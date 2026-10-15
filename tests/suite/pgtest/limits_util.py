"""Builders for the limits interplay scenarios (scenarios/interplay_limits.py):
several subsystems at their limits at once.

The limits, as the code states them (Rust paths name items of
reims-vgpu-core):
  - 64 FIFOs at binary version 61 and up: the root and children 1-63
    (maxFIFOCount, which the child FIFO define checks).
  - 128 pending stamps a FIFO (fifo::stamps::STAMP_QUEUE_CAPACITY).
  - 256 tasks, IDs 0-255 (device::tasks::MAX_TASKS); each remote task
    (1-255 under isolation) is a service instance of its own (the
    service's Info.plist sets _MultipleInstances).
  - the root FIFO's 0x20000 bytes; a child's 4096 ring pages.
  - in the blocks setup, the root task's range allocator of 0xffffff bytes
    in 16 KiB granules, which maps the root node, the root FIFO, every
    child's ring and page list, every local task's root page and every
    display's shared page.
  - 8 display ports (device::config::MAX_DISPLAY_PORT_COUNT), 128 modes a
    page (display::page::MAX_MODES), 1024 gamma entries, cursor glyphs of
    256 x 256.
  - IOSurface mapping IDs below 8192.

Everything here composes the per-area builders (fifo.py, task.py,
newops_util.py, interplay_util.py); nothing is imported from the
implementation.
"""
import struct

from . import abi
from .abi import PAGE
from .newops_util import cmd_seq

# Blit record 306, fill: {u32 buffer, u64 offset, u64 length, u8 value},
# 8 + 24 bytes.
FILL = 306
FILL_REC = 32
SEG = 8  # the segment header


def fill_seg(ref, off, n, byte):
    """One blit segment holding one fill."""
    body = abi.REC_HDR.pack(FILL, FILL_REC) + struct.pack("<IQQB3x", ref, off, n, byte)
    return abi.SEG_HDR.pack(SEG + len(body), abi.ENC_BLIT, 0, 0) + body


def exec3(tid, cmdbufs, resources):
    """ExecIndirect3 (67): {u32 taskID, u32 resourceCount, u32
    cmdBufferCount}, {u32 ref, u32 flags} each, {u64 VA, u64 length}
    each."""
    p = struct.pack("<III", tid, len(resources), len(cmdbufs))
    p += b"".join(struct.pack("<II", r, f) for r, f in resources)
    p += b"".join(struct.pack("<QQ", va, n) for va, n in cmdbufs)
    return p


def buffer_record(length, va_page, flags=0):
    """A buffer (type 1): {u64 length, u32 task VA page, u8 flags}."""
    return struct.pack("<QIB3x", length, va_page, flags)


class Space:
    """One task address space that many tasks share: a page table of
    `depth` levels (4096 u32 entries a level, 12-bit indices top-down,
    bit 31 of a leaf writable) over `pages` data
    pages. `leaf(vpn)` may name the RAM page of VA page vpn (default: the
    space's own contiguous pages). Every task defined over it sees the same
    bytes at the same VAs, so their results land side by side and one dump
    shows them all."""

    def __init__(self, s, pages, depth=1, leaf=None, table=None, vpns=None, length=None):
        self.s = s
        self.npages = pages
        self.depth = depth
        self.length = pages * PAGE if length is None else length
        self.data = s.alloc_page(pages) if leaf is None else None
        self.leaf = leaf or (lambda v: self.data + v)
        self._table_alloc = table or s.alloc_page
        self._tables = {}
        self.pt_root = self._table(0, 0)
        # The leaves of one last-level table go in one write. vpns: the VA
        # pages mapped (default 0..pages-1); the rest are holes.
        rows = {}
        for v in (range(pages) if vpns is None else vpns):
            prefix = 0
            for level in range(depth):
                shift = 12 * (depth - 1 - level)
                idx = (v >> shift) & 0xFFF
                t = self._table(level, prefix)
                if level == depth - 1:
                    rows.setdefault(t, {})[idx] = s.pfn(self.leaf(v)) | abi.PTE_WRITABLE
                else:
                    nxt = self._table(level + 1, (prefix << 12) | idx)
                    rows.setdefault(t, {})[idx] = s.pfn(nxt)
                prefix = (prefix << 12) | idx
        for t, row in rows.items():
            lo, hi = min(row), max(row)
            s.mem(s.pa(t, 4 * lo), struct.pack("<%dI" % (hi - lo + 1), *[row.get(i, 0) for i in range(lo, hi + 1)]))

    def _table(self, level, prefix):
        if (level, prefix) not in self._tables:
            self._tables[(level, prefix)] = self._table_alloc()
        return self._tables[(level, prefix)]

    def gpa(self, va):
        return self.s.pa(self.leaf(va // PAGE), va % PAGE)

    def write(self, va, data):
        data = bytes(data)
        pos = 0
        while pos < len(data):
            a = va + pos
            n = min(len(data) - pos, PAGE - a % PAGE)
            self.s.mem(self.gpa(a), data[pos:pos + n])
            pos += n

    def dump(self, va, n, label):
        self.s.dump(self.gpa(va), n, label=label)

    def root_bytes(self, obj_off=0, obj_len=0):
        """The task root (guest::records::TaskRoot2): table root page,
        depth, object list offset and length; the host writes the length
        at +0x14."""
        return abi.TASK_ROOT.pack(self.s.pfn(self.pt_root), self.depth, obj_off, obj_len, 0, 0, 0)


class Roots:
    """Task root pages for tasks first..first+count-1 over one Space, in
    contiguous RAM pages, written with one mem_seq (stride one page)."""

    def __init__(self, s, space, first, count):
        self.s = s
        self.space = space
        self.first = first
        self.count = count
        self.page0 = s.alloc_page(count)
        s.mem_seq(s.pa(self.page0), count, PAGE, space.root_bytes())

    def page(self, tid):
        return self.page0 + tid - self.first

    def pfn(self, tid):
        return self.s.pfn(self.page(tid))


# ---- 32-byte and larger commands in bulk (newops_util.cmd_seq) ----

def ring_seq(s, f, count, template, fields=()):
    """cmd_seq on any FIFO: the root FIFO, or a child whose ring pages are
    physically contiguous (ChildFIFO without scatter)."""
    if f.fid == 0:
        return cmd_seq(s, f, count, template, fields)
    assert f.pages == list(range(f.pages[0], f.pages[0] + len(f.pages)))

    class _View:
        pass
    v = _View()
    v.written, v.stamp, v.start = f.written, f.stamp, 0
    v.ring_size = f.ring_size
    v.base_gpa = lambda: s.pa(f.pages[0])
    last = cmd_seq(s, v, count, template, fields)
    f.written, f.stamp = v.written, v.stamp
    return last


def define_seq(s, f, roots, first, count, length):
    """DefineTask2 (56) {u32 taskID << 1, u64 length, u32 task root page}
    for tasks first.., padded to 32 bytes."""
    t = abi.HDR.pack(abi.OP_DEFINE_TASK2, 0, 32, 0) + struct.pack("<IQI", 0, length, 0) + bytes(4)
    return cmd_seq(s, f, count, t, fields=[(12, 4, first << 1, 2), (24, 4, roots.pfn(first), 1)])


def set_list_seq(s, f, first, count, list_va, list_len):
    """SetObjectList (51) {u32 taskID, u32 list page, u32 length}, 32 bytes
    with a barrier on f's own previous stamp: 51 completes asynchronously
    (under isolation, by the service's reply), and a command that waits for
    the one before it to publish publishes alone, one interrupt each (see
    exec_seq)."""
    t = abi.HDR.pack(abi.OP_SET_OBJECT_LIST, 1, 32, 0) + abi.BARRIER.pack(f.stamp_index, 0) + struct.pack(
        "<III", 0, list_va // PAGE, list_len)
    return ring_seq(s, f, count, t, fields=[(20, 4, first, 1), (16, 4, f.stamp, 1)])


def delete_seq(s, f, first, count, step=1):
    """DeleteTask (32) {u32 taskID}, padded to 32 bytes."""
    t = abi.HDR.pack(abi.OP_DELETE_TASK, 0, 32, 0) + bytes(20)
    return ring_seq(s, f, count, t, fields=[(12, 4, first, step)])


def exec_seq(s, f, first, count, resources, cb_va0, cb_step, cb_len, size, chain=True, res_step=None):
    """count ExecIndirect3 (67) commands, one a task from `first`, each with
    one command buffer at cb_va0 + k * cb_step and the same resource list
    (with res_step, the first resource's ref steps by it);
    with chain=True each carries a barrier on its own FIFO's previous stamp
    (pushed, not yet signalled: it waits for the exec before it, and is not
    circular), so they complete, and publish, one at a time. Padded to size, which must divide the ring."""
    nb = 1 if chain else 0
    hdr = abi.HDR.pack(abi.OP_EXEC_INDIRECT3, nb, size, 0)
    bar = abi.BARRIER.pack(f.stamp_index, 0) if chain else b""
    body = exec3(0, [(0, cb_len)], resources)
    t = hdr + bar + body
    assert len(t) <= size
    t += bytes(size - len(t))
    base = 12 + 8 * nb
    fields = [(base, 4, first, 1), (base + 12 + 8 * len(resources), 8, cb_va0, cb_step)]
    if res_step is not None:
        fields.append((base + 12, 4, resources[0][0], res_step))
    if chain:
        fields.append((16, 4, f.stamp, 1))  # the previous stamp: the first waits on what is already published
    return ring_seq(s, f, count, t, fields=fields)


def fill_cmdbufs(s, space, va0, step, count, ref0, ref_step, off0, off_step, n, byte0, byte_step=1):
    """count command buffers, one fill segment each, at va0 + k * step, in
    one mem_seq (the runner writes the whole span, gaps included, so the
    buffers own it): buffer ref0 + k * ref_step, at off0 + k * off_step, n
    bytes of byte0 + k * byte_step (below 0x100). The space's pages must be
    contiguous in RAM over the span. Returns the segment's length."""
    seg = fill_seg(ref0, 0, n, 0)
    assert byte0 + (count - 1) * byte_step < 0x100
    # Segment header 8, record header 8, then ref @16, offset @20, length
    # @28, value (u8, 3 pad bytes that stay 0) @36.
    s.mem_seq(space.gpa(va0), count, step, seg,
              fields=[(16, 4, ref0, ref_step), (20, 8, off0, off_step), (36, 4, byte0, byte_step)])
    return len(seg)


def buffer_list(s, space, list_va, rec_va, count, length, va_page):
    """An object list of `count` buffers (type 1), all over the same pages
    (VA page va_page, `length` bytes): entry 0 nil, entry i {1 | 16 << 8,
    rec_va + 16 (i - 1)} (guest::records::ObjectEntry), and the records
    packed at rec_va. Two mem_seqs whose copies abut. Returns the
    list's length in bytes."""
    s.mem(space.gpa(list_va), abi.OBJ_ENTRY.pack(0, 0))
    s.mem_seq(space.gpa(list_va + 12), count, 12, abi.OBJ_ENTRY.pack(abi.T_BUFFER | 16 << 8, 0),
              fields=[(4, 8, rec_va, 16)])
    s.mem_seq(space.gpa(rec_va), count, 16, buffer_record(length, va_page))
    return 12 * (count + 1)


def align(s, f, size):
    """One NOP, padded (its header's length covers the pad; a longer
    payload is accepted), that brings f's written count to a multiple of
    size, submitted and waited for, so that cmd_seq can follow."""
    gap = -f.written % size
    if gap == 0:
        return None
    if gap < abi.HDR.size:
        gap += size
    st = f.cmd(abi.OP_NOP, length=gap, pad_to=gap)
    f.doorbell()
    f.wait_stamp(st, label="aligning NOP")
    return st


def skip_to(f, target, barriers=()):
    """A NOP whose header's length carries f's written count to ring offset
    `target`: only its header (and barriers) are written, and its payload
    is whatever the ring holds, which a NOP never reads (a payload longer
    than the command needs is accepted). The
    header lands wherever the ring is, straddling a page or the ring's end
    as it falls (FIFO.raw). Returns the NOP's stamp."""
    ring = f.ring_size()
    head = abi.HDR.size + abi.BARRIER.size * len(barriers)
    n = (target - f.written) % ring
    assert head <= n <= ring, (hex(f.written % ring), hex(target))
    f.stamp += 1
    f.raw(abi.HDR.pack(abi.OP_NOP, len(barriers), n, f.stamp) + b"".join(abi.BARRIER.pack(i, v) for i, v in barriers))
    f.written += n - head
    return f.stamp
