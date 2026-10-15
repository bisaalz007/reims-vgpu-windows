"""Builders for the tasks-and-memory scenarios (docs/testing.md, sections
6.10-6.11): tasks whose page tables the scenario lays out leaf by leaf, raw
task commands with any field, and the heap-texture payload.

Everything here restates the guest ABI from byte offsets, as task.py does;
nothing is taken from the implementation.
"""
import struct

from . import abi
from .abi import PAGE
from .task import GPUTask, ObjectList, Task, buffer_record, rec, seg

# Blit fill (the blit decoder's record 306): {u32 buffer ref, u64 offset,
# u64 length, u8 value} (the same layout smoke_blit uses).
BLIT_FILL = 306
BLIT_COPY = 301


def fill_rec(ref, off, n, byte):
    return rec(BLIT_FILL, struct.pack("<IQQB3x", ref, off, n, byte))


class XTask(Task):
    """A task whose leaves the scenario chooses.

    leaves: {vpn: ram_page}, the guest page each VA page maps to; vpns not
      in it are holes. By default every page of `pages` gets a data page of
      its own, allocated contiguously (as Task does), or one apart with
      scatter=True (so no two are physically adjacent).
    writable: True, False, or a set of vpns whose leaves carry bit 31
      (memory::page_table: bit 31 of a leaf is the writable bit).
    alias: map every non-hole vpn to this one RAM page (for tasks larger than
      guest RAM).
    mapped: only these vpns get leaves (with alias or allocation); the rest
      are holes. Defaults to range(pages) minus holes.
    """

    def __init__(self, s, tid=1, pages=16, depth=1, writable=True, holes=(), length=None,
                 leaves=None, scatter=False, alias=None, mapped=None):
        self.s = s
        self.tid = tid
        self.npages = pages
        self.depth = depth
        self.length = pages * PAGE if length is None else length
        self.root = s.alloc_page()
        self.holes = set(holes)
        self.writable = writable
        self._tables = {}
        self.pt_root = self._table(0, 0)
        if mapped is None:
            mapped = [v for v in range(pages) if v not in self.holes]
        if leaves is None:
            leaves = {}
            if alias is not None:
                for v in mapped:
                    leaves[v] = alias
            else:
                for v in mapped:
                    leaves[v] = s.alloc_page()
                    if scatter:
                        s.alloc_page()
        self.leaves = dict(leaves)
        pg = sorted(self.leaves.values())
        self.data = pg[0] if pg else s._next_page
        self._data_end = (pg[-1] + 1) if pg else s._next_page
        for v in sorted(self.leaves):
            self._map_leaf(v)
        self.write_root()

    def _map_leaf(self, vpn):
        s = self.s
        d = self.depth
        prefix = 0
        for level in range(d):
            shift = 12 * (d - 1 - level)
            idx = (vpn >> shift) & 0xFFF
            table = self._table(level, prefix)
            if level == d - 1:
                w = self.writable is True or (self.writable not in (True, False) and vpn in self.writable)
                s.u32(s.pa(table, 4 * idx), s.pfn(self.leaves[vpn]) | (abi.PTE_WRITABLE if w else 0))
            else:
                nxt = self._table(level + 1, (prefix << 12) | idx)
                s.u32(s.pa(table, 4 * idx), s.pfn(nxt))
            prefix = (prefix << 12) | idx

    def table_gpa(self, level, prefix, idx=0):
        """The guest address of entry idx of the level-`level` table reached
        through `prefix` (the higher indices, concatenated)."""
        return self.s.pa(self._tables[(level, prefix)], 4 * idx)

    def gpa(self, va):
        return self.s.pa(self.leaves[va // PAGE], va % PAGE)

    def diff(self, label=None):
        self.s.diff(label=label, gpa=self.s.pa(self.data), length=(self._data_end - self.data) * PAGE)


class XGPUTask(XTask):
    """GPUTask's layout (task.py) over an XTask: page 0 the object list,
    pages 1-3 records, pages 4-7 command buffers, 8.. buffer contents."""

    def __init__(self, s, tid=1, pages=64, **kw):
        super().__init__(s, tid=tid, pages=pages, **kw)
        self.ol = ObjectList(self, 0, 1 * PAGE)
        self._cmd = 4 * PAGE
        self._blob = 8 * PAGE

    buffer = GPUTask.buffer
    reserve = GPUTask.reserve
    blob = GPUTask.blob
    cmdbuf = GPUTask.cmdbuf
    reset_cmdbufs = GPUTask.reset_cmdbufs
    install = GPUTask.install


# ---- raw commands (the FIFO payloads, any field) ----

def define_raw(s, word0, length, root_pfn, **kw):
    """DefineTask2 (56): {u32 bit 0 | taskID << 1, u64 length, u32 task root
    page}."""
    return s.submit(abi.OP_DEFINE_TASK2, struct.pack("<IQI", word0 & 0xFFFFFFFF, length, root_pfn), **kw)


def map_raw(s, tid, off, n, **kw):
    """MapMemory2 (57): {u32 taskID, u64 offset, u64 length}."""
    return s.submit(abi.OP_MAP_MEMORY2, struct.pack("<IQQ", tid, off, n), **kw)


def unmap_raw(s, tid, off, n, **kw):
    """UnmapMemory (34): {u32 taskID, u64 offset, u64 length}."""
    return s.submit(abi.OP_UNMAP_MEMORY, struct.pack("<IQQ", tid, off, n), **kw)


def delete_raw(s, tid, **kw):
    """DeleteTask (32): {u32 taskID}."""
    return s.submit(abi.OP_DELETE_TASK, struct.pack("<I", tid), **kw)


def set_list_raw(s, tid, page, n, **kw):
    """SetObjectList (51): {u32 taskID, u32 list page (task VA page), u32
    length}; placement offset and length 0."""
    return s.submit(abi.OP_SET_OBJECT_LIST, struct.pack("<III", tid, page, n), **kw)


def set_lists_raw(s, tid, obj_off, place_off, obj_len, place_len, **kw):
    """SetObjectAndPlacementList (68): taskID@0 objVO u64@4 placeVO u64@12
    objLen@20 placeLen@24."""
    return s.submit(abi.OP_SET_OBJECT_AND_PLACEMENT_LIST,
                    struct.pack("<IQQII", tid, obj_off, place_off, obj_len, place_len), **kw)


def refs_payload(tid, refs):
    """{u32 taskID, u32 count} + count x u32 reference: opcodes 53, 62, 63
    (reims-vgpu-core's tasks::ResourceReference for each reference)."""
    return struct.pack("<II", tid, len(refs)) + b"".join(struct.pack("<I", r) for r in refs)


def invalidations_payload(tid, items):
    """Opcode 52: {u32 taskID, u32 count} + count x {u32 reference, four
    u8 flags at +4..+7} (reims-vgpu-core's tasks::ResourceInvalidation, 8
    bytes; invalidate_resources tests them).
    items: (reference, (b4, b5, b6, b7))."""
    return struct.pack("<II", tid, len(items)) + b"".join(
        struct.pack("<IBBBB", r, *flags) for r, flags in items)


# ---- heap texture size and alignment (64) ----

def tex_desc(ttype=2, usage=1, fmt=70, w=256, h=256, d=1, mips=1, samples=1, array=1, options=0x20, prot=0):
    """A v1 serialized texture descriptor, 32 bytes (reims-vgpu-core's
    deserializer::texture_descriptor::SerializedTextureDescriptor): u32
    {type:4, framebufferOnly:1, isDrawable:1, allowGPUOptimizedContents:1,
    :1, usage:8, pixelFormat:16}, u32 width, height, depth, u16
    mipmapLevelCount, sampleCount, arrayLength, resourceOptions, u64
    protectionOptions."""
    word0 = (ttype & 0xF) | (usage & 0xFF) << 8 | (fmt & 0xFFFF) << 16
    return struct.pack("<IIIIHHHHQ", word0, w, h, d, mips, samples, array, options, prot)


def heap_ser(desc=None, cmd=22, pad=b""):
    """The serializer payload: u32 command 22, u32 unread, then the
    descriptor."""
    return struct.pack("<II", cmd, 0) + (tex_desc() if desc is None else desc) + pad


def heap_payload(tid, reply_va, reply_len, ser, payload_len=None):
    """HeapTextureSizeAndAlign (64): taskID@0 replyVO u64@4 replyLen u64@12
    payloadLen u32@20, then the payload."""
    n = len(ser) if payload_len is None else payload_len
    return struct.pack("<IQQI", tid, reply_va, reply_len, n) + ser


def heap_metal(s, fmt=70, w=256, h=256, usage=1, options=0x20, label=None):
    """The value the reply must hold, computed from Metal in the test
    (docs/testing.md, section 5.8)."""
    kw = dict(format=fmt, w=w, h=h, usage=usage, options=options)
    if label:
        kw["label"] = label
    return s.metal("heapTextureSizeAndAlign", **kw)


def fault_and_resume(s, label):
    """After a command that faults: the fault state, then resume (0x1024)."""
    s.fault_state(label)
    s.resume_root()


def maybe_fault(s, opcode, payload, label, timeout_ms=500):
    """A command whose outcome differs between setups (faults in some,
    completes in others; docs/testing.md, section 6.10's divergence table).
    Waits a short while for a fault, records the fault state, resumes (a no-op
    when nothing faulted: StampQueue::resume_from_fault finds no faulted head),
    then waits for the command's stamp, which either path publishes."""
    st = s.cmd(opcode, payload)
    s.doorbell()
    s.wait(mmio=abi.R_FAULT_MASK, mmio_eq=1, timeout_ms=timeout_ms, label="fault? " + label)
    s.fault_state(label)
    s.resume_root()
    s.fifo.wait_stamp(st, label="stamp after " + label)
    return st


def exec3_payload(tid, cmdbufs=(), resources=()):
    """ExecIndirect3 (67): {u32 taskID, u32 resourceCount, u32
    cmdBufferCount}, resourceCount x {u32 ref, u32 flags}, cmdBufferCount x
    {u64 VA, u64 length}."""
    p = struct.pack("<III", tid, len(resources), len(cmdbufs))
    p += b"".join(struct.pack("<II", r, f) for r, f in resources)
    p += b"".join(struct.pack("<QQ", va, n) for va, n in cmdbufs)
    return p


def probe_list(s, tid, page, npages, label, expect="ok"):
    """Makes the host map [page, page + npages) of task memory as an object
    list: SetObjectList (51) over those pages, then an ExecIndirect3 (67)
    with no command buffers, which maps the object list before anything else
    (ensure_object_list_mapped, in process and in the service). In the
    blocks setup this is the lazy path: the range tracker's test_range and
    visit_unmapped_ranges, then a map_memory walk per unmapped run. The
    list's length ends 12 bytes short of the last page's end, so it stays
    below the task's end when the range reaches it."""
    set_list_raw(s, tid, page, npages * PAGE - 12)
    p = exec3_payload(tid)
    if expect == "maybe":
        return maybe_fault(s, abi.OP_EXEC_INDIRECT3, p, label)
    if expect == "fault":
        st = s.submit(abi.OP_EXEC_INDIRECT3, p, fault=True, label=label)
        fault_and_resume(s, label)
        s.fifo.wait_stamp(st)
        return st
    return s.submit(abi.OP_EXEC_INDIRECT3, p, label=label)
