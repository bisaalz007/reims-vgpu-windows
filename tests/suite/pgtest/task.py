"""Guest tasks, their page tables, object lists and command buffers
(docs/testing.md, sections 6.10-6.13)."""
import struct

from . import abi
from .abi import PAGE


class Task:
    """A guest task: a task root page (reims-vgpu-core's
    guest::records::TaskRoot2), a 1-3 level page table (4096 u32 entries a
    level, 12-bit indices top-down, bit 31 of a leaf makes it writable, 0 is
    a hole), and data pages.

    pages: the task's length in pages. writable: every leaf writable, or a
    set of VA page numbers that are. holes: VA page numbers left unmapped.
    """

    def __init__(self, s, tid=1, pages=16, depth=1, writable=True, holes=(), length=None):
        self.s = s
        self.tid = tid
        self.npages = pages
        self.depth = depth
        self.length = pages * PAGE if length is None else length
        self.root = s.alloc_page()
        self.data = s.alloc_page(pages)
        self.holes = set(holes)
        self.writable = writable
        self._tables = {}  # (level, prefix) -> RAM page
        self.pt_root = self._table(0, 0)
        for v in range(pages):
            if v in self.holes:
                continue
            self._map_leaf(v)
        self.write_root()

    # ---- the page table ----

    def _table(self, level, prefix):
        key = (level, prefix)
        if key not in self._tables:
            self._tables[key] = self.s.alloc_page()
        return self._tables[key]

    def _map_leaf(self, vpn):
        s = self.s
        d = self.depth
        prefix = 0
        for level in range(d):
            shift = 12 * (d - 1 - level)
            idx = (vpn >> shift) & 0xFFF
            table = self._table(level, prefix)
            if level == d - 1:
                w = self.writable is True or (self.writable and vpn in self.writable)
                s.u32(s.pa(table, 4 * idx), s.pfn(self.data + vpn) | (abi.PTE_WRITABLE if w else 0))
            else:
                nxt = self._table(level + 1, (prefix << 12) | idx)
                s.u32(s.pa(table, 4 * idx), s.pfn(nxt))
            prefix = (prefix << 12) | idx

    def write_root(self, pt_root=None, depth=None, obj_off=0, obj_len=0, length=0, place_off=0, place_len=0):
        """The task root, as the guest writes it before define. The host
        writes the task length at +0x14."""
        s = self.s
        s.mem(s.pa(self.root), abi.TASK_ROOT.pack(
            s.pfn(self.pt_root) if pt_root is None else pt_root,
            self.depth if depth is None else depth,
            obj_off, obj_len, length, place_off, place_len))

    # ---- task virtual memory ----

    def gpa(self, va):
        return self.s.pa(self.data + va // PAGE, va % PAGE)

    def write(self, va, data):
        pos = 0
        data = bytes(data)
        while pos < len(data):
            a = va + pos
            n = min(len(data) - pos, PAGE - a % PAGE)
            self.s.mem(self.gpa(a), data[pos:pos + n])
            pos += n

    def fill(self, va, length, byte=0):
        self.s.fill(self.gpa(va), length, byte)

    def dump(self, va, length, label=None):
        self.s.dump(self.gpa(va), length, label=label)

    def diff(self, label=None):
        """Diffs only this task's data pages."""
        self.s.diff(label=label, gpa=self.s.pa(self.data), length=self.npages * PAGE)

    # ---- commands ----

    def define(self, length=None, root=None, bit0=0, fifo=None, **kw):
        """DefineTask2 (56): {u32 bit 0 | taskID << 1, u64 length, u32 task
        root page}."""
        payload = struct.pack("<IQI", (self.tid << 1) | bit0,
                              self.length if length is None else length,
                              self.s.pfn(self.root) if root is None else root)
        return (fifo or self.s).submit(abi.OP_DEFINE_TASK2, payload, **kw)

    def map(self, offset=0, length=None, **kw):
        """MapMemory2 (57): {u32 taskID, u64 offset, u64 length}."""
        return self.s.submit(abi.OP_MAP_MEMORY2, struct.pack(
            "<IQQ", self.tid, offset, self.length if length is None else length), **kw)

    def unmap(self, offset=0, length=None, **kw):
        """UnmapMemory (34): {u32 taskID, u64 offset, u64 length}."""
        return self.s.submit(abi.OP_UNMAP_MEMORY, struct.pack(
            "<IQQ", self.tid, offset, self.length if length is None else length), **kw)

    def delete(self, **kw):
        """DeleteTask (32): {u32 taskID}."""
        return self.s.submit(abi.OP_DELETE_TASK, struct.pack("<I", self.tid), **kw)

    def set_object_list(self, va, length, **kw):
        """SetObjectList (51): {u32 taskID, u32 list page (task VA page),
        u32 length in bytes}."""
        return self.s.submit(abi.OP_SET_OBJECT_LIST, struct.pack("<III", self.tid, va // PAGE, length), **kw)

    def set_object_and_placement_list(self, obj_va, place_va, obj_len, place_len, **kw):
        """SetObjectAndPlacementList (68): taskID@0 objVO u64@4 placeVO
        u64@12 objLen@20 placeLen@24."""
        return self.s.submit(abi.OP_SET_OBJECT_AND_PLACEMENT_LIST, struct.pack(
            "<IQQII", self.tid, obj_va, place_va, obj_len, place_len), **kw)

    def exec3(self, cmdbufs, resources=(), opcode=abi.OP_EXEC_INDIRECT3, **kw):
        """ExecIndirect3 (67) or InfoIndirect (69): {u32 taskID, u32
        resourceCount, u32 cmdBufferCount}, resourceCount x {u32 reference,
        u32 flags}, cmdBufferCount x {u64 task VA, u64 length}."""
        p = struct.pack("<III", self.tid, len(resources), len(cmdbufs))
        p += b"".join(struct.pack("<II", r, f) for r, f in resources)
        p += b"".join(struct.pack("<QQ", va, n) for va, n in cmdbufs)
        return self.s.submit(opcode, p, **kw)

    def exec2(self, cmdbufs, resources=(), **kw):
        """ExecIndirect2 (55): resource records are 24 bytes: {u32
        reference, u8 invalidateHost, u8 invalidateGuest, u8 synchronize,
        pad, u64, u64}."""
        p = struct.pack("<III", self.tid, len(resources), len(cmdbufs))
        p += b"".join(struct.pack("<IBBBxQQ", r, ih, ig, sy, 0, 0) for r, ih, ig, sy in resources)
        p += b"".join(struct.pack("<QQ", va, n) for va, n in cmdbufs)
        return self.s.submit(abi.OP_EXEC_INDIRECT2, p, **kw)


class ObjectList:
    """An object list (guest::records::ObjectEntry, 12 bytes each) and the
    records it names, in task memory. Reference 0 is nil (entry 0 stays
    empty). Records are packed from records_va on, 16-byte aligned."""

    def __init__(self, task, list_va, records_va):
        self.task = task
        self.list_va = list_va
        self.records_va = records_va
        self._rec = records_va
        self.entries = [(0, 0, 0)]  # (type, length, va)

    def add(self, typ, record, align=16):
        """Adds a record; returns its reference."""
        record = bytes(record)
        va = self._rec
        self.task.write(va, record)
        self._rec = (va + len(record) + align - 1) // align * align
        self.entries.append((typ, len(record), va))
        return len(self.entries) - 1

    def add_raw(self, typ, length, va):
        """An entry naming whatever is at va, with any length."""
        self.entries.append((typ, length, va))
        return len(self.entries) - 1

    def blob_va(self, data, align=PAGE):
        """Places arbitrary bytes (a shader, a buffer's contents) after the
        records, aligned; returns their VA."""
        va = (self._rec + align - 1) // align * align
        self.task.write(va, bytes(data))
        self._rec = va + len(data)
        return va

    def reserve(self, length, align=PAGE):
        va = (self._rec + align - 1) // align * align
        self._rec = va + length
        return va

    def write(self):
        data = b"".join(abi.OBJ_ENTRY.pack(t | (n << 8), va) for t, n, va in self.entries)
        self.task.write(self.list_va, data)
        return len(data)

    def length(self):
        return 12 * len(self.entries)

    def install(self, **kw):
        """Writes the list and sets it with opcode 51."""
        self.write()
        return self.task.set_object_list(self.list_va, self.length(), **kw)


def seg(enc, records=b"", cont_prev=0, cont_next=0, length=None):
    """A segment: the 8-byte header (reims-vgpu-core's
    deserializer::records::SegmentHeader), then its records. length
    overrides the header's length (the whole segment)."""
    body = bytes(records)
    n = abi.SEG_HDR.size + len(body) if length is None else length
    return abi.SEG_HDR.pack(n & 0xFFFFFFFF, enc, cont_prev, cont_next) + body


def rec(cmd, payload=b"", length=None):
    """A decoder record: {u32 commandID, u32 length incl. header}, payload."""
    payload = bytes(payload)
    n = abi.REC_HDR.size + len(payload) if length is None else length
    return abi.REC_HDR.pack(cmd, n & 0xFFFFFFFF) + payload


def buffer_record(length, va_page, flags=0):
    """A buffer (type 1): {u64 length, u32 task VA page, u8 flags}, 16 bytes;
    flags bit 0 private, bit 1 read-only."""
    return struct.pack("<QIB3x", length, va_page, flags)


class GPUTask(Task):
    """A task with an object list and room for records, command buffers and
    buffer contents, laid out in fixed VA regions:

      page 0          object list
      pages 1-3       records
      pages 4-7       command buffers (one segment each, cmdbuf())
      pages 8..       blobs and buffer contents (ol.blob_va / reserve)
    """

    def __init__(self, s, tid=1, pages=64, **kw):
        super().__init__(s, tid=tid, pages=pages, **kw)
        self.ol = ObjectList(self, 0, 1 * PAGE)
        self.ol._blob_start = 8 * PAGE
        self._cmd = 4 * PAGE

    def buffer(self, length=PAGE, flags=0, contents=None, va=None):
        """A shared buffer over task pages; returns (ref, va)."""
        if va is None:
            va = self.reserve(max(length, 1))
        if contents is not None:
            self.write(va, contents)
        return self.ol.add(abi.T_BUFFER, buffer_record(length, va // PAGE, flags)), va

    def reserve(self, length, align=PAGE):
        start = max(self.ol._rec, getattr(self, "_blob", 8 * PAGE))
        va = (start + align - 1) // align * align
        self._blob = va + length
        return va

    def blob(self, data, align=PAGE):
        va = self.reserve(len(data), align)
        self.write(va, data)
        return va

    def cmdbuf(self, data):
        """Places one command-buffer range (one segment) in the command
        region; returns (va, length)."""
        va = self._cmd
        self.write(va, data)
        self._cmd = (va + len(data) + 0xFF) // 0x100 * 0x100
        assert self._cmd <= 8 * PAGE, "command region full"
        return (va, len(data))

    def reset_cmdbufs(self):
        self._cmd = 4 * PAGE

    def install(self, **kw):
        """Writes the object list and sets it with opcode 51."""
        return self.ol.install(**kw)
