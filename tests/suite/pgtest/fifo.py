"""The root FIFO and child FIFOs (docs/testing.md, sections 6.5-6.7)."""
import struct

from . import abi
from .abi import PAGE


class FIFO:
    """What both kinds share: a ring, a free-running written count, stamps."""

    stamp_index = 0

    def __init__(self, s):
        self.s = s
        self.written = 0
        self.stamp = 0

    def ring_size(self):
        raise NotImplementedError

    def ring_write(self, off, data):
        raise NotImplementedError

    def raw(self, data):
        """Writes bytes into the ring at the written count, wrapping at the
        ring's end (a command may straddle it), and advances the count."""
        size = self.ring_size()
        pos = 0
        while pos < len(data):
            off = (self.written + pos) % size
            n = min(len(data) - pos, size - off)
            self.ring_write(off, data[pos:pos + n])
            pos += n
        self.written += len(data)

    def cmd(self, opcode, payload=b"", barriers=(), length=None, stamp=None, pad_to=None):
        """One command: the 12-byte header (reims-vgpu-core's
        fifo::wire::FifoCommand), barriers {u32 fifoIndex, u32 value}, then
        the payload. length overrides the header's length field; stamp
        overrides the stamp value (the next one by default). Returns the
        stamp value."""
        if stamp is None:
            self.stamp += 1
            stamp = self.stamp
        body = b"".join(abi.BARRIER.pack(i, v) for i, v in barriers) + bytes(payload)
        n = abi.HDR.size + len(body) if length is None else length
        data = abi.HDR.pack(opcode, len(barriers), n & 0xFFFFFFFF, stamp & 0xFFFFFFFF) + body
        if pad_to is not None and len(data) < pad_to:
            data += bytes(pad_to - len(data))
        self.raw(data)
        return stamp

    def stamp_gpa(self):
        """Stamp i is the u32 at the root FIFO mapping's base + 4i
        (create_root_fifo maps it; DeviceCore::signal_stamp writes it)."""
        return self.s.fifo.base_gpa() + 4 * self.stamp_index

    def wait_stamp(self, value, timeout_ms=5000, label=None, **extra):
        self.s.wait(gpa=self.stamp_gpa(), ge=value, timeout_ms=timeout_ms,
                    label=label or "stamp %d on FIFO %d" % (value, self.fid), **extra)


class RootFIFO(FIFO):
    fid = 0

    def __init__(self, s, pages=2, start=0x400):
        super().__init__(s)
        self.pages = pages
        self.page = s.alloc_page(pages)
        self.length = pages * PAGE
        self.start = start

    def base_gpa(self):
        return self.s.pa(self.page)

    def ring_size(self):
        return self.length - self.start

    def ring_write(self, off, data):
        self.s.mem(self.base_gpa() + self.start + off, data)

    def enable(self, length=None, start=None, base=None, value=1):
        s = self.s
        s.w(abi.R_FIFO_BASE, s.pfn(self.page) if base is None else base)
        s.w(abi.R_FIFO_LENGTH, self.length if length is None else length)
        s.w(abi.R_FIFO_START, self.start if start is None else start)
        s.w(abi.R_FIFO_ENABLE, value)

    def doorbell(self, written=None):
        self.s.w(abi.R_FIFO_WRITTEN, (self.written if written is None else written) & 0xFFFFFFFF)


class ChildFIFO(FIFO):
    """A child FIFO: a node in the root page (0x400 + 0x14 (id - 1)), a page
    list page of u32 guest page numbers ending at the first 0, and a ring over
    those pages, virtually contiguous (section 6.7)."""

    def __init__(self, s, fid, stamp_index=None, pages=1, scatter=False):
        super().__init__(s)
        self.fid = fid
        self.stamp_index = fid if stamp_index is None else stamp_index
        self.list_page = s.alloc_page()
        if scatter:
            # Pages that are not physically contiguous: one apart each.
            self.pages = []
            for _ in range(pages):
                self.pages.append(s.alloc_page())
                s.alloc_page()
        else:
            first = s.alloc_page(pages)
            self.pages = [first + i for i in range(pages)]

    def node_gpa(self, field=0):
        return self.s.pa(self.s.root_page, abi.ROOT_CHILD_NODE + abi.CHILD_NODE_SIZE * (self.fid - 1) + field)

    def ring_size(self):
        return len(self.pages) * PAGE

    def ring_write(self, off, data):
        pos = 0
        while pos < len(data):
            pg, o = divmod(off + pos, PAGE)
            n = min(len(data) - pos, PAGE - o)
            self.s.mem(self.s.pa(self.pages[pg], o), data[pos:pos + n])
            pos += n

    def setup_node(self, page_list=None, stamp_index=None):
        """Writes the node's guest fields and the page list."""
        s = self.s
        plist = [s.pfn(p) for p in self.pages] if page_list is None else page_list
        s.mem(s.pa(self.list_page), struct.pack("<%dI" % (len(plist) + 1), *plist, 0))
        # {u32 written, u32 readCount, u32 faultOffset, u16 stampIndex, u16
        # running, u32 pageListPage}; the host owns readCount, faultOffset
        # and running.
        s.mem(self.node_gpa(0), struct.pack("<I", 0))
        s.mem(self.node_gpa(0xC), struct.pack("<H", self.stamp_index if stamp_index is None else stamp_index))
        s.mem(self.node_gpa(0x10), struct.pack("<I", s.pfn(self.list_page)))

    def define(self, **kw):
        """Opcode 48 on the root FIFO."""
        return self.s.submit(abi.OP_DEFINE_CHILD_FIFO, struct.pack("<I", self.fid), **kw)

    def delete(self, **kw):
        return self.s.submit(abi.OP_DELETE_CHILD_FIFO, struct.pack("<I", self.fid), **kw)

    def doorbell(self, written=None):
        """The guest stores `written` in the node, then writes 0x1020 = id."""
        s = self.s
        s.mem(self.node_gpa(0), struct.pack("<I", (self.written if written is None else written) & 0xFFFFFFFF))
        s.w(abi.R_WAKE_CHILD, self.fid)

    def submit(self, opcode, payload=b"", fault=False, timeout_ms=5000, label=None, **kw):
        st = self.cmd(opcode, payload, **kw)
        self.doorbell()
        if fault:
            self.s.wait(gpa=self.node_gpa(8), ne=0xFFFFFFFF, timeout_ms=timeout_ms,
                        label=label or "child %d fault" % self.fid)
            # The node's fault offset is written at once; 0x102c follows from
            # signal_fault_in_fifo, which runs asynchronously on the FIFO
            # queue, so a read of it must wait too.
            self.s.wait(mmio=abi.R_FAULT_MASK, mmio_ne=0, timeout_ms=timeout_ms,
                        label=(label or "child %d fault" % self.fid) + ": 0x102c")
        else:
            self.wait_stamp(st, timeout_ms=timeout_ms, label=label)
        return st
