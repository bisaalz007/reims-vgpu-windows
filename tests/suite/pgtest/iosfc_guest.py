"""The IOSurface host device's guest side (docs/testing.md, section 6.23):
its registers, the command ring, the descriptor table and the descriptors,
v1 (0x200 bytes) and v2 (0x400 bytes), and a surface's page tables.

Layouts are reims-vgpu-core's iosurface::descriptor; the device is its
iosurface module. vm_page_size is 16 KiB on the hosts the suite runs on, so
a descriptor page holds 32 v1 or 16 v2 descriptors, and a page-table page
4096 entries. Registers go through QEMU's 32-bit calls
(apple-gfx-mmio.c:45-58): a value written or read keeps its low 32 bits,
zero-extended.
"""
import struct

from .abi import PAGE

# Registers (iosurface::regs, mmio_read and mmio_write).
I_RING_BASE = 0x1000
I_RING_SIZE = 0x1008
I_DESC_BASE = 0x1010
I_WRITE = 0x1018
I_READ = 0x1020
I_ONE = 0x1028
I_DESC_ENABLE = 0x1030
IREGS = (I_RING_BASE, I_RING_SIZE, I_DESC_BASE, I_WRITE, I_READ, I_ONE, I_DESC_ENABLE)

CMD_MAP, CMD_UNMAP, CMD_TEST = 1, 2, 3

# IOReturn values (IOKit/IOReturn.h).
SUCCESS = 0
NO_MEMORY = 0xE00002BD
BAD_ARGUMENT = 0xE00002C2
UNSUPPORTED = 0xE00002C7
BUSY = 0xE00002D5


def pte(page_number, valid=1, is_table=0):
    """A descriptor-table or page-table entry: {valid:1, isTable:1,
    pageNumber:30} (IoSurfaceDescriptorPageTableEntry)."""
    return (valid & 1) | (is_table & 1) << 1 | (page_number & 0x3FFFFFFF) << 2


def plane_v1(width=0, height=0, bpr=0, bpe=0, offset=0, base=0, size=0, ew=0, eh=0, comps=0, depths=(),
             names=(), types=(), ranges=(), offsets=()):
    """IOSurfaceDevicePlaneDescriptor, 0x40 bytes."""
    def arr(v):
        return bytes(list(v) + [0] * (4 - len(v)))
    return (struct.pack("<IIIII", 0, 0, offset, base, size)
            + struct.pack("<II", (ew & 0xFF) | (width & 0xFFFFFF) << 8, (eh & 0xFF) | (height & 0xFFFFFF) << 8)
            + struct.pack("<IHH", bpr, bpe, 0)
            + struct.pack("<BBBB", 0, 0, 0, comps) + bytes(4)
            + arr(depths) + arr(offsets) + arr(names) + arr(types) + arr(ranges))


def desc_v1(root, width, height, fmt=b"BGRA", alloc=None, bpr=None, bpe=4, valid=1, mapped=0, planes=(),
            offset=0, read_only=0, ew=1, eh=1):
    """IOSurfaceDeviceDescriptor, 0x200 bytes: the 0x40-byte base and 7
    planes."""
    bpr = width * bpe if bpr is None else bpr
    alloc = bpr * height if alloc is None else alloc
    fmtv = struct.unpack(">I", fmt)[0] if isinstance(fmt, bytes) else fmt
    base = struct.pack("<IIIII", (valid & 1) | (mapped & 1) << 1 | (root & 0x3FFFFFFF) << 2, fmtv, offset, 0, alloc)
    base += struct.pack("<II", (ew & 0xFF) | (width & 0xFFFFFF) << 8, (eh & 0xFF) | (height & 0xFFFFFF) << 8)
    base += struct.pack("<IHH", bpr, bpe, 0)
    base += struct.pack("<BBBB", len(planes), 0, 0, read_only & 1) + bytes(4) + struct.pack("<I", 0) + bytes(16)
    assert len(base) == 0x40
    ps = b"".join(planes) + bytes(0x40 * (7 - len(planes)))
    return base + ps


def plane_v2(width=0, height=0, bpr=0, bpe=0, offset=0, base=0, size=0, ew=0, eh=0, comps=0, depths=(),
             offsets=(), names=(), types=(), ranges=()):
    """IOSurfaceDevicePlaneDescriptor2, 0x80 bytes."""
    def arr(v):
        return bytes(list(v) + [0] * (4 - len(v)))
    p = struct.pack("<IIQQQQQQ", 0, 0, offset, base, size, width, height, bpr)
    p += struct.pack("<HBBBBBB", bpe, ew, eh, 0, 0, 0, comps)
    p += struct.pack("<IIII", 0, 0, 0, 0)
    p += arr(depths) + arr(offsets) + arr(names) + arr(types) + arr(ranges) + bytes(28)
    assert len(p) == 0x80
    return p


def desc_v2(root, width, height, fmt=b"BGRA", alloc=None, bpr=None, bpe=4, valid=1, mapped=0, planes=(),
            offset=0, read_only=0, host_only=0, protection=0, plane_count=None, ew=1, eh=1):
    """IOSurfaceDeviceDescriptor2, 0x400 bytes: the 0x80-byte base and 7
    planes."""
    bpr = width * bpe if bpr is None else bpr
    alloc = bpr * height if alloc is None else alloc
    fmtv = struct.unpack(">I", fmt)[0] if isinstance(fmt, bytes) else fmt
    b = struct.pack("<IIQQQQQQ", (valid & 1) | (mapped & 1) << 1 | (root & 0x3FFFFFFF) << 2, fmtv, offset, 0,
                    alloc, width, height, bpr)
    b += struct.pack("<HHBBBBB", bpe, 0, ew, eh, len(planes) if plane_count is None else plane_count, 0, 0)
    b += struct.pack("<BBB", (read_only & 1) | (host_only & 1) << 1, 0, 0)
    b += struct.pack("<IIIII", 0, 0, 0, 0, 0) + struct.pack("<Q", protection) + bytes(32)
    assert len(b) == 0x80, len(b)
    ps = b"".join(planes) + bytes(0x80 * (7 - len(planes)))
    return b + ps


class IOSFC:
    """A guest driver for the IOSurface host device: a ring of `entries`
    16-byte commands (a whole number of 16 KiB pages: the memory map
    asserts that a mapping's length is a page multiple), a
    descriptor table page, descriptor pages allocated on demand. v2 selects
    0x400-byte descriptors (0x1030 bit 0)."""

    def __init__(self, s, entries=1024, v2=False):
        self.s = s
        self.v2 = v2
        self.entries = entries
        self.ring = s.alloc_page()
        self.table = s.alloc_page()
        self.desc_pages = {}
        self.written = 0
        # A new process's IOSurface interrupts count from 0 (Scenario.phase).
        s.on_phase(lambda: setattr(self, "irqs", 0))

    def per_page(self):
        return PAGE // (0x400 if self.v2 else 0x200)

    def start(self, enable=True, ring=True, table=True):
        s = self.s
        if enable:
            s.iosfc_w(I_DESC_ENABLE, 1 if self.v2 else 0)
        if table:
            s.iosfc_w(I_DESC_BASE, s.pa(self.table))
        if ring:
            s.iosfc_w(I_RING_BASE, s.pa(self.ring))
            s.iosfc_w(I_RING_SIZE, self.entries)

    def desc_gpa(self, mid):
        """The guest address of descriptor mid; allocates and links its
        descriptor page into the table."""
        pi, idx = divmod(mid, self.per_page())
        if pi not in self.desc_pages:
            p = self.s.alloc_page()
            self.desc_pages[pi] = p
            self.s.u32(self.s.pa(self.table, 4 * pi), pte(self.s.pfn(p)))
        return self.s.pa(self.desc_pages[pi], idx * (0x400 if self.v2 else 0x200))

    def put_desc(self, mid, data):
        self.s.mem(self.desc_gpa(mid), data)

    def surface_pages(self, n):
        """n contiguous data pages and a one-level page table naming them;
        returns (table page number, first data RAM page)."""
        s = self.s
        pt = s.alloc_page()
        data = s.alloc_page(n)
        s.mem(s.pa(pt), b"".join(struct.pack("<I", pte(s.pfn(data + i))) for i in range(n)))
        return s.pfn(pt), data

    def cmd_gpa(self, i):
        return self.s.pa(self.ring, 16 * (i % self.entries))

    def submit(self, *cmds, wait=True):
        """Writes (command, mappingID) entries at the write counter, rings
        0x1018, and waits for one interrupt a command. Returns the entries'
        indices."""
        s = self.s
        idx = []
        for c, mid in cmds:
            s.mem(self.cmd_gpa(self.written), struct.pack("<IIiI", c, mid, 0x7EADBEEF, 0))
            idx.append(self.written)
            self.written += 1
        s.iosfc_w(I_WRITE, self.written)
        if wait:
            self.irqs = getattr(self, "irqs", 0) + len(cmds)
            s.wait(iosfc_irqs=self.irqs, label="iosfc irqs %d" % self.irqs)
        return idx

    def status(self, idx, label=None):
        """Dumps the ring entries (their status words)."""
        for i in idx:
            self.s.dump(self.cmd_gpa(i), 16, label=label)
