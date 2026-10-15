"""The IOSurface host device (docs/testing.md, section 6.23), through QEMU's
32-bit MMIO declaration: its registers, the command ring (map 1, unmap 2,
test 3, anything else unsupported), descriptor tables v1 and v2, every
IOReturn status a command writes back, the descriptors' mapped bits, one
interrupt per command, and host-only surfaces.

Map and isolate setups only: apple-gfx-mmio creates the IOSurface host
device with a memory map, which it needs (apple-gfx-mmio.c:134-162), and
the device does not depend on the RvgDevice's setup at all.
"""
import struct

from pgtest.abi import PAGE
from pgtest.iosfc_guest import (BAD_ARGUMENT, BUSY, CMD_MAP, CMD_TEST, CMD_UNMAP, I_DESC_BASE, I_DESC_ENABLE, I_ONE,
                                I_READ, I_RING_BASE, I_RING_SIZE, I_WRITE, IOSFC, IREGS, NO_MEMORY, desc_v1, desc_v2,
                                plane_v1, plane_v2, pte)
from pgtest.registry import scenario
from pgtest.scenario import Scenario

SETUPS = ("map", "isolate")
W, H = 16, 8


def _s(name, doc, **kw):
    return Scenario(name, setups=SETUPS, iosfc=True, doc=doc, **kw)


def _regs(s, label):
    for r in IREGS:
        s.iosfc_r(r, label)


@scenario(setups=SETUPS)
def iosfc_registers():
    """Each register read back after a write, 32 bits wide: ring base and
    size, descriptor base (it maps a page), write counter, the read counter
    (written only while no ring is mapped), 0x1028 always 1, descriptor
    enable (bit 0 only; writing it clears the descriptor base). Other
    offsets, unaligned ones included, read 0xffffffff and ignore writes. A
    ring size with no base maps nothing and is not stored."""
    s = _s("iosfc_registers", iosfc_registers.__doc__)
    _regs(s, "fresh")
    for off in (0x0, 0x1004, 0x1034, 0x2000, 0xFFFC):
        s.iosfc_r(off, "unhandled")
    s.iosfc_w(I_RING_SIZE, 1024)
    s.iosfc_r(I_RING_SIZE, "size with no base")
    s.iosfc_w(I_READ, 5)
    s.iosfc_r(I_READ, "read counter, no ring")
    ring = s.alloc_page()
    table = s.alloc_page()
    s.iosfc_w(I_RING_BASE, s.pa(ring))
    s.iosfc_w(I_RING_SIZE, 1024)
    s.iosfc_w(I_READ, 9)
    s.iosfc_r(I_READ, "read counter, ring mapped")
    s.iosfc_w(I_DESC_BASE, s.pa(table))
    s.iosfc_w(I_DESC_ENABLE, 3)
    _regs(s, "enable 3")
    s.iosfc_w(I_DESC_BASE, s.pa(table))
    s.iosfc_w(I_WRITE, 5)
    s.iosfc_w(0x1004, 0x55)
    s.iosfc_w(0x2000, 0x55)
    s.quiesce()
    s.events("no descriptors: the counters stop the ring")
    _regs(s, "written")
    s.iosfc_w(I_RING_BASE, 0)
    s.iosfc_w(I_READ, 0)
    _regs(s, "ring base 0")
    s.diff("guest memory")
    return s


@scenario(setups=SETUPS)
def iosfc_map_test_unmap():
    """Map, test and unmap one BGRA surface over one guest page, v1 and v2:
    map sets the descriptor's mapped bit (bit 1) and writes status 0; test
    writes i x 8 into each 8-byte word of the surface, into the guest page;
    unmap clears the bit. A second map of a mapped id succeeds without
    re-creating it; unmapping an unmapped id succeeds; testing it fails
    (BadArgument). One interrupt a command."""
    out = []
    for v2 in (False, True):
        name = "iosfc_map_test_unmap_" + ("v2" if v2 else "v1")
        s = _s(name, iosfc_map_test_unmap.__doc__)
        d = IOSFC(s, v2=v2)
        d.start()
        root, data = d.surface_pages(1)
        mk = desc_v2 if v2 else desc_v1
        d.put_desc(1, mk(root, W, H))
        idx = d.submit((CMD_MAP, 1))
        d.status(idx, "map")
        s.dump(d.desc_gpa(1), 8, "descriptor after map")
        idx = d.submit((CMD_TEST, 1))
        d.status(idx, "test")
        s.diff("after test", gpa=s.pa(data), length=PAGE)
        idx = d.submit((CMD_MAP, 1), (CMD_UNMAP, 1), (CMD_UNMAP, 1), (CMD_TEST, 1))
        d.status(idx, "map again, unmap twice, test")
        s.dump(d.desc_gpa(1), 8, "descriptor after unmap")
        s.events("interrupts")
        _regs(s, "end")
        out.append(s)
    return out


@scenario(setups=SETUPS)
def iosfc_map_errors():
    """Map's statuses: id 0 and 8192 BadArgument; a descriptor table entry
    that is not valid NoMemory, one outside guest RAM NoMemory; a
    descriptor without its valid bit BadArgument; a page-table root outside
    RAM NoMemory; a root entry that is invalid or a table BadArgument; a
    surface page outside RAM NoMemory; 8 planes or a plane with 5
    components NoMemory; a descriptor IOSurface refuses (no size)
    BadArgument. Unmap and test of id 0 or 8192 BadArgument; other commands
    Unsupported. Each writes its status and raises one interrupt."""
    out = []
    for v2 in (False, True):
        name = "iosfc_map_errors_" + ("v2" if v2 else "v1")
        s = _s(name, iosfc_map_errors.__doc__)
        d = IOSFC(s, v2=v2)
        d.start()
        mk, pl = (desc_v2, plane_v2) if v2 else (desc_v1, plane_v1)
        per = d.per_page()
        root, data = d.surface_pages(1)
        bad_root = s.alloc_page()
        s.u32(s.pa(bad_root), pte(s.pfn(data), valid=0))
        table_root = s.alloc_page()
        s.u32(s.pa(table_root), pte(s.pfn(data), is_table=1))
        out_root = s.alloc_page()
        s.u32(s.pa(out_root), pte(0x10))
        cases = [
            ("id 0", 0, None),
            ("id 8192", 8192, None),
            ("descriptor invalid", 1, mk(root, W, H, valid=0)),
            ("root outside RAM", 2, mk(0x10, W, H)),
            ("root entry invalid", 3, mk(s.pfn(bad_root), W, H)),
            ("root entry is a table", 4, mk(s.pfn(table_root), W, H)),
            ("surface page outside RAM", 5, mk(s.pfn(out_root), W, H)),
            ("8 planes", 6, mk(root, W, H, planes=[pl(W, H, 64, 4)] * 7 + ([] if v2 else []))),
            ("5 components", 7, mk(root, W, H, planes=[pl(W, H, 64, 4, comps=5)])),
            ("4 components", 8, mk(root, W, H, planes=[pl(W, H, 64, 4, size=512, comps=4, depths=(8, 8, 8, 8),
                                                            offsets=(0, 8, 16, 24), names=(1, 2, 3, 4),
                                                            types=(1, 1, 1, 1), ranges=(1, 1, 1, 1))])),
            ("zero size", 9, mk(root, 0, 0, alloc=0)),
            ("power-of-two elements", 10, mk(root, W, H, ew=0x81, eh=0x80)),
        ]
        if v2:
            cases[7] = ("8 planes", 6, mk(root, W, H, plane_count=8))
        else:
            cases[7] = ("8 planes", 6, mk(root, W, H, planes=[pl(W, H, 64, 4)] * 7)[:0x24]
                        + bytes([8]) + mk(root, W, H, planes=[pl(W, H, 64, 4)] * 7)[0x25:])
        for label, mid, desc in cases:
            if desc is not None:
                d.put_desc(mid, desc)
        # An id whose descriptor-table entry is invalid, and one whose
        # descriptor page is outside RAM.
        invalid_page_id = 2 * per + 1
        s.u32(s.pa(d.table, 8), pte(s.pfn(data), valid=0))
        outside_page_id = 3 * per + 1
        s.u32(s.pa(d.table, 12), pte(0x10))
        cmds = [(CMD_MAP, mid) for _, mid, _ in cases] + [(CMD_MAP, invalid_page_id), (CMD_MAP, outside_page_id)]
        cmds += [(CMD_UNMAP, 0), (CMD_UNMAP, 8192), (CMD_TEST, 0), (CMD_TEST, 8192), (CMD_TEST, 11),
                 (0, 1), (4, 1), (0xFFFFFFFF, 1)]
        idx = d.submit(*cmds)
        labels = [c[0] for c in cases] + ["table entry invalid", "descriptor page outside RAM", "unmap 0",
                                          "unmap 8192", "test 0", "test 8192", "test unmapped", "command 0",
                                          "command 4", "command 0xffffffff"]
        for i, label in zip(idx, labels):
            d.status([i], label)
        for label, mid, desc in cases:
            if desc is not None:
                s.dump(d.desc_gpa(mid), 4, "descriptor word 0: " + label)
        s.events("interrupts")
        _regs(s, "end")
        out.append(s)
    return out


@scenario(setups=SETUPS)
def iosfc_two_level():
    """A surface of more than 4096 pages takes a two-level page table: root
    entries must be valid tables (else BadArgument), their tables must map
    (else NoMemory), and each table entry must be a valid page (else
    NoMemory) that maps (else NoMemory). A surface over 4096 x 4096 pages is
    NoMemory at once. v2 descriptors (a 64-bit allocSize); every page of the
    good surface is the same guest page."""
    s = _s("iosfc_two_level", iosfc_two_level.__doc__)
    d = IOSFC(s, v2=True)
    d.start()
    data = s.alloc_page()
    n = 4097
    alloc = n * PAGE
    t1, t2 = s.alloc_page(), s.alloc_page()
    s.mem(s.pa(t1), struct.pack("<I", pte(s.pfn(data))) * 4096)
    s.mem(s.pa(t2), struct.pack("<I", pte(s.pfn(data))))

    def root(entries):
        r = s.alloc_page()
        s.mem(s.pa(r), b"".join(struct.pack("<I", e) for e in entries))
        return s.pfn(r)

    bad_t2 = s.alloc_page()
    s.mem(s.pa(bad_t2), struct.pack("<I", pte(s.pfn(data), valid=0)))
    tab_t2 = s.alloc_page()
    s.mem(s.pa(tab_t2), struct.pack("<I", pte(s.pfn(data), is_table=1)))
    out_t2 = s.alloc_page()
    s.mem(s.pa(out_t2), struct.pack("<I", pte(0x10)))
    T = lambda p: pte(s.pfn(p), is_table=1)
    cases = [
        ("good", root([T(t1), T(t2)])),
        ("root entry not a table", root([T(t1), pte(s.pfn(t2))])),
        ("root entry invalid", root([T(t1), pte(s.pfn(t2), valid=0, is_table=1)])),
        ("table outside RAM", root([T(t1), pte(0x10, is_table=1)])),
        ("table entry invalid", root([T(t1), T(bad_t2)])),
        ("table entry is a table", root([T(t1), T(tab_t2)])),
        ("table entry outside RAM", root([T(t1), T(out_t2)])),
    ]
    for i, (label, r) in enumerate(cases):
        d.put_desc(1 + i, desc_v2(r, 64, 64, alloc=alloc, bpr=256))
    d.put_desc(9, desc_v2(cases[0][1], 64, 64, alloc=(4096 * 4096 + 1) * PAGE, bpr=256))
    idx = d.submit(*[(CMD_MAP, 1 + i) for i in range(len(cases))], (CMD_MAP, 9))
    for i, label in zip(idx, [c[0] for c in cases] + ["over 4096 x 4096 pages"]):
        d.status([i], label)
    idx = d.submit((CMD_UNMAP, 1))
    d.status(idx, "unmap the good one")
    s.events("interrupts")
    return s


@scenario(setups=SETUPS)
def iosfc_host_only():
    """Host-only surfaces (v2 bit 1 of +0x41) need no page table and never
    touch guest memory: test writes the host's copy. Their total, in whole
    pages, is capped at 1 GiB: one page, 512 MiB and 512 MiB less two pages
    fit, a fourth one-page surface reaches exactly 1 GiB and fits (the check
    is >), a fifth does not (NoMemory); unmap gives the budget back."""
    s = _s("iosfc_host_only", iosfc_host_only.__doc__)
    d = IOSFC(s, v2=True)
    d.start()
    d.put_desc(1, desc_v2(0, W, H, host_only=1))
    d.put_desc(2, desc_v2(0, 8192, 16384, bpr=32768, host_only=1))
    d.put_desc(3, desc_v2(0, 8192, 16383, bpr=32768, host_only=1))
    d.put_desc(4, desc_v2(0, W, H, host_only=1))
    d.put_desc(5, desc_v2(0, W, H, host_only=1))
    idx = d.submit((CMD_MAP, 1), (CMD_TEST, 1), (CMD_MAP, 2), (CMD_MAP, 3), (CMD_MAP, 4), (CMD_MAP, 5))
    for i, label in zip(idx, ["map 16x8", "test", "map 512 MiB", "map 512 MiB - 32 KiB", "map to 1 GiB",
                              "map past 1 GiB"]):
        d.status([i], label)
    idx = d.submit((CMD_UNMAP, 1), (CMD_MAP, 5), (CMD_UNMAP, 2), (CMD_UNMAP, 3), (CMD_UNMAP, 4), (CMD_UNMAP, 5))
    for i, label in zip(idx, ["unmap 16x8", "map again", "unmap", "unmap", "unmap", "unmap"]):
        d.status([i], label)
    s.diff("guest memory")
    s.events("interrupts")
    return s


@scenario(setups=SETUPS)
def iosfc_host_only_teardown():
    """A descriptor base write tears host-only surfaces down and gives
    their pages back to the 1 GiB budget: two 512 MiB surfaces, the
    base written again, and both map again, which they could not if the
    budget were still spent."""
    s = _s("iosfc_host_only_teardown", iosfc_host_only_teardown.__doc__)
    d = IOSFC(s, v2=True)
    d.start()
    d.put_desc(1, desc_v2(0, 8192, 16384, bpr=32768, host_only=1))
    d.put_desc(2, desc_v2(0, 8192, 16384, bpr=32768, host_only=1))
    d.put_desc(3, desc_v2(0, W, H, host_only=1))
    idx = d.submit((CMD_MAP, 1), (CMD_MAP, 2), (CMD_MAP, 3))
    d.status(idx, "1 GiB, then one page past it")
    s.iosfc_w(I_DESC_BASE, s.pa(d.table))
    idx = d.submit((CMD_MAP, 1), (CMD_MAP, 2), (CMD_UNMAP, 1), (CMD_UNMAP, 2))
    d.status(idx, "after the base write")
    s.events("interrupts")
    return s


@scenario(setups=SETUPS)
def iosfc_ring():
    """The ring: commands run from the read counter to the write counter,
    entry i at i mod size, wrapping; a doorbell with no descriptor table
    runs nothing, and setting the table later runs nothing until the next
    doorbell. A write counter below the read counter runs nothing. The read
    counter is writable while no ring is mapped: a ring base write unmaps
    it; a ring too large for guest RAM does not map."""
    s = _s("iosfc_ring", iosfc_ring.__doc__)
    d = IOSFC(s)
    s.iosfc_w(I_READ, 1022)
    s.iosfc_w(I_WRITE, 1022)
    d.written = 1022
    d.start(table=False)
    d.submit((0, 1), (0, 2), wait=False)
    s.quiesce()
    s.events("no descriptor table")
    s.iosfc_w(I_DESC_BASE, s.pa(d.table))
    s.quiesce()
    s.events("table set, no doorbell")
    s.iosfc_w(I_WRITE, d.written)
    d.irqs = 2
    s.wait(iosfc_irqs=2)
    d.submit((0, 3), (0, 4), (0, 5))
    s.dump(d.cmd_gpa(1022), 32, "ring entries 1022, 1023")
    s.dump(d.cmd_gpa(0), 48, "ring entries 0-2, wrapped")
    _regs(s, "wrapped")
    s.iosfc_w(I_WRITE, 3)
    s.quiesce()
    s.events("write counter below the read counter")
    _regs(s, "counter below")
    s.iosfc_w(I_RING_BASE, s.pa(d.ring))
    s.iosfc_w(I_RING_SIZE, 0x10000000)
    s.iosfc_w(I_READ, 0)
    s.iosfc_w(I_WRITE, 1)
    s.quiesce()
    s.events("ring too large to map")
    _regs(s, "ring too large")
    s.iosfc_w(I_RING_SIZE, 1024)
    s.mem(d.cmd_gpa(0), struct.pack("<IIiI", 9, 1, 0, 0))
    s.iosfc_w(I_WRITE, 1)
    s.wait(iosfc_irqs=6)
    s.dump(d.cmd_gpa(0), 16, "ring entry 0 at counter 0")
    s.events("end")
    return s


@scenario(setups=SETUPS)
def iosfc_ring_size_unaligned():
    """A ring size whose 16-byte entries are not a whole number of 16 KiB
    pages asserts in the memory map (totalLength % 16 KiB) when the size
    is written with a base set: the host dies at that write."""
    s = _s("iosfc_ring_size_unaligned", iosfc_ring_size_unaligned.__doc__)
    ring = s.alloc_page()
    s.iosfc_w(I_RING_BASE, s.pa(ring))
    s.iosfc_w(I_RING_SIZE, 4)
    return s


@scenario(setups=SETUPS)
def iosfc_descriptor_base_change():
    """Writing the descriptor base (or the enable) tears every surface down,
    leaving the guest's mapped bits as they were; mapping again creates it
    anew. Switching v1 to v2 re-reads the table with 0x400-byte descriptors:
    the same id lands elsewhere."""
    s = _s("iosfc_descriptor_base_change", iosfc_descriptor_base_change.__doc__)
    d = IOSFC(s)
    d.start()
    root, data = d.surface_pages(1)
    d.put_desc(1, desc_v1(root, W, H))
    d.put_desc(2, desc_v1(root, W, H))
    d.submit((CMD_MAP, 1), (CMD_MAP, 2))
    s.iosfc_w(I_DESC_BASE, s.pa(d.table))
    idx = d.submit((CMD_TEST, 1), (CMD_UNMAP, 2), (CMD_MAP, 1), (CMD_TEST, 1))
    d.status(idx, "after the base write")
    s.dump(d.desc_gpa(1), 4, "descriptor 1")
    s.dump(d.desc_gpa(2), 4, "descriptor 2")
    s.iosfc_w(I_DESC_ENABLE, 1)
    s.iosfc_w(I_DESC_BASE, s.pa(d.table))
    idx = d.submit((CMD_TEST, 1), (CMD_MAP, 1))
    d.status(idx, "v2 over v1 descriptors")
    s.diff("surface page", gpa=s.pa(data), length=PAGE)
    _regs(s, "end")
    s.events("end")
    return s


@scenario(setups=SETUPS)
def iosfc_surface_options():
    """Descriptor fields that reach the IOSurface: a nonzero offset (test
    writes from the base minus the offset), read-only (the surface is made
    with client direction 2), planes, v1 descriptors with a read-only bit.
    Each map's status and each test's writes into guest memory are
    locked."""
    out = []
    for v2 in (False, True):
        s = _s("iosfc_surface_options_" + ("v2" if v2 else "v1"), iosfc_surface_options.__doc__)
        d = IOSFC(s, v2=v2)
        d.start()
        mk, pl = (desc_v2, plane_v2) if v2 else (desc_v1, plane_v1)
        pages = []
        cases = [("offset 256", dict(offset=256)),
                 ("read-only", dict(read_only=1)),
                 ("one plane", dict(planes=[pl(W, H, 64, 4, size=512)])),
                 ("two planes", dict(planes=[pl(W, H, 64, 4, size=512), pl(W // 2, H // 2, 32, 4, offset=512,
                                                                           size=128)], alloc=1024)),
                 ("RGBA 8 bytes a pixel", dict(fmt=b"RGhA", bpe=8))]
        for i, (label, kw) in enumerate(cases):
            root, data = d.surface_pages(1)
            pages.append(data)
            d.put_desc(1 + i, mk(root, W, H, **kw))
        cmds = []
        for i in range(len(cases)):
            cmds += [(CMD_MAP, 1 + i), (CMD_TEST, 1 + i)]
        idx = d.submit(*cmds)
        for j, (label, _) in enumerate(cases):
            d.status(idx[2 * j:2 * j + 2], label)
            s.diff(label, gpa=s.pa(pages[j]), length=PAGE)
        s.events("end")
        out.append(s)
    return out
