"""The mapped-range tracker, the IOSurface host device and the guest display
at their edges: each scenario locks one behaviour, as its docstring says. The
docstrings say what the code does, which is what the expected files record.
"""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.display_guest import display, icc_profile, p_icc, p_swap
from pgtest.iosfc_guest import CMD_MAP, CMD_TEST, I_DESC_BASE, I_DESC_ENABLE, IOSFC, desc_v1, desc_v2, plane_v1, plane_v2
from pgtest.newops_util import AliasedTask, bring_up_big
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.tasks_lib import XTask, probe_list

BLOCKS = ("blocks",)
MAPPED = ("map", "isolate")
W, H = 16, 8


def steps(s, t, plan):
    """Maps, unmaps and lazy probes (probe_list), each followed by the
    events it caused (as scenarios/tasks_gaps.py's steps): the host's
    map_memory calls show what the tracker thought unmapped."""
    for kind, p, n in plan:
        label = "%s [%d, %d)" % (kind, p, p + n)
        if kind == "map":
            t.map(p * PAGE, n * PAGE)
        elif kind == "unmap":
            t.unmap(p * PAGE, n * PAGE)
        else:
            probe_list(s, t.tid, p, n, label)
        s.events(label)


# ---- the mapped-range tracker ----

@scenario(setups=BLOCKS)
def mutgap4_sys_tracker_split_kept_last_run():
    """RunBlock::split gives the kept half the span of all its runs, up to
    the end of its last run. A full block of 240 one-page runs at the even
    pages 100-578 and a run at 700 (a second block); a run at 650 splits the
    full block, keeping [100..338] (120 runs) and moving [340..578] to a new
    block that takes 650. A probe of page 338, the kept half's last run,
    finds it mapped (no map_memory); one of page 336 too; one of page 339, the
    gap after it, maps that page. Blocks only."""
    s = Scenario("mutgap4_sys_tracker_split_kept_last_run", setups=BLOCKS,
                 doc=mutgap4_sys_tracker_split_kept_last_run.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=1100, alias=s.alloc_page())
    t.define()
    for p in range(100, 580, 2):
        t.map(p * PAGE, PAGE)
    s.events("240 runs from page 100")
    steps(s, t, [("map", 700, 1), ("map", 650, 1), ("probe", 338, 1), ("probe", 336, 1), ("probe", 339, 1)])
    t.delete()
    s.events("end")
    return s


@scenario(setups=BLOCKS)
def mutgap4_sys_tracker_merge_then_unmap():
    """A map that overlaps the end of one run and the start of the next is
    merged into the first, and the clean-up that follows trims the next run
    to what the merged run leaves of it (RunBlock::add): runs [0, 2) and [3,
    5), then a map of [1, 4) leaves [0, 4) and [4, 5). An unmap of [3, 4)
    then trims [0, 4) to [0, 3), so a probe of page 3 maps it again (one
    map_memory call), and a probe of [0, 3) and of [4, 5) maps nothing. Blocks
    only."""
    s = Scenario("mutgap4_sys_tracker_merge_then_unmap", setups=BLOCKS,
                 doc=mutgap4_sys_tracker_merge_then_unmap.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=64, alias=s.alloc_page())
    t.define()
    steps(s, t, [("map", 0, 2), ("map", 3, 2), ("map", 1, 3), ("unmap", 3, 1), ("probe", 3, 1),
                 ("probe", 0, 3), ("probe", 4, 1)])
    t.delete()
    s.events("end")
    return s


@scenario(setups=BLOCKS)
def mutgap4_sys_tracker_unmap_stops_at_empty_block():
    """An unmap stops at the first block that compares after it, and an
    emptied block compares after everything (remove_from_range;
    Run::compare). A full block of 240 one-page runs at the even pages
    100-578 and a run at 1000 (past the full last block: a new block B1); a
    run at 650 splits the full block ([100..338] and [340..578] + 650); a map
    of [339, 651) joins the first half, and the second half, inside it at its
    end, is emptied and kept. An unmap of [1000, 1001) stops at the empty
    block, so B1 keeps its run; a map of [1001, 1002) joins B1, behind the
    empty block, and a probe of page 1001 maps it again (one map_memory call),
    since the scan stops at the empty block. Blocks only."""
    s = Scenario("mutgap4_sys_tracker_unmap_stops_at_empty_block", setups=BLOCKS,
                 doc=mutgap4_sys_tracker_unmap_stops_at_empty_block.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=1100, alias=s.alloc_page())
    t.define()
    for p in range(100, 580, 2):
        t.map(p * PAGE, PAGE)
    s.events("240 runs from page 100")
    steps(s, t, [("map", 1000, 1), ("map", 650, 1), ("map", 339, 312), ("unmap", 1000, 1), ("map", 1001, 1),
                 ("probe", 1001, 1)])
    t.delete()
    s.events("end")
    return s


@scenario(setups=BLOCKS)
def mutgap4_sys_tracker_compact_then_unmap_block():
    """Compaction moves a partly stolen block's span to its new first run, so
    a compacted block's span is exactly its runs. As
    mutgap_mem_tracker_compact_stale_span up to its compaction: a task of 16
    x 4096 pages, every leaf one RAM page, 30,720 one-page runs at every
    other page (128 blocks of 240), 200 runs unmapped in each of blocks 10
    and 11, and a map into a gap near the end of block 127 that compacts to
    127 blocks: block 10 takes block 11's 40 runs and 160 of block 12's, and
    every later block k keeps the last 80 runs of old block k + 1 and takes
    the first 160 of old block k + 2. Block 20 then holds [10400, 10879)
    (runs at the even pages 10400-10878). An unmap of exactly that range
    equals block 20's span, and the block is removed; a probe of page 10880,
    block 21's first run, maps nothing, and one of page 10878, unmapped, maps
    it. Blocks only; block calls as counts and a digest."""
    s = Scenario("mutgap4_sys_tracker_compact_then_unmap_block", setups=BLOCKS, watchdog_s=120,
                 doc=mutgap4_sys_tracker_compact_then_unmap_block.__doc__)
    f = bring_up_big(s)
    t = AliasedTask(s, tid=1, tables=16)
    t.define()
    total = 240 * 128
    half = f.ring_size() // 32 // 2
    k = 0
    while k < total:
        n = min(half, total - k)
        st = t.map_seq(f, n, 2 * k, 2)
        f.doorbell()
        f.wait_stamp(st, timeout_ms=60000)
        k += n
    s.r(abi.R_FAULT_MASK, "no fault after the fill")
    s.events("30,720 runs mapped", blocks_digest=True)

    def blk(b, i):
        return 2 * (240 * b + i)

    t.unmap(blk(10, 20), 400)
    t.unmap(blk(11, 20), 400)
    t.map(blk(127, 200) + 1)
    s.r(abi.R_FAULT_MASK, "compacted to 127 blocks")
    s.events("unmaps in blocks 10 and 11, a map that compacts", blocks_digest=True)
    t.unmap(10400, 479)
    s.events("unmap [10400, 10879)", blocks_digest=True)
    probe_list(s, 1, 10880, 1, "probe [10880, 10881)")
    s.events("probe [10880, 10881)", blocks_digest=True)
    probe_list(s, 1, 10878, 1, "probe [10878, 10879)")
    s.events("probe [10878, 10879)", blocks_digest=True)
    return s


# ---- the IOSurface host device ----

def _patched(data, off, value, size=1):
    b = bytearray(data)
    b[off:off + size] = value.to_bytes(size, "little")
    return bytes(b)


def _good_plane(mk, i=0, **kw):
    """A 16x8 plane of 1x1 elements based at its offset, with four 8-bit
    components, which IOSurface accepts (as mutgap3_res's good plane)."""
    args = dict(offset=512 * i, base=512 * i, size=512, ew=1, eh=1, comps=4, offsets=(0, 8, 16, 24),
                depths=(8, 8, 8, 8), names=(1, 2, 3, 4), types=(1, 1, 1, 1), ranges=(1, 1, 1, 1))
    args.update(kw)
    return mk(W, H, 64, 4, **args)


def _map_each(s, d, descs, labels):
    """Puts each descriptor at ids 1.. over a data page of its own, then maps
    and tests each in turn, locking both statuses and the data page."""
    pages = []
    for i, mkdesc in enumerate(descs):
        root, data = d.surface_pages(1)
        pages.append(data)
        d.put_desc(1 + i, mkdesc(root))
    for i, label in enumerate(labels):
        idx = d.submit((CMD_MAP, 1 + i), (CMD_TEST, 1 + i))
        d.status(idx, label)
        s.diff(label + ": data page", gpa=s.pa(pages[i]), length=PAGE)


def _both(fn, name, doc):
    out = []
    for v2 in (False, True):
        s = Scenario(name + ("_v2" if v2 else "_v1"), setups=MAPPED, iosfc=True, doc=doc)
        d = IOSFC(s, v2=v2)
        d.start()
        fn(s, d, v2)
        s.events("end")
        out.append(s)
    return out


@scenario(setups=MAPPED)
def mutgap4_sys_iosfc_base_write_then_teardown():
    """A descriptor base write (and a descriptor enable write, which writes the
    base 0) releases each cached descriptor-page mapping and forgets it, so the
    device has none to release at teardown: a v1 surface mapped on descriptor
    page 0, then the write, and nothing mapped again; the runner's teardown
    ends normally."""
    out = []
    for kind in ("base", "enable"):
        s = Scenario("mutgap4_sys_iosfc_%s_write_then_teardown" % kind, setups=MAPPED, iosfc=True,
                     doc=mutgap4_sys_iosfc_base_write_then_teardown.__doc__)
        d = IOSFC(s)
        d.start()
        root, data = d.surface_pages(1)
        d.put_desc(1, desc_v1(root, W, H))
        idx = d.submit((CMD_MAP, 1))
        d.status(idx, "mapped")
        if kind == "base":
            s.iosfc_w(I_DESC_BASE, s.pa(d.table))
        else:
            s.iosfc_w(I_DESC_ENABLE, 1)
        s.events("end")
        out.append(s)
    return out


@scenario(setups=MAPPED)
def mutgap4_sys_iosfc_offset():
    """A surface's offset is kept for the test command, which writes its
    pattern from the surface's base address minus the offset: 16x8 BGRA
    surfaces (bytesPerRow 64) with offset 256 and an allocation of 1024, and
    with offset 512 and an allocation of 1024, v1 and v2, each mapped then
    tested; each status and data page is locked. Measured: both map, and the
    test's 1024-byte pattern starts at the data page's first byte."""
    def body(s, d, v2):
        mk = desc_v2 if v2 else desc_v1
        _map_each(s, d, [lambda r: mk(r, W, H, offset=256, alloc=1024),
                         lambda r: mk(r, W, H, offset=512, alloc=1024)],
                  ["offset 256, alloc 1024", "offset 512, alloc 1024"])
    return _both(body, "mutgap4_sys_iosfc_offset", mutgap4_sys_iosfc_offset.__doc__)


@scenario(setups=MAPPED)
def mutgap4_sys_iosfc_plane_size():
    """A plane's size reaches the IOSurface as kIOSurfacePlaneSize: one-plane
    16x8 BGRA surfaces (a good plane, bytesPerRow 64) whose plane size is 256
    (less than its rows), 512 (its rows) and 1024 (with an allocation of
    1024), v1 and v2, each mapped then tested; each status and data page is
    locked. Measured: the 256-byte plane is refused (BadArgument, the test
    too, the page untouched); the others map and the test writes the
    allocation."""
    def body(s, d, v2):
        mk, pl = (desc_v2, plane_v2) if v2 else (desc_v1, plane_v1)
        cases = [(256, 512), (512, 512), (1024, 1024)]
        _map_each(s, d, [lambda r, sz=sz, a=a: mk(r, W, H, alloc=a, planes=[_good_plane(pl, size=sz)])
                         for sz, a in cases],
                  ["plane size %d, alloc %d" % c for c in cases])
    return _both(body, "mutgap4_sys_iosfc_plane_size", mutgap4_sys_iosfc_plane_size.__doc__)


@scenario(setups=MAPPED)
def mutgap4_sys_iosfc_plane_extended_left():
    """A plane's extendedPixelsOnLeft reaches the IOSurface as
    kIOSurfacePlaneExtendedPixelsOnLeft: one-plane 16x8 BGRA surfaces (a good
    plane) with 8 extra columns on the left in a plane of 512 bytes
    (bytesPerRow 64) and in one of 768 bytes (bytesPerRow 96), then 0, v1
    (plane byte 0x28) and v2 (plane +0x40), each mapped then tested; each
    status and data page is locked. Measured: the 512-byte plane with 8 left
    columns is refused (BadArgument), the 768-byte one maps (the test writes
    768 bytes), and the plane with none maps."""
    def body(s, d, v2):
        mk, pl = (desc_v2, plane_v2) if v2 else (desc_v1, plane_v1)
        off, sz = (0x40, 4) if v2 else (0x28, 1)
        cases = [(8, 64, 512), (8, 96, 768), (0, 64, 512)]

        def one(r, ext, bpr, size):
            p = _patched(pl(W, H, bpr, 4, offset=0, base=0, size=size, ew=1, eh=1, comps=4, offsets=(0, 8, 16, 24),
                            depths=(8, 8, 8, 8), names=(1, 2, 3, 4), types=(1, 1, 1, 1), ranges=(1, 1, 1, 1)),
                         off, ext, sz)
            return mk(r, W, H, alloc=max(size, 512), bpr=bpr, planes=[p])
        _map_each(s, d, [lambda r, c=c: one(r, *c) for c in cases],
                  ["extendedPixelsOnLeft %d, bytesPerRow %d, plane size %d" % c for c in cases])
    return _both(body, "mutgap4_sys_iosfc_plane_extended_left", mutgap4_sys_iosfc_plane_extended_left.__doc__)


@scenario(setups=MAPPED)
def mutgap4_sys_iosfc_address_format():
    """A descriptor's base addressFormat reaches the IOSurface as
    kIOSurfaceAddressFormat (a v1 descriptor's is copied into the v2 form):
    16x8 BGRA surfaces, no planes, with address format 1, 2, 3 and 0 (v1 base
    byte 0x25, v2 base +0x3f), the first of them the process's first map,
    each mapped then tested; each status and data page is locked. Measured:
    IOSurface ignores the key, every format maps and tests as format 0 (so
    neither the key nor its copy from v1 shows here)."""
    def body(s, d, v2):
        mk = desc_v2 if v2 else desc_v1
        off = 0x3F if v2 else 0x25
        fmts = (1, 2, 3, 0)
        _map_each(s, d, [lambda r, f=f: _patched(mk(r, W, H), off, f) for f in fmts],
                  ["addressFormat %d" % f for f in fmts])
    return _both(body, "mutgap4_sys_iosfc_address_format", mutgap4_sys_iosfc_address_format.__doc__)


@scenario(setups=MAPPED)
def mutgap4_sys_iosfc_base_extended_pixels():
    """A descriptor's base extendedPixelsOnRight and extendedPixelsOnBottom
    reach the IOSurface as the surface-level kIOSurfacePlaneExtendedPixels
    keys (a v1 descriptor's are copied into the v2 form): planeless 16x8 BGRA
    surfaces with 16 extra columns on the right in 512 bytes (bytesPerRow 64)
    and in 1024 bytes (bytesPerRow 128), then 8 extra rows at the bottom in
    512 and in 1024 bytes, then none (v1 base bytes 0x2a and 0x2b; v2 base
    +0x4c and +0x50), the first of them the process's first map, each mapped
    then tested; each status and data page is locked. Measured: for a
    planeless surface IOSurface ignores both keys (unlike the plane-level
    ones), and every surface maps and the test writes its allocation (so
    neither the keys nor their copy from v1 shows here)."""
    def body(s, d, v2):
        mk = desc_v2 if v2 else desc_v1
        roff, boff, sz = (0x4C, 0x50, 4) if v2 else (0x2A, 0x2B, 1)
        cases = [("right", 16, 64, 512), ("right", 16, 128, 1024), ("bottom", 8, 64, 512),
                 ("bottom", 8, 64, 1024), ("none", 0, 64, 512)]

        def one(r, kind, n, bpr, alloc):
            data = mk(r, W, H, bpr=bpr, alloc=alloc)
            if kind == "right":
                data = _patched(data, roff, n, sz)
            elif kind == "bottom":
                data = _patched(data, boff, n, sz)
            return data
        _map_each(s, d, [lambda r, c=c: one(r, *c) for c in cases],
                  ["extended %s %d, bytesPerRow %d, alloc %d" % c for c in cases])
    return _both(body, "mutgap4_sys_iosfc_base_extended_pixels",
                 mutgap4_sys_iosfc_base_extended_pixels.__doc__)


@scenario(setups=MAPPED)
def mutgap4_sys_iosfc_v1_width():
    """A v1 descriptor's width is copied into the v2 form the map works on
    and reaches the IOSurface as kIOSurfaceWidth: v1 BGRA surfaces of height
    8, an 8-wide one (bytesPerRow 32, 256 bytes) first, then 4096 wide in
    bytesPerRow 64, 17 wide in bytesPerRow 64, and 16 wide in bytesPerRow 64;
    each mapped then tested; each status and data page is locked. Measured:
    widths 4096 and 17 are refused (BadArgument), 8 and 16 map."""
    s = Scenario("mutgap4_sys_iosfc_v1_width", setups=MAPPED, iosfc=True, doc=mutgap4_sys_iosfc_v1_width.__doc__)
    d = IOSFC(s, v2=False)
    d.start()
    cases = [(8, 32, 256), (4096, 64, 512), (17, 64, 512), (16, 64, 512)]
    _map_each(s, d, [lambda r, c=c: desc_v1(r, c[0], H, bpr=c[1], alloc=c[2]) for c in cases],
              ["width %d, bytesPerRow %d, alloc %d" % c for c in cases])
    s.events("end")
    return s


@scenario(setups=MAPPED)
def mutgap4_sys_iosfc_v1_after_host_only():
    """A v1 descriptor maps a surface over guest pages whatever the v2
    descriptor mapped before it was: the upgrade to the v2 form clears
    hostOnly. v2 descriptors: eight host-only 16x8 surfaces mapped (and their
    descriptor copies freed), then the enable written 0 and the table again
    (v1), and a v1 16x8 surface over a data page, mapped then tested: the
    test writes its pattern into the data page. Measured: without that clear
    the map does the same (the freed copy never hands its hostOnly bit to the
    next map on this host)."""
    s = Scenario("mutgap4_sys_iosfc_v1_after_host_only", setups=MAPPED, iosfc=True,
                 doc=mutgap4_sys_iosfc_v1_after_host_only.__doc__)
    d = IOSFC(s, v2=True)
    d.start()
    for i in range(8):
        d.put_desc(1 + i, desc_v2(0, W, H, host_only=1))
    idx = d.submit(*[(CMD_MAP, 1 + i) for i in range(8)])
    d.status(idx, "host-only maps")
    s.iosfc_w(I_DESC_ENABLE, 0)
    d.v2 = False
    d.desc_pages = {}
    table = s.alloc_page()
    d.table = table
    s.iosfc_w(I_DESC_BASE, s.pa(table))
    root, data = d.surface_pages(1)
    d.put_desc(1, desc_v1(root, W, H))
    idx = d.submit((CMD_MAP, 1), (CMD_TEST, 1))
    d.status(idx, "v1 map and test")
    s.diff("v1 surface: data page", gpa=s.pa(data), length=PAGE)
    s.events("end")
    return s


@scenario(setups=MAPPED)
def mutgap4_sys_iosfc_v1_fourth_component():
    """A v1 plane's fourth component is copied into the v2 form and reaches
    the IOSurface: one-plane 16x8 BGRA v1 surfaces with four components, the
    fourth's bit depth 16 (40 bits in 4 bytes), then 200, then 8 with name
    0xff and range 0xff, then the good plane (8, name 4), each mapped then
    tested; each status and data page is locked. Measured: IOSurface
    validates none of them, and every one maps and tests as the good plane
    (so how the fourth component is copied does not show here)."""
    s = Scenario("mutgap4_sys_iosfc_v1_fourth_component", setups=MAPPED, iosfc=True,
                 doc=mutgap4_sys_iosfc_v1_fourth_component.__doc__)
    d = IOSFC(s, v2=False)
    d.start()
    cases = [("depth 16", dict(depths=(8, 8, 8, 16))), ("depth 200", dict(depths=(8, 8, 8, 200))),
             ("name 0xff, range 0xff", dict(names=(1, 2, 3, 0xFF), ranges=(1, 1, 1, 0xFF))),
             ("good", {})]
    _map_each(s, d, [lambda r, kw=kw: desc_v1(r, W, H, planes=[_good_plane(plane_v1, **kw)]) for _, kw in cases],
              ["fourth component " + c for c, _ in cases])
    s.events("end")
    return s


# ---- the guest display ----

def _display_setup(s, surfaces):
    from scenarios.display_mapper import _setup
    return _setup(s, surfaces)


@scenario(setups=MAPPED, host=True)
def mutgap4_sys_mapper_rgba_bytes_per_element():
    """Opcode 8 checks an 'RGBA' surface's bytes per element against 4: an
    'RGBA' surface at 4 bytes presents; one at 8 bytes throws "invalid bytes
    per pixel (8) != (4)" (no new frame, the present count unchanged, the
    stamp signalled); the first again presents."""
    s = Scenario("mutgap4_sys_mapper_rgba_bytes_per_element", setups=MAPPED, host=True, iosfc=True,
                 displays=[display(0)], doc=mutgap4_sys_mapper_rgba_bytes_per_element.__doc__)
    g, d, pages = _display_setup(s, {1: (b"RGBA", 4, {}), 2: (b"RGBA", 8, {})})
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="RGBA at 4")
    s.wait(cb="newFrame")
    s.events("RGBA at 4")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 2), label="RGBA at 8")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="RGBA at 4 again")
    s.wait(cb="newFrame", cb_count=2)
    s.events("after")
    g.diff_page(0, "page")
    s.api("present_counts", display=0)
    return s


@scenario(setups=MAPPED, host=True)
def mutgap4_sys_mapper_icc_cleared():
    """Clearing the guest ICC profile (opcode 12 with length 0) releases the
    guest display's colour space and forgets it: a valid profile, a mapped
    present (the display retains the space), the profile cleared, a present
    (the display releases its copy and takes none, with mode_change), the
    profile cleared again, a present, a valid profile again and a present,
    and the display released at the end; nothing dies."""
    s = Scenario("mutgap4_sys_mapper_icc_cleared", setups=MAPPED, host=True, iosfc=True,
                 displays=[display(0)], doc=mutgap4_sys_mapper_icc_cleared.__doc__)
    g, d, pages = _display_setup(s, {1: (b"BGRA", 4, {})})
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="first")
    s.wait(cb="newFrame")
    n = 1
    prof = icc_profile(2.2)
    va = g.t0.blob(prof)
    for label, length in (("profile", len(prof)), ("cleared", 0), ("cleared again", 0), ("profile again", len(prof))):
        s.submit(abi.OP_DISPLAY_SET_GUEST_ICC_PROFILE, p_icc(0, length, va, len(prof)), label=label)
        s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="present after " + label)
        n += 1
        s.wait(cb="newFrame", cb_count=n)
        s.wait(cb="modeChange", cb_count=n)
        s.events(label)
    g.diff_page(0, "page")
    s.api("release_display", display=0)
    s.events("released")
    return s
