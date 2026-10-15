"""IOSurface host device coverage gaps (docs/testing.md, section 6.23): a
device made without a memory map, ring registers written in orders the
per-area module leaves out, element dimensions in their power-of-two form,
and malformed rings of commands; and mapped surfaces shared by displays.

The device has no save state: the C API gives it none, and a suspend
carries nothing of it.
"""
from pgtest import abi
from pgtest.abi import PAGE
from pgtest.display_guest import DisplayGuest, display, p_swap, pattern
from pgtest.iosfc_guest import (CMD_MAP, CMD_TEST, CMD_UNMAP, I_READ, I_RING_BASE, I_RING_SIZE, I_WRITE, IOSFC,
                                IREGS, desc_v1, desc_v2, plane_v1, plane_v2)
from pgtest.registry import scenario
from pgtest.scenario import Scenario

MAPPED = ("map", "isolate")
W, H = 16, 8


def _regs(s, label):
    for r in IREGS:
        s.iosfc_r(r, label)


@scenario
def iosfc_gap_no_memory_map():
    """An IOSurface host device config without a memory map:
    rvg_iosurface_device_create refuses it (RVG_ERR_NO_MEMORY_MAP). The
    RvgDevice, made with use_iosurface_mapper, then has no mapper service,
    and every opcode 8 is "Invalid mappingID" with its stamp signalled.
    Refused before the RvgDevice is created and after it alike. Every setup:
    the IOSurface host device does not depend on the RvgDevice's."""
    out = []
    for after in (False, True):
        name = "iosfc_gap_no_memory_map" + ("_after" if after else "")
        s = Scenario(name, iosfc=True, iosfc_after=after, displays=[display(0)], doc=iosfc_gap_no_memory_map.__doc__)
        s._config["iosfc_no_map"] = True
        g = DisplayGuest(s)
        g.connect(0, enabled=0xF)
        g.ack(0, 1)
        s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="mapping 1")
        s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 0), label="mapping 0")
        s.fault_state("after")
        s.events("presents")
        g.diff_page(0, "page")
        out.append(s)
    return out


@scenario(setups=MAPPED)
def iosfc_gap_ring_register_orders():
    """The ring's registers in the orders the guest may write them: a
    doorbell before any ring runs nothing and leaves the read counter; a
    ring base with no size, or a size of 0 written after it, maps
    nothing, and a doorbell then runs nothing either; so does a ring
    without a descriptor table. The read counter is guest-writable only
    while no ring is mapped; once the size maps the ring, commands run
    from it to the write counter, and a later write to it is ignored."""
    s = Scenario("iosfc_gap_ring_register_orders", setups=MAPPED, iosfc=True,
                 doc=iosfc_gap_ring_register_orders.__doc__)
    d = IOSFC(s)
    root, data = d.surface_pages(1)
    d.put_desc(1, desc_v1(root, W, H))
    s.iosfc_w(I_WRITE, 1)
    s.quiesce()
    s.events("doorbell with no ring")
    _regs(s, "no ring")
    s.iosfc_w(I_RING_BASE, s.pa(d.ring))
    s.iosfc_w(I_WRITE, 1)
    s.quiesce()
    s.events("base, no size")
    s.iosfc_w(I_RING_SIZE, 0)
    s.iosfc_w(I_WRITE, 1)
    s.quiesce()
    s.events("base, size 0")
    _regs(s, "base, size 0")
    s.iosfc_w(I_READ, 2)
    s.iosfc_w(I_RING_SIZE, d.entries)
    s.iosfc_w(I_WRITE, 2)
    s.quiesce()
    s.events("ring, no descriptor table")
    _regs(s, "ring, no descriptor table")
    s.iosfc_w(I_READ, 0)
    s.iosfc_r(I_READ, "read counter written with the ring mapped")
    d.start(enable=True, ring=False)
    # Commands at entries 2 and 3: the read counter is 2.
    d.written = 2
    idx = d.submit((CMD_MAP, 1), (CMD_TEST, 1))
    d.status(idx, "from the read counter")
    _regs(s, "ran")
    s.diff("surface", gpa=s.pa(data), length=0x40)
    return s


@scenario(setups=MAPPED)
def iosfc_gap_element_dimensions():
    """Element widths and heights with bit 7 set are powers of two,
    1 << (value & 0x7f), in the descriptor and in its planes, v1 and v2:
    0x80 is 1, 0x81 is 2, and each such surface maps and tests; 0x9f is
    1 << 31, which IOSurfaceCreate refuses (BadArgument). Each map's status
    and each test's writes into guest memory are locked."""
    out = []
    for v2 in (False, True):
        s = Scenario("iosfc_gap_element_dimensions_" + ("v2" if v2 else "v1"), setups=MAPPED, iosfc=True,
                     doc=iosfc_gap_element_dimensions.__doc__)
        d = IOSFC(s, v2=v2)
        d.start()
        mk, pl = (desc_v2, plane_v2) if v2 else (desc_v1, plane_v1)
        cases = [("0x80 x 0x80", dict(ew=0x80, eh=0x80)),
                 ("0x81 x 1, 8 bytes an element", dict(ew=0x81, eh=1, bpe=8, bpr=64)),
                 ("1 x 0x81", dict(ew=1, eh=0x81, bpe=4, bpr=64, alloc=64 * H)),
                 ("0x81 x 0x81", dict(ew=0x81, eh=0x81, bpe=16, bpr=128, alloc=128 * H // 2)),
                 ("plane 0x81 x 0x80", dict(planes=[pl(W, H, 64, 8, size=512, ew=0x81, eh=0x80)])),
                 ("0x9f x 1", dict(ew=0x9F, eh=1))]
        pages = []
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


@scenario(setups=MAPPED)
def iosfc_gap_mixed_ring():
    """One doorbell over a ring of mixed commands, run in order with one
    interrupt each and every status written back: unsupported commands (0,
    4, 0xffffffff) with any mapping ID, a map, the same map again (success,
    nothing re-created), a test and an unmap of an ID never mapped, an
    unmap, a test after it, and IDs 0 and 8191."""
    s = Scenario("iosfc_gap_mixed_ring", setups=MAPPED, iosfc=True, doc=iosfc_gap_mixed_ring.__doc__)
    d = IOSFC(s, v2=True)
    d.start()
    root, data = d.surface_pages(1)
    d.put_desc(1, desc_v2(root, W, H))
    cmds = [(0, 1), (4, 1), (0xFFFFFFFF, 0), (CMD_MAP, 1), (CMD_MAP, 1), (CMD_TEST, 2), (CMD_UNMAP, 2),
            (CMD_TEST, 1), (CMD_UNMAP, 1), (CMD_TEST, 1), (CMD_MAP, 0), (CMD_MAP, 8191), (CMD_UNMAP, 8191),
            (CMD_TEST, 8191)]
    idx = d.submit(*cmds)
    d.status(idx, "statuses")
    s.dump(d.desc_gpa(1), 8, label="descriptor 1's first word")
    s.events("mixed ring")
    _regs(s, "end")
    return s


@scenario(setups=MAPPED, host=True)
def iosfc_gap_surface_on_two_displays():
    """One mapped surface presented on two displays: each display's frame
    holds it, so an unmap is Busy until both have presented another
    surface; then it unmaps, and presenting its ID is "Invalid
    mappingID". The second surface's frames are the IOSurface over guest
    pages: a guest write shows in both displays' encodes."""
    s = Scenario("iosfc_gap_surface_on_two_displays", setups=MAPPED, host=True, iosfc=True,
                 descriptor={"port_count": 2}, displays=[display(0), display(1)],
                 doc=iosfc_gap_surface_on_two_displays.__doc__)
    g = DisplayGuest(s)
    for p in (0, 1):
        g.connect(p, enabled=0xF)
        g.ack(p, 1)
    d = IOSFC(s, v2=True)
    d.start()
    pages = {}
    for mid in (1, 2):
        root, data = d.surface_pages(1)
        s.mem(s.pa(data), pattern(W, H, 0x20 * mid))
        pages[mid] = data
        d.put_desc(mid, desc_v2(root, W, H))
    d.submit((CMD_MAP, 1), (CMD_MAP, 2))
    for p in (0, 1):
        s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(p, 1), label="surface 1 on port %d" % p)
    s.wait(cb="newFrame", cb_count=2)
    d.status(d.submit((CMD_UNMAP, 1)), "unmap 1, on both")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 2), label="surface 2 on port 0")
    s.wait(cb="newFrame", cb_count=3)
    d.status(d.submit((CMD_UNMAP, 1)), "unmap 1, on port 1")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(1, 2), label="surface 2 on port 1")
    s.wait(cb="newFrame", cb_count=4)
    d.status(d.submit((CMD_UNMAP, 1)), "unmap 1, on neither")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="surface 1 unmapped")
    s.mem(s.pa(pages[2]), pattern(W, H, 0x77))
    for p in (0, 1):
        s.encode(p, W, H, label="port %d after a guest write" % p, samples=[[0, 0], [W - 1, H - 1]])
    s.quiesce(100)
    s.events("end")
    for p in (0, 1):
        g.diff_page(p, "port %d" % p)
    return s
