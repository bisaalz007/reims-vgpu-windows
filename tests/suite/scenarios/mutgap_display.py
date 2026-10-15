"""Boundaries, page fields and statuses in the displays, EFI and the
IOSurface host device: each scenario here locks one, as its docstring says.
"""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.display_guest import (SS_EXPECTED_ACK, DisplayGuest, display, floats, gamma_words, p_glyph, p_present3,
                                  p_props, p_sleep, p_swap, pattern)
from pgtest.efi_guest import EFI
from pgtest.iosfc_guest import (CMD_MAP, CMD_TEST, I_DESC_BASE, I_DESC_ENABLE, IOSFC, desc_v1, desc_v2,
                                plane_v1, plane_v2)
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from scenarios.display_cursor import glyph_pixels, glyph_sum
from scenarios.display_mapper import _encode, _setup
from scenarios.suspend import resume_fresh, suspend

MAPPED = ("map", "isolate")
W, H = 16, 8


# ---- the guest display ----

@scenario
def mutgap_disp_settings_overwritten():
    """The page's settings are guest-writable; the settings check puts the
    host's values back and counts each as a change. In state 2, with the
    guest's overwrite of colorSpace.redPrimary.x (+0x2c), greenPrimary.x
    (+0x34) or origin.x (+0x4c, u16) and a mode list identical to the page's,
    the state machine raises 12, bumps expectedAck and sets ackPending: one
    update each, acknowledged before the next."""
    s = Scenario("mutgap_disp_settings_overwritten", displays=[display(0)],
                 doc=mutgap_disp_settings_overwritten.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0, enabled=0xF)
    g.ack(0, 1)
    s.events("connected")
    g.dump_host(0, "before")
    for n, (label, off, data) in enumerate((("redPrimary.x", 0x2C, struct.pack("<f", 0.0)),
                                            ("greenPrimary.x", 0x34, struct.pack("<f", 0.0)),
                                            ("origin.x", 0x4C, struct.pack("<H", 0)))):
        s.mem(g.ss(0, off), data)
        s.api("set_modes", display=0, modes=[[64, 32, 60]])
        s.wait(gpa=g.ss(0, SS_EXPECTED_ACK), eq=n + 2, label=label + " update")
        s.events(label)
        s.dump(g.ss(0, 0x2C), 0x24, label + ": colour space and origin")
        s.dump(g.ss(0, 0xFA8), 0x1C, label + ": ack state")
        g.ack(0, n + 2)
    s.events("end")
    g.diff_page(0, "page")
    return s


@scenario
def mutgap_disp_restore_asleep_after_present():
    """A display presented (opcode 7) and then put to sleep restores asleep:
    the restore takes the sleep branch, not the present one, since the page
    says asleep; the state machine then runs, and a mode list that differs
    from the page's raises 12."""
    s = Scenario("mutgap_disp_restore_asleep_after_present", displays=[display(0)],
                 doc=mutgap_disp_restore_asleep_after_present.__doc__)
    g = DisplayGuest(s)
    r, va = g.backing(16, 8, pixels=pattern(16, 8))
    g.install()
    g.connect(0, enabled=0xF)
    g.ack(0, 1)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, r))
    s.wait(cb="newFrame")
    s.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(0, 1))
    s.events("presented, asleep")
    suspend(s)
    s.api("recreate", config={"op": "config", "displays": [display(0, modes=[[128, 64, 30]])]})
    s.api("willResume")
    s.api("didResume")
    s.wait(gpa=g.ss(0, SS_EXPECTED_ACK), eq=2, label="the restore's update")
    s.events("resumed")
    g.diff_page(0, "resumed")
    s.api("present_counts", display=0)
    return s


@scenario
def mutgap_disp_unknown_then_matrix():
    """SetProperties stops at the first property it does not know: the throw
    is caught by the command, which signals its stamp, so a colour matrix
    after it is never written; the same matrix first in the list is."""
    s = Scenario("mutgap_disp_unknown_then_matrix", displays=[display(0)], doc=mutgap_disp_unknown_then_matrix.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0)
    g.ack(0, 1)
    m = floats(0.5, 0, 0, 0.1, 0, 0.25, 0, 0.2, 0, 0, 2.0, 0.3)
    s.submit(abi.OP_DISPLAY_SET_PROPERTIES, p_props(0, [(1, [1, 2]), (0, m)]), label="unknown, then a matrix")
    s.fault_state("unknown, then a matrix")
    g.diff_page(0, "unknown, then a matrix")
    s.submit(abi.OP_DISPLAY_SET_PROPERTIES, p_props(0, [(0, m), (1, [1, 2])]), label="a matrix, then unknown")
    g.diff_page(0, "a matrix, then unknown")
    s.events("end")
    return s


@scenario
def mutgap_disp_cursor_width_257():
    """A glyph 257 pixels wide is refused ("invalid cursor parameters")
    though every other check passes: stride 1028, mapped length 1028 inside
    task 0, the right checksum. Width 256 with the same bytes is accepted."""
    s = Scenario("mutgap_disp_cursor_width_257", displays=[display(0)], doc=mutgap_disp_cursor_width_257.__doc__)
    g = DisplayGuest(s)
    g.connect(0)
    g.ack(0, 1)
    va = 8 * PAGE
    px = glyph_pixels(257, 1, 1028, seed=5)
    g.t0.write(va, px)
    s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(0, va, 1028, 1028, 257, 1, 0, 0, glyph_sum(257, 1, 1028, px)),
             fault=True, label="257x1")
    s.fault_state("257x1")
    s.resume_root()
    s.submit(abi.OP_NOP)
    s.quiesce(100)
    s.events("257x1")
    g.diff_page(0, "257x1")
    s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(0, va, 1028, 1028, 256, 1, 0, 0, glyph_sum(256, 1, 1028, px)),
             label="256x1")
    s.wait(cb="cursorGlyph")
    s.events("256x1")
    g.diff_page(0, "256x1")
    return s


# ---- EFI ----

@scenario(host=True)
def mutgap_disp_efi_width_max():
    """An EFI mode exactly as wide as the host's largest 2D texture
    (maxTextureWidth2D, 16384 here) is accepted: a 16384 x 1 framebuffer over
    four pages is presented. Trace blocks, so the one present follows the
    rebuild."""
    s = Scenario("mutgap_disp_efi_width_max", host=True, displays=[display(0)], descriptor={"trace": True},
                 doc=mutgap_disp_efi_width_max.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    e = EFI(s, pages=4)
    e.write_frame(16384, 1, 65536, seed=7)
    e.setup(16384, 1, 65536, length=4 * PAGE)
    s.wait(cb="newFrame", label="the 16384 x 1 frame")
    s.events("presented")
    s.api("present_counts", display=0)
    e.regs("registers")
    return s


# ---- mapped presents ----

@scenario(setups=MAPPED, host=True)
def mutgap_disp_mapper_texture_refused_dies():
    """A mapped BGRA surface 16400 pixels wide, over the 16384-texel limit:
    the IOSurface host device maps it (nothing limits a descriptor's width),
    and opcode 8 asks Metal for a texture on it. Metal's own descriptor
    validation aborts the host inside newTextureWithDescriptor:iosurface:
    ("width (16400) greater than the maximum allowed size of 16384"), with
    MTL_DEBUG_LAYER unset, so the call never returns nil and the library's own
    death on a nil texture is never reached; a bytesPerRow not aligned to 16
    and a height of 16400 abort the same way. A guest kills the host with one
    descriptor. Map and isolate only: the mapper needs a memory map."""
    s = Scenario("mutgap_disp_mapper_texture_refused_dies", setups=MAPPED, host=True, iosfc=True,
                 displays=[display(0)], doc=mutgap_disp_mapper_texture_refused_dies.__doc__)
    g, d, pages = _setup(s, {})
    d.start()
    root, data = d.surface_pages(5)
    d.put_desc(1, desc_v2(root, 16400, 1))
    idx = d.submit((CMD_MAP, 1))
    d.status(idx, "map 16400 x 1")
    s.events("mapped")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), fault="either", label="present 16400 x 1")
    s.events("after")
    return s


def half_pixels(w, h, values):
    """RGBA16Float pixels: pixel x of every row is (values[x], values[x],
    values[x], 1.0)."""
    row = b"".join(struct.pack("<4e", v, v, v, 1.0) for v in values[:w])
    return row * h


@scenario(setups=MAPPED, host=True)
def mutgap_disp_gamma_out_of_range():
    """A gamma table applied to an 'RGhA' (RGBA16Float) surface whose
    components run from -0.5 to 1.75: the table is 16 entries a channel,
    linear from 0.1 to 0.4 (red), 0.2 to 0.3 (green) and 0.3 to 0.6 (blue),
    so its slopes at 0 and at 1 are not 1; the encoded row shows what the
    gamma shader does with components below 0 and above 1."""
    s = Scenario("mutgap_disp_gamma_out_of_range", setups=MAPPED, host=True, iosfc=True, displays=[display(0)],
                 doc=mutgap_disp_gamma_out_of_range.__doc__)
    g, d, pages = _setup(s, {1: (b"RGhA", 8, {})})
    vals = [-0.5 + 0.15 * x for x in range(W)]
    s.mem(s.pa(pages[1]), half_pixels(W, H, vals))
    ends = ((0.1, 0.4), (0.2, 0.3), (0.3, 0.6))
    words, csum = gamma_words(16, lambda c, i, n: ends[c][0] + (ends[c][1] - ends[c][0]) * i / (n - 1))
    gva = g.t0.blob(struct.pack("<%dI" % len(words), *words))
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1, gva, 12 * 16, 16, csum), label="gamma")
    s.wait(cb="newFrame")
    s.events("presented")
    s.encode(0, W, H, label="row 0", samples=[[x, 0] for x in range(W)])
    return s


# ---- the IOSurface host device ----

@scenario(setups=MAPPED)
def mutgap_disp_iosfc_seven_planes():
    """Seven planes, the most a descriptor holds, pass the plane-count check
    ("exceeds maximum allowed (7)"), v1 and v2: the map's status is
    IOSurfaceCreate's answer for seven 16 x 8 planes, not NoMemory:
    BadArgument for planes whose element size is 0, and for 1 x 1-element
    planes whose base is not their offset; success for seven 1 x 1-element
    planes each based at its offset."""
    out = []
    for v2 in (False, True):
        name = "mutgap_disp_iosfc_seven_planes_" + ("v2" if v2 else "v1")
        s = Scenario(name, setups=MAPPED, iosfc=True, doc=mutgap_disp_iosfc_seven_planes.__doc__)
        d = IOSFC(s, v2=v2)
        d.start()
        mk, pl = (desc_v2, plane_v2) if v2 else (desc_v1, plane_v1)
        root, data = d.surface_pages(1)
        d.put_desc(1, mk(root, W, H, planes=[pl(W, H, 64, 4, offset=512 * i, size=512) for i in range(7)]))
        d.put_desc(2, mk(root, W, H, planes=[pl(W, H, 64, 4)] * 7))
        d.put_desc(3, mk(root, W, H, alloc=7 * 512,
                         planes=[pl(W, H, 64, 4, offset=512 * i, size=512, ew=1, eh=1) for i in range(7)]))
        d.put_desc(4, mk(root, W, H, alloc=7 * 512,
                         planes=[pl(W, H, 64, 4, offset=512 * i, base=512 * i, size=512, ew=1, eh=1)
                                 for i in range(7)]))
        idx = d.submit((CMD_MAP, 1), (CMD_MAP, 2), (CMD_MAP, 3), (CMD_MAP, 4))
        d.status(idx, "seven planes")
        for mid in (1, 2, 3, 4):
            s.dump(d.desc_gpa(mid), 4, "descriptor %d word 0" % mid)
        s.events("end")
        out.append(s)
    return out


def _switch_to_v1(s, d):
    """Clears 0x1030 (v1 descriptors; the write clears the descriptor
    base) and gives the device a fresh descriptor table."""
    s.iosfc_w(I_DESC_ENABLE, 0)
    d.v2 = False
    d.table = s.alloc_page()
    d.desc_pages = {}
    s.iosfc_w(I_DESC_BASE, s.pa(d.table))


@scenario(setups=MAPPED)
def mutgap_disp_iosfc_v1_planes_upgraded():
    """A v1 descriptor's seven planes are each copied into the v2 form the
    map works on. The device first maps a v2 descriptor whose planes 1-6 have
    five components (NoMemory), leaving those bytes in the heap block the
    next map's descriptor may reuse; then, switched to v1, it maps seven good
    16 x 8 planes of 1 x 1 elements, each based at its offset and with four
    8-bit components (success; a plane left unconverted would carry other
    bytes), one such plane (success), and one plane whose components' bit
    offsets are 1, 3, 5 and 200, which IOSurface accepts too: the offsets are
    only metadata."""
    s = Scenario("mutgap_disp_iosfc_v1_planes_upgraded", setups=MAPPED, iosfc=True,
                 doc=mutgap_disp_iosfc_v1_planes_upgraded.__doc__)
    d = IOSFC(s, v2=True)
    d.start()
    root, data = d.surface_pages(1)
    comp = dict(depths=(8, 8, 8, 8), names=(1, 2, 3, 4), types=(1, 1, 1, 1), ranges=(1, 1, 1, 1))
    poison = [plane_v2(W, H, 64, 4, offset=512 * i, size=512, ew=1, eh=1, comps=4 if i == 0 else 5,
                       offsets=(1, 3, 5, 200), **comp) for i in range(7)]
    d.put_desc(1, desc_v2(root, W, H, alloc=7 * 512, planes=poison))
    idx = d.submit((CMD_MAP, 1))
    d.status(idx, "v2, five components in planes 1-6")
    _switch_to_v1(s, d)

    def good(i, offsets=(0, 8, 16, 24)):
        return plane_v1(W, H, 64, 4, offset=512 * i, base=512 * i, size=512, ew=1, eh=1, comps=4, offsets=offsets,
                        **comp)
    d.put_desc(2, desc_v1(root, W, H, alloc=7 * 512, planes=[good(i) for i in range(7)]))
    d.put_desc(3, desc_v1(root, W, H, planes=[good(0)]))
    d.put_desc(4, desc_v1(root, W, H, planes=[good(0, offsets=(1, 3, 5, 200))]))
    for mid in (2, 3, 4):
        idx = d.submit((CMD_MAP, mid))
        d.status(idx, "v1 descriptor %d" % mid)
    for mid in (2, 3, 4):
        s.dump(d.desc_gpa(mid), 4, "descriptor %d word 0" % mid)
    s.events("end")
    return s


@scenario(setups=MAPPED)
def mutgap_disp_iosfc_test_id_max():
    """Test (3) of mapping ID 0xffffffff is out of range: BadArgument and one
    interrupt, with surface 1 mapped; testing surface 1 then fills it."""
    s = Scenario("mutgap_disp_iosfc_test_id_max", setups=MAPPED, iosfc=True, doc=mutgap_disp_iosfc_test_id_max.__doc__)
    d = IOSFC(s, v2=True)
    d.start()
    root, data = d.surface_pages(1)
    d.put_desc(1, desc_v2(root, W, H))
    idx = d.submit((CMD_MAP, 1), (CMD_TEST, 0xFFFFFFFF), (CMD_TEST, 0x80000000))
    d.status(idx, "map 1, test 0xffffffff, test 0x80000000")
    idx = d.submit((CMD_TEST, 1))
    d.status(idx, "test 1")
    s.diff("surface after test 1", gpa=s.pa(data), length=W * H * 4)
    s.events("end")
    return s


@scenario(setups=MAPPED)
def mutgap_disp_iosfc_test_id_8192_page1():
    """Test of mapping ID 8192 is out of range also while descriptor page 1
    is mapped (by a map of ID 16, the first v2 descriptor on that page):
    BadArgument, one interrupt."""
    s = Scenario("mutgap_disp_iosfc_test_id_8192_page1", setups=MAPPED, iosfc=True,
                 doc=mutgap_disp_iosfc_test_id_8192_page1.__doc__)
    d = IOSFC(s, v2=True)
    d.start()
    root, data = d.surface_pages(1)
    first = d.per_page()
    d.put_desc(first, desc_v2(root, W, H))
    idx = d.submit((CMD_MAP, first), (CMD_TEST, 8192), (CMD_TEST, first))
    d.status(idx, "map 16, test 8192, test 16")
    s.events("end")
    return s



