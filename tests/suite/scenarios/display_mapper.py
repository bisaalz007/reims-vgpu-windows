"""Presents of IOSurface mapper surfaces (opcode 8, DisplaySwapMapping;
docs/testing.md, section 6.21): the surface is mapped by the IOSurface host
device over guest pages, and presented by mapping ID.

Map and isolate setups only: the IOSurface host device exists only with a
memory map (apple-gfx-mmio.c). Pixels are drawn by the library's shaders, so
the scenarios are keyed by host.
"""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.display_guest import DisplayGuest, display, gamma_words, icc_profile, p_icc, p_swap
from pgtest.iosfc_guest import BUSY, CMD_MAP, CMD_UNMAP, I_DESC_BASE, IOSFC, desc_v2
from pgtest.registry import scenario
from pgtest.scenario import Scenario

SETUPS = ("map", "isolate")
W, H = 16, 8


def surface_pixels(w, h, bpe=4, seed=0):
    if bpe == 8:
        # RGBA16Float: 0.5 in every channel but alpha 1.0.
        return struct.pack("<4H", 0x3800, 0x3400, 0x3000, 0x3C00) * (w * h)
    return b"".join(bytes([(x * 8 + seed) & 0xFF, (y * 16) & 0xFF, (x + y + seed) & 0xFF, 0xFF])
                    for y in range(h) for x in range(w))


def _setup(s, surfaces, version=abi.MAX_VERSION, iosfc=True):
    """Task 0, a connected display on port 0, and the IOSurface host device
    with v2 descriptors; surfaces: {id: (fmt, bpe, kwargs)}. Maps each and
    returns (guest, device, {id: data page})."""
    g = DisplayGuest(s, version=version)
    g.connect(0, enabled=0xF)
    g.ack(0, 1)
    d = IOSFC(s, v2=True)
    pages = {}
    if iosfc:
        d.start()
        for mid, (fmt, bpe, kw) in surfaces.items():
            root, data = d.surface_pages(1)
            s.mem(s.pa(data), surface_pixels(W, H, bpe, seed=mid))
            pages[mid] = data
            d.put_desc(mid, desc_v2(root, W, H, fmt=fmt, bpe=bpe, **kw))
        if surfaces:
            d.submit(*[(CMD_MAP, m) for m in surfaces])
    return g, d, pages


def _encode(s, label, **kw):
    s.encode(0, W, H, label=label, samples=[[0, 0], [W - 1, H - 1], [5, 3]], **kw)


@scenario(setups=SETUPS, host=True)
def display_mapper_present():
    """Opcode 8 presents a mapped BGRA surface: mode_change, a new frame, the
    present interrupt, +0xf98 counts it and +0xfb0 holds the mapping ID. The
    frame is the IOSurface over guest pages: a guest write shows in the next
    encode. While the display holds it, unmapping the surface is Busy."""
    s = Scenario("display_mapper_present", setups=SETUPS, host=True, iosfc=True, displays=[display(0)],
                 doc=display_mapper_present.__doc__)
    g, d, pages = _setup(s, {1: (b"BGRA", 4, {}), 2: (b"RGBA", 4, {})})
    s.events("mapped")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1))
    s.wait(cb="newFrame")
    s.events("presented")
    g.diff_page(0, "page")
    _encode(s, "surface 1")
    s.mem(s.pa(pages[1]), surface_pixels(W, H, seed=0x40))
    _encode(s, "after a guest write")
    idx = d.submit((CMD_UNMAP, 1))
    d.status(idx, "unmap while presented")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 2), label="RGBA")
    s.wait(cb="newFrame", cb_count=2)
    s.events("RGBA")
    _encode(s, "surface 2")
    idx = d.submit((CMD_UNMAP, 1))
    d.status(idx, "unmap after the display let go")
    s.api("present_counts", display=0)
    g.diff_page(0, "page at the end")
    return s


@scenario(setups=SETUPS, host=True)
def display_mapper_formats():
    """Opcode 8's formats: 'BGRA' and 'RGBA' at 4 bytes a pixel, 'RGhA' at 8
    (RGBA16Float; below version 22 it logs once and presents anyway);
    another format throws "invalid pixel format", and a bytes-per-element
    mismatch throws; each failure still signals the stamp."""
    out = []
    for version in (abi.MAX_VERSION, 21):
        name = "display_mapper_formats" + ("" if version == abi.MAX_VERSION else "_v21")
        s = Scenario(name, setups=SETUPS, host=True, iosfc=True, displays=[display(0)],
                     doc=display_mapper_formats.__doc__)
        g, d, pages = _setup(s, {1: (b"RGhA", 8, {}), 2: (b"ARGB", 4, {}), 3: (b"RGhA", 4, {}),
                                 4: (b"BGRA", 8, {})}, version=version)
        s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="RGhA")
        s.wait(cb="newFrame")
        s.events("RGhA")
        _encode(s, "RGhA")
        for label, mid in (("ARGB", 2), ("RGhA at 4 bytes", 3), ("BGRA at 8 bytes", 4)):
            s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, mid), label=label)
        s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="RGhA again")
        s.wait(cb="newFrame", cb_count=2)
        s.events("refusals")
        g.diff_page(0, "page")
        s.api("present_counts", display=0)
        out.append(s)
    return out


@scenario(setups=SETUPS, host=True)
def display_mapper_errors():
    """Opcode 8's refusals, each signalling the stamp: an unmapped ID, ID 0
    and 8192 ("Invalid mappingID"), a taskID other than 0, a port without a
    page, a short payload; below version 10 the command is unsupported.
    Without task 0: "Invalid task"."""
    s = Scenario("display_mapper_errors", setups=SETUPS, host=True, iosfc=True, displays=[display(0)],
                 doc=display_mapper_errors.__doc__)
    g, d, pages = _setup(s, {1: (b"BGRA", 4, {})})
    for label, payload in (("unmapped id", p_swap(0, 5)), ("id 0", p_swap(0, 0)), ("id 8192", p_swap(0, 8192)),
                           ("id 2^40", p_swap(0, 1 << 40)), ("task 1", p_swap(0, 1, task=1)),
                           ("port 1", p_swap(1, 1)), ("short", struct.pack("<II", 0, 0))):
        s.submit(abi.OP_DISPLAY_SWAP_MAPPING, payload, label=label)
    s.fault_state("after")
    s.w(abi.R_VERSION, 6)
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="version 6")
    s.quiesce(100)
    s.events("refusals")
    g.diff_page(0, "page")
    s.api("present_counts", display=0)
    return s


@scenario(setups=SETUPS, host=True)
def display_mapper_no_task0():
    """Opcode 8 with no task 0 raises from DeviceCore::get_kernel_task before
    the mapper is asked; the guest display's own check for no kernel task is
    dead. The stamp is signalled."""
    s = Scenario("display_mapper_no_task0", setups=SETUPS, host=True, iosfc=True, displays=[display(0)],
                 doc=display_mapper_no_task0.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0)
    g.ack(0, 1)
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1))
    s.events("end")
    g.diff_page(0, "page")
    return s


@scenario(setups=SETUPS, host=True)
def display_mapper_no_mapper():
    """A device without the IOSurface mapper (use_iosurface_mapper false, or
    no IOSurface host device in the process): the device has no mapper
    service, and every opcode 8 is "Invalid mappingID"."""
    s = Scenario("display_mapper_no_mapper", setups=SETUPS, host=True, displays=[display(0)],
                 doc=display_mapper_no_mapper.__doc__)
    g = DisplayGuest(s)
    g.connect(0)
    g.ack(0, 1)
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1))
    s.events("end")
    g.diff_page(0, "page")
    return s


@scenario(setups=SETUPS, host=True)
def display_mapper_gamma_icc():
    """Opcode 8 with a gamma table (stored at +0xf80, switching to the gamma
    shader), and a guest ICC profile: the profile marks the guest colour
    space dirty, and the next mapped present passes it on, which calls
    mode_change though nothing else changed. A profile the host cannot parse
    gives a nil colour space, still dirty."""
    s = Scenario("display_mapper_gamma_icc", setups=SETUPS, host=True, iosfc=True, displays=[display(0)],
                 doc=display_mapper_gamma_icc.__doc__)
    g, d, pages = _setup(s, {1: (b"BGRA", 4, {})})
    words, csum = gamma_words(64, lambda c, i, n: (i / (n - 1)) ** 2)
    gva = g.t0.blob(struct.pack("<%dI" % len(words), *words))
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1, gva, 12 * 64, 64, csum), label="gamma")
    s.wait(cb="newFrame")
    s.events("gamma")
    g.diff_page(0, "gamma")
    _encode(s, "gamma")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1, gva, 12 * 64, 1025, csum), label="1025 entries")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1, gva, 12 * 64, 64, csum + 1), label="bad checksum")
    va = g.t0.blob(b"\x00" * 128)
    s.submit(abi.OP_DISPLAY_SET_GUEST_ICC_PROFILE, p_icc(0, 128, va, 128))
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="after the ICC profile")
    s.wait(cb="newFrame", cb_count=2)
    s.wait(cb="modeChange", cb_count=2)
    s.events("ICC")
    g.diff_page(0, "ICC")
    _encode(s, "after the ICC profile")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="clean")
    s.wait(cb="newFrame", cb_count=3)
    s.events("clean")
    return s


@scenario(setups=SETUPS, host=True)
def display_mapper_icc_profiles():
    """Valid guest ICC profiles (pgtest.display_guest.icc_profile, gamma 2.2
    then 1.8): each gives the guest display a colour space, releasing the one
    before, and the next mapped present swaps the display's retained copy and
    calls mode_change. Releasing the display then frees the copy it holds."""
    s = Scenario("display_mapper_icc_profiles", setups=SETUPS, host=True, iosfc=True, displays=[display(0)],
                 doc=display_mapper_icc_profiles.__doc__)
    g, d, pages = _setup(s, {1: (b"BGRA", 4, {})})
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="first")
    s.wait(cb="newFrame")
    n = 1
    for gamma in (2.2, 1.8):
        prof = icc_profile(gamma)
        va = g.t0.blob(prof)
        s.submit(abi.OP_DISPLAY_SET_GUEST_ICC_PROFILE, p_icc(0, len(prof), va, len(prof)), label="gamma %s" % gamma)
        s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="present after gamma %s" % gamma)
        n += 1
        s.wait(cb="newFrame", cb_count=n)
        s.wait(cb="modeChange", cb_count=n)
        s.events("gamma %s" % gamma)
        _encode(s, "gamma %s" % gamma)
    g.diff_page(0, "page")
    s.api("release_display", display=0)
    s.events("released")
    return s


@scenario(setups=SETUPS, host=True)
def display_mapper_teardown_while_presented():
    """A descriptor base write while the display holds the presented surface:
    HostDeviceCore::teardown_surfaces releases the device's reference anyway,
    logging the outstanding references, the display keeps showing its frame,
    the mapping ID presents "Invalid mappingID" until mapped again, and then
    presents the new surface."""
    s = Scenario("display_mapper_teardown_while_presented", setups=SETUPS, host=True, iosfc=True,
                 displays=[display(0)], doc=display_mapper_teardown_while_presented.__doc__)
    g, d, pages = _setup(s, {1: (b"BGRA", 4, {})})
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="presented")
    s.wait(cb="newFrame")
    s.iosfc_w(I_DESC_BASE, s.pa(d.table))
    _encode(s, "after the teardown")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="torn down")
    s.quiesce(100)
    s.events("torn down")
    s.mem(s.pa(pages[1]), surface_pixels(W, H, seed=0x40))
    idx = d.submit((CMD_MAP, 1))
    d.status(idx, "mapped again")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="again")
    s.wait(cb="newFrame", cb_count=2)
    s.events("again")
    _encode(s, "mapped again")
    g.diff_page(0, "page")
    return s


@scenario(setups=SETUPS, host=True)
def display_mapper_protection():
    """A v2 descriptor's protection options reach the surface, and opcode 8
    passes them to DisplayCore::present_frame, which calls mode_change for the
    change; the encode then sets them on the host's command buffer. A surface
    IOSurface refuses to protect fails to map instead: its status is locked
    either way."""
    s = Scenario("display_mapper_protection", setups=SETUPS, host=True, iosfc=True, displays=[display(0)],
                 doc=display_mapper_protection.__doc__)
    g, d, pages = _setup(s, {1: (b"BGRA", 4, {}), 2: (b"BGRA", 4, {"protection": 1})})
    d.status([0, 1], "maps")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="plain")
    s.wait(cb="newFrame")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 2), label="protected")
    s.quiesce(100)
    s.events("presents")
    g.diff_page(0, "page")
    s.api("present_counts", display=0)
    return s
