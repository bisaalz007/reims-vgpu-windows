"""The cursor (docs/testing.md, sections 4.6 and 6.21): the glyph (opcode
4), show (5), the doorbell (0x1220) and rvg_display_cursor_position.
"""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.display_guest import SS_CURSOR_POS, SS_GUEST_CURSOR_SHOW, DisplayGuest, display, p_glyph, p_show
from pgtest.registry import scenario
from pgtest.scenario import Scenario


def glyph_pixels(w, h, stride, seed=1):
    """BGRA words, row y at y * stride; padding bytes are 0xee (they are
    never summed or copied)."""
    rows = []
    for y in range(h):
        row = b"".join(struct.pack("<I", ((x * 0x10 + seed) & 0xFF) | ((y * 0x20 & 0xFF) << 8)
                                   | (((x + y + seed) & 0xFF) << 16) | 0x80000000) for x in range(w))
        rows.append(row + b"\xee" * (stride - len(row)))
    return b"".join(rows)


def glyph_sum(w, h, stride, data):
    t = 0
    for y in range(h):
        for x in range(w):
            t += struct.unpack_from("<I", data, y * stride + 4 * x)[0]
    return t & 0xFFFFFFFF


@scenario
def display_cursor_glyph():
    """A glyph with its checksum: cursor_glyph gets a w x h RGBA8 bitmap,
    swizzled from the guest's BGRA, and the hotspot; the page records the glyph
    at +0xf00. A checksum of 0 skips the check; a stride larger than 4w skips
    the padding. The glyph's bytes after the call are the host's copy: a later
    guest write does not change it."""
    s = Scenario("display_cursor_glyph", displays=[display(0)], doc=display_cursor_glyph.__doc__)
    g = DisplayGuest(s)
    g.connect(0)
    g.ack(0, 1)
    va = 8 * PAGE
    for label, (w, h, stride, hx, hy, use_sum) in (
            ("4x3 packed", (4, 3, 16, 1, 2, True)),
            ("5x2 stride 32, checksum 0", (5, 2, 32, 4, 1, False)),
            ("1x1", (1, 1, 4, 0, 0, True))):
        data = glyph_pixels(w, h, stride)
        g.t0.write(va, data)
        csum = glyph_sum(w, h, stride, data) if use_sum else 0
        s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(0, va, len(data), stride, w, h, hx, hy, csum), label=label)
        s.wait(cb="cursorGlyph", cb_count={"4x3 packed": 1, "5x2 stride 32, checksum 0": 2, "1x1": 3}[label])
        s.events(label)
        g.diff_page(0, label)
    return s


@scenario
def display_cursor_glyph_sizes():
    """Width and height 1-256 pass; 0 and 257 fault ("invalid cursor
    parameters"); so do (h - 1) x stride + 4w overflowing 64 bits, or exceeding
    mappedLength, a task other than 0, a port without a page, and a checksum
    mismatch, which skips cursor_glyph. A mapped length past the largest local
    task faults before any allocation, and one past the task's end faults in
    the copy. The 256 x 256 glyph is hashed."""
    s = Scenario("display_cursor_glyph_sizes", displays=[display(0)], doc=display_cursor_glyph_sizes.__doc__)
    g = DisplayGuest(s, pages=72)
    g.connect(0)
    g.ack(0, 1)
    va = 8 * PAGE
    big = glyph_pixels(256, 256, 1024, seed=3)
    g.t0.write(va, big)
    s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(0, va, len(big), 1024, 256, 256, 255, 255,
                                                  glyph_sum(256, 256, 1024, big)), label="256x256")
    s.wait(cb="cursorGlyph")
    s.events("256x256")
    g.diff_page(0, "256x256")
    bad = (("width 0", p_glyph(0, va, 64, 4, 0, 1)),
           ("width 257", p_glyph(0, va, 0x100000, 1028, 257, 1)),
           ("height 0", p_glyph(0, va, 64, 4, 1, 0)),
           ("height 257", p_glyph(0, va, 0x100000, 4, 1, 257)),
           ("stride overflow", p_glyph(0, va, 0x100, 1 << 63, 1, 3)),
           ("add overflow", p_glyph(0, va, 0x100, (1 << 64) - 4, 2, 2)),
           ("needed > mapped", p_glyph(0, va, 4 * 3 * 2 - 1, 12, 3, 2)),
           ("task 1", p_glyph(0, va, 64, 16, 4, 1, task=1)),
           ("port 1", p_glyph(1, va, 64, 16, 4, 1)),
           ("checksum mismatch", p_glyph(0, va, 16, 16, 4, 1, csum=1)),
           ("mapped length too large to allocate", p_glyph(0, va, 1 << 62, 16, 4, 1)),
           ("past the task's end", p_glyph(0, 71 * PAGE, 2 * PAGE, 16, 4, 1)),
           ("short", struct.pack("<II", 0, 0)))
    for label, payload in bad:
        s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, payload, fault=True, label=label)
        s.fault_state(label)
        s.resume_root()
        s.submit(abi.OP_NOP)
    s.quiesce(100)
    s.events("refusals")
    g.diff_page(0, "after the refusals")
    return s


@scenario
def display_cursor_no_glyph_handler():
    """A display without a cursor_glyph callback: the page's glyph maximum is
    0, the glyph is still checked and recorded in the page, and no call is
    made; its checksum is not verified, because nothing copies it. A port with
    no display behaves alike."""
    s = Scenario("display_cursor_no_glyph_handler", descriptor={"port_count": 2},
                 displays=[display(0, handlers=["newFrame", "cursorShow"])],
                 doc=display_cursor_no_glyph_handler.__doc__)
    g = DisplayGuest(s)
    g.connect(0)
    g.connect(1)
    va = 8 * PAGE
    data = glyph_pixels(4, 4, 16)
    g.t0.write(va, data)
    for p in (0, 1):
        s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(p, va, len(data), 16, 4, 4, 1, 1, 0x1234), label="port %d" % p)
        s.submit(abi.OP_DISPLAY_CURSOR_SHOW, p_show(p, 1), label="show on port %d" % p)
    s.wait(cb="cursorShow")
    s.events("glyphs")
    for p in (0, 1):
        g.diff_page(p, "port %d" % p)
    return s


@scenario
def display_cursor_show_doorbell():
    """Show (5) calls cursor_show and records +0xf20; the doorbell (0x1220 =
    port) calls cursor_show only when the guest's +0xe04 differs from +0xf20,
    then cursor_move every time, in that order, on the display thread.
    rvg_display_cursor_position reads +0xe00 {x, y}. A doorbell for a port
    without a page, or out of range, is ignored."""
    s = Scenario("display_cursor_show_doorbell", descriptor={"port_count": 2}, displays=[display(0)],
                 doc=display_cursor_show_doorbell.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0)
    g.ack(0, 1)
    s.api("cursorPosition", display=0)
    s.submit(abi.OP_DISPLAY_CURSOR_SHOW, p_show(0, 1))
    s.wait(cb="cursorShow")
    s.events("show 1")
    g.diff_page(0, "show 1")
    s.u32(g.ss(0, SS_GUEST_CURSOR_SHOW), 0)
    s.u32(g.ss(0, SS_CURSOR_POS), 123 | 456 << 16)
    s.w(abi.R_DISPLAY_DOORBELL, 0)
    s.wait(cb="cursorMove")
    s.events("doorbell, guest hides", order=True)
    s.api("cursorPosition", display=0)
    g.diff_page(0, "after the doorbell")
    s.w(abi.R_DISPLAY_DOORBELL, 0)
    s.wait(cb="cursorMove", cb_count=2)
    s.events("doorbell, same show", order=True)
    s.submit(abi.OP_DISPLAY_CURSOR_SHOW, p_show(0, 7), label="show 7")
    s.wait(cb="cursorShow", cb_count=3)
    s.events("show 7")
    g.diff_page(0, "show 7")
    s.u32(g.ss(0, SS_CURSOR_POS), 0xFFFFFFFF)
    s.api("cursorPosition", display=0)
    for port in (1, 2, 0x10000):
        s.w(abi.R_DISPLAY_DOORBELL, port)
    s.quiesce(100)
    s.events("doorbells elsewhere")
    for label, payload in (("show on port 1", p_show(1, 1)), ("short", struct.pack("<I", 0))):
        s.submit(abi.OP_DISPLAY_CURSOR_SHOW, payload, fault=True, label=label)
        s.fault_state(label)
        s.resume_root()
        s.submit(abi.OP_NOP)
    s.events("end")
    return s
