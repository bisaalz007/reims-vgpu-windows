"""Guest presents (docs/testing.md, section 6.21): opcodes 6 and 7 present an
IOSurface backing (object type 4) from task 0; the frame reaches the host's
encode (DisplayCore::encode_current_frame). Gamma tables (7), the colour
matrix, sleep (9), and the present counts.

Pixels are drawn by the library's shaders, so these scenarios are keyed by
the host.
"""
import struct

from pgtest import abi
from pgtest.display_guest import (IDENTITY, DisplayGuest, SS_CM_FORCE, SS_CM_READ, SS_CM_WRITE, display,
                                  floats, gamma_words, p_present2, p_present3, p_props, p_sleep, pattern)
from pgtest.registry import scenario
from pgtest.scenario import Scenario

W, H = 16, 8


def _guest(s, enabled=0xF, frames=((b"BGRA", 0),), version=abi.MAX_VERSION):
    """A connected, acked display on port 0 and task 0 with one backing per
    (format, seed) in frames. Returns (guest, [refs], [vas])."""
    g = DisplayGuest(s, version=version)
    refs, vas = [], []
    for fmt, seed in frames:
        r, va = g.backing(W, H, fmt=fmt, pixels=pattern(W, H, seed))
        refs.append(r)
        vas.append(va)
    g.install()
    g.connect(0, enabled=enabled)
    g.ack(0, 1)
    return g, refs, vas


def _encode(s, label, **kw):
    s.encode(0, W, H, label=label, samples=[[0, 0], [W - 1, H - 1], [5, 3]], **kw)


@scenario(host=True)
def display_present_basic():
    """Opcode 7 presents a BGRA backing: mode_change({16, 8}, 'BGRA'), a new
    frame, raisedInterrupts |= 2 with its interrupt before the stamp's,
    present count +1 at +0xf98 and the surface at +0xfa0. The encode reads
    it back and counts one host present; a second encode of the same frame
    does not. Presenting it again does not re-copy from guest memory (the
    backing stays host-valid)."""
    s = Scenario("display_present_basic", host=True, displays=[display(0)], doc=display_present_basic.__doc__)
    g, refs, vas = _guest(s)
    s.events("connected")
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, refs[0]))
    s.wait(cb="newFrame")
    s.events("presented")
    s.r(abi.R_IRQ_DISPLAY)
    g.diff_page(0, "page after the present")
    s.api("present_counts", display=0)
    _encode(s, "the frame")
    _encode(s, "the frame again")
    s.api("present_counts", display=0)
    g.t0.write(vas[0], pattern(W, H, 0x40))
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, refs[0]), label="again, after a guest write")
    s.wait(cb="newFrame", cb_count=2)
    s.events("presented again")
    _encode(s, "the frame after the guest write")
    s.api("present_counts", display=0)
    g.diff_page(0, "page after the second present")
    return s


@scenario(host=True)
def display_present_formats():
    """'BGRA' and 'RGBA' backings present (the view takes the IOSurface's
    format); any other IOSurface format throws "invalid pixel format", and
    opcode 7 still signals its stamp while opcode 6 faults. Switching
    format calls mode_change with the new OSType."""
    s = Scenario("display_present_formats", host=True, displays=[display(0)], doc=display_present_formats.__doc__)
    g, refs, vas = _guest(s, frames=((b"BGRA", 0), (b"RGBA", 3), (b"2vuy", 5), (0x12345678, 7)))
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, refs[1]), label="RGBA")
    s.wait(cb="newFrame")
    s.events("RGBA")
    _encode(s, "RGBA")
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, refs[0]), label="BGRA")
    s.wait(cb="modeChange", cb_count=2)
    s.wait(cb="newFrame", cb_count=2)
    s.events("BGRA")
    _encode(s, "BGRA")
    for label, ref in (("2vuy", refs[2]), ("0x12345678", refs[3])):
        s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, ref), label=label + " (7)")
        s.fault_state(label + " (7)")
        s.submit(abi.OP_DISPLAY_TRANSACTION2, p_present2(0, ref), fault=True, label=label + " (6)")
        s.fault_state(label + " (6)")
        s.resume_root()
        s.submit(abi.OP_NOP)
    s.events("bad formats")
    g.diff_page(0, "page")
    s.api("present_counts", display=0)
    return s


@scenario(host=True)
def display_present_opcode6():
    """Opcode 6 (DisplayTransaction2) presents like 7 without a gamma table;
    a taskID other than 0 faults ("only reference a kernel task"), and so do
    a port without a page, a reference that is not a backing, and a short
    payload."""
    s = Scenario("display_present_opcode6", host=True, displays=[display(0)], doc=display_present_opcode6.__doc__)
    g, refs, vas = _guest(s)
    s.submit(abi.OP_DISPLAY_TRANSACTION2, p_present2(0, refs[0]))
    s.wait(cb="newFrame")
    s.events("presented")
    _encode(s, "the frame")
    for label, payload in (("task 1", p_present2(0, refs[0], task=1)), ("port 3", p_present2(3, refs[0])),
                           ("reference 0", p_present2(0, 0)), ("reference 40", p_present2(0, 40)),
                           ("short", struct.pack("<II", 0, refs[0]))):
        s.submit(abi.OP_DISPLAY_TRANSACTION2, payload, fault=True, label=label)
        s.fault_state(label)
        s.resume_root()
        s.submit(abi.OP_NOP)
    s.events("refusals")
    g.diff_page(0, "page")
    return s


@scenario(host=True)
def display_present3_errors():
    """Opcode 7's refusals, each logged and the stamp signalled anyway: task
    1, no page on the port, a reference with no backing, a short payload. No
    task 0 at all: "Invalid task"."""
    s = Scenario("display_present3_errors", host=True, displays=[display(0)], doc=display_present3_errors.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0)
    g.ack(0, 1)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, 1), label="no task 0")
    for label, payload in (("task 1", p_present3(0, 1, task=1)), ("port 3", p_present3(3, 1)),
                           ("short", struct.pack("<III", 0, 0, 1))):
        s.submit(abi.OP_DISPLAY_TRANSACTION3, payload, label=label)
    s.fault_state("after")
    s.events("refusals")
    g.diff_page(0, "page")
    s.api("present_counts", display=0)
    return s


@scenario(host=True)
def display_present_no_newframe():
    """A display without a new_frame callback (or a port with no display): the
    guest display skips DisplayCore::present_frame and runs the completion at
    once (raisedInterrupts |= 2), and still counts the present in the page. No
    mode_change, no frame for the encode."""
    s = Scenario("display_present_no_newframe", host=True, descriptor={"port_count": 2},
                 displays=[display(0, handlers=["modeChange", "cursorMove"])],
                 doc=display_present_no_newframe.__doc__)
    g, refs, vas = _guest(s)
    g.connect(1, enabled=0xF)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, refs[0]), label="port 0")
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(1, refs[0]), label="port 1, no display")
    s.events("presented")
    s.r(abi.R_IRQ_DISPLAY)
    g.diff_page(0, "port 0")
    g.diff_page(1, "port 1")
    _encode(s, "no frame")
    s.api("present_counts", display=0)
    return s


@scenario(host=True)
def display_present_gamma():
    """A gamma table with opcode 7: entryCount <= 1024 and
    4 x count <= mappedLength, else a throw before the frame; a checksum
    mismatch raises in DisplayCore::update_gamma_table_locked after
    mode_change, so the present is not counted. A good table is stored at
    +0xf80 and switches the display to the gamma shader for good; a table of
    another size re-creates the table texture. The mapped length is always
    at least 12 x count: the host reads 12 x count bytes."""
    s = Scenario("display_present_gamma", host=True, displays=[display(0)], doc=display_present_gamma.__doc__)
    g, refs, vas = _guest(s)
    n = 256
    words, csum = gamma_words(n, lambda c, i, n: (i / (n - 1)) ** (1.0 + c * 0.5))
    gva = g.t0.blob(struct.pack("<%dI" % len(words), *words))
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, refs[0]), label="no gamma")
    s.wait(cb="newFrame")
    _encode(s, "no gamma")
    for label, args in (("1025 entries", (gva, 0x4000, 1025, 0)),
                        ("mapped < 4 count", (gva, 4 * n - 4, n, csum)),
                        ("bad checksum", (gva, 12 * n, n, csum ^ 1))):
        s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, refs[0], *args), label=label)
        g.diff_page(0, label)
    s.events("refusals")
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, refs[0], gva, 12 * n, n, csum), label="gamma 256")
    s.wait(cb="newFrame", cb_count=2)
    s.events("gamma 256")
    g.diff_page(0, "gamma 256")
    _encode(s, "gamma 256")
    words2, csum2 = gamma_words(16, lambda c, i, n: 1.0 - i / (n - 1))
    gva2 = g.t0.blob(struct.pack("<%dI" % len(words2), *words2))
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, refs[0], gva2, 12 * 16, 16, csum2), label="gamma 16")
    s.wait(cb="newFrame", cb_count=3)
    _encode(s, "gamma 16, inverted")
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, refs[0]), label="no table: the gamma shader stays")
    s.wait(cb="newFrame", cb_count=4)
    _encode(s, "no table after one")
    s.api("present_counts", display=0)
    s.events("end")
    return s


@scenario
def display_present_gamma_one():
    """A gamma table of one entry kills the host: the slope step is 1/0, and
    the red slope at one reads gammaTable[entryCount - 2] with a u32 index,
    0xffffffff, 16 GiB past the table; the host dies after the present's
    doorbell. docs/testing.md (6.21) says it reads index -1; it reads
    2^32 - 1. The death is on the display's queue, not the FIFO's, so the
    present's stamp may land before it: the scenario ends in a wait only the
    death ends (0x102c never reads 2), where it always lands."""
    s = Scenario("display_present_gamma_one", displays=[display(0)], doc=display_present_gamma_one.__doc__)
    g, refs, vas = _guest(s)
    words, csum = gamma_words(1, lambda c, i, n: 0.5)
    gva = g.t0.blob(struct.pack("<3I", *words))
    s.fifo.cmd(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, refs[0], gva, 12, 1, csum))
    s.fifo.doorbell()
    s.wait(mmio=abi.R_FAULT_MASK, mmio_eq=2, timeout_ms=10000, label="the one-entry gamma present: the host dies")
    return s


@scenario(host=True)
def display_present_color_matrix():
    """The colour matrix reaches the gamma shader: property 0 (opcode 10)
    marks the matrix dirty and the next present swaps the read and write
    indices and loads it; a present with nothing dirty keeps it; the guest's
    force-read byte reloads the read index."""
    s = Scenario("display_present_color_matrix", host=True, displays=[display(0)],
                 doc=display_present_color_matrix.__doc__)
    g, refs, vas = _guest(s)
    words, csum = gamma_words(256)
    gva = g.t0.blob(struct.pack("<768I", *words))
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, refs[0], gva, 12 * 256, 256, csum), label="linear gamma")
    s.wait(cb="newFrame")
    _encode(s, "identity")
    g.diff_page(0, "after the first present")
    swap = floats(0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0)
    s.submit(abi.OP_DISPLAY_SET_PROPERTIES, p_props(0, [(0, swap)]))
    g.diff_page(0, "matrix written")
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, refs[0]), label="dirty")
    s.wait(cb="newFrame", cb_count=2)
    g.diff_page(0, "after the dirty present")
    _encode(s, "red and blue swapped")
    half = floats(0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0)
    s.submit(abi.OP_DISPLAY_SET_PROPERTIES, p_props(0, [(0, half)]))
    s.mem(g.ss(0, 0xA73), b"\x00")
    s.mem(g.ss(0, SS_CM_FORCE), b"\x01")
    s.mem(g.ss(0, SS_CM_READ), b"\x00")
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, refs[0]), label="force read of index 0")
    s.wait(cb="newFrame", cb_count=3)
    g.diff_page(0, "after the forced read")
    _encode(s, "forced read")
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, refs[0]), label="clean")
    s.wait(cb="newFrame", cb_count=4)
    g.diff_page(0, "after the clean present")
    _encode(s, "clean")
    return s


@scenario(host=True)
def display_encode_region():
    """The encode's region is the viewport of the frame's quad; the clear
    still covers the whole texture. Encoding into an RGBA8 texture builds
    another pipeline."""
    s = Scenario("display_encode_region", host=True, displays=[display(0)], doc=display_encode_region.__doc__)
    g, refs, vas = _guest(s)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, refs[0]))
    s.wait(cb="newFrame")
    for label, kw in (("left half", dict(region=[0, 0, W // 2, H])),
                      ("offset quarter", dict(region=[4, 2, W // 2, H // 2])),
                      ("RGBA8 target", dict(format=70)),
                      ("sRGB target", dict(format=81)),
                      ("region past the texture", dict(region=[8, 4, W, H]))):
        _encode(s, label, **kw)
    s.encode(0, 32, 16, label="upscaled 2x", samples=[[0, 0], [31, 15], [10, 6]])
    s.api("present_counts", display=0)
    return s


@scenario
def display_sleep():
    """Sleep (9): asleep, a present updates the frame but neither calls
    new_frame nor counts a guest present (the page still counts it); waking
    signals the current frame (new_frame, +1). The page's +0xfb8 follows.
    Below version 21 the command faults; a short payload faults."""
    s = Scenario("display_sleep", displays=[display(0)], doc=display_sleep.__doc__)
    g, refs, vas = _guest(s)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, refs[0]))
    s.wait(cb="newFrame")
    s.events("awake present")
    s.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(0, 1))
    g.diff_page(0, "asleep")
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, refs[0]), label="asleep present")
    s.quiesce(100)
    s.events("asleep present")
    s.api("present_counts", display=0)
    g.diff_page(0, "after the asleep present")
    s.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(0, 0))
    s.wait(cb="newFrame", cb_count=2)
    s.events("woken")
    s.api("present_counts", display=0)
    g.diff_page(0, "awake")
    s.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(0, 5), label="asleep = 5")
    g.diff_page(0, "asleep 5")
    for label, payload in (("short", struct.pack("<I", 0)), ("port 2", p_sleep(2, 1))):
        s.submit(abi.OP_DISPLAY_SLEEP_STATE, payload, fault=True, label=label)
        s.fault_state(label)
        s.resume_root()
        s.submit(abi.OP_NOP)
    s.w(abi.R_VERSION, 20)
    s.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(0, 0), fault=True, label="version 20")
    s.fault_state("version 20")
    s.events("end")
    return s


@scenario
def display_sleep_no_frame():
    """Waking with no frame presented yet: DisplayCore::signal_current_frame
    returns early, nothing is called. A display without new_frame gets no wake
    call either."""
    s = Scenario("display_sleep_no_frame", descriptor={"port_count": 2},
                 displays=[display(0), display(1, handlers=["cursorMove"])], doc=display_sleep_no_frame.__doc__)
    g = DisplayGuest(s, task0=False)
    for p in (0, 1):
        g.connect(p)
        g.ack(p, 1)
        s.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(p, 1))
        s.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(p, 0))
    s.quiesce(100)
    s.events("end")
    for p in (0, 1):
        g.diff_page(p, "port %d" % p)
    return s
