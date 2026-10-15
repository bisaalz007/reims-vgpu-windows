"""Display coverage gaps (docs/testing.md, section 6.21): the shared-page
protocol's paths the per-area modules leave out, and displays across
hot-plug, reset and suspend.

Scenarios that encode are keyed by the host: the library's shaders draw
the pixels.
"""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.display_guest import (SS_ACK_PENDING, SS_GUEST_CURSOR_SHOW, SS_MODE_COUNT, SS_STATE, DisplayGuest,
                                  display, gamma_words, p_glyph, p_present2, p_present3, p_props, p_show,
                                  p_sleep, p_swap, pattern)
from pgtest.iosfc_guest import CMD_MAP, CMD_UNMAP, IOSFC, desc_v2
from pgtest.registry import scenario
from pgtest.scenario import Scenario

W, H = 16, 8
ALL = 0xF
NO_PRESENT_IRQ = ALL & ~2  # enabledInterrupts without the present bit (1 << 1)


def _glyph(w, h, stride, seed=1):
    """BGRA glyph words, row y at y * stride, and their u32 sum."""
    rows = []
    for y in range(h):
        row = b"".join(struct.pack("<I", ((x * 0x10 + seed) & 0xFF) | ((y * 0x20 & 0xFF) << 8)
                                   | (((x + y + seed) & 0xFF) << 16) | 0x80000000) for x in range(w))
        rows.append(row + b"\xee" * (stride - len(row)))
    data = b"".join(rows)
    total = 0
    for y in range(h):
        for x in range(w):
            total += struct.unpack_from("<I", data, y * stride + 4 * x)[0]
    return data, total & 0xFFFFFFFF


def _enc(s, label, d=0, w=W, h=H, **kw):
    s.encode(d, w, h, label=label, samples=[[0, 0], [w - 1, h - 1], [w // 3, h // 2]], **kw)


# ---- the present interrupt and clients that take no call ----

@scenario
def display_gap_present_irq_masked():
    """Opcodes 6 and 7 with the present bit (1 << 1) not enabled at +0x104:
    the completion still ORs 2 into raisedInterrupts but pokes no port, so
    0x1014 stays 0 and only the stamps interrupt; enabling the bit later
    does not raise the latched 2 either, until the next present."""
    s = Scenario("display_gap_present_irq_masked", displays=[display(0)],
                 doc=display_gap_present_irq_masked.__doc__)
    g = DisplayGuest(s)
    ref, _ = g.backing(W, H, pixels=pattern(W, H, 1))
    g.install()
    g.connect(0, enabled=NO_PRESENT_IRQ)
    g.ack(0, 1)
    s.events("connected")
    s.r(abi.R_IRQ_DISPLAY, "connected")
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, ref), label="7, masked")
    s.wait(cb="newFrame")
    s.submit(abi.OP_DISPLAY_TRANSACTION2, p_present2(0, ref), label="6, masked")
    s.wait(cb="newFrame", cb_count=2)
    s.events("two masked presents")
    s.r(abi.R_IRQ_DISPLAY, "after masked presents")
    g.diff_page(0, "masked")
    s.u32(g.ss(0, 0x104), ALL)
    s.submit(abi.OP_NOP, label="enabled, nothing new")
    s.events("enabled")
    s.r(abi.R_IRQ_DISPLAY, "enabled")
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, ref), label="7, enabled")
    s.wait(cb="newFrame", cb_count=3)
    s.events("enabled present")
    s.r(abi.R_IRQ_DISPLAY, "enabled present")
    g.diff_page(0, "enabled")
    s.api("present_counts", display=0)
    return s


@scenario
def display_gap_clients_without_calls():
    """Every guest command on a port whose client takes no call: ports 0
    and 1 have a page and no display, port 2's display has only a
    mode_change callback. The present runs the
    completion at once (raised |= 2, port poked) and counts in the page;
    the glyph is checked and recorded; show records +0xf20; the doorbell
    does nothing; sleep and wake only record +0xfb8; the colour matrix is
    written. No callback is ever called, and the displays count no guest
    present."""
    s = Scenario("display_gap_clients_without_calls", descriptor={"port_count": 3},
                 displays=[display(2, handlers=["modeChange"])],
                 doc=display_gap_clients_without_calls.__doc__)
    g = DisplayGuest(s)
    ref, _ = g.backing(W, H, pixels=pattern(W, H, 2))
    g.install()
    data, csum = _glyph(4, 2, 16)
    gva = g.t0.blob(data)
    for p in (0, 1, 2):
        g.connect(p, enabled=ALL)
    g.ack(2, 1)
    s.events("connected")
    for p in (0, 1, 2):
        n = "port %d: " % p
        s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(p, ref), label=n + "present 7")
        s.submit(abi.OP_DISPLAY_TRANSACTION2, p_present2(p, ref), label=n + "present 6")
        s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(p, gva, len(data), 16, 4, 2, 1, 1, csum), label=n + "glyph")
        s.submit(abi.OP_DISPLAY_CURSOR_SHOW, p_show(p, 1), label=n + "show")
        s.u32(g.ss(p, SS_GUEST_CURSOR_SHOW), 0)
        s.w(abi.R_DISPLAY_DOORBELL, p)
        s.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(p, 1), label=n + "sleep")
        s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(p, ref), label=n + "present asleep")
        s.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(p, 0), label=n + "wake")
        s.submit(abi.OP_DISPLAY_SET_PROPERTIES, p_props(p, [(0, [0x3F800000] * 12)]), label=n + "matrix")
    s.quiesce(100)
    s.events("commands")
    s.r(abi.R_IRQ_DISPLAY, "commands")
    for p in (0, 1, 2):
        g.diff_page(p, "port %d" % p)
    s.api("present_counts", display=0)
    s.api("display_props", display=0)
    return s


# ---- what calls mode_change ----

@scenario
def display_gap_mode_change_triggers():
    """DisplayCore::present_frame calls mode_change when the width, the height,
    the IOSurface format or the protection differ from the last present's: each
    of width only, height only and format only calls it once, a repeat does
    not. A display without a mode_change callback still tracks the mode and
    shows the frames, with no call."""
    s = Scenario("display_gap_mode_change_triggers", host=True, descriptor={"port_count": 2},
                 displays=[display(0), display(1, handlers=["newFrame"])],
                 doc=display_gap_mode_change_triggers.__doc__)
    g = DisplayGuest(s)
    frames = [(16, 8, b"BGRA"), (16, 12, b"BGRA"), (20, 12, b"BGRA"), (20, 12, b"RGBA"), (20, 12, b"RGBA")]
    refs = []
    for i, (w, h, fmt) in enumerate(frames):
        r, _ = g.backing(w, h, fmt=fmt, pixels=pattern(w, h, 0x10 * i))
        refs.append(r)
    g.install()
    for p in (0, 1):
        g.connect(p, enabled=ALL)
        g.ack(p, 1)
    s.events("connected")
    n = 0
    for (w, h, fmt), r in zip(frames, refs):
        label = "%dx%d %s" % (w, h, fmt.decode())
        n += 1
        for p in (0, 1):
            s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(p, r), label=label + " on %d" % p)
        s.wait(cb="newFrame", cb_count=2 * n)
        s.events(label)
    _enc(s, "display 0, last", 0, 20, 12)
    _enc(s, "display 1, last", 1, 20, 12)
    for d in (0, 1):
        s.api("present_counts", display=d)
    return s


# ---- the state machine ----

@scenario
def display_gap_bogus_resend_without_list():
    """A bogus state (7) after the mode list was consumed: the guest sets
    ackPending again and acks expectedAck, and the machine resends (12)
    from the page's own mode count, with no list to compare, bumping
    expectedAck. A state of 0xffffffff does the same. With a list pending
    the resend also rewrites the modes."""
    s = Scenario("display_gap_bogus_resend_without_list", displays=[display(0, modes=((64, 32, 60), (32, 16, 30)))],
                 doc=display_gap_bogus_resend_without_list.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0, enabled=ALL)
    g.ack(0, 1)
    s.events("connected")
    ack = 1
    for label, state in (("state 7", 7), ("state 0xffffffff", 0xFFFFFFFF)):
        s.u32(g.ss(0, SS_STATE), state)
        s.mem(g.ss(0, SS_ACK_PENDING), b"\x01")
        g.ack(0, ack, label=label)
        s.events(label, order=True)
        g.diff_page(0, label)
        ack += 1
        g.ack(0, ack, label=label + ", acked")
        s.events(label + ", acked", order=True)
    s.u32(g.ss(0, SS_STATE), 3)
    s.api("set_modes", display=0, modes=[[64, 32, 60]], label="state 3 with a new list")
    s.wait(irqs=10)
    s.events("state 3 with a list", order=True)
    g.diff_page(0, "state 3 with a list")
    return s


# ---- gamma tables and the pipeline cache ----

@scenario(host=True)
def display_gap_gamma_sizes():
    """Gamma tables of 2, 3, 1023 and 1024 entries (opcode 7), each re-creating
    the table texture, then a present with no table: the gamma shader stays.
    Encoding into BGRA8Unorm_sRGB (81) with the gamma pipeline cached for
    BGRA8Unorm (80) builds a second pipeline: the cache's key
    (pipeline::PipelineKey) is the target's pixel format and whether the gamma
    table applies, so the two formats share none; RGBA8Unorm (70) and its sRGB
    twin (71) likewise."""
    s = Scenario("display_gap_gamma_sizes", host=True, displays=[display(0)], doc=display_gap_gamma_sizes.__doc__)
    g = DisplayGuest(s, pages=128)
    ref, _ = g.backing(W, H, pixels=pattern(W, H, 5))
    g.install()
    g.connect(0, enabled=ALL)
    g.ack(0, 1)
    n = 0
    for count, curve in ((2, lambda c, i, n: [0.25, 0.75][i]),
                         (3, lambda c, i, n: (i / 2.0) ** (1 + c)),
                         (1023, lambda c, i, n: 1.0 - i / (n - 1)),
                         (1024, lambda c, i, n: (i / (n - 1)) ** 0.5)):
        words, csum = gamma_words(count, curve)
        gva = g.t0.blob(struct.pack("<%dI" % len(words), *words))
        n += 1
        s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, ref, gva, 12 * count, count, csum),
                 label="%d entries" % count)
        s.wait(cb="newFrame", cb_count=n)
        g.diff_page(0, "%d entries" % count)
        _enc(s, "%d entries" % count)
    n += 1
    s.submit(abi.OP_DISPLAY_TRANSACTION2, p_present2(0, ref), label="no table")
    s.wait(cb="newFrame", cb_count=n)
    _enc(s, "no table")
    for fmt in (81, 80, 70, 71, 81):
        _enc(s, "format %d" % fmt, format=fmt)
    s.events("end")
    return s


# ---- hot plug and unplug while presenting ----

@scenario
def display_gap_unplug_while_presenting():
    """Releasing a presenting display detaches its client: the guest display
    raises 8 (disconnect) and zeroes the mode count; the guest's commands then
    reach a guest display with no client (the present completes at once, the
    glyph and show are only recorded, the doorbell and sleep do nothing a host
    sees). A display plugged into the port again becomes the client and
    connects (4) from state 0; after the ack a present reaches it."""
    s = Scenario("display_gap_unplug_while_presenting", displays=[display(0)],
                 doc=display_gap_unplug_while_presenting.__doc__)
    g = DisplayGuest(s)
    ref, _ = g.backing(W, H, pixels=pattern(W, H, 7))
    g.install()
    data, csum = _glyph(2, 2, 8)
    gva = g.t0.blob(data)
    g.connect(0, enabled=ALL)
    g.ack(0, 1)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, ref), label="presented")
    s.wait(cb="newFrame")
    s.events("presented")
    s.api("release_display", display=0)
    s.wait(irqs=8)
    s.events("unplugged", order=True)
    g.diff_page(0, "unplugged")
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, ref), label="present, no client")
    s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(0, gva, len(data), 8, 2, 2, 0, 1, csum), label="glyph, no client")
    s.submit(abi.OP_DISPLAY_CURSOR_SHOW, p_show(0, 1), label="show, no client")
    s.w(abi.R_DISPLAY_DOORBELL, 0)
    s.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(0, 1), label="sleep, no client")
    s.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(0, 0), label="wake, no client")
    s.quiesce(100)
    s.events("no client")
    g.diff_page(0, "no client")
    s.api("new_display", spec=display(0, serial=2000, modes=((32, 16, 60),)))
    s.wait(irqs=15)
    s.events("plugged again", order=True)
    g.diff_page(0, "plugged again")
    g.ack(0, 2)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, ref), label="present to the new display")
    s.wait(cb="newFrame", cb_count=2)
    s.events("new display presented")
    s.api("present_counts", display=1)
    s.api("display_props", display=1)
    return s


# ---- displays around reset ----

@scenario
def display_gap_reset_with_displays():
    """A reset (DeviceCore::reset) with three ports, two with pages (port 0
    presenting, port 1 with its connect ack pending): the root page write of 0
    detaches both guest displays. Port 0's disconnects (8, one interrupt, mode
    count 0); port 1's stays in state 1 with the ack pending ("bogus ack"),
    raising nothing. The root words are cleared, and 0x1014 and the version
    read 0. Brought up again on the same pages, port 0 reconnects from state 0
    (4, expectedAck 2) while port 1's page, still waiting for its ack, is left
    as it was; after the acks a present reaches the same display, which counts
    it after the one before the reset."""
    s = Scenario("display_gap_reset_with_displays", descriptor={"port_count": 3},
                 displays=[display(0), display(1), display(2)], doc=display_gap_reset_with_displays.__doc__)
    g = DisplayGuest(s)
    ref, _ = g.backing(W, H, pixels=pattern(W, H, 9))
    g.install()
    g.connect(0, enabled=ALL)
    g.ack(0, 1)
    g.connect(1, enabled=ALL)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, ref), label="presented")
    s.wait(cb="newFrame")
    s.events("before the reset")
    s.api("reset")
    s.quiesce(100)
    s.events("reset")
    for p in (0, 1):
        g.diff_page(p, "port %d after the reset" % p)
    s.dump(s.pa(s.root_page, 0x900), 16, label="root words 0x240..")
    s.r(abi.R_IRQ_DISPLAY, "after the reset")
    s.r(abi.R_VERSION, "after the reset")
    for d in range(3):
        s.api("present_counts", display=d)
    # Brought up again on the same pages and task 0.
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    s.fifo.enable()
    s.fifo.written = 0
    g.t0.define()
    g.install()
    for p in (0, 1):
        g.connect(p)
    s.events("reconnected", order=True)
    for p in (0, 1):
        g.diff_page(p, "port %d reconnected" % p)
    g.ack(0, 2)
    g.ack(1, 1)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, ref), label="present after the reset")
    s.wait(cb="newFrame", cb_count=2)
    s.events("presented again")
    for p in (0, 1):
        g.diff_page(p, "port %d acked" % p)
    s.api("present_counts", display=0)
    return s


# ---- displays around suspend ----

@scenario
def display_gap_suspend_restore_variants():
    """Displays that restore without re-presenting: port 0 connected and
    acked, never presented, comes back on a display with no mode list, so
    the machine runs on the restored page's modes alone (no change, nothing
    raised); port 1 with an ack pending stays pending ("bogus ack"); port 2
    with a glyph and show re-sends both to the new display; port 3's
    display had no modes, so its page restores with a mode count of 0 and
    connects from the new display's list (4); port 4, acked, whose guest
    zeroed the mode count, is not "restored" (the page has no modes) but
    takes the new display's list as an update (12, expectedAck 2, state 1),
    not a disconnect; port 5 presented and then went to sleep, so the
    restore only puts it to sleep again (no frame, no call)."""
    specs = [display(0), display(1), display(2), display(3, modes=None), display(4), display(5)]
    s = Scenario("display_gap_suspend_restore_variants", descriptor={"port_count": 6}, displays=specs,
                 doc=display_gap_suspend_restore_variants.__doc__)
    g = DisplayGuest(s)
    ref, _ = g.backing(W, H, pixels=pattern(W, H, 4))
    g.install()
    data, csum = _glyph(3, 2, 12)
    gva = g.t0.blob(data)
    for p in range(6):
        g.connect(p, enabled=ALL)
    for p in (0, 2, 4, 5):
        g.ack(p, 1)
    s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(2, gva, len(data), 12, 3, 2, 2, 1, csum))
    s.submit(abi.OP_DISPLAY_CURSOR_SHOW, p_show(2, 1))
    s.wait(cb="cursorShow")
    s.mem(g.ss(4, SS_MODE_COUNT), struct.pack("<H", 0))
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(5, ref))
    s.wait(cb="newFrame")
    s.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(5, 1))
    s.events("before")
    for p in range(6):
        g.diff_page(p, "port %d before" % p)
    s.api("willSuspend")
    s.api("finishSuspend")
    new = [display(0, modes=None), display(1), display(2), display(3, modes=((48, 24, 60),)), display(4),
           display(5)]
    s.api("recreate", config={"op": "config", "descriptor": {"port_count": 6}, "displays": new})
    s.api("willResume")
    s.api("didResume")
    s.wait(cb="cursorShow", cb_count=2)
    s.events("resumed")
    for p in range(6):
        g.diff_page(p, "port %d resumed" % p)
    g.ack(1, 1, label="port 1's pending ack")
    s.events("port 1 acked", order=True)
    s.api("present_counts", display=5)
    return s


@scenario
def display_gap_restore_without_task0():
    """A display that presented, whose guest then deleted task 0 (the resume
    re-creates no task 0): the restore re-presents through
    GuestDisplay::present_mapped_surface, whose DeviceCore::get_kernel_task
    raises (the task lookup raises and never answers none, so the guest
    display's own check for no kernel task cannot fire);
    GuestDisplay::finish_restore catches it and skips the state machine call
    after it. Nothing is called; the page keeps its state."""
    s = Scenario("display_gap_restore_without_task0", displays=[display(0)],
                 doc=display_gap_restore_without_task0.__doc__)
    g = DisplayGuest(s)
    ref, _ = g.backing(W, H, pixels=pattern(W, H, 6))
    g.install()
    g.connect(0, enabled=ALL)
    g.ack(0, 1)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, ref))
    s.wait(cb="newFrame")
    g.t0.delete()
    s.events("presented, task 0 deleted")
    g.diff_page(0, "before")
    s.api("willSuspend")
    s.api("finishSuspend")
    s.api("recreate", config={"op": "config", "displays": [display(0, modes=((32, 16, 60),))]})
    s.api("willResume")
    s.api("didResume")
    s.quiesce(100)
    s.events("resumed")
    g.diff_page(0, "resumed")
    s.api("present_counts", display=1)
    return s


@scenario
def display_gap_resume_nil_with_displays():
    """rvg_device_will_resume with no state (NULL) on a fresh device with
    displays: refused like empty data (RVG_RESUME_ERROR_INVALID_CONTENT, 2),
    and the displays stay parked (no guest display:
    rvg_display_cursor_position reads 0xffff, 0xffff), with nothing called."""
    s = Scenario("display_gap_resume_nil_with_displays", descriptor={"port_count": 2},
                 displays=[display(0), display(1)], doc=display_gap_resume_nil_with_displays.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0, enabled=ALL)
    g.ack(0, 1)
    s.api("willSuspend")
    s.api("finishSuspend")
    s.api("recreate", config={"op": "config", "descriptor": {"port_count": 2}, "displays": [display(0), display(1)]})
    s.api("willResume", nil=True, label="nil")
    s.api("willResume", slot="never filled", label="empty data")
    s.quiesce(100)
    s.events("after")
    for d in (0, 1):
        s.api("display_props", display=d)
    return s


# ---- mapped presents (opcode 8) ----

MAPPED = ("map", "isolate")


@scenario(setups=MAPPED)
def display_gap_mapped_without_calls():
    """Opcode 8 with the present bit masked on port 0 (raised |= 2, no
    poke), to ports without a display (1 and 2) and to a display without
    a new_frame callback (3): the
    completion runs at once and the page counts the present and
    records the mapping ID, with no call. Map and isolate: the
    IOSurface host device needs a memory map."""
    s = Scenario("display_gap_mapped_without_calls", setups=MAPPED, iosfc=True, descriptor={"port_count": 4},
                 displays=[display(0), display(3, handlers=["modeChange"])],
                 doc=display_gap_mapped_without_calls.__doc__)
    g = DisplayGuest(s)
    g.connect(0, enabled=NO_PRESENT_IRQ)
    g.ack(0, 1)
    g.connect(1, enabled=ALL)
    g.connect(2, enabled=ALL)
    g.connect(3, enabled=ALL)
    g.ack(3, 1)
    d = IOSFC(s, v2=True)
    d.start()
    root, data = d.surface_pages(1)
    s.mem(s.pa(data), pattern(W, H, 3))
    d.put_desc(1, desc_v2(root, W, H))
    d.submit((CMD_MAP, 1))
    s.events("mapped")
    s.r(abi.R_IRQ_DISPLAY, "mapped")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="port 0, masked")
    s.wait(cb="newFrame")
    for p in (1, 2, 3):
        s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(p, 1), label="port %d" % p)
    s.quiesce(100)
    s.events("presents")
    s.r(abi.R_IRQ_DISPLAY, "presents")
    for p in range(4):
        g.diff_page(p, "port %d" % p)
    idx = d.submit((CMD_UNMAP, 1))
    d.status(idx, "unmap: only port 0's display holds it")
    return s


# ---- cursor glyphs and positions at the edges ----

def _glyph_rows(w, h, stride, seed):
    """A glyph whose rows overlap or skip as the stride says: the bytes the
    host reads are (h - 1) x stride + 4w, each row 4w bytes at y x stride.
    Returns (bytes, sum) over what the host sums."""
    n = (h - 1) * stride + 4 * w
    data = bytes(((i * 7 + seed) & 0x7F) | 0x80 for i in range(n))
    total = 0
    for y in range(h):
        for x in range(w):
            total += struct.unpack_from("<I", data, y * stride + 4 * x)[0]
    return data, total & 0xFFFFFFFF


@scenario
def display_gap_cursor_edges():
    """Glyph parameters at their edges, each passing with its checksum: a
    stride smaller than 4w (rows overlap) and a stride of 0 (every row is
    the first), a mapped length exactly (h - 1) x stride + 4w, 256 x 1 and
    1 x 256, and a hotspot outside the glyph (passed on as it is). One
    byte less of mapped length faults. The cursor position at its extremes
    reads back as {x, y}."""
    s = Scenario("display_gap_cursor_edges", displays=[display(0)], doc=display_gap_cursor_edges.__doc__)
    g = DisplayGuest(s, pages=80)
    g.connect(0, enabled=ALL)
    g.ack(0, 1)
    n = 0
    for label, (w, h, stride, hx, hy) in (("stride 4 < 4w", (4, 3, 4, 0, 0)),
                                          ("stride 0", (3, 4, 0, 1, 1)),
                                          ("256 x 1", (256, 1, 1024, 255, 0)),
                                          ("1 x 256", (1, 256, 4, 0, 255)),
                                          ("hotspot outside", (2, 2, 8, 300, 0xFFFF))):
        data, csum = _glyph_rows(w, h, stride, n)
        va = g.t0.blob(data)
        n += 1
        s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(0, va, len(data), stride, w, h, hx, hy, csum), label=label)
        s.wait(cb="cursorGlyph", cb_count=n)
        s.events(label)
        s.dump(g.ss(0, 0xF00), 0x2C, label=label)
    data, csum = _glyph_rows(3, 2, 20, 9)
    va = g.t0.blob(data)
    s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(0, va, len(data) - 1, 20, 3, 2, 0, 0, csum), fault=True,
             label="one byte short")
    s.fault_state("one byte short")
    s.resume_root()
    s.submit(abi.OP_NOP)
    for label, pos in (("0, 0", 0), ("0xffff, 0", 0xFFFF), ("0, 0xffff", 0xFFFF0000), ("0xffff, 0xffff", 0xFFFFFFFF)):
        s.u32(g.ss(0, 0xE00), pos)
        s.w(abi.R_DISPLAY_DOORBELL, 0)
        s.api("cursorPosition", display=0, label=label)
    s.wait(cb="cursorMove", cb_count=4)
    s.events("positions")
    return s


# ---- a display released before suspend ----

@scenario
def display_gap_release_before_suspend():
    """A display unplugged from a connected port, then a suspend: the page
    stays with the device (its guest display, mode count 0, state 0 after the
    disconnect), and on the migration target the port's new display connects
    from the restored page (4); the other port, presenting, restores through
    the mapped-surface path ("Invalid mappingID", caught) and keeps its
    state."""
    s = Scenario("display_gap_release_before_suspend", descriptor={"port_count": 2},
                 displays=[display(0), display(1)], doc=display_gap_release_before_suspend.__doc__)
    g = DisplayGuest(s)
    ref, _ = g.backing(W, H, pixels=pattern(W, H, 8))
    g.install()
    for p in (0, 1):
        g.connect(p, enabled=ALL)
        g.ack(p, 1)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, ref))
    s.wait(cb="newFrame")
    s.api("release_display", display=1)
    s.wait(irqs=11)
    s.events("released")
    for p in (0, 1):
        g.diff_page(p, "port %d before" % p)
    s.api("willSuspend")
    s.api("finishSuspend")
    s.api("recreate", config={"op": "config", "descriptor": {"port_count": 2},
                              "displays": [display(0), display(1, serial=2001, modes=((48, 24, 60),))]})
    s.api("willResume")
    s.api("didResume")
    s.wait(irqs=13)
    s.events("resumed")
    for p in (0, 1):
        g.diff_page(p, "port %d resumed" % p)
    g.ack(1, 2)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(1, ref), label="present on the new port 1")
    s.wait(cb="newFrame", cb_count=2)
    s.events("presented on port 1")
    return s
