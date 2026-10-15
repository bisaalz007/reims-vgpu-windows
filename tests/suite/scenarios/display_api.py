"""The display host API (docs/testing.md, sections 4.1 and 4.6):
rvg_display_create and its refusals, one to eight displays on one to eight
ports, what the host reads of a display, the modes, release (hot unplug),
the callback subsets QEMU gives, and encodes before any frame.
"""
from pgtest import abi
from pgtest.display_guest import DisplayGuest, display
from pgtest.registry import scenario
from pgtest.scenario import Scenario


@scenario
def display_new_refusals():
    """rvg_display_create refuses a port >= the port count (1 by default,
    RVG_ERR_PORT_OUT_OF_RANGE), a port already in use (RVG_ERR_PORT_IN_USE),
    and a serial number already reserved anywhere in the process
    (RVG_ERR_SERIAL_IN_USE). A refused display unreserves its serial again, and
    releasing a display frees both its port and its serial."""
    s = Scenario("display_new_refusals", displays=[display(0, serial=5)], doc=display_new_refusals.__doc__)
    for label, spec in (("port 1 of 1", display(1, serial=6)),
                        ("port 0 in use", display(0, serial=7)),
                        ("port 2^32", display(1 << 32, serial=8)),
                        ("serial 5 reserved", display(0, serial=5))):
        s.mark(label)
        s.api("new_display", spec=spec)
    s.api("release_display", display=0)
    s.api("new_display", spec=display(0, serial=7), label="serial 7 after port 0 in use refused it")
    s.api("release_display", display=5)
    s.api("new_display", spec=display(0, serial=5), label="serial 5 after release")
    s.api("display_props", display=6)
    s.api("display_props", display=1, label="a refused display")
    return s


@scenario
def display_ports():
    """One to eight ports (descriptor displayPortCount): a display on each,
    and a shared page on each, which sets bit port in 0x1014; 0x122c reads
    the count at version 245 and count != 0 below it, 0x1234 the connection
    types (0). A display on port = count is refused."""
    out = []
    for n in range(1, 9):
        name = "display_ports_%d" % n
        s = Scenario(name, displays=[display(p) for p in range(n)], descriptor={"port_count": n},
                     doc=display_ports.__doc__)
        g = DisplayGuest(s, task0=False)
        s.r(abi.R_PORT_COUNT)
        s.r(abi.R_PORT_TYPES)
        s.api("new_display", spec=display(n, serial=99), label="port = count")
        for p in range(n):
            g.connect(p, enabled=0xF)
        s.events("connected")
        s.r(abi.R_IRQ_DISPLAY, "all ports' bits")
        s.dump(s.pa(s.root_page, abi.ROOT_DISPLAY_WORD), 4 * 8, "root display words")
        for p in range(n):
            g.diff_page(p, "port %d" % p)
        s.w(abi.R_VERSION, 244)
        s.r(abi.R_VERSION)
        s.r(abi.R_PORT_COUNT, "below 245")
        s.r(abi.R_PORT_TYPES, "below 245")
        out.append(s)
    return out


@scenario
def display_port_count_invalid():
    """A config display_port_count of 0 or 9 is refused: no device;
    RVG_MAX_DISPLAY_PORTS is 8."""
    out = []
    for n in (0, 9):
        s = Scenario("display_port_count_invalid_%d" % n, descriptor={"port_count": n},
                     doc=display_port_count_invalid.__doc__)
        out.append(s)
    s = Scenario("display_port_count_max", doc=display_port_count_invalid.__doc__)
    s.api("maxPorts")
    out.append(s)
    return out


@scenario
def display_properties():
    """What the host reads of a display (its cursor position and present
    counts), and what the page shows of its config: the default name "Apple
    Virtual" (13 bytes), a 14-byte name (zero bytes in the page), a 13-byte
    UTF-8 name, sizes in mm rounded, the serial number. The cursor position
    is {0xffff, 0xffff} with no page, and the page's +0xe00 once
    connected."""
    s = Scenario("display_properties", descriptor={"port_count": 4},
                 displays=[{"port": 0, "serial": 11, "modes": [[64, 32, 60]]},
                           display(1, serial=12, name="fourteen bytes", mm=[123.4, 56.5]),
                           display(2, serial=0xFFFFFFFF, name="été à Pari", mm=[0.4, 0.6]),
                           display(3, serial=0, name="", mm=[65535.4, 1])],
                 doc=display_properties.__doc__)
    g = DisplayGuest(s, task0=False)
    for d in range(4):
        s.api("display_props", display=d)
    for p in range(4):
        g.connect(p)
    for p in range(4):
        g.diff_page(p, "port %d" % p)
    s.u32(g.ss(1, 0xE00), 0x00200010)
    s.api("cursorPosition", display=1)
    s.api("cursorPosition", display=0)
    s.api("present_counts", display=0)
    return s


@scenario
def display_handler_subsets():
    """The callbacks QEMU gives, in subsets: the page's glyph maximum is 256
    only with a cursor_glyph callback, and +0x20 has bit 0 for cursor_show
    and bit 1 for cursor_move. A doorbell calls only the callbacks that are
    given. Port 6 has a page and no display: nothing is called for it."""
    subsets = [[], ["newFrame"], ["cursorGlyph"], ["cursorShow"], ["cursorMove"],
               ["newFrame", "modeChange", "cursorGlyph", "cursorShow", "cursorMove"]]
    specs = [display(p, handlers=h) for p, h in enumerate(subsets)]
    s = Scenario("display_handler_subsets", descriptor={"port_count": 7}, displays=specs,
                 doc=display_handler_subsets.__doc__)
    g = DisplayGuest(s, task0=False)
    for p in range(7):
        g.connect(p)
        s.u32(g.ss(p, 0xE04), 1)
    for p in range(7):
        g.diff_page(p, "port %d" % p)
        s.w(abi.R_DISPLAY_DOORBELL, p)
    s.wait(cb="cursorMove", cb_count=2)
    s.wait(cb="cursorShow", cb_count=2)
    s.events("doorbells")
    for p in range(7):
        g.diff_page(p, "port %d after the doorbell" % p)
    return s


@scenario
def display_encode_refusals():
    """rvg_display_encode_frame with no frame yet clears the target to
    (0,0,0,1) and succeeds, and the host present count stays 0. A region
    smaller than the target does not limit the clear."""
    s = Scenario("display_encode_refusals", displays=[display(0)], doc=display_encode_refusals.__doc__)
    s.encode(0, 8, 4, region=[2, 1, 3, 2], label="a region, no frame")
    s.encode(0, 8, 4, format=70, label="RGBA8Unorm, no frame")
    s.api("present_counts", display=0)
    return s


@scenario
def display_release_with_page():
    """Hot unplug with a page connected, and a fresh display on the same
    port: release detaches the client (8, mode count 0), and the new
    display reconnects (4) from state 0. The serial of the released display
    can be used again."""
    s = Scenario("display_release_with_page", displays=[display(0, serial=40)],
                 doc=display_release_with_page.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0, enabled=0xF)
    g.ack(0, 1)
    s.events("connected")
    s.api("release_display", display=0)
    s.wait(irqs=4)
    s.events("released", order=True)
    g.diff_page(0, "released")
    s.api("display_props", display=0)
    s.api("new_display", spec=display(0, serial=40, name="again", modes=[[32, 16, 60]]))
    s.wait(irqs=5)
    s.events("replugged", order=True)
    g.diff_page(0, "replugged")
    g.ack(0, 2)
    g.disconnect(0)
    s.events("disconnected", order=True)
    g.diff_page(0, "disconnected")
    s.api("release_display", display=1)
    s.quiesce()
    s.events("released with no page")
    return s
