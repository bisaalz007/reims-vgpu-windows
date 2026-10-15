"""The display's shared state page and its state machine (docs/testing.md,
section 6.21): opcode 1 set, replace and clear, the ack protocol (2), the
guest's interrupt enables and 0x1014, mode list changes after connect,
disconnect, the guest's mode count, colour matrices and display properties
(10), compositor parameters (11), and the ICC profile (12).
"""
import struct

from pgtest import abi
from pgtest.display_guest import (DisplayGuest, IDENTITY, SS_CM_DIRTY, SS_CM_FORCE, SS_CM_READ, SS_CM_WRITE,
                                  SS_ACK_PENDING, SS_ENABLED, SS_LIVE_RESIZE, SS_MODE_COUNT, SS_STATE, display, floats,
                                  p_ack, p_compositor, p_icc, p_props, p_set_shared)
from pgtest.registry import scenario
from pgtest.scenario import Scenario

ALL = 0xF  # every raisedInterrupts bit the host sets: 2, 4, 8


def _irq_regs(s, label):
    """0x1014 (display bits, read-to-clear) then 0x1018; read after an
    events checkpoint."""
    s.r(abi.R_IRQ_DISPLAY, label)
    s.r(abi.R_IRQ_STAMP, label)


@scenario
def display_connect_ack():
    """Opcode 1 fills the page (serial, name, port, sizes, modes, the
    identity colour matrix, expectedAck 1, ackPending, state 1) and raises
    interrupt 4, then the stamp's; the ack (2) of expectedAck moves the state
    to 2 with no interrupt."""
    s = Scenario("display_connect_ack", displays=[display(0, modes=((64, 32, 60), (1920, 1080, 59.94)))],
                 doc=display_connect_ack.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0, enabled=ALL)
    s.events("connected", order=True)
    _irq_regs(s, "connected")
    g.diff_page(0, "page after connect")
    s.r(abi.R_IRQ_DISPLAY, "0x1014 read again")
    g.ack(0, 1)
    s.events("acked", order=True)
    _irq_regs(s, "acked")
    g.diff_page(0, "page after ack")
    s.diff("elsewhere")
    s.api("display_props", display=0)
    return s


@scenario
def display_ack_errors():
    """An ack with nothing pending faults ("unexpected ack"), and so does one
    that is not expectedAck ("invalid ack"); neither changes the page. An ack
    on a port with no page, or out of range, faults in
    DeviceCore::get_guest_display_at_port."""
    s = Scenario("display_ack_errors", displays=[display(0)], doc=display_ack_errors.__doc__)
    g = DisplayGuest(s, task0=False)
    g.ack(0, 1, fault=True, label="ack before any page")
    s.fault_state("no page")
    s.resume_root()
    s.submit(abi.OP_NOP)
    g.ack(9, 1, fault=True, label="port out of range")
    s.resume_root()
    s.submit(abi.OP_NOP)
    g.connect(0, enabled=ALL)
    g.ack(0, 2, fault=True, label="invalid ack")
    s.fault_state("invalid ack")
    s.resume_root()
    s.submit(abi.OP_NOP)
    g.diff_page(0, "page after the invalid ack")
    g.ack(0, 1)
    g.ack(0, 1, fault=True, label="unexpected ack")
    s.fault_state("unexpected ack")
    s.resume_root()
    s.submit(abi.OP_NOP)
    s.events("end of acks", order=True)
    g.diff_page(0, "page after the unexpected ack")
    return s


@scenario
def display_interrupt_enables():
    """raisedInterrupts is set whatever the guest enabled; 0x1014 and the
    interrupt follow only the enabled bits. Two ports: port 1's bit is
    1 << 1."""
    out = []
    for name, en in (("none", 0), ("connect_only", 4), ("disconnect_only", 8)):
        n = "display_interrupt_enables_" + name
        s = Scenario(n, displays=[display(0), display(1)], descriptor={"port_count": 2},
                     doc=display_interrupt_enables.__doc__)
        g = DisplayGuest(s, task0=False)
        g.connect(1, enabled=en)
        s.events("port 1 connected")
        _irq_regs(s, "port 1 connected")
        g.ack(1, 1)
        g.disconnect(1)
        s.events("port 1 disconnected")
        _irq_regs(s, "port 1 disconnected")
        g.diff_page(1, "port 1's page")
        out.append(s)
    return out


@scenario
def display_set_page_errors():
    """Opcode 1's refusals, each a fault: port >= the port count, a page on a
    port in use, a clear of a port not in use, and a page with no root page. A
    short payload faults too (Command::check_size). A page outside guest RAM
    fails to map (GuestDisplay::new raises) in the map setups."""
    s = Scenario("display_set_page_errors", displays=[display(0)], doc=display_set_page_errors.__doc__)
    g = DisplayGuest(s, task0=False)
    p = g.page(0)
    for label, payload in (("port out of range", p_set_shared(1, s.pfn(p))),
                           ("clear a port not in use", p_set_shared(0, 0)),
                           ("short payload", struct.pack("<I", 0))):
        s.submit(abi.OP_DISPLAY_SET_SHARED_STATE_PAGE, payload, fault=True, label=label)
        s.fault_state(label)
        s.resume_root()
        s.submit(abi.OP_NOP)
    g.connect(0)
    q = s.alloc_page()
    s.submit(abi.OP_DISPLAY_SET_SHARED_STATE_PAGE, p_set_shared(0, s.pfn(q)), fault=True, label="port in use")
    s.fault_state("port in use")
    s.resume_root()
    s.submit(abi.OP_NOP)
    s.events("refusals")
    s.diff("after the refusals")
    return s


@scenario
def display_set_page_unmapped():
    """A shared page outside guest RAM: it cannot be mapped (the memory map
    refuses it; in the blocks setup map_memory answers NO), GuestDisplay::new
    raises and the command faults."""
    s = Scenario("display_set_page_unmapped", displays=[display(0)],
                 doc=display_set_page_unmapped.__doc__)
    DisplayGuest(s, task0=False)
    s.submit(abi.OP_DISPLAY_SET_SHARED_STATE_PAGE, p_set_shared(0, 0x10), fault=True)
    s.fault_state("unmapped page")
    s.events("end")
    return s


@scenario
def display_page_replace():
    """Clearing a port's page (opcode 1, page 0) detaches the guest display:
    interrupt 8 is raised into the old page, the root word 0x900 + 4 port goes
    to 0, and a new page connects again from state 0 with expectedAck 1."""
    s = Scenario("display_page_replace", displays=[display(0)], doc=display_page_replace.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0, enabled=ALL)
    s.dump(s.pa(s.root_page, abi.ROOT_DISPLAY_WORD), 8, "root display words")
    g.ack(0, 1)
    g.disconnect(0)
    s.events("disconnected", order=True)
    _irq_regs(s, "disconnected")
    s.dump(s.pa(s.root_page, abi.ROOT_DISPLAY_WORD), 8, "root display words after clear")
    g.diff_page(0, "old page")
    old = g.pages.pop(0)
    g.connect(0, enabled=ALL)
    s.events("reconnected", order=True)
    g.diff_page(0, "new page")
    s.dump(s.pa(old), 0x10, "old page head")
    g.ack(0, 1)
    s.events("acked")
    return s


@scenario
def display_page_no_display():
    """A page on a port no display is attached to: the guest display starts
    with no client, state 0 stays 0, no interrupt; attaching a display later
    runs the machine. Releasing the display (hot unplug) then raises 8 and
    clears the mode count."""
    s = Scenario("display_page_no_display", doc=display_page_no_display.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0, enabled=ALL)
    s.events("page with no display")
    g.diff_page(0, "page with no display")
    s.api("new_display", spec=display(0))
    s.wait(irqs=2)
    s.events("display attached", order=True)
    _irq_regs(s, "display attached")
    g.diff_page(0, "page after attach")
    g.ack(0, 1)
    s.api("release_display", display=0)
    s.wait(irqs=4)
    s.events("display released", order=True)
    _irq_regs(s, "display released")
    g.diff_page(0, "page after release")
    s.api("new_display", spec=display(0, serial=77, name="second", modes=((800, 600, 30),)))
    s.wait(irqs=5)
    s.events("another display attached", order=True)
    g.diff_page(0, "page after the second attach")
    s.api("display_props", display=1)
    return s


@scenario
def display_page_no_modes():
    """A display with no mode list: connecting leaves state 0 and raises
    nothing; setting a mode list later runs the machine (4). A nil or empty
    list is ignored."""
    s = Scenario("display_page_no_modes", displays=[display(0, modes=None)], doc=display_page_no_modes.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0, enabled=ALL)
    s.events("no modes")
    g.diff_page(0, "no modes")
    s.api("set_modes", display=0, nil=True, label="nil")
    s.api("set_modes", display=0, modes=[], label="empty")
    s.quiesce()
    s.events("after nil and empty")
    s.api("display_props", display=0)
    s.api("set_modes", display=0, modes=[[1024, 768, 60], [640, 480, 75.5]])
    s.wait(irqs=2)
    s.events("modes set", order=True)
    g.diff_page(0, "modes set")
    return s


@scenario
def display_modes_after_connect():
    """Mode list changes after the ack: a changed list raises 12 (update)
    with expectedAck 2; an identical list changes nothing and raises
    nothing; a change while an ack is pending ("bogus ack") only updates
    the list, and the ack then raises the update. Modes beyond 128 are
    cut to 128."""
    s = Scenario("display_modes_after_connect", displays=[display(0)], doc=display_modes_after_connect.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0, enabled=ALL)
    g.ack(0, 1)
    s.events("connected")
    s.api("set_modes", display=0, modes=[[64, 32, 60], [128, 64, 30]])
    s.wait(irqs=4)
    s.events("changed", order=True)
    _irq_regs(s, "changed")
    g.diff_page(0, "changed")
    s.api("set_modes", display=0, modes=[[256, 256, 60]], label="while an ack is pending")
    s.quiesce()
    s.events("pending")
    g.diff_page(0, "pending")
    g.ack(0, 2)
    s.events("acked 2", order=True)
    g.diff_page(0, "acked 2")
    g.ack(0, 3)
    s.api("set_modes", display=0, modes=[[256, 256, 60]], label="identical")
    s.quiesce()
    s.events("identical")
    g.diff_page(0, "identical")
    many = [[16 + i, 8 + i, 60] for i in range(130)]
    s.api("set_modes", display=0, modes=many, label="130 modes")
    s.wait(irqs=8)
    s.events("130 modes")
    g.diff_page(0, "130 modes")
    s.api("display_props", display=0)
    return s


@scenario
def display_mode_fields():
    """Each mode field is compared on its own: width, height, refresh (16.16,
    rounded), and the EDR and 10 bpc bits, which the defaults leave 0.
    Invalid modes (0 size) are nil and the runner drops them; the rest are
    written in order."""
    s = Scenario("display_mode_fields", displays=[display(0)], doc=display_mode_fields.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0, enabled=ALL)
    g.ack(0, 1)
    ack = 1
    for label, modes in (("width", [[65, 32, 60]]), ("height", [[65, 33, 60]]),
                         ("refresh", [[65, 33, 59.999]]), ("zero sizes", [[0, 0, 60], [66, 33, 59.999]]),
                         ("fractional refresh", [[66, 33, 1 / 3.0]])):
        s.api("set_modes", display=0, modes=modes, label=label)
        s.quiesce()
        s.events(label)
        g.diff_page(0, label)
        ack += 1
        g.ack(0, ack, fault=False)
    s.events("end")
    return s


@scenario
def display_guest_state_values():
    """The state word is guest-writable. A value other than 0-2 is a "bogus
    state", handled as 2: with a nonzero guest mode count it resends (12)
    even when nothing changed; with a mode count of 0 and no pending list it
    disconnects (8). A guest state 1 with nothing pending moves on to 2.
    Interrupt totals count the stamps' too."""
    s = Scenario("display_guest_state_values", displays=[display(0)], doc=display_guest_state_values.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0, enabled=ALL)
    g.ack(0, 1)
    s.events("connected")
    s.u32(g.ss(0, SS_STATE), 7)
    s.api("set_modes", display=0, modes=[[64, 32, 60]], label="bogus state, same modes")
    s.wait(irqs=4)
    s.events("bogus state", order=True)
    g.diff_page(0, "bogus state")
    g.ack(0, 2)
    s.u32(g.ss(0, SS_STATE), 7)
    s.mem(g.ss(0, SS_MODE_COUNT), struct.pack("<H", 0))
    s.mem(g.ss(0, SS_ACK_PENDING), b"\x01")
    g.ack(0, 2, label="bogus state, mode count 0, through an ack")
    s.events("bogus, no modes", order=True)
    g.diff_page(0, "bogus, no modes")
    s.api("set_modes", display=0, modes=[[64, 32, 60]], label="state 0 again")
    s.wait(irqs=8)
    s.events("reconnect", order=True)
    g.diff_page(0, "reconnect")
    g.ack(0, 3)
    s.u32(g.ss(0, SS_STATE), 1)
    s.api("set_modes", display=0, modes=[[64, 32, 30]], label="state 1, nothing pending")
    s.wait(irqs=10)
    s.events("state 1", order=True)
    g.diff_page(0, "state 1")
    return s


@scenario
def display_guest_mode_count_zero():
    """A guest that zeroes the mode count while an update is pending: the ack
    runs the machine with no new list, which disconnects (8) and leaves state
    0; the next mode list reconnects (4). A list that arrives while the count
    is 0 rewrites it instead."""
    s = Scenario("display_guest_mode_count_zero", displays=[display(0)], doc=display_guest_mode_count_zero.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0, enabled=ALL)
    g.ack(0, 1)
    s.api("set_modes", display=0, modes=[[64, 32, 30]])
    s.wait(irqs=4)
    s.mem(g.ss(0, SS_MODE_COUNT), struct.pack("<H", 0))
    g.ack(0, 2)
    s.events("disconnect", order=True)
    g.diff_page(0, "disconnect")
    s.api("set_modes", display=0, modes=[[64, 32, 60]], label="again")
    s.wait(irqs=7)
    s.events("reconnect", order=True)
    g.diff_page(0, "reconnect")
    g.ack(0, 3)
    s.mem(g.ss(0, SS_MODE_COUNT), struct.pack("<H", 0))
    s.api("set_modes", display=0, modes=[[64, 32, 60]], label="count 0, then a list")
    s.wait(irqs=9)
    s.events("rewritten", order=True)
    g.diff_page(0, "rewritten")
    return s


@scenario
def display_guest_mode_count_over():
    """A guest mode count above 128 asserts in GuestDisplay::sync_mode_list at
    the next state machine run that updates the modes: the host dies at the
    set_modes call."""
    s = Scenario("display_guest_mode_count_over", displays=[display(0)],
                 doc=display_guest_mode_count_over.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0, enabled=ALL)
    g.ack(0, 1)
    s.events("connected")
    s.mem(g.ss(0, SS_MODE_COUNT), struct.pack("<H", 129))
    s.api("set_modes", display=0, modes=[[64, 32, 60]])
    s.events("after")
    return s


@scenario
def display_color_matrix():
    """SetProperties (10) property 0 writes 12 floats into
    colorMatrices[writeIndex] and sets dirty; a wrong word count and an
    unknown property throw, and the command still signals its stamp. The write
    index is a guest byte: 255 writes 48 bytes at 0xa10 + 255 * 48, inside the
    page."""
    s = Scenario("display_color_matrix", displays=[display(0)], doc=display_color_matrix.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0)
    g.ack(0, 1)
    s.events("connected")
    m = floats(0.5, 0, 0, 0.1, 0, 0.25, 0, 0.2, 0, 0, 2.0, 0.3)
    s.submit(abi.OP_DISPLAY_SET_PROPERTIES, p_props(0, [(0, m)]))
    g.diff_page(0, "matrix 1")
    s.submit(abi.OP_DISPLAY_SET_PROPERTIES, p_props(0, [(0, IDENTITY), (0, m)]), label="two properties")
    g.diff_page(0, "two properties")
    for label, props in (("11 words", [(0, m[:11])]), ("unknown property", [(1, [1, 2])]),
                         ("property -1", [(0xFFFFFFFF, [])])):
        s.submit(abi.OP_DISPLAY_SET_PROPERTIES, p_props(0, props), label=label)
        s.fault_state(label)
        g.diff_page(0, label)
    s.mem(g.ss(0, SS_CM_WRITE), bytes([255]))
    s.submit(abi.OP_DISPLAY_SET_PROPERTIES, p_props(0, [(0, m)]), label="write index 255")
    g.diff_page(0, "write index 255")
    s.events("end")
    return s


@scenario
def display_set_properties_payloads():
    """SetProperties' payload checks: fewer bytes than a property header or
    its words throws, a word count whose byte size overflows u32 throws
    ("Overflow"), a port without a page throws; every one signals the
    stamp. Below version 40 the command is unsupported, and still
    signals."""
    s = Scenario("display_set_properties_payloads", displays=[display(0)],
                 doc=display_set_properties_payloads.__doc__)
    g = DisplayGuest(s, task0=False)
    s.submit(abi.OP_DISPLAY_SET_PROPERTIES, p_props(0, [(0, IDENTITY)]), label="no page")
    g.connect(0)
    g.ack(0, 1)
    for label, payload in (
            ("short header", struct.pack("<I", 0)),
            ("count 1, no property", struct.pack("<II", 0, 1)),
            ("words missing", struct.pack("<IIII", 0, 1, 0, 12) + b"\0" * 8),
            ("word count overflows", struct.pack("<IIII", 0, 1, 0, 0x40000000)),
            ("count 0", struct.pack("<II", 0, 0))):
        s.submit(abi.OP_DISPLAY_SET_PROPERTIES, payload, label=label)
    s.fault_state("after")
    g.diff_page(0, "page")
    s.w(abi.R_VERSION, 30)
    s.submit(abi.OP_DISPLAY_SET_PROPERTIES, p_props(0, [(0, IDENTITY)]), label="version 30")
    s.r(abi.R_VERSION)
    g.diff_page(0, "page at version 30")
    s.events("end")
    return s


@scenario
def display_compositor_parameters():
    """CompositorParameters (11): validated, then dropped: the C API has no
    callback for it, and nothing reaches the page. Below version 108 it
    faults, a short payload faults, and a port without a page faults."""
    s = Scenario("display_compositor_parameters", displays=[display(0)],
                 doc=display_compositor_parameters.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0)
    g.ack(0, 1)
    s.submit(abi.OP_DISPLAY_COMPOSITOR_PARAMETERS, p_compositor(0, 10, 20, 5, 2))
    g.diff_page(0, "after")
    for label, payload in (("short", struct.pack("<I", 0)), ("port 3", p_compositor(3, 0, 0, 0, 0))):
        s.submit(abi.OP_DISPLAY_COMPOSITOR_PARAMETERS, payload, fault=True, label=label)
        s.fault_state(label)
        s.resume_root()
        s.submit(abi.OP_NOP)
    s.w(abi.R_VERSION, 107)
    s.submit(abi.OP_DISPLAY_COMPOSITOR_PARAMETERS, p_compositor(0, 1, 2, 3, 4), fault=True, label="version 107")
    s.fault_state("version 107")
    s.events("end")
    return s


@scenario
def display_icc_profile():
    """SetGuestICCProfile (12) stores the VA and length at +0xfc8 and reads
    the profile from task 0; length > mappedLength throws, and every
    failure still signals the stamp. A dirty colour space makes the next
    mapped-surface present call mode_change."""
    s = Scenario("display_icc_profile", displays=[display(0)], doc=display_icc_profile.__doc__)
    g = DisplayGuest(s)
    g.connect(0)
    g.ack(0, 1)
    va = g.t0.blob(b"not an ICC profile" + bytes(46))
    s.submit(abi.OP_DISPLAY_SET_GUEST_ICC_PROFILE, p_icc(0, 64, va, 64), label="garbage profile")
    g.diff_page(0, "garbage profile")
    s.submit(abi.OP_DISPLAY_SET_GUEST_ICC_PROFILE, p_icc(0, 65, va, 64), label="length > mapped")
    s.submit(abi.OP_DISPLAY_SET_GUEST_ICC_PROFILE, p_icc(0, 0, 0, 0), label="cleared")
    g.diff_page(0, "cleared")
    s.submit(abi.OP_DISPLAY_SET_GUEST_ICC_PROFILE, p_icc(0, 16, 0x7FFF0000, 16), label="VA outside the task")
    s.submit(abi.OP_DISPLAY_SET_GUEST_ICC_PROFILE, struct.pack("<I", 0), label="short")
    s.submit(abi.OP_DISPLAY_SET_GUEST_ICC_PROFILE, p_icc(5, 0, 0, 0), label="port 5")
    s.fault_state("after")
    s.w(abi.R_VERSION, 207)
    s.submit(abi.OP_DISPLAY_SET_GUEST_ICC_PROFILE, p_icc(0, 0, 0, 0), label="version 207")
    s.fault_state("version 207")
    s.events("end")
    return s


@scenario
def display_live_resize_latch():
    """The guest's compositorSupportsLiveResize byte (+0x51) is latched on
    the first doorbell that sees it set; it only reaches the private
    responsiveness handler, so the guest sees nothing change."""
    s = Scenario("display_live_resize_latch", displays=[display(0)], doc=display_live_resize_latch.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0)
    g.ack(0, 1)
    s.mem(g.ss(0, SS_LIVE_RESIZE), b"\x01")
    s.w(abi.R_DISPLAY_DOORBELL, 0)
    s.wait(cb="cursorMove")
    s.w(abi.R_DISPLAY_DOORBELL, 0)
    s.wait(cb="cursorMove", cb_count=2)
    s.events("doorbells")
    g.diff_page(0, "page")
    return s
