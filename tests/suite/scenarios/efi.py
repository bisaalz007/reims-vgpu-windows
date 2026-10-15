"""The EFI framebuffer (docs/testing.md, section 6.22): its registers,
enable and disable, mode select, the mapping refusals, the conditions under
which the frame is rebuilt, trace ranges against the 100 ms timer, the
display it binds, and its teardown on a version change and on reset.

Most scenarios give the device the trace callbacks, so that every present
follows a rebuild or a trace_fire (section 5.4); a scenario waits for each
present's new_frame before the next register write, so presents never
coalesce.
"""
from pgtest import abi
from pgtest.abi import PAGE
from pgtest.display_guest import display
from pgtest.efi_guest import EFI, frame
from pgtest.registry import scenario
from pgtest.scenario import Scenario

TRACE = {"trace": True}
MODES = ((64, 32, 60), (800, 600, 60), (16384, 16384, 30))


def _presents(s, n, label=None):
    s.wait(cb="newFrame", cb_count=n, label=label)


@scenario
def efi_registers():
    """The EFI registers before EFI exists: writes of start, length, depth,
    mode and stride are stored though there is no EFI display; the mode
    count, selected mode and stride alignment read 0. Enabling creates it:
    0x1200 reads the value written, 0x1204 the display's mode count, 0x1224
    the host's alignment. A second enable keeps the display. Disabling
    zeroes all six stored registers."""
    s = Scenario("efi_registers", host=True, displays=[display(0, modes=MODES)], doc=efi_registers.__doc__)
    e = EFI(s)
    e.regs("fresh")
    s.w(abi.R_EFI_START, 0x1234)
    s.w(abi.R_EFI_LENGTH, 0x5678)
    s.w(abi.R_EFI_DEPTH, 24)
    s.w(abi.R_EFI_MODE, 0x00100020)
    s.w(abi.R_EFI_STRIDE, 0x40)
    s.w(abi.R_EFI_MODE_SELECT, 1)
    e.regs("written without EFI")
    s.w(abi.R_EFI_ENABLE, 7)
    e.regs("enabled with 7")
    s.w(abi.R_EFI_ENABLE, 1)
    e.regs("enabled again with 1")
    s.w(abi.R_EFI_ENABLE, 0)
    e.regs("disabled")
    s.w(abi.R_EFI_ENABLE, 0)
    e.regs("disabled twice")
    s.events("end")
    return s


@scenario
def efi_mode_select():
    """Mode select indexes the display's mode list: 0x120c reads w | h << 16
    of the selected mode, 0xffffffff past the end; the index resets to 0
    when the display is bound. With no display, the count is 0 and every
    index is past the end."""
    out = []
    s = Scenario("efi_mode_select", displays=[display(0, modes=MODES)], doc=efi_mode_select.__doc__)
    s.w(abi.R_EFI_ENABLE, 1)
    for i in (None, 0, 1, 2, 3, 0xFFFFFFFF, 0):
        if i is not None:
            s.w(abi.R_EFI_MODE_SELECT, i)
        s.r(abi.R_EFI_MODE_COUNT)
        s.r(abi.R_EFI_MODE_SELECTED)
    s.api("set_modes", display=0, modes=[[320, 200, 70]])
    s.r(abi.R_EFI_MODE_COUNT, "after a new list")
    s.r(abi.R_EFI_MODE_SELECTED, "after a new list")
    out.append(s)
    s = Scenario("efi_mode_select_no_display", doc=efi_mode_select.__doc__)
    s.w(abi.R_EFI_ENABLE, 1)
    for i in (0, 1):
        s.w(abi.R_EFI_MODE_SELECT, i)
        s.r(abi.R_EFI_MODE_COUNT)
        s.r(abi.R_EFI_MODE_SELECTED)
    out.append(s)
    return out


@scenario(host=True)
def efi_trace_present():
    """With trace callbacks, a valid framebuffer adds one trace range over
    height x stride rounded up to 16 KiB, and presents once; a trace_fire
    inside it presents again, one outside does not. The frame is a
    zero-copy view: a guest write shows in the next encode. Each later
    register write removes the range, and a valid result adds it again and
    presents."""
    s = Scenario("efi_trace_present", host=True, displays=[display(0)], descriptor=TRACE,
                 doc=efi_trace_present.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    e = EFI(s, pages=2)
    e.write_frame(16, 8, 64)
    e.setup(16, 8, 64, length=2 * PAGE)
    _presents(s, 1)
    s.events("first present")
    s.encode(0, 16, 8, label="first")
    s.api("present_counts", display=0)
    e.write_frame(16, 8, 64, seed=0x80)
    s.encode(0, 16, 8, label="after a guest write, before any present")
    s.trace_fire(e.gpa(0x10), 4)
    _presents(s, 2)
    s.trace_fire(e.gpa(PAGE + 0x10), 4)
    s.trace_fire(e.gpa(-4), 4)
    s.quiesce(100)
    s.events("trace fires")
    s.api("present_counts", display=0)
    s.encode(0, 16, 8, label="after the fire")
    s.w(abi.R_EFI_STRIDE, 128)
    _presents(s, 3)
    s.events("stride 128")
    s.encode(0, 16, 8, label="stride 128")
    s.w(abi.R_EFI_MODE, 8 | 4 << 16)
    _presents(s, 4)
    s.events("mode 8x4")
    s.encode(0, 8, 4, label="mode 8x4")
    s.api("display_props", display=0)
    e.regs("end")
    return s


@scenario
def efi_mapping_refusals():
    """The mapping (on start or length writes) needs both nonzero, a length
    that is a 16 KiB multiple and at most 64 MiB, and guest memory that
    maps; otherwise it logs and nothing is presented. A task that failed to
    map is destroyed. Each case starts from a fresh EFI display (0x1200 = 0,
    then 1), because a refusal after a valid mapping leaves a stale task
    (efi_stale_task)."""
    s = Scenario("efi_mapping_refusals", displays=[display(0)], descriptor=TRACE, doc=efi_mapping_refusals.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    e = EFI(s)
    for label, start, length in (("length 0", s.pfn(e.page), 0),
                                 ("start 0", 0, PAGE),
                                 ("length not a page multiple", s.pfn(e.page), PAGE + 4),
                                 ("length over 64 MiB", s.pfn(e.page), 0x4004000),
                                 ("64 MiB, past the end of RAM", s.pfn(e.page), 0x4000000),
                                 ("outside guest RAM", 0x10, PAGE)):
        s.mark(label)
        s.w(abi.R_EFI_ENABLE, 1)
        s.w(abi.R_EFI_MODE, 16 | 8 << 16)
        s.w(abi.R_EFI_DEPTH, 32)
        s.w(abi.R_EFI_STRIDE, 64)
        s.w(abi.R_EFI_START, start)
        s.w(abi.R_EFI_LENGTH, length)
        s.quiesce()
        s.events(label)
        e.regs(label)
        s.w(abi.R_EFI_ENABLE, 0)
        s.events(label + ", disabled")
    s.w(abi.R_EFI_ENABLE, 1)
    e.setup(16, 8, 64, enable=False)
    _presents(s, 1)
    s.events("valid")
    e.regs("end")
    return s


@scenario
def efi_stale_task():
    """A mapping refusal after a valid mapping: the task rebuild
    (EfiDisplay::rebuild_task) destroys the task but returns early with the
    task still marked ready, so the mode rebuild maps the new range with the
    stale task. Start 0 or a range outside RAM then fails to map and clears the
    task; length 0 maps an empty range; an unaligned length asserts in the
    memory map (totalLength % 16 KiB, a host death). In the blocks setup the
    host is asked to unmap the task it was just told to destroy (its record
    says "destroyed"), and for start 0, length 0, an over-long or an unaligned
    length its map_memory callback is called with a NULL task, which the
    runner's host refuses."""
    out = []
    for name, reg, val in (("start_0", abi.R_EFI_START, 0), ("length_0", abi.R_EFI_LENGTH, 0),
                           ("outside_ram", abi.R_EFI_START, 0x10),
                           ("length_over_64m", abi.R_EFI_LENGTH, 0x4004000),
                           ("unaligned_length", abi.R_EFI_LENGTH, PAGE + 4)):
        s = Scenario("efi_stale_task_" + name, displays=[display(0)], descriptor=TRACE, doc=efi_stale_task.__doc__)
        s.w(abi.R_VERSION, abi.MAX_VERSION)
        e = EFI(s)
        e.setup(16, 8, 64)
        _presents(s, 1)
        s.events("valid")
        s.w(reg, val)
        s.quiesce()
        s.events(name)
        e.regs(name)
        s.w(abi.R_EFI_STRIDE, 128)
        s.quiesce()
        s.events("a later write")
        s.api("present_counts", display=0)
        out.append(s)
    return out


@scenario
def efi_rebuild_conditions():
    """The frame is rebuilt on every register write only when the task is
    mapped, the mode is nonzero, depth is 32, the stride (0 means 4 x
    width) is at least 4 x width and a multiple of 0x1224's alignment, and
    height x stride fits the length; a mode wider or taller than the
    device's 2D limit becomes 0 x 0. Otherwise nothing is presented. The
    alignment is 16 on the hosts measured; 8 is never a multiple of it."""
    s = Scenario("efi_rebuild_conditions", displays=[display(0)], descriptor=TRACE,
                 doc=efi_rebuild_conditions.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    e = EFI(s)
    e.setup(16, 8, 64)
    _presents(s, 1)
    s.events("valid")
    n = 1
    for label, reg, bad, good in (("depth 24", abi.R_EFI_DEPTH, 24, 32),
                                  ("stride below 4w", abi.R_EFI_STRIDE, 60, 64),
                                  ("stride not aligned", abi.R_EFI_STRIDE, 72, 64),
                                  ("mode 0", abi.R_EFI_MODE, 0, 16 | 8 << 16),
                                  ("height x stride over the length", abi.R_EFI_MODE, 16 | 300 << 16, 16 | 8 << 16),
                                  ("width over the 2D limit", abi.R_EFI_MODE, 16385 | 8 << 16, 16 | 8 << 16),
                                  ("height over the 2D limit", abi.R_EFI_MODE, 16 | 16385 << 16, 16 | 8 << 16),
                                  ("stride 0: 4 x width", abi.R_EFI_STRIDE, None, None)):
        s.mark(label)
        if bad is None:
            s.w(abi.R_EFI_STRIDE, 0)
            n += 1
            _presents(s, n, label)
            s.events(label)
            continue
        s.w(reg, bad)
        s.quiesce()
        s.events(label)
        e.regs(label)
        s.w(reg, good)
        n += 1
        _presents(s, n, label + ", restored")
        s.events(label + ", restored")
    s.api("present_counts", display=0)
    return s


@scenario(host=True)
def efi_timer():
    """Without trace callbacks the frame is presented by a 100 ms timer; the
    frame reaches the encode. How many timer presents land before a
    checkpoint is timing, so this scenario ends in a death (writing 2 to
    0x1000 while the FIFO is enabled asserts) before any events
    checkpoint could count them."""
    s = Scenario("efi_timer", host=True, displays=[display(0)], doc=efi_timer.__doc__)
    s.bring_up()
    e = EFI(s)
    e.write_frame(16, 8, 64, seed=3)
    e.setup(16, 8, 64)
    _presents(s, 2)
    s.encode(0, 16, 8, label="timer frame")
    s.w(abi.R_FIFO_ENABLE, 2)
    return s


@scenario
def efi_timer_no_display():
    """EFI with no display bound, on the timer: the first tick finds no
    display and stops presenting (cancels its own timer). A display
    attached afterwards is not bound, because EFI binds only at
    creation: nothing is ever presented."""
    s = Scenario("efi_timer_no_display", doc=efi_timer_no_display.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    e = EFI(s)
    e.setup(16, 8, 64)
    s.api("new_display", spec=display(0))
    s.quiesce(300)
    s.events("no present")
    s.api("present_counts", display=0)
    s.encode(0, 16, 8, label="no frame")
    return s


@scenario
def efi_binding():
    """EFI binds the display parked at port 0 when EFI is created, and only
    then: created before any display it binds none; a display with a shared
    page is not parked (its guest display holds it), so EFI binds none either;
    releasing the bound display unbinds it, and a display on port 1 is never
    bound. Trace blocks make every present deterministic."""
    out = []
    # Created before the display.
    s = Scenario("efi_binding_before_display", descriptor=TRACE, doc=efi_binding.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    e = EFI(s)
    s.w(abi.R_EFI_ENABLE, 1)
    s.api("new_display", spec=display(0))
    e.setup(16, 8, 64, enable=False)
    s.quiesce(100)
    s.events("no present")
    s.w(abi.R_EFI_ENABLE, 0)
    s.w(abi.R_EFI_ENABLE, 1)
    e.setup(16, 8, 64, enable=False)
    _presents(s, 1, "re-created: bound")
    s.events("bound after re-creation")
    out.append(s)
    # A display with a shared page.
    s = Scenario("efi_binding_display_with_page", descriptor=TRACE, displays=[display(0)], doc=efi_binding.__doc__)
    s.bring_up()
    page = s.alloc_page()
    import struct
    s.submit(abi.OP_DISPLAY_SET_SHARED_STATE_PAGE, struct.pack("<II", 0, s.pfn(page)))
    e = EFI(s)
    e.setup(16, 8, 64)
    s.quiesce(100)
    s.events("no present")
    s.api("present_counts", display=0)
    s.r(abi.R_EFI_MODE_COUNT)
    out.append(s)
    # Released.
    s = Scenario("efi_binding_released", descriptor={"trace": True, "port_count": 2},
                 displays=[display(0), display(1)], doc=efi_binding.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    e = EFI(s)
    e.setup(16, 8, 64)
    _presents(s, 1)
    s.events("bound")
    s.r(abi.R_EFI_MODE_COUNT, "bound")
    s.api("release_display", display=0)
    s.r(abi.R_EFI_MODE_COUNT, "released")
    s.w(abi.R_EFI_STRIDE, 128)
    s.quiesce(100)
    s.events("released: a rebuild presents nothing")
    s.api("present_counts", display=1)
    out.append(s)
    return out


@scenario
def efi_teardown():
    """EFI is torn down (the trace range removed, the task destroyed, the six
    registers zeroed) by a version without efiDisplay (2, and 0), by a reset
    (DeviceCore::reset), and by 0x1200 = 0; version 3 keeps it."""
    out = []
    for name, action in (("version_2", ("w", abi.R_VERSION, 2)), ("version_0", ("w", abi.R_VERSION, 0)),
                         ("version_3", ("w", abi.R_VERSION, 3)), ("reset", ("api", "reset")),
                         ("disable", ("w", abi.R_EFI_ENABLE, 0)), ("stop", ("api", "stop"))):
        s = Scenario("efi_teardown_" + name, displays=[display(0)], descriptor=TRACE, doc=efi_teardown.__doc__)
        s.w(abi.R_VERSION, abi.MAX_VERSION)
        e = EFI(s)
        e.setup(16, 8, 64)
        _presents(s, 1)
        s.events("presented")
        if action[0] == "w":
            s.w(action[1], action[2])
        else:
            s.api(action[1])
        s.quiesce()
        s.events(name)
        e.regs(name)
        s.r(abi.R_VERSION)
        s.trace_fire(e.gpa(), 4)
        s.quiesce(100)
        s.events("after a trace fire")
        out.append(s)
    return out


@scenario
def efi_before_version():
    """EFI registers work before any binary version is written (features are 0,
    and DeviceCore::set_efi_display does not check them); writing version 2
    then tears it down."""
    s = Scenario("efi_before_version", displays=[display(0)], descriptor=TRACE, doc=efi_before_version.__doc__)
    e = EFI(s)
    e.setup(16, 8, 64)
    _presents(s, 1)
    s.events("presented")
    e.regs("before a version")
    s.w(abi.R_VERSION, 2)
    s.events("version 2")
    e.regs("version 2")
    return s


@scenario
def efi_display_without_newframe():
    """An EFI present to a display without a new_frame callback asserts in
    DisplayCore::present_frame, and the host dies. Without trace callbacks the
    first timer tick presents. The death is on EFI's present queue, after the
    tick has queued mode_change to the display's queue, so the scenario ends in
    a wait only the death ends (0x102c never reads 2) rather than one for
    mode_change, which could return first."""
    s = Scenario("efi_display_without_newframe", displays=[display(0, handlers=["modeChange"])],
                 doc=efi_display_without_newframe.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    e = EFI(s)
    e.setup(16, 8, 64)
    s.wait(mmio=abi.R_FAULT_MASK, mmio_eq=2, timeout_ms=10000, label="the first tick: the host dies")
    return s


@scenario
def efi_trace_add_only():
    """A config with add_trace_range and no remove_trace_range is refused
    at device creation (RVG_ERR_INVALID_ARGUMENT); one with only
    remove_trace_range is accepted, and EFI
    then falls back to the timer (here with no display,
so the first tick stops it)."""
    out = [Scenario("efi_trace_add_only", descriptor={"trace": {"add": True}}, doc=efi_trace_add_only.__doc__)]
    s = Scenario("efi_trace_remove_only", descriptor={"trace": {"remove": True}}, doc=efi_trace_add_only.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    e = EFI(s)
    e.setup(16, 8, 64)
    s.quiesce(300)
    s.events("timer, no display")
    s.w(abi.R_EFI_ENABLE, 0)
    out.append(s)
    return out
