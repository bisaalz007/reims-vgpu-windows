"""EFI framebuffer coverage gaps (docs/testing.md, section 6.22): the 100 ms
timer without a death, the EFI display handing over to the guest's driver
on the same display, and a bound display unplugged while the timer runs.

The timer's new_frame count before a checkpoint is timing, so the timer
scenarios collapse consecutive new_frame callbacks of one display into one
(the events op's "collapse"), stop the frame (depth 0) before each
checkpoint, and wait for mode_change, which each rebuild of another size
calls exactly once.
"""
from pgtest import abi
from pgtest.abi import PAGE
from pgtest.display_guest import DisplayGuest, display, p_present3, pattern
from pgtest.efi_guest import EFI
from pgtest.registry import scenario
from pgtest.scenario import Scenario

TRACE = {"trace": True}


def _events(s, label):
    """An events checkpoint with new_frame runs collapsed."""
    s.quiesce(250)
    s.op(op="events", label=label, collapse=["newFrame"])


@scenario(host=True)
def efi_gap_timer_collapsed():
    """The timer's presents, counted as runs: a valid frame starts the timer,
    whose first tick calls mode_change and a new_frame, then only new_frame
    calls; depth 0 stops it (the rebuild cancels the timer and no frame is
    made). The frame stays a zero-copy view of guest memory while stopped: a
    guest write shows in the encode. Another mode and depth 32 again restart it
    (mode_change for the new size); a reset (DeviceCore::reset) while it runs
    tears EFI down and zeroes its registers, and nothing is presented after."""
    s = Scenario("efi_gap_timer_collapsed", host=True, displays=[display(0)], doc=efi_gap_timer_collapsed.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    e = EFI(s, pages=2)
    e.write_frame(16, 8, 64, seed=1)
    e.setup(16, 8, 64, length=2 * PAGE)
    s.wait(cb="modeChange", label="first tick")
    s.w(abi.R_EFI_DEPTH, 0)
    _events(s, "16x8, stopped")
    s.encode(0, 16, 8, label="16x8", samples=[[0, 0], [15, 7]])
    e.write_frame(16, 8, 64, seed=0x40)
    s.encode(0, 16, 8, label="16x8 after a guest write, stopped", samples=[[0, 0], [15, 7]])
    s.w(abi.R_EFI_MODE, 32 | 12 << 16)
    s.w(abi.R_EFI_STRIDE, 128)
    e.write_frame(32, 12, 128, seed=0x80)
    s.w(abi.R_EFI_DEPTH, 32)
    s.wait(cb="modeChange", cb_count=2, label="32x12")
    s.w(abi.R_EFI_DEPTH, 0)
    _events(s, "32x12, stopped")
    s.encode(0, 32, 12, label="32x12", samples=[[0, 0], [31, 11]])
    s.w(abi.R_EFI_MODE, 16 | 8 << 16)
    s.w(abi.R_EFI_DEPTH, 32)
    s.wait(cb="modeChange", cb_count=3, label="16x8 again")
    s.api("reset")
    _events(s, "running, then reset")
    e.regs("after the reset")
    return s


@scenario(host=True)
def efi_gap_handoff_to_guest():
    """EFI and the guest's driver present to the same display: EFI (trace
    blocks, so each present follows a rebuild or a trace_fire) presents 16x8;
    the guest's shared page attaches the guest display, which drops the current
    frame (the encode clears); after the ack the guest presents 32x16
    (mode_change); a trace_fire still presents EFI's frame through its weak
    display reference (mode_change back to 16x8); EFI disabled (0x1200 = 0)
    removes its trace range, and the guest's next present is 32x16 again."""
    s = Scenario("efi_gap_handoff_to_guest", host=True, displays=[display(0)], descriptor=TRACE,
                 doc=efi_gap_handoff_to_guest.__doc__)
    g = DisplayGuest(s)
    ref, _ = g.backing(32, 16, pixels=pattern(32, 16, 3))
    g.install()
    e = EFI(s)
    e.write_frame(16, 8, 64, seed=5)
    e.setup(16, 8, 64)
    s.wait(cb="newFrame")
    s.events("EFI presented")
    s.encode(0, 16, 8, label="EFI frame", samples=[[0, 0], [15, 7]])
    g.connect(0, enabled=0xF)
    s.events("page attached")
    s.encode(0, 16, 8, label="after the attach", samples=[[0, 0], [15, 7]])
    g.ack(0, 1)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, ref), label="guest present")
    s.wait(cb="newFrame", cb_count=2)
    s.events("guest presented")
    s.encode(0, 32, 16, label="guest frame", samples=[[0, 0], [31, 15]])
    s.trace_fire(e.gpa(0x10), 4)
    s.wait(cb="newFrame", cb_count=3)
    s.events("EFI again")
    s.encode(0, 16, 8, label="EFI frame again", samples=[[0, 0], [15, 7]])
    s.w(abi.R_EFI_ENABLE, 0)
    s.trace_fire(e.gpa(0x10), 4)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, ref), label="guest present after EFI")
    s.wait(cb="newFrame", cb_count=4)
    s.events("EFI disabled")
    s.encode(0, 32, 16, label="guest frame again", samples=[[0, 0], [31, 15]])
    g.diff_page(0, "page")
    s.api("present_counts", display=0)
    return s


@scenario
def efi_gap_timer_unplug():
    """The bound display unplugged while the timer runs: detaching port 0
    unbinds EFI, and the next tick finds no display and stops presenting.
    A display plugged in again is not bound, and a rebuild (a stride
    write) restarts the timer only for its first tick to stop it again:
    nothing is presented."""
    s = Scenario("efi_gap_timer_unplug", displays=[display(0)], doc=efi_gap_timer_unplug.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    e = EFI(s)
    e.write_frame(16, 8, 64)
    e.setup(16, 8, 64)
    s.wait(cb="modeChange", label="first tick")
    s.api("release_display", display=0)
    _events(s, "unplugged")
    s.api("new_display", spec=display(0, serial=2000))
    s.w(abi.R_EFI_STRIDE, 128)
    _events(s, "plugged again, rebuilt")
    s.api("present_counts", display=1)
    e.regs("end")
    return s
