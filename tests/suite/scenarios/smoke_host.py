"""The host side's shortest paths: a display and an EFI present read back,
the IOSurface host device's registers, and suspend and resume onto a fresh
device (docs/testing.md, sections 6.21-6.24)."""
import struct

from pgtest import abi
from pgtest.registry import scenario
from pgtest.scenario import Scenario

DISPLAY = {"port": 0, "serial": 101, "name": "suite", "modes": [[64, 32, 60], [1920, 1080, 60]]}


def efi_frame(s, w=64, h=32, stride=256):
    """A BGRA frame in guest memory: pixel (x, y) = {x, y, x ^ y, 0xff}."""
    page = s.alloc_page()
    rows = []
    for y in range(h):
        row = b"".join(bytes([x & 0xFF, y & 0xFF, (x ^ y) & 0xFF, 0xFF]) for x in range(w))
        rows.append(row + bytes(stride - len(row)))
    s.mem(s.pa(page), b"".join(rows))
    return page


@scenario
def smoke_efi_present():
    """An EFI framebuffer present (section 6.22's measured sequence), then
    the frame read back through rvg_display_encode_frame.

    smoke_efi_present_trace has the add_trace_range callback, so EFI presents
    once, at once: exactly one new_frame, and the display's guestPresentCount
    and hostPresentCount are 1 after the encode.

    smoke_efi_present has none, so a 100 ms timer presents: its first tick
    calls mode_change and a new_frame, and every later tick another new_frame.
    How many ticks fall before a checkpoint is timing (docs/testing.md,
    decision 8), so it locks what the timer establishes: mode_change before
    the first new_frame, a present counted (guestPresentCount at least 1), the
    registers and the frame while it runs; then depth 0 stops the timer (the
    rebuild cancels it and makes no frame), and the callbacks are compared
    with consecutive new_frame calls collapsed into one (efi_gaps' method). The
    display's properties are read before EFI starts, where both present
    counts are 0, since after it guestPresentCount is the tick count."""
    out = []
    for trace in (False, True):
        name = "smoke_efi_present" + ("_trace" if trace else "")
        s = Scenario(name, host=True, displays=[DISPLAY], descriptor={"trace": True} if trace else None,
                     doc=smoke_efi_present.__doc__)
        s.w(abi.R_VERSION, abi.MAX_VERSION)
        page = efi_frame(s)
        if not trace:
            s.api("display_props", display=0)
        s.w(abi.R_EFI_ENABLE, 1)
        s.w(abi.R_EFI_MODE, 64 | 32 << 16)
        s.w(abi.R_EFI_DEPTH, 32)
        s.w(abi.R_EFI_STRIDE, 256)
        s.w(abi.R_EFI_START, s.pfn(page))
        s.w(abi.R_EFI_LENGTH, 0x4000)
        s.wait(cb="modeChange", timeout_ms=5000)
        s.wait(cb="newFrame", timeout_ms=5000)
        if trace:
            s.events("presented")
        else:
            s.wait(display=0, present_count=1, timeout_ms=5000, label="a present counted")
        for r in (abi.R_EFI_ENABLE, abi.R_EFI_MODE_COUNT, abi.R_EFI_MODE_SELECTED, abi.R_EFI_START,
                  abi.R_EFI_LENGTH, abi.R_EFI_DEPTH, abi.R_EFI_MODE, abi.R_EFI_STRIDE_ALIGN, abi.R_EFI_STRIDE):
            s.r(r)
        s.encode(0, 64, 32, samples=[[0, 0], [63, 31], [5, 7]])
        if trace:
            s.api("display_props", display=0)
        else:
            s.w(abi.R_EFI_DEPTH, 0)
            s.quiesce(250)
            s.op(op="events", label="presented, then stopped", collapse=["newFrame"])
        out.append(s)
    return out


@scenario(setups=("map", "isolate"))
def smoke_iosfc_registers():
    """The IOSurface host device's registers, fresh and after writes, through
    QEMU's 32-bit declaration (section 6.23)."""
    s = Scenario("smoke_iosfc_registers", setups=("map", "isolate"), iosfc=True, doc=smoke_iosfc_registers.__doc__)
    s.op(op="iosfc_sweep", **{"from": "0x0", "to": "0x1040", "step": 8})
    s.iosfc_w(0x1030, 1)
    s.iosfc_r(0x1030)
    s.iosfc_r(0x1028)
    s.iosfc_r(0x2000, "unhandled")
    return s


@scenario
def smoke_suspend_resume():
    """Suspend a running device, resume the blob on a fresh one over the same
    guest memory, and run a command on it (section 6.24)."""
    s = Scenario("smoke_suspend_resume", host=True, displays=[DISPLAY], doc=smoke_suspend_resume.__doc__)
    f = s.bring_up()
    st = s.submit(abi.OP_NOP)
    s.submit(abi.OP_GET_DEVICE_INFO, struct.pack("<III", 45, 64, s.pfn(s.alloc_page())))
    s.events("before suspend")
    s.api("willSuspend")
    s.api("finishSuspend")
    s.api("recreate")
    s.api("willResume")
    s.api("didResume")
    s.events("resumed")
    s.sweep(label="registers after resume")
    st = f.cmd(abi.OP_NOP)
    f.doorbell()
    f.wait_stamp(st)
    s.events("a NOP after resume")
    s.diff("after resume")
    s.api("willResume", hex="00", label="a short blob")
    s.api("willResume", hex=struct.pack("<IIII", 0x53475652, 2, 0, 0).hex(), label="version 2")
    return s
