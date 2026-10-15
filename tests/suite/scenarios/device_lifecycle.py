"""The device's lifecycle through the host API (reset, pause, unpause, stop)
and the interrupt and resume registers (docs/testing.md, sections 4.4,
6.2 and 7.3).

DeviceCore::pause quiesces and drains every FIFO and pauses each guest
display; DeviceCore::unpause restarts them and always raises one interrupt;
DeviceCore::interrupt raises only while running; DeviceCore::stop is
DeviceCore::set_root_page(0); DeviceCore::reset also sets version 0 and tears
EFI down. A guest display asserts on pause twice and on unpause without pause.
"""
import struct

from pgtest import abi
from pgtest.fifo import ChildFIFO
from pgtest.registry import scenario
from pgtest.scenario import Scenario


def regs(s, label):
    s.sweep(0x1000, 0x1300, label=label)


def display_page(s, port=0):
    """DisplaySetSharedStatePage (1): {u32 port, u32 page} gives the port a
    guest display, with or without a host display."""
    page = s.alloc_page()
    s.submit(abi.OP_DISPLAY_SET_SHARED_STATE_PAGE, struct.pack("<II", port, s.pfn(page)))
    return page


@scenario
def device_lifecycle_pause_unpause():
    """While paused the root FIFO is drained: a NOP rung then is not run
    (its stamp wait times out), and the unpause restarts the FIFO, which
    runs it; unpause raises one interrupt of its own. Registers and guest
    memory across the pause."""
    s = Scenario("device_lifecycle_pause_unpause", doc=device_lifecycle_pause_unpause.__doc__)
    f = s.bring_up()
    s.submit(abi.OP_NOP)
    s.events("running")
    s.api("pause")
    s.events("paused")
    regs(s, "paused")
    st = f.cmd(abi.OP_NOP)
    f.doorbell()
    f.wait_stamp(st, timeout_ms=500, label="a NOP while paused")
    s.events("a NOP rung while paused")
    s.r(abi.R_FIFO_READ)
    s.api("unpause")
    f.wait_stamp(st)
    s.events("unpaused: the NOP ran")
    regs(s, "unpaused")
    s.diff("after")
    return s


@scenario
def device_lifecycle_pause_quirks():
    """Without a guest display nothing asserts: pause twice, then unpause once,
    runs the next NOP; unpause while running starts the FIFO again and raises
    an interrupt. Pause and unpause with no root page or FIFO at all raise only
    unpause's interrupt."""
    s = Scenario("device_lifecycle_pause_quirks", doc=device_lifecycle_pause_quirks.__doc__)
    s.api("pause")
    s.api("unpause")
    s.events("pause and unpause, no FIFO")
    f = s.bring_up()
    s.api("pause")
    s.api("pause")
    s.events("paused twice")
    s.api("unpause")
    s.submit(abi.OP_NOP)
    s.events("unpaused once, a NOP")
    s.api("unpause")
    s.events("unpause while running")
    s.submit(abi.OP_NOP)
    s.submit(abi.OP_NOP)
    s.events("two NOPs")
    regs(s, "after")
    return s


@scenario
def device_lifecycle_pause_twice_with_display_dies():
    """With a guest display (a shared state page on port 0), a second pause
    asserts in the guest display: the host dies."""
    s = Scenario("device_lifecycle_pause_twice_with_display_dies",
                 doc=device_lifecycle_pause_twice_with_display_dies.__doc__)
    s.bring_up()
    display_page(s)
    s.events("a nub")
    s.api("pause")
    s.events("paused")
    s.api("pause")
    return s


@scenario
def device_lifecycle_unpause_without_pause_with_display_dies():
    """With a guest display, unpause without a pause asserts in the guest
    display: the host dies."""
    s = Scenario("device_lifecycle_unpause_without_pause_with_display_dies",
                 doc=device_lifecycle_unpause_without_pause_with_display_dies.__doc__)
    s.bring_up()
    display_page(s)
    s.events("a nub")
    s.api("unpause")
    return s


@scenario
def device_lifecycle_pause_with_display():
    """Pause and unpause with a guest display and a child FIFO: the child is
    drained and restarted too (its running flag), a NOP on it waits for the
    unpause, and the guest display survives."""
    s = Scenario("device_lifecycle_pause_with_display", doc=device_lifecycle_pause_with_display.__doc__)
    s.bring_up()
    page = display_page(s)
    c = ChildFIFO(s, 1)
    c.setup_node()
    c.define()
    c.submit(abi.OP_NOP)
    s.events("a nub and a child")
    s.diff("before the pause")
    s.api("pause")
    s.events("paused")
    s.diff("paused")
    st = c.cmd(abi.OP_NOP)
    c.doorbell()
    c.wait_stamp(st, timeout_ms=500, label="child NOP while paused")
    s.api("unpause")
    c.wait_stamp(st)
    s.events("unpaused")
    s.diff("after")
    return s


@scenario
def device_lifecycle_stop():
    """stop is a root page of 0: the FIFO and the root go, the version stays; a
    new root page and FIFO then work. With a guest display its root word is
    cleared. stop twice is harmless."""
    s = Scenario("device_lifecycle_stop", doc=device_lifecycle_stop.__doc__)
    f = s.bring_up()
    display_page(s)
    s.submit(abi.OP_NOP)
    s.events("running")
    s.api("stop")
    s.events("stopped")
    regs(s, "stopped")
    s.diff("stopped")
    s.api("stop")
    regs(s, "stopped twice")
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    s.w(abi.R_FIFO_ENABLE, 1)
    f.written = 0
    s.submit(abi.OP_NOP)
    s.events("restarted")
    regs(s, "restarted")
    s.diff("restarted")
    return s


@scenario
def device_lifecycle_reset():
    """reset tears the root down, sets binary version 0 (so maxFIFOCount
    32) and tears EFI down. Registers before and after; after it the guest
    negotiates again and runs. reset on a fresh device is harmless."""
    s = Scenario("device_lifecycle_reset", doc=device_lifecycle_reset.__doc__)
    s.api("reset")
    regs(s, "reset fresh")
    f = s.bring_up()
    s.w(abi.R_EFI_ENABLE, 1)
    s.w(abi.R_EFI_START, 0x77)
    s.submit(abi.OP_NOP)
    s.events("running")
    regs(s, "before reset")
    s.api("reset")
    s.events("reset")
    regs(s, "after reset")
    s.diff("after reset")
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    s.w(abi.R_FIFO_ENABLE, 1)
    f.written = 0
    s.submit(abi.OP_NOP)
    s.events("a NOP at version 0")
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    s.submit(abi.OP_NOP)
    s.events("a NOP at 245")
    regs(s, "renegotiated")
    s.diff("after")
    return s


@scenario
def device_lifecycle_paused_interrupts():
    """DeviceCore::interrupt raises only while running, and DeviceCore::pause
    sets the paused state only after draining: pausing with the root FIFO and a
    child faulted publishes what the drain completes, with interrupts, before
    the state changes. Resumes (0x1024, 0x1028) while paused then raise
    nothing; unpause raises its own."""
    s = Scenario("device_lifecycle_paused_interrupts", doc=device_lifecycle_paused_interrupts.__doc__)
    f = s.bring_up()
    c = ChildFIFO(s, 1)
    c.setup_node()
    c.define()
    c.submit(99, fault=True)
    f.cmd(99)
    st = f.cmd(abi.OP_NOP)
    f.doorbell()
    s.wait_fault()
    s.wait(mmio=abi.R_FAULT_OFFSET, mmio_ne=0xFFFFFFFF)
    s.events("root and child faulted")
    s.fault_state("both faulted")
    s.api("pause")
    s.events("paused")
    s.fault_state("paused")
    s.diff("paused")
    s.w(abi.R_RESUME_ROOT, 0)
    s.events("root resumed while paused")
    s.fault_state("root resumed while paused")
    s.w(abi.R_RESUME_CHILD, 1)
    s.events("child resumed while paused")
    s.fault_state("resumed while paused")
    s.api("unpause")
    f.wait_stamp(st)
    s.events("unpaused")
    s.fault_state("unpaused")
    s.diff("after")
    return s


@scenario
def device_lifecycle_resume_registers():
    """0x1024 and 0x1028. A root resume with a child still faulted raises
    the interrupt again and 0x102c stays 1; the child's resume clears it.
    Resuming a FIFO that is not faulted, with no FIFO at all, a child id
    that is not defined, 63, and 64 (past the child slots, resuming nothing,
    and whose 1 << 64 shifts as 1 << 0 on arm64: it clears the root's fault
    bit without resuming it); ten resumes of a healthy
    child report a stamp-wait timeout each, later ones only log. Wakes
    (0x1020) of 0, 64, 0xffffffff and a child that does not exist do
    nothing."""
    s = Scenario("device_lifecycle_resume_registers", doc=device_lifecycle_resume_registers.__doc__)
    s.w(abi.R_RESUME_ROOT, 0)
    s.w(abi.R_RESUME_CHILD, 0)
    s.w(abi.R_RESUME_CHILD, 5)
    s.events("resumes with no FIFO")
    f = s.bring_up()
    for v in (0, 64, 0xFFFFFFFF, 5, 63):
        s.w(abi.R_WAKE_CHILD, v)
    c = ChildFIFO(s, 1)
    c.setup_node()
    c.define()
    c.submit(99, fault=True)
    f.cmd(99)
    st = f.cmd(abi.OP_NOP)
    f.doorbell()
    s.wait_fault()
    s.wait(mmio=abi.R_FAULT_OFFSET, mmio_ne=0xFFFFFFFF)
    s.events("root and child faulted")
    s.fault_state("both faulted")
    s.w(abi.R_RESUME_ROOT, 0)
    f.wait_stamp(st)
    s.events("root resumed, child still faulted")
    s.fault_state("root resumed")
    s.w(abi.R_RESUME_CHILD, 1)
    s.events("child resumed")
    s.fault_state("child resumed")
    s.diff("after the resumes")
    s.w(abi.R_RESUME_ROOT, 0)
    for n in range(11):
        s.w(abi.R_RESUME_CHILD, 1)
    s.w(abi.R_RESUME_CHILD, 7)
    s.w(abi.R_RESUME_CHILD, 63)
    s.events("resumes of healthy FIFOs")
    s.fault_state("healthy")
    c.submit(abi.OP_NOP)
    s.submit(abi.OP_NOP)
    s.events("both still run")
    # 64: the root's fault bit cleared, the root left faulted.
    st2 = s.fifo.cmd(99)
    st3 = s.fifo.cmd(abi.OP_NOP)
    f.doorbell()
    s.wait_fault()
    s.events("root faulted")
    s.w(abi.R_RESUME_CHILD, 64)
    s.events("0x1028 = 64")
    s.fault_state("0x1028 = 64")
    f.wait_stamp(st3, timeout_ms=500, label="the root stays faulted")
    s.w(abi.R_RESUME_ROOT, 0)
    f.wait_stamp(st3)
    s.events("root resumed")
    s.fault_state("root resumed")
    regs(s, "end")
    return s


@scenario
def device_lifecycle_resume_child_order():
    """A child resume while the root is still faulted raises the interrupt
    again and 0x102c stays 1; 0x1028 = 0 then clears the root's fault bit
    (1 << 0) and resumes child FIFO slot 0, which is empty: 0x102c reads 0
    but the root stays faulted until 0x1024."""
    s = Scenario("device_lifecycle_resume_child_order", doc=device_lifecycle_resume_child_order.__doc__)
    f = s.bring_up()
    c = ChildFIFO(s, 1)
    c.setup_node()
    c.define()
    c.submit(99, fault=True)
    f.cmd(99)
    st = f.cmd(abi.OP_NOP)
    f.doorbell()
    s.wait(mmio=abi.R_FAULT_OFFSET, mmio_ne=0xFFFFFFFF)
    s.events("root and child faulted")
    s.w(abi.R_RESUME_CHILD, 1)
    s.events("child resumed first")
    s.fault_state("child resumed first")
    s.w(abi.R_RESUME_CHILD, 0)
    s.events("0x1028 = 0")
    s.fault_state("0x1028 = 0")
    f.wait_stamp(st, timeout_ms=500, label="the root stays faulted")
    s.w(abi.R_RESUME_ROOT, 0)
    f.wait_stamp(st)
    s.events("root resumed")
    s.fault_state("root resumed")
    s.diff("after")
    return s


@scenario
def device_lifecycle_resume_child_65_dies():
    """0x1028 = 65 names no child FIFO: it clears bit 1 of the fault mask
    and resumes nothing, and the device runs on. The name is kept from when
    the host died there."""
    s = Scenario("device_lifecycle_resume_child_65_dies", doc=device_lifecycle_resume_child_65_dies.__doc__)
    s.bring_up()
    s.submit(abi.OP_NOP)
    s.events("running")
    s.w(abi.R_RESUME_CHILD, 65)
    return s


@scenario
def device_lifecycle_child_start_fails():
    """A child FIFO whose ring cannot be mapped (its page list names a page
    past guest RAM) is still defined, but Fifo::start fails: nothing runs on
    it, and unpause's restart of it fails again with only a log. A child whose
    page list names one page 1100 times: in the blocks setup its 17.6 MB ring
    does not fit the root task's 16 MB range allocator, so it never runs; with
    a memory map it does. Both are deleted cleanly."""
    s = Scenario("device_lifecycle_child_start_fails", doc=device_lifecycle_child_start_fails.__doc__)
    s.bring_up()
    past = (s.ram[0][0] + s.ram[0][1]) >> 14
    bad = ChildFIFO(s, 1)
    bad.setup_node(page_list=[s.pfn(bad.pages[0]), past])
    bad.define()
    big = ChildFIFO(s, 2)
    big.setup_node(page_list=[s.pfn(big.pages[0])] * 1100)
    big.define()
    s.events("defined")
    s.diff("defined", gpa=s.pa(s.root_page), length=abi.PAGE)
    for c in (bad, big):
        st = c.cmd(abi.OP_NOP)
        c.doorbell()
        c.wait_stamp(st, timeout_ms=1000, label="child %d NOP" % c.fid)
    s.api("pause")
    s.api("unpause")
    s.events("paused and unpaused")
    bad.delete()
    big.delete()
    s.events("deleted")
    s.diff("after", gpa=s.pa(s.root_page), length=abi.PAGE)
    return s
