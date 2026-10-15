"""Host-side paths no other scenario reaches, and the public suspend,
resume, pause, stop and reset call orders no other scenario runs.

Every input is one the C API allows (reims_vgpu.h: every host callback is
optional) or the guest ABI. The runner's ops willSuspend, finishSuspend,
willResume, didResume, pause, unpause, stop, reset and cursorPosition call
the C API's rvg_device_will_suspend, rvg_device_finish_suspend,
rvg_device_will_resume, rvg_device_did_resume, rvg_device_pause,
rvg_device_unpause, rvg_device_stop, rvg_device_reset and
rvg_display_cursor_position.
The run states the device keeps (RunState): 0 Running, 1 EnteringSuspend,
2 Suspended, 3 LeavingSuspendEarly, 4 Resuming (DeviceCore::did_resume
until its rebuild ends), 5 Paused. DeviceCore::interrupt raises only in
state 0.

It also locks deaths a guest can reach: DeviceCore::add_wait's check that
the waiter's entry is idle (DeviceDeath::WaiterBusy), through
DeviceCore::finish_wait's 32-bit mask, and the guest display's check that
it is not paused (its CommandWhilePaused death) on eight guest commands,
through a root FIFO the guest restarts while the host has the device
paused.
"""
from pgtest import abi
from pgtest.abi import PAGE
from pgtest.display_guest import (DisplayGuest, display, p_ack, p_compositor, p_glyph, p_present3, p_props, p_show,
                                  p_sleep, p_swap)
from pgtest.fifo import ChildFIFO
from pgtest.fifo_util import child, track_stamps
from pgtest.limits_util import skip_to
from pgtest.registry import scenario
from pgtest.scenario import Scenario, h
from pgtest.suspend_guest import blob, state
from pgtest.task import Task

from scenarios.fifo_stamps import duplicate


# ---- the pending-stamp ring ----

@scenario
def fresh_dev_stamp_duplicate_fault_of_faulted_dies():
    """The fourth pair of the fifo_stamp_duplicate family: behind a faulted
    head (stamp 1) a second command with stamp 5 faults too and waits
    pending, still unpublished; a third command with stamp 5 then faults.
    StampQueue::fault_stamp finds the earlier twin already faulted and ends
    the process (fifo::DeathKind::FaultTwice): the host dies at the
    doorbell."""
    return duplicate("fresh_dev_stamp_duplicate_fault_of_faulted_dies", 99,
                     fresh_dev_stamp_duplicate_fault_of_faulted_dies.__doc__, third=99)


# ---- task and memory callbacks left absent (every one is optional in
# reims_vgpu.h; ALL_BLOCKS names them as the runner's config does) ----

ALL_BLOCKS = ("createTask", "destroyTask", "mapMemory", "unmapMemory", "readMemory")


def without(*missing):
    return {"blocks": {b: True for b in ALL_BLOCKS if b not in missing}}


@scenario(setups=("blocks",))
def fresh_dev_blocks_no_destroy_task_reset_dies():
    """A blocks-setup host whose RvgHostCallbacks leave destroy_task absent:
    bring-up and NOPs work (nothing destroys a task yet), then reset
    (DeviceCore::reset) writes root page 0, and DeviceCore::teardown_root,
    finding no destroy_task before it destroys the root task, ends the
    process (DeviceDeath::TeardownWithoutDestroyTask): the host dies at the
    reset. No user task is defined: the teardown would release it first,
    through DeviceCore::destroy_task, and the host would die there instead
    (DeviceDeath::NilBlock, as fresh_dev_blocks_no_destroy_task_delete_dies).
    Blocks setup only: the task and memory callbacks exist only there (a
    memory map with any of them is refused with RVG_ERR_MEMORY_CONFLICT,
    device_create_map_with_a_block)."""
    s = Scenario("fresh_dev_blocks_no_destroy_task_reset_dies", setups=("blocks",),
                 descriptor=without("destroyTask"), doc=fresh_dev_blocks_no_destroy_task_reset_dies.__doc__)
    s.bring_up()
    s.submit(abi.OP_NOP)
    s.submit(abi.OP_NOP)
    s.events("up")
    s.api("reset")
    return s


@scenario(setups=("blocks",))
def fresh_dev_blocks_no_destroy_task_delete_dies():
    """A blocks-setup host that leaves destroy_task absent, deleting a user
    task: DeviceCore::delete_task_id releases the task, whose release
    (LocalTask's drop) calls DeviceCore::destroy_task, which finds no
    destroy_task and ends the process (DeviceDeath::NilBlock): the host dies
    at the DeleteTask doorbell. Blocks setup only, as
    fresh_dev_blocks_no_destroy_task_reset_dies."""
    s = Scenario("fresh_dev_blocks_no_destroy_task_delete_dies", setups=("blocks",),
                 descriptor=without("destroyTask"), doc=fresh_dev_blocks_no_destroy_task_delete_dies.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=4)
    t.define()
    t.map()
    s.events("task up")
    t.delete(timeout_ms=10000, label="the host dies here")
    return s


@scenario(setups=("blocks",))
def fresh_dev_blocks_no_map_memory_dies():
    """A blocks-setup host that leaves map_memory absent: the device is
    created and its registers work, but the first root-page write
    (DeviceCore::set_root_page) creates the root task (create_task) and maps
    the root node into it, where DeviceCore::root_task_mapping finds no
    map_memory and ends the process (DeviceDeath::NilBlock): the host dies
    at that write. Blocks setup only, as
    fresh_dev_blocks_no_destroy_task_reset_dies."""
    s = Scenario("fresh_dev_blocks_no_map_memory_dies", setups=("blocks",),
                 descriptor=without("mapMemory"), doc=fresh_dev_blocks_no_map_memory_dies.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    s.sweep(0x1000, 0x1040, label="registers")
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    return s


@scenario(setups=("blocks",))
def fresh_dev_blocks_no_unmap_memory_dies():
    """A blocks-setup host that leaves unmap_memory absent: bring-up and a
    NOP work, since nothing unmaps; stop's root-page write of 0 then tears
    the root down, and the first of its root-task mappings released (the
    root FIFO's and the root node's) unmaps through MappingSink::unmap_range,
    which finds no unmap_memory and ends the process
    (DeviceDeath::NilBlock): the host dies at stop. No user task is
    defined: with one mapped, a root-task mapping of its page table is
    released at a moment timing decides, and the host dies there instead
    (measured: at the task's map or at a later command, varying by run).
    Blocks setup only, as
    fresh_dev_blocks_no_destroy_task_reset_dies."""
    s = Scenario("fresh_dev_blocks_no_unmap_memory_dies", setups=("blocks",),
                 descriptor=without("unmapMemory"), doc=fresh_dev_blocks_no_unmap_memory_dies.__doc__)
    s.bring_up()
    s.submit(abi.OP_NOP)
    s.submit(abi.OP_NOP)
    s.events("up")
    s.api("stop")
    return s


# ---- suspend, resume, pause, stop and reset orders ----

def guest(s, display_page=True):
    """A running guest: root FIFO, child FIFO 1, and (display_page) a
    connected, acknowledged display on port 0."""
    g = DisplayGuest(s, task0=False)
    c = ChildFIFO(s, 1)
    c.setup_node()
    c.define()
    if display_page:
        g.connect(0, enabled=0xF)
        g.ack(0, 1)
    s.submit(abi.OP_NOP)
    c.submit(abi.OP_NOP)
    s.events("running")
    return g, c


def regs(s, label):
    s.sweep(0x1000, 0x1040, label=label)


def suspend(s, slot="s", **kw):
    s.api("willSuspend", **kw)
    s.api("finishSuspend", slot=slot, **kw)


def restart_root(s):
    """The guest brings the root FIFO up again on its root page: 0x1000 is
    written 0 first, since the kept enable mark would make a write of 1 do
    nothing."""
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    s.w(abi.R_FIFO_ENABLE, 0)
    s.fifo.enable()
    s.fifo.written = 0


def scen(name, fn, displays=True, **kw):
    return Scenario(name, displays=[display(0)] if displays else None, doc=fn.__doc__, **kw)


@scenario
def fresh_dev_suspend_finish_twice_dies():
    """rvg_device_finish_suspend twice: the first sets run state 2, so the
    second fails DeviceCore::finish_suspend's check that the state is
    EnteringSuspend (DeviceDeath::SuspendNotBegun) and the host dies there. No
    display page, on purpose: without the store of 2 the second call would pass
    the check and, with no guest display to pause twice, return a blob; with a
    guest display it would die at GuestDisplay::pause's own check (its
    PauseTwice death) instead, at the same op
    (fresh_dev_suspend_second_suspend_dies)."""
    s = scen("fresh_dev_suspend_finish_twice_dies", fresh_dev_suspend_finish_twice_dies, displays=False)
    guest(s, display_page=False)
    suspend(s)
    s.events("suspended")
    s.api("finishSuspend", slot="t", label="again")
    return s


@scenario
def fresh_dev_suspend_will_twice():
    """rvg_device_will_suspend twice, then rvg_device_finish_suspend: the
    second rvg_device_will_suspend quiesces FIFOs already quiesced and changes
    nothing, so the blob is the one a single rvg_device_will_suspend writes,
    and it resumes on a fresh device that runs on."""
    s = scen("fresh_dev_suspend_will_twice", fresh_dev_suspend_will_twice)
    g, c = guest(s)
    s.api("willSuspend")
    s.api("willSuspend", label="again")
    s.events("after two willSuspends")
    s.api("finishSuspend", slot="s")
    s.api("recreate")
    s.api("willResume", slot="s")
    s.api("didResume")
    s.events("resumed")
    s.submit(abi.OP_NOP)
    c.submit(abi.OP_NOP)
    s.events("end")
    return s


@scenario
def fresh_dev_suspend_did_twice():
    """rvg_device_did_resume twice after one resume onto a fresh device: the
    second re-writes the root page, whose teardown finds the root FIFO the
    first rebuilt, tears it and child FIFO 1 down and clears the enable mark,
    so neither FIFO is rebuilt: 0x1000 reads 0 and neither NOP is answered, as
    in suspend_in_place. The display page is restored again and one more
    interrupt raised."""
    s = scen("fresh_dev_suspend_did_twice", fresh_dev_suspend_did_twice)
    g, c = guest(s)
    suspend(s)
    s.api("recreate")
    s.api("willResume", slot="s")
    s.api("didResume")
    s.events("resumed")
    regs(s, "after one didResume")
    s.api("didResume", label="again")
    s.events("after the second didResume")
    regs(s, "after two")
    g.diff_page(0, "page")
    s.submit(abi.OP_NOP, timeout_ms=500, label="a NOP")
    c.submit(abi.OP_NOP, timeout_ms=500, label="a child NOP")
    s.events("end")
    return s


@scenario
def fresh_dev_suspend_did_without_will():
    """rvg_device_did_resume on a running device with no suspend at all: its
    root-page write tears the running root FIFO and child FIFO 1 down and
    clears the enable mark, so no FIFO is rebuilt (0x1000 reads 0, neither NOP
    is answered); the display page is restored and interrupts are raised."""
    s = scen("fresh_dev_suspend_did_without_will", fresh_dev_suspend_did_without_will)
    g, c = guest(s)
    s.api("didResume")
    s.events("after didResume")
    regs(s, "registers")
    g.diff_page(0, "page")
    s.submit(abi.OP_NOP, timeout_ms=500, label="a NOP")
    c.submit(abi.OP_NOP, timeout_ms=500, label="a child NOP")
    s.events("end")
    return s


@scenario
def fresh_dev_suspend_did_on_fresh_device():
    """rvg_device_did_resume on a device that was never versioned or given a
    root page: root page 0 builds nothing, and it raises its one interrupt. The
    guest then brings the device up as usual."""
    s = scen("fresh_dev_suspend_did_on_fresh_device", fresh_dev_suspend_did_on_fresh_device)
    s.api("didResume")
    s.events("after didResume")
    regs(s, "registers")
    s.bring_up()
    s.submit(abi.OP_NOP)
    s.events("end")
    return s


@scenario
def fresh_dev_suspend_will_then_pause():
    """rvg_device_will_suspend, then pause instead of
    rvg_device_finish_suspend: pause quiesces and drains the already quiesced
    FIFOs, pauses the guest display and sets run state 5; unpause restarts the
    FIFOs: the suspend is abandoned and the guest runs on."""
    s = scen("fresh_dev_suspend_will_then_pause", fresh_dev_suspend_will_then_pause)
    g, c = guest(s)
    s.api("willSuspend")
    s.api("pause")
    s.events("paused")
    s.api("unpause")
    s.events("unpaused")
    s.submit(abi.OP_NOP)
    c.submit(abi.OP_NOP)
    s.api("cursorPosition", display=0)
    s.events("end")
    return s


@scenario
def fresh_dev_suspend_pause_then_finish_dies():
    """pause, then rvg_device_will_suspend is skipped and
    rvg_device_finish_suspend called: run state 5 fails
    DeviceCore::finish_suspend's check that the state is EnteringSuspend
    (DeviceDeath::SuspendNotBegun): the host dies at
    rvg_device_finish_suspend."""
    s = scen("fresh_dev_suspend_pause_then_finish_dies", fresh_dev_suspend_pause_then_finish_dies)
    guest(s)
    s.api("pause")
    s.events("paused")
    s.api("finishSuspend")
    return s


@scenario
def fresh_dev_suspend_will_pause_finish_dies():
    """rvg_device_will_suspend, pause, rvg_device_finish_suspend: pause
    overwrote run state 1 with 5, so rvg_device_finish_suspend's check fails
    (DeviceDeath::SuspendNotBegun) and the host dies."""
    s = scen("fresh_dev_suspend_will_pause_finish_dies", fresh_dev_suspend_will_pause_finish_dies)
    guest(s)
    s.api("willSuspend")
    s.api("pause")
    s.events("paused")
    s.api("finishSuspend")
    return s


def will_then(name, fn, call):
    s = scen(name, fn)
    g, c = guest(s)
    s.api("willSuspend")
    s.api(call)
    s.events("after " + call)
    regs(s, "registers after " + call)
    g.diff_page(0, "page after " + call)
    s.api("finishSuspend", slot="s")
    s.events("finished")
    return s


@scenario
def fresh_dev_suspend_will_then_stop():
    """rvg_device_will_suspend, stop, rvg_device_finish_suspend: stop writes
    root page 0 and tears the quiesced FIFOs and the display page down while
    the run state stays 1; rvg_device_finish_suspend then serializes a device
    with no root, and its blob says so."""
    return will_then("fresh_dev_suspend_will_then_stop", fresh_dev_suspend_will_then_stop, "stop")


@scenario
def fresh_dev_suspend_will_then_reset():
    """rvg_device_will_suspend, reset, rvg_device_finish_suspend: reset also
    zeroes the binary version, the EFI display and the device-info max key,
    which the blob then carries."""
    return will_then("fresh_dev_suspend_will_then_reset", fresh_dev_suspend_will_then_reset, "reset")


@scenario
def fresh_dev_suspend_finish_then_pause_dies():
    """rvg_device_finish_suspend pauses the guest display; pause then pauses it
    again, and GuestDisplay::pause, finding it paused already, ends the process
    (its PauseTwice death): the host dies at pause."""
    s = scen("fresh_dev_suspend_finish_then_pause_dies", fresh_dev_suspend_finish_then_pause_dies)
    guest(s)
    suspend(s)
    s.events("suspended")
    s.api("pause")
    return s


@scenario
def fresh_dev_suspend_finish_then_pause_no_display():
    """With no display page, pause and unpause after rvg_device_finish_suspend
    find no root FIFO and no guest display: nothing dies, unpause sets run
    state 0 and raises one interrupt, but no FIFO runs (0x1000 still reads 1)
    until the guest writes 0x1000 = 0 and enables the root FIFO again; its NOP
    then completes with an interrupt."""
    s = scen("fresh_dev_suspend_finish_then_pause_no_display", fresh_dev_suspend_finish_then_pause_no_display,
             displays=False)
    g, c = guest(s, display_page=False)
    suspend(s)
    s.events("suspended")
    s.api("pause")
    s.events("paused")
    s.api("unpause")
    s.events("unpaused")
    regs(s, "registers")
    s.submit(abi.OP_NOP, timeout_ms=500, label="a NOP")
    restart_root(s)
    s.submit(abi.OP_NOP)
    s.events("the guest restarts the root")
    return s


def finish_then(name, fn, call):
    s = scen(name, fn)
    g, c = guest(s)
    suspend(s)
    s.events("suspended")
    s.api(call)
    s.events("after " + call)
    regs(s, "registers after " + call)
    g.diff_page(0, "page after " + call)
    restart_root(s)
    s.submit(abi.OP_NOP)
    s.events("the guest restarts the root")
    regs(s, "registers after the restart")
    return s


@scenario
def fresh_dev_suspend_finish_then_stop():
    """rvg_device_finish_suspend, then stop: the root and the display page are
    torn down with the guest display still paused by rvg_device_finish_suspend,
    and nothing dies. The run state stays 2, so when the guest brings the root
    FIFO up again its NOP completes and the stamp lands (0x1018), but no
    interrupt is raised (an interrupt is raised only in state 0)."""
    return finish_then("fresh_dev_suspend_finish_then_stop", fresh_dev_suspend_finish_then_stop, "stop")


@scenario
def fresh_dev_suspend_finish_then_reset():
    """rvg_device_finish_suspend, then reset: as
    fresh_dev_suspend_finish_then_stop, and the binary version, EFI display and
    device-info max key are zeroed too; the guest restarting the root after it
    writes no version, and its NOP still completes, with no interrupt."""
    return finish_then("fresh_dev_suspend_finish_then_reset", fresh_dev_suspend_finish_then_reset, "reset")


@scenario
def fresh_dev_suspend_second_suspend_dies():
    """A second suspend without a resume, with a display page: the second
    rvg_device_will_suspend finds no root FIFO and no children
    (rvg_device_finish_suspend released them) and passes, but the second
    rvg_device_finish_suspend pauses the guest display the first one paused,
    and GuestDisplay::pause ends the process (its PauseTwice death): the host
    dies at the second rvg_device_finish_suspend."""
    s = scen("fresh_dev_suspend_second_suspend_dies", fresh_dev_suspend_second_suspend_dies)
    guest(s)
    suspend(s, slot="a")
    s.events("first")
    suspend(s, slot="b", label="again")
    return s


@scenario
def fresh_dev_suspend_second_suspend_resume_dies():
    """A second rvg_device_will_suspend and rvg_device_finish_suspend without a
    resume, and no display page (so no guest display dies): the second blob is
    written after the first rvg_device_finish_suspend released the root FIFO,
    so keys 6-8 and 19 (written, read, start, fault offset), which read 0 with
    no root FIFO (DeviceCore::serialize_suspend_state), are 0, while key 4
    still carries the enable mark. rvg_device_will_resume accepts it on a fresh
    device; in rvg_device_did_resume, DeviceCore::resume_root_fifo's
    DeviceCore::create_root_fifo refuses start 0 and makes no FIFO, and the
    resume ends the process (DeviceDeath::ResumedRootFifoMissing): the host
    dies at rvg_device_did_resume."""
    s = scen("fresh_dev_suspend_second_suspend_resume_dies", fresh_dev_suspend_second_suspend_resume_dies,
             displays=False)
    guest(s, display_page=False)
    suspend(s, slot="a")
    s.events("first")
    suspend(s, slot="b", label="again")
    s.events("second")
    s.api("recreate")
    s.api("willResume", slot="b")
    s.api("didResume")
    return s


def resumed_pending(name, fn, **kw):
    """A guest suspended and a fresh device given its blob:
    rvg_device_will_resume has succeeded (run state 3) and
    rvg_device_did_resume has not been called."""
    s = scen(name, fn, **kw)
    g, c = guest(s)
    suspend(s)
    s.api("recreate")
    s.api("willResume", slot="s")
    return s, g, c


@scenario
def fresh_dev_suspend_mmio_before_did():
    """Between a successful rvg_device_will_resume and rvg_device_did_resume
    the registers read the blob's values (0x1000 1, 0x101c the root page,
    0x1034 the version), though nothing is rebuilt and nothing raises. A
    root-FIFO doorbell then only stores the written offset, since there is no
    root FIFO to poke; rvg_device_did_resume rebuilds the FIFO at the blob's
    offsets and its NOP is not run then (0x100c still reads the old offset); it
    runs, and its stamp publishes, with the next doorbell."""
    s, g, c = resumed_pending("fresh_dev_suspend_mmio_before_did", fresh_dev_suspend_mmio_before_did)
    regs(s, "after willResume")
    s.r(abi.R_VERSION, "version")
    s.events("nothing yet")
    st = s.fifo.cmd(abi.OP_NOP)
    s.fifo.doorbell()
    s.quiesce(100)
    s.events("doorbell before didResume")
    s.api("didResume")
    s.fifo.wait_stamp(st, timeout_ms=500, label="the NOP")
    s.events("resumed")
    regs(s, "after didResume")
    s.submit(abi.OP_NOP)
    c.submit(abi.OP_NOP)
    s.events("end")
    return s


@scenario
def fresh_dev_suspend_root_page_before_did_dies():
    """A root-page write between a successful rvg_device_will_resume and
    rvg_device_did_resume: the run state is still LeavingSuspendEarly, so
    DeviceCore::new_contiguous_mapping, mapping the root node, ends the process
    (DeviceDeath::MappingWhileLeavingSuspend) and the host dies, as after a
    refused rvg_device_will_resume (suspend_failed_resume_then_map)."""
    s, g, c = resumed_pending("fresh_dev_suspend_root_page_before_did_dies",
                              fresh_dev_suspend_root_page_before_did_dies)
    s.events("resume pending")
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    return s


@scenario
def fresh_dev_suspend_display_doorbell_before_did():
    """A display doorbell between a successful rvg_device_will_resume and
    rvg_device_did_resume: the recreated display has no page yet, so it rings
    nothing; rvg_device_did_resume then restores the page as usual."""
    s, g, c = resumed_pending("fresh_dev_suspend_display_doorbell_before_did",
                              fresh_dev_suspend_display_doorbell_before_did)
    s.w(abi.R_DISPLAY_DOORBELL, 0)
    s.quiesce(100)
    s.events("doorbell before didResume")
    s.api("didResume")
    s.events("resumed")
    g.diff_page(0, "page")
    s.submit(abi.OP_NOP)
    s.events("end")
    return s


@scenario
def fresh_dev_suspend_resume_then_unpause():
    """unpause in place of rvg_device_did_resume after a successful
    rvg_device_will_resume (the start Virtualization.framework takes for a
    paused rather than a restored machine): run state 0 and one interrupt, but
    nothing is rebuilt, so the root FIFO never answers though 0x1000 reads 1. A
    root-page write maps the root again but builds no FIFO; the guest's
    0x1000 = 0, then its usual enable, does, and a NOP runs."""
    s, g, c = resumed_pending("fresh_dev_suspend_resume_then_unpause", fresh_dev_suspend_resume_then_unpause)
    s.api("unpause")
    s.events("unpaused")
    regs(s, "registers")
    s.submit(abi.OP_NOP, timeout_ms=500, label="a NOP")
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    s.events("root page again")
    regs(s, "registers after the root page")
    restart_root(s)
    s.submit(abi.OP_NOP)
    s.events("root FIFO restarted")
    return s


@scenario
def fresh_dev_suspend_resume_running():
    """rvg_device_will_resume and rvg_device_did_resume on a running device
    (itself resumed from the blob it is given again): rvg_device_will_resume
    accepts it and overwrites the kept registers (0x100c reads the blob's read
    offset), and rvg_device_did_resume's root-page write tears the running
    FIFOs down and clears the enable mark, so no FIFO is rebuilt and neither
    NOP is answered."""
    s = scen("fresh_dev_suspend_resume_running", fresh_dev_suspend_resume_running)
    g, c = guest(s)
    suspend(s)
    s.api("recreate")
    s.api("willResume", slot="s")
    s.api("didResume")
    s.events("resumed")
    s.submit(abi.OP_NOP)
    s.events("running again")
    s.api("willResume", slot="s", label="on the running device")
    regs(s, "after willResume")
    s.api("didResume", label="on the running device")
    s.events("after the second resume")
    regs(s, "registers")
    g.diff_page(0, "page")
    s.submit(abi.OP_NOP, timeout_ms=500, label="a NOP")
    c.submit(abi.OP_NOP, timeout_ms=500, label="a child NOP")
    s.events("end")
    return s


@scenario
def fresh_dev_suspend_will_then_resume():
    """rvg_device_will_suspend, then rvg_device_will_resume and
    rvg_device_did_resume with an earlier blob and no
    rvg_device_finish_suspend: rvg_device_will_resume accepts the blob on the
    quiesced device, and rvg_device_did_resume's root-page write tears the
    quiesced root FIFO and child FIFO down and clears the enable mark, so
    neither is rebuilt and neither NOP is answered."""
    s = scen("fresh_dev_suspend_will_then_resume", fresh_dev_suspend_will_then_resume)
    g, c = guest(s)
    suspend(s)
    s.api("recreate")
    s.api("willResume", slot="s")
    s.api("didResume")
    s.submit(abi.OP_NOP)
    s.events("resumed once")
    s.api("willSuspend")
    s.api("willResume", slot="s", label="after willSuspend")
    s.api("didResume", label="after willSuspend")
    s.events("resumed without finishSuspend")
    regs(s, "registers")
    s.submit(abi.OP_NOP, timeout_ms=500, label="a NOP")
    c.submit(abi.OP_NOP, timeout_ms=500, label="a child NOP")
    s.events("end")
    return s


@scenario
def fresh_dev_suspend_pause_then_resume():
    """A paused device given a blob: rvg_device_will_resume and
    rvg_device_did_resume on it. pause left the FIFOs quiesced and the guest
    display paused; rvg_device_did_resume's root-page write tears the FIFOs
    down and clears the enable mark, so no FIFO is rebuilt and the NOP is not
    answered; the display is restored, and the guest display is no longer
    paused (rvg_display_cursor_position, whose GuestDisplay::cursor_position
    dies on a paused guest display, reads it)."""
    s = scen("fresh_dev_suspend_pause_then_resume", fresh_dev_suspend_pause_then_resume)
    g, c = guest(s)
    suspend(s)
    s.api("recreate")
    s.api("willResume", slot="s")
    s.api("didResume")
    s.submit(abi.OP_NOP)
    s.events("resumed once")
    s.api("pause")
    s.api("willResume", slot="s", label="paused")
    s.api("didResume", label="paused")
    s.events("resumed from paused")
    regs(s, "registers")
    g.diff_page(0, "page")
    s.submit(abi.OP_NOP, timeout_ms=500, label="a NOP")
    s.api("cursorPosition", display=0)
    s.events("end")
    return s


@scenario
def fresh_dev_suspend_did_after_failed_will():
    """A refused rvg_device_will_resume (port count 3, InvalidDisplayPortCount)
    leaves run state 3; rvg_device_did_resume then sets state 4 before its
    root-page write, so no mapping dies (compare
    suspend_failed_resume_then_map). The write tears the running FIFOs down and
    clears the enable mark, so no FIFO is rebuilt and neither NOP is answered;
    the display page is restored."""
    s = scen("fresh_dev_suspend_did_after_failed_will", fresh_dev_suspend_did_after_failed_will)
    g, c = guest(s)
    s.api("willResume", hex=blob(state(ports=3)).hex(), label="port count 3")
    s.api("didResume")
    s.events("after didResume")
    regs(s, "registers")
    g.diff_page(0, "page")
    s.submit(abi.OP_NOP, timeout_ms=500, label="a NOP")
    c.submit(abi.OP_NOP, timeout_ms=500, label="a child NOP")
    s.events("end")
    return s


@scenario
def fresh_dev_lifecycle_stop_reset_orders():
    """stop and reset in both orders, then pause, stop while paused, and
    unpause: stop and reset are root-page writes of 0 (reset also zeroes
    the version, EFI and the device-info max key), and a second of either
    changes nothing once the root is gone (every register reads 0). The
    guest brings the device up again; stop while paused tears its root and
    display page down without dying, unpause raises one interrupt, and
    rvg_display_cursor_position then reads (0xffff, 0xffff)."""
    s = scen("fresh_dev_lifecycle_stop_reset_orders", fresh_dev_lifecycle_stop_reset_orders)
    g, c = guest(s)
    s.api("stop")
    s.api("reset")
    s.events("stop, reset")
    regs(s, "after stop, reset")
    s.api("reset", label="again")
    s.api("stop", label="again")
    regs(s, "after reset, stop")
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    s.fifo.enable()
    s.fifo.written = 0
    s.submit(abi.OP_NOP)
    g.connect(0, enabled=0xF)
    s.events("up again")
    s.api("pause")
    s.api("stop", label="paused")
    s.events("stopped while paused")
    s.api("unpause")
    s.events("unpaused")
    regs(s, "registers")
    s.api("cursorPosition", display=0)
    return s


@scenario
def fresh_dev_suspend_will_then_did():
    """rvg_device_will_suspend, then rvg_device_did_resume with no
    rvg_device_finish_suspend and no rvg_device_will_resume: the suspend is
    abandoned by a resume of the live device. rvg_device_did_resume's root-page
    write tears the quiesced FIFOs down, and the run state goes to 0."""
    s = scen("fresh_dev_suspend_will_then_did", fresh_dev_suspend_will_then_did)
    g, c = guest(s)
    s.api("willSuspend")
    s.api("didResume")
    s.events("after didResume")
    regs(s, "registers")
    g.diff_page(0, "page")
    s.submit(abi.OP_NOP, timeout_ms=500, label="a NOP")
    s.api("cursorPosition", display=0)
    restart_root(s)
    s.submit(abi.OP_NOP)
    s.events("the guest restarts the root")
    return s


@scenario
def fresh_dev_suspend_finish_then_did_child_dies():
    """rvg_device_finish_suspend, then rvg_device_did_resume on the same device
    with no rvg_device_will_resume. Its root-page write finds the enable mark
    rvg_device_finish_suspend left, clears it in the teardown, so no root FIFO
    is rebuilt (0x1000 reads 0, a NOP is not answered) and the stamp memory
    rvg_device_finish_suspend cleared stays null; child FIFO 1, whose node is
    still marked running, is rebuilt, and the display restore ends the guest
    display's pause (rvg_display_cursor_position reads it). The child's NOP
    then signals its stamp through the null stamp memory: the host dies in the
    wait after the child's doorbell. The same death as
    interplay_life_resume_child_without_root_fifo_dies, reached in place with
    no blob."""
    s = scen("fresh_dev_suspend_finish_then_did_child_dies", fresh_dev_suspend_finish_then_did_child_dies)
    g, c = guest(s)
    suspend(s)
    s.api("didResume")
    s.events("after didResume")
    regs(s, "registers")
    g.diff_page(0, "page")
    s.submit(abi.OP_NOP, timeout_ms=500, label="a NOP")
    s.api("cursorPosition", display=0)
    s.events("before the child")
    c.submit(abi.OP_NOP, timeout_ms=10000, label="the host dies here")
    return s


@scenario
def fresh_dev_suspend_will_resume_twice():
    """Two successful rvg_device_will_resume calls with the same blob on a
    fresh device, then rvg_device_did_resume: the second overwrites the first's
    registers with the same values, and the rebuild is the one a single
    rvg_device_will_resume gives."""
    s = scen("fresh_dev_suspend_will_resume_twice", fresh_dev_suspend_will_resume_twice)
    g, c = guest(s)
    suspend(s)
    s.api("recreate")
    s.api("willResume", slot="s")
    s.api("willResume", slot="s", label="again")
    s.api("didResume")
    s.events("resumed")
    regs(s, "registers")
    s.submit(abi.OP_NOP)
    c.submit(abi.OP_NOP)
    s.events("end")
    return s


@scenario
def fresh_dev_suspend_reset_before_did():
    """reset between a successful rvg_device_will_resume and
    rvg_device_did_resume: reset's root-page write of 0 maps nothing, so the
    LeavingSuspendEarly checks (DeviceDeath::MappingWhileLeavingSuspend) are
    not reached; it replaces the root page the blob set with 0 and zeroes the
    binary version, the EFI display and the device-info max key, but keeps the
    blob's FIFO enable mark. rvg_device_did_resume then writes root page 0 and
    rebuilds nothing: 0x1000 reads 1, and neither NOP is answered."""
    s, g, c = resumed_pending("fresh_dev_suspend_reset_before_did", fresh_dev_suspend_reset_before_did)
    s.api("reset")
    regs(s, "after reset")
    s.api("didResume")
    s.events("resumed")
    regs(s, "registers")
    g.diff_page(0, "page")
    s.submit(abi.OP_NOP, timeout_ms=500, label="a NOP")
    c.submit(abi.OP_NOP, timeout_ms=500, label="a child NOP")
    s.events("end")
    return s



# ---- the waiting-info mask aliased by a delete ----

SLOW_PAGES = 512  # an 8 MiB ring naming one page 512 times, as interplay_limits


@scenario
def fresh_dev_wait_alias_delete_dies():
    """DeviceCore::finish_wait clears its waiter's bit with a 32-bit mask
    (num::bit32, the shift taken mod 32), so deleting child 33 (opcode 49,
    DeviceCore::delete_fifo) clears bit 1 of the waiting word
    (waiting_stamps) while child 1 waits on a barrier, leaving child 1's
    waiting entry (its next index) set. Child 2, sharing stamp index 1
    (nothing checks uniqueness), then waits on a barrier:
    DeviceCore::add_wait's mask test passes (bit 1 is clear), and its check
    that the waiter's entry is idle ends the process
    (DeviceDeath::WaiterBusy): the host dies.
    Child 1 and child 2 wait on child 3's first stamp, which never comes (a
    wait on the root would be released by the delete, itself a root
    command). Child 2 reaches its wait only after an 8 MiB NOP, so the death
    lands in the wait after its doorbell. Compare
    interplay_limits_child_wait_alias_dies, the same aliasing in
    DeviceCore::add_wait itself."""
    s = Scenario("fresh_dev_wait_alias_delete_dies", watchdog_s=15, doc=fresh_dev_wait_alias_delete_dies.__doc__)
    s.bring_up()
    track_stamps(s, 34)
    c1 = child(s, 1)
    c2 = ChildFIFO(s, 2, stamp_index=1)
    c2.setup_node(page_list=[s.pfn(c2.pages[0])] * SLOW_PAGES)
    c2.pages = [c2.pages[0]] * SLOW_PAGES
    c2.define()
    s.fifos[2] = c2
    child(s, 3)
    c33 = child(s, 33)
    s.events("children defined")
    a = c1.cmd(abi.OP_NOP, barriers=[(3, 1)])
    c1.doorbell()
    c1.wait_stamp(a, timeout_ms=300, label="child 1 waits on child 3")
    c33.delete()
    s.events("child 33 deleted, child 1 still waiting")
    skip_to(c2, SLOW_PAGES * PAGE - 0x100)
    b = c2.cmd(abi.OP_NOP, barriers=[(3, 1)])
    c2.doorbell()
    c2.wait_stamp(b, timeout_ms=5000, label="child 2 waits under index 1: the host dies")
    return s


# ---- a display command on a root FIFO restarted while paused ----

def paused_nub(name, fn, opcode, payload, after="pause"):
    """A running guest with port 0's guest display, paused by `after`
    (pause, or a whole suspend); the guest then restarts the root FIFO by
    writing 0x1000 = 0 and 1 (nothing checks the run state, and no root-page
    write tears the guest display down), a NOP runs on it, and `opcode` for
    port 0 reaches the paused guest display."""
    s = scen(name, fn)
    g, c = guest(s)
    if after == "pause":
        s.api("pause")
    else:
        suspend(s)
    s.events("paused")
    s.w(abi.R_FIFO_ENABLE, 0)
    s.fifo.written = 0
    s.w(abi.R_FIFO_ENABLE, 1)
    s.submit(abi.OP_NOP, label="a NOP on the restarted root FIFO")
    s.events("the root FIFO runs, paused")
    s.r(abi.R_IRQ_STAMP)
    # The doorbell is quiet: the guest display can end the process before the
    # write's record is written, so the death's op would depend on which
    # thread won.
    st = s.fifo.cmd(opcode, payload)
    s.op(op="mmio_w", off=h(abi.R_FIFO_WRITTEN), val=h(s.fifo.written & 0xFFFFFFFF), quiet=True)
    s.fifo.wait_stamp(st, timeout_ms=10000, label="the host dies here")
    return s


def paused_doc(what):
    return ("""While the host has the device paused, the guest restarts the root
    FIFO (0x1000 = 0, then 1): pause drained the old one and paused the
    guest displays, but the new FIFO runs, its NOP's stamp lands (0x1018)
    with no interrupt (run state 5). %s then reaches the paused guest
    display, whose first check, that it is not paused, ends the process
    (its CommandWhilePaused death): the host dies in the wait after the
    doorbell.""" % what)


@scenario
def fresh_dev_paused_nub_ack_dies():
    return paused_nub("fresh_dev_paused_nub_ack_dies", fresh_dev_paused_nub_ack_dies,
                      abi.OP_DISPLAY_ACK, p_ack(0, 1))


fresh_dev_paused_nub_ack_dies.__doc__ = paused_doc("Ack (2), GuestDisplay::ack,")


@scenario
def fresh_dev_paused_nub_cursor_glyph_dies():
    return paused_nub("fresh_dev_paused_nub_cursor_glyph_dies", fresh_dev_paused_nub_cursor_glyph_dies,
                      abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(0, 0, PAGE, 16, 4, 4))


fresh_dev_paused_nub_cursor_glyph_dies.__doc__ = paused_doc(
    "CursorGlyph (4), GuestDisplay::cursor_glyph, which checks before it reads the glyph,")


@scenario
def fresh_dev_paused_nub_cursor_show_dies():
    return paused_nub("fresh_dev_paused_nub_cursor_show_dies", fresh_dev_paused_nub_cursor_show_dies,
                      abi.OP_DISPLAY_CURSOR_SHOW, p_show(0, 1))


fresh_dev_paused_nub_cursor_show_dies.__doc__ = paused_doc("CursorShow (5), GuestDisplay::cursor_show,")


@scenario
def fresh_dev_paused_nub_present_dies():
    return paused_nub("fresh_dev_paused_nub_present_dies", fresh_dev_paused_nub_present_dies,
                      abi.OP_DISPLAY_TRANSACTION3, p_present3(0, 1))


fresh_dev_paused_nub_present_dies.__doc__ = paused_doc(
    "DisplayTransaction3 (7), GuestDisplay::present_surface, which checks before it looks up the surface,")


@scenario
def fresh_dev_paused_nub_swap_mapping_dies():
    return paused_nub("fresh_dev_paused_nub_swap_mapping_dies", fresh_dev_paused_nub_swap_mapping_dies,
                      abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1))


fresh_dev_paused_nub_swap_mapping_dies.__doc__ = paused_doc(
    "DisplaySwapMapping (8), GuestDisplay::present_mapped_surface, which checks before it looks up the "
    "mapping (no IOSurface host device is needed),")


@scenario
def fresh_dev_paused_nub_sleep_dies():
    return paused_nub("fresh_dev_paused_nub_sleep_dies", fresh_dev_paused_nub_sleep_dies,
                      abi.OP_DISPLAY_SLEEP_STATE, p_sleep(0, 1))


fresh_dev_paused_nub_sleep_dies.__doc__ = paused_doc("SleepState (9), GuestDisplay::sleep_state,")


@scenario
def fresh_dev_paused_nub_set_properties_dies():
    return paused_nub("fresh_dev_paused_nub_set_properties_dies", fresh_dev_paused_nub_set_properties_dies,
                      abi.OP_DISPLAY_SET_PROPERTIES, p_props(0, [(0, [0])]))


fresh_dev_paused_nub_set_properties_dies.__doc__ = paused_doc(
    "SetProperties (10) with one property, GuestDisplay::set_property,")


@scenario
def fresh_dev_paused_nub_compositor_dies():
    return paused_nub("fresh_dev_paused_nub_compositor_dies", fresh_dev_paused_nub_compositor_dies,
                      abi.OP_DISPLAY_COMPOSITOR_PARAMETERS, p_compositor(0, 10, 20, 5, 2))


fresh_dev_paused_nub_compositor_dies.__doc__ = paused_doc(
    "CompositorParameters (11), GuestDisplay::compositor_parameters,")


@scenario
def fresh_dev_suspended_nub_ack_dies():
    return paused_nub("fresh_dev_suspended_nub_ack_dies", fresh_dev_suspended_nub_ack_dies,
                      abi.OP_DISPLAY_ACK, p_ack(0, 1), after="suspend")


fresh_dev_suspended_nub_ack_dies.__doc__ = paused_doc("Ack (2), GuestDisplay::ack,").replace(
    "While the host has the device paused", "After willSuspend and finishSuspend, which pause the guest "
    "displays and leave the enable mark at 1").replace(
    "pause drained the old one and paused the\n    guest displays", "finishSuspend released the old one").replace(
    "(run state 5)", "(run state 2)")
