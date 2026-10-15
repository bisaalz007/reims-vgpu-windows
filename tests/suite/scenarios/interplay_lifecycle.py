"""Host lifecycle calls in the middle of guest work (docs/testing.md,
sections 4.7, 6.24, 7.3 and 7.4): pause and unpause, the four suspend
calls, reset, stop, recreate, release and display release, landing while
commands are queued, blocked on a barrier, held behind a fault or a Delay,
and the lifetimes of objects, tasks, FIFOs and mappings while work that
names them waits.

Where a round trip is meant to be transparent, a scenario runs the same
two-part workload (pgtest/lifecycle_util.py) on region A uninterrupted and
on region B interrupted, and compares the two.

On the device side, pause quiesces and drains the FIFOs and pauses the guest
displays; rvg_device_will_suspend sets run state 1 and quiesces;
rvg_device_finish_suspend drains, flushes and tears down the tasks and
serializes. On the FIFO side, a barrier wait that a quiesce wakes abandons the
command without advancing, and the pending-stamp queue's pump publishes a
faulted head as signalled once quiesced, while the stamp queue's fault latch is
cleared only by StampQueue::resume_from_fault on a faulted head.
"""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.fifo import ChildFIFO
from pgtest.lifecycle_util import Rig, node, resume_fresh, stamps, suspend
from pgtest.registry import scenario
from pgtest.scenario import Scenario


# ---- transparency: the same work with and without the interruption ----

@scenario
def interplay_life_pause_queued():
    """Region B's second part is rung while the device is paused: it does
    not run (its stamp wait times out), unpause runs it, and B ends equal
    to the uninterrupted A. Paused, the stamp and the registers are
    unchanged; unpause raises its own interrupt and the exec's."""
    s = Scenario("interplay_life_pause_queued", doc=interplay_life_pause_queued.__doc__)
    g = Rig(s)
    g.control()
    g.run("B", 0)
    s.events("A done, B's first part")
    s.api("pause")
    st = g.queue("B", 1)
    g.f.wait_stamp(st, timeout_ms=300, label="B's second part while paused")
    s.r(abi.R_FIFO_READ, "paused")
    s.events("paused, rung")
    s.api("unpause")
    g.f.wait_stamp(st)
    s.events("unpaused")
    g.compare()
    return s


@scenario
def interplay_life_pause_blocked():
    """Region B's second part waits on the gate's barrier when the device
    pauses: the quiesce wakes the wait, which abandons the command without
    advancing (0x100c stays before it); unpause parses it again and it
    waits again; the gate's NOP releases it, and B ends equal to A."""
    s = Scenario("interplay_life_pause_blocked", doc=interplay_life_pause_blocked.__doc__)
    g = Rig(s)
    g.control()
    g.run("B", 0)
    st = g.queue("B", 1, barriers=g.gate_barrier())
    g.f.wait_stamp(st, timeout_ms=300, label="B's second part blocked on the gate")
    s.r(abi.R_FIFO_READ, "blocked")
    s.events("blocked")
    s.api("pause")
    s.r(abi.R_FIFO_READ, "paused")
    s.events("paused")
    s.api("unpause")
    g.f.wait_stamp(st, timeout_ms=300, label="still blocked after unpause")
    s.events("unpaused")
    g.open_gate()
    g.f.wait_stamp(st)
    s.events("released")
    s.r(abi.R_FIFO_READ, "released")
    g.compare()
    return s


@scenario
def interplay_life_suspend_between_parts():
    """Suspend after region B's first part and resume on a fresh device
    before its second: the guest rings part two without redoing anything
    (the task, its object list and the gate come back from the root page).
    With the private buffer passed with flags 0, in-process
    (blocks, map) rvg_device_finish_suspend does not write it back: the fresh
    device makes it again from its guest pages (0x77 bytes), and B's
    destination differs from A's where the private copy lands (from +0x200).
    Under isolation the service keeps a private buffer in its guest pages (A's
    and B's pages hold the private contents all along) and B ends as A. Passed
    with RES_INVALIDATE_GUEST (_dirty), in-process rvg_device_finish_suspend
    writes both regions' private buffers back, and B ends as A in every
    setup."""
    out = []
    for suffix, flags in (("", 0), ("_dirty", abi.RES_INVALIDATE_GUEST)):
        name = "interplay_life_suspend_between_parts" + suffix
        s = Scenario(name, doc=interplay_life_suspend_between_parts.__doc__)
        g = Rig(s, priv_flags=flags)
        g.control()
        g.run("B", 0)
        s.events("before")
        suspend(s)
        g.priv_guest("suspended")
        resume_fresh(s)
        s.events("resumed")
        s.sweep(0x1000, 0x1040, label="registers after the resume")
        g.run("B", 1)
        g.open_gate()
        s.events("B's second part, the gate")
        g.compare()
        g.priv_guest("end")
        out.append(s)
    return out


@scenario
def interplay_life_suspend_blocked():
    """Region B's second part is blocked on the gate when the host
    suspends: rvg_device_will_suspend quiesces, rvg_device_finish_suspend wakes
    and abandons the wait, and the blob's read count (key 7) stops before the
    command. The fresh device parses it again, where it waits again; the gate
    releases it. As in interplay_life_suspend_between_parts, B's private buffer
    did not survive in-process (B differs from A from +0x201); under isolation
    B ends as A."""
    s = Scenario("interplay_life_suspend_blocked", doc=interplay_life_suspend_blocked.__doc__)
    g = Rig(s)
    g.control()
    g.run("B", 0)
    st = g.queue("B", 1, barriers=g.gate_barrier())
    g.f.wait_stamp(st, timeout_ms=300, label="blocked on the gate")
    s.r(abi.R_FIFO_READ, "blocked")
    s.events("blocked")
    suspend(s)
    s.events("suspended")
    resume_fresh(s)
    g.f.wait_stamp(st, timeout_ms=300, label="blocked again after the resume")
    s.r(abi.R_FIFO_READ, "resumed")
    s.events("resumed")
    g.open_gate()
    g.f.wait_stamp(st)
    s.events("released")
    g.compare()
    return s


@scenario
def interplay_life_will_suspend_rung():
    """Between rvg_device_will_suspend and rvg_device_finish_suspend the FIFOs
    are quiesced and the run state suppresses interrupts: part two rung then
    does not run. The suspend completes, and on the fresh device the root FIFO
    resumes from the blob's counts, where written already counts part two: it
    runs without a new doorbell. B's private buffer did not survive in-process
    (B differs from A); under isolation B ends as A."""
    s = Scenario("interplay_life_will_suspend_rung", doc=interplay_life_will_suspend_rung.__doc__)
    g = Rig(s)
    g.control()
    g.run("B", 0)
    s.events("before")
    s.api("willSuspend")
    st = g.queue("B", 1)
    g.f.wait_stamp(st, timeout_ms=300, label="rung after willSuspend")
    s.r(abi.R_FIFO_READ, "after willSuspend")
    s.events("rung after willSuspend")
    s.api("finishSuspend", slot="s")
    s.events("suspended")
    resume_fresh(s)
    g.f.wait_stamp(st)
    s.events("resumed")
    g.compare()
    return s


@scenario
def interplay_life_will_suspend_unpause():
    """A host that calls rvg_device_will_suspend and then changes its mind with
    unpause (no guest display, so nothing asserts): unpause sets the run state
    back to running and restarts the quiesced FIFOs, so the part rung in
    between runs and B ends as A. A suspend cycle abandoned this way is
    transparent."""
    s = Scenario("interplay_life_will_suspend_unpause", doc=interplay_life_will_suspend_unpause.__doc__)
    g = Rig(s)
    g.control()
    g.run("B", 0)
    s.api("willSuspend")
    st = g.queue("B", 1)
    g.f.wait_stamp(st, timeout_ms=300, label="rung after willSuspend")
    s.events("after willSuspend")
    s.api("unpause")
    g.f.wait_stamp(st)
    s.events("unpaused")
    g.compare()
    g.open_gate()
    s.events("the gate still runs")
    return s


# ---- faults and pending stamps across pause and suspend ----

@scenario
def interplay_life_pause_over_fault():
    """A pause while the root FIFO is faulted with region B's second part
    held behind the faulted head: the quiesce publishes the held stamp (the
    exec ran; B equals A), but 0x102c stays 1 and 0x103c keeps the offset.
    The queue's faulted flag survives the pause: after unpause and 0x1024
    (whose head is no longer faulted), a new fault is recorded but never
    reported: no fault offset, no 0x102c, the stamps behind it held. A
    second 0x1024 finds the faulted head and releases them, and a third
    fault is reported as usual."""
    s = Scenario("interplay_life_pause_over_fault", doc=interplay_life_pause_over_fault.__doc__)
    g = Rig(s)
    g.control()
    g.run("B", 0)
    f = g.f
    f.cmd(99)
    st = f.cmd(abi.OP_EXEC_INDIRECT3, g.payload("B", 1))
    f.doorbell()
    s.wait_fault()
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written, label="both read")
    f.wait_stamp(st, timeout_ms=300, label="held behind the fault")
    s.fault_state("faulted")
    s.events("faulted")
    s.api("pause")
    f.wait_stamp(st, label="published by the pause")
    s.fault_state("paused")
    s.events("paused")
    s.api("unpause")
    s.fault_state("unpaused")
    s.resume_root()
    s.fault_state("0x1024")
    s.events("unpaused and resumed")
    g.compare()
    f.cmd(99)
    st2 = f.cmd(abi.OP_NOP)
    f.doorbell()
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written, label="the second fault read")
    f.wait_stamp(st2, timeout_ms=300, label="a NOP behind the second fault")
    s.wait_fault(timeout_ms=300, label="the second fault")
    s.fault_state("second fault")
    s.events("second fault")
    s.resume_root()
    f.wait_stamp(st2)
    s.fault_state("resumed again")
    s.events("resumed again")
    s.submit(99, fault=True, label="a third fault")
    s.fault_state("third fault")
    s.resume_root()
    s.submit(abi.OP_NOP)
    s.events("end")
    return s


@scenario
def interplay_life_pause_over_child_fault():
    """The same on the gate (a child FIFO): paused while faulted with a NOP
    held behind the fault, the NOP publishes, the node keeps its fault
    offset; after unpause and 0x1028, the next fault on the child is not
    reported (the node's fault offset stays 0xffffffff), until a second
    0x1028."""
    s = Scenario("interplay_life_pause_over_child_fault", doc=interplay_life_pause_over_child_fault.__doc__)
    g = Rig(s)
    c = g.gate
    c.cmd(99)
    st = c.cmd(abi.OP_NOP)
    c.doorbell()
    s.wait(gpa=c.node_gpa(8), ne=0xFFFFFFFF, label="child faulted")
    s.wait(gpa=c.node_gpa(4), eq=c.written, label="both read")
    c.wait_stamp(st, timeout_ms=300, label="held behind the fault")
    node(s, c, "faulted")
    s.fault_state("faulted")
    s.events("faulted")
    s.api("pause")
    c.wait_stamp(st, label="published by the pause")
    node(s, c, "paused")
    s.api("unpause")
    s.w(abi.R_RESUME_CHILD, c.fid)
    node(s, c, "resumed")
    s.fault_state("resumed")
    s.events("unpaused and resumed")
    c.cmd(99)
    st2 = c.cmd(abi.OP_NOP)
    c.doorbell()
    s.wait(gpa=c.node_gpa(4), eq=c.written, label="the second fault read")
    c.wait_stamp(st2, timeout_ms=300, label="a NOP behind the second fault")
    node(s, c, "second fault")
    s.fault_state("second fault")
    s.events("second fault")
    s.w(abi.R_RESUME_CHILD, c.fid)
    c.wait_stamp(st2)
    node(s, c, "resumed again")
    s.fault_state("resumed again")
    s.events("end")
    return s


@scenario
def interplay_life_suspend_over_fault():
    """A suspend while the root FIFO is faulted with region B's second part
    held: rvg_device_finish_suspend's drain publishes it, and the blob carries
    the fault (keys 11 and 19). The fresh device resumes faulted (0x102c 1),
    and its queue is new: 0x1024 resumes, and a later fault is reported as
    usual (unlike after a pause)."""
    s = Scenario("interplay_life_suspend_over_fault", doc=interplay_life_suspend_over_fault.__doc__)
    g = Rig(s)
    g.control()
    g.run("B", 0)
    f = g.f
    f.cmd(99)
    st = f.cmd(abi.OP_EXEC_INDIRECT3, g.payload("B", 1))
    f.doorbell()
    s.wait_fault()
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written, label="both read")
    f.wait_stamp(st, timeout_ms=300, label="held behind the fault")
    s.events("faulted")
    suspend(s)
    f.wait_stamp(st, label="published by finishSuspend")
    s.events("suspended")
    resume_fresh(s)
    s.events("resumed")
    s.fault_state("resumed")
    g.compare()
    s.resume_root()
    s.fault_state("0x1024")
    s.submit(99, fault=True, label="a fault after the resume")
    s.fault_state("fault after the resume")
    s.resume_root()
    s.submit(abi.OP_NOP)
    s.events("end")
    return s


# ---- Delay: every lifecycle call that drains hangs behind it ----

def _delay_rig(name, doc, child=False):
    s = Scenario(name, watchdog_s=8, doc=doc)
    g = Rig(s)
    g.control()
    f = g.gate if child else g.f
    a = f.cmd(abi.OP_NOP)
    f.cmd(abi.OP_DELAY, bytes(8))
    c = f.cmd(abi.OP_NOP)
    f.doorbell()
    f.wait_stamp(a)
    if child:
        s.wait(gpa=f.node_gpa(4), eq=f.written, label="the delay read")
    else:
        s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written, label="the delay read")
    f.wait_stamp(c, timeout_ms=300, label="a NOP behind the delay")
    s.events("a delay pending")
    return s, g


@scenario
def interplay_life_delay():
    """A Delay (61) never signals, and every lifecycle call that drains the
    FIFO holding it waits for it forever: pause, rvg_device_finish_suspend
    (rvg_device_will_suspend only quiesces, and returns), stop, reset, release
    (the device's shutdown stops it) and recreate (which releases first), with
    the Delay on the root FIFO; pause and rvg_device_finish_suspend with it on
    a child FIFO. Each hangs at that call: the watchdog fires there.
    rvg_device_will_suspend alone is locked returning, and a NOP rung after it
    is not run."""
    out = []
    for call in ("pause", "finish_suspend", "stop", "reset", "release", "recreate", "child_pause",
                 "child_finish_suspend"):
        s, g = _delay_rig("interplay_life_delay_" + call, interplay_life_delay.__doc__, child=call.startswith("child"))
        if call == "child_finish_suspend":
            s.api("willSuspend")
            s.events("willSuspend returned")
            s.api("finishSuspend", slot="s")
        elif call == "finish_suspend":
            s.api("willSuspend")
            st = s.fifo.cmd(abi.OP_NOP)
            s.fifo.doorbell()
            s.fifo.wait_stamp(st, timeout_ms=300, label="a NOP after willSuspend")
            s.events("willSuspend returned")
            s.api("finishSuspend", slot="s")
        elif call == "child_pause":
            s.api("pause")
        else:
            s.api(call)
        s.mark("not reached")
        out.append(s)
    return out


@scenario
def interplay_life_pause_in_flight():
    """Region B's second part rung, and the device paused as soon as the
    FIFO has read it (0x100c), without waiting for its stamp: the exec may
    still be on the GPU, and the pause's drain waits for it. Its stamp is
    published by then or by the drain (one interrupt either way, the run
    state still running), unpause raises its own, and B ends as A."""
    s = Scenario("interplay_life_pause_in_flight", doc=interplay_life_pause_in_flight.__doc__)
    g = Rig(s)
    g.control()
    g.run("B", 0)
    s.events("before")
    f = g.f
    st = f.cmd(abi.OP_EXEC_INDIRECT3, g.payload("B", 1))
    f.doorbell()
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written, label="read")
    s.api("pause")
    f.wait_stamp(st, label="published by the time the pause returned")
    s.api("unpause")
    s.events("paused and unpaused")
    g.compare()
    return s


# ---- lifetimes: what queued work names goes away before it runs ----

def _held(s, g):
    """Region B's second part queued on the root FIFO behind the gate."""
    st = g.queue("B", 1, barriers=g.gate_barrier())
    g.f.wait_stamp(st, timeout_ms=300, label="B's second part held by the gate")
    return st


@scenario
def interplay_life_delete_queued_buffer():
    """Region B's second part reads the source and B's private buffer; while
    it waits on the gate, the worker FIFO (child 2) deletes one of them:
    DeleteResource (37) of the source, of the private buffer, or
    DeleteObject (40, operation 1000) of the private buffer. Released, the
    exec makes the deleted buffer again from its object-list record: the
    shared source from guest memory (B ends as A), and in-process the
    private buffer from its guest pages (0x77 bytes, not what part one
    copied: B differs from A from +0x200). Under isolation the private
    buffer lives in its guest pages, so B ends as A."""
    out = []
    for suffix, what in (("src", "src"), ("priv", "priv"), ("priv_object", "priv_object")):
        s = Scenario("interplay_life_delete_queued_buffer_" + suffix, doc=interplay_life_delete_queued_buffer.__doc__)
        g = Rig(s, worker=2)
        g.control()
        g.run("B", 0)
        st = _held(s, g)
        w = g.worker
        if what == "src":
            w.submit(abi.OP_DELETE_RESOURCE, struct.pack("<II", 1, g.src), label="delete the source")
        elif what == "priv":
            w.submit(abi.OP_DELETE_RESOURCE, struct.pack("<II", 1, g.regions["B"].priv), label="delete B's private buffer")
        else:
            w.submit(abi.OP_DELETE_OBJECT, struct.pack("<IIII", 1, 1000, 12, g.regions["B"].priv),
                     label="delete B's private buffer (40)")
        s.events("deleted while held")
        g.open_gate()
        g.f.wait_stamp(st)
        s.events("released")
        s.fault_state("released")
        g.compare()
        g.priv_guest("end")
        out.append(s)
    return out


@scenario
def interplay_life_delete_queued_task():
    """Task 1 deleted (32, on the worker FIFO) while region B's second part
    waits on the gate for it: released, the exec names no task and faults
    the root FIFO. The guest defines task 1 again (its root and object list
    as before) and runs both of B's parts: B ends as A."""
    s = Scenario("interplay_life_delete_queued_task", doc=interplay_life_delete_queued_task.__doc__)
    g = Rig(s, worker=2)
    g.control()
    g.run("B", 0)
    st = _held(s, g)
    g.worker.submit(abi.OP_DELETE_TASK, struct.pack("<I", 1), label="delete task 1")
    s.dump(s.pa(s.root_page, 4), 4, "root word of task 1")
    s.events("deleted while held")
    g.open_gate()
    s.wait_fault(label="the exec faults")
    s.fault_state("released")
    s.events("released")
    s.resume_root()
    g.f.wait_stamp(st)
    g.t.define()
    g.t.install()
    g.run("B", 0)
    g.run("B", 1)
    s.events("redone")
    g.compare()
    return s


@scenario
def interplay_life_unmap_queued():
    """UnmapMemory (34, on the worker FIFO) over what region B's held second
    part uses: its command buffer's page, B's destination, or the object
    list's page. With a memory map the unmap only drops the object list's
    mapping if it overlaps; in the blocks setup the host's unmap_memory is
    called too, and the task maps the range again on use. Released, the
    exec runs in each case; MapMemory2 (57) of the task, and part two again,
    end the scenario."""
    out = []
    for suffix, va, n in (("cmdbuf", 4 * PAGE, PAGE), ("dst", None, PAGE), ("objlist", 0, PAGE)):
        s = Scenario("interplay_life_unmap_queued_" + suffix, doc=interplay_life_unmap_queued.__doc__)
        g = Rig(s, worker=2)
        g.control()
        g.run("B", 0)
        if va is None:
            va = g.regions["B"].dst_va
        st = _held(s, g)
        g.worker.submit(abi.OP_UNMAP_MEMORY, struct.pack("<IQQ", 1, va, n), label="unmap")
        s.events("unmapped while held")
        g.open_gate()
        g.f.wait_stamp(st, label="released")
        s.fault_state("released")
        s.events("released")
        g.compare("A and B after the release")
        g.t.map()
        g.run("B", 1)
        s.events("mapped again, part two again")
        g.compare("A and B, part two again")
        out.append(s)
    return out


@scenario
def interplay_life_delete_blocked_child():
    """The worker FIFO (child 2) holds a NOP and region B's second part,
    the exec blocked on the gate, when the root deletes child 2 (49): the
    delete quiesces it, which wakes and abandons the wait (the exec never
    runs); the node's running flag drops, its readCount stays before the
    exec, and the next node's page-list page is zeroed. Defined again (48)
    the FIFO starts from readCount 0 and parses its ring from the start:
    the NOP runs a second time (the guest zeroed its stamp slot, which
    reads 1 again) and the exec blocks on the gate again, which
    releases it; B ends as A."""
    s = Scenario("interplay_life_delete_blocked_child", doc=interplay_life_delete_blocked_child.__doc__)
    g = Rig(s, worker=2)
    g.control()
    g.run("B", 0)
    w = g.worker
    w.submit(abi.OP_NOP)
    st = w.cmd(abi.OP_EXEC_INDIRECT3, g.payload("B", 1), barriers=g.gate_barrier())
    w.doorbell()
    w.wait_stamp(st, timeout_ms=300, label="the worker's exec held by the gate")
    node(s, w, "held")
    s.events("held")
    w.delete()
    node(s, w, "deleted")
    s.dump(s.pa(s.root_page, abi.ROOT_CHILD_NODE + 2 * abi.CHILD_NODE_SIZE), abi.CHILD_NODE_SIZE, "node 3")
    stamps(s, 3, "stamp slots after the delete")
    s.events("deleted")
    s.u32(s.fifo.base_gpa() + 4 * w.stamp_index, 0)
    w.define()
    s.wait(gpa=s.fifo.base_gpa() + 4 * w.stamp_index, eq=1, label="the NOP replayed")
    w.wait_stamp(st, timeout_ms=300, label="the exec, held again")
    node(s, w, "defined again")
    stamps(s, 3, "stamp slots after the define")
    s.events("defined again")
    g.open_gate()
    w.wait_stamp(st)
    s.events("released")
    g.compare()
    return s


# ---- stop, reset and recreate with work held ----

@scenario
def interplay_life_teardown_held():
    """stop, reset, recreate (a fresh device, no resume) or release while
    region B's second part waits on the gate on the root FIFO and the worker
    FIFO waits on it too: the teardown wakes and abandons both waits and
    returns (no hang: nothing is pending). After release (the host drops
    the device) only the root page is locked. What the guest sees after it:
    registers, the root page, the nodes. The guest then does everything
    again (version after reset or recreate, root page, FIFO from 0, the
    gate, task 1 and its object list) and runs both of B's parts: B ends as
    A, which ran before the teardown."""
    out = []
    for call in ("stop", "reset", "recreate", "release"):
        s = Scenario("interplay_life_teardown_held_" + call, doc=interplay_life_teardown_held.__doc__)
        g = Rig(s, worker=2)
        g.control()
        g.run("B", 0)
        st = _held(s, g)
        w = g.worker
        wst = w.cmd(abi.OP_NOP, barriers=g.gate_barrier())
        w.doorbell()
        w.wait_stamp(wst, timeout_ms=300, label="the worker held by the gate")
        s.events("held")
        s.api(call)
        s.events(call)
        if call == "release":
            s.diff(call, gpa=s.pa(s.root_page), length=PAGE)
            out.append(s)
            continue
        s.sweep(0x1000, 0x1040, label="registers after " + call)
        s.diff(call, gpa=s.pa(s.root_page), length=PAGE)
        stamps(s, 3, "stamp slots after " + call)
        if call != "stop":
            s.w(abi.R_VERSION, abi.MAX_VERSION)
        s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
        g.f.written = 0
        g.f.enable()
        g.gate.written = 0
        g.gate.setup_node()
        g.gate.define()
        g.t.define()
        g.t.install()
        g.run("B", 0)
        g.run("B", 1)
        s.events("redone")
        g.compare()
        out.append(s)
    return out


# ---- resume onto a fresh device: what comes back, what the guest redoes ----

@scenario
def interplay_life_resume_rich():
    """A device with the root FIFO, the gate (child 1), a worker (child 2)
    deleted before the suspend, child 3 defined before that delete and
    faulted, task 1 (the rig) and task 2 (object and placement lists set
    with 68), suspended and resumed on a fresh device. What comes back:
    the root FIFO and its counts, the gate, task 1 with its object list
    (region B's second part runs without a redo). What does not: child 2
    (deleted: its running flag is 0), and child 3, whose page-list page the
    delete of child 2 zeroed (W3): it is created from its running node and
    fails to start, so its fault bit is set but nothing runs on it; the
    guest writes its page list again, deletes it (49) and defines it (48).
    Task 2 comes back from its root word with the object list the task
    root holds."""
    s = Scenario("interplay_life_resume_rich", stamp_slots=4, doc=interplay_life_resume_rich.__doc__)
    g = Rig(s, worker=2)
    c3 = ChildFIFO(s, 3)
    c3.setup_node()
    c3.define()
    c3.submit(abi.OP_NOP)
    from pgtest.task import GPUTask, seg
    from pgtest.lifecycle_util import fill as bfill
    t2 = GPUTask(s, tid=2, pages=16)
    b2, b2_va = t2.buffer()
    t2.define()
    t2.ol.write()
    place = t2.reserve(PAGE)
    t2.set_object_and_placement_list(0, place, t2.ol.length(), 16 * len(t2.ol.entries))
    cb2 = t2.cmdbuf(seg(abi.ENC_BLIT, bfill(b2, 0, 0x40, 0x5E)))
    t2.exec3([cb2], resources=[(b2, 5)])
    g.control()
    g.run("B", 0)
    g.worker.delete()
    c3.submit(99, fault=True)
    s.fault_state("before")
    s.events("before")
    s.diff("before the suspend", gpa=s.pa(s.root_page), length=PAGE)
    suspend(s)
    resume_fresh(s)
    s.events("resumed")
    s.sweep(0x1000, 0x1040, label="registers after the resume")
    s.fault_state("resumed")
    s.diff("after the resume", gpa=s.pa(s.root_page), length=PAGE)
    g.run("B", 1)
    g.compare()
    g.open_gate()
    t2.fill(b2_va, 0x40, 0)
    t2.exec3([cb2], resources=[(b2, 5)], label="task 2 after the resume")
    t2.dump(b2_va, 0x40, "task 2's buffer")
    st = c3.cmd(abi.OP_NOP)
    c3.doorbell()
    c3.wait_stamp(st, timeout_ms=300, label="child 3 after the resume")
    s.w(abi.R_RESUME_CHILD, 3)
    s.fault_state("child 3 resumed (0x1028)")
    c3.wait_stamp(st, timeout_ms=300, label="child 3 after 0x1028")
    c3.setup_node()
    c3.written = 0
    f = g.f
    f.cmd(abi.OP_DEFINE_CHILD_FIFO, struct.pack("<I", 3))
    f.doorbell()
    s.wait(mmio=abi.R_FAULT_OFFSET, mmio_ne=0xFFFFFFFF, label="child 3 defined again while it exists")
    s.fault_state("define of child 3 refused")
    s.resume_root()
    c3.delete()
    c3.define()
    s.fault_state("child 3 deleted and defined")
    c3.submit(abi.OP_NOP, label="child 3 runs again")
    g.worker.written = 0
    g.worker.setup_node()
    g.worker.define()
    g.worker.submit(abi.OP_NOP, label="child 2 defined again")
    s.fault_state("end")
    s.events("redone")
    return s


def _node(fid, written=0, read=0, fault=0xFFFFFFFF, stamp_index=None, running=1, page_list=0):
    return struct.pack("<IIIHHI", written, read, fault, fid if stamp_index is None else stamp_index, running,
                       page_list)


@scenario
def interplay_life_resume_child_without_root_fifo_dies():
    """A resume blob with a root page but no root FIFO, over a running node
    whose FIFO holds a written NOP: the restored child runs it and signals
    its stamp into the stamp memory the missing root FIFO would have
    provided (DeviceCore::signal_stamp writes through a null stamp memory):
    the host dies."""
    from pgtest.suspend_guest import blob, state
    s = Scenario("interplay_life_resume_child_without_root_fifo_dies",
                 doc=interplay_life_resume_child_without_root_fifo_dies.__doc__)
    root = s.root_page
    pl, ring = s.alloc_page(), s.alloc_page()
    s.u32(s.pa(pl), s.pfn(ring), 0)
    s.mem(s.pa(ring), abi.HDR.pack(abi.OP_NOP, 0, 12, 7))
    s.mem(s.pa(root, abi.ROOT_CHILD_NODE), _node(1, written=12, page_list=s.pfn(pl)))
    s.api("willResume", hex=blob(state(root=s.pfn(root))).hex(), label="resume")
    s.api("didResume")
    s.quiesce(200)
    s.events("resumed")
    return s


# ---- displays: presents held across the lifecycle, and what they present going away ----

def _display_guest(s, gate=True):
    """A connected, acked display on port 0, task 0 with one BGRA backing
    (16 x 8), and optionally a gate (child FIFO 1)."""
    from pgtest.display_guest import DisplayGuest, pattern
    g = DisplayGuest(s)
    r, va = g.backing(16, 8, pixels=pattern(16, 8, 3))
    g.install()
    g.connect(0, enabled=0xF)
    g.ack(0, 1)
    c = None
    if gate:
        c = ChildFIFO(s, 1)
        c.setup_node()
        c.define()
    return g, r, va, c


@scenario(host=True)
def interplay_life_display_delete_presented():
    """The backing a display shows is deleted (54) after its present: the
    display keeps the frame it has (an encode still reads it), and a
    present of the same reference makes the backing again from task 0's
    object list and guest memory, which the guest changed meanwhile: the
    new frame shows the new bytes. Pixels are drawn by the library's
    shaders: keyed by the host."""
    from pgtest.display_guest import p_present3, pattern
    s = Scenario("interplay_life_display_delete_presented", host=True, displays=[{"port": 0, "serial": 1000,
                 "name": "suite", "modes": [[64, 32, 60]]}], doc=interplay_life_display_delete_presented.__doc__)
    g, r, va, _ = _display_guest(s, gate=False)
    samples = [[0, 0], [15, 7], [5, 3]]
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, r))
    s.wait(cb="newFrame")
    s.events("presented")
    s.encode(0, 16, 8, label="presented", samples=samples)
    s.submit(abi.OP_DELETE_IOSURFACE_BACKING2, struct.pack("<II", r, 0), label="delete the presented backing")
    s.events("deleted")
    s.encode(0, 16, 8, label="after the delete", samples=samples)
    s.api("present_counts", display=0)
    g.t0.write(va, pattern(16, 8, 0x60))
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, r), label="present the deleted reference")
    s.wait(cb="newFrame", cb_count=2)
    s.events("presented again")
    s.encode(0, 16, 8, label="presented again", samples=samples)
    s.api("present_counts", display=0)
    g.diff_page(0, "page")
    return s


@scenario
def interplay_life_display_held_present():
    """A present (7) held on the root FIFO behind the gate while the host
    pauses and unpauses (the guest display is paused and unpaused around it,
    and the present still waits), then released: it presents. Then another
    is held while the host releases the display (hot unplug: the page
    reports the disconnect); released, it presents to the guest display with
    no client: the stamp is signalled and no callback is made. A new display
    on the port reconnects, and a present reaches it."""
    from pgtest.display_guest import p_present3
    s = Scenario("interplay_life_display_held_present", displays=[{"port": 0, "serial": 1000, "name": "suite",
                 "modes": [[64, 32, 60]]}], doc=interplay_life_display_held_present.__doc__)
    g, r, va, c = _display_guest(s)
    f = s.fifo
    s.events("connected")
    st = f.cmd(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, r), barriers=[(1, 1)])
    f.doorbell()
    f.wait_stamp(st, timeout_ms=300, label="the present held")
    s.api("pause")
    s.api("unpause")
    f.wait_stamp(st, timeout_ms=300, label="the present held after unpause")
    s.events("paused and unpaused")
    c.submit(abi.OP_NOP)
    f.wait_stamp(st)
    s.wait(cb="newFrame")
    s.events("released: presented")
    st = f.cmd(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, r), barriers=[(1, 2)])
    f.doorbell()
    f.wait_stamp(st, timeout_ms=300, label="another present held")
    s.api("release_display", display=0)
    s.quiesce(100)
    s.events("display released")
    g.diff_page(0, "page after the release")
    c.submit(abi.OP_NOP)
    s.wait_any([dict(gpa=f.stamp_gpa(), ge=st), dict(mmio=abi.R_FAULT_MASK, mmio_eq=1)],
               label="the present to a released display")
    s.quiesce(100)
    s.events("released: presented with no client")
    s.fault_state("after")
    s.api("new_display", spec={"port": 0, "serial": 1001, "name": "again", "modes": [[64, 32, 60]]})
    s.quiesce(100)
    s.events("a new display")
    g.diff_page(0, "page after the new display")
    g.ack(0, 2)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, r), label="present to the new display")
    s.wait(cb="newFrame", cb_count=2)
    s.events("presented to the new display")
    s.api("present_counts", display=1)
    return s


@scenario
def interplay_life_display_suspend_held_present():
    """A present held behind the gate when the host suspends:
    rvg_device_finish_suspend abandons the wait, and the restore on the fresh
    device does not present the last frame again (a present of opcode 7
    restores through the mapping path only); the root FIFO parses the held
    present again and waits; the gate releases it and the restored display gets
    its frame."""
    from pgtest.display_guest import p_present3
    s = Scenario("interplay_life_display_suspend_held_present", displays=[{"port": 0, "serial": 1000,
                 "name": "suite", "modes": [[64, 32, 60]]}], doc=interplay_life_display_suspend_held_present.__doc__)
    g, r, va, c = _display_guest(s)
    f = s.fifo
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, r))
    s.wait(cb="newFrame")
    s.events("presented")
    st = f.cmd(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, r), barriers=[(1, 1)])
    f.doorbell()
    f.wait_stamp(st, timeout_ms=300, label="the present held")
    suspend(s)
    g.diff_page(0, "suspended")
    resume_fresh(s)
    f.wait_stamp(st, timeout_ms=300, label="held again after the resume")
    s.quiesce(100)
    s.events("resumed")
    g.diff_page(0, "resumed")
    c.submit(abi.OP_NOP)
    f.wait_stamp(st)
    s.wait(cb="newFrame")
    s.events("released")
    g.diff_page(0, "presented after the resume")
    s.api("present_counts", display=0)
    return s


# ---- abandoned suspends ----

@scenario
def interplay_life_finish_suspend_unpause():
    """A host that abandons a suspend after rvg_device_finish_suspend and calls
    unpause instead of resuming (no guest display): rvg_device_finish_suspend
    dropped the root FIFO, the children and the tasks but left 0x1000 reading
    1, so a doorbell is never answered. The guest disables and enables its FIFO
    (0x1000 = 0, 1): the new FIFO starts at read 0, and the next doorbell (the
    guest's written count, unchanged) replays the whole ring from its start:
    the gate's define, task 1's define and object list, both region A's parts,
    B's first part and the unanswered NOP run again, every stamp in order (how
    many interrupts publish them is timing, and is not locked). B's second part
    then runs, and B equals A in every setup: the replay made B's private
    buffer again."""
    s = Scenario("interplay_life_finish_suspend_unpause", doc=interplay_life_finish_suspend_unpause.__doc__)
    g = Rig(s)
    g.control()
    g.run("B", 0)
    suspend(s)
    s.api("unpause")
    s.events("abandoned")
    s.sweep(0x1000, 0x1040, label="registers")
    f = g.f
    st = s.submit(abi.OP_NOP, timeout_ms=300, label="a NOP after the abandoned suspend")
    s.w(abi.R_FIFO_ENABLE, 0)
    s.w(abi.R_FIFO_ENABLE, 1)
    s.sweep(0x1000, 0x1040, label="registers after 0x1000 = 0, 1")
    s.u32(f.stamp_gpa(), 0)
    f.doorbell()
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written, label="the ring read again")
    f.wait_stamp(st, label="the replay's last stamp")
    # The replay is one batch with GPU work in it: one interrupt publishes
    # every stamp ready at the head (pump), so how many interrupts it
    # raises is timing (B's exec and the NOP behind it share one, or not):
    # counted collapsed.
    s.events("replayed", collapse=["irq"])
    s.sweep(0x1000, 0x1040, label="registers after the replay")
    stamps(s, 2, "stamp slots")
    g.run("B", 1)
    g.compare()
    return s


@scenario
def interplay_life_will_suspend_unpause_display_dies():
    """rvg_device_will_suspend, then unpause, with a guest display:
    rvg_device_will_suspend never paused the guest display, and the guest
    display asserts it was paused: the host dies at unpause."""
    from pgtest.display_guest import DisplayGuest
    s = Scenario("interplay_life_will_suspend_unpause_display_dies", displays=[{"port": 0, "serial": 1000,
                 "name": "suite", "modes": [[64, 32, 60]]}], doc=interplay_life_will_suspend_unpause_display_dies.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0, enabled=0xF)
    g.ack(0, 1)
    s.api("willSuspend")
    s.events("willSuspend")
    s.api("unpause")
    s.events("after")
    return s


# ---- stamp indices shared while a FIFO waits ----

@scenario
def interplay_life_define_over_waiting_index_dies():
    """Child 1 has stamp index 5 and waits on a barrier, which records the
    wait under index 5 (DeviceCore::wait_stamps keys it by the FIFO's stamp
    index). Defining child 5 then finds that index's waiting info in use:
    DeviceCore::create_fifo asserts it is free, and the host dies at the
    define."""
    s = Scenario("interplay_life_define_over_waiting_index_dies", stamp_slots=6,
                 doc=interplay_life_define_over_waiting_index_dies.__doc__)
    s.bring_up()
    a = ChildFIFO(s, 1, stamp_index=5)
    a.setup_node()
    a.define()
    b = ChildFIFO(s, 2)
    b.setup_node()
    b.define()
    st = a.cmd(abi.OP_NOP, barriers=[(2, 1)])
    a.doorbell()
    a.wait_stamp(st, timeout_ms=300, label="child 1 waits under index 5")
    s.events("waiting")
    c = ChildFIFO(s, 5)
    c.setup_node()
    c.define()
    return s


@scenario
def interplay_life_two_waiters_one_index_dies():
    """Children 1 and 2 share stamp index 3 (nothing checks uniqueness) and
    each waits on a barrier: the second wait finds index 3 already waiting,
    and DeviceCore::add_wait asserts: the host dies."""
    s = Scenario("interplay_life_two_waiters_one_index_dies", stamp_slots=6,
                 doc=interplay_life_two_waiters_one_index_dies.__doc__)
    s.bring_up()
    a = ChildFIFO(s, 1, stamp_index=3)
    b = ChildFIFO(s, 2, stamp_index=3)
    c = ChildFIFO(s, 4)
    for x in (a, b, c):
        x.setup_node()
        x.define()
    st = a.cmd(abi.OP_NOP, barriers=[(4, 1)])
    a.doorbell()
    a.wait_stamp(st, timeout_ms=300, label="child 1 waits under index 3")
    s.events("one waiting")
    st = b.cmd(abi.OP_NOP, barriers=[(4, 1)])
    b.doorbell()
    b.wait_stamp(st, timeout_ms=300, label="child 2 waits under index 3")
    s.events("after")
    return s


@scenario
def interplay_life_display_reset_held():
    """reset while a present waits on the gate with a display connected:
    the teardown abandons the present and tears the guest display down: the
    page's raised bits read 0xc (the disconnect added to the connect), one
    interrupt, and port 0's root word is cleared. The display stays attached
    to the device. The guest negotiates again, brings up the root FIFO and
    task 0, connects the port and acks, and presents: the frame arrives."""
    from pgtest.display_guest import p_present3
    s = Scenario("interplay_life_display_reset_held", displays=[{"port": 0, "serial": 1000, "name": "suite",
                 "modes": [[64, 32, 60]]}], doc=interplay_life_display_reset_held.__doc__)
    g, r, va, c = _display_guest(s)
    f = s.fifo
    st = f.cmd(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, r), barriers=[(1, 1)])
    f.doorbell()
    f.wait_stamp(st, timeout_ms=300, label="the present held")
    s.events("held")
    s.api("reset")
    s.quiesce(100)
    s.events("reset")
    s.dump(s.pa(s.root_page, abi.ROOT_DISPLAY_WORD), 4, "root word of port 0")
    g.diff_page(0, "page after the reset")
    s.api("display_props", display=0)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    f.written = 0
    f.enable()
    g.t0.define()
    g.install()
    g.connect(0)
    s.quiesce(100)
    s.events("connected again")
    g.diff_page(0, "page after the connect")
    g.ack(0, 2)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, r), label="present after the reset")
    s.wait(cb="newFrame")
    s.events("presented")
    s.api("present_counts", display=0)
    return s
