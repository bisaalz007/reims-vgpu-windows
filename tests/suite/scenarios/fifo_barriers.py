"""Barriers: {u32 fifoIndex, u32 value} after a command's header, waited on
before the command is dispatched (docs/testing.md, section 6.6, step 5).

A barrier is satisfied when (int32_t)(slot - value) >= 0, the slot being
the published stamp at the root FIFO's base + 4 x fifoIndex. An index at or
above maxFIFOCount (64 at version 245, 32 below 61) throws; a wait whose
chain of waiters comes back to the waiting FIFO while the stamps it needs
are not even pending throws "Circular wait". Both fault the command, after
its bytes were consumed. An unsatisfiable barrier blocks its FIFO before
the read count moves; the wait re-checks on every stamp signalled anywhere,
and on its own every 2 s.
"""
from pgtest import abi
from pgtest.fifo_util import child, probe, resume_child, stamps, track_stamps
from pgtest.registry import scenario
from pgtest.scenario import Scenario

F, S = "fault", "signal"


@scenario
def fifo_barrier_root():
    """Barriers on the root FIFO: an index whose slot already holds the
    value, index 63 with value 0, a value behind the slot by serial
    arithmetic (0x90000005 against 5), and several barriers at once all run
    the command. Index 64 and index 0xffffffff fault it. A barrier on the
    root's own slot for a value beyond the last pushed stamp is a circular
    wait and faults; 0x103c then names the command's ring offset. 32
    barriers on index 0 hit the same check each time."""
    s = Scenario("fifo_barrier_root", doc=fifo_barrier_root.__doc__)
    f = s.bring_up()
    track_stamps(s, 1)
    for _ in range(5):
        f.cmd(abi.OP_NOP)
    f.doorbell()
    f.wait_stamp(5)
    probe(s, abi.OP_NOP, expect=S, barriers=[(0, 5)], label="index 0, value 5 (reached)")
    probe(s, abi.OP_NOP, expect=S, barriers=[(63, 0)], label="index 63, value 0")
    probe(s, abi.OP_NOP, expect=S, barriers=[(0, 0x90000005)], label="index 0, value 0x90000005 (behind)")
    probe(s, abi.OP_NOP, expect=S, barriers=[(0, 1), (1, 0), (2, 0), (0, 8)], label="four barriers")
    probe(s, abi.OP_NOP, expect=S, barriers=[(0, 3)] * 32, label="32 barriers")
    probe(s, abi.OP_NOP, expect=F, barriers=[(64, 0)], label="index 64")
    probe(s, abi.OP_NOP, expect=F, barriers=[(0xFFFFFFFF, 0)], label="index 0xffffffff")
    probe(s, abi.OP_NOP, expect=F, barriers=[(0, 1), (64, 0)], label="index 64 after a satisfied one")
    st = f.stamp
    probe(s, abi.OP_NOP, expect=F, barriers=[(0, st + 2)], label="index 0, a value not yet pushed: circular")
    probe(s, abi.OP_NOP, expect=S, label="a NOP after")
    s.r(abi.R_FIFO_READ)
    s.events("root barriers")
    return s


@scenario
def fifo_barrier_version_60():
    """At version 60 maxFIFOCount is 32 (features::features_for_version): a
    barrier on index 31 runs, index 32 faults."""
    s = Scenario("fifo_barrier_version_60", doc=fifo_barrier_version_60.__doc__)
    s.bring_up(version=60)
    s.r(abi.R_VERSION)
    probe(s, abi.OP_NOP, expect=S, barriers=[(31, 0)], label="index 31")
    probe(s, abi.OP_NOP, expect=F, barriers=[(32, 0)], label="index 32")
    s.events("version 60")
    return s


@scenario
def fifo_barrier_blocked_wakes_at_timeout():
    """A barrier on stamp slot 5, which no FIFO signals, blocks the root FIFO
    before the command is consumed: its stamp and 0x100c stay put, with no
    fault. The guest then stores 1 in slot 5 itself. Nothing signals, but the
    wait wakes every 2 s on its own, finds the slot reached ("Woke up from
    timeout"), and the command and the NOP after it run."""
    s = Scenario("fifo_barrier_blocked_wakes_at_timeout", doc=fifo_barrier_blocked_wakes_at_timeout.__doc__)
    f = s.bring_up()
    track_stamps(s, 6)
    a = f.cmd(abi.OP_NOP)
    b = f.cmd(abi.OP_NOP, barriers=[(5, 1)])
    c = f.cmd(abi.OP_NOP)
    f.doorbell()
    f.wait_stamp(a)
    f.wait_stamp(b, timeout_ms=500, label="blocked on slot 5")
    s.r(abi.R_FIFO_READ)
    s.fault_state("blocked, not faulted")
    s.u32(f.base_gpa() + 4 * 5, 1)
    f.wait_stamp(c, timeout_ms=6000, label="woken by the 2 s timeout")
    s.r(abi.R_FIFO_READ)
    s.events("blocked, then woken", irq_stamps=True)
    stamps(s, 6)
    return s


@scenario
def fifo_barrier_blocked_wakes_on_signal():
    """A barrier on slot 5 blocks the root FIFO past its 2 s timeout (the
    wait logs, reports every FIFO's pending stamps, and then waits with no
    timeout). The guest stores 1 in slot 5; the root stays blocked until a
    stamp is signalled anywhere: a NOP on child FIFO 1 wakes it, and the
    root's command runs after the child's stamp is published."""
    s = Scenario("fifo_barrier_blocked_wakes_on_signal", doc=fifo_barrier_blocked_wakes_on_signal.__doc__)
    f = s.bring_up()
    track_stamps(s, 6)
    c1 = child(s, 1)
    b = f.cmd(abi.OP_NOP, barriers=[(5, 1)])
    f.doorbell()
    f.wait_stamp(b, timeout_ms=3000, label="blocked past the 2 s timeout")
    s.u32(f.base_gpa() + 4 * 5, 1)
    f.wait_stamp(b, timeout_ms=500, label="still blocked: nothing signalled")
    st = c1.cmd(abi.OP_NOP)
    c1.doorbell()
    c1.wait_stamp(st)
    f.wait_stamp(b, label="woken by child 1's stamp")
    s.events("child 1's stamp, then the root's", order=True, irq_stamps=True)
    stamps(s, 6)
    return s


@scenario
def fifo_barrier_between_fifos():
    """Barriers across FIFOs. Child 1 waits for root stamp 3 while the root is
    at 1; two root NOPs then release it, and its stamp is published after the
    root's (the wait re-checks under the lock DeviceCore::signal_stamp holds
    while it raises the root's interrupt). Then the root waits for child 1's
    stamp 3, and the child's NOPs release it."""
    s = Scenario("fifo_barrier_between_fifos", doc=fifo_barrier_between_fifos.__doc__)
    f = s.bring_up()
    track_stamps(s, 2)
    c1 = child(s, 1)
    s.events("defined")
    st = c1.cmd(abi.OP_NOP, barriers=[(0, 3)])
    c1.doorbell()
    c1.wait_stamp(st, timeout_ms=500, label="child 1 blocked on root stamp 3")
    s.dump(c1.node_gpa(4), 4, "child 1 read count, not advanced")
    f.cmd(abi.OP_NOP)
    r = f.cmd(abi.OP_NOP)
    f.doorbell()
    f.wait_stamp(r)
    c1.wait_stamp(st)
    s.events("the root's two stamps, then child 1's", order=True, irq_stamps=True)
    rb = f.cmd(abi.OP_NOP, barriers=[(1, 3)])
    f.doorbell()
    f.wait_stamp(rb, timeout_ms=500, label="root blocked on child 1 stamp 3")
    c1.cmd(abi.OP_NOP)
    c3 = c1.cmd(abi.OP_NOP)
    c1.doorbell()
    c1.wait_stamp(c3)
    f.wait_stamp(rb)
    s.events("child 1's two stamps, then the root's", order=True, irq_stamps=True)
    stamps(s, 2)
    return s


@scenario
def fifo_barrier_circular():
    """A circular wait across two FIFOs: the root blocks on child 1's stamp
    5, then child 1 issues a command waiting on root stamp 3, which the root
    has not pushed (its last pushed stamp is 1): the chain root -> child 1
    -> root throws "Circular wait" in child 1, which faults at node +8 = 0.
    0x1028 = 1 resumes it; child 1's NOPs up to stamp 5 then release the
    root. (The root's wait is registered first: the scenario waits 500 ms on
    its blocked stamp before child 1's command.)"""
    s = Scenario("fifo_barrier_circular", doc=fifo_barrier_circular.__doc__)
    f = s.bring_up()
    track_stamps(s, 2)
    c1 = child(s, 1)
    f.cmd(abi.OP_NOP)
    rb = f.cmd(abi.OP_NOP, barriers=[(1, 5)])
    f.doorbell()
    f.wait_stamp(rb, timeout_ms=500, label="root blocked on child 1 stamp 5")
    probe(s, abi.OP_NOP, expect=F, fifo=c1, barriers=[(0, 3)], label="child 1 waits on root stamp 3: circular")
    for _ in range(4):
        st = c1.cmd(abi.OP_NOP)
    c1.doorbell()
    c1.wait_stamp(st)
    f.wait_stamp(rb)
    s.fault_state("both running")
    s.events("circular, resumed, released", irq_stamps=True)
    stamps(s, 2)
    return s


@scenario
def fifo_barrier_self_wait_pending():
    """A barrier on the root's own slot for a stamp that is pushed but never
    signalled (behind a Delay) is not circular: the value is pending, so the
    wait is registered and blocks for good. The FIFO stays blocked with no
    fault; teardown quiesces the wait, then hangs draining the Delay: the
    outcome is the watchdog in teardown."""
    s = Scenario("fifo_barrier_self_wait_pending", watchdog_s=8, doc=fifo_barrier_self_wait_pending.__doc__)
    f = s.bring_up()
    a = f.cmd(abi.OP_NOP)
    d = f.cmd(abi.OP_DELAY, bytes(8))
    b = f.cmd(abi.OP_NOP, barriers=[(0, d)])
    f.doorbell()
    f.wait_stamp(a)
    f.wait_stamp(b, timeout_ms=500, label="blocked on the Delay's stamp")
    s.fault_state("not faulted")
    s.r(abi.R_FIFO_READ)
    s.events("blocked")
    return s


@scenario
def fifo_barrier_resume_report():
    """0x1028 on a FIFO that is not faulted logs and synchronously reports
    every FIFO's wait and pending stamps, here with the root blocked on
    slot 5 behind a Delay of its own, child 2 blocked on slot 6, and child
    1 holding 11 unpublished stamps behind a Delay. None of it reaches the
    guest: the stamps, 0x102c and the interrupts are unchanged. Teardown
    drains child 1's Delay and hangs."""
    s = Scenario("fifo_barrier_resume_report", watchdog_s=8, doc=fifo_barrier_resume_report.__doc__)
    f = s.bring_up()
    track_stamps(s, 3)
    c1 = child(s, 1)
    c1.cmd(abi.OP_DELAY, bytes(8))
    for _ in range(11):
        last = c1.cmd(abi.OP_NOP)
    c1.doorbell()
    s.wait(gpa=c1.node_gpa(4), eq=c1.written, label="child 1 read everything")
    c2 = child(s, 2)
    st2 = c2.cmd(abi.OP_NOP, barriers=[(6, 1)])
    c2.doorbell()
    f.cmd(abi.OP_DELAY, bytes(8))
    b = f.cmd(abi.OP_NOP, barriers=[(5, 1)])
    f.doorbell()
    f.wait_stamp(b, timeout_ms=500, label="root blocked on slot 5 behind a Delay")
    c2.wait_stamp(st2, timeout_ms=100, label="child 2 blocked on slot 6")
    s.events("before the reports")
    resume_child(s, 1)
    resume_child(s, 2)
    resume_child(s, 0)
    s.fault_state("after 0x1028 = 1, 2, 0")
    c1.wait_stamp(last, timeout_ms=300, label="child 1 still behind its Delay")
    s.events("the reports changed nothing")
    stamps(s, 3)
    return s
