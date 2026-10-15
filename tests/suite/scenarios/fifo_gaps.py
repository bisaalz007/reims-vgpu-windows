"""FIFO and stamp corner cases the per-area modules leave open
(docs/testing.md, sections 6.6 and 6.7): the pending-stamp queue's fault
latch across a pause, barriers compared across the signed wrap of the
stamp space, and what a teardown does to a faulted head.

The pending-stamp queue's StampQueue::pump reports a faulted head only while
the queue is not quiesced and its latch (faulted) is clear, sets the latch when
it reports, and only StampQueue::resume_from_fault clears it, and only when it
finds a faulted head. The barrier wait is DeviceCore::wait_stamp, which
compares (int32_t)(slot - value) >= 0.
"""
from pgtest import abi
from pgtest.fifo_util import child, child_fault, stamps, track_stamps
from pgtest.registry import scenario
from pgtest.scenario import Scenario

BAD = 99  # an opcode the FIFO's dispatch does not know: log and fault


@scenario
def fifo_gap_fault_latch_after_pause():
    """A faulted head that a pause publishes leaves the queue's fault latch
    set: pause quiesces the queue, and pump() then publishes the faulted
    head as if signalled without clearing _faulted. The guest's 0x1024
    after unpause clears 0x102c but finds no faulted head ("Stamp at head
    isn't faulted"), so the latch stays. A NOP still publishes; the next
    faulting command is then never reported: its stamp is held with 0x102c
    0 and 0x103c 0xffffffff, and the NOP behind it waits (a 500 ms wait
    that times out, and ok false, is the behaviour). A second 0x1024 finds
    that faulted head, clears the latch and publishes both; a third fault
    is reported again."""
    s = Scenario("fifo_gap_fault_latch_after_pause", doc=fifo_gap_fault_latch_after_pause.__doc__)
    f = s.bring_up()
    track_stamps(s, 1)
    s.submit(BAD, fault=True, label="first fault")
    s.fault_state("first fault")
    s.api("pause")
    s.events("paused", irq_stamps=True)
    s.fault_state("paused")
    s.api("unpause")
    s.resume_root()
    s.events("unpaused and resumed")
    s.fault_state("unpaused and resumed")
    s.submit(abi.OP_NOP, label="a NOP publishes")
    bad = f.cmd(BAD)
    nop = f.cmd(abi.OP_NOP)
    f.doorbell()
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written, label="both consumed")
    f.wait_stamp(nop, timeout_ms=500, label="the NOP behind the silent fault")
    s.fault_state("silent fault")
    s.events("silent fault", irq_stamps=True)
    stamps(s, 1)
    s.resume_root()
    f.wait_stamp(nop, label="released by the second 0x1024")
    s.fault_state("after the second 0x1024")
    s.events("after the second 0x1024", irq_stamps=True)
    s.submit(BAD, fault=True, label="third fault")
    s.fault_state("third fault")
    s.resume_root()
    s.submit(abi.OP_NOP)
    s.events("end", irq_stamps=True)
    stamps(s, 1)
    return s


@scenario
def fifo_gap_fault_latch_after_pause_child():
    """The latch of fifo_gap_fault_latch_after_pause on a child FIFO: its
    fault published by pause, 0x1028 after unpause, then a faulting command
    on the child is silent (its node's fault offset stays 0xffffffff,
    0x102c 0, the NOP behind it held: a 500 ms wait that times out is the
    behaviour), until a second 0x1028 releases both."""
    s = Scenario("fifo_gap_fault_latch_after_pause_child", stamp_slots=2,
                 doc=fifo_gap_fault_latch_after_pause_child.__doc__)
    s.bring_up()
    c = child(s, 1)
    c.cmd(BAD)
    c.doorbell()
    child_fault(s, c, label="child fault")
    s.fault_state("child fault")
    s.dump(c.node_gpa(), 0x14, "node after the fault")
    s.api("pause")
    s.api("unpause")
    s.w(abi.R_RESUME_CHILD, 1)
    s.events("unpaused and resumed")
    s.r(abi.R_FAULT_MASK, "unpaused and resumed")
    c.submit(abi.OP_NOP, label="a NOP publishes")
    c.cmd(BAD)
    nop = c.cmd(abi.OP_NOP)
    c.doorbell()
    s.wait(gpa=c.node_gpa(4), eq=c.written, label="both consumed")
    c.wait_stamp(nop, timeout_ms=500, label="the NOP behind the silent fault")
    s.r(abi.R_FAULT_MASK, "silent fault")
    s.dump(c.node_gpa(), 0x14, "node after the silent fault")
    s.events("silent fault")
    s.w(abi.R_RESUME_CHILD, 1)
    c.wait_stamp(nop, label="released by the second 0x1028")
    s.r(abi.R_FAULT_MASK, "after the second 0x1028")
    s.dump(c.node_gpa(), 0x14, "node after the second 0x1028")
    s.events("end")
    stamps(s, 2)
    return s


@scenario
def fifo_gap_barrier_signed_distance():
    """Barriers compare (int32_t)(slot - value) >= 0, so they wrap: with
    root slot 0 at 0xfffffffe, a barrier on value 0x7fffffff (distance
    2^31 - 1) passes at once, while one on 0x7ffffffe (distance 2^31,
    negative) blocks child 1 (a 300 ms wait that times out is the
    behaviour). Root stamps 2 and 0x7ffffffd leave it blocked; 0x7ffffffe
    releases it. A barrier on 5 while the slot reads 0xfffffffb blocks
    until the stamp wraps through 0 to 5."""
    s = Scenario("fifo_gap_barrier_signed_distance", stamp_slots=2, doc=fifo_gap_barrier_signed_distance.__doc__)
    f = s.bring_up()
    c = child(s, 1)
    s.submit(abi.OP_NOP, stamp=0xFFFFFFFE, label="root 0xfffffffe")
    c.submit(abi.OP_NOP, barriers=[(0, 0x7FFFFFFF)], label="barrier 0:0x7fffffff passes")
    st = c.cmd(abi.OP_NOP, barriers=[(0, 0x7FFFFFFE)])
    c.doorbell()
    c.wait_stamp(st, timeout_ms=300, label="blocked on 0:0x7ffffffe")
    for v in (2, 0x7FFFFFFD):
        s.submit(abi.OP_NOP, stamp=v, label="root 0x%x" % v)
        c.wait_stamp(st, timeout_ms=300, label="still blocked at root 0x%x" % v)
    s.submit(abi.OP_NOP, stamp=0x7FFFFFFE, label="root 0x7ffffffe")
    c.wait_stamp(st, label="released at 0x7ffffffe")
    s.submit(abi.OP_NOP, stamp=0xFFFFFFFB, label="root 0xfffffffb")
    st = c.cmd(abi.OP_NOP, barriers=[(0, 5)])
    c.doorbell()
    c.wait_stamp(st, timeout_ms=300, label="blocked on 0:5")
    s.submit(abi.OP_NOP, stamp=0, label="root 0")
    c.wait_stamp(st, timeout_ms=300, label="still blocked at root 0")
    s.submit(abi.OP_NOP, stamp=5, label="root 5")
    c.wait_stamp(st, label="released at 5")
    s.fault_state("end")
    s.events("end")
    stamps(s, 2)
    return s


@scenario
def fifo_gap_teardown_faulted_head():
    """0x1000 = 0 with a faulted head and two NOPs behind it: the teardown
    stops and drains the queue quiesced, which publishes the faulted head
    and everything behind it into the stamp slot before the mapping goes,
    and clears the fault mask: 0x102c reads 0 and 0x103c 0 (no FIFO). A
    FIFO enabled again on the same pages starts with fresh counts and no
    fault, and 0x1024 on it changes nothing; its first NOP (stamp 11)
    publishes."""
    s = Scenario("fifo_gap_teardown_faulted_head", doc=fifo_gap_teardown_faulted_head.__doc__)
    f = s.bring_up()
    track_stamps(s, 1)
    s.submit(abi.OP_NOP)
    f.cmd(BAD)
    f.cmd(abi.OP_NOP)
    f.cmd(abi.OP_NOP)
    f.doorbell()
    s.wait_fault(label="faulted head")
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written, label="all consumed")
    s.fault_state("faulted, two behind")
    stamps(s, 1, label="before teardown")
    s.w(abi.R_FIFO_ENABLE, 0)
    s.events("torn down", irq_stamps=True)
    stamps(s, 1, label="after teardown")
    s.fault_state("torn down")
    s.r(abi.R_FIFO_READ, "torn down")
    f.written = 0
    f.stamp = 10  # above what the slot holds, so the wait below means this NOP
    f.enable()
    s.r(abi.R_FIFO_ENABLE, "enabled again")
    s.fault_state("enabled again")
    s.resume_root()
    s.fault_state("0x1024 on the new FIFO")
    s.submit(abi.OP_NOP)
    s.r(abi.R_FIFO_READ, "a NOP on the new FIFO")
    s.events("end", irq_stamps=True)
    return s
