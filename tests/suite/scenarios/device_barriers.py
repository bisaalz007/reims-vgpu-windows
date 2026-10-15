"""Barrier waits between FIFOs, as the device implements them, and the
stamp-wait report a guest can trigger (docs/testing.md, section 6.6 step 5).

DeviceCore::wait_stamps raises for an index at or above maxFIFOCount;
DeviceCore::wait_stamp returns at once when the slot already reached the value,
else records the wait (DeviceCore::add_wait, which raises a circular wait when
the chain of waits returns to the waiter) and sleeps, logging and posting a
report every 2 s, until the slot reaches the value or the FIFO is quiesced.
DeviceCore::report_stamp_wait_timeout walks the waiting FIFOs through
DeviceCore::fifo_for_stamp_index, which asserts that a child FIFO found at a
stamp index has that stamp index.
"""
from pgtest import abi
from pgtest.fifo import ChildFIFO
from pgtest.registry import scenario
from pgtest.scenario import Scenario


def children(s, *specs):
    out = []
    for fid, stamp_index in specs:
        c = ChildFIFO(s, fid, stamp_index=stamp_index)
        c.setup_node()
        c.define()
        out.append(c)
    return out


@scenario
def device_barriers_basic():
    """A barrier already satisfied passes at once; one on a stamp not yet
    reached blocks its FIFO until another FIFO publishes it; a barrier index
    of 64 (maxFIFOCount at 245) faults the command with "Invalid stamp
    index"; the root FIFO can wait on a child."""
    s = Scenario("device_barriers_basic", stamp_slots=3, doc=device_barriers_basic.__doc__)
    f = s.bring_up()
    a, b = children(s, (1, 1), (2, 2))
    b.submit(abi.OP_NOP)
    a.submit(abi.OP_NOP, barriers=[(2, 1)], label="satisfied barrier")
    st = a.cmd(abi.OP_NOP, barriers=[(2, 3)])
    a.doorbell()
    a.wait_stamp(st, timeout_ms=300, label="blocked on 2:3")
    b.submit(abi.OP_NOP)
    a.wait_stamp(st, timeout_ms=300, label="still blocked on 2:3")
    b.submit(abi.OP_NOP)
    a.wait_stamp(st, label="released at 2:3")
    s.events("child barriers")
    rst = f.cmd(abi.OP_NOP, barriers=[(1, 3)])
    f.doorbell()
    f.wait_stamp(rst, timeout_ms=300, label="root blocked on 1:3")
    a.submit(abi.OP_NOP)
    f.wait_stamp(rst, label="root released")
    s.events("root barrier")
    a.submit(abi.OP_NOP, barriers=[(64, 0)], fault=True, label="barrier index 64")
    s.r(abi.R_FAULT_MASK)
    s.dump(a.node_gpa(0), 0x14, "child 1's node after the fault")
    s.events("index 64")
    s.dump(s.fifo.base_gpa(), 12, "stamp slots")
    return s


@scenario
def device_barriers_circular():
    """Child 1 waits on child 2's stamp; child 2 then waits on child 1's: the
    chain returns to child 2 and its command faults with "Circular wait".
    Child 2 resumed, child 1 is released by it."""
    s = Scenario("device_barriers_circular", stamp_slots=3, doc=device_barriers_circular.__doc__)
    s.bring_up()
    a, b = children(s, (1, 1), (2, 2))
    st = a.cmd(abi.OP_NOP, barriers=[(2, 2)])
    a.doorbell()
    a.wait_stamp(st, timeout_ms=300, label="child 1 blocked on 2:2")
    b.submit(abi.OP_NOP, barriers=[(1, 1)], fault=True, label="child 2 waits on 1:1")
    s.events("circular")
    s.dump(b.node_gpa(0), 0x14, "child 2's node")
    s.r(abi.R_FAULT_MASK)
    s.w(abi.R_RESUME_CHILD, 2)
    b.submit(abi.OP_NOP)
    a.wait_stamp(st, label="child 1 released")
    s.events("released")
    s.dump(s.fifo.base_gpa(), 12, "stamp slots")
    return s


@scenario
def device_barriers_report():
    """A barrier blocked for longer than 2 s logs and posts a stamp-wait
    report, then keeps waiting; a resume of a FIFO that is not faulted
    (0x1028) posts one at once. The reports walk the waiting FIFOs (child 1,
    the root) and every FIFO's pending commands, and change nothing a guest
    sees: the FIFOs are released by the stamps they wait for. A waiting FIFO
    whose stamp index names no FIFO is skipped."""
    s = Scenario("device_barriers_report", stamp_slots=4, doc=device_barriers_report.__doc__)
    f = s.bring_up()
    a, b, c = children(s, (1, 1), (2, 2), (4, 3))
    st_a = a.cmd(abi.OP_NOP, barriers=[(2, 1)])
    a.doorbell()
    st_c = c.cmd(abi.OP_NOP, barriers=[(2, 2)])
    c.doorbell()
    rst = f.cmd(abi.OP_NOP, barriers=[(1, 1)])
    f.doorbell()
    a.wait_stamp(st_a, timeout_ms=2600, label="child 1 blocked past the 2 s report")
    s.w(abi.R_RESUME_CHILD, 5)
    s.events("blocked, reported")
    s.fault_state("blocked")
    b.submit(abi.OP_NOP)
    a.wait_stamp(st_a, label="child 1 released")
    f.wait_stamp(rst, label="root released")
    b.submit(abi.OP_NOP)
    c.wait_stamp(st_c, label="child 4 released")
    s.events("released")
    s.dump(s.fifo.base_gpa(), 16, "stamp slots")
    return s


@scenario
def device_barriers_report_stamp_index_mismatch_dies():
    """A stamp-wait report while child 1 (stamp index 2) waits: the report
    looks the waiter up by stamp index, finds child 2 in the device's child
    FIFO slot 2, whose stamp index is 5, and asserts: the host dies at the
    0x1028 write that asked for the report."""
    s = Scenario("device_barriers_report_stamp_index_mismatch_dies", stamp_slots=6,
                 doc=device_barriers_report_stamp_index_mismatch_dies.__doc__)
    s.bring_up()
    a, b = children(s, (1, 2), (2, 5))
    st = a.cmd(abi.OP_NOP, barriers=[(5, 1)])
    a.doorbell()
    a.wait_stamp(st, timeout_ms=300, label="child 1 blocked on 5:1")
    s.events("blocked")
    s.w(abi.R_RESUME_CHILD, 3)
    return s


@scenario
def device_barriers_pause_while_blocked():
    """pause while a child is blocked on a barrier: the quiesce wakes the
    wait, which gives up; unpause restarts the child, which waits again
    and runs once the stamp arrives."""
    s = Scenario("device_barriers_pause_while_blocked", stamp_slots=3, doc=device_barriers_pause_while_blocked.__doc__)
    s.bring_up()
    a, b = children(s, (1, 1), (2, 2))
    st = a.cmd(abi.OP_NOP, barriers=[(2, 1)])
    a.doorbell()
    a.wait_stamp(st, timeout_ms=300, label="child 1 blocked on 2:1")
    s.api("pause")
    s.events("paused")
    s.dump(a.node_gpa(0), 0x14, "child 1's node, paused")
    s.api("unpause")
    a.wait_stamp(st, timeout_ms=300, label="child 1 still blocked")
    b.submit(abi.OP_NOP)
    a.wait_stamp(st, label="child 1 released")
    s.events("released")
    s.dump(a.node_gpa(0), 0x14, "child 1's node")
    return s


@scenario
def device_barriers_chain_pending():
    """DeviceCore::add_wait's walk stops at a FIFO that is itself waiting when
    that FIFO has already pushed the awaited stamp: child 2 faults (stamp 1),
    runs a NOP (stamp 2, pushed, held behind the fault), and blocks on child 3;
    child 1 then waits for child 2's stamp 2, which is no cycle. Resuming child
    2 publishes 2 and releases child 1; child 3's stamp releases child 2."""
    s = Scenario("device_barriers_chain_pending", stamp_slots=4, doc=device_barriers_chain_pending.__doc__)
    s.bring_up()
    a, b, c = children(s, (1, 1), (2, 2), (3, 3))
    b.cmd(99)
    b.cmd(abi.OP_NOP)
    st_b = b.cmd(abi.OP_NOP, barriers=[(3, 1)])
    b.doorbell()
    s.wait(gpa=b.node_gpa(8), ne=0xFFFFFFFF, label="child 2 faulted")
    s.wait(gpa=b.node_gpa(4), eq=b.written - abi.HDR.size - abi.BARRIER.size, label="child 2 at its barrier")
    st_a = a.cmd(abi.OP_NOP, barriers=[(2, 2)])
    a.doorbell()
    a.wait_stamp(st_a, timeout_ms=300, label="child 1 blocked on 2:2")
    s.events("child 2 faulted and blocked, child 1 blocked")
    s.dump(s.fifo.base_gpa(), 16, "stamp slots")
    s.w(abi.R_RESUME_CHILD, 2)
    a.wait_stamp(st_a, label="child 1 released")
    b.wait_stamp(st_b, timeout_ms=300, label="child 2 still blocked on 3:1")
    c.submit(abi.OP_NOP)
    b.wait_stamp(st_b, label="child 2 released")
    s.events("released")
    s.dump(s.fifo.base_gpa(), 16, "stamp slots")
    return s


@scenario
def device_barriers_report_pending():
    """A stamp-wait report (asked for by 0x1028 on a FIFO that is not
    faulted) while the root and child 1 each hold commands behind a faulted
    head: each FIFO's pending list is non-empty and logged. Nothing a guest
    sees changes; both resume."""
    s = Scenario("device_barriers_report_pending", stamp_slots=2, doc=device_barriers_report_pending.__doc__)
    f = s.bring_up()
    (a,) = children(s, (1, 1))
    a.cmd(99)
    st_a = a.cmd(abi.OP_NOP)
    a.doorbell()
    s.wait(gpa=a.node_gpa(4), eq=a.written, label="child 1 consumed both")
    f.cmd(99)
    st = f.cmd(abi.OP_NOP)
    f.doorbell()
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written, label="root consumed both")
    s.wait(mmio=abi.R_FAULT_OFFSET, mmio_ne=0xFFFFFFFF, label="root faulted")
    s.events("both faulted")
    s.w(abi.R_RESUME_CHILD, 5)
    s.events("reported")
    s.fault_state("reported")
    s.w(abi.R_RESUME_CHILD, 1)
    s.w(abi.R_RESUME_ROOT, 0)
    a.wait_stamp(st_a)
    f.wait_stamp(st)
    s.events("resumed")
    s.fault_state("resumed")
    return s
