"""Boundaries and orders in the device and its FIFOs: each scenario here
locks one boundary or one order, as its docstring says.
"""
import struct

from pgtest import abi
from pgtest.display_guest import display
from pgtest.fifo import ChildFIFO, RootFIFO
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.suspend_guest import blob, state
from pgtest.task import Task
from scenarios.device_barriers import children
from scenarios.suspend import resume_fresh, suspend, will_resume


@scenario
def mutgap_dev_root_fifo_start_low_bits():
    """The root FIFO's creation refuses a start with any of its low ten bits
    set: 0x401, 1 and 0x801, whose only low bit is bit 0, each leave 0x1000
    reading 0; 0x400 then enables and a NOP runs."""
    s = Scenario("mutgap_dev_root_fifo_start_low_bits", doc=mutgap_dev_root_fifo_start_low_bits.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    f = RootFIFO(s, pages=2)
    s.fifo = f
    s.fifos[0] = f
    for start in (0x401, 0x1, 0x801):
        f.enable(start=start)
        s.r(abi.R_FIFO_ENABLE, "start 0x%x" % start)
    s.events("after the refusals")
    f.enable()
    s.r(abi.R_FIFO_ENABLE, "start 0x400")
    s.submit(abi.OP_NOP)
    s.events("a NOP")
    return s


@scenario
def mutgap_dev_root_fifo_refused_then_suspend():
    """A root FIFO whose start is not below its length is refused after its
    mapping was made: the root FIFO's creation releases the mapping and
    clears its pointer, so rvg_device_finish_suspend's release of it releases
    nil. 0x1000 reads 0; in the blocks setup the mapping's task calls are
    locked; the blob says the FIFO is off."""
    s = Scenario("mutgap_dev_root_fifo_refused_then_suspend",
                 doc=mutgap_dev_root_fifo_refused_then_suspend.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    f = RootFIFO(s, pages=2)
    f.enable(start=0x8000)
    s.r(abi.R_FIFO_ENABLE, "start 0x8000, length 0x8000")
    s.events("refused")
    suspend(s)
    s.events("suspended")
    s.sweep(0x1000, 0x1040, label="registers after finishSuspend")
    return s


@scenario
def mutgap_dev_task_length_one_past():
    """A task of 0x800000001 bytes, one past 32 GiB: in-process define
    refuses it ("Task size too large") before any create_task call, and the
    define faults. The isolate setup's service has no such limit
    (tasks_define_too_large) and defines it, length not a page multiple and
    all: the root word is set and nothing faults. Nothing is submitted after
    the define."""
    s = Scenario("mutgap_dev_task_length_one_past", doc=mutgap_dev_task_length_one_past.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=4, depth=3)
    t.define(length=0x800000001, fault="either", timeout_ms=3000, label="define 0x800000001")
    s.fault_state("after the define")
    s.dump(s.pa(s.root_page, 4), 4, "root word of task 1")
    s.events("end")
    return s


@scenario
def mutgap_dev_child63_resume():
    """Child FIFO 63, the highest at version 245 (maxFIFOCount 64), survives
    a suspend and a resume on a fresh device: rvg_device_did_resume recreates
    every child whose node's running word is set, ids 1 to 63, and a NOP rung
    on 63 afterwards runs."""
    s = Scenario("mutgap_dev_child63_resume", doc=mutgap_dev_child63_resume.__doc__)
    s.bring_up()
    c = ChildFIFO(s, 63)
    c.setup_node()
    c.define()
    c.submit(abi.OP_NOP)
    s.events("before")
    suspend(s)
    resume_fresh(s)
    s.events("resumed")
    s.dump(c.node_gpa(0), abi.CHILD_NODE_SIZE, "node 63 after the resume")
    c.submit(abi.OP_NOP, timeout_ms=3000, label="a NOP on child 63 after the resume")
    s.events("end")
    s.dump(s.fifo.base_gpa() + 4 * 63, 4, "stamp slot 63")
    return s


@scenario
def mutgap_dev_child63_pause():
    """pause quiesces and drains children 1-63; unpause restarts the same 63,
    so child FIFO 63 runs a NOP rung after unpause, with its interrupt."""
    s = Scenario("mutgap_dev_child63_pause", doc=mutgap_dev_child63_pause.__doc__)
    s.bring_up()
    c = ChildFIFO(s, 63)
    c.setup_node()
    c.define()
    c.submit(abi.OP_NOP)
    s.events("before")
    s.api("pause")
    s.api("unpause")
    c.submit(abi.OP_NOP, timeout_ms=3000, label="a NOP on child 63 after unpause")
    s.events("end")
    s.dump(s.fifo.base_gpa() + 4 * 63, 4, "stamp slot 63")
    return s


@scenario
def mutgap_dev_unpause_eight_ports():
    """At the largest port count, 8, with a display created on port 0 and no
    page: unpause visits the eight ports' guest displays and no more. Pause,
    unpause and a NOP complete."""
    s = Scenario("mutgap_dev_unpause_eight_ports", descriptor={"port_count": 8}, displays=[display(0)],
                 doc=mutgap_dev_unpause_eight_ports.__doc__)
    s.bring_up()
    s.submit(abi.OP_NOP)
    s.api("pause")
    s.api("unpause")
    s.submit(abi.OP_NOP)
    s.events("end")
    return s


@scenario
def mutgap_dev_circular_then_wait():
    """After the circular wait of device_barriers_circular, the faulting
    waiter's wait is finished, so the same FIFO can wait again: child 2,
    resumed, releases child 1, then blocks on child 1's next stamp and is
    released when child 1 publishes it. No fault, no death."""
    s = Scenario("mutgap_dev_circular_then_wait", stamp_slots=3, doc=mutgap_dev_circular_then_wait.__doc__)
    s.bring_up()
    a, b = children(s, (1, 1), (2, 2))
    st = a.cmd(abi.OP_NOP, barriers=[(2, 2)])
    a.doorbell()
    a.wait_stamp(st, timeout_ms=300, label="child 1 blocked on 2:2")
    b.submit(abi.OP_NOP, barriers=[(1, 1)], fault=True, label="child 2 waits on 1:1")
    s.w(abi.R_RESUME_CHILD, 2)
    b.submit(abi.OP_NOP)
    a.wait_stamp(st, label="child 1 released")
    s.events("circular, resumed, released")
    st_b = b.cmd(abi.OP_NOP, barriers=[(1, st + 1)])
    b.doorbell()
    b.wait_stamp(st_b, timeout_ms=300, label="child 2 blocked on 1:%d" % (st + 1))
    a.submit(abi.OP_NOP)
    b.wait_stamp(st_b, label="child 2 released")
    s.events("waited again")
    s.fault_state("end")
    s.dump(s.fifo.base_gpa(), 12, "stamp slots")
    return s


@scenario
def mutgap_dev_barrier_report_once():
    """A barrier wait posts one stamp-wait report per timeout and then waits
    without a timeout until a stamp is signalled. Child 1 (stamp index 1)
    blocks on 7:1 at time 0 and is reported at 2 s, with only its own index
    waiting. At 3.5 s child 3, stamp index 2, blocks on 7:1 too, which
    signals nothing; child 2 has stamp index 7, so a report walking index 2
    would find it and assert (as
    device_gap_barrier_timeout_report_mismatch_dies). No report runs before
    child 2 publishes 7:1 at about 4.75 s, releasing both, before child 3's
    own 2 s timeout: no death. (A wait that timed out twice and reported
    twice would walk index 2.)"""
    s = Scenario("mutgap_dev_barrier_report_once", stamp_slots=8, doc=mutgap_dev_barrier_report_once.__doc__)
    s.bring_up()
    a, b, c = children(s, (1, 1), (2, 7), (3, 2))
    st_a = a.cmd(abi.OP_NOP, barriers=[(7, 1)])
    a.doorbell()
    a.wait_stamp(st_a, timeout_ms=3500, label="child 1 blocked past its 2 s report")
    st_c = c.cmd(abi.OP_NOP, barriers=[(7, 1)])
    c.doorbell()
    c.wait_stamp(st_c, timeout_ms=1250, label="child 3 blocked on 7:1")
    b.submit(abi.OP_NOP)
    a.wait_stamp(st_a, label="child 1 released")
    c.wait_stamp(st_c, label="child 3 released")
    s.events("released")
    s.fault_state("end")
    s.dump(s.fifo.base_gpa(), 32, "stamp slots")
    return s


@scenario(host=True)
def mutgap_dev_resume_info_max_key():
    """rvg_device_will_resume takes the device-info max key from the blob, so
    the next rvg_device_finish_suspend's blob carries it: a crafted blob with
    max key 2 ({1, 1}) resumed on the fresh device gives a blob with max key 2,
    the host's {1, ...} pair and length 116; the blob of a device that asked
    for max key 45, resumed on a fresh device, gives max key 45 again."""
    out = []
    s = Scenario("mutgap_dev_resume_info_max_key", host=True, doc=mutgap_dev_resume_info_max_key.__doc__)
    will_resume(s, blob(state(), [(1, 1)], info_max=2), "max key 2")
    s.api("didResume")
    suspend(s)
    out.append(s)
    s = Scenario("mutgap_dev_resume_info_max_key_45", host=True, doc=mutgap_dev_resume_info_max_key.__doc__)
    s.bring_up()
    s.submit(abi.OP_GET_DEVICE_INFO, struct.pack("<III", 45, 64, s.pfn(s.alloc_page())))
    suspend(s, slot="first")
    resume_fresh(s, slot="first")
    s.submit(abi.OP_NOP)
    suspend(s, slot="second")
    out.append(s)
    return out


@scenario(host=True)
def mutgap_dev_reset_info_max_key():
    """reset sets the device-info max key back to 1: after a device-info
    query with max key 2 and a reset, rvg_device_finish_suspend's blob has max
    key 1, no info pairs and length 108."""
    s = Scenario("mutgap_dev_reset_info_max_key", host=True, doc=mutgap_dev_reset_info_max_key.__doc__)
    s.bring_up()
    s.submit(abi.OP_GET_DEVICE_INFO, struct.pack("<III", 2, 64, s.pfn(s.alloc_page())))
    s.api("reset")
    suspend(s)
    return s
