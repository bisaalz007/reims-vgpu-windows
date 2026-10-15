"""Device paths the per-area modules leave open (docs/testing.md, sections
6.2, 6.6 and 7.4): the stamp-wait report a blocked barrier posts on its own
after 2 s, and a 0x1028 index far past the child FIFO table.

DeviceCore::wait_stamp posts a report to the device's stamp-wait report source
when a wait times out; the source's handler runs
DeviceCore::report_stamp_wait_timeout, which looks every waiting stamp index up
through DeviceCore::fifo_for_stamp_index. DeviceCore::resume_child_fifo takes
any n, and resumes a child only in the 64 child FIFO slots.
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
        s.fifos[fid] = c
        out.append(c)
    return out


@scenario
def device_gap_barrier_timeout_report_mismatch_dies():
    """The report a barrier posts by itself: child 1 (stamp index 2) blocks on
    5:1, which nothing signals; child 2 has stamp index 5. After 2 s the wait
    times out and posts the stamp-wait report, whose walk looks stamp index 2
    up, finds child 2 in the device's child FIFO slot 2 with stamp index 5, and
    asserts. No register write asks for it: the host dies inside a 4 s wait on
    child 1's stamp, which is the op the outcome names (docs/testing.md 7.4,
    which had this path read but not locked)."""
    s = Scenario("device_gap_barrier_timeout_report_mismatch_dies", stamp_slots=6,
                 doc=device_gap_barrier_timeout_report_mismatch_dies.__doc__)
    s.bring_up()
    a, b = children(s, (1, 2), (2, 5))
    st = a.cmd(abi.OP_NOP, barriers=[(5, 1)])
    a.doorbell()
    a.wait_stamp(st, timeout_ms=300, label="child 1 blocked on 5:1")
    s.events("blocked")
    a.wait_stamp(st, timeout_ms=4000, label="the 2 s report kills the host")
    s.events("not reached")
    return s


@scenario
def device_gap_barrier_timeout_report_matching():
    """The control of device_gap_barrier_timeout_report_mismatch_dies: every
    child's stamp index is its id, so the 2 s report finds each waiter where
    its index says, logs and returns. Child 1 (stamp index 1) blocks on 2:1
    past the report (a 3 s wait that times out is the behaviour); child 2
    then publishes 1 and child 1 runs. No fault, one interrupt per stamp."""
    s = Scenario("device_gap_barrier_timeout_report_matching", stamp_slots=3,
                 doc=device_gap_barrier_timeout_report_matching.__doc__)
    s.bring_up()
    a, b = children(s, (1, 1), (2, 2))
    st = a.cmd(abi.OP_NOP, barriers=[(2, 1)])
    a.doorbell()
    a.wait_stamp(st, timeout_ms=3000, label="child 1 blocked past the 2 s report")
    s.events("blocked")
    s.fault_state("blocked")
    b.submit(abi.OP_NOP)
    a.wait_stamp(st, label="child 1 released")
    s.fault_state("released")
    s.events("end")
    s.dump(s.fifo.base_gpa(), 12, "stamp slots 0-2")
    return s


@scenario
def device_gap_resume_child_huge_dies():
    """0x1028 = 0xffffffff: the resume logs that the index is past
    maxFIFOCount, clears bit 63 of the fault mask (1 << n shifts by n mod 64
    on arm64), reports the waits (the FIFO is not faulted), and resumes
    nothing: past the 64 slots there is no FIFO, as for n = 65
    (fifo_child_resume_65_dies). The device runs on; the name is kept from
    when the host died there."""
    s = Scenario("device_gap_resume_child_huge_dies", doc=device_gap_resume_child_huge_dies.__doc__)
    s.bring_up()
    s.submit(abi.OP_NOP)
    s.events("before")
    s.w(abi.R_RESUME_CHILD, 0xFFFFFFFF)
    s.events("not reached")
    return s


@scenario
def device_gap_version_fingerprint():
    """Every version docs/testing.md 6.3 lists as valid (2-6, 10, 11, 20-22,
    30, 31, 40-43, 60-62, 100-113, 200-209, 230-235, 237, 238, 245), then 0, 1
    and 236, written in turn on a device with two display ports, each with EFI
    enabled just before it: 0x1200 shows whether the version kept EFI
    (efiDisplay from 3), 0x1034 what was negotiated, 0x122c the port count as
    the guest sees it (2 at 245, else count != 0) and 0x1234 the connection
    types (masked below 245). Then the same, downwards. 234 is not a case of
    features::features_for_version: it steps down to 233, so 48 versions are
    valid, not 49."""
    s = Scenario("device_gap_version_fingerprint", descriptor={"port_count": 2},
                 doc=device_gap_version_fingerprint.__doc__)
    listed = (list(range(2, 7)) + [10, 11, 20, 21, 22, 30, 31, 40, 41, 42, 43, 60, 61, 62] + list(range(100, 114))
              + list(range(200, 210)) + list(range(230, 236)) + [237, 238, 245])
    order = listed + [0, 1, 236] + list(reversed(listed))
    for v in order:
        s.w(abi.R_EFI_ENABLE, 1)
        s.w(abi.R_VERSION, v)
        for r in (abi.R_EFI_ENABLE, abi.R_VERSION, abi.R_PORT_COUNT, abi.R_PORT_TYPES):
            s.r(r, "version %d" % v)
    s.events("end")
    return s
