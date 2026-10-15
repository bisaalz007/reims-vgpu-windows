"""Stamps and faults: the pending-stamp ring, how stamps publish, and how
faults stop and resume them (docs/testing.md, section 6.6).

Every dispatched command pushes (stamp value, opcode) into a 128-slot ring.
Completion marks the first pending slot with its value; the pump publishes
the value of the last contiguously completed slot from the head, stores it
in the FIFO's stamp slot, ORs 1 << stampIndex into 0x1018 and raises the
interrupt, under one lock. A faulted head stops the pump, records its
offset (0x103c) and raises a fault interrupt; 0x1024 treats the faulted
head as signalled.
"""
from pgtest import abi
from pgtest.fifo_util import stamps, track_stamps
from pgtest.registry import scenario
from pgtest.scenario import Scenario


@scenario
def fifo_stamp_values():
    """A stamp is the command's own value, published as is: values going
    backwards (5 then 3), 0xffffffff, 0, and the same value twice in a row
    (the first completed before the second is pushed) each land in slot 0,
    with one interrupt each, the slot already holding the value the
    interrupt announces."""
    s = Scenario("fifo_stamp_values", doc=fifo_stamp_values.__doc__)
    f = s.bring_up()
    track_stamps(s, 1)
    for n, v in enumerate((5, 3, 0xFFFFFFFF, 0, 7, 7)):
        f.cmd(abi.OP_NOP, stamp=v)
        f.doorbell()
        s.wait(gpa=f.stamp_gpa(), eq=v, irqs=n + 1, label="stamp 0x%x" % v)
        s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written)
        s.events("stamp 0x%x" % v, irq_stamps=True)
        s.r(abi.R_IRQ_STAMP, "0x1018")
    return s


@scenario
def fifo_stamp_publish_in_order():
    """Stamps publish in command order: behind a faulted head, three NOPs
    complete but publish nothing; 0x1024 then publishes the last of them in
    one interrupt. A second faulted command behind the first faults again
    as soon as the first is resumed: the stamp before it is published (one
    interrupt), then its fault interrupt, with 0x103c at its offset."""
    s = Scenario("fifo_stamp_publish_in_order", doc=fifo_stamp_publish_in_order.__doc__)
    f = s.bring_up()
    track_stamps(s, 1)
    f.cmd(99)
    for _ in range(3):
        last = f.cmd(abi.OP_NOP)
    f.doorbell()
    s.wait_fault()
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written)
    s.events("faulted head", irq_stamps=True)
    s.fault_state("faulted at 0")
    stamps(s, 1, "nothing published")
    s.resume_root()
    f.wait_stamp(last)
    s.events("one interrupt publishes stamp 4", irq_stamps=True)
    s.fault_state("resumed")
    a = f.cmd(99)
    f.cmd(abi.OP_NOP)
    f.cmd(99)
    c = f.cmd(abi.OP_NOP)
    f.doorbell()
    s.wait_fault()
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written)
    s.fault_state("first of two faults")
    s.events("first fault", irq_stamps=True)
    s.resume_root()
    f.wait_stamp(a + 1)
    s.wait_fault()
    s.events("stamp 6, then the second fault", order=True, irq_stamps=True)
    s.fault_state("second of two faults")
    s.resume_root()
    f.wait_stamp(c)
    s.events("stamp 8", irq_stamps=True)
    s.fault_state("resumed twice")
    # Left faulted: teardown's quiesced pump publishes past it.
    f.cmd(99)
    f.doorbell()
    s.wait_fault()
    s.events("faulted at the end")
    return s


@scenario
def fifo_stamp_ring_full():
    """The pending ring holds 128 stamps. Behind a faulted head, 127 NOPs
    complete unpublished; the 129th command is read and advanced (0x100c
    counts it) and then blocks pushing its stamp, so the FIFO reads no
    further. 0x1024 publishes the 128 in one interrupt, which frees the
    ring: the blocked command and the one after it then publish one
    interrupt each."""
    s = Scenario("fifo_stamp_ring_full", doc=fifo_stamp_ring_full.__doc__)
    f = s.bring_up()
    track_stamps(s, 1)
    f.cmd(99)
    for _ in range(127):
        f.cmd(abi.OP_NOP)
    f.cmd(abi.OP_NOP)
    last = f.cmd(abi.OP_NOP)
    f.doorbell()
    s.wait_fault()
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=129 * 12, label="129 commands read")
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=130 * 12, timeout_ms=500, label="the 130th is not read")
    s.fault_state("full")
    stamps(s, 1, "nothing published")
    s.events("fault")
    s.resume_root()
    f.wait_stamp(last)
    s.r(abi.R_FIFO_READ)
    s.events("128, 129, 130", irq_stamps=True)
    return s


@scenario
def fifo_stamp_fault_resume_registers():
    """0x102c reads 1 while any FIFO is faulted and is not cleared by
    reading; 0x1018 is read-to-clear; 0x1024 with nothing faulted changes
    nothing and raises no interrupt ("Stamp at head isn't faulted")."""
    s = Scenario("fifo_stamp_fault_resume_registers", doc=fifo_stamp_fault_resume_registers.__doc__)
    f = s.bring_up()
    track_stamps(s, 1)
    s.resume_root()
    st = s.submit(abi.OP_NOP)
    s.wait(gpa=f.stamp_gpa(), ge=st, irqs=1)
    s.r(abi.R_IRQ_STAMP, "0x1018")
    s.r(abi.R_IRQ_STAMP, "0x1018 again")
    s.resume_root()
    s.events("0x1024 while healthy", irq_stamps=True)
    s.fault_state("healthy")
    st = f.cmd(99)
    f.doorbell()
    s.wait_fault()
    for i in range(3):
        s.r(abi.R_FAULT_MASK, "0x102c read %d" % i)
    s.r(abi.R_IRQ_STAMP, "0x1018 after a fault")
    s.resume_root()
    f.wait_stamp(st)
    s.wait(gpa=f.stamp_gpa(), ge=st, irqs=3)
    s.r(abi.R_IRQ_STAMP, "0x1018 after the resume")
    s.resume_root()
    s.events("resumed, then 0x1024 again", irq_stamps=True)
    s.fault_state("healthy again")
    return s


def duplicate(name, second, doc, third=None):
    """Behind a faulted head (stamp 1), a NOP with stamp 5 completes and
    waits unpublished; then `second` with stamp 5 again."""
    s = Scenario(name, doc=doc)
    f = s.bring_up()
    track_stamps(s, 1)
    f.cmd(99, stamp=1)
    f.cmd(abi.OP_NOP if third is None else third, stamp=5)
    f.doorbell()
    s.wait_fault()
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written)
    s.events("faulted head, stamp 5 pending")
    f.cmd(second, stamp=5)
    f.doorbell()
    # The assert fires on the FIFO's thread; this wait holds the scenario
    # at a fixed op until it does.
    s.wait(gpa=f.stamp_gpa(), eq=0x99, timeout_ms=10000, label="the host dies here")
    return s


@scenario
def fifo_stamp_duplicate_signal_dies():
    """Two commands with the same stamp value in flight: completing the
    second finds the first, already signalled, and asserts: the host
    dies at the doorbell."""
    return duplicate("fifo_stamp_duplicate_signal_dies", abi.OP_NOP, fifo_stamp_duplicate_signal_dies.__doc__)


@scenario
def fifo_stamp_duplicate_fault_dies():
    """Faulting a command whose stamp value an earlier, signalled command in
    flight shares asserts: the host dies."""
    return duplicate("fifo_stamp_duplicate_fault_dies", 99, fifo_stamp_duplicate_fault_dies.__doc__)


@scenario
def fifo_stamp_duplicate_of_faulted_dies():
    """Completing a command whose stamp value an earlier, faulted command in
    flight shares asserts: the host dies."""
    return duplicate("fifo_stamp_duplicate_of_faulted_dies", abi.OP_NOP,
                     fifo_stamp_duplicate_of_faulted_dies.__doc__, third=99)
