"""Child FIFOs (docs/testing.md, section 6.7): define and delete (opcodes 48
and 49, root FIFO only), the node fields the host writes, page lists, the
wake (0x1020), faults and resume (0x1028), stamp indices, and the delete
that zeroes the next node's pageListPage.

A node is 0x14 bytes at root + 0x400 + 0x14 x (id - 1): u32 written
(guest), u32 readCount (host: 0 at define), u32 faultOffset (host:
0xffffffff at define and resume), u16 stampIndex (guest), u16 running
(host), u32 pageListPage (guest). The page list is a page of u32 guest page
numbers ending at the first 0, at most 4096; the ring is those pages,
virtually contiguous.
"""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.fifo_util import (NODE_FAULT, NODE_READ, child, child_fault, node, probe, resume_child, stamps,
                              track_stamps, wait_running)
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import Task

F, S = "fault", "signal"


@scenario
def fifo_child_define_delete():
    """Define child 1: the host sets readCount 0, faultOffset 0xffffffff and
    running 1 in its node. A NOP on it publishes stamp slot 1 and moves
    readCount. Delete: running goes to 0, and the host zeroes the u32 at
    root + 0x410 + 0x14 x id, which is node 2's pageListPage, not node 1's
    (docs/testing.md, W3). Redefining child 1 restarts its node. Child 63,
    the last, zeroes root + 0x8fc, the word before the display words."""
    s = Scenario("fifo_child_define_delete", doc=fifo_child_define_delete.__doc__)
    f = s.bring_up()
    track_stamps(s, 2)
    root = s.pa(s.root_page)
    s.u32(root + 0x400 + 0x14 + 0x10, 0xDEADBEEF)  # node 2's pageListPage
    s.u32(root + 0x8FC, 0xFEEDFACE)
    c = child(s, 1)
    wait_running(s, c)
    node(s, 1, "defined")
    st = c.submit(abi.OP_NOP)
    node(s, 1, "after a NOP")
    c.delete()
    wait_running(s, c, 0)
    node(s, 1, "deleted")
    node(s, 2, "node 2 after deleting 1")
    s.u32(c.node_gpa(4), 0x1234)  # a stale readCount: define resets it
    c.define()
    wait_running(s, c)
    # The ring still holds the first NOP (write count 0xc): the restarted
    # FIFO reads it again from 0, so readCount ends at 0xc, not 0.
    s.wait(gpa=c.node_gpa(4), eq=0xC, label="readCount after the replayed NOP")
    node(s, 1, "redefined")
    st = c.submit(abi.OP_NOP, stamp=st + 1)
    node(s, 1, "after a NOP")
    c63 = child(s, 63)
    c63.submit(abi.OP_NOP)
    c63.delete()
    s.dump(root + 0x8FC, 4, "root + 0x8fc after deleting 63")
    s.events("define, delete", irq_stamps=True)
    s.diff("root page and stamps", gpa=root, length=PAGE)
    return s


@scenario
def fifo_child_define_errors():
    """Refused defines and deletes fault on the root FIFO: define id 0, 64 and
    0xffffffff (DeviceCore::create_fifo raises: an invalid FIFO id); define an
    id already defined; define with node stampIndex 64 (Fifo::new_child_on
    makes no FIFO at stampIndex >= maxFIFOCount); a 3-byte payload for each;
    delete id 0, 64, and an id never defined (DeviceCore::delete_fifo answers
    false). A longer payload is fine. None of it disturbs child 1."""
    s = Scenario("fifo_child_define_errors", doc=fifo_child_define_errors.__doc__)
    s.bring_up()
    c1 = child(s, 1)
    u = lambda v: struct.pack("<I", v)
    for fid in (0, 64, 0xFFFFFFFF, 1):
        probe(s, abi.OP_DEFINE_CHILD_FIFO, u(fid), F, label="define %d" % fid)
    c2 = child(s, 2, define=False, stamp_index=64)
    probe(s, abi.OP_DEFINE_CHILD_FIFO, u(2), F, label="define 2 with stampIndex 64")
    node(s, 2, "node 2 after the refused define")
    probe(s, abi.OP_DEFINE_CHILD_FIFO, b"\x03\x00\x00", F, label="define, 3 bytes")
    probe(s, abi.OP_DELETE_CHILD_FIFO, b"\x01\x00\x00", F, label="delete, 3 bytes")
    for fid in (0, 64, 5):
        probe(s, abi.OP_DELETE_CHILD_FIFO, u(fid), F, label="delete %d" % fid)
    c3 = child(s, 3, define=False)
    probe(s, abi.OP_DEFINE_CHILD_FIFO, u(3) + b"\xff" * 8, S, label="define 3, 12 bytes")
    c1.submit(abi.OP_NOP)
    c3.submit(abi.OP_NOP)
    probe(s, abi.OP_DELETE_CHILD_FIFO, u(3) + b"\xff" * 4, S, label="delete 3, 8 bytes")
    s.events("define and delete errors")
    return s


@scenario
def fifo_child_version_60():
    """At version 60 maxFIFOCount is 32: child 31 can be defined, 32 cannot,
    and a node stampIndex of 32 is refused."""
    s = Scenario("fifo_child_version_60", doc=fifo_child_version_60.__doc__)
    s.bring_up(version=60)
    track_stamps(s, 32)
    c31 = child(s, 31)
    c31.submit(abi.OP_NOP)
    probe(s, abi.OP_DEFINE_CHILD_FIFO, struct.pack("<I", 32), F, label="define 32")
    child(s, 5, define=False, stamp_index=32)
    probe(s, abi.OP_DEFINE_CHILD_FIFO, struct.pack("<I", 5), F, label="define 5 with stampIndex 32")
    s.events("version 60")
    stamps(s, 32)
    return s


@scenario
def fifo_child_root_only():
    """Define (48) and delete (49) are accepted only on the root FIFO: on
    child 1 both fault, at the command's offset in child 1's ring (node
    +8), 0x102c reads 1, and 0x1028 = 1 resumes it. The ids they name are
    untouched: child 2 is never defined, child 1 keeps running."""
    s = Scenario("fifo_child_root_only", doc=fifo_child_root_only.__doc__)
    s.bring_up()
    track_stamps(s, 3)
    c1 = child(s, 1)
    child(s, 2, define=False)
    probe(s, abi.OP_DEFINE_CHILD_FIFO, struct.pack("<I", 2), F, fifo=c1, label="define 2 on child 1")
    probe(s, abi.OP_DELETE_CHILD_FIFO, struct.pack("<I", 1), F, fifo=c1, label="delete 1 on child 1")
    probe(s, abi.OP_DELETE_CHILD_FIFO, b"", F, fifo=c1, label="delete, no payload, on child 1")
    probe(s, abi.OP_NOP, expect=S, fifo=c1, label="NOP on child 1")
    node(s, 1)
    node(s, 2)
    s.events("root-only commands", irq_stamps=True)
    return s


@scenario
def fifo_child_commands():
    """Commands on a child FIFO frame and dispatch as on the root: an
    invalid opcode, a deprecated one and a short payload fault at their
    ring offsets (node +8; 0x103c stays 0xffffffff); Debug, GetDeviceInfo
    with maxCount 0 on a real page, and display commands 7 and 10 signal; a
    barrier count over 64 and an underflowing barrier count fault. The
    root-only guard aside, nothing is specific to the root."""
    s = Scenario("fifo_child_commands", doc=fifo_child_commands.__doc__)
    s.bring_up()
    track_stamps(s, 2)
    c = child(s, 1)
    page = s.alloc_page()
    probe(s, 99, b"", F, fifo=c, label="invalid 99")
    s.r(abi.R_FAULT_OFFSET, "0x103c while a child is faulted")
    probe(s, 3, b"", F, fifo=c, label="deprecated 3")
    probe(s, abi.OP_GET_DEVICE_INFO, bytes(11), F, fifo=c, label="device info, 11 bytes")
    probe(s, abi.OP_DEBUG, b"hello\0", S, fifo=c, label="debug")
    probe(s, abi.OP_GET_DEVICE_INFO, struct.pack("<III", 0, 0, s.pfn(page)), S, fifo=c, label="device info, count 0")
    probe(s, abi.OP_DISPLAY_TRANSACTION3, bytes(36), S, fifo=c, label="display 7, no display")
    probe(s, abi.OP_DISPLAY_SET_PROPERTIES, bytes(3), S, fifo=c, label="display 10, 3 bytes")
    probe(s, abi.OP_DISPLAY_ACK, bytes(8), F, fifo=c, label="display 2, no display")
    # A barrier count over 64: the child consumes only the header, so the
    # 528 barrier bytes (zeros) are skipped as 44 headers of length 0 before
    # the NOP after them runs.
    c.stamp += 1
    c.raw(abi.HDR.pack(abi.OP_NOP, 66, 12 + 66 * 8, c.stamp) + bytes(66 * 8))
    nop = c.cmd(abi.OP_NOP)
    c.doorbell()
    child_fault(s, c)
    # The FIFO thread goes on reading the 44 empty headers and the NOP while
    # the fault publishes: wait for its read count before the dump, here and
    # at the next fault (docs/testing.md 5.4).
    s.wait(gpa=c.node_gpa(NODE_READ), eq=c.written, label="child 1 read its ring")
    node(s, 1, "66 barriers")
    resume_child(s, 1)
    c.wait_stamp(nop)
    # One barrier in a 12-byte command: its barrier does not fit.
    c.stamp += 1
    c.raw(abi.HDR.pack(abi.OP_NOP, 1, 12, c.stamp))
    nop = c.cmd(abi.OP_NOP)
    c.doorbell()
    child_fault(s, c)
    s.wait(gpa=c.node_gpa(NODE_READ), eq=c.written, label="child 1 read its ring again")
    node(s, 1, "a barrier in a 12-byte command")
    resume_child(s, 1)
    c.wait_stamp(nop)
    node(s, 1, "resumed")
    s.events("child commands", irq_stamps=True)
    return s


@scenario
def fifo_child_bad_length():
    """A header of length 0 or over the ring on a child FIFO is skipped
    without a stamp, as on the root, but the child counts what it read
    locally: nothing is parsed past the node's `written`, and the next
    doorbell's NOP runs. The node's readCount then includes the skipped
    header."""
    s = Scenario("fifo_child_bad_length", doc=fifo_child_bad_length.__doc__)
    s.bring_up()
    track_stamps(s, 2)
    c = child(s, 1)
    c.submit(abi.OP_NOP)
    for n in (0, PAGE + 1):
        c.raw(abi.HDR.pack(abi.OP_NOP, 0, n, 0x99))
        # Stale bytes past `written`: a NOP the child must not run.
        c.ring_write(c.written, abi.HDR.pack(abi.OP_NOP, 0, 12, 0x77))
        c.doorbell()
        st = c.cmd(abi.OP_NOP)
        c.doorbell()
        c.wait_stamp(st)
        s.wait(gpa=c.node_gpa(4), eq=c.written)
        node(s, 1, "after a length-%d header and a NOP" % n)
    s.events("bad lengths", irq_stamps=True)
    return s


@scenario
def fifo_child_page_lists():
    """Rings over page lists: one page; three contiguous pages; three
    scattered pages, each followed in guest memory by a page of 0xff. On the
    scattered ring, a barrier whose index straddles each page boundary and
    the ring's end reads 0 (a read of the physically next page would give
    0xffff.... and fault), and a header straddles the end. readCount runs on
    past the ring's size."""
    s = Scenario("fifo_child_page_lists", doc=fifo_child_page_lists.__doc__)
    s.bring_up()
    track_stamps(s, 4)
    one = child(s, 1, pages=1)
    three = child(s, 2, pages=3)
    sc = child(s, 3, pages=3, scatter=True)
    for p in sc.pages:
        s.fill(s.pa(p + 1), PAGE, 0xFF)
    one.submit(abi.OP_NOP)
    three.submit(abi.OP_NOP, bytes(2 * PAGE))
    three.submit(abi.OP_NOP)

    def pad_to(c, off):
        n = (off - c.written % c.ring_size()) % c.ring_size()
        return c.submit(abi.OP_NOP, bytes(n - 12))

    for boundary in (PAGE, 2 * PAGE, 3 * PAGE):
        pad_to(sc, boundary - 14)
        sc.submit(abi.OP_NOP, barriers=[(0, 1)], label="index split at 0x%x" % boundary)
    pad_to(sc, 3 * PAGE - 6)
    sc.submit(abi.OP_NOP, label="a header split at the ring's end")
    for c in (one, three, sc):
        node(s, c.fid)
    s.fault_state("no fault")
    s.events("page lists", irq_stamps=True)
    stamps(s, 4)
    return s


@scenario
def fifo_child_page_list_empty():
    """A page list whose first entry is 0 maps nothing: the ring's mapping
    (Fifo::map_ring) fails, so the FIFO never starts (running stays 0) and a
    woken command never runs; yet the define signals, because
    DeviceCore::create_fifo ignores Fifo::start's result. Deleting it works."""
    s = Scenario("fifo_child_page_list_empty", doc=fifo_child_page_list_empty.__doc__)
    s.bring_up()
    track_stamps(s, 2)
    c = child(s, 1, page_list=[])
    node(s, 1, "defined, not running")
    st = c.cmd(abi.OP_NOP)
    c.doorbell()
    c.wait_stamp(st, timeout_ms=500, label="never runs")
    c.delete()
    node(s, 1, "deleted")
    s.events("empty page list")
    return s


@scenario
def fifo_child_page_list_unmappable():
    """A page list naming a page outside guest RAM fails to map: the FIFO is
    defined (the define signals) but never runs, as with an empty list. A
    pageListPage outside guest RAM does the same."""
    s = Scenario("fifo_child_page_list_unmappable", doc=fifo_child_page_list_unmappable.__doc__)
    s.bring_up()
    track_stamps(s, 3)
    c1 = child(s, 1, page_list=[0x7FFFF])
    node(s, 1, "a page outside RAM")
    c2 = child(s, 2, define=False)
    s.u32(c2.node_gpa(0x10), 0x7FFFF)
    c2.define()
    node(s, 2, "a page list outside RAM")
    for c in (c1, c2):
        st = c.cmd(abi.OP_NOP)
        c.doorbell()
        c.wait_stamp(st, timeout_ms=500, label="child %d never runs" % c.fid)
    c1.delete()
    c2.delete()
    s.events("unmappable")
    return s


@scenario
def fifo_child_page_list_full():
    """A page list with no 0 in its 4096 entries: the scan stops at 4096, here
    every entry naming the same page, a 64 MiB ring. With a memory map (map,
    isolate) it maps and runs. In the blocks setup the root task's 16 MiB range
    allocator cannot place it (DeviceCore::new_range_mapping): the define still
    signals, and the FIFO never runs. Deleting it works in both."""
    s = Scenario("fifo_child_page_list_full", doc=fifo_child_page_list_full.__doc__)
    s.bring_up()
    track_stamps(s, 2)
    ring = s.alloc_page()
    c = child(s, 1, page_list=[s.pfn(ring)] * 4096)
    c.pages = [ring] * 4096  # the ring the guest writes: one page, 4096 times
    node(s, 1, "defined")
    st = c.cmd(abi.OP_NOP)
    c.doorbell()
    c.wait_stamp(st, timeout_ms=1000, label="a NOP on the 4096-page ring")
    node(s, 1, "after the NOP")
    c.delete()
    s.events("full page list")
    return s


@scenario
def fifo_child_prefilled():
    """Commands already in a child's ring, with `written` stored in its node,
    run as soon as it is defined, without a wake (0x1020): the new FIFO reads
    the node's `written` before it first waits. The child may publish before
    or after the root publishes the define's stamp, so the interrupts are
    counted, and the stamp slots read, but their snapshots are not
    ordered."""
    s = Scenario("fifo_child_prefilled", doc=fifo_child_prefilled.__doc__)
    s.bring_up()
    track_stamps(s, 2)
    c = child(s, 1, define=False)
    for _ in range(3):
        st = c.cmd(abi.OP_NOP)
    s.u32(c.node_gpa(0), c.written)
    c.define()
    c.wait_stamp(st)
    node(s, 1)
    s.events("prefilled")
    s.dump(s.fifo.stamp_gpa(), 8, "stamp slots 0 and 1")
    return s


@scenario
def fifo_child_wake():
    """0x1020 wakes child n: 0 and 64 or more only log, and an id with no
    FIFO does nothing; none of them changes guest memory or raises an
    interrupt. Then child 1's commands across several doorbells, each a
    wake."""
    s = Scenario("fifo_child_wake", doc=fifo_child_wake.__doc__)
    s.bring_up()
    track_stamps(s, 2)
    c = child(s, 1)
    s.events("defined")
    for n in (0, 64, 0xFFFFFFFF, 5):
        s.w(abi.R_WAKE_CHILD, n)
    st = s.submit(abi.OP_NOP)
    s.events("stray wakes, then a root NOP")
    s.diff("the stray wakes wrote nothing")
    for i in range(3):
        st = c.submit(abi.OP_NOP, bytes(4 * i))
    node(s, 1)
    s.events("three doorbells", irq_stamps=True)
    return s


@scenario
def fifo_child_fault_resume():
    """A child fault: node +8 holds the ring offset, 0x102c reads 1, 0x103c
    stays 0xffffffff, the stamp waits. 0x1028 = 1 resumes it: node +8 goes
    back to 0xffffffff and the stamp publishes. With the root and child 1
    both faulted, resuming one raises an interrupt because the other's fault
    remains; 0x102c reads 1 until both are resumed, in either order. A
    faulted child that is deleted clears its fault bit, and the drain
    publishes its faulted stamp (the pending-stamp queue's pump,
    quiesced)."""
    s = Scenario("fifo_child_fault_resume", doc=fifo_child_fault_resume.__doc__)
    f = s.bring_up()
    track_stamps(s, 3)
    c = child(s, 1)
    c.submit(abi.OP_NOP)
    probe(s, 99, b"", F, fifo=c, label="child 1 invalid")
    node(s, 1, "resumed")
    s.events("one child fault", irq_stamps=True)
    rs = f.cmd(99)
    f.doorbell()
    s.wait_fault()
    cs = c.cmd(99)
    c.doorbell()
    s.wait(gpa=c.node_gpa(NODE_FAULT), ne=0xFFFFFFFF)
    s.events("root and child 1 faulted", irq_stamps=True)
    resume_child(s, 1)
    c.wait_stamp(cs)
    s.events("child 1 resumed: the root's fault raises again", irq_stamps=True)
    s.fault_state("root still faulted")
    s.resume_root()
    f.wait_stamp(rs)
    s.events("root resumed", irq_stamps=True)
    s.fault_state("both resumed")
    rs = f.cmd(99)
    f.doorbell()
    s.wait_fault()
    cs = c.cmd(99)
    c.doorbell()
    s.wait(gpa=c.node_gpa(NODE_FAULT), ne=0xFFFFFFFF)
    s.events("both faulted again", irq_stamps=True)
    s.resume_root()
    f.wait_stamp(rs)
    s.events("root resumed first: child 1's fault raises again", irq_stamps=True)
    s.fault_state("child 1 still faulted")
    resume_child(s, 1)
    c.wait_stamp(cs)
    s.fault_state("both resumed again")
    # Deleting a faulted child: the drain publishes its faulted stamp.
    c2 = child(s, 2)
    c2.cmd(99)
    c2.doorbell()
    child_fault(s, c2)
    c2.delete()
    s.fault_state("child 2 deleted while faulted")
    s.events("deleted while faulted", irq_stamps=True)
    return s


@scenario
def fifo_child_stamp_index_mismatch():
    """A child's stamp slot and fault bit come from its node's stampIndex,
    while 0x1028 takes the FIFO id: child 1 with stampIndex 3 publishes in
    slot 3, and its fault sets bit 3. 0x1028 = 1 clears bit 1 (not set) and
    resumes child 1, whose stamp then publishes, but bit 3 stays: 0x102c
    still reads 1 and the resume raises an interrupt. 0x1028 = 3 clears bit
    3 and resumes no FIFO."""
    s = Scenario("fifo_child_stamp_index_mismatch", doc=fifo_child_stamp_index_mismatch.__doc__)
    s.bring_up()
    track_stamps(s, 4)
    c = child(s, 1, stamp_index=3)
    c.submit(abi.OP_NOP)
    s.events("child 1's NOP in slot 3", irq_stamps=True)
    st = c.cmd(99)
    c.doorbell()
    child_fault(s, c)
    s.events("fault")
    resume_child(s, 1)
    c.wait_stamp(st)
    s.events("0x1028 = 1", irq_stamps=True)
    s.fault_state("bit 3 still set")
    node(s, 1)
    resume_child(s, 3)
    s.wait(mmio=abi.R_FAULT_MASK, mmio_eq=0)
    s.events("0x1028 = 3", irq_stamps=True)
    s.fault_state("cleared")
    stamps(s, 4)
    return s


@scenario
def fifo_child_stamp_index_alias():
    """The stamp interrupt bit is 1U << stampIndex, a 32-bit shift:
    stampIndex 33 sets bit 1 of 0x1018, 63 sets bit 31, 32 sets bit 0,
    like the root. Each child's stamp lands in its own slot. A child with
    stampIndex 0 shares the root's slot 0 and bit 0 (a node's stampIndex
    is not checked against 0 or others)."""
    s = Scenario("fifo_child_stamp_index_alias", doc=fifo_child_stamp_index_alias.__doc__)
    f = s.bring_up()
    track_stamps(s, 64)
    irqs = 4  # the four defines below
    cs = [child(s, fid, stamp_index=si) for fid, si in ((1, 33), (2, 63), (3, 32), (4, 0))]
    s.events("defined")
    s.r(abi.R_IRQ_STAMP, "0x1018 after the defines")
    for c in cs:
        st = c.cmd(abi.OP_NOP, stamp=0x100 + c.fid)
        c.doorbell()
        irqs += 1
        s.wait(gpa=c.stamp_gpa(), eq=st, irqs=irqs, label="child %d, stampIndex %d" % (c.fid, c.stamp_index))
        s.r(abi.R_IRQ_STAMP, "0x1018 after child %d (stampIndex %d)" % (c.fid, c.stamp_index))
    st = s.submit(abi.OP_NOP, stamp=0x200)
    s.r(abi.R_IRQ_STAMP, "0x1018 after a root NOP")
    s.events("aliased bits")
    stamps(s, 64)
    return s


@scenario
def fifo_child_delete_with_task():
    """Deleting a child FIFO tells every task (Task::fifo_deleted, in the
    service under isolation) and waits for each to answer before the delete
    signals."""
    s = Scenario("fifo_child_delete_with_task", doc=fifo_child_delete_with_task.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=4)
    t.define()
    t2 = Task(s, tid=2, pages=4)
    t2.define()
    c = child(s, 1)
    c.submit(abi.OP_NOP)
    c.delete()
    node(s, 1)
    t.delete()
    t2.delete()
    s.events("deleted with two tasks")
    return s


@scenario
def fifo_child_delete_delay_hangs():
    """Deleting a child FIFO drains its pending stamps; a Delay on it never
    completes, so opcode 49 hangs the root FIFO: its stamp never publishes,
    and teardown then hangs (the watchdog)."""
    s = Scenario("fifo_child_delete_delay_hangs", watchdog_s=8, doc=fifo_child_delete_delay_hangs.__doc__)
    f = s.bring_up()
    c = child(s, 1)
    c.cmd(abi.OP_DELAY, bytes(8))
    c.doorbell()
    s.wait(gpa=c.node_gpa(4), eq=c.written, label="the Delay read")
    st = f.cmd(abi.OP_DELETE_CHILD_FIFO, struct.pack("<I", 1))
    f.doorbell()
    f.wait_stamp(st, timeout_ms=500, label="the delete never completes")
    s.events("hung")
    return s


@scenario
def fifo_child_resume_64():
    """0x1028 = 64 is past the 64 child slots: the mask shift 1ULL << 64
    wraps to bit 0, the root's, and no FIFO is resumed. With the root
    faulted, it clears the root's bit, so 0x102c reads 0; the root is still
    stuck until 0x1024."""
    s = Scenario("fifo_child_resume_64", doc=fifo_child_resume_64.__doc__)
    f = s.bring_up()
    track_stamps(s, 1)
    resume_child(s, 64)
    s.fault_state("0x1028 = 64 while healthy")
    st = f.cmd(99)
    f.doorbell()
    s.wait_fault()
    resume_child(s, 64)
    s.fault_state("0x1028 = 64 while the root is faulted")
    f.wait_stamp(st, timeout_ms=500, label="the root is still stuck")
    s.resume_root()
    f.wait_stamp(st)
    s.fault_state("0x1024")
    s.events("resume 64", irq_stamps=True)
    return s


@scenario
def fifo_child_resume_65_dies():
    """0x1028 = 65 names no child FIFO: past the 64 slots there is none. It
    clears bit 1 of the fault mask (the shift is by n mod 64) and resumes
    nothing, as a wake ignores the id; the device runs on (docs/testing.md,
    section 6.2). The name is kept from when the host died there."""
    s = Scenario("fifo_child_resume_65_dies", doc=fifo_child_resume_65_dies.__doc__)
    s.bring_up()
    s.submit(abi.OP_NOP)
    resume_child(s, 65)
    s.fault_state("not reached")
    return s


@scenario
def fifo_child_report_mismatch_dies():
    """A report of stamp waits (0x1028 on a FIFO that is not faulted) looks
    each waiting stampIndex up as a child id and asserts that the child
    there has that stampIndex. Child 1 with stampIndex 3 blocks on a
    barrier (registering stampIndex 3 as waiting; the scenario waits 500 ms
    for it), child 3 has stampIndex 5: 0x1028 = 2 then kills the host."""
    s = Scenario("fifo_child_report_mismatch_dies", doc=fifo_child_report_mismatch_dies.__doc__)
    s.bring_up()
    c1 = child(s, 1, stamp_index=3)
    child(s, 3, stamp_index=5)
    st = c1.cmd(abi.OP_NOP, barriers=[(7, 1)])
    c1.doorbell()
    c1.wait_stamp(st, timeout_ms=500, label="child 1 blocked on slot 7")
    resume_child(s, 2)
    s.fault_state("not reached")
    return s


@scenario
def fifo_child_version_lowered():
    """The stamp index check runs at signal time against the current
    maxFIFOCount, with `>`. Children with stampIndex 40 and 32 are defined at
    version 245 (maxFIFOCount 64); after 0x1034 = 60 (maxFIFOCount 32),
    stampIndex 40's stamps are dropped ("Invalid stamp index"), with no
    interrupt, while stampIndex 32 still publishes, as 32 > 32 is false.
    Barriers on index 40 now fault. Back at 245, 40 publishes again: the
    dropped stamp was consumed, so the slot jumps."""
    s = Scenario("fifo_child_version_lowered", doc=fifo_child_version_lowered.__doc__)
    f = s.bring_up()
    track_stamps(s, 41)
    c40 = child(s, 1, stamp_index=40)
    c32 = child(s, 2, stamp_index=32)
    c40.submit(abi.OP_NOP)
    c32.submit(abi.OP_NOP)
    s.w(abi.R_VERSION, 60)
    s.r(abi.R_VERSION)
    s.events("at 245")
    st = c40.cmd(abi.OP_NOP)
    c40.doorbell()
    s.wait(gpa=c40.node_gpa(4), eq=c40.written, label="child 1 read its NOP")
    c40.wait_stamp(st, timeout_ms=500, label="stampIndex 40 at version 60: dropped")
    c32.submit(abi.OP_NOP)
    probe(s, abi.OP_NOP, expect=F, barriers=[(40, 0)], label="a barrier on index 40 at version 60")
    s.events("at 60", irq_stamps=True)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    c40.submit(abi.OP_NOP)
    s.events("back at 245", irq_stamps=True)
    stamps(s, 41)
    return s


@scenario
def fifo_child_report_missing_fifo():
    """A report of stamp waits (0x1028 on a FIFO that is not faulted) skips a
    waiting stampIndex that names no child (DeviceCore::fifo_for_stamp_index
    finds none and logs it): child 1 with stampIndex 3 blocks on a barrier and
    no child 3 exists. Nothing reaches the guest; slot 7 written by the guest
    then releases child 1 at the wait's next wake, a stamp signalled elsewhere
    (a root NOP)."""
    s = Scenario("fifo_child_report_missing_fifo", doc=fifo_child_report_missing_fifo.__doc__)
    f = s.bring_up()
    track_stamps(s, 8)
    c1 = child(s, 1, stamp_index=3)
    st = c1.cmd(abi.OP_NOP, barriers=[(7, 1)])
    c1.doorbell()
    c1.wait_stamp(st, timeout_ms=500, label="child 1 blocked on slot 7")
    resume_child(s, 2)
    s.fault_state("after the report")
    s.u32(f.base_gpa() + 4 * 7, 1)
    s.submit(abi.OP_NOP)
    c1.wait_stamp(st)
    s.events("released", order=True, irq_stamps=True)
    stamps(s, 8)
    return s
