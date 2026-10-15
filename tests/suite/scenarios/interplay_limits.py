"""Several subsystems at their limits at once: each limit is locked alone
elsewhere (fifo_stamp_ring_full, tasks_define_ids,
newops_fan_out_fan_in_blocks, fifo_child_page_list_full,
display_gap_gamma_sizes, display_gap_cursor_edges, memory_gap_many_ranges);
these lock what happens when they meet. pgtest/limits_util.py lists the
limits and where the code sets them.
"""
import struct

from pgtest import abi
from pgtest import render_lib as R
from pgtest.abi import PAGE
from pgtest.compute_objects import Blit as B, Compute as C
from pgtest.display_guest import (SS_ENABLED, DisplayGuest, display, gamma_words, p_glyph, p_present3,
                                  p_set_shared, p_show, pattern)
from pgtest.fifo import ChildFIFO
from pgtest.fifo_util import NODE_FAULT, NODE_READ, child, resume_child, track_stamps
from pgtest.interplay_util import INC, World, blit, compute
from pgtest.iosfc_guest import CMD_MAP, CMD_TEST, CMD_UNMAP, I_READ, I_WRITE, desc_v2, pte
from pgtest.limits_util import (FILL_REC, Roots, Space, align, buffer_list, buffer_record, define_seq, delete_seq,
                                exec3, exec_seq, fill_cmdbufs, ring_seq, set_list_seq, skip_to)
from pgtest.newops_util import bring_up_big
from pgtest.registry import scenario
from pgtest.resources_build import (BGRA8Unorm, OPT_SHARED, desc1, heap_buffer_rec, heap_rec, iosurface_op,
                                    mapper_ref_rec)
from pgtest.scenario import RAM_BASE, RAM_SIZE, Scenario
from pgtest.session import FH, FW, Session
from pgtest.task import seg


# ---- the pending stamp queue full on several FIFOs ----

@scenario
def interplay_limits_pending_full_many():
    """The 128-slot pending stamp queue full on four FIFOs at once: the root
    and children 1-3 each hold a faulted head (opcode 99) and 129 NOPs, so
    each reads 129 commands (the 129th read and advanced, then blocked
    pushing its stamp) and publishes nothing. Child 2's backlog carries a
    second opcode 99 at stamp 70. Child 4 holds a NOP gated on the last
    stamp of all four. The fault mask has four bits and 0x102c reads 1 until
    the last is resumed. Resumed one at a time (child 3, the root, child 1,
    child 2): while another FIFO is still faulted, the resume first raises
    an interrupt of its own, with nothing new published
    (DeviceCore::resume_child_fifo and DeviceCore::resume_fifo raise one
    when the fault mask is still nonzero); then the FIFO publishes its 128
    in one interrupt, the blocked push goes on, and 129 and 130 follow one
    interrupt each. Child 2, resumed last (the mask then empty, no extra
    interrupt), publishes 69 and faults again at its second 99 (its node's
    fault offset moves to that command), with 0x102c 1 again; a second
    0x1028 = 2 finishes it and releases child 4."""
    s = Scenario("interplay_limits_pending_full_many", doc=interplay_limits_pending_full_many.__doc__)
    f = s.bring_up(fifo_pages=2)
    track_stamps(s, 5)
    c = {fid: child(s, fid) for fid in (1, 2, 3, 4)}
    base_read = f.written
    base_stamp = f.stamp
    last = {}
    for fifo in (f, c[1], c[2], c[3]):
        fifo.cmd(99)
        for k in range(129):
            st = fifo.cmd(99 if (fifo is c[2] and k == 68) else abi.OP_NOP)
        last[fifo.fid] = st
    for fifo in (c[3], c[2], c[1]):
        fifo.doorbell()
    f.doorbell()
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=base_read + 129 * 12, label="root: 129 commands read")
    for fid in (1, 2, 3):
        s.wait(gpa=c[fid].node_gpa(NODE_READ), eq=129 * 12, label="child %d: 129 commands read" % fid)
    x = c[4].cmd(abi.OP_NOP, barriers=[(0, last[0]), (1, last[1]), (2, last[2]), (3, last[3])])
    c[4].doorbell()
    c[4].wait_stamp(x, timeout_ms=300, label="child 4 blocked on all four")
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=base_read + 130 * 12, timeout_ms=300, label="root: the 130th is not read")
    s.fault_state("four full queues")
    for fid in (1, 2, 3, 4):
        s.dump(c[fid].node_gpa(0), abi.CHILD_NODE_SIZE, "node %d, full" % fid)
    s.dump(f.base_gpa(), 20, "stamp slots, full")
    s.events("four FIFOs faulted and full")

    resume_child(s, 3)
    c[3].wait_stamp(last[3])
    s.events("child 3: 128, 129, 130", order=True, irq_stamps=True)
    s.fault_state("child 3 resumed")
    s.resume_root()
    f.wait_stamp(last[0])
    s.events("root: 128, 129, 130", order=True, irq_stamps=True)
    s.r(abi.R_FIFO_READ, "root drained")
    resume_child(s, 1)
    c[1].wait_stamp(last[1])
    s.events("child 1: 128, 129, 130", order=True, irq_stamps=True)
    s.fault_state("child 2 still faulted")
    resume_child(s, 2)
    s.wait(gpa=c[2].node_gpa(NODE_FAULT), ne=0xFFFFFFFF, label="child 2 faults again")
    s.wait(gpa=c[2].stamp_gpa(), eq=69, label="child 2 published 69")
    s.events("child 2: 69, then its second fault", order=True, irq_stamps=True)
    s.dump(c[2].node_gpa(0), abi.CHILD_NODE_SIZE, "node 2, second fault")
    s.fault_state("child 2's second fault")
    c[4].wait_stamp(x, timeout_ms=300, label="child 4 still blocked")
    resume_child(s, 2)
    c[2].wait_stamp(last[2])
    c[4].wait_stamp(x)
    s.events("child 2 drained, child 4 released", order=True, irq_stamps=True)
    s.fault_state("all resumed")
    s.dump(f.base_gpa(), 20, "stamp slots, drained")
    assert base_stamp == 4
    return s


# ---- every task ID at once ----

# The shared space of the task scenarios: VA page 0 the object list (256
# buffers, records from 0x1000), page 1 the buffers' page (every buffer is
# this page), pages 2-3 command buffers of rounds 0 and 1 (task t's at
# 0x40 t).
LIST_VA, REC_VA, OUT_VA, CB_VA = 0, 0x1000, PAGE, 2 * PAGE
NTASKS = 256


def _task_space(s):
    sp = Space(s, 4)
    ol_len = buffer_list(s, sp, LIST_VA, REC_VA, NTASKS, PAGE, OUT_VA // PAGE)
    return sp, ol_len


def _cmdbufs(s, sp, rnd):
    """Task t's command buffer of round rnd (t from 1): fill buffer t (every
    buffer the same page) at 0x100 rnd + t with one byte of value t."""
    return fill_cmdbufs(s, sp, CB_VA + rnd * PAGE + 0x40, 0x40, NTASKS - 1, 1, 1, 0x100 * rnd + 1, 1, 1, 1, 1)


def _execs(s, f, first, count, rnd, seg_len):
    """Execs of tasks first.., chained on f, each listing its own buffer."""
    return exec_seq(s, f, first, count, [(first, 0)], CB_VA + rnd * PAGE + 0x40 * first, 0x40, seg_len, 0x40,
                    res_step=1)


@scenario
def interplay_limits_all_tasks():
    """Every task ID at once: tasks 0-255 defined over one shared address
    space (the same page table; one task root page each), each given the
    same object list of 256 buffers, all over one page; task 256 refused
    ("Invalid task ID"). Tasks 1-255 each run an exec that fills byte t of
    the page through buffer t, chained on the root FIFO one after another.
    Then, interleaved: child 1 holds the execs of tasks 128-255 (round 1)
    behind a NOP gated on the root, while the root deletes tasks 1-127 and
    then opens the gate; child 1's execs all run, and the task words of the
    deleted tasks read 0. An exec for deleted task 5 faults ("Invalid
    task"). Tasks 1-127 are defined again, their lists set, and run round 1.
    Under isolation tasks 1-255 are 255 service instances, of which 127 are
    ended and started again. Block calls are locked as counts and a
    digest."""
    s = Scenario("interplay_limits_all_tasks", watchdog_s=120, doc=interplay_limits_all_tasks.__doc__)
    f = bring_up_big(s)
    track_stamps(s, 2)
    c1 = child(s, 1, pages=4)
    sp, ol_len = _task_space(s)
    roots = Roots(s, sp, 0, NTASKS + 1)
    seg0 = _cmdbufs(s, sp, 0)
    _cmdbufs(s, sp, 1)
    align(s, f, 32)
    define_seq(s, f, roots, 0, NTASKS, sp.length)
    st = set_list_seq(s, f, 0, NTASKS, LIST_VA, ol_len)
    f.doorbell()
    f.wait_stamp(st, timeout_ms=60000, label="256 tasks defined, lists set")
    s.fault_state("256 tasks")
    s.dump(s.pa(s.root_page), 4 * NTASKS, "task words")
    s.dump(s.pa(roots.page(255)), 0x28, "task 255's root")
    s.events("256 tasks defined", blocks_digest=True)
    s.submit(abi.OP_DEFINE_TASK2, struct.pack("<IQI", NTASKS << 1, sp.length, roots.pfn(NTASKS)), fault=True,
             label="task 256")
    s.fault_state("task 256")
    s.resume_root()
    align(s, f, 0x40)
    st = _execs(s, f, 1, 255, 0, seg0)
    f.doorbell()
    f.wait_stamp(st, timeout_ms=60000, label="255 execs")
    sp.dump(OUT_VA, 0x100, "round 0")
    s.events("255 execs, one at a time", blocks_digest=True)

    # Child 1: a NOP gated on the root's next stamp, then tasks 128-255.
    align(s, f, 32)
    gate = f.stamp + 127  # the last delete's stamp
    c1.cmd(abi.OP_NOP, barriers=[(0, gate)], length=0x40, pad_to=0x40)
    last1 = _execs(s, c1, 128, 128, 1, seg0)
    c1.doorbell()
    c1.wait_stamp(1, timeout_ms=300, label="child 1 held on the root")
    delete_seq(s, f, 1, 127)
    f.doorbell()
    f.wait_stamp(gate, timeout_ms=60000, label="127 tasks deleted")
    c1.wait_stamp(last1, timeout_ms=60000, label="child 1's 128 execs")
    s.dump(s.pa(s.root_page), 4 * NTASKS, "task words, 1-127 deleted")
    sp.dump(OUT_VA + 0x100, 0x100, "round 1, tasks 128-255")
    s.events("deletes, then child 1's execs", blocks_digest=True)
    s.submit(abi.OP_EXEC_INDIRECT3, exec3(5, [(CB_VA + 0x40 * 5, seg0)], [(5, 0)]), fault=True,
             label="exec of deleted task 5")
    s.fault_state("deleted task 5")
    s.resume_root()

    align(s, f, 32)
    define_seq(s, f, roots, 1, 127, sp.length)
    st = set_list_seq(s, f, 1, 127, LIST_VA, ol_len)
    f.doorbell()
    f.wait_stamp(st, timeout_ms=60000, label="1-127 defined again")
    align(s, f, 0x40)
    st = _execs(s, f, 1, 127, 1, seg0)
    f.doorbell()
    f.wait_stamp(st, timeout_ms=60000, label="round 1, tasks 1-127")
    sp.dump(OUT_VA, 0x200, "both rounds")
    s.dump(s.pa(s.root_page), 4 * NTASKS, "task words, all defined")
    s.fault_state("end")
    s.events("redefined, round 1", blocks_digest=True)
    s.services = NTASKS - 1  # under isolation, tasks 1-255 are services at once (cli.run_all)
    return s


# ---- every child FIFO, each with its own task, a display, a suspend ----

KIDS = tuple(range(1, 64))


def _kid_execs(s, kids, cs, sp, rnd, seg_len, gate):
    """Child fid's exec of task fid (round rnd's command buffer, its own
    buffer), gated on the root's stamp `gate`; every child rung."""
    sts = {}
    for fid in kids:
        va = CB_VA + rnd * PAGE + 0x40 * fid
        sts[fid] = cs[fid].cmd(abi.OP_EXEC_INDIRECT3, exec3(fid, [(va, seg_len)], [(fid, 0)]), barriers=[(0, gate)])
    for fid in kids:
        cs[fid].doorbell()
    return sts


LOW, HIGH = tuple(range(1, 32)), tuple(range(32, 64))


@scenario
def interplay_limits_children_suspend():
    """All 63 child FIFOs (maxFIFOCount 64 at version 245), each carrying an
    exec for a task of its own (tasks 1-63 over one shared space, each
    filling its own byte), gated on the root. At most 32 of them may wait at
    once: DeviceCore::add_wait marks a waiter with a 32-bit `1 << waiter`,
    so child k and child k + 32 share a bit and asserting that the bit is
    clear kills the host (interplay_limits_child_wait_alias_dies). So
    children 1-31 wait on the root while it presents a frame on port 0, then
    open; then children 32-63 wait (32 waiters, child 32 on bit 0) and open;
    their block calls, from many threads, sorted. Round 1: children 32-63
    are rung gated again, and the host suspends while they wait:
    rvg_device_finish_suspend abandons the 32 waits, and the fresh device
    re-creates 63 children and 63 tasks (under isolation, 63 new service
    instances) from the root page; the 32 parse their rings again and wait
    again. The root presents the other frame and opens the gate; then children
    1-31 run round 1. The page holds both rounds."""
    x = Session("interplay_limits_children_suspend", interplay_limits_children_suspend.__doc__, host=False)
    s = x.s
    x.negotiate(asks=())
    x.root()
    f = s.fifo
    track_stamps(s, 64)
    x.kernel_task(frames=2)
    x.connect(0)
    sp = Space(s, 4)
    ol_len = buffer_list(s, sp, LIST_VA, REC_VA, 64, PAGE, OUT_VA // PAGE)
    roots = Roots(s, sp, 1, 63)
    seg_len = fill_cmdbufs(s, sp, CB_VA + 0x40, 0x40, 63, 1, 1, 1, 1, 1, 1, 1)
    fill_cmdbufs(s, sp, CB_VA + PAGE + 0x40, 0x40, 63, 1, 1, 0x41, 1, 1, 0x41, 1)
    align(s, f, 32)
    define_seq(s, f, roots, 1, 63, sp.length)
    st = set_list_seq(s, f, 1, 63, LIST_VA, ol_len)
    f.doorbell()
    f.wait_stamp(st, timeout_ms=30000, label="63 tasks")  # 63 services start: slow under load
    cs = {fid: x.child(fid) for fid in KIDS}
    s.events("63 tasks, 63 children", sort_blocks=True, blocks_digest=True)

    def run(kids, rnd, label, present=None, before_gate=None):
        gate = f.stamp + (2 if present is not None else 1)
        sts = _kid_execs(s, kids, cs, sp, rnd, seg_len, gate)
        cs[kids[-1]].wait_stamp(sts[kids[-1]], timeout_ms=300, label="child %d waits for the gate" % kids[-1])
        if before_gate:
            before_gate(sts, kids)
        if present is not None:
            x.present(0, present, encode=False)
        st = s.submit(abi.OP_NOP, label="the gate")
        assert st == gate
        for fid in kids:
            cs[fid].wait_stamp(sts[fid])
        s.events(label, sort_blocks=True, blocks_digest=True)

    run(LOW, 0, "a present, then children 1-31", present=0)
    run(HIGH, 0, "children 32-63")
    sp.dump(OUT_VA, 0x40, "round 0")

    def suspend(sts, kids):
        s.events("children 32-63 waiting", sort_blocks=True, blocks_digest=True)
        x.suspend()
        s.events("suspended", sort_blocks=True, blocks_digest=True)
        x.resume()
        cs[32].wait_stamp(sts[32], timeout_ms=300, label="child 32 waits again after the resume")
        s.events("resumed", sort_blocks=True, blocks_digest=True)
        s.dump(s.pa(s.root_page, abi.ROOT_CHILD_NODE), 0x14 * 63, "child nodes after the resume")

    run(HIGH, 1, "a present, then children 32-63", present=1, before_gate=suspend)
    run(LOW, 1, "children 1-31")
    sp.dump(OUT_VA, 0x80, "both rounds")
    x.page(0, "end")
    s.dump(f.base_gpa(), 4 * 64, "stamp slots")
    s.fault_state("end")
    return s


# ---- waiters above 31: DeviceCore::add_wait's 32-bit mask ----
#
# DeviceCore::add_wait marks the waiting FIFO with the 32-bit
# `num::bit32(waiter)`, `1 << waiter` with the shift taken modulo 32, so a
# waiter k >= 32 marks bit k - 32 of the 64-bit waiting word
# (waiting_stamps), while the circular-wait walk tests bit `next` of that
# word, which no waiter above 31 ever sets. DeviceCore::finish_wait clears
# the same aliased bit.


@scenario
def interplay_limits_high_children_circular():
    """A circular wait among children above 31 goes undetected. Children 8
    and 9 wait on each other's first stamp: 9's DeviceCore::add_wait walks
    to 8 and back and throws "Circular wait", so child 9 faults (node fault
    offset 0) while 8 waits; 0x1028 = 9 publishes 9's stamp and releases 8.
    Children 40 and 41 do the same: 40 marks bit 8 and 41 bit 9, the walk
    from 41 tests bit 40, finds it clear, and returns, so both wait forever,
    with no fault (0x102c 0) and their read counts 0. Deleting child 41 (49)
    abandons its wait; defined again over a ring holding a plain NOP, it
    publishes stamp 1, which releases child 40. Then child 33 waits on the
    root, marking child 1's bit, and child 5 waits on idle child 1: the walk
    follows the alias into child 1's empty entry, logs, and child 5 simply
    waits; both are released in turn."""
    s = Scenario("interplay_limits_high_children_circular", doc=interplay_limits_high_children_circular.__doc__)
    s.bring_up()
    track_stamps(s, 42)
    c = {fid: child(s, fid) for fid in (1, 5, 8, 9, 33, 40, 41)}
    a = c[8].cmd(abi.OP_NOP, barriers=[(9, 1)])
    c[8].doorbell()
    c[8].wait_stamp(a, timeout_ms=300, label="child 8 waits on child 9")
    b = c[9].cmd(abi.OP_NOP, barriers=[(8, 1)])
    c[9].doorbell()
    s.wait(gpa=c[9].node_gpa(NODE_FAULT), ne=0xFFFFFFFF, label="child 9: circular wait")
    # 0x102c follows the node's fault offset from the block
    # DeviceCore::signal_fault_in_fifo queues asynchronously on the FIFO
    # queue: wait for it before reading (docs/testing.md 5.4).
    s.wait_fault()
    s.fault_state("child 9 faulted")
    for fid in (8, 9):
        s.dump(c[fid].node_gpa(0), abi.CHILD_NODE_SIZE, "node %d" % fid)
    s.events("children 8 and 9")
    resume_child(s, 9)
    c[8].wait_stamp(a)
    s.events("child 9 resumed, child 8 released", order=True, irq_stamps=True)
    s.fault_state("8 and 9 done")

    x = c[40].cmd(abi.OP_NOP, barriers=[(41, 1)])
    c[40].doorbell()
    c[40].wait_stamp(x, timeout_ms=300, label="child 40 waits on child 41")
    y = c[41].cmd(abi.OP_NOP, barriers=[(40, 1)])
    c[41].doorbell()
    c[41].wait_stamp(y, timeout_ms=300, label="child 41 waits on child 40: no circular wait")
    c[40].wait_stamp(x, timeout_ms=300, label="child 40 still waits")
    s.fault_state("no fault")
    for fid in (40, 41):
        s.dump(c[fid].node_gpa(0), abi.CHILD_NODE_SIZE, "node %d, deadlocked" % fid)
    s.events("children 40 and 41 deadlocked")
    c[41].delete()
    s.dump(c[41].node_gpa(0), abi.CHILD_NODE_SIZE, "node 41 deleted")
    c[40].wait_stamp(x, timeout_ms=300, label="child 40 still waits after the delete")
    c[41].written = 0
    c[41].stamp = 0
    c[41].cmd(abi.OP_NOP)
    c[41].setup_node()
    c[41].define()
    c[41].doorbell()
    c[40].wait_stamp(x)
    s.events("child 41 again, child 40 released", order=True, irq_stamps=True)
    s.dump(s.fifo.base_gpa() + 4 * 40, 8, "slots 40, 41")

    # Child 33 waits on the root (marking bit 1); child 5 then waits on
    # child 1, which is idle: the walk finds bit 1 set, reads child 1's
    # entry, whose next index is UINT32_MAX, logs "Got invalid next waiting
    # index" and lets child 5 wait.
    r = s.fifo.stamp + 1
    z = c[33].cmd(abi.OP_NOP, barriers=[(0, r)])
    c[33].doorbell()
    c[33].wait_stamp(z, timeout_ms=300, label="child 33 waits on the root")
    v = c[5].cmd(abi.OP_NOP, barriers=[(1, 1)])
    c[5].doorbell()
    c[5].wait_stamp(v, timeout_ms=300, label="child 5 waits on child 1")
    s.fault_state("no fault")
    s.submit(abi.OP_NOP, label="the root releases child 33")
    c[33].wait_stamp(z)
    c[1].submit(abi.OP_NOP, label="child 1 releases child 5")
    c[5].wait_stamp(v)
    s.events("children 33 and 5 released", order=True, irq_stamps=True)
    s.fault_state("end")
    return s


SLOW_PAGES = 512     # an 8 MiB ring, every entry of its page list one page


def _slow_child(s, fid):
    """A child whose ring is 512 entries naming one page (8 MiB; the blocks
    setup's root task range holds 1024 pages). Its first command is a NOP
    whose header's length covers the ring but for 0x100 bytes, so that the
    thread copies 8 MiB of payload before it parses the command after it:
    the host dies well after the doorbell's MMIO write has returned and
    been recorded, during the wait that follows, whatever the load."""
    c = ChildFIFO(s, fid)
    c.setup_node(page_list=[s.pfn(c.pages[0])] * SLOW_PAGES)
    c.pages = [c.pages[0]] * SLOW_PAGES
    c.define()
    s.fifos[fid] = c
    skip_to(c, SLOW_PAGES * PAGE - 0x100)
    return c


@scenario
def interplay_limits_child_wait_alias_dies():
    """Two waiters whose ids differ by 32 kill the host: child 1 waits on
    the root's next stamp (bit 1), then child 33 waits on it too, marks bit
    1 again, and DeviceCore::add_wait's check that the waiter's bit is clear
    in the waiting word dies, in every setup (the device is the host's). The
    root and child 32 alias the same way
    (interplay_limits_root_wait_alias_dies). Child 33 reaches its wait only
    after an 8 MiB NOP, so the death lands in the wait after its
    doorbell."""
    s = Scenario("interplay_limits_child_wait_alias_dies", watchdog_s=15,
                 doc=interplay_limits_child_wait_alias_dies.__doc__)
    f = s.bring_up()
    track_stamps(s, 34)
    c1 = child(s, 1)
    c33 = _slow_child(s, 33)
    a = c1.cmd(abi.OP_NOP, barriers=[(0, f.stamp + 1)])
    c1.doorbell()
    c1.wait_stamp(a, timeout_ms=300, label="child 1 waits on the root")
    s.events("child 1 waiting")
    b = c33.cmd(abi.OP_NOP, barriers=[(0, f.stamp + 1)])
    c33.doorbell()
    c33.wait_stamp(b, timeout_ms=3000, label="child 33 waits on the root: the host dies")
    return s


@scenario
def interplay_limits_root_wait_alias_dies():
    """The root waits on child 2's first stamp (bit 0); child 32 then waits
    on it too, marks bit 0 (1 << 32 is 1 << 0), and the assert aborts the
    host. Child 32 reaches its wait only after an 8 MiB NOP, so the death
    lands in the wait after its doorbell."""
    s = Scenario("interplay_limits_root_wait_alias_dies", watchdog_s=15,
                 doc=interplay_limits_root_wait_alias_dies.__doc__)
    f = s.bring_up()
    track_stamps(s, 33)
    c2 = child(s, 2)
    c32 = _slow_child(s, 32)
    a = f.cmd(abi.OP_NOP, barriers=[(2, 1)])
    f.doorbell()
    f.wait_stamp(a, timeout_ms=300, label="the root waits on child 2")
    s.events("the root waiting")
    b = c32.cmd(abi.OP_NOP, barriers=[(2, 1)])
    c32.doorbell()
    c32.wait_stamp(b, timeout_ms=3000, label="child 32 waits on child 2: the host dies")
    return s


# ---- the blocks setup's root task range, full ----

RING_PAGES = 13


def _ring_pages(fid):
    """13 pages a ring, except child 59's 5: with the root node (1), the
    root FIFO (8) and 256 task roots, children 1-58 take 754 pages, and
    child 59 the last 5 of the range's 1024 (measured: with a 4-page child
    59 one page is left, which a connect takes)."""
    return 5 if fid == 59 else RING_PAGES


@scenario
def interplay_limits_root_range_full():
    """In the blocks setup the root task's range allocator (0xffffff bytes in
    16 KiB granules) maps the root node, the root FIFO, each local task's root
    page, each child's page list for a moment and its ring, each display's
    shared page and each device info page. Filled exactly, 1024 pages, by the
    root node, the largest root FIFO (8 pages), all 256 tasks (256 root pages)
    and children 1-59 (rings of 13 pages, every entry of a page list naming one
    page, and child 59's of 5): children 60-63 cannot be placed, still signal
    their define and never run (-start's result is ignored). With nothing left,
    a display connect (opcode 1) and GetDeviceInfo (58) are tried on port 0:
    both fault. Deleting tasks 200-255 frees 56 pages: a connect on port 1 and
    the device info work, and a child that never ran, deleted and defined
    again, runs. In the map and isolate setups nothing is allocated from that
    range (under isolation only task 0 is local), and every child runs, and the
    connect on port 0 and the device info work at once."""
    s = Scenario("interplay_limits_root_range_full", watchdog_s=120, displays=[display(0), display(1)],
                 descriptor={"port_count": 2},
                 doc=interplay_limits_root_range_full.__doc__)
    f = bring_up_big(s)
    track_stamps(s, 64)
    sp, ol_len = _task_space(s)
    roots = Roots(s, sp, 0, NTASKS)
    align(s, f, 32)
    st = define_seq(s, f, roots, 0, NTASKS, sp.length)
    f.doorbell()
    f.wait_stamp(st, timeout_ms=60000, label="256 tasks")
    s.fault_state("256 tasks")
    cs = {}
    for fid in KIDS:
        c = cs[fid] = ChildFIFO(s, fid)
        c.setup_node(page_list=[s.pfn(c.pages[0])] * _ring_pages(fid))
        s.fifos[fid] = c
        f.cmd(abi.OP_DEFINE_CHILD_FIFO, struct.pack("<I", fid))
    f.doorbell()
    f.wait_stamp(f.stamp, label="63 children defined")
    s.fault_state("63 children")
    s.events("256 tasks, 63 children", blocks_digest=True)
    sts = {fid: cs[fid].cmd(abi.OP_NOP) for fid in KIDS}
    for fid in KIDS:
        cs[fid].doorbell()
    for fid in KIDS:
        cs[fid].wait_stamp(sts[fid], timeout_ms=300, label="child %d's NOP" % fid)
    s.events("a NOP on each child", blocks_digest=True)
    page = s.alloc_page()
    s.u32(s.pa(page, SS_ENABLED), 0xF)
    s.submit(abi.OP_DISPLAY_SET_SHARED_STATE_PAGE, p_set_shared(0, s.pfn(page)), fault="either",
             label="connect with the range full")
    s.fault_state("connect")
    s.resume_root()
    info = s.alloc_page()
    s.submit(abi.OP_GET_DEVICE_INFO, struct.pack("<III", 3, 4, s.pfn(info)), fault="either",
             label="device info with the range full")
    s.fault_state("device info")
    s.resume_root()
    s.submit(abi.OP_NOP, label="resumed")
    s.dump(s.pa(info), 0x20, "device info page")
    s.events("connect and device info", blocks_digest=True)

    align(s, f, 32)
    st = delete_seq(s, f, 200, 56)
    f.doorbell()
    f.wait_stamp(st, label="tasks 200-255 deleted")
    page1 = s.alloc_page()
    s.u32(s.pa(page1, SS_ENABLED), 0xF)
    s.submit(abi.OP_DISPLAY_SET_SHARED_STATE_PAGE, p_set_shared(1, s.pfn(page1)), label="connect port 1")
    s.submit(abi.OP_GET_DEVICE_INFO, struct.pack("<III", 3, 4, s.pfn(info)), label="device info again")
    s.dump(s.pa(info), 0x20, "device info page")
    c = cs[63]
    c.delete()
    c.written = 0
    c.stamp = 1
    st = c.cmd(abi.OP_NOP)
    c.setup_node(page_list=[s.pfn(c.pages[0])] * _ring_pages(63))
    c.define()
    c.doorbell()
    c.wait_stamp(st, label="child 63 again")
    s.dump(s.pa(s.root_page, abi.ROOT_CHILD_NODE), 0x14 * 63, "child nodes")
    s.fault_state("end")
    s.events("tasks deleted, connect, device info, child 63", blocks_digest=True)
    s.services = NTASKS - 1  # under isolation, tasks 1-255 are services at once (cli.run_all)
    return s


# ---- eight ports at their limits, tasks running, a reset ----

PORTS = 8
GLYPH = 256          # the largest glyph
GAMMA = 1024         # the largest gamma table
MODES = [(64 + 8 * i, 32 + (i % 4), 60) for i in range(130)]   # past the page's 128


def _glyph_sum(start, step, n):
    """The sum of n u32 pixels start + k * step (mod 2^32), which is what
    the host sums over a glyph whose stride is 4w (every pixel once)."""
    return (n * start + step * n * (n - 1) // 2) & 0xFFFFFFFF


@scenario
def interplay_limits_displays_reset():
    """All eight display ports (RVG_MAX_DISPLAY_PORTS) busy at once, each
    display offering 130 modes (the page lists 128), each connected and
    acked, given the largest cursor glyph (256 x 256; a 256 KiB glyph
    written with mem_seq) and shown, and presented with the largest gamma
    table (1024 entries), while tasks 1-8 run execs on children 1-8. Then
    tasks 1-8 queue execs on their children gated on a root stamp that never
    comes, and the host resets: the teardown disconnects all eight guest
    displays (eight interrupts; ports 0 and 7 read mode count 0) and drops
    the children and tasks with their waits, but leaves the task words in
    the root page as they were (the display words are cleared). Brought up
    again on the same pages, port 7 reconnects and presents, and child 1
    with task 1 runs again. The children's block calls (blocks setup)
    interleave with the root's by timing, and are locked sorted."""
    s = Scenario("interplay_limits_displays_reset", watchdog_s=60, descriptor={"port_count": PORTS},
                 displays=[display(p, modes=MODES) for p in range(PORTS)], doc=interplay_limits_displays_reset.__doc__)
    g = DisplayGuest(s, pages=48)
    f = s.fifo
    track_stamps(s, 9)
    refs = [g.backing(16, 8, pixels=pattern(16, 8, 0x21 * i))[0] for i in range(2)]
    glyph_va = g.t0.reserve(4 * GLYPH * GLYPH)
    s.mem_seq(g.t0.gpa(glyph_va), GLYPH * GLYPH, 4, b"\0" * 4, fields=[(0, 4, 0x80402010, 0x00010203)])
    gsum = _glyph_sum(0x80402010, 0x00010203, GLYPH * GLYPH)
    words, csum = gamma_words(GAMMA)
    gva = g.t0.blob(struct.pack("<%dI" % len(words), *words))
    g.install()
    sp = Space(s, 4)
    ol_len = buffer_list(s, sp, LIST_VA, REC_VA, 16, PAGE, OUT_VA // PAGE)
    roots = Roots(s, sp, 1, 8)
    seg_len = fill_cmdbufs(s, sp, CB_VA + 0x40, 0x40, 8, 1, 1, 0x10, 0x10, 0x10, 1, 1)
    fill_cmdbufs(s, sp, CB_VA + PAGE + 0x40, 0x40, 8, 1, 1, 0x110, 0x10, 0x10, 0x11, 1)
    align(s, f, 32)
    define_seq(s, f, roots, 1, 8, sp.length)
    st = set_list_seq(s, f, 1, 8, LIST_VA, ol_len)
    f.doorbell()
    f.wait_stamp(st, label="tasks 1-8")
    cs = {fid: child(s, fid) for fid in range(1, 9)}
    for p in range(PORTS):
        g.connect(p, enabled=0xF)
        g.ack(p, 1)
    s.events("eight ports connected")
    s.dump(g.ss(0, 0x200), 0x10, "port 0: ack and mode count")
    s.dump(g.ss(0, 0x210 + 16 * 126), 0x30, "port 0: modes 126-128")
    s.dump(g.ss(7, 0x208), 4, "port 7: mode count")

    sts = {fid: cs[fid].cmd(abi.OP_EXEC_INDIRECT3, exec3(fid, [(CB_VA + 0x40 * fid, seg_len)], [(fid, 0)]))
           for fid in cs}
    for fid in cs:
        cs[fid].doorbell()
    n = 0
    for p in range(PORTS):
        s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(p, glyph_va, 4 * GLYPH * GLYPH, 4 * GLYPH, GLYPH, GLYPH,
                                                       GLYPH - 1, GLYPH - 1, gsum), label="glyph on %d" % p)
        s.submit(abi.OP_DISPLAY_CURSOR_SHOW, p_show(p, 1), label="show on %d" % p)
        s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(p, refs[p % 2], gva, 12 * GAMMA, GAMMA, csum),
                 label="present on %d" % p)
        n += 1
        s.wait(cb="newFrame", cb_count=n, label="frame %d" % n)
        s.wait(cb="cursorShow", cb_count=n, label="show %d" % n)
    for fid in cs:
        cs[fid].wait_stamp(sts[fid])
    s.events("eight glyphs, shows and presents, eight execs", sort_blocks=True)
    for p in (0, 7):
        s.dump(g.ss(p, 0xF00), 0xB0, "port %d: glyph, gamma, present" % p)
    sp.dump(OUT_VA, 0x90, "round 0")

    gate = f.stamp + 100
    sts = {fid: cs[fid].cmd(abi.OP_EXEC_INDIRECT3, exec3(fid, [(CB_VA + PAGE + 0x40 * fid, seg_len)], [(fid, 0)]),
                            barriers=[(0, gate)]) for fid in cs}
    for fid in cs:
        cs[fid].doorbell()
    cs[8].wait_stamp(sts[8], timeout_ms=300, label="child 8 waits")
    s.api("reset")
    s.quiesce(100)
    s.events("reset")
    s.r(abi.R_FIFO_ENABLE, "after the reset")
    s.r(abi.R_VERSION, "after the reset")
    s.dump(s.pa(s.root_page), 0x40, "task words after the reset")
    s.dump(s.pa(s.root_page, abi.ROOT_DISPLAY_WORD), 4 * PORTS, "display words after the reset")
    for p in (0, 7):
        s.dump(g.ss(p, 0x100), 8, "port %d: interrupts after the reset" % p)
        s.dump(g.ss(p, 0x208), 4, "port %d: mode count after the reset" % p)
    for p in range(PORTS):
        s.api("present_counts", display=p)
    sp.dump(OUT_VA + 0x100, 0x90, "the gated round never ran")

    s.w(abi.R_VERSION, abi.MAX_VERSION)
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    f.enable()
    f.written = 0
    g.t0.define()
    g.install()
    g.connect(7)
    s.events("port 7 reconnected", order=True)
    g.ack(7, 2)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(7, refs[0]), label="present on 7 after the reset")
    s.wait(cb="newFrame", cb_count=n + 1)
    t1 = struct.pack("<IQI", 1 << 1, sp.length, roots.pfn(1))
    s.submit(abi.OP_DEFINE_TASK2, t1, label="task 1 again")
    s.submit(abi.OP_SET_OBJECT_LIST, struct.pack("<III", 1, LIST_VA // PAGE, ol_len), label="task 1's list")
    c1 = cs[1]
    c1.written = 0
    c1.stamp = 2  # above what its stamp slot already holds from before the reset
    c1.setup_node()
    c1.define()
    st = c1.submit(abi.OP_EXEC_INDIRECT3, exec3(1, [(CB_VA + PAGE + 0x40, seg_len)], [(1, 0)]),
                   label="child 1 after the reset")
    s.events("after the reset")
    sp.dump(OUT_VA + 0x100, 0x90, "end")
    s.api("present_counts", display=7)
    s.fault_state("end")
    return s


# ---- ring wraps on the largest root FIFO and scattered child rings ----

EXEC_SIZE = 0x70


@scenario
def interplay_limits_ring_wraps():
    """A token passed through the largest root FIFO (0x20000, a ring of
    0x1fc00 bytes) and two children over scattered pages (child 1: 16
    pages, child 2: 5, every ring page followed in RAM by a page of 0xff),
    ten rounds: each exec increments a counter (compute_inc) and copies it
    into log[k]. Before each exec, a NOP whose header's length carries the
    ring to a chosen offset (only its header written; its payload is the
    ring's stale bytes), so that across the rounds each FIFO's exec header
    straddles the ring's end (6 bytes before it), an exec's payload
    straddles it, and on the children an exec's header straddles a page
    boundary (5 bytes before it) and a payload another; each FIFO laps its
    ring five times. Every command waits for the one
    before it (the NOP on the previous FIFO's exec, the exec on its own
    NOP), so the 60 interrupts come in token order, and log[k] = k + 1.
    Each round is rung once, as a guest must: a doorbell covering more
    than a ring's worth would overwrite commands not yet read."""
    w = World("interplay_limits_ring_wraps", interplay_limits_ring_wraps.__doc__, root_pages=8)
    s = w.s
    for fid, n in ((1, 16), (2, 5)):
        w.c[fid] = child(s, fid, define=False, pages=n, scatter=True)
        for p in w.c[fid].pages:
            s.fill(s.pa(p + 1), PAGE, 0xFF)
    track_stamps(s, 3)
    w.nslots = 3
    k_ref, k_va = w.buffer()
    log, log_va = w.buffer()
    inc = w.kernel(INC)
    w.ready()
    order = (0, 1, 2)

    def targets(f, r):
        ring = f.ring_size()
        return (0x100, ring - 6, ring - 0x30, PAGE - 5, 2 * PAGE - 0x20)[r % 5] % ring

    prev = None  # (fid, stamp) of the last exec
    k = 0
    for r in range(10):
        for fid in order:
            f = w.fifo(fid)
            t = targets(f, r)
            bars = [prev] if prev else []
            # A round's commands on one FIFO go in one doorbell, so they must
            # not reach their own first header, which the ring still holds.
            assert (t - f.written) % f.ring_size() + EXEC_SIZE <= f.ring_size()
            nop = skip_to(f, t, barriers=bars)
            st = w.exec_cmd(fid, compute(C.set_pipeline(inc), C.set_buffers(0, [(k_ref, 0)]),
                                         C.dispatch_threads((1, 1, 1), (1, 1, 1))),
                            blit(B.copy_buffer_to_buffer(k_ref, log, 0, 4 * k, 4)),
                            barriers=[(f.stamp_index, nop)], size=EXEC_SIZE)
            prev = (f.stamp_index, st)
            k += 1
        for fid in (2, 1, 0):
            w.fifo(fid).doorbell()
        w.fifo(prev[0]).wait_stamp(prev[1])
        w.t.reset_cmdbufs()
        if r % 5 == 4:
            s.events("rounds %d-%d in token order" % (r - 4, r), order=True, irq_stamps=True)
    w.t.dump(k_va, 4, "counter")
    w.t.dump(log_va, 4 * k, "log")
    w.node(1)
    w.node(2)
    s.r(abi.R_FIFO_READ, "root read count")
    w.stamps()
    s.fault_state("end")
    return s


# ---- the IOSurface host device: a full ring, many surfaces, render, present ----

NSURF = 256
MID_LAST = 8191      # the largest mapping ID
RING = 1024          # ring entries: one 16 KiB page of 16-byte commands


def _mid(k):
    """Surface k's mapping ID: the last slot (15) of descriptor page k."""
    return 16 * k + 15


@scenario(setups=("map", "isolate"))
def interplay_limits_iosurface_ring():
    """The IOSurface host device (map and isolate only: QEMU creates it only
    with a memory map) at its limits. 256 surfaces, one on each of 256
    descriptor pages (v2, the last slot of each: IDs 15, 31, ... 4095, all
    over one guest page), and ID 8191 (the largest, on descriptor page 511,
    over a page of its own). One doorbell fills the whole 1024-entry ring:
    257 maps, a map of 8192 (BadArgument), and 766 tests of 8191, which
    write their pattern into its page; 1024 interrupts, the read counter at
    1024. Task 1 renders clears through mapper textures (type 11) into
    surfaces 15 and 8191, 8191 is presented (opcode 8), rendered into again
    while the display holds it. A second full lap of the ring unmaps all
    256: 15 is Busy, because task 1's mapper texture still holds it (a
    retain count other than 1; under isolation too), and the other 255
    unmap; 8191 is Busy (the display and the texture hold it); 15 is tested
    (still mapped: its pattern lands in the shared page) and mapped again
    (already mapped: success); the read counter ends at 2048."""
    x = Session("interplay_limits_iosurface_ring", interplay_limits_iosurface_ring.__doc__, setups=("map", "isolate"),
                host=False, iosfc=True)
    s = x.s
    x.negotiate(asks=())
    x.root()
    x.kernel_task()
    x.connect(0)
    x.iosfc_start()
    d = x.iosfc
    # One shared surface page and 8191's own, each under a one-page table.
    root_a, data_a = d.surface_pages(1)
    root_b, data_b = d.surface_pages(1)
    s.mem(s.pa(data_a), pattern(FW, FH, 0x10))
    s.mem(s.pa(data_b), pattern(FW, FH, 0x70))
    # Descriptor pages 0-255 contiguous, one descriptor each at slot 15;
    # page 511 for 8191. The table's entries {valid, isTable, pageNumber}.
    dp = s.alloc_page(NSURF)
    s.mem_seq(s.pa(dp, 15 * 0x400), NSURF, PAGE, desc_v2(root_a, FW, FH))
    s.mem_seq(s.pa(d.table), NSURF, 4, struct.pack("<I", pte(0)), fields=[(0, 4, pte(s.pfn(dp)), 4)])
    d.desc_pages.update({k: dp + k for k in range(NSURF)})
    d.put_desc(MID_LAST, desc_v2(root_b, FW, FH))

    def ring(entries, label):
        """Writes entries (cmd, id) at the write counter, each run of one
        command whose IDs step evenly in one mem_seq, rings once, and waits
        for them all."""
        k = 0
        while k < len(entries):
            cmd, mid = entries[k]
            step = entries[k + 1][1] - mid if k + 1 < len(entries) and entries[k + 1][0] == cmd else 0
            j = k + 1
            while j < len(entries) and entries[j][0] == cmd and entries[j][1] - entries[j - 1][1] == step:
                j += 1
            s.mem_seq(d.cmd_gpa(d.written + k), j - k, 16, struct.pack("<IIiI", cmd, mid, 0x7EADBEEF, 0),
                      fields=[(4, 4, mid, step)] if step else ())
            k = j
        assert len(entries) == RING
        d.written += len(entries)
        s.iosfc_w(I_WRITE, d.written)
        d.irqs = getattr(d, "irqs", 0) + len(entries)
        s.wait(iosfc_irqs=d.irqs, label=label, timeout_ms=20000)
        s.iosfc_r(I_READ, label)
        s.dump(s.pa(d.ring), PAGE, label + ": the ring")
        s.dump(s.pa(d.ring), 0x40, label + ": entries 0-3")
        s.dump(s.pa(d.ring, 16 * 254), 0x60, label + ": entries 254-259")
        s.dump(s.pa(d.ring, 16 * (RING - 2)), 0x20, label + ": entries 1022-1023")

    ring([(CMD_MAP, _mid(k)) for k in range(NSURF)] + [(CMD_MAP, MID_LAST), (CMD_MAP, MID_LAST + 1)]
         + [(CMD_TEST, MID_LAST)] * (RING - NSURF - 2), "lap 1")
    s.events("lap 1")
    s.dump(s.pa(data_b), 0x40, "8191's page after the tests")

    def extra(t):
        dsc = desc1(fmt=BGRA8Unorm, w=FW, h=FH, usage=1 | 2 | 4, options=OPT_SHARED)
        t.ma = t.ol.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(_mid(0)) + iosurface_op(12, dsc))
        t.mb = t.ol.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(MID_LAST) + iosurface_op(12, dsc))
        t.extra += [t.ma, t.mb]

    t = x.task(1, 0x11, extra=extra)

    def clear(tex, rgba, label):
        rec = R.describe_pass({0: dict(tex=tex, clear=rgba)}, width=FW, height=FH)
        x.run(0, t, [seg(abi.ENC_RENDER, rec)], label)

    clear(t.ma, (0.0, 0.5, 0.0, 1.0), "clear into surface 15")
    clear(t.mb, (1.0, 0.0, 0.0, 1.0), "clear into surface 8191")
    s.dump(s.pa(data_a), FW * FH * 4, "the shared page")
    s.dump(s.pa(data_b), FW * FH * 4, "8191's page")
    x.swap(0, MID_LAST, encode=False)
    clear(t.mb, (0.0, 0.0, 1.0, 1.0), "clear into 8191 while presented")
    s.dump(s.pa(data_b), 0x40, "8191's page, presented")
    x.page(0, "presenting 8191")

    ring([(CMD_UNMAP, _mid(k)) for k in range(NSURF)] + [(CMD_UNMAP, MID_LAST), (CMD_TEST, _mid(0)),
                                                         (CMD_MAP, _mid(0))]
         + [(CMD_TEST, _mid(0))] * (RING - NSURF - 3), "lap 2")
    s.events("lap 2")
    s.dump(s.pa(data_a), 0x40, "the shared page after lap 2")
    s.fault_state("end")
    return s


# ---- many RAM ranges, depth-3 tables across them ----

XBASE = RAM_BASE + RAM_SIZE + 0x1000000   # past the first range, with a hole
NRANGES = 15         # the runner holds 16 ranges (runner/pgrun.m:148)
RPAGES = 8
NBUFS = 16


@scenario
def interplay_limits_ranges_depth3():
    """Guest RAM in 16 ranges: the first, and 15 of 8 pages each with a
    range-sized hole after each, listed to the device in reverse order (the
    memory map sorts them). Three tasks of 32 GiB (0x800000000, the
    in-process maximum) share one depth-3 page table (the deepest) whose
    tables, task roots and data pages are dealt out over the 15 ranges in
    turn: the first eight VA pages (object list, records, command buffers)
    in the first range, and 16 two-page buffers at VA pages 0x8000 k +
    0xfff, each across two leaf tables and so across two ranges. Each task
    runs one exec of 16 fills over all 16 buffers: task 1 across each
    buffer's page boundary, task 2 at its start, task 3 at its end. Every
    page of every buffer lands in its own range, in every setup (no run of
    pages crosses a range, so the blocks setup maps each as it is)."""
    ram = [(RAM_BASE, RAM_SIZE)] + [(XBASE + 2 * i * RPAGES * PAGE, RPAGES * PAGE) for i in reversed(range(NRANGES))]
    s = Scenario("interplay_limits_ranges_depth3", ram=ram, doc=interplay_limits_ranges_depth3.__doc__)
    f = s.bring_up()
    track_stamps(s, 1)
    # Pages dealt out one range at a time: consecutive pages are never in
    # the same range.
    pool = [(XBASE - RAM_BASE) // PAGE + 2 * i * RPAGES + j for j in range(RPAGES) for i in range(NRANGES)]
    main = s.alloc_page(8)
    vbuf = [0x8000 * k + 0xFFF for k in range(NBUFS)]
    leaves = {v: main + v for v in range(8)}
    for v in vbuf:
        leaves[v] = pool.pop()
        leaves[v + 1] = pool.pop()
    rng = lambda p: (p - (XBASE - RAM_BASE) // PAGE) // (2 * RPAGES)  # noqa: E731
    assert all(rng(leaves[v]) != rng(leaves[v + 1]) for v in vbuf)
    sp = Space(s, 0, depth=3, leaf=lambda v: leaves[v], table=lambda: pool.pop(0), vpns=sorted(leaves),
               length=0x800000000)
    roots = [pool.pop(0) for _ in range(3)]
    for r in roots:
        s.mem(s.pa(r), sp.root_bytes())
    # The object list: 16 two-page buffers.
    recs = b"".join(buffer_record(2 * PAGE, v) for v in vbuf)
    sp.write(0x400, recs)
    sp.write(0, abi.OBJ_ENTRY.pack(0, 0) + b"".join(abi.OBJ_ENTRY.pack(abi.T_BUFFER | 16 << 8, 0x400 + 16 * i)
                                                   for i in range(NBUFS)))
    ol_len = 12 * (NBUFS + 1)
    res = [(r, 0) for r in range(1, NBUFS + 1)]
    spans = {1: (PAGE - 8, 16, 0x11), 2: (0, 16, 0x22), 3: (2 * PAGE - 16, 16, 0x33)}
    cbs = {}
    for tid, (off, n, byte) in spans.items():
        body = b"".join(abi.REC_HDR.pack(306, 32) + struct.pack("<IQQB3x", r, off, n, byte) for r, _ in res)
        seg_ = abi.SEG_HDR.pack(8 + len(body), abi.ENC_BLIT, 0, 0) + body
        va = 2 * PAGE + 0x1000 * tid
        sp.write(va, seg_)
        cbs[tid] = (va, len(seg_))
    s.events("laid out")
    for tid, r in zip((1, 2, 3), roots):
        s.submit(abi.OP_DEFINE_TASK2, struct.pack("<IQI", tid << 1, sp.length, s.pfn(r)), label="task %d" % tid)
        s.submit(abi.OP_SET_OBJECT_LIST, struct.pack("<III", tid, 0, ol_len), label="task %d's list" % tid)
    s.fault_state("defined")
    for tid in (1, 2, 3):
        s.submit(abi.OP_EXEC_INDIRECT3, exec3(tid, [cbs[tid]], res), label="task %d's fills" % tid)
    s.fault_state("filled")
    s.events("three tasks, 48 fills")
    for k, v in enumerate(vbuf):
        s.dump(sp.gpa(v * PAGE), 0x10, "buffer %d, first page's start" % k)
        s.dump(sp.gpa(v * PAGE + PAGE - 0x10), 0x10, "buffer %d, first page's end" % k)
        s.dump(sp.gpa((v + 1) * PAGE), 0x10, "buffer %d, second page's start" % k)
        s.dump(sp.gpa((v + 1) * PAGE + PAGE - 0x10), 0x10, "buffer %d, second page's end" % k)
    s.dump(s.pa(roots[2]), 0x28, "task 3's root")
    for tid in (1, 2, 3):
        s.submit(abi.OP_DELETE_TASK, struct.pack("<I", tid), label="delete task %d" % tid)
    s.events("deleted")
    return s


# ---- the largest object and placement lists, a heap packed full ----

NOBJ = 4096          # buffers, all over one page
HEAP_PAGES = 16


@scenario
def interplay_limits_objects_heap():
    """One task with 4113 objects: 4096 buffers (all over one page), a
    private heap of 16 pages and 16 page-sized heap buffers packed into it
    with no gap. The object and placement lists are set with opcode 68 at
    their largest for this task (12 and 16 bytes an entry, both ending
    inside it); only the service records a placement list (its task root
    fields). One exec, rung on the largest root FIFO, lists all 4113 objects
    and fills buffers 1 and 4096 and every heap buffer, copying each heap
    buffer out, which shows each placed at its offset. A 17th heap buffer,
    at the heap's end, fails to be made and its exec faults. Then all 4096
    buffers are deleted (opcode 40, 1024 to a doorbell, each waiting for the
    one before). An exec of buffer 4096 then completes and its fill lands:
    the object list still names it, and the task makes it again from its
    record. The heap buffers still work."""
    s = Scenario("interplay_limits_objects_heap", watchdog_s=120, doc=interplay_limits_objects_heap.__doc__)
    f = bring_up_big(s)
    track_stamps(s, 1)
    nent = NOBJ + HEAP_PAGES + 3              # nil, buffers, heap, heap buffers, the 17th
    list_va, place_va = 0, 4 * PAGE           # 4 pages, 5 pages
    rec_va, out_va, cb_va = 9 * PAGE, 13 * PAGE, 14 * PAGE
    heap_va, hrec_va = 15 * PAGE, 31 * PAGE
    sp = Space(s, 32)
    buffer_list(s, sp, list_va, rec_va, NOBJ, PAGE, out_va // PAGE)
    heap = NOBJ + 1
    hbufs = list(range(heap + 1, heap + 1 + HEAP_PAGES))
    extra = heap + 1 + HEAP_PAGES
    recs = heap_rec(heap_va // PAGE, HEAP_PAGES * PAGE, 1)
    entries = [abi.OBJ_ENTRY.pack(abi.T_HEAP | len(recs) << 8, hrec_va)]
    for i in range(HEAP_PAGES + 1):
        va = hrec_va + 0x10 + 0x20 * i
        sp.write(va, heap_buffer_rec(heap, i * PAGE, PAGE))
        entries.append(abi.OBJ_ENTRY.pack(abi.T_HEAP_BUFFER | 0x14 << 8, va))
    sp.write(hrec_va, recs)
    sp.write(list_va + 12 * (NOBJ + 1), b"".join(entries))
    obj_len, place_len = 12 * nent, 16 * nent
    assert obj_len <= 4 * PAGE and place_len <= 5 * PAGE
    roots = Roots(s, sp, 1, 1)
    s.submit(abi.OP_DEFINE_TASK2, struct.pack("<IQI", 1 << 1, sp.length, roots.pfn(1)), label="task 1")
    s.submit(abi.OP_SET_OBJECT_AND_PLACEMENT_LIST, struct.pack("<IQQII", 1, list_va, place_va, obj_len, place_len),
             label="the largest lists")
    s.dump(s.pa(roots.page(1)), 0x28, "task root")

    # The exec: every object listed; fills of buffers 1 and 4096; each heap
    # buffer filled with its index and copied out to 0x100 + 0x10 i.
    body = rec_fill(1, 0, 0x10, 0xA1) + rec_fill(NOBJ, 0x10, 0x10, 0xA2)
    for i, hb in enumerate(hbufs):
        body += rec_fill(hb, 0, PAGE, 0x40 + i) + rec_copy(hb, NOBJ, PAGE - 0x10, 0x100 + 0x10 * i, 0x10)
    cb = abi.SEG_HDR.pack(8 + len(body), abi.ENC_BLIT, 0, 0) + body
    sp.write(cb_va, cb)
    everything = [(r, 0) for r in range(1, extra)]
    st = s.submit(abi.OP_EXEC_INDIRECT3, exec3(1, [(cb_va, len(cb))], everything), label="an exec of 4113 objects",
                  timeout_ms=20000)
    sp.dump(out_va, 0x200, "the buffers' page")
    s.fault_state("after the exec")
    s.events("lists set, the exec", blocks_digest=True)
    cb2 = abi.SEG_HDR.pack(8 + FILL_REC, abi.ENC_BLIT, 0, 0) + rec_fill(extra, 0, 0x10, 0x55)
    sp.write(cb_va + 0x1000, cb2)
    s.submit(abi.OP_EXEC_INDIRECT3, exec3(1, [(cb_va + 0x1000, len(cb2))], [(heap, 0), (extra, 0)]), fault=True,
             label="a heap buffer at the heap's end")
    s.fault_state("the 17th heap buffer")
    s.resume_root()

    # Delete every buffer: {taskID, operation 1000 (resource), length 12, ref}.
    tmpl = (abi.HDR.pack(abi.OP_DELETE_OBJECT, 1, 0x40, 0) + abi.BARRIER.pack(0, 0)
            + struct.pack("<IIII", 1, 1000, 12, 0))
    tmpl += bytes(0x40 - len(tmpl))
    for q in range(4):
        align(s, f, 0x40)
        st = ring_seq(s, f, NOBJ // 4, tmpl, fields=[(32, 4, 1 + q * NOBJ // 4, 1), (16, 4, f.stamp, 1)])
        f.doorbell()
        f.wait_stamp(st, timeout_ms=30000, label="1024 buffers deleted")
    s.fault_state("4096 deleted")
    s.events("4096 deletes", blocks_digest=True)
    cb3 = abi.SEG_HDR.pack(8 + FILL_REC, abi.ENC_BLIT, 0, 0) + rec_fill(NOBJ, 0, 0x10, 0x66)
    sp.write(cb_va + 0x2000, cb3)
    s.submit(abi.OP_EXEC_INDIRECT3, exec3(1, [(cb_va + 0x2000, len(cb3))], [(NOBJ, 0)]), fault="either",
             label="buffer 4096 after its delete")
    s.fault_state("deleted buffer")
    sp.dump(out_va, 0x20, "buffer 4096's fill after its delete")
    cb4 = abi.SEG_HDR.pack(8 + FILL_REC + 0x28, abi.ENC_BLIT, 0, 0) + rec_fill(hbufs[3], 0, 0x10, 0x77) + rec_copy(
        hbufs[3], hbufs[0], 0, 0, 0x10)
    sp.write(cb_va + 0x3000, cb4)
    s.submit(abi.OP_EXEC_INDIRECT3, exec3(1, [(cb_va + 0x3000, len(cb4))], [(heap, 0)] + [(h, 0) for h in hbufs]),
             label="the heap buffers after the deletes")
    s.fault_state("end")
    s.events("after the deletes", blocks_digest=True)
    return s


def rec_fill(ref, off, n, byte):
    """Blit 306: {u32 buffer, u64 offset, u64 length, u8 value}."""
    return abi.REC_HDR.pack(306, FILL_REC) + struct.pack("<IQQB3x", ref, off, n, byte)


def rec_copy(src, dst, soff, doff, n):
    """Blit 301: {u32 src, u32 dst, u64 srcOffset, u64 dstOffset, u64 size}."""
    return abi.REC_HDR.pack(301, 0x28) + struct.pack("<IIQQQ", src, dst, soff, doff, n)
