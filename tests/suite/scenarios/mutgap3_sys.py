"""Boundaries and orders in the task service, the tasks, the mapped-range
tracker, the FIFO and the device: each scenario locks one, as its docstring
says.
"""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.newops_util import AliasedTask, bring_up_big
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.tasks_lib import (XTask, exec3_payload, heap_payload, heap_ser, maybe_fault, probe_list,
                              set_lists_raw)

OP64 = abi.OP_HEAP_TEXTURE_SIZE_AND_ALIGN


# ---- the service's reply mapping ----

@scenario
def mutgap3_sys_heap_reply_one_past_page():
    """HeapTextureSizeAndAlign (64) with a 16-byte reply at 2 * PAGE - 15:
    the reply ends one byte into VA page 2. The service maps the reply's
    pages with the end rounded up, (offset + length + 0x3fff) & ~0x3fff, so
    both pages are mapped and all 16 bytes land, the last at 2 * PAGE; the
    map setup's task maps with the same rounding. The blocks setup does not:
    its lazy map asks the range tracker for the run of pages, which rounds
    with 0x3ffe (calculate_page_run_unaligned), one page short when the range
    ends one byte into a page. Only VA page 1 is mapped (one map_memory), and
    the 16th byte goes to the task block's unmapped page 2: the guest's byte
    at 2 * PAGE stays 0xee. The bytes around the reply were 0xee. Host values
    (the size and alignment Metal reports)."""
    s = Scenario("mutgap3_sys_heap_reply_one_past_page", host=True,
                 doc=mutgap3_sys_heap_reply_one_past_page.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=8)
    t.define()
    t.fill(2 * PAGE - 0x20, 0x20, 0xEE)
    t.fill(2 * PAGE, 0x20, 0xEE)
    s.submit(OP64, heap_payload(1, 2 * PAGE - 15, 16, heap_ser()), label="reply at 2 * PAGE - 15")
    t.dump(2 * PAGE - 0x20, 0x20, "VA page 1's last 0x20 bytes")
    t.dump(2 * PAGE, 0x20, "VA page 2's first 0x20 bytes")
    t.delete()
    s.events("end")
    return s


# ---- which unmaps drop the placement list's mapping ----

@scenario
def mutgap3_sys_placement_unmap_bounds():
    """An UnmapMemory (34) drops the task's placement-list mapping only when
    its range overlaps the list: offset < list end and list offset < end (in
    the service and in an in-process task). The placement list is at VA 6 *
    PAGE, 48 bytes (SetObjectAndPlacementList, 68), and an exec maps both
    lists. What the guest sees of the mapping: VA page 6's leaf is then made
    a hole, and an exec with no command buffers re-walks the page table only
    for a list whose mapping was dropped ("Failed to map placement list"), so
    it faults if it was dropped and completes if the old mapping is kept.
    First an unmap that starts exactly at the list's end (6 * PAGE + 48, one
    page long), then an unmap of the list's own page; the leaf is restored
    after each. Where the setups differ (the in-process tasks, and an
    unaligned unmap), that is locked too; each command waits for either
    outcome."""
    s = Scenario("mutgap3_sys_placement_unmap_bounds", doc=mutgap3_sys_placement_unmap_bounds.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=12)
    t.define()
    leaf6 = t.table_gpa(0, 0, 6)
    good = s.pfn(t.leaves[6]) | abi.PTE_WRITABLE
    maybe_fault(s, abi.OP_SET_OBJECT_AND_PLACEMENT_LIST,
                struct.pack("<IQQII", 1, PAGE, 6 * PAGE, 36, 48), "lists at PAGE and 6 * PAGE")
    maybe_fault(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1), "an exec maps both lists")
    for label, off in (("an unmap from the list's end", 6 * PAGE + 48),
                       ("an unmap of the list's page", 6 * PAGE)):
        maybe_fault(s, abi.OP_UNMAP_MEMORY, struct.pack("<IQQ", 1, off, PAGE), label)
        s.u32(leaf6, 0)
        maybe_fault(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1), "exec over a hole, after " + label)
        s.u32(leaf6, good)
        maybe_fault(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1), "exec, leaf restored, after " + label)
    t.dump(6 * PAGE, 0x30, "the placement list")
    t.delete()
    s.events("end")
    return s


# ---- the mapped-range tracker ----

@scenario
def mutgap3_sys_tracker_17_inner_holes():
    """visit_unmapped_ranges holds 16 unmapped runs; a 17th inside the range
    is refused ("Unmapped range too complex"). Single pages 2, 4, ..., 34 are
    mapped (17 runs), then [35, 40), a run of its own (adjacent runs stay
    apart). A lazy probe of [0, 38) finds 17 holes, each before a mapped
    page, and no tail (the last run covers it), so the 17th is refused at the
    Inside case, not at the tail's check: the blocks setup cannot map the
    list and the exec faults with no map_memory call. The other setups map the
    list whole."""
    s = Scenario("mutgap3_sys_tracker_17_inner_holes", doc=mutgap3_sys_tracker_17_inner_holes.__doc__)
    s.bring_up()
    base = s.alloc_page()
    t = XTask(s, tid=1, pages=40, alias=base)
    t.define()
    for p in range(2, 36, 2):
        t.map(p * PAGE, PAGE)
    t.map(35 * PAGE, 5 * PAGE)
    s.events("17 single pages and [35, 40) mapped")
    probe_list(s, 1, 0, 38, "17 inner holes", expect="maybe")
    s.events("17 inner holes")
    t.delete()
    return s


@scenario(setups=("blocks",))
def mutgap3_sys_tracker_compact_keeps_runs():
    """Compaction moves the runs a block keeps after the previous block takes
    its first ones. The fill of mutgap_mem_tracker_compact_stale_span: 128
    blocks of 240 one-page runs at every other page of a 16 x 4096-page task,
    200 runs unmapped in each of blocks 10 and 11, and a map near the end of
    block 127 that compacts: block 10 takes block 11's 40 runs, then 160 of
    block 12's, which keeps its runs 160-239. A lazy probe of old block 12's
    run 200 then finds it mapped (no map_memory call), as does one of old
    block 60's run 10; a probe of an unmapped page of old block 12 (run 20's
    neighbour) maps just it. Blocks only: the tracker exists only there
    (mutgap_memory.py's docstring); block calls as counts and a digest."""
    s = Scenario("mutgap3_sys_tracker_compact_keeps_runs", setups=("blocks",), watchdog_s=120,
                 doc=mutgap3_sys_tracker_compact_keeps_runs.__doc__)
    f = bring_up_big(s)
    t = AliasedTask(s, tid=1, tables=16)
    t.define()
    total = 240 * 128
    half = f.ring_size() // 32 // 2
    k = 0
    while k < total:
        n = min(half, total - k)
        st = t.map_seq(f, n, 2 * k, 2)
        f.doorbell()
        f.wait_stamp(st, timeout_ms=60000)
        k += n
    s.r(abi.R_FAULT_MASK, "no fault after the fill")

    def blk(b, i):
        return 2 * (240 * b + i)

    t.unmap(blk(10, 20), 400)
    t.unmap(blk(11, 20), 400)
    t.map(blk(127, 200) + 1)
    s.r(abi.R_FAULT_MASK, "compacted to 127 blocks")
    s.events("unmaps in blocks 10 and 11, a map that compacts", blocks_digest=True)
    probe_list(s, 1, blk(12, 200), 1, "old block 12's run 200", expect="maybe")
    s.events("old block 12's run 200")
    probe_list(s, 1, blk(60, 10), 1, "old block 60's run 10", expect="maybe")
    s.events("old block 60's run 10")
    probe_list(s, 1, blk(12, 20) + 1, 1, "the page after old block 12's run 20", expect="maybe")
    s.events("the page after old block 12's run 20")
    return s


# ---- deleteFIFO waits for the tasks' answers without holding the task mutex ----

DEADLOCK_BUFFERS = 1300
DEADLOCK_NOPS = 50


@scenario(setups=("isolate",))
def mutgap3_sys_delete_fifo_during_exec():
    """Deleting a FIFO tells every task (fifoDeleted) and waits for all their
    answers on a condition variable, which releases the task mutex while it
    waits. The service answers fifoDeleted on the task's serial queue, behind
    an exec still preparing its resources, and that exec asks the host for a
    shared texture's handle, which takes the task mutex (to find the kernel
    task). Task 1 lists %d shared buffers and then a shared texture (type 16)
    over a backing in task 0; an exec on child FIFO 3 names them all, in that
    order, so the service spends its first milliseconds creating buffers. The
    FIFO thread waits for the service's reply to an exec, so the delete comes
    from another FIFO. Once FIFO 3's node shows the exec read (its read count
    is published before the command runs), the root FIFO runs %d NOPs (time
    for FIFO 3 to send the exec: the send looks the task up under the same
    mutex) and then deletes child FIFO 2 while the service is still at the
    buffers. Both complete: the handle request gets the mutex while the
    delete waits, and the delete's stamp comes after the exec's reply. (A
    delete that waited holding the mutex would hang, but only with that
    timing, the service still creating buffers when the delete takes the
    mutex; the outcome here does not depend on it.) Isolate only: the
    in-process tasks answer fifoDeleted at once."""
    from pgtest.fifo_util import child
    from pgtest.resources_build import RTask, buffer_rec, shared_backing_rec, shared_texture_rec, simple_dim
    from scenarios.textures_shared import define_both, desc1
    s = Scenario("mutgap3_sys_delete_fifo_during_exec", setups=("isolate",), watchdog_s=8,
                 doc=mutgap3_sys_delete_fifo_during_exec.__doc__)
    s.bring_up()
    k, t = RTask(s, tid=0, pages=32), RTask(s, tid=1, pages=12)
    b_va = k.area(PAGE, contents=bytes(range(64)))
    backing = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, b_va // PAGE, [simple_dim(4, 4)],
                                                             desc1(options=0x20, usage=3)))
    refs = [t.add(abi.T_BUFFER, buffer_rec(PAGE, 8)) for _ in range(DEADLOCK_BUFFERS)]
    refs.append(t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(backing)))
    define_both(k, t)
    child(s, 2)
    c3 = child(s, 3, pages=2)
    st3 = c3.cmd(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [], [(r, 0) for r in refs]))
    c3.doorbell()
    s.wait(gpa=c3.node_gpa(4), eq=c3.written, label="FIFO 3 has read its exec")
    for _ in range(DEADLOCK_NOPS):
        s.cmd(abi.OP_NOP)
    st = s.cmd(abi.OP_DELETE_CHILD_FIFO, struct.pack("<I", 2))
    s.doorbell()
    s.fifo.wait_stamp(st, label="the delete of FIFO 2")
    c3.wait_stamp(st3, label="the exec on FIFO 3")
    s.fault_state("after both")
    s.events("end")
    return s


mutgap3_sys_delete_fifo_during_exec.__doc__ %= (DEADLOCK_BUFFERS, DEADLOCK_NOPS)
