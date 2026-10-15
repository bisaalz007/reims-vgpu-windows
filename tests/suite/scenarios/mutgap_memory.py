"""The blocks setup's mapped-range tracker where the suite's other tracker
scenarios leave a choice unobserved: a run block holding more runs than an
add allows, a remove whose binary search stops past the run it should split,
a block that ends exactly where a new run ends, a full last block and whether
a run past it opens a new block or splits it, and compaction next to an
emptied block or leaving a stale span (docs/testing.md, sections 6.10 and
7.1).

Blocks only: the tracker exists only for tasks over the host's task blocks;
in the map and isolate setups maps and unmaps go through the memory map,
which has none. What the guest sees of the tracker is which ranges the host
is asked to map when an object list is used (probe_list: the lazy path,
test_range then visit_unmapped_ranges), whether a map faults ("Range too
complex"), and whether the host dies at one of its asserts.
"""
from pgtest import abi
from pgtest.abi import PAGE
from pgtest.newops_util import AliasedTask, bring_up_big
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.resources_build import RTask, b_fill_buf
from pgtest.task import seg
from pgtest.tasks_lib import XTask, exec3_payload, probe_list

BLOCKS = ("blocks",)


def steps(s, t, plan):
    """Maps, unmaps (in pages) and lazy probes (probe_list: an object list
    over the pages, mapped by an exec with no command buffers), each
    followed by the events it caused: the host's map_memory calls show what
    the tracker thought unmapped."""
    for kind, p, n in plan:
        label = "%s [%d, %d)" % (kind, p, p + n)
        if kind == "map":
            t.map(p * PAGE, n * PAGE)
        elif kind == "unmap":
            t.unmap(p * PAGE, n * PAGE)
        else:
            probe_list(s, t.tid, p, n, label)
        s.events(label)


@scenario(setups=BLOCKS)
def mutgap_mem_tracker_block_256_runs():
    """A run block holding 256 runs, then a remove on it. Adds stop at 240
    runs a block (add_to_range), but an unmap inside a run splits it in place
    (RunBlock::remove's Contains case), with no limit but the array's 256.
    240 three-page runs at pages 4i fill one block; unmapping the middle page
    of 16 of them takes it to exactly 256 runs (the 16th insert still finds
    255 < 256). The next unmap of a whole run in that block reaches
    RunBlock::remove with 256 runs, and its assert (fewer than 256 runs)
    kills the host, although that remove would not have split anything."""
    s = Scenario("mutgap_mem_tracker_block_256_runs", setups=BLOCKS,
                 doc=mutgap_mem_tracker_block_256_runs.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=1024, alias=s.alloc_page())
    t.define()
    for i in range(240):
        t.map(4 * i * PAGE, 3 * PAGE)
    s.events("240 runs of three pages", blocks_digest=True)
    for i in range(16):
        t.unmap((4 * i + 1) * PAGE, PAGE)
    s.events("16 runs split: 256 runs in the block", blocks_digest=True)
    s.r(abi.R_FAULT_MASK, "no fault")
    t.unmap(400 * PAGE, 3 * PAGE, label="unmap a whole run of the 256-run block")
    s.events("after")
    return s


@scenario(setups=BLOCKS)
def mutgap_mem_tracker_remove_search():
    """RunBlock::remove's binary search can stop on a run after the one that
    contains the removed range, and the scan from there then never sees it.
    Runs [0,1), [4,5), [8,12) and [16,17); an unmap of page 10: the search
    ends at index 3 ([16,17), which compares Greater and is skipped), so
    [8,12) is not split and the tracker still counts page 10 as mapped,
    although the host was asked to unmap it. A probe of page 10 then asks the
    host to map nothing; a probe of page 12, never mapped, maps it."""
    s = Scenario("mutgap_mem_tracker_remove_search", setups=BLOCKS,
                 doc=mutgap_mem_tracker_remove_search.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=64, alias=s.alloc_page())
    t.define()
    steps(s, t, [
        ("map", 0, 1),
        ("map", 4, 1),
        ("map", 8, 4),
        ("map", 16, 1),
        ("unmap", 10, 1),
        ("probe", 10, 1),
        ("probe", 12, 1),
    ])
    t.delete()
    s.events("end")
    return s


@scenario(setups=BLOCKS)
def mutgap_mem_tracker_block_end_edge():
    """A new run that ends exactly where a later block ends. 481 one-page
    runs at the even pages 0-960 fill blocks [0, 479) and [480, 959) and
    start a third at 960. A map of [479, 959) is adjacent to the full first
    block, which splits; it joins the split's second half, and the next
    block, [480, 959), shares only its end with the run: Run::compare calls
    that InsideEdge, so add_to_range removes the run from it and leaves it
    empty, not removed. An empty block compares Greater than everything and
    stops test_range and visit_unmapped_ranges: a probe of page 960, mapped
    since the start, asks the host to map it again, and so does a second one,
    since the run it adds is already in the last block (RunBlock::add returns
    on Equal) and the empty block stays."""
    s = Scenario("mutgap_mem_tracker_block_end_edge", setups=BLOCKS,
                 doc=mutgap_mem_tracker_block_end_edge.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=1100, alias=s.alloc_page())
    t.define()
    for p in range(0, 962, 2):
        t.map(p * PAGE, PAGE)
    s.events("481 runs", blocks_digest=True)
    steps(s, t, [
        ("map", 479, 480),
        ("probe", 960, 1),
        ("probe", 960, 1),
    ])
    t.delete()
    s.events("end")
    return s


@scenario(setups=BLOCKS)
def mutgap_mem_tracker_full_last_block():
    """A run past a full last block opens a new block (add_to_range) rather
    than splitting it. 240 one-page runs at the even pages 0-478 fill block
    0; a map of page 600 starts block 1. An unmap of the whole task then
    removes only the block it contains whole, the first, and stops
    (remove_from_range; docs/testing.md, section 7.1): all of pages 0-478
    count as unmapped, so a probe of page 300 asks the host to map it, while
    one of page 600, which the tracker still records, maps nothing."""
    s = Scenario("mutgap_mem_tracker_full_last_block", setups=BLOCKS,
                 doc=mutgap_mem_tracker_full_last_block.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=1024, alias=s.alloc_page())
    t.define()
    for p in range(0, 480, 2):
        t.map(p * PAGE, PAGE)
    s.events("240 runs", blocks_digest=True)
    steps(s, t, [
        ("map", 600, 1),
        ("unmap", 0, 1024),
        ("probe", 300, 1),
        ("probe", 600, 1),
    ])
    t.delete()
    s.events("end")
    return s


@scenario(setups=BLOCKS)
def mutgap_mem_tracker_compact_empty_block():
    """Compaction next to an emptied run block kills the host. As in
    tasks_gap_tracker_empty_block, a full block of 240 runs from page 100, a
    block at page 700, a run at 650 that splits the first, and a map of [640,
    701) that removes 700's run from the last block, which stays in the list,
    empty; one more run at page 101 makes block 0 hold 121. An empty block
    compares Greater than everything, so the 15,119 one-page runs that
    follow, at every other page from 702, all go to the block before it,
    which splits each time it is full (the comparison is Greater there) until
    there are 128 blocks. The next run finds that block full and 128 blocks,
    and compact packs the blocks from the front; their total is not a
    multiple of 240, so the last non-empty block is left short with the empty
    block after it, and the steal of nothing asserts (compact asserts it
    steals at least one run): the host dies at that map. Blocks only; one
    task of 15 x 4096 pages, every leaf one RAM page
    (newops_util.AliasedTask), the runs written with mem_seq into the largest
    root FIFO."""
    s = Scenario("mutgap_mem_tracker_compact_empty_block", setups=BLOCKS, watchdog_s=120,
                 doc=mutgap_mem_tracker_compact_empty_block.__doc__)
    f = bring_up_big(s)
    t = AliasedTask(s, tid=1, tables=15)
    t.define()
    st = t.map_seq(f, 240, 100, 2)
    f.doorbell()
    f.wait_stamp(st, timeout_ms=60000)
    for vpn, n in ((700, 1), (650, 1), (640, 61), (101, 1)):
        t.map(vpn, n)
    s.r(abi.R_FAULT_MASK, "no fault: an empty block after two")
    s.events("the empty block", blocks_digest=True)
    half = f.ring_size() // 32 // 2
    total = 15119
    k = 0
    while k < total:
        n = min(half, total - k)
        st = t.map_seq(f, n, 702 + 2 * k, 2)
        f.doorbell()
        f.wait_stamp(st, timeout_ms=60000)
        k += n
    s.r(abi.R_FAULT_MASK, "no fault after 15,119 runs")
    s.events("15,119 runs: 128 blocks", blocks_digest=True)
    t.map(702 + 2 * total, label="the run that compacts")
    s.events("after", blocks_digest=True)
    return s


@scenario(setups=BLOCKS)
def mutgap_mem_tracker_compact_stale_span():
    """Compaction leaves the last block's span right. A task of 16 x 4096
    pages, every leaf one RAM page, and 30,720 one-page runs at every other
    page fill 128 blocks of 240 (as newops_range_tracker_limit). 200 runs
    unmapped in each of blocks 10 and 11; a map into a gap near the end of
    block 127 finds 128 blocks and compacts: block 10 takes block 11's 40
    runs and it is removed, and every later block takes 160 runs from the
    next, whose span is set to its remaining runs; the last, block 126 now,
    keeps 80 runs and the span of just those, [61280, 61439), and the map
    adds a run to it. 159 one-page runs after it, from page 61440, fill it to
    240 and its span to [61280, 61757). A run at page 61758 is past it
    (Less), and with 127 blocks it opens block 127. A map into a gap in the
    first half of block 126 (page 61281) then finds it full and 128 blocks,
    compacts, which frees nothing, and faults "Range too complex" after the
    host mapped the page. Blocks only; block calls as counts and a digest."""
    s = Scenario("mutgap_mem_tracker_compact_stale_span", setups=BLOCKS, watchdog_s=120,
                 doc=mutgap_mem_tracker_compact_stale_span.__doc__)
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
    s.events("30,720 runs mapped", blocks_digest=True)

    def blk(b, i):
        return 2 * (240 * b + i)

    t.unmap(blk(10, 20), 400)
    t.unmap(blk(11, 20), 400)
    t.map(blk(127, 200) + 1)
    s.r(abi.R_FAULT_MASK, "compacted to 127 blocks")
    s.events("unmaps in blocks 10 and 11, a map that compacts", blocks_digest=True)
    st = t.map_seq(f, 159, 61440, 2)
    f.doorbell()
    f.wait_stamp(st)
    s.r(abi.R_FAULT_MASK, "the last block filled")
    t.map(61758)
    s.r(abi.R_FAULT_MASK, "a run past the full last block")
    s.events("159 runs, then one past them", blocks_digest=True)
    t.map(61281, fault=True, label="a map into block 126's first half")
    s.fault_state("range too complex")
    s.resume_root()
    s.events("the last map", blocks_digest=True)
    return s


# ---- task memory as the cursors, memory map and page-table walk see it ----

@scenario(setups=("isolate",), host=True)
def mutgap_mem_reply_split():
    """Info replies written across a page boundary where the next VA page's
    guest page is not adjacent (a leaf apart): the service's writable cursor
    over the reply buffer has two ranges there, and a sequential write that
    fills the first range goes on at the start of the second. 451 (heap
    texture size and align, 16 bytes) at reply offset 0x3ff8 puts 8 bytes at
    the end of VA page 8 and 8 at the start of VA page 9; a second 451 at
    0x4010 lands wholly in page 9. Isolate only: in-process a reply aborts
    the host (info_reply_aborts_host). The values are the host's (s.metal
    gives them)."""
    from pgtest.exec_objects import info_heap_size
    from pgtest.tasks_lib import XGPUTask
    s = Scenario("mutgap_mem_reply_split", setups=("isolate",), host=True, doc=mutgap_mem_reply_split.__doc__)
    s.bring_up()
    t = XGPUTask(s, tid=1, pages=12, scatter=True)
    r, r_va = t.buffer(length=2 * PAGE, va=8 * PAGE)
    t.fill(r_va + PAGE - 0x10, 0x10, 0xEE)
    t.fill(r_va + PAGE, 0x30, 0xEE)
    t.define()
    t.install()
    desc = dict(fmt=70, w=256, h=256, usage=1, options=0x20)
    cb = t.cmdbuf(seg(abi.ENC_INFO, info_heap_size(r, PAGE - 8, **desc) + info_heap_size(r, PAGE + 0x10, **desc)))
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb]), label="two 451 replies")
    t.dump(r_va + PAGE - 0x10, 0x10, "the end of VA page 8")
    t.dump(r_va + PAGE, 0x30, "the start of VA page 9")
    s.dump(s.pa(t.leaves[8] + 1), 0x10, "the guest page after VA page 8's")
    s.metal("heapTextureSizeAndAlign", **{"format": 70, "w": 256, "h": 256, "usage": 1, "options": 0x20})
    t.delete()
    s.events("end")
    return s


def _icb_split_task(s, pages=96):
    from scenarios.icb import ICBTask
    return ICBTask(s, pages=pages)


@scenario(setups=("isolate",))
def mutgap_mem_icb_args_split():
    """An ICB write-back whose command crosses a page boundary between guest
    pages that are not adjacent. A render ICB of 57 commands with a guest
    layout of commandSize 0x120 and draw arguments at 0xf8: command 56 starts
    at 0x3f00, so its 0x22 bytes of arguments start 8 bytes before the page's
    end. The ICB's second VA page is moved to a guest page far from its first
    (its leaf rewritten before define). Listed with invalidateGuest it is
    prepared and marked guest-invalid; the guest's copy around the boundary
    is then set to 0xee, and SynchronizeResources (53) writes the commands
    back through the service's writable cursor: the arguments are written at
    an offset into the command sub-cursor, and the part past the first
    range's end continues at the start of the second, leaving the guest page
    that follows the first one in RAM (set to 0x55) as it was. Isolate only:
    in-process there is no writable cursor."""
    import struct as st
    from pgtest.exec_objects import ICB_DRAW, icb_descriptor, icb_record
    from pgtest.icb_cmds import layout
    from scenarios.icb import sync_refs
    s = Scenario("mutgap_mem_icb_args_split", setups=("isolate",), doc=mutgap_mem_icb_args_split.__doc__)
    s.bring_up()
    w = _icb_split_task(s)
    size, n = 0x120, 57
    cmds = bytearray()
    for i in range(n):
        c = bytearray(size)
        st.pack_into("<III", c, 0, 1, 0, w.rpso)
        c[0xF8:0xF8 + 0x22] = st.pack("<HQQQQ", 3, i, 3, 1, 0)
        cmds += c
    va = w.t.reserve(2 * PAGE)
    w.t.write(va, bytes(cmds))
    lay = layout("render", commandArgumentsOffset=0xF8, commandSize=size)
    icb = w.t.ol.add(abi.T_ICB, icb_record(n * size, va // PAGE, icb_descriptor(ICB_DRAW, vbufs=1, fbufs=1), lay,
                                           max_count=n))
    far = s.alloc_page()
    vpn2 = va // PAGE + 1
    near = w.t.data + vpn2  # the guest page VA page vpn2 was made over
    s.mem(s.pa(far), bytes(cmds[PAGE:]))
    s.u32(s.pa(w.t.pt_root, 4 * vpn2), s.pfn(far) | abi.PTE_WRITABLE)
    w.start()
    w.t.exec_segments([seg(abi.ENC_BLIT)], resources=w.res() + [(icb, abi.RES_INVALIDATE_GUEST)],
                      label="prepared, guest invalidated")
    s.fault_state("prepared")
    s.fill(s.pa(near - 1, PAGE - 0x20), 0x20, 0xEE)
    s.fill(s.pa(far), 0x20, 0xEE)
    s.fill(s.pa(near), 0x20, 0x55)
    sync_refs(s, 1, [icb], label="synchronize")
    s.dump(s.pa(near - 1, PAGE - 0x20), 0x20, "the end of the ICB's first page")
    s.dump(s.pa(far), 0x20, "the start of its second page")
    s.dump(s.pa(near), 0x20, "the guest page after its first page")
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def mutgap_mem_icb_binds_at_end():
    """An ICB write-back whose kernel bind array ends exactly at the end of
    its command: a compute layout with kernelBufferBindOffset 0xec and one
    kernel bind (0x14 bytes), so the array is the command's last bytes. The
    sub-cursor for it fits (a sub-cursor may end at its parent's end), and
    SynchronizeResources (53) writes the bind (reference, GPU address,
    offset) back. Isolate only (the writable cursor is the service's); keyed
    by the host: the bind carries a GPU address."""
    import struct as st
    from pgtest.exec_objects import ICB_DISPATCH_THREADS, icb_bind, icb_descriptor, icb_record
    from pgtest.icb_cmds import compute_cmd, layout
    from scenarios.icb import sync_refs
    s = Scenario("mutgap_mem_icb_binds_at_end", setups=("isolate",), host=True,
                 doc=mutgap_mem_icb_binds_at_end.__doc__)
    s.bring_up()
    w = _icb_split_task(s)
    c = bytearray(compute_cmd(pso=w.cpso, grid=(4, 1, 1), tg=(4, 1, 1)))
    c[0xEC:0x100] = icb_bind(w.out, 0, 0x10)
    va = w.t.reserve(PAGE)
    w.t.write(va, bytes(c))
    lay = layout("compute", kernelBufferBindOffset=0xEC)
    icb = w.t.ol.add(abi.T_ICB, icb_record(0x100, va // PAGE, icb_descriptor(ICB_DISPATCH_THREADS, kbufs=1), lay))
    w.start()
    w.t.exec_segments([seg(abi.ENC_BLIT)], resources=w.res() + [(icb, abi.RES_INVALIDATE_GUEST)],
                      label="prepared, guest invalidated")
    s.fault_state("prepared")
    w.t.fill(va, 0x100, 0xEE)
    sync_refs(s, 1, [icb], label="synchronize")
    w.t.dump(va, 0x100, "the command written back")
    s.events("end")
    return s


def _zero_first(name, doc, setup):
    from pgtest.scenario import RAM_BASE, RAM_SIZE
    ranges = [[RAM_BASE, 0, RAM_BASE], [RAM_BASE, RAM_SIZE, RAM_BASE]]
    s = Scenario(name, setups=(setup,), descriptor={"map_ranges": [[hex(a), hex(n), hex(h)] for a, n, h in ranges]},
                 doc=doc)
    s.bring_up()
    s.submit(abi.OP_NOP)
    s.events("a NOP")
    return s


@scenario(setups=("map",))
def mutgap_mem_map_zero_length_first():
    """A memory map whose first range is zero-length and starts where the
    second, all of RAM, starts. The memory map sorts ranges by start with a
    strict comparison, so two ranges with one start keep their order, and the
    overlap check passes: the zero-length range ends where it starts. The
    device is created, a NOP runs, and a task's fill lands through the map.
    Map setup only: the blocks setup has no map, and in isolate the
    zero-length range faults the first define
    (mutgap_mem_map_zero_length_first_isolate)."""
    s = _zero_first("mutgap_mem_map_zero_length_first", mutgap_mem_map_zero_length_first.__doc__, "map")
    t = RTask(s, pages=12)
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_buf(b, 0, 0x10, 0x77), resources=[(b, 0)], label="a fill")
    t.dump(b_va, 0x10, "B")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",))
def mutgap_mem_map_zero_length_first_isolate():
    """The same memory map with isolation: the device is created and a NOP
    runs, but the first DefineTask2 faults: an isolated task needs every
    range of the map to be whole pages, and a zero-length range is not, so
    none is made. Locked as a fault at the define."""
    s = _zero_first("mutgap_mem_map_zero_length_first_isolate", mutgap_mem_map_zero_length_first_isolate.__doc__,
                    "isolate")
    t = RTask(s, pages=12)
    t.define()
    return s


@scenario
def mutgap_mem_icb_zero_length():
    """An ICB whose record gives it length 0, at a page-aligned VA past page
    0, listed in two execs. Preparing it asks the task for a cursor of length
    0: in-process the copy walks a size of 0 and the page-table walk refuses
    it ("Zero length") without touching the task's iterator; "Failed to make
    cursor for prepare", and the exec goes on. The service refuses any
    zero-length cursor itself. Execs that read task memory keep working, and
    so does the task's deletion."""
    from pgtest.exec_objects import ICB_DISPATCH_THREADS, icb_descriptor, icb_record
    from pgtest.icb_cmds import layout
    s = Scenario("mutgap_mem_icb_zero_length", doc=mutgap_mem_icb_zero_length.__doc__)
    s.bring_up()
    w = _icb_split_task(s)
    va = w.t.reserve(PAGE)
    icb = w.t.ol.add(abi.T_ICB, icb_record(0, va // PAGE, icb_descriptor(ICB_DISPATCH_THREADS, kbufs=1),
                                           layout("compute")))
    w.start()
    for n in range(2):
        w.t.exec_segments([seg(abi.ENC_BLIT, b_fill_buf(w.out, 0x10 * n, 0x10, 0x60 + n))],
                          resources=w.res() + [(icb, 0)], label="exec %d listing the ICB" % n)
        s.fault_state("exec %d" % n)
    w.t.exec_segments([seg(abi.ENC_BLIT, b_fill_buf(w.out, 0x20, 0x10, 0x70))], resources=w.res(),
                      label="an exec without it")
    w.t.dump(w.out_va, 0x30, "out")
    w.t.delete()
    s.events("end")
    return s
