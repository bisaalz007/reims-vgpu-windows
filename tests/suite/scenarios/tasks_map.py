"""MapMemory2 (57) and UnmapMemory (34) in each setup, the blocks setup's
map_memory walk and its coalescing, the range tracker behind lazy mapping, and
the unmap of overlapping object lists (docs/testing.md, sections 4.5, 6.10,
7.1)."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import ObjectList, Task, buffer_record, seg
from pgtest.tasks_lib import (XGPUTask, XTask, exec3_payload, fault_and_resume,
                              fill_rec, map_raw, maybe_fault, probe_list,
                              set_list_raw, unmap_raw)

U64 = 1 << 64


@scenario
def tasks_map_ranges():
    """Map and unmap ranges, valid and not. In the blocks setup a map walks
    the page table and calls the host's map_memory, and refuses a misaligned
    offset or length, a zero length, a range beyond the task or one that
    overflows; in the map and isolate setups a map is a no-op that always
    completes. An unmap checks the range against the task in-process (blocks
    and map fault), the range tracker refuses misaligned and empty ranges
    (blocks only), and the service does no check at all. An unknown task ID
    faults in every setup ("Invalid task")."""
    s = Scenario("tasks_map_ranges", doc=tasks_map_ranges.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=8)
    t.define()
    t.map()
    t.unmap()
    t.map(2 * PAGE, 3 * PAGE)
    s.events("valid map, unmap, map")
    M, U = abi.OP_MAP_MEMORY2, abi.OP_UNMAP_MEMORY
    cases = [
        (M, 0x100, PAGE, "map, misaligned offset"),
        (M, 0, 0x100, "map, misaligned length"),
        (M, 0, 0, "map, zero length"),
        (M, 7 * PAGE, 2 * PAGE, "map, beyond the task"),
        (M, U64 - PAGE, 2 * PAGE, "map, offset + length overflows"),
        (U, 0x100, PAGE, "unmap, misaligned offset"),
        (U, 0, 0x100, "unmap, misaligned length"),
        (U, 0, 0, "unmap, zero length"),
        (U, 7 * PAGE, 2 * PAGE, "unmap, beyond the task"),
        (U, 8 * PAGE, 0, "unmap, empty at the task's end"),
        (U, U64 - PAGE, 2 * PAGE, "unmap, offset + length overflows"),
    ]
    for op, off, n, label in cases:
        maybe_fault(s, op, struct.pack("<IQQ", 1, off, n), label)
    s.events("the edge cases")
    map_raw(s, 2, 0, PAGE, fault=True)
    fault_and_resume(s, "map, unknown task")
    unmap_raw(s, 2, 0, PAGE, fault=True)
    fault_and_resume(s, "unmap, unknown task")
    t.delete()
    s.diff("end")
    s.events("end")
    return s


def coalesce_task(s, tid, depth):
    """Twelve pages whose leaves test the blocks setup's run building: three
    physically contiguous writable pages, two read-only ones, a jump, a
    writable pair, a page that goes backwards, a read-only page, and a
    writable pair."""
    base = s.alloc_page(16)
    phys = [0, 1, 2, 3, 4, 10, 11, 12, 8, 9, 13, 14]
    ro = {3, 4, 5, 9}
    leaves = {v: base + p for v, p in enumerate(phys)}
    return XTask(s, tid=tid, pages=12, depth=depth, leaves=leaves,
                 writable={v for v in range(12) if v not in ro})


@scenario
def tasks_map_coalesce():
    """The blocks setup's map_memory calls for a map of a whole task at depth
    1, 2 and 3 (their page tables read through read_memory, a partial read of
    the root table and whole lower tables): a run is extended only while the
    next page follows the run's *first* page, so runs are at most 2 pages,
    and a run flushed mid-walk is mapped with the *next* page's writability
    (section 7.1). Then a map across a hole: the pages before it are mapped,
    then "Found a hole in the page table" faults the command. In the other
    setups every map completes and no callback is called."""
    s = Scenario("tasks_map_coalesce", doc=tasks_map_coalesce.__doc__)
    s.bring_up()
    for tid, depth in ((1, 1), (2, 2), (3, 3)):
        t = coalesce_task(s, tid, depth)
        t.define()
        t.map()
        s.events("depth %d" % depth)
    h = XTask(s, tid=4, pages=8, holes={5})
    h.define()
    maybe_fault(s, abi.OP_MAP_MEMORY2, struct.pack("<IQQ", 4, 0, 8 * PAGE), "map across a hole")
    h.map(0, 4 * PAGE)
    s.events("across a hole")
    s.diff("end")
    return s


@scenario
def tasks_map_depths():
    """Page-table walks of each depth across table boundaries: depth 2 over
    VA pages 4094-4097 (two level-1 tables) and depth 3 over the same pages
    (two level-2 tables under one level-1 table), each leaf its own scattered
    guest page. A blit fill of a buffer spanning the boundary lands in
    exactly the guest pages the leaves name (the page-table walk's descent
    and climb; the task's buffer mapping, in-process and in the service)."""
    s = Scenario("tasks_map_depths", doc=tasks_map_depths.__doc__)
    s.bring_up()
    for tid, depth in ((1, 2), (2, 3)):
        mapped = list(range(8)) + [4094, 4095, 4096, 4097]
        t = XGPUTask(s, tid=tid, pages=4098, depth=depth, mapped=mapped, scatter=True)
        ref, va = t.buffer(length=4 * PAGE, va=4094 * PAGE)
        t.define()
        t.install()
        cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(ref, PAGE - 0x10, 2 * PAGE + 0x20, 0x5A)))
        s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(tid, [cb], [(ref, 0)]))
        for v, off in ((4094, PAGE - 0x20), (4095, 0), (4096, PAGE - 0x20), (4097, 0)):
            t.dump(v * PAGE + off, 0x20, "depth %d, VA page %d + %#x" % (depth, v, off))
        t.delete()
    s.events("end")
    return s


@scenario
def tasks_map_unmapped_holes():
    """Lazy mapping of a long range through the object list: 16 mapped pages
    with a hole before, between and after them is 17 unmapped runs, one past
    what the tracker's walk of unmapped ranges holds ("Unmapped range too
    complex"): the blocks setup cannot map the list and the exec faults. One
    run fewer maps each hole with its own walk. The other setups map the list
    whole every time."""
    s = Scenario("tasks_map_unmapped_holes", doc=tasks_map_unmapped_holes.__doc__)
    s.bring_up()
    base = s.alloc_page()
    t = XTask(s, tid=1, pages=40, alias=base)
    t.define()
    for p in range(2, 34, 2):
        t.map(p * PAGE, PAGE)
    s.events("16 single pages mapped")
    probe_list(s, 1, 0, 35, "17 holes", expect="maybe")
    s.events("17 holes")
    probe_list(s, 1, 2, 33, "16 holes", expect="maybe")
    s.events("16 holes")
    probe_list(s, 1, 0, 35, "the rest", expect="maybe")
    s.events("the rest")
    t.delete()
    return s


@scenario(setups=("blocks",))
def tasks_map_tracker():
    """The range tracker's bookkeeping, probed through lazy mapping (blocks
    only: the tracker exists only for tasks over the host's task blocks; in
    the other setups the same maps and unmaps are the no-ops tasks_map_ranges
    locks). Each case clears the task, maps and unmaps a pattern, then maps
    an object list over a window; the map_memory calls that follow are the
    runs the tracker thought unmapped. Adjacent runs are kept apart,
    overlapping ones merge, a nested one is dropped, a containing one
    absorbs, an unmap splits, trims or removes runs (each case of the run
    comparison, and the run block's add and remove)."""
    s = Scenario("tasks_map_tracker", setups=("blocks",), doc=tasks_map_tracker.__doc__)
    s.bring_up()
    base = s.alloc_page()
    N = 24
    t = XTask(s, tid=1, pages=N, alias=base)
    t.define()
    M, U = "map", "unmap"
    cases = [
        ("overlapping", [(M, 2, 4), (M, 4, 4)]),
        ("adjacent", [(M, 2, 2), (M, 4, 2)]),
        ("nested", [(M, 2, 8), (M, 4, 2)]),
        ("containing", [(M, 4, 2), (M, 2, 8)]),
        ("added before", [(M, 8, 2), (M, 2, 2)]),
        ("joining three", [(M, 2, 2), (M, 6, 2), (M, 10, 2), (M, 3, 8)]),
        ("covering two exactly", [(M, 2, 2), (M, 6, 2), (M, 2, 6)]),
        ("same run twice", [(M, 4, 2), (M, 4, 2)]),
        ("split by an unmap", [(M, 0, 12), (U, 4, 2)]),
        ("unmap trims the start", [(M, 4, 6), (U, 2, 4)]),
        ("unmap trims the end", [(M, 4, 6), (U, 8, 4)]),
        ("unmap removes one run", [(M, 2, 2), (M, 6, 2), (U, 6, 2)]),
        ("unmap spans runs", [(M, 2, 2), (M, 6, 2), (M, 10, 2), (U, 3, 8)]),
        ("unmap before and after", [(M, 6, 2), (U, 0, 2), (U, 12, 2)]),
        ("unmap an edge of a run", [(M, 4, 4), (U, 4, 2)]),
    ]
    for label, steps in cases:
        t.unmap()
        for kind, p, n in steps:
            (t.map if kind == M else t.unmap)(p * PAGE, n * PAGE)
        s.events("set up: " + label)
        probe_list(s, 1, 0, 14, label)
        s.events("probe: " + label)
        probe_list(s, 1, 0, 14, label + ", again")
        s.events("probe again: " + label)
    t.delete()
    return s


@scenario(setups=("blocks",))
def tasks_map_tracker_blocks():
    """More runs than a run block holds (240): 240 single-page runs fill the
    first block, the next run past them starts a second, and a run inside the
    full first block splits it. Then an unmap of the whole task (section
    7.1): it removes the first block and stops, so the second block's runs
    stay recorded although the host unmapped them; an object list there is
    not mapped again, reads the host's zero-filled pages, and its entry is
    "invalid", while an object list in the first block's range is mapped
    again and works. Blocks only, for the reason tasks_map_tracker gives."""
    s = Scenario("tasks_map_tracker_blocks", setups=("blocks",), doc=tasks_map_tracker_blocks.__doc__)
    s.bring_up()
    alias = s.alloc_page()
    special = {600: s.alloc_page(), 601: s.alloc_page(), 602: s.alloc_page(),
               100: s.alloc_page(), 101: s.alloc_page(), 102: s.alloc_page()}
    leaves = {v: alias for v in range(1024)}
    leaves.update(special)
    t = XTask(s, tid=1, pages=1024, leaves=leaves)
    t.define()
    for p in range(0, 480, 2):
        t.map(p * PAGE, PAGE)
    s.events("240 runs")
    for p in (600, 601, 602):
        t.map(p * PAGE, PAGE)
    t.map(1 * PAGE, PAGE)
    t.map(3 * PAGE, PAGE)
    s.events("a second block, and a split")
    # Objects for both probes: list page L, record page L+1 (a buffer over
    # page L+2), filled by a command buffer read from page 4. The first list
    # names the buffer as reference 1, the second as reference 2: objects
    # are cached by reference, so the second must not reuse the first's.
    cbs = []
    for n, L in enumerate((100, 600)):
        ol = ObjectList(t, L * PAGE, (L + 1) * PAGE)
        if n:
            ol.add_raw(abi.T_BUFFER, 16, (L + 1) * PAGE + 0x100)
        ol.add(abi.T_BUFFER, buffer_record(PAGE, L + 2))
        ol.write()
        data = seg(abi.ENC_BLIT, fill_rec(n + 1, 0, 0x40, 0xC3))
        t.write(4 * PAGE + 0x100 * n, data)
        cbs.append((4 * PAGE + 0x100 * n, len(data)))
    t.unmap()
    s.events("unmapped whole")
    set_list_raw(s, 1, 100, 24)
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cbs[0]], [(1, 0)]))
    t.dump(102 * PAGE, 0x50, "the first block's buffer, filled")
    set_list_raw(s, 1, 600, 36)
    st = s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cbs[1]], [(2, 0)]), fault=True)
    fault_and_resume(s, "the second block's list reads zeros")
    s.fifo.wait_stamp(st)
    t.dump(602 * PAGE, 0x50, "the second block's buffer, untouched")
    s.events("probes")
    t.delete()
    return s


@scenario
def tasks_map_unmap_lists():
    """An unmap that overlaps the object list makes the host map it again on
    next use; one that does not leaves the old mapping in use. The guest
    points the list's page at another guest page holding a different list:
    until an unmap covers the list, the host keeps reading the old one
    (in-process and in the service). Each exec fills a buffer named by a
    reference not used before, so which list answered shows in which guest
    page the fill lands."""
    s = Scenario("tasks_map_unmap_lists", doc=tasks_map_unmap_lists.__doc__)
    s.bring_up()
    t = XGPUTask(s, tid=1, pages=24)
    # List A (the page table's page 0) and list B (a spare guest page): refs
    # 1-4 are buffers; A names VA pages 8-11, B names 12-15.
    refs_a = [t.buffer(length=PAGE, va=(8 + i) * PAGE)[0] for i in range(4)]
    t.ol.write()
    list_b_page = s.alloc_page()
    rec_b_va = 3 * PAGE
    b_entries = [(0, 0, 0)]
    for i in range(4):
        va = rec_b_va + 16 * i
        t.write(va, buffer_record(PAGE, 12 + i))
        b_entries.append((abi.T_BUFFER, 16, va))
    s.mem(s.pa(list_b_page), b"".join(abi.OBJ_ENTRY.pack(ty | (n << 8), va) for ty, n, va in b_entries))
    t.define()
    t.install()

    def fill(ref, label):
        t.reset_cmdbufs()
        cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(ref, 0, 0x10, 0xA0 + ref)))
        s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(ref, 0)]), label=label)

    fill(1, "ref 1 through list A")
    s.u32(t.table_gpa(0, 0, 0), s.pfn(list_b_page) | abi.PTE_WRITABLE)
    fill(2, "ref 2 after the leaf moved")
    unmap_raw(s, 1, 20 * PAGE, PAGE)
    fill(3, "ref 3 after an unmap elsewhere")
    unmap_raw(s, 1, 0, PAGE)
    fill(4, "ref 4 after an unmap of the list")
    for p in range(8, 16):
        t.dump(p * PAGE, 0x10, "VA page %d" % p)
    t.delete()
    s.events("end")
    return s


def build_blocks(s, t, n=481):
    """n single-page runs at the even pages from 0: 240 fill a run block."""
    for p in range(0, 2 * n, 2):
        t.map(p * PAGE, PAGE)
    s.events("%d runs" % n)


def tracker_steps(s, t, steps):
    for kind, p, n in steps:
        label = "%s [%d, %d)" % (kind, p, p + n)
        if kind == "map":
            t.map(p * PAGE, n * PAGE)
        elif kind == "unmap":
            t.unmap(p * PAGE, n * PAGE)
        else:
            probe_list(s, 1, p, n, label)
        s.events(label)


@scenario(setups=("blocks",))
def tasks_map_tracker_multiblock():
    """The range tracker across several run blocks (blocks only, for the
    reason tasks_map_tracker gives): 481 single-page runs fill two blocks and
    start a third; then maps that span blocks or merge many runs inside one,
    unmaps inside a block, across two, of a whole block and past the end,
    lazy probes over several blocks (the map_memory calls each probe makes are
    what the tracker thought unmapped). The last probe, over [0, 1010) after
    a map of [0, 1000) across the fragmented blocks, kills the host: adding a
    lazily mapped run reaches one of the tracker's aborts."""
    s = Scenario("tasks_map_tracker_multiblock", setups=("blocks",), doc=tasks_map_tracker_multiblock.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=1100, alias=s.alloc_page())
    t.define()
    build_blocks(s, t)
    tracker_steps(s, t, [
        ("probe", 0, 20),
        ("map", 470, 20),
        ("map", 700, 100),
        ("unmap", 100, 20),
        ("unmap", 450, 50),
        ("unmap", 960, 1),
        ("probe", 440, 80),
        ("map", 1000, 4),
        ("unmap", 1050, 10),
        ("unmap", 999, 6),
        ("map", 0, 1000),
        ("probe", 0, 1010),
    ])
    return s


@scenario(setups=("blocks",))
def tasks_map_tracker_multiblock_unmap():
    """The range tracker across run blocks, continued (blocks only): an unmap
    of the whole task removes only the first block (section 7.1), so a probe
    in the second block's range maps nothing; unmaps that trim blocks from
    either side; then 17 single-page runs, and a probe over them with 18
    unmapped holes ("Unmapped range too complex" from inside the scan) and
    one with 16, which maps."""
    s = Scenario("tasks_map_tracker_multiblock_unmap", setups=("blocks",),
                 doc=tasks_map_tracker_multiblock_unmap.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=1100, alias=s.alloc_page())
    t.define()
    build_blocks(s, t)
    tracker_steps(s, t, [
        ("unmap", 0, 1100),
        ("probe", 600, 10),
        ("unmap", 470, 30),
        ("unmap", 900, 100),
        ("unmap", 0, 1100),
        ("unmap", 0, 1100),
        ("probe", 700, 20),
    ])
    t.unmap()
    for p in range(2, 36, 2):
        t.map(p * PAGE, PAGE)
    probe_list(s, 1, 0, 40, "18 holes", expect="fault")
    s.events("18 holes")
    probe_list(s, 1, 2, 33, "16 holes")
    s.events("16 holes")
    t.delete()
    return s
