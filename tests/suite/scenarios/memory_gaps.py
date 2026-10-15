"""Guest memory as the host describes it: RAM in several ranges with holes
between them, task roots and page tables in any range, buffers physically
contiguous across range boundaries, leaves in holes and beyond every range,
and memory maps whose ranges alias other ranges' host memory, have no host
memory, or end inside a page (docs/testing.md, sections 3.3, 6.10)."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.fifo import ChildFIFO
from pgtest.gaps_memory import (R0_PAGES, Pool, fill_cb, fill_exec, page_of, placed_task,
                                redirect_writes)
from pgtest.registry import scenario
from pgtest.scenario import RAM_BASE, RAM_SIZE, Scenario
from pgtest.resources_build import b_copy_buf_buf
from pgtest.task import ObjectList, buffer_record, seg
from pgtest.tasks_lib import (XGPUTask, exec3_payload, fault_and_resume,
                              fill_rec, maybe_fault, set_list_raw)

# Guest RAM in five ranges: R0, the scenario's own (alloc_page hands out
# from it); a hole of 16 pages; R1 (8 pages), R2 (1 page) and R3 (4 pages),
# each starting where the one before ends; and R4, 4 pages at 4 GiB.
R0 = (RAM_BASE, RAM_SIZE)
HOLE = RAM_BASE + RAM_SIZE
R1 = (HOLE + 16 * PAGE, 8 * PAGE)
R2 = (R1[0] + R1[1], PAGE)
R3 = (R2[0] + R2[1], 4 * PAGE)
R4 = (0x100000000, 4 * PAGE)
FIVE = [R0, R1, R2, R3, R4]


def pg(r, i=0):
    """Page i of range r, as a page index from RAM_BASE."""
    return page_of(r[0]) + i


@scenario
def memory_gap_ram_holes():
    """Guest RAM in five ranges with a hole between the first and the rest. A
    depth-2 task whose task root and first-level table are in R1 and whose
    second-level table is in R4 (the page-table walk reads each through the
    range that holds it); a buffer of four pages physically contiguous across
    R1, R2 and R3 (the memory map maps it in three pieces), one in R4, and
    one whose page is in the hole. In the map and isolate setups the fill
    across the ranges lands in all four pages; in the blocks setup the
    page-table walk coalesces the four pages into one physical range, which
    the host (as QEMU's) cannot map across its ranges, so the command faults.
    The page in the hole faults in every setup: no range holds it
    ("Incomplete range" in-process; the host's map_memory fails in blocks)."""
    s = Scenario("memory_gap_ram_holes", ram=FIVE, doc=memory_gap_ram_holes.__doc__)
    s.bring_up()
    pool = Pool([pg(R1, 0), pg(R1, 1), pg(R4, 0)])
    leaves = {v: s.alloc_page() for v in range(8)}
    leaves.update({8: pg(R1, 7), 9: pg(R2), 10: pg(R3, 0), 11: pg(R3, 1),
                   12: page_of(HOLE) + 3, 13: pg(R4, 1)})
    t = placed_task(s, pool, cls=XGPUTask, tid=1, pages=14, depth=2, leaves=leaves)
    across, _ = t.buffer(length=4 * PAGE, va=8 * PAGE)
    hole, _ = t.buffer(length=PAGE, va=12 * PAGE)
    far, _ = t.buffer(length=PAGE, va=13 * PAGE)
    t.define()
    s.dump(s.pa(pg(R1, 0), 0x14), 8, "the task length, in the task root in R1")
    t.install()
    cb = fill_cb(t, across, PAGE - 0x10, 2 * PAGE + 0x20, 0x5A)
    maybe_fault(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(across, 0)]),
                "a buffer across R1, R2 and R3", timeout_ms=1000)
    s.dump(R2[0] - 0x20, 0x20, "R1's end")
    s.dump(R2[0], 0x10, "R2's start")
    s.dump(R3[0] - 0x10, 0x10, "R2's end")
    s.dump(R3[0], 0x10, "R3's start")
    s.dump(R3[0] + PAGE, 0x20, "R3's second page")
    fill_exec(s, t, far, 0, 0x20, 0x6B)
    s.dump(R4[0] + PAGE, 0x30, "the buffer in R4")
    fill_exec(s, t, hole, 0, 0x20, 0x7C, fault=True)
    fault_and_resume(s, "a buffer in the hole")
    s.events("fills")
    t.delete()
    s.events("deleted")
    return s


@scenario(setups=("map", "isolate"))
def memory_gap_alias_range():
    """A memory map with a second range, at 4 GiB, whose host memory is four
    pages of the first range: the library uses whatever host address the
    host gives a range. A task's page table and its command buffer's page are
    reached only through the alias, and a fill of a buffer over two aliased
    pages lands in the first range's pages. Map and isolate setups only: the
    blocks setup has no memory map."""
    alias = 0x100000000
    s = Scenario("memory_gap_alias_range", setups=("map", "isolate"),
                 doc=memory_gap_alias_range.__doc__)
    backing = s.alloc_page(4)
    s._config["descriptor"] = {"map_ranges": [[hex(RAM_BASE), hex(RAM_SIZE), hex(RAM_BASE)],
                                              [hex(alias), hex(4 * PAGE), hex(s.pa(backing))]]}
    s.bring_up()
    a = page_of(alias)
    leaves = {v: s.alloc_page() for v in range(8)}
    leaves.update({4: a + 2, 8: a + 0, 9: a + 1})
    pool = Pool([s.alloc_page(), a + 3])
    since = len(s.ops)
    t = placed_task(s, pool, cls=XGPUTask, tid=1, pages=10, leaves=leaves)
    buf, _ = t.buffer(length=2 * PAGE, va=8 * PAGE)
    t.define()
    t.install()
    cb = fill_cb(t, buf, PAGE - 0x10, 0x20, 0x3D)
    redirect_writes(s, since, alias, 4 * PAGE, s.pa(backing))
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(buf, 0)]))
    s.dump(s.pa(backing, PAGE - 0x10), 0x20, "the fill, in the first range's pages")
    t.delete()
    s.events("end")
    return s


NULL_HOST = [[hex(RAM_BASE), hex(RAM_SIZE), hex(RAM_BASE)], [hex(R4[0]), hex(R4[1]), "0x1"]]


@scenario(setups=("map", "isolate"))
def memory_gap_null_host_define():
    """A memory map with a range the host gives no memory (host address
    NULL): the device is created and runs, and in the map setup a task is
    defined and deleted. In the isolate setup the first define faults: an
    isolated task needs every range of the map to be whole pages at a
    page-aligned host address that is not NULL, so none is made. Map and
    isolate setups only: the blocks setup has no memory map."""
    s = Scenario("memory_gap_null_host_define", setups=("map", "isolate"),
                 descriptor={"map_ranges": NULL_HOST}, doc=memory_gap_null_host_define.__doc__)
    s.bring_up()
    s.submit(abi.OP_NOP)
    s.events("running")
    t = XGPUTask(s, tid=1, pages=8)
    t.define()
    s.events("defined")
    t.delete()
    s.events("deleted")
    return s


@scenario(setups=("map",))
def memory_gap_null_host_map():
    """In-process, a range the host gives no memory (host address NULL)
    cannot be mapped: mach_vm_remap from address 0 plus the offset fails. A
    buffer over one of its pages faults its exec (the whole range in one
    piece), and so does a buffer over its last page and the first page of the
    range after it, which has memory (the first of two pieces fails); a task
    whose task root page is there faults its define. A buffer over the
    following range alone, and one in the first range, work. Map setup only:
    the blocks setup has no memory map, and in the isolate setup the first
    define faults (memory_gap_null_host_define)."""
    s = Scenario("memory_gap_null_host_map", setups=("map",), doc=memory_gap_null_host_map.__doc__)
    backing = s.alloc_page(4)
    after = R4[0] + R4[1]
    s._config["descriptor"] = {"map_ranges": NULL_HOST + [[hex(after), hex(4 * PAGE), hex(s.pa(backing))]]}
    s.bring_up()
    leaves = {v: s.alloc_page() for v in range(9)}
    leaves.update({9: pg(R4, 1), 10: pg(R4, 3), 11: page_of(after), 12: page_of(after) + 1})
    t = XGPUTask(s, tid=1, pages=13, leaves=leaves)
    null_buf, _ = t.buffer(length=PAGE, va=9 * PAGE)
    across, _ = t.buffer(length=2 * PAGE, va=10 * PAGE)
    ok_buf, _ = t.buffer(length=PAGE, va=8 * PAGE)
    next_buf, _ = t.buffer(length=PAGE, va=12 * PAGE)
    t.define()
    t.install()
    fill_exec(s, t, null_buf, 0, 0x20, 0x11, fault=True)
    fault_and_resume(s, "a buffer over the range without memory")
    fill_exec(s, t, across, PAGE - 0x10, 0x20, 0x12, fault=True)
    fault_and_resume(s, "a buffer from it into the range after")
    fill_exec(s, t, next_buf, 0, 0x20, 0x13)
    s.dump(s.pa(backing + 1), 0x30, "the range after, through its backing")
    fill_exec(s, t, ok_buf, 0, 0x20, 0x22)
    t.dump(8 * PAGE, 0x30, "a buffer in the first range")
    u = XGPUTask(s, tid=2, pages=8)
    u.root = pg(R4, 0)
    u.define(fault=True)
    fault_and_resume(s, "a task root in the range without memory")
    s.r(abi.R_FIFO_READ)
    t.delete()
    s.events("end")
    return s


@scenario(setups=("map", "isolate"))
def memory_gap_partial_range():
    """A memory map whose only range ends 8 KiB into its last 16 KiB page. In
    the map setup the range is used as given: a command buffer in the page's
    first half is read (a copy through the memory map) and runs, and a buffer
    over that page cannot be mapped (the range covers half of it: "Incomplete
    range"), so its exec faults. In the isolate setup no isolated task is
    made: the range is not whole pages, which an isolated task needs, so the
    define faults, and so does every later command for the task ("Invalid
    task"). Map and isolate setups only: the blocks setup has no memory
    map."""
    s = Scenario("memory_gap_partial_range", setups=("map", "isolate"),
                 descriptor={"map_ranges": [[hex(RAM_BASE), hex(RAM_SIZE - 0x2000), hex(RAM_BASE)]]},
                 doc=memory_gap_partial_range.__doc__)
    s.bring_up()
    last = R0_PAGES - 1
    leaves = {v: s.alloc_page() for v in range(10)}
    leaves[4] = last
    leaves[9] = last
    t = XGPUTask(s, tid=1, pages=10, leaves=leaves)
    ok_buf, _ = t.buffer(length=PAGE, va=8 * PAGE)
    part_buf, _ = t.buffer(length=PAGE, va=9 * PAGE)
    st = t.define(fault="either", label="define")
    fault_and_resume(s, "define")
    s.fifo.wait_stamp(st)
    maybe_fault(s, abi.OP_SET_OBJECT_LIST, struct.pack("<III", 1, 0, t.ol.write()), "set the object list")
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(ok_buf, 0, 0x20, 0x33)))
    maybe_fault(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(ok_buf, 0)]),
                "a command buffer in the page's first half")
    t.dump(8 * PAGE, 0x30, "its fill")
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(part_buf, 0, 0x20, 0x44)))
    maybe_fault(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(part_buf, 0)]),
                "a buffer over the partly covered page")
    s.dump(s.pa(last, 0x100), 0x20, "that page, untouched")
    maybe_fault(s, abi.OP_DELETE_TASK, struct.pack("<I", 1), "delete")
    s.events("end")
    return s


MANY = HOLE + 64 * PAGE  # twelve one-page ranges from here, each where the last ends


@scenario
def memory_gap_many_ranges():
    """Guest RAM in thirteen ranges: the first, then twelve of one page each,
    physically adjacent, listed to the device in reverse order (the memory
    map sorts them). A child FIFO whose three ring pages are three of those
    ranges, with a command straddling each page boundary (the ring is one
    range mapping of three pieces); a depth-3 task whose task root and three
    tables are four more of them; and a buffer over the last five, one
    physical run across five ranges, filled from its first page to its last.
    In the map and isolate setups the fill lands in all five; in the blocks
    setup the host cannot map a range that crosses its ranges and the exec
    faults (as in memory_gap_ram_holes), while the child FIFO, whose pages
    the root task maps one range each, runs in every setup."""
    ram = [R0] + [(MANY + i * PAGE, PAGE) for i in reversed(range(12))]
    s = Scenario("memory_gap_many_ranges", ram=ram, doc=memory_gap_many_ranges.__doc__)
    s.bring_up()
    m = page_of(MANY)
    c = ChildFIFO(s, 1)
    c.pages = [m, m + 1, m + 2]
    c.setup_node()
    c.define()
    for n in range(3):
        c.cmd(abi.OP_NOP, pad_to=PAGE - 4 if n == 0 else PAGE)
    st = c.cmd(abi.OP_NOP)
    c.doorbell()
    c.wait_stamp(st, label="child FIFO: commands across its three ranges")
    s.dump(c.node_gpa(0), abi.CHILD_NODE_SIZE, "child node")
    pool = Pool([m + 3, m + 4, m + 5, m + 6])
    leaves = {v: s.alloc_page() for v in range(8)}
    leaves.update({8 + i: m + 7 + i for i in range(5)})
    t = placed_task(s, pool, cls=XGPUTask, tid=1, pages=13, depth=3, leaves=leaves)
    buf, _ = t.buffer(length=5 * PAGE, va=8 * PAGE)
    t.define()
    t.install()
    cb = fill_cb(t, buf, 0x10, 5 * PAGE - 0x20, 0x4E)
    maybe_fault(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(buf, 0)]),
                "a buffer across five ranges", timeout_ms=1000)
    for i in range(5):
        s.dump(s.pa(m + 7 + i), 0x20, "range %d's start" % (8 + i))
        s.dump(s.pa(m + 7 + i, PAGE - 0x20), 0x20, "range %d's end" % (8 + i))
    t.delete()
    c.delete()
    s.events("end")
    return s


@scenario
def memory_gap_shared_tables():
    """Two tasks over one page table: task 2's task root names task 1's
    page-table root, so the two see the same guest pages at the same VAs (the
    host walks whatever table the root names). Task 2's object list is at VA
    page 2, task 1's at 0. Task 1 fills a buffer at VA 8; task 2, whose
    buffer record names the same VA page, copies it into its own buffer at VA
    9: the copy shows the fill. Then task 2 fills the page and task 1 copies
    it. In the blocks setup each task maps the pages into its own host task;
    in the isolate setup each task has its own service, both over the one
    shared guest RAM."""
    s = Scenario("memory_gap_shared_tables", doc=memory_gap_shared_tables.__doc__)
    s.bring_up()
    t = XGPUTask(s, tid=1, pages=16)
    a, a_va = t.buffer(length=PAGE, va=8 * PAGE)
    back, back_va = t.buffer(length=PAGE, va=10 * PAGE)
    t.define()
    t.install()
    u = XGPUTask(s, tid=2, pages=16)
    u.write_root(pt_root=s.pfn(t.pt_root))
    ol = ObjectList(t, 2 * PAGE, 3 * PAGE)
    ua = ol.add(abi.T_BUFFER, buffer_record(PAGE, 8))
    uout = ol.add(abi.T_BUFFER, buffer_record(PAGE, 9))
    ol.write()
    u.define()
    set_list_raw(s, 2, 2, ol.length())
    cb = fill_cb(t, a, 0, 0x20, 0x5A)
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(a, 0)]), label="task 1 fills VA 8")
    t.write(6 * PAGE, seg(abi.ENC_BLIT, b_copy_buf_buf(ua, uout, 0, 0, 0x20)))
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(2, [(6 * PAGE, 8 + 0x30)], [(ua, 0), (uout, 0)]),
             label="task 2 copies VA 8 to VA 9")
    t.dump(9 * PAGE, 0x20, "task 2's copy, at VA 9")
    t.write(6 * PAGE + 0x100, seg(abi.ENC_BLIT, fill_rec(ua, 0x20, 0x20, 0x6B)))
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(2, [(6 * PAGE + 0x100, 8 + 0x20)], [(ua, 0)]),
             label="task 2 fills VA 8")
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, b_copy_buf_buf(a, back, 0, 0, 0x40)))
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(a, 0), (back, 0)]), label="task 1 copies VA 8 to VA 10")
    t.dump(back_va, 0x40, "task 1's copy, at VA 10")
    u.delete()
    t.delete()
    s.events("end")
    return s


@scenario
def memory_gap_table_rewrite():
    """The guest rewrites a leaf after the host has used it. Buffer A's
    object keeps the pages it was made over in every setup (objects are
    cached by reference): a fill through A after VA 8's leaf moves from page
    P1 to P2 lands in P1. A new buffer B over VA 8 differs by setup: in the
    blocks setup VA 8 is still mapped in the host task (the range tracker
    says so), so B's fill lands in P1 too; in the map and isolate setups a
    buffer is a new mapping of the table as it is now, and it lands in P2.
    After an unmap of VA 8 (34) a third buffer C lands in P2 in every setup:
    the blocks setup maps the page again. Command buffers are read through
    the table at every exec: after VA 4's leaf moves to page P3, which holds
    another command buffer, the same range runs P3's."""
    s = Scenario("memory_gap_table_rewrite", doc=memory_gap_table_rewrite.__doc__)
    s.bring_up()
    leaves = {v: s.alloc_page() for v in range(12)}
    p1, p2, p3 = leaves[8], s.alloc_page(), s.alloc_page()
    t = XGPUTask(s, tid=1, pages=12, leaves=leaves)
    a, _ = t.buffer(length=PAGE, va=8 * PAGE)
    t.define()
    t.install()

    def run(ref, off, byte, label):
        cb = fill_cb(t, ref, off, 0x10, byte)
        s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(ref, 0)]), label=label)

    def pages(label):
        s.dump(s.pa(p1), 0x50, "P1 " + label)
        s.dump(s.pa(p2), 0x50, "P2 " + label)

    run(a, 0, 0x11, "A, before the rewrite")
    leaf8 = t.table_gpa(0, 0, 8)
    s.u32(leaf8, s.pfn(p2) | abi.PTE_WRITABLE)
    run(a, 0x10, 0x22, "A, after VA 8 moved to P2")
    b = t.ol.add(abi.T_BUFFER, buffer_record(PAGE, 8))
    t.ol.install()
    run(b, 0x20, 0x33, "B, made after the rewrite")
    pages("after A and B")
    t.unmap(8 * PAGE, PAGE)
    c = t.ol.add(abi.T_BUFFER, buffer_record(PAGE, 8))
    t.ol.install()
    run(c, 0x30, 0x44, "C, after an unmap of VA 8")
    pages("after C")
    other = seg(abi.ENC_BLIT, fill_rec(c, 0x40, 0x10, 0x55))
    s.mem(s.pa(p3), other)
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(c, 0x40, 0x10, 0x66)))
    s.u32(t.table_gpa(0, 0, 4), s.pfn(p3) | abi.PTE_WRITABLE)
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(c, 0)]), label="VA 4 moved to P3")
    pages("after the command buffer's page moved")
    t.delete()
    s.events("end")
    return s


@scenario
def memory_gap_huge_task():
    """A task of 32 GiB (0x800000000, the in-process maximum) at depth 3,
    with leaves only for its first eight VA pages (object list, records,
    command buffers) and three far pages: the last page of the first
    second-level table's range (VA page 4095), the first of the next (4096),
    and the task's very last page (2^21 - 1). A buffer over pages 4095 and
    4096 crosses from one last-level table to the next; one over the last
    page ends exactly at the task's end. Fills of both land in every setup;
    in the blocks setup the host task is a 32 GiB reservation and only the
    pages used are mapped into it."""
    s = Scenario("memory_gap_huge_task", doc=memory_gap_huge_task.__doc__)
    s.bring_up()
    last = (0x800000000 // PAGE) - 1
    leaves = {v: s.alloc_page() for v in range(8)}
    leaves.update({4095: s.alloc_page(), 4096: s.alloc_page(), last: s.alloc_page()})
    t = XGPUTask(s, tid=1, pages=8, depth=3, leaves=leaves, length=0x800000000)
    across, _ = t.buffer(length=2 * PAGE, va=4095 * PAGE)
    end, _ = t.buffer(length=PAGE, va=last * PAGE)
    t.define()
    s.dump(s.pa(t.root, 0x14), 8, "the task length")
    t.install()
    cb = fill_cb(t, across, PAGE - 0x10, 0x20, 0x71)
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(across, 0)]), label="across two tables")
    cb = fill_cb(t, end, PAGE - 0x20, 0x20, 0x72)
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(end, 0)]), label="the task's last bytes")
    s.dump(s.pa(leaves[4095], PAGE - 0x20), 0x20, "VA page 4095's end")
    s.dump(s.pa(leaves[4096]), 0x20, "VA page 4096's start")
    s.dump(s.pa(leaves[last], PAGE - 0x30), 0x30, "the last page's end")
    s.events("fills")
    t.delete()
    s.events("end")
    return s
