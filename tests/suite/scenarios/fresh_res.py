"""Resources and tasks: the paths in memory, resources, textures, the
resource manager, the tasks and the service that a guest or the host (the
C API) can reach and no other scenario reaches.
"""
import struct

from pgtest import abi
from pgtest import fresh_res_util as U
from pgtest.abi import PAGE
from pgtest.compute_objects import Compute as C, Rig
from pgtest.covgap_mem_util import HOLE_PAGE, HOLED, XRTask, own_leaves
from pgtest.lifetime_util import either
from pgtest.registry import scenario
from pgtest.resources_build import (RTask, b_copy_buf_buf, b_copy_tex_buf, b_fill_buf, desc1, simple_dim,
                                    texture_rec)
from pgtest.scenario import Scenario
from pgtest.task import seg
from pgtest.tasks_lib import XGPUTask, XTask, exec3_payload, fill_rec

# fresh_res_spin.metal: one thread that polls c[0] (guest memory, through a
# shared buffer) until it reads nonzero or it has polled c[1] times, then
# writes c[2]: 0x600d if c[0] ended it, 0xdead if the cap did. Here c[0]
# stays 0, so the kernel is fixed work: it ends at its cap, and its length
# is the cap's, about 59.5 us a poll on the reference host alone
# (measured: 40,000 polls 2.38 s, 100,000 5.95 s, 220,000 13.09 s), and
# only longer under load (40,000 took 20 s beside 60 other spinning
# processes). The guest cannot end it early: Metal makes a CPU write to a
# shared buffer visible to a running kernel at no defined time, and a
# guest-held kernel that missed its release wedged every full round
# (pgtest/spin_util.py). Every CPU write to c is made before the exec's
# commit. A kernel still running when its process exits makes the GPU
# restart: each cap ends well inside its scenario's watchdog.
SPIN = "shaders/fresh_res_spin.metallib"
AT_CAP = 0xDEAD


def _spin_rig(name, doc, cap, watchdog_s):
    """A task with the spin kernel and its buffer c, warmed up: one exec
    runs the kernel with c[0] already set, so the pipeline is compiled and
    the kernel's path proven before the exec that runs it to its cap.
    Leaves c[0] and c[2] zero again and c[1] = cap (a guest write, made
    before the next exec is committed). Returns (rig, c's VA, the exec's
    segment).

    The scenarios run exclusive (README rule 10): the kernel keeps a GPU
    thread busy for seconds, and nothing else of the suite may slow it or
    be slowed by it."""
    r = Rig(name, doc, watchdog_s=watchdog_s, exclusive=True)
    c, c_va = r.buffer(contents=struct.pack("<IIII", 1, cap, 0, 0))
    pso = r.kernel(SPIN)
    r.ready()
    sg = seg(abi.ENC_COMPUTE, C.set_pipeline(pso) + C.set_buffers(0, [(c, 0)]) +
             C.dispatch_threads((1, 1, 1), (1, 1, 1)))
    r.run(sg, label="warm-up: c[0] already set, the kernel ends at once")
    r.dump(c_va, 12, "c after the warm-up")
    r.t.write(c_va, struct.pack("<III", 0, cap, 0))
    return r, c_va, sg


def _start(r, sg):
    """Starts the spin exec (67) without waiting; returns its stamp."""
    r.t.reset_cmdbufs()
    cb = r.t.cmdbuf(sg)
    st = r.s.fifo.cmd(abi.OP_EXEC_INDIRECT3, exec3_payload(r.t.tid, [cb], r.resources()))
    r.s.fifo.doorbell()
    return st


# InfoIndirect's case: 220,000 polls, 13.1 s alone. The checkpoint while it
# runs is taken at 8 s, 3 s past 69's 5 s timeout and 5 s before the
# kernel can end; the exec's stamp is waited for up to 40 s (3x).
INFO_CAP = 220000
INFO_HELD_MS = 8000


@scenario
def fresh_res_info_while_kernel_held():
    """InfoIndirect (69) behind an exec whose kernel runs for 13 s.

    The exec runs fresh_res_spin as fixed work: 220,000 polls of a word the
    guest leaves 0, 13.1 s on the reference host alone and only longer
    under load, far past the 5 s below. 69 follows on the same FIFO, in
    the same doorbell.

    In process (blocks, map), 69 waits at most 5 s for the FIFO event, which
    only the exec's completion signals, then reports status 3
    (`waitUntilSignaledValue:timeoutMS:5000`): a fault. Nothing is published
    while the kernel runs, not even that fault (the checkpoint at 8 s): the
    fault is published only after the exec's stamp, once the kernel has
    ended (pump: a stamp behind an unsignalled one waits).

    With isolation, the service's exec signals the FIFO event when its
    command buffer is scheduled (`encodeSignalEventScheduled`), not when it
    completes, so the service's 5 s wait is met at once and 69 completes;
    its stamp still follows the exec's. docs/testing.md 7.4 listed this
    timeout as needing a GPU hang: a guest kernel that runs past 5 s is
    enough."""
    name = "fresh_res_info_while_kernel_held"
    r, c_va, sg = _spin_rig(name, fresh_res_info_while_kernel_held.__doc__, INFO_CAP, watchdog_s=60)
    s = r.s
    st_exec = _start(r, sg)
    st_info = s.fifo.cmd(abi.OP_INFO_INDIRECT, exec3_payload(r.t.tid))
    s.fifo.doorbell()
    s.wait_any([dict(gpa=s.fifo.stamp_gpa(), ge=st_exec), dict(gpa=s.fifo.stamp_gpa(), ge=st_info),
                dict(mmio=abi.R_FAULT_MASK, mmio_eq=1)], timeout_ms=INFO_HELD_MS,
               label="8 s into the kernel: the exec's stamp (0), 69's (1), a fault (2), or none")
    s.events("while the kernel runs")
    s.fifo.wait_stamp(st_exec, timeout_ms=40000, label="the exec's stamp, once the kernel ends")
    s.wait_any([dict(gpa=s.fifo.stamp_gpa(), ge=st_info), dict(mmio=abi.R_FAULT_MASK, mmio_eq=1)],
               timeout_ms=10000, label="69: its stamp (0) or a fault (1)")
    s.fault_state("after 69")
    s.resume_root()
    s.fifo.wait_stamp(st_info, label="69's stamp, after the resume")
    r.dump(c_va, 12, "c: the kernel ended at its cap")
    return r.end()


# DeleteTask's case: 110,000 polls, 6.5 s alone; the checkpoint while it
# runs is taken at 2 s, 4.5 s before the kernel can end.
DELETE_CAP = 110000
DELETE_HELD_MS = 2000


@scenario
def fresh_res_delete_while_kernel_held():
    """DeleteTask (32) while the task's exec is still running on the GPU.

    The exec runs fresh_res_spin as fixed work: 110,000 polls, 6.5 s on
    the reference host alone and only longer under load; the delete is
    queued right behind it, in the same doorbell, and the checkpoint is
    taken 2 s in.

    With isolation, releasing the service's task drains the outstanding GPU
    token, waiting for the exec's completion, and with a memory map (map)
    the delete waits behind the exec too: nothing is published while the
    kernel runs, and after it ends the exec's stamp comes, then the
    delete's, and c[2] says the cap ended the kernel. How many interrupts
    publish those two stamps is timing, so the last checkpoint counts them
    collapsed.

    In the blocks setup the delete does not wait for the GPU: its host
    calls (task 0's unmap_memory of the task's root page, then destroy_task)
    are made while the kernel runs, as the checkpoint shows. The kernel
    keeps running over guest RAM, which the host still has, ends at its
    cap, and the stamps come in the same order.
    (Run beside another process's GPU fault, the exec is discarded as its
    victim: its stamp comes early and c[2] stays 0. Hence exclusive.)"""
    name = "fresh_res_delete_while_kernel_held"
    r, c_va, sg = _spin_rig(name, fresh_res_delete_while_kernel_held.__doc__, DELETE_CAP, watchdog_s=30)
    s = r.s
    st_exec = _start(r, sg)
    st_del = s.fifo.cmd(abi.OP_DELETE_TASK, struct.pack("<I", r.t.tid))
    s.fifo.doorbell()
    s.wait_any([dict(gpa=s.fifo.stamp_gpa(), ge=st_exec), dict(gpa=s.fifo.stamp_gpa(), ge=st_del),
                dict(mmio=abi.R_FAULT_MASK, mmio_eq=1)], timeout_ms=DELETE_HELD_MS,
               label="2 s into the kernel: the exec's stamp (0), the delete's (1), a fault (2), or none")
    s.events("while the kernel runs")
    s.fifo.wait_stamp(st_exec, timeout_ms=20000, label="the exec's stamp, once the kernel ends")
    s.fifo.wait_stamp(st_del, label="the delete's stamp")
    s.fault_state("after the delete")
    r.dump(c_va, 12, "c: the kernel ended at its cap")
    # The exec's stamp and the delete's are published by one interrupt or by
    # two: with isolation the drain wakes the FIFO thread (the delete) before
    # the exec's token is signalled, and which thread signals its stamp first
    # is timing (docs/testing.md 5.4).
    s.events("end", collapse=["irq"])
    return s


# ---- the range tracker ----

def _access(s, tid, page, npages, label):
    """Makes the host map [page, page + npages) as an object list:
    SetObjectList (51) over it, then an ExecIndirect3 (67) with no command
    buffers, which maps the object list first (the service's
    GpuTask::ensure_object_list_mapped). The list's length is the range's less
    4 bytes, as the fuzzer's access was. Either may fault: the pages hold no
    valid list."""
    either(s, abi.OP_SET_OBJECT_LIST, struct.pack("<III", tid, page, npages * PAGE - 4), label + ": the list")
    either(s, abi.OP_EXEC_INDIRECT3, exec3_payload(tid), label + ": an exec maps it")


@scenario
def fresh_res_tracker_test_block_end():
    """RangeTracker::test_range across two run blocks. 240 one-page MapMemory2
    commands (57) fill the tracker with 240 runs; a map of page 5 then splits
    them into two blocks of 120 runs (pages 0-119, 120-239). An access of page
    119, the last run of block 0 (an object list there, mapped by an exec),
    makes test_range empty the range in block 0 and meet block 1
    GreaterAdjacent: nothing is mapped again. Every VA page aliases one RAM
    page.

    The tracker is the in-process task's (blocks). With a memory map (map)
    and in the service (isolate), MapMemory2 is handled otherwise, and that
    difference is locked too. The 240 maps' block calls are reported as a
    digest; what the split and the access call is listed."""
    s = Scenario("fresh_res_tracker_test_block_end", watchdog_s=60, doc=fresh_res_tracker_test_block_end.__doc__)
    s.bring_up()
    alias = s.alloc_page()
    t = XTask(s, tid=1, pages=1024, alias=alias)
    t.define()
    for p in range(240):
        t.map(p * PAGE, PAGE, fault="either", label="map page %d" % p)
    s.events("240 one-page maps", blocks_digest=True)
    t.map(5 * PAGE, PAGE, fault="either", label="page 5 again: the split")
    s.events("the split")
    _access(s, 1, 119, 1, "page 119")
    s.events("the access of page 119")
    t.delete()
    s.events("end")
    return s


@scenario
def fresh_res_tracker_contains():
    """RangeTracker::add_to_range where a later run block's span contains the
    new run. The input is a 962-command guest sequence a fuzzer of the tracker
    found (pgtest/fresh_res_util.py says how): 959 one-page MapMemory2 commands
    (57) over a 2,796-page task, an access of pages 1246-1247 (an object list,
    mapped by an exec), then MapMemory2 of 403 pages at 1323 and of 2 at 1722.
    Every VA page aliases one RAM page. The tracker is the in-process task's
    (blocks); map and isolate lock what the same commands do there. Block calls
    are reported as a digest before the access and listed after it."""
    s = Scenario("fresh_res_tracker_contains", watchdog_s=120, doc=fresh_res_tracker_contains.__doc__)
    s.bring_up()
    alias = s.alloc_page()
    t = XTask(s, tid=1, pages=U.TRACKER_CONTAINS_TASK_PAGES, alias=alias)
    t.define()
    for p in U.TRACKER_CONTAINS_PAGES:
        t.map(p * PAGE, PAGE, fault="either", label="map page %d" % p)
    s.events("959 one-page maps", blocks_digest=True)
    a, n = U.TRACKER_CONTAINS_ACCESS
    _access(s, 1, a, n, "pages %d-%d" % (a, a + n - 1))
    s.events("the access")
    for a, n in U.TRACKER_CONTAINS_TAIL:
        t.map(a * PAGE, n * PAGE, fault="either", label="map %d pages at %d" % (n, a))
    s.events("the last two maps", blocks_digest=True)
    t.delete()
    s.events("end")
    return s


# ---- paging textures and buffers ----

@scenario
def fresh_res_texture_over_hole():
    """A private texture (type 2) whose one page is a leaf no RAM range holds
    (RAM in two ranges with a hole between), listed in an exec that copies it
    to a buffer: the texture cannot page in, and the exec faults. In the
    service the texture's transfer buffer comes from
    GpuTask::new_buffer_for_virtual_page, whose GpuTask::new_virtual_mapping
    gets no mapping of the page from the memory map, and it raises; in process
    the mapping fails through the host's map_memory (blocks) or the memory map
    (map). The heap twin is covgap_mem_heap_over_hole. The next exec, a fill
    over RAM, runs."""
    s = Scenario("fresh_res_texture_over_hole", ram=HOLED, doc=fresh_res_texture_over_hole.__doc__)
    s.bring_up()
    leaves = own_leaves(s, 10)
    leaves[10] = HOLE_PAGE
    t = XRTask(s, tid=1, pages=11, leaves=leaves)
    tex = t.add(abi.T_TEXTURE, texture_rec(PAGE, 10, [simple_dim(4, 4)], desc1()))
    b, b_va = t.shared(PAGE, fill=0x00)
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(tex, b, 0, 16, 64, (4, 4, 1)), resources=[(tex, 0), (b, 0)], fault=True,
           label="a texture over the hole")
    t.blit(b_fill_buf(b, 0, 0x10, 0x6B), resources=[(b, 0)], label="after it, a fill over RAM")
    t.dump(b_va, 0x20, "B")
    t.delete()
    s.events("end")
    return s


@scenario
def fresh_res_read_only_pairs():
    """Resources over two read-only pages, whose page-table walk meets a second
    read-only page after the first has already cleared `writable` (bit 31 of a
    leaf). A shared buffer (type 1) over VA pages 8-9 is read across its page
    boundary by a copy into a writable buffer: in the service a detach buffer,
    whose address ranges are walked by
    GpuTask::address_ranges_for_virtual_page. A private texture (type 2) of
    64 x 128 RGBA8 over pages 10-11 is copied out, texel rows 0 and 127: in the
    service its transfer buffer is a mapping of both pages. In process the
    blocks setup reports each mapping's read-only flag."""
    s = Scenario("fresh_res_read_only_pairs", doc=fresh_res_read_only_pairs.__doc__)
    s.bring_up()
    t = XRTask(s, tid=1, pages=14, writable=set(range(14)) - {8, 9, 10, 11})
    ro, _ = t.buffer(length=2 * PAGE, va=8 * PAGE)
    t.write(8 * PAGE, bytes(range(0x100)) * 64)
    t.write(9 * PAGE, bytes(0xFF - (i & 0xFF) for i in range(0x100)) * 64)
    tex = t.add(abi.T_TEXTURE, texture_rec(2 * PAGE, 10, [simple_dim(64, 128)], desc1(w=64, h=128)))
    t.write(10 * PAGE, bytes((i * 7) & 0xFF for i in range(PAGE)))
    t.write(11 * PAGE, bytes((i * 13 + 1) & 0xFF for i in range(PAGE)))
    b, b_va = t.buffer(length=PAGE, va=12 * PAGE)
    t.fill(b_va, PAGE, 0)
    t.define()
    t.install()
    either(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1, [_blit_cmd(t, b_copy_buf_buf(ro, b, PAGE - 0x10, 0, 0x20))],
                                                   [(ro, 0), (b, 0)]),
           "copy across the read-only buffer's two pages")
    either(s, abi.OP_EXEC_INDIRECT3,
           exec3_payload(1, [_blit_cmd(t, b_copy_tex_buf(tex, b, 0x40, 16, 16, (4, 1, 1)) +
                                         b_copy_tex_buf(tex, b, 0x50, 16, 16, (4, 1, 1), origin=(0, 127, 0)))],
                         [(tex, 0), (b, 0)]),
           "copy rows 0 and 127 of the read-only texture")
    t.dump(b_va, 0x60, "B")
    t.dump(8 * PAGE, 0x10, "the read-only buffer's first bytes")
    t.delete()
    s.events("end")
    return s


def _blit_cmd(t, records):
    t.reset_cmdbufs()
    return t.cmdbuf(seg(abi.ENC_BLIT, records))


# ---- placement lists at rebuild (isolate) ----

@scenario(setups=("isolate",))
def fresh_res_reserve_odd_entries():
    """TaskCore::rebuild_task's ResourceManager::reserve_objects over entry
    kinds no other scenario puts in a placement task: a type 3 (mipmapped
    texture) and a type 9 (serializer texture) entry, with zero placements,
    reach their case labels and reserve nothing; a type 7 entry whose record is
    4 bytes, with a nonzero placement, goes to ResourceManager::reserve_object,
    whose header read fails, and reserves nothing either. None of the records
    is ever made: an exec with no command buffers maps the lists (and so runs
    reserve_objects), and completes, and a fill of a buffer B runs. The
    placement entries of the three are unchanged. Isolate only: in process no
    task has argument buffers, so a placement list is never read."""
    from scenarios.textures_placement import install_with_placements
    s = Scenario("fresh_res_reserve_odd_entries", setups=("isolate",), doc=fresh_res_reserve_odd_entries.__doc__)
    s.bring_up()
    t = RTask(s, pages=16)
    rec_va = t.area(PAGE, fill=0x00)
    m3 = t.ol.add_raw(abi.T_TEXTURE_MIPGEN, 0x80, rec_va)
    m9 = t.ol.add_raw(abi.T_SERIALIZER_TEXTURE9, 0x80, rec_va + 0x100)
    short = t.ol.add_raw(abi.T_SERIALIZED, 4, rec_va + 0x200)
    b, b_va = t.shared(PAGE, fill=0x00)
    t.define()
    pl = install_with_placements(t, {short: (0x40, 0x40)})
    either(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1), "an exec maps the lists")
    t.blit(b_fill_buf(b, 0, 0x10, 0x3C), resources=[(b, 0)], label="a fill")
    t.dump(b_va, 0x10, "B")
    t.dump(pl, 16 * short + 16, "placement list, up to the type 7 entry (B's holds a host GPU address)")
    t.delete()
    s.events("end")
    return s


# ---- the page-table walk ----

@scenario(setups=("isolate",))
def fresh_res_walk_climb_two_levels():
    """A depth-3 task of 2^39 bytes with leaves for VA pages 0-7 and for
    P - 1 and P, P = 4096 * 4096 = 2^24: the last page under root entry 0
    and the first under root entry 1. A buffer over P - 1 and P is filled
    across the boundary. After page P - 1 the walk leaves its leaf table,
    climbs to level 1, finds that level's next index past its table too
    (4096 from its base), and climbs again to the root in one step, then
    descends under root entry 1: both pages are filled.
    mutgap4_svc_walk_second_root_entry climbs one level. Isolate only:
    in-process tasks stop at 32 GiB."""
    s = Scenario("fresh_res_walk_climb_two_levels", setups=("isolate",),
                 doc=fresh_res_walk_climb_two_levels.__doc__)
    s.bring_up()
    p = 4096 * 4096
    leaves = {v: s.alloc_page() for v in range(8)}
    leaves.update({p - 1: s.alloc_page(), p: s.alloc_page()})
    t = XGPUTask(s, tid=1, pages=8, depth=3, leaves=leaves, length=1 << 39)
    across, _ = t.buffer(length=2 * PAGE, va=(p - 1) * PAGE)
    t.define()
    t.install()
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(across, PAGE - 0x10, 0x20, 0x72)))
    either(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(across, 0)]), "a fill across two root entries")
    s.dump(s.pa(leaves[p - 1], PAGE - 0x20), 0x20, "VA page P - 1's end")
    s.dump(s.pa(leaves[p]), 0x20, "VA page P's start")
    t.delete()
    s.events("end")
    return s


# ---- objects whose records lie past the task's end ----

@scenario
def fresh_res_objects_outside_task():
    """Compute objects whose object-list entry names a record past the task's
    end: a compute pipeline, a sampler and a fence. No cursor can be made over
    the record (in each type's ResourceManager::new_object), so the command
    that names it fails and the exec faults. resources_record_outside_task does
    this for buffers, textures and heaps. A last exec dispatches the good
    kernel. A function past the end is a death instead (Metal's
    "computeFunction must not be nil", compute_death_function_*)."""
    s_name = "fresh_res_objects_outside_task"
    r = Rig(s_name, fresh_res_objects_outside_task.__doc__, pages=16)
    end = 16 * PAGE
    out, out_va = r.buffer()
    bad_pso = r.raw(abi.T_SERIALIZED, b"")
    bad_sampler = r.raw(abi.T_SERIALIZED, b"")
    bad_fence = r.raw(abi.T_SERIALIZED, b"")
    good = r.kernel()
    for ref, n in ((bad_pso, 0x68), (bad_sampler, 0x40), (bad_fence, 0x10)):
        typ, _, _ = r.t.ol.entries[ref]
        r.t.ol.entries[ref] = (typ, n, end + ref * PAGE)
    r.ready()
    r.compute(C.set_pipeline(bad_pso), fault=True, label="a pipeline past the end")
    r.compute(C.set_pipeline(good) + C.set_samplers(0, [bad_sampler]), fault=True, label="a sampler past the end")
    r.compute(C.set_pipeline(good) + C.fence(0xD4, bad_fence), fault=True, label="a fence past the end")
    r.compute(C.set_pipeline(good) + C.set_buffers(0, [(out, 0)]) + C.dispatch_threads((4, 1, 1), (4, 1, 1)),
              label="the good kernel")
    r.dump(out_va, 0x10, "out")
    return r.end()
