"""Task paths the other task modules leave out: the blocks setup's range
tracker at its edges, and the kernel task's object list failing under another
task's lookups (docs/testing.md, sections 6.10, 6.20)."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.resources_build import (BGRA8Unorm, OPT_SHARED, RTask, b_copy_tex_buf, b_fill_buf,
                                    b_fill_tex_color, backing_rec, cmd_delete_shared_backing, desc1,
                                    iosurface_op, ref_texture_rec, shared_backing_rec,
                                    shared_texture_rec, simple_dim)
from pgtest.scenario import Scenario
from pgtest.task import seg
from pgtest.tasks_lib import XGPUTask, XTask, exec3_payload, fill_rec, probe_list, set_list_raw

FULL = (4, 4, 1)
BGRA = 0x42475241  # 'BGRA', an IOSurface pixel format


def steps(s, t, plan):
    """Maps, unmaps and lazy probes (probe_list: an object list over the
    pages, mapped by an exec), each followed by the events it caused: the
    host's map_memory calls show what the tracker thought unmapped."""
    for kind, p, n in plan:
        label = "%s [%d, %d)" % (kind, p, p + n)
        if kind == "map":
            t.map(p * PAGE, n * PAGE)
        elif kind == "unmap":
            t.unmap(p * PAGE, n * PAGE)
        else:
            probe_list(s, t.tid, p, n, label)
        s.events(label)


@scenario(setups=("blocks",))
def tasks_gap_tracker_edges():
    """The range tracker's edge cases, each probed by what the host is asked
    to map next. A probe that starts inside a run and runs past it (a partial
    overlap, which the scan subtracts) maps only the rest; one that ends
    inside a later run maps only its start; a map of a run's first pages
    again is merged into it. Blocks only: the tracker is the blocks setup's
    (tasks_map_tracker)."""
    s = Scenario("tasks_gap_tracker_edges", setups=("blocks",), doc=tasks_gap_tracker_edges.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=1100, alias=s.alloc_page())
    t.define()
    steps(s, t, [
        ("map", 4, 4),
        ("probe", 6, 4),
        ("map", 20, 4),
        ("probe", 18, 4),
        ("map", 30, 4),
        ("map", 30, 2),
        ("probe", 28, 8),
    ])
    t.delete()
    s.events("end")
    return s


def two_blocks(s):
    """A task of 1100 pages, every leaf one guest page, with 240 single-page
    runs at the even pages from 100 (a full run block: a block holds 240
    runs) and a run at page 700, which starts a second block."""
    t = XTask(s, tid=1, pages=1100, alias=s.alloc_page())
    t.define()
    for p in range(100, 580, 2):
        t.map(p * PAGE, PAGE)
    s.events("240 runs from page 100")
    steps(s, t, [("map", 700, 1)])
    return t


@scenario(setups=("blocks",))
def tasks_gap_tracker_blocks():
    """The range tracker across run blocks, at their edges (blocks only, for
    the reason tasks_map_tracker gives). A full block of 240 runs and a
    second block at page 700; a run at 650 finds the first block before it
    and the second after it and splits the full first block. A probe [690,
    700), ending exactly where the second block starts, is unmapped (the walk
    of unmapped ranges stops at the adjacent block) and maps all of itself,
    which makes the first blocks' last run end where the second block starts.
    An unmap [0, 100), ending where the first block starts, touches nothing.
    A map of [700, 701) again joins the earlier block, and the second block,
    now equal to it, is removed; a probe over it maps nothing more, and one
    in the first block maps its gaps."""
    s = Scenario("tasks_gap_tracker_blocks", setups=("blocks",), doc=tasks_gap_tracker_blocks.__doc__)
    s.bring_up()
    t = two_blocks(s)
    steps(s, t, [
        ("map", 650, 1),
        ("probe", 690, 10),
        ("unmap", 0, 100),
        ("map", 700, 1),
        ("probe", 690, 11),
        ("probe", 100, 4),
    ])
    t.delete()
    s.events("end")
    return s


@scenario(setups=("blocks",))
def tasks_gap_tracker_empty_block():
    """A run block the tracker empties and keeps (blocks only, for the reason
    tasks_map_tracker gives). With a full block, a split, and a second block
    holding only [700, 701) (as in tasks_gap_tracker_blocks), a map of [640,
    701) is added to the earlier block, and the later block, inside the new
    run at its end, has that run removed (the add scans the later blocks): it
    is left with no runs and an empty span, still in the list. An empty block
    compares as after everything: a probe past it maps, and the runs it adds
    go to the block before; a probe inside the new run maps nothing; one over
    a run added before every block maps only its gaps. An unmap of the whole
    task removes only the first block, as always (docs/testing.md, section
    7.1): a probe in that range maps it again, while one over [690, 710) maps
    only [701, 710), the tracker still holding the host-unmapped [640,
    701)."""
    s = Scenario("tasks_gap_tracker_empty_block", setups=("blocks",), doc=tasks_gap_tracker_empty_block.__doc__)
    s.bring_up()
    t = two_blocks(s)
    steps(s, t, [
        ("map", 650, 1),
        ("map", 640, 61),
        ("probe", 800, 4),
        ("probe", 690, 11),
        ("map", 10, 2),
        ("probe", 8, 6),
        ("unmap", 0, 1100),
        ("probe", 100, 6),
        ("probe", 690, 20),
    ])
    t.delete()
    s.events("end")
    return s


def kernel_and_user(s, kholes=()):
    """Task 0 (the kernel task) with pages 30 and 31 left as holes when
    asked, and task 1; each with its object list at VA 0."""
    return RTask(s, tid=0, pages=32, holes=kholes), RTask(s, tid=1)


@scenario
def tasks_gap_kernel_lists():
    """The kernel task's side of shared textures when its objects are gone or
    its object list cannot be mapped. A shared texture (type 16) made from a
    shared texture backing (type 17) in task 0 works; after opcode 65 deletes
    the backing, a fill of that texture with invalidateGuest still completes:
    the exec's end tells the kernel task to mark the backing's guest copy
    invalid, and the kernel task only logs that it has no such backing (from
    the service, through the host). Then task 0's object list is moved onto a
    hole in its page table (opcode 51 only records it): a second shared
    texture, whose handle is asked of the kernel task, cannot be made
    ("Failed to map object list"; the host turns that into a nil handle for
    the service), and its exec faults. With the list moved back, the same
    texture works."""
    s = Scenario("tasks_gap_kernel_lists", doc=tasks_gap_kernel_lists.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s, kholes=(30, 31))
    px = bytes((0x30 + i) & 0xFF for i in range(64))
    bk_va = k.area(PAGE, contents=px)
    gone = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, bk_va // PAGE, [simple_dim(4, 4)],
                                                           desc1(options=0x20, usage=3)))
    bk2_va = k.area(PAGE, contents=px[::-1])
    kept = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, bk2_va // PAGE, [simple_dim(4, 4)],
                                                           desc1(options=0x20, usage=3)))
    s_gone = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(gone))
    s_late = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kept))
    b, b_va = t.shared(PAGE)
    k.define()
    k.install()
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(s_gone, b, 0, 16, 64, FULL), resources=[(s_gone, 0), (b, 0)],
           label="a shared texture over backing 1")
    t.dump(b_va, 0x40, "B: backing 1's pixels")
    cmd_delete_shared_backing(s, gone, label="65 deletes backing 1")
    t.blit(b_fill_tex_color(s_gone, (2, 2, 1), (1.0, 0, 0, 1.0), 70), resources=[(s_gone, 2), (b, 0)],
           label="invalidateGuest of a texture whose backing is gone")
    s.events("after the invalidation")
    set_list_raw(s, 0, 30, k.ol.length())
    t.blit(b_copy_tex_buf(s_late, b, 0x40, 16, 64, FULL), resources=[(s_late, 0), (b, 0)], fault=True,
           label="a shared texture while task 0's list is over a hole")
    set_list_raw(s, 0, 0, k.ol.length())
    t.blit(b_copy_tex_buf(s_late, b, 0x80, 16, 64, FULL), resources=[(s_late, 0), (b, 0)],
           label="the same texture with task 0's list back")
    t.dump(b_va + 0x40, 0x80, "B+0x40: nothing, then backing 2's pixels")
    t.delete()
    k.delete()
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def tasks_gap_kernel_list_backing():
    """A texture over an IOSurface backing (type 5 over type 4) finds its
    backing through the kernel task's object list. With task 0's list moved
    onto a hole in its page table the list cannot be mapped ("Failed to map
    object list") and the exec faults; moved back, the texture reads the
    backing. In-process only: in the service the backing lookup aborts
    (textures_backing_ref_service)."""
    s = Scenario("tasks_gap_kernel_list_backing", setups=("blocks", "map"),
                 doc=tasks_gap_kernel_list_backing.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s, kholes=(30, 31))
    px = bytes((i * 5) & 0xFF for i in range(64))
    bk_va = k.area(PAGE, contents=bytes(0x100) + px)
    bk = k.add(abi.T_IOSURFACE_BACKING, backing_rec(PAGE, bk_va // PAGE, BGRA, [(0x100, 4, 4, 16, 4)]))
    r = t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=iosurface_op(12, desc1(fmt=BGRA8Unorm,
                                                                                     options=OPT_SHARED))))
    b, b_va = t.shared(PAGE)
    k.define()
    k.install()
    t.define()
    t.install()
    set_list_raw(s, 0, 30, k.ol.length())
    t.blit(b_copy_tex_buf(r, b, 0, 16, 64, FULL), resources=[(r, 0), (b, 0)], fault=True,
           label="a type 5 texture while task 0's list is over a hole")
    set_list_raw(s, 0, 0, k.ol.length())
    t.blit(b_copy_tex_buf(r, b, 0x40, 16, 64, FULL), resources=[(r, 0), (b, 0)],
           label="with task 0's list back")
    t.dump(b_va, 0x80, "B: nothing, then the backing's pixels")
    t.delete()
    k.delete()
    s.events("end")
    return s


@scenario
def tasks_gap_cmdbuf_huge_length():
    """A command buffer of length 2^62 at task VA 0x10000. In-process the
    cursor is a copy, and the copy's buffer is allocated before any range
    check: malloc(2^62) fails ("Failed malloc"), the cursor is null, and the
    segment decoder dereferences it (docs/testing.md, sections 6.12, 7.4):
    the host dies. The service refuses the range ("Out of range cursor
    request") and dereferences its null cursor too: the service dies, the
    exec faults, and so does the next command on the task. A fill before
    shows the task working."""
    s = Scenario("tasks_gap_cmdbuf_huge_length", doc=tasks_gap_cmdbuf_huge_length.__doc__)
    s.bring_up()
    t = XGPUTask(s, tid=1, pages=16)
    buf, buf_va = t.buffer(length=PAGE, va=8 * PAGE)
    t.define()
    t.install()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(buf, 0, 8, 0x11)))
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(buf, 0)]), label="before")
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [(4 * PAGE, 1 << 62)], [(buf, 0)]), fault=True,
             label="a command buffer of 2^62 bytes")
    s.fault_state("a command buffer of 2^62 bytes")
    s.resume_root()
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(buf, 8, 8, 0x22)))
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(buf, 0)]), fault=True, label="after")
    s.fault_state("after")
    t.dump(buf_va, 0x10, "the buffer")
    s.events("end")
    return s
