"""Whole guest sessions (docs/testing.md, sections 1, 4 and 6): every
subsystem the way a macOS guest driver uses it across its lifetime, in one
scenario, so that a refactor that breaks how two subsystems interact shows
even where each one alone still passes.

A session negotiates the version and reads device info, boots on the EFI
framebuffer and hands the display over to the driver, connects display
pages, defines the kernel task and several GPU tasks with buffers, a heap,
textures, functions from the committed metallibs and pipelines, runs blit,
compute and render work on the root FIFO and child FIFOs whose results land
in guest memory, presents frames and encodes them, moves the cursor, drives
the IOSurface host device, suspends and resumes in the middle, and tears
everything down. The steps are pgtest/session.py's.

Where two paths must produce the same bytes (the same work on two tasks, or
before and after a resume), the session compares them in guest memory
(cmp). Frames the host encodes come from the library's shaders, and
device info reports the host, so most sessions are keyed by the host.
"""
import struct

from pgtest import abi
from pgtest import render_lib as R
from pgtest.abi import PAGE
from pgtest.exec_objects import pipeline_record, render_blob
from pgtest.iosfc_guest import CMD_TEST, CMD_UNMAP
from pgtest.registry import scenario
from pgtest.resources_build import BGRA8Unorm, OPT_SHARED, b_fill_buf, desc1, iosurface_op, mapper_ref_rec
from pgtest.session import COMPUTE_OFF, FH, FW, PHASE, RENDER_OFF, RESULT, Session
from pgtest.task import seg


def _render_slots_equal(sess, a, pa, b, pb, label):
    sess.s.cmp(a.result_gpa(pa) + RENDER_OFF, b.result_gpa(pb) + RENDER_OFF, 0x100, label=label)


def _resume_display(sess, cursor_ports=(0,)):
    """After rvg_device_did_resume the restored guest displays replay each
    port's cursor glyph and show; a frame last presented with opcode 7 is not
    presented again."""
    for _ in cursor_ports:
        sess.expect("cursorGlyph", label="glyph replayed")
        sess.expect("cursorShow", label="show replayed")
    sess.s.events("resumed")


@scenario(host=True)
def interplay_session_workstation():
    """One head, two apps. Version negotiation (0x100 answered with 0xf5,
    a request with bit 31 ignored), EFI's frame presented and encoded,
    device info, then the driver: task 0 with two backings, EFI off before
    the display page connects (the handover), a frame and the cursor. Tasks
    1 and 2 run blit, compute (add_one) and render work on child FIFOs 1
    and 2, task 2's exec waiting on task 1's stamp (a barrier); both render the same triangle, so their render
    results are the same bytes. Suspend and resume on a fresh device in the
    middle: the resumed device replays the cursor, runs the same work with
    the tasks swapped between the child FIFOs, and every result equals its
    pre-suspend twin. A frame after the resume, then orderly teardown and
    release."""
    x = Session("interplay_session_workstation", interplay_session_workstation.__doc__, efi=True)
    s = x.s
    x.negotiate()
    x.efi_boot()
    x.root()
    x.device_info()
    x.kernel_task()
    x.efi_off()
    x.connect(0)
    x.page(0, "connected after EFI")
    x.present(0, 0)
    x.cursor(0, 10, 4)
    x.page(0, "cursor")
    t1 = x.task(1, 0x10)
    t2 = x.task(2, 0x80)
    x.child(1)
    x.child(2)
    s.events("objects defined")
    x.batch([(1, t1, t1.blit_segments(0, 0x3C) + t1.compute_segments(0) + t1.render_segments(0), ()),
             (2, t2, t2.blit_segments(0, 0x3C) + t2.compute_segments(0) + t2.render_segments(0), ())],
            "two apps", serial=True)
    x.results(t1, 0)
    x.results(t2, 0)
    _render_slots_equal(x, t1, 0, t2, 0, "the same render on two tasks")
    x.present(0, 1)
    x.cursor(0, 12, 6, glyph=False)
    s.events("before suspend")
    x.suspend()
    x.resume()
    _resume_display(x)
    s.sweep(0x1000, 0x1040, label="registers after resume")
    x.page(0, "resumed")
    x.batch([(2, t1, t1.blit_segments(1, 0x3C) + t1.compute_segments(1) + t1.render_segments(1), ()),
             (1, t2, t2.blit_segments(1, 0x3C) + t2.compute_segments(1) + t2.render_segments(1), ())],
            "two apps after resume, FIFOs swapped", serial=True)
    x.same(t1, 0, t1, 1, "task 1: before and after the resume")
    x.same(t2, 0, t2, 1, "task 2: before and after the resume")
    x.present(0, 0)
    x.cursor(0, 1, 1, glyph=False)
    x.page(0, "after the resume's frame")
    x.teardown()
    return s


@scenario(host=True)
def interplay_session_efi_live_handover():
    """The driver connects port 0's page while EFI still owns the display:
    the guest display takes the display that EFI had bound, then EFI keeps
    rebuilding on its own register writes, and the driver turns EFI off only
    after its first frame: EFI's rebuilds still reach the display (new_frame,
    a guest present each) though the guest display holds it. The connect
    itself presents nothing. Tasks run on the root FIFO between the two;
    teardown at the end."""
    x = Session("interplay_session_efi_live_handover", interplay_session_efi_live_handover.__doc__, efi=True)
    s = x.s
    x.negotiate(asks=())
    x.efi_boot(seed=0x44)
    s.events("EFI booted")
    x.root()
    x.kernel_task()
    x.connect(0)
    s.quiesce(100)
    s.events("connected while EFI is live")
    x.page(0, "connected while EFI is live")
    x.efi.write_frame(FW, FH, 64, seed=0x66)
    s.w(abi.R_EFI_STRIDE, 128)
    x.expect("newFrame", label="EFI rebuilt at stride 128")
    s.w(abi.R_EFI_STRIDE, 64)
    x.expect("newFrame", label="EFI rebuilt at stride 64")
    s.events("EFI rebuilt twice under the nub")
    s.api("present_counts", display=0)
    t = x.task(1, 0x21)
    x.phase(t, 0)
    x.results(t, 0)
    x.present(0, 1)
    x.efi_off()
    s.quiesce(100)
    s.events("EFI off")
    x.present(0, 0)
    x.phase(t, 1, byte=0x5A)
    x.same(t, 0, t, 1, "the same work twice")
    x.teardown()
    return s


@scenario
def interplay_session_dependencies():
    """Work across FIFOs ordered only by barriers: child 1's blit fills a
    buffer, child 2's compute reads it (a barrier on child 1's stamp), and
    the root FIFO's render and copy wait for child 2 (a barrier on its
    stamp). The root FIFO is rung first and child 1 last, so each waits on
    the next. Then the chain the other way round, two tasks sharing it, and
    all four FIFOs chained the same way. At the end nothing has faulted: the fault state and each
    node. No display: nothing here reports the host."""
    x = Session("interplay_session_dependencies", interplay_session_dependencies.__doc__, host=False, ports=1,
                displays=())
    s = x.s
    x.negotiate(asks=(0x7FFFFFFF,))
    x.root()
    t1 = x.task(1, 0x31)
    t2 = x.task(2, 0x62)
    c1 = x.child(1)
    c2 = x.child(2)
    c3 = x.child(3, pages=2)
    for phase, (a, b, root_task) in enumerate(((c1, c2, t1), (c3, c1, t2))):
        scratch = phase * PHASE + RESULT
        feed = t1.cmdbuf(seg(abi.ENC_BLIT, b_fill_buf(t1.out, scratch, 0x100, 0x07 + phase)))
        p_feed = struct.pack("<III", t1.tid, len(t1.resources()), 1) + b"".join(
            struct.pack("<II", r, f) for r, f in t1.resources()) + struct.pack("<QQ", *feed)
        sa = a.cmd(abi.OP_EXEC_INDIRECT3, p_feed)
        sb = b.cmd(abi.OP_EXEC_INDIRECT3, t1.payload(t1.compute_segments(phase, src=t1.out, src_off=scratch)),
                   barriers=[(a.stamp_index, sa)])
        sr = s.fifo.cmd(abi.OP_EXEC_INDIRECT3, root_task.payload(root_task.render_segments(phase)),
                        barriers=[(b.stamp_index, sb)])
        s.doorbell()
        b.doorbell()
        a.doorbell()
        s.fifo.wait_stamp(sr, label="chain %d: the root's render" % phase)
        b.wait_stamp(sb, label="chain %d: the compute" % phase)
        a.wait_stamp(sa, label="chain %d: the fill" % phase)
        t1.reset_cmdbufs()
        t2.reset_cmdbufs()
        s.dump(t1.result_gpa(phase) + COMPUTE_OFF, 0x100, "chain %d: add_one over the fill" % phase)
        s.dump(root_task.result_gpa(phase) + RENDER_OFF, 0x100, "chain %d: the render" % phase)
    s.cmp(t1.result_gpa(0) + RENDER_OFF, t2.result_gpa(1) + RENDER_OFF, 0x100, label="the two renders")
    # Every FIFO, each exec waiting on the one before.
    x.batch([(0, t1, t1.blit_segments(2, 1) + t1.compute_segments(2), ()),
             (1, t2, t2.blit_segments(2, 2) + t2.compute_segments(2), ()),
             (2, t1, t1.render_segments(2), ()),
             (3, t2, t2.render_segments(2), ())], "four FIFOs, chained", serial=True)
    x.results(t1, 2)
    x.results(t2, 2)
    s.dump(s.fifo.base_gpa(), 16, "stamp slots 0-3")
    for fid in (1, 2, 3):
        s.dump(x.children[fid].node_gpa(), abi.CHILD_NODE_SIZE, "node %d" % fid)
    s.fault_state("end")
    x.teardown()
    return s


def _mapper_objects(t):
    """Mapper textures (type 11) over mappings 1 and 2, 16 x 8 BGRA,
    shared, usable as render targets, and a render pipeline whose
    colour attachment is BGRA8Unorm (80)."""
    d = desc1(fmt=BGRA8Unorm, w=FW, h=FH, usage=1 | 2 | 4, options=OPT_SHARED)
    t.m1 = t.ol.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(1) + iosurface_op(12, d))
    t.m2 = t.ol.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(2) + iosurface_op(12, d))
    t.bgra = t.ol.add(abi.T_SERIALIZED, pipeline_record(14, render_blob(t.f_vert, t.f_frag, BGRA8Unorm)))
    t.extra += [t.m1, t.m2]


def _compose(x, t, tex, clear, draw, label, fid=0):
    """The guest's compositor: a pass over a mapped surface, cleared, and
    the triangle when draw is set. The pixels land in the surface's guest
    page at once."""
    recs = [R.describe_pass({0: dict(tex=tex, clear=clear)}, width=FW, height=FH)]
    if draw:
        recs += [R.pipeline_state(t.bgra), R.draw16(3, 0, 3)]
    x.run(fid, t, [seg(abi.ENC_RENDER, b"".join(recs))], label)


@scenario(setups=("map", "isolate"), host=True)
def interplay_session_iosurface():
    """A compositor over the IOSurface host device (map and isolate only:
    QEMU creates that device only with a memory map, apple-gfx-mmio.c). Two
    16 x 8 BGRA surfaces are mapped; task 1 renders frames into them through
    mapper textures, on the root FIFO and on child 1, and the driver
    presents each with DisplaySwapMapping (8), double-buffered; the host
    encodes every frame. The surface's test command writes its pattern
    between frames. Suspend and resume in the middle: the restore presents
    the last mapped surface again (GuestDisplay::finish_restore, a
    new_frame), and the resumed device renders the same frame into the same
    surface. Teardown deletes the mapper textures before unmapping their
    surfaces."""
    x = Session("interplay_session_iosurface", interplay_session_iosurface.__doc__, setups=("map", "isolate"),
                iosfc=True)
    s = x.s
    x.negotiate()
    x.root()
    x.device_info()
    x.kernel_task()
    x.connect(0)
    x.iosfc_start()
    x.surface(1, 3)
    x.surface(2, 5)
    t = x.task(1, 0x11, extra=_mapper_objects)
    x.child(1)
    x.cursor(0, 3, 3)
    _compose(x, t, t.m1, (0.0, 0.5, 0.0, 1.0), False, "frame 1: clear into surface 1")
    s.dump(s.pa(x.surfaces[1]), FW * FH * 4, "surface 1")
    x.swap(0, 1)
    _compose(x, t, t.m2, (0.0, 0.0, 0.0, 1.0), True, "frame 2: the triangle into surface 2", fid=1)
    s.dump(s.pa(x.surfaces[2]), FW * FH * 4, "surface 2")
    x.swap(0, 2)
    x.iosfc_cmd(CMD_TEST, 1, "test surface 1")
    s.dump(s.pa(x.surfaces[1]), FW * FH * 4, "surface 1 after test")
    x.swap(0, 1)
    x.phase(t, 0, fid=1)
    x.results(t, 0)
    s.events("before suspend")
    x.suspend()
    x.resume()
    x.expect("newFrame", label="the restore presents mapping 1")
    _resume_display(x)
    x.encode(0, "restored frame")
    _compose(x, t, t.m2, (0.0, 0.0, 0.0, 1.0), True, "frame 2 again after the resume", fid=1)
    s.dump(s.pa(x.surfaces[2]), FW * FH * 4, "surface 2 after the resume")
    x.swap(0, 2)
    x.phase(t, 1, fid=0)
    x.same(t, 0, t, 1, "task 1 before and after the resume")
    x.teardown(release=False)
    for mid in (1, 2):
        x.iosfc_cmd(CMD_UNMAP, mid, "unmap %d after teardown" % mid)
    s.events("surfaces unmapped")
    s.api("release")
    return s


@scenario(setups=("map", "isolate"), host=True)
def interplay_session_concurrent():
    """Three apps at once: tasks 1-3 each run a full phase on their own
    FIFO (root, child 1, child 2) with no barriers, rung together, twice,
    with a present, a cursor move and device info between the rounds; each
    task's two rounds are the same bytes and the three renders are equal.
    Map and isolate only: in the blocks setup every exec reads its command
    buffers through the host's read_memory callback, so concurrent FIFOs
    interleave block calls by timing, which is no order (decision 8)."""
    x = Session("interplay_session_concurrent", interplay_session_concurrent.__doc__, setups=("map", "isolate"))
    s = x.s
    x.negotiate()
    x.root()
    x.kernel_task()
    x.connect(0)
    ts = [x.task(i, 0x40 * i) for i in (1, 2, 3)]
    x.child(1)
    x.child(2, pages=2)
    for rnd in (0, 1):
        x.batch([(fid, t, t.blit_segments(rnd, 0x99) + t.compute_segments(rnd) + t.render_segments(rnd), ())
                 for fid, t in zip((0, 1, 2), ts)], "round %d" % rnd)
        for t in ts:
            x.results(t, rnd)
        x.present(0, rnd)
        x.cursor(0, 4 + rnd, 2, glyph=(rnd == 0))
        x.device_info(max_key=10 + rnd)
    for t in ts:
        x.same(t, 0, t, 1, "task %d: two rounds" % t.tid)
    _render_slots_equal(x, ts[0], 0, ts[2], 1, "task 1 and task 3 render the same")
    x.teardown()
    return s


@scenario(host=True)
def interplay_session_multihead():
    """Three ports, two displays at creation. The driver connects ports 0
    and 1, then port 2's page with no display; a display plugged into port
    2 later runs its state machine (the mode count appears) and the driver
    acks. Frames and cursors on every head, three tasks on the root FIFO
    and children 1-3 (each exec waiting on the one before), then port 1's
    display is unplugged: the page's mode count is cleared, a present on
    that port reaches no display, and the other heads carry on. Teardown
    disconnects every port."""
    x = Session("interplay_session_multihead", interplay_session_multihead.__doc__, ports=3, displays=(0, 1))
    s = x.s
    x.negotiate(asks=(0xF4, 0x10000))
    x.root()
    x.device_info()
    x.kernel_task(frames=3)
    x.connect(0)
    x.connect(1)
    x.connect(2, has_display=False)
    x.page(2, "no display yet")
    x.hotplug(2)
    s.events("three heads", order=True)
    for port in (0, 1, 2):
        x.page(port, "connected")
    for port in (0, 1, 2):
        x.present(port, port)
        x.cursor(port, 2 * port + 1, port)
    ts = [x.task(i, 0x21 * i) for i in (1, 2, 3)]
    for fid in (1, 2, 3):
        x.child(fid)
    x.batch([(fid, t, t.blit_segments(0, 0x10 * fid) + t.compute_segments(0) + t.render_segments(0), ())
             for fid, t in zip((1, 2, 3), ts)], "three apps on three children", serial=True)
    for t in ts:
        x.results(t, 0)
    s.events("work done")
    x.unplug(1, 1)
    s.events("port 1 unplugged", order=True)
    x.page(1, "unplugged")
    s.submit(abi.OP_DISPLAY_TRANSACTION3, struct.pack("<IIIQQII", 1, 0, x.backings[0][0], 0, 0, 0, 0),
             label="present on the unplugged port")
    s.quiesce(100)
    s.events("present on the unplugged port")
    x.page(1, "after a present with no display")
    x.present(2, 0)
    x.present(0, 2)
    x.batch([(0, ts[1], ts[1].blit_segments(1, 0x20) + ts[1].compute_segments(1) + ts[1].render_segments(1), ()),
             (3, ts[0], ts[0].blit_segments(1, 0x10) + ts[0].compute_segments(1) + ts[0].render_segments(1), ())],
            "two apps, other FIFOs", serial=True)
    x.same(ts[0], 0, ts[0], 1, "task 1 again, on child 3")
    x.same(ts[1], 0, ts[1], 1, "task 2 again, on the root FIFO")
    for port in (0, 2):
        s.api("present_counts", display=port)
    x.teardown()
    return s


@scenario(host=True)
def interplay_session_suspend_twice():
    """Two migrations in one session, with the guest changing between them.
    Before the first: tasks 1 and 2 on children 1 and 2, a frame, the
    cursor. After it the resumed device runs the same work (equal), task 3
    is defined, child 3 too, and task 2 is deleted with its objects. After
    the second the resumed device knows tasks 1 and 3 and children 1-3, and
    their work equals what ran before either suspend. Frames and the cursor
    across both; teardown on the third device."""
    x = Session("interplay_session_suspend_twice", interplay_session_suspend_twice.__doc__)
    s = x.s
    x.negotiate()
    x.root()
    x.kernel_task()
    x.connect(0)
    x.present(0, 0)
    x.cursor(0, 7, 3)
    t1 = x.task(1, 0x13)
    t2 = x.task(2, 0x26)
    x.child(1)
    x.child(2)
    x.batch([(1, t1, t1.blit_segments(0, 0x77) + t1.compute_segments(0) + t1.render_segments(0), ()),
             (2, t2, t2.blit_segments(0, 0x77) + t2.compute_segments(0) + t2.render_segments(0), ())],
            "before the first suspend", serial=True)
    x.results(t1, 0)
    x.results(t2, 0)
    s.events("before the first suspend")
    x.suspend("first suspend")
    x.resume("first resume")
    _resume_display(x)
    x.batch([(1, t1, t1.blit_segments(1, 0x77) + t1.compute_segments(1) + t1.render_segments(1), ()),
             (2, t2, t2.blit_segments(1, 0x77) + t2.compute_segments(1) + t2.render_segments(1), ())],
            "after the first resume", serial=True)
    x.same(t1, 0, t1, 1, "task 1 after the first resume")
    x.same(t2, 0, t2, 1, "task 2 after the first resume")
    t3 = x.task(3, 0x13)
    x.child(3)
    x.delete_objects(t2)
    t2.delete(label="delete task 2")
    x.tasks.remove(t2)
    x.run(3, t3, t3.blit_segments(0, 0x77) + t3.compute_segments(0) + t3.render_segments(0), "task 3 on child 3")
    x.same(t1, 0, t3, 0, "task 3, of task 1's seed")
    x.present(0, 1)
    x.cursor(0, 9, 5, glyph=False)
    s.dump(s.pa(s.root_page), 0x10, "task words before the second suspend")
    s.events("before the second suspend")
    x.suspend("second suspend")
    x.resume("second resume")
    _resume_display(x)
    s.sweep(0x1000, 0x1040, label="registers after the second resume")
    x.batch([(3, t1, t1.blit_segments(2, 0x77) + t1.compute_segments(2) + t1.render_segments(2), ()),
             (1, t3, t3.blit_segments(1, 0x77) + t3.compute_segments(1) + t3.render_segments(1), ()),
             (2, t1, t1.render_segments(3), ())],
            "after the second resume", serial=True)
    x.same(t1, 0, t1, 2, "task 1 after the second resume")
    x.same(t3, 0, t3, 1, "task 3 after the second resume")
    _render_slots_equal(x, t1, 0, t1, 3, "task 1's render on child 2")
    x.present(0, 0)
    x.teardown()
    return s


def _old_guest(name, doc, version, ports):
    x = Session(name, doc, ports=ports)
    s = x.s
    x.negotiate(asks=(0xF5,), version=version)
    x.root()
    x.device_info()
    x.kernel_task()
    x.connect(0)
    x.present(0, 0)
    x.cursor(0, 5, 5)
    t = x.task(1, 0x55)
    x.child(1)
    x.run(1, t, t.blit_segments(0, 0x42) + t.compute_segments(0) + t.render_segments(0), "work on child 1")
    x.results(t, 0)
    x.suspend()
    x.resume()
    _resume_display(x)
    s.r(abi.R_VERSION, "resumed")
    s.r(abi.R_PORT_COUNT, "resumed")
    x.run(0, t, t.blit_segments(1, 0x42) + t.compute_segments(1) + t.render_segments(1), "work on the root FIFO")
    x.same(t, 0, t, 1, "before and after the resume")
    x.present(0, 1)
    x.teardown()
    return s


@scenario(host=True)
def interplay_session_old_guest():
    """Older drivers on a two-port device: a guest that settles on version
    235 or 110 sees one port (multiple displays are a 245 feature,
    getAPVFeaturesForVersion), gets the device info its version gates, and
    runs a whole session on port 0 with execs (67, from 109), a child
    FIFO, a frame, the cursor and a migration, which keeps its version."""
    return [_old_guest("interplay_session_old_guest_235", interplay_session_old_guest.__doc__, 235, 2),
            _old_guest("interplay_session_old_guest_110", interplay_session_old_guest.__doc__, 110, 2)]


ORDERS = {
    # name: the steps in order, the number of tasks, the FIFOs they run on
    "displays_first": (("display", "tasks", "work", "frame", "cursor"), 1, (0,)),
    "tasks_first": (("tasks", "work", "display", "cursor", "frame"), 2, (1, 0)),
    "interleaved": (("tasks", "display", "work", "frame", "work2", "cursor"), 3, (2, 1, 0)),
}


@scenario(host=True)
def interplay_session_orders():
    """The same session built in different orders: displays before tasks,
    tasks (and their work) before any display page, and the two
    interleaved, with 1, 2 or 3 tasks on different FIFOs. Whatever the
    order, every task's work gives the bytes the first task's does (each
    compared with a fresh run of the same seed), and each ends in orderly
    teardown."""
    out = []
    for name, (steps, ntasks, fids) in ORDERS.items():
        x = Session("interplay_session_orders_" + name, interplay_session_orders.__doc__)
        s = x.s
        x.negotiate(asks=())
        x.root()
        x.kernel_task()
        for fid in sorted(set(fids) - {0}):
            x.child(fid)
        ts = []
        for step in steps:
            if step == "display":
                x.connect(0)
                x.page(0, "connected")
            elif step == "tasks":
                ts = [x.task(i + 1, 0x19) for i in range(ntasks)]
            elif step in ("work", "work2"):
                ph = 0 if step == "work" else 1
                x.batch([(fid, t, t.blit_segments(ph, 0x61) + t.compute_segments(ph) + t.render_segments(ph), ())
                         for fid, t in zip(fids, ts)], step, serial=True)
            elif step == "frame":
                x.present(0, 0)
            elif step == "cursor":
                x.cursor(0, 8, 8)
            s.events(step)
        x.results(ts[0], 0)
        for t in ts[1:]:
            x.same(ts[0], 0, t, 0, "task %d equals task 1" % t.tid)
        if "work2" in steps:
            for t in ts:
                x.same(t, 0, t, 1, "task %d's second round" % t.tid)
        x.teardown()
        out.append(s)
    return out
