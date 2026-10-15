"""Guest ABI versions across whole sessions (docs/testing.md, sections 6.3,
6.8, 6.9, 6.21, 6.22, 6.24): the per-area scenarios test one gate at a
time; these run a whole guest at each version where some gate flips and
lock everything it sees at once, and guests that change version mid-life.

The gates (pgtest/versions_util.py GATES, each the lowest version whose case
in features::features_for_version sets the flag, with what reads it):
   3  efiDisplay: a version write without it tears EFI down
   5  supportsSharedTextures (type 16, key 19); off again for 7-99
  10  displayMapperSurface (8), synchronizeAndDiscardResources (62),
      mapperIOSurfaces (an in-process assert)
  11  discardResources (63)
  21  displaySleep (9)
  22  supportsRGhAPresents (the guest display logs, and allows)
  31  dualPlaneTextures (types 12 and 18, key 12); off at 40
  40  displaySetProperties (10)
  43  metalHeaps (64; keys 14-16)
  60  bufferFromIOSurface (key 17)
  61  maxFIFOCount 64 (else 32 for any written version; 0 before one)
  62  s8ByteCountForD32S8 1 (else 4)
 105  supportsProgrammableSamplePositions (key 21)
 106  supportsVariableRasterizationRateMaps (66)
 107  supportsTileShadersImageBlocks (keys 23-27)
 108  supportsDisplayCompositorParameters (11)
 109  supportsCmdExecIndirect3 (67)
 111  supportsDynamicAttributeStride (key 28 bit 1)
 113  supportsMultiplaneMippedBlitTextures (dual-plane blit textures and
      shared texture backings)
 202  supportsCommandBufferJump (key 36)
 204  supportsHeapHazardTracking (the service's detach heaps)
 207  supportsArgumentBuffers (68; the service's object and placement
      lists; read-only buffers and heaps in the service)
 208  supportsDisplaySetGuestICCProfile (12)
 231  supportsIndirectCommandBuffers (the service's GPU-address registration)
 235  supportsInfoIndirect (69)
 238  supportsDisplayPortConnectionType (the resume's check)
 245  supportsMultipleDisplays (0x122c, 0x1234)
Requests that are not versions: 1 and 236 are stored with bit 31 set
(unsupported: no flag, a migration refuses them); 0 is kept (no flag); 234
steps down to 233, 244 to 238; bit 31 is ignored.

What the device reads as the version at use, not at creation: every FIFO
gate and device info read the device's current features; an in-process
task reads the device's too, but a service task keeps the copy it was
created with. So a version changed while tasks live splits the setups.
"""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.display_guest import p_compositor
from pgtest.fifo import ChildFIFO
from pgtest.registry import scenario
from pgtest.resources_build import texture_rec
from pgtest.session import FH, FW
from pgtest.versions_util import (DS, VGuest, ds_dims, effective, fifo_count, heap_query, outcome, resume_display,
                                  u32s)

# The requests whose sessions are locked: each version where a gate flips,
# the one below where it matters, and the requests that are not versions.
GATE_SESSIONS = (0, 1, 2, 3, 5, 10, 11, 21, 31, 40, 41, 43, 60, 61, 62, 100, 106, 108, 109, 113, 204, 207, 208,
                 231, 234, 235, 236, 237, 238, 245)


def gate_session(req):
    """The whole session at request `req`; see interplay_versions_gate."""
    name = "interplay_versions_gate_%d" % req
    x = VGuest(name, interplay_versions_gate.__doc__)
    s = x.s
    x.set_version(req, "settled")
    x.root()
    x.device_info()
    x.kernel()
    x.connect(0)
    x.present(0, 0)
    x.cursor(0, 6, 3)
    x.page(0, "frame and cursor")
    x.display_ops(0)
    x.connect(1)
    x.present(1, 1)
    x.page(1, "port 1: connected and presented")
    top = x.children_at_limit()
    worker = x.child(1)
    for c in (worker, top):
        c.submit(abi.OP_NOP, barriers=[(0, s.fifo.stamp)], label="child %d: a NOP behind the root" % c.fid)
    t = x.probe_task(1)
    x.task_ops(t)
    s.events("before the suspend")
    x.suspend()
    if x.unsupported():
        # rvg_device_will_resume sets the blob's version first and
        # refuses one it marks unsupported (InvalidGuestVersion); the
        # migration ends there.
        s.api("recreate", label="resume")
        s.api("willResume", label="resume: refused")
        s.r(abi.R_VERSION, "refused")
        return s
    x.resume()
    resume_display(x)
    s.r(abi.R_VERSION, "resumed")
    s.r(abi.R_PORT_COUNT, "resumed")
    s.r(abi.R_PORT_TYPES, "resumed")
    s.fault_state("resumed")
    s.dump(s.pa(s.root_page, abi.ROOT_CHILD_NODE), abi.CHILD_NODE_SIZE, "resumed: child 1's node")
    s.dump(top.node_gpa(), abi.CHILD_NODE_SIZE, "resumed: child %d's node" % top.fid)
    for c in (worker, top):
        c.submit(abi.OP_NOP, label="child %d after the resume" % c.fid)
    t.run55(10, label="task 1: 55 after the resume")
    t.run67(11, x.has("exec3"), label="task 1: 67 after the resume")
    t.run67(12, x.has("infoIndirect"), opcode=abi.OP_INFO_INDIRECT, label="task 1: 69 after the resume")
    t.dump_probe("after the resume")
    x.present(0, 0)
    x.present(1, 0, encode=False)
    x.finish()
    return s


@scenario(host=True)
def interplay_versions_gate():
    """A whole guest at each version where a gate flips (the module's list)
    and at the requests that are not versions (0, 1, 234, 236): a two-port
    device (0x1034, 0x122c, 0x1234 as the guest reads them), the root FIFO,
    device info at max key 45, task 0 with two backings, an ICC profile and
    a shared texture backing; port 0 connected, a frame, the cursor through
    the doorbell 0x1220 (whose displayDoorbell flag nothing reads), then
    sleep/wake (9), a colour matrix (10), compositor parameters (11), the
    ICC profile (12) and a frame; port 1 connected and presented to, below
    245 too (the port checks use the host's count); the highest child FIFO
    the version allows and the next refused; task 1 running 55, 67, 69, 62,
    63, 64, each gated object type in its own 55 (textures 12 and 16,
    tracked and read-only heaps, a read-only buffer: whether the setups
    agree is locked per setup), 68, and a 55 after it. Then a migration at
    that version: the blob, the fresh device's 0x1034, the children and
    task 1 restored, the gated execs again, a frame on each port, and
    teardown. The unsupported requests end at the refused
    rvg_device_will_resume."""
    return [gate_session(v) for v in GATE_SESSIONS]


# ---- a guest walking the versions on one device ----

WALK = (2, 3, 5, 6, 10, 11, 20, 21, 30, 31, 40, 41, 42, 43, 60, 61, 62, 100, 105, 106, 107, 108, 109, 110,
        111, 112, 113, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 230, 231, 232, 233, 235, 237, 238, 245,
        0, 1, 236, 0x80000000, 0xF4, 0x7F)


@scenario(host=True)
def interplay_versions_walk():
    """One guest that writes every valid version in turn (and 0, 1, 236, a
    request with bit 31, 0xf4 and 0x7f last) while a task, a display, two
    child FIFOs and EFI's registers live, and after each write probes what
    that version decides: 0x1034, 0x122c, 0x1234, and 0x1200 (EFI is
    enabled again before each write); device info
    (max key 45); the displays' 9 and 11; the task's 67, 69, 62, 63, 64 and
    a two-plane texture listed in a 55 (a new object each time: the task
    reads the device's current features in-process and its own copy in the
    service, so the setups differ); the top child the version allows
    defined and deleted, and a NOP on child 31 (whose stamp index every
    version allows). Nothing is torn down between versions."""
    x = VGuest("interplay_versions_walk", interplay_versions_walk.__doc__, watchdog_s=240)
    s = x.s
    x.set_version(245, "first")
    x.root()
    x.kernel()
    x.connect(0)
    x.present(0, 0)
    c31 = x.child(31)
    t = x.probe_task(1)
    extra = []
    # One fresh two-plane texture per step, so each is created at that
    # step's version (objects are cached by reference once made), with the
    # stencil texel the step's version asks for. They share two data
    # pages; their records go in a region of their own, past the command
    # buffers (GPUTask keeps pages 1-3 for records, 4-7 for command
    # buffers).
    p1, p4 = t.area(PAGE) // PAGE, t.area(PAGE) // PAGE
    t.ol._rec = t.area(8 * PAGE)
    for _ in WALK:
        extra.append((t.add(abi.T_DUAL_PLANE_TEXTURE, texture_rec(PAGE, p1, ds_dims(s8=1), DS)),
                      t.add(abi.T_DUAL_PLANE_TEXTURE, texture_rec(PAGE, p4, ds_dims(s8=4), DS))))
    assert t.ol._rec <= t._blob
    t.install(label="task 1's longer list")
    for i, req in enumerate(WALK):
        s.w(abi.R_EFI_ENABLE, 1)
        x.set_version(req, "walk 0x%x" % req)
        lab = " at 0x%x" % req
        s.r(abi.R_EFI_ENABLE, "EFI" + lab)
        x.device_info()
        outcome(s, abi.OP_DISPLAY_SLEEP_STATE, struct.pack("<Ib3x", 0, 0), x.has("displaySleep"), "9" + lab)
        outcome(s, abi.OP_DISPLAY_COMPOSITOR_PARAMETERS, p_compositor(0, 1, 2, 3, 1), x.has("compositor"),
                "11" + lab)
        t.run67(1, x.has("exec3"), label="67" + lab)
        t.run67(2, x.has("infoIndirect"), opcode=abi.OP_INFO_INDIRECT, label="69" + lab)
        outcome(s, abi.OP_SYNCHRONIZE_AND_DISCARD_RESOURCES, u32s(1, 0), x.has("syncAndDiscard"), "62" + lab)
        outcome(s, abi.OP_DISCARD_RESOURCES, u32s(1, 0), x.has("discardResources"), "63" + lab)
        outcome(s, abi.OP_HEAP_TEXTURE_SIZE_AND_ALIGN, heap_query(1, t.reply_va), x.has("metalHeaps"), "64" + lab)
        s8one, s8four = extra[i]
        t.run55(3, ok=None, extra=[(s8one if x.has("s8one") else s8four, 0)], label="type 12" + lab)
        n = fifo_count(x.v)
        top = ChildFIFO(s, n - 1)
        top.setup_node()
        if n - 1 != 31:
            top.define(label="define child %d%s" % (n - 1, lab))
            top.delete(label="delete child %d%s" % (n - 1, lab))
        c31.submit(abi.OP_NOP, label="child 31" + lab)
        t.dump_probe("walk" + lab)
        t.fill(t.probe_va, 0x10 * t.SLOTS, 0)
    s.events("walked")
    x.set_version(245, "last")
    x.finish()
    return s


# ---- the setups split when the version changes under live tasks ----

def _new_objects(x, t):
    """A second set of gated objects appended to task t's list and the list
    set again (51): new references, so new creations, at the version the
    task reads now. Returns [(name, ref)]."""
    from pgtest.versions_util import DS, DS2, ds_dims
    from pgtest.resources_build import buffer_rec, heap_rec, shared_texture_rec, texture_rec
    s8 = 1 if x.has("s8one") else 4
    objs = [
        ("two-plane D32S8 (new)", t.add(abi.T_DUAL_PLANE_TEXTURE,
                                        texture_rec(PAGE, t.area(PAGE) // PAGE, ds_dims(s8=s8), DS))),
        ("two-plane D32S8, 2 levels (new)", t.add(abi.T_DUAL_PLANE_TEXTURE, texture_rec(
            PAGE, t.area(PAGE) // PAGE, ds_dims(s8=s8, levels=2), DS2))),
        ("shared texture (new)", t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(x.k17))),
        ("tracked heap (new)", t.add(abi.T_HEAP, heap_rec(t.area(4 * PAGE) // PAGE, 4 * PAGE, 3))),
        ("read-only buffer (new)", t.add(abi.T_BUFFER, buffer_rec(PAGE, t.area(PAGE) // PAGE, 2))),
        ("read-only heap (new)", t.add(abi.T_HEAP, heap_rec(t.area(4 * PAGE) // PAGE, 4 * PAGE, 5))),
    ]
    t.install(label="task %d's list, longer" % t.tid)
    return objs


def _live_change(first, second):
    name = "interplay_versions_live_%d_to_%d" % (first, second)
    x = VGuest(name, interplay_versions_live.__doc__, ports=1)
    s = x.s
    x.set_version(first, "first")
    x.root()
    x.kernel()
    a = x.probe_task(1)
    a.objects_probe(0, label=" at %d" % first)
    x.set_version(second, "changed")
    a.objects_probe(6, label=" listed again at %d" % second)
    b = x.probe_task(2)
    b.objects_probe(0, label=" (task 2, defined at %d)" % second)
    for i, (nm, ref) in enumerate(_new_objects(x, a)):
        a.run55(12 + i, ok=None, extra=[(ref, 0)], label="task 1: %s at %d" % (nm, second))
    a.run67(18, x.has("exec3"), label="task 1: 67 at %d" % second)
    a.dump(a.probe_va, 0x130, "task 1 probe")
    b.dump(b.probe_va, 0x60, "task 2 probe")
    x.set_version(first, "back")
    b.objects_probe(6, label=" (task 2's listed again at %d)" % first)
    b.dump(b.probe_va + 0x60, 0x60, "task 2 probe, back at %d" % first)
    s.events("end")
    x.finish()
    return s


@scenario(host=True)
def interplay_versions_live():
    """The version changed while tasks live (245 to 30 and back, and 30 to
    245 and back). Task 1, defined at the first version, lists each gated
    object (two-plane textures, one and two levels; a shared texture; a
    tracked heap; a read-only buffer and heap) at that version, then again
    after the change (made objects are cached by reference; a refused one
    is tried again), then a second set of the same kinds under new
    references; task 2 is defined after the change and lists its own. The
    device's gates follow the change at once (67 at the new version);
    in-process tasks create objects by the device's current features, the
    service's by the copy the task was created with, so the setups differ
    exactly where task 1's new objects meet a flag the two versions
    disagree on. The guest lays out each two-plane texture's stencil plane
    with its current version's texel (4 bytes below 62, 1 from it:
    s8ByteCountForD32S8), so a texture the 30 guest writes is refused by a
    task that reads 245's features (task 1 in the service; task 2
    in-process once back at 245) though both allow two-plane textures.
    Back at the first version, task 2's objects are listed again."""
    return [_live_change(245, 30), _live_change(30, 245)]


# ---- a migration over a lowered version ----

@scenario
def interplay_versions_suspend_lowered():
    """Children 1, 31, 40 and 63 defined at 245 (maxFIFOCount 64), each with a
    NOP; the guest then lowers the version to 60 (32 FIFOs) and migrates. The
    blob carries 60, and the fresh device rebuilds only the children below its
    maxFIFOCount (rvg_device_did_resume walks ids 1 to maxFIFOCount - 1): 1 and
    31 run on, while 40 and 63, whose nodes still say running, are not FIFOs
    there (a doorbell on them does nothing). Raised to 245 again, the guest
    defines 40 and 63 anew (their nodes as it left them: a define over a
    running node is a new FIFO) and each runs a NOP."""
    x = VGuest("interplay_versions_suspend_lowered", interplay_versions_suspend_lowered.__doc__, ports=1,
               host=False)
    s = x.s
    s.track_stamps(64)
    x.set_version(245, "first")
    x.root()
    kids = [x.child(fid) for fid in (1, 31, 40, 63)]
    for c in kids:
        c.submit(abi.OP_NOP, label="child %d at 245" % c.fid)
    x.set_version(60, "lowered")
    s.events("lowered")
    x.suspend()
    x.resume()
    s.events("resumed")
    s.r(abi.R_VERSION, "resumed")
    s.fault_state("resumed")
    s.dump(s.pa(s.root_page, abi.ROOT_CHILD_NODE), abi.CHILD_NODE_SIZE * 63, "resumed: child nodes")
    for c in kids[:2]:
        c.submit(abi.OP_NOP, label="child %d after the resume" % c.fid)
    for c in kids[2:]:
        st = c.cmd(abi.OP_NOP)
        c.doorbell()
        c.wait_stamp(st, timeout_ms=300, label="child %d after the resume: not a FIFO" % c.fid)
    s.events("after the resume at 60")
    x.set_version(245, "raised")
    for c in kids[2:]:
        c.written = 0
        c.setup_node()
        c.define(label="define child %d again" % c.fid)
        c.submit(abi.OP_NOP, label="child %d defined again" % c.fid)
    s.dump(s.fifo.base_gpa(), 0x100, "stamp slots")
    s.events("end")
    x.finish()
    return s


@scenario(host=True)
def interplay_versions_suspend_info():
    """The blob's device info is written at the version the device has when
    it suspends, up to the last query's max key
    (DeviceCore::serialize_suspend_state; device_info::write_device_info
    reads the device's current features): a guest that queried at 245 and
    lowered to 43 migrates a 43 blob (no keys 17, 19, 21, 23-27, 36; key 28
    without bit 1), and one that queried at 2 and raised to 245 a 245 blob.
    Each resumes onto a fresh device, whose capability check runs at the
    blob's version and passes; the resumed guest queries again, and a second
    migration carries the max key the first blob named."""
    out = []
    for first, second in ((245, 43), (2, 245)):
        name = "interplay_versions_suspend_info_%d_%d" % (first, second)
        x = VGuest(name, interplay_versions_suspend_info.__doc__, ports=1)
        s = x.s
        x.set_version(first, "queried at")
        x.root()
        x.device_info()
        x.set_version(second, "suspended at")
        x.suspend(label="first")
        x.resume(label="first")
        s.events("resumed")
        s.r(abi.R_VERSION, "resumed")
        x.device_info()
        x.set_version(first, "back")
        x.suspend(label="second")
        x.resume(label="second")
        s.r(abi.R_VERSION, "resumed again")
        s.submit(abi.OP_NOP, label="the root FIFO after two migrations")
        s.events("end")
        out.append(s)
    return out


@scenario(host=True)
def interplay_versions_resume_raise():
    """A 110 guest (67, no 69, 32 FIFOs) with a task, a child and a frame
    migrates; the resumed device is at 110 and the guest's work runs there
    as before, then the guest raises its version to 245 on the resumed
    device: 69 and child 63 become available at once, device info grows,
    the restored task's objects keep working, and a second migration
    carries 245. The work before, between and after the migrations gives
    the same bytes."""
    from pgtest.session import SessionTask
    from pgtest.versions_util import work
    x = VGuest("interplay_versions_resume_raise", interplay_versions_resume_raise.__doc__, ports=1)
    s = x.s
    x.set_version(110, "first")
    x.root()
    x.kernel()
    x.connect(0)
    x.present(0, 0)
    x.cursor(0, 3, 3)
    c = x.child(1)
    t = x.task(2, 0x33)
    p = x.probe_task(1)
    work(x, t, 0, fid=1)
    x.results(t, 0)
    x.suspend(label="at 110")
    x.resume(label="at 110")
    resume_display(x)
    s.r(abi.R_VERSION, "resumed")
    work(x, t, 1, fid=1)
    x.same(t, 0, t, 1, "at 110, before and after the migration")
    p.run67(1, False, opcode=abi.OP_INFO_INDIRECT, label="69 at 110 after the resume")
    x.set_version(245, "raised on the resumed device")
    p.run67(2, True, opcode=abi.OP_INFO_INDIRECT, label="69 at 245")
    c63 = x.child(63)
    c63.submit(abi.OP_NOP, label="child 63 at 245")
    x.device_info()
    work(x, t, 2, fid=63)
    x.same(t, 0, t, 2, "at 245 on child 63")
    p.dump_probe("after the raise")
    x.suspend(label="at 245")
    x.resume(label="at 245")
    resume_display(x)
    s.r(abi.R_VERSION, "resumed at 245")
    work(x, t, 3, fid=63)
    x.same(t, 0, t, 3, "after the second migration, on child 63")
    x.present(0, 1)
    x.finish()
    return s


# ---- stop and reset, and a guest that sets up again at another version ----

def _resetup(call, first, second):
    """call: "reset" or "stop"; second: the version the guest writes when it
    sets up again (None: it writes none)."""
    from pgtest.versions_util import work
    tag = "none" if second is None else str(second)
    name = "interplay_versions_resetup_%s_%d_%s" % (call, first, tag)
    x = VGuest(name, interplay_versions_resetup.__doc__, ports=1)
    s = x.s
    x.set_version(first, "first")
    x.root()
    x.device_info()
    x.kernel()
    x.connect(0)
    x.present(0, 0)
    x.cursor(0, 4, 4)
    x.child(1)
    t = x.task(1, 0x61)
    work(x, t, 0, fid=1)
    x.results(t, 0)
    s.events("before " + call)
    s.api(call)
    s.events(call)
    s.sweep(0x1000, 0x1040, label="registers after " + call)
    s.r(abi.R_PORT_COUNT, "after " + call)
    s.diff("root page after " + call, gpa=s.pa(s.root_page), length=PAGE)
    # Setting up again: the version (or none), the root page, the FIFO from
    # written 0, the child, task 0 and task 1 (their pages kept their
    # contents: nothing on the host did), the display page.
    if second is not None:
        x.set_version(second, "second")
    else:
        s.r(abi.R_VERSION, "no version written")
        x.v = effective(0) if call == "reset" else x.v
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    s.fifo.written = 0
    s.fifo.stamp = 0
    s.fifo.enable()
    s.submit(abi.OP_NOP, label="first NOP again")
    outcome(s, abi.OP_EXEC_INDIRECT2, struct.pack("<III", 1, 0, 0), False, "task 1 before it is defined again")
    c = x.children[1]
    c.written = 0
    c.setup_node()
    c.define(label="define child 1 again")
    x.t0.define(label="define task 0 again")
    x.t0.install(label="task 0's list again")
    t.define(label="define task 1 again")
    t.install(label="task 1's list again")
    del x.pages[0]
    x.connect(0)
    x.page(0, "connected again")
    x.present(0, 1)
    work(x, t, 1, fid=1)
    x.same(t, 0, t, 1, "the work before the %s and after" % call)
    outcome(s, abi.OP_INFO_INDIRECT, struct.pack("<III", 1, 0, 0), x.has("infoIndirect"), "69 with no ranges")
    x.suspend()
    x.resume()
    s.events("resumed")
    s.r(abi.R_VERSION, "resumed")
    work(x, t, 2, fid=1)
    x.same(t, 0, t, 2, "after the migration")
    x.finish()
    return s


@scenario(host=True)
def interplay_versions_resetup():
    """reset or stop under a live session, then the guest sets everything
    up again, at another version or at none. -reset tears the root down
    (FIFOs, tasks, displays) and writes version 0, EFI off and device-info
    max key 1; -stop only writes root page 0. What carries over is what
    the device holds outside the root: the version after stop, 0 after
    reset (so a guest that writes none gets version 0's features: 67 and
    69 refused, 32 FIFOs), and the device-info max key the next blob uses.
    Tasks do not come back from the root page's task words (only a resume
    reads them): an exec on task 1 faults until it is defined again. The
    work before the teardown, after the new setup, and after a migration
    at the end gives the same bytes, by 67 or 55 as the version allows."""
    return [_resetup("reset", 245, 110), _resetup("reset", 110, 245), _resetup("reset", 245, None),
            _resetup("stop", 110, None), _resetup("stop", 110, 245)]


# ---- EFI and the driver's version ----

def _efi_case(name, doc, versions, efi_first=True, suspend=False):
    x = VGuest(name, doc, ports=1, efi=True)
    s = x.s
    if efi_first:
        x.efi_boot(seed=0x21)
    for v in versions:
        x.set_version(v, "driver 0x%x" % v)
        x.efi.regs("after version 0x%x" % v) if x.efi else None
        s.quiesce(100)
        s.events("after version 0x%x" % v)
    if not efi_first:
        x.efi_boot(seed=0x42)
    x.root()
    x.kernel()
    if suspend:
        s.events("before the suspend")
        x.suspend()
        x.resume()
        s.quiesce(100)
        s.events("resumed")
        s.r(abi.R_VERSION, "resumed")
        x.efi.regs("resumed")
        s.encode(0, FW, FH, label="the display after the resume", samples=[[0, 0], [FW - 1, FH - 1]])
        x.set_version(x.v, "the version written again")
        x.efi.regs("after the version again")
    x.connect(0)
    x.page(0, "connected")
    x.present(0, 0)
    s.w(abi.R_EFI_ENABLE, 0)
    x.efi.regs("EFI off")
    s.quiesce(100)
    s.events("EFI off")
    x.present(0, 1)
    x.finish()
    return s


@scenario(host=True)
def interplay_versions_efi():
    """EFI and the driver's version: the firmware's framebuffer comes up
    before any version (_features all zero, nothing tears it down), then
    the driver writes its version: 0 and 2 (no efiDisplay) tear EFI down
    and zero its registers, 3 and 245 keep it presenting; 2 then 245
    cannot bring it back. A guest at 2 that enables EFI after its version
    write keeps it (the flag is checked only at a version write), and a
    migration then restores it on the fresh device (rvg_device_did_resume
    replays the EFI registers after rvg_device_will_resume set version 2) until
    the guest writes its version again. Each driver then connects port 0,
    presents, turns EFI off and presents again."""
    d = interplay_versions_efi.__doc__
    return [_efi_case("interplay_versions_efi_0", d, (0,)),
            _efi_case("interplay_versions_efi_2", d, (2,)),
            _efi_case("interplay_versions_efi_3", d, (3,)),
            _efi_case("interplay_versions_efi_245", d, (245,)),
            _efi_case("interplay_versions_efi_2_245", d, (2, 245)),
            _efi_case("interplay_versions_efi_after_2", d, (2,), efi_first=False, suspend=True)]


# ---- what the service's tasks copy at their creation ----

@scenario(setups=("isolate",))
def interplay_versions_icb_registration():
    """Only the service registers GPU addresses, and only for tasks created
    at 231 or later (supportsIndirectCommandBuffers, copied into the task at
    its creation). Task 1 is defined at 230 and task 2 after the guest
    raises to 231; each runs the same compute ICB (bound to its out buffer
    by reference, which needs no registration) listed with invalidateGuest,
    then SynchronizeResources (53) writes the ICB's command back: the bind's
    reference comes from the registered address (fillBufferBind), so task
    1's reads ~0 with offset 0 and task 2's its buffer. Only the bind's
    reference and offset are dumped: the GPU address is the host's. The
    service alone: in-process the write-back aborts
    (icb_sync_in_process)."""
    from scenarios.icb import ICBTask, sync_refs
    from pgtest.icb_cmds import compute_cmd
    s_ = VGuest("interplay_versions_icb_registration", interplay_versions_icb_registration.__doc__, ports=1,
                setups=("isolate",), host=False)
    s = s_.s
    s_.set_version(230, "first")
    s_.root()
    ws = []
    for tid, v in ((1, 230), (2, 231)):
        if v != s_.v:
            s_.set_version(v, "raised")
        w = ICBTask(s, tid=tid)
        a = w.icb([compute_cmd(pso=w.cpso, binds=[(w.out, 0, 0x10)], grid=(4, 1, 1), tg=(4, 1, 1))])
        w.start()
        ws.append((w, a))
    for w, a in ws:
        tid = w.t.tid
        w.run_compute(a, flags=abi.RES_INVALIDATE_GUEST, label="task %d: execute, guest invalidated" % tid)
        w.out_dump("task %d: executed" % tid, 0x20)
        va, _ = w.icbs[a]
        w.t.fill(va, 0x100, 0xEE)
        sync_refs(s, tid, [a], label="task %d: synchronize" % tid)
        w.t.dump(va, 0x10, "task %d: type, barrier, pipeline" % tid)
        w.t.dump(va + 0x10, 4, "task %d: the bind's reference" % tid)
        w.t.dump(va + 0x10 + 12, 8, "task %d: the bind's offset" % tid)
    s.events("end")
    for w, _ in ws:
        w.t.delete(label="delete task %d" % w.t.tid)
    return s


@scenario(setups=("isolate",))
def interplay_versions_argbuf_snapshot():
    """With argument buffers enabled (the descriptor, as QEMU's
    apple-gfx-mmio sets it), a service task keeps its object and placement
    lists by the version it was created at (supportsArgumentBuffers, 207)
    while the opcode's gate reads the device's. Task 1 is defined at 206 and
    task 2 after the guest raises to 207; both set an object and placement
    list with 68 at 207, then list a read-only buffer (refused in the
    service below 207): the task roots show which recorded a placement list,
    the probe which buffer was made. 68 at 206 is refused first."""
    from pgtest.resources_build import buffer_rec
    x = VGuest("interplay_versions_argbuf_snapshot", interplay_versions_argbuf_snapshot.__doc__, ports=1,
               setups=("isolate",), host=False)
    x.s._config["descriptor"] = {"enableArgumentBuffers": True}
    s = x.s
    x.set_version(206, "first")
    x.root()
    from pgtest.resources_build import RTask
    ts = []
    for tid, v in ((1, 206), (2, 207)):
        if v != x.v:
            x.set_version(v, "raised")
        t = RTask(s, tid=tid, pages=32)
        t.probe, t.probe_va = t.shared(PAGE)
        t.ro = t.add(abi.T_BUFFER, buffer_rec(PAGE, t.area(PAGE) // PAGE, 2))
        t.place = t.area(PAGE)
        t.define(label="define task %d at %d" % (tid, v))
        if tid == 1:
            t.ol.write()
            outcome(s, abi.OP_SET_OBJECT_AND_PLACEMENT_LIST,
                    struct.pack("<IQQII", 1, 0, t.place, t.ol.length(), 16 * len(t.ol.entries)), False,
                    "68 at 206")
        ts.append(t)
    for t in ts:
        t.ol.write()
        outcome(s, abi.OP_SET_OBJECT_AND_PLACEMENT_LIST,
                struct.pack("<IQQII", t.tid, 0, t.place, t.ol.length(), 16 * len(t.ol.entries)), True,
                "task %d: 68 at 207" % t.tid)
        s.dump(s.pa(t.root), abi.TASK_ROOT.size, "task %d root after 68" % t.tid)
        from pgtest.resources_build import b_fill_buf
        from pgtest.task import seg
        t.reset_cmdbufs()
        cb = t.cmdbuf(seg(abi.ENC_BLIT, b_fill_buf(t.probe, 0, 0x10, 0x77)))
        outcome(s, abi.OP_EXEC_INDIRECT3, struct.pack("<IIIIIIIQQ", t.tid, 2, 1, t.probe, 0, t.ro, 0, cb[0], cb[1]),
                None, "task %d: a read-only buffer" % t.tid)
        t.dump(t.probe_va, 0x10, "task %d probe" % t.tid)
    s.events("end")
    for t in ts:
        t.delete(label="delete task %d" % t.tid)
    return s
