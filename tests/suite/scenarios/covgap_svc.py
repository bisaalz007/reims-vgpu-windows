"""The tasks and the GPU task service: paths the rest of the suite leaves
unreached there.

What they lock:
- An object list set at an offset that is not page-aligned (68 accepts one:
  setObjectList tests the previous offsets) is read from that offset in the
  blocks setup, and cannot be mapped with a memory map: the task's
  ResourceHost::new_virtual_mapping refuses the offset, in-process and in
  the service alike, ObjectLists::new_mapping catches that and returns no
  mapping, and the exec faults.
- Paging that throws inside an encoder: the encoder the throw left open,
  blit or compute, is still ended, and the paging command buffer committed.
  When it is the exec's synchronize that throws, the exec's command buffer
  and the synchronize command buffer are both committed, and in-process the
  exec's completion handler runs twice: once from the exec's failure (status
  2) and once from the synchronize command buffer's completion. The second
  completion of the stamp kills the host (the pending stamp's asserts)
  unless another command's stamp is pending then, a race; the scenarios here
  make the throwing exec the last command. In the service the exec replies 2
  and the synchronize command buffer's completion sends success; the host
  keeps the first to arrive and the other finds its token gone. Which
  arrives first is timing unless something orders it: the isolate scenarios
  make the exec's command buffer run a kernel of fixed work, seconds against
  the reply's milliseconds (pgtest/spin_util.py).
"""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.compute_objects import Compute as C, Rig, function_record
from pgtest.covgap_svc_util import SYNC_BACK, last_exec, point, punch
from pgtest.registry import scenario
from pgtest.resources_build import (RGBA8Unorm, XR10_2P_A8, RTask, b_fill_tex_color, cmd_discard, desc1,
                                    simple_dim, texture_rec)
from pgtest.scenario import Scenario
from pgtest.spin_util import Spin
from pgtest.task import GPUTask, seg
from pgtest.tasks_lib import exec3_payload, fault_and_resume, fill_rec, set_lists_raw

E3 = abi.OP_EXEC_INDIRECT3
IN_PROCESS = ("blocks", "map")


def _fill_exec(t, buf, off, byte, resources):
    """An ExecIndirect3 payload of one blit segment filling 0x10 bytes of
    buf at off with byte."""
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(buf, off, 0x10, byte)))
    return exec3_payload(t.tid, [cb], resources)


# ---- the object list at an offset that is not page-aligned ----

def _misaligned(name, doc, setups, fault):
    s = Scenario(name, setups=setups, doc=doc)
    s.bring_up()
    t = GPUTask(s, pages=16)
    buf, bva = t.buffer(PAGE)
    t.define()
    t.ol.list_va = 12
    n = t.ol.write()
    set_lists_raw(s, 1, 12, 0, n, 0, label="68: the object list at VA 12")
    label = "an exec through the list at VA 12"
    s.submit(E3, _fill_exec(t, buf, 0, 0x5A, [(buf, 0)]), fault=fault, label=label)
    if fault:
        fault_and_resume(s, label)
    t.dump(bva, 0x20, "the buffer")
    t.delete()
    s.events("end")
    return s


@scenario
def covgap_svc_objlist_misaligned():
    """An object list set with 68 at task VA 12, where the guest wrote it:
    the call is accepted, since setObjectList tests the offsets it had
    before. The blocks setup maps the list through the task's own mapping,
    at any offset, and the exec's fill through reference 1 lands. With a
    memory map the list is mapped by ResourceHost::new_virtual_mapping, which
    refuses an offset that is not page-aligned ("Attempting to map memory with
    a non-page-aligned offset"); ObjectLists::new_mapping catches it and
    returns no mapping ("Failed to map object list"), and the exec faults with
    the buffer untouched: covgap_svc_objlist_misaligned_memmap."""
    return [_misaligned("covgap_svc_objlist_misaligned", covgap_svc_objlist_misaligned.__doc__, ("blocks",),
                        False),
            _misaligned("covgap_svc_objlist_misaligned_memmap", covgap_svc_objlist_misaligned.__doc__,
                        ("map", "isolate"), True)]


# ---- paging over a hole ----

def _xr10(t):
    """A 64x4 texture of pixel format 550 (type 12), two planes over two
    task pages, as textures_dual_plane_xr10 builds it; returns (ref, VA)."""
    from scenarios.textures_dual_plane import _xr10_rec
    data = bytes((i * 37 + 11) & 0xFF for i in range(64 * 4 * 4))
    x, va = _xr10_rec(t, XR10_2P_A8, contents=data)
    t.write(va + PAGE, data[::-1])
    return x, va


@scenario
def covgap_svc_xr10_hole_prepare():
    """A texture of pixel format 550, which pages on a compute encoder,
    whose second task page is a hole. Listing it makes its transfer buffer
    over both pages, and the walk throws "Found a hole in the page table"
    inside the compute encoder: that encoder is still ended and the paging
    command buffer committed, and the exec faults in every setup, its fill
    of B not run. The next exec, without the texture, fills B. Host-keyed:
    whether Metal makes a format-550 texture is the host's."""
    s = Scenario("covgap_svc_xr10_hole_prepare", host=True, doc=covgap_svc_xr10_hole_prepare.__doc__)
    s.bring_up()
    t = RTask(s, pages=96)
    x, xva = _xr10(t)
    b, bva = t.shared(PAGE)
    punch(t, xva // PAGE + 1)
    t.define()
    t.install()
    s.submit(E3, _fill_exec(t, b, 0, 0x5A, [(x, 0), (b, 0)]), fault=True, label="an exec listing X")
    fault_and_resume(s, "an exec listing X")
    s.submit(E3, _fill_exec(t, b, 0x10, 0x5B, [(b, 0)]), label="an exec without X")
    t.dump(bva, 0x20, "B")
    t.delete()
    s.events("end")
    return s


def _normal_texture(t):
    va = t.area(PAGE, contents=b"\xA1" * 0x40)
    return t.add(abi.T_TEXTURE, texture_rec(PAGE, va // PAGE, [simple_dim(4, 4)], desc1())), va, 1


def _xr10_texture(t):
    x, va = _xr10(t)
    return x, va, 2


def _sync_hole(name, doc, make, fill, host, setups):
    """X is written by the GPU (paged in on the way), its last task page
    made a hole, the range unmapped (34) so the blocks setup walks the
    table again, and X discarded (63), which drops its transfer buffer.
    An exec listing X with invalidateGuest and synchronize then fills B:
    X's host copy is still valid, so preparing it makes nothing, and the
    synchronize remakes the transfer buffer over the hole and throws.

    In the service the exec is completed twice, by the failure's reply (2)
    and by the synchronize command buffer's completion (success), and the
    host keeps the first to arrive (pgtest/spin_util.py). The isolate
    scenarios run a kernel of fixed work (about 2 s) in the exec after the
    fill, so the fault comes first; once the kernel has ended (c[2] =
    0xdead) the late success arrives to nothing (no interrupt, the FIFO
    still faulted). They run exclusive, as pgtest/spin_util.py says."""
    spin = setups != IN_PROCESS
    s = Scenario(name, host=host, setups=setups, doc=doc,
                 **(dict(watchdog_s=90, exclusive=True) if spin else {}))
    s.bring_up()
    t = RTask(s, pages=96)
    x, xva, npages = make(t)
    b, bva = t.shared(PAGE)
    if spin:
        sp = Spin(t)
    t.define()
    t.install()
    if spin:
        sp.warm_up()
    t.blit(*fill(x), resources=[(x, 0), (b, 0)], label="X paged in, the GPU writes it")
    punch(t, xva // PAGE + npages - 1)
    t.unmap(xva, npages * PAGE, label="34 over X")
    cmd_discard(s, 1, [x], label="63 of X")
    if not spin:
        p = _fill_exec(t, b, 0, 0x5A, [(x, SYNC_BACK), (b, 0)])
        last_exec(s, E3, p, label="an exec synchronizing X: the host dies")
        return s
    t.reset_cmdbufs()
    p = sp.payload([t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(b, 0, 0x10, 0x5A)))], [(x, SYNC_BACK), (b, 0)])
    label = "an exec synchronizing X"
    st = s.submit(E3, p, fault=True, timeout_ms=30000, label=label)
    s.fault_state(label)
    s.events("the fault, the kernel still running")
    sp.wait_end()
    s.events("the exec's command buffer completed, after the fault")
    fault_and_resume(s, "after the kernel ended")
    s.fifo.wait_stamp(st)
    s.submit(E3, _fill_exec(t, b, 0x10, 0x5B, [(b, 0)]), label="an exec without X")
    t.dump(bva, 0x20, "B")
    t.dump(xva, 0x40, "X's first page")
    t.delete()
    s.events("end")
    return s


@scenario
def covgap_svc_texture_hole_synchronize():
    """A normal texture (type 2, 4x4 RGBA8Unorm), which pages on a blit
    encoder, synchronized over a hole (the setup is _sync_hole's): the throw
    leaves the blit encoder open, which is still ended, and the synchronize
    command buffer is committed. In-process the exec faults (status 2, from
    its failure) and the synchronize command buffer's completion runs its
    handler again: the stamp is completed twice, and the host dies on the
    pending stamp's asserts (the scenario's last command). In the service
    the exec faults and the task goes on:
    covgap_svc_texture_hole_synchronize_isolate, where the exec's fill of B
    ran (its command buffer was committed before the throw), the next exec
    fills B, and X's guest page keeps the guest's bytes."""
    doc = covgap_svc_texture_hole_synchronize.__doc__
    fill = lambda x: [b_fill_tex_color(x, (4, 1, 1), (1.0, 0.0, 0.0, 1.0), RGBA8Unorm)]
    return [_sync_hole("covgap_svc_texture_hole_synchronize", doc, _normal_texture, fill, False, IN_PROCESS),
            _sync_hole("covgap_svc_texture_hole_synchronize_isolate", doc, _normal_texture, fill, False,
                       ("isolate",))]


@scenario
def covgap_svc_xr10_hole_synchronize():
    """As covgap_svc_texture_hole_synchronize, with a texture of pixel
    format 550, which pages on a compute encoder: the throw leaves the
    compute encoder open. The first exec only pages it in. Host-keyed,
    as covgap_svc_xr10_hole_prepare."""
    doc = covgap_svc_xr10_hole_synchronize.__doc__
    return [_sync_hole("covgap_svc_xr10_hole_synchronize", doc, _xr10_texture, lambda x: [], True, IN_PROCESS),
            _sync_hole("covgap_svc_xr10_hole_synchronize_isolate", doc, _xr10_texture, lambda x: [], True,
                       ("isolate",))]


@scenario
def covgap_svc_xr10_hole_sync53():
    """The same texture of pixel format 550 as covgap_svc_xr10_hole_prepare,
    paged in, its second page made a hole, unmapped (34) and discarded (63),
    then listed with invalidateGuest only, and synchronized with 53:
    SynchronizeResources pages it out on a compute encoder, the transfer
    buffer's walk throws, and 53 catches it per resource, logs "Exception
    while synchronizing resource, continuing" and completes, in every setup.
    Host-keyed."""
    s = Scenario("covgap_svc_xr10_hole_sync53", host=True, doc=covgap_svc_xr10_hole_sync53.__doc__)
    s.bring_up()
    t = RTask(s, pages=96)
    x, xva = _xr10(t)
    b, bva = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(resources=[(x, 0), (b, 0)], label="X paged in")
    punch(t, xva // PAGE + 1)
    t.unmap(xva, 2 * PAGE, label="34 over X")
    cmd_discard(s, 1, [x], label="63 of X")
    s.submit(E3, _fill_exec(t, b, 0, 0x5A, [(x, abi.RES_INVALIDATE_GUEST), (b, 0)]),
             label="an exec invalidating X")
    s.submit(abi.OP_SYNCHRONIZE_RESOURCES, struct.pack("<III", 1, 1, x), label="53 of X")
    s.submit(E3, _fill_exec(t, b, 0x10, 0x5B, [(b, 0)]), label="an exec without X")
    t.dump(bva, 0x20, "B")
    t.delete()
    s.events("end")
    return s


# ---- an ICB listed with synchronize ----

def _icb_world(s):
    from scenarios.icb import ICBTask, exec_ranged, use_resources
    from pgtest.icb_cmds import compute_cmd
    w = ICBTask(s)
    a = w.icb([compute_cmd(pso=w.cpso, binds=[(w.out, 0, 0)], grid=(4, 1, 1), tg=(4, 1, 1))])
    w.start()
    w.t.reset_cmdbufs()
    cb = w.t.cmdbuf(seg(abi.ENC_COMPUTE, use_resources(w.out) + exec_ranged(a, 0, 1)))
    return w, a, exec3_payload(w.t.tid, [cb], w.res(a, flags=SYNC_BACK))


@scenario(setups=IN_PROCESS)
def covgap_svc_icb_exec_synchronize():
    """A compute ICB executed with invalidateGuest and synchronize. The ICB
    pages by CPU, but in-process LocalTask::complete_resources sorts it by
    needsComputePaging alone into the blit list, and asks it to synchronize
    in the blit encoder, which an ICB does not implement: the unrecognised
    request throws inside the blit encoder, the encoder is still ended, and
    the exec's stamp is completed twice, as in
    covgap_svc_texture_hole_synchronize: the host dies. The last command.
    The service skips a CPU-paged resource there:
    covgap_svc_icb_exec_synchronize_isolate."""
    s = Scenario("covgap_svc_icb_exec_synchronize", setups=IN_PROCESS,
                 doc=covgap_svc_icb_exec_synchronize.__doc__)
    s.bring_up()
    w, a, p = _icb_world(s)
    last_exec(s, E3, p, label="an exec, the ICB synchronized: the host dies")
    return s


@scenario(setups=("isolate",))
def covgap_svc_icb_exec_synchronize_isolate():
    """The service's complete_resources (gpu_task/paging.rs) skips a resource
    that pages by CPU, so an ICB listed with invalidateGuest and synchronize is
    not written back: the exec completes, its dispatch writes out[gid] = 2 gid,
    and the next exec completes too (in-process:
    covgap_svc_icb_exec_synchronize)."""
    s = Scenario("covgap_svc_icb_exec_synchronize_isolate", setups=("isolate",),
                 doc=covgap_svc_icb_exec_synchronize_isolate.__doc__)
    s.bring_up()
    w, a, p = _icb_world(s)
    s.submit(E3, p, label="an exec, the ICB synchronized")
    w.out_dump("out", 0x20)
    s.submit(E3, _fill_exec(w.t, w.out, 0x20, 0x5A, w.res(a)), label="the next exec")
    w.out_dump("out after", 0x40)
    w.t.delete()
    s.events("end")
    return s


# ---- a function whose data range overflows ----

@scenario
def covgap_svc_function_offset_overflow():
    """A function (type 6) whose metallib is at task VA 2^64 - 0x10, 0x100
    bytes: the copy out of task memory refuses the range, whose end
    overflows, "Failed to get function data", and Metal aborts building
    the pipeline over the nil function, as
    mutgap_svc_function_at_task_end. The abort is inside Metal, so the host
    dies in every setup, the isolate one included, and the next exec is
    never reached."""
    r = Rig("covgap_svc_function_offset_overflow", covgap_svc_function_offset_overflow.__doc__, pages=64)
    out, _ = r.buffer()
    fn = r.raw(abi.T_FUNCTION, function_record((1 << 64) - 0x10, 0x100))
    pso = r.pipeline(fn)
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)),
              fault=True, label="dispatch")
    r.compute(C.set_buffers(0, [(out, 0)]), fault=True, label="the task's service is gone")
    r.s.events("end")
    return r.s


# ---- a deletion past the object list ----

@scenario
def covgap_svc_delete_past_list():
    """DeleteResource (37) of reference 50 in a task whose object list has
    two entries, after an exec mapped both lists. The resource manager clears
    the reference's placement entry: with argument buffers (the service) the
    placement list is mapped and has a length, and getPlacementListEntry
    refuses an index past the object list's count ("ObjectRef ref(50) out of
    range(2)"), so nothing is written: the guest's bytes where slot 50 would
    be stay. In-process there is no placement list. 37 of the buffer then
    zeroes its slot (reference 1) in the service. Every command completes."""
    s = Scenario("covgap_svc_delete_past_list", doc=covgap_svc_delete_past_list.__doc__)
    s.bring_up()
    t = GPUTask(s, pages=16)
    b, _ = t.buffer(PAGE, va=10 * PAGE)
    t.define()
    n = t.ol.write()
    pva = 12 * PAGE
    t.write(pva + 50 * 16, b"\xA5" * 16)
    set_lists_raw(s, 1, 0, pva, n, 16 * (n // 12), label="68: two entries, two placements")
    s.submit(E3, _fill_exec(t, b, 0, 0x5A, [(b, 0)]), label="an exec maps both lists")
    s.submit(abi.OP_DELETE_RESOURCE, struct.pack("<II", 1, 50), label="37 of reference 50")
    t.dump(pva + 50 * 16, 16, "where slot 50 would be")
    s.submit(abi.OP_DELETE_RESOURCE, struct.pack("<II", 1, b), label="37 of B")
    t.dump(pva, 32, "slots 0 and 1")
    t.delete()
    s.events("end")
    return s


# ---- a function whose data the service cannot read ----

@scenario(setups=("isolate",))
def covgap_svc_function_outside_ram():
    """A function (type 6) whose metallib's task page is mapped by a leaf
    naming a page outside guest RAM. The service copies the function's bytes
    out of task memory through the memory map, whose read of that page fails
    and writes nothing, so the copy stays zeros. Metal cannot make a library
    of it, the function is nil, and Metal aborts building the pipeline, as
    covgap_svc_function_offset_overflow: the abort is inside
    Metal, so the host dies at the exec, although the task is isolated, and
    the next exec is never reached. Isolate only: the in-process copy of a
    page outside RAM is not locked here (docs/testing.md, section 7.5)."""
    from pgtest.gaps_memory import page_of
    from pgtest.scenario import RAM_BASE, RAM_SIZE
    r = Rig("covgap_svc_function_outside_ram", covgap_svc_function_outside_ram.__doc__, pages=64,
            setups=("isolate",))
    out, _ = r.buffer()
    fn = r.raw(abi.T_FUNCTION, function_record(40 * PAGE, 0x100))
    pso = r.pipeline(fn)
    point(r.t, 40, r.s.pfn(page_of(RAM_BASE + 2 * RAM_SIZE)))
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)),
              fault=True, label="dispatch")
    r.compute(C.set_buffers(0, [(out, 0)]), fault=True, label="the task's service is gone")
    r.s.events("end")
    return r.s


# ---- a GPU error with a synchronized resource, then InfoIndirect ----

@scenario(exclusive=True)
def covgap_svc_gpu_fault_sync_info():
    """A draw of primitive type 9, which the GPU fails (render_draw_type9),
    in an exec that also lists a private buffer with invalidateGuest and
    synchronize. The exec's completion waits for the synchronize command
    buffer, which finds the exec's command buffer in error and completes
    the exec with status 3 ("Sync command buffer failed"): the exec faults.
    An InfoIndirect (69) on the same FIFO then waits for the FIFO event to
    reach the value the failed exec signals, and it does: 69 completes at
    once, not after its 5 s timeout. exclusive: it faults the GPU."""
    from scenarios.render_draw import DrawTask
    from pgtest import render_lib as R
    s = Scenario("covgap_svc_gpu_fault_sync_info", exclusive=True, doc=covgap_svc_gpu_fault_sync_info.__doc__)
    s.bring_up()
    d = DrawTask(s)
    p, _ = d.t.buffer(PAGE, flags=1)
    d.t.define()
    d.t.install()
    d.t.reset_cmdbufs()
    cb = d.t.cmdbuf(seg(abi.ENC_RENDER, b"".join([d.pass_(), R.pipeline_state(d.p), d.vb(), R.draw16(9, 0, 3)])))
    res = [(r, 0) for r in d.t.bufs] + [(p, SYNC_BACK)]
    label = "type 9, a private buffer synchronized"
    st = s.submit(E3, exec3_payload(1, [cb], res), fault=True, timeout_ms=30000, label=label)
    fault_and_resume(s, label)
    s.fifo.wait_stamp(st)
    s.submit(abi.OP_INFO_INDIRECT, exec3_payload(1), timeout_ms=2000, label="69 after it")
    d.t.delete()
    s.events("end")
    return s


# ---- InfoIndirect listing a reference that makes nothing ----

@scenario
def covgap_svc_info_bad_resource():
    """InfoIndirect (69) listing reference 9 in a task whose object list has
    two entries: preparing it asks for the resource, and the lookup throws
    "Failed to get resource", which 69 catches ("Info command buffer
    failed", in-process and in the service alike): it faults in every setup,
    and an InfoIndirect without it completes."""
    s = Scenario("covgap_svc_info_bad_resource", doc=covgap_svc_info_bad_resource.__doc__)
    s.bring_up()
    t = GPUTask(s, pages=16)
    t.buffer(PAGE)
    t.define()
    t.install()
    label = "69 listing reference 9"
    st = s.submit(abi.OP_INFO_INDIRECT, exec3_payload(1, [], [(9, 0)]), fault=True, label=label)
    fault_and_resume(s, label)
    s.fifo.wait_stamp(st)
    s.submit(abi.OP_INFO_INDIRECT, exec3_payload(1), label="69 without it")
    t.delete()
    s.events("end")
    return s


# ---- a reply range whose end wraps ----

@scenario
def covgap_svc_heap_reply_wraps():
    """HeapTextureSizeAndAlign (64) with its 16-byte reply at task VA
    2^64 - 16. In-process the range check refuses it before anything. The
    service rounds the reply out to whole pages first, and the page end
    wraps to 0, so it asks for one page at 2^64 - 0x4000, whose end
    overflows the range check. It faults in every setup; the next 64,
    with its reply in the task, completes. Host values (the size and
    alignment Metal reports)."""
    from pgtest.tasks_lib import heap_payload, heap_ser
    s = Scenario("covgap_svc_heap_reply_wraps", host=True, doc=covgap_svc_heap_reply_wraps.__doc__)
    s.bring_up()
    t = GPUTask(s, pages=8)
    t.define()
    op = abi.OP_HEAP_TEXTURE_SIZE_AND_ALIGN
    label = "64, the reply at 2^64 - 16"
    st = s.submit(op, heap_payload(1, (1 << 64) - 16, 16, heap_ser()), fault=True, label=label)
    fault_and_resume(s, label)
    s.fifo.wait_stamp(st)
    s.submit(op, heap_payload(1, PAGE, 16, heap_ser()), label="64, the reply at PAGE")
    t.dump(PAGE, 16, "the reply")
    t.delete()
    s.events("end")
    return s


# ---- the kernel task deleted under a shared texture ----

def _kernel_deleted(name, doc, setups, fault):
    from pgtest.resources_build import b_copy_tex_buf, shared_backing_rec, shared_texture_rec
    s = Scenario(name, setups=setups, doc=doc)
    s.bring_up()
    k, t = RTask(s, tid=0, pages=32), RTask(s, tid=1)
    px = bytes((0x30 + i) & 0xFF for i in range(64))
    bk_va = k.area(PAGE, contents=px)
    bk = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, bk_va // PAGE, [simple_dim(4, 4)],
                                                         desc1(options=0x20, usage=3)))
    st_ = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(bk))
    b, bva = t.shared(PAGE)
    k.define()
    k.install()
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(st_, b, 0, 16, 64, (4, 4, 1)), resources=[(st_, 0), (b, 0)],
           label="a shared texture over task 0's backing")
    k.delete(label="32 of task 0")
    label = "invalidateGuest of the shared texture, task 0 gone"
    st = s.submit(E3, _fill_exec(t, b, 0x40, 0x5A, [(st_, abi.RES_INVALIDATE_GUEST), (b, 0)]), fault=fault,
                  label=label)
    if fault:
        fault_and_resume(s, label)
        s.fifo.wait_stamp(st)
    s.submit(E3, _fill_exec(t, b, 0x50, 0x5B, [(b, 0)]), label="an exec without it")
    t.dump(bva, 0x60, "B")
    t.delete()
    s.events("end")
    return s


@scenario
def covgap_svc_kernel_task_deleted():
    """A shared texture (type 16) in task 1, made from a shared texture
    backing (type 17) in task 0 and used once; then task 0 is deleted (32)
    and an exec lists the texture with invalidateGuest. At the exec's end
    the kernel task is asked to mark the backing's guest copy invalid, and
    DeviceCore::get_kernel_task raises "Invalid task". In-process that is
    inside LocalTask::prepare_resources, so the exec faults before its
    command buffer is made, its fill not run. From the service the host's
    side of the task's link catches it and logs
    (DeviceLink::invalidate_shared_texture_backings): the exec completes
    (covgap_svc_kernel_task_deleted_isolate).
    The next exec fills B."""
    doc = covgap_svc_kernel_task_deleted.__doc__
    return [_kernel_deleted("covgap_svc_kernel_task_deleted", doc, IN_PROCESS, True),
            _kernel_deleted("covgap_svc_kernel_task_deleted_isolate", doc, ("isolate",), False)]


# ---- setting the lists: the edges the other list scenarios keep clear of ----

@scenario
def covgap_svc_placement_edges():
    """SetObjectAndPlacementList (68) at the edges of the object-list
    manager's checks, the task root's list fields dumped after each:
    - an object list whose offset + length overflows (2^64 - 12, 36
      bytes), and a placement list whose offset + length overflows: both
      refused in every setup, the fields zeroed;
    - an empty object list with a placement offset and no placement length:
      accepted (no entries, so no placement is required); with argument
      buffers (the service) the offset is stored and no mapping is made;
    - a valid pair after it: the old placement list has an offset and no
      length, so there is nothing to copy, and it is accepted;
    - the placement list's range unmapped (34, which drops its mapping),
      its page made a hole, and the lists moved: in the service the old
      list must be mapped again to be copied from, cannot be (the walk throws,
      and ObjectLists::new_mapping returns none), and the call fails, "Unable
      to do placement list copy", zeroing the task root's four fields but not
      the stored offsets. In-process there is no placement list and it is
      accepted.
    Where a call's outcome differs between setups it is waited for as
    tasks_objlist_placement does (tasks_lib.maybe_fault)."""
    from pgtest.tasks_lib import XTask, maybe_fault, unmap_raw
    s = Scenario("covgap_svc_placement_edges", doc=covgap_svc_placement_edges.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=12)
    t.define()
    op = abi.OP_SET_OBJECT_AND_PLACEMENT_LIST

    def case(label, oo, po, ol, pl):
        maybe_fault(s, op, struct.pack("<IQQII", 1, oo, po, ol, pl), label)
        s.dump(s.pa(t.root, 8), 0x20, label + ": the task root's list fields")

    case("object list offset + length overflows", (1 << 64) - 12, 0, 36, 0)
    case("placement offset + length overflows", PAGE, (1 << 64) - 12, 36, 48)
    case("no entries, a placement offset without a length", PAGE, 4 * PAGE, 0, 0)
    case("valid, after it", PAGE, 5 * PAGE, 36, 48)
    unmap_raw(s, 1, 5 * PAGE, PAGE, label="34 over the placement list")
    punch(t, 5)
    case("moved, the old placement list over a hole", PAGE, 6 * PAGE, 36, 48)
    t.delete()
    s.events("end")
    return s
