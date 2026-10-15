"""The compute and info decoders, the resource manager, textures, ICBs and
the task's object lists at their edges: each scenario locks one behaviour, as
its docstring says; the docstrings say what the code does, which is what the
expected files record."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.compute_objects import Blit as B, Compute as C, Rig, stage_in_blob
from pgtest.exec_objects import DYNAMIC_STRIDE, ObjTask, info_heap_size, rate_map_record
from pgtest.registry import scenario
from pgtest.resources_build import (BC1_RGBA, BGRA8Unorm, OPT_SHARED, R8Unorm, RTask, b_copy_tex_buf, b_fill_buf,
                                    backing_rec, cmd_delete_object, desc1, desc2, dimension, iosurface_op, mip,
                                    ref_texture_rec, shared_backing_rec, shared_texture_rec, simple_dim, texture_rec)
from pgtest.scenario import Scenario
from pgtest.task import Task, buffer_record, rec, seg
from pgtest.tasks_lib import exec3_payload, maybe_fault, set_list_raw

FULL = (4, 4, 1)
BGRA = 0x42475241  # 'BGRA', an IOSurface pixel format


def u32s(*v):
    return struct.pack("<%dI" % len(v), *v)


def _pattern(n, seed):
    return bytes((i * 13 + seed * 29 + (i >> 8)) & 0xFF for i in range(n))


# ---- the compute decoder ----

@scenario
def mutgap2_res_stage_in_region_after_indirect():
    """compute_stage_in's pipeline (attribute 0 from buffer 1, stride 4, per
    thread in x; stage_in_k writes in.v + 0x100) with the two stage-in
    records the other way round: SetStageInRegionIndirect (0xd2) first, whose
    region has size 6, and a dispatch of 8 threads, where threads 6 and 7
    read element 5; then SetStageInRegion (0xd1) with size 8 at the origin
    replaces it, and a second 8-thread dispatch reads elements 0-7."""
    r = Rig("mutgap2_res_stage_in_region_after_indirect", mutgap2_res_stage_in_region_after_indirect.__doc__)
    src, _ = r.buffer(contents=u32s(*range(0x10, 0x30)))
    args, _ = r.buffer(contents=bytes(8) + u32s(0, 0, 0, 6, 1, 1))
    out, out_va = r.buffer()
    f = r.function("shaders/compute_stagein.metallib")
    pso = r.pipeline(blob=stage_in_blob(f))
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0), (src, 0)]), C.stage_in_region_indirect(args, 8),
              C.dispatch_threads((8, 1, 1), (8, 1, 1)),
              C.set_buffer_offset(0, 0x40), C.stage_in_region((8, 1, 1), (0, 0, 0)),
              C.dispatch_threads((8, 1, 1), (8, 1, 1)))
    r.dump(out_va, 0x60, "out: the indirect region, then the direct one")
    return r.end()


@scenario
def mutgap2_res_stage_in_indirect_offset_1():
    """SetStageInRegionIndirect (0xd2) at indirect-buffer offsets 1 and 5:
    the offset must be 4-aligned (offset & 3), so each throws "Invalid
    indirect buffer offset" and its exec faults, nothing of it committed."""
    from scenarios.compute_errors import Checks
    k = Checks("mutgap2_res_stage_in_indirect_offset_1", mutgap2_res_stage_in_indirect_offset_1.__doc__)
    k.fault("offset 1", C.stage_in_region_indirect(k.out, 1))
    k.fault("offset 5", C.stage_in_region_indirect(k.out, 5))
    return k.end()


# ---- the info decoder ----

class InfoRig:
    """A service task with a reply buffer, a one-layer and a two-layer
    rasterization rate map, and 453 targets (isolate: in-process replies
    abort the host, info_reply_aborts_host)."""

    def __init__(self, s, param_len=0x6030):
        t = self.t = ObjTask(s, pages=64)
        self.reply, self.reply_va = t.buffer(PAGE)
        self.rmap = t.ol.add(abi.T_SERIALIZED, rate_map_record(w=256, h=256, samples=[([0.25, 1.0], [0.25, 1.0])]))
        self.rmap2 = t.ol.add(abi.T_SERIALIZED, rate_map_record(
            w=256, h=256, layers=((2, 2), (2, 2)), samples=[([0.25, 1.0], [0.25, 1.0]), ([1.0, 0.25], [1.0, 0.25])]))
        self.exact, self.exact_va = t.buffer(param_len)
        t.define()
        t.install()

    def info(self, records, resources=(), **kw):
        return self.t.exec_segments([seg(abi.ENC_INFO, b"".join(records))], resources=resources, **kw)


@scenario(setups=("isolate",), host=True)
def mutgap2_res_info_rate_map_two_layers():
    """452 GetRasterizationRateMapInfo on a two-layer map (256x256, 2x2
    samples a layer) writes the info header and then one 16-bit 2D size per
    layer, layer 0's and layer 1's physical sizes; the bytes after them keep
    their 0xEE fill. Host-keyed: the sizes are Metal's."""
    s = Scenario("mutgap2_res_info_rate_map_two_layers", setups=("isolate",), host=True,
                 doc=mutgap2_res_info_rate_map_two_layers.__doc__)
    s.bring_up()
    w = InfoRig(s)
    w.t.fill(w.reply_va + 0x400, 0x30, 0xEE)
    w.info([rec(452, struct.pack("<IIQQ", w.rmap2, w.reply, 0x400, 0))], label="452 two layers")
    s.fault_state("452 two layers")
    w.t.dump(w.reply_va + 0x400, 0x30, "452 two layers: header, two layer sizes")
    s.events("end")
    return s


@scenario(setups=("isolate",))
def mutgap2_res_info_456_count_edges():
    """456 MapPhysicalToScreenCoordinatesMultiple reads coordinateCount pairs
    from a copy of the reply cursor and writes each one's mapping in place:
    count 1 at the reply buffer's last 8 bytes reads and writes exactly them
    ((0, 0) maps to (0, 0)) and completes; count 2 of the pairs (0, 0) and
    (0, 0), followed by a 0xEE sentinel pair, maps the two and leaves the
    sentinel; count 0 writes nothing."""
    s = Scenario("mutgap2_res_info_456_count_edges", setups=("isolate",), doc=mutgap2_res_info_456_count_edges.__doc__)
    s.bring_up()
    w = InfoRig(s)
    R = w.reply
    w.info([rec(456, struct.pack("<IIQQII", w.rmap, R, PAGE - 8, 0, 0, 1))], label="456 one pair at the end")
    s.fault_state("456 one pair at the end")
    w.t.dump(w.reply_va + PAGE - 8, 8, "456 one pair at the end")
    w.t.fill(w.reply_va + 0x210, 8, 0xEE)
    w.info([rec(456, struct.pack("<IIQQII", w.rmap, R, 0x200, 0, 0, 2))], label="456 two pairs, a sentinel")
    s.fault_state("456 two pairs, a sentinel")
    w.t.dump(w.reply_va + 0x200, 0x18, "456 two pairs, then the sentinel")
    w.t.fill(w.reply_va + 0x300, 8, 0xEE)
    w.info([rec(456, struct.pack("<IIQQII", w.rmap, R, 0x300, 0, 0, 0))], label="456 no pairs, a sentinel")
    w.t.dump(w.reply_va + 0x300, 8, "456 no pairs: the sentinel")
    s.events("end")
    return s


@scenario(setups=("blocks",), host=True)
def mutgap2_res_info_453_exact_fit():
    """453 CopyRasterizationRateParameterBuffer into a shared buffer whose
    length is exactly the map's parameter size (0x6030 on this host): the
    check is offset + size > length, so the copy at offset 0 fits and
    completes; at offset 0x10 it does not ("Invalid buffer length"), and the
    exec faults. Blocks only: with a memory map (map, isolate) a buffer whose
    length is not a page multiple cannot be made, and its exec faults
    whatever it holds. Host-keyed: the parameter size and data are
    Metal's."""
    s = Scenario("mutgap2_res_info_453_exact_fit", setups=("blocks",), host=True,
                 doc=mutgap2_res_info_453_exact_fit.__doc__)
    s.bring_up()
    w = InfoRig(s)
    res = [(w.exact, 0)]
    w.info([rec(453, struct.pack("<IIQ", w.rmap, w.exact, 0))], resources=res, label="453 exact fit")
    s.fault_state("453 exact fit")
    w.t.dump(w.exact_va, 0x40, "453 exact fit: the first bytes")
    w.t.dump(w.exact_va + 0x6030 - 0x10, 0x10, "453 exact fit: the last bytes")
    w.info([rec(453, struct.pack("<IIQ", w.rmap, w.exact, 0x10))], resources=res, fault=True,
           label="453 at 0x10: 0x10 too long")
    s.fault_state("453 at 0x10")
    s.resume_root()
    s.events("end")
    return s


# ---- placements (service tasks with argument buffers) ----

@scenario(setups=("isolate",))
def mutgap2_res_placement_delete_objects():
    """A deletion (40) of a rasterization rate map (operation 1014), a
    depth-stencil state (1002), a fence (1009), a compute pipeline (1006) and
    a render pipeline (1007) forgets each and zeroes its 16-byte placement
    entry in guest memory, though none was ever made. The entries hold guest
    patterns ({0x72, 0x72}, {0x5555 + i, 0x6666 + i}), installed with opcode
    68; an exec maps the lists. Isolate only (placements)."""
    from pgtest.exec_objects import depth_stencil_record
    from scenarios.textures_placement import install_with_placements
    s = Scenario("mutgap2_res_placement_delete_objects", setups=("isolate",),
                 doc=mutgap2_res_placement_delete_objects.__doc__)
    s.bring_up()
    t = RTask(s)
    from pgtest.compute_objects import compute_blob, compute_pipeline_record, fence_record
    from pgtest import render_lib as R
    rm = t.add(abi.T_SERIALIZED, rate_map_record())
    ds = t.add(abi.T_SERIALIZED, depth_stencil_record())
    fe = t.add(abi.T_SERIALIZED, fence_record())
    cp = t.add(abi.T_SERIALIZED, compute_pipeline_record(compute_blob(0)))
    rp_blob = R.pipeline_blob("basic", 0, 0)
    rp = t.add(abi.T_SERIALIZED, struct.pack("<IIII", 14, 0, 0, len(rp_blob)) + rp_blob)
    b, b_va = t.shared(PAGE)
    t.define()
    pl = install_with_placements(t, {rm: (0x72, 0x72), ds: (0x5555, 0x6666), fe: (0x5556, 0x6667),
                                     cp: (0x5557, 0x6668), rp: (0x5558, 0x6669)})
    t.blit(b_fill_buf(b, 0, 0x10, 1), resources=[(b, 0)], label="an exec: the lists are mapped")
    t.dump(pl, 16 * len(t.ol.entries), "placement list before")
    cmd_delete_object(s, 1, 1014, rm, label="40, 1014 on the rate map")
    t.dump(pl, 16 * len(t.ol.entries), "placement list after 1014")
    cmd_delete_object(s, 1, 1002, ds, label="40, 1002 on the depth-stencil state")
    t.dump(pl, 16 * len(t.ol.entries), "placement list after 1002")
    for op, ref, what in ((1009, fe, "the fence"), (1006, cp, "the compute pipeline"),
                          (1007, rp, "the render pipeline")):
        cmd_delete_object(s, 1, op, ref, label="40, %d on %s" % (op, what))
        t.dump(pl, 16 * len(t.ol.entries), "placement list after %d" % op)
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def mutgap2_res_shared_backing_placed():
    """Shared texture backings (type 17, 4x4 RGBA8Unorm, private) in a
    service task with a placement list, used directly by a copy out in that
    task: made with commandType 48 (v1 descriptor) and 53 (v2), each once
    with a zero entry (made unplaced, no resourceID check) and once with a
    forced entry {0x50 + i, 0x50 + i}, where it is made at that index and its
    gpuResourceID, equal to the index on this host, matches. Every backing
    writes its resourceIndex and gpuResourceID back, and every copy shows the
    guest's texels. A fifth (48) with entry {0x52, 0x1234} and a sixth (53)
    with {0x53, 0x1234} are made at their index, but their gpuResourceID is
    not 0x1234: "Failed to correctly place texture", each exec faults and
    each entry keeps the guest's values. Host-keyed: the indices and IDs are
    the driver's."""
    from scenarios.textures_placement import install_with_placements
    s = Scenario("mutgap2_res_shared_backing_placed", setups=("isolate",), host=True,
                 doc=mutgap2_res_shared_backing_placed.__doc__)
    s.bring_up()
    t = RTask(s, pages=96)
    kbs = []
    for i, (ct, d) in enumerate(((48, desc1(options=0x20, usage=3)), (53, desc2(options=0x20, usage=3)),
                                 (48, desc1(options=0x20, usage=3)), (53, desc2(options=0x20, usage=3)))):
        va = t.area(PAGE, contents=_pattern(64, i))
        kbs.append(t.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, va // PAGE, [simple_dim(4, 4)], d,
                                                                   command_type=ct)))
    wrong = t.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, t.area(PAGE, contents=_pattern(64, 4)) // PAGE,
                                                            [simple_dim(4, 4)], desc1(options=0x20, usage=3)))
    wrong2 = t.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, t.area(PAGE, contents=_pattern(64, 5)) // PAGE,
                                                             [simple_dim(4, 4)], desc2(options=0x20, usage=3),
                                                             command_type=53))
    b, b_va = t.shared(PAGE)
    t.define()
    pl = install_with_placements(t, {kbs[2]: (0x50, 0x50), kbs[3]: (0x51, 0x51), wrong: (0x52, 0x1234),
                                     wrong2: (0x53, 0x1234)})
    labels = ("48, zero entry", "53, zero entry", "48, entry 0x50", "53, entry 0x51")
    for i, (kb, label) in enumerate(zip(kbs, labels)):
        t.blit(b_copy_tex_buf(kb, b, 0x40 * i, 16, 64, FULL), resources=[(kb, 0), (b, 0)], label=label)
    t.blit(b_copy_tex_buf(wrong, b, 0x100, 16, 64, FULL), resources=[(wrong, 0), (b, 0)], fault=True,
           label="48, entry {0x52, 0x1234}")
    t.blit(b_copy_tex_buf(wrong2, b, 0x140, 16, 64, FULL), resources=[(wrong2, 0), (b, 0)], fault=True,
           label="53, entry {0x53, 0x1234}")
    t.dump(b_va, 0x180, "B: each backing's texels, then nothing twice")
    t.dump(pl, 16 * len(t.ol.entries), "placement list")
    t.delete()
    s.events("end")
    return s


# ---- texture records ----

@scenario
def mutgap2_res_texture_record_edges():
    """The texture dimension check at three edges, each object listed in its
    own exec beside a probe buffer: a BC1 8x8 texture whose dimension says
    4x4x1 blocks of 16 bytes (BC1's are 8) is refused at the block comparison
    ("Compressed texture block size mismatch") though its rows and size would
    fit; a 4x4 RGBA8 texture with mips 0 and 1 both ending at 0x40 is refused
    ("Non-increasing mip end"); a 4x4 RGBA8 texture whose one slice of
    exactly one page fills its one-page allocation is made (the check is
    calculated > length); the BC1 texture described with 8x4 blocks is
    refused at the block comparison too; and a two-level BC1 texture whose
    mip 1 ends where mip 0 does (0x20) is refused ("Non-increasing mip end"
    on the compressed path)."""
    from scenarios.resources import creation_case
    from scenarios.textures import _tex_case
    bcd = desc1(fmt=BC1_RGBA, w=8, h=8)
    cases = [
        _tex_case([dimension([mip(0, 64, 32, 8, 8)], block=(4, 4, 1, 16), compressed=1)], bcd),
        _tex_case([dimension([mip(0, 64, 16, 4, 4), mip(0x30, 16, 8, 2, 2)], slice_bytes=0x50)], desc1(mips=2)),
        _tex_case([dimension([mip(0, 64, 16, 4, 4)], slice_bytes=PAGE)], length=PAGE, faults=False),
        _tex_case([dimension([mip(0, 32, 16, 8, 8)], block=(8, 4, 1, 8), compressed=1)], bcd),
        _tex_case([dimension([mip(0, 32, 16, 8, 8), mip(0x18, 8, 8, 4, 4)], block=(4, 4, 1, 8), compressed=1)],
                  desc1(fmt=BC1_RGBA, w=8, h=8, mips=2)),
    ]
    return creation_case("mutgap2_res_texture_record_edges", mutgap2_res_texture_record_edges.__doc__, cases)


@scenario
def mutgap2_res_shared_backing_r8():
    """A shared texture backing (type 17) in task 0 of a 1-byte format,
    R8Unorm 4x4 with tight rows (bytesPerRow 4, equal to the width in
    pixels), over 16 known bytes, read from task 1 through a type 16 texture:
    asking for the handle pages the backing in (whose assert is bytesPerRow
    >= width), and the copy out shows the 16 bytes."""
    from scenarios.textures_shared import define_both, kernel_and_user
    s = Scenario("mutgap2_res_shared_backing_r8", doc=mutgap2_res_shared_backing_r8.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    k_va = k.area(PAGE, contents=bytes(range(0x21, 0x31)))
    kb = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, k_va // PAGE, [simple_dim(4, 4, bpp=1)],
                                                         desc1(fmt=R8Unorm, options=0x20, usage=3)))
    st = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kb))
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    t.blit(b_copy_tex_buf(st, b, 0, 4, 16, FULL), resources=[(st, 0), (b, 0)], label="copy the R8 texture out")
    t.dump(b_va, 0x20, "B")
    t.delete()
    k.delete()
    s.events("end")
    return s


# ---- object lists ----

def _small_task(s, pages=8):
    """A task of `pages` pages with no object list set."""
    return Task(s, tid=1, pages=pages)


@scenario
def mutgap2_res_objlist_refused_then_exec():
    """A refused SetObjectList (51) after an accepted one: the list at page 2
    (36 bytes) is accepted, then a list at page 7 of an 8-page task (ending
    at the task's end) is refused, which zeroes the manager's offset, length
    and count. An ExecIndirect3 (67) with no command buffers then maps the
    object list first. In the blocks setup that maps [offset, length) = [0,
    0), which the range tracker refuses ("Invalid length"): the exec fails
    and faults, with no map_memory call. With a memory map (map, isolate) the
    list mapping is [0, the previous list's page-rounded length), which
    works: the exec completes."""
    s = Scenario("mutgap2_res_objlist_refused_then_exec", doc=mutgap2_res_objlist_refused_then_exec.__doc__)
    s.bring_up()
    t = _small_task(s)
    t.define()
    set_list_raw(s, 1, 2, 36, label="a list at page 2")
    set_list_raw(s, 1, 7, PAGE, fault=True, label="a list at page 7, to the task's end")
    s.fault_state("refused")
    s.resume_root()
    s.events("before the exec")
    maybe_fault(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1), "an empty exec after the refusal")
    s.events("after the exec")
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap2_res_objlist_refused_reply_ref0():
    """After a refused SetObjectList the object count is 0. An info exec
    whose 451 HeapTextureSizeAndAlign names reply buffer reference 0 looks
    the reference up in the object list directly (the writable cursor's
    lookup, which has no reference-0 check), and with count 0 the lookup
    fails ("ObjectRef ref(0) out of range(0)") or the list is not mapped:
    "Failed to get cursor for buffer", the exec faults in every setup, and
    the page a buffer entry at VA 0 would name stays zero. The task goes on
    (a second such exec faults the same way)."""
    s = Scenario("mutgap2_res_objlist_refused_reply_ref0", doc=mutgap2_res_objlist_refused_reply_ref0.__doc__)
    s.bring_up()
    t = _small_task(s)
    # At VA 0 an APVObjectEntry naming a buffer record at VA 2 * PAGE over
    # page 3, as ObjectList.write() packs them (pgtest/task.py).
    t.write(0, abi.OBJ_ENTRY.pack(abi.T_BUFFER | (16 << 8), 2 * PAGE))
    t.write(2 * PAGE, buffer_record(PAGE, 3))
    cb = seg(abi.ENC_INFO, info_heap_size(0, 0, fmt=70, w=16, h=16, usage=1, options=0x20))
    t.write(4 * PAGE, cb)
    t.define()
    set_list_raw(s, 1, 5, 12, label="a list at page 5")
    set_list_raw(s, 1, 7, PAGE, fault=True, label="a list at page 7, to the task's end")
    s.fault_state("refused")
    s.resume_root()
    for i in range(2):
        label = "451 into reply reference 0, %d" % i
        s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [(4 * PAGE, len(cb))]), fault=True, label=label)
        s.fault_state(label)
        s.resume_root()
    t.dump(3 * PAGE, 0x20, "page 3: nothing written")
    t.delete()
    s.events("end")
    return s


# ---- the kernel task's object list and cached objects ----

@scenario(setups=("blocks", "map"))
def mutgap2_res_kernel_list_cached_backing():
    """The backing lookup maps the kernel task's object list before it looks
    at anything, cached or not. A type 5 texture over an IOSurface backing is
    used while task 0's list is in place, so the backing is made and cached;
    task 0's list is then moved onto a hole in its page table (51 only
    records it), and a second type 5 texture over the same backing cannot be
    made ("Failed to map object list"): its exec faults. With the list moved
    back it works. In-process only: in the service the backing lookup
    aborts."""
    from scenarios.tasks_gaps import kernel_and_user
    s = Scenario("mutgap2_res_kernel_list_cached_backing", setups=("blocks", "map"),
                 doc=mutgap2_res_kernel_list_cached_backing.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s, kholes=(30, 31))
    px = bytes((i * 5) & 0xFF for i in range(64))
    bk_va = k.area(PAGE, contents=bytes(0x100) + px)
    bk = k.add(abi.T_IOSURFACE_BACKING, backing_rec(PAGE, bk_va // PAGE, BGRA, [(0x100, 4, 4, 16, 4)]))
    op = iosurface_op(12, desc1(fmt=BGRA8Unorm, options=OPT_SHARED))
    r = t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=op))
    r2 = t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=op))
    b, b_va = t.shared(PAGE)
    k.define()
    k.install()
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(r, b, 0, 16, 64, FULL), resources=[(r, 0), (b, 0)], label="the first texture")
    set_list_raw(s, 0, 30, k.ol.length())
    t.blit(b_copy_tex_buf(r2, b, 0x40, 16, 64, FULL), resources=[(r2, 0), (b, 0)], fault=True,
           label="a second texture while task 0's list is over a hole")
    set_list_raw(s, 0, 0, k.ol.length())
    t.blit(b_copy_tex_buf(r2, b, 0x80, 16, 64, FULL), resources=[(r2, 0), (b, 0)],
           label="the second texture with task 0's list back")
    t.dump(b_va, 0xC0, "B: the backing's pixels, nothing, the pixels")
    t.delete()
    k.delete()
    s.events("end")
    return s


@scenario
def mutgap2_res_kernel_list_cached_shared_backing():
    """The shared texture handle lookup maps the kernel task's object list
    before it looks at anything, cached or not. A type 16 texture over a
    shared backing (type 17) is used while task 0's list is in place, so the
    backing is made and cached; task 0's list is then moved onto a hole in
    its page table, and a second type 16 texture over the same backing cannot
    get its handle ("Failed to map object list"; through the host a nil
    handle for the service): its exec faults. With the list moved back it
    works."""
    from scenarios.tasks_gaps import kernel_and_user
    s = Scenario("mutgap2_res_kernel_list_cached_shared_backing",
                 doc=mutgap2_res_kernel_list_cached_shared_backing.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s, kholes=(30, 31))
    px = bytes((0x30 + i) & 0xFF for i in range(64))
    bk_va = k.area(PAGE, contents=px[::-1])
    kept = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, bk_va // PAGE, [simple_dim(4, 4)],
                                                           desc1(options=0x20, usage=3)))
    s1 = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kept))
    s2 = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kept))
    b, b_va = t.shared(PAGE)
    k.define()
    k.install()
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(s1, b, 0, 16, 64, FULL), resources=[(s1, 0), (b, 0)], label="the first shared texture")
    set_list_raw(s, 0, 30, k.ol.length())
    t.blit(b_copy_tex_buf(s2, b, 0x40, 16, 64, FULL), resources=[(s2, 0), (b, 0)], fault=True,
           label="a second shared texture while task 0's list is over a hole")
    set_list_raw(s, 0, 0, k.ol.length())
    t.blit(b_copy_tex_buf(s2, b, 0x80, 16, 64, FULL), resources=[(s2, 0), (b, 0)],
           label="the second shared texture with task 0's list back")
    t.dump(b_va, 0xC0, "B: the backing's pixels, nothing, the pixels")
    t.delete()
    k.delete()
    s.events("end")
    return s


# ---- ICBs ----

@scenario
def mutgap2_res_icb_prepare_fails_again():
    """An ICB whose prepare fails stays host-invalid (the error is only
    logged and the host copy is not marked valid), so every exec that lists
    it prepares it again from guest memory. Its command 0 dispatches into out
    at 0x40 and its command 1 has type 1, which prepare refuses: executed,
    only command 0's output shows. The guest then rewrites command 1 as a
    dispatch into out at 0x80, and the next exec (resource flags 0) prepares
    both: both outputs show."""
    from pgtest.icb_cmds import CMD, compute_cmd
    from scenarios.icb import ICBTask
    s = Scenario("mutgap2_res_icb_prepare_fails_again", doc=mutgap2_res_icb_prepare_fails_again.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    ok_c = lambda off: compute_cmd(pso=w.cpso, binds=[(O, 0, off)], grid=(2, 1, 1), tg=(2, 1, 1))
    icb = w.icb([ok_c(0x40), compute_cmd(ctype=1, pso=w.cpso)])
    w.start()
    w.clear_out()
    w.run_compute(icb, 0, 2, label="command 1 of type 1")
    w.out_dump("command 1 of type 1", 0x100)
    va, _ = w.icbs[icb]
    w.t.write(va + CMD, ok_c(0x80))
    w.clear_out()
    w.run_compute(icb, 0, 2, label="command 1 rewritten, flags 0")
    w.out_dump("command 1 rewritten", 0x100)
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def mutgap2_res_icb_render_two_strides():
    """The ICB write-back (icb_sync's rig: an exec lists the ICB with
    invalidateGuest, the guest copy is overwritten with 0xEE, 53 writes it
    back) of a dynamic-stride render ICB with two vertex binds and strides 16
    and 32: the render write-back writes every vertex bind and its stride, so
    both binds (reference, GPU address, offset) and both strides are written
    back. GPU addresses are the host's: keyed by the host. Isolate only:
    in-process the write-back aborts (icb_sync_in_process)."""
    from pgtest.icb_cmds import render_cmd
    from scenarios.icb import ICBTask, sync_refs
    s = Scenario("mutgap2_res_icb_render_two_strides", setups=("isolate",), host=True,
                 doc=mutgap2_res_icb_render_two_strides.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    r2 = w.icb([render_cmd(pso=w.rpso, vbinds=[(O, 0, 0x40), (w.tb, 0, 0x50)], strides=[16, 32])], kind="render",
               flags=DYNAMIC_STRIDE, vbufs=2)
    w.start()
    w.t.exec_segments([seg(abi.ENC_BLIT)], resources=w.res() + [(r2, abi.RES_INVALIDATE_GUEST)],
                      label="prepared, guest invalidated")
    s.fault_state("prepared")
    va, size = w.icbs[r2]
    w.t.fill(va, size, 0xEE)
    sync_refs(s, w.t.tid, [r2], label="synchronize")
    w.t.dump(va, size, "written back")
    s.events("end")
    return s


@scenario(setups=("map", "isolate"), iosfc=True)
def mutgap2_res_icb_mapper_bind_by_address():
    """A compute ICB whose kernel bind 0 names no reference (~0) and GPU
    address 0x40, listed after a mapper-ref buffer (type 15, over mapped
    surface 1) in the exec that runs it. A bind by address is resolved
    through the task's registered GPU addresses (the highest registered
    address at or below it). A service task registers each buffer's GPU
    address when it makes it, and a mapper buffer's before its prepare makes
    its Metal buffer: as address 0. An earlier exec lists the mapper buffer,
    so it is made and prepared (its Metal buffer exists), and the pass that
    executes the ICB makes it resident (0x87). In isolate address 0x40 is
    then the mapper buffer at offset 0x40: the command is prepared, and
    double_gid writes 0, 2, 4, 6 into the surface's page at 0x40. In-process
    (map) nothing is registered: the bind fails, prepare stops before the
    command (its error only logged), and the page keeps its pattern. The
    IOSurface host device needs a memory map: map and isolate."""
    from pgtest.icb_cmds import compute_cmd
    from pgtest.resources_build import mapper_ref_rec
    from scenarios.icb import ICBTask
    from scenarios.resources_mapper import _surfaces
    s = Scenario("mutgap2_res_icb_mapper_bind_by_address", setups=("map", "isolate"), iosfc=True,
                 doc=mutgap2_res_icb_mapper_bind_by_address.__doc__)
    s.bring_up()
    d, pages = _surfaces(s, {1: (64, 16, 4, b"BGRA", None)})
    w = ICBTask(s)
    m = w.t.ol.add(abi.T_MAPPER_BUFFER, mapper_ref_rec(1))
    icb = w.icb([compute_cmd(pso=w.cpso, binds=[(0xFFFFFFFF, 0x40, 0)], grid=(4, 1, 1), tg=(4, 1, 1))])
    w.start()
    w.t.exec_segments([seg(abi.ENC_BLIT, b_fill_buf(w.out, 0, 4, 1))], resources=[(m, 0), (w.out, 0)],
                      label="an exec listing the mapper buffer: it is made and prepared")
    w.run_compute(icb, 0, 1, res=[(m, 0)] + w.res(icb), extra=C.use_resources([m]),
                  label="the ICB, a bind by address 0x40")
    s.dump(s.pa(pages[1]), 0x60, "surface 1's page")
    s.events("end")
    return s


# ---- mapper textures (the IOSurface host device) ----

@scenario(setups=("map", "isolate"), iosfc=True)
def mutgap2_res_mapper_texture_row_17():
    """A one-component 8-bit IOSurface (L008, 16x4, one byte an element) with
    rows of 17 bytes, and a mapper-ref texture (type 11, R8Unorm 16x4,
    shader-read) over it: validateIOSurfaceDescriptor needs the surface's row
    bytes aligned to the device's IOSurface texture alignment, which 17 is
    not ("Invalid row byte alignment"), so creation fails and the exec that
    lists it faults; the probe slot stays zero. A texture over a surface with
    rows of 16 bytes is made and its exec runs. The IOSurface host device
    needs a memory map: map and isolate."""
    from pgtest.resources_build import mapper_ref_rec
    from scenarios.resources_mapper import _surfaces
    s = Scenario("mutgap2_res_mapper_texture_row_17", setups=("map", "isolate"), iosfc=True,
                 doc=mutgap2_res_mapper_texture_row_17.__doc__)
    s.bring_up()
    d, pages = _surfaces(s, {4: (16, 4, 1, b"L008", 17), 5: (16, 4, 1, b"L008", 16)})
    t = RTask(s)
    d8 = desc1(fmt=R8Unorm, w=16, h=4, options=OPT_SHARED, usage=1)
    bad = t.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(4) + iosurface_op(12, d8))
    good = t.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(5) + iosurface_op(12, d8))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_buf(b, 0, 0x10, 1), resources=[(bad, 0), (b, 0)], fault=True, label="rows of 17 bytes")
    t.blit(b_fill_buf(b, 0x10, 0x10, 2), resources=[(good, 0), (b, 0)], label="rows of 16 bytes")
    t.dump(b_va, 0x20, "probe buffer: which execs ran")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("map", "isolate"), iosfc=True)
def mutgap2_res_mapper_texture_failed_then_unmap():
    """A mapper-ref texture (type 11) over mapped surface 2 (64x16 BGRA)
    whose descriptor is taller than the surface fails validation after the
    surface was retained, and the failure releases it (in the service through
    the host): its exec faults. The IOSurface host device then unmaps the
    surface: nothing else holds it, so the ring entry's status is success.
    Map and isolate (the IOSurface host device needs a memory map)."""
    from pgtest.iosfc_guest import CMD_UNMAP
    from pgtest.resources_build import mapper_ref_rec
    from scenarios.resources_mapper import _bgra, _surfaces
    s = Scenario("mutgap2_res_mapper_texture_failed_then_unmap", setups=("map", "isolate"), iosfc=True,
                 doc=mutgap2_res_mapper_texture_failed_then_unmap.__doc__)
    s.bring_up()
    d, pages = _surfaces(s, {2: (64, 16, 4, b"BGRA", None)})
    t = RTask(s)
    x = t.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(2) + iosurface_op(12, _bgra(h=32)))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_buf(b, 0, 0x10, 1), resources=[(x, 0), (b, 0)], fault=True, label="a texture taller than the surface")
    idx = d.submit((CMD_UNMAP, 2))
    d.status(idx, "unmap after the failed texture")
    t.dump(b_va, 0x10, "probe buffer")
    t.delete()
    s.events("end")
    return s


# ---- task memory: MapMemory2 and UnmapMemory at odd offsets ----

@scenario
def mutgap2_res_map_offset_4001():
    """MapMemory2 (57) at offset 0x4001, length one page, on a defined task
    not yet mapped. In the blocks setup the map refuses a non-page-aligned
    offset before it walks the page table, so it faults and the task's
    page-table iterator is untouched; with a memory map MapMemory2 does
    nothing and completes. A second task with pages of its own is then
    defined, and mapping task 1's first two pages walks task 1's table: the
    host's map_memory calls name task 1's pages. Both tasks are then
    deleted."""
    from pgtest.tasks_lib import XTask
    s = Scenario("mutgap2_res_map_offset_4001", doc=mutgap2_res_map_offset_4001.__doc__)
    s.bring_up()
    t1 = Task(s, tid=1, pages=8)
    t1.define()
    s.events("task 1 defined")
    maybe_fault(s, abi.OP_MAP_MEMORY2, struct.pack("<IQQ", 1, 0x4001, PAGE), "map at 0x4001")
    s.events("after the misaligned map")
    t2 = XTask(s, tid=2, pages=8, scatter=True)
    t2.define()
    t1.map(0, 2 * PAGE)
    s.events("task 2 defined, task 1 mapped")
    t2.delete()
    t1.delete()
    s.events("end")
    return s


@scenario
def mutgap2_res_unmap_odd():
    """UnmapMemory (34) at offset 1, length one page, and at offset 0, length
    one page and a byte, on a mapped task. In the blocks setup the range
    tracker's alignment checks are the only ones: "Invalid offset", "Invalid
    length", each command faults before the host's unmap_memory is called.
    With a memory map there is no tracker and no unmap call: both
    complete."""
    s = Scenario("mutgap2_res_unmap_odd", doc=mutgap2_res_unmap_odd.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=8)
    t.define()
    t.map()
    s.events("defined and mapped")
    maybe_fault(s, abi.OP_UNMAP_MEMORY, struct.pack("<IQQ", 1, 1, PAGE), "unmap at offset 1")
    s.events("after the unmap at offset 1")
    maybe_fault(s, abi.OP_UNMAP_MEMORY, struct.pack("<IQQ", 1, 0, PAGE + 1), "unmap of a page and a byte")
    s.events("after the unmap of a page and a byte")
    t.unmap(0, PAGE)
    s.events("unmap of page 0")
    t.delete()
    s.events("end")
    return s


# ---- the service's exec completion ----

@scenario(setups=("isolate",))
def mutgap2_res_exec_error_with_sync():
    """In the service, an exec whose command buffer ends in error while its
    synchronize buffer completes: a fill of a shared buffer the exec does not
    list fails on this host's GPU ("Insufficient Memory",
    exec_empty_resource_list_isolate), and a second shared buffer listed with
    the synchronize flag makes the exec wait for a sync buffer. The token
    completes with 3 unless both buffers completed: the exec faults. A later
    exec that lists the buffer fills it."""
    s = Scenario("mutgap2_res_exec_error_with_sync", setups=("isolate",), doc=mutgap2_res_exec_error_with_sync.__doc__)
    s.bring_up()
    t = RTask(s)
    b, b_va = t.shared(PAGE)
    b2, b2_va = t.shared(PAGE, fill=0x33)
    t.define()
    t.install()
    t.blit(b_fill_buf(b, 0, 0x10, 0x5A), resources=[(b2, abi.RES_SYNCHRONIZE)], fault=True,
           label="an unlisted fill, a synchronize resource")
    t.blit(b_fill_buf(b, 0x10, 0x10, 0x5B), resources=[(b, 0)], label="listed")
    t.dump(b_va, 0x20, "B")
    t.dump(b2_va, 0x10, "B2")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",))
def mutgap2_res_texture_last_page():
    """A normal texture (type 2, 4x4 RGBA8Unorm) whose guest page is the last
    page of a 16-page task (VA page 15). Paging it in makes a transfer buffer
    over [15 pages, 16 pages), whose mapping refuses only a range ending past
    the task's end: the copy out shows the texels. Isolate only (the
    service's mapping; mutgap_svc_texture_past_end is the page after it)."""
    from scenarios.mutgap_service import TEXELS, past_end_task
    s = Scenario("mutgap2_res_texture_last_page", setups=("isolate",), doc=mutgap2_res_texture_last_page.__doc__)
    s.bring_up()
    t = past_end_task(s)
    t.write(15 * PAGE, TEXELS)
    x = t.add(abi.T_TEXTURE, texture_rec(PAGE, 15, [simple_dim(4, 4)], desc1()))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(x, b, 0, 16, 64, FULL), resources=[(x, 0), (b, 0)], label="copy the texture out")
    t.dump(b_va, 0x40, "B")
    t.delete()
    s.events("end")
    return s


# ---- resource lists in the compute decoder ----

@scenario(setups=("isolate",))
def mutgap2_res_resource_lists_placement():
    """Over-long UseResources (0x87: count 2, one ref), UseHeaps (0x86: count
    2, one ref) and MemoryBarrierWithResources (0xd6: count 2, one ref) throw
    "Too many resources specified" before they look any ref up, so the
    objects they name, two textures and a heap no exec lists, are never made:
    their placement entries, zero, stay zero (a made texture or heap writes
    its index or GPU address into a zero entry, textures_placement_written).
    Isolate only (placements)."""
    from pgtest.resources_build import heap_rec, texture_rec as tr
    from scenarios.textures_placement import install_with_placements
    s = Scenario("mutgap2_res_resource_lists_placement", setups=("isolate",),
                 doc=mutgap2_res_resource_lists_placement.__doc__)
    s.bring_up()
    t = RTask(s, pages=96)
    tex = t.add(abi.T_TEXTURE, tr(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc1()))
    h = t.add(abi.T_HEAP, heap_rec(t.area(4 * PAGE) // PAGE, 4 * PAGE, 0))
    tex2 = t.add(abi.T_TEXTURE, tr(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc1()))
    b, b_va = t.shared(PAGE)
    t.define()
    pl = install_with_placements(t, {})
    for label, r in (("0x87 count 2, one texture", C.use_resources([tex], count=2)),
                     ("0x86 count 2, one heap", C.use_heaps([h], count=2)),
                     ("0xd6 count 2, one texture", C.barrier_resources([tex2], count=2))):
        t.reset_cmdbufs()
        cb = t.cmdbuf(seg(abi.ENC_COMPUTE, r))
        t.exec3([cb], resources=[(b, 0)], fault=True, label=label)
        s.fault_state(label)
        s.resume_root()
    t.dump(pl, 16 * len(t.ol.entries), "placement list")
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap2_res_use_lists_257():
    """UseResources (0x87) and UseHeaps (0x86) at the stack list's edge: 256
    refs fill the 256-entry stack list and 257 take the heap-allocated one.
    All succeed, and the dispatch after them writes its output; a later exec
    runs."""
    r = Rig("mutgap2_res_use_lists_257", mutgap2_res_use_lists_257.__doc__)
    out, out_va = r.buffer()
    heap = r.heap(listed=True)
    pso = r.kernel()
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0)]), C.use_resources([out] * 256),
              C.use_resources([out] * 257), C.use_heaps([heap] * 256), C.use_heaps([heap] * 257),
              C.dispatch_threads((8, 1, 1), (8, 1, 1)))
    r.dump(out_va, 0x24, "out")
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0x40)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)))
    r.dump(out_va + 0x40, 0x14, "out at 0x40: a later dispatch")
    return r.end()


@scenario
def mutgap2_res_concurrent_barrier_resources():
    """compute_serial_dispatches' two dispatches in a Concurrent pass
    (SetComputePassDispatchType 1 first), ordered by a
    MemoryBarrierWithResources (0xd6) naming the buffer the first writes and
    the second reads: the second sees the doubled values, and B holds the
    same bytes as with a scope barrier. Run twice."""
    from scenarios.compute import ADD
    r = Rig("mutgap2_res_concurrent_barrier_resources", mutgap2_res_concurrent_barrier_resources.__doc__)
    a, a_va = r.buffer()
    b, b_va = r.buffer()
    dbl = r.kernel()
    add = r.kernel(ADD)
    r.ready()
    for i in range(2):
        r.run(seg(abi.ENC_BLIT, B.fill(a, 0, 0x80, 0) + B.fill(b, 0, 0x80, 0)))
        r.compute(C.dispatch_type(1) + C.set_pipeline(dbl) + C.set_buffers(0, [(a, 0)])
                  + C.dispatch_threads((32, 1, 1), (32, 1, 1)) + C.barrier_resources([a])
                  + C.set_pipeline(add) + C.set_buffers(0, [(b, 0), (a, 0)]) + C.dispatch_threads((32, 1, 1), (32, 1, 1)))
        r.dump(b_va, 0x84, "B, pass %d" % i)
    return r.end()


@scenario
def mutgap2_res_compute_first_overlong_after_unread():
    """A compute segment whose first record, a SetComputePassDispatchType
    (0xdb), says its length runs past the segment, after an earlier compute
    segment of the same exec whose SetBufferOffset (0xcf) left 16 bytes
    unread: the pass-descriptor peek fails to make its sub-cursor and throws
    "Failed to read": the exec faults, nothing of it committed, and the task
    keeps working."""
    from scenarios.compute_errors import Checks
    k = Checks("mutgap2_res_compute_first_overlong_after_unread",
               mutgap2_res_compute_first_overlong_after_unread.__doc__)
    first = seg(abi.ENC_COMPUTE, k.prefix() + rec(0xCF, struct.pack("<IQ", 0, 0) + bytes(16)))
    second = seg(abi.ENC_COMPUTE, rec(0xDB, u32s(0), length=0x100))
    k.r.run(first, second, fault=True, label="0xdb past the segment after 16 unread bytes")
    k.r.compute(k.prefix(), label="a later dispatch")
    k.r.dump(k.out_va, 0x14, "out: only the later dispatch")
    return k.r.end()


@scenario(setups=("isolate",))
def mutgap2_res_info_overlong_after_unread():
    """An info segment of two 461 BufferHostInfo records: the first names the
    reply buffer's own host info and carries 0x20 bytes it never reads; the
    second says its length runs past the segment. The first reply is written;
    the second record's sub-cursor fails and the decoder throws "Failed to
    read": the exec faults, and a later info exec still writes its reply.
    Isolate only (replies abort in-process). Only whether each reply was
    written is dumped (its first byte against a fill)."""
    from pgtest.exec_objects import info_ref
    s = Scenario("mutgap2_res_info_overlong_after_unread", setups=("isolate",),
                 doc=mutgap2_res_info_overlong_after_unread.__doc__)
    s.bring_up()
    w = InfoRig(s)
    R = w.reply
    w.t.fill(w.reply_va, 0x40, 0xEE)
    first = rec(461, struct.pack("<IIQ", w.exact, R, 0x10) + bytes(0x20))
    second = rec(461, struct.pack("<IIQ", w.exact, R, 0x20), length=0x100)
    w.info([first, second], fault=True, label="461 with 0x20 unread bytes, then one past the segment")
    s.fault_state("past the segment")
    s.resume_root()
    w.info([info_ref(461, w.exact, R, 0x30)], label="a later 461")
    w.t.dump(w.reply_va + 0x20, 8, "the second record's reply slot: not written")
    s.events("end")
    return s


# ---- ICB layouts ----

@scenario
def mutgap2_res_icb_barrier_at_command_end():
    """A compute ICB whose layout puts the barrier at 0xFC, so its 4 bytes
    end exactly at commandSize (0x100): prepare's positioned read is refused
    only past the command's end, so the command is prepared and dispatches
    double_gid into out."""
    from pgtest.icb_cmds import compute_cmd, layout
    from scenarios.icb import ICBTask
    s = Scenario("mutgap2_res_icb_barrier_at_command_end", doc=mutgap2_res_icb_barrier_at_command_end.__doc__)
    s.bring_up()
    w = ICBTask(s)
    icb = w.icb([compute_cmd(pso=w.cpso, binds=[(w.out, 0, 0)], grid=(8, 1, 1), tg=(8, 1, 1))],
                lay=layout("compute", barrierOffset=0xFC))
    w.start()
    w.clear_out()
    w.run_compute(icb, 0, 1, label="barrier at 0xfc")
    w.out_dump("out", 0x24)
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def mutgap2_res_icb_heap_buffer_bind():
    """The ICB write-back (icb_sync's rig) of a compute ICB whose kernel bind
    0 is a buffer placed in a heap (type 14 at offset 0x100 of a type 13
    heap) at offset 0x10: in a service task a heap's cover buffer is
    registered by GPU address when the heap is made, so the bind's write-back
    finds the heap and writes the bind back as the heap's reference and the
    bind's offset within it (0x110). GPU addresses are the host's: keyed by
    the host."""
    from pgtest.icb_cmds import compute_cmd
    from pgtest.resources_build import heap_buffer_rec, heap_rec
    from scenarios.icb import ICBTask, sync_refs
    s = Scenario("mutgap2_res_icb_heap_buffer_bind", setups=("isolate",), host=True,
                 doc=mutgap2_res_icb_heap_buffer_bind.__doc__)
    s.bring_up()
    w = ICBTask(s)
    h = w.t.ol.add(abi.T_HEAP, heap_rec(w.t.reserve(4 * PAGE) // PAGE, 4 * PAGE, 1))
    hb = w.t.ol.add(abi.T_HEAP_BUFFER, heap_buffer_rec(h, 0x100, PAGE))
    icb = w.icb([compute_cmd(pso=w.cpso, binds=[(hb, 0, 0x10)], grid=(4, 1, 1), tg=(4, 1, 1))])
    w.start()
    w.t.exec_segments([seg(abi.ENC_BLIT)], resources=w.res() + [(h, 0), (hb, 0), (icb, abi.RES_INVALIDATE_GUEST)],
                      label="prepared, guest invalidated")
    s.fault_state("prepared")
    va, size = w.icbs[icb]
    w.t.fill(va, size, 0xEE)
    sync_refs(s, w.t.tid, [icb], label="synchronize")
    w.t.dump(va, 0x40, "written back")
    s.events("end")
    return s


# ---- backings and heaps in-process ----

@scenario(setups=("blocks", "map"))
def mutgap2_res_backing_four_planes():
    """An IOSurface backing (type 4) in task 0 whose record says four planes
    and carries all four: the backing's creation refuses more than three
    before it reads them ("Too many planes (4) in backing resource"), so a
    type 5 texture over it in task 1 cannot be made and its exec faults; the
    task goes on. In-process only (type 5 textures abort the service)."""
    from scenarios.textures_shared import define_both, kernel_and_user
    s = Scenario("mutgap2_res_backing_four_planes", setups=("blocks", "map"),
                 doc=mutgap2_res_backing_four_planes.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    va = k.area(PAGE, contents=_pattern(PAGE, 5))
    bk = k.add(abi.T_IOSURFACE_BACKING, backing_rec(PAGE, va // PAGE, BGRA, [(0, 4, 4, 16, 4)] * 4))
    r = t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=iosurface_op(12, desc1(fmt=BGRA8Unorm,
                                                                                   options=OPT_SHARED))))
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    t.blit(b_fill_buf(b, 0, 0x10, 1), resources=[(r, 0), (b, 0)], fault=True, label="a texture over four planes")
    t.blit(b_fill_buf(b, 0x10, 0x10, 2), resources=[(b, 0)], label="the task goes on")
    t.dump(b_va, 0x20, "probe buffer")
    t.delete()
    k.delete()
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def mutgap2_res_legacy_heap_sync_valid():
    """In-process a heap is a legacy heap, made with its guest copy valid. A
    fill of a buffer placed in it (listed with flags 0) stays on the host; 53
    then finds the guest copy valid and copies nothing back (needsSync): the
    heap's pages keep their pattern. A copy out of the heap buffer shows the
    fill."""
    from pgtest.resources_build import b_copy_buf_buf, cmd_synchronize, heap_buffer_rec, heap_rec
    s = Scenario("mutgap2_res_legacy_heap_sync_valid", setups=("blocks", "map"),
                 doc=mutgap2_res_legacy_heap_sync_valid.__doc__)
    s.bring_up()
    t = RTask(s, pages=80)
    h_va = t.area(4 * PAGE, contents=_pattern(0x100, 6))
    h = t.add(abi.T_HEAP, heap_rec(h_va // PAGE, 4 * PAGE, 1))
    hb = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(h, 0, PAGE))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_buf(hb, 0, 0x10, 0xA1), resources=[(h, 0), (hb, 0), (b, 0)], label="fill the heap buffer")
    cmd_synchronize(s, 1, [h], label="53 on the heap")
    t.dump(h_va, 0x20, "the heap's pages")
    t.blit(b_copy_buf_buf(hb, b, 0, 0, 0x20), resources=[(h, 0), (hb, 0), (b, 0)], label="copy the heap buffer out")
    t.dump(b_va, 0x20, "B")
    t.delete()
    s.events("end")
    return s


# ---- the object and placement list's alignment check ----

@scenario
def mutgap2_res_objlist_misaligned_by_one():
    """SetObjectAndPlacementList (68)'s alignment test checks the offsets the
    previous accepted call stored, not the new ones (as
    tasks_objlist_placement shows): an object list at PAGE + 1 is accepted
    and stored, so the next call, a valid one, is refused for it and zeroes
    the offsets, and the call after that is accepted. The same for a
    placement list at 6 pages + 1, whose offset is stored only with argument
    buffers (the service): there the call after it is refused, in-process
    accepted. Each result shows in the task root."""
    from scenarios.tasks_objlist import root_fields
    from pgtest.tasks_lib import XTask
    s = Scenario("mutgap2_res_objlist_misaligned_by_one", doc=mutgap2_res_objlist_misaligned_by_one.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=12)
    t.define()
    root_fields(s, t, "defined")
    for label, (oo, po, ol, pl) in (("valid", (PAGE, 6 * PAGE, 36, 48)),
                                    ("misaligned by one byte", (PAGE + 1, 6 * PAGE, 36, 48)),
                                    ("valid, after the one-byte misalignment", (PAGE, 6 * PAGE, 36, 48)),
                                    ("valid, after that", (PAGE, 6 * PAGE, 36, 48)),
                                    ("placement list misaligned by one byte", (PAGE, 6 * PAGE + 1, 36, 48)),
                                    ("valid, after the placement misalignment", (2 * PAGE, 7 * PAGE, 36, 48)),
                                    ("valid, once more", (2 * PAGE, 7 * PAGE, 36, 48))):
        maybe_fault(s, abi.OP_SET_OBJECT_AND_PLACEMENT_LIST, struct.pack("<IQQII", 1, oo, po, ol, pl), label)
        root_fields(s, t, label)
    t.delete()
    s.events("end")
    return s


# ---- forced placements, used ----

@scenario(setups=("isolate",), host=True)
def mutgap2_res_placement_forced_used():
    """Textures with forced placements {0x40 + i, 0x40 + i}
    (textures_placement_forced's entries) used for real, not only named by
    314: a memoryless depth texture (commandType 1) as a render pass's depth
    attachment (load Clear, store DontCare) beside a colour target cleared to
    green, textures over a buffer (commandTypes 9 and 55) copied out, and
    textures in a heap (21 and 56) filled from a buffer and copied out. Each
    is made at its index and its gpuResourceID matches, so the pass clears
    the target and each copy shows the buffer's bytes. Host-keyed: the IDs
    are the driver's."""
    from pgtest import render_lib as R
    from pgtest.resources_build import Depth32Float, ser_memoryless, ser_on_buffer
    from scenarios.textures_placement import install_with_placements
    s = Scenario("mutgap2_res_placement_forced_used", setups=("isolate",), host=True,
                 doc=mutgap2_res_placement_forced_used.__doc__)
    s.bring_up()
    t = R.RenderTask(s, pages=96)
    target, _, target_va = t.target(4, 4)
    ml = t.ol.add(abi.T_SERIALIZER_TEXTURE, ser_memoryless(desc1(fmt=Depth32Float, options=0x30, usage=4)))
    sh, sh_va = t.buffer(PAGE, contents=_pattern(0x200, 7))
    v1 = t.ol.add(abi.T_SERIALIZER_TEXTURE, ser_on_buffer(sh, 0, 16, desc1(options=OPT_SHARED)))
    v2 = t.ol.add(abi.T_SERIALIZER_TEXTURE, ser_on_buffer(sh, 0x100, 16, desc2(options=OPT_SHARED), v2=True))
    from pgtest.resources_build import b_copy_buf_tex, heap_rec, ser_in_heap
    h_va = t.reserve(4 * PAGE)
    h = t.ol.add(abi.T_HEAP, heap_rec(h_va // PAGE, 4 * PAGE, 1))
    ht1 = t.ol.add(abi.T_SERIALIZER_TEXTURE, ser_in_heap(h, 0, desc1(usage=3)))
    ht2 = t.ol.add(abi.T_SERIALIZER_TEXTURE, ser_in_heap(h, PAGE, desc2(usage=3), v2=True))
    b, b_va = t.buffer(PAGE)
    t.define()
    pl = install_with_placements(t, {ml: (0x40, 0x40), v1: (0x41, 0x41), v2: (0x42, 0x42), ht1: (0x43, 0x43),
                                     ht2: (0x44, 0x44)})
    t.fill(target_va, 0x40, 0x11)
    t.render([R.describe_pass({0: dict(tex=target, clear=(0.0, 1.0, 0.0, 1.0))}, width=4, height=4,
                              depth=dict(tex=ml, load=R.LOAD_CLEAR, store=R.STORE_DONTCARE, clear=1.0))],
             label="a pass with the memoryless depth texture")
    s.fault_state("the pass")
    t.dump(target_va, 0x40, "the target")
    for i, v in enumerate((v1, v2)):
        t.exec_segments([seg(abi.ENC_BLIT, b_copy_tex_buf(v, b, 0x40 * i, 16, 64, FULL))],
                        label="copy out of the texture over a buffer, %d" % i)
        s.fault_state("copy %d" % i)
    for i, ht in enumerate((ht1, ht2)):
        t.exec_segments([seg(abi.ENC_BLIT, b_copy_buf_tex(sh, ht, 0x80 + 0x40 * i, 16, 64, FULL)
                             + b_copy_tex_buf(ht, b, 0x80 + 0x40 * i, 16, 64, FULL))],
                        resources=[h] + t.bufs, label="through the texture in the heap, %d" % i)
        s.fault_state("heap %d" % i)
    t.dump(b_va, 0x100, "B: the four textures' bytes")
    t.dump(pl, 16 * len(t.ol.entries), "placement list")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def mutgap2_res_ratemap_reset_list_unmappable():
    """ResetRasterizationRateMap (66) in-process maps the task's object lists
    before it looks the map up. A rate map is made by a draw; the task's
    object list is then moved onto a hole in its page table (51 only records
    it), and 66 on the map throws "Failed to map object list": the command
    faults and the root FIFO resumes. (With the list mapped, 66 on an
    existing map kills the host in-process: render_ratemap_reset_death.)"""
    from pgtest import render_lib as R
    from scenarios.render_ratemap import HALF, H, W, RateTask, reset, run
    s = Scenario("mutgap2_res_ratemap_reset_list_unmappable", setups=("blocks", "map"),
                 doc=mutgap2_res_ratemap_reset_list_unmappable.__doc__)
    s.bring_up()

    # RateTask's objects (scenarios/render_ratemap.py:22-41) in a task whose
    # VA page 60 is a hole.
    rt = RateTask.__new__(RateTask)
    t = rt.t = R.RenderTask(s, holes=(60,))
    rt.p = t.pipeline("basic", v=t.function("shaders/render_vbuf.metallib"),
                      f=t.function("shaders/render_color.metallib"))
    rt.tex, _, rt.va = t.target(W, H)
    rt.pos, _ = t.buffer(contents=R.float4s(*R.FULL))
    rt.col, _ = t.buffer(contents=R.float4s((1.0, 0.0, 0.0, 1.0)))
    rt.buf, _ = t.buffer()
    m = rt.t.ratemap(W, H, [HALF], mutable=1)
    rt.start()
    run(s, rt, "create the map", m)
    set_list_raw(s, rt.t.tid, 60, rt.t.ol.length(), label="the list onto a hole")
    reset(s, rt, m, "66 with the list over a hole", fault=True)
    s.events("end")
    return s


@scenario
def mutgap2_res_compute_overlong_after_fence():
    """In one compute segment, an UpdateFence (0xd4) whose record carries 4
    bytes its 4-byte command never reads, then, as the segment's last bytes,
    a record whose length runs past the segment: its sub-cursor fails and the
    decoder throws "Failed to read" before any command sees it: the exec
    faults, nothing of it committed, and the task keeps working."""
    from scenarios.compute_errors import Checks
    k = Checks("mutgap2_res_compute_overlong_after_fence", mutgap2_res_compute_overlong_after_fence.__doc__)
    k.fault("0xd4 with 4 unread bytes, then a record past the segment",
            rec(0xD4, struct.pack("<II", k.fence, 0)), rec(0xD4, b"", length=0x40))
    k.r.compute(k.prefix(), label="a later dispatch")
    k.r.dump(k.out_va, 0x14, "out: only the later dispatch")
    return k.r.end()


@scenario(setups=("isolate",))
def mutgap2_res_heap_pinned_huge():
    """A private heap and a private buffer of 2^36 bytes in a service task
    whose placement entries pin their GPU addresses (a nonzero first word):
    the temporary allocation succeeds and Metal then makes no heap ("Failed
    to create placed heap") and no buffer, so each exec that lists one
    faults; an address is written back only when the entry's is zero, and
    both entries keep the guest's pins. Isolate only (detach resources and
    placements are the service's)."""
    from pgtest.resources_build import heap_rec
    from scenarios.textures_placement import PINNED, install_with_placements
    s = Scenario("mutgap2_res_heap_pinned_huge", setups=("isolate",), doc=mutgap2_res_heap_pinned_huge.__doc__)
    s.bring_up()
    t = RTask(s)
    va = t.area(PAGE)
    h = t.add(abi.T_HEAP, heap_rec(va // PAGE, 1 << 36, 1))
    hb = t.add(abi.T_BUFFER, buffer_record(1 << 36, va // PAGE, 1))
    b, b_va = t.shared(PAGE)
    t.define()
    pl = install_with_placements(t, {h: (PINNED + 0x200000, 0), hb: (PINNED + 0x400000, 0)})
    t.blit(b_fill_buf(b, 0, 0x10, 1), resources=[(h, 0), (b, 0)], fault=True, label="the heap of 2^36 bytes")
    t.blit(b_fill_buf(b, 0, 0x10, 2), resources=[(hb, 0), (b, 0)], fault=True, label="the buffer of 2^36 bytes")
    t.dump(pl, 16 * len(t.ol.entries), "placement list")
    t.dump(b_va, 0x10, "probe")
    t.delete()
    s.events("end")
    return s


# ---- the compute decoder ----

def _two_level_texture(r):
    """A normal 4x4 RGBA8 texture of two levels, level 0 red and level 1
    green (as mutgap_dec_render_tile_sampler_lod lays it out), listed in
    every exec of the Rig; returns its reference."""
    va = r.t.reserve(PAGE)
    r.t.write(va, bytes.fromhex("ff0000ff") * 16 + bytes.fromhex("00ff00ff") * 4)
    ref = r.t.ol.add(abi.T_TEXTURE, texture_rec(PAGE, va // PAGE, [simple_dim(4, 4, levels=2)],
                                                desc1(w=4, h=4, mips=2, usage=1)))
    r.buffers.append(ref)
    return ref


@scenario
def mutgap2_res_compute_sampler_lod_clamps():
    """0xcd SetSamplerStatesLODClamp
    (setSamplerStates:lodMinClamps:lodMaxClamps:withRange:) with a mipmapped
    sampler over a two-level texture (level 0 red, level 1 green). sample_k
    samples at level 0 (fixtures/shaders/compute_sample.metal): clamps (1, 1)
    give green, (0, 1000) red. sample_l1 samples at level 1
    (fixtures/shaders/mutres2_sample_l1.metal): clamps (0, 0) give red, (0,
    1000) green."""
    r = Rig("mutgap2_res_compute_sampler_lod_clamps", mutgap2_res_compute_sampler_lod_clamps.__doc__)
    tex = _two_level_texture(r)
    out, out_va = r.buffer()
    smp = r.sampler(mip_filter=1)
    p0 = r.kernel("shaders/compute_sample.metallib")
    p1 = r.kernel("shaders/mutres2_sample_l1.metallib")
    r.ready()
    for i, (p, lo, hi) in enumerate(((p0, 1.0, 1.0), (p0, 0.0, 1000.0), (p1, 0.0, 0.0), (p1, 0.0, 1000.0))):
        r.compute(C.set_pipeline(p), C.set_textures(0, [tex]), C.set_samplers_lod(0, [(smp, lo, hi)]),
                  C.set_buffers(0, [(out, 0x10 * i)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)))
    r.dump(out_va, 0x40, "out: level 0 clamped to (1, 1), (0, 1000); level 1 clamped to (0, 0), (0, 1000)")
    return r.end()


@scenario
def mutgap2_res_stage_in_region_twice():
    """compute_stage_in's pipeline with two SetStageInRegion (0xd1) records
    in one pass: size 8 for the first 8-thread dispatch, then size 6 for the
    second, whose threads 6 and 7 then read element 5, as in
    compute_stage_in's indirect pass."""
    r = Rig("mutgap2_res_stage_in_region_twice", mutgap2_res_stage_in_region_twice.__doc__)
    src, _ = r.buffer(contents=u32s(*range(0x10, 0x30)))
    out, out_va = r.buffer()
    f = r.function("shaders/compute_stagein.metallib")
    pso = r.pipeline(blob=stage_in_blob(f))
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0), (src, 0)]), C.stage_in_region((8, 1, 1), (0, 0, 0)),
              C.dispatch_threads((8, 1, 1), (8, 1, 1)),
              C.set_buffer_offset(0, 0x40), C.stage_in_region((6, 1, 1), (0, 0, 0)),
              C.dispatch_threads((8, 1, 1), (8, 1, 1)))
    r.dump(out_va, 0x60, "out: region size 8, then 6")
    return r.end()


@scenario
def mutgap2_res_compute_icb_ref0_empty_range():
    """ExecuteCommandsInBufferRanged (0xe4) with ICB reference 0 and the
    empty range {0, 0}: the lookup gives nil, and the decoder raises "Failed
    to get indirect command buffer" before the range check that an empty
    range would pass: the exec faults, and the task keeps working."""
    from scenarios.compute_errors import Checks
    k = Checks("mutgap2_res_compute_icb_ref0_empty_range", mutgap2_res_compute_icb_ref0_empty_range.__doc__)
    k.fault("0xe4 ICB 0, range {0, 0}", C.execute_icb(0, 0, 0))
    k.r.compute(k.prefix(), label="a later dispatch")
    k.r.dump(k.out_va, 0x14, "out: only the later dispatch")
    return k.r.end()


# ---- the range tracker's run blocks (blocks setup) ----

def _tracker_task(s):
    """A 1024-page task every leaf of which is one RAM page (as
    tasks_map_tracker_blocks builds it), defined, not mapped."""
    from pgtest.tasks_lib import XTask
    alias = s.alloc_page()
    t = XTask(s, tid=1, pages=1024, leaves={v: alias for v in range(1024)})
    t.define()
    return t


@scenario(setups=("blocks",))
def mutgap2_res_tracker_split_adjacent():
    """240 single-page maps of pages 0-239, each run adjacent to the last and
    kept as its own run (RunBlock::add), fill the first run block (240 runs).
    A map of page 5, inside it, splits the block into two of 120 runs; the
    halves abut, which the split's assert allows (<= kCmpLessAdjacent), and
    the map completes (page 5 is already recorded: no host map_memory call).
    Blocks only: the tracker is used only without a memory map."""
    s = Scenario("mutgap2_res_tracker_split_adjacent", setups=("blocks",),
                 doc=mutgap2_res_tracker_split_adjacent.__doc__)
    s.bring_up()
    t = _tracker_task(s)
    for p in range(240):
        t.map(p * PAGE, PAGE)
    s.events("240 adjacent runs", blocks_digest=True)
    t.map(5 * PAGE, PAGE, label="page 5 again: a split")
    t.map(300 * PAGE, PAGE, label="page 300")
    s.events("after the split")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("blocks",))
def mutgap2_res_tracker_split_odd_death():
    """A host death (docs/testing.md, section 7.4). A run block holds 239
    single-page runs (every other page from 0) and one run of three pages
    (480-482): 240 runs. An unmap of page 481 splits that run in two inside
    the block (remove_from_range, RunBlock::remove), so the block holds 241,
    one more than the 240 at which add splits a block first. The next map
    into the block splits it: split keeps oldRunCount - half runs instead of
    half, 121 + 121 != 241, and its assert aborts the host (blocks: the
    tracker is used only without a memory map)."""
    from pgtest.exec_objects import submit_last
    s = Scenario("mutgap2_res_tracker_split_odd_death", setups=("blocks",),
                 doc=mutgap2_res_tracker_split_odd_death.__doc__)
    s.bring_up()
    t = _tracker_task(s)
    for p in range(0, 478, 2):
        t.map(p * PAGE, PAGE)
    t.map(480 * PAGE, 3 * PAGE)
    s.events("239 single-page runs and one of three pages", blocks_digest=True)
    t.unmap(481 * PAGE, PAGE, label="unmap page 481: 241 runs")
    s.events("after the unmap")
    submit_last(s, abi.OP_MAP_MEMORY2, struct.pack("<IQQ", 1, 1 * PAGE, PAGE), label="map page 1: the split")
    return s


# ---- shared buffers (in-process) ----

@scenario(setups=("blocks", "map"))
def mutgap2_res_shared_buffer_discarded_bind():
    """In-process a discarded shared buffer drops its Metal buffer, and
    Resource::get_buffer on it raises: the buffer was invalidated. A compute
    pass that only binds it (0xcb, whose getter is the nullable one) with
    the buffer left out of the exec's list reaches that raise: the exec
    faults. Listed again, it is re-created and a dispatch writes it.
    In-process only (the service's buffers are detach buffers)."""
    from pgtest.resources_build import cmd_discard
    r = Rig("mutgap2_res_shared_buffer_discarded_bind", mutgap2_res_shared_buffer_discarded_bind.__doc__,
            setups=("blocks", "map"))
    a, a_va = r.buffer(listed=False)
    out, out_va = r.buffer()
    pso = r.kernel()
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(a, 0)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)),
              resources=r.resources([(a, 0)]), label="a dispatch into A")
    cmd_discard(r.s, 1, [a], label="63 on A")
    r.compute(C.set_buffers(0, [(a, 0)]), fault=True, label="A bound after the discard, not listed")
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(a, 0x20)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)),
              resources=r.resources([(a, 0)]), label="A listed again")
    r.dump(a_va, 0x30, "A")
    return r.end()


# ---- the page-table walk's last root entry ----

@scenario
def mutgap2_res_page_table_last_root_entry():
    """A depth-1 task of 4096 pages (the most depth 1 covers) whose buffer
    lies on VA page 4095: the walk's root read starts at entry 4095, a
    table's last (the walk refuses only a start past 4095), so the buffer's
    page is found and a fill lands."""
    from pgtest.tasks_lib import XGPUTask, fill_rec
    s = Scenario("mutgap2_res_page_table_last_root_entry", doc=mutgap2_res_page_table_last_root_entry.__doc__)
    s.bring_up()
    t = XGPUTask(s, tid=1, pages=4096, depth=1, mapped=list(range(8)) + [4095])
    ref, va = t.buffer(length=PAGE, va=4095 * PAGE)
    t.define()
    t.install()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(ref, 0, 0x20, 0x5A)))
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(ref, 0)]), label="fill the buffer on page 4095")
    s.fault_state("fill")
    t.dump(4095 * PAGE, 0x30, "page 4095")
    t.delete()
    s.events("end")
    return s


# ---- the ICB write-back at a buffer's end ----

@scenario(setups=("isolate",), host=True)
def mutgap2_res_icb_bind_at_buffer_end():
    """The ICB write-back (icb_sync's rig) of a compute ICB whose kernel bind
    0 names a shared buffer P by reference at offset 0x4000, its length. P's
    GPU address is pinned by its placement entry, far from every other
    buffer, so nothing is registered at P's end. The bind's write-back
    resolves the bind's address by the registered buffer at or below it and
    refuses an offset not below that buffer's length ("Buffer too small"):
    the bind is written back as reference ~0, the address, offset 0.
    Host-keyed (the rest of the ICB's write-back)."""
    from pgtest.icb_cmds import compute_cmd
    from scenarios.icb import ICBTask, sync_refs
    from scenarios.textures_placement import PINNED, install_with_placements
    s = Scenario("mutgap2_res_icb_bind_at_buffer_end", setups=("isolate",), host=True,
                 doc=mutgap2_res_icb_bind_at_buffer_end.__doc__)
    s.bring_up()
    w = ICBTask(s)
    P, _ = w.t.buffer(PAGE)
    icb = w.icb([compute_cmd(pso=w.cpso, binds=[(P, 0, PAGE)], grid=(4, 1, 1), tg=(4, 1, 1))])
    w.t.define()
    install_with_placements(w.t, {P: (PINNED + 0x300000, 0)})
    w.t.exec_segments([seg(abi.ENC_BLIT)], resources=w.res() + [(P, 0), (icb, abi.RES_INVALIDATE_GUEST)],
                      label="prepared, guest invalidated")
    s.fault_state("prepared")
    va, size = w.icbs[icb]
    w.t.fill(va, size, 0xEE)
    sync_refs(s, w.t.tid, [icb], label="synchronize")
    w.t.dump(va + 0x10, 0x14, "written back: bind 0")
    s.events("end")
    return s


# ---- an ICB at a forced placement ----

@scenario(setups=("isolate",), host=True)
def mutgap2_res_icb_placement_forced():
    """An ICB whose placement entry forces resourceIndex 0x70 with
    gpuResourceID 0x70 (for textures the ID equals the index on this host,
    textures_placement_forced): it is made at that index and the ID check
    passes, so executing it dispatches double_gid into out. Host-keyed: the
    IDs are the driver's."""
    from pgtest.icb_cmds import compute_cmd
    from scenarios.icb import ICBTask
    from scenarios.textures_placement import install_with_placements
    s = Scenario("mutgap2_res_icb_placement_forced", setups=("isolate",), host=True,
                 doc=mutgap2_res_icb_placement_forced.__doc__)
    s.bring_up()
    w = ICBTask(s)
    icb = w.icb([compute_cmd(pso=w.cpso, binds=[(w.out, 0, 0)], grid=(8, 1, 1), tg=(8, 1, 1))])
    w.t.define()
    pl = install_with_placements(w.t, {icb: (0x70, 0x70)})
    w.clear_out()
    w.run_compute(icb, 0, 1, label="the ICB at index 0x70")
    w.out_dump("out", 0x24)
    w.t.dump(pl + 16 * icb, 16, "the ICB's placement entry")
    s.events("end")
    return s


# ---- an untracked heap below version 204 ----

@scenario(setups=("isolate",))
def mutgap2_res_untracked_heap_v203():
    """Below binary version 204 the service refuses tracked heaps
    (supportsHeapHazardTracking), but an untracked private heap is made: its
    exec completes, while a tracked one's faults. Isolate only (detach heaps
    are the service's)."""
    from scenarios.resources import creation_case
    from pgtest.resources_build import heap_rec
    return creation_case("mutgap2_res_untracked_heap_v203", mutgap2_res_untracked_heap_v203.__doc__,
                         [(abi.T_HEAP, heap_rec(20, 4 * PAGE, 1), None, False),
                          (abi.T_HEAP, heap_rec(24, 4 * PAGE, 3), None, True)],
                         setups=("isolate",), version=203)


# ---- object lists over read-only leaves, and placement lists at 0 ----

@scenario(setups=("map", "isolate"))
def mutgap2_res_objlist_readonly_leaf():
    """An object list on a page whose leaf lacks the writable bit: with a
    memory map (map, isolate) the list is mapped read-only, so the buffer it
    names is found and a fill lands. (Blocks maps it through the legacy
    manager, which ignores the flag.)"""
    from pgtest.tasks_lib import XGPUTask, fill_rec
    s = Scenario("mutgap2_res_objlist_readonly_leaf", setups=("map", "isolate"),
                 doc=mutgap2_res_objlist_readonly_leaf.__doc__)
    s.bring_up()
    t = XGPUTask(s, tid=1, pages=16, writable=set(range(1, 16)))
    ref, va = t.buffer(length=PAGE, va=8 * PAGE)
    t.define()
    t.install()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(ref, 0, 0x10, 0x6C)))
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(ref, 0)]), label="a fill, the list read-only")
    s.fault_state("fill")
    t.dump(8 * PAGE, 0x20, "the buffer")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",))
def mutgap2_res_placement_list_at_zero():
    """With argument buffers (the service), 68 with a placement list at
    offset 0 and a length is accepted (offset 0 skips the size check and the
    mapping); a second 68 that moves it to page 10 then finds an old list
    with a length that was never mapped, "Unable to do placement list copy":
    it faults and zeroes the task root's four list fields. Isolate only
    (placement offsets are stored only with argument buffers)."""
    from scenarios.tasks_objlist import root_fields
    from pgtest.tasks_lib import XGPUTask
    s = Scenario("mutgap2_res_placement_list_at_zero", setups=("isolate",),
                 doc=mutgap2_res_placement_list_at_zero.__doc__)
    s.bring_up()
    t = XGPUTask(s, tid=1, pages=12)
    t.buffer(length=PAGE, va=8 * PAGE)
    t.define()
    n = t.ol.write()
    for label, po in (("68: a placement list at offset 0", 0), ("68: the list moved to page 10", 10 * PAGE)):
        maybe_fault(s, abi.OP_SET_OBJECT_AND_PLACEMENT_LIST, struct.pack("<IQQII", 1, 0, po, n, 32), label)
        root_fields(s, t, label)
    t.delete()
    s.events("end")
    return s


# ---- mapper textures with rotation and a zero placement entry ----

@scenario(setups=("isolate",), host=True, iosfc=True)
def mutgap2_res_mapper_rotation_zero_entry():
    """Mapper-ref textures made with rotation (commandTypes 47 and 57) in a
    service task with placements, their entries zero: made unforced, with no
    resourceID check, so each copy out completes. Host-keyed as
    resources_mapper_placement."""
    from pgtest.resources_build import mapper_ref_rec
    from scenarios.resources_mapper import _bgra, _surfaces
    from scenarios.textures_placement import install_with_placements
    s = Scenario("mutgap2_res_mapper_rotation_zero_entry", setups=("isolate",), host=True, iosfc=True,
                 doc=mutgap2_res_mapper_rotation_zero_entry.__doc__)
    s.bring_up()
    _surfaces(s, {2: (64, 16, 4, b"BGRA", None)})
    t = RTask(s)
    m47 = t.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(2) + iosurface_op(47, _bgra()))
    m57 = t.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(2) + iosurface_op(
        57, desc2(fmt=BGRA8Unorm, w=64, h=16, options=OPT_SHARED, usage=3)))
    b, b_va = t.shared(PAGE)
    t.define()
    pl = install_with_placements(t, {})
    for i, r in enumerate((m47, m57)):
        t.blit(b_copy_tex_buf(r, b, 0x100 + 0x10 * i, 256, 1024, (1, 1, 1)), b_fill_buf(b, 0x10 * i, 4, i + 1),
               resources=[(b, 0)], label="object %d" % r)
    t.dump(b_va, 0x120, "probe buffer")
    t.dump(pl, 16 * len(t.ol.entries), "placement list")
    t.delete()
    s.events("end")
    return s


# ---- a deletion 11 bytes long (in-process) ----

@scenario(setups=("blocks", "map"))
def mutgap2_res_delete_object_11_bytes():
    """DeleteObject (40) whose payload is 15 bytes: taskID, operation 1000,
    length 11, and three bytes of a buffer's reference. The FIFO passes the
    task the payload from the operation on, 11 bytes, and the resource
    manager refuses a deletion under 12 bytes, deleting nothing; the task
    reports success either way. The buffer was made by an exec, and its
    object-list entry was then rewritten to an invalid type, which a made
    object never re-reads: a copy out of it afterwards still completes with
    its bytes. A 12-byte deletion of an unused reference first leaves byte 15
    of the FIFO's payload zero. In-process only (the service copies exactly
    the 11 bytes, and the reference's fourth byte would be whatever its heap
    holds)."""
    from pgtest.resources_build import b_copy_buf_buf
    s = Scenario("mutgap2_res_delete_object_11_bytes", setups=("blocks", "map"),
                 doc=mutgap2_res_delete_object_11_bytes.__doc__)
    s.bring_up()
    t = RTask(s)
    a, a_va = t.shared(PAGE, contents=_pattern(0x40, 8))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_buf_buf(a, b, 0, 0, 0x20), resources=[(a, 0), (b, 0)], label="A is made")
    typ, n, va = t.ol.entries[a]
    t.ol.entries[a] = (0x7F, n, va)
    t.ol.write()
    cmd_delete_object(s, 1, 1000, 50, label="40 on an unused reference, 12 bytes")
    s.submit(abi.OP_DELETE_OBJECT, struct.pack("<III", 1, 1000, 11) + struct.pack("<I", a)[:3],
             label="40 on A, 11 bytes")
    t.blit(b_copy_buf_buf(a, b, 0x10, 0x20, 0x20), resources=[(a, 0), (b, 0)], label="A copied out again")
    t.dump(b_va, 0x40, "B")
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap2_res_icb_command_size_0x30():
    """A compute ICB whose layout makes each command 0x30 bytes, the size of
    the concurrentDispatchThreads arguments (two MTLSize), read at offset 0:
    that positioned read spans the whole command sub-cursor, which the
    cursor's size check allows (only a length past the cursor is refused),
    and the dispatch is encoded. Every field overlaps: grid (64,1,1) and
    threadgroup (64,1,1) at 0, the command type is grid.x's low word (0x40),
    the barrier its high word (0). The ICB inherits its pipeline and buffers
    from the encoder (0xd0 and 0xcb), so no bind or pipeline field is read.
    The kernel dispatch arguments that follow (0x38 bytes at 0) do not fit
    the command, so prepare then fails, which only logs: the dispatch already
    encoded still runs, writing double_gid over 64 threads into out."""
    from pgtest.exec_objects import INH_BUFFERS, INH_PIPELINE, c_buffers, c_pipeline
    from pgtest.icb_cmds import layout
    from scenarios.icb import ICBTask
    s = Scenario("mutgap2_res_icb_command_size_0x30", doc=mutgap2_res_icb_command_size_0x30.__doc__)
    s.bring_up()
    w = ICBTask(s)
    cmd = struct.pack("<6Q", 0x40, 1, 1, 0x40, 1, 1)
    icb = w.icb([cmd], flags=INH_PIPELINE | INH_BUFFERS,
                lay=layout("compute", commandSize=0x30, commandTypeOffset=0, barrierOffset=4,
                           commandArgumentsOffset=0, kernelDispatchArgumentsOffset=0))
    w.start()
    w.clear_out()
    w.run_compute(icb, 0, 1, extra=c_pipeline(w.cpso) + c_buffers(0, [(w.out, 0)]), label="0x30-byte commands")
    w.out_dump("out", 0x104)
    s.events("end")
    return s


@scenario(setups=("isolate",))
def mutgap2_res_placement_list_stale_after_refusal():
    """SetObjectAndPlacementList (68) in a service task (placement lists are
    kept only with argument buffers): a placement list at 6 pages + 1 is
    stored though its mapping fails ("Unable to do placement list copy"), so
    the next call, valid with its placement list at 7 pages, is refused for
    the stored misaligned offset and the refusal zeroes the stored offsets. A
    fourth call with the placement list at 8 pages is then accepted with
    nothing to copy: the old list is gone, so the 48 bytes at 8 pages stay as
    the guest wrote them, and the list at 7 pages is untouched."""
    from scenarios.tasks_objlist import root_fields
    from pgtest.tasks_lib import XTask
    s = Scenario("mutgap2_res_placement_list_stale_after_refusal", setups=("isolate",),
                 doc=mutgap2_res_placement_list_stale_after_refusal.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=12)
    s.mem(t.gpa(7 * PAGE), _pattern(0x30, 11))
    s.mem(t.gpa(8 * PAGE), _pattern(0x30, 12))
    t.define()
    for label, (oo, po, ol, pl) in (("valid", (PAGE, 6 * PAGE, 36, 48)),
                                    ("placement list misaligned by one byte", (PAGE, 6 * PAGE + 1, 36, 48)),
                                    ("valid, placement list at 7 pages", (2 * PAGE, 7 * PAGE, 36, 48)),
                                    ("valid, placement list at 8 pages", (2 * PAGE, 8 * PAGE, 36, 48))):
        maybe_fault(s, abi.OP_SET_OBJECT_AND_PLACEMENT_LIST, struct.pack("<IQQII", 1, oo, po, ol, pl), label)
        root_fields(s, t, label)
    s.dump(t.gpa(7 * PAGE), 0x30, "placement list at 7 pages")
    s.dump(t.gpa(8 * PAGE), 0x30, "placement list at 8 pages")
    t.delete()
    s.events("end")
    return s
