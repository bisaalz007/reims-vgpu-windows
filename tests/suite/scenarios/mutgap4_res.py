"""The resource manager (samplers, textures, pipelines, ICBs), the
single-plane and serializer textures and the mapper-ref buffer at their
edges: each scenario locks one behaviour, as its docstring says; the
docstrings say what the code does, which is what the expected files
record."""
import struct

from pgtest import abi
from pgtest import render_lib as R
from pgtest.abi import PAGE
from pgtest.compute_objects import Compute as C, Rig
from pgtest.exec_objects import ICB_DISPATCH, ICB_DRAW_INDEXED_PATCHES
from pgtest.icb_cmds import compute_cmd, draw_indexed_patches_args, render_cmd, tess_args
from pgtest.registry import scenario
from pgtest.resources_build import (BC1_RGBA, RGBA8Unorm, TT_2D, TT_CubeArray, RTask, b_copy_buf_buf,
                                    b_copy_tex_buf, b_fill_buf, b_fill_tex_color, cmd_delete_object, cmd_discard,
                                    cmd_invalidate, desc1, desc2, dimension, mapper_ref_rec, mip, ser_memoryless,
                                    ser_view_type, simple_dim, texture_rec)
from pgtest.scenario import Scenario

FULL = (4, 4, 1)
S8 = abi.T_SERIALIZER_TEXTURE


# ---- samplers ----

@scenario
def mutgap4_res_sampler_descriptor_fields():
    """Three sampler descriptor fields reach Metal, each shown by a pair of
    samplers that differ in it alone, over the two-level 4x4 texture of
    mutgap2_res (level 0 red, level 1 green): lodMaxClamp: sample_l1
    (fixtures/shaders/mutres2_sample_l1.metal) samples at level 1, which a
    mipmapped (Nearest) sampler with lodMaxClamp 0 holds at level 0 (red),
    and one with 1000 does not (green). borderColor: sample_k
    (fixtures/shaders/compute_sample.metal) samples left of the texture,
    which address mode ClampToBorderColor (5) answers with the border colour:
    OpaqueWhite (2) reads ffffffff, OpaqueBlack (1) ff000000. maxAnisotropy:
    sample_grad (fixtures/shaders/mutres4_sample_grad.metal) samples with
    gradients 16 times longer in x than in y; a Linear, mipmapped sampler
    with maxAnisotropy 16 samples level 0 (red), the same sampler with 1
    level 1 (green)."""
    from scenarios.mutgap2_res import _two_level_texture
    r = Rig("mutgap4_res_sampler_descriptor_fields", mutgap4_res_sampler_descriptor_fields.__doc__)
    tex = _two_level_texture(r)
    out, out_va = r.buffer()
    lod0 = r.sampler(mip_filter=1, lod_max=0.0)
    lod1000 = r.sampler(mip_filter=1)
    white = r.sampler(s_mode=5, t_mode=5, border=2)
    black = r.sampler(s_mode=5, t_mode=5, border=1)
    aniso16 = r.sampler(min_filter=1, mag_filter=1, mip_filter=2, max_aniso=16)
    aniso1 = r.sampler(min_filter=1, mag_filter=1, mip_filter=2, max_aniso=1)
    p_l1 = r.kernel("shaders/mutres2_sample_l1.metallib")
    p_k = r.kernel("shaders/compute_sample.metallib")
    p_g = r.kernel("shaders/mutres4_sample_grad.metallib")
    r.ready()
    recs = []
    for i, (p, smp) in enumerate(((p_l1, lod0), (p_l1, lod1000), (p_k, white), (p_k, black), (p_g, aniso16),
                                  (p_g, aniso1))):
        recs += [C.set_pipeline(p), C.set_textures(0, [tex]), C.set_samplers(0, [smp]),
                 C.set_buffers(0, [(out, 0x10 * i)]), C.dispatch_threads((4, 1, 1), (4, 1, 1))]
    r.compute(*recs)
    r.dump(out_va, 0x60, "out: lodMaxClamp 0, 1000; border white, black; maxAnisotropy 16, 1")
    return r.end()


# ---- texture descriptors and placement ----

@scenario
def mutgap4_res_v2_swizzle_alpha_channels():
    """A v2 descriptor's swizzle of 5 (MTLTextureSwizzleAlpha) in the green
    and in the blue channel reaches Metal (a channel below 6 is Metal's; the
    counterpart of mutgap3_res_v2_swizzles_sampled's red): 4x1 RGBA8Unorm
    textures (type 2, commandType 52) over the texels {R, G, B, A} = {11 22
    33 80}, {44 55 66 40}, {77 88 99 c0}, {aa bb cc 10}, sampled by sample_k
    (Repeat: thread i reads texel i) with swizzle (2, 5, 4, 5), which reads
    alpha as green, and (2, 3, 5, 5), alpha as blue."""
    r = Rig("mutgap4_res_v2_swizzle_alpha_channels", mutgap4_res_v2_swizzle_alpha_channels.__doc__)
    t = r.t
    texels = struct.pack("<4I", 0x80332211, 0x40665544, 0xC0998877, 0x10CCBBAA)
    texs = []
    for sw in ((2, 5, 4, 5), (2, 3, 5, 5)):
        va = t.reserve(PAGE)
        t.write(va, texels)
        ref = t.ol.add(abi.T_TEXTURE, texture_rec(PAGE, va // PAGE, [simple_dim(4, 1)],
                                                  desc2(w=4, h=1, usage=1, swizzle=sw), command_type=52))
        r.buffers.append(ref)
        texs.append(ref)
    out, out_va = r.buffer()
    smp = r.sampler(s_mode=2, t_mode=2)
    pso = r.kernel("shaders/compute_sample.metallib")
    r.ready()
    recs = [C.set_pipeline(pso), C.set_samplers(0, [smp])]
    for i, tex in enumerate(texs):
        recs += [C.set_textures(0, [tex]), C.set_buffers(0, [(out, 0x10 * i)]),
                 C.dispatch_threads((4, 1, 1), (4, 1, 1))]
    r.compute(*recs)
    r.dump(out_va, 0x20, "samples: (2,5,4,5), (2,3,5,5)")
    return r.end()


@scenario(setups=("isolate",), host=True)
def mutgap4_res_v2_placement_reserved_index():
    """mutgap_svc_placement_reserved_index with the texture made from a v2
    descriptor (commandType 52): its entry {1, 1} names its place when the
    service first maps the lists, so the rebuild reserves index 1; X1, a v1
    texture with a zero entry, is made first and gets another index, and X0
    is then made at index 1 by force (v2's setResourceIndex: and
    setForceResourceIndex:YES) and its copy lands in B. Both entries and B
    are dumped (measured: X1 is given 2). Host-keyed (Metal's indices)."""
    from scenarios.textures_placement import install_with_placements
    s = Scenario("mutgap4_res_v2_placement_reserved_index", setups=("isolate",), host=True,
                 doc=mutgap4_res_v2_placement_reserved_index.__doc__)
    s.bring_up()
    t = RTask(s, pages=32)
    x0_va = t.area(PAGE, contents=bytes((i * 7 + 3) & 0xFF for i in range(0x40)))
    x0 = t.add(abi.T_TEXTURE, texture_rec(PAGE, x0_va // PAGE, [simple_dim(4, 4)], desc2(usage=0x13),
                                          command_type=52))
    x1 = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc1(usage=0x13)))
    b, b_va = t.shared(PAGE)
    t.define()
    pl = install_with_placements(t, {x0: (1, 1)})
    t.blit(b_copy_tex_buf(x1, b, 0, 16, 64, FULL), resources=[(x1, 0), (b, 0)], label="make X1")
    t.dump(pl + 16 * x0, 32, "X0's and X1's entries after X1")
    t.blit(b_copy_tex_buf(x0, b, 0x40, 16, 64, FULL), resources=[(x0, 0), (b, 0)],
           label="make X0 (v2) at its forced index")
    t.dump(pl + 16 * x0, 32, "X0's and X1's entries after X0")
    t.dump(b_va + 0x40, 0x40, "B+0x40: X0's texels")
    t.delete()
    s.events("end")
    return s



@scenario(setups=("isolate",), host=True)
def mutgap4_res_v2_placement_index_taken():
    """A v2 texture (commandType 52) whose placement entry, written after the
    lists were first mapped (so nothing reserved it), forces the index a live
    texture already holds: X1 (v1, zero entry) is made first and is given
    index 1; the guest then writes X0's entry {1, 1}, and X0 is made by a
    copy out of it. X0's and X1's entries and B are dumped. Host-keyed
    (Metal's indices)."""
    from scenarios.textures_placement import install_with_placements
    s = Scenario("mutgap4_res_v2_placement_index_taken", setups=("isolate",), host=True,
                 doc=mutgap4_res_v2_placement_index_taken.__doc__)
    s.bring_up()
    t = RTask(s, pages=32)
    x0_va = t.area(PAGE, contents=bytes((i * 7 + 3) & 0xFF for i in range(0x40)))
    x0 = t.add(abi.T_TEXTURE, texture_rec(PAGE, x0_va // PAGE, [simple_dim(4, 4)], desc2(usage=0x13),
                                          command_type=52))
    x1 = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc1(usage=0x13)))
    b, b_va = t.shared(PAGE)
    t.define()
    pl = install_with_placements(t, {})
    t.blit(b_copy_tex_buf(x1, b, 0, 16, 64, FULL), resources=[(x1, 0), (b, 0)], label="make X1")
    t.dump(pl + 16 * x0, 32, "X0's and X1's entries after X1")
    t.write(pl + 16 * x0, abi.PLACEMENT.pack(1, 1))
    from pgtest.task import seg
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, b_copy_tex_buf(x0, b, 0x40, 16, 64, FULL)))
    st = t.exec3([cb], resources=[(x0, 0), (b, 0)], fault="either", label="make X0 (v2) forced at X1's index")
    s.fault_state("make X0")
    s.resume_root()
    s.fifo.wait_stamp(st, label="its stamp")
    t.blit(b_fill_buf(b, 0x80, 0x10, 0x6B), resources=[(b, 0)], label="the next exec")
    t.dump(pl + 16 * x0, 32, "X0's and X1's entries after X0")
    t.dump(b_va + 0x40, 0x50, "B+0x40: X0's texels, then the next exec's fill")
    t.delete()
    s.events("end")
    return s

def _index_reuse(name, doc, make):
    """A texture X (zero entry) is made first, then the object make(t, x)
    returns (zero entry), copied from or synchronized; its entry is dumped;
    40 (operation 1000) deletes it, which clears its entry; then a texture Y
    with a zero entry is made and its entry dumped."""
    from scenarios.textures_placement import install_with_placements
    s = Scenario(name, setups=("isolate",), host=True, doc=doc)
    s.bring_up()
    t = RTask(s, pages=32)
    x = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc1(usage=0x13)))
    v, use = make(t, x)
    y = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc1(usage=0x13)))
    b, b_va = t.shared(PAGE)
    t.define()
    pl = install_with_placements(t, {})
    t.blit(b_copy_tex_buf(x, b, 0, 16, 64, FULL), resources=[(x, 0), (b, 0)], label="make X")
    t.blit(use(v, b), resources=[(x, 0), (b, 0)], label="make the object")
    t.dump(pl + 16 * x, 16, "X's entry")
    t.dump(pl + 16 * v, 16, "the object's entry")
    cmd_delete_object(s, 1, 1000, v, label="40, 1000 on the object")
    t.dump(pl + 16 * v, 16, "the object's entry after 40")
    t.blit(b_copy_tex_buf(y, b, 0x40, 16, 64, FULL), resources=[(y, 0), (b, 0)], label="make Y")
    t.dump(pl + 16 * y, 16, "Y's entry")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def mutgap4_res_swizzle_view_index_freed():
    """A swizzle view's MTLTexture is released once its resource holds it, so
    deleting the view (40) frees its Metal resource index: a view of X with
    swizzle (5, 4, 3, 2) and a zero entry is made by a copy through it and
    writes the index Metal gives it; after 40 a new texture Y made with a
    zero entry is given an index and ID Metal chooses, both dumped. Measured:
    X gets 1, the view 2, and Y the freed 2. Host-keyed (Metal's indices)."""
    return _index_reuse("mutgap4_res_swizzle_view_index_freed", mutgap4_res_swizzle_view_index_freed.__doc__,
                        lambda t, x: (t.add(S8, ser_view_type(x, RGBA8Unorm, TT_2D, swizzle=(5, 4, 3, 2))),
                                      lambda v, b: b_copy_tex_buf(v, b, 0x80, 16, 64, FULL)))


@scenario(setups=("isolate",), host=True)
def mutgap4_res_memoryless_index_freed():
    """A memoryless serializer texture's MTLTexture is released once its
    discardable resource holds it, so deleting it (40) frees its Metal
    resource index: one made by a 314 (a SynchronizeResource on it creates
    it; docs/testing.md, section 7.1) writes the index Metal gives it; after
    40 a new texture Y made with a zero entry is given an index and ID Metal
    chooses, both dumped. Measured: X gets 1, the memoryless texture 2, and Y
    the freed 2. Host-keyed (Metal's indices)."""
    from pgtest.resources_build import b_sync_resource
    return _index_reuse("mutgap4_res_memoryless_index_freed", mutgap4_res_memoryless_index_freed.__doc__,
                        lambda t, x: (t.add(S8, ser_memoryless(desc1(options=0x30, usage=4))),
                                      lambda v, b: b_sync_resource(v)))


# ---- pipelines and their placement entries ----

@scenario(setups=("isolate",), host=True)
def mutgap4_res_render_pipeline_unforced_entry():
    """A render pipeline whose placement entry has resourceID 0 is not
    placed, but its entry is still written back with the index and ID Metal
    gave it: ICBTask's ICB-capable render pipeline with the entry {0x77, 0},
    used by a render ICB's draw, reads Metal's index and ID afterwards
    (measured {1, 1}), not 0x77. Host-keyed (the driver's IDs)."""
    from scenarios.icb import ICBTask
    from scenarios.textures_placement import install_with_placements
    s = Scenario("mutgap4_res_render_pipeline_unforced_entry", setups=("isolate",), host=True,
                 doc=mutgap4_res_render_pipeline_unforced_entry.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    icb = w.icb([render_cmd(pso=w.rpso, vbinds=[(O, 0, 0)], fbinds=[(O, 0, 0x10)])], kind="render")
    w.t.define()
    pl = install_with_placements(w.t, {w.rpso: (0x77, 0)})
    w.t.write(w.tb_va, bytes(0x40))
    w.run_render(icb, 0, 1, label="the draw")
    w.pixels("the draw")
    w.t.dump(pl + 16 * w.rpso, 16, "the render pipeline's placement entry")
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def mutgap4_res_compute_pipeline_forced():
    """The compute counterpart of mutgap3_res_render_pipeline_forced_taken
    and _forced_index (a placed compute pipeline is made with
    setForceResourceIndex:YES): three ICB-capable compute pipelines, P (zero
    entry), Q (entry {1, 1}) and S (entry {5, 5}), dispatched in that order
    in one exec, so P is made first and writes the index and ID Metal gives
    it, Q is made at index 1 by force and S at the free index 5. Each
    dispatch writes its output. Measured: P is given {1, 1}, Q is made at
    index 1 too ({1, 1}, as the render pipelines were), S at {5, 5}.
    Host-keyed (the driver's indices)."""
    r = Rig("mutgap4_res_compute_pipeline_forced", mutgap4_res_compute_pipeline_forced.__doc__,
            setups=("isolate",), host=True)
    out, out_va = r.buffer()
    p = r.kernel(icb=True)
    q = r.kernel(icb=True)
    sp = r.kernel(icb=True)
    t = r.t
    t.define()
    from scenarios.textures_placement import install_with_placements
    pl = install_with_placements(t, {q: (1, 1), sp: (5, 5)})
    r.irqs += 2
    r.compute(C.set_pipeline(p), C.set_buffers(0, [(out, 0)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)),
              C.set_pipeline(q), C.set_buffers(0, [(out, 0x10)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)),
              C.set_pipeline(sp), C.set_buffers(0, [(out, 0x20)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)))
    r.dump(out_va, 0x30, "out")
    for ref, name in ((p, "P"), (q, "Q"), (sp, "S")):
        t.dump(pl + 16 * ref, 16, "%s's placement entry" % name)
    return r.end()


# ---- ICBs ----

@scenario
def mutgap4_res_icb_threadgroup_memory():
    """A compute ICB whose descriptor has one kernel threadgroup memory bind
    (maxKernelThreadgroupMemoryBindCount 1) and whose command sets length 64
    at index 0 (prepare), executing tg_reverse
    (fixtures/shaders/compute_tgmem.metal) over two threadgroups of 16: each
    group is reversed through threadgroup memory 0, out[gid] = 3 x the
    mirrored gid, as compute_threadgroup_memory's direct dispatch."""
    from scenarios.icb import ICBTask
    s = Scenario("mutgap4_res_icb_threadgroup_memory", doc=mutgap4_res_icb_threadgroup_memory.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    tg = w.t.compute_pipeline("shaders/compute_tgmem.metallib", icb=True)
    a = w.icb([compute_cmd(ctype=0x20, pso=tg, binds=[(O, 0, 0)], tg_lengths=[64], grid=(2, 1, 1),
                           tg=(16, 1, 1))], command_types=ICB_DISPATCH, ktg=1)
    w.start()
    w.clear_out()
    w.run_compute(a, 0, 1, label="tg_reverse from the ICB")
    w.out_dump("out", 0x80)
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def mutgap4_res_icb_sync_patch_start_tgmem():
    """The ICB write-back (53 after an exec that lists each ICB with
    invalidateGuest; icb_sync's rig) of an indexed-patches command whose
    patchStart is 3 and patchCount 2, and of a compute command whose one
    threadgroup memory length is 96 (maxKernelThreadgroupMemoryBindCount 1,
    which the write-back reads). GPU addresses are the host's: keyed by the
    host. Isolate only: in-process the write-back aborts
    (icb_sync_in_process)."""
    from scenarios.icb import ICBTask, sync_refs
    s = Scenario("mutgap4_res_icb_sync_patch_start_tgmem", setups=("isolate",), host=True,
                 doc=mutgap4_res_icb_sync_patch_start_tgmem.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    idx, _ = w.t.buffer(PAGE, contents=struct.pack("<3H", 0, 1, 2))
    patches = w.icb([render_cmd(ctype=8, pso=w.rpso,
                                args=draw_indexed_patches_args(points=3, start=3, count=2, index=(idx, 0, 0),
                                                               cp_index=(idx, 0, 2)),
                                tess=tess_args(idx, 0, 0, 0, 1.0))],
                    kind="render", command_types=ICB_DRAW_INDEXED_PATCHES)
    tg = w.t.compute_pipeline("shaders/compute_tgmem.metallib", icb=True)
    comp = w.icb([compute_cmd(ctype=0x20, pso=tg, binds=[(O, 0, 0)], tg_lengths=[96], grid=(2, 1, 1),
                              tg=(16, 1, 1))], command_types=ICB_DISPATCH, ktg=1)
    icbs = [patches, comp]
    w.start()
    w.t.exec_segments([seg_blit()], resources=w.res() + [(idx, 0)] + [(i, abi.RES_INVALIDATE_GUEST) for i in icbs],
                      label="prepared, guest invalidated")
    s.fault_state("prepared")
    for i in icbs:
        va, size = w.icbs[i]
        w.t.fill(va, size, 0xEE)
    sync_refs(s, w.t.tid, icbs, label="synchronize")
    for i, name in zip(icbs, ("indexed patches, patchStart 3", "compute, threadgroup memory 96")):
        va, size = w.icbs[i]
        w.t.dump(va, 0x100, "written back: " + name)
    s.events("end")
    return s


def seg_blit():
    from pgtest.task import seg
    return seg(abi.ENC_BLIT)


@scenario
def mutgap4_res_icb_inherit_raster_state():
    """A render ICB's inheritCullMode and inheritFrontFacingWinding bits
    (bits 8 and 9 of the descriptor's inherit bits) reach Metal: an ICB with
    both draws the fixture triangle in two passes whose encoder first sets
    cull mode Back (0x6b) and then front-facing winding Clockwise or
    CounterClockwise (0x73), so one of the two passes culls the triangle. An
    ICB with inheritDepthTestBounds (bit 11) draws it in a pass with the
    encoder's default bounds (nothing a guest sends sets them: the render
    decoder has no depth-test-bounds command and prepare sets none). Each
    pass's 4x4 target is dumped. Measured: Clockwise culls the triangle (the
    clear colour stays), CounterClockwise draws it, and the depth-test-bounds
    ICB draws it."""
    from scenarios.icb import ICBTask
    s = Scenario("mutgap4_res_icb_inherit_raster_state", doc=mutgap4_res_icb_inherit_raster_state.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    cmd = render_cmd(pso=w.rpso, vbinds=[(O, 0, 0)], fbinds=[(O, 0, 0x10)])
    raster = w.icb([cmd], kind="render", flags=(1 << 8) | (1 << 9))
    bounds = w.icb([cmd], kind="render", flags=1 << 11)
    w.start()
    for winding, label in ((0, "cull Back, winding Clockwise"), (1, "cull Back, winding CounterClockwise")):
        w.t.write(w.tb_va, bytes(0x40))
        w.run_render(raster, 0, 1, extra=R.value(0x6B, 2) + R.value(0x73, winding), label=label)
        w.pixels(label)
    w.t.write(w.tb_va, bytes(0x40))
    w.run_render(bounds, 0, 1, label="inheritDepthTestBounds")
    w.pixels("inheritDepthTestBounds")
    s.events("end")
    return s


# ---- single-plane textures ----

@scenario
def mutgap4_res_mipgen_two_levels():
    """A type 3 texture (mipmaps generated) of exactly two levels: prepare
    pages level 0 and generates level 1 on the GPU (a single level would not
    generate). A 4x4 RGBA8Unorm texture whose level 0 is 40 30 20 10 in every
    texel and whose guest level-1 bytes are 0xee (not paged): level 1 read
    back with 302 is the average, 40 30 20 10."""
    s = Scenario("mutgap4_res_mipgen_two_levels", doc=mutgap4_res_mipgen_two_levels.__doc__)
    s.bring_up()
    t = RTask(s)
    va = t.area(PAGE, contents=bytes.fromhex("40302010") * 16 + b"\xEE" * 16)
    g = t.add(abi.T_TEXTURE_MIPGEN, texture_rec(PAGE, va // PAGE, [simple_dim(4, 4, levels=2)],
                                                desc1(mips=2)))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(g, b, 0, 16, 64, FULL), b_copy_tex_buf(g, b, 0x40, 8, 16, (2, 2, 1), level=1),
           resources=[(g, 0), (b, 0)], label="read both levels")
    t.dump(b_va, 0x50, "B: level 0, then level 1")
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap4_res_cube_array_pages():
    """A cube array of two cubes pages face slice 6 x cube + face from guest
    offset bytesPerSlice x that slice: a 4x4 RGBA8Unorm cube array (type 2,
    arrayLength 2, 0x40 bytes a face) whose face slice k holds 0x10 + k in
    every byte. Texel (0, 0) of each of the 12 slices is copied out (302);
    then slice 7 is filled with green and synchronized (flags 6), and every
    slice is written back to its own offset."""
    s = Scenario("mutgap4_res_cube_array_pages", doc=mutgap4_res_cube_array_pages.__doc__)
    s.bring_up()
    t = RTask(s)
    va = t.area(PAGE, contents=b"".join(bytes([0x10 + k]) * 0x40 for k in range(12)))
    c = t.add(abi.T_TEXTURE, texture_rec(PAGE, va // PAGE, [dimension([mip(0, 64, 16, 4, 4)], array=2, cube=1,
                                                                      slice_bytes=0x40)],
                                         desc1(tt=TT_CubeArray, array=2)))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(*[b_copy_tex_buf(c, b, 4 * k, 4, 4, (1, 1, 1), slice_=k) for k in range(12)],
           resources=[(c, 0), (b, 0)], label="texel 0 of each slice")
    t.dump(b_va, 0x30, "B: slices 0-11")
    t.blit(b_fill_tex_color(c, FULL, (0.0, 1.0, 0.0, 1.0), RGBA8Unorm, slice_=7), resources=[(c, 6), (b, 0)],
           label="fill slice 7, synchronize")
    t.dump(va, 0x300, "the cube array's pages")
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap4_res_sync_after_discard():
    """A discard (63) releases a normal texture's transfer buffer and leaves
    it host-valid, so the next prepare only devolatilizes it; a synchronize
    after a GPU write then makes the transfer buffer again and writes the
    texture into the guest's pages. A 4x4 RGBA8Unorm texture (type 2) is
    copied out, filled red and synchronized; 63; filled green and
    synchronized again; its pages show green."""
    s = Scenario("mutgap4_res_sync_after_discard", doc=mutgap4_res_sync_after_discard.__doc__)
    s.bring_up()
    t = RTask(s)
    va = t.area(PAGE, contents=bytes((i * 5 + 1) & 0xFF for i in range(0x40)))
    a = t.add(abi.T_TEXTURE, texture_rec(PAGE, va // PAGE, [simple_dim(4, 4)], desc1()))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(a, b, 0, 16, 64, FULL), resources=[(a, 0), (b, 0)], label="paged in")
    t.blit(b_fill_tex_color(a, FULL, (1.0, 0.0, 0.0, 1.0), RGBA8Unorm), resources=[(a, 6), (b, 0)],
           label="fill red, synchronize")
    t.dump(va, 0x40, "A's pages: red")
    cmd_discard(s, 1, [a], label="63")
    t.blit(b_fill_tex_color(a, FULL, (0.0, 1.0, 0.0, 1.0), RGBA8Unorm), resources=[(a, 6), (b, 0)],
           label="fill green, synchronize")
    t.dump(va, 0x40, "A's pages: green")
    t.blit(b_copy_tex_buf(a, b, 0x40, 16, 64, FULL), resources=[(a, 0), (b, 0)], label="copied out")
    t.dump(b_va, 0x80, "B: as paged in, then green")
    t.delete()
    s.events("end")
    return s


def _view_world(name, doc, setups, spin=False, **kw):
    """A normal texture X and a view of it (type 8, commandType 27), made by
    a copy through it; 52 then marks the view host-valid and guest-invalid,
    so it needs a synchronize. Returns (s, t, v, b, b_va), and with spin=True
    also the task's pgtest.spin_util.Spin, warmed up before the copy."""
    s = Scenario(name, setups=setups, doc=doc, **kw)
    s.bring_up()
    t = RTask(s)
    x_va = t.area(PAGE, contents=bytes((i * 3 + 7) & 0xFF for i in range(0x40)))
    x = t.add(abi.T_TEXTURE, texture_rec(PAGE, x_va // PAGE, [simple_dim(4, 4)], desc1(usage=0x13)))
    v = t.add(S8, ser_view_type(x, RGBA8Unorm, TT_2D))
    b, b_va = t.shared(PAGE)
    if spin:
        from pgtest.spin_util import Spin
        sp = Spin(t)
    t.define()
    t.install()
    if spin:
        sp.warm_up()
    t.blit(b_copy_tex_buf(v, b, 0, 16, 64, FULL), resources=[(x, 0), (b, 0)], label="copy through the view")
    t.dump(b_va, 0x40, "B: X through the view")
    cmd_invalidate(s, 1, [(v, 0, 1, 1, 0)], label="52: the view host-valid, guest-invalid")
    s.fault_state("after 52")
    if spin:
        return s, t, v, b, b_va, sp
    return s, t, v, b, b_va


def _view_sync_exec(t, v, b, spin=None):
    """An ExecIndirect3 payload: a fill of B+0x40 with 0x5a, listing the
    view with the synchronize flag (4) and B. spin: a Spin, whose kernel
    runs after the fill, in a second command-buffer range of the exec."""
    from pgtest.task import seg
    t.reset_cmdbufs()
    cbs = [t.cmdbuf(seg(abi.ENC_BLIT, b_fill_buf(b, 0x40, 0x10, 0x5A)))]
    res = [(v, abi.RES_SYNCHRONIZE), (b, 0)]
    if spin:
        return spin.payload(cbs, res)
    return (struct.pack("<III", t.tid, len(res), len(cbs)) + b"".join(struct.pack("<II", r, f) for r, f in res)
            + b"".join(struct.pack("<QQ", va, n) for va, n in cbs))


@scenario(setups=("blocks", "map"))
def mutgap4_res_view_synchronize_dies():
    """In-process, a view that needs a synchronize (_view_world) listed with
    the synchronize flag in an exec throws "Attempting to synchronize a view
    resource" inside the exec's completion, after the exec's command buffer
    is committed: the error path completes its stamp with 2, and the
    synchronize buffer, committed on the way out, completes it again: the
    host dies on the pending stamp's assert (its pending count), as
    covgap_svc_icb_exec_synchronize (docs/testing.md, section 7.4). The last
    command (covgap_svc_util.last_exec). The service:
    mutgap4_res_view_synchronize_isolate."""
    from pgtest.covgap_svc_util import last_exec
    s, t, v, b, b_va = _view_world("mutgap4_res_view_synchronize_dies", mutgap4_res_view_synchronize_dies.__doc__,
                                   ("blocks", "map"))
    last_exec(s, abi.OP_EXEC_INDIRECT3, _view_sync_exec(t, v, b),
              label="an exec, the view synchronized: the host dies")
    return s


@scenario(setups=("isolate",))
def mutgap4_res_view_synchronize_isolate():
    """In the service, the view of _view_world listed with the synchronize
    flag beside a fill of B+0x40: the service's completion of the exec throws
    after the exec's command buffer is committed, and its error path replies
    2: the guest sees a fault (the exec's fill still lands). After the resume
    the next exec (a fill of B+0x50 with 0x6b) completes, and B+0x40 shows
    both fills. In-process the host dies: mutgap4_res_view_synchronize_dies.

    The service completes the exec a second time: the synchronize buffer,
    committed on the way out, sends success once the exec's command buffer
    has completed, and the host keeps the first completion to reach it
    (pgtest/spin_util.py has the race). So the exec also runs a kernel of
    fixed work after the fill (about 2 s): locked are the fault first, while
    that kernel runs (0x102c = 1, one interrupt), and then the second
    completion arriving to nothing: once the kernel has ended (c[2] = 0xdead)
    no interrupt, no stamp, the FIFO still faulted until the resume.
    Exclusive, as pgtest/spin_util.py says."""
    s, t, v, b, b_va, sp = _view_world(
        "mutgap4_res_view_synchronize_isolate", mutgap4_res_view_synchronize_isolate.__doc__, ("isolate",),
        spin=True, watchdog_s=90, exclusive=True)
    s.submit(abi.OP_EXEC_INDIRECT3, _view_sync_exec(t, v, b, spin=sp), fault=True, timeout_ms=30000,
             label="the exec listing the view with synchronize")
    s.fault_state("the exec listing the view with synchronize")
    s.events("the fault, the kernel still running")
    sp.wait_end()
    s.events("the exec's command buffer completed, after the fault")
    s.fault_state("after the kernel ended")
    s.resume_root()
    t.blit(b_fill_buf(b, 0x50, 0x10, 0x6B), resources=[(b, 0)], label="the next exec")
    t.dump(b_va + 0x40, 0x20, "B+0x40")
    t.delete()
    s.events("end")
    return s


# ---- texture records ----

@scenario
def mutgap4_res_compressed_slice_bytes():
    """The texture dimension check's compressed path accepts a mip whose end
    fits a nonzero bytesPerSlice: BC1 8x8 textures of one level (32 bytes)
    with bytesPerSlice 0x20 (the mip ends at the slice's end) and 0x40, each
    listed in its own exec beside a probe buffer; both are made and both
    execs complete."""
    from scenarios.resources import creation_case
    from scenarios.textures import _tex_case
    bcd = desc1(fmt=BC1_RGBA, w=8, h=8)
    cases = [_tex_case([dimension([mip(0, 32, 16, 8, 8)], block=(4, 4, 1, 8), compressed=1, slice_bytes=sb)], bcd,
                       faults=False) for sb in (0x20, 0x40)]
    return creation_case("mutgap4_res_compressed_slice_bytes", mutgap4_res_compressed_slice_bytes.__doc__, cases)


# ---- mapper-ref buffers ----

@scenario(setups=("map", "isolate"))
def mutgap4_res_mapper_buffer_discard_keeps_mapping():
    """A mapper-ref buffer supports discard: 63 releases its surface but
    keeps the resource and its mapping ID, so after the guest rewrites the
    buffer's object record to name mapping 2, the next exec retains surface 1
    again and a copy out of the buffer reads surface 1's page, not surface
    2's. Surfaces 1 and 2 are mapped over their own pages, each with its own
    pattern (resources_mapper's _surfaces). Map and isolate only (the
    IOSurface host device needs a memory map)."""
    from scenarios.resources_mapper import W, H, _surfaces
    s = Scenario("mutgap4_res_mapper_buffer_discard_keeps_mapping", setups=("map", "isolate"), iosfc=True,
                 doc=mutgap4_res_mapper_buffer_discard_keeps_mapping.__doc__)
    s.bring_up()
    d, pages = _surfaces(s, {1: (W, H, 4, b"BGRA", None), 2: (W, H, 4, b"BGRA", None)})
    t = RTask(s)
    m = t.add(abi.T_MAPPER_BUFFER, mapper_ref_rec(1))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_buf_buf(m, b, 0, 0, 0x20), resources=[(m, 0), (b, 0)], label="copy out: surface 1")
    cmd_discard(s, 1, [m], label="63 on it")
    t.write(t.ol.entries[m][2], mapper_ref_rec(2))
    t.blit(b_copy_buf_buf(m, b, 0, 0x20, 0x20), resources=[(m, 0), (b, 0)],
           label="copy out after 63 and the record's rewrite")
    t.dump(b_va, 0x40, "B: before, then after")
    t.delete()
    s.events("end")
    return s
