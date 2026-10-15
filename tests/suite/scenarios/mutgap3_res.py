"""The resource manager, textures, heaps, ICBs, the displays and the
IOSurface host device at their edges: each scenario locks one behaviour, as
its docstring says; the docstrings say what the code does, which is what the
expected files record."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.compute_objects import Compute as C, Rig
from pgtest.display_guest import DisplayGuest, display
from pgtest.exec_objects import info_heap_size
from pgtest.icb_cmds import CMD, compute_cmd, render_cmd
from pgtest.iosfc_guest import (CMD_MAP, CMD_TEST, I_DESC_BASE, IOSFC, desc_v1, desc_v2, plane_v1,
                                plane_v2)
from pgtest.registry import scenario
from pgtest.resources_build import (BC1_RGBA, Depth32Float_Stencil8, RGBA8Unorm, TT_3D, TT_Cube, TT_CubeArray,
                                    XR10_2P_A8, RTask, b_copy_buf_buf, b_copy_tex_buf, b_fill_buf, b_fill_tex_color,
                                    backing_rec, cmd_synchronize, desc1, desc2, dimension, heap_buffer_rec,
                                    heap_rec, iosurface_op, mip, ref_texture_rec, shared_backing_rec,
                                    shared_texture_rec, simple_dim, texture_rec)
from pgtest.scenario import Scenario
from pgtest.task import seg

FULL = (4, 4, 1)
BGRA = 0x42475241  # 'BGRA', an IOSurface pixel format


def _pattern(n, seed):
    return bytes((i * 13 + seed * 29 + (i >> 8)) & 0xFF for i in range(n))


# ---- texture records ----

@scenario
def mutgap3_res_texture_record_clamps():
    """The texture dimension check accepts three textures, each listed in its
    own exec beside a probe buffer, and each exec completes: a 4x4 RGBA8 cube
    whose six faces of 0xa00 bytes fill 0x3c00 of its one-page (0x4000)
    allocation (the allocation check counts 6 slices a cube; 7 would not
    fit); an RGBA8 2x8 texture of four levels (2x8, 1x4, 1x2, 1x1), whose
    expected width is clamped to 1 after each halving, so levels 2 and 3 are
    compared with width 1; and a BC1 8x32 texture of six levels whose width
    reaches 1 at level 3 and stays 1 through level 5, the compressed path's
    clamp."""
    from scenarios.resources import creation_case
    from scenarios.textures import _tex_case
    cases = [
        _tex_case([dimension([mip(0, 64, 16, 4, 4)], cube=1, slice_bytes=0xA00)], desc1(tt=TT_Cube), length=PAGE,
                  faults=False),
        _tex_case([dimension([mip(0, 64, 8, 2, 8), mip(0x40, 16, 4, 1, 4), mip(0x50, 8, 4, 1, 2),
                              mip(0x58, 4, 4, 1, 1)])], desc1(w=2, h=8, mips=4, usage=1), faults=False),
        _tex_case([dimension([mip(0, 128, 16, 8, 32), mip(0x80, 32, 8, 4, 16), mip(0xA0, 16, 8, 2, 8),
                              mip(0xB0, 8, 8, 1, 4), mip(0xB8, 8, 8, 1, 2), mip(0xC0, 8, 8, 1, 1)],
                             block=(4, 4, 1, 8), compressed=1)],
                  desc1(fmt=BC1_RGBA, w=8, h=32, mips=6, usage=1), faults=False),
    ]
    return creation_case("mutgap3_res_texture_record_clamps", mutgap3_res_texture_record_clamps.__doc__, cases)


# ---- descriptors: sampler lodMinClamp, v2 swizzles, v2 sampleCount ----

@scenario
def mutgap3_res_sampler_lod_min():
    """A sampler's lodMinClamp reaches Metal: a mipmapped (Nearest) sampler
    with lodMinClamp 1.0 over a two-level texture (level 0 red, level 1
    green) samples level 1 where sample_k asks for level 0
    (fixtures/shaders/compute_sample.metal), so its four samples are green;
    the same sampler with lodMinClamp 0 samples red."""
    from scenarios.mutgap2_res import _two_level_texture
    r = Rig("mutgap3_res_sampler_lod_min", mutgap3_res_sampler_lod_min.__doc__)
    tex = _two_level_texture(r)
    out, out_va = r.buffer()
    s1 = r.sampler(mip_filter=1, lod_min=1.0)
    s0 = r.sampler(mip_filter=1, lod_min=0.0)
    p = r.kernel("shaders/compute_sample.metallib")
    r.ready()
    for i, smp in enumerate((s1, s0)):
        r.compute(C.set_pipeline(p), C.set_textures(0, [tex]), C.set_samplers(0, [smp]),
                  C.set_buffers(0, [(out, 0x10 * i)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)))
    r.dump(out_va, 0x20, "out: lodMinClamp 1, then 0")
    return r.end()


@scenario
def mutgap3_res_v2_swizzles_sampled():
    """A v2 descriptor's swizzle (a channel below 6 is Metal's
    MTLTextureSwizzle, 6 and above One) as sample_k reads it (Repeat, so
    thread i reads texel i). Four 4x1 RGBA8Unorm textures (type 2,
    commandType 52) over the same texels {R, G, B, A} = {11 22 33 80}, {44 55
    66 40}, {77 88 99 c0}, {aa bb cc 10}: swizzle (2, 3, 4, 5) reads them as
    they are; (5, 3, 4, 5) puts alpha in red; (2, 6, 4, 5) reads green as One
    (0xff); (2, 3, 6, 5) reads blue as One."""
    r = Rig("mutgap3_res_v2_swizzles_sampled", mutgap3_res_v2_swizzles_sampled.__doc__)
    t = r.t
    texels = struct.pack("<4I", 0x80332211, 0x40665544, 0xC0998877, 0x10CCBBAA)
    swizzles = ((2, 3, 4, 5), (5, 3, 4, 5), (2, 6, 4, 5), (2, 3, 6, 5))
    texs = []
    for sw in swizzles:
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
    r.dump(out_va, 0x40, "samples: (2,3,4,5), (5,3,4,5), (2,6,4,5), (2,3,6,5)")
    return r.end()


@scenario(setups=("isolate",), host=True)
def mutgap3_res_heap_size_v2_samples():
    """469 HeapTextureSizeAndAlignWithDescriptor2 passes the v2 descriptor's
    sampleCount to Metal: a 256x256 RGBA8 render target, 2D with one sample
    and 2DMultisample with four, replies different sizes. Isolate only
    (in-process info replies abort the host, info_reply_aborts_host).
    Host-keyed: the sizes are Metal's."""
    from scenarios.mutgap2_res import InfoRig
    s = Scenario("mutgap3_res_heap_size_v2_samples", setups=("isolate",), host=True,
                 doc=mutgap3_res_heap_size_v2_samples.__doc__)
    s.bring_up()
    w = InfoRig(s)
    R = w.reply
    w.info([info_heap_size(R, 0x00, v2=True, fmt=70, w=256, h=256, usage=4, options=0x20),
            info_heap_size(R, 0x10, v2=True, ttype=4, samples=4, fmt=70, w=256, h=256, usage=4, options=0x20)],
           label="469, one sample and four")
    s.fault_state("469")
    w.t.dump(w.reply_va, 0x20, "469: 2D x1, then 2DMultisample x4 {size, align}")
    s.events("end")
    return s


# ---- ICBs ----

@scenario
def mutgap3_res_icb_compute_reset_again():
    """Prepare resets a compute ICB command whose guest type is 0, the
    compute counterpart of mutgap_tex_icb_reset_again. Command 0 dispatches
    double_gid over two threads; executed, out shows 0 and 2. The guest then
    writes type 0 into it and lists the ICB with invalidateHost, so prepare
    runs again: executed into a cleared out, the reset command dispatches
    nothing and out stays zero."""
    from scenarios.icb import ICBTask
    s = Scenario("mutgap3_res_icb_compute_reset_again", doc=mutgap3_res_icb_compute_reset_again.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    a = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0)], grid=(2, 1, 1), tg=(2, 1, 1))])
    w.start()
    w.clear_out()
    w.run_compute(a, 0, 1, label="the dispatch")
    w.out_dump("dispatched", 0x10)
    va, _ = w.icbs[a]
    w.t.write(va, compute_cmd(ctype=0, pso=w.cpso, binds=[(O, 0, 0)], grid=(2, 1, 1), tg=(2, 1, 1)))
    w.clear_out()
    w.run_compute(a, 0, 1, flags=abi.RES_INVALIDATE_HOST, label="type 0, prepared again")
    w.out_dump("type 0, prepared again", 0x10)
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def mutgap3_res_icb_sync_spare_slot():
    """The compute write-back (SynchronizeResources 53 after an exec that
    lists the ICB with invalidateGuest; icb_sync's rig) writes [buffer size]
    commands, no more: an ICB of one command whose guest length has room for
    two gets its command 0 written back, and the guest's 0xEE in the second
    slot stays. Host-keyed: command 0 holds GPU addresses."""
    from scenarios.icb import ICBTask, sync_refs
    s = Scenario("mutgap3_res_icb_sync_spare_slot", setups=("isolate",), host=True,
                 doc=mutgap3_res_icb_sync_spare_slot.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    a = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0x10)], grid=(2, 1, 1), tg=(2, 1, 1))], length=2 * CMD)
    w.start()
    va, _ = w.icbs[a]
    w.t.exec_segments([seg(abi.ENC_BLIT)], resources=w.res() + [(a, abi.RES_INVALIDATE_GUEST)],
                      label="prepared, guest invalidated")
    s.fault_state("prepared")
    w.t.fill(va, 2 * CMD, 0xEE)
    sync_refs(s, w.t.tid, [a], label="synchronize")
    w.t.dump(va, CMD, "command 0 written back")
    w.t.dump(va + CMD, CMD, "the spare slot")
    s.events("end")
    return s


@scenario
def mutgap3_res_icb_inherit_viewports():
    """A render ICB's inheritViewports bit (bit 13 of the descriptor's
    inherit bits) reaches Metal. Two render ICBs draw the fixture triangle
    over the whole 4x4 target, one with the bit and one without; each runs in
    a pass whose encoder first sets the viewport (0, 0, 2, 4), the left half.
    Measured: both draw only the left half (the encoder's viewport applies to
    an ICB's draws whatever the bit says)."""
    from scenarios.icb import ICBTask
    from pgtest import render_lib as R
    s = Scenario("mutgap3_res_icb_inherit_viewports", doc=mutgap3_res_icb_inherit_viewports.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    cmd = render_cmd(pso=w.rpso, vbinds=[(O, 0, 0)], fbinds=[(O, 0, 0x10)])
    inh = w.icb([cmd], kind="render", flags=1 << 13)
    plain = w.icb([cmd], kind="render")
    w.start()
    for ref, label in ((inh, "inheritViewports"), (plain, "no inherit bits")):
        w.t.write(w.tb_va, bytes(0x40))
        w.run_render(ref, 0, 1, extra=R.viewport(0.0, 0.0, 2.0, 4.0), label=label)
        w.pixels(label)
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def mutgap3_res_render_pipeline_forced_index():
    """A render pipeline whose placement entry has a nonzero resourceID is
    made at the entry's resourceIndex (setResourceIndex: and
    setForceResourceIndex:) and writes its index and ID back: an ICB-capable
    render pipeline (ICBTask's) placed at {3, 1}, used by a render ICB's
    draw, reads {3, 3} afterwards (the compute counterpart,
    compute_pipeline_placement, reads {3, 3} too). Host-keyed: the IDs are
    the driver's."""
    from scenarios.icb import ICBTask
    from scenarios.textures_placement import install_with_placements
    s = Scenario("mutgap3_res_render_pipeline_forced_index", setups=("isolate",), host=True,
                 doc=mutgap3_res_render_pipeline_forced_index.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    icb = w.icb([render_cmd(pso=w.rpso, vbinds=[(O, 0, 0)], fbinds=[(O, 0, 0x10)])], kind="render")
    w.t.define()
    pl = install_with_placements(w.t, {w.rpso: (3, 1)})
    w.t.write(w.tb_va, bytes(0x40))
    w.run_render(icb, 0, 1, label="the draw")
    w.pixels("the draw")
    w.t.dump(pl + 16 * w.rpso, 16, "the render pipeline's placement entry")
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def mutgap3_res_render_pipeline_forced_taken():
    """A render pipeline whose placement entry forces a resourceIndex another
    render pipeline already holds: two ICB-capable render pipelines, P (zero
    entry) and Q (entry {1, 1}); a render ICB drawing with P runs first, so P
    is made first and writes the index and ID Metal gives it; then one
    drawing with Q, made at index 1 by force. Both draw the triangle.
    Measured: P is given index and ID 1, and Q is made at 1 too, its entry
    {1, 1} as P's (Metal forces an index another pipeline holds). Host-keyed:
    the indices are the driver's."""
    from scenarios.icb import ICBTask
    from scenarios.textures_placement import install_with_placements
    s = Scenario("mutgap3_res_render_pipeline_forced_taken", setups=("isolate",), host=True,
                 doc=mutgap3_res_render_pipeline_forced_taken.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    q = w.t.render_pipeline(icb=True)
    ip = w.icb([render_cmd(pso=w.rpso, vbinds=[(O, 0, 0)], fbinds=[(O, 0, 0x10)])], kind="render")
    iq = w.icb([render_cmd(pso=q, vbinds=[(O, 0, 0)], fbinds=[(O, 0, 0x10)])], kind="render")
    w.t.define()
    pl = install_with_placements(w.t, {q: (1, 1)})
    for ref, label in ((ip, "P's draw"), (iq, "Q's draw")):
        w.t.write(w.tb_va, bytes(0x40))
        w.run_render(ref, 0, 1, label=label)
        w.pixels(label)
    w.t.dump(pl + 16 * w.rpso, 16, "P's placement entry")
    w.t.dump(pl + 16 * q, 16, "Q's placement entry")
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def mutgap3_res_icb_reservation():
    """ICB indices a placement list holds are reserved when the lists are
    first mapped (by the task's rebuild): ICB A's entry is {1, 1}, the index
    and ID a fresh service's first ICB gets (icb_placement), and ICB B's is
    zero. B is executed first, so it is made first, where Metal places it
    (index and ID 2); A is then made at index 1 and its ID check passes. Both
    dispatch double_gid into out. Host-keyed: the indices are the
    driver's."""
    from scenarios.icb import ICBTask
    from scenarios.textures_placement import install_with_placements
    s = Scenario("mutgap3_res_icb_reservation", setups=("isolate",), host=True,
                 doc=mutgap3_res_icb_reservation.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    a = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0)], grid=(2, 1, 1), tg=(2, 1, 1))])
    b = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0x10)], grid=(2, 1, 1), tg=(2, 1, 1))])
    w.t.define()
    pl = install_with_placements(w.t, {a: (1, 1)})
    w.clear_out()
    w.run_compute(b, 0, 1, label="B first")
    w.t.dump(pl + 16 * b, 16, "B's entry")
    w.run_compute(a, 0, 1, label="then A")
    w.t.dump(pl + 16 * a, 16, "A's entry")
    w.out_dump("out: A's, then B's", 0x20)
    s.events("end")
    return s


# ---- textures' guest validity ----

@scenario(setups=("blocks", "map"))
def mutgap3_res_backing_sync_twice():
    """An IOSurface backing's synchronize marks its guest copy valid: a fill
    through a type 5 texture with invalidateGuest only, then 53 on task 0
    naming the backing, pages the fill out; the guest then writes 0x55 over
    the backing's first 16 bytes, and a second 53 finds the guest copy valid
    and writes nothing: the 0x55 stay. In-process only: a type 5 texture
    aborts the service (textures_backing_ref_service)."""
    from scenarios.textures_shared import define_both, kernel_and_user
    from pgtest.resources_build import BGRA8Unorm, OPT_SHARED
    s = Scenario("mutgap3_res_backing_sync_twice", setups=("blocks", "map"),
                 doc=mutgap3_res_backing_sync_twice.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    bk_va = k.area(PAGE, contents=bytes(0x100) + _pattern(64, 3))
    bk = k.add(abi.T_IOSURFACE_BACKING, backing_rec(PAGE, bk_va // PAGE, BGRA, [(0x100, 4, 4, 16, 4)]))
    tx = t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=iosurface_op(12, desc1(fmt=BGRA8Unorm,
                                                                                     options=OPT_SHARED))))
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    t.blit(b_fill_tex_color(tx, FULL, (0, 0, 1.0, 1.0), BGRA8Unorm), resources=[(tx, 2), (b, 0)],
           label="fill, invalidateGuest only")
    cmd_synchronize(s, 0, [bk], label="53 on the backing")
    k.dump(bk_va + 0x100, 0x40, "the backing's pages after 53")
    k.write(bk_va + 0x100, b"\x55" * 16)
    cmd_synchronize(s, 0, [bk], label="53 again")
    k.dump(bk_va + 0x100, 0x40, "the backing's pages after the second 53")
    t.delete()
    k.delete()
    s.events("end")
    return s


def _xr10_dims(w=64, h=4):
    """Two planes of 4 bytes a texel, color at +0 and alpha at +PAGE, as
    mutgap_service's builder."""
    return [dimension([mip(0, w * 4 * h, w * 4, w, h)], block=(1, 1, 1, 4)),
            dimension([mip(0, w * 4 * h, w * 4, w, h)], base=PAGE, block=(1, 1, 1, 4))]


@scenario
def mutgap3_res_xr10_host_only_fill():
    """A compute-paged texture (pixel format 550, type 12) starts with its
    guest copy valid: an exec fills it listed with flags 0, which changes
    only the host copy and leaves the guest copy valid; the guest then writes
    0xAA over both planes' first 0x40 bytes, and 53 naming the texture finds
    nothing to synchronize (needsSync): both planes keep the 0xAA."""
    s = Scenario("mutgap3_res_xr10_host_only_fill", doc=mutgap3_res_xr10_host_only_fill.__doc__)
    s.bring_up()
    t = RTask(s, pages=96)
    va = t.area(2 * PAGE, contents=_pattern(0x400, 4))
    x = t.add(abi.T_DUAL_PLANE_TEXTURE, texture_rec(2 * PAGE, va // PAGE, _xr10_dims(),
                                                    desc1(fmt=XR10_2P_A8, w=64, h=4, usage=3)))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_tex_color(x, (4, 2, 1), (1.0, 0.5, 0.25, 1.0), XR10_2P_A8), b_fill_buf(b, 0, 0x10, 0x01),
           resources=[(x, 0), (b, 0)], label="fill it, flags 0")
    t.fill(va, 0x40, 0xAA)
    t.fill(va + PAGE, 0x40, 0xAA)
    cmd_synchronize(s, 1, [x], label="53 naming it")
    t.dump(va, 0x40, "color plane after 53")
    t.dump(va + PAGE, 0x40, "alpha plane after 53")
    t.delete()
    s.events("end")
    return s


def _ds_cube_dims(arrays, depth_base=0, stencil_base=0x400):
    """Depth32Float_Stencil8 cube faces of 4x4 (mutgap_textures' builder):
    plane 0 64 bytes a face at depth_base, plane 1 16 bytes a face at
    stencil_base."""
    return [dimension([mip(0, 64, 16, 4, 4)], array=arrays, cube=1, base=depth_base, slice_bytes=64,
                      block=(1, 1, 1, 4)),
            dimension([mip(0, 16, 4, 4, 4)], array=arrays, cube=1, base=stencil_base, slice_bytes=16,
                      block=(1, 1, 1, 1))]


@scenario
def mutgap3_res_dual_plane_cube_array_slice_6():
    """A two-plane Depth32Float_Stencil8 cube array of two cubes (type 12):
    array slice 1's face 0 pages to texture slice 6 = 1 x 6 + 0 from guest
    offset bytesPerSlice x 6 of each plane. Copying texture slice 6's two
    planes out shows the guest's depth floats at +0x180 and stencil bytes at
    +0x460."""
    from scenarios.mutgap_textures import _depth_faces
    s = Scenario("mutgap3_res_dual_plane_cube_array_slice_6", doc=mutgap3_res_dual_plane_cube_array_slice_6.__doc__)
    s.bring_up()
    t = RTask(s)
    va = t.area(2 * PAGE, contents=_depth_faces(12))
    t.write(va + 0x400, bytes(range(0x40, 0x40 + 12 * 16)))
    d = t.add(abi.T_DUAL_PLANE_TEXTURE, texture_rec(2 * PAGE, va // PAGE, _ds_cube_dims(2),
                                                    desc1(fmt=Depth32Float_Stencil8, tt=TT_CubeArray, array=2,
                                                          usage=3)))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(d, b, 0, 16, 64, FULL, slice_=6, options=1),
           b_copy_tex_buf(d, b, 0x40, 4, 16, FULL, slice_=6, options=2),
           resources=[(d, 0), (b, 0)], label="copy texture slice 6's two planes out")
    t.dump(b_va, 0x50, "B: slice 6's depth, then its stencil")
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap3_res_shared_backing_3d():
    """A shared texture backing (type 17) in task 0 that is an RGBA8 4x4x2 3D
    texture over 128 distinct guest bytes, read from task 1 through a type 16
    texture: the backing pages in with bytesPerImage = bytesPerRow x rows,
    rows = ceil(height / blockHeight) = 4, so depth slice 1 comes from +0x40,
    and a copy of the whole volume out shows the 128 bytes in order."""
    from scenarios.textures_shared import define_both, kernel_and_user
    s = Scenario("mutgap3_res_shared_backing_3d", doc=mutgap3_res_shared_backing_3d.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    k_va = k.area(PAGE, contents=_pattern(128, 5))
    kb = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, k_va // PAGE,
                                                         [dimension([mip(0, 128, 16, 4, 4, 2)], block=(1, 1, 1, 4))],
                                                         desc1(w=4, h=4, d=2, tt=TT_3D, options=0x20, usage=3)))
    st = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kb))
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    t.blit(b_copy_tex_buf(st, b, 0, 16, 64, (4, 4, 2)), resources=[(st, 0), (b, 0)], label="copy the volume out")
    t.dump(b_va, 0x80, "B: depth slices 0 and 1")
    t.delete()
    k.delete()
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def mutgap3_res_legacy_heap_leaf_rewrite():
    """An in-process heap (a legacy heap) keeps the transfer buffer its first
    prepare made (it makes one only when it has none): exec 1 copies a heap
    buffer out; the guest then moves the heap's first VA page to another RAM
    page holding other bytes, and exec 2 lists the heap with invalidateHost
    (flag 1), so it is paged in again from the same buffer, which maps the
    old RAM page (in the blocks setup the host task's mapping is the old page
    anyway, memory_gap_table_rewrite): both copies show the old page's bytes.
    The heap counterpart of mutgap_tex_leaf_rewrite."""
    s = Scenario("mutgap3_res_legacy_heap_leaf_rewrite", setups=("blocks", "map"),
                 doc=mutgap3_res_legacy_heap_leaf_rewrite.__doc__)
    s.bring_up()
    t = RTask(s, pages=80)
    h_va = t.area(4 * PAGE, contents=_pattern(0x100, 6))
    h = t.add(abi.T_HEAP, heap_rec(h_va // PAGE, 4 * PAGE, 1))
    hb = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(h, 0, PAGE))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_buf_buf(hb, b, 0, 0, 0x40), resources=[(h, 0), (hb, 0), (b, 0)], label="first copy out")
    other = s.alloc_page()
    s.mem(s.pa(other), _pattern(0x100, 7))
    s.u32(s.pa(t.pt_root, 4 * (h_va // PAGE)), s.pfn(other) | abi.PTE_WRITABLE)
    t.blit(b_copy_buf_buf(hb, b, 0, 0x40, 0x40), resources=[(h, 1), (hb, 0), (b, 0)],
           label="after the leaf moved, listed with invalidateHost")
    t.dump(b_va, 0x80, "B: before and after the leaf moved")
    t.delete()
    s.events("end")
    return s


# ---- displays ----

@scenario
def mutgap3_res_display_doorbell_paused_dies():
    """The display doorbell (0x1220) rung while the device is paused
    (rvg_device_pause, which pauses every guest display) reaches
    GuestDisplay::doorbell with no run-state check, whose first act is to
    assert that the guest display is not paused: the host dies (SIGABRT), on
    a port whose guest display has no display at all. A guest kills the host
    with one register write while the hypervisor has it paused."""
    from scenarios.device_lifecycle import display_page
    s = Scenario("mutgap3_res_display_doorbell_paused_dies", doc=mutgap3_res_display_doorbell_paused_dies.__doc__)
    s.bring_up()
    display_page(s)
    s.events("a nub without a display")
    s.api("pause")
    s.events("paused")
    s.w(abi.R_DISPLAY_DOORBELL, 0)
    s.quiesce(100)
    s.api("unpause")
    s.events("after the doorbell")
    return s


# ---- the IOSurface host device ----

MAPPED = ("map", "isolate")
W, H = 16, 8


@scenario(setups=MAPPED)
def mutgap3_res_iosfc_base_drops_every_page():
    """Writing the descriptor base drops every descriptor page's cached
    mapping, not only page 0's: mapping ID 17 (v2: 16 descriptors a page, so
    on descriptor page 1) from table T1, whose page 1 holds a valid
    descriptor, then unmapping it; then a new base, table T2, whose page 1
    holds an invalid descriptor for ID 17: mapping 17 again reads T2's page
    (the entry's status is not 0) and leaves T1's page as it was."""
    from pgtest.iosfc_guest import CMD_UNMAP, pte
    s = Scenario("mutgap3_res_iosfc_base_drops_every_page", setups=MAPPED, iosfc=True,
                 doc=mutgap3_res_iosfc_base_drops_every_page.__doc__)
    d = IOSFC(s, v2=True)
    d.start()
    root, data = d.surface_pages(1)
    d.put_desc(17, desc_v2(root, W, H))
    idx = d.submit((CMD_MAP, 17), (CMD_UNMAP, 17))
    d.status(idx, "T1: map 17, unmap 17")
    p1 = d.desc_gpa(17)
    t2 = s.alloc_page()
    p2 = s.alloc_page()
    s.u32(s.pa(t2, 4), pte(s.pfn(p2)))
    s.mem(s.pa(p2, 0x400), desc_v2(root, W, H, valid=0))
    s.iosfc_w(I_DESC_BASE, s.pa(t2))
    idx = d.submit((CMD_MAP, 17))
    d.status(idx, "T2: map 17")
    s.dump(p1, 4, "T1's descriptor 17, word 0")
    s.dump(s.pa(p2, 0x400), 4, "T2's descriptor 17, word 0")
    s.events("end")
    return s


def _patched(data, off, value):
    b = bytearray(data)
    b[off] = value
    return bytes(b)


def _good_plane(mk, i=0, **kw):
    """A 16x8 plane of 1x1 elements based at its offset, with four 8-bit
    components, which IOSurface accepts (mutgap_disp_iosfc_v1_planes_upgraded's
    good plane)."""
    args = dict(offset=512 * i, base=512 * i, size=512, ew=1, eh=1, comps=4, offsets=(0, 8, 16, 24),
                depths=(8, 8, 8, 8), names=(1, 2, 3, 4), types=(1, 1, 1, 1), ranges=(1, 1, 1, 1))
    args.update(kw)
    return mk(W, H, 64, 4, **args)


@scenario(setups=MAPPED)
def mutgap3_res_iosfc_compression_type():
    """A descriptor's base compressionType reaches the IOSurface as
    kIOSurfacePlaneCompressionType (a v1 descriptor's is copied into the v2
    form the map works on): v1 and v2, a 16x8 BGRA surface with
    compressionType 1, 2 and 0, then a one-plane surface (a good plane) with
    compressionType 1 and 0, each on its own data page, mapped then tested
    (3); each map's and test's status and each data page after the test are
    locked. Measured: every map succeeds and every type gives the same test
    writes as type 0 (IOSurface takes no compression from the surface-level
    key). The v1 case with type 1 is the process's first map of a nonzero
    type."""
    out = []
    for v2 in (False, True):
        name = "mutgap3_res_iosfc_compression_type_" + ("v2" if v2 else "v1")
        s = Scenario(name, setups=MAPPED, iosfc=True, doc=mutgap3_res_iosfc_compression_type.__doc__)
        d = IOSFC(s, v2=v2)
        d.start()
        off = 0x40 if v2 else 0x26
        mk, pl = (desc_v2, plane_v2) if v2 else (desc_v1, plane_v1)
        pages = []
        cases = [(1, {}), (2, {}), (0, {}), (1, dict(planes=[_good_plane(pl)])), (0, dict(planes=[_good_plane(pl)]))]
        labels = []
        for i, (ct, kw) in enumerate(cases):
            root, data = d.surface_pages(1)
            pages.append(data)
            d.put_desc(1 + i, _patched(mk(root, W, H, **kw), off, ct))
            labels.append("compressionType %d%s" % (ct, ", one plane" if kw else ""))
        for i, label in enumerate(labels):
            idx = d.submit((CMD_MAP, 1 + i), (CMD_TEST, 1 + i))
            d.status(idx, label)
            s.diff(label + ": data page", gpa=s.pa(pages[i]), length=PAGE)
        s.events("end")
        out.append(s)
    return out


@scenario(setups=MAPPED)
def mutgap3_res_iosfc_v1_plane_extended_bottom():
    """A v1 plane's extendedPixelsOnBottom (plane byte 0x2b) is copied into
    the v2 form and reaches the IOSurface as
    kIOSurfacePlaneExtendedPixelsOnBottom: one-plane 16x8 BGRA v1 descriptors
    (a good plane of 1x1 elements, 512 bytes), the first mapped in the
    process with 8 extra rows at the bottom, then with 0, then with 8 and a
    plane and allocation of 1024 bytes; each is mapped then tested (3), and
    each map's and test's status and data page are locked. Measured:
    IOSurface refuses 8 extra rows in a 512-byte plane (BadArgument, the test
    too, the page untouched), makes the 0-row plane (the test writes its 8
    rows), and makes the 1024-byte plane with 8 extra rows (the test writes
    16 rows)."""
    s = Scenario("mutgap3_res_iosfc_v1_plane_extended_bottom", setups=MAPPED, iosfc=True,
                 doc=mutgap3_res_iosfc_v1_plane_extended_bottom.__doc__)
    d = IOSFC(s, v2=False)
    d.start()
    pages = []
    cases = [(8, 512), (0, 512), (8, 1024)]
    for i, (ext, size) in enumerate(cases):
        root, data = d.surface_pages(1)
        pages.append(data)
        pl = _patched(_good_plane(plane_v1, size=size), 0x2B, ext)
        d.put_desc(1 + i, desc_v1(root, W, H, alloc=max(size, 512), planes=[pl]))
    for i, (ext, size) in enumerate(cases):
        label = "extendedPixelsOnBottom %d, plane size %d" % (ext, size)
        idx = d.submit((CMD_MAP, 1 + i), (CMD_TEST, 1 + i))
        d.status(idx, label)
        s.diff(label + ": data page", gpa=s.pa(pages[i]), length=PAGE)
    s.events("end")
    return s
