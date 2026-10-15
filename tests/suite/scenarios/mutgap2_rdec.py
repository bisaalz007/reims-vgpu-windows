"""The render decoder and the rate-map reset at their edges: each scenario
locks one behaviour, as its docstring says.
"""
import struct

from pgtest import abi
from pgtest import render_lib as R
from pgtest.abi import PAGE
from pgtest.compute_objects import Blit as B
from pgtest.mutgap2_rdec_util import Resolve, Resident, amp_target
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import rec, seg
from scenarios.render_state import LESS, NPIX, TRI, W, H, StateTask, case
from scenarios.render_tile import TileTask, case as tile_case

U16 = 0


@scenario(exclusive=True)
def mutgap2_rdec_primitive_type_10():
    """Primitive type 10 in every draw form
    mutgap_dec_render_primitive_type_10 leaves out: 0x00, 0x01, 0x03, 0x05,
    0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x1c, 0x1d, 0x10 and 0x11 each check
    the primitive type, which throws "Invalid primitive type", a decode
    error: the exec's command buffer is never committed (docs/testing.md,
    6.12), so neither the blit fill in its first segment nor the pass's clear
    lands. exclusive: a draw that reaches Metal with type 10 faults the GPU
    (render_draw_type9)."""
    from scenarios.render_draw import DrawTask
    s = Scenario("mutgap2_rdec_primitive_type_10", exclusive=True, doc=mutgap2_rdec_primitive_type_10.__doc__)
    s.bring_up()
    d = DrawTask(s)
    t = d.t
    out, out_va = t.buffer()
    t.define()
    t.install()
    a, i16 = d.args, d.idx16
    for label, draw in (("0x00", R.draw64(10, 0, 3)),
                        ("0x01", R.draw16(10, 0, 3)),
                        ("0x03", R.draw_inst16(10, 0, 3, 1)),
                        ("0x05", R.draw_inst_base16(10, 0, 3, 1, 0)),
                        ("0x06", R.draw_idx64(10, U16, i16, 3, 0)),
                        ("0x07", R.draw_idx16(10, U16, i16, 3, 0)),
                        ("0x08", R.draw_idx_inst64(10, U16, i16, 3, 0, 1)),
                        ("0x09", R.draw_idx_inst16(10, U16, i16, 3, 0, 1)),
                        ("0x0a", R.draw_idx_base64(10, U16, i16, 0, 3, 1, 0, 0)),
                        ("0x0b", R.draw_idx_base16(10, U16, i16, 0, 3, 1, 0, 0)),
                        ("0x1c", R.draw_idx_base64(10, U16, i16, 0, 3, 1, 0, 0, cmd=0x1C)),
                        ("0x1d", R.draw_idx_base16(10, U16, i16, 0, 3, 1, 0, 0, cmd=0x1D)),
                        ("0x10", R.draw_indirect(10, a, 0)),
                        ("0x11", R.draw_idx_indirect(10, U16, i16, a, 0, 16))):
        label += " type 10"
        t.fill(d.va, NPIX, 0x11)
        t.fill(out_va, 8, 0x11)
        t.exec_segments([seg(abi.ENC_BLIT, B.fill(out, 0, 8, 0x5A)),
                         seg(abi.ENC_RENDER, d.pass_() + R.pipeline_state(d.p) + d.vb() + draw)],
                        fault=True, timeout_ms=30000, label=label)
        s.fault_state(label)
        s.resume_root()
        t.dump(out_va, 8, label + ": the fill's buffer")
        t.dump(d.va, NPIX, label + ": the target")
    t.delete()
    s.events("end")
    return s


def two_level(st):
    """A 4x4 RGBA8 texture of two levels (type 2) over task pages: level 0
    red, level 1 green; listed in every exec. And a sampler with nearest
    mipmap filtering."""
    from pgtest.resources_build import desc1, simple_dim, texture_rec
    t = st.t
    va = t.reserve(PAGE)
    t.write(va, bytes.fromhex("ff0000ff") * 16 + bytes.fromhex("00ff00ff") * 4)
    tex = t.ol.add(abi.T_TEXTURE, texture_rec(PAGE, va // PAGE, [simple_dim(4, 4, levels=2)],
                                              desc1(w=4, h=4, mips=2, usage=1)))
    t.bufs.append(tex)
    return tex, t.sampler(mip=1)


@scenario
def mutgap2_rdec_sampler_lod_min():
    """The LOD minimum clamps of 0x80 SetVertexSamplerStatesLODClamp and 0x71
    SetFragmentSamplerStatesLODClamp reach Metal: a two-level texture (level
    0 red, level 1 green) through a mipmapped sampler, sampled at level 0 by
    render_vtex and at level 0 by render_ftex's gradients: clamps (1, 1) give
    green, (0, 0) red."""
    s = Scenario("mutgap2_rdec_sampler_lod_min", doc=mutgap2_rdec_sampler_lod_min.__doc__)
    s.bring_up()
    st = StateTask(s)
    tex, smp = two_level(st)
    st.start()
    p = R.pipeline_state
    for lo, hi in ((1.0, 1.0), (0.0, 0.0)):
        case(s, st, "0x80 clamps (%g, %g)" % (lo, hi),
             [p(st.p_vtex), R.refs_range(0x81, 0, [tex]), R.samplers_lod(0x80, 0, [(smp, lo, hi)]),
              R.draw16(TRI, 0, 3)])
        case(s, st, "0x71 clamps (%g, %g)" % (lo, hi),
             [p(st.p_ftex), R.refs_range(0x72, 0, [tex]), R.samplers_lod(0x71, 0, [(smp, lo, hi)]),
              R.draw16(TRI, 0, 3)])
    st.t.delete()
    s.events("end")
    return s


@scenario
def mutgap2_rdec_tile_samplers():
    """0x9f SetTileSamplerStates binds its samplers
    (setTileSamplerStates:withRange:): mutdec_tilesample samples a two-level
    texture (level 0 red, level 1 green) at level 0 through sampler 1. 0xa0
    first binds a sampler with clamps (0, 0) there, then 0x9f replaces it
    with a sampler whose descriptor's lodMinClamp is 1: green. 0xa0 alone:
    red."""
    s = Scenario("mutgap2_rdec_tile_samplers", doc=mutgap2_rdec_tile_samplers.__doc__)
    s.bring_up()
    tt = TileTask(s)
    t = tt.t
    p = t.pipeline("tile", t=t.function("shaders/mutdec_tilesample.metallib"))
    tex, smp = two_level(tt)
    smp1 = t.sampler(mip=1, lmin=1.0)
    tt.start()
    base = [R.pipeline_state(p), R.refs_range(0xA1, 0, [tex]), R.samplers_lod(0xA0, 1, [(smp, 0.0, 0.0)])]
    tile_case(s, tt, "0xa0 (0, 0), then 0x9f lodMinClamp 1", base + [R.refs_range(0x9F, 1, [smp1]),
                                                                     R.dispatch_tile(16, 16)])
    tile_case(s, tt, "0xa0 (0, 0) alone", base + [R.dispatch_tile(16, 16)])
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap2_rdec_tile_region_origin():
    """0xa2 DispatchThreadsPerTileInRegion and 0xa3 (WithIndex, layer 0) pass
    the region's origin on a 32x16 target cleared red, with 16x16 tiles (two
    tiles): tile.metal writes green where it runs. The region is in pixels:
    origin (0, 0) or (1, 0) with size 1x1 runs the first tile, origin (16, 0)
    with size 16x16 the second."""
    s = Scenario("mutgap2_rdec_tile_region_origin", doc=mutgap2_rdec_tile_region_origin.__doc__)
    s.bring_up()
    tt = TileTask(s)
    t = tt.t
    tex, _, va = t.target(32, 16)
    tt.start()
    n = 32 * 16 * 4
    for cmd, origin, size in ((0xA2, (0, 0, 0), (1, 1, 1)), (0xA2, (1, 0, 0), (1, 1, 1)),
                              (0xA2, (16, 0, 0), (16, 16, 1)), (0xA3, (1, 0, 0), (1, 1, 1)),
                              (0xA3, (16, 0, 0), (16, 16, 1))):
        label = "0x%02x origin %s size %s" % (cmd, origin, size)
        t.fill(va, n, 0x11)
        t.render([R.tile_size(16, 16), R.describe_pass({0: dict(tex=tex, clear=(1.0, 0.0, 0.0, 1.0))},
                                                       width=32, height=16),
                  R.pipeline_state(tt.p_tile), R.dispatch_tile_region(cmd, (16, 16, 1), origin, size)], label=label)
        s.fault_state(label)
        t.dump(va, n, label)
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap2_rdec_memoryless_depth_load():
    """A memoryless depth attachment with load action Load is rewritten to
    Clear: with clearDepth 0.6 a red draw at depth 0.5 with depth Less
    passes, with 0.4 it does not."""
    s = Scenario("mutgap2_rdec_memoryless_depth_load", doc=mutgap2_rdec_memoryless_depth_load.__doc__)
    s.bring_up()
    st = StateTask(s)
    less = st.t.depth_stencil(compare=LESS, write=0)
    st.start()
    for clear in (0.6, 0.4):
        case(s, st, "depth load Load, clear %g, red at 0.5 with Less" % clear,
             [R.pipeline_state(st.p_depth), R.depth_stencil_state(less), R.buffers(0x7D, 0, [(st.params, 0)]),
              R.draw16(TRI, 0, 3)], depth=True,
             pass_kw=dict(depth=dict(tex=st.ds, load=R.LOAD_LOAD, store=R.STORE_DONTCARE, clear=clear)))
    st.t.delete()
    s.events("end")
    return s


@scenario
def mutgap2_rdec_depth_stencil_resolve():
    """Depth and stencil resolves of a 4-sample memoryless depth-stencil
    attachment into 1-sample private D32S8 textures P and Q, each read back
    by a later pass that loads it: a red draw at depth 0.5 with depth Less
    shows whether the depth is the resolved 1.0 (drawn) or the 0.3 stored
    before (not); stencil Equal shows 3 (resolved) or 5 (stored before).
    Resolved: depth through the pass descriptor's resolve texture; depth
    stored Unknown on the memoryless attachment (kept), then 0x69
    SetDepthStoreAction MultisampleResolve, or 0x6a (which dispatches to the
    same decoder); without either nothing is resolved; 0x97
    SetDepthResolveTexture moves the resolve from P to Q; stencil stored
    Unknown (kept), then 0x78 MultisampleResolve; 0x98
    SetStencilResolveTexture moves the stencil resolve to Q."""
    s = Scenario("mutgap2_rdec_depth_stencil_resolve", doc=mutgap2_rdec_depth_stencil_resolve.__doc__)
    s.bring_up()
    r = Resolve(s)
    P, Q = r.P, r.Q
    r.run("depth resolved by the descriptor", depth=dict(resolve=P, store=R.STORE_RESOLVE), check=[("depth", P)])
    r.run("depth store Unknown, then 0x69 MultisampleResolve", depth=dict(resolve=P, store=R.STORE_UNKNOWN),
          records=[R.value(0x69, R.STORE_RESOLVE)], check=[("depth", P)])
    r.run("depth store Unknown, then 0x6a MultisampleResolve", depth=dict(resolve=P, store=R.STORE_UNKNOWN),
          records=[R.value(0x6A, R.STORE_RESOLVE)], check=[("depth", P)])
    r.run("depth store Unknown, no 0x69", depth=dict(resolve=P, store=R.STORE_UNKNOWN), check=[("depth", P)])
    r.run("depth resolved to P, 0x97 Q", depth=dict(resolve=P, store=R.STORE_RESOLVE),
          records=[R.ds_resolve(0x97, Q, 0, 0, 0, 0)], check=[("depth", P), ("depth", Q)])
    r.run("stencil store Unknown, then 0x78 MultisampleResolve", stencil=dict(resolve=P, store=R.STORE_UNKNOWN),
          records=[R.value(0x78, R.STORE_RESOLVE)], check=[("stencil", P)])
    r.run("stencil resolved to P, 0x98 Q", stencil=dict(resolve=P, store=R.STORE_RESOLVE),
          records=[R.ds_resolve(0x98, Q, 0, 0, 0, 0)], check=[("stencil", P), ("stencil", Q)])
    r.st.t.delete()
    s.events("end")
    return s


@scenario
def mutgap2_rdec_amplification_offsets():
    """0x9a SetVertexAmplificationCount passes each view mapping's two
    offsets, drawn with the amp2 pipeline (vert.metal's full-target triangle,
    frag.metal's orange) and two views, after a 0x9a of zero mappings in the
    same pass: render target array offsets (0, 1) into a two-slice array
    target draw both slices, (0, 0) only slice 0; viewport offsets (0, 1)
    with 0x83 viewports on the left and right halves draw both halves, (0, 0)
    only the left."""
    s = Scenario("mutgap2_rdec_amplification_offsets", doc=mutgap2_rdec_amplification_offsets.__doc__)
    s.bring_up()
    st = StateTask(s)
    t = st.t
    p_amp = t.pipeline("amp2", v=st.fn["vert"], f=st.fn["frag"])
    arr, arr_va = amp_target(t)
    st.start()
    zero = R.amp_count([(0, 0), (0, 0)])
    vps = R.viewports([(0.0, 0.0, 4.0, 8.0, 0.0, 1.0), (4.0, 0.0, 4.0, 8.0, 0.0, 1.0)])
    for maps in ([(0, 0), (0, 1)], [(0, 0), (0, 0)]):
        label = "render target array offsets (%d, %d)" % (maps[0][1], maps[1][1])
        t.fill(arr_va, 2 * NPIX, 0x11)
        t.render([R.describe_pass({0: dict(tex=arr, clear=(0.0, 0.0, 0.0, 1.0))}, width=W, height=H,
                                  array_length=2), R.pipeline_state(p_amp), zero, R.amp_count(maps),
                  R.draw16(TRI, 0, 3)], label=label)
        s.fault_state(label)
        t.dump(arr_va, 2 * NPIX, label + ": slices 0 and 1")
    for maps in ([(0, 0), (1, 0)], [(0, 0), (0, 0)]):
        case(s, st, "viewport offsets (%d, %d)" % (maps[0][0], maps[1][0]),
             [R.pipeline_state(p_amp), vps, zero, R.amp_count(maps), R.draw16(TRI, 0, 3)])
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap2_rdec_restart_transform():
    """0x91 SetPrimitiveRestartEnabled and 0x8f SetViewportTransformEnabled
    reach the encoder. An indexed triangle strip over 0, 1, 2, 0xffff, 3, 4,
    5, where vertex 0xffff is a real position (A at 0-2, B at 3-5), drawn
    without 0x91 (0xffff restarts: A and B) and after 0x91 {enabled, index}
    of {0, 0}, {1, 0}, {0, 15} and {1, 15}; the full-target triangle with the
    viewport transform enabled and disabled. Measured: {0, 0} and {1, 0} draw
    alike and unlike no 0x91, {0, 15} and {1, 15} cover the target; what the
    SPI's two arguments mean on this GPU is unknown (0x91 checks its index
    against maxViewportCount). With the transform disabled the clip-space
    triangle covers one pixel."""
    s = Scenario("mutgap2_rdec_restart_transform", doc=mutgap2_rdec_restart_transform.__doc__)
    s.bring_up()
    st = StateTask(s, pages=320)
    t = st.t
    # Vertex 0xffff of a 1 MiB position buffer: A at 0-2, B at 3-5.
    from scenarios.render_state import A, B as TB
    big, big_va = t.buffer(length=0x100000)
    t.write(big_va, R.float4s(*(A + TB)))
    t.write(big_va + 0xFFFF * 16, R.float4s((1.0, -1.0, 0.0, 1.0)))
    idx, _ = t.buffer(contents=struct.pack("<7H", 0, 1, 2, 0xFFFF, 3, 4, 5))
    st.start()
    strip = [R.pipeline_state(st.p_vbuf), R.buffers(0x7D, 0, [(big, 0), (st.col, 0)]),
             R.draw_idx16(4, U16, idx, 7, 0)]
    case(s, st, "strip, no 0x91", strip)
    for en, i in ((0, 0), (1, 0), (0, 15), (1, 15)):
        case(s, st, "strip, 0x91 {enabled %d, index %d}" % (en, i), [R.u32(0x91, en, i)] + strip)
    full = [R.pipeline_state(st.p_vbuf), R.buffers(0x7D, 0, [(st.pos, 144), (st.col, 0)]), R.draw16(TRI, 0, 3)]
    for en in (1, 0):
        case(s, st, "full triangle, 0x8f %d" % en, [R.u32(0x8F, en)] + full)
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap2_rdec_use_resources_stages():
    """0x89 UseResourcesWithStages makes its resources resident for an
    indirect command buffer's draw, which reads its vertex buffers only
    through the ICB: the draw covers the target only after 0x89 names both
    buffers, also as entries 1 and 2 after an unrelated buffer; a count of 0
    is accepted. 0x1b UseHeapsWithStages and 0x86 UseHeaps, each with its own
    private heap whose buffer holds the vertices: in the service a heap's
    first draw needs it, and after it a later exec without it draws too
    (measured: the residency outlasts the exec); in-process (blocks, map) a
    heap's buffer is drawn from without it."""
    s = Scenario("mutgap2_rdec_use_resources_stages", doc=mutgap2_rdec_use_resources_stages.__doc__)
    s.bring_up()
    r = Resident(s)
    r.run("no 0x89", [])
    r.run("0x89 [pos, col]", [R.use_resources_stages([r.pos, r.col], 1, 1)])
    r.run("0x89 [other, pos, col]", [R.use_resources_stages([r.other, r.pos, r.col], 1, 1)])
    r.run("0x89 count 0, then [pos, col]", [R.use_resources_stages([], 1, 1),
                                            R.use_resources_stages([r.pos, r.col], 1, 1)])
    (h1, h2), (i1, i2) = r.heaps, r.icb_heaps
    r.run("heap 1's buffer, no 0x1b", [], icb=i1)
    r.run("heap 1's buffer, 0x1b [heap 1]", [R.use_heaps_stages([h1], 1)], icb=i1)
    r.run("heap 1's buffer again, no 0x1b", [], icb=i1)
    r.run("heap 2's buffer, no 0x86", [], icb=i2)
    r.run("heap 2's buffer, 0x86 [heap 2]", [R.use_heaps([h2])], icb=i2)
    r.t.delete()
    s.events("end")
    return s


# ---- rate map resets ----

@scenario(setups=("isolate",), host=True)
def mutgap2_rdec_rate_map_reset_two_layers():
    """66 on a mutable map of two layers: the rebuilt descriptor sets both
    layers' descriptors, then reads both layers' samples, and the reset
    succeeds; 454 maps (200, 200) before and after, and a second 66 succeeds
    too. Isolate only: in-process 66 on an existing map aborts the host
    (exec_reset_rate_map_in_process). host: 454's values are Metal's."""
    from pgtest.exec_objects import ObjTask, rate_map_record
    s = Scenario("mutgap2_rdec_rate_map_reset_two_layers", setups=("isolate",), host=True,
                 doc=mutgap2_rdec_rate_map_reset_two_layers.__doc__)
    s.bring_up()
    t = ObjTask(s)
    reply, reply_va = t.buffer(PAGE)
    body = rate_map_record(w=256, h=256, layers=((2, 2), (2, 2)), mutable=1, min_factor=0.25,
                           samples=[([0.25, 1.0], [0.25, 1.0]), ([1.0, 0.5], [1.0, 0.5])])
    mut = t.ol.add(abi.T_SERIALIZED, body)
    t.define()
    t.install()
    t.fill(reply_va, 0x20, 0xEE)

    def query(off, label):
        t.exec_segments([seg(abi.ENC_INFO, rec(454, struct.pack("<IIQIff", mut, reply, off, 0, 200.0, 200.0))
                             + rec(454, struct.pack("<IIQIff", mut, reply, off + 8, 1, 200.0, 200.0)))],
                        resources=[(reply, 0)], label=label)
        s.fault_state(label)
        t.dump(reply_va + off, 0x10, label)
    query(0, "454 layers 0 and 1, before")
    for i in range(2):
        s.submit(abi.OP_RESET_RASTERIZATION_RATE_MAP, struct.pack("<II", t.tid, mut), label="66 #%d" % i)
        s.fault_state("66 #%d" % i)
    query(0x10, "454 layers 0 and 1, after")
    s.events("end")
    return s


@scenario(setups=("isolate",))
def mutgap2_rdec_rate_map_reset_8_bytes():
    """66 on a mutable map whose object-list record is exactly 8 bytes, the
    header length the reset checks for, starting with an operation other than
    0x32: the reset returns at once with success, and so does a second
    one."""
    from scenarios.exec import RateMaps
    s = Scenario("mutgap2_rdec_rate_map_reset_8_bytes", setups=("isolate",),
                 doc=mutgap2_rdec_rate_map_reset_8_bytes.__doc__)
    s.bring_up()
    m = RateMaps(s)
    t = m.t
    m.create(m.mut, "create the mutable map")
    t.write(m.mut_va, struct.pack("<I", 51))
    typ, n, va = t.ol.entries[m.mut]
    t.ol.entries[m.mut] = (typ, 8, va)
    t.ol.write()
    m.reset(m.mut, label="66 with an 8-byte record, operation 51")
    m.reset(m.mut, label="66 again")
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def mutgap2_rdec_use_heaps_count_placement():
    """Commands that look objects up, checked before they do, in a service
    task with a placement list (opcode 68) whose entries start zero: an
    object nothing has made yet (none is listed in the exec) is made by its
    first lookup and writes its placement entry (a heap or buffer its GPU
    address; an argument-buffer sampler its resourceIndex and ID). 0x86
    UseHeaps, 0x1b UseHeapsWithStages, 0x87 UseResources and 0x89
    UseResourcesWithStages with count 2 and one ref: the size check throws
    "Too many resources specified" before any lookup, so each exec faults and
    its object's entry stays zero. 0x8e SetFragmentSamplerState at index 16
    (maxVertexSamplers) raises "Invalid sampler index" before its lookup: the
    entry stays zero; at index 0 the sampler is looked up (its entry written)
    and the exec faults at the setter (docs/testing.md 6.16: this encoder has
    no lodBias setter). Last, an exec that lists heap 1 and names it in 0x86
    makes it. Isolate only: placement entries are written only in the
    service. host: the written values are the host's."""
    from scenarios.textures_placement import install_with_placements
    from pgtest.task import buffer_record
    s = Scenario("mutgap2_rdec_use_heaps_count_placement", setups=("isolate",), host=True,
                 doc=mutgap2_rdec_use_heaps_count_placement.__doc__)
    s.bring_up()
    t = R.RenderTask(s)
    tex, _, va = t.target(W, H)
    objs = {}
    objs["heap 1"] = t.heap()
    objs["heap 2"] = t.heap()
    for name in ("buffer 1", "buffer 2"):
        objs[name] = t.ol.add(abi.T_BUFFER, buffer_record(PAGE, t.reserve(PAGE) // PAGE))
    objs["sampler 1"] = t.sampler(argbuf=1)
    objs["sampler 2"] = t.sampler(argbuf=1)
    t.define()
    pl = install_with_placements(t, {})
    p = R.describe_pass({0: dict(tex=tex, clear=(0.0, 0.0, 0.0, 1.0))}, width=W, height=H)

    def entries(label):
        for name, ref in objs.items():
            t.dump(pl + 16 * ref, 16, "%s: %s's placement entry" % (label, name))
    entries("before")
    for label, r in (("0x86 count 2, [heap 1]", R.use_heaps([objs["heap 1"]], count=2)),
                     ("0x1b count 2, [heap 2]", R.use_heaps_stages([objs["heap 2"]], 1, count=2)),
                     ("0x87 count 2, [buffer 1]", R.use_resources([objs["buffer 1"]], 1, count=2)),
                     ("0x89 count 2, [buffer 2]", R.use_resources_stages([objs["buffer 2"]], 1, 1, count=2)),
                     ("0x8e sampler 2 at index 16", R.sampler_state(0x8E, objs["sampler 2"], 0.0, 1000.0, 0.0, 16)),
                     ("0x8e sampler 1 at index 0", R.sampler_state(0x8E, objs["sampler 1"], 0.0, 1000.0, 0.0, 0))):
        t.render([p, r], fault=True, label=label)
        s.fault_state(label)
        s.resume_root()
    entries("after the faults")
    t.render([p, R.use_heaps([objs["heap 1"]])], resources=t.bufs + [objs["heap 1"]],
             label="0x86 [heap 1], heap 1 listed")
    s.fault_state("0x86 [heap 1], heap 1 listed")
    t.dump(pl + 16 * objs["heap 1"], 16, "after heap 1 is made: its placement entry")
    t.delete()
    s.events("end")
    return s


def eight_two_level(st):
    """An 8x8 RGBA8 texture of two levels (type 2): level 0 red, level 1
    (4x4) green; listed in every exec. render_ftex's coordinates step two
    texels a pixel on it, so its implicit LOD is 1."""
    from pgtest.resources_build import desc1, simple_dim, texture_rec
    t = st.t
    va = t.reserve(PAGE)
    t.write(va, bytes.fromhex("ff0000ff") * 64 + bytes.fromhex("00ff00ff") * 16)
    tex = t.ol.add(abi.T_TEXTURE, texture_rec(PAGE, va // PAGE, [simple_dim(8, 8, levels=2)],
                                              desc1(w=8, h=8, mips=2, usage=1)))
    t.bufs.append(tex)
    return tex


@scenario
def mutgap2_rdec_sampler_bindings():
    """Vertex and fragment sampler bindings over two-level textures (level 0
    red, level 1 green) through a mipmapped sampler: each binding replaces
    the one before it in the pass, and each LOD clamp reaches Metal, also
    after an earlier binding in the same segment left other clamps.
    render_vtex samples level 0, fixtures/shaders/rdec_vtex1.metal level 1,
    render_ftex at LOD 0 on the 4x4 texture and LOD 1 on the 8x8 one. 0x80
    (0, 0) then (1, 1): green; 0x71 likewise; the maximum clamp: (0, 0) red,
    (0, 0) then (0, 1000) green, for 0x80 at level 1 and 0x71 at LOD 1; 0x80
    (1, 1) then 0x7f SetVertexSamplerStates with the plain sampler: red; 0x7f
    with a sampler whose lodMinClamp is 1: green, then 0x80 (0, 0) over it:
    red. And two draws in one pass, scissored (0x75) to the left and right
    halves, with different minimum clamps: each half shows its own."""
    s = Scenario("mutgap2_rdec_sampler_bindings", doc=mutgap2_rdec_sampler_bindings.__doc__)
    s.bring_up()
    st = StateTask(s)
    t = st.t
    tex4, smp = two_level(st)
    tex8 = eight_two_level(st)
    smp1 = t.sampler(mip=1, lmin=1.0)
    p_vtex1 = t.pipeline("basic", v=t.function("shaders/rdec_vtex1.metallib"), f=st.fn["render_color"])
    st.start()
    p = R.pipeline_state
    d = R.draw16(TRI, 0, 3)
    v0 = [p(st.p_vtex), R.refs_range(0x81, 0, [tex4])]
    v1 = [p(p_vtex1), R.refs_range(0x81, 0, [tex4])]
    f0 = [p(st.p_ftex), R.refs_range(0x72, 0, [tex4])]
    f1 = [p(st.p_ftex), R.refs_range(0x72, 0, [tex8])]
    lod = lambda cmd, lo, hi: R.samplers_lod(cmd, 0, [(smp, lo, hi)])
    for label, recs in (
            ("0x80 (0, 0) then (1, 1), level 0", v0 + [lod(0x80, 0.0, 0.0), lod(0x80, 1.0, 1.0), d]),
            ("0x71 (0, 0) then (1, 1), LOD 0", f0 + [lod(0x71, 0.0, 0.0), lod(0x71, 1.0, 1.0), d]),
            ("0x80 (0, 0), level 1", v1 + [lod(0x80, 0.0, 0.0), d]),
            ("0x80 (0, 0) then (0, 1000), level 1", v1 + [lod(0x80, 0.0, 0.0), lod(0x80, 0.0, 1000.0), d]),
            ("0x71 (0, 0), LOD 1", f1 + [lod(0x71, 0.0, 0.0), d]),
            ("0x71 (0, 0) then (0, 1000), LOD 1", f1 + [lod(0x71, 0.0, 0.0), lod(0x71, 0.0, 1000.0), d]),
            ("0x71 (0, 0) on the left half, (1, 1) on the right, LOD 0",
             f0 + [R.scissor(0, 0, 4, 8), lod(0x71, 0.0, 0.0), d, R.scissor(4, 0, 4, 8), lod(0x71, 1.0, 1.0), d]),
            ("0x71 (1, 1) on the left half, (0, 0) on the right, LOD 0",
             f0 + [R.scissor(0, 0, 4, 8), lod(0x71, 1.0, 1.0), d, R.scissor(4, 0, 4, 8), lod(0x71, 0.0, 0.0), d]),
            ("0x80 (0, 0) on the left half, (1, 1) on the right, level 0",
             v0 + [R.scissor(0, 0, 4, 8), lod(0x80, 0.0, 0.0), d, R.scissor(4, 0, 4, 8), lod(0x80, 1.0, 1.0), d]),
            ("0x80 (1, 1) then 0x7f plain, level 0", v0 + [lod(0x80, 1.0, 1.0), R.refs_range(0x7F, 0, [smp]), d]),
            ("0x7f lodMinClamp 1, level 0", v0 + [R.refs_range(0x7F, 0, [smp1]), d]),
            ("0x7f lodMinClamp 1 then 0x80 (0, 0), level 0", v0 + [R.refs_range(0x7F, 0, [smp1]), lod(0x80, 0.0, 0.0),
                                                                   d])):
        case(s, st, label, recs)
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap2_rdec_draw_state():
    """Encoder state a draw shows: 0x90 SetProvokingVertexMode with a
    flat-interpolated colour per vertex (fixtures/shaders/rdec_flat_v.metal
    and rdec_flat_f.metal; vertices red, green, blue), without 0x90 and with
    modes 0 and 1; 0x7b SetTessellationFactorScale on a triangle patch whose
    factors are 1 (render_draw_patches' pipeline), without it and with scales
    0 and 2; 0x0d DrawPatches16 through a patch index buffer [1, 0] at offset
    0: patch 1, B. Measured: without 0x90 the first vertex provokes (red),
    mode 1 the last (blue); the factor scale changes nothing drawn on this
    host, 0 included. Then 0x15 ExecuteCommandsInBufferRanged with ICB 0 and
    range {0, 0}: the nil ICB throws "Failed to get indirect command buffer
    for ref 0" before the range is checked, and the exec faults."""
    from scenarios.render_draw import DrawTask
    s = Scenario("mutgap2_rdec_draw_state", doc=mutgap2_rdec_draw_state.__doc__)
    s.bring_up()
    d = DrawTask(s)
    t = d.t
    pf = t.pipeline("basic", v=t.function("shaders/rdec_flat_v.metallib"),
                    f=t.function("shaders/rdec_flat_f.metallib"))
    rgb, _ = t.buffer(contents=R.float4s((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)))
    t.define()
    t.install()

    def draw(label, recs):
        t.fill(d.va, NPIX, 0x11)
        t.render([d.pass_()] + recs, label=label)
        s.fault_state(label)
        t.dump(d.va, NPIX, label)
    flat = [R.pipeline_state(pf), R.buffers(0x7D, 0, [(d.pos, 0), (rgb, 0)])]
    tri = R.draw16(TRI, 0, 3)
    draw("flat colours, no 0x90", flat + [tri])
    for mode in (0, 1):
        draw("flat colours, 0x90 %d" % mode, flat + [R.u32(0x90, mode), tri])
    patch = R.draw_patches16(3, 0, 1, 0, 0, 1, 0)
    draw("patch 0, no 0x7b", d.tess() + [patch])
    for scale in (0.0, 2.0):
        draw("patch 0, 0x7b %g" % scale, d.tess() + [R.tess_factor_scale(scale), patch])
    draw("0x0d patch index buffer at 0: [1]", d.tess() + [R.draw_patches16(3, 0, 1, d.patch_idx, 0, 1, 0)])
    t.render([d.pass_(), R.exec_icb_ranged(0, 0, 0)], fault=True, label="0x15 ICB 0, range {0, 0}")
    s.fault_state("0x15 ICB 0, range {0, 0}")
    s.resume_root()
    draw("a draw after it", [R.pipeline_state(d.p), d.vb(), tri])
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",))
def mutgap2_rdec_rate_map_reset_short_entry():
    """66 reads the map's object-list entry again, and its cursor runs from
    12 bytes in to 4 bytes past the entry's length. The mutable map's 48-byte
    record listed as 44 bytes: the cursor ends exactly where the samples end,
    and the reset succeeds; as 43 bytes it is one byte short, "Failed to read
    samples" is thrown uncaught in the service, the service dies, and the 66
    faults; so does the next one. Isolate only: in-process 66 on an existing
    map aborts the host (exec_reset_rate_map_in_process)."""
    from scenarios.exec import RateMaps
    s = Scenario("mutgap2_rdec_rate_map_reset_short_entry", setups=("isolate",),
                 doc=mutgap2_rdec_rate_map_reset_short_entry.__doc__)
    s.bring_up()
    m = RateMaps(s)
    t = m.t
    assert len(m.rec_m) == 48
    m.create(m.mut, "create the mutable map")
    typ, n, va = t.ol.entries[m.mut]
    for length, fault in ((44, False), (43, True), (48, True)):
        t.ol.entries[m.mut] = (typ, length, va)
        t.ol.write()
        m.reset(m.mut, fault=fault, label="66, the entry %d bytes" % length)
    s.events("end")
    return s


@scenario(exclusive=True)
def mutgap2_rdec_primitive_type_9():
    """Primitive type 9 passes the decoder's primitive-type check (only >= 10
    throws), so the exec's command buffer is committed with a blit fill in
    its first segment and a pass drawing 0x01 with type 9 in its second, and
    Metal reports an error for it (render_draw_type9): the exec faults. The
    fill's buffer and the target show what of the committed buffer ran.
    Measured: the fill lands, the pass's clear does not. exclusive: the draw
    faults the GPU."""
    from scenarios.render_draw import DrawTask
    s = Scenario("mutgap2_rdec_primitive_type_9", exclusive=True, doc=mutgap2_rdec_primitive_type_9.__doc__)
    s.bring_up()
    d = DrawTask(s)
    t = d.t
    out, out_va = t.buffer()
    t.define()
    t.install()
    label = "0x01 type 9"
    t.fill(d.va, NPIX, 0x11)
    t.fill(out_va, 8, 0x11)
    t.exec_segments([seg(abi.ENC_BLIT, B.fill(out, 0, 8, 0x5A)),
                     seg(abi.ENC_RENDER, d.pass_() + R.pipeline_state(d.p) + d.vb() + R.draw16(9, 0, 3))],
                    fault=True, timeout_ms=30000, label=label)
    s.fault_state(label)
    s.resume_root()
    t.dump(out_va, 8, label + ": the fill's buffer")
    t.dump(d.va, NPIX, label + ": the target")
    t.delete()
    s.events("end")
    return s
