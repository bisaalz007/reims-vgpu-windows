"""The decoders at their edges: each scenario locks one behaviour of a
decoder, as its docstring says; the docstrings say what the code does, which
is what the expected files record."""
import struct

from pgtest import abi
from pgtest import render_lib as R
from pgtest.compute_objects import Blit as B, Compute as C, Rig
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import rec, seg
from scenarios.blit_errors import Checks
from scenarios.exec import RateMaps
from scenarios.info import World
from scenarios.render_state import ALWAYS, EQUAL, NPIX, PRIVATE, TRI, W, H, StateTask, case
from scenarios.render_tile import TileTask, case as tile_case



# ---- the blit decoder ----

@scenario
def mutgap_dec_blit_overlong_after_unread():
    """A record whose length runs past the segment, after a record that left
    bytes unread: an UpdateFence (316) with 4 bytes of padding its 4-byte
    command never reads. The record's sub-cursor fails and the decoder throws
    "Failed to read" before any command sees the failed cursor: the exec
    faults, nothing of it is committed, and the task keeps working (a later
    fill lands)."""
    k = Checks("mutgap_dec_blit_overlong_after_unread", mutgap_dec_blit_overlong_after_unread.__doc__)
    k.fault("316 with 4 unread bytes, then a record past the segment",
            rec(316, struct.pack("<II", k.fence, 0)), rec(316, b"", length=0x40))
    k.r.blit(B.fill(k.out, 4, 4, 0x77), label="a fill after the fault")
    k.r.dump(k.out_va, 0x8, "out: only the later fill")
    return k.r.end()


@scenario
def mutgap_dec_blit_icb_ref0_empty_range():
    """ICB ref 0 with an empty range, which the range check would accept
    against a nil ICB's size 0: the decoder's own non-null getter throws "Got
    invalid indirect command buffer" first, for OptimizeIndirectCommandBuffer
    (312), ResetCommandsInBuffer (313) and CopyIndirectCommandBuffer's source
    (305), so each exec faults and none of it runs; the task keeps
    working."""
    k = Checks("mutgap_dec_blit_icb_ref0_empty_range", mutgap_dec_blit_icb_ref0_empty_range.__doc__)
    k.fault("312 ICB 0, range {0, 0}", B.icb_range(312, 0, 0, 0))
    k.fault("313 ICB 0, range {0, 0}", B.icb_range(313, 0, 0, 0))
    k.fault("305 source 0, range {0, 0} to index 0", B.copy_icb(0, k.icb, 0, 0, 0))
    k.r.blit(B.fill(k.out, 4, 4, 0x77), label="a fill after the faults")
    k.r.dump(k.out_va, 0x8, "out: only the later fill")
    return k.r.end()


# ---- the compute decoder ----

@scenario
def mutgap_dec_compute_barrier_256_257():
    """MemoryBarrierWithResources (0xd6) at the stack list's edge: 256 refs
    fill the 256-entry stack list and 257 take the heap-allocated one. Both
    succeed, and the dispatch after them writes its output."""
    r = Rig("mutgap_dec_compute_barrier_256_257", mutgap_dec_compute_barrier_256_257.__doc__)
    out, out_va = r.buffer()
    pso = r.kernel()
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0)]), C.barrier_resources([out] * 256),
              C.barrier_resources([out] * 257), C.dispatch_threads((8, 1, 1), (8, 1, 1)))
    r.dump(out_va, 0x24, "out")
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0x40)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)))
    r.dump(out_va + 0x40, 0x14, "out at 0x40: a later dispatch")
    return r.end()


# ---- the info decoder ----

@scenario
def mutgap_dec_info_parameter_offset_1():
    """453 CopyRasterizationRateParameterBuffer at offset 1: an odd offset
    fails the alignment mask (offset & (align - 1)), "Invalid buffer
    alignment", and the buffer is untouched."""
    s = Scenario("mutgap_dec_info_parameter_offset_1", doc=mutgap_dec_info_parameter_offset_1.__doc__)
    s.bring_up()
    w = World(s)
    w.t.fill(w.param_va, 0x10, 0xEE)
    w.fails("453 offset 1", [rec(453, struct.pack("<IIQ", w.rmap, w.param, 1))], resources=[(w.param, 0)],
            dump=(w.param_va, 0x10))
    s.events("end")
    return s


# ---- rate map resets ----

@scenario(setups=("isolate",))
def mutgap_dec_rate_map_reset_7_bytes():
    """66 on a mutable map whose object-list record is 7 bytes and starts
    with an operation other than 0x32: the header length check refuses
    anything under 8 bytes before the operation is looked at ("Expected
    Header of 8 bytes"), uncaught in the service, which dies: the 66 faults,
    and so does a second one. Isolate only: in-process 66 on an existing map
    aborts the host (exec_reset_rate_map_in_process)."""
    s = Scenario("mutgap_dec_rate_map_reset_7_bytes", setups=("isolate",),
                 doc=mutgap_dec_rate_map_reset_7_bytes.__doc__)
    s.bring_up()
    m = RateMaps(s)
    t = m.t
    m.create(m.mut, "create the mutable map")
    t.write(m.mut_va, struct.pack("<I", 51))
    typ, n, va = t.ol.entries[m.mut]
    t.ol.entries[m.mut] = (typ, 7, va)
    t.ol.write()
    m.reset(m.mut, fault=True, label="66 with a 7-byte record, operation 51")
    m.reset(m.mut, fault=True, label="66 again: the service is gone")
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def mutgap_dec_rate_map_reset_min_factor():
    """66 hands Metal a descriptor rebuilt from the record, with mutability 1
    and the record's minFactor only when the record says mutability 1. The
    mutable map (created by a 452 query) is reset from records that differ
    only there; after each, 452 (parameter size and alignment, granularity,
    physical size), 454 (200, 200) and 453 (the parameter data) show the map.
    Measured: before any reset the physical size and the mapping are 0;
    after, they follow the samples, and neither the record's mutability nor
    its minFactor changes them. host: every value is Metal's."""
    from pgtest.exec_objects import ObjTask, rate_map_record
    s = Scenario("mutgap_dec_rate_map_reset_min_factor", setups=("isolate",), host=True,
                 doc=mutgap_dec_rate_map_reset_min_factor.__doc__)
    s.bring_up()
    t = ObjTask(s)
    reply, reply_va = t.buffer(abi.PAGE)
    param, param_va = t.buffer(2 * abi.PAGE)
    rec_m = rate_map_record(w=256, h=256, mutable=1, min_factor=0.25, samples=[([0.25, 1.0], [0.25, 1.0])])
    mut = t.ol.add(abi.T_SERIALIZED, rec_m)
    mut_va = t.ol.entries[mut][2]
    t.define()
    t.install()
    res = [(reply, 0), (param, 0)]
    I = abi.ENC_INFO

    def show(i, label):
        t.fill(reply_va + 0x40 * i, 0x40, 0xEE)
        t.exec_segments([seg(I, rec(452, struct.pack("<IIQQ", mut, reply, 0x40 * i, 0))
                             + rec(454, struct.pack("<IIQIff", mut, reply, 0x40 * i + 0x20, 0, 200.0, 200.0))
                             + rec(453, struct.pack("<IIQ", mut, param, 0)))], resources=res, label=label)
        s.fault_state(label)
        t.dump(reply_va + 0x40 * i, 0x30, label + ": 452 and 454")
        t.dump(param_va, 0x6030, label + ": 453")
    show(0, "before")
    body = bytearray(rec_m)
    # After the 8-byte header: u32 unread, u16 w, u16 h, u32 layers,
    # u32 mutability @0x14, f32 minFactor @0x18 (exec_objects.rate_map_record);
    # the samples from 0x20.
    steps = [("mutability 1, minFactor 0.5, samples 0.25", 1, 0.5, (0.25, 1.0, 0.25, 1.0)),
             ("mutability 0, minFactor 0.5, samples 0.25", 0, 0.5, (0.25, 1.0, 0.25, 1.0)),
             ("mutability 1, minFactor 0.9, samples 0.25", 1, 0.9, (0.25, 1.0, 0.25, 1.0)),
             ("mutability 0, minFactor 0.9, samples 0.25", 0, 0.9, (0.25, 1.0, 0.25, 1.0))]
    for i, (label, m_, mf, samples) in enumerate(steps):
        struct.pack_into("<If", body, 0x14, m_, mf)
        struct.pack_into("<4f", body, 0x20, *samples)
        t.write(mut_va, bytes(body))
        s.submit(abi.OP_RESET_RASTERIZATION_RATE_MAP, struct.pack("<II", t.tid, mut), label="66: " + label)
        s.fault_state("66: " + label)
        show(i + 1, label)
    s.events("end")
    return s


# ---- the render decoder ----

@scenario
def mutgap_dec_render_use_heaps_stages():
    """0x1b UseHeaps with stages reads and looks up every ref it counts: with
    no 0x86 before it, count 0 is an empty list and the draw runs; a heap
    then a buffer fails on the buffer, and the exec faults."""
    s = Scenario("mutgap_dec_render_use_heaps_stages", doc=mutgap_dec_render_use_heaps_stages.__doc__)
    s.bring_up()
    st = StateTask(s)
    heap = st.t.heap()
    st.t.bufs.append(heap)
    st.start()
    draw = [R.pipeline_state(st.p_vbuf), st.vb(), R.draw16(TRI, 0, 3)]
    case(s, st, "0x1b no heaps", [R.use_heaps_stages([], 1)] + draw)
    case(s, st, "0x1b a heap then a buffer", [R.use_heaps_stages([heap, st.pos], 1)] + draw, fault=True)
    case(s, st, "0x1b a buffer then a heap", [R.use_heaps_stages([st.pos, heap], 1)] + draw, fault=True)
    case(s, st, "0x1b two heaps", [R.use_heaps_stages([heap, heap], 1)] + draw)
    st.t.delete()
    s.events("end")
    return s


@scenario
def mutgap_dec_render_overlong_after_unread():
    """The render counterpart of mutgap_dec_blit_overlong_after_unread:
    inside a pass, 0x74 SetRenderPipelineState with 4 bytes of padding it
    never reads, then a record whose length runs past the segment. The
    record's sub-cursor fails and "Failed to read" is thrown before any
    command sees the failed cursor: the exec faults, and the task keeps
    working (a later pass draws)."""
    s = Scenario("mutgap_dec_render_overlong_after_unread", doc=mutgap_dec_render_overlong_after_unread.__doc__)
    s.bring_up()
    st = StateTask(s)
    st.start()
    case(s, st, "0x74 with 4 unread bytes, then a record past the segment",
         [rec(0x74, struct.pack("<II", st.p_vbuf, 0)), rec(0x74, b"", length=0x40)], fault=True)
    case(s, st, "a draw after the fault", [R.pipeline_state(st.p_vbuf), st.vb(), R.draw16(TRI, 0, 3)])
    st.t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",))
def mutgap_dec_render_tile_dims_after_null_reply():
    """0xa4 GetTileDimensions with reply buffer 0 throws "Failed to get
    cursor for buffer", a decode fault the service survives: the next 0xa4
    writes its 16x16 reply. Isolate only: in-process a reply aborts the host
    (render_tile_dimensions_host)."""
    s = Scenario("mutgap_dec_render_tile_dims_after_null_reply", setups=("isolate",),
                 doc=mutgap_dec_render_tile_dims_after_null_reply.__doc__)
    s.bring_up()
    tt = TileTask(s)
    tt.start()
    tt.t.fill(tt.reply_va, 8, 0xEE)
    tile_case(s, tt, "reply buffer 0", [R.tile_dims(0, 0)], fault=True)
    tile_case(s, tt, "16x16 @0 after", [R.tile_dims(tt.reply, 0)], dump=False)
    tt.t.dump(tt.reply_va, 8, "reply")
    tt.t.delete()
    s.events("end")
    return s


@scenario
def mutgap_dec_render_stencil_store_action():
    """0x78 SetStencilStoreAction resolves a stencil store action of Unknown
    on a stored (private) attachment: pass 1 stores stencil 5, pass 2 clears
    it to 3 with store Unknown and 0x78 Store, pass 3 loads it and draws red
    with stencil Equal against reference 3, which passes; against 5 it does
    not. A pass 2 without 0x78 leaves stencil 5 (Unknown is not stored)."""
    s = Scenario("mutgap_dec_render_stencil_store_action", doc=mutgap_dec_render_stencil_store_action.__doc__)
    s.bring_up()
    st = StateTask(s)
    sten = st.t.depth_stencil(compare=ALWAYS, write=0, front=(EQUAL, 0, 0, 0, 0xFF, 0xFF),
                              back=(EQUAL, 0, 0, 0, 0xFF, 0xFF))
    # A private depth-stencil texture on a private heap, as render_state_depth
    # (type 8, commandType 21).
    heap = st.t.heap(length=16 * abi.PAGE)
    st.t.bufs.append(heap)
    private = st.t.ol.add(abi.T_SERIALIZER_TEXTURE, R.op_header(21) + struct.pack("<II", 0, heap) + R.tex_desc(
        W, H, R.D32S8, usage=R.USAGE_RT, options=PRIVATE) + struct.pack("<IQ", 0, 0))
    st.start()
    pd = [R.pipeline_state(st.p_depth)]
    d0 = [R.buffers(0x7D, 0, [(st.params, 0)]), R.draw16(TRI, 0, 3)]

    def ds(sload, sstore, sclear):
        return dict(depth=dict(tex=private, load=R.LOAD_CLEAR, store=R.STORE_DONTCARE, clear=1.0),
                    stencil=dict(tex=private, load=sload, store=sstore, clear=sclear))
    for tag, resolve in (("with 0x78 Store", [R.value(0x78, 1)]), ("without 0x78", [])):
        case(s, st, "store stencil 5", [], depth=True, pass_kw=ds(R.LOAD_CLEAR, R.STORE_STORE, 5), dump=False)
        case(s, st, "clear stencil to 3, store Unknown, " + tag, resolve, depth=True,
             pass_kw=ds(R.LOAD_CLEAR, R.STORE_UNKNOWN, 3), dump=False)
        for ref in (3, 5):
            case(s, st, "%s: load, stencil Equal %d" % (tag, ref),
                 pd + [R.depth_stencil_state(sten), R.stencil_ref(ref, ref)] + d0, depth=True,
                 pass_kw=ds(R.LOAD_LOAD, R.STORE_DONTCARE, 0))
    st.t.delete()
    s.events("end")
    return s


def private_ds(st, fmt=R.D32S8):
    """A private 8x8 depth-stencil texture at offset 0 of a private heap, as
    render_state_depth builds it (type 8, commandType 21: {u32, u32 heap, v1
    descriptor, u32, u64 offset}). The heap is listed in every exec."""
    heap = st.t.heap(length=16 * abi.PAGE)
    st.t.bufs.append(heap)
    return st.t.ol.add(abi.T_SERIALIZER_TEXTURE, R.op_header(21) + struct.pack("<II", 0, heap) + R.tex_desc(
        W, H, fmt, usage=R.USAGE_RT, options=PRIVATE) + struct.pack("<IQ", 0, 0))


def stencil_equal(st):
    return st.t.depth_stencil(compare=ALWAYS, write=0, front=(EQUAL, 0, 0, 0, 0xFF, 0xFF),
                              back=(EQUAL, 0, 0, 0, 0xFF, 0xFF))


@scenario
def mutgap_dec_render_memoryless_stencil_load():
    """A memoryless stencil attachment with load action Load: the decoder
    rewrites any load other than DontCare or Clear to Clear, so the stencil
    holds clearStencil 0xA5: a draw with stencil Equal passes against
    reference 0xA5 and fails against 0xA4. The depth attachment (the same
    texture) loads with Load too, rewritten the same way."""
    s = Scenario("mutgap_dec_render_memoryless_stencil_load", doc=mutgap_dec_render_memoryless_stencil_load.__doc__)
    s.bring_up()
    st = StateTask(s)
    sten = stencil_equal(st)
    st.start()
    kw = dict(depth=dict(tex=st.ds, load=R.LOAD_LOAD, store=R.STORE_DONTCARE, clear=1.0),
              stencil=dict(tex=st.ds, load=R.LOAD_LOAD, store=R.STORE_DONTCARE, clear=0xA5))
    d0 = [R.buffers(0x7D, 0, [(st.params, 0)]), R.draw16(TRI, 0, 3)]
    for ref in (0xA5, 0xA4):
        case(s, st, "stencil load Load, clear 0xA5, Equal 0x%x" % ref,
             [R.pipeline_state(st.p_depth), R.depth_stencil_state(sten), R.stencil_ref(ref, ref)] + d0,
             depth=True, pass_kw=kw)
    st.t.delete()
    s.events("end")
    return s


@scenario
def mutgap_dec_render_depth_stencil_cleared():
    """0x94 SetDepthCleared and 0x95 SetStencilCleared on a private
    attachment that holds loaded values: pass 1 stores depth 0.5 and stencil
    5; each later pass loads them and draws green at depth 0.7 with depth
    Less (fails), or red with stencil Equal 5 (passes), with and without the
    hint before the draw. Measured: the hints change nothing drawn on this
    host."""
    s = Scenario("mutgap_dec_render_depth_stencil_cleared", doc=mutgap_dec_render_depth_stencil_cleared.__doc__)
    s.bring_up()
    st = StateTask(s)
    less = st.t.depth_stencil(compare=1, write=1)
    sten = stencil_equal(st)
    private = private_ds(st)
    st.start()
    pd = [R.pipeline_state(st.p_depth)]
    d = lambda i: [R.buffers(0x7D, 0, [(st.params, 32 * i)]), R.draw16(TRI, 0, 3)]
    stored = dict(depth=dict(tex=private, load=R.LOAD_CLEAR, store=R.STORE_STORE, clear=1.0),
                  stencil=dict(tex=private, load=R.LOAD_CLEAR, store=R.STORE_STORE, clear=5))
    loaded = dict(depth=dict(tex=private, load=R.LOAD_LOAD, store=R.STORE_DONTCARE, clear=1.0),
                  stencil=dict(tex=private, load=R.LOAD_LOAD, store=R.STORE_DONTCARE, clear=3))
    for tag, hint in (("", []), (", 0x94", [rec(0x94)])):
        case(s, st, "store depth 0.5, stencil 5", pd + [R.depth_stencil_state(less)] + d(0), depth=True,
             pass_kw=stored, dump=False)
        case(s, st, "loaded%s: green (0.7) with Less" % tag, pd + [R.depth_stencil_state(less)] + hint + d(1),
             depth=True, pass_kw=loaded)
    for tag, hint in (("", []), (", 0x95", [rec(0x95)])):
        case(s, st, "store depth 0.5, stencil 5", pd + [R.depth_stencil_state(less)] + d(0), depth=True,
             pass_kw=stored, dump=False)
        case(s, st, "loaded%s: red with stencil Equal 5" % tag,
             pd + [R.depth_stencil_state(sten), R.stencil_ref(5, 5)] + hint + d(0), depth=True, pass_kw=loaded)
    st.t.delete()
    s.events("end")
    return s


@scenario
def mutgap_dec_render_fragment_vertex_barrier():
    """0x17 memoryBarrierWithScope:afterStages:beforeStages: between two
    draws of one pass: the first draw's fragments store 1 into a buffer and
    draw green everywhere; the second draw's vertices read that buffer and
    cover the target (orange) only if they see the 1 (fixtures/shaders/
    mutdec_fwrite.metal, mutdec_vread.metal). Measured: with a buffers
    barrier after the fragment stage and before the vertex stage, as without
    one, they do not (this tile-based GPU runs the pass's vertex stages
    first, and the barrier does not split the pass), so the target is green
    and the buffer holds 1 either way.
    """
    s = Scenario("mutgap_dec_render_fragment_vertex_barrier",
                 doc=mutgap_dec_render_fragment_vertex_barrier.__doc__)
    s.bring_up()
    st = StateTask(s)
    fw = st.t.function("shaders/mutdec_fwrite.metallib")
    fv = st.t.function("shaders/mutdec_vread.metallib")
    pa = st.t.pipeline("basic", v=st.fn["vert"], f=fw)
    pb = st.t.pipeline("basic", v=fv, f=st.fn["frag"])
    flag, flag_va = st.t.buffer()
    st.start()
    a = [R.pipeline_state(pa), R.buffers(0x6E, 0, [(flag, 0)]), R.draw16(TRI, 0, 3)]
    b = [R.pipeline_state(pb), R.buffers(0x7D, 0, [(flag, 0)]), R.draw16(TRI, 0, 3)]
    for label, mid in (("0x17 buffers, after fragment, before vertex", [R.barrier_scope(1, 2, 1)]),
                       ("no barrier", [])):
        st.t.fill(flag_va, 4, 0)
        case(s, st, label, a + mid + b)
        st.t.dump(flag_va, 4, label + ": the buffer")
    st.t.delete()
    s.events("end")
    return s


class Msaa:
    """A StateTask with 4-sample memoryless colour attachments (ms, ms2),
    two resolve targets over shared buffers (res, res2), the msaa4 pipeline
    (render_vbuf and render_color) and render_pass_msaa's red triangle with
    an edge across the target."""

    def __init__(self, s, stencil=False):
        st = self.st = StateTask(s)
        t = st.t
        self.p = t.pipeline("msaa4", v=st.fn["render_vbuf"], f=st.fn["render_color"])
        self.pos, _ = t.buffer(contents=R.float4s((-1.0, -1.0, 0.0, 1.0), (1.0, -0.6, 0.0, 1.0),
                                                  (-0.3, 1.0, 0.0, 1.0)))
        self.ms = t.memoryless(W, H, R.RGBA8, ttype=R.TEX_2D_MS, samples=4)
        self.ms2 = t.memoryless(W, H, R.RGBA8, ttype=R.TEX_2D_MS, samples=4)
        self.res, _, self.va = t.target(W, H)
        self.res2, _, self.va2 = t.target(W, H)
        if stencil:
            self.ds_ms = t.memoryless(W, H, R.D32S8, ttype=R.TEX_2D_MS, samples=4)
            self.private = private_ds(st)
            self.sten = stencil_equal(st)
        st.start()

    def draw(self):
        return [R.pipeline_state(self.p), R.buffers(0x7D, 0, [(self.pos, 0), (self.st.col, 0)]), R.draw16(TRI, 0, 3)]

    def run(self, s, label, colors, records, **kw):
        t = self.st.t
        t.fill(self.va, NPIX, 0x11)
        t.fill(self.va2, NPIX, 0x11)
        t.render([R.describe_pass(colors, width=W, height=H, **kw)] + records, label=label)
        s.fault_state(label)
        t.dump(self.va, NPIX, label + ": res")
        t.dump(self.va2, NPIX, label + ": res2")


@scenario(host=True)
def mutgap_dec_render_msaa_memoryless():
    """A 4-sample memoryless colour attachment resolved into a texture over a
    shared buffer, with a red triangle over part of it: load Load is
    rewritten to Clear, so the uncovered pixels resolve to the blue clear
    colour; store Unknown is kept and 0x66 SetColorStoreAction
    MultisampleResolve resolves it; 0x96 SetColorResolveTexture with a second
    attachment bound (so index 0 passes the >= max check) moves the resolve
    to res2. host: the edge pixels are the GPU's sample positions."""
    s = Scenario("mutgap_dec_render_msaa_memoryless", host=True, doc=mutgap_dec_render_msaa_memoryless.__doc__)
    s.bring_up()
    m = Msaa(s)
    blue = (0.0, 0.0, 1.0, 1.0)
    m.run(s, "load Load, store resolve", {0: dict(tex=m.ms, resolve=m.res, load=R.LOAD_LOAD, store=R.STORE_RESOLVE,
                                                   clear=blue)}, m.draw())
    m.run(s, "store Unknown, then 0x66 MultisampleResolve",
          {0: dict(tex=m.ms, resolve=m.res, store=R.STORE_UNKNOWN, clear=blue)},
          [R.color_store_action(R.STORE_RESOLVE, 0)] + m.draw())
    m.run(s, "two attachments, 0x96 res2 at index 0",
          {0: dict(tex=m.ms, resolve=m.res, store=R.STORE_RESOLVE, clear=blue),
           1: dict(tex=m.ms2, store=R.STORE_DONTCARE, clear=blue)},
          [R.color_resolve(m.res2, 0, 0, 0, 0, 0)] + m.draw())
    m.st.t.delete()
    s.events("end")
    return s


@scenario
def mutgap_dec_render_stencil_resolve():
    """A 4-sample memoryless depth-stencil attachment's stencil, cleared to 3
    and stored MultisampleResolve into a 1-sample private D32S8 texture named
    as its resolve texture: a later pass loads the resolved stencil and draws
    red with stencil Equal 3 (drawn) and 5 (not), after a pass that stored 5
    there."""
    s = Scenario("mutgap_dec_render_stencil_resolve", doc=mutgap_dec_render_stencil_resolve.__doc__)
    s.bring_up()
    m = Msaa(s, stencil=True)
    st = m.st
    pd = [R.pipeline_state(st.p_depth)]
    d0 = [R.buffers(0x7D, 0, [(st.params, 0)]), R.draw16(TRI, 0, 3)]
    for ref in (3, 5):
        case(s, st, "store stencil 5 in the private texture", [], depth=True, dump=False,
             pass_kw=dict(depth=dict(tex=m.private, load=R.LOAD_CLEAR, store=R.STORE_STORE, clear=1.0),
                          stencil=dict(tex=m.private, load=R.LOAD_CLEAR, store=R.STORE_STORE, clear=5)))
        st.t.render([R.describe_pass({0: dict(tex=m.ms, resolve=m.res, store=R.STORE_RESOLVE,
                                              clear=(0.0, 0.0, 1.0, 1.0))}, width=W, height=H,
                                     stencil=dict(tex=m.ds_ms, resolve=m.private, load=R.LOAD_CLEAR,
                                                  store=R.STORE_RESOLVE, clear=3))],
                    label="4-sample stencil 3 resolved into the private texture")
        case(s, st, "load it, stencil Equal %d" % ref, pd + [R.depth_stencil_state(m.sten), R.stencil_ref(ref, ref)]
             + d0, depth=True,
             pass_kw=dict(depth=dict(tex=m.private, load=R.LOAD_CLEAR, store=R.STORE_DONTCARE, clear=1.0),
                          stencil=dict(tex=m.private, load=R.LOAD_LOAD, store=R.STORE_DONTCARE, clear=0)))
    st.t.delete()
    s.events("end")
    return s


@scenario
def mutgap_dec_render_tile_threadgroup_memory():
    """0x9c SetThreadgroupMemoryLength in a render pass
    (setThreadgroupMemoryLength:offset:atIndex:) places a tile function's
    threadgroup memory in what 0x23 reserved for the pass
    (threadgroupMemoryLength). fixtures/shaders/mutdec_tiletg.metal goes
    through 2 KiB of it and writes blue back to colour 0, over a red clear.
    Then, in one pass of 4 KiB, three tile dispatches: blue stored at offset
    0, green at offset 2048, and a read at offset 0 (mutdec_tgwrite_blue,
    mutdec_tgwrite_green, mutdec_tgread): the pass's threadgroup memory
    persists across its tile dispatches, and the read finds blue."""
    s = Scenario("mutgap_dec_render_tile_threadgroup_memory", doc=mutgap_dec_render_tile_threadgroup_memory.__doc__)
    s.bring_up()
    tt = TileTask(s)
    t = tt.t
    p = t.pipeline("tile", t=t.function("shaders/mutdec_tiletg.metallib"))
    pb = t.pipeline("tile", t=t.function("shaders/mutdec_tgwrite_blue.metallib"))
    pg = t.pipeline("tile", t=t.function("shaders/mutdec_tgwrite_green.metallib"))
    pr = t.pipeline("tile", t=t.function("shaders/mutdec_tgread.metallib"))
    tt.start()

    def run(label, tg, records):
        t.fill(tt.va, NPIX, 0x11)
        t.render([R.tile_size(16, 16), R.threadgroup_memory_length(tg),
                  R.describe_pass({0: dict(tex=tt.tex, clear=(1.0, 0.0, 0.0, 1.0))}, width=W, height=H)] + records,
                 label=label)
        s.fault_state(label)
        t.dump(tt.va, NPIX, label)
    run("0x23 2048, 0x9c 2048 at 0, mutdec_tiletg", 2048,
        [R.pipeline_state(p), R.tg_memory(2048, 0, 0), R.dispatch_tile(16, 16)])
    run("0x23 4096: blue at 0, green at 2048, read at 0", 4096,
        [R.pipeline_state(pb), R.tg_memory(2048, 0, 0), R.dispatch_tile(16, 16),
         R.pipeline_state(pg), R.tg_memory(2048, 2048, 0), R.dispatch_tile(16, 16),
         R.pipeline_state(pr), R.tg_memory(2048, 0, 0), R.dispatch_tile(16, 16)])
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap_dec_compute_imageblock():
    """0xd8 SetImageblockWidth (setImageblockWidth:height:; this GPU
    supportsImageBlocks) sizes a kernel's explicit imageblock:
    fixtures/shaders/mutdec_imageblock.metal writes get_width() and
    get_height() to its output; 8x8 then 16x16 over one 16x16 threadgroup."""
    r = Rig("mutgap_dec_compute_imageblock", mutgap_dec_compute_imageblock.__doc__)
    out, out_va = r.buffer()
    pso = r.kernel("shaders/mutdec_imageblock.metallib")
    r.ready()
    for i, (w, h) in enumerate(((8, 8), (16, 16))):
        r.t.fill(out_va + 8 * i, 8, 0xEE)
        r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 8 * i)]), C.imageblock(w, h),
                  C.dispatch_threadgroups((1, 1, 1), (16, 16, 1)), label="0xd8 %dx%d" % (w, h))
    r.dump(out_va, 0x10, "out")
    return r.end()


@scenario
def mutgap_dec_render_tile_sampler_lod():
    """0xa0 SetTileSamplerStatesLODClamp
    (setTileSamplerStates:lodMinClamps:lodMaxClamps:withRange:) with a tile
    function that samples at level 0 (fixtures/shaders/
    mutdec_tilesample.metal) a two-level texture (type 2, 4x4 RGBA8: level 0
    red, level 1 green) through a mipmapped sampler at index 1: clamps (1, 1)
    give green, (0, 0) and (0, 1000) red."""
    from pgtest.resources_build import desc1, simple_dim, texture_rec
    s = Scenario("mutgap_dec_render_tile_sampler_lod", doc=mutgap_dec_render_tile_sampler_lod.__doc__)
    s.bring_up()
    tt = TileTask(s)
    t = tt.t
    fn = t.function("shaders/mutdec_tilesample.metallib")
    p = t.pipeline("tile", t=fn)
    smp = t.sampler(mip=1)
    va = t.reserve(abi.PAGE)
    t.write(va, bytes.fromhex("ff0000ff") * 16 + bytes.fromhex("00ff00ff") * 4)
    tex = t.ol.add(abi.T_TEXTURE, texture_rec(abi.PAGE, va // abi.PAGE, [simple_dim(4, 4, levels=2)],
                                              desc1(w=4, h=4, mips=2, usage=1)))
    t.bufs.append(tex)
    tt.start()
    for lo, hi in ((1.0, 1.0), (0.0, 0.0), (0.0, 1000.0)):
        tile_case(s, tt, "0xa0 sampler 1 clamps (%g, %g)" % (lo, hi),
                  [R.pipeline_state(p), R.refs_range(0xA1, 0, [tex]), R.samplers_lod(0xA0, 1, [(smp, lo, hi)]),
                   R.dispatch_tile(16, 16)])
    t.delete()
    s.events("end")
    return s


@scenario(exclusive=True)
def mutgap_dec_render_primitive_type_10():
    """Primitive type 10 in 0x02 DrawInstancedPrimitives64 and 0x04
    DrawInstancedBasePrimitives64: the decoder's primitive-type check throws
    "Invalid primitive type", a decode error, so the exec's command buffer is
    never committed (docs/testing.md, 6.12): neither the blit fill in the
    exec's first segment nor the pass's clear lands. exclusive: a draw that
    reaches Metal with type 10 faults the GPU (render_draw_type9)."""
    from scenarios.render_draw import DrawTask
    s = Scenario("mutgap_dec_render_primitive_type_10", exclusive=True, doc=mutgap_dec_render_primitive_type_10.__doc__)
    s.bring_up()
    d = DrawTask(s)
    t = d.t
    out, out_va = t.buffer()
    t.define()
    t.install()
    for label, draw in (("0x02 type 10", R.draw_inst64(10, 0, 3, 1)),
                        ("0x04 type 10", R.draw_inst_base64(10, 0, 3, 1, 0))):
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


@scenario
def mutgap_dec_render_fault_releases_encoder():
    """200 render execs, each faulting after its encoder exists (0x16
    RenderBarrierResources is always unsupported), then a pass that draws:
    the decoder's fault path ends and releases the encoder, and more faults
    than a command queue's 64 command buffers leave the next exec to
    complete. Measured: without the release (a leaked, ended encoder per
    fault) the draw completes as well, so the leak is not visible to the
    guest here."""
    s = Scenario("mutgap_dec_render_fault_releases_encoder", doc=mutgap_dec_render_fault_releases_encoder.__doc__)
    s.bring_up()
    st = StateTask(s)
    st.start()
    for i in range(200):
        st.t.render([st.pass_(), rec(0x16, bytes(8))], fault=True, label="fault %d" % i)
        s.resume_root()
    s.fault_state("after 200 faults")
    case(s, st, "a draw after them", [R.pipeline_state(st.p_vbuf), st.vb(), R.draw16(TRI, 0, 3)])
    st.t.delete()
    s.events("end")
    return s


@scenario
def mutgap_dec_render_default_color_samples():
    """0x1f SetDefaultColorSampleCount in the pass prefix ([descriptor
    setDefaultColorSampleCount:]), with counts that disagree with the pass: 4
    on a pass with a 1-sample colour attachment, then 4 (with 0x1e 4) on a
    pass with no attachments, sized 8x8, drawing with a 4-sample pipeline
    with no colour attachment. Measured: both complete, and the first draws
    what a count of 1 draws; the count changes nothing observable on this
    host."""
    s = Scenario("mutgap_dec_render_default_color_samples", doc=mutgap_dec_render_default_color_samples.__doc__)
    s.bring_up()
    st = StateTask(s)
    p4 = st.t.pipeline("nocolor4", v=st.fn["vert"])
    st.start()
    t = st.t
    t.fill(st.va, NPIX, 0x11)
    t.render([R.default_color_samples(4), st.pass_(), R.pipeline_state(st.p_vbuf), st.vb(), R.draw16(TRI, 0, 3)],
             label="0x1f 4, 1-sample attachment")
    s.fault_state("0x1f 4, 1-sample attachment")
    t.dump(st.va, NPIX, "0x1f 4, 1-sample attachment")
    t.render([R.default_raster_samples(4), R.default_color_samples(4), R.describe_pass({}, width=W, height=H),
              R.pipeline_state(p4), R.draw16(TRI, 0, 3)], label="0x1e 4, 0x1f 4, no attachments")
    s.fault_state("0x1e 4, 0x1f 4, no attachments")
    st.t.delete()
    s.events("end")
    return s
