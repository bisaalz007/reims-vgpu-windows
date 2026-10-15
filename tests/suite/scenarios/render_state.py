"""Render encoder state 0x65-0xa6 (docs/testing.md, section 6.16): each
command set, and where the state changes what the GPU draws, drawn into a
texture over a shared buffer and read from guest memory; each check hit."""
import struct

from pgtest import abi
from pgtest import render_lib as R
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import rec

W = H = 8
NPIX = W * H * 4
TRI = 3
A = [(-1.0, -1.0, 0.0, 1.0), (0.0, -1.0, 0.0, 1.0), (-1.0, 1.0, 0.0, 1.0)]    # counter-clockwise
A_CW = [(-1.0, -1.0, 0.0, 1.0), (-1.0, 1.0, 0.0, 1.0), (0.0, -1.0, 0.0, 1.0)]  # clockwise
B = [(0.0, 1.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0), (1.0, -1.0, 0.0, 1.0)]
RED, GREEN, BLUE, WHITE = (1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0)
CLAMP, MIRROR_CLAMP, REPEAT, MIRROR_REPEAT, CLAMP_ZERO = 0, 1, 2, 3, 4
NEAREST, LINEAR = 0, 1
LESS, LESS_EQUAL, EQUAL, ALWAYS = 1, 3, 2, 7
PRIVATE = 0x20  # MTLResourceStorageModePrivate


class StateTask:
    def __init__(self, s, pages=64):
        t = self.t = R.RenderTask(s, pages=pages)
        fn = {n: t.function("shaders/%s.metallib" % n) for n in (
            "vert", "frag", "render_vbuf", "render_color", "render_fbuf", "render_ftex", "render_vtex",
            "render_vdesc", "render_depth")}
        self.fn = fn
        self.p_vbuf = t.pipeline("basic", v=fn["render_vbuf"], f=fn["render_color"])
        self.p_fbuf = t.pipeline("basic", v=fn["vert"], f=fn["render_fbuf"])
        self.p_blend = t.pipeline("blend", v=fn["vert"], f=fn["render_fbuf"])
        self.p_ftex = t.pipeline("basic", v=fn["vert"], f=fn["render_ftex"])
        self.p_vtex = t.pipeline("basic", v=fn["render_vtex"], f=fn["render_color"])
        self.p_vdesc = t.pipeline("vdesc", v=fn["render_vdesc"], f=fn["render_color"])
        self.p_depth = t.pipeline("depth", v=fn["render_depth"], f=fn["render_color"])
        self.p_basic = t.pipeline("basic", v=fn["vert"], f=fn["frag"])
        self.tex, _, self.va = t.target(W, H)
        self.pos, _ = t.buffer(contents=R.float4s(*(A + B + A_CW + R.FULL)))
        self.col, _ = t.buffer(contents=R.float4s(RED, GREEN, BLUE, WHITE))
        # render_depth's Params {float4 z, float4 colour}: z 0.5 red, 0.7 green,
        # 1.5 blue, 0.5 green, 0.45 white.
        params = b"".join(R.float4s((z, 0.0, 0.0, 0.0), c) for z, c in (
            (0.5, RED), (0.7, GREEN), (1.5, BLUE), (0.5, GREEN), (0.45, WHITE)))
        self.params, _ = t.buffer(contents=params)
        # A 2x2 source texture over a buffer: red, green / blue, white.
        self.src, _, sva = t.target(2, 2, usage=R.USAGE_READ,
                                    contents=bytes.fromhex("ff0000ff00ff00ff") + bytes(8) + bytes.fromhex(
                                        "0000ffffffffffff") + bytes(8))
        # render_vdesc's vertices: float2 position, uchar4 colour, stride 12.
        vd = b"".join(struct.pack("<ff4B", x, y, *c) for (x, y), c in (
            ((-1.0, -1.0), (255, 0, 0, 255)), ((1.0, -1.0), (0, 255, 0, 255)), ((-1.0, 1.0), (0, 0, 255, 255))))
        self.vd, _ = t.buffer(contents=vd)
        self.ds = t.memoryless(W, H, R.D32S8)
        self.samplers = {}

    def sampler(self, name, **kw):
        self.samplers[name] = self.t.sampler(**kw)
        return self.samplers[name]

    def pass_(self, with_ds=False, **kw):
        extra = {}
        if with_ds:
            extra = dict(depth=dict(tex=self.ds, load=R.LOAD_CLEAR, store=R.STORE_DONTCARE, clear=1.0),
                         stencil=dict(tex=self.ds, load=R.LOAD_CLEAR, store=R.STORE_DONTCARE, clear=3))
            extra.update(kw)
        return R.describe_pass({0: dict(tex=self.tex, clear=(0.0, 0.0, 0.0, 1.0))}, width=W, height=H, **extra)

    def vb(self):
        return R.buffers(0x7D, 0, [(self.pos, 0), (self.col, 0)])

    def start(self):
        self.t.define()
        self.t.install()


def case(s, st, label, records, fault=False, dump=True, depth=False, pass_kw=None):
    t = st.t
    if dump and not fault:
        t.fill(st.va, NPIX, 0x11)
    t.render([st.pass_(with_ds=depth, **(pass_kw or {}))] + records, fault=fault, label=label)
    if fault:
        s.fault_state(label)
        s.resume_root()
    elif dump:
        t.dump(st.va, NPIX, label)


@scenario
def render_state_vertex():
    """Vertex-stage bindings, drawn: 0x7d SetVertexBuffers (positions from
    buffer 0, colours from buffer 1), 0x7e SetVertexBufferOffset (buffer 0
    from vertex 3: B instead of A), 0xa5 SetVertexBuffersWithStride and 0xa6
    SetVertexBufferOffsetWithStride (the host supports dynamic attribute
    strides), a vertex-descriptor pipeline fed from one interleaved buffer,
    and 0x81 SetVertexTextures with 0x7f and 0x80 samplers (render_vtex reads
    the texel at (0.75, 0.25))."""
    s = Scenario("render_state_vertex", doc=render_state_vertex.__doc__)
    s.bring_up()
    st = StateTask(s)
    near = st.sampler("near")
    st.start()
    p = R.pipeline_state
    case(s, st, "0x7d: A", [p(st.p_vbuf), st.vb(), R.draw16(TRI, 0, 3)])
    case(s, st, "0x7e offset 48: B", [p(st.p_vbuf), st.vb(), R.buffer_offset(0x7E, 0, 48), R.draw16(TRI, 0, 3)])
    case(s, st, "0x7d at 1, then at 0 with offset 16", [p(st.p_vbuf), R.buffers(0x7D, 1, [(st.col, 16)]),
                                                        R.buffers(0x7D, 0, [(st.pos, 16)]), R.draw16(TRI, 0, 3)])
    case(s, st, "0xa5 strides: A", [p(st.p_vbuf), R.buffers_stride(0, [(st.pos, 0, 16), (st.col, 0, 16)]),
                                   R.draw16(TRI, 0, 3)])
    case(s, st, "0xa6 offset 48, stride 16: B", [p(st.p_vbuf), st.vb(), R.buffer_offset_stride(0, 48, 16),
                                                R.draw16(TRI, 0, 3)])
    case(s, st, "vertex descriptor", [p(st.p_vdesc), R.buffers(0x7D, 0, [(st.vd, 0)]), R.draw16(TRI, 0, 3)])
    case(s, st, "0x81 + 0x7f: texel (1, 0)", [p(st.p_vtex), R.refs_range(0x81, 0, [st.src]),
                                             R.refs_range(0x7F, 0, [near]), R.draw16(TRI, 0, 3)])
    case(s, st, "0x81 + 0x80 LOD clamp", [p(st.p_vtex), R.refs_range(0x81, 0, [st.src]),
                                         R.samplers_lod(0x80, 0, [(near, 0.0, 0.0)]), R.draw16(TRI, 0, 3)])
    st.t.delete()
    s.events("end")
    return s


@scenario
def render_state_fragment():
    """Fragment-stage bindings, drawn: 0x6e SetFragmentBuffers and 0x6f
    SetFragmentBufferOffset (render_fbuf returns buffer 0's first float4),
    0x72 SetFragmentTextures with 0x70 samplers of each address mode and
    filter, and 0x71 LOD-clamped samplers (render_ftex samples from -0.5 to
    1.5 across the target), 0x65 SetBlendColor with a pipeline that blends
    by it."""
    s = Scenario("render_state_fragment", doc=render_state_fragment.__doc__)
    s.bring_up()
    st = StateTask(s)
    smp = [(name, st.sampler(name, min_f=f, mag_f=f, s=a, t=a)) for name, f, a in (
        ("nearest clamp", NEAREST, CLAMP), ("nearest repeat", NEAREST, REPEAT),
        ("nearest mirror repeat", NEAREST, MIRROR_REPEAT), ("nearest clamp to zero", NEAREST, CLAMP_ZERO),
        ("linear clamp", LINEAR, CLAMP), ("linear mirror clamp", LINEAR, MIRROR_CLAMP))]
    st.start()
    p = R.pipeline_state
    case(s, st, "0x6e: red", [p(st.p_fbuf), R.buffers(0x6E, 0, [(st.col, 0)]), R.draw16(TRI, 0, 3)])
    case(s, st, "0x6e offset 32: blue", [p(st.p_fbuf), R.buffers(0x6E, 0, [(st.col, 32)]), R.draw16(TRI, 0, 3)])
    case(s, st, "0x6f offset 16: green", [p(st.p_fbuf), R.buffers(0x6E, 0, [(st.col, 0)]),
                                          R.buffer_offset(0x6F, 0, 16), R.draw16(TRI, 0, 3)])
    for name, ref in smp:
        case(s, st, "0x70 " + name, [p(st.p_ftex), R.refs_range(0x72, 0, [st.src]), R.refs_range(0x70, 0, [ref]),
                                     R.draw16(TRI, 0, 3)])
    case(s, st, "0x71 nearest repeat", [p(st.p_ftex), R.refs_range(0x72, 0, [st.src]),
                                        R.samplers_lod(0x71, 0, [(smp[1][1], 0.0, 4.0)]), R.draw16(TRI, 0, 3)])
    case(s, st, "no sampler", [p(st.p_ftex), R.refs_range(0x72, 0, [st.src]), R.draw16(TRI, 0, 3)])
    case(s, st, "0x65 blend colour (0.5, 0.25, 1, 0.5) over green", [
        p(st.p_blend), R.buffers(0x6E, 0, [(st.col, 16)]), R.blend_color(0.5, 0.25, 1.0, 0.5), R.draw16(TRI, 0, 3)],
         pass_kw=None)
    st.t.delete()
    s.events("end")
    return s


@scenario
def render_state_raster():
    """Rasterizer state, drawn: 0x75 SetScissorRect and 0x76 SetScissorRects,
    0x82 SetViewport and 0x83 SetViewports, 0x7c SetTriangleFillMode lines,
    0x6b SetCullMode with 0x73 SetFrontFacingWinding over a clockwise and a
    counter-clockwise triangle, 0x74 twice with the same pipeline (kept) and
    with another (replaced)."""
    s = Scenario("render_state_raster", doc=render_state_raster.__doc__)
    s.bring_up()
    st = StateTask(s)
    st.start()
    p = R.pipeline_state
    full = [p(st.p_vbuf), st.vb(), R.buffer_offset(0x7E, 0, 144)]
    case(s, st, "full target", full + [R.draw16(TRI, 0, 3)])
    case(s, st, "0x75 (1, 2, 3, 4)", full + [R.scissor(1, 2, 3, 4), R.draw16(TRI, 0, 3)])
    case(s, st, "0x76 one rect (4, 0, 4, 2)", full + [R.scissors([(4, 0, 4, 2)]), R.draw16(TRI, 0, 3)])
    case(s, st, "0x82 (2, 2, 4, 4)", full + [R.viewport(2.0, 2.0, 4.0, 4.0), R.draw16(TRI, 0, 3)])
    case(s, st, "0x83 one viewport (0, 4, 8, 4)", full + [R.viewports([(0.0, 4.0, 8.0, 4.0, 0.0, 1.0)]),
                                                         R.draw16(TRI, 0, 3)])
    case(s, st, "0x7c lines", [p(st.p_vbuf), st.vb(), R.value(0x7C, 1), R.draw16(TRI, 0, 3)])
    case(s, st, "0x7c fill", [p(st.p_vbuf), st.vb(), R.value(0x7C, 0), R.draw16(TRI, 0, 3)])
    for cull, wind in ((1, 0), (2, 0), (2, 1), (0, 1)):
        case(s, st, "0x6b cull %d, 0x73 winding %d: A ccw, A cw shifted" % (cull, wind),
             [p(st.p_vbuf), st.vb(), R.value(0x6B, cull), R.value(0x73, wind), R.draw16(TRI, 0, 3),
              R.buffers(0x7D, 1, [(st.col, 16)]), R.buffer_offset(0x7E, 0, 96), R.viewport(4.0, 0.0, 8.0, 8.0),
              R.draw16(TRI, 0, 3)])
    case(s, st, "0x74 the same pipeline twice", [p(st.p_vbuf), p(st.p_vbuf), st.vb(), R.draw16(TRI, 0, 3)])
    case(s, st, "0x74 replaced", [p(st.p_basic), p(st.p_vbuf), st.vb(), R.draw16(TRI, 0, 3)])
    st.t.delete()
    s.events("end")
    return s


@scenario
def render_state_depth():
    """Depth and stencil, drawn over a memoryless depth-stencil attachment
    cleared to depth 1 and stencil 3: 0x68 SetDepthStencilState (compare Less
    keeps the nearer red over a farther green; Always lets green win; a
    stencil compare Equal passes with 0x77 reference 3 and fails with 4),
    0x6c SetDepthBias pushing an equal-depth draw behind, 0x6d
    SetDepthClipMode clip or clamp for a triangle at z 1.5; 0x69, 0x6a
    (decoded as SetDepthStoreAction too), 0x78 and 0x79 on the memoryless
    attachment; 0x94 and 0x95 SetDepthCleared and SetStencilCleared. A
    private depth-stencil attachment keeps its load and store actions (only a
    memoryless one has them rewritten): a pass that stores depth 0.5 and a
    pass that loads it, where a farther green fails Less."""
    s = Scenario("render_state_depth", doc=render_state_depth.__doc__)
    s.bring_up()
    st = StateTask(s)
    less = st.t.depth_stencil(compare=LESS, write=1)
    always = st.t.depth_stencil(compare=ALWAYS, write=1)
    lequal = st.t.depth_stencil(compare=LESS_EQUAL, write=1)
    sten = st.t.depth_stencil(compare=ALWAYS, write=0, front=(EQUAL, 0, 0, 0, 0xFF, 0xFF),
                              back=(EQUAL, 0, 0, 0, 0xFF, 0xFF))
    front_only = st.t.depth_stencil(compare=ALWAYS, write=0, front=(EQUAL, 0, 0, 0, 0xFF, 0xFF))
    # A private depth-stencil texture at offset 0 of a private heap (type 8,
    # commandType 21: {u32, u32 heap, v1 descriptor, u32, u64 offset}).
    # The heap is listed in every exec.
    heap = st.t.heap(length=16 * abi.PAGE)
    st.t.bufs.append(heap)
    private = st.t.ol.add(abi.T_SERIALIZER_TEXTURE, R.op_header(21) + struct.pack("<II", 0, heap) + R.tex_desc(
        W, H, R.D32S8, usage=R.USAGE_RT, options=PRIVATE) + struct.pack("<IQ", 0, 0))
    st.start()
    p = R.pipeline_state
    pd = [p(st.p_depth)]
    par = lambda i: R.buffers(0x7D, 0, [(st.params, 32 * i)])
    d = lambda i: [par(i), R.draw16(TRI, 0, 3)]
    case(s, st, "0x68 Less: red (0.5) then green (0.7)", pd + [R.depth_stencil_state(less)] + d(0) + d(1), depth=True)
    case(s, st, "0x68 Always: red then green", pd + [R.depth_stencil_state(always)] + d(0) + d(1), depth=True)
    case(s, st, "stencil Equal, 0x77 ref 3", pd + [R.depth_stencil_state(sten), R.stencil_ref(3, 3)] + d(0),
         depth=True)
    case(s, st, "stencil Equal, 0x77 ref 4", pd + [R.depth_stencil_state(sten), R.stencil_ref(4, 4)] + d(0),
         depth=True)
    case(s, st, "front stencil only, ref 4", pd + [R.depth_stencil_state(front_only), R.stencil_ref(4, 3)] + d(0),
         depth=True)
    case(s, st, "LessEqual: red then green at the same depth", pd + [R.depth_stencil_state(lequal)] + d(0) + d(3),
         depth=True)
    case(s, st, "LessEqual, 0x6c bias 1e6 on the second", pd + [R.depth_stencil_state(lequal)] + d(0) + [
        R.depth_bias(1e6, 0.0, 0.0)] + d(3), depth=True)
    case(s, st, "LessEqual, 0x6c bias 1e6 clamped to 0.01", pd + [R.depth_stencil_state(lequal)] + d(0) + [
        R.depth_bias(1e6, 0.0, 0.01)] + d(3), depth=True)
    case(s, st, "0x6d clip: z 1.5 not drawn", pd + [R.depth_stencil_state(always), R.value(0x6D, 0)] + d(2),
         depth=True)
    case(s, st, "0x6d clamp: z 1.5 drawn", pd + [R.depth_stencil_state(always), R.value(0x6D, 1)] + d(2),
         depth=True)
    unknown = dict(depth=dict(tex=st.ds, load=R.LOAD_CLEAR, store=R.STORE_UNKNOWN, clear=1.0),
                   stencil=dict(tex=st.ds, load=R.LOAD_CLEAR, store=R.STORE_UNKNOWN, clear=3))
    case(s, st, "0x69 and 0x78 resolve Unknown store actions", pd + [R.value(0x69, 0), R.value(0x78, 0)] + d(4),
         depth=True, pass_kw=unknown)
    case(s, st, "0x6a and 0x78", pd + [R.value(0x6A, 0), R.value(0x78, 0)] + d(4), depth=True, pass_kw=unknown)
    case(s, st, "0x79 stencil options, 0x94, 0x95", pd + [R.value(0x79, 0), rec(0x94), rec(0x95)] + d(4), depth=True)
    stored = dict(depth=dict(tex=private, load=R.LOAD_CLEAR, store=R.STORE_STORE, clear=1.0),
                  stencil=dict(tex=private, load=R.LOAD_CLEAR, store=R.STORE_STORE, clear=3))
    loaded = dict(depth=dict(tex=private, load=R.LOAD_LOAD, store=R.STORE_STORE, clear=1.0),
                  stencil=dict(tex=private, load=R.LOAD_LOAD, store=R.STORE_STORE, clear=3))
    case(s, st, "private attachment, stored: red (0.5)", pd + [R.depth_stencil_state(less)] + d(0), depth=True,
         pass_kw=stored)
    case(s, st, "private attachment, loaded: green (0.7) fails Less", pd + [R.depth_stencil_state(less)] + d(1),
         depth=True, pass_kw=loaded)
    case(s, st, "0x68 ref 0", pd + [R.depth_stencil_state(0)], fault=True, depth=True)
    case(s, st, "0x68 a pipeline", pd + [R.depth_stencil_state(st.p_basic)], fault=True, depth=True)
    case(s, st, "0x68 a buffer", pd + [R.depth_stencil_state(st.pos)], fault=True, depth=True)
    st.t.delete()
    s.events("end")
    return s


@scenario
def render_state_misc():
    """State with no pixel of its own, each set inside a pass that then
    draws: 0x17 memory barrier scope, 0x85 TextureBarrier, 0x18 and 0x19
    fences (ref 0 throws "Got invalid Fence"; a buffer ref faults), 0x86 and
    0x1b UseHeaps (a private heap; a count past the record throws "Too many
    resources specified"; 257 refs take the heap-allocated list), 0x87 and
    0x89 UseResources (a ref that is not a resource resolves to nil), 0x7b
    tessellation factor scale, 0x88 line width, 0x8a alpha test, 0x8b point
    size, 0x8f viewport transform, 0x90 provoking vertex, 0x91 primitive
    restart, 0x99 and 0x9a vertex amplification with an amplification-capable
    pipeline, 0x97 and 0x98 depth and stencil resolve textures, 0x96 colour
    resolve with two attachments. 0x16 RenderBarrierResources always throws
    "unsupported"."""
    s = Scenario("render_state_misc", doc=render_state_misc.__doc__)
    s.bring_up()
    st = StateTask(s)
    t = st.t
    fence = t.fence()
    heap = t.heap()
    # Listed in every exec: in isolate an exec that uses a heap the exec
    # does not list never completes, and its stamp holds every later one.
    t.bufs.append(heap)
    p_amp = t.pipeline("amp2", v=st.fn["vert"], f=st.fn["frag"])
    p_two = t.pipeline("two", v=st.fn["vert"], f=st.fn["frag"])
    tex2, _, va2 = t.target(W, H)
    st.start()
    p = R.pipeline_state
    draw = [p(st.p_vbuf), st.vb(), R.draw16(TRI, 0, 3)]
    case(s, st, "0x17, 0x85", [R.barrier_scope(1, 1, 2)] + draw + [rec(0x85), R.barrier_scope(4, 2, 1)] + draw)
    case(s, st, "0x18, 0x19", [R.wait_fence(fence, 1)] + draw + [R.update_fence(fence, 2)])
    case(s, st, "0x86, 0x1b, 0x87, 0x89", [R.use_heaps([heap]), R.use_heaps_stages([heap], 3),
                                          R.use_resources([st.pos, st.col, st.tex], 1),
                                          R.use_resources_stages([st.pos], 1, 1), R.use_heaps([]),
                                          R.use_resources([], 1)] + draw)
    case(s, st, "257 heaps and resources", [R.use_heaps([heap] * 257), R.use_heaps_stages([heap] * 257, 1),
                                            R.use_resources([st.pos] * 257, 1),
                                            R.use_resources_stages([st.pos] * 257, 1, 1)] + draw)
    case(s, st, "0x7b, 0x88, 0x8a, 0x8b, 0x8f, 0x90, 0x91", [
        R.tess_factor_scale(2.0), R.f32(0x88, 2.0), R.f32(0x8A, 0.5), R.f32(0x8B, 4.0), R.u32(0x8F, 1),
        R.u32(0x90, 1), R.u32(0x91, 1, 15)] + draw)
    case(s, st, "0x99, 0x9a", [p(p_amp), R.amp_mode(0, 0), R.amp_count([]), R.amp_count([(0, 0), (0, 0)]),
                               R.amp_count([(0, 0)]), R.draw16(TRI, 0, 3)])
    t.fill(va2, NPIX, 0x11)
    case(s, st, "0x97, 0x98", [R.ds_resolve(0x97, tex2, 0, 0, 0, 0), R.ds_resolve(0x98, tex2, 0, 0, 0, 0)] + draw,
         depth=True)
    t.dump(va2, NPIX, "the resolve texture")
    t.fill(va2, NPIX, 0x11)
    t.render([R.describe_pass({0: dict(tex=st.tex, clear=(0.0, 0.0, 0.0, 1.0)),
                               1: dict(tex=tex2, clear=(0.0, 0.0, 1.0, 1.0))}, width=W, height=H),
              p(p_two), R.color_resolve(st.tex, 0, 0, 0, 0, 0), R.draw16(TRI, 0, 3)], label="0x96 index 0 of two")
    t.dump(st.va, NPIX, "attachment 0")
    t.dump(va2, NPIX, "attachment 1")
    case(s, st, "0x16 RenderBarrierResources: always unsupported", [rec(0x16, bytes(8))], fault=True)
    case(s, st, "0x16 with no payload", [rec(0x16)], fault=True)
    case(s, st, "0x18 ref 0", [R.update_fence(0, 1)], fault=True)
    case(s, st, "0x19 ref 0", [R.wait_fence(0, 1)], fault=True)
    case(s, st, "0x18 a buffer", [R.update_fence(st.pos, 1)], fault=True)
    case(s, st, "0x86 count past the record", [R.use_heaps([heap], count=2)], fault=True)
    case(s, st, "0x1b count past the record", [R.use_heaps_stages([heap], 1, count=2)], fault=True)
    case(s, st, "0x87 count past the record", [R.use_resources([st.pos], 1, count=2)], fault=True)
    case(s, st, "0x89 count past the record", [R.use_resources_stages([st.pos], 1, 1, count=2)], fault=True)
    case(s, st, "0x86 a buffer", [R.use_heaps([st.pos])], fault=True)
    case(s, st, "0x87 a pipeline: taken as nil, and the draw runs", [R.use_resources([st.p_vbuf], 1)] + draw)
    st.t.delete()
    s.events("end")
    return s


def limit_cases(st):
    """(label, records, faults) around each bound the decoder takes from the
    host's device (maxVertexBuffers 31 and so on on this host)."""
    b, smp, tx = st.pos, st.samplers["near"], st.src
    out = []
    for cmd, lim in ((0x6E, 31), (0x7D, 31), (0x9D, 31)):
        out.append(("0x%02x 1 at %d" % (cmd, lim - 1), [R.buffers(cmd, lim - 1, [(b, 0)])], False))
        out.append(("0x%02x 1 at %d" % (cmd, lim), [R.buffers(cmd, lim, [(b, 0)])], True))
        out.append(("0x%02x %d from 0, ref 0" % (cmd, lim), [R.buffers(cmd, 0, [(0, 0)] * lim)], False))
        out.append(("0x%02x 2 entries, 1 given" % cmd, [R.buffers(cmd, 0, [(b, 0)], count=2)], True))
    out.append(("0xa5 1 at 30", [R.buffers_stride(30, [(b, 0, 16)])], False))
    out.append(("0xa5 1 at 31", [R.buffers_stride(31, [(b, 0, 16)])], True))
    out.append(("0xa5 2 entries, 1 given", [R.buffers_stride(0, [(b, 0, 16)], count=2)], True))
    for cmd in (0x6F, 0x7E, 0x9E):
        out.append(("0x%02x index 30" % cmd, [R.buffers(0x7D if cmd == 0x7E else 0x6E if cmd == 0x6F else 0x9D, 30,
                                                         [(b, 0)]), R.buffer_offset(cmd, 30, 16)], False))
        out.append(("0x%02x index 31" % cmd, [R.buffer_offset(cmd, 31, 0)], True))
    out.append(("0xa6 index 30", [R.buffers_stride(30, [(b, 0, 16)]), R.buffer_offset_stride(30, 16, 16)], False))
    out.append(("0xa6 index 31", [R.buffer_offset_stride(31, 0, 16)], True))
    for cmd, lod in ((0x70, False), (0x7F, False), (0x9F, False), (0x71, True), (0x80, True), (0xA0, True)):
        if lod:
            out.append(("0x%02x 1 at 15" % cmd, [R.samplers_lod(cmd, 15, [(smp, 0.0, 1.0)])], False))
            out.append(("0x%02x 1 at 16" % cmd, [R.samplers_lod(cmd, 16, [(smp, 0.0, 1.0)])], True))
            out.append(("0x%02x 16 from 0" % cmd, [R.samplers_lod(cmd, 0, [(0, 0.0, 1.0)] * 16)], False))
            out.append(("0x%02x 2 entries, 1 given" % cmd, [R.samplers_lod(cmd, 0, [(smp, 0.0, 1.0)], count=2)],
                        True))
        else:
            out.append(("0x%02x 1 at 15" % cmd, [R.refs_range(cmd, 15, [smp])], False))
            out.append(("0x%02x 1 at 16" % cmd, [R.refs_range(cmd, 16, [smp])], True))
            out.append(("0x%02x 16 from 0" % cmd, [R.refs_range(cmd, 0, [0] * 16)], False))
            out.append(("0x%02x 2 entries, 1 given" % cmd, [R.refs_range(cmd, 0, [smp], count=2)], True))
    for cmd in (0x72, 0x81, 0xA1):
        out.append(("0x%02x 1 at 127" % cmd, [R.refs_range(cmd, 127, [tx])], False))
        out.append(("0x%02x 1 at 128" % cmd, [R.refs_range(cmd, 128, [tx])], True))
        out.append(("0x%02x 128 from 0" % cmd, [R.refs_range(cmd, 0, [0] * 128)], False))
        out.append(("0x%02x 2 entries, 1 given" % cmd, [R.refs_range(cmd, 0, [tx], count=2)], True))
    out.append(("0x70 a texture as a sampler", [R.refs_range(0x70, 0, [tx])], True))
    out.append(("0x72 a sampler as a texture", [R.refs_range(0x72, 0, [smp])], True))
    out.append(("0x76 16 rects", [R.scissors([(0, 0, 8, 8)] * 16)], False))
    out.append(("0x76 17 rects", [R.scissors([(0, 0, 8, 8)] * 17)], True))
    out.append(("0x76 2 rects, 1 given", [R.scissors([(0, 0, 8, 8)], count=2)], True))
    out.append(("0x83 16 viewports", [R.viewports([(0.0, 0.0, 8.0, 8.0, 0.0, 1.0)] * 16)], False))
    out.append(("0x83 17 viewports", [R.viewports([(0.0, 0.0, 8.0, 8.0, 0.0, 1.0)] * 17)], True))
    out.append(("0x83 2 viewports, 1 given", [R.viewports([(0.0, 0.0, 8.0, 8.0, 0.0, 1.0)], count=2)], True))
    out.append(("0x91 index 15", [R.u32(0x91, 1, 15)], False))
    out.append(("0x91 index 16 (bounded by maxViewportCount)", [R.u32(0x91, 1, 16)], True))
    out.append(("0x9a 3 mappings", [R.amp_count([(0, 0)] * 3)], True))
    out.append(("0x9a 2 mappings, 1 given", [R.amp_count([(0, 0)], count=2)], True))
    out.append(("0x9c index 30", [R.tg_memory(0, 0, 30)], False))
    out.append(("0x9c index 31", [R.tg_memory(0, 0, 31)], True))
    out.append(("0x8c index 5 (the host lacks setClipPlane)", [R.clip_plane((0.0, 0.0, 1.0, 0.0), 5)], True))
    out.append(("0x8c index 6", [R.clip_plane((0.0, 0.0, 1.0, 0.0), 6)], True))
    out.append(("0x8d index 15 (the host lacks the lodBias setter)", [R.sampler_state(0x8D, smp, 0.0, 1.0, 0.0, 15)],
                True))
    out.append(("0x8d index 16", [R.sampler_state(0x8D, smp, 0.0, 1.0, 0.0, 16)], True))
    out.append(("0x8e index 15", [R.sampler_state(0x8E, smp, 0.0, 1.0, 0.0, 15)], True))
    out.append(("0x8e index 16 (bounded by maxVertexSamplers)", [R.sampler_state(0x8E, smp, 0.0, 1.0, 0.0, 16)],
                True))
    out.append(("0x92 (the host lacks the front/back fill setter)", [R.fill_front_back(1, 0)], True))
    out.append(("0x93 (the host lacks setTransformFeedbackState:)", [R.u32(0x93, 1)], True))
    return out


@scenario(host=True)
def render_state_limits():
    """Each bound the decoder reads from the host's device, on both sides:
    buffer ranges and indices (maxVertexBuffers, maxFragmentBuffers,
    maxTileBuffers: 31 here), sampler ranges (16), texture ranges (128),
    scissor and viewport counts and 0x91's primitive-restart index
    (maxViewportCount 16 for all three), 0x9a's amplification count (2),
    0x9c's threadgroup memory index (maxComputeLocalMemorySizes 31), 0x8d and
    0x8e's sampler index (maxVertexSamplers for both, though 0x8e's is a
    fragment sampler; invisible here where both limits are 16), 0x8c's clip
    plane index (< 6); and a range whose entries run out. Where the check
    passes, what Metal does follows: this host's render encoder lacks
    setClipPlane:, the lodBias sampler setters, the front/back fill setter
    and setTransformFeedbackState:, so those fault with an unrecognized
    selector. host: every bound and selector is the host's."""
    s = Scenario("render_state_limits", host=True, doc=render_state_limits.__doc__)
    s.bring_up()
    st = StateTask(s)
    st.sampler("near")
    st.start()
    for label, records, faults in limit_cases(st):
        case(s, st, label, records, fault=faults, dump=False)
    st.t.delete()
    s.events("end")
    return s
