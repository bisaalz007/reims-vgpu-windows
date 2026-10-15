"""Builders for the render decoder (segment type 0) and the render-side
state objects: render pipelines (type 7, commandType 14 and 51), depth-stencil
states (4), samplers (3), rasterization rate maps (50), fences (13), and the
textures a render pass targets (docs/testing.md, sections 6.16, 6.18, 6.19).

Layouts name the reims-vgpu-core module or item that reads them.
"""
import os
import struct

from . import abi
from .abi import PAGE
from .scenario import FIXTURES
from .task import GPUTask, rec, seg

# ---- serialized pipeline descriptors (section 6.18) ----
# Serialized on the host by fixtures/shaders/render_blobs.m (Metal's
# -serializeRenderPipelineDescriptor: and the tile variant), with sentinel
# function refs that pipeline_blob() replaces by object-list references. A
# blob does not depend on which functions it names, only on their refs, so
# one blob serves every vertex/fragment pair of the same shape.
SENT_V, SENT_F, SENT_T = b"\xa1" * 4, b"\xb2" * 4, b"\xc3" * 4
BLOBS = {
    # vertex V, fragment F, colour[0] RGBA8Unorm
    "basic": "030804140000000104a1a1a1a10204b2b2b2b200010000000800000002000400000000010446000000",
    # the same, colour[0] blending: rgb = src * blendColor + dst, a = src * blendAlpha
    "blend": "030804140000000104a1a1a1a10204b2b2b2b20001000000080000000600040000000001044600000002040100000003040b00000004040100000006040d000000",
    # a vertex descriptor: attr 0 float2 @0, attr 1 uchar4normalized @8, buffer 0, stride 12
    "vdesc": "0403041c0000000804800000000104a1a1a1a10204b2b2b2b20000000200041000000001043c000000000000020000000c000000190000000200040000000001041d00000003000401000000010409000000020408000000020000000c000000190000000200040000000003040c0000000200040000000003040c0000000000010000000800000002000400000000010446000000",
    # depth and stencil Depth32Float_Stencil8 (260)
    "depth": "050804200000000904040100000a04040100000104a1a1a1a10204b2b2b2b200010000000800000002000400000000010446000000",
    # rasterSampleCount 4
    "msaa4": "0404040400000008041c0000000104a1a1a1a10204b2b2b2b2000000010000000800000002000400000000010446000000",
    # a vertex function alone: no fragment function, no colour attachment
    "nocolor": "010104a1a1a1a1",
    # the same, rasterSampleCount 4
    "nocolor4": "020404040000000104a1a1a1a1",
    # colour[0] and colour[1] RGBA8Unorm
    "two": "030804140000000104a1a1a1a10204b2b2b2b200020000000c000000190000000200040000000001044600000002000401000000010446000000",
    # maxVertexAmplificationCount 2
    "amp2": "0408041c0000001a04020000000104a1a1a1a10204b2b2b2b2000000010000000800000002000400000000010446000000",
    # tessellation: integer partitioning, triangle patches from render_tess
    "tess": "0408041c0000000c04010000000104a1a1a1a10204b2b2b2b2000000010000000800000002000400000000010446000000",
    # the same, with UInt16 control-point indices
    "tess_idx": "050804200000000c04010000001004010000000104a1a1a1a10204b2b2b2b200010000000800000002000400000000010446000000",
    # tile: function T, threadgroupSizeMatchesTileSize, colour[0] RGBA8Unorm
    "tile": "030404010000000304140000000104c3c3c3c300010000000800000002000400000000010446000000",
}


def pipeline_blob(name, v=None, f=None, t=None):
    """A serialized descriptor with its function sentinels replaced by
    object-list references (a function is its functionRef, section 6.18)."""
    b = bytes.fromhex(BLOBS[name])
    for sent, ref in ((SENT_V, v), (SENT_F, f), (SENT_T, t)):
        if sent in b:
            assert ref is not None, "%s needs a function for %s" % (name, sent.hex())
            b = b.replace(sent, struct.pack("<I", ref))
    return b


# ---- pixel formats, actions (Metal's enums) ----
RGBA8 = 70
R32F = 55
D32S8 = 260
LOAD_DONTCARE, LOAD_LOAD, LOAD_CLEAR = 0, 1, 2
STORE_DONTCARE, STORE_STORE, STORE_RESOLVE, STORE_STORE_RESOLVE, STORE_UNKNOWN = 0, 1, 2, 3, 4
TEX_2D, TEX_2D_MS, TEX_2D_ARRAY = 2, 4, 3
USAGE_READ, USAGE_WRITE, USAGE_RT = 1, 2, 4
MEMORYLESS = 0x30  # MTLResourceStorageModeMemoryless


def tex_desc(w, h, fmt=RGBA8, ttype=TEX_2D, usage=USAGE_READ | USAGE_RT, depth=1, mips=1,
             samples=1, array=1, options=0, protection=0):
    """The serialized texture descriptor v1, 32 bytes
    (`deserializer::texture_descriptor::SerializedTextureDescriptor`): u32
    {textureType:4, framebufferOnly:1, isDrawable:1,
    allowGPUOptimizedContents:1, :1, usage:8, pixelFormat:16}, width,
    height, depth, u16 mips, samples, arrayLength, resourceOptions, u64
    protectionOptions."""
    w0 = (ttype & 0xF) | ((usage & 0xFF) << 8) | ((fmt & 0xFFFF) << 16)
    return struct.pack("<IIIIHHHHQ", w0, w, h, depth, mips, samples, array, options, protection)


def op_header(command_type):
    """The operation header {u32 commandType, u32 unread}
    (`resources::manager::records::OperationHeader`)."""
    return struct.pack("<II", command_type, 0)


# ---- the render pass (section 6.16) ----

def attachment(tex=0, resolve=0, level=0, slice_=0, depth_plane=0, rlevel=0, rslice=0, rdepth=0,
               load=LOAD_CLEAR, store=STORE_STORE, opts=0):
    """The first 0x1c bytes every attachment shares
    (`deserializer::records::CmdDescribeRenderPass`): texture ref, resolve
    texture ref, then u16 level, slice, depthPlane, resolveLevel,
    resolveSlice, resolveDepthPlane, loadAction, storeAction,
    storeActionOptions, and one u16 nothing reads."""
    return struct.pack("<IIHHHHHHHHHH", tex, resolve, level, slice_, depth_plane, rlevel, rslice, rdepth,
                       load, store, opts, 0)


def describe_pass(colors=None, depth=None, stencil=None, vis=0, array_length=0, width=0, height=0):
    """0x1a DescribeRenderPass, 0x248 bytes
    (`deserializer::records::CmdDescribeRenderPass`): depth (0x28: common,
    f64 clearDepth @0x1c, u16 depthResolveFilter @0x24), stencil (0x24:
    common, u32 clearStencil @0x1c, u16 stencilResolveFilter @0x20), 8
    colours (0x3c: common, f64 r, g, b, a), u32 visibility buffer ref
    @0x22c, u64 renderTargetArrayLength, width, height.

    colors: {index: dict(attachment kwargs, clear=(r, g, b, a))}; depth and
    stencil: dicts of attachment kwargs plus clear= and filter=."""
    out = b""
    d = dict(depth or {})
    dclear, dfilter = d.pop("clear", 1.0), d.pop("filter", 0)
    out += (attachment(**d) if depth else bytes(0x1c)) + struct.pack("<dHH", dclear if depth else 0.0,
                                                                    dfilter if depth else 0, 0)
    st = dict(stencil or {})
    sclear, sfilter = st.pop("clear", 0), st.pop("filter", 0)
    out += (attachment(**st) if stencil else bytes(0x1c)) + struct.pack("<IHH", sclear if stencil else 0,
                                                                      sfilter if stencil else 0, 0)
    for i in range(8):
        c = dict((colors or {}).get(i) or {})
        clear = c.pop("clear", (0.0, 0.0, 0.0, 0.0))
        if (colors or {}).get(i) is None:
            out += bytes(0x3c)
        else:
            out += attachment(**c) + struct.pack("<dddd", *clear)
    out += struct.pack("<IQQQ", vis, array_length, width, height)
    assert len(out) == 0x248
    return rec(0x1A, out)


# The pass prefix (`deserializer::render::pass`).
def default_raster_samples(n):
    return rec(0x1E, struct.pack("<I", n))


def default_color_samples(n):
    return rec(0x1F, struct.pack("<I", n))


def sample_positions(points, count=None):
    """0x20: {u32 count} then count x {f32 x, f32 y}; count > 8 throws."""
    n = len(points) if count is None else count
    return rec(0x20, struct.pack("<I", n) + b"".join(struct.pack("<ff", x, y) for x, y in points))


def rate_map(ref):
    return rec(0x21, struct.pack("<I", ref))


def imageblock_sample_length(n):
    return rec(0x22, struct.pack("<I", n))


def threadgroup_memory_length(n):
    return rec(0x23, struct.pack("<I", n))


def tile_size(w, h):
    return rec(0x24, struct.pack("<HH", w, h))


# ---- main-loop commands (`deserializer::render`'s dispatch) ----
# Payload layouts are `deserializer::records`'; natural structs include their
# tail padding.

def draw16(prim, start, count):
    """0x01 DrawPrimitives16 {u32 primitive, u16 start, u16 count}."""
    return rec(0x01, struct.pack("<IHH", prim, start, count))


def draw64(prim, start, count):
    """0x00 DrawPrimitives64, packed 0x14: {u32 primitive, u64 start, u64 count}."""
    return rec(0x00, struct.pack("<IQQ", prim, start, count))


def draw_inst64(prim, start, count, inst):
    """0x02, packed 0x1c: {u64 start, u64 count, u64 instances, u16 primitive} + 2 pad."""
    return rec(0x02, struct.pack("<QQQHxx", start, count, inst, prim))


def draw_inst16(prim, start, count, inst):
    """0x03 {u16 start, u16 count, u16 instances, u16 primitive}."""
    return rec(0x03, struct.pack("<HHHH", start, count, inst, prim))


def draw_inst_base64(prim, start, count, inst, base):
    """0x04, packed 0x24: {u64 start, count, instances, baseInstance, u16 primitive} + 2 pad."""
    return rec(0x04, struct.pack("<QQQQHxx", start, count, inst, base, prim))


def draw_inst_base16(prim, start, count, inst, base):
    """0x05, packed 0xc: {u16 start, count, instances, baseInstance, primitive} + 2 pad."""
    return rec(0x05, struct.pack("<HHHHHxx", start, count, inst, base, prim))


def draw_idx64(prim, itype, ibuf, count, off):
    """0x06 {u16 primitive, u16 indexType, u32 indexBuffer, u64 count, u64 offset}."""
    return rec(0x06, struct.pack("<HHIQQ", prim, itype, ibuf, count, off))


def draw_idx16(prim, itype, ibuf, count, off):
    """0x07 {u16 primitive, u16 indexType, u32 indexBuffer, u16 count, u16 offset}."""
    return rec(0x07, struct.pack("<HHIHH", prim, itype, ibuf, count, off))


def draw_idx_inst64(prim, itype, ibuf, count, off, inst):
    """0x08 {.., u64 count, u64 offset, u64 instances}."""
    return rec(0x08, struct.pack("<HHIQQQ", prim, itype, ibuf, count, off, inst))


def draw_idx_inst16(prim, itype, ibuf, count, off, inst):
    """0x09 {.., u16 count, u16 offset, u16 instances} + 2 pad."""
    return rec(0x09, struct.pack("<HHIHHHxx", prim, itype, ibuf, count, off, inst))


def draw_idx_base64(prim, itype, ibuf, off, count, inst, bv, bi, cmd=0x0A):
    """0x0a (and 0x1c) {u16 primitive, u16 indexType, u32 indexBuffer, u64
    offset, u64 count, u64 instances, u64 baseVertex, u64 baseInstance}."""
    return rec(cmd, struct.pack("<HHIQQQQQ", prim, itype, ibuf, off, count, inst, bv & (2**64 - 1), bi))


def draw_idx_base16(prim, itype, ibuf, off, count, inst, bv, bi, cmd=0x0B):
    """0x0b (and 0x1d) {u16 primitive, u16 indexType, u32 indexBuffer, u16
    offset, u16 count, u16 instances, u16 baseVertex, u16 baseInstance} + 2
    pad. 0x0b loads baseVertex zero-extended, 0x1d sign-extended."""
    return rec(cmd, struct.pack("<HHIHHHHHxx", prim, itype, ibuf, off, count, inst, bv & 0xFFFF, bi))


def draw_patches64(cp, start, count, pib, pio, inst, base):
    """0x0c, packed 0x30: {u64 patchStart, u64 patchCount, u32 patchIndexBuffer,
    u64 its offset, u64 instances, u64 baseInstance, u16 controlPoints} + 2 pad."""
    return rec(0x0C, struct.pack("<QQIQQQHxx", start, count, pib, pio, inst, base, cp))


def draw_patches16(cp, start, count, pib, pio, inst, base):
    """0x0d {u16 start, u16 count, u32 patchIndexBuffer, u16 offset, u16
    instances, u16 baseInstance, u16 controlPoints}."""
    return rec(0x0D, struct.pack("<HHIHHHH", start, count, pib, pio, inst, base, cp))


def draw_idx_patches64(cp, start, count, pib, pio, cib, cio, inst, base):
    """0x0e, packed 0x3c."""
    return rec(0x0E, struct.pack("<QQIQIQQQHxx", start, count, pib, pio, cib, cio, inst, base, cp))


def draw_idx_patches16(cp, start, count, pib, pio, cib, cio, inst, base):
    """0x0f, packed 0x18: {u16 start, count, u32 pib, u16 pio, u32 cib, u16
    cio, u16 instances, u16 baseInstance, u16 controlPoints} + 2 pad."""
    return rec(0x0F, struct.pack("<HHIHIHHHHxx", start, count, pib, pio, cib, cio, inst, base, cp))


def draw_indirect(prim, ib, off):
    """0x10 {u64 indirectOffset, u32 indirectBuffer, u16 primitive} + 2 pad."""
    return rec(0x10, struct.pack("<QIHxx", off, ib, prim))


def draw_idx_indirect(prim, itype, ibuf, ib, ioff, off):
    """0x11, packed 0x1c: {u16 primitive, u16 indexType, u32 indexBuffer, u32
    indirectBuffer, u64 indexOffset, u64 indirectOffset}."""
    return rec(0x11, struct.pack("<HHIIQQ", prim, itype, ibuf, ib, ioff, off))


def draw_patches_indirect(cp, pib, ib, pio, off):
    """0x12, packed 0x1c: {u32 patchIndexBuffer, u32 indirectBuffer, u64
    patchIndexOffset, u64 indirectOffset, u16 controlPoints} + 2 pad."""
    return rec(0x12, struct.pack("<IIQQHxx", pib, ib, pio, off, cp))


def draw_idx_patches_indirect(cp, pib, cib, ib, pio, cio, off):
    """0x13, packed 0x28: {u32 pib, u32 cib, u32 indirect, u64 pio, u64 cio,
    u64 indirectOffset, u16 controlPoints} + 2 pad."""
    return rec(0x13, struct.pack("<IIIQQQHxx", pib, cib, ib, pio, cio, off, cp))


def exec_icb(icb, rangebuf, off):
    """0x14 ExecuteCommandsInBuffer {u32 icb, u32 rangeBuffer, u64 offset}."""
    return rec(0x14, struct.pack("<IIQ", icb, rangebuf, off))


def exec_icb_ranged(icb, loc, n):
    """0x15, packed 0x14: {u32 icb, u64 location, u64 length}."""
    return rec(0x15, struct.pack("<IQQ", icb, loc, n))


def barrier_scope(scope, after, before):
    """0x17 {u16 scope, u8 afterStages, u8 beforeStages}."""
    return rec(0x17, struct.pack("<HBB", scope, after, before))


def update_fence(ref, stages):
    return rec(0x18, struct.pack("<II", ref, stages))


def wait_fence(ref, stages):
    return rec(0x19, struct.pack("<II", ref, stages))


def use_heaps_stages(refs, stages, count=None):
    """0x1b {u32 count, u16 stages} + pad, then count x u32 heap refs."""
    return rec(0x1B, struct.pack("<IHxx", len(refs) if count is None else count, stages)
               + b"".join(struct.pack("<I", r) for r in refs))


def value(cmd, v):
    """A CmdSetValue command {u64 value}: 0x69, 0x6a, 0x6b, 0x6d, 0x73,
    0x78, 0x79, 0x7c."""
    return rec(cmd, struct.pack("<Q", v))


def blend_color(r, g, b, a):
    return rec(0x65, struct.pack("<ffff", r, g, b, a))


def color_store_action(action, index):
    """0x66 {u32 storeAction, u32 index}; index > max colour index throws."""
    return rec(0x66, struct.pack("<II", action, index))


def color_store_options(opts, index):
    """0x67, packed 0xc: {u64 options, u32 index}."""
    return rec(0x67, struct.pack("<QI", opts, index))


def depth_stencil_state(ref):
    return rec(0x68, struct.pack("<I", ref))


def depth_bias(bias, slope, clamp):
    return rec(0x6C, struct.pack("<fff", bias, slope, clamp))


def buffers(cmd, loc, entries, count=None):
    """0x6e/0x7d/0x9d {u32 location, u32 length} then length x packed {u32
    ref, u64 offset}."""
    return rec(cmd, struct.pack("<II", loc, len(entries) if count is None else count)
               + b"".join(struct.pack("<IQ", r, o) for r, o in entries))


def buffers_stride(loc, entries, count=None):
    """0xa5 {u32 location, u32 length} then packed {u32 ref, u64 offset, u64 stride}."""
    return rec(0xA5, struct.pack("<II", loc, len(entries) if count is None else count)
               + b"".join(struct.pack("<IQQ", r, o, st) for r, o, st in entries))


def buffer_offset(cmd, index, off):
    """0x6f/0x7e/0x9e, packed 0xc: {u32 index, u64 offset}."""
    return rec(cmd, struct.pack("<IQ", index, off))


def buffer_offset_stride(index, off, stride):
    """0xa6, packed 0x14: {u32 index, u64 offset, u64 stride}."""
    return rec(0xA6, struct.pack("<IQQ", index, off, stride))


def refs_range(cmd, loc, refs, count=None):
    """Samplers 0x70/0x7f/0x9f and textures 0x72/0x81/0xa1: {u32 location, u32
    length} then length x u32 refs."""
    return rec(cmd, struct.pack("<II", loc, len(refs) if count is None else count)
               + b"".join(struct.pack("<I", r) for r in refs))


def samplers_lod(cmd, loc, entries, count=None):
    """0x71/0x80/0xa0: {u32 location, u32 length} then {u32 ref, f32 min, f32 max}."""
    return rec(cmd, struct.pack("<II", loc, len(entries) if count is None else count)
               + b"".join(struct.pack("<Iff", r, a, b) for r, a, b in entries))


def pipeline_state(ref):
    """0x74 SetRenderPipelineState {u32 ref}; ref 0 throws."""
    return rec(0x74, struct.pack("<I", ref))


def scissor(x, y, w, h):
    return rec(0x75, struct.pack("<QQQQ", x, y, w, h))


def scissors(rects, count=None):
    """0x76 {u64 count} then count x {u64 x, y, w, h}; count > maxViewportCount throws."""
    return rec(0x76, struct.pack("<Q", len(rects) if count is None else count)
               + b"".join(struct.pack("<QQQQ", *r) for r in rects))


def stencil_ref(front, back):
    return rec(0x77, struct.pack("<II", front, back))


def tess_factor_buffer(ref, off, stride):
    """0x7a, packed 0x14: {u32 buffer, u64 offset, u64 instanceStride}."""
    return rec(0x7A, struct.pack("<IQQ", ref, off, stride))


def tess_factor_scale(v):
    return rec(0x7B, struct.pack("<f", v))


def viewport(x, y, w, h, zn=0.0, zf=1.0):
    return rec(0x82, struct.pack("<dddddd", x, y, w, h, zn, zf))


def viewports(vps, count=None):
    """0x83 {u32 count}, then count x 6 f64."""
    return rec(0x83, struct.pack("<I", len(vps) if count is None else count)
               + b"".join(struct.pack("<dddddd", *v) for v in vps))


def visibility_mode(mode, off):
    """0x84 {u64 offset, u64 mode}."""
    return rec(0x84, struct.pack("<QQ", off, mode))


def use_heaps(refs, count=None):
    return rec(0x86, struct.pack("<I", len(refs) if count is None else count)
               + b"".join(struct.pack("<I", r) for r in refs))


def use_resources(refs, usage, count=None):
    """0x87 {u32 count, u32 usage} then u32 refs."""
    return rec(0x87, struct.pack("<II", len(refs) if count is None else count, usage)
               + b"".join(struct.pack("<I", r) for r in refs))


def use_resources_stages(refs, usage, stages, count=None):
    """0x89 {u32 count, u16 usage, u16 stages} then u32 refs."""
    return rec(0x89, struct.pack("<IHH", len(refs) if count is None else count, usage, stages)
               + b"".join(struct.pack("<I", r) for r in refs))


def f32(cmd, v):
    """One f32: 0x7b, 0x88 line width, 0x8a alpha test, 0x8b point size."""
    return rec(cmd, struct.pack("<f", v))


def u32(cmd, *vals):
    return rec(cmd, struct.pack("<%dI" % len(vals), *vals))


def clip_plane(p, index):
    return rec(0x8C, struct.pack("<ffffI", p[0], p[1], p[2], p[3], index))


def sampler_state(cmd, ref, lmin, lmax, bias, index):
    """0x8d/0x8e {u32 ref, f32 lodMin, f32 lodMax, f32 lodBias, u32 index}."""
    return rec(cmd, struct.pack("<IfffI", ref, lmin, lmax, bias, index))


def fill_front_back(front, back):
    return rec(0x92, struct.pack("<HH", front, back))


def color_resolve(tex, slice_, plane, level, yinv, index):
    """0x96 {u32 tex, u16 slice, depthPlane, level, yInvert, index} + 2 pad;
    index >= max colour index throws."""
    return rec(0x96, struct.pack("<IHHHHHxx", tex, slice_, plane, level, yinv, index))


def ds_resolve(cmd, tex, slice_, plane, level, yinv):
    """0x97/0x98 {u32 tex, u16 slice, depthPlane, level, yInvert}."""
    return rec(cmd, struct.pack("<IHHHH", tex, slice_, plane, level, yinv))


def amp_mode(mode, v):
    return rec(0x99, struct.pack("<II", mode, v))


def amp_count(maps, count=None):
    """0x9a {u32 count} then count x {u32 viewportOffset, u32 rtaOffset}."""
    return rec(0x9A, struct.pack("<I", len(maps) if count is None else count)
               + b"".join(struct.pack("<II", a, b) for a, b in maps))


def dispatch_tile(w, h, d=1):
    """0x9b {MTLSize threadsPerTile} (3 x u64)."""
    return rec(0x9B, struct.pack("<QQQ", w, h, d))


def tg_memory(length, off, index):
    """0x9c, packed 0x14: {u64 length, u64 offset, u32 index}."""
    return rec(0x9C, struct.pack("<QQI", length, off, index))


def dispatch_tile_region(cmd, tpt, origin, size, rta=0):
    """0xa2/0xa3, packed 0x4c: MTLSize threadsPerTile, MTLOrigin, MTLSize,
    u32 renderTargetArrayIndex (0xa2 ignores it)."""
    return rec(cmd, struct.pack("<QQQQQQQQQI", *tpt, *origin, *size, rta))


def tile_dims(buf, off):
    """0xa4 GetTileDimensions, packed 0xc: {u32 replyBuffer, u64 offset};
    replies {u16 tileWidth, u16 tileHeight}."""
    return rec(0xA4, struct.pack("<IQ", buf, off))


# ---- a task that renders ----

class RenderTask(GPUTask):
    """A GPUTask with builders for functions, pipelines, state objects and
    textures over its buffers. Buffers default to one page each."""

    def __init__(self, s, tid=1, pages=64, **kw):
        super().__init__(s, tid=tid, pages=pages, **kw)
        self.bufs = []

    def buffer(self, length=PAGE, flags=0, contents=None, va=None):
        ref, va = super().buffer(length, flags, contents, va)
        if not flags & 1:
            self.bufs.append(ref)
        return ref, va

    def function(self, fixture):
        """A function (type 6, >= 20 bytes): {u64 task VA of the metallib, u32
        its length, u32 commandType 15, u32 unread}
        (`resources::manager::create::function`)."""
        n = os.path.getsize(os.path.join(FIXTURES, fixture))
        va = self.reserve(n)
        self.s.mem_file(self.gpa(va), fixture)  # a task's VA pages are consecutive RAM pages
        return self.ol.add(abi.T_FUNCTION, struct.pack("<QIII", va, n, 15, 0))

    def pipeline(self, name, v=None, f=None, t=None, command_type=None, blob=None):
        """A render pipeline (type 7): {u32 commandType 14 (51 tile), u32
        unread, u32 unread, u32 descriptorLength}, then the descriptor
        (`resources::manager::create::pipeline`). The
        record is exactly 16 + descriptorLength (section 6.18)."""
        b = pipeline_blob(name, v, f, t) if blob is None else blob
        ct = command_type if command_type is not None else (51 if name == "tile" else 14)
        return self.ol.add(abi.T_SERIALIZED, struct.pack("<IIII", ct, 0, 0, len(b)) + b)

    def target(self, w, h, fmt=RGBA8, bpp=4, usage=USAGE_READ | USAGE_RT, contents=None, ttype=TEX_2D,
               array=1, bpr=None):
        """A texture over a shared buffer (type 8, commandType 9): {u32 9, u32
        unread, u32 unread, u32 bufferRef, u64 offset, u64 bytesPerRow}, then
        a v1 descriptor (`resources::manager::texture`, section 6.19).
        Returns (texture ref, buffer ref, buffer VA)."""
        bpr = bpr or max(16, w * bpp)
        n = bpr * h * array
        # A shared buffer's length must be page-aligned ("Attempting to map
        # memory with a non-page-aligned length").
        bref, va = self.buffer((n + PAGE - 1) // PAGE * PAGE, contents=contents)
        rec_ = op_header(9) + struct.pack("<IIQQ", 0, bref, 0, bpr) + tex_desc(
            w, h, fmt, ttype=ttype, usage=usage, array=array)
        return self.ol.add(abi.T_SERIALIZER_TEXTURE, rec_), bref, va

    def memoryless(self, w, h, fmt=RGBA8, ttype=TEX_2D, samples=1, usage=USAGE_RT, command_type=1,
                   options=MEMORYLESS):
        """A memoryless serializer texture (type 8, commandType 1): header,
        u32 unread, v1 descriptor with resourceOptions memoryless."""
        return self.ol.add(abi.T_SERIALIZER_TEXTURE, op_header(command_type) + struct.pack("<I", 0) + tex_desc(
            w, h, fmt, ttype=ttype, usage=usage, samples=samples, options=options))

    def sampler(self, min_f=0, mag_f=0, s=0, t=0, r=0, mip=0, aniso=1, norm=1, lod_avg=0, compare=0,
                border=0, argbuf=0, reduction=0, lmin=0.0, lmax=1000.0, command_type=3):
        """A sampler (type 7, commandType 3): header, then the
        NewSamplerState operation (0x1c): u32 unread; u32
        {minFilter:2, magFilter:2, lodAverage:1, compare:3, s:4, t:4, r:4,
        border:4, mipFilter:2, maxAnisotropy:5, normalized:1}; u32
        {supportArgumentBuffers:1, reductionMode:3}; f32 lodMin, lodMax; 8
        bytes unread (`resources::manager::create::sampler`)."""
        w1 = (min_f | mag_f << 2 | lod_avg << 4 | compare << 5 | s << 8 | t << 12 | r << 16 | border << 20
              | mip << 24 | aniso << 26 | norm << 31)
        w2 = argbuf | reduction << 1
        return self.ol.add(abi.T_SERIALIZED, op_header(command_type) + struct.pack(
            "<IIIff8x", 0, w1, w2, lmin, lmax))

    def depth_stencil(self, compare=7, write=1, front=None, back=None, command_type=4):
        """A depth-stencil state (type 7, commandType 4): header, then the
        NewDepthStencilState operation (0x20): u32 unread; u32
        {compare:3, write:1, hasFront:1, hasBack:1}; front and back {u32
        {compare:3, fail:3, depthFail:3, pass:3}, u32 readMask, u32 writeMask}
        (`resources::manager::create::depth_stencil`).
        front/back: (compare, fail, depthFail, pass, read, write)."""
        def face(f):
            if f is None:
                return struct.pack("<III", 0, 0, 0)
            c, fl, df, p, rm, wm = f
            return struct.pack("<III", c | fl << 3 | df << 6 | p << 9, rm, wm)
        w = compare | write << 3 | (front is not None) << 4 | (back is not None) << 5
        return self.ol.add(abi.T_SERIALIZED, op_header(command_type) + struct.pack("<II", 0, w)
                           + face(front) + face(back))

    def fence(self, command_type=13):
        """A fence (type 7, commandType 13): the header alone
        (`resources::manager::create::fence`)."""
        return self.ol.add(abi.T_SERIALIZED, op_header(command_type))

    def ratemap(self, w, h, layers, mutable=0, min_factor=0.0, command_type=50, raw_tail=None, count=None):
        """A rasterization rate map (type 7, commandType 50): header, u32
        unread, {u16 screenW, u16 screenH, u32 layerCount, u32 mutability, f32
        minFactor}, then layerCount x {u16 h, u16 v}, then each layer's
        horizontal and vertical f32 samples
        (`resources::manager::create::rate_map`).
        layers: see ratemap_record."""
        return self.ol.add(abi.T_SERIALIZED, ratemap_record(w, h, layers, mutable, min_factor, command_type, raw_tail,
                                                            count))

    def heap(self, length=PAGE, flags=1):
        """A heap (type 13), packed {u32 page, u64 length, u8 flags}; bit 0
        private (in-process heaps must be private, section 6.11). The record
        is read as 16 bytes (`resources::manager::records::ObjectHeap`), so
        it is padded."""
        va = self.reserve(length)
        return self.ol.add(abi.T_HEAP, struct.pack("<IQB3x", va // PAGE, length, flags))

    def icb(self, max_count=1, command_types=1, inherit=3, length=PAGE):
        """An indirect command buffer (type 20): {u64 length, u32 page}, header
        54, the NewIcbCommand operation (0x50): the 0x14 descriptor, the 0x34
        layout, u32 maxCommandCount, u16 options
        (`resources::manager::icb::records`)."""
        va = self.reserve(length)
        desc = struct.pack("<I9BxHI", command_types, 0, 0, 0, 0, 0, 0, 0, 0, 0, inherit, 0)
        assert len(desc) == 0x14
        layout = bytes(0x34)
        op = desc + layout + struct.pack("<IHxx", max_count, 0)
        assert len(op) == 0x50
        return self.ol.add(abi.T_ICB, struct.pack("<QI", length, va // PAGE) + op_header(54) + op)

    # ---- execution ----

    def render(self, records, fault=False, resources=None, cont_prev=0, cont_next=0, label=None, **kw):
        """One exec (67) of one render segment. In the isolate setup every
        buffer the GPU touches must be listed (section 6.12), so every shared
        buffer of the task is, with flags 0."""
        if self._cmd > 7 * PAGE:
            self.reset_cmdbufs()
        cb = self.cmdbuf(seg(abi.ENC_RENDER, b"".join(records), cont_prev, cont_next))
        res = [(r, 0) for r in (self.bufs if resources is None else resources)]
        return self.exec3([cb], resources=res, fault=fault, label=label, **kw)

    def exec3_cmd(self, records, resources=None, barriers=()):
        """Writes an exec (67) of one render segment into the root FIFO
        without ringing or waiting; returns its stamp. With barriers, the
        FIFO holds the command until those stamps arrive."""
        if self._cmd > 7 * PAGE:
            self.reset_cmdbufs()
        va, n = self.cmdbuf(seg(abi.ENC_RENDER, b"".join(records)))
        res = [(r, 0) for r in (self.bufs if resources is None else resources)]
        p = struct.pack("<III", self.tid, len(res), 1) + b"".join(struct.pack("<II", r, f) for r, f in res)
        p += struct.pack("<QQ", va, n)
        return self.s.fifo.cmd(abi.OP_EXEC_INDIRECT3, p, barriers=barriers)

    def exec_segments(self, segments, fault=False, resources=None, label=None, **kw):
        """One exec of several command-buffer ranges, one segment each."""
        if self._cmd > 6 * PAGE:
            self.reset_cmdbufs()
        cbs = [self.cmdbuf(sg) for sg in segments]
        res = [(r, 0) for r in (self.bufs if resources is None else resources)]
        return self.exec3(cbs, resources=res, fault=fault, label=label, **kw)


def ratemap_record(w, h, layers, mutable=0, min_factor=0.0, command_type=50, raw_tail=None, count=None):
    """layers: [(horizontal samples, vertical samples)], each a list of f32
    rates or a count of 1.0s. count overrides layerCount; raw_tail replaces
    everything after the layer headers."""
    def rates(x):
        return [1.0] * x if isinstance(x, int) else list(x)
    n = len(layers) if count is None else count
    body = op_header(command_type) + struct.pack("<IHHIIf", 0, w, h, n, mutable, min_factor)
    body += b"".join(struct.pack("<HH", len(rates(hs)), len(rates(vs))) for hs, vs in layers)
    if raw_tail is not None:
        return body + raw_tail
    for hs, vs in layers:
        body += struct.pack("<%df" % len(rates(hs)), *rates(hs))
        body += struct.pack("<%df" % len(rates(vs)), *rates(vs))
    return body


def float4s(*vals):
    return b"".join(struct.pack("<ffff", *v) for v in vals)


# Positions (clip space) of a full-target triangle and of the left and right
# halves' covering triangles, for render_vbuf.metal.
FULL = [(-1.0, -1.0, 0.0, 1.0), (3.0, -1.0, 0.0, 1.0), (-1.0, 3.0, 0.0, 1.0)]
LEFT = [(-1.0, -1.0, 0.0, 1.0), (0.0, -1.0, 0.0, 1.0), (-1.0, 3.0, 0.0, 1.0)]
