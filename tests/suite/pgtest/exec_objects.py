"""Records for the objects the exec, info and ICB scenarios name: functions,
pipelines, samplers, depth-stencil states, rate maps, heaps, textures over
buffers, CPU buffers and indirect command buffers (docs/testing.md,
sections 6.11, 6.17-6.19), plus the info decoder's records and a task that
holds them all.

Every layout names the reims-vgpu-core module or item that reads it;
nothing is imported from the library's code.
"""
import struct

from . import abi
from .abi import PAGE
from .task import GPUTask, rec, seg

# ---- object records ----


def function_record(va, length):
    """Type 6: {u64 task VA of the metallib, u32 length, u32 commandType 15,
    u32 unread}, 20 bytes (resources::manager::create::function)."""
    return struct.pack("<QIII", va, length, 15, 0)


def pipeline_record(command_type, blob):
    """Type 7 pipeline: {u32 commandType (11 compute, 14 render), u32, u32,
    u32 descriptorLength}, then the serialized descriptor; exactly 16 +
    descriptorLength, because Metal is handed entry length - 16 bytes
    (resources::manager::create::pipeline)."""
    return struct.pack("<IIII", command_type, 0, 0, len(blob)) + bytes(blob)


def compute_blob(fn_ref, icb=False):
    """Metal's compact property list for a compute descriptor with only its
    function (docs/testing.md, section 6.18): 01 00 04 <R>. With icb,
    supportIndirectCommandBuffers (key 7) as well: 02 07 04 01000000 00 04
    <R> (serialized on this host by -[MTLDevice
    serializeComputePipelineDescriptor:])."""
    if icb:
        return bytes([2, 7, 4, 1, 0, 0, 0, 0, 4]) + struct.pack("<I", fn_ref)
    return bytes([1, 0, 4]) + struct.pack("<I", fn_ref)


def render_blob(vert_ref, frag_ref, fmt=70, icb=False):
    """A render descriptor: vertex, fragment, colour[0] pixel format
    (docs/testing.md, section 6.18). With icb,
    supportIndirectCommandBuffers (key 0x19), which moves the attachment
    block from 0x14 to 0x1c (serialized on this host as compute_blob's)."""
    if icb:
        head = (bytes([0x04, 0x08, 0x04, 0x1C, 0, 0, 0, 0x19, 0x04, 1, 0, 0, 0, 0x01, 0x04]) + struct.pack("<I", vert_ref)
                + bytes([0x02, 0x04]) + struct.pack("<I", frag_ref) + bytes(3))
    else:
        head = (bytes([0x03, 0x08, 0x04, 0x14, 0, 0, 0, 0x01, 0x04]) + struct.pack("<I", vert_ref)
                + bytes([0x02, 0x04]) + struct.pack("<I", frag_ref) + bytes(1))
    return head + bytes([1, 0, 0, 0, 8, 0, 0, 0, 0x02, 0x00, 0x04, 0, 0, 0, 0, 0x01, 0x04]) + struct.pack("<I", fmt)


def sampler_record(support_argument_buffers=1, min_filter=0, mag_filter=1, lod_min=0.0, lod_max=8.0):
    """Type 7, commandType 3: header {3, unread}, then
    PGOperationNewSamplerStateWithDescriptor (0x1c): u32 unread, u32 {minFilter:2,
    magFilter:2, lodAverage:1, compare:3, s:4, t:4, r:4, border:4, mipFilter:2,
    maxAnisotropy:5, normalized:1}, u32 {supportArgumentBuffers:1,
    reductionMode:3}, f32 lodMin, f32 lodMax, 8 unread
    (resources::manager::create::sampler, NewSamplerState)."""
    w0 = min_filter | (mag_filter << 2) | (1 << 22) | (1 << 31)  # maxAnisotropy 1, normalized
    w1 = support_argument_buffers & 1
    return struct.pack("<II", 3, 0) + struct.pack("<IIIff", 0, w0, w1, lod_min, lod_max) + bytes(8)


def depth_stencil_record(compare=1, write=1):
    """Type 7, commandType 4: header, then
    PGOperationNewDepthStencilStateWithDescriptor (0x20): u32 unread, u32
    {compare:3, write:1, hasFront:1, hasBack:1}, front and back {u32 bits,
    readMask, writeMask} (resources::manager::create::depth_stencil,
    NewDepthStencilState)."""
    face = struct.pack("<III", 7, 0xFF, 0xFF)  # compare always
    return struct.pack("<II", 4, 0) + struct.pack("<II", 0, compare | (write << 3)) + face + face


def rate_map_record(w=64, h=32, layers=((2, 2),), mutable=0, min_factor=0.0, samples=None):
    """Type 7, commandType 50: header, then
    PGOperationNewRasterizationRateMapWithDescriptor (0x14): u32 unread, u16
    screenW, u16 screenH, u32 layerCount, u32 mutability, f32 minFactor; then
    layerCount x {u16 h, u16 v}; then each layer's h and v floats
    (resources::manager::create::rate_map, NewRateMap)."""
    out = struct.pack("<II", 50, 0) + struct.pack("<IHHIIf", 0, w, h, len(layers), mutable, min_factor)
    for hc, vc in layers:
        out += struct.pack("<HH", hc, vc)
    for i, (hc, vc) in enumerate(layers):
        hs, vs = samples[i] if samples else ([1.0] * hc, [1.0] * vc)
        out += struct.pack("<%df" % hc, *hs) + struct.pack("<%df" % vc, *vs)
    return out


def heap_record(va_page, length, flags):
    """Type 13: {u32 page, u64 length, u8 flags} at +0, +4, +0xc, read as 16
    bytes; flags bit 0 private, 1 tracked, 2 read-only
    (resources::manager::records::ObjectHeap)."""
    return struct.pack("<IQB3x", va_page, length, flags)


def cpu_buffer_record(length, va_page):
    """Type 19: packed {u64 length, u32 task VA page}, 12 bytes
    (resources::manager::records::ObjectCpuBuffer)."""
    return struct.pack("<QI", length, va_page)


def texture_desc_v1(fmt=70, w=4, h=4, usage=5, ttype=2, mips=1, samples=1, array=1, options=0):
    """Serialized texture descriptor v1, 32 bytes
    (deserializer::texture_descriptor::SerializedTextureDescriptor)."""
    word0 = ttype | (usage << 8) | (fmt << 16)
    return struct.pack("<IIIIHHHHQ", word0, w, h, 1, mips, samples, array, options, 0)


def buffer_texture_record(buf_ref, offset=0, bpr=16, **desc):
    """Type 8, commandType 9: {u32 9, u32, u32, u32 bufferRef, u64 offset,
    u64 bytesPerRow}, then a v1 descriptor (docs/testing.md, section 6.19)."""
    return struct.pack("<IIIIQQ", 9, 0, 0, buf_ref, offset, bpr) + texture_desc_v1(**desc)


# ---- indirect command buffers ----

# MTLIndirectCommandType bits.
ICB_DRAW = 1
ICB_DRAW_INDEXED = 2
ICB_DRAW_PATCHES = 4
ICB_DRAW_INDEXED_PATCHES = 8
ICB_DISPATCH = 0x20
ICB_DISPATCH_THREADS = 0x40

# Descriptor flag bits (the u16 at +0xe of
# PGSerializedIndirectCommandBufferDescriptor).
INH_PIPELINE = 1
INH_BUFFERS = 2
SUPPORT_RAYTRACING = 4
DYNAMIC_STRIDE = 8


def icb_layout(commandTypeOffset=0, barrierOffset=4, kernelDispatchArgumentsOffset=0x70,
               tessellationFactorOffset=0xb0, pipelineStateOffset=8, vertexBufferBindOffset=0x10,
               fragmentBufferBindOffset=0x40, objectBufferBindOffset=0, meshBufferBindOffset=0,
               kernelBufferBindOffset=0x10, attributeStrideOffset=0xd0,
               objectThreadgroupMemoryLengthOffset=0, threadgroupMemoryLengthOffset=0x68,
               commandArgumentsOffset=0x78, commandSize=0x100):
    """PGIndirectCommandLayout, 52 bytes: four u16 (commandType, barrier,
    kernelDispatchArguments, tessellationFactor offsets) then eleven u32
    (pipelineState, vertex, fragment, object, mesh, kernel bind offsets,
    attributeStride, objectThreadgroupMemoryLength, threadgroupMemoryLength,
    commandArguments offsets, commandSize)
    (resources::icb::IndirectCommandLayout). The defaults are this suite's own
    layout: one 0x100-byte command per slot."""
    return struct.pack("<HHHHIIIIIIIIIII", commandTypeOffset, barrierOffset, kernelDispatchArgumentsOffset,
                       tessellationFactorOffset, pipelineStateOffset, vertexBufferBindOffset,
                       fragmentBufferBindOffset, objectBufferBindOffset, meshBufferBindOffset,
                       kernelBufferBindOffset, attributeStrideOffset, objectThreadgroupMemoryLengthOffset,
                       threadgroupMemoryLengthOffset, commandArgumentsOffset, commandSize)


def icb_descriptor(command_types, vbufs=0, fbufs=0, kbufs=0, ktg=0, flags=0, override=0,
                   obufs=0, mbufs=0, otg=0, scissors=0, viewports=0):
    """PGSerializedIndirectCommandBufferDescriptor, 0x14 bytes: u32
    commandTypes, u8 max vertex, fragment, kernel, object, mesh buffer bind
    counts, u8 kernel and object threadgroup memory bind counts, u8 max
    scissor rects, u8 max viewports, u8 unread, u16 inherit bits (bit 0
    pipeline, 1 buffers, 2 ray tracing, 3 dynamic attribute stride, 4
    depth-stencil ...), u32 allowOverrideRenderStates
    (resources::manager::icb::records::SerializedIcbDescriptor)."""
    return struct.pack("<IBBBBBBBBBBHI", command_types, vbufs, fbufs, kbufs, obufs, mbufs, ktg, otg,
                       scissors, viewports, 0, flags, override)


def icb_record(length, va_page, descriptor, layout=None, max_count=1, options=0, command_type=54,
               op_len=None):
    """Type 20: APVObjectIndirectCommandBuffer {u64 length, u32 page} (12),
    the operation header {u32 commandType 54, u32 unread}, then
    PGOperationNewIndirectCommandBufferWithDescriptor (0x50): descriptor
    (0x14), layout (0x34), u32 maxCommandCount, u16 options, 2 pad
    (resources::manager::icb::records, ObjectIcb and NewIcbCommand)."""
    op = descriptor + (layout if layout is not None else icb_layout()) + struct.pack("<IH2x", max_count, options)
    if op_len is not None:
        op = op[:op_len]
    return struct.pack("<QI", length, va_page) + struct.pack("<II", command_type, 0) + op


def icb_bind(ref, gpu=0, offset=0):
    """PGIndirectCommandBufferBind, packed 0x14: {u32 bufferReference, u64
    gpuAddress, u64 offset}; a reference of ~0 looks the GPU address up
    (resources::manager::records::IndirectCommandBufferBind)."""
    return struct.pack("<IQQ", ref, gpu, offset)


# ---- compute records (deserializer::compute) ----

def c_pipeline(ref):
    """0xd0 SetPipelineState {u32 ref}."""
    return rec(0xD0, struct.pack("<I", ref))


def c_buffers(loc, entries):
    """0xcb SetBuffers {u32 rangeLocation, u32 rangeLength}, then that many
    packed {u32 ref, u64 offset}."""
    return rec(0xCB, struct.pack("<II", loc, len(entries)) + b"".join(struct.pack("<IQ", r, o) for r, o in entries))


def c_dispatch_threads(grid, tg):
    """0xca DispatchThreads {MTLSize threadsPerGrid, MTLSize threadsPerThreadgroup}."""
    return rec(0xCA, struct.pack("<6Q", *grid, *tg))


# ---- render records (deserializer::render) ----

def render_pass(color_tex, clear=(0.0, 0.0, 1.0, 1.0), w=4, h=4, load=2, store=1, prefix=b""):
    """0x1a DescribeRenderPass (0x248, packed): depth (0x28) and stencil
    (0x24) attachments, 8 colour attachments of 0x3c ({u32 texture, u32
    resolve, u16 level, slice, depthPlane, resolve level, slice, plane, u16
    load, store, storeOptions, u16, f64 clear RGBA}), u32 visibility
    buffer, u64 renderTargetArrayLength, width, height. prefix: pass
    commands that come before it (0x1e-0x24)."""
    body = bytearray(0x248)
    c0 = 0x28 + 0x24
    struct.pack_into("<II", body, c0, color_tex, 0)
    struct.pack_into("<HH", body, c0 + 0x14, load, store)
    struct.pack_into("<4d", body, c0 + 0x1c, *clear)
    struct.pack_into("<QQQ", body, 0x230, 0, w, h)
    return bytes(prefix) + rec(0x1A, bytes(body))


def r_pipeline(ref):
    """0x74 SetRenderPipelineState {u32 ref}."""
    return rec(0x74, struct.pack("<I", ref))


def r_draw(prim=3, start=0, count=3):
    """0x01 DrawPrimitives16 {u32 primitiveType, u16 start, u16 count}."""
    return rec(0x01, struct.pack("<IHH", prim, start, count))


# ---- info records (deserializer::info) ----

def info_ref(cmd, obj, reply, off=0):
    """450, 457, 461-468: {u32 object, u32 replyBuf, u64 offset}, 0x10."""
    return rec(cmd, struct.pack("<IIQ", obj, reply, off))


def info_heap_size(reply, off=0, v2=False, **desc):
    """451: v1 descriptor (0x20), replyBuf@0x20, offset@0x24 (0x2c, packed).
    469: v2 descriptor (0x27, packed), replyBuf@0x28, offset@0x2c (0x34,
    pack 4)."""
    if not v2:
        return rec(451, texture_desc_v1(**desc) + struct.pack("<IQ", reply, off))
    return rec(469, texture_desc_v2(**desc) + b"\0" + struct.pack("<IQ", reply, off))


def texture_desc_v2(fmt=70, w=4, h=4, usage=5, ttype=2, mips=1, samples=1, array=1, options=0x20,
                    swizzle=(2, 3, 4, 5), write_swizzle=0):
    """Serialized texture descriptor v2, packed, 0x27 bytes: byte 0
    {textureType:4, framebufferOnly:1, isDrawable:1, allowGPUOptimized:1,
    writeSwizzleEnabled:1}, u16 pixelFormat @1, u32 usage @3, width @7,
    height @0xb, depth @0xf, u16 mips @0x13, samples @0x15, array @0x17,
    options @0x19, u64 protection @0x1b, swizzle RGBA @0x23
    (deserializer::texture_descriptor::SerializedTextureDescriptor2)."""
    return (struct.pack("<BHIIIIHHHHQ", ttype | (write_swizzle << 7), fmt, usage, w, h, 1, mips, samples,
                        array, options, 0) + bytes(swizzle))


def submit_last(s, opcode, payload, nops=256, label=None):
    """Submits a command expected to kill the process, behind nops NOPs in
    the same doorbell, then waits for a fault that never comes. The NOPs
    keep the FIFO busy past the doorbell's MMIO write, so the death lands
    in the wait op every run rather than racing the doorbell op
    (docs/testing.md, section 5.5: the op a process dies at is part of the
    outcome). A death inside Metal can also land on another thread after
    the call that caused it has returned: Metal aborts on its compiler's
    callback thread after a render pass with no attachment and a target
    size (render_pass_no_target, interplay_decodewalk_24). So the fatal
    command goes last, its doorbell is rung last, and the process dies in
    the wait after it, whichever thread the death is on."""
    f = s.fifo
    for _ in range(nops):
        f.cmd(abi.OP_NOP)
    f.cmd(opcode, payload)
    f.doorbell()
    s.wait_fault(label=label)


class ObjTask(GPUTask):
    """A GPUTask that also places fixtures (metallibs) in its blob region and
    builds the objects the info and ICB scenarios use."""

    def fixture(self, rel):
        va = self.reserve(0x2000)
        n = self.s.mem_file(self.gpa(va), rel)
        assert n <= 0x2000
        return va, n

    def function(self, rel):
        va, n = self.fixture(rel)
        return self.ol.add(abi.T_FUNCTION, function_record(va, n))

    def compute_pipeline(self, rel="shaders/kernel.metallib", icb=False):
        fn = self.function(rel)
        return self.ol.add(abi.T_SERIALIZED, pipeline_record(11, compute_blob(fn, icb)))

    def render_pipeline(self, fmt=70, icb=False):
        v = self.function("shaders/vert.metallib")
        f = self.function("shaders/frag.metallib")
        return self.ol.add(abi.T_SERIALIZED, pipeline_record(14, render_blob(v, f, fmt, icb)))

    def relation_kernel(self):
        """fixtures/shaders/rel.metal: for pair i of u64s in buffer 0, u32
        bits in buffer 1: 1 equal, 2 first nonzero, 4 second nonzero. How a scenario locks a relation between host
        values that differ from run to run (docs/testing.md, section 5.8)."""
        if not hasattr(self, "_rel"):
            self._rel = self.compute_pipeline("shaders/rel.metallib")
        return self._rel

    def relate(self, pairs_ref, out_ref, count):
        """Executes the relation kernel over count pairs; the caller dumps
        out_ref. Both buffers are listed (the isolate setup needs every
        buffer the GPU touches listed, section 6.12)."""
        self.exec_segments([seg(abi.ENC_COMPUTE, c_pipeline(self._rel) + c_buffers(0, [(pairs_ref, 0), (out_ref, 0)])
                                + c_dispatch_threads((count, 1, 1), (count, 1, 1)))],
                           resources=[(pairs_ref, 0), (out_ref, 0)], label="relations")

    def exec_last(self, segments, resources=(), opcode=abi.OP_EXEC_INDIRECT3, label=None):
        """exec_segments for a command expected to kill the process
        (submit_last)."""
        self.reset_cmdbufs()
        cbs = [self.cmdbuf(sg) for sg in segments]
        p = struct.pack("<III", self.tid, len(resources), len(cbs))
        p += b"".join(struct.pack("<II", r, f) for r, f in resources)
        p += b"".join(struct.pack("<QQ", va, n) for va, n in cbs)
        submit_last(self.s, opcode, p, label=label)

    def exec_segments(self, segments, resources=(), opcode=abi.OP_EXEC_INDIRECT3, **kw):
        """Places each segment as its own range and executes them."""
        self.reset_cmdbufs()
        cbs = [self.cmdbuf(sg) for sg in segments]
        return self.exec3(cbs, resources=resources, opcode=opcode, **kw)
