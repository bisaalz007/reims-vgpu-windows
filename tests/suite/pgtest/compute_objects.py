"""Builders for the objects the blit and compute decoders reference:
functions, compute pipelines, fences, samplers, textures over buffers,
indirect command buffers and heaps, and the decoders' records
(docs/testing.md, sections 6.11, 6.14, 6.15, 6.18, 6.19).

Each layout names the reims-vgpu-core module or item that reads it.
Nothing here is imported from the library's code.
"""
import struct

from . import abi
from .abi import PAGE
from .scenario import Scenario
from .task import GPUTask, rec, seg

# ---- object records ----

# PGOperationHeader (resources::manager::records::OperationHeader): {u32
# commandType, u32 unread}, 8 bytes, after the per-type prefix.
OP_HDR = struct.Struct("<II")

CT_SAMPLER = 3
CT_TEXTURE_ON_BUFFER = 9
CT_COMPUTE_PIPELINE = 11
CT_FENCE = 13
CT_FUNCTION = 15
CT_ICB = 54

# MTLPixelFormat values.
PF_R8UNORM = 10
PF_R32UINT = 53
PF_RGBA8UNORM = 70
PF_RGBA8UINT = 73
PF_BC1_RGBA = 130

# MTLTextureUsage bits.
U_READ, U_WRITE, U_RT, U_PFV = 1, 2, 4, 0x10
TEX_2D = 2
TEX_2D_ARRAY = 3


def function_record(va, length, command_type=CT_FUNCTION):
    """A function (type 6): ObjectFunction {u64 task VA of the metallib,
    u32 its length} (packed, 12 bytes), then OperationHeader {u32
    commandType 15, u32 unread} (resources::manager::create::function)."""
    return struct.pack("<QI", va, length) + OP_HDR.pack(command_type, 0)


def compute_pipeline_record(blob, descriptor_length=None, command_type=CT_COMPUTE_PIPELINE):
    """A compute pipeline (type 7, commandType 11): OperationHeader, then
    {u32 unread, u32 descriptorLength}, then the serialized descriptor
    (resources::manager::create::pipeline). Metal
    is handed entry length - 16 bytes, so the record is exactly 16 + the
    blob (section 6.18)."""
    n = len(blob) if descriptor_length is None else descriptor_length
    return OP_HDR.pack(command_type, 0) + struct.pack("<II", 0, n) + bytes(blob)


def compute_blob(fn_ref, icb=False, max_threads=None):
    """Metal's serialized compute pipeline descriptor (section 6.18): a u8
    count, then {u8 key, u8 size 4, u32 value}; key 0 the function ref, key 7
    supportIndirectCommandBuffers, key 8 maxTotalThreadsPerThreadgroup
    (keys measured with -[MTLDevice serializeComputePipelineDescriptor:] on
    this host)."""
    entries = []
    if icb:
        entries.append((7, 1))
    if max_threads is not None:
        entries.append((8, max_threads))
    entries.append((0, fn_ref))
    return bytes([len(entries)]) + b"".join(struct.pack("<BBI", k, 4, v) for k, v in entries)


def stage_in_blob(fn_ref):
    """A compute descriptor with a stage-input descriptor: attribute 0 UInt
    (format 36) from buffer 1, layout 1 stride 4, step function
    ThreadPositionInGridX (5). Serialized on this host with
    -serializeComputePipelineDescriptor: from stagein.metallib, the function
    written as ref fn_ref; the nested blocks follow the entries at the
    offsets key 3 and the arrays give."""
    blob = bytearray.fromhex(
        "02 03 04 10 00 00 00 00 04 00 00 00 00 00 00 00 02 00 04 10 00 00 00 01 04 2c 00 00 00 00 00 00"
        " 01 00 00 00 08 00 00 00 03 00 04 00 00 00 00 01 04 24 00 00 00 03 04 01 00 00 00 00 01 00 00 00"
        " 08 00 00 00 03 00 04 01 00 00 00 01 04 05 00 00 00 03 04 04 00 00 00")
    struct.pack_into("<I", blob, 9, fn_ref)
    return bytes(blob)


def fence_record(command_type=CT_FENCE):
    """A fence (type 7, commandType 13): the header alone
    (resources::manager::create::fence)."""
    return OP_HDR.pack(command_type, 0)


def sampler_record(min_filter=0, mag_filter=0, s_mode=0, t_mode=0, r_mode=0, mip_filter=0,
                   normalized=1, lod_min=0.0, lod_max=1000.0, compare=0, max_aniso=1,
                   border=0, arg_buffers=0, reduction=0):
    """A sampler (type 7, commandType 3): OperationHeader, then
    NewSamplerState (0x1c, resources::manager::create::sampler): u32
    unread; u32 {minFilter:2, magFilter:2, lodAverage:1, compare:3, s:4,
    t:4, r:4, border:4, mipFilter:2, maxAnisotropy:5, normalized:1}; u32
    {supportArgumentBuffers:1, reductionMode:3}; f32 lodMin; f32 lodMax; 8
    bytes unread. Address modes: 0 ClampToEdge, 2 Repeat."""
    bits = (min_filter | mag_filter << 2 | compare << 5 | s_mode << 8 | t_mode << 12 | r_mode << 16
            | border << 20 | mip_filter << 24 | max_aniso << 26 | normalized << 31)
    bits2 = arg_buffers | reduction << 1
    return OP_HDR.pack(CT_SAMPLER, 0) + struct.pack("<IIIff8x", 0, bits, bits2, lod_min, lod_max)


def texture_descriptor(w, h=1, d=1, fmt=PF_RGBA8UNORM, usage=U_READ | U_WRITE | U_RT, ttype=TEX_2D,
                       mips=1, samples=1, array=1, options=0, protection=0):
    """Serialized texture descriptor v1, 32 bytes
    (deserializer::texture_descriptor::SerializedTextureDescriptor): u32
    {textureType:4, framebufferOnly:1, isDrawable:1,
    allowGPUOptimizedContents:1, :1, usage:8, pixelFormat:16}, u32 width,
    height, depth, u16 mips, samples, arrayLength, resourceOptions, u64
    protectionOptions."""
    word0 = ttype | usage << 8 | fmt << 16
    return struct.pack("<IIIIHHHHQ", word0, w, h, d, mips, samples, array, options, protection)


def texture_on_buffer_record(buf_ref, offset, bytes_per_row, desc, command_type=CT_TEXTURE_ON_BUFFER):
    """A texture over a buffer (type 8, commandType 9): OperationHeader,
    then OpViewWithBuffer {u32 unread, u32 bufferRef, u64 offset, u64
    bytesPerRow, v1 descriptor}, 0x38 bytes (resources::manager::texture)."""
    return OP_HDR.pack(command_type, 0) + struct.pack("<IIQQ", 0, buf_ref, offset, bytes_per_row) + desc


def heap_record(page, length, flags=1):
    """A heap (type 13): packed {u32 page, u64 length, u8 flags} read as 16
    bytes; flags bit 0 private (resources::manager::records::ObjectHeap).
    In-process heaps must be private."""
    return struct.pack("<IQB3x", page, length, flags)


# PGIndirectCommandLayout (resources::icb::IndirectCommandLayout): u16
# commandTypeOffset, barrierOffset, kernelDispatchArgumentsOffset,
# tessellationFactorOffset, then u32 pipelineStateOffset, vertex, fragment,
# object, mesh, kernel buffer bind offsets, attributeStrideOffset,
# objectThreadgroupMemoryLengthOffset, threadgroupMemoryLengthOffset,
# commandArgumentsOffset, commandSize: 52 bytes.
ICB_LAYOUT = struct.Struct("<HHHHIIIIIIIIIII")
assert ICB_LAYOUT.size == 0x34
# The compute command layout this suite uses, one command every 0x100 bytes:
ICMD_TYPE = 0x00          # u32 MTLIndirectCommandType (0 none: reset)
ICMD_BARRIER = 0x04       # u32 hasBarrier
ICMD_DISPATCH = 0x08      # PGIndirectCommandDispatch, 0x38: stage-in size, origin, imageblock w, h
ICMD_PIPELINE = 0x40      # u32 pipeline reference
ICMD_KBUF = 0x48          # maxKernelBufferBindCount x {u32 ref, u64 gpuAddress, u64 offset}
ICMD_TGMEM = 0x80         # maxKernelThreadgroupMemoryBindCount x u64
ICMD_ARGS = 0xa0          # {MTLSize, MTLSize}: the dispatch
ICMD_SIZE = 0x100
ICMD_CONCURRENT_DISPATCH = 0x20          # MTLIndirectCommandTypeConcurrentDispatch
ICMD_CONCURRENT_DISPATCH_THREADS = 0x40  # MTLIndirectCommandTypeConcurrentDispatchThreads


def icb_record(page, length, max_commands, command_types=ICMD_CONCURRENT_DISPATCH, kernel_buffers=1,
               tg_mem=0, options=0, inherit_pipeline=0, inherit_buffers=0):
    """An indirect command buffer (type 20): APVObjectIndirectCommandBuffer
    {u64 length, u32 page} (packed, 12), PGOperationHeader (54), then
    PGOperationNewIndirectCommandBufferWithDescriptor (0x50): descriptor
    0x14 {u32 commandTypes, u8 maxVertex, maxFragment, maxKernel, maxObject,
    maxMesh buffer binds, maxKernelThreadgroupMemory, maxObjectThreadgroupMemory,
    maxScissor, maxViewport, unread, u16 inherit bits (bit 0 pipeline, 1
    buffers), u32 allowOverrideRenderStates}, layout 0x34, u32
    maxCommandCount, u16 options (resources::manager::icb::records,
    NewIcbCommand)."""
    desc = struct.pack("<IBBBBBBBBBBHI", command_types, 0, 0, kernel_buffers, 0, 0, tg_mem, 0, 0, 0, 0,
                       inherit_pipeline | inherit_buffers << 1, 0)
    assert len(desc) == 0x14
    layout = ICB_LAYOUT.pack(ICMD_TYPE, ICMD_BARRIER, ICMD_DISPATCH, 0, ICMD_PIPELINE, 0, 0, 0, 0,
                             ICMD_KBUF, 0, 0, ICMD_TGMEM, ICMD_ARGS, ICMD_SIZE)
    op = desc + layout + struct.pack("<IH2x", max_commands, options)
    assert len(op) == 0x50
    return struct.pack("<QI", length, page) + OP_HDR.pack(CT_ICB, 0) + op


def icb_compute_command(ctype=ICMD_CONCURRENT_DISPATCH, pipeline=0, buffers=(), grid=(1, 1, 1),
                        tptg=(1, 1, 1), barrier=0, tg_lengths=()):
    """One indirect compute command in the layout above (read by
    resources::manager::icb::prepare, written back by its synchronize):
    buffers are (reference, offset) binds (gpuAddress 0)."""
    c = bytearray(ICMD_SIZE)
    struct.pack_into("<II", c, ICMD_TYPE, ctype, barrier)
    struct.pack_into("<I", c, ICMD_PIPELINE, pipeline)
    for i, (ref, off) in enumerate(buffers):
        struct.pack_into("<IQQ", c, ICMD_KBUF + 0x14 * i, ref, 0, off)
    for i, n in enumerate(tg_lengths):
        struct.pack_into("<Q", c, ICMD_TGMEM + 8 * i, n)
    struct.pack_into("<6Q", c, ICMD_ARGS, *grid, *tptg)
    return bytes(c)


# ---- decoder records (deserializer::records) ----

def size3(v):
    return struct.pack("<3Q", *v)


class Blit:
    """Blit decoder records, 300-323 (deserializer::blit, layouts
    deserializer::records; MTLSize and MTLOrigin are 3 x u64)."""

    @staticmethod
    def copy_buffer_to_texture(buf, tex, src_off, bpr, bpi, size, origin, slice_=0, level=0, options=0):
        """300, PGCmdCopyFromBufferToTexture, 0x58."""
        return rec(300, struct.pack("<IIQQQ", buf, tex, src_off, bpr, bpi) + size3(size) + size3(origin)
                   + struct.pack("<HHI", slice_, level, options))

    @staticmethod
    def copy_buffer_to_buffer(src, dst, src_off, dst_off, size):
        """301, PGCmdCopyFromBufferToBuffer, 0x20."""
        return rec(301, struct.pack("<IIQQQ", src, dst, src_off, dst_off, size))

    @staticmethod
    def copy_texture_to_buffer(tex, buf, origin, size, dst_off, bpr, bpi, slice_=0, level=0, options=0):
        """302, PGCmdCopyFromTextureToBuffer, 0x58 (tail padding included)."""
        return rec(302, struct.pack("<II", tex, buf) + size3(origin) + size3(size)
                   + struct.pack("<QQQHHH2x", dst_off, bpr, bpi, slice_, level, options))

    @staticmethod
    def copy_texture_to_texture(src, dst, src_origin, size, dst_origin, ss=0, sl=0, ds=0, dl=0):
        """303, PGCmdCopyFromTextureToTexture, 0x58."""
        return rec(303, struct.pack("<II", src, dst) + size3(src_origin) + size3(size) + size3(dst_origin)
                   + struct.pack("<HHHH", ss, sl, ds, dl))

    @staticmethod
    def copy_texture_to_texture_options(src, dst, src_origin, size, dst_origin, ss=0, sl=0, ds=0, dl=0,
                                        options=0):
        """304, PGCmdCopyFromTextureToTextureWithOptions, packed 0x5c."""
        return rec(304, struct.pack("<II", src, dst) + size3(src_origin) + size3(size) + size3(dst_origin)
                   + struct.pack("<HHHHI", ss, sl, ds, dl, options))

    @staticmethod
    def copy_icb(src, dst, loc, length, dst_index):
        """305, PGCmdCopyIndirectCommandBuffer, 0x20."""
        return rec(305, struct.pack("<IIQQQ", src, dst, loc, length, dst_index))

    @staticmethod
    def fill(buf, loc, length, value):
        """306, PGCmdFillBuffer, packed 0x18."""
        return rec(306, struct.pack("<IQQB3x", buf, loc, length, value))

    @staticmethod
    def mipmaps(tex):
        """307, PGCmdGenerateMipmaps, 4."""
        return rec(307, struct.pack("<I", tex))

    @staticmethod
    def optimize(cmd, tex):
        """308 CPU / 309 GPU, PGCmdOptimize, 4."""
        return rec(cmd, struct.pack("<I", tex))

    @staticmethod
    def optimize_image(cmd, tex, slice_=0, level=0):
        """310 CPU / 311 GPU, PGCmdOptimizeImage, 8."""
        return rec(cmd, struct.pack("<IHH", tex, slice_, level))

    @staticmethod
    def icb_range(cmd, icb, loc, length):
        """312 optimize / 313 reset, PGCmdOptimizeResetIndirectBuffer, packed 0x14."""
        return rec(cmd, struct.pack("<IQQ", icb, loc, length))

    @staticmethod
    def synchronize(ref):
        """314, PGCmdSynchronize, 4."""
        return rec(314, struct.pack("<I", ref))

    @staticmethod
    def synchronize_image(tex, slice_=0, level=0):
        """315, PGCmdSynchronizeImage, 8."""
        return rec(315, struct.pack("<IHH", tex, slice_, level))

    @staticmethod
    def fence(cmd, fence):
        """316 update / 317 wait, PGCmdFenceOp, 4."""
        return rec(cmd, struct.pack("<I", fence))

    @staticmethod
    def copy_slices(src, dst, ss=0, sl=0, ds=0, dl=0, slices=1, levels=1):
        """318, PGCmdCopyFromTextureToTextureWithNumSliceLevel, 0x14."""
        return rec(318, struct.pack("<IIHHHHHH", src, dst, ss, sl, ds, dl, slices, levels))

    @staticmethod
    def fill_pattern(buf, loc, length, pattern):
        """319, PGCmdFillBufferWithPattern, packed 0x18."""
        return rec(319, struct.pack("<IQQI", buf, loc, length, pattern))

    @staticmethod
    def fill_texture_bytes(tex, level, slice_, size, origin, buf, off, length):
        """320, PGCmdFillTextureWithBytes, packed 0x4c: the region's size
        before its origin."""
        return rec(320, struct.pack("<IHH", tex, level, slice_) + size3(size) + size3(origin)
                   + struct.pack("<IQQ", buf, off, length))

    @staticmethod
    def fill_texture_color(tex, level, slice_, size, origin, color, fmt):
        """321, PGCmdFillTextureWithColor, packed 0x5c: color 4 x f64, u16
        pixelFormat, 2 bytes of padding."""
        return rec(321, struct.pack("<IHH", tex, level, slice_) + size3(size) + size3(origin)
                   + struct.pack("<4dH2x", *color, fmt))

    @staticmethod
    def invalidate(tex):
        """322, PGCmdInvalidateCompressedTexture, 4."""
        return rec(322, struct.pack("<I", tex))

    @staticmethod
    def invalidate_image(tex, slice_=0, level=0):
        """323, PGCmdInvalidateCompressedTextureImage, 8."""
        return rec(323, struct.pack("<IHH", tex, slice_, level))


class Compute:
    """Compute decoder records (deserializer::compute, layouts
    deserializer::records)."""

    @staticmethod
    def dispatch_threadgroups(groups, tptg):
        """0xc8, PGCmdDispatchThreadgroups, 0x30."""
        return rec(0xC8, size3(groups) + size3(tptg))

    @staticmethod
    def dispatch_threadgroups_indirect(tptg, off, buf):
        """0xc9, PGCmdDispatchThreadgroupsIndirect, packed 0x24."""
        return rec(0xC9, size3(tptg) + struct.pack("<QI", off, buf))

    @staticmethod
    def dispatch_threads(grid, tptg):
        """0xca, PGCmdDispatchThreads, 0x30."""
        return rec(0xCA, size3(grid) + size3(tptg))

    @staticmethod
    def set_buffers(loc, entries, count=None):
        """0xcb: PGCmdSetBuffers {u32 location, u32 length}, then length x
        PGCmdSetBufferEntry {u32 ref, u64 offset} (packed 0xc)."""
        n = len(entries) if count is None else count
        return rec(0xCB, struct.pack("<II", loc, n) + b"".join(struct.pack("<IQ", r, o) for r, o in entries))

    @staticmethod
    def set_samplers(loc, refs, count=None):
        """0xcc: {u32 location, u32 length}, then length x u32 ref."""
        n = len(refs) if count is None else count
        return rec(0xCC, struct.pack("<II", loc, n) + b"".join(struct.pack("<I", r) for r in refs))

    @staticmethod
    def set_samplers_lod(loc, entries, count=None):
        """0xcd: {u32 location, u32 length}, then length x {u32 ref, f32
        lodMin, f32 lodMax}."""
        n = len(entries) if count is None else count
        return rec(0xCD, struct.pack("<II", loc, n) + b"".join(struct.pack("<Iff", *e) for e in entries))

    @staticmethod
    def set_textures(loc, refs, count=None):
        """0xce: {u32 location, u32 length}, then length x u32 ref."""
        n = len(refs) if count is None else count
        return rec(0xCE, struct.pack("<II", loc, n) + b"".join(struct.pack("<I", r) for r in refs))

    @staticmethod
    def set_buffer_offset(index, off):
        """0xcf, PGCmdSetBufferOffset {u32 index, u64 offset}, packed 0xc."""
        return rec(0xCF, struct.pack("<IQ", index, off))

    @staticmethod
    def set_pipeline(ref):
        """0xd0, {u32 ref}."""
        return rec(0xD0, struct.pack("<I", ref))

    @staticmethod
    def stage_in_region(size, origin):
        """0xd1, PGCmdSetComputeStageInRegion: size first, then origin, 0x30."""
        return rec(0xD1, size3(size) + size3(origin))

    @staticmethod
    def stage_in_region_indirect(buf, off):
        """0xd2, PGCmdSetComputeStageInRegionIndirect {u32 ref, u64 offset},
        packed 0xc."""
        return rec(0xD2, struct.pack("<IQ", buf, off))

    @staticmethod
    def threadgroup_memory(length, index):
        """0xd3, PGCmdSetThreadgroupMemoryLength {u64 length, u32 index},
        packed 0xc."""
        return rec(0xD3, struct.pack("<QI", length, index))

    @staticmethod
    def fence(cmd, fence):
        """0xd4 update / 0xd5 wait, {u32 ref}."""
        return rec(cmd, struct.pack("<I", fence))

    @staticmethod
    def barrier_resources(refs, count=None):
        """0xd6: {u32 count}, then count x u32 ref."""
        n = len(refs) if count is None else count
        return rec(0xD6, struct.pack("<I", n) + b"".join(struct.pack("<I", r) for r in refs))

    @staticmethod
    def barrier_scope(scope):
        """0xd7, PGCmdMemoryBarrierScope, a u32 with 16 bits used."""
        return rec(0xD7, struct.pack("<I", scope))

    @staticmethod
    def imageblock(w, h):
        """0xd8, PGCmdSetImageblockWidth {u32 width, u32 height}."""
        return rec(0xD8, struct.pack("<II", w, h))

    @staticmethod
    def set_buffers_stride(loc, entries, count=None):
        """0xd9: {u32 location, u32 length}, then length x {u32 ref, u64
        offset, u64 attributeStride} (packed 0x14)."""
        n = len(entries) if count is None else count
        return rec(0xD9, struct.pack("<II", loc, n) + b"".join(struct.pack("<IQQ", *e) for e in entries))

    @staticmethod
    def set_buffer_offset_stride(index, off, stride):
        """0xda, PGCmdSetBufferOffsetWithStride {u32 index, u64 offset, u64
        stride}, packed 0x14."""
        return rec(0xDA, struct.pack("<IQQ", index, off, stride))

    @staticmethod
    def dispatch_type(t):
        """0xdb, PGCmdSetComputePassDescriptorDispatchType {u32}."""
        return rec(0xDB, struct.pack("<I", t))

    @staticmethod
    def jump(cmd, buf=None, off=0, comparison=0, value=0):
        """0xdc StartDoWhile, 0xdf EndWhile, 0xe1 StartElse, 0xe2 EndIf: no
        payload; 0xdd EndDoWhile, 0xde StartWhile, 0xe0 StartIf: packed 0x14
        {u32 buffer ref, u64 offset, u32 MTLCompareFunction, u32 value}."""
        if buf is None:
            return rec(cmd)
        return rec(cmd, struct.pack("<IQII", buf, off, comparison, value))

    @staticmethod
    def flush():
        """0xe3 InsertCompressedTextureReinterpretationFlush: no payload."""
        return rec(0xE3)

    @staticmethod
    def execute_icb(icb, loc, length):
        """0xe4, PGCmdExecuteCommandsInBufferRanged {u32 icb, u64 location,
        u64 length}, packed 0x14."""
        return rec(0xE4, struct.pack("<IQQ", icb, loc, length))

    @staticmethod
    def execute_icb_indirect(icb, range_buf, off):
        """0xe5, PGCmdExecuteCommandsInBuffer {u32 icb, u32 range buffer,
        u64 offset}, 0x10."""
        return rec(0xE5, struct.pack("<IIQ", icb, range_buf, off))

    @staticmethod
    def dispatch_threads_indirect(off, buf):
        """0xe6, PGCmdDispatchThreadsIndirect {u64 offset, u32 buffer},
        packed 0xc."""
        return rec(0xE6, struct.pack("<QI", off, buf))

    @staticmethod
    def use_heaps(refs, count=None):
        """0x86: {u32 count}, then count x u32 ref."""
        n = len(refs) if count is None else count
        return rec(0x86, struct.pack("<I", n) + b"".join(struct.pack("<I", r) for r in refs))

    @staticmethod
    def use_resources(refs, usage=3, count=None):
        """0x87: {u32 count, u32 MTLResourceUsage}, then count x u32 ref."""
        n = len(refs) if count is None else count
        return rec(0x87, struct.pack("<II", n, usage) + b"".join(struct.pack("<I", r) for r in refs))


# ---- a GPU scenario ----

class Rig:
    """A scenario with one GPU task (GPUTask: object list at VA 0, records
    at VA pages 1-3, command buffers at pages 4-7, blobs from page 8), and
    helpers to add objects and run segments.

    Refs are handed out as objects are added; call ready() once every
    object is in, which defines the task and installs the object list."""

    def __init__(self, name, doc, pages=64, **kw):
        self.s = Scenario(name, doc=doc, **kw)
        self.s.bring_up()
        self.t = GPUTask(self.s, pages=pages)
        self.buffers = []  # refs the GPU touches: listed in every exec (section 6.12)
        self.irqs = 0  # interrupts so far, while every command completed or faulted once
        # A new process's interrupts count from 0 (Scenario.phase).
        self.s.on_phase(lambda: setattr(self, "irqs", 0))

    # objects

    def buffer(self, length=PAGE, contents=None, flags=0, listed=True):
        ref, va = self.t.buffer(length=length, flags=flags, contents=contents)
        if listed:
            self.buffers.append(ref)
        return ref, va

    def function(self, fixture="shaders/kernel.metallib", command_type=CT_FUNCTION, length=None):
        va = self.t.reserve(0x2000)
        n = self.s.mem_file(self.t.gpa(va), fixture)
        return self.t.ol.add(abi.T_FUNCTION, function_record(va, n if length is None else length, command_type))

    def function_bytes(self, data, command_type=CT_FUNCTION, length=None):
        va = self.t.blob(data)
        return self.t.ol.add(abi.T_FUNCTION, function_record(va, len(data) if length is None else length,
                                                             command_type))

    def pipeline(self, fn_ref=None, blob=None, **kw):
        if blob is None:
            blob = compute_blob(fn_ref, **kw)
        return self.t.ol.add(abi.T_SERIALIZED, compute_pipeline_record(blob))

    def kernel(self, fixture="shaders/kernel.metallib", **kw):
        """A function and a pipeline over it; returns the pipeline's ref.
        The function's ref is the pipeline's minus one."""
        f = self.function(fixture)
        return self.pipeline(f, **kw)

    def fence(self):
        return self.t.ol.add(abi.T_SERIALIZED, fence_record())

    def sampler(self, **kw):
        return self.t.ol.add(abi.T_SERIALIZED, sampler_record(**kw))

    def texture(self, buf_ref, w, h=1, bpr=None, offset=0, typ=abi.T_SERIALIZER_TEXTURE, **kw):
        """A texture over a buffer; bytesPerRow defaults to 4w rounded up to
        16, what minimumLinearTextureAlignment asks of RGBA8 on this GPU."""
        if bpr is None:
            bpr = (4 * w + 15) // 16 * 16
        return self.t.ol.add(typ, texture_on_buffer_record(buf_ref, offset, bpr, texture_descriptor(w, h, **kw)))

    def heap(self, length=PAGE, flags=1, listed=False):
        va = self.t.reserve(length)
        ref = self.t.ol.add(abi.T_HEAP, heap_record(va // PAGE, length, flags))
        if listed:
            self.buffers.append(ref)
        return ref

    def icb(self, commands, max_commands=None, listed=True, **kw):
        """An ICB whose guest copy holds these commands (icb_compute_command).
        listed: in every exec's resource list, which prepares it from guest
        memory before the exec."""
        n = len(commands) if max_commands is None else max_commands
        length = ICMD_SIZE * n
        va = self.t.reserve(max(length, 1))
        data = b"".join(commands)
        if data:
            self.t.write(va, data)
        ref = self.t.ol.add(abi.T_ICB, icb_record(va // PAGE, length, n, **kw))
        if listed:
            self.buffers.append(ref)
        return ref

    def raw(self, typ, record):
        return self.t.ol.add(typ, record)

    def ready(self):
        self.t.define()
        self.t.install()
        self.irqs += 2

    # execution

    def resources(self, extra=()):
        return [(r, 0) for r in self.buffers] + list(extra)

    def run(self, *segments, fault=False, either=False, label=None, resources=None):
        """One exec (67) of these segments, one command-buffer range each.
        With fault=True, waits for the fault, records it and resumes.
        With either=True, for an exec that faults in some setups and
        completes in others: waits for its interrupt (a stamp and a fault
        raise one each), records the fault state, resumes, and waits for the
        stamp, which a resume publishes. The interrupt count is unknown
        after that, so either=True works once per scenario."""
        self.t.reset_cmdbufs()
        cbs = [self.t.cmdbuf(sg) for sg in segments]
        res = self.resources() if resources is None else resources
        if either:
            assert self.irqs is not None, "either=True once per scenario"
            p = struct.pack("<III", self.t.tid, len(res), len(cbs))
            p += b"".join(struct.pack("<II", r_, f) for r_, f in res)
            p += b"".join(struct.pack("<QQ", va, n) for va, n in cbs)
            st = self.s.fifo.cmd(abi.OP_EXEC_INDIRECT3, p)
            self.s.doorbell()
            self.s.wait(irqs=self.irqs + 1, label=label or "the exec's interrupt")
            self.irqs = None
            self.s.fault_state(label)
            self.s.resume_root()
            self.s.fifo.wait_stamp(st, label="its stamp, after the resume")
            return
        self.t.exec3(cbs, resources=res, fault=fault, label=label)
        if self.irqs is not None:
            self.irqs = None if fault else self.irqs + 1
        if fault:
            self.s.fault_state(label)
            self.s.resume_root()

    def blit(self, *records, **kw):
        self.run(seg(abi.ENC_BLIT, b"".join(records)), **kw)

    def compute(self, *records, **kw):
        self.run(seg(abi.ENC_COMPUTE, b"".join(records)), **kw)

    def dump(self, va, n, label=None):
        self.t.dump(va, n, label)

    def end(self):
        self.t.delete()
        self.s.events("end")
        return self.s
