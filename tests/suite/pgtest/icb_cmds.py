"""Indirect command buffers as the guest lays them out: the suite's two
command layouts and each command's bytes (docs/testing.md, section 6.19,
and section 4.3's ICB write-back).

The library reads a command of layout.commandSize bytes per slot, each
field at the offset the layout gives it: prepare reads them into the Metal
ICB, synchronize writes the Metal ICB's commands back in the same places
(reims-vgpu-core's resources::manager::icb::prepare and ::synchronize).
The commands below name their structs in resources::manager::icb::records
(the bind: IndirectCommandBufferBind, resources::manager::records).
"""
import struct

from .exec_objects import icb_bind, icb_layout

CMD = 0x100  # commandSize of both layouts

# A compute layout: type @0, barrier @4, pipeline @8, kernel binds @0x10
# (4 x 0x14), threadgroup memory lengths @0x60 (2 x 8), dispatch arguments
# @0x70 (2 MTLSize), kernel dispatch arguments (Dispatch, 0x38) @0xa0,
# attribute strides @0xd8 (4 x 8).
COMPUTE = dict(commandTypeOffset=0, barrierOffset=4, kernelDispatchArgumentsOffset=0xA0,
               tessellationFactorOffset=0, pipelineStateOffset=8, vertexBufferBindOffset=0,
               fragmentBufferBindOffset=0, kernelBufferBindOffset=0x10, attributeStrideOffset=0xD8,
               threadgroupMemoryLengthOffset=0x60, commandArgumentsOffset=0x70, commandSize=CMD)

# A render layout: type @0, barrier @4, pipeline @8, vertex binds @0x10 (2 x
# 0x14), fragment binds @0x38 (2 x 0x14), draw arguments @0x60 (up to 0x4c),
# tessellation factor arguments @0xb0 (0x20), attribute strides @0xd0 (2 x 8).
RENDER = dict(commandTypeOffset=0, barrierOffset=4, kernelDispatchArgumentsOffset=0,
              tessellationFactorOffset=0xB0, pipelineStateOffset=8, vertexBufferBindOffset=0x10,
              fragmentBufferBindOffset=0x38, kernelBufferBindOffset=0, attributeStrideOffset=0xD0,
              threadgroupMemoryLengthOffset=0, commandArgumentsOffset=0x60, commandSize=CMD)


def layout(kind, **over):
    d = dict(COMPUTE if kind == "compute" else RENDER)
    d.update(over)
    return icb_layout(**d)


def compute_cmd(ctype=0x40, pso=0, binds=(), tg_lengths=(), barrier=0, grid=(16, 1, 1), tg=(16, 1, 1),
                stage_size=(0, 0, 0), stage_origin=(0, 0, 0), imageblock=(0, 0), strides=()):
    """One compute command in the COMPUTE layout. ctype: 0 none, 0x20
    concurrentDispatchThreadgroups (grid is threadgroups), 0x40
    concurrentDispatchThreads (ConcurrentDispatch, two MTLSize); binds:
    (ref, gpuAddress, offset) each (IndirectCommandBufferBind); the dispatch
    arguments (Dispatch: stage-in size, origin, imageblock w, h)."""
    b = bytearray(CMD)
    struct.pack_into("<III", b, 0, ctype, barrier, pso)
    for i, bind in enumerate(binds):
        b[0x10 + 0x14 * i:0x10 + 0x14 * (i + 1)] = icb_bind(*bind)
    for i, n in enumerate(tg_lengths):
        struct.pack_into("<Q", b, 0x60 + 8 * i, n)
    struct.pack_into("<6Q", b, 0x70, *grid, *tg)
    struct.pack_into("<6QII", b, 0xA0, *stage_size, *stage_origin, *imageblock)
    for i, n in enumerate(strides):
        struct.pack_into("<Q", b, 0xD8 + 8 * i, n)
    return bytes(b)


def render_cmd(ctype=1, pso=0, vbinds=(), fbinds=(), barrier=0, prim=3, start=0, count=3, instances=1,
               base_instance=0, args=None, tess=None, strides=()):
    """One render command in the RENDER layout. ctype 1 draw
    (DrawPrimitives: packed {u16 primitive, u64 start, u64
    count, u64 instances, u64 baseInstance}); args replaces the argument
    bytes for the other kinds; tess the tessellation factor bytes."""
    b = bytearray(CMD)
    struct.pack_into("<III", b, 0, ctype, barrier, pso)
    for i, bind in enumerate(vbinds):
        b[0x10 + 0x14 * i:0x10 + 0x14 * (i + 1)] = icb_bind(*bind)
    for i, bind in enumerate(fbinds):
        b[0x38 + 0x14 * i:0x38 + 0x14 * (i + 1)] = icb_bind(*bind)
    a = args if args is not None else struct.pack("<HQQQQ", prim, start, count, instances, base_instance)
    b[0x60:0x60 + len(a)] = a
    if tess is not None:
        b[0xB0:0xB0 + len(tess)] = tess
    for i, n in enumerate(strides):
        struct.pack_into("<Q", b, 0xD0 + 8 * i, n)
    return bytes(b)


def draw_indexed_args(prim=3, index_type=0, index_ref=0, count=3, index_gpu=0, index_off=0, instances=1,
                      base_vertex=0, base_instance=0):
    """DrawIndexedPrimitives, 0x38: u16 primitive, u16 indexType, u32 index
    buffer ref, u64 indexCount, u64 index buffer GPU address, u64 offset,
    u64 instances, i64 baseVertex, u64 baseInstance."""
    return struct.pack("<HHIQQQQqQ", prim, index_type, index_ref, count, index_gpu, index_off, instances,
                       base_vertex, base_instance)


def draw_patches_args(points=3, start=0, count=1, index=(0, 0, 0), instances=1, base_instance=0):
    """DrawPatches, packed 0x38: u16 control points, u64 patchStart,
    patchCount, u32 patch index ref, u64 GPU address, u64 offset, u64
    instances, u64 baseInstance."""
    return struct.pack("<HQQIQQQQ", points, start, count, index[0], index[1], index[2], instances, base_instance) + bytes(2)


def draw_indexed_patches_args(points=3, start=0, count=1, index=(0, 0, 0), cp_index=(0, 0, 0), instances=1,
                              base_instance=0):
    """DrawIndexedPatches, packed 0x4c."""
    return (struct.pack("<HQQIQQIQQQQ", points, start, count, index[0], index[1], index[2], cp_index[0], cp_index[1],
                        cp_index[2], instances, base_instance) + bytes(2))


def tess_args(ref=0, gpu=0, off=0, stride=0, scale=1.0):
    """TessellationFactor, packed 0x20: u32 ref, u64 GPU address, u64
    offset, u64 instanceStride, f32 scale."""
    return struct.pack("<IQQQf", ref, gpu, off, stride, scale)
