"""The compute decoder (encoder type 1), with the GPU's work read back out of
guest memory (docs/testing.md, sections 4.10, 6.15, 6.18).

Every scenario builds one GPU task with shared buffers over guest pages,
lists each of them in every exec's resource list (the isolate setup needs
it, section 6.12), and dumps the buffers the kernels wrote. The kernels are
tests/suite/fixtures/shaders/*.metal."""
import struct

from pgtest import abi
from pgtest.compute_objects import Blit as B, Compute as C, Rig, icb_compute_command, stage_in_blob
from pgtest.registry import scenario
from pgtest.task import seg

ADD = "shaders/compute_add.metallib"
INC = "shaders/compute_inc.metallib"


def u32s(*v):
    return struct.pack("<%dI" % len(v), *v)


@scenario
def compute_dispatch_threads():
    """SetComputePipelineState, SetBuffers and DispatchThreads run the
    double_gid kernel: out[gid] = 2 gid for 64 threads, and nothing past
    them."""
    r = Rig("compute_dispatch_threads", compute_dispatch_threads.__doc__)
    out, out_va = r.buffer()
    pso = r.kernel()
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0)]), C.dispatch_threads((64, 1, 1), (64, 1, 1)))
    r.dump(out_va, 0x110, "out")
    return r.end()


@scenario
def compute_dispatch_threadgroups():
    """DispatchThreadgroups (0xc8): 3 groups of 16 threads write 48
    values."""
    r = Rig("compute_dispatch_threadgroups", compute_dispatch_threadgroups.__doc__)
    out, out_va = r.buffer()
    pso = r.kernel()
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0)]), C.dispatch_threadgroups((3, 1, 1), (16, 1, 1)))
    r.dump(out_va, 0xd0, "out")
    return r.end()


@scenario
def compute_dispatch_indirect():
    """The two indirect dispatches read their sizes from a guest buffer:
    DispatchThreadgroupsIndirect (0xc9) takes {2, 1, 1} groups at offset
    0x10 with 8 threads each; DispatchThreadsIndirect (0xe6) takes
    {threadsPerGrid 5, 1, 1, threadsPerThreadgroup 4, 1, 1} at offset
    0x40."""
    r = Rig("compute_dispatch_indirect", compute_dispatch_indirect.__doc__)
    args = bytearray(0x80)
    args[0x10:0x1c] = u32s(2, 1, 1)
    args[0x40:0x58] = u32s(5, 1, 1, 4, 1, 1)
    arg, _ = r.buffer(contents=bytes(args))
    a, a_va = r.buffer()
    b, b_va = r.buffer()
    pso = r.kernel()
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(a, 0)]), C.dispatch_threadgroups_indirect((8, 1, 1), 0x10, arg))
    r.dump(a_va, 0x50, "threadgroups indirect")
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(b, 0)]), C.dispatch_threads_indirect(0x40, arg))
    r.dump(b_va, 0x20, "threads indirect")
    return r.end()


@scenario
def compute_buffer_offsets():
    """Buffer offsets reach the kernel: SetBuffers with offset 0x40,
    SetBufferOffset (0xcf) moving index 0 to 0x100, and the stride forms
    SetBuffers-with-stride (0xd9) and SetBufferOffset-with-stride (0xda),
    which this GPU supports (supportsDynamicAttributeStride)."""
    r = Rig("compute_buffer_offsets", compute_buffer_offsets.__doc__)
    out, out_va = r.buffer(length=0x4000)
    pso = r.kernel()
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0x40)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)),
              C.set_buffer_offset(0, 0x100), C.dispatch_threads((4, 1, 1), (4, 1, 1)),
              C.set_buffers_stride(0, [(out, 0x200, 4)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)),
              C.set_buffer_offset_stride(0, 0x300, 4), C.dispatch_threads((4, 1, 1), (4, 1, 1)))
    r.dump(out_va, 0x320, "out")
    return r.end()


@scenario
def compute_two_buffers():
    """SetBuffers binds a range: location 0 the output, location 1 the input
    at offset 8; add_one writes in[gid + 2] + 1. A second SetBuffers at
    location 1 alone rebinds only the input."""
    r = Rig("compute_two_buffers", compute_two_buffers.__doc__)
    src, _ = r.buffer(contents=u32s(*range(0x1000, 0x1010)))
    src2, _ = r.buffer(contents=u32s(*range(0x2000, 0x2010)))
    out, out_va = r.buffer()
    pso = r.kernel(ADD)
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0), (src, 8)]), C.dispatch_threads((8, 1, 1), (8, 1, 1)),
              C.set_buffer_offset(0, 0x20), C.set_buffers(1, [(src2, 0)]), C.dispatch_threads((8, 1, 1), (8, 1, 1)))
    r.dump(out_va, 0x40, "out")
    return r.end()


@scenario
def compute_order_in_exec():
    """Work in one exec runs in encoded order: a blit fills A with 0x05, a
    compute pass writes B = A + 1, a second compute pass (a new encoder)
    doubles into A, and a blit copies B after A. The final bytes show each
    step saw the one before (decision 8)."""
    r = Rig("compute_order_in_exec", compute_order_in_exec.__doc__)
    a, a_va = r.buffer()
    b, b_va = r.buffer()
    add = r.kernel(ADD)
    dbl = r.kernel()
    r.ready()
    r.run(seg(abi.ENC_BLIT, B.fill(a, 0, 0x40, 0x05)),
          seg(abi.ENC_COMPUTE, C.set_pipeline(add) + C.set_buffers(0, [(b, 0), (a, 0)])
              + C.dispatch_threads((16, 1, 1), (16, 1, 1))),
          seg(abi.ENC_COMPUTE, C.set_pipeline(dbl) + C.set_buffers(0, [(a, 0)])
              + C.dispatch_threads((8, 1, 1), (8, 1, 1))),
          seg(abi.ENC_BLIT, B.copy_buffer_to_buffer(b, a, 0, 0x40, 0x40)))
    r.dump(a_va, 0x90, "A")
    r.dump(b_va, 0x50, "B")
    return r.end()


@scenario
def compute_serial_dispatches():
    """Two dispatches in one encoder: the second reads what the first wrote.
    The pass descriptor's dispatch type is Serial by default, and so with an
    explicit SetComputePassDispatchType(0) as the first record; with
    Concurrent (1) a MemoryBarrierWithScope(buffers) between them orders
    them."""
    r = Rig("compute_serial_dispatches", compute_serial_dispatches.__doc__)
    a, a_va = r.buffer()
    b, b_va = r.buffer()
    dbl = r.kernel()
    add = r.kernel(ADD)
    r.ready()
    body = (C.set_pipeline(dbl) + C.set_buffers(0, [(a, 0)]) + C.dispatch_threads((32, 1, 1), (32, 1, 1))
            + C.set_pipeline(add) + C.set_buffers(0, [(b, 0), (a, 0)]))
    run = C.dispatch_threads((32, 1, 1), (32, 1, 1))
    for i, (prefix, mid) in enumerate(((b"", b""), (C.dispatch_type(0), b""), (C.dispatch_type(1), C.barrier_scope(1)))):
        r.run(seg(abi.ENC_BLIT, B.fill(a, 0, 0x80, 0) + B.fill(b, 0, 0x80, 0)))
        r.compute(prefix + body + mid + run)
        r.dump(b_va, 0x84, "B, pass %d" % i)
    return r.end()


@scenario
def compute_threadgroup_memory():
    """SetThreadgroupMemoryLength (0xd3) sizes threadgroup memory 0, which
    tg_reverse uses to reverse each group of 16: out[gid] = 3 * (the
    mirrored gid)."""
    r = Rig("compute_threadgroup_memory", compute_threadgroup_memory.__doc__)
    out, out_va = r.buffer()
    pso = r.kernel("shaders/compute_tgmem.metallib")
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0)]), C.threadgroup_memory(64, 0),
              C.dispatch_threadgroups((2, 1, 1), (16, 1, 1)))
    r.dump(out_va, 0x80, "out")
    return r.end()


@scenario
def compute_texture_write():
    """SetTextures (0xce) binds a 4x4 RGBA8 texture over a shared buffer
    (type 8, commandType 9); tex_write stores {x, y, 0x80, 0xff} in each
    texel, which lands in the buffer's rows of 16 bytes."""
    r = Rig("compute_texture_write", compute_texture_write.__doc__)
    buf, buf_va = r.buffer()
    tex = r.texture(buf, 4, 4)
    pso = r.kernel("shaders/compute_texwrite.metallib")
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_textures(0, [tex]), C.dispatch_threads((4, 4, 1), (4, 4, 1)))
    r.dump(buf_va, 0x50, "texture rows")
    return r.end()


@scenario
def compute_samplers():
    """SetSamplers (0xcc) and SetSamplers with LOD clamps (0xcd) bind a
    sampler whose address mode decides the texel sample_k reads left of a
    4x1 texture: ClampToEdge (0) gives texel 0 four times, Repeat (2) gives
    texels 0-3, and with the LOD-clamp form ClampToZero (4) gives zero."""
    r = Rig("compute_samplers", compute_samplers.__doc__)
    img, _ = r.buffer(contents=u32s(0xff0000ff, 0xff00ff00, 0xffff0000, 0x80808080))
    out, out_va = r.buffer()
    tex = r.texture(img, 4, 1)
    clamp = r.sampler(s_mode=0, t_mode=0)
    repeat = r.sampler(s_mode=2, t_mode=2)
    zero = r.sampler(s_mode=4, t_mode=4)
    pso = r.kernel("shaders/compute_sample.metallib")
    r.ready()
    common = C.set_pipeline(pso) + C.set_textures(0, [tex])
    r.compute(common, C.set_samplers(0, [clamp]), C.set_buffers(0, [(out, 0)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)),
              C.set_samplers(0, [repeat]), C.set_buffers(0, [(out, 0x10)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)),
              C.set_samplers_lod(0, [(zero, 0.0, 4.0)]), C.set_buffers(0, [(out, 0x20)]),
              C.dispatch_threads((4, 1, 1), (4, 1, 1)))
    r.dump(out_va, 0x30, "samples: clamp, repeat, zero")
    return r.end()


@scenario
def compute_stage_in():
    """A pipeline with a stage-input descriptor (attribute 0 from buffer 1,
    stride 4, per thread in x) and SetStageInRegion (0xd1): stage_in_k
    writes in.v + 0x100. Then SetStageInRegionIndirect (0xd2) takes the
    region from a buffer at offset 8."""
    r = Rig("compute_stage_in", compute_stage_in.__doc__)
    src, _ = r.buffer(contents=u32s(*range(0x10, 0x30)))
    args, _ = r.buffer(contents=bytes(8) + u32s(0, 0, 0, 6, 1, 1))
    out, out_va = r.buffer()
    f = r.function("shaders/compute_stagein.metallib")
    pso = r.pipeline(blob=stage_in_blob(f))
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0), (src, 0)]), C.stage_in_region((8, 1, 1), (0, 0, 0)),
              C.dispatch_threads((8, 1, 1), (8, 1, 1)),
              C.set_buffer_offset(0, 0x40), C.stage_in_region_indirect(args, 8),
              C.dispatch_threads((8, 1, 1), (8, 1, 1)))
    r.dump(out_va, 0x60, "out")
    return r.end()


@scenario
def compute_fences():
    """Fences (type 7, commandType 13) order work across encoders: a blit
    fills A and updates fence 1, a compute pass waits for it, writes
    B = A + 1 and updates fence 2, and a blit waits for fence 2 and copies B
    into C."""
    r = Rig("compute_fences", compute_fences.__doc__)
    a, _ = r.buffer()
    b, _ = r.buffer()
    c, c_va = r.buffer()
    f1 = r.fence()
    f2 = r.fence()
    add = r.kernel(ADD)
    r.ready()
    r.run(seg(abi.ENC_BLIT, B.fill(a, 0, 0x40, 0x21) + B.fence(316, f1)),
          seg(abi.ENC_COMPUTE, C.fence(0xd5, f1) + C.set_pipeline(add) + C.set_buffers(0, [(b, 0), (a, 0)])
              + C.dispatch_threads((16, 1, 1), (16, 1, 1)) + C.fence(0xd4, f2)),
          seg(abi.ENC_BLIT, B.fence(317, f2) + B.copy_buffer_to_buffer(b, c, 0, 0, 0x40)))
    r.dump(c_va, 0x50, "C")
    return r.end()


@scenario
def compute_resident_and_barriers():
    """The residency and barrier records succeed and leave the dispatch's
    result as it was: UseResources (0x87), UseHeaps (0x86) on a private
    heap, MemoryBarrierWithResources (0xd6), MemoryBarrierWithScope (0xd7),
    SetImageblockWidth (0xd8, this GPU supportsImageBlocks) and
    InsertCompressedTextureReinterpretationFlush (0xe3). The lists of 300
    take the heap-allocated path (count > 256). The heap is in the exec's
    resource list (compute_err_unlisted_heap has it unlisted)."""
    r = Rig("compute_resident_and_barriers", compute_resident_and_barriers.__doc__)
    out, out_va = r.buffer()
    heap = r.heap(listed=True)
    pso = r.kernel()
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0)]),
              C.use_resources([out]), C.use_resources([out] * 300, usage=1), C.use_resources([]),
              C.use_heaps([heap]), C.use_heaps([heap] * 300), C.use_heaps([]),
              C.barrier_resources([out]), C.barrier_resources([out] * 300), C.barrier_resources([]),
              C.barrier_scope(1), C.barrier_scope(2), C.imageblock(8, 8), C.flush(),
              C.dispatch_threads((8, 1, 1), (8, 1, 1)))
    r.dump(out_va, 0x24, "out")
    return r.end()


@scenario
def compute_empty_segments():
    """A compute segment with no records opens and ends an encoder and
    succeeds; so does one whose only record is SetComputePassDispatchType,
    and one with 1-7 trailing bytes after a record."""
    r = Rig("compute_empty_segments", compute_empty_segments.__doc__)
    out, out_va = r.buffer()
    pso = r.kernel()
    r.ready()
    r.compute()
    r.compute(C.dispatch_type(0))
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)), b"\x01\x02\x03")
    r.dump(out_va, 0x14, "out")
    return r.end()


@scenario
def compute_continuation():
    """A compute pass split over segments: the first sets the pipeline and
    says continuedByNext, the second continues it and dispatches with the
    pipeline still set; a third segment continues again and ends the
    encoder. The pipeline is released at the end, so a fresh segment that
    dispatches without setting one faults."""
    r = Rig("compute_continuation", compute_continuation.__doc__)
    out, out_va = r.buffer()
    pso = r.kernel()
    r.ready()
    r.run(seg(abi.ENC_COMPUTE, C.set_pipeline(pso) + C.set_buffers(0, [(out, 0)]), cont_next=1),
          seg(abi.ENC_COMPUTE, C.dispatch_threads((4, 1, 1), (4, 1, 1)), cont_prev=1, cont_next=1),
          seg(abi.ENC_COMPUTE, C.set_buffer_offset(0, 0x20) + C.dispatch_threads((4, 1, 1), (4, 1, 1)), cont_prev=1))
    r.dump(out_va, 0x34, "out")
    r.run(seg(abi.ENC_COMPUTE, C.set_pipeline(pso) + C.set_buffers(0, [(out, 0x40)])),
          seg(abi.ENC_COMPUTE, C.set_buffers(0, [(out, 0x40)]) + C.dispatch_threads((4, 1, 1), (4, 1, 1))),
          fault=True, label="a new pass has no pipeline")
    r.dump(out_va, 0x54, "out, unchanged")
    return r.end()


@scenario
def compute_conditionals():
    """Command-buffer jumps, which this GPU supports
    (supportsCommandBufferJump): EncodeStartIf (0xe0) on the guest word 7 with
    a compare function and a reference value, a dispatch, EncodeStartElse
    (0xe1), another dispatch into the next 0x20 bytes, EncodeEndIf (0xe2).
    Three passes: Equal (2) 7, Equal 8 and NotEqual (5) 7; which branch each
    ran is what the GPU did, locked as it is."""
    r = Rig("compute_conditionals", compute_conditionals.__doc__)
    cond, _ = r.buffer(contents=u32s(7, 9))
    a, a_va = r.buffer()
    pso = r.kernel()
    r.ready()
    for i, (cmp_, val) in enumerate(((2, 7), (2, 8), (5, 7))):
        r.compute(C.set_pipeline(pso),
                  C.jump(0xe0, cond, 0, cmp_, val),
                  C.set_buffers(0, [(a, 0x40 * i)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)),
                  C.jump(0xe1),
                  C.set_buffers(0, [(a, 0x40 * i + 0x20)]), C.dispatch_threads((2, 1, 1), (2, 1, 1)),
                  C.jump(0xe2))
    r.dump(a_va, 0xc0, "if (7 == 7), if (7 == 8), if (7 != 7)")
    return r.end()


@scenario
def compute_loops():
    """Command-buffer loops: EncodeStartDoWhile (0xdc) ... EncodeEndDoWhile
    (0xdd, while c[0] < 5) runs inc_k five times; EncodeStartWhile (0xde,
    while c[0] < 3) ... EncodeEndWhile (0xdf) runs it three times, and not
    at all when the guest word starts at 3. MTLCompareFunctionLess is 1."""
    r = Rig("compute_loops", compute_loops.__doc__)
    c1, c1_va = r.buffer()
    c2, c2_va = r.buffer()
    c3, c3_va = r.buffer(contents=u32s(3))
    pso = r.kernel(INC)
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(c1, 0)]), C.jump(0xdc),
              C.dispatch_threads((1, 1, 1), (1, 1, 1)), C.barrier_scope(1), C.jump(0xdd, c1, 0, 1, 5))
    r.dump(c1_va, 0x40, "do-while")
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(c2, 0)]), C.jump(0xde, c2, 0, 1, 3),
              C.dispatch_threads((1, 1, 1), (1, 1, 1)), C.barrier_scope(1), C.jump(0xdf))
    r.dump(c2_va, 0x40, "while")
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(c3, 0)]), C.jump(0xde, c3, 0, 1, 3),
              C.dispatch_threads((1, 1, 1), (1, 1, 1)), C.jump(0xdf))
    r.dump(c3_va, 0x40, "while, never")
    return r.end()


@scenario
def compute_indirect_command_buffer():
    """ExecuteCommandsInBuffer runs an indirect command buffer the guest
    wrote (listed in the exec, so prepared from guest memory): command 0
    dispatches double_gid (1 group of 8) into A, command 1 add_one reading A
    into B, command 2 is empty. The ICB's buffers reach its commands only
    once UseResources makes them resident: 0xe4 over [0, 2) without it
    leaves A and B alone, with it writes both. 0xe5 then takes the range
    {u32 location, u32 length} = {1, 1} from a buffer at offset 0x10, so
    only the second command runs again, into a B the guest refilled."""
    r = Rig("compute_indirect_command_buffer", compute_indirect_command_buffer.__doc__)
    a, a_va = r.buffer()
    b, b_va = r.buffer()
    rng, _ = r.buffer(contents=bytes(0x10) + u32s(1, 1))
    dbl = r.kernel(icb=True)
    add = r.kernel(ADD, icb=True)
    cmds = [icb_compute_command(pipeline=dbl, buffers=[(a, 0)], grid=(1, 1, 1), tptg=(8, 1, 1)),
            icb_compute_command(pipeline=add, buffers=[(b, 0), (a, 0)], grid=(1, 1, 1), tptg=(8, 1, 1), barrier=1),
            icb_compute_command(ctype=0)]
    icb = r.icb(cmds, kernel_buffers=2)
    r.ready()
    r.compute(C.execute_icb(icb, 0, 2))
    r.dump(a_va, 0x24, "A, not resident")
    r.dump(b_va, 0x24, "B, not resident")
    r.compute(C.use_resources([a, b]), C.execute_icb(icb, 0, 2))
    r.dump(a_va, 0x24, "A")
    r.dump(b_va, 0x24, "B")
    r.run(seg(abi.ENC_BLIT, B.fill(b, 0, 0x20, 0xee) + B.fill(a, 0, 4, 0x40)))
    r.compute(C.use_resources([a, b, rng]), C.execute_icb_indirect(icb, rng, 0x10))
    r.dump(b_va, 0x24, "B after the indirect range")
    return r.end()


@scenario
def compute_unbalanced_jumps():
    """Command-buffer jump records out of balance: EncodeEndIf, EncodeEndWhile,
    EncodeEndDoWhile and EncodeStartElse with no start, EncodeStartIf and
    EncodeStartWhile on buffer ref 0 (the nullable getter's nil) closed
    properly, and EncodeStartDoWhile never closed. Each pass sets a pipeline
    and dispatches double_gid after the record; the decoder accepts every
    record (an End* that returns NO only logs), the exec completes, and what
    the GPU wrote is locked per pass."""
    r = Rig("compute_unbalanced_jumps", compute_unbalanced_jumps.__doc__)
    out, out_va = r.buffer()
    pso = r.kernel()
    r.ready()
    passes = [("EndIf", [C.jump(0xe2)]), ("EndWhile", [C.jump(0xdf)]),
              ("EndDoWhile", [C.jump(0xdd, out, 0x3f00, 1, 5)]), ("StartElse", [C.jump(0xe1)]),
              ("StartIf nil", [C.jump(0xe0, 0, 0, 2, 0), C.jump(0xe2)]),
              ("StartWhile nil", [C.jump(0xde, 0, 0, 2, 1), C.jump(0xdf)]),
              ("StartDoWhile, never closed", [C.jump(0xdc)])]
    for i, (label, recs) in enumerate(passes):
        r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0x20 * i)]), *recs,
                  C.dispatch_threads((4, 1, 1), (4, 1, 1)), label=label)
    r.dump(out_va, 0x20 * len(passes), "out, 0x20 a pass")
    return r.end()


@scenario
def compute_delete_and_recreate():
    """Objects are created on first use and cached, and deletion (opcode 40)
    drops the cache entry. The guest rewrites pipeline P's record to name
    add_one's function instead of double_gid's: P still runs double_gid.
    After DeleteObject 1006 (compute pipeline) P is created again from the
    new record and runs add_one. Then the guest names double_gid's function
    again in P, but points that function's record at add_one's metallib, and
    deletes the function (1005) and P (1006): add_one runs. A fence deleted
    (1009) and used again is created again. DeleteObject's payload is {u32
    taskID, u32 operation, u32 length, u32 ref}."""
    from pgtest.compute_objects import compute_blob, compute_pipeline_record, function_record
    r = Rig("compute_delete_and_recreate", compute_delete_and_recreate.__doc__)
    src, _ = r.buffer(contents=u32s(*range(0x40, 0x50)))
    out, out_va = r.buffer()
    t, s = r.t, r.s
    libs = {}
    for name, fixture in (("dbl", "shaders/kernel.metallib"), ("add", ADD)):
        va = t.reserve(0x2000)
        libs[name] = (va, s.mem_file(t.gpa(va), fixture))
    dbl_fn = t.ol.add(abi.T_FUNCTION, function_record(*libs["dbl"]))
    add_fn = t.ol.add(abi.T_FUNCTION, function_record(*libs["add"]))
    pso = r.pipeline(dbl_fn)
    fence = r.fence()
    va_of = {ref: va for ref, (_, _, va) in enumerate(t.ol.entries)}
    r.ready()

    def dispatch(off):
        r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, off), (src, 0)]), C.fence(0xd4, fence),
                  C.dispatch_threads((4, 1, 1), (4, 1, 1)))

    def delete(op, ref):
        s.submit(abi.OP_DELETE_OBJECT, struct.pack("<IIII", t.tid, op, 8, ref), label="delete %d ref %d" % (op, ref))

    dispatch(0)
    t.write(va_of[pso], compute_pipeline_record(compute_blob(add_fn)))
    dispatch(0x10)
    delete(1006, pso)
    dispatch(0x20)
    t.write(va_of[pso], compute_pipeline_record(compute_blob(dbl_fn)))
    t.write(va_of[dbl_fn], function_record(*libs["add"]))
    delete(1005, dbl_fn)
    delete(1006, pso)
    delete(1009, fence)
    dispatch(0x30)
    r.dump(out_va, 0x40, "out: double_gid, still cached, add_one, add_one through double_gid's function")
    return r.end()
