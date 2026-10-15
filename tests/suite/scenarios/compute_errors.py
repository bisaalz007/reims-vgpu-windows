"""The compute decoder's checks (docs/testing.md, sections 6.12, 6.13, 6.15).

A decode error faults the exec and the command buffer is never committed
(section 6.12), so each faulting exec below also carries a dispatch ahead of
the bad record that would have written the output buffer: the dump after
the faults shows nothing ran. Every fault is followed by a resume (0x1024)."""
import struct

from pgtest import abi
from pgtest.compute_objects import Blit as B, Compute as C, Rig, compute_blob, compute_pipeline_record
from pgtest.registry import scenario
from pgtest.task import rec, seg


def u32s(*v):
    return struct.pack("<%dI" % len(v), *v)


class Checks:
    """A rig with an output buffer, a double_gid pipeline, a texture, a
    sampler, a fence, a heap and an ICB, for faulting records to name."""

    def __init__(self, name, doc, extra=None, **kw):
        r = self.r = Rig(name, doc, **kw)
        self.out, self.out_va = r.buffer()
        self.img, _ = r.buffer()
        self.tex = r.texture(self.img, 4, 1)
        self.pso = r.kernel()
        self.fn = self.pso - 1
        self.sampler = r.sampler()
        self.fence = r.fence()
        self.heap = r.heap(listed=True)
        self.icb = r.icb([bytes(0x100)] * 2)
        self.invalid = r.raw(0, b"")  # an entry of type 0
        self.extra = [] if extra is None else extra(r)
        if not isinstance(self.extra, list):
            self.extra = [self.extra]
        self.past = r.t.ol.length() // 12  # past the end of the object list
        r.ready()

    def prefix(self):
        return C.set_pipeline(self.pso) + C.set_buffers(0, [(self.out, 0)]) + C.dispatch_threads((4, 1, 1), (4, 1, 1))

    def fault(self, label, *records, prefix=True):
        self.r.compute((self.prefix() if prefix else b"") + b"".join(records), fault=True, label=label)

    def end(self):
        self.r.dump(self.out_va, 0x14, "out: no faulting exec ran")
        return self.r.end()


@scenario
def compute_err_dispatch_without_pipeline():
    """Each of the four dispatches throws "Attempting to dispatch without
    pipeline set" when no SetComputePipelineState came first in the
    pass."""
    k = Checks("compute_err_dispatch_without_pipeline", compute_err_dispatch_without_pipeline.__doc__)
    arg = k.out
    k.fault("0xc8", C.set_buffers(0, [(k.out, 0)]), C.dispatch_threadgroups((1, 1, 1), (4, 1, 1)), prefix=False)
    k.fault("0xc9", C.set_buffers(0, [(k.out, 0)]), C.dispatch_threadgroups_indirect((4, 1, 1), 0, arg), prefix=False)
    k.fault("0xca", C.set_buffers(0, [(k.out, 0)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)), prefix=False)
    k.fault("0xe6", C.set_buffers(0, [(k.out, 0)]), C.dispatch_threads_indirect(0, arg), prefix=False)
    return k.end()


@scenario
def compute_err_pipeline_refs():
    """SetComputePipelineState faults for ref 0 ("Invalid compute pipeline
    state"), and for refs that do not make a compute pipeline: a buffer
    (type 1), a fence and a sampler (type 7, commandTypes 13 and 3), a
    function (type 6), an entry of type 0 ("Attempting to fetch invalid
    object") and a ref past the list."""
    k = Checks("compute_err_pipeline_refs", compute_err_pipeline_refs.__doc__)
    for label, ref in (("ref 0", 0), ("a buffer", k.out), ("a fence", k.fence), ("a sampler", k.sampler),
                       ("a function", k.fn), ("type 0", k.invalid), ("past the list", k.past)):
        k.fault(label, C.set_pipeline(ref))
    return k.end()


@scenario
def compute_err_buffer_refs():
    """SetBuffers and its stride form fault for a texture (a resource with no
    buffer), and so do the indirect dispatches' and the jump records' buffer
    refs; SetBuffers faults for a pipeline ("Unexpected object type (7) for
    resource"), an entry of type 0, a ref past the list, and a count larger
    than the entries that follow ("Failed to read"). Ref 0 binds nil and
    succeeds."""
    k = Checks("compute_err_buffer_refs", compute_err_buffer_refs.__doc__)
    for label, ref in (("a texture", k.tex), ("a pipeline", k.pso), ("type 0", k.invalid),
                       ("past the list", k.past)):
        k.fault("0xcb " + label, C.set_buffers(1, [(ref, 0)]))
        k.fault("0xd9 " + label, C.set_buffers_stride(1, [(ref, 0, 4)]))
    k.fault("0xc9 indirect buffer a texture", C.dispatch_threadgroups_indirect((4, 1, 1), 0, k.tex))
    k.fault("0xe6 indirect buffer a texture", C.dispatch_threads_indirect(0, k.tex))
    k.fault("0xdd buffer a texture", C.jump(0xdc), C.jump(0xdd, k.tex, 0, 1, 1))
    k.fault("0xde buffer a texture", C.jump(0xde, k.tex, 0, 1, 1))
    k.fault("0xe0 buffer a texture", C.jump(0xe0, k.tex, 0, 2, 0))
    k.fault("0xcb short", C.set_buffers(1, [(k.out, 0)], count=2))
    k.fault("0xd9 short", C.set_buffers_stride(1, [(k.out, 0, 4)], count=2))
    k.r.compute(C.set_buffers(1, [(0, 0)]), C.set_buffers_stride(2, [(0, 0, 0)]))
    return k.end()


@scenario
def compute_err_texture_sampler_refs():
    """SetTextures faults for a buffer ref (a resource with no texture), a
    pipeline and a ref past the list; SetSamplers and its LOD-clamp form
    fault for a buffer ("Invalid object type (1) for sampler state"), a
    fence (commandType 13), and a short list. Ref 0 binds nil and
    succeeds."""
    k = Checks("compute_err_texture_sampler_refs", compute_err_texture_sampler_refs.__doc__)
    for label, ref in (("a buffer", k.out), ("a pipeline", k.pso), ("past the list", k.past)):
        k.fault("0xce " + label, C.set_textures(0, [ref]))
    k.fault("0xce short", C.set_textures(0, [k.tex], count=2))
    for label, ref in (("a buffer", k.out), ("a fence", k.fence), ("past the list", k.past)):
        k.fault("0xcc " + label, C.set_samplers(0, [ref]))
        k.fault("0xcd " + label, C.set_samplers_lod(0, [(ref, 0.0, 1.0)]))
    k.fault("0xcc short", C.set_samplers(0, [k.sampler], count=2))
    k.fault("0xcd short", C.set_samplers_lod(0, [(k.sampler, 0.0, 1.0)], count=2))
    k.r.compute(C.set_textures(0, [0]), C.set_samplers(0, [0]), C.set_samplers_lod(1, [(0, 0.0, 1.0)]))
    return k.end()


@scenario
def compute_err_counts():
    """Binding ranges past any GPU's limits fault before a single entry is
    read: SetBuffers and its stride form ("Invalid buffer count"),
    SetSamplers and the LOD form ("Invalid sampler count"), SetTextures
    ("Invalid texture count"), SetBufferOffset and its stride form
    ("Invalid buffer index"), SetThreadgroupMemoryLength ("Invalid
    threadgroup memory size index"), each at location or index 0x10000."""
    k = Checks("compute_err_counts", compute_err_counts.__doc__)
    big = 0x10000
    k.fault("0xcb", C.set_buffers(big, [], count=1))
    k.fault("0xd9", C.set_buffers_stride(big, [], count=1))
    k.fault("0xcc", C.set_samplers(big, [], count=1))
    k.fault("0xcd", C.set_samplers_lod(big, [], count=1))
    k.fault("0xce", C.set_textures(big, [], count=1))
    k.fault("0xcf", C.set_buffer_offset(big, 0))
    k.fault("0xda", C.set_buffer_offset_stride(big, 0, 0))
    k.fault("0xd3", C.threadgroup_memory(16, big))
    return k.end()


@scenario
def compute_err_resource_lists():
    """UseResources, UseHeaps and MemoryBarrierWithResources throw "Too many
    resources specified" when the count needs more bytes than the record
    has, whether under or over the 256 the stack holds. UseHeaps faults for
    a ref that is not a heap (a buffer, a ref past the list); UseResources
    and the barrier look refs up with the nullable getter, so a pipeline or
    an unknown ref is nil and the pass runs."""
    k = Checks("compute_err_resource_lists", compute_err_resource_lists.__doc__)
    k.fault("0x87 2 of 1", C.use_resources([k.out], count=2))
    k.fault("0x87 300 of 1", C.use_resources([k.out], count=300))
    k.fault("0x86 2 of 1", C.use_heaps([k.heap], count=2))
    k.fault("0x86 300 of 1", C.use_heaps([k.heap], count=300))
    k.fault("0xd6 2 of 1", C.barrier_resources([k.out], count=2))
    k.fault("0xd6 300 of 1", C.barrier_resources([k.out], count=300))
    k.fault("0x86 a buffer", C.use_heaps([k.out]))
    k.fault("0x86 past the list", C.use_heaps([k.past]))
    k.fault("0x87 short header", rec(0x87, u32s(0)))
    k.r.compute(C.use_resources([k.pso, k.past, 0]), C.barrier_resources([k.pso, k.past, 0]), C.use_heaps([0]))
    return k.end()


@scenario
def compute_err_stage_in_indirect():
    """SetStageInRegionIndirect throws "Invalid indirect buffer" for ref 0
    and "Invalid indirect buffer offset" for an offset not 4-aligned; a
    texture ref faults in the lookup."""
    k = Checks("compute_err_stage_in_indirect", compute_err_stage_in_indirect.__doc__)
    k.fault("ref 0", C.stage_in_region_indirect(0, 0))
    k.fault("offset 2", C.stage_in_region_indirect(k.out, 2))
    k.fault("a texture", C.stage_in_region_indirect(k.tex, 0))
    return k.end()


@scenario
def compute_err_fences():
    """UpdateFence and WaitForFence need a fence: ref 0 throws through the
    non-null getter, a buffer is "Invalid object type (1) for fence", a
    sampler "Invalid serializer command type (3)", and a ref past the list
    fails the lookup."""
    k = Checks("compute_err_fences", compute_err_fences.__doc__,
               extra=lambda r: r.raw(abi.T_SERIALIZED, struct.pack("<I", 13)))
    for label, ref in (("ref 0", 0), ("a buffer", k.out), ("a sampler", k.sampler), ("past the list", k.past),
                       ("a 4-byte record", k.extra[0])):
        k.fault("0xd4 " + label, C.fence(0xd4, ref))
        k.fault("0xd5 " + label, C.fence(0xd5, ref))
    return k.end()


@scenario
def compute_err_icb():
    """ExecuteCommandsInBuffer (ranged, 0xe4) faults for ICB ref 0 ("Failed
    to get indirect command buffer"), a buffer ref, a range past the ICB's 2
    commands, and a range whose end overflows; the indirect form (0xe5) for
    ICB ref 0, range buffer ref 0 ("Failed to get indirect range buffer",
    which names the ICB ref), and a range buffer that is a texture."""
    k = Checks("compute_err_icb", compute_err_icb.__doc__)
    k.fault("0xe4 ref 0", C.execute_icb(0, 0, 1))
    k.fault("0xe4 a buffer", C.execute_icb(k.out, 0, 1))
    k.fault("0xe4 [1, 3)", C.execute_icb(k.icb, 1, 2))
    k.fault("0xe4 overflow", C.execute_icb(k.icb, 1, 0xFFFFFFFFFFFFFFFF))
    k.fault("0xe5 ref 0", C.execute_icb_indirect(0, k.out, 0))
    k.fault("0xe5 range ref 0", C.execute_icb_indirect(k.icb, 0, 0))
    k.fault("0xe5 range a texture", C.execute_icb_indirect(k.icb, k.tex, 0))
    k.r.compute(C.execute_icb(k.icb, 2, 0), C.execute_icb(k.icb, 0, 2))
    return k.end()


@scenario
def compute_err_framing():
    """How the compute decoder reads its records: an unknown command
    (0x100), SetComputePassDispatchType anywhere but first, a record length
    under 8 ("Underflow"), a length past the segment ("Failed to read"), a
    record shorter than its command (DispatchThreads with 8 bytes), and the
    first record's peek: a length under 8, a length past the segment, a
    short 0xdb, and a segment of 4 bytes that cannot hold a header."""
    k = Checks("compute_err_framing", compute_err_framing.__doc__)
    k.fault("unknown 0x100", rec(0x100, u32s(0)))
    k.fault("0xdb second", C.dispatch_type(0))
    k.fault("length 4", rec(0xca, length=4))
    k.fault("length past the segment", rec(0xca, bytes(0x30), length=0x100))
    k.fault("short 0xca", rec(0xca, bytes(8)))
    k.fault("first: length 4", rec(0xdb, u32s(0), length=4), prefix=False)
    k.fault("first: length past the segment", rec(0xdb, u32s(0), length=0x100), prefix=False)
    k.fault("first: short 0xdb", rec(0xdb), prefix=False)
    k.fault("first: 4 bytes", b"\x01\x02\x03\x04", prefix=False)
    k.fault("short 0xdd", rec(0xdd, u32s(k.out)))
    k.fault("short 0xde", rec(0xde, u32s(k.out)))
    k.fault("short 0xe0", rec(0xe0, u32s(k.out)))
    return k.end()


@scenario(setups=("isolate",))
def compute_err_unlisted_heap():
    """In the isolate setup, UseHeaps on a heap the exec does not list
    makes the command buffer fail (section 6.12's "Insufficient Memory"),
    and the exec faults; listed, the same pass writes its output. Isolate
    only: in-process heaps are host memory, and there it runs (see
    compute_resident_and_barriers)."""
    r = Rig("compute_err_unlisted_heap", compute_err_unlisted_heap.__doc__, setups=("isolate",))
    out, out_va = r.buffer()
    heap = r.heap()
    pso = r.kernel()
    r.ready()
    body = (C.set_pipeline(pso), C.set_buffers(0, [(out, 0)]), C.use_heaps([heap]),
            C.dispatch_threads((4, 1, 1), (4, 1, 1)))
    r.compute(*body, fault=True, label="unlisted")
    r.dump(out_va, 0x14, "out, unlisted")
    r.compute(*body, resources=r.resources([(heap, 0)]))
    r.dump(out_va, 0x14, "out, listed")
    return r.end()


@scenario
def compute_err_pipeline_records():
    """Compute pipeline records that fail without a death: an empty
    descriptor (Metal returns a nil descriptor, and building a pipeline
    from it fails: "Failed to create compute pipeline"), a vertex function
    as the compute function (the same error), a record of the header alone
    ("Failed to read command"), a descriptorLength past the record
    ("Insufficient data in command"), and a type-7 record of 4 bytes
    ("Failed to read serializer header")."""
    def extra(r):
        fn = r.function()
        vert = r.function("shaders/vert.metallib")
        return [r.pipeline(blob=b""), r.pipeline(vert),
                r.raw(abi.T_SERIALIZED, struct.pack("<II", 11, 0)),
                r.raw(abi.T_SERIALIZED, compute_pipeline_record(compute_blob(fn), descriptor_length=20)),
                r.raw(abi.T_SERIALIZED, struct.pack("<I", 11))]
    k = Checks("compute_err_pipeline_records", compute_err_pipeline_records.__doc__, extra=extra)
    for label, ref in zip(("empty descriptor", "a vertex function", "header only", "descriptorLength 20 of 7",
                           "4 bytes"), k.extra):
        k.fault(label, C.set_pipeline(ref))
    return k.end()


@scenario
def compute_err_protected_encoder():
    """A protection envelope (encoder type 5) around a compute pass, with
    options 2 and with all bits set: the command buffer takes the options
    (the SPI setProtectionOptions:), the compute decoder still gets an
    encoder, and the exec faults with nothing written. Which protection
    options these are is Metal's; options 1 and 3 are deaths
    (compute_death_protection_options)."""
    r = Rig("compute_err_protected_encoder", compute_err_protected_encoder.__doc__)
    out, out_va = r.buffer()
    pso = r.kernel()
    r.ready()
    body = C.set_pipeline(pso) + C.set_buffers(0, [(out, 0)]) + C.dispatch_threads((4, 1, 1), (4, 1, 1))
    inner = abi.SEG_HDR.pack(8 + len(body), abi.ENC_COMPUTE, 0, 0) + body
    for opt in (2, 0xFFFFFFFFFFFFFFFF):
        r.run(abi.SEG_HDR.pack(16 + len(inner), abi.ENC_PROTECTION, 0, 0) + struct.pack("<Q", opt) + inner,
              fault=True, label="options %#x" % opt)
    r.dump(out_va, 0x14, "out")
    return r.end()


@scenario(host=True)
def compute_err_limits():
    """Each binding check at this GPU's limit (maxComputeBuffers 31,
    maxComputeSamplers 16, maxComputeTextures 128,
    maxComputeLocalMemorySizes 31): a range ending exactly at the limit
    binds (NSMaxRange > max fails), one ending past it faults; an index one
    below the limit binds and the limit itself faults (index >= max)."""
    k = Checks("compute_err_limits", compute_err_limits.__doc__, host=True)
    ok = [C.set_buffers(30, [(0, 0)]), C.set_buffers_stride(30, [(0, 0, 0)]), C.set_samplers(15, [0]),
          C.set_samplers_lod(15, [(0, 0.0, 1.0)]), C.set_textures(127, [0]), C.set_buffer_offset(30, 0),
          C.set_buffer_offset_stride(30, 0, 0), C.threadgroup_memory(16, 30)]
    k.r.compute(k.prefix(), *ok)
    k.r.dump(k.out_va, 0x14, "out: the pass at the limits ran")
    k.r.run(seg(abi.ENC_BLIT, B.fill(k.out, 0, 0x14, 0)))
    k.fault("0xcb [31, 32)", C.set_buffers(31, [(0, 0)]))
    k.fault("0xd9 [31, 32)", C.set_buffers_stride(31, [(0, 0, 0)]))
    k.fault("0xcc [16, 17)", C.set_samplers(16, [0]))
    k.fault("0xcd [16, 17)", C.set_samplers_lod(16, [(0, 0.0, 1.0)]))
    k.fault("0xce [128, 129)", C.set_textures(128, [0]))
    k.fault("0xcf index 31", C.set_buffer_offset(31, 0))
    k.fault("0xda index 31", C.set_buffer_offset_stride(31, 0, 0))
    k.fault("0xd3 index 31", C.threadgroup_memory(16, 31))
    return k.end()
