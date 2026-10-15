"""Guest-triggerable deaths through the compute path (docs/testing.md,
sections 4.8, 6.18, 7.4).

Each scenario ends in one inside Metal (Metal aborts, or dereferences
nil), so the host process dies at the exec in every setup: an isolated
task's commands run on threads of the host's process, and a crash inside
Metal there ends the process too (section 4.8). The second exec, which
binds a buffer on the same task, is never reached. Only the outcome, the
process and the op are locked, not the signal (decision 4).

Functions are looked up only through Metal's descriptor deserialization,
whose function resolver (resources::manager::resolver's
function_with_function_ref) gives Metal nil for any lookup that fails;
Metal then aborts building a compute pipeline with no function. So every
function error below is a death, and each reaches its own check."""
import struct

from pgtest import abi
from pgtest.compute_objects import Compute as C, Rig, compute_blob, function_record
from pgtest.registry import scenario


def death(name, doc, build):
    """build(rig) returns the pipeline ref to set. The exec sets it and
    dispatches, and the host dies there; the second exec only binds a
    buffer, and is never reached."""
    r = Rig(name, doc)
    out, out_va = r.buffer()
    pso = build(r, out)
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)),
              fault=True, label="the death")
    r.compute(C.set_buffers(0, [(out, 0)]), fault=True, label="the task's service is gone")
    r.s.events("end")
    return r.s


def fn_death(suffix, what, build):
    name = "compute_death_function_" + suffix

    def f():
        return death(name, "A compute pipeline whose function ref resolves to nil: %s. "
                     "Metal aborts (section 6.18)." % what, build)
    f.__name__ = name
    f.__doc__ = what
    scenario(f)


fn_death("ref0", "the descriptor names function ref 0, which is nil by definition",
         lambda r, o: r.pipeline(0))
fn_death("wrong_type", "the ref names a buffer: \"Invalid object type (1) for function\"",
         lambda r, o: r.pipeline(o))
fn_death("short_record", "the function record is 8 bytes: \"Failed to read function header\"",
         lambda r, o: r.pipeline(r.raw(abi.T_FUNCTION, struct.pack("<Q", 0x20000))))
fn_death("short_header", "the record is 16 bytes, so its PGOperationHeader is cut: "
         "\"Failed to read serializer header\"",
         lambda r, o: r.pipeline(r.raw(abi.T_FUNCTION, struct.pack("<QII", 0x20000, 100, 15))))
fn_death("data_out_of_range", "the metallib's VA lies past the task: \"Failed to get function "
         "data\"",
         lambda r, o: r.pipeline(r.raw(abi.T_FUNCTION, function_record(0x4000000, 100))))
fn_death("command_type", "commandType 14 instead of 15: \"Invalid serializer command type\"",
         lambda r, o: r.pipeline(r.function(command_type=14)))
fn_death("truncated_library", "the first 1000 bytes of kernel.metallib: Metal's \"Invalid library "
         "file\", \"Failed to create library\"",
         lambda r, o: r.pipeline(r.function("shaders/kernel_trunc.metallib")))
fn_death("empty_library", "a function of length 0: an empty library, \"Failed to create library\"",
         lambda r, o: r.pipeline(r.function(length=0)))
fn_death("two_functions", "two.metallib holds two kernels: \"Unexpected function count in "
         "library (2)\"",
         lambda r, o: r.pipeline(r.function("shaders/two.metallib")))


@scenario
def compute_death_function_constant():
    """fc.metallib's kernel has an unspecialized [[function_constant]]: the
    function is created, and building the pipeline aborts inside Metal
    (section 6.18)."""
    return death("compute_death_function_constant", compute_death_function_constant.__doc__,
                 lambda r, o: r.pipeline(r.function("shaders/fc.metallib")))


def blob_death(suffix, what, blob_of):
    name = "compute_death_blob_" + suffix

    def f():
        def build(r, o):
            fn = r.function()
            return r.pipeline(blob=blob_of(fn))
        return death(name, "A malformed serialized compute descriptor: %s. Metal aborts deserializing "
                     "it (section 6.18)." % what, build)
    f.__name__ = name
    scenario(f)


blob_death("garbage", "64 bytes of 0xab", lambda fn: b"\xab" * 64)
blob_death("truncated1", "the valid 7-byte blob cut to 1 byte", lambda fn: compute_blob(fn)[:1])
blob_death("truncated5", "the valid 7-byte blob cut to 5 bytes", lambda fn: compute_blob(fn)[:5])


@scenario
def compute_death_threadgroups_indirect_nil():
    """DispatchThreadgroupsIndirect with indirect buffer ref 0: the nullable
    getter gives nil and the decoder hands it to Metal, which dereferences
    it."""
    r = Rig("compute_death_threadgroups_indirect_nil", compute_death_threadgroups_indirect_nil.__doc__)
    out, _ = r.buffer()
    pso = r.kernel()
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0)]), C.dispatch_threadgroups_indirect((4, 1, 1), 0, 0),
              fault=True, label="the death")
    r.compute(C.set_buffers(0, [(out, 0)]), fault=True, label="the task's service is gone")
    r.s.events("end")
    return r.s


@scenario
def compute_death_threads_indirect_nil():
    """DispatchThreadsIndirect with indirect buffer ref 0: nil reaches
    Metal."""
    r = Rig("compute_death_threads_indirect_nil", compute_death_threads_indirect_nil.__doc__)
    out, _ = r.buffer()
    pso = r.kernel()
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0)]), C.dispatch_threads_indirect(0, 0),
              fault=True, label="the death")
    r.compute(C.set_buffers(0, [(out, 0)]), fault=True, label="the task's service is gone")
    r.s.events("end")
    return r.s


@scenario
def compute_death_second_encoder():
    """A compute segment that says continuedByNext, then a compute segment
    that does not say continuesPrevious: the carried decoder is reused and
    opens a second compute encoder over the open one (section 6.13 rule
    4). Metal aborts."""
    from pgtest.task import seg
    r = Rig("compute_death_second_encoder", compute_death_second_encoder.__doc__)
    out, _ = r.buffer()
    pso = r.kernel()
    r.ready()
    r.run(seg(abi.ENC_COMPUTE, C.set_pipeline(pso) + C.set_buffers(0, [(out, 0)]), cont_next=1),
          seg(abi.ENC_COMPUTE, C.set_pipeline(pso) + C.set_buffers(0, [(out, 0x40)])
              + C.dispatch_threads((4, 1, 1), (4, 1, 1))),
          fault=True, label="the death")
    r.compute(C.set_buffers(0, [(out, 0)]), fault=True, label="the task's service is gone")
    r.s.events("end")
    return r.s


@scenario
def compute_death_protection_options():
    """A protection envelope (encoder type 5) with options 1 around a compute
    pass: the command buffer takes them (setProtectionOptions:), and opening
    the compute encoder under them aborts inside Metal: the host dies in
    every setup."""
    import struct
    r = Rig("compute_death_protection_options", compute_death_protection_options.__doc__)
    out, _ = r.buffer()
    pso = r.kernel()
    r.ready()
    body = C.set_pipeline(pso) + C.set_buffers(0, [(out, 0)]) + C.dispatch_threads((4, 1, 1), (4, 1, 1))
    inner = abi.SEG_HDR.pack(8 + len(body), abi.ENC_COMPUTE, 0, 0) + body
    r.run(abi.SEG_HDR.pack(16 + len(inner), abi.ENC_PROTECTION, 0, 0) + struct.pack("<Q", 1) + inner,
          fault=True, label="the death")
    r.compute(C.set_buffers(0, [(out, 0)]), fault=True, label="the task's service is gone")
    r.s.events("end")
    return r.s
