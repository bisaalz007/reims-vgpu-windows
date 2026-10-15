"""A kernel of fixed work, for the service's exec that completes twice.

When completing the resources of the service's exec raises after the
exec's command buffer is committed (a view listed with synchronize, a
transfer buffer remade over a hole), the service completes the exec twice:
its catch replies 2, and the synchronize command buffer it committed later
sends completeAsyncTokenID with success, once the exec's command buffer has
completed (reims-vgpu-core's gpu_task::exec). The host keeps whichever
reaches it first and drops the other (tasks::remote::completions: the later
one finds no token). Left to the GPU's speed that is a race, and under load
the success won: no fault, where the expected files have one.

The scenarios that meet it run fresh_res's spin kernel (fixtures/shaders/
fresh_res_spin.metal) in the same exec, after the work that matters, as a
fixed amount of work: c[0] stays 0, so the one thread polls until its cap
of c[1] polls and ends by itself, writing c[2] = 0xdead. The exec's command
buffer cannot complete, and the second completion cannot be sent, before
that; the cap is 40,000 polls, about 2 s on the reference host (2M polls
are about 100 s). The reply that faults the exec is sent right after the
exec's commit, in milliseconds, so the fault comes first by a margin of
three orders of magnitude, where a plain fill's command buffer completes in
microseconds and under load won the race.

Load does not only make the kernel slower. Right after a GPU restart (the
exclusive case before such a scenario often restarts the GPU), another
process's GPU work gets this one long dispatch killed by the GPU's hang
watchdog about 0.5 s after the commit ("Caused GPU Hang Error"): c[2]
stays 0 and the wait for it times out. Measured on two builds, runs
failed alike: with no load 0/13 and 0/14; with a 1080p frame drawn every
33 ms 3/10 and 1/10; every 8 ms 10/10 each. A plain Metal program running
the same kernel is killed the same way. The same work split into many
short serial dispatches survived every such run, but the scenarios keep
the one long dispatch: keep other GPU clients (a VM, video) off the
machine during full runs.

Why not let the guest release the kernel after it has seen the fault
(fresh_res's design, causal on paper): Metal makes a CPU write to a shared
buffer visible to the GPU only at command-buffer boundaries, and a running
kernel can keep reading a stale c[0]. In full rounds at -j 32 and 48 every
held kernel of a round (fresh_res's too) missed the guest's release for
30 s, and in one bless run a kernel polled to its cap though the guest had
released it. Here every CPU write to c happens before the exec is
committed, so nothing depends on that visibility; the kernel's length is
its own.

Such a scenario runs exclusive=True (README rule 10): alone, after the
"hold" cases, so no held kernel of another case shares the GPU with it.
"""
import struct

from . import abi
from .abi import PAGE
from .compute_objects import Compute as C, compute_blob, compute_pipeline_record, function_record
from .task import seg

SPIN = "shaders/fresh_res_spin.metallib"
SPIN_CAP = 40000
RELEASED = 0x600D
AT_CAP = 0xDEAD


class Spin:
    """The spin kernel's objects in a GPUTask t (before t.define()): its
    buffer c (shared, c[0] = 1 so that the warm-up ends at once), a
    function and a pipeline. seg is the compute segment that runs it."""

    def __init__(self, t):
        self.t = t
        self.c, self.c_va = t.buffer(PAGE, contents=struct.pack("<IIII", 1, SPIN_CAP, 0, 0))
        f_va = t.reserve(0x2000)
        f = t.ol.add(abi.T_FUNCTION, function_record(f_va, t.s.mem_file(t.gpa(f_va), SPIN)))
        pso = t.ol.add(abi.T_SERIALIZED, compute_pipeline_record(compute_blob(f)))
        self.seg = seg(abi.ENC_COMPUTE, C.set_pipeline(pso) + C.set_buffers(0, [(self.c, 0)]) +
                       C.dispatch_threads((1, 1, 1), (1, 1, 1)))

    def warm_up(self):
        """After t.install(): one exec runs the kernel with c[0] already
        set, so the pipeline is compiled and the kernel's path proven
        before the exec that runs it to its cap; then c[0] and c[2] are
        zero again, written before that exec is committed."""
        t = self.t
        t.reset_cmdbufs()
        t.exec3([t.cmdbuf(self.seg)], resources=[(self.c, 0)],
                label="warm-up: c[0] already set, the kernel ends at once")
        t.dump(self.c_va, 12, "c after the warm-up")
        t.write(self.c_va, struct.pack("<III", 0, SPIN_CAP, 0))

    def payload(self, cmdbufs, resources):
        """An ExecIndirect3 payload: these command-buffer ranges, then the
        spin in a range of its own; c listed after these resources."""
        t = self.t
        cbs = list(cmdbufs) + [t.cmdbuf(self.seg)]
        res = list(resources) + [(self.c, 0)]
        return (struct.pack("<III", t.tid, len(res), len(cbs))
                + b"".join(struct.pack("<II", r, f) for r, f in res)
                + b"".join(struct.pack("<QQ", va, n) for va, n in cbs))

    def wait_end(self, label="the kernel ended at its cap: c[2] = 0xdead"):
        """Waits until the kernel has run its fixed work (c[2] = 0xdead): the
        exec's command buffer then completes."""
        t = self.t
        t.s.wait(gpa=t.gpa(self.c_va + 8), eq=AT_CAP, timeout_ms=60000, label=label)
