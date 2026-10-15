"""Builders for the interplay scenarios (scenarios/interplay_fifo.py): a
root FIFO, child FIFOs and one or more GPU tasks whose execs any FIFO can
carry, gated on each other's stamps (docs/testing.md, sections 6.5-6.7,
6.12).

What the implementation orders, and so what these builders lean on:
- A FIFO publishes its stamps in command order: the FIFO thread pushes each
  command's stamp before dispatching it, and the pending-stamp queue's pump
  publishes only the last contiguous signalled slot from the head
  (reims-vgpu-core's fifo::stamps::StampQueue: push_stamp, signal_stamp,
  pump).
- A barrier {index, value} blocks the FIFO, before the command is consumed,
  until the published slot `index` reaches `value` by serial arithmetic
  (device::waits). A command gated on another FIFO's stamp therefore runs
  after that stamp's command completed: an exec's stamp is published from
  its command buffer's completion (in the service, from the reply).
- DeviceCore::signal_stamp stores the slot and raises the interrupt under
  the signal mutex a blocked barrier re-checks under, so the publisher's
  interrupt comes before anything the release lets run.

What it does not order: two FIFOs that are not gated on each other run on
their own threads, so their interleaving is timing. And an async command
(an exec) followed on the same FIFO by commands the guest wrote in the same
doorbell may be published alone or together with them, depending on when
its command buffer completes; every exec here is therefore either the last
thing its FIFO holds until its stamp is waited for, or followed only by
commands gated on its own publication.

Nothing here is imported from the library's code.
"""
import struct

from . import abi
from .abi import PAGE
from .compute_objects import compute_blob, compute_pipeline_record, fence_record, function_record
from .fifo_util import child, track_stamps
from .scenario import Scenario
from .task import GPUTask, seg

ADD = "shaders/compute_add.metallib"   # out[gid] = in[gid] + 1
INC = "shaders/compute_inc.metallib"   # c[0] += 1; c[c[0]] = c[0] * 0x11 below 16


def u32s(*v):
    return struct.pack("<%dI" % len(v), *v)


def exec3_payload(tid, cmdbufs, resources):
    """ExecIndirect3 (67): {u32 taskID, u32 resourceCount, u32
    cmdBufferCount}, resourceCount x {u32 reference, u32 flags},
    cmdBufferCount x {u64 task VA, u64 length}."""
    p = struct.pack("<III", tid, len(resources), len(cmdbufs))
    p += b"".join(struct.pack("<II", r, f) for r, f in resources)
    p += b"".join(struct.pack("<QQ", va, n) for va, n in cmdbufs)
    return p


class World:
    """A scenario with the root FIFO, child FIFOs `children` (ids, stamp
    index = id unless `stamp_indices` says otherwise), and GPU tasks
    (GPUTask: object list at VA 0, records at pages 1-3, command buffers at
    pages 4-7, which are never reused within a scenario so that an exec's
    command buffer is intact whenever its FIFO gets to it).

    Objects are added per task; ready() defines every task and installs its
    object list on the root FIFO (two root stamps a task)."""

    def __init__(self, name, doc, children=(), slots=None, tasks=1, stamp_indices=None, child_pages=None,
                 root_pages=2, **kw):
        self.s = Scenario(name, doc=doc, **kw)
        self.f = self.s.bring_up(fifo_pages=root_pages)
        self.tasks = [GPUTask(self.s, tid=i + 1, pages=64) for i in range(tasks)]
        self.t = self.tasks[0]
        self.listed = {t.tid: [] for t in self.tasks}
        self.c = {}
        si = stamp_indices or {}
        cp = child_pages or {}
        for fid in children:
            kw_c = {"define": False, "pages": cp.get(fid, 1)}
            if fid in si:
                kw_c["stamp_index"] = si[fid]
            self.c[fid] = child(self.s, fid, **kw_c)
        n = slots if slots is not None else 1 + max([self.c[i].stamp_index for i in self.c] or [0])
        track_stamps(self.s, n)
        self.nslots = n

    def fifo(self, fid):
        return self.f if fid == 0 else self.c[fid]

    # ---- objects ----

    def buffer(self, length=PAGE, contents=None, flags=0, task=None, listed=True):
        """A buffer (type 1) over task pages; flags bit 0 private. Returns
        (ref, va). Listed buffers go into every exec's resource list with
        flags 0 (section 6.12)."""
        t = task or self.t
        ref, va = t.buffer(length=length, flags=flags, contents=contents)
        if listed:
            self.listed[t.tid].append(ref)
        return ref, va

    def kernel(self, fixture, task=None):
        """A function (type 6) and a compute pipeline (type 7, commandType
        11) over it; returns the pipeline's ref."""
        t = task or self.t
        va = t.reserve(0x2000)
        n = self.s.mem_file(t.gpa(va), fixture)
        fn = t.ol.add(abi.T_FUNCTION, function_record(va, n))
        return t.ol.add(abi.T_SERIALIZED, compute_pipeline_record(compute_blob(fn)))

    def fence(self, task=None):
        """A fence (type 7, commandType 13)."""
        return (task or self.t).ol.add(abi.T_SERIALIZED, fence_record())

    # ---- bring-up ----

    def define_children(self):
        """Opcode 48 for every child, in id order, on the root FIFO."""
        for fid in sorted(self.c):
            self.c[fid].define()

    def ready(self, children=True):
        """Defines the children (unless children=False), then every task and
        its object list."""
        if children:
            self.define_children()
        for t in self.tasks:
            t.define()
            t.install()

    # ---- execs ----

    def payload(self, segments, task=None, resources=None):
        """An exec payload: each segment in a command buffer range of its
        own, every listed buffer of the task in its resource list."""
        t = task or self.t
        cbs = [t.cmdbuf(sg) for sg in segments]
        res = [(r, 0) for r in self.listed[t.tid]] if resources is None else resources
        return exec3_payload(t.tid, cbs, res)

    def exec_cmd(self, fid, *segments, barriers=(), task=None, resources=None, size=None, stamp=None,
                 opcode=abi.OP_EXEC_INDIRECT3):
        """Writes an exec (67, or 69 with opcode=) on FIFO fid without ringing
        its doorbell; returns its stamp. size: the whole command's length, its
        payload padded with zeros (a longer payload is accepted, section 6.8),
        to move the ring along."""
        return self.fifo(fid).cmd(opcode, self.payload(segments, task, resources),
                                  barriers=barriers, length=size, pad_to=size, stamp=stamp)

    def exec(self, fid, *segments, barriers=(), wait=True, label=None, **kw):
        """An exec on FIFO fid, doorbell rung; waits for its stamp unless
        wait=False. Returns the stamp."""
        st = self.exec_cmd(fid, *segments, barriers=barriers, **kw)
        f = self.fifo(fid)
        f.doorbell()
        if wait:
            f.wait_stamp(st, label=label)
        return st

    def nop(self, fid, barriers=(), wait=True, label=None, stamp=None):
        f = self.fifo(fid)
        st = f.cmd(abi.OP_NOP, barriers=barriers, stamp=stamp)
        f.doorbell()
        if wait:
            f.wait_stamp(st, label=label)
        return st

    def blocked(self, fid, st, label, timeout_ms=300):
        """Waits a short while on a stamp the scenario expects NOT to be
        published (the wait's observation says ok: false). It also gives the
        FIFO's thread the time to register its barrier wait, where a
        scenario depends on that (the prior art: fifo_barrier_circular)."""
        self.fifo(fid).wait_stamp(st, timeout_ms=timeout_ms, label=label)

    def stamps(self, label="stamp slots"):
        self.s.dump(self.f.base_gpa(), 4 * self.nslots, label=label)

    def node(self, fid, label=None):
        self.s.dump(self.c[fid].node_gpa(0), abi.CHILD_NODE_SIZE, label=label or "node %d" % fid)


def blit(*records):
    return seg(abi.ENC_BLIT, b"".join(records))


def compute(*records):
    return seg(abi.ENC_COMPUTE, b"".join(records))
