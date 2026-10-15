"""Builders for the differential and isolation scenarios
(scenarios/interplay_diff.py, scenarios/interplay_isolation.py).

A differential check runs two encodings of the same work into two guest
regions and compares them with Scenario.cmp, so that a change that breaks
one encoding and not the other is caught even where the absolute result is
a host value. Everything here restates the guest ABI from byte offsets
(docs/testing.md, sections 6.10-6.16), and nothing is taken from the
library's code.
"""
import struct

from . import abi
from .abi import PAGE
from .compute_objects import (Blit as B, Compute as C, compute_blob, compute_pipeline_record, function_record)
from .fifo_util import child
from .task import seg
from .tasks_lib import XGPUTask

BLIT, COMPUTE, RENDER, INFO = abi.ENC_BLIT, abi.ENC_COMPUTE, abi.ENC_RENDER, abi.ENC_INFO

# fixtures/shaders: double_gid (out[gid] = 2 gid), add_one (out[gid] =
# in[gid] + 1), inc_k (one thread: c[0] += 1, c[c[0]] = 0x11 c[0]).
K_DOUBLE = "shaders/kernel.metallib"
K_ADD = "shaders/compute_add.metallib"
K_INC = "shaders/compute_inc.metallib"
K_TEXWRITE = "shaders/compute_texwrite.metallib"


def pattern(n, mul=7, add=3):
    return bytes((i * mul + add) & 0xFF for i in range(n))


def exec_payload(tid, cbs, resources=(), opcode=abi.OP_EXEC_INDIRECT3):
    """The payload of 55, 67 or 69 (reims-vgpu-core's fifo::commands::exec):
    {u32 taskID, u32 resourceCount, u32 cmdBufferCount}, the
    resource records, then cmdBufferCount x {u64 task VA, u64 length}.
    resources are (reference, flags) with the 67/69 flag bits (bit 0
    invalidateHost, 1 invalidateGuest, 2 synchronize);
    for 55 they become the 24-byte record {u32 reference, u8
    invalidateHost, u8 invalidateGuest, u8 synchronize, pad, u64, u64}, or
    are given as (reference, ih, ig, sy) bytes as they are."""
    p = struct.pack("<III", tid, len(resources), len(cbs))
    for r in resources:
        if opcode == abi.OP_EXEC_INDIRECT2:
            if len(r) == 2:
                ref, f = r
                r = (ref, f & 1, (f >> 1) & 1, (f >> 2) & 1)
            p += struct.pack("<IBBBxQQ", r[0], r[1], r[2], r[3], 0, 0)
        else:
            p += struct.pack("<II", *r)
    p += b"".join(struct.pack("<QQ", va, n) for va, n in cbs)
    return p


class Lab(XGPUTask):
    """A GPU task (GPUTask's layout: object list at VA 0, records at pages
    1-3, command buffers at pages 4-7, data from page 8) over a page table
    the scenario may shape (XGPUTask: depth, scatter, leaves), and able to
    run its execs on any FIFO, root or child.

    listed: the references every exec lists with flags 0 (the isolate setup
    needs every guest buffer the GPU touches listed, docs/testing.md
    section 6.12)."""

    def __init__(self, s, tid=1, pages=64, **kw):
        super().__init__(s, tid=tid, pages=pages, **kw)
        self.listed = []

    # ---- objects ----

    def add(self, typ, record):
        return self.ol.add(typ, record)

    def shared(self, length=PAGE, contents=None, flags=0, listed=True, va=None):
        """A buffer (type 1) over its own task pages (or at va); returns
        (ref, va)."""
        ref, va = self.buffer(length=length, flags=flags, contents=contents, va=va)
        if listed:
            self.listed.append(ref)
        return ref, va

    def function(self, fixture):
        """A function (type 6): {u64 task VA of the metallib, u32 length,
        u32 commandType 15, u32 unread}. The metallib sits in pages of its
        own, each written where the task maps it."""
        import os
        from .scenario import FIXTURES
        data = open(os.path.join(FIXTURES, fixture), "rb").read()
        va = self.reserve(len(data))
        if all(self.gpa(va + i) == self.gpa(va) + i for i in range(0, len(data), PAGE)):
            self.s.mem_file(self.gpa(va), fixture)
        else:
            self.write(va, data)
        return self.add(abi.T_FUNCTION, function_record(va, len(data)))

    def kernel(self, fixture):
        """A function (type 6) and a compute pipeline (type 7, commandType
        11) over it."""
        fn = self.function(fixture)
        return self.add(abi.T_SERIALIZED, compute_pipeline_record(compute_blob(fn)))

    def ready(self, fifo=None):
        self.define(fifo=fifo)
        self.install()

    # ---- execution ----

    def resources(self, extra=()):
        return [(r, 0) for r in self.listed] + list(extra)

    def place(self, segments):
        """Writes each segment as its own command-buffer range; returns the
        ranges. The command region (pages 4-7) wraps when it fills."""
        if self._cmd + sum(len(sg) + 0x100 for sg in segments) > 8 * PAGE:
            self.reset_cmdbufs()
        return [self.cmdbuf(sg) for sg in segments]

    def cmd(self, segments, resources=None, opcode=abi.OP_EXEC_INDIRECT3, fifo=None, ranges=None):
        """Writes one exec into a FIFO's ring without ringing; returns its
        stamp."""
        cbs = ranges if ranges is not None else self.place(segments)
        res = self.resources() if resources is None else resources
        return (fifo or self.s.fifo).cmd(opcode, exec_payload(self.tid, cbs, res, opcode))

    def run(self, segments, resources=None, opcode=abi.OP_EXEC_INDIRECT3, fifo=None, fault=False, label=None,
            ranges=None):
        """One exec, rung and waited for (its stamp, or with fault=True its
        fault, whose state is then read and the FIFO resumed)."""
        f = fifo or self.s.fifo
        cbs = ranges if ranges is not None else self.place(segments)
        res = self.resources() if resources is None else resources
        sub = self.s.submit if f.fid == 0 else f.submit
        st = sub(opcode, exec_payload(self.tid, cbs, res, opcode), fault=fault, label=label)
        if fault:
            fault_state(self.s, f, label)
            resume(self.s, f)
        return st


def fault_state(s, f, label):
    """The fault as the guest reads it: 0x102c and 0x103c for the root
    FIFO, 0x102c and the node's fault offset for a child."""
    s.r(abi.R_FAULT_MASK, label=label)
    if f.fid == 0:
        s.r(abi.R_FAULT_OFFSET, label=label)
    else:
        s.dump(f.node_gpa(8), 4, label=(label or "") + ": node fault offset")


def resume(s, f):
    """0x1024 for the root FIFO, 0x1028 = id for a child
    (resume_child_fifo)."""
    if f.fid == 0:
        s.resume_root()
    else:
        s.w(abi.R_RESUME_CHILD, f.fid)


def cmp_va(s, ta, va, tb, vb, n, label):
    """Compares n bytes at task VA va of ta with those at vb of tb, one
    comparison a page run (a task's pages need not be adjacent in guest
    RAM)."""
    pos = 0
    part = 0
    while pos < n:
        k = min(n - pos, PAGE - (va + pos) % PAGE, PAGE - (vb + pos) % PAGE)
        s.cmp(ta.gpa(va + pos), tb.gpa(vb + pos), k, label="%s [%d]" % (label, part) if n > k or part else label)
        pos += k
        part += 1


def dump_va(t, va, n, label):
    """A dump of n bytes at task VA va, one a page run (see cmp_va)."""
    pos = 0
    part = 0
    while pos < n:
        k = min(n - pos, PAGE - (va + pos) % PAGE)
        t.s.dump(t.gpa(va + pos), k, label="%s [%d]" % (label, part) if n > k or part else label)
        pos += k
        part += 1


# ---- workloads ----

def inc(pso, buf, off):
    """One inc_k dispatch over buf at off: pipeline, buffer 0, one thread
    (deserializer::compute)."""
    return C.set_pipeline(pso) + C.set_buffers(0, [(buf, off)]) + C.dispatch_threads((1, 1, 1), (1, 1, 1))


class Workload:
    """The same order-dependent work, expressed as steps that encodings
    split differently: each step is (encoder type, records). Its result is
    in one buffer region of 0x400 bytes at `off`:

      +0x000  inc_k's counter and marks (16 u32)
      +0x100  a blit copy of +0 after three dispatches
      +0x200  a blit fill of 0x10 bytes of 0x77
      +0x300  a blit copy of +0 at the end
    """

    def __init__(self, inc_pso):
        self.pso = inc_pso

    def steps(self, buf, off=0):
        d = lambda: (COMPUTE, inc(self.pso, buf, off))
        return [d(), d(), d(),
                (BLIT, B.copy_buffer_to_buffer(buf, buf, off, off + 0x100, 0x40)),
                d(), d(),
                (BLIT, B.fill(buf, off + 0x200, 0x10, 0x77) + B.copy_buffer_to_buffer(buf, buf, off, off + 0x300, 0x40))]


def merge(steps):
    """Adjacent steps of the same encoder type as one segment each."""
    out = []
    for enc, r in steps:
        if out and out[-1][0] == enc:
            out[-1] = (enc, out[-1][1] + r)
        else:
            out.append((enc, r))
    return [seg(e, r) for e, r in out]


def one_each(steps):
    """Every step its own segment."""
    return [seg(e, r) for e, r in steps]


def continued(steps):
    """Adjacent steps of the same type as a chain of continued segments: the
    first says continuedByNext, the middle ones both, the last
    continuesPrevious, so they share one encoder (deserializer::walk). A
    continued compute segment after the first drops its SetPipelineState:
    the encoder keeps it."""
    groups = []
    for enc, r in steps:
        if groups and groups[-1][0] == enc:
            groups[-1][1].append(r)
        else:
            groups.append((enc, [r]))
    out = []
    for enc, rs in groups:
        n = len(rs)
        for i, r in enumerate(rs):
            if i and enc == COMPUTE and r.startswith(C.set_pipeline(0)[:4]):
                r = r[len(C.set_pipeline(0)):]
            out.append(seg(enc, r, cont_prev=1 if i else 0, cont_next=1 if i < n - 1 else 0))
    return out


def children(s, *fids):
    """Child FIFOs, defined on the root FIFO (fifo_util.child)."""
    return [child(s, fid) for fid in fids]


def refs_payload(tid, refs):
    return struct.pack("<II", tid, len(refs)) + b"".join(struct.pack("<I", r) for r in refs)
