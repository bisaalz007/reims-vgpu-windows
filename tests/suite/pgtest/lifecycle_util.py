"""Builders for the lifecycle interplay scenarios (scenarios/interplay_lifecycle.py):
host lifecycle calls that land in the middle of guest work.

A Rig is a guest with a root FIFO, a gate (a child FIFO whose stamp the
other FIFOs' barriers wait on, so that a command can be held queued for as
long as the scenario likes), and GPU task 1 with a source buffer and two
regions, A and B, each a shared destination buffer and a private buffer.
The same two-part workload runs on region A without interruption and on
region B with a lifecycle call between (or under) its parts; a cmp of the
two destinations then says whether the interruption was transparent.

Layouts are the guest ABI as abi.py, fifo.py and task.py restate it: the
exec payload and its resource flags, barriers, the blit records (fill 306,
copy 301), the buffer record. Nothing here is taken from the library's
code.
"""
import struct

from . import abi
from .abi import PAGE
from .fifo import ChildFIFO
from .task import GPUTask, rec, seg

# Resource flags a region's exec passes: the shared destination is
# invalidated on the host and synchronized back (5), the source only
# invalidated (1), the private buffer neither (0) by default; with
# RES_INVALIDATE_GUEST (2) the host marks the guest's copy stale, which is
# what makes a flush write it back (suspend_writeback).
F_DST = abi.RES_INVALIDATE_HOST | abi.RES_SYNCHRONIZE
F_SRC = abi.RES_INVALIDATE_HOST
F_PRIV = 0

DST_LEN = 0x480  # what the workload writes of a destination


def fill(ref, off, n, byte):
    """Blit 306: {u32 buffer, u64 offset, u64 length, u8 value}."""
    return rec(306, struct.pack("<IQQB3x", ref, off, n, byte))


def copy(src, dst, soff, doff, n):
    """Blit 301: {u32 src, u32 dst, u64 srcOffset, u64 dstOffset, u64 size}."""
    return rec(301, struct.pack("<IIQQQ", src, dst, soff, doff, n))


def exec3_payload(tid, cmdbufs, resources=()):
    """ExecIndirect3 (67): {u32 taskID, u32 resourceCount, u32
    cmdBufferCount}, resources {u32 ref, u32 flags}, command buffers {u64 VA,
    u64 length}."""
    p = struct.pack("<III", tid, len(resources), len(cmdbufs))
    p += b"".join(struct.pack("<II", r, f) for r, f in resources)
    p += b"".join(struct.pack("<QQ", va, n) for va, n in cmdbufs)
    return p


class Region:
    def __init__(self, name, dst, dst_va, priv, priv_va):
        self.name = name
        self.dst, self.dst_va, self.priv, self.priv_va = dst, dst_va, priv, priv_va


class Rig:
    """The guest: bring-up, the gate (child FIFO `gate_fid`), task 1 with a
    source buffer (bytes 0..255 repeated) and regions A and B, the object
    list installed. Each part of the workload is one command buffer placed
    once per region, so the two regions run identical records."""

    def __init__(self, s, gate_fid=1, task_pages=32, worker=None, priv_flags=F_PRIV):
        self.s = s
        self.priv_flags = priv_flags
        self.f = s.bring_up()
        self.gate = ChildFIFO(s, gate_fid)
        self.gate.setup_node()
        self.gate.define()
        self.gate_value = 0
        self.worker = None
        if worker is not None:
            self.worker = ChildFIFO(s, worker)
            self.worker.setup_node()
            self.worker.define()
        t = self.t = GPUTask(s, tid=1, pages=task_pages)
        self.src, self.src_va = t.buffer(contents=bytes(range(256)) * 4)
        self.regions = {}
        for name in ("A", "B"):
            d, dva = t.buffer()
            # The private buffer's guest pages hold 0x77 bytes: a buffer the
            # host makes again from guest memory shows them.
            p, pva = t.buffer(flags=1, contents=b"\x77" * 0x200)
            self.regions[name] = Region(name, d, dva, p, pva)
        t.define()
        t.install()
        self.cb = {}
        for name, r in self.regions.items():
            self.cb[name] = (t.cmdbuf(seg(abi.ENC_BLIT, self.part1(r))), t.cmdbuf(seg(abi.ENC_BLIT, self.part2(r))))

    # ---- the workload ----

    def part1(self, r):
        """Fill the destination, copy the source into the private buffer and
        fill its second half."""
        return (fill(r.dst, 0, 0x200, 0x11) + copy(self.src, r.priv, 0, 0, 0x100)
                + fill(r.priv, 0x100, 0x100, 0x33))

    def part2(self, r):
        """Copy the private buffer out, overlay a source range and fill a
        tail: its result depends on part 1 having run first."""
        return (copy(r.priv, r.dst, 0, 0x200, 0x200) + copy(self.src, r.dst, 0x80, 0x40, 0x80)
                + fill(r.dst, 0x400, 0x80, 0x44))

    def resources(self, r):
        return [(self.src, F_SRC), (r.dst, F_DST), (r.priv, self.priv_flags)]

    def payload(self, name, part):
        r = self.regions[name]
        return exec3_payload(self.t.tid, [self.cb[name][part]], self.resources(r))

    def run(self, name, part, fifo=None, **kw):
        """One part on one region, submitted and waited for."""
        return (fifo or self.s).submit(abi.OP_EXEC_INDIRECT3, self.payload(name, part), **kw)

    def queue(self, name, part, fifo=None, barriers=()):
        """Rings one part without waiting; returns its stamp."""
        f = fifo or self.s.fifo
        st = f.cmd(abi.OP_EXEC_INDIRECT3, self.payload(name, part), barriers=barriers)
        f.doorbell()
        return st

    def control(self):
        """Region A, uninterrupted."""
        self.run("A", 0)
        self.run("A", 1)

    # ---- the gate ----

    def gate_barrier(self):
        """A barrier on the gate's next stamp."""
        return [(self.gate.stamp_index, self.gate_value + 1)]

    def open_gate(self, **kw):
        """A NOP on the gate: its stamp releases what waits on it."""
        self.gate_value += 1
        return self.gate.submit(abi.OP_NOP, **kw)

    # ---- checkpoints ----

    def compare(self, label="A and B"):
        """The two destinations, compared, and B's dumped."""
        s = self.s
        a, b = self.regions["A"], self.regions["B"]
        s.cmp(self.t.gpa(a.dst_va), self.t.gpa(b.dst_va), DST_LEN, label=label)
        self.t.dump(b.dst_va, DST_LEN, label="B's destination")

    def priv_guest(self, label):
        """The guest pages under both private buffers (written only by a
        writeback)."""
        for name in ("A", "B"):
            self.t.dump(self.regions[name].priv_va, 0x200, label="%s: %s's private pages" % (label, name))


def suspend(s, slot="s"):
    s.api("willSuspend")
    s.api("finishSuspend", slot=slot)


def resume_fresh(s, slot="s", label=None):
    """A migration: a fresh device over the same guest RAM, the blob, and
    rvg_device_did_resume (as scenarios/suspend.py does)."""
    s.api("recreate")
    s.api("willResume", slot=slot, label=label)
    s.api("didResume")


def node(s, c, label):
    s.dump(c.node_gpa(0), abi.CHILD_NODE_SIZE, label=label)


def stamps(s, n, label):
    s.dump(s.fifo.base_gpa(), 4 * n, label=label)
