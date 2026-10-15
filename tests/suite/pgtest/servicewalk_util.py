"""Service walks (scenarios/interplay_servicewalk.py): one seeded walker that
drives many GPU tasks at once, each with its own objects, across the root
FIFO and child FIFOs, and that creates, deletes, faults and finally kills
them while the others run.

Under isolation (setup isolate, the device config's process_isolation;
docs/testing.md, section 3.3) every task but task 0 is a remote task with a
service of its own, on threads of its own: a define (56) is a handshake
that starts one, a delete (32) ends it, and a death in its own code is a
fault only its task sees (section 6.10, "A task whose service died"); a
death inside Metal ends the host. In the blocks and map setups the same
tasks are local tasks in the host. The walker runs the same programs in
every setup where the behaviour is the same, and records the divergences
(sections 6.10 and 7.2) where it is not.

The walker composes the builders the per-area scenarios use:
  - pgtest/diff_util.py (Lab, Workload, exec_payload, fault_state,
    resume): a task whose execs any FIFO can carry, and inc_k's
    order-dependent workload;
  - pgtest/compute_objects.py: blit and compute records, compute pipelines
    (type 7, commandType 11) over functions (type 6);
  - pgtest/exec_objects.py and pgtest/icb_cmds.py: an indirect command
    buffer (type 20) of one double_gid dispatch, prepared by the CPU from
    the guest's command and executed from a compute pass (0x87, 0xe4); a
    texture over a buffer (type 8, commandType 9) that a render pass clears;
  - pgtest/resources_build.py: a private heap (type 13) and a buffer placed
    in it (type 14), a read-only buffer (type 1, flag bit 1);
  - pgtest/lifetime_util.py: objects deleted (37, 40) and made again at
    their next use, the outcome the setups decide (either());
  - pgtest/tasks_lib.py: HeapTextureSizeAndAlign (64);
  - the deaths of section 7.4 (scenarios/interplay_isolation.py's triggers).

A step is one batch: a set of nodes, each one command on one FIFO, written
in creation order and rung together. Every task has at most one chain of
nodes a batch; a node of the chain on another FIFO than the one before it
waits on that one's stamp with a barrier, and every node after the first
on a FIFO waits on its FIFO-previous node (a barrier on the FIFO's own
stamp index), so every command is published alone and interrupt counts
are exact (pgtest/interplay_util.py: an exec followed by commands in the
same doorbell may be published with them, depending on timing). Nodes:
  - a program: one exec (67, or 55 at random) of the task, writing one
    0x400-byte slot of its output buffer (below);
  - lifecycle: DefineTask2 (56) and SetObjectList (51) of a new task, whose
    chain may go on with its first program; DeleteTask (32) of a task the
    batch does not otherwise use;
  - resource churn: DeleteResource (37) of the source or output buffer,
    DeleteObject (40) of a compute pipeline or its function, Discard (63)
    and Synchronize (53) of the output buffer, or the guest rewriting the
    source buffer's bytes; each object is made again from its record at
    its next use;
  - info: GetComputeInfo (59) of a pipeline and HeapTextureSizeAndAlign
    (64), each into a reply area the guest filled with 0xEE first;
  - at most one fault a batch, the last node on its FIFO, nothing waiting
    on it: a decode error (an unknown blit record, 399, after a fill of the
    task's slot), a record whose length runs past its segment, a reference
    past the object list, an unknown opcode (99), or an exec of a task
    deleted by the same batch, on another FIFO, behind the delete.

Programs (every result a 0x400-byte slot whose bytes the model knows):
  - blit: a fill, a copy of the source buffer, a second fill;
  - double: a fill, then double_gid and add_one of it in one compute
    encoder (out[i] = 2i, then 2i + 1);
  - inc: a zero fill, then diff_util's Workload (inc_k's counter and marks,
    copied between dispatches);
  - heapfill / heapcopy: a fill of the heap buffer (and of the slot), and in
    a later batch a copy of the heap buffer into the slot;
  - icb: the ICB's double_gid dispatch into its own buffer, copied into the
    slot;
  - render: a render pass that clears the texture over a buffer to one of
    four colours, copied into the slot (copy_texture_to_buffer);
  - info (isolate walks only): a fill of the slot, then an info segment
    (type 4) whose 451 reply the service writes into the info page at
    decode time; in-process a reply aborts the host (section 7.4), so the
    all-setup walks meet it only as an ending;
  - ro: a fill of the read-only buffer, which lands in-process and is
    dropped by the service (section 6.10's table): dumped, not modelled.

The first tasks are defined in one order across the FIFOs (each define
waits on the one before it): the blocks setup's create_task numbers the
host's tasks in the order it is called (runner/pgrun.m), and its block
calls name them. Later batches define at most one task each, and in the
walks that run in the blocks setup a delete in the same batch waits on the
define: without a memory map a define maps the task's root page into the
root task at an offset taken from a first-fit allocator (reims-vgpu-core's
collections::first_fit), and a delete frees its task's, so a define and a
delete in flight at once on two FIFOs make the define's offset, which the
map_memory and unmap_memory records name, timing.

The model: every slot's bytes (a faulted exec commits nothing, docs 6.12),
the heap buffer's first 0x100 bytes (unknown after the task is deleted:
in-process the heap's host copy dies with the task, and the service's heap
is the guest pages, section 6.10's "DeleteTask after GPU writes" row), and
every info reply equal to the first reply of its kind. After each batch
every slot the batch touched is compared (cmp) with the model's bytes,
which the guest writes into a page of its own first.

Divergences (all-setup walks), each a root command whose outcome the setups
decide (lifetime_util.either: the observation says whether it completed or
faulted), while a batch of work on the child FIFOs is in flight: MapMemory2
(57) at an offset that is not page-aligned (blocks fault), UnmapMemory (34)
past the task's end (blocks and map fault), GetComputeInfo (59) with a
count of 0 inside a page (blocks fault).

Endings, last:
  - all-setup walks may end with one death trigger on task 1's exec, while
    the child FIFOs run the other tasks' work: a range past the task's end
    (null cursor: host death in-process, the task's own death under
    isolation), a last segment that says continuedByNext (Metal asserts at
    the commit, inside Metal: a host death in every setup), a page-aligned
    zero-length range (as the range past the end), an info reply (451: an
    in-process task's reply cursor aborts, the service writes it), or
    GetComputeInfo on pipeline 0 (in-process zero values, the service
    asserts and dies). The wait the host dies in is labelled
    [dies]. Where the process lives on, task 1's next exec completes or
    faults (either()), and it is deleted, defined again and run; the other
    tasks run a last batch and are compared.
  - isolate walks end with one to three deaths at once, each on its own
    task and FIFO, beside the others' work; every other task keeps running
    and its bytes stay right; each dead task's next exec faults, and a
    delete and a define give it a new service that runs its program. A
    death inside Metal (open_encoder) ends the process instead, and none of
    that is reached. It lands asynchronously to the runner's ops, so its
    command goes behind 256 NOPs, its FIFO's doorbell is rung last and its
    fault is waited for first: the process dies in that wait every run.

Every wait succeeds but those labelled "completed (0) or faulted (1)" (whose
"which" is the setup's outcome) and those labelled [dies];
python3 -m pgtest.servicewalk_util reads the expected files back and checks
it, with every cmp equal.
"""
import json
import os
import struct

from . import abi
from .abi import PAGE
from .compute_objects import Blit as B, Compute as C, compute_pipeline_record
from .diff_util import (BLIT, COMPUTE, K_ADD, K_DOUBLE, K_INC, RENDER, INFO, Lab, Workload, exec_payload,
                        fault_state, merge, pattern, resume)
from .exec_objects import (ICB_DISPATCH_THREADS, buffer_texture_record, compute_blob, icb_descriptor, icb_record,
                           info_heap_size, render_pass)
from .fifo_util import HEALTHY, NODE_FAULT, child
from .icb_cmds import CMD, compute_cmd, layout
from .lifetime_util import either
from .randgen import Rng
from .resources_build import heap_buffer_rec, heap_rec
from .scenario import RAM_BASE, SETUPS, Scenario
from .session import DEL_COMPUTE_PSO, DEL_FUNCTION
from .task import rec, seg
from .tasks_lib import heap_payload, heap_ser

E3, E2 = abi.OP_EXEC_INDIRECT3, abi.OP_EXEC_INDIRECT2
ISO = ("isolate",)
SLOT = 0x400
DIES = " [dies]"
EITHER = "completed (0) or faulted (1)"
FILL_REPLY = 0xEE
# Clear colours whose RGBA8Unorm bytes are exact (0.0 and 1.0).
COLOURS = ((0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (1.0, 1.0, 1.0, 0.0))
FAULTS = ("decode", "overlong", "badref", "opcode", "deleted")
DEATHS_ALL = ("past_end", "open_encoder", "zero_length", "info_reply", "compute_info_0")
DEATHS_ISO = ("past_end", "open_encoder", "zero_length", "compute_info_0")
# The deaths inside Metal: the process ends under isolation too.
DEATHS_METAL = ("open_encoder",)


SRC_LEN = 0x100      # the source buffer's patterned bytes (the rest of its page is zero)


def emit_bytes(s, gpa, data):
    """Writes data at gpa with few, small ops: runs of one byte as fill,
    runs of u32s in arithmetic progression as mem_seq, the rest as mem."""
    data = bytes(data)
    n = len(data)
    i = 0
    raw = bytearray()
    raw_at = 0

    def flush():
        if raw:
            s.mem(gpa + raw_at, bytes(raw))
            raw.clear()
    while i < n:
        j = i
        while j < n and data[j] == data[i]:
            j += 1
        if j - i >= 16:
            flush()
            s.fill(gpa + i, j - i, data[i])
            i = j
            raw_at = i
            continue
        if i % 4 == 0 and i + 32 <= n:
            w = struct.unpack_from("<8I", data, i)
            step = (w[1] - w[0]) & 0xFFFFFFFF
            if all((w[k + 1] - w[k]) & 0xFFFFFFFF == step for k in range(7)):
                k = i + 32
                while k + 4 <= n and (struct.unpack_from("<I", data, k)[0] - struct.unpack_from("<I", data, k - 4)[0]) \
                        & 0xFFFFFFFF == step:
                    k += 4
                flush()
                s.mem_seq(gpa + i, (k - i) // 4, 4, bytes(4), fields=[(0, 4, w[0], step)])
                i = k
                raw_at = i
                continue
        if not raw:
            raw_at = i
        raw.append(data[i])
        i += 1
    flush()


# ---- a task ----

class SvcTask(Lab):
    """One walker task: Lab's layout (object list at VA 0, records at pages
    1-3, command buffers at pages 4-7, data from page 8), with the objects
    its profile asks for, each over its own task pages:

      src      a shared buffer (type 1) of a pattern, one page;
      out      the output buffer, `out_pages` pages of 0x400-byte slots;
      compute  an info page (replies), double_gid, add_one and inc_k
               (functions, type 6, and compute pipelines, type 7);
      icb      a pipeline over double_gid that supports ICBs, an ICB (type
               20) of one concurrentDispatchThreads command binding icb_out,
               and icb_out;
      heap     a private heap (type 13, flags 1) of one page, and a buffer
               placed in it (type 14) over the whole of it;
      render   a buffer and a 4x4 RGBA8Unorm texture over it (type 8,
               commandType 9, bytesPerRow 16);
      ro       a read-only buffer (type 1, flags 2).

    Every exec lists every buffer, the heap, the heap buffer and the ICB
    with flags 0 (the service needs every guest buffer the GPU touches
    listed, docs/testing.md section 6.12)."""

    def __init__(self, s, tid, profile, out_pages=1):
        need = 8 + 1 + out_pages
        if "compute" in profile:
            need += 4
        if "icb" in profile:
            need += 2
        if "heap" in profile:
            need += 1
        if "render" in profile:
            need += 1
        if "ro" in profile:
            need += 1
        super().__init__(s, tid=tid, pages=need)
        self.profile = frozenset(profile)
        self.src_bytes = bytearray(pattern(SRC_LEN, 5 + 2 * tid, tid))
        self.src, self.src_va = self.shared(contents=bytes(self.src_bytes))
        self.out, self.out_va = self.shared(length=out_pages * PAGE)
        self.nslots = out_pages * PAGE // SLOT
        self.slots = [bytes(SLOT)] * self.nslots
        self.next_slot = 0
        if "compute" in profile:
            self.info_va = self.reserve(PAGE)
            # The info page as a buffer too: the info decoder's replies
            # (type 4 segments) name a buffer.
            self.reply, _ = self.shared(va=self.info_va)
            self.fn_dbl = self.function(K_DOUBLE)
            self.dbl = self.add(abi.T_SERIALIZED, compute_pipeline_record(compute_blob(self.fn_dbl)))
            self.add1 = self.kernel(K_ADD)
            self.inc = self.kernel(K_INC)
            self.info_next = 0
        if "icb" in profile:
            self.icb_pso = self.add(abi.T_SERIALIZED, compute_pipeline_record(compute_blob(self.fn_dbl, icb=True)))
            self.icb_out, self.icb_out_va = self.shared()
            va = self.reserve(PAGE)
            self.write(va, compute_cmd(pso=self.icb_pso, binds=[(self.icb_out, 0, 0)], grid=(64, 1, 1),
                                       tg=(64, 1, 1)))
            self.icb = self.add(abi.T_ICB, icb_record(CMD, va // PAGE, icb_descriptor(ICB_DISPATCH_THREADS, kbufs=1),
                                                      layout("compute"), max_count=1))
            self.listed.append(self.icb)
        if "heap" in profile:
            self.heap_va = self.reserve(PAGE)
            self.heap = self.add(abi.T_HEAP, heap_rec(self.heap_va // PAGE, PAGE, flags=1))
            self.hb = self.add(abi.T_HEAP_BUFFER, heap_buffer_rec(self.heap, 0, PAGE))
            self.listed += [self.heap, self.hb]
            self.heap_bytes = None      # the heap buffer's first 0x100 bytes, when the model knows them
        if "render" in profile:
            self.tb, self.tb_va = self.shared()
            self.tex = self.add(abi.T_SERIALIZER_TEXTURE, buffer_texture_record(self.tb))
        if "ro" in profile:
            self.ro, self.ro_va = self.shared(flags=2)
        self.ol.write()
        self.live = False
        self.dead = False                 # its service died (isolate)

    # ---- commands, as payloads ----

    def define_payload(self):
        """DefineTask2 (56): {u32 taskID << 1, u64 length, u32 task root
        page}."""
        return struct.pack("<IQI", self.tid << 1, self.length, self.s.pfn(self.root))

    def list_payload(self):
        """SetObjectList (51): {u32 taskID, u32 list page, u32 length}."""
        return struct.pack("<III", self.tid, self.ol.list_va // PAGE, self.ol.length())

    def delete_payload(self):
        return struct.pack("<I", self.tid)

    def exec(self, segments, opcode=E3, ranges=None):
        cbs = ranges if ranges is not None else self.place(segments)
        return exec_payload(self.tid, cbs, self.resources(), opcode)

    def info_off(self):
        """The next 0x100 of the info page, filled with 0xEE by the guest
        first, so that a reply that is not written shows."""
        off = (self.info_next % 64) * 0x100
        self.info_next += 1
        self.fill(self.info_va + off, 0x40, FILL_REPLY)
        return off

    def slot_va(self, k):
        return self.out_va + k * SLOT

    def take_slot(self):
        k = self.next_slot
        self.next_slot = (k + 1) % self.nslots
        return k


# ---- programs: (segments, the slot's bytes after it) ----

def prog_blit(t, off, b):
    segs = [seg(BLIT, B.fill(t.out, off, 0x100, b) + B.copy_buffer_to_buffer(t.src, t.out, 0, off + 0x100, 0x100)
                + B.fill(t.out, off + 0x200, 0x200, b ^ 0xFF))]
    m = bytes([b]) * 0x100 + bytes(t.src_bytes[:0x100]) + bytes([b ^ 0xFF]) * 0x200
    return segs, m


def prog_double(t, off, b):
    segs = [seg(BLIT, B.fill(t.out, off + 0x200, 0x200, b)),
            seg(COMPUTE, C.set_pipeline(t.dbl) + C.set_buffers(0, [(t.out, off)])
                + C.dispatch_threads((64, 1, 1), (64, 1, 1))
                + C.set_pipeline(t.add1) + C.set_buffers(0, [(t.out, off + 0x100), (t.out, off)])
                + C.dispatch_threads((64, 1, 1), (64, 1, 1)))]
    m = struct.pack("<64I", *[2 * i for i in range(64)]) + struct.pack("<64I", *[2 * i + 1 for i in range(64)])
    return segs, m + bytes([b]) * 0x200


def _inc_model():
    """diff_util.Workload's bytes over a zeroed 0x400: three inc_k (c[0] +=
    1, c[c[0]] = 0x11 c[0]), a copy of 0x40 bytes to +0x100, two more, a
    fill of 0x10 bytes of 0x77 at +0x200, a copy of 0x40 to +0x300."""
    m = bytearray(SLOT)
    c = [0] * 16

    def inc():
        c[0] += 1
        if c[0] < 16:
            c[c[0]] = 0x11 * c[0]
    for _ in range(3):
        inc()
    m[0x100:0x140] = struct.pack("<16I", *c)
    inc()
    inc()
    m[0:0x40] = struct.pack("<16I", *c)
    m[0x200:0x210] = b"\x77" * 0x10
    m[0x300:0x340] = struct.pack("<16I", *c)
    return bytes(m)


INC_MODEL = _inc_model()


def prog_inc(t, off, b):
    steps = [(BLIT, B.fill(t.out, off, SLOT, 0))] + Workload(t.inc).steps(t.out, off)
    return merge(steps), INC_MODEL


def prog_heapfill(t, off, b):
    segs = [seg(BLIT, B.fill(t.hb, 0, 0x100, b) + B.fill(t.out, off, SLOT, b ^ 0x0F))]
    return segs, bytes([b ^ 0x0F]) * SLOT


def prog_heapcopy(t, off, b):
    segs = [seg(BLIT, B.copy_buffer_to_buffer(t.hb, t.out, 0, off, 0x100) + B.fill(t.out, off + 0x100, 0x300, b))]
    return segs, bytes(t.heap_bytes) + bytes([b]) * 0x300


def prog_icb(t, off, b):
    segs = [seg(BLIT, B.fill(t.out, off + 0x100, 0x300, b)),
            seg(COMPUTE, C.use_resources([t.icb_out]) + C.execute_icb(t.icb, 0, 1)),
            seg(BLIT, B.copy_buffer_to_buffer(t.icb_out, t.out, 0, off, 0x100))]
    return segs, struct.pack("<64I", *[2 * i for i in range(64)]) + bytes([b]) * 0x300


def prog_render(t, off, b):
    ci = b % len(COLOURS)
    segs = [seg(BLIT, B.fill(t.out, off + 0x40, SLOT - 0x40, b)),
            seg(RENDER, render_pass(t.tex, clear=COLOURS[ci])),
            seg(BLIT, B.copy_texture_to_buffer(t.tex, t.out, (0, 0, 0), (4, 4, 1), off, 16, 64))]
    px = bytes(round(v * 255) for v in COLOURS[ci])
    return segs, px * 16 + bytes([b]) * (SLOT - 0x40)


def prog_info(t, off, b):
    """A fill of the slot, then an info segment: 451 (heap texture size
    and align of a 256x256 RGBA8Unorm texture, v1 descriptor) into the info
    page, written by the CPU at decode time (docs/testing.md 6.17). The
    service only: in-process the reply aborts the host (7.4)."""
    ro = t.info_off()
    segs = [seg(BLIT, B.fill(t.out, off, SLOT, b)),
            seg(INFO, info_heap_size(t.reply, ro, fmt=70, w=256, h=256, usage=1, options=0x20))]
    t._info_reply = ro
    return segs, bytes([b]) * SLOT


PROGRAMS = {"blit": prog_blit, "double": prog_double, "inc": prog_inc, "heapfill": prog_heapfill,
            "heapcopy": prog_heapcopy, "icb": prog_icb, "render": prog_render, "info": prog_info}


def programs_for(t, iso=False):
    out = ["blit"]
    if "compute" in t.profile:
        out += ["double", "inc"]
        if iso:
            out.append("info")
    if "heap" in t.profile:
        out.append("heapfill")
        if t.heap_bytes is not None:
            out.append("heapcopy")
    if "icb" in t.profile:
        out.append("icb")
    if "render" in t.profile:
        out.append("render")
    return out


# ---- the walker ----

class Node:
    def __init__(self, fifo, opcode, payload, label, task=None):
        self.fifo, self.opcode, self.payload, self.label, self.task = fifo, opcode, payload, label, task
        self.bars = []
        self.stamp = None
        self.fault = False


class Walk:
    """One program: the scenario and the model, built step by step from the
    seed."""

    def __init__(self, name, doc, seed, steps, setups, ntasks, children, ending, big=False):
        self.r = Rng(seed * 0x9E3779B1 ^ 0x5E5E5E5E)
        self.seed, self.steps, self.setups, self.ending, self.big = seed, steps, setups, ending, big
        ram = (128 << 20) if big else (64 << 20)
        self.s = s = Scenario(name, setups=setups, host=True, watchdog_s=110 if big else 60,
                              ram=[(RAM_BASE, ram)], doc=doc)
        s.bring_up(fifo_pages=4 if big else 2)
        self.kids = [child(s, fid) for fid in range(1, children + 1)]
        self.fifos = [s.fifo] + self.kids
        self.model_page = s.alloc_page()
        self.first_reply = {}             # kind -> gpa of the first reply of that kind
        self.tasks = {}
        self.ntasks = ntasks
        self.batch_no = 0
        # Task 1 is always the full profile (the endings use it); the rest
        # draw theirs.
        for tid in range(1, ntasks + 1):
            if tid == 1:
                prof = ("compute", "icb", "heap", "render", "ro")
            elif big and tid > 6:
                prof = ()
            else:
                prof = self.r.sample(("compute", "icb", "heap", "render", "ro"), self.r.below(6))
                if "icb" in prof and "compute" not in prof:
                    prof = list(prof) + ["compute"]
            self.tasks[tid] = SvcTask(s, tid, prof, out_pages=1 if big and tid > 6 else 2)

    # ---- helpers ----

    def live(self):
        return [t for t in self.tasks.values() if t.live and not t.dead]

    def label(self, text):
        return "b%d: %s" % (self.batch_no, text)

    def compare(self, t, k, label):
        """The slot's model bytes into the model page, then cmp."""
        s = self.s
        emit_bytes(s, s.pa(self.model_page), t.slots[k])
        s.cmp(t.gpa(t.slot_va(k)), s.pa(self.model_page), SLOT, label=label)

    def reply(self, t, kind, off, n, label):
        """An info reply: the first of its kind is dumped (a host value),
        every later one compared with it."""
        gpa = t.gpa(t.info_va + off)
        if kind not in self.first_reply:
            self.first_reply[kind] = gpa
            self.s.dump(gpa, n, label=label + ": the first reply")
        else:
            self.s.cmp(gpa, self.first_reply[kind], n, label=label + " == the first reply")

    # ---- one batch ----

    def batch(self, lifecycle=True, faults=True, fifos=None, extra=None):
        """Builds, rings and checks one batch. extra(nodes, add) may add
        nodes (the endings). Returns the nodes."""
        r, s = self.r, self.s
        self.batch_no += 1
        fifos = fifos or self.fifos
        nodes = []
        chain_last = {}      # tid -> the task's last node this batch
        touched = []         # (task, slot)
        busy = set()
        posts = []           # checks after the batch

        def add(node, after=None):
            prev = after if after is not None else (chain_last.get(node.task.tid) if node.task else None)
            if prev is not None and prev.fifo is not node.fifo:
                node.bars.append((prev.fifo.stamp_index, prev))
            nodes.append(node)
            if node.task is not None:
                chain_last[node.task.tid] = node
            return node

        live = self.live()
        free = [t for t in self.tasks.values() if not t.live]
        # Lifecycle: a new task, a task deleted.
        deleted = None
        defined = None
        if lifecycle and free and r.chance(0.35):
            t = r.choice(free)
            f = r.choice(fifos)
            defined = add(Node(f, abi.OP_DEFINE_TASK2, t.define_payload(), "56 task %d" % t.tid, t))
            add(Node(f if r.chance(0.6) else r.choice(fifos), abi.OP_SET_OBJECT_LIST, t.list_payload(),
                     "51 task %d" % t.tid, t))
            t.live = True
            if "heap" in t.profile:
                t.heap_bytes = None
            busy.add(t.tid)
            if r.chance(0.6):
                self._program(t, r.choice(fifos), add, touched, posts)
        if lifecycle and len(live) > 2 and r.chance(0.3):
            cands = [t for t in live if t.tid != 1 and t.tid not in busy]
            if cands:
                deleted = r.choice(cands)
                # After the batch's define where a setup has no memory map:
                # see the module's docstring.
                add(Node(r.choice(fifos), abi.OP_DELETE_TASK, deleted.delete_payload(), "32 task %d" % deleted.tid,
                         deleted), after=defined if self.setups != ISO else None)
                busy.add(deleted.tid)
        # Programs, churn and info.
        users = [t for t in live if t.tid not in busy]
        k = min(len(users), 1 + r.below(max(1, min(len(users), 2 * len(fifos) + 2))))
        if self.big:
            k = min(len(users), 20 + r.below(60))
        for t in r.sample(users, k):
            if r.chance(0.2):
                self._churn(t, r.choice(fifos), add, posts)
            self._program(t, r.choice(fifos), add, touched, posts)
            if "compute" in t.profile and r.chance(0.15):
                self._info(t, r.choice(fifos), add, posts)
            if r.chance(0.2):
                self._program(t, r.choice(fifos), add, touched, posts)
        # One fault, last on its FIFO.
        fault = None
        if faults and r.chance(0.3):
            kinds = [k_ for k_ in FAULTS if k_ != "deleted" or deleted is not None]
            kind = r.choice(kinds)
            f = r.choice(fifos)
            if kind == "deleted":
                fault = Node(f, E3, deleted.exec(prog_blit(deleted, 0, 0x33)[0]),
                             "exec of task %d, deleted by this batch" % deleted.tid, None)
                add(fault, after=chain_last[deleted.tid])
            elif kind == "opcode":
                fault = add(Node(f, 99, b"", "unknown opcode 99", None))
            else:
                t = r.choice(users) if users else r.choice(self.live())
                fault = self._fault(t, kind, f, add, touched)
            fault.fault = True
        if extra:
            extra(nodes, add, touched, posts)
        # Write in creation order: each node after the first on its FIFO
        # waits on its FIFO-previous node.
        last = {}
        for n in nodes:
            bars = [(i, p.stamp) for i, p in n.bars]
            prev = last.get(n.fifo.fid)
            if prev is not None:
                bars.insert(0, (n.fifo.stamp_index, prev.stamp))
            n.stamp = n.fifo.cmd(n.opcode, n.payload, barriers=bars)
            last[n.fifo.fid] = n
        used = [f for f in fifos if f.fid in last]
        for f in r.sample(used, len(used)):
            f.doorbell()
        faulted = [n for n in nodes if n.fault]
        for f in used:
            ok = [n for n in nodes if n.fifo is f and not n.fault]
            if ok:
                f.wait_stamp(ok[-1].stamp, label=self.label("FIFO %d: %s, stamp %d" % (f.fid, ok[-1].label,
                                                                                       ok[-1].stamp)))
        for n in faulted:
            self._recover(n)
        if deleted is not None:
            deleted.live = False
        for t, k_ in touched:
            self.compare(t, k_, self.label("task %d slot %d" % (t.tid, k_)))
        for p in posts:
            p()
        return nodes

    def _program(self, t, f, add, touched, posts=None, name=None, opcode=None):
        r = self.r
        name = name or r.choice(programs_for(t, self.setups == ISO and posts is not None))
        k = t.take_slot()
        b = r.below(256)
        segs, m = PROGRAMS[name](t, t.slot_va(k) - t.out_va, b)
        if name == "heapfill":
            t.heap_bytes = bytes([b]) * 0x100
        op = opcode or (E2 if r.chance(0.25) else E3)
        n = add(Node(f, op, t.exec(segs, op), "task %d %s -> slot %d" % (t.tid, name, k), t))
        t.slots[k] = m
        touched.append((t, k))
        if name == "info":
            ro = t._info_reply
            posts.append(lambda: self.reply(t, "451", ro, 0x10, self.label("task %d 451" % t.tid)))
        return n

    def _churn(self, t, f, add, posts):
        r = self.r
        kinds = ["del_src", "del_out", "discard_out", "sync_out", "rewrite_src"]
        if "compute" in t.profile:
            kinds += ["del_pso", "del_function"]
        kind = r.choice(kinds)
        if kind == "rewrite_src":
            # The guest rewrites the source's bytes: a shared buffer is the
            # guest pages in every setup.
            t.src_bytes = bytearray(pattern(SRC_LEN, 3 + r.below(200), r.below(256)))
            t.write(t.src_va, bytes(t.src_bytes))
            return
        if kind == "del_src":
            op, p, lab = abi.OP_DELETE_RESOURCE, struct.pack("<II", t.tid, t.src), "37 src"
        elif kind == "del_out":
            op, p, lab = abi.OP_DELETE_RESOURCE, struct.pack("<II", t.tid, t.out), "37 out"
        elif kind == "discard_out":
            op, p, lab = abi.OP_DISCARD_RESOURCES, struct.pack("<III", t.tid, 1, t.out), "63 out"
        elif kind == "sync_out":
            op, p, lab = abi.OP_SYNCHRONIZE_RESOURCES, struct.pack("<III", t.tid, 1, t.out), "53 out"
        elif kind == "del_pso":
            op, p, lab = abi.OP_DELETE_OBJECT, struct.pack("<IIII", t.tid, DEL_COMPUTE_PSO, 12, t.dbl), "40 pipeline"
        else:
            op, p, lab = abi.OP_DELETE_OBJECT, struct.pack("<IIII", t.tid, DEL_FUNCTION, 12, t.fn_dbl), "40 function"
        add(Node(f, op, p, "task %d %s" % (t.tid, lab), t))

    def _info(self, t, f, add, posts):
        """59 of double_gid's pipeline (4 keys, 32 bytes) or 64 (16 bytes),
        into the info page, at an offset the guest filled with 0xEE."""
        r = self.r
        off = t.info_off()
        if r.chance(0.5):
            p = struct.pack("<IIIIQ", t.tid, t.dbl, 5, 4, t.info_va + off)
            add(Node(f, abi.OP_GET_COMPUTE_INFO, p, "task %d 59" % t.tid, t))
            posts.append(lambda: self.reply(t, "59", off, 0x20, self.label("task %d 59" % t.tid)))
        else:
            p = heap_payload(t.tid, t.info_va + off, 16, heap_ser())
            add(Node(f, abi.OP_HEAP_TEXTURE_SIZE_AND_ALIGN, p, "task %d 64" % t.tid, t))
            posts.append(lambda: self.reply(t, "64", off, 0x10, self.label("task %d 64" % t.tid)))

    def _fault(self, t, kind, f, add, touched):
        """A faulting exec of t. Its fill of the slot is never committed:
        nothing of a faulted exec reaches the GPU (docs/testing.md 6.12)."""
        k = t.take_slot()
        off = t.slot_va(k) - t.out_va
        fill = B.fill(t.out, off, SLOT, 0xA5)
        if kind == "decode":
            segs = [seg(BLIT, fill + rec(399, bytes(8)))]
        elif kind == "overlong":
            segs = [seg(BLIT, fill + rec(306, struct.pack("<IQQB3x", t.out, off, 0x10, 0x5A), length=0x400))]
        else:
            segs = [seg(BLIT, fill + B.fill(len(t.ol.entries) + 40, 0, 0x10, 0x5A))]
        touched.append((t, k))
        return add(Node(f, E3, t.exec(segs), "task %d %s fault, slot %d" % (t.tid, kind, k), t))

    def _recover(self, n):
        """Waits for the fault, reads its state, resumes, and waits for the
        stamp the resume publishes."""
        s, f = self.s, n.fifo
        lab = self.label(n.label)
        if f.fid:
            s.wait(gpa=f.node_gpa(NODE_FAULT), ne=HEALTHY, label=lab + ": child %d faults" % f.fid)
        s.wait_fault(label=lab + ": 0x102c")
        fault_state(s, f, lab)
        resume(s, f)
        f.wait_stamp(n.stamp, label=lab + ": resumed, stamp %d" % n.stamp)

    # ---- divergences (all-setup walks) ----

    def divergence(self):
        r, s = self.r, self.s
        self.batch_no += 1
        live = self.live()
        t = r.choice(live)
        kinds = ["map_misaligned", "unmap_past_end"]
        if "compute" in t.profile:
            kinds.append("compute_info_count_0")
        kind = r.choice(kinds)
        # Work on the child FIFOs, in flight around the root's command.
        others = [o for o in live if o is not t]
        ring = []
        touched = []
        for o in r.sample(others, min(len(others), len(self.kids))):
            f = self.kids[len(ring) % len(self.kids)]
            k = o.take_slot()
            b = r.below(256)
            segs, m = prog_blit(o, o.slot_va(k) - o.out_va, b)
            o.slots[k] = m
            touched.append((o, k))
            ring.append((f, f.cmd(E3, o.exec(segs))))
            f.doorbell()
            if len(ring) == len(self.kids):
                break
        if kind == "map_misaligned":
            either(s, abi.OP_MAP_MEMORY2, struct.pack("<IQQ", t.tid, 0x100, PAGE),
                   self.label("57 task %d at 0x100" % t.tid))
        elif kind == "unmap_past_end":
            either(s, abi.OP_UNMAP_MEMORY, struct.pack("<IQQ", t.tid, t.length - PAGE, 2 * PAGE),
                   self.label("34 task %d past its end" % t.tid))
        else:
            off = t.info_off()
            either(s, abi.OP_GET_COMPUTE_INFO, struct.pack("<IIIIQ", t.tid, t.dbl, 5, 0, t.info_va + off),
                   self.label("59 task %d count 0" % t.tid))
            t.dump(t.info_va + off, 0x20, self.label("59 count 0: the reply area"))
        for f, st in ring:
            f.wait_stamp(st, label=self.label("FIFO %d beside the divergence" % f.fid))
        for o, k in touched:
            self.compare(o, k, self.label("task %d slot %d" % (o.tid, k)))

    def ro_probe(self):
        """The read-only buffer filled by the GPU, dumped: in-process the
        fill lands, the service drops it (docs/testing.md 6.10)."""
        cands = [t for t in self.live() if "ro" in t.profile]
        if not cands:
            return
        t = self.r.choice(cands)
        self.batch_no += 1
        b = self.r.below(256)
        t.run([seg(BLIT, B.fill(t.ro, 0, 0x40, b))], label=self.label("task %d fills its read-only buffer" % t.tid))
        t.dump(t.ro_va, 0x40, self.label("task %d's read-only buffer" % t.tid))

    # ---- the program ----

    def start(self):
        """Defines the first tasks, spread across the FIFOs."""
        r, s = self.r, self.s
        n0 = self.ntasks if self.big else max(2, (self.ntasks + 1) // 2)
        first = [self.tasks[tid] for tid in range(1, n0 + 1)]
        self.batch_no += 1
        last = {}
        prev_define = None
        for t in first:
            f = r.choice(self.fifos)
            for op, p, lab in ((abi.OP_DEFINE_TASK2, t.define_payload(), "56"),
                               (abi.OP_SET_OBJECT_LIST, t.list_payload(), "51")):
                prev = last.get(f.fid)
                bars = [(f.stamp_index, prev)] if prev is not None else []
                if op == abi.OP_DEFINE_TASK2 and prev_define is not None and prev_define[0] is not f:
                    # Defines in one order across the FIFOs: the blocks
                    # setup's create_task numbers the host's tasks in the
                    # order it is called (runner/pgrun.m).
                    bars.append((prev_define[0].stamp_index, prev_define[1]))
                last[f.fid] = f.cmd(op, p, barriers=bars)
                if op == abi.OP_DEFINE_TASK2:
                    prev_define = (f, last[f.fid])
            t.live = True
        for f in self.fifos:
            if f.fid in last:
                f.doorbell()
        for f in self.fifos:
            if f.fid in last:
                f.wait_stamp(last[f.fid], timeout_ms=60000,
                             label=self.label("FIFO %d: tasks defined, stamp %d" % (f.fid, last[f.fid])))
        s.fault_state(self.label("tasks defined"))
        s.dump(s.pa(s.root_page), 4 * (n0 + 1), self.label("task words"))
        s.events(self.label("tasks defined"), sort_blocks=True, blocks_digest=True)

    def run(self):
        r, s = self.r, self.s
        self.start()
        for i in range(self.steps):
            x = r.below(100)
            if self.setups != ISO and x < 10 and len(self.live()) > 1 and self.kids:
                self.divergence()
            elif x < 15:
                self.ro_probe()
            else:
                self.batch()
            if r.chance(0.3) or i == self.steps - 1:
                s.r(abi.R_FAULT_MASK, label=self.label("0x102c"))
                s.events(self.label("events"), sort_blocks=True, blocks_digest=True)
        for t in self.tasks.values():
            if "heap" in t.profile and t.live:
                t.dump(t.heap_va, 0x100, self.label("task %d's heap pages" % t.tid))
        if self.ending:
            self.end()
        # Everything deleted, in order, so teardown has nothing left.
        self.batch_no += 1
        for t in sorted(self.tasks.values(), key=lambda t: t.tid):
            if t.live:
                s.submit(abi.OP_DELETE_TASK, t.delete_payload(), label=self.label("32 task %d" % t.tid))
                t.live = False
        s.dump(s.pa(s.root_page), 4 * (self.ntasks + 1), self.label("task words at the end"))
        s.events(self.label("end"), sort_blocks=True, blocks_digest=True)
        return s

    # ---- the endings ----

    def death_payload(self, t, kind):
        """The killing command of task t (docs/testing.md 7.4)."""
        if kind == "past_end":
            return E3, t.exec(None, ranges=[(t.npages * PAGE, 0x40)])
        if kind == "zero_length":
            return E3, t.exec(None, ranges=[(4 * PAGE, 0)])
        if kind == "open_encoder":
            return E3, t.exec([seg(BLIT, B.fill(t.out, 0, 0x10, 0x11), cont_next=1)])
        if kind == "compute_info_0":
            off = t._death_reply = t.info_off()
            return abi.OP_GET_COMPUTE_INFO, struct.pack("<IIIIQ", t.tid, 0, 5, 4, t.info_va + off)
        if kind == "info_reply":
            off = t._death_reply = t.info_off()
            return E3, t.exec([seg(INFO, info_heap_size(t.reply, off, fmt=70, w=256, h=256, usage=1,
                                                        options=0x20))])
        raise ValueError(kind)

    def end(self):
        if self.setups == ISO:
            # Task 1 (every object) takes 59 on pipeline 0 when it is one.
            kinds = sorted(self.ending, key=lambda k: k != "compute_info_0")
        else:
            kinds = (self.ending,)
        s = self.s
        # Enough tasks for every death and one beside them.
        while len(self.live()) < len(kinds) + 1:
            t = min((t for t in self.tasks.values() if not t.live), key=lambda t: t.tid)
            s.submit(abi.OP_DEFINE_TASK2, t.define_payload(), label=self.label("56 task %d" % t.tid))
            s.submit(abi.OP_SET_OBJECT_LIST, t.list_payload(), label=self.label("51 task %d" % t.tid))
            t.live = True
            if "heap" in t.profile:
                t.heap_bytes = None
        live = self.live()
        victims = [self.tasks[1]] + [t for t in live if t.tid != 1][:len(kinds) - 1]
        assert self.tasks[1].live and len(victims) == len(kinds)
        s = self.s
        self.batch_no += 1
        # Each death on its own FIFO, the root's last; the others' work
        # beside them on FIFOs that carry no death.
        dfifos = [s.fifo] + self.kids[:len(kinds) - 1]
        others = [t for t in live if t not in victims]
        wfifos = [f for f in self.kids if f not in dfifos] or []
        ring = []
        touched = []
        for i, o in enumerate(others):
            if not wfifos:
                break
            f = wfifos[i % len(wfifos)]
            k = o.take_slot()
            segs, m = prog_blit(o, o.slot_va(k) - o.out_va, self.r.below(256))
            o.slots[k] = m
            touched.append((o, k))
            prev = [st for ff, st in ring if ff is f]
            ring.append((f, f.cmd(E3, o.exec(segs), barriers=[(f.stamp_index, prev[-1])] if prev else [])))
        deaths = []
        for t, kind, f in zip(victims, kinds, dfifos):
            op, p = self.death_payload(t, kind)
            if self.setups != ISO or kind in DEATHS_METAL:
                # 256 NOPs ahead of it keep the root FIFO busy past its
                # doorbell's register write, so a host death lands in the
                # wait that follows every run (exec_objects.submit_last).
                for _ in range(256):
                    f.cmd(abi.OP_NOP)
            deaths.append((t, kind, f, f.cmd(op, p)))
        for f in {id(f): f for f, _ in ring}.values():
            f.doorbell()
        # A death inside Metal ends the process: its doorbell last, its wait first, so the process
        # dies in that wait whatever the other deaths' timing.
        deaths_last = sorted(reversed(deaths), key=lambda d: d[1] in DEATHS_METAL)
        for t, kind, f, st in deaths_last:
            f.doorbell()
        # The deaths first: in-process the host dies in the first wait.
        if self.setups == ISO:
            for t, kind, f, st in sorted(deaths, key=lambda d: d[1] not in DEATHS_METAL):
                if f.fid:
                    s.wait(gpa=f.node_gpa(NODE_FAULT), ne=HEALTHY,
                           label=self.label("task %d %s: child %d faults" % (t.tid, kind, f.fid)))
                else:
                    s.wait(mmio=abi.R_FAULT_OFFSET, mmio_ne=HEALTHY,
                           label=self.label("task %d %s: the root FIFO faults" % (t.tid, kind)))
            s.wait_fault(label=self.label("the deaths: 0x102c"))
            for t, kind, f, st in deaths:
                fault_state(s, f, self.label("task %d %s" % (t.tid, kind)))
            for t, kind, f, st in deaths:
                resume(s, f)
                f.wait_stamp(st, label=self.label("task %d %s: resumed, stamp %d" % (t.tid, kind, st)))
        else:
            t, kind, f, st = deaths[0]
            if kind in ("compute_info_0", "info_reply"):
                # In-process 59 on pipeline 0 writes zero values and 451
                # aborts the host; the service dies on the first and writes
                # the second: the observation says which.
                s.wait_any([dict(gpa=f.stamp_gpa(), ge=st), dict(mmio=abi.R_FAULT_MASK, mmio_eq=1)],
                           label=self.label("task 1 %s: %s" % (kind, EITHER))
                           + (DIES if kind == "info_reply" else ""))
                s.fault_state(self.label("task 1 %s" % kind))
                s.resume_root()
                f.wait_stamp(st, label=self.label("task 1 %s: stamp %d" % (kind, st)))
                t.dump(t.info_va + t._death_reply, 0x20, self.label("task 1 %s: the reply area" % kind))
            else:
                s.wait_fault(label=self.label("task 1 %s: the host dies or the service" % kind) + DIES)
                fault_state(s, f, self.label("task 1 %s" % kind))
                resume(s, f)
                f.wait_stamp(st, label=self.label("task 1 %s: resumed, stamp %d" % (kind, st)))
        for f, st in ring:
            f.wait_stamp(st, label=self.label("FIFO %d beside the deaths, stamp %d" % (f.fid, st)))
        for t in victims:
            t.dead = True
        for o, k in touched:
            self.compare(o, k, self.label("task %d slot %d, beside the deaths" % (o.tid, k)))
        s.events(self.label("the deaths"), sort_blocks=True, blocks_digest=True)
        # After: every other task runs a batch; each victim's next exec
        # completes or faults, then it is deleted, defined again and run.
        survivors = [t for t in self.live() if t not in victims]
        if survivors:
            self.batch(lifecycle=False, faults=False, fifos=self.kids or None)
        self.batch_no += 1
        for t in victims:
            t.dead = False
            k = t.take_slot()
            segs, m = prog_blit(t, t.slot_va(k) - t.out_va, 0x6D)
            if self.setups == ISO:
                t.run(segs, fault=True, label=self.label("task %d after its death" % t.tid))
            else:
                either(s, E3, t.exec(segs), self.label("task %d after the ending" % t.tid))
                t.slots[k] = None
            t.dump(t.slot_va(k), 0x20, self.label("task %d slot %d after the ending" % (t.tid, k)))
            s.submit(abi.OP_DELETE_TASK, t.delete_payload(), label=self.label("32 task %d" % t.tid))
            s.submit(abi.OP_DEFINE_TASK2, t.define_payload(), label=self.label("56 task %d again" % t.tid))
            s.submit(abi.OP_SET_OBJECT_LIST, t.list_payload(), label=self.label("51 task %d again" % t.tid))
            if "heap" in t.profile:
                t.heap_bytes = None
            k = t.take_slot()
            segs, m = prog_blit(t, t.slot_va(k) - t.out_va, 0x7E)
            t.run(segs, label=self.label("task %d again: blit" % t.tid))
            t.slots[k] = m
            self.compare(t, k, self.label("task %d again: slot %d" % (t.tid, k)))
        s.fault_state(self.label("after the ending"))


def build(name, doc, seed, steps, setups=SETUPS, ntasks=6, children=4, ending=None, big=False):
    return Walk(name, doc, seed, steps, setups, ntasks, children, ending, big).run()


# ---- reading the expected files back ----

def check(pattern_=None):
    """Every cmp equal; every wait ok but the "either" waits (whose "which"
    is the setup's) and [dies]; every scenario completed but those whose
    ending kills the host in-process."""
    import glob
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "expected",
                        "interplay_servicewalk")
    bad = 0
    for path in sorted(glob.glob(os.path.join(base, "*", "*.jsonl"))):
        if pattern_ and pattern_ not in path:
            continue
        for line in open(path):
            o = json.loads(line)
            ev = o.get("ev")
            if ev == "cmp" and not o.get("equal"):
                print("%s: cmp %s differs" % (path, o.get("label")))
                bad += 1
            if ev == "wait" and not o.get("ok") and EITHER not in o.get("label", "") \
                    and DIES not in o.get("label", ""):
                print("%s: wait %s failed" % (path, o.get("label")))
                bad += 1
            if ev == "outcome" and o["kind"] not in ("completed", "died"):
                print("%s: outcome %s" % (path, o))
                bad += 1
    print("checked; %d problems" % bad)
    return bad


if __name__ == "__main__":
    import sys
    sys.exit(1 if check(sys.argv[1] if len(sys.argv) > 1 else None) else 0)
