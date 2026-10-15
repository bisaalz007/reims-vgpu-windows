"""Marathons (scenarios/interplay_marathon.py): one seeded walker that drives
every subsystem at once in one device session, step by step, so that each
subsystem's state changes while the others' work is in flight.

The walker composes the generators that walk one subsystem each:

  - pgtest/randdisplay.py: the display walk, whole: its Walk is the
    Marathon's base class, so every display step is one of its actions
    (connects, acks, presents of backings and mapper surfaces with gamma
    tables and colour matrices, glyphs, cursors, sleep, ICC, hot plugs,
    mode lists, scribbles, pause with host changes and commands rung while
    paused, suspend onto a fresh device, reset and stop, EFI), and its
    model predicts every display observation as it does alone;
  - pgtest/dagwalk.py and pgtest/randmix.py: GPU work as nodes of a
    barrier DAG over the root FIFO and up to eight child FIFOs and one to
    three tasks, each node an order witness (randmix.dag_segments: its
    chain's counter, add_one over its first predecessor, its predecessors'
    first words, and a scratch-buffer fill), compared with a control;
  - pgtest/randfault.py: faulting commands (decode errors, an exec of a
    task that does not exist, unknown and deprecated opcodes) on a child
    FIFO, with work on other FIFOs blocked on the faulted stamp, recovered
    later by a resume (0x1028) or by deleting the child;
  - pgtest/lifetime_util.py: the scratch buffer deleted (37, 40), retyped
    (its object-list entry rewritten in place) and made again, pipelines
    and functions deleted, a task deleted and defined again;
  - pgtest/memlive_util.py: a scratch page moved the guest's way (34, the
    leaf pointed at a fresh page, 57); command-buffer pages of held work
    moved to fresh pages, leaf by leaf or, in a task two levels deep, by
    swapping the whole leaf table, the old pages poisoned.

Unlike those, nothing here is a phase. Work is written in batches whose
nodes wait on the gate (child FIFO 20) or on a faulted stamp, directly or
through their barriers, and stay in flight across any number of later
steps: other batches (which stack up behind held work on its FIFOs, or
run beside it), display steps, faults, recoveries, lifetime and memory
changes, child FIFOs deleted and defined again, pause (with memory
changes made while paused), and suspend onto a fresh device. A release
step opens the gate; a recovery step resumes or deletes a faulted child.
Reset and stop are taken only with nothing in flight.

The model (one per program, kept by the walker):
  - GPU slots: before each batch runs, a control task runs the batch's
    nodes on the root FIFO, in creation order, into the control region
    (dagwalk's differential control, per batch). Every slot of a node that
    ran is compared with its control slot when it has run, at checks, and
    at the end; each chain's counter once its chain has nothing in
    flight. A faulting node's slot is compared with its control slot,
    which nothing writes: nothing of a faulted exec is committed.
  - Locks: every node carries the set of things it waits on ("gate", a
    faulted node), the union of its barrier targets' locks and its
    FIFO-previous node's. A node with none completes in its step (the
    walker waits for its stamp); one with some is waited for, 200 ms, to
    see that it does not complete ([held]), and completes in the step that
    removes its last lock.
  - Scratch buffers: each task's scratch entry names VA page A or B; each
    VA page's leaf names a RAM page; the object is made when the first exec
    that lists it runs, over the RAM page the entry's leaf names then, and
    stays bound to that page until a delete, a task delete, a suspend or a
    teardown forgets it (lifetime_util's and memlive_util's
    observations). Each fill lands in the bound page; the model's bytes
    of every page in play are compared with the guest's.
  - Command buffers: a held exec walks its command buffer when it runs
    (memlive_util): after its pages move and the old ones are poisoned,
    its slot still equals the control.
  - Interrupts and 0x1018: one interrupt and one stamp bit per published
    stamp (each command on a FIFO is published alone: a node gets a
    barrier on its FIFO-previous node when that one is in flight and not
    an ancestor, dagwalk's rule) and one per fault; the display model's
    counts are kept with them, so its predictions hold across GPU steps.
  - Fault registers: 0x102c waited for at the model's value (1 while any
    child is faulted), 0x103c and every live child's node fault offset
    read, labelled with the model's state.

Invariants the walker keeps (the composed generators give the evidence):
  - Nothing in flight is on the root FIFO: the display, the lifetime and
    memory commands and the control use it, and every root command is
    waited for in its step.
  - A faulted child gets nothing more until its recovery; faults are on
    idle children only, at most two at once; pause, suspend, reset and
    stop are taken only with no fault outstanding (a pause over a fault
    latches it, randmix 7.1), and reset and stop only with nothing in
    flight. rvg_device_will_suspend then unpause is never taken (7.3, 7.4).
  - While paused nothing is rung but what the display walk rings: the
    GPU side only rewrites guest memory (command-buffer pages, an
    object-list entry).
  - Deleting child n zeroes node n+1's page list; the guest writes it
    back at once. A child is deleted only idle. Barriers name nodes of the
    current incarnation of the device (not before a reset) on a FIFO that
    is defined.
  - One display is active a step, and GPU steps make no display callback,
    so the display walk's waits by running totals stay exact.
  - The blocks setup's calls are reported sorted where FIFOs run at once
    (GPU steps, also digested) and during lifecycle calls: a pause over
    GPU tasks unmaps their pages (as each mapping is dropped,
    reims-vgpu-core's memory::mapping::Mapping) in an order only timing
    decides.

Programs that end in a death end with the display walk's death actions
(docs/testing.md 7.3, 7.4), with work still held on the gate.

Nothing here is imported from the library's code.
"""
import json
import struct

from . import abi
from .abi import PAGE
from .compute_objects import compute_blob, compute_pipeline_record, function_record
from .dagwalk import ADD, CNT, INC, MAX_CHAINS
from .diff_util import exec_payload
from .fifo import ChildFIFO
from .fifo_util import HEALTHY, NODE_FAULT, NODE_PAGE_LIST
from .lifetime_util import delete_object, delete_resource, rewrite_entry
from .memlive_util import leaf, set_leaf, write_table
from .randdisplay import Death, Walk, check_obs
from .randfault import UNKNOWN_OPS
from .randgen import Rng
from .randmix import CONTROL_GROUP, DECODE, FILLS, HELD, MAX_PREDS, SCR_LEN, SLOT, SLOT_BASE, dag_segments, slot_off
from .resources_build import b_fill_buf
from .session import DEL_COMPUTE_PSO, DEL_FUNCTION, DEL_RESOURCE
from .task import GPUTask, buffer_record, rec, seg
from .tasks_lib import XGPUTask

E3, E2 = abi.OP_EXEC_INDIRECT3, abi.OP_EXEC_INDIRECT2
GATE = 20
HOLD_MS = 200
# A task's VA pages: 0 the object list, 1-3 records, 4-11 command buffers,
# 12 and 13 the kernels, 14 and 15 the scratch pages A and B, 16 the
# control's scratch; then SRC, the control region and the live region,
# the same guest pages in every task.
CMD_FIRST, CMD_END = 4, 12
SCR_VPN = {"A": 14, "B": 15, "C": 16}
PRIVATE = 17
REG_PAGES = 8
SHARED = 1 + 2 * REG_PAGES
MAX_SLOTS = (REG_PAGES * PAGE - SLOT_BASE) // SLOT
MARKERS = (0x21, 0x42, 0x63, 0x84, 0xA5, 0xC6, 0xE7)


class MarTask(XGPUTask):
    """randmix's MixTask without surfaces, with a larger command region and
    live region, and a depth of 1 or 2: inc_k's function and pipeline,
    add_one's, SRC, the control region, the live region, the scratch buffer
    (entry on page A), the control's scratch (page C), in that order in
    every task, so a reference means the same in each."""

    def __init__(self, s, tid, shared, depth=1):
        leaves = {v: s.alloc_page() for v in range(PRIVATE)}
        leaves.update({PRIVATE + i: pg for i, pg in enumerate(shared)})
        super().__init__(s, tid=tid, pages=PRIVATE + SHARED, leaves=leaves, depth=depth)
        self._blob = 12 * PAGE
        self.fn_inc, self.pso_inc = self._kernel(INC, 12)
        self.fn_add, self.pso_add = self._kernel(ADD, 13)
        base = PRIVATE * PAGE
        self.src, self.src_va = self.buffer(PAGE, va=base)
        self.reg_c, self.reg_c_va = self.buffer(REG_PAGES * PAGE, va=base + PAGE)
        self.reg_l, self.reg_l_va = self.buffer(REG_PAGES * PAGE, va=base + (1 + REG_PAGES) * PAGE)
        self.scr, _ = self.buffer(PAGE, va=SCR_VPN["A"] * PAGE)
        self.scr_c, _ = self.buffer(PAGE, va=SCR_VPN["C"] * PAGE)
        self.reset_cmdbufs()

    def _kernel(self, fixture, vpn):
        va = self.reserve(0x2000)
        assert va == vpn * PAGE
        n = self.s.mem_file(self.gpa(va), fixture)
        fn = self.ol.add(abi.T_FUNCTION, function_record(va, n))
        return fn, self.ol.add(abi.T_SERIALIZED, compute_pipeline_record(compute_blob(fn)))

    def cmdbuf(self, data):
        va = self._cmd
        self.write(va, data)
        self.cmdlog.append((va, bytes(data)))
        self._cmd = (va + len(data) + 0xFF) // 0x100 * 0x100
        assert self._cmd <= CMD_END * PAGE, "command region full"
        return (va, len(data))

    def room(self, segs):
        return CMD_END * PAGE - self._cmd >= sum((len(sg) + 0xFF) // 0x100 * 0x100 for sg in segs)

    def reset_cmdbufs(self):
        self._cmd = CMD_FIRST * PAGE
        self.cmdlog = []

    def used_cmd_pages(self):
        return list(range(CMD_FIRST, (self._cmd + PAGE - 1) // PAGE))


class Node:
    """One command: a dag node (randmix's order witness) or a faulting
    command. preds feed the node's data; after are barrier-only targets;
    gate a barrier on the gate's next stamp. locks: what it waits on;
    anc: the nodes in flight when it was written that finish before it
    runs."""

    def __init__(self, nid, kind, task, op=E3):
        self.id, self.kind, self.task, self.op = nid, kind, task, op
        self.fid = None
        self.preds, self.after = [], []
        self.gate = False
        self.bars = []
        self.slot = self.chain = self.fill = self.fault = None
        self.stamp = None
        self.locks = set()
        self.anc = set()
        self.done = False
        self.epoch = 0

    def desc(self):
        bars = ",".join("gate" if b == "gate" else "n%d" % b for b in self.bars)
        if self.kind == "dag":
            what = "t%d c%d s%d%s%s" % (self.task + 1, self.chain, self.slot,
                                       " fill %02x" % self.fill if self.fill is not None else "",
                                       " op55" if self.op == E2 else "")
        else:
            what = "FAULT %s/%d s%d" % (self.fault[0], self.fault[1], self.slot)
        return "n%d F%d %s%s" % (self.id, self.fid, what, " <" + bars if bars else "")


class _View:
    """What randmix.dag_segments reads of a program: .nodes[id], which
    here also reaches the batch being built."""

    class _Nodes:
        def __init__(self, m):
            self.m = m

        def __getitem__(self, i):
            return self.m.node(i)

    def __init__(self, m):
        self.nodes = self._Nodes(m)


class Marathon(Walk):
    """randdisplay's Walk, with GPU work, faults, lifetime and memory
    changes interleaved step by step (the module docstring)."""

    def __init__(self, seed, steps, doc, death=None):
        super().__init__(seed, steps, doc)
        self.version = abi.MAX_VERSION
        self.death = death
        self.name = self.s.name = "interplay_marathon_%d" % seed
        self.s._config["watchdog_s"] = 120 + steps // 2
        q = self.q = Rng(seed * 0x9E3779B1 + 0x6D61)
        self.nchildren = q.choice((2, 3, 4, 6, 8))
        self.ntasks = q.choice((1, 2, 2, 3))
        self.depths = [2 if q.chance(0.4) else 1 for _ in range(self.ntasks)]
        self.gate_p = q.choice((0.25, 0.4, 0.6))
        self.faults_on = q.chance(0.75)
        self.batch_max = q.choice((3, 4, 6))
        self.teardown_on = q.chance(0.5)
        self.nodes = []
        self.tails = []
        self.nslots = 0
        self.pend = {}
        self.faulted = {}
        self.fifos = {}
        self.defined = set(range(1, self.nchildren + 1))
        self.last_fill = {}
        self.epoch = 0
        self.hist = {}
        self._batch = []
        self._sort = False
        self.view = _View(self)

    # ---- the display model's interrupt and 0x1018 bookkeeping ----

    def sig(self, idx):
        """A stamp published on stamp slot idx: one interrupt, one bit of
        0x1018 (the stamp's signal)."""
        self.r1018 |= 1 << idx
        self.interrupt()

    def rsub(self, opcode, payload, label):
        """A root FIFO command, waited for."""
        st = self.s.submit(opcode, payload, label=label)
        self.predict(ok=True)
        self.sig(0)
        return st

    def gevents(self, label):
        """events after a GPU step: FIFOs ran at once, so the blocks
        setup's calls are sorted and digested; the display model's counts
        (every interrupt the step raised) predicted."""
        self.wait_callbacks()
        self.s.events(label, sort_blocks=True, blocks_digest=True)
        p = {"irqs": self.irqs}
        if self.cb:
            p["callbacks"] = list(self.cb)
        self.predict(**p)
        self.cb = []
        self._pend = 0
        self.irqs = 0
        self.regs(label)

    def events(self, label):
        """The display walk's events; during a lifecycle call with GPU
        tasks alive (pause, unpause, suspend, reset, stop, the end), the
        blocks setup's calls are reported sorted: the pause's unmaps of a
        task's pages come in an order only timing decides (measured,
        interplay_marathon_11)."""
        self.wait_callbacks()
        self.s.events(label, sort_blocks=self._sort)
        p = {"irqs": self.irqs}
        if self.cb:
            p["callbacks"] = list(self.cb)
        self.predict(**p)
        self.cb = []
        self._pend = 0
        self.irqs = 0

    def sorted_call(self, fn, *args):
        self._sort = True
        try:
            return fn(*args)
        finally:
            self._sort = False

    def a_pause(self):
        self.sorted_call(Walk.a_pause, self)

    def a_unpause(self):
        self.sorted_call(Walk.a_unpause, self)

    def a_suspend(self):
        self.sorted_call(Walk.a_suspend, self)

    def a_reset(self, how):
        self.sorted_call(Walk.a_reset, self, how)

    def final_check(self):
        self.sorted_call(Walk.final_check, self)

    def final_death(self):
        self.sorted_call(Walk.final_death, self)

    def model_cmp(self, gpa, expect, label):
        """The model's bytes into the guest's model page, then a cmp."""
        s = self.s
        s.mem(s.pa(self.mpage), expect)
        s.cmp(gpa, s.pa(self.mpage), len(expect), label=label + " against the model")
        self.predict(equal=True)

    # ---- set-up ----

    def build(self):
        s = self.s
        self.root_words = {}
        s.w(abi.R_VERSION, self.version)
        s.r(abi.R_VERSION, "version")
        self.predict(val=hex(self.version))
        s.r(abi.R_PORT_COUNT, "port count")
        self.predict(val=hex(self.nports))
        s.r(abi.R_PORT_TYPES, "port types")
        self.predict(val="0x0")
        self.t0 = GPUTask(s, tid=0, pages=160)
        self.seq_regions = []
        self.lay_task0()
        if self.surfaces_on:
            self.lay_surfaces()
        self.bring_up(first=True)
        if self.efi_on:
            self.efi_boot()
        self.gpu_setup()
        try:
            for i in range(self.steps):
                self.step_no = i
                self.m_step()
            if self.death:
                self.gpu_before_death()
                self.final_death()
            else:
                self.gpu_final()
                self.final_check()
        except Death:
            pass
        self.s.doc = self.doc.strip() + "\n\nThis program: " + self.summary() + " " + self.gpu_summary()
        return self.s

    def fifo(self, fid):
        return self.s.fifo if fid == 0 else self.fifos[fid]

    def define(self, fid, label=None):
        c = self.fifos.get(fid)
        if c is None:
            c = self.fifos[fid] = ChildFIFO(self.s, fid)
        c.written = 0
        c.setup_node()
        self.rsub(abi.OP_DEFINE_CHILD_FIFO, struct.pack("<I", fid), label or "define child %d" % fid)
        self.pend.setdefault(fid, [])
        return c

    def delete(self, fid, label):
        """49 on child fid, which zeroes node fid+1's pageListPage; the
        guest writes it back, so the next start of that child maps its
        ring."""
        s = self.s
        c = self.fifos[fid]
        self.rsub(abi.OP_DELETE_CHILD_FIFO, struct.pack("<I", fid), label)
        s.dump(c.node_gpa(0), abi.CHILD_NODE_SIZE, "%s: node %d" % (label, fid))
        # Node fid+1 without its read offset, which moves while that child runs.
        s.dump(c.node_gpa(0) + abi.CHILD_NODE_SIZE + 8, abi.CHILD_NODE_SIZE - 8, "%s: node %d from +8" % (label, fid + 1))
        nxt = self.fifos.get(fid + 1)
        if nxt is not None and ((fid + 1) in self.defined or fid + 1 == GATE):
            s.u32(nxt.node_gpa(NODE_PAGE_LIST), s.pfn(nxt.list_page))

    def define_task(self, t, label):
        payload = struct.pack("<IQI", t.tid << 1, t.length, self.s.pfn(t.root))
        self.rsub(abi.OP_DEFINE_TASK2, payload, "%s: define task %d" % (label, t.tid))
        t.ol.write()
        self.rsub(abi.OP_SET_OBJECT_LIST, struct.pack("<III", t.tid, t.ol.list_va // PAGE, t.ol.length()),
                  "%s: task %d's object list" % (label, t.tid))

    def gpu_setup(self):
        s = self.s
        self.s.mark("GPU set-up: children %s, the gate, tasks %s (depths %s), control task %d" % (
            sorted(self.defined), list(range(1, self.ntasks + 1)), self.depths, self.ntasks + 1))
        for fid in sorted(self.defined):
            self.define(fid)
        self.gate = self.define(GATE, "define the gate")
        self.shared = s.alloc_page(SHARED)
        self.mpage = s.alloc_page()
        self.tasks = [MarTask(s, tid + 1, [self.shared + i for i in range(SHARED)], self.depths[tid])
                      for tid in range(self.ntasks)]
        self.ctl = MarTask(s, self.ntasks + 1, [self.shared + i for i in range(SHARED)])
        t = self.tasks[0]
        t.write(t.src_va, struct.pack("<256I", *[0x1000 * (i // 16 + 1) + i for i in range(256)]))
        self.scr = []
        for t in self.tasks:
            pages = {pg: t.leaves[SCR_VPN[pg]] for pg in ("A", "B")}
            for p in pages.values():
                s.fill(s.pa(p), SCR_LEN, 0)
            self.scr.append(dict(entry="A", leaf=pages, cached=None,
                                 bytes={p: bytes(SCR_LEN) for p in pages.values()}))
        for t in self.tasks + [self.ctl]:
            self.define_task(t, "set-up")
        self.gevents("GPU set up")
        self.observe("GPU set up")

    def gpu_mark(self, text):
        return self.mark("[gpu] " + text)

    # ---- ancestry and locks ----

    def closure(self, targets):
        """For barrier targets: the nodes in flight that finish before, and
        the locks inherited."""
        anc, locks = set(), set()
        for b in targets:
            nb = self.node(b)
            if nb.done:
                continue
            pl = self.pend[nb.fid]
            for u in pl[:pl.index(b) + 1]:
                anc.add(u)
                anc |= self.node(u).anc
            locks |= nb.locks
        return anc, locks

    def barrier_ok(self, m):
        """Whether m may be a barrier target: of this incarnation of the
        device, on a FIFO that is defined."""
        return m.epoch == self.epoch and (m.fid == 0 or m.fid in self.defined)

    # ---- writing work ----

    def make_node(self, batch, hot, forced=(), exclude=(), gate_ok=True):
        """A dag node (not committed until it fits its task's command
        region); None if it does not fit."""
        q = self.q
        ti = q.below(self.ntasks)
        t = self.tasks[ti]
        nd = Node(len(self.nodes) + len(batch), "dag", ti, E2 if q.chance(0.2) else E3)
        nd.epoch = self.epoch
        nd.preds = list(forced)
        cand = [m for m in (self.nodes[-10:] + batch) if m.slot is not None and self.barrier_ok(m)]
        if q.chance(0.5):
            # Beside the work in flight rather than behind it.
            cand = [m for m in cand if not m.locks]
        if cand and q.chance(0.8):
            cnt = 1 if not q.chance(0.2) else 2 + q.below(3)
            nd.preds += [m.id for m in q.sample(cand[-q.choice((2, 4, 12)):], cnt)]
        nd.preds = list(dict.fromkeys(nd.preds))[:MAX_PREDS - 2]
        if q.chance(0.35):
            nd.fill = q.choice(FILLS)
            lf = self.last_fill.get(ti)
            if lf is not None and not self.node(lf).done and lf not in self.closure(nd.preds)[0] \
                    and lf not in nd.preds:
                nd.preds.append(lf)
        nd.gate = gate_ok and q.chance(self.gate_p)
        anc, locks = self.closure(nd.preds)
        cont = [c for c, tl in enumerate(self.tails) if self.node(tl).done or tl in anc]
        if cont and (q.chance(0.8) or len(self.tails) >= MAX_CHAINS):
            nd.chain = q.choice(cont)
        elif len(self.tails) >= MAX_CHAINS:
            nd.chain = q.below(len(self.tails))
            nd.after.append(self.tails[nd.chain])
        else:
            nd.chain = len(self.tails)
        anc, locks = self.closure(nd.preds + nd.after)
        can_root = not locks and not nd.gate and 0 not in exclude
        fifos = [f for f in sorted(self.defined) if f not in self.faulted and f not in exclude]
        if can_root:
            fifos.append(0)
        if not fifos:
            return None
        hh = [f for f in hot if f in fifos]
        nd.fid = q.choice(hh if hh and q.chance(0.7) else fifos)
        pl = self.pend.get(nd.fid, [])
        prev = pl[-1] if pl else None
        if prev is not None and prev not in anc:
            if self.node(prev).slot is not None and len(nd.preds) < MAX_PREDS:
                nd.preds.append(prev)
            else:
                nd.after.append(prev)
        anc, locks = self.closure(nd.preds + nd.after)
        if nd.gate:
            locks.add("gate")
        assert nd.fid != 0 or not locks
        nd.anc, nd.locks = anc, locks
        nd.slot = self.nslots + len([m for m in batch if m.slot is not None])
        nd.segs = dag_segments(self.view, t, nd, t.reg_l, t.scr)
        if not t.room(nd.segs + sum((m.segs for m in batch if m.task == ti and m.kind == "dag"), [])):
            return None
        bars = list(dict.fromkeys(nd.preds + nd.after)) + (["gate"] if nd.gate else [])
        nd.bars = q.sample(bars, len(bars))
        if nd.chain == len(self.tails):
            self.tails.append(nd.id)
        else:
            self.tails[nd.chain] = nd.id
        if nd.fill is not None:
            self.last_fill[ti] = nd.id
        return nd

    def node(self, nid):
        return self.nodes[nid] if nid < len(self.nodes) else self._batch[nid - len(self.nodes)]

    def commit(self, nd):
        """The node joins the program: its slot, chain tail, FIFO queue."""
        assert nd.id == len(self.nodes)
        self.nodes.append(nd)
        if nd.slot is not None:
            assert nd.slot == self.nslots
            self.nslots += 1
        self.pend.setdefault(nd.fid, []).append(nd.id)

    def barriers(self, nd):
        out = []
        for b in nd.bars:
            if b == "gate":
                out.append((GATE, self.gate.stamp + 1))
            else:
                m = self.nodes[b]
                out.append((self.fifo(m.fid).stamp_index, m.stamp))
        return out

    def write_node(self, nd):
        t = self.tasks[nd.task]
        f = self.fifo(nd.fid)
        if nd.kind == "dag":
            res = [(t.src, 0), (t.reg_l, 0)] + ([(t.scr, 0)] if nd.fill is not None else [])
            cbs = [t.cmdbuf(sg) for sg in nd.segs]
            nd.stamp = f.cmd(nd.op, exec_payload(t.tid, cbs, res, nd.op), barriers=self.barriers(nd))
        else:
            op, payload = self.bad_command(nd, t)
            nd.stamp = f.cmd(op, payload, barriers=self.barriers(nd))

    def bad_command(self, n, t):
        """randfault's faulting commands, as randmix writes them: a good
        blit into the node's slot, then the error (the DECODE kinds); an
        exec of a task that does not exist; an unknown or deprecated
        opcode."""
        k, v = n.fault
        if k in DECODE:
            good = seg(abi.ENC_BLIT, b_fill_buf(t.reg_l, slot_off(n.slot), 0x40, 0xEE))
            if k == "record":
                bad = seg(abi.ENC_BLIT, rec((299, 324, 0)[v], struct.pack("<I", t.reg_l)))
            elif k == "overlong":
                bad = seg(abi.ENC_BLIT, rec(306, bytes(0x18), length=0x40 + 0x10 * v))
            elif k == "ref":
                bad = seg(abi.ENC_BLIT, b_fill_buf((0, 0x7FF0, len(t.ol.entries))[v], 0, 0x10, 0x99))
            else:
                bad = seg((3, 6, 0xFF)[v], b"")
            cbs = [t.cmdbuf(good), t.cmdbuf(bad)]
            return n.op, exec_payload(t.tid, cbs, [(t.src, 0), (t.reg_l, 0)], n.op)
        if k == "task":
            return E3, exec_payload(v, [(4 * PAGE, 0x10)], [])
        return v, struct.pack("<II", 0x0BAD0000 | v, n.id)

    def control(self, nodes, label):
        """The control task runs the nodes, in creation order, on the root
        FIFO, into the control region."""
        tc = self.ctl
        dag = [n for n in nodes if n.kind == "dag"]
        for i in range(0, len(dag), CONTROL_GROUP):
            grp = dag[i:i + CONTROL_GROUP]
            segs = []
            for n in grp:
                segs += dag_segments(self.view, tc, n, tc.reg_c, tc.scr_c)
            cbs = [tc.cmdbuf(sg) for sg in segs]
            self.rsub(E3, exec_payload(tc.tid, cbs, [(tc.src, 0), (tc.reg_c, 0), (tc.scr_c, 0)]),
                      "%s: control of n%d-n%d" % (label, grp[0].id, grp[-1].id))
            tc.reset_cmdbufs()

    def ring(self, nodes):
        for fid in reversed(list(dict.fromkeys(n.fid for n in nodes))):
            self.fifo(fid).doorbell()

    def wait_last(self, nodes, what):
        last = {}
        for n in nodes:
            last[n.fid] = n
        for fid, n in sorted(last.items()):
            self.fifo(fid).wait_stamp(n.stamp, label="n%d (%s)" % (n.id, what))
            self.predict(ok=True)

    def held_proof(self, nodes, label):
        """Waits HOLD_MS for the first locked node, to see its stamp does
        not come."""
        locked = [n for n in nodes if n.locks and n.kind == "dag"]
        if locked:
            n = locked[0]
            self.fifo(n.fid).wait_stamp(n.stamp, timeout_ms=HOLD_MS, label="%s: n%d waits on %s%s" % (
                label, n.id, ", ".join(sorted("the gate" if k == "gate" else "fault n%d" % k[1] for k in n.locks)),
                HELD))
            self.predict(ok=False)

    def finish(self, nodes):
        """Nodes that ran: the model's effects in creation order (nodes that
        touch one scratch are ordered by ancestry), their stamps, done."""
        for n in sorted(nodes, key=lambda n: n.id):
            if n.kind == "dag" and n.fill is not None:
                m = self.scr[n.task]
                if m["cached"] is None:
                    m["cached"] = m["leaf"][m["entry"]]
                m["bytes"][m["cached"]] = bytes([n.fill]) * SCR_LEN
            self.sig(self.fifo(n.fid).stamp_index)
            self.pend[n.fid].remove(n.id)
            n.done = True
            n.locks = set()

    def idle_tasks(self):
        """Resets the command region of every task with nothing in flight."""
        busy = {self.nodes[i].task for pl in self.pend.values() for i in pl if self.nodes[i].kind == "dag"}
        for ti, t in enumerate(self.tasks):
            if ti not in busy:
                t.reset_cmdbufs()

    def pending(self):
        return [self.nodes[i] for pl in self.pend.values() for i in pl]

    def cmp_slots(self, nodes, label):
        """The slots of nodes that ran (or faulted), in runs of consecutive
        slots, against the control."""
        s = self.s
        slots = sorted(n.slot for n in nodes if n.slot is not None)
        runs = []
        for x in slots:
            if runs and runs[-1][1] == x - 1:
                runs[-1][1] = x
            else:
                runs.append([x, x])
        for a, b in runs:
            s.cmp(self.live(slot_off(a)), self.ctrl(slot_off(a)), SLOT * (b - a + 1),
                  label="%s: slots %d-%d against the control" % (label, a, b))
            self.predict(equal=True)

    def live(self, off):
        return self.s.pa(self.shared, (1 + REG_PAGES) * PAGE + off)

    def ctrl(self, off):
        return self.s.pa(self.shared, PAGE + off)

    def scratch_cmp(self, ti, label):
        m = self.scr[ti]
        pages = sorted(set(m["leaf"].values()) | ({m["cached"]} if m["cached"] else set()))
        for pg in pages:
            self.model_cmp(self.s.pa(pg), m["bytes"][pg], "%s: task %d's scratch RAM page %d (entry %s on %s, "
                           "bound %s)" % (label, ti + 1, pg, m["entry"], m["leaf"][m["entry"]],
                                          m["cached"] or "nothing"))

    def observe(self, label):
        s = self.s
        mask = 1 if self.faulted else 0
        s.wait(mmio=abi.R_FAULT_MASK, mmio_eq=mask, label="%s: 0x102c %d (model)" % (label, mask))
        self.predict(ok=True)
        s.r(abi.R_FAULT_OFFSET, label="%s: root ok" % label)
        self.predict(val=hex(HEALTHY))
        for fid in sorted(self.defined):
            st = "faulted" if fid in self.faulted else "ok"
            s.dump(self.fifos[fid].node_gpa(NODE_FAULT), 4, "%s: child %d %s" % (label, fid, st))
            if st == "ok":
                self.predict(hex="ffffffff")

    # ==== the steps ====

    def m_step(self):
        q = self.q
        if self.paused:
            table = [("display", 10), ("retype", 2)]
            if any(n.kind == "dag" for n in self.pending()):
                table.append(("cmd_move", 4))
        else:
            left = MAX_SLOTS - self.nslots
            held = [n for n in self.pending() if "gate" in n.locks]
            idle = [f for f in self.defined if f not in self.faulted and not self.pend.get(f)]
            quiet = not self.faulted
            table = [("display", 30), ("life", 5), ("scr_move", 3), ("check", 1), ("churn", 2)]
            if left > 24:
                table.append(("dag", 20))
            if held:
                table.append(("release", 6 + (12 if len(held) > 10 else 0)))
            if self.faults_on and idle and len(self.faulted) < 2 and left > 24:
                table.append(("fault", 5))
            if self.faulted:
                table.append(("recover", 6 + 8 * len(self.faulted)))
            if any(n.kind == "dag" for n in self.pending()):
                table.append(("cmd_move", 4))
            if quiet:
                table += [("pause", 3), ("suspend", 3)]
                if self.teardown_on and not self.pending():
                    table.append(("teardown", 4))
        kind = q.weighted(table)
        self.hist[kind] = self.hist.get(kind, 0) + 1
        getattr(self, "g_" + kind)()
        self.idle_tasks()

    def legal(self):
        """The display walk's actions, without its lifecycle ones: pause,
        suspend, reset and stop are the marathon's own steps."""
        return [a for a in Walk.legal(self) if a[0][0] not in ("pause", "suspend", "reset")]

    def g_display(self):
        self.one_step()

    # ---- GPU work ----

    def g_dag(self, gate_ok=True):
        q = self.q
        fifos = [0] + sorted(f for f in self.defined if f not in self.faulted)
        hot = q.sample(fifos, min(len(fifos), q.choice((2, 3, 5))))
        self._batch = batch = []
        for _ in range(1 + q.below(self.batch_max)):
            nd = self.make_node(batch, hot, gate_ok=gate_ok)
            if nd is None:
                break
            batch.append(nd)
            self.pend.setdefault(nd.fid, []).append(nd.id)   # for the next node's FIFO-previous
        for nd in batch:
            self.pend[nd.fid].remove(nd.id)
        if not batch:
            self._batch = []
            return
        lab = self.gpu_mark("batch: " + "; ".join(n.desc() for n in batch))
        self.control(batch, lab)
        for nd in batch:
            self.commit(nd)
        self._batch = []
        for nd in batch:
            self.write_node(nd)
        self.ring(batch)
        free = [n for n in batch if not n.locks]
        self.wait_last(free, "free")
        self.held_proof(batch, lab)
        self.finish(free)
        self.gevents(lab)
        self.cmp_slots(free, lab)
        for ti in sorted({n.task for n in free if n.fill is not None}):
            self.scratch_cmp(ti, lab)

    def unlock(self, key, lab):
        """Removes a lock from every node in flight; waits for the nodes it
        frees; returns them."""
        freed = []
        for n in self.pending():
            if key in n.locks:
                n.locks.discard(key)
                if not n.locks and n.kind == "dag":
                    freed.append(n)
        freed.sort(key=lambda n: n.id)
        self.wait_last(freed, "released by %s" % lab)
        return freed

    def g_release(self):
        lab = self.gpu_mark("the gate opens over %d nodes" % len([n for n in self.pending() if "gate" in n.locks]))
        self.gate.submit(abi.OP_NOP, label=lab)
        self.predict(ok=True)
        self.sig(GATE)
        freed = self.unlock("gate", lab)
        self.finish(freed)
        self.gevents(lab)
        self.cmp_slots(freed, lab)
        for ti in sorted({n.task for n in freed if n.fill is not None}):
            self.scratch_cmp(ti, lab)

    # ---- faults ----

    def g_fault(self):
        q = self.q
        idle = sorted(f for f in self.defined if f not in self.faulted and not self.pend.get(f))
        fid = q.choice(idle)
        ti = q.below(self.ntasks)
        if not self.tasks[ti].room([bytes(0x400)]):
            return
        k = q.choice(list(DECODE) + ["task", "opcode"])
        v = q.below(3) if k in DECODE else q.choice((5, 6, 7, 9)) if k == "task" else q.choice(UNKNOWN_OPS)
        F = Node(len(self.nodes), "fault", ti, E2 if q.chance(0.2) else E3)
        F.epoch = self.epoch
        F.fid = fid
        F.fault = (k, v)
        F.slot = self.nslots
        F.locks = {("fault", F.id)}
        self.commit(F)
        # Dependents: work on other FIFOs whose barriers wait on F's stamp.
        fifos = [0] + sorted(f for f in self.defined if f not in self.faulted and f != fid)
        hot = q.sample(fifos, min(len(fifos), 3))
        self._batch = deps = []
        for _ in range(q.choice((0, 1, 1, 2, 3))):
            nd = self.make_node(deps, hot, forced=[F.id], exclude=(0, fid))
            if nd is None:
                break
            deps.append(nd)
            self.pend.setdefault(nd.fid, []).append(nd.id)
        for nd in deps:
            self.pend[nd.fid].remove(nd.id)
        lab = self.gpu_mark("fault on child %d: %s; dependents: %s" % (
            fid, F.desc(), "; ".join(n.desc() for n in deps) or "none"))
        if deps:
            self.control(deps, lab)
        for nd in deps:
            self.commit(nd)
        self._batch = []
        self.write_node(F)
        for nd in deps:
            self.write_node(nd)
        self.ring(deps + [F])
        s = self.s
        s.wait(gpa=self.fifos[fid].node_gpa(NODE_FAULT), ne=HEALTHY, label=lab + ": node fault offset")
        self.predict(ok=True)
        s.wait(mmio=abi.R_FAULT_MASK, mmio_ne=0, label=lab + ": 0x102c")
        self.predict(ok=True)
        self.faulted[fid] = F
        self.interrupt()
        self.held_proof(deps, lab)
        self.gevents(lab)
        self.observe(lab)

    def g_recover(self):
        q = self.q
        fid = q.choice(sorted(self.faulted))
        F = self.faulted[fid]
        how = q.choice(("resume", "resume", "delete"))
        lab = self.gpu_mark("recover child %d (n%d) by %s" % (fid, F.id, how))
        c = self.fifos[fid]
        del self.faulted[fid]
        if how == "resume":
            self.s.w(abi.R_RESUME_CHILD, fid)
            if self.faulted:
                self.interrupt()
            c.wait_stamp(F.stamp, label=lab + ": the resume publishes the faulted stamp")
            self.predict(ok=True)
        else:
            self.delete(fid, lab + ": delete child %d" % fid)
            c.wait_stamp(F.stamp, label=lab + ": the delete publishes the faulted stamp")
            self.predict(ok=True)
        self.sig(fid)
        self.pend[fid].remove(F.id)
        F.done = True
        F.locks = set()
        freed = self.unlock(("fault", F.id), lab)
        self.finish(freed)
        if how == "delete":
            self.define(fid, lab + ": define child %d again" % fid)
        self.gevents(lab)
        self.observe(lab)
        self.cmp_slots([F] + freed, lab)
        for ti in sorted({n.task for n in freed if n.fill is not None}):
            self.scratch_cmp(ti, lab)

    # ---- lifetime ----

    def g_life(self):
        q = self.q
        ti = q.below(self.ntasks)
        t = self.tasks[ti]
        m = self.scr[ti]
        k = q.weighted([("del37", 3), ("del40", 2), ("retype", 3), ("remake", 3), ("pso", 1), ("fnpso", 1),
                        ("task", 1)])
        lab = self.gpu_mark("lifetime: task %d %s, %d nodes of it in flight" % (
            t.tid, k, len([n for n in self.pending() if n.task == ti and n.kind == "dag"])))
        if k in ("del37", "remake"):
            op, pl = delete_resource(t.tid, t.scr)
            self.rsub(op, pl, lab + ": 37 the scratch")
            m["cached"] = None
        elif k == "del40":
            op, pl = delete_object(t.tid, DEL_RESOURCE, t.scr)
            self.rsub(op, pl, lab + ": 40 the scratch")
            m["cached"] = None
        elif k in ("pso", "fnpso"):
            which = q.choice(("inc", "add"))
            pso, fn = (t.pso_inc, t.fn_inc) if which == "inc" else (t.pso_add, t.fn_add)
            op, pl = delete_object(t.tid, DEL_COMPUTE_PSO, pso)
            self.rsub(op, pl, lab + ": 40 the %s pipeline" % which)
            if k == "fnpso":
                op, pl = delete_object(t.tid, DEL_FUNCTION, fn)
                self.rsub(op, pl, lab + ": 40 the %s function" % which)
        elif k == "task":
            self.rsub(abi.OP_DELETE_TASK, struct.pack("<I", t.tid), lab + ": delete task %d" % t.tid)
            self.define_task(t, lab + ": again")
            m["cached"] = None
        if k in ("retype", "remake"):
            self.retype(ti, lab)
        self.gevents(lab)
        self.scratch_cmp(ti, lab)

    def retype(self, ti, lab):
        """The scratch entry rewritten in place to name the other page; no
        command follows (the list is read in place)."""
        t, m = self.tasks[ti], self.scr[ti]
        to = "B" if m["entry"] == "A" else "A"
        rewrite_entry(t, t.scr, abi.T_BUFFER, record=buffer_record(PAGE, SCR_VPN[to]))
        m["entry"] = to
        self.s.mark("%s: task %d's scratch entry now names page %s" % (lab, t.tid, to))

    def g_retype(self):
        ti = self.q.below(self.ntasks)
        self.retype(ti, "step %d: [gpu] lifetime while paused" % self.step_no)

    # ---- memory ----

    def g_scr_move(self):
        """The guest moves a scratch VA page: 34, the leaf pointed at a
        fresh page holding a marker, 57."""
        q, s = self.q, self.s
        ti = q.below(self.ntasks)
        t, m = self.tasks[ti], self.scr[ti]
        pg = q.choice(("A", "B"))
        vpn = SCR_VPN[pg]
        new = s.alloc_page()
        mark = q.choice(MARKERS)
        lab = self.gpu_mark("memory: task %d's scratch page %s (VA page %d) moved from RAM page %d to %d "
                            "(34, leaf, 57), %d nodes of it in flight" % (
                                t.tid, pg, vpn, m["leaf"][pg], new,
                                len([n for n in self.pending() if n.task == ti and n.kind == "dag"])))
        self.rsub(abi.OP_UNMAP_MEMORY, struct.pack("<IQQ", t.tid, vpn * PAGE, PAGE), lab + ": 34")
        s.fill(s.pa(new), SCR_LEN, mark)
        set_leaf(t, vpn, new)
        t.leaves[vpn] = new
        self.rsub(abi.OP_MAP_MEMORY2, struct.pack("<IQQ", t.tid, vpn * PAGE, PAGE), lab + ": 57")
        m["leaf"][pg] = new
        m["bytes"][new] = bytes([mark]) * SCR_LEN
        self.gevents(lab)
        self.scratch_cmp(ti, lab)

    def g_cmd_move(self):
        """Command-buffer pages of work in flight moved to fresh pages: one
        leaf, or (a task two deep) the whole leaf table swapped for a copy
        whose every command page is new. The old pages are poisoned."""
        q, s = self.q, self.s
        busy = sorted({n.task for n in self.pending() if n.kind == "dag"})
        ti = q.choice(busy)
        t = self.tasks[ti]
        used = t.used_cmd_pages()
        whole = t.depth == 2 and q.chance(0.5)
        vpns = used if whole else [q.choice(used)]
        moved = {}
        for v in vpns:
            new = s.alloc_page()
            for va, data in t.cmdlog:
                lo, hi = max(va, v * PAGE), min(va + len(data), (v + 1) * PAGE)
                if lo < hi:
                    s.mem(s.pa(new, lo - v * PAGE), data[lo - va:hi - va])
            moved[v] = new
        lab = self.gpu_mark("memory%s: task %d's command pages %s moved to RAM pages %s%s" % (
            " while paused" if self.paused else "", t.tid, vpns, [moved[v] for v in vpns],
            " by a new leaf table" if whole else ""))
        old = {v: t.leaves[v] for v in vpns}
        for v in vpns:
            t.leaves[v] = moved[v]
        if whole:
            table = write_table(s, {v: leaf(s, p) for v, p in t.leaves.items()})
            s.u32(t.table_gpa(0, 0, 0), s.pfn(table))
            t._tables[(1, 0)] = table
        else:
            for v in vpns:
                set_leaf(t, v, moved[v])
        for v in vpns:
            s.fill(s.pa(old[v]), PAGE, 0xFF)

    # ---- child FIFOs ----

    def g_churn(self):
        q = self.q
        idle = sorted(f for f in self.defined if f not in self.faulted and not self.pend.get(f))
        dead = [f for f in range(1, self.nchildren + 1) if f not in self.defined]
        acts = []
        if idle and len(self.defined) > 1 and q.chance(0.7):
            fid = q.choice(idle)
            acts.append(("delete", fid))
            if q.chance(0.5):
                acts.append(("define", fid))
        for fid in dead:
            if q.chance(0.6):
                acts.append(("define", fid))
        if not acts:
            return
        lab = self.gpu_mark("child FIFOs: %s" % ", ".join("%s %d" % a for a in acts))
        for k, fid in acts:
            if k == "delete":
                self.defined.discard(fid)
                self.delete(fid, lab + ": delete child %d" % fid)
            else:
                self.define(fid, lab + ": define child %d" % fid)
                self.defined.add(fid)
        self.gevents(lab)
        self.observe(lab)

    # ---- lifecycle ----

    def g_pause(self):
        self.gpu_mark("pause, %d nodes in flight" % len(self.pending()))
        self.a_pause()

    def g_suspend(self):
        self.gpu_mark("suspend, %d nodes in flight" % len(self.pending()))
        self.a_suspend()
        for m in self.scr:
            m["cached"] = None

    def g_teardown(self):
        how = self.q.choice(("reset", "stop"))
        self.a_reset(how)
        lab = self.gpu_mark("after %s: the GPU set-up again" % how)
        for fid in sorted(self.defined):
            self.define(fid, lab + ": define child %d" % fid)
        self.define(GATE, lab + ": define the gate")
        for t in self.tasks + [self.ctl]:
            self.define_task(t, lab)
        for m in self.scr:
            m["cached"] = None
        self.epoch += 1
        self.gevents(lab)
        self.observe(lab)

    def g_check(self):
        s = self.s
        lab = self.gpu_mark("checkpoint")
        self.gevents(lab)
        self.observe(lab)
        done = [n for n in self.nodes if n.done and n.slot is not None]
        self.cmp_slots(done, lab)
        pend_chains = {n.chain for n in self.pending() if n.kind == "dag"}
        quiet = [c for c in range(len(self.tails)) if c not in pend_chains]
        runs = []
        for c in quiet:
            if runs and runs[-1][1] == c - 1:
                runs[-1][1] = c
            else:
                runs.append([c, c])
        for a, b in runs:
            s.cmp(self.live(a * CNT), self.ctrl(a * CNT), CNT * (b - a + 1),
                  label="%s: chain counters %d-%d against the control" % (lab, a, b))
            self.predict(equal=True)
        for ti in range(self.ntasks):
            self.scratch_cmp(ti, lab)

    # ---- the end ----

    def settle_all(self):
        if self.paused:
            self.a_unpause()
        while self.faulted:
            fid = sorted(self.faulted)[0]
            F = self.faulted[fid]
            lab = self.gpu_mark("the end: resume child %d (n%d)" % (fid, F.id))
            del self.faulted[fid]
            self.s.w(abi.R_RESUME_CHILD, fid)
            if self.faulted:
                self.interrupt()
            self.fifos[fid].wait_stamp(F.stamp, label=lab + ": the faulted stamp")
            self.predict(ok=True)
            self.sig(fid)
            self.pend[fid].remove(F.id)
            F.done = True
            F.locks = set()
            freed = self.unlock(("fault", F.id), lab)
            self.finish(freed)
            self.gevents(lab)
            self.cmp_slots([F] + freed, lab)

    def gpu_final(self):
        s = self.s
        self.step_no = self.steps
        self.settle_all()
        if any("gate" in n.locks for n in self.pending()):
            self.g_release()
        assert not self.pending()
        lab = self.gpu_mark("the end")
        self.gevents(lab)
        self.observe(lab)
        if self.nslots:
            s.cmp(self.live(slot_off(0)), self.ctrl(slot_off(0)), SLOT * self.nslots,
                  label="%s: every slot, 0-%d, against the control" % (lab, self.nslots - 1))
            self.predict(equal=True)
        if self.tails:
            s.cmp(self.live(0), self.ctrl(0), CNT * len(self.tails),
                  label="%s: the %d chain counters against the control" % (lab, len(self.tails)))
            self.predict(equal=True)
            s.dump(self.ctrl(0), CNT * len(self.tails), "%s: the control's chain counters" % lab)
        for ti in range(self.ntasks):
            self.scratch_cmp(ti, lab)
        s.dump(s.fifo.base_gpa(), 4 * (GATE + 1), "%s: stamp slots" % lab)
        s.dump(s.pa(s.root_page, abi.ROOT_CHILD_NODE), abi.CHILD_NODE_SIZE * GATE, "%s: child nodes 1-20" % lab)

    def gpu_before_death(self):
        """Faults recovered; then work held on the gate, in flight when the
        host dies."""
        self.step_no = self.steps
        self.settle_all()
        if not any("gate" in n.locks for n in self.pending()):
            for _ in range(4):
                self.g_dag(gate_ok=True)
                if any("gate" in n.locks for n in self.pending()):
                    break

    def gpu_summary(self):
        kinds = {}
        for n in self.nodes:
            k = n.kind if n.kind == "dag" else "fault %s" % n.fault[0]
            kinds[k] = kinds.get(k, 0) + 1
        return "GPU: %d children, %d task%s (depths %s), gate p %g; steps: %s; nodes: %s." % (
            self.nchildren, self.ntasks, "s" if self.ntasks > 1 else "", self.depths, self.gate_p,
            ", ".join("%s %d" % kv for kv in sorted(self.hist.items())),
            ", ".join("%s %d" % kv for kv in sorted(kinds.items())))


# ---- entry points ----

def marathon(seed, steps, doc="", death=None):
    m = Marathon(seed, steps, doc, death=death)
    m.build()
    return m


def build(seed, steps, doc, death=None):
    return marathon(seed, steps, doc, death).s


def check(paths=None):
    """Every expected (or, with paths, output) file of interplay_marathon
    against the model: every prediction, every wait but [held] and [dies]
    ones succeeded, and the outcome."""
    import glob
    import os
    from scenarios.interplay_marathon import SEEDS, __doc__ as doc
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = paths or os.path.join(here, "expected")
    bad = 0
    for seed, steps, death in SEEDS:
        w = marathon(seed, steps, doc, death)
        for p in sorted(glob.glob(os.path.join(root, "interplay_marathon", w.name, "*.jsonl"))):
            lines = [json.loads(l) for l in open(p) if l.strip() and not l.startswith("#")]
            obs = [o for o in lines if "ev" in o or "kind" in o]
            complaints = check_obs(w, obs)
            for o in obs:
                if o.get("ev") == "cmp" and o.get("equal") is False:
                    complaints.append("cmp differs: %s" % o.get("label"))
            for c in complaints:
                print("%s: %s" % (os.path.relpath(p, here), c))
            bad += len(complaints)
    return bad


if __name__ == "__main__":
    import sys
    sys.exit(1 if check(sys.argv[1] if len(sys.argv) > 1 else None) else 0)
