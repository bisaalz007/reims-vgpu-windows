"""Seeded dependency graphs of GPU work over many FIFOs (scenarios/
interplay_dag.py): a generator of DAGs whose nodes are execs spread over the
root FIFO, up to 16 child FIFOs and several tasks, and an emitter that turns
one into a Scenario whose every result proves the order the graph implies.

A node is one exec (67, or 55) on one FIFO and one task. Its edges are
barriers {stampIndex, value} on the stamps of the nodes it depends on, on
other FIFOs or its own. The implementation orders exactly this much
(pgtest/interplay_util.py says more):
  - a FIFO publishes its stamps in command order (fifo::stamps::StampQueue's
    pump), and an exec's stamp is signalled when its command buffer
    completed (in the service, at the reply);
  - a barrier holds its FIFO before the command is consumed until the slot
    reaches the value, each barrier of a command in list order
    (device::waits, wait_stamps);
  so a node runs after every node whose stamp it waits on, and after
  every node published before those (the FIFO prefix of each, and their
  own ancestors): its ancestors. Nothing else is ordered: two nodes
  neither of which is the other's ancestor run on their own threads.

Order witnesses. Every node, in one exec, writes only its own 0x100-byte
slot, and the counter of its chain:
  slot + 0x00  (0x40) its chain's counter block after inc_k ran on it
               (fixtures/shaders/compute_inc.metal: c[0] += 1, c[c[0]] =
               0x11 c[0] below 16): the node's position in its chain
  slot + 0x40  (0x40) add_one (compute_add.metal) over its first
               predecessor's +0x40 (a source: over SRC): the length of the
               first-predecessor path, 16 words
  slot + 0x80  (0x80) for each predecessor, up to 16, the first word of
               its +0x00 and of its +0x40: what it had written when this
               node read it
A chain is a sequence of nodes each of which has the one before it among
its ancestors, so its counter's increments are totally ordered; a node
reads only its ancestors' slots. Every byte written is therefore fixed by
the graph alone, whatever the interleaving, and a node that ran before one
of its ancestors finished leaves a wrong byte.

The control run (as pgtest/randgen.py's): before the walk, one
control run on the root FIFO executes every node in creation order (a
topological order), eight to an exec, into a control region laid out as
the live one; after each phase the live slots are compared with the
control's, and at the end the chain counters. The control's counters and
two of its slots are dumped, so the expected files hold the absolute bytes
too.

Invariants the generator keeps:
  - Each exec on a FIFO is published alone: a node whose FIFO-previous
    node is not among its ancestors gets a barrier on it (a barrier on the
    FIFO's own slot for a value it pushed is not circular: add_wait's walk
    finds no cycle). An exec followed on its FIFO by other
    commands would otherwise publish with them or alone by timing, and the
    interrupt count with it (pgtest/interplay_util.py). So the phase's
    interrupt count is exact; their order is recorded only when the phase's
    nodes are totally ordered (every node an ancestor of the next).
  - Blocks-setup block calls from FIFOs at once are reported sorted and
    digested (events(sort_blocks, blocks_digest)).
  - At most 18 FIFOs wait at once (16 children, the root, the pressure
    FIFO), below the 32 of the waiter mask (docs/testing.md, 6.7 and 7.4);
    every child's stampIndex is its id. (One hand-written shape,
    interplay_dag_alias_report, uses child 33 on purpose.)
  - Held work waits on the gate (child 20, NOPs only) and is released by
    one NOP. Before a pause, a suspend or the release, the guest waits for
    every node that does not depend on the gate, so nothing is in flight:
    the held FIFOs are blocked in a barrier, or behind one. A pause, or a
    suspend's rvg_device_finish_suspend, quiesces the FIFOs and wakes their
    waits (signal_fifos), which give up before the FIFO advances past the
    command, and the command is parsed again after the unpause or the resume.
    Never rvg_device_will_suspend then unpause over held work:
    rvg_device_will_suspend wakes no wait, and the unpause starts a second
    thread on each blocked FIFO (scenarios/interplay_dag.py,
    interplay_dag_abandon_blocked*). A hold is 300 ms, or 2.5 s: past the 2 s
    after which a blocked barrier logs and posts a stamp-wait report and waits
    again without a timeout (report_stamp_wait_timeout).
  - Child FIFOs are deleted and defined again only between phases, idle.
    A child keeps its stamp count across that, so a later barrier on one
    of its old stamps passes at once (the slot holds it or a later value) and a
    new one waits for the new incarnation. Deleting child n zeroes node
    n+1's pageListPage (docs/testing.md 7.1), which the next start of
    child n+1 (an unpause's, a resume's) maps its ring from: the guest
    writes node n+1's +0x10 back after each delete. The gate's neighbour 19
    is never used.

Rust paths name items of reims-vgpu-core. Nothing here is imported from the
library's code.
"""
import struct

from . import abi
from .abi import PAGE
from .compute_objects import Blit as B, Compute as C, compute_blob, compute_pipeline_record, function_record
from .diff_util import exec_payload
from .fifo import ChildFIFO
from .fifo_util import NODE_PAGE_LIST, NODE_READ, child_fault, resume_child, track_stamps
from .randgen import Rng
from .scenario import Scenario
from .task import seg
from .tasks_lib import XGPUTask

INC = "shaders/compute_inc.metallib"
ADD = "shaders/compute_add.metallib"

MAX_CHILD = 16
PRESS = 18               # the pending-pressure FIFO of hand-written shapes
GATE = 20                # child 19 is never defined, so never deleted
SLOTS = GATE + 1         # stamp slots 0-20
PRIVATE = 10             # a task's own VA pages: list, records, command buffers, kernels
REG_PAGES = 2
SHARED = 1 + 2 * REG_PAGES   # SRC, the control region, the live region
CNT = 0x40               # a chain's counter block
MAX_CHAINS = 32
SLOT_BASE = MAX_CHAINS * CNT
SLOT = 0x100
MAX_NODES = (REG_PAGES * PAGE - SLOT_BASE) // SLOT
MAX_PREDS = 16
CONTROL_GROUP = 8
HOLD_MS = 300
HOLD_LONG_MS = 2500      # past the 2 s timer of a barrier's wait
EVENTS = ("hold", "hold_long", "pause", "pause_long", "suspend")


# ---- the graph ----

class Node:
    """An exec: FIFO fid, task index `task`, chain `chain`, data
    predecessors `preds` (node ids; preds[0] feeds add_one), barrier list
    `bars` (("n", id) or ("gate",), in the order the command lists them),
    opcode `op`."""

    def __init__(self, nid, phase, fid, task, op):
        self.id, self.phase, self.fid, self.task, self.op = nid, phase, fid, task, op
        self.chain = None
        self.preds = []
        self.bars = []
        self.stamp = None

    def desc(self):
        bars = ",".join("gate" if b[0] == "gate" else "%d" % b[1] for b in self.bars)
        return "n%d F%d t%d c%d%s%s" % (self.id, self.fid, self.task + 1, self.chain,
                                        " op55" if self.op == abi.OP_EXEC_INDIRECT2 else "",
                                        " <" + bars if bars else "")


class Phase:
    """Nodes written, rung, and waited for together. pre: FIFO actions
    before it (("delete", fid), ("define", fid)). event: None, or what the
    guest does while the gated nodes are held (EVENTS). pressure: child
    PRESS carries a faulted head and 129 NOPs ahead of its nodes."""

    def __init__(self, index):
        self.index = index
        self.nodes = []
        self.pre = []
        self.event = None
        self.pressure = False
        self.anc = {}          # node id -> set of in-phase ancestors
        self.gated = set()

    def total(self):
        """Whether every node is an ancestor of every later one."""
        ids = [n.id for n in self.nodes]
        return all(ids[i] in self.anc[ids[j]] for j in range(len(ids)) for i in range(j))


class Graph:
    """The program: children defined at set-up, tasks, phases, nodes."""

    def __init__(self, children, tasks):
        self.children = list(children)
        self.tasks = tasks
        self.nodes = []
        self.phases = []
        self.tails = []        # chain -> its last node's id
        self.defined = set(children)
        self.tags = []

    def uses_gate(self):
        return any(p.event for p in self.phases)

    def fifo_prefix(self, ph, w):
        """w and the nodes of its FIFO before it, in its phase."""
        nw = self.nodes[w]
        return [n.id for n in ph.nodes if n.fid == nw.fid and n.id <= w]

    def ancestors(self, ph, v):
        """Every in-phase node published before v's barriers pass: each
        barrier target, the nodes before it on its FIFO (published in
        order), and their ancestors."""
        out = set()
        for b in self.nodes[v].bars:
            if b[0] != "n" or self.nodes[b[1]].phase != ph.index:
                continue
            for u in self.fifo_prefix(ph, b[1]):
                out.add(u)
                out |= ph.anc.get(u, set())
        return out

    # ---- building ----

    def phase(self, pre=(), event=None, pressure=False):
        ph = Phase(len(self.phases))
        for a in pre:
            ph.pre.append(a)
            if a[0] == "delete":
                self.defined.discard(a[1])
            else:
                self.defined.add(a[1])
        ph.event = event
        ph.pressure = pressure
        self.phases.append(ph)
        return ph

    def node(self, fid, preds=(), task=0, chain=None, gate=False, op=abi.OP_EXEC_INDIRECT3, order=None,
             self_bar=True):
        """Adds a node to the last phase. chain: an existing chain (its tail
        must end up among the node's ancestors or in an earlier phase), or
        None for a new one. order: the barrier list's order, a permutation
        of range(len(bars)). Adds the barrier on the FIFO-previous node when
        it is not already an ancestor (self_bar=False refuses, for a shape
        that must not have one)."""
        ph = self.phases[-1]
        assert fid == 0 or fid in self.defined or fid == PRESS, "FIFO %d not defined" % fid
        n = Node(len(self.nodes), ph.index, fid, task, op)
        self.nodes.append(n)
        ph.nodes.append(n)
        n.preds = list(preds)
        n.bars = [("n", p) for p in n.preds]
        if gate:
            n.bars.append(("gate",))
        prev = [m.id for m in ph.nodes[:-1] if m.fid == fid]
        anc = self.ancestors(ph, n.id)
        if prev and prev[-1] not in anc:
            assert self_bar, "node %d needs a barrier on %d" % (n.id, prev[-1])
            n.preds.append(prev[-1])
            n.bars.insert(0, ("n", prev[-1]))
            anc = self.ancestors(ph, n.id)
        if chain is None:
            assert len(self.tails) < MAX_CHAINS
            chain = len(self.tails)
            self.tails.append(None)
        else:
            t = self.tails[chain]
            assert self.nodes[t].phase != ph.index or t in anc, "chain %d's tail %d is not an ancestor of %d" % (
                chain, t, n.id)
        n.chain = chain
        self.tails[chain] = n.id
        if order is not None:
            n.bars = [n.bars[i] for i in order]
        ph.anc[n.id] = anc
        assert n.id < MAX_NODES and len(n.preds) <= MAX_PREDS
        return n

    def finish(self):
        for ph in self.phases:
            direct = {n.id for n in ph.nodes if ("gate",) in n.bars}
            ph.gated = {n.id for n in ph.nodes if n.id in direct or ph.anc[n.id] & direct}
            assert not ph.event or ph.gated, "phase %d holds nothing" % ph.index
            assert ph.event or not direct, "phase %d gates without an event" % ph.index
        return self


def generate(seed, nodes_hint):
    """A random graph: 1-16 children, 1-3 tasks, 1-4 phases, each of up to
    `nodes_hint` nodes, some held on the gate across an event, children
    deleted and defined again between phases."""
    r = Rng(seed)
    k = r.choice((2, 4, 6, 8, 10, 12, 14, 16, 16))
    ntasks = r.choice((1, 1, 2, 3))
    nphases = r.choice((1, 2, 2, 3, 3, 4))
    window = r.choice((1, 2, 3, 6, 12, 100))   # deep ... wide
    join = r.choice((0.05, 0.15, 0.3))         # how often a node joins several
    passed = r.choice((0.0, 0.1, 0.3))         # a predecessor from an earlier phase
    churn = r.chance(0.5)
    g = Graph(range(1, k + 1), ntasks)
    g.tags = ["children=%d" % k, "tasks=%d" % ntasks, "window=%d" % window]
    total = 0
    for pi in range(nphases):
        pre = []
        if pi and churn:
            dead = [f for f in range(1, k + 1) if f not in g.defined]
            for fid in dead:
                if r.chance(0.6):
                    pre.append(("define", fid))
            live = sorted(g.defined)
            for fid in r.sample(live, r.choice((1, 1, 2))):
                if len(live) <= 1:
                    break
                pre.append(("delete", fid))
                live.remove(fid)
                if r.chance(0.6):
                    pre.append(("define", fid))
                    live.append(fid)
        event = r.weighted([(None, 30), ("hold", 12), ("hold_long", 4), ("pause", 18), ("pause_long", 4),
                            ("suspend", 12)])
        ph = g.phase(pre=pre, event=event)
        fifos = [0] + sorted(g.defined)
        hot = r.sample(fifos, min(len(fifos), r.choice((2, 4, 17))))
        n = min(nodes_hint - 2 + r.below(5), MAX_NODES - total)
        if n < 2:
            g.phases.pop()
            break
        total += n
        gate_p = r.choice((0.2, 0.4, 0.7)) if event else 0
        for i in range(n):
            fid = r.choice(hot if r.chance(0.7) else fifos)
            here = [m.id for m in ph.nodes]
            earlier = [m.id for m in g.nodes if m.phase < ph.index]
            preds = []
            if here and not (i == 0 or r.chance(0.15)):
                cnt = 1
                if r.chance(join):
                    cnt = 2 + r.below(3)
                cand = here[-window:]
                preds = r.sample(cand, cnt)
            if earlier and r.chance(passed):
                preds += r.sample(earlier, 1 + r.below(2))
            preds = list(dict.fromkeys(preds))[:MAX_PREDS - 2]
            # A chain to continue: one whose tail will be an ancestor.
            anc_now = set()
            for p in preds:
                if g.nodes[p].phase == ph.index:
                    anc_now |= set(g.fifo_prefix(ph, p))
                    for u in g.fifo_prefix(ph, p):
                        anc_now |= ph.anc[u]
            prev = [m.id for m in ph.nodes if m.fid == fid]
            if prev:
                anc_now |= set(g.fifo_prefix(ph, prev[-1]))
                anc_now |= ph.anc[prev[-1]]
            cont = [c for c, t in enumerate(g.tails)
                    if g.nodes[t].phase < ph.index or t in anc_now]
            chain = None
            if cont and (r.chance(0.8) or len(g.tails) >= MAX_CHAINS):
                chain = r.choice(cont)
            elif len(g.tails) >= MAX_CHAINS:
                chain = r.below(len(g.tails))
                if g.tails[chain] not in preds:
                    preds.append(g.tails[chain])
            gate = event is not None and r.chance(gate_p)
            op = abi.OP_EXEC_INDIRECT2 if r.chance(0.2) else abi.OP_EXEC_INDIRECT3
            node = g.node(fid, preds, task=r.below(ntasks), chain=chain, gate=gate, op=op)
            perm = list(range(len(node.bars)))
            node.bars = [node.bars[j] for j in r.sample(perm, len(perm))]
        if event and not any(("gate",) in m.bars for m in ph.nodes):
            m = r.choice(ph.nodes)
            m.bars.append(("gate",))
    return g.finish()


# ---- the emitter ----

class DagTask(XGPUTask):
    """A task of PRIVATE pages of its own (object list at VA 0, records at
    pages 1-3, command buffers at 4-7, the two kernels from page 8), then
    SHARED pages every task maps to the same guest pages: SRC (a ramp),
    the control region and the live region. Every task's object list names
    the same objects in the same order, so a reference means the same in
    each: inc_k's function and pipeline, add_one's, SRC, the control
    region, the live region."""

    def __init__(self, s, tid, shared):
        leaves = {v: s.alloc_page() for v in range(PRIVATE)}
        leaves.update({PRIVATE + i: p for i, p in enumerate(shared)})
        super().__init__(s, tid=tid, pages=PRIVATE + SHARED, leaves=leaves)
        self.pso_inc = self._kernel(INC)
        self.pso_add = self._kernel(ADD)
        base = PRIVATE * PAGE
        self.src, self.src_va = self.buffer(PAGE, va=base)
        self.reg_c, self.reg_c_va = self.buffer(REG_PAGES * PAGE, va=base + PAGE)
        self.reg_l, self.reg_l_va = self.buffer(REG_PAGES * PAGE, va=base + (1 + REG_PAGES) * PAGE)

    def _kernel(self, fixture):
        va = self.reserve(0x2000)
        n = self.s.mem_file(self.gpa(va), fixture)
        fn = self.ol.add(abi.T_FUNCTION, function_record(va, n))
        return self.ol.add(abi.T_SERIALIZED, compute_pipeline_record(compute_blob(fn)))

    def resources(self):
        return [(self.src, 0), (self.reg_c, 0), (self.reg_l, 0)]


def slot_off(nid):
    return SLOT_BASE + nid * SLOT


def node_segments(t, n, reg):
    """Node n's two segments on task t, writing into region `reg` (the
    control's or the live one): a compute pass (inc_k on its chain's
    counter, then add_one of 16 words), then a blit pass (the counter
    block copied out, and each predecessor's two first words)."""
    o = slot_off(n.id)
    if n.preds:
        src = (reg, slot_off(n.preds[0]) + 0x40)
    else:
        src = (t.src, 0x40 * (n.id % 4))
    comp = (C.set_pipeline(t.pso_inc) + C.set_buffers(0, [(reg, n.chain * CNT)])
            + C.dispatch_threads((1, 1, 1), (1, 1, 1))
            + C.set_pipeline(t.pso_add) + C.set_buffers(0, [(reg, o + 0x40), src])
            + C.dispatch_threads((16, 1, 1), (16, 1, 1)))
    blit = B.copy_buffer_to_buffer(reg, reg, n.chain * CNT, o, CNT)
    for i, p in enumerate(n.preds):
        po = slot_off(p)
        blit += B.copy_buffer_to_buffer(reg, reg, po, o + 0x80 + 8 * i, 4)
        blit += B.copy_buffer_to_buffer(reg, reg, po + 0x40, o + 0x84 + 8 * i, 4)
    return [seg(abi.ENC_COMPUTE, comp), seg(abi.ENC_BLIT, blit)]


class Emitter:
    def __init__(self, name, doc, g, watchdog_s=60, repair=True):
        self.g = g
        self.repair = repair
        long_holds = sum(1 for p in g.phases if p.event in ("hold_long", "pause_long"))
        self.s = s = Scenario(name, watchdog_s=watchdog_s + 3 * long_holds, doc=doc)
        self.root = s.bring_up()
        track_stamps(s, SLOTS)
        self.fifos = {}
        self.shared = s.alloc_page(SHARED)
        self.tasks = [DagTask(s, tid, [self.shared + i for i in range(SHARED)]) for tid in range(1, g.tasks + 1)]
        t = self.tasks[0]
        t.write(t.src_va, struct.pack("<256I", *[0x1000 * (i // 16 + 1) + i for i in range(256)]))

    def fifo(self, fid):
        return self.root if fid == 0 else self.fifos[fid]

    def events(self, label, order=False):
        self.s.events(label, order=order, irq_stamps=order, sort_blocks=True, blocks_digest=True)

    def stamps(self, label):
        self.s.dump(self.root.base_gpa(), 4 * SLOTS, label)

    def define(self, fid):
        c = self.fifos.get(fid)
        if c is None:
            c = self.fifos[fid] = ChildFIFO(self.s, fid)
        c.written = 0
        c.setup_node()
        c.define(label="define child %d" % fid)
        return c

    def delete(self, fid):
        s = self.s
        c = self.fifos[fid]
        c.delete(label="delete child %d" % fid)
        s.dump(c.node_gpa(0), abi.CHILD_NODE_SIZE, "node %d after deleting it" % fid)
        # Node fid+1 without its read offset, which moves while that child runs.
        s.dump(c.node_gpa(0) + abi.CHILD_NODE_SIZE + 8, abi.CHILD_NODE_SIZE - 8, "node %d from +8 after deleting %d" % (fid + 1, fid))
        nxt = self.fifos.get(fid + 1)
        if self.repair and nxt is not None and (fid + 1) in self.live:
            # The delete zeroed node fid+1's pageListPage; the guest
            # writes it back, so the next start maps its ring again.
            s.u32(nxt.node_gpa(NODE_PAGE_LIST), s.pfn(nxt.list_page))
        self.live.discard(fid)

    # ---- set-up and control ----

    def setup(self):
        s, g = self.s, self.g
        self.live = set()
        for fid in g.children:
            self.define(fid)
            self.live.add(fid)
        if any(p.pressure for p in g.phases):
            self.define(PRESS)
        if g.uses_gate():
            self.gate = self.define(GATE)
        for t in self.tasks:
            t.define(label="define task %d" % t.tid)
            t.install(label="task %d's object list" % t.tid)
        self.control()
        self.events("set up")

    def control(self):
        s, g = self.s, self.g
        t = self.tasks[0]
        for i in range(0, len(g.nodes), CONTROL_GROUP):
            grp = g.nodes[i:i + CONTROL_GROUP]
            segs = []
            for n in grp:
                segs += node_segments(t, n, t.reg_c)
            cbs = [t.cmdbuf(sg) for sg in segs]
            s.submit(abi.OP_EXEC_INDIRECT3, exec_payload(t.tid, cbs, t.resources()),
                     label="control: nodes %d-%d" % (grp[0].id, grp[-1].id))
            t.reset_cmdbufs()
        nch = len(g.tails)
        s.dump(t.gpa(t.reg_c_va), nch * CNT, "control: the %d chain counters" % nch)
        widest = max(g.nodes, key=lambda n: (len(n.preds), n.id))
        for n in sorted({g.nodes[-1].id, widest.id}):
            s.dump(t.gpa(t.reg_c_va + slot_off(n)), SLOT, "control: node %d's slot" % n)

    # ---- phases ----

    def write_node(self, n):
        t = self.tasks[n.task]
        f = self.fifo(n.fid)
        bars = []
        for b in n.bars:
            if b[0] == "gate":
                bars.append((GATE, self.gate.stamp + 1))
            else:
                p = self.g.nodes[b[1]]
                bars.append((p.fid, p.stamp))
        cbs = [t.cmdbuf(sg) for sg in node_segments(t, n, t.reg_l)]
        at = f.written
        n.stamp = f.cmd(n.op, exec_payload(t.tid, cbs, t.resources(), n.op), barriers=bars)
        n.length = f.written - at

    def pre(self, ph):
        """The phase's FIFO actions, on the root FIFO, all idle."""
        for a in ph.pre:
            if a[0] == "delete":
                self.delete(a[1])
            else:
                self.define(a[1])
                self.live.add(a[1])

    def write_ring(self, ph):
        """Writes every node (the pressure FIFO's faulted head and NOPs
        first), then rings the FIFOs in reverse order of first use, so the
        later ones wait first. Returns the pressure FIFO, if any."""
        press = None
        if ph.pressure:
            press = self.fifos[PRESS]
            press.cmd(99)
            for _ in range(129):
                press.cmd(abi.OP_NOP)
        for n in ph.nodes:
            self.write_node(n)
        fids = list(dict.fromkeys([n.fid for n in ph.nodes] + ([PRESS] if press else [])))
        for fid in reversed(fids):
            self.fifo(fid).doorbell()
        if press is not None:
            # The faulted head and 127 NOPs fill the 128 pending slots; the
            # 128th NOP is read and advanced, and its push blocks
            # (fifo::stamps::StampQueue::push_stamp).
            child_fault(self.s, press, label="the pressure FIFO faulted")
            self.s.wait(gpa=press.node_gpa(NODE_READ), eq=12 * 129,
                        label="its thread blocked pushing its 128th NOP's stamp")
        return press

    def hold(self, ph):
        """Waits for every node that does not depend on the gate, holds the
        rest HOLD_MS or HOLD_LONG_MS, and does the phase's event."""
        s = self.s
        free = [n for n in ph.nodes if n.id not in ph.gated]
        self.wait_last(free, "not held")
        held = [n for n in ph.nodes if n.id in ph.gated]
        ms = HOLD_LONG_MS if ph.event in ("hold_long", "pause_long") else HOLD_MS
        self.fifo(held[0].fid).wait_stamp(held[0].stamp, timeout_ms=ms,
                                          label="node %d held (%d ms)" % (held[0].id, ms))
        self.events("phase %d: held" % ph.index, order=ph.total())
        self.stamps("phase %d: stamps while held" % ph.index)
        if ph.event in ("pause", "pause_long"):
            s.api("pause")
            s.api("unpause")
            self.events("phase %d: paused and unpaused" % ph.index)
        elif ph.event == "suspend":
            s.api("willSuspend")
            s.api("finishSuspend")
            s.api("recreate")
            s.api("willResume")
            s.api("didResume")
            self.events("phase %d: suspended, resumed on a fresh device" % ph.index)

    def open_gate(self, ph):
        self.gate.submit(abi.OP_NOP, label="phase %d: the gate opens" % ph.index)

    def done(self, ph):
        """Every node's stamp, the events, the stamp slots, and the phase's
        slots against the control."""
        s = self.s
        self.wait_last(ph.nodes, "done")
        self.events("phase %d done" % ph.index, order=ph.total())
        self.stamps("phase %d: stamps" % ph.index)
        first = ph.nodes[0].id
        s.cmp(s.pa(self.shared, (1 + REG_PAGES) * PAGE + slot_off(first)),
              s.pa(self.shared, PAGE + slot_off(first)), SLOT * len(ph.nodes),
              label="phase %d: slots of nodes %d-%d against the control" % (ph.index, first, ph.nodes[-1].id))
        for t in self.tasks:
            t.reset_cmdbufs()

    def run_phase(self, ph):
        s = self.s
        self.pre(ph)
        s.mark("phase %d: %s%s" % (ph.index, ph.event or "no event", ", pressure" if ph.pressure else ""))
        press = self.write_ring(ph)
        if ph.event:
            self.hold(ph)
            self.open_gate(ph)
        if press is not None:
            s.dump(press.node_gpa(0), abi.CHILD_NODE_SIZE, "the pressure FIFO: 128 pending")
            resume_child(s, PRESS)
        self.done(ph)

    def wait_last(self, nodes, what):
        last = {}
        for n in nodes:
            last[n.fid] = n
        for fid, n in last.items():
            self.fifo(fid).wait_stamp(n.stamp, label="node %d (%s)" % (n.id, what))

    def final(self):
        s, g = self.s, self.g
        s.mark("the end")
        nch = len(g.tails)
        s.cmp(s.pa(self.shared, (1 + REG_PAGES) * PAGE), s.pa(self.shared, PAGE), nch * CNT,
              label="the %d chain counters against the control" % nch)
        s.dump(s.pa(s.root_page, abi.ROOT_CHILD_NODE), abi.CHILD_NODE_SIZE * GATE, "child nodes 1-20")
        s.r(abi.R_FIFO_READ, "the end")
        s.fault_state("the end")

    def emit(self):
        self.setup()
        for ph in self.g.phases:
            self.run_phase(ph)
        self.final()
        return self.s


def summary(g):
    ev = {}
    for p in g.phases:
        k = p.event or "none"
        ev[k] = ev.get(k, 0) + 1
    joins = sum(1 for n in g.nodes if len(n.preds) > 1)
    churn = sum(len(p.pre) for p in g.phases)
    return "%s; %d nodes in %d phases (%s), %d chains, %d joins, %d FIFO deletes and defines" % (
        ", ".join(g.tags), len(g.nodes), len(g.phases), ", ".join("%s %d" % kv for kv in sorted(ev.items())),
        len(g.tails), joins, churn)


def build(seed, nodes_hint, doc):
    g = generate(seed, nodes_hint)
    name = "interplay_dag_%d" % seed
    return Emitter(name, doc.strip() + "\n\nThis graph: " + summary(g) + ".", g).emit()
