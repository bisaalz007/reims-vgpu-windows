"""Seeded random guest programs that mix every walked dimension
(scenarios/interplay_mix.py): a walker over what these generators walk
one at a time.

  - pgtest/dagwalk.py: barrier DAGs of GPU work over the root FIFO and
    child FIFOs, whose every byte proves the order the graph implies;
  - pgtest/lifetime_util.py: objects deleted (37, 40), retyped (the
    object-list entry rewritten in place) and made again while work that
    names them is held;
  - pgtest/scanout_util.py: GPU-rendered frames in mapper surfaces
    (IOSurface host device, type 11 mapper textures) presented on displays
    with opcode 8, and task 0's backings with opcode 7;
  - pgtest/randfault.py: injected faults (decode errors, an invalid task,
    unknown opcodes), and the guest's recoveries (a resume, deleting the
    faulted child, a pause, a suspend);
  - pgtest/randgen.py: the lifecycle (pause and unpause, work rung while
    paused, suspend onto a fresh device and resume, rvg_device_will_suspend and
    unpause, reset, stop or a fresh device with the guest's set-up again),
    cursors and child FIFOs deleted and defined again.

A program is a list of rounds. Most rounds are a phase: a DAG of nodes, each
one command on one FIFO, ordered only by barriers {stampIndex, value} on
other nodes' stamps. A node is GPU work (dag: dagwalk's order witness, into
the live region, with a scratch buffer part), a render into a mapper
surface (render: a clear to an exact colour, the triangle or not), a present
(op 8 of a surface, op 7 of a backing), a faulting command (fault), or GPU
work blocked on the faulting command's stamp (a dependent). Some nodes wait
on the gate (child 20) while the guest does the phase's event: a hold, a
pause, a suspend, or lifetime actions on objects the held nodes name, then
opens it. The other rounds are the lifecycle, idle.

One model predicts every observation (the emitter keeps it and writes it
into every label and every expected byte):
  - dag slots and chain counters: the differential control (dagwalk's):
    before the walk, task 1 runs every dag node in creation order on the
    root FIFO, into a control region; after each phase the live slots of
    the phase are compared with it, and at checks and at the end every
    slot so far and the chain counters. A faulting node's slot is compared
    with its control slot, which nothing writes: nothing of a faulted exec
    is committed.
  - surfaces: the model's bytes (the surface's initial pixels, or the last
    render's exact colour: clears of 0, 0.5, 1 and the triangle's (1, 0.5,
    0.25, 1)) are written to a page of the guest's and compared (cmp) with
    the surface's page after every phase that rendered it;
  - scratch buffers: each task's scratch reference names page A or B (its
    object-list entry) and is cached as whichever the first exec that
    listed it made it from. A delete (37 or 40 of op 1000) forgets it, a
    suspend or a teardown forgets every object; a retype rewrites the entry
    and changes nothing until the object is forgotten. Each fill lands in the
    page the model says; both pages are compared with the model's bytes;
  - displays: one new_frame per present, and after a resume one for each
    port whose page has an opcode-8 mapping (the last one, even behind an
    opcode-7 present, docs/testing.md 7.1), and the cursor glyph and show
    replayed on each port that set one;
  - faults: 0x102c waited for at the model's value, 0x103c and every live
    child's node fault offset labelled with the model's state for its FIFO
    (ok, faulted, paused, restored: randfault's), and each dependent held
    until the recovery publishes the faulted stamp;
  - waits: every wait succeeds but those the model says time out, whose
    labels end "[held]" (check_expected reads it back).

Invariants the generator keeps (each earlier generator's docstring gives
the evidence):
  - Each command on a FIFO is published alone (dagwalk): a node whose
    FIFO-previous node is not an ancestor gets a barrier on it, so the
    interrupt counts are exact.
  - Every node that writes what another reads or writes is ordered with it
    by ancestry: nodes of one chain, the scratch users of one task, the
    renders and presents of one surface. At most one present a phase, waited
    for by its new_frame before the phase's event: new_frame calls coalesce
    (docs/testing.md 4.6), and presents on two displays at once call back
    in an order only timing decides.
  - At most 8 children, the root and the gate wait: below the waiter mask's
    32 (docs/testing.md 7.4). Deleting child n zeroes node n+1's
    pageListPage; the guest writes it back at once, so every FIFO start (an
    unpause's, a resume's) maps its ring again.
  - Held work waits on the gate, 300 ms, with everything not held finished
    first, so nothing is in flight at the event. Lifetime actions go on the
    root FIFO, so a phase with them holds nothing on it. No display command
    and no cursor doorbell while paused; pause and unpause pair.
    rvg_device_will_suspend then unpause is taken only idle and only with no
    display (with a guest display it asserts, 7.3; over a blocked FIFO it
    leaves two threads and can kill the host, 7.4), and with a pause between
    them: even idle, rvg_device_will_suspend wakes no FIFO thread, so unpause
    would start a second one on every FIFO (7.3). A fresh device is preceded by
    a host stop: releasing a live device under isolation tears its root down
    during the release, with an outcome that depends on timing (7.3).
  - One fault a phase, the last command on its FIFO, never held; its
    dependents are the last on their child FIFOs, never on the root (the
    guest's recovery uses it), below the faulted FIFO when a suspend
    recovers it (make_fault), and nothing depends on them. A pause over a
    fault leaves the queue's latch set (7.1): the guest resumes the FIFO to
    clear the mask, and the model never faults a latched FIFO again until a
    suspend, a teardown or a delete clears it.
  - Mapper surfaces exist only with a memory map: programs with surfaces run
    in the map and isolate setups; the others in all three.
"""
import json
import struct

from . import abi
from .abi import PAGE
from .compute_objects import Blit as B, Compute as C, compute_blob, compute_pipeline_record, function_record
from .dagwalk import ADD, CNT, GATE, INC, MAX_CHAINS
from .diff_util import exec_payload
from .display_guest import p_present3, p_swap
from .exec_objects import pipeline_record, render_blob
from .fifo import ChildFIFO
from .fifo_util import HEALTHY, NODE_FAULT, NODE_PAGE_LIST, track_stamps
from .lifetime_util import delete_object, delete_resource, rewrite_entry
from .randfault import UNKNOWN_OPS
from .randgen import Rng
from .resources_build import BGRA8Unorm, OPT_SHARED, b_fill_buf, desc1, iosurface_op, mapper_ref_rec
from .scanout_util import FH, FRAME, FW, MAPPED, Scanout, frame_bgra
from .scenario import SETUPS
from .session import DEL_COMPUTE_PSO, DEL_FUNCTION, DEL_RESOURCE
from .task import buffer_record, rec, seg
from .tasks_lib import XGPUTask

E3, E2 = abi.OP_EXEC_INDIRECT3, abi.OP_EXEC_INDIRECT2
VERT = "shaders/vert.metallib"
FRAG = "shaders/frag.metallib"

# A task's own VA pages: 0 the object list, 1-3 records, 4-7 command
# buffers, 8 and 9 the kernels, 10 and 11 the render functions, 12 and 13
# the scratch pages A and B, 14 the control's scratch; then the pages every
# task maps to the same guest pages: SRC, the control region, the live
# region.
PRIVATE = 15
SCR_VPN = {"A": 12, "B": 13, "C": 14}
REG_PAGES = 4
SHARED = 1 + 2 * REG_PAGES
SLOT_BASE = MAX_CHAINS * CNT
SLOT = 0x100
MAX_SLOTS = (REG_PAGES * PAGE - SLOT_BASE) // SLOT
MAX_PREDS = 8                 # slot + 0x80: 8 predecessors' two words
SCR_OFF, SCR_LEN = 0xC0, 0x40
CONTROL_GROUP = 8
MAX_CHILD = 8
HOLD_MS = 300
FILLS = (0x11, 0x5A, 0xC3, 0x3C, 0x96)
COLOURS = ((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0),
           (0.5, 0.5, 0.5, 1.0))
TRIANGLE = bytes([0x40, 0x80, 0xFF, 0xFF]) * (FW * FH)   # tri_frag's (1, 0.5, 0.25, 1), BGRA
DECODE = ("record", "overlong", "ref", "encoder")
HELD = " [held]"              # the suffix of every wait the model says times out


# ---- the program ----

class Features:
    """What a seed's program may use."""

    def __init__(self, seed, rounds):
        r = Rng(seed ^ 0x313A313A313A)
        self.rounds = rounds
        self.ports = r.choice((0, 1, 1, 2, 2))
        self.surfaces = (1 + r.below(3)) if self.ports and r.chance(0.75) else 0
        self.tasks = r.choice((1, 2, 2))
        self.children = r.choice((2, 3, 4, 6, 8))
        self.teardown = r.chance(0.6)
        self.fault_rate = r.choice((0.2, 0.35, 0.5))
        self.nodes = r.choice((4, 6, 9))
        self.gate_p = r.choice((0.3, 0.5, 0.8))
        # A task's service killed by an exec (isolate only): a program with
        # two tasks, so the phase's other work runs on the other one.
        self.death = self.tasks == 2 and r.chance(0.2)

    def setups(self):
        if self.death:
            return ("isolate",)
        return MAPPED if self.surfaces else SETUPS

    def tags(self):
        t = ["ports=%d" % self.ports, "surfaces=%d" % self.surfaces, "tasks=%d" % self.tasks,
             "children=%d" % self.children, "fault rate %g" % self.fault_rate]
        if self.teardown:
            t.append("teardown")
        if self.death:
            t.append("death")
        return t


class Node:
    """One command. kind: dag, render, present, fault. preds: data
    predecessors (nodes with slots: their slot feeds add_one and the
    predecessor words); after: barrier-only predecessors; gate: a barrier on
    the gate. dep: blocked on the phase's fault."""

    def __init__(self, nid, phase, fid, kind, task=0, op=E3):
        self.id, self.phase, self.fid, self.kind, self.task, self.op = nid, phase, fid, kind, task, op
        self.preds = []
        self.after = []
        self.gate = False
        self.bars = []
        self.slot = None
        self.chain = None
        self.fill = None          # dag: the scratch part's byte
        self.render = None        # (mid, colour index or "tri")
        self.present = None       # (port, "map", mid) or (port, "backing", frame)
        self.fault = None         # (kind, variant)
        self.dep = False
        self.circ = None          # a circular wait's first waiter: the faulting node's id
        self.gated = False
        self.stamp = None

    def desc(self):
        bars = ",".join("gate" if b == "gate" else str(b) for b in self.bars)
        k = self.kind
        if k == "dag":
            what = "dag t%d c%d s%d%s%s" % (self.task + 1, self.chain, self.slot,
                                           " fill %02x" % self.fill if self.fill is not None else "",
                                           " op55" if self.op == E2 else "")
        elif k == "render":
            mid, c = self.render
            what = "render t%d surface %d %s" % (self.task + 1, mid, c if c == "tri" else "colour %d" % c)
        elif k == "nop":
            what = "NOP waiting on n%d's stamp" % self.circ
        elif k == "present":
            port, how, v = self.present
            what = "present %s %d on port %d" % ("mapping" if how == "map" else "backing", v, port)
        else:
            what = "FAULT %s/%d s%d" % (self.fault[0], self.fault[1], self.slot)
        return "n%d F%d %s%s%s" % (self.id, self.fid, what, " dep" if self.dep else "", " <" + bars if bars else "")


class Phase:
    def __init__(self, index, event):
        self.index = index
        self.event = event        # None, hold, pause, suspend, ring
        self.nodes = []
        self.anc = {}
        self.life = []
        self.fault = None
        self.deps = []
        self.recovery = None

    def total(self):
        ids = [n.id for n in self.nodes]
        return all(ids[i] in self.anc[ids[j]] for j in range(len(ids)) for i in range(j))


class Program:
    def __init__(self, f):
        self.f = f
        self.nodes = []
        self.tails = []
        self.nslots = 0
        self.acts = []
        self.defined = set(range(1, f.children + 1))
        self.latched = set()

    def dag_nodes(self):
        return [n for n in self.nodes if n.kind == "dag"]


class Gen:
    """The walk's generator: random structure over the legal actions."""

    def __init__(self, seed, rounds):
        self.f = f = Features(seed, rounds)
        self.r = Rng(seed)
        self.p = Program(f)

    # ---- ancestry ----

    def prefix(self, ph, nid):
        n = self.p.nodes[nid]
        return [m.id for m in ph.nodes if m.fid == n.fid and m.id <= nid]

    def ancestors(self, ph, targets):
        out = set()
        for b in targets:
            if self.p.nodes[b].phase != ph.index:
                continue
            for u in self.prefix(ph, b):
                out.add(u)
                out |= ph.anc.get(u, set())
        return out

    def gated(self, ph, anc):
        return any(self.p.nodes[a].gated for a in anc)

    # ---- nodes ----

    def add(self, ph, n, gate=False):
        """Places node n on its FIFO: the FIFO-previous barrier, ancestry,
        gating, a chain for dag nodes, and a shuffled barrier list."""
        r, p = self.r, self.p
        prev = [m for m in ph.nodes if m.fid == n.fid]
        anc = self.ancestors(ph, n.preds + n.after)
        if prev and prev[-1].id not in anc:
            if n.kind == "dag" and prev[-1].slot is not None and len(n.preds) < MAX_PREDS:
                n.preds.append(prev[-1].id)
            else:
                n.after.append(prev[-1].id)
            anc = self.ancestors(ph, n.preds + n.after)
        n.gate = gate
        n.gated = gate or self.gated(ph, anc)
        if n.kind == "dag":
            cont = [c for c, t in enumerate(p.tails)
                    if p.nodes[t].phase < ph.index or t in anc]
            if cont and (r.chance(0.8) or len(p.tails) >= MAX_CHAINS):
                n.chain = r.choice(cont)
            elif len(p.tails) >= MAX_CHAINS:
                n.chain = r.below(len(p.tails))
                n.after.append(p.tails[n.chain])
                anc = self.ancestors(ph, n.preds + n.after)
                n.gated = gate or self.gated(ph, anc)
            else:
                n.chain = len(p.tails)
                p.tails.append(None)
            p.tails[n.chain] = n.id
        if n.kind in ("dag", "fault"):
            n.slot = p.nslots
            p.nslots += 1
            assert p.nslots <= MAX_SLOTS
        bars = [x for x in dict.fromkeys(n.preds + n.after)] + (["gate"] if gate else [])
        n.bars = r.sample(bars, len(bars))
        ph.anc[n.id] = anc
        ph.nodes.append(n)
        p.nodes.append(n)
        return n

    def new(self, ph, fid, kind, **kw):
        return Node(len(self.p.nodes), ph.index, fid, kind, **kw)

    def last_toucher(self, ph, key):
        out = None
        for m in ph.nodes:
            if key == ("scr", m.task) and m.kind == "dag" and m.fill is not None:
                out = m
            if m.kind == "render" and key == ("surf", m.render[0]):
                out = m
            if m.kind == "present" and m.present[1] == "map" and key == ("surf", m.present[2]):
                out = m
        return out

    def order_after(self, ph, n, key, data=False):
        """n after the phase's last node touching key, if that is not
        already an ancestor."""
        last = self.last_toucher(ph, key)
        if last is None:
            return
        if last.id in self.ancestors(ph, n.preds + n.after):
            return
        if data and len(n.preds) < MAX_PREDS:
            n.preds.append(last.id)
        else:
            n.after.append(last.id)

    def pick_fifo(self, ph, n, hot, fifos, gated_ok_root):
        """A FIFO for n: a hot one mostly. A node that will be gated is
        kept off the root FIFO in a phase with lifetime actions."""
        r = self.r
        anc = self.ancestors(ph, n.preds + n.after)
        will_gate = n.gate or self.gated(ph, anc)
        cand = hot if r.chance(0.7) else fifos
        if will_gate and not gated_ok_root:
            cand = [x for x in cand if x != 0] or [x for x in fifos if x != 0]
        return r.choice(cand)

    def phase(self, event):
        r, p, f = self.r, self.p, self.f
        ph = Phase(sum(1 for a in p.acts if a[0] == "phase"), event)
        held = event in ("hold", "pause", "suspend")
        has_life = held and r.chance(0.6)
        with_fault = event in (None, "hold") and r.chance(f.fault_rate)
        # A death phase: the dying task runs nothing else in the phase.
        dt = r.below(2) if with_fault and f.death and r.chance(0.5) else None

        def task():
            return r.below(f.tasks) if dt is None else 1 - dt
        fifos = [0] + sorted(p.defined)
        hot = r.sample(fifos, min(len(fifos), r.choice((2, 3, 5))))
        n = f.nodes - 2 + r.below(5)
        n = min(n, MAX_SLOTS - p.nslots - 4)
        ports_free = list(range(f.ports)) if event != "ring" else []
        for i in range(max(n, 1)):
            table = [("dag", 60)]
            if f.surfaces and event != "ring":
                table.append(("render", 16))
            if ports_free:
                table.append(("present", 10))
            kind = r.weighted(table)
            gate = held and r.chance(f.gate_p)
            here = [m for m in ph.nodes if m.slot is not None]
            if kind == "dag":
                nd = self.new(ph, 0, "dag", task=task(), op=E2 if r.chance(0.2) else E3)
                if here and r.chance(0.85):
                    cnt = 1 if not r.chance(0.2) else 2 + r.below(3)
                    nd.preds = [m.id for m in r.sample(here[-r.choice((1, 2, 4, 50)):], cnt)]
                earlier = [m for m in p.nodes if m.slot is not None and m.phase < ph.index]
                if earlier and r.chance(0.2):
                    nd.preds.append(r.choice(earlier).id)
                nd.preds = list(dict.fromkeys(nd.preds))[:MAX_PREDS - 2]
                if r.chance(0.35):
                    nd.fill = r.choice(FILLS)
                    self.order_after(ph, nd, ("scr", nd.task), data=True)
            elif kind == "render":
                mid = 1 + r.below(f.surfaces)
                nd = self.new(ph, 0, "render", task=task())
                nd.render = (mid, "tri" if r.chance(0.3) else r.below(len(COLOURS)))
                self.order_after(ph, nd, ("surf", mid))
            else:
                # One present a phase: presents on two displays at once
                # call back in either order (interplay_mix_3, _38 before).
                port = r.choice(ports_free)
                ports_free = []
                nd = self.new(ph, 0, "present")
                if f.surfaces and r.chance(0.75):
                    mid = 1 + r.below(f.surfaces)
                    nd.present = (port, "map", mid)
                    self.order_after(ph, nd, ("surf", mid))
                else:
                    nd.present = (port, "backing", r.below(2))
            nd.gate = gate
            nd.fid = self.pick_fifo(ph, nd, hot, fifos, not has_life)
            self.add(ph, nd, gate=gate)
        if held and not any(m.gated for m in ph.nodes):
            nd = self.new(ph, r.choice(sorted(p.defined)), "dag", task=task())
            self.add(ph, nd, gate=True)
        if with_fault:
            self.make_fault(ph, fifos, has_life, dt, task)
        if has_life:
            ph.life = self.life_actions(ph, dt)
        return ph

    def make_fault(self, ph, fifos, has_life, dt, task):
        r, p, f = self.r, self.p, self.f

        def last_free(fid):
            prev = [m for m in ph.nodes if m.fid == fid]
            return not prev or not prev[-1].gated

        cand = [x for x in fifos if x not in p.latched and last_free(x) and not (x == 0 and has_life)]
        if not cand:
            return
        fid = r.choice(cand)
        waiters = [x for x in sorted(p.defined) if x != fid and last_free(x)]
        kinds = list(DECODE) + ["task", "opcode"] + (["circular", "circular"] if waiters else [])
        k = "death" if dt is not None else r.choice(kinds)
        if k in DECODE:
            v = r.below(3)
        elif k == "task":
            v = r.choice((5, 6, 7, 9))
        elif k == "opcode":
            v = r.choice(UNKNOWN_OPS)
        else:
            v = 0
        rec_choices = [("resume", 5), ("pause", 2), ("suspend", 2)]
        if fid:
            rec_choices.append(("delete", 3))
        recovery = r.weighted(rec_choices)
        # Recovered by a suspend, a dependent waits on a child below the
        # faulted FIFO: rvg_device_will_suspend quiesces the FIFOs in index
        # order, and the faulted one's quiesce publishes its head (the stamp
        # queue's quiesce and pump), which wakes the dependent's FIFO; one not
        # yet quiesced runs the dependent on the old device, one already
        # quiesced leaves it to the new one (docs/testing.md 7.3).
        below = [x for x in waiters if x < fid] if recovery == "suspend" else waiters
        A = None
        if k == "circular":
            # The first waiter: a NOP on another child waiting on the
            # faulting NOP's stamp; the faulting NOP then waits on the
            # waiter's: "Circular wait".
            w = [x for x in below if not (recovery == "delete" and x == fid + 1)]
            if not w:
                return
            A = self.new(ph, r.choice(w), "nop")
            A.dep = True
            A.circ = A.id + 1
            self.add(ph, A)
            assert not A.gated
            ph.deps.append(A)
        F = self.new(ph, fid, "fault", task=dt if dt is not None else task(), op=E2 if r.chance(0.2) else E3)
        F.fault = (k, v)
        if A is not None:
            assert F.id == A.circ
            F.after.append(A.id)
        self.add(ph, F)
        assert not F.gated
        ph.fault = F
        ph.recovery = recovery
        used = {A.fid} if A is not None else set()
        others = [x for x in sorted(p.defined) if x != fid and last_free(x) and x not in used
                  and not (ph.recovery == "delete" and x == fid + 1)
                  and not (ph.recovery == "suspend" and x > fid)]
        for g in r.sample(others, r.choice((0, 1, 1, 2))):
            D = self.new(ph, g, "dag", task=task())
            D.preds = [F.id]
            free = [m for m in ph.nodes if m.slot is not None and not m.gated and m is not F and not m.dep]
            if free and r.chance(0.4):
                D.preds.append(r.choice(free).id)
            D.dep = True
            self.add(ph, D)
            assert not D.gated
            ph.deps.append(D)

    def life_actions(self, ph, dt):
        """Lifetime actions while the gated nodes are held: on a task that
        a gated node uses, mostly; never on a task whose service the phase
        kills, where every command faults until the guest defines it again
        (the recovery, after the gate opens)."""
        r, f = self.r, self.f
        gated = [m for m in ph.nodes if m.gated]
        out = []
        for _ in range(1 + r.below(2)):
            t = r.choice(gated).task if gated and r.chance(0.7) else r.below(f.tasks)
            if t == dt:
                t = 1 - dt
            table = [("scr", 10), ("pso", 3), ("fnpso", 2), ("task", 2)]
            if f.surfaces:
                table.append(("mtex", 3))
            k = r.weighted(table)
            if k == "scr":
                out.append(("scr", t, r.choice(("del37", "del40", "retype", "retype", "remake", "remake"))))
            elif k in ("pso", "fnpso"):
                out.append((k, t, r.choice(("inc", "add"))))
            elif k == "task":
                out.append(("task", t))
            else:
                out.append(("mtex", t, 1 + r.below(f.surfaces)))
        return out

    # ---- idle rounds ----

    def idle(self, kind):
        r, p, f = self.r, self.p, self.f
        if kind == "churn":
            acts = []
            live = sorted(p.defined)
            dead = [x for x in range(1, MAX_CHILD + 1) if x not in p.defined]
            for fid in r.sample(live, r.choice((1, 1, 2))):
                if len(p.defined) <= 1:
                    break
                acts.append(("delete", fid))
                p.defined.discard(fid)
                p.latched.discard(fid)
                if r.chance(0.5):
                    acts.append(("define", fid))
                    p.defined.add(fid)
            for fid in dead:
                if r.chance(0.4):
                    acts.append(("define", fid))
                    p.defined.add(fid)
            return ("churn", acts)
        if kind in ("suspend", "teardown"):
            p.latched.clear()
        if kind == "teardown":
            return ("teardown", r.choice(("reset", "stop", "recreate")))
        if kind == "cursor":
            return ("cursor", r.below(f.ports), r.below(FW), r.below(FH))
        if kind == "life":
            k = r.weighted([("scr", 6), ("pso", 1), ("task", 1)] + ([("mtex", 1)] if f.surfaces else []))
            t = r.below(f.tasks)
            if k == "scr":
                return ("life", [("scr", t, r.choice(("del37", "del40", "retype", "remake")))])
            if k == "pso":
                return ("life", [("fnpso" if r.chance(0.5) else "pso", t, r.choice(("inc", "add")))])
            if k == "task":
                return ("life", [("task", t)])
            return ("life", [("mtex", t, 1 + r.below(f.surfaces))])
        return (kind,)

    def generate(self):
        r, p, f = self.r, self.p, self.f
        p.acts.append(("setup",))
        for i in range(f.rounds):
            left = MAX_SLOTS - p.nslots
            table = [("check", 1), ("pause", 2), ("suspend", 2), ("churn", 2), ("life", 2)]
            if left > 16:
                table.append(("phase", 22))
            if f.ports == 0:
                table.append(("will", 1))
            if f.teardown:
                table.append(("teardown", 2))
            if f.ports:
                table.append(("cursor", 2))
            kind = r.weighted(table) if i else "phase"
            if kind == "phase":
                ev = r.weighted([(None, 30), ("hold", 10), ("pause", 12), ("suspend", 10), ("ring", 6)])
                ph = self.phase(ev)
                if ph.fault is not None and ph.recovery == "pause":
                    p.latched.add(ph.fault.fid)
                if ph.event == "suspend" or (ph.fault is not None and ph.recovery == "suspend"):
                    p.latched.clear()
                if ph.fault is not None and ph.recovery == "delete":
                    p.latched.discard(ph.fault.fid)
                p.acts.append(("phase", ph))
            else:
                p.acts.append(self.idle(kind))
        p.acts.append(("final",))
        return p


def generate(seed, rounds):
    return Gen(seed, rounds).generate()


# ---- the tasks ----

class MixTask(XGPUTask):
    """dagwalk's DagTask with more objects, in the same order in every
    task: inc_k's function and pipeline, add_one's, SRC, the control
    region, the live region, the scratch buffer (entry on page A), the
    control's scratch (page C); with surfaces, the render functions, a
    BGRA8Unorm render pipeline and a mapper texture (type 11) over each
    surface. SRC and the regions are
    the same guest pages in every task."""

    def __init__(self, s, tid, shared, surfaces):
        leaves = {v: s.alloc_page() for v in range(PRIVATE)}
        leaves.update({PRIVATE + i: pg for i, pg in enumerate(shared)})
        super().__init__(s, tid=tid, pages=PRIVATE + SHARED, leaves=leaves)
        self.fn_inc, self.pso_inc = self._kernel(INC, 8)
        self.fn_add, self.pso_add = self._kernel(ADD, 9)
        base = PRIVATE * PAGE
        self.src, self.src_va = self.buffer(PAGE, va=base)
        self.reg_c, self.reg_c_va = self.buffer(REG_PAGES * PAGE, va=base + PAGE)
        self.reg_l, self.reg_l_va = self.buffer(REG_PAGES * PAGE, va=base + (1 + REG_PAGES) * PAGE)
        self.scr, _ = self.buffer(PAGE, va=SCR_VPN["A"] * PAGE)
        self.scr_c, _ = self.buffer(PAGE, va=SCR_VPN["C"] * PAGE)
        self.mtex = {}
        if surfaces:
            self.f_vert = self._function(VERT, 10)
            self.f_frag = self._function(FRAG, 11)
            self.bgra = self.ol.add(abi.T_SERIALIZED, pipeline_record(14, render_blob(self.f_vert, self.f_frag,
                                                                                        BGRA8Unorm)))
            d = desc1(fmt=BGRA8Unorm, w=FW, h=FH, usage=1 | 2 | 4, options=OPT_SHARED)
            for mid in range(1, surfaces + 1):
                self.mtex[mid] = self.ol.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(mid) + iosurface_op(12, d))

    def _function(self, fixture, vpn):
        va = self.reserve(0x2000)
        assert va == vpn * PAGE
        n = self.s.mem_file(self.gpa(va), fixture)
        return self.ol.add(abi.T_FUNCTION, function_record(va, n))

    def _kernel(self, fixture, vpn):
        fn = self._function(fixture, vpn)
        return fn, self.ol.add(abi.T_SERIALIZED, compute_pipeline_record(compute_blob(fn)))

    def scr_gpa(self, page):
        return self.gpa(SCR_VPN[page] * PAGE)


def slot_off(slot):
    return SLOT_BASE + slot * SLOT


def dag_segments(p, t, n, reg, scr):
    """dagwalk's node segments (inc_k on the chain's counter, add_one over
    the first predecessor's +0x40 or SRC; the counter block and each
    predecessor's two first words copied into the slot), and the scratch
    part: the scratch buffer filled with the node's byte and copied to
    slot + 0xc0."""
    o = slot_off(n.slot)
    if n.preds:
        src = (reg, slot_off(p.nodes[n.preds[0]].slot) + 0x40)
    else:
        src = (t.src, 0x40 * (n.slot % 4))
    comp = (C.set_pipeline(t.pso_inc) + C.set_buffers(0, [(reg, n.chain * CNT)])
            + C.dispatch_threads((1, 1, 1), (1, 1, 1))
            + C.set_pipeline(t.pso_add) + C.set_buffers(0, [(reg, o + 0x40), src])
            + C.dispatch_threads((16, 1, 1), (16, 1, 1)))
    blit = B.copy_buffer_to_buffer(reg, reg, n.chain * CNT, o, CNT)
    for i, q in enumerate(n.preds):
        po = slot_off(p.nodes[q].slot)
        blit += B.copy_buffer_to_buffer(reg, reg, po, o + 0x80 + 8 * i, 4)
        blit += B.copy_buffer_to_buffer(reg, reg, po + 0x40, o + 0x84 + 8 * i, 4)
    if n.fill is not None:
        blit += b_fill_buf(scr, 0, SCR_LEN, n.fill) + B.copy_buffer_to_buffer(scr, reg, 0, o + SCR_OFF, SCR_LEN)
    return [seg(abi.ENC_COMPUTE, comp), seg(abi.ENC_BLIT, blit)]


# ---- the emitter ----

class Emitter:
    def __init__(self, name, doc, p):
        self.p = p
        f = self.f = p.f
        self.x = x = Scanout(name, doc, ports=max(f.ports, 1), displays=tuple(range(f.ports)),
                             setups=f.setups(), host=f.ports > 0, iosfc=f.surfaces > 0,
                             watchdog_s=90 + 6 * f.rounds)
        self.s = x.s
        self.fifos = {}
        # The model.
        self.ports = {q: dict(glyph=False, mapping=0) for q in range(f.ports)}
        self.surf = {}
        self.scr = [dict(entry="A", cached=None, A=bytes(SCR_LEN), B=bytes(SCR_LEN)) for _ in range(f.tasks)]
        self.fst = {}
        self.defined = set(range(1, f.children + 1))
        self.step = 0

    # ---- helpers ----

    def fifo(self, fid):
        return self.s.fifo if fid == 0 else self.fifos[fid]

    def events(self, label, order=False, irq_stamps=None, collapse=None):
        """irq_stamps (default: order): each interrupt's snapshot of the
        stamp slots. Only one FIFO's may be snapshotted: a FIFO stores its
        stamp and wakes the barriers waiting on it before it raises its
        interrupt, so a FIFO the barrier
        releases can publish its own stamp before that snapshot is taken;
        whether it did is timing (docs/testing.md, decision 8). collapse:
        Scenario.events'."""
        self.s.events(label, order=order, irq_stamps=order if irq_stamps is None else irq_stamps,
                      sort_blocks=True, blocks_digest=True, collapse=collapse)

    def stamps(self, label):
        self.s.dump(self.s.fifo.base_gpa(), 4 * (GATE + 1), label)

    def define(self, fid):
        c = self.fifos.get(fid)
        if c is None:
            c = self.fifos[fid] = ChildFIFO(self.s, fid)
        c.written = 0
        c.setup_node()
        c.define(label="define child %d" % fid)
        return c

    def delete(self, fid, label=None):
        """49 on child fid, which zeroes node fid+1's pageListPage; the guest
        writes it back, so the next start of that FIFO maps its ring."""
        s = self.s
        c = self.fifos[fid]
        c.delete(label=label or "delete child %d" % fid)
        s.dump(c.node_gpa(0), abi.CHILD_NODE_SIZE, "node %d after deleting it" % fid)
        # Node fid+1 without its read offset, which moves while that child runs.
        s.dump(c.node_gpa(0) + abi.CHILD_NODE_SIZE + 8, abi.CHILD_NODE_SIZE - 8, "node %d from +8 after deleting %d" % (fid + 1, fid))
        nxt = self.fifos.get(fid + 1)
        if nxt is not None and (fid + 1) in self.defined:
            s.u32(nxt.node_gpa(NODE_PAGE_LIST), s.pfn(nxt.list_page))

    def model_cmp(self, gpa, expect, label):
        """The model's bytes into the guest's page, then a cmp."""
        self.s.mem(self.s.pa(self.mpage), expect)
        self.s.cmp(gpa, self.s.pa(self.mpage), len(expect), label=label + " against the model")

    def live(self, off):
        return self.s.pa(self.shared, (1 + REG_PAGES) * PAGE + off)

    def ctrl(self, off):
        return self.s.pa(self.shared, PAGE + off)

    # ---- the fault registers ----

    def observe(self, label):
        s = self.s
        mask = 1 if self.fst else 0
        s.wait(mmio=abi.R_FAULT_MASK, mmio_eq=mask, label="%s: 0x102c %d (model)" % (label, mask))
        s.r(abi.R_FAULT_OFFSET, label="%s: root %s" % (label, self.fst.get(0, "ok")))
        for fid in sorted(self.defined):
            s.dump(self.fifos[fid].node_gpa(NODE_FAULT), 4, "%s: child %d %s" % (label, fid, self.fst.get(fid, "ok")))

    def wait_fault(self, fid, label):
        s = self.s
        if fid == 0:
            s.wait(mmio=abi.R_FAULT_OFFSET, mmio_ne=HEALTHY, label=label + ": 0x103c")
        else:
            s.wait(gpa=self.fifos[fid].node_gpa(NODE_FAULT), ne=HEALTHY, label=label + ": node fault offset")
        s.wait(mmio=abi.R_FAULT_MASK, mmio_ne=0, label=label + ": 0x102c")

    def resume_fifo(self, fid):
        if fid == 0:
            self.s.resume_root()
        else:
            self.s.w(abi.R_RESUME_CHILD, fid)

    # ---- set-up ----

    def emit(self):
        for a in self.p.acts:
            getattr(self, "a_" + a[0])(*a[1:])
            self.step += 1
        return self.s

    def a_setup(self):
        x, s, f = self.x, self.s, self.f
        s.mark("set-up")
        x.negotiate(asks=())
        x.root()
        track_stamps(s, GATE + 1)
        if f.ports:
            x.kernel_task(frames=2)
            for q in range(f.ports):
                x.connect(q)
        if f.surfaces:
            x.iosfc_start()
            for mid in range(1, f.surfaces + 1):
                seed = 0x10 * mid + 3
                x.surface(mid, seed)
                self.surf[mid] = bytes(((i * 5 + seed) & 0xFF) if i % 4 != 3 else 0xFF for i in range(FRAME))
        for fid in sorted(self.defined):
            self.define(fid)
        self.gate = self.define(GATE)
        self.shared = s.alloc_page(SHARED)
        self.mpage = s.alloc_page()
        self.tasks = [MixTask(s, tid, [self.shared + i for i in range(SHARED)], f.surfaces)
                      for tid in range(1, f.tasks + 1)]
        t = self.tasks[0]
        t.write(t.src_va, struct.pack("<256I", *[0x1000 * (i // 16 + 1) + i for i in range(256)]))
        for t in self.tasks:
            for pg in ("A", "B"):
                s.fill(t.scr_gpa(pg), SCR_LEN, 0)
            t.define(label="define task %d" % t.tid)
            t.install(label="task %d's object list" % t.tid)
        self.control()
        self.events("set up")

    def control(self):
        s, p = self.s, self.p
        t = self.tasks[0]
        dag = p.dag_nodes()
        for i in range(0, len(dag), CONTROL_GROUP):
            grp = dag[i:i + CONTROL_GROUP]
            segs = []
            for n in grp:
                segs += dag_segments(p, t, n, t.reg_c, t.scr_c)
            cbs = [t.cmdbuf(sg) for sg in segs]
            s.submit(E3, exec_payload(t.tid, cbs, [(t.src, 0), (t.reg_c, 0), (t.scr_c, 0)]),
                     label="control: nodes %d-%d" % (grp[0].id, grp[-1].id))
            t.reset_cmdbufs()
        nch = len(p.tails)
        s.dump(self.ctrl(0), nch * CNT, "control: the %d chain counters" % nch)
        if not dag:
            return
        widest = max(dag, key=lambda n: (len(n.preds), n.fill is not None, n.id))
        for n in sorted({dag[-1].id, widest.id}):
            s.dump(self.ctrl(slot_off(p.nodes[n].slot)), SLOT, "control: node %d's slot" % n)

    # ---- the model's effects ----

    def ran(self, nodes):
        """What nodes that ran did, in creation order (every two that touch
        one thing are ordered by ancestry, so creation order is theirs)."""
        for n in sorted(nodes, key=lambda n: n.id):
            if n.kind == "dag" and n.fill is not None:
                m = self.scr[n.task]
                pg = m["cached"] or m["entry"]
                m["cached"] = pg
                m[pg] = bytes([n.fill]) * SCR_LEN
            elif n.kind == "render":
                mid, c = n.render
                self.surf[mid] = TRIANGLE if c == "tri" else frame_bgra(COLOURS[c])
            elif n.kind == "present" and n.present[1] == "map":
                self.ports[n.present[0]]["mapping"] = n.present[2]

    def forget_all(self):
        for m in self.scr:
            m["cached"] = None

    def life(self, actions, label):
        """Lifetime actions on the root FIFO, and the model's view of them."""
        s = self.s
        for a in actions:
            k, ti = a[0], a[1]
            t = self.tasks[ti]
            m = self.scr[ti]
            if k == "scr":
                how = a[2]
                if how in ("del37", "remake"):
                    op, pl = delete_resource(t.tid, t.scr)
                    s.submit(op, pl, label="%s: 37 task %d's scratch" % (label, t.tid))
                    m["cached"] = None
                elif how == "del40":
                    op, pl = delete_object(t.tid, DEL_RESOURCE, t.scr)
                    s.submit(op, pl, label="%s: 40 task %d's scratch" % (label, t.tid))
                    m["cached"] = None
                if how in ("retype", "remake"):
                    to = "B" if m["entry"] == "A" else "A"
                    rewrite_entry(t, t.scr, abi.T_BUFFER, record=buffer_record(PAGE, SCR_VPN[to]))
                    m["entry"] = to
                s.mark("%s: task %d's scratch %s: entry %s, cached %s" % (label, t.tid, how, m["entry"],
                                                                         m["cached"] or "nothing"))
            elif k in ("pso", "fnpso"):
                pso, fn = (t.pso_inc, t.fn_inc) if a[2] == "inc" else (t.pso_add, t.fn_add)
                op, pl = delete_object(t.tid, DEL_COMPUTE_PSO, pso)
                s.submit(op, pl, label="%s: 40 task %d's %s pipeline" % (label, t.tid, a[2]))
                if k == "fnpso":
                    op, pl = delete_object(t.tid, DEL_FUNCTION, fn)
                    s.submit(op, pl, label="%s: 40 task %d's %s function" % (label, t.tid, a[2]))
            elif k == "task":
                t.delete(label="%s: delete task %d" % (label, t.tid))
                t.define(label="%s: define task %d again" % (label, t.tid))
                t.install(label="%s: task %d's object list again" % (label, t.tid))
                m["cached"] = None
            else:
                op, pl = delete_resource(t.tid, t.mtex[a[2]])
                s.submit(op, pl, label="%s: 37 task %d's mapper texture %d" % (label, t.tid, a[2]))

    # ---- lifecycle ----

    def suspend_resume(self, label):
        """rvg_device_will_suspend, rvg_device_finish_suspend, a fresh device,
        rvg_device_will_resume, rvg_device_did_resume; then the replays the
        model predicts."""
        x = self.x
        x.suspend(label=label)
        x.resume(label=label)
        glyphs = [q for q, m in sorted(self.ports.items()) if m["glyph"]]
        frames = [q for q, m in sorted(self.ports.items()) if m["mapping"]]
        if glyphs:
            x.expect("cursorGlyph", n=len(glyphs), label="%s: glyphs replayed on ports %s" % (label, glyphs))
            x.expect("cursorShow", n=len(glyphs), label="%s: shows replayed on ports %s" % (label, glyphs))
        if frames:
            x.expect("newFrame", n=len(frames), label="%s: mappings %s presented again on ports %s" % (
                label, [self.ports[q]["mapping"] for q in frames], frames))
        self.forget_all()

    # ---- phases ----

    def write_node(self, n):
        p = self.p
        f = self.fifo(n.fid)
        want = n.stamp
        self._write(n, f)
        assert want is None or want == n.stamp, (n.desc(), want, n.stamp)

    def _write(self, n, f):
        p = self.p
        bars = []
        for b in n.bars:
            if b == "gate":
                bars.append((GATE, self.gate.stamp + 1))
            else:
                q = p.nodes[b]
                bars.append((self.fifo(q.fid).stamp_index, q.stamp))
        t = self.tasks[n.task]
        if n.kind == "dag":
            res = [(t.src, 0), (t.reg_l, 0)] + ([(t.scr, 0)] if n.fill is not None else [])
            cbs = [t.cmdbuf(sg) for sg in dag_segments(p, t, n, t.reg_l, t.scr)]
            n.stamp = f.cmd(n.op, exec_payload(t.tid, cbs, res, n.op), barriers=bars)
        elif n.kind == "render":
            mid, c = n.render
            segs = Scanout.compose(t, t.mtex[mid], COLOURS[4] if c == "tri" else COLOURS[c], draw=c == "tri")
            cbs = [t.cmdbuf(sg) for sg in segs]
            n.stamp = f.cmd(E3, exec_payload(t.tid, cbs, [(t.mtex[mid], 0)]), barriers=bars)
        elif n.kind == "nop":
            F = p.nodes[n.circ]
            bars.append((self.fifo(F.fid).stamp_index, F.stamp))
            n.stamp = f.cmd(abi.OP_NOP, barriers=bars)
        elif n.kind == "present":
            port, how, v = n.present
            if how == "map":
                n.stamp = f.cmd(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(port, v), barriers=bars)
            else:
                n.stamp = f.cmd(abi.OP_DISPLAY_TRANSACTION3, p_present3(port, self.x.backings[v][0]), barriers=bars)
        else:
            op, payload = self.bad_command(n, t)
            n.stamp = f.cmd(op, payload, barriers=bars)

    def bad_command(self, n, t):
        """randfault's faulting commands: a good blit into the node's slot,
        then the error (DECODE kinds); an exec of a task that does not
        exist; an unknown or deprecated opcode."""
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
        if k == "circular":
            return abi.OP_NOP, b""
        if k == "death":
            # A range starting at the task's end: the service dereferences a
            # null cursor and dies.
            return E3, exec_payload(t.tid, [(t.npages * PAGE, 0x40)], [(t.src, 0), (t.reg_l, 0)])
        return v, struct.pack("<II", 0x0BAD0000 | v, n.id)

    def wait_last(self, nodes, what):
        last = {}
        for n in nodes:
            last[n.fid] = n
        for fid, n in sorted(last.items()):
            self.fifo(fid).wait_stamp(n.stamp, label="n%d (%s)" % (n.id, what))

    def frames(self, nodes, label):
        """The new_frame of each present that ran (one per port a phase)."""
        k = sum(1 for n in nodes if n.kind == "present")
        if k:
            self.x.expect("newFrame", n=k, label="%s: %d presents" % (label, k))

    def circular(self, ph):
        """Whether the phase's fault is a circular wait, whose two NOPs are
        written only when the rest has run."""
        return ph.fault is not None and ph.fault.fault[0] == "circular"

    def fault_now(self, ph, lab):
        """A circular wait's two NOPs, in order: the waiter first, seen
        blocked, then the NOP that closes the circle; then the fault."""
        if self.circular(ph):
            F = ph.fault
            A = next(d for d in ph.deps if d.kind == "nop")
            self.write_node(A)
            self.fifo(A.fid).doorbell()
            self.fifo(A.fid).wait_stamp(A.stamp, timeout_ms=HOLD_MS,
                                        label="%s: n%d waits on n%d's stamp%s" % (lab, A.id, F.id, HELD))
            self.write_node(F)
            self.fifo(F.fid).doorbell()
        self.fault_seen(ph, lab)

    def fault_seen(self, ph, lab):
        F = ph.fault
        self.wait_fault(F.fid, lab)
        for D in ph.deps:
            self.fifo(D.fid).wait_stamp(D.stamp, timeout_ms=HOLD_MS,
                                        label="%s: n%d blocked on the faulted stamp%s" % (lab, D.id, HELD))
        self.fst[F.fid] = "faulted"
        self.observe(lab)

    def recover(self, ph, lab):
        s, F = self.s, ph.fault
        how = ph.recovery
        s.mark("%s: recovery by %s" % (lab, how))
        fb = self.fifo(F.fid)
        if how == "resume":
            self.resume_fifo(F.fid)
            fb.wait_stamp(F.stamp, label="%s: the resume publishes the faulted stamp" % lab)
            del self.fst[F.fid]
        elif how == "delete":
            self.delete(F.fid, label="%s: delete faulted child %d" % (lab, F.fid))
            fb.wait_stamp(F.stamp, label="%s: the delete publishes the faulted stamp" % lab)
            del self.fst[F.fid]
            self.define(F.fid)
        elif how == "pause":
            s.api("pause", label=lab)
            s.api("unpause", label=lab)
            fb.wait_stamp(F.stamp, label="%s: the pause publishes the faulted stamp" % lab)
            self.fst[F.fid] = "paused"
        else:
            self.suspend_resume(lab)
            fb.wait_stamp(F.stamp, label="%s: finishSuspend publishes the faulted stamp" % lab)
            self.fst[F.fid] = "restored"
        self.wait_last(ph.deps, lab + ": released")
        self.ran(ph.deps)
        # A suspend releases the dependents in rvg_device_finish_suspend, when
        # the old device publishes the faulted stamp and interrupts nothing (an
        # interrupt is raised only in the Running run state), and they run on
        # the old device or on the new one after rvg_device_did_resume: how
        # many interrupts the recovery raises is timing (3 or 2 in
        # interplay_mix_44), so they are counted collapsed. The other
        # recoveries publish while running.
        self.events(lab + ": recovered by " + how, collapse=["irq"] if how == "suspend" else None)
        self.observe(lab + ": after the " + how)
        if how in ("pause", "suspend"):
            self.resume_fifo(F.fid)
            del self.fst[F.fid]
            self.observe(lab + ": after the " + how + ", resumed")
            self.events(lab + ": resumed")
        if F.fault[0] == "death":
            # Every later command on the task faults: the guest deletes it
            # and defines it again, which gives it a new service.
            t = self.tasks[F.task]
            s.mark("%s: task %d's service died: delete it, define it again" % (lab, t.tid))
            t.delete(label="%s: delete dead task %d" % (lab, t.tid))
            t.define(label="%s: define task %d again" % (lab, t.tid))
            t.install(label="%s: task %d's object list again" % (lab, t.tid))
            self.scr[F.task]["cached"] = None

    def a_phase(self, ph):
        s, x, p = self.s, self.x, self.p
        lab = "phase %d" % ph.index
        s.mark("%s: %s%s%s; %s" % (
            lab, ph.event or "no event", "; life %s" % ph.life if ph.life else "",
            "; fault on FIFO %d, recovery %s" % (ph.fault.fid, ph.recovery) if ph.fault else "",
            "; ".join(n.desc() for n in ph.nodes)))
        blocked = set(n.id for n in ph.deps) | ({ph.fault.id} if ph.fault else set())
        if ph.event == "ring":
            s.api("pause", label=lab)
        late = {n.id for n in ph.nodes if self.circular(ph) and (n is ph.fault or n.kind == "nop")}
        for n in ph.nodes:
            if n.id in late:
                n.stamp = self.fifo(n.fid).stamp + 1
            else:
                self.write_node(n)
        fids = list(dict.fromkeys(n.fid for n in ph.nodes if n.id not in late))
        for fid in reversed(fids):
            self.fifo(fid).doorbell()
        pending = [n for n in ph.nodes if n.id not in blocked]
        seen_fault = False
        if ph.event == "ring":
            n0 = ph.nodes[0]
            self.fifo(n0.fid).wait_stamp(n0.stamp, timeout_ms=HOLD_MS,
                                         label="%s: n%d rung while paused, not run%s" % (lab, n0.id, HELD))
            s.api("unpause", label=lab)
        elif ph.event in ("hold", "pause", "suspend"):
            free = [n for n in pending if not n.gated]
            held = [n for n in pending if n.gated]
            self.wait_last(free, "not held")
            self.frames(free, lab + ": not held")
            self.ran(free)
            h0 = held[0]
            self.fifo(h0.fid).wait_stamp(h0.stamp, timeout_ms=HOLD_MS, label="%s: n%d gated%s" % (lab, h0.id, HELD))
            if ph.fault is not None:
                self.fault_now(ph, lab + ": fault")
                seen_fault = True
            self.events(lab + ": held", order=False)
            self.stamps(lab + ": stamps while held")
            if ph.life:
                self.life(ph.life, lab)
            if ph.event == "pause":
                s.api("pause", label=lab)
                s.api("unpause", label=lab)
                self.events(lab + ": paused and unpaused over held work")
            elif ph.event == "suspend":
                self.suspend_resume(lab)
                self.events(lab + ": suspended and resumed over held work")
                s.sweep(0x1000, 0x1040, label=lab + ": registers after the resume")
            self.gate.submit(abi.OP_NOP, label="%s: the gate opens" % lab)
            pending = held
        self.wait_last(pending, "done")
        self.frames(pending, lab + ": done")
        self.ran(pending)
        if ph.fault is not None and not seen_fault:
            self.fault_now(ph, lab + ": fault")
        simple = ph.fault is None and all(n.kind == "dag" for n in ph.nodes)
        order = simple and ph.total()
        self.events(lab + " done", order=order, irq_stamps=order and len({n.fid for n in ph.nodes}) == 1)
        self.stamps(lab + ": stamps")
        if ph.fault is not None:
            self.recover(ph, lab + ": fault")
            self.stamps(lab + ": stamps after the recovery")
        self.check_phase(ph, lab)

    def check_phase(self, ph, lab):
        s, p = self.s, self.p
        slots = [n.slot for n in ph.nodes if n.slot is not None]
        if slots:
            a, b = min(slots), max(slots)
            s.cmp(self.live(slot_off(a)), self.ctrl(slot_off(a)), SLOT * (b - a + 1),
                  label="%s: slots %d-%d against the control" % (lab, a, b))
        mids = sorted({n.render[0] for n in ph.nodes if n.kind == "render"})
        for mid in mids:
            self.model_cmp(s.pa(self.x.surfaces[mid]), self.surf[mid], "%s: surface %d" % (lab, mid))
        touched = sorted({n.task for n in ph.nodes if n.kind == "dag" and n.fill is not None}
                         | {a[1] for a in ph.life if a[0] == "scr"})
        for ti in touched:
            self.scratch_cmp(ti, lab)
        ports = sorted({n.present[0] for n in ph.nodes if n.kind == "present"})
        for q in ports:
            self.x.encode(q, "%s: port %d" % (lab, q))
        for t in self.tasks:
            t.reset_cmdbufs()

    def scratch_cmp(self, ti, lab):
        t, m = self.tasks[ti], self.scr[ti]
        for pg in ("A", "B"):
            self.model_cmp(t.scr_gpa(pg), m[pg], "%s: task %d's scratch page %s (entry %s, cached %s)" % (
                lab, t.tid, pg, m["entry"], m["cached"] or "nothing"))

    # ---- idle rounds ----

    def a_churn(self, acts):
        self.s.mark("%d: child FIFOs: %s" % (self.step, ", ".join("%s %d" % a for a in acts)))
        for k, fid in acts:
            if k == "delete":
                self.defined.discard(fid)
                self.delete(fid)
            else:
                self.define(fid)
                self.defined.add(fid)

    def a_pause(self):
        s = self.s
        s.mark("%d: pause, unpause" % self.step)
        s.api("pause")
        s.api("unpause")
        self.events("paused and unpaused")

    def a_suspend(self):
        s = self.s
        s.mark("%d: suspend, resume on a fresh device" % self.step)
        self.suspend_resume("suspend %d" % self.step)
        self.events("resumed")
        s.sweep(0x1000, 0x1040, label="registers after the resume")
        self.observe("resumed")

    def a_will(self):
        s = self.s
        # The pause ends the FIFO threads rvg_device_will_suspend left waiting
        # before unpause starts new ones (docs/testing.md 7.3;
        # pgtest/randgen.py).
        s.mark("%d: willSuspend, then pause and unpause (no display)" % self.step)
        s.api("willSuspend")
        s.api("pause")
        s.api("unpause")
        self.events("suspend abandoned")

    def a_cursor(self, port, cx, cy):
        m = self.ports[port]
        self.s.mark("%d: cursor on port %d at %d, %d%s" % (self.step, port, cx, cy,
                                                           "" if m["glyph"] else " with its glyph"))
        self.x.cursor(port, cx, cy, glyph=not m["glyph"])
        m["glyph"] = True

    def a_life(self, actions):
        lab = "%d: lifetime" % self.step
        self.s.mark("%s %s, nothing held" % (lab, actions))
        self.life(actions, lab)
        for ti in sorted({a[1] for a in actions if a[0] == "scr"}):
            self.scratch_cmp(ti, lab)

    def a_check(self):
        s, p = self.s, self.p
        s.mark("%d: checkpoint" % self.step)
        self.events("checkpoint")
        self.observe("checkpoint")
        if p.nslots:
            n = max(n.slot for n in p.nodes if n.slot is not None and n.stamp is not None) + 1
            s.cmp(self.live(slot_off(0)), self.ctrl(slot_off(0)), SLOT * n,
                  label="checkpoint: slots 0-%d against the control" % (n - 1))
        for mid in sorted(self.surf):
            self.model_cmp(s.pa(self.x.surfaces[mid]), self.surf[mid], "checkpoint: surface %d" % mid)
        for ti in range(self.f.tasks):
            self.scratch_cmp(ti, "checkpoint")

    def a_teardown(self, how):
        x, s, f = self.x, self.s, self.f
        s.mark("%d: %s; the guest does its set-up again" % (self.step, how))
        if how == "recreate":
            # A live device is stopped before its release (docs/testing.md
            # 7.3; pgtest/randgen.py).
            s.api("stop")
        s.api(how)
        self.events(how)
        s.sweep(0x1000, 0x1040, label="registers after " + how)
        if how != "stop":
            s.w(abi.R_VERSION, abi.MAX_VERSION)
        s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
        s.fifo.written = 0
        s.fifo.enable()
        for fid in sorted(self.defined):
            self.define(fid)
        self.define(GATE)
        if x.t0 is not None:
            x.t0.define(label="define task 0 again")
            x.t0.install(label="task 0's object list again")
        for t in self.tasks:
            t.define(label="define task %d again" % t.tid)
            t.install(label="task %d's object list again" % t.tid)
        x.pages = {}
        for q in range(f.ports):
            x.connect(q)
            self.ports[q] = dict(glyph=False, mapping=0)
        self.forget_all()
        self.fst.clear()
        self.events("set up again")
        self.observe("set up again")

    def a_final(self):
        x, s, p = self.x, self.s, self.p
        s.mark("the end")
        self.events("end of the walk")
        self.observe("end of the walk")
        nch = len(p.tails)
        s.cmp(self.live(0), self.ctrl(0), nch * CNT, label="the %d chain counters against the control" % nch)
        s.cmp(self.live(slot_off(0)), self.ctrl(slot_off(0)), SLOT * p.nslots,
              label="every slot, 0-%d, against the control" % (p.nslots - 1))
        for mid in sorted(self.surf):
            self.model_cmp(s.pa(x.surfaces[mid]), self.surf[mid], "the end: surface %d" % mid)
        self.stamps("stamp slots")
        s.dump(s.pa(s.root_page, abi.ROOT_CHILD_NODE), abi.CHILD_NODE_SIZE * GATE, "child nodes 1-20")
        for t in self.tasks:
            t.delete(label="delete task %d" % t.tid)
        for q in sorted(x.pages):
            x.disconnect(q)
        if x.t0 is not None:
            x.t0.delete(label="delete task 0")
        for fid in sorted(self.defined):
            self.fifos[fid].delete(label="delete child %d" % fid)
        self.gate.delete(label="delete the gate")
        s.submit(abi.OP_NOP, label="last NOP")
        self.events("torn down")
        s.fault_state("torn down")
        s.api("release")
        self.events("released")


def summary(p):
    hist = {}
    for a in p.acts[1:-1]:
        k = a[0]
        if k == "phase":
            k = "phase(%s)" % (a[1].event or "-")
        hist[k] = hist.get(k, 0) + 1
    kinds = {}
    for n in p.nodes:
        kinds[n.kind] = kinds.get(n.kind, 0) + 1
    faults = [a[1] for a in p.acts if a[0] == "phase" and a[1].fault is not None]
    life = sum(len(a[1].life) for a in p.acts if a[0] == "phase")
    return "%s; %d rounds: %s; nodes: %s; faults: %s; %d lifetime actions while held" % (
        ", ".join(p.f.tags()), len(p.acts) - 2, ", ".join("%s %d" % kv for kv in sorted(hist.items())),
        ", ".join("%s %d" % kv for kv in sorted(kinds.items())),
        ", ".join("%s on %d by %s" % (ph.fault.fault[0], ph.fault.fid, ph.recovery) for ph in faults) or "none",
        life)


def build(seed, rounds, doc):
    p = generate(seed, rounds)
    name = "interplay_mix_%d" % seed
    return Emitter(name, doc.strip() + "\n\nThis program: " + summary(p) + ".", p).emit()


# ---- reading the expected files back ----

def check_expected(path):
    """The model against what the expected file records: every wait
    succeeded but those whose label ends [held], which must time out; every
    cmp is equal; every fault register a label gives a state for reads
    0xffffffff exactly when the state is ok. Returns a list of complaints."""
    bad = []
    for line in open(path):
        if not line.startswith("{"):
            continue
        o = json.loads(line)
        lab = o.get("label") or ""
        ev = o.get("ev")
        if ev == "wait":
            if lab.endswith(HELD.strip()) or lab.endswith(HELD):
                if o.get("ok") is not False:
                    bad.append("wait did not time out: %s" % lab)
            elif o.get("ok") is False:
                bad.append("wait failed: %s" % lab)
        elif ev == "cmp" and not o.get("equal"):
            bad.append("cmp differs: %s (%s)" % (lab, {k: v for k, v in o.items() if k not in ("ev", "label")}))
        elif ev in ("died", "watchdog", "error"):
            bad.append("%s: %s" % (ev, line.strip()[:200]))
        state = None
        for st in ("ok", "faulted", "paused", "restored"):
            if lab.endswith(" " + st):
                state = st
        if state is None:
            continue
        if ev == "mmio_r":
            val = int(o["val"], 16)
        elif ev == "dump" and "hex" in o:
            val = struct.unpack("<I", bytes.fromhex(o["hex"])[:4])[0]
        else:
            continue
        if (val == HEALTHY) != (state == "ok"):
            bad.append("%s: 0x%x" % (lab, val))
    return bad
