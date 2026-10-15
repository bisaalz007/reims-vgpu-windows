"""Seeded random guest programs with faults (scenarios/interplay_faultwalk.py):
pgtest/randgen.py's walk over legal multi-subsystem work, with guest errors
injected at random points and recovered from as a guest driver would.

randgen.py offers only legal actions. Here the same kinds of work (tasks
and scratch objects made and deleted, blit, compute and render items on the
root FIFO and child FIFOs, serialized by barriers or at once, work held
behind the gate, displays, pause, suspend onto a fresh device, reset, stop
and a fresh device) go on around faults, and the model predicts exactly
which FIFO faults, what it holds back, and what survives:

  - faulting execs (67 or 55), decode errors inside an otherwise good
    command buffer: an unknown blit record (299, 324, 0: "Unrecognized
    command"), a record longer than its segment ("Failed to read"), a
    reference past the object list or 0, an unknown encoder type
    (blit_err_framing, exec_encoder_types);
  - an out-of-range task VA: GetComputeInfo (59) with its reply at the
    task's last 8 bytes ("Address out of range for task", the service
    "Failed to make mapping"; compute_info_errors);
  - an exec naming a task that does not exist, never defined or deleted
    ("Invalid task");
  - an opcode the FIFO does not know, or a deprecated one;
  - a circular barrier wait between a child (below 32) and another FIFO:
    the second waiter throws "Circular wait" and faults, and the first
    stays blocked until the
    faulted stamp is published;
  - in the isolate setup, a service death: an exec range past the task's
    end, a null cursor the service dereferences (exec_null_cursor_past_end,
    interplay_iso_death_beside_healthy). Every
    later command on that task faults (docs/testing.md, section 7.4); the
    guest deletes it (32) and defines it again over the same memory, which
    gives it a new service, and carries on.

Faults happen on the root FIFO and on child FIFOs, alone or with other
work: items on other FIFOs run at once with the faulting command, an item
held behind the gate across it, and a dependent item on another child FIFO
whose barrier names the faulting command's stamp, which stays blocked
until that stamp is published. The guest resumes a faulted FIFO (0x1024,
0x1028 = id) at a random later point, or deletes the faulted child, and
sometimes pauses and unpauses or suspends and resumes first.

What the model keeps, from the code and from the scenarios that measured it:
  - A reported fault: the FIFO's fault offset (0x103c for the root, node
    +8 for a child) leaves 0xffffffff at once; bit stampIndex of the fault
    mask (0x102c reads it as a boolean) follows asynchronously, on the FIFO
    queue (reims-vgpu-core's DeviceCore::signal_fault_in_fifo), so a fault
    is waited for by its offset, then by 0x102c. The stamp is held until a
    resume, which publishes it and releases what waits on it
    (fifo::stamps::StampQueue).
  - A pause publishes a faulted head as if signalled but leaves 0x102c,
    the fault offset and the queue's fault latch set (the stamp queue's
    pump; fifo_gap_fault_latch_after_pause): what
    waited on the stamp runs after the unpause; the guest's resume then
    finds no faulted head and clears only the mask, so the FIFO's next
    fault is silent (no offset, no 0x102c, its stamp and the NOP behind it
    held) until a second resume ("paused" below, then "latched").
  - rvg_device_finish_suspend drains a faulted head too; the fresh device
    restores the fault as registers only (key 19, the node's fault offset;
    suspend_resume_faults, suspend_gap_restored_fault): the queue is new, the
    latch gone, and a resume clears the mask ("restored").
  - Deleting a child FIFO clears its fault bit and publishes its faulted
    head (DeviceCore::delete_fifo; interplay_fifo_delete_faulted_releases).
    Waking a deleted child does nothing (DeviceCore::wake_fifo).
  - A resume of a FIFO that is not faulted logs, and for a child also
    reports stamp waits, and raises an interrupt when another FIFO is
    faulted (DeviceCore::resume_fifo and resume_child_fifo).
  - A faulted FIFO takes no new work until the guest resumes it, and a
    FIFO blocked on a faulted stamp none until that is published; while the
    root FIFO is faulted only child FIFOs, cursor moves (0x1220), pause,
    suspend and resumes are used. Faults, dependents and circular waits use
    only child FIFOs whose page list is intact (randgen's rule: a delete
    zeroes the next child's page list), and a child is not deleted while
    it or the next one is blocked or faulted.
  - A task's command buffers are reused only after everything written in
    them ran: a task with a blocked item is busy.
  - Deleted scratch buffers (40) stay in their task's resource list and
    may be named again: the next exec makes them again from their record
    (resources_delete_objects), so the item still equals its control.

The check is randgen's control run for every item that runs
(companions, held and dependent items included), plus, at every fault, a
resume and every checkpoint: 0x102c (waited for at the model's value, so a
wrong model shows as a wait that fails), 0x103c and every live child's node
fault offset, each labelled with the state the model gives its FIFO
(check_expected() reads them back). A faulting exec's slot is compared with
its control slot, which nothing wrote: nothing of a faulted exec is
committed. Programs with a service death run in the isolate setup only.

Nothing here is imported from the library's code.
"""
import json
import struct

from . import abi
from .abi import PAGE
from .diff_util import exec_payload
from .randgen import BYTES, CLEARS, COMPACT, GATE, MAX_CHILD, SLOT, SLOTS, Emitter, Item, Model, Rng, TaskM
from .resources_build import b_fill_buf
from .session import FH, FW
from .task import rec, seg

HEALTHY = 0xFFFFFFFF
NODE_READ, NODE_FAULT = 4, 8
DECODE = ("record", "overlong", "ref", "encoder")
SYNC = ("opcode", "task")        # faults the FIFO thread records before its next command
UNKNOWN_OPS = (13, 99, 29, 70, 3, 31)   # 3 and 31 are deprecated
HOLD_MS = 300                   # a wait expected to time out: the stamp is held


class FFeatures:
    """What a seed's program may use. No IOSurface host device, no info
    work and no hot plug: the faults are the point; displays stay."""

    def __init__(self, seed, steps):
        r = Rng(seed ^ 0xFA17FA17FA17)
        self.compact = steps > COMPACT
        self.ports = r.choice((0, 0, 1, 1, 2))
        self.iosfc = self.info = self.plug = False
        self.death = r.chance(0.35)
        self.concurrent = r.chance(0.7)
        self.teardown = r.chance(0.5)
        self.scratch = r.chance(0.7)
        self.device_info = r.chance(0.3)
        self.held = r.chance(0.5)
        self.fault_rate = r.choice((10, 16, 24))

    def setups(self):
        return ("isolate",) if self.death else ("blocks", "map", "isolate")

    def tags(self):
        t = ["ports=%d" % self.ports, "fault rate %d" % self.fault_rate]
        for k in ("death", "concurrent", "teardown", "scratch", "device_info", "held", "compact"):
            if getattr(self, k):
                t.append(k)
        return t


class Fault:
    """One injected error on FIFO `fid`. kind: DECODE kinds, "task_va",
    "task", "opcode", "death", "circular". item: the faulting exec's slot
    and good parts (an Item that is not in the program's items). comp:
    items run at once on other FIFOs; held: an item held behind the gate
    across the fault; dep: an item on another child gated on the faulting
    command's stamp; waiter: circular's first waiter, blocked on it.
    silent: the FIFO is latched, so the fault is not reported, and the
    guest, which knows, resumes it at once."""

    def __init__(self, n, fid, kind):
        self.n, self.fid, self.kind = n, fid, kind
        self.item = None
        self.key = None
        self.variant = 0
        self.comp = []
        self.held = None
        self.dep = None
        self.waiter = None
        self.silent = False
        self.stamp = None       # the emitter's
        self.wstamp = None      # circular: the waiter's blocked stamp

    def desc(self):
        k = self.kind
        if k == "circular":
            return "circular wait: FIFO %d waits on FIFO %d, which then waits on it" % (self.waiter, self.fid)
        out = "%s on FIFO %d" % (k, self.fid)
        if self.item is not None:
            out += " (task %d slot %d)" % (self.item.tid, self.item.slot)
        if self.silent:
            out += ", silent"
        return out


class Pending:
    """Something blocked on a faulted stamp: a dependent item, or circular's
    waiter (item None)."""

    def __init__(self, fid, item, fault):
        self.fid, self.item, self.fault = fid, item, fault


class FModel(Model):
    def __init__(self, f):
        super().__init__(f)
        self.fst = {}        # fid -> "faulted" | "paused" | "restored"
        self.heads = {}      # fid -> the Fault whose stamp the FIFO holds (state faulted)
        self.latched = set()
        self.blocked = {}    # fid -> Pending
        self.dead = set()    # task keys whose service died
        self.busy = set()    # task keys with a blocked item
        self.gone_children = set()
        self.nfault = 0

    def mask(self):
        return 1 if self.fst else 0

    def view(self):
        """What each FIFO's fault registers should say: its state."""
        v = {0: self.fst.get(0, "ok")}
        for fid in sorted(self.children):
            v[fid] = self.fst.get(fid, "ok")
        return self.mask(), v


def _parts(r, t):
    """randgen's _parts without info work; a scratch part may name a buffer
    deleted before (it is made again)."""
    parts = []
    blit = r.chance(0.55)
    if blit:
        parts.append(("blit", r.choice(BYTES)))
    names = t.scratch + t.gone
    if names and r.chance(0.35):
        parts.append(("scratch", r.choice(names), r.choice(BYTES)))
    c = r.below(3)
    if c == 1:
        parts.append(("compute",))
    elif c == 2 and (t.blits or blit):
        parts.append(("chain", None if blit and (not t.blits or r.chance(0.3)) else r.choice(t.blits[-6:])))
    if r.chance(0.45) or not parts:
        parts.append(("render", r.below(len(CLEARS)), r.chance(0.6)))
    return parts


def _new_item(r, m, t, fid, program=True):
    """randgen's _new_item; with program=False the item is a faulting
    exec's good part: it takes a slot but is not run by the control."""
    parts = _parts(r, t)
    op = abi.OP_EXEC_INDIRECT2 if r.chance(0.2) else abi.OP_EXEC_INDIRECT3
    it = Item(len(m.items) if program else -1, t.key, t.tid, fid, t.slot, op)
    t.slot += 1
    for p in parts:
        if p[0] == "chain":
            p = ("chain", it if p[1] is None else p[1])
        if p[0] == "scratch" and p[1] in t.gone:
            t.gone.remove(p[1])
            t.scratch.append(p[1])
        it.parts.append(p)
    if program:
        m.items.append(it)
        if "blit" in it.kinds():
            t.blits.append(it)
    return it


def generate(seed, steps):
    f = FFeatures(seed, steps)
    r = Rng(seed)
    m = FModel(f)
    acts = []

    def new_task():
        tid = m.free_tid()
        t = TaskM(len(m.incarnations) + 1, tid, r.choice((1, 1, 2)))
        t.gone = []
        m.tasks[tid] = t
        m.incarnations.append(t)
        return t

    def take_broken():
        out = sorted(fid for fid, z in m.children.items() if z)
        for fid in out:
            m.children[fid] = False
            # The guest's recovery deletes and defines it: whatever fault
            # it held is gone with it.
            m.fst.pop(fid, None)
            m.heads.pop(fid, None)
            m.latched.discard(fid)
        for fid in out:
            if fid + 1 in m.children and fid + 1 not in out:
                m.children[fid + 1] = True
        return out

    def usable_tasks():
        return [t for t in sorted(m.tasks.values(), key=lambda t: t.tid)
                if t.slot < SLOTS - 1 and t.key not in m.dead and t.key not in m.busy]

    def free(fid):
        return fid not in m.fst and fid not in m.blocked

    def ready_fifos():
        return [fid for fid in m.fifos() if free(fid)]

    def healthy_ready():
        return [fid for fid in m.healthy_fifos() if free(fid)]

    def root_ok():
        return free(0)

    def resume_expect():
        ports = [p for p in m.ports.values() if p.connected]
        glyphs = [p.port for p in ports if p.glyph and p.attached]
        m.displays = [p for p in m.displays if p is not None]
        return glyphs, [], list(m.displays)

    def release(fid):
        """The pendings blocked on FIFO fid's faulted stamp: they run."""
        out = []
        for w, p in sorted(m.blocked.items()):
            if p.fault.fid == fid:
                out.append(p)
        for p in out:
            del m.blocked[p.fid]
            if p.item is not None:
                m.busy.discard(p.item.task)
        return out

    def release_all():
        out = []
        for fid in sorted(m.heads):
            out += release(fid)
        return out

    def new_fault(fid, kind):
        m.nfault += 1
        return Fault(m.nfault, fid, kind)

    def fault_on(fid, F):
        """The model after a reported fault on fid."""
        m.fst[fid] = "faulted"
        m.heads[fid] = F

    # Set-up: displays, children 1 and 2 and the gate, tasks 1 and 2.
    for p in m.ports.values():
        p.connected = True
        p.fresh_page()
    first = [new_task() for _ in range(2)]
    for fid in (1, 2):
        m.children[fid] = False
    acts.append(("setup", [(t.key, t.tid, t.depth) for t in first], [1, 2]))

    for step in range(steps):
        table = []
        ut = usable_tasks()
        rf = ready_fifos()
        hr = healthy_ready()
        rok = root_ok()
        calm = not m.fst and not m.blocked
        if ut and rf:
            table.append(("work", 26))
        if f.held and ut and calm and hr:
            table.append(("held", 3))
        if rok and m.ready_ports():
            table.append(("present", 5))
        if m.ready_ports() and (rok or any(p.glyph for p in m.ready_ports())):
            table.append(("cursor", 4))
        if rok and len(m.tasks) < 4:
            table.append(("task_new", 8 if not ut else 3))
        dels = [t for t in m.tasks.values() if t.key not in m.busy]
        if rok and len(m.tasks) > 1 and dels:
            table.append(("task_del", 3))
        if rok and f.scratch and any(t.made < 4 and t.key not in m.dead for t in m.tasks.values()):
            table.append(("scratch_new", 3))
        if rok and any(t.scratch and t.key not in m.dead for t in m.tasks.values()):
            table.append(("scratch_del", 2))
        if rok and len(m.children) < MAX_CHILD:
            table.append(("child_new", 3))
        cdel = [fid for fid in m.children if fid not in m.blocked and fid + 1 not in m.blocked
                and fid + 1 not in m.fst]
        if rok and cdel:
            table.append(("child_del", 3))
        if m.gone_children:
            table.append(("ring_deleted", 1))
        # A child blocked on a faulted stamp whose page list a delete zeroed
        # would not start again after the pause or resume that releases it.
        if not [fid for fid in m.blocked if m.children.get(fid)]:
            table.append(("pause", 3))
            table.append(("suspend", 3))
        if f.teardown and not m.blocked:
            table.append(("teardown", 2))
        if f.ports == 0 and calm:
            table.append(("will_suspend_unpause", 1))
        if f.device_info and rok:
            table.append(("device_info", 1))
        table.append(("check", 2))
        if rok and m.dead:
            table.append(("revive", 8))
        # The errors.
        if hr:
            table.append(("fault", f.fault_rate))
            circ = [(a, b) for a in hr if a != 0 for b in hr if b != a and b not in m.latched]
            if circ:
                table.append(("circular", 3))
            if f.death and ut and [x for x in hr if x not in m.latched]:
                table.append(("death", 3))
        if m.fst:
            table.append(("resume", 12))
        if [x for x in rf if x not in m.latched]:
            table.append(("spurious_resume", 1))
        kind = r.weighted(table)

        if kind == "work":
            n = 1 + r.below(min(3, len(rf)))
            fids = r.sample(rf, n)
            items = []
            conc = f.concurrent and n > 1 and r.chance(0.5)
            used = set()
            for fid in fids:
                cand = [t for t in usable_tasks() if not (conc and t.tid in used)]
                if not cand:
                    break
                t = r.choice(cand)
                used.add(t.tid)
                items.append(_new_item(r, m, t, fid))
            if conc and any(p[0] == "chain" and p[1] in items and p[1] is not it for it in items for p in it.parts):
                conc = False
            acts.append(("work", items, not conc))
        elif kind == "held":
            fid = r.choice(hr)
            item = _new_item(r, m, r.choice(ut), fid)
            how = r.choice(("none", "pause", "suspend"))
            if how == "pause":
                acts.append(("held", item, how, take_broken()))
            elif how == "suspend":
                glyphs, frames, ports = resume_expect()
                m.dead.clear()
                acts.append(("held", item, how, take_broken(), glyphs, frames, ports))
            else:
                acts.append(("held", item, how, []))
        elif kind == "present":
            p = r.choice(m.ready_ports())
            p.presented = True
            acts.append(("present", p.port, r.below(2), r.chance(0.25)))
        elif kind == "cursor":
            p = r.choice([p for p in m.ready_ports() if rok or p.glyph])
            acts.append(("cursor", p.port, r.below(FW), r.below(FH), not p.glyph))
            p.glyph = True
        elif kind == "task_new":
            t = new_task()
            acts.append(("task_new", t.key, t.tid, t.depth))
        elif kind == "task_del":
            t = r.choice(sorted(dels, key=lambda t: t.tid))
            del m.tasks[t.tid]
            dead = t.key in m.dead
            m.dead.discard(t.key)
            acts.append(("task_del", t.key, r.chance(0.5) and not dead))
        elif kind == "scratch_new":
            t = r.choice([t for t in sorted(m.tasks.values(), key=lambda t: t.tid)
                          if t.made < 4 and t.key not in m.dead])
            m.nscratch += 1
            t.scratch.append(m.nscratch)
            t.made += 1
            acts.append(("scratch_new", t.key, m.nscratch))
        elif kind == "scratch_del":
            t = r.choice([t for t in sorted(m.tasks.values(), key=lambda t: t.tid)
                          if t.scratch and t.key not in m.dead])
            sid = r.choice(t.scratch)
            t.scratch.remove(sid)
            t.gone.append(sid)
            acts.append(("scratch_del", t.key, sid))
        elif kind == "child_new":
            fid = r.choice([i for i in range(1, MAX_CHILD + 1) if i not in m.children])
            m.children[fid] = False
            m.gone_children.discard(fid)
            acts.append(("child_new", fid))
        elif kind == "child_del":
            fid = r.choice(sorted(cdel))
            state = m.fst.pop(fid, "ok")
            m.latched.discard(fid)
            rel = release(fid)
            m.heads.pop(fid, None)
            del m.children[fid]
            m.gone_children.add(fid)
            if fid + 1 in m.children:
                m.children[fid + 1] = True
            acts.append(("child_del", fid, state, rel, m.view()))
        elif kind == "ring_deleted":
            acts.append(("ring_deleted", r.choice(sorted(m.gone_children))))
        elif kind == "pause":
            item = None
            if ut and hr and r.chance(0.5):
                item = _new_item(r, m, r.choice(ut), r.choice(hr))
            rel = release_all()
            for fid in sorted(m.heads):
                m.fst[fid] = "paused"
                m.latched.add(fid)
            m.heads.clear()
            acts.append(("pause", item, take_broken(), rel, m.view()))
        elif kind == "suspend":
            glyphs, frames, ports = resume_expect()
            rel = release_all()
            for fid in list(m.fst):
                m.fst[fid] = "restored"
            m.heads.clear()
            m.latched.clear()
            m.dead.clear()
            acts.append(("suspend", glyphs, frames, take_broken(), ports, rel, m.view()))
        elif kind == "will_suspend_unpause":
            acts.append(("will_suspend_unpause", take_broken()))
        elif kind == "teardown":
            how = r.choice(("reset", "stop", "recreate"))
            if how == "recreate":
                m.displays = [p for p in m.displays if p is not None]
            for p in m.ports.values():
                p.connected = p.attached
                p.fresh_page()
            for fid in m.children:
                m.children[fid] = False
            m.fst.clear()
            m.heads.clear()
            m.latched.clear()
            m.dead.clear()
            acts.append(("teardown", how, sorted(t.key for t in m.tasks.values()), sorted(m.children),
                         sorted(p.port for p in m.ports.values() if p.connected), list(m.displays)))
        elif kind == "device_info":
            acts.append(("device_info", r.choice((10, 20, 45))))
        elif kind == "check":
            acts.append(("check", m.view()))
        elif kind == "revive":
            key = r.choice(sorted(m.dead))
            m.dead.discard(key)
            acts.append(("revive", key))
        elif kind == "fault":
            fid = r.choice(hr)
            latched = fid in m.latched
            kinds = list(SYNC) if latched else list(DECODE) + ["task_va", "task", "opcode"]
            if not ut:
                kinds = [k for k in kinds if k in ("task", "opcode")]
            k = r.choice(kinds)
            F = new_fault(fid, k)
            F.silent = latched
            busy = set()
            if k in DECODE or k == "task_va":
                t = r.choice(ut)
                F.key = t.key
                busy.add(t.key)
                if k in DECODE:
                    F.item = _new_item(r, m, t, fid, program=False)
                F.variant = r.below(3)
            elif k == "task":
                F.variant = r.choice([x for x in range(1, 10) if x not in m.tasks])
            else:
                F.variant = r.choice(UNKNOWN_OPS)
            others = [x for x in rf if x != fid]
            if not latched and others and r.chance(0.6):
                for x in r.sample(others, 1 + r.below(min(2, len(others)))):
                    cand = [t for t in usable_tasks() if t.key not in busy]
                    if not cand:
                        break
                    t = r.choice(cand)
                    busy.add(t.key)
                    F.comp.append(_new_item(r, m, t, x))
                others = [x for x in others if x not in [c.fid for c in F.comp]]
            if not latched and others and r.chance(0.3):
                cand = [t for t in usable_tasks() if t.key not in busy]
                if cand:
                    x = r.choice(others)
                    t = r.choice(cand)
                    busy.add(t.key)
                    F.held = _new_item(r, m, t, x)
                    others.remove(x)
            deps = [x for x in others if x != 0 and x in hr]
            if not latched and deps and r.chance(0.45):
                cand = [t for t in usable_tasks() if t.key not in busy]
                if cand:
                    t = r.choice(cand)
                    F.dep = _new_item(r, m, t, r.choice(deps))
            if not latched:
                fault_on(fid, F)
                if F.dep is not None:
                    m.blocked[F.dep.fid] = Pending(F.dep.fid, F.dep, F)
                    m.busy.add(F.dep.task)
            else:
                m.latched.discard(fid)
            acts.append(("fault", F, m.view()))
        elif kind == "circular":
            a, b = r.choice(circ)
            F = new_fault(b, "circular")
            F.waiter = a
            fault_on(b, F)
            m.blocked[a] = Pending(a, None, F)
            acts.append(("circular", F, m.view()))
        elif kind == "death":
            fid = r.choice([x for x in hr if x not in m.latched])
            t = r.choice(ut)
            F = new_fault(fid, "death")
            F.key = t.key
            others = [x for x in rf if x != fid]
            if others and r.chance(0.6):
                for x in r.sample(others, 1 + r.below(min(2, len(others)))):
                    cand = [u for u in usable_tasks() if u.key != t.key and u.key not in [c.task for c in F.comp]]
                    if not cand:
                        break
                    F.comp.append(_new_item(r, m, r.choice(cand), x))
            fault_on(fid, F)
            m.dead.add(t.key)
            acts.append(("fault", F, m.view()))
        elif kind == "resume":
            fid = r.choice(sorted(m.fst))
            state = m.fst.pop(fid)
            rel = release(fid) if state == "faulted" else []
            F = m.heads.pop(fid, None)
            if state == "faulted":
                m.latched.discard(fid)
            acts.append(("resume", fid, state, F, rel, m.view()))
        elif kind == "spurious_resume":
            fid = r.choice([x for x in rf if x not in m.latched])
            acts.append(("spurious_resume", fid, m.view()))
    # The guest resumes whatever is still faulted before its teardown.
    for fid in sorted(m.fst):
        state = m.fst.pop(fid)
        rel = release(fid) if state == "faulted" else []
        F = m.heads.pop(fid, None)
        if state == "faulted":
            m.latched.discard(fid)
        acts.append(("resume", fid, state, F, rel, m.view()))
    acts.append(("final", sorted(m.dead)))
    return f, acts, m.items, m.incarnations


# ---- the emitter ----

class FEmitter(Emitter):

    def __init__(self, *a):
        super().__init__(*a)
        self.dead = set()
        self.pend = {}          # fid -> (stamp, item or None) of what waits there

    # ---- observations ----

    def observe(self, label, view):
        """0x102c, waited for at the model's value (a wrong model shows as a
        wait that fails), 0x103c and every live child's node fault offset,
        each labelled with the model's state for its FIFO."""
        s = self.s
        mask, states = view
        s.wait(mmio=abi.R_FAULT_MASK, mmio_eq=mask, label="%s: 0x102c %d (model)" % (label, mask))
        s.r(abi.R_FAULT_OFFSET, label="%s: root %s" % (label, states[0]))
        for fid in sorted(states):
            if fid:
                s.dump(self.fifos[fid].node_gpa(NODE_FAULT), 4, "%s: child %d %s" % (label, fid, states[fid]))

    def wait_fault(self, fid, label):
        s = self.s
        if fid == 0:
            s.wait(mmio=abi.R_FAULT_OFFSET, mmio_ne=HEALTHY, label=label + ": 0x103c")
        else:
            s.wait(gpa=self.fifos[fid].node_gpa(NODE_FAULT), ne=HEALTHY, label=label + ": node fault offset")
        s.wait(mmio=abi.R_FAULT_MASK, mmio_ne=0, label=label + ": 0x102c")

    def wait_read(self, fid, label):
        """Everything written on FIFO fid consumed: 0x100c for the root, the
        node's readCount for a child."""
        f = self.fifo(fid)
        if fid == 0:
            self.s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written & 0xFFFFFFFF, label=label)
        else:
            self.s.wait(gpa=f.node_gpa(NODE_READ), eq=f.written & 0xFFFFFFFF, label=label)

    def resume_fifo(self, fid):
        if fid == 0:
            self.s.resume_root()
        else:
            self.s.w(abi.R_RESUME_CHILD, fid)

    def item_cmd(self, it, barriers=()):
        t, segs = self.segments(it)
        f = self.fifo(it.fid)
        return f, f.cmd(it.op, t.exec_payload(segs, it.op), barriers=barriers)

    def finish(self, it, st, label):
        """An item that was written and rung: its stamp, then its slot."""
        self.fifo(it.fid).wait_stamp(st, label=label)
        self.tasks[it.task].reset_cmdbufs()
        self.ran(it)

    def released(self, rel, label):
        """Pendings a publication released: each dependent's stamp, then its
        slot; circular's waiter's blocked NOP."""
        for p in rel:
            st, it = self.pend.pop(p.fid)
            if it is None:
                self.fifo(p.fid).wait_stamp(st, label="%s: FIFO %d's waiting NOP runs" % (label, p.fid))
            else:
                self.finish(it, st, "%s: item %d runs" % (label, it.id))

    def zero_slot(self, key, slot, label):
        """A faulted exec's slot against its control slot, which nothing
        wrote: nothing of the exec was committed."""
        if self.f.compact:
            return
        t = self.tasks[key]
        self.s.cmp(t.slot_gpa(slot), self.ctrl.slot_gpa((key - 1) * SLOTS + slot), SLOT, label=label)

    # ---- the faults ----

    def bad_command(self, F):
        """The faulting command's opcode and payload."""
        k = F.kind
        if k in DECODE:
            it = F.item
            t, segs = self.segments(it)
            v = F.variant
            if k == "record":
                bad = seg(abi.ENC_BLIT, rec((299, 324, 0)[v], struct.pack("<I", t.out)))
            elif k == "overlong":
                bad = seg(abi.ENC_BLIT, rec(306, bytes(0x18), length=0x40 + 0x10 * v))
            elif k == "ref":
                bad = seg(abi.ENC_BLIT, b_fill_buf((0, 0x7FF0, len(t.ol.entries))[v], 0, 0x10, 0x99))
            else:
                bad = seg((3, 6, 0xFF)[v], b"")
            return it.op, t.exec_payload(segs + [bad], it.op)
        if k == "task_va":
            t = self.tasks[F.key]
            return abi.OP_GET_COMPUTE_INFO, struct.pack("<IIIIQ", t.tid, t.cpso, 5, 4, t.length - 8 * (1 + F.variant))
        if k == "task":
            return abi.OP_EXEC_INDIRECT3, exec_payload(F.variant, [(4 * PAGE, 0x10)], [])
        if k == "opcode":
            return F.variant, struct.pack("<II", 0x0BAD0000 | F.variant, F.n)
        if k == "death":
            t = self.tasks[F.key]
            return abi.OP_EXEC_INDIRECT3, exec_payload(t.tid, [(t.npages * PAGE, 0x40)], t.resources())
        raise ValueError(k)

    def a_fault(self, F, view):
        s = self.s
        lab = "fault %d" % F.n
        self.mark("%s: %s%s%s%s" % (lab, F.desc(),
                                   "; at once: " + "; ".join(it.desc() for it in F.comp) if F.comp else "",
                                   "; held behind the gate: " + F.held.desc() if F.held else "",
                                   "; gated on it: " + F.dep.desc() if F.dep else ""))
        g = self.gate
        if F.held is not None:
            hf, hst = self.item_cmd(F.held, barriers=[(g.stamp_index, g.stamp + 1)])
            hf.doorbell()
            hf.wait_stamp(hst, timeout_ms=HOLD_MS, label="%s: item %d held" % (lab, F.held.id))
        written = [self.item_cmd(it) for it in F.comp]
        fb = self.fifo(F.fid)
        op, payload = self.bad_command(F)
        F.stamp = fb.cmd(op, payload)
        nop = fb.cmd(abi.OP_NOP) if F.silent else None
        if F.dep is not None:
            df, dst = self.item_cmd(F.dep, barriers=[(fb.stamp_index, F.stamp)])
            self.pend[F.dep.fid] = (dst, F.dep)
        rung = [f for f, _ in written] + [fb] + ([df] if F.dep is not None else [])
        for f in rung:
            f.doorbell()
        for it, (f, st) in zip(F.comp, written):
            self.finish(it, st, "%s: item %d beside it" % (lab, it.id))
        if F.silent:
            self.wait_read(F.fid, "%s: the command and a NOP behind it consumed" % lab)
            fb.wait_stamp(nop, timeout_ms=HOLD_MS, label="%s: silent, the NOP held" % lab)
            self.observe(lab + " (silent)", view)
            self.resume_fifo(F.fid)
            fb.wait_stamp(nop, label="%s: the second resume publishes both" % lab)
        else:
            self.wait_fault(F.fid, lab)
            if F.dep is not None:
                df.wait_stamp(dst, timeout_ms=HOLD_MS, label="%s: item %d held on the faulted stamp" % (lab, F.dep.id))
        if F.item is not None:
            self.zero_slot(F.item.task, F.item.slot, "%s: its slot, never committed" % lab)
            self.tasks[F.item.task].reset_cmdbufs()
        self.observe(lab, view)
        if F.held is not None:
            g.submit(abi.OP_NOP, label="%s: the gate opens" % lab)
            self.finish(F.held, hst, "%s: item %d released" % (lab, F.held.id))
        if F.kind == "death":
            self.dead.add(F.key)
        self.events(lab)

    def a_circular(self, F, view):
        s = self.s
        lab = "fault %d" % F.n
        self.mark("%s: %s" % (lab, F.desc()))
        a, b = self.fifo(F.waiter), self.fifo(F.fid)
        a0 = a.cmd(abi.OP_NOP)
        bst = b.stamp + 1
        a1 = a.cmd(abi.OP_NOP, barriers=[(b.stamp_index, bst)])
        a.doorbell()
        a.wait_stamp(a0, label="%s: FIFO %d's first NOP" % (lab, a.fid))
        a.wait_stamp(a1, timeout_ms=HOLD_MS, label="%s: FIFO %d waits on %d:%d" % (lab, a.fid, b.stamp_index, bst))
        F.stamp = b.cmd(abi.OP_NOP, barriers=[(a.stamp_index, a1)])
        assert F.stamp == bst
        b.doorbell()
        self.wait_fault(b.fid, lab)
        self.pend[a.fid] = (a1, None)
        self.observe(lab, view)
        self.events(lab)

    def a_resume(self, fid, state, F, rel, view):
        lab = "resume %d" % fid
        self.mark("%s (%s%s)" % (lab, state, ", fault %d" % F.n if F is not None else ""))
        self.resume_fifo(fid)
        if F is not None:
            self.fifo(fid).wait_stamp(F.stamp, label="%s: the faulted stamp published" % lab)
        self.released(rel, lab)
        self.observe(lab, view)
        self.events(lab)

    def a_spurious_resume(self, fid, view):
        lab = "resume %d, not faulted" % fid
        self.mark(lab)
        self.resume_fifo(fid)
        self.observe(lab, view)
        self.events(lab)

    def a_revive(self, key):
        t = self.tasks[key]
        self.mark("task %d's service died: delete it, define it again" % t.tid)
        t.delete(label="delete dead task %d" % t.tid)
        t.define(label="define task %d again" % t.tid)
        t.install(label="task %d's object list again" % t.tid)
        self.dead.discard(key)

    def a_ring_deleted(self, fid):
        c = self.fifos[fid]
        self.mark("a NOP rung on deleted child %d" % fid)
        st = c.cmd(abi.OP_NOP)
        c.doorbell()
        c.wait_stamp(st, timeout_ms=HOLD_MS, label="deleted child %d: not run" % fid)
        self.s.dump(c.node_gpa(0), abi.CHILD_NODE_SIZE, "node %d, deleted" % fid)

    # ---- randgen's actions, with the faults' state ----

    def a_check(self, view):
        self.mark("checkpoint")
        self.events("checkpoint")
        self.observe("checkpoint", view)
        for key in sorted(self.tasks):
            self.compare_task(key)

    def a_child_del(self, fid, state, rel, view):
        lab = "delete child %d" % fid
        self.mark(lab + ("" if state == "ok" else " (%s)" % state))
        self.fifos[fid].delete(label=lab)
        del self.x.children[fid]
        self.released(rel, lab)
        self.s.dump(self.fifos[fid].node_gpa(0), abi.CHILD_NODE_SIZE, "node %d" % fid)
        # Node fid+1 without its read offset, which moves while that child runs.
        self.s.dump(self.fifos[fid].node_gpa(0) + abi.CHILD_NODE_SIZE + 8, abi.CHILD_NODE_SIZE - 8, "node %d from +8" % (fid + 1))
        if state != "ok" or rel:
            self.observe(lab, view)
            self.events(lab)

    def a_task_del(self, key, objects):
        super().a_task_del(key, objects)
        self.dead.discard(key)

    def a_scratch_del(self, key, sid):
        """40 on the buffer; it stays listed and nameable (made again)."""
        t = self.tasks[key]
        ref = t.scratch[sid]
        self.mark("task %d: delete scratch buffer %d" % (t.tid, ref))
        self.s.submit(abi.OP_DELETE_OBJECT, struct.pack("<IIII", t.tid, 1000, 12, ref),
                      label="task %d: delete ref %d" % (t.tid, ref))

    def a_pause(self, item, broken, rel, view):
        s = self.s
        self.mark("pause, unpause" + ("; rung while paused: " + item.desc() if item else ""))
        s.api("pause")
        if item is not None:
            f, st = self.item_cmd(item)
            f.doorbell()
            f.wait_stamp(st, timeout_ms=HOLD_MS, label="item %d rung while paused: not run" % item.id)
        s.api("unpause")
        if item is not None:
            self.finish(item, st, "item %d after unpause" % item.id)
        self.released(rel, "the unpause")
        self.events("unpaused")
        self.recover(broken, "the unpause")
        self.observe("unpaused", view)

    def a_suspend(self, glyphs, frames, broken, ports, rel, view):
        self.mark("suspend, resume on a fresh device")
        x = self.x
        x.suspend()
        x.resume(config=self.config(ports) if ports != list(range(self.f.ports)) else None)
        self.reindex(ports)
        if glyphs:
            x.expect("cursorGlyph", n=len(glyphs), label="glyphs replayed on ports %s" % glyphs)
            x.expect("cursorShow", n=len(glyphs), label="shows replayed on ports %s" % glyphs)
        self.released(rel, "the resume")
        self.events("resumed")
        if not self.f.compact:
            self.s.sweep(0x1000, 0x1040, label="registers after the resume")
        for port in sorted(x.pages):
            self.page(port, "resumed")
        self.recover(broken, "the resume")
        self.observe("resumed", view)
        self.dead.clear()

    def a_teardown(self, *a):
        super().a_teardown(*a)
        self.dead.clear()

    def a_held(self, it, how, broken, *resume):
        super().a_held(it, how, broken, *resume)
        if how == "suspend":
            self.dead.clear()

    def a_final(self, dead):
        x, s = self.x, self.s
        self.mark("the end: teardown")
        self.events("end of the walk")
        s.fault_state("end of the walk")
        for key in sorted(self.tasks):
            self.compare_task(key)
        s.dump(s.fifo.base_gpa(), 4 * (GATE + 1), "stamp slots")
        for key, t in sorted(self.tasks.items(), key=lambda kt: kt[1].tid):
            if key not in self.dead:
                x.delete_objects(t)
            t.delete(label="delete task %d" % t.tid)
        if x.t0 is not None:
            x.delete_backings()
        for port in sorted(x.pages):
            x.disconnect(port)
        if x.t0 is not None:
            x.t0.delete(label="delete task 0")
        for fid in sorted(x.children):
            x.children[fid].delete(label="delete child %d" % fid)
        self.gate.delete(label="delete the gate")
        s.submit(abi.OP_NOP, label="last NOP")
        self.events("torn down")
        s.dump(s.pa(s.root_page), 0x40, "root page: task words")
        s.dump(s.pa(s.root_page, abi.ROOT_CHILD_NODE), abi.CHILD_NODE_SIZE * GATE, "root page: child nodes")
        s.fault_state("torn down")
        s.api("release")
        self.events("released")


def summary(f, acts):
    hist = {}
    kinds = {}
    for a in acts[1:-1]:
        hist[a[0]] = hist.get(a[0], 0) + 1
        if a[0] in ("fault", "circular"):
            kinds[a[1].kind] = kinds.get(a[1].kind, 0) + 1
    return "%s; %d steps: %s; faults: %s" % (
        ", ".join(f.tags()), len(acts) - 2, ", ".join("%s %d" % (k, v) for k, v in sorted(hist.items())),
        ", ".join("%s %d" % (k, v) for k, v in sorted(kinds.items())) or "none")


def build(seed, steps, doc):
    f, acts, items, inc = generate(seed, steps)
    name = "interplay_faultwalk_%d" % seed
    e = FEmitter(name, doc.strip() + "\n\nThis program: " + summary(f, acts) + ".", seed, f, acts, items, inc)
    return e.emit()


# ---- reading the expected files back ----

def check_expected(path):
    """The model's labels against what the expected file records: every
    fault register a label gives a state for reads 0xffffffff exactly when
    the state is ok, and no wait failed but those expected to (HOLD_MS).
    Returns a list of complaints."""
    bad = []
    for line in open(path):
        if not line.startswith("{"):
            continue
        o = json.loads(line)
        lab = o.get("label") or ""
        if o.get("ev") == "wait" and o.get("ok") is False and not any(
                w in lab for w in ("held", "not run", "waits on", "silent")):
            bad.append("wait failed: %s" % lab)
        state = None
        for st in ("ok", "faulted", "paused", "restored"):
            if lab.endswith(" " + st):
                state = st
        if state is None:
            continue
        if o.get("ev") == "mmio_r":
            val = int(o["val"], 16)
        elif o.get("ev") == "dump":
            val = struct.unpack("<I", bytes.fromhex(o["hex"])[:4])[0] if "hex" in o else None
        else:
            continue
        if val is None:
            continue
        if (val == HEALTHY) != (state == "ok"):
            bad.append("%s: 0x%x" % (lab, val))
    return bad
