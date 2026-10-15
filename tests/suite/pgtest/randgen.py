"""Seeded random guest programs (scenarios/interplay_rand.py): a generator
that walks, from a seed, over legal guest and host actions across every
subsystem, and an emitter that turns the walk into a Scenario.

Hand-written sessions reach only the combinations someone thought of. Here a
seed picks the program: tasks (one- or two-level page tables) and objects
created and deleted, blit, compute, render and info work on the root FIFO
and child FIFOs, through opcodes 67, 55 and 69, serialized by barriers or
running at once, work held behind a barrier across a pause or a suspend,
display connects, disconnects, presents, cursor moves and hot plugs,
IOSurface maps, tests, unmaps and swaps, pause and unpause (with work rung
while paused), suspend onto a fresh device and resume, task and FIFO
deletion, and reset, stop or a fresh device with the guest redoing its
set-up. The generator keeps a model of what the guest and the host hold,
and offers only actions that model says are legal, so every step is one a
careful guest driver may take and every checkpoint is deterministic.

The control run. Every piece of GPU work the walk runs (an "item") writes
into a slot of its own, 0x400 bytes of its task's out buffer, that nothing
else writes. Before the walk, on the pristine device, a control task
runs every item of the program in program order on the root FIFO, with
nothing between them, into a control region laid out as the live tasks'
out buffers are (one region per task incarnation, the same slot in it).
Whatever the walk did around an item (suspends, resets, pauses, FIFOs and
tasks coming and going, other work running at once), its slot must hold
the bytes the item gives uninterrupted: a cmp of the live slot with the
control slot after each item, or in a long ("compact") program one cmp of
each task's slots run so far at checkpoints, before the task is deleted,
and at the end. The control results of each distinct kind of work are
dumped once, so the expected files record the absolute bytes too.

The items reuse pgtest/session.py's SessionTask work (blit, add_one,
the triangle); what they add is layout only:

  slot + 0x000  blit: a fill, a copy of the source, the private heap
                buffer filled and copied out (0x100)
  slot + 0x100  compute: add_one over the source, or ("chain") over an
                earlier item's blit part (0x100)
  slot + 0x200  render: the 8 x 8 target cleared, the triangle or not,
                copied out (0x100)
  slot + 0x300  scratch: a buffer object the walk created, filled and
                copied out (0x80)
  slot + 0x380  info (isolate only): 450, 457 or 451's reply

Invariants the model keeps (docs/testing.md, sections 6.5-6.7, 6.21,
6.24, 7.3, 7.4):
  - One exec per FIFO in a batch, each waited for: an exec followed on its
    FIFO by more of the guest's commands may publish with them or alone
    (pgtest/interplay_util.py), so interrupt counts would be timing.
  - Items in one batch that share a task, or where one reads another's
    result, are serialized with barriers on the previous item's stamp
    (Session.batch). Unserialized batches run different tasks on different
    FIFOs at once; events then report block calls sorted and digested.
  - Held work waits on a barrier on the gate (child FIFO 6, which nothing
    else uses) for less than the 2 s after which a blocked FIFO reports; a
    pause or a suspend abandons the wait and the command is parsed again.
  - Nothing touches a display while the device is paused (the doorbell and
    cursor read the guest display's page, which asserts), and pause and
    unpause always pair (a second pause asserts with a guest display).
    rvg_device_will_suspend then unpause is taken only with no guest display
    (with one it asserts, section 7.3).
  - A child FIFO keeps its stamp count across deletes, resets and resumes,
    so a wait on its slot never passes on an old value. Deleting child n
    zeroes node n+1's page list. Every unpause starts every child again,
    and a FIFO's start maps its page list again, as a resume does: a live
    child n+1 then does not start after an unpause (a pause's, or an
    abandoned rvg_device_will_suspend's) or a resume, and a doorbell on it runs
    nothing. The guest writes its page list again, deletes it and defines it
    (section 6.24); that delete zeroes the next node in turn.
  - A display's expected ack is 1 on a fresh page (the guest display's
    state machine, in its first state); a display plugged again into a
    connected page raises it by one. After a resume the guest display
    replays the cursor glyph and show when the page holds a glyph, and
    presents the page's presentedMappingID again when anything was
    presented; mapping 0 or an unmapped surface throws inside the guest
    display's restore, which presents nothing.
  - Two host sequences leave their outcome to the scheduler, so the walk
    never takes them (docs/testing.md, section 7.4):
    * rvg_device_will_suspend quiesces every FIFO but wakes none, so each FIFO
      thread idle waiting for ring bytes stays in its wait, and unpause's start
      adds a second thread on the FIFO; the two consume the ring with no lock,
      and a later drain waits for one exit only, so the second thread can tear
      down the next start's mapping. An abandoned suspend is therefore
      rvg_device_will_suspend, then a pause, whose drain ends each old thread,
      then unpause.
    * Releasing a device whose root is live tears the root down during the
      release: each child's delete notifies every remote task from a block
      that holds the device being released, and what those notifications
      reach then depends on timing. A fresh device is therefore preceded
      by a host stop, which tears the root down while the device lives.
  - Info replies abort an in-process host: programs with info work run in
    the isolate setup only. The IOSurface host device exists only with a
    memory map (map and isolate).
"""
import copy
import struct

from . import abi
from . import render_lib as R
from .abi import PAGE
from .compute_objects import compute_blob, compute_pipeline_record
from .diff_util import exec_payload
from .display_guest import SS_EXPECTED_ACK, SS_MODE_COUNT, SS_PRESENT_COUNT, SS_RAISED, display
from .exec_objects import info_heap_size, info_ref, pipeline_record, render_blob
from .fifo import ChildFIFO
from .iosfc_guest import CMD_TEST, CMD_UNMAP
from .resources_build import b_copy_buf_buf, b_fill_buf, heap_buffer_rec, heap_rec
from .session import DEL_RESOURCE, FH, FW, PHASE, TH, TW, Session, SessionTask
from .task import GPUTask, seg

SLOT = PHASE             # 0x400 a slot
SCRATCH_OFF = 0x300
INFO_OFF = 0x380
CTRL_TID = 200           # the control task (task IDs are below 256)
MAX_CHILD = 4
GATE = 6                 # the gate: a child FIFO only held work waits on
LIVE_OUT_PAGES = 4
SLOTS = LIVE_OUT_PAGES * PAGE // SLOT   # 64 slots a live task
COMPACT = 60             # programs longer than this compare at checkpoints
BYTES = (0x11, 0x5A, 0xC3, 0x3C)
CLEARS = ((0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0))
# Info replies (docs/testing.md 6.17): the bytes the reply writes that are
# its fields. 450 writes 0x1c, of which the last 3 are padding; 457 writes
# 0xc, of which 2 are padding. The padding is zeros, and never compared.
INFO_LEN = {450: 0x19, 457: 10, 451: 0x10}
M64 = (1 << 64) - 1


class Rng:
    """splitmix64: the same stream on every Python."""

    def __init__(self, seed):
        self.x = seed & M64

    def next(self):
        self.x = (self.x + 0x9E3779B97F4A7C15) & M64
        z = self.x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def below(self, n):
        return self.next() % n

    def chance(self, p):
        return self.below(1 << 20) < int(p * (1 << 20))

    def choice(self, seq):
        seq = list(seq)
        return seq[self.below(len(seq))]

    def sample(self, seq, k):
        seq = list(seq)
        out = []
        for _ in range(min(k, len(seq))):
            out.append(seq.pop(self.below(len(seq))))
        return out

    def weighted(self, table):
        total = sum(w for _, w in table)
        r = self.below(total)
        for v, w in table:
            if r < w:
                return v
            r -= w
        raise AssertionError


# ---- the program ----

class Item:
    """One piece of GPU work: task incarnation `task` (task ID `tid`), FIFO
    `fid`, its slot, the exec opcode, and its parts in the order they run:
    ("blit", byte), ("scratch", scratch id, byte), ("compute",),
    ("chain", source item), ("render", clear index, draw), ("info", cmd)."""

    def __init__(self, iid, task, tid, fid, slot, op):
        self.id, self.task, self.tid, self.fid, self.slot, self.op = iid, task, tid, fid, slot, op
        self.parts = []

    def cslot(self):
        """Its slot in the control task's out buffer."""
        return (self.task - 1) * SLOTS + self.slot

    def kinds(self):
        return [p[0] for p in self.parts]

    def info(self):
        return next((p[1] for p in self.parts if p[0] == "info"), None)

    def desc(self):
        out = []
        for p in self.parts:
            if p[0] == "blit":
                out.append("blit %02x" % p[1])
            elif p[0] == "scratch":
                out.append("scratch %02x" % p[2])
            elif p[0] == "chain":
                out.append("chain<%d" % p[1].id)
            elif p[0] == "render":
                out.append("render c%d%s" % (p[1], "+tri" if p[2] else ""))
            elif p[0] == "info":
                out.append("info %d" % p[1])
            else:
                out.append(p[0])
        return "item %d (task %d slot %d, FIFO %d, %d: %s)" % (self.id, self.tid, self.slot, self.fid, self.op,
                                                              ", ".join(out))


class Features:
    """What a seed's program may use."""

    def __init__(self, seed, steps):
        r = Rng(seed ^ 0x5EED5EED5EED)
        self.compact = steps > COMPACT
        self.ports = r.choice((0, 1, 1, 1, 2, 2))
        self.iosfc = self.ports > 0 and r.chance(0.3)
        self.info = r.chance(0.2) and not self.compact
        self.concurrent = r.chance(0.6)
        self.teardown = r.chance(0.7)
        self.scratch = r.chance(0.7)
        self.plug = self.ports > 0 and r.chance(0.5)
        self.device_info = r.chance(0.4)
        self.held = r.chance(0.6)

    def setups(self):
        if self.info:
            return ("isolate",)
        if self.iosfc:
            return ("map", "isolate")
        return ("blocks", "map", "isolate")

    def tags(self):
        t = ["ports=%d" % self.ports]
        for k in ("iosfc", "info", "concurrent", "teardown", "scratch", "plug", "device_info", "held", "compact"):
            if getattr(self, k):
                t.append(k)
        return t


class TaskM:
    def __init__(self, key, tid, depth):
        self.key, self.tid, self.depth = key, tid, depth
        self.slot = 0
        self.scratch = []       # live scratch ids
        self.made = 0           # scratch objects ever made
        self.blits = []         # items with a blit part


class PortM:
    def __init__(self, port):
        self.port = port
        self.connected = False
        self.attached = True
        self.glyph = False
        self.presented = False
        self.mapping = 0         # the page's presentedMappingID
        self.ack = 0             # the page's expectedAck

    def fresh_page(self):
        self.glyph = self.presented = False
        self.mapping = 0
        self.ack = 1 if self.attached else 0


class Model:
    """What the guest knows the device holds."""

    def __init__(self, f):
        self.f = f
        self.tasks = {}          # tid -> TaskM
        self.incarnations = []   # every TaskM, by key - 1
        self.children = {}       # fid -> zeroed (bool)
        self.ports = {p: PortM(p) for p in range(f.ports)}
        self.surfaces = {}       # mid -> mapped (bool)
        self.items = []
        self.nscratch = 0
        self.displays = [p for p in range(f.ports)]  # the host's display list: port or None

    def fifos(self):
        return [0] + sorted(self.children)

    def healthy_fifos(self):
        return [0] + sorted(f for f, z in self.children.items() if not z)

    def free_tid(self):
        return next(t for t in range(1, 8) if t not in self.tasks)

    def ready_ports(self):
        return [p for p in self.ports.values() if p.connected and p.attached]


def _parts(r, t, info):
    """A random item's parts for task t."""
    parts = []
    if info and r.chance(0.5):
        return [("info", r.choice((450, 457, 451)))]
    blit = r.chance(0.55)
    if blit:
        parts.append(("blit", r.choice(BYTES)))
    if t.scratch and r.chance(0.35):
        parts.append(("scratch", r.choice(t.scratch), r.choice(BYTES)))
    c = r.below(3)
    if c == 1:
        parts.append(("compute",))
    elif c == 2 and (t.blits or blit):
        parts.append(("chain", None if blit and (not t.blits or r.chance(0.3)) else r.choice(t.blits[-6:])))
    if r.chance(0.45) or not parts:
        parts.append(("render", r.below(len(CLEARS)), r.chance(0.6)))
    if info and r.chance(0.3):
        parts.append(("info", r.choice((450, 457, 451))))
    return parts


def _new_item(r, m, t, fid, info):
    parts = _parts(r, t, info)
    if all(p[0] == "info" for p in parts) and r.chance(0.5):
        op = abi.OP_INFO_INDIRECT
    else:
        op = abi.OP_EXEC_INDIRECT2 if r.chance(0.2) else abi.OP_EXEC_INDIRECT3
    it = Item(len(m.items), t.key, t.tid, fid, t.slot, op)
    t.slot += 1
    for p in parts:
        if p[0] == "chain":
            p = ("chain", it if p[1] is None else p[1])
        it.parts.append(p)
    m.items.append(it)
    if "blit" in it.kinds():
        t.blits.append(it)
    return it


def generate(seed, steps):
    """The program of `seed`: its features, a list of actions (each a tuple
    whose first element names it), its items and its task incarnations."""
    f = Features(seed, steps)
    r = Rng(seed)
    m = Model(f)
    acts = []

    def new_task():
        tid = m.free_tid()
        t = TaskM(len(m.incarnations) + 1, tid, r.choice((1, 1, 2)))
        m.tasks[tid] = t
        m.incarnations.append(t)
        return t

    def take_broken():
        """The live children whose page list a delete zeroed: the next
        start of the FIFOs leaves them stopped."""
        out = sorted(fid for fid, z in m.children.items() if z)
        for fid in out:
            m.children[fid] = False
        # The guest's recovery deletes each of them (49), which zeroes the
        # next node's page list in turn, unless that one is recovered after.
        for fid in out:
            if fid + 1 in m.children and fid + 1 not in out:
                m.children[fid + 1] = True
        return out

    def usable_tasks():
        return [t for t in sorted(m.tasks.values(), key=lambda t: t.tid) if t.slot < SLOTS]

    def resume_expect():
        ports = [p for p in m.ports.values() if p.connected]
        glyphs = [p.port for p in ports if p.glyph and p.attached]
        frames = [p.port for p in ports if p.attached and p.presented and m.surfaces.get(p.mapping)]
        m.displays = [p for p in m.displays if p is not None]
        return glyphs, frames, list(m.displays)

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
        if ut:
            table.append(("work", 30))
            if f.held:
                table.append(("held", 4))
        if m.ready_ports():
            table.append(("present", 7))
            table.append(("cursor", 5))
        if len(m.tasks) < 4:
            table.append(("task_new", 8 if not ut else 3))
        if len(m.tasks) > 1:
            table.append(("task_del", 3))
        if f.scratch and any(t.made < 4 for t in m.tasks.values()):
            table.append(("scratch_new", 3))
        if any(t.scratch for t in m.tasks.values()):
            table.append(("scratch_del", 2))
        if len(m.children) < MAX_CHILD:
            table.append(("child_new", 3))
        if m.children:
            table.append(("child_del", 3))
        table.append(("pause", 3))
        table.append(("suspend", 3))
        if f.teardown:
            table.append(("teardown", 3))
        if f.ports == 0:
            table.append(("will_suspend_unpause", 1))
        if f.device_info:
            table.append(("device_info", 1))
        table.append(("check", 2))
        if f.iosfc:
            if len(m.surfaces) < 3:
                table.append(("surface_map", 6))
            if any(m.surfaces.values()):
                table.append(("surface_test", 4))
                table.append(("surface_unmap", 1))
                if m.ready_ports():
                    table.append(("swap", 8))
            if [mid for mid, v in m.surfaces.items() if not v]:
                table.append(("surface_remap", 2))
        if f.ports:
            if [p for p in m.ports.values() if p.connected]:
                table.append(("port_off", 1))
            if [p for p in m.ports.values() if not p.connected]:
                table.append(("port_on", 3))
        if f.plug:
            if [p for p in m.ports.values() if p.connected and p.attached]:
                table.append(("unplug", 3))
            if [p for p in m.ports.values() if p.connected and not p.attached]:
                table.append(("hotplug", 5))
        kind = r.weighted(table)

        if kind == "work":
            fids = m.fifos()
            n = 1 + r.below(min(3, len(fids)))
            fids = r.sample(fids, n)
            items = []
            conc = f.concurrent and n > 1 and r.chance(0.5)
            used = set()
            for fid in fids:
                cand = [t for t in usable_tasks() if not (conc and t.tid in used)]
                if not cand:
                    break
                t = r.choice(cand)
                used.add(t.tid)
                items.append(_new_item(r, m, t, fid, f.info))
            if conc and any(p[0] == "chain" and p[1] in items and p[1] is not it for it in items for p in it.parts):
                conc = False
            acts.append(("work", items, not conc))
        elif kind == "held":
            fid = r.choice(m.healthy_fifos())
            item = _new_item(r, m, r.choice(ut), fid, f.info)
            how = r.choice(("none", "pause", "suspend"))
            if how == "pause":
                acts.append(("held", item, how, take_broken()))
            elif how == "suspend":
                glyphs, frames, ports = resume_expect()
                acts.append(("held", item, how, take_broken(), glyphs, frames, ports))
            else:
                acts.append(("held", item, how, []))
        elif kind == "present":
            p = r.choice(m.ready_ports())
            p.presented = True
            acts.append(("present", p.port, r.below(2), r.chance(0.25)))
        elif kind == "cursor":
            p = r.choice(m.ready_ports())
            acts.append(("cursor", p.port, r.below(FW), r.below(FH), not p.glyph))
            p.glyph = True
        elif kind == "task_new":
            t = new_task()
            acts.append(("task_new", t.key, t.tid, t.depth))
        elif kind == "task_del":
            t = r.choice(sorted(m.tasks.values(), key=lambda t: t.tid))
            del m.tasks[t.tid]
            acts.append(("task_del", t.key, r.chance(0.5)))
        elif kind == "scratch_new":
            t = r.choice([t for t in sorted(m.tasks.values(), key=lambda t: t.tid) if t.made < 4])
            m.nscratch += 1
            t.scratch.append(m.nscratch)
            t.made += 1
            acts.append(("scratch_new", t.key, m.nscratch))
        elif kind == "scratch_del":
            t = r.choice([t for t in sorted(m.tasks.values(), key=lambda t: t.tid) if t.scratch])
            sid = r.choice(t.scratch)
            t.scratch.remove(sid)
            acts.append(("scratch_del", t.key, sid))
        elif kind == "child_new":
            fid = r.choice([i for i in range(1, MAX_CHILD + 1) if i not in m.children])
            m.children[fid] = False
            acts.append(("child_new", fid))
        elif kind == "child_del":
            fid = r.choice(sorted(m.children))
            del m.children[fid]
            if fid + 1 in m.children:
                m.children[fid + 1] = True
            acts.append(("child_del", fid))
        elif kind == "pause":
            item = None
            if ut and r.chance(0.5):
                item = _new_item(r, m, r.choice(ut), r.choice(m.healthy_fifos()), False)
            acts.append(("pause", item, take_broken()))
        elif kind == "suspend":
            glyphs, frames, ports = resume_expect()
            acts.append(("suspend", glyphs, frames, take_broken(), ports))
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
            acts.append(("teardown", how, sorted(t.key for t in m.tasks.values()), sorted(m.children),
                         sorted(p.port for p in m.ports.values() if p.connected), list(m.displays)))
        elif kind == "device_info":
            acts.append(("device_info", r.choice((10, 20, 45))))
        elif kind == "check":
            acts.append(("check",))
        elif kind == "surface_map":
            mid = 1 + len(m.surfaces)
            m.surfaces[mid] = True
            acts.append(("surface_map", mid, 1 + r.below(0x40)))
        elif kind == "surface_remap":
            mid = r.choice(sorted(mid for mid, v in m.surfaces.items() if not v))
            m.surfaces[mid] = True
            acts.append(("surface_map", mid, 1 + r.below(0x40)))
        elif kind == "surface_test":
            acts.append(("surface_test", r.choice(sorted(mid for mid, v in m.surfaces.items() if v))))
        elif kind == "surface_unmap":
            mid = r.choice(sorted(mid for mid, v in m.surfaces.items() if v))
            m.surfaces[mid] = False
            acts.append(("surface_unmap", mid))
        elif kind == "swap":
            p = r.choice(m.ready_ports())
            mid = r.choice(sorted(mid for mid, v in m.surfaces.items() if v))
            p.presented = True
            p.mapping = mid
            acts.append(("swap", p.port, mid, r.chance(0.25)))
        elif kind == "port_off":
            p = r.choice([p for p in m.ports.values() if p.connected])
            p.connected = False
            acts.append(("port_off", p.port))
        elif kind == "port_on":
            p = r.choice([p for p in m.ports.values() if not p.connected])
            p.connected = True
            p.fresh_page()
            acts.append(("port_on", p.port, p.attached))
        elif kind == "unplug":
            p = r.choice([p for p in m.ports.values() if p.connected and p.attached])
            p.attached = False
            idx = m.displays.index(p.port)
            m.displays[idx] = None
            acts.append(("unplug", p.port, idx))
        elif kind == "hotplug":
            p = r.choice([p for p in m.ports.values() if p.connected and not p.attached])
            p.attached = True
            p.ack += 1
            m.displays.append(p.port)
            acts.append(("hotplug", p.port, p.ack))
    acts.append(("final",))
    return f, acts, m.items, m.incarnations


# ---- the emitter ----

class RandTask(SessionTask):
    """SessionTask's objects, in its order, over a page table of `depth`
    levels, with an out buffer of `out_pages` (16 slots a page) and scratch
    buffers the walk adds."""

    def __init__(self, s, tid, seed, pages=64, out_pages=LIVE_OUT_PAGES, depth=1):
        GPUTask.__init__(self, s, tid=tid, pages=pages, depth=depth)
        self.seed = seed
        self.shared = []
        self.src, self.src_va = self._buf(PAGE, bytes((i * 7 + seed) & 0xFF for i in range(0x100)))
        self.out, self.out_va = self._buf(out_pages * PAGE)
        self.heap_va = self.reserve(4 * PAGE)
        self.heap = self.ol.add(abi.T_HEAP, heap_rec(self.heap_va // PAGE, 4 * PAGE, 1))
        self.hbuf = self.ol.add(abi.T_HEAP_BUFFER, heap_buffer_rec(self.heap, PAGE, PAGE))
        self.f_add = self._function("shaders/compute_add.metallib")
        self.cpso = self.ol.add(abi.T_SERIALIZED, compute_pipeline_record(compute_blob(self.f_add)))
        self.f_vert = self._function("shaders/vert.metallib")
        self.f_frag = self._function("shaders/frag.metallib")
        self.rpso = self.ol.add(abi.T_SERIALIZED, pipeline_record(14, render_blob(self.f_vert, self.f_frag, 70)))
        self.tbuf, self.tbuf_va = self._buf(PAGE)
        self.tex = self.ol.add(abi.T_SERIALIZER_TEXTURE, R.op_header(9) + struct.pack(
            "<IIQQ", 0, self.tbuf, 0, 32) + R.tex_desc(TW, TH))
        self.extra = []
        self.scratch = {}

    def add_scratch(self, sid):
        """A shared buffer (type 1) of one page, listed in every exec."""
        ref, _ = self.buffer(PAGE)
        self.scratch[sid] = ref
        self.extra.append(ref)
        return ref

    def slot_gpa(self, slot, off=0):
        return self.gpa(self.out_va + slot * SLOT + off)

    def exec_payload(self, segments, op):
        """An exec (67 or 69: {ref, flags} resources; 55: 24-byte resource
        records) of these segments, one command buffer range each, every
        listed buffer in its resource list."""
        cbs = [self.cmdbuf(sg) for sg in segments]
        return exec_payload(self.tid, cbs, self.resources(), op)


def item_segments(t, slot, it, chain_slot, scratch_ref):
    """The segments of item `it` on task t at `slot`; chain_slot is the slot
    of the item whose blit part a chain reads, scratch_ref the scratch
    buffer's reference in t."""
    segs = []
    base = slot * SLOT
    for p in it.parts:
        k = p[0]
        if k == "blit":
            segs += t.blit_segments(slot, p[1])
        elif k == "scratch":
            segs.append(seg(abi.ENC_BLIT, b_fill_buf(scratch_ref, 0, 0x80, p[2])
                            + b_copy_buf_buf(scratch_ref, t.out, 0, base + SCRATCH_OFF, 0x80)))
        elif k == "compute":
            segs += t.compute_segments(slot)
        elif k == "chain":
            segs += t.compute_segments(slot, src=t.out, src_off=chain_slot * SLOT)
        elif k == "render":
            segs += t.render_segments(slot, clear=CLEARS[p[1]], draw=p[2])
        elif k == "info":
            cmd = p[1]
            off = base + INFO_OFF
            if cmd == 450:
                r = info_ref(450, t.cpso, t.out, off)
            elif cmd == 457:
                r = info_ref(457, t.rpso, t.out, off)
            else:
                r = info_heap_size(t.out, off, fmt=70, w=256, h=256, usage=1, options=0x20)
            segs.append(seg(abi.ENC_INFO, r))
    return segs


def signature(it):
    """What an item's control dump is keyed by: each part's own inputs."""
    out = []
    for p in it.parts:
        if p[0] == "chain":
            src = next(q for q in p[1].parts if q[0] == "blit")
            out.append(("chain", src[1]))
        elif p[0] == "scratch":
            out.append(("scratch", p[2]))
        else:
            out.append(p)
    return out


PART_AT = {"blit": (0x000, 0x100), "compute": (0x100, 0x100), "chain": (0x100, 0x100),
           "render": (0x200, 0x100), "scratch": (SCRATCH_OFF, 0x80)}


class Emitter:
    def __init__(self, name, doc, seed, f, acts, items, incarnations):
        self.f, self.acts, self.items, self.inc = f, acts, items, incarnations
        self.pseed = seed & 0xFF
        ports = f.ports
        self.x = x = Session(name, doc, ports=max(ports, 1), displays=tuple(range(ports)), setups=f.setups(),
                             host=self.host(), iosfc=f.iosfc, watchdog_s=60)
        self.s = x.s
        self.tasks = {}          # incarnation -> RandTask
        self.fifos = {}          # fid -> ChildFIFO, for good
        self.disp = {p: p for p in range(ports)}   # port -> display index
        self.specs = {p: display(p, name="head%d" % p, modes=x.modes) for p in range(ports)}
        self.displays = list(range(ports))         # the runner's list: port or None
        self.step = 0

    def host(self):
        """Keyed by the host when it reports the host: device info, info
        replies, and frames the host encodes."""
        f = self.f
        return bool(f.ports or f.info or f.device_info)

    # ---- helpers ----

    def mark(self, text):
        """The step's mark; a compact program's marks name only the
        action, to keep its expected files readable."""
        if self.f.compact:
            text = self.acts[self.step][0]
        self.s.mark("%d: %s" % (self.step, text))

    def events(self, label):
        self.s.events(label, sort_blocks=True, blocks_digest=True)

    def child(self, fid):
        c = self.fifos.get(fid)
        if c is None:
            c = self.fifos[fid] = ChildFIFO(self.s, fid)
        c.written = 0
        c.setup_node()
        c.define(label="define child %d" % fid)
        if fid != GATE:
            self.x.children[fid] = c
        return c

    def new_task(self, key, tid, depth):
        t = RandTask(self.s, tid, self.pseed, depth=depth)
        t.define(label="define task %d" % tid)
        t.install(label="task %d's object list" % tid)
        self.tasks[key] = t
        return t

    def segments(self, it, control=False):
        if control:
            t = self.ctrl
            slot = it.cslot()
        else:
            t = self.tasks[it.task]
            slot = it.slot
        chain = next((p[1] for p in it.parts if p[0] == "chain"), None)
        chain_slot = None if chain is None else (chain.cslot() if control else chain.slot)
        sc = next((p[1] for p in it.parts if p[0] == "scratch"), None)
        ref = None
        if sc is not None:
            ref = self.ctrl_scratch if control else t.scratch[sc]
        return t, item_segments(t, slot, it, chain_slot, ref)

    def fifo(self, fid):
        return self.s.fifo if fid == 0 else self.fifos[fid]

    def batch(self, items, serial, label):
        """Session.batch with each item's own opcode: every exec written,
        then the FIFOs rung (serialized: last first, each exec with a
        barrier on the one before it), then every stamp waited for."""
        written = []
        prev = None
        for it in items:
            t, segs = self.segments(it)
            f = self.fifo(it.fid)
            bars = []
            if serial and prev is not None and prev[0] is not f:
                bars.append((prev[0].stamp_index, prev[1]))
            st = f.cmd(it.op, t.exec_payload(segs, it.op), barriers=bars)
            written.append((f, st, t))
            prev = (f, st)
        rung = list(dict.fromkeys(f for f, _, _ in written))
        for f in reversed(rung) if serial else rung:
            f.doorbell()
        for f, st, t in written:
            f.wait_stamp(st, label="%s: FIFO %d stamp %d" % (label, f.fid, st))
            t.reset_cmdbufs()

    def compare(self, it):
        """An item's slot against its control slot (not in a compact
        program, which compares whole tasks at checkpoints)."""
        if self.f.compact:
            return
        t = self.tasks[it.task]
        s = self.s
        a, b = t.slot_gpa(it.slot), self.ctrl.slot_gpa(it.cslot())
        cmd = it.info()
        if cmd is None:
            s.cmp(a, b, SLOT, label="item %d against its control" % it.id)
        else:
            s.cmp(a, b, INFO_OFF, label="item %d against its control" % it.id)
            s.cmp(a + INFO_OFF, b + INFO_OFF, INFO_LEN[cmd], label="item %d: info %d reply" % (it.id, cmd))

    def compare_task(self, key):
        """A compact program's check: every slot task incarnation `key` has
        run so far against its control region (no info work in a compact
        program, so every byte compared is defined)."""
        if not self.f.compact:
            return
        n = self.done.get(key, 0)
        if n:
            t = self.tasks[key]
            self.s.cmp(t.slot_gpa(0), self.ctrl.slot_gpa((key - 1) * SLOTS), n * SLOT,
                       label="task %d's %d slots against the control" % (t.tid, n))

    def ran(self, it):
        self.done[it.task] = max(self.done.get(it.task, 0), it.slot + 1)
        self.compare(it)

    # ---- the control ----

    def control(self):
        s, x = self.s, self.x
        out_pages = max(1, len(self.inc)) * LIVE_OUT_PAGES
        c = self.ctrl = RandTask(s, CTRL_TID, self.pseed, pages=32 + out_pages, out_pages=out_pages)
        self.ctrl_scratch = c.add_scratch(0)
        c.define(label="define the control task")
        c.install(label="the control task's object list")
        # One exec an item; a compact program's control runs eight items
        # an exec, their segments in item order.
        group = 8 if self.f.compact else 1
        for i in range(0, len(self.items), group):
            segs = []
            for it in self.items[i:i + group]:
                segs += self.segments(it, control=True)[1]
            x.run(0, c, segs, "control: " + "; ".join(it.desc() for it in self.items[i:i + group]))
        seen = []
        for it in self.items:
            for p, sig in zip(it.parts, signature(it)):
                if sig in seen:
                    continue
                seen.append(sig)
                if p[0] == "info":
                    s.dump(c.slot_gpa(it.cslot(), INFO_OFF), INFO_LEN[p[1]], "control: info %d" % p[1])
                else:
                    off, ln = PART_AT[p[0]]
                    s.dump(c.slot_gpa(it.cslot(), off), ln, "control: %s" % " ".join(str(v) for v in sig))
        c.delete(label="delete the control task")
        self.events("control done")
        self.done = {}

    # ---- the walk ----

    def emit(self):
        for a in self.acts:
            getattr(self, "a_" + a[0])(*a[1:])
            self.step += 1
        return self.s

    def a_setup(self, tasks, fids):
        x = self.x
        self.mark("set-up")
        x.negotiate(asks=())
        x.root()
        self.control()
        if self.f.ports:
            x.kernel_task()
            for p in range(self.f.ports):
                x.connect(p)
        if self.f.iosfc:
            x.iosfc_start()
        for fid in fids:
            self.child(fid)
        self.gate = self.child(GATE)
        for key, tid, depth in tasks:
            self.new_task(key, tid, depth)
        self.events("set up")

    def a_work(self, items, serial):
        self.mark("work%s: %s" % ("" if serial else " at once", "; ".join(it.desc() for it in items)))
        self.batch(items, serial, "step %d" % self.step)
        for it in items:
            self.ran(it)

    def a_held(self, it, how, broken, *resume):
        s = self.s
        self.mark("held behind the gate%s: %s" % ({"none": "", "pause": " across a pause",
                                                     "suspend": " across a suspend"}[how], it.desc()))
        t, segs = self.segments(it)
        f = self.fifo(it.fid)
        g = self.gate
        st = f.cmd(it.op, t.exec_payload(segs, it.op), barriers=[(g.stamp_index, g.stamp + 1)])
        f.doorbell()
        f.wait_stamp(st, timeout_ms=300, label="item %d held" % it.id)
        if how == "pause":
            s.api("pause")
            s.api("unpause")
            self.events("paused and unpaused over held work")
        elif how == "suspend":
            self.suspend_resume([], *resume)
        g.submit(abi.OP_NOP, label="the gate opens")
        f.wait_stamp(st, label="item %d released" % it.id)
        t.reset_cmdbufs()
        self.ran(it)
        # Children that did not start are recovered once the held work ran:
        # the recovery's commands go on the root FIFO, which may be the one
        # holding it.
        self.recover(broken, "the unpause" if how == "pause" else "the resume")

    def a_present(self, port, frame, enc):
        self.mark("present backing %d on port %d" % (frame, port))
        self.x.present(port, frame, display_index=self.disp[port], encode=enc and not self.f.compact)

    def a_cursor(self, port, cx, cy, glyph):
        self.mark("cursor on port %d at %d, %d%s" % (port, cx, cy, " with its glyph" if glyph else ""))
        self.x.cursor(port, cx, cy, glyph=glyph)

    def a_task_new(self, key, tid, depth):
        self.mark("new task %d (%d-level page table)" % (tid, depth))
        self.new_task(key, tid, depth)

    def a_task_del(self, key, objects):
        t = self.tasks[key]
        self.mark("delete task %d%s" % (t.tid, " and its objects first" if objects else ""))
        self.compare_task(key)
        if objects:
            self.x.delete_objects(t)
        t.delete(label="delete task %d" % t.tid)
        del self.tasks[key]

    def a_scratch_new(self, key, sid):
        t = self.tasks[key]
        self.mark("task %d: a scratch buffer, the object list set again" % t.tid)
        t.add_scratch(sid)
        t.install(label="task %d's longer object list" % t.tid)

    def a_scratch_del(self, key, sid):
        t = self.tasks[key]
        ref = t.scratch.pop(sid)
        t.extra.remove(ref)
        self.mark("task %d: delete scratch buffer %d" % (t.tid, ref))
        self.s.submit(abi.OP_DELETE_OBJECT, struct.pack("<IIII", t.tid, DEL_RESOURCE, 12, ref),
                      label="task %d: delete ref %d" % (t.tid, ref))

    def a_child_new(self, fid):
        self.mark("define child %d" % fid)
        self.child(fid)

    def a_child_del(self, fid):
        self.mark("delete child %d" % fid)
        self.fifos[fid].delete(label="delete child %d" % fid)
        del self.x.children[fid]
        self.s.dump(self.fifos[fid].node_gpa(0), abi.CHILD_NODE_SIZE, "node %d" % fid)
        # Node fid+1 without its read offset, which moves while that child runs.
        self.s.dump(self.fifos[fid].node_gpa(0) + abi.CHILD_NODE_SIZE + 8, abi.CHILD_NODE_SIZE - 8, "node %d from +8" % (fid + 1))

    def recover(self, broken, when):
        """Each child whose page list a delete of the child before it
        zeroed did not start again: its node, a NOP rung on it that does
        not run, then the guest writes its page list again, deletes it and
        defines it."""
        s = self.s
        for fid in broken:
            c = self.fifos[fid]
            s.dump(c.node_gpa(0), abi.CHILD_NODE_SIZE, "child %d after %s" % (fid, when))
            st = c.cmd(abi.OP_NOP)
            c.doorbell()
            c.wait_stamp(st, timeout_ms=300, label="child %d after %s: a NOP is not run" % (fid, when))
            c.written = 0
            c.setup_node()
            c.delete(label="delete child %d, which did not start" % fid)
            c.define(label="define child %d again" % fid)

    def a_pause(self, item, broken):
        s = self.s
        self.mark("pause, unpause" + ("; rung while paused: " + item.desc() if item else ""))
        s.api("pause")
        if item is not None:
            t, segs = self.segments(item)
            f = self.fifo(item.fid)
            st = f.cmd(item.op, t.exec_payload(segs, item.op))
            f.doorbell()
            f.wait_stamp(st, timeout_ms=300, label="item %d rung while paused: not run" % item.id)
        s.api("unpause")
        if item is not None:
            f.wait_stamp(st, label="item %d after unpause" % item.id)
            t.reset_cmdbufs()
            self.ran(item)
        self.events("unpaused")
        self.recover(broken, "the unpause")

    def config(self, ports):
        cfg = copy.deepcopy(self.s._config)
        cfg.pop("op", None)
        cfg["displays"] = [self.specs[p] for p in ports]
        return cfg

    def reindex(self, ports):
        self.displays = list(ports)
        self.disp = {p: i for i, p in enumerate(ports)}

    def suspend_resume(self, broken, glyphs, frames, ports):
        x, s = self.x, self.s
        x.suspend()
        x.resume(config=self.config(ports) if ports != list(range(self.f.ports)) else None)
        self.reindex(ports)
        if glyphs:
            x.expect("cursorGlyph", n=len(glyphs), label="glyphs replayed on ports %s" % glyphs)
            x.expect("cursorShow", n=len(glyphs), label="shows replayed on ports %s" % glyphs)
        if frames:
            x.expect("newFrame", n=len(frames), label="mappings presented again on ports %s" % frames)
        self.events("resumed")
        if not self.f.compact:
            s.sweep(0x1000, 0x1040, label="registers after the resume")
        for port in sorted(x.pages):
            self.page(port, "resumed")
        self.recover(broken, "the resume")

    def a_suspend(self, glyphs, frames, broken, ports):
        self.mark("suspend, resume on a fresh device")
        self.suspend_resume(broken, glyphs, frames, ports)

    def a_will_suspend_unpause(self, broken):
        # The pause ends the threads rvg_device_will_suspend left waiting
        # before unpause starts new ones (the invariants above;
        # docs/testing.md, 7.4).
        self.mark("willSuspend, then pause and unpause")
        self.s.api("willSuspend")
        self.s.api("pause")
        self.s.api("unpause")
        self.events("suspend abandoned")
        self.recover(broken, "the unpause")

    def a_teardown(self, how, keys, fids, ports, displays):
        x, s = self.x, self.s
        self.mark("%s; the guest does its set-up again" % how)
        if how == "recreate":
            # The host stops the old device before releasing it (the
            # invariants above; docs/testing.md, 7.4).
            s.api("stop")
            s.api("recreate", config=self.config(displays))
            self.reindex(displays)
        else:
            s.api(how)
        self.events(how)
        s.sweep(0x1000, 0x1040, label="registers after " + how)
        if how != "stop":
            s.w(abi.R_VERSION, abi.MAX_VERSION)
        s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
        s.fifo.written = 0
        s.fifo.enable()
        x.children = {}
        for fid in fids:
            self.child(fid)
        self.child(GATE)
        if x.t0 is not None:
            x.t0.define(label="define task 0 again")
            x.t0.install(label="task 0's object list again")
        for key in keys:
            t = self.tasks[key]
            t.define(label="define task %d again" % t.tid)
            t.install(label="task %d's object list again" % t.tid)
        x.pages = {}
        for p in ports:
            x.connect(p)
        self.events("set up again")

    def a_device_info(self, max_key):
        self.mark("device info to key %d" % max_key)
        self.x.device_info(max_key=max_key)

    def a_check(self):
        self.mark("checkpoint")
        self.events("checkpoint")
        self.s.fault_state("checkpoint")
        for key in sorted(self.tasks):
            self.compare_task(key)

    def page(self, port, label):
        s = self.s
        if self.f.compact:
            return
        for off, n in ((SS_RAISED, 8), (SS_EXPECTED_ACK, 0x10), (SS_PRESENT_COUNT, 0x28)):
            s.dump(self.x.ss(port, off), n, "port %d: %s (+0x%x)" % (port, label, off))

    def a_port_off(self, port):
        self.mark("disconnect port %d" % port)
        self.x.disconnect(port)
        self.page(port, "disconnected")
        del self.x.pages[port]

    def a_port_on(self, port, attached):
        self.mark("connect port %d%s" % (port, "" if attached else ", no display"))
        self.x.connect(port, has_display=attached)
        self.page(port, "connected")

    def a_surface_map(self, mid, pat):
        self.mark("map surface %d" % mid)
        self.x.surface(mid, pat)

    def a_surface_test(self, mid):
        self.mark("test surface %d" % mid)
        self.x.iosfc_cmd(CMD_TEST, mid, "test %d" % mid)
        self.s.dump(self.s.pa(self.x.surfaces[mid]), 0x40, "surface %d after its test" % mid)

    def a_surface_unmap(self, mid):
        self.mark("unmap surface %d" % mid)
        self.x.iosfc_cmd(CMD_UNMAP, mid, "unmap %d" % mid)

    def a_swap(self, port, mid, enc):
        self.mark("swap to mapping %d on port %d" % (mid, port))
        self.x.swap(port, mid, display_index=self.disp[port], encode=enc and not self.f.compact)

    def a_unplug(self, port, idx):
        self.mark("unplug port %d's display" % port)
        self.x.unplug(idx, port)
        self.displays[idx] = None
        self.page(port, "unplugged")

    def a_hotplug(self, port, ack):
        x, s = self.x, self.s
        self.mark("plug a display into port %d" % port)
        s.api("new_display", spec=self.specs[port])
        s.wait(gpa=x.ss(port, SS_MODE_COUNT), mask=0xFFFF, ne=0, label="modes on port %d" % port)
        x.ack(port, ack)
        self.displays.append(port)
        self.disp[port] = len(self.displays) - 1
        self.page(port, "plugged")

    def a_final(self):
        x, s = self.x, self.s
        self.mark("the end: teardown")
        self.events("end of the walk")
        s.fault_state("end of the walk")
        for key in sorted(self.tasks):
            self.compare_task(key)
        s.dump(s.fifo.base_gpa(), 4 * (GATE + 1), "stamp slots")
        for t in sorted(self.tasks.values(), key=lambda t: t.tid):
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
    for a in acts[1:-1]:
        hist[a[0]] = hist.get(a[0], 0) + 1
    return "%s; %d steps: %s" % (", ".join(f.tags()), len(acts) - 2,
                                 ", ".join("%s %d" % (k, v) for k, v in sorted(hist.items())))


def build(seed, steps, doc):
    f, acts, items, inc = generate(seed, steps)
    name = "interplay_rand_%d" % seed
    e = Emitter(name, doc.strip() + "\n\nThis program: " + summary(f, acts) + ".", seed, f, acts, items, inc)
    s = e.emit()
    if f.compact:
        # A compact program's waits are named by the step's mark alone,
        # but for those expected to time out (held, paused, not started).
        for o in s.ops:
            if o.get("op") == "wait" and o.get("timeout_ms") == 5000:
                o.pop("label", None)
    return s
