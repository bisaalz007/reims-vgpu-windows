"""Seeded walks over the IOSurface host device and its consumers
(scenarios/interplay_iosfcwalk.py), with a model that predicts every
command's status, every surface's bytes and ids, and what each consumer
sees.

The IOSurface host device (docs/testing.md 6.23) is the MMIO device QEMU's
apple-gfx-mmio creates beside the device (apple-gfx-mmio.c:134-197). A
program drives it and its consumers at once, one random legal action a
step:

  - the ring: batches of map (1), unmap (2) and test (3) commands on one
    doorbell, with ids out of range (0, 8192, 0xffffffff), an id whose
    descriptor page the table does not name (NoMemory), a descriptor
    without its valid bit (BadArgument) and commands the device does not
    know (Unsupported), each with its status written back and one
    interrupt;
  - the ring reprogrammed mid-stream: moved to another ring of 1024 or
    2048 entries with the counters kept, so the next command is read at
    the counter modulo the new size, sometimes after the guest wrote
    commands into the old ring and rang only after the move (the device
    runs what the new ring holds at those slots: commands of an earlier
    use, or zeros, which are Unsupported); rewound with 0x1020, written
    only while no ring is mapped, so that commands run again or are
    skipped; and rung while no ring is mapped, which runs nothing until a
    doorbell after the ring is mapped again;
  - the descriptor table: rewritten (0x1010), which drops every surface
    whatever holds it, and the descriptor format switched (0x1030, v1
    0x200-byte and v2 0x400-byte descriptors) with a new table, and a
    doorbell between the two, which runs nothing while the table is
    unset;
  - the consumers: a GPU task per surface with a mapper buffer (type 15)
    over it and a source buffer, whose fills and copies land in the
    surface's guest page; opcode 8 presents of the surfaces and opcode 7
    presents of task 0's backings on one or two heads; the buffer deleted
    (37) and made again by the next exec that lists it;
  - held work: fills and presents on child FIFOs and the root FIFO, some
    held on a gate FIFO, some rung while the device is paused, while the
    guest unmaps, remaps, tests, rewrites or tears down the surfaces they
    name;
  - the device's lifecycle: pause (with ring commands, rewrites and work
    rung while paused), the suspend onto a fresh device (work held on the
    gate across it) and reset, with the guest's set-up again.
  - Some programs end in a death docs/testing.md 7.3 and 7.4 list, with
    work held on the gate: the ring's size written unaligned (the memory
    map asserts, iosfc_ring_size_unaligned), a present of a surface wider
    than Metal's largest texture (mutgap_disp_mapper_texture_refused_dies),
    or a second pause.

The model:
  - the device: its registers (ring base and size, whether the ring is
    mapped, the write and read counters, descriptor base and enable),
    each ring's 16-byte entries as guest memory holds them, which
    descriptor page the table names, each descriptor's mapped bit (set
    by a map, cleared by an unmap, left alone by a teardown), and each
    id's host surface as a generation (a map makes a new one, an unmap
    or a teardown drops it);
  - holders: a surface is Busy to unmap while its current generation is
    held by a head showing it or by a task's mapper buffer. Measured with
    this module's first programs (map and isolate alike): a head holds the
    surface of its last accepted opcode 8 until it shows something else,
    its page is disconnected, the device is reset (the frame is dropped:
    an encode is black) or released; a mapper buffer holds from the first
    exec that lists it until 37 deletes it or the device goes (reset, or
    the suspend's fresh device); a teardown leaves both holding the old
    surface, over the same guest page, so their writes and views still
    reach the page while the new surface of the id is not held;
  - under isolation the host holds a buffer's surface through the remote
    task's delegate, whose table keeps the first answer for an id, nil
    included: after an exec whose buffer found no surface, a buffer made
    later is held by the service only, and an unmap the in-process setup
    refuses (Busy) goes through, until 37 releases the buffer and erases
    the entry (interplay_iosfcwalk_refused_retain). One Model runs per
    setup; where their statuses split the ids mapped, the guest maps the
    split ids again at once (both accept), so every later wait means the
    same in both, and the setups' own statuses and bytes are predicted
    each;
  - mode_change: a display's first frame, and the first opcode 8 on a page
    connected with opcode 1 (the guest display's guest colour space starts
    dirty and only opcode 8 hands it to the display), so a port that showed
    backings first calls it again at its first surface; a guest display the
    resume restores starts clean (measured);
  - an exec that lists a buffer not made while its id has no surface
    faults its child FIFO ("Attempting to create mapper ref against
    invalid surface"); the guest maps the id again or not, resumes the
    child (0x1028) and the faulted stamp is published, with the page
    untouched;
  - an opcode 8 of an id with no surface is signalled with no callback,
    and the head keeps its frame; the page's mapping stays the last
    accepted one, which the resume re-presents when its id has a surface
    then (a new_frame, and the new head holds it), and not otherwise;
  - every surface's 512 bytes (a mirror page and a cmp), each head's
    encoded frame (no gamma table: the display copies the surface,
    alpha included, docs/testing.md 6.21), the callbacks and IOSurface
    interrupts of every events checkpoint, the registers, the
    descriptors' first words and every ring entry the device ran.
python3 -m pgtest.iosfcwalk_util reads the expected files back against it.

Invariants the generator keeps, so every checkpoint is deterministic:
  - Work that does not run at once forms one chain: each item carries a
    barrier on the one before it (and on its FIFO's previous command), so
    the chain runs in order, and a fault holds what follows until the
    guest resumes the child; work that runs at once goes only on a FIFO
    with nothing held and is waited for before the next step.
  - At most one present in the chain, so new_frame never coalesces and
    callbacks arrive in one order.
  - Fills go on child FIFOs only, so every fault is a child's.
  - Ring commands are rung only when the walker waits for their
    interrupts (or shows that none come); a register is never written
    while the device runs commands.

Nothing here is imported from the library's code.
"""
import hashlib
import json
import struct

from . import abi
from .abi import PAGE
from .diff_util import exec_payload
from .display_guest import p_present3, p_swap, pattern
from .fifo import ChildFIFO
from .iosfc_guest import (BAD_ARGUMENT, BUSY, CMD_MAP, CMD_TEST, CMD_UNMAP, I_DESC_BASE, I_DESC_ENABLE, I_ONE,
                          I_READ, I_RING_BASE, I_RING_SIZE, I_WRITE, NO_MEMORY, SUCCESS, UNSUPPORTED, desc_v1,
                          desc_v2, pte)
from .lifetime_util import delete_resource
from .randgen import Rng
from .resources_build import b_copy_buf_buf, b_fill_buf, mapper_ref_rec
from .scanout_util import FRAME, MAPPED, Scanout
from .session import FH, FW
from .task import GPUTask, seg

GATE = 7
WORKERS = (1, 2, 3)
HOLD_MS = 300
HELD = " [held]"
DIES = " [dies]"
POOL = (1, 2, 3, 5, 15, 16, 17, 33, 47, 100, 8191)
INVALID_ID = 4          # its descriptor has no valid bit
UNLINKED_ID = 4000      # its descriptor page is not in the table (v1 page 125, v2 page 250)
WIDE_ID = 60            # the death's 16400 x 1 surface
BAD_IDS = (0, 8192, 0xFFFFFFFF)
BAD_CMDS = (0, 4, 9, 0x100)
DEATHS = ("ring_unaligned", "wide_present", "pause_twice")
POISON = 0x7EADBEEF     # the status word the guest writes with each command
HEALTHY = 0xFFFFFFFF
NODE_FAULT = 8
M32 = 0xFFFFFFFF
BGRA = "0x42475241"
ABSENT = "absent"       # no entry in the host delegate's surface table
NIL = "nil"             # an entry holding nil (a retain the device refused)


def test_bytes(n=FRAME):
    """What test (3) writes into a surface of n bytes: i x 8 into each
    8-byte word."""
    return b"".join(struct.pack("<Q", 8 * i) for i in range(n // 8))


class Table:
    """A descriptor table's layout: its page, the descriptor pages it names
    (linked when a descriptor is first written there), the format."""

    def __init__(self, s, v2, key):
        self.s = s
        self.v2 = v2
        self.key = key
        self.page = s.alloc_page()
        self.pages = {}

    def per_page(self):
        return PAGE // (0x400 if self.v2 else 0x200)

    def gpa(self, mid):
        pi, idx = divmod(mid, self.per_page())
        return self.s.pa(self.pages[pi], idx * (0x400 if self.v2 else 0x200))

    def linked(self, mid):
        return mid // self.per_page() in self.pages

    def put(self, mid, root, w=FW, h=FH, valid=1):
        """Writes the descriptor; returns its first word."""
        s = self.s
        pi = mid // self.per_page()
        if pi not in self.pages:
            self.pages[pi] = s.alloc_page()
            s.u32(s.pa(self.page, 4 * pi), pte(s.pfn(self.pages[pi])))
        mk = desc_v2 if self.v2 else desc_v1
        s.mem(self.gpa(mid), mk(root, w, h, valid=valid))
        return (valid & 1) | (root & 0x3FFFFFFF) << 2


class Ring:
    """A command ring over whole pages."""

    def __init__(self, s, entries, key):
        self.s = s
        self.entries = entries
        self.key = key
        self.page = s.alloc_page(entries * 16 // PAGE)

    def gpa(self, slot=0):
        return self.s.pa(self.page, 16 * slot)


class Head:
    def __init__(self, port):
        self.port = port
        self.frame = None        # ("surf", mid, gen) | ("back", k)
        self.known = False       # the display has had a frame: a new size or format calls mode_change
        self.dirty = True        # the guest display's guest colour space is dirty: the next opcode 8 calls mode_change
        self.mapping = None      # the page's last accepted opcode-8 mapping


class Model:
    """What one host setup does. In-process (map) a task's mapper buffer
    retains its surface from the device; under isolation the service asks
    the host, which keeps each id's surface in a table (reims-vgpu-core's
    tasks::remote::endpoint): an entry already there, even nil from a
    retain the device refused, is kept and the new surface is not held; the
    service's release of its last reference erases the entry."""

    def __init__(self, setup, mids, ports):
        self.setup = setup
        self.iso = setup == "isolate"
        self.gen = {m: None for m in mids}
        self.raw = {}
        self.word0 = {}          # (table key, mid) -> the descriptor's first word
        self.mem = {}            # (ring key, slot) -> [command, id, status, 0]
        self.made = {m: None for m in mids}
        self.host = {m: ABSENT for m in mids}
        self.heads = {p: Head(p) for p in range(ports)}
        self.cbs = []
        self.ngen = 0

    def new_gen(self):
        self.ngen += 1
        return self.ngen

    def entry_hex(self, ring, slot):
        c, m, st, z = self.mem.get((ring.key, slot), [0, 0, 0, 0])
        return struct.pack("<IIII", c & M32, m & M32, st & M32, z & M32).hex()

    def held(self, mid):
        out = set()
        for h in self.heads.values():
            if h.frame and h.frame[0] == "surf" and h.frame[1] == mid:
                out.add(h.frame[2])
        if self.iso:
            if self.host.get(mid) not in (ABSENT, NIL, None):
                out.add(self.host[mid])
        elif self.made.get(mid) is not None:
            out.add(self.made[mid])
        return out

    def execute(self, w, c, mid):
        if c == CMD_MAP:
            return self.do_map(w, mid)
        if c == CMD_UNMAP:
            return self.do_unmap(w, mid)
        if c == CMD_TEST:
            return self.do_test(w, mid)
        return UNSUPPORTED

    def do_map(self, w, mid):
        if mid == 0 or mid >= 8192:
            return BAD_ARGUMENT
        if self.gen.get(mid) is not None:
            return SUCCESS
        t = w.dtab
        if not t.linked(mid):
            return NO_MEMORY
        k = (t.key, mid)
        if k not in self.word0 or not (self.word0[k] & 1):
            return BAD_ARGUMENT
        self.gen[mid] = self.new_gen()
        self.word0[k] |= 2
        return SUCCESS

    def do_unmap(self, w, mid):
        if mid == 0 or mid >= 8192:
            return BAD_ARGUMENT
        if self.gen.get(mid) is None:
            return SUCCESS
        if self.gen[mid] in self.held(mid):
            return BUSY
        self.gen[mid] = None
        self.word0[(w.dtab.key, mid)] &= ~2
        return SUCCESS

    def do_test(self, w, mid):
        if mid == 0 or mid >= 8192:
            return BAD_ARGUMENT
        if self.gen.get(mid) is None:
            return BAD_ARGUMENT
        self.raw[mid][:] = test_bytes()
        w.touched.add(mid)
        return SUCCESS

    def frame_cbs(self, h, mapped):
        if not h.known or (mapped and h.dirty):
            self.cbs.append({"cb": "modeChange", "display": h.port, "fmt": BGRA, "h": FH, "w": FW})
            h.known = True
        if mapped:
            h.dirty = False
        self.cbs.append({"cb": "newFrame", "display": h.port})


class FillTask(GPUTask):
    """A GPU task over one surface: a mapper buffer (type 15) over its id and
    a source buffer of the task's pattern; command buffers go round pages
    4-7 in 0x100 steps, so held execs keep theirs."""

    def __init__(self, s, tid, mid):
        super().__init__(s, tid=tid, pages=16)
        self.mid = mid
        self.mbuf = self.ol.add(abi.T_MAPPER_BUFFER, mapper_ref_rec(mid))
        self.src_bytes = bytes((i * 11 + tid * 29) & 0xFF for i in range(0x800))
        self.src, self.src_va = self.buffer(PAGE, contents=self.src_bytes)
        self.slot = 0

    def cmdbuf(self, data):
        va = 4 * PAGE + 0x100 * (self.slot % 256)
        self.slot += 1
        assert len(data) <= 0x100
        self.write(va, data)
        return (va, len(data))


class Item:
    """One command of GPU or display work: kind fill or present."""

    def __init__(self, iid, kind, fid):
        self.id, self.kind, self.fid = iid, kind, fid
        self.mid = None
        self.frame = None
        self.port = None
        self.ops = []
        self.gated = False
        self.stamp = None
        self.fifo = None
        self.offset = None       # the command's offset in its child's ring
        self.state = "new"       # new, queued, done, faulted, abandoned

    def desc(self):
        if self.kind == "fill":
            w = "fill surface %d: %s" % (self.mid, ", ".join(
                ("fill %x+%x %02x" % (o[1], o[2], o[3])) if o[0] == "fill" else ("copy src+%x -> %x+%x" % o[1:])
                for o in self.ops))
        elif self.mid is not None:
            w = "opcode 8 of %d on port %d" % (self.mid, self.port)
        else:
            w = "opcode 7 of backing %d on port %d" % (self.frame, self.port)
        return "i%d F%d %s%s" % (self.id, self.fid, w, " (gated)" if self.gated else "")


class Features:
    def __init__(self, seed, steps, death):
        r = Rng(seed ^ 0x105FC0A1C)
        self.steps = steps
        self.death = death
        self.ports = r.choice((1, 2, 2))
        self.v2 = r.chance(0.6)
        n = 2 + r.below(3)
        self.mids = sorted(r.sample(POOL, n))
        self.entries = r.choice((1024, 1024, 2048))
        self.gate_p = r.choice((0.3, 0.5, 0.7))
        self.lifecycle = r.choice((0.5, 1.0, 1.5))

    def tags(self):
        t = ["ports=%d" % self.ports, "descriptors v%d" % (2 if self.v2 else 1),
             "ids %s" % ",".join(str(m) for m in self.mids), "ring %d" % self.entries,
             "gate p %g" % self.gate_p, "lifecycle weight %g" % self.lifecycle]
        if self.death:
            t.append("ends in death: " + self.death)
        return t


class Walk:
    def __init__(self, seed, steps, doc, death=None):
        self.seed = seed
        self.f = f = Features(seed, steps, death)
        self.r = Rng(seed)
        self.name = "interplay_iosfcwalk_%d" % seed
        self.doc = doc.strip() + "\n\nThis program: " + "; ".join(f.tags()) + "."
        self.x = x = Scanout(self.name, self.doc, ports=f.ports, setups=MAPPED, host=True, iosfc=True,
                             modes=((64, 32, 60), (FW, FH, 60)), watchdog_s=110)
        self.s = x.s
        self.models = [Model(st, f.mids, f.ports) for st in MAPPED]
        self.pred = {}
        self.irqs = 0            # IOSurface interrupts since the last events
        self.total_irqs = 0
        self.items = []
        self.queue = []          # items not yet run, in chain order
        self.prev = {}           # fid -> the stamp of the FIFO's last command
        self.paused = False
        self.dies = False
        self.step_no = 0
        self.hist = {}
        self.touched = set()
        self.reframed = set()
        self.diverged = 0

    # ---- predictions ----

    def predict(self, **fields):
        self.pred[len(self.s.ops) - 1] = fields

    def predict_each(self, fn):
        """A prediction per setup: fn(model) gives the fields."""
        per = {m.setup: fn(m) for m in self.models}
        vals = list(per.values())
        self.pred[len(self.s.ops) - 1] = vals[0] if all(v == vals[0] for v in vals) else {"by_setup": per}

    def count(self, k):
        self.hist[k] = self.hist.get(k, 0) + 1

    def mark(self, text):
        lab = "step %d: %s" % (self.step_no, text)
        self.s.mark(lab)
        return lab

    def events(self, label):
        self.s.events(label)
        irqs = self.irqs or None
        self.predict_each(lambda m: {"iosfc_irqs": irqs, "callbacks": m.cbs or None})
        for m in self.models:
            m.cbs = []
        self.irqs = 0

    def same(self, fn, what):
        vals = [fn(m) for m in self.models]
        assert all(v == vals[0] for v in vals), "%s: the setups diverge: %s" % (what, vals)
        return vals[0]

    # ---- set-up ----

    def build(self):
        f, x, s = self.f, self.x, self.s
        x.negotiate(asks=())
        x.root()
        x.kernel_task()
        for p in range(f.ports):
            x.connect(p)
        self.tables = []
        self.rings = []
        self.new_ring(f.entries)
        self.ring = None         # the ring the device has mapped (base and size written)
        self.base = None         # the ring whose base 0x1000 holds (None: 0)
        self.size_reg = 0
        self.W = self.R = 0
        self.enable = 1 if f.v2 else 0
        self.dtab = None
        s.iosfc_w(I_DESC_ENABLE, self.enable)
        t = self.new_table(f.v2)
        self.set_table(t)
        self.program_ring(self.rings[0])
        self.pages = {}
        for mid in f.mids:
            root, data = self.surface_pages()
            raw = bytes(self.r.below(256) for _ in range(FRAME))
            s.mem(s.pa(data), raw)
            self.pages[mid] = (data, root, s.alloc_page())
            for m in self.models:
                m.raw[mid] = bytearray(raw)
        self.write_descs(t)
        self.mark("map every surface")
        self.ring_cmds([(CMD_MAP, m) for m in f.mids], "map every surface")
        self.tasks = {}
        for i, mid in enumerate(f.mids):
            tk = FillTask(s, 2 + i, mid)
            tk.define(label="define task %d (surface %d)" % (tk.tid, mid))
            tk.install(label="task %d's object list" % tk.tid)
            self.tasks[mid] = tk
        self.children = {}
        for fid in WORKERS:
            self.define_child(fid)
        self.define_child(GATE)
        x.gate = self.children[GATE]
        self.events("set up")
        self.check_all("set up")
        for self.step_no in range(1, f.steps + 1):
            self.step()
        self.step_no = f.steps + 1
        if f.death:
            self.die(f.death)
        else:
            self.finish()
        s.doc = self.doc + " Steps: " + self.summary() + "."
        return s

    def new_ring(self, entries):
        g = Ring(self.s, entries, len(self.rings))
        self.rings.append(g)
        return g

    def new_table(self, v2):
        t = Table(self.s, v2, len(self.tables))
        self.tables.append(t)
        return t

    def write_descs(self, t):
        """Every surface's descriptor, and the invalid one, in table t."""
        for mid in self.f.mids:
            w0 = t.put(mid, self.pages[mid][1])
            for m in self.models:
                m.word0[(t.key, mid)] = w0
        w0 = t.put(INVALID_ID, self.pages[self.f.mids[0]][1], valid=0)
        for m in self.models:
            m.word0[(t.key, INVALID_ID)] = w0

    def surface_pages(self, n=1):
        s = self.s
        pt = s.alloc_page()
        data = s.alloc_page(n)
        s.mem(s.pa(pt), b"".join(struct.pack("<I", pte(s.pfn(data + i))) for i in range(n)))
        return s.pfn(pt), data

    def define_child(self, fid):
        s = self.s
        c = self.children.get(fid)
        if c is None:
            c = self.children[fid] = ChildFIFO(s, fid)
            s.fifos[fid] = c
        c.written = 0
        self.prev[fid] = None
        c.setup_node()
        c.define(label="define child %d" % fid)
        self.x.children[fid] = c
        return c

    def fifo(self, fid):
        return self.s.fifo if fid == 0 else self.children[fid]

    # ---- the device's registers ----

    def set_table(self, table):
        """0x1010: every surface dropped, the descriptor pages forgotten,
        the new table mapped."""
        self.s.iosfc_w(I_DESC_BASE, self.s.pa(table.page) if table else 0)
        self.drop_all()
        self.dtab = table

    def drop_all(self):
        for m in self.models:
            for mid in m.gen:
                m.gen[mid] = None

    def program_ring(self, ring):
        """0x1000 then 0x1008: the ring mapped, the counters kept."""
        s = self.s
        s.iosfc_w(I_RING_BASE, ring.gpa())
        s.iosfc_w(I_RING_SIZE, ring.entries)
        self.base = ring
        self.ring = ring
        self.size_reg = ring.entries

    def unmap_ring(self):
        self.s.iosfc_w(I_RING_BASE, 0)
        self.base = None
        self.ring = None

    def regs(self, label):
        s = self.s
        vals = {I_RING_BASE: self.base.gpa() if self.base else 0, I_RING_SIZE: self.size_reg,
                I_DESC_BASE: self.s.pa(self.dtab.page) if self.dtab else 0, I_WRITE: self.W, I_READ: self.R,
                I_ONE: 1, I_DESC_ENABLE: self.enable}
        for off in (I_RING_BASE, I_RING_SIZE, I_DESC_BASE, I_WRITE, I_READ, I_ONE, I_DESC_ENABLE):
            s.iosfc_r(off, label)
            self.predict(val=hex(vals[off] & M32))

    # ---- the ring ----

    def write_cmds(self, ring, cmds):
        """Writes commands at the write counter into `ring` (guest memory
        and the models), not rung; returns their counters."""
        s = self.s
        out = []
        for c, mid in cmds:
            slot = self.W % ring.entries
            for m in self.models:
                m.mem[(ring.key, slot)] = [c, mid, POISON, 0]
            s.mem(ring.gpa(slot), struct.pack("<IIII", c & M32, mid & M32, POISON, 0))
            out.append(self.W)
            self.W += 1
        return out

    def bell(self):
        """0x1018 = the write counter: runs the ring from the read counter
        when the ring and the table are mapped; returns the
        entries it ran."""
        self.s.iosfc_w(I_WRITE, self.W)
        ran = []
        if self.ring is None or self.dtab is None or not self.size_reg:
            return ran
        g = self.ring
        while self.R < self.W:
            slot = self.R % self.size_reg
            for m in self.models:
                e = m.mem.get((g.key, slot), [0, 0, 0, 0])
                st = m.execute(self, e[0], e[1])
                m.mem[(g.key, slot)] = [e[0], e[1], st, e[3]]
            ran.append((g, slot))
            self.R += 1
            self.irqs += 1
            self.total_irqs += 1
        return ran

    def wait_ran(self, ran, label):
        s = self.s
        if ran:
            s.wait(iosfc_irqs=self.total_irqs, label="%s: iosfc irqs %d" % (label, self.total_irqs))
        for g, slot in ran:
            s.dump(g.gpa(slot), 16, label)
            self.predict_each(lambda m: {"hex": m.entry_hex(g, slot)})
        if ran:
            self.converge(label)

    def converge(self, label):
        """The setups keep the same ids mapped: where a status differed
        (an unmap Busy in one and done in the other), the guest maps the id
        again, which both accept, so what later work waits for is the same
        in both."""
        split = [mid for mid in self.f.mids if len(set(m.gen[mid] is None for m in self.models)) > 1]
        if not split:
            return
        self.diverged += 1
        self.count("setups diverged")
        lab = label + ": the setups differ on %s: map again" % split
        self.write_cmds(self.ring, [(CMD_MAP, mid) for mid in split])
        ran = self.bell()
        self.wait_ran(ran, lab)

    def ring_cmds(self, cmds, label):
        """Commands written and rung on the mapped ring, waited for; their
        entries dumped."""
        self.write_cmds(self.ring, cmds)
        ran = self.bell()
        self.wait_ran(ran, label)
        return ran

    # ---- checkpoints ----

    def cmp_surface(self, mid, label):
        """The model's bytes into the mirror page, then a cmp; where the
        setups' bytes differ, a dump instead."""
        s = self.s
        data, _, mirror = self.pages[mid]
        raws = [bytes(m.raw[mid]) for m in self.models]
        if all(r == raws[0] for r in raws):
            s.mem(s.pa(mirror), raws[0])
            s.cmp(s.pa(data), s.pa(mirror), FRAME, label="%s: surface %d against the model" % (label, mid))
            self.predict(equal=True)
        else:
            s.dump(s.pa(data), FRAME, "%s: surface %d (the setups differ)" % (label, mid))
            self.predict_each(lambda m: {"hex": bytes(m.raw[mid]).hex()})

    def encode(self, port, label):
        if self.same(lambda m: m.heads[port].frame is None, "a frame on head %d" % port):
            return
        self.s.encode(port, FW, FH, label="%s: head %d" % (label, port),
                      samples=[[0, 0], [FW - 1, FH - 1], [5, 3]])

        def img(m):
            fr = m.heads[port].frame
            b = bytes(m.raw[fr[1]]) if fr[0] == "surf" else pattern(FW, FH, 0x30 * fr[1] + 1)
            return {"ok": True, "status": 4, "sha256": hashlib.sha256(b).hexdigest(), "hex": b.hex()}
        self.predict_each(img)

    def descriptors(self, label):
        s = self.s
        for mid in self.f.mids:
            if self.dtab is None:
                continue
            s.dump(self.dtab.gpa(mid), 4, "%s: descriptor %d's first word" % (label, mid))
            k = (self.dtab.key, mid)
            self.predict_each(lambda m: {"hex": struct.pack("<I", m.word0[k]).hex()})

    def showing(self, port, mids):
        return any(m.heads[port].frame and m.heads[port].frame[0] == "surf" and m.heads[port].frame[1] in mids
                   for m in self.models)

    def settle(self, label):
        """The step's checkpoint: events, every surface the step touched
        against the model, and every head that shows one of them (or that
        the step changed) encoded."""
        self.events(label)
        for mid in sorted(self.touched):
            self.cmp_surface(mid, label)
        for p in range(self.f.ports):
            if p in self.reframed or self.showing(p, self.touched):
                self.encode(p, label)
        self.touched = set()
        self.reframed = set()

    def check_all(self, label):
        self.regs(label)
        self.descriptors(label)
        for mid in self.f.mids:
            self.cmp_surface(mid, label)
        for p in range(self.f.ports):
            self.encode(p, label)

    # ---- items ----

    def held_fifos(self):
        return set(it.fid for it in self.queue)

    def new_item(self, kind, fid):
        it = Item(len(self.items), kind, fid)
        self.items.append(it)
        return it

    def fill_ops(self):
        r = self.r
        ops = []
        for _ in range(1 + r.below(2)):
            if r.chance(0.6):
                off = 16 * r.below(FRAME // 16)
                n = 16 * (1 + r.below((FRAME - off) // 16))
                ops.append(("fill", off, n, r.below(256)))
            else:
                doff = 16 * r.below(FRAME // 16)
                n = 16 * (1 + r.below((FRAME - doff) // 16))
                ops.append(("copy", 16 * r.below(0x30), doff, n))
        return ops

    def write_item(self, it, gate):
        """Writes it (not rung): barriers on its FIFO's previous command,
        on the chain's last item, and on the gate."""
        x = self.x
        f = self.fifo(it.fid)
        bars = []
        if self.prev.get(it.fid) is not None:
            bars.append((f.stamp_index, self.prev[it.fid]))
        if self.queue:
            last = self.queue[-1]
            bars.append((last.fifo.stamp_index, last.stamp))
        if gate:
            it.gated = True
            bars.append(x.gate_barrier())
        if it.fid:
            it.offset = f.written % f.ring_size()
        if it.kind == "fill":
            t = self.tasks[it.mid]
            recs = b""
            for o in it.ops:
                if o[0] == "fill":
                    recs += b_fill_buf(t.mbuf, o[1], o[2], o[3])
                else:
                    recs += b_copy_buf_buf(t.src, t.mbuf, o[1], o[2], o[3])
            cb = t.cmdbuf(seg(abi.ENC_BLIT, recs))
            it.stamp = f.cmd(abi.OP_EXEC_INDIRECT3, exec_payload(t.tid, [cb], [(t.src, 0), (t.mbuf, 0)]),
                             barriers=bars)
        elif it.mid is not None:
            it.stamp = f.cmd(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(it.port, it.mid), barriers=bars)
        else:
            it.stamp = f.cmd(abi.OP_DISPLAY_TRANSACTION3, p_present3(it.port, self.x.backings[it.frame][0]),
                             barriers=bars)
        it.fifo = f
        self.prev[it.fid] = it.stamp

    def run_item(self, it, label):
        """The models run it (its FIFO reached it) and the walker waits for
        what the guest sees: the stamp, a new_frame, or the fault, the
        recovery and the stamp."""
        s, x = self.s, self.x
        lab = "%s: i%d" % (label, it.id)
        if it.kind == "fill":
            mid = it.mid
            fault = self.same(lambda m: m.made[mid] is None and m.gen[mid] is None, "i%d faults" % it.id)
            if fault:
                for m in self.models:
                    if m.iso and m.host[mid] == ABSENT:
                        m.host[mid] = NIL
                c = it.fifo
                s.wait(gpa=c.node_gpa(NODE_FAULT), ne=HEALTHY, label=lab + ": child %d faulted" % it.fid)
                s.wait(mmio=abi.R_FAULT_MASK, mmio_ne=0, label=lab + ": 0x102c")
                s.dump(c.node_gpa(NODE_FAULT), 4, lab + ": the fault offset")
                self.predict(hex=struct.pack("<I", it.offset).hex())
                if self.r.chance(0.5):
                    self.ring_cmds([(CMD_MAP, mid)], lab + ": the guest maps %d again" % mid)
                s.w(abi.R_RESUME_CHILD, it.fid)
                c.wait_stamp(it.stamp, label=lab + ": the faulted stamp published")
                it.state = "faulted"
                self.count("fault")
                return
            for m in self.models:
                if m.made[mid] is None:
                    m.made[mid] = m.gen[mid]
                    if m.iso and m.host[mid] == ABSENT:
                        m.host[mid] = m.gen[mid]
                t = self.tasks[mid]
                for o in it.ops:
                    if o[0] == "fill":
                        m.raw[mid][o[1]:o[1] + o[2]] = bytes([o[3]]) * o[2]
                    else:
                        m.raw[mid][o[2]:o[2] + o[3]] = t.src_bytes[o[1]:o[1] + o[3]]
            it.fifo.wait_stamp(it.stamp, label=lab)
            self.touched.add(mid)
            it.state = "done"
            return
        it.fifo.wait_stamp(it.stamp, label=lab)
        it.state = "done"
        if it.mid is not None:
            mid = it.mid
            ok = self.same(lambda m: m.gen.get(mid) is not None, "i%d accepted" % it.id)
            if not ok:
                self.count("refused present")
                return
            for m in self.models:
                h = m.heads[it.port]
                h.frame = ("surf", mid, m.gen[mid])
                h.mapping = mid
                m.frame_cbs(h, True)
        else:
            for m in self.models:
                h = m.heads[it.port]
                h.frame = ("back", it.frame)
                m.frame_cbs(h, False)
        x.expect("newFrame", label=lab + ": newFrame on head %d" % it.port)
        self.reframed.add(it.port)

    def resolve_queue(self, label):
        for it in list(self.queue):
            self.run_item(it, label)
        self.queue = []

    # ---- the steps ----

    def step(self):
        r, f = self.r, self.f
        self.touched = set()
        self.reframed = set()
        table = [("cmds", 10), ("fill", 7), ("present", 5), ("rewrite", 2), ("teardown", 2), ("format", 1),
                 ("relocate", 2), ("stale", 1), ("rewind", 2), ("unmapped_bell", 1), ("regs", 1),
                 ("delete_buf", 2)]
        if self.queue:
            table.append(("open", 5))
        lw = f.lifecycle
        table += [("pause", int(3 * lw)), ("suspend", int(2 * lw)), ("reset", int(1 * lw))]
        table = [(k, w) for k, w in table if w > 0]
        a = r.weighted(table)
        getattr(self, "a_" + a)()
        if self.step_no % 12 == 0:
            self.check_all("step %d: every surface" % self.step_no)

    # -- the ring --

    def pick_cmd(self):
        r = self.r
        k = r.weighted([("good", 14), ("bad_id", 1), ("unlinked", 1), ("invalid", 1), ("unknown", 1)])
        if k == "good":
            c = r.weighted([(CMD_MAP, 4), (CMD_UNMAP, 4), (CMD_TEST, 3)])
            return c, r.choice(self.f.mids)
        if k == "bad_id":
            return r.choice((CMD_MAP, CMD_UNMAP, CMD_TEST)), r.choice(BAD_IDS)
        if k == "unlinked":
            return r.choice((CMD_MAP, CMD_UNMAP, CMD_TEST)), UNLINKED_ID
        if k == "invalid":
            return CMD_MAP, INVALID_ID
        return r.choice(BAD_CMDS), r.choice(self.f.mids)

    def cmd_text(self, cmds):
        names = {CMD_MAP: "map", CMD_UNMAP: "unmap", CMD_TEST: "test"}
        return ", ".join("%s %d" % (names.get(c, "cmd %d" % c), m) for c, m in cmds)

    def a_cmds(self):
        self.count("ring")
        cmds = [self.pick_cmd() for _ in range(1 + self.r.below(5))]
        lab = self.mark("ring: " + self.cmd_text(cmds))
        self.ring_cmds(cmds, lab)
        if self.r.chance(0.3):
            self.descriptors(lab)
        self.settle(lab)

    def a_relocate(self):
        """The ring moved to another (or a new one), the counters kept."""
        r = self.r
        self.count("ring moved")
        if len(self.rings) < 3 and r.chance(0.5):
            self.new_ring(r.choice((1024, 2048)))
        cands = [g for g in self.rings if g is not self.ring] or self.rings
        g = r.choice(cands)
        lab = self.mark("the ring moved to ring %d (%d entries) at counters %d/%d" % (
            g.key, g.entries, self.W, self.R))
        self.program_ring(g)
        cmds = [self.pick_cmd() for _ in range(1 + r.below(3))]
        self.ring_cmds(cmds, lab + ": " + self.cmd_text(cmds))
        self.settle(lab)

    def a_stale(self):
        """Commands written into the ring, then the ring moved before the
        doorbell: the device runs what the new ring holds at those slots."""
        r, s = self.r, self.s
        self.count("ring moved before the doorbell")
        if len(self.rings) < 3 and r.chance(0.4):
            self.new_ring(r.choice((1024, 2048)))
        cands = [g for g in self.rings if g is not self.ring]
        if not cands:
            cands = [self.new_ring(r.choice((1024, 2048)))]
        g = r.choice(cands)
        old = self.ring
        cmds = [self.pick_cmd() for _ in range(1 + r.below(3))]
        lab = self.mark("written into ring %d (%s), then the ring moved to ring %d and rung" % (
            old.key, self.cmd_text(cmds), g.key))
        ctrs = self.write_cmds(old, cmds)
        self.program_ring(g)
        ran = self.bell()
        self.wait_ran(ran, lab)
        for c in ctrs:
            slot = c % old.entries
            s.dump(old.gpa(slot), 16, lab + ": the old ring's entry, never run")
            self.predict_each(lambda m: {"hex": m.entry_hex(old, slot)})
        self.settle(lab)

    def a_rewind(self):
        """0x1020 with no ring mapped: the read counter moved back (the
        commands run again) or up to the write counter (the commands
        written and not rung are skipped)."""
        r, s = self.r, self.s
        g = self.ring
        back = min(self.R, 1 + r.below(4))
        if r.chance(0.4) or back == 0:
            self.count("ring skipped")
            cmds = [self.pick_cmd() for _ in range(1 + r.below(3))]
            lab = self.mark("skip: %s written, the read counter set past them" % self.cmd_text(cmds))
            ctrs = self.write_cmds(g, cmds)
            self.unmap_ring()
            s.iosfc_w(I_READ, self.W)
            self.R = self.W
            self.program_ring(g)
            self.bell()
            self.events(lab + ": nothing runs")
            for c in ctrs:
                slot = c % g.entries
                s.dump(g.gpa(slot), 16, lab + ": skipped")
                self.predict_each(lambda m: {"hex": m.entry_hex(g, slot)})
        else:
            self.count("ring rewound")
            lab = self.mark("rewind the read counter by %d: those commands run again" % back)
            self.unmap_ring()
            s.iosfc_w(I_READ, self.R - back)
            self.R -= back
            self.program_ring(g)
            ran = self.bell()
            self.wait_ran(ran, lab)
        self.settle(lab)

    def a_unmapped_bell(self):
        """Rung while no ring is mapped: nothing runs, mapping the ring
        again runs nothing, the next doorbell runs them."""
        r, s = self.r, self.s
        self.count("rung unmapped")
        g = self.ring
        cmds = [self.pick_cmd() for _ in range(1 + r.below(3))]
        lab = self.mark("rung with the ring unmapped: %s" % self.cmd_text(cmds))
        self.unmap_ring()
        self.write_cmds(g, cmds)
        self.bell()
        self.program_ring(g)
        self.events(lab + ": nothing ran, the ring mapped again")
        s.iosfc_r(I_READ, lab)
        self.predict(val=hex(self.R & M32))
        ran = self.bell()
        self.wait_ran(ran, lab + ": rung again")
        self.settle(lab)

    def a_regs(self):
        self.count("registers")
        lab = self.mark("registers")
        self.regs(lab)

    def a_teardown(self):
        """0x1010 rewritten with the same table: every surface dropped,
        whatever holds it; the mapped bits stay in the descriptors."""
        r = self.r
        self.count("teardown")
        lab = self.mark("the descriptor base rewritten: every surface dropped")
        self.set_table(self.dtab)
        self.descriptors(lab)
        if r.chance(0.6):
            again = r.sample(self.f.mids, 1 + r.below(len(self.f.mids)))
            self.ring_cmds([(CMD_MAP, m) for m in again], lab + ": mapped again: %s" % again)
        self.settle(lab)

    def a_format(self):
        """0x1030: the other descriptor format, the table cleared; a
        doorbell before the new table runs nothing; the new table's
        descriptors; then the surfaces mapped again."""
        r, s = self.r, self.s
        self.count("format switched")
        v2 = not self.dtab.v2
        lab = self.mark("descriptors switched to v%d with a new table" % (2 if v2 else 1))
        self.enable = 1 if v2 else 0
        s.iosfc_w(I_DESC_ENABLE, self.enable)
        self.drop_all()
        self.dtab = None
        if r.chance(0.5):
            self.write_cmds(self.ring, [(CMD_MAP, r.choice(self.f.mids))])
            self.bell()
            self.events(lab + ": rung with no table, nothing ran")
        t = self.new_table(v2)
        self.write_descs(t)
        self.set_table(t)
        again = r.sample(self.f.mids, 1 + r.below(len(self.f.mids)))
        self.ring_cmds([(CMD_MAP, m) for m in again], lab + ": mapped: %s" % again)
        self.descriptors(lab)
        self.settle(lab)

    # -- the consumers --

    def free_fifos(self, children_only):
        held = self.held_fifos()
        c = [f for f in WORKERS if f not in held]
        if not children_only and 0 not in held:
            c = [0] + c
        return c

    def gated_now(self):
        """Whether the next item joins the chain: always while paused or
        when something is queued; otherwise by the program's rate."""
        return self.paused or bool(self.queue) or self.r.chance(self.f.gate_p)

    def a_fill(self):
        r = self.r
        self.count("fill")
        mid = r.choice(self.f.mids)
        chain = self.gated_now()
        fids = list(WORKERS) if chain else self.free_fifos(True)
        if not fids:
            chain, fids = True, list(WORKERS)
        it = self.new_item("fill", r.choice(fids))
        it.mid = mid
        it.ops = self.fill_ops()
        self.issue(it, chain)

    def a_present(self):
        r = self.r
        if any(q.kind == "present" for q in self.queue):
            return self.a_fill()
        self.count("present")
        chain = self.gated_now()
        fids = [0] + list(WORKERS) if chain else self.free_fifos(False)
        if not fids:
            chain, fids = True, [0] + list(WORKERS)
        it = self.new_item("present", r.choice(fids))
        it.port = r.below(self.f.ports)
        if r.chance(0.75):
            it.mid = UNLINKED_ID if r.chance(0.08) else r.choice(self.f.mids)
        else:
            it.frame = r.below(2)
        self.issue(it, chain)

    def issue(self, it, chain):
        gate = chain and not self.paused and not self.queue
        lab = self.mark("%s%s" % (it.desc() if not gate else it.desc() + " (gated)",
                                  " (chained)" if chain and not gate else ""))
        self.write_item(it, gate)
        it.fifo.doorbell()
        if chain:
            it.state = "queued"
            first = not self.queue
            self.queue.append(it)
            if first:
                it.fifo.wait_stamp(it.stamp, timeout_ms=HOLD_MS, label=lab + ": not run" + HELD)
            self.settle(lab)
            return
        self.run_item(it, lab)
        self.settle(lab)

    def a_open(self):
        self.count("gate opened")
        lab = self.mark("the gate opens: %s" % "; ".join("i%d" % it.id for it in self.queue))
        self.x.open_gate(label=lab + ": gate NOP")
        self.resolve_queue(lab)
        self.settle(lab)

    def a_rewrite(self):
        """The guest writes a surface's page itself."""
        r, s = self.r, self.s
        self.count("rewrite")
        mid = r.choice(self.f.mids)
        off = 16 * r.below(FRAME // 16)
        n = 16 * (1 + r.below((FRAME - off) // 16))
        data = bytes(r.below(256) for _ in range(n))
        lab = self.mark("the guest rewrites surface %d's bytes %x+%x" % (mid, off, n))
        s.mem(s.pa(self.pages[mid][0], off), data)
        for m in self.models:
            m.raw[mid][off:off + n] = data
        self.touched.add(mid)
        self.settle(lab)

    def a_delete_buf(self):
        """37 on a mapper buffer: a made buffer's hold goes (and under
        isolation the service's release erases the host's entry); the next
        exec makes it again from the list's entry. On a buffer not made,
        37 releases nothing."""
        r, s = self.r, self.s
        if 0 in self.held_fifos():
            return self.a_cmds()
        cand = [m for m in self.f.mids if not any(q.kind == "fill" and q.mid == m for q in self.queue)]
        if not cand:
            return self.a_cmds()
        self.count("37")
        mid = r.choice(cand)
        t = self.tasks[mid]
        lab = self.mark("37 on task %d's mapper buffer over %d" % (t.tid, mid))
        op, pl = delete_resource(t.tid, t.mbuf)
        s.submit(op, pl, label=lab)
        for m in self.models:
            if m.made[mid] is not None:
                m.made[mid] = None
                m.host[mid] = ABSENT
        if r.chance(0.6):
            self.ring_cmds([(CMD_UNMAP, mid)], lab + ": unmap")
        self.settle(lab)

    # -- the lifecycle --

    def a_pause(self):
        r, s = self.r, self.s
        self.count("pause")
        lab = self.mark("pause")
        s.api("pause", label=lab)
        self.paused = True
        self.events(lab)
        for _ in range(1 + r.below(4)):
            k = r.weighted([("cmds", 4), ("fill", 3), ("present", 1), ("rewrite", 1), ("relocate", 1)])
            getattr(self, "a_" + k)()
        lab = self.mark("unpause")
        s.api("unpause", label=lab)
        self.paused = False
        if self.queue and not self.queue[0].gated:
            self.resolve_queue(lab + ": rung while paused")
        self.settle(lab)

    def a_suspend(self):
        """The suspend onto a fresh device, with the chain held on the gate:
        the heads are new, the tasks' buffers go, and the resume
        re-presents each page's mapping whose id has a surface."""
        x = self.x
        self.count("suspend")
        lab = self.mark("suspend onto a fresh device%s" % (
            " over %s" % "; ".join("i%d" % it.id for it in self.queue) if self.queue else ""))
        x.suspend(label=lab)
        self.events(lab + ": suspended")
        x.resume(label=lab)
        n = 0
        for m in self.models:
            n = 0
            for mid in self.f.mids:
                m.made[mid] = None
                m.host[mid] = ABSENT
            for p in sorted(m.heads):
                old = m.heads[p]
                h = m.heads[p] = Head(p)
                h.dirty = False
                h.mapping = old.mapping
                if old.mapping is not None and m.gen.get(old.mapping) is not None:
                    h.frame = ("surf", old.mapping, m.gen[old.mapping])
                    m.frame_cbs(h, True)
                    self.reframed.add(p)
                    n += 1
        if n:
            x.expect("newFrame", n=n, label=lab + ": the mappings presented again")
        self.settle(lab + ": resumed")

    def a_reset(self):
        """reset: the heads' frames and the tasks' buffers go, and work held
        on the gate is abandoned; the guest does its set-up again."""
        s, x = self.s, self.x
        self.count("reset")
        lab = self.mark("reset%s" % (" abandoning %s" % "; ".join("i%d" % it.id for it in self.queue)
                                     if self.queue else ""))
        s.api("reset", label=lab)
        for it in self.queue:
            it.state = "abandoned"
        self.queue = []
        for m in self.models:
            for h in m.heads.values():
                h.frame = None
                h.mapping = None
                h.dirty = True
            for mid in self.f.mids:
                m.made[mid] = None
                m.host[mid] = ABSENT
        self.events(lab)
        s.w(abi.R_VERSION, x.version)
        s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
        s.fifo.written = 0
        self.prev[0] = None
        s.fifo.enable()
        s.submit(abi.OP_NOP, label=lab + ": first NOP again")
        x.t0.define(label=lab + ": define task 0 again")
        x.t0.install(label=lab + ": task 0's object list again")
        for mid in self.f.mids:
            t = self.tasks[mid]
            t.define(label=lab + ": define task %d again" % t.tid)
            t.install(label=lab + ": task %d's object list again" % t.tid)
        for fid in WORKERS + (GATE,):
            self.define_child(fid)
        for p in range(self.f.ports):
            x.connect(p)
        self.settle(lab + ": set up again")

    # -- the end --

    def finish(self):
        if self.queue and self.queue[0].gated:
            self.a_open()
        lab = self.mark("the end")
        self.check_all(lab)
        self.events(lab)

    def die(self, death):
        r, s = self.r, self.s
        if not self.queue:
            it = self.new_item("fill", r.choice(WORKERS))
            it.mid = r.choice(self.f.mids)
            it.ops = self.fill_ops()
            self.issue(it, True)
        self.dies = True
        if death == "ring_unaligned":
            self.mark("the ring's size written as 1000 entries with a base set: the host dies")
            s.iosfc_w(I_RING_SIZE, 1000)
            s.quiesce(200)
        elif death == "wide_present":
            lab = self.mark("surface %d, 16400 x 1, mapped and presented: the host dies" % WIDE_ID)
            root, data = self.surface_pages(5)
            w0 = self.dtab.put(WIDE_ID, root, w=16400, h=1)
            for m in self.models:
                m.word0[(self.dtab.key, WIDE_ID)] = w0
                m.gen[WIDE_ID] = None
            self.ring_cmds([(CMD_MAP, WIDE_ID)], lab)
            s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, WIDE_ID), fault="either",
                     label=lab + ": the present" + DIES)
        else:
            lab = self.mark("pause twice: the host dies")
            s.api("pause", label=lab)
            s.api("pause", label=lab + DIES)
            s.quiesce(200)

    def summary(self):
        return ", ".join("%s %d" % kv for kv in sorted(self.hist.items()))


# ---- entry points ----

def walk(seed, steps, doc="", death=None):
    w = Walk(seed, steps, doc, death)
    w.build()
    return w


def build(seed, steps, doc, death=None):
    return walk(seed, steps, doc, death).s


def check_obs(w, obs, setup):
    """Compares a run's observations with the walk's predictions for that
    setup; returns complaints."""
    bad = []
    by_i = {}
    for o in obs:
        if isinstance(o.get("i"), int):
            by_i.setdefault(o["i"], []).append(o)
    for i, p in sorted(w.pred.items()):
        if "by_setup" in p:
            p = p["by_setup"][setup]
        got = by_i.get(i)
        if not got:
            if w.dies:
                continue
            bad.append("op %d: no observation (want %s)" % (i, json.dumps(p)[:200]))
            continue
        o = got[-1]
        for k, v in p.items():
            if o.get(k) != v:
                bad.append("op %d %s: %s want %s got %s" % (i, o.get("label", ""), k, json.dumps(v)[:400],
                                                            json.dumps(o.get(k))[:400]))
                break
    out = obs[-1] if obs else {}
    want = "died" if w.dies else "completed"
    if out.get("kind") != want:
        bad.append("outcome %s, the model says %s" % (out.get("kind"), want))
    for o in obs:
        lab = o.get("label", "")
        if o.get("ev") == "wait" and o.get("ok") is False and not (lab.endswith(HELD) or lab.endswith(DIES)):
            bad.append("wait failed: %s" % lab)
        if o.get("ev") == "wait" and o.get("ok") is True and lab.endswith(HELD):
            bad.append("held work ran: %s" % lab)
        if o.get("ev") == "cmp" and o.get("equal") is False:
            bad.append("cmp differs: %s" % lab)
    return bad


def check(paths=None):
    """Every expected file of interplay_iosfcwalk against the model."""
    import glob
    import os
    from scenarios.interplay_iosfcwalk import SEEDS, __doc__ as doc
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = paths or os.path.join(here, "expected")
    bad = 0
    for seed, steps, death in SEEDS:
        w = walk(seed, steps, doc, death)
        for p in sorted(glob.glob(os.path.join(root, "interplay_iosfcwalk", w.name, "*.jsonl"))):
            setup = os.path.basename(p).split("@")[0].split(".")[0]
            lines = [json.loads(l) for l in open(p) if l.strip() and not l.startswith("#")]
            obs = [o for o in lines if "ev" in o or "kind" in o]
            complaints = check_obs(w, obs, setup)
            for c in complaints:
                print("%s: %s" % (os.path.relpath(p, here), c))
            bad += len(complaints)
    return bad


if __name__ == "__main__":
    import sys
    sys.exit(1 if check(sys.argv[1] if len(sys.argv) > 1 else None) else 0)
