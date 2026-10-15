"""Seeded walks that inject device lifecycle events into live mixed work
(scenarios/interplay_lifewalk.py).

The earlier walkers put the lifecycle between rounds of work (randgen's
idle pause, suspend, reset, stop and fresh device), hold one phase's work
across a pause or a suspend (randmix), or walk display state around the
lifecycle (randdisplay). Here every round builds a live state first and
then lands one lifecycle event in it, chosen at random:

  - live work: GPU execs on task 1 (a pgtest/session.py SessionTask) on the
    root FIFO and worker child FIFOs, each writing a slot of the task's out
    buffer, some chained to an earlier item's slot (an order witness),
    some held behind the gate (child 7) or behind another held item, and
    before a pause one read by its FIFO and not yet waited for; fills of
    IOSurface mapper surfaces through a mapper buffer (type 15,
    pgtest/scanout_util.py); a present of a task-0 backing (7) or of a mapper
    surface (8), itself held or not; a faulting command (an unknown opcode) on
    a worker child; the cursor; and in some programs the EFI framebuffer at
    boot, presenting through trace fires;
  - the event: pause and unpause, with work rung while paused and, with a
    display, a new mode list while paused; the suspend sequence
    (rvg_device_will_suspend, work rung, rvg_device_finish_suspend, a fresh
    device, rvg_device_will_resume, rvg_device_did_resume, where that work runs
    without a new doorbell); rvg_device_will_suspend then unpause (no guest
    display only: with one it asserts, docs/testing.md 7.3); reset, stop and a
    fresh device, after which the guest does its set-up again and rings the
    abandoned work again; the root FIFO disabled and enabled (0x1000 = 0, 1),
    which deletes every child FIFO and abandons what they held; a binary
    version change, or a dip below 3 and back (which tears EFI down); the host
    unplugging and plugging a display, or giving it a new mode list; a worker
    child FIFO deleted with its held work and defined again, or deleted for
    good, or a new one defined; or nothing;
  - the resolution: the gate opens, faults are recovered the way the event
    leaves them, abandoned work is rung again, and every present's new_frame
    and the resume's replays are waited for.

The model (what the walk writes into labels, waits and mirror pages):
  - slots: each item's 0x100 bytes of task 1's out buffer: [0, 0x40) the
    chain predecessor's fill byte (copied from its slot) or the item's byte
    ^ 0xff, [0x40, 0x80) the item's byte, [0x80, 0xc0) 0x40 bytes of the
    source from an offset of the item's. After each round the model's
    bytes go into a mirror page and a cmp compares every slot the round
    touched; at the end every slot. Held, rung-while-paused, rung-after-
    rvg_device_will_suspend and abandoned-then-rung-again items all end as the
    model says, so every lifecycle event is transparent to the GPU work;
  - surfaces: the last fill's byte over the whole 16 x 8 frame, compared
    with a mirror the same way, and encoded after each present of it;
  - stamps: every item's stamp is waited for, and the held ones with a
    300 ms wait labelled [held] that times out;
  - faults (randfault's rules): a fault is waited for by its node's fault
    offset and 0x102c; a pause publishes the faulted stamp and leaves
    0x102c set and the child latched (never faulted again in the walk
    until a teardown or a delete clears it); a suspend drains it, and the
    fresh device restores it as a register; reset, stop, a fresh device,
    0x1000 = 0 and a delete of the child clear it; otherwise the guest's
    0x1028 publishes it. 0x102c is waited for at 0 after every recovery;
  - displays: one new_frame for each present that ran, and after a resume
    the cursor glyph and show replayed on each port that set one and a
    new_frame on each port whose page had an opcode-8 mapping
    (GuestDisplay::finish_restore); acks follow scanout_util's count;
  - mode lists: the host display's and the page's, apart. A new list is
    one that differs from both. A resume onto the target's display runs
    each restored guest display's state machine, so a page whose list
    differs gets an update to ack, unless the port presented without an
    opcode-8 mapping: finish_restore then re-presents mapping 0, which
    raises and skips the state machine
    (interplay_lifewalk_resume_page_modes).

Invariants the generator keeps so that every checkpoint is deterministic:
  - Each command on a FIFO carries a barrier on the command before it on
    that FIFO, so each publishes alone and interrupt counts are exact
    (dagwalk's rule); a chained item on another FIFO carries a barrier on
    its predecessor; a present of a surface on the fill before it.
  - At most one present a round (new_frame calls coalesce, 4.6), and none
    on a port whose display the round unplugs. Nothing touches a display
    from the guest while paused (7.3).
  - Held work is released only by the gate, within its 300 ms hold plus
    the event, well below the 2 s after which a blocked FIFO reports.
  - Deleting child n zeroes node n+1's pageListPage: the guest writes it
    back at once, so every start of a child's FIFO thread maps its ring.
  - rvg_device_will_suspend then unpause only with no display and nothing held
    (over a blocked FIFO it leaves two threads, 7.4).
  - An item in flight (read, not waited for) only before a pause, whose
    drain publishes it with one interrupt either way: before a suspend,
    rvg_device_will_suspend suppresses interrupts, and whether the exec's came
    first is timing (measured: the count differed between runs).
  - The events whose guest side uses the root FIFO (49 and 48, an ack)
    keep the round's work off it, and every ack follows the release of
    what the root FIFO held.
  - EFI runs only while port 0's page is not connected, and ends (0x1200 =
    0, or a version dip) before any reset, stop, fresh device, suspend or
    the guest connecting port 0.
  - Mapper surfaces need a memory map: programs with them run in the map
    and isolate setups, the others in all three.
  - A program that ends in a death (docs/testing.md 7.3) does so with work
    held: the death is its last action.

Nothing here is imported from the library's code.
"""
from . import abi
from .abi import PAGE
from .fifo import ChildFIFO
from .randgen import Rng
from .resources_build import b_copy_buf_buf, b_fill_buf
from .scanout_util import FRAME, MAPPED, Scanout
from .scenario import SETUPS
from .session import FH, FW
from .task import seg

GATE = 7
SLOT = 0x100
HOLD_MS = 300
HELD = " [held]"
BYTES = (0x11, 0x2B, 0x5A, 0x6C, 0x93, 0xA5, 0xC3, 0xD7, 0x3C, 0xE1)
UNKNOWN_OP = 13                 # the FIFO's default case: "unknown opcode"
MODES_A = ((64, 32, 60), (FW, FH, 60))
MODES_B = ((FW, FH, 60), (32, 16, 30))
MODES_C = ((32, 16, 60), (FW, FH, 60), (64, 32, 30))
MODE_LISTS = (MODES_A, MODES_B, MODES_C)
VERSIONS = (245, 238, 237, 235, 233, 231, 230, 209, 208, 207, 204, 200, 113, 110, 109)
DIPS = (2, 1, 0)                 # below efiDisplay (3): EFI torn down (set_binary_version)
DEATHS_NUB = ("pause_twice", "doorbell_paused", "will_unpause_nub")
DEATHS_ANY = ("finish_without_will", "failed_resume")
DEATHS = DEATHS_NUB + DEATHS_ANY
NODE_FAULT = 8
HEALTHY = 0xFFFFFFFF


class Features:
    def __init__(self, seed, rounds, death):
        r = Rng(seed ^ 0x11FE11FE11FE)
        self.rounds = rounds
        self.ports = r.choice((0, 1, 1, 2))
        if death in DEATHS_NUB:
            self.ports = max(self.ports, 1)
        self.surfaces = (1 + r.below(2)) if self.ports and r.chance(0.6) else 0
        self.efi = bool(self.ports) and r.chance(0.35)
        self.workers = r.choice((2, 3, 4))
        self.held_p = r.choice((0.3, 0.5, 0.7))
        self.fault_p = r.choice((0.0, 0.2, 0.35))
        self.death = death or None

    def setups(self):
        return MAPPED if self.surfaces else SETUPS

    def tags(self):
        t = ["ports=%d" % self.ports, "surfaces=%d" % self.surfaces, "workers=%d" % self.workers]
        if self.efi:
            t.append("EFI at boot")
        if self.fault_p:
            t.append("fault rate %g" % self.fault_p)
        if self.death:
            t.append("ends in death: " + self.death)
        return t


class Item:
    """One command of a round. kind: exec, surf, present, fault."""

    def __init__(self, iid, kind, fid):
        self.id, self.kind, self.fid = iid, kind, fid
        self.gate = False       # a barrier on the gate
        self.pred = None        # exec: the chain predecessor item
        self.after = []         # items this one has barriers on (besides the FIFO's previous)
        self.byte = None
        self.slot = None
        self.src_off = 0
        self.mid = None         # surf, present of a mapping
        self.port = None
        self.frame = None       # present of a backing
        self.stamp = None
        self.fifo_obj = None
        self.phase = "pre"      # pre, during
        self.inflight = False
        self.done = False

    def desc(self):
        k = self.kind
        if k == "exec":
            w = "exec s%d %02x%s" % (self.slot, self.byte, " <i%d" % self.pred.id if self.pred else "")
        elif k == "surf":
            w = "fill surface %d %02x" % (self.mid, self.byte)
        elif k == "present":
            w = ("swap %d" % self.mid if self.mid is not None else "backing %d" % self.frame) + " on port %d" % self.port
        else:
            w = "FAULT"
        return "i%d F%d %s%s%s" % (self.id, self.fid, w, " gate" if self.gate else "",
                                   " inflight" if self.inflight else "")


class Walk:
    def __init__(self, seed, rounds, doc, death=False):
        self.seed = seed
        self.f = f = Features(seed, rounds, death)
        self.r = Rng(seed)
        name = "interplay_lifewalk_%d" % seed
        self.x = x = Scanout(name, doc, ports=max(f.ports, 1), displays=tuple(range(f.ports)),
                             setups=f.setups(), host=bool(f.ports), efi=f.efi, iosfc=bool(f.surfaces),
                             modes=MODES_A, watchdog_s=90)
        self.s = x.s
        self.items = []
        self.nslots = 0
        self.slot_bytes = {}
        self.chain_tails = []
        self.workers = {}           # fid -> ChildFIFO, the live ones
        self.fifos = {}             # fid -> ChildFIFO, for good
        self.latched = set()
        self.version = abi.MAX_VERSION
        self.disp = {p: p for p in range(f.ports)}
        # Each port's mode lists: the host display's, and the one the
        # shared page holds (a resume keeps the page's, and a new mode list
        # equal to it raises no update).
        self.modes = {p: MODES_A for p in range(f.ports)}
        self.page_modes = {p: MODES_A for p in range(f.ports)}
        self.glyph = set()
        self.mapping = {}           # port -> mid of the last opcode 8
        self.connected = set()
        self.surf_bytes = {}
        self.last_surf = {}         # mid -> the last surf item written for it
        self.prev = {}              # fid -> the stamp of the last command written on it
        self.efi_on = False
        self.round = 0

    # ---- helpers ----

    def fifo(self, fid):
        return self.s.fifo if fid == 0 else self.fifos[fid]

    def events(self, label):
        self.s.events(label, sort_blocks=True, blocks_digest=True)

    def child(self, fid):
        c = self.fifos.get(fid)
        if c is None:
            c = self.fifos[fid] = ChildFIFO(self.s, fid)
            self.s.fifos[fid] = c
        c.written = 0
        self.prev[fid] = None
        c.setup_node()
        c.define(label="define child %d" % fid)
        self.x.children[fid] = c
        return c

    def restore_next(self, fid):
        """The delete of child fid zeroed node fid+1's pageListPage; the
        guest writes it back."""
        n = self.fifos.get(fid + 1)
        if n is not None and (fid + 1 in self.workers or fid + 1 == GATE):
            self.s.u32(n.node_gpa(0x10), self.s.pfn(n.list_page))

    def define_gate(self):
        self.x.gate = self.child(GATE)

    def src_bytes(self, off, n):
        seed = self.task.seed
        return bytes(((i * 7 + seed) & 0xFF) if i < 0x100 else 0 for i in range(off, off + n))

    def model_slot(self, it):
        head = bytes([self.slot_bytes[it.pred.slot][0x40]]) * 0x40 if it.pred else bytes([it.byte ^ 0xFF]) * 0x40
        return head + bytes([it.byte]) * 0x40 + self.src_bytes(it.src_off, 0x40) + bytes(0x40)

    # ---- set-up ----

    def setup(self, first):
        x, s, f = self.x, self.s, self.f
        if first:
            x.negotiate(asks=())
            x.root()
        else:
            s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
            s.fifo.written = 0
            self.prev[0] = None
            s.fifo.enable()
            s.submit(abi.OP_NOP, label="first NOP again")
        self.workers = {}
        x.children = {}
        if first:
            self.worker_ids = set(range(1, f.workers + 1))
        for fid in sorted(self.worker_ids):
            self.workers[fid] = self.child(fid)
        self.define_gate()
        if first:
            if f.ports:
                x.kernel_task()
            if f.surfaces:
                x.iosfc_start()
                for mid in range(1, f.surfaces + 1):
                    x.surface(mid, 0x10 * mid + 3)
                    self.surf_bytes[mid] = None
            self.task = x.task(1, self.seed & 0xFF, extra=x.mapper_objects((), buffers=range(1, f.surfaces + 1)))
            self.mirror = s.alloc_page(4)
            self.smirror = {mid: s.alloc_page() for mid in range(1, f.surfaces + 1)}
        else:
            if x.t0 is not None:
                x.t0.define(label="define task 0 again")
                x.t0.install(label="task 0's object list again")
            self.task.define(label="define task 1 again")
            self.task.install(label="task 1's object list again")
        self.connected = set()
        self.glyph = set()
        self.mapping = {}
        self.presented = set()      # ports whose page counted a present
        for p in range(f.ports):
            if p == 0 and self.efi_on:
                continue
            x.connect(p)
            self.connected.add(p)
        self.events("set up" if first else "set up again")

    # ---- items ----

    def new_item(self, kind, fid):
        it = Item(len(self.items), kind, fid)
        self.items.append(it)
        return it

    def gen_items(self, event, allow_held, allow_fault, exclude=()):
        """The round's items, in creation order."""
        r, f = self.r, self.f
        out = []
        fids = [0] + sorted(self.workers)
        n = 1 + r.below(4)
        faulted = None
        if allow_fault and r.chance(f.fault_p):
            cand = [c for c in sorted(self.workers) if c not in self.latched]
            if cand:
                faulted = r.choice(cand)
        live = [fid for fid in fids if fid != faulted and fid not in exclude]
        if not live:
            faulted = None
            live = [fid for fid in fids if fid not in exclude]
        if not live:
            return out
        present_done = False
        for _ in range(n):
            table = [("exec", 6)]
            if f.surfaces:
                table.append(("surf", 3))
            if self.connected and not present_done:
                table.append(("present", 2))
            kind = r.weighted(table)
            it = self.new_item(kind, r.choice(live))
            it.gate = allow_held and r.chance(f.held_p)
            if kind == "exec":
                it.byte = r.choice(BYTES)
                it.slot = self.nslots
                self.nslots += 1
                assert self.nslots <= 4 * PAGE // SLOT
                it.src_off = 8 * r.below(0x18)
                if self.chain_tails and r.chance(0.6):
                    c = r.below(len(self.chain_tails))
                    it.pred = self.chain_tails[c]
                    self.chain_tails[c] = it
                else:
                    self.chain_tails.append(it)
            elif kind == "surf":
                it.mid = 1 + r.below(f.surfaces)
                it.byte = r.choice(BYTES)
                if self.last_surf.get(it.mid) is not None:
                    it.after.append(self.last_surf[it.mid])
                self.last_surf[it.mid] = it
            else:
                present_done = True
                it.port = r.choice(sorted(self.connected))
                if f.surfaces and r.chance(0.6):
                    it.mid = 1 + r.below(f.surfaces)
                    if self.last_surf.get(it.mid) is not None:
                        it.after.append(self.last_surf[it.mid])
                else:
                    it.frame = r.below(2)
            out.append(it)
        if faulted is not None:
            it = self.new_item("fault", faulted)
            out.append(it)
        return out

    def held(self, it, items=None):
        """Whether it cannot run until the gate opens: it waits on the gate,
        on an item that cannot, or sits behind one on its FIFO."""
        if it.done:
            return False
        if it.gate:
            return True
        for m in it.after + ([it.pred] if it.pred is not None else []):
            if not m.done and self.held(m):
                return True
        for m in self.items[:it.id]:
            if m.fid == it.fid and not m.done and m.kind != "fault" and m.fifo_obj is it.fifo_obj \
                    and m.stamp is not None and self.held(m):
                return True
        return False

    def static_held(self, it, batch):
        """held(), for a batch not yet written: every earlier round's work
        has run."""
        if it.gate:
            return True
        for m in it.after + ([it.pred] if it.pred is not None else []):
            if not m.done and m in batch and self.static_held(m, batch):
                return True
        return any(m.fid == it.fid and self.static_held(m, batch) for m in batch[:batch.index(it)])

    def payload(self, it):
        t = self.task
        if it.kind == "exec":
            o = it.slot * SLOT
            if it.pred is not None:
                head = b_copy_buf_buf(t.out, t.out, it.pred.slot * SLOT + 0x40, o, 0x40)
            else:
                head = b_fill_buf(t.out, o, 0x40, it.byte ^ 0xFF)
            recs = head + b_fill_buf(t.out, o + 0x40, 0x40, it.byte) + b_copy_buf_buf(t.src, t.out, it.src_off,
                                                                                       o + 0x80, 0x40)
            return t.payload([seg(abi.ENC_BLIT, recs)])
        return t.payload([seg(abi.ENC_BLIT, b_fill_buf(t.mbuf[it.mid], 0, FRAME, it.byte))])

    def write(self, it, gate=True):
        """Writes it on its FIFO (not rung): barriers on the FIFO's
        previous command, on what it waits for, and on the gate."""
        x = self.x
        f = self.fifo(it.fid)
        bars = []
        if self.prev.get(it.fid) is not None:
            bars.append((f.stamp_index, self.prev[it.fid]))
        for m in it.after + ([it.pred] if it.pred is not None else []):
            if not m.done:
                bars.append((self.fifo(m.fid).stamp_index, m.stamp))
        if it.gate and gate:
            bars.append(x.gate_barrier())
        if it.kind in ("exec", "surf"):
            it.stamp = f.cmd(abi.OP_EXEC_INDIRECT3, self.payload(it), barriers=bars)
        elif it.kind == "present":
            if it.mid is not None:
                it.stamp = x.swap_cmd(it.fid, it.port, it.mid, barriers=bars)
            else:
                it.stamp = x.present_cmd(it.fid, it.port, it.frame, barriers=bars)
        else:
            it.stamp = f.cmd(UNKNOWN_OP, bytes(4), barriers=bars)
        it.fifo_obj = f
        self.prev[it.fid] = it.stamp

    def ring_batch(self, batch, gate=True):
        """Writes a batch and rings its FIFOs."""
        for it in batch:
            self.write(it, gate=gate)
        for fid in dict.fromkeys(it.fid for it in batch):
            self.fifo(fid).doorbell()

    def complete(self, it, label):
        """Waits for an item that runs, and applies it to the model."""
        s, x = self.s, self.x
        it.fifo_obj.wait_stamp(it.stamp, label="%s: i%d" % (label, it.id))
        self.ran(it)

    def ran(self, it):
        it.done = True
        if it.kind == "exec":
            self.slot_bytes[it.slot] = self.model_slot(it)
        elif it.kind == "surf":
            self.surf_bytes[it.mid] = bytes([it.byte]) * FRAME
        elif it.kind == "present":
            x = self.x
            self.presented.add(it.port)
            if it.mid is not None:
                self.mapping[it.port] = it.mid
                x.expect("newFrame", label="mapping %d on port %d" % (it.mid, it.port))
            else:
                x.expect("newFrame", label="backing %d on port %d" % (it.frame, it.port))
            self.encodes.append(it)

    def wait_held(self, items, label):
        held = [it for it in items if it.kind != "fault" and not it.inflight and self.held(it)]
        if held:
            it = held[0]
            it.fifo_obj.wait_stamp(it.stamp, timeout_ms=HOLD_MS, label="%s: i%d%s" % (label, it.id, HELD))

    # ---- faults ----

    def wait_fault(self, it):
        s = self.s
        c = self.fifos[it.fid]
        s.wait(gpa=c.node_gpa(NODE_FAULT), ne=HEALTHY, label="child %d faulted" % it.fid)
        s.wait(mmio=abi.R_FAULT_MASK, mmio_ne=0, label="child %d faulted: 0x102c" % it.fid)

    def recover(self, it, how):
        """The guest's recovery of the round's fault, after the event."""
        s = self.s
        c = self.fifos[it.fid]
        if how == "gone":
            it.done = True
            s.r(abi.R_FAULT_MASK, "fault cleared by the event")
            return
        s.r(abi.R_FAULT_MASK, "child %d: fault %s" % (it.fid, how))
        s.dump(c.node_gpa(NODE_FAULT), 4, "child %d: fault offset (%s)" % (it.fid, how))
        s.w(abi.R_RESUME_CHILD, it.fid)
        s.wait(mmio=abi.R_FAULT_MASK, mmio_eq=0, label="child %d resumed: 0x102c" % it.fid)
        if how == "latched":
            self.latched.add(it.fid)
        c.wait_stamp(it.stamp, label="child %d: the faulted stamp published" % it.fid)
        it.done = True

    # ---- the round ----

    def choose_event(self):
        r, f = self.r, self.f
        if self.efi_on:
            table = [("none", 2), ("pause", 3), ("toggle", 2), ("child", 2), ("version", 2), ("efi_end", 2)]
        else:
            table = [("none", 1), ("pause", 5), ("suspend", 5), ("reset", 2), ("stop", 2), ("recreate", 2),
                     ("toggle", 3), ("version", 3), ("child", 3)]
            if not f.ports:
                table.append(("will_unpause", 3))
            if f.ports:
                table.append(("display", 4))
        return r.weighted(table)

    def do_round(self, death=None):
        x, s, r, f = self.x, self.s, self.r, self.f
        event = death or self.choose_event()
        self.encodes = []
        allow_held = event != "will_unpause"
        allow_fault = event not in ("will_unpause", "efi_end") and death is None
        # The cursor, now and then, before the work (never while paused).
        if self.connected and not death and r.chance(0.25):
            port = r.choice(sorted(self.connected))
            glyph = port not in self.glyph or r.chance(0.3)
            s.mark("round %d: cursor on port %d" % (self.round, port))
            x.cursor(port, 1 + r.below(60), 1 + r.below(30), glyph=glyph)
            self.glyph.add(port)
        # Events whose guest side uses the root FIFO (49 and 48, an ack)
        # keep work off it, so nothing held there blocks them.
        pre = self.gen_items(event, allow_held, allow_fault, exclude=(0,) if event in ("child", "display") else ())
        fault_fid = next((it.fid for it in pre if it.kind == "fault"), None)
        during = []
        if event in ("pause", "suspend", "will_unpause") and r.chance(0.6):
            ex = ((fault_fid,) if fault_fid is not None else ()) + ((0,) if event in ("child", "display") else ())
            for it in self.gen_items(event, False, False, exclude=ex):
                if it.kind == "fault":
                    continue
                if it.kind == "present":
                    # No display command while paused or suspending.
                    it.kind = "exec"
                    it.port = it.mid = it.frame = None
                    it.byte = r.choice(BYTES)
                    it.slot = self.nslots
                    self.nslots += 1
                    it.src_off = 0
                    it.after = []
                    self.chain_tails.append(it)
                it.phase = "during"
                it.gate = False
                during.append(it)
        fault = next((it for it in pre if it.kind == "fault"), None)
        body = [it for it in pre if it.kind != "fault"]
        # In flight: the last ungated item of a round with a pause, when
        # nothing waits on it; the pause's drain publishes it, with its one
        # interrupt either way (interplay_life_pause_in_flight). Not before a
        # suspend: rvg_device_will_suspend suppresses interrupts, so whether
        # the exec's comes before it is timing.
        if event in ("pause", "suspend") and r.chance(0.4) and event == "pause" and body and body[-1].kind != "present" \
                and not self.static_held(body[-1], body):
            body[-1].inflight = True
        s.mark("round %d: %s; %s%s" % (self.round, event, "; ".join(it.desc() for it in pre),
                                        (" | during: " + "; ".join(it.desc() for it in during)) if during else ""))
        self.ring_batch(body)
        if fault is not None:
            self.ring_batch([fault])
        for it in body:
            if it.inflight:
                f_ = it.fifo_obj
                if it.fid == 0:
                    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f_.written & 0xFFFFFFFF, label="i%d read" % it.id)
                else:
                    s.wait(gpa=f_.node_gpa(4), eq=f_.written, label="i%d read" % it.id)
            elif not self.held(it):
                self.complete(it, "before the event")
        if fault is not None:
            self.wait_fault(fault)
        self.wait_held(body, "held at the event")
        getattr(self, "ev_" + event)(body, during, fault)
        if death:
            return
        self.check(body + during)
        self.round += 1

    # ---- resolutions shared by events ----

    def release(self, items, label="released"):
        """The gate opens, if anything waits on it, and every item that has
        not run is waited for in creation order."""
        x = self.x
        if any(it.gate and not it.done for it in items):
            x.open_gate(label="round %d: the gate opens" % self.round)
        for it in items:
            if not it.done and it.kind != "fault":
                self.complete(it, label)

    def ring_again(self, items, wait=True):
        """Work the event abandoned, rung again as fresh commands (the gate
        dropped: nothing holds it now)."""
        again = [it for it in items if not it.done and it.kind != "fault"]
        if not again:
            return
        for it in again:
            it.gate = False
            it.inflight = False
        self.s.mark("round %d: rung again: %s" % (self.round, "; ".join("i%d" % it.id for it in again)))
        self.ring_batch(again)
        if wait:
            for it in again:
                self.complete(it, "rung again")

    def during_ring(self, during, label):
        if during:
            self.ring_batch(during)
            it = during[0]
            it.fifo_obj.wait_stamp(it.stamp, timeout_ms=HOLD_MS, label="%s: i%d%s" % (label, it.id, HELD))

    def inflight_done(self, body):
        for it in body:
            if it.inflight:
                self.complete(it, "in flight at the event")

    def replays(self):
        """What the resume replays: glyph and show on every port that set a
        glyph, a new_frame on every port whose page had an opcode-8 mapping."""
        x = self.x
        glyphs = sorted(self.glyph & self.connected)
        frames = sorted(p for p in self.mapping if p in self.connected)
        if glyphs:
            x.expect("cursorGlyph", n=len(glyphs), label="glyphs replayed on ports %s" % glyphs)
            x.expect("cursorShow", n=len(glyphs), label="shows replayed on ports %s" % glyphs)
        if frames:
            x.expect("newFrame", n=len(frames), label="mappings presented again on ports %s" % frames)

    # ---- events ----

    def ev_none(self, body, during, fault):
        self.release(body)
        if fault:
            self.recover(fault, "kept")

    def ev_pause(self, body, during, fault):
        s, x, r = self.s, self.x, self.r
        s.api("pause", label="round %d" % self.round)
        self.inflight_done(body)
        if fault:
            # The pause publishes the faulted stamp (fifo::stamps).
            fault.fifo_obj.wait_stamp(fault.stamp, label="the pause published the faulted stamp")
        self.during_ring(during, "rung while paused")
        moded = None
        if self.connected and r.chance(0.35):
            moded = r.choice(sorted(self.connected))
            self.new_modes(moded)
            x.set_modes(self.disp[moded], self.modes[moded], label="port %d's new modes while paused" % moded)
        self.events("paused")
        s.api("unpause", label="round %d" % self.round)
        if moded is not None:
            s.wait(gpa=x.ss(moded, 0xFA9), mask=0xFF, eq=1, label="port %d's update pending" % moded)
        for it in during:
            if not self.held(it):
                self.complete(it, "after the unpause")
        if fault:
            self.recover(fault, "latched")
        self.release(body + during)
        if moded is not None:
            # The ack goes on the root FIFO, after what it held.
            x.ack_next(moded)

    def ev_suspend(self, body, during, fault):
        s, x = self.s, self.x
        s.api("willSuspend", label="round %d" % self.round)
        self.during_ring(during, "rung after willSuspend")
        s.api("finishSuspend", label="round %d" % self.round)
        self.inflight_done(body)
        if fault:
            fault.fifo_obj.wait_stamp(fault.stamp, label="finishSuspend published the faulted stamp")
        self.events("suspended")
        s.api("recreate", label="round %d: the migration target" % self.round)
        s.api("willResume", label="round %d" % self.round)
        s.api("didResume", label="round %d" % self.round)
        self.disp = {p: p for p in range(self.f.ports)}
        self.ndisp = self.f.ports
        self.modes = {p: MODES_A for p in range(self.f.ports)}
        self.replays()
        # The restored guest display runs its state machine against the
        # target's display: a page whose list differs gets the display's, and
        # an update (12) the guest acks (interplay_lifewalk_resume_page_modes).
        # But finish_restore re-presents the page's mapping first when the
        # port ever presented, and mapping 0 (opcode 7 only) raises, which
        # skips the state machine.
        updated = [p for p in sorted(self.connected) if self.page_modes[p] != self.modes[p]
                   and (p not in self.presented or p in self.mapping)]
        for p in updated:
            s.wait(gpa=x.ss(p, 0xFA9), mask=0xFF, eq=1, label="port %d: the resume's update pending" % p)
            self.page_modes[p] = self.modes[p]
        for it in during:
            if not self.held(it):
                self.complete(it, "after the resume, without a doorbell")
        self.events("resumed")
        if fault:
            self.recover(fault, "restored")
        self.release(body + during)
        for p in updated:
            x.ack_next(p)

    def ev_will_unpause(self, body, during, fault):
        s = self.s
        s.api("willSuspend", label="round %d" % self.round)
        self.during_ring(during, "rung after willSuspend")
        s.api("unpause", label="round %d: the suspend abandoned" % self.round)
        for it in during:
            self.complete(it, "after the unpause")
        self.release(body + during)

    def teardown(self, how, body, fault):
        s = self.s
        if how == "recreate":
            s.api("recreate", label="round %d: a fresh device" % self.round)
            self.disp = {p: p for p in range(self.f.ports)}
            self.ndisp = self.f.ports
            self.modes = {p: MODES_A for p in range(self.f.ports)}
            self.page_modes = dict(self.modes)
        else:
            s.api(how, label="round %d" % self.round)
            self.page_modes = dict(self.modes)
        self.events(how)
        s.sweep(0x1000, 0x1040, label="registers after " + how)
        if how != "stop":
            s.w(abi.R_VERSION, self.version)
        self.latched = set()
        if fault:
            self.recover(fault, "gone")
        self.setup(False)
        self.ring_again(body)

    def ev_reset(self, body, during, fault):
        self.teardown("reset", body, fault)

    def ev_stop(self, body, during, fault):
        self.teardown("stop", body, fault)

    def ev_recreate(self, body, during, fault):
        self.teardown("recreate", body, fault)

    def ev_toggle(self, body, during, fault):
        s = self.s
        s.w(abi.R_FIFO_ENABLE, 0)
        s.r(abi.R_FIFO_ENABLE, "round %d: disabled" % self.round)
        s.r(abi.R_FAULT_MASK, "round %d: disabled" % self.round)
        self.events("root FIFO disabled")
        s.fifo.written = 0
        self.prev[0] = None
        s.w(abi.R_FIFO_ENABLE, 1)
        s.submit(abi.OP_NOP, label="round %d: the new root FIFO" % self.round)
        self.latched = set()
        if fault:
            self.recover(fault, "gone")
        x = self.x
        x.children = {}
        for fid in sorted(self.workers):
            self.workers[fid] = self.child(fid)
        self.define_gate()
        self.ring_again(body)

    def ev_version(self, body, during, fault):
        s, r = self.s, self.r
        if not self.efi_on and r.chance(0.4):
            dip = r.choice(DIPS)
            s.w(abi.R_VERSION, dip)
            s.r(abi.R_VERSION, "round %d: dipped" % self.round)
            s.w(abi.R_VERSION, self.version)
        else:
            self.version = r.choice(VERSIONS)
            s.w(abi.R_VERSION, self.version)
        s.r(abi.R_VERSION, "round %d" % self.round)
        s.r(abi.R_PORT_COUNT, "round %d" % self.round)
        self.release(body)
        if fault:
            self.recover(fault, "kept")

    def ev_efi_end(self, body, during, fault):
        s, x, r = self.s, self.x, self.r
        if r.chance(0.5):
            s.w(abi.R_VERSION, r.choice(DIPS))
            s.w(abi.R_VERSION, self.version)
            x.efi.regs("EFI after a version dip")
        else:
            x.efi_off()
        self.efi_on = False
        self.release(body)
        x.connect(0)
        self.connected.add(0)
        self.presented.discard(0)
        self.mapping.pop(0, None)

    def ev_display(self, body, during, fault):
        s, x, r = self.s, self.x, self.r
        busy = {it.port for it in body if it.kind == "present"}
        free = [p for p in sorted(self.connected) if p not in busy]
        if free and r.chance(0.6):
            port = r.choice(free)
            s.mark("round %d: unplug and plug port %d" % (self.round, port))
            x.unplug(self.disp[port], port)
            self.release(body)
            x.hotplug(port, modes=self.modes[port])
            self.page_modes[port] = self.modes[port]
            self.disp[port] = self.next_index()
        else:
            port = r.choice(sorted(self.connected))
            self.new_modes(port)
            x.set_modes(self.disp[port], self.modes[port], label="port %d's new modes" % port)
            s.wait(gpa=x.ss(port, 0xFA9), mask=0xFF, eq=1, label="port %d's update pending" % port)
            x.ack_next(port)
            self.release(body)
        if fault:
            self.recover(fault, "kept")

    def new_modes(self, port):
        self.modes[port] = self.r.choice([m for m in MODE_LISTS if m not in (self.modes[port], self.page_modes[port])])
        self.page_modes[port] = self.modes[port]

    def next_index(self):
        self.ndisp += 1
        return self.ndisp - 1

    def ev_child(self, body, during, fault):
        """A worker child deleted (49) and defined again (48) with its held
        work rung again; or deleted for good when nothing of the round is
        on it; or a new worker defined."""
        s, r = self.s, self.r
        idle = [fid for fid in sorted(self.workers) if not any(it.fid == fid for it in body)]
        roll = r.below(4)
        if roll == 0 and idle and len(self.workers) > 1:
            fid = r.choice(idle)
            self.workers[fid].delete(label="round %d: delete child %d for good" % (self.round, fid))
            self.restore_next(fid)
            self.latched.discard(fid)
            del self.workers[fid]
            del self.x.children[fid]
            self.worker_ids.discard(fid)
            s.dump(self.fifos[fid].node_gpa(0), abi.CHILD_NODE_SIZE, "child %d's node, deleted" % fid)
            self.release(body)
            if fault is not None:
                self.recover(fault, "kept")
            return
        free = [fid for fid in range(1, GATE) if fid not in self.workers]
        if roll == 1 and free:
            fid = r.choice(free)
            self.workers[fid] = self.child(fid)
            self.worker_ids.add(fid)
            self.release(body)
            if fault is not None:
                self.recover(fault, "kept")
            return
        fid = fault.fid if fault is not None and r.chance(0.5) else r.choice(sorted(self.workers))
        c = self.workers[fid]
        c.delete(label="round %d: delete child %d" % (self.round, fid))
        self.restore_next(fid)
        self.latched.discard(fid)
        if fault is not None and fault.fid == fid:
            self.recover(fault, "gone")
        lost = [it for it in body if it.fid == fid and not it.done and it.kind != "fault"]
        self.child(fid)
        self.ring_again(lost, wait=False)
        self.release(body)
        if fault is not None and not fault.done:
            self.recover(fault, "kept")

    # ---- deaths ----

    def ev_pause_twice(self, body, during, fault):
        s = self.s
        s.api("pause", label="the first pause")
        s.api("pause", label="the second pause: the nub asserts")

    def ev_doorbell_paused(self, body, during, fault):
        s = self.s
        s.api("pause")
        s.w(abi.R_DISPLAY_DOORBELL, 0)

    def ev_will_unpause_nub(self, body, during, fault):
        s = self.s
        s.api("willSuspend")
        s.api("unpause", label="the nub was never paused: it asserts")

    def ev_finish_without_will(self, body, during, fault):
        self.s.api("finishSuspend", label="without willSuspend: asserts")

    def ev_failed_resume(self, body, during, fault):
        s = self.s
        s.api("willSuspend")
        s.api("finishSuspend")
        s.api("recreate")
        s.api("willResume", nil=True, label="nil state: refused, run state left LeavingSuspendEarly")
        s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))

    # ---- checks ----

    def check(self, items):
        s, x = self.s, self.x
        for it in self.encodes:
            if it.mid is not None:
                x.encode(self.disp[it.port], "round %d: mapping %d on port %d" % (self.round, it.mid, it.port))
            else:
                x.encode(self.disp[it.port], "round %d: backing %d on port %d" % (self.round, it.frame, it.port))
        slots = sorted({it.slot for it in items if it.kind == "exec"})
        if slots:
            lo, hi = slots[0], slots[-1] + 1
            self.cmp_slots(lo, hi, "round %d: slots %d-%d" % (self.round, lo, hi - 1))
        for mid in sorted({it.mid for it in items if it.kind == "surf"}):
            s.mem(s.pa(self.smirror[mid]), self.surf_bytes[mid])
            s.cmp(s.pa(x.surfaces[mid]), s.pa(self.smirror[mid]), FRAME, label="round %d: surface %d" % (self.round, mid))
        self.events("round %d" % self.round)
        s.r(abi.R_FAULT_MASK, "round %d" % self.round)
        self.task.reset_cmdbufs()

    def cmp_slots(self, lo, hi, label):
        s = self.s
        data = b"".join(self.slot_bytes.get(i, bytes(SLOT)) for i in range(lo, hi))
        s.mem(s.pa(self.mirror) + lo * SLOT, data)
        s.cmp(self.task.gpa(self.task.out_va) + lo * SLOT, s.pa(self.mirror) + lo * SLOT, (hi - lo) * SLOT,
              label=label)

    # ---- EFI ----

    def efi_boot(self):
        x = self.x
        self.efi_on = True
        x.efi_boot(seed=0x20 + (self.seed & 0x1F))

    def efi_frame(self):
        x, s = self.x, self.s
        seed = self.r.below(256)
        s.mark("round %d: an EFI frame" % self.round)
        x.efi.write_frame(FW, FH, 64, seed=seed)
        s.trace_fire(x.efi.gpa(0x10), 4)
        x.expect("newFrame", label="EFI frame %02x" % seed)
        x.encode(0, "EFI frame %02x" % seed)

    # ---- the walk ----

    def emit(self):
        f, s, x = self.f, self.s, self.x
        self.ndisp = f.ports
        if f.efi:
            self.efi_on = True
        self.setup(True)
        if f.efi:
            self.efi_boot()
        rounds = f.rounds
        for i in range(rounds):
            if self.efi_on and i >= min(3, rounds - 1):
                # The era ends in time for the lifecycle that needs it off.
                s.mark("round %d: EFI ends" % self.round)
                self.ev_efi_end([], [], None)
                self.round += 1
                continue
            if self.efi_on and self.r.chance(0.5):
                self.efi_frame()
            self.do_round()
        if f.death:
            if self.efi_on:
                self.ev_efi_end([], [], None)
            self.do_round(death=f.death)
            return s
        self.final()
        return s

    def final(self):
        s, x = self.s, self.x
        s.mark("the end")
        if self.nslots:
            self.cmp_slots(0, self.nslots, "every slot")
        s.dump(s.fifo.base_gpa(), 4 * (GATE + 1), "stamp slots")
        s.fault_state("end of the walk")
        for p in sorted(self.connected):
            s.dump(x.ss(p, 0xF98), 0x20, "port %d: present count, surface, mapping" % p)
        self.events("end of the walk")
        s.api("release")
        self.events("released")


def summary(w):
    return ", ".join(w.f.tags()) + "; %d rounds" % w.f.rounds


def build(seed, rounds, doc, death=False):
    w = Walk(seed, rounds, doc.strip(), death=death)
    s = w.emit()
    s.doc = doc.strip() + "\n\nThis program: " + summary(w) + "."
    return s


def check_expected(path, death=None, ops=None):
    """The model against what the expected file records: every wait
    succeeded but those whose label ends [held], which must time out; every
    cmp is equal; the run completes, or, for a program with a death, dies at
    its last op (ops: the scenario's op count). Returns a list of
    complaints."""
    import json
    bad = []
    last = None
    for line in open(path):
        if not line.startswith("{"):
            continue
        o = json.loads(line)
        lab = o.get("label") or ""
        ev = o.get("ev")
        if ev == "wait":
            if lab.endswith(HELD):
                if o.get("ok") is not False:
                    bad.append("wait did not time out: %s" % lab)
            elif o.get("ok") is False:
                bad.append("wait failed: %s" % lab)
        elif ev == "cmp" and not o.get("equal"):
            bad.append("cmp differs: %s" % lab)
        elif ev == "api" and o.get("ok") is False and "nil state" not in lab:
            bad.append("api refused: %s" % line.strip()[:200])
        last = o
    kind = (last or {}).get("kind") if (last or {}).get("ev") == "outcome" else None
    if death is None and kind != "completed":
        bad.append("does not complete: %s" % last)
    # "after": the last op that completed, the one before the death.
    if death is not None and (kind != "died" or last.get("after") != ops - 2):
        bad.append("does not die at its last op (%d): %s" % (ops - 1, last))
    return bad
