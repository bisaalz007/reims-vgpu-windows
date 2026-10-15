"""The scenario builder: a guest driver that writes ops for the runner.

A Scenario is the guest and the host's script at once: it lays out guest
memory, writes registers, commands and records, waits for what the guest can
see, and takes checkpoints. It knows the guest ABI only through abi.py and
the builders beside it, never through the implementation's source.

Checkpoints (diff, dump, events, register reads) are the observations the
expected file locks. Take one only after a wait on something the guest sees
(a stamp, a register that reading does not clear, a callback); the events
checkpoint also settles, so interrupt and callback counts are exact.
"""
import hashlib
import json
import os
import struct

from . import abi
from .abi import PAGE

SETUPS = ("blocks", "map", "isolate")
RAM_BASE = 0x40000000
RAM_SIZE = 64 << 20

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")


def h(v):
    return hex(v)


class Scenario:
    """One scenario: a list of runner ops.

    setups: the host setups it runs in (docs/testing.md, section 3.3). A
      scenario split into phases (phase()) runs each phase in its own
      process, and a setup may then be a chain, one setup a phase: a tuple
      ("map", "isolate") or the string "map-isolate". A plain setup runs
      every phase in it. Chains are listed one by one, never a cross product.
    host: True if its observations report the host (the GPU, the OS), so its
      expected files are keyed by the host.
    exclusive: True if it faults or hangs the GPU, or crashes Metal's
      compiler service. A GPU error discards other processes' command
      buffers too ("victim of GPU error"), and each compiler crash costs
      Metal several retries and a crash report, which under load outlast
      the watchdog; so such a scenario runs alone, after the others.
    services: how many isolated tasks' services the scenario holds at once
      under isolation (one a task, each on threads of the runner's
      process), when that is many more than a few: the orchestrator keeps
      such cases from ramping up together (cli.run_all). Not part of the
      generated scenario.
    """

    services = 0

    def __init__(self, name, setups=SETUPS, host=False, watchdog_s=30, ram=None,
                 displays=None, descriptor=None, iosfc=False, iosfc_after=False,
                 iosfc_no_map=False, stamp_slots=2, exclusive=False, doc=""):
        self.name = name
        self.exclusive = exclusive
        self.setups = tuple(s if isinstance(s, str) else "-".join(s) for s in setups)
        self.host = host
        self.doc = doc
        self.ops = []
        self.ram = ram or [(RAM_BASE, RAM_SIZE)]
        self._next_page = 1  # page 0 of RAM is never handed out: page 0 means "none"
        self.root_page = self.alloc_page()
        self.fifo = None
        self.fifos = {}
        cfg = {"op": "config", "watchdog_s": watchdog_s,
               "ram": [[h(a), h(n)] for a, n in self.ram],
               "stamp_slots": stamp_slots}
        if displays:
            cfg["displays"] = displays
        if descriptor:
            cfg["descriptor"] = descriptor
        if iosfc:
            cfg["iosfc"] = True
            if iosfc_after:
                cfg["iosfc_after"] = True
            if iosfc_no_map:
                cfg["iosfc_no_map"] = True
        self._config = cfg
        self.ops.append(cfg)
        self._on_phase = []  # callbacks run at each phase(): per-process counters

    # ---- guest memory ----

    def pfn(self, page):
        """The guest page number of RAM page `page`, as registers and
        payloads take them (GPA >> 14)."""
        return (self.ram[0][0] >> 14) + page

    def pa(self, page, off=0):
        """The guest physical address of RAM page `page`, plus off."""
        return self.ram[0][0] + page * PAGE + off

    def alloc_page(self, n=1):
        """Hands out n contiguous RAM pages; returns the first's index."""
        p = self._next_page
        self._next_page += n
        assert self._next_page * PAGE <= self.ram[0][1], "guest RAM exhausted"
        return p

    def op(self, **kw):
        self.ops.append(kw)
        return kw

    def mem(self, gpa, data):
        if data:
            self.op(op="mem", gpa=h(gpa), hex=bytes(data).hex())

    def mem_seq(self, gpa, count, stride, template, fields=()):
        """count copies of template at gpa + k * stride; each field (off,
        size, start, step) stores start + k * step, little-endian, size
        bytes (4 or 8), at off in copy k. For structures too large to spell
        out: thousands of commands, or page tables of thousands of entries."""
        self.op(op="mem_seq", gpa=h(gpa), count=count, stride=stride, hex=bytes(template).hex(),
                fields=[[o, n, h(a), h(b)] for o, n, a, b in fields])

    def fill(self, gpa, length, byte=0):
        self.op(op="fill", gpa=h(gpa), len=h(length), byte=byte)

    def mem_file(self, gpa, rel):
        """Writes a fixture (tests/suite/fixtures/rel) at gpa; returns its
        length. The runner checks the file's SHA-256."""
        data = open(os.path.join(FIXTURES, rel), "rb").read()
        self.op(op="mem_file", gpa=h(gpa), file=rel, sha256=hashlib.sha256(data).hexdigest())
        return len(data)

    def u32(self, gpa, *vals):
        self.mem(gpa, struct.pack("<%dI" % len(vals), *vals))

    # ---- registers ----

    def w(self, off, val):
        self.op(op="mmio_w", off=h(off), val=h(val & 0xFFFFFFFF))

    def r(self, off, label=None):
        o = self.op(op="mmio_r", off=h(off))
        if label:
            o["label"] = label

    def sweep(self, lo=0, hi=0x4000, step=4, label=None):
        o = self.op(op="mmio_sweep", **{"from": h(lo), "to": h(hi), "step": step})
        if label:
            o["label"] = label

    def iosfc_w(self, off, val):
        self.op(op="iosfc_w", off=h(off), val=h(val & 0xFFFFFFFF))

    def iosfc_r(self, off, label=None):
        o = self.op(op="iosfc_r", off=h(off))
        if label:
            o["label"] = label

    # ---- waits and checkpoints ----

    _HEX = ("gpa", "eq", "ge", "ne", "mask", "mmio", "mmio_eq", "mmio_ne")

    def _cond(self, cond):
        return {k: (h(v) if k in self._HEX else v) for k, v in cond.items()}

    def wait(self, timeout_ms=5000, label=None, **cond):
        o = dict(op="wait", timeout_ms=timeout_ms)
        o.update(self._cond(cond))
        if label:
            o["label"] = label
        self.ops.append(o)

    def wait_any(self, conds, timeout_ms=5000, label=None):
        """Waits until one of several conditions holds; the observation says
        which (the first in list order that holds)."""
        o = dict(op="wait", timeout_ms=timeout_ms, any=[self._cond(c) for c in conds])
        if label:
            o["label"] = label
        self.ops.append(o)

    def track_stamps(self, slots):
        """How many stamp slots each interrupt's snapshot holds (events with
        irq_stamps=True)."""
        self._config["stamp_slots"] = slots

    def diff(self, label=None, gpa=None, length=None):
        o = self.op(op="diff")
        if label:
            o["label"] = label
        if gpa is not None:
            o["gpa"], o["len"] = h(gpa), h(length)

    def dump(self, gpa, length, label=None):
        o = self.op(op="dump", gpa=h(gpa), len=h(length))
        if label:
            o["label"] = label

    def cmp(self, a, b, length, label=None, skip=None):
        """Compares two guest ranges (a differential check): the observation
        says whether they are equal, and where they first differ. skip:
        (offset, length) ranges left out, such as a reply's padding."""
        o = self.op(op="cmp", a=h(a), b=h(b), len=h(length))
        if skip:
            o["skip"] = [[h(x), h(n)] for x, n in skip]
        if label:
            o["label"] = label

    def events(self, label=None, order=False, irq_stamps=False, quiet_ms=None, collapse=None,
               sort_blocks=False, blocks_digest=False, collapse_per_display=False, sort_callbacks=False):
        """collapse: callback kinds whose consecutive repeats for one display
        count as one event (the EFI timer's new_frame), and "irq": interrupts
        count by runs, since one interrupt publishes every ready stamp at the
        head of a FIFO (the pump of reims-vgpu-core's
        `fifo::stamps::StampQueue`), so after a batch with an asynchronous
        command (an exec, whose completion signals it) how many of its
        stamps share one is timing (docs/testing.md 5.4). Only what
        the checkpoint orders separates runs: with order, a callback, an
        IOSurface interrupt or an ordered block call; without it, nothing,
        so irqs is 1 if any interrupt came and 0 if none did. A run's
        irq_stamps entry is its last interrupt's snapshot. With
        collapse_per_display, a repeat of that display's previous callback
        whatever came between (presents coalesce however they interleave).
        sort_callbacks: report callbacks sorted and leave them out of order,
        for displays presenting at once. sort_blocks: report
        the host's block calls sorted and leave them out of order, for FIFOs
        running at once, whose calls interleave by timing (blocks setup).
        blocks_digest: report block calls as counts per kind and a SHA-256
        of their records, for tens of thousands of them. The final events
        at teardown reports only what came after the last checkpoint."""
        o = self.op(op="events")
        if label:
            o["label"] = label
        if order:
            o["order"] = True
        if irq_stamps:
            o["irq_stamps"] = True
        if quiet_ms is not None:
            o["quiet_ms"] = quiet_ms
        if collapse:
            o["collapse"] = list(collapse)
        if collapse_per_display:
            o["collapse_per_display"] = True
        if sort_callbacks:
            o["sort_callbacks"] = True
        if sort_blocks:
            o["sort_blocks"] = True
        if blocks_digest:
            o["blocks_digest"] = True

    def quiesce(self, quiet_ms=40):
        self.op(op="quiesce", quiet_ms=quiet_ms)

    def mark(self, label):
        self.op(op="mark", label=label)

    def api(self, name, **kw):
        return self.op(op="api", name=name, **kw)

    def encode(self, display=0, w=64, h_=32, **kw):
        return self.op(op="encode", display=display, w=w, h=h_, **kw)

    def encode_commit(self, held, label=None):
        """Commits the frame encode(hold=True) returned as held number
        `held` (0 for the first), and reports its pixels."""
        o = self.op(op="encode_commit", held=held)
        if label:
            o["label"] = label

    def metal(self, query, **kw):
        return self.op(op="metal", query=query, **kw)

    def trace_fire(self, gpa, length):
        self.op(op="trace_fire", gpa=h(gpa), len=h(length))

    # ---- the root FIFO (section 6.5) ----

    def bring_up(self, version=abi.MAX_VERSION, fifo_pages=2, start=0x400, root=True):
        """The minimal sequence (section 6.25 a): version first (it sets the
        features that gate FIFO counts and commands), root page, FIFO base,
        length and start, enable."""
        from .fifo import RootFIFO
        if version is not None:
            self.w(abi.R_VERSION, version)
        if root:
            self.w(abi.R_ROOT_PAGE, self.pfn(self.root_page))
        self.fifo = RootFIFO(self, fifo_pages, start)
        # Interrupts snapshot the root FIFO's stamp slots (events(irq_stamps=True)).
        self._config.setdefault("stamp_gpa", h(self.fifo.base_gpa()))
        self.fifo.enable()
        self.fifos[0] = self.fifo
        return self.fifo

    def cmd(self, opcode, payload=b"", **kw):
        return self.fifo.cmd(opcode, payload, **kw)

    def doorbell(self):
        self.fifo.doorbell()

    def submit(self, opcode, payload=b"", fault=False, timeout_ms=5000, label=None, **kw):
        """One command on the root FIFO: write, ring, and wait until it
        completed (its stamp is visible) or, with fault=True, until the device
        reports a fault (0x102c reads 1), or with fault="either", until one
        of the two (the observation says which: 0 completed, 1 faulted)."""
        st = self.fifo.cmd(opcode, payload, **kw)
        self.fifo.doorbell()
        if fault == "either":
            self.wait_any([dict(gpa=self.fifo.stamp_gpa(), ge=st), dict(mmio=abi.R_FAULT_MASK, mmio_eq=1)],
                          timeout_ms=timeout_ms, label=label or "stamp %d or a fault" % st)
        elif fault:
            self.wait_fault(timeout_ms=timeout_ms, label=label)
        else:
            self.fifo.wait_stamp(st, timeout_ms=timeout_ms, label=label)
        return st

    def wait_fault(self, timeout_ms=5000, label=None):
        self.wait(mmio=abi.R_FAULT_MASK, mmio_eq=1, timeout_ms=timeout_ms, label=label or "fault")

    def resume_root(self):
        """Resume the root FIFO after a fault (0x1024)."""
        self.w(abi.R_RESUME_ROOT, 0)

    def fault_state(self, label=None):
        """The guest-visible fault state: 0x102c, 0x103c."""
        self.r(abi.R_FAULT_MASK, label=label)
        self.r(abi.R_FAULT_OFFSET, label=label)

    # ---- phases: a migration to a new process ----

    def on_phase(self, fn):
        """Registers fn(), called at every phase(): a builder that counts
        what the runner counts per process (interrupts, callbacks, blocks)
        starts again at 0 there."""
        self._on_phase.append(fn)

    def phases(self):
        return 1 + sum(1 for o in self.ops if o.get("op") == "phase")

    def phase_setups(self, setup):
        """The setup of each phase for one of self.setups."""
        chain = setup.split("-")
        n = self.phases()
        if len(chain) == 1:
            return chain * n
        assert len(chain) == n, "%s: setup %s for %d phases" % (self.name, setup, n)
        return chain

    def handoff_save(self, image="0"):
        """Hands guest RAM (and the scenario's shadow of it) and the blob
        slots on to a later phase, as image `image`."""
        self.op(op="handoff_save", image=image)

    def phase(self, save=True, load=True, image="0", ram=True, blobs=True, **cfg_over):
        """Ends this phase (its process tears down as at a scenario's end)
        and starts the next in a new runner process: saves the handoff image
        (unless save=False: the next phase loads an earlier one), then a
        config like the first, changed by cfg_over (ram_ranges=[(gpa, len),
        ...] for other guest RAM; displays=[...], descriptor=..., iosfc=...;
        a value of None drops the key), that loads it (ram=False: blobs only,
        RAM zeroed). Guest addresses (pa, pfn) stay the first phase's.
        Everything the runner counts starts again at 0, so the builders
        registered with on_phase reset their counts."""
        if save:
            self.handoff_save(image)
        self.op(op="phase")
        cfg = json.loads(json.dumps(self._config))
        for k, v in cfg_over.items():
            if k == "ram_ranges":
                cfg["ram"] = [[h(a), h(n)] for a, n in v]
            elif v is None:
                cfg.pop(k, None)
            else:
                cfg[k] = v
        if load:
            spec = {"image": image}
            if not ram:
                spec["ram"] = False
            if not blobs:
                spec["blobs"] = False
            cfg["handoff_load"] = spec
        self.ops.append(cfg)
        for fn in self._on_phase:
            fn()
        return cfg

    # ---- output ----

    def jsonl(self):
        lines = ["# %s: %s" % (self.name, self.doc.strip().replace("\n", " "))] if self.doc else []
        lines += [json.dumps(o, sort_keys=True) for o in self.ops]
        return "\n".join(lines) + "\n"
