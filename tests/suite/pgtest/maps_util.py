"""The `maps` generator (scenarios/native_maps.py): a device whose memory
map has 24 to 64 ranges, with shared starts and zero-length ranges, and a
task whose buffers lie across them.

What it reaches that the suite's other maps (16 ranges at most) do not:
many ranges, and shared starts in any descriptor order. The memory map
sorts its ranges by start and then length (reims-vgpu-core's
memory::map::MemoryMap::new), so zero-length ranges come before the range
with a length that shares their start, whatever order the host listed
them in. It then refuses the map when a range starts before the end of
the one sorted before it: only the previous range's end is compared, not
the furthest end so far. A map is accepted when no two ranges with a
length share a start or overlap, however its descriptor lists them.

After the sort, nothing else sees that order: a lookup takes the first
range that contains an address, and a zero-length range contains none
(MemoryMap's virtual_address_for_physical and new_range_mapping, both
first match in stored order). The workload then maps buffers over pages
the ranges hold, across adjacent ranges (a mapping takes a piece of each
range) and into gaps between them (refused: the exec faults).

The ranges are identity ranges of the scenario's RAM (a range's guest
address is its host backing's), so the builders' guest writes are what the
device reads. Setups: map and isolate (the blocks setup has no memory
map).

`accepts` below restates the rule, to predict which maps the device
accepts; the generator builds its workload only for maps it accepts.
"""
import random

from . import abi
from .abi import PAGE
from .scenario import RAM_BASE, RAM_SIZE, Scenario
from .task import seg
from .tasks_lib import XGPUTask, exec3_payload, fault_and_resume, fill_rec

# ---- the memory map's rule ----


def sort_key(r):
    """The memory map's order: by start, then length. Python's sort is
    stable, as the map's is, so ranges equal in both keep the descriptor's
    order."""
    return (r[0], r[1])


def accepted(ranges):
    """The memory map's check after the sort: each range starts at or
    after the end of the one before it (only the previous end counts); an
    overflowing end is refused. ranges: [(start, length, ...)] in stored
    order."""
    end = 0
    for r in ranges:
        if r[0] < end:
            return False
        if r[0] + r[1] >= 1 << 64:
            return False
        end = r[0] + r[1]
    return True


def accepts(desc):
    """Whether the memory map accepts a descriptor [(start, length, host)]
    in addRange order."""
    return accepted(sorted(desc, key=sort_key))


# ---- the generator ----

STRUCT_PAGES = 128   # RAM pages 0-127: the scenario's own (alloc_page), one range
DATA0 = 160          # the data ranges start at this RAM page
FAR = 0x200000000    # zero-length ranges past RAM start here


def _place_run(order, members, first):
    """Reorders, in place, the descriptor positions `order` holds for the
    members of one run of equal starts, so that the one with a length takes
    the first (first=True) or the last of those positions."""
    pos = sorted(i for i, r in enumerate(order) if r in members)
    nonzero = [r for r in members if r[1]]
    zeros = [order[i] for i in pos if not order[i][1]]  # the shuffle's order
    seq = (nonzero + zeros) if first else (zeros + nonzero)
    for i, r in zip(pos, seq):
        order[i] = r


class MapPlan:
    """A memory map: the scenario's own range, K data ranges with gaps,
    and zero-length ranges, in a descriptor order."""

    def __init__(self, r, n):
        self.n = n
        struct = (RAM_BASE, STRUCT_PAGES * PAGE, RAM_BASE)
        zeros_min = 2
        k = r.randint(max(4, n // 3), max(4, n - 1 - zeros_min - r.randint(0, n // 3)))
        k = min(k, n - 1 - zeros_min)
        if r.random() < 0.2:
            k = n - 1  # no zero-length range: the service gets every range
        z = n - 1 - k
        self.pieces = []           # (first RAM page, pages)
        page = DATA0
        for _ in range(k):
            page += 0 if r.random() < 0.6 else r.randint(1, 2)
            npg = r.randint(1, 4)
            self.pieces.append((page, npg))
            page += npg
        self.data_end = page
        assert self.data_end * PAGE <= RAM_SIZE
        nonzero = [struct] + [(RAM_BASE + p * PAGE, n_ * PAGE, RAM_BASE + p * PAGE) for p, n_ in self.pieces]
        starts_nz = [x[0] for x in nonzero]
        # Ends no other range starts at: a zero-length range there follows
        # the range that ends there. (At an end another range starts at, it
        # would make one more mixed run.)
        lone_ends = sorted({x[0] + x[1] for x in nonzero} - set(starts_nz))
        # How many runs mix zero-length ranges with the range that has a
        # length (usually 1-2).
        mixed = r.choices((0, 1, 2, 3), (0.2, 0.5, 0.2, 0.1))[0]
        mixed = min(mixed, z)
        mixed_starts = r.sample(starts_nz, mixed) if mixed else []
        zeros = []
        for i in range(z):
            host = RAM_BASE + r.randrange(STRUCT_PAGES) * PAGE
            if i < mixed:
                start = mixed_starts[i]
            else:
                pick = r.random()
                if pick < 0.08 and mixed_starts:
                    start = r.choice(mixed_starts)       # one more in a mixed run
                elif pick < 0.55 and lone_ends:
                    start = r.choice(lone_ends)          # where a range ends, before a gap
                elif pick < 0.8:
                    gaps = self.gap_pages()
                    start = RAM_BASE + (r.choice(gaps) if gaps else self.data_end) * PAGE
                else:
                    start = FAR + r.randrange(4) * PAGE  # past RAM, zeros only
            zeros.append((start, 0, host))
        self.ranges = nonzero + zeros
        # Tag every range so equal tuples stay distinct in the sort.
        self.ranges = [(a, b, c, i) for i, (a, b, c) in enumerate(self.ranges)]
        self.zeros = z
        runs = {}
        for x in self.ranges:
            runs.setdefault(x[0], []).append(x)
        self.mixed_runs = sum(1 for v in runs.values() if len(v) > 1 and any(y[1] for y in v))
        order = list(self.ranges)
        r.shuffle(order)
        self.mode = r.choices(("shuffled", "nonzero_last", "nonzero_first"), (0.4, 0.4, 0.2))[0]
        if self.mode != "shuffled":
            for v in runs.values():
                if len(v) > 1 and any(y[1] for y in v):
                    _place_run(order, v, self.mode == "nonzero_first")
        self.order = order
        self.ok = accepts([(a, b, c) for a, b, c, _ in order])

    def gap_pages(self):
        out, prev = [], DATA0
        for p, n in self.pieces:
            out.extend(range(prev, p))
            prev = p + n
        return out

    def mapped(self, page):
        """Whether RAM page `page` lies in a range with a length."""
        if page < STRUCT_PAGES:
            return True
        return any(p <= page < p + n for p, n in self.pieces)

    def descriptor(self):
        return {"map_ranges": [[hex(a), hex(b), hex(c)] for a, b, c, _ in self.order]}

    def params(self):
        return "ranges=%d pieces=%d zeros=%d mixed=%d order=%s map=%s" % (
            len(self.order), len(self.pieces), self.zeros, self.mixed_runs, self.mode,
            "ok" if self.ok else "refused")


def _workload(s, r, plan, steps):
    """A task whose buffers lie over the data ranges: fills, copies and
    dumps; a buffer with a page outside every range faults its exec."""
    s.bring_up()
    s.submit(abi.OP_NOP, label="the map made")
    data_pages = list(range(DATA0, plan.data_end + 2))
    nbuf = r.randint(2, 6)
    leaves = {v: s.alloc_page() for v in range(8)}
    bufs = []
    va_page = 8
    for _ in range(nbuf):
        npg = r.randint(1, 3)
        if r.random() < 0.75:
            p0 = r.choice(data_pages)
            pages = [min(p0 + i, plan.data_end + 1) for i in range(npg)]
        else:
            pages = [r.choice(data_pages) for _ in range(npg)]
        for i, p in enumerate(pages):
            leaves[va_page + i] = p
        bufs.append((va_page, npg, all(plan.mapped(p) for p in pages)))
        va_page += npg
    assert s._next_page <= STRUCT_PAGES, "the scenario's pages outgrew its range"
    t = XGPUTask(s, tid=r.randint(1, 8), pages=va_page, leaves=leaves)
    refs = []
    for va, npg, ok in bufs:
        ref, _ = t.buffer(length=npg * PAGE, va=va * PAGE, flags=1 if r.random() < 0.3 else 0)
        refs.append((ref, va * PAGE, npg * PAGE, ok))
    assert s._next_page <= STRUCT_PAGES, "the scenario's pages outgrew its range"
    t.define(label="define")
    t.install()
    for k in range(steps):
        ref, va, n, ok = r.choice(refs)
        off = r.randrange(0, n, 0x100)
        ln = min(n - off, r.choice((0x40, 0x100, 0x2000, PAGE, 2 * PAGE)))
        byte = (0x11 * (k + 1) + 3) & 0xFF
        recs = fill_rec(ref, off, ln, byte)
        res = [(ref, 0)]
        other = r.choice(refs)
        if other[0] != ref and other[3] and ok and r.random() < 0.3:
            m = min(0x100, other[2], n)
            recs += fill_rec(other[0], 0, m, byte ^ 0xFF)
            res.append((other[0], 0))
        t.reset_cmdbufs()
        cb = t.cmdbuf(seg(abi.ENC_BLIT, recs))
        label = "fill %d: buffer %d at +%#x (%s)" % (k, ref, off, "mapped" if ok else "outside every range")
        if ok:
            s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(t.tid, [cb], res), label=label)
            t.dump(va + off, min(ln, 0x40), label=label)
            if ln > 0x40:
                t.dump(va + off + ln - 0x20, 0x20, label=label + ", its end")
        else:
            s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(t.tid, [cb], res), fault=True, label=label)
            fault_and_resume(s, label)
    s.events("the fills")
    t.delete(label="delete")
    s.events("end")


def build(seed, n, doc, steps=None):
    """The program for seed: a map of n ranges (24-64); its workload when
    the device accepts the map. Returns (Scenario, MapPlan)."""
    r = random.Random("maps:%d:%d" % (seed, n))
    plan = MapPlan(r, n)
    s = Scenario("native_maps_%d" % seed, setups=("map", "isolate"), descriptor=plan.descriptor(),
                 watchdog_s=30, doc=doc)
    if plan.ok:
        _workload(s, r, plan, steps if steps is not None else r.randint(3, 10))
    else:
        # A refused map: no device. Nothing to drive; the record of the
        # device's creation is the observation.
        s.mark("the map refused: no device")
    return s, plan
