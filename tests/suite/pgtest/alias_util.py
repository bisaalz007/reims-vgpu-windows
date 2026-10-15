"""The `alias` generator (scenarios/native_alias.py): dirty resources that
alias one another's guest memory, then a suspend whose flush writes them
back.

Every task's data pages 8.. map to one pool of guest RAM pages, so the
private buffers and textures of every task lie over the same bytes, at
page starts the generator draws. Execs write them on the GPU under
invalidateGuest (flag 2), which leaves their guest copies stale; some
are written without it, synchronized (53) or discarded (63) first.
rvg_device_will_suspend and rvg_device_finish_suspend then flush every task
(reims-vgpu-core's ResourceManager::flush_resources,
resources/manager/flush.rs): tasks in ascending task ID, and within a task
every stale resource in the resource cache's order, CPU-paged ones first, then
one blit encoder and one compute encoder. Each write-back copies the whole
resource over the pool, so the pool's bytes after the flush say which resource
was written back last over each byte: the order of the task table and of the
resource caches, which the suite otherwise sees only in single-resource
flushes.

The references are spread by filler records the program never names (a
cache holds only what an exec looked up), so the cache's key order and its
insertion order (the order execs first name resources) differ.

In the service (isolate) buffers are detach resources over guest memory:
their writes land at once and nothing of theirs is flushed, while its
textures are paged as in-process. Every setup's outcome is recorded as it
runs.

Measured (2026-09-27): resources of one task that alias one another
(`intra`) make the outcome nondeterministic. The flush's copies share one
blit encoder, and Metal runs them concurrently, so the bytes where two of
them overlap come out interleaved differently on every run; so do two
aliasing fills in one exec. Such programs differ from run to run of the
same build: through aliasing, the resource cache's order is not
observable on this host, only the task table's (each task's flush commits
and waits before the next task's starts). So the suite's scenarios
(scenarios/native_alias.py) alias across tasks only; build(intra=True)
still makes the racing programs.
"""
import random

from . import abi
from .abi import PAGE
from .resources_build import RGBA8Unorm, b_fill_buf, b_fill_tex_color, buffer_rec, desc1, simple_dim, texture_rec
from .resources_build import cmd_discard, cmd_synchronize
from .task import seg
from .tasks_lib import XGPUTask, exec3_payload

FLAG_DIRTY = abi.RES_INVALIDATE_GUEST


class _Res:
    def __init__(self, ref, kind, va, length, w=0, h=0):
        self.ref, self.kind, self.va, self.length, self.w, self.h = ref, kind, va, length, w, h


class AliasTask(XGPUTask):
    """An XGPUTask whose VA pages 0-7 are its own and 8.. the shared pool."""

    def add(self, typ, record):
        return self.ol.add(typ, record)


def _filler(t):
    """A record no command names: it only moves the next reference up."""
    return t.add(abi.T_BUFFER, buffer_rec(0x100, 8))


def build(seed, n, doc, intra):
    """The program for seed: n resources in all (2-24) over 1-3 tasks and a
    pool of guest pages, 3-14 writes, then the suspend. With intra
    resources of one task may alias one another; without it only resources
    of different tasks do. Returns (Scenario, params)."""
    from .scenario import Scenario
    r = random.Random("alias:%d:%d" % (seed, n))
    s = Scenario("native_alias_%d" % seed, doc=doc)
    s.bring_up()
    ntasks = r.choices((1, 2, 3), (0.45, 0.35, 0.2))[0]
    if not intra and ntasks == 1:
        ntasks = 2
    tids = sorted(r.sample(range(1, 48), ntasks), key=lambda _: r.random())
    counts = [1] * ntasks
    for _ in range(max(0, n - ntasks)):
        counts[r.randrange(ntasks)] += 1
    # Each resource's pages: a buffer 1 or 2 pages, a texture 1 (lengths are
    # whole pages: a mapping of a part of a page is refused, "Attempting to
    # map memory with a non-page-aligned length"). Without intra, a task's
    # resources lie on pages of their own, which the other tasks' share.
    shapes = [[r.choice(("b1", "b1", "b2", "t")) for _ in range(m)] for m in counts]
    sizes = [[2 if x == "b2" else 1 for x in sh] for sh in shapes]
    npool = r.randint(2, 5) if intra else max(sum(z) for z in sizes)
    pool = s.alloc_page(npool)
    # The pool's first bytes: a pattern, so a write-back of bytes a
    # resource paged in (not written) is told from one of zeros.
    for i in range(npool):
        s.fill(s.pa(pool + i), PAGE, 0x40 + i)
    tasks = []
    for tid, shape, size in zip(tids, shapes, sizes):
        leaves = {v: s.alloc_page() for v in range(8)}
        leaves.update({8 + i: pool + i for i in range(npool)})
        t = AliasTask(s, tid=tid, pages=8 + npool, leaves=leaves)
        if intra:
            pages = [r.randrange(npool - z + 1) for z in size]
        else:
            order = list(range(len(size)))
            r.shuffle(order)
            at, first = r.randrange(npool - sum(size) + 1), {}
            for i in order:
                first[i] = at
                at += size[i]
            pages = [first[i] for i in range(len(size))]
        res = []
        for kind, z, page in zip(shape, size, pages):
            for _ in range(r.choice((0, 0, 1, 2, 5, 11))):
                _filler(t)
            if kind == "t":
                w, h = r.choice(((16, 16), (32, 8), (64, 16), (64, 64)))
                ref = t.add(abi.T_TEXTURE, texture_rec(PAGE, 8 + page, [simple_dim(w, h)],
                                                       desc1(fmt=RGBA8Unorm, w=w, h=h, usage=3)))
                res.append(_Res(ref, "texture", (8 + page) * PAGE, w * h * 4, w, h))
            else:
                ref = t.add(abi.T_BUFFER, buffer_rec(z * PAGE, 8 + page, 1))
                res.append(_Res(ref, "buffer", (8 + page) * PAGE, z * PAGE))
        tasks.append((t, res))
    for t, _ in tasks:
        t.define(label="define task %d" % t.tid)
        t.install(label="object list of task %d" % t.tid)

    steps = r.randint(3, 14)
    kinds = {"dirty": 0, "clean": 0, "sync": 0, "discard": 0}
    for k in range(steps):
        t, res = r.choice(tasks)
        pick = r.sample(res, min(len(res), r.choice((1, 1, 2, 3))))
        recs, flags = b"", []
        for x in pick:
            byte = (0x13 * (k + 1) + 7 * x.ref) & 0xFF
            if x.kind == "buffer":
                off = r.randrange(0, x.length, 0x40)
                ln = min(x.length - off, r.choice((0x40, 0x100, 0x400, x.length)))
                recs += b_fill_buf(x.ref, off, ln, byte)
            else:
                ox, oy = r.randrange(x.w), r.randrange(x.h)
                size = (r.randint(1, x.w - ox), r.randint(1, x.h - oy), 1)
                color = (byte / 255.0, ((byte * 3) & 0xFF) / 255.0, ((byte * 5) & 0xFF) / 255.0, 1.0)
                recs += b_fill_tex_color(x.ref, size, color, RGBA8Unorm, origin=(ox, oy, 0))
            f = FLAG_DIRTY if r.random() < 0.8 else 0
            kinds["dirty" if f else "clean"] += 1
            flags.append((x.ref, f))
        t.reset_cmdbufs()
        cb = t.cmdbuf(seg(abi.ENC_BLIT, recs))
        s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(t.tid, [cb], flags),
                 label="write %d: task %d, %s" % (k, t.tid, ", ".join(str(x.ref) for x in pick)))
        u = r.random()
        if u < 0.1:
            x = r.choice(pick)
            cmd_synchronize(s, t.tid, [x.ref], label="synchronize %d of task %d" % (x.ref, t.tid))
            kinds["sync"] += 1
        elif u < 0.15:
            x = r.choice(pick)
            cmd_discard(s, t.tid, [x.ref], label="discard %d of task %d" % (x.ref, t.tid))
            kinds["discard"] += 1
    s.diff("before the suspend", gpa=s.pa(pool), length=npool * PAGE)
    s.api("willSuspend")
    s.diff("after willSuspend", gpa=s.pa(pool), length=npool * PAGE)
    s.api("finishSuspend", slot="s")
    s.diff("after finishSuspend", gpa=s.pa(pool), length=npool * PAGE)
    s.events("end")
    params = "resources=%d tasks=%s pool=%d intra=%s writes=%d (%s)" % (
        n, ",".join(str(t.tid) for t, _ in tasks), npool, "yes" if intra else "no", steps,
        " ".join("%s %d" % kv for kv in sorted(kinds.items())))
    return s, params
