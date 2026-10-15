"""Dirty resources over one another's guest memory, then a suspend
(pgtest/alias_util.py): the alias generator's programs at fixed seeds.

Two or three tasks, with IDs drawn from 1-47, map their data pages to one
pool of guest pages; each task's private buffers and textures lie on pages of
their own, which the other tasks' resources share. Execs write them on the
GPU, most under invalidateGuest (their guest copies go stale), some without
it, some synchronized (53) or discarded (63) after. rvg_device_will_suspend
flushes every task: in ascending task ID, each task's stale resources written
back and waited for before the next task's. Where resources of two tasks
overlap, the pool's bytes after the flush say which task's flush came last;
rvg_device_finish_suspend then finds nothing stale. Every setup runs: in the
service buffers are detach resources over guest memory (their writes land at
once, nothing of theirs is flushed) and textures are paged as in-process.

Resources of one task that alias one another are left out: their write-backs
share one blit encoder, which Metal runs concurrently, so their bytes
interleave differently on every run (alias_util's measurement).
"""
from pgtest.alias_util import build
from pgtest.registry import scenario

# (seed, resources)
CASES = [(1, 3), (4, 5), (1, 8), (5, 8), (2, 12)]


@scenario
def native_alias():
    out = []
    for seed, n in CASES:
        s, params = build(seed, n, __doc__, intra=False)
        s.name = "native_alias_%d_%d" % (n, seed)
        out.append(s)
    return out
