"""Builders for the guest-memory-under-live-work scenarios
(scenarios/interplay_memlive.py): a guest that rewrites its page tables,
leaf by leaf or a whole intermediate table at a time, moves its pages
between RAM ranges, aliases pages between tasks (and maps page-table pages
as data), maps and unmaps ranges, and grows a task, while work that uses
the memory is queued, held behind a barrier on another FIFO, or running.

The guest ABI is restated from byte offsets, as task.py and tasks_lib.py
do (docs/testing.md, section 6.10): a page-table level is one guest page of
4096 u32 entries, the VA page number splits top-down into 12-bit indices,
an entry's bits 0-30 are the guest page of the next table or of the data,
bit 31 of a leaf makes it writable, 0 is a hole. Rust paths name items of
reims-vgpu-core; nothing here is taken from the library's code.

What the scenarios lean on (read from the code, and locked by them):
- The page-table root and depth are read from the task root once, at
  define (setup_task_root, in tasks::local::new and, in the service,
  gpu_task::handshake); the tables below the root are read again at every
  walk (memory::page_table::PageTableWalker).
- A command buffer is walked through the tables at every exec, when the
  exec is processed (in process a copy; the service a cursor in place),
  and a FIFO copies a command's payload before it waits on the command's
  barriers: a held exec reads its command buffer and makes its objects
  only when released.
- A buffer's guest pages are bound when its host object is made: a shared
  buffer's Metal buffer and a private buffer's transfer buffer
  (resources::buffer::shared and private), or the service's detach
  backing (resources::buffer::detach). invalidateHost only marks the host
  copy stale: the copy is paged again from the pages bound before.
"""
import struct

from . import abi
from .abi import PAGE
from .diff_util import exec_payload
from .lifetime_util import E3, Life


# ---- page tables ----

def indices(t, vpn):
    """The table index at each level of t's walk for VA page vpn."""
    d = t.depth
    return [(vpn >> (12 * (d - 1 - level))) & 0xFFF for level in range(d)]


def prefix_of(t, vpn, level):
    """The prefix (the higher indices, concatenated) of the level-`level`
    table that vpn's walk reads."""
    p = 0
    for idx in indices(t, vpn)[:level]:
        p = (p << 12) | idx
    return p


def entry_gpa(t, vpn, level=None):
    """Where the entry for vpn at `level` (default the leaf) is in guest
    memory, in the tables t wrote."""
    level = t.depth - 1 if level is None else level
    return t.table_gpa(level, prefix_of(t, vpn, level), indices(t, vpn)[level])


def leaf(s, page, writable=True):
    return s.pfn(page) | (abi.PTE_WRITABLE if writable else 0)


def set_leaf(t, vpn, page, writable=True):
    """The guest points vpn's leaf at RAM page `page` (no command follows)."""
    t.s.u32(entry_gpa(t, vpn), leaf(t.s, page, writable))


def clear_leaf(t, vpn):
    """The guest punches a hole: vpn's leaf becomes 0."""
    t.s.u32(entry_gpa(t, vpn), 0)


def table_entries(t, level, prefix):
    """The entries t wrote into its level-`level` table under `prefix`:
    {index: u32}."""
    s = t.s
    out = {}
    for vpn, page in t.leaves.items():
        if prefix_of(t, vpn, level) != prefix:
            continue
        idx = indices(t, vpn)[level]
        if level == t.depth - 1:
            w = t.writable is True or (t.writable not in (True, False) and vpn in t.writable)
            out[idx] = leaf(s, page, w)
        else:
            out[idx] = s.pfn(t._tables[(level + 1, (prefix << 12) | idx)])
    return out


def write_table(s, entries, page=None):
    """A page-table page holding `entries` ({index: u32}), the rest holes;
    a fresh page unless one is given. Returns the page."""
    if page is None:
        page = s.alloc_page()
    for idx in sorted(entries):
        s.u32(s.pa(page, 4 * idx), entries[idx])
    return page


def spare(s, byte=None, data=None):
    """A fresh guest page, filled with `byte` or holding `data`."""
    page = s.alloc_page()
    if byte is not None:
        s.fill(s.pa(page), PAGE, byte)
    if data is not None:
        s.mem(s.pa(page), data)
    return page


# ---- a guest with a gate ----

class Rig(Life):
    """lifetime_util.Life (bring-up, a gate on child FIFO 1 and a worker on
    child FIFO 2) with task 1 as a Lab of `pages` VA pages, whose Lab
    keywords (depth, leaves, length, scatter) the scenario may set, or
    whatever make(s) returns."""

    def __init__(self, s, pages=48, worker=True, make=None, **lab):
        from .diff_util import Lab
        super().__init__(s, pages=pages, worker=worker, task=False)
        self.t = make(s) if make else Lab(s, tid=1, pages=pages, **lab)

    def hold_on(self, fifo, t, ranges, resources, label, opcode=E3, need=1):
        """An exec of t on `fifo` (root or child) behind the gate, rung; for
        the root FIFO waits 300 ms to see its stamp does not come. Returns
        its stamp."""
        st = fifo.cmd(opcode, exec_payload(t.tid, ranges, resources, opcode), barriers=self.barrier(need))
        fifo.doorbell()
        if fifo.fid == 0:
            fifo.wait_stamp(st, timeout_ms=300, label="%s: held by the gate" % label)
        return st

    def settle(self, st, fifo, label):
        """After the gate opens: on the root FIFO waits for stamp `st` or a
        fault (the observation says which), reads the fault state and
        resumes; then waits for the stamp. On a child FIFO only for the
        stamp: use it there for work that completes."""
        s = self.s
        if fifo.fid == 0:
            s.wait_any([dict(gpa=fifo.stamp_gpa(), ge=st), dict(mmio=abi.R_FAULT_MASK, mmio_eq=1)],
                       label="%s: completed (0) or faulted (1)" % label)
            s.fault_state(label)
            s.resume_root()
        fifo.wait_stamp(st, label="%s: stamp" % label)


def exec_on(fifo, t, ranges, resources, barriers=(), opcode=E3):
    """Writes an exec of t into `fifo` (with barriers) without ringing;
    returns its stamp."""
    return fifo.cmd(opcode, exec_payload(t.tid, ranges, resources, opcode), barriers=barriers)


def define_payload(tid, length, root_pfn):
    """DefineTask2 (56): {u32 taskID << 1, u64 length, u32 task root page}."""
    return struct.pack("<IQI", tid << 1, length, root_pfn)


def clone_tree(t, level, prefix, overrides=None):
    """A copy of t's level-`level` table under `prefix` and every table
    below it, each in a fresh page, with the leaves in `overrides` ({vpn:
    u32 entry, 0 for a hole}) changed; returns the new table's page. The
    guest points an intermediate entry at it to swap a whole subtree."""
    s = t.s
    overrides = overrides or {}
    entries = table_entries(t, level, prefix)
    if level < t.depth - 1:
        for idx in list(entries):
            entries[idx] = s.pfn(clone_tree(t, level + 1, (prefix << 12) | idx, overrides))
    else:
        for vpn, val in overrides.items():
            if prefix_of(t, vpn, level) == prefix:
                entries[indices(t, vpn)[level]] = val
        entries = {i: v for i, v in entries.items() if v}
    return write_table(s, entries)
