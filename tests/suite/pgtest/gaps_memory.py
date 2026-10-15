"""Builders for the memory, task and service gap scenarios
(scenarios/memory_gaps.py, tasks_gaps.py, service_gaps.py): guest RAM laid
out in several ranges with holes between them, tasks whose task root and
page-table pages live wherever the scenario says, and memory maps that
register ranges the way a host may (docs/testing.md, sections 3.3, 6.10).

Page numbers here are indices from RAM_BASE, as Scenario.pa and pfn take
them, whichever RAM range they fall in; nothing is taken from the
implementation.
"""
import struct

from . import abi
from .abi import PAGE
from .scenario import RAM_BASE, RAM_SIZE
from .task import ObjectList, seg
from .tasks_lib import XGPUTask, XTask, exec3_payload, fill_rec

R0_PAGES = RAM_SIZE // PAGE  # the first range's pages: where alloc_page hands out


def page_of(gpa):
    """The page index (from RAM_BASE) of a guest physical address."""
    assert (gpa - RAM_BASE) % PAGE == 0
    return (gpa - RAM_BASE) // PAGE


class Pool:
    """Hands out pages from a list, in order: where a task's root and page
    tables go."""

    def __init__(self, pages):
        self.pages = list(pages)

    def take(self):
        return self.pages.pop(0)


def placed_task(s, pool, cls=XTask, **kw):
    """A task (cls: XTask or XGPUTask) whose root page is pool's first page
    and whose tables follow from pool. XTask allocates its root page from
    the first range before anything else; this takes the root from the pool
    instead and writes the root there (reims-vgpu-core's
    guest::records::TaskRoot2)."""
    root = pool.take()

    class _T(cls):
        def _table(self, level, prefix):
            key = (level, prefix)
            if key not in self._tables:
                self._tables[key] = pool.take()
            return self._tables[key]

    t = _T(s, **kw)
    t.root = root
    t.write_root()
    return t


def fill_cb(t, ref, off, n, byte):
    """One blit segment filling [off, off + n) of buffer ref with byte, in
    the task's command region."""
    t.reset_cmdbufs()
    return t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(ref, off, n, byte)))


def fill_exec(s, t, ref, off, n, byte, resources=None, **kw):
    """ExecIndirect3 (67) of one fill of buffer ref."""
    cb = fill_cb(t, ref, off, n, byte)
    res = [(ref, 0)] if resources is None else resources
    return s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(t.tid, [cb], res), **kw)


def redirect_writes(s, since, frm, length, to):
    """Rewrites the scenario's guest-memory writes (op mem) made since op
    index `since` that fall in [frm, frm + length) to the same offsets from
    `to`: the guest's writes to guest-physical pages that the host's memory
    map backs with other RAM (an alias range) land, as the runner holds
    them, where that RAM is. A write straddling the range's edge is a
    scenario error."""
    for op in s.ops[since:]:
        if op.get("op") != "mem":
            continue
        gpa = int(op["gpa"], 16)
        n = len(op["hex"]) // 2
        if frm <= gpa < frm + length:
            assert gpa + n <= frm + length, "write straddles the alias range"
            op["gpa"] = hex(to + gpa - frm)
