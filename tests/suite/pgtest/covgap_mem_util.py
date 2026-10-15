"""Builders for the memory, resource and texture scenarios of
scenarios/covgap_mem.py: guest RAM with a hole or a range that
ends inside a page, and tasks whose leaves the scenario chooses with the
resource helpers of RTask (docs/testing.md, sections 3.3, 6.10, 6.19).

Page numbers are indices from RAM_BASE, as Scenario.pa and pfn take them;
nothing is taken from the implementation.
"""
from .abi import PAGE
from .resources_build import RTask
from .scenario import RAM_BASE, RAM_SIZE
from .tasks_lib import XGPUTask

# Guest RAM in two ranges: R0, the scenario's own (alloc_page hands out from
# it), then a hole of 16 pages, then R1 of 8 pages.
R0 = (RAM_BASE, RAM_SIZE)
HOLE = RAM_BASE + RAM_SIZE
R1 = (HOLE + 16 * PAGE, 8 * PAGE)
HOLED = [R0, R1]


def page_of(gpa):
    """The page index (from RAM_BASE) of a guest physical address."""
    assert (gpa - RAM_BASE) % PAGE == 0
    return (gpa - RAM_BASE) // PAGE


HOLE_PAGE = page_of(HOLE) + 3  # a page no RAM range holds


class XRTask(XGPUTask):
    """RTask's resource helpers (add, area, shared, blit) over an XGPUTask,
    whose leaves the scenario chooses (tasks_lib.XTask): page 0 the object
    list, pages 1-3 records, 4-7 command buffers, 8.. data."""
    add = RTask.add
    area = RTask.area
    shared = RTask.shared
    blit = RTask.blit


def own_leaves(s, n):
    """{vpn: ram_page} for VA pages 0..n-1, each its own freshly allocated
    page in R0."""
    return {v: s.alloc_page() for v in range(n)}
