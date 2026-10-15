"""The root node page (0x101c) and the root FIFO's creation, its registers
and its teardown (docs/testing.md, sections 6.4 and 6.5).

DeviceCore::set_root_page tears down the previous root
(DeviceCore::teardown_root: FIFOs, tasks, displays, the interrupt bits, and in
the blocks setup the root task), then maps the new page;
DeviceCore::create_root_fifo refuses, with only a log, when there is no root
node, the length is above 0x20000 or not a multiple of 0x4000, the start offset
is 0 or not 0x400-aligned, the mapping fails, or the start offset is not below
the length.
"""
import struct

from pgtest import abi
from pgtest.fifo import ChildFIFO, RootFIFO
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import Task

FIFO_REGS = (abi.R_FIFO_ENABLE, abi.R_FIFO_READ, abi.R_ROOT_PAGE, abi.R_FAULT_MASK, abi.R_VERSION,
             abi.R_FAULT_OFFSET)


def regs(s, label):
    s.sweep(0x1000, 0x1040, label=label)


def root_fifo(s, pages=2, start=0x400):
    f = RootFIFO(s, pages, start)
    s.fifo = f
    s.fifos[0] = f
    return f


@scenario
def device_root_fifo_refusals():
    """Each of DeviceCore::create_root_fifo's six refusals leaves 0x1000
    reading 0 and nothing else changed; after each the guest fixes the one
    register and a NOP runs. The refusals: no root page; length 0x24000 (above
    0x20000); length 0x4400 (not page-sized); length 0; start 0; start 0x200
    (not 0x400-aligned); a base page outside guest RAM and base page 0 (the
    mapping fails); start 0x8000 with length 0x8000 and start 0xc000 with
    length 0x8000 (start not below length). A length of exactly 0x20000 is
    accepted."""
    s = Scenario("device_root_fifo_refusals", doc=device_root_fifo_refusals.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    f = root_fifo(s, pages=8)  # 0x20000 bytes
    good = dict(length=0x8000, start=0x400, base=s.pfn(f.page))
    outside = (s.ram[0][0] + s.ram[0][1]) >> 14

    def attempt(label, **kw):
        args = dict(good)
        args.update(kw)
        f.enable(**args)
        s.r(abi.R_FIFO_ENABLE, label)

    attempt("no root page")
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    for label, kw in (("length 0x24000", dict(length=0x24000)),
                      ("length 0x4400", dict(length=0x4400)),
                      ("length 0", dict(length=0)),
                      ("start 0", dict(start=0)),
                      ("start 0x200", dict(start=0x200)),
                      ("start 0x3ff", dict(start=0x3FF)),
                      ("base page past guest RAM", dict(base=outside)),
                      ("base page 0", dict(base=0)),
                      ("base page 0xffffffff", dict(base=0xFFFFFFFF)),
                      ("start 0x8000, length 0x8000", dict(start=0x8000)),
                      ("start 0xc000, length 0x8000", dict(start=0xC000))):
        attempt(label, **kw)
    s.events("after the refusals")
    regs(s, "after the refusals")
    s.diff("after the refusals")
    # Accepted: the largest length, a start near its end.
    attempt("length 0x20000, start 0x1fc00", length=0x20000, start=0x1FC00)
    f.length, f.start = 0x20000, 0x1FC00
    s.submit(abi.OP_NOP)
    s.submit(abi.OP_NOP)
    s.events("two NOPs on a 0x400-byte ring")
    regs(s, "running")
    s.diff("after")
    return s


@scenario
def device_root_fifo_registers_while_enabled():
    """While the root FIFO is enabled, writes to its length (0x1004), start
    (0x1010) and base page (0x1030) are logged and ignored; a disable and
    re-enable uses the values from before, and only writes made while
    disabled take effect."""
    s = Scenario("device_root_fifo_registers_while_enabled", doc=device_root_fifo_registers_while_enabled.__doc__)
    f = s.bring_up()
    s.submit(abi.OP_NOP)
    other = s.alloc_page(2)
    s.w(abi.R_FIFO_LENGTH, 0x4000)
    s.w(abi.R_FIFO_START, 0x800)
    s.w(abi.R_FIFO_BASE, s.pfn(other))
    s.submit(abi.OP_NOP)
    s.events("two NOPs, the writes between them ignored")
    regs(s, "enabled")
    s.w(abi.R_FIFO_ENABLE, 0)
    s.w(abi.R_FIFO_ENABLE, 1)
    f.written = 0
    s.submit(abi.OP_NOP)
    s.events("re-enabled with the old registers")
    regs(s, "re-enabled")
    s.diff("the old ring and stamp page")
    s.w(abi.R_FIFO_ENABLE, 0)
    s.w(abi.R_FIFO_LENGTH, 0x4000)
    s.w(abi.R_FIFO_START, 0x800)
    s.w(abi.R_FIFO_BASE, s.pfn(other))
    s.w(abi.R_FIFO_ENABLE, 1)
    f.page, f.length, f.start, f.written = other, 0x4000, 0x800, 0
    s.submit(abi.OP_NOP)
    s.events("re-enabled on the new registers")
    regs(s, "moved")
    s.diff("the new ring and stamp page")
    return s


@scenario
def device_root_page_changes():
    """0x101c: reads back what was written; a new page tears the old root
    down (the FIFO is disabled, interrupt and fault bits cleared, 0x103c
    reads 0 with no FIFO), leaves the old page's task words as they were,
    and the FIFO must be enabled again on the new root; writing the same
    page again also tears down; 0 leaves no root, and a FIFO cannot be
    enabled. In the blocks setup each root is a root task: create_task and
    its mappings, destroy_task at teardown."""
    s = Scenario("device_root_page_changes", doc=device_root_page_changes.__doc__)
    s.r(abi.R_ROOT_PAGE, "fresh")
    s.w(abi.R_ROOT_PAGE, 0)
    s.r(abi.R_ROOT_PAGE, "0 with no root")
    s.events("0 with no root")
    f = s.bring_up()
    t = Task(s, tid=3, pages=2)
    t.define()
    # A stamp bit left set, and a fault left standing.
    s.submit(abi.OP_NOP)
    s.submit(99, fault=True)
    s.events("a NOP and a fault")
    s.r(abi.R_FAULT_MASK)
    s.r(abi.R_FAULT_OFFSET)
    s.diff("root word 3 set")
    first = s.root_page
    second = s.alloc_page()
    s.w(abi.R_ROOT_PAGE, s.pfn(second))
    s.events("a new root page")
    regs(s, "on the new root page")
    s.diff("the old root page after the change")
    s.root_page = second
    f.written = 0
    s.w(abi.R_FIFO_ENABLE, 1)
    s.submit(abi.OP_NOP)
    s.events("a NOP on the new root")
    regs(s, "the FIFO on the new root")
    s.w(abi.R_ROOT_PAGE, s.pfn(second))
    s.events("the same page again")
    regs(s, "the same page again")
    s.w(abi.R_ROOT_PAGE, 0)
    s.events("root page 0")
    regs(s, "root page 0")
    s.w(abi.R_FIFO_ENABLE, 1)
    s.r(abi.R_FIFO_ENABLE, "enable without a root")
    s.w(abi.R_ROOT_PAGE, s.pfn(first))
    s.w(abi.R_FIFO_ENABLE, 1)
    s.r(abi.R_FIFO_ENABLE, "enable on the first page again")
    s.events("back on the first page")
    s.diff("after")
    return s


@scenario
def device_root_page_unmappable():
    """A root page the device cannot map (past guest RAM, or 0xffffffff) is
    still what 0x101c reads, but there is no root node: the root FIFO is
    refused. In the blocks setup the root task is created, its mapping
    fails, and it is destroyed at once. A later good page works."""
    s = Scenario("device_root_page_unmappable", doc=device_root_page_unmappable.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    f = root_fifo(s)
    for page in ((s.ram[0][0] + s.ram[0][1]) >> 14, 0xFFFFFFFF):
        s.w(abi.R_ROOT_PAGE, page)
        s.r(abi.R_ROOT_PAGE)
        f.enable()
        s.r(abi.R_FIFO_ENABLE, "enable on an unmapped root")
        s.events("an unmappable root page")
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    s.w(abi.R_FIFO_ENABLE, 1)
    s.submit(abi.OP_NOP)
    s.events("a good page")
    regs(s, "a good page")
    return s


@scenario
def device_root_page_teardown():
    """A root page change with everything live on the old root: child FIFOs
    1 and 2 (the FIFO teardown deletes each, zeroing the next node's page
    list word), a task (its word stays), and a display shared state page on
    port 0 (its word is cleared while running). The new root starts
    empty."""
    s = Scenario("device_root_page_teardown", doc=device_root_page_teardown.__doc__)
    s.bring_up()
    c1 = ChildFIFO(s, 1)
    c2 = ChildFIFO(s, 2)
    c1.setup_node()
    c2.setup_node()
    s.u32(s.pa(s.root_page, abi.ROOT_CHILD_NODE + abi.CHILD_NODE_SIZE * 2 + 0x10), 0xDEADBEEF)
    c1.define()
    c2.define()
    c1.submit(abi.OP_NOP)
    t = Task(s, tid=1, pages=2)
    t.define()
    display_page = s.alloc_page()
    s.submit(abi.OP_DISPLAY_SET_SHARED_STATE_PAGE, struct.pack("<II", 0, s.pfn(display_page)))
    s.events("children, a task and a display page")
    s.diff("the old root in use", gpa=s.pa(s.root_page), length=abi.PAGE)
    s.diff("the rest", gpa=s.pa(s.root_page + 1), length=0x100000)
    new = s.alloc_page()
    s.w(abi.R_ROOT_PAGE, s.pfn(new))
    s.events("the root page changed")
    regs(s, "on the new root")
    s.diff("the old root after the change", gpa=s.pa(s.root_page), length=abi.PAGE)
    s.diff("the rest after the change")
    return s


@scenario
def device_root_fifo_teardown_with_children():
    """0x1000 = 0 with child FIFOs defined tears the children down with the
    root FIFO: each child's running flag goes to 0 and the next node's page
    list word is zeroed; the fault mask is cleared. Re-enabling brings the
    root FIFO back, the children stay gone: a wake of child 1 does
    nothing."""
    s = Scenario("device_root_fifo_teardown_with_children", doc=device_root_fifo_teardown_with_children.__doc__)
    f = s.bring_up()
    c1 = ChildFIFO(s, 1)
    c1.setup_node()
    s.u32(s.pa(s.root_page, abi.ROOT_CHILD_NODE + abi.CHILD_NODE_SIZE + 0x10), 0xDEADBEEF)
    c1.define()
    c1.submit(99, fault=True)
    s.events("child 1 faulted")
    s.r(abi.R_FAULT_MASK)
    s.diff("child 1 defined and faulted", gpa=s.pa(s.root_page), length=abi.PAGE)
    s.w(abi.R_FIFO_ENABLE, 0)
    s.events("disabled")
    regs(s, "disabled")
    s.diff("after the teardown", gpa=s.pa(s.root_page), length=abi.PAGE)
    s.w(abi.R_FIFO_ENABLE, 1)
    f.written = 0
    s.submit(abi.OP_NOP)
    c1.cmd(abi.OP_NOP)
    c1.doorbell()
    s.submit(abi.OP_NOP)
    s.events("re-enabled; child 1 woken")
    regs(s, "re-enabled")
    s.diff("after", gpa=s.pa(s.root_page), length=abi.PAGE)
    return s


@scenario
def device_root_child_fifo_ids():
    """The device's child FIFO table: DefineChildFIFO (48) of an id already
    defined, of a node whose stamp index is at or above maxFIFOCount (64 at
    245; 40 at version 60), and DeleteChildFIFO (49) of ids 0, 64 and one
    never defined, each fault; the table is unchanged."""
    s = Scenario("device_root_child_fifo_ids", doc=device_root_child_fifo_ids.__doc__)
    s.bring_up()

    def faulting(opcode, fid, label):
        s.submit(opcode, struct.pack("<I", fid), fault=True, label=label)
        s.fault_state(label)
        s.resume_root()
        s.fifo.wait_stamp(s.fifo.stamp)

    c1 = ChildFIFO(s, 1)
    c1.setup_node()
    c1.define()
    faulting(abi.OP_DEFINE_CHILD_FIFO, 1, "define 1 again")
    c2 = ChildFIFO(s, 2, stamp_index=64)
    c2.setup_node()
    faulting(abi.OP_DEFINE_CHILD_FIFO, 2, "define 2 with stamp index 64")
    s.w(abi.R_VERSION, 60)
    c3 = ChildFIFO(s, 3, stamp_index=40)
    c3.setup_node()
    faulting(abi.OP_DEFINE_CHILD_FIFO, 3, "define 3 with stamp index 40 at version 60")
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    for fid in (0, 64, 5):
        faulting(abi.OP_DELETE_CHILD_FIFO, fid, "delete %d" % fid)
    c1.submit(abi.OP_NOP)
    s.events("end")
    s.diff("the root page", gpa=s.pa(s.root_page), length=abi.PAGE)
    return s
