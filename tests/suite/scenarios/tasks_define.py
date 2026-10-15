"""DefineTask2 (56) and DeleteTask (32): every refusal, the root node word
and the task root's length (W4, W6), and each host setup's task
implementation (docs/testing.md, sections 6.10 and 7.2)."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import Task
from pgtest.tasks_lib import (XTask, define_raw, delete_raw, fault_and_resume,
                              unmap_raw)

REPLACE_PHYSICAL = abi.OP_REPLACE_PHYSICAL


def barrier(s, tid):
    """ReplacePhysical (60) of reference 0: a command every setup completes
    with no effect (reference 0 names no host resource). In the isolate setup
    it runs on the service's serial queue, so once its stamp is visible,
    every earlier piece of work the service queued for the task (the define's
    object list) has run."""
    s.submit(REPLACE_PHYSICAL, struct.pack("<II", tid, 0))


@scenario
def tasks_define_lifecycle():
    """Define, redefine (refused: "Host task already exists"), delete, delete
    again (refused: "Failed to delete task"), and define the same ID again;
    the highest ID 255; IDs 256 and above refused by define and by delete;
    bit 0 of word 0 ignored. Locks W4 (root word[taskID]), W6 (task root
    +0x14) and, in the blocks setup, the create_task/destroy_task calls in
    order."""
    s = Scenario("tasks_define_lifecycle", doc=tasks_define_lifecycle.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=8)
    s.diff("bring-up")
    t.define()
    s.diff("task 1 defined")
    t.define(fault=True)
    fault_and_resume(s, "task 1 defined twice")
    s.diff("after the refused redefinition")
    t.delete()
    s.diff("task 1 deleted")
    t.delete(fault=True)
    fault_and_resume(s, "task 1 deleted twice")
    t.define()
    s.diff("task 1 defined again")
    s.events("lifecycle of task 1")

    hi = Task(s, tid=255, pages=4)
    hi.define()
    s.diff("task 255 defined")
    hi.delete()
    s.diff("task 255 deleted")

    t256 = Task(s, tid=256, pages=4)
    t256.define(fault=True)
    fault_and_resume(s, "task 256")
    define_raw(s, 0xFFFFFFFE, 4 * PAGE, s.pfn(t256.root), fault=True)
    fault_and_resume(s, "task 0x7fffffff")
    delete_raw(s, 256, fault=True)
    fault_and_resume(s, "delete task 256")
    delete_raw(s, 7, fault=True)
    fault_and_resume(s, "delete task 7, never defined")
    s.diff("after the refused IDs")

    b = Task(s, tid=2, pages=4)
    b.define(bit0=1)
    s.diff("task 2 defined with bit 0 set")
    b.delete()
    t.delete()
    s.diff("tasks 2 and 1 deleted")
    s.events("end")
    return s


@scenario
def tasks_define_refusals():
    """Each define refusal in turn, and what it leaves in guest memory
    (nothing): length 0 ("Task Length Zero"), page-table depth 0, 4 and
    0xffffffff and a length beyond the table's coverage ("Invalid Page Table
    Depth"), a task root outside guest RAM (the mapping fails). Then the
    edges that pass: a length of exactly 4096 pages at depth 1, and a length
    that is not a whole page. In the blocks setup the refused tasks still
    call create_task, then destroy_task."""
    s = Scenario("tasks_define_refusals", doc=tasks_define_refusals.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=4)
    s.diff("bring-up")
    t.define(length=0, fault=True)
    fault_and_resume(s, "length 0")
    for depth in (0, 4, 0xFFFFFFFF):
        t.write_root(depth=depth)
        t.define(fault=True)
        fault_and_resume(s, "depth %#x" % depth)
    t.write_root()
    t.define(length=4096 * PAGE + PAGE, fault=True)
    fault_and_resume(s, "one page beyond depth 1's coverage")
    t.define(length=4096 * PAGE + 1, fault=True)
    fault_and_resume(s, "one byte beyond depth 1's coverage")
    outside = s.pfn(0) + (0x10000000 >> 14)  # 256 MiB past RAM's base: beyond the 64 MiB of RAM
    t.define(root=outside, fault=True)
    fault_and_resume(s, "task root outside guest RAM")
    s.diff("after the refusals")
    s.events("refusals")
    t.define(length=4096 * PAGE)
    s.diff("depth 1, exactly its coverage")
    t.delete()
    t.define(length=3 * PAGE + 0x123)
    s.diff("a length that is not a whole page")
    t.delete()
    s.diff("deleted")
    s.events("end")
    return s


@scenario
def tasks_define_too_large():
    """A task longer than 32 GiB (0x800000000): refused in-process ("Task
    size too large"); the service allows it up to the page table's coverage
    (docs/testing.md, section 6.10's divergence table). Exactly 32 GiB passes
    everywhere. Depth 3 covers both."""
    s = Scenario("tasks_define_too_large", doc=tasks_define_too_large.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=4, depth=3)
    t.define(length=0x800000000)
    s.diff("exactly 32 GiB")
    t.delete()
    st = t.define(length=0x800000000 + PAGE, fault=True, timeout_ms=1500)
    s.fault_state("32 GiB and a page")
    s.resume_root()
    s.fifo.wait_stamp(st)
    s.diff("32 GiB and a page")
    s.events("end")
    return s


@scenario
def tasks_define_root_zero():
    """A task root page of 0. In-process, the task's setup skips the task
    root and then reads the object list from a NULL task root: the host dies
    at the define. The service refuses it ("Task Root setup failed"), the
    define faults, and the ID stays free."""
    s = Scenario("tasks_define_root_zero", doc=tasks_define_root_zero.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=4)
    t.define(root=0, fault=True)
    fault_and_resume(s, "task root 0")
    s.diff("after the refused define")
    t.define()
    s.diff("defined properly")
    t.delete()
    s.events("end")
    return s


@scenario
def tasks_define_kernel_task():
    """Task 0, the kernel task, is an in-process task in every setup. It is
    defined, used by a map and an unmap, and deleted like any other; its
    object list is taken from its task root at define time."""
    s = Scenario("tasks_define_kernel_task", doc=tasks_define_kernel_task.__doc__)
    s.bring_up()
    t = Task(s, tid=0, pages=4)
    t.write_root(obj_off=PAGE, obj_len=24)
    t.define()
    s.diff("task 0 defined")
    t.map()
    unmap_raw(s, 0, 0, 4 * PAGE)
    s.diff("mapped and unmapped")
    t.delete()
    s.diff("task 0 deleted")
    s.events("end")
    return s
