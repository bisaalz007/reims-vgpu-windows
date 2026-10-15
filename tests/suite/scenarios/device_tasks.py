"""The device's task table: define, look up and delete through the FIFO
commands (docs/testing.md, sections 6.8 and 6.10).

DeviceCore::create_task_id raises for an id of 256 or more, an id already
defined, an in-process task longer than 0x800000000, and a task whose creation
fails; DeviceCore::get_task_id raises for an unknown id,
DeviceCore::get_user_task_id for id 0, DeviceCore::get_kernel_task when task 0
is not defined; DeviceCore::delete_task_id raises for 256 or more and answers
false for an unknown id, which DeleteTask (32) turns into a fault.
"""
import struct

from pgtest import abi
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import Task


def faulting(s, opcode, payload, label):
    s.submit(opcode, payload, fault=True, label=label)
    s.fault_state(label)
    s.resume_root()
    s.fifo.wait_stamp(s.fifo.stamp)


@scenario
def device_tasks_refusals():
    """DefineTask2 (56) with id 256, 0x7fffffff, an id already defined, and
    length 0 (the task's init fails, "Failed to create host task").
    DeleteTask (32) of 256 and of an id never defined. MapMemory2 (57) of
    an unknown task. A good task is mapped, unmapped and deleted, and then
    deleted again. Each fault
    leaves the root words as they were."""
    s = Scenario("device_tasks_refusals", doc=device_tasks_refusals.__doc__)
    s.bring_up()
    t1 = Task(s, tid=1, pages=2)
    t1.define()
    for tid, label in ((256, "define 256"), (0x7FFFFFFF, "define 0x7fffffff"), (1, "define 1 again")):
        faulting(s, abi.OP_DEFINE_TASK2, struct.pack("<IQI", tid << 1, 0x8000, s.pfn(t1.root)), label)
    faulting(s, abi.OP_DEFINE_TASK2, struct.pack("<IQI", 2 << 1, 0, s.pfn(t1.root)), "define 2 with length 0")
    s.diff("after the define refusals", gpa=s.pa(s.root_page), length=abi.PAGE)
    for tid, label in ((256, "delete 256"), (0xFFFFFFFF, "delete 0xffffffff"), (9, "delete 9, never defined")):
        faulting(s, abi.OP_DELETE_TASK, struct.pack("<I", tid), label)
    faulting(s, abi.OP_MAP_MEMORY2, struct.pack("<IQQ", 9, 0, 0x4000), "map task 9, never defined")
    t1.map()
    t1.unmap()
    t1.delete()
    faulting(s, abi.OP_DELETE_TASK, struct.pack("<I", 1), "delete 1 twice")
    s.events("end")
    s.diff("after", gpa=s.pa(s.root_page), length=abi.PAGE)
    return s


@scenario
def device_tasks_kernel_and_user_lookups():
    """Commands that need the kernel task (task 0) fault while it is not
    defined (DeviceCore::get_kernel_task raises: there is no task 0):
    DeleteIOSurfaceBacking2 (54) with task 0 and DeleteSharedTextureBacking
    (65). Commands that need a user task fault for task 0: ExecIndirect3 (67)
    and GetComputeInfo (59) naming task 0."""
    s = Scenario("device_tasks_kernel_and_user_lookups", doc=device_tasks_kernel_and_user_lookups.__doc__)
    s.bring_up()
    faulting(s, abi.OP_DELETE_IOSURFACE_BACKING2, struct.pack("<II", 5, 0), "54 without task 0")
    faulting(s, abi.OP_DELETE_SHARED_TEXTURE_BACKING, struct.pack("<I", 5), "65 without task 0")
    faulting(s, abi.OP_EXEC_INDIRECT3, struct.pack("<III", 0, 0, 0), "67 on task 0")
    faulting(s, abi.OP_GET_COMPUTE_INFO, struct.pack("<IIIIQ", 0, 1, 4, 4, 0), "59 on task 0")
    s.events("end")
    return s


@scenario
def device_tasks_long():
    """A task longer than 0x800000000 faults its define in every setup:
    in-process (blocks, map) with "Task size too large"; the isolate
    setup's service task is created without that check and fails on its
    own."""
    s = Scenario("device_tasks_long", doc=device_tasks_long.__doc__)
    s.bring_up()
    big = Task(s, tid=3, pages=2)
    s.submit(abi.OP_DEFINE_TASK2, struct.pack("<IQI", 3 << 1, 0x800004000, s.pfn(big.root)), fault=True,
             label="define 3, 0x800004000 bytes")
    s.fault_state("after the define")
    s.events("after the define")
    s.diff("after the define", gpa=s.pa(s.root_page), length=abi.PAGE)
    s.resume_root()
    s.submit(abi.OP_NOP)
    s.events("resumed")
    return s
