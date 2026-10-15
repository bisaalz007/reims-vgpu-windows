"""Tasks over more than one guest RAM range, copied (private) buffers under
the synchronize and discard commands, the task flush at suspend, child FIFO
deletion as the tasks see it, and a task's service dying (docs/testing.md,
sections 4.8, 4.10, 6.10)."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.fifo import ChildFIFO
from pgtest.registry import scenario
from pgtest.scenario import RAM_BASE, RAM_SIZE, Scenario
from pgtest.task import Task, buffer_record, seg
from pgtest.tasks_lib import (XGPUTask, XTask, exec3_payload, fault_and_resume,
                              fill_rec, heap_payload, heap_ser,
                              invalidations_payload, maybe_fault, refs_payload)

RAM1 = RAM_BASE + RAM_SIZE      # a second range, physically adjacent to the first
RAM1_PAGE = RAM_SIZE // PAGE    # its first page, as an index from RAM_BASE


@scenario
def tasks_memory_ranges():
    """Guest RAM in two ranges, the second starting where the first ends. A
    buffer whose two pages are the first range's last page and the second's
    first is physically contiguous across the ranges: the memory map maps it
    in two pieces and the fill lands in both. A leaf naming a page outside
    guest RAM cannot be mapped: in the blocks setup the host's map_memory
    fails ("Failed to map"), elsewhere the range mapping does; either way the
    command faults. The blocks setup's host (as QEMU's) maps one physical
    range per call within one RAM range, so there the buffer across the
    ranges fails too."""
    ram = [(RAM_BASE, RAM_SIZE), (RAM1, 4 * PAGE)]
    s = Scenario("tasks_memory_ranges", ram=ram, doc=tasks_memory_ranges.__doc__)
    s.bring_up()
    outside = (RAM1 + 64 * PAGE - RAM_BASE) // PAGE
    leaves = {v: s.alloc_page() for v in range(8)}
    leaves.update({8: RAM1_PAGE - 1, 9: RAM1_PAGE, 10: outside, 11: RAM1_PAGE + 1})
    t = XGPUTask(s, tid=1, pages=12, leaves=leaves)
    across, _ = t.buffer(length=2 * PAGE, va=8 * PAGE)
    out, _ = t.buffer(length=PAGE, va=10 * PAGE)
    t.define()
    t.install()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(across, PAGE - 0x10, 0x20, 0x6B)))
    maybe_fault(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(across, 0)]),
                "a buffer across the two ranges", timeout_ms=1000)
    s.dump(RAM1 - 0x20, 0x20, "the fill: the first range's end")
    s.dump(RAM1, 0x20, "the fill: the second range's start")
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(out, 0, 0x10, 0x6C)))
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(out, 0)]), fault=True)
    fault_and_resume(s, "a buffer outside guest RAM")
    s.submit(abi.OP_HEAP_TEXTURE_SIZE_AND_ALIGN, heap_payload(1, 10 * PAGE, 16, heap_ser()), fault=True)
    fault_and_resume(s, "a reply outside guest RAM")
    maybe_fault(s, abi.OP_MAP_MEMORY2, struct.pack("<IQQ", 1, 8 * PAGE, 4 * PAGE), "map across the ranges and outside")
    s.dump(RAM1, 0x10, "the second range")
    t.delete()
    s.events("end")
    return s


def private_task(s, tid=1):
    """A task with one private buffer (record flags bit 0) over VA page 8,
    initially 0x11s: the host copies it in and out."""
    t = XGPUTask(s, tid=tid, pages=16)
    ref, va = t.buffer(length=PAGE, flags=1, va=8 * PAGE, contents=b"\x11" * 0x40)
    return t, ref


def fill_exec(s, t, ref, off, byte, flags, **kw):
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(ref, off, 0x10, byte)))
    return s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(t.tid, [cb], [(ref, flags)]), **kw)


@scenario
def tasks_objects_private():
    """A private buffer is a copy, and a synchronize copies it out only when it
    needs one (Resource::needs_sync): guest-invalid and host-valid
    (section 4.10). In turn: a fill with the invalidate-host flag (the guest's
    contents are copied in first); SynchronizeResources (53); the guest
    rewrites its copy and InvalidateResources (52) marks the host copy stale
    before a fill and a synchronize; a fill with the synchronize flag; a fill
    with the invalidate-guest flag, after which SynchronizeAndDiscardResources
    (62) copies out every fill so far; a fill, DiscardResources (63) and 53; 52
    setting host-valid and clearing guest-valid, then 53, which copies again.
    In-process, nothing reaches the guest page before the invalidate-guest
    fill. In the service a private buffer is a Shared detach buffer, so each
    fill lands at once (section 6.10). The guest page is dumped after each."""
    s = Scenario("tasks_objects_private", doc=tasks_objects_private.__doc__)
    s.bring_up()
    t, ref = private_task(s)
    t.define()
    t.install()
    show = lambda label: t.dump(8 * PAGE, 0x40, label)
    fill_exec(s, t, ref, 0, 0xA1, abi.RES_INVALIDATE_HOST)
    show("after a fill with invalidate-host")
    s.submit(abi.OP_SYNCHRONIZE_RESOURCES, refs_payload(1, [ref]))
    show("after SynchronizeResources")
    t.write(8 * PAGE + 0x30, b"\x22" * 0x10)
    s.submit(abi.OP_INVALIDATE_RESOURCES, invalidations_payload(1, [(ref, (1, 0, 0, 1))]))
    fill_exec(s, t, ref, 0x10, 0xA2, 0)
    s.submit(abi.OP_SYNCHRONIZE_RESOURCES, refs_payload(1, [ref]))
    show("after invalidating the host copy, a fill, and a synchronize")
    fill_exec(s, t, ref, 0x20, 0xA3, abi.RES_SYNCHRONIZE)
    show("after a fill with synchronize")
    fill_exec(s, t, ref, 0x00, 0xA4, abi.RES_INVALIDATE_GUEST)
    show("after a fill with invalidate-guest")
    s.submit(abi.OP_SYNCHRONIZE_AND_DISCARD_RESOURCES, refs_payload(1, [ref]))
    show("after SynchronizeAndDiscardResources")
    fill_exec(s, t, ref, 0x10, 0xA5, 0)
    s.submit(abi.OP_DISCARD_RESOURCES, refs_payload(1, [ref]))
    s.submit(abi.OP_SYNCHRONIZE_RESOURCES, refs_payload(1, [ref]))
    show("after a fill, DiscardResources and a synchronize")
    s.submit(abi.OP_INVALIDATE_RESOURCES, invalidations_payload(1, [(ref, (0, 1, 1, 0))]))
    s.submit(abi.OP_SYNCHRONIZE_RESOURCES, refs_payload(1, [ref]))
    show("after setting host valid, clearing guest valid, and a synchronize")
    s.submit(abi.OP_REPLACE_PHYSICAL, struct.pack("<II", 1, ref))
    fill_exec(s, t, ref, 0x30, 0xA6, abi.RES_SYNCHRONIZE)
    show("after ReplacePhysical and a fill with synchronize")
    t.delete()
    s.events("end")
    return s


@scenario
def tasks_flush_on_suspend():
    """rvg_device_finish_suspend flushes every task, in-process or in its
    service, before it writes the blob (section 4.7), but the flush, like a
    synchronize, copies out only resources that needsSync (guest-invalid and
    host-valid). A private buffer filled under an exec without the
    invalidate-guest flag is not one: in-process the guest page keeps its old
    bytes through rvg_device_finish_suspend. In the service the fill landed at
    once (a Shared detach buffer). suspend_writeback locks the flush that does
    copy. The blob reports the host (device info), so the expected file is
    keyed by it."""
    s = Scenario("tasks_flush_on_suspend", host=True, doc=tasks_flush_on_suspend.__doc__)
    s.bring_up()
    t, ref = private_task(s)
    t.define()
    t.install()
    fill_exec(s, t, ref, 0, 0xB1, abi.RES_INVALIDATE_HOST)
    t.dump(8 * PAGE, 0x20, "before the suspend")
    s.api("willSuspend")
    s.api("finishSuspend")
    t.dump(8 * PAGE, 0x20, "after finishSuspend")
    s.events("end")
    return s


@scenario
def tasks_child_fifo_delete():
    """Deleting a child FIFO tells every task: in-process tasks answer at
    once; a service forgets the FIFO's event, and the host clears the FIFO's
    bit, so the next exec from a FIFO with that stamp index sends the event
    again. Execs from a child FIFO fill a buffer before and after."""
    s = Scenario("tasks_child_fifo_delete", doc=tasks_child_fifo_delete.__doc__)
    s.bring_up()
    t = XGPUTask(s, tid=1, pages=16)
    ref, _ = t.buffer(length=PAGE, va=8 * PAGE)
    k = Task(s, tid=0, pages=4)
    t.define()
    k.define()
    t.install()
    stamp = 0
    for n, byte in enumerate((0xC1, 0xC2)):
        f = ChildFIFO(s, 1)
        f.stamp = stamp  # the stamp slot keeps its value across the delete
        f.setup_node()
        f.define()
        t.reset_cmdbufs()
        cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(ref, 0x10 * n, 0x10, byte)))
        f.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(ref, 0)]))
        f.delete()
        stamp = f.stamp
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(ref, 0x20, 0x10, 0xC3)))
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(ref, 0)]))
    t.dump(8 * PAGE, 0x30, "the buffer")
    t.delete()
    k.delete()
    s.events("end")
    return s


@scenario
def tasks_service_crash():
    """A walk of a page table whose root is 0 asserts. In-process that kills
    the host at the command. With isolation only the task dies: the command
    faults, every later command to the task faults (the host's end of the
    task's link completes its tokens with status 4 once the task has died),
    the task can still be deleted, and its ID defined again (section
    4.8)."""
    s = Scenario("tasks_service_crash", doc=tasks_service_crash.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=4)
    t.write_root(pt_root=0)
    t.define()
    s.submit(abi.OP_HEAP_TEXTURE_SIZE_AND_ALIGN, heap_payload(1, 0x40, 16, heap_ser()), fault=True)
    fault_and_resume(s, "the walk")
    s.submit(abi.OP_REPLACE_PHYSICAL, struct.pack("<II", 1, 0), fault=True)
    fault_and_resume(s, "a command after the crash")
    s.submit(abi.OP_SET_OBJECT_LIST, struct.pack("<III", 1, 0, 12), fault=True)
    fault_and_resume(s, "another")
    t.delete()
    t.write_root()
    t.define()
    s.submit(abi.OP_REPLACE_PHYSICAL, struct.pack("<II", 1, 0))
    s.diff("defined again")
    t.delete()
    s.events("end")
    return s
