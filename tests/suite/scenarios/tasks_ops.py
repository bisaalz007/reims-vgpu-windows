"""The task commands that act on objects by reference: HeapTextureSizeAndAlign
(64), ReplacePhysical (60), InvalidateResources (52), SynchronizeResources
(53), SynchronizeAndDiscardResources (62), DiscardResources (63),
DeleteResource (37), DeleteObject (40), DeleteIOSurfaceBacking2 (54),
DeleteSharedTextureBacking (65), and the refusals of GetComputeInfo (59),
ResetRasterizationRateMap (66) and InfoIndirect (69) that the task decides
(docs/testing.md, sections 6.8, 6.10, 6.11)."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import Task, buffer_record, seg
from pgtest.tasks_lib import (XGPUTask, XTask, exec3_payload, fault_and_resume,
                              fill_rec, heap_metal, heap_payload, heap_ser,
                              invalidations_payload, maybe_fault, refs_payload,
                              set_list_raw, tex_desc)

OP64 = abi.OP_HEAP_TEXTURE_SIZE_AND_ALIGN


@scenario
def tasks_heap_refusals():
    """HeapTextureSizeAndAlign (64) refused, with nothing written: a reply
    length under 16 or above 0x3fff ("Invalid reply length"), a serializer
    payload under 8 bytes, under 0x28 bytes, or not command 22 ("Invalid
    command size"), a payload length beyond the command or one that
    overflows, task 0 ("Invalid task ID for user task"), an unknown task, a
    reply beyond the task, a reply over a page-table hole."""
    s = Scenario("tasks_heap_refusals", doc=tasks_heap_refusals.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=8, holes={6})
    t.define()
    t.map(0, 4 * PAGE)
    ser = heap_ser()
    cases = [
        ("reply length 15", heap_payload(1, 0x40, 15, ser)),
        ("reply length 0x4000", heap_payload(1, 0x40, 0x4000, ser)),
        ("payload of 4 bytes", heap_payload(1, 0x40, 16, ser[:4])),
        ("payload of 0x27 bytes", heap_payload(1, 0x40, 16, ser[:0x27])),
        ("command 21", heap_payload(1, 0x40, 16, heap_ser(cmd=21))),
        ("payload length beyond the command", heap_payload(1, 0x40, 16, ser, payload_len=len(ser) + 1)),
        ("payload length overflows", heap_payload(1, 0x40, 16, ser, payload_len=0xFFFFFFFF)),
        ("task 0", heap_payload(0, 0x40, 16, ser)),
        ("unknown task", heap_payload(5, 0x40, 16, ser)),
        ("reply beyond the task", heap_payload(1, 8 * PAGE - 8, 16, ser)),
        ("reply over a hole", heap_payload(1, 6 * PAGE, 16, ser)),
    ]
    for label, p in cases:
        s.submit(OP64, p, fault=True, label=label)
        fault_and_resume(s, label)
    t.dump(0x40, 0x10, "reply area, untouched")
    t.delete()
    s.events("end")
    return s


@scenario
def tasks_heap_success():
    """HeapTextureSizeAndAlign (64) answered: the reply, {u64 size, u64
    align}, is written at the reply offset. The test's own Metal query is
    recorded beside it, with allowGPUOptimizedContents at Metal's default
    (YES); the guest's descriptor clears that bit, and the two differ
    (section 6.10). A reply of 16 bytes and one of 0x3fff straddling two
    pages; a longer payload than the descriptor needs. Over a read-only page
    the reply is refused where the task maps with writability (map: "Failed
    to make mapping"; the service), and written anyway in the blocks setup,
    whose mapping ignores it. Below version 43 the command is unsupported."""
    s = Scenario("tasks_heap_success", host=True, doc=tasks_heap_success.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=8, writable={0, 1, 2, 3, 4, 5, 7})
    t.define()
    s.submit(OP64, heap_payload(1, 0x40, 16, heap_ser()))
    t.dump(0x40, 0x10, "256x256 RGBA8Unorm, private")
    heap_metal(s, label="Metal, 256x256 RGBA8Unorm")
    d2 = tex_desc(fmt=80, w=512, h=128, usage=5, options=0x20)
    s.submit(OP64, heap_payload(1, 2 * PAGE - 8, 0x3FFF, heap_ser(d2, pad=b"\xee" * 24)))
    t.dump(2 * PAGE - 8, 0x10, "512x128 BGRA8Unorm across a page boundary")
    heap_metal(s, fmt=80, w=512, h=128, usage=5, label="Metal, 512x128 BGRA8Unorm")
    maybe_fault(s, OP64, heap_payload(1, 6 * PAGE + 0x100, 16, heap_ser()), "reply over a read-only page")
    t.dump(6 * PAGE + 0x100, 0x10, "the read-only page")
    t.delete()
    s.events("end")
    return s


@scenario
def tasks_heap_version():
    """HeapTextureSizeAndAlign (64) is gated on the metalHeaps feature,
    binary version 43 (section 6.8): version 42 refuses it before looking
    at the payload."""
    s = Scenario("tasks_heap_version", doc=tasks_heap_version.__doc__)
    s.bring_up(version=42)
    t = Task(s, tid=1, pages=4)
    t.define()
    s.submit(OP64, heap_payload(1, 0x40, 16, heap_ser()), fault=True)
    fault_and_resume(s, "version 42")
    t.dump(0x40, 0x10, "reply area, untouched")
    t.delete()
    s.events("end")
    return s


class Probe:
    """A task with one buffer object, reference 1, whose record the scenario
    can point at another page: objects are created once and cached by
    reference, so after a command that drops the object, the next use reads
    the record again and the fill lands in the new page."""

    def __init__(self, s, tid=1):
        self.s = s
        self.t = XGPUTask(s, tid=tid, pages=40)
        self.ref, _ = self.t.buffer(length=PAGE, va=8 * PAGE)
        self.rec_va = self.t.ol.entries[self.ref][2]
        self.page = 8
        self.byte = 0x30

    def fill(self, label):
        """Fills 16 bytes of whatever reference 1 now is."""
        t = self.t
        t.reset_cmdbufs()
        self.byte += 1
        cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(self.ref, 0, 0x10, self.byte)))
        self.s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(t.tid, [cb], [(self.ref, 0)]), label=label)

    def repoint(self):
        """Points reference 1's record at the next page."""
        self.page += 1
        self.t.write(self.rec_va, buffer_record(PAGE, self.page))

    def show(self, label):
        for p in range(8, self.page + 1):
            self.t.dump(p * PAGE, 0x10, "%s: VA page %d" % (label, p))


def drop_case(s, pr, label, opcode, payload):
    """Fill, repoint the record, run the command, fill again, then diff the
    task's pages: the second fill lands where the first did if the command
    left the object cached, in the record's new page if it dropped it."""
    pr.fill(label + ": before")
    pr.repoint()
    s.submit(opcode, payload, label=label)
    pr.fill(label + ": after")
    pr.t.diff(label + ": where the two fills landed")


@scenario
def tasks_objects_drop():
    """Which commands drop a buffer object: DeleteResource (37), DeleteObject
    (40) with operations 1000, 1001 and 1012; and which leave it:
    DiscardResources (63) and SynchronizeAndDiscardResources (62) (a shared
    buffer supports discard, so it is discarded, not deleted),
    ReplacePhysical (60), SynchronizeResources (53), InvalidateResources
    (52), DeleteObject with the operations of other object kinds (1002, 1003,
    1005, 1006, 1007, 1009, 1014), the unhandled ones (1004, 1008, 1010,
    1011, 1013), an invalid one, and one whose payload is under 12 bytes.
    Commands on references never created, or 0, are no-ops."""
    s = Scenario("tasks_objects_drop", doc=tasks_objects_drop.__doc__)
    s.bring_up()
    pr = Probe(s)
    pr.t.define()
    pr.t.install()
    pr.t.diff("before")
    tp = struct.pack

    def delobj(op, ref=1, length=12, extra=b""):
        return tp("<IIII", 1, op, length, ref) + extra

    cases = [
        ("DeleteResource", abi.OP_DELETE_RESOURCE, tp("<II", 1, 1)),
        ("DeleteObject 1000", abi.OP_DELETE_OBJECT, delobj(1000)),
        ("DeleteObject 1001", abi.OP_DELETE_OBJECT, delobj(1001)),
        ("DeleteObject 1012", abi.OP_DELETE_OBJECT, delobj(1012)),
        ("DiscardResources", abi.OP_DISCARD_RESOURCES, refs_payload(1, [1])),
        ("SynchronizeAndDiscardResources", abi.OP_SYNCHRONIZE_AND_DISCARD_RESOURCES, refs_payload(1, [0, 1, 7])),
        ("ReplacePhysical", abi.OP_REPLACE_PHYSICAL, tp("<II", 1, 1)),
        ("SynchronizeResources", abi.OP_SYNCHRONIZE_RESOURCES, refs_payload(1, [1, 0, 7])),
        ("InvalidateResources", abi.OP_INVALIDATE_RESOURCES,
         invalidations_payload(1, [(1, (1, 0, 1, 0)), (1, (0, 1, 0, 1)), (7, (1, 1, 1, 1)), (0, (1, 1, 1, 1))])),
    ]
    for op in (1002, 1003, 1005, 1006, 1007, 1009, 1014, 1004, 1008, 1010, 1011, 1013, 999):
        cases.append(("DeleteObject %d" % op, abi.OP_DELETE_OBJECT, delobj(op)))
    cases.append(("DeleteObject 1000, 8 bytes", abi.OP_DELETE_OBJECT, tp("<III", 1, 1000, 0)))
    for label, op, p in cases:
        drop_case(s, pr, label, op, p)
    for label, op, p in [
        ("DeleteResource 0", abi.OP_DELETE_RESOURCE, tp("<II", 1, 0)),
        ("DeleteResource 7", abi.OP_DELETE_RESOURCE, tp("<II", 1, 7)),
        ("ReplacePhysical 7", abi.OP_REPLACE_PHYSICAL, tp("<II", 1, 7)),
        ("DiscardResources 0 and 7", abi.OP_DISCARD_RESOURCES, refs_payload(1, [0, 7])),
        ("DeleteObject 1000, extra payload", abi.OP_DELETE_OBJECT, delobj(1000, ref=7, extra=b"\xff" * 8)),
    ]:
        s.submit(op, p, label=label)
    pr.t.delete()
    s.events("end")
    return s


@scenario
def tasks_objects_refusals():
    """The object commands' refusals: an unknown task for each
    (DeleteResource 37, DeleteObject 40, ReplacePhysical 60,
    InvalidateResources 52, SynchronizeResources 53,
    SynchronizeAndDiscardResources 62, DiscardResources 63); a count whose
    list does not fit the command, and one whose size overflows ("Overflow in
    Cmd..."); DeleteObject with a length beyond the payload or one that
    overflows."""
    s = Scenario("tasks_objects_refusals", doc=tasks_objects_refusals.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=4)
    t.define()
    tp = struct.pack
    cases = [
        ("DeleteResource, unknown task", abi.OP_DELETE_RESOURCE, tp("<II", 3, 1)),
        ("DeleteObject, unknown task", abi.OP_DELETE_OBJECT, tp("<IIII", 3, 1000, 12, 1)),
        ("ReplacePhysical, unknown task", abi.OP_REPLACE_PHYSICAL, tp("<II", 3, 1)),
        ("InvalidateResources, unknown task", abi.OP_INVALIDATE_RESOURCES, invalidations_payload(3, [(1, (1, 0, 0, 0))])),
        ("SynchronizeResources, unknown task", abi.OP_SYNCHRONIZE_RESOURCES, refs_payload(3, [1])),
        ("SynchronizeAndDiscardResources, unknown task", abi.OP_SYNCHRONIZE_AND_DISCARD_RESOURCES, refs_payload(3, [1])),
        ("DiscardResources, unknown task", abi.OP_DISCARD_RESOURCES, refs_payload(3, [1])),
        ("InvalidateResources, count beyond the command", abi.OP_INVALIDATE_RESOURCES, tp("<II", 1, 2) + b"\0" * 8),
        ("InvalidateResources, count overflows", abi.OP_INVALIDATE_RESOURCES, tp("<II", 1, 0x20000000)),
        ("SynchronizeResources, count beyond the command", abi.OP_SYNCHRONIZE_RESOURCES, tp("<III", 1, 2, 1)),
        ("SynchronizeResources, count overflows", abi.OP_SYNCHRONIZE_RESOURCES, tp("<II", 1, 0x40000000)),
        ("SynchronizeAndDiscardResources, count overflows", abi.OP_SYNCHRONIZE_AND_DISCARD_RESOURCES, tp("<II", 1, 0x40000000)),
        ("DiscardResources, count beyond the command", abi.OP_DISCARD_RESOURCES, tp("<III", 1, 2, 1)),
        ("DiscardResources, count overflows", abi.OP_DISCARD_RESOURCES, tp("<II", 1, 0x40000000)),
        ("DeleteObject, length beyond the payload", abi.OP_DELETE_OBJECT, tp("<IIII", 1, 1000, 13, 1)),
        ("DeleteObject, length overflows", abi.OP_DELETE_OBJECT, tp("<IIII", 1, 1000, 0xFFFFFFFD, 1)),
    ]
    for label, op, p in cases:
        s.submit(op, p, fault=True, label=label)
        fault_and_resume(s, label)
    t.delete()
    s.events("end")
    return s


@scenario
def tasks_kernel_backings():
    """DeleteIOSurfaceBacking2 (54) and DeleteSharedTextureBacking (65) act
    on the kernel task, task 0: a task ID other than 0 in 54 is refused
    ("This command may only reference a kernel task"); with no kernel task
    both fault ("Invalid task"); with a kernel task whose object list cannot
    be mapped (length 0), 54 faults ("Failed to map object list") and 65
    completes; with a list, both complete, for references never created
    too."""
    s = Scenario("tasks_kernel_backings", doc=tasks_kernel_backings.__doc__)
    s.bring_up()
    tp = struct.pack
    s.submit(abi.OP_DELETE_IOSURFACE_BACKING2, tp("<II", 1, 1), fault=True)
    fault_and_resume(s, "54, task 1")
    s.submit(abi.OP_DELETE_IOSURFACE_BACKING2, tp("<II", 1, 0), fault=True)
    fault_and_resume(s, "54, no kernel task")
    s.submit(abi.OP_DELETE_SHARED_TEXTURE_BACKING, tp("<I", 1), fault=True)
    fault_and_resume(s, "65, no kernel task")
    k = Task(s, tid=0, pages=4)
    k.define()
    s.submit(abi.OP_DELETE_IOSURFACE_BACKING2, tp("<II", 1, 0), fault=True)
    fault_and_resume(s, "54, kernel task without a list")
    s.submit(abi.OP_DELETE_SHARED_TEXTURE_BACKING, tp("<I", 1))
    set_list_raw(s, 0, 1, 36)
    s.submit(abi.OP_DELETE_IOSURFACE_BACKING2, tp("<II", 1, 0))
    s.submit(abi.OP_DELETE_IOSURFACE_BACKING2, tp("<II", 0, 0))
    s.submit(abi.OP_DELETE_SHARED_TEXTURE_BACKING, tp("<I", 2))
    k.delete()
    s.events("end")
    return s


@scenario
def tasks_task_refusals():
    """Refusals the task itself decides for commands whose success belongs to
    other areas: an exec on a task with no object list (the list cannot be
    mapped: in-process status 2; the service checks the list as well),
    InfoIndirect (69) likewise; GetComputeInfo (59) with no list, with a
    reference that is not a compute pipeline ("Failed to get pipeline"), or
    for task 0; ResetRasterizationRateMap (66) with reference 0 ("Invalid
    rate map reference"), and with a reference never created (in-process:
    fault; the service: success, section 7.1)."""
    s = Scenario("tasks_task_refusals", doc=tasks_task_refusals.__doc__)
    s.bring_up()
    t = XGPUTask(s, tid=1, pages=16)
    ref, _ = t.buffer(length=PAGE, va=8 * PAGE)
    t.define()
    tp = struct.pack
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1), fault=True)
    fault_and_resume(s, "exec, no object list")
    s.submit(abi.OP_INFO_INDIRECT, exec3_payload(1), fault=True)
    fault_and_resume(s, "info, no object list")
    s.submit(abi.OP_GET_COMPUTE_INFO, tp("<IIIIQ", 1, ref, 5, 4, 0x100), fault=True)
    fault_and_resume(s, "compute info, no object list")
    s.submit(abi.OP_RESET_RASTERIZATION_RATE_MAP, tp("<II", 1, 1), fault=True)
    fault_and_resume(s, "rate map, no object list")
    t.install()
    s.submit(abi.OP_GET_COMPUTE_INFO, tp("<IIIIQ", 1, ref, 5, 4, 0x100), fault=True)
    fault_and_resume(s, "compute info, a buffer reference")
    s.submit(abi.OP_GET_COMPUTE_INFO, tp("<IIIIQ", 0, ref, 5, 4, 0x100), fault=True)
    fault_and_resume(s, "compute info, task 0")
    s.submit(abi.OP_RESET_RASTERIZATION_RATE_MAP, tp("<II", 1, 0), fault=True)
    fault_and_resume(s, "rate map 0")
    maybe_fault(s, abi.OP_RESET_RASTERIZATION_RATE_MAP, tp("<II", 1, ref), "rate map never created")
    s.submit(abi.OP_INFO_INDIRECT, exec3_payload(1))
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1))
    t.dump(0x100, 0x20, "compute info reply area, untouched")
    t.delete()
    s.events("end")
    return s
