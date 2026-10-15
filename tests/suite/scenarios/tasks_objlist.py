"""Object and placement lists: SetObjectList (51), SetObjectAndPlacementList
(68), the task root's list at define time, their validation and what they
write into the task root (W7; docs/testing.md, section 6.11), and records
read at the task's very end (section 6.10)."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import Task, buffer_record, seg
from pgtest.tasks_lib import (XGPUTask, XTask, exec3_payload, fault_and_resume,
                              fill_rec, maybe_fault, set_list_raw,
                              set_lists_raw, unmap_raw)


def root_fields(s, t, label):
    """The task root's list fields: object list offset (+0x08) and length
    (+0x10), task length (+0x14), placement list offset (+0x1c) and length
    (+0x24)."""
    s.dump(s.pa(t.root, 8), 0x20, label)


@scenario
def tasks_objlist_set():
    """SetObjectList (51): a valid list is recorded in the task root; one
    that ends at or beyond the task's end is refused, faults, and zeroes the
    task root's list fields (the check is `end >= taskLength`, so a list
    ending exactly at the end is refused); a length that is not a multiple
    of 12 is accepted. The placement fields are never written by opcode 51.
    An unknown task faults."""
    s = Scenario("tasks_objlist_set", doc=tasks_objlist_set.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=8)
    t.define()
    root_fields(s, t, "defined")
    set_list_raw(s, 1, 2, 36)
    root_fields(s, t, "valid list at page 2")
    set_list_raw(s, 1, 7, PAGE, fault=True)
    fault_and_resume(s, "a list ending exactly at the task's end")
    root_fields(s, t, "after the refusal")
    set_list_raw(s, 1, 3, 25)
    root_fields(s, t, "25 bytes")
    set_list_raw(s, 1, 7, PAGE + 1, fault=True)
    fault_and_resume(s, "a list beyond the task's end")
    root_fields(s, t, "after the second refusal")
    set_list_raw(s, 1, 0xFFFFFFFF, 12, fault=True)
    fault_and_resume(s, "a list page far beyond the task")
    set_list_raw(s, 1, 7, PAGE - 1)
    root_fields(s, t, "a list ending one byte before the end")
    set_list_raw(s, 1, 0, 0)
    root_fields(s, t, "an empty list")
    set_list_raw(s, 9, 0, 12, fault=True)
    fault_and_resume(s, "unknown task")
    t.delete()
    s.events("end")
    return s


@scenario
def tasks_objlist_root_list():
    """The task root's object list is taken at define time (by the service in
    its handshake): a valid one stays, one ending at the task's end is zeroed
    in the task root although the define succeeds, and a misaligned offset
    passes (only the previous offsets' alignment is tested, and at define
    time they are 0). With argument buffers the service also takes the
    placement fields. ReplacePhysical of reference 0 after each define orders
    the service's queued work before the observation."""
    s = Scenario("tasks_objlist_root_list", doc=tasks_objlist_root_list.__doc__)
    s.bring_up()
    cases = [
        ("valid", dict(obj_off=PAGE, obj_len=36, place_off=2 * PAGE, place_len=48)),
        ("ending at the task's end", dict(obj_off=7 * PAGE, obj_len=PAGE)),
        ("misaligned", dict(obj_off=PAGE + 4, obj_len=36, place_off=2 * PAGE + 8, place_len=48)),
        ("placement too short", dict(obj_off=PAGE, obj_len=36, place_off=2 * PAGE, place_len=47)),
        ("placement at the end", dict(obj_off=PAGE, obj_len=36, place_off=7 * PAGE, place_len=PAGE)),
    ]
    for label, kw in cases:
        t = Task(s, tid=1, pages=8)
        t.write_root(**kw)
        t.define()
        s.submit(abi.OP_REPLACE_PHYSICAL, struct.pack("<II", 1, 0))
        root_fields(s, t, label)
        t.delete()
    s.events("end")
    return s


@scenario
def tasks_objlist_placement():
    """SetObjectAndPlacementList (68), which needs the argument-buffer
    feature, is accepted in every setup. Every validation the list's setter
    makes: offset + length overflow, a list or placement list ending at the
    task's end, a placement list shorter than 16 bytes an entry, and the
    alignment check (a misaligned offset is accepted, and the call after it
    is refused for it; the refusal zeroes the offsets, so the next call
    passes). Placement offsets are stored only with argument buffers, so a
    misaligned placement list matters only in the service. A placement list
    moved to new pages is copied there (min of both lengths); one over a
    read-only page cannot be mapped, "Unable to do placement list copy". Each
    result shows in the task root."""
    s = Scenario("tasks_objlist_placement", doc=tasks_objlist_placement.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=12, writable={v for v in range(12) if v != 9})
    t.define()
    t.write(4 * PAGE, bytes(range(0x40)))  # the first placement list's contents
    root_fields(s, t, "defined")
    cases = [
        ("valid", (PAGE, 4 * PAGE, 36, 48)),
        ("moved: the placement list is copied", (PAGE, 5 * PAGE, 36, 64)),
        ("object list overflows", ((1 << 64) - PAGE, 5 * PAGE, 36, 48)),
        ("placement list overflows", (PAGE, (1 << 64) - PAGE, 36, 48)),
        ("object list at the end", (11 * PAGE, 5 * PAGE, PAGE, 48)),
        ("placement list at the end", (PAGE, 11 * PAGE, 36, PAGE)),
        ("placement list too short", (PAGE, 5 * PAGE, 36, 47)),
        ("no placement list", (PAGE, 0, 36, 0)),
        ("valid again", (PAGE, 6 * PAGE, 36, 48)),
        ("misaligned object list", (PAGE + 12, 6 * PAGE, 36, 48)),
        ("valid, after a misaligned one", (PAGE, 6 * PAGE, 36, 48)),
        ("valid, after the refusal", (PAGE, 6 * PAGE, 36, 48)),
        ("misaligned placement list", (PAGE, 6 * PAGE + 16, 36, 48)),
        ("valid, after a misaligned placement", (2 * PAGE, 7 * PAGE, 36, 48)),
        ("valid", (2 * PAGE, 7 * PAGE, 36, 48)),
        ("placement over a read-only page", (2 * PAGE, 9 * PAGE, 36, 48)),
        ("valid", (2 * PAGE, 7 * PAGE, 24, 32)),
        ("longer list", (2 * PAGE, 8 * PAGE, 48, 64)),
    ]
    for label, (oo, po, ol, pl) in cases:
        maybe_fault(s, abi.OP_SET_OBJECT_AND_PLACEMENT_LIST,
                    struct.pack("<IQQII", 1, oo, po, ol, pl), label)
        root_fields(s, t, label)
        if label.startswith("moved"):
            t.dump(5 * PAGE, 0x50, "the new placement list")
    unmap_raw(s, 1, 8 * PAGE, PAGE)
    unmap_raw(s, 1, 2 * PAGE, PAGE)
    root_fields(s, t, "after unmaps of both lists")
    for p in (4, 5, 6, 7, 8):
        t.dump(p * PAGE, 0x50, "VA page %d" % p)
    set_lists_raw(s, 2, 0, 0, 0, 0, fault=True)
    fault_and_resume(s, "unknown task")
    t.delete()
    s.events("end")
    return s


@scenario
def tasks_objlist_record_at_end():
    """A buffer record that ends exactly at the task's end: in-process record
    reads copy out of the task, which refuses `end >= taskLength`, so the
    buffer cannot be created and the exec faults; the service reads in place
    and accepts `end == taskLength`, and the fill lands (section 7.1). A
    record one byte shorter works everywhere."""
    s = Scenario("tasks_objlist_record_at_end", doc=tasks_objlist_record_at_end.__doc__)
    s.bring_up()
    t = XGPUTask(s, tid=1, pages=12)
    end = 12 * PAGE
    t.write(end - 16, buffer_record(PAGE, 8))
    t.write(end - 33, buffer_record(PAGE, 9))
    t.ol.add_raw(abi.T_BUFFER, 16, end - 16)
    t.ol.add_raw(abi.T_BUFFER, 16, end - 33)
    t.define()
    t.install()
    for ref, byte in ((1, 0x11), (2, 0x22)):
        t.reset_cmdbufs()
        cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(ref, 0, 0x10, byte)))
        maybe_fault(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(ref, 0)]),
                    "record %d" % ref, timeout_ms=1000)
    t.dump(8 * PAGE, 0x10, "buffer over VA page 8")
    t.dump(9 * PAGE, 0x10, "buffer over VA page 9")
    t.delete()
    s.events("end")
    return s


@scenario
def tasks_objlist_cmdbuf_at_end():
    """A command buffer that ends exactly at the task's end. In-process the
    copy refuses it, the cursor is null, and the segment decoder reads
    through it: the host dies at the exec. The service reads it in place and
    the fill lands (docs/testing.md, sections 6.10 and 7.4)."""
    s = Scenario("tasks_objlist_cmdbuf_at_end", doc=tasks_objlist_cmdbuf_at_end.__doc__)
    s.bring_up()
    t = XGPUTask(s, tid=1, pages=12)
    ref, va = t.buffer(length=PAGE, va=8 * PAGE)
    t.define()
    t.install()
    data = seg(abi.ENC_BLIT, fill_rec(ref, 0, 0x10, 0x77))
    t.write(12 * PAGE - len(data), data)
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [(12 * PAGE - len(data), len(data))], [(ref, 0)]))
    t.dump(8 * PAGE, 0x10, "buffer over VA page 8")
    t.delete()
    s.events("end")
    return s
