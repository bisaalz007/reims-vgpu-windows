"""The object layer common to every resource: creation by type, references
of the wrong kind, deletion, and buffer records
(docs/testing.md, sections 6.11, 6.20).

The resource manager makes a resource by the entry's type, and creates an
object on first reference and caches it."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.resources_build import (RGBA8Unorm, RTask, b_copy_buf_buf, b_copy_tex_buf, b_fill_buf, b_fill_tex_color,
                                    b_sync_resource, buffer_rec, cmd_delete_object, cmd_delete_resource, cpu_buffer_rec,
                                    desc1, simple_dim, texture_rec)
from pgtest.scenario import Scenario
from scenarios.resources import creation_case


@scenario
def resources_object_types():
    """Every object type listed in an exec's resources, one object and one
    exec each: types 0 ("Attempting to fetch invalid object"), 6 (a
    function), 7, 10, 19 (a CPU buffer is not a resource) and 21-255
    ("Unexpected object type (N) for resource"); a reference past the list; a
    buffer and a texture that are made. Type 20 (an indirect command buffer)
    with a short record reaches its creator and fails there."""
    cases = [
        (0, b"", 0, True),
        (abi.T_FUNCTION, bytes(16), None, True),
        (abi.T_SERIALIZED, struct.pack("<II", 3, 0) + bytes(0x20), None, True),
        (10, bytes(16), None, True),
        (abi.T_CPU_BUFFER, cpu_buffer_rec(PAGE, 20), None, True),
        (21, bytes(16), None, True),
        (0x7F, bytes(16), None, True),
        (0xFF, bytes(16), None, True),
        (abi.T_ICB, bytes(8), None, True),
        (abi.T_BUFFER, buffer_rec(PAGE, 20), None, False),
        (abi.T_TEXTURE, texture_rec(PAGE, 21, [simple_dim(4, 4)], desc1()), None, False),
    ]
    out = creation_case("resources_object_types", resources_object_types.__doc__, cases)
    # A reference past the object list, in its own scenario-sized addition.
    return out


@scenario
def resources_buffer_records():
    """Buffer (type 1) records, each its own object listed in its own exec: a
    short record ("Failed to read buffer descriptor"); lengths of one page,
    two pages, 0x100 and 0x4001 bytes (a length that is not a page multiple:
    the memory-map setup and the service refuse the no-copy buffer, the
    blocks setup maps it); a length of 0; a page outside the task; a private
    buffer of 0x100 bytes (made; its paging fails the same way); flags with
    unknown bits (only bits 0 and 1 are read)."""
    cases = [
        (abi.T_BUFFER, buffer_rec(PAGE, 20), 15, True),
        (abi.T_BUFFER, buffer_rec(PAGE, 20), None, False),
        (abi.T_BUFFER, buffer_rec(2 * PAGE, 21), None, False),
        (abi.T_BUFFER, buffer_rec(0x100, 23), None, {"blocks": False, "map": True, "service": True}),
        (abi.T_BUFFER, buffer_rec(PAGE + 1, 24), None, {"blocks": False, "map": True, "service": True}),
        (abi.T_BUFFER, buffer_rec(0, 26), None, True),
        (abi.T_BUFFER, buffer_rec(PAGE, 500), None, True),
        (abi.T_BUFFER, buffer_rec(0x100, 27, 1), None, {"blocks": False, "map": True, "service": True}),
        (abi.T_BUFFER, buffer_rec(PAGE, 28, 0xFC), None, False),
        (abi.T_BUFFER, buffer_rec(PAGE, 29, 0xFD), None, False),
    ]
    return creation_case("resources_buffer_records", resources_buffer_records.__doc__, cases)


@scenario
def resources_wrong_kind():
    """References of the wrong kind in blit records: a texture where a buffer
    is expected (301), a buffer where a texture is expected (302, 321), a
    heap-less object where a heap is expected: each class that lacks the
    getter sends doesNotRecognizeSelector, the decoder faults the exec.
    Reference 0 where a non-null object is needed, and a reference past the
    list ("Failed to create Buffer"). 314 SynchronizeResource takes a
    nullable resource: an unknown reference is nil there, not a fault."""
    s = Scenario("resources_wrong_kind", doc=resources_wrong_kind.__doc__)
    s.bring_up()
    t = RTask(s)
    b, b_va = t.shared(PAGE)
    tex = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc1()))
    t.define()
    t.install()
    for label, recs, res in (
            ("301 from a texture", [b_copy_buf_buf(tex, b, 0, 0, 0x10)], [(tex, 0), (b, 0)]),
            ("302 from a buffer", [b_copy_tex_buf(b, b, 0, 16, 64, (4, 4, 1))], [(b, 0)]),
            ("321 on a buffer", [b_fill_tex_color(b, (1, 1, 1), (1, 0, 0, 1), RGBA8Unorm)], [(b, 0)]),
            ("301 from reference 0", [b_copy_buf_buf(0, b, 0, 0, 0x10)], [(b, 0)]),
            ("301 from a reference past the list", [b_copy_buf_buf(40, b, 0, 0, 0x10)], [(b, 0)]),
            ("302 from a reference past the list", [b_copy_tex_buf(40, b, 0, 16, 64, (4, 4, 1))], [(b, 0)])):
        t.blit(*recs, resources=res, fault=True, label=label)
    t.blit(b_sync_resource(40), b_sync_resource(0), b_sync_resource(tex), b_fill_buf(b, 0, 0x10, 0x3C),
           resources=[(tex, 0), (b, 0)], label="314 on an unknown reference, on 0, on a texture")
    t.dump(b_va, 0x10, "B: the exec ran")
    t.delete()
    s.events("end")
    return s


@scenario
def resources_delete_objects():
    """Deleting resources: 37 (DeleteResource) and 40 (DeleteObject with
    operations 1000, 1001 and 1012) forget the object; the next reference
    makes it again from its record, so a changed record is read afresh. A
    deleted private buffer drops its host copy: its next use pages in the
    guest's bytes. 40 with an operation it does not handle ("Invalid command
    for deletion") or a payload under 12 bytes returns false, which no caller
    reads: the command completes. Every other handled operation (1002, 1003,
    1005-1007, 1009, 1014) on a buffer's reference forgets nothing of the
    buffer."""
    s = Scenario("resources_delete_objects", doc=resources_delete_objects.__doc__)
    s.bring_up()
    t = RTask(s)
    p_va = t.area(PAGE, fill=0x11)
    p = t.add(abi.T_BUFFER, buffer_rec(PAGE, p_va // PAGE, 1))
    alt_va = t.area(PAGE, fill=0x0A)
    b, b_va = t.shared(PAGE)
    rec_va = t.ol.entries[p][2]
    t.define()
    t.install()

    def probe(off, label):
        t.blit(b_copy_buf_buf(p, b, 0, off, 0x10), resources=[(p, 0), (b, 0)], label=label)

    probe(0, "page in the private buffer")
    t.write(p_va, b"\x22" * 0x10)
    probe(0x10, "a guest write, not seen: the host copy is valid")
    cmd_delete_resource(s, 1, p, label="37")
    probe(0x20, "after 37: made again, paged in")
    t.write(p_va, b"\x33" * 0x10)
    for i, op in enumerate((1000, 1001, 1012)):
        t.write(p_va, bytes([0x40 + i]) * 0x10)
        cmd_delete_object(s, 1, op, p, label="40 with operation %d" % op)
        probe(0x30 + 0x10 * i, "after 40/%d" % op)
    for op in (1002, 1003, 1005, 1006, 1007, 1009, 1014):
        cmd_delete_object(s, 1, op, p, label="40 with operation %d" % op)
    t.write(p_va, b"\x55" * 0x10)
    probe(0x60, "after the other operations: the buffer was kept")
    for op in (1004, 1008, 1010, 1011, 1013, 999):
        cmd_delete_object(s, 1, op, p, label="40 with operation %d" % op)
    s.submit(abi.OP_DELETE_OBJECT, struct.pack("<III", 1, 1000, 8), label="40 with an 8-byte payload")
    s.r(abi.R_FAULT_MASK, "no fault")
    t.write(rec_va, buffer_rec(PAGE, alt_va // PAGE, 1))
    t.write(alt_va, b"\x66" * 0x10)
    cmd_delete_resource(s, 1, p, label="37 after the record changed")
    probe(0x70, "the new record's pages")
    cmd_delete_resource(s, 1, 40, label="37 on an unknown reference")
    t.dump(b_va, 0x80, "B")
    t.delete()
    s.events("end")
    return s


@scenario
def textures_invalid_pixel_format():
    """A normal texture of MTLPixelFormatInvalid (0): Metal aborts making it
    (newTextureWithDescriptor:). The abort is inside Metal, so the host dies
    in every setup, an isolated task's included. The scenario's last
    command."""
    s = Scenario("textures_invalid_pixel_format", doc=textures_invalid_pixel_format.__doc__)
    s.bring_up()
    t = RTask(s)
    tex = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc1(fmt=0)))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_buf(b, 0, 0x10, 1), resources=[(tex, 0), (b, 0)], fault=True, label="pixel format 0")
    s.events("end")
    return s
