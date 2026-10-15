"""The service, the tasks, the page-table walk and the FIFO's display
commands at their edges: each scenario locks one behaviour, as its docstring
says (docs/testing.md, sections 4.10, 6.10-6.13, 6.21)."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.display_guest import (SS_COLOR_MATRICES, DisplayGuest, display, p_glyph)
from pgtest.exec_objects import ObjTask
from pgtest.registry import scenario
from pgtest.resources_build import (RGBA8Unorm, b_copy_buf_buf, b_fill_buf, b_fill_tex_color, buffer_rec, desc1,
                                    simple_dim, texture_rec)
from pgtest.scenario import Scenario
from pgtest.task import rec, seg
from pgtest.tasks_lib import (XGPUTask, XTask, exec3_payload, fill_rec, heap_payload, heap_ser,
                              invalidations_payload, maybe_fault, refs_payload)

PATTERN = bytes((i * 7 + 3) & 0xFF for i in range(0x100))


def settle(s, st, label):
    """A command that completes or faults, whichever the setup does: waits
    for its stamp or the fault, records the fault state, resumes (a no-op
    when nothing faulted) and waits for the stamp."""
    s.wait_any([dict(gpa=s.fifo.stamp_gpa(), ge=st), dict(mmio=abi.R_FAULT_MASK, mmio_eq=1)],
               label=label)
    s.fault_state(label)
    s.resume_root()
    s.fifo.wait_stamp(st, label="stamp after " + label)


def either(s, opcode, payload, label):
    st = s.cmd(opcode, payload)
    s.doorbell()
    settle(s, st, label)
    return st


# ---- GetComputeInfo's reply mapping ----

@scenario(host=True)
def mutgap4_svc_compute_info_read_only_then_writable():
    """GetComputeInfo (59) with its reply on a read-only leaf, then the same
    query with the reply on a writable page. The service asks for a writable
    mapping (which throws "Failed to make mapping"): the first command faults
    and the service lives, so the second writes its entries. In-process the
    map setup faults the same way and the blocks setup writes the first reply
    through the host task's mapping (compute_info_read_only_reply). Keyed by
    the host: the entries are the host GPU's limits."""
    s = Scenario("mutgap4_svc_compute_info_read_only_then_writable", host=True,
                 doc=mutgap4_svc_compute_info_read_only_then_writable.__doc__)
    s.bring_up()
    from pgtest.compute_objects import compute_blob, compute_pipeline_record, function_record
    from pgtest.task import GPUTask
    t = GPUTask(s, pages=64, writable=set(range(64)) - {40})
    va = t.reserve(0x2000)
    n = s.mem_file(t.gpa(va), "shaders/kernel.metallib")
    fn = t.ol.add(abi.T_FUNCTION, function_record(va, n))
    pso = t.ol.add(abi.T_SERIALIZED, compute_pipeline_record(compute_blob(fn)))
    t.fill(40 * PAGE, 0x40, 0xee)
    t.fill(41 * PAGE, 0x40, 0xee)
    t.define()
    t.install()
    either(s, abi.OP_GET_COMPUTE_INFO, struct.pack("<IIIIQ", t.tid, pso, 5, 4, 40 * PAGE), "the read-only reply")
    t.dump(40 * PAGE, 0x40, "the read-only reply")
    either(s, abi.OP_GET_COMPUTE_INFO, struct.pack("<IIIIQ", t.tid, pso, 5, 4, 41 * PAGE), "a writable reply next")
    t.dump(41 * PAGE, 0x40, "the writable reply")
    t.delete()
    s.events("end")
    return s


# ---- ResetRasterizationRateMap when the lists cannot be mapped ----

@scenario(setups=("isolate",), host=True)
def mutgap4_svc_rate_map_reset_placement_unmappable():
    """ResetRasterizationRateMap (66) while the task's placement list is on a
    read-only leaf. A mutable rate map (type 7, 256x256, samples {0.25, 1.0})
    is made by an info 454 query, which maps (200, 200) to physical
    coordinates; the guest then writes other samples into its record ({1.0,
    0.25}) and moves the lists with SetObjectAndPlacementList (68): the
    object list where it was, the placement list on VA page 40, read-only. 66
    must map both lists first: the object list maps, the placement list needs
    a writable mapping and does not, so 66 replies 2 and returns with the map
    untouched: the command faults. With the placement list moved to a
    writable page (68 again), 454 still maps (200, 200) as before: the new
    samples were never read. Isolate only (placement lists and 66 are the
    service's). Keyed by the host: 454's coordinates are Metal's."""
    from pgtest.exec_objects import rate_map_record
    s = Scenario("mutgap4_svc_rate_map_reset_placement_unmappable", setups=("isolate",), host=True,
                 doc=mutgap4_svc_rate_map_reset_placement_unmappable.__doc__)
    s.bring_up()
    t = ObjTask(s, pages=64, writable=set(range(64)) - {40})
    reply, reply_va = t.buffer(PAGE)
    rec_m = rate_map_record(w=256, h=256, mutable=1, min_factor=0.25, samples=[([0.25, 1.0], [0.25, 1.0])])
    m = t.ol.add(abi.T_SERIALIZED, rec_m)
    m_va = t.ol.entries[m][2]
    good = t.reserve(PAGE)
    t.define()
    t.install()

    def query(off, label):
        t.exec_segments([seg(abi.ENC_INFO, rec(454, struct.pack("<IIQIff", m, reply, off, 0, 200.0, 200.0)))],
                        label=label)
        s.fault_state(label)
        t.dump(reply_va + off, 8, label)

    query(0, "454 makes the map")
    body = bytearray(rec_m)
    struct.pack_into("<4f", body, 0x1c + 4, 1.0, 0.25, 1.0, 0.25)
    t.write(m_va, bytes(body))
    n = len(t.ol.entries)
    either(s, abi.OP_SET_OBJECT_AND_PLACEMENT_LIST, struct.pack("<IQQII", 1, 0, 40 * PAGE, t.ol.length(), 16 * n),
           "68: the placement list on a read-only leaf")
    either(s, abi.OP_RESET_RASTERIZATION_RATE_MAP, struct.pack("<II", 1, m), "66, the placement list unmappable")
    either(s, abi.OP_SET_OBJECT_AND_PLACEMENT_LIST, struct.pack("<IQQII", 1, 0, good, t.ol.length(), 16 * n),
           "68: the placement list on a writable page")
    query(8, "454 after")
    t.delete()
    s.events("end")
    return s


# ---- an exec's completion reads exactly the listed resources ----

FAR_VPN = 0x100000  # VA 0x4_0000_0000: bit 34 of a VA is bit 2 of the high word


@scenario
def mutgap4_svc_complete_reads_listed_only():
    """An exec's completion walks its resource list, count entries of {u32
    reference, u32 flags}, and the first command buffer's {u64 VA, u64
    length} follows the list: here that VA is 0x4_0000_0000 + X, read as a
    record it would name texture X with the synchronize flag (bit 2 of its
    high word). A depth-2 task of 0x100001 pages (16 GiB and a page) with a
    leaf at VA page 0x100000. A normal texture X (type 2, 4x4 RGBA8Unorm,
    blit-paged) is filled on the GPU by an exec that lists it with
    invalidateGuest only, so its host copy is newer; the guest writes 0xB7
    over its pages. A second exec, whose command buffer is at 0x4_0000_0000 +
    X, lists only a shared buffer S with synchronize: nothing synchronizes X,
    and its pages keep the 0xB7. The same in every setup."""
    s = Scenario("mutgap4_svc_complete_reads_listed_only", doc=mutgap4_svc_complete_reads_listed_only.__doc__)
    s.bring_up()
    leaves = {v: s.alloc_page() for v in range(16)}
    leaves[FAR_VPN] = s.alloc_page()
    t = XGPUTask(s, tid=1, pages=16, depth=2, leaves=leaves, length=(FAR_VPN + 1) * PAGE)
    x_va = t.reserve(PAGE)
    t.write(x_va, b"\xA1" * 0x40)
    x = t.ol.add(abi.T_TEXTURE, texture_rec(PAGE, x_va // PAGE, [simple_dim(4, 4)], desc1()))
    b, b_va = t.buffer(PAGE)
    t.define()
    t.install()
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, b_fill_tex_color(x, (4, 4, 1), (1.0, 0.5, 0.0, 1.0), RGBA8Unorm)))
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(x, abi.RES_INVALIDATE_GUEST)]),
             label="fill X, invalidateGuest")
    t.fill(x_va, 0x40, 0xB7)
    data = seg(abi.ENC_BLIT, fill_rec(b, 0, 0x10, 0x5C))
    far = FAR_VPN * PAGE + x
    t.write(far, data)
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [(far, len(data))], [(b, abi.RES_SYNCHRONIZE)]),
             label="a command buffer at 0x4_0000_0000 + X, S synchronized")
    t.dump(x_va, 0x40, "X's pages")
    t.dump(b_va, 0x20, "S")
    t.delete()
    s.events("end")
    return s


# ---- the synchronize flag on an indirect command buffer ----

@scenario(setups=("isolate",))
def mutgap4_svc_icb_synchronize_flag():
    """An indirect command buffer (type 20) listed with invalidateGuest in
    one exec (it is prepared and run, then marked guest-invalid while host
    valid: it needs a sync), then with synchronize (flag 4) in the next. The
    service's completion of the exec leaves out every CPU-paged resource: the
    exec completes, the guest's copy of the commands (overwritten with 0xee
    between the two) is not written, and a third run shows the ICB still
    runs. Isolate only: the in-process task has no such test, so the ICB is
    asked to synchronize in an encoder, which it does not implement
    (doesNotRecognizeSelector); the exec faults (the error path calls its
    completion handler with 2), and the sync command buffer, committed on the
    way out, later calls the same handler again, which kills the host in some
    runs and not in others: nothing there can be locked."""
    from pgtest.icb_cmds import compute_cmd
    from scenarios.icb import ICBTask, exec_ranged, use_resources
    s = Scenario("mutgap4_svc_icb_synchronize_flag", setups=("isolate",), doc=mutgap4_svc_icb_synchronize_flag.__doc__)
    s.bring_up()
    w = ICBTask(s)
    a = w.icb([compute_cmd(pso=w.cpso, binds=[(w.out, 0, 0)], grid=(4, 1, 1), tg=(4, 1, 1))])
    w.start()
    va, size = w.icbs[a]
    w.run_compute(a, flags=abi.RES_INVALIDATE_GUEST, label="execute, guest invalidated")
    w.out_dump("first run", 0x10)
    w.clear_out()
    w.t.fill(va, size, 0xEE)
    body = use_resources(w.out) + exec_ranged(a, 0, 1)
    w.t.reset_cmdbufs()
    cb = w.t.cmdbuf(seg(abi.ENC_COMPUTE, body))
    either(s, abi.OP_EXEC_INDIRECT3, exec3_payload(w.t.tid, [cb], w.res(a, flags=abi.RES_SYNCHRONIZE)),
           "execute, synchronize")
    w.t.dump(va, 0x40, "the ICB's guest pages")
    w.out_dump("second run", 0x10)
    w.clear_out()
    w.run_compute(a, label="execute again")
    w.out_dump("third run", 0x10)
    s.events("end")
    return s


# ---- the task rebuild's reservations ----

def _placements(t, places):
    """The object list and a placement list on its own page, entry i from
    places.get(i, (0, 0)), set with 68 before any exec maps them (as
    scenarios/mutgap_service.py's _placement_list). Returns its VA."""
    n = len(t.ol.entries)
    va = t.reserve(PAGE)
    t.write(va, b"".join(abi.PLACEMENT.pack(*places.get(i, (0, 0))) for i in range(n)))
    t.ol.write()
    t.set_object_and_placement_list(0, va, t.ol.length(), 16 * n)
    return va


@scenario(setups=("isolate",), host=True)
def mutgap4_svc_icb_placement_reserved_index():
    """A placement list that already names an ICB's place when the service
    first maps it: entry A = {resourceIndex 1, gpuResourceID 1}, the index
    and ID this host's Metal gives the first ICB of a fresh service
    (icb_placement). The task's rebuild reserves the index of every type 20
    entry with a nonzero ID, so ICB B, made first with a zero entry, gets
    another index, written into its entry, and A is then made at its index 1
    and runs. Isolate only (placements); host-keyed (Metal's indices)."""
    from pgtest.icb_cmds import compute_cmd
    from scenarios.icb import ICBTask
    s = Scenario("mutgap4_svc_icb_placement_reserved_index", setups=("isolate",), host=True,
                 doc=mutgap4_svc_icb_placement_reserved_index.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    a = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0)], grid=(2, 1, 1), tg=(2, 1, 1))])
    b = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0x10)], grid=(2, 1, 1), tg=(2, 1, 1))])
    t = w.t
    t.define()
    pl = _placements(t, {a: (1, 1)})
    w.run_compute(b, label="make and run B")
    t.dump(pl + 16 * a, 32, "A's and B's entries after B")
    w.run_compute(a, label="make and run A at its index")
    t.dump(pl + 16 * a, 32, "A's and B's entries after A")
    w.out_dump("both ran", 0x20)
    s.events("end")
    return s


@scenario(setups=("isolate",))
def mutgap4_svc_placement_unmakeable_buffer():
    """A placement list, set before the service first maps it, whose entry
    for a buffer (type 1) is nonzero, so the service's task rebuild discards
    that buffer. The buffer's record lies past the task's end, so making it
    fails ("Failed to create memory cursor") and nothing is discarded; the
    lists are mapped and the exec, a fill of another buffer, completes.
    Isolate only (placements are the service's)."""
    s = Scenario("mutgap4_svc_placement_unmakeable_buffer", setups=("isolate",),
                 doc=mutgap4_svc_placement_unmakeable_buffer.__doc__)
    s.bring_up()
    t = XGPUTask(s, tid=1, pages=16)
    bad = t.ol.add_raw(abi.T_BUFFER, 16, 20 * PAGE)
    b, b_va = t.buffer(PAGE)
    t.define()
    pl = _placements(t, {bad: (1, 0)})
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(b, 0, 0x10, 0x3D)))
    either(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(b, 0)]), "the first exec: -rebuildTask")
    t.dump(b_va, 0x20, "B")
    t.dump(pl, 16 * len(t.ol.entries), "the placement list")
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(b, 0x10, 0x10, 0x3E)))
    either(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(b, 0)]), "a second exec")
    t.dump(b_va, 0x20, "B after")
    t.delete()
    s.events("end")
    return s


# ---- InvalidateResources and SynchronizeAndDiscardResources ----

@scenario
def mutgap4_svc_invalidate_set_guest_valid():
    """InvalidateResources (52) with only the set-guest-valid flag (the
    entry's fourth flag byte) on a private buffer whose host copy is newer: a
    fill listed with invalidateGuest left it guest-invalid and host-valid, so
    it needs a sync. The guest writes 0x33 over its page, 52 marks the guest
    copy valid, and SynchronizeResources (53) then has nothing to copy out:
    the 0x33 stays. In the service the private buffer is a detach buffer over
    the guest's pages, which the fill wrote at once and which ignores guest
    validity (tasks_objects_private)."""
    s = Scenario("mutgap4_svc_invalidate_set_guest_valid", doc=mutgap4_svc_invalidate_set_guest_valid.__doc__)
    s.bring_up()
    t = XGPUTask(s, tid=1, pages=16)
    ref, _ = t.buffer(length=PAGE, flags=1, va=8 * PAGE, contents=b"\x11" * 0x40)
    t.define()
    t.install()
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(ref, 0, 0x10, 0xA1)))
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(ref, abi.RES_INVALIDATE_GUEST)]),
             label="a fill, invalidateGuest")
    t.dump(8 * PAGE, 0x20, "the page after the fill")
    t.write(8 * PAGE, b"\x33" * 0x40)
    s.submit(abi.OP_INVALIDATE_RESOURCES, invalidations_payload(1, [(ref, (0, 0, 0, 1))]),
             label="52: set guest valid")
    s.submit(abi.OP_SYNCHRONIZE_RESOURCES, refs_payload(1, [ref]), label="53")
    t.dump(8 * PAGE, 0x40, "the page after 53")
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap4_svc_sync_discard_rewalks():
    """SynchronizeAndDiscardResources (62) discards what supports it. A
    shared buffer B on VA page 9, whose leaf points at guest page A (pattern
    X), is copied into an output buffer; 62 names B; the guest then points VA
    page 9's leaf at guest page C (pattern Y) and B is copied again. In the
    service B is a detach buffer whose discard drops its backing, so the
    second copy walks the page table again and reads Y. The map setup reads Y
    too: discard drops the in-process buffer's no-copy mapping, which the
    next use makes again from the page table. The blocks setup reads X twice:
    the host task's mapping of VA page 9 is what it reads, and the guest's
    leaf does not change it."""
    s = Scenario("mutgap4_svc_sync_discard_rewalks", doc=mutgap4_svc_sync_discard_rewalks.__doc__)
    s.bring_up()
    t = XGPUTask(s, tid=1, pages=16)
    c = s.alloc_page()
    s.mem(s.pa(c), bytes(reversed(PATTERN)))
    ref = t.ol.add(abi.T_BUFFER, buffer_rec(PAGE, 9))
    t.write(9 * PAGE, PATTERN)
    out, out_va = t.buffer(PAGE, va=10 * PAGE)
    t.define()
    t.install()

    def copy(off, label):
        t.reset_cmdbufs()
        cb = t.cmdbuf(seg(abi.ENC_BLIT, b_copy_buf_buf(ref, out, 0, off, 0x20)))
        s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(ref, 0), (out, 0)]), label=label)

    copy(0, "copy B")
    s.submit(abi.OP_SYNCHRONIZE_AND_DISCARD_RESOURCES, refs_payload(1, [ref]), label="62 on B")
    s.u32(t.table_gpa(0, 0, 9), s.pfn(c) | abi.PTE_WRITABLE)
    copy(0x20, "copy B after its leaf moved")
    t.dump(out_va, 0x40, "the two copies")
    t.delete()
    s.events("end")
    return s


# ---- HeapTextureSizeAndAlign's reply mapping ----

@scenario(host=True)
def mutgap4_svc_heap_reply_one_byte_into_hole():
    """HeapTextureSizeAndAlign (64) whose 16-byte reply starts 15 bytes
    before the end of VA page 2, so its last byte is the first of page 3, a
    hole. The memory-map setups map the pages from the offset's page to the
    one holding offset + length, rounded up by 0x3fff: page 3 is a hole, the
    mapping fails ("Failed to make mapping") and the command faults, page 2
    untouched. The blocks setup asks the range tracker for the page run,
    which rounds with 0x3ffe (mutgap3_sys_heap_reply_one_past_page): only
    page 2 is mapped, the hole is never walked, and the reply is written, its
    16th byte into the task block's unmapped page 3. Keyed by the host: the
    blocks setup's reply holds Metal's size and alignment."""
    s = Scenario("mutgap4_svc_heap_reply_one_byte_into_hole", host=True,
                 doc=mutgap4_svc_heap_reply_one_byte_into_hole.__doc__)
    s.bring_up()
    t = XTask(s, tid=1, pages=5, holes={3})
    t.define()
    t.fill(2 * PAGE, PAGE, 0xEE)
    maybe_fault(s, abi.OP_HEAP_TEXTURE_SIZE_AND_ALIGN, heap_payload(1, 3 * PAGE - 15, 16, heap_ser()),
                "a reply 1 byte into the hole")
    t.dump(3 * PAGE - 0x20, 0x20, "page 2's end")
    t.delete()
    s.events("end")
    return s


# ---- stale payload bytes: the FIFO's payload buffer is per thread ----

@scenario
def mutgap4_svc_glyph_short_after_nop():
    """The FIFO copies each command's payload into one buffer it reuses, so
    bytes a short payload lacks are the previous command's. A NOP (30)
    carries a well-formed 44-byte CursorGlyph record for port 0 and task 0
    (with its checksum), then CursorGlyph (4) comes with no payload: its size
    check faults ("Invalid command length") before reading anything, no
    cursor callback fires and the page's glyph record stays zero. The same
    glyph sent whole then reaches cursor_glyph."""
    from scenarios.display_cursor import glyph_pixels, glyph_sum
    s = Scenario("mutgap4_svc_glyph_short_after_nop", displays=[display(0)],
                 doc=mutgap4_svc_glyph_short_after_nop.__doc__)
    g = DisplayGuest(s)
    g.connect(0)
    g.ack(0, 1)
    va = 8 * PAGE
    data = glyph_pixels(4, 3, 16)
    g.t0.write(va, data)
    glyph = p_glyph(0, va, len(data), 16, 4, 3, 1, 2, glyph_sum(4, 3, 16, data))
    s.submit(abi.OP_NOP, glyph + b"\0" * 4, label="a NOP carrying a glyph record")
    either(s, abi.OP_DISPLAY_CURSOR_GLYPH, b"", "CursorGlyph without a payload")
    s.events("after the short glyph")
    s.dump(g.ss(0, 0xF00), 0x30, "the page's glyph record")
    s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, glyph, label="the glyph, whole")
    s.wait(cb="cursorGlyph", label="its callback")
    s.events("after the whole glyph")
    return s


@scenario
def mutgap4_svc_set_properties_short_matrix():
    """SetProperties (10) whose colour matrix (property 0, 12 words) has only
    8 of its words in the payload (48 bytes: {port 0, count 1}, {0, 12}, 8
    words), after a NOP whose 64-byte payload left distinctive words in the
    FIFO's reused payload buffer where the missing 4 would be. The remaining
    size is the payload's less its 8-byte header, so the property's check
    refuses it, nothing is applied, and the stamp signals: the shared page's
    matrices and dirty flag are unchanged."""
    s = Scenario("mutgap4_svc_set_properties_short_matrix", displays=[display(0)],
                 doc=mutgap4_svc_set_properties_short_matrix.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0)
    g.ack(0, 1)
    s.dump(g.ss(0, SS_COLOR_MATRICES), 0x68, "matrices and flags before")
    s.submit(abi.OP_NOP, struct.pack("<16I", *[0x3F800000 + i for i in range(16)]), label="a NOP of 64 bytes")
    words = struct.pack("<8I", *[0x3E800000 + i for i in range(8)])
    s.submit(abi.OP_DISPLAY_SET_PROPERTIES, struct.pack("<IIII", 0, 1, 0, 12) + words,
             label="a matrix with 8 of its 12 words")
    s.fault_state("after")
    s.dump(g.ss(0, SS_COLOR_MATRICES), 0x68, "matrices and flags after")
    s.events("end")
    return s


# ---- the page-table walk under a second root entry ----

@scenario(setups=("isolate",))
def mutgap4_svc_walk_second_root_entry():
    """A depth-3 task of 2^39 bytes (the service accepts any length up to
    4096^3 pages at depth 3; in-process tasks stop at 32 GiB) with leaves for
    VA pages 0-7 and for P - 1 and P, P = 4096 * 4096 + 4096: both under the
    second root entry, the last page of one last-level table and the first of
    the next. A buffer over P - 1 and P is filled across the boundary. The
    walk climbs from the leaf table to level 1 and steps to its next entry,
    relative to the level's saved base (rootIndex << 12): both pages are
    filled. Isolate only (the length)."""
    s = Scenario("mutgap4_svc_walk_second_root_entry", setups=("isolate",),
                 doc=mutgap4_svc_walk_second_root_entry.__doc__)
    s.bring_up()
    p = 4096 * 4096 + 4096
    leaves = {v: s.alloc_page() for v in range(8)}
    leaves.update({p - 1: s.alloc_page(), p: s.alloc_page()})
    t = XGPUTask(s, tid=1, pages=8, depth=3, leaves=leaves, length=1 << 39)
    across, _ = t.buffer(length=2 * PAGE, va=(p - 1) * PAGE)
    t.define()
    s.dump(s.pa(t.root, 0x14), 8, "the task length")
    t.install()
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(across, PAGE - 0x10, 0x20, 0x71)))
    either(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(across, 0)]), "a fill across two leaf tables")
    s.dump(s.pa(leaves[p - 1], PAGE - 0x20), 0x20, "VA page P - 1's end")
    s.dump(s.pa(leaves[p]), 0x20, "VA page P's start")
    t.delete()
    s.events("end")
    return s
