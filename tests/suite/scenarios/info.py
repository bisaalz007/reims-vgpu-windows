"""The info decoder (encoder type 4, commands 450-469): replies the CPU
writes into guest memory at decode time, and every check before them
(docs/testing.md, sections 4.3 W12, 6.17, 7.4).

In-process tasks (blocks, map) abort the host on the first reply write,
because an in-process task's writable cursor aborts; so every successful
reply is locked in the isolate setup only, and in-process each command's
checks that throw before the reply cursor is fetched are locked instead.
"""
import struct

from pgtest import abi
from pgtest.exec_objects import (ObjTask, buffer_texture_record, cpu_buffer_record, depth_stencil_record,
                                 heap_record, icb_descriptor, icb_record, info_heap_size, info_ref,
                                 rate_map_record, sampler_record, texture_desc_v1, texture_desc_v2,
                                 ICB_DISPATCH_THREADS)
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import rec, seg

PAGE = abi.PAGE
INFO = abi.ENC_INFO


RO_PAGE = 95  # the read-only page of World's task (Task writable=...)


class World:
    """One task holding one of every object an info command names, laid out
    the same way in every scenario that uses it. Its last page (VA page 95)
    is mapped read-only (a leaf without bit 31)."""

    def __init__(self, s, tid=1):
        t = self.t = ObjTask(s, tid=tid, pages=96, writable=set(range(96)) - {RO_PAGE})
        self.reply, self.reply_va = t.buffer(PAGE)
        self.data, self.data_va = t.buffer(PAGE, contents=bytes(range(256)) * 4)
        cva = t.reserve(PAGE)
        self.cpu_va = cva
        self.cpu = t.ol.add(19, cpu_buffer_record(PAGE, cva // PAGE))
        self.cpso = t.compute_pipeline()
        self.rpso = t.render_pipeline()
        self.sampler = t.ol.add(abi.T_SERIALIZED, sampler_record())
        self.dss = t.ol.add(abi.T_SERIALIZED, depth_stencil_record())
        # 256x256, two samples a direction, the first at quarter rate: a
        # mapping that is not the identity.
        self.rmap = t.ol.add(abi.T_SERIALIZED, rate_map_record(w=256, h=256, samples=[([0.25, 1.0], [0.25, 1.0])]))
        self.tex = t.ol.add(abi.T_SERIALIZER_TEXTURE, buffer_texture_record(self.data))
        hva = t.reserve(4 * PAGE)
        self.heap = t.ol.add(abi.T_HEAP, heap_record(hva // PAGE, 4 * PAGE, 1))
        iva = t.reserve(PAGE)
        self.icb = t.ol.add(abi.T_ICB, icb_record(PAGE, iva // PAGE, icb_descriptor(ICB_DISPATCH_THREADS, kbufs=1)))
        self.bad = t.ol.add_raw(10, 16, 1 * PAGE)  # type 10 names nothing
        self.pairs, self.pairs_va = t.buffer(PAGE)
        self.out, self.out_va = t.buffer(PAGE)
        t.relation_kernel()
        # 453's targets: the rate map's parameter data is 0x6030 bytes here.
        self.param, self.param_va = t.buffer(2 * PAGE)
        self.small, self.small_va = t.buffer(0x100)
        self.private, self.private_va = t.buffer(2 * PAGE, flags=1)
        # Reply targets that fail: read-only pages, a short buffer record,
        # a short CPU buffer record, a record outside the task, and a
        # texture.
        self.ro = t.ol.add(abi.T_BUFFER, struct.pack("<QIB3x", PAGE, RO_PAGE, 0))
        self.ro_va = RO_PAGE * PAGE
        self.short_buf = t.ol.add_raw(abi.T_BUFFER, 8, t.ol.entries[self.reply][2])
        self.short_cpu = t.ol.add_raw(19, 8, t.ol.entries[self.cpu][2])
        self.outside = t.ol.add_raw(abi.T_BUFFER, 16, 200 * PAGE)
        # A CPU buffer is not a resource: as a buffer it is refused.
        self.nobj = len(t.ol.entries)  # the first reference past the list
        t.define()
        t.install()

    def info(self, records, resources=(), opcode=abi.OP_EXEC_INDIRECT3, **kw):
        """One info segment holding records, executed on its own."""
        return self.t.exec_segments([seg(INFO, b"".join(records))], resources=resources, opcode=opcode, **kw)

    def fails(self, label, records, dump=None, **kw):
        """An info exec that faults: the fault state, what the reply region
        holds (nothing written), then resume."""
        self.info(records, fault=True, label=label, **kw)
        self.t.s.fault_state(label)
        if dump:
            self.t.dump(dump[0], dump[1], label + ": reply region")
        self.t.s.resume_root()

    def dump_reply(self, off, n, label):
        self.t.dump(self.reply_va + off, n, label)


@scenario(setups=("isolate",))
def info_replies():
    """Every info command's reply, as the service writes it at decode
    time, each in its own exec at its own offset of one shared reply
    buffer. Isolate only: in-process a reply aborts the host
    (info_reply_aborts_host). Host values (GPU addresses, resource IDs,
    limits): keyed by the host."""
    s = Scenario("info_replies", setups=("isolate",), host=True, doc=info_replies.__doc__)
    s.bring_up()
    w = World(s)
    R = w.reply
    cmds = [
        # 450's reply is 0x19 bytes of fields; the 3 after them are
        # padding, zeros, and not dumped.
        ("450 compute pipeline info", [info_ref(450, w.cpso, R, 0x000)], 0x000, 0x19),
        ("451 heap texture size and align", [info_heap_size(R, 0x020, fmt=70, w=256, h=256, usage=1, options=0x20)], 0x020, 0x10),
        ("452 rate map info", [rec(452, struct.pack("<IIQQ", w.rmap, R, 0x040, 0))], 0x040, 0x20),
        ("454 map screen to physical", [rec(454, struct.pack("<IIQIff", w.rmap, R, 0x060, 0, 10.0, 20.0))], 0x060, 8),
        ("455 map physical to screen", [rec(455, struct.pack("<IIQIff", w.rmap, R, 0x068, 0, 5.0, 6.0))], 0x068, 8),
        ("457 render pipeline info", [info_ref(457, w.rpso, R, 0x080)], 0x080, 10),
        ("458 render imageblock length", [rec(458, struct.pack("<IQQQIQ", w.rpso, 8, 8, 1, R, 0x090))], 0x090, 4),
        ("459 compute imageblock length", [rec(459, struct.pack("<IQQQIQ", w.cpso, 8, 8, 1, R, 0x098))], 0x098, 4),
        ("460 unique id: texture", [rec(460, struct.pack("<IIIQ", w.tex, 1, R, 0x0a0))], 0x0a0, 8),
        ("460 unique id: ICB", [rec(460, struct.pack("<IIIQ", w.icb, 5, R, 0x0a8))], 0x0a8, 8),
        ("461 buffer host info", [info_ref(461, w.data, R, 0x0c0)], 0x0c0, 8),
        ("462 texture host info", [info_ref(462, w.tex, R, 0x0c8)], 0x0c8, 16),
        ("463 heap host info", [info_ref(463, w.heap, R, 0x0d8)], 0x0d8, 8),
        ("464 sampler host info", [info_ref(464, w.sampler, R, 0x0e0)], 0x0e0, 16),
        ("465 ICB host info", [info_ref(465, w.icb, R, 0x0f0)], 0x0f0, 16),
        ("466 render pipeline host info", [info_ref(466, w.rpso, R, 0x100)], 0x100, 16),
        ("467 compute pipeline host info", [info_ref(467, w.cpso, R, 0x110)], 0x110, 16),
        ("469 heap texture size and align v2", [info_heap_size(R, 0x130, v2=True, fmt=70, w=256, h=256, usage=1, options=0x20)], 0x130, 0x10),
    ]
    for label, records, off, n in cmds:
        w.info(records, label=label)
        s.r(abi.R_FAULT_MASK, label)
        w.dump_reply(off, n, label)
    # 468 sends gpuResourceID to the depth-stencil state, which Metal's does
    # not implement: doesNotRecognizeSelector, caught, a fault, and nothing
    # written.
    w.info([info_ref(468, w.dss, R, 0x120)], fault=True, label="468 depth-stencil host info")
    s.fault_state("468")
    w.dump_reply(0x120, 16, "468 depth-stencil host info: nothing written")
    s.resume_root()
    # A pipeline's unique identifier differs from run to run: locked as
    # relations (pgtest/exec_objects.py relate): the same pipeline twice is
    # equal and nonzero, the render and compute pipelines differ.
    P = w.pairs
    w.info([rec(460, struct.pack("<IIIQ", w.rpso, 9, P, 0x00)), rec(460, struct.pack("<IIIQ", w.rpso, 9, P, 0x08)),
            rec(460, struct.pack("<IIIQ", w.cpso, 10, P, 0x10)), rec(460, struct.pack("<IIIQ", w.cpso, 10, P, 0x18)),
            rec(460, struct.pack("<IIIQ", w.rpso, 9, P, 0x20)), rec(460, struct.pack("<IIIQ", w.cpso, 10, P, 0x28))],
           label="460 unique id: pipelines")
    w.t.relate(P, w.out, 3)
    w.t.dump(w.out_va, 12, "460 pipeline identifiers: rpso==rpso, cpso==cpso, rpso vs cpso")
    s.metal("heapTextureSizeAndAlign", format=70, w=256, h=256, usage=1, options=0x20)
    s.events("end")
    return s


@scenario(setups=("isolate",))
def info_replies_more():
    """The rest of the reply paths, isolate only: a CPU buffer (type 19) and
    a private buffer as reply targets (newWritableMemoryCursorForBuffer
    reads their pages directly); 456 mapping coordinates in place; several
    records in one segment; the same replies through InfoIndirect (69); and
    453, which writes through Metal rather than a cursor."""
    s = Scenario("info_replies_more", setups=("isolate",), host=True, doc=info_replies_more.__doc__)
    s.bring_up()
    w = World(s)
    t, R = w.t, w.reply
    w.info([info_ref(461, w.data, w.cpu, 0x10)], label="461 into a CPU buffer")
    t.dump(w.cpu_va, 0x20, "CPU buffer")
    w.info([info_ref(461, w.data, w.private, 0x8)], label="461 into a private buffer")
    t.dump(w.private_va, 0x18, "private buffer's pages")
    # 456: count x {f32, f32} read from the reply cursor's copy and each
    # replaced by its mapping.
    t.write(w.reply_va + 0x200, struct.pack("<6f", 16.0, 8.0, 40.0, 20.0, 47.0, 23.0))
    w.info([rec(456, struct.pack("<IIQQII", w.rmap, R, 0x200, 0, 0, 3))], label="456 three coordinates in place")
    w.dump_reply(0x200, 0x20, "456")
    w.info([rec(456, struct.pack("<IIQQII", w.rmap, R, 0x240, 0, 0, 0))], label="456 no coordinates")
    w.dump_reply(0x240, 8, "456 count 0")
    # Several records in one segment, each written in turn.
    w.info([info_ref(461, w.data, R, 0x300), info_ref(463, w.heap, R, 0x308), info_ref(465, w.icb, R, 0x310)],
           label="three records")
    w.dump_reply(0x300, 0x20, "three records")
    # The same through InfoIndirect.
    w.info([info_ref(461, w.data, R, 0x340), info_ref(462, w.tex, R, 0x348)], opcode=abi.OP_INFO_INDIRECT,
           label="InfoIndirect")
    w.dump_reply(0x340, 0x18, "InfoIndirect replies")
    # 452 with a two-layer map is covered by mutgap2_res_info_rate_map_two_layers; 453 here.
    w.info([rec(453, struct.pack("<IIQ", w.rmap, w.param, 0))], resources=[(w.param, 0)],
           label="453 parameter buffer")
    t.dump(w.param_va, 0x6030, "453 parameter data")
    t.dump(w.param_va + 0x6030, 0x10, "453: nothing past the size")
    s.events("end")
    return s


@scenario
def info_copy_parameter_buffer():
    """453 CopyRasterizationRateParameterBuffer writes through Metal
    (copyParameterDataToBuffer:offset:), not through a reply cursor, so it
    works in every setup: into a shared buffer over guest pages, directly.
    Its checks: the map and the buffer non-null, shared storage,
    offset + size within the buffer (and not overflowing), offset
    aligned."""
    s = Scenario("info_copy_parameter_buffer", host=True, doc=info_copy_parameter_buffer.__doc__)
    s.bring_up()
    w = World(s)
    t = w.t
    res = [(w.param, 0)]
    w.info([rec(453, struct.pack("<IIQ", w.rmap, w.param, 0x1000))], resources=res, label="453 at 0x1000")
    s.r(abi.R_FAULT_MASK)
    t.dump(w.param_va, 0x10, "453: below the offset")
    t.dump(w.param_va + 0x1000, 0x6030, "453 parameter data at 0x1000")
    w.fails("453 map ref 0", [rec(453, struct.pack("<IIQ", 0, w.param, 0))])
    w.fails("453 buffer ref 0", [rec(453, struct.pack("<IIQ", w.rmap, 0, 0))])
    w.fails("453 buffer too small", [rec(453, struct.pack("<IIQ", w.rmap, w.small, 0))], resources=[(w.small, 0)],
            dump=(w.small_va, 0x10))
    w.fails("453 offset past the end", [rec(453, struct.pack("<IIQ", w.rmap, w.param, 0x2000))], resources=res)
    w.fails("453 offset overflows", [rec(453, struct.pack("<IIQ", w.rmap, w.param, 0xFFFFFFFFFFFFFFF0))],
            resources=res)
    w.fails("453 misaligned offset", [rec(453, struct.pack("<IIQ", w.rmap, w.param, 2))], resources=res,
            dump=(w.param_va, 0x10))
    w.fails("453 short record", [rec(453, struct.pack("<IIQ", w.rmap, w.param, 0)[:12])], resources=res)
    s.events("end")
    return s


@scenario
def info_copy_parameter_buffer_private():
    """453 into a private buffer (type 1, flags bit 0): in-process the
    buffer's storage is private and checkBuffer refuses it ("Invalid buffer
    storage mode"): a fault. In the service it is shared storage over the
    guest pages, and the copy lands in guest memory."""
    out = []
    for name, setups, fault in (("info_copy_parameter_buffer_private", ("blocks", "map"), True),
                                ("info_copy_parameter_buffer_private_isolate", ("isolate",), False)):
        s = Scenario(name, setups=setups, host=True, doc=info_copy_parameter_buffer_private.__doc__)
        s.bring_up()
        w = World(s)
        w.info([rec(453, struct.pack("<IIQ", w.rmap, w.private, 0))], resources=[(w.private, 0)], fault=fault,
               label="453 into a private buffer")
        s.fault_state()
        w.t.dump(w.private_va, 0x6040, "private buffer's pages")
        s.events("end")
        out.append(s)
    return out


def _before_cursor(w):
    """Info records that fail before the reply cursor is fetched, or whose
    reply cursor cannot be made (so nothing aborts in-process)."""
    R = w.reply
    X = w.nobj + 5  # a reference past the object list
    z = lambda cmd: rec(cmd, b"")
    return [
        # readOrThrow of a short payload: "Failed to read"
        ("450 short", [rec(450, struct.pack("<IIQ", w.cpso, R, 0)[:15])]),
        ("451 short", [rec(451, (texture_desc_v1() + struct.pack("<IQ", R, 0))[:0x2b])]),
        ("452 short", [rec(452, struct.pack("<IIQ", w.rmap, R, 0))]),
        ("454 short", [rec(454, struct.pack("<IIQIf", w.rmap, R, 0, 0, 1.0))]),
        ("455 short", [z(455)]),
        ("456 short", [rec(456, struct.pack("<IIQQI", w.rmap, R, 0, 0, 0))]),
        ("457 short", [z(457)]),
        ("458 short", [rec(458, struct.pack("<IQQQI", w.rpso, 8, 8, 1, R))]),
        ("459 short", [z(459)]),
        ("460 short", [rec(460, struct.pack("<III", w.tex, 1, R))]),
        ("461 short", [z(461)]),
        ("462 short", [z(462)]),
        ("463 short", [z(463)]),
        ("464 short", [z(464)]),
        ("465 short", [z(465)]),
        ("466 short", [z(466)]),
        ("467 short", [z(467)]),
        ("468 short", [z(468)]),
        ("469 short", [rec(469, (texture_desc_v2() + b"\0" + struct.pack("<IQ", R, 0))[:0x33])]),
        # The object reference 0: the NonNull getters throw "Got invalid"
        # (the ICB's is the info decoder's own).
        ("450 pipeline ref 0", [info_ref(450, 0, R)]),
        ("452 rate map ref 0", [rec(452, struct.pack("<IIQQ", 0, R, 0, 0))]),
        ("454 rate map ref 0", [rec(454, struct.pack("<IIQIff", 0, R, 0, 0, 1.0, 1.0))]),
        ("455 rate map ref 0", [rec(455, struct.pack("<IIQIff", 0, R, 0, 0, 1.0, 1.0))]),
        ("456 rate map ref 0", [rec(456, struct.pack("<IIQQII", 0, R, 0, 0, 0, 1))]),
        ("457 pipeline ref 0", [info_ref(457, 0, R)]),
        ("458 pipeline ref 0", [rec(458, struct.pack("<IQQQIQ", 0, 8, 8, 1, R, 0))]),
        ("459 pipeline ref 0", [rec(459, struct.pack("<IQQQIQ", 0, 8, 8, 1, R, 0))]),
        ("461 buffer ref 0", [info_ref(461, 0, R)]),
        ("462 texture ref 0", [info_ref(462, 0, R)]),
        ("463 heap ref 0", [info_ref(463, 0, R)]),
        ("464 sampler ref 0", [info_ref(464, 0, R)]),
        ("465 ICB ref 0", [info_ref(465, 0, R)]),
        ("466 pipeline ref 0", [info_ref(466, 0, R)]),
        ("467 pipeline ref 0", [info_ref(467, 0, R)]),
        ("468 depth-stencil ref 0", [info_ref(468, 0, R)]),
        # An object of the wrong kind: the delegate throws "Failed to create
        # <kind>".
        ("450 a render pipeline", [info_ref(450, w.rpso, R)]),
        ("457 a compute pipeline", [info_ref(457, w.cpso, R)]),
        ("461 a CPU buffer", [info_ref(461, w.cpu, R)]),
        ("461 type 10", [info_ref(461, w.bad, R)]),
        ("461 past the list", [info_ref(461, X, R)]),
        ("462 a buffer", [info_ref(462, w.data, R)]),
        ("463 a buffer", [info_ref(463, w.data, R)]),
        ("464 a depth-stencil state", [info_ref(464, w.dss, R)]),
        ("465 a buffer", [info_ref(465, w.data, R)]),
        # The reply cursor cannot be made: "Failed to get cursor for buffer"
        # (ResourceManager::new_writable_memory_cursor_for_buffer: types 1 and
        # 19 only, the record read, the entry's range).
        ("450 reply ref 0", [info_ref(450, w.cpso, 0)]),
        ("451 reply ref 0", [info_heap_size(0, 0)]),
        ("452 reply a texture", [rec(452, struct.pack("<IIQQ", w.rmap, w.tex, 0, 0))]),
        ("454 reply a pipeline", [rec(454, struct.pack("<IIQIff", w.rmap, w.cpso, 0, 0, 1.0, 1.0))]),
        ("455 reply type 10", [rec(455, struct.pack("<IIQIff", w.rmap, w.bad, 0, 0, 1.0, 1.0))]),
        ("456 reply past the list", [rec(456, struct.pack("<IIQQII", w.rmap, X, 0, 0, 0, 1))]),
        ("457 reply a short buffer record", [info_ref(457, w.rpso, w.short_buf)]),
        ("458 reply a short CPU buffer record", [rec(458, struct.pack("<IQQQIQ", w.rpso, 8, 8, 1, w.short_cpu, 0))]),
        ("459 reply outside the task", [rec(459, struct.pack("<IQQQIQ", w.cpso, 8, 8, 1, w.outside, 0))]),
        ("460 reply ref 0, bad type", [rec(460, struct.pack("<IIIQ", w.tex, 2, 0, 0))]),
        ("461 reply a heap", [info_ref(461, w.data, w.heap)]),
        ("462 reply a sampler", [info_ref(462, w.tex, w.sampler)]),
        ("463 reply an ICB", [info_ref(463, w.heap, w.icb)]),
        ("464 reply ref 0", [info_ref(464, w.sampler, 0)]),
        ("465 reply ref 0", [info_ref(465, w.icb, 0)]),
        ("466 reply ref 0", [info_ref(466, w.rpso, 0)]),
        ("467 reply ref 0", [info_ref(467, w.cpso, 0)]),
        ("468 reply ref 0", [info_ref(468, w.dss, 0)]),
        ("469 reply ref 0", [info_heap_size(0, 0, v2=True)]),
        # The segment's framing: an unknown command, a record length under
        # 8 ("Underflow") and one past the segment ("Failed to read").
        ("unknown command 449", [rec(449, b"")]),
        ("unknown command 470", [rec(470, bytes(16))]),
        ("record length 7", [rec(461, b"", length=7)]),
        ("record length 0", [rec(461, b"", length=0)]),
        ("record length past the segment", [rec(461, struct.pack("<IIQ", w.data, R, 0), length=0x100)]),
    ]


@scenario
def info_errors():
    """Every info command's checks that throw before its reply cursor is
    fetched, and every way the reply cursor cannot be made: each exec
    faults, nothing is written, and the FIFO resumes. None of them reaches
    the abort in-process (the checks are the info decoder's, the
    deserializer's NonNull getters and the reply buffer's cursor)."""
    s = Scenario("info_errors", host=False, doc=info_errors.__doc__)
    s.bring_up()
    w = World(s)
    for label, records in _before_cursor(w):
        w.fails(label, records)
    w.t.dump(w.reply_va, 0x40, "reply buffer: untouched")
    s.events("end")
    return s


@scenario
def info_framing():
    """What the info decoder accepts without writing anything: an empty
    segment, 1-7 trailing bytes after the last record, a record longer
    than its payload (the rest ignored), and an info segment with no
    records through InfoIndirect. Records that write are isolate-only
    (info_replies), so these carry none."""
    s = Scenario("info_framing", doc=info_framing.__doc__)
    s.bring_up()
    w = World(s)
    w.info([], label="an empty info segment")
    for n in range(1, 8):
        w.info([bytes(range(0x40, 0x40 + n))], label="%d trailing bytes" % n)
    w.info([], opcode=abi.OP_INFO_INDIRECT, label="InfoIndirect, empty")
    s.r(abi.R_FAULT_MASK)
    s.events("end")
    return s


def _after_cursor(w):
    """Records that fail after the reply cursor was fetched: isolate only
    (in-process the fetch itself aborts)."""
    R = w.reply
    return [
        ("456 coordinates past the end", [rec(456, struct.pack("<IIQQII", w.rmap, R, PAGE - 8, 0, 0, 2))],
         (w.reply_va + PAGE - 8, 8)),
        ("450 offset at the end", [info_ref(450, w.cpso, R, PAGE)]),
        ("450 reply straddles the end", [info_ref(450, w.cpso, R, PAGE - 0x10)]),
        ("451 offset past the end", [info_heap_size(R, 2 * PAGE, fmt=70, w=16, h=16, usage=1, options=0x20)]),
        ("452 offset past the end", [rec(452, struct.pack("<IIQQ", w.rmap, R, PAGE, 0))]),
        ("452 layer sizes past the end", [rec(452, struct.pack("<IIQQ", w.rmap, R, PAGE - 0x14, 0))]),
        ("454 offset past the end", [rec(454, struct.pack("<IIQIff", w.rmap, R, PAGE, 0, 1.0, 1.0))]),
        ("455 offset past the end", [rec(455, struct.pack("<IIQIff", w.rmap, R, PAGE, 0, 1.0, 1.0))]),
        ("457 offset past the end", [info_ref(457, w.rpso, R, PAGE)]),
        ("458 offset past the end", [rec(458, struct.pack("<IQQQIQ", w.rpso, 8, 8, 1, R, PAGE))]),
        ("459 offset past the end", [rec(459, struct.pack("<IQQQIQ", w.cpso, 8, 8, 1, R, PAGE))]),
        ("460 offset past the end", [rec(460, struct.pack("<IIIQ", w.tex, 1, R, PAGE))]),
        ("460 object type 2", [rec(460, struct.pack("<IIIQ", w.tex, 2, R, 0))]),
        ("460 object type 0", [rec(460, struct.pack("<IIIQ", w.tex, 0, R, 0))]),
        ("460 texture ref 0", [rec(460, struct.pack("<IIIQ", 0, 1, R, 0))]),
        ("460 ICB ref 0", [rec(460, struct.pack("<IIIQ", 0, 5, R, 0))]),
        ("460 render pipeline ref 0", [rec(460, struct.pack("<IIIQ", 0, 9, R, 0))]),
        ("460 compute pipeline ref 0", [rec(460, struct.pack("<IIIQ", 0, 10, R, 0))]),
        ("461 offset past the end", [info_ref(461, w.data, R, PAGE)]),
        ("462 offset past the end", [info_ref(462, w.tex, R, PAGE)]),
        ("463 offset past the end", [info_ref(463, w.heap, R, PAGE)]),
        ("464 offset past the end", [info_ref(464, w.sampler, R, PAGE)]),
        ("465 offset past the end", [info_ref(465, w.icb, R, PAGE)]),
        ("466 offset past the end", [info_ref(466, w.rpso, R, PAGE)]),
        ("467 offset past the end", [info_ref(467, w.cpso, R, PAGE)]),
        ("469 offset past the end", [info_heap_size(R, PAGE, v2=True, fmt=70, w=16, h=16, usage=1, options=0x20)]),
        ("461 a read-only reply buffer", [info_ref(461, w.data, w.ro, 0)]),
    ]


@scenario(setups=("isolate",))
def info_errors_isolate():
    """Info records that fail after their reply cursor is fetched: a reply
    offset at or past the end of the reply buffer (the write throws "Failed
    to write to cursor"), 460's object-type check and NonNull getters,
    which follow the fetch, 456 reading coordinates past the end, and a
    reply buffer over a read-only page (the service's writable cursor
    refuses it). Nothing is written, except where a reply straddles the
    end: write<T> checks the whole reply first."""
    s = Scenario("info_errors_isolate", setups=("isolate",), doc=info_errors_isolate.__doc__)
    s.bring_up()
    w = World(s)
    t = w.t
    t.write(w.reply_va + PAGE - 8, struct.pack("<ff", 40.0, 20.0))
    for label, records, *dump in _after_cursor(w):
        w.fails(label, records, dump=dump[0] if dump else None)
    t.dump(w.reply_va, 0x40, "reply buffer start")
    t.dump(w.reply_va + PAGE - 0x20, 0x20, "reply buffer end")
    t.dump(w.ro_va, 0x10, "read-only page")
    s.events("end")
    return s


@scenario(setups=("isolate",))
def info_reply_before_error():
    """A decode error after a reply was written: the reply stays in guest
    memory (the CPU wrote it at decode time), the exec faults, and no
    segment of the exec reaches the GPU: a blit fill in the segment before
    is never committed (docs/testing.md, section 6.12)."""
    s = Scenario("info_reply_before_error", setups=("isolate",), host=True, doc=info_reply_before_error.__doc__)
    s.bring_up()
    w = World(s)
    t = w.t
    fill = rec(306, struct.pack("<IQQB3x", w.param, 0, 0x100, 0xAB))
    t.exec_segments([seg(abi.ENC_BLIT, fill),
                     seg(INFO, info_ref(461, w.data, w.reply, 0) + rec(449, b""))],
                    resources=[(w.param, 0)], fault=True, label="fill, reply, unknown command")
    s.fault_state("after")
    w.dump_reply(0, 8, "461's reply stays")
    t.dump(w.param_va, 0x10, "the fill never ran")
    s.resume_root()
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def info_reply_aborts_host():
    """In-process, the first info reply aborts the host: the reply cursor
    comes from the in-process task's writable cursor, which aborts
    (docs/testing.md, section 7.4). Checks before the fetch pass
    (info_errors); here 461 with a valid buffer and reply buffer, through
    ExecIndirect3."""
    s = Scenario("info_reply_aborts_host", setups=("blocks", "map"), doc=info_reply_aborts_host.__doc__)
    s.bring_up()
    w = World(s)
    w.t.exec_last([seg(INFO, info_ref(461, w.data, w.reply, 0))], label="461: the host dies")
    return s


@scenario(setups=("blocks", "map"))
def info_reply_aborts_host_cpu_buffer():
    """The same death with a CPU buffer (type 19) as the reply target,
    through InfoIndirect (69)."""
    s = Scenario("info_reply_aborts_host_cpu_buffer", setups=("blocks", "map"),
                 doc=info_reply_aborts_host_cpu_buffer.__doc__)
    s.bring_up()
    w = World(s)
    w.t.exec_last([seg(INFO, info_ref(450, w.cpso, w.cpu, 0))], opcode=abi.OP_INFO_INDIRECT,
                  label="450: the host dies")
    return s


@scenario(setups=("blocks", "map"))
def info_460_bad_type_aborts_host():
    """460 fetches its reply cursor before it checks the object type
    (docs/testing.md, section 7.1): an invalid object type with a valid
    reply buffer still aborts the host in-process, where the isolate
    setup faults (info_errors_isolate)."""
    s = Scenario("info_460_bad_type_aborts_host", setups=("blocks", "map"),
                 doc=info_460_bad_type_aborts_host.__doc__)
    s.bring_up()
    w = World(s)
    w.t.exec_last([seg(INFO, rec(460, struct.pack("<IIIQ", w.tex, 2, w.reply, 0)))], label="460 type 2: the host dies")
    return s
