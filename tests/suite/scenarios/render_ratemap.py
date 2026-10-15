"""Rasterization rate maps: created from type-7 records (commandType 50) when
the pass prefix's 0x21 names them, and reset from their record by opcode 66
ResetRasterizationRateMap (in-process, which aborts on an existing map, or in
the service). docs/testing.md, sections 6.19, 7.1 and 7.4."""
import struct

from pgtest import abi
from pgtest import render_lib as R
from pgtest.registry import scenario
from pgtest.scenario import Scenario

W = H = 8
NPIX = W * H * 4
IN_PROCESS = ("blocks", "map")
TRI = 3


class RateTask:
    def __init__(self, s):
        t = self.t = R.RenderTask(s)
        vbuf = t.function("shaders/render_vbuf.metallib")
        color = t.function("shaders/render_color.metallib")
        self.p = t.pipeline("basic", v=vbuf, f=color)
        self.tex, _, self.va = t.target(W, H)
        self.pos, _ = t.buffer(contents=R.float4s(*R.FULL))
        self.col, _ = t.buffer(contents=R.float4s((1.0, 0.0, 0.0, 1.0)))
        self.buf, _ = t.buffer()

    def draw(self, ref):
        pre = [R.rate_map(ref)] if ref is not None else []
        return pre + [R.describe_pass({0: dict(tex=self.tex, clear=(0.0, 0.0, 1.0, 1.0))}, width=W, height=H),
                      R.pipeline_state(self.p), R.buffers(0x7D, 0, [(self.pos, 0), (self.col, 0)]),
                      R.scissor(0, 0, 3, 5), R.draw16(TRI, 0, 3)]

    def start(self):
        self.t.define()
        self.t.install()


def run(s, rt, label, ref, fault=False, dump=True):
    t = rt.t
    if dump and not fault:
        t.fill(rt.va, NPIX, 0x11)
    t.render(rt.draw(ref), fault=fault, label=label)
    if fault:
        s.fault_state(label)
        s.resume_root()
    elif dump:
        t.dump(rt.va, NPIX, label)


def reset(s, rt, ref, label, fault=False, payload=None):
    """ResetRasterizationRateMap (66): {u32 taskID, u32 objectReference}."""
    s.submit(abi.OP_RESET_RASTERIZATION_RATE_MAP, struct.pack("<II", rt.t.tid, ref) if payload is None else payload,
             fault=fault, label=label)
    if fault:
        s.fault_state(label)
        s.resume_root()


HALF = ([1.0, 0.5], [1.0])  # left half full rate, right half half rate


@scenario
def render_ratemap_pass():
    """Rate maps in a pass (0x21): a scissored red draw over a blue clear,
    with no map, with a one-layer map whose right half runs at half rate, the
    same reference again (cached), a mutable map with a minimum factor.
    Failures: a non-type-7 entry, commandType 3, a record too short for the
    header or the 0x14-byte command, a layer count past the layer headers
    ("Failed to read layer at index"), horizontal or vertical samples cut
    short, a layer with no samples (Metal makes no map). A map with no layers
    is made."""
    s = Scenario("render_ratemap_pass", doc=render_ratemap_pass.__doc__)
    s.bring_up()
    rt = RateTask(s)
    t = rt.t
    half = t.ratemap(W, H, [HALF])
    mut = t.ratemap(W, H, [([0.5, 1.0], [1.0, 0.5])], mutable=1, min_factor=0.25)
    two = t.ratemap(W, H, [HALF, HALF])
    bad = [("a buffer entry", rt.buf),
           ("commandType 3", t.ratemap(W, H, [HALF], command_type=3)),
           ("4-byte record", t.ol.add(abi.T_SERIALIZED, struct.pack("<I", 50))),
           ("header only", t.ol.add(abi.T_SERIALIZED, R.op_header(50))),
           ("0x10-byte command", t.ol.add(abi.T_SERIALIZED, R.op_header(50) + bytes(0x10))),
           ("layer count 2, one layer header", t.ratemap(W, H, [HALF], count=2, raw_tail=b"")),
           ("horizontal samples cut short", t.ratemap(W, H, [HALF], raw_tail=struct.pack("<f", 1.0))),
           ("vertical samples cut short", t.ratemap(W, H, [HALF], raw_tail=struct.pack("<ff", 1.0, 0.5))),
           ("a layer with no samples", t.ratemap(W, H, [([], [])]))]
    none = t.ratemap(W, H, [])
    rt.start()
    run(s, rt, "no map", None)
    run(s, rt, "0x21 ref 0", 0)
    run(s, rt, "half rate on the right", half)
    run(s, rt, "the same map, cached", half)
    run(s, rt, "mutable, min factor 0.25", mut)
    run(s, rt, "two layers", two)
    run(s, rt, "no layers", none)
    for label, ref in bad:
        run(s, rt, label, ref, fault=True)
    t.delete()
    s.events("end")
    return s


@scenario
def render_ratemap_reset_checks():
    """Opcode 66's checks that do not depend on a map: a payload under 8
    bytes, reference 0 ("Invalid rate map reference"), and a task with no
    object list, which fails to map it (in-process the exception faults the
    command; the service replies 2). Each faults."""
    s = Scenario("render_ratemap_reset_checks", doc=render_ratemap_reset_checks.__doc__)
    s.bring_up()
    rt = RateTask(s)
    bare = R.RenderTask(s, tid=2, pages=4)
    rt.start()
    bare.define()
    reset(s, rt, 0, "short payload", fault=True, payload=struct.pack("<I", rt.t.tid))
    reset(s, rt, 0, "reference 0", fault=True)
    reset(s, bare_as(rt, bare), 1, "a task with no object list", fault=True)
    rt.t.delete()
    bare.delete()
    s.events("end")
    return s


def bare_as(rt, task):
    class _T:
        t = task
    return _T


@scenario(setups=("isolate",))
def render_ratemap_reset_version():
    """Below binary version 106, opcode 66 is "unsupported in this binary
    version" and faults. One setup: the check is the host FIFO's, before any
    task."""
    s = Scenario("render_ratemap_reset_version", setups=("isolate",), doc=render_ratemap_reset_version.__doc__)
    s.bring_up(version=105)
    rt = RateTask(s)
    rt.start()
    reset(s, rt, 1, "version 105", fault=True)
    rt.t.delete()
    s.events("end")
    return s


@scenario(setups=IN_PROCESS)
def render_ratemap_reset_death():
    """Opcode 66 on a rate map that exists, in-process: an in-process task's
    read of the entry aborts (section 7.4), and the host dies at the
    command."""
    s = Scenario("render_ratemap_reset_death", setups=IN_PROCESS, doc=render_ratemap_reset_death.__doc__)
    s.bring_up()
    rt = RateTask(s)
    m = rt.t.ratemap(W, H, [HALF], mutable=1)
    rt.start()
    run(s, rt, "create the map", m)
    # The reset waits behind a GPU draw (a barrier on its stamp), so that the
    # host dies while the runner waits, not while it rings the doorbell.
    st = rt.t.exec3_cmd(rt.draw(m))
    s.fifo.cmd(abi.OP_RESET_RASTERIZATION_RATE_MAP, struct.pack("<II", rt.t.tid, m), barriers=[(0, st)])
    s.fifo.doorbell()
    s.wait_fault(label="reset it")
    return s


@scenario(setups=("isolate",))
def render_ratemap_reset():
    """Opcode 66 in the service on a mutable map a pass created: the record
    is read again and the map reset from it. Rewritten samples show in the
    next pass; a record whose first word is not 0x32 is ignored; a new
    descriptor that is mutable sets its minimum factor; an entry whose type
    is no longer 7 is skipped (the reply is still success); an entry past a
    shrunk object list makes the service reply 2, a fault."""
    s = Scenario("render_ratemap_reset", setups=("isolate",), doc=render_ratemap_reset.__doc__)
    s.bring_up()
    rt = RateTask(s)
    t = rt.t
    m = t.ratemap(W, H, [HALF], mutable=1, min_factor=0.25)
    rt.start()
    typ, n, va = t.ol.entries[m]
    run(s, rt, "created", m)
    reset(s, rt, m, "reset from the same record")
    run(s, rt, "after the reset", m)
    t.write(va, R.ratemap_record(W, H, [([0.5, 1.0], [0.5, 1.0])], mutable=0))
    reset(s, rt, m, "reset from rewritten samples, immutable descriptor")
    run(s, rt, "after it", m)
    t.write(va, R.ratemap_record(W, H, [([1.0, 1.0], [1.0, 1.0])], mutable=1, min_factor=0.5))
    reset(s, rt, m, "reset from a mutable descriptor")
    run(s, rt, "after it", m)
    t.write(va, R.ratemap_record(W, H, [([0.5, 0.5], [0.5, 0.5])], command_type=0x33))
    reset(s, rt, m, "first word 0x33: ignored")
    run(s, rt, "unchanged", m)
    t.ol.entries[m] = (abi.T_BUFFER, n, va)
    t.install()
    reset(s, rt, m, "entry type 1: skipped")
    t.ol.entries[m] = (typ, n, va)
    t.ol.entries = t.ol.entries[:m]
    t.install()
    reset(s, rt, m, "entry past the list", fault=True)
    t.delete()
    s.events("end")
    return s


def service_death(name, record, doc):
    """A reset whose record makes the service raise outside any handler: the
    task dies (a death in its own code, which ends only the task), the
    command faults, and the task's next exec faults."""
    sname = "render_ratemap_reset_dies_" + name
    s = Scenario(sname, setups=("isolate",), doc=doc)
    s.bring_up()
    rt = RateTask(s)
    t = rt.t
    m = t.ratemap(W, H, [HALF], mutable=1)
    rt.start()
    typ, n, va = t.ol.entries[m]
    run(s, rt, "created", m)
    if name == "immutable":
        m = t.ratemap(W, H, [HALF], mutable=0)
        t.install()
        run(s, rt, "an immutable map", m)
    else:
        t.write(va, record)
        t.ol.entries[m] = (typ, len(record), va)
        t.install()
    reset(s, rt, m, "reset", fault=True)
    run(s, rt, "the next exec", m, fault=True)
    s.events("end")
    return s


@scenario(setups=("isolate",))
def render_ratemap_reset_dies():
    """Resets the task does not survive (the raise is inside the service's
    job, GpuTask::reset_rasterization_rate_map, where nothing catches it): an
    immutable map; a
    record under 8 bytes (the header length check); a record too short for
    the descriptor; horizontal samples missing; vertical samples missing. The
    reset's cursor runs 4 bytes past the record (section 7.5), so each record
    is built to fail before reading them."""
    hdr = R.op_header(50) + struct.pack("<I", 0)
    desc = struct.pack("<HHIIf", W, H, 1, 1, 0.0)
    return [
        service_death("immutable", None, "Opcode 66 on an immutable map: the reset raises (the map is "
                      "immutable) and the task dies."),
        service_death("short_header", struct.pack("<I", 50), "Opcode 66 when the map's record is now 4 bytes: "
                      "the header-length check raises and the task dies."),
        service_death("short_descriptor", hdr + desc[:8], "Opcode 66 when the record is 20 bytes: the descriptor "
                      "read from +0xc raises and the task dies."),
        service_death("no_samples", hdr + desc + struct.pack("<HH", 2, 1), "Opcode 66 when the record ends after "
                      "its layer header: reading the horizontal samples raises and the task dies."),
        service_death("zero_samples", hdr + desc + struct.pack("<HH", 0, 0), "Opcode 66 when the record's layer "
                      "has no samples: Metal makes no layer descriptor, the reset raises and the task dies."),
        service_death("no_vertical", hdr + desc + struct.pack("<HH", 1, 2) + struct.pack("<f", 1.0),
                      "Opcode 66 when the record ends before the vertical samples: that read raises and the "
                      "task dies."),
    ]


def absent_map(isolate):
    name = "render_ratemap_reset_absent" + ("" if isolate else "_in_process")
    doc = ("Opcode 66 on a rate map no pass has created, and on a reference past the list: %s (section "
           "7.1). The map is created afterwards by a pass as usual." % (
               "the service replies success to both" if isolate else "the in-process task faults both"))
    s = Scenario(name, setups=("isolate",) if isolate else IN_PROCESS, doc=doc)
    s.bring_up()
    rt = RateTask(s)
    absent = rt.t.ratemap(W, H, [HALF], mutable=1)
    rt.start()
    reset(s, rt, absent, "absent map", fault=not isolate)
    reset(s, rt, 200, "past the list", fault=not isolate)
    run(s, rt, "the map afterwards", absent)
    rt.t.delete()
    s.events("end")
    return s


@scenario
def render_ratemap_reset_absent():
    """Opcode 66 on an absent map, in each setup group (section 7.1)."""
    return [absent_map(True), absent_map(False)]


@scenario
def render_ratemap_three_layers():
    """A rate map record with three layers: Metal crashes adding the third
    layer to the descriptor, before the code's own layer-count assert. The
    crash is inside Metal, so the host dies in every setup, an isolated
    task's included."""
    s = Scenario("render_ratemap_three_layers", doc=render_ratemap_three_layers.__doc__)
    s.bring_up()
    rt = RateTask(s)
    m = rt.t.ratemap(W, H, [HALF, HALF, HALF])
    rt.start()
    run(s, rt, "three layers", m, fault=True)
    run(s, rt, "the task's next exec", None, fault=True)
    s.events("end")
    return s
