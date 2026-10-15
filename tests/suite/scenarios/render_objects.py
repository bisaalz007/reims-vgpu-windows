"""Render-side state objects as the guest describes them in type-7 records,
created lazily when a render command first names them (docs/testing.md,
sections 6.11, 6.18, 6.19): render and tile pipelines, depth-stencil states,
samplers, fences, and every way a record fails. A failed creation makes the
delegate throw "Failed to create <kind>", so the exec faults."""
import struct

from pgtest import abi
from pgtest import render_lib as R
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.scenario import Scenario

W = H = 4
NPIX = W * H * 4


class ObjTask:
    def __init__(self, s):
        t = self.t = R.RenderTask(s)
        self.vert = t.function("shaders/vert.metallib")
        self.frag = t.function("shaders/frag.metallib")
        self.kernel = t.function("shaders/kernel.metallib")
        self.tile = t.function("shaders/tile.metallib")
        self.tex, _, self.va = t.target(W, H)
        self.buf, _ = t.buffer()

    def add7(self, data):
        return self.t.ol.add(abi.T_SERIALIZED, data)

    def pass_(self):
        return R.describe_pass({0: dict(tex=self.tex, clear=(0.0, 0.0, 1.0, 1.0))}, width=W, height=H)

    def start(self):
        self.t.define()
        self.t.install()


def case(s, o, label, records, fault=False, dump=True, **kw):
    t = o.t
    if dump and not fault:
        t.fill(o.va, NPIX, 0x11)
    t.render([o.pass_()] + records, fault=fault, label=label, **kw)
    if fault:
        s.fault_state(label)
        s.resume_root()
    elif dump:
        t.dump(o.va, NPIX, label)


def rp_record(command_type, blob, length=None, pad=b""):
    """{u32 commandType, u32 unread, u32 unread, u32 descriptorLength} then
    the descriptor."""
    return struct.pack("<IIII", command_type, 0, 0, len(blob) if length is None else length) + blob + pad


@scenario
def render_objects_pipeline():
    """Render pipelines reached by 0x74: the render (14) and tile (51)
    descriptors draw and dispatch, and a reference reused in a later exec
    comes from the cache. Failures: a non-type-7 entry ("Invalid object
    type"), commandType 99 ("Invalid serializer command type"), a record too
    short for the operation header, for the command, or for its
    descriptorLength ("Insufficient data in command"), an empty descriptor
    (Metal gives no descriptor, then no pipeline: "Failed to create render
    pipeline"), a kernel as the vertex function and a vertex function as the
    tile function (Metal's error), and 0x74 with ref 0."""
    s = Scenario("render_objects_pipeline", exclusive=True, watchdog_s=90, doc=render_objects_pipeline.__doc__)
    s.bring_up()
    o = ObjTask(s)
    basic = R.pipeline_blob("basic", v=o.vert, f=o.frag)
    tile = R.pipeline_blob("tile", t=o.tile)
    good = o.add7(rp_record(14, basic))
    good_tile = o.add7(rp_record(51, tile))
    bad = [
        ("a buffer entry", o.buf),
        ("commandType 99", o.add7(rp_record(99, basic))),
        ("4-byte record", o.add7(struct.pack("<I", 14))),
        ("12-byte record, 14", o.add7(struct.pack("<III", 14, 0, 0))),
        ("12-byte record, 51", o.add7(struct.pack("<III", 51, 0, 0))),
        ("descriptorLength past the record, 14", o.add7(rp_record(14, basic, length=len(basic) + 1))),
        ("descriptorLength past the record, 51", o.add7(rp_record(51, tile, length=len(tile) + 1))),
        ("empty descriptor, 14", o.add7(rp_record(14, b""))),
        ("empty descriptor, 51", o.add7(rp_record(51, b""))),
        ("a kernel as the vertex function", o.add7(rp_record(14, R.pipeline_blob("basic", v=o.kernel, f=o.frag)))),
        ("a vertex function as the tile function", o.add7(rp_record(51, R.pipeline_blob("tile", t=o.vert)))),
    ]
    o.start()
    draw = [R.pipeline_state(good), R.draw16(3, 0, 3)]
    case(s, o, "render pipeline", draw)
    case(s, o, "the same reference, cached", draw)
    t = o.t
    t.fill(o.va, NPIX, 0x11)
    t.render([R.tile_size(16, 16), o.pass_(), R.pipeline_state(good_tile), R.dispatch_tile(16, 16)],
             label="tile pipeline")
    t.dump(o.va, NPIX, "tile pipeline")
    for label, ref in bad:
        case(s, o, label, [R.pipeline_state(ref)], fault=True, timeout_ms=30000)
    case(s, o, "0x74 ref 0", [R.pipeline_state(0)], fault=True)
    case(s, o, "0x74 ref past the list", [R.pipeline_state(500)], fault=True)
    o.t.delete()
    s.events("end")
    return s


def malformed(name, blob_fn):
    doc = ("A malformed serialized render descriptor (%s): Metal aborts inside "
           "newRenderPipelineDescriptorWithSerializedData: or the pipeline build "
           "(section 6.18). The abort is inside Metal, so the host dies at the exec "
           "in every setup, an isolated task's included." % name)

    def build():
        sname = "render_objects_malformed_" + name
        s = Scenario(sname, doc=doc)
        s.bring_up()
        o = ObjTask(s)
        ref = o.add7(rp_record(14, blob_fn(o)))
        good = o.add7(rp_record(14, R.pipeline_blob("basic", v=o.vert, f=o.frag)))
        o.start()
        case(s, o, "0x74 the malformed pipeline", [R.pipeline_state(ref)], fault=True)
        case(s, o, "0x74 a good pipeline", [R.pipeline_state(good), R.draw16(3, 0, 3)], fault=True)
        s.events("end")
        return s
    return build


@scenario
def render_objects_malformed():
    """Death tests, one scenario each (section 7.4): 64 bytes of garbage, a
    valid blob truncated to 5 bytes, a vertex function ref that resolves to
    nil (ref 0), and a function ref naming a buffer."""
    out = []
    for name, fn in (
            ("garbage", lambda o: bytes((i * 37 + 11) & 0xFF for i in range(64))),
            ("truncated", lambda o: R.pipeline_blob("basic", v=o.vert, f=o.frag)[:5]),
            ("nil_vertex", lambda o: R.pipeline_blob("basic", v=0, f=o.frag)),
            ("buffer_vertex", lambda o: R.pipeline_blob("basic", v=o.buf, f=o.frag))):
        out.append(malformed(name, fn)())
    return out


def ds_record(command_type, body):
    return R.op_header(command_type) + body


@scenario
def render_objects_depth_stencil():
    """Depth-stencil states reached by 0x68: with front and back stencils,
    front only, back only, neither (each face copied only when its flag is
    set), every compare function; failures: a non-type-7 entry, commandType 3
    (a sampler's), a record too short for the header or the 0x20-byte
    command."""
    s = Scenario("render_objects_depth_stencil", doc=render_objects_depth_stencil.__doc__)
    s.bring_up()
    o = ObjTask(s)
    t = o.t
    face = (7, 1, 2, 3, 0xF0, 0x0F)
    good = [("front and back", t.depth_stencil(compare=3, write=1, front=face, back=(2, 7, 6, 5, 1, 2))),
            ("front only", t.depth_stencil(compare=1, write=0, front=face)),
            ("back only", t.depth_stencil(compare=6, write=1, back=face)),
            ("neither", t.depth_stencil(compare=0, write=0))]
    good += [("compare %d" % c, t.depth_stencil(compare=c, write=1)) for c in (2, 4, 5, 7)]
    bad = [("a buffer entry", o.buf),
           ("commandType 3", t.depth_stencil(command_type=3)),
           ("4-byte record", o.add7(struct.pack("<I", 4))),
           ("header only", o.add7(R.op_header(4))),
           ("0x1c-byte command", o.add7(R.op_header(4) + bytes(0x1C)))]
    o.start()
    for label, ref in good:
        case(s, o, label, [R.depth_stencil_state(ref)], dump=False)
    for label, ref in bad:
        case(s, o, label, [R.depth_stencil_state(ref)], fault=True)
    o.t.delete()
    s.events("end")
    return s


@scenario
def render_objects_sampler():
    """Samplers reached by 0x70: every field of the packed descriptor at
    non-zero values (filters, lodAverage, compare, the three address modes,
    border colour, mip filter, anisotropy, normalized coordinates,
    argument-buffer support, reduction mode, LOD clamps); failures: a
    non-type-7 entry, commandType 4 (a depth-stencil's), a record too short
    for the header or the 0x1c-byte command."""
    s = Scenario("render_objects_sampler", doc=render_objects_sampler.__doc__)
    s.bring_up()
    o = ObjTask(s)
    t = o.t
    good = [("defaults", t.sampler()),
            ("linear, lodAverage, compare Less, clamp to border, opaque white, mip linear, aniso 4",
             t.sampler(min_f=1, mag_f=1, lod_avg=1, compare=1, s=5, t=5, r=5, border=2, mip=2, aniso=4)),
            ("pixel coordinates, reduction min, argument buffers",
             t.sampler(norm=0, reduction=1, argbuf=1, lmin=1.0, lmax=2.0)),
            ("mirror repeat, opaque black border", t.sampler(s=3, t=3, r=3, border=1, reduction=2))]
    bad = [("a buffer entry", o.buf),
           ("commandType 4", t.sampler(command_type=4)),
           ("4-byte record", o.add7(struct.pack("<I", 3))),
           ("header only", o.add7(R.op_header(3))),
           ("0x18-byte command", o.add7(R.op_header(3) + bytes(0x18)))]
    o.start()
    for label, ref in good:
        case(s, o, label, [R.refs_range(0x70, 0, [ref])], dump=False)
    for label, ref in bad:
        case(s, o, label, [R.refs_range(0x70, 0, [ref])], fault=True)
    o.t.delete()
    s.events("end")
    return s


@scenario
def render_objects_fence():
    """Fences reached by 0x18 and 0x19: a header alone makes one; a
    non-type-7 entry, commandType 3 and a short record fail."""
    s = Scenario("render_objects_fence", doc=render_objects_fence.__doc__)
    s.bring_up()
    o = ObjTask(s)
    t = o.t
    good = t.fence()
    bad = [("a buffer entry", o.buf), ("commandType 3", t.fence(command_type=3)),
           ("4-byte record", o.add7(struct.pack("<I", 13)))]
    o.start()
    case(s, o, "update then wait", [R.update_fence(good, 1), R.wait_fence(good, 1)], dump=False)
    for label, ref in bad:
        case(s, o, label, [R.update_fence(ref, 1)], fault=True)
    o.t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",))
def render_objects_placement():
    """Placements (section 6.11): a service task with argument buffers takes
    a placement list with its object list (opcode 68), and the state objects
    read and write their entries. A depth-stencil state zeroes its entry's
    first word; a pipeline with a nonzero resourceID is created at the
    entry's resourceIndex, one with resourceID 0 wherever Metal puts it; a
    sampler with a nonzero resourceIndex whose gpuResourceID comes out
    different fails ("Failed to create sampler state with correct
    resourceID"), one with index 0 is not checked, and one with
    argument-buffer support writes its entry back (those values are the
    host's, so not dumped). Isolate only: in-process tasks never get a
    placement list."""
    s = Scenario("render_objects_placement", setups=("isolate",), doc=render_objects_placement.__doc__)
    s.bring_up()
    o = ObjTask(s)
    t = o.t
    ds = t.depth_stencil(compare=1, write=1)
    rp = o.add7(rp_record(14, R.pipeline_blob("basic", v=o.vert, f=o.frag)))
    rp_zero = o.add7(rp_record(14, R.pipeline_blob("basic", v=o.vert, f=o.frag)))
    smp_bad = t.sampler()
    smp_plain = t.sampler(argbuf=0)
    smp_ab = t.sampler(argbuf=1)
    smp_id = t.sampler(argbuf=0, min_f=1)
    t.define()
    place_va = t.reserve(PAGE)
    n = len(t.ol.entries)
    # Zero placements for everything else: a buffer's is its pinned GPU
    # address and a texture's its resource index.
    places = [(0, 0)] * n
    places[ds] = (0xEEEEEEEEEEEEEEEE, 0xEEEEEEEEEEEEEEEE)
    places[rp] = (3, 1)
    places[rp_zero] = (3, 0)
    places[smp_bad] = (5, 0x1234)
    places[smp_plain] = (0, 0)
    places[smp_id] = (0, 0x1234)
    t.write(place_va, b"".join(abi.PLACEMENT.pack(a, b) for a, b in places))
    t.ol.write()
    t.set_object_and_placement_list(0, place_va, t.ol.length(), 16 * n)
    t.dump(place_va + 16 * ds, 16, "depth-stencil placement before")
    case(s, o, "depth-stencil state", [R.depth_stencil_state(ds)], dump=False)
    t.dump(place_va + 16 * ds, 16, "depth-stencil placement after")
    case(s, o, "pipeline at resource index 3", [R.pipeline_state(rp), R.draw16(3, 0, 3)])
    case(s, o, "pipeline with index 3, resourceID 0 (not placed)", [R.pipeline_state(rp_zero), R.draw16(3, 0, 3)])
    case(s, o, "sampler without argument buffers", [R.refs_range(0x70, 0, [smp_plain])], dump=False)
    t.dump(place_va + 16 * smp_plain, 16, "that sampler's placement")
    case(s, o, "sampler with argument buffers (its placement is written: host values, not observed)",
         [R.refs_range(0x70, 0, [smp_ab])], dump=False)
    case(s, o, "sampler at index 0, resourceID 0x1234 (not checked)", [R.refs_range(0x70, 0, [smp_id])], dump=False)
    t.dump(place_va + 16 * smp_id, 16, "that sampler's placement")
    case(s, o, "sampler at index 5, resourceID 0x1234", [R.refs_range(0x70, 0, [smp_bad])], fault=True)
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def render_objects_placement_sampler():
    """A sampler placed where Metal puts it: resourceIndex 7 and resourceID
    7, the gpuResourceID this host's AGX driver gives a sampler created at
    resource index 7, passes the check and the sampler is used. Isolate only
    (placements); host: the resource ID is the driver's."""
    s = Scenario("render_objects_placement_sampler", setups=("isolate",), host=True,
                 doc=render_objects_placement_sampler.__doc__)
    s.bring_up()
    o = ObjTask(s)
    t = o.t
    smp = t.sampler()
    t.define()
    place_va = t.reserve(PAGE)
    n = len(t.ol.entries)
    places = [(0, 0)] * n
    places[smp] = (7, 7)
    t.write(place_va, b"".join(abi.PLACEMENT.pack(a, b) for a, b in places))
    t.ol.write()
    t.set_object_and_placement_list(0, place_va, t.ol.length(), 16 * n)
    case(s, o, "sampler at index 7, resourceID 7", [R.refs_range(0x70, 0, [smp])], dump=False)
    t.dump(place_va + 16 * smp, 16, "that sampler's placement")
    t.delete()
    s.events("end")
    return s
