"""The render pass: 0x1a DescribeRenderPass's attachments, actions, clear
values, resolve and visibility buffers and target size, the pass prefix
records 0x1e-0x24, and how render segments are framed (docs/testing.md,
sections 6.13 and 6.16).
"""
import struct

from pgtest import abi
from pgtest import render_lib as R
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import rec, seg

CLEAR = (0.25, 0.5, 0.75, 1.0)  # 40 80 bf ff


def gpu_task(s, pages=64):
    """A task with the suite's functions and the pipelines most passes use.
    Returns (task, dict of refs)."""
    t = R.RenderTask(s, pages=pages)
    o = {}
    o["vert"] = t.function("shaders/vert.metallib")
    o["frag"] = t.function("shaders/frag.metallib")
    o["basic"] = t.pipeline("basic", v=o["vert"], f=o["frag"])
    return t, o


def step(s, t, label, records, fault=False, **kw):
    """One exec of one render segment. A decode error faults the exec:
    the fault state is observed and the root FIFO resumed."""
    t.render(records, fault=fault, label=label, **kw)
    if fault:
        s.fault_state(label)
        s.resume_root()


def start(s, t):
    t.define()
    t.install()


@scenario
def render_basic():
    """Section 6.16's measurement: colour 0 a texture over a shared buffer,
    load Clear, store Store, the vert/frag pipeline, DrawPrimitives16
    {triangle, 0, 3}: every pixel is ff8040ff in guest memory."""
    s = Scenario("render_basic", doc=render_basic.__doc__)
    s.bring_up()
    t, o = gpu_task(s)
    tex, buf, va = t.target(4, 4)
    start(s, t)
    t.fill(va, 64, 0x11)
    t.render([R.describe_pass({0: dict(tex=tex, clear=(0.0, 0.0, 1.0, 1.0))}, width=4, height=4),
              R.pipeline_state(o["basic"]), R.draw16(3, 0, 3)])
    s.r(abi.R_FAULT_MASK)
    t.dump(va, 64, "target")
    t.delete()
    s.events("end")
    return s


@scenario
def render_pass_actions():
    """Colour load and store actions and clear values: Clear+Store writes the
    clear colour; Load keeps what the buffer held (a scissored draw over it);
    Clear+DontCare leaves guest memory as it was; Unknown resolved later by
    0x66 SetColorStoreAction stores; 0x67 store options at index 0; slice 1
    of a 2D-array target over a buffer."""
    s = Scenario("render_pass_actions", doc=render_pass_actions.__doc__)
    s.bring_up()
    t, o = gpu_task(s)
    a, _, ava = t.target(4, 4)
    b, _, bva = t.target(4, 4, ttype=R.TEX_2D_ARRAY, array=2)
    start(s, t)
    ca = lambda **kw: R.describe_pass({0: dict(tex=a, clear=CLEAR, **kw)}, width=4, height=4)
    t.fill(ava, 64, 0x11)
    step(s, t, "clear, store", [ca()])
    t.dump(ava, 64, "cleared")
    t.write(ava, bytes(range(64)))
    step(s, t, "load, scissored draw", [ca(load=R.LOAD_LOAD), R.pipeline_state(o["basic"]), R.scissor(0, 0, 2, 2),
                                         R.draw16(3, 0, 3)])
    t.dump(ava, 64, "loaded, top-left 2x2 drawn")
    t.fill(ava, 64, 0x22)
    step(s, t, "clear, don't care", [ca(store=R.STORE_DONTCARE)])
    t.dump(ava, 64, "not stored")
    t.fill(ava, 64, 0x33)
    step(s, t, "clear, unknown, then 0x66 store", [ca(store=R.STORE_UNKNOWN), R.color_store_action(R.STORE_STORE, 0)])
    t.dump(ava, 64, "stored by 0x66")
    t.fill(ava, 64, 0x44)
    step(s, t, "clear, 0x67 options 0", [ca(), R.color_store_options(0, 0)])
    t.dump(ava, 64, "stored")
    t.fill(bva, 128, 0x55)
    step(s, t, "slice 1 of a 2D array", [R.describe_pass({0: dict(tex=b, slice_=1, clear=CLEAR)}, array_length=2,
                                                          width=4, height=4)])
    t.dump(bva, 128, "2D array, both slices")
    t.delete()
    s.events("end")
    return s


@scenario
def render_pass_eight():
    """Eight colour attachments, each cleared to its own colour; then only 0,
    3 and 7 (the others are skipped). The last attachment with a texture sets
    the max colour index that 0x66, 0x67 (index >) and 0x96 (index >=, so
    with only attachment 0 bound, index 0 fails) check."""
    s = Scenario("render_pass_eight", doc=render_pass_eight.__doc__)
    s.bring_up()
    t, o = gpu_task(s)
    tg = [t.target(4, 4) for _ in range(8)]
    start(s, t)
    for _, _, va in tg:
        t.fill(va, 64, 0x11)
    cols = {i: dict(tex=tg[i][0], clear=(i / 7.0, 1.0 - i / 7.0, 0.5, 1.0)) for i in range(8)}
    step(s, t, "eight", [R.describe_pass(cols, width=4, height=4)])
    for i, (_, _, va) in enumerate(tg):
        t.dump(va, 16, "attachment %d, row 0" % i)
    some = {i: dict(tex=tg[i][0], clear=(1.0, 1.0, 1.0, 1.0)) for i in (0, 3, 7)}
    step(s, t, "0, 3, 7", [R.describe_pass(some, width=4, height=4)])
    for i, (_, _, va) in enumerate(tg):
        t.dump(va, 16, "attachment %d, row 0" % i)
    only0 = R.describe_pass({0: dict(tex=tg[0][0], clear=CLEAR)}, width=4, height=4)
    only3 = R.describe_pass({3: dict(tex=tg[3][0], clear=CLEAR)}, width=4, height=4)
    all8 = R.describe_pass(cols, width=4, height=4)
    step(s, t, "only 0: 0x66 index 0", [only0, R.color_store_action(R.STORE_STORE, 0)])
    step(s, t, "only 0: 0x66 index 1", [only0, R.color_store_action(R.STORE_STORE, 1)], fault=True)
    step(s, t, "only 0: 0x67 index 1", [only0, R.color_store_options(0, 1)], fault=True)
    step(s, t, "only 0: 0x96 index 0", [only0, R.color_resolve(tg[1][0], 0, 0, 0, 0, 0)], fault=True)
    step(s, t, "only 3: 0x66 index 3", [only3, R.color_store_action(R.STORE_STORE, 3)])
    step(s, t, "only 3: 0x66 index 4", [only3, R.color_store_action(R.STORE_STORE, 4)], fault=True)
    step(s, t, "only 3: 0x96 index 3", [only3, R.color_resolve(tg[1][0], 0, 0, 0, 0, 3)], fault=True)
    step(s, t, "all 8: 0x66 index 7", [all8, R.color_store_action(R.STORE_STORE, 7)])
    step(s, t, "all 8: 0x67 index 7", [all8, R.color_store_options(0, 7)])
    step(s, t, "all 8: 0x66 index 8", [all8, R.color_store_action(R.STORE_STORE, 8)], fault=True)
    step(s, t, "all 8: 0x67 index 8", [all8, R.color_store_options(0, 8)], fault=True)
    step(s, t, "all 8: 0x96 index 7", [all8, R.color_resolve(tg[1][0], 0, 0, 0, 0, 7)], fault=True)
    nocol = [R.default_raster_samples(1), R.describe_pass({}, width=4, height=4)]
    step(s, t, "no colour: 0x66 index 0", nocol + [R.color_store_action(0, 0)])
    step(s, t, "no colour: 0x66 index 1", nocol + [R.color_store_action(0, 1)], fault=True)
    t.delete()
    s.events("end")
    return s


@scenario
def render_pass_memoryless():
    """Memoryless attachments (serializer textures, commandType 1): a store
    action other than DontCare, Unknown or MultisampleResolve becomes
    DontCare, and a load action other than DontCare or Clear becomes Clear,
    for depth, stencil and colour; resolve textures, unused (store Store or
    DontCare), on each attachment. Nothing of a memoryless texture reaches
    guest memory; each of those passes must complete. Last, depth store
    MultisampleResolve with one sample and no resolve texture: Metal
    crashes, and the host dies in every setup, an isolated task's
    included."""
    s = Scenario("render_pass_memoryless", doc=render_pass_memoryless.__doc__)
    s.bring_up()
    t, o = gpu_task(s)
    ds = t.memoryless(4, 4, R.D32S8)
    cm = t.memoryless(4, 4, R.RGBA8)
    a, _, ava = t.target(4, 4)
    b, _, bva = t.target(4, 4)
    start(s, t)
    t.fill(ava, 64, 0x11)
    for load, store in ((R.LOAD_LOAD, R.STORE_STORE), (R.LOAD_DONTCARE, R.STORE_DONTCARE),
                        (R.LOAD_CLEAR, R.STORE_UNKNOWN), (R.LOAD_LOAD, R.STORE_STORE_RESOLVE)):
        lbl = "load %d store %d" % (load, store)
        d = dict(tex=ds, load=load, store=store, clear=0.5)
        st = dict(tex=ds, load=load, store=store, clear=7)
        step(s, t, "depth+stencil " + lbl, [R.describe_pass({0: dict(tex=a, clear=CLEAR)}, depth=d, stencil=st,
                                                            width=4, height=4)])
        step(s, t, "colour 1 memoryless " + lbl, [R.describe_pass(
            {0: dict(tex=a, clear=CLEAR), 1: dict(tex=cm, load=load, store=store, clear=CLEAR)}, width=4, height=4)])
    t.dump(ava, 64, "colour 0 after the passes")
    t.fill(ava, 64, 0x22)
    t.fill(bva, 64, 0x22)
    step(s, t, "unused resolve textures on depth, stencil, colour", [R.describe_pass(
        {0: dict(tex=a, resolve=b, clear=CLEAR)},
        depth=dict(tex=ds, resolve=b, load=R.LOAD_CLEAR, store=R.STORE_DONTCARE, clear=1.0, filter=1),
        stencil=dict(tex=ds, resolve=b, load=R.LOAD_CLEAR, store=R.STORE_DONTCARE, clear=3, filter=1),
        width=4, height=4)])
    t.dump(ava, 64, "colour 0")
    t.dump(bva, 64, "the resolve texture")
    step(s, t, "depth store MultisampleResolve, 1 sample, no resolve texture", [R.describe_pass(
        {0: dict(tex=a, clear=CLEAR)}, depth=dict(tex=ds, load=R.LOAD_CLEAR, store=R.STORE_RESOLVE),
        width=4, height=4)], fault=True)
    s.events("end")
    return s


@scenario
def render_pass_memoryless_stencil_resolve():
    """A memoryless stencil attachment keeps store MultisampleResolve like
    depth does; with one sample and no resolve texture Metal crashes, as for
    depth in render_pass_memoryless: the host dies in every setup."""
    s = Scenario("render_pass_memoryless_stencil_resolve", doc=render_pass_memoryless_stencil_resolve.__doc__)
    s.bring_up()
    t, o = gpu_task(s)
    ds = t.memoryless(4, 4, R.D32S8)
    a, _, ava = t.target(4, 4)
    start(s, t)
    step(s, t, "stencil store MultisampleResolve, 1 sample, no resolve texture", [R.describe_pass(
        {0: dict(tex=a, clear=CLEAR)}, stencil=dict(tex=ds, load=R.LOAD_CLEAR, store=R.STORE_RESOLVE),
        width=4, height=4)], fault=True)
    s.events("end")
    return s


def msaa_task(s):
    t = R.RenderTask(s)
    o = {}
    o["vbuf"] = t.function("shaders/render_vbuf.metallib")
    o["color"] = t.function("shaders/render_color.metallib")
    o["p"] = t.pipeline("msaa4", v=o["vbuf"], f=o["color"])
    # A triangle with an edge across the target, red.
    o["pos"], _ = t.buffer(contents=R.float4s((-1.0, -1.0, 0.0, 1.0), (1.0, -0.6, 0.0, 1.0), (-0.3, 1.0, 0.0, 1.0)))
    o["col"], _ = t.buffer(contents=R.float4s((1.0, 0.0, 0.0, 1.0)))
    o["ms"] = t.memoryless(8, 8, R.RGBA8, ttype=R.TEX_2D_MS, samples=4)
    o["res"], _, o["va"] = t.target(8, 8)
    return t, o


@scenario(host=True)
def render_pass_msaa():
    """MSAA: a 4-sample memoryless colour attachment resolved into a texture
    over a shared buffer (store MultisampleResolve), a triangle whose edges
    cross pixels; then the same with 0x20 SetProgrammableSamplePositions,
    which moves the edge pixels' coverage; then store
    StoreAndMultisampleResolve, which the memoryless rule turns into
    DontCare, so nothing is resolved. Count 9 throws; a count longer than its
    points fails to read. Last, 8 positions (the most the decoder takes) on a
    4-sample pass: Metal aborts, and the host dies in every setup, an
    isolated task's included. host: the default sample positions are the
    GPU's."""
    s = Scenario("render_pass_msaa", host=True, doc=render_pass_msaa.__doc__)
    s.bring_up()
    t, o = msaa_task(s)
    start(s, t)
    draw = [R.pipeline_state(o["p"]), R.buffers(0x7D, 0, [(o["pos"], 0), (o["col"], 0)]), R.draw16(3, 0, 3)]
    ps = lambda store=R.STORE_RESOLVE: R.describe_pass(
        {0: dict(tex=o["ms"], resolve=o["res"], store=store, clear=(0.0, 0.0, 0.0, 1.0))}, width=8, height=8)
    t.fill(o["va"], 256, 0x11)
    step(s, t, "resolve", [ps()] + draw)
    t.dump(o["va"], 256, "resolved, default positions")
    t.fill(o["va"], 256, 0x11)
    pos = [(0.125, 0.125), (0.875, 0.125), (0.125, 0.875), (0.875, 0.875)]
    step(s, t, "resolve, sample positions", [R.sample_positions(pos), ps()] + draw)
    t.dump(o["va"], 256, "resolved, corner positions")
    t.fill(o["va"], 256, 0x11)
    step(s, t, "store and resolve, memoryless", [ps(R.STORE_STORE_RESOLVE)] + draw)
    t.dump(o["va"], 256, "not resolved")
    step(s, t, "9 sample positions", [R.sample_positions([(0.5, 0.5)] * 9), ps()], fault=True)
    step(s, t, "count 2, one point", [R.sample_positions([(0.5, 0.5)], count=2), ps()], fault=True)
    step(s, t, "8 sample positions on 4 samples", [R.sample_positions([(0.5, 0.5)] * 8), ps()], fault=True)
    s.events("end")
    return s


@scenario
def render_pass_visibility():
    """Visibility results: a pass with no attachments, only a target size and
    0x1e SetDefaultRasterSampleCount, a vertex-only pipeline, and 0x84
    SetVisibilityResultMode Counting, Boolean and Disabled at three offsets
    of the pass's visibility buffer. Counting counts samples: 64 for a full
    8x8 target at 1 sample, 256 at 4. A visibility ref that is not a buffer,
    or out of the list, faults."""
    s = Scenario("render_pass_visibility", doc=render_pass_visibility.__doc__)
    s.bring_up()
    t, o = gpu_task(s)
    p1 = t.pipeline("nocolor", v=o["vert"])
    p4 = t.pipeline("nocolor4", v=o["vert"])
    vis, vva = t.buffer()
    a, _, ava = t.target(4, 4)
    start(s, t)
    t.fill(vva, 64, 0xEE)
    body = lambda p: [R.pipeline_state(p), R.visibility_mode(2, 0), R.draw16(3, 0, 3), R.visibility_mode(1, 8),
                      R.draw16(3, 0, 3), R.visibility_mode(0, 16), R.draw16(3, 0, 3)]
    step(s, t, "1 sample", [R.default_raster_samples(1), R.describe_pass({}, vis=vis, width=8, height=8)] + body(p1))
    t.dump(vva, 32, "counting @0, boolean @8, disabled @16")
    t.fill(vva, 64, 0xEE)
    step(s, t, "4 samples", [R.default_raster_samples(4), R.describe_pass({}, vis=vis, width=8, height=8)] + body(p4))
    t.dump(vva, 32, "counting @0, boolean @8, disabled @16")
    t.fill(vva, 64, 0xEE)
    step(s, t, "with a colour attachment", [R.describe_pass({0: dict(tex=a, clear=CLEAR)}, vis=vis, width=4, height=4),
                                            R.pipeline_state(o["basic"]), R.visibility_mode(2, 24), R.draw16(3, 0, 3)])
    t.dump(vva, 32, "counting @24")
    t.dump(ava, 64, "colour")
    step(s, t, "visibility ref a texture", [R.describe_pass({}, vis=a, width=4, height=4)], fault=True)
    step(s, t, "visibility ref past the list", [R.describe_pass({}, vis=200, width=4, height=4)], fault=True)
    t.delete()
    s.events("end")
    return s


@scenario
def render_pass_prefix():
    """The pass prefix: 0x1f SetDefaultColorSampleCount, 0x21
    SetVariableRateRasterizationMap with ref 0 (nil), 0x22
    ImageBlockSampleLength, 0x23 SetThreadGroupMemoryLength, 0x24
    SetTileParameters, each then a clear. Errors: any other id before 0x1a
    ("Expected a render pass descriptor command"); a record length under 8,
    which wraps (length - 8 as u32) and fails to read; a prefix record too
    short for its payload; a segment that ends before 0x1a; a short 0x1a;
    0x1a and the prefix ids in the main loop ("Unrecognized command")."""
    s = Scenario("render_pass_prefix", doc=render_pass_prefix.__doc__)
    s.bring_up()
    t, o = gpu_task(s)
    a, _, ava = t.target(4, 4)
    start(s, t)
    ps = R.describe_pass({0: dict(tex=a, clear=CLEAR)}, width=4, height=4)
    t.fill(ava, 64, 0x11)
    step(s, t, "0x1f", [R.default_color_samples(1), ps])
    t.dump(ava, 16, "row 0")
    step(s, t, "0x21 ref 0", [R.rate_map(0), ps])
    step(s, t, "0x22", [R.imageblock_sample_length(32), ps])
    step(s, t, "0x23", [R.threadgroup_memory_length(256), ps])
    step(s, t, "0x24", [R.tile_size(16, 16), ps])
    step(s, t, "all of them", [R.default_raster_samples(1), R.default_color_samples(1), R.rate_map(0),
                               R.imageblock_sample_length(32), R.threadgroup_memory_length(0), R.tile_size(32, 32), ps])
    step(s, t, "a draw before the pass", [R.draw16(3, 0, 3), ps], fault=True)
    step(s, t, "0x65 before the pass", [R.blend_color(0, 0, 0, 0), ps], fault=True)
    step(s, t, "prefix length 4", [rec(0x1E, b"", length=4) + struct.pack("<I", 1), ps], fault=True)
    step(s, t, "prefix length 0", [rec(0x1E, b"", length=0), ps], fault=True)
    step(s, t, "0x1e without payload", [rec(0x1E, b""), ps], fault=True)
    step(s, t, "0x24 with 2 bytes", [rec(0x24, b"\x10\x00"), ps], fault=True)
    step(s, t, "prefix only", [R.default_raster_samples(1)], fault=True)
    step(s, t, "empty segment", [], fault=True)
    step(s, t, "0x1a short", [rec(0x1A, bytes(0x200))], fault=True)
    step(s, t, "0x1a twice", [ps, ps], fault=True)
    step(s, t, "0x1e in the main loop", [ps, R.default_raster_samples(1)], fault=True)
    step(s, t, "0x24 in the main loop", [ps, R.tile_size(16, 16)], fault=True)
    t.dump(ava, 16, "row 0")
    t.delete()
    s.events("end")
    return s


@scenario
def render_segments():
    """How render segments are framed (section 6.13): a pass continued into a
    second command-buffer range of the same exec (continuedByNext, then
    continuesPrevious), which keeps the encoder and the pipeline; trailing
    bytes under a record header are ignored; a record length under 8 throws
    "Underflow"; one past the segment fails to read; ids outside the table
    ("Unrecognized command") fault."""
    s = Scenario("render_segments", doc=render_segments.__doc__)
    s.bring_up()
    t, o = gpu_task(s)
    a, _, ava = t.target(4, 4)
    start(s, t)
    ps = R.describe_pass({0: dict(tex=a, clear=CLEAR)}, width=4, height=4)
    t.fill(ava, 64, 0x11)
    s1 = seg(abi.ENC_RENDER, ps + R.pipeline_state(o["basic"]) + R.scissor(0, 0, 2, 2) + R.draw16(3, 0, 3), cont_next=1)
    s2 = seg(abi.ENC_RENDER, R.scissor(2, 2, 2, 2) + R.draw16(3, 0, 3), cont_prev=1)
    t.exec_segments([s1, s2], label="continued")
    t.dump(ava, 64, "top-left and bottom-right drawn")
    t.fill(ava, 64, 0x11)
    step(s, t, "7 trailing bytes", [ps, b"\x01" * 7])
    t.dump(ava, 64, "cleared")
    step(s, t, "record length 4", [ps, rec(0x65, b"", length=4)], fault=True)
    step(s, t, "record length past the segment", [ps, rec(0x65, bytes(16), length=0x100)], fault=True)
    for cid in (0x25, 0x64, 0xA7, 0xFFFFFFFF):
        step(s, t, "id 0x%x" % cid, [ps, rec(cid, bytes(16))], fault=True)
    t.dump(ava, 64, "unchanged by the faulted execs")
    t.delete()
    s.events("end")
    return s


@scenario
def render_pass_no_target():
    """A 0x1a with no attachment and no target size, after 0x1e: Metal
    returns no encoder, and the decoder throws "Failed to create render
    encoder". Then a pass with no attachment but a 4x4 target size, without
    0x1e: a pass with no attachment and no default raster sample count makes
    Metal abort, on its compiler's callback thread once
    renderCommandEncoderWithDescriptor: has returned, so the host dies in the
    wait after the exec's doorbell. That kills the host in every setup, the
    isolate one included, since the abort is inside Metal."""
    s = Scenario("render_pass_no_target", exclusive=True, watchdog_s=90, doc=render_pass_no_target.__doc__)
    s.bring_up()
    t, o = gpu_task(s)
    start(s, t)
    step(s, t, "empty pass after 0x1e", [R.default_raster_samples(1), R.describe_pass({})], fault=True)
    step(s, t, "no attachment, no sample count", [R.describe_pass({}, width=4, height=4)], fault=True)
    step(s, t, "the next exec", [R.default_raster_samples(1), R.describe_pass({}, width=4, height=4)], fault=True)
    t.delete()
    s.events("end")
    return s


def open_encoder(name, doc, segments_fn):
    s = Scenario(name, doc=doc)
    s.bring_up()
    t, o = gpu_task(s)
    a, _, ava = t.target(4, 4)
    b, _, bva = t.target(4, 4)
    start(s, t)
    t.fill(ava, 64, 0x11)
    t.fill(bva, 64, 0x11)
    pa = R.describe_pass({0: dict(tex=a, clear=CLEAR)}, width=4, height=4)
    pb = R.describe_pass({0: dict(tex=b, clear=(1.0, 1.0, 0.0, 1.0))}, width=4, height=4)
    t.exec_segments(segments_fn(pa, pb), label="the exec", fault=True)
    s.fault_state("the exec")
    s.resume_root()
    t.dump(ava, 16, "A row 0")
    t.dump(bva, 16, "B row 0")
    step(s, t, "the task's next exec", [pa], fault=True)
    s.events("end")
    return s


@scenario
def render_segments_open():
    """Render encoders left open by segment flags (section 6.13 items 4 and
    5), each a death: a pass whose segment says continuedByNext with nothing
    after it in the exec, so the encoder is never ended before the commit;
    and a second pass in a segment that does not continue the first while the
    decoder is carried, which opens a new encoder over the open one. Metal
    aborts, and the host dies in every setup, an isolated task's
    included."""
    return [
        open_encoder("render_segments_open_last", render_segments_open.__doc__,
                     lambda pa, pb: [seg(abi.ENC_RENDER, pa, cont_next=1)]),
        open_encoder("render_segments_open_twice", render_segments_open.__doc__,
                     lambda pa, pb: [seg(abi.ENC_RENDER, pa, cont_next=1), seg(abi.ENC_RENDER, pb)]),
    ]


@scenario
def render_pass_unknown_store():
    """A colour attachment stored with Unknown that no 0x66 resolves before
    the encoder ends: what Metal does is locked (render_pass_actions resolves
    one with 0x66)."""
    s = Scenario("render_pass_unknown_store", doc=render_pass_unknown_store.__doc__)
    s.bring_up()
    t, o = gpu_task(s)
    a, _, ava = t.target(4, 4)
    start(s, t)
    t.fill(ava, 64, 0x11)
    step(s, t, "Unknown, never resolved", [R.describe_pass({0: dict(tex=a, clear=CLEAR, store=R.STORE_UNKNOWN)},
                                                           width=4, height=4)])
    t.dump(ava, 64, "target")
    s.events("end")
    return s
