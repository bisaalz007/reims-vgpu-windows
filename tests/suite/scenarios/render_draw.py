"""Render draws 0x00-0x13, 0x1c and 0x1d: each form draws into a texture over
a shared buffer, and each check is hit (docs/testing.md, section 6.16). The
vertex function (fixtures/shaders/render_vbuf.metal) reads positions from
vertex buffer 0 by vertex_id, and its colour from buffer 1 by instance_id,
shifted right half the target per instance; so vertex start, counts,
instances, base instance, base vertex and indices all show in the pixels."""
import struct

from pgtest import abi
from pgtest import render_lib as R
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import rec

TRI, POINT, LINE, LINE_STRIP, TRI_STRIP = 3, 0, 1, 2, 4
U16, U32 = 0, 1
W = H = 8
NPIX = W * H * 4

# Clip-space triangles: A lower left, B upper right, C a band across the top.
A = [(-1.0, -1.0, 0.0, 1.0), (0.0, -1.0, 0.0, 1.0), (-1.0, 1.0, 0.0, 1.0)]
B = [(0.0, 1.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0), (1.0, -1.0, 0.0, 1.0)]
C = [(-1.0, 1.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)]
COLOURS = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0)]
HIGH = 65536  # C sits at vertex 65536: indices 32768.. plus a zero-extended 0x8000


def half(v):
    """An IEEE half, for MTLTriangleTessellationFactorsHalf."""
    return struct.unpack("<H", struct.pack("<e", v))[0]


class DrawTask:
    """Functions, pipelines, a W x H target and every buffer the draws read,
    in a task large enough for a vertex buffer past vertex 65538."""

    def __init__(self, s):
        t = self.t = R.RenderTask(s, pages=160)
        vbuf = t.function("shaders/render_vbuf.metallib")
        color = t.function("shaders/render_color.metallib")
        tess = t.function("shaders/render_tess.metallib")
        self.p = t.pipeline("basic", v=vbuf, f=color)
        self.ptess = t.pipeline("tess", v=tess, f=color)
        self.ptess_idx = t.pipeline("tess_idx", v=tess, f=color)
        self.tex, _, self.va = t.target(W, H)
        npos = (HIGH + 3) * 16
        self.pos, pva = t.buffer(length=(npos + PAGE - 1) // PAGE * PAGE)
        t.write(pva, R.float4s(*(A + B)))
        t.write(pva + HIGH * 16, R.float4s(*C))
        self.col, _ = t.buffer(contents=R.float4s(*COLOURS))
        # 0-5 @0, 1-3 @12, 32768-32770 @20 (offsets 4-byte aligned).
        idx16 = struct.pack("<10H", 0, 1, 2, 3, 4, 5, 1, 2, 3, 0) + struct.pack("<3H", 32768, 32769, 32770)
        self.idx16, _ = t.buffer(contents=idx16)
        self.idx32, _ = t.buffer(contents=struct.pack("<6I", 0, 1, 2, 3, 4, 5))
        # MTLDrawPrimitivesIndirectArguments {vertexCount, instanceCount,
        # vertexStart, baseInstance} @0; MTLDrawIndexedPrimitivesIndirectArguments
        # {indexCount, instanceCount, indexStart, baseVertex, baseInstance} @16;
        # MTLDrawPatchIndirectArguments {patchCount, instanceCount, patchStart,
        # baseInstance} @48.
        args = struct.pack("<4I", 3, 1, 3, 0) + struct.pack("<3IiI", 3, 1, 0, 3, 1) + bytes(12)
        args += struct.pack("<4I", 1, 1, 1, 2)
        self.args, _ = t.buffer(contents=args)
        # Two triangle patches, A's and B's corners, as float2 control points.
        cps = b"".join(struct.pack("<ff", p[0], p[1]) for p in A + B)
        self.cps, _ = t.buffer(contents=cps)
        one = half(1.0)
        self.factors, _ = t.buffer(contents=struct.pack("<4H", one, one, one, one) * 4)
        self.patch_idx, _ = t.buffer(contents=struct.pack("<2I", 1, 0))
        self.cp_idx, _ = t.buffer(contents=struct.pack("<6H", 0, 1, 2, 3, 4, 5))

    def pass_(self):
        return R.describe_pass({0: dict(tex=self.tex, clear=(0.0, 0.0, 0.0, 1.0))}, width=W, height=H)

    def vb(self):
        return R.buffers(0x7D, 0, [(self.pos, 0), (self.col, 0)])

    def tess(self, indexed=False):
        return [R.pipeline_state(self.ptess_idx if indexed else self.ptess), R.buffers(0x7D, 0, [(self.cps, 0)]),
                R.tess_factor_buffer(self.factors, 0, 0)]


def draw_case(s, d, label, draw, fault=False, pipeline=True, tess=None):
    """One exec: the pass, the pipeline and vertex buffers unless told
    otherwise, the draw. A drawn case dumps the target."""
    t = d.t
    if not fault:
        t.fill(d.va, NPIX, 0x11)
    pre = ([] if not pipeline else d.tess(tess == "idx") if tess else [R.pipeline_state(d.p), d.vb()])
    t.render([d.pass_()] + pre + [draw], fault=fault, label=label)
    if fault:
        s.fault_state(label)
        s.resume_root()
    else:
        t.dump(d.va, NPIX, label)


@scenario
def render_draw_primitives():
    """The non-indexed draws 0x00-0x05 and every primitive type 0-4, with
    their checks: no pipeline ("Attempting to draw with nil pipeline
    state"), a primitive type >= 10 ("Invalid primitive type"), a short
    record."""
    s = Scenario("render_draw_primitives", doc=render_draw_primitives.__doc__)
    s.bring_up()
    d = DrawTask(s)
    d.t.define()
    d.t.install()
    cases = [
        ("0x00 A", R.draw64(TRI, 0, 3)),
        ("0x01 B", R.draw16(TRI, 3, 3)),
        ("0x02 A, 2 instances", R.draw_inst64(TRI, 0, 3, 2)),
        ("0x03 B, 2 instances", R.draw_inst16(TRI, 3, 3, 2)),
        ("0x04 A, base instance 2", R.draw_inst_base64(TRI, 0, 3, 1, 2)),
        ("0x05 A, 2 instances from 1", R.draw_inst_base16(TRI, 0, 3, 2, 1)),
        ("0x01 points", R.draw16(POINT, 0, 6)),
        ("0x01 lines", R.draw16(LINE, 0, 6)),
        ("0x01 line strip", R.draw16(LINE_STRIP, 0, 6)),
        ("0x01 triangle strip", R.draw16(TRI_STRIP, 0, 6)),
    ]
    for label, dr in cases:
        draw_case(s, d, label, dr)
    for cmd, dr in ((0x00, R.draw64(TRI, 0, 3)), (0x01, R.draw16(TRI, 0, 3)), (0x02, R.draw_inst64(TRI, 0, 3, 1)),
                    (0x03, R.draw_inst16(TRI, 0, 3, 1)), (0x04, R.draw_inst_base64(TRI, 0, 3, 1, 0)),
                    (0x05, R.draw_inst_base16(TRI, 0, 3, 1, 0))):
        draw_case(s, d, "0x%02x no pipeline" % cmd, dr, fault=True, pipeline=False)
    for cmd, dr in ((0x00, R.draw64(10, 0, 3)), (0x01, R.draw16(10, 0, 3)), (0x02, R.draw_inst64(0xFFFF, 0, 3, 1)),
                    (0x03, R.draw_inst16(10, 0, 3, 1)), (0x04, R.draw_inst_base64(10, 0, 3, 1, 0)),
                    (0x05, R.draw_inst_base16(10, 0, 3, 1, 0))):
        draw_case(s, d, "0x%02x primitive type >= 10" % cmd, dr, fault=True)
    for cmd in range(0x00, 0x06):
        draw_case(s, d, "0x%02x short" % cmd, rec(cmd, b"\x03\x00"), fault=True)
    d.t.delete()
    s.events("end")
    return s


@scenario
def render_draw_indexed():
    """The indexed draws 0x06-0x0b, 0x1c and 0x1d: 16- and 32-bit indices,
    offsets, instances, base vertex and base instance. The two 16-bit forms
    differ: 0x0b loads its 16-bit baseVertex zero-extended and 0x1d
    sign-extended, so 0x8000 over indices 32768.. draws vertex 65536.. (C)
    with 0x0b and vertex 0.. (A) with 0x1d; the 64-bit pair both take -1.
    Checks: no pipeline, index buffer ref 0 ("nil index buffer"), an index
    buffer ref that is a texture, primitive type >= 10, a short record."""
    s = Scenario("render_draw_indexed", doc=render_draw_indexed.__doc__)
    s.bring_up()
    d = DrawTask(s)
    d.t.define()
    d.t.install()
    i16, i32 = d.idx16, d.idx32
    cases = [
        ("0x06 u16 from index 3: B", R.draw_idx64(TRI, U16, i16, 3, 6)),
        ("0x07 u32: A", R.draw_idx16(TRI, U32, i32, 3, 0)),
        ("0x08 u16, 2 instances: A", R.draw_idx_inst64(TRI, U16, i16, 3, 0, 2)),
        ("0x09 u32 from index 3, 2 instances: B", R.draw_idx_inst16(TRI, U32, i32, 3, 12, 2)),
        ("0x0a base vertex 3, base instance 1: B", R.draw_idx_base64(TRI, U16, i16, 0, 3, 1, 3, 1)),
        ("0x0a base vertex -1 over 1, 2, 3: A", R.draw_idx_base64(TRI, U16, i16, 12, 3, 1, -1, 0)),
        ("0x0b base vertex 0x8000 over 32768..: C", R.draw_idx_base16(TRI, U16, i16, 20, 3, 1, 0x8000, 2)),
        ("0x0b base vertex 3: B", R.draw_idx_base16(TRI, U16, i16, 0, 3, 1, 3, 0)),
        ("0x1c base vertex -1 over 1, 2, 3: A", R.draw_idx_base64(TRI, U16, i16, 12, 3, 1, -1, 3, cmd=0x1C)),
        ("0x1d base vertex 0x8000 over 32768..: A", R.draw_idx_base16(TRI, U16, i16, 20, 3, 1, 0x8000, 0, cmd=0x1D)),
        ("0x1d base vertex 3, 2 instances: B", R.draw_idx_base16(TRI, U16, i16, 0, 3, 2, 3, 0, cmd=0x1D)),
    ]
    for label, dr in cases:
        draw_case(s, d, label, dr)
    forms = [
        (0x06, lambda p, ib: R.draw_idx64(p, U16, ib, 3, 0)),
        (0x07, lambda p, ib: R.draw_idx16(p, U16, ib, 3, 0)),
        (0x08, lambda p, ib: R.draw_idx_inst64(p, U16, ib, 3, 0, 1)),
        (0x09, lambda p, ib: R.draw_idx_inst16(p, U16, ib, 3, 0, 1)),
        (0x0A, lambda p, ib: R.draw_idx_base64(p, U16, ib, 0, 3, 1, 0, 0)),
        (0x0B, lambda p, ib: R.draw_idx_base16(p, U16, ib, 0, 3, 1, 0, 0)),
        (0x1C, lambda p, ib: R.draw_idx_base64(p, U16, ib, 0, 3, 1, 0, 0, cmd=0x1C)),
        (0x1D, lambda p, ib: R.draw_idx_base16(p, U16, ib, 0, 3, 1, 0, 0, cmd=0x1D)),
    ]
    for cmd, f in forms:
        draw_case(s, d, "0x%02x no pipeline" % cmd, f(TRI, i16), fault=True, pipeline=False)
        draw_case(s, d, "0x%02x index buffer 0" % cmd, f(TRI, 0), fault=True)
        draw_case(s, d, "0x%02x primitive type >= 10" % cmd, f(10, i16), fault=True)
    for cmd, f in forms[:2]:
        draw_case(s, d, "0x%02x index buffer a texture" % cmd, f(TRI, d.tex), fault=True)
        draw_case(s, d, "0x%02x short" % cmd, rec(cmd, b"\x03\x00"), fault=True)
    d.t.delete()
    s.events("end")
    return s


@scenario
def render_draw_patches():
    """Tessellated draws 0x0c-0x0f with a triangle-patch pipeline, a
    tessellation factor buffer (0x7a) and control points in buffer 0: patch
    start and count, a patch index buffer (ref 0 is nil) and a control-point
    index buffer. Checks: no pipeline, control-point index buffer ref 0 ("nil
    control point index buffer")."""
    s = Scenario("render_draw_patches", doc=render_draw_patches.__doc__)
    s.bring_up()
    d = DrawTask(s)
    d.t.define()
    d.t.install()
    pi, ci = d.patch_idx, d.cp_idx
    cases = [
        ("0x0c patch 0: A", R.draw_patches64(3, 0, 1, 0, 0, 1, 0), "plain"),
        ("0x0c patches 0-1, index buffer [1, 0]", R.draw_patches64(3, 0, 2, pi, 0, 1, 0), "plain"),
        ("0x0d patch 1: B", R.draw_patches16(3, 1, 1, 0, 0, 1, 0), "plain"),
        ("0x0d patch index buffer at 4: [0]", R.draw_patches16(3, 0, 1, pi, 4, 1, 0), "plain"),
        ("0x0e patch 0", R.draw_idx_patches64(3, 0, 1, 0, 0, ci, 0, 1, 0), "idx"),
        ("0x0e patch index buffer [1]", R.draw_idx_patches64(3, 0, 1, pi, 0, ci, 0, 1, 0), "idx"),
        ("0x0f patch 1", R.draw_idx_patches16(3, 1, 1, 0, 0, ci, 0, 1, 0), "idx"),
        ("0x0f patch index buffer [1]", R.draw_idx_patches16(3, 0, 1, pi, 0, ci, 0, 1, 0), "idx"),
    ]
    for label, dr, kind in cases:
        draw_case(s, d, label, dr, tess=kind)
    for cmd, dr in ((0x0C, R.draw_patches64(3, 0, 1, pi, 0, 1, 0)), (0x0D, R.draw_patches16(3, 0, 1, 0, 0, 1, 0)),
                    (0x0E, R.draw_idx_patches64(3, 0, 1, 0, 0, ci, 0, 1, 0)),
                    (0x0F, R.draw_idx_patches16(3, 0, 1, pi, 0, ci, 0, 1, 0))):
        draw_case(s, d, "0x%02x no pipeline" % cmd, dr, fault=True, pipeline=False)
    draw_case(s, d, "0x0e control-point buffer 0", R.draw_idx_patches64(3, 0, 1, 0, 0, 0, 0, 1, 0), fault=True,
              tess="idx")
    draw_case(s, d, "0x0f control-point buffer 0", R.draw_idx_patches16(3, 0, 1, pi, 0, 0, 0, 1, 0), fault=True,
              tess="idx")
    draw_case(s, d, "0x0c patch index buffer a texture", R.draw_patches64(3, 0, 1, d.tex, 0, 1, 0), fault=True,
              tess="plain")
    d.t.delete()
    s.events("end")
    return s


@scenario
def render_draw_indirect():
    """Indirect draws 0x10-0x13: arguments read by the GPU from a shared
    buffer (MTLDraw*IndirectArguments). Checks, in the decoder's order: no
    pipeline, indirect buffer ref 0, index or control-point buffer ref 0,
    primitive type >= 10."""
    s = Scenario("render_draw_indirect", doc=render_draw_indirect.__doc__)
    s.bring_up()
    d = DrawTask(s)
    d.t.define()
    d.t.install()
    a, i16, pi, ci = d.args, d.idx16, d.patch_idx, d.cp_idx
    draw_case(s, d, "0x10 {3, 1, 3, 0}: B", R.draw_indirect(TRI, a, 0))
    draw_case(s, d, "0x11 {3, 1, 0, base vertex 3, base instance 1}: B", R.draw_idx_indirect(TRI, U16, i16, a, 0, 16))
    draw_case(s, d, "0x12 {1, 1, patch 1, base instance 2}", R.draw_patches_indirect(3, 0, a, 0, 48), tess="plain")
    draw_case(s, d, "0x12 patch index buffer", R.draw_patches_indirect(3, pi, a, 0, 48), tess="plain")
    draw_case(s, d, "0x13 {1, 1, patch 1, base instance 2}", R.draw_idx_patches_indirect(3, 0, ci, a, 0, 0, 48),
              tess="idx")
    draw_case(s, d, "0x13 patch index buffer", R.draw_idx_patches_indirect(3, pi, ci, a, 0, 0, 48), tess="idx")
    for label, dr, kw in (
            ("0x10 no pipeline", R.draw_indirect(TRI, a, 0), dict(pipeline=False)),
            ("0x10 indirect buffer 0", R.draw_indirect(TRI, 0, 0), {}),
            ("0x10 primitive type >= 10", R.draw_indirect(10, a, 0), {}),
            ("0x11 no pipeline", R.draw_idx_indirect(TRI, U16, i16, a, 0, 16), dict(pipeline=False)),
            ("0x11 indirect buffer 0", R.draw_idx_indirect(TRI, U16, i16, 0, 0, 16), {}),
            ("0x11 index buffer 0", R.draw_idx_indirect(TRI, U16, 0, a, 0, 16), {}),
            ("0x11 primitive type >= 10", R.draw_idx_indirect(10, U16, i16, a, 0, 16), {}),
            ("0x12 no pipeline", R.draw_patches_indirect(3, 0, a, 0, 48), dict(pipeline=False)),
            ("0x12 indirect buffer 0", R.draw_patches_indirect(3, 0, 0, 0, 48), dict(tess="plain")),
            ("0x13 no pipeline", R.draw_idx_patches_indirect(3, 0, ci, a, 0, 0, 48), dict(pipeline=False)),
            ("0x13 indirect buffer 0", R.draw_idx_patches_indirect(3, 0, ci, 0, 0, 0, 48), dict(tess="idx")),
            ("0x13 control-point buffer 0", R.draw_idx_patches_indirect(3, 0, 0, a, 0, 0, 48), dict(tess="idx")),
            ("0x10 short", rec(0x10, bytes(8)), {}),
            ("0x13 short", rec(0x13, bytes(8)), {})):
        draw_case(s, d, label, dr, fault=True, **kw)
    d.t.delete()
    s.events("end")
    return s


@scenario
def render_draw_odd_types():
    """Values the decoder passes to Metal unchecked: primitive types 5-8 pass
    the decoder's primitive-type check (only >= 10 throws) and draw; no draw
    checks its index type, and index type 2 draws nothing."""
    s = Scenario("render_draw_odd_types", doc=render_draw_odd_types.__doc__)
    s.bring_up()
    d = DrawTask(s)
    d.t.define()
    d.t.install()
    for prim in range(5, 9):
        draw_case(s, d, "0x01 primitive type %d" % prim, R.draw16(prim, 0, 3))
    draw_case(s, d, "0x07 index type 2", R.draw_idx16(TRI, 2, d.idx16, 3, 0))
    draw_case(s, d, "0x07 index type 0xffff", R.draw_idx16(TRI, 0xFFFF, d.idx16, 3, 0))
    d.t.delete()
    s.events("end")
    return s


@scenario
def render_draw_type9():
    """Primitive type 9 passes the decoder's check too, and the GPU does not
    complete the command buffer: Metal reports an error and the exec faults
    (status 3)."""
    s = Scenario("render_draw_type9", exclusive=True, doc=render_draw_type9.__doc__)
    s.bring_up()
    d = DrawTask(s)
    d.t.define()
    d.t.install()
    d.t.render([d.pass_(), R.pipeline_state(d.p), d.vb(), R.draw16(9, 0, 3)], fault=True, timeout_ms=30000,
               label="primitive type 9")
    s.fault_state("primitive type 9")
    s.resume_root()
    draw_case(s, d, "the next draw", R.draw16(TRI, 0, 3))
    d.t.delete()
    s.events("end")
    return s


@scenario
def render_draw_icb():
    """Indirect command buffers executed in a render pass: 0x15 over empty
    ranges of a one-command ICB, and 0x14 with a range read by the GPU from a
    buffer (MTLIndirectCommandBufferExecutionRange {u32 location, u32
    length}), which the GPU fails even when empty; a range over the command
    the guest never encoded makes the command buffer fail on the GPU too:
    both fault. Checks: ICB ref 0 ("Failed to get indirect command buffer"),
    a buffer as the ICB, a range past the ICB's size or whose end overflows
    ("Invalid indirect command buffer range"), range buffer ref 0 ("Failed to
    get indirect range buffer")."""
    s = Scenario("render_draw_icb", exclusive=True, doc=render_draw_icb.__doc__)
    s.bring_up()
    d = DrawTask(s)
    icb = d.t.icb(max_count=1)
    rng, _ = d.t.buffer(contents=struct.pack("<II", 0, 0))
    d.t.define()
    d.t.install()
    draw_case(s, d, "0x15 [0, 0)", R.exec_icb_ranged(icb, 0, 0))
    draw_case(s, d, "0x15 [1, 1)", R.exec_icb_ranged(icb, 1, 0))
    draw_case(s, d, "0x14 range {0, 0} from a buffer: the GPU faults", R.exec_icb(icb, rng, 0), fault=True)
    draw_case(s, d, "0x15 [0, 1): the unencoded command faults the GPU", R.exec_icb_ranged(icb, 0, 1), fault=True)
    draw_case(s, d, "0x15 ICB 0", R.exec_icb_ranged(0, 0, 1), fault=True)
    draw_case(s, d, "0x15 a buffer as the ICB", R.exec_icb_ranged(d.pos, 0, 1), fault=True)
    draw_case(s, d, "0x15 [1, 2)", R.exec_icb_ranged(icb, 1, 1), fault=True)
    draw_case(s, d, "0x15 end overflows", R.exec_icb_ranged(icb, 2**64 - 1, 2), fault=True)
    draw_case(s, d, "0x14 ICB 0", R.exec_icb(0, rng, 0), fault=True)
    draw_case(s, d, "0x14 range buffer 0", R.exec_icb(icb, 0, 0), fault=True)
    d.t.delete()
    s.events("end")
    return s
