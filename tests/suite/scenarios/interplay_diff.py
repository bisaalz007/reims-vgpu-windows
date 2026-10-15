"""Differential checks: two encodings of the same work, run into two guest
regions and compared with cmp (docs/testing.md, sections 4.10, 6.10-6.16).

Each scenario states which encodings are supposed to agree and locks that
they do, byte for byte, so a change that breaks one encoding and not the
other is caught even where the absolute result is a host value. Where the
result is computed rather than reported, one copy is dumped as well.

Unless a docstring says otherwise, a scenario runs in all three setups and
its three expected files agree but for their first lines (the setup's name
and the binding) and the host block calls the blocks setup's events record:
blocks and map run an in-process task, with the legacy and the memory-map
object list managers, isolate runs a task in the service, and none of that
shows in the results. interplay_diff_setups_agree takes no events checkpoint
of its own, so its three files are identical but for their first lines and
the block calls in the blocks file's closing events record (the runner's).
"""
import struct

from pgtest import abi
from pgtest import render_lib as R
from pgtest.abi import PAGE
from pgtest.compute_objects import Blit as B, Compute as C
from pgtest.diff_util import (BLIT, COMPUTE, INFO, K_ADD, K_DOUBLE, K_INC, K_TEXWRITE, Lab, Workload, children,
                              cmp_va, continued, dump_va, merge, one_each, pattern)
from pgtest.exec_objects import info_heap_size, info_ref, rate_map_record
from pgtest.registry import scenario
from pgtest.resources_build import (RGBA8Unorm, b_copy_buf_tex, b_copy_tex_buf, cmd_invalidate, cmd_synchronize,
                                    desc1, heap_buffer_rec, heap_rec, simple_dim, texture_rec)
from pgtest.scenario import Scenario
from pgtest.task import rec, seg


def _work(t, pso_inc, pso_dbl, src, buf):
    """Workload's steps into buf at 0 (merged by encoder), then double_gid
    over 64 u32s at buf+0x400 and a blit copy of src's first 0x100 bytes to
    buf+0x600."""
    return (merge(Workload(pso_inc).steps(buf))
            + [seg(COMPUTE, C.set_pipeline(pso_dbl) + C.set_buffers(0, [(buf, 0x400)])
                   + C.dispatch_threads((64, 1, 1), (64, 1, 1))),
               seg(BLIT, B.copy_buffer_to_buffer(src, buf, 0, 0x600, 0x100))])


@scenario
def interplay_diff_exec_opcodes():
    """ExecIndirect2 (55) and ExecIndirect3 (67) run the same segments to
    the same bytes: the Workload steps, a double_gid dispatch and a copy,
    into a shared buffer each. Their resource records convert to the same
    flags: a private buffer listed with 55's bytes {0, 1, 1}, with {0, 3, 3}
    (each byte cut to its low bit), and with 67's flags 6 is synchronized
    back to the same bytes. InfoIndirect (69) decodes the same segments and
    never commits them: its buffer equals one nothing touched."""
    s = Scenario("interplay_diff_exec_opcodes", doc=interplay_diff_exec_opcodes.__doc__)
    s.bring_up()
    t = Lab(s)
    inc_, dbl = t.kernel(K_INC), t.kernel(K_DOUBLE)
    src, _ = t.shared(contents=pattern(0x100))
    o55, v55 = t.shared()
    o67, v67 = t.shared()
    o69, v69 = t.shared()
    _, vnone = t.shared()
    p55, vp55 = t.shared(flags=1, listed=False)
    p55b, vp55b = t.shared(flags=1, listed=False)
    p67, vp67 = t.shared(flags=1, listed=False)
    t.ready()
    t.run(_work(t, inc_, dbl, src, o55), opcode=abi.OP_EXEC_INDIRECT2, label="55")
    t.run(_work(t, inc_, dbl, src, o67), label="67")
    t.run(_work(t, inc_, dbl, src, o69), opcode=abi.OP_INFO_INDIRECT, label="69")
    t.run(_work(t, inc_, dbl, src, p55), opcode=abi.OP_EXEC_INDIRECT2,
          resources=t.resources([(p55, 0, 1, 1)]), label="55 private {0, 1, 1}")
    t.run(_work(t, inc_, dbl, src, p55b), opcode=abi.OP_EXEC_INDIRECT2,
          resources=t.resources([(p55b, 0, 3, 3)]), label="55 private {0, 3, 3}")
    t.run(_work(t, inc_, dbl, src, p67), resources=t.resources([(p67, 6)]), label="67 private flags 6")
    t.dump(v67, 0x700, "67's buffer")
    s.cmp(t.gpa(v55), t.gpa(v67), 0x800, label="55 == 67")
    s.cmp(t.gpa(v69), t.gpa(vnone), 0x800, label="69 == untouched")
    s.cmp(t.gpa(vp55), t.gpa(v67), 0x800, label="55 private {0, 1, 1} == 67")
    s.cmp(t.gpa(vp55b), t.gpa(v67), 0x800, label="55 private {0, 3, 3} == 67")
    s.cmp(t.gpa(vp67), t.gpa(v67), 0x800, label="67 private flags 6 == 67")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",))
def interplay_diff_info_opcodes():
    """The same info records (450 compute pipeline info, 451 heap texture
    size and align, 452 rate map info, 461 buffer host info, 467 compute
    pipeline host info) through 67, 55 and 69, as one segment, one segment
    a record, and a chain of continued info segments: every encoding writes
    the same reply bytes, host values as they are. 450's reply is packed
    and 4-aligned, 0x19 bytes of fields and 3 of tail padding, zeros, which
    are never compared (69 decodes and never commits). Isolate only:
    in-process the first reply write aborts the host."""
    s = Scenario("interplay_diff_info_opcodes", setups=("isolate",), doc=interplay_diff_info_opcodes.__doc__)
    s.bring_up()
    t = Lab(s)
    pso = t.kernel(K_DOUBLE)
    data, _ = t.shared(contents=pattern(0x100))
    reply, rva = t.shared()
    rmap = t.add(abi.T_SERIALIZED, rate_map_record(w=256, h=256, samples=[([0.25, 1.0], [0.25, 1.0])]))
    t.ready()

    def records(base):
        return [info_ref(450, pso, reply, base + 0x00),
                info_heap_size(reply, base + 0x20, fmt=70, w=256, h=256, usage=1, options=0x20),
                rec(452, struct.pack("<IIQQ", rmap, reply, base + 0x40, 0)),
                info_ref(461, data, reply, base + 0x60),
                info_ref(467, pso, reply, base + 0x70)]

    cases = [("67 one segment", abi.OP_EXEC_INDIRECT3, lambda r: [seg(INFO, b"".join(r))]),
             ("55 one segment", abi.OP_EXEC_INDIRECT2, lambda r: [seg(INFO, b"".join(r))]),
             ("69 one segment", abi.OP_INFO_INDIRECT, lambda r: [seg(INFO, b"".join(r))]),
             ("67 a segment each", abi.OP_EXEC_INDIRECT3, lambda r: [seg(INFO, x) for x in r]),
             ("69 a segment each", abi.OP_INFO_INDIRECT, lambda r: [seg(INFO, x) for x in r]),
             ("67 continued", abi.OP_EXEC_INDIRECT3, lambda r: continued([(INFO, x) for x in r]))]
    for i, (label, op, enc) in enumerate(cases):
        t.run(enc(records(0x80 * i)), opcode=op, resources=[], label=label)
    t.dump(rva, 0x19, "67 one segment: 450")
    t.dump(rva + 0x1C, 0x64, "67 one segment: 451, 452, 461, 467")
    for i, (label, _, _) in enumerate(cases[1:], 1):
        s.cmp(t.gpa(rva), t.gpa(rva + 0x80 * i), 0x19, label="%s == 67 one segment: 450" % label)
        s.cmp(t.gpa(rva + 0x1C), t.gpa(rva + 0x80 * i + 0x1C), 0x64, label="%s == 67 one segment: the rest" % label)
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_diff_segment_split():
    """The Workload (inc_k dispatches and blit copies whose results depend
    on their order) in six encodings, each into its own buffer: one segment
    per run of an encoder type; one segment per step; runs as chains of
    continued segments (continuesPrevious/continuedByNext, one encoder, the
    pipeline set once); each step its own exec, rung together; the steps
    split across two execs mid-run; and the first encoding on a child FIFO.
    All six agree (each range holds one segment, docs/testing.md section
    6.12)."""
    s = Scenario("interplay_diff_segment_split", doc=interplay_diff_segment_split.__doc__)
    s.bring_up()
    (c1,) = children(s, 1)
    t = Lab(s)
    pso = t.kernel(K_INC)
    bufs = [t.shared() for _ in range(6)]
    t.ready()
    w = Workload(pso)
    (b0, v0), (b1, _), (b2, _), (b3, _), (b4, _), (b5, _) = bufs
    t.run(merge(w.steps(b0)), label="a segment per run")
    t.run(one_each(w.steps(b1)), label="a segment per step")
    t.run(continued(w.steps(b2)), label="continued runs")
    last = None
    t.reset_cmdbufs()
    for step in w.steps(b3):
        last = t.cmd(one_each([step]))
    s.doorbell()
    s.fifo.wait_stamp(last, label="an exec per step")
    st = w.steps(b4)
    t.cmd(merge(st[:2]))
    last = t.cmd(merge(st[2:]))
    s.doorbell()
    s.fifo.wait_stamp(last, label="two execs, split mid-run")
    t.run(merge(w.steps(b5)), fifo=c1, label="a segment per run, child FIFO 1")
    t.dump(v0, 0x340, "a segment per run")
    for i, label in enumerate(("a segment per step", "continued runs", "an exec per step",
                               "two execs, split mid-run", "child FIFO 1"), 1):
        s.cmp(t.gpa(v0), t.gpa(bufs[i][1]), 0x400, label="%s == a segment per run" % label)
    t.delete()
    s.events("end")
    return s


def _depth_task(s, tid, depth, **kw):
    """A task with the same objects at the same references and the same
    contents, whatever its page table."""
    t = Lab(s, tid=tid, depth=depth, **kw)
    t.pso_inc, t.pso_dbl, t.pso_add = t.kernel(K_INC), t.kernel(K_DOUBLE), t.kernel(K_ADD)
    t.src, t.src_va = t.shared(4 * PAGE, contents=pattern(4 * PAGE, 13, 1))
    t.dst, t.dst_va = t.shared(4 * PAGE)
    t.out, t.out_va = t.shared()
    t.ready()
    return t


def _depth_work(t):
    """The Workload into out, double_gid, a copy from src; then add_one over
    4 pages of src into dst (dst = src + 1 a u32) and a blit copy across
    page boundaries (src 0x2000..0xa000 to dst 0x3000 on)."""
    t.run(_work(t, t.pso_inc, t.pso_dbl, t.src, t.out), label="task %d: workload" % t.tid)
    t.run([seg(COMPUTE, C.set_pipeline(t.pso_add) + C.set_buffers(0, [(t.dst, 0), (t.src, 0)])
               + C.dispatch_threads((PAGE, 1, 1), (64, 1, 1))),
           seg(BLIT, B.copy_buffer_to_buffer(t.src, t.dst, 0x2000, 0x3000, 0x8000))],
          label="task %d: across pages" % t.tid)


@scenario
def interplay_diff_page_table_depth():
    """The same objects, data and work in three tasks whose page tables have
    1, 2 and 3 levels (the service walks its own): the results agree byte
    for byte."""
    s = Scenario("interplay_diff_page_table_depth", doc=interplay_diff_page_table_depth.__doc__)
    s.bring_up()
    ts = [_depth_task(s, tid, depth) for tid, depth in ((1, 1), (2, 2), (3, 3))]
    for t in ts:
        _depth_work(t)
    a = ts[0]
    a.dump(a.out_va, 0x700, "depth 1: workload")
    dump_va(a, a.dst_va, 4 * PAGE, "depth 1: dst")
    for t in ts[1:]:
        cmp_va(s, a, a.out_va, t, t.out_va, 0x800, "depth %d workload == depth 1" % t.depth)
        cmp_va(s, a, a.dst_va, t, t.dst_va, 4 * PAGE, "depth %d dst == depth 1" % t.depth)
    for t in ts:
        t.delete()
    s.events("end")
    return s


@scenario
def interplay_diff_page_table_high():
    """The work of interplay_diff_page_table_depth with its buffers at task
    VA 64 MiB and above, in a 2-level table: the VA page number's top
    12-bit index is 1 there, not 0, so the walk takes the second entry of
    the root table (section 6.10: indices split top-down). Only VA pages
    0-15 and 4096-4111 are mapped; the rest are holes nothing touches. The
    results equal a 1-level task's."""
    s = Scenario("interplay_diff_page_table_high", doc=interplay_diff_page_table_high.__doc__)
    s.bring_up()
    a = _depth_task(s, 1, 1)
    hi = 4096
    t = Lab(s, tid=2, depth=2, pages=hi + 16, mapped=list(range(16)) + list(range(hi, hi + 16)))
    t.pso_inc, t.pso_dbl, t.pso_add = t.kernel(K_INC), t.kernel(K_DOUBLE), t.kernel(K_ADD)
    t.src, t.src_va = t.shared(4 * PAGE, contents=pattern(4 * PAGE, 13, 1), va=hi * PAGE)
    t.dst, t.dst_va = t.shared(4 * PAGE, va=(hi + 4) * PAGE)
    t.out, t.out_va = t.shared(va=(hi + 8) * PAGE)
    t.ready()
    for x in (a, t):
        _depth_work(x)
    cmp_va(s, a, a.out_va, t, t.out_va, 0x800, "VA 64 MiB workload == depth 1")
    cmp_va(s, a, a.dst_va, t, t.dst_va, 4 * PAGE, "VA 64 MiB dst == depth 1")
    a.dump(a.out_va, 0x100, "depth 1: double_gid")
    for x in (a, t):
        x.delete()
    s.events("end")
    return s


@scenario
def interplay_diff_fragmented_pages():
    """The same work in three tasks whose data pages lie differently in
    guest RAM: one after another, one page apart each (no two adjacent), and
    in reverse order. Buffers of four pages then span guest pages that are
    not contiguous, and a copy and a dispatch cross their boundaries; the
    results agree byte for byte (in-process through the task's mapping; in
    the service through the shared ranges)."""
    s = Scenario("interplay_diff_fragmented_pages", doc=interplay_diff_fragmented_pages.__doc__)
    s.bring_up()
    a = _depth_task(s, 1, 1)
    b = _depth_task(s, 2, 1, scatter=True)
    first = s.alloc_page(64)
    c = _depth_task(s, 3, 1, leaves={v: first + 63 - v for v in range(64)})
    for t in (a, b, c):
        _depth_work(t)
    dump_va(a, a.dst_va, 4 * PAGE, "contiguous: dst")
    for t, name in ((b, "scattered"), (c, "reversed")):
        cmp_va(s, a, a.out_va, t, t.out_va, 0x800, "%s workload == contiguous" % name)
        cmp_va(s, a, a.dst_va, t, t.dst_va, 4 * PAGE, "%s dst == contiguous" % name)
    for t in (a, b, c):
        t.delete()
    s.events("end")
    return s


@scenario
def interplay_diff_object_lists():
    """One set of objects, and the same work, in four tasks whose object
    lists are set differently: from the task root's fields at define time
    (the manager's setObjectList), with SetObjectList (51), with
    SetObjectAndPlacementList (68) and a placement list of 16 bytes an
    entry, and with 51 over a list whose first three entries are extra
    buffers, so every object sits three references later. The results agree
    byte for byte. Per setup the manager differs too: legacy in blocks,
    memory-map in map and isolate."""
    s = Scenario("interplay_diff_object_lists", doc=interplay_diff_object_lists.__doc__)
    s.bring_up()
    ts = []
    for tid, how in ((1, "root"), (2, "51"), (3, "68"), (4, "shifted")):
        t = Lab(s, tid=tid)
        if how == "shifted":
            for _ in range(3):
                t.shared()
            t.listed = []
        t.pso_inc, t.pso_dbl = t.kernel(K_INC), t.kernel(K_DOUBLE)
        t.src, _ = t.shared(contents=pattern(0x100, 5, 9))
        t.out, t.out_va = t.shared()
        t.how = how
        if how == "root":
            t.write_root(obj_off=0, obj_len=t.ol.write())
            t.define()
        elif how == "68":
            n = len(t.ol.entries)
            place = t.reserve(PAGE)
            t.ol.write()
            t.define()
            t.set_object_and_placement_list(0, place, t.ol.length(), 16 * n)
        else:
            t.ready()
        ts.append(t)
    for t in ts:
        t.run(_work(t, t.pso_inc, t.pso_dbl, t.src, t.out), label="list from %s" % t.how)
    a = ts[0]
    a.dump(a.out_va, 0x700, "list from the task root")
    for t in ts[1:]:
        cmp_va(s, a, a.out_va, t, t.out_va, 0x800, "list from %s == from the task root" % t.how)
    for t in ts:
        t.delete()
    s.events("end")
    return s


@scenario
def interplay_diff_heap_buffer():
    """The same bytes as a buffer (type 1) and as a buffer placed in a
    private heap (types 13 and 14, at offset 0x4000 of a four-page heap):
    a blit copy out of each, and a fill into each followed by an exec that
    lists the heap and the buffer with invalidateGuest + synchronize
    (flags 6), give the same bytes, both in a shared result buffer and in
    the guest pages behind them."""
    s = Scenario("interplay_diff_heap_buffer", doc=interplay_diff_heap_buffer.__doc__)
    s.bring_up()
    t = Lab(s, pages=80)
    data = pattern(PAGE, 11, 5)
    plain, pva = t.shared(contents=data, listed=False)
    hva = t.reserve(4 * PAGE)
    t.write(hva + PAGE, data)
    heap = t.add(abi.T_HEAP, heap_rec(hva // PAGE, 4 * PAGE, flags=1))
    hb = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(heap, PAGE, PAGE))
    out, ova = t.shared()
    t.ready()
    t.run([seg(BLIT, B.copy_buffer_to_buffer(plain, out, 0, 0, 0x400))],
          resources=t.resources([(plain, 0)]), label="copy out of the buffer")
    t.run([seg(BLIT, B.copy_buffer_to_buffer(hb, out, 0, 0x400, 0x400))],
          resources=t.resources([(heap, 0), (hb, 0)]), label="copy out of the heap buffer")
    def fill(b, at):
        return seg(BLIT, B.fill(b, 0x100, 0x80, 0xC4) + B.copy_buffer_to_buffer(b, out, 0x80, at, 0x100))

    t.run([fill(plain, 0x800)], resources=t.resources([(plain, 6)]), label="fill the buffer, flags 6")
    t.run([fill(hb, 0x900)], resources=t.resources([(heap, 6), (hb, 0)]), label="fill the heap buffer, heap flags 6")
    t.dump(ova + 0x800, 0x100, "the buffer after its fill, copied out")
    s.cmp(t.gpa(ova), t.gpa(ova + 0x400), 0x400, label="copies: heap buffer == buffer")
    s.cmp(t.gpa(ova + 0x800), t.gpa(ova + 0x900), 0x100, label="after the fills: heap buffer == buffer")
    s.cmp(t.gpa(pva), t.gpa(hva + PAGE), PAGE, label="guest pages: heap buffer == buffer")
    t.delete()
    s.events("end")
    return s


def _tex(t, w, h):
    """A private RGBA8Unorm type-2 texture over a fresh task page, usage
    read and write."""
    va = t.reserve(PAGE)
    return t.add(abi.T_TEXTURE, texture_rec(PAGE, va // PAGE, [simple_dim(w, h)], desc1(fmt=RGBA8Unorm, w=w, h=h))), va


@scenario
def interplay_diff_texture_upload():
    """A 16x16 RGBA8Unorm texture (type 2) written by the tex_write kernel
    ({x, y, 0x80, 0xff} a texel), and another uploaded by a blit copy from
    a buffer holding those bytes (300); both listed with invalidateGuest +
    synchronize (flags 6), so each is copied back into its guest pages.
    The pages agree, and so do copies of both textures into a shared
    buffer (302)."""
    s = Scenario("interplay_diff_texture_upload", doc=interplay_diff_texture_upload.__doc__)
    s.bring_up()
    t = Lab(s)
    w = h = 16
    texels = b"".join(bytes([x, y, 0x80, 0xFF]) for y in range(h) for x in range(w))
    pso = t.kernel(K_TEXWRITE)
    up, upva = t.shared(contents=texels)
    out, ova = t.shared()
    a, ava = _tex(t, w, h)
    b, bva = _tex(t, w, h)
    t.ready()
    t.run([seg(COMPUTE, C.set_pipeline(pso) + C.set_textures(0, [a]) + C.dispatch_threads((w, h, 1), (8, 8, 1)))],
          resources=t.resources([(a, 6)]), label="written by compute")
    t.run([seg(BLIT, b_copy_buf_tex(up, b, 0, 4 * w, 4 * w * h, (w, h, 1)))],
          resources=t.resources([(b, 6)]), label="uploaded by blit")
    t.run([seg(BLIT, b_copy_tex_buf(a, out, 0, 4 * w, 4 * w * h, (w, h, 1))
               + b_copy_tex_buf(b, out, 0x400, 4 * w, 4 * w * h, (w, h, 1)))],
          resources=t.resources([(a, 0), (b, 0)]), label="both copied out")
    t.dump(ava, 0x400, "the computed texture's pages")
    s.cmp(t.gpa(ava), t.gpa(bva), 0x400, label="pages: blit == compute")
    s.cmp(t.gpa(ova), t.gpa(ova + 0x400), 0x400, label="copies: blit == compute")
    s.cmp(t.gpa(ava), t.gpa(upva), 0x400, label="pages == the texels the blit uploaded")
    t.delete()
    s.events("end")
    return s


# Clip-space triangles: A the lower left half, C a band across the top that
# overlaps A, so which is drawn last shows (render_draw.py's A and C).
TRI_A = [(-1.0, -1.0, 0.0, 1.0), (1.0, -1.0, 0.0, 1.0), (-1.0, 1.0, 0.0, 1.0)]
TRI_C = [(-1.0, 1.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0), (0.0, -0.5, 0.0, 1.0)]
COLOURS = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)]


@scenario
def interplay_diff_render_split():
    """Two overlapping triangles drawn into an 8x8 RGBA8 texture over a
    shared buffer (type 8, commandType 9), in three encodings: one pass
    (clear, A, C, store); two passes in two execs, the second loading what
    the first stored; the same two passes as two segments of one exec. The
    targets agree byte for byte."""
    s = Scenario("interplay_diff_render_split", doc=interplay_diff_render_split.__doc__)
    s.bring_up()
    t = Lab(s)
    v, f = t.function("shaders/render_vbuf.metallib"), t.function("shaders/render_color.metallib")
    blob = R.pipeline_blob("basic", v, f)
    pso = t.add(abi.T_SERIALIZED, struct.pack("<IIII", 14, 0, 0, len(blob)) + blob)
    pos, _ = t.shared(contents=R.float4s(*(TRI_A + TRI_C)))
    col, _ = t.shared(contents=R.float4s(*COLOURS))
    targets = []
    for _ in range(3):
        bref, va = t.shared()
        tex = t.add(abi.T_SERIALIZER_TEXTURE, R.op_header(9) + struct.pack("<IIQQ", 0, bref, 0, 32)
                    + R.tex_desc(8, 8, usage=R.USAGE_READ | R.USAGE_RT))
        targets.append((tex, va))
    t.ready()

    def pass_(tex, load):
        return R.describe_pass({0: dict(tex=tex, load=load, clear=(0.0, 0.0, 0.0, 1.0))}, width=8, height=8)

    def draw(start, colour):
        return (R.pipeline_state(pso) + R.buffers(0x7D, 0, [(pos, 0), (col, 16 * colour)])
                + R.draw16(3, start, 3))

    (t1, v1), (t2, v2), (t3, v3) = targets
    t.run([seg(abi.ENC_RENDER, pass_(t1, R.LOAD_CLEAR) + draw(0, 0) + draw(3, 1))], label="one pass")
    t.run([seg(abi.ENC_RENDER, pass_(t2, R.LOAD_CLEAR) + draw(0, 0))], label="two execs: first pass")
    t.run([seg(abi.ENC_RENDER, pass_(t2, R.LOAD_LOAD) + draw(3, 1))], label="two execs: second pass, load")
    t.run([seg(abi.ENC_RENDER, pass_(t3, R.LOAD_CLEAR) + draw(0, 0)),
           seg(abi.ENC_RENDER, pass_(t3, R.LOAD_LOAD) + draw(3, 1))], label="one exec, two passes")
    t.dump(v1, 0x100, "one pass")
    s.cmp(t.gpa(v1), t.gpa(v2), 0x100, label="two execs == one pass")
    s.cmp(t.gpa(v1), t.gpa(v3), 0x100, label="one exec, two passes == one pass")
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_diff_setups_agree():
    """A mixed workload whose three expected files (blocks, map, isolate)
    are identical but for their first lines (the setup) and the host block
    calls the blocks file's closing events record lists: the Workload
    through 55, 67 and a child FIFO; add_one over four pages; a texture
    written by compute and synchronized into its pages; a private buffer
    synchronized by 52 and 53; a task deleted and its ID defined again. So a
    divergence between an in-process task with either object list manager
    and a task in the service shows as a difference between the files, not
    only between encodings."""
    s = Scenario("interplay_diff_setups_agree", doc=interplay_diff_setups_agree.__doc__)
    s.bring_up()
    (c1,) = children(s, 1)
    t = _depth_task(s, 1, 2)
    u = Lab(s, tid=2)
    pso_tw = u.kernel(K_TEXWRITE)
    tex, tva = _tex(u, 8, 8)
    priv, pva = u.shared(flags=1, listed=False)
    u.ready()
    _depth_work(t)
    t.run(merge(Workload(t.pso_inc).steps(t.out, 0x800)), opcode=abi.OP_EXEC_INDIRECT2, label="55")
    t.run(merge(Workload(t.pso_inc).steps(t.out, 0xC00)), fifo=c1, label="child FIFO 1")
    u.run([seg(COMPUTE, C.set_pipeline(pso_tw) + C.set_textures(0, [tex]) + C.dispatch_threads((8, 8, 1), (8, 8, 1))),
           seg(BLIT, B.fill(priv, 0, 0x40, 0x3C))], resources=u.resources([(tex, 6), (priv, 0)]),
          label="texture and private buffer")
    cmd_invalidate(s, 2, [(priv, 0, 0, 1, 0)], label="52: the private buffer's guest copy invalid")
    cmd_synchronize(s, 2, [priv], label="53")
    t.dump(t.out_va, 0x1000, "workloads")
    dump_va(t, t.dst_va, 4 * PAGE, "dst")
    u.dump(tva, 0x100, "texture pages")
    u.dump(pva, 0x80, "private buffer pages")
    u.delete()
    u2 = Lab(s, tid=2)
    pso2 = u2.kernel(K_INC)
    b2, bva2 = u2.shared()
    u2.ready()
    u2.run(merge(Workload(pso2).steps(b2)), label="task 2 defined again")
    cmp_va(s, t, t.out_va, u2, bva2, 0x400, "task 2 again == task 1")
    u2.delete()
    t.delete()
    s.submit(abi.OP_NOP, label="the end")
    s.r(abi.R_FAULT_MASK, label="the end")
    return s
