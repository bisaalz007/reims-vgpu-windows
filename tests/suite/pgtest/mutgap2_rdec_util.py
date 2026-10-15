"""Builders for scenarios/mutgap2_rdec.py: depth and stencil resolves read
back through later passes, and an indirect command buffer whose draw reads
its buffers only through the ICB (so only residency makes them visible).
"""
import struct

from . import abi
from . import render_lib as R
from .abi import PAGE
from .exec_objects import ICB_DRAW, icb_descriptor, icb_record, render_blob
from .icb_cmds import CMD, layout, render_cmd

W = H = 8
NPIX = W * H * 4
TRI = 3
LESS = 1
RED = (1.0, 0.0, 0.0, 1.0)


def amp_target(t):
    """An 8x8 two-slice 2D array RGBA8 texture over a shared buffer
    (type 8, commandType 9, section 6.19). Returns (texture ref, VA)."""
    tex, _, va = t.target(W, H, ttype=R.TEX_2D_ARRAY, array=2)
    return tex, va


class Resolve:
    """A StateTask with a 4-sample memoryless colour and depth-stencil
    attachment (ms, ds_ms) and two 1-sample private D32S8 textures (P, Q),
    each at offset 0 of its own private heap (type 8, commandType 21), as
    mutgap_dec_render_stencil_resolve builds them."""

    def __init__(self, s):
        from scenarios.mutgap_decoders import private_ds, stencil_equal
        from scenarios.render_state import StateTask
        self.s = s
        st = self.st = StateTask(s)
        t = st.t
        self.ms = t.memoryless(W, H, R.RGBA8, ttype=R.TEX_2D_MS, samples=4)
        self.ds_ms = t.memoryless(W, H, R.D32S8, ttype=R.TEX_2D_MS, samples=4)
        self.P = private_ds(st)
        self.Q = private_ds(st)
        self.less = t.depth_stencil(compare=LESS, write=0)
        self.sten = stencil_equal(st)
        st.start()

    def _pass(self, label, records, pass_kw):
        from scenarios.render_state import case
        case(self.s, self.st, label, records, depth=True, dump=True, pass_kw=pass_kw)

    def prestore(self):
        """Depth 0.3 and stencil 5 stored into P and Q (clear and store)."""
        from scenarios.render_state import case
        for x, name in ((self.P, "P"), (self.Q, "Q")):
            case(self.s, self.st, "store depth 0.3, stencil 5 in " + name, [], depth=True, dump=False,
                 pass_kw=dict(depth=dict(tex=x, load=R.LOAD_CLEAR, store=R.STORE_STORE, clear=0.3),
                              stencil=dict(tex=x, load=R.LOAD_CLEAR, store=R.STORE_STORE, clear=5)))

    def run(self, label, depth=None, stencil=None, records=(), check=()):
        """P and Q prestored; then a pass over ms and ds_ms clearing depth to
        1.0 and stencil to 3 (each stored DontCare unless depth/stencil
        say otherwise) with records; then, for each (kind, texture) in
        check, a pass that loads the texture and draws red at depth 0.5
        with depth Less (kind "depth") or with stencil Equal 3 ("stencil")."""
        s, st = self.s, self.st
        self.prestore()
        d = dict(tex=self.ds_ms, load=R.LOAD_CLEAR, store=R.STORE_DONTCARE, clear=1.0)
        d.update(depth or {})
        sc = dict(tex=self.ds_ms, load=R.LOAD_CLEAR, store=R.STORE_DONTCARE, clear=3)
        sc.update(stencil or {})
        st.t.render([R.describe_pass({0: dict(tex=self.ms, store=R.STORE_DONTCARE, clear=(0.0, 0.0, 1.0, 1.0))},
                                     width=W, height=H, depth=d, stencil=sc)] + list(records), label=label)
        s.fault_state(label)
        names = {self.P: "P", self.Q: "Q"}
        for kind, x in check:
            loaded = dict(depth=dict(tex=x, load=R.LOAD_LOAD, store=R.STORE_DONTCARE, clear=1.0),
                          stencil=dict(tex=x, load=R.LOAD_LOAD, store=R.STORE_DONTCARE, clear=0))
            draw = [R.buffers(0x7D, 0, [(st.params, 0)]), R.draw16(TRI, 0, 3)]
            if kind == "depth":
                recs = [R.pipeline_state(st.p_depth), R.depth_stencil_state(self.less)] + draw
                what = "red at 0.5 with depth Less over %s's depth" % names[x]
            else:
                recs = [R.pipeline_state(st.p_depth), R.depth_stencil_state(self.sten), R.stencil_ref(3, 3)] + draw
                what = "red with stencil Equal 3 over %s's stencil" % names[x]
            self._pass("%s: %s" % (label, what), recs, loaded)


class Resident:
    """A render task whose ICBs (type 20, one draw command each in the
    icb_cmds RENDER layout) draw with an ICB-capable render_vbuf/render_color
    pipeline (exec_objects.render_blob, icb) and bind their two vertex
    buffers in the command: icb_buf binds shared buffers pos (the
    full-target triangle) and col (red); icb_heap binds one buffer placed in
    a private heap (type 14 in type 13) holding the same positions at 0
    and colour at 0x100, whose bytes come from the heap's guest pages
    (resources_heap_paging). The target is 8x8 over a shared buffer.
    Everything is listed in every exec (ICBs are prepared from guest memory
    when listed, section 6.19)."""

    def __init__(self, s):
        self.s = s
        t = self.t = R.RenderTask(s)
        vb = t.function("shaders/render_vbuf.metallib")
        cf = t.function("shaders/render_color.metallib")
        self.pso = t.pipeline("icb", blob=render_blob(vb, cf, icb=True))
        self.tex, _, self.va = t.target(W, H)
        self.pos, _ = t.buffer(contents=R.float4s(*R.FULL))
        self.col, _ = t.buffer(contents=R.float4s(RED))
        self.other, _ = t.buffer()
        # Two private heaps, each with one buffer holding the positions at 0
        # and the colour at 0x100. Type 13 {u32 page, u64 length, u8 flags:
        # private}, read as 16 bytes; type 14 {u32 heap, u64 offset, u64
        # length}.
        self.heaps, self.icb_heaps = [], []
        for _ in range(2):
            hva = t.reserve(PAGE)
            t.write(hva, R.float4s(*R.FULL))
            t.write(hva + 0x100, R.float4s(RED))
            heap = t.ol.add(abi.T_HEAP, struct.pack("<IQB3x", hva // PAGE, PAGE, 1))
            hbuf = t.ol.add(abi.T_HEAP_BUFFER, struct.pack("<IQQ", heap, 0, PAGE))
            t.bufs += [heap, hbuf]
            self.heaps.append(heap)
            self.icb_heaps.append(self.icb([(hbuf, 0, 0), (hbuf, 0, 0x100)]))
        self.icb_buf = self.icb([(self.pos, 0, 0), (self.col, 0, 0)])
        t.define()
        t.install()

    def icb(self, vbinds):
        t = self.t
        va = t.reserve(PAGE)
        t.write(va, render_cmd(pso=self.pso, vbinds=vbinds))
        ref = t.ol.add(abi.T_ICB, icb_record(CMD, va // PAGE, icb_descriptor(ICB_DRAW, vbufs=2), layout("render")))
        t.bufs.append(ref)
        return ref

    def run(self, label, records, icb=None):
        """A pass clearing the target black, records, then 0x15 over the
        ICB's command 0; dumps the target."""
        t = self.t
        t.fill(self.va, NPIX, 0x11)
        t.render([R.describe_pass({0: dict(tex=self.tex, clear=(0.0, 0.0, 0.0, 1.0))}, width=W, height=H)]
                 + list(records) + [R.exec_icb_ranged(self.icb_buf if icb is None else icb, 0, 1)], label=label)
        self.s.fault_state(label)
        t.dump(self.va, NPIX, label)
