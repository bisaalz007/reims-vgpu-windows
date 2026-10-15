"""Builders for the display pipeline interplay scenarios
(scenarios/interplay_scanout.py): the guest renders with the GPU into
surfaces that become display frames, and presents them while the host and
the guest change the displays around it.

A Scanout is a Session (pgtest/session.py) with what a compositor adds:
- mapper surfaces: IOSurfaces the IOSurface host device maps over guest
  pages, and in a GPU task a mapper texture (type 11) and a mapper buffer
  (type 15) over each, so that render, blit and compute work lands in the
  surface, which is also the surface opcode 8 presents;
- ref textures (type 5) over task 0's IOSurface backings, the surfaces
  opcodes 6 and 7 present; in-process only, since a type 5 texture aborts
  the service (docs/testing.md, section 7.4);
- presents and execs written on any FIFO, gated by barriers on other
  FIFOs' stamps, so that the order of GPU writes and presents is the one
  the guest established, not timing;
- a gate: a child FIFO whose NOPs release what waits on its stamp;
- the expected ack a port's guest display will take next, which the guest
  tracks from the interrupts it saw (4 and 12 each bump +0x200).

Frames: the IOSurface is not copied at a present; the display's frame is a
texture view of it, so what an encode shows is the surface as it is at the
encode. The expected files record the surface's guest page with a dump (the
GPU's writes, computed by committed shaders and exact clear colours) and
the frame with an encode (the library's shaders draw it: host-keyed, rule 5).

Layouts are the ones the per-area builders give.
"""
import struct

from . import abi
from . import render_lib as R
from .abi import PAGE
from .display_guest import (SS_CURSOR_POS, SS_GUEST_CURSOR_SHOW, SS_MODE_COUNT, display, gamma_words, p_present3,
                            p_props, p_show, p_swap)
from .exec_objects import pipeline_record, render_blob
from .fifo import ChildFIFO
from .fifo_util import HEALTHY, NODE_FAULT
from .iosfc_guest import CMD_MAP, CMD_TEST, CMD_UNMAP, IOSFC, desc_v2
from .resources_build import (BGRA8Unorm, OPT_SHARED, b_copy_tex_buf, b_fill_buf, b_fill_tex_color, desc1,
                              iosurface_op, mapper_ref_rec, ref_texture_rec)
from .session import FH, FW, Session
from .task import rec, seg

FRAME = FW * FH * 4          # one 16 x 8 BGRA frame: 512 bytes
MAPPED = ("map", "isolate")  # the IOSurface host device needs a memory map (apple-gfx-mmio.c)
IN_PROCESS = ("blocks", "map")  # ref textures (type 5) abort the service

# Exec resource flags (reims-vgpu-core's
# guest::records::CmdExecIndirectResource3): invalidate the host copy and
# synchronize it back, so a GPU write reaches guest memory in every setup.
F_SYNC = abi.RES_INVALIDATE_HOST | abi.RES_SYNCHRONIZE

RED = (1.0, 0.0, 0.0, 1.0)
GREEN = (0.0, 1.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)
GREY = (0.5, 0.5, 0.5, 1.0)


def frame_bgra(color):
    """The bytes a 16 x 8 BGRA8Unorm target cleared to an exactly
    representable colour holds (0, 0.5 and 1 give 0x00, 0x80, 0xff)."""
    def u8(c):
        return int(c * 255 + 0.5)
    r, g, b, a = color
    return bytes([u8(b), u8(g), u8(r), u8(a)]) * (FW * FH)


class Scanout(Session):
    """A Session with mapper surfaces, a gate, and presents on any FIFO.

    acks: the value each port's next ack must carry (expectedAck), which
    the scenario bumps with bump() for every connect (4) or update (12) the
    guest display raises."""

    def __init__(self, name, doc, **kw):
        super().__init__(name, doc, **kw)
        self.acks = {}
        self.gate = None
        self.gate_value = 0

    # ---- display pages and acks ----

    def connect(self, port, has_display=True):
        """Opcode 1 with a fresh page (all interrupts enabled) and, with a
        display, the ack of expectedAck (1 for a new page)."""
        self.acks[port] = 1 if has_display else 0
        super().connect(port, has_display=False)
        if has_display:
            self.ack(port, self.acks[port])

    def bump(self, port, n=1):
        self.acks[port] = self.acks.get(port, 0) + n
        return self.acks[port]

    def ack_next(self, port):
        """The ack of the value the guest display expects after one more 4 or
        12."""
        self.ack(port, self.bump(port))

    def hotplug(self, port, name=None, modes=None):
        """A display plugged into a port whose page is connected: the guest
        display connects (4) and writes the mode list; the guest waits for
        the mode count and acks."""
        self.s.api("new_display", spec=display(port, name=name or "head%d" % port,
                                                modes=self.modes if modes is None else modes))
        self.s.wait(gpa=self.ss(port, SS_MODE_COUNT), mask=0xFFFF, ne=0, label="modes on port %d" % port)
        self.ack_next(port)

    def set_modes(self, index, modes, label=None):
        self.s.api("set_modes", display=index, modes=[list(m) for m in modes], label=label)

    def hide_cursor(self, port):
        """Show 0 (5), then the guest clears +0xe04 and rings the doorbell:
        the doorbell finds the guest's show equal to the host's and calls
        only cursor_move."""
        s = self.s
        s.submit(abi.OP_DISPLAY_CURSOR_SHOW, p_show(port, 0), label="hide on %d" % port)
        self.expect("cursorShow")
        s.u32(self.ss(port, SS_GUEST_CURSOR_SHOW), 0)
        s.w(abi.R_DISPLAY_DOORBELL, port)
        self.expect("cursorMove")

    def move_cursor(self, port, x, y):
        s = self.s
        s.u32(self.ss(port, SS_CURSOR_POS), x | y << 16)
        s.w(abi.R_DISPLAY_DOORBELL, port)
        self.expect("cursorMove")

    def matrix(self, port, words, label="colour matrix"):
        """SetProperties (10), property 0: 12 floats into the write index,
        dirty."""
        self.s.submit(abi.OP_DISPLAY_SET_PROPERTIES, p_props(port, [(0, words)]), label=label)

    def gamma(self, n, curve):
        """A gamma table in task 0: (va, mapped length, count, checksum)."""
        words, csum = gamma_words(n, curve)
        va = self.t0.blob(struct.pack("<%dI" % len(words), *words))
        return (va, 12 * n, n, csum)

    # ---- the gate ----

    def make_gate(self, fid):
        self.gate = self.child(fid)
        return self.gate

    def gate_barrier(self, ahead=1):
        return (self.gate.stamp_index, self.gate_value + ahead)

    def open_gate(self, label=None):
        self.gate_value += 1
        return self.gate.submit(abi.OP_NOP, label=label or "open the gate (%d)" % self.gate_value)

    # ---- the IOSurface host device and mapper objects ----

    def surface(self, mid, seed, w=FW, h=FH, fmt=b"BGRA", bpe=4, pixels=None):
        """A w x h surface over one guest page, mapped (command 1)."""
        d = self.iosfc
        root, data = d.surface_pages(1)
        if pixels is None:
            pixels = bytes(((i * 5 + seed) & 0xFF) if i % 4 != 3 else 0xFF for i in range(w * h * bpe))
        self.s.mem(self.s.pa(data), pixels)
        d.put_desc(mid, desc_v2(root, w, h, fmt=fmt, bpe=bpe))
        d.status(d.submit((CMD_MAP, mid)), "map surface %d" % mid)
        self.surfaces[mid] = data
        return data

    def iosfc_ring(self, cmds, label):
        """Several commands on one doorbell, run in order with an
        interrupt each; their status words dumped."""
        self.iosfc.status(self.iosfc.submit(*cmds), label)

    def encode_sized(self, index, w, h, label):
        self.s.encode(index, w, h, label=label, samples=[[0, 0], [w - 1, h - 1], [w // 2, h // 2]])

    def dump_surface(self, mid, label=None, length=FRAME):
        self.s.dump(self.s.pa(self.surfaces[mid]), length, label or "surface %d" % mid)

    def mapper_objects(self, mids, buffers=(), sizes=None):
        """An `extra` for task(): a mapper texture (type 11, 16 x 8 BGRA,
        shared, read/write/render target) over each mapping ID in mids, a
        mapper buffer (type 15) over each in buffers, and a render pipeline
        whose colour attachment is BGRA8Unorm. t.mtex[mid], t.mbuf[mid],
        t.bgra. sizes: {mid: (w, h, Metal pixel format)} for other shapes."""
        sizes = sizes or {}

        def extra(t):
            t.mtex, t.mbuf = {}, {}
            for mid in mids:
                w, h, fmt = sizes.get(mid, (FW, FH, BGRA8Unorm))
                d = desc1(fmt=fmt, w=w, h=h, usage=1 | 2 | 4, options=OPT_SHARED)
                t.mtex[mid] = t.ol.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(mid) + iosurface_op(12, d))
                t.extra.append(t.mtex[mid])
            for mid in buffers:
                t.mbuf[mid] = t.ol.add(abi.T_MAPPER_BUFFER, mapper_ref_rec(mid))
                t.extra.append(t.mbuf[mid])
            t.bgra = t.ol.add(abi.T_SERIALIZED, pipeline_record(14, render_blob(t.f_vert, t.f_frag, BGRA8Unorm)))
        return extra

    def ref_objects(self, frames):
        """An `extra` for task(): a ref texture (type 5) over each of task
        0's backings in frames (indices into self.backings), 16 x 8 BGRA,
        shared, read/write/render target; t.rtex[frame], and t.bgra."""
        def extra(t):
            d = desc1(fmt=BGRA8Unorm, w=FW, h=FH, usage=1 | 2 | 4, options=OPT_SHARED)
            t.rtex = {}
            for i in frames:
                t.rtex[i] = t.ol.add(abi.T_BACKING_TEXTURE, ref_texture_rec(self.backings[i][0], op=iosurface_op(12, d)))
                t.extra.append(t.rtex[i])
            t.bgra = t.ol.add(abi.T_SERIALIZED, pipeline_record(14, render_blob(t.f_vert, t.f_frag, BGRA8Unorm)))
        return extra

    # ---- work, as segments ----

    @staticmethod
    def compose(t, tex, clear, draw=False, w=FW, h=FH):
        """A render pass over tex (a w x h BGRA texture), cleared, and the
        full-target triangle of (1, 0.5, 0.25, 1) when draw is set."""
        recs = [R.describe_pass({0: dict(tex=tex, clear=clear)}, width=w, height=h)]
        if draw:
            recs += [R.pipeline_state(t.bgra), R.draw16(3, 0, 3)]
        return [seg(abi.ENC_RENDER, b"".join(recs))]

    @staticmethod
    def fill_rect(tex, color, origin=(0, 0, 0), size=(FW, FH, 1)):
        """Blit 321: fill a region of tex with a colour."""
        return [seg(abi.ENC_BLIT, b_fill_tex_color(tex, size, color, BGRA8Unorm, origin=origin))]

    @staticmethod
    def fill_bytes(buf, off, n, byte):
        return [seg(abi.ENC_BLIT, b_fill_buf(buf, off, n, byte))]

    @staticmethod
    def readback(t, tex, phase):
        """Copies tex's 16 x 8 texels into the task's out buffer at
        phase x 0x400 (a blit): the GPU's view of the surface."""
        return [seg(abi.ENC_BLIT, b_copy_tex_buf(tex, t.out, phase * 0x400, FW * 4, FRAME, (FW, FH, 1)))]

    @staticmethod
    def bad_blit():
        """A blit record the decoder does not know (399): the exec faults
        its FIFO without touching the GPU."""
        return [seg(abi.ENC_BLIT, rec(399, bytes(16)))]

    @staticmethod
    def compute_into(t, buf, off=0):
        """add_one (fixtures/shaders/compute_add.metal) of the task's source
        into buf at off: 64 words, source word + 1."""
        from .compute_objects import Compute as C
        return [seg(abi.ENC_COMPUTE, C.set_pipeline(t.cpso) + C.set_buffers(0, [(buf, off), (t.src, 0)])
                    + C.dispatch_threads((64, 1, 1), (64, 1, 1)))]

    def death_payload(self, t):
        """An exec whose command buffer range starts at the task's end: the
        service dereferences a null cursor and dies
        (exec_null_cursor_past_end). Isolate only: in-process it kills the
        host."""
        res = t.resources()
        return (struct.pack("<III", t.tid, len(res), 1) + b"".join(struct.pack("<II", r, f) for r, f in res)
                + struct.pack("<QQ", t.length, 0x100))

    # ---- commands on any FIFO ----

    def exec_cmd(self, fid, t, segments, barriers=(), flags=None):
        """An exec (67) of segments on FIFO fid, not rung; returns its
        stamp. flags: {ref: flags} overriding the resource list's 0s."""
        f = self.fifo(fid)
        p = t.payload(segments)
        if flags:
            res = [(r, flags.get(r, 0)) for r, _ in t.resources()]
            cbs = p[12 + 8 * len(res):]
            p = struct.pack("<III", t.tid, len(res), len(segments)) + b"".join(
                struct.pack("<II", r, fl) for r, fl in res) + cbs
        return f.cmd(abi.OP_EXEC_INDIRECT3, p, barriers=barriers)

    def swap_cmd(self, fid, port, mid, barriers=(), gamma=None):
        """DisplaySwapMapping (8) of mapping mid on port, on FIFO fid, not
        rung; returns its stamp."""
        g = gamma or (0, 0, 0, 0)
        return self.fifo(fid).cmd(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(port, mid, *g), barriers=barriers)

    def present_cmd(self, fid, port, frame, barriers=(), gamma=None):
        """DisplayTransaction3 (7) of backing `frame` on port, on FIFO fid,
        not rung; returns its stamp."""
        g = gamma or (0, 0, 0, 0)
        return self.fifo(fid).cmd(abi.OP_DISPLAY_TRANSACTION3, p_present3(port, self.backings[frame][0], *g),
                                  barriers=barriers)

    def ring(self, *fids):
        for fid in fids:
            self.fifo(fid).doorbell()

    def wait_stamp(self, fid, st, label=None):
        self.fifo(fid).wait_stamp(st, label=label)

    def go(self, fid, st, label=None):
        """Rings fid and waits for stamp st."""
        self.ring(fid)
        self.wait_stamp(fid, st, label)

    def swap_on(self, fid, port, mid, index=None, barriers=(), gamma=None, encode=True):
        """Opcode 8 on FIFO fid, rung, its stamp and new_frame waited for,
        and the frame encoded."""
        st = self.swap_cmd(fid, port, mid, barriers=barriers, gamma=gamma)
        self.go(fid, st, "swap %d on port %d, FIFO %d" % (mid, port, fid))
        self.expect("newFrame", label="mapping %d on port %d" % (mid, port))
        if encode:
            self.encode(port if index is None else index, "mapping %d on port %d" % (mid, port))
        return st

    def node_fault(self, fid, label=None):
        c = self.children[fid]
        self.s.wait(gpa=c.node_gpa(NODE_FAULT), ne=HEALTHY, label=label or "child %d faulted" % fid)
        self.s.wait_fault()

    def node(self, fid, label):
        self.s.dump(self.children[fid].node_gpa(), abi.CHILD_NODE_SIZE, label)

    def stamps(self, n, label="stamp slots"):
        self.s.dump(self.s.fifo.base_gpa(), 4 * n, label)


__all__ = ["Scanout", "FRAME", "MAPPED", "IN_PROCESS", "F_SYNC", "RED", "GREEN", "BLUE", "WHITE", "GREY",
           "frame_bgra", "FW", "FH", "CMD_MAP", "CMD_UNMAP", "CMD_TEST", "ChildFIFO", "IOSFC", "PAGE"]
