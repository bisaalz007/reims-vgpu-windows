"""Whole guest sessions: a small library of steps that compose into a guest
driver's lifetime, for the interplay scenarios (scenarios/interplay_session.py).

A Session is a Scenario plus what a macOS guest driver keeps between steps:
the negotiated version, the kernel task (task 0) with its IOSurface backings
and cursor glyphs, the display pages, the GPU tasks with their objects, the
child FIFOs, and how many display callbacks of each kind the host has seen
so far (the runner counts callbacks cumulatively, across displays and across
a recreated device, so every wait names the running total).

Every layout below is the one the per-area builders already restate; this
module only composes them:
  - the root FIFO, child FIFOs and barriers: fifo.py
  - tasks, object lists, segments: task.py
  - render and compute objects: render_lib.py, compute_objects.py,
    exec_objects.py
  - heaps and heap buffers, blit records, deletions: resources_build.py
  - the display's shared page and opcodes 1-12: display_guest.py
  - EFI: efi_guest.py
  - the IOSurface host device: iosfc_guest.py
  - suspend and resume: rvg_device_will_suspend and the rest of the C
    API's migration calls, as scenarios/suspend.py drives them

Nothing here is imported from the library's code.
"""
import struct

from . import abi
from . import render_lib as R
from .abi import PAGE
from .compute_objects import Compute as C, compute_blob, compute_pipeline_record
from .display_guest import (SS_CURSOR_POS, SS_GUEST_CURSOR_SHOW, SS_MODE_COUNT, display, p_ack, p_glyph,
                            p_present3, p_set_shared, p_show, p_swap, pattern)
from .efi_guest import EFI
from .exec_objects import pipeline_record, render_blob
from .fifo import ChildFIFO
from .iosfc_guest import CMD_MAP, CMD_TEST, CMD_UNMAP, IOSFC, desc_v2
from .resources_build import (BGRA8Unorm, OPT_SHARED, b_copy_buf_buf, b_copy_tex_buf, b_fill_buf, desc1,
                              heap_buffer_rec, heap_rec, iosurface_op, mapper_ref_rec)
from .scenario import SETUPS, Scenario
from .task import GPUTask, seg

# Operation IDs of opcode 40, the serialized delete (reims-vgpu-core's
# resources::manager::cache, Deletion).
DEL_RESOURCE, DEL_FUNCTION, DEL_COMPUTE_PSO, DEL_RENDER_PSO = 1000, 1005, 1006, 1007

# Each phase of GPU work writes 0x300 bytes of results into a task's out
# buffer, at phase x PHASE: the blit's 0x100, the compute's 0x100, then the
# render target read back through a blit (8 x 8 RGBA8).
PHASE = 0x400
BLIT_OFF, COMPUTE_OFF, RENDER_OFF = 0x000, 0x100, 0x200
RESULT = 0x300
TW = TH = 8          # the render target
FW, FH = 16, 8       # frames the displays show


def u32s(*v):
    return struct.pack("<%dI" % len(v), *v)


class SessionTask(GPUTask):
    """A GPU task holding what a guest app's session uses: a source buffer
    of its own pattern, an out buffer of 4 pages that each phase writes into,
    a private heap with a buffer placed in it, add_one's function and
    compute pipeline (fixtures/shaders/compute_add.metal), tri_vert and
    tri_frag's functions and render pipeline (vert.metal, frag.metal: one
    full-target triangle of (1, 0.5, 0.25, 1)), and an 8 x 8 RGBA8 render
    target over a shared buffer (type 8, commandType 9). Object references
    are handed out in that order, the same in every task."""

    def __init__(self, s, tid, seed, pages=64):
        super().__init__(s, tid=tid, pages=pages)
        self.seed = seed
        self.shared = []  # every shared buffer: listed in each exec (section 6.12)
        self.src, self.src_va = self._buf(PAGE, bytes((i * 7 + seed) & 0xFF for i in range(0x100)))
        self.out, self.out_va = self._buf(4 * PAGE)
        self.heap_va = self.reserve(4 * PAGE)
        self.heap = self.ol.add(abi.T_HEAP, heap_rec(self.heap_va // PAGE, 4 * PAGE, 1))
        self.hbuf = self.ol.add(abi.T_HEAP_BUFFER, heap_buffer_rec(self.heap, PAGE, PAGE))
        self.f_add = self._function("shaders/compute_add.metallib")
        self.cpso = self.ol.add(abi.T_SERIALIZED, compute_pipeline_record(compute_blob(self.f_add)))
        self.f_vert = self._function("shaders/vert.metallib")
        self.f_frag = self._function("shaders/frag.metallib")
        self.rpso = self.ol.add(abi.T_SERIALIZED, pipeline_record(14, render_blob(self.f_vert, self.f_frag, 70)))
        self.tbuf, self.tbuf_va = self._buf(PAGE)
        self.tex = self.ol.add(abi.T_SERIALIZER_TEXTURE, R.op_header(9) + struct.pack(
            "<IIQQ", 0, self.tbuf, 0, 32) + R.tex_desc(TW, TH))
        self.extra = []  # objects a session adds (mapper textures, ...)

    def _buf(self, length, contents=None):
        ref, va = self.buffer(length, contents=contents)
        self.shared.append(ref)
        return ref, va

    def _function(self, fixture):
        va = self.reserve(0x2000)
        n = self.s.mem_file(self.gpa(va), fixture)
        return self.ol.add(abi.T_FUNCTION, struct.pack("<QIII", va, n, 15, 0))

    def resources(self):
        return [(r, 0) for r in self.shared + [self.heap, self.hbuf] + self.extra]

    def payload(self, segments):
        """An ExecIndirect3 (67) payload of these segments, one range each."""
        cbs = [self.cmdbuf(sg) for sg in segments]
        res = self.resources()
        p = struct.pack("<III", self.tid, len(res), len(cbs))
        p += b"".join(struct.pack("<II", r, f) for r, f in res)
        p += b"".join(struct.pack("<QQ", va, n) for va, n in cbs)
        return p

    # ---- the work, as segments ----

    def blit_segments(self, phase, byte):
        """A fill, a copy of the source, and the heap buffer filled and
        copied out: 0x100 bytes at the phase's blit slot."""
        o = phase * PHASE + BLIT_OFF
        return [seg(abi.ENC_BLIT, b_fill_buf(self.out, o, 0x40, byte)
                    + b_copy_buf_buf(self.src, self.out, 0, o + 0x40, 0x40)
                    + b_fill_buf(self.hbuf, 0, 0x80, byte ^ 0xFF)
                    + b_copy_buf_buf(self.hbuf, self.out, 0, o + 0x80, 0x80))]

    def compute_segments(self, phase, src=None, src_off=0):
        """add_one over 64 words of the source (or another buffer): 0x100
        bytes at the phase's compute slot."""
        o = phase * PHASE + COMPUTE_OFF
        return [seg(abi.ENC_COMPUTE, C.set_pipeline(self.cpso)
                    + C.set_buffers(0, [(self.out, o), (self.src if src is None else src, src_off)])
                    + C.dispatch_threads((64, 1, 1), (64, 1, 1)))]

    def render_segments(self, phase, clear=(0.0, 0.0, 1.0, 1.0), draw=True):
        """A pass over the target cleared to `clear`, the triangle, then a
        blit segment copying the target into the phase's render slot."""
        o = phase * PHASE + RENDER_OFF
        recs = [R.describe_pass({0: dict(tex=self.tex, clear=clear)}, width=TW, height=TH)]
        if draw:
            recs += [R.pipeline_state(self.rpso), R.draw16(3, 0, 3)]
        return [seg(abi.ENC_RENDER, b"".join(recs)),
                seg(abi.ENC_BLIT, b_copy_tex_buf(self.tex, self.out, o, 32, 32 * TH, (TW, TH, 1)))]

    def result_gpa(self, phase):
        return self.gpa(self.out_va + phase * PHASE)


class Session:
    """A guest driver's session over one Scenario. ports: display ports the
    device has; displays: the ports whose displays exist at creation.
    efi: trace callbacks, so every EFI present is deterministic (efi.py).
    iosfc: the IOSurface host device (map and isolate only)."""

    def __init__(self, name, doc, ports=1, displays=None, setups=SETUPS, host=True, efi=False, iosfc=False,
                 modes=((64, 32, 60), (FW, FH, 60)), watchdog_s=120, exclusive=False):
        self.ports = ports
        self.modes = modes
        desc = {}
        if ports > 1:
            desc["port_count"] = ports
        if efi:
            desc["trace"] = True
        disp = [display(p, name="head%d" % p, modes=modes) for p in (range(ports) if displays is None else displays)]
        self.s = Scenario(name, setups=setups, host=host, watchdog_s=watchdog_s, displays=disp,
                          descriptor=desc or None, iosfc=iosfc, exclusive=exclusive, doc=doc)
        self.n = {}             # callbacks seen so far, by kind
        # A new process's callbacks count from 0 (Scenario.phase).
        self.s.on_phase(lambda: setattr(self, "n", {}))
        self.pages = {}         # port -> shared page
        self.t0 = None
        self.tasks = []
        self.children = {}
        self.backings = []      # (ref, va) in task 0
        self.efi = None
        self.iosfc = None
        self.version = None

    # ---- callbacks ----

    def expect(self, kind, n=1, label=None):
        """Waits until the host has called `kind` n more times."""
        self.n[kind] = self.n.get(kind, 0) + n
        self.s.wait(cb=kind, cb_count=self.n[kind], label=label or "%s %d" % (kind, self.n[kind]))

    # ---- bring-up ----

    def negotiate(self, asks=(0x100, 0x800000F0), version=abi.MAX_VERSION):
        """Version negotiation as a guest probes it: requests that the
        device answers with another version (above 0xf5 it offers 0xf5; a
        request with bit 31 set is ignored, set_binary_version), each read
        back with the port count
        and connection types, then the version the driver settles on."""
        s = self.s
        for v in asks:
            s.w(abi.R_VERSION, v)
            s.r(abi.R_VERSION, "asked 0x%x" % v)
        s.w(abi.R_VERSION, version)
        s.r(abi.R_VERSION, "settled")
        s.r(abi.R_PORT_COUNT, "settled")
        s.r(abi.R_PORT_TYPES, "settled")
        self.version = version

    def root(self, fifo_pages=2):
        """Root page, root FIFO, enable (docs/testing.md, section 6.25 a)."""
        self.s.bring_up(version=None, fifo_pages=fifo_pages)
        self.s.submit(abi.OP_NOP, label="first NOP")

    def device_info(self, max_key=45):
        """GetDeviceInfo (58) {maxKey, maxCount, page} into a page filled
        with 0xa5 first."""
        s = self.s
        page = s.alloc_page()
        s.fill(s.pa(page), 0x180, 0xA5)
        s.submit(abi.OP_GET_DEVICE_INFO, struct.pack("<III", max_key, 64, s.pfn(page)), label="device info")
        s.dump(s.pa(page), 0x180, "device info")

    # ---- EFI and the handover ----

    def efi_boot(self, seed=0x21):
        """The firmware's framebuffer: 16 x 8, stride 64, over one page; EFI
        binds the display parked at port 0 and presents it once."""
        s = self.s
        self.efi = EFI(s)
        self.efi.write_frame(FW, FH, 64, seed=seed)
        self.efi.setup(FW, FH, 64)
        self.expect("newFrame", label="EFI's first frame")
        self.efi.regs("EFI up")
        s.encode(0, FW, FH, label="EFI frame")

    def efi_off(self):
        """0x1200 = 0: EFI torn down, its registers zeroed (:1745-1758)."""
        self.s.w(abi.R_EFI_ENABLE, 0)
        self.efi.regs("EFI off")

    # ---- the kernel task and displays ----

    def kernel_task(self, pages=32, frames=2):
        """Task 0: the display commands' task. Holds `frames` BGRA backings
        (type 4) of 16 x 8, each its own pattern, and a cursor glyph at VA
        page 8."""
        t = self.t0 = GPUTask(self.s, tid=0, pages=pages)
        self.glyph_va = t.reserve(PAGE)
        t.write(self.glyph_va, struct.pack("<I", 0x80402010) * 4 + struct.pack("<I", 0xFF00FF00) * 4)
        for i in range(frames):
            va = t.reserve(PAGE)
            t.write(va, pattern(FW, FH, 0x30 * i + 1))
            ref = t.ol.add(abi.T_IOSURFACE_BACKING, struct.pack("<QIIB3x", PAGE, va // PAGE,
                                                                 struct.unpack(">I", b"BGRA")[0], 1)
                           + struct.pack("<IIII", 0, FW, FH, (FW * 4) | 4 << 24))
            self.backings.append((ref, va))
        t.define(label="define task 0")
        t.install(label="task 0's object list")

    def connect(self, port, has_display=True):
        """Opcode 1 with a fresh page (all interrupts enabled), then, when a
        display is attached, the ack of expectedAck 1."""
        s = self.s
        p = self.pages[port] = s.alloc_page()
        s.u32(s.pa(p, 0x104), 0xF)
        s.submit(abi.OP_DISPLAY_SET_SHARED_STATE_PAGE, p_set_shared(port, s.pfn(p)), label="connect %d" % port)
        if has_display:
            self.ack(port, 1)

    def ack(self, port, value):
        self.s.submit(abi.OP_DISPLAY_ACK, p_ack(port, value), label="ack %d on %d" % (value, port))

    def ss(self, port, off=0):
        return self.s.pa(self.pages[port], off)

    def page(self, port, label):
        self.s.diff(label="port %d: %s" % (port, label), gpa=self.ss(port), length=PAGE)

    def hotplug(self, port):
        """A display attached to a port whose page is already connected: the
        guest display's state machine runs (DeviceCore::attach_display) and
        writes the mode list into the page; the guest waits for the mode
        count, then acks."""
        self.s.api("new_display", spec=display(port, name="head%d" % port, modes=self.modes))
        self.s.wait(gpa=self.ss(port, SS_MODE_COUNT), mask=0xFFFF, ne=0, label="modes on port %d" % port)
        self.ack(port, 1)

    def unplug(self, index, port):
        """Releasing a display (hot unplug, detach_display): the host
        clears the page's mode count."""
        self.s.api("release_display", display=index)
        self.s.wait(gpa=self.ss(port, SS_MODE_COUNT), mask=0xFFFF, eq=0, label="port %d unplugged" % port)

    def disconnect(self, port):
        self.s.submit(abi.OP_DISPLAY_SET_SHARED_STATE_PAGE, p_set_shared(port, 0), label="disconnect %d" % port)

    def present(self, port, frame, display_index=None, encode=True):
        """DisplayTransaction3 (7) of backing `frame` on port: new_frame, then
        the host encodes it, as QEMU's apple_gfx_do_render does
        (apple-gfx.c:52-78)."""
        ref, _ = self.backings[frame]
        self.s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(port, ref), label="present %d on %d" % (frame, port))
        self.expect("newFrame", label="frame %d on port %d" % (frame, port))
        if encode:
            self.encode(port if display_index is None else display_index, "frame %d on port %d" % (frame, port))

    def encode(self, index, label):
        self.s.encode(index, FW, FH, label=label, samples=[[0, 0], [FW - 1, FH - 1], [5, 3]])

    def cursor(self, port, x, y, glyph=True):
        """A glyph (4) of 4 x 2 from task 0 with hotspot (1, 1), show (5),
        then the guest moves it: +0xe00 and the doorbell 0x1220 = port,
        which calls cursor_move."""
        s = self.s
        if glyph:
            s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(port, self.glyph_va, 32, 16, 4, 2, 1, 1, 0),
                     label="glyph on %d" % port)
            self.expect("cursorGlyph")
            s.submit(abi.OP_DISPLAY_CURSOR_SHOW, p_show(port, 1), label="show on %d" % port)
            self.expect("cursorShow")
            s.u32(self.ss(port, SS_GUEST_CURSOR_SHOW), 1)
        s.u32(self.ss(port, SS_CURSOR_POS), x | y << 16)
        s.w(abi.R_DISPLAY_DOORBELL, port)
        self.expect("cursorMove")

    # ---- GPU tasks and FIFOs ----

    def task(self, tid, seed, extra=None):
        """A SessionTask defined and its object list installed; extra(t) may
        add objects before the list is written."""
        t = SessionTask(self.s, tid, seed)
        if extra:
            extra(t)
        t.define(label="define task %d" % tid)
        t.install(label="task %d's object list" % tid)
        self.tasks.append(t)
        return t

    def child(self, fid, pages=1):
        c = ChildFIFO(self.s, fid, pages=pages)
        c.setup_node()
        c.define(label="define child %d" % fid)
        self.s.fifos[fid] = c
        self.children[fid] = c
        return c

    def fifo(self, fid):
        return self.s.fifo if fid == 0 else self.children[fid]

    def batch(self, items, label, serial=False):
        """Several execs, each on its own FIFO, written first, then each
        FIFO rung, then each stamp waited for. items: (fid, task, segments,
        barriers). Returns {fid: stamp}.

        Unserialized, the FIFOs run concurrently. In the blocks setup that
        makes the host's block calls (every exec reads its command buffers
        through read_memory) interleave by timing, which is no order the
        implementation establishes (docs/testing.md, decision 8), so a
        scenario that runs in blocks passes serial=True: each exec then
        carries a barrier on the previous exec's stamp when that one is on
        another FIFO, and the FIFOs are rung last first, so each waits on the
        one before."""
        written = []
        prev = None
        for fid, t, segments, barriers in items:
            f = self.fifo(fid)
            bars = list(barriers)
            if serial and prev is not None and prev[0] is not f:
                bars.append((prev[0].stamp_index, prev[1]))
            st = f.cmd(abi.OP_EXEC_INDIRECT3, t.payload(segments), barriers=bars)
            written.append((f, st))
            prev = (f, st)
        rung = list(dict.fromkeys(f for f, _ in written))
        for f in reversed(rung) if serial else rung:
            f.doorbell()
        for f, st in written:
            f.wait_stamp(st, label="%s: FIFO %d stamp %d" % (label, f.fid, st))
        for t in {id(t): t for _, t, _, _ in items}.values():
            t.reset_cmdbufs()
        return {f.fid: st for f, st in written}

    def run(self, fid, t, segments, label):
        return self.batch([(fid, t, segments, ())], label, serial=True)

    def phase(self, t, phase, fid=0, byte=0x5A, label=None):
        """One task's blit, compute and render work, in one exec."""
        lab = label or "task %d phase %d" % (t.tid, phase)
        self.run(fid, t, t.blit_segments(phase, byte) + t.compute_segments(phase) + t.render_segments(phase), lab)

    def results(self, t, phase, label=None):
        self.s.dump(t.result_gpa(phase), RESULT, label or "task %d phase %d results" % (t.tid, phase))

    def same(self, a, pa, b, pb, label):
        """A differential check: two phases' results are the same bytes."""
        self.s.cmp(a.result_gpa(pa), b.result_gpa(pb), RESULT, label=label)

    # ---- the IOSurface host device ----

    def iosfc_start(self):
        self.iosfc = IOSFC(self.s, v2=True)
        self.iosfc.start()
        self.surfaces = {}

    def surface(self, mid, seed):
        """A 16 x 8 BGRA surface over one guest page, mapped with command 1."""
        d = self.iosfc
        root, data = d.surface_pages(1)
        self.s.mem(self.s.pa(data), pattern(FW, FH, seed))
        d.put_desc(mid, desc_v2(root, FW, FH))
        d.status(d.submit((CMD_MAP, mid)), "map surface %d" % mid)
        self.surfaces[mid] = data
        return data

    def iosfc_cmd(self, cmd, mid, label):
        self.iosfc.status(self.iosfc.submit((cmd, mid)), label)

    def swap(self, port, mid, display_index=None, encode=True):
        """DisplaySwapMapping (8): present mapping `mid`."""
        self.s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(port, mid), label="swap %d on %d" % (mid, port))
        self.expect("newFrame", label="mapping %d on port %d" % (mid, port))
        if encode:
            self.encode(port if display_index is None else display_index, "mapping %d on port %d" % (mid, port))

    # ---- suspend and resume ----

    def suspend(self, label="suspend"):
        self.s.api("willSuspend", label=label)
        self.s.api("finishSuspend", label=label)

    def resume(self, label="resume", config=None):
        """A migration target: a fresh device over the same guest RAM with
        the scenario's displays, the blob, rvg_device_did_resume."""
        if config is None:
            self.s.api("recreate", label=label)
        else:
            self.s.api("recreate", label=label, config=config)
        self.s.api("willResume", label=label)
        self.s.api("didResume", label=label)

    # ---- teardown ----

    def delete_objects(self, t):
        """Opcode 40 for each object, pipelines before the functions they
        name, resources last."""
        s = self.s
        pre = ((DEL_RENDER_PSO, t.bgra),) if hasattr(t, "bgra") else ()
        for op, ref in pre + ((DEL_RENDER_PSO, t.rpso), (DEL_COMPUTE_PSO, t.cpso), (DEL_FUNCTION, t.f_vert),
                        (DEL_FUNCTION, t.f_frag), (DEL_FUNCTION, t.f_add), (DEL_RESOURCE, t.tex),
                        (DEL_RESOURCE, t.hbuf), (DEL_RESOURCE, t.heap)) + tuple(
                            (DEL_RESOURCE, r) for r in t.extra + t.shared):
            s.submit(abi.OP_DELETE_OBJECT, struct.pack("<IIII", t.tid, op, 12, ref),
                     label="task %d: delete %d ref %d" % (t.tid, op, ref))

    def delete_backings(self):
        """54 {backing ref, task 0} for each backing."""
        for ref, _ in self.backings:
            self.s.submit(abi.OP_DELETE_IOSURFACE_BACKING2, struct.pack("<II", ref, 0), label="delete backing %d" % ref)

    def teardown(self, release=True):
        """Objects, tasks, child FIFOs, display pages, then the host
        releases the device."""
        s = self.s
        for t in self.tasks:
            self.delete_objects(t)
            t.delete(label="delete task %d" % t.tid)
        if self.t0 is not None:
            self.delete_backings()
        for port in sorted(self.pages):
            self.disconnect(port)
        if self.t0 is not None:
            self.t0.delete(label="delete task 0")
        for fid in sorted(self.children):
            self.children[fid].delete(label="delete child %d" % fid)
        s.submit(abi.OP_NOP, label="last NOP")
        s.events("torn down")
        s.dump(s.pa(s.root_page), 0x40, "root page: task words")
        s.dump(s.pa(s.root_page, abi.ROOT_CHILD_NODE), 0x14 * 4, "root page: child nodes")
        s.dump(s.pa(s.root_page, abi.ROOT_DISPLAY_WORD), 0x10, "root page: display words")
        s.fault_state("torn down")
        if release:
            s.api("release")
            s.events("released")
