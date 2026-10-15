"""A guest's whole life, the way a macOS guest drives apple-gfx under QEMU,
in one session (scenarios/interplay_boot.py).

Where the marathon (pgtest/marathon_util.py) picks each step at random
among every legal one, a boot program follows the phases of a real guest's
life, and the seed varies what happens inside each phase: how many frames,
presents and batches, the order of independent steps, and which subsystems
are busy when a phase ends.

  1. Firmware: the EFI framebuffer (0x1200-0x1228) presents on port 0's
     parked display at the fresh device's version 0 (efi_before_version),
     or after the driver's version negotiation, and its frames go on
     (trace fires) while the driver comes up.
  2. The driver's takeover: version negotiation (asks the device answers
     with 0xf5 or ignores), the root page and root FIFO, task 0 with its
     backings, glyph regions and gamma tables, device info (58), the child
     FIFOs, the gate and the GPU tasks; EFI turned off (0x1200 = 0), and
     only then port 0 connected: each head's page (opcode 1, every
     interrupt enabled), its ack, the cursor glyph, show and position.
  3. The window server: the IOSurface host device's ring and descriptor
     table, every mapper surface mapped, and a compositor task (tid past
     the GPU tasks and the control) holding a mapper buffer (type 15) over
     each surface and a window buffer, whose pixels (the guest writes
     them) the GPU copies into a surface (blit 301) before the frame is
     presented with opcode 8 from a FIFO behind the copy.
  4. Steady state: compositions (GPU copy, then opcode 8, with or without a
     gamma table), opcode-7 presents of task 0's backings, colour
     matrices, ICC profiles, cursor moves and glyphs, beside app-like GPU
     work (the marathon's dag batches over the root FIFO, children and
     tasks, some held on the gate), app faults and their recovery, and
     lifetime and memory changes.
  5. Hot plug and mode changes: the host unplugs and plugs displays and
     sets new mode lists; the driver acks each update.
  6. Sleep and wake of the heads (9), with presents while asleep.
  7. Now and then a host pause, then the suspend onto a fresh device
     (rvg_device_will_suspend, rvg_device_finish_suspend, recreate,
     rvg_device_will_resume, rvg_device_did_resume), with work held on the gate
     across it, and steady state after the resume.
  8. Shutdown: every held batch released and every result compared; the
     cursors hidden, the heads put to sleep or not and disconnected; the
     compositor's mapper buffers deleted (37) and its task deleted, every
     surface unmapped; the backings deleted (54); the child FIFOs, the gate
     and every task deleted (49, 32), task 0 last; then the root FIFO
     disabled, and the device released, or reset (a reboot) and the
     firmware's EFI frames again.

Before phases 2 (EFI off), 5, 6, 7 and 8 the seed decides what is busy:
GPU batches held on the gate, batches that run free, a fault on an idle
child (not before a pause or a suspend, randmix 7.1), EFI frames.

Programs with a death stop at a realistic moment instead of going on
(docs/testing.md 7.3, 7.4), with work held on the gate: a one-entry gamma
table in steady state, a guest mode count over 128 when the host sets a
new list in the hot-plug phase, and at the suspend a second pause, or the
cursor doorbell or rvg_display_cursor_position while the host has the device
paused.

The model is the display walk's (pgtest/randdisplay.py) and the
marathon's, which this class extends: every callback, interrupt, 0x1014
and 0x1018, shared page, encoded pixel, present count, GPU slot and chain
counter, scratch page and fault register is predicted. What this module
adds is predicted the same way: a surface the GPU wrote is compared with
the model's bytes and encoded on every display that shows it (the frame is
a view of the surface, docs/testing.md 6.21), the unmap statuses, and the
root FIFO's enable register after the shutdown. python3 -m
pgtest.boot_util reads the expected files back against the model.

Invariants, on top of the marathon's:
  - EFI is on only while port 0 has no page, and is off before port 0
    connects, a pause, a suspend or a reset.
  - A composition's copy runs on the root FIFO or on an idle child, and
    its present on the root FIFO or on another idle child, with a barrier
    on the copy's stamp: each stamp publishes alone, and the present
    follows the copy.
  - Surfaces are written only in 8-bit formats, with palette bytes and
    alpha 0xff, so the model predicts their pixels.
  - Surfaces are unmapped only in the shutdown, after the compositor's
    mapper buffers are deleted and every head disconnected, so nothing
    holds them.

QEMU (vendor/qemu/hw/display/apple-gfx.c) creates one display at port 0
with its mode list before the guest runs, encodes a frame on each
new_frame, and resets the device on a device reset; it gives no
trace-range callbacks, so its EFI presents come from the 100 ms timer.
These programs give them (as randdisplay does) so that every EFI present
is one new_frame, and use the suspend and pause calls a migration would
(QEMU's apple-gfx makes none yet).

Nothing here is imported from the library's code.
"""
import json
import struct

from . import abi
from .diff_util import exec_payload
from .display_guest import p_set_shared, p_show, p_swap
from .efi_guest import EFI
from .iosfc_guest import CMD_UNMAP, SUCCESS
from .lifetime_util import delete_resource
from .marathon_util import E3, GATE, Marathon
from .randdisplay import EFI_SIZES, EFI_STRIDE, M32, O_ACKP, O_ENABLED, Death, PageM, check_obs, pixels
from .randgen import Rng
from .resources_build import b_copy_buf_buf, mapper_ref_rec
from .task import GPUTask, seg

# Phase lengths: steady-state steps before the hot-plug phase, while the
# heads sleep, and after the resume.
LENGTHS = {"short": ((6, 10), (1, 3), (4, 8)),
           "medium": ((14, 26), (2, 5), (10, 18)),
           "long": ((70, 110), (4, 8), (60, 90))}
# Where each death lands.
DEATH_PHASE = {"gamma_one": "steady", "mode_count_over": "hotplug", "cursor_paused": "sleep",
               "doorbell_paused": "suspend", "pause_twice": "suspend"}
MISC = ("matrix", "icc", "compositor", "glyph", "move", "encode", "rewrite_backing", "present6", "clear_raised")


class Boot(Marathon):
    """One boot program (the module docstring)."""

    def __init__(self, seed, doc, length="medium", death=None):
        super().__init__(seed, 0, doc, death=death)
        self.name = self.s.name = "interplay_boot_%d" % seed
        p = self.p = Rng(seed * 0xB0075 + 0x1D)
        self.length = length
        self.efi_on = True
        self.efi_first = p.chance(0.7)
        self.gpu_early = p.chance(0.5)      # GPU set-up before EFI goes off
        self.info_first = p.chance(0.5)
        self.late_port = p.chance(0.4)      # port 1 connected while EFI still shows
        self.defer_ack = p.chance(0.3)
        self.pause_on = p.chance(0.4)
        self.sleep_first = p.chance(0.5)    # heads asleep before they disconnect
        self.disable_fifo = p.chance(0.6)
        self.reboot = p.chance(0.45) and death is None
        lo = LENGTHS[length]
        self.n_steady = [lo[0][0] + p.below(lo[0][1] - lo[0][0] + 1), lo[1][0] + p.below(lo[1][1] - lo[1][0] + 1),
                         lo[2][0] + p.below(lo[2][1] - lo[2][0] + 1)]
        total = sum(self.n_steady) + 40
        self.s._config["watchdog_s"] = 120 + total
        self.phase = "boot"
        self.nstep = 0
        self.comp = None
        self.phist = {}
        self.fills = 0

    # ---- labels ----

    def mark(self, text):
        self.nstep += 1
        lab = "%s %d: %s" % (self.phase, self.nstep, text)
        self.s.mark(lab)
        return lab

    def enter(self, phase):
        self.phase = phase
        self.s.mark("== %s" % phase)

    def did(self, kind):
        self.phist[kind] = self.phist.get(kind, 0) + 1

    # ---- firmware: EFI (reims-vgpu-core's display::efi) ----

    def efi_up(self):
        """0x1200 = 1 and the measured sequence, on port 0's parked display:
        mode_change and the first frame; the registers."""
        s, p = self.s, self.p
        d = self.parked[0]
        if not hasattr(self, "efi_fb"):
            self.efi_fb = EFI(s)
        w, h = p.choice(EFI_SIZES)
        lab = self.mark("EFI on port 0's display %d, %dx%d" % (d.index, w, h))
        raw = self.efi_frame(w, h)
        self.efi = dict(w=w, h=h, raw=raw)
        self.efi_fb.setup(w, h, EFI_STRIDE)
        self.efi_bound = d
        self.efi_present(d)
        self.events(lab)
        self.efi_regs(d, lab)
        self.encode(d, lab)
        self.did("efi frame")

    def efi_tick(self):
        """The firmware (or the console before the driver's takeover) draws
        a frame: a trace hit presents it; now and then a mode change."""
        s, p = self.s, self.p
        d = self.efi_bound
        if d is None:
            return
        if p.chance(0.25):
            w, h = p.choice(EFI_SIZES)
            lab = self.mark("EFI mode %dx%d" % (w, h))
            raw = self.efi_frame(w, h)
            self.efi.update(w=w, h=h, raw=raw)
            s.w(abi.R_EFI_MODE, w | h << 16)
        else:
            lab = self.mark("an EFI frame")
            self.efi["raw"] = self.efi_frame(self.efi["w"], self.efi["h"])
            s.trace_fire(self.efi_fb.gpa(), abi.PAGE)
        self.efi_present(d)
        self.events(lab)
        self.encode(d, lab)
        self.did("efi frame")

    def efi_down(self):
        """The driver's handover: the mode it selects read back, then 0x1200
        = 0, which zeroes the registers; the display keeps a frame over a
        mapping EFI released, never encoded again."""
        s, p = self.s, self.p
        d = self.efi_bound
        lab = self.mark("EFI off")
        if d is not None:
            sel = p.below(len(d.modes) + 1) if d.modes else 0
            s.w(abi.R_EFI_MODE_SELECT, sel)
            s.r(abi.R_EFI_MODE_SELECTED, "%s: mode %d" % (lab, sel))
            m = d.modes[sel] if d.modes and sel < len(d.modes) else None
            self.predict(val=hex(m[0] | m[1] << 16) if m else hex(M32))
            d.frame = ("stale",)
        s.w(abi.R_EFI_ENABLE, 0)
        self.efi_bound = None
        self.efi = None
        self.events(lab)
        for reg in (abi.R_EFI_ENABLE, abi.R_EFI_START, abi.R_EFI_LENGTH, abi.R_EFI_MODE):
            s.r(reg, lab)
            self.predict(val="0x0")

    # ---- the driver ----

    def negotiate(self):
        """What a guest probes before it settles on 245: 0x100 is answered
        with 0xf5; a request with bit 31 set is ignored; 0xf4 is not a
        version and is answered with the one below it that is
        (set_binary_version)."""
        s, p = self.s, self.p
        lab = self.mark("version negotiation")
        asks = p.sample([0x100, 0x800000F0, 0xF4], 1 + p.below(3))
        for v in asks:
            s.w(abi.R_VERSION, v)
            s.r(abi.R_VERSION, "%s: asked 0x%x" % (lab, v))
            if v == 0x100:
                self.predict(val=hex(0xF5))
        s.w(abi.R_VERSION, self.version)
        s.r(abi.R_VERSION, lab + ": settled")
        self.predict(val=hex(self.version))
        s.r(abi.R_PORT_COUNT, lab + ": port count")
        self.predict(val=hex(self.nports))
        s.r(abi.R_PORT_TYPES, lab + ": port types")
        self.predict(val="0x0")

    def device_info(self):
        """GetDeviceInfo (58) {maxKey 45, maxCount 64, page} into a page of
        0xa5: host-keyed."""
        s = self.s
        lab = self.mark("device info")
        page = s.alloc_page()
        s.fill(s.pa(page), 0x180, 0xA5)
        self.rsub(abi.OP_GET_DEVICE_INFO, struct.pack("<III", 45, 64, s.pfn(page)), lab)
        s.dump(s.pa(page), 0x180, lab)

    def driver_up(self):
        """Root page, root FIFO, a NOP, task 0 and its object list
        (randdisplay's bring_up), without the IOSurface host device: the
        window server starts it later."""
        on = self.surfaces_on
        self.surfaces_on = False
        self.mark("root page, root FIFO, task 0")
        self.bring_up(first=True)
        self.surfaces_on = on

    def connect_head(self, port):
        """Opcode 1 on a fresh page with every interrupt enabled."""
        s = self.s
        pg = PageM(s.alloc_page(), None)
        self.guest_write(pg, O_ENABLED, "I", 0xF)
        lab = self.mark("connect port %d" % port)
        self.submit(abi.OP_DISPLAY_SET_SHARED_STATE_PAGE, p_set_shared(port, s.pfn(pg.ram)), lab)
        self.attach_page(port, pg)
        self.last_page[port] = pg
        self.settle(port, lab)
        self.root_dump(lab)
        self.did("connect")

    def acks(self):
        """The driver acks every update pending on a page."""
        for port in sorted(self.nubs):
            for _ in range(3):
                if not self.nubs[port].page.b[O_ACKP]:
                    break
                self.a_ack(port)
                self.did("ack")

    def show(self, port, show):
        lab = self.mark("cursor show %d on port %d" % (show, port))
        self.submit(abi.OP_DISPLAY_CURSOR_SHOW, p_show(port, show), lab)
        self.m_show(self.nubs[port], show)
        self.settle(port, lab)

    def cursor_up(self, port):
        self.a_glyph(port)
        self.show(port, 1)
        self.a_move(port)
        self.did("cursor")

    def heads(self, client=True):
        return [p for p in sorted(self.nubs) if not client or self.nubs[p].client is not None]

    # ---- the window server ----

    def window_server(self):
        """The IOSurface host device, every surface mapped, and the
        compositor's task: a mapper buffer (type 15, reims-vgpu-core's
        resources::buffer::mapper_ref) over each surface."""
        s = self.s
        self.mark("the IOSurface host device and the surfaces")
        self.iosfc.start()
        for mid in sorted(self.surf):
            self.map_surface(mid)
        t = self.comp = GPUTask(s, tid=self.ntasks + 2, pages=16)
        t.mbuf = {mid: t.ol.add(abi.T_MAPPER_BUFFER, mapper_ref_rec(mid)) for mid in sorted(self.surf)}
        t.win, t.win_va = t.buffer(abi.PAGE)
        lab = self.mark("the compositor's task %d" % t.tid)
        self.define_task(t, lab)
        self.gevents(lab)

    def idle_fifos(self):
        return [f for f in sorted(self.defined) if f not in self.faulted and not self.pend.get(f)]

    def compose(self, port=None, mid=None):
        """The compositor: the guest writes a window's pixels into its
        buffer, the GPU copies them into a surface (blit 301 onto the
        surface's mapper buffer, the source invalidated so the host reads
        the guest's new bytes) on the root FIFO or an idle child, then
        opcode 8 presents the surface on a head, from the root FIFO or
        another idle child, with a barrier on the copy's stamp. The copy
        shows at once on every display that shows the surface (a frame is
        a view); the present's new_frame follows. Pixels have alpha 0xff:
        the display copies alpha through where it has no gamma table
        (measured, a fill of 00 encodes 00000000)."""
        s, p = self.s, self.p
        heads = self.heads()
        if port is None:
            if not heads:
                return self.gpu_work()
            port = p.choice(heads)
        fillable = [m for m in sorted(self.surf) if self.surf[m].mapped and self.surf[m].fmt != "RGhA"]
        if mid is None:
            if not fillable:
                return self.a_present_mapped(port)
            mid = p.choice(fillable)
        sf = self.surf[mid]
        n = sf.w * sf.h * 4
        raw = pixels(p, sf.w, sf.h, sf.fmt)
        idle = self.idle_fifos()
        ffid = p.choice([0] + idle)
        pfid = p.choice([0] + [f for f in idle if f != ffid])
        g = self.pick_gamma()
        t = self.comp
        t.write(t.win_va, raw)
        cb = t.cmdbuf(seg(abi.ENC_BLIT, b_copy_buf_buf(t.win, t.mbuf[mid], 0, 0, n)))
        lab = self.mark("compose: a window copied into surface %d (%dx%d %s) on FIFO %d, presented on port %d "
                        "from FIFO %d%s" % (mid, sf.w, sf.h, sf.fmt, ffid, port, pfid,
                                            ", gamma %s %d%s" % (g.kind, g.n, " (refused)" if g.bad else "")
                                            if g else ""))
        ff, pf = self.fifo(ffid), self.fifo(pfid)
        fst = ff.cmd(E3, exec_payload(t.tid, [cb], [(t.win, abi.RES_INVALIDATE_HOST), (t.mbuf[mid], 0)]))
        pst = pf.cmd(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(port, mid, *(g.payload() if g else ())),
                     barriers=[(ff.stamp_index, fst)])
        pf.doorbell()
        if pf is not ff:
            ff.doorbell()
        pf.wait_stamp(pst, label=lab + ": the present")
        self.predict(ok=True)
        ff.wait_stamp(fst, label=lab + ": the copy")
        self.predict(ok=True)
        t.reset_cmdbufs()
        sf.raw = raw
        self.sig(ff.stamp_index)
        self.sig(pf.stamp_index)
        self.m_present_mapped(port, mid, g)
        self.model_cmp(s.pa(sf.page), sf.raw, lab + ": surface %d" % mid)
        self.after_present(port, lab)
        for d in self.live_displays():
            if d.frame == ("surface", mid) and d is not self.plugged.get(port):
                self.encode(d, lab + ": another head's view")
        self.fills += 1
        self.did("compose")

    # ---- GPU ----

    def gpu_work(self, gate_p=None):
        """A batch of app work (the marathon's dag step)."""
        if self.nslots > 700:
            return
        keep = self.gate_p
        if gate_p is not None:
            self.gate_p = gate_p
        try:
            self.g_dag(gate_ok=gate_p != 0)
        finally:
            self.gate_p = keep
        self.idle_tasks()
        self.did("gpu")

    def held(self):
        return [n for n in self.pending() if "gate" in n.locks]

    def release(self):
        if self.held():
            self.g_release()
            self.idle_tasks()
            self.did("release")

    def busy(self, allow_fault=True):
        """What is running when a phase ends: batches held on the gate,
        batches that run free, a fault on an idle child."""
        p = self.p
        what = []
        if p.chance(0.6):
            what.append("held")
        if p.chance(0.4):
            what.append("free")
        if allow_fault and self.faults_on and p.chance(0.3) and self.idle_fifos() and len(self.faulted) < 2:
            what.append("fault")
        self.s.mark("%s: busy at the transition: %s" % (self.phase, ", ".join(what) or "nothing"))
        for w in p.sample(what, len(what)):
            if w == "held":
                self.gpu_work(gate_p=1.0)
            elif w == "free":
                self.gpu_work(gate_p=0)
            elif self.idle_fifos() and len(self.faulted) < 2:
                self.g_fault()
                self.did("fault")
        return what

    def resolve(self):
        """After a transition the guest opens the gate and recovers its
        faults, in the seed's order."""
        p = self.p
        acts = []
        if self.held():
            acts.append("release")
        if self.faulted:
            acts.append("recover")
        for a in p.sample(acts, len(acts)):
            if a == "release":
                self.release()
            else:
                while self.faulted:
                    self.g_recover()
                    self.idle_tasks()
                    self.did("recover")

    def quiet(self):
        """No fault outstanding (pause and suspend need it)."""
        while self.faulted:
            self.g_recover()
            self.idle_tasks()

    # ---- steady state ----

    def steady(self, n):
        for _ in range(n):
            self.steady_step()

    def steady_step(self):
        p = self.p
        heads = self.heads()
        table = [("gpu", 8), ("check", 1)]
        if heads:
            table += [("move", 8), ("present", 5), ("misc", 4)]
            if self.surfaces_on:
                table.append(("compose", 10))
        if self.held():
            table.append(("release", 5))
        if self.faults_on and self.idle_fifos() and len(self.faulted) < 2:
            table.append(("fault", 1))
        if self.faulted:
            table.append(("recover", 6))
        table += [("life", 1), ("scr_move", 1)]
        kind = p.weighted(table)
        if kind == "gpu":
            self.gpu_work()
        elif kind == "check":
            self.g_check()
        elif kind == "move":
            self.a_move(p.choice(heads))
        elif kind == "present":
            self.a_present_backing(p.choice(heads))
        elif kind == "compose":
            self.compose()
        elif kind == "misc":
            self.misc()
        elif kind == "release":
            self.release()
        elif kind == "fault":
            self.g_fault()
        elif kind == "recover":
            self.g_recover()
        elif kind == "life":
            self.g_life()
        elif kind == "scr_move":
            self.g_scr_move()
        if kind != "misc":
            self.did(kind)
        self.idle_tasks()

    def misc(self):
        p = self.p
        heads = self.heads()
        k = p.choice(MISC)
        self.did("misc " + k)
        if k == "rewrite_backing":
            self.a_rewrite_backing()
        elif k == "encode":
            self.a_encode(p.choice(heads))
        else:
            getattr(self, "a_" + k)(p.choice(heads))

    # ---- hot plug, sleep, pause, suspend ----

    def hotplug(self):
        """The host changes the displays: an empty port gets a display, one
        is unplugged and a new one plugged in, a mode list changes; the
        driver acks each update, with steady steps between."""
        p = self.p
        acts = []
        for port in range(self.nports):
            if port not in self.plugged:
                acts.append(("plug", port))
        port = p.below(self.nports)
        if port in self.plugged and p.chance(0.7):
            acts.append(("replug", port))
        for port in range(self.nports):
            if p.chance(0.6):
                acts.append(("modes", port))
        if not acts:
            acts.append(("modes", 0))
        for k, port in p.sample(acts, len(acts)):
            if k == "plug" and port not in self.plugged:
                self.a_plug(port)
            elif k == "replug":
                if port in self.plugged:
                    self.a_unplug(port)
                    self.acks()
                    if p.chance(0.5):
                        self.steady_step()
                self.a_plug(port)
            elif k == "modes":
                if port not in self.plugged:
                    self.a_plug(port)
                else:
                    self.a_set_modes(port)
            self.did("hotplug " + k)
            if p.chance(0.3):
                self.steady_step()
            self.acks()
            if p.chance(0.5):
                self.steady_step()
        for port in range(self.nports):
            if port in self.plugged and port not in self.nubs:
                self.connect_head(port)
                self.acks()

    def sleep_heads(self, asleep):
        for port in self.p.sample(self.heads(client=False), len(self.heads(client=False))):
            nub = self.nubs[port]
            if nub.asleep != bool(asleep):
                self.a_sleep(port)
                self.did("sleep" if asleep else "wake")

    def pause_cycle(self):
        """The host pauses the device (QEMU's migration pre-copy would), maybe
        changes a display while it is paused, and unpauses."""
        self.a_pause()
        if self.p.chance(0.5):
            port = self.p.below(self.nports)
            self.a_host_while_paused(port)
        self.a_unpause()
        self.acks()
        self.did("pause")

    # ---- shutdown ----

    def shutdown(self):
        s, p = self.s, self.p
        self.gpu_final()
        for port in self.heads(client=False):
            if self.nubs[port].page.b[0xF20]:
                self.show(port, 0)
        if self.sleep_first:
            self.sleep_heads(1)
        for port in p.sample(sorted(self.nubs), len(self.nubs)):
            self.a_disconnect(port)
            self.did("disconnect")
        if self.comp is not None:
            t = self.comp
            lab = self.mark("the window server exits: its mapper buffers deleted (37), task %d deleted" % t.tid)
            for mid in sorted(t.mbuf):
                op, pl = delete_resource(t.tid, t.mbuf[mid])
                self.rsub(op, pl, lab + ": 37 surface %d's buffer" % mid)
            self.rsub(abi.OP_DELETE_TASK, struct.pack("<I", t.tid), lab + ": 32")
            self.gevents(lab)
            shown = set(d.frame[1] for d in self.disps if d.alive and d.frame and d.frame[0] == "surface")
            for mid in sorted(self.surf):
                sf = self.surf[mid]
                if not sf.mapped:
                    continue
                lab = self.mark("unmap surface %d%s" % (mid, " (still shown)" if mid in shown else ""))
                idx = self.iosfc.submit((CMD_UNMAP, mid))
                self.iosfc.status(idx, lab)
                if mid not in shown:
                    self.predict(hex=struct.pack("<IIII", CMD_UNMAP, mid, SUCCESS, 0).hex())
                    sf.mapped = False
            self.checkpoint(lab, pages=False)
        lab = self.mark("delete the backings (54)")
        for b in self.backings:
            self.submit(abi.OP_DELETE_IOSURFACE_BACKING2, struct.pack("<II", b.ref, 0), lab + ": %d" % b.ref)
            b.host = None
        self.checkpoint(lab, pages=False)
        lab = self.mark("the FIFOs and tasks torn down")
        for fid in p.sample(sorted(self.defined), len(self.defined)) + [GATE]:
            self.delete(fid, lab + ": delete child %d" % fid)
        self.defined = set()
        for t in p.sample(self.tasks + [self.ctl], len(self.tasks) + 1):
            self.rsub(abi.OP_DELETE_TASK, struct.pack("<I", t.tid), lab + ": delete task %d" % t.tid)
        self.rsub(abi.OP_DELETE_TASK, struct.pack("<I", 0), lab + ": delete task 0")
        self.gevents(lab)
        s.dump(s.pa(s.root_page), 0x40, lab + ": root page task words")
        s.dump(s.pa(s.root_page, abi.ROOT_CHILD_NODE), abi.CHILD_NODE_SIZE * GATE, lab + ": child nodes 1-20")
        self.observe(lab)
        self.final_check()
        if self.disable_fifo:
            lab = self.mark("the root FIFO disabled (0x1000 = 0)")
            s.w(abi.R_FIFO_ENABLE, 0)
            s.r(abi.R_FIFO_ENABLE, lab)
            self.predict(val="0x0")
            self.events(lab)
        if self.reboot:
            self.enter("reboot")
            lab = self.mark("reset (a PCI reset: the guest reboots)")
            s.api("reset", label=lab)
            self.r1014 = self.r1018 = 0
            self.checkpoint(lab, pages=False)
            self.root_dump(lab)
            if 0 not in self.parked:
                self.a_plug(0)
            self.efi_up()
            for _ in range(p.below(3)):
                self.efi_tick()
            if p.chance(0.5):
                self.efi_down()
            self.events("rebooted to the firmware")

    # ---- the program ----

    def build(self):
        s, p = self.s, self.p
        self.root_words = {}
        self.t0 = GPUTask(s, tid=0, pages=160)
        self.seq_regions = []
        self.lay_task0()
        if self.surfaces_on:
            self.lay_surfaces()
        try:
            self.program()
        except Death:
            pass
        self.s.doc = self.doc.strip() + "\n\nThis program: " + self.boot_summary()
        return self.s

    def program(self):
        s, p = self.s, self.p
        self.enter("firmware")
        if self.efi_first:
            self.efi_up()
            for _ in range(p.below(3)):
                self.efi_tick()
        self.enter("driver")
        self.negotiate()
        if not self.efi_first:
            self.efi_up()
        self.driver_up()
        if p.chance(0.6):
            self.efi_tick()
        if self.info_first:
            self.device_info()
        if self.gpu_early:
            self.gpu_setup()
            if p.chance(0.6):
                self.efi_tick()
        if not self.info_first:
            self.device_info()
        ports = p.sample([q for q in range(self.nports) if q in self.plugged], len(self.plugged))
        if self.late_port and 1 in ports:
            self.connect_head(1)
            if not self.defer_ack:
                self.acks()
            ports.remove(1)
            self.efi_tick()
        if self.gpu_early:
            self.busy(allow_fault=True)
            self.efi_tick()
        self.efi_down()
        if not self.gpu_early:
            self.gpu_setup()
        for port in ports:
            self.connect_head(port)
            if not self.defer_ack:
                self.acks()
        self.acks()
        self.resolve()
        cursor_ports = self.heads()
        for port in p.sample(cursor_ports, 1 + p.below(max(len(cursor_ports), 1))) if cursor_ports else ():
            self.cursor_up(port)
        self.enter("window server")
        if self.surfaces_on:
            self.window_server()
            for port in self.heads():
                self.compose(port)
        else:
            for port in self.heads():
                self.a_present_backing(port)
                self.did("present")
        self.enter("steady")
        self.steady(self.n_steady[0])
        if self.death == "gamma_one":
            return self.die()
        self.enter("hotplug")
        self.busy()
        self.hotplug()
        if self.death == "mode_count_over":
            return self.die()
        self.resolve()
        self.enter("sleep")
        self.busy()
        self.sleep_heads(1)
        if self.death == "cursor_paused":
            return self.die()
        self.steady(self.n_steady[1])
        self.sleep_heads(0)
        self.resolve()
        self.enter("suspend")
        self.quiet()
        if self.pause_on:
            self.busy(allow_fault=False)
            self.pause_cycle()
            self.resolve()
            self.quiet()
        if self.death in ("doorbell_paused", "pause_twice"):
            return self.die()
        self.busy(allow_fault=False)
        self.g_suspend()
        self.did("suspend")
        self.acks()
        self.resolve()
        self.enter("steady after the resume")
        if self.surfaces_on:
            for port in self.heads():
                if self.p.chance(0.5):
                    self.compose(port)
        self.steady(self.n_steady[2])
        self.enter("shutdown")
        self.busy()
        self.shutdown()

    def die(self):
        self.enter("death: " + self.death)
        self.gpu_before_death()
        self.final_death()

    def boot_summary(self):
        return "%d port%s, %s, EFI %s, GPU set-up %s EFI goes off, %d children, %d task%s; steady %s; %s%s; %s." % (
            self.nports, "s" if self.nports > 1 else "",
            "mapper surfaces and a compositor" if self.surfaces_on else "no mapper surfaces",
            "at version 0" if self.efi_first else "after the version", "before" if self.gpu_early else "after",
            self.nchildren, self.ntasks, "s" if self.ntasks > 1 else "", "/".join(str(n) for n in self.n_steady),
            ", ".join("%s %d" % kv for kv in sorted(self.phist.items())),
            ("; a pause before the suspend" if self.pause_on else ""),
            ("ends in a death (%s)" % self.death if self.death else
             "ends in a reboot to the firmware" if self.reboot else "ends released"))


# ---- entry points ----

def boot(seed, doc="", length="medium", death=None):
    b = Boot(seed, doc, length, death)
    b.build()
    return b


def build(seed, doc, length="medium", death=None):
    return boot(seed, doc, length, death).s


def check(paths=None):
    """Every expected file of interplay_boot against the model."""
    import glob
    import os
    from scenarios.interplay_boot import SEEDS, __doc__ as doc
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = paths or os.path.join(here, "expected")
    bad = 0
    for seed, length, death in SEEDS:
        b = boot(seed, doc, length, death)
        for pth in sorted(glob.glob(os.path.join(root, "interplay_boot", b.name, "*.jsonl"))):
            lines = [json.loads(l) for l in open(pth) if l.strip() and not l.startswith("#")]
            obs = [o for o in lines if "ev" in o or "kind" in o]
            complaints = check_obs(b, obs)
            for o in obs:
                if o.get("ev") == "cmp" and o.get("equal") is False:
                    complaints.append("cmp differs: %s" % o.get("label"))
            for c in complaints:
                print("%s: %s" % (os.path.relpath(pth, here), c))
            bad += len(complaints)
    return bad


if __name__ == "__main__":
    import sys
    sys.exit(1 if check(sys.argv[1] if len(sys.argv) > 1 else None) else 0)
