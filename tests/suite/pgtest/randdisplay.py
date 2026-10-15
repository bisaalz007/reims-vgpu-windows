"""Seeded random walks over display state (scenarios/interplay_displaywalk.py),
with a model that predicts every host-visible observation.

A program drives one or two display ports (a port count the guest reads at
0x122c, which below version 245 is only "nonzero")
and the EFI framebuffer through random legal actions:

  - the guest: connect a port on a fresh page or on the page it used before
    (opcode 1), disconnect it, ack (2), present a task-0 backing (6, 7) or an
    IOSurface mapper surface (8) with or without a gamma table (sizes 2 to
    1024), a cursor glyph (4; 1 to 256 wide and high, strides with padding,
    checksum or none), show and hide (5), move with the doorbell 0x1220 and
    +0xe00/+0xe04, a colour matrix (10), an ICC profile (12), sleep and wake
    (9), compositor parameters (11), protection options written at +0x68,
    its mode count zeroed or a bogus state written, raisedInterrupts cleared,
    backings rewritten and deleted (54), surfaces rewritten, unmapped and
    mapped again, refused presents (no page, an unmapped mapping, a gamma
    table over 1024, a pixel format opcode 7 refuses; a gamma table with a
    wrong checksum or a mapped length of 2^62, which the display refuses
    after mode_change and before the frame) and
    commands its version gates (10 below 40 and 12 below 208 signal and do
    nothing), and a field the host owns overwritten (the settings, the
    display UUID, a mode's width or flags), which the next update puts
    back and counts as a change (the fields in turn across the seeds);
  - the host: hot unplug (release a display) and plug (a new display, a new
    serial), a new mode list (set_modes, sometimes the same one, sometimes
    130 modes), frames encoded, rvg_display_cursor_position and present counts
    read;
  - the lifecycle: pause and unpause, with display plugs, unplugs, mode
    lists and guest commands rung while paused; suspend onto a fresh device
    (the target's displays chosen per program) and resume; reset and stop,
    with the guest's set-up again; the EFI framebuffer at boot and after a
    reset, framed through trace ranges, its mode changed, then turned off;
  - a final action that kills the host, in some programs (docs/testing.md
    7.3, 7.4): a one-entry gamma table, the doorbell or
    rvg_display_cursor_position while paused, a second pause, a guest mode
    count above 128 at the next update.

The model is the guest display's and the display's code restated
(reims-vgpu-core's display::guest_display, display::frame, display::gamma,
display::efi and the device's display plumbing, device::displays): the
shared page byte for byte (the state machine, the settings and mode list it
writes, presents, the colour matrix indices, glyph, show, gamma, ICC and
sleep records, restore), each display's last mode (mode_change), its current
frame, gamma table, colour matrix and present counts, the interrupts each
poke and stamp raises, and the pixels an encode draws. It predicts, for
every checkpoint:

  - the callbacks (kind, display, and their contents: mode_change's size and
    format, cursor_glyph's RGBA bytes and SHA-256, cursor_show's flag) and
    the interrupt count; each callback kind is also waited for by its
    running total, so a missing one shows as a failed wait;
  - every shared page: the model's bytes go into a mirror page of the
    guest's and a cmp compares all 16 KiB with the live page; a wait on
    the page's expectedAck and state at the model's values after each step;
  - 0x1014 (the display ports poked since the last read) and 0x1018;
  - the root page's display words;
  - the encoded frame: its SHA-256 and pixels, from the surface's bytes
    through the display's pipeline (below);
  - present counts, rvg_display_cursor_position, the EFI mode registers, the
    IOSurface host device's statuses, and the outcome (completed, or died after
    a final death action).

pgtest.randdisplay.check() rebuilds each program and compares every
prediction with the expected file (python3 -m pgtest.randdisplay).

What the model takes from measurement rather than from the code: the
display's fragment shaders (displayPresentFragment and its gamma variant in
crates/reims-vgpu-metal/src/gpu/shaders.metal). Measured with this module's
first programs (and consistent with interplay_scanout_gamma_matrix_frames):
until a gamma table arrives the display copies the frame (the colour matrix is
not applied); once one has arrived (the gamma pipeline, for good) each channel
is looked up in its table and then the colour matrix row c gives
output c = m[c][0] r + m[c][1] g + m[c][2] b + m[c][3]; the result is stored as
unorm8, rounded to nearest (0.5 gives 0x80). The model predicts pixels only
where the lookup cannot depend on the sampler's rounding: tables that are
constant per channel, 256- and 1024-entry identity ramps over 8-bit surfaces,
and the two-entry ramp (a threshold at 0.5) over values away from 0.5. Surface
bytes come from a palette whose values are multiples of 4 and never 0x80,
matrix coefficients from {0, 1/4, 1/2, 3/4, 1} and offsets from {0, 1/4}, so no
result is a rounding tie. Other combinations (a ramp over an 'RGhA' surface)
are encoded and recorded, not predicted.

Invariants the generator keeps, so every checkpoint is deterministic:
  - One display is active a step; its callbacks are waited for by running
    totals before the checkpoint, so new_frame (a DATA_ADD source) never
    coalesces: at most one new_frame a display a step. A resume replays on
    every port in port order, from one thread.
  - Nothing touches a guest display while paused (7.3); pause and unpause
    pair.
    Commands rung while paused run after the unpause's state machines.
  - EFI runs only while port 0's display is parked (no page), is bound
    only at creation, and is turned off before the guest connects port 0,
    a reset, a stop or a suspend; after it is off, its frame is not encoded
    (the view is over a mapping it released).
  - A mapper surface is unmapped only while no display shows it (else
    Busy, docs 6.21) and presented again only after a map.
  - Mapper surfaces need a memory map: programs with them run in map and
    isolate, the others in all three setups.

Nothing here is imported from the library's code.
"""
import hashlib
import json
import math
import struct

from . import abi
from .abi import PAGE
from .display_guest import (display, icc_profile, p_ack, p_compositor, p_glyph, p_icc, p_present2, p_present3,
                            p_props, p_set_shared, p_show, p_sleep, p_swap)
from .efi_guest import EFI
from .iosfc_guest import CMD_MAP, CMD_UNMAP, IOSFC, SUCCESS, desc_v2
from .randgen import Rng
from .scenario import SETUPS, Scenario
from .task import GPUTask

MAPPED = ("map", "isolate")
PAL = (0x00, 0x20, 0x40, 0x60, 0xA0, 0xC0, 0xE0, 0xFC)   # multiples of 4, far from 0x80
HALF_PAL = (0.0, 0.25, 0.75, 1.0)
COEF = (0.0, 0.25, 0.5, 0.75, 1.0)
GAMMA_C = (0.0, 0.25, 0.5, 0.75, 1.0)
FMT = {"BGRA": 0x42475241, "RGBA": 0x52474241, "RGhA": 0x52476841}
NAMES = ("head", "panel", "Studio", "proj", "tv", "LCD-7", "wall")
SIZES_8 = ((16, 8), (8, 4), (32, 16), (4, 4), (12, 8), (20, 4))
SIZES_H = ((4, 4), (8, 2), (2, 6))
MODES = ((64, 32, 60), (16, 8, 60), (1920, 1080, 60), (1280, 800, 59.94), (32, 16, 30), (800, 600, 75),
         (2560, 1440, 120), (8, 4, 24))
EFI_SIZES = ((16, 8), (32, 8), (8, 16), (24, 4))
EFI_STRIDE = 256
VERSIONS = (245, 245, 245, 245, 238, 208, 109, 108, 40, 22, 21, 10)
DEATHS = ("gamma_one", "doorbell_paused", "cursor_paused", "pause_twice", "mode_count_over")
M32 = 0xFFFFFFFF

# The page (reims-vgpu-core's display::page).
O_SERIAL, O_NAME, O_PORT, O_MMR, O_GLYPHMAX, O_1C, O_HANDLERS = 0x0, 0x4, 0x12, 0x14, 0x18, 0x1C, 0x20
O_MMF, O_CS, O_ORIGIN, O_SCALE, O_EPOCH, O_CONN, O_PROT, O_NITS = 0x24, 0x2C, 0x4C, 0x50, 0x58, 0x60, 0x68, 0x70
O_RAISED, O_ENABLED, O_EXPACK, O_COUNT, O_MODES = 0x100, 0x104, 0x200, 0x208, 0x210
O_CM, O_CM_READ, O_CM_FORCE, O_CM_WRITE, O_CM_DIRTY, O_UUID = 0xA10, 0xA70, 0xA71, 0xA72, 0xA73, 0xA74
O_CURSOR, O_GSHOW = 0xE00, 0xE04
O_GLYPH, O_GLEN, O_GSTRIDE, O_GF18, O_GSUM, O_HSHOW, O_GW = 0xF00, 0xF08, 0xF10, 0xF18, 0xF1C, 0xF20, 0xF22
O_GAMMA, O_PCOUNT, O_PSURF, O_FA4, O_ACKP, O_PMAP, O_ASLEEP, O_LASTACK, O_STATE, O_ICC = (
    0xF80, 0xF98, 0xFA0, 0xFA4, 0xFA9, 0xFB0, 0xFB8, 0xFBC, 0xFC0, 0xFC8)
# The display descriptor's defaults (display::descriptor) and the runner's
# 400 x 300 mm (runner/pgrun.m display_descriptor).
COLOR_SPACE = (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.312727, 0.329023)


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def fbits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def half(v):
    return struct.pack("<e", v)


def u8(v):
    """A float stored as unorm8 (0.5 gives 0x80, measured)."""
    v = min(max(v, 0.0), 1.0)
    return int(math.floor(v * 255.0 + 0.5))


class Death(Exception):
    """The model's host died: the program ends here."""


class Refused(Exception):
    """An exception the host throws inside a present, which the FIFO
    catches and logs before it signals the stamp."""


# ---- the model: pages, displays, guest displays, surfaces ----

class PageM:
    """A shared page as guest memory holds it: the guest's writes and the
    host's. mirror: a page of the guest's the model's bytes are copied into
    for a cmp."""

    def __init__(self, ram_page, mirror):
        self.ram = ram_page
        self.b = bytearray(PAGE)
        self.mirror = mirror
        self.mirrored = bytearray(PAGE)

    def get(self, off, fmt):
        return struct.unpack_from("<" + fmt, self.b, off)[0]

    def put(self, off, fmt, v):
        struct.pack_into("<" + fmt, self.b, off, v)


class HostSurf:
    """An IOSurface a frame views, with fixed contents (a backing's copy)."""

    def __init__(self, w, h, fmt, raw):
        self.w, self.h, self.fmt, self.raw = w, h, fmt, bytes(raw)


class Backing:
    def __init__(self, ref, va, w, h, fmt, raw):
        self.ref, self.va, self.w, self.h, self.fmt = ref, va, w, h, fmt
        self.raw = raw          # the guest pages' pixels
        self.host = None        # HostSurf: the host copy, made at the first present


class Surface:
    """A mapper surface: an IOSurface over guest pages, a live view."""

    def __init__(self, mid, page, root, w, h, fmt, raw):
        self.mid, self.page, self.root, self.w, self.h, self.fmt, self.raw = mid, page, root, w, h, fmt, raw
        self.mapped = False


class Gamma:
    """A gamma table in task 0: kind const (per-channel values), ramp or
    thresh (the two-entry ramp)."""

    def __init__(self, n, kind, consts, va, csum, bad=False, mapped=None):
        self.n, self.kind, self.consts, self.va, self.csum, self.bad = n, kind, consts, va, csum, bad
        self.mapped = 12 * n if mapped is None else mapped

    def payload(self):
        return (self.va, self.mapped, self.n, self.csum)


class DispM:
    """A display (Display) the host created: its index in the runner's
    list, its spec, and the display state the model needs."""

    def __init__(self, index, port, serial, name, modes):
        self.index, self.port, self.serial, self.name = index, port, serial, name
        self.modes = [tuple(m) for m in modes] if modes else None   # the display's mode_list
        self.nub = None
        self.w = self.h = self.fmt = 0
        self.prot = 0
        self.cs_dirty = False
        self.frame = None        # ("host", HostSurf) | ("surface", mid) | ("efi",) | ("stale",)
        self.encoded = False
        self.gpc = self.hpc = 0
        self.gamma = None        # the last Gamma once one arrived: the gamma pipeline
        self.matrix = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0]
        self.alive = True

    def spec(self):
        return display(self.port, serial=self.serial, name=self.name, modes=self.modes)


class NubM:
    def __init__(self, port, page):
        self.port, self.page = port, page
        self.client = None
        self.modelist = None
        self.paused = self.will_restore = self.did_restore = False
        self.cs_dirty = False
        self.asleep = False


# ---- pixels ----

def chans(fmt, raw, w, h):
    """[(r, g, b)] per pixel: bytes 0-255 for 8-bit formats, floats for RGhA."""
    out = []
    if fmt == "RGhA":
        for i in range(w * h):
            r, g, b, _ = struct.unpack_from("<4e", raw, 8 * i)
            out.append((r, g, b))
    else:
        for i in range(w * h):
            p = raw[4 * i:4 * i + 4]
            out.append((p[2], p[1], p[0]) if fmt == "BGRA" else (p[0], p[1], p[2]))
    return out


def render(d, w, h, fmt, raw):
    """What an encode of a w x h frame at the frame's size draws, BGRA8, or
    None where the model does not predict it."""
    px = chans(fmt, raw, w, h)
    out = bytearray()
    eight = fmt != "RGhA"
    for (r, g, b) in px:
        if d.gamma is None:
            if eight:
                out += bytes((b, g, r, 0xFF))
            else:
                out += bytes((u8(b), u8(g), u8(r), 0xFF))
            continue
        gm = d.gamma
        vals = []
        for c, v in enumerate((r, g, b)):
            x = v / 255.0 if eight else v
            if gm.kind == "const":
                vals.append(gm.consts[c])
            elif gm.kind == "thresh":
                vals.append(1.0 if x > 0.5 else 0.0)
            else:
                if not eight:
                    return None
                vals.append(x)
        m = d.matrix
        o = [m[4 * c] * vals[0] + m[4 * c + 1] * vals[1] + m[4 * c + 2] * vals[2] + m[4 * c + 3] for c in range(3)]
        out += bytes((u8(o[2]), u8(o[1]), u8(o[0]), 0xFF))
    return bytes(out)


def pixels(rng, w, h, fmt):
    if fmt == "RGhA":
        return b"".join(half(rng.choice(HALF_PAL)) + half(rng.choice(HALF_PAL)) + half(rng.choice(HALF_PAL))
                        + half(1.0) for _ in range(w * h))
    return b"".join(bytes((rng.choice(PAL), rng.choice(PAL), rng.choice(PAL), 0xFF)) for _ in range(w * h))


def cstr_differ(a, b):
    """strncmp(a, b, len) != 0: compared up to the first NUL."""
    for x, y in zip(a, b):
        if x != y:
            return True
        if x == 0:
            return False
    return False


# ---- the program ----

class Walk:
    """One program: generated and emitted in one pass; the model runs
    alongside and records a prediction for every observation it can."""

    def __init__(self, seed, steps, doc):
        self.seed, self.steps = seed, steps
        r = self.r = Rng(seed * 7919 + 17)
        self.nports = 1 + r.below(2)
        self.version = r.choice(VERSIONS)
        self.surfaces_on = r.chance(0.6)
        self.efi_on = r.chance(0.45)
        self.death = r.choice(DEATHS) if r.chance(0.3) else None
        self.setups = MAPPED if self.surfaces_on else SETUPS
        self.name = "interplay_displaywalk_%d" % seed
        cfg_disp = [self.new_spec(p, 1000 + p) for p in range(self.nports)]
        desc = {"trace": True}
        if self.nports > 1:
            desc["port_count"] = self.nports
        self.s = Scenario(self.name, setups=self.setups, host=True, watchdog_s=60 + 3 * steps,
                          displays=[d for d in cfg_disp], descriptor=desc, iosfc=self.surfaces_on,
                          ram=None, doc="")
        self.doc = doc
        self.pred = {}
        self.desc_steps = []
        # the model
        self.disps = []                 # DispM by runner index
        for p, spec in enumerate(cfg_disp):
            self.disps.append(DispM(p, p, spec["serial"], spec["name"], [tuple(m) for m in spec["modes"]]))
        self.parked = {p: self.disps[p] for p in range(self.nports)}   # the device's display_client
        self.plugged = {p: self.disps[p] for p in range(self.nports)}  # the port's display
        self.nubs = {}                  # port -> NubM
        self.pages = {}                 # RAM page -> PageM
        self.last_page = {}             # port -> PageM last connected
        self.enabled = {}               # port -> mask the guest writes
        self.run_state = 0
        self.irqs = 0
        self.r1014 = 0
        self.r1018 = 0
        self.cb = []                    # callbacks since the last events
        self.totals = {}
        self.serials = 2000
        self.efi = None                 # the EFI model while it is on
        self.efi_bound = None
        self.queued = []                # commands rung while paused
        self.paused = False
        self.surf = {}
        self.dies = False
        self.scribbles = 0

    # ---- helpers: predictions, callbacks, interrupts ----

    def predict(self, **fields):
        self.pred[len(self.s.ops) - 1] = fields

    def callback(self, d, kind, **kw):
        c = {"cb": kind, "display": d.index}
        c.update(kw)
        self.cb.append(c)

    def interrupt(self):
        if self.run_state == 0:
            self.irqs += 1

    def poke(self, port):
        self.r1014 |= 1 << port
        self.interrupt()

    def stamp(self):
        self.r1018 |= 1
        self.interrupt()

    def new_spec(self, port, serial):
        r = self.r
        n = 1 + r.below(3)
        modes = [r.choice(MODES) for _ in range(n)]
        name = "%s%d" % (r.choice(NAMES), port)
        return display(port, serial=serial, name=name, modes=modes)

    # ---- the page ----

    def page_of(self, port):
        n = self.nubs.get(port)
        return n.page if n else None

    def guest_write(self, pg, off, fmt, v):
        pg.put(off, fmt, v)
        self.s.mem(self.s.pa(pg.ram, off), struct.pack("<" + fmt, v))

    # ---- settings and mode list ----

    def update_settings(self, nub):
        pg, c = nub.page, nub.client
        ch = False

        def setf(off, fmt, v):
            nonlocal ch
            if pg.get(off, fmt) != v:
                pg.put(off, fmt, v)
                ch = True
        setf(O_SERIAL, "I", c.serial)
        setf(O_MMF, "f", f32(400.0))
        setf(O_MMF + 4, "f", f32(300.0))
        pg.put(O_MMR, "H", int(pg.get(O_MMF, "f") + 0.5) & 0xFFFF)
        pg.put(O_MMR + 2, "H", int(pg.get(O_MMF + 4, "f") + 0.5) & 0xFFFF)
        for i, v in enumerate(COLOR_SPACE):
            setf(O_CS + 4 * i, "f", f32(v))
        setf(O_ORIGIN, "H", 0xFFFF)
        setf(O_ORIGIN + 2, "H", 0xFFFF)
        setf(O_SCALE, "B", 0)
        for i in range(3):
            setf(O_NITS + 4 * i, "f", 0.0)
        setf(O_EPOCH, "Q", 0)
        setf(O_CONN, "i", 0)
        name = c.name.encode()[:13].ljust(14, b"\0")
        if cstr_differ(pg.b[O_NAME:O_NAME + 14], name):
            pg.b[O_NAME:O_NAME + 14] = name
            ch = True
        if pg.b[O_UUID] != 0:
            pg.b[O_UUID:O_UUID + 39] = bytes(39)
            ch = True
        setf(O_GLYPHMAX, "H", 256)
        setf(O_GLYPHMAX + 2, "H", 256)
        setf(O_HANDLERS, "I", 3)
        return ch

    def update_modes(self, nub):
        pg = nub.page
        if pg.get(O_COUNT, "H") > 128:
            raise Death("mode count over 128")
        modes = nub.modelist
        n = min(len(modes), 128)
        ch = False
        if pg.get(O_COUNT, "H") != n:
            pg.put(O_COUNT, "H", n)
            ch = True
        for i in range(n):
            w, h, hz = modes[i]
            o = O_MODES + 16 * i
            rr = int(math.floor(hz * 65536.0 + 0.5)) & M32
            for off, fmt, v in ((0, "H", w), (2, "H", h), (4, "I", rr)):
                if pg.get(o + off, fmt) != v:
                    pg.put(o + off, fmt, v)
                    ch = True
            if pg.b[o + 8] & 3:
                pg.b[o + 8] &= ~3 & 0xFF
                ch = True
        nub.modelist = None
        return ch

    # ---- the state machine ----

    def sm(self, nub):
        if nub.paused or nub.will_restore:
            return
        pg = nub.page
        bogus = False
        while True:
            state = pg.get(O_STATE, "I")
            if state == 0:
                if nub.client is not None and nub.modelist is not None:
                    self.update_settings(nub)
                    self.update_modes(nub)
                    pg.put(O_PROT, "Q", 0)
                    nub.cs_dirty = True
                    pg.put(O_EXPACK, "I", (pg.get(O_EXPACK, "I") + 1) & M32)
                    pg.b[O_ACKP] = 1
                    state, intr = 1, 4
                else:
                    state, intr = 0, 0
                nub.did_restore = False
                break
            if state == 1:
                if pg.b[O_ACKP]:
                    pg.put(O_STATE, "I", 1)
                    return
                pg.put(O_STATE, "I", 2)
                continue
            if state != 2:
                bogus = True
            intr = 0
            count = pg.get(O_COUNT, "H")
            restored = nub.did_restore and count != 0
            resend = bogus and count != 0
            if nub.client is not None and (nub.modelist is not None or restored or resend):
                sc = self.update_settings(nub)
                mc = self.update_modes(nub) if nub.modelist is not None else False
                if sc or mc or resend:
                    pg.put(O_EXPACK, "I", (pg.get(O_EXPACK, "I") + 1) & M32)
                    pg.b[O_ACKP] = 1
                    state, intr = 1, 12
            if nub.client is None or pg.get(O_COUNT, "H") == 0:
                pg.put(O_COUNT, "H", 0)
                state, intr = 0, 8
            nub.did_restore = False
            break
        if intr:
            pg.put(O_RAISED, "I", pg.get(O_RAISED, "I") | intr)
            if pg.get(O_ENABLED, "I") & intr:
                self.poke(nub.port)
        pg.put(O_STATE, "I", state)

    # ---- guest display and display plumbing ----

    def nub_init(self, port, pg):
        nub = NubM(port, pg)
        pg.put(O_PORT, "H", port)
        pg.put(O_1C, "I", 1024)
        wi = pg.b[O_CM_WRITE]
        for c in range(3):
            pg.put(O_CM + 48 * wi + 16 * c + 4 * c, "f", 1.0)
        pg.b[O_CM_DIRTY] = 1
        return nub

    def disp_set_nub(self, d, nub):
        d.frame = None
        d.nub = nub
        if nub is not None and d.modes:
            nub.modelist = list(d.modes)
            self.sm(nub)

    def nub_set_client(self, nub, d):
        nub.client = d
        self.sm(nub)

    def attach_page(self, port, pg, restoring=False):
        nub = self.nub_init(port, pg)
        if restoring:
            nub.will_restore = True
        else:
            self.root_word(port, self.s.pfn(pg.ram))
        self.nubs[port] = nub
        d = self.parked.pop(port, None)
        if d is not None:
            self.disp_set_nub(d, nub)
            self.nub_set_client(nub, d)
        return nub

    def detach_page(self, port):
        nub = self.nubs.pop(port)
        d = nub.client
        self.nub_set_client(nub, None)
        if d is not None:
            self.disp_set_nub(d, None)
            self.parked[port] = d
        if self.run_state == 0:
            self.root_word(port, 0)

    def root_word(self, port, v):
        self.root_words[port] = v

    def disp_set_modes(self, d, modes):
        if not modes:
            return
        d.modes = [tuple(m) for m in modes]
        if d.nub is not None:
            d.nub.modelist = list(d.modes)
            self.sm(d.nub)

    def signal_frame(self, d):
        if d.frame is None:
            return
        d.encoded = False
        d.gpc += 1
        self.callback(d, "newFrame")

    # ---- presents ----

    def matrix_dirty(self, pg):
        """GuestDisplay::take_dirty_color_matrix: the matrix passed, or
        None."""
        if not pg.b[O_CM_FORCE]:
            if not pg.b[O_CM_DIRTY]:
                return None
            pg.b[O_CM_READ] = pg.b[O_CM_WRITE]
            pg.b[O_CM_WRITE] = 0 if pg.b[O_CM_WRITE] else 1
            pg.b[O_CM_DIRTY] = 0
        ri = pg.b[O_CM_READ]
        m = [pg.get(O_CM + 48 * ri + 4 * k, "f") for k in range(12)]
        pg.b[O_CM_FORCE] = 0
        return m

    def present_frame(self, d, frame, w, h, fmt, gamma, matrix, nub, efi=False):
        """The display's present_frame and the completion."""
        if d.w != w or d.h != h or d.fmt != FMT[fmt] or d.prot != 0 or d.cs_dirty:
            d.cs_dirty = False
            self.callback(d, "modeChange", w=w, h=h, fmt=hex(FMT[fmt]))
            d.w, d.h, d.fmt, d.prot = w, h, FMT[fmt], 0
        if gamma is not None:
            if gamma.n == 1:
                raise Death("a one-entry gamma table")
            if gamma.bad:
                raise Refused("corrupt gammaTable")
            d.gamma = gamma
        d.frame = frame
        if matrix is not None:
            d.matrix = list(matrix)
        if efi:
            self.signal_frame(d)
        else:
            self.completion(nub)

    def completion(self, nub):
        pg = nub.page
        pg.put(O_RAISED, "I", pg.get(O_RAISED, "I") | 2)
        if pg.get(O_ENABLED, "I") & 2:
            self.poke(nub.port)

    def gamma_record(self, pg, gamma):
        if gamma is None:
            return
        pg.put(O_GAMMA, "Q", gamma.va)
        pg.put(O_GAMMA + 8, "Q", gamma.mapped)
        pg.put(O_GAMMA + 16, "I", gamma.n)
        pg.put(O_GAMMA + 20, "I", gamma.csum)

    def m_present_backing(self, port, b, gamma):
        """GuestDisplay::present_surface. Returns False when refused."""
        nub = self.nubs.get(port)
        if nub is None:
            return False
        pg = nub.page
        if b.host is None:
            b.host = HostSurf(b.w, b.h, b.fmt, b.raw)
        if b.fmt not in ("BGRA", "RGBA"):
            return False
        if gamma is not None and gamma.n > 1024:
            return False
        self.gamma_record(pg, gamma)
        d = nub.client
        if d is not None:
            m = self.matrix_dirty(pg)
            try:
                self.present_frame(d, ("host", b.host), b.w, b.h, b.fmt, gamma, m, nub)
            except Refused:
                return False
            if not nub.asleep:
                self.signal_frame(d)
        else:
            self.completion(nub)
        pg.put(O_PCOUNT, "Q", pg.get(O_PCOUNT, "Q") + 1)
        pg.put(O_PSURF, "I", b.ref)
        pg.put(O_FA4, "I", 0)
        return True

    def m_present_mapped(self, port, mid, gamma, nub=None):
        """GuestDisplay::present_mapped_surface. Returns False when
        refused."""
        nub = nub or self.nubs.get(port)
        if nub is None:
            return False
        sf = self.surf.get(mid)
        if sf is None or not sf.mapped:
            return False
        pg = nub.page
        if gamma is not None and gamma.n > 1024:
            return False
        self.gamma_record(pg, gamma)
        d = nub.client
        if d is not None:
            if nub.cs_dirty:
                nub.cs_dirty = False
                d.cs_dirty = True
            m = self.matrix_dirty(pg)
            try:
                self.present_frame(d, ("surface", mid), sf.w, sf.h, sf.fmt, gamma, m, nub)
            except Refused:
                return False
            if not nub.asleep:
                self.signal_frame(d)
        else:
            self.completion(nub)
        pg.put(O_PCOUNT, "Q", pg.get(O_PCOUNT, "Q") + 1)
        pg.put(O_PMAP, "Q", mid)
        pg.put(O_FA4, "I", 0)
        return True

    # ---- cursor (display::guest_display::cursor) ----

    def m_glyph(self, nub, va, length, stride, w, h, hx, hy, csum, words_at):
        pg = nub.page
        if w == 0 or w > 256 or h == 0 or h > 256:
            return False
        if (h - 1) * stride + 4 * w > length:
            return False
        d = nub.client
        if d is not None:
            rgba = bytearray()
            total = 0
            for y in range(h):
                for x in range(w):
                    p = words_at(va + y * stride + 4 * x)
                    total = (total + p) & M32
                    q = (p & 0xFF00FF00) | ((p & 0xFF) << 16) | ((p >> 16) & 0xFF)
                    rgba += struct.pack("<I", q)
            if csum and csum != total:
                return False
            c = {"alpha": True, "bitmapFormat": 0, "bpr": 4 * w, "bps": 8, "h": h, "hot": [hx, hy],
                 "planar": False, "sha256": hashlib.sha256(bytes(rgba)).hexdigest(), "spp": 4, "w": w}
            if len(rgba) <= 1024:
                c["hex"] = bytes(rgba).hex()
            self.callback(d, "cursorGlyph", **c)
        pg.put(O_GLYPH, "Q", va)
        pg.put(O_GLEN, "Q", length)
        pg.put(O_GSTRIDE, "Q", stride)
        pg.put(O_GF18, "I", 0)
        pg.put(O_GSUM, "I", csum)
        pg.put(O_GW, "H", w)
        pg.put(O_GW + 2, "H", h)
        pg.put(O_GW + 4, "H", hx)
        pg.put(O_GW + 6, "H", hy)
        return True

    def m_show(self, nub, show):
        if nub.client is not None:
            self.callback(nub.client, "cursorShow", show=bool(show))
        nub.page.b[O_HSHOW] = 1 if show else 0

    def m_sleep(self, nub, asleep):
        d = nub.client
        if d is not None and not asleep:
            self.signal_frame(d)
        nub.asleep = bool(asleep)
        nub.page.b[O_ASLEEP] = 1 if asleep else 0

    # ---- task 0's contents ----

    def word_at(self, va):
        """A u32 of task 0's memory the model wrote (glyph regions)."""
        for (lo, hi, start, step) in self.seq_regions:
            if lo <= va < hi:
                return (start + ((va - lo) // 4) * step) & M32
        raise AssertionError("glyph read outside the model's regions at 0x%x" % va)

    # ==== emission ====

    def build(self):
        s = self.s
        r = self.r
        self.root_words = {}
        s.w(abi.R_VERSION, self.version)
        s.r(abi.R_VERSION, "version")
        self.predict(val=hex(self.version))
        s.r(abi.R_PORT_COUNT, "port count")
        self.predict(val=hex(self.nports if self.version >= 245 else 1))
        s.r(abi.R_PORT_TYPES, "port types")
        self.predict(val="0x0")
        self.t0 = GPUTask(s, tid=0, pages=160)
        self.seq_regions = []
        self.lay_task0()
        if self.surfaces_on:
            self.lay_surfaces()
        self.bring_up(first=True)
        steps = self.steps
        if self.efi_on:
            self.efi_boot()
        try:
            for i in range(steps):
                self.step_no = i
                self.one_step()
            if self.death:
                self.final_death()
            else:
                self.final_check()
        except Death:
            pass
        self.s.doc = self.doc.strip() + "\n\nThis program: " + self.summary()
        return self.s

    def summary(self):
        kinds = {}
        for k in self.desc_steps:
            kinds[k] = kinds.get(k, 0) + 1
        return "%d port%s, version %d, %s; %s%s." % (
            self.nports, "s" if self.nports > 1 else "", self.version,
            "mapper surfaces" if self.surfaces_on else "no mapper surfaces",
            ", ".join("%s %d" % kv for kv in sorted(kinds.items())),
            "; ends in a death (%s)" % self.death if self.death else "")

    # ---- task 0: backings, glyph regions, gamma tables, ICC ----

    def lay_task0(self):
        r, t = self.r, self.t0
        self.backings = []
        nb = 2 + r.below(3)
        for i in range(nb):
            w, h = r.choice(SIZES_8)
            fmt = "RGBA" if r.chance(0.3) else "BGRA"
            raw = pixels(r, w, h, fmt)
            va = t.reserve(PAGE)
            t.write(va, raw)
            rec = struct.pack("<QIIB3x", PAGE, va // PAGE, FMT[fmt], 1) + struct.pack(
                "<IIII", 0, w, h, (w * 4) | 4 << 24)
            ref = t.ol.add(abi.T_IOSURFACE_BACKING, rec)
            self.backings.append(Backing(ref, va, w, h, fmt, raw))
        # A backing opcode 7 refuses: an IOSurface of format 'RGhA' (8 bytes).
        va = t.reserve(PAGE)
        rec = struct.pack("<QIIB3x", PAGE, va // PAGE, FMT["RGhA"], 1) + struct.pack("<IIII", 0, 4, 4, 32 | 8 << 24)
        self.bad_backing = Backing(t.ol.add(abi.T_IOSURFACE_BACKING, rec), va, 4, 4, "RGhA", bytes(128))
        # Glyph regions: two a port (a new glyph goes into the one the page
        # does not name) and one large region, contents start + k * step.
        self.glyph_slots = {}
        for p in range(self.nports):
            self.glyph_slots[p] = [self.seq_region(4 * PAGE) for _ in range(2)]
        self.big_glyph = self.seq_region(17 * PAGE)
        # Gamma tables.
        self.gammas = []
        for kind, n in (("ramp", 256), ("thresh", 2), ("ramp", 1024)):
            self.gammas.append(self.gamma_table(n, kind))
        for _ in range(3 + r.below(3)):
            self.gammas.append(self.gamma_table(r.choice((2, 3, 16, 64, 256, 1000, 1024)), "const"))
        self.gamma_big = self.gamma_table(1025, "const", write=False)
        g = self.gammas[3]
        self.gamma_bad = Gamma(g.n, g.kind, g.consts, g.va, (g.csum + 1) & M32, bad=True)
        # A mapped length the display cannot allocate: the gamma table's
        # update throws "Failed to allocate buffer"; the guest display's
        # checks pass it.
        self.gamma_huge = Gamma(g.n, g.kind, g.consts, g.va, g.csum, bad=True, mapped=1 << 62)
        self.gamma_one = self.gamma_table(1, "const")
        prof = icc_profile(r.choice((1.8, 2.2, 2.4)))
        self.icc = (t.blob(prof), len(prof))

    def seq_region(self, length):
        t, r = self.t0, self.r
        va = t.reserve(length)
        start, step = r.next() & M32, (r.next() | 1) & M32
        self.s.mem_seq(t.gpa(va), length // 4 if length <= PAGE else PAGE // 4, 4, bytes(4), [(0, 4, start, step)])
        pos = PAGE
        while pos < length:
            self.s.mem_seq(t.gpa(va + pos), PAGE // 4, 4, bytes(4), [(0, 4, (start + (pos // 4) * step) & M32, step)])
            pos += PAGE
        self.seq_regions.append((va, va + length, start, step))
        return va

    def gamma_table(self, n, kind, write=True):
        t, r = self.t0, self.r
        if kind == "const":
            consts = tuple(r.choice(GAMMA_C) for _ in range(3))
            words = []
            for c in range(3):
                words += [fbits(consts[c])] * n
        elif kind == "thresh":
            consts = None
            words = [fbits(0.0), fbits(1.0)] * 3
        else:
            consts = None
            words = [fbits(i / (n - 1)) for i in range(n)] * 3
        csum = sum(words) & M32
        va = t.reserve(12 * max(n, 1))
        if write:
            if kind == "const":
                for c in range(3):
                    self.s.mem_seq(t.gpa(va + 4 * n * c), n, 4, bytes(4), [(0, 4, fbits(consts[c]), 0)])
            else:
                t.write(va, struct.pack("<%dI" % len(words), *words))
        return Gamma(n, kind, consts, va, csum)

    def lay_surfaces(self):
        r, s = self.r, self.s
        self.iosfc = IOSFC(s, v2=True)
        self.surf = {}
        n = 2 + r.below(3)
        for mid in range(1, n + 1):
            if r.chance(0.25):
                w, h = r.choice(SIZES_H)
                fmt = "RGhA"
            else:
                w, h = r.choice(SIZES_8)
                fmt = "RGBA" if r.chance(0.3) else "BGRA"
            root, data = self.iosfc.surface_pages(1)
            raw = pixels(r, w, h, fmt)
            s.mem(s.pa(data), raw)
            self.surf[mid] = Surface(mid, data, root, w, h, fmt, raw)

    # ---- bring-up and teardown ----

    def bring_up(self, first=False):
        """Root page, root FIFO, task 0 and its object list; the first time
        also the IOSurface host device and every surface mapped."""
        s = self.s
        if first:
            s.bring_up(version=None)
        else:
            s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
            s.fifo.written = 0
            s.fifo.enable()
        self.submit(abi.OP_NOP, b"", "NOP")
        self.t0.define(label="define task 0")
        self.stamp()
        self.t0.install(label="task 0's object list")
        self.stamp()
        for b in self.backings:
            b.host = None
        if first and self.surfaces_on:
            self.iosfc.start()
            for mid in sorted(self.surf):
                self.map_surface(mid)

    def map_surface(self, mid):
        sf = self.surf[mid]
        d = self.iosfc
        d.put_desc(mid, desc_v2(sf.root, sf.w, sf.h, fmt=sf.fmt.encode(), bpe=8 if sf.fmt == "RGhA" else 4))
        idx = d.submit((CMD_MAP, mid))
        d.status(idx, "map surface %d" % mid)
        self.predict(hex=struct.pack("<IIII", CMD_MAP, mid, SUCCESS, 0).hex())
        sf.mapped = True

    def submit(self, opcode, payload, label):
        """One command on the root FIFO, waited for by its stamp, which
        raises an interrupt."""
        self.s.submit(opcode, payload, label=label)
        self.stamp()

    # ---- the EFI framebuffer ----

    def efi_boot(self):
        """EFI at boot or after a reset: port 0's parked display bound, a
        frame, trace fires, maybe a mode change, then off."""
        s, r = self.s, self.r
        d = self.parked.get(0)
        if d is None or 0 in self.nubs:
            return
        self.desc_steps.append("efi")
        s.mark("EFI on port 0's display %d" % d.index)
        if not hasattr(self, "efi_fb"):
            self.efi_fb = EFI(s)
        w, h = r.choice(EFI_SIZES)
        raw = self.efi_frame(w, h)
        self.efi = dict(w=w, h=h, raw=raw)
        self.efi_fb.setup(w, h, EFI_STRIDE)
        self.efi_bound = d
        self.efi_present(d)
        self.events("EFI up")
        self.efi_regs(d, "EFI up")
        self.encode(d, "EFI frame")
        for _ in range(r.below(3)):
            raw = self.efi_frame(w, h)
            self.efi["raw"] = raw
            s.trace_fire(self.efi_fb.gpa(), PAGE)
            self.efi_present(d)
            self.events("EFI frame")
            self.encode(d, "EFI frame")
        if r.chance(0.5):
            w2, h2 = r.choice(EFI_SIZES)
            raw = self.efi_frame(w2, h2)
            self.efi.update(w=w2, h=h2, raw=raw)
            s.w(abi.R_EFI_MODE, w2 | h2 << 16)
            self.efi_present(d)
            self.events("EFI mode %dx%d" % (w2, h2))
            self.encode(d, "EFI frame")
        sel = r.below(len(d.modes) + 1) if d.modes else 0
        s.w(abi.R_EFI_MODE_SELECT, sel)
        s.r(abi.R_EFI_MODE_SELECTED, "EFI mode %d" % sel)
        m = d.modes[sel] if d.modes and sel < len(d.modes) else None
        self.predict(val=hex(m[0] | m[1] << 16) if m else hex(M32))
        s.w(abi.R_EFI_ENABLE, 0)
        self.efi_bound = None
        self.efi = None
        d.frame = ("stale",)
        self.events("EFI off")
        for reg in (abi.R_EFI_ENABLE, abi.R_EFI_START, abi.R_EFI_LENGTH, abi.R_EFI_MODE):
            s.r(reg, "EFI off")
            self.predict(val="0x0")

    def efi_frame(self, w, h):
        r = self.r
        rows = []
        raw = b""
        for y in range(h):
            row = b"".join(bytes((r.choice(PAL), r.choice(PAL), r.choice(PAL), 0xFF)) for _ in range(w))
            raw += row
            rows.append(row + bytes(EFI_STRIDE - len(row)))
        self.s.mem(self.efi_fb.gpa(), b"".join(rows))
        return raw

    def efi_present(self, d):
        e = self.efi
        self.present_frame(d, ("efi",), e["w"], e["h"], "BGRA", None, None, None, efi=True)
        self.wait_callbacks()

    def efi_regs(self, d, label):
        s = self.s
        s.r(abi.R_EFI_ENABLE, label)
        self.predict(val="0x1")
        s.r(abi.R_EFI_MODE_COUNT, label)
        self.predict(val=hex(len(d.modes) if d.modes else 0))
        s.r(abi.R_EFI_MODE, label)
        self.predict(val=hex(self.efi["w"] | self.efi["h"] << 16))

    # ---- observation ----

    def wait_callbacks(self):
        """Waits for each kind the model added since the last wait, by its
        running total."""
        pend = getattr(self, "_pend", 0)
        new = self.cb[pend:]
        self._pend = len(self.cb)
        kinds = []
        for c in new:
            if c["cb"] not in kinds:
                kinds.append(c["cb"])
        for c in new:
            self.totals[c["cb"]] = self.totals.get(c["cb"], 0) + 1
        for k in kinds:
            self.s.wait(cb=k, cb_count=self.totals[k], label="%s %d (model)" % (k, self.totals[k]))
            self.predict(ok=True)

    def events(self, label):
        self.wait_callbacks()
        self.s.events(label)
        p = {"irqs": self.irqs}
        if self.cb:
            p["callbacks"] = list(self.cb)
        self.predict(**p)
        self.cb = []
        self._pend = 0
        self.irqs = 0

    def regs(self, label):
        s = self.s
        s.r(abi.R_IRQ_DISPLAY, label)
        self.predict(val=hex(self.r1014))
        self.r1014 = 0
        s.r(abi.R_IRQ_STAMP, label)
        self.predict(val=hex(self.r1018))
        self.r1018 = 0

    def check_pages(self, label, ports=None):
        s = self.s
        for port in sorted(self.nubs) if ports is None else ports:
            pg = self.nubs[port].page if port in self.nubs else self.last_page.get(port)
            if pg is None:
                continue
            self.cmp_page(pg, "%s: port %d's page" % (label, port))

    def cmp_page(self, pg, label):
        s = self.s
        if pg.mirror is None:
            pg.mirror = s.alloc_page()
        i = 0
        b, m = pg.b, pg.mirrored
        while i < PAGE:
            if b[i] == m[i]:
                i += 1
                continue
            j = i
            last = i
            while j < PAGE and j - last < 16:
                if b[j] != m[j]:
                    last = j
                j += 1
            s.mem(s.pa(pg.mirror, i), bytes(b[i:last + 1]))
            m[i:last + 1] = b[i:last + 1]
            i = last + 1
        s.cmp(s.pa(pg.ram), s.pa(pg.mirror), PAGE, label=label + " against the model")
        self.predict(equal=True)

    def root_dump(self, label):
        s = self.s
        s.dump(s.pa(s.root_page, abi.ROOT_DISPLAY_WORD), 4 * self.nports, label)
        self.predict(hex=b"".join(struct.pack("<I", self.root_words.get(p, 0)) for p in range(self.nports)).hex())

    def frame_bytes(self, d):
        f = d.frame
        if f is None:
            return None
        if f[0] == "host":
            hs = f[1]
            return hs.w, hs.h, hs.fmt, hs.raw
        if f[0] == "surface":
            sf = self.surf[f[1]]
            return sf.w, sf.h, sf.fmt, sf.raw
        if f[0] == "efi":
            e = self.efi
            return e["w"], e["h"], "BGRA", e["raw"]
        return "stale"

    def encode(self, d, label):
        fb = self.frame_bytes(d)
        if fb == "stale":
            return
        if fb is None:
            w, h = 4, 4
            img = bytes((0, 0, 0, 0xFF)) * 16
        else:
            w, h, fmt, raw = fb
            img = render(d, w, h, fmt, raw)
        self.s.encode(d.index, w, h, label="%s (display %d%s)" % (label, d.index, "" if img else ", not predicted"))
        if fb is not None and not d.encoded:
            d.hpc += 1
            d.encoded = True
        if img is None:
            self.predict(ok=True, status=4)
        else:
            p = dict(ok=True, status=4, sha256=hashlib.sha256(img).hexdigest())
            if len(img) <= 4096:
                p["hex"] = img.hex()
            self.predict(**p)

    def counts(self, d):
        self.s.api("present_counts", display=d.index, label="display %d" % d.index)
        self.predict(guest=d.gpc, host=d.hpc)

    def checkpoint(self, label, pages=True):
        self.events(label)
        self.regs(label)
        if pages:
            self.check_pages(label)

    # ==== the steps ====

    def live_displays(self):
        return [d for d in self.disps if d.alive]

    def one_step(self):
        r = self.r
        acts = self.legal()
        name, args = r.weighted([(a, w) for a, w in acts])
        self.desc_steps.append(name)
        getattr(self, "a_" + name)(*args)

    def legal(self):
        """(("action", args), weight) for every legal action now."""
        r = self.r
        out = []
        port = r.below(self.nports)
        nub = self.nubs.get(port)
        d = self.plugged.get(port)
        if self.paused:
            out.append((("unpause", ()), 6))
            out.append((("host_while_paused", (port,)), 4))
            # One command a port: two presents on one display after the
            # unpause would call new_frame at once, and it coalesces.
            queued = any(q[1] == port for q in self.queued)
            out.append((("ring_while_paused", (port,)), 6 if nub and not queued else 0))
            return [a for a in out if a[1]]
        if nub is None:
            out.append((("connect", (port,)), 10))
            out.append((("refused_present", (port,)), 1))
            out.append((("doorbell_no_page", (port,)), 1))
        else:
            pg = nub.page
            pend = bool(pg.b[O_ACKP])
            out.append((("ack", (port,)), 12 if pend else 0))
            out.append((("disconnect", (port,)), 2))
            out.append((("reconnect", (port,)), 1))
            out.append((("present_backing", (port,)), 10))
            out.append((("present6", (port,)), 2))
            if self.surfaces_on:
                out.append((("present_mapped", (port,)), 10))
            out.append((("glyph", (port,)), 4))
            out.append((("show", (port,)), 3))
            out.append((("move", (port,)), 4))
            out.append((("matrix", (port,)), 3))
            out.append((("icc", (port,)), 2))
            out.append((("protection", (port,)), 1))
            out.append((("sleep", (port,)), 2 if self.version >= 21 else 0))
            out.append((("compositor", (port,)), 1 if self.version >= 108 else 0))
            out.append((("zero_modes", (port,)), 1 if not pend else 0))
            out.append((("bogus_state", (port,)), 1 if not pend else 0))
            out.append((("clear_raised", (port,)), 2))
            out.append((("scribble", (port,)), 5))
            out.append((("refused_present", (port,)), 1))
        if d is not None:
            out.append((("unplug", (port,)), 2))
            out.append((("set_modes", (port,)), 4))
            out.append((("encode", (port,)), 2))
        else:
            out.append((("plug", (port,)), 5))
        out.append((("rewrite_backing", ()), 2))
        out.append((("delete_backing", ()), 1))
        if self.surfaces_on:
            out.append((("rewrite_surface", ()), 2))
            out.append((("remap_surface", ()), 1))
        out.append((("pause", ()), 2))
        out.append((("suspend", ()), 2))
        out.append((("reset", ("reset",)), 1))
        out.append((("reset", ("stop",)), 1))
        return [a for a in out if a[1]]

    # ---- per-step checkpoint ----

    def settle(self, port, label):
        """A wait on the port's page at the model's expectedAck and state,
        then the checkpoint."""
        nub = self.nubs.get(port)
        if nub is not None:
            pg = nub.page
            self.s.wait(gpa=self.s.pa(pg.ram, O_EXPACK), eq=pg.get(O_EXPACK, "I"), label="%s: expectedAck (model)" % label)
            self.predict(ok=True)
            self.s.wait(gpa=self.s.pa(pg.ram, O_STATE), eq=pg.get(O_STATE, "I"), label="%s: state (model)" % label)
            self.predict(ok=True)
        self.checkpoint(label)

    def mark(self, text):
        self.s.mark("step %d: %s" % (self.step_no, text))
        return "step %d: %s" % (self.step_no, text)

    # ---- connect, disconnect, ack ----

    def a_connect(self, port, reuse=None):
        s, r = self.s, self.r
        if reuse is None and port in self.last_page and self.last_page[port] not in [
                n.page for n in self.nubs.values()] and r.chance(0.3):
            reuse = self.last_page[port]
        if reuse is not None:
            pg = reuse
        else:
            pg = PageM(s.alloc_page(), None)
            mask = r.choice((0xF, 0xF, 0xE, 0xC, 0x6, 0x2, 0x0, 0xA))
            self.guest_write(pg, O_ENABLED, "I", mask)
        lab = self.mark("connect port %d on %s page (enabled 0x%x)" % (
            port, "its old" if reuse else "a fresh", pg.get(O_ENABLED, "I")))
        self.submit(abi.OP_DISPLAY_SET_SHARED_STATE_PAGE, p_set_shared(port, s.pfn(pg.ram)), lab)
        self.attach_page(port, pg)
        self.last_page[port] = pg
        self.settle(port, lab)
        self.root_dump(lab)

    def a_disconnect(self, port):
        lab = self.mark("disconnect port %d" % port)
        pg = self.nubs[port].page
        self.submit(abi.OP_DISPLAY_SET_SHARED_STATE_PAGE, p_set_shared(port, 0), lab)
        self.detach_page(port)
        self.checkpoint(lab, pages=False)
        self.cmp_page(pg, lab + ": the page left")
        self.root_dump(lab)
        d = self.plugged.get(port)
        if d is not None and self.r.chance(0.5):
            self.encode(d, lab)

    def a_reconnect(self, port):
        """The guest resets the port: disconnect, then connect on the same
        page (its expectedAck goes on) or on a fresh one."""
        pg = self.nubs[port].page
        self.a_disconnect(port)
        self.a_connect(port, reuse=pg if self.r.chance(0.6) else None)

    def a_ack(self, port):
        pg = self.nubs[port].page
        v = pg.get(O_EXPACK, "I")
        lab = self.mark("ack %d on port %d" % (v, port))
        self.submit(abi.OP_DISPLAY_ACK, p_ack(port, v), lab)
        nub = self.nubs[port]
        pg.put(O_LASTACK, "I", v)
        pg.b[O_ACKP] = 0
        self.sm(nub)
        self.settle(port, lab)

    # ---- host plug, unplug, modes ----

    def a_plug(self, port):
        s, r = self.s, self.r
        self.serials += 1
        spec = self.new_spec(port, self.serials)
        if r.chance(0.1):
            spec["modes"] = [[16 + 8 * (i % 40), 8 + 4 * (i // 40), 60] for i in range(130)]
        lab = self.mark("plug display %d (serial %d, %s) into port %d" % (len(self.disps), spec["serial"], spec["name"], port))
        s.api("new_display", spec=spec)
        d = DispM(len(self.disps), port, spec["serial"], spec["name"], None)
        self.predict(display=d.index, ok=True, modes_valid=len(spec["modes"]))
        self.disps.append(d)
        self.plugged[port] = d
        nub = self.nubs.get(port)
        if nub is not None:
            self.nub_set_client(nub, d)
            d.nub = nub
        else:
            self.parked[port] = d
        self.disp_set_modes(d, [tuple(m) for m in spec["modes"]])
        self.settle(port, lab)

    def a_unplug(self, port):
        d = self.plugged.pop(port)
        lab = self.mark("unplug display %d from port %d" % (d.index, port))
        self.s.api("release_display", display=d.index, label=lab)
        d.alive = False
        nub = self.nubs.get(port)
        if nub is not None:
            self.nub_set_client(nub, None)
        else:
            self.parked.pop(port, None)
        if port == 0:
            self.efi_bound = None
        self.settle(port, lab)

    def a_set_modes(self, port):
        r = self.r
        d = self.plugged[port]
        if r.chance(0.2) and d.modes:
            modes = list(d.modes)
        elif r.chance(0.08):
            modes = [(8 + 8 * (i % 50), 4 + 2 * (i // 50), 30) for i in range(131)]
        else:
            modes = [r.choice(MODES) for _ in range(1 + r.below(4))]
        lab = self.mark("set_modes on display %d: %d mode%s" % (d.index, len(modes), "s" if len(modes) > 1 else ""))
        self.s.api("set_modes", display=d.index, modes=[list(m) for m in modes], label=lab)
        self.predict(modes_valid=len(modes))
        self.disp_set_modes(d, modes)
        self.settle(port, lab)

    # ---- presents ----

    def pick_gamma(self):
        """None, a table, or now and then one whose checksum is wrong: the
        display throws after mode_change and before the frame."""
        r = self.r
        if r.chance(0.55):
            return None
        if r.chance(0.08):
            return self.gamma_bad if r.chance(0.5) else self.gamma_huge
        return r.choice(self.gammas)

    def after_present(self, port, lab, encode=True):
        self.settle(port, lab)
        d = self.plugged.get(port)
        nub = self.nubs.get(port)
        if d is not None and nub is not None and nub.client is d and encode:
            self.encode(d, lab)
            if self.r.chance(0.3):
                self.counts(d)

    def a_present_backing(self, port):
        r = self.r
        b = r.choice(self.backings)
        g = self.pick_gamma()
        lab = self.mark("present backing %d (%dx%d %s) on port %d%s" % (
            b.ref, b.w, b.h, b.fmt, port, ", gamma %s %d%s" % (g.kind, g.n, (" with a wrong checksum" if g.mapped == 12 * g.n else " mapped 2^62") if g.bad else "") if g else ""))
        self.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(port, b.ref, *(g.payload() if g else ())), lab)
        self.m_present_backing(port, b, g)
        self.after_present(port, lab)

    def a_present6(self, port):
        b = self.r.choice(self.backings)
        lab = self.mark("present backing %d with opcode 6 on port %d" % (b.ref, port))
        self.submit(abi.OP_DISPLAY_TRANSACTION2, p_present2(port, b.ref), lab)
        self.m_present_backing(port, b, None)
        self.after_present(port, lab)

    def a_present_mapped(self, port):
        r = self.r
        mapped = [m for m in sorted(self.surf) if self.surf[m].mapped]
        if not mapped:
            return self.a_present_backing(port)
        mid = r.choice(mapped)
        sf = self.surf[mid]
        g = self.pick_gamma()
        lab = self.mark("swap to mapping %d (%dx%d %s) on port %d%s" % (
            mid, sf.w, sf.h, sf.fmt, port, ", gamma %s %d%s" % (g.kind, g.n, (" with a wrong checksum" if g.mapped == 12 * g.n else " mapped 2^62") if g.bad else "") if g else ""))
        self.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(port, mid, *(g.payload() if g else ())), lab)
        self.m_present_mapped(port, mid, g)
        self.after_present(port, lab)

    def a_refused_present(self, port):
        """A present the host refuses and signals: no page, an unmapped
        mapping, a gamma table over 1024 entries, or a backing whose format
        opcode 7 does not present."""
        r = self.r
        nub = self.nubs.get(port)
        choices = ["bad_format", "gamma_big"] if nub else ["no_page"]
        if self.surfaces_on:
            choices.append("unmapped")
        why = r.choice(choices)
        lab = self.mark("a refused present on port %d (%s)" % (port, why))
        if why == "unmapped" or (why == "no_page" and self.surfaces_on and r.chance(0.5)):
            mid = 99 if why == "unmapped" else r.choice(sorted(self.surf))
            self.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(port, mid), lab)
        elif why == "gamma_big":
            b = r.choice(self.backings)
            self.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(port, b.ref, *self.gamma_big.payload()), lab)
            if b.host is None:
                b.host = HostSurf(b.w, b.h, b.fmt, b.raw)
        elif why == "bad_format":
            self.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(port, self.bad_backing.ref), lab)
        else:
            self.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(port, self.backings[0].ref), lab)
        self.settle(port, lab)

    # ---- cursor ----

    def a_glyph(self, port):
        r, s = self.r, self.s
        nub = self.nubs[port]
        pg = nub.page
        big = r.chance(0.12)
        if big:
            w, h = r.choice(((256, 256), (255, 256), (1, 256), (256, 1), (200, 100), (129, 64), (256, 255)))
            base = self.big_glyph
            room = 17 * PAGE
        else:
            w = r.choice((1, 2, 3, 4, 7, 8, 16, 32, 64))
            h = r.choice((1, 2, 4, 5, 16, 32, 64))
            slots = self.glyph_slots[port]
            base = slots[0] if pg.get(O_GLYPH, "Q") != slots[0] else slots[1]
            room = 4 * PAGE
        pad = r.choice((0, 0, 4, 16, 60))
        stride = 4 * w + pad
        while (h - 1) * stride + 4 * w > room:
            stride = 4 * w
            if (h - 1) * stride + 4 * w > room:
                h //= 2
        need = (h - 1) * stride + 4 * w
        length = need + r.choice((0, 0, 4, 64))
        length = min(length, room)
        hx, hy = r.below(300), r.below(300)
        total = 0
        for y in range(h):
            for x in range(w):
                total = (total + self.word_at(base + y * stride + 4 * x)) & M32
        csum = total if r.chance(0.5) else 0
        lab = self.mark("glyph %dx%d stride %d on port %d (hot %d,%d, %s)" % (
            w, h, stride, port, hx, hy, "checksum" if csum else "no checksum"))
        self.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(port, base, length, stride, w, h, hx, hy, csum), lab)
        ok = self.m_glyph(nub, base, length, stride, w, h, hx, hy, csum, self.word_at)
        assert ok
        self.settle(port, lab)

    def a_show(self, port):
        show = self.r.below(2)
        lab = self.mark("cursor show %d on port %d" % (show, port))
        self.submit(abi.OP_DISPLAY_CURSOR_SHOW, p_show(port, show), lab)
        self.m_show(self.nubs[port], show)
        self.settle(port, lab)

    def a_move(self, port):
        r, s = self.r, self.s
        nub = self.nubs[port]
        pg = nub.page
        x, y = r.below(0x10000), r.below(0x10000)
        gshow = r.below(2) if r.chance(0.5) else None
        lab = self.mark("move the cursor on port %d to %d,%d%s" % (
            port, x, y, "" if gshow is None else ", guest show %d" % gshow))
        self.guest_write(pg, O_CURSOR, "I", x | y << 16)
        if gshow is not None:
            self.guest_write(pg, O_GSHOW, "B", gshow)
        if r.chance(0.2):
            self.guest_write(pg, 0x51, "B", 1)
        s.w(abi.R_DISPLAY_DOORBELL, port)
        d = nub.client
        if d is not None:
            show = pg.b[O_GSHOW] != 0
            if show != (pg.b[O_HSHOW] != 0):
                pg.b[O_HSHOW] = 1 if show else 0
                self.callback(d, "cursorShow", show=show)
            self.callback(d, "cursorMove")
        self.settle(port, lab)
        pd = self.plugged.get(port)
        if pd is not None and r.chance(0.6):
            s.api("cursorPosition", display=pd.index, label=lab)
            if pd.nub is not None:
                self.predict(x=x, y=y)
            else:
                self.predict(x=0xFFFF, y=0xFFFF)

    def a_doorbell_no_page(self, port):
        lab = self.mark("doorbell on port %d, which has no page" % port)
        self.s.w(abi.R_DISPLAY_DOORBELL, port)
        self.s.quiesce()
        self.checkpoint(lab)

    # ---- properties, ICC, sleep, compositor ----

    def a_matrix(self, port):
        r = self.r
        kind = r.below(4)
        if kind == 0:
            m = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0]
        elif kind == 1:
            perm = r.sample(range(3), 3)
            m = []
            for c in range(3):
                row = [0.0] * 4
                row[perm[c]] = 1.0
                m += row
        else:
            m = []
            for c in range(3):
                m += [r.choice(COEF), r.choice(COEF), r.choice(COEF), r.choice((0.0, 0.0, 0.25))]
        words = [fbits(v) for v in m]
        unknown = r.chance(0.15)
        props = ([(7, [1, 2])] if unknown else []) + [(0, words)]
        gated = self.version < 40
        lab = self.mark("colour matrix on port %d%s%s" % (port, " after an unknown property" if unknown else "",
                                                          " (below version 40)" if gated else ""))
        self.submit(abi.OP_DISPLAY_SET_PROPERTIES, p_props(port, props), lab)
        if not gated and not unknown:
            pg = self.nubs[port].page
            wi = pg.b[O_CM_WRITE]
            for k in range(12):
                pg.put(O_CM + 48 * wi + 4 * k, "f", f32(m[k]))
            pg.b[O_CM_DIRTY] = 1
        self.settle(port, lab)

    def a_icc(self, port):
        r = self.r
        va, n = self.icc
        kind = r.below(3)
        gated = self.version < 208
        if kind == 0:
            length, mapped = n, n
        elif kind == 1:
            length, mapped = 0, 0
        else:
            length, mapped = n, n - 1   # refused: length > mappedLength
        lab = self.mark("ICC profile on port %d (length %d, mapped %d)%s" % (port, length, mapped,
                                                                              " below version 208" if gated else ""))
        self.submit(abi.OP_DISPLAY_SET_GUEST_ICC_PROFILE, p_icc(port, length, va, mapped), lab)
        if not gated and length <= mapped:
            nub = self.nubs[port]
            nub.page.put(O_ICC, "Q", va)
            nub.page.put(O_ICC + 8, "I", length)
            nub.cs_dirty = True
        self.settle(port, lab)

    def a_protection(self, port):
        v = self.r.choice((1, 2, 3, 0x10, 0))
        lab = self.mark("the guest writes protection 0x%x at +0x68 on port %d" % (v, port))
        self.guest_write(self.nubs[port].page, O_PROT, "Q", v)
        self.settle(port, lab)

    def a_sleep(self, port):
        nub = self.nubs[port]
        asleep = 0 if nub.asleep else 1
        lab = self.mark("%s port %d" % ("sleep" if asleep else "wake", port))
        self.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(port, asleep), lab)
        self.m_sleep(nub, asleep)
        self.settle(port, lab)
        d = nub.client
        if d is not None and not asleep and self.r.chance(0.5):
            self.encode(d, lab)
            self.counts(d)

    def a_compositor(self, port):
        r = self.r
        lab = self.mark("compositor parameters on port %d" % port)
        self.submit(abi.OP_DISPLAY_COMPOSITOR_PARAMETERS, p_compositor(port, r.below(100), r.below(100),
                                                                      r.below(1000), r.below(4)), lab)
        self.settle(port, lab)

    def a_zero_modes(self, port):
        lab = self.mark("the guest zeroes port %d's mode count" % port)
        self.guest_write(self.nubs[port].page, O_COUNT, "H", 0)
        self.settle(port, lab)

    def a_bogus_state(self, port):
        v = self.r.choice((7, 3, 0xFFFFFFFF))
        lab = self.mark("the guest writes state 0x%x on port %d" % (v, port))
        self.guest_write(self.nubs[port].page, O_STATE, "I", v)
        self.settle(port, lab)

    def a_scribble(self, port):
        """The guest overwrites a field the host owns: the next state
        machine run that updates settings (or the mode list, for a mode)
        puts it back and counts a change."""
        r = self.r
        pg = self.nubs[port].page
        fields = ([(O_SCALE, "B"), (O_NITS, "f"), (O_NITS + 4, "f"), (O_NITS + 8, "f"), (O_EPOCH, "Q"),
                   (O_CONN, "i"), ("uuid", None), ("mode", 1), ("mode", 2), ("mode", 0), (O_SERIAL, "I"),
                   (O_MMF, "f"), (O_MMF + 4, "f"), (O_ORIGIN, "H"), (O_ORIGIN + 2, "H"), (O_GLYPHMAX, "H"),
                   (O_GLYPHMAX + 2, "H"), (O_HANDLERS, "I"), (O_CS + 4 * r.below(8), "f"), ("name", None)])
        # The fields in turn, from a place the seed picks, so that the
        # programs together overwrite each of them.
        off, fmt = fields[(self.seed + self.scribbles) % len(fields)]
        self.scribbles += 1
        if off == "name":
            n = pg.b[O_NAME:O_NAME + 14].find(0)
            if n >= 0 and n < 13 and r.chance(0.5):
                what = "junk after the name's NUL"
                data = bytes([0x5A] * (13 - n))
                at = O_NAME + n + 1
            else:
                what = "the name"
                data = b"guest!"
                at = O_NAME
        elif off == "uuid":
            what, data, at = "the display UUID", b"GUEST-UUID", O_UUID
        elif off == "mode":
            i = r.below(max(pg.get(O_COUNT, "H"), 1))
            if fmt:
                what, data, at = "mode %d's flags (%d)" % (i, fmt), bytes([fmt]), O_MODES + 16 * i + 8
            else:
                what, data, at = "mode %d's width" % i, struct.pack("<H", 7777), O_MODES + 16 * i
        else:
            v = {"f": 123.25, "B": 3, "H": 0x1234, "I": 0x51, "Q": 0x77, "i": -2}[fmt]
            what, data, at = "+0x%x" % off, struct.pack("<" + fmt, v), off
        nub = self.nubs[port]
        d = nub.client
        again = d is not None and d.modes and r.chance(0.8)
        lab = self.mark("the guest overwrites %s on port %d%s" % (
            what, port, "; the host sets the same mode list again" if again else ""))
        pg.b[at:at + len(data)] = data
        self.s.mem(self.s.pa(pg.ram, at), data)
        if again:
            self.s.api("set_modes", display=d.index, modes=[list(m) for m in d.modes], label=lab)
            self.predict(modes_valid=len(d.modes))
            self.disp_set_modes(d, list(d.modes))
        self.settle(port, lab)

    def a_clear_raised(self, port):
        lab = self.mark("the guest clears port %d's raisedInterrupts" % port)
        self.guest_write(self.nubs[port].page, O_RAISED, "I", 0)
        self.settle(port, lab)

    # ---- guest memory under frames ----

    def a_rewrite_backing(self):
        r = self.r
        b = r.choice(self.backings)
        lab = self.mark("the guest rewrites backing %d's pixels" % b.ref)
        b.raw = pixels(r, b.w, b.h, b.fmt)
        self.t0.write(b.va, b.raw)
        self.encode_all(lab)

    def a_delete_backing(self):
        b = self.r.choice(self.backings)
        lab = self.mark("delete backing %d (54)" % b.ref)
        self.submit(abi.OP_DELETE_IOSURFACE_BACKING2, struct.pack("<II", b.ref, 0), lab)
        b.host = None
        self.checkpoint(lab)
        self.encode_all(lab)

    def a_rewrite_surface(self):
        r, s = self.r, self.s
        mid = r.choice(sorted(self.surf))
        sf = self.surf[mid]
        lab = self.mark("the guest rewrites surface %d" % mid)
        sf.raw = pixels(r, sf.w, sf.h, sf.fmt)
        s.mem(s.pa(sf.page), sf.raw)
        self.encode_all(lab)

    def a_remap_surface(self):
        r = self.r
        shown = set(d.frame[1] for d in self.disps if d.alive and d.frame and d.frame[0] == "surface")
        free = [m for m in sorted(self.surf) if m not in shown]
        if not free:
            return self.a_rewrite_surface()
        mid = r.choice(free)
        sf = self.surf[mid]
        lab = self.mark("%s surface %d" % ("map" if not sf.mapped else "unmap", mid))
        if sf.mapped:
            idx = self.iosfc.submit((CMD_UNMAP, mid))
            self.iosfc.status(idx, lab)
            self.predict(hex=struct.pack("<IIII", CMD_UNMAP, mid, SUCCESS, 0).hex())
            sf.mapped = False
        else:
            self.map_surface(mid)
        self.checkpoint(lab, pages=False)

    def encode_all(self, lab):
        for d in self.live_displays():
            if d.frame is not None and d.frame[0] != "stale":
                self.encode(d, lab)

    def a_encode(self, port):
        d = self.plugged[port]
        lab = self.mark("encode display %d" % d.index)
        self.encode(d, lab)
        self.counts(d)

    # ---- lifecycle ----

    def a_pause(self):
        lab = self.mark("pause")
        self.s.api("pause", label=lab)
        for port in sorted(self.nubs):
            self.nubs[port].paused = True
        self.run_state = 5
        self.paused = True
        self.checkpoint(lab, pages=False)

    def a_host_while_paused(self, port):
        r = self.r
        d = self.plugged.get(port)
        if d is None:
            self.a_plug(port)
        elif r.chance(0.5):
            self.a_set_modes(port)
        else:
            self.a_unplug(port)

    def a_ring_while_paused(self, port):
        """A command rung while paused: it runs after the unpause."""
        r, s = self.r, self.s
        nub = self.nubs[port]
        pg = nub.page
        if pg.b[O_ACKP] and r.chance(0.6):
            v = pg.get(O_EXPACK, "I")
            st = s.fifo.cmd(abi.OP_DISPLAY_ACK, p_ack(port, v))
            self.queued.append(("ack", port, v, st))
            what = "ack %d" % v
        else:
            b = r.choice(self.backings)
            st = s.fifo.cmd(abi.OP_DISPLAY_TRANSACTION3, p_present3(port, b.ref))
            self.queued.append(("present", port, b, st))
            what = "present backing %d" % b.ref
        lab = self.mark("rung while paused on port %d: %s" % (port, what))
        s.fifo.doorbell()
        s.fifo.wait_stamp(st, timeout_ms=300, label=lab + ": not run [held]")
        self.predict(ok=False)

    def a_unpause(self):
        s = self.s
        lab = self.mark("unpause")
        s.api("unpause", label=lab)
        self.run_state = 0
        self.paused = False
        for port in sorted(self.nubs):
            self.nubs[port].paused = False
            self.sm(self.nubs[port])
        self.interrupt()
        last = None
        for q in self.queued:
            if q[0] == "ack":
                _, port, v, st = q
                pg = self.nubs[port].page
                pg.put(O_LASTACK, "I", v)
                pg.b[O_ACKP] = 0
                self.sm(self.nubs[port])
            else:
                _, port, b, st = q
                self.m_present_backing(port, b, None)
            self.stamp()
            last = st
        self.queued = []
        if last is not None:
            s.fifo.wait_stamp(last, label=lab + ": the rung commands")
            self.predict(ok=True)
        self.checkpoint(lab)
        self.root_dump(lab)

    def a_suspend(self):
        s, r = self.s, self.r
        lab = self.mark("suspend, resume on a fresh device")
        s.api("willSuspend", label=lab)
        self.run_state = 1
        s.api("finishSuspend", label=lab)
        self.predict(ok=True)
        for b in self.backings:
            b.host = None
        for port in sorted(self.nubs):
            self.nubs[port].paused = True
        # The target's displays: those the source had, as the host creates
        # them again, sometimes one more or one fewer.
        ports = [p for p in range(self.nports) if p in self.plugged]
        if r.chance(0.25):
            p = r.below(self.nports)
            if p in ports:
                ports.remove(p)
            else:
                ports.append(p)
                ports.sort()
        specs = []
        for p in ports:
            d = self.plugged.get(p)
            if d is not None and d.modes:
                specs.append(d.spec())
            else:
                self.serials += 1
                specs.append(self.new_spec(p, self.serials))
        cfg = {k: v for k, v in s._config.items() if k != "op"}
        cfg["displays"] = specs
        s.api("recreate", label=lab, config=cfg)
        for d in self.disps:
            d.alive = False
        self.disps = []
        self.plugged, self.parked = {}, {}
        old = self.nubs
        self.nubs = {}
        for i, spec in enumerate(specs):
            d = DispM(i, spec["port"], spec["serial"], spec["name"], [tuple(m) for m in spec["modes"]])
            self.disps.append(d)
            self.plugged[d.port] = d
            self.parked[d.port] = d
        self.run_state = 3
        s.api("willResume", label=lab)
        self.predict(ok=True)
        s.api("didResume", label=lab)
        for port in sorted(old):
            if self.root_words.get(port):
                self.attach_page(port, old[port].page, restoring=True)
                self.nubs[port].asleep = False
        self.run_state = 0
        for port in sorted(self.nubs):
            self.finish_restore(self.nubs[port])
        self.interrupt()
        self.checkpoint(lab)
        self.root_dump(lab)
        for d in self.live_displays():
            if d.frame is not None:
                self.encode(d, lab)
                self.counts(d)

    def finish_restore(self, nub):
        """GuestDisplay::finish_restore."""
        pg = nub.page
        nub.will_restore = False
        nub.did_restore = True
        if pg.get(O_GLEN, "Q"):
            ok = self.m_glyph(nub, pg.get(O_GLYPH, "Q"), pg.get(O_GLEN, "Q"), pg.get(O_GSTRIDE, "Q"),
                              pg.get(O_GW, "H"), pg.get(O_GW + 2, "H"), pg.get(O_GW + 4, "H"),
                              pg.get(O_GW + 6, "H"), pg.get(O_GSUM, "I"), self.word_at)
            if not ok:
                return
            self.m_show(nub, pg.b[O_HSHOW])
        pg.b[O_CM_FORCE] = 1
        if pg.get(O_PCOUNT, "Q") and not pg.b[O_ASLEEP]:
            n = pg.get(O_GAMMA + 16, "I")
            gamma = None
            if n:
                gamma = next(g for g in self.gammas + [self.gamma_one, self.gamma_bad, self.gamma_huge]
                             if g.va == pg.get(O_GAMMA, "Q") and g.n == n and g.csum == pg.get(O_GAMMA + 20, "I")
                             and g.mapped == pg.get(O_GAMMA + 8, "Q"))
            if not self.m_present_mapped(nub.port, pg.get(O_PMAP, "Q"), gamma, nub=nub):
                return
        elif pg.b[O_ASLEEP]:
            self.m_sleep(nub, 1)
        self.sm(nub)

    def a_reset(self, how):
        s, r = self.s, self.r
        lab = self.mark(how)
        s.api(how, label=lab)
        for port in sorted(self.nubs):
            self.detach_page(port)
        self.r1014 = 0
        self.r1018 = 0
        for b in self.backings:
            b.host = None
        self.checkpoint(lab, pages=False)
        for port in sorted(self.last_page):
            self.cmp_page(self.last_page[port], "%s: port %d's page left" % (lab, port))
        self.root_dump(lab)
        if how == "reset":
            s.w(abi.R_VERSION, self.version)
            if self.efi_on and r.chance(0.6):
                self.bring_up()
                self.efi_boot()
                self.checkpoint(lab + ": set up again")
                return
        self.bring_up()
        self.checkpoint(lab + ": set up again")

    # ---- the end ----

    def final_check(self):
        s = self.s
        self.step_no = self.steps
        if self.paused:
            self.a_unpause()
        lab = self.mark("the end")
        self.submit(abi.OP_NOP, b"", lab)
        self.checkpoint(lab)
        self.root_dump(lab)
        for d in self.live_displays():
            if d.frame is not None and d.frame[0] != "stale":
                self.encode(d, lab)
            self.counts(d)

    def final_death(self):
        """A death as the last action (docs/testing.md 7.3, 7.4)."""
        s, r = self.s, self.r
        self.step_no = self.steps
        kind = self.death
        clients = [p for p in sorted(self.nubs) if self.nubs[p].client is not None]
        if self.paused:
            self.a_unpause()
        if kind == "gamma_one":
            if not clients:
                p = r.below(self.nports)
                if p not in self.plugged:
                    self.a_plug(p)
                if p not in self.nubs:
                    self.a_connect(p)
                clients = [p]
            p = clients[0]
            b = self.backings[0]
            lab = self.mark("present with a one-entry gamma table on port %d: the host dies" % p)
            self.checkpoint(lab + ": before")
            self.s.fifo.cmd(abi.OP_DISPLAY_TRANSACTION3, p_present3(p, b.ref, *self.gamma_one.payload()))
            self.s.fifo.doorbell()
            # The death is the display's, on its
            # queue, not the FIFO's: the present's stamp can land before it,
            # and a run could then end before the death. So the program
            # ends in a wait only the death ends (0x102c never reads 2), and
            # the death always lands in it.
            self.s.wait(mmio=abi.R_FAULT_MASK, mmio_eq=2, timeout_ms=10000, label=lab + " [dies]")
        elif kind in ("doorbell_paused", "cursor_paused", "pause_twice"):
            if not self.nubs:
                p = r.below(self.nports)
                self.a_connect(p)
            p = sorted(self.nubs)[0]
            if kind == "cursor_paused" and self.nubs[p].client is None:
                if p not in self.plugged:
                    self.a_plug(p)
            lab = self.mark("pause, then %s: the host dies" % {
                "doorbell_paused": "the doorbell on port %d" % p, "cursor_paused": "cursorPosition on port %d" % p,
                "pause_twice": "pause again"}[kind])
            s.api("pause", label=lab)
            self.run_state = 5
            self.checkpoint(lab + ": paused", pages=False)
            if kind == "doorbell_paused":
                s.w(abi.R_DISPLAY_DOORBELL, p)
            elif kind == "cursor_paused":
                s.api("cursorPosition", display=self.nubs[p].client.index, label=lab)
            else:
                s.api("pause", label=lab)
        else:
            ok = [q for q in clients if self.nubs[q].page.get(O_STATE, "I") == 2]
            if not ok:
                p = 0
                if p not in self.plugged:
                    self.a_plug(p)
                if p in self.nubs:
                    self.a_disconnect(p)
                self.a_connect(p)
                for _ in range(3):
                    if self.nubs[p].page.b[O_ACKP]:
                        self.a_ack(p)
                clients = [p]
            p = next(q for q in clients if self.nubs[q].page.get(O_STATE, "I") == 2)
            d = self.nubs[p].client
            lab = self.mark("the guest writes mode count 200 on port %d, then a new mode list: the host dies" % p)
            self.guest_write(self.nubs[p].page, O_COUNT, "H", 200)
            self.checkpoint(lab + ": before")
            s.api("set_modes", display=d.index, modes=[[48, 24, 60]], label=lab)
        self.dies = True


# ---- entry points ----

def build(seed, steps, doc):
    w = Walk(seed, steps, doc)
    return w.build()


def walk(seed, steps, doc=""):
    w = Walk(seed, steps, doc)
    w.build()
    return w


def check_obs(w, obs):
    """Compares a run's observations (normalized) with the walk's
    predictions; returns complaints."""
    bad = []
    by_i = {}
    for o in obs:
        if isinstance(o.get("i"), int):
            by_i.setdefault(o["i"], []).append(o)
    for i, p in sorted(w.pred.items()):
        got = by_i.get(i)
        if not got:
            if getattr(w, "dies", False):
                continue
            bad.append("op %d: no observation (want %s)" % (i, json.dumps(p)[:200]))
            continue
        o = got[-1]
        for k, v in p.items():
            if o.get(k) != v:
                lab = o.get("label", "")
                if k == "callbacks":
                    bad.append("op %d %s: callbacks\n   want %s\n   got  %s" % (
                        i, lab, json.dumps(v)[:1500], json.dumps(o.get(k))[:1500]))
                else:
                    bad.append("op %d %s: %s want %s got %s" % (i, lab, k, json.dumps(v)[:300], json.dumps(o.get(k))[:300]))
                break
    out = obs[-1] if obs else {}
    want = "died" if getattr(w, "dies", False) else "completed"
    if out.get("kind") != want:
        bad.append("outcome %s, the model says %s" % (out.get("kind"), want))
    for o in obs:
        if o.get("ev") == "wait" and o.get("ok") is False and not (o.get("label", "").endswith("[held]")
                                                                   or o.get("label", "").endswith("[dies]")):
            bad.append("wait failed: %s" % o.get("label"))
    return bad


def check(paths=None):
    """Every expected file of interplay_displaywalk against the model."""
    import glob
    import os
    from scenarios.interplay_displaywalk import SEEDS, __doc__ as doc
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bad = 0
    for seed, steps in SEEDS:
        w = walk(seed, steps, doc)
        for p in sorted(glob.glob(os.path.join(here, "expected", "interplay_displaywalk", w.name, "*.jsonl"))):
            lines = [json.loads(l) for l in open(p) if l.strip()]
            complaints = check_obs(w, lines[1:])
            for c in complaints:
                print("%s: %s" % (os.path.relpath(p, here), c))
            bad += len(complaints)
    return bad


if __name__ == "__main__":
    import sys
    sys.exit(1 if check() else 0)
