"""The display's guest side: the shared state page, the display opcodes'
payloads, and IOSurface backings in task 0 (docs/testing.md, section 6.21).

Offsets are restated here; Rust paths name items of reims-vgpu-core.
Nothing here is imported from the library's code.
"""
import struct

from . import abi
from .abi import PAGE
from .task import GPUTask

# ---- the shared state page (display::page) ----
SS_SERIAL = 0x000
SS_NAME = 0x004            # char[14]
SS_PORT = 0x012            # u16
SS_MM = 0x014              # u16 w, u16 h, rounded
SS_GLYPH_MAX = 0x018       # u16 w, u16 h
SS_1C = 0x01C              # u32 1024 at init
SS_CURSOR_HANDLERS = 0x020  # u32: 1 show, 2 move
SS_LIVE_RESIZE = 0x051     # u8, guest
SS_PROTECTION = 0x068      # u64
SS_RAISED = 0x100          # u32 raisedInterrupts (host ORs)
SS_ENABLED = 0x104         # u32 enabledInterrupts (guest)
SS_EXPECTED_ACK = 0x200    # u32
SS_MODE_COUNT = 0x208      # u16
SS_MODES = 0x210           # 128 x 16 bytes
SS_COLOR_MATRICES = 0xA10  # float[2][3][4]
SS_CM_READ = 0xA70         # u8 read index
SS_CM_FORCE = 0xA71        # u8 force read
SS_CM_WRITE = 0xA72        # u8 write index
SS_CM_DIRTY = 0xA73        # u8 dirty
SS_CURSOR_POS = 0xE00      # u32 {u16 x, u16 y}
SS_GUEST_CURSOR_SHOW = 0xE04  # bool, guest
SS_GLYPH = 0xF00           # glyph record, to 0xf2a
SS_HOST_CURSOR_SHOW = 0xF20
SS_GAMMA = 0xF80           # gamma record, to 0xf98
SS_PRESENT_COUNT = 0xF98   # u64
SS_PRESENTED = 0xFA0       # u32 surface; 0xfa4 = 0
SS_ACK_PENDING = 0xFA9     # bool
SS_PRESENTED_MAPPING = 0xFB0  # u64
SS_ASLEEP = 0xFB8          # bool
SS_LAST_ACK = 0xFBC        # u32
SS_STATE = 0xFC0           # u32
SS_ICC = 0xFC8             # u64 VA, u32 length at 0xfd0

# raisedInterrupts bits (the guest display's state machine, and the present
# completion).
IRQ_PRESENT = 2
IRQ_CONNECT = 4
IRQ_DISCONNECT = 8
IRQ_UPDATE = 12

# The host's part of the page, as ranges a scenario dumps; the rest of the
# page is guest-owned or never written, and a dump of it would only repeat
# what the scenario wrote.
HOST_RANGES = [(0x000, 0x80), (0x100, 0x8), (0x200, 0x20), (0xA10, 0x90), (0xE00, 0x8),
               (0xF00, 0xD8)]


def mode_bytes(w, h, hz):
    """One mode as the host writes it: {u16 w, u16 h, u32 refresh 16.16,
    bits}, 16 bytes (GuestDisplay::sync_mode_list writes them)."""
    return struct.pack("<HHIB7x", w, h, int(hz * 65536 + 0.5), 0)


# ---- payloads, #pragma pack(4) ----

def p_set_shared(port, page):
    return struct.pack("<II", port, page)


def p_ack(port, ack):
    return struct.pack("<II", port, ack)


def p_glyph(port, va, mapped, stride, w, h, hx=0, hy=0, csum=0, task=0):
    """CursorGlyph (4): 44 bytes."""
    return struct.pack("<IIQQQHHHHI", port, task, va, mapped, stride, w, h, hx, hy, csum)


def p_show(port, show):
    """CursorShow (5): {u32 port, BOOL show}, 8 bytes with the pack's pad."""
    return struct.pack("<Ib3x", port, show)


def p_present2(port, ref, task=0):
    """DisplayTransaction2 (6): {u32 port, u32 surface ref, u32 taskID}."""
    return struct.pack("<III", port, ref, task)


def p_present3(port, ref, gva=0, glen=0, gcount=0, gsum=0, task=0):
    """DisplayTransaction3 (7): 36 bytes."""
    return struct.pack("<IIIQQII", port, task, ref, gva, glen, gcount, gsum)


def p_swap(port, mapping, gva=0, glen=0, gcount=0, gsum=0, task=0):
    """DisplaySwapMapping (8): 40 bytes."""
    return struct.pack("<IIQQQII", port, task, mapping, gva, glen, gcount, gsum)


def p_sleep(port, asleep):
    return struct.pack("<Ib3x", port, asleep)


def p_props(port, props):
    """SetProperties (10): {u32 port, u32 count}, then per property {u32
    property, u32 wordCount, words}."""
    body = b"".join(struct.pack("<II", p, len(w)) + struct.pack("<%dI" % len(w), *w) for p, w in props)
    return struct.pack("<II", port, len(props)) + body


def p_compositor(port, x, y, epoch, scale):
    """CompositorParameters (11): {u32 port, {u16 x, u16 y}, u64 epoch, u8
    scale}, 20 bytes."""
    return struct.pack("<IHHQB3x", port, x, y, epoch, scale)


def p_icc(port, length, va, mapped):
    """SetGuestICCProfile (12): {u32 port, u32 length, u64 VA, u64 mapped}."""
    return struct.pack("<IIQQ", port, length, va, mapped)


def floats(*v):
    return [struct.unpack("<I", struct.pack("<f", x))[0] for x in v]


IDENTITY = floats(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)


def gamma_words(n, curve=lambda c, i, n: i / max(n - 1, 1)):
    """A gamma table of n entries a channel, 3n floats as u32 words, and its
    checksum: the u32 sum of w[i] + w[n+i] + w[2n+i] (display::gamma)."""
    words = []
    for c in range(3):
        words += floats(*[curve(c, i, n) for i in range(n)])
    csum = sum(words) & 0xFFFFFFFF
    return words, csum


# ---- a guest with displays ----

class DisplayGuest:
    """A brought-up device with task 0 (the kernel task the display commands
    read from) and one shared state page a port.

    Task 0 is a GPUTask: page 0 its object list, pages 1-3 records, 8.. data
    (glyphs, gamma tables, backings)."""

    def __init__(self, s, version=abi.MAX_VERSION, task0=True, pages=64):
        self.s = s
        s.bring_up(version=version)
        self.pages = {}
        self.t0 = None
        if task0:
            self.t0 = GPUTask(s, tid=0, pages=pages)
            self.t0.define()

    def page(self, port, enabled=None):
        """Allocates port's shared page; enabled writes +0x104 first."""
        s = self.s
        p = s.alloc_page()
        self.pages[port] = p
        if enabled is not None:
            s.u32(s.pa(p, SS_ENABLED), enabled)
        return p

    def ss(self, port, off=0):
        return self.s.pa(self.pages[port], off)

    def connect(self, port=0, enabled=None, **kw):
        """Opcode 1 with a fresh page for port."""
        if port not in self.pages:
            self.page(port, enabled)
        return self.s.submit(abi.OP_DISPLAY_SET_SHARED_STATE_PAGE,
                             p_set_shared(port, self.s.pfn(self.pages[port])), **kw)

    def disconnect(self, port=0, **kw):
        return self.s.submit(abi.OP_DISPLAY_SET_SHARED_STATE_PAGE, p_set_shared(port, 0), **kw)

    def ack(self, port=0, value=None, **kw):
        """Opcode 2; value defaults to the expectedAck the scenario tracks
        (1 after the first connect)."""
        return self.s.submit(abi.OP_DISPLAY_ACK, p_ack(port, 1 if value is None else value), **kw)

    def dump_host(self, port=0, label=None):
        """The host-written parts of the page."""
        for off, n in HOST_RANGES:
            self.s.dump(self.ss(port, off), n, label=(label or "page") + " +0x%x" % off)

    def dump_page(self, port=0, label=None):
        """The whole shared page (a SHA-256 over 16 KiB, plus the host ranges)."""
        self.s.dump(self.ss(port), PAGE, label=label)

    def diff_page(self, port=0, label=None):
        self.s.diff(label=label, gpa=self.ss(port), length=PAGE)

    # ---- task 0 data ----

    def backing(self, w, h, fmt=b"BGRA", bpr=None, bpe=4, pixels=None, planes=None, length=None):
        """An IOSurface backing (type 4) in task 0: the 0x14-byte record
        {u64 length, u32 task VA page, u32 pixel format, u8 planeCount},
        then planeCount x {u32 offset, u32 width, u32 height, u32
        bytesPerRow:24 | bytesPerElement << 24}
        (resources::texture::backing::BackingPlane). The pixels go in task
        pages. Returns (ref, va)."""
        t = self.t0
        bpr = w * bpe if bpr is None else bpr
        size = bpr * h
        length = (size + PAGE - 1) // PAGE * PAGE if length is None else length
        va = t.reserve(max(length, PAGE))
        if pixels is not None:
            t.write(va, pixels)
        planes = [(0, w, h, bpr, bpe)] if planes is None else planes
        fmtv = struct.unpack(">I", fmt)[0] if isinstance(fmt, bytes) else fmt
        rec = struct.pack("<QIIB3x", length, va // PAGE, fmtv, len(planes))
        rec += b"".join(struct.pack("<IIII", o, pw, ph, (pb & 0xFFFFFF) | (pe << 24)) for o, pw, ph, pb, pe in planes)
        return t.ol.add(abi.T_IOSURFACE_BACKING, rec), va

    def install(self):
        return self.t0.install()


def pattern(w, h, seed=0):
    """BGRA pixels: (x, y) = {x*4+seed, y*8, (x^y)+seed, 0xff}."""
    return b"".join(bytes([(x * 4 + seed) & 0xFF, (y * 8) & 0xFF, ((x ^ y) + seed) & 0xFF, 0xFF])
                    for y in range(h) for x in range(w))


def display(port=0, serial=None, name="suite", modes=((64, 32, 60),), **kw):
    """A display spec for Scenario(displays=[...]) or api new_display."""
    d = {"port": port, "serial": 1000 + port if serial is None else serial, "name": name}
    if modes is not None:
        d["modes"] = [list(m) for m in modes]
    d.update(kw)
    return d


def icc_profile(gamma=2.2):
    """A minimal ICC v2 RGB display profile, enough for
    CGColorSpaceCreateWithICCData: the 128-byte header, a tag table, desc,
    wtpt (D50), the sRGB-ish rXYZ/gXYZ/bXYZ primaries (Bradford-adapted to
    D50) and one curv tag with a pure gamma shared by the three TRCs."""
    def s15(v):
        return struct.pack(">i", int(round(v * 65536)))

    def xyz(x, y, z):
        return b"XYZ " + bytes(4) + s15(x) + s15(y) + s15(z)

    text = b"suite\x00"
    desc = b"desc" + bytes(4) + struct.pack(">I", len(text)) + text + bytes(4 + 4 + 2 + 1 + 67)
    curv = b"curv" + bytes(4) + struct.pack(">IH", 1, int(round(gamma * 256))) + bytes(2)
    tags = [(b"desc", desc), (b"wtpt", xyz(0.9642, 1.0, 0.8249)),
            (b"rXYZ", xyz(0.4361, 0.2225, 0.0139)), (b"gXYZ", xyz(0.3851, 0.7169, 0.0971)),
            (b"bXYZ", xyz(0.1431, 0.0606, 0.7141)), (b"rTRC", curv), (b"gTRC", curv), (b"bTRC", curv)]
    table = struct.pack(">I", len(tags))
    body = b""
    offset = 128 + 4 + 12 * len(tags)
    placed = {}
    for sig, data in tags:
        if data not in placed:
            placed[data] = offset + len(body)
            body += data + bytes((-len(data)) % 4)
        table += sig + struct.pack(">II", placed[data], len(data))
    size = offset + len(body)
    header = struct.pack(">I", size) + b"none" + struct.pack(">I", 0x02100000) + b"mntrRGB XYZ "
    header += struct.pack(">6H", 2026, 1, 1, 0, 0, 0) + b"acspAPPL" + bytes(4) + b"none" + bytes(4)
    header += bytes(8) + bytes(4) + s15(0.9642) + s15(1.0) + s15(0.8249) + b"none"
    header += bytes(128 - len(header))
    assert len(header) == 128
    return header + table + body
