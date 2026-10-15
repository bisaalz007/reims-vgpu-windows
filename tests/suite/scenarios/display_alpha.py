"""What the display does with a frame's alpha (docs/testing.md, sections
6.21 and 6.22): a 4 x 2 frame whose pixels carry nonzero colours with
alpha 0x00, 0x7f (0.5 in 'RGhA'), 0xff and one more value a row, presented
and encoded at its own size, so every encoded byte is one of the frame's
pixels.

Each present scenario runs one display through five presents of the same
frame, each waited for (new_frame) and encoded:
  1. no gamma table and an identity colour matrix: the copy shader;
  2. a colour matrix swapping red and blue (10, property 0), still no
     gamma table: the copy shader ignores the matrix;
  3. the matrix back to identity and a 256-entry identity gamma table: the
     gamma shader from then on;
  4. the swapping matrix again, with the table;
  5. no table in the present: the gamma shader stays.
The surfaces are opcode 8 mapper surfaces (the IOSurface host device;
'BGRA', 'RGBA', 'RGhA'; map and isolate only, as it needs a memory map) and
opcode 7 task-0 backings ('BGRA' and 'RGBA', every setup). The EFI scenarios
present a 4 x 2 framebuffer, alone and after the guest's present with a
gamma table and the swapping matrix.

What the display does (measured on these scenarios): the copy shader, used
until a gamma table arrives, carries each pixel's alpha into the encoded
frame unchanged ('RGhA' converted to unorm8, 0.5 to 0x80) and ignores the
colour matrix, for opcode 7, opcode 8 and EFI alike; the colour is not
premultiplied or blended. Once a table has arrived, the gamma shader writes
alpha 0xff for every pixel whatever the surface's alpha, and applies the
matrix, for the rest of the display's life, EFI's frames included.

The pixels are the guest's own bytes, so nothing but the encodes (drawn by
the library's shaders: host-keyed, rule 5) and the callbacks is locked. Rows
are 64 bytes apart (the surfaces' bytesPerRow, the backings' and EFI's stride).
"""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.display_guest import IDENTITY, floats, gamma_words, p_present3, p_swap
from pgtest.efi_guest import EFI
from pgtest.iosfc_guest import CMD_MAP, desc_v2
from pgtest.registry import scenario
from pgtest.scanout_util import IN_PROCESS, MAPPED, Scanout
from pgtest.scenario import SETUPS
from pgtest.task import GPUTask

W, H, BPR = 4, 2, 64

# (r, g, b) a row, and the alphas along it.
ROWS = [((0xA0, 0x60, 0x20), (0x00, 0x7F, 0xFF, 0x40)),
        ((0x30, 0x90, 0xF0), (0xFF, 0x7F, 0x00, 0xC0))]
# 'RGhA': half floats, alphas 0, 0.5, 1, 0.25 and 1, 0.5, 0, 0.75.
HROWS = [((0.625, 0.375, 0.125), (0.0, 0.5, 1.0, 0.25)),
         ((0.1875, 0.5625, 0.9375), (1.0, 0.5, 0.0, 0.75))]

SWAP_RB = floats(0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0)


def pixels(fmt):
    """The 4 x 2 frame in fmt, rows BPR bytes apart (the padding 0)."""
    out = b""
    if fmt == b"RGhA":
        for (r, g, b), alphas in HROWS:
            row = b"".join(struct.pack("<4e", r, g, b, a) for a in alphas)
            out += row + bytes(BPR - len(row))
        return out
    for (r, g, b), alphas in ROWS:
        row = b"".join(bytes((b, g, r, a) if fmt == b"BGRA" else (r, g, b, a)) for a in alphas)
        out += row + bytes(BPR - len(row))
    return out


def _kernel_task(x, fmt=None):
    """Task 0 (the display commands' task, and the gamma tables'), with one
    4 x 2 backing (type 4: {u64 length, u32 VA page, u32 format, u8
    planeCount}, then {u32 offset, u32 width, u32 height, u32
    bytesPerRow | bytesPerElement << 24}) when fmt is given."""
    t = x.t0 = GPUTask(x.s, tid=0, pages=32)
    if fmt is not None:
        va = t.reserve(PAGE)
        t.write(va, pixels(fmt))
        ref = t.ol.add(abi.T_IOSURFACE_BACKING, struct.pack("<QIIB3x", PAGE, va // PAGE, struct.unpack(">I", fmt)[0], 1)
                       + struct.pack("<IIII", 0, W, H, BPR | 4 << 24))
        x.backings.append((ref, va))
    t.define(label="define task 0")
    t.install(label="task 0's object list")


def _mapper_surface(x, mid, fmt):
    """A 4 x 2 mapper surface over one guest page, mapped (command 1)."""
    d = x.iosfc
    root, data = d.surface_pages(1)
    x.s.mem(x.s.pa(data), pixels(fmt))
    d.put_desc(mid, desc_v2(root, W, H, fmt=fmt, bpr=BPR, bpe=8 if fmt == b"RGhA" else 4))
    d.status(d.submit((CMD_MAP, mid)), "map surface %d" % mid)


def _encode(x, label):
    x.s.encode(0, W, H, label=label)


def _five_presents(x, present):
    """The five presents of the module's docstring; present(gamma) writes
    one on the root FIFO and returns its stamp."""
    gamma = x.gamma(256, lambda c, i, n: i / (n - 1))
    steps = [("no gamma table, identity matrix", None, None),
             ("no gamma table, red and blue swapped", SWAP_RB, None),
             ("identity gamma table, identity matrix", IDENTITY, gamma),
             ("identity gamma table, red and blue swapped", SWAP_RB, gamma),
             ("no table in the present, after one", None, None)]
    for label, matrix, g in steps:
        if matrix is not None:
            x.matrix(0, matrix, label="matrix: " + label)
        x.go(0, present(g), label)
        x.expect("newFrame", label=label)
        _encode(x, label)


def _mapped(fmt):
    tag = fmt.decode().lower()
    name = "display_alpha_swap_" + tag
    doc = ("Opcode 8 of a 4 x 2 '%s' mapper surface whose pixels carry alpha "
           "0x00, 0x7f, 0xff (in 'RGhA' 0, 0.5, 1) and one more a row, with nonzero "
           "colours, through the module's five presents: no table, the matrix "
           "without a table, an identity table, the table with the matrix, and no "
           "table after one. "
           "Each frame is encoded at 4 x 2." % fmt.decode())
    x = Scanout(name, doc, setups=MAPPED, host=True, iosfc=True)
    x.negotiate(asks=())
    x.root()
    _kernel_task(x)
    x.connect(0)
    x.iosfc_start()
    _mapper_surface(x, 1, fmt)
    _five_presents(x, lambda g: x.swap_cmd(0, 0, 1, gamma=g))
    x.s.events("end")
    return x.s


def _backing(fmt):
    tag = fmt.decode().lower()
    name = "display_alpha_present_" + tag
    doc = ("Opcode 7 of a 4 x 2 '%s' task-0 backing whose pixels carry alpha "
           "0x00, 0x7f, 0xff and one more a row, with nonzero colours, through "
           "the module's five presents: no table, the matrix without a table, "
           "an identity table, the table with the matrix, and no table after one. Each frame is "
           "encoded at 4 x 2." % fmt.decode())
    x = Scanout(name, doc, setups=SETUPS, host=True)
    x.negotiate(asks=())
    x.root()
    _kernel_task(x, fmt)
    x.connect(0)
    _five_presents(x, lambda g: x.present_cmd(0, 0, 0, gamma=g))
    x.s.events("end")
    return x.s


@scenario
def display_alpha_presents():
    return [_mapped(f) for f in (b"BGRA", b"RGBA", b"RGhA")] + [_backing(f) for f in (b"BGRA", b"RGBA")]


def _efi(x):
    """EFI up on port 0's display with the 4 x 2 frame (trace callbacks, so
    the setup presents once, docs/testing.md 6.22)."""
    s = x.s
    x.efi = EFI(s)
    s.mem(x.efi.gpa(), pixels(b"BGRA"))
    x.efi.setup(W, H, BPR)
    x.expect("newFrame", label="EFI's first frame")
    _encode(x, "EFI frame")


@scenario
def display_alpha_efi():
    """The EFI framebuffer (a zero-copy BGRA8 view of guest memory) of
    the 4 x 2 frame, alpha 0x00, 0x7f, 0xff and one more a row,
    presented before any guest present and encoded at 4 x 2."""
    x = Scanout("display_alpha_efi", display_alpha_efi.__doc__, host=True, efi=True)
    x.negotiate(asks=())
    _efi(x)
    x.s.events("end")
    return x.s


@scenario
def display_alpha_efi_after_gamma():
    """EFI's frame after the guest presented with a gamma table: EFI shows
    the 4 x 2 frame (encoded), the guest connects port 0 while EFI is live
    and presents a 4 x 2 'BGRA' backing (opcode 7) with an identity
    256-entry table and a matrix swapping red and blue (encoded), then a
    trace hit presents EFI's frame again through its weak display reference
    (a new_frame; no mode_change, as size and format are the backing's), now
    through the gamma shader and the matrix the display kept: alpha 0xff,
    red and blue swapped (encoded) (efi_gap_handoff_to_guest)."""
    x = Scanout("display_alpha_efi_after_gamma", display_alpha_efi_after_gamma.__doc__, host=True, efi=True)
    s = x.s
    x.negotiate(asks=())
    _efi(x)
    x.root()
    _kernel_task(x, b"BGRA")
    x.connect(0)
    x.matrix(0, SWAP_RB, label="matrix: red and blue swapped")
    gamma = x.gamma(256, lambda c, i, n: i / (n - 1))
    x.go(0, x.present_cmd(0, 0, 0, gamma=gamma), "backing with the table")
    x.expect("newFrame", label="backing with the table")
    _encode(x, "backing, identity table, red and blue swapped")
    s.trace_fire(x.efi.gpa(0x10), 4)
    x.expect("newFrame", label="EFI again")
    _encode(x, "EFI frame after the table")
    s.events("end")
    return s
