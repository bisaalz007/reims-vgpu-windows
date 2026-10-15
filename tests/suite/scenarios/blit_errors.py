"""The blit decoder's checks (docs/testing.md, sections 6.12-6.14).

A decode error faults the exec and the command buffer is never committed
(section 6.12). So every faulting exec below starts with a FillBuffer of the
output buffer that would have landed had anything run: the dump after the
faults shows none did. Every fault is followed by a resume (0x1024)."""
import struct

from pgtest import abi
from pgtest.compute_objects import Blit as B, PF_RGBA8UNORM, Rig
from pgtest.registry import scenario
from pgtest.task import rec


class Checks:
    """A rig with an output buffer, a buffer and a texture over one, a
    private buffer, a fence, two ICBs of two commands, and refs of the
    wrong kinds for faulting records to name."""

    def __init__(self, name, doc):
        r = self.r = Rig(name, doc)
        self.out, self.out_va = r.buffer()
        self.buf, _ = r.buffer()
        self.tb, _ = r.buffer()
        self.tex = r.texture(self.tb, 4, 4)
        self.priv, _ = r.buffer(flags=1)
        self.fence = r.fence()
        self.icb = r.icb([bytes(0x100)] * 2)
        self.icb2 = r.icb([bytes(0x100)] * 2)
        self.pso = r.kernel()
        self.past = len(r.t.ol.entries)
        r.ready()

    def fault(self, label, *records):
        self.r.blit(B.fill(self.out, 0, 4, 0x99), *records, fault=True, label=label)

    def end(self):
        self.r.dump(self.out_va, 0x8, "out: no faulting exec ran")
        return self.r.end()


def all_records(k, tex, buf, icb, fence):
    """One record of each of the 24 commands, naming these refs."""
    t, b = tex, buf
    return [
        (300, B.copy_buffer_to_texture(b, t, 0, 16, 64, (1, 1, 1), (0, 0, 0))),
        (301, B.copy_buffer_to_buffer(b, b, 0, 0x10, 4)),
        (302, B.copy_texture_to_buffer(t, b, (0, 0, 0), (1, 1, 1), 0, 16, 64)),
        (303, B.copy_texture_to_texture(t, t, (0, 0, 0), (1, 1, 1), (1, 0, 0))),
        (304, B.copy_texture_to_texture_options(t, t, (0, 0, 0), (1, 1, 1), (1, 0, 0))),
        (305, B.copy_icb(icb, icb, 0, 1, 1)),
        (306, B.fill(b, 0, 4, 1)),
        (307, B.mipmaps(t)),
        (308, B.optimize(308, t)),
        (309, B.optimize(309, t)),
        (310, B.optimize_image(310, t)),
        (311, B.optimize_image(311, t)),
        (312, B.icb_range(312, icb, 0, 1)),
        (313, B.icb_range(313, icb, 0, 1)),
        (314, B.synchronize(b)),
        (315, B.synchronize_image(t)),
        (316, B.fence(316, fence)),
        (317, B.fence(317, fence)),
        (318, B.copy_slices(t, t)),
        (319, B.fill_pattern(b, 0, 4, 1)),
        (320, B.fill_texture_bytes(t, 0, 0, (1, 1, 1), (0, 0, 0), b, 0, 4)),
        (321, B.fill_texture_color(t, 0, 0, (1, 1, 1), (0, 0, 0), (1, 1, 1, 1), PF_RGBA8UNORM)),
        (322, B.invalidate(t)),
        (323, B.invalidate_image(t)),
    ]


def shorten(record):
    """The same record with its last payload byte cut, and the header's
    length saying so: readOrThrow<T> fails ("Failed to read")."""
    cmd, n = struct.unpack_from("<II", record)
    return struct.pack("<II", cmd, n - 1) + record[8:n - 1]


@scenario
def blit_err_short_records():
    """Every one of the 24 commands faults when its record is one byte
    shorter than its command struct, tail padding included (0x58 for 302,
    section 6.13)."""
    k = Checks("blit_err_short_records", blit_err_short_records.__doc__)
    for cmd, record in all_records(k, k.tex, k.buf, k.icb, k.fence):
        k.fault("%d short" % cmd, shorten(record))
    return k.end()


@scenario
def blit_err_null_refs():
    """Every command that needs an object faults for ref 0: buffers,
    textures and fences through the deserializer's non-null getters ("Got
    invalid ..."), ICBs through the decoder's own ("Got invalid indirect
    command buffer"). SynchronizeResource (314) looks its ref up with the
    nullable getter and runs with nil (blit_texture_maintenance)."""
    k = Checks("blit_err_null_refs", blit_err_null_refs.__doc__)
    for cmd, record in all_records(k, 0, 0, 0, 0):
        if cmd != 314:
            k.fault("%d, all refs 0" % cmd, record)
    # The second reference of each two-object command, alone at 0.
    k.fault("300 texture 0", B.copy_buffer_to_texture(k.buf, 0, 0, 16, 64, (1, 1, 1), (0, 0, 0)))
    k.fault("301 destination 0", B.copy_buffer_to_buffer(k.buf, 0, 0, 0, 4))
    k.fault("302 buffer 0", B.copy_texture_to_buffer(k.tex, 0, (0, 0, 0), (1, 1, 1), 0, 16, 64))
    k.fault("303 destination 0", B.copy_texture_to_texture(k.tex, 0, (0, 0, 0), (1, 1, 1), (0, 0, 0)))
    k.fault("304 destination 0", B.copy_texture_to_texture_options(k.tex, 0, (0, 0, 0), (1, 1, 1), (0, 0, 0)))
    k.fault("305 destination 0", B.copy_icb(k.icb, 0, 0, 1, 0))
    k.fault("318 destination 0", B.copy_slices(k.tex, 0))
    k.fault("320 buffer 0", B.fill_texture_bytes(k.tex, 0, 0, (1, 1, 1), (0, 0, 0), 0, 0, 4))
    return k.end()


@scenario
def blit_err_wrong_refs():
    """Refs of the wrong kind or past the object list fault: a texture where
    a buffer is expected (no buffer in its resource), a buffer where a
    texture or an ICB is, a pipeline (type 7) where any resource is
    ("Unexpected object type (7) for resource"), a buffer where a fence is
    ("Invalid object type (1) for fence"), and a ref past the list."""
    k = Checks("blit_err_wrong_refs", blit_err_wrong_refs.__doc__)
    for label, (tex, buf, icb, fence) in (("texture as buffer", (k.tex, k.tex, k.icb, k.fence)),
                                          ("buffer as texture", (k.buf, k.buf, k.icb, k.fence)),
                                          ("pipeline", (k.pso, k.pso, k.pso, k.fence)),
                                          ("past the list", (k.past, k.past, k.past, k.past))):
        for cmd, record in all_records(k, tex, buf, icb, fence):
            if cmd == 314:
                continue
            if label == "texture as buffer" and cmd not in (301, 306, 319):
                continue
            if label == "buffer as texture" and cmd in (301, 305, 306, 312, 313, 316, 317, 319):
                continue
            if label == "pipeline" and cmd in (316, 317):
                continue
            k.fault("%d, %s" % (cmd, label), record)
    k.fault("305, buffer as ICB", B.copy_icb(k.buf, k.icb, 0, 1, 0))
    k.fault("312, buffer as ICB", B.icb_range(312, k.buf, 0, 1))
    k.fault("316, buffer as fence", B.fence(316, k.buf))
    k.fault("317, buffer as fence", B.fence(317, k.buf))
    return k.end()


@scenario
def blit_err_ranges():
    """The range checks, each at the first value that fails and at an end
    that overflows 64 bits: FillBuffer and FillBufferWithPattern past the
    buffer's 0x4000 bytes ("Invalid buffer range"), CopyIndirectCommandBuffer
    past the source ICB's 2 commands ("Invalid indirect command buffer
    range") and at destination index 2 ("Invalid destination index"),
    OptimizeIndirectCommandBuffer and ResetCommandsInBuffer past 2."""
    k = Checks("blit_err_ranges", blit_err_ranges.__doc__)
    big = 0xFFFFFFFFFFFFFFFF
    k.fault("306 [0x3ff0, 0x4001)", B.fill(k.buf, 0x3ff0, 0x11, 1))
    k.fault("306 overflow", B.fill(k.buf, 1, big, 1))
    k.fault("319 [0x3ff0, 0x4001)", B.fill_pattern(k.buf, 0x3ff0, 0x11, 1))
    k.fault("319 overflow", B.fill_pattern(k.buf, 1, big, 1))
    k.fault("305 [1, 3)", B.copy_icb(k.icb, k.icb2, 1, 2, 0))
    k.fault("305 overflow", B.copy_icb(k.icb, k.icb2, 1, big, 0))
    k.fault("305 index 2", B.copy_icb(k.icb, k.icb2, 0, 1, 2))
    for cmd in (312, 313):
        k.fault("%d [1, 3)" % cmd, B.icb_range(cmd, k.icb, 1, 2))
        k.fault("%d overflow" % cmd, B.icb_range(cmd, k.icb, 1, big))
    k.r.blit(B.copy_icb(k.icb, k.icb2, 0, 2, 1), B.icb_range(312, k.icb, 0, 2), B.icb_range(313, k.icb2, 2, 0))
    return k.end()


@scenario
def blit_err_fill_texture_bytes():
    """FillTextureWithBytes's own checks: a length below the pixel format's
    block size (3 bytes for RGBA8Unorm, whose block is 4: "Buffer length for
    format ... < minLength"), an offset and length past the buffer, and an
    end that overflows ("Invalid buffer length")."""
    k = Checks("blit_err_fill_texture_bytes", blit_err_fill_texture_bytes.__doc__)

    def ftb(buf, off, n):
        return B.fill_texture_bytes(k.tex, 0, 0, (1, 1, 1), (0, 0, 0), buf, off, n)
    k.fault("length 3", ftb(k.buf, 0, 3))
    k.fault("[0x3ffd, 0x4001)", ftb(k.buf, 0x3ffd, 4))
    k.fault("overflow", ftb(k.buf, 0xFFFFFFFFFFFFFFFE, 4))
    k.r.blit(ftb(k.buf, 0x3ffc, 4))
    return k.end()


@scenario
def blit_err_framing():
    """How the blit decoder reads records: an unknown command id (299, 324
    and 0: "UnrecognizedCommand"), a record length under 8 ("Underflow"),
    and a length past the segment ("Failed to read")."""
    k = Checks("blit_err_framing", blit_err_framing.__doc__)
    for cmd in (299, 324, 0):
        k.fault("command %d" % cmd, rec(cmd, struct.pack("<I", k.buf)))
    k.fault("length 4", rec(306, length=4))
    k.fault("length 7", rec(306, length=7))
    k.fault("length past the segment", rec(306, bytes(0x18), length=0x40))
    return k.end()


@scenario
def blit_err_fill_texture_bytes_private():
    """FillTextureWithBytes from a buffer created private (flags bit 0).
    In-process it is a Metal private buffer: "Invalid buffer storage mode",
    and the exec faults. In the service a private buffer is a detach buffer
    over the guest's pages with Shared storage, so the check passes and the
    texel is filled with the buffer's guest bytes."""
    r = Rig("blit_err_fill_texture_bytes_private", blit_err_fill_texture_bytes_private.__doc__)
    tb, t_va = r.buffer()
    tex = r.texture(tb, 4, 4)
    priv, _ = r.buffer(flags=1, contents=bytes([0x10, 0x20, 0x30, 0x40]))
    r.ready()
    r.blit(B.fill_texture_bytes(tex, 0, 0, (2, 1, 1), (1, 1, 0), priv, 0, 4), either=True, label="private buffer")
    r.dump(t_va, 0x40, "texture")
    return r.end()
