"""The blit decoder (encoder type 2), with the GPU's work read back out of
guest memory (docs/testing.md, sections 4.10, 6.14).

Buffers are shared (type 1, flags 0) over guest pages, and textures are
textures over those buffers (type 8, commandType 9: linear RGBA8, rows of
bytesPerRow bytes), so every blit's result lands in guest memory. Every
buffer the GPU touches is in each exec's resource list (section 6.12)."""
import struct

from pgtest import abi
from pgtest.compute_objects import (Blit as B, Compute as C, PF_R8UNORM, PF_RGBA8UINT, PF_RGBA8UNORM, Rig,
                                    icb_compute_command)
from pgtest.registry import scenario
from pgtest.task import seg


def u32s(*v):
    return struct.pack("<%dI" % len(v), *v)


def ramp(n, start=0):
    return bytes((start + i) & 0xFF for i in range(n))


@scenario
def blit_fill_buffer():
    """FillBuffer (306) sets a byte range of a shared buffer: 0x20 bytes of
    0xab at 0x10, a 1-byte fill at 0x40, a 0-byte fill at 0x50 that writes
    nothing, and a fill ending exactly at the buffer's end (the range check
    allows end == length)."""
    r = Rig("blit_fill_buffer", blit_fill_buffer.__doc__)
    a, a_va = r.buffer()
    r.ready()
    r.blit(B.fill(a, 0x10, 0x20, 0xab), B.fill(a, 0x40, 1, 0x5a), B.fill(a, 0x50, 0, 0x77),
           B.fill(a, 0x3ff0, 0x10, 0xc3))
    r.dump(a_va, 0x60, "A")
    r.dump(a_va + 0x3fe0, 0x20, "A's end")
    return r.end()


@scenario
def blit_fill_pattern():
    """FillBufferWithPattern (319) repeats a 4-byte pattern over the range:
    0x20 bytes at 0x10, and 0x10 bytes ending at the buffer's end."""
    r = Rig("blit_fill_pattern", blit_fill_pattern.__doc__)
    a, a_va = r.buffer()
    r.ready()
    r.blit(B.fill_pattern(a, 0x10, 0x20, 0x04030201), B.fill_pattern(a, 0x3ff0, 0x10, 0xdeadbeef))
    r.dump(a_va, 0x40, "A")
    r.dump(a_va + 0x3fe0, 0x20, "A's end")
    return r.end()


@scenario
def blit_copy_buffer():
    """CopyFromBufferToBuffer (301) at source and destination offsets, within
    one buffer and between two, and a copy of size 0."""
    r = Rig("blit_copy_buffer", blit_copy_buffer.__doc__)
    a, a_va = r.buffer(contents=ramp(0x100))
    b, b_va = r.buffer()
    r.ready()
    r.blit(B.copy_buffer_to_buffer(a, b, 0x10, 0x40, 0x20), B.copy_buffer_to_buffer(a, a, 0, 0x100, 0x10),
           B.copy_buffer_to_buffer(a, b, 0, 0, 0))
    r.dump(a_va, 0x120, "A")
    r.dump(b_va, 0x80, "B")
    return r.end()


@scenario
def blit_buffer_texture_copies():
    """The copies between buffers and textures, over 4x4 RGBA8 textures on
    buffers with 16-byte rows: CopyFromBufferToTexture (300) puts a 2x2 block
    of the source at texel (1, 1) of T1; CopyFromTextureToBuffer (302) reads
    T1's region (1, 1) 2x2 back into a buffer; CopyFromTextureToTexture (303)
    copies T1's 2x2 at (1, 1) to T2 at (0, 2); the options form (304) copies
    all of T1 into T3; the slices-and-levels form (318) copies T1 into T4
    whole."""
    r = Rig("blit_buffer_texture_copies", blit_buffer_texture_copies.__doc__)
    src, _ = r.buffer(contents=ramp(0x40, 0x40))
    t1b, t1_va = r.buffer()
    t2b, t2_va = r.buffer()
    t3b, t3_va = r.buffer()
    t4b, t4_va = r.buffer()
    out, out_va = r.buffer()
    t1, t2, t3, t4 = (r.texture(b_, 4, 4) for b_ in (t1b, t2b, t3b, t4b))
    r.ready()
    r.blit(B.copy_buffer_to_texture(src, t1, 0x8, 0x10, 0x20, (2, 2, 1), (1, 1, 0)),
           B.copy_texture_to_buffer(t1, out, (1, 1, 0), (2, 2, 1), 0x4, 8, 16),
           B.copy_texture_to_texture(t1, t2, (1, 1, 0), (2, 2, 1), (0, 2, 0)),
           B.copy_texture_to_texture_options(t1, t3, (0, 0, 0), (4, 4, 1), (0, 0, 0)),
           B.copy_slices(t1, t4))
    for label, va in (("T1", t1_va), ("T2", t2_va), ("T3", t3_va), ("T4", t4_va)):
        r.dump(va, 0x40, label)
    r.dump(out_va, 0x18, "out")
    return r.end()


@scenario
def blit_fill_texture():
    """FillTextureWithColor (321) fills a region of a 4x4 RGBA8Unorm texture
    with a colour of four f64 in a given pixel format; FillTextureWithBytes
    (320) repeats bytes read out of a shared buffer at decode time (here 4
    bytes, one RGBA8 texel, at offset 8) over another region."""
    r = Rig("blit_fill_texture", blit_fill_texture.__doc__)
    tb, t_va = r.buffer()
    src, _ = r.buffer(contents=bytes(8) + bytes([0x11, 0x22, 0x33, 0x44]) + bytes(4))
    tex = r.texture(tb, 4, 4)
    r.ready()
    r.blit(B.fill_texture_color(tex, 0, 0, (2, 2, 1), (0, 0, 0), (1.0, 0.5, 0.25, 1.0), PF_RGBA8UNORM),
           B.fill_texture_bytes(tex, 0, 0, (2, 2, 1), (2, 2, 0), src, 8, 4))
    r.dump(t_va, 0x40, "texture")
    return r.end()


@scenario
def blit_texture_maintenance():
    """The texture-maintenance commands on a texture over a buffer succeed
    and leave its texels as they were: GenerateMipmaps (307), Optimize for
    CPU and GPU access (308, 309) and their one-image forms (310, 311),
    SynchronizeResource (314) on a texture, a buffer, ref 0 and an unknown
    ref (the nullable getter gives nil), SynchronizeTexture (315), and
    InvalidateCompressedTexture (322, 323)."""
    r = Rig("blit_texture_maintenance", blit_texture_maintenance.__doc__)
    tb, t_va = r.buffer(contents=ramp(0x40, 0x80))
    tex = r.texture(tb, 4, 4)
    r.ready()
    past = len(r.t.ol.entries)
    r.blit(B.mipmaps(tex), B.optimize(308, tex), B.optimize(309, tex), B.optimize_image(310, tex),
           B.optimize_image(311, tex), B.synchronize(tex), B.synchronize(tb), B.synchronize(0), B.synchronize(past),
           B.synchronize_image(tex), B.invalidate(tex), B.invalidate_image(tex))
    r.dump(t_va, 0x40, "texture")
    return r.end()


@scenario
def blit_fences():
    """UpdateFence (316) and WaitForFence (317) between blit encoders in one
    exec: a fill updates the fence, the next encoder waits for it and copies
    the filled bytes on."""
    r = Rig("blit_fences", blit_fences.__doc__)
    a, _ = r.buffer()
    b, b_va = r.buffer()
    f = r.fence()
    r.ready()
    r.run(seg(abi.ENC_BLIT, B.fill(a, 0, 0x20, 0x66) + B.fence(316, f)),
          seg(abi.ENC_BLIT, B.fence(317, f) + B.copy_buffer_to_buffer(a, b, 0, 0x10, 0x20)))
    r.dump(b_va, 0x40, "B")
    return r.end()


@scenario
def blit_continuation():
    """A blit pass split over segments: the first says continuedByNext and
    keeps its encoder open, the second continues it; then an exec whose
    blit segments each end their own encoder. Both copy in encoded
    order."""
    r = Rig("blit_continuation", blit_continuation.__doc__)
    a, a_va = r.buffer()
    r.ready()
    r.run(seg(abi.ENC_BLIT, B.fill(a, 0, 0x10, 0x11), cont_next=1),
          seg(abi.ENC_BLIT, B.copy_buffer_to_buffer(a, a, 0, 0x10, 0x10), cont_prev=1))
    r.run(seg(abi.ENC_BLIT, B.fill(a, 0x20, 0x10, 0x22)),
          seg(abi.ENC_BLIT, B.copy_buffer_to_buffer(a, a, 0x20, 0x30, 0x10)))
    r.dump(a_va, 0x50, "A")
    return r.end()


@scenario
def blit_framing():
    """What the blit decoder skips: an empty segment, and 1-7 bytes after the
    last record (it loops only while 8 bytes remain); a record longer than
    its command is fine."""
    r = Rig("blit_framing", blit_framing.__doc__)
    a, a_va = r.buffer()
    r.ready()
    r.blit()
    r.blit(B.fill(a, 0, 4, 0x31), b"\x01\x02\x03\x04\x05\x06\x07")
    r.blit(B.fill(a, 4, 4, 0x32)[:4] + struct.pack("<I", 0x28) + B.fill(a, 4, 4, 0x32)[8:] + bytes(8))
    r.dump(a_va, 0x10, "A")
    return r.end()


@scenario
def blit_indirect_command_buffers():
    """The blit commands on indirect command buffers, seen through a compute
    pass that executes them into guest memory: CopyIndirectCommandBuffer
    (305) copies command 0 of ICB A (double_gid into X) to index 1 of ICB B,
    whose own command 0 is empty; executing B [0, 2) writes X. Then
    ResetCommandsInBuffer (313) empties B's range [1, 1), and executing it
    writes nothing into a zeroed X; OptimizeIndirectCommandBuffer (312) on A
    leaves it working."""
    r = Rig("blit_indirect_command_buffers", blit_indirect_command_buffers.__doc__)
    x, x_va = r.buffer()
    pso = r.kernel(icb=True)
    icb_a = r.icb([icb_compute_command(pipeline=pso, buffers=[(x, 0)], grid=(1, 1, 1), tptg=(8, 1, 1)),
                   icb_compute_command(ctype=0)])
    icb_b = r.icb([icb_compute_command(ctype=0), icb_compute_command(ctype=0)])
    r.ready()
    run = C.use_resources([x])
    r.blit(B.copy_icb(icb_a, icb_b, 0, 1, 1))
    r.compute(run, C.execute_icb(icb_b, 0, 2))
    r.dump(x_va, 0x24, "X, after executing B")
    r.run(seg(abi.ENC_BLIT, B.fill(x, 0, 0x20, 0) + B.icb_range(313, icb_b, 1, 1)))
    r.compute(run, C.execute_icb(icb_b, 0, 2))
    r.dump(x_va, 0x24, "X, after executing B reset")
    r.blit(B.icb_range(312, icb_a, 0, 2), B.fill(x, 0, 0x20, 0))
    r.compute(run, C.execute_icb(icb_a, 0, 1))
    r.dump(x_va, 0x24, "X, after executing A optimized")
    return r.end()


@scenario
def blit_texture_slices():
    """The slice fields of the blit records, on 4x4x2 RGBA8 2D-array textures
    over buffers (type 8, textureType 3, created linear with bytesPerImage 0,
    which puts slice 1 at 0x80): CopyFromBufferToTexture into slice 1 of A;
    CopyFromTextureToBuffer out of row 1 of A's slice 1;
    CopyFromTextureToTexture from A's slice 1 to B's slice 0; the
    slices-and-levels form copying both of A's slices into C;
    FillTextureWithColor into texel (3, 3) of C's slice 1 in RGBA8Uint (73)
    with the colour (200, 100, 50, 255); FillTextureWithBytes into B's slice
    1. The diffs show every byte each buffer gained."""
    r = Rig("blit_texture_slices", blit_texture_slices.__doc__)
    src, _ = r.buffer(contents=ramp(0x40, 0x40))
    bytes_, _ = r.buffer(contents=bytes([0xa1, 0xa2, 0xa3, 0xa4]))
    ab, a_va = r.buffer()
    bb, b_va = r.buffer()
    cb, c_va = r.buffer()
    out, out_va = r.buffer()
    a, b, c = (r.texture(x, 4, 4, ttype=3, array=2) for x in (ab, bb, cb))
    r.ready()
    r.blit(B.copy_buffer_to_texture(src, a, 0, 16, 64, (4, 4, 1), (0, 0, 0), slice_=1),
           B.copy_texture_to_buffer(a, out, (0, 1, 0), (4, 1, 1), 0, 16, 16, slice_=1),
           B.copy_texture_to_texture(a, b, (0, 0, 0), (2, 2, 1), (2, 2, 0), ss=1, ds=0),
           B.copy_slices(a, c, slices=2),
           B.fill_texture_color(c, 0, 1, (1, 1, 1), (3, 3, 0), (200.0, 100.0, 50.0, 255.0), PF_RGBA8UINT),
           B.fill_texture_bytes(b, 0, 1, (1, 1, 1), (0, 0, 0), bytes_, 0, 4))
    for label, va in (("A", a_va), ("B", b_va), ("C", c_va), ("out", out_va)):
        r.s.diff(label + ": every byte changed", gpa=r.t.gpa(va), length=0x4000)
    return r.end()


@scenario
def blit_unlisted_buffer():
    """A blit fill into a shared buffer the exec's resource list leaves out.
    In-process the buffer is host memory and the fill lands; in the isolate
    setup the GPU cannot reach the guest's pages, the command buffer fails
    and the exec faults with nothing written (section 6.12). Listed, the
    same fill lands in every setup."""
    r = Rig("blit_unlisted_buffer", blit_unlisted_buffer.__doc__)
    a, a_va = r.buffer(listed=False)
    r.ready()
    r.blit(B.fill(a, 0, 0x10, 0x3c), either=True, label="unlisted")
    r.dump(a_va, 0x20, "A, unlisted")
    r.blit(B.fill(a, 0x10, 0x10, 0x3d), resources=[(a, 0)])
    r.dump(a_va, 0x20, "A, listed")
    return r.end()
