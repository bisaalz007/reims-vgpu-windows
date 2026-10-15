"""Two-plane textures: object type 12 (feature dualPlaneTextures) and type
18 shared backings (docs/testing.md, sections 4.10, 6.19).

A type 12 record carries two dimension blocks back to back, each followed by
its mips. The device's pixel-format table describes Depth32Float_Stencil8 as
two planes, and it pages by blits, a plane at a time; pixel formats 550 and
551 page through the BlitIn/BlitOutRGB_2P_XR10_A8 compute kernels of the
library's shaders."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.resources_build import (Depth32Float_Stencil8, XR10_2P_A8, XR10_2P_A8_SRGB, TT_2DArray, TT_Cube, TT_2DMS,
                                    RTask, b_copy_buf_tex, b_copy_tex_buf, b_fill_buf, desc1, dimension, mip,
                                    shared_backing_rec, texture_rec)
from pgtest.scenario import Scenario

T12 = abi.T_DUAL_PLANE_TEXTURE
DEPTH = struct.pack("<16f", *[i / 16.0 for i in range(16)])
STENCIL = bytes(range(0x30, 0x40))


def ds_dims(levels=1, s8=1, w=4, h=4):
    """Depth32Float_Stencil8: plane 0, 4 bytes a texel at +0; plane 1, s8
    bytes a texel at +0x100 (baseOffset), each level after the last."""
    def plane(bpp, base):
        mips, off, ww, hh = [], 0, w, h
        for _ in range(levels):
            mips.append(mip(off, ww * hh * bpp, ww * bpp, ww, hh))
            off += ww * hh * bpp
            ww, hh = max(ww // 2, 1), max(hh // 2, 1)
        return dimension(mips, base=base, block=(1, 1, 1, bpp))
    return [plane(4, 0), plane(s8, 0x100)]


@scenario
def textures_dual_plane_depth_stencil():
    """A 4x4 Depth32Float_Stencil8 texture (type 12) over guest bytes: depth
    floats at +0, stencil bytes at +0x100. Listing it pages both planes in
    (copyFromBuffer with DepthFromDepthStencil, then StencilFromDepthStencil
    options); 302 with the same options reads each plane back into a shared
    buffer. 300 writes new stencil bytes, and flags 6 page both planes back
    out to the guest. A two-level texture pages both levels of both planes."""
    s = Scenario("textures_dual_plane_depth_stencil", doc=textures_dual_plane_depth_stencil.__doc__)
    s.bring_up()
    t = RTask(s)
    va = t.area(PAGE, contents=DEPTH)
    t.write(va + 0x100, STENCIL)
    d = t.add(T12, texture_rec(PAGE, va // PAGE, ds_dims(), desc1(fmt=Depth32Float_Stencil8, usage=3)))
    va2 = t.area(PAGE, contents=DEPTH + DEPTH[:16])
    t.write(va2 + 0x100, STENCIL + STENCIL[:4])
    d2 = t.add(T12, texture_rec(PAGE, va2 // PAGE, ds_dims(levels=2),
                                desc1(fmt=Depth32Float_Stencil8, usage=3, w=4, h=4, mips=2)))
    b, b_va = t.shared(PAGE)
    t.write(b_va + 0x200, bytes(range(0x80, 0x90)))
    t.define()
    t.install()
    full = (4, 4, 1)
    t.blit(b_copy_tex_buf(d, b, 0, 16, 64, full, options=1), b_copy_tex_buf(d, b, 0x40, 4, 16, full, options=2),
           b_copy_tex_buf(d2, b, 0x80, 8, 16, (2, 2, 1), level=1, options=1),
           b_copy_tex_buf(d2, b, 0xC0, 2, 4, (2, 2, 1), level=1, options=2),
           resources=[(d, 0), (d2, 0), (b, 0)], label="read both planes")
    t.dump(b_va, 0xD0, "B")
    t.blit(b_copy_buf_tex(b, d, 0x200, 4, 16, full, options=2), resources=[(d, 6), (b, 0)],
           label="write the stencil plane, synchronize")
    t.dump(va, 0x40, "depth plane pages")
    t.dump(va + 0x100, 0x10, "stencil plane pages")
    t.delete()
    s.events("end")
    return s


def _xr10_rec(t, fmt, w=64, h=4, contents=None, dims=None, desc=None):
    va = t.area(2 * PAGE, contents=contents)
    if dims is None:
        dims = [dimension([mip(0, w * 4 * h, w * 4, w, h)], block=(1, 1, 1, 4)),
                dimension([mip(0, w * 4 * h, w * 4, w, h)], base=PAGE, block=(1, 1, 1, 4))]
    return t.add(T12, texture_rec(2 * PAGE, va // PAGE, dims, desc or desc1(fmt=fmt, w=w, h=h, usage=3))), va


@scenario
def textures_dual_plane_xr10():
    """Pixel formats 550 and 551 (not public), 64x4, as type 12 textures: a
    color plane at +0 and an alpha plane at +0x4000, each 4 bytes a texel as
    the device's pixel-format table describes them. Listing one pages it in
    on a compute encoder through BlitInRGB_2P_XR10_A8 and flags 6 page it
    back out through BlitOutRGB_2P_XR10_A8, over plane textures made on the
    transfer buffer. The round trip's bytes are the library's shaders'
    arithmetic: host-keyed."""
    s = Scenario("textures_dual_plane_xr10", host=True, doc=textures_dual_plane_xr10.__doc__)
    s.bring_up()
    t = RTask(s, pages=96)
    data = bytes((i * 37 + 11) & 0xFF for i in range(64 * 4 * 4))
    x0, x0_va = _xr10_rec(t, XR10_2P_A8, contents=data)
    t.write(x0_va + PAGE, data[::-1])
    x1, x1_va = _xr10_rec(t, XR10_2P_A8_SRGB, contents=data)
    t.write(x1_va + PAGE, data[::-1])
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_buf(b, 0, 0x10, 0x01), resources=[(x0, 6), (x1, 6), (b, 0)], label="page in and out")
    for va, name in ((x0_va, "550"), (x1_va, "551")):
        t.dump(va, 0x400, name + " color plane")
        t.dump(va + PAGE, 0x400, name + " alpha plane")
    t.dump(b_va, 0x10, "B")
    t.delete()
    s.events("end")
    return s


@scenario
def textures_dual_plane_records():
    """Two-plane refusals, each its own object listed in its own exec: a
    one-plane format as type 12 ("Dual plane mismatch"); a stencil plane
    whose texel size is not 1 at this version (the stencil texel size the
    device expects is 1 from version 62); a short second dimension; 550 with
    two levels ("Multi plane mipped textures are unsupported"), as a cube
    (Metal makes none), as an array ("Multi plane cube, array, compressed and
    multi-sample textures are unsupported"); a type 18 of 550 (no shared
    texture resource is made for it). The good records close it."""
    from scenarios.resources import creation_case
    ok_desc = desc1(fmt=Depth32Float_Stencil8, usage=3)
    xr = [dimension([mip(0, 256 * 4, 256, 64, 4)], block=(1, 1, 1, 4)),
          dimension([mip(0, 256 * 4, 256, 64, 4)], base=PAGE, block=(1, 1, 1, 4))]
    xr2 = [dimension([mip(0, 256 * 4, 256, 64, 4), mip(0x400, 128 * 2, 128, 32, 2)], block=(1, 1, 1, 4)),
           dimension([mip(0, 256 * 4, 256, 64, 4), mip(0x400, 128 * 2, 128, 32, 2)], base=PAGE, block=(1, 1, 1, 4))]
    rec = lambda dims, d, length=2 * PAGE: texture_rec(length, 8, dims, d)
    cases = [
        (T12, rec([dimension([mip(0, 64, 16, 4, 4)]), dimension([mip(0, 64, 16, 4, 4)], base=0x100)], desc1(usage=3)),
         None, True),
        (T12, rec(ds_dims(s8=4), ok_desc), None, True),
        (T12, rec(ds_dims(), ok_desc), 12 + 0x3c + 0x17, True),
        (T12, rec(xr2, desc1(fmt=XR10_2P_A8, w=64, h=4, mips=2, usage=3)), None, True),
        (T12, rec([dimension([mip(0, 1024, 256, 64, 64)], cube=1, slice_bytes=0x400, block=(1, 1, 1, 4)),
                   dimension([mip(0, 1024, 256, 64, 64)], cube=1, slice_bytes=0x400, base=0x1800, block=(1, 1, 1, 4))],
                  desc1(fmt=XR10_2P_A8, w=64, h=64, tt=TT_Cube, usage=3), length=4 * PAGE), None, True),
        (T12, rec([dimension([mip(0, 1024, 256, 64, 4)], array=2, slice_bytes=0x400, block=(1, 1, 1, 4)),
                   dimension([mip(0, 1024, 256, 64, 4)], array=2, slice_bytes=0x400, base=PAGE, block=(1, 1, 1, 4))],
                  desc1(fmt=XR10_2P_A8, w=64, h=4, tt=TT_2DArray, array=2, usage=3)), None, True),
        (T12, rec(ds_dims(), ok_desc), None, False),
        (T12, rec(xr, desc1(fmt=XR10_2P_A8, w=64, h=4, usage=3)), None, False),
    ]
    return creation_case("textures_dual_plane_records", textures_dual_plane_records.__doc__, cases, host=True, pages=96)


@scenario
def textures_dual_plane_shared_xr10():
    """A type 18 shared backing of pixel format 550 in task 0, reached from a
    type 16 texture in task 1: no shared texture resource is made for 550 and
    551, so the backing is not made and the handle is not found. Host-keyed:
    whether Metal makes a shareable 550 texture is the host's."""
    from scenarios.textures_shared import kernel_and_user, two_task_probe
    from pgtest.resources_build import shared_texture_rec
    s = Scenario("textures_dual_plane_shared_xr10", host=True, doc=textures_dual_plane_shared_xr10.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    xr = [dimension([mip(0, 256 * 4, 256, 64, 4)], block=(1, 1, 1, 4)),
          dimension([mip(0, 256 * 4, 256, 64, 4)], base=PAGE, block=(1, 1, 1, 4))]
    kb = k.add(abi.T_SHARED_BACKING2, shared_backing_rec(2 * PAGE, 8, xr, desc1(fmt=XR10_2P_A8, w=64, h=4, usage=3)))
    kd = k.add(abi.T_SHARED_BACKING2, shared_backing_rec(PAGE, 10, ds_dims(levels=2),
                                                         desc1(fmt=Depth32Float_Stencil8, usage=3, mips=2)))
    cases = [(t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kb)), True),
             (t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kd)), False)]
    return two_task_probe(s, k, t, cases)


@scenario
def textures_dual_plane_xr10_bad_row():
    """A 63x4 texture of pixel format 550: its plane textures over the
    transfer buffer have rows of 252 bytes, which Metal refuses by aborting,
    not by returning nil. The abort is inside Metal, so the host dies in
    every setup, an isolated task's included. This is the scenario's last
    command."""
    s = Scenario("textures_dual_plane_xr10_bad_row", host=True, doc=textures_dual_plane_xr10_bad_row.__doc__)
    s.bring_up()
    t = RTask(s)
    odd, _ = _xr10_rec(t, XR10_2P_A8, w=63)
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_buf(b, 0, 0x10, 0x02), resources=[(odd, 0), (b, 0)], fault=True, label="63 texels a row")
    s.events("end")
    return s


@scenario
def textures_dual_plane_xr10_multisample():
    """A multisampled (2DMS, 2 samples) texture of pixel format 550: Metal
    aborts making it, so the host dies in every setup, an isolated task's
    included. The last command."""
    s = Scenario("textures_dual_plane_xr10_multisample", host=True, doc=textures_dual_plane_xr10_multisample.__doc__)
    s.bring_up()
    t = RTask(s)
    ms, _ = _xr10_rec(t, XR10_2P_A8, dims=[dimension([mip(0, 2048, 512, 64, 4)], block=(1, 1, 1, 4)),
                                            dimension([mip(0, 2048, 512, 64, 4)], base=PAGE, block=(1, 1, 1, 4))],
                      desc=desc1(fmt=XR10_2P_A8, w=64, h=4, tt=TT_2DMS, samples=2, usage=4))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_buf(b, 0, 0x10, 0x02), resources=[(ms, 0), (b, 0)], fault=True, label="a multisampled 550")
    s.events("end")
    return s
