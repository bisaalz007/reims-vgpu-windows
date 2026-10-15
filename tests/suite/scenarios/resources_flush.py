"""rvg_device_finish_suspend flushes every task's resources before tearing the
tasks down (W14; docs/testing.md, sections 4.10, 6.24): each resource with a
valid host copy and a stale guest copy is paged out, CPU-paged ones first, then
blit-paged ones on one blit encoder and compute-paged ones on one compute
encoder of one command buffer. An eviction that throws is logged ("Exception
while evicting task, continuing") and the rest go on."""
from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.resources_build import (BGRA8Unorm, Depth32Float_Stencil8, RGBA8Unorm, XR10_2P_A8, RTask, b_copy_buf_tex,
                                    b_copy_tex_tex, b_fill_buf, b_fill_tex_color, buffer_rec, cmd_invalidate, desc1,
                                    dimension, heap_buffer_rec, heap_rec, mip, ser_view_format, simple_dim, texture_rec)
from pgtest.scenario import Scenario

PATTERN = bytes((i * 7 + 3) & 0xFF for i in range(0x100))
FULL = (4, 4, 1)
# The colour and alpha planes of the 550 texture a copy reads, unlike the
# bytes of the texture it writes.
XR10_SRC_COLOR = bytes((i * 53 + 29) & 0xFF for i in range(0x400))
XR10_SRC_ALPHA = bytes((i * 19 + 0x55) & 0xFF for i in range(0x400))


def _suspend(s):
    s.api("willSuspend")
    s.api("finishSuspend", slot="s")


@scenario
def resources_flush_on_suspend():
    """A private buffer, a normal texture, a two-plane Depth32Float_Stencil8
    texture and a private heap, each written on the GPU by an exec that lists
    it with invalidateGuest only (flag 2), so their guest pages are stale; a
    second private buffer left clean. rvg_device_finish_suspend pages the dirty
    ones out: their writes are in guest memory after it, the clean buffer's
    pages are untouched. A view marked host-valid and guest-invalid by 52 is
    flushed too, and its synchronize throws ("Attempting to synchronize a view
    resource"), which is logged, and the others are still flushed. In the
    service the buffers and heap are detach resources over guest memory: their
    writes were there at once."""
    from scenarios.textures_dual_plane import DEPTH, STENCIL, ds_dims
    s = Scenario("resources_flush_on_suspend", doc=resources_flush_on_suspend.__doc__)
    s.bring_up()
    t = RTask(s, pages=96)
    p_va = t.area(PAGE, contents=PATTERN)
    p = t.add(abi.T_BUFFER, buffer_rec(PAGE, p_va // PAGE, 1))
    c_va = t.area(PAGE, contents=PATTERN)
    c = t.add(abi.T_BUFFER, buffer_rec(PAGE, c_va // PAGE, 1))
    x_va = t.area(PAGE, contents=PATTERN[:64])
    x = t.add(abi.T_TEXTURE, texture_rec(PAGE, x_va // PAGE, [simple_dim(4, 4)], desc1(usage=0x13)))
    v = t.add(abi.T_SERIALIZER_TEXTURE, ser_view_format(x, BGRA8Unorm))
    d_va = t.area(PAGE, contents=DEPTH)
    t.write(d_va + 0x100, STENCIL)
    d = t.add(abi.T_DUAL_PLANE_TEXTURE, texture_rec(PAGE, d_va // PAGE, ds_dims(),
                                                    desc1(fmt=Depth32Float_Stencil8, usage=3)))
    h_va = t.area(4 * PAGE, contents=PATTERN)
    h = t.add(abi.T_HEAP, heap_rec(h_va // PAGE, 4 * PAGE, 1))
    hb = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(h, 0x100, PAGE))
    b, b_va = t.shared(PAGE)
    t.write(b_va, bytes(range(0xE0, 0xF0)))
    t.define()
    t.install()
    t.blit(b_fill_buf(p, 0x10, 0x10, 0xA1), b_fill_tex_color(x, (2, 2, 1), (0, 1.0, 0, 1.0), RGBA8Unorm),
           b_copy_buf_tex(b, d, 0, 4, 16, FULL, options=2), b_fill_buf(hb, 0, 0x10, 0xB2),
           resources=[(p, 2), (c, 0), (x, 2), (d, 2), (h, 2), (hb, 0), (b, 0)],
           label="write four resources, invalidateGuest only")
    cmd_invalidate(s, 1, [(v, 0, 1, 1, 0)], label="52: the view host-valid, guest-invalid")
    t.dump(p_va, 0x20, "private buffer before")
    t.dump(x_va, 0x40, "texture before")
    t.dump(d_va + 0x100, 0x10, "stencil plane before")
    t.dump(h_va + 0x100, 0x10, "heap at 0x100 before")
    _suspend(s)
    t.dump(p_va, 0x20, "private buffer after finishSuspend")
    t.dump(c_va, 0x20, "clean private buffer after finishSuspend")
    t.dump(x_va, 0x40, "texture after finishSuspend")
    t.dump(d_va + 0x100, 0x10, "stencil plane after finishSuspend")
    t.dump(h_va + 0x100, 0x10, "heap at 0x100 after finishSuspend")
    s.events("end")
    return s


@scenario(host=True)
def textures_flush_xr10_on_suspend():
    """Two type 12 textures of pixel format 550, compute-paged: one written
    by an exec with invalidateGuest only is paged out by
    rvg_device_finish_suspend on the compute encoder; the other, whose pages
    are past the task's end, is marked host-valid and guest-invalid by 52
    without ever being paged in, and its eviction throws (the task has no
    pages there), which is logged.

    The exec writes the first texture twice on the GPU. 303 copies texels
    8-15 of row 1 of a third 550 texture, listed with flags 0 and paged in
    from other guest bytes, into texels 4-11 of row 0, inside the dumps. 321
    fills texels 60-63 of rows 2 and 3, outside them: Apple's fill of a 550
    texture writes bytes that change with every host boot (docs/lessons.md).
    The colour plane before the suspend holds the guest's bytes untouched.
    After it, the dumps (the first 16 texels of row 0 of each plane) hold
    what BlitOutRGB_2P_XR10_A8 wrote from the texture, so they show the
    page-out ran after the copy, from the texture it wrote. In the colour
    plane, texels 0-3 and 12-15 are the texture's own guest bytes as
    BlitInRGB_2P_XR10_A8 paged them in, and texels 4-11 the source's, each
    with its top two bits set. The alpha plane reads zero: the texture's own
    alpha plane is zero, and the copied texels' alpha reads zero too, not
    the source's. Host-keyed: the library's shaders' arithmetic."""
    s = Scenario("textures_flush_xr10_on_suspend", host=True, doc=textures_flush_xr10_on_suspend.__doc__)
    s.bring_up()
    t = RTask(s, pages=96)
    w, h = 64, 4
    dims = [dimension([mip(0, w * 4 * h, w * 4, w, h)], block=(1, 1, 1, 4)),
            dimension([mip(0, w * 4 * h, w * 4, w, h)], base=PAGE, block=(1, 1, 1, 4))]
    xd = desc1(fmt=XR10_2P_A8, w=w, h=h, usage=3)
    x_va = t.area(2 * PAGE, contents=bytes((i * 37 + 11) & 0xFF for i in range(w * 4 * h)))
    x = t.add(abi.T_DUAL_PLANE_TEXTURE, texture_rec(2 * PAGE, x_va // PAGE, dims, xd))
    y_va = t.area(2 * PAGE, contents=XR10_SRC_COLOR)
    t.write(y_va + PAGE, XR10_SRC_ALPHA)
    y = t.add(abi.T_DUAL_PLANE_TEXTURE, texture_rec(2 * PAGE, y_va // PAGE, dims, xd))
    far = t.add(abi.T_DUAL_PLANE_TEXTURE, texture_rec(2 * PAGE, 500, dims, xd))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_tex_color(x, (4, 2, 1), (1.0, 0.5, 0.25, 1.0), XR10_2P_A8, origin=(60, 2, 0)),
           b_copy_tex_tex(y, x, (8, 1, 1), sorigin=(8, 1, 0), dorigin=(4, 0, 0)),
           resources=[(x, 2), (y, 0), (b, 0)], label="write the 550 texture, invalidateGuest only")
    t.blit(b_fill_buf(b, 0, 0x10, 0x01), b_fill_tex_color(far, (1, 1, 1), (1.0, 0, 0, 1.0), XR10_2P_A8),
           resources=[(b, 0)], label="make the far texture without paging it")
    cmd_invalidate(s, 1, [(far, 0, 1, 1, 0)], label="52: the far texture host-valid, guest-invalid")
    t.dump(x_va, 0x40, "color plane before")
    _suspend(s)
    t.dump(x_va, 0x40, "color plane after finishSuspend")
    t.dump(x_va + PAGE, 0x40, "alpha plane after finishSuspend")
    s.events("end")
    return s
