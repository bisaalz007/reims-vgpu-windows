"""Texture paths the per-area scenarios leave: two-plane compute textures
prepared, synchronized and discarded twice; shared texture backings
prepared again after a discard and synchronized twice; shared textures
at a version with no two-plane textures; and two cases once thought to
hang, which complete (docs/testing.md, sections 4.10, 6.19, 6.20)."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.resources_build import (BGRA8Unorm, Depth32Float_Stencil8, OPT_SHARED, RGBA16Float, RGBA8Unorm,
                                    XR10_2P_A8, RTask, b_copy_buf_tex, b_copy_tex_buf, b_fill_buf, b_fill_tex_color,
                                    b_sync_resource, cmd_discard, cmd_invalidate, cmd_synchronize, desc1, dimension, iosurface_op,
                                    mapper_ref_rec, mip, shared_backing_rec, shared_texture_rec, simple_dim,
                                    texture_rec)
from pgtest.scenario import Scenario
from pgtest.task import seg

T12 = abi.T_DUAL_PLANE_TEXTURE
PATTERN = bytes((i * 7 + 3) & 0xFF for i in range(0x100))
FULL = (4, 4, 1)


def _xr10_dims(w=64, h=4, w1=None):
    """Two planes of 4 bytes a texel, color at +0 and alpha at +0x4000; w1
    overrides the alpha plane's mip width."""
    return [dimension([mip(0, w * 4 * h, w * 4, w, h)], block=(1, 1, 1, 4)),
            dimension([mip(0, w * 4 * h, w * 4, w if w1 is None else w1, h)], base=PAGE, block=(1, 1, 1, 4))]


@scenario
def textures_gap_xr10_twice():
    """A 64x4 texture of pixel format 550 (compute-paged) listed twice in one
    exec: the first entry pages it in on the compute encoder, the second
    finds it host-valid. Listed twice with invalidateGuest + synchronize
    (flags 6), it is synchronized once and the second finds its guest copy
    valid; so again through 53 naming it twice after an exec with flags 2.
    With neither copy marked valid by 52, the next exec prepares it anyway,
    logging "Trying to prepare from invalid guest resource". 63 discards it,
    releasing its two plane textures and its transfer buffer, and the next
    exec pages it in again, making both anew. A 550 texture whose alpha
    plane's mip is 32 texels wide fails the texture dimension check ("Size
    mismatch") and its exec faults. Only the probe buffer is observed: the
    paged bytes are the library's shaders' arithmetic
    (textures_dual_plane_xr10)."""
    s = Scenario("textures_gap_xr10_twice", doc=textures_gap_xr10_twice.__doc__)
    s.bring_up()
    t = RTask(s, pages=96)
    va = t.area(2 * PAGE, contents=PATTERN * 4)
    x = t.add(T12, texture_rec(2 * PAGE, va // PAGE, _xr10_dims(), desc1(fmt=XR10_2P_A8, w=64, h=4, usage=3)))
    bad = t.add(T12, texture_rec(2 * PAGE, va // PAGE, _xr10_dims(w1=32), desc1(fmt=XR10_2P_A8, w=64, h=4, usage=3)))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_buf(b, 0x00, 0x10, 0x01), resources=[(x, 0), (x, 0), (b, 0)], label="listed twice")
    t.blit(b_fill_tex_color(x, (4, 2, 1), (1.0, 0.5, 0.25, 1.0), XR10_2P_A8), b_fill_buf(b, 0x10, 0x10, 0x02),
           resources=[(x, 6), (x, 6), (b, 0)], label="written, listed twice with flags 6")
    t.blit(b_fill_tex_color(x, (4, 2, 1), (0.25, 0.5, 1.0, 1.0), XR10_2P_A8), b_fill_buf(b, 0x20, 0x10, 0x03),
           resources=[(x, 2), (b, 0)], label="written, invalidateGuest only")
    cmd_synchronize(s, 1, [x, x], label="53 naming it twice")
    cmd_invalidate(s, 1, [(x, 1, 0, 1, 0)], label="52: neither copy valid")
    t.blit(b_fill_buf(b, 0x50, 0x10, 0x06), resources=[(x, 0), (b, 0)], label="prepared from an invalid guest copy")
    cmd_discard(s, 1, [x], label="63 on it")
    t.blit(b_fill_buf(b, 0x30, 0x10, 0x04), resources=[(x, 0), (b, 0)], label="paged in again after 63")
    t.blit(b_fill_buf(b, 0x40, 0x10, 0x05), resources=[(bad, 0), (b, 0)], fault=True,
           label="an alpha plane 32 texels wide")
    t.dump(b_va, 0x60, "probe buffer: which execs ran")
    t.delete()
    s.events("end")
    return s


@scenario
def textures_gap_shared_backing_twice():
    """A one-plane (type 17) and a two-plane Depth32Float_Stencil8 (type 18)
    shared texture backing in task 0, used from task 1 through type 16
    textures. Asking for a handle prepares a backing that needs it. 63 on
    task 0 discards both backings, which leaves them host-valid but volatile,
    so the next handle prepares each again: the devolatilize finds its
    contents kept and the prepare returns early. A fill through the type 16
    texture with invalidateGuest only marks the backing's guest copy invalid
    through task 0; 53 on task 0 naming each backing twice pages each out
    once, the second finding its guest copy valid. Then 52 on task 0 marks
    neither copy of either backing valid, and new type 16 textures prepare
    both from their guest pages anyway, logging "Trying to prepare from
    invalid guest resource": the pages as 53 left them."""
    from scenarios.textures_dual_plane import DEPTH, STENCIL, ds_dims
    from scenarios.textures_shared import kernel_and_user, define_both
    s = Scenario("textures_gap_shared_backing_twice", doc=textures_gap_shared_backing_twice.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    k1_va = k.area(PAGE, contents=PATTERN[:64])
    k1 = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, k1_va // PAGE, [simple_dim(4, 4)],
                                                         desc1(options=0x20, usage=3)))
    k2_va = k.area(PAGE, contents=DEPTH)
    k.write(k2_va + 0x100, STENCIL)
    k2 = k.add(abi.T_SHARED_BACKING2, shared_backing_rec(PAGE, k2_va // PAGE, ds_dims(),
                                                         desc1(fmt=Depth32Float_Stencil8, usage=3)))
    s1 = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(k1))
    s2 = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(k2))
    s1b = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(k1))
    s2b = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(k2))
    s1c = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(k1))
    s2c = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(k2))
    b, b_va = t.shared(PAGE)
    t.write(b_va + 0x800, bytes(range(0x50, 0x60)))
    define_both(k, t)
    t.blit(b_copy_tex_buf(s1, b, 0, 16, 64, FULL), b_copy_tex_buf(s2, b, 0x40, 4, 16, FULL, options=2),
           resources=[(s1, 0), (s2, 0), (b, 0)], label="read both backings")
    t.dump(b_va, 0x50, "B")
    cmd_discard(s, 0, [k1, k2], label="63 on task 0, on both backings")
    k.write(k1_va, b"\x11" * 16)
    t.blit(b_copy_tex_buf(s1b, b, 0x80, 16, 64, FULL), b_copy_tex_buf(s2b, b, 0xC0, 4, 16, FULL, options=2),
           resources=[(s1b, 0), (s2b, 0), (b, 0)], label="new textures: each backing prepared again")
    t.dump(b_va + 0x80, 0x50, "B+0x80: the kept contents")
    t.blit(b_fill_tex_color(s1, (2, 2, 1), (1.0, 0, 0, 1.0), RGBA8Unorm),
           b_copy_buf_tex(b, s2, 0x800, 4, 16, FULL, options=2),
           resources=[(s1, 2), (s2, 2), (b, 0)], label="write through both, invalidateGuest only")
    k.dump(k1_va, 0x40, "backing 17's pages before 53")
    cmd_synchronize(s, 0, [k1, k1, k2, k2], label="53 on task 0 naming each backing twice")
    k.dump(k1_va, 0x40, "backing 17's pages after 53")
    k.dump(k2_va + 0x100, 0x10, "backing 18's stencil pages after 53")
    cmd_invalidate(s, 0, [(k1, 1, 0, 1, 0), (k2, 1, 0, 1, 0)], label="52 on task 0: neither copy of either valid")
    t.blit(b_copy_tex_buf(s1c, b, 0x100, 16, 64, FULL), b_copy_tex_buf(s2c, b, 0x140, 4, 16, FULL, options=2),
           resources=[(s1c, 0), (s2c, 0), (b, 0)], label="new textures: each backing prepared from invalid copies")
    t.dump(b_va + 0x100, 0x50, "B+0x100: the guest pages paged in again")
    t.delete()
    k.delete()
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def textures_gap_backing_twice():
    """An IOSurface backing (type 4) in task 0 and a texture over it in task
    1 (type 5). A fill through the texture with invalidateGuest only marks
    the backing's guest copy invalid (the ref texture forwards validity to
    it); 53 on task 0 naming the backing twice pages it out once, and the
    second finds its guest copy valid. The ref texture listed twice prepares
    the backing once. In-process only: a type 5 texture aborts the service
    (textures_backing_ref_service)."""
    from scenarios.textures_shared import BGRA, kernel_and_user, define_both
    from pgtest.resources_build import backing_rec, ref_texture_rec
    s = Scenario("textures_gap_backing_twice", setups=("blocks", "map"), doc=textures_gap_backing_twice.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    bk_va = k.area(PAGE, contents=bytes(0x100) + PATTERN[:64])
    bk = k.add(abi.T_IOSURFACE_BACKING, backing_rec(PAGE, bk_va // PAGE, BGRA, [(0x100, 4, 4, 16, 4)]))
    r = t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=iosurface_op(12, desc1(fmt=BGRA8Unorm, options=OPT_SHARED,
                                                                                    usage=3))))
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    t.blit(b_copy_tex_buf(r, b, 0, 16, 64, FULL), resources=[(r, 0), (r, 0), (b, 0)], label="listed twice")
    t.dump(b_va, 0x40, "B")
    t.blit(b_fill_tex_color(r, (2, 2, 1), (0, 1.0, 0, 1.0), BGRA8Unorm), resources=[(r, 2), (b, 0)],
           label="fill, invalidateGuest only")
    k.dump(bk_va + 0x100, 0x40, "the backing's pages before 53")
    cmd_synchronize(s, 0, [bk, bk], label="53 on task 0 naming the backing twice")
    k.dump(bk_va + 0x100, 0x40, "the backing's pages after 53")
    t.delete()
    k.delete()
    s.events("end")
    return s


@scenario
def textures_gap_v6_shared_dual_plane():
    """At version 6 shared textures are supported and two-plane textures are
    not (supportsSharedTextures from 5, dualPlaneTextures from 31): a type 16
    texture over a type 18 backing asks task 0 for the backing, which is
    refused ("Attempting to use unsupported dual-plane textures"), and its
    exec faults; one over a type 17 backing is made. Execs are opcode 55 (67
    comes at 109)."""
    from scenarios.textures_dual_plane import ds_dims
    from scenarios.textures_shared import kernel_and_user, two_task_probe
    s = Scenario("textures_gap_v6_shared_dual_plane", doc=textures_gap_v6_shared_dual_plane.__doc__)
    s.bring_up(version=6)
    s.pg_version = 6
    k, t = kernel_and_user(s)
    k17 = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, 8, [simple_dim(4, 4)], desc1(usage=3)))
    k18 = k.add(abi.T_SHARED_BACKING2, shared_backing_rec(PAGE, 9, ds_dims(s8=4),
                                                          desc1(fmt=Depth32Float_Stencil8, usage=3)))
    cases = [(t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(k18)), True),
             (t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(k17)), False)]
    return two_task_probe(s, k, t, cases)


@scenario(host=True)
def textures_gap_pvrtc_shared_backing():
    """A 16x16 PVRTC_RGBA_4BPP shared texture backing (type 17, compressed
    4x4 blocks of 8 bytes) in task 0, used from task 1: a PVRTC backing pages
    with MTLBlitOptionRowLinearPVRTC. Whether Metal makes a shareable PVRTC
    texture is the host's: host-keyed. It does, and a copy out through the
    type 16 texture shows the guest's blocks."""
    from scenarios.textures_shared import kernel_and_user, define_both
    s = Scenario("textures_gap_pvrtc_shared_backing", host=True, doc=textures_gap_pvrtc_shared_backing.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    k_va = k.area(PAGE, contents=PATTERN[:128])
    dim = dimension([mip(0, 128, 32, 16, 16)], block=(4, 4, 1, 8), compressed=1)
    kb = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, k_va // PAGE, [dim],
                                                         desc1(fmt=162, w=16, h=16, options=0x20, usage=1)))
    st = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kb))
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    t.blit(b_copy_tex_buf(st, b, 0, 32, 128, (16, 16, 1)), b_fill_buf(b, 0x100, 0x10, 0x01),
           resources=[(st, 0), (b, 0)], label="copy out of the PVRTC shared texture")
    t.dump(b_va, 0x110, "B")
    t.delete()
    k.delete()
    s.events("end")
    return s


@scenario(host=True)
def textures_gap_heap_gpu_optimized():
    """Whether a texture fits in a heap depends on its
    allowGPUOptimizedContents bit (the descriptor's bit 6, which the texture
    descriptor passes to Metal). Opcode 64 answers heapTextureSizeAndAlign
    for a 64x64 RGBA8Unorm shader-read texture without and with the bit, and
    the test asks Metal the same (allow_gpu_optimized): the replies must
    equal Metal's. Then a one-page heap takes the texture without the bit at
    offset 0 (it needs one page) and refuses it with the bit (it needs more):
    Metal makes no texture, "Failed to create texture", and the exec faults.
    A two-page heap takes the texture with the bit. The same through
    commandType 56 (a v2 descriptor, whose bit 6 of byte 0 is the same flag).
    Host-keyed: the sizes are the host's."""
    from pgtest.resources_build import heap_rec, ser_in_heap
    from pgtest.tasks_lib import heap_payload, heap_ser
    s = Scenario("textures_gap_heap_gpu_optimized", host=True, doc=textures_gap_heap_gpu_optimized.__doc__)
    s.bring_up()
    t = RTask(s, pages=96)
    d0 = desc1(fmt=RGBA8Unorm, w=64, h=64, usage=1)
    d1 = desc1(fmt=RGBA8Unorm, w=64, h=64, usage=1, gpuopt=1)
    h1_va = t.area(PAGE, contents=PATTERN * 64)
    h1 = t.add(abi.T_HEAP, heap_rec(h1_va // PAGE, PAGE, 1))
    h2_va = t.area(2 * PAGE, contents=PATTERN * 128)
    h2 = t.add(abi.T_HEAP, heap_rec(h2_va // PAGE, 2 * PAGE, 1))
    x0 = t.add(abi.T_SERIALIZER_TEXTURE, ser_in_heap(h1, 0, d0))
    x1 = t.add(abi.T_SERIALIZER_TEXTURE, ser_in_heap(h1, 0, d1))
    x2 = t.add(abi.T_SERIALIZER_TEXTURE, ser_in_heap(h2, 0, d1))
    from pgtest.resources_build import desc2
    e0 = desc2(fmt=RGBA8Unorm, w=64, h=64, usage=1)
    e1 = desc2(fmt=RGBA8Unorm, w=64, h=64, usage=1, gpuopt=1)
    y0 = t.add(abi.T_SERIALIZER_TEXTURE, ser_in_heap(h1, 0, e0, v2=True))
    y1 = t.add(abi.T_SERIALIZER_TEXTURE, ser_in_heap(h1, 0, e1, v2=True))
    reply_va = t.area(PAGE)
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    for g, d in ((0, d0), (1, d1)):
        s.submit(abi.OP_HEAP_TEXTURE_SIZE_AND_ALIGN, heap_payload(1, reply_va + 0x10 * g, 16, heap_ser(d)),
                 label="64, allowGPUOptimizedContents %d" % g)
        s.metal("heapTextureSizeAndAlign", format=RGBA8Unorm, w=64, h=64, usage=1, options=0x20,
                allow_gpu_optimized=bool(g), label="Metal, allowGPUOptimizedContents %d" % g)
    t.dump(reply_va, 0x20, "64's replies {size, align}")
    for i, (x, h, faults, label) in enumerate(((x0, h1, False, "one page, without the bit"),
                                               (x1, h1, True, "one page, with the bit"),
                                               (x2, h2, False, "two pages, with the bit"),
                                               (y0, h1, False, "one page, v2 descriptor without the bit"),
                                               (y1, h1, True, "one page, v2 descriptor with the bit"))):
        t.blit(b_fill_buf(b, 0x10 * i, 0x10, i + 1), b_copy_tex_buf(x, b, 0x100 + 0x40 * i, 256, 1024, (1, 1, 1)),
               resources=[(h, 0), (b, 0)], fault=faults, label=label)
    t.dump(b_va, 0x50, "probe buffer: which execs ran")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def textures_gap_misplaced_sync():
    """Each placeable texture kind (textures_placement_forced's) with a
    placement entry {resourceIndex 0x40 + i, gpuResourceID 0x1234}, whose
    creation therefore fails ("Failed to correctly place texture"), named by
    314 SynchronizeResource in an exec that does not list it. 314 takes a
    nullable resource: the failed creation gives nil, and every exec
    completes, its probe slot written, with no fault. (A wait for a fault
    here times out, which once read as a hang: it is not one.) The entries
    keep the guest's values."""
    from scenarios.textures_placement import _placed_objects, install_with_placements
    s = Scenario("textures_gap_misplaced_sync", setups=("isolate",), host=True, doc=textures_gap_misplaced_sync.__doc__)
    s.bring_up()
    t = RTask(s, pages=128)
    base, sh, h, objs = _placed_objects(t, None)
    b, b_va = t.shared(PAGE)
    t.define()
    pl = install_with_placements(t, {ref: (0x40 + i, 0x1234) for i, (_, ref) in enumerate(objs)})
    for i, (label, ref) in enumerate(objs):
        t.blit(b_sync_resource(ref), b_fill_buf(b, 0x10 * i, 0x10, i + 1),
               resources=[(base, 0), (sh, 0), (h, 0), (b, 0)], label="314: " + label)
    s.r(abi.R_FAULT_MASK, "no fault")
    t.dump(b_va, 0x10 * len(objs), "probe buffer: every exec ran")
    t.dump(pl, 16 * len(t.ol.entries), "placement list")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("map", "isolate"))
def textures_gap_rgba16f_mapper():
    """A 32x16 RGBA16Float mapper-ref texture (type 11, commandType 12) over
    a 64x16 BGRA surface of 4 bytes an element: its rows (256 bytes) fit the
    surface's, the surface allows pixel-size casting (the IOSurface device
    sets no kIOSurfacePixelSizeCastingAllowed), so
    validateIOSurfaceDescriptor passes it, and Metal makes it. Every exec
    completes: listing it, a copy out (the surface's bytes, 8 a texel), a
    fill (half floats in the surface's page). (A wait for a fault here times
    out, which once read as a hang: it is not one.) The IOSurface host device
    needs a memory map: map and isolate."""
    from scenarios.resources_mapper import _surfaces
    s = Scenario("textures_gap_rgba16f_mapper", setups=("map", "isolate"), iosfc=True,
                 doc=textures_gap_rgba16f_mapper.__doc__)
    s.bring_up()
    d, pages = _surfaces(s, {2: (64, 16, 4, b"BGRA", None)})
    t = RTask(s)
    x = t.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(2) + iosurface_op(
        12, desc1(fmt=RGBA16Float, w=32, h=16, options=OPT_SHARED, usage=3)))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_buf(b, 0, 0x10, 1), resources=[(x, 0), (b, 0)], label="list it")
    t.blit(b_copy_tex_buf(x, b, 0x100, 256, 4096, (4, 4, 1)), resources=[(x, 0), (b, 0)], label="copy out")
    t.dump(b_va, 0x10, "B: the first exec ran")
    t.dump(b_va + 0x100, 0x20, "B+0x100: a row of the surface")
    t.blit(b_fill_tex_color(x, (2, 1, 1), (1.0, 0, 0, 1.0), RGBA16Float), resources=[(x, 0), (b, 0)], label="fill")
    s.dump(s.pa(pages[2]), 0x20, "surface 2's page")
    s.r(abi.R_FAULT_MASK, "no fault")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def textures_gap_backing_missing_plane():
    """Type 5 textures asking for plane 1 of a one-plane IOSurface backing,
    through commandTypes 12, 47 and 57. The backing's surface has a plane
    list, so validateIOSurfaceDescriptor checks nothing, and Metal makes no
    texture: "Failed to create texture" through each of 12, 47 and 57; each
    exec faults. Plane 0 through 57 is made. In-process only: a type 5
    texture aborts the service."""
    from scenarios.textures_shared import BGRA, kernel_and_user, two_task_probe
    from pgtest.resources_build import backing_rec, desc2, ref_texture_rec
    s = Scenario("textures_gap_backing_missing_plane", setups=("blocks", "map"),
                 doc=textures_gap_backing_missing_plane.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    bk = k.add(abi.T_IOSURFACE_BACKING, backing_rec(PAGE, k.area(PAGE) // PAGE, BGRA, [(0, 4, 4, 16, 4)]))
    d1 = desc1(fmt=BGRA8Unorm, options=OPT_SHARED, usage=3)
    d2 = desc2(fmt=BGRA8Unorm, options=OPT_SHARED, usage=3)
    cases = [(t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=iosurface_op(12, d1, plane=1))), True),
             (t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=iosurface_op(47, d1, plane=1))), True),
             (t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=iosurface_op(57, d2, plane=1))), True),
             (t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=iosurface_op(57, d2, plane=0))), False)]
    return two_task_probe(s, k, t, cases)


@scenario
def textures_gap_shared_backing_v2_shared_storage():
    """A type 17 shared texture backing made with commandType 53 (a v2
    descriptor) of shared storage: Metal makes no shareable texture of it,
    "Failed to create texture", so the type 16 texture over it faults its
    exec; the same backing of private storage is made."""
    from scenarios.textures_shared import kernel_and_user, two_task_probe
    from pgtest.resources_build import desc2
    s = Scenario("textures_gap_shared_backing_v2_shared_storage",
                 doc=textures_gap_shared_backing_v2_shared_storage.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    ks = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, k.area(PAGE) // PAGE, [simple_dim(4, 4)],
                                                         desc2(options=OPT_SHARED, usage=3), command_type=53))
    kp = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, k.area(PAGE) // PAGE, [simple_dim(4, 4)],
                                                         desc2(options=0x20, usage=3), command_type=53))
    cases = [(t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(ks)), True),
             (t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kp)), False)]
    return two_task_probe(s, k, t, cases)


@scenario(host=True)
def textures_gap_dimension_branches():
    """The texture dimension check over shapes the other records leave, each
    a type 2 texture listed in its own exec: a cube array (textureType
    CubeArray counts as a cube) is made; a 4x4x4 3D texture of three levels,
    whose depth halves with each, is made; a BC1 3D texture of two levels,
    the compressed path's halving, is made; a second level whose height is
    wrong and a 3D first level whose depth is wrong are refused ("Size
    mismatch at mip"), and their execs fault. Host-keyed: whether Metal makes
    a BC1 3D texture is the host's (as in textures_formats)."""
    from scenarios.resources import creation_case
    from pgtest.resources_build import TT_3D, TT_CubeArray
    cube_array = texture_rec(PAGE, 8, [dimension([mip(0, 64, 16, 4, 4)], array=2, cube=1, slice_bytes=64,
                                                 block=(1, 1, 1, 4))],
                             desc1(w=4, h=4, tt=TT_CubeArray, array=2, usage=1))
    d3 = texture_rec(PAGE, 8, [dimension([mip(0, 256, 16, 4, 4, 4), mip(256, 32, 8, 2, 2, 2), mip(288, 4, 4, 1, 1, 1)],
                                         block=(1, 1, 1, 4))],
                     desc1(w=4, h=4, d=4, tt=TT_3D, mips=3, usage=1))
    bc3d = texture_rec(PAGE, 8, [dimension([mip(0, 128, 16, 8, 8, 4), mip(128, 16, 8, 4, 4, 2)], block=(4, 4, 1, 8),
                                           compressed=1)],
                       desc1(fmt=130, w=8, h=8, d=4, tt=TT_3D, mips=2, usage=1))
    bad_h = texture_rec(PAGE, 8, [dimension([mip(0, 64, 16, 4, 4), mip(64, 8, 8, 2, 1)], block=(1, 1, 1, 4))],
                        desc1(w=4, h=4, mips=2, usage=1))
    bad_d = texture_rec(PAGE, 8, [dimension([mip(0, 64, 16, 4, 4, 1)], block=(1, 1, 1, 4))],
                        desc1(w=4, h=4, d=2, tt=TT_3D, usage=1))
    cases = [(abi.T_TEXTURE, cube_array, None, False), (abi.T_TEXTURE, d3, None, False),
             (abi.T_TEXTURE, bc3d, None, False), (abi.T_TEXTURE, bad_h, None, True),
             (abi.T_TEXTURE, bad_d, None, True)]
    return creation_case("textures_gap_dimension_branches", textures_gap_dimension_branches.__doc__, cases, host=True)


@scenario(host=True)
def textures_gap_shared_xr10_srgb():
    """A type 18 shared backing of pixel format 551 (the sRGB counterpart of
    550) in task 0, reached from a type 16 texture in task 1: no shared
    texture resource is made for 551, as for 550, so the backing is not made
    and the texture's exec faults. Host-keyed as
    textures_dual_plane_shared_xr10."""
    from pgtest.resources_build import XR10_2P_A8_SRGB
    from scenarios.textures_shared import kernel_and_user, two_task_probe
    s = Scenario("textures_gap_shared_xr10_srgb", host=True, doc=textures_gap_shared_xr10_srgb.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    kb = k.add(abi.T_SHARED_BACKING2, shared_backing_rec(2 * PAGE, 8, _xr10_dims(),
                                                         desc1(fmt=XR10_2P_A8_SRGB, w=64, h=4, usage=3)))
    return two_task_probe(s, k, t, [(t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kb)), True)])


@scenario
def textures_gap_serializer_v2():
    """Serializer textures (type 8) with v2 descriptors where only v1 ones
    were, each made by its first use in a copy with only the buffers listed
    (listing a serializer texture itself faults its prepare): a 2D array of
    two slices over a shared buffer through commandType 55 takes
    newLinearTextureWithDescriptor and its second slice copies out the
    buffer's bytes at +0x280 (slices 0x80 apart); 55 over an unknown buffer
    reference and 56 in an unknown heap fail ("Failed to get base buffer" /
    "base heap") and fault their copies. A normal texture (type 2,
    commandType 52) whose swizzle names channels 6, 7 and 0xff reads each as
    One, and is made when listed (shader-read only: Metal asserts on a
    swizzled texture that may be written)."""
    from pgtest.resources_build import TT_2DArray, desc2, ser_in_heap, ser_on_buffer
    s = Scenario("textures_gap_serializer_v2", doc=textures_gap_serializer_v2.__doc__)
    s.bring_up()
    t = RTask(s)
    buf, buf_va = t.shared(PAGE, contents=PATTERN * 4)
    arr = t.add(abi.T_SERIALIZER_TEXTURE, ser_on_buffer(buf, 0x200, 16, desc2(tt=TT_2DArray, array=2, options=OPT_SHARED),
                                                        v2=True))
    bad55 = t.add(abi.T_SERIALIZER_TEXTURE, ser_on_buffer(60, 0, 16, desc2(options=OPT_SHARED), v2=True))
    bad56 = t.add(abi.T_SERIALIZER_TEXTURE, ser_in_heap(61, 0, desc2(usage=3), v2=True))
    sw = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)],
                                          desc2(usage=1, swizzle=(6, 7, 0xFF, 3)), command_type=52))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(arr, b, 0, 16, 64, FULL, slice_=1), resources=[(buf, 0), (b, 0)],
           label="copy slice 1 of the v2 array")
    t.dump(b_va, 0x40, "slice 1")
    t.blit(b_copy_tex_buf(bad55, b, 0x40, 16, 64, FULL), resources=[(buf, 0), (b, 0)], fault=True,
           label="55 over an unknown buffer")
    t.blit(b_copy_tex_buf(bad56, b, 0x40, 16, 64, FULL), resources=[(buf, 0), (b, 0)], fault=True,
           label="56 in an unknown heap")
    t.blit(b_fill_buf(b, 0x40, 0x10, 4), resources=[(sw, 0), (b, 0)], label="the swizzled texture listed")
    t.dump(b_va + 0x40, 0x10, "B+0x40")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("map", "isolate"))
def textures_gap_mapper_rotation():
    """Mapper-ref textures (type 11) through commandTypes 47 and 57 with
    rotations 1, 3 and 4, which swap the descriptor's width and height for
    validate_iosurface_descriptor: a 64x16 descriptor over the 64x16 surface is
    then 16 wide and 64 tall, taller than the surface ("Descriptor
    height*arrayLength (64) > IOSurface height (16)"), and each exec faults
    before Metal is asked. Rotation 2 swaps nothing, and is made through 47 and
    57. The descriptors are framebuffer-only render targets: Metal asserts on a
    rotated texture that is not framebufferOnly."""
    from scenarios.resources_mapper import _surfaces, _bgra
    from pgtest.resources_build import desc2
    s = Scenario("textures_gap_mapper_rotation", setups=("map", "isolate"), iosfc=True,
                 doc=textures_gap_mapper_rotation.__doc__)
    s.bring_up()
    _surfaces(s, {2: (64, 16, 4, b"BGRA", None)})
    t = RTask(s)
    cases = []
    f1 = _bgra(fbonly=1, usage=4)
    f2 = desc2(fmt=BGRA8Unorm, w=64, h=16, options=OPT_SHARED, usage=4, fbonly=1)
    for rot in (1, 3, 4, 2):
        cases.append((t.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(2) + iosurface_op(47, f1, rotation=rot)), rot != 2))
        cases.append((t.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(2) + iosurface_op(57, f2, rotation=rot)), rot != 2))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    for i, (r, faults) in enumerate(cases):
        t.blit(b_fill_buf(b, 0x10 * i, 0x10, i + 1), resources=[(r, 0), (b, 0)], fault=faults, label="object %d" % r)
    t.dump(b_va, 0x10 * len(cases), "probe buffer: which execs ran")
    t.delete()
    s.events("end")
    return s
