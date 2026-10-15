"""Serializer textures (object types 8 and 9, which are not told apart):
memoryless textures, views of another texture, textures over a buffer and
textures placed in a heap (docs/testing.md, section 6.19).

A texture over a shared buffer is the no-copy path for texels: a blit into it
lands in the buffer's guest pages. A view pages nothing: listing one in an
exec's resources makes its prepare throw ("Attempting to prepare a view
resource")."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.resources_build import (RGBA8Unorm, BGRA8Unorm, TT_2D, TT_2DArray, OPT_SHARED, RTask, b_copy_tex_buf,
                                    b_fill_buf, b_fill_tex_color, buffer_rec, cmd_discard, cmd_invalidate,
                                    cmd_sync_discard, cmd_synchronize, desc1, desc2, heap_rec, ser_in_heap,
                                    ser_memoryless, ser_on_buffer, ser_view_format, ser_view_type, simple_dim,
                                    texture_rec)
from pgtest.scenario import Scenario

S8, S9 = abi.T_SERIALIZER_TEXTURE, abi.T_SERIALIZER_TEXTURE9
FULL = (4, 4, 1)
TEXELS = bytes((i * 9 + 2) & 0xFF for i in range(64))


@scenario
def textures_on_buffer():
    """Textures over a shared buffer (commandType 9 with a v1 descriptor, 55
    with a v2), made with the buffer's resource options: a fill through the
    texture is in the buffer's guest pages with no flags, and a guest write
    is seen by the next copy out. A 2D array takes
    newLinearTextureWithDescriptor. Type 9 names the same creator. In the
    service the buffer is a detach buffer and the texture over it pages with
    it."""
    s = Scenario("textures_on_buffer", doc=textures_on_buffer.__doc__)
    s.bring_up()
    t = RTask(s)
    buf, buf_va = t.shared(PAGE, contents=TEXELS * 4)
    v1 = t.add(S8, ser_on_buffer(buf, 0, 16, desc1(options=OPT_SHARED)))
    v2 = t.add(S9, ser_on_buffer(buf, 0x100, 16, desc2(options=OPT_SHARED), v2=True))
    arr = t.add(S8, ser_on_buffer(buf, 0x200, 16, desc1(tt=TT_2DArray, array=2, options=OPT_SHARED)))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(v1, b, 0, 16, 64, FULL), b_copy_tex_buf(v2, b, 0x40, 16, 64, FULL),
           b_copy_tex_buf(arr, b, 0x80, 16, 64, FULL, slice_=1), resources=[(buf, 0), (b, 0)],
           label="copy out of the three textures")
    t.dump(b_va, 0xC0, "B")
    t.blit(b_fill_tex_color(v1, (2, 2, 1), (1.0, 0, 0, 1.0), RGBA8Unorm),
           b_fill_tex_color(v2, (2, 2, 1), (0, 1.0, 0, 1.0), RGBA8Unorm, origin=(2, 2, 0)),
           b_fill_tex_color(arr, (4, 1, 1), (0, 0, 1.0, 1.0), RGBA8Unorm, slice_=0),
           resources=[(buf, 0), (b, 0)], label="fill through the textures, no flags")
    t.dump(buf_va, 0x300, "the buffer's pages")
    t.write(buf_va, b"\x5A" * 16)
    t.blit(b_copy_tex_buf(v1, b, 0xC0, 16, 64, FULL), resources=[(buf, 0), (b, 0)], label="a guest write")
    t.dump(b_va + 0xC0, 0x40, "B+0xc0")
    t.delete()
    s.events("end")
    return s


@scenario
def textures_views():
    """Views of a normal texture: commandType 7 (a pixel format view), 8
    (type, levels and slices) and 27 (with a swizzle). A view is a child
    resource of its base: a discardable one over a discardable base. The base
    pages; the views read and write its texels. Listing a view in an exec's
    resources makes its prepare throw (a fault). A view marked with 52
    (setHostValid, clearGuestValid) needs a synchronize, which throws inside
    opcode 53 and is logged and skipped. 63 discards a view (nothing) and 62
    synchronizes and discards it."""
    s = Scenario("textures_views", doc=textures_views.__doc__)
    s.bring_up()
    t = RTask(s)
    base_va = t.area(PAGE, contents=TEXELS)
    base = t.add(abi.T_TEXTURE, texture_rec(PAGE, base_va // PAGE, [simple_dim(4, 4)],
                                            desc1(usage=0x13)))  # read, write, pixel format view
    fv = t.add(S8, ser_view_format(base, BGRA8Unorm))
    tv = t.add(S8, ser_view_type(base, RGBA8Unorm, TT_2DArray))
    sv = t.add(S9, ser_view_type(base, RGBA8Unorm, TT_2D, swizzle=(5, 4, 3, 2)))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(fv, b, 0, 16, 64, FULL), b_copy_tex_buf(tv, b, 0x40, 16, 64, FULL),
           b_copy_tex_buf(sv, b, 0x80, 16, 64, FULL), resources=[(base, 0), (b, 0)],
           label="read the base through three views")
    t.dump(b_va, 0xC0, "B")
    t.blit(b_fill_tex_color(fv, (2, 2, 1), (1.0, 0, 0, 1.0), BGRA8Unorm), resources=[(base, 6), (b, 0)],
           label="fill through the BGRA view, synchronize the base")
    t.dump(base_va, 0x40, "the base's pages")
    t.blit(b_fill_buf(b, 0x100, 0x10, 0x11), resources=[(fv, 0), (b, 0)], fault=True, label="a view in the resources")
    cmd_invalidate(s, 1, [(tv, 0, 1, 1, 0)], label="52: the view host-valid, guest-invalid")
    cmd_synchronize(s, 1, [tv], label="53 on the view: its synchronize throws, logged")
    cmd_discard(s, 1, [fv], label="63 on a view")
    cmd_sync_discard(s, 1, [tv], label="62 on a view")
    t.blit(b_copy_tex_buf(fv, b, 0x100, 16, 64, FULL), b_copy_tex_buf(tv, b, 0x140, 16, 64, FULL),
           resources=[(base, 0), (b, 0)], label="the views after 63 and 62")
    t.dump(b_va + 0x100, 0x80, "B+0x100")
    t.delete()
    s.events("end")
    return s


@scenario
def textures_in_heap():
    """Textures placed in a heap (commandType 21 with a v1 descriptor, 56
    with a v2), made with the heap's resource options. In-process the heap is
    private and pages as a whole: a texture in it shows the guest bytes at
    its offset once the heap is listed, and a fill with the heap listed with
    flags 6 lands in the heap's pages. In the service the heap is a shared
    detach heap over the guest pages."""
    s = Scenario("textures_in_heap", doc=textures_in_heap.__doc__)
    s.bring_up()
    t = RTask(s, pages=80)
    h_va = t.area(4 * PAGE, contents=TEXELS)
    t.write(h_va + PAGE, TEXELS[::-1])
    h = t.add(abi.T_HEAP, heap_rec(h_va // PAGE, 4 * PAGE, flags=1))
    t1 = t.add(S8, ser_in_heap(h, 0, desc1(usage=3)))
    t2 = t.add(S9, ser_in_heap(h, PAGE, desc2(usage=3), v2=True))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(t1, b, 0, 16, 64, FULL), b_copy_tex_buf(t2, b, 0x40, 16, 64, FULL),
           resources=[(h, 0), (b, 0)], label="read two heap textures")
    t.dump(b_va, 0x80, "B")
    t.blit(b_fill_tex_color(t1, (2, 2, 1), (0, 0, 1.0, 1.0), RGBA8Unorm), resources=[(h, 6), (b, 0)],
           label="fill a heap texture, synchronize the heap")
    t.dump(h_va, 0x40, "the heap's pages")
    t.delete()
    s.events("end")
    return s


@scenario
def textures_serializer_records():
    """Serializer texture refusals and variants, each its own object listed
    in its own exec: a short header; an unknown commandType; a memoryless
    texture (commandType 1 and 52, which are made); a non-memoryless one
    ("Invalid storage mode"); short operations for 1, 52, 7, 8, 27, 9, 55,
    21, 56; views of reference 0, of an unknown reference, of a buffer (no
    Resource::get_texture: its kind raises Unrecognized); a texture on
    reference 0, on a texture (no get_buffer); a heap texture on reference 0
    (Metal makes nothing: "Failed to create texture") and on a buffer (no
    get_heap). Views (7, 8, 27), a texture on a buffer and one in a heap
    whose base is past the object list ("Failed to get base texture", "...
    base buffer", "... base heap"). Every serializer texture made faults
    when listed in an exec's resources: none is host-valid, so each is
    prepared, and a serializer texture's prepare throws."""
    from scenarios.resources import creation_case
    ml = desc1(options=0x30, usage=4)
    base_rec = texture_rec(PAGE, 20, [simple_dim(4, 4)], desc1(usage=0x13))
    cases = [
        (S8, ser_memoryless(ml), 7, True),                                         # 1
        (S8, struct.pack("<II", 2, 0) + bytes(0x40), None, True),                   # 2 unknown type
        (S8, ser_memoryless(ml), None, True),                                       # 3 memoryless v1
        (S8, ser_memoryless(desc2(options=0x30, usage=4), command_type=52), None, True),  # 4 v2
        (S8, ser_memoryless(desc1()), None, True),                                  # 5 private
        (S8, ser_memoryless(ml), 8 + 0x23, True),
        (S8, ser_memoryless(desc2(options=0x30, usage=4), command_type=52), 8 + 0x2b, True),
        (S8, ser_view_format(20, BGRA8Unorm), 8 + 0xb, True),
        (S8, ser_view_type(20, RGBA8Unorm, TT_2D), 8 + 0x2b, True),
        (S8, ser_view_type(20, RGBA8Unorm, TT_2D, swizzle=(2, 3, 4, 5)), 8 + 0x2f, True),   # 10
        (S8, ser_on_buffer(21, 0, 16), 8 + 0x37, True),
        (S8, ser_on_buffer(21, 0, 16, v2=True), 8 + 0x3f, True),
        (S8, ser_in_heap(22, 0), 8 + 0x33, True),
        (S8, ser_in_heap(22, 0, v2=True), 8 + 0x3b, True),
        (S8, ser_view_format(0, BGRA8Unorm), None, True),                          # 15
        (S8, ser_view_type(40, RGBA8Unorm, TT_2D), None, True),
        (S8, ser_view_type(21, RGBA8Unorm, TT_2D, swizzle=(2, 3, 4, 5)), None, True),
        (S8, ser_on_buffer(0, 0, 16), None, True),
        (S8, ser_on_buffer(20, 0, 16), None, True),
        (abi.T_TEXTURE, base_rec, None, None),                                      # 20 a base texture
        (abi.T_BUFFER, buffer_rec(PAGE, 21), None, None),                           # 21 a buffer
        (abi.T_HEAP, heap_rec(22, 4 * PAGE, 1), None, None),                        # 22 a heap
        (S8, ser_in_heap(0, 0), None, True),
        (S8, ser_in_heap(21, 0), None, True),
        (S8, ser_view_format(20, BGRA8Unorm), None, True),                         # 25 good view: prepare throws
        (S8, ser_on_buffer(21, 0, 16), None, True),                                 # 26 on a buffer
        (S9, ser_in_heap(22, 0), None, True),                                       # 27 in a heap
        (S8, ser_view_format(40, BGRA8Unorm), None, True),                         # 28 bases past the list
        (S8, ser_view_type(40, RGBA8Unorm, TT_2D, swizzle=(2, 3, 4, 5)), None, True),
        (S8, ser_on_buffer(40, 0, 16), None, True),
        (S8, ser_in_heap(40, 0), None, True),
    ]
    return creation_case("textures_serializer_records", textures_serializer_records.__doc__, cases, pages=96)
