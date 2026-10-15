"""Normal textures (object types 2, 3): their records, paging in and out of
guest memory, mip levels, arrays, cubes and 3D textures
(docs/testing.md, sections 4.10, 6.19).

A normal texture's bytes live in guest pages the record names; an exec that
lists the texture copies them in (prepare), and invalidateGuest plus
synchronize copy the texture back out. The blit decoder is the probe: 321
FillTextureWithColor writes a texture, 302 CopyFromTextureToBuffer reads one
into a shared buffer."""
from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.resources_build import (RGBA8Unorm, R8Unorm, TT_2DArray, TT_3D, TT_Cube, OPT_PRIVATE, OPT_SHARED, RTask,
                                    b_copy_buf_tex, b_copy_tex_buf, b_copy_tex_tex, b_fill_tex_color, b_gen_mipmaps,
                                    cmd_discard, cmd_invalidate, cmd_sync_discard, cmd_synchronize, desc1, desc2,
                                    simple_dim, texture_rec)
from pgtest.scenario import Scenario

TEXELS = bytes((i * 5 + 1) & 0xFF for i in range(4 * 4 * 4))


def tex(t, fmt=RGBA8Unorm, w=4, h=4, bpp=4, contents=None, typ=abi.T_TEXTURE, v2=False, levels=1, options=OPT_PRIVATE,
        **dkw):
    """A normal texture with its bytes in a fresh task page; returns (ref, va)."""
    va = t.area(PAGE, contents=contents)
    dim = simple_dim(w, h, bpp=bpp, levels=levels)
    if v2:
        d = desc2(fmt=fmt, w=w, h=h, mips=levels, options=options, **dkw)
        return t.add(typ, texture_rec(PAGE, va // PAGE, [dim], d, command_type=52)), va
    d = desc1(fmt=fmt, w=w, h=h, mips=levels, options=options, **dkw)
    return t.add(typ, texture_rec(PAGE, va // PAGE, [dim], d)), va


@scenario
def textures_normal_roundtrip():
    """A 4x4 RGBA8Unorm texture (type 2, descriptor v1, private) over guest
    bytes: listed in an exec it is paged in, so a copy to a shared buffer
    shows the guest's texels; a fill with no flags stays on the host; with
    invalidateGuest + synchronize (flags 6) the fill is copied back into the
    guest's pages; opcode 52 + 53 does the same; invalidateHost (flag 1)
    re-reads a guest write. The same for a descriptor v2 (commandType 52)
    texture with shared storage."""
    s = Scenario("textures_normal_roundtrip", doc=textures_normal_roundtrip.__doc__)
    s.bring_up()
    t = RTask(s)
    a, a_va = tex(t, contents=TEXELS)
    c, c_va = tex(t, contents=TEXELS[::-1], v2=True, options=OPT_SHARED)
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    full = (4, 4, 1)
    t.blit(b_copy_tex_buf(a, b, 0, 16, 64, full), b_copy_tex_buf(c, b, 0x40, 16, 64, full),
           resources=[(a, 0), (c, 0), (b, 0)], label="copy both textures out")
    t.dump(b_va, 0x80, "B: both textures as paged in")
    t.blit(b_fill_tex_color(a, (2, 2, 1), (1.0, 0.0, 0.0, 1.0), RGBA8Unorm),
           b_copy_tex_buf(a, b, 0x80, 16, 64, full), resources=[(a, 0), (b, 0)], label="fill A, no flags")
    t.dump(a_va, 0x40, "A's pages: unchanged")
    t.dump(b_va + 0x80, 0x40, "B+0x80: A on the host")
    t.blit(b_fill_tex_color(a, (2, 2, 1), (0.0, 1.0, 0.0, 1.0), RGBA8Unorm, origin=(2, 2, 0)),
           resources=[(a, 6), (b, 0)], label="fill A, invalidateGuest + synchronize")
    t.dump(a_va, 0x40, "A's pages: synchronized")
    t.blit(b_fill_tex_color(c, (4, 1, 1), (0.0, 0.0, 1.0, 1.0), RGBA8Unorm), resources=[(c, 0), (b, 0)],
           label="fill C")
    cmd_invalidate(s, 1, [(c, 0, 0, 1, 0)], label="52: C's guest copy invalid")
    cmd_synchronize(s, 1, [c], label="53: C copied back")
    t.dump(c_va, 0x40, "C's pages")
    t.write(a_va, b"\xAA" * 16)
    t.blit(b_copy_tex_buf(a, b, 0xC0, 16, 64, full), resources=[(a, 1), (b, 0)], label="A with invalidateHost")
    t.dump(b_va + 0xC0, 0x40, "B+0xc0: A re-read from the guest")
    t.blit(b_copy_buf_tex(b, a, 0, 16, 64, full), b_copy_tex_tex(a, c, full), resources=[(a, 6), (c, 6), (b, 0)],
           label="buffer to A, A to C, both synchronized")
    t.dump(a_va, 0x40, "A's pages")
    t.dump(c_va, 0x40, "C's pages")
    cmd_sync_discard(s, 1, [a, c], label="62")
    cmd_discard(s, 1, [a, c], label="63")
    t.blit(b_copy_tex_buf(a, b, 0x100, 16, 64, full), resources=[(a, 0), (b, 0)], label="A after a discard")
    t.dump(b_va + 0x100, 0x40, "B+0x100")
    t.delete()
    s.events("end")
    return s


@scenario
def textures_mip_levels():
    """Mip levels. Type 2 with 3 levels pages each level from its own offset
    in the guest's bytes; type 3 (generate mipmaps) pages level 0 only and
    generates the rest on the GPU, even though its record describes 3 levels.
    A type 3 texture with one level does not generate. 307 GenerateMipmaps on
    a type 2 texture. Each level is read back with 302."""
    s = Scenario("textures_mip_levels", doc=textures_mip_levels.__doc__)
    s.bring_up()
    t = RTask(s)
    # 8x8 R8Unorm: levels of 64, 16 and 4 bytes, packed.
    lv = bytes(range(0x40)) + bytes(0x80 + i for i in range(0x10)) + bytes([0xF0, 0xF1, 0xF2, 0xF3])
    a, a_va = tex(t, fmt=R8Unorm, w=8, h=8, bpp=1, levels=3, contents=lv)
    g, g_va = tex(t, fmt=R8Unorm, w=8, h=8, bpp=1, levels=3, contents=bytes([0x40]) * 0x40 + bytes([0x11]) * 0x14,
                  typ=abi.T_TEXTURE_MIPGEN)
    one, one_va = tex(t, fmt=R8Unorm, w=8, h=8, bpp=1, levels=1, contents=bytes([0x22]) * 0x40,
                      typ=abi.T_TEXTURE_MIPGEN)
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    recs = []
    for i, (r, off) in enumerate(((a, 0), (g, 0x100))):
        recs += [b_copy_tex_buf(r, b, off, 8, 64, (8, 8, 1), level=0),
                 b_copy_tex_buf(r, b, off + 0x40, 4, 16, (4, 4, 1), level=1),
                 b_copy_tex_buf(r, b, off + 0x50, 2, 4, (2, 2, 1), level=2)]
    recs.append(b_copy_tex_buf(one, b, 0x200, 8, 64, (8, 8, 1)))
    t.blit(*recs, resources=[(a, 0), (g, 0), (one, 0), (b, 0)], label="read every level")
    t.dump(b_va, 0x240, "B: A's levels, G's levels, the one-level texture")
    t.blit(b_fill_tex_color(a, (8, 8, 1), (1.0, 0, 0, 1.0), R8Unorm), b_gen_mipmaps(a),
           b_copy_tex_buf(a, b, 0x300, 4, 16, (4, 4, 1), level=1), resources=[(a, 6), (b, 0)],
           label="fill level 0, generate, synchronize")
    t.dump(b_va + 0x300, 0x10, "B+0x300: A's level 1 after 307")
    t.dump(a_va, 0x54, "A's pages: all three levels synchronized")
    t.blit(b_fill_tex_color(g, (8, 8, 1), (0.5, 0, 0, 1.0), R8Unorm), resources=[(g, 6), (b, 0)],
           label="G synchronized")
    t.dump(g_va, 0x54, "G's pages: level 0 only")
    t.delete()
    s.events("end")
    return s


@scenario
def textures_slices():
    """Arrays, cubes and 3D textures page slice by slice: a slice's bytes are
    at baseOffset + bytesPerSlice x slice + the mip's offset, and a cube has
    6 faces a slice. Each texture is read back slice by slice, then filled
    and synchronized."""
    s = Scenario("textures_slices", doc=textures_slices.__doc__)
    s.bring_up()
    t = RTask(s)
    from pgtest.resources_build import dimension, mip
    b, b_va = t.shared(PAGE)
    # 2D array of 3 slices, 2x2 RGBA8, 0x40 bytes a slice, base offset 0x20.
    arr_va = t.area(PAGE, contents=bytes(0x20) + b"".join(bytes([0x10 * (s_ + 1)]) * 0x40 for s_ in range(3)))
    arr = t.add(abi.T_TEXTURE, texture_rec(PAGE, arr_va // PAGE, [dimension([mip(0, 16, 8, 2, 2)], array=3, base=0x20,
                                                                             slice_bytes=0x40)],
                                           desc1(w=2, h=2, tt=TT_2DArray, array=3)))
    # A cube, 2x2 RGBA8, 6 faces of 0x10 bytes.
    cube_va = t.area(PAGE, contents=b"".join(bytes([0xC0 + f]) * 0x10 for f in range(6)))
    cube = t.add(abi.T_TEXTURE, texture_rec(PAGE, cube_va // PAGE, [dimension([mip(0, 16, 8, 2, 2)], cube=1,
                                                                              slice_bytes=0x10)],
                                            desc1(w=2, h=2, tt=TT_Cube)))
    # A 3D texture, 2x2x2 RGBA8.
    v_va = t.area(PAGE, contents=bytes(range(0x20)))
    vol = t.add(abi.T_TEXTURE, texture_rec(PAGE, v_va // PAGE, [dimension([mip(0, 32, 8, 2, 2, 2)])],
                                           desc1(w=2, h=2, d=2, tt=TT_3D)))
    t.define()
    t.install()
    recs = [b_copy_tex_buf(arr, b, 0x10 * i, 8, 16, (2, 2, 1), slice_=i) for i in range(3)]
    recs += [b_copy_tex_buf(cube, b, 0x40 + 0x10 * f, 8, 16, (2, 2, 1), slice_=f) for f in range(6)]
    recs.append(b_copy_tex_buf(vol, b, 0xA0, 8, 16, (2, 2, 2)))
    t.blit(*recs, resources=[(arr, 0), (cube, 0), (vol, 0), (b, 0)], label="read every slice")
    t.dump(b_va, 0xC0, "B: array slices, cube faces, the volume")
    t.blit(b_fill_tex_color(arr, (2, 2, 1), (0, 0, 1.0, 1.0), RGBA8Unorm, slice_=1),
           b_fill_tex_color(cube, (2, 2, 1), (0, 1.0, 0, 1.0), RGBA8Unorm, slice_=5),
           b_fill_tex_color(vol, (2, 2, 1), (1.0, 1.0, 0, 1.0), RGBA8Unorm, origin=(0, 0, 1)),
           resources=[(arr, 6), (cube, 6), (vol, 6), (b, 0)], label="fill one slice of each, synchronize")
    t.dump(arr_va, 0xE0, "array pages")
    t.dump(cube_va, 0x60, "cube pages")
    t.dump(v_va, 0x20, "volume pages")
    t.delete()
    s.events("end")
    return s


def _tex_case(dims, desc=None, length=PAGE, typ=abi.T_TEXTURE, command_type=1, op=None, entry_len=None, faults=True):
    """One normal-texture record for creation_case, its bytes on page 8."""
    return (typ, texture_rec(length, 8, dims, desc, command_type=command_type, op=op), entry_len, faults)


def _cut(record, n):
    return record[:n]


@scenario
def textures_normal_record_checks():
    """Every refusal of a normal texture's record and of the texture
    dimension check, one object each, each listed in its own exec so that its
    creation's fault is its own; the probe buffer shows which execs ran.
    Short reads, an unknown commandType, a memoryless descriptor, a texture
    Metal refuses, then the dimension checks in the order the code makes
    them, for an uncompressed format (kind 0) and a block-compressed one
    (kind 1, BC1_RGBA: 4x4 blocks of 8 bytes). A passing uncompressed,
    compressed and type 3 record close each group."""
    from pgtest.resources_build import BC1_RGBA, Depth32Float_Stencil8, dimension, mip, texture_rec as tr
    from scenarios.resources import creation_case
    ok = simple_dim(4, 4)
    good = tr(PAGE, 8, [ok], desc1())
    big = 1 << 63
    cases = [
        (abi.T_TEXTURE, good, 11, True),                                # short APVObjectTexture
        (abi.T_TEXTURE, good, 12 + 0x17, True),                         # short dimension
        (abi.T_TEXTURE, good, 12 + 0x18 + 0x23, True),                  # short mips
        (abi.T_TEXTURE, good, 12 + 0x3c + 7, True),                     # short operation header
        _tex_case([ok], command_type=3),                                # unknown commandType
        (abi.T_TEXTURE, good, 12 + 0x3c + 8 + 0x23, True),              # short v1 operation
        (abi.T_TEXTURE, tr(PAGE, 8, [ok], desc2(), command_type=52), 12 + 0x3c + 8 + 0x2b, True),  # short v2
        _tex_case([ok], desc1(options=0x30, usage=4)),                  # memoryless
        _tex_case([ok], desc1(fmt=Depth32Float_Stencil8)),              # 1 plane, a two-plane format
        _tex_case([dimension([mip(0, 64, 16, 4, 4)], slice_bytes=1 << 32)]),        # u32 overflow
        # A length that is not a page multiple: in-process with the memory
        # map the transfer buffer is refused at commit; the blocks setup
        # maps it.
        _tex_case([ok], length=0x3f, faults={"blocks": False, "map": True, "service": True}),
        _tex_case([dimension([mip(0, 64, 16, 4, 4)], slice_bytes=0x40)], length=0x3f),  # allocation too small
        _tex_case([dimension([mip(0, 64, 16, 4, 4)], array=2)]),        # array length
        _tex_case([simple_dim(4, 4, levels=2)]),                        # mip count
        _tex_case([dimension([mip(0, 64, 16, 4, 4)], cube=1)]),         # cubeness
        _tex_case([dimension([mip(0, 64, 16, 4, 4)], compressed=1)]),   # compression
        _tex_case([dimension([mip(0, 64, 16, 4, 4)], block=(2, 1, 1, 4))]),
        _tex_case([dimension([mip(0, 64, 16, 4, 4)], block=(1, 2, 1, 4))]),
        _tex_case([dimension([mip(0, 64, 16, 4, 4)], block=(1, 1, 2, 4))]),
        _tex_case([dimension([mip(0, 64, 16, 4, 4)], block=(1, 1, 1, 2))]),  # pixel size
        _tex_case([dimension([mip(0, 64, 16, 3, 4)])]),                 # width at mip 0
        _tex_case([dimension([mip(0, 64, 16, 4, 3)])]),                 # height
        _tex_case([dimension([mip(0, 64, 16, 4, 4, 2)])]),              # depth
        _tex_case([dimension([mip(0, 64, 12, 4, 4)])]),                 # row length
        _tex_case([dimension([mip(0, 72, 18, 4, 4)])]),                 # row not a pixel multiple
        _tex_case([dimension([mip(0, 64, big, 4, 4)])]),                # totalMipSize overflow
        _tex_case([dimension([mip((1 << 64) - 16, 64, 16, 4, 4)])]),    # mipEnd overflow
        _tex_case([dimension([mip(0, 63, 16, 4, 4)])]),                 # declared size
        _tex_case([dimension([mip(0, 64, 16, 4, 4)], slice_bytes=0x3f)], length=0x3f * 1),  # slice size
        _tex_case([dimension([mip(0x10, 64, 16, 4, 4)], slice_bytes=0x4f)]),                # mip end > slice
        _tex_case([dimension([mip(0, 64, 16, 4, 4), mip(0, 16, 8, 2, 2)])], desc1(mips=2)),  # non-increasing
        _tex_case([dimension([mip(0, 64, 16, 4, 4), mip(0x40, 16, 8, 2, 2)], slice_bytes=0x50)], desc1(mips=2),
                  faults=False),
        _tex_case([simple_dim(4, 4, levels=2)], typ=abi.T_TEXTURE_MIPGEN, faults=False),   # type 3: count unchecked
        _tex_case([ok], faults=False),
    ]
    bc = lambda mips, **kw: dimension(mips, block=kw.pop("block", (4, 4, 1, 8)), compressed=kw.pop("compressed", 1), **kw)
    bcd = desc1(fmt=BC1_RGBA, w=8, h=8)
    cases += [
        _tex_case([bc([mip(0, 32, 16, 8, 8)], compressed=0)], bcd),     # compression mismatch
        _tex_case([bc([mip(0, 32, 16, 8, 8)], block=(4, 4, 1, 16))], bcd),
        _tex_case([bc([mip(0, 32, 16, 8, 8)], block=(2, 4, 1, 8))], bcd),
        _tex_case([bc([mip(0, 32, 16, 8, 8)], block=(4, 2, 1, 8))], bcd),
        _tex_case([bc([mip(0, 32, 16, 8, 8)], block=(4, 4, 2, 8))], bcd),
        _tex_case([bc([mip(0, 32, 16, 4, 8)])], bcd),                   # size at mip
        _tex_case([bc([mip(0, 32, 8, 8, 8)])], bcd),                    # row shorter than 2 blocks
        _tex_case([bc([mip(0, 40, 20, 8, 8)])], bcd),                   # row not a block multiple
        _tex_case([bc([mip(0, 32, big, 8, 8)])], bcd),                  # overflow
        _tex_case([bc([mip((1 << 64) - 8, 32, 16, 8, 8)])], bcd),       # mipEnd overflow
        _tex_case([bc([mip(0, 31, 16, 8, 8)])], bcd),                   # declared size
        _tex_case([bc([mip(0, 32, 16, 8, 8)], slice_bytes=0x1f)], bcd),  # mip end > slice
        _tex_case([bc([mip(0, 32, 16, 8, 8), mip(0x10, 8, 8, 4, 4)])], desc1(fmt=BC1_RGBA, w=8, h=8, mips=2)),
        _tex_case([bc([mip(0, 32, 16, 8, 8), mip(0x20, 8, 8, 4, 4)])], desc1(fmt=BC1_RGBA, w=8, h=8, mips=2),
                  faults=False),
        _tex_case([bc([mip(0, 32, 16, 8, 8)])], bcd, faults=False),
    ]
    return creation_case("textures_normal_record_checks", textures_normal_record_checks.__doc__, cases)
