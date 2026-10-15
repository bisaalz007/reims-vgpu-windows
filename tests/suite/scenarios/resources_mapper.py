"""Mapper-ref buffers (type 15) and textures (type 11): resources over an
IOSurface the IOSurface host device mapped over guest pages, named by its
mapping ID (docs/testing.md, sections 6.11, 6.23).

A type 15 record is {u64 mappingID}; its buffer is made on first prepare from
the mapper's surface. A type 11 record is {u64 mappingID}, an operation
header and an IOSurface texture operation (12, 47 or 57), checked against the
surface by validateIOSurfaceDescriptor. The IOSurface host device exists only
with a memory map (apple-gfx-mmio.c), so map and isolate setups only. The
surfaces alias guest pages: GPU writes land there at once."""
from pgtest import abi
from pgtest.abi import PAGE
from pgtest.iosfc_guest import CMD_MAP, CMD_UNMAP, IOSFC, desc_v2
from pgtest.registry import scenario
from pgtest.resources_build import (BGRA8Unorm, OPT_SHARED, TT_2DArray, TT_3D, TT_TextureBuffer, RTask, b_copy_buf_buf,
                                    b_copy_tex_buf, b_fill_buf, b_fill_tex_color, cmd_delete_resource, cmd_discard,
                                    cmd_synchronize, desc1, desc2, iosurface_op, mapper_ref_rec, op_hdr)
from pgtest.scenario import Scenario

SETUPS = ("map", "isolate")
W, H = 64, 16  # one 4 KiB BGRA surface, rows of 256 bytes


def _surfaces(s, spec):
    """Starts the IOSurface host device (v2 descriptors) and maps one
    surface per {mappingID: (width, height, bpe, fmt, bpr)} over its own
    guest page, filled with a pattern. Returns {mappingID: data page}."""
    d = IOSFC(s, v2=True)
    d.start()
    pages = {}
    for mid, (w, h, bpe, fmt, bpr) in spec.items():
        root, data = d.surface_pages(1)
        s.mem(s.pa(data), bytes((i * 3 + mid) & 0xFF for i in range(PAGE)))
        pages[mid] = data
        d.put_desc(mid, desc_v2(root, w, h, fmt=fmt, bpe=bpe, bpr=bpr))
    d.submit(*[(CMD_MAP, m) for m in spec])
    return d, pages


def _bgra(w=W, h=H, **kw):
    kw.setdefault("usage", 3)
    return desc1(fmt=BGRA8Unorm, w=w, h=h, options=OPT_SHARED, **kw)


@scenario(setups=SETUPS)
def resources_mapper_buffer():
    """A mapper-ref buffer over mapped surface 1: a copy out of it shows the
    surface's guest page; a fill of it lands in that page at once; 53 has
    nothing to do (it never needs a synchronize). 63 discards it, releasing
    the surface, and the next exec retains it again. After 37 deletes it, the
    device can unmap the surface. A buffer over mapping ID 99, which nothing
    mapped, is made, and its prepare throws ("Attempting to create mapper ref
    against invalid surface"): its exec faults. A 7-byte record fails
    ("Failed to read mapper ref buffer descriptor")."""
    s = Scenario("resources_mapper_buffer", setups=SETUPS, iosfc=True, doc=resources_mapper_buffer.__doc__)
    s.bring_up()
    d, pages = _surfaces(s, {1: (W, H, 4, b"BGRA", None)})
    t = RTask(s)
    m = t.add(abi.T_MAPPER_BUFFER, mapper_ref_rec(1))
    bad = t.add(abi.T_MAPPER_BUFFER, mapper_ref_rec(99))
    short_va = t.ol._rec
    t.write(short_va, mapper_ref_rec(1))
    t.ol._rec += 16
    short = t.ol.add_raw(abi.T_MAPPER_BUFFER, 7, short_va)
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_buf_buf(m, b, 0, 0, 0x40), resources=[(m, 0), (b, 0)], label="copy out of the mapper buffer")
    t.dump(b_va, 0x40, "B: the surface's page")
    t.blit(b_fill_buf(m, 0x10, 0x10, 0xC5), resources=[(m, 0), (b, 0)], label="fill the mapper buffer")
    s.dump(s.pa(pages[1]), 0x40, "surface 1's page")
    cmd_synchronize(s, 1, [m], label="53 on it")
    cmd_discard(s, 1, [m], label="63 on it")
    s.mem(s.pa(pages[1]), b"\x7A" * 0x10)
    t.blit(b_copy_buf_buf(m, b, 0, 0x40, 0x20), resources=[(m, 0), (b, 0)], label="after 63")
    t.dump(b_va + 0x40, 0x20, "B+0x40")
    t.blit(b_fill_buf(b, 0x80, 0x10, 0x01), resources=[(bad, 0), (b, 0)], fault=True, label="mapping ID 99")
    t.blit(b_fill_buf(b, 0x90, 0x10, 0x02), resources=[(short, 0), (b, 0)], fault=True,
           label="a record of 7 bytes")
    idx = d.submit((CMD_UNMAP, 1))
    d.status(idx, "unmap while the buffer holds the surface")
    cmd_delete_resource(s, 1, m, label="37 on the mapper buffer")
    idx = d.submit((CMD_UNMAP, 1))
    d.status(idx, "unmap after 37")
    t.dump(b_va + 0x80, 0x20, "B+0x80")
    t.delete()
    s.events("end")
    return s


@scenario(setups=SETUPS)
def resources_mapper_texture():
    """Mapper-ref textures over mapped surface 2 (64x16 BGRA, rows of 256
    bytes), commandTypes 12, 47 and 57: a copy out shows the surface's guest
    page; a fill lands there at once. A texture over mapping ID 99 fails
    ("Attempting to create mapper ref against invalid surface"). 37 deletes
    one, releasing its surface."""
    s = Scenario("resources_mapper_texture", setups=SETUPS, iosfc=True, doc=resources_mapper_texture.__doc__)
    s.bring_up()
    d, pages = _surfaces(s, {2: (W, H, 4, b"BGRA", None)})
    t = RTask(s)
    x12 = t.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(2) + iosurface_op(12, _bgra()))
    x47 = t.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(2) + iosurface_op(47, _bgra()))
    x57 = t.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(2) + iosurface_op(
        57, desc2(fmt=BGRA8Unorm, w=W, h=H, options=OPT_SHARED, usage=3)))
    bad = t.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(99) + iosurface_op(12, _bgra()))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(x12, b, 0, 256, 1024, (4, 4, 1)), b_copy_tex_buf(x47, b, 0x400, 256, 1024, (4, 4, 1)),
           b_copy_tex_buf(x57, b, 0x800, 256, 1024, (4, 4, 1)),
           resources=[(x12, 0), (x47, 0), (x57, 0), (b, 0)], label="copy out of three textures")
    t.dump(b_va, 0x10, "B via 12")
    t.dump(b_va + 0x400, 0x10, "B via 47")
    t.dump(b_va + 0x800, 0x10, "B via 57")
    t.blit(b_fill_tex_color(x57, (2, 1, 1), (1.0, 0, 0, 1.0), BGRA8Unorm), resources=[(x57, 0), (b, 0)],
           label="fill through 57")
    s.dump(s.pa(pages[2]), 0x10, "surface 2's page")
    t.blit(b_fill_buf(b, 0xC00, 0x10, 0x01), resources=[(bad, 0), (b, 0)], fault=True, label="mapping ID 99")
    cmd_delete_resource(s, 1, x12, label="37 on a mapper texture")
    t.blit(b_copy_tex_buf(x12, b, 0xC10, 256, 1024, (1, 1, 1)), resources=[(x12, 0), (b, 0)],
           label="made again after 37")
    t.dump(b_va + 0xC00, 0x20, "B+0xc00")
    t.delete()
    s.events("end")
    return s


@scenario(setups=SETUPS)
def resources_mapper_texture_records():
    """Type 11 records and validateIOSurfaceDescriptor's refusals over a
    one-plane surface, each its own object listed in its own exec: a short
    record and a short header; an unknown commandType (the surface is
    released); short 12, 47 and 57 operations; memoryless 12 ("Invalid
    storage mode (3)") and 47 descriptors; a nonzero plane ("Non zero plane
    in non-planar IOSurface"); a descriptor wider than the surface's rows;
    rows not aligned to iosurfaceTextureAlignmentBytes (surface 3, rows of
    260 bytes) for a shader-read texture and for a read-write one (the
    read-only alignment is used only when usage & 7 is ShaderRead); taller
    than the surface, or an array of two slices; two mip levels; a 3D texture
    of depth 2; a texture buffer ("must be of type MTLTextureType2D, linear
    MTLTextureType2DArray or MTLTextureTypeTextureBuffer", though the check
    refuses exactly TextureBuffer); 47 and 57 descriptors wider than the
    rows. Then the good ones."""
    s = Scenario("resources_mapper_texture_records", setups=SETUPS, iosfc=True,
                 doc=resources_mapper_texture_records.__doc__)
    s.bring_up()
    d, pages = _surfaces(s, {2: (W, H, 4, b"BGRA", None), 3: (W, H, 4, b"BGRA", 260)})
    t = RTask(s, pages=64)
    good = mapper_ref_rec(2) + iosurface_op(12, _bgra())

    def raw(record, n):
        va = t.ol._rec
        t.write(va, record)
        t.ol._rec = (va + len(record) + 15) // 16 * 16
        return t.ol.add_raw(abi.T_MAPPER_TEXTURE, n, va)

    M = lambda *a, **k: t.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(k.pop("mid", 2)) + iosurface_op(*a, **k))
    cases = [
        (raw(good, 7), True),
        (raw(good, 8 + 7), True),
        (t.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(2) + op_hdr(13) + bytes(0x28)), True),
        (raw(good, 8 + 8 + 0x27), True),
        (raw(mapper_ref_rec(2) + iosurface_op(47, _bgra()), 8 + 8 + 0x27), True),
        (raw(mapper_ref_rec(2) + iosurface_op(57, desc2(fmt=BGRA8Unorm, w=W, h=H, options=OPT_SHARED)),
             8 + 8 + 0x2f), True),
        (M(12, desc1(fmt=BGRA8Unorm, w=W, h=H, options=0x30, usage=4)), True),
        (M(47, desc1(fmt=BGRA8Unorm, w=W, h=H, options=0x30, usage=4)), True),
        (M(12, _bgra(), plane=1), True),
        (M(12, _bgra(w=128)), True),
        (M(12, _bgra(usage=1), mid=3), True),
        (M(12, _bgra(usage=3), mid=3), True),
        (M(12, _bgra(h=32)), True),
        (M(12, _bgra(tt=TT_2DArray, array=2)), True),
        (M(12, _bgra(mips=2)), True),
        (M(12, _bgra(tt=TT_3D, d=2)), True),
        (M(12, _bgra(tt=TT_TextureBuffer, h=1)), True),
        (M(47, _bgra(w=128)), True),
        (M(57, desc2(fmt=BGRA8Unorm, w=128, h=H, options=OPT_SHARED, usage=3)), True),
        (M(12, _bgra(usage=1)), False),
        (M(47, _bgra(), rotation=0), False),
    ]
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    for i, (r, faults) in enumerate(cases):
        t.blit(b_fill_buf(b, i * 0x10, 0x10, i + 1), resources=[(r, 0), (b, 0)], fault=faults, label="object %d" % r)
    t.dump(b_va, 0x10 * len(cases), "probe buffer: which execs ran")
    t.delete()
    s.events("end")
    return s


def _death(name, doc, desc):
    s = Scenario(name, setups=SETUPS, iosfc=True, doc=doc)
    s.bring_up()
    _surfaces(s, {2: (W, H, 4, b"BGRA", None)})
    t = RTask(s)
    x = t.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(2) + iosurface_op(12, desc))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_buf(b, 0, 0x10, 0x01), resources=[(x, 0), (b, 0)], fault=True, label="the texture")
    s.events("end")
    return s


@scenario(setups=SETUPS)
def resources_mapper_texture_array():
    """A 64x8 2D array of two slices over a 64x16 surface: it passes
    the manager's validate_iosurface_descriptor (8 x 2 rows fit), and Metal
    aborts making it. The abort is inside Metal, so the host dies in every
    setup, an isolated task's included. The last command."""
    return _death("resources_mapper_texture_array", resources_mapper_texture_array.__doc__,
                  _bgra(tt=TT_2DArray, h=8, array=2))


@scenario(setups=("isolate",), host=True)
def resources_mapper_placement():
    """Mapper-ref textures in a service task with placements (opcode 68): one
    with a zero entry, which is not written back (a mapper-ref texture writes
    no placement); 12 and 47 with entries {index, the gpuResourceID Metal
    gives}, made at that index; 12 and 47 with a wrong gpuResourceID, which
    fail ("Failed to correctly place texture") and fault their exec.
    Host-keyed: resource IDs are the host's."""
    from scenarios.textures_placement import install_with_placements
    s = Scenario("resources_mapper_placement", setups=("isolate",), host=True, iosfc=True,
                 doc=resources_mapper_placement.__doc__)
    s.bring_up()
    _surfaces(s, {2: (W, H, 4, b"BGRA", None)})
    t = RTask(s)
    M = lambda ct: t.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(2) + iosurface_op(ct, _bgra()))
    cases = [(M(12), (0, 0), False), (M(12), (0x40, 0x40), False), (M(47), (0x41, 0x41), False),
             (M(12), (0x42, 0x1234), True), (M(47), (0x43, 0x1234), True)]
    b, b_va = t.shared(PAGE)
    t.define()
    pl = install_with_placements(t, {r: p for r, p, _ in cases})
    for i, (r, _, faults) in enumerate(cases):
        t.blit(b_copy_tex_buf(r, b, 0x100 + 0x10 * i, 256, 1024, (1, 1, 1)), b_fill_buf(b, 0x10 * i, 4, i + 1),
               resources=[(b, 0)], fault=faults, label="object %d" % r)
    t.dump(b_va, 0x150, "probe buffer")
    t.dump(pl, 16 * len(t.ol.entries), "placement list")
    t.delete()
    s.events("end")
    return s
