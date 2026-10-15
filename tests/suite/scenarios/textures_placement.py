"""Placement entries (W8; docs/testing.md, sections 6.11, 6.20): a service
task with argument buffers takes a placement list beside its object list
(opcode 68), one 16-byte entry {u64 resourceIndex or GPU address, u64
gpuResourceID} per object. In-process tasks never get one.

A texture whose entry has a nonzero resourceIndex is made at that index, and
its gpuResourceID must then equal the entry's ("Failed to correctly place
texture"); every texture made writes its index and ID back. A buffer or heap
whose entry's address is 0 writes its GPU address back. The values written
are the host's Metal device's; the scenarios are host-keyed."""
from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.resources_build import (BGRA8Unorm, RGBA8Unorm, Depth32Float_Stencil8, OPT_SHARED, TT_2D, RTask,
                                    b_copy_buf_buf, b_copy_tex_buf, b_fill_buf, b_sync_resource, buffer_rec, desc1,
                                    desc2,
                                    heap_buffer_rec, heap_rec, ser_in_heap, ser_memoryless, ser_on_buffer,
                                    ser_view_format, ser_view_type, simple_dim, texture_rec)
from pgtest.scenario import Scenario

S8 = abi.T_SERIALIZER_TEXTURE
PINNED = 0x10000400000
FULL = (4, 4, 1)


def install_with_placements(t, places):
    """Writes the object list and a placement list on its own page, entry i
    from places.get(i, (0, 0)), and sets both with opcode 68. Returns the
    placement list's VA."""
    n = len(t.ol.entries)
    va = t.reserve(PAGE)
    t.write(va, b"".join(abi.PLACEMENT.pack(*places.get(i, (0, 0))) for i in range(n)))
    t.ol.write()
    t.set_object_and_placement_list(0, va, t.ol.length(), 16 * n)
    return va


@scenario(setups=("isolate",), host=True)
def textures_placement_written():
    """Every kind of texture made with a zero placement entry writes its
    resourceIndex and gpuResourceID into the entry: a normal texture (v1
    and v2 descriptors), a two-plane one, a memoryless one, three views, a
    texture over a buffer and one in a heap; a buffer and a heap write
    their GPU address (host values). Entries of objects never made stay
    zero."""
    s = Scenario("textures_placement_written", setups=("isolate",), host=True, doc=textures_placement_written.__doc__)
    s.bring_up()
    t = RTask(s, pages=96)
    from scenarios.textures_dual_plane import ds_dims
    x1 = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc1(usage=0x13)))
    x2 = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc2(), command_type=52))
    d = t.add(abi.T_DUAL_PLANE_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, ds_dims(),
                                                    desc1(fmt=Depth32Float_Stencil8, usage=3)))
    ml = t.add(S8, ser_memoryless(desc1(options=0x30, usage=4)))
    v7 = t.add(S8, ser_view_format(x1, BGRA8Unorm))
    v8 = t.add(S8, ser_view_type(x1, RGBA8Unorm, TT_2D))
    v27 = t.add(S8, ser_view_type(x1, RGBA8Unorm, TT_2D, swizzle=(5, 4, 3, 2)))
    sh, _ = t.shared(PAGE)
    ob = t.add(S8, ser_on_buffer(sh, 0, 16, desc1(options=OPT_SHARED)))
    h = t.add(abi.T_HEAP, heap_rec(t.area(4 * PAGE) // PAGE, 4 * PAGE, 1))
    hb = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(h, 0, PAGE))
    ht = t.add(S8, ser_in_heap(h, PAGE, desc1(usage=3)))
    never = t.add(abi.T_TEXTURE, texture_rec(PAGE, 8, [simple_dim(4, 4)], desc1()))
    b, b_va = t.shared(PAGE)
    t.define()
    pl = install_with_placements(t, {})
    t.blit(b_copy_tex_buf(x1, b, 0, 16, 64, FULL), b_copy_tex_buf(x2, b, 0, 16, 64, FULL),
           b_copy_tex_buf(v7, b, 0, 16, 64, FULL), b_copy_tex_buf(v8, b, 0, 16, 64, FULL),
           b_copy_tex_buf(v27, b, 0, 16, 64, FULL), b_copy_tex_buf(ob, b, 0, 16, 64, FULL),
           b_copy_tex_buf(ht, b, 0, 16, 64, FULL), b_fill_buf(hb, 0, 4, 1), b_sync_resource(ml),
           resources=[(x1, 0), (x2, 0), (d, 0), (sh, 0), (h, 0), (hb, 0), (b, 0)], label="make them")
    t.dump(pl, 16 * len(t.ol.entries), "placement list")
    t.delete()
    s.events("end")
    return s


def _placed_objects(t, forced):
    """One object of each placeable kind whose creation reads its placement
    entry; returns [(label, ref)]. forced: the entry each gets, by label."""
    from scenarios.textures_dual_plane import ds_dims
    base = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc1(usage=0x13)))
    sh, _ = t.shared(PAGE)
    h = t.add(abi.T_HEAP, heap_rec(t.area(4 * PAGE) // PAGE, 4 * PAGE, 1))
    objs = [
        ("normal, v1", abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc1())),
        ("normal, v2", abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc2(),
                                                   command_type=52)),
        ("two-plane", abi.T_DUAL_PLANE_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, ds_dims(),
                                                             desc1(fmt=Depth32Float_Stencil8, usage=3))),
        ("memoryless", S8, ser_memoryless(desc1(options=0x30, usage=4))),
        ("view 7", S8, ser_view_format(base, BGRA8Unorm)),
        ("view 8", S8, ser_view_type(base, RGBA8Unorm, TT_2D)),
        ("view 27", S8, ser_view_type(base, RGBA8Unorm, TT_2D, swizzle=(5, 4, 3, 2))),
        ("on a buffer, v1", S8, ser_on_buffer(sh, 0, 16, desc1(options=OPT_SHARED))),
        ("on a buffer, v2", S8, ser_on_buffer(sh, 0x100, 16, desc2(options=OPT_SHARED), v2=True)),
        ("in a heap, v1", S8, ser_in_heap(h, 0, desc1(usage=3))),
        ("in a heap, v2", S8, ser_in_heap(h, PAGE, desc2(usage=3), v2=True)),
    ]
    return base, sh, h, [(label, t.add(typ, rec)) for label, typ, rec in objs]


def _forced(name, doc, good):
    """Each placeable texture kind with a forced resourceIndex (0x40 + i),
    its resourceID either the one Metal gives (good) or not; each made by
    314 SynchronizeResource in its own exec, whose probe slot shows whether
    it ran."""
    s = Scenario(name, setups=("isolate",), host=True, doc=doc)
    s.bring_up()
    t = RTask(s, pages=128)
    base, sh, h, objs = _placed_objects(t, None)
    extra = {}
    if good:
        from pgtest.exec_objects import depth_stencil_record, sampler_record
        pb_va = t.area(PAGE, fill=0x3C)
        extra = {t.add(abi.T_SERIALIZED, sampler_record()): (0x60, 0x60),
                 t.add(abi.T_SERIALIZED, depth_stencil_record()): (0x61, 0x61),
                 t.add(abi.T_ICB, bytes(16)): (0x62, 0x62),
                 t.add(abi.T_MAPPER_TEXTURE, bytes(16)): (0x63, 0x63),
                 t.add(abi.T_FUNCTION, bytes(16)): (0x64, 0x64),
                 t.add(abi.T_BUFFER, buffer_rec(PAGE, pb_va // PAGE)): (PINNED, 0),
                 t.add(abi.T_HEAP, heap_rec(t.area(4 * PAGE) // PAGE, 4 * PAGE, 1)): (PINNED + 0x100000, 0)}
    b, b_va = t.shared(PAGE)
    t.define()
    places = dict(extra)
    for i, (label, ref) in enumerate(objs):
        idx = 0x40 + i
        places[ref] = (idx, idx if good else 0x1234)
    pl = install_with_placements(t, places)
    t.dump(pl, 16 * len(t.ol.entries), "placement list before")
    for i, (label, ref) in enumerate(objs):
        probe = b_sync_resource(ref) if good else b_copy_tex_buf(ref, b, 0x100, 16, 64, FULL)
        t.blit(probe, b_fill_buf(b, 0x10 * i, 0x10, i + 1),
               resources=[(base, 0), (sh, 0), (h, 0), (b, 0)], fault=not good, label=label)
    if good:
        pb = [r for r, v in extra.items() if v == (PINNED, 0)][0]
        t.blit(b_copy_buf_buf(pb, b, 0, 0x200, 0x10), resources=[(pb, 0), (b, 0)], label="the pinned buffer")
        t.dump(b_va + 0x200, 0x10, "the pinned buffer's bytes")
    t.dump(b_va, 0x10 * len(objs), "probe buffer: which execs ran")
    t.dump(pl, 16 * len(t.ol.entries), "placement list after")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def textures_placement_forced():
    """Every placeable texture kind (normal v1 and v2, two-plane, memoryless,
    views 7, 8 and 27, on a buffer and in a heap, v1 and v2) with an entry
    {resourceIndex 0x40 + i, gpuResourceID the same}: each is made at that
    index, and its gpuResourceID, equal to its index on this host, matches.
    The first exec maps the lists, and the task's rebuild reserves every
    nonzero-ID texture index first."""
    return _forced("textures_placement_forced", textures_placement_forced.__doc__, True)


@scenario(setups=("isolate",), host=True)
def textures_placement_misplaced():
    """The same kinds with entries {0x40 + i, 0x1234}: each texture is made
    at its index, its gpuResourceID is not 0x1234, and creation fails
    ("Failed to correctly place texture"): every exec faults, and the
    entries keep the guest's values."""
    return _forced("textures_placement_misplaced", textures_placement_misplaced.__doc__, False)


@scenario(setups=("isolate",), host=True)
def textures_placement_shared():
    """Shared textures (type 16) from a task 0 backing, in a service task
    with placements: one with a zero entry writes its resourceIndex and
    gpuResourceID back; one with resourceIndex 0x50 is made at that index
    (Metal's newSharedTextureWithHandle:withResourceIndex:), with no check of
    the resourceID, and writes both back. Task 0 is in-process and never has
    placements, so a backing's placement paths are never taken."""
    from scenarios.textures_shared import kernel_and_user
    from pgtest.resources_build import shared_backing_rec, shared_texture_rec
    s = Scenario("textures_placement_shared", setups=("isolate",), host=True, doc=textures_placement_shared.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    px = bytes((0x40 + i) & 0xFF for i in range(64))
    kb_va = k.area(PAGE, contents=px)
    kb = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, kb_va // PAGE, [simple_dim(4, 4)],
                                                         desc1(options=0x20, usage=3)))
    s0 = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kb))
    s1 = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kb))
    b, b_va = t.shared(PAGE)
    k.define()
    k.install()
    t.define()
    pl = install_with_placements(t, {s1: (0x50, 0x7777)})
    t.blit(b_copy_tex_buf(s0, b, 0, 16, 64, FULL), b_copy_tex_buf(s1, b, 0x40, 16, 64, FULL),
           resources=[(s0, 0), (s1, 0), (b, 0)], label="read the backing through both")
    t.dump(b_va, 0x80, "B")
    t.dump(pl, 16 * len(t.ol.entries), "placement list")
    t.delete()
    k.delete()
    s.events("end")
    return s
