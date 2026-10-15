"""Resource paths the per-area scenarios leave: a resource listed twice in
one exec or one synchronize, records whose lengths no host can back, the
resource getters 314 reaches on every kind, and deletion of placed
objects (docs/testing.md, sections 4.10, 6.11, 6.19, 6.20).

An exec's prepare collects every listed entry whose resource needs a prepare
before it prepares any (in-process and in the service), and its completion
collects every entry with the synchronize flag whose resource needs a
synchronize before it synchronizes any; opcode 53 does the same. So a
resource listed twice is prepared or synchronized twice, and the second call
finds the work done and returns early, for every kind of resource."""
from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.resources_build import (Depth32Float_Stencil8, RGBA8Unorm, RTask, b_copy_buf_buf, b_copy_buf_tex,
                                    b_copy_tex_buf, b_fill_buf, b_fill_tex_color, buffer_rec, cmd_discard,
                                    cmd_synchronize, desc1, heap_buffer_rec, heap_rec, simple_dim, texture_rec)
from pgtest.scenario import Scenario
from pgtest.task import seg

PATTERN = bytes((i * 7 + 3) & 0xFF for i in range(0x100))
FULL = (4, 4, 1)


def _kinds(t):
    """A private buffer, a shared buffer, a private heap with a buffer at
    0x100, a 4x4 RGBA8 texture and a 4x4 Depth32Float_Stencil8 texture,
    each over patterned guest pages. Returns {name: (ref, va)}."""
    from scenarios.textures_dual_plane import DEPTH, STENCIL, ds_dims
    out = {}
    va = t.area(PAGE, contents=PATTERN)
    out["private buffer"] = (t.add(abi.T_BUFFER, buffer_rec(PAGE, va // PAGE, 1)), va)
    va = t.area(PAGE, contents=PATTERN[::-1])
    out["shared buffer"] = (t.add(abi.T_BUFFER, buffer_rec(PAGE, va // PAGE)), va)
    va = t.area(4 * PAGE, contents=PATTERN * 4)
    h = t.add(abi.T_HEAP, heap_rec(va // PAGE, 4 * PAGE, 1))
    out["heap"] = (h, va)
    out["heap buffer"] = (t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(h, 0x100, PAGE)), va + 0x100)
    va = t.area(PAGE, contents=PATTERN[:64])
    out["texture"] = (t.add(abi.T_TEXTURE, texture_rec(PAGE, va // PAGE, [simple_dim(4, 4)], desc1(usage=0x13))), va)
    va = t.area(PAGE, contents=DEPTH)
    t.write(va + 0x100, STENCIL)
    out["two-plane"] = (t.add(abi.T_DUAL_PLANE_TEXTURE, texture_rec(
        PAGE, va // PAGE, ds_dims(), desc1(fmt=Depth32Float_Stencil8, usage=3))), va)
    return out


def _twice(refs, flags):
    return [e for r in refs for e in ((r, flags), (r, flags))]


@scenario
def resources_gap_listed_twice():
    """Each kind (a private buffer, a shared buffer, a private heap and a
    buffer in it, a normal texture, a two-plane Depth32Float_Stencil8
    texture) listed twice in one exec. The shared buffer is discarded by 63
    first, so that it needs a prepare. The first entry prepares each, the
    second finds it prepared: a copy out of each shows the guest bytes, paged
    in once. Then an exec writes each, every entry listed twice with
    invalidateGuest + synchronize (flags 6): each is synchronized once and
    its guest pages show the write. Then an exec writes each with
    invalidateGuest only (flags 2), and 53 names every one twice: the second
    synchronize of each finds its guest copy valid. In the service the
    buffers and the heap are detach resources over guest memory (CPU-paged),
    whose writes are in guest memory at once."""
    s = Scenario("resources_gap_listed_twice", doc=resources_gap_listed_twice.__doc__)
    s.bring_up()
    t = RTask(s, pages=96)
    k = _kinds(t)
    b, b_va = t.shared(PAGE)
    t.write(b_va + 0x800, bytes(range(0x40, 0x50)))
    t.define()
    t.install()
    refs = [r for r, _ in k.values()]
    t.blit(b_fill_buf(b, 0xF00, 0x10, 0xEE), resources=[(k["shared buffer"][0], 0), (b, 0)],
           label="make the shared buffer")
    cmd_discard(s, 1, [k["shared buffer"][0]], label="63 on the shared buffer")
    t.blit(b_copy_buf_buf(k["private buffer"][0], b, 0, 0x000, 0x20),
           b_copy_buf_buf(k["shared buffer"][0], b, 0, 0x020, 0x20),
           b_copy_buf_buf(k["heap buffer"][0], b, 0, 0x040, 0x20),
           b_copy_tex_buf(k["texture"][0], b, 0x80, 16, 64, FULL),
           b_copy_tex_buf(k["two-plane"][0], b, 0xC0, 16, 64, FULL, options=1),
           b_copy_tex_buf(k["two-plane"][0], b, 0x100, 4, 16, FULL, options=2),
           resources=_twice(refs, 0) + [(b, 0)], label="each listed twice")
    t.dump(b_va, 0x110, "B: each resource's guest bytes")
    t.blit(b_fill_buf(k["private buffer"][0], 0x10, 0x10, 0xA1),
           b_fill_buf(k["heap buffer"][0], 0, 0x10, 0xB2),
           b_fill_tex_color(k["texture"][0], (2, 2, 1), (0, 1.0, 0, 1.0), RGBA8Unorm),
           b_copy_buf_tex(b, k["two-plane"][0], 0x800, 4, 16, FULL, options=2),
           resources=_twice(refs, 6) + [(b, 0)], label="write each, listed twice with flags 6")
    t.dump(k["private buffer"][1], 0x20, "private buffer pages")
    t.dump(k["heap"][1] + 0x100, 0x20, "heap pages at 0x100")
    t.dump(k["texture"][1], 0x40, "texture pages")
    t.dump(k["two-plane"][1] + 0x100, 0x10, "stencil plane pages")
    t.blit(b_fill_buf(k["private buffer"][0], 0x10, 0x10, 0xC3),
           b_fill_buf(k["heap buffer"][0], 0, 0x10, 0xD4),
           b_fill_tex_color(k["texture"][0], (2, 2, 1), (0, 0, 1.0, 1.0), RGBA8Unorm),
           b_copy_buf_tex(b, k["two-plane"][0], 0x0, 4, 16, FULL, options=2),
           resources=[(r, 2) for r in refs] + [(b, 0)], label="write each, invalidateGuest only")
    t.dump(k["private buffer"][1], 0x20, "private buffer pages before 53")
    cmd_synchronize(s, 1, [r for r in refs for _ in (0, 1)], label="53 naming each twice")
    t.dump(k["private buffer"][1], 0x20, "private buffer pages after 53")
    t.dump(k["heap"][1] + 0x100, 0x20, "heap pages at 0x100 after 53")
    t.dump(k["texture"][1], 0x40, "texture pages after 53")
    t.dump(k["two-plane"][1] + 0x100, 0x10, "stencil plane pages after 53")
    t.delete()
    s.events("end")
    return s


def probe_each(s, t, cases, b, b_va, label="probe buffer: which execs ran"):
    """Lists each object in its own exec beside the probe buffer b, whose
    slot i the exec fills; cases: [(label, ref, faults)]."""
    from pgtest.task import seg
    for i, (lab, r, faults) in enumerate(cases):
        t.reset_cmdbufs()
        cb = t.cmdbuf(seg(abi.ENC_BLIT, b_fill_buf(b, 0x10 * i, 0x10, (i + 1) & 0xFF)))
        t.exec3([cb], resources=[(r, 0), (b, 0)], fault=faults, label=lab)
        if faults:
            s.fault_state(lab)
            s.resume_root()
    t.dump(b_va, 0x10 * len(cases), label)


@scenario
def resources_gap_huge_lengths():
    """Buffer and heap records whose lengths no host can back (2^60 and 2^36
    bytes), each listed in its own exec; every exec faults, each for its
    setup's reason. In-process: a private buffer's newBufferWithLength:
    returns nil, a shared buffer's pages are not in the task and its creation
    throws ("Failed to get guest pointer"), a private (legacy) heap's
    newHeapWithDescriptor: returns nil, and a shared heap is refused for not
    being private. In the service every buffer and heap is a detach resource:
    its temporary allocation of 2^60 bytes fails ("Failed to make temporary
    allocation"); one of 2^36 bytes is allocated, and Metal then makes no
    heap ("Failed to create placed heap") or no buffer."""
    s = Scenario("resources_gap_huge_lengths", doc=resources_gap_huge_lengths.__doc__)
    s.bring_up()
    t = RTask(s)
    va = t.area(PAGE)
    H = 1 << 60
    G = 1 << 36
    cases = [
        ("private buffer of 2^60 bytes", t.add(abi.T_BUFFER, buffer_rec(H, va // PAGE, 1)), True),
        ("shared buffer of 2^60 bytes", t.add(abi.T_BUFFER, buffer_rec(H, va // PAGE)), True),
        ("private heap of 2^60 bytes", t.add(abi.T_HEAP, heap_rec(va // PAGE, H, 1)), True),
        ("shared heap of 2^60 bytes", t.add(abi.T_HEAP, heap_rec(va // PAGE, H, 0)), True),
        ("private heap of 2^36 bytes", t.add(abi.T_HEAP, heap_rec(va // PAGE, G, 1)), True),
        ("shared heap of 2^36 bytes", t.add(abi.T_HEAP, heap_rec(va // PAGE, G, 0)), True),
        ("private buffer of 2^36 bytes", t.add(abi.T_BUFFER, buffer_rec(G, va // PAGE, 1)), True),
    ]
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    probe_each(s, t, cases, b, b_va)
    t.delete()
    s.events("end")
    return s


def _view_probe(s, t, view, base, b, b_va, off, label):
    """Copies the view's texels to B+off in an exec listing base and b."""
    t.blit(b_copy_tex_buf(view, b, off, 16, 64, FULL), resources=[(base, 0), (b, 0)], label=label)
    t.dump(b_va + off, 0x40, "B+%#x" % off)


@scenario
def resources_gap_shared_texture_view():
    """A shared texture (type 16) over a task 0 backing, and a pixel-format
    view of it (type 8, commandType 7). A shared texture supports discard, so
    its view is a discardable one, whose discard does nothing. A copy out
    through the view shows task 0's bytes; 314 takes Resource::get_resource
    of the shared texture and of the view; 63 on both, 62 and 60 on the view
    each complete, and the view still reads the same texels."""
    from scenarios.textures_shared import kernel_and_user, define_both
    from pgtest.resources_build import (BGRA8Unorm, b_sync_resource, cmd_replace_physical, cmd_sync_discard,
                                        ser_view_format, shared_backing_rec, shared_texture_rec)
    s = Scenario("resources_gap_shared_texture_view", doc=resources_gap_shared_texture_view.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    kb_va = k.area(PAGE, contents=PATTERN[:64])
    kb = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, kb_va // PAGE, [simple_dim(4, 4)],
                                                         desc1(options=0x20, usage=3)))
    st = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kb))
    v = t.add(abi.T_SERIALIZER_TEXTURE, ser_view_format(st, BGRA8Unorm))
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    _view_probe(s, t, v, st, b, b_va, 0, "copy out through the view")
    t.blit(b_sync_resource(st), b_sync_resource(v), b_fill_buf(b, 0x40, 0x10, 0x31),
           resources=[(st, 0), (b, 0)], label="314 on the shared texture and on the view")
    t.dump(b_va + 0x40, 0x10, "B+0x40: the exec ran")
    cmd_discard(s, 1, [v, st], label="63 on the view and the shared texture")
    _view_probe(s, t, v, st, b, b_va, 0x80, "the view after 63")
    cmd_sync_discard(s, 1, [v], label="62 on the view")
    _view_probe(s, t, v, st, b, b_va, 0xC0, "the view after 62")
    cmd_replace_physical(s, 1, v, label="60 on the view")
    _view_probe(s, t, v, st, b, b_va, 0x100, "the view after 60")
    t.delete()
    k.delete()
    s.events("end")
    return s


@scenario(setups=("map", "isolate"))
def resources_gap_mapper_views():
    """A mapper-ref texture (type 11) and a mapper-ref buffer (type 15) over
    mapped surface 2, and a pixel-format view of the texture. A mapper
    texture does not support discard, so its view is a plain serializer
    texture: 63 and 62 on the view forget it and the next exec makes it
    again; 60 on it calls its discard, which throws ("Attempting to discard a
    view resource"), and 60 faults. 63 on the mapper texture forgets it too,
    and it is made again. 53 on the mapper texture asks needsSync, which
    reads its guest validity (always YES). 314 takes Resource::get_resource
    of the mapper texture and the mapper buffer. The mapper buffer listed
    twice in one exec is prepared twice; the second finds its buffer
    made."""
    from scenarios.resources_mapper import _surfaces, _bgra
    from pgtest.resources_build import (BGRA8Unorm, b_sync_resource, cmd_replace_physical, cmd_sync_discard,
                                        iosurface_op, mapper_ref_rec, ser_view_format)
    s = Scenario("resources_gap_mapper_views", setups=("map", "isolate"), iosfc=True,
                 doc=resources_gap_mapper_views.__doc__)
    s.bring_up()
    d, pages = _surfaces(s, {2: (64, 16, 4, b"BGRA", None)})
    t = RTask(s)
    m = t.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(2) + iosurface_op(12, _bgra()))
    mb = t.add(abi.T_MAPPER_BUFFER, mapper_ref_rec(2))
    v = t.add(abi.T_SERIALIZER_TEXTURE, ser_view_format(m, BGRA8Unorm))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(v, b, 0, 256, 1024, FULL), b_copy_buf_buf(mb, b, 0, 0x400, 0x20),
           resources=[(m, 0), (mb, 0), (mb, 0), (b, 0)], label="through the view; the mapper buffer listed twice")
    t.dump(b_va, 0x10, "B: the view's texels")
    t.dump(b_va + 0x400, 0x20, "B+0x400: the mapper buffer's bytes")
    t.blit(b_sync_resource(m), b_sync_resource(mb), b_sync_resource(v), b_fill_buf(b, 0x40, 0x10, 0x31),
           resources=[(m, 0), (mb, 0), (b, 0)], label="314 on the mapper texture, the mapper buffer, the view")
    t.dump(b_va + 0x40, 0x10, "B+0x40: the exec ran")
    cmd_synchronize(s, 1, [m, mb], label="53 on the mapper texture and buffer")
    cmd_discard(s, 1, [v], label="63 on the view")
    s.mem(s.pa(pages[2]), b"\x6B" * 0x10)
    t.blit(b_copy_tex_buf(v, b, 0x80, 256, 1024, (1, 1, 1)), resources=[(m, 0), (b, 0)], label="the view after 63")
    t.dump(b_va + 0x80, 0x10, "B+0x80")
    cmd_sync_discard(s, 1, [v], label="62 on the view")
    t.blit(b_copy_tex_buf(v, b, 0xC0, 256, 1024, (1, 1, 1)), resources=[(m, 0), (b, 0)], label="the view after 62")
    t.dump(b_va + 0xC0, 0x10, "B+0xc0")
    cmd_replace_physical(s, 1, v, fault=True, label="60 on the view")
    s.fault_state("60 on the view")
    s.resume_root()
    cmd_discard(s, 1, [m], label="63 on the mapper texture")
    t.blit(b_copy_tex_buf(m, b, 0x100, 256, 1024, (1, 1, 1)), resources=[(m, 0), (b, 0)],
           label="the mapper texture after 63")
    t.dump(b_va + 0x100, 0x10, "B+0x100")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def resources_gap_ref_texture_view():
    """A texture over an IOSurface backing in task 0 (type 5) and a
    pixel-format view of it. A ref texture does not support discard, so its
    view is a plain serializer texture: 60 on the view throws from its
    discard and faults. 314 takes Resource::get_resource of the ref texture.
    Then 54 deletes the backing from task 0's list (the ref texture keeps
    it), a fill through the ref texture with invalidateGuest only (flag 2)
    leaves the backing's guest copy invalid, and rvg_device_finish_suspend
    flushes: task 0 no longer holds the backing, and task 1's flush finds the
    ref texture needing a synchronize (it reads its backing's validity) and
    evicts it, which does nothing. The fill never reaches the backing's pages.
    In-process only: in the service a type 5 texture aborts
    (textures_backing_ref_service)."""
    from scenarios.textures_shared import BGRA, kernel_and_user, define_both
    from pgtest.resources_build import (BGRA8Unorm, OPT_SHARED, b_sync_resource, backing_rec, cmd_replace_physical,
                                        iosurface_op, ref_texture_rec, ser_view_format)
    s = Scenario("resources_gap_ref_texture_view", setups=("blocks", "map"), doc=resources_gap_ref_texture_view.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    bk_va = k.area(PAGE, contents=bytes(0x100) + PATTERN[:64])
    bk = k.add(abi.T_IOSURFACE_BACKING, backing_rec(PAGE, bk_va // PAGE, BGRA, [(0x100, 4, 4, 16, 4)]))
    r = t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=iosurface_op(12, desc1(fmt=0x50, options=OPT_SHARED,
                                                                                    usage=3))))
    v = t.add(abi.T_SERIALIZER_TEXTURE, ser_view_format(r, BGRA8Unorm))
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    t.blit(b_copy_tex_buf(v, b, 0, 16, 64, FULL), resources=[(r, 0), (b, 0)], label="through the view")
    t.dump(b_va, 0x40, "B: the view's texels")
    t.blit(b_sync_resource(r), b_sync_resource(v), b_fill_buf(b, 0x40, 0x10, 0x31),
           resources=[(r, 0), (b, 0)], label="314 on the ref texture and the view")
    t.dump(b_va + 0x40, 0x10, "B+0x40: the exec ran")
    cmd_replace_physical(s, 1, v, fault=True, label="60 on the view")
    s.fault_state("60 on the view")
    s.resume_root()
    from pgtest.resources_build import cmd_delete_backing
    cmd_delete_backing(s, bk, label="54: task 0 forgets the backing")
    t.blit(b_fill_tex_color(r, (2, 2, 1), (1.0, 0, 0, 1.0), BGRA8Unorm), resources=[(r, 2), (b, 0)],
           label="fill through the ref texture, invalidateGuest only")
    k.dump(bk_va + 0x100, 0x40, "the backing's pages before the suspend")
    s.api("willSuspend")
    s.api("finishSuspend", slot="s")
    k.dump(bk_va + 0x100, 0x40, "the backing's pages after finishSuspend")
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def resources_gap_placement_delete():
    """A service task with argument buffers takes a placement list (opcode
    68). When its lists are first mapped, the task's rebuild reserves
    objects, walking every entry: a type 7 entry with a nonzero resourceID is
    reserved only if its record is a sampler (commandType 3), so a
    depth-stencil record there (commandType 4) and a type 7 entry whose
    record lies past the task's end reserve nothing. A texture and a buffer
    with zero entries are made by an exec and write their placement back (a
    resourceIndex and gpuResourceID, a GPU address: host values). Then
    deleting either clears its entry: 37 on the texture and 40 (operation
    1000) on the buffer each delete the object and clear its placement, which
    zeroes the entry. An exec makes both again, writing their entries anew.
    Host-keyed: the written values are the host's."""
    from scenarios.textures_placement import install_with_placements
    from pgtest.exec_objects import depth_stencil_record
    from pgtest.resources_build import cmd_delete_object, cmd_delete_resource
    s = Scenario("resources_gap_placement_delete", setups=("isolate",), host=True,
                 doc=resources_gap_placement_delete.__doc__)
    s.bring_up()
    t = RTask(s)
    x = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc1(usage=3)))
    bf, bf_va = t.shared(PAGE, contents=PATTERN)
    ds = t.add(abi.T_SERIALIZED, depth_stencil_record())
    far = t.ol.add_raw(abi.T_SERIALIZED, 0x28, t.length + PAGE)
    b, b_va = t.shared(PAGE)
    t.define()
    pl = install_with_placements(t, {ds: (0x70, 0x70), far: (0x71, 0x71)})
    t.dump(pl, 16 * len(t.ol.entries), "placement list before")
    make = lambda label, off: t.blit(b_copy_tex_buf(x, b, off, 16, 64, FULL), b_copy_buf_buf(bf, b, 0, off + 0x40, 0x10),
                                     resources=[(x, 0), (bf, 0), (b, 0)], label=label)
    make("make the texture and the buffer", 0)
    t.dump(pl, 16 * len(t.ol.entries), "placement list: written back")
    cmd_delete_resource(s, 1, x, label="37 on the texture")
    cmd_delete_object(s, 1, 1000, bf, label="40, operation 1000, on the buffer")
    t.dump(pl, 16 * len(t.ol.entries), "placement list: both entries cleared")
    make("make both again", 0x80)
    t.dump(pl, 16 * len(t.ol.entries), "placement list: written back again")
    t.dump(b_va, 0xD0, "B")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",))
def resources_gap_icb():
    """Indirect command buffers (type 20) in the service, which prepares and
    synchronizes them on the CPU. A compute ICB listed twice in one exec is
    prepared once: the second prepare finds it host-valid; executed, it
    dispatches double_gid into the output buffer. Listed with invalidateGuest
    (flag 2), then named twice by 53, it is written back once and the second
    synchronize finds its guest copy valid. Then binds by GPU address
    (reference ~0) that no buffer holds: the service registers every buffer's
    GPU address and looks a bind's address up as the registered address at or
    below it. 0xfffffffffffffff0 finds the highest buffer, and its offset is
    past that buffer's end ("Buffer too small"); after 37 deletes every
    buffer, the addresses stay registered (a delete forgets only the object),
    and the same address finds a buffer that no longer exists ("Resource not
    found for GPU address"). Either way the bind is skipped and the ICB (one
    reset command) is prepared. Only the output buffer is observed: the ICB's
    write-back holds GPU addresses, the host's."""
    from scenarios.icb import ICBTask, sync_refs
    from pgtest.icb_cmds import compute_cmd
    from pgtest.resources_build import cmd_delete_resource, b_sync_resource
    s = Scenario("resources_gap_icb", setups=("isolate",), doc=resources_gap_icb.__doc__)
    s.bring_up()
    w = ICBTask(s)
    O = w.out
    icb = w.icb([compute_cmd(pso=w.cpso, binds=[(O, 0, 0)], grid=(8, 1, 1), tg=(8, 1, 1))])
    HIGH = 0xFFFFFFFFFFFFFFF0
    j1 = w.icb([compute_cmd(ctype=0, binds=[(0xFFFFFFFF, HIGH, 0)])])
    j2 = w.icb([compute_cmd(ctype=0, binds=[(0xFFFFFFFF, HIGH, 0)])])
    w.start()
    w.run_compute(icb, 0, 1, res=w.res(icb) + [(icb, 0)], label="listed twice, executed")
    w.out_dump("double_gid", 0x20)
    w.clear_out()
    w.run_compute(icb, 0, 1, flags=abi.RES_INVALIDATE_GUEST, label="executed, invalidateGuest")
    sync_refs(s, w.t.tid, [icb, icb], label="53 naming it twice")
    w.out_dump("double_gid again", 0x20)
    w.t.exec_segments([seg(abi.ENC_BLIT, b_sync_resource(0))], resources=w.res(j1),
                      label="a bind past the highest buffer's end")
    for r in (w.out, w.tb, w.rng):
        cmd_delete_resource(s, w.t.tid, r, label="37 on buffer %d" % r)
    w.t.exec_segments([seg(abi.ENC_BLIT, b_sync_resource(0))], resources=[(j2, 0)],
                      label="a bind to a deleted buffer's address")
    s.r(abi.R_FAULT_MASK, "no fault")
    s.events("end")
    return s
