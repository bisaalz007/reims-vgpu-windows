"""Paging edge cases common to buffers and textures: resources whose host
and guest copies are both marked invalid, SynchronizeResource (314) on each
kind of resource, and resources whose pages lie outside their task
(docs/testing.md, sections 4.10, 6.11).

Opcode 52 sets and clears a resource's validity bits as asked, with no check.
A prepare with neither copy valid logs "Trying to prepare from invalid guest
resource" and pages the guest bytes in anyway. Every synchronize is gated on
needsSync (guest invalid, host valid), so the "Trying to synchronize from
invalid host resource" logs are unreachable.
"""
from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.resources_build import (BC1_RGBA, Depth32Float_Stencil8, RGBA8Unorm, BGRA8Unorm, OPT_SHARED, RTask,
                                    b_copy_buf_buf, b_copy_buf_tex, b_copy_tex_buf, b_fill_buf, b_fill_tex_color,
                                    b_sync_resource, buffer_rec, cmd_discard, cmd_invalidate, cmd_synchronize, desc1,
                                    dimension, heap_buffer_rec, heap_rec, mip, ser_on_buffer, ser_view_format,
                                    simple_dim, texture_rec)
from pgtest.scenario import Scenario

FULL = (4, 4, 1)
PATTERN = bytes((i * 7 + 3) & 0xFF for i in range(0x100))
BOTH = (1, 0, 1, 0)  # clearHostValid, clearGuestValid


@scenario
def resources_both_invalid():
    """A private buffer, a normal texture, a two-plane Depth32Float_Stencil8
    texture and a private heap (through a buffer in it; "Trying to prepare
    from invalid guest heap"), each paged in, written on the GPU, then marked
    with neither copy valid (52: clearHostValid and clearGuestValid). 53 then
    does nothing: none of them needs a synchronize (the host copy is not
    valid), so the guest pages keep their bytes. Marked invalid again, the
    next exec prepares each anyway: the guest bytes, rewritten by the test,
    are paged in (the "invalid guest" log). In the service the buffer is a
    detach buffer over the guest pages, whose validity setters do nothing:
    its fill is in the guest pages at once."""
    from scenarios.textures_dual_plane import DEPTH, STENCIL, ds_dims
    s = Scenario("resources_both_invalid", doc=resources_both_invalid.__doc__)
    s.bring_up()
    t = RTask(s)
    p_va = t.area(PAGE, contents=PATTERN)
    p = t.add(abi.T_BUFFER, buffer_rec(PAGE, p_va // PAGE, 1))
    x_va = t.area(PAGE, contents=PATTERN[:64])
    x = t.add(abi.T_TEXTURE, texture_rec(PAGE, x_va // PAGE, [simple_dim(4, 4)], desc1()))
    d_va = t.area(PAGE, contents=DEPTH)
    t.write(d_va + 0x100, STENCIL)
    d = t.add(abi.T_DUAL_PLANE_TEXTURE, texture_rec(PAGE, d_va // PAGE, ds_dims(),
                                                    desc1(fmt=Depth32Float_Stencil8, usage=3)))
    h_va = t.area(4 * PAGE, contents=PATTERN)
    h = t.add(abi.T_HEAP, heap_rec(h_va // PAGE, 4 * PAGE, 1))
    hb = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(h, 0, PAGE))
    b, b_va = t.shared(PAGE)
    t.write(b_va + 0x400, bytes(range(0xA0, 0xB0)))
    t.define()
    t.install()
    t.blit(b_fill_buf(p, 0, 0x10, 0x5A), b_fill_tex_color(x, (2, 2, 1), (1.0, 0, 0, 1.0), RGBA8Unorm),
           b_copy_buf_tex(b, d, 0x400, 4, 16, FULL, options=2), b_fill_buf(hb, 0, 0x10, 0x6B),
           resources=[(p, 0), (x, 0), (d, 0), (h, 0), (hb, 0), (b, 0)], label="page in and write all four")
    cmd_invalidate(s, 1, [(p,) + BOTH, (x,) + BOTH, (d,) + BOTH, (h,) + BOTH], label="52: neither copy valid")
    cmd_synchronize(s, 1, [p, x, d, h], label="53: nothing needs a synchronize")
    t.dump(p_va, 0x20, "private buffer pages")
    t.dump(x_va, 0x40, "texture pages")
    t.dump(d_va + 0x100, 0x10, "stencil plane pages")
    t.dump(h_va, 0x10, "heap pages")
    t.write(h_va, b"\x44" * 0x10)
    t.write(p_va, b"\x11" * 0x10)
    t.write(x_va, b"\x22" * 0x10)
    t.write(d_va + 0x100, b"\x03" * 0x10)
    cmd_invalidate(s, 1, [(p,) + BOTH, (x,) + BOTH, (d,) + BOTH, (h,) + BOTH], label="52 again")
    t.blit(b_copy_buf_buf(p, b, 0, 0, 0x20), b_copy_tex_buf(x, b, 0x40, 16, 64, FULL),
           b_copy_tex_buf(d, b, 0x80, 4, 16, FULL, options=2), b_copy_buf_buf(hb, b, 0, 0x90, 0x10),
           resources=[(p, 0), (x, 0), (d, 0), (h, 0), (hb, 0), (b, 0)], label="prepare from an invalid guest copy")
    t.dump(b_va, 0xA0, "B")
    t.delete()
    s.events("end")
    return s


def _sync_kinds(name, doc, setups, discarded_faults):
    s = Scenario(name, setups=setups, doc=doc)
    s.bring_up()
    t = RTask(s, pages=80)
    p = t.add(abi.T_BUFFER, buffer_rec(PAGE, t.area(PAGE) // PAGE, 1))
    sh, _ = t.shared(PAGE)
    h = t.add(abi.T_HEAP, heap_rec(t.area(4 * PAGE) // PAGE, 4 * PAGE, 1))
    hb = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(h, 0, PAGE))
    x = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc1(usage=0x13)))
    v = t.add(abi.T_SERIALIZER_TEXTURE, ser_view_format(x, BGRA8Unorm))
    ob = t.add(abi.T_SERIALIZER_TEXTURE, ser_on_buffer(sh, 0, 16, desc1(options=OPT_SHARED)))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    cases = [("a private buffer", p, [(p, 0)], False), ("a shared buffer", sh, [(sh, 0)], False),
             ("a heap buffer", hb, [(h, 0), (hb, 0)], False), ("a heap", h, [(h, 0)], True),
             ("a texture", x, [(x, 0)], False), ("a view", v, [(x, 0)], False),
             ("a texture over a buffer", ob, [(sh, 0)], False)]
    for i, (label, ref, res, faults) in enumerate(cases):
        t.blit(b_sync_resource(ref), b_fill_buf(b, 0x10 * i, 0x10, i + 1), resources=res + [(b, 0)], fault=faults,
               label="314 on " + label)
    cmd_discard(s, 1, [sh], label="63 on the shared buffer")
    t.blit(b_sync_resource(sh), b_fill_buf(b, 0x80, 0x10, 0x80), resources=[(b, 0)], fault=discarded_faults,
           label="314 on a discarded shared buffer, not listed")
    t.dump(b_va, 0x90, "probe buffer: which execs ran")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def resources_synchronize_resource_kinds_inproc():
    """314 SynchronizeResource takes Resource::get_resource of any resource:
    a private buffer, a shared buffer, a heap buffer, a normal texture, a
    view and a texture over a buffer each give their Metal resource; a heap
    has no Resource::get_resource (its kind raises Unrecognized), and the
    exec faults. A shared buffer that 63 discarded has no buffer (its
    get_resource raises: the buffer was invalidated): that exec faults too.
    Each exec ends with a fill of the probe buffer, so its slot shows
    whether it ran."""
    return _sync_kinds("resources_synchronize_resource_kinds_inproc",
                       resources_synchronize_resource_kinds_inproc.__doc__, ("blocks", "map"), True)


@scenario(setups=("isolate",))
def resources_synchronize_resource_kinds_service():
    """As resources_synchronize_resource_kinds_inproc, in the service: the
    heap (a detach heap) faults the same way, but the shared buffer is a
    detach buffer, whose discard detaches its backing and keeps the buffer,
    so 314 on it after 63 succeeds."""
    return _sync_kinds("resources_synchronize_resource_kinds_service",
                       resources_synchronize_resource_kinds_service.__doc__, ("isolate",), False)


@scenario
def resources_pages_outside_task():
    """Resources whose record names pages past the task's end: each is made
    (nothing reads its pages at creation), and the exec that lists it faults
    when its prepare asks the task for a transfer buffer: the task throws
    ("Failed to get guest pointer", in-process and in the service), so the
    resources' own "Failed to create buffer" checks of a nil buffer are never
    reached. A two-plane texture of a block-compressed format is refused at
    creation ("Multi plane compressed textures are unsupported")."""
    from scenarios.resources import creation_case
    from scenarios.textures_dual_plane import ds_dims
    bc = [dimension([mip(0, 32, 16, 8, 8)], block=(4, 4, 1, 8), compressed=1),
          dimension([mip(0, 32, 16, 8, 8)], base=0x100, block=(4, 4, 1, 8), compressed=1)]
    T12 = abi.T_DUAL_PLANE_TEXTURE
    cases = [
        (abi.T_TEXTURE, texture_rec(PAGE, 500, [simple_dim(4, 4)], desc1()), None, True),
        (abi.T_TEXTURE_MIPGEN, texture_rec(PAGE, 500, [simple_dim(4, 4)], desc1()), None, True),
        (T12, texture_rec(PAGE, 500, ds_dims(), desc1(fmt=Depth32Float_Stencil8, usage=3)), None, True),
        (T12, texture_rec(PAGE, 8, bc, desc1(fmt=BC1_RGBA, w=8, h=8)), None, True),
        (abi.T_BUFFER, buffer_rec(PAGE, 500, 1), None, True),
        (abi.T_HEAP, heap_rec(500, 4 * PAGE, 1), None, True),
    ]
    return creation_case("resources_pages_outside_task", resources_pages_outside_task.__doc__, cases)


@scenario
def resources_record_outside_task():
    """Object-list entries whose record lies past the task's end, or runs
    over it: no cursor can be made over the record ("Failed to create memory
    cursor"), so each object fails, and the exec listing it faults. Each is
    listed in its own exec; the last exec lists a good buffer."""
    s = Scenario("resources_record_outside_task", doc=resources_record_outside_task.__doc__)
    s.bring_up()
    t = RTask(s, pages=16)
    end = 16 * PAGE
    refs = [t.ol.add_raw(abi.T_BUFFER, 16, 500 * PAGE), t.ol.add_raw(abi.T_BUFFER, 16, end - 8),
            t.ol.add_raw(abi.T_TEXTURE, 0x80, end), t.ol.add_raw(abi.T_HEAP, 16, 1 << 40)]
    good = t.add(abi.T_BUFFER, buffer_rec(PAGE, 9))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    for i, r in enumerate(refs + [good]):
        t.blit(b_fill_buf(b, 0x10 * i, 0x10, i + 1), resources=[(r, 0), (b, 0)], fault=r != good,
               label="object %d" % r)
    t.dump(b_va, 0x50, "probe buffer: which execs ran")
    t.delete()
    s.events("end")
    return s
