"""Textures whose storage lives in the kernel task (task 0) and is used from
another task: IOSurface backings (type 4) and textures over them (type 5),
shared texture backings (types 17, 18) and shared textures made from their
handles (type 16) (docs/testing.md, sections 6.11, 6.19, 6.20).

Task 0 is always in-process, and no exec may name it ("This command may not
reference a kernel task"). Its objects are created when another task asks for
them: a type 5 record finds its backing through the kernel task's object
list, a lookup that aborts in the service; a type 16 record gets its handle
from the kernel task, through the host in the service."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.resources_build import (BGRA8Unorm, OPT_SHARED, RTask, b_copy_tex_buf, b_fill_buf, b_fill_tex_color,
                                    backing_rec, cmd_delete_backing, cmd_discard, cmd_invalidate, cmd_synchronize,
                                    desc1, desc2, iosurface_op, ref_texture_rec)
from pgtest.scenario import Scenario

BGRA = 0x42475241  # 'BGRA', an IOSurface pixel format
FULL = (4, 4, 1)


def kernel_and_user(s, kpages=32):
    """Task 0 (the kernel task) and task 1, each with its own object list.
    Returns (k, t); define and install both with define_both."""
    return RTask(s, tid=0, pages=kpages), RTask(s, tid=1)


def define_both(k, t):
    k.define()
    k.install()
    t.define()
    t.install()


@scenario(setups=("blocks", "map"))
def textures_backing_ref():
    """An IOSurface backing in task 0 (type 4: one BGRA plane of 4x4 at
    +0x100 of its pages) and three textures over it in task 1 (type 5,
    commandTypes 12, 47 and 57). Each pages in through the backing (a ref
    texture forwards prepare, synchronize and validity to it): a copy out
    shows the guest bytes; a fill with flags 6 lands in task 0's pages;
    opcode 53 on task 0 names the backing itself; invalidateHost re-reads.
    Opcode 54 deletes the backing from task 0's list (a taskID other than 0
    faults, "This command may only reference a kernel task"); the textures
    keep theirs. 63 on a ref texture does nothing (it does not support
    discard); 63 on task 0 discards the backing's transfer buffer; with
    neither copy marked valid by 52, the next prepare logs "Trying to prepare
    backing from invalid guest" and pages the guest bytes in. In-process
    only: in the service the backing lookup aborts
    (textures_backing_ref_service)."""
    s = Scenario("textures_backing_ref", setups=("blocks", "map"), doc=textures_backing_ref.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    px = bytes((i * 3) & 0xFF for i in range(64))
    bk_va = k.area(PAGE, contents=bytes(0x100) + px)
    bk = k.add(abi.T_IOSURFACE_BACKING, backing_rec(PAGE, bk_va // PAGE, BGRA, [(0x100, 4, 4, 16, 4)]))
    d = desc1(fmt=BGRA8Unorm, options=OPT_SHARED)
    t12 = t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=iosurface_op(12, d)))
    t47 = t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=iosurface_op(47, d)))
    t57 = t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=iosurface_op(57, desc2(fmt=BGRA8Unorm, options=OPT_SHARED))))
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    t.blit(b_copy_tex_buf(t12, b, 0, 16, 64, FULL), b_copy_tex_buf(t47, b, 0x40, 16, 64, FULL),
           b_copy_tex_buf(t57, b, 0x80, 16, 64, FULL), resources=[(t12, 0), (t47, 0), (t57, 0), (b, 0)],
           label="read the backing through three textures")
    t.dump(b_va, 0xC0, "B")
    t.blit(b_fill_tex_color(t12, (2, 2, 1), (1.0, 0, 0, 1.0), BGRA8Unorm), resources=[(t12, 6), (b, 0)],
           label="fill, synchronize")
    k.dump(bk_va + 0x100, 0x40, "the backing's pages")
    t.blit(b_fill_tex_color(t47, (2, 2, 1), (0, 1.0, 0, 1.0), BGRA8Unorm, origin=(2, 2, 0)),
           resources=[(t47, 2), (b, 0)], label="fill, invalidateGuest only")
    cmd_synchronize(s, 1, [t47], label="53 on task 1 through the ref texture")
    k.dump(bk_va + 0x100, 0x40, "the backing's pages after 53")
    t.blit(b_fill_tex_color(t57, (1, 1, 1), (0, 0, 1.0, 1.0), BGRA8Unorm), resources=[(t57, 2), (b, 0)],
           label="fill through 57, invalidateGuest")
    cmd_synchronize(s, 0, [bk], label="53 on task 0, on the backing itself")
    k.dump(bk_va + 0x100, 0x40, "the backing's pages after 53 on the backing")
    k.write(bk_va + 0x100, b"\x77" * 16)
    t.blit(b_copy_tex_buf(t12, b, 0x100, 16, 64, FULL), resources=[(t12, 1), (b, 0)], label="invalidateHost")
    t.dump(b_va + 0x100, 0x40, "B+0x100: re-read")
    cmd_delete_backing(s, bk, tid=1, fault=True, label="54 with task 1")
    s.fault_state("54 with task 1")
    s.resume_root()
    cmd_delete_backing(s, bk, label="54")
    cmd_delete_backing(s, 60, label="54 on an unknown reference")
    k.write(bk_va + 0x100, b"\x66" * 16)
    t.blit(b_copy_tex_buf(t12, b, 0x140, 16, 64, FULL), resources=[(t12, 1), (b, 0)],
           label="the ref texture keeps its backing")
    t.dump(b_va + 0x140, 0x40, "B+0x140")
    cmd_discard(s, 1, [t12], label="63 on a ref texture: it does not support discard")
    t.blit(b_copy_tex_buf(t12, b, 0x180, 16, 64, FULL), resources=[(t12, 0), (b, 0)], label="after 63")
    t.dump(b_va + 0x180, 0x40, "B+0x180")
    cmd_discard(s, 0, [bk], label="63 on task 0, on the backing: its transfer buffer is dropped")
    cmd_invalidate(s, 0, [(bk, 1, 0, 1, 0)], label="52 on task 0: neither of the backing's copies valid")
    k.write(bk_va + 0x100, b"\x55" * 16)
    t.blit(b_copy_tex_buf(t12, b, 0x1C0, 16, 64, FULL), resources=[(t12, 0), (b, 0)],
           label="prepare from an invalid guest copy, through the ref texture")
    t.dump(b_va + 0x1C0, 0x40, "B+0x1c0")
    t.delete()
    k.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",))
def textures_backing_ref_service():
    """A type 5 texture in a service task: the backing lookup aborts in the
    service, so the service dies and the exec faults; every later command on
    that task faults too (docs/testing.md, section 6.20). The host survives.
    Isolate only: this is the service's behaviour."""
    s = Scenario("textures_backing_ref_service", setups=("isolate",), doc=textures_backing_ref_service.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    bk_va = k.area(PAGE)
    bk = k.add(abi.T_IOSURFACE_BACKING, backing_rec(PAGE, bk_va // PAGE, BGRA, [(0, 4, 4, 16, 4)]))
    r = t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=iosurface_op(12, desc1(fmt=BGRA8Unorm, options=OPT_SHARED))))
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    t.blit(b_fill_buf(b, 0, 0x10, 0x11), resources=[(r, 0), (b, 0)], fault=True, label="a type 5 texture in the service")
    t.blit(b_fill_buf(b, 0, 0x10, 0x22), resources=[(b, 0)], fault=True, label="the task after its service died")
    t.dump(b_va, 0x10, "B")
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def textures_backing_records():
    """IOSurface backing (type 4) refusals and variants in task 0, each
    reached from its own type 5 texture in task 1, each listed in its own
    exec: a short record, a plane count of 4 ("Too many planes") or 0
    ("Invalid plane count"), short planes, a stride under width x bytes per
    element and a plane past the allocation, an entry of type 0 and one that
    is a buffer (the lookup requires an IOSurface backing), an unknown
    reference; then each bytes-per-element the switch knows (16, 8, 4, 2, 1)
    and three planes, each viewed as a texture of the matching size. Then
    type 5 refusals over a good backing: a short record, a nonzero task ID, a
    short header, an unknown commandType (12, 47 and 57 are known), short
    12/47/57 operations, memoryless descriptors, a plane the surface lacks
    (Metal returns nil: "Failed to create texture")."""
    s = Scenario("textures_backing_records", setups=("blocks", "map"), doc=textures_backing_records.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s, kpages=48)
    good = [(0, 4, 4, 16, 4)]
    backings = [  # (record, entry length or None, bytes per element for the view)
        (backing_rec(PAGE, 8, BGRA, good), 0x13, 4),
        (backing_rec(PAGE, 8, BGRA, good, count=4), None, 4),
        (backing_rec(PAGE, 8, BGRA, []), None, 4),
        (backing_rec(PAGE, 8, BGRA, good, count=2), None, 4),
        (backing_rec(PAGE, 8, BGRA, [(0, 4, 4, 15, 4)]), None, 4),
        (backing_rec(PAGE, 8, BGRA, [(0x3FC1, 4, 4, 16, 4)]), None, 4),
    ]
    fmts = {16: 123, 8: 113, 4: 70, 2: 30, 1: 10}  # RGBA32Uint, RGBA16Uint, RGBA8Unorm, RG8Unorm, R8Unorm
    for bpe in (16, 8, 4, 2, 1):
        backings.append((backing_rec(PAGE, 8, BGRA, [(0, 4, 4, 4 * bpe, bpe)]), None, bpe))
    backings.append((backing_rec(PAGE, 8, 0x34323076, [(0, 4, 4, 16, 1), (0x100, 2, 2, 16, 2), (0x200, 2, 2, 16, 2)]),
                     None, 1))
    krefs = []
    for rec_, n, bpe in backings:
        if n is None:
            krefs.append((k.add(abi.T_IOSURFACE_BACKING, rec_), bpe))
        else:
            va = k.ol._rec
            k.write(va, rec_)
            k.ol._rec = (va + len(rec_) + 15) // 16 * 16
            krefs.append((k.ol.add_raw(abi.T_IOSURFACE_BACKING, n, va), bpe))
    kinvalid = k.ol.add_raw(0, 0, 0)
    kbuf = k.add(abi.T_BUFFER, struct.pack("<QIB3x", PAGE, 9, 0))
    cases = []  # (ref, faults)
    for i, (r, bpe) in enumerate(krefs):
        cases.append((t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(r, op=iosurface_op(12, desc1(fmt=fmts[bpe],
                                                                                              options=OPT_SHARED)))),
                      i < 6))
    for r in (kinvalid, kbuf, 60):
        cases.append((t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(r, op=iosurface_op(12, desc1(
            fmt=BGRA8Unorm, options=OPT_SHARED)))), True))
    bk4 = krefs[8][0]
    d = desc1(fmt=70, options=OPT_SHARED)
    ml = desc1(fmt=70, options=0x30)

    def raw(record, n):
        va = t.ol._rec
        t.write(va, record)
        t.ol._rec = (va + len(record) + 15) // 16 * 16
        return t.ol.add_raw(abi.T_BACKING_TEXTURE, n, va)

    cases += [
        (raw(ref_texture_rec(bk4, op=iosurface_op(12, d)), 7), True),
        (t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk4, task_id=1, op=iosurface_op(12, d))), True),
        (raw(ref_texture_rec(bk4, op=iosurface_op(12, d)), 8 + 7), True),
        (t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk4, op=struct.pack("<II", 13, 0) + bytes(0x28))), True),
        (raw(ref_texture_rec(bk4, op=iosurface_op(12, d)), 8 + 8 + 0x27), True),
        (raw(ref_texture_rec(bk4, op=iosurface_op(47, d)), 8 + 8 + 0x27), True),
        (raw(ref_texture_rec(bk4, op=iosurface_op(57, desc2(fmt=70, options=OPT_SHARED))), 8 + 8 + 0x2f), True),
        (t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk4, op=iosurface_op(12, ml))), True),
        (t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk4, op=iosurface_op(47, ml))), True),
        (t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk4, op=iosurface_op(57, desc2(fmt=70, options=0x30)))), True),
        (t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk4, op=iosurface_op(12, d, plane=1))), True),
    ]
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    for i, (r, faults) in enumerate(cases):
        t.blit(b_fill_buf(b, i * 0x10, 0x10, (i + 1) & 0xFF), resources=[(r, 0), (b, 0)], fault=faults,
               label="object %d" % r)
    t.dump(b_va, 0x10 * len(cases), "probe buffer: which execs ran")
    t.delete()
    k.delete()
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def textures_backing_bad_element_size():
    """A backing plane of 3 bytes per element has no pixel format (the
    backing's pixel-format mapping logs "Invalid pixel bytes" and returns
    MTLPixelFormatInvalid), and Metal aborts making a texture of it over the
    IOSurface: the host dies at the exec whose type 5 texture reaches the
    backing. In-process only: in the service, the backing lookup aborts first
    (textures_backing_ref_service)."""
    s = Scenario("textures_backing_bad_element_size", setups=("blocks", "map"),
                 doc=textures_backing_bad_element_size.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    bk = k.add(abi.T_IOSURFACE_BACKING, backing_rec(PAGE, 8, BGRA, [(0, 4, 4, 12, 3)]))
    r = t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=iosurface_op(12, desc1(fmt=BGRA8Unorm, options=OPT_SHARED))))
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    t.blit(b_fill_buf(b, 0, 0x10, 0x11), resources=[(r, 0), (b, 0)], label="the host dies here")
    return s


@scenario
def textures_shared_texture():
    """A shared texture backing in task 0 (type 17: a 4x4 RGBA8 texture made
    shareable, commandType 48 with a v1 descriptor and 53 with a v2) and
    shared textures in task 1 made from its handle (type 16, commandType 49).
    Asking for the handle pages the backing in from task 0's pages, so a copy
    out of the type 16 texture shows them. A fill of the type 16 texture with
    invalidateGuest marks the backing's guest copy invalid through the kernel
    task; opcode 53 on task 0 then copies the backing out to its pages.
    Opcode 65 deletes a backing; a type 16 texture made after that cannot
    find it. A type 16 texture's discard, prepare and synchronize are empty:
    63 and an exec listing it with flags 7 run and change nothing."""
    s = Scenario("textures_shared_texture", doc=textures_shared_texture.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    from pgtest.resources_build import shared_backing_rec, shared_texture_rec, simple_dim, cmd_delete_shared_backing
    px = bytes((0x40 + i) & 0xFF for i in range(64))
    b1_va = k.area(PAGE, contents=px)
    b1 = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, b1_va // PAGE, [simple_dim(4, 4)],
                                                         desc1(options=0x20, usage=3)))
    b2_va = k.area(PAGE, contents=px[::-1])
    b2 = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, b2_va // PAGE, [simple_dim(4, 4)],
                                                         desc2(options=0x20, usage=3), command_type=53))
    s1 = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(b1))
    s1b = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(b1))
    s2 = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(b2))
    late = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(b2))
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    t.blit(b_copy_tex_buf(s1, b, 0, 16, 64, FULL), b_copy_tex_buf(s2, b, 0x40, 16, 64, FULL),
           resources=[(s1, 0), (s2, 0), (b, 0)], label="read both backings through shared textures")
    t.dump(b_va, 0x80, "B")
    t.blit(b_fill_tex_color(s1, (2, 2, 1), (1.0, 0, 0, 1.0), 70), resources=[(s1, 2), (b, 0)],
           label="fill through the handle, invalidateGuest")
    t.blit(b_copy_tex_buf(s1b, b, 0x80, 16, 64, FULL), resources=[(s1b, 6), (b, 0)],
           label="a second texture from the same handle sees the fill")
    t.dump(b_va + 0x80, 0x40, "B+0x80")
    k.dump(b1_va, 0x40, "backing 1's pages before 53")
    cmd_synchronize(s, 0, [b1, b2], label="53 on task 0")
    k.dump(b1_va, 0x40, "backing 1's pages after 53")
    k.dump(b2_va, 0x40, "backing 2's pages after 53")
    cmd_delete_shared_backing(s, b2, label="65 on backing 2")
    cmd_delete_shared_backing(s, 60, label="65 on an unknown reference")
    k.write(b2_va, b"\x99" * 16)
    t.blit(b_copy_tex_buf(late, b, 0xC0, 16, 64, FULL), resources=[(late, 0), (b, 0)],
           label="a shared texture made after 65")
    t.dump(b_va + 0xC0, 0x40, "B+0xc0")
    cmd_discard(s, 1, [s1], label="63 on a shared texture: its discard does nothing")
    t.blit(b_fill_buf(b, 0x100, 0x10, 0x5D), resources=[(s1, 7), (b, 0)],
           label="a shared texture with flags 7: its prepare and synchronize do nothing")
    t.dump(b_va + 0x100, 0x10, "B+0x100")
    t.delete()
    k.delete()
    s.events("end")
    return s


def two_task_probe(s, k, t, cases, label="probe buffer: which execs ran"):
    """Lists each task 1 object in its own exec beside a probe buffer.
    cases: [(ref, faults)]; a creation that throws faults the exec before
    anything runs, so its probe slot stays zero."""
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    for i, (r, faults) in enumerate(cases):
        t.blit(b_fill_buf(b, i * 0x10, 0x10, (i + 1) & 0xFF), resources=[(r, 0), (b, 0)], fault=faults,
               label="object %d" % r)
    t.dump(b_va, 0x10 * len(cases), label)
    t.delete()
    k.delete()
    s.events("end")
    return s


def _raw(task, typ, record, n):
    va = task.ol._rec
    task.write(va, record)
    task.ol._rec = (va + len(record) + 15) // 16 * 16
    return task.ol.add_raw(typ, n, va)


@scenario
def textures_shared_records():
    """Shared texture backing (types 17, 18) and shared texture (type 16)
    refusals. Each backing in task 0 is reached from its own type 16 texture
    in task 1, each listed in its own exec. Backings: short record, short
    dimension, a mip count of 0 ("Invalid mip level count"), short mips,
    short header, an unknown commandType (48 and 53 are known), short 48 and
    53 operations, memoryless 48 and 53 descriptors, shared storage (Metal
    makes no shareable texture of it: "Failed to create texture"), a
    dimension the texture fails (array length), two mips, a two-plane
    Depth32Float_Stencil8 backing (type 18), a type 18 of a one-plane format,
    a buffer where a backing should be (it has no shared texture handle).
    Shared textures: short record, short header, an unknown commandType (49
    is known), an unknown backing ID. In the service the handle comes through
    the host and a missing one reads "Failed to get shared texture
    handle"."""
    from pgtest.resources_build import shared_backing_rec, shared_texture_rec, simple_dim, dimension, mip
    s = Scenario("textures_shared_records", doc=textures_shared_records.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s, kpages=48)
    T17, T18 = abi.T_SHARED_BACKING1, abi.T_SHARED_BACKING2
    good = shared_backing_rec(PAGE, 8, [simple_dim(4, 4)], desc1(usage=3))
    good2 = shared_backing_rec(PAGE, 8, [simple_dim(4, 4)], desc2(usage=3), command_type=53)
    ds_dims = [dimension([mip(0, 64, 16, 4, 4)]), dimension([mip(0x40, 16, 4, 4, 4)], block=(1, 1, 1, 1))]
    kcases = [
        (T17, good, 11, True),
        (T17, good, 12 + 0x17, True),
        (T17, shared_backing_rec(PAGE, 8, [dimension([], levels=0)], desc1(usage=3)), None, True),
        (T17, good, 12 + 0x18 + 0x23, True),
        (T17, good, 12 + 0x3c + 7, True),
        (T17, shared_backing_rec(PAGE, 8, [simple_dim(4, 4)], desc1(usage=3), command_type=49), None, True),
        (T17, good, 12 + 0x3c + 8 + 0x23, True),
        (T17, good2, 12 + 0x3c + 8 + 0x2b, True),
        (T17, shared_backing_rec(PAGE, 8, [simple_dim(4, 4)], desc1(usage=4, options=0x30)), None, True),
        (T17, shared_backing_rec(PAGE, 8, [simple_dim(4, 4)], desc2(usage=4, options=0x30), command_type=53), None,
         True),
        (T17, shared_backing_rec(PAGE, 8, [simple_dim(4, 4)], desc1(usage=3, options=0)), None, True),
        (T17, shared_backing_rec(PAGE, 8, [simple_dim(4, 4, array=2)], desc1(usage=3)), None, True),
        (T17, shared_backing_rec(PAGE, 8, [simple_dim(8, 8, levels=2)], desc1(w=8, h=8, mips=2, usage=3)), None,
         False),
        (T18, shared_backing_rec(PAGE, 8, ds_dims, desc1(fmt=260, usage=3)), None, False),
        (T18, shared_backing_rec(PAGE, 8, [simple_dim(4, 4), simple_dim(4, 4)], desc1(usage=3)), None, True),
        (abi.T_BUFFER, struct.pack("<QIB3x", PAGE, 9, 0), None, True),
        (T17, good, None, False),
    ]
    cases = []
    for typ, rec_, n, faults in kcases:
        r = k.add(typ, rec_) if n is None else _raw(k, typ, rec_, n)
        cases.append((t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(r)), faults))
    kgood = r
    cases += [
        (_raw(t, abi.T_SHARED_TEXTURE, shared_texture_rec(kgood), 3), True),
        (_raw(t, abi.T_SHARED_TEXTURE, shared_texture_rec(kgood), 11), True),
        (t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kgood, command_type=50)), True),
        (t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(60)), True),
    ]
    return two_task_probe(s, k, t, cases)


@scenario
def textures_shared_backing_dual_plane():
    """A two-plane Depth32Float_Stencil8 shared texture backing (type 18) in
    task 0, used from task 1 through a type 16 texture: a copy out of each
    plane shows task 0's pages (paged in when the handle is asked for); a
    copy into the stencil plane with invalidateGuest marks the backing's
    guest copy invalid, and 53 on task 0 pages both planes out, a plane at a
    time. A type 18 backing of BC1 is refused ("Multi plane compressed shared
    backing texture is unsupported"), so the type 16 texture over it
    fails."""
    from pgtest.resources_build import (BC1_RGBA, Depth32Float_Stencil8, b_copy_buf_tex, dimension, mip,
                                        shared_backing_rec, shared_texture_rec)
    from scenarios.textures_dual_plane import DEPTH, STENCIL, ds_dims
    s = Scenario("textures_shared_backing_dual_plane", doc=textures_shared_backing_dual_plane.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    kb_va = k.area(PAGE, contents=DEPTH)
    k.write(kb_va + 0x100, STENCIL)
    kb = k.add(abi.T_SHARED_BACKING2, shared_backing_rec(PAGE, kb_va // PAGE, ds_dims(),
                                                         desc1(fmt=Depth32Float_Stencil8, usage=3)))
    bc = [dimension([mip(0, 32, 16, 8, 8)], block=(4, 4, 1, 8), compressed=1),
          dimension([mip(0, 32, 16, 8, 8)], base=0x100, block=(4, 4, 1, 8), compressed=1)]
    kc = k.add(abi.T_SHARED_BACKING2, shared_backing_rec(PAGE, k.area(PAGE) // PAGE, bc,
                                                         desc1(fmt=BC1_RGBA, w=8, h=8, usage=1)))
    x = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kb))
    xc = t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(kc))
    b, b_va = t.shared(PAGE)
    t.write(b_va + 0x200, bytes(range(0x90, 0xA0)))
    define_both(k, t)
    t.blit(b_copy_tex_buf(x, b, 0, 16, 64, FULL, options=1), b_copy_tex_buf(x, b, 0x40, 4, 16, FULL, options=2),
           resources=[(x, 0), (b, 0)], label="read both planes through the shared texture")
    t.dump(b_va, 0x50, "B")
    t.blit(b_copy_buf_tex(b, x, 0x200, 4, 16, FULL, options=2), resources=[(x, 2), (b, 0)],
           label="write the stencil plane, invalidateGuest")
    k.dump(kb_va + 0x100, 0x10, "stencil plane pages before 53")
    cmd_synchronize(s, 0, [kb], label="53 on task 0")
    k.dump(kb_va, 0x40, "depth plane pages after 53")
    k.dump(kb_va + 0x100, 0x10, "stencil plane pages after 53")
    t.blit(b_fill_buf(b, 0x100, 0x10, 0x01), resources=[(xc, 0), (b, 0)], fault=True,
           label="a shared texture over a BC1 two-plane backing")
    t.dump(b_va + 0x100, 0x10, "B+0x100")
    t.delete()
    k.delete()
    s.events("end")
    return s
