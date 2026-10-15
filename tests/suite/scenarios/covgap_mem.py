"""Memory, resources, textures and the resource manager: the reachable
paths the rest of the suite leaves unreached there (docs/testing.md,
sections 6.10, 6.11, 6.17, 6.19, 6.23).

pgtest/covgap_mem_util.py lays out guest RAM with a hole (R0, 16 pages of
nothing, R1) and gives RTask's resource helpers to tasks whose leaves the
scenario chooses. Unless a docstring says otherwise a scenario runs in all
three setups, and where they differ the difference is locked.
"""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.covgap_mem_util import HOLE_PAGE, HOLED, XRTask, own_leaves
from pgtest.registry import scenario
from pgtest.resources_build import (BC1_RGBA, BGRA8Unorm, OPT_SHARED, RTask, b_copy_buf_buf, b_copy_tex_buf, b_fill_buf,
                                    backing_rec, desc1, dimension, heap_buffer_rec, heap_rec,
                                    iosurface_op, mip, ref_texture_rec, texture_rec)
from pgtest.scenario import RAM_BASE, RAM_SIZE, Scenario
from pgtest.task import seg
from pgtest.tasks_lib import exec3_payload

BGRA = 0x42475241  # 'BGRA', an IOSurface pixel format


@scenario
def covgap_mem_heap_over_hole():
    """A private heap (type 13, flags 1) of one page whose only leaf names
    a guest page no RAM range holds, and a buffer placed in it (type 14),
    listed with the heap in an exec that fills the heap buffer. The heap
    cannot reach its pages in any setup and the exec faults: in-process the
    legacy heap's first prepare makes its transfer buffer through the
    host's map_memory (blocks) or the memory map ("Incomplete range", map);
    in the service the detach heap's first prepare asks Metal to replace
    the heap's backing with the page's host address, which the service's
    memory map does not have, and "Failed to replace heap backing" is
    thrown. A heap over RAM, listed after, pages and is filled."""
    s = Scenario("covgap_mem_heap_over_hole", ram=HOLED, doc=covgap_mem_heap_over_hole.__doc__)
    s.bring_up()
    leaves = own_leaves(s, 10)
    leaves[10] = HOLE_PAGE
    t = XRTask(s, tid=1, pages=11, leaves=leaves)
    h = t.add(abi.T_HEAP, heap_rec(10, PAGE, flags=1))
    hb = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(h, 0, PAGE))
    ok = t.add(abi.T_HEAP, heap_rec(9, PAGE, flags=1))
    okb = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(ok, 0, PAGE))
    b, b_va = t.shared(PAGE, fill=0x00)
    t.define()
    t.install()
    t.blit(b_fill_buf(hb, 0, 0x10, 0x5A), resources=[(h, 0), (hb, 0), (b, 0)], fault=True,
           label="a heap over the hole")
    t.blit(b_fill_buf(okb, 0, 0x10, 0x6B), b_copy_buf_buf(okb, b, 0, 0, 0x20),
           resources=[(ok, 6), (okb, 0), (b, 0)], label="a heap over RAM")
    t.dump(b_va, 0x20, "B")
    t.dump(9 * PAGE, 0x20, "the heap's pages after flags 6")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",))
def covgap_mem_reply_in_hole():
    """An info reply (451, heap texture size and align) into a buffer
    record whose page is one no RAM range holds. The reply's writable cursor is
    made from the record alone, without making the buffer, and the service
    builds it from the host address its memory map gives each page: none for
    this one (MemoryMap::virtual_address_for_physical returns NULL), so the
    cursor's range starts at address 0, and the write there is a death in the
    task's own code, which ends only the task: the exec faults. Isolate only:
    in-process every reply aborts the host (info_reply_aborts_host). The exec
    is the scenario's last command."""
    from pgtest.exec_objects import info_heap_size
    s = Scenario("covgap_mem_reply_in_hole", setups=("isolate",), ram=HOLED,
                 doc=covgap_mem_reply_in_hole.__doc__)
    s.bring_up()
    leaves = own_leaves(s, 9)
    leaves[9] = HOLE_PAGE
    t = XRTask(s, tid=1, pages=10, leaves=leaves)
    r = t.add(abi.T_BUFFER, struct.pack("<QIB3x", PAGE, 9, 0))
    t.define()
    t.install()
    cb = t.cmdbuf(seg(abi.ENC_INFO, info_heap_size(r, 0x10, fmt=70, w=16, h=16, usage=1, options=0x20)))
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb]), fault=True, label="a reply into the hole")
    s.fault_state("after the reply")
    return s


@scenario(setups=("isolate",))
def covgap_mem_icb_args_past_page():
    """An ICB write-back whose command's draw arguments lie wholly in the
    next VA page, whose guest page is not adjacent to the first's. A render
    ICB of 56 commands with a guest layout of commandSize 0x128 and draw
    arguments at 0xf8: command 55 starts at 0x3f98, 0x68 bytes before the
    page's end, so its arguments start 0x90 bytes into the second page. The
    write at that offset into the command's sub-cursor skips the whole of
    the first range (the offset past the range's end) and lands at +0x90 of
    the far page; the guest page after the first one in RAM (0x55) is
    untouched. Listed with invalidateGuest it is prepared and marked
    guest-invalid, the guest's copy set to 0xee around the boundary, and
    SynchronizeResources (53) writes the commands back through the
    service's writable cursor. Isolate only: in-process there is no
    writable cursor."""
    from pgtest.exec_objects import ICB_DRAW, icb_descriptor, icb_record
    from pgtest.icb_cmds import layout
    from scenarios.icb import ICBTask, sync_refs
    s = Scenario("covgap_mem_icb_args_past_page", setups=("isolate",), doc=covgap_mem_icb_args_past_page.__doc__)
    s.bring_up()
    w = ICBTask(s, pages=96)
    size, n = 0x128, 56
    cmds = bytearray()
    for i in range(n):
        c = bytearray(size)
        struct.pack_into("<III", c, 0, 1, 0, w.rpso)
        c[0xF8:0xF8 + 0x22] = struct.pack("<HQQQQ", 3, i, 3, 1, 0)
        cmds += c
    va = w.t.reserve(2 * PAGE)
    w.t.write(va, bytes(cmds))
    lay = layout("render", commandArgumentsOffset=0xF8, commandSize=size)
    icb = w.t.ol.add(abi.T_ICB, icb_record(n * size, va // PAGE, icb_descriptor(ICB_DRAW, vbufs=1, fbufs=1), lay,
                                           max_count=n))
    far = s.alloc_page()
    vpn2 = va // PAGE + 1
    near = w.t.data + vpn2
    s.mem(s.pa(far), bytes(cmds[PAGE:]))
    s.u32(s.pa(w.t.pt_root, 4 * vpn2), s.pfn(far) | abi.PTE_WRITABLE)
    w.start()
    w.t.exec_segments([seg(abi.ENC_BLIT)], resources=w.res() + [(icb, abi.RES_INVALIDATE_GUEST)],
                      label="prepared, guest invalidated")
    s.fault_state("prepared")
    s.fill(s.pa(near - 1, PAGE - 0x100), 0x100, 0xEE)
    s.fill(s.pa(far), 0xC0, 0xEE)
    s.fill(s.pa(near), 0xC0, 0x55)
    sync_refs(s, 1, [icb], label="synchronize")
    s.dump(s.pa(near - 1, PAGE - 0x100), 0x100, "the end of the ICB's first page")
    s.dump(s.pa(far), 0xC0, "the start of its second page")
    s.dump(s.pa(near), 0xC0, "the guest page after its first page")
    s.events("end")
    return s


@scenario
def covgap_mem_mip_slice_fits():
    """An 8x8 BC1_RGBA texture (type 2, 4x4 blocks of 8 bytes) of two
    levels whose dimension declares a slice of 0x30 bytes, which both
    levels' ends (0x20 and 0x28) fit: the compressed path's mip-end check
    passes on its first test, where every other compressed texture of the
    suite declares no slice size. Listing it pages both levels in; copies
    of each level out to a buffer show the guest bytes."""
    s = Scenario("covgap_mem_mip_slice_fits", doc=covgap_mem_mip_slice_fits.__doc__)
    s.bring_up()
    t = RTask(s)
    px = bytes((i * 7 + 3) & 0xFF for i in range(0x30))
    va = t.area(PAGE, contents=px)
    dims = [dimension([mip(0, 32, 16, 8, 8), mip(0x20, 8, 8, 4, 4)], slice_bytes=0x30, block=(4, 4, 1, 8),
                      compressed=1)]
    tex = t.add(abi.T_TEXTURE, texture_rec(PAGE, va // PAGE, dims, desc1(fmt=BC1_RGBA, w=8, h=8, mips=2)))
    b, b_va = t.shared(PAGE, fill=0x00)
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(tex, b, 0, 16, 32, (8, 8, 1)), b_copy_tex_buf(tex, b, 0x40, 8, 8, (4, 4, 1), level=1),
           resources=[(tex, 0), (b, 0)], label="both levels out")
    t.dump(b_va, 0x50, "B")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("map", "isolate"))
def covgap_mem_iosfc_straddle():
    """The IOSurface host device over guest RAM whose second range is 0x6000
    bytes: its second 16 KiB page is half outside it. A surface whose page
    table names that page cannot be mapped: the device's memory map finds
    the range holding the page's start and logs "Get straddles map ranges",
    and map writes NoMemory. A surface over the range's first page, and one
    over RAM, map. Map and isolate setups only, as iosfc.py: apple-gfx
    creates the IOSurface host device only with a memory map, and the device
    does not depend on the RvgDevice's setup."""
    from pgtest.iosfc_guest import CMD_MAP, IOSFC, desc_v1, pte
    x = RAM_BASE + RAM_SIZE + 16 * PAGE
    s = Scenario("covgap_mem_iosfc_straddle", setups=("map", "isolate"), iosfc=True,
                 ram=[(RAM_BASE, RAM_SIZE), (x, 0x6000)], doc=covgap_mem_iosfc_straddle.__doc__)
    d = IOSFC(s)
    d.start()
    xp = (x - RAM_BASE) // PAGE
    roots = []
    for page in (xp + 1, xp, None):
        pt = s.alloc_page()
        if page is None:
            page = s.alloc_page()
        s.u32(s.pa(pt), pte(s.pfn(page)))
        roots.append(s.pfn(pt))
    for mid, root in enumerate(roots, start=1):
        d.put_desc(mid, desc_v1(root, 16, 8))
    idx = d.submit((CMD_MAP, 1), (CMD_MAP, 2), (CMD_MAP, 3))
    for i, label in zip(idx, ("the straddling page", "the range's first page", "a page in RAM")):
        d.status([i], label)
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def covgap_mem_backing_zero_width():
    """An IOSurface backing (type 4) in task 0 whose one plane is 0 texels
    wide: its checks pass (a row of 0 texels fits any stride, and 0 rows
    fit the allocation), and the exec in task 1 that lists a type 5 texture
    over the backing kills the host as the backing makes its plane's
    texture: Metal fails its assertion "MTLTextureDescriptor has width of
    zero" (SIGABRT, measured), so the nil check after it is never reached.
    In-process only: in the service GpuTask's backing_for_id ends the task
    (textures_backing_ref_service). The exec is the scenario's last
    command."""
    from scenarios.textures_shared import define_both, kernel_and_user
    s = Scenario("covgap_mem_backing_zero_width", setups=("blocks", "map"),
                 doc=covgap_mem_backing_zero_width.__doc__)
    s.bring_up()
    k, t = kernel_and_user(s)
    bk_va = k.area(PAGE, fill=0x11)
    bk = k.add(abi.T_IOSURFACE_BACKING, backing_rec(PAGE, bk_va // PAGE, BGRA, [(0, 0, 4, 16, 4)]))
    r = t.add(abi.T_BACKING_TEXTURE, ref_texture_rec(bk, op=iosurface_op(12, desc1(fmt=BGRA8Unorm,
                                                                                    options=OPT_SHARED))))
    b, b_va = t.shared(PAGE)
    define_both(k, t)
    t.blit(b_fill_buf(b, 0, 0x10, 0x22), resources=[(r, 0), (b, 0)], label="the host dies here")
    return s


@scenario(setups=("isolate",))
def covgap_mem_icb_writeback_without_pipeline():
    """An ICB write-back of render commands that have no pipeline: a render
    ICB of three commands in the suite's render layout, the first a draw
    with a pipeline, the second a reset (type 0) and the third a draw, both
    with pipeline reference 0, so the prepare sets no pipeline on them.
    Listed with invalidateGuest it is prepared and marked guest-invalid,
    the guest's copy is set to 0xee, and SynchronizeResources (53) writes
    each command back: for a render command the pipeline's unique
    identifier is looked up whatever the command's type, and one no
    pipeline was registered under writes reference 0. Isolate only:
    in-process there is no writable cursor."""
    from pgtest.icb_cmds import CMD, render_cmd
    from scenarios.icb import ICBTask, sync_refs
    s = Scenario("covgap_mem_icb_writeback_without_pipeline", setups=("isolate",),
                 doc=covgap_mem_icb_writeback_without_pipeline.__doc__)
    s.bring_up()
    w = ICBTask(s)
    icb = w.icb([render_cmd(pso=w.rpso), render_cmd(ctype=0, pso=0), render_cmd(pso=0)], kind="render")
    va, size = w.icbs[icb]
    w.start()
    w.t.exec_segments([seg(abi.ENC_BLIT)], resources=w.res() + [(icb, abi.RES_INVALIDATE_GUEST)],
                      label="prepared, guest invalidated")
    s.fault_state("prepared")
    w.t.fill(va, 3 * CMD, 0xEE)
    sync_refs(s, 1, [icb], label="synchronize")
    for i in range(3):
        w.t.dump(va + i * CMD, 0x10, "command %d's type, barrier and pipeline" % i)
    s.events("end")
    return s


