"""Boundaries, page fields and deaths in the device, its FIFOs, the displays,
EFI, the IOSurface host device and the suspend blob: each scenario locks one,
as its docstring says.
"""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.display_guest import (SS_EXPECTED_ACK, SS_MODE_COUNT, DisplayGuest, display, floats,
                                  gamma_words, p_glyph, p_icc, p_present3, p_props, pattern)
from pgtest.efi_guest import EFI
from pgtest.fifo import ChildFIFO, RootFIFO
from pgtest.fifo_util import child, resume_child
from pgtest.iosfc_guest import CMD_MAP, CMD_TEST, I_RING_SIZE, IOSFC, desc_v1, desc_v2, pte
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.suspend_guest import blob, state
from pgtest.task import GPUTask, Task, rec, seg
from scenarios.display_cursor import glyph_pixels, glyph_sum
from scenarios.iosfc_gaps import _regs
from scenarios.suspend import resume_fresh, suspend, will_resume
from scenarios.suspend_gaps import adopt_root, node_words, running_root

MAPPED = ("map", "isolate")
TRACE = {"trace": True}


# ---- the suspend blob ----

@scenario
def mutgap2_dev_blob_key_at_max():
    """The blob's device-info max key is carried, not checked against its
    pairs: a device-info key 2 under max key 2 is accepted, as under max
    key 3. A pair's key must be one a resume reads (1-42 or 44): keys 0, 43
    and 45 are InvalidContent. The device is never resumed."""
    s = Scenario("mutgap2_dev_blob_key_at_max", doc=mutgap2_dev_blob_key_at_max.__doc__)
    will_resume(s, blob(state(), [(2, 0)], info_max=2), "info key 2, max 2")
    will_resume(s, blob(state(), [(2, 0)], info_max=3), "info key 2, max 3")
    will_resume(s, blob(state(), [(0, 0)]), "info key 0")
    will_resume(s, blob(state(), [(43, 0)]), "info key 43")
    will_resume(s, blob(state(), [(45, 0)]), "info key 45")
    s.r(abi.R_VERSION, "after the last")
    return s


# ---- rvg_device_did_resume ----

@scenario
def mutgap2_dev_resume_running_high_byte():
    """rvg_device_did_resume rebuilds a child FIFO for a node whose u16 running
    word (+0xe) is nonzero, whatever its bytes: node 3 with running 0x100 and a
    NOP between its read count and written is rebuilt and started, the NOP
    runs, and the node's running word reads 1 afterwards. The root FIFO, from
    the blob, runs a NOP."""
    s = Scenario("mutgap2_dev_resume_running_high_byte", stamp_slots=4,
                 doc=mutgap2_dev_resume_running_high_byte.__doc__)
    f = RootFIFO(s, 2, 0x400)
    adopt_root(s, f)
    c = ChildFIFO(s, 3)
    c.setup_node()
    c.cmd(abi.OP_NOP)
    node_words(s, 3, written=c.written, running=0x100, plist=s.pfn(c.list_page))
    will_resume(s, blob(running_root(s, f)), "resume")
    s.api("didResume")
    c.wait_stamp(1, label="node 3's NOP")
    s.fault_state("resumed")
    s.events("resumed")
    s.dump(c.node_gpa(0), abi.CHILD_NODE_SIZE, "node 3")
    s.submit(abi.OP_NOP)
    s.events("end")
    s.dump(f.base_gpa(), 16, "stamp slots 0-3")
    return s


@scenario
def mutgap2_dev_resume_task_255():
    """rvg_device_did_resume recreates every task whose root word is set, ids 0
    to 255: GPU task 255 with a buffer, suspended and resumed on a fresh
    device, runs a blit fill afterwards."""
    s = Scenario("mutgap2_dev_resume_task_255", doc=mutgap2_dev_resume_task_255.__doc__)
    s.bring_up()
    t = GPUTask(s, tid=255, pages=16)
    b, b_va = t.buffer()
    t.define()
    t.install()
    s.dump(s.pa(s.root_page, 4 * 255), 4, "root word 255")
    suspend(s)
    resume_fresh(s)
    s.events("resumed")
    fill = struct.pack("<IQQB3x", b, 0, 0x40, 0xCD)
    t.exec3([t.cmdbuf(seg(abi.ENC_BLIT, rec(306, fill)))], resources=[(b, 5)])
    t.dump(b_va, 0x80, "buffer after the fill")
    s.fault_state("end")
    s.events("end")
    return s


# ---- wake and stamp-wait reports ----

@scenario
def mutgap2_dev_wake_64_while_faulted():
    """A wake (0x1020) of 64 is refused ("invalid fifo index in wake") also
    while the root FIFO is faulted, when the word past the 64 child slots
    (the fault mask) is nonzero: the wake does nothing, the root FIFO resumes
    and a NOP runs."""
    s = Scenario("mutgap2_dev_wake_64_while_faulted", doc=mutgap2_dev_wake_64_while_faulted.__doc__)
    s.bring_up()
    s.submit(99, fault=True, label="unknown opcode")
    s.fault_state("faulted")
    s.w(abi.R_WAKE_CHILD, 64)
    s.r(abi.R_FAULT_OFFSET, "after the wake")
    s.fault_state("after the wake")
    s.resume_root()
    s.submit(abi.OP_NOP)
    s.events("end")
    return s


@scenario
def mutgap2_dev_report_stamp_index_1_dies():
    """A stamp-wait report (0x1028 on a FIFO that is not faulted) looks stamp
    index 1 up as child 1, like every index but 0: child 2 with stamp index 1
    blocks on a barrier, child 1 has stamp index 4, and the report asserts
    that child 1's stamp index is 1: the host dies at the 0x1028 write."""
    s = Scenario("mutgap2_dev_report_stamp_index_1_dies", doc=mutgap2_dev_report_stamp_index_1_dies.__doc__)
    s.bring_up()
    c2 = child(s, 2, stamp_index=1)
    child(s, 1, stamp_index=4)
    st = c2.cmd(abi.OP_NOP, barriers=[(7, 1)])
    c2.doorbell()
    c2.wait_stamp(st, timeout_ms=500, label="child 2 blocked on slot 7")
    resume_child(s, 2)
    s.fault_state("not reached")
    return s


# ---- mappings after a failed resume (run state LeavingSuspendEarly) ----

@scenario
def mutgap2_dev_failed_resume_then_map_task():
    """After a failed rvg_device_will_resume (no state: InvalidContent, run
    state left LeavingSuspendEarly), a MapMemory2 of a task defined before it:
    in the blocks setup the device's call that maps into the host task asserts
    and the host dies at the map. The map and isolate setups map through the
    memory map, never reach that call, and the map's stamp lands with no block
    call."""
    s = Scenario("mutgap2_dev_failed_resume_then_map_task", doc=mutgap2_dev_failed_resume_then_map_task.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=8)
    t.define()
    s.events("defined")
    s.api("willResume", nil=True, label="nil")
    t.map()
    s.events("after the map")
    return s


@scenario
def mutgap2_dev_failed_resume_then_unmap_task():
    """After a failed rvg_device_will_resume (no state), an UnmapMemory of a
    task mapped before it: in the blocks setup the device's call that unmaps
    from the host task asserts and the host dies at the unmap. In the map and
    isolate setups the unmap's stamp lands."""
    s = Scenario("mutgap2_dev_failed_resume_then_unmap_task",
                 doc=mutgap2_dev_failed_resume_then_unmap_task.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=8)
    t.define()
    t.map()
    s.events("mapped")
    s.api("willResume", nil=True, label="nil")
    t.unmap()
    s.events("after the unmap")
    return s


@scenario
def mutgap2_dev_failed_resume_then_efi():
    """After a failed rvg_device_will_resume (no state), programming an EFI
    framebuffer: the length write creates the EFI task and its task mapping
    asserts, in every setup: the host dies at the length write."""
    s = Scenario("mutgap2_dev_failed_resume_then_efi", displays=[display(0)], descriptor=TRACE,
                 doc=mutgap2_dev_failed_resume_then_efi.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    s.api("willResume", nil=True, label="nil")
    e = EFI(s)
    e.write_frame(16, 8, 64)
    e.setup(16, 8, 64)
    s.events("after the length write")
    return s


# ---- display commands ----

@scenario
def mutgap2_dev_present3_task_1():
    """DisplayTransaction3 naming task 1 is refused even with task 0 and a
    good backing: the stamp is signalled, nothing is presented, and the
    page's present count stays 0. The same payload naming task 0 then
    presents."""
    s = Scenario("mutgap2_dev_present3_task_1", displays=[display(0)], doc=mutgap2_dev_present3_task_1.__doc__)
    g = DisplayGuest(s)
    r, va = g.backing(16, 8, pixels=pattern(16, 8))
    g.install()
    g.connect(0, enabled=0xF)
    g.ack(0, 1)
    s.events("connected")
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, r, task=1), label="task 1")
    s.quiesce(100)
    s.events("task 1")
    g.diff_page(0, "task 1")
    s.api("present_counts", display=0)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, r), label="task 0")
    s.wait(cb="newFrame")
    s.events("task 0")
    g.diff_page(0, "task 0")
    s.api("present_counts", display=0)
    return s


@scenario
def mutgap2_dev_icc_below_version():
    """SetGuestICCProfile below version 208 is unsupported: at version 207 a
    nonzero profile VA and length are not stored (+0xfc8 keeps 0) and the
    stamp signals. At version 245 the same command stores them."""
    s = Scenario("mutgap2_dev_icc_below_version", displays=[display(0)], doc=mutgap2_dev_icc_below_version.__doc__)
    g = DisplayGuest(s)
    g.connect(0)
    g.ack(0, 1)
    va = g.t0.blob(b"not an ICC profile" + bytes(46))
    s.w(abi.R_VERSION, 207)
    s.submit(abi.OP_DISPLAY_SET_GUEST_ICC_PROFILE, p_icc(0, 64, va, 64), label="version 207")
    s.fault_state("version 207")
    g.diff_page(0, "version 207")
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    s.submit(abi.OP_DISPLAY_SET_GUEST_ICC_PROFILE, p_icc(0, 64, va, 64), label="version 245")
    g.diff_page(0, "version 245")
    s.events("end")
    return s


@scenario
def mutgap2_dev_properties_below_version():
    """SetProperties below version 40 is unsupported: at version 30 a
    non-identity colour matrix is not written to the page, and the stamp
    signals. At version 40 it is."""
    s = Scenario("mutgap2_dev_properties_below_version", displays=[display(0)],
                 doc=mutgap2_dev_properties_below_version.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0)
    g.ack(0, 1)
    m = floats(0.5, 0, 0, 0.1, 0, 0.25, 0, 0.2, 0, 0, 2.0, 0.3)
    s.w(abi.R_VERSION, 30)
    s.submit(abi.OP_DISPLAY_SET_PROPERTIES, p_props(0, [(0, m)]), label="version 30")
    s.fault_state("version 30")
    g.diff_page(0, "version 30")
    s.w(abi.R_VERSION, 40)
    s.submit(abi.OP_DISPLAY_SET_PROPERTIES, p_props(0, [(0, m)]), label="version 40")
    g.diff_page(0, "version 40")
    s.events("end")
    return s


@scenario
def mutgap2_dev_mode_count_128():
    """A page mode count of exactly 128 (the most the host writes) passes the
    mode-list check's assert: with the guest's count set to 128, a mode-list
    update rewrites it to 1 and raises the update."""
    s = Scenario("mutgap2_dev_mode_count_128", displays=[display(0)], doc=mutgap2_dev_mode_count_128.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0, enabled=0xF)
    g.ack(0, 1)
    s.events("connected")
    s.mem(g.ss(0, SS_MODE_COUNT), struct.pack("<H", 128))
    s.api("set_modes", display=0, modes=[[64, 32, 60]])
    s.wait(gpa=g.ss(0, SS_EXPECTED_ACK), eq=2, label="the update")
    s.events("updated")
    g.diff_page(0, "updated")
    return s


@scenario
def mutgap2_dev_gamma_exact_length():
    """A gamma table whose mapped length is exactly 4 x entryCount is
    accepted: on port 1, which has no display (so nothing reads the table), a
    present with 4 entries and mapped length 16 records the gamma record at
    +0xf80 and counts the present; mapped length 15 is refused and records
    nothing."""
    s = Scenario("mutgap2_dev_gamma_exact_length", descriptor={"port_count": 2}, displays=[display(0)],
                 doc=mutgap2_dev_gamma_exact_length.__doc__)
    g = DisplayGuest(s)
    r, _ = g.backing(16, 8, pixels=pattern(16, 8, 5))
    g.install()
    g.connect(1, enabled=0xF)
    words, csum = gamma_words(4)
    gva = g.t0.blob(struct.pack("<%dI" % len(words), *words))
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(1, r, gva, 15, 4, csum), label="mapped 15")
    s.quiesce(100)
    s.events("mapped 15")
    g.diff_page(1, "mapped 15")
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(1, r, gva, 16, 4, csum), label="mapped 16")
    s.wait(gpa=g.ss(1, 0xF98), eq=1, label="present count 1")
    s.events("mapped 16")
    g.diff_page(1, "mapped 16")
    return s


@scenario
def mutgap2_dev_cursor_height_257():
    """A glyph 257 pixels high is refused ("invalid cursor parameters")
    though every other check passes: 1 x 257, stride 4, mapped length 1028
    inside task 0, the right checksum. 1 x 256 with the same bytes is
    accepted."""
    s = Scenario("mutgap2_dev_cursor_height_257", displays=[display(0)], doc=mutgap2_dev_cursor_height_257.__doc__)
    g = DisplayGuest(s)
    g.connect(0)
    g.ack(0, 1)
    va = 8 * PAGE
    px = glyph_pixels(1, 257, 4, seed=5)
    g.t0.write(va, px)
    s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(0, va, 1028, 4, 1, 257, 0, 0, glyph_sum(1, 257, 4, px)),
             fault=True, label="1x257")
    s.fault_state("1x257")
    s.resume_root()
    s.submit(abi.OP_NOP)
    s.quiesce(100)
    s.events("1x257")
    g.diff_page(0, "1x257")
    s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(0, va, 1028, 4, 1, 256, 0, 0, glyph_sum(1, 256, 4, px)),
             label="1x256")
    s.wait(cb="cursorGlyph")
    s.events("1x256")
    g.diff_page(0, "1x256")
    return s


@scenario
def mutgap2_dev_cursor_height_0_stride_0():
    """A glyph of height 0 is refused (the parameter check tests `height == 0`)
    also where nothing after the parameter check would refuse it: 4 x 0,
    stride 0 ((h - 1) x stride does not overflow), mapped length 16, on a
    display with no cursor_glyph callback (nothing builds an image). The page
    keeps no glyph record. 4 x 1 with stride 0 is then accepted and
    recorded."""
    s = Scenario("mutgap2_dev_cursor_height_0_stride_0",
                 displays=[display(0, handlers=["newFrame", "cursorShow"])],
                 doc=mutgap2_dev_cursor_height_0_stride_0.__doc__)
    g = DisplayGuest(s)
    g.connect(0)
    g.ack(0, 1)
    va = 8 * PAGE
    g.t0.write(va, glyph_pixels(4, 1, 16))
    s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(0, va, 16, 0, 4, 0), fault=True, label="4x0 stride 0")
    s.fault_state("4x0 stride 0")
    s.resume_root()
    s.submit(abi.OP_NOP)
    s.quiesce(100)
    s.events("4x0 stride 0")
    g.diff_page(0, "4x0 stride 0")
    s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(0, va, 16, 0, 4, 1), label="4x1 stride 0")
    s.events("4x1 stride 0")
    g.diff_page(0, "4x1 stride 0")
    return s


# ---- EFI ----

@scenario(host=True)
def mutgap2_dev_efi_height_max():
    """An EFI mode exactly as tall as the host's largest 2D texture
    (maxTextureHeight2D, 16384 here) is accepted: a 1 x 16384 framebuffer,
    stride 16, over 16 pages is presented. Trace blocks, so the one present
    follows the rebuild."""
    s = Scenario("mutgap2_dev_efi_height_max", host=True, displays=[display(0)], descriptor=TRACE,
                 doc=mutgap2_dev_efi_height_max.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    e = EFI(s, pages=16)
    e.write_frame(1, 16384, 16, seed=7)
    e.setup(1, 16384, 16, length=16 * PAGE)
    s.wait(cb="newFrame", label="the 1 x 16384 frame")
    s.events("presented")
    s.api("present_counts", display=0)
    e.regs("registers")
    return s


@scenario
def mutgap2_dev_efi_too_tall_stride_0():
    """A mode taller than the 2D limit becomes 0 x 0: with the stride
    register 0 the stride is then 4 x 0 = 0, which passes the stride checks,
    so the framebuffer is mapped again and a trace range of length 0 is
    added; nothing is presented. Restoring the mode presents the 16 x 8 frame
    (three guest presents in all)."""
    s = Scenario("mutgap2_dev_efi_too_tall_stride_0", displays=[display(0)], descriptor=TRACE,
                 doc=mutgap2_dev_efi_too_tall_stride_0.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    e = EFI(s)
    e.setup(16, 8, 64)
    s.wait(cb="newFrame", cb_count=1)
    s.w(abi.R_EFI_STRIDE, 0)
    s.wait(cb="newFrame", cb_count=2)
    s.events("stride 0")
    s.w(abi.R_EFI_MODE, 16 | 16385 << 16)
    s.quiesce(200)
    s.events("too tall, stride 0")
    e.regs("too tall, stride 0")
    s.w(abi.R_EFI_MODE, 16 | 8 << 16)
    s.quiesce(200)
    s.events("restored")
    s.api("present_counts", display=0)
    return s


# ---- the IOSurface host device ----

@scenario(setups=MAPPED)
def mutgap2_dev_iosfc_ring_size_0():
    """A ring size of 0 written while a ring is mapped is ignored: 0x1008
    still reads the old size and the next command runs from the same ring.
    Map and isolate only: the IOSurface host device needs a memory map."""
    s = Scenario("mutgap2_dev_iosfc_ring_size_0", setups=MAPPED, iosfc=True,
                 doc=mutgap2_dev_iosfc_ring_size_0.__doc__)
    d = IOSFC(s)
    d.start()
    root, data = d.surface_pages(1)
    d.put_desc(1, desc_v1(root, 16, 8))
    idx = d.submit((CMD_MAP, 1))
    d.status(idx, "map")
    s.iosfc_w(I_RING_SIZE, 0)
    _regs(s, "size 0 after a ring")
    idx = d.submit((CMD_TEST, 1))
    d.status(idx, "test after size 0")
    _regs(s, "after the test")
    s.diff("surface", gpa=s.pa(data), length=0x40)
    return s


@scenario(setups=MAPPED)
def mutgap2_dev_iosfc_4096_tables():
    """A v2 surface of 4096 x 4095 + 1 pages needs exactly 4096 page-table
    pages, which the device allows: the map goes on to walk the root table,
    whose entry 0 is invalid, so the status is BadArgument ("root table
    invalid"), not NoMemory. So is a surface of exactly 4096 x 4096 pages,
    the most the page-count check allows; 4096 x 4096 + 1 pages is NoMemory.
    Map and isolate only."""
    s = Scenario("mutgap2_dev_iosfc_4096_tables", setups=MAPPED, iosfc=True,
                 doc=mutgap2_dev_iosfc_4096_tables.__doc__)
    d = IOSFC(s, v2=True)
    d.start()
    r = s.alloc_page()
    s.mem(s.pa(r), struct.pack("<I", pte(0, valid=0, is_table=1)))
    d.put_desc(1, desc_v2(s.pfn(r), 64, 64, alloc=(4096 * 4095 + 1) * PAGE, bpr=256))
    d.put_desc(2, desc_v2(s.pfn(r), 64, 64, alloc=(4096 * 4096 + 1) * PAGE, bpr=256))
    d.put_desc(3, desc_v2(s.pfn(r), 64, 64, alloc=4096 * 4096 * PAGE, bpr=256))
    idx = d.submit((CMD_MAP, 1), (CMD_MAP, 2), (CMD_MAP, 3))
    d.status(idx[:1], "4096 x 4095 + 1 pages")
    d.status(idx[1:2], "4096 x 4096 + 1 pages")
    d.status(idx[2:], "4096 x 4096 pages")
    s.events("end")
    return s


@scenario(setups=MAPPED)
def mutgap2_dev_iosfc_host_only_texture():
    """A host-only surface is created writable (readOnly false), a
    guest-backed surface described read-only gets kIOSurfaceClientDirection
    2, and Metal honours the direction. A mapper-ref texture (commandType 57)
    over each, a red fill of its first two pixels and a copy out to a shared
    buffer: over host-only mapping 2 (fresh, zeroed) the fill lands; over
    read-only mapping 3 it is silently dropped, with no fault, and the copy
    shows the guest page's bytes, which stay as written. Map and isolate
    only: the mapper needs a memory map."""
    from pgtest.resources_build import (BGRA8Unorm, OPT_SHARED, RTask, b_copy_tex_buf, b_fill_tex_color, desc2,
                                        iosurface_op, mapper_ref_rec)
    s = Scenario("mutgap2_dev_iosfc_host_only_texture", setups=MAPPED, iosfc=True,
                 doc=mutgap2_dev_iosfc_host_only_texture.__doc__)
    s.bring_up()
    W, H = 64, 16
    d = IOSFC(s, v2=True)
    d.start()
    d.put_desc(2, desc_v2(0, W, H, host_only=1))
    root, data = d.surface_pages(1)
    s.mem(s.pa(data), bytes((i * 3 + 3) & 0xFF for i in range(PAGE)))
    d.put_desc(3, desc_v2(root, W, H, read_only=1))
    idx = d.submit((CMD_MAP, 2), (CMD_MAP, 3))
    d.status(idx, "map host-only 2, read-only 3")
    t = RTask(s)
    tex = {}
    for mid in (2, 3):
        tex[mid] = t.add(abi.T_MAPPER_TEXTURE, mapper_ref_rec(mid) + iosurface_op(
            57, desc2(fmt=BGRA8Unorm, w=W, h=H, options=OPT_SHARED, usage=3)))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    for n, mid in enumerate((2, 3)):
        x = tex[mid]
        t.blit(b_fill_tex_color(x, (2, 1, 1), (1.0, 0, 0, 1.0), BGRA8Unorm), resources=[(x, 0), (b, 0)],
               label="fill mapping %d" % mid)
        t.blit(b_copy_tex_buf(x, b, 0x400 * n, 256, 1024, (4, 1, 1)), resources=[(x, 0), (b, 0)],
               label="copy out of mapping %d" % mid)
        t.dump(b_va + 0x400 * n, 0x10, "mapping %d, row 0" % mid)
    s.dump(s.pa(data), 0x10, "mapping 3's guest page")
    s.fault_state("end")
    s.events("end")
    return s


# ---- FIFO creation and barriers, EFI, displays, IOSurface, suspend ----

@scenario
def mutgap2_dev_create_fifo_0_while_index_0_waits():
    """CreateFIFO (48) of ID 0 is refused up front ("Invalid FIFO ID"),
    before the check of that ID's wait record: child 1 with stamp index 0 is
    blocked on slot 3, so the wait record of index 0 is in use, and the
    create only faults the root FIFO, which resumes and runs a NOP."""
    s = Scenario("mutgap2_dev_create_fifo_0_while_index_0_waits", stamp_slots=4,
                 doc=mutgap2_dev_create_fifo_0_while_index_0_waits.__doc__)
    s.bring_up()
    a = child(s, 1, stamp_index=0)
    a.cmd(abi.OP_NOP, barriers=[(3, 1)])
    a.doorbell()
    # Child 1's stamps would land in slot 0, the root FIFO's, so a stamp
    # wait cannot show it blocked; it has 300 ms to block.
    s.quiesce(300)
    s.submit(abi.OP_DEFINE_CHILD_FIFO, struct.pack("<I", 0), fault=True, label="create FIFO 0")
    s.fault_state("create FIFO 0")
    s.resume_root()
    s.submit(abi.OP_NOP)
    s.events("end")
    return s


@scenario
def mutgap2_dev_create_fifo_while_index_waits_dies():
    """FIFO creation asserts that the new ID's wait record is free, but wait
    records are indexed by the waiting FIFO's stamp index, not its ID: child
    1 with stamp index 2 blocks on slot 3, and CreateFIFO 2 then kills the
    host."""
    s = Scenario("mutgap2_dev_create_fifo_while_index_waits_dies", stamp_slots=4,
                 doc=mutgap2_dev_create_fifo_while_index_waits_dies.__doc__)
    s.bring_up()
    a = child(s, 1, stamp_index=2)
    st = a.cmd(abi.OP_NOP, barriers=[(3, 1)])
    a.doorbell()
    a.wait_stamp(st, timeout_ms=500, label="child 1 blocked on 3:1 (times out)")
    s.events("blocked")
    child(s, 2, stamp_index=3)
    s.events("not reached")
    return s


@scenario
def mutgap2_dev_barrier_count_65():
    """A command with 65 barriers is faulted ("Too many barriers") before any
    barrier is read, and only its header is consumed: its 528 zero bytes then
    parse as 44 zero-length headers, skipped, and the NOP after them (stamp
    3) runs when 0x1024 resumes the faulted head."""
    s = Scenario("mutgap2_dev_barrier_count_65", doc=mutgap2_dev_barrier_count_65.__doc__)
    f = s.bring_up()
    s.submit(abi.OP_NOP)
    f.raw(abi.HDR.pack(abi.OP_NOP, 65, 12 + 528, 2) + bytes(528))
    st = f.cmd(abi.OP_NOP, stamp=3)
    f.doorbell()
    s.wait_fault()
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written)
    s.fault_state("65 zero barriers")
    s.dump(f.base_gpa(), 4, "stamp slot 0 while faulted")
    s.resume_root()
    f.wait_stamp(st)
    s.events("resumed", irq_stamps=True)
    s.dump(f.base_gpa(), 4, "stamp slot 0")
    return s


@scenario
def mutgap2_dev_efi_length_page_plus_1():
    """An EFI framebuffer length of 0x4001, whose only unaligned bit is bit
    0, is refused ("not page aligned"): no task, no mapping, no present. A
    length of two pages then maps and presents."""
    s = Scenario("mutgap2_dev_efi_length_page_plus_1", displays=[display(0)], descriptor=TRACE,
                 doc=mutgap2_dev_efi_length_page_plus_1.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    e = EFI(s, pages=2)
    e.write_frame(16, 8, 64)
    e.setup(16, 8, 64, length=PAGE + 1)
    s.quiesce(200)
    s.events("length 0x4001")
    e.regs("length 0x4001")
    s.w(abi.R_EFI_LENGTH, 2 * PAGE)
    s.wait(cb="newFrame")
    s.events("length 0x8000")
    s.api("present_counts", display=0)
    return s


@scenario
def mutgap2_dev_present_guest_protection():
    """The display's protection options are read from the guest's page
    (+0x68); a backing passes when its IOSurface asks for no protection bit
    the options lack ((options | requirements) == options): with the guest's
    +0x68 set to 1 after the connect, a backing with no requirements still
    presents."""
    s = Scenario("mutgap2_dev_present_guest_protection", displays=[display(0)],
                 doc=mutgap2_dev_present_guest_protection.__doc__)
    g = DisplayGuest(s)
    r, va = g.backing(16, 8, pixels=pattern(16, 8))
    g.install()
    g.connect(0, enabled=0xF)
    g.ack(0, 1)
    s.mem(g.ss(0, 0x68), struct.pack("<Q", 1))
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, r))
    s.wait(cb="newFrame")
    s.events("presented")
    g.diff_page(0, "presented")
    s.api("present_counts", display=0)
    return s


@scenario
def mutgap2_dev_cursor_width_0_no_handler():
    """A glyph of width 0 is refused (the parameter check tests `width == 0`)
    also on a display with no cursor_glyph callback, where nothing would build
    an image: 0 x 1, stride 4, mapped 64 faults, and the page keeps the 4 x 1
    glyph recorded before it."""
    s = Scenario("mutgap2_dev_cursor_width_0_no_handler",
                 displays=[display(0, handlers=["newFrame", "cursorShow"])],
                 doc=mutgap2_dev_cursor_width_0_no_handler.__doc__)
    g = DisplayGuest(s)
    g.connect(0)
    g.ack(0, 1)
    va = 8 * PAGE
    g.t0.write(va, glyph_pixels(4, 1, 16))
    s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(0, va, 16, 16, 4, 1), label="4x1")
    g.diff_page(0, "4x1")
    s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(0, va, 64, 4, 0, 1), fault=True, label="0x1")
    s.fault_state("0x1")
    s.resume_root()
    s.submit(abi.OP_NOP)
    s.quiesce(100)
    s.events("0x1")
    g.diff_page(0, "0x1")
    return s


@scenario(setups=MAPPED)
def mutgap2_dev_swap_8192_page_1_mapped():
    """DisplaySwapMapping of mapping ID 8192 is "Invalid mappingID" also
    while descriptor page 1 is mapped (by a map of ID 16, the first v2
    descriptor on that page): the stamp signals, nothing is presented, and a
    NOP follows. Map and isolate only."""
    from pgtest.display_guest import p_swap
    from scenarios.display_mapper import _setup
    s = Scenario("mutgap2_dev_swap_8192_page_1_mapped", setups=MAPPED, iosfc=True, displays=[display(0)],
                 doc=mutgap2_dev_swap_8192_page_1_mapped.__doc__)
    g, d, pages = _setup(s, {})
    first = d.per_page()
    root, data = d.surface_pages(1)
    d.put_desc(first, desc_v2(root, 16, 8))
    idx = d.submit((CMD_MAP, first))
    d.status(idx, "map 16")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 8192), label="swap to 8192")
    s.submit(abi.OP_NOP)
    s.quiesce(100)
    s.events("end")
    g.diff_page(0, "page")
    s.api("present_counts", display=0)
    return s


@scenario(setups=MAPPED)
def mutgap2_dev_iosfc_id_0_valid_descriptor():
    """Mapping ID 0 is out of range for map, test and unmap even when
    descriptor 0 is valid: BadArgument three times, and the descriptor is
    untouched; ID 1 with the same descriptor maps. v1 and v2. Map and isolate
    only."""
    out = []
    for v2_ in (False, True):
        name = "mutgap2_dev_iosfc_id_0_valid_descriptor_" + ("v2" if v2_ else "v1")
        s = Scenario(name, setups=MAPPED, iosfc=True, doc=mutgap2_dev_iosfc_id_0_valid_descriptor.__doc__)
        d = IOSFC(s, v2=v2_)
        d.start()
        mk = desc_v2 if v2_ else desc_v1
        root, data = d.surface_pages(1)
        d.put_desc(0, mk(root, 16, 8))
        d.put_desc(1, mk(root, 16, 8))
        idx = d.submit((CMD_MAP, 0), (CMD_TEST, 0), (2, 0), (CMD_MAP, 1))
        d.status(idx, "map, test, unmap 0; map 1")
        s.dump(d.desc_gpa(0), 16, "descriptor 0")
        s.events("end")
        out.append(s)
    return out


@scenario
def mutgap2_dev_finish_suspend_fresh_dies():
    """rvg_device_finish_suspend without rvg_device_will_suspend asserts the
    run state also on a fresh device with only a version, which has no FIFO to
    drain, no task and no display to pause: the host dies at the call
    (suspend_host_asserts_finish_without_will dies there too, but with a
    running FIFO that nothing quiesced)."""
    s = Scenario("mutgap2_dev_finish_suspend_fresh_dies", doc=mutgap2_dev_finish_suspend_fresh_dies.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    s.r(abi.R_VERSION, "before")
    s.api("finishSuspend", slot="s")
    s.events("not reached")
    return s
