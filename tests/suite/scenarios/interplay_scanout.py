"""The display pipeline end to end (docs/testing.md, sections 6.21-6.24):
the guest renders with the GPU into surfaces that become display frames,
and presents them across several displays while the host and the guest
change everything around them: modes, hot plug and unplug, the cursor,
gamma and the colour matrix, pause, suspend and resume onto a fresh
device, stop and reset with the guest redoing its set-up, and the IOSurface
host device mapping, unmapping and testing the surfaces at the same time.

The surfaces are IOSurface mapper surfaces over guest pages, rendered
through mapper textures (type 11) and mapper buffers (type 15) and
presented with opcode 8; task 0's IOSurface backings, written through ref
textures (type 5) and presented with opcode 7; and the EFI framebuffer.
Where a GPU write and a present touch the same surface, barriers order them
(the order is the guest's, so the frame it implies is locked). The steps
are pgtest/scanout_util.py's, over pgtest/session.py's.

Every scenario that encodes a frame is keyed by the host: the library's
shaders draw it (rule 5). The surfaces' guest pages, which the GPU writes with
exact clear colours and the committed shaders, are dumped as well. Scenarios
over mapper surfaces run in the map and isolate setups only: the IOSurface
host device exists only with a memory map (apple-gfx-mmio.c). Scenarios over
ref textures run in blocks and map only: a type 5 texture aborts the service
(docs/testing.md, section 7.4).
"""
from pgtest import abi
from pgtest.display_guest import SS_MODE_COUNT, display, floats, p_swap
from pgtest.fifo_util import resume_child
from pgtest.registry import scenario
from pgtest.resources_build import cmd_delete_resource
from pgtest.scanout_util import (BLUE, CMD_MAP, CMD_TEST, CMD_UNMAP, FH, FRAME, FW, GREEN, GREY, IN_PROCESS, MAPPED,
                                 RED, WHITE, Scanout)


def _compositor(name, doc, ports=1, displays=None, surfaces=((1, 3), (2, 5)), children=(1, 2), gate=3,
                buffers=(), setups=MAPPED, host=True, efi=False, frames=2):
    """The compositor's session: task 0 with `frames` backings, a page on
    every port (acked where a display is attached), the IOSurface host
    device with `surfaces` mapped, task 1 with a mapper texture over each
    surface (and a mapper buffer over each in buffers), child FIFOs
    `children` and the gate."""
    x = Scanout(name, doc, ports=ports, displays=displays, setups=setups, host=host, iosfc=bool(surfaces),
                efi=efi)
    x.negotiate(asks=())
    x.root()
    x.kernel_task(frames=frames)
    attached = range(ports) if displays is None else displays
    for p in range(ports):
        x.connect(p, has_display=p in attached)
    if surfaces:
        x.iosfc_start()
        for mid, seed in surfaces:
            x.surface(mid, seed)
    mids = [m for m, _ in surfaces]
    t = x.task(1, 0x11, extra=x.mapper_objects(mids, buffers) if surfaces else None)
    for fid in children:
        x.child(fid)
    if gate:
        x.make_gate(gate)
    return x, t


# ---- GPU writes and presents ordered by barriers ----


@scenario(setups=MAPPED, host=True)
def interplay_scanout_render_present_render():
    """A present between two GPU writes of the same surface, ordered by
    barriers only: child 1 clears surface 1 red (gated on the gate), the
    root presents it with opcode 8 gated on child 1's stamp, and child 2
    clears it green gated on the root's present and on the gate's second
    stamp. The FIFOs are rung root first. The first gate releases the red
    render and the present: new_frame, and the frame is red. The second
    releases the green render: no new_frame, and the frame the host encodes
    is green, because the display holds the surface, not a copy of it. Then
    child 1 draws the triangle into surface 2 and child 2 presents it (a
    barrier on child 1), and the root presents surface 1 again."""
    x, t = _compositor("interplay_scanout_render_present_render", interplay_scanout_render_present_render.__doc__)
    s = x.s
    st1 = x.exec_cmd(1, t, x.compose(t, t.mtex[1], RED), barriers=[x.gate_barrier(1)])
    stp = x.swap_cmd(0, 0, 1, barriers=[(1, st1)])
    st2 = x.exec_cmd(2, t, x.compose(t, t.mtex[1], GREEN), barriers=[(0, stp), x.gate_barrier(2)])
    x.ring(0, 2, 1)
    x.open_gate()
    x.wait_stamp(0, stp, "the present, after the red render")
    x.expect("newFrame", label="surface 1 presented")
    x.encode(0, "red: the render the present waited for")
    x.dump_surface(1, "surface 1: red")
    s.events("red presented")
    x.open_gate()
    x.wait_stamp(2, st2, "the green render, after the present")
    t.reset_cmdbufs()
    x.encode(0, "green: the display shows the surface as it is now")
    x.dump_surface(1, "surface 1: green")
    s.events("green rendered, nothing presented")
    st3 = x.exec_cmd(1, t, x.compose(t, t.mtex[2], BLUE, draw=True))
    st4 = x.swap_cmd(2, 0, 2, barriers=[(1, st3)])
    x.ring(2, 1)
    x.wait_stamp(2, st4, "surface 2 presented from child 2")
    t.reset_cmdbufs()
    x.expect("newFrame", label="surface 2 presented")
    x.encode(0, "the triangle on blue, from child 2")
    x.dump_surface(2, "surface 2: the triangle")
    x.swap_on(0, 0, 1)
    x.page(0, "after three presents")
    s.api("present_counts", display=0)
    s.events("end")
    return s


@scenario(setups=MAPPED, host=True)
def interplay_scanout_present_from_faulted_neighbour():
    """Frames presented from child 2 while its neighbour child 1 faults.
    Child 1's exec names a blit record the decoder does not know (399) and
    faults its FIFO; child 2 renders surface 1 and presents it, twice, and
    the frames reach the host. A present written on child 1 behind its
    fault still runs (new_frame, the page counts it) but its stamp is held
    until 0x1028 resumes child 1, which then presents again. The nodes,
    stamp slots and fault registers are locked at each step."""
    x, t = _compositor("interplay_scanout_present_from_faulted_neighbour",
                       interplay_scanout_present_from_faulted_neighbour.__doc__, gate=None)
    s = x.s
    x.stamps(3, "stamp slots before")
    x.exec_cmd(1, t, x.bad_blit())
    x.ring(1)
    x.node_fault(1)
    t.reset_cmdbufs()
    x.node(1, "child 1 faulted")
    s.fault_state("child 1 faulted")
    st = x.exec_cmd(2, t, x.compose(t, t.mtex[1], GREEN, draw=True))
    x.go(2, st, "child 2 renders beside the fault")
    t.reset_cmdbufs()
    x.swap_on(2, 0, 1)
    x.dump_surface(1, "surface 1 from child 2")
    held = x.swap_cmd(1, 0, 2)
    x.ring(1)
    x.expect("newFrame", label="the present behind child 1's fault runs")
    x.encode(0, "surface 2, presented behind the fault")
    x.stamps(3, "child 1's stamps held behind the fault")
    x.page(0, "presented behind the fault")
    s.events("present behind the fault")
    st = x.exec_cmd(2, t, x.compose(t, t.mtex[1], RED))
    x.go(2, st, "child 2 renders again")
    t.reset_cmdbufs()
    x.swap_on(2, 0, 1)
    resume_child(s, 1)
    x.wait_stamp(1, held, "child 1 resumed: its held stamps publish")
    x.node(1, "child 1 resumed")
    s.fault_state("resumed")
    x.swap_on(1, 0, 2)
    x.stamps(3, "stamp slots at the end")
    s.events("end")
    return s


@scenario(setups=MAPPED, host=True)
def interplay_scanout_two_heads_one_surface():
    """One surface on two heads, rendered between presents. Surface 1 is
    presented on ports 0 and 1; child 1 then draws into it, and both heads
    encode the new contents with no present. Task 1's mapper texture is
    deleted (37), so only the displays hold the surface: an unmap is Busy
    while either shows it, surface 2 moves port 0, the unmap is still Busy,
    surface 2 moves port 1, and the unmap succeeds; a present of mapping 1
    is then "Invalid mappingID" (its stamp signalled, nothing called), and
    mapping it again over the same guest page presents what the GPU
    drew."""
    x, t = _compositor("interplay_scanout_two_heads_one_surface", interplay_scanout_two_heads_one_surface.__doc__,
                       ports=2, gate=None, children=(1,))
    s = x.s
    x.swap_on(0, 0, 1)
    x.swap_on(0, 1, 1)
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], GREY, draw=True))
    x.go(1, st, "child 1 draws into the shown surface")
    t.reset_cmdbufs()
    x.encode(0, "head 0 after the draw")
    x.encode(1, "head 1 after the draw")
    x.dump_surface(1, "surface 1 drawn")
    cmd_delete_resource(s, 1, t.mtex[1], label="37: task 1 lets surface 1 go")
    x.iosfc_cmd(CMD_UNMAP, 1, "unmap 1 while both heads show it")
    x.swap_on(0, 0, 2)
    x.iosfc_cmd(CMD_UNMAP, 1, "unmap 1 while head 1 shows it")
    x.swap_on(0, 1, 2)
    x.iosfc_cmd(CMD_UNMAP, 1, "unmap 1, shown nowhere")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="present unmapped mapping 1")
    s.quiesce(100)
    s.events("present of an unmapped surface")
    x.iosfc_cmd(CMD_MAP, 1, "map 1 again over the drawn page")
    x.swap_on(0, 1, 1)
    s.api("present_counts", display=0)
    s.api("present_counts", display=1)
    x.page(0, "port 0")
    x.page(1, "port 1")
    s.events("end")
    return s


@scenario(setups=MAPPED, host=True)
def interplay_scanout_unplug_under_held_present():
    """A present held behind a GPU render (itself held on the gate) while
    the host unplugs the display: the guest display disconnects (8, mode
    count 0), and when the gate opens the render lands in the surface and
    the present reaches a guest display with no client: it completes at once
    (raised |= 2, the page counts it and records the mapping) and nothing is
    called. A display plugged into the port again connects (4, expectedAck
    2), the guest acks, and a present of the same surface reaches the new
    display (index 1): mode_change and new_frame, and the frame the GPU
    drew."""
    x, t = _compositor("interplay_scanout_unplug_under_held_present",
                       interplay_scanout_unplug_under_held_present.__doc__, children=(1,))
    s = x.s
    x.swap_on(0, 0, 2)
    st1 = x.exec_cmd(1, t, x.compose(t, t.mtex[1], BLUE, draw=True), barriers=[x.gate_barrier()])
    stp = x.swap_cmd(0, 0, 1, barriers=[(1, st1)])
    x.ring(0, 1)
    x.unplug(0, 0)
    s.events("unplugged under the held present")
    x.page(0, "unplugged")
    x.open_gate()
    x.wait_stamp(0, stp, "the present, with no client")
    t.reset_cmdbufs()
    s.quiesce(100)
    s.events("presented to no client")
    x.page(0, "presented to no client")
    x.dump_surface(1, "surface 1 drawn")
    x.hotplug(0)
    s.events("plugged again")
    x.swap_on(0, 0, 1, index=1)
    s.api("present_counts", display=1)
    x.page(0, "the new display presenting")
    s.events("end")
    return s


@scenario(host=True)
def interplay_scanout_hotplug_while_paused():
    """Hot plug and a mode list change while the device is paused, between
    GPU work and presents: port 0 presents and task 1 runs a phase; paused,
    a display plugged into port 1 (its page connected with none) and a new
    mode list on port 0 run no state machine (the guest displays are
    paused), so the pages do not change. unpause runs each port's machine in
    port order: port 0 raises 12 (update, expectedAck 2) and port 1 raises 4
    (connect, expectedAck 1). After the acks both heads present, and task
    1's second phase equals its first."""
    x = Scanout("interplay_scanout_hotplug_while_paused", interplay_scanout_hotplug_while_paused.__doc__,
                ports=2, displays=(0,))
    s = x.s
    x.negotiate(asks=())
    x.root()
    x.kernel_task()
    x.connect(0)
    x.connect(1, has_display=False)
    t = x.task(1, 0x21)
    x.present(0, 0)
    x.phase(t, 0)
    x.results(t, 0)
    s.api("pause")
    s.api("new_display", spec=display(1, name="head1", modes=x.modes))
    x.set_modes(0, ((FW, FH, 60), (32, 16, 30)), label="port 0's new modes while paused")
    s.quiesce(100)
    s.events("paused: plugged and modes changed")
    x.page(0, "paused")
    x.page(1, "paused")
    s.api("unpause")
    s.wait(gpa=x.ss(1, SS_MODE_COUNT), mask=0xFFFF, ne=0, label="port 1's modes after unpause")
    s.events("unpaused")
    s.r(abi.R_IRQ_DISPLAY, "unpaused")
    x.page(0, "unpaused")
    x.page(1, "unpaused")
    x.ack_next(0)
    x.ack_next(1)
    x.present(0, 1)
    x.present(1, 0, display_index=1)
    x.phase(t, 1)
    x.same(t, 0, t, 1, "task 1 across the pause")
    s.events("end")
    return s


@scenario(setups=MAPPED)
def interplay_scanout_mapped_gamma_one_dies():
    """Opcode 8 with a gamma table of one entry, after GPU frames with good
    gamma tables: child 1 renders surface 1, the root presents it with a
    256-entry table (stored at +0xf80), then with one of 16; then a table of
    one entry kills the host at the present, as opcode 7's does (the slope
    step is 1/0 and the red slope reads index 0xffffffff), while the display
    presents the mapped surface. No encode: nothing here reports the host.
    The death is on the display's queue, so the present's stamp may land
    before it: the scenario ends in a wait only the death ends (0x102c never
    reads 2)."""
    x, t = _compositor("interplay_scanout_mapped_gamma_one_dies", interplay_scanout_mapped_gamma_one_dies.__doc__,
                       host=False, gate=None, children=(1,))
    s = x.s
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], GREY, draw=True))
    x.go(1, st, "child 1 draws")
    t.reset_cmdbufs()
    x.swap_on(0, 0, 1, gamma=x.gamma(256, lambda c, i, n: (i / (n - 1)) ** 1.5), encode=False)
    x.page(0, "gamma 256")
    x.swap_on(0, 0, 1, gamma=x.gamma(16, lambda c, i, n: i / (n - 1)), encode=False)
    x.page(0, "gamma 16")
    s.events("two gamma tables")
    g1 = x.gamma(1, lambda c, i, n: 0.5)
    s.fifo.cmd(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1, *g1))
    s.fifo.doorbell()
    s.wait(mmio=abi.R_FAULT_MASK, mmio_eq=2, timeout_ms=10000, label="gamma of one entry: the host dies")
    return s


# ---- backings written by the GPU (opcode 7) ----


@scenario(setups=IN_PROCESS, host=True)
def interplay_scanout_ref_texture_frames():
    """Task 0's backings written by the GPU through ref textures (type 5) in
    task 1, and presented with opcode 7. Backing 0 is presented (its guest
    pages are copied into its IOSurface once); a blit fills it red through
    the ref texture and the next encode shows red with no present; child 1
    draws the triangle into backing 1 and the root presents it gated on
    child 1's stamp. The guest then writes a new pattern into backing 0's
    pages and presents it again: the backing stays host-valid, so the frame
    is still the GPU's red, and its guest pages still hold what the guest
    wrote. In-process only: a type 5 texture aborts the service."""
    x = Scanout("interplay_scanout_ref_texture_frames", interplay_scanout_ref_texture_frames.__doc__,
                setups=IN_PROCESS)
    s = x.s
    x.negotiate(asks=())
    x.root()
    x.kernel_task(frames=2)
    x.connect(0)
    t = x.task(1, 0x31, extra=x.ref_objects((0, 1)))
    x.child(1)
    x.present(0, 0)
    st = x.exec_cmd(0, t, x.fill_rect(t.rtex[0], RED))
    x.go(0, st, "fill backing 0 red through the ref texture")
    t.reset_cmdbufs()
    x.encode(0, "backing 0 after the fill, not presented again")
    st1 = x.exec_cmd(1, t, x.compose(t, t.rtex[1], BLUE, draw=True))
    stp = x.present_cmd(0, 0, 1, barriers=[(1, st1)])
    x.ring(0, 1)
    x.wait_stamp(0, stp, "backing 1 presented after child 1's draw")
    t.reset_cmdbufs()
    x.expect("newFrame", label="backing 1")
    x.encode(0, "backing 1: the triangle")
    st = x.exec_cmd(0, t, x.readback(t, t.rtex[1], 0) + x.readback(t, t.rtex[0], 1))
    x.go(0, st, "read both backings back through the GPU")
    t.reset_cmdbufs()
    s.dump(t.gpa(t.out_va), FRAME, "backing 1 as the GPU sees it")
    s.dump(t.gpa(t.out_va + 0x400), FRAME, "backing 0 as the GPU sees it")
    from pgtest.display_guest import pattern
    x.t0.write(x.backings[0][1], pattern(FW, FH, 0x77))
    x.present(0, 0)
    s.dump(x.t0.gpa(x.backings[0][1]), 0x40, "backing 0's guest pages")
    x.page(0, "after four presents")
    s.api("present_counts", display=0)
    s.events("end")
    return s


# ---- the cursor over GPU frames ----


@scenario(setups=MAPPED, host=True)
def interplay_scanout_cursor_over_frames():
    """The cursor on two heads while the GPU renders and presents: a glyph
    and show on head 0, moves between frames rendered on child 1 and
    presented from child 2; a hide (show 0, then a doorbell that finds the
    guest's +0xe04 equal to the host's and calls only cursor_move); a glyph
    and show on head 1 whose moves interleave with head 0's frames; the
    positions the host reads (rvg_display_cursor_position) and the pages'
    cursor fields."""
    x, t = _compositor("interplay_scanout_cursor_over_frames", interplay_scanout_cursor_over_frames.__doc__,
                       ports=2, gate=None)
    s = x.s
    x.cursor(0, 3, 2)
    for i, color in enumerate((RED, GREEN)):
        st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], color, draw=(i == 1)))
        stp = x.swap_cmd(2, 0, 1, barriers=[(1, st)])
        x.ring(2, 1)
        x.wait_stamp(2, stp, "frame %d presented from child 2" % i)
        t.reset_cmdbufs()
        x.expect("newFrame", label="frame %d" % i)
        x.encode(0, "head 0, frame %d" % i)
        x.move_cursor(0, 5 + i, 1 + i)
        s.api("cursorPosition", display=0, label="after frame %d" % i)
    x.hide_cursor(0)
    x.cursor(1, 7, 7)
    x.swap_on(0, 1, 2)
    x.move_cursor(1, 15, 7)
    x.swap_on(0, 0, 2)
    x.move_cursor(0, 0, 0)
    s.api("cursorPosition", display=0, label="head 0")
    s.api("cursorPosition", display=1, label="head 1")
    s.events("cursors and frames")
    for p in (0, 1):
        s.dump(x.ss(p, 0xE00), 8, "port %d: cursor position and guest show" % p)
        s.dump(x.ss(p, 0xF00), 0x28, "port %d: glyph record and host show" % p)
    return s


# ---- modes and frame sizes ----


@scenario(setups=MAPPED, host=True)
def interplay_scanout_mode_switch_surfaces():
    """A resolution change carried out by the guest: the host's mode list
    grows to 32 x 16 (12, expectedAck 2), the guest acks, renders a 32 x 16
    surface (mapping 3) on child 1 and presents it: mode_change {32, 16}
    before the new_frame; the host encodes 32 x 16. The mode list shrinks
    back to 16 x 8 (12, expectedAck 3) while a 16 x 8 frame is held on the
    gate behind a render; after the ack the gate opens, the present calls
    mode_change {16, 8}, and the host encodes it. Each size change calls
    mode_change once."""
    x = Scanout("interplay_scanout_mode_switch_surfaces", interplay_scanout_mode_switch_surfaces.__doc__,
                setups=MAPPED, iosfc=True, modes=((FW, FH, 60),))
    s = x.s
    x.negotiate(asks=())
    x.root()
    x.kernel_task()
    x.connect(0)
    x.iosfc_start()
    x.surface(1, 3)
    x.surface(3, 9, w=32, h=16)
    from pgtest.resources_build import BGRA8Unorm
    t = x.task(1, 0x41, extra=x.mapper_objects((1, 3), sizes={3: (32, 16, BGRA8Unorm)}))
    x.child(1)
    x.make_gate(2)
    x.swap_on(0, 0, 1)
    x.set_modes(0, ((FW, FH, 60), (32, 16, 60)), label="32 x 16 offered")
    s.wait(gpa=x.ss(0, 0xFA9), mask=0xFF, eq=1, label="the update's ack pending")
    s.events("mode list grew")
    x.page(0, "mode list grew")
    x.ack_next(0)
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[3], BLUE, draw=True, w=32, h=16))
    stp = x.swap_cmd(0, 0, 3, barriers=[(1, st)])
    x.ring(0, 1)
    x.wait_stamp(0, stp, "32 x 16 presented")
    t.reset_cmdbufs()
    x.expect("newFrame", label="32 x 16")
    x.encode_sized(0, 32, 16, "32 x 16")
    s.dump(s.pa(x.surfaces[3]), 32 * 16 * 4, "surface 3")
    s.events("32 x 16 presented")
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], GREEN), barriers=[x.gate_barrier()])
    stp = x.swap_cmd(0, 0, 1, barriers=[(1, st)])
    x.ring(0, 1)
    x.set_modes(0, ((FW, FH, 60),), label="back to 16 x 8")
    s.wait(gpa=x.ss(0, 0xFA9), mask=0xFF, eq=1, label="the second update's ack pending")
    x.page(0, "mode list shrank")
    s.events("mode list shrank")
    x.open_gate()
    x.wait_stamp(0, stp, "16 x 8 presented after the gate")
    t.reset_cmdbufs()
    x.expect("newFrame", label="16 x 8 again")
    x.ack_next(0)
    x.encode(0, "16 x 8 again")
    x.encode_sized(0, 32, 16, "16 x 8 scaled into 32 x 16")
    x.page(0, "acked")
    s.events("end")
    return s


# ---- gamma and the colour matrix ----


@scenario(setups=MAPPED, host=True)
def interplay_scanout_gamma_matrix_frames():
    """Gamma tables and colour matrices over GPU frames, mixing opcodes 7
    and 8 on one display: a grey surface rendered on child 1 is presented
    with no table, then with a 64-entry table (the gamma shader from then
    on); a colour matrix swapping red and blue (10) applies at the next
    present, which is backing 0 through opcode 7 with its own 16-entry
    table; then the surface again with no table and nothing dirty. Each
    frame is encoded; the page's gamma record, colour matrix indices and
    presented surface and mapping are locked."""
    x, t = _compositor("interplay_scanout_gamma_matrix_frames", interplay_scanout_gamma_matrix_frames.__doc__,
                       gate=None, children=(1,))
    s = x.s
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], GREY))
    x.go(1, st, "grey")
    t.reset_cmdbufs()
    x.swap_on(0, 0, 1)
    x.swap_on(0, 0, 1, gamma=x.gamma(64, lambda c, i, n: (i / (n - 1)) ** (0.5 + c * 0.5)))
    x.page(0, "gamma 64")
    x.matrix(0, floats(0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0))
    g16 = x.gamma(16, lambda c, i, n: 1.0 - i / (n - 1))
    st = x.present_cmd(0, 0, 0, gamma=g16)
    x.go(0, st, "backing 0 with gamma 16, the matrix dirty")
    x.expect("newFrame", label="backing 0")
    x.encode(0, "backing 0, inverted, red and blue swapped")
    x.page(0, "after the backing")
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], WHITE, draw=True))
    x.go(1, st, "the triangle")
    t.reset_cmdbufs()
    x.swap_on(0, 0, 1)
    x.page(0, "the surface again")
    s.api("present_counts", display=0)
    s.events("end")
    return s


# ---- sleep ----


@scenario(setups=MAPPED, host=True)
def interplay_scanout_sleep_while_rendering():
    """The display asleep (9) while the guest renders and presents: a
    present while asleep updates the frame without new_frame (the page
    counts it, the display does not); the GPU draws into the presented
    surface; waking signals the current frame (one new_frame), and its
    encode shows the drawing. A second head stays awake throughout and
    takes its frames."""
    from pgtest.display_guest import p_sleep
    x, t = _compositor("interplay_scanout_sleep_while_rendering", interplay_scanout_sleep_while_rendering.__doc__,
                       ports=2, gate=None, children=(1,))
    s = x.s
    x.swap_on(0, 0, 2)
    s.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(0, 1), label="port 0 asleep")
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], RED))
    x.go(1, st, "red")
    t.reset_cmdbufs()
    st = x.swap_cmd(0, 0, 1)
    x.go(0, st, "present while asleep")
    x.swap_on(0, 1, 2)
    s.api("present_counts", display=0, label="asleep")
    s.events("asleep present")
    x.page(0, "asleep")
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], BLUE, draw=True))
    x.go(1, st, "the triangle into the asleep frame")
    t.reset_cmdbufs()
    s.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(0, 0), label="port 0 awake")
    x.expect("newFrame", label="the wake signals the current frame")
    x.encode(0, "woken: the triangle")
    s.api("present_counts", display=0, label="awake")
    x.page(0, "awake")
    s.events("end")
    return s


# ---- suspend and resume ----


@scenario(setups=MAPPED, host=True)
def interplay_scanout_suspend_two_heads():
    """A migration with two heads mid-composition. Head 0 shows surface 1,
    rendered red on child 1, with a cursor; head 1 shows surface 2 with a
    64-entry gamma table, and then backing 0 through opcode 7 (so its page
    records surface 0 and still mapping 2). On the fresh device each guest
    display replays its cursor and re-presents its last mapping: head 1 gets
    mapping 2 again, with the page's gamma table, though its last frame was
    the backing. Task 1 and the children come back from the root page: child
    1 draws into surface 1, child 2 presents it (a barrier on child 1), and
    an unmap of surface 1 is Busy (the restored display and task 1's mapper
    texture hold it)."""
    x, t = _compositor("interplay_scanout_suspend_two_heads", interplay_scanout_suspend_two_heads.__doc__, ports=2,
                       gate=None)
    s = x.s
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], RED))
    stp = x.swap_cmd(2, 0, 1, barriers=[(1, st)])
    x.ring(2, 1)
    x.wait_stamp(2, stp, "surface 1 on head 0")
    t.reset_cmdbufs()
    x.expect("newFrame", label="head 0")
    x.cursor(0, 4, 4)
    x.swap_on(0, 1, 2, gamma=x.gamma(64, lambda c, i, n: (i / (n - 1)) ** 2))
    st = x.present_cmd(0, 1, 0)
    x.go(0, st, "backing 0 on head 1")
    x.expect("newFrame", label="backing 0 on head 1")
    x.encode(1, "backing 0 on head 1, through the gamma table")
    x.page(1, "head 1 before the suspend")
    s.events("before the suspend")
    x.suspend()
    x.resume()
    x.expect("cursorGlyph", label="head 0's glyph replayed")
    x.expect("cursorShow", label="head 0's show replayed")
    x.expect("newFrame", n=2, label="both heads re-presented")
    s.events("resumed")
    x.encode(0, "head 0 restored")
    x.encode(1, "head 1 restored: mapping 2, not the backing")
    x.page(0, "head 0 restored")
    x.page(1, "head 1 restored")
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], BLUE, draw=True))
    stp = x.swap_cmd(2, 0, 1, barriers=[(1, st)])
    x.ring(2, 1)
    x.wait_stamp(2, stp, "surface 1 again after the resume")
    t.reset_cmdbufs()
    x.expect("newFrame", label="head 0 after the resume")
    x.encode(0, "head 0: the triangle after the resume")
    x.iosfc_cmd(CMD_UNMAP, 1, "unmap surface 1 after the resume")
    s.events("end")
    return s


@scenario(setups=MAPPED, host=True)
def interplay_scanout_suspend_held_frame():
    """A frame held behind the GPU when the host suspends: child 1's render
    of surface 1 waits on the gate and the root's present of it waits on
    child 1. rvg_device_finish_suspend abandons both waits; on the fresh device
    the restore re-presents surface 2 (the last presented), and the root FIFO
    and child 1 parse their commands again and wait again. The gate opens: the
    render lands and the present follows it; the frame is the render's."""
    x, t = _compositor("interplay_scanout_suspend_held_frame", interplay_scanout_suspend_held_frame.__doc__,
                       children=(1,), gate=2)
    s = x.s
    x.swap_on(0, 0, 2)
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], GREEN, draw=True), barriers=[x.gate_barrier()])
    stp = x.swap_cmd(0, 0, 1, barriers=[(1, st)])
    x.ring(0, 1)
    s.r(abi.R_FIFO_READ, "the root FIFO holds the present")
    x.dump_surface(1, "surface 1 before the suspend")
    x.suspend()
    x.resume()
    x.expect("newFrame", label="the restore re-presents surface 2")
    s.events("resumed")
    x.encode(0, "restored: surface 2")
    x.open_gate()
    x.wait_stamp(0, stp, "the held present, after the resume")
    t.reset_cmdbufs()
    x.expect("newFrame", label="surface 1")
    x.encode(0, "the held frame")
    x.dump_surface(1, "surface 1 rendered after the resume")
    x.page(0, "end")
    s.events("end")
    return s


# ---- stop and reset, and the guest's set-up again ----


def _redo(x, frames=2, surfaces=True, mids=(1, 2), children=(1,)):
    """The guest's set-up after reset or stop: a fresh root page, the
    version (reset zeroes it), the root FIFO, task 0, the display pages,
    task 1 and the children, as a driver that starts over does."""
    s = x.s
    s.root_page = s.alloc_page()
    x.negotiate(asks=())
    x.root()
    x.backings = []
    x.kernel_task(frames=frames)
    for p in sorted(x.pages):
        x.connect(p)
    x.tasks = []
    t = x.task(1, 0x11, extra=x.mapper_objects(mids) if surfaces else None)
    for fid in children:
        x.child(fid)
    return t


@scenario(setups=MAPPED, host=True)
def interplay_scanout_reset_redo():
    """reset in the middle of a composition on two heads, and the guest
    starting over. Before: surfaces 1 and 2 rendered on child 1 and shown
    on heads 0 and 1. reset tears the guest displays down (8 on both pages,
    the root words cleared) and the task with its mapper textures, so
    surface 2 now unmaps (nothing holds it) and maps again. The guest redoes
    everything on fresh pages: both heads connect (4) and take their acks,
    task 1 and child 1 are defined again, surface 1 is rendered green and
    surface 2 gets the same triangle as before, and both are presented."""
    x, t = _compositor("interplay_scanout_reset_redo", interplay_scanout_reset_redo.__doc__, ports=2, gate=None,
                       children=(1,))
    s = x.s
    for mid, port, color in ((1, 0, RED), (2, 1, BLUE)):
        st = x.exec_cmd(1, t, x.compose(t, t.mtex[mid], color, draw=(mid == 2)))
        x.go(1, st, "render surface %d" % mid)
        t.reset_cmdbufs()
        x.swap_on(0, port, mid)
    x.dump_surface(2, "surface 2: the triangle")
    s.events("before the reset")
    old = dict(x.pages)
    s.api("reset")
    s.events("reset")
    for p in (0, 1):
        s.diff(label="port %d: reset" % p, gpa=s.pa(old[p]), length=0x4000)
    s.fault_state("reset")
    x.iosfc_cmd(CMD_UNMAP, 2, "unmap surface 2 after the reset")
    x.iosfc_cmd(CMD_MAP, 2, "map surface 2 again")
    t = _redo(x)
    s.events("set up again")
    for mid, port, color in ((1, 0, GREEN), (2, 1, BLUE)):
        st = x.exec_cmd(1, t, x.compose(t, t.mtex[mid], color, draw=(mid == 2)))
        x.go(1, st, "render surface %d again" % mid)
        t.reset_cmdbufs()
        x.swap_on(0, port, mid)
    x.dump_surface(1, "surface 1: green now")
    x.dump_surface(2, "surface 2: the triangle again")
    x.page(0, "port 0 set up again")
    x.page(1, "port 1 set up again")
    s.events("end")
    return s


@scenario(host=True)
def interplay_scanout_stop_redo():
    """stop (the root page cleared, the version kept) between frames and
    GPU work, and the guest starting over without negotiating again: before
    it, backings are presented on two heads and task 1 runs a phase; stop
    disconnects both guest displays (8) and drops the task; the guest brings
    up a fresh root page, task 0, the pages (4 each, acked 1) and task 1,
    and the same phase gives the same bytes; the heads present again."""
    x = Scanout("interplay_scanout_stop_redo", interplay_scanout_stop_redo.__doc__, ports=2)
    s = x.s
    x.negotiate(asks=())
    x.root()
    x.kernel_task()
    x.connect(0)
    x.connect(1)
    t = x.task(1, 0x51)
    x.present(0, 0)
    x.present(1, 1)
    x.phase(t, 0)
    first = t
    s.api("stop")
    s.events("stopped")
    s.r(abi.R_VERSION, "stopped")
    s.fault_state("stopped")
    s.root_page = s.alloc_page()
    x.root()
    x.backings = []
    x.kernel_task()
    for p in (0, 1):
        x.connect(p)
    x.tasks = []
    t = x.task(1, 0x51)
    x.phase(t, 0)
    x.same(first, 0, t, 0, "the same phase before and after stop")
    x.present(0, 1)
    x.present(1, 0)
    x.page(0, "port 0 again")
    s.events("end")
    return s


# ---- the EFI framebuffer ----


def _efi_start(x, gpa, w=FW, h=FH, stride=64):
    """EFI over an existing guest page (the measured order: enable, mode,
    depth, stride, start, length)."""
    s = x.s
    s.w(abi.R_EFI_ENABLE, 1)
    s.w(abi.R_EFI_MODE, w | h << 16)
    s.w(abi.R_EFI_DEPTH, 32)
    s.w(abi.R_EFI_STRIDE, stride)
    s.w(abi.R_EFI_START, gpa >> 14)
    s.w(abi.R_EFI_LENGTH, 0x4000)


@scenario(host=True)
def interplay_scanout_efi_written_by_gpu():
    """The EFI framebuffer over a GPU buffer's guest page: task 1's shared
    buffer is EFI's framebuffer (trace callbacks, so each present follows a
    rebuild or a trace hit). A blit fills it and add_one writes its first
    256 bytes, each synchronized back to guest memory; the host's trace hit
    over the page presents again, and the frame is what the GPU wrote. Then
    the driver takes over: task 0, port 0's page, a backing presented, EFI
    off."""
    x = Scanout("interplay_scanout_efi_written_by_gpu", interplay_scanout_efi_written_by_gpu.__doc__, efi=True)
    s = x.s
    x.negotiate(asks=())
    x.root()
    t = x.task(1, 0x61)
    fb, fb_va = t.out, t.out_va
    gpa = t.gpa(fb_va)
    from pgtest.efi_guest import EFI
    x.efi = EFI(s)  # its registers; the framebuffer is the task's page
    _efi_start(x, gpa)
    x.expect("newFrame", label="EFI's first frame")
    x.encode(0, "EFI over the zeroed buffer")
    st = x.exec_cmd(0, t, x.fill_bytes(fb, 0, 0x4000, 0x7F), flags={fb: 5})
    x.go(0, st, "the GPU fills the framebuffer")
    t.reset_cmdbufs()
    s.trace_fire(gpa, 0x40)
    x.expect("newFrame", label="the trace hit")
    x.encode(0, "EFI: the GPU's fill")
    st = x.exec_cmd(0, t, x.compute_into(t, fb, 0x40), flags={fb: 5})
    x.go(0, st, "add_one into the framebuffer")
    t.reset_cmdbufs()
    s.dump(gpa, 0x180, "the framebuffer's first rows")
    s.trace_fire(gpa + 0x40, 0x100)
    x.expect("newFrame", label="the second trace hit")
    x.encode(0, "EFI: add_one's rows")
    s.events("EFI frames")
    x.kernel_task()
    x.connect(0)
    x.present(0, 0)
    x.efi_off()
    x.present(0, 1)
    s.events("end")
    return s


@scenario(host=True)
def interplay_scanout_efi_reset_rebind():
    """EFI, the driver, reset, and EFI again: EFI binds port 0's display and
    presents; the driver connects, presents a backing and runs a GPU phase;
    reset tears down EFI (its registers zeroed), the guest display (8) and
    the task. Enabling EFI again binds the display the reset parked, and
    presents its frame. The driver then starts over on a fresh root page and
    its phase gives the same bytes."""
    x = Scanout("interplay_scanout_efi_reset_rebind", interplay_scanout_efi_reset_rebind.__doc__, efi=True)
    s = x.s
    x.negotiate(asks=())
    x.efi_boot(seed=0x10)
    x.root()
    x.kernel_task()
    x.connect(0)
    x.present(0, 1)
    t = x.task(1, 0x71)
    x.phase(t, 0)
    first = t
    s.api("reset")
    s.events("reset")
    x.efi.regs("after the reset")
    x.efi.write_frame(FW, FH, 64, seed=0x50)
    x.efi.setup(FW, FH, 64)
    x.expect("newFrame", label="EFI again")
    x.efi.regs("EFI again")
    x.encode(0, "EFI's frame after the reset")
    s.events("EFI again")
    s.root_page = s.alloc_page()
    x.negotiate(asks=())
    x.root()
    x.backings = []
    x.kernel_task()
    x.connect(0)
    x.efi_off()
    x.present(0, 0)
    x.tasks = []
    t = x.task(1, 0x71)
    x.phase(t, 0)
    x.same(first, 0, t, 0, "the phase before and after the reset")
    s.events("end")
    return s


@scenario(host=True)
def interplay_scanout_efi_trace_while_paused():
    """EFI presents while the device is paused: its presents go to the
    display directly, not through a guest display, so a trace hit while
    paused presents (a new_frame). Then the driver's port 0 page is connected
    (the guest display takes the display) and presents a backing; paused
    again, a trace hit still presents EFI's frame through the display, and
    after unpause the guest presents its own again."""
    x = Scanout("interplay_scanout_efi_trace_while_paused", interplay_scanout_efi_trace_while_paused.__doc__,
                efi=True)
    s = x.s
    x.negotiate(asks=())
    x.efi_boot(seed=0x33)
    x.root()
    s.api("pause")
    x.efi.write_frame(FW, FH, 64, seed=0x66)
    s.trace_fire(x.efi.gpa(), 0x40)
    x.expect("newFrame", label="EFI while paused")
    x.encode(0, "EFI while paused")
    s.api("unpause")
    s.events("paused EFI")
    x.kernel_task()
    x.connect(0)
    x.present(0, 0)
    s.api("pause", label="again")
    s.trace_fire(x.efi.gpa(), 0x40)
    x.expect("newFrame", label="EFI under the paused nub")
    x.encode(0, "EFI's frame under the paused nub")
    s.api("unpause", label="again")
    x.present(0, 1)
    s.events("end")
    x.page(0, "end")
    return s


# ---- the IOSurface host device under the display ----


@scenario(setups=MAPPED, host=True)
def interplay_scanout_iosfc_test_under_frame():
    """The IOSurface host device writes a presented surface: test (3) of
    surface 1 while head 0 shows it writes i x 8 into each 8-byte word of
    the guest page, and the next encode shows it with no present; child 1
    then fills an 8 x 4 corner red through the mapper texture, the encode
    shows both, and the GPU reads the surface back into task 1's out buffer
    (the same bytes as the guest page). Surface 2, tested before its first
    present, presents the test pattern."""
    x, t = _compositor("interplay_scanout_iosfc_test_under_frame", interplay_scanout_iosfc_test_under_frame.__doc__,
                       gate=None, children=(1,))
    s = x.s
    x.swap_on(0, 0, 1)
    x.iosfc_cmd(CMD_TEST, 1, "test surface 1 under the display")
    x.dump_surface(1, "surface 1 after the test")
    x.encode(0, "the test pattern, not presented again")
    st = x.exec_cmd(1, t, x.fill_rect(t.mtex[1], RED, size=(8, 4, 1)))
    x.go(1, st, "a red corner")
    t.reset_cmdbufs()
    x.encode(0, "the test pattern and the red corner")
    st = x.exec_cmd(1, t, x.readback(t, t.mtex[1], 0))
    x.go(1, st, "read surface 1 back")
    t.reset_cmdbufs()
    s.cmp(t.gpa(t.out_va), s.pa(x.surfaces[1]), FRAME, label="the GPU's view equals the guest page")
    x.iosfc_cmd(CMD_TEST, 2, "test surface 2")
    x.swap_on(0, 0, 2)
    x.page(0, "end")
    s.events("end")
    return s


@scenario(setups=MAPPED, host=True)
def interplay_scanout_iosfc_teardown_under_holders():
    """A descriptor base write while head 0 shows surface 1 and task 1's
    mapper texture, already used to render it red, holds it: the device
    drops its surfaces, but the display and the texture keep the IOSurface
    over the same guest page. Child 1 draws through the old texture: the
    page and the frame show it. A present of mapping 1 is "Invalid
    mappingID" (stamp signalled, nothing called) until the guest maps it
    again, which makes a new IOSurface over the same page: its present
    shows the drawing, and a later render through the old texture shows in
    the new surface too, since both alias the guest page. Surface 2's
    mapper texture was made by the first exec (every exec lists all of
    task 1's objects), so it too outlives the teardown: a render through
    it completes and turns surface 2's page white, though a present of
    mapping 2 is refused."""
    x, t = _compositor("interplay_scanout_iosfc_teardown_under_holders",
                       interplay_scanout_iosfc_teardown_under_holders.__doc__, gate=None, children=(1,))
    s = x.s
    from pgtest.iosfc_guest import I_DESC_BASE
    x.swap_on(0, 0, 1)
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], RED))
    x.go(1, st, "red through the texture")
    t.reset_cmdbufs()
    x.encode(0, "red")
    s.iosfc_w(I_DESC_BASE, s.pa(x.iosfc.table))
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], BLUE, draw=True))
    x.go(1, st, "draw through the texture over the dropped surface")
    t.reset_cmdbufs()
    x.dump_surface(1, "surface 1's page")
    x.encode(0, "the dropped surface, drawn")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="present the dropped mapping")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 2), label="present the other dropped mapping")
    s.quiesce(100)
    s.events("presents of dropped mappings")
    x.page(0, "after the refused presents")
    x.iosfc_cmd(CMD_MAP, 1, "map 1 again")
    x.swap_on(0, 0, 1)
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], GREEN))
    x.go(1, st, "green through the old texture")
    t.reset_cmdbufs()
    x.encode(0, "the new surface shows the old texture's green")
    x.iosfc_cmd(CMD_UNMAP, 1, "unmap 1: the display holds the new surface")
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[2], WHITE))
    x.go(1, st, "white through surface 2's texture")
    t.reset_cmdbufs()
    x.node(1, "child 1")
    x.dump_surface(2, "surface 2's page", length=0x40)
    s.events("end")
    return s


@scenario(setups=MAPPED, host=True)
def interplay_scanout_iosfc_ring_under_held_present():
    """The IOSurface host device's ring runs while a present of surface 3
    waits on the gate: one doorbell maps 3, tests 1 (shown on head 0),
    unmaps 2 and unmaps 3, each with its interrupt and status in order.
    Both unmaps succeed: nothing holds 3 yet, and task 1's mapper texture
    over 2 is made only by an exec that lists it, and none has run. The
    gate opens and the present of 3 is "Invalid mappingID". The guest
    maps 3 again and presents it; then 3 is Busy to unmap; head 0 goes
    back to surface 1, showing the test pattern."""
    x, t = _compositor("interplay_scanout_iosfc_ring_under_held_present",
                       interplay_scanout_iosfc_ring_under_held_present.__doc__, children=(1,))
    s = x.s
    from pgtest.iosfc_guest import desc_v2
    root, data = x.iosfc.surface_pages(1)
    s.mem(s.pa(data), bytes(range(256)) * 2)
    x.iosfc.put_desc(3, desc_v2(root, FW, FH))
    x.surfaces[3] = data
    x.swap_on(0, 0, 1)
    stp = x.swap_cmd(0, 0, 3, barriers=[x.gate_barrier()])
    x.ring(0)
    x.iosfc_ring([(CMD_MAP, 3), (CMD_TEST, 1), (CMD_UNMAP, 2), (CMD_UNMAP, 3)], "one doorbell, four commands")
    x.open_gate()
    x.wait_stamp(0, stp, "the held present of 3")
    s.quiesce(100)
    s.events("the held present found 3 unmapped")
    x.page(0, "after the refused present")
    x.iosfc_cmd(CMD_MAP, 3, "map 3 again")
    x.swap_on(0, 0, 3)
    x.iosfc_cmd(CMD_UNMAP, 3, "unmap 3 while shown")
    x.swap_on(0, 0, 1)
    x.dump_surface(3, "surface 3's page", length=0x40)
    s.events("end")
    return s


@scenario(setups=MAPPED, host=True)
def interplay_scanout_mapper_buffer_frames():
    """Frames written as bytes: task 1's mapper buffer (type 15) over
    surface 1. add_one writes 64 words of the task's source plus one into
    it, a blit fills the rest with 0x40, and child 2 presents it gated on
    child 1's exec; the frame is those bytes read as BGRA. A second exec
    fills the whole surface through the buffer while head 0 shows it; the
    encode follows the buffer's write with no present."""
    x, t = _compositor("interplay_scanout_mapper_buffer_frames", interplay_scanout_mapper_buffer_frames.__doc__,
                       gate=None, buffers=(1,))
    s = x.s
    b = t.mbuf[1]
    st = x.exec_cmd(1, t, x.compute_into(t, b, 0) + x.fill_bytes(b, 0x100, FRAME - 0x100, 0x40))
    stp = x.swap_cmd(2, 0, 1, barriers=[(1, st)])
    x.ring(2, 1)
    x.wait_stamp(2, stp, "the buffer's frame, presented from child 2")
    t.reset_cmdbufs()
    x.expect("newFrame", label="surface 1")
    x.encode(0, "add_one's words and the fill")
    x.dump_surface(1, "surface 1")
    st = x.exec_cmd(1, t, x.fill_bytes(b, 0, FRAME, 0xC0))
    x.go(1, st, "fill the shown surface through the buffer")
    t.reset_cmdbufs()
    x.encode(0, "the fill, not presented again")
    x.iosfc_cmd(CMD_UNMAP, 1, "unmap 1: the buffer and the display hold it")
    s.events("end")
    return s


# ---- order across FIFOs ----


@scenario(setups=MAPPED, host=True)
def interplay_scanout_frame_chain():
    """Six frames on one head from three FIFOs, each render gated on the
    previous present and each present on its render: child 1 renders and
    child 2 presents surface 1, the root renders and child 1 presents
    surface 2, child 2 renders and the root presents surface 1, and so on.
    Everything is written first and the FIFOs are rung in reverse. Each
    render after the first also waits on the gate, which the host opens
    only after it has seen the frame before (its new_frame) and encoded it:
    new_frame is a dispatch data source, and two presents closer than its
    handler would merge into one call. So every frame is encoded, and they
    are the renders in the order the barriers make."""
    x, t = _compositor("interplay_scanout_frame_chain", interplay_scanout_frame_chain.__doc__)
    s = x.s
    colors = (RED, GREEN, BLUE, WHITE, GREY, (0.0, 0.5, 1.0, 1.0))
    plan = [((1, 2), 1), ((0, 1), 2), ((2, 0), 1), ((1, 2), 2), ((0, 1), 1), ((2, 0), 2)]
    prev = None
    stamps = []
    for i, ((rf, pf), mid) in enumerate(plan):
        bars = ([prev, x.gate_barrier(i)] if prev else [])
        st = x.exec_cmd(rf, t, x.compose(t, t.mtex[mid], colors[i], draw=(i == 3)), barriers=bars)
        sp = x.swap_cmd(pf, 0, mid, barriers=[(rf, st)])
        prev = (pf, sp)
        stamps.append((pf, sp))
    x.ring(0, 2, 1)
    for i, (pf, sp) in enumerate(stamps):
        x.wait_stamp(pf, sp, "frame %d presented" % i)
        x.expect("newFrame", label="frame %d" % i)
        x.encode(0, "frame %d" % i)
        if i + 1 < len(stamps):
            x.open_gate()
    t.reset_cmdbufs()
    x.dump_surface(1, "surface 1: frame 4")
    x.dump_surface(2, "surface 2: frame 5")
    x.stamps(4, "stamp slots")
    s.api("present_counts", display=0)
    x.page(0, "end")
    s.events("end")
    return s


@scenario(setups=MAPPED, host=True)
def interplay_scanout_held_encode_sees_render():
    """A frame the host encodes and holds (encode with hold, its command
    buffer not yet committed) while the GPU draws into the surface it
    shows: committed after child 1's draw completed, it draws the surface
    as it is at the commit, the drawing, not the pattern shown when it was
    encoded. It counts as a host present. A present of surface 2 after
    the commit finds no frame in flight and presents at once."""
    x, t = _compositor("interplay_scanout_held_encode_sees_render", interplay_scanout_held_encode_sees_render.__doc__,
                       gate=None, children=(1,))
    s = x.s
    x.swap_on(0, 0, 1)
    s.encode(0, FW, FH, hold=True, label="held", samples=[[0, 0]])
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], BLUE, draw=True))
    x.go(1, st, "draw into the held frame's surface")
    t.reset_cmdbufs()
    s.encode_commit(0, label="the held frame, committed after the draw")
    x.swap_on(0, 0, 2)
    s.api("present_counts", display=0)
    s.events("end")
    return s


# ---- the guest's page, the host's display, and pause ----


@scenario(setups=MAPPED, host=True)
def interplay_scanout_disconnect_under_render():
    """The guest disconnects head 1's page (opcode 1, page 0) on the root
    FIFO while child 1 renders for it and child 2's present to port 1 is
    gated on both: the guest display goes (8 into the old page, the root
    word cleared), and the present then finds no page on port 1: refused,
    its stamp signalled, nothing called. Head 0 carries on. The guest
    connects a fresh page on port 1 (4, expectedAck 1 again: a new page
    starts at 0), acks, and the gated render's surface presents there."""
    x, t = _compositor("interplay_scanout_disconnect_under_render", interplay_scanout_disconnect_under_render.__doc__,
                       ports=2, gate=None)
    s = x.s
    x.swap_on(0, 1, 2)
    old = x.pages[1]
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], GREEN, draw=True))
    sd = s.fifo.cmd(abi.OP_DISPLAY_SET_SHARED_STATE_PAGE, __import__("struct").pack("<II", 1, 0))
    sp = x.swap_cmd(2, 1, 1, barriers=[(1, st), (0, sd)])
    x.ring(2, 0, 1)
    x.wait_stamp(2, sp, "the present to the disconnected port")
    t.reset_cmdbufs()
    s.quiesce(100)
    s.events("disconnected under the render")
    s.diff(label="port 1's old page", gpa=s.pa(old), length=0x4000)
    s.dump(s.pa(s.root_page, abi.ROOT_DISPLAY_WORD), 8, "root page: display words")
    s.fault_state("after the refused present")
    x.swap_on(0, 0, 1)
    x.connect(1)
    x.swap_on(0, 1, 1)
    x.page(1, "port 1's new page")
    s.events("end")
    return s


@scenario(host=True)
def interplay_scanout_release_while_paused():
    """The host unplugs a presenting display while the device is paused,
    with GPU work queued: the detach runs no state machine (the guest
    display is paused), so the page keeps its modes; the NOP-gated phase
    waits. unpause runs the machine with no client: 8 (disconnect), mode
    count 0, state 0; the queued phase runs, and presents to the port
    complete at once with nothing called. A display plugged in again
    connects (4, expectedAck 2) and takes the next frame; the second phase
    equals the first."""
    x = Scanout("interplay_scanout_release_while_paused", interplay_scanout_release_while_paused.__doc__)
    s = x.s
    x.negotiate(asks=())
    x.root()
    x.kernel_task()
    x.connect(0)
    t = x.task(1, 0x23)
    x.present(0, 0)
    x.phase(t, 0)
    s.api("pause")
    s.api("release_display", display=0)
    st = x.exec_cmd(0, t, t.blit_segments(1, 0x5A) + t.compute_segments(1) + t.render_segments(1))
    x.ring(0)
    s.quiesce(100)
    s.events("released while paused")
    x.page(0, "released while paused")
    s.api("unpause")
    x.wait_stamp(0, st, "the phase queued while paused")
    t.reset_cmdbufs()
    s.wait(gpa=x.ss(0, SS_MODE_COUNT), mask=0xFFFF, eq=0, label="port 0 disconnected")
    s.events("unpaused")
    x.page(0, "unpaused")
    x.same(t, 0, t, 1, "the phase across the pause")
    st = x.present_cmd(0, 0, 1)
    x.go(0, st, "a present with no display")
    s.quiesce(100)
    s.events("a present with no display")
    x.hotplug(0)
    x.present(0, 1, display_index=1)
    x.page(0, "the new display")
    s.events("end")
    return s


@scenario(setups=MAPPED, host=True)
def interplay_scanout_update_pending_presents():
    """Presents while a mode-list update waits for its ack: the host adds a
    mode (12, expectedAck 2, ackPending), the guest keeps rendering and
    presenting from two children, and every frame reaches the host with the
    ack still pending; an identical list then changes nothing; a second
    change while the ack is pending only updates the list ("bogus ack"),
    and the ack raises the update it held (12, expectedAck 3), which the
    guest acks too."""
    x, t = _compositor("interplay_scanout_update_pending_presents", interplay_scanout_update_pending_presents.__doc__,
                       gate=None)
    s = x.s
    modes2 = x.modes + ((32, 16, 60),)
    x.set_modes(0, modes2, label="a mode added")
    s.wait(gpa=x.ss(0, 0xFA9), mask=0xFF, eq=1, label="the update's ack pending")
    x.bump(0)
    for i, (fid, color) in enumerate(((1, RED), (2, GREEN))):
        st = x.exec_cmd(fid, t, x.compose(t, t.mtex[1], color))
        x.go(fid, st, "render %d" % i)
        t.reset_cmdbufs()
        x.swap_on(3 - fid, 0, 1)
    x.set_modes(0, modes2, label="the same list")
    x.set_modes(0, ((FW, FH, 60),), label="another list, ack pending")
    s.quiesce(100)
    s.events("updates while presenting")
    x.page(0, "ack pending")
    x.ack(0, x.acks[0])
    s.wait(gpa=x.ss(0, 0x200), eq=3, label="the held update raised")
    s.events("acked")
    x.page(0, "acked")
    x.ack_next(0)
    x.swap_on(0, 0, 2)
    x.page(0, "end")
    s.events("end")
    return s


# ---- other pixel formats ----


@scenario(setups=MAPPED, host=True)
def interplay_scanout_rgha_rendered():
    """An 'RGhA' surface (RGBA16Float, 8 bytes a pixel) rendered by the
    GPU through an RGBA16Float mapper texture and presented with opcode 8,
    between BGRA frames on the same head: each format change calls
    mode_change with the new OSType; the half-float clear colour is in the
    surface's guest page exactly."""
    x = Scanout("interplay_scanout_rgha_rendered", interplay_scanout_rgha_rendered.__doc__, setups=MAPPED,
                iosfc=True)
    s = x.s
    x.negotiate(asks=())
    x.root()
    x.kernel_task()
    x.connect(0)
    x.iosfc_start()
    x.surface(1, 3)
    x.surface(4, 0, fmt=b"RGhA", bpe=8, pixels=bytes(FW * FH * 8))
    t = x.task(1, 0x19, extra=x.mapper_objects((1, 4), sizes={4: (FW, FH, 115)}))
    x.child(1)
    x.swap_on(0, 0, 1)
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[4], (0.25, 0.5, 1.0, 1.0)))
    x.go(1, st, "clear the half-float surface")
    t.reset_cmdbufs()
    s.dump(s.pa(x.surfaces[4]), 0x40, "surface 4: half floats")
    x.swap_on(0, 0, 4)
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], GREY))
    x.go(1, st, "grey")
    t.reset_cmdbufs()
    x.swap_on(0, 0, 1)
    s.events("end")
    return s


# ---- a service death beside the display ----


@scenario(setups=("isolate",), host=True)
def interplay_scanout_service_death_beside_frames():
    """Task 1's service dies (an exec range at the task's end) after it
    rendered surface 1, which head 0 shows: the exec faults child 1, the
    frame stays (an encode draws it), and task 2, in its own service,
    renders surface 2 and presents it from child 2. With task 1's service
    gone and head 0 moved to surface 2, surface 1 unmaps (the texture the
    dead service held is no host reference); a present of mapping 1 is
    then refused. Isolate only: in-process the death
    kills the host."""
    x, t = _compositor("interplay_scanout_service_death_beside_frames",
                       interplay_scanout_service_death_beside_frames.__doc__, gate=None, setups=("isolate",))
    s = x.s
    t2 = x.task(2, 0x22, extra=x.mapper_objects((2,)))
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], RED))
    x.go(1, st, "task 1 renders surface 1")
    t.reset_cmdbufs()
    x.swap_on(0, 0, 1)
    x.children[1].cmd(abi.OP_EXEC_INDIRECT3, x.death_payload(t))
    x.ring(1)
    x.node_fault(1, "task 1's service dies")
    x.node(1, "child 1 faulted")
    x.encode(0, "the frame task 1 rendered")
    st = x.exec_cmd(2, t2, x.compose(t2, t2.mtex[2], BLUE, draw=True))
    sp = x.swap_cmd(2, 0, 2)
    x.go(2, sp, "task 2 renders and presents from child 2")
    t2.reset_cmdbufs()
    x.expect("newFrame", label="surface 2")
    x.encode(0, "task 2's frame")
    x.iosfc_cmd(CMD_UNMAP, 1, "unmap surface 1 after task 1's death")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1), label="present surface 1 again")
    s.quiesce(100)
    s.events("end")
    x.page(0, "end")
    return s


# ---- opcode 6 from a child FIFO ----


@scenario(setups=IN_PROCESS, host=True)
def interplay_scanout_opcode6_children():
    """Opcode 6 (DisplayTransaction2) from child FIFOs with backings the
    GPU wrote through ref textures: child 1 presents backing 0 after a red
    fill, child 2 presents backing 1 after the triangle; then child 1's
    opcode 6 of a reference that is not a backing faults child 1 (opcode 6
    faults where 7 signals), while child 2 keeps presenting; 0x1028
    resumes child 1, which presents again. In-process only: ref textures
    abort the service."""
    from pgtest.display_guest import p_present2
    x = Scanout("interplay_scanout_opcode6_children", interplay_scanout_opcode6_children.__doc__,
                setups=IN_PROCESS)
    s = x.s
    x.negotiate(asks=())
    x.root()
    x.kernel_task()
    x.connect(0)
    t = x.task(1, 0x29, extra=x.ref_objects((0, 1)))
    x.child(1)
    x.child(2)

    def present6(fid, frame, barriers=(), ref=None):
        r = x.backings[frame][0] if ref is None else ref
        return x.fifo(fid).cmd(abi.OP_DISPLAY_TRANSACTION2, p_present2(0, r), barriers=barriers)

    st = x.exec_cmd(0, t, x.fill_rect(t.rtex[0], RED))
    sp = present6(1, 0, barriers=[(0, st)])
    x.ring(1, 0)
    x.wait_stamp(1, sp, "backing 0 from child 1")
    t.reset_cmdbufs()
    x.expect("newFrame", label="backing 0")
    x.encode(0, "backing 0, red")
    st = x.exec_cmd(0, t, x.compose(t, t.rtex[1], BLUE, draw=True))
    sp = present6(2, 1, barriers=[(0, st)])
    x.ring(2, 0)
    x.wait_stamp(2, sp, "backing 1 from child 2")
    t.reset_cmdbufs()
    x.expect("newFrame", label="backing 1")
    x.encode(0, "backing 1, the triangle")
    present6(1, 0, ref=t.rtex[0] + 40)
    x.ring(1)
    x.node_fault(1, "opcode 6 of a reference that is no backing")
    x.node(1, "child 1 faulted")
    sp = present6(2, 0)
    x.go(2, sp, "child 2 presents beside the fault")
    x.expect("newFrame", label="backing 0 from child 2")
    resume_child(s, 1)
    sp = present6(1, 1)
    x.go(1, sp, "child 1 resumed, presents")
    x.expect("newFrame", label="backing 1 from child 1")
    x.encode(0, "backing 1 again")
    x.node(1, "child 1 resumed")
    s.fault_state("end")
    s.events("end")
    return s


# ---- four heads ----


@scenario(setups=MAPPED, host=True)
def interplay_scanout_four_heads():
    """Four heads, each fed by its own FIFO: surfaces 1-4 rendered by child
    1 in one exec, then each presented from a different FIFO (root,
    children 1-3), each present gated on the one before and the first on
    the render, so the heads' callbacks come in port order; head 2 also gets backing 0
    through opcode 7. Each head's frame is encoded, the 0x1014 bits and
    every page's present count and presented surface are locked; head 3 is
    unplugged and replugged between frames (8, then 4 with expectedAck 2)."""
    x, t = _compositor("interplay_scanout_four_heads", interplay_scanout_four_heads.__doc__, ports=4,
                       surfaces=((1, 3), (2, 5), (3, 7), (4, 9)), children=(1, 2, 3), gate=None)
    s = x.s
    segs = []
    for mid, color in ((1, RED), (2, GREEN), (3, BLUE), (4, GREY)):
        segs += x.compose(t, t.mtex[mid], color, draw=(mid == 4))
    st = x.exec_cmd(1, t, segs)
    sps = []
    prev = (1, st)
    for port, fid in ((0, 0), (1, 1), (2, 2), (3, 3)):
        sps.append((fid, x.swap_cmd(fid, port, port + 1, barriers=[prev])))
        prev = sps[-1]
    x.ring(0, 3, 2, 1)
    for fid, sp in sps:
        x.wait_stamp(fid, sp, "head %d presented" % fid)
    t.reset_cmdbufs()
    x.expect("newFrame", n=4, label="four heads")
    for port in range(4):
        x.encode(port, "head %d" % port)
    s.events("four heads")
    s.r(abi.R_IRQ_DISPLAY, "four heads")
    st = x.present_cmd(2, 2, 0)
    x.go(2, st, "backing 0 on head 2 from child 2")
    x.expect("newFrame", label="backing 0 on head 2")
    x.encode(2, "head 2: backing 0")
    x.unplug(3, 3)
    x.hotplug(3)
    x.swap_on(3, 3, 1, index=4)
    for port in range(4):
        s.dump(x.ss(port, 0xF98), 0x20, "port %d: present count, surface, mapping" % port)
    s.events("end")
    return s


# ---- deaths, each the scenario's last action ----


@scenario(setups=MAPPED)
def interplay_scanout_doorbell_paused_dies():
    """The cursor doorbell (0x1220) while the device is paused, after GPU
    frames and cursor moves: GuestDisplay::doorbell asserts it is not paused
    and the host dies at the register write. No encode: nothing here reports
    the host."""
    x, t = _compositor("interplay_scanout_doorbell_paused_dies", interplay_scanout_doorbell_paused_dies.__doc__,
                       host=False, gate=None, children=(1,))
    s = x.s
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], RED))
    x.go(1, st, "red")
    t.reset_cmdbufs()
    x.swap_on(0, 0, 1, encode=False)
    x.cursor(0, 2, 2)
    s.api("pause")
    s.events("paused")
    s.w(abi.R_DISPLAY_DOORBELL, 0)
    return s


@scenario
def interplay_scanout_cursor_position_paused_dies():
    """The host reads a display's cursor position (rvg_display_cursor_position)
    while the device is paused, after backings were presented from a child FIFO
    and a phase ran: GuestDisplay::cursor_position's read of the page asserts
    it is not paused and the host dies at the call."""
    x = Scanout("interplay_scanout_cursor_position_paused_dies",
                interplay_scanout_cursor_position_paused_dies.__doc__, host=False)
    s = x.s
    x.negotiate(asks=())
    x.root()
    x.kernel_task()
    x.connect(0)
    t = x.task(1, 0x41)
    x.child(1)
    st = x.present_cmd(1, 0, 0)
    x.go(1, st, "backing 0 from child 1")
    x.expect("newFrame")
    x.cursor(0, 9, 3)
    x.phase(t, 0, fid=1)
    s.api("cursorPosition", display=0, label="running")
    s.api("pause")
    s.events("paused")
    s.api("cursorPosition", display=0, label="paused")
    return s


@scenario(setups=MAPPED)
def interplay_scanout_mode_count_over_dies():
    """A guest that writes a mode count above 128 at +0x208 while frames
    flow: presents do not look at it; the next state machine run that
    updates the modes (the host's new mode list) asserts in
    GuestDisplay::sync_mode_list and the host dies at the set_modes call."""
    x, t = _compositor("interplay_scanout_mode_count_over_dies", interplay_scanout_mode_count_over_dies.__doc__,
                       host=False, gate=None, children=(1,))
    s = x.s
    s.u32(x.ss(0, SS_MODE_COUNT), 200)
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], BLUE))
    x.go(1, st, "blue")
    t.reset_cmdbufs()
    x.swap_on(0, 0, 1, encode=False)
    x.swap_on(1, 0, 2, encode=False)
    x.page(0, "a mode count of 200")
    s.events("presented")
    x.set_modes(0, ((32, 16, 60),), label="a new mode list")
    return s


# ---- colour spaces and compositor parameters between frames ----


@scenario(setups=MAPPED, host=True)
def interplay_scanout_icc_between_frames():
    """A guest ICC profile (12) and compositor parameters (11) between GPU
    frames: the profile makes the next mapped present call mode_change
    though nothing else changed, once; a backing presented with opcode 7
    in between does not carry it. Compositor parameters reach nothing a
    host sees. The page's ICC record and colour-space fields are
    locked."""
    from pgtest.display_guest import icc_profile, p_compositor, p_icc
    x, t = _compositor("interplay_scanout_icc_between_frames", interplay_scanout_icc_between_frames.__doc__,
                       gate=None, children=(1,))
    s = x.s
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], GREY, draw=True))
    x.go(1, st, "the triangle")
    t.reset_cmdbufs()
    x.swap_on(0, 0, 1)
    prof = icc_profile(1.8)
    va = x.t0.blob(prof)
    s.submit(abi.OP_DISPLAY_SET_GUEST_ICC_PROFILE, p_icc(0, len(prof), va, len(prof)), label="an ICC profile")
    s.submit(abi.OP_DISPLAY_COMPOSITOR_PARAMETERS, p_compositor(0, 3, 4, 7, 2), label="compositor parameters")
    x.present(0, 0)
    s.events("the backing after the profile")
    x.swap_on(0, 0, 1)
    s.events("the mapped present after the profile")
    x.swap_on(0, 0, 2)
    s.events("the next mapped present")
    s.dump(x.ss(0, 0xFC8), 0x10, "ICC record")
    x.page(0, "end")
    return s


# ---- more of what a restore does ----


@scenario(setups=MAPPED, host=True)
def interplay_scanout_suspend_asleep_hidden():
    """A migration with head 0 asleep after a GPU frame and head 1's cursor
    hidden: on the fresh device head 0 is only put to sleep again (no
    present, GuestDisplay::finish_restore), and head 1's glyph is replayed
    with its show of 0. Waking head 0 then has no frame to signal (the new
    display never had one); the guest's next present after a render shows.
    Head 1 presents on as before."""
    from pgtest.display_guest import p_sleep
    x, t = _compositor("interplay_scanout_suspend_asleep_hidden", interplay_scanout_suspend_asleep_hidden.__doc__,
                       ports=2, gate=None, children=(1,))
    s = x.s
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], RED))
    x.go(1, st, "red")
    t.reset_cmdbufs()
    x.swap_on(0, 0, 1)
    s.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(0, 1), label="head 0 asleep")
    x.swap_on(0, 1, 2)
    x.cursor(1, 6, 6)
    x.hide_cursor(1)
    s.events("before the suspend")
    x.suspend()
    x.resume()
    x.expect("cursorGlyph", label="head 1's glyph replayed")
    x.expect("cursorShow", label="head 1's show (0) replayed")
    x.expect("newFrame", label="head 1 re-presented")
    s.events("resumed")
    x.page(0, "head 0 restored")
    s.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(0, 0), label="head 0 awake")
    s.quiesce(100)
    s.events("woken with no frame")
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], GREEN, draw=True))
    x.go(1, st, "a render after the wake")
    t.reset_cmdbufs()
    x.swap_on(0, 0, 1)
    x.swap_on(0, 1, 1)
    s.events("end")
    return s


@scenario(setups=MAPPED, host=True)
def interplay_scanout_suspend_after_teardown():
    """A migration after the guest rewrote the IOSurface host device's
    descriptor base (every surface dropped) while head 0 showed surface 1:
    the restore re-presents mapping 1, which the device no longer has
    (the mapped present raises for a mapping with no surface, and
    GuestDisplay::finish_restore catches it and skips its state machine
    call), so nothing is presented. The guest maps 1 again; its first render
    on the resumed device faults child 1, because the exec lists every
    object of task 1 and the mapper texture over mapping 2, made anew, finds
    no surface ("Attempting to create mapper ref against invalid surface").
    Once 2 is mapped again and child 1 resumed, the render lands in the new
    surface 1 and is presented."""
    from pgtest.iosfc_guest import I_DESC_BASE
    x, t = _compositor("interplay_scanout_suspend_after_teardown", interplay_scanout_suspend_after_teardown.__doc__,
                       gate=None, children=(1,))
    s = x.s
    x.swap_on(0, 0, 1)
    s.iosfc_w(I_DESC_BASE, s.pa(x.iosfc.table))
    s.events("torn down")
    x.suspend()
    x.resume()
    s.quiesce(100)
    s.events("resumed: nothing presented")
    x.page(0, "resumed")
    x.iosfc_cmd(CMD_MAP, 1, "map 1 again")
    x.exec_cmd(1, t, x.compose(t, t.mtex[1], BLUE, draw=True))
    x.ring(1)
    x.node_fault(1, "the exec lists the texture over unmapped 2")
    t.reset_cmdbufs()
    x.node(1, "child 1 faulted")
    x.dump_surface(1, "surface 1 untouched")
    x.iosfc_cmd(CMD_MAP, 2, "map 2 again")
    resume_child(s, 1)
    x.wait_stamp(1, x.children[1].stamp, "the faulted stamp, published by the resume")
    st = x.exec_cmd(1, t, x.compose(t, t.mtex[1], BLUE, draw=True))
    x.go(1, st, "render into the new surface")
    t.reset_cmdbufs()
    x.swap_on(0, 0, 1)
    x.dump_surface(1, "surface 1")
    x.page(0, "end")
    s.events("end")
    return s
