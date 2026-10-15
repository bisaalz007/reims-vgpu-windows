"""Seeded walks that land device lifecycle events in live mixed work
(pgtest/lifewalk_util.py): each seed is one scenario,
interplay_lifewalk_<seed>, whose program the generator builds from the seed
alone.

Each round of a walk builds a live state and lands one lifecycle event in
it. The live state composes what the earlier walkers drive apart: GPU
execs on the root FIFO and worker child FIFOs, chained into order
witnesses (dagwalk, randmix), some held behind a gate FIFO or behind held
work, some rung and only read (in flight); IOSurface mapper surfaces
filled by the GPU through mapper buffers and presented with opcode 8, and
task 0's backings presented with opcode 7 (scanout); cursors; a faulted
worker child (randfault); and in some programs the EFI framebuffer at
boot. The event is pause and unpause (with work rung while paused and a
new mode list), the suspend sequence onto a fresh device (with work rung
after rvg_device_will_suspend), rvg_device_will_suspend then unpause (no
display), reset, stop, a fresh device, the root FIFO disabled and enabled
again, a version change or a dip below EFI's version, the host unplugging and
plugging a display or changing its modes, a worker child deleted with its work
and defined again, or nothing. The guest then resolves it as a driver would:
opens the gate, recovers the fault the way the event left it, redoes its set-up
and rings abandoned work again.

The model predicts every observation: every item's slot bytes and every
surface's bytes (written into mirror pages and compared), each stamp
(waited for; held ones time out in waits labelled [held]), the fault
registers after each recovery, one new_frame per present and the resume's
replays of glyphs, shows and mappings. Programs with mapper surfaces run
in the map and isolate setups, the others in all three; some programs end
in a death docs/testing.md 7.3 lists, with work held, as their last
action.
"""
from pgtest.lifewalk_util import DEATHS, build
from pgtest.registry import scenario

# (seed, rounds, death): short programs, long ones, and ones that end in
# each death, twice.
SEEDS = ([(s, 4 + (s * 5) % 5, None) for s in range(1, 13)]
         + [(s, 16 + (s * 7) % 9, None) for s in range(13, 19)]
         + [(s, 5 + (s * 3) % 7, DEATHS[s % 5]) for s in range(19, 29)])


@scenario
def interplay_lifewalk():
    return [build(seed, rounds, __doc__, death=death) for seed, rounds, death in SEEDS]




@scenario(host=True)
def interplay_lifewalk_resume_page_modes():
    """A resume onto a display whose mode list differs from the one the
    guest acked. With nothing presented on the port,
    GuestDisplay::finish_restore runs the guest display's state machine: the
    target display's list goes into the page with an update (12),
    expectedAck 3 and ackPending set, which the guest acks. A list the host
    sets while that ack is pending only updates the list; the ack raises the
    held update (expectedAck 4); after its ack, a list equal to the page's
    raises nothing and a new one raises 12 again. With a backing presented
    (opcode 7) before the suspend (_after_present), the restore first
    re-presents the page's mapping, 0, which throws and skips the state
    machine: the page keeps the acked list, nothing is pending, and the host
    setting the page's list raises nothing, where a new one raises 12. Found
    by the lifecycle walk, whose model tracks the page's list apart from the
    display's."""
    from pgtest.lifewalk_util import MODES_A, MODES_B, MODES_C
    from pgtest.scanout_util import Scanout
    out = []
    for suffix in ("", "_after_present"):
        name = "interplay_lifewalk_resume_page_modes" + suffix
        x = Scanout(name, interplay_lifewalk_resume_page_modes.__doc__, ports=1, displays=(0,), modes=MODES_A)
        s = x.s
        x.negotiate(asks=())
        x.root()
        x.connect(0)
        if suffix:
            x.kernel_task()
        x.set_modes(0, MODES_B, label="list B")
        s.wait(gpa=x.ss(0, 0xFA9), mask=0xFF, eq=1, label="B's update pending")
        x.ack_next(0)
        if suffix:
            x.present(0, 0)
        s.dump(x.ss(0, 0x200), 0x40, "B acked: expectedAck, count, modes")
        s.events("B acked")
        x.suspend()
        x.resume()
        if suffix:
            s.quiesce(100)
            s.events("resumed onto a display with list A")
            s.dump(x.ss(0, 0x200), 0x40, "resumed: the page keeps list B")
            s.dump(x.ss(0, 0xFA8), 0x18, "resumed: nothing pending")
            x.set_modes(0, MODES_B, label="B again: the page's list")
            s.quiesce(100)
            s.events("B again")
            s.dump(x.ss(0, 0xFA8), 0x18, "B again: nothing pending")
            x.set_modes(0, MODES_C, label="list C")
            s.wait(gpa=x.ss(0, 0xFA9), mask=0xFF, eq=1, label="C's update pending")
            s.events("C")
            x.ack_next(0)
            s.dump(x.ss(0, 0x200), 0x40, "C acked")
            s.events("end")
            out.append(s)
            continue
        s.wait(gpa=x.ss(0, 0xFA9), mask=0xFF, eq=1, label="the resume's update pending")
        s.events("resumed onto a display with list A")
        s.dump(x.ss(0, 0x200), 0x40, "resumed: list A, expectedAck 3")
        s.dump(x.ss(0, 0xFA8), 0x18, "resumed: ackPending, lastAck")
        x.set_modes(0, MODES_C, label="C while the ack is pending")
        s.quiesce(100)
        s.events("C while pending")
        s.dump(x.ss(0, 0x200), 0x40, "C while pending: the page unchanged")
        x.ack_next(0)
        s.wait(gpa=x.ss(0, 0x200), eq=4, label="the ack raised the held update")
        s.events("acked: the held update")
        s.dump(x.ss(0, 0x200), 0x40, "the held update: list C")
        x.ack_next(0)
        x.set_modes(0, MODES_C, label="C again: the page's list")
        s.quiesce(100)
        s.events("C again")
        s.dump(x.ss(0, 0xFA8), 0x18, "C again: nothing pending")
        x.set_modes(0, MODES_B, label="list B")
        s.wait(gpa=x.ss(0, 0xFA9), mask=0xFF, eq=1, label="B's update pending")
        x.ack_next(0)
        s.dump(x.ss(0, 0x200), 0x40, "B acked")
        s.events("end")
        out.append(s)
    return out
