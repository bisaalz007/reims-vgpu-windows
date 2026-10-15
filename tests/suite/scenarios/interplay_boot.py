"""A guest's whole life (pgtest/boot_util.py): each seed is one scenario,
interplay_boot_<seed>, whose program the generator builds from the seed
alone: one device session that goes through the phases a macOS guest goes
through under QEMU's apple-gfx, end to end, with every subsystem busy
where a real guest's would be.

The firmware's EFI frames on port 0's display (at version 0, before the
driver, or after its version negotiation); the driver's takeover (version
negotiation, root page and root FIFO, task 0, device info, child FIFOs, the
gate and GPU tasks, EFI turned off, each head's shared page connected and
acked, the cursor's glyph, show and position); the window server (the
IOSurface host device's ring and table, mapper surfaces, a compositor task
with a mapper buffer over each); steady state (the GPU copies a window into a
surface and the compositor presents it with opcode 8 from a FIFO behind
the copy,
opcode-7 presents of task 0's backings, gamma tables, colour matrices, ICC
profiles, cursor moves and glyphs, beside the marathon's app batches, app
faults and their recovery, lifetime and memory changes); the host
hot-plugging displays and setting mode lists, each update acked; the heads
put to sleep, presents while asleep, woken; a host pause now and then;
the suspend onto a fresh device, with work held on the gate across it, and
steady state after the resume; and the shutdown: everything released and
compared, cursors hidden, heads disconnected, the compositor's buffers and
task deleted, the surfaces unmapped, the backings deleted, the child FIFOs,
the gate and every task deleted, the root FIFO disabled, and the device
released, or reset and booted to the firmware's EFI frames again. Before
each phase ends the seed decides what is busy (work held on the gate, work
running free, a faulted child, EFI frames).

The model is the display walk's and the marathon's (pgtest/randdisplay.py,
pgtest/marathon_util.py), extended to the compositor's copies: every
callback, interrupt, shared page, encoded pixel, present count, GPU slot,
scratch page, surface and fault register is predicted, and every wait
succeeds but those labelled [held] and [dies]; python3 -m pgtest.boot_util
reads the expected files back against it. Programs with mapper surfaces run
in the map and isolate setups, the others in all three. Some programs stop
at a realistic moment in a death docs/testing.md 7.3 and 7.4 list, with
work held on the gate: a one-entry gamma table in steady state, a guest
mode count over 128 when the host sets a new list, rvg_display_cursor_position
while the host has the device paused in the sleep phase, and at the suspend the
cursor doorbell while paused or a second pause.
"""
from pgtest.boot_util import build
from pgtest.registry import scenario

# (seed, length, death)
SEEDS = ([(s, "medium", None) for s in range(1, 11)]
         + [(s, "short", None) for s in range(11, 15)]
         + [(s, "long", None) for s in range(15, 19)]
         + [(19, "medium", "gamma_one"), (20, "medium", "mode_count_over"), (21, "short", "cursor_paused"),
            (22, "medium", "doorbell_paused"), (23, "short", "pause_twice"), (24, "long", "gamma_one")])


@scenario
def interplay_boot():
    return [build(seed, __doc__, length, death) for seed, length, death in SEEDS]
