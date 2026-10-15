"""Seeded random walks over display state (pgtest/randdisplay.py): each seed
is one scenario, interplay_displaywalk_<seed>, whose program the generator
builds from the seed alone.

A walk drives one or two display ports and the EFI framebuffer through
random legal guest and host actions: connects on fresh and reused pages,
acks, disconnects, hot plugs and unplugs, mode lists (the same one, new
ones, 130 modes), presents of task 0's backings (opcodes 6, 7) and of
mapper surfaces (8) with gamma tables of 2 to 1024 entries, colour
matrices, ICC profiles, protection options, sleep and wake, cursor glyphs
from 1 x 1 to 256 x 256, shows, moves and the doorbell, refused presents
and commands the version gates, the guest rewriting backings and surfaces
under the frames, backings deleted and surfaces unmapped and mapped again;
pause and unpause with host changes and guest commands rung while paused,
suspend onto a fresh device and resume, reset and stop with the guest's
set-up again, and EFI at boot and after a reset. Some programs end in a
death docs/testing.md lists (7.3, 7.4).

A model restated from the guest display's and the display's code predicts
every observation: the callbacks and interrupts, every shared page byte for
byte (compared with a mirror of the model's bytes), 0x1014 and 0x1018, the
root page's display words, the encoded pixels, present counts and cursor
positions (pgtest.randdisplay.check reads the expected files back; python3
-m pgtest.randdisplay). Programs with mapper surfaces run in map and
isolate, the others in all three setups. The encoded frames make every
program host-keyed.
"""
from pgtest.randdisplay import build
from pgtest.registry import scenario

# (seed, steps): short programs, medium ones, long ones.
SEEDS = ([(s, 6 + (s * 5) % 9) for s in range(1, 21)]
         + [(s, 20 + (s * 7) % 21) for s in range(21, 37)]
         + [(s, 60 + (s * 11) % 41) for s in range(37, 45)])


@scenario
def interplay_displaywalk():
    return [build(seed, steps, __doc__) for seed, steps in SEEDS]
