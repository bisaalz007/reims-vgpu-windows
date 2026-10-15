"""Seeded random guest programs (pgtest/randgen.py): each seed is one
scenario, interplay_rand_<seed>, whose program the generator builds from
the seed alone.

A program is a walk over legal guest and host actions across every
subsystem: tasks and their objects made and deleted, blit, compute, render
and info work on the root FIFO and child FIFOs (serialized by barriers, or
at once on different tasks), display presents, cursor moves and hot plugs,
IOSurface maps, tests and swaps, pause and unpause, suspend and resume on a
fresh device, child FIFOs deleted and defined again, and reset, stop or a
fresh device with the guest doing its set-up again. After every step the
guest can observe, the scenario waits for it; checkpoints are guest memory,
registers and event counts, never timing.

The check is differential: before the walk a control task runs every item of
work of the program, uninterrupted, into control slots on the pristine
device; after each item of the walk its slot is compared with its control
slot, so a lifecycle call or a concurrent FIFO that changes what the GPU
computed shows as a cmp that is not equal. Each distinct kind of work's
control result is dumped once. The setups a program runs in follow from what
it uses: info replies abort an in-process host, so a program with info work
runs in isolate only; the IOSurface host device needs a memory map (map and
isolate).

What the walks found, and the model now keeps: deleting child FIFO n zeroes
node n+1's page list (W3), and every unpause starts every child FIFO again,
mapping its page list again, just as a resume does. So a live child n+1
stops at the next pause and unpause (or rvg_device_will_suspend and unpause) as
well as at a resume: its node still says running, a NOP rung on it never runs,
and a barrier on its stamp blocks the FIFO that waits. The programs lock it
where it happens ("a NOP is not run"), and the guest recovers the child by
writing its page list again, deleting it and defining it.
"""
from pgtest.randgen import build
from pgtest.registry import scenario

# (seed, steps): short programs for quick diagnosis, medium ones, and a few
# long ones (compact: compared at checkpoints, pgtest/randgen.py).
SEEDS = ([(s, 8 + (s * 7) % 17) for s in range(1, 25)]
         + [(s, 30 + (s * 11) % 31) for s in range(25, 39)]
         + [(39, 120), (40, 150), (41, 180), (42, 200), (43, 200), (44, 200)]
         + [(s, 15 + (s * 13) % 40) for s in range(45, 53)])


@scenario
def interplay_rand():
    return [build(seed, steps, __doc__) for seed, steps in SEEDS]
