"""Seeded random guest programs that mix every walked dimension
(pgtest/randmix.py): each seed is one scenario, interplay_mix_<seed>,
whose program the generator builds from the seed alone.

One walk mixes what interplay_dag, interplay_lifetime, interplay_scanout,
interplay_faultwalk and interplay_rand walk apart: barrier DAGs of GPU work
over the root FIFO, up to eight child FIFOs and one or two tasks, each node's
bytes an order witness (dagwalk's), with a scratch-buffer part; GPU renders
into IOSurface mapper surfaces (type 11) and presents of them (opcode 8) and
of task 0's backings (opcode 7) on one or two displays, ordered by barriers;
work held on a gate FIFO while the guest deletes (37, 40), retypes (the
object-list entry rewritten in place) or remakes the objects it names, or
pauses, or suspends onto a fresh device; faulting commands (decode errors,
an invalid task, unknown opcodes) with work blocked on their stamps, and the
guest's recovery by a resume, by deleting the faulted child, by a pause or
by a suspend; work rung while paused; and idle rounds of pause, suspend,
rvg_device_will_suspend and unpause (no display), reset, stop or a fresh device
with the guest's set-up again, cursors, child FIFOs deleted and defined again,
and lifetime actions.

One model predicts every observation: the differential control for the GPU
work (every dag node, run uninterrupted on the pristine device, compared
slot by slot after each phase, and every slot and chain counter at the
end); the model's bytes for every surface and scratch page, written into a
page of the guest's and compared; one new_frame per present and the replays
after each resume; the fault registers waited for or labelled with the
model's state; and every wait succeeds but those labelled [held], which
time out (pgtest.randmix.check_expected reads the expected files back).
Programs with mapper surfaces run in the map and isolate setups (the
IOSurface host device needs a memory map); the others in all three.
"""
from pgtest.randmix import build
from pgtest.registry import scenario

# (seed, rounds): short programs for diagnosis, medium ones, long ones.
SEEDS = ([(s, 4 + (s * 5) % 6) for s in range(1, 25)]
         + [(s, 12 + (s * 7) % 9) for s in range(25, 39)]
         + [(s, 30 + (s * 3) % 11) for s in range(39, 45)]
         + [(s, 8 + (s * 5) % 11) for s in range(45, 51)])


@scenario
def interplay_mix():
    return [build(seed, rounds, __doc__) for seed, rounds in SEEDS]
