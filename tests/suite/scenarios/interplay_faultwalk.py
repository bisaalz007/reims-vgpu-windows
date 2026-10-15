"""Seeded random guest programs with faults (pgtest/randfault.py): each
seed is one scenario, interplay_faultwalk_<seed>, whose program the
generator builds from the seed alone.

The walk is interplay_rand's (tasks and objects, blit, compute and render
items on the root FIFO and child FIFOs, work held behind the gate,
displays, pause, suspend, reset, stop, a fresh device), with guest errors
injected at random points: faulting execs (an unknown record, a record
longer than its segment, a reference past the object list or 0, an
unknown encoder type), GetComputeInfo with its reply past the task's end,
an exec naming a task that does not exist, unknown and deprecated opcodes,
a circular barrier wait between two FIFOs below 32, a NOP rung on a
deleted child FIFO, a resume of a FIFO that is not faulted, and in the
isolate setup a task's service death (an exec range past the task's end),
after which the guest deletes the task and defines it again. Faults land
on the root FIFO and child FIFOs, alone or while other FIFOs run work at
once, hold work behind the gate, or wait on the faulting command's stamp.
The guest resumes each faulted FIFO (0x1024, 0x1028) at a random later
point, sometimes after a pause and unpause or a suspend and resume, or
deletes the faulted child.

What the walk checks: every item of work that runs (beside a fault, held
across it, or released by its resume) is compared with its control slot, run
uninterrupted before the walk; a faulted exec's slot is compared with its
control slot, which nothing wrote; and at every fault, resume and checkpoint
0x102c is waited for at the model's value, and 0x103c and every live child's
node fault offset are recorded, labelled with the state the model gives that
FIFO (ok, faulted, paused: published by a pause, still flagged; restored:
flagged by a resume from suspend).
"""
from pgtest.randfault import build
from pgtest.registry import scenario

# (seed, steps): short programs, medium ones, and long ("compact") ones
# compared at checkpoints.
SEEDS = ([(s, 8 + (s * 7) % 17) for s in range(1, 21)]
         + [(s, 30 + (s * 11) % 31) for s in range(21, 35)]
         + [(35, 100), (36, 120), (37, 140), (38, 160), (39, 110), (40, 130)]
         + [(s, 15 + (s * 13) % 40) for s in range(41, 47)])


@scenario
def interplay_faultwalk():
    return [build(seed, steps, __doc__) for seed, steps in SEEDS]
