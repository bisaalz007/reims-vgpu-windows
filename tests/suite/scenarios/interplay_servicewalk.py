"""Service walks (pgtest/servicewalk_util.py): each seed is one scenario,
interplay_servicewalk_<seed>, whose program the walker builds from the seed
alone: many GPU tasks at once, each with its own buffers, pipelines, heap,
ICB and render target, their execs interleaved across the root FIFO and
child FIFOs, while tasks are defined and deleted, their objects deleted and
made again, info replies requested, and faults injected into one task at
a time; every other task keeps running, and every slot it writes is
compared with the model's bytes.

Under isolation each task is a service of its own, on threads of the host's
process (docs/testing.md 3.3, 6.10); in the blocks and map setups the same
programs run in the host, with the same results, and the divergences of
section 6.10's table are locked where they are met: MapMemory2 at an
unaligned offset, UnmapMemory past a task's end, GetComputeInfo with count 0,
a GPU fill of a read-only buffer, and a private heap's guest pages.

Programs end last in the deaths of section 7.4. The all-setup walks that
end in a death end in one on task 1's exec, beside the others' work: in
the blocks and map setups a host death (a range past the task's end, a
last segment continued by next, a zero-length range, an info reply).
Under isolation a death in the task's own code (a range past its end, a
zero-length range) is a fault after which every other task keeps running,
and task 1 is deleted and defined again with a new service; the last
segment continued by next asserts inside Metal, which ends the host there
too (interplay_servicewalk_3 and 13); the service writes the info reply;
GetComputeInfo on pipeline 0 writes zeros in-process and kills the task's
service. The isolate walks end in one to three deaths at once on
different FIFOs, and lock that only the dead tasks' commands fault
afterwards, unless one of them is an encoder left open, inside Metal,
which ends the host in its wait (interplay_servicewalk_22, 24, 25, 26, 28
and 33).

python3 -m pgtest.servicewalk_util checks the expected files: every cmp
equal, every wait ok but the either() waits and the [dies] ones.
"""
from pgtest.servicewalk_util import build
from pgtest.registry import scenario

ISO = ("isolate",)

# (seed, steps, setups, tasks, children, ending, big)
SEEDS = [
    # all setups: short, each ending one way
    (1, 8, None, 4, 2, None, False),
    (2, 10, None, 5, 3, "past_end", False),
    (3, 12, None, 6, 4, "open_encoder", False),
    (4, 12, None, 6, 3, "zero_length", False),
    (5, 14, None, 6, 4, "info_reply", False),
    (6, 14, None, 8, 5, "compute_info_0", False),
    # all setups: medium
    (7, 30, None, 8, 6, None, False),
    (8, 40, None, 10, 4, "past_end", False),
    (9, 60, None, 12, 6, None, False),
    (10, 80, None, 10, 5, "compute_info_0", False),
    (11, 50, None, 16, 6, "info_reply", False),
    # all setups: long
    (12, 200, None, 12, 6, None, False),
    (13, 200, None, 20, 6, "open_encoder", False),
    # isolate: the service's own walks (info replies among the programs),
    # ending in one to three deaths at once
    (21, 12, ISO, 6, 3, ("past_end",), False),
    (22, 16, ISO, 8, 4, ("open_encoder", "zero_length"), False),
    (23, 20, ISO, 10, 5, ("compute_info_0", "past_end", "zero_length"), False),
    (24, 30, ISO, 12, 6, ("zero_length", "open_encoder"), False),
    (25, 60, ISO, 16, 6, ("past_end", "compute_info_0", "open_encoder"), False),
    (26, 40, ISO, 24, 6, ("open_encoder",), False),
    (27, 150, ISO, 16, 6, ("compute_info_0", "zero_length"), False),
    (28, 200, ISO, 24, 6, ("past_end", "open_encoder", "zero_length"), False),
    # up to the task limit: tasks 1-255 at once
    (31, 16, None, 255, 6, None, True),
    (32, 12, ISO, 255, 6, ("past_end", "zero_length"), True),
    (33, 60, ISO, 64, 6, ("open_encoder",), False),
]


@scenario
def interplay_servicewalk():
    out = []
    for seed, steps, setups, tasks, children, ending, big in SEEDS:
        kw = {} if setups is None else {"setups": setups}
        sc = build("interplay_servicewalk_%d" % seed, __doc__, seed, steps, ntasks=tasks, children=children,
                   ending=ending, big=big, **kw)
        sc.services = tasks  # under isolation a service a task, at most all at once (cli.run_all)
        out.append(sc)
    return out
