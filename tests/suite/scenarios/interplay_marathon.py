"""Marathons (pgtest/marathon_util.py): each seed is one scenario,
interplay_marathon_<seed>, whose program the walker builds from the seed
alone: one device session in which every subsystem is driven at once,
step by step, so each subsystem's state changes while the others' work is
in flight.

A step is one action of one subsystem, chosen at random among the legal
ones: a display action (pgtest/randdisplay.py's walk: connects, acks,
presents of backings and mapper surfaces with gamma tables and colour
matrices, glyphs and cursors, sleep, ICC profiles, hot plugs and mode
lists, scribbles, backings and surfaces rewritten, deleted, unmapped;
with EFI at boot); a batch of GPU work (pgtest/dagwalk.py's order
witnesses over the root FIFO, child FIFOs and one to three tasks, ordered
by barriers, some nodes waiting on the gate, on earlier held work or on a
faulted stamp); the gate opening; a fault on an idle child
(pgtest/randfault.py's decode errors, an exec of a task that does not
exist, unknown and deprecated opcodes) with work blocked on it, and its
recovery by a resume or a delete, steps later; a lifetime action
(pgtest/lifetime_util.py: the scratch buffer deleted, retyped in place or
made again, pipelines and functions deleted, a task deleted and defined
again) while work that names it is held; a memory change
(pgtest/memlive_util.py: a scratch page moved by 34, a new leaf and 57;
held work's command-buffer pages moved leaf by leaf or by a whole new
leaf table, the old pages poisoned); child FIFOs deleted and defined
again; pause (while paused, display host changes, commands rung, and the
guest rewriting command pages and object-list entries), suspend onto a
fresh device over held work, and reset or stop with the set-up again.

One model predicts every observation: the display walk's (callbacks,
interrupts, 0x1014 and 0x1018, every shared page, the encoded pixels,
present counts), kept exact across the GPU steps by counting their
interrupts and stamp bits; every GPU slot against a control task's run of
the same node, and the chain counters; every scratch page against the
model of which RAM page the scratch object is bound to; the fault
registers; and every wait succeeds but those labelled [held] (work shown
not to run within 200 ms) and [dies]. python3 -m pgtest.marathon_util
reads the expected files back against the model. Programs with mapper
surfaces run in the map and isolate setups, the others in all three.
Some programs end in a death docs/testing.md lists (7.3, 7.4) with work
still held on the gate: a one-entry gamma table, the display doorbell or
rvg_display_cursor_position while paused, a second pause, a guest mode count
over 128.
"""
from pgtest.marathon_util import build
from pgtest.registry import scenario

# (seed, steps, death): medium programs, long ones (hundreds of steps), and
# programs that end in each death the display walk knows, one of them long.
SEEDS = ([(s, 30 + (s * 7) % 41, None) for s in range(1, 15)]
         + [(15, 200, None), (16, 250, None), (17, 300, None), (18, 400, None), (43, 300, None)]
         + [(19, 40, "gamma_one"), (20, 50, "pause_twice"), (21, 40, "doorbell_paused"),
            (22, 45, "cursor_paused"), (23, 60, "mode_count_over"), (24, 200, "pause_twice")])


@scenario
def interplay_marathon():
    return [build(seed, steps, __doc__, death) for seed, steps, death in SEEDS]
