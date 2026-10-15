"""The host releases or recreates the device with guest work pending
(pgtest/teardown_util.py): the teardown generator's programs at fixed seeds.

The guest brings the device up with a gate child FIFO and task 1 (region A
run whole, region B's first part run), and more tasks with IDs drawn from
2-63, each with buffers an exec made (so the host holds their mappings).
Then, by seed: region B's second part queued on the root FIFO behind the
gate (a pending stamp); a NOP or another task's exec held on a worker
child FIFO behind the gate; a task deleted while its exec is held;
rvg_device_will_suspend returned without rvg_device_finish_suspend; an EFI
framebuffer on its 100 ms timer bound to display 0. Then the host releases the
device (its last release shuts it down: in the blocks setup the host sees
unmap_memory and destroy_task for every task, in ascending task ID) or
recreates it (release, then a fresh device over the same RAM that the guest
brings up and runs a NOP on). The events of the teardown and the root page are
observed. What the runner cannot express (a release on another thread, from
inside a host block or callback, racing a running command, or with a report
queued) is left to the rig (teardown_util's docstring).
"""
from pgtest.registry import scenario
from pgtest.teardown_util import build

# (seed, buffers of the extra tasks): each names its pending work and call
CASES = [
    (1, 1),   # a NOP on the worker and B held; release
    (8, 1),   # task 42's exec held, the task deleted, B held; recreate
    (6, 1),   # B held, the EFI timer; release
    (5, 2),   # a NOP on the worker, rvg_device_will_suspend, the EFI timer; release
    (3, 6),   # B held, rvg_device_will_suspend; release
    (9, 6),   # task 12's exec held, B held, the EFI timer; release
    (2, 9),   # B held, four tasks; recreate
    (6, 9),   # task 43's exec held, the task deleted; release
]


@scenario
def native_teardown():
    out = []
    for seed, n in CASES:
        s, params = build(seed, n, __doc__)
        s.name = "native_teardown_%d_%d" % (n, seed)
        out.append(s)
    return out
