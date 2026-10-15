"""The `teardown` generator (scenarios/native_teardown.py): the host
releases the device, or recreates it, in the middle of guest work, as far
as the runner can express it.

The guest: lifecycle_util's Rig (the root FIFO, a gate child FIFO, task 1
with regions A and B), region A run to the end, and 0-4 more tasks with
IDs drawn from 2-63 (their teardowns go in ascending task ID: the task
table, reims-vgpu-core's collections::bucket_order::BucketOrderMap),
each with shared and private buffers made by an exec, so the host holds
their mappings (the blocks
setup's map_memory calls). Then, drawn by the seed, the device is left with
work pending:
- region B's second part queued on the root FIFO behind the gate (a
  pending stamp);
- a NOP, or another task's exec, queued on a worker child FIFO behind the
  gate;
- then a task deleted, or unmapped (34), while that worker item is held
  (it is the task's own exec or another item);
- rvg_device_will_suspend returned (the FIFOs quiesced) but no
  rvg_device_finish_suspend;
- an EFI framebuffer on the 100 ms timer, bound to display 0, whose mode
  has height 0: the frame's size is 0, so its texture is nil and every
  tick presents nothing, while the timer, the EFI task and its mapping
  stay live until the teardown.
Then the host call: release (the host's last reference goes: the device's
shutdown tears it down, the displays first), or recreate (release, then a
fresh device over the same guest RAM, which the guest brings up again and
runs a NOP on). What the guest and the host see is observed: the host's
block calls (destroy_task and unmap_memory in the task table's order, in the
blocks setup), interrupts, callbacks, the root page.

Why the EFI frame is empty: a frame the timer presents calls new_frame
back, asynchronously, every 100 ms until the release drops the display,
so whether one lands between the last checkpoint and the release is
timing. Measured under load (a load average near 50 on 16 cores): 1 of 24
EFI program-setups ending in a release differed between two runs of the
same build (teardown:7049, the release's window held `cb:newFrame` in two
runs of three), and native_teardown_2_5[isolate] failed 1 time in 90. No
checkpoint option drops a callback kind, so the program shows nothing: a
live timer whose ticks present nothing, which the teardown still tears
down (the EFI mapping's unmap_memory and its destroy_task lead the release's
blocks in the blocks setup, whether the last tick or the shutdown stops
the presentation: both run on the present queue, before the shutdown goes
on).

What the runner cannot express is left to the rig, reims-vgpu-core's unit
tests on its fake platform and host (device::root::rig): a release on a
thread other than the runner's main thread; a release from
inside a host block or a display callback; a release that races a
command the GPU is running (every pending command here is held by a
barrier, so what the teardown finds is decided before it starts); and a
report queued at the release. The runner's `release` also drops its
displays before the device (release_all), so a device released before its
displays is not expressible either.
"""
import random

from . import abi
from .abi import PAGE
from .display_guest import display
from .efi_guest import EFI
from .lifecycle_util import Rig
from .scenario import Scenario
from .task import seg
from .tasks_lib import XGPUTask, exec3_payload, fill_rec


# The EFI frame: 16 wide, height 0, stride 64 (valid: 4 x width, 16-aligned).
EFI_W, EFI_H, EFI_STRIDE = 16, 0, 64


def build(seed, n, doc):
    """The program for seed: n is the number of extra tasks' buffers
    (1-12). Returns (Scenario, params)."""
    r = random.Random("teardown:%d:%d" % (seed, n))
    efi = r.random() < 0.25
    kw = {"displays": [display(0)]} if efi else {}
    s = Scenario("native_teardown_%d" % seed, watchdog_s=20, doc=doc, **kw)
    worker = 2 if r.random() < 0.6 else None
    g = Rig(s, worker=worker)
    g.control()
    g.run("B", 0)

    # More tasks, each with buffers the host maps when an exec makes them.
    ntasks = min(4, max(0, n // 3)) if n > 2 else r.randint(0, 1)
    tids = r.sample(range(2, 64), ntasks)
    extra = []
    left = n
    for i, tid in enumerate(tids):
        m = max(1, left // (ntasks - i)) if i < ntasks - 1 else max(1, left)
        left -= m
        t = XGPUTask(s, tid=tid, pages=8 + 2 * m, scatter=r.random() < 0.3)
        bufs = []
        for _ in range(m):
            ref, va = t.buffer(length=PAGE if r.random() < 0.7 else 2 * PAGE, flags=1 if r.random() < 0.3 else 0)
            bufs.append((ref, va))
        t.define(label="define task %d" % tid)
        t.install(label="object list of task %d" % tid)
        t.reset_cmdbufs()
        cb = t.cmdbuf(seg(abi.ENC_BLIT, b"".join(fill_rec(ref, 0, 0x40, 0x20 + tid) for ref, _ in bufs)))
        s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(tid, [cb], [(ref, 0) for ref, _ in bufs]),
                 label="task %d: its buffers made" % tid)
        extra.append((t, bufs))

    if efi:
        # The timer starts at the last register write, and never presents.
        EFI(s).setup(EFI_W, EFI_H, EFI_STRIDE)

    pending = []
    root_held = r.random() < 0.7
    if worker and r.random() < 0.7:
        w = g.worker
        if extra and r.random() < 0.5:
            t, bufs = r.choice(extra)
            t.reset_cmdbufs()
            ref = r.choice(bufs)[0]
            cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(ref, 0x40, 0x40, 0x99)))
            st = w.cmd(abi.OP_EXEC_INDIRECT3, exec3_payload(t.tid, [cb], [(ref, 0)]), barriers=g.gate_barrier())
            pending.append("task %d held on the worker" % t.tid)
        else:
            st = w.cmd(abi.OP_NOP, barriers=g.gate_barrier())
            pending.append("NOP held on the worker")
        w.doorbell()
        w.wait_stamp(st, timeout_ms=300, label="the worker held by the gate")
        if extra and r.random() < 0.4:
            t, _ = r.choice(extra)
            if r.random() < 0.5:
                t.delete(label="delete task %d, its exec held" % t.tid)
                extra.remove((t, _))
                pending.append("task %d deleted" % t.tid)
            else:
                t.unmap(label="unmap task %d, its exec held" % t.tid)
                pending.append("task %d unmapped" % t.tid)
    # Last on the root FIFO: anything after it there would wait behind it.
    if root_held:
        g.queue("B", 1, barriers=g.gate_barrier())
        pending.append("B held")
    if r.random() < 0.2:
        s.api("willSuspend")
        pending.append("willSuspend")
    if efi:
        pending.append("EFI timer")
    s.events("held", order=True)

    call = r.choices(("release", "recreate"), (0.7, 0.3))[0]
    s.api(call)
    s.events(call, order=True)
    s.diff(call, gpa=s.pa(s.root_page), length=PAGE)
    if call == "recreate":
        s.sweep(0x1000, 0x1040, label="registers of the new device")
        s.w(abi.R_VERSION, abi.MAX_VERSION)
        s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
        g.f.written = 0
        g.f.enable()
        s.submit(abi.OP_NOP, label="a NOP on the new device")
        s.events("the new device", order=True)
    params = "tasks=%s pending=[%s] call=%s" % (",".join(str(t) for t in [1] + tids), "; ".join(pending), call)
    return s, params
