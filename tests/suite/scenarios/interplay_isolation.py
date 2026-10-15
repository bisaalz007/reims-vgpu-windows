"""Process isolation across tasks: one service a task, on threads of its own
(setup isolate, process_isolation; docs/testing.md, section 3.3), and what a
task's death does to its task, its FIFO, and every other task and FIFO.

Every scenario here is isolate only: in-process there is no service, and
the deaths used below kill the host itself (exec_null_cursor_past_end,
exec_open_encoder_at_commit), which ends a scenario.

A death in a task's own code ends that task and nothing else
(tasks::isolation): the host's end of the task's link hears of it and
completes every outstanding token with 4, and every later command on the
task completes with 4 at once. Every FIFO handler faults on a status other
than 1. A crash inside Metal is not the task's: it ends the host process,
isolated task or not (docs/testing.md, section 4.8). Deaths used:
  - an exec range past the task's end: a null cursor the service
    dereferences (exec_null_cursor), the task's own death;
  - a last segment that says continuedByNext: Metal asserts in
    encodeSignalEvent:value: (exec_open_encoder_at_commit), which ends the
    host, so a scenario that uses it ends there;
  - a zero-length range: the service refuses it ("Zero length cursor
    request") and dereferences the null cursor
    (exec_zero_length_range_aligned_isolate), the task's own death;
  - GetComputeInfo (59) on pipeline 0: the service asserts the pipeline
    (compute_info_ref_zero), the task's own death.
None faults the GPU.

Healthy tasks carry the same work with and without a death beside them,
into two regions each, and the regions are compared (cmp).
"""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.compute_objects import Blit as B, Compute as C
from pgtest.diff_util import (BLIT, COMPUTE, K_ADD, K_DOUBLE, K_INC, Lab, Workload, children, exec_payload,
                              fault_state, merge, pattern, refs_payload, resume)
from pgtest.fifo_util import NODE_FAULT
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import rec, seg
from pgtest.tasks_lib import heap_payload, heap_ser

ISO = ("isolate",)
REGION = 0x800
DEATH_PAST_END = "past_end"
DEATH_OPEN_ENCODER = "open_encoder"
DEATH_COMPUTE_INFO = "compute_info"
DEATH_ZERO_LENGTH = "zero_length"


class Healthy(Lab):
    """A task whose work (four execs) writes REGION bytes into one of its
    result buffers: double_gid into +0; add_one of that into +0x100; a copy
    of src and a fill at +0x200; the Workload at +0x400."""

    def __init__(self, s, tid, regions=4, fifo=None):
        super().__init__(s, tid=tid)
        self.inc_, self.dbl, self.add1 = self.kernel(K_INC), self.kernel(K_DOUBLE), self.kernel(K_ADD)
        self.src, _ = self.shared(contents=pattern(0x100, 3 + tid, tid))
        self.outs = [self.shared() for _ in range(regions)]
        self.ready(fifo)

    def execs(self, buf):
        return [[seg(COMPUTE, C.set_pipeline(self.dbl) + C.set_buffers(0, [(buf, 0)])
                     + C.dispatch_threads((64, 1, 1), (64, 1, 1)))],
                [seg(COMPUTE, C.set_pipeline(self.add1) + C.set_buffers(0, [(buf, 0x100), (buf, 0)])
                     + C.dispatch_threads((64, 1, 1), (64, 1, 1)))],
                [seg(BLIT, B.copy_buffer_to_buffer(self.src, buf, 0, 0x200, 0x100) + B.fill(buf, 0x300, 0x10, 0x5A))],
                merge(Workload(self.inc_).steps(buf, 0x400))]

    def queue(self, region, fifo=None):
        """Writes the four execs into a FIFO's ring without ringing; returns
        the last stamp."""
        buf = self.outs[region][0]
        self.reset_cmdbufs()
        last = None
        for segments in self.execs(buf):
            last = self.cmd(segments, fifo=fifo)
        return last

    def va(self, region):
        return self.outs[region][1]


class Doomed(Lab):
    """A task with one buffer, whose service a command kills."""

    def __init__(self, s, tid, fifo=None):
        super().__init__(s, tid=tid)
        self.buf, self.buf_va = self.shared()
        self.ready(fifo)

    def death(self, kind, fifo=None):
        """Writes the killing exec into a FIFO's ring; returns its stamp."""
        if kind == DEATH_PAST_END:
            return self.cmd(None, ranges=[(self.npages * PAGE, 0x40)], fifo=fifo)
        if kind == DEATH_ZERO_LENGTH:
            return self.cmd(None, ranges=[(4 * PAGE, 0)], fifo=fifo)
        if kind == DEATH_COMPUTE_INFO:
            return (fifo or self.s.fifo).cmd(abi.OP_GET_COMPUTE_INFO,
                                             struct.pack("<IIIIQ", self.tid, 0, 5, 4, 60 * PAGE))
        return self.cmd([seg(BLIT, B.fill(self.buf, 0, 0x10, 0x11), cont_next=1)], fifo=fifo)

    def fill(self, byte):
        return [seg(BLIT, B.fill(self.buf, 0, 0x40, byte))]


def ring(s, *fifos):
    for f in fifos:
        f.doorbell()


def wait_all(fifos_stamps, label):
    for f, st in fifos_stamps:
        f.wait_stamp(st, label="%s: FIFO %d stamp %d" % (label, f.fid, st))


def nodes(s, fifos, label):
    """The child nodes' fault offsets (UINT32_MAX when healthy)."""
    for f in fifos:
        s.dump(f.node_gpa(NODE_FAULT), 4, label="%s: child %d fault offset" % (label, f.fid))


@scenario(setups=ISO)
def interplay_iso_death_beside_healthy():
    """Task 1's service dies on the root FIFO (an exec range past the task's
    end) while tasks 2 and 3 run four execs each on child FIFOs 1 and 2,
    rung before it. The healthy work completes, and its bytes equal those of
    the same work run before, with no death beside it; the child FIFOs never
    fault. The death is a fault of the root FIFO at the exec; after the
    resume, task 1's next exec faults too (its token source is crashed). 56
    on task 1 is refused ("Host task already exists"); 32 deletes it; a task
    1 defined again gets a new service and runs the healthy work to the same
    bytes; tasks 2 and 3 do too, a third time."""
    s = Scenario("interplay_iso_death_beside_healthy", setups=ISO, doc=interplay_iso_death_beside_healthy.__doc__)
    s.bring_up()
    c1, c2 = children(s, 1, 2)
    d = Doomed(s, 1)
    h2, h3 = Healthy(s, 2), Healthy(s, 3)
    a, b = h2.queue(0, c1), h3.queue(0, c2)
    ring(s, c1, c2)
    wait_all([(c1, a), (c2, b)], "no death")
    a, b = h2.queue(1, c1), h3.queue(1, c2)
    st = d.death(DEATH_PAST_END)
    ring(s, c1, c2, s.fifo)
    s.wait_fault(label="task 1's service dies")
    wait_all([(c1, a), (c2, b)], "beside the death")
    fault_state(s, s.fifo, "the death")
    nodes(s, (c1, c2), "the death")
    resume(s, s.fifo)
    s.fifo.wait_stamp(st, label="the death's stamp, after the resume")
    h2.dump(h2.va(0), REGION, "task 2's work")
    s.cmp(h2.gpa(h2.va(0)), h2.gpa(h2.va(1)), REGION, label="task 2: beside the death == alone")
    s.cmp(h3.gpa(h3.va(0)), h3.gpa(h3.va(1)), REGION, label="task 3: beside the death == alone")
    d.run(d.fill(0x22), fault=True, label="task 1 after its death")
    d.dump(d.buf_va, 0x40, "task 1's buffer")
    d.define(fault=True, label="56 on the dead task 1")
    fault_state(s, s.fifo, "56 on the dead task 1")
    resume(s, s.fifo)
    d.delete(label="32 on the dead task 1")
    n = Healthy(s, 1)
    n.run(n.execs(n.outs[0][0])[0], label="task 1 again: exec 1")
    for i, segments in enumerate(n.execs(n.outs[0][0])[1:], 2):
        n.run(segments, label="task 1 again: exec %d" % i)
    s.cmp(n.gpa(n.va(0)) + 0x400, h2.gpa(h2.va(0)) + 0x400, REGION - 0x400,
          label="task 1 again: Workload == task 2's")
    n.dump(n.va(0), 0x200, "task 1 again: double_gid and add_one")
    a, b = h2.queue(2, c1), h3.queue(2, c2)
    ring(s, c1, c2)
    wait_all([(c1, a), (c2, b)], "after the death")
    s.cmp(h2.gpa(h2.va(0)), h2.gpa(h2.va(2)), REGION, label="task 2: after == alone")
    s.cmp(h3.gpa(h3.va(0)), h3.gpa(h3.va(2)), REGION, label="task 3: after == alone")
    for t in (n, h2, h3):
        t.delete()
    s.events("end")
    return s


@scenario(setups=ISO)
def interplay_iso_death_on_child():
    """Task 2's service dies on child FIFO 1 (an exec range past the task's
    end) while task 1 works on the root FIFO and task 3 on child FIFO 2.
    Only child FIFO 1 faults (its node's fault offset, and 0x102c); the
    others complete with the bytes they give alone. After 0x1028 resumes
    child 1, task 2's next exec there faults again; task 3's work on child
    1 completes. Task 2 is deleted and defined again through child FIFO 1,
    and works."""
    s = Scenario("interplay_iso_death_on_child", setups=ISO, doc=interplay_iso_death_on_child.__doc__)
    s.bring_up()
    c1, c2 = children(s, 1, 2)
    h1 = Healthy(s, 1)
    d = Doomed(s, 2)
    h3 = Healthy(s, 3)
    a, b = h1.queue(0), h3.queue(0, c2)
    ring(s, s.fifo, c2)
    wait_all([(s.fifo, a), (c2, b)], "no death")
    a, b = h1.queue(1), h3.queue(1, c2)
    st = d.death(DEATH_PAST_END, fifo=c1)
    ring(s, s.fifo, c2, c1)
    s.wait(gpa=c1.node_gpa(NODE_FAULT), ne=0xFFFFFFFF, label="child 1 faults")
    s.wait_fault(label="0x102c")
    wait_all([(s.fifo, a), (c2, b)], "beside the death")
    fault_state(s, c1, "the death")
    s.r(abi.R_FAULT_OFFSET, label="the root FIFO's fault offset")
    nodes(s, (c2,), "the death")
    resume(s, c1)
    c1.wait_stamp(st, label="the death's stamp, after the resume")
    s.r(abi.R_FAULT_MASK, label="after the resume")
    s.cmp(h1.gpa(h1.va(0)), h1.gpa(h1.va(1)), REGION, label="task 1: beside the death == alone")
    s.cmp(h3.gpa(h3.va(0)), h3.gpa(h3.va(1)), REGION, label="task 3: beside the death == alone")
    d.run(d.fill(0x22), fifo=c1, fault=True, label="task 2 after its death, child 1")
    s.r(abi.R_FAULT_MASK, label="after the child's resume")
    b = h3.queue(2, c1)
    ring(s, c1)
    c1.wait_stamp(b, label="task 3's work on child 1")
    s.cmp(h3.gpa(h3.va(0)), h3.gpa(h3.va(2)), REGION, label="task 3 on child 1 == alone")
    d.delete(label="32 on the dead task 2")
    n = Healthy(s, 2, fifo=c1)
    last = n.queue(0, c1)
    ring(s, c1)
    c1.wait_stamp(last, label="task 2 again, on child 1")
    s.cmp(n.gpa(n.va(0)) + 0x400, h1.gpa(h1.va(0)) + 0x400, REGION - 0x400,
          label="task 2 again: Workload == task 1's")
    n.dump(n.va(0), 0x400, "task 2 again")
    for t in (h1, n, h3):
        t.delete()
    s.events("end")
    return s


@scenario(setups=ISO)
def interplay_iso_faults_stay_apart():
    """Four tasks, four kinds of outcome at once: task 1 faults without
    dying (a blit record the decoder does not know, 399) on the root FIFO,
    with task 4's work queued behind it there; task 3 dies on child FIFO 2
    (a range past its end, a death in its own code); and task 2's command
    on child FIFO 1 (continuedByNext on the last segment) makes Metal
    assert at the commit. That assert is inside Metal, so it ends the host
    process although task 2 is isolated (docs/testing.md, section 4.8), and
    the run ends there. The rest of the program is never reached: each
    FIFO's fault and resume, task 4's bytes against those it gives alone,
    task 1 working again after its decode fault, and tasks 2 and 3 faulting
    on their next exec.

    Metal's assert ends the process asynchronously to the runner's ops:
    task 2's command goes behind 256 NOPs and child FIFO 1's doorbell is
    rung last, so the process dies in the wait for child 1's fault every
    run."""
    s = Scenario("interplay_iso_faults_stay_apart", setups=ISO, doc=interplay_iso_faults_stay_apart.__doc__)
    s.bring_up()
    c1, c2 = children(s, 1, 2)
    t1 = Doomed(s, 1)
    t2 = Doomed(s, 2)
    t3 = Doomed(s, 3)
    h4 = Healthy(s, 4)
    last = h4.queue(0)
    s.doorbell()
    s.fifo.wait_stamp(last, label="task 4 alone")
    st1 = t1.cmd([seg(BLIT, rec(399, b"\0" * 8))])
    last = h4.queue(1)
    for _ in range(256):
        c1.cmd(abi.OP_NOP)
    st2 = t2.death(DEATH_OPEN_ENCODER, fifo=c1)
    st3 = t3.death(DEATH_PAST_END, fifo=c2)
    ring(s, s.fifo, c2, c1)
    s.wait(gpa=c1.node_gpa(NODE_FAULT), ne=0xFFFFFFFF, label="child 1 faults")
    s.wait(gpa=c2.node_gpa(NODE_FAULT), ne=0xFFFFFFFF, label="child 2 faults")
    s.wait(mmio=abi.R_FAULT_OFFSET, mmio_ne=0xFFFFFFFF, label="the root FIFO faults")
    fault_state(s, s.fifo, "task 1's decode fault")
    nodes(s, (c1, c2), "the deaths")
    resume(s, s.fifo)
    resume(s, c1)
    resume(s, c2)
    s.fifo.wait_stamp(last, label="task 4 behind task 1's fault")
    c1.wait_stamp(st2, label="task 2's death's stamp")
    c2.wait_stamp(st3, label="task 3's death's stamp")
    s.r(abi.R_FAULT_MASK, label="after the resumes")
    s.cmp(h4.gpa(h4.va(0)), h4.gpa(h4.va(1)), REGION, label="task 4: behind the faults == alone")
    t1.run(t1.fill(0x31), label="task 1 after its decode fault")
    t1.dump(t1.buf_va, 0x40, "task 1's buffer")
    t2.run(t2.fill(0x32), fifo=c1, fault=True, label="task 2 after its death")
    t3.run(t3.fill(0x33), fifo=c2, fault=True, label="task 3 after its death")
    s.r(abi.R_FAULT_MASK, label="after the dead tasks' faults")
    t2.dump(t2.buf_va, 0x40, "task 2's buffer")
    t3.dump(t3.buf_va, 0x40, "task 3's buffer")
    last = h4.queue(2)
    s.doorbell()
    s.fifo.wait_stamp(last, label="task 4 after")
    s.cmp(h4.gpa(h4.va(0)), h4.gpa(h4.va(2)), REGION, label="task 4: after == alone")
    for t in (t1, t2, t3, h4):
        t.delete()
    s.events("end")
    return s


def _probe(s, opcode, payload, expect, label):
    """One command on the dead task: its stamp, or its fault, the fault
    state and the resume, then the stamp the resume publishes."""
    st = s.submit(opcode, payload, fault=(expect == "fault"), label=label)
    if expect == "fault":
        fault_state(s, s.fifo, label)
        resume(s, s.fifo)
        s.fifo.wait_stamp(st, label=label + ": resumed")
    else:
        s.r(abi.R_FAULT_MASK, label=label)


@scenario(setups=ISO)
def interplay_iso_dead_task_commands():
    """Every command that names a task, on a task whose service died (an
    exec range past its end): each one that asks the service anything gets a
    token its crashed token source signals 4 at once, and faults: 51, 68,
    52, 53, 62, 63, 34, 37, 64, 55, 67, 69, and 60, 40, 59 (pipeline 0) and
    66 (map 0), which the live task is not asked (59 on pipeline 0 kills a
    live service, compute_info_ref_zero). MapMemory2 (57) asks nothing and
    completes. Defining a child FIFO and deleting it (48, 49) calls every
    task's Task::fifo_deleted, the dead one's included, which completes at once;
    both complete. Then 32 deletes the task, 56 defines it again with a new
    service, and the same commands complete or fault as a live task's do; a
    task beside it (task 2) never faulted."""
    s = Scenario("interplay_iso_dead_task_commands", setups=ISO, doc=interplay_iso_dead_task_commands.__doc__)
    s.bring_up()
    d = Doomed(s, 1)
    h = Healthy(s, 2)
    last = h.queue(0)
    s.doorbell()
    s.fifo.wait_stamp(last, label="task 2 alone")
    st = d.death(DEATH_PAST_END)
    s.doorbell()
    s.wait_fault(label="task 1's service dies")
    fault_state(s, s.fifo, "the death")
    resume(s, s.fifo)
    s.fifo.wait_stamp(st, label="the death's stamp")
    tid, ref, cb = 1, d.buf, d.place(d.fill(0x44))
    heap_va = 60 * PAGE

    def table(dead):
        f = "fault" if dead else "signal"
        who = "dead" if dead else "live"
        rows = [
            (abi.OP_SET_OBJECT_LIST, struct.pack("<III", tid, 0, d.ol.length()), f, "51"),
            (abi.OP_SET_OBJECT_AND_PLACEMENT_LIST, struct.pack("<IQQII", tid, 0, 56 * PAGE, d.ol.length(),
                                                               16 * len(d.ol.entries)), f, "68"),
            (abi.OP_INVALIDATE_RESOURCES, struct.pack("<IIIBBBB", tid, 1, ref, 0, 0, 1, 0), f, "52"),
            (abi.OP_SYNCHRONIZE_RESOURCES, refs_payload(tid, [ref]), f, "53"),
            (abi.OP_SYNCHRONIZE_AND_DISCARD_RESOURCES, refs_payload(tid, [ref]), f, "62"),
            (abi.OP_DISCARD_RESOURCES, refs_payload(tid, [ref]), f, "63"),
            (abi.OP_MAP_MEMORY2, struct.pack("<IQQ", tid, 0, PAGE), "signal", "57"),
            (abi.OP_UNMAP_MEMORY, struct.pack("<IQQ", tid, 62 * PAGE, PAGE), f, "34"),
            (abi.OP_DELETE_RESOURCE, struct.pack("<II", tid, ref), f, "37"),
            (abi.OP_HEAP_TEXTURE_SIZE_AND_ALIGN, heap_payload(tid, 61 * PAGE, 16, heap_ser()), f, "64"),
            (abi.OP_EXEC_INDIRECT2, exec_payload(tid, cb, [(ref, 0)], abi.OP_EXEC_INDIRECT2), f, "55"),
            (abi.OP_EXEC_INDIRECT3, exec_payload(tid, cb, [(ref, 0)]), f, "67"),
            (abi.OP_INFO_INDIRECT, exec_payload(tid, cb, [(ref, 0)], abi.OP_INFO_INDIRECT), f, "69"),
        ]
        for op, payload, expect, name in rows:
            _probe(s, op, payload, expect, "%s on the %s task" % (name, who))
        d.dump(d.buf_va, 0x40, "the %s task's buffer" % who)

    table(True)
    for op, payload, name in ((abi.OP_REPLACE_PHYSICAL, struct.pack("<II", tid, ref), "60"),
                              (abi.OP_DELETE_OBJECT, struct.pack("<IIII", tid, 1003, 12, ref), "40"),
                              (abi.OP_GET_COMPUTE_INFO, struct.pack("<IIIIQ", tid, 0, 1, 1, 63 * PAGE), "59"),
                              (abi.OP_RESET_RASTERIZATION_RATE_MAP, struct.pack("<II", tid, 0), "66")):
        _probe(s, op, payload, "fault", "%s on the dead task" % name)
    (c1,) = children(s, 1)
    c1.delete(label="49 with a dead task")
    d.delete(label="32 on the dead task")
    d.define(label="56: task 1 again")
    d.install(label="51: task 1 again")
    table(False)
    last = h.queue(1)
    s.doorbell()
    s.fifo.wait_stamp(last, label="task 2 after")
    s.cmp(h.gpa(h.va(0)), h.gpa(h.va(1)), REGION, label="task 2: after == alone")
    d.delete()
    h.delete()
    s.events("end")
    return s


@scenario(setups=ISO)
def interplay_iso_suspend_with_dead_task():
    """A suspend while task 1's service is dead and task 2's alive: the
    suspend flushes every task, the dead one's flush gets a token
    signalled 4 at once and returns; the blob is written. Resumed on a
    fresh device, task 1 is rebuilt from its root word with a new service,
    so it works again, and task 2 gives the bytes it gave before."""
    s = Scenario("interplay_iso_suspend_with_dead_task", setups=ISO,
                 doc=interplay_iso_suspend_with_dead_task.__doc__)
    s.bring_up()
    d = Doomed(s, 1)
    h = Healthy(s, 2)
    last = h.queue(0)
    s.doorbell()
    s.fifo.wait_stamp(last, label="task 2 alone")
    st = d.death(DEATH_PAST_END)
    s.doorbell()
    s.wait_fault(label="task 1's service dies")
    fault_state(s, s.fifo, "the death")
    resume(s, s.fifo)
    s.fifo.wait_stamp(st, label="the death's stamp")
    s.events("before the suspend")
    s.api("willSuspend")
    s.api("finishSuspend", slot="s")
    s.api("recreate")
    s.api("willResume", slot="s")
    s.api("didResume")
    s.events("resumed")
    d.run(d.fill(0x55), label="task 1 after the resume")
    d.dump(d.buf_va, 0x40, "task 1's buffer")
    last = h.queue(1)
    s.doorbell()
    s.fifo.wait_stamp(last, label="task 2 after the resume")
    s.cmp(h.gpa(h.va(0)), h.gpa(h.va(1)), REGION, label="task 2: after the resume == before")
    d.delete()
    h.delete()
    s.events("end")
    return s


@scenario(setups=ISO)
def interplay_iso_four_deaths():
    """Four tasks, each to be killed by a different trigger in turn on the
    root FIFO (a range past the end, continuedByNext on the last segment, a
    zero-length range, 59 on pipeline 0), while task 5 runs its work on
    child FIFO 1, rung before the first death. The first ends task 1 alone:
    one fault of the root FIFO, whose resume publishes the stamp. The
    second makes Metal assert at the commit, which ends the host process
    although task 2 is isolated (docs/testing.md, section 4.8): the run
    ends there. The rest of the program is never reached: the other two
    deaths, task 5's bytes against those it gives alone, the dead tasks'
    faulting execs, and their deletes and new services."""
    s = Scenario("interplay_iso_four_deaths", setups=ISO, doc=interplay_iso_four_deaths.__doc__)
    s.bring_up()
    (c1,) = children(s, 1)
    kinds = (DEATH_PAST_END, DEATH_OPEN_ENCODER, DEATH_ZERO_LENGTH, DEATH_COMPUTE_INFO)
    dead = [Doomed(s, tid) for tid in range(1, 5)]
    h = Healthy(s, 5)
    a = h.queue(0, c1)
    ring(s, c1)
    c1.wait_stamp(a, label="task 5 alone")
    a = h.queue(1, c1)
    ring(s, c1)
    for d, kind in zip(dead, kinds):
        st = d.death(kind)
        s.doorbell()
        s.wait_fault(label="task %d: %s" % (d.tid, kind))
        fault_state(s, s.fifo, "task %d: %s" % (d.tid, kind))
        resume(s, s.fifo)
        s.fifo.wait_stamp(st, label="task %d: its stamp, after the resume" % d.tid)
    c1.wait_stamp(a, label="task 5 beside the deaths")
    nodes(s, (c1,), "after the deaths")
    b = h.queue(2, c1)
    ring(s, c1)
    c1.wait_stamp(b, label="task 5 after the deaths")
    s.cmp(h.gpa(h.va(0)), h.gpa(h.va(1)), REGION, label="task 5: beside the deaths == alone")
    s.cmp(h.gpa(h.va(0)), h.gpa(h.va(2)), REGION, label="task 5: after == alone")
    for d in dead:
        d.run(d.fill(0x60 + d.tid), fault=True, label="task %d after its death" % d.tid)
    for d in dead:
        d.delete(label="32 on task %d" % d.tid)
    for d in dead:
        d.define(label="56: task %d again" % d.tid)
        d.install(label="51: task %d again" % d.tid)
        d.run(d.fill(0x70 + d.tid), label="task %d again" % d.tid)
        d.dump(d.buf_va, 0x40, "task %d's buffer" % d.tid)
    for d in dead + [h]:
        d.delete()
    s.events("end")
    return s
