"""Order under concurrency, as dependency graphs (pgtest/dagwalk.py): GPU
work spread over the root FIFO, up to 16 child FIFOs and several tasks,
whose edges are barriers on other FIFOs' stamps and on the FIFO's own
earlier ones. Each seed is one scenario, interplay_dag_<seed>; the shapes
after them are written by hand.

Every node is one exec that leaves an order witness: it increments its
chain's counter (inc_k) and copies the counter block out, runs add_one over
its first predecessor's result, and copies each predecessor's first words
into its slot. A chain's nodes are ordered by the graph, and a node reads
only its ancestors, so every byte is fixed by the graph whatever the
interleaving; a control run of every node, one after another in creation
order on the root FIFO, is what every phase's slots and the chain counters
are compared with. Also locked: the stamps every FIFO publishes after each
phase and while work is held, the interrupt count of each phase (each exec
publishes alone), the interrupts' order and snapshots where a phase is
totally ordered, and the host's block calls, sorted and digested where FIFOs
run at once.

What varies: width and depth, joins over several FIFOs (successive
barriers), predecessors from earlier phases (their stamps already passed),
work held behind the gate for 300 ms or for 2.5 s (past the blocked-barrier
report) across nothing, a pause and unpause, or a suspend and a resume on a
fresh device, child FIFOs deleted and defined again between phases
(barriers on the old incarnation's stamps pass at once), up to three tasks
over the same guest pages, opcodes 67 and 55.

What the implementation does not order, and so nothing here assumes: execs
on one FIFO are ordered on the GPU only when they share a task. Two
consecutive execs of different tasks on one FIFO, with no barrier between
them, run at once (measured: the second, a copy of a counter the first
increments 1100 times, read 0 in-process and in the service after the
first round); in-process both tasks' buffers are different MTLBuffers over
the same guest pages on the one exec queue, so Metal tracks no hazard
between them, and in the service the second task's command buffer waits
only for the FIFO event the first signals when it is scheduled. Their
stamps still publish in order. So every edge here is a barrier, the one on
the FIFO's previous node included.
"""
import struct

from pgtest import abi
from pgtest.compute_objects import Blit as B, Compute as C
from pgtest.dagwalk import GATE, PRESS, SHARED, DagTask, Emitter, Graph, build, node_segments
from pgtest.diff_util import exec_payload
from pgtest.fifo import ChildFIFO
from pgtest.fifo_util import child_fault, resume_child, track_stamps
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import seg

# (seed, nodes a phase, about): varied sizes; generate() picks the rest.
SEEDS = [(s, 6 + (s * 7) % 23) for s in range(1, 46)]


@scenario
def interplay_dag():
    return [build(seed, n, __doc__) for seed, n in SEEDS]


def emit(name, doc, g, **kw):
    return Emitter(name, doc, g.finish(), **kw).emit()


@scenario
def interplay_dag_diamond():
    """Diamonds. Phase 0: the root's node A feeds children 1-4 (each B_i
    waits on A), child 5's J joins them through four successive barriers
    (listed B3, B1, B4, B2), and the root's K follows J (its barrier on A,
    the root's previous node, is implied: A is J's ancestor). Phase 1: the
    same shape over tasks 1 and 2 with A held on the gate, so J's FIFO and
    the Bs block on barriers and the root blocks on J while the guest holds
    for 2.5 s, past the 2 s blocked-barrier report, then pauses and
    unpauses: the waits are quiesced and parsed again, and the gate
    releases the whole diamond. Phase 2: a join whose barriers have all
    passed (on phase 0 and 1 nodes) joined with a fresh one."""
    g = Graph(range(1, 6), 2)
    g.phase()
    a = g.node(0)
    bs = [g.node(fid, [a.id], chain=0 if fid == 1 else None) for fid in range(1, 5)]
    j = g.node(5, [b.id for b in bs], chain=0, order=[2, 0, 3, 1])
    g.node(0, [j.id], chain=0)
    g.phase(event="pause_long")
    a = g.node(0, gate=True, task=1, chain=1)
    bs = [g.node(fid, [a.id], task=fid % 2, chain=1 if fid == 1 else None) for fid in range(1, 5)]
    j = g.node(5, [b.id for b in bs], chain=1, task=1)
    g.node(0, [j.id], chain=1)
    g.phase()
    x = g.node(3, [0, 5, 6, 11], chain=2)
    g.node(4, [x.id, 4, 9], chain=0)
    return emit("interplay_dag_diamond", interplay_dag_diamond.__doc__, g)


@scenario
def interplay_dag_fan_in_16():
    """Sixteen children each run a source node, on its own chain, held on
    the gate: all sixteen block on the gate's stamp and the root blocks on
    its join, a node with sixteen barriers, one on each child (seventeen
    FIFOs waiting at once, under the waiter mask's 32, docs/testing.md
    6.7). The gate's NOP releases the sixteen; the join reads every
    child's result. Phase 1 fans out from the root to the sixteen (each
    continuing its chain) and joins them again on the root, sixteen
    barriers on stamps not yet pushed when it is parsed."""
    g = Graph(range(1, 17), 3)
    g.phase(event="hold")
    srcs = [g.node(fid, gate=True, task=fid % 3) for fid in range(1, 17)]
    g.node(0, [n.id for n in srcs])
    g.phase()
    r = g.node(0, [16], chain=16)
    outs = [g.node(fid, [r.id], chain=fid - 1, task=fid % 3) for fid in range(1, 17)]
    g.node(0, [n.id for n in reversed(outs)], chain=16)
    return emit("interplay_dag_fan_in_16", interplay_dag_fan_in_16.__doc__, g)


def chain_over(g, fids, n, first_gate=False, task_mod=1, start_chain=None):
    prev = None
    chain = start_chain
    out = []
    for k in range(n):
        fid = fids[k % len(fids)]
        node = g.node(fid, [prev] if prev is not None else [], chain=chain, gate=first_gate and k == 0,
                      task=k % task_mod, op=abi.OP_EXEC_INDIRECT2 if k % 5 == 4 else abi.OP_EXEC_INDIRECT3)
        chain = node.chain
        prev = node.id
        out.append(node)
    return out


@scenario
def interplay_dag_long_chain_16():
    """One chain of 51 nodes over the root and all sixteen children, three
    times round (0, 1, ..., 16), each waiting on the one before, written in
    order and rung last FIFO first: the order is total, so the 51
    interrupts and the stamp slots each one saw are locked in order, the
    counter reads 51 and logged its first fifteen steps. Phase 1 runs the
    same chain round again, from child 16 down to the root, over three
    tasks, with its first node held on the gate: sixteen FIFOs wait on
    the one before, the root on child 1."""
    g = Graph(range(1, 17), 3)
    g.phase()
    chain_over(g, list(range(17)), 51)
    g.phase(event="hold")
    chain_over(g, list(range(16, -1, -1)), 17, first_gate=True, task_mod=3, start_chain=0)
    return emit("interplay_dag_long_chain_16", interplay_dag_long_chain_16.__doc__, g)


@scenario
def interplay_dag_chain_across_suspend():
    """A chain over the root and children 1-8 and two tasks: its first nine
    nodes run, the tenth waits on the gate as well, and the eleven after
    it wait down the chain. The guest suspends (rvg_device_will_suspend
    quiesces every wait, rvg_device_finish_suspend drains) and resumes on a
    fresh device, which recreates the children (their nodes say running,
    docs/testing.md 6.24) and parses each held command again; the gate's NOP
    then releases the rest in order. Total order: interrupts and snapshots
    locked in order."""
    g = Graph(range(1, 9), 2)
    g.phase(event="suspend")
    ns = chain_over(g, list(range(9)), 21, task_mod=2)
    ns[9].bars.append(("gate",))
    return emit("interplay_dag_chain_across_suspend", interplay_dag_chain_across_suspend.__doc__, g)


@scenario
def interplay_dag_chain_pause_past_report():
    """A chain over children 1-6 and the root held on the gate for 2.5 s:
    every FIFO downstream is blocked in a barrier past the 2 s timeout, has
    logged and posted a stamp-wait report and now waits without a timeout
    (every child's stampIndex is its id, so the report's lookup holds). A
    pause wakes and quiesces them all, the unpause starts every FIFO again,
    and they block again; the gate's NOP releases the chain in order. The
    report changes nothing the guest sees."""
    g = Graph(range(1, 7), 1)
    g.phase(event="pause_long")
    chain_over(g, [3, 1, 6, 0, 2, 5, 4], 14, first_gate=True)
    return emit("interplay_dag_chain_pause_past_report", interplay_dag_chain_pause_past_report.__doc__, g)


@scenario
def interplay_dag_redefined_chain():
    """Barrier chains through child FIFOs deleted and defined again. Phase
    0 runs two chains over children 1-6 and the root. Then child 3 is
    deleted and defined again, and child 5 deleted: each delete zeroes the
    next node's pageListPage (W3, the dumps), and the guest writes 4's and
    6's back. Phase 1 runs a chain through the new child 3 (its stamps go
    on from the old incarnation's), with barriers also on phase 0's nodes
    on child 3 and on the deleted child 5, whose slots still hold later
    values, so they pass at once; it is held across a pause, whose unpause
    starts every child again from its node (4 and 6 map their rings from
    the page lists the guest wrote back). Phase 2 defines 5 again and runs
    through it across a suspend and resume."""
    g = Graph(range(1, 7), 2)
    g.phase()
    c0 = chain_over(g, [0, 1, 2, 3, 4, 5, 6], 14, task_mod=2)
    c1 = chain_over(g, [6, 5, 3, 1], 6)
    on3 = [n.id for n in c0 + c1 if n.fid == 3]
    on5 = [n.id for n in c0 + c1 if n.fid == 5]
    g.phase(pre=[("delete", 3), ("define", 3), ("delete", 5)], event="pause")
    a = g.node(3, [on3[0], on5[-1]], gate=True, chain=c0[-1].chain)
    b = g.node(4, [a.id, on3[-1]], task=1)
    c = g.node(3, [b.id], chain=a.chain)
    g.node(6, [c.id, on5[0]], chain=c1[-1].chain, task=1)
    g.node(0, [c.id], chain=b.chain)
    g.phase(pre=[("define", 5)], event="suspend")
    x = g.node(5, [on5[-1]], gate=True, chain=a.chain)
    y = g.node(4, [x.id], chain=x.chain, task=1)
    g.node(5, [y.id], chain=x.chain)
    g.node(6, [x.id], chain=c1[-1].chain)
    return emit("interplay_dag_redefined_chain", interplay_dag_redefined_chain.__doc__, g)


@scenario
def interplay_dag_pressure():
    """Pending-queue pressure inside a graph: the pressure FIFO (child 18)
    holds a faulted head (opcode 99) and 129 NOPs, so its thread blocks
    pushing the 128th NOP's stamp with 128 pending, and its two nodes wait
    behind them. Children 1 and 4 wait on its first node, and its second on
    child 3, whose chain with child 2 runs meanwhile. 0x1028 = 18 publishes
    the 128 in one interrupt, the last two NOPs one each, then the pressure
    FIFO's nodes and the ones waiting on them run."""
    g = Graph(range(1, 5), 2)
    g.phase(pressure=True)
    p1 = g.node(PRESS)
    n1 = g.node(1, [p1.id], chain=p1.chain)
    c2 = g.node(2, task=1)
    c3 = g.node(3, [c2.id], chain=c2.chain)
    g.node(PRESS, [c3.id], chain=c3.chain, task=1)
    g.node(4, [n1.id, c2.id], chain=n1.chain)
    return emit("interplay_dag_pressure", interplay_dag_pressure.__doc__, g)


@scenario
def interplay_dag_w3_unrepaired():
    """A chain through the child after a deleted one, when the guest does
    not write its page list back. Phase 0 runs a chain over the root and
    children 1-6. Deleting child 3 zeroes node 4's pageListPage (W3);
    phase 1 then holds a chain 1 -> 4 -> 5 -> 6 on the gate (with an
    independent one on child 2 and the root) across a pause, and the
    unpause's -start of child 4 fails on the zeroed page list: its node
    still says running, and when the gate opens child 1's node runs, while
    child 4's is never parsed, so 5 and 6 stay blocked on it. The guest
    recovers child 4 (its page list written again, 49, which zeroes node
    5's in turn, then 48) and writes the lost command again at the start
    of its ring with the same stamp: the chain completes and matches the
    control."""
    g = Graph(range(1, 7), 1)
    g.phase()
    chain_over(g, [0, 1, 2, 3, 4, 5, 6], 7)
    g.phase(pre=[("delete", 3)], event="pause")
    a = g.node(1, [6], gate=True, chain=0)
    b = g.node(4, [a.id], chain=0)
    c = g.node(5, [b.id], chain=0)
    d = g.node(6, [c.id], chain=0)
    e = g.node(2, [3])
    f = g.node(0, [e.id], chain=e.chain)
    g.finish()
    em = Emitter("interplay_dag_w3_unrepaired", interplay_dag_w3_unrepaired.__doc__, g, repair=False)
    s = em.s
    em.setup()
    em.run_phase(g.phases[0])
    ph = g.phases[1]
    em.pre(ph)
    s.mark("phase 1: pause, child 4 not repaired")
    em.write_ring(ph)
    em.hold(ph)
    em.open_gate(ph)
    em.wait_last([a, e, f], "not behind child 4")
    c4 = em.fifos[4]
    c4.wait_stamp(b.stamp, timeout_ms=300, label="node %d on child 4: never parsed" % b.id)
    em.events("phase 1: the gate opened, child 4 did not start")
    em.stamps("phase 1: stamps, child 4's node lost")
    s.dump(c4.node_gpa(0), abi.CHILD_NODE_SIZE * 2, "nodes 4 and 5: 4 running, never started")
    # Recovery: the page list written again, 49, 48; the lost command
    # written again at the ring's start with its stamp.
    c4.written = 0
    c4.setup_node()
    c4.delete(label="delete child 4, which did not start")
    s.dump(c4.node_gpa(0), abi.CHILD_NODE_SIZE * 2, "nodes 4 and 5 after deleting 4")
    c4.define(label="define child 4 again")
    t = em.tasks[b.task]
    cbs = [t.cmdbuf(sg) for sg in node_segments(t, b, t.reg_l)]
    c4.cmd(b.op, exec_payload(t.tid, cbs, t.resources(), b.op), barriers=[(1, a.stamp)], stamp=b.stamp)
    c4.doorbell()
    em.done(ph)
    em.final()
    return s


def abandon_graph():
    """Child 1's node runs; child 2's waits on it and on the gate; child
    3's waits on child 2's."""
    g = Graph(range(1, 4), 1)
    g.phase(event="hold")
    a = g.node(1)
    b = g.node(2, [a.id], gate=True, chain=a.chain)
    c = g.node(3, [b.id], chain=a.chain)
    return g.finish(), (a, b, c)


def abandon(name, doc, watchdog_s=60):
    """Holds children 2 and 3 blocked, then rvg_device_will_suspend and
    unpause."""
    g, nodes = abandon_graph()
    em = Emitter(name, doc, g, watchdog_s=watchdog_s)
    s = em.s
    em.setup()
    ph = g.phases[0]
    s.mark("phase 0: held, then willSuspend and unpause")
    em.write_ring(ph)
    em.hold(ph)
    for fid in (2, 3):
        s.dump(em.fifos[fid].node_gpa(0), abi.CHILD_NODE_SIZE, "node %d blocked" % fid)
    s.api("willSuspend")
    s.api("unpause")
    em.events("willSuspend, unpause")
    s.fault_state("abandoned")
    return em, nodes


@scenario
def interplay_dag_abandon_blocked():
    """rvg_device_will_suspend then unpause (an abandoned suspend) while
    children 2 and 3 are blocked on barriers. rvg_device_will_suspend quiesces
    every FIFO but, unlike pause and rvg_device_finish_suspend, does not wake
    the waits (it never signals the FIFOs), so both FIFOs' threads stay in
    DeviceCore::wait_stamp with their waits registered, and unpause's
    Fifo::start clears the quiesced flag and starts a second thread on each.
    The new threads find nothing to read: a child counts what it read locally,
    and the old thread had read the blocked command. The gate's NOP then wakes
    the old threads, whose Fifo::barrier_wait now sees the flag clear: they run
    the held work, which completes in order and matches the control. The new
    threads are left waiting for a doorbell; the scenario rings neither FIFO
    again (which thread a doorbell wakes would be timing)."""
    em, (a, b, c) = abandon("interplay_dag_abandon_blocked", interplay_dag_abandon_blocked.__doc__)
    ph = em.g.phases[0]
    em.open_gate(ph)
    em.done(ph)
    em.final()
    return em.s


@scenario
def interplay_dag_abandon_blocked_dies():
    """The abandoned suspend of interplay_dag_abandon_blocked, but the guest
    then rings the blocked child with more bytes before it opens the gate.
    Child 2's blocked command is a NOP on the gate's stamp with a 4 MiB payload
    of zeros (its ring is 257 pages). After rvg_device_will_suspend and
    unpause, child 2's new thread starts reading at the offset Fifo::map_ring
    took from the node's readCount, before the blocked command, but counts as
    read what the old thread read, so it waits for more bytes. The guest's
    doorbell claims 4 MiB more; the new thread parses the blocked command
    again, copying its payload, and registers child 2's wait while the old
    thread's is still registered: DeviceCore::add_wait's check that the
    waiter's bit is clear in the waiting word dies, host death at that doorbell
    (the 4 MiB copy puts it after the doorbell's record, as in
    interplay_limits_child_wait_alias_dies). (A random walk found the abandoned
    suspend over blocked work. With the root FIFO blocked instead, the new root
    thread reads on at once from past the blocked command, which the old thread
    had consumed, into unwritten ring bytes it skips
    (docs/testing.md 6.6, step 1) until it wraps and replays the ring from its
    start, and the define of the blocked child asserts: measured, but racing
    with unpause's return, so not locked.)"""
    s = Scenario("interplay_dag_abandon_blocked_dies", watchdog_s=30,
                 doc=interplay_dag_abandon_blocked_dies.__doc__)
    s.bring_up()
    track_stamps(s, GATE + 1)
    gate = ChildFIFO(s, GATE)
    gate.setup_node()
    gate.define()
    c2 = ChildFIFO(s, 2, pages=257)
    c2.setup_node()
    c2.define()
    s.events("set up")
    big = 4 << 20
    hdr = struct.pack("<HHII", abi.OP_NOP, 1, big, 1) + struct.pack("<II", GATE, 1)
    c2.raw(hdr)
    c2.written += big - len(hdr)   # the payload: zeros, as the guest's RAM already is
    c2.stamp = 1
    c2.doorbell()
    c2.wait_stamp(1, timeout_ms=300, label="child 2 blocked on the gate")
    s.dump(c2.node_gpa(0), abi.CHILD_NODE_SIZE, "node 2 blocked: read count 0")
    s.api("willSuspend")
    s.api("unpause")
    s.events("willSuspend, unpause")
    s.fault_state("abandoned")
    c2.written += big
    c2.doorbell()
    c2.wait_stamp(1, timeout_ms=2000, label="not reached: the second thread's wait asserts")
    return s


@scenario
def interplay_dag_alias_report():
    """The stamp-wait report over an aliased waiter does not kill the host.
    Child 33's node waits on the gate for 2.5 s, and the root's waits on
    it: the waiter mask is 32-bit (1 << 33 sets bit 1, docs/testing.md 6.7
    and 7.4), so the report posted at 2 s walks waiting stamp indices 0 and
    1. For 1 it finds child 1, which exists and is idle, reads its waiting
    info's next index, UINT32_MAX, logs "waiting on invalid next
    waitingIndex" and stops walking. Nothing the guest sees changes: the
    gate releases child 33, then the root. (Child 1 must not wait itself:
    that is the alias death, interplay_limits_child_wait_alias_dies.)"""
    g = Graph([1, 33], 1)
    g.phase(event="hold_long")
    a = g.node(33, gate=True)
    g.node(0, [a.id], chain=a.chain)
    g.phase()
    b = g.node(1, [a.id], chain=a.chain)
    g.node(33, [b.id], chain=a.chain)
    s = emit("interplay_dag_alias_report", interplay_dag_alias_report.__doc__, g)
    s.dump(s.fifo.base_gpa() + 4 * 33, 4, "child 33's stamp slot")
    return s


@scenario
def interplay_dag_program_order_same_task():
    """Same-FIFO program order orders GPU work when the execs share a task,
    with no barrier between them. Child 1 carries opcode 99 (it faults),
    then exec A, which runs inc_k 800 times on a counter in one compute
    pass, then exec B, which copies the counter out, both on task 1 (B as
    opcode 55): they run behind the faulted head, one command buffer after
    the other on the task's queue (in-process the device's exec queue; in
    the service the task's), and Metal orders them on the counter buffer
    they share. The guest waits for B's copy to read 800 (guest memory: the
    stamps are held behind the fault, so no stamp can say B finished, and
    none is published alone, so the interrupt count is fixed). 0x1028 = 1
    then publishes all three stamps in one interrupt. The same pair on two
    tasks is not ordered (the module's docstring), and is not locked."""
    n = 800
    s = Scenario("interplay_dag_program_order_same_task", doc=interplay_dag_program_order_same_task.__doc__)
    s.bring_up()
    track_stamps(s, 2)
    shared = s.alloc_page(SHARED)
    t = DagTask(s, 1, [shared + i for i in range(SHARED)])
    c = ChildFIFO(s, 1)
    c.setup_node()
    c.define()
    t.define()
    t.install()
    s.events("set up")
    c.cmd(99)
    inc = (C.set_pipeline(t.pso_inc) + C.set_buffers(0, [(t.reg_l, 0)])
           + C.dispatch_threads((1, 1, 1), (1, 1, 1)) * n)
    c.cmd(abi.OP_EXEC_INDIRECT3, exec_payload(t.tid, [t.cmdbuf(seg(abi.ENC_COMPUTE, inc))], t.resources()))
    cp = B.copy_buffer_to_buffer(t.reg_l, t.reg_l, 0, 0x100, 0x40)
    last = c.cmd(abi.OP_EXEC_INDIRECT2, exec_payload(t.tid, [t.cmdbuf(seg(abi.ENC_BLIT, cp))], t.resources(),
                                                     abi.OP_EXEC_INDIRECT2))
    c.doorbell()
    child_fault(s, c)
    out = t.gpa(t.reg_l_va + 0x100)
    s.wait(gpa=out, eq=n, label="B's copy of the counter reads %d" % n)
    s.dump(out, 0x40, "B's copy")
    s.dump(c.stamp_gpa(), 4, "child 1's slot: nothing published")
    s.events("child 1's fault")
    resume_child(s, 1)
    c.wait_stamp(last)
    s.events("0x1028 = 1: three stamps in one interrupt", order=True, irq_stamps=True)
    s.r(abi.R_FAULT_MASK, "resumed")
    return s
