"""Scenarios the runner's mem_seq, held encodes and sorted block calls make
possible: a range tracker filled to its limit, a frame held in flight while
the display waits for it, and concurrent FIFOs in the blocks setup.
"""
import struct

from pgtest import abi
from pgtest.display_guest import DisplayGuest, display, p_present3, pattern
from pgtest.compute_objects import Blit as B
from pgtest.interplay_util import World, blit
from pgtest.newops_util import AliasedTask, bring_up_big
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.session import RENDER_OFF, Session

# The tracker's shape: 128 run blocks of at most 240 runs each.
RUNS_PER_BLOCK = 240
RUN_BLOCKS = 128


def _run_vpn(k):
    """Run k of the fill: one page at VA page 2k, so no two runs touch."""
    return 2 * k


@scenario
def newops_range_tracker_limit():
    """A blocks-setup task's mapped-range tracker filled to its limit and
    past it.

    One task of 15 x 4096 pages at depth 2, every leaf the same RAM page, and
    30,720 one-page MapMemory2 commands at every other page (written with
    mem_seq, 2032 at a time into alternate halves of the largest root FIFO)
    fill 128 blocks of 240 runs: each run past a full last block opens a new
    one. Then:
    1. a map into a gap of block 0 finds all 128 blocks full, compacts them,
       which moves nothing, and faults "Range too complex" after the host
       has mapped the page;
    2. one run unmapped in block 64 leaves it 239; a map into a gap near the
       end of block 127 compacts: every block from 64 on takes one run from
       the next, block 127 has room, and the map completes;
    3. a map into block 0 faults again (all full once more);
    4. 200 runs unmapped in each of blocks 10 and 11; a map into block 0
       compacts: block 10 takes all of block 11's 40 runs and removes it,
       the rest cascade, and with 127 blocks the full block 0 is split and
       the map completes;
    5. a 1000-page map and a 1500-page unmap across several blocks.
    Blocks only: map and isolate map through the memory map, which has no
    tracker. Block calls are locked as counts and a digest: one FIFO, one
    task, each batch waited for, so their order is the command order."""
    s = Scenario("newops_range_tracker_limit", setups=("blocks",), watchdog_s=120,
                 doc=newops_range_tracker_limit.__doc__)
    f = bring_up_big(s)
    t = AliasedTask(s, tid=1, tables=15)
    t.define()
    s.events("task defined", blocks_digest=True)

    total = RUNS_PER_BLOCK * RUN_BLOCKS
    half = f.ring_size() // 32 // 2
    k = 0
    while k < total:
        n = min(half, total - k)
        st = t.map_seq(f, n, _run_vpn(k), 2)
        f.doorbell()
        f.wait_stamp(st, timeout_ms=60000)
        k += n
    s.r(abi.R_FAULT_MASK, "no fault after the fill")
    s.events("30,720 runs mapped", blocks_digest=True)

    def blk(b, i):
        return _run_vpn(RUNS_PER_BLOCK * b + i)

    # 1. Everything full: compact() moves nothing, then "Range too complex".
    t.map(blk(0, 0) + 1, fault=True)
    s.fault_state("range too complex, all full")
    s.dump(f.stamp_gpa(), 4, "stamp after the fault")
    s.events("map into block 0: host mapped, then the fault", blocks_digest=True)
    s.resume_root()

    # 2. One run out of block 64; a map near the end of block 127 compacts.
    t.unmap(blk(64, 100))
    t.map(blk(127, 238) + 1)
    s.r(abi.R_FAULT_MASK, "compact made room in block 127")
    s.events("unmap in block 64, map into block 127", blocks_digest=True)

    # 3. Full again.
    t.map(blk(0, 1) + 1, fault=True)
    s.fault_state("range too complex, full again")
    s.events("map into block 0 again", blocks_digest=True)
    s.resume_root()

    # 4. Blocks 10 and 11 drop to 40 runs each; compact removes block 11.
    t.unmap(blk(10, 20), 400)
    t.unmap(blk(11, 20), 400)
    t.map(blk(0, 2) + 1)
    s.r(abi.R_FAULT_MASK, "compact removed a block; block 0 split")
    t.map(blk(0, 3) + 1)
    s.r(abi.R_FAULT_MASK, "room in block 0's half")
    s.events("unmap 400 runs, two maps into block 0", blocks_digest=True)

    # 5. One map over 1000 pages from a gap: it swallows the runs it covers
    #    in its own block, removes a block wholly inside it and trims the
    #    one it ends in. One unmap over 1500 pages trims the blocks at either
    #    end and removes those inside.
    t.map(_run_vpn(12001) + 1, 1000)
    s.r(abi.R_FAULT_MASK, "a map across blocks")
    t.unmap(_run_vpn(20001) + 1, 1500)
    s.r(abi.R_FAULT_MASK, "an unmap across blocks")
    s.events("a map and an unmap across blocks", blocks_digest=True)
    return s


# ---- a frame held in flight ----
#
# Every frame encoded (rvg_display_encode_frame) adds one frame in flight, and
# its command buffer's completion handler takes it away, waking the waiters
# when none is left. A guest present, and a guest connect or disconnect (a
# reset's teardown of the displays too), wait for none in flight, for at
# most 2 s, and then go on regardless. The runner's encode(hold=True) keeps
# the command buffer uncommitted until encode_commit, so nothing completes the
# frame until the scenario says. Pixels come from the library's shaders:
# these are keyed by the host.

W, H = 16, 8


def _held_guest(s, frames=(0,), connect=True):
    """Task 0 with one BGRA backing per seed; port 0 connected and acked."""
    g = DisplayGuest(s)
    refs = [g.backing(W, H, pixels=pattern(W, H, seed))[0] for seed in frames]
    g.install()
    if connect:
        g.connect(0, enabled=0xF)
        g.ack(0, 1)
    return g, refs


def _enc(s, label, **kw):
    s.encode(0, W, H, label=label, samples=[[0, 0], [W - 1, H - 1], [5, 3]], **kw)


def _present(s, ref):
    """Opcode 7 presents ref on port 0: written and rung, not waited for."""
    st = s.fifo.cmd(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, ref))
    s.doorbell()
    return st


@scenario(host=True)
def newops_held_present_timeout():
    """A guest present while an encoded frame is still in flight waits 2 s for
    it, then logs an error and goes on: its stamp is not there after 1 s (a
    wait that times out, locked as such), and is there by 5 s, with its
    new_frame; the held frame, committed afterwards, still draws the first
    frame's pixels."""
    s = Scenario("newops_held_present_timeout", host=True, displays=[display(0)],
                 doc=newops_held_present_timeout.__doc__)
    g, refs = _held_guest(s, frames=(0, 0x40))
    s.fifo.wait_stamp(_present(s, refs[0]))
    s.wait(cb="newFrame")
    s.events("first frame presented")
    _enc(s, "the first frame, held", hold=True)
    st = _present(s, refs[1])
    s.fifo.wait_stamp(st, timeout_ms=1000, label="second present, 1 s: still waiting")
    s.fifo.wait_stamp(st, timeout_ms=5000, label="second present, after the 2 s timeout")
    s.wait(cb="newFrame", cb_count=2)
    s.events("second frame presented")
    s.encode_commit(0, label="the held frame, committed")
    _enc(s, "the second frame")
    s.api("present_counts", display=0)
    s.events("after the commit")
    return s


@scenario(host=True)
def newops_held_present_released():
    """A guest present waits for a held frame; committing the frame while it
    waits completes it, the completion handler finds none left in flight and
    wakes the present, which goes on before its 2 s: its stamp is not there
    after 300 ms, and is once the commit returns."""
    s = Scenario("newops_held_present_released", host=True, displays=[display(0)],
                 doc=newops_held_present_released.__doc__)
    g, refs = _held_guest(s, frames=(0, 0x40))
    st = _present(s, refs[0])
    s.fifo.wait_stamp(st)
    s.wait(cb="newFrame")
    _enc(s, "the first frame, held", hold=True)
    st = _present(s, refs[1])
    s.fifo.wait_stamp(st, timeout_ms=300, label="second present, 300 ms: waiting")
    s.encode_commit(0, label="the held frame, committed while the present waits")
    s.fifo.wait_stamp(st, timeout_ms=1000, label="second present, woken by the completion")
    s.wait(cb="newFrame", cb_count=2)
    s.events("second frame presented")
    _enc(s, "the second frame")
    return s


@scenario(host=True)
def newops_held_two_frames():
    """Two frames held: committing the first leaves one in flight, so its
    completion wakes nobody and the waiting present stays blocked (no stamp
    300 ms later); committing the second wakes it."""
    s = Scenario("newops_held_two_frames", host=True, displays=[display(0)],
                 doc=newops_held_two_frames.__doc__)
    g, refs = _held_guest(s, frames=(0, 0x40))
    st = _present(s, refs[0])
    s.fifo.wait_stamp(st)
    s.wait(cb="newFrame")
    _enc(s, "held 0", hold=True)
    _enc(s, "held 1", hold=True)
    st = _present(s, refs[1])
    s.fifo.wait_stamp(st, timeout_ms=300, label="present, two in flight: waiting")
    s.encode_commit(0, label="held 0 committed")
    s.fifo.wait_stamp(st, timeout_ms=300, label="present, one still in flight: waiting")
    s.encode_commit(1, label="held 1 committed")
    s.fifo.wait_stamp(st, timeout_ms=1000, label="present, none in flight")
    s.wait(cb="newFrame", cb_count=2)
    s.events("second frame presented")
    return s


@scenario(host=True)
def newops_held_disconnect():
    """A guest disconnect (opcode 1, page 0) with a frame held: the
    display's detach of its guest display (DisplayCore::set_guest_display)
    waits 2 s for it, then drops the current frame and the guest display (no
    callback): the held frame, committed after, still draws the frame it was
    encoded with, and a new encode draws none."""
    s = Scenario("newops_held_disconnect", host=True, displays=[display(0)],
                 doc=newops_held_disconnect.__doc__)
    g, refs = _held_guest(s)
    st = _present(s, refs[0])
    s.fifo.wait_stamp(st)
    s.wait(cb="newFrame")
    s.events("presented")
    _enc(s, "held", hold=True)
    st = s.fifo.cmd(abi.OP_DISPLAY_SET_SHARED_STATE_PAGE, struct.pack("<II", 0, 0))
    s.doorbell()
    s.fifo.wait_stamp(st, timeout_ms=1000, label="disconnect, 1 s: waiting")
    s.fifo.wait_stamp(st, timeout_ms=5000, label="disconnect, after the 2 s timeout")
    s.events("disconnected")
    s.encode_commit(0, label="the held frame, committed after the disconnect")
    _enc(s, "after the disconnect")
    return s


@scenario(host=True)
def newops_held_connect():
    """A display attached before the guest connects has no guest display;
    an encode then draws no frame but still counts one in flight, and the
    guest's connect (opcode 1) waits 2 s for it before it attaches the guest
    display."""
    s = Scenario("newops_held_connect", host=True, displays=[display(0)],
                 doc=newops_held_connect.__doc__)
    g, refs = _held_guest(s, connect=False)
    s.events("no nub yet")
    _enc(s, "held, no frame", hold=True)
    g.page(0, 0xF)
    st = s.fifo.cmd(abi.OP_DISPLAY_SET_SHARED_STATE_PAGE, struct.pack("<II", 0, s.pfn(g.pages[0])))
    s.doorbell()
    s.fifo.wait_stamp(st, timeout_ms=1000, label="connect, 1 s: waiting")
    s.fifo.wait_stamp(st, timeout_ms=5000, label="connect, after the 2 s timeout")
    s.events("connected")
    s.encode_commit(0, label="the held frame")
    g.ack(0, 1)
    st = _present(s, refs[0])
    s.fifo.wait_stamp(st)
    s.wait(cb="newFrame")
    s.events("presented")
    _enc(s, "the frame")
    return s


@scenario(host=True)
def newops_held_reset():
    """rvg_device_reset with a frame held: the root's teardown disconnects
    port 0, whose detach of its guest display waits the 2 s out; reset
    returns (no callback), the guest sees the FIFO and root page registers
    cleared, the held frame commits after with the frame it was encoded
    with, and a new encode draws none."""
    s = Scenario("newops_held_reset", host=True, displays=[display(0)],
                 doc=newops_held_reset.__doc__)
    g, refs = _held_guest(s)
    st = _present(s, refs[0])
    s.fifo.wait_stamp(st)
    s.wait(cb="newFrame")
    s.events("presented")
    _enc(s, "held", hold=True)
    s.api("reset")
    s.events("reset")
    s.r(abi.R_FIFO_ENABLE, "after reset")
    s.r(abi.R_ROOT_PAGE, "after reset")
    s.encode_commit(0, label="the held frame, after the reset")
    _enc(s, "after the reset")
    return s


# ---- concurrent FIFOs in the blocks setup, block calls sorted ----
#
# In the blocks setup every exec reads its command buffers (and page-table
# entries) through the host's read_memory callback, from its own FIFO's thread,
# so FIFOs running at once interleave their block calls by timing, which is
# no order (docs/testing.md, decision 8). events(sort_blocks=True) locks
# them as a sorted multiset instead: the set of calls, not their order.


@scenario(host=True)
def newops_session_concurrent_blocks():
    """interplay_session_concurrent in the blocks setup: tasks 1-3 each run
    a full phase on their own FIFO (root, child 1, child 2) with no
    barriers, rung together, twice, with a present, a cursor move and
    device info between the rounds. Each round's block calls are locked
    sorted (the FIFOs' read_memory calls interleave by timing); the results
    are the same bytes in both rounds and on every task."""
    x = Session("newops_session_concurrent_blocks", newops_session_concurrent_blocks.__doc__,
                setups=("blocks",))
    s = x.s
    x.negotiate()
    x.root()
    x.kernel_task()
    x.connect(0)
    ts = [x.task(i, 0x40 * i) for i in (1, 2, 3)]
    x.child(1)
    x.child(2, pages=2)
    s.events("set up")
    for rnd in (0, 1):
        x.batch([(fid, t, t.blit_segments(rnd, 0x99) + t.compute_segments(rnd) + t.render_segments(rnd), ())
                 for fid, t in zip((0, 1, 2), ts)], "round %d" % rnd)
        s.events("round %d: three FIFOs at once" % rnd, sort_blocks=True)
        for t in ts:
            x.results(t, rnd)
        x.present(0, rnd)
        x.cursor(0, 4 + rnd, 2, glyph=(rnd == 0))
        x.device_info(max_key=10 + rnd)
    for t in ts:
        x.same(t, 0, t, 1, "task %d: two rounds" % t.tid)
    s.cmp(ts[0].result_gpa(0) + RENDER_OFF, ts[2].result_gpa(1) + RENDER_OFF, 0x100,
          label="task 1 and task 3 render the same")
    x.teardown()
    return s


PATTERN = 0x04030201


@scenario
def newops_fan_out_fan_in_blocks():
    """interplay_fifo_fan_out_fan_in in the blocks setup: all 63 children
    (maxFIFOCount 64 at version 245) copy their own 4 bytes of A into B,
    gated on the root's exec that fills A; released together, their 63
    execs read their command buffers through read_memory from 63 threads,
    locked as a sorted multiset. The root's NOP with 63 barriers publishes
    after all of them, and B holds all of A."""
    kids = tuple(range(1, 64))
    w = World("newops_fan_out_fan_in_blocks", newops_fan_out_fan_in_blocks.__doc__, children=kids, slots=64,
              setups=("blocks",))
    s = w.s
    a, a_va = w.buffer()
    b, b_va = w.buffer()
    w.ready()
    rs = w.f.stamp + 1
    for fid in kids:
        w.exec_cmd(fid, blit(B.copy_buffer_to_buffer(a, b, 4 * fid, 4 * fid, 4)), barriers=[(0, rs)])
    s.events("63 children defined")
    w.f.cmd(abi.OP_EXEC_INDIRECT3, w.payload([blit(B.fill_pattern(a, 0, 0x100, PATTERN))]))
    fan_in = w.f.cmd(abi.OP_NOP, barriers=[(fid, 1) for fid in kids])
    w.f.doorbell()
    w.f.wait_stamp(rs)
    w.f.wait_stamp(fan_in, timeout_ms=500, label="the fan-in waits for the children")
    s.events("the root's exec")
    for fid in kids:
        w.c[fid].doorbell()
    w.f.wait_stamp(fan_in)
    s.events("63 children, the fan-in", sort_blocks=True)
    w.t.dump(b_va, 0x100, "B")
    w.stamps()
    return s
