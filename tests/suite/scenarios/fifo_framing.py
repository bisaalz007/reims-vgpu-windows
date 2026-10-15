"""Command framing on the root FIFO: every branch of the FIFO thread's loop
(Fifo::run) before the dispatch (docs/testing.md, section 6.6), the ring's
wrap, and many commands per doorbell.

The root FIFO reads a header's bytes into a local cursor, and publishes the
read count (0x100c) only when a command is dispatched. A header rejected
before that leaves the published count behind, so the bytes still
"available" are counted from the old read count, past the cursor: the FIFO
goes on to parse ring bytes the guest never counted in 0x1008.
"""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.fifo_util import stamps, track_stamps
from pgtest.registry import scenario
from pgtest.scenario import Scenario

HDR = abi.HDR.pack
BAR = abi.BARRIER.pack


def stale_after_bad_header(name, bad_length, doc):
    """Two NOPs, then a header with a bad length as the only command the
    doorbell counts. The ring bytes after it, which 0x1008 never counts,
    hold a NOP (stamp 3) and then a command whose barrier waits on stamp
    slot 5 for 1, which nothing signals: the FIFO parses the stale NOP,
    publishes stamp 3, advances 0x100c past 0x1008, and blocks in the
    barrier."""
    s = Scenario(name, doc=doc)
    f = s.bring_up()
    track_stamps(s, 1)
    f.cmd(abi.OP_NOP)
    st = f.cmd(abi.OP_NOP)
    f.doorbell()
    f.wait_stamp(st)
    s.events("two NOPs")
    bad_at = f.written
    f.raw(HDR(abi.OP_NOP, 0, bad_length, 0x99))
    f.ring_write(bad_at + 12, HDR(abi.OP_NOP, 0, 12, 3))
    f.ring_write(bad_at + 24, HDR(abi.OP_NOP, 1, 20, 4) + BAR(5, 1))
    f.doorbell()
    s.wait(gpa=f.stamp_gpa(), eq=3, label="stamp 3, from bytes past 0x1008")
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=bad_at + 24, label="0x100c past 0x1008")
    s.events("the stale NOP", irq_stamps=True)
    s.r(abi.R_FIFO_READ)
    s.fault_state("no fault")
    s.diff("only stamp slot 0 changed")
    return s


@scenario
def fifo_frame_length_zero_stale():
    """A header of length 0 (< 12) is skipped without a fault or a stamp and
    without publishing the read count; the FIFO then parses the stale bytes
    after it (docs/testing.md, section 7.1)."""
    return stale_after_bad_header("fifo_frame_length_zero_stale", 0, fifo_frame_length_zero_stale.__doc__ + stale_after_bad_header.__doc__)


@scenario
def fifo_frame_length_11_stale():
    """A header of length 11, one short of the header, is skipped as length
    0 is, and the stale bytes after it parse."""
    return stale_after_bad_header("fifo_frame_length_11_stale", 11, fifo_frame_length_11_stale.__doc__ + stale_after_bad_header.__doc__)


@scenario
def fifo_frame_length_over_ring_stale():
    """A header longer than the ring (0x7c00 bytes: length 0x8000 - start
    0x400) by one byte is skipped the same way."""
    return stale_after_bad_header("fifo_frame_length_over_ring_stale", 0x7C01,
                                  fifo_frame_length_over_ring_stale.__doc__ + stale_after_bad_header.__doc__)


@scenario
def fifo_frame_length_skip():
    """Bad lengths (0, 11, the ring size + 1, 0xffffffff) inside one doorbell
    with a NOP after them: each header's 12 bytes are skipped, nothing is
    signalled or faulted for them, and the NOP after them runs; the read
    count then covers the skipped headers too."""
    s = Scenario("fifo_frame_length_skip", doc=fifo_frame_length_skip.__doc__)
    f = s.bring_up()
    track_stamps(s, 1)
    f.cmd(abi.OP_NOP)
    for n in (0, 11, 0x7C01, 0xFFFFFFFF):
        f.raw(HDR(abi.OP_NOP, 0, n, 0x90))
    st = f.cmd(abi.OP_NOP)
    f.doorbell()
    f.wait_stamp(st)
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written)
    s.events("one NOP, four skipped headers, one NOP", irq_stamps=True)
    s.r(abi.R_FIFO_READ)
    s.fault_state("no fault")
    stamps(s, 1)
    return s


@scenario
def fifo_frame_whole_ring():
    """A command exactly as long as the ring (0x3c00: one page, start
    0x400) is valid (length <= commandLength); the next command starts at
    ring offset 0 again, and the read count runs on past the ring's
    size."""
    s = Scenario("fifo_frame_whole_ring", doc=fifo_frame_whole_ring.__doc__)
    f = s.bring_up(fifo_pages=1)
    ring = f.ring_size()
    s.submit(abi.OP_NOP, bytes(ring - 12))
    s.r(abi.R_FIFO_READ)
    s.submit(abi.OP_NOP)
    s.r(abi.R_FIFO_READ)
    s.submit(abi.OP_NOP, bytes(ring - 12))
    s.r(abi.R_FIFO_READ)
    s.events("three commands")
    stamps(s, 1)
    return s


@scenario
def fifo_frame_wrap():
    """Commands straddling the ring's end (one page, start 0x400: a ring of
    0x3c00 bytes) are read across it: a header split 6 + 6; a DefineChildFIFO
    whose 4-byte id is split 1 + 3 (the page after the ring holds 0xff, so a
    read past the end would name no valid FIFO); a barrier whose index is split
    2 + 2 (0xff bytes would make it out of range); and a faulting command whose
    header starts 8 bytes before the end, whose fault offset (0x103c) is that
    ring offset, not the free-running count."""
    s = Scenario("fifo_frame_wrap", doc=fifo_frame_wrap.__doc__)
    f = s.bring_up(fifo_pages=1)
    after = s.alloc_page()
    s.fill(s.pa(after), PAGE, 0xFF)
    from pgtest.fifo_util import child, node
    c = child(s, 1, define=False)
    ring = f.ring_size()

    def pad_to(off):
        """A NOP whose payload brings the ring offset to `off`."""
        cur = f.written % ring
        n = (off - cur) % ring
        assert n >= 12
        return s.submit(abi.OP_NOP, bytes(n - 12), label="pad to 0x%x" % off)

    pad_to(ring - 6)
    s.submit(abi.OP_NOP, label="a header split 6 + 6")
    s.r(abi.R_FIFO_READ)
    pad_to(ring - 13)
    s.submit(abi.OP_DEFINE_CHILD_FIFO, struct.pack("<I", 1), label="define child 1, its id split 1 + 3")
    node(s, 1, "child 1 defined")
    pad_to(ring - 14)
    s.submit(abi.OP_NOP, barriers=[(0, 1)], label="a barrier whose index is split 2 + 2")
    pad_to(ring - 8)
    st = f.cmd(99)
    f.doorbell()
    s.wait_fault()
    s.fault_state("a faulting header split 8 + 4")
    s.r(abi.R_FIFO_READ)
    s.resume_root()
    f.wait_stamp(st)
    s.fault_state("resumed")
    s.events("wrapped")
    s.diff("stamps and child 1's node; the page after the ring unchanged")
    return s


@scenario
def fifo_frame_many_per_doorbell():
    """300 commands of mixed lengths in one doorbell: NOPs with 0-40 payload
    bytes and Debug commands. Each is signalled at the head, so each
    publishes its own stamp and raises one interrupt, in order (the
    pending-stamp queue's pump)."""
    s = Scenario("fifo_frame_many_per_doorbell", doc=fifo_frame_many_per_doorbell.__doc__)
    f = s.bring_up()
    track_stamps(s, 1)
    for i in range(300):
        if i % 7 == 3:
            st = f.cmd(abi.OP_DEBUG, b"x" * (1 + i % 13))
        else:
            st = f.cmd(abi.OP_NOP, bytes(i % 41))
    f.doorbell()
    f.wait_stamp(st, irqs=300)
    s.events("300 commands", irq_stamps=True)
    s.r(abi.R_FIFO_READ)
    stamps(s, 1)
    return s


@scenario
def fifo_frame_underflow():
    """A barrier count whose barriers do not fit the length faults the
    command: its stamp is pushed and faulted (the separate "insufficient
    bytes" check tests the same condition, so it never fires). Only the
    header is consumed: with length 12 and one barrier, the next command
    parses cleanly; with length 16 and two barriers, the header's 4 extra
    bytes become the start of the next header, here a NOP (stamp 5) the
    guest spelled across the boundary."""
    s = Scenario("fifo_frame_underflow", doc=fifo_frame_underflow.__doc__)
    f = s.bring_up()
    track_stamps(s, 1)
    f.cmd(abi.OP_NOP)
    f.raw(HDR(abi.OP_NOP, 1, 12, 2))
    st = f.cmd(abi.OP_NOP, stamp=3)
    f.doorbell()
    s.wait_fault()
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written)
    s.events("one barrier in 12 bytes", irq_stamps=True)
    s.fault_state("faulted")
    s.r(abi.R_FIFO_READ)
    s.resume_root()
    f.wait_stamp(st)
    s.events("resumed", irq_stamps=True)
    # length 16, two barriers: 4 bytes left over, then 8 more make a NOP.
    f.raw(HDR(abi.OP_NOP, 2, 16, 4) + struct.pack("<HH", abi.OP_NOP, 0))
    f.raw(struct.pack("<II", 12, 5))
    st = f.cmd(abi.OP_NOP, stamp=6)
    f.doorbell()
    s.wait_fault()
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written)
    s.fault_state("two barriers in 16 bytes")
    s.resume_root()
    f.wait_stamp(6)
    s.events("the NOP spelled across the boundary ran", irq_stamps=True)
    stamps(s, 1)
    return s


@scenario
def fifo_frame_barrier_count():
    """64 barriers, all satisfied (index 0, value 0), run the command. 66
    barriers fault it before any barrier is read (barrierCount > 64), and
    only its header is consumed: its 528 barrier bytes are then parsed as
    commands. Zeros parse as 44 headers of length 0, skipped; 44 NOPs
    spelled into them run (stamps 3-46), and publish with the rest when
    0x1024 resumes the faulted head."""
    s = Scenario("fifo_frame_barrier_count", doc=fifo_frame_barrier_count.__doc__)
    f = s.bring_up()
    track_stamps(s, 1)
    s.submit(abi.OP_NOP, barriers=[(0, 0)] * 64, label="64 barriers")
    s.r(abi.R_FIFO_READ)
    # 66 barriers of zeros.
    f.raw(HDR(abi.OP_NOP, 66, 12 + 66 * 8, 2) + bytes(66 * 8))
    st = f.cmd(abi.OP_NOP, stamp=3)
    f.doorbell()
    s.wait_fault()
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written)
    s.fault_state("66 zero barriers")
    s.resume_root()
    f.wait_stamp(st)
    s.events("zero barriers", irq_stamps=True)
    # 66 barriers that spell 44 NOPs.
    smuggled = b"".join(HDR(abi.OP_NOP, 0, 12, 5 + i) for i in range(44))
    f.raw(HDR(abi.OP_NOP, 66, 12 + 66 * 8, 4) + smuggled)
    st = f.cmd(abi.OP_NOP, stamp=49)
    f.doorbell()
    s.wait_fault()
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written)
    s.fault_state("66 barriers spelling NOPs")
    stamps(s, 1, "stamp slot while faulted")
    s.resume_root()
    f.wait_stamp(st)
    s.events("the smuggled NOPs published with the resume", irq_stamps=True)
    stamps(s, 1)
    return s


@scenario
def fifo_frame_partial_root():
    """The root FIFO never waits inside a command: bytes are "available"
    while 0x1008 differs from the published read count, and that count
    moves only at dispatch. A doorbell covering only the header of a
    40-byte NOP runs the whole NOP from the ring bytes past 0x1008,
    publishes its stamp, and leaves 0x100c past 0x1008; the FIFO then
    parses on, here into a command blocked on stamp slot 5."""
    s = Scenario("fifo_frame_partial_root", doc=fifo_frame_partial_root.__doc__)
    f = s.bring_up()
    track_stamps(s, 1)
    s.submit(abi.OP_NOP)
    at = f.written
    st = f.cmd(abi.OP_NOP, bytes(28))
    f.raw(HDR(abi.OP_NOP, 1, 20, st + 1) + BAR(5, 1))
    f.doorbell(written=at + 12)
    f.wait_stamp(st, label="the whole NOP ran")
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=at + 40)
    s.r(abi.R_FIFO_READ)
    s.fault_state("no fault")
    s.events("header only", irq_stamps=True)
    return s


@scenario
def fifo_frame_partial_child():
    """A child FIFO counts what it has read locally, so a doorbell covering
    part of a command makes it wait for the rest: a 40-byte NOP with only
    its header counted does not run, and runs at the next doorbell;
    likewise a command cut inside its barriers. The node's readCount moves
    only at dispatch. The scenario ends with child 1 cut inside a payload
    and child 2 inside its barriers: teardown ends both reads (the payload
    read's `continue` and the barrier read's `break`) without dispatching
    them."""
    s = Scenario("fifo_frame_partial_child", doc=fifo_frame_partial_child.__doc__)
    s.bring_up()
    track_stamps(s, 3)
    from pgtest.fifo_util import child, node
    c = child(s, 1)
    c2 = child(s, 2)
    st = c.cmd(abi.OP_NOP, bytes(28))
    c.doorbell(written=12)
    c.wait_stamp(st, timeout_ms=500, label="header only: waiting for the payload")
    node(s, 1, "waiting for the payload")
    c.doorbell()
    c.wait_stamp(st)
    node(s, 1, "completed")
    st = c.cmd(abi.OP_NOP, barriers=[(0, 1), (0, 1)])
    c.doorbell(written=c.written - 12)
    c.wait_stamp(st, timeout_ms=500, label="cut inside the barriers")
    c.doorbell()
    c.wait_stamp(st)
    node(s, 1, "completed")
    s.events("partial commands completed", irq_stamps=True)
    st = c.cmd(abi.OP_NOP, bytes(28))
    c.doorbell(written=c.written - 1)
    st2 = c2.cmd(abi.OP_NOP, barriers=[(0, 1), (0, 1)])
    c2.doorbell(written=c2.written - 4)
    c.wait_stamp(st, timeout_ms=300, label="child 1 cut inside its payload")
    c2.wait_stamp(st2, timeout_ms=300, label="child 2 cut inside its barriers")
    node(s, 1, "left pending")
    node(s, 2, "left pending")
    s.events("left pending")
    return s
