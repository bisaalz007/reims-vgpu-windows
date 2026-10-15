"""Seeded walks over the IOSurface host device and its consumers
(pgtest/iosfcwalk_util.py): each seed is one scenario,
interplay_iosfcwalk_<seed>, whose program the walker builds from the seed
alone.

The IOSurface host device QEMU's apple-gfx-mmio creates beside the
RvgDevice (docs/testing.md 6.23) is driven at random through its ring and
registers: batches of map, unmap and test commands with bad ids, ids whose
descriptor page the table does not name, an invalid descriptor and
unknown commands; the ring moved to rings of other sizes with the
counters kept, commands written into the old ring and rung after the
move, the read counter rewound (commands run again) or moved past
commands never rung, a doorbell while no ring is mapped; the descriptor
table rewritten (every surface dropped) and the descriptor format
switched with a doorbell before the new table. Interleaved with its
consumers: a GPU task per surface filling and copying into it through a
mapper buffer (type 15), deleted (37) and made again; opcode 8 presents of
the surfaces and opcode 7 presents of task 0's backings on one or two
heads; work held on a gate FIFO or rung while paused while the guest
unmaps, remaps, tests, rewrites or tears down the surfaces it names (a
fill whose buffer is not made faults its child when its id has no
surface; a present of such an id is refused); and the device's lifecycle:
pause with the IOSurface device and the guest busy, the suspend onto a
fresh device with work held across it, and reset with the guest's set-up
again.

The model predicts every ring entry's status, the descriptors' mapped bits,
the registers, every surface's bytes (against a mirror page), every frame
the heads encode, the callbacks and IOSurface interrupts at every
checkpoint, and which work runs, faults or is refused; every wait succeeds
but those labelled [held] (work shown not to run within 300 ms) and [dies].
python3 -m pgtest.iosfcwalk_util reads the expected files back against it.
Map and isolate only: the IOSurface host device needs a memory map
(apple-gfx-mmio.c:134-162). Some programs end in a death with work held on
the gate: the ring's size written unaligned (the memory map asserts), a
present of a 16400-pixel-wide surface (Metal aborts making its texture), a
second pause.
"""
from pgtest.iosfcwalk_util import build
from pgtest.registry import scenario

# (seed, steps, death): medium programs, long ones (hundreds of steps), and
# programs that end in each death, one of each long.
SEEDS = ([(s, 25 + (s * 13) % 30, None) for s in range(1, 15)]
         + [(s, 220 + (s * 37) % 120, None) for s in range(15, 20)]
         + [(20, 30, "ring_unaligned"), (21, 45, "wide_present"), (22, 35, "pause_twice"),
            (23, 150, "ring_unaligned"), (24, 80, "wide_present"), (25, 120, "pause_twice")])


@scenario(setups=("map", "isolate"), host=True)
def interplay_iosfcwalk():
    return [build(seed, steps, __doc__, death) for seed, steps, death in SEEDS]


@scenario(setups=("map", "isolate"))
def interplay_iosfcwalk_refused_retain():
    """Found by the walk: under isolation, a mapper buffer whose first
    exec found no surface leaves the host holding nothing for its id until
    the buffer is released. The service asks the host for the surface; the
    host's delegate stores the device's answer with try_emplace, nil too,
    and the service caches only a surface it got. Task 2's buffer over
    surface 1 is listed while 1 is unmapped: the exec faults child 1
    ("Attempting to create mapper ref against invalid surface"). The guest
    maps 1 and resumes the child; the next exec makes the buffer and fills
    the page. Unmap 1 is then Busy in-process (the buffer retains the
    device's surface) and done under isolation (the delegate's nil entry
    was kept, so the host holds nothing); test 1 follows it (done
    in-process, BadArgument under isolation), while the service's buffer
    still writes the page. After the guest maps 1 again and 37 deletes the
    buffer (the service's release erases the entry), the next exec retains
    the surface anew, and unmap 1 is Busy in both setups."""
    from pgtest.diff_util import exec_payload
    from pgtest.iosfcwalk_util import FillTask
    from pgtest.lifetime_util import delete_resource
    from pgtest.resources_build import b_fill_buf
    from pgtest.scanout_util import CMD_MAP, CMD_TEST, CMD_UNMAP, FRAME, Scanout
    from pgtest import abi
    from pgtest.task import seg
    x = Scanout("interplay_iosfcwalk_refused_retain", interplay_iosfcwalk_refused_retain.__doc__, displays=(),
                setups=("map", "isolate"), host=False, iosfc=True)
    s = x.s
    x.negotiate(asks=())
    x.root()
    x.iosfc_start()
    x.surface(1, 0x21)
    t = FillTask(s, 2, 1)
    t.define(label="define task 2")
    t.install(label="task 2's object list")
    c = x.child(1)

    def fill(byte):
        cb = t.cmdbuf(seg(abi.ENC_BLIT, b_fill_buf(t.mbuf, 0, 0x40, byte)))
        return c.cmd(abi.OP_EXEC_INDIRECT3, exec_payload(t.tid, [cb], [(t.src, 0), (t.mbuf, 0)]))

    x.iosfc_cmd(CMD_UNMAP, 1, "unmap 1: nothing holds it")
    st = fill(0x11)
    c.doorbell()
    x.node_fault(1, "the buffer's surface refused: child 1 faults")
    x.node(1, "child 1 faulted")
    x.iosfc_cmd(CMD_MAP, 1, "map 1 again")
    s.w(abi.R_RESUME_CHILD, 1)
    c.wait_stamp(st, label="the faulted stamp published")
    st = fill(0x22)
    x.go(1, st, "the buffer made: 0x22 into the page")
    x.dump_surface(1, "surface 1: 0x22", length=0x48)
    x.iosfc_cmd(CMD_UNMAP, 1, "unmap 1: Busy in-process, done under isolation")
    x.iosfc_cmd(CMD_TEST, 1, "test 1: done in-process, BadArgument under isolation")
    st = fill(0x33)
    x.go(1, st, "the buffer still writes the page")
    x.dump_surface(1, "surface 1: 0x33 over what the test left", length=0x48)
    x.iosfc_cmd(CMD_MAP, 1, "map 1 again")
    op, pl = delete_resource(t.tid, t.mbuf)
    s.submit(op, pl, label="37 on the buffer")
    x.iosfc_cmd(CMD_UNMAP, 1, "unmap 1 after 37: done")
    x.iosfc_cmd(CMD_MAP, 1, "map 1 again")
    st = fill(0x44)
    x.go(1, st, "the buffer made again")
    x.iosfc_cmd(CMD_UNMAP, 1, "unmap 1: Busy in both")
    x.dump_surface(1, "surface 1: 0x44", length=0x48)
    s.events("end")
    return s
