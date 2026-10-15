"""The prototype's scenarios, on the suite's library: the shortest path
through each part of the machine (docs/testing.md, section 6.25)."""
import struct

from pgtest import abi
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import GPUTask, Task, rec, seg


@scenario
def smoke_nop():
    """One NOP, then three with one doorbell (section 6.25 a)."""
    s = Scenario("smoke_nop", doc=smoke_nop.__doc__)
    f = s.bring_up()
    s.r(abi.R_FIFO_ENABLE)
    s.diff("bring-up")
    st = s.submit(abi.OP_NOP)
    s.events("one NOP")
    s.r(abi.R_FIFO_READ)
    s.r(abi.R_IRQ_STAMP, "stamp bits")
    s.r(abi.R_IRQ_STAMP, "stamp bits, read again")
    s.r(abi.R_FAULT_MASK)
    s.r(abi.R_FAULT_OFFSET)
    s.diff("after one NOP")
    for _ in range(3):
        st = f.cmd(abi.OP_NOP)
    f.doorbell()
    f.wait_stamp(st)
    s.events("three NOPs")
    s.r(abi.R_FIFO_READ)
    s.diff("after three NOPs")
    return s


@scenario
def smoke_fault():
    """An invalid opcode between two NOPs faults; 0x1024 resumes past it."""
    s = Scenario("smoke_fault", doc=smoke_fault.__doc__)
    f = s.bring_up()
    a = f.cmd(abi.OP_NOP)
    f.cmd(99)
    c = f.cmd(abi.OP_NOP)
    f.doorbell()
    s.wait_fault()
    f.wait_stamp(a)
    s.events("stamp 1 and the fault")
    s.fault_state("faulted")
    s.r(abi.R_FIFO_READ)
    s.diff("faulted")
    s.resume_root()
    f.wait_stamp(c)
    s.events("resumed")
    s.fault_state("resumed")
    s.diff("resumed")
    return s


@scenario
def smoke_task():
    """Define, map, query the heap texture size, and delete a task
    (section 6.25 b, 6.10)."""
    out = []
    s = Scenario("smoke_task", host=True, doc=smoke_task.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=8)
    s.diff("bring-up")
    t.define()
    s.diff("defined")
    t.map()
    word0 = 2 | (1 << 8) | (70 << 16)  # 2D, ShaderRead, RGBA8Unorm
    desc = struct.pack("<IIIIHHHHQ", word0, 256, 256, 1, 1, 1, 1, 0x20, 0)
    ser = struct.pack("<II", 22, 0) + desc
    s.submit(abi.OP_HEAP_TEXTURE_SIZE_AND_ALIGN, struct.pack("<IQQI", 1, 0x40, 16, len(ser)) + ser)
    t.dump(0x40, 16, "MTLSizeAndAlign reply")
    s.metal("heapTextureSizeAndAlign", format=70, w=256, h=256, usage=1, options=0x20)
    t.delete()
    s.diff("deleted")
    s.events("end")
    out.append(s)
    return out


@scenario
def smoke_blit():
    """A blit fill and copy into guest memory (section 6.14's measurement)."""
    s = Scenario("smoke_blit", doc=smoke_blit.__doc__)
    s.bring_up()
    t = GPUTask(s)
    a, a_va = t.buffer(contents=bytes(i & 0xFF for i in range(0x100)))
    b, b_va = t.buffer()
    t.define()
    t.install()
    fill = struct.pack("<IQQB3x", b, 0, 0x100, 0xAB)
    copy = struct.pack("<IIQQQ", a, b, 0, 0x200, 0x100)
    cb = t.cmdbuf(seg(abi.ENC_BLIT, rec(306, fill) + rec(301, copy)))
    t.exec3([cb], resources=[(a, 5), (b, 5)])
    s.r(abi.R_FAULT_MASK)
    t.dump(b_va, 0x300, "buffer B")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("map", "isolate"))
def smoke_crash():
    """A page-table root of 0, then a walk: the walk's read of the root
    asserts. The host dies without isolation; with it the service does, and
    the command faults."""
    s = Scenario("smoke_crash", setups=("map", "isolate"), doc=smoke_crash.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=8)
    t.write_root(pt_root=0)
    t.define()
    word0 = 2 | (1 << 8) | (70 << 16)
    ser = struct.pack("<II", 22, 0) + struct.pack("<IIIIHHHHQ", word0, 256, 256, 1, 1, 1, 1, 0x20, 0)
    s.submit(abi.OP_HEAP_TEXTURE_SIZE_AND_ALIGN, struct.pack("<IQQI", 1, 0x40, 16, len(ser)) + ser, fault=True)
    s.fault_state("after the walk")
    s.events("end")
    return s


@scenario
def smoke_delay():
    """CmdDelay (61) never signals its stamp; teardown then hangs."""
    s = Scenario("smoke_delay", watchdog_s=8, doc=smoke_delay.__doc__)
    f = s.bring_up()
    a = f.cmd(abi.OP_NOP)
    f.cmd(abi.OP_DELAY, struct.pack("<Q", 1000))
    c = f.cmd(abi.OP_NOP)
    f.doorbell()
    f.wait_stamp(a)
    f.wait_stamp(c, timeout_ms=500, label="stamp 3, behind the delay")
    s.events("after")
    s.r(abi.R_FIFO_READ)
    s.r(abi.R_FAULT_MASK)
    return s
