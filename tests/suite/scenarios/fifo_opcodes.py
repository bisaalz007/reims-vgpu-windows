"""Every opcode's framing: the payload size check of each of the 38 live
handlers (short, exact, long), the 15 deprecated ids, invalid ids, the
feature gates on both sides of their version and with no version written,
Debug, NOP and Delay (docs/testing.md, section 6.8).

The class of each outcome is locked: F, the handler's failure faults the
stamp; S, the display commands 7, 8, 10 and 12 log and signal anyway; A, the
task's completion handler signals status 1 and faults any other at the
latched offset. With no task or display, exact payloads fault in the later
validation they reach (the task or display lookup), which is the point: the
size check passed, and the stamp's class is the handler's.
"""
import struct

from pgtest import abi
from pgtest.fifo_util import probe, track_stamps
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import Task

F, S = "fault", "signal"

# The payload each handler checks, in bytes (packed to 4), and the class of
# an exact payload of zeros with no task and no display. Variable tails (10,
# 40, 52, 53, 55, 62-64, 67, 69) are checked in the tail scenarios below.
SIZES = {
    0: (1, S),    # Debug: >= 1 byte, the last NUL-terminated
    1: (8, F),    # DisplaySetSharedStatePage: page 0 on a free port, "not in use"
    2: (8, F),    # DisplayAck: no guest display
    4: (44, F),   # DisplayCursorGlyph
    5: (8, F),    # DisplayCursorShow: {u32, BOOL} packed to 8
    6: (12, F),   # DisplayTransaction2_DEPRECATED
    7: (36, S),   # DisplayTransaction3
    8: (40, S),   # DisplaySwapMapping
    9: (8, F),    # DisplaySleepState
    10: (8, S),   # DisplaySetProperties, count 0
    11: (20, F),  # DisplayCompositorParameters: u8 at 16, packed to 20
    12: (24, S),  # DisplaySetGuestICCProfile
    30: (0, S),   # NOP
    32: (4, F),   # DeleteTask: no task 0, DeviceCore::delete_task_id: false
    34: (20, F),  # UnmapMemory
    37: (8, F),   # DeleteResource
    40: (12, F),  # DeleteObject: length 0 needs >= 4 bytes, so 12 passes both
    48: (4, F),   # DefineChildFIFO: id 0
    49: (4, F),   # DeleteChildFIFO: id 0
    51: (12, F),  # SetObjectList
    52: (8, F),   # InvalidateResources, count 0
    53: (8, F),   # SynchronizeResources, count 0
    54: (8, F),   # DeleteIOSurfaceBacking2: no kernel task
    55: (12, F),  # ExecIndirect2: taskID 0
    56: (16, F),  # DefineTask2: length 0, "Failed to create host task"
    57: (20, F),  # MapMemory2
    58: (12, F),  # GetDeviceInfo: destination page 0 does not map
    59: (24, F),  # GetComputeInfo: taskID 0
    60: (8, F),   # ReplacePhysical
    62: (8, F),   # SynchronizeAndDiscardResources
    63: (8, F),   # DiscardResources
    64: (24, F),  # HeapTextureSizeAndAlign: taskID 0
    65: (4, F),   # DeleteSharedTextureBacking: no kernel task
    66: (8, F),   # ResetRasterizationRateMap: taskID 0
    67: (12, F),  # ExecIndirect3: taskID 0
    68: (28, F),  # SetObjectAndPlacementList
    69: (12, F),  # InfoIndirect: taskID 0
}
SIGNAL_ANYWAY = (7, 8, 10, 12)  # display commands that log and signal anyway
DISPLAY_OPS = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
TASK_OPS = [op for op in SIZES if op >= 30]


def sweep(s, ops):
    """Short (size - 1), exact and long (size + 4) payloads of zeros."""
    for op in ops:
        size, cls = SIZES[op]
        if size:
            # A short payload throws at the size check; the signal-anyway
            # class catches that too.
            probe(s, op, bytes(size - 1), S if op in SIGNAL_ANYWAY else F, label="op %d short" % op)
        probe(s, op, bytes(size), cls, label="op %d exact" % op)
        probe(s, op, bytes(size + 4), cls, label="op %d long" % op)


@scenario
def fifo_op_sizes_display():
    """Display opcodes 0-12 with no display: a short payload faults every
    one but the signal-anyway class (7, 8, 10, 12), which logs and signals;
    exact and long payloads reach the port lookup and fault or signal by the
    same classes. Debug (0) needs one byte."""
    s = Scenario("fifo_op_sizes_display", doc=fifo_op_sizes_display.__doc__)
    s.bring_up()
    sweep(s, DISPLAY_OPS)
    s.events("display sweep")
    s.r(abi.R_FIFO_READ)
    return s


@scenario
def fifo_op_sizes_task():
    """The 26 live opcodes from 30 on (Delay aside), with no task defined:
    short payloads fault at the size check; exact and long payloads of
    zeros fault at the task lookup (GetDeviceInfo at mapping page 0), and
    NOP signals whatever its payload."""
    s = Scenario("fifo_op_sizes_task", doc=fifo_op_sizes_task.__doc__)
    s.bring_up()
    sweep(s, TASK_OPS)
    s.events("task sweep")
    s.r(abi.R_FIFO_READ)
    return s


@scenario
def fifo_op_deprecated_invalid():
    """The 15 deprecated ids fault through -CmdDeprecated: whatever their
    payload; ids with no case, 13-29, 70, 99 and 0xffff, log and fault in
    the switch's default. Each is resumed with 0x1024, which publishes
    its stamp."""
    s = Scenario("fifo_op_deprecated_invalid", doc=fifo_op_deprecated_invalid.__doc__)
    s.bring_up()
    for op in abi.DEPRECATED_OPCODES:
        probe(s, op, b"", F, label="deprecated %d, no payload" % op)
        probe(s, op, bytes(range(1, 17)), F, label="deprecated %d, 16 bytes" % op)
    for op in list(range(13, 30)) + [70, 99, 0xFFFF]:
        probe(s, op, b"", F, label="invalid %d" % op)
    probe(s, 70, bytes(64), F, label="invalid 70, 64 bytes")
    s.events("deprecated and invalid")
    s.r(abi.R_FIFO_READ)
    return s


def heap_query(tid, reply_va):
    """HeapTextureSizeAndAlign (64): taskID@0 replyVO u64@4 replyLen u64@12
    payloadLen u32@20, then a serialized texture descriptor, as smoke_task
    sends it (2D RGBA8Unorm 256x256, private)."""
    word0 = 2 | (1 << 8) | (70 << 16)
    ser = struct.pack("<II", 22, 0) + struct.pack("<IIIIHHHHQ", word0, 256, 256, 1, 1, 1, 1, 0x20, 0)
    return struct.pack("<IQQI", tid, reply_va, 16, len(ser)) + ser


# The gated opcodes: (opcode, the lowest version that sets the feature it
# checks, a payload that passes every check but the gate with task 1
# defined, and the class at the gate's version).
GATES = [
    (8, 10, bytes(40), S),                        # displayMapperSurface
    (9, 21, bytes(8), F),                         # displaySleep
    (10, 40, bytes(8), S),                        # displaySetProperties
    (11, 108, bytes(20), F),                      # supportsDisplayCompositorParameters
    (12, 208, bytes(24), S),                      # supportsDisplaySetGuestICCProfile
    (62, 10, struct.pack("<II", 1, 0), S),        # synchronizeAndDiscardResources
    (63, 11, struct.pack("<II", 1, 0), S),        # discardResources
    (64, 43, heap_query(1, 0x40), S),             # metalHeaps
    (66, 106, struct.pack("<II", 1, 0), F),       # supportsVariableRasterizationRateMaps
    (67, 109, struct.pack("<III", 1, 0, 0), F),   # supportsCmdExecIndirect3
    (68, 207, struct.pack("<IQQII", 1, 0, 0, 0, 0), S),  # supportsArgumentBuffers
    (69, 235, struct.pack("<III", 1, 0, 0), F),   # supportsInfoIndirect
]
BELOW = {10: 6, 11: 10, 21: 20, 40: 31, 43: 42, 106: 105, 108: 107, 109: 108,
         207: 206, 208: 207, 235: 233}  # the valid version just below each gate


@scenario
def fifo_op_gates():
    """Each feature-gated opcode just below its version and at it, with task
    1 defined and mapped (docs/testing.md, section 6.3): below, the handler
    throws "unsupported" and the stamp faults (F) or signals anyway (8, 10,
    12); at the version the same payload goes on to the task or the display
    lookup. 0x1034 is written between commands; the FIFO keeps running."""
    s = Scenario("fifo_op_gates", doc=fifo_op_gates.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=8)
    t.define()
    t.map()
    for op, gate, payload, cls in GATES:
        s.w(abi.R_VERSION, BELOW[gate])
        s.r(abi.R_VERSION)
        probe(s, op, payload, S if op in SIGNAL_ANYWAY else F, label="op %d at version %d" % (op, BELOW[gate]))
        s.w(abi.R_VERSION, gate)
        s.r(abi.R_VERSION)
        probe(s, op, payload, cls, label="op %d at version %d" % (op, gate))
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    t.delete()
    s.events("gates")
    return s


@scenario
def fifo_op_no_version():
    """No binary version written: the device's features are zero
    (docs/testing.md, section 6.3). NOP and GetDeviceInfo still run; every
    gated opcode is refused whatever its payload, by its class; child FIFO 1
    cannot be defined (maxFIFOCount 0: DeviceCore::create_fifo raises); a
    barrier on the root's own index 0 is out of range and faults; the stamp
    index check `stamp > maxFIFOCount` still lets index 0 publish."""
    s = Scenario("fifo_op_no_version", doc=fifo_op_no_version.__doc__)
    s.bring_up(version=None)
    s.r(abi.R_VERSION)
    probe(s, abi.OP_NOP, label="NOP")
    info = s.alloc_page()
    probe(s, abi.OP_GET_DEVICE_INFO, struct.pack("<III", 2, 1, s.pfn(info)), S, label="device info, key 1")
    for op, gate, payload, cls in GATES:
        probe(s, op, payload, S if op in SIGNAL_ANYWAY else F, label="op %d with no version" % op)
    probe(s, abi.OP_DEFINE_CHILD_FIFO, struct.pack("<I", 1), F, label="define child 1")
    probe(s, abi.OP_NOP, b"", F, label="barrier on index 0", barriers=[(0, 0)])
    s.events("no version")
    return s


@scenario
def fifo_op_debug():
    """Debug (0) needs one payload byte and NUL-terminates its last byte in the
    host's copy only: guest memory is unchanged. Payloads of 1, 2 and 255 bytes
    signal; the ring bytes are not rewritten."""
    s = Scenario("fifo_op_debug", doc=fifo_op_debug.__doc__)
    s.bring_up()
    for n in (1, 2, 255):
        probe(s, abi.OP_DEBUG, bytes([0x41 + (i % 26) for i in range(n)]), S, label="debug, %d bytes" % n)
    s.diff("the ring is unchanged; only the stamp slot moved")
    return s


@scenario
def fifo_op_delay():
    """Delay (61) with its 8-byte payload never signals: later stamps on the
    FIFO complete but stay unpublished behind it, with no fault; 0x1024 then
    finds no faulted head and changes nothing. Teardown drains the FIFO and
    hangs: the outcome is the watchdog, in teardown (docs/testing.md,
    section 7.4)."""
    s = Scenario("fifo_op_delay", watchdog_s=8, doc=fifo_op_delay.__doc__)
    f = s.bring_up()
    a = f.cmd(abi.OP_NOP)
    f.cmd(abi.OP_DELAY, struct.pack("<Q", 0))
    c = f.cmd(abi.OP_NOP)
    f.doorbell()
    f.wait_stamp(a)
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=12 + 20 + 12, label="all three read")
    f.wait_stamp(c, timeout_ms=500, label="stamp 3, behind the delay")
    s.fault_state("no fault")
    s.resume_root()
    f.wait_stamp(c, timeout_ms=500, label="stamp 3 after 0x1024")
    s.events("after the delay")
    return s


@scenario
def fifo_op_delay_long():
    """Delay with a 16-byte payload never signals either (the check is >=);
    a 7-byte one faults and resumes normally. Teardown hangs on the long
    one."""
    s = Scenario("fifo_op_delay_long", watchdog_s=8, doc=fifo_op_delay_long.__doc__)
    f = s.bring_up()
    probe(s, abi.OP_DELAY, bytes(7), F, label="delay, 7 bytes")
    d = f.cmd(abi.OP_DELAY, bytes(16))
    f.doorbell()
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written, label="the delay read")
    f.wait_stamp(d, timeout_ms=500, label="the delay's stamp")
    s.events("after the delay")
    return s


@scenario
def fifo_op_delay_disable():
    """Disabling the root FIFO (0x1000 = 0) behind a Delay hangs the MMIO write
    itself: DeviceCore::teardown_fifos drains a stamp that never completes. The
    watchdog fires at that op."""
    s = Scenario("fifo_op_delay_disable", watchdog_s=8, doc=fifo_op_delay_disable.__doc__)
    f = s.bring_up()
    a = f.cmd(abi.OP_NOP)
    f.cmd(abi.OP_DELAY, bytes(8))
    f.doorbell()
    f.wait_stamp(a)
    s.wait(mmio=abi.R_FIFO_READ, mmio_eq=f.written, label="the delay read")
    s.w(abi.R_FIFO_ENABLE, 0)
    s.mark("not reached")
    return s


def u32s(*v):
    return struct.pack("<%dI" % len(v), *v)


@scenario
def fifo_op_tails():
    """The variable-length payloads' second checks, with user task 1 and the
    kernel task 0 defined: each count's records short by one byte fault;
    counts whose record bytes overflow u32 fault ("Overflow in Cmd...",
    os_mul/os_add checks); zero counts and exact tails reach the task, whose
    completion handler signals status 1 (A). Kernel-only commands (54)
    refuse a user task, and user-only ones (55, 59, 64, 66, 67, 69) refuse
    task 0."""
    s = Scenario("fifo_op_tails", doc=fifo_op_tails.__doc__)
    s.bring_up()
    k = Task(s, tid=0, pages=4)
    k.define()
    t = Task(s, tid=1, pages=8)
    t.define()
    t.map()
    info = s.alloc_page()
    cases = [
        # (label, opcode, payload, class)
        ("delete task 7, none", 32, u32s(7), F),
        ("delete task 256", 32, u32s(256), F),
        ("unmap 1 page", 34, struct.pack("<IQQ", 1, 0, 0x4000), S),
        ("map 1 page", 57, struct.pack("<IQQ", 1, 0, 0x4000), S),
        ("delete resource 5", 37, u32s(1, 5), S),
        ("delete object, length 0", 40, u32s(1, 0, 0), S),
        ("delete object, length 9, 8 bytes follow", 40, u32s(1, 0, 9), F),
        ("delete object, length overflows", 40, u32s(1, 0, 0xFFFFFFFC), F),
        ("object list, empty", 51, u32s(1, 0, 0), S),
        ("object list past the task's end", 51, u32s(1, 0x100, 12), F),
        ("invalidate 0", 52, u32s(1, 0), S),
        ("invalidate 1, 7 record bytes", 52, u32s(1, 1) + bytes(7), F),
        ("invalidate 0x1fffffff overflows", 52, u32s(1, 0x1FFFFFFF), F),
        ("synchronize 0", 53, u32s(1, 0), S),
        ("synchronize 1, 3 record bytes", 53, u32s(1, 1) + bytes(3), F),
        ("synchronize 0x3fffffff overflows", 53, u32s(1, 0x3FFFFFFF), F),
        ("synchronize and discard 1, 3 record bytes", 62, u32s(1, 1) + bytes(3), F),
        ("synchronize and discard 0x3fffffff overflows", 62, u32s(1, 0x3FFFFFFF), F),
        ("discard 1, 3 record bytes", 63, u32s(1, 1) + bytes(3), F),
        ("discard 0x3fffffff overflows", 63, u32s(1, 0x3FFFFFFF), F),
        ("delete backing 5, kernel task", 54, u32s(5, 0), F),
        ("delete backing 5, task 1", 54, u32s(5, 1), F),
        ("exec2, nothing", 55, u32s(1, 0, 0), F),
        ("exec2, task 0", 55, u32s(0, 0, 0), F),
        ("exec2, 1 resource, 23 bytes", 55, u32s(1, 1, 0) + bytes(23), F),
        ("exec2, 1 command buffer, 15 bytes", 55, u32s(1, 0, 1) + bytes(15), F),
        ("exec2, resources overflow", 55, u32s(1, 0x0AAAAAAB, 0), F),
        ("exec2, command buffers overflow", 55, u32s(1, 0, 0x10000000), F),
        ("exec2, the sum overflows", 55, u32s(1, 0x06000000, 0x07000000), F),
        ("device info, 1 pair", 58, u32s(2, 1, s.pfn(info)), S),
        ("compute info, pipeline 0", 59, struct.pack("<IIIIQ", 1, 0, 0, 0, 0), F),
        ("compute info, task 0", 59, struct.pack("<IIIIQ", 0, 0, 0, 0, 0), F),
        ("replace physical 5", 60, u32s(1, 5), S),
        ("heap texture size", 64, heap_query(1, 0x40), S),
        ("heap texture size, task 0", 64, heap_query(0, 0x40), F),
        ("heap texture size, 4-byte payload absent", 64, struct.pack("<IQQI", 1, 0x40, 16, 4), F),
        ("heap texture size, payload length overflows", 64, struct.pack("<IQQI", 1, 0x40, 16, 0xFFFFFFF0), F),
        ("delete shared texture backing 5", 65, u32s(5), S),
        ("reset rate map, task 0", 66, u32s(0, 0), F),
        ("exec3, nothing", 67, u32s(1, 0, 0), F),
        ("exec3, task 0", 67, u32s(0, 0, 0), F),
        ("exec3, 1 resource, 7 bytes", 67, u32s(1, 1, 0) + bytes(7), F),
        ("exec3, resources overflow", 67, u32s(1, 0x20000000, 0), F),
        ("exec3, command buffers overflow", 67, u32s(1, 0, 0x10000000), F),
        ("exec3, the sum overflows", 67, u32s(1, 0x10000000, 0x08000000), F),
        ("placement lists, empty", 68, struct.pack("<IQQII", 1, 0, 0, 0, 0), S),
        ("info indirect, nothing", 69, u32s(1, 0, 0), F),
        ("info indirect, task 0", 69, u32s(0, 0, 0), F),
        ("info indirect, 1 command buffer, 15 bytes", 69, u32s(1, 0, 1) + bytes(15), F),
        ("info indirect, the sum overflows", 69, u32s(1, 0x10000000, 0x08000000), F),
        ("delete task 1", 32, u32s(1), S),
        ("delete task 0", 32, u32s(0), S),
    ]
    for label, op, payload, cls in cases:
        probe(s, op, payload, cls, label=label)
    s.events("tails")
    return s


@scenario
def fifo_op_display_nub():
    """The display opcodes' framing past the port lookup, with a shared state
    page set on port 0 (opcode 1) and no host display attached. Only the
    stamp's class is locked here, not the shared page (the display scenarios
    lock that). DisplaySetProperties walks its property records: a record cut
    short, a word count whose byte count overflows u32, values cut short, and
    an unknown property each end the walk, and the command signals anyway; a
    full colour matrix (property 0, 12 words) and an empty list signal too.
    Kernel-only commands name task 1 and fault; ICC profile lengths over the
    mapped length signal anyway."""
    s = Scenario("fifo_op_display_nub", doc=fifo_op_display_nub.__doc__)
    s.bring_up()
    shared = s.alloc_page()
    u = lambda *v: struct.pack("<%dI" % len(v), *v)
    probe(s, 1, u(0, s.pfn(shared)), S, label="set the shared page")
    probe(s, 1, u(0, s.pfn(shared)), F, label="set it again: in use")
    probe(s, 1, u(1, s.pfn(shared)), F, label="port 1: out of range")
    probe(s, 2, u(0, 0), F, label="ack with nothing pending")
    probe(s, 4, u(0, 1) + bytes(36), F, label="cursor glyph, task 1")
    probe(s, 5, u(0, 1), S, label="cursor show")
    probe(s, 6, u(0, 0, 1), F, label="transaction2, task 1")
    probe(s, 7, u(0, 1) + bytes(28), S, label="transaction3, task 1")
    probe(s, 8, u(0, 1) + bytes(32), S, label="swap mapping, task 1")
    probe(s, 9, u(0, 0), S, label="sleep state awake")
    prop = lambda p, words: u(p, len(words)) + u(*words)
    for label, body in (
            ("no properties", u(0, 0)),
            ("a colour matrix", u(0, 1) + prop(0, [0x3F800000] * 12)),
            ("a record cut short", u(0, 1) + u(0)),
            ("a word count overflowing", u(0, 1) + u(0, 0x40000000)),
            ("values cut short", u(0, 1) + u(0, 12) + u(*[0] * 11)),
            ("an unknown property", u(0, 1) + prop(7, [1])),
            ("two, the second unknown", u(0, 2) + prop(0, [0] * 12) + prop(9, []))):
        probe(s, 10, body, S, label="set properties: " + label)
    probe(s, 11, u(0, 0) + bytes(12), S, label="compositor parameters")
    probe(s, 12, u(0, 5) + struct.pack("<QQ", 0, 4), S, label="ICC profile longer than mapped")
    probe(s, 12, u(0, 0) + struct.pack("<QQ", 0, 0), S, label="ICC profile, length 0")
    probe(s, 1, u(0, 0), S, label="release the shared page")
    probe(s, 1, u(0, 0), F, label="release it again: not in use")
    s.events("display commands")
    return s


@scenario(setups=("isolate",))
def fifo_op_async_faults():
    """The asynchronous class (A): a task's completion handler faults the
    stamp at the offset latched when the command was read, for any status
    other than 1. Only a service task answers with another status on its
    own, so this runs under isolation only: task 1's page table root is 0,
    its first walk kills its service (docs/testing.md, section 5.5), and
    every later command on it then faults, with 0x103c at its own ring
    offset, while the root FIFO goes on."""
    s = Scenario("fifo_op_async_faults", setups=("isolate",), doc=fifo_op_async_faults.__doc__)
    s.bring_up()
    t = Task(s, tid=1, pages=4)
    t.write_root(pt_root=0)
    t.define()
    probe(s, 64, heap_query(1, 0x40), F, label="the walk that kills the service")
    cases = [
        ("unmap", 34, struct.pack("<IQQ", 1, 0, 0x4000)),
        ("delete resource", 37, u32s(1, 5)),
        ("delete object", 40, u32s(1, 1000, 8, 5)),
        ("object list", 51, u32s(1, 0, 0)),
        ("invalidate", 52, u32s(1, 0)),
        ("synchronize", 53, u32s(1, 0)),
        ("exec2", 55, u32s(1, 0, 0)),
        ("compute info", 59, struct.pack("<IIIIQ", 1, 1, 0, 0, 0)),
        ("replace physical", 60, u32s(1, 5)),
        ("synchronize and discard", 62, u32s(1, 0)),
        ("discard", 63, u32s(1, 0)),
        ("heap texture size", 64, heap_query(1, 0x40)),
        ("reset rate map", 66, u32s(1, 1)),
        ("exec3", 67, u32s(1, 0, 0)),
        ("placement lists", 68, struct.pack("<IQQII", 1, 0, 0, 0, 0)),
        ("info indirect", 69, u32s(1, 0, 0)),
    ]
    for label, op, payload in cases:
        probe(s, op, payload, F, label=label + " on the dead task")
    # MapMemory2 on a memory-map task never reaches the service: it still
    # completes.
    probe(s, 57, struct.pack("<IQQ", 1, 0, 0x4000), S, label="map on the dead task")
    probe(s, abi.OP_NOP, expect=S, label="the root goes on")
    s.events("async faults")
    return s


@scenario
def fifo_op_kernel_task():
    """The kernel task (0) with an object list: DeleteIOSurfaceBacking2 (54)
    and DeleteSharedTextureBacking (65) of an unknown reference complete
    with status 1 through the kernel task's handler; without an object list,
    54 throws "Failed to map object list" and faults."""
    s = Scenario("fifo_op_kernel_task", doc=fifo_op_kernel_task.__doc__)
    s.bring_up()
    k = Task(s, tid=0, pages=4)
    k.define()
    k.map()
    probe(s, 54, u32s(5, 0), F, label="delete backing, no object list")
    k.write(0, bytes(12))
    k.set_object_list(0, 12)
    probe(s, 54, u32s(5, 0), S, label="delete backing 5")
    probe(s, 65, u32s(5), S, label="delete shared texture backing 5")
    k.delete()
    s.events("kernel task")
    return s


@scenario
def fifo_op_exec_empty():
    """The three execution opcodes with no command buffers and no resources,
    on a task with an object list: each completes with status 1 and signals;
    InfoIndirect first waits for the FIFO's event to reach the executions'
    count. Without an object list (task 2) each completes with status 2 and
    faults at its offset."""
    s = Scenario("fifo_op_exec_empty", doc=fifo_op_exec_empty.__doc__)
    s.bring_up()
    from pgtest.task import GPUTask
    t = GPUTask(s, tid=1, pages=16)
    t.define()
    t.map()
    t.install()
    t2 = Task(s, tid=2, pages=4)
    t2.define()
    for tid, cls in ((1, S), (2, F)):
        probe(s, 69, u32s(tid, 0, 0), cls, label="info indirect, task %d" % tid)
        probe(s, 55, u32s(tid, 0, 0), cls, label="exec2, task %d" % tid)
        probe(s, 67, u32s(tid, 0, 0), cls, label="exec3, task %d" % tid)
        probe(s, 69, u32s(tid, 0, 0), cls, label="info indirect after executions, task %d" % tid)
    t.delete()
    t2.delete()
    s.events("empty executions")
    return s
