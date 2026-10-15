"""The device's main BAR: every register's read and write, the index
arithmetic that aliases and wraps offsets, the read-to-clear registers, and
the root FIFO's enable register (docs/testing.md, sections 6.2 and 6.5).

DeviceCore::mmio_read and DeviceCore::mmio_write both index with
mmio::register_index, (uint16_t)((offset - 0x1000) / 4) in 64-bit arithmetic:
an offset aliases every 0x40000 bytes, rounds down to a multiple of 4, and
offsets below 0x1000 wrap to indices 0xfc00-0xffff, which nothing handles.
"""
from pgtest import abi
from pgtest.registry import scenario
from pgtest.scenario import Scenario

# The registers a read handles, and the ones a write does not: read-only, or
# not a register at all.
READ_ONLY = [abi.R_FIFO_READ, abi.R_IRQ_DISPLAY, abi.R_IRQ_STAMP, abi.R_FAULT_MASK, abi.R_FAULT_OFFSET,
             abi.R_EFI_MODE_COUNT, abi.R_EFI_MODE_SELECTED, abi.R_EFI_STRIDE_ALIGN, abi.R_PORT_COUNT,
             abi.R_PORT_TYPES]
UNHANDLED = [0x0, 0x4, 0x800, 0xFFC, 0x1040, 0x1044, 0x10FC, 0x1100, 0x11FC, 0x1230, 0x1238, 0x123C, 0x1240,
             0x2000, 0x3FFC, 0x4000, 0x3F000]
# An offset whose index wraps: 2^64 - 0x40000 + off has the same uint16_t
# index as off.
WRAP = 1 << 64


def wrapped(off, k=1):
    """off - k * 0x40000, as a 64-bit offset: the same register."""
    return (off - k * 0x40000) % WRAP


@scenario
def device_mmio_fresh():
    """A fresh device's registers: 0x122c reads 1 (count != 0 before any
    version) and every other offset in 0-0x4000 reads 0; 0x122c's
    aliases (0x4122c, unaligned 0x122d-0x122f, and a 64-bit offset that
    wraps) read 1 too, while offsets below 0x1000 wrap to unhandled
    indices and read 0."""
    s = Scenario("device_mmio_fresh", doc=device_mmio_fresh.__doc__)
    s.sweep(0, 0x4000, label="fresh, 0-0x4000")
    s.sweep(0x40000, 0x41400, label="fresh, 0x40000-0x41400: aliases of 0x0-0x1400")
    for off in (0x122D, 0x122E, 0x122F, 0x4122C, 0x8122C, 0x4122F, wrapped(0x122C), wrapped(0x122C, 0x1000),
                0x1000 + 0x40000 * 0x10000 + 0x22C, 0xFFFFFFFFFFFFFFFC, 0xFFFFFFFFFFFFF000, 0x1, 0x3, 0xFFF):
        s.r(off)
    return s


@scenario
def device_mmio_running():
    """Every readable register with a root FIFO running after one NOP, read
    at its offset, at its alias 0x40000 up, unaligned (rounded down), and
    through a 64-bit wrap; 0x1018 is read-to-clear, so only the first sweep
    sees the stamp bit. Then every read-only register and unhandled offset
    is written with 0xffffffff and a sweep shows nothing moved (the device
    ignores them)."""
    s = Scenario("device_mmio_running", doc=device_mmio_running.__doc__)
    s.bring_up()
    s.submit(abi.OP_NOP)
    s.events("one NOP")
    s.sweep(0x1000, 0x1300, label="running")
    s.sweep(0x41000, 0x41300, label="running, the alias 0x40000 up; 0x1018 was cleared")
    for off in (0x1001, 0x1002, 0x1003, 0x100D, 0x100E, 0x100F, 0x101D, 0x101F, 0x1035, 0x1037, 0x103D, 0x103F,
                wrapped(abi.R_ROOT_PAGE), wrapped(abi.R_VERSION, 7), wrapped(abi.R_FAULT_OFFSET),
                wrapped(abi.R_FIFO_READ)):
        s.r(off)
    for off in READ_ONLY + UNHANDLED + [abi.R_TRACE_MARKER]:
        s.w(off, 0xFFFFFFFF)
    # EFI mode select without EFI goes nowhere; a display doorbell on a
    # port with no guest display, or past the port count, is caught and
    # logged.
    for off, v in ((abi.R_EFI_MODE_SELECT, 0), (abi.R_EFI_MODE_SELECT, 0xFFFFFFFF), (abi.R_DISPLAY_DOORBELL, 0),
                   (abi.R_DISPLAY_DOORBELL, 1), (abi.R_DISPLAY_DOORBELL, 0xFFFFFFFF)):
        s.w(off, v)
    s.sweep(0x1000, 0x1300, label="after writes to read-only and unhandled offsets")
    s.sweep(0, 0x1000, label="below 0x1000, after the writes")
    s.sweep(0x1300, 0x4000, label="0x1300-0x4000, after the writes")
    s.diff("guest memory after the writes")
    st = s.submit(abi.OP_NOP)
    s.events("a NOP after the writes")
    s.r(abi.R_FIFO_READ)
    s.r(abi.R_IRQ_STAMP)
    return s


@scenario
def device_mmio_aliased_writes():
    """Writes land on the register their uint16_t index names: the whole
    bring-up through the alias 0x40000 up, unaligned offsets and 64-bit
    wraps, then a NOP rung through an alias."""
    s = Scenario("device_mmio_aliased_writes", doc=device_mmio_aliased_writes.__doc__)
    s.w(abi.R_VERSION + 0x40000, 0x3E)
    s.r(abi.R_VERSION, "0x41034 = 0x3e")
    s.w(abi.R_VERSION + 2, 0x2B)
    s.r(abi.R_VERSION, "0x1036 = 0x2b")
    s.w(wrapped(abi.R_VERSION + 3, 3), 0x7F)
    s.r(abi.R_VERSION, "a wrapped 0x1037 = 0x7f")
    s.w(0x1000 + 0x40000 * 0x10000 + 0x34, abi.MAX_VERSION)
    s.r(abi.R_VERSION, "0x1000 + 0x40000 * 0x10000 + 0x34 = 0xf5")
    s.w(abi.R_ROOT_PAGE + 0x80000, s.pfn(s.root_page))
    s.r(abi.R_ROOT_PAGE)
    from pgtest.fifo import RootFIFO
    f = RootFIFO(s, 2, 0x400)
    s.fifo = f
    s.fifos[0] = f
    s.w(abi.R_FIFO_BASE + 0x40001, s.pfn(f.page))
    s.w(wrapped(abi.R_FIFO_LENGTH), f.length)
    s.w(abi.R_FIFO_START + 0x40003, f.start)
    s.w(abi.R_FIFO_ENABLE + 0xC0000, 1)
    s.r(abi.R_FIFO_ENABLE)
    st = f.cmd(abi.OP_NOP)
    s.w(abi.R_FIFO_WRITTEN + 0x40002, f.written)
    f.wait_stamp(st)
    s.events("a NOP rung through 0x4100a")
    s.r(abi.R_FIFO_READ + 0x40000)
    s.r(abi.R_IRQ_STAMP + 0x40001)
    s.r(abi.R_IRQ_STAMP)
    s.diff("after")
    return s


@scenario
def device_mmio_enable_register():
    """0x1000: a write equal to the enabled flag does nothing, 0 tears down,
    any nonzero value enables and reads back as 1; refused without a root
    page. Registers across disable and re-enable, and a NOP on each enabled
    FIFO (the ring restarts at the start offset)."""
    s = Scenario("device_mmio_enable_register", doc=device_mmio_enable_register.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    from pgtest.fifo import RootFIFO
    f = RootFIFO(s, 2, 0x400)
    s.fifo = f
    s.fifos[0] = f
    s.w(abi.R_FIFO_ENABLE, 0)
    s.r(abi.R_FIFO_ENABLE, "0 while disabled")
    f.enable(value=2)
    s.r(abi.R_FIFO_ENABLE, "2 without a root page: refused")
    s.w(abi.R_ROOT_PAGE, s.pfn(s.root_page))
    s.w(abi.R_FIFO_ENABLE, 2)
    s.r(abi.R_FIFO_ENABLE, "2 with a root page: enabled")
    s.w(abi.R_FIFO_ENABLE, 1)
    s.r(abi.R_FIFO_ENABLE, "1 while enabled: nothing")
    s.submit(abi.OP_NOP)
    s.events("a NOP")
    s.sweep(0x1000, 0x1040, label="enabled")
    s.w(abi.R_FIFO_ENABLE, 0)
    s.sweep(0x1000, 0x1040, label="disabled")
    s.w(abi.R_FIFO_ENABLE, 0)
    s.r(abi.R_FIFO_ENABLE, "0 again")
    s.w(abi.R_FIFO_ENABLE, 0xFFFFFFFF)
    s.r(abi.R_FIFO_ENABLE, "0xffffffff: enabled")
    f.written = 0
    s.submit(abi.OP_NOP)
    s.events("a NOP on the re-enabled FIFO")
    s.sweep(0x1000, 0x1040, label="re-enabled")
    s.diff("after")
    return s


@scenario
def device_mmio_enable_other_value_dies():
    """Writing a nonzero value other than 1 to 0x1000 while the root FIFO is
    enabled calls DeviceCore::enable_fifo, which asserts that the root FIFO is
    not enabled: the host dies."""
    out = []
    for v in (2, 0x80000000):
        name = "device_mmio_enable_other_value_dies_%x" % v
        s = Scenario(name, doc=device_mmio_enable_other_value_dies.__doc__)
        s.bring_up()
        s.submit(abi.OP_NOP)
        s.events("running")
        s.w(abi.R_FIFO_ENABLE, v)
        out.append(s)
    return out


@scenario
def device_mmio_trace_marker():
    """0x1038, the trace marker, only emits a kdebug event when Metal tracing
    is on: no register or memory changes."""
    s = Scenario("device_mmio_trace_marker", doc=device_mmio_trace_marker.__doc__)
    s.bring_up()
    for v in (0, 1, 0xDEADBEEF, 0xFFFFFFFF):
        s.w(abi.R_TRACE_MARKER, v)
    s.sweep(0x1000, 0x1300, label="after trace markers")
    s.submit(abi.OP_NOP)
    s.events("a NOP")
    s.diff("after")
    return s
