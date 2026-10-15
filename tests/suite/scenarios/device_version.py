"""Binary version negotiation through 0x1034, and the feature gates a guest
can see from registers and simple commands (docs/testing.md, section 6.3).

Setting the binary version: a request with bit 31 set is ignored; 1 and 236 are
stored with bit 31 set; an unknown request above 0xf5 becomes 0xf5, below it
steps down to the nearest valid version, and 0 stays 0; the result is clamped
by the device feature level (0 here: 245) and the features recomputed
(features::features_for_version), tearing EFI down when the version has no
efiDisplay.
"""
import struct

from pgtest import abi
from pgtest.fifo import ChildFIFO
from pgtest.registry import scenario
from pgtest.scenario import Scenario

EXTRA = [0x100, 0x101, 0x1F5, 0x10000, 0x100F5, 0x7FFFFFFF, 0x80000000, 0x800000F5, 0xFFFFFFFF, 0xFFFFFF0B]


@scenario
def device_version_sweep():
    """Every request from 0 to 0x100, then values above it and with bit 31
    set, each followed by reads of 0x1034, 0x122c (the port count: 1 below
    245, with one port) and 0x1234 (connection types, 0 here). The first
    write, 0x80000000, is ignored on a fresh device."""
    s = Scenario("device_version_sweep", doc=device_version_sweep.__doc__)
    s.r(abi.R_VERSION, "fresh")
    s.w(abi.R_VERSION, 0x80000000)
    s.r(abi.R_VERSION, "0x80000000 on a fresh device")
    for v in list(range(0, 0x101)) + EXTRA:
        s.w(abi.R_VERSION, v)
        s.r(abi.R_VERSION)
        s.r(abi.R_PORT_COUNT)
        s.r(abi.R_PORT_TYPES)
    return s


@scenario
def device_version_ports():
    """0x122c with 2 and 8 display ports: the host's count at version 245
    only; below it, and before any version, 1 (count != 0). 0x1234 is
    `& 3` below 245."""
    out = []
    for n in (2, 8):
        name = "device_version_ports_%d" % n
        s = Scenario(name, descriptor={"port_count": n}, doc=device_version_ports.__doc__)
        s.r(abi.R_PORT_COUNT, "fresh")
        s.r(abi.R_PORT_TYPES, "fresh")
        for v in (245, 238, 0, 245, 1, 245, 236, 245, 0x10000, 244, 0x80000000, 0xF5):
            s.w(abi.R_VERSION, v)
            s.r(abi.R_VERSION)
            s.r(abi.R_PORT_COUNT)
            s.r(abi.R_PORT_TYPES)
        out.append(s)
    return out


@scenario
def device_version_efi_gate():
    """EFI (0x1200) is enabled by any nonzero write and reads back what was
    written; a version write whose features lack efiDisplay (below 3: 0, 2;
    the unsupported 1 and 236; and before any version is written nothing
    tears it down) disables it and zeroes the EFI registers. The EFI
    registers are read after each version."""
    s = Scenario("device_version_efi_gate", doc=device_version_efi_gate.__doc__)

    def efi(label):
        for r in (abi.R_EFI_ENABLE, abi.R_EFI_MODE_COUNT, abi.R_EFI_MODE_SELECTED, abi.R_EFI_START,
                  abi.R_EFI_LENGTH, abi.R_EFI_DEPTH, abi.R_EFI_MODE, abi.R_EFI_STRIDE):
            s.r(r, label)

    s.w(abi.R_EFI_ENABLE, 1)
    s.w(abi.R_EFI_START, 0x1234)
    s.w(abi.R_EFI_DEPTH, 32)
    efi("enabled before any version")
    for v in (3, 2, 245, 0, 4, 1, 7, 236, 3, 0x80000000, 0x1000):
        s.w(abi.R_EFI_ENABLE, 7)
        s.w(abi.R_EFI_LENGTH, 0x4000)
        s.w(abi.R_EFI_MODE, 64 | 32 << 16)
        s.w(abi.R_EFI_STRIDE, 256)
        s.w(abi.R_VERSION, v)
        s.r(abi.R_VERSION)
        efi("after version 0x%x" % v)
    s.w(abi.R_EFI_ENABLE, 0)
    efi("disabled")
    return s


@scenario
def device_version_fifo_count():
    """maxFIFOCount as the guest sees it, through DefineChildFIFO (48) of ids
    31 and 32 at each version: 0 before any version (every define faults), 32
    for any written version below 61 including 0 and the unsupported ones
    (features::features_for_version starts from 32 before it matches the
    version), 64 from 61."""
    s = Scenario("device_version_fifo_count", doc=device_version_fifo_count.__doc__)
    s.bring_up(version=None)
    kids = {31: ChildFIFO(s, 31), 32: ChildFIFO(s, 32)}
    # What the device's code allows at each version (the ids below the count
    # are defined, the rest fault); the expected file records what it does.
    for v, count in ((None, 0), (0, 32), (1, 32), (236, 32), (60, 32), (61, 64), (0x80000000, 64), (62, 64),
                     (245, 64), (2, 32)):
        if v is not None:
            s.w(abi.R_VERSION, v)
        s.r(abi.R_VERSION)
        for fid, k in kids.items():
            k.setup_node()
            ok = fid < count
            k.define(fault=not ok, label="define %d at %s" % (fid, v))
            s.r(abi.R_FAULT_MASK)
            if not ok:
                s.r(abi.R_FAULT_OFFSET)
                s.resume_root()
                s.fifo.wait_stamp(s.fifo.stamp)
            else:
                k.delete()
        s.events("version %s" % v)
        s.diff("version %s" % v, gpa=s.pa(s.root_page), length=abi.PAGE)
    return s


@scenario
def device_version_stamp_index_above_count():
    """A child FIFO whose stamp index (40) was valid when it was defined at
    version 245 (64 FIFOs) signals nothing once the guest lowers the version to
    60 (32 FIFOs): DeviceCore::signal_stamp drops indices above maxFIFOCount,
    so the stamp slot stays and no interrupt comes; back at 245 the next stamp
    is published. Index 32 at version 60 is still published: the check is `>`,
    not `>=`."""
    s = Scenario("device_version_stamp_index_above_count", stamp_slots=41,
                 doc=device_version_stamp_index_above_count.__doc__)
    s.bring_up()
    c40 = ChildFIFO(s, 1, stamp_index=40)
    c32 = ChildFIFO(s, 2, stamp_index=32)
    for c in (c40, c32):
        c.setup_node()
        c.define()
    c40.submit(abi.OP_NOP)
    c32.submit(abi.OP_NOP)
    s.events("both children at 245")
    s.r(abi.R_IRQ_STAMP)
    s.w(abi.R_VERSION, 60)
    st = c40.cmd(abi.OP_NOP)
    c40.doorbell()
    s.wait(gpa=c40.node_gpa(4), eq=c40.written, label="child 40's read count")
    s.quiesce()
    c32.submit(abi.OP_NOP)
    s.events("at version 60")
    s.r(abi.R_IRQ_STAMP)
    s.dump(s.fifo.base_gpa(), 0xA4, "stamp slots at 60")
    s.w(abi.R_VERSION, 245)
    c40.submit(abi.OP_NOP)
    s.events("back at 245")
    s.r(abi.R_IRQ_STAMP)
    s.dump(s.fifo.base_gpa(), 0xA4, "stamp slots at 245")
    return s
