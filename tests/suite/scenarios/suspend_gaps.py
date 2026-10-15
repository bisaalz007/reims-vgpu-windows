"""Suspend and resume paths the suspend module leaves open (docs/testing.md,
section 6.24): no state, blobs that set every register, what
rvg_device_did_resume rebuilds from child FIFO nodes it cannot run and from a
task root it cannot read, and the capability check at the binary versions and
host setups where the host's own device info changes.

The capability check compares the blob's device info against the host's own
device info at the blob's binary version.
"""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.fifo import ChildFIFO, RootFIFO
from pgtest.fifo_util import probe
from pgtest.registry import scenario
from pgtest.scenario import Scenario, h
from pgtest.suspend_guest import blob, state

OUTSIDE = 0x3FFFFF  # a guest page number far past guest RAM


def will_resume(s, data, label):
    s.api("willResume", hex=data.hex(), label=label)


def adopt_root(s, f):
    """Makes f the scenario's root FIFO without writing its registers: the
    blob carries them."""
    s.fifo = f
    s.fifos[0] = f
    s._config.setdefault("stamp_gpa", h(f.base_gpa()))


@scenario
def suspend_gap_nil_state():
    """rvg_device_will_resume with no state (a NULL state pointer), which it
    refuses: InvalidContent (2), and the run state is left
    LeavingSuspendEarly. A valid blob afterwards is accepted, and
    rvg_device_did_resume on it brings the device back: a root page, a FIFO
    and a NOP work."""
    s = Scenario("suspend_gap_nil_state", doc=suspend_gap_nil_state.__doc__)
    s.api("willResume", nil=True, label="nil")
    s.r(abi.R_VERSION, "after nil")
    will_resume(s, blob(state()), "a valid blob after nil")
    s.api("didResume")
    s.events("resumed")
    s.r(abi.R_VERSION, "resumed")
    s.bring_up(version=None)
    s.submit(abi.OP_NOP)
    s.events("a NOP")
    return s


def every_key(root, fifo, version, enabled, written, read, fault_offset, types):
    """Every register, by the device-state key that names it, each a value a
    register can show."""
    return [(1, root), (2, version), (3, fifo.s.pfn(fifo.page)), (4, enabled), (5, fifo.length), (6, written),
            (7, read), (8, fifo.start), (9, 0x5), (10, 0x3), (11, 1), (12, 0), (13, 0x123), (14, 0x8000),
            (15, 32), (16, 0x200040), (17, 0x100), (18, 1), (19, fault_offset), (21, types)]


@scenario
def suspend_gap_every_key():
    """A blob setting every register, the root FIFO disabled, at binary
    version 237 (connection types are checked from 238, so key 21 = 2
    passes): after rvg_device_did_resume the registers show what the blob
    restored and what it did not. The pending display and stamp interrupt bits
    (keys 9, 10) read back once (read-to-clear); the fault key 11 is ignored
    (0x102c 0); the EFI registers 0x1210-0x121c and 0x1228 read the blob's
    values though EFI stays off; the connection types are not restored.
    0x1000 = 1 alone then creates the root FIFO from the blob's base, length
    and start, with fresh counts (not keys 6 and 7): a NOP at the ring's start
    runs."""
    s = Scenario("suspend_gap_every_key", doc=suspend_gap_every_key.__doc__)
    f = RootFIFO(s, 2, 0x400)
    kv = every_key(s.pfn(s.root_page), f, 237, 0, 0x30, 0x18, 0x18, 2)
    will_resume(s, blob(kv), "every key, FIFO off")
    s.api("didResume")
    s.events("resumed")
    s.sweep(0x1000, 0x1240, label="registers after didResume")
    s.r(abi.R_IRQ_DISPLAY, "0x1014 again")
    s.r(abi.R_IRQ_STAMP, "0x1018 again")
    adopt_root(s, f)
    s.w(abi.R_FIFO_ENABLE, 1)
    s.r(abi.R_FIFO_ENABLE, "0x1000 = 1 alone")
    s.submit(abi.OP_NOP)
    s.r(abi.R_FIFO_READ, "a NOP")
    s.events("a NOP")
    return s


@scenario
def suspend_gap_restored_fault():
    """A blob whose root FIFO is enabled with read 0x18, written 0x30 and
    fault offset (key 19) 0x18: rvg_device_did_resume recreates the FIFO with
    those counts, 0x103c reads 0x18 and 0x102c 1, but the fault is only a
    register: the queue is fresh, so the two NOPs between read and written
    (stamps 7 and 8) run and publish while 0x102c still reads 1. 0x1024 clears
    it ("Stamp at head isn't faulted"), and the next NOP runs. A fault offset
    of 0, the register a crafted blob leaves unset, restores the same fault at
    offset 0: only 0xffffffff means no fault."""
    out = []
    for kind in ("at_read", "at_0"):
        name = "suspend_gap_restored_fault_" + kind
        s = Scenario(name, doc=suspend_gap_restored_fault.__doc__)
        f = RootFIFO(s, 2, 0x400)
        f.written = 0x18
        f.cmd(abi.OP_NOP, stamp=7)
        f.cmd(abi.OP_NOP, stamp=8)
        kv = every_key(s.pfn(s.root_page), f, 245, 1, 0x30, 0x18, 0x18, 0)
        if kind == "at_0":
            kv = [(k, v) for k, v in kv if k != 19]
        adopt_root(s, f)
        will_resume(s, blob(kv), "FIFO read 0x18, written 0x30, fault " + kind.replace("_", " "))
        s.api("didResume")
        f.wait_stamp(8, label="the NOPs behind the restored fault")
        s.fault_state("restored fault")
        s.r(abi.R_FIFO_READ, "restored fault")
        s.events("resumed")
        s.resume_root()
        s.fault_state("after 0x1024")
        f.stamp = 8
        s.submit(abi.OP_NOP)
        s.r(abi.R_FIFO_READ, "a NOP after 0x1024")
        s.events("end")
        out.append(s)
    return out


def running_root(s, f):
    """Device state for a root FIFO enabled on f with nothing pending and no
    fault: key 19 = 0xffffffff, since a fault offset of 0 resumes the FIFO
    faulted at offset 0 (suspend_gap_restored_fault_at_0)."""
    return [(1, s.pfn(s.root_page)), (2, 0xF5), (3, s.pfn(f.page)), (4, 1), (5, f.length), (6, 0), (7, 0),
            (8, f.start), (18, 1), (19, 0xFFFFFFFF)]


def node_words(s, fid, written=0, read=0, fault=0xFFFFFFFF, stamp_index=None, running=1, plist=0):
    s.mem(s.pa(s.root_page, abi.ROOT_CHILD_NODE + abi.CHILD_NODE_SIZE * (fid - 1)),
          struct.pack("<IIIHHI", written, read, fault, fid if stamp_index is None else stamp_index, running, plist))


@scenario
def suspend_gap_resume_child_nodes():
    """rvg_device_did_resume rebuilds a child FIFO for every node whose running
    word (+0xe) is nonzero, restoring its counts and fault offset from the
    node. Node 1's stamp index 100 is past the 64 FIFOs: the FIFO is not
    created ("Failed to set up FIFO during resume"). Node 2's page list is
    outside guest RAM: the FIFO is created but does not start ("Failed to start
    child fifo"), and waking it does nothing. Node 3 resumes with a NOP between
    its read count and written: it runs. Node 4 resumes with fault offset 0:
    0x102c reads 1 (bit 4 of the mask), yet its pending NOP runs and publishes,
    the queue being fresh; 0x1028 = 4 clears the mask. Node 5 has running 0 and
    a valid ring: it is not a FIFO, and waking it does nothing. The root FIFO,
    from the blob (key 19 = 0xffffffff), runs a NOP."""
    s = Scenario("suspend_gap_resume_child_nodes", stamp_slots=6, doc=suspend_gap_resume_child_nodes.__doc__)
    f = RootFIFO(s, 2, 0x400)
    adopt_root(s, f)
    node_words(s, 1, stamp_index=100, plist=s.pfn(s.alloc_page()))
    node_words(s, 2, plist=OUTSIDE)
    kids = {}
    for fid, fault, running in ((3, 0xFFFFFFFF, 1), (4, 0, 1), (5, 0xFFFFFFFF, 0)):
        c = ChildFIFO(s, fid)
        c.setup_node()
        c.cmd(abi.OP_NOP)
        node_words(s, fid, written=c.written, fault=fault, running=running, plist=s.pfn(c.list_page))
        kids[fid] = c
    kv = running_root(s, f)
    will_resume(s, blob(kv), "resume")
    s.api("didResume")
    kids[3].wait_stamp(1, label="node 3's NOP")
    kids[4].wait_stamp(1, label="node 4's NOP")
    s.fault_state("resumed")
    s.events("resumed")
    s.dump(s.pa(s.root_page, abi.ROOT_CHILD_NODE), 5 * abi.CHILD_NODE_SIZE, "nodes 1-5")
    s.w(abi.R_RESUME_CHILD, 4)
    s.fault_state("0x1028 = 4")
    for fid in (1, 2, 5):
        s.w(abi.R_WAKE_CHILD, fid)
    s.submit(abi.OP_NOP)
    s.events("end")
    s.dump(s.pa(s.root_page, abi.ROOT_CHILD_NODE), 5 * abi.CHILD_NODE_SIZE, "nodes 1-5 at the end")
    s.dump(f.base_gpa(), 24, "stamp slots 0-5")
    return s


@scenario(setups=("map", "isolate"))
def suspend_gap_task_root_unreadable():
    """rvg_device_did_resume with root word 0 (the kernel task) naming a
    page outside guest RAM: the task root read fails (logged) and writes
    nothing, so the task's length reads 0 and its creation is refused, which
    the resume catches and logs: no kernel task. A command that needs it
    (DeleteSharedTextureBacking, 65) then faults, and the root word keeps
    the blob's page. Map and isolate setups only: the blocks setup is not
    locked (there too the length reads 0, and the task is refused before
    create_task is called; docs/testing.md 7.5); task 0 is local in every
    setup, so isolation never sees it."""
    s = Scenario("suspend_gap_task_root_unreadable", setups=("map", "isolate"),
                 doc=suspend_gap_task_root_unreadable.__doc__)
    f = RootFIFO(s, 2, 0x400)
    adopt_root(s, f)
    s.u32(s.pa(s.root_page), OUTSIDE)
    kv = running_root(s, f)
    will_resume(s, blob(kv), "resume")
    s.api("didResume")
    s.events("resumed")
    probe(s, abi.OP_DELETE_SHARED_TEXTURE_BACKING, struct.pack("<I", 1), expect="fault",
          label="65 without a kernel task")
    s.dump(s.pa(s.root_page), 8, "root words 0-1")
    s.events("end")
    return s


# ---- the capability check where the host's device info changes ----

@scenario(host=True)
def suspend_gap_capabilities_versions():
    """device_info::check_capabilities compares the guest's pairs with
    device_info::write_device_info's at the blob's binary version
    (DeviceCore::will_resume applies it with DeviceCore::set_binary_version
    first), so a key is undefined, or a value lower, on one side of its
    feature's version and not on the other: keys 15 and 16 (metalHeaps, 43),
    24-27 (supportsTileShadersImageBlocks, 107) fail as undefined below and
    pass from there (at 0, but key 15, an alignment the host's must not
    exceed, at 0xffffffff); key 12 = 1 fails below dualPlaneTextures (31),
    at 40 (where the override turns it off) and passes at 41; key 28 = 2
    (dynamic attribute stride, 111) fails at 110 and passes at 111; key 17 =
    1 (bufferFromIOSurface, 60) fails at 43 and passes at 60 with a memory
    map (the blocks setup never has it, as the device info shows). At binary
    version 0 (valid only as a request) the host writes no feature key: 14
    fails."""
    s = Scenario("suspend_gap_capabilities_versions", host=True, doc=suspend_gap_capabilities_versions.__doc__)
    cases = []
    for k in (15, 16):
        cases += [(42, k, 0), (43, k, 0xFFFFFFFF if k == 15 else 0)]
    for k in (24, 25, 26, 27):
        cases += [(106, k, 0), (107, k, 0)]
    cases += [(30, 12, 1), (31, 12, 1), (40, 12, 1), (41, 12, 1), (110, 28, 2), (111, 28, 2),
              (43, 17, 1), (60, 17, 1), (0, 14, 0), (0, 12, 0), (0, 1, 0)]
    for version, k, v in cases:
        will_resume(s, blob(state(version=version), [(k, v)]), "version %d, key %d = 0x%x" % (version, k, v))
    s.r(abi.R_VERSION, "after the last")
    return s


@scenario(host=True)
def suspend_gap_capabilities_setups():
    """The capability check's flags where the host setup decides them, with
    enableArgumentBuffers set as QEMU's apple-gfx-mmio sets it: key 33 bit
    0x20 (argument buffers) and key 42 bits 1 and 2 need isolation; key 33
    bits 1 and 2 (large user and kernel tasks) need a memory map. Each case
    is one blob; which pass is the setup's own device info. Every setup runs
    it: the differences are behaviour (docs/testing.md 7.2)."""
    s = Scenario("suspend_gap_capabilities_setups", host=True, descriptor={"enableArgumentBuffers": True},
                 doc=suspend_gap_capabilities_setups.__doc__)
    for k, v in ((33, 0x1), (33, 0x2), (33, 0x20), (33, 0x800), (42, 1), (42, 2), (42, 3), (42, 4)):
        will_resume(s, blob(state(), [(k, v)]), "key %d = 0x%x" % (k, v))
    return s
