"""Helpers for the FIFO scenarios (scenarios/fifo*.py): stamp snapshots,
child FIFO nodes, and command probes (docs/testing.md, sections 6.5-6.8).

Everything here is built on scenario.py and fifo.py; Rust names are
reims-vgpu-core's.
"""
import struct

from . import abi
from .fifo import ChildFIFO

# The node's host-written fields: readCount +4, faultOffset +8, running
# (u16) +0xe.
NODE_READ = 0x4
NODE_FAULT = 0x8
NODE_STAMP_INDEX = 0xC
NODE_RUNNING = 0xE
NODE_PAGE_LIST = 0x10

PROBE_MS = 5000  # how long a probe waits for its stamp or its fault

HEALTHY = 0xFFFFFFFF  # a node's or 0x103c's fault offset when not faulted


def track_stamps(s, slots=2):
    """Makes the runner snapshot `slots` stamp slots, from the root FIFO's
    mapping base (DeviceCore::signal_stamp), inside
    every interrupt it records. Call after bring_up()."""
    s._config["stamp_gpa"] = hex(s.fifo.base_gpa())
    s._config["stamp_slots"] = slots


def stamps(s, n, label=None, first=0):
    """Dumps stamp slots first..first+n-1 (the u32s at the root FIFO's base)."""
    s.dump(s.fifo.base_gpa() + 4 * first, 4 * n, label=label or "stamp slots %d-%d" % (first, first + n - 1))


def node(s, fid, label=None):
    """Dumps child FIFO fid's 0x14-byte node in the root page."""
    s.dump(s.pa(s.root_page, abi.ROOT_CHILD_NODE + abi.CHILD_NODE_SIZE * (fid - 1)), abi.CHILD_NODE_SIZE,
           label=label or "node %d" % fid)


def child(s, fid, define=True, **kw):
    """A child FIFO with its node and page list written; defined with opcode
    48 on the root FIFO unless define=False."""
    c = ChildFIFO(s, fid, **{k: v for k, v in kw.items() if k in ("stamp_index", "pages", "scatter")})
    c.setup_node(**{k: v for k, v in kw.items() if k in ("page_list",)})
    if define:
        c.define()
    s.fifos[fid] = c
    return c


def wait_running(s, c, value=1, timeout_ms=5000):
    """A child FIFO's node +0xe (running) is written when its FIFO thread
    starts and stops (fifo::thread, start and stop)."""
    s.wait(gpa=c.node_gpa(NODE_RUNNING - 2), mask=0xFFFF0000, eq=value << 16, timeout_ms=timeout_ms,
           label="child %d running=%d" % (c.fid, value))


def child_fault(s, c, timeout_ms=5000, label=None):
    """Waits for a child FIFO's node fault offset to leave UINT32_MAX (its
    set_fault_offset), and for 0x102c."""
    s.wait(gpa=c.node_gpa(NODE_FAULT), ne=HEALTHY, timeout_ms=timeout_ms, label=label or "child %d fault" % c.fid)
    s.wait_fault()


def resume_child(s, n):
    """0x1028 = n (DeviceCore::resume_child_fifo)."""
    s.w(abi.R_RESUME_CHILD, n)


def probe(s, opcode, payload=b"", expect="signal", label=None, fifo=None, **kw):
    """One command whose outcome the scenario states: 'signal' waits for its
    stamp; 'fault' waits for the fault, reads 0x102c and 0x103c (or the
    child's node), resumes, and waits for the stamp the resume publishes.
    Returns the stamp."""
    f = fifo or s.fifo
    kw_wait = {"timeout_ms": PROBE_MS}
    lab = label or "op %d, %d payload bytes" % (opcode, len(payload))
    st = f.cmd(opcode, payload, **kw)
    f.doorbell()
    if expect == "signal":
        f.wait_stamp(st, label=lab + ": signalled", **kw_wait)
        s.r(abi.R_FAULT_MASK, label=lab)
    elif expect == "fault":
        if f.fid == 0:
            s.wait_fault(label=lab + ": fault", **kw_wait)
            s.fault_state(label=lab)
            s.resume_root()
        else:
            child_fault(s, f, label=lab + ": fault", **kw_wait)
            s.dump(f.node_gpa(NODE_FAULT), 4, label=lab + ": node fault offset")
            resume_child(s, f.fid)
        f.wait_stamp(st, label=lab + ": resumed", **kw_wait)
    else:
        raise ValueError(expect)
    return st


def pack_u32(*vals):
    return struct.pack("<%dI" % len(vals), *vals)
