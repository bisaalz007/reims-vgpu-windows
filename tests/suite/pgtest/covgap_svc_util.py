"""Builders for the task and service scenarios of scenarios/covgap_svc.py.

The guest ABI is restated from byte offsets, as task.py and tasks_lib.py
do (docs/testing.md, section 6.10): a one-level page table is one guest
page of 4096 u32 leaves, a leaf's bits 0-30 are the guest page, bit 31
makes it writable, and 0 is a hole (reims-vgpu-core's memory::page_table).
Nothing here is taken from the implementation.
"""
from . import abi

# An exec resource's flags {invalidateHost: bit 0, invalidateGuest: bit 1,
# synchronize: bit 2}: the GPU wrote it, page it out.
SYNC_BACK = abi.RES_INVALIDATE_GUEST | abi.RES_SYNCHRONIZE


def leaf_gpa(t, vpn):
    """Where VA page vpn's leaf is, for a one-level task (Task or XTask)."""
    assert t.depth == 1
    return t.s.pa(t.pt_root, 4 * vpn)


def punch(t, vpn):
    """The guest makes VA page vpn a hole (its leaf 0); no command follows.
    The tables below the root are read again at every walk."""
    t.s.u32(leaf_gpa(t, vpn), 0)


def point(t, vpn, pfn, writable=True):
    """The guest points VA page vpn's leaf at guest page number pfn."""
    t.s.u32(leaf_gpa(t, vpn), pfn | (abi.PTE_WRITABLE if writable else 0))


def last_exec(s, opcode, payload, label, nops=256):
    """A command expected to kill the host after it faults, or before: the
    scenario's last command. Behind nops NOPs in the same doorbell (as
    exec_objects.submit_last), then a wait for a fault-mask value the root
    FIFO never shows (bit 1), so the death lands in that wait every run,
    whether the fault was raised first or not."""
    f = s.fifo
    for _ in range(nops):
        f.cmd(abi.OP_NOP)
    f.cmd(opcode, payload)
    f.doorbell()
    s.wait(mmio=abi.R_FAULT_MASK, mmio_eq=2, timeout_ms=10000, label=label)
