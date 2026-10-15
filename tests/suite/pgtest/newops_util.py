"""Builders for the scenarios mem_seq, held encodes and sorted block calls
unlock (scenarios/newops.py).
"""
import struct

from . import abi
from .abi import PAGE
from .fifo import RootFIFO

# The root FIFO's largest length: the root FIFO's creation refuses more than
# 0x20000, and a length that is not a multiple of the page (reims-vgpu-core's
# device::root::ROOT_FIFO_MAX_LENGTH).
ROOT_FIFO_MAX = 0x20000


def bring_up_big(s, start=0x400):
    """bring_up with the largest root FIFO the device takes: 8 pages, a ring
    of 0x20000 - start bytes (0x1FC00 = 4064 commands of 32 bytes)."""
    return s.bring_up(fifo_pages=ROOT_FIFO_MAX // PAGE, start=start)


def cmd_seq(s, fifo, count, template, fields=(), stamp_off=8):
    """count copies of a fixed-size command on fifo, in one or two mem_seq
    ops (split where the ring wraps; the command size must divide the
    ring's, so none straddles it). The header's stamp at stamp_off steps by
    one from the FIFO's next stamp; fields are (off, size, start, step) as
    mem_seq takes them, start meaning copy 0's value. Advances the written
    count and the stamp; returns the last stamp."""
    size = len(template)
    ring = fifo.ring_size()
    assert ring % size == 0, "commands would straddle the ring's end"
    assert fifo.written % size == 0, "the ring is not at a command boundary"
    first = fifo.stamp + 1
    done = 0
    while done < count:
        off = fifo.written % ring
        n = min(count - done, (ring - off) // size)
        fl = [(stamp_off, 4, first + done, 1)]
        fl += [(o, sz, a + done * st, st) for o, sz, a, st in fields]
        s.mem_seq(fifo.base_gpa() + fifo.start + off, n, size, template, fields=fl)
        fifo.written += n * size
        done += n
    fifo.stamp = first + count - 1
    return fifo.stamp


class AliasedTask:
    """A task whose every page is one guest RAM page: a depth-2 page table
    (4096 u32 entries a level, bit 31 of a leaf writable) of `tables`
    second-level tables, each one mem_seq of 4096 identical leaves. The task
    spans tables * 4096 pages, and costs tables + 3 RAM pages."""

    def __init__(self, s, tid=1, tables=15):
        self.s = s
        self.tid = tid
        self.root = s.alloc_page()
        self.l1 = s.alloc_page()
        self.data = s.alloc_page()
        self.l2 = [s.alloc_page() for _ in range(tables)]
        self.npages = tables * 4096
        self.length = self.npages * PAGE
        leaf = struct.pack("<I", s.pfn(self.data) | abi.PTE_WRITABLE)
        for t in self.l2:
            s.mem_seq(s.pa(t), 4096, 4, leaf)
        s.u32(s.pa(self.l1), *[s.pfn(t) for t in self.l2])
        # The task root (guest::records::TaskRoot2): table root page, depth 2.
        s.mem(s.pa(self.root), abi.TASK_ROOT.pack(s.pfn(self.l1), 2, 0, 0, 0, 0, 0))

    def define(self, **kw):
        """DefineTask2 (56), padded to 32 bytes (the
        header's length says 32; the command reads its 16) so that the
        32-byte commands after it keep to the ring's 32-byte grid."""
        kw.setdefault("length", 32)
        kw.setdefault("pad_to", 32)
        return self.s.submit(abi.OP_DEFINE_TASK2, struct.pack(
            "<IQI", self.tid << 1, self.length, self.s.pfn(self.root)), **kw)

    def map(self, vpn, pages=1, **kw):
        """MapMemory2 (57): {u32 taskID, u64 offset, u64 length}."""
        return self.s.submit(abi.OP_MAP_MEMORY2, struct.pack(
            "<IQQ", self.tid, vpn * PAGE, pages * PAGE), **kw)

    def unmap(self, vpn, pages=1, **kw):
        """UnmapMemory (34): {u32 taskID, u64 offset, u64 length}."""
        return self.s.submit(abi.OP_UNMAP_MEMORY, struct.pack(
            "<IQQ", self.tid, vpn * PAGE, pages * PAGE), **kw)

    def map_seq(self, fifo, count, vpn0, vpn_step, pages=1):
        """count MapMemory2 commands of `pages` pages each, at VA pages vpn0,
        vpn0 + vpn_step, ...: 32 bytes each, written with cmd_seq."""
        template = abi.HDR.pack(abi.OP_MAP_MEMORY2, 0, 32, 0) + struct.pack(
            "<IQQ", self.tid, 0, pages * PAGE)
        return cmd_seq(self.s, fifo, count, template,
                       fields=[(16, 8, vpn0 * PAGE, vpn_step * PAGE)])
