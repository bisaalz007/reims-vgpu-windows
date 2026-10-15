"""The EFI framebuffer's guest side (docs/testing.md, section 6.22): the
registers 0x1200-0x1228 (reims-vgpu-core's device::mmio and display::efi)
and a frame in guest memory."""
import struct

from . import abi
from .abi import PAGE

EFI_REGS = (abi.R_EFI_ENABLE, abi.R_EFI_MODE_COUNT, abi.R_EFI_MODE_SELECTED, abi.R_EFI_START, abi.R_EFI_LENGTH,
            abi.R_EFI_DEPTH, abi.R_EFI_MODE, abi.R_EFI_STRIDE_ALIGN, abi.R_EFI_STRIDE)


def frame(w, h, stride, seed=0):
    """BGRA pixels, (x, y) = {x + seed, y, (x ^ y) + seed, 0xff}; the
    padding past 4w is 0x55."""
    rows = []
    for y in range(h):
        row = b"".join(bytes([(x + seed) & 0xFF, y & 0xFF, ((x ^ y) + seed) & 0xFF, 0xFF]) for x in range(w))
        rows.append(row + b"\x55" * (stride - len(row)))
    return b"".join(rows)


class EFI:
    """An EFI framebuffer at `pages` RAM pages from `page`."""

    def __init__(self, s, pages=1):
        self.s = s
        self.page = s.alloc_page(pages)
        self.pages = pages

    def gpa(self, off=0):
        return self.s.pa(self.page, off)

    def write_frame(self, w, h, stride, seed=0):
        self.s.mem(self.gpa(), frame(w, h, stride, seed))

    def regs(self, label=None):
        for r in EFI_REGS:
            self.s.r(r, label)

    def setup(self, w=64, h=32, stride=256, depth=32, length=None, enable=True, start=None):
        """The measured sequence: enable, mode, depth, stride, start, length
        (section 6.22)."""
        s = self.s
        if enable:
            s.w(abi.R_EFI_ENABLE, 1)
        s.w(abi.R_EFI_MODE, w | h << 16)
        s.w(abi.R_EFI_DEPTH, depth)
        s.w(abi.R_EFI_STRIDE, stride)
        s.w(abi.R_EFI_START, s.pfn(self.page) if start is None else start)
        s.w(abi.R_EFI_LENGTH, self.pages * PAGE if length is None else length)
