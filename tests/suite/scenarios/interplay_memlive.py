"""Guest memory under live GPU work: the guest rewrites page-table leaves
and intermediate tables, clears them and flips their read-only bit, aliases
pages between tasks (a task's page-table pages among them), maps and
unmaps ranges, moves a task's pages from one RAM range to another, and
grows a task, while work that uses the memory is queued, held behind a
barrier on another FIFO, or running (docs/testing.md, sections 6.7,
6.10-6.12).

pgtest/memlive_util.py restates the page-table layout and says what the
scenarios lean on. Unless a docstring says otherwise a scenario runs in all
three setups, and where they differ (section 6.10's table: the blocks
setup maps pages into a host task once and keeps them until an unmap; the
map setup maps the table as it is whenever it makes an object; the service
backs objects with the guest pages themselves) the difference is locked.
"""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.diff_util import Lab, cmp_va, exec_payload
from pgtest.lifetime_util import (B, IG, IH, SY, blit, either, refs_payload, run_either, tex2d, tex_fill, tex_out,
                                  texels)
from pgtest.memlive_util import (Rig, clear_leaf, clone_tree, define_payload, entry_gpa, exec_on, leaf, set_leaf, spare,
                                 table_entries, write_table)
from pgtest.registry import scenario
from pgtest.resources_build import heap_buffer_rec, heap_rec
from pgtest.scenario import Scenario
from pgtest.task import seg

E3 = abi.OP_EXEC_INDIRECT3
BLUE = (0.0, 0.0, 1.0, 1.0)
GREEN = (0.0, 1.0, 0.0, 1.0)


def _scn(fn, suffix="", **kw):
    return Scenario(fn.__name__ + suffix, doc=fn.__doc__, **kw)


# ---- four kinds of object over one page each ----

class Four:
    """A shared buffer (type 1), a private buffer (type 1, flag bit 0), a
    private heap (type 13, one page) with a buffer placed at its start
    (type 14), and a 4x4 RGBA8Unorm texture (type 2), each over a page of
    its own holding its own bytes, and a result buffer (out) of two
    pages."""

    def __init__(self, t):
        self.t = t
        self.sh, self.shva = t.shared(contents=b"\x31" * PAGE, listed=False)
        self.pr, self.prva = t.shared(contents=b"\x32" * PAGE, flags=1, listed=False)
        self.hva = t.reserve(PAGE)
        t.write(self.hva, b"\x33" * PAGE)
        self.h = t.add(abi.T_HEAP, heap_rec(self.hva // PAGE, PAGE, 1))
        self.hb = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(self.h, 0, PAGE))
        self.tx, self.txva = tex2d(t, contents=texels(4, 4, seed=1))
        self.out, self.ova = t.shared(length=2 * PAGE, listed=False)

    def vas(self):
        return [self.shva, self.prva, self.hva, self.txva]

    def res(self, flags):
        return [(self.sh, flags), (self.pr, flags), (self.h, flags), (self.hb, 0), (self.tx, flags), (self.out, 0)]

    def read_recs(self, slot):
        """Blit records copying each object's first bytes to out at slot *
        0x100: the shared buffer's 0x20 at +0, the private buffer's at
        +0x20, the heap buffer's at +0x40, the texture's 0x40 at +0x60."""
        o = slot * 0x100
        cp = B.copy_buffer_to_buffer
        return (cp(self.sh, self.out, 0, o, 0x20) + cp(self.pr, self.out, 0, o + 0x20, 0x20)
                + cp(self.hb, self.out, 0, o + 0x40, 0x20) + tex_out(self.tx, self.out, o + 0x60))

    def read(self, slot, flags, label):
        """read_recs in an exec whose outcome the setup decides."""
        return run_either(self.t, blit(self.read_recs(slot)), self.res(flags), label)

    def fill(self, byte, colour, flags, label):
        """Fills 0x20 bytes at +0x40 of each buffer with `byte` and the
        whole texture with `colour`."""
        recs = (B.fill(self.sh, 0x40, 0x20, byte) + B.fill(self.pr, 0x40, 0x20, byte + 1)
                + B.fill(self.hb, 0x40, 0x20, byte + 2) + tex_fill(self.tx, colour))
        return run_either(self.t, blit(recs), self.res(flags), label)

    def pages(self, pages, label):
        """The first 0x60 bytes of each page in `pages` (RAM page indices)."""
        s = self.t.s
        for name, page in zip(("shared", "private", "heap", "texture"), pages):
            s.dump(s.pa(page), 0x60, "%s: %s's page" % (label, name))

    def old_pages(self):
        return [self.t.leaves[va // PAGE] for va in self.vas()]


def _four(s, pages=48, more=None):
    """Bring-up, task 1 with Four's objects (and whatever `more` adds),
    defined and its object list installed."""
    s.bring_up()
    t = Lab(s, pages=pages)
    f = Four(t)
    extra = more(t) if more else None
    t.ready()
    return (t, f, extra) if more else (t, f)


def _retarget(t, f, bytes_=(0xA1, 0xA2, 0xA3, 0xA4)):
    """Points each object's leaf at a fresh page holding `bytes_`; returns
    the new pages."""
    s = t.s
    new = []
    for va, b in zip(f.vas(), bytes_):
        p = spare(s, byte=b)
        set_leaf(t, va // PAGE, p)
        new.append(p)
    return new


# ---- leaves rewritten between execs ----

@scenario
def interplay_memlive_retarget_flags():
    """Each kind of object is read once, then the guest points its leaf at
    another page holding other bytes. A live object keeps the pages it was
    bound to when it was made, and invalidateHost does not rebind it: it
    pages the host copy again from the same old pages (a private buffer's
    transfer buffer; a legacy heap's; a texture's), or, for a shared buffer
    and the service's detach resources, nothing at all (a shared buffer is
    always host-valid while it has its Metal buffer; a detach buffer
    prepares only when detached). The reads with flags 0 and with
    invalidateHost (1) after the rewrite are locked beside the one before;
    the old and the new pages are dumped."""
    s = _scn(interplay_memlive_retarget_flags)
    t, f = _four(s)
    old = f.old_pages()
    f.read(0, 0, "before the rewrite")
    new = _retarget(t, f)
    f.read(1, 0, "flags 0, after the rewrite")
    f.read(2, IH, "invalidateHost, after the rewrite")
    t.dump(f.ova, 0x300, "reads")
    f.pages(old, "old")
    f.pages(new, "new")
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_retarget_writeback():
    """Each kind of object is filled with invalidateGuest and synchronize
    (6), which writes the GPU's bytes back to the guest pages it is bound
    to; then the guest points each leaf at a fresh page, and a second fill
    with 6 is written back again. The second write-back lands in the old
    pages in every setup, not in the pages the table now names: a private
    buffer's and a heap's synchronize copy into the transfer buffer made
    at first use, a shared buffer and a detach resource are the old pages.
    A read with invalidateHost then shows what the host copies hold."""
    s = _scn(interplay_memlive_retarget_writeback)
    t, f = _four(s)
    old = f.old_pages()
    f.fill(0x51, BLUE, IG | SY, "first fill, written back")
    f.pages(old, "after the first write-back")
    new = _retarget(t, f)
    f.fill(0x61, GREEN, IG | SY, "second fill after the rewrite, written back")
    f.pages(old, "old, after the second write-back")
    f.pages(new, "new, after the second write-back")
    f.read(0, IH, "invalidateHost")
    t.dump(f.ova, 0xA0, "read")
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_retarget_discard():
    """Each kind of object is read, the guest points its leaf at a fresh
    page, and DiscardResources (63) drops what can be dropped: a shared
    buffer's Metal buffer, a private buffer's, a heap's and a texture's
    transfer buffer (they turn volatile and stay host-valid), a detach
    resource's backing. Then a read with flags 0 and one with
    invalidateHost. In the map setup the flags-0 read makes only the shared
    buffer again (it has no Metal buffer), over the new page; the others
    read their host copies (the old bytes) until invalidateHost pages them
    from the new pages. In the service the shared buffer, the private
    buffer and the heap are detach resources, attached again at the next
    exec that lists them whatever its flags: all new at flags 0 but the
    texture, a host copy, new with invalidateHost. In the blocks setup
    every read shows the old pages, because the host task still maps them
    at those VAs; only an UnmapMemory (34) of the four pages and another 63
    bring the new pages in, and then every setup reads them."""
    s = _scn(interplay_memlive_retarget_discard)
    t, f = _four(s)
    f.read(0, 0, "before")
    new = _retarget(t, f)
    drop = [f.sh, f.pr, f.h, f.tx]
    s.submit(abi.OP_DISCARD_RESOURCES, refs_payload(1, drop), label="63 after the rewrite")
    f.read(1, 0, "flags 0 after 63")
    f.read(2, IH, "invalidateHost after 63")
    lo = min(f.vas())
    t.unmap(lo, max(f.vas()) + PAGE - lo, label="34 over the four pages")
    s.submit(abi.OP_DISCARD_RESOURCES, refs_payload(1, drop), label="63 after the unmap")
    f.read(3, IH, "invalidateHost after 34 and 63")
    t.dump(f.ova, 0x400, "reads")
    f.pages(new, "new")
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_hole_under_live():
    """The guest clears the leaves (holes) under four live objects. Nothing
    walks the table for a live object, so reads with flags 0 and with
    invalidateHost still run over the pages bound at first use, in every
    setup. A fifth object, a buffer never used before over a hole, faults
    its first exec ("Found a hole in the page table"): in the blocks setup
    the lazy map walks the hole, in the map setup the new mapping does, in
    the service the backing does. With its leaf restored it is made and
    runs."""
    s = _scn(interplay_memlive_hole_under_live)
    t, f, (late, lva) = _four(s, more=lambda t: t.shared(contents=b"\x3A" * PAGE, listed=False))
    f.read(0, 0, "before")
    for va in f.vas() + [lva]:
        clear_leaf(t, va // PAGE)
    f.read(1, 0, "flags 0 over holes")
    f.read(2, IH, "invalidateHost over holes")
    use = blit(B.copy_buffer_to_buffer(late, f.out, 0, 0x300, 0x20))
    run_either(t, use, [(late, 0), (f.out, 0)], "a new buffer over a hole")
    set_leaf(t, lva // PAGE, t.leaves[lva // PAGE])
    run_either(t, use, [(late, 0), (f.out, 0)], "its leaf restored")
    t.dump(f.ova, 0x320, "reads")
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_readonly_flip():
    """Bit 31 of a live shared buffer's leaf flipped both ways. A buffer
    made over a writable leaf takes fills after the leaf turns read-only:
    its mapping or backing was made writable. Discarded (63) and listed
    again it is bound read-only: in the map setup the mapping of a range
    with a read-only page is read-only and in the service the backing
    carries the leaves' read-only bit, and the GPU's fill is dropped
    without a fault; the blocks setup's host task maps it writable (and
    keeps its old mapping), so fills land. The leaf made writable again
    changes nothing until another 63."""
    s = _scn(interplay_memlive_readonly_flip)
    s.bring_up()
    t = Lab(s, pages=40)
    a, ava = t.shared(contents=b"\x40" * PAGE, listed=False)
    t.ready()
    vpn = ava // PAGE
    page = t.leaves[vpn]
    steps = [("writable", None, 0x41), ("read-only leaf, live buffer", False, 0x42),
             ("read-only leaf, after 63", "63", 0x43), ("writable leaf, bound read-only", True, 0x44),
             ("writable leaf, after 63", "63", 0x45)]
    for k, (label, change, byte) in enumerate(steps):
        if change is False or change is True:
            set_leaf(t, vpn, page, writable=change)
        elif change == "63":
            s.submit(abi.OP_DISCARD_RESOURCES, refs_payload(1, [a]), label="63: " + label)
        run_either(t, blit(B.fill(a, 0x10 * k, 0x10, byte)), [(a, 0)], label)
    s.dump(s.pa(page), 0x50, "the buffer's page")
    t.delete()
    s.events("end")
    return s


# ---- intermediate tables ----

HI = 4096  # the first VA page of the second last-level table (64 MiB)


def _hi_task(s, depth):
    """A task of depth 2 or 3 with VA pages 0-15 (object list, records,
    command buffers, out at VA 8) and HI..HI+3 mapped, each to a page of
    its own: at depth 2 HI is under entry 1 of the root table, at depth 3
    under entry 1 of the first second-level table."""
    leaves = {v: s.alloc_page() for v in range(16)}
    leaves.update({HI + i: s.alloc_page() for i in range(4)})
    return Lab(s, pages=16, depth=depth, leaves=leaves, length=(HI + 8) * PAGE)


def _hi_objects(t):
    """A at HI (0xA0s), B at HI+1 (0xB0s), C at HI+2 (0xC0s), out at VA
    8; HI+3 is left for command buffers."""
    a, _ = t.shared(contents=b"\xA0" * PAGE, va=HI * PAGE, listed=False)
    b, _ = t.shared(contents=b"\xB0" * PAGE, va=(HI + 1) * PAGE, listed=False)
    c, _ = t.shared(contents=b"\xC0" * PAGE, va=(HI + 2) * PAGE, listed=False)
    out, ova = t.shared(listed=False)
    return a, b, c, out, ova


def _hi_cmdbuf(t, segment):
    """The segment at VA HI+3; returns its range."""
    va = (HI + 3) * PAGE
    t.write(va, segment)
    return [(va, len(segment))]


def _intermediate(depth):
    s = Scenario("interplay_memlive_intermediate_depth%d" % depth, doc=interplay_memlive_intermediate.__doc__)
    s.bring_up()
    t = _hi_task(s, depth)
    a, b, c, out, ova = _hi_objects(t)
    t.ready()
    cp = B.copy_buffer_to_buffer
    run_either(t, blit(cp(a, out, 0, 0, 0x20)), [(a, 0), (out, 0)], "A read through the table as defined")
    x = seg(abi.ENC_BLIT, cp(a, out, 0, 0x20, 0x20) + cp(b, out, 0, 0x40, 0x20))
    y = seg(abi.ENC_BLIT, cp(a, out, 0, 0x60, 0x20) + cp(b, out, 0, 0x80, 0x20))
    ranges = _hi_cmdbuf(t, x)
    level = depth - 2  # the entry that names HI's last-level table
    entries = table_entries(t, depth - 1, HI >> 12)
    entries.update({0: leaf(s, spare(s, byte=0xA1)), 1: leaf(s, spare(s, byte=0xB1)), 3: leaf(s, spare(s, data=y))})
    clone = write_table(s, entries)
    swap = entry_gpa(t, HI, level)
    s.u32(swap, s.pfn(clone))
    either(s, E3, exec_payload(1, ranges, [(a, 0), (b, 0), (out, 0)]), "the command buffer at HI+3 after the swap")
    s.u32(swap, 0)
    run_either(t, blit(cp(a, out, 0, 0xA0, 0x20)), [(a, 0), (out, 0)], "A with the entry cleared")
    run_either(t, blit(cp(c, out, 0, 0xC0, 0x20)), [(c, 0), (out, 0)], "C, never made, with the entry cleared")
    s.u32(swap, s.pfn(clone))
    run_either(t, blit(cp(c, out, 0, 0xE0, 0x20)), [(c, 0), (out, 0)], "C with the copy back")
    t.dump(ova, 0x100, "reads")
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_intermediate():
    """An intermediate page-table entry rewritten between execs, at depth 2
    (the root table's entry 1, which names the last-level table of VA pages
    4096-8191) and at depth 3 (entry 1 of the first second-level table, the
    same VA pages). Buffer A at VA page 4096 is read once; then the guest
    points the entry at a copy of the table whose leaves for A, for B
    (never made) and for the command-buffer page 4099 name other pages. The
    walk reads every table below the root at every walk: the command buffer
    at 4099 is read from the new page and runs its other copies; A, live,
    reads its old page; B is made over the copy's leaf. With the entry
    cleared, A still reads (nothing walks for a live object) and C, never
    made, faults ("Found a hole in the page table"); with the copy back C
    is made over it. In the blocks setup a buffer is a lazy map of the host
    task, which maps whatever the table says when a range is first used."""
    return [_intermediate(2), _intermediate(3)]


def _held_intermediate(depth):
    s = Scenario("interplay_memlive_held_intermediate_depth%d" % depth, doc=interplay_memlive_held_intermediate.__doc__)
    g = Rig(s, worker=False, make=lambda s: _hi_task(s, depth))
    t = g.t
    a, b, c, out, ova = _hi_objects(t)
    t.ready()
    res = [(a, 0), (b, 0), (out, 0)]
    cp = B.copy_buffer_to_buffer
    run_either(t, blit(cp(a, out, 0, 0, 0x20)), [(a, 0), (out, 0)], "A read")
    x = seg(abi.ENC_BLIT, cp(a, out, 0, 0x20, 0x20) + cp(b, out, 0, 0x40, 0x20))
    y = seg(abi.ENC_BLIT, cp(a, out, 0, 0x60, 0x20) + cp(b, out, 0, 0x80, 0x20))
    ranges = _hi_cmdbuf(t, x)
    st = g.hold_on(g.f, t, ranges, res, "the command buffer at HI+3")
    over = {HI: leaf(s, spare(s, byte=0xA1)), HI + 1: leaf(s, spare(s, byte=0xB1)), HI + 3: leaf(s, spare(s, data=y))}
    if depth == 2:
        s.u32(entry_gpa(t, HI, 0), s.pfn(clone_tree(t, 1, 1, over)))
    else:
        s.u32(entry_gpa(t, HI, 0), s.pfn(clone_tree(t, 1, 0, over)))
    g.open_gate()
    g.settle(st, g.f, "released after the swap")
    t.dump(ova, 0xA0, "reads")
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_held_intermediate():
    """An exec whose command buffer is at VA page 4099 waits on the gate
    (a barrier on child FIFO 1's next stamp) while the guest swaps a
    subtree: at depth 2 the root table's entry 1 (the last-level table of
    VA pages 4096-8191), at depth 3 the root table's entry 0 (the whole
    second-level table, VA pages 0-16M, the object list and every command
    buffer among them), each for a copy whose leaves for A, B and page
    4099 name other pages. The FIFO copied the exec's payload before its
    barrier but reads its command buffer and makes its objects only when
    released: it runs the copy's command buffer, A (read before the hold)
    keeps its old page, and B is made over the copy's leaf."""
    return [_held_intermediate(2), _held_intermediate(3)]


# ---- leaves rewritten while an exec is held ----

@scenario
def interplay_memlive_held_cmdbuf():
    """Command-buffer pages rewritten while their execs wait on the gate.
    Exec 1 (root FIFO) has its segment at VA page 4; exec 2 (worker FIFO,
    child 2, behind the same gate) has a 40-byte fill segment starting 36
    bytes before VA page 6, so only its fill value and padding are in page
    6; exec 3 (root FIFO) reads its segment at VA page 5, whose leaf the
    guest makes read-only. While they wait the guest points page 4's leaf
    at a page holding another segment and page 6's at a page holding
    another fill value. Each FIFO copied its command before the barrier,
    but a command buffer is walked when the exec runs: released, exec 1
    runs the other segment, exec 2 runs a segment spliced from two pages
    (the first page's header and fill fields, the new page's value), and
    exec 3 runs from the read-only page (a command buffer is only read).
    The two FIFOs run at once after the gate opens, so the blocks setup's
    read_memory calls are reported sorted."""
    s = _scn(interplay_memlive_held_cmdbuf)
    g = Rig(s, pages=24)
    t = g.t
    out, ova = t.shared(listed=False)
    t.ready()
    res = [(out, 0)]
    one = seg(abi.ENC_BLIT, B.fill(out, 0, 0x20, 0x11))
    other = seg(abi.ENC_BLIT, B.fill(out, 0x20, 0x20, 0x22))
    t.write(4 * PAGE, one)
    va2 = 6 * PAGE - 36
    two = seg(abi.ENC_BLIT, B.fill(out, 0x40, 0x20, 0x33))
    two_other = seg(abi.ENC_BLIT, B.fill(out, 0x40, 0x20, 0x44))
    assert len(two) == 40 and two[:36] == two_other[:36]
    t.write(va2, two)
    three = seg(abi.ENC_BLIT, B.fill(out, 0x60, 0x20, 0x55))
    t.write(5 * PAGE + 0x100, three)
    st1 = g.hold_on(g.f, t, [(4 * PAGE, len(one))], res, "exec 1, page 4")
    st2 = g.hold_on(g.worker, t, [(va2, len(two))], res, "exec 2 on the worker, across pages 5 and 6")
    st3 = g.hold_on(g.f, t, [(5 * PAGE + 0x100, len(three))], res, "exec 3, page 5")
    set_leaf(t, 4, spare(s, data=other))
    set_leaf(t, 6, spare(s, data=two_other[36:]))
    set_leaf(t, 5, t.leaves[5], writable=False)
    g.open_gate()
    g.settle(st1, g.f, "exec 1")
    g.settle(st3, g.f, "exec 3")
    g.worker.wait_stamp(st2, label="exec 2")
    s.events("the three execs", sort_blocks=True)
    t.dump(ova, 0x80, "fills")
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_held_first_use():
    """An exec that fills a shared buffer made before (S), and a shared
    buffer (N), a private buffer (P), a two-page private heap with a buffer
    at its second page (H) and a texture (X) that no exec has named yet,
    waits on the gate while the guest points each one's leaves at other
    pages (for the heap only its second page). Released, the exec makes
    N, P, H and X from the table as it is then, in every setup: N's fill
    lands in its new page; P, H and X are paged from their new pages (a
    copy out shows the new bytes). S, live, fills its old page."""
    s = _scn(interplay_memlive_held_first_use)
    g = Rig(s, pages=40)
    t = g.t
    sh, shva = t.shared(contents=b"\x10" * PAGE, listed=False)
    n, nva = t.shared(contents=b"\x20" * PAGE, listed=False)
    p, pva = t.shared(contents=b"\x30" * PAGE, flags=1, listed=False)
    hva = t.reserve(2 * PAGE)
    t.write(hva, b"\x40" * 2 * PAGE)
    h = t.add(abi.T_HEAP, heap_rec(hva // PAGE, 2 * PAGE, 1))
    hb = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(h, PAGE, PAGE))
    x, xva = tex2d(t, contents=texels(4, 4, seed=3))
    out, ova = t.shared(listed=False)
    t.ready()
    run_either(t, blit(B.copy_buffer_to_buffer(sh, out, 0, 0, 0x10)), [(sh, 0), (out, 0)], "S made")
    cp = B.copy_buffer_to_buffer
    recs = (B.fill(sh, 0x10, 0x10, 0x1F) + B.fill(n, 0x10, 0x10, 0x2F) + cp(n, out, 0, 0x20, 0x10)
            + cp(p, out, 0, 0x30, 0x10) + cp(hb, out, 0, 0x40, 0x10) + tex_out(x, out, 0x50))
    res = [(sh, 0), (n, 0), (p, 0), (h, 0), (hb, 0), (x, 0), (out, 0)]
    st = g.hold(blit(recs), res, "first use of N, P, H and X")
    new = {}
    for va, byte in ((shva, 0x11), (nva, 0x21), (pva, 0x31), (hva + PAGE, 0x41), (xva, 0x51)):
        new[va] = spare(s, byte=byte)
        set_leaf(t, va // PAGE, new[va])
    g.release(st, "released")
    t.dump(ova, 0x90, "reads")
    for va, name in ((shva, "S"), (nva, "N")):
        s.dump(s.pa(t.leaves[va // PAGE]), 0x20, "%s's old page" % name)
        s.dump(s.pa(new[va]), 0x20, "%s's new page" % name)
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_held_first_use_hole_readonly():
    """Two execs, each holding on the gate a first use of a shared buffer
    whose leaf the guest changes while it waits. The first buffer's leaf
    becomes a hole: released, the exec faults in every setup, making the
    object ("Found a hole in the page table"); with the leaf restored the
    next exec makes it and fills it. The second buffer's leaf loses its
    writable bit: released, the object is made read-only in the map setup
    and in the service and the fill is dropped without a fault; the blocks
    setup maps it writable and the fill lands."""
    s = _scn(interplay_memlive_held_first_use_hole_readonly)
    g = Rig(s, pages=24)
    t = g.t
    a, ava = t.shared(contents=b"\x60" * PAGE, listed=False)
    b, bva = t.shared(contents=b"\x70" * PAGE, listed=False)
    t.ready()
    st = g.hold(blit(B.fill(a, 0, 0x10, 0x61)), [(a, 0)], "A's first use")
    clear_leaf(t, ava // PAGE)
    g.release(st, "released over a hole")
    set_leaf(t, ava // PAGE, t.leaves[ava // PAGE])
    run_either(t, blit(B.fill(a, 0x10, 0x10, 0x62)), [(a, 0)], "A, its leaf restored")
    st = g.hold(blit(B.fill(b, 0, 0x10, 0x71)), [(b, 0)], "B's first use")
    set_leaf(t, bva // PAGE, t.leaves[bva // PAGE], writable=False)
    g.release(st, "released over a read-only leaf")
    t.dump(ava, 0x30, "A's page")
    t.dump(bva, 0x20, "B's page")
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_held_discard_retarget():
    """An exec that reads the four kinds of object (a shared and a private
    buffer, a private heap's buffer, a texture), all made and read before,
    waits on the gate while the worker FIFO (child 2) discards them (63)
    and the guest points their leaves at other pages. Released with flags
    0 and then again with invalidateHost, it shows what each setup makes
    again over the new pages: a discard defers the rebinding to the next
    exec that pages the object, and what pages it is the setup's (the
    retarget_discard scenario's table, here decided at release time)."""
    s = _scn(interplay_memlive_held_discard_retarget)
    g = Rig(s, pages=48)
    t = g.t
    f = Four(t)
    t.ready()
    f.read(0, 0, "before")
    st = g.hold(blit(f.read_recs(1)), f.res(0), "a read, flags 0")
    g.work(abi.OP_DISCARD_RESOURCES, refs_payload(1, [f.sh, f.pr, f.h, f.tx]), "63 on the worker")
    _retarget(t, f)
    g.release(st, "released")
    f.read(2, IH, "invalidateHost")
    t.dump(f.ova, 0x300, "reads")
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_inflight_rewrite():
    """Five execs rung in one doorbell fill a live shared buffer and a live
    private buffer (the private one with invalidateGuest and synchronize,
    6, so each is written back) at five offsets; right after the doorbell,
    without waiting, the guest points both leaves at other pages and
    writes those. Whenever each exec runs against the rewrite, its objects
    are already bound: every fill and every write-back lands in the old
    pages, and the new pages keep the guest's bytes."""
    s = _scn(interplay_memlive_inflight_rewrite)
    s.bring_up()
    t = Lab(s, pages=32)
    a, ava = t.shared(contents=b"\x80" * PAGE, listed=False)
    p, pva = t.shared(contents=b"\x90" * PAGE, flags=1, listed=False)
    t.ready()
    run_either(t, blit(B.fill(a, 0x100, 1, 0x8F) + B.fill(p, 0x100, 1, 0x9F)), [(a, 0), (p, IG | SY)], "made")
    last = None
    for k in range(5):
        last = t.cmd(blit(B.fill(a, 0x10 * k, 0x10, 0x81 + k) + B.fill(p, 0x10 * k, 0x10, 0x91 + k)),
                     resources=[(a, 0), (p, IG | SY)])
    s.doorbell()
    na, np_ = s.alloc_page(), s.alloc_page()
    set_leaf(t, ava // PAGE, na)
    set_leaf(t, pva // PAGE, np_)
    s.fill(s.pa(na), PAGE, 0x88)
    s.fill(s.pa(np_), PAGE, 0x98)
    s.fifo.wait_stamp(last, label="the fifth exec")
    s.fault_state("after five")
    for va, page, name in ((ava, na, "shared"), (pva, np_, "private")):
        s.dump(s.pa(t.leaves[va // PAGE]), 0x60, "%s: old page" % name)
        s.dump(s.pa(page), 0x60, "%s: new page" % name)
    t.delete()
    s.events("end")
    return s


# ---- two tasks over the same guest pages ----

def _alias_pair(g):
    """Task 1 (g.t) with a shared buffer A (VA 8, 0x0As) and a private
    buffer P (VA 9, 0x0Bs); task 2, whose VA pages 8 and 9 are task 1's
    pages 8 and 9 (its other pages its own), with a shared A2 and a private
    P2 over them and out2 at VA 10. Both defined, lists installed."""
    s = g.s
    t1 = g.t
    a, ava = t1.shared(contents=b"\x0A" * PAGE, va=8 * PAGE, listed=False)
    p, pva = t1.shared(contents=b"\x0B" * PAGE, flags=1, va=9 * PAGE, listed=False)
    out1, o1va = t1.shared(va=10 * PAGE, listed=False)
    leaves = {v: s.alloc_page() for v in range(16)}
    leaves[8], leaves[9] = t1.leaves[8], t1.leaves[9]
    t2 = Lab(s, tid=2, pages=16, leaves=leaves)
    a2, _ = t2.shared(va=8 * PAGE, listed=False)
    p2, _ = t2.shared(flags=1, va=9 * PAGE, listed=False)
    out2, o2va = t2.shared(va=10 * PAGE, listed=False)
    t1.ready()
    t2.ready()
    return t1, t2, (a, p, out1, o1va), (a2, p2, out2, o2va)


@scenario
def interplay_memlive_alias_ordered():
    """Two tasks over the same two guest pages, their work ordered only by
    barriers across two FIFOs and rung at once: task 1 on the root FIFO
    fills its shared buffer A and its private buffer P (written back:
    invalidateGuest and synchronize, 6); task 2 on the worker (child 2),
    behind a barrier on that exec's stamp, copies its A2 and P2 (P2 with
    invalidateHost) out; task 1, behind task 2's stamp, fills again; task
    2, behind that, copies again with flags 0. The shared buffers are the
    same RAM in every setup, so task 2 sees both of task 1's fills. Task
    2's private buffer is paged at its first read: in-process it is then a
    host copy that flags 0 never pages again, so its second read misses
    task 1's second fill; in the service it is the guest pages, and sees
    it (docs/testing.md section 6.10: detach resources)."""
    s = _scn(interplay_memlive_alias_ordered)
    g = Rig(s)
    t1, t2, (a, p, out1, o1va), (a2, p2, out2, o2va) = _alias_pair(g)
    cp = B.copy_buffer_to_buffer
    f, w = g.f, g.worker
    e1 = exec_on(f, t1, t1.place(blit(B.fill(a, 0, 0x20, 0x11) + B.fill(p, 0, 0x20, 0x12))), [(a, 0), (p, IG | SY)])
    e2 = exec_on(w, t2, t2.place(blit(cp(a2, out2, 0, 0, 0x40) + cp(p2, out2, 0, 0x40, 0x40))),
                 [(a2, 0), (p2, IH), (out2, 0)], barriers=[(0, e1)])
    e3 = exec_on(f, t1, t1.place(blit(B.fill(a, 0x20, 0x20, 0x21) + B.fill(p, 0x20, 0x20, 0x22))),
                 [(a, 0), (p, IG | SY)], barriers=[(w.stamp_index, e2)])
    e4 = exec_on(w, t2, t2.place(blit(cp(a2, out2, 0, 0x80, 0x40) + cp(p2, out2, 0, 0xC0, 0x40))),
                 [(a2, 0), (p2, 0), (out2, 0)], barriers=[(0, e3)])
    f.doorbell()
    w.doorbell()
    f.wait_stamp(e3, label="task 1's second fill")
    w.wait_stamp(e4, label="task 2's second copy")
    s.fault_state("after the four")
    t2.dump(o2va, 0x100, "task 2's copies")
    t1.dump(9 * PAGE, 0x40, "the private pages")
    t2.delete()
    t1.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_alias_writeback_race():
    """Two tasks' private buffers over one guest page, each filling a
    different half with invalidateGuest and synchronize (6): task 1 its
    first 0x20 bytes, then task 2 (paged from the page after task 1's
    write-back, invalidateHost) bytes 0x20-0x3f, then task 1 bytes
    0x40-0x5f with flags 6 again. In-process a synchronize copies the whole
    host copy back over the page, so each write-back erases what the other
    task wrote since its copy was paged: the page ends with task 1's host
    copy, and task 2's bytes are lost. In the service both buffers are the
    page itself, and every fill survives."""
    s = _scn(interplay_memlive_alias_writeback_race)
    g = Rig(s, worker=False)
    t1, t2, (a, p, out1, o1va), (a2, p2, out2, o2va) = _alias_pair(g)
    run_either(t1, blit(B.fill(p, 0, 0x20, 0x31)), [(p, IG | SY)], "task 1 fills 0-0x1f")
    run_either(t2, blit(B.fill(p2, 0x20, 0x20, 0x32)), [(p2, IH | IG | SY)], "task 2 fills 0x20-0x3f")
    t1.dump(9 * PAGE, 0x60, "the page after task 2's write-back")
    run_either(t1, blit(B.fill(p, 0x40, 0x20, 0x33)), [(p, IG | SY)], "task 1 fills 0x40-0x5f")
    t1.dump(9 * PAGE, 0x60, "the page after task 1's second write-back")
    t2.delete()
    t1.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_alias_held_retarget():
    """Task 1's fill of the shared page waits on the gate (root FIFO); task
    2's copy of the page, never made, waits behind it on the worker FIFO
    (a barrier on task 1's stamp). While both wait the guest points task
    2's leaf for that page at another page. Released, task 1's fill lands
    in the old page and task 2 makes its buffer over the new one and
    copies the new page's bytes, not the fill: aliasing is whatever the
    tables say when each object is made."""
    s = _scn(interplay_memlive_alias_held_retarget)
    g = Rig(s)
    t1, t2, (a, p, out1, o1va), (a2, p2, out2, o2va) = _alias_pair(g)
    st1 = g.hold_on(g.f, t1, t1.place(blit(B.fill(a, 0, 0x40, 0x41))), [(a, 0)], "task 1's fill")
    e2 = exec_on(g.worker, t2, t2.place(blit(B.copy_buffer_to_buffer(a2, out2, 0, 0, 0x40))), [(a2, 0), (out2, 0)],
                 barriers=[(0, st1)])
    g.worker.doorbell()
    new = spare(s, byte=0x4F)
    s.u32(entry_gpa(t2, 8), leaf(s, new))
    g.open_gate()
    g.settle(st1, g.f, "task 1's fill")
    g.worker.wait_stamp(e2, label="task 2's copy")
    t2.dump(o2va, 0x40, "task 2's copy")
    t1.dump(8 * PAGE, 0x40, "the old page")
    s.dump(s.pa(new), 0x40, "the new page")
    t2.delete()
    t1.delete()
    s.events("end")
    return s


def _pt_task(s, tid, pages, extra):
    """A Lab whose VA pages 0..pages-1 have pages of their own, plus the
    leaves in `extra` ({vpn: RAM page}); its length reaches the last."""
    leaves = {v: s.alloc_page() for v in range(pages)}
    leaves.update(extra)
    return Lab(s, tid=tid, pages=pages, leaves=leaves, length=(max(leaves) + 1) * PAGE)


@scenario
def interplay_memlive_alias_page_table():
    """Task 2 maps task 1's page table as data: its VA page 10 is task 1's
    (only) table page. Task 2 copies the table out (it equals the guest's
    table), then writes leaves into it with the GPU: a blit copy of a u32
    from its own buffer to entry 12 points task 1's VA page 12 (buffer B,
    never made) at another page, and one to entry 5 points task 1's
    command-buffer page 5 at a page holding another segment. Task 1's next
    execs walk the tables the GPU wrote: B is made over the new page and
    the other segment runs, in every setup (the blocks setup's walks read
    guest RAM through read_memory, which the GPU's write reached). Last,
    task 1's read of buffer C (VA 13, never made) waits on the gate while
    task 2, on the worker FIFO, rewrites entry 13: released, C is made over
    the page task 2 wrote."""
    s = _scn(interplay_memlive_alias_page_table)
    g = Rig(s, pages=16)
    t1 = g.t
    b, _ = t1.shared(contents=b"\x5B" * PAGE, va=12 * PAGE, listed=False)
    c, _ = t1.shared(contents=b"\x5C" * PAGE, va=13 * PAGE, listed=False)
    out1, o1va = t1.shared(va=8 * PAGE, listed=False)
    nb, nc = spare(s, byte=0x6B), spare(s, byte=0x6C)
    cmd_old = seg(abi.ENC_BLIT, B.fill(out1, 0x40, 0x20, 0x7A))
    cmd_new = seg(abi.ENC_BLIT, B.fill(out1, 0x60, 0x20, 0x7C))
    t1.write(5 * PAGE, cmd_old)
    ncmd = spare(s, data=cmd_new)
    t2 = _pt_task(s, 2, 12, {12: t1.pt_root})
    v, _ = t2.shared(contents=struct.pack("<III", leaf(s, nb), leaf(s, ncmd), leaf(s, nc)), va=8 * PAGE, listed=False)
    out2, o2va = t2.shared(va=9 * PAGE, listed=False)
    pt, _ = t2.shared(va=12 * PAGE, listed=False)
    t1.ready()
    t2.ready()
    cp = B.copy_buffer_to_buffer
    run_either(t2, blit(cp(pt, out2, 0, 0, 0x40)), [(pt, 0), (out2, 0)], "task 2 reads task 1's table")
    s.cmp(t2.gpa(o2va), s.pa(t1.pt_root), 0x40, label="the copy == task 1's table")
    run_either(t2, blit(cp(v, pt, 0, 4 * 12, 4) + cp(v, pt, 4, 4 * 5, 4)), [(v, 0), (pt, 0)],
               "task 2 writes task 1's entries 12 and 5")
    s.dump(s.pa(t1.pt_root, 4 * 5), 4, "task 1's entry 5")
    s.dump(s.pa(t1.pt_root, 4 * 12), 4, "task 1's entry 12")
    run_either(t1, blit(cp(b, out1, 0, 0, 0x20)), [(b, 0), (out1, 0)], "task 1 reads B")
    either(s, E3, exec_payload(1, [(5 * PAGE, len(cmd_old))], [(out1, 0)]), "task 1's command buffer at page 5")
    st = g.hold_on(g.f, t1, t1.place(blit(cp(c, out1, 0, 0x20, 0x20))), [(c, 0), (out1, 0)], "task 1 reads C")
    t2.reset_cmdbufs()
    g.work(E3, exec_payload(2, t2.place(blit(cp(v, pt, 8, 4 * 13, 4))), [(v, 0), (pt, 0)]),
           "task 2 writes entry 13 on the worker")
    g.open_gate()
    g.settle(st, g.f, "task 1 reads C")
    t1.dump(o1va, 0x80, "task 1's reads")
    t2.delete()
    t1.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_self_mapped_table():
    """A task whose VA page 20 is its own page-table page, as a shared
    buffer T. It reads its table through T, then writes its own leaf for VA
    page 12 with a GPU copy (buffer B, never made, is then made over the
    new page); then writes the leaf of page 20 itself, pointing T's VA at
    another page. T, live, still reads the table page (its copy of entry 20
    shows the new leaf). Discarded (63) and listed with invalidateHost, T
    is made again over the table as it is now: the other page in the map
    setup and in the service; in the blocks setup the host task keeps the
    table page mapped at that VA."""
    s = _scn(interplay_memlive_self_mapped_table)
    s.bring_up()
    nb, nt = spare(s, byte=0x9B), spare(s, byte=0x9E)
    t = Lab(s, pages=24)
    tb, _ = t.shared(va=20 * PAGE, listed=False)
    s.u32(t.table_gpa(0, 0, 20), leaf(s, t.pt_root))
    b, _ = t.shared(contents=b"\x5B" * PAGE, va=12 * PAGE, listed=False)
    v, _ = t.shared(contents=struct.pack("<II", leaf(s, nb), leaf(s, nt)), va=8 * PAGE, listed=False)
    out, ova = t.shared(va=9 * PAGE, listed=False)
    t.ready()
    cp = B.copy_buffer_to_buffer
    run_either(t, blit(cp(tb, out, 0, 0, 0x60)), [(tb, 0), (out, 0)], "T reads the table")
    s.cmp(t.gpa(ova), s.pa(t.pt_root), 0x60, label="T's copy == the table")
    run_either(t, blit(cp(v, tb, 0, 4 * 12, 4)), [(v, 0), (tb, 0)], "the GPU writes entry 12")
    run_either(t, blit(cp(b, out, 0, 0x60, 0x20)), [(b, 0), (out, 0)], "B, made after")
    run_either(t, blit(cp(v, tb, 4, 4 * 20, 4)), [(v, 0), (tb, 0)], "the GPU writes entry 20, T's own")
    run_either(t, blit(cp(tb, out, 4 * 20, 0x80, 4)), [(tb, 0), (out, 0)], "T, live, reads entry 20")
    s.submit(abi.OP_DISCARD_RESOURCES, refs_payload(1, [tb]), label="63: T")
    run_either(t, blit(cp(tb, out, 0, 0xA0, 0x20)), [(tb, IH), (out, 0)], "T made again")
    t.dump(ova + 0x60, 0x60, "reads")
    s.dump(s.pa(t.pt_root, 4 * 12), 4, "entry 12")
    s.dump(s.pa(t.pt_root, 4 * 20), 4, "entry 20")
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_shared_table_held():
    """Two tasks over one page table and one object list (task 2's task
    root names task 1's table root and task 1's list, as
    memory_gap_shared_tables does), so both see the same VAs: task 2's
    first read of buffers A and B waits on the gate while the guest points
    the shared leaves of A's and B's pages at other pages, and task 1 then
    reads both. Task 1's A, made before the rewrite, keeps its old page;
    its B, first made after, is made over the new one (task 1 runs on the
worker FIFO, child 2: the root FIFO is held). Released, task 2
    makes its own A and B over the rewritten leaves: both of its reads show
    the new pages (objects are a task's own; the table is shared)."""
    s = _scn(interplay_memlive_shared_table_held)
    g = Rig(s, pages=16)
    t1 = g.t
    a, _ = t1.shared(contents=b"\xAA" * PAGE, va=8 * PAGE, listed=False)
    b, _ = t1.shared(contents=b"\xBB" * PAGE, va=9 * PAGE, listed=False)
    out, ova = t1.shared(va=10 * PAGE, listed=False)
    out2, o2va = t1.shared(va=11 * PAGE, listed=False)
    t1.ready()
    root2 = s.alloc_page()
    s.mem(s.pa(root2), abi.TASK_ROOT.pack(s.pfn(t1.pt_root), 1, 0, t1.ol.length(), 0, 0, 0))
    s.submit(abi.OP_DEFINE_TASK2, define_payload(2, t1.length, s.pfn(root2)), label="task 2 over task 1's table")
    cp = B.copy_buffer_to_buffer
    run_either(t1, blit(cp(a, out, 0, 0, 0x20)), [(a, 0), (out, 0)], "task 1 reads A")
    ranges = t1.place(blit(cp(b, out2, 0, 0, 0x20) + cp(a, out2, 0, 0x20, 0x20)))
    st = g.f.cmd(E3, exec_payload(2, ranges, [(b, 0), (a, 0), (out2, 0)]), barriers=g.barrier())
    g.f.doorbell()
    g.f.wait_stamp(st, timeout_ms=300, label="task 2 reads B and A: held by the gate")
    set_leaf(t1, 9, spare(s, byte=0xBC))
    set_leaf(t1, 8, spare(s, byte=0xAC))
    g.work(E3, exec_payload(1, t1.place(blit(cp(b, out, 0, 0x20, 0x20) + cp(a, out, 0, 0x40, 0x20))),
                            [(b, 0), (a, 0), (out, 0)]), "task 1 reads B and A on the worker")
    g.open_gate()
    g.settle(st, g.f, "task 2 reads B and A")
    t1.dump(ova, 0x60, "task 1's reads")
    t1.dump(o2va, 0x40, "task 2's reads")
    s.submit(abi.OP_DELETE_TASK, struct.pack("<I", 2), label="32: task 2")
    t1.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_shared_subtree():
    """Task 2's page table reaches into task 1's: task 1 is 2 deep, task 2 3
    deep, and entry 1 of task 2's second-level table names task 1's
    last-level table for VA pages 4096-8191, so both tasks see the same
    leaves there (a walk reads whatever page an entry names, whatever table
    it came from). Task 2's first read of A and B at VA pages 4096 and 4097
    waits on the gate while the guest rewrites the shared leaf of page 4097
    and points task 1's own root entry 1 at a copy of the last-level table
    whose leaf for page 4096 names another page. Released, task 2 walks its
    own entry to the shared table: B is made over the rewritten leaf, and A
    over the unchanged one, not the copy's. Task 1 then walks the copy: its
    B (never made) is made over the copy's leaf, the old page."""
    s = _scn(interplay_memlive_shared_subtree)
    g = Rig(s, worker=False, make=lambda s: _hi_task(s, 2))
    t1 = g.t
    a, b, c, out, ova = _hi_objects(t1)
    t1.ready()
    cp = B.copy_buffer_to_buffer
    run_either(t1, blit(cp(a, out, 0, 0, 0x20)), [(a, 0), (out, 0)], "task 1 reads A")
    leaves = {v: s.alloc_page() for v in range(16)}
    leaves[HI] = t1.leaves[HI]
    t2 = Lab(s, tid=2, pages=16, depth=3, leaves=leaves, length=(HI + 8) * PAGE)
    s.u32(entry_gpa(t2, HI, 1), s.pfn(t1._tables[(1, HI >> 12)]))
    a2, _ = t2.shared(va=HI * PAGE, listed=False)
    b2, _ = t2.shared(va=(HI + 1) * PAGE, listed=False)
    out2, o2va = t2.shared(va=8 * PAGE, listed=False)
    t2.ready()
    st = g.hold_on(g.f, t2, t2.place(blit(cp(a2, out2, 0, 0, 0x20) + cp(b2, out2, 0, 0x20, 0x20))),
                   [(a2, 0), (b2, 0), (out2, 0)], "task 2 reads A and B")
    set_leaf(t1, HI + 1, spare(s, byte=0xB7))
    entries = table_entries(t1, 1, HI >> 12)
    entries[0] = leaf(s, spare(s, byte=0xA7))
    s.u32(entry_gpa(t1, HI, 0), s.pfn(write_table(s, entries)))
    g.open_gate()
    g.settle(st, g.f, "task 2 reads A and B")
    run_either(t1, blit(cp(b, out, 0, 0x20, 0x20)), [(b, 0), (out, 0)], "task 1 reads B")
    t1.dump(ova, 0x40, "task 1's reads")
    t2.dump(o2va, 0x40, "task 2's reads")
    t2.delete()
    t1.delete()
    s.events("end")
    return s


# ---- map and unmap (57, 34) under live work ----

@scenario
def interplay_memlive_held_unmap():
    """An exec that fills a live shared buffer S and a shared buffer N no
    exec has named waits on the gate while the worker FIFO unmaps (34) both
    pages. In the blocks setup the unmap calls the host's unmap_memory,
    which replaces the host task's pages at those VAs; in the map setup and
    in the service it only unmaps object lists that overlap
    (docs/testing.md section 6.10). Released, S fills the pages it was made
    over and N is made (in the blocks setup the range tracker maps its page
    again). A MapMemory2 (57) on the worker, and fills through both again,
    follow. Every fill lands in the guest pages in every setup: the blocks
    setup's Metal buffer over S keeps the guest page it was made over after
    the host task's VA is replaced."""
    s = _scn(interplay_memlive_held_unmap)
    g = Rig(s, pages=24)
    t = g.t
    sh, shva = t.shared(contents=b"\xC1" * PAGE, listed=False)
    n, nva = t.shared(contents=b"\xC2" * PAGE, listed=False)
    t.ready()
    run_either(t, blit(B.fill(sh, 0, 0x10, 0xC3)), [(sh, 0)], "S made")
    st = g.hold(blit(B.fill(sh, 0x10, 0x10, 0xC4) + B.fill(n, 0x10, 0x10, 0xC5)), [(sh, 0), (n, 0)], "fills")
    g.work(abi.OP_UNMAP_MEMORY, struct.pack("<IQQ", 1, shva, 2 * PAGE), "34 on the worker")
    g.release(st, "released")
    t.dump(shva, 0x30, "S's page")
    t.dump(nva, 0x30, "N's page")
    g.work(abi.OP_MAP_MEMORY2, struct.pack("<IQQ", 1, shva, 2 * PAGE), "57 on the worker")
    run_either(t, blit(B.fill(sh, 0x20, 0x10, 0xC6) + B.fill(n, 0x20, 0x10, 0xC7)), [(sh, 0), (n, 0)], "after 57")
    t.dump(shva, 0x30, "S's page after 57")
    t.dump(nva, 0x30, "N's page after 57")
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_unmap_retarget_map():
    """The guest's sequence for moving a live buffer's page: UnmapMemory
    (34) of its VA, the leaf pointed at another page, MapMemory2 (57) of
    the VA. A fill through the live buffer S still lands in its old page
    in every setup; a new buffer N over the same VA lands in the new page
    in every setup (in the blocks setup 57 mapped the new page into the
    host task). S discarded (63) and listed with invalidateHost is then
    made over the new page too."""
    s = _scn(interplay_memlive_unmap_retarget_map)
    s.bring_up()
    t = Lab(s, pages=24)
    sh, shva = t.shared(contents=b"\xD1" * PAGE, listed=False)
    n = t.add(abi.T_BUFFER, struct.pack("<QIB3x", PAGE, shva // PAGE, 0))
    t.ready()
    run_either(t, blit(B.fill(sh, 0, 0x10, 0xD2)), [(sh, 0)], "S made")
    t.unmap(shva, PAGE, label="34")
    new = spare(s, byte=0xD9)
    set_leaf(t, shva // PAGE, new)
    t.map(shva, PAGE, label="57")
    run_either(t, blit(B.fill(sh, 0x10, 0x10, 0xD3)), [(sh, 0)], "S")
    run_either(t, blit(B.fill(n, 0x20, 0x10, 0xD4)), [(n, 0)], "N over the same VA")
    s.submit(abi.OP_DISCARD_RESOURCES, refs_payload(1, [sh]), label="63: S")
    run_either(t, blit(B.fill(sh, 0x30, 0x10, 0xD5)), [(sh, IH)], "S made again")
    s.dump(s.pa(t.leaves[shva // PAGE]), 0x40, "the old page")
    s.dump(s.pa(new), 0x40, "the new page")
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_held_list_page_moved():
    """The object list's page moved while execs that resolve references
    through it wait on the gate. The list lives at VA 0; the guest writes
    a second list in another page, the same but for entries 1 and 2, which
    name buffers over other pages. A first exec maps the list. Exec 1
    (reference 1, never made) waits while the guest points VA 0's leaf at
    the second list: released, it resolves reference 1 through the list as
    mapped at its first use, so the old list's buffer is made: the list is
    read in place through a mapping kept until an unmap or another 51, not
    through the table. Exec 2 (reference 2) waits while the worker FIFO
    unmaps (34) VA 0, which drops the list's mapping (unmapOverlapping):
    released, the list is mapped again through the table, and reference 2
    resolves through the second list."""
    from pgtest.task import buffer_record
    s = _scn(interplay_memlive_held_list_page_moved)
    g = Rig(s, pages=24)
    t = g.t
    r1, _ = t.shared(contents=b"\xE8" * PAGE, va=8 * PAGE, listed=False)
    r2, _ = t.shared(contents=b"\xEA" * PAGE, va=10 * PAGE, listed=False)
    out, ova = t.shared(va=12 * PAGE, listed=False)
    t.write(9 * PAGE, b"\xE9" * PAGE)
    t.write(11 * PAGE, b"\xEB" * PAGE)
    alt = []
    for ref, page in ((r1, 9), (r2, 11)):
        va = t.ol._rec
        t.write(va, buffer_record(PAGE, page))
        t.ol._rec += 16
        alt.append((ref, va))
    t.ready()
    entries = list(t.ol.entries)
    for ref, va in alt:
        entries[ref] = (abi.T_BUFFER, 16, va)
    other = spare(s, data=b"".join(abi.OBJ_ENTRY.pack(ty | (n << 8), va) for ty, n, va in entries))
    cp = B.copy_buffer_to_buffer
    run_either(t, blit(B.fill(out, 0x40, 0x10, 0xE0)), [(out, 0)], "an exec maps the list")
    st = g.hold(blit(cp(r1, out, 0, 0, 0x20)), [(r1, 0), (out, 0)], "exec 1: reference 1")
    set_leaf(t, 0, other)
    g.release(st, "exec 1")
    st = g.hold(blit(cp(r2, out, 0, 0x20, 0x20)), [(r2, 0), (out, 0)], "exec 2: reference 2")
    g.work(abi.OP_UNMAP_MEMORY, struct.pack("<IQQ", 1, 0, PAGE), "34 of VA 0 on the worker")
    g.release(st, "exec 2")
    t.dump(ova, 0x50, "reads")
    t.delete()
    s.events("end")
    return s


# ---- pages moving between RAM ranges ----

HI_RAM = 0x200000000  # a second RAM range, at 8 GiB
HI_PAGES = 32


def _hi_ram():
    from pgtest.scenario import RAM_BASE, RAM_SIZE
    return [(RAM_BASE, RAM_SIZE), (HI_RAM, HI_PAGES * PAGE)]


def _hi_page(i):
    """Page i of the second range, as a page index from RAM_BASE."""
    from pgtest.scenario import RAM_BASE
    return (HI_RAM - RAM_BASE) // PAGE + i


@scenario
def interplay_memlive_pages_move_ranges():
    """Guest RAM in two ranges (the first at 1 GiB, 32 pages at 8 GiB),
    each its own range of the memory map (map, isolate) and of the host's
    RAM (blocks). Task 1 in the first range reads its four kinds of object;
    the guest writes each object's bytes, changed, into a page of the
    second range and points the leaf there, and moves the command-buffer
    page too (a segment that fills out). Live objects keep their first-range
    pages; a discard (63) and invalidateHost make them again over the second
    range in the map setup and in the service, and not in the blocks setup,
    whose host task keeps the old pages until an unmap (34), after which
    they follow too. The moved command buffer runs from the second range in
    every setup."""
    s = _scn(interplay_memlive_pages_move_ranges, ram=_hi_ram())
    t, f = _four(s)
    f.read(0, 0, "before")
    k = 0
    for va, b in zip(f.vas(), (0xB1, 0xB2, 0xB3, 0xB4)):
        p = _hi_page(k)
        k += 1
        s.fill(s.pa(p), PAGE, b)
        set_leaf(t, va // PAGE, p)
    f.read(1, 0, "moved, live")
    drop = [f.sh, f.pr, f.h, f.tx]
    s.submit(abi.OP_DISCARD_RESOURCES, refs_payload(1, drop), label="63")
    f.read(2, IH, "after 63, invalidateHost")
    lo = min(f.vas())
    t.unmap(lo, max(f.vas()) + PAGE - lo, label="34 over the four pages")
    s.submit(abi.OP_DISCARD_RESOURCES, refs_payload(1, drop), label="63 again")
    f.read(3, IH, "after 34 and 63, invalidateHost")
    cmd = seg(abi.ENC_BLIT, B.fill(f.out, 0x400, 0x20, 0xBC))
    s.mem(s.pa(_hi_page(k)), cmd)
    set_leaf(t, 7, _hi_page(k))
    either(s, E3, exec_payload(1, [(7 * PAGE, len(cmd))], [(f.out, 0)]), "the command buffer in the second range")
    t.dump(f.ova, 0x420, "reads")
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_held_pages_move_ranges():
    """A task whose task root, page tables and pages are all in the second
    RAM range (8 GiB), defined beside task 1 in the first. Task 1's first
    reads of a shared buffer, a private buffer and a texture wait on the
    gate while the guest moves their pages into the second range, onto the
    very pages task 2's buffers name. Released, task 1 makes them there,
    and task 2, on the worker FIFO, fills its buffer: task 1's next read
    shows the fill (both tasks' buffers are the same RAM) in every
    setup."""
    from pgtest.gaps_memory import Pool, placed_task
    s = _scn(interplay_memlive_held_pages_move_ranges, ram=_hi_ram())
    g = Rig(s, pages=32)
    t1 = g.t
    sh, shva = t1.shared(contents=b"\x51" * PAGE, listed=False)
    pr, prva = t1.shared(contents=b"\x52" * PAGE, flags=1, listed=False)
    tx, txva = tex2d(t1, contents=texels(4, 4, seed=5))
    out, ova = t1.shared(listed=False)
    t1.ready()
    pool = Pool(_hi_page(i) for i in range(2, 4))
    leaves = {v: _hi_page(4 + v) for v in range(12)}
    t2 = placed_task(s, pool, cls=Lab, tid=2, pages=12, leaves=leaves)
    b2, b2va = t2.shared(va=8 * PAGE, listed=False)
    for v in range(8, 12):
        s.fill(s.pa(leaves[v]), PAGE, 0x60 + v)
    g.work(abi.OP_DEFINE_TASK2, define_payload(2, t2.length, s.pfn(t2.root)), "56 for task 2 on the worker")
    set_list_on(g, t2)
    cp = B.copy_buffer_to_buffer
    st = g.hold(blit(cp(sh, out, 0, 0, 0x20) + cp(pr, out, 0, 0x20, 0x20) + tex_out(tx, out, 0x40)),
                [(sh, 0), (pr, 0), (tx, 0), (out, 0)], "task 1's first reads")
    set_leaf(t1, shva // PAGE, leaves[8])
    set_leaf(t1, prva // PAGE, leaves[9])
    set_leaf(t1, txva // PAGE, leaves[10])
    g.release(st, "released")
    g.work(E3, exec_payload(2, t2.place(blit(B.fill(b2, 0x10, 0x10, 0x6F))), [(b2, 0)]), "task 2 fills on the worker")
    run_either(t1, blit(cp(sh, out, 0, 0x80, 0x20)), [(sh, 0), (out, 0)], "task 1 reads again")
    t1.dump(ova, 0xA0, "task 1's reads")
    t2.delete()
    t1.delete()
    s.events("end")
    return s


def set_list_on(g, t):
    """SetObjectList (51) for t on the worker FIFO."""
    t.ol.write()
    g.work(abi.OP_SET_OBJECT_LIST, struct.pack("<III", t.tid, t.ol.list_va // PAGE, t.ol.length()),
           "51 for task %d on the worker" % t.tid)


@scenario(setups=("map", "isolate"))
def interplay_memlive_migration_drops_range():
    """A suspend, and a resume on a fresh device (a migration) whose memory
    map lists only the first RAM range: task 1, in the first range, has a
    shared buffer whose page is in the second (8 GiB) range, used before.
    rvg_device_did_resume rebuilds the task from its root word, and the buffer,
    made again at its next use, finds its guest page in no range of the new map
    and faults, while a buffer in the first range works. The map and isolate
    setups only: the blocks setup has no memory map (its host calls map guest
    RAM through callbacks). The memory map is fixed at device creation
    (RvgDeviceConfig.memory_map, copied at rvg_device_create; QEMU builds it
    once, apple-gfx.c:505-513), so only a migration can drop a range."""
    from pgtest.lifecycle_util import suspend
    from pgtest.scenario import RAM_BASE, RAM_SIZE
    s = _scn(interplay_memlive_migration_drops_range, ram=_hi_ram(), setups=("map", "isolate"))
    s.bring_up()
    leaves = {v: s.alloc_page() for v in range(12)}
    leaves[9] = _hi_page(0)
    t = Lab(s, pages=12, leaves=leaves)
    a, ava = t.shared(contents=b"\x71" * PAGE, va=8 * PAGE, listed=False)
    b, bva = t.shared(va=9 * PAGE, listed=False)
    s.fill(s.pa(_hi_page(0)), PAGE, 0x72)
    t.ready()
    cp = B.copy_buffer_to_buffer
    run_either(t, blit(cp(b, a, 0, 0, 0x10)), [(a, 0), (b, 0)], "before the migration")
    suspend(s)
    s.api("recreate", config={"op": "config", "descriptor": {
        "map_ranges": [[hex(RAM_BASE), hex(RAM_SIZE), hex(RAM_BASE)]]}})
    s.api("willResume", slot="s", label="resume")
    s.api("didResume")
    run_either(t, blit(B.fill(a, 0x10, 0x10, 0x73)), [(a, 0)], "A, in the first range")
    run_either(t, blit(cp(b, a, 0, 0x20, 0x10)), [(a, 0), (b, 0)], "B, in the dropped range")
    t.dump(ava, 0x30, "A")
    t.delete()
    s.events("end")
    return s


# ---- a task that grows ----

def _grown_table(t, extra, override=None):
    """A depth-2 table for t's VA pages 0-15 as t maps them, plus `extra`
    ({vpn: RAM page}) and `override` ({vpn: RAM page}): a root page whose
    entry 0 names one last-level page. Returns the root page."""
    s = t.s
    entries = {}
    pages = dict(t.leaves)
    pages.update(override or {})
    for v, p in pages.items():
        entries[v] = leaf(s, p)
    for v, p in extra.items():
        entries[v] = leaf(s, p)
    last = write_table(s, entries)
    return write_table(s, {0: s.pfn(last)})


@scenario
def interplay_memlive_grow_refused():
    """A task of 16 pages (depth 1) with live objects asks to grow:
    DefineTask2 (56) again for its ID with 32 pages is refused ("Host task
    already exists") and the objects carry on. A buffer at VA page 20 is
    past the task's end and faults. The guest then rewrites the task root
    in place, depth 2 and a new table root that maps 32 pages (VA page 9,
    buffer B's, to another page) and length 32 pages at +0x14: nothing
    changes, because the table's root and depth are read once, at define,
    and the length is the host's: B, first used now, is made over the old
    table's leaf, and page 20 still faults. Deleted (32) and defined again
    over the rewritten root with 32 pages, the task is 32 pages deep 2: B
    is made over the new leaf, and the buffer at page 20 runs."""
    s = _scn(interplay_memlive_grow_refused)
    s.bring_up()
    t = Lab(s, pages=16)
    a, ava = t.shared(contents=b"\x21" * PAGE, va=8 * PAGE, listed=False)
    b, bva = t.shared(contents=b"\x22" * PAGE, va=9 * PAGE, listed=False)
    out, ova = t.shared(va=10 * PAGE, listed=False)
    far = t.add(abi.T_BUFFER, struct.pack("<QIB3x", PAGE, 20, 0))
    t.ready()
    cp = B.copy_buffer_to_buffer
    run_either(t, blit(cp(a, out, 0, 0, 0x20)), [(a, 0), (out, 0)], "A")
    either(s, abi.OP_DEFINE_TASK2, define_payload(1, 32 * PAGE, s.pfn(t.root)), "56 again, 32 pages")
    run_either(t, blit(cp(a, out, 0, 0x20, 0x20)), [(a, 0), (out, 0)], "A after the refusal")
    run_either(t, blit(cp(out, far, 0, 0, 0x20)), [(far, 0), (out, 0)], "page 20, past the end")
    root = _grown_table(t, {20: spare(s, byte=0x2F)}, {9: spare(s, byte=0x29)})
    s.mem(s.pa(t.root), abi.TASK_ROOT.pack(s.pfn(root), 2, 0, t.ol.length(), 32 * PAGE, 0, 0))
    run_either(t, blit(cp(b, out, 0, 0x40, 0x20)), [(b, 0), (out, 0)], "B, the root rewritten in place")
    run_either(t, blit(cp(far, out, 0, 0x60, 0x20)), [(far, 0), (out, 0)], "page 20, the root rewritten")
    t.delete()
    either(s, abi.OP_DEFINE_TASK2, define_payload(1, 32 * PAGE, s.pfn(t.root)), "56 after 32, 32 pages")
    s.dump(s.pa(t.root), 0x28, "the task root")
    run_either(t, blit(cp(b, out, 0, 0x80, 0x20)), [(b, 0), (out, 0)], "B in the grown task")
    run_either(t, blit(cp(far, out, 0, 0xA0, 0x20)), [(far, 0), (out, 0)], "page 20 in the grown task")
    t.dump(ova, 0xC0, "reads")
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_grow_held():
    """An exec whose command buffer is at VA page 21 and which fills a
    buffer at page 24, both past the end of a 16-page task, waits on the
    gate while the worker FIFO deletes the task (32) and defines it again
    (56) over its root, which the guest has rewritten meanwhile: depth 2,
    32 pages, pages 0-15 as before and pages 21 and 24 mapped. The exec
    names only a task ID, VAs and references: released, it runs on the
    grown task (its object list made again from the root's fields),
    reading its command buffer at the new end. A second exec at the new
    end follows."""
    s = _scn(interplay_memlive_grow_held)
    g = Rig(s, pages=16)
    t = g.t
    far = t.add(abi.T_BUFFER, struct.pack("<QIB3x", PAGE, 24, 0))
    out, ova = t.shared(va=10 * PAGE, listed=False)
    t.ready()
    run_either(t, blit(B.fill(out, 0, 0x10, 0x31)), [(out, 0)], "out, in the small task")
    cmd = seg(abi.ENC_BLIT, B.fill(far, 0, 0x20, 0x32) + B.fill(out, 0x10, 0x10, 0x33))
    cpage, fpage = spare(s, data=cmd), spare(s, byte=0x30)
    st = g.hold_on(g.f, t, [(21 * PAGE, len(cmd))], [(far, 0), (out, 0)], "the exec at page 21")
    root = _grown_table(t, {21: cpage, 24: fpage})
    s.mem(s.pa(t.root), abi.TASK_ROOT.pack(s.pfn(root), 2, 0, t.ol.length(), 0, 0, 0))
    g.work(abi.OP_DELETE_TASK, struct.pack("<I", 1), "32 on the worker")
    g.work(abi.OP_DEFINE_TASK2, define_payload(1, 32 * PAGE, s.pfn(t.root)), "56 on the worker: 32 pages")
    g.open_gate()
    g.settle(st, g.f, "released")
    either(s, E3, exec_payload(1, t.place(blit(B.copy_buffer_to_buffer(far, out, 0, 0x20, 0x20))),
                               [(far, 0), (out, 0)]), "a read at the new end")
    t.dump(ova, 0x40, "out")
    s.dump(s.pa(fpage), 0x20, "page 24")
    t.delete()
    s.events("end")
    return s


# ---- invalidation flags and page moves ----

@scenario
def interplay_memlive_retarget_flags_exec2():
    """interplay_memlive_retarget_flags and _discard's sequence twice in one
    task, over two sets of the four kinds of object: once through
    ExecIndirect3 (67, flags as bits) and once through ExecIndirect2 (55,
    one byte a flag, each cut to its low bit: a byte of 3 invalidates the
    host and a byte of 2 does nothing, docs/testing.md section 6.12). Each
    set is read before its leaves move, after (flags 0), with
    invalidateHost, after a discard (63) with flags 0, and with
    invalidateHost again. The two sets' reads are equal in every setup
    (compared), and set 1's are dumped."""
    s = _scn(interplay_memlive_retarget_flags_exec2)
    s.bring_up()
    t = Lab(s, pages=64)
    f3, f2 = Four(t), Four(t)
    t.ready()
    E2 = abi.OP_EXEC_INDIRECT2

    def read(f, slot, opcode, flags, label):
        recs = f.read_recs(slot)
        res = [(r, *flags) if opcode == E2 else (r, flags) for r in (f.sh, f.pr, f.h, f.tx)]
        res += [(f.hb, 0, 0, 0) if opcode == E2 else (f.hb, 0), (f.out, 0, 0, 0) if opcode == E2 else (f.out, 0)]
        run_either(t, blit(recs), res, label, opcode=opcode)

    for f, opcode, ih, zero in ((f3, E3, IH, 0), (f2, E2, (3, 0, 0), (2, 0, 0))):
        tag = "67" if opcode == E3 else "55"
        read(f, 0, opcode, zero, tag + ": before")
        _retarget(t, f)
        read(f, 1, opcode, zero, tag + ": flags 0")
        read(f, 2, opcode, ih, tag + ": invalidateHost")
        s.submit(abi.OP_DISCARD_RESOURCES, refs_payload(1, [f.sh, f.pr, f.h, f.tx]), label=tag + ": 63")
        read(f, 3, opcode, zero, tag + ": flags 0 after 63")
        read(f, 4, opcode, ih, tag + ": invalidateHost after 63")
    cmp_va(s, t, f3.ova, t, f2.ova, 0x500, "55's reads == 67's")
    t.dump(f3.ova, 0x500, "67's reads")
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_retarget_invalidate_52():
    """The four kinds of object are filled with flags 0 (0x51-0x53 at
    +0x40, the texture blue), their leaves move, and InvalidateResources
    (52) clears each one's host-valid bit. An exec with flags 0 then pages
    every host copy again from the pages bound at first use, not the new
    ones: the private buffer's, the heap's and the texture's
    unsynchronized fills are overwritten by the old guest bytes
    in-process, and the texture's in the service (its buffers and heap are
    the guest pages, which hold their fills). A 52 clearing guest-valid
    and a SynchronizeResources (53) then write each host copy back into
    the same old pages; the shared buffer's fill is in its old page in
    every setup, and the new pages keep the guest's bytes."""
    s = _scn(interplay_memlive_retarget_invalidate_52)
    t, f = _four(s)
    old = f.old_pages()
    f.fill(0x51, BLUE, 0, "fills, flags 0")
    new = _retarget(t, f)
    refs = [f.sh, f.pr, f.h, f.tx]
    from pgtest.lifetime_util import invalidate_payload
    s.submit(abi.OP_INVALIDATE_RESOURCES, invalidate_payload(1, [(r, (1, 0, 0, 0)) for r in refs]),
             label="52: clearHostValid")
    f.read(0, 0, "flags 0 after 52")
    s.submit(abi.OP_INVALIDATE_RESOURCES, invalidate_payload(1, [(r, (0, 0, 1, 0)) for r in refs]),
             label="52: clearGuestValid")
    either(s, abi.OP_SYNCHRONIZE_RESOURCES, refs_payload(1, refs), "53")
    t.dump(f.ova, 0xA0, "read")
    f.pages(old, "old")
    f.pages(new, "new")
    t.delete()
    s.events("end")
    return s


@scenario
def interplay_memlive_migrate_protocol():
    """A guest moving three objects' pages the way that works in every
    setup, checked against a control task that never moves them. Both tasks
    fill a shared buffer, a private buffer and a private heap's buffer with
    invalidateGuest and synchronize (6), so the guest pages hold the GPU's
    bytes; task 1 then copies each object into a helper buffer over a
    fresh page (the GPU copies the host copy), points the objects' leaves
    at the fresh pages, unmaps (34) the old VAs (the blocks setup's host
    task maps them again at their next use), discards the objects (63),
    and scribbles over the old pages, as a guest reusing them would. Both
    tasks then fill again and read everything out with invalidateHost: the
    two tasks' reads are equal in every setup (compared)."""
    s = _scn(interplay_memlive_migrate_protocol)
    s.bring_up()
    cp = B.copy_buffer_to_buffer

    def build(tid):
        t = Lab(s, tid=tid, pages=32)
        o = {}
        o["sh"], o["shva"] = t.shared(contents=pattern_bytes(1), listed=False)
        o["pr"], o["prva"] = t.shared(contents=pattern_bytes(2), flags=1, listed=False)
        o["hva"] = t.reserve(PAGE)
        t.write(o["hva"], pattern_bytes(3))
        o["h"] = t.add(abi.T_HEAP, heap_rec(o["hva"] // PAGE, PAGE, 1))
        o["hb"] = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(o["h"], 0, PAGE))
        o["m"] = [t.shared(listed=False) for _ in range(3)]
        o["out"], o["ova"] = t.shared(length=PAGE, listed=False)
        t.ready()
        return t, o

    def work(t, o, byte, flags, label):
        recs = B.fill(o["sh"], 0x100, 0x40, byte) + B.fill(o["pr"], 0x140, 0x40, byte + 1) + B.fill(
            o["hb"], 0x180, 0x40, byte + 2)
        run_either(t, blit(recs), [(o["sh"], flags), (o["pr"], flags), (o["h"], flags), (o["hb"], 0)], label)

    def read(t, o, label):
        recs = cp(o["sh"], o["out"], 0, 0, 0x200) + cp(o["pr"], o["out"], 0, 0x200, 0x200) + cp(
            o["hb"], o["out"], 0, 0x400, 0x200)
        run_either(t, blit(recs), [(o["sh"], IH), (o["pr"], IH), (o["h"], IH), (o["hb"], 0), (o["out"], 0)], label)

    t1, o1 = build(1)
    t2, o2 = build(2)
    for t, o in ((t1, o1), (t2, o2)):
        work(t, o, 0x40, IG | SY, "task %d: fills, written back" % t.tid)
    mrefs = [m for m, _ in o1["m"]]
    recs = cp(o1["sh"], mrefs[0], 0, 0, PAGE) + cp(o1["pr"], mrefs[1], 0, 0, PAGE) + cp(o1["hb"], mrefs[2], 0, 0, PAGE)
    run_either(t1, blit(recs), [(o1["sh"], 0), (o1["pr"], 0), (o1["h"], 0), (o1["hb"], 0)] + [(m, 0) for m in mrefs],
               "task 1 copies each object to its new page")
    olds = []
    for va, (_, mva) in zip((o1["shva"], o1["prva"], o1["hva"]), o1["m"]):
        olds.append(t1.leaves[va // PAGE])
        set_leaf(t1, va // PAGE, t1.leaves[mva // PAGE])
        t1.unmap(va, PAGE, label="34 at VA page %d" % (va // PAGE))
    s.submit(abi.OP_DISCARD_RESOURCES, refs_payload(1, [o1["sh"], o1["pr"], o1["h"]]), label="63: task 1's three")
    for p in olds:
        s.fill(s.pa(p), PAGE, 0xEE)
    for t, o in ((t1, o1), (t2, o2)):
        work(t, o, 0x50, IH, "task %d: fills again" % t.tid)
        read(t, o, "task %d: reads" % t.tid)
    cmp_va(s, t1, o1["ova"], t2, o2["ova"], 0x600, "task 1 == task 2")
    t2.delete()
    t1.delete()
    s.events("end")
    return s


def pattern_bytes(seed):
    return bytes((i * 7 + seed * 31 + (i >> 8)) & 0xFF for i in range(PAGE))
