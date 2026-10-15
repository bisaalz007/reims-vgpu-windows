"""Heaps (type 13) and buffers placed in them (type 14)
(docs/testing.md, sections 4.10, 6.11, 6.20).

In-process a heap is a legacy heap: a private placement heap whose bytes are
paged through one transfer buffer covering it; it must be private, and a
shared one fails silently. In the service on a detach-backing GPU it is a
detach heap over the guest pages, shared, whose buffers land in guest memory
at once."""
from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.resources_build import (RTask, b_copy_buf_buf, b_fill_buf, buffer_rec, cmd_discard, cmd_invalidate,
                                    cmd_sync_discard, cmd_synchronize, heap_buffer_rec, heap_rec)
from pgtest.scenario import Scenario

HEAP = 4 * PAGE


@scenario
def resources_heap_paging():
    """A private heap of four pages over patterned guest bytes, two buffers
    placed in it (type 14, at 0 and at 0x8000) and a tracked private heap
    (flag bit 1) with one. Listing the heap pages it in, so a copy out of a
    heap buffer shows the guest bytes at its offset; a fill of a heap buffer
    stays on the host until the heap is listed with flags 6, or marked with
    52 and synchronized with 53. The legacy heap's synchronize sets its host
    validity, not its guest validity, so a second 53 copies again. A heap
    buffer does not page (its prepare and synchronize are empty). 63 deletes
    an in-process heap buffer (it does not support discard) and discards the
    heap; 62 synchronizes and discards. In the service every write is in
    guest memory at once."""
    s = Scenario("resources_heap_paging", doc=resources_heap_paging.__doc__)
    s.bring_up()
    t = RTask(s, pages=80)
    h_va = t.area(HEAP, contents=bytes((i * 11) & 0xFF for i in range(0x100)))
    t.write(h_va + 0x8000, bytes((i * 13 + 5) & 0xFF for i in range(0x100)))
    h = t.add(abi.T_HEAP, heap_rec(h_va // PAGE, HEAP, flags=1))
    hb0 = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(h, 0, PAGE))
    hb1 = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(h, 0x8000, PAGE))
    tr_va = t.area(HEAP, fill=0x5C)
    tr = t.add(abi.T_HEAP, heap_rec(tr_va // PAGE, HEAP, flags=3))
    trb = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(tr, 0x4000, PAGE))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_buf_buf(hb0, b, 0, 0, 0x40), b_copy_buf_buf(hb1, b, 0, 0x40, 0x40),
           b_copy_buf_buf(trb, b, 0, 0x80, 0x40),
           resources=[(h, 0), (hb0, 0), (hb1, 0), (tr, 0), (trb, 0), (b, 0)], label="read three heap buffers")
    t.dump(b_va, 0xC0, "B")
    t.blit(b_fill_buf(hb0, 0, 0x10, 0xA1), b_fill_buf(hb1, 0x10, 0x10, 0xB2),
           resources=[(h, 0), (hb0, 0), (hb1, 0), (b, 0)], label="fill two heap buffers")
    t.dump(h_va, 0x20, "heap pages at 0")
    t.dump(h_va + 0x8000, 0x20, "heap pages at 0x8000")
    t.blit(b_fill_buf(hb0, 0x20, 0x10, 0xC3), resources=[(h, 6), (hb0, 0), (b, 0)],
           label="fill, the heap listed with flags 6")
    t.dump(h_va, 0x40, "heap pages at 0 after the synchronizing exec")
    t.blit(b_fill_buf(hb1, 0x20, 0x10, 0xD4), resources=[(h, 0), (hb1, 0), (b, 0)], label="fill again")
    cmd_invalidate(s, 1, [(h, 0, 0, 1, 0), (hb1, 0, 0, 1, 0)], label="52 on the heap and a heap buffer")
    cmd_synchronize(s, 1, [h, hb1], label="53")
    t.dump(h_va + 0x8000, 0x40, "heap pages at 0x8000 after 53")
    t.write(h_va + 0x8000, b"\xEE" * 0x10)
    cmd_synchronize(s, 1, [h], label="53 again")
    t.dump(h_va + 0x8000, 0x40, "heap pages at 0x8000 after a second 53")
    t.blit(b_fill_buf(trb, 0, 0x10, 0x7E), resources=[(tr, 6), (trb, 0), (b, 0)], label="the tracked heap")
    t.dump(tr_va + 0x4000, 0x20, "tracked heap pages at 0x4000")
    cmd_discard(s, 1, [hb0, h], label="63 on a heap buffer and its heap")
    t.blit(b_copy_buf_buf(hb0, b, 0, 0x100, 0x40), resources=[(h, 0), (hb0, 0), (b, 0)],
           label="a heap buffer after 63")
    t.dump(b_va + 0x100, 0x40, "B+0x100")
    cmd_sync_discard(s, 1, [tr, trb], label="62 on the tracked heap and its buffer")
    t.blit(b_copy_buf_buf(trb, b, 0, 0x140, 0x40), resources=[(tr, 0), (trb, 0), (b, 0)],
           label="the tracked heap's buffer after 62")
    t.dump(b_va + 0x140, 0x40, "B+0x140")
    t.delete()
    s.events("end")
    return s


@scenario
def resources_heap_records():
    """Heap and heap-buffer records, each its own object listed in its own
    exec: a short heap record; a shared heap (flags 0), which in-process
    returns nil silently ("Failed to create heap host resource"); a shared
    read-only heap (flag bit 2); private, tracked and read-only private
    heaps; a short heap-buffer record; heap-buffer references to nothing (0:
    a nil heap makes no buffer, "Failed to create heap buffer"), to an
    unknown object, to a buffer (which has no Resource::get_heap: its kind
    raises Unrecognized, a fault), to a heap that failed; a heap buffer past
    its heap's end; a good one. A heap used where a buffer is expected: the
    in-process heap has no Resource::get_buffer (a fault), the service's
    returns its cover buffer."""
    from scenarios.resources import creation_case
    H = abi.T_HEAP
    cases = [
        (H, heap_rec(8, HEAP, 1), 11, True),                 # 1 short
        (H, heap_rec(8, HEAP, 0), None, {"inproc": True, "service": False}),   # 2 shared
        (H, heap_rec(8, HEAP, 4), None, {"inproc": True, "service": False}),   # 3 shared read-only
        (H, heap_rec(8, HEAP, 1), None, False),               # 4 private
        (H, heap_rec(12, HEAP, 3), None, False),              # 5 private tracked
        (H, heap_rec(16, HEAP, 5), None, False),              # 6 private read-only
        (abi.T_HEAP_BUFFER, heap_buffer_rec(4, 0, PAGE), 0x13, True),
        (abi.T_HEAP_BUFFER, heap_buffer_rec(0, 0, PAGE), None, True),
        (abi.T_HEAP_BUFFER, heap_buffer_rec(40, 0, PAGE), None, True),
        (abi.T_HEAP_BUFFER, heap_buffer_rec(13, 0, PAGE), None, True),
        (abi.T_HEAP_BUFFER, heap_buffer_rec(2, 0, PAGE), None, {"inproc": True, "service": False}),
        (abi.T_HEAP_BUFFER, heap_buffer_rec(4, 0, PAGE), None, False),
        (abi.T_BUFFER, buffer_rec(PAGE, 20), None, False),    # 13 a buffer
    ]
    return creation_case("resources_heap_records", resources_heap_records.__doc__, cases)


def exec_expect(t, label, recs, res, faults):
    """One blit exec whose outcome the caller states per setup group."""
    t.blit(*recs, resources=res, fault=faults, label=label)


def _heap_setups(name, doc, inproc):
    sts = ("blocks", "map") if inproc else ("isolate",)
    s = Scenario(name, setups=sts, doc=doc)
    s.bring_up()
    t = RTask(s, pages=80)
    sh_va = t.area(HEAP, fill=0x31)
    sh = t.add(abi.T_HEAP, heap_rec(sh_va // PAGE, HEAP, 0))
    shb = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(sh, PAGE, PAGE))
    ro_va = t.area(HEAP, fill=0x42)
    ro = t.add(abi.T_HEAP, heap_rec(ro_va // PAGE, HEAP, 4))
    rob = t.add(abi.T_HEAP_BUFFER, heap_buffer_rec(ro, 0, PAGE))
    pr_va = t.area(HEAP, fill=0x53)
    pr = t.add(abi.T_HEAP, heap_rec(pr_va // PAGE, HEAP, 1))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    exec_expect(t, "a shared heap and its buffer", [b_fill_buf(shb, 0, 0x10, 0x99), b_copy_buf_buf(shb, b, 0, 0, 0x20)],
                [(sh, 0), (shb, 0), (b, 0)], inproc)
    t.dump(sh_va + PAGE, 0x20, "the shared heap's pages")
    exec_expect(t, "a read-only heap and its buffer", [b_copy_buf_buf(rob, b, 0, 0x20, 0x20)],
                [(ro, 0), (rob, 0), (b, 0)], inproc)
    exec_expect(t, "a private heap as a blit buffer", [b_copy_buf_buf(pr, b, 0, 0x40, 0x20)], [(pr, 0), (b, 0)], inproc)
    t.dump(b_va, 0x60, "B")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def resources_heap_setups_inproc():
    """A shared heap, a read-only heap and a heap used as a blit buffer,
    in-process: the first two are refused (a legacy heap must be private),
    and a legacy heap has no Resource::get_buffer (its kind raises
    Unrecognized, a fault). Compare resources_heap_setups_service."""
    return _heap_setups("resources_heap_setups_inproc", resources_heap_setups_inproc.__doc__, True)


@scenario(setups=("isolate",))
def resources_heap_setups_service():
    """The same records in the service: a shared detach heap over guest pages
    whose buffer's writes are in guest memory at once, a read-only heap
    (argument buffers are on), and a heap as a blit buffer, which is its
    cover buffer."""
    return _heap_setups("resources_heap_setups_service", resources_heap_setups_service.__doc__, False)
