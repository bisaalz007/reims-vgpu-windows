"""Task memory as the command paths read it: command buffers over pages the
host sees apart (scattered, or in other RAM ranges), buffers over read-only
leaves, and a command buffer over a page outside guest RAM, which only the
service reads in place (docs/testing.md, section 6.10)."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.gaps_memory import page_of
from pgtest.registry import scenario
from pgtest.iosfc_guest import CMD_MAP, IOSFC, desc_v2
from pgtest.resources_build import RTask, b_copy_buf_buf, b_fill_buf, mapper_ref_rec
from pgtest.scenario import RAM_BASE, RAM_SIZE, Scenario
from pgtest.task import rec, seg
from pgtest.tasks_lib import (XGPUTask, exec3_payload, fault_and_resume, fill_rec, maybe_fault,
                              refs_payload)

R1 = (RAM_BASE + RAM_SIZE + 4 * PAGE, 4 * PAGE)  # a second RAM range, after a hole


def fills(ref, base, byte, n=3):
    """n fills of 0x10 bytes of buffer ref at base, base + 0x10...: 0x20
    bytes a record."""
    return b"".join(fill_rec(ref, base + 0x10 * i, 0x10, byte + i) for i in range(n))


@scenario
def service_gap_cmdbuf_scattered():
    """Command buffers that cross a page boundary where the next page is
    elsewhere: VA pages 4, 5 and 6 are a first-range page, a page of a second
    RAM range, and a first-range page far from the first. Each command buffer
    (one blit segment of three fills, the middle record straddling the
    boundary) is read as two pieces: the service's cursor starts a new range
    where the host addresses stop being adjacent, in-process the copy walks
    page by page. A third command buffer crosses physically adjacent pages
    and is one piece. Every fill lands in every setup."""
    s = Scenario("service_gap_cmdbuf_scattered", ram=[(RAM_BASE, RAM_SIZE), R1],
                 doc=service_gap_cmdbuf_scattered.__doc__)
    s.bring_up()
    leaves = {v: s.alloc_page() for v in range(4)}
    far = s.alloc_page(40) + 39
    leaves.update({4: s.alloc_page(), 5: page_of(R1[0]) + 1, 6: far})
    pair = s.alloc_page(2)
    leaves.update({7: s.alloc_page(), 8: s.alloc_page(), 9: pair, 10: pair + 1})
    t = XGPUTask(s, tid=1, pages=11, leaves=leaves)
    buf, _ = t.buffer(length=PAGE, va=8 * PAGE)
    t.define()
    t.install()
    cbs = []
    for n, va in enumerate((5 * PAGE - 0x38, 6 * PAGE - 0x18, 10 * PAGE - 0x30)):
        data = seg(abi.ENC_BLIT, fills(buf, 0x40 * n, 0x10 * (n + 1)))
        t.write(va, data)
        cbs.append((va, len(data)))
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, cbs, [(buf, 0)]), label="three command buffers")
    t.dump(8 * PAGE, 0xC0, "the buffer")
    t.delete()
    s.events("end")
    return s


@scenario
def service_gap_readonly_leaves():
    """Buffers over leaves without the writable bit (bit 31), read by a copy
    into a writable buffer: one over a read-only page, and one over a
    writable page and a read-only one. Mapping takes the whole range
    read-only when any page is (in the service, its address ranges report
    readOnly), and the copies read the guest's bytes in every setup."""
    s = Scenario("service_gap_readonly_leaves", doc=service_gap_readonly_leaves.__doc__)
    s.bring_up()
    t = XGPUTask(s, tid=1, pages=12, writable={v for v in range(12) if v not in (9, 11)})
    ro, ro_va = t.buffer(length=PAGE, va=9 * PAGE, contents=bytes(range(0x40)))
    mixed, mixed_va = t.buffer(length=2 * PAGE, va=10 * PAGE)
    t.write(mixed_va + PAGE - 0x20, bytes(0xA0 + i for i in range(0x40)))
    out, out_va = t.buffer(length=PAGE, va=8 * PAGE)
    t.define()
    t.install()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, b_copy_buf_buf(ro, out, 0, 0, 0x40)
                      + b_copy_buf_buf(mixed, out, PAGE - 0x20, 0x40, 0x40)))
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(ro, 0), (mixed, 0), (out, 0)]),
             label="copies out of read-only leaves")
    t.dump(out_va, 0x80, "the copies")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",))
def service_gap_cmdbuf_outside_ram():
    """A command buffer whose second page's leaf names a page outside guest
    RAM. The service reads command buffers in place: the memory map has no
    host address for that page (its lookup returns NULL), the cursor's second
    range starts at that NULL, and decoding the record that crosses into it
    kills the service: the exec faults, and so does the next command for that
    task, which a live service would have run. A task defined after it gets a
    new service and runs. Isolate only: in-process the command buffer is
    copied, the page's read fails and writes nothing, and the decoder reads
    the copy's zeroed tail, which no scenario locks (docs/testing.md,
    section 7.5)."""
    s = Scenario("service_gap_cmdbuf_outside_ram", setups=("isolate",),
                 doc=service_gap_cmdbuf_outside_ram.__doc__)
    s.bring_up()
    outside = page_of(RAM_BASE + 2 * RAM_SIZE)
    leaves = {v: s.alloc_page() for v in range(10)}
    leaves[5] = outside
    t = XGPUTask(s, tid=1, pages=10, leaves=leaves)
    buf, _ = t.buffer(length=PAGE, va=8 * PAGE)
    t.define()
    t.install()
    data = seg(abi.ENC_BLIT, fills(buf, 0, 0x21))
    va = 5 * PAGE - 0x28
    t.write(va, data[:0x28])
    st = s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [(va, len(data))], [(buf, 0)]), fault=True,
                  label="a command buffer into a page outside RAM")
    fault_and_resume(s, "a command buffer into a page outside RAM")
    s.fifo.wait_stamp(st)
    t.dump(8 * PAGE, 0x40, "the buffer")
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(buf, 0, 0x10, 0x55)))
    st = s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(buf, 0)]), fault=True,
                  label="the task after its service died")
    fault_and_resume(s, "the task after its service died")
    s.fifo.wait_stamp(st)
    u = XGPUTask(s, tid=2, pages=10)
    ubuf, _ = u.buffer(length=PAGE, va=8 * PAGE)
    u.define()
    u.install()
    cb = u.cmdbuf(seg(abi.ENC_BLIT, fill_rec(ubuf, 0, 0x10, 0x66)))
    s.submit(abi.OP_EXEC_INDIRECT3, exec3_payload(2, [cb], [(ubuf, 0)]), label="a new task")
    u.dump(8 * PAGE, 0x20, "the new task's buffer")
    u.delete()
    s.events("end")
    return s


@scenario
def service_gap_placement_unmappable():
    """A task's first placement list set over a read-only leaf, with
    SetObjectAndPlacementList (68). In the service the placement list must be
    mapped writable; that mapping fails and is left unmapped, and since there
    was no earlier placement list to copy, the command still succeeds. The
    next exec must map it and cannot ("Failed to map placement list"), so it
    faults, while a synchronize (53) of a buffer, which maps no list,
    completes. A second 68 moving the placement list to a writable page maps
    the old one read-only to copy from and succeeds, and the exec then runs.
    In-process, 68 is accepted (the device has the argument-buffer feature)
    but the task's list manager has no argument buffers: it records the
    object list and never the placement list, and every exec runs."""
    s = Scenario("service_gap_placement_unmappable", doc=service_gap_placement_unmappable.__doc__)
    s.bring_up()
    t = XGPUTask(s, tid=1, pages=12, writable={v for v in range(12) if v != 9})
    buf, buf_va = t.buffer(length=PAGE, va=8 * PAGE)
    t.define()
    n = t.ol.write()
    maybe_fault(s, abi.OP_SET_OBJECT_AND_PLACEMENT_LIST, struct.pack("<IQQII", 1, 0, 9 * PAGE, n, 16 * 2),
                "68: the first placement list, over a read-only leaf")
    s.dump(s.pa(t.root, 8), 0x20, "the task root's list fields")
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(buf, 0, 0x10, 0x31)))
    maybe_fault(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(buf, 0)]), "an exec")
    maybe_fault(s, abi.OP_SYNCHRONIZE_RESOURCES, refs_payload(1, [buf]), "53 of the buffer")
    t.dump(buf_va, 0x20, "the buffer")
    maybe_fault(s, abi.OP_SET_OBJECT_AND_PLACEMENT_LIST, struct.pack("<IQQII", 1, 0, 10 * PAGE, n, 16 * 2),
                "68: the placement list moved to a writable page")
    s.dump(s.pa(t.root, 8), 0x20, "the task root's list fields, again")
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(buf, 0x10, 0x10, 0x42)))
    maybe_fault(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(buf, 0)]), "an exec, after the move")
    t.dump(buf_va, 0x20, "the buffer, after")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("map", "isolate"))
def service_gap_mapper_unknown_twice():
    """A mapper-ref buffer (type 15) over mapping ID 99, which nothing
    mapped, used by three execs, then one over mapped surface 1. The mapper
    has no surface for 99, so every exec using it faults ("Attempting to
    create mapper ref against invalid surface", resources_mapper_buffer). In
    the service each attempt asks the host again (the service caches only
    surfaces it got), and from the second on the host already holds an entry
    for 99 ("Failed to emplace surface"); the outcome is the same. The buffer
    over surface 1 still works. Map and isolate setups only: the IOSurface
    host device needs a memory map."""
    s = Scenario("service_gap_mapper_unknown_twice", setups=("map", "isolate"), iosfc=True,
                 doc=service_gap_mapper_unknown_twice.__doc__)
    s.bring_up()
    d = IOSFC(s, v2=True)
    d.start()
    root, data = d.surface_pages(1)
    s.mem(s.pa(data), bytes((i * 7) & 0xFF for i in range(0x100)))
    d.put_desc(1, desc_v2(root, 64, 16, fmt=b"BGRA", bpe=4))
    d.submit((CMD_MAP, 1))
    t = RTask(s)
    bad = t.add(abi.T_MAPPER_BUFFER, mapper_ref_rec(99))
    good = t.add(abi.T_MAPPER_BUFFER, mapper_ref_rec(1))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    for n in range(3):
        t.blit(b_fill_buf(b, 0x10 * n, 0x10, 0x10 + n), resources=[(bad, 0), (b, 0)], fault=True,
               label="mapping ID 99, attempt %d" % (n + 1))
    t.blit(b_copy_buf_buf(good, b, 0, 0x40, 0x40), resources=[(good, 0), (b, 0)], label="surface 1")
    t.dump(b_va, 0x80, "B")
    t.delete()
    s.events("end")
    return s
