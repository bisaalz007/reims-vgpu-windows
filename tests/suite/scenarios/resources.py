"""Buffers (type 1) and their data paths: shared buffers over guest pages,
private buffers paged through transfer copies, read-only buffers, and the
exec resource flags and FIFO commands that move their bytes
(docs/testing.md, sections 4.10, 6.11, 6.12).

The blit decoder is the probe: 306 FillBuffer and 301 CopyFromBufferToBuffer
put bytes into a buffer and copy them into a shared buffer, whose pages the
test reads."""
from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.resources_build import (RTask, b_copy_buf_buf, b_fill_buf, buffer_rec, cmd_discard, cmd_invalidate,
                                    cmd_replace_physical, cmd_sync_discard, cmd_synchronize)
from pgtest.scenario import Scenario

PATTERN = bytes((i * 7 + 3) & 0xFF for i in range(0x100))


@scenario
def resources_private_buffer():
    """A private buffer (type 1, flags bit 0). In-process it is a private
    buffer resource: an exec that lists it copies its guest pages in
    (prepare), GPU writes stay on the host until invalidateGuest (bit 1) plus
    synchronize (bit 2) copy them back, and invalidateHost (bit 0) re-reads
    the guest pages. In the service on a detach-backing GPU every buffer is a
    detach buffer over the guest pages, so GPU writes land in guest memory at
    once: a setup divergence."""
    s = Scenario("resources_private_buffer", doc=resources_private_buffer.__doc__)
    s.bring_up()
    t = RTask(s)
    a_va = t.area(PAGE, contents=PATTERN)
    a = t.add(abi.T_BUFFER, buffer_rec(PAGE, a_va // PAGE, flags=1))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_buf_buf(a, b, 0, 0, 0x100), resources=[(a, 0), (b, 0)], label="copy A to B")
    t.dump(b_va, 0x100, "B: A's guest bytes, paged in")
    t.blit(b_fill_buf(a, 0x10, 0x20, 0xCD), resources=[(a, 0), (b, 0)], label="fill A")
    t.dump(a_va, 0x100, "A's pages after a fill with no flags")
    t.blit(b_copy_buf_buf(a, b, 0, 0x100, 0x100), resources=[(a, 0), (b, 0)], label="copy A to B+0x100")
    t.dump(b_va + 0x100, 0x100, "B+0x100: the host copy of A")
    cmd_synchronize(s, 1, [a, b], label="synchronize A and B (guest valid: nothing to do)")
    t.dump(a_va, 0x100, "A's pages after opcode 53")
    t.blit(b_fill_buf(a, 0x40, 0x10, 0xEF), resources=[(a, 6), (b, 0)], label="fill A, invalidateGuest+synchronize")
    t.dump(a_va, 0x100, "A's pages after the synchronizing exec")
    t.write(a_va, b"\x55" * 8)
    t.blit(b_copy_buf_buf(a, b, 0, 0x200, 0x10), resources=[(a, 0), (b, 0)], label="copy without invalidateHost")
    t.blit(b_copy_buf_buf(a, b, 0, 0x210, 0x10), resources=[(a, 1), (b, 0)], label="copy with invalidateHost")
    t.dump(b_va + 0x200, 0x20, "B+0x200: before and after invalidateHost")
    t.delete()
    s.events("end")
    return s


@scenario
def resources_private_buffer_sync_commands():
    """A private buffer whose host copy the guest marks invalid for the guest
    with opcode 52 (clearGuestValid), then copies back with opcode 53
    (synchronizeResources) or 62 (which also discards: a private buffer
    supports discard, so it goes volatile and drops its transfer buffer).
    Opcode 63 discards; opcode 60 (replacePhysical) discards too. After a
    discard the next prepare devolatilizes and pages in again."""
    s = Scenario("resources_private_buffer_sync_commands", doc=resources_private_buffer_sync_commands.__doc__)
    s.bring_up()
    t = RTask(s)
    a_va = t.area(PAGE, contents=PATTERN)
    a = t.add(abi.T_BUFFER, buffer_rec(PAGE, a_va // PAGE, flags=1))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_buf(a, 0, 0x20, 0x11), resources=[(a, 0), (b, 0)], label="fill A with 0x11")
    cmd_invalidate(s, 1, [(a, 0, 0, 1, 0), (b, 0, 0, 1, 0), (0, 0, 0, 1, 0), (40, 0, 0, 1, 0)],
                   label="52: clear A's guest validity; B, 0 and 40 are ignored")
    cmd_synchronize(s, 1, [a, b, 0, 40], label="53: copies A back")
    t.dump(a_va, 0x40, "A's pages after 52 + 53")
    t.blit(b_fill_buf(a, 0x20, 0x20, 0x22), resources=[(a, 0), (b, 0)], label="fill A with 0x22")
    cmd_invalidate(s, 1, [(a, 0, 0, 1, 0)], label="52 again")
    cmd_sync_discard(s, 1, [a, b, 0], label="62: copy back, then discard")
    t.dump(a_va, 0x40, "A's pages after 62")
    t.write(a_va, b"\x77" * 0x10)
    t.blit(b_copy_buf_buf(a, b, 0, 0, 0x40), resources=[(a, 0), (b, 0)], label="copy A (re-paged after discard)")
    t.dump(b_va, 0x40, "B: A after the discard")
    cmd_discard(s, 1, [a, b, 0, 40], label="63: discard A and B")
    t.write(a_va + 0x10, b"\x88" * 0x10)
    t.blit(b_copy_buf_buf(a, b, 0, 0x40, 0x40), resources=[(a, 0), (b, 0)], label="copy A after 63")
    t.dump(b_va + 0x40, 0x40, "B+0x40")
    cmd_replace_physical(s, 1, a, label="60 on A")
    cmd_replace_physical(s, 1, 40, label="60 on an unknown reference")
    t.write(a_va + 0x20, b"\x99" * 0x10)
    t.blit(b_copy_buf_buf(a, b, 0, 0x80, 0x40), resources=[(a, 0), (b, 0)], label="copy A after 60")
    t.dump(b_va + 0x80, 0x40, "B+0x80")
    t.delete()
    s.events("end")
    return s


@scenario
def resources_shared_buffer_paths():
    """A shared buffer (type 1, flags 0) wraps the guest pages with no copy:
    GPU writes are in guest memory without any flag, a guest write is seen by
    the next exec without invalidateHost, opcode 53 does nothing (needsSync
    is NO), and a discard (63, 62, 60) drops the Metal buffer, which the next
    listing re-creates (needsCPUPaging, prepareWithCPU). A buffer used by a
    blit without being in the resource list after a discard throws "Getting
    buffer from invalidated resource" in-process."""
    s = Scenario("resources_shared_buffer_paths", doc=resources_shared_buffer_paths.__doc__)
    s.bring_up()
    t = RTask(s)
    a, a_va = t.shared(PAGE, contents=PATTERN)
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_buf(a, 0x100, 0x10, 0x5A), b_copy_buf_buf(a, b, 0, 0, 0x110),
           resources=[(a, 0), (b, 0)], label="fill and copy")
    t.dump(a_va + 0x100, 0x10, "A's pages: the fill, no flags")
    t.dump(b_va, 0x110, "B")
    t.write(a_va, b"\x42" * 0x10)
    t.blit(b_copy_buf_buf(a, b, 0, 0x200, 0x10), resources=[(a, 0), (b, 0)], label="copy a guest write")
    t.dump(b_va + 0x200, 0x10, "B+0x200")
    cmd_invalidate(s, 1, [(a, 1, 0, 1, 0)], label="52 on a shared buffer")
    cmd_synchronize(s, 1, [a, b], label="53 on shared buffers")
    cmd_sync_discard(s, 1, [a], label="62 on a shared buffer")
    t.blit(b_copy_buf_buf(a, b, 0x10, 0x300, 0x10), resources=[(a, 0), (b, 0)], label="copy after 62")
    t.dump(b_va + 0x300, 0x10, "B+0x300")
    cmd_discard(s, 1, [a], label="63 on a shared buffer")
    t.blit(b_copy_buf_buf(a, b, 0x20, 0x310, 0x10), resources=[(b, 0)], fault=True,
           label="A used after a discard, not listed")
    t.dump(b_va + 0x310, 0x10, "B+0x310")
    t.blit(b_copy_buf_buf(a, b, 0x20, 0x320, 0x10), resources=[(a, 0), (b, 0)], label="A listed again")
    t.dump(b_va + 0x320, 0x10, "B+0x320")
    t.delete()
    s.events("end")
    return s


@scenario
def resources_readonly_buffer():
    """A read-only buffer (type 1, flags bit 1). In-process the bit is
    ignored (a shared buffer); in the service a detach buffer takes the
    private 0x1000000 resource option when the features support argument
    buffers. A blit fill into it, then a copy out of it."""
    s = Scenario("resources_readonly_buffer", doc=resources_readonly_buffer.__doc__)
    s.bring_up()
    t = RTask(s)
    a_va = t.area(PAGE, contents=PATTERN)
    a = t.add(abi.T_BUFFER, buffer_rec(PAGE, a_va // PAGE, flags=2))
    p_va = t.area(PAGE, contents=PATTERN)
    p = t.add(abi.T_BUFFER, buffer_rec(PAGE, p_va // PAGE, flags=3))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_buf_buf(a, b, 0, 0, 0x100), b_copy_buf_buf(p, b, 0, 0x100, 0x100),
           resources=[(a, 0), (p, 0), (b, 0)], label="copy out of read-only buffers")
    t.dump(b_va, 0x200, "B")
    t.blit(b_fill_buf(a, 0, 0x10, 0x3C), resources=[(a, 0), (b, 0)], label="fill a read-only buffer")
    t.dump(a_va, 0x20, "A's pages")
    t.delete()
    s.events("end")
    return s


def creation_case(name, doc, records, setups=None, version=abi.MAX_VERSION, host=False, iosfc=False, probe_len=0x10,
                  tid=1, pages=64):
    """Creates each record as its own object by listing it in an exec beside
    a shared probe buffer, one exec per object, so each one's outcome is its
    own. records: (type, record bytes, entry length or None, faults); faults
    None lists the object in no exec (a target for other records), and a
    dict {"inproc": ..., "service": ...} says the outcome differs between
    the in-process setups (blocks, map) and the service (isolate): the case
    is then built twice, NAME_inproc and NAME_service. A creation that
    throws faults the exec before anything is committed, so the probe's
    slot for it stays zero. Returns a list of scenarios."""
    def outcome(faults, setup):
        if not isinstance(faults, dict):
            return faults
        if setup in faults:
            return faults[setup]
        return faults["service" if setup == "isolate" else "inproc"]

    all_setups = setups or ("blocks", "map", "isolate")
    sig = {st: tuple(outcome(r[3], st) for r in records) for st in all_setups}
    groups = []
    for st in all_setups:
        for g in groups:
            if sig[g[0]] == sig[st]:
                g.append(st)
                break
        else:
            groups.append([st])
    if len(groups) == 1:
        groups = [("", setups, all_setups[0])]
    else:
        names = {("blocks", "map"): "_inproc", ("isolate",): "_service"}
        groups = [(names.get(tuple(g), "_" + "_".join(g)), tuple(g), g[0]) for g in groups]
    out = []
    for suffix, sts, first in groups:
        kw = {"setups": sts} if sts else {}
        s = Scenario(name + suffix, doc=doc, host=host, iosfc=iosfc, **kw)
        s.bring_up(version=version)
        s.pg_version = version
        t = RTask(s, tid=tid, pages=pages)
        refs = []
        for typ, record, length, faults in records:
            faults = outcome(faults, first)
            if length is None:
                refs.append((t.add(typ, record), faults))
            else:
                va = t.ol._rec
                t.write(va, record)
                t.ol._rec = (va + max(len(record), 1) + 15) // 16 * 16
                refs.append((t.ol.add_raw(typ, length, va), faults))
        b, b_va = t.shared(PAGE)
        t.define()
        t.install()
        for i, (r, faults) in enumerate(refs):
            if faults is None:
                continue
            t.blit(b_fill_buf(b, i * probe_len, probe_len, (i + 1) & 0xFF), resources=[(r, 0), (b, 0)],
                   fault=faults, label="object %d" % r)
        t.dump(b_va, probe_len * len(refs), "probe buffer: which execs ran")
        t.delete()
        s.events("end")
        out.append(s)
    return out
