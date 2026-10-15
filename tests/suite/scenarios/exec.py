"""Execution framing: ExecIndirect2 (55), ExecIndirect3 (67) and
InfoIndirect (69), their resource lists and command-buffer ranges, and the
segments each range holds (docs/testing.md, sections 6.12, 6.13).

What the GPU did is read back as a blit fill into a shared buffer over
guest pages: a fill that is in guest memory ran, one that is not never
reached the GPU.
"""
import struct

from pgtest import abi
from pgtest.exec_objects import ObjTask, submit_last
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import GPUTask, rec, seg

PAGE = abi.PAGE
BLIT, COMPUTE, RENDER, INFO, PROT = abi.ENC_BLIT, abi.ENC_COMPUTE, abi.ENC_RENDER, abi.ENC_INFO, abi.ENC_PROTECTION


def fill(ref, off, n, byte):
    """Blit 306 FillBuffer {u32 ref, u64 offset, u64 length, u8 value}."""
    return rec(306, struct.pack("<IQQB3x", ref, off, n, byte))


class Target:
    """A task with one shared buffer (the fill target) and an object list."""

    def __init__(self, s, tid=1, pages=64, **kw):
        t = self.t = GPUTask(s, tid=tid, pages=pages, **kw)
        self.buf, self.va = t.buffer(PAGE)
        self.res = [(self.buf, 0)]
        t.define()
        t.install()

    def run(self, segments, resources=None, opcode=abi.OP_EXEC_INDIRECT3, fault=False, label=None, ranges=None):
        """Places each segment as its own range (or the given ranges) and
        executes them; then the fault state and the buffer's first bytes."""
        t = self.t
        t.reset_cmdbufs()
        cbs = ranges if ranges is not None else [t.cmdbuf(sg) for sg in segments]
        res = self.res if resources is None else resources
        if opcode == abi.OP_EXEC_INDIRECT2:
            t.exec2(cbs, resources=[(r, (f & 1), (f >> 1) & 1, (f >> 2) & 1) for r, f in res], fault=fault, label=label)
        else:
            t.exec3(cbs, resources=res, opcode=opcode, fault=fault, label=label)
        t.s.fault_state(label)
        t.dump(self.va, 0x40, label)
        if fault:
            t.s.resume_root()

    def clear(self):
        """Zeroes the buffer's first bytes from the guest side."""
        self.t.write(self.va, bytes(0x40))


@scenario
def exec_opcodes():
    """The three opcodes run the same segments: a blit fill through 55 and
    67 reaches the GPU; through 69 (InfoIndirect) the command buffer is
    decoded and never committed, so the fill never runs."""
    s = Scenario("exec_opcodes", doc=exec_opcodes.__doc__)
    s.bring_up()
    x = Target(s)
    x.run([seg(BLIT, fill(x.buf, 0, 0x10, 0x55))], opcode=abi.OP_EXEC_INDIRECT2, label="55")
    x.run([seg(BLIT, fill(x.buf, 0x10, 0x10, 0x67))], label="67")
    x.run([seg(BLIT, fill(x.buf, 0x20, 0x10, 0x69))], opcode=abi.OP_INFO_INDIRECT, label="69: never committed")
    x.run([seg(BLIT, fill(x.buf, 0x30, 0x10, 0x67))], label="67 after 69")
    s.events("end")
    return s


@scenario
def exec_payload_errors():
    """Every refusal of the payload, for each opcode: taskID 0 ("This command
    may not reference a kernel task"), a payload under 12 bytes, the size
    arithmetic overflowing 32 bits in each of its three steps ("Overflow in
    CmdExecIndirect2/3"; 69 says 3), a payload shorter than its counts say,
    and a task that does not exist ("Invalid task"). Each faults at the FIFO,
    before the task sees it."""
    s = Scenario("exec_payload_errors", doc=exec_payload_errors.__doc__)
    s.bring_up()
    x = Target(s)
    cb = struct.pack("<QQ", 4 * PAGE, 8)
    for op, rsize, name in ((abi.OP_EXEC_INDIRECT2, 24, "55"), (abi.OP_EXEC_INDIRECT3, 8, "67"),
                            (abi.OP_INFO_INDIRECT, 8, "69")):
        cases = [
            ("task 0", struct.pack("<III", 0, 0, 1) + cb),
            ("11 bytes", struct.pack("<III", 1, 0, 0)[:11]),
            ("resourceCount overflows", struct.pack("<III", 1, (1 << 32) // rsize + 1, 0)),
            ("cmdBufferCount overflows", struct.pack("<III", 1, 0, 0x10000000)),
            ("the sum overflows", struct.pack("<III", 1, 0, 0x0FFFFFFF) + bytes(4)),
            ("one resource missing", struct.pack("<III", 1, 1, 0)),
            ("one range short", struct.pack("<III", 1, 0, 1) + cb[:15]),
            ("task 2 does not exist", struct.pack("<III", 2, 0, 1) + cb),
        ]
        for label, payload in cases:
            label = "%s %s" % (name, label)
            s.submit(op, payload, fault=True, label=label)
            s.fault_state(label)
            s.resume_root()
    st = s.submit(abi.OP_NOP)
    s.events("end")
    return s


@scenario
def exec_version_gates():
    """ExecIndirect3 needs binary version 109 or later and InfoIndirect 235
    or later ("This command is unsupported in this binary version");
    ExecIndirect2 has no gate. At 108 and 233 each refused opcode faults;
    at 109 and 235 they run."""
    out = []
    for version, ok3, ok69 in ((108, False, False), (109, True, False), (233, True, False), (235, True, True)):
        name = "exec_version_gates_%d" % version
        s = Scenario(name, doc=exec_version_gates.__doc__)
        s.bring_up(version=version)
        x = Target(s)
        x.run([seg(BLIT, fill(x.buf, 0, 0x10, 0x55))], opcode=abi.OP_EXEC_INDIRECT2, label="55")
        x.run([seg(BLIT, fill(x.buf, 0x10, 0x10, 0x67))], fault=not ok3, label="67")
        x.run([seg(BLIT, fill(x.buf, 0x20, 0x10, 0x69))], opcode=abi.OP_INFO_INDIRECT, fault=not ok69, label="69")
        s.events("end")
        out.append(s)
    return out


@scenario
def exec_no_object_list():
    """A task whose object list was never set cannot map it: each opcode
    faults (status 2) without running anything, in-process and in the
    service alike. A range under 8 bytes, which would be skipped, does not
    help."""
    s = Scenario("exec_no_object_list", doc=exec_no_object_list.__doc__)
    s.bring_up()
    t = GPUTask(s)
    t.define()
    for op, name in ((abi.OP_EXEC_INDIRECT2, "55"), (abi.OP_EXEC_INDIRECT3, "67"), (abi.OP_INFO_INDIRECT, "69")):
        if op == abi.OP_EXEC_INDIRECT2:
            t.exec2([(4 * PAGE, 4)], fault=True, label=name)
        else:
            t.exec3([(4 * PAGE, 4)], opcode=op, fault=True, label=name)
        s.fault_state(name)
        s.resume_root()
    s.events("end")
    return s


@scenario
def exec_ranges():
    """Command-buffer ranges: each holds exactly one segment. One 8-byte
    header is read, the cursor truncated to its length, and bytes after it
    are ignored; a range under 8 bytes is skipped; zero ranges run nothing
    and complete. Several ranges run in order: the later fill wins where
    they overlap."""
    s = Scenario("exec_ranges", doc=exec_ranges.__doc__)
    s.bring_up()
    x = Target(s)
    t = x.t
    x.run([], ranges=[], label="no ranges")
    # A second segment after the first in the same range is ignored.
    t.reset_cmdbufs()
    one = seg(BLIT, fill(x.buf, 0, 0x10, 0x11)) + seg(BLIT, fill(x.buf, 0x10, 0x10, 0x22))
    va, n = t.cmdbuf(one)
    x.run(None, ranges=[(va, n)], label="two segments in one range: the second ignored")
    for k in range(1, 8):
        x.run(None, ranges=[(va, k)], label="a range of %d bytes: skipped" % k)
    # Ranges under 8 bytes between real ones.
    t.reset_cmdbufs()
    a = t.cmdbuf(seg(BLIT, fill(x.buf, 0x20, 0x10, 0x33)))
    b = t.cmdbuf(seg(BLIT, fill(x.buf, 0x28, 0x10, 0x44)))
    x.run(None, ranges=[(a[0], 3), a, (b[0], 7), b], label="short ranges between two fills")
    # A range longer than its segment: the rest is ignored.
    x.run(None, ranges=[(a[0], a[1] + 0x80)], label="a range longer than its segment")
    s.events("end")
    return s


@scenario
def exec_segment_lengths():
    """The segment header's length: exactly 8 is an empty segment; under 8
    ("already past current offset") or past the range ("Cannot truncate")
    throws "Failed to truncate", and the exec faults with nothing
    committed."""
    s = Scenario("exec_segment_lengths", doc=exec_segment_lengths.__doc__)
    s.bring_up()
    x = Target(s)
    good = seg(BLIT, fill(x.buf, 0, 0x10, 0x11))
    x.run([seg(BLIT, b"", length=8), good], label="an 8-byte segment, then a fill")
    x.clear()
    for n in (0, 7):
        x.run([good, seg(BLIT, b"", length=n)], fault=True, label="length %d" % n)
    x.run([good, seg(BLIT, fill(x.buf, 0, 0x10, 0x22), length=0x100)], fault=True, label="length past the range")
    x.run([good, seg(BLIT, fill(x.buf, 0, 0x10, 0x22), length=0xFFFFFFFF)], fault=True, label="length 2^32-1")
    s.events("end")
    return s


@scenario
def exec_encoder_types():
    """Encoder types: 0 render, 1 compute, 2 blit, 4 info create a decoder;
    any other throws "Invalid encoder type" and faults. An empty blit, info
    or compute segment completes; the fill after it runs."""
    s = Scenario("exec_encoder_types", doc=exec_encoder_types.__doc__)
    s.bring_up()
    x = Target(s)
    for enc in (BLIT, INFO, COMPUTE):
        x.clear()
        x.run([seg(enc), seg(BLIT, fill(x.buf, 0, 0x10, 0x10 + enc))], label="empty type %d, then a fill" % enc)
    for enc in (3, 6, 7, 0xFF):
        x.clear()
        x.run([seg(BLIT, fill(x.buf, 0, 0x10, 0x77)), seg(enc)], fault=True, label="type %d" % enc)
    s.events("end")
    return s


@scenario
def exec_continuation():
    """Segment continuation: continuedByNext keeps the decoder for the next
    segment, which continues it with continuesPrevious; a type that differs
    from the carried decoder's throws "Encoder type doesn't match for
    continuation"; with no decoder carried the type compared is 0, so every
    type but render (type 0) throws that mismatch, and render throws "Previous
    stage didn't specify continuation". A carried decoder is reused whatever
    the next segment's type when it does not say continuesPrevious: a blit
    decoder decodes a segment marked info or compute. Any error: nothing of the
    exec is committed."""
    s = Scenario("exec_continuation", doc=exec_continuation.__doc__)
    s.bring_up()
    x = Target(s)
    b = x.buf
    x.run([seg(BLIT, fill(b, 0, 8, 0x11), cont_next=1), seg(BLIT, fill(b, 8, 8, 0x12), cont_prev=1)],
          label="blit continued by blit")
    x.clear()
    x.run([seg(BLIT, fill(b, 0, 8, 0x21), cont_next=1), seg(BLIT, fill(b, 8, 8, 0x22), cont_prev=1, cont_next=1),
           seg(BLIT, fill(b, 16, 8, 0x23), cont_prev=1)], label="three blit segments continued")
    x.clear()
    for enc in (INFO, COMPUTE, RENDER, 3):
        x.run([seg(BLIT, fill(b, 0, 8, 0x31), cont_next=1), seg(enc, fill(b, 8, 8, 0x32))],
              label="blit carried into a type %d segment" % enc)
        x.clear()
    for enc in (INFO, COMPUTE, RENDER):
        x.run([seg(BLIT, fill(b, 0, 8, 0x41), cont_next=1), seg(enc, fill(b, 8, 8, 0x42), cont_prev=1)],
              fault=True, label="blit continued as type %d: mismatch" % enc)
    x.run([seg(INFO, cont_next=1), seg(BLIT, fill(b, 8, 8, 0x43), cont_prev=1)], fault=True,
          label="info continued as blit: mismatch")
    x.run([seg(INFO, cont_next=1), seg(INFO, cont_prev=1), seg(BLIT, fill(b, 0, 8, 0x44))],
          label="info continued by info")
    x.clear()
    for enc in (BLIT, COMPUTE, INFO, 3, RENDER):
        x.run([seg(BLIT, fill(b, 0, 8, 0x51)), seg(enc, b"", cont_prev=1)], fault=True,
              label="type %d continues nothing" % enc)
    x.run([seg(BLIT, fill(b, 0, 8, 0x61)), seg(INFO, cont_next=1)], label="info continued by nothing")
    s.events("end")
    return s


@scenario
def exec_protection_envelope():
    """Encoder type 5, the protection envelope: a u64 of protection options
    goes to the command buffer, then the next 8 bytes are the real segment
    header, used without truncating to its length (the outer header's length
    bounds it). If either read fails the segment returns nil and is skipped,
    with no error. An inner type 5 is an invalid encoder type. The inner
    header's continuation flags apply."""
    s = Scenario("exec_protection_envelope", doc=exec_protection_envelope.__doc__)
    s.bring_up()
    x = Target(s)
    b = x.buf

    def env(options, inner, length=None, inner_len=None):
        hdr = seg(inner[0], b"", inner[2], inner[3], length=inner_len if inner_len is not None else 8 + len(inner[1]))
        return seg(PROT, struct.pack("<Q", options) + hdr + inner[1], length=length)

    x.run([env(0, (BLIT, fill(b, 0, 8, 0x11), 0, 0))], label="options 0, a blit fill")
    x.clear()
    x.run([env(0, (BLIT, fill(b, 0, 8, 0x21), 0, 0), inner_len=0)], label="inner length 0: not applied")
    x.clear()
    x.run([env(0, (BLIT, fill(b, 0, 8, 0x31), 0, 0), inner_len=0xFFFFFFFF)], label="inner length 2^32-1: not applied")
    x.clear()
    x.run([env(0, (BLIT, fill(b, 0, 8, 0x41), 0, 1)), seg(BLIT, fill(b, 8, 8, 0x42), cont_prev=1)],
          label="the inner header continued by the next segment")
    x.clear()
    x.run([seg(BLIT, fill(b, 0, 8, 0x51)), env(0, (BLIT, fill(b, 8, 8, 0x52), 0, 0))],
          label="an envelope after an ended blit encoder")
    x.clear()
    for n in (8, 12, 15):
        x.run([seg(PROT, bytes(n - 8), length=n), seg(BLIT, fill(b, 0, 8, 0x60 + n))],
              label="outer length %d: options unread, skipped" % n)
        x.clear()
    for n in (16, 20, 23):
        x.run([seg(PROT, bytes(n - 8), length=n), seg(BLIT, fill(b, 0, 8, 0x60 + n))],
              label="outer length %d: inner header unread, skipped" % n)
        x.clear()
    x.run([env(0, (PROT, b"", 0, 0))], fault=True, label="an envelope in an envelope")
    x.run([env(0, (BLIT, fill(b, 0, 8, 0x71), 1, 0))], fault=True, label="inner continuesPrevious with nothing carried")
    s.events("end")
    return s


def envelope(options, enc, body, cont_prev=0, cont_next=0):
    return seg(PROT, struct.pack("<Q", options) + seg(enc, body, cont_prev, cont_next))


@scenario
def exec_protection_envelope_open_encoder():
    """A protection envelope while a carried blit encoder is still open:
    setProtectionOptions: asserts in Metal ("with uncommitted encoder").
    The assert is inside Metal, so the host dies in every setup, an
    isolated task's included (docs/testing.md, section 4.8), and the next
    exec is never reached."""
    s = Scenario("exec_protection_envelope_open_encoder", doc=exec_protection_envelope_open_encoder.__doc__)
    s.bring_up()
    x = Target(s)
    b = x.buf
    x.run([seg(BLIT, fill(b, 0, 8, 0x51), cont_next=1), envelope(0, BLIT, fill(b, 8, 8, 0x52), cont_prev=1)],
          fault=True, label="an envelope continuing a blit")
    x.run([seg(BLIT, fill(b, 0, 8, 0x53))], fault=True, label="the task's service is gone")
    s.events("end")
    return s


@scenario
def exec_trailing_bytes():
    """A decoder loops while at least 8 bytes remain: 1-7 trailing bytes
    after the last record are ignored; a record length under 8 throws
    "Underflow"; one past the segment "Failed to read" (the blit decoder's
    loop, as the info decoder's)."""
    s = Scenario("exec_trailing_bytes", doc=exec_trailing_bytes.__doc__)
    s.bring_up()
    x = Target(s)
    b = x.buf
    for k in range(1, 8):
        x.run([seg(BLIT, fill(b, 0, 8, 0x10 + k) + bytes(range(0xF0, 0xF0 + k)))], label="%d trailing bytes" % k)
    x.clear()
    for n in (0, 7):
        x.run([seg(BLIT, fill(b, 0, 8, 0x21) + rec(306, b"", length=n))], fault=True, label="record length %d" % n)
    x.run([seg(BLIT, fill(b, 0, 8, 0x22) + rec(306, bytes(24), length=0x40))], fault=True,
          label="record past the segment")
    s.events("end")
    return s


def copy(src, dst, soff, doff, n):
    """Blit 301 CopyFromBuffer {u32 src, u32 dst, u64 srcOffset, u64
    dstOffset, u64 size} (tests/suite/scenarios/smoke.py smoke_blit)."""
    return rec(301, struct.pack("<IIQQQ", src, dst, soff, doff, n))


class Paged:
    """A private buffer P (paged by copies) and a shared buffer S."""

    def __init__(self, s):
        t = self.t = ObjTask(s)
        self.p, self.p_va = t.buffer(PAGE, flags=1, contents=b"\xA1" * 0x40)
        self.sh, self.s_va = t.buffer(PAGE)
        self.cpso = t.compute_pipeline()
        t.define()
        t.install()

    def exec(self, segments, pflags, label, opcode=abi.OP_EXEC_INDIRECT3, fault=False):
        t = self.t
        t.reset_cmdbufs()
        cbs = [t.cmdbuf(sg) for sg in segments]
        if opcode == abi.OP_EXEC_INDIRECT2:
            # pflags is the three flag bytes as given.
            t.exec2(cbs, resources=[(self.p,) + tuple(pflags), (self.sh, 0, 0, 0)], fault=fault, label=label)
        else:
            t.exec3(cbs, resources=[(self.p, pflags), (self.sh, 0)], opcode=opcode, fault=fault, label=label)
        t.s.fault_state(label)
        t.dump(self.s_va, 0x40, label + ": shared")
        t.dump(self.p_va, 0x40, label + ": private's guest pages")


@scenario
def exec_resource_flags():
    """The resource flags of ExecIndirect3 (bit 0 invalidateHost, bit 1
    invalidateGuest, bit 2 synchronize), seen through a private buffer,
    which is paged by copies: the first use prepares it from guest memory;
    later guest writes reach it only with invalidateHost; GPU writes reach
    guest memory only when invalidateGuest marked the guest copy stale and
    synchronize writes it back at completion."""
    s = Scenario("exec_resource_flags", doc=exec_resource_flags.__doc__)
    s.bring_up()
    x = Paged(s)
    t, P, S = x.t, x.p, x.sh
    x.exec([seg(BLIT, copy(P, S, 0, 0, 0x10))], 0, "first use: prepared")
    t.write(x.p_va, b"\xB2" * 0x40)
    x.exec([seg(BLIT, copy(P, S, 0, 0x10, 0x10))], 0, "flags 0: stale host copy")
    x.exec([seg(BLIT, copy(P, S, 0, 0x20, 0x10))], 1, "invalidateHost: fresh")
    x.exec([seg(BLIT, fill(P, 0, 0x10, 0xC3))], 0, "a fill, flags 0: guest untouched")
    x.exec([seg(BLIT, fill(P, 0, 0x10, 0xC4))], 4, "a fill, synchronize alone: guest still valid, untouched")
    x.exec([seg(BLIT, fill(P, 0x10, 0x10, 0xC5))], 2, "a fill, invalidateGuest: not written back yet")
    x.exec([], 4, "synchronize: written back")
    x.exec([seg(BLIT, fill(P, 0x20, 0x10, 0xC6))], 6, "invalidateGuest and synchronize: written back at completion")
    x.exec([seg(BLIT, fill(P, 0x30, 0x10, 0xC7))], 7, "all three")
    s.events("end")
    return s


@scenario
def exec2_resource_flags():
    """ExecIndirect2's 24-byte resource records carry each flag as a byte,
    and each is cut to its bit, in-process and in the service alike: an
    invalidateHost byte of 2 is 0, of 3 is 1."""
    s = Scenario("exec2_resource_flags", doc=exec2_resource_flags.__doc__)
    s.bring_up()
    x = Paged(s)
    t, P, S = x.t, x.p, x.sh
    E2 = abi.OP_EXEC_INDIRECT2
    x.exec([seg(BLIT, copy(P, S, 0, 0, 0x10))], (0, 0, 0), "first use", opcode=E2)
    t.write(x.p_va, b"\xB2" * 0x40)
    x.exec([seg(BLIT, copy(P, S, 0, 0x10, 0x10))], (2, 0, 0), "invalidateHost byte 2: stale", opcode=E2)
    x.exec([seg(BLIT, copy(P, S, 0, 0x20, 0x10))], (3, 0, 0), "invalidateHost byte 3: fresh", opcode=E2)
    x.exec([seg(BLIT, fill(P, 0, 0x10, 0xC3))], (0, 0xFE, 0xFF), "invalidateGuest byte 0xfe, synchronize 0xff",
           opcode=E2)
    x.exec([seg(BLIT, fill(P, 0x10, 0x10, 0xC4))], (0, 1, 1), "invalidateGuest 1, synchronize 1", opcode=E2)
    s.events("end")
    return s


@scenario
def exec_resource_list_errors():
    """What a resource list may name, in-process and in the service alike: a
    pipeline is "Unexpected object type (7) for resource", a reference past
    the list or of type 10 fails to be created; each faults with nothing
    run. Reference 0 is nil and passes."""
    s = Scenario("exec_resource_list_errors", doc=exec_resource_list_errors.__doc__)
    s.bring_up()
    t = ObjTask(s)
    sh, s_va = t.buffer(PAGE)
    cpso = t.compute_pipeline()
    bad = t.ol.add_raw(10, 16, PAGE)
    t.define()
    t.install()
    past = len(t.ol.entries) + 3
    for label, res, fault in (("a pipeline", [(sh, 0), (cpso, 0)], True),
                              ("past the list", [(sh, 0), (past, 1)], True),
                              ("type 10", [(sh, 0), (bad, 4)], True),
                              ("reference 0", [(0, 7), (sh, 0)], False),
                              ("the same buffer twice", [(sh, 1), (sh, 6)], False)):
        t.reset_cmdbufs()
        t.exec3([t.cmdbuf(seg(BLIT, fill(sh, 0, 0x10, len(label))))], resources=res, fault=fault, label=label)
        s.fault_state(label)
        t.dump(s_va, 0x10, label)
        if fault:
            s.resume_root()
        t.write(s_va, bytes(0x10))
    s.events("end")
    return s


@scenario
def exec_empty_resource_list():
    """A fill of a shared buffer the resource list does not name: in-process
    it runs. In the service, on a GPU with detach backing, the buffer is not
    made resident and the command buffer fails ("Insufficient Memory"):
    the exec faults (status 3) after decoding succeeded
    (docs/testing.md, section 6.12)."""
    out = []
    for name, setups, fault in (("exec_empty_resource_list", ("blocks", "map"), False),
                                ("exec_empty_resource_list_isolate", ("isolate",), True)):
        s = Scenario(name, setups=setups, doc=exec_empty_resource_list.__doc__)
        s.bring_up()
        x = Target(s)
        x.run([seg(BLIT, fill(x.buf, 0, 0x10, 0x5A))], resources=[], fault=fault, label="an empty list")
        x.run([seg(BLIT, fill(x.buf, 0x10, 0x10, 0x5B))], label="listed")
        s.events("end")
        out.append(s)
    return out


def _death(name, doc, build, setups=("blocks", "map", "isolate"), watchdog_s=30):
    s = Scenario(name, setups=setups, doc=doc, watchdog_s=watchdog_s)
    s.bring_up()
    x = Target(s, holes=(40,))
    build(s, x)
    s.events("end")
    return s


@scenario
def exec_null_cursor():
    """Nothing checks for a null command-buffer cursor (docs/testing.md,
    sections 6.12, 7.4): a range whose cursor cannot be made is
    dereferenced. Each case is its own scenario, a death test: in-process
    the host dies; in the isolate setup the null cursor is a death in the
    task's own code, which ends only that task: the exec faults, and the
    next command on the task faults too. A range past the
    task's end, one over a hole in the page table, and a zero-length range
    at a page boundary (in-process LocalTask::cursor makes an empty cursor
    there, which also has no header to read: it completes; the service
    refuses any zero length)."""
    doc = exec_null_cursor.__doc__
    out = []

    def case(ranges, label):
        def build(s, x):
            x.run([seg(BLIT, fill(x.buf, 0, 8, 0x11))], label="before")
            x.t.exec3(ranges, resources=x.res, fault=True, label=label)
            s.fault_state(label)
            s.resume_root()
            x.run([seg(BLIT, fill(x.buf, 8, 8, 0x22))], fault=True, label="after")
        return build

    out.append(_death("exec_null_cursor_past_end", doc, case([(64 * PAGE, 0x40)], "a range past the task")))
    out.append(_death("exec_null_cursor_hole", doc, case([(40 * PAGE, 0x40)], "a range over a hole")))
    out.append(_death("exec_null_cursor_overflow", doc, case([(0xFFFFFFFFFFFFFFF0, 0x40)], "offset + length overflows")))
    return out


def _split(name, doc, build, fault_in_process, watchdog_s=20):
    """Two scenarios where the setups' waits differ: name (blocks, map)
    and name_isolate. build(s, x, fault) takes whether the exec faults."""
    return [_death(name, doc, lambda s, x: build(s, x, fault_in_process), setups=("blocks", "map"),
                   watchdog_s=watchdog_s),
            _death(name + "_isolate", doc, lambda s, x: build(s, x, True), setups=("isolate",),
                   watchdog_s=watchdog_s)]


@scenario
def exec_zero_length_range():
    """A zero-length range. In-process a range that is not page-aligned makes
    an empty cursor (LocalTask::cursor walks one page), and an empty
    cursor has no header: skipped, and the exec completes. A page-aligned
    one dies in the host. The service refuses every zero length ("Zero
    length cursor request") and dereferences the null cursor, a death in
    the task's own code that ends only the task: the exec faults, and so
    does the next (docs/testing.md, section 6.12)."""
    out = []
    for name, off, fault in (("exec_zero_length_range_unaligned", 4 * PAGE + 0x10, False),
                             ("exec_zero_length_range_aligned", 4 * PAGE, True)):
        def build(s, x, fault, off=off):
            x.run([seg(BLIT, fill(x.buf, 0, 8, 0x11))], label="before")
            x.t.exec3([(off, 0)], resources=x.res, fault=fault, label="zero length at %#x" % off)
            s.fault_state("zero length")
            if fault:
                s.resume_root()
            x.run([seg(BLIT, fill(x.buf, 8, 8, 0x22))], fault=fault, label="after")
        out += _split(name, exec_zero_length_range.__doc__, build, fault)
    return out


@scenario
def exec_range_at_task_end():
    """A command buffer ending exactly at the task's end: in-process
    LocalTask::cursor's range check refuses end >= the task's length, the
    cursor is null, and Deserializer::decode_segments dies on it: the host
    dies. The service
    accepts end == length and runs it (docs/testing.md, sections 6.10,
    7.1)."""
    def build(s, x):
        t = x.t
        data = seg(BLIT, fill(x.buf, 0, 8, 0x5E))
        va = 64 * PAGE - len(data)
        t.write(va, data)
        x.run(None, ranges=[(va, len(data))], label="a range ending at the task's end")
    return _death("exec_range_at_task_end", exec_range_at_task_end.__doc__, build)


@scenario
def exec_open_encoder_at_commit():
    """The last segment of an exec says continuedByNext: its blit encoder is
    never ended, and encodeSignalEvent:value: asserts in Metal ("with
    uncommitted encoder") before the commit (docs/testing.md, section 6.13,
    effect unknown until now). The assert is inside Metal, so the host dies
    in every setup, an isolated task's included (docs/testing.md, section
    4.8), and the next exec is never reached."""
    def build(s, x, fault):
        x.run([seg(BLIT, fill(x.buf, 0, 8, 0x11))], label="before")
        x.run([seg(BLIT, fill(x.buf, 8, 8, 0x22), cont_next=1)], fault=fault,
              label="continuedByNext on the last segment")
        x.run([seg(BLIT, fill(x.buf, 16, 8, 0x33))], fault=fault, label="after")
    return _split("exec_open_encoder_at_commit", exec_open_encoder_at_commit.__doc__, build, True)


@scenario
def exec_carried_compute_reopened():
    """A carried compute decoder meets a segment that does not continue it:
    the decoder is reused and opens a second compute encoder over its open
    one, and Metal asserts ("A command encoder is already encoding to this
    command buffer"; docs/testing.md, section 6.13, unknown until now). The
    assert is inside Metal, so the host dies in every setup, an isolated
    task's included (section 4.8), and the next exec is never reached."""
    def build(s, x, fault):
        x.run([seg(BLIT, fill(x.buf, 0, 8, 0x11))], label="before")
        x.run([seg(COMPUTE, b"", cont_next=1), seg(COMPUTE, b"")], fault=fault,
              label="compute carried into a new compute segment")
        x.run([seg(BLIT, fill(x.buf, 16, 8, 0x33))], fault=fault, label="after")
    return _split("exec_carried_compute_reopened", exec_carried_compute_reopened.__doc__, build, True)


@scenario
def exec_decoder_getters():
    """The deserializer's object getters the decoders call, each reached
    once: compute textures, samplers, heaps, resources and a fence around
    a dispatch whose output is read back; a render pass with a rate map
    reference 0, a render pipeline and a depth-stencil state around a draw
    into a texture over a buffer. A fence reference 0 is the NonNull
    getter's "Got invalid Fence"."""
    from pgtest.exec_objects import (buffer_texture_record, c_buffers, c_dispatch_threads, c_pipeline,
                                     depth_stencil_record, heap_record, r_draw, r_pipeline, render_pass,
                                     sampler_record)
    s = Scenario("exec_decoder_getters", doc=exec_decoder_getters.__doc__)
    s.bring_up()
    t = ObjTask(s)
    out, out_va = t.buffer(PAGE)
    tb, tb_va = t.buffer(PAGE)
    tex = t.ol.add(abi.T_SERIALIZER_TEXTURE, buffer_texture_record(tb))
    sampler = t.ol.add(abi.T_SERIALIZED, sampler_record())
    hva = t.reserve(4 * PAGE)
    heap = t.ol.add(abi.T_HEAP, heap_record(hva // PAGE, 4 * PAGE, 1))
    fence = t.ol.add(abi.T_SERIALIZED, struct.pack("<II", 13, 0))
    dss = t.ol.add(abi.T_SERIALIZED, depth_stencil_record())
    cpso = t.compute_pipeline()
    rpso = t.render_pipeline()
    t.define()
    t.install()
    res = [(out, 0), (tb, 0), (heap, 0)]
    comp = (c_pipeline(cpso) + c_buffers(0, [(out, 0)])
            + rec(0xCE, struct.pack("<IIII", 0, 2, 0, tex))
            + rec(0xCC, struct.pack("<IIII", 0, 2, 0, sampler))
            + rec(0x86, struct.pack("<II", 1, heap))
            + rec(0x87, struct.pack("<IIII", 2, 1, out, tex))
            + rec(0xD4, struct.pack("<I", fence))
            + c_dispatch_threads((16, 1, 1), (16, 1, 1)))
    t.exec_segments([seg(COMPUTE, comp)], resources=res, label="compute")
    s.fault_state("compute")
    t.dump(out_va, 0x40, "compute output")
    rend = (render_pass(tex, prefix=rec(0x21, struct.pack("<I", 0))) + r_pipeline(rpso)
            + rec(0x68, struct.pack("<I", dss)) + r_draw())
    t.exec_segments([seg(RENDER, rend)], resources=res, label="render")
    s.fault_state("render")
    t.dump(tb_va, 0x40, "render target")
    t.exec_segments([seg(COMPUTE, c_pipeline(cpso) + rec(0xD4, struct.pack("<I", 0)))], resources=res,
                    fault=True, label="fence ref 0")
    s.fault_state("fence ref 0")
    s.resume_root()
    s.events("end")
    return s


class RateMaps:
    """A task with two rate maps (mutable and immutable), a shared reply
    buffer and a CPU-free way to create a map in-process: info 452 names
    the map (created on first use) and a reply buffer reference 0, which
    fails after the map exists and before any reply is written."""

    def __init__(self, s):
        from pgtest.exec_objects import rate_map_record
        t = self.t = ObjTask(s)
        self.reply, self.reply_va = t.buffer(PAGE)
        self.rec_m = rate_map_record(w=256, h=256, mutable=1, min_factor=0.25, samples=[([0.25, 1.0], [0.25, 1.0])])
        self.mut = t.ol.add(abi.T_SERIALIZED, self.rec_m)
        self.imm = t.ol.add(abi.T_SERIALIZED, rate_map_record(w=256, h=256, samples=[([0.25, 1.0], [0.25, 1.0])]))
        self.unused = t.ol.add(abi.T_SERIALIZED, rate_map_record())
        self.mut_va = t.ol.entries[self.mut][2]
        t.define()
        t.install()

    def create(self, ref, label):
        self.t.exec_segments([seg(INFO, rec(452, struct.pack("<IIQQ", ref, 0, 0, 0)))], fault=True, label=label)
        self.t.s.resume_root()

    def query(self, ref, off, label):
        """454 map screen to physical of (200, 200), isolate only."""
        self.t.exec_segments([seg(INFO, rec(454, struct.pack("<IIQIff", ref, self.reply, off, 0, 200.0, 200.0)))],
                             label=label)
        self.t.s.fault_state(label)
        self.t.dump(self.reply_va + off, 8, label)

    def reset(self, ref, fault=False, label=None):
        self.t.s.submit(abi.OP_RESET_RASTERIZATION_RATE_MAP, struct.pack("<II", self.t.tid, ref), fault=fault,
                        label=label)
        self.t.s.fault_state(label)
        if fault:
            self.t.s.resume_root()


@scenario(setups=("isolate",))
def exec_reset_rate_map():
    """ResetRasterizationRateMap (66) in the service: the map named must
    exist; its object-list record is read again and handed to
    resetRasterizationRateMapWithSerializedData, which rebuilds the
    descriptor from the record's samples when its commandType is 0x32 and
    does nothing otherwise. A map never created replies success; reference
    0 faults. 454 shows the mapping before and after."""
    s = Scenario("exec_reset_rate_map", setups=("isolate",), host=True, doc=exec_reset_rate_map.__doc__)
    s.bring_up()
    m = RateMaps(s)
    t = m.t
    m.query(m.mut, 0, "454 before")
    # New samples in the same record: the second zone at quarter rate.
    body = bytearray(m.rec_m)
    struct.pack_into("<4f", body, 0x1c + 4, 1.0, 0.25, 1.0, 0.25)
    t.write(m.mut_va, bytes(body))
    m.query(m.mut, 8, "454 after the record changed, before 66")
    m.reset(m.mut, label="66 on the mutable map")
    m.query(m.mut, 0x10, "454 after 66")
    struct.pack_into("<I", body, 0, 51)
    struct.pack_into("<4f", body, 0x1c + 4, 0.25, 1.0, 0.25, 1.0)
    t.write(m.mut_va, bytes(body))
    m.reset(m.mut, label="66 with commandType 51: ignored")
    m.query(m.mut, 0x18, "454 unchanged")
    m.reset(m.unused, label="66 on a map never created: success")
    m.reset(0, fault=True, label="66 on reference 0")
    s.events("end")
    return s


@scenario(setups=("isolate",))
def exec_reset_rate_map_immutable():
    """66 on an immutable map: the reset raises (the map is immutable)
    inside the service's job, GpuTask::reset_rasterization_rate_map, where
    nothing catches it: the raise ends the task, 66 faults, and so does
    every later command on the task."""
    s = Scenario("exec_reset_rate_map_immutable", setups=("isolate",), doc=exec_reset_rate_map_immutable.__doc__)
    s.bring_up()
    m = RateMaps(s)
    m.create(m.imm, "create the immutable map")
    m.reset(m.imm, fault=True, label="66 on the immutable map")
    m.reset(m.mut, fault=True, label="66 after: the service is gone")
    s.events("end")
    return s


@scenario(setups=("isolate",))
def exec_reset_rate_map_short():
    """66 when the map's record no longer holds its samples: the reset reads
    through a cursor over the record from +0xc, and a sample read past it
    raises, uncaught in the service's job: the task ends. The record
    shortened by 8 bytes in the object list entry."""
    s = Scenario("exec_reset_rate_map_short", setups=("isolate",), doc=exec_reset_rate_map_short.__doc__)
    s.bring_up()
    m = RateMaps(s)
    t = m.t
    m.create(m.mut, "create the mutable map")
    typ, n, va = t.ol.entries[m.mut]
    t.ol.entries[m.mut] = (typ, n - 12, va)
    t.ol.write()
    m.reset(m.mut, fault=True, label="66 with its samples cut short")
    s.events("end")
    return s


@scenario(setups=("isolate",))
def exec_reset_rate_map_tiny():
    """66 when the map's record is under 8 bytes: the reset's size check
    (Deserializer::reset_rasterization_rate_map_with_serialized_data)
    raises, uncaught in the service's job: the task ends."""
    s = Scenario("exec_reset_rate_map_tiny", setups=("isolate",), doc=exec_reset_rate_map_tiny.__doc__)
    s.bring_up()
    m = RateMaps(s)
    t = m.t
    m.create(m.mut, "create the mutable map")
    typ, n, va = t.ol.entries[m.mut]
    t.ol.entries[m.mut] = (typ, 4, va)
    t.ol.write()
    m.reset(m.mut, fault=True, label="66 with a 4-byte record")
    s.events("end")
    return s


@scenario(setups=("isolate",))
def exec_reset_rate_map_under_command():
    """66 when the map's record holds its header but not the 0x14 bytes of
    the command (here 0x10 bytes): nothing checks the command's length, so
    the reset goes on and reads the descriptor through an 8-byte cursor:
    its first 16-byte read raises, uncaught in the service's job, and the
    task ends (exec_reset_rate_map_short)."""
    s = Scenario("exec_reset_rate_map_under_command", setups=("isolate",),
                 doc=exec_reset_rate_map_under_command.__doc__)
    s.bring_up()
    m = RateMaps(s)
    t = m.t
    m.create(m.mut, "create the mutable map")
    typ, n, va = t.ol.entries[m.mut]
    t.ol.entries[m.mut] = (typ, 0x10, va)
    t.ol.write()
    m.reset(m.mut, fault=True, label="66 with a 16-byte record")
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def exec_reset_rate_map_in_process():
    """66 in-process: reference 0 and a map never created fault (status 2;
    the service replies success for the second, exec_reset_rate_map); an
    existing map reaches the in-process task's object lookup, which aborts:
    the host dies (docs/testing.md, section 7.4)."""
    s = Scenario("exec_reset_rate_map_in_process", setups=("blocks", "map"),
                 doc=exec_reset_rate_map_in_process.__doc__)
    s.bring_up()
    m = RateMaps(s)
    m.reset(0, fault=True, label="66 on reference 0")
    m.reset(m.mut, fault=True, label="66 on a map never created")
    m.create(m.mut, "create the map")
    submit_last(s, abi.OP_RESET_RASTERIZATION_RATE_MAP, struct.pack("<II", m.t.tid, m.mut),
                label="66 on an existing map: the host dies")
    return s

