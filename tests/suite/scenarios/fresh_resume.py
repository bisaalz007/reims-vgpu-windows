"""Suspend in one process, resume in another (docs/testing.md, section 6.24):
the migration a hypervisor makes with rvg_device_will_suspend,
rvg_device_finish_suspend, rvg_device_will_resume and rvg_device_did_resume,
each side a runner process of its own. The phases hand guest RAM and the blob
on (Scenario.phase, tests/suite/README.md "Phases").

What one process cannot show, and these lock: guest RAM at new host
addresses; process-wide state fresh on the target (display serial numbers,
the last IOSurface host device, host task numbers); a migration between
setups; one image resumed twice; and target RAM that differs. Then the blob
itself, mutated: every truncation at a field boundary and every u32 field set
to boundary values, against the blob's parse and rvg_device_will_resume's
checks.
"""
import copy
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.display_guest import DisplayGuest, display, p_glyph, p_present3, p_show, p_swap, pattern
from pgtest.efi_guest import EFI
from pgtest.fifo import ChildFIFO
from pgtest.iosfc_guest import CMD_MAP, CMD_TEST, IOSFC, IREGS, desc_v2
from pgtest.registry import scenario
from pgtest.scenario import RAM_BASE, RAM_SIZE, Scenario
from pgtest.suspend_guest import blob, state
from pgtest.task import GPUTask, rec, seg

SAME = ("blocks", "map", "isolate")
CROSS = tuple((a, b) for a in SAME for b in SAME if a != b)


def suspend(s, slot="s"):
    s.api("willSuspend")
    s.api("finishSuspend", slot=slot)


def resume(s, slot="s", label=None):
    kw = {"label": label} if label else {}
    s.api("willResume", slot=slot, **kw)
    s.api("didResume", **kw)


def rewind_point(*objs):
    """The builder state of objs (FIFO counters, a task's allocators), so
    that a later phase that loads an earlier image can write its commands
    where that image's guest would."""
    def plain(v):
        return isinstance(v, (int, float, str, bytes, type(None))) or (
            isinstance(v, (list, tuple, dict, set)) and all(plain(x) for x in (v.values() if isinstance(v, dict) else v)))
    keep = lambda o, k: not (isinstance(o, Scenario) and k != "_next_page")  # never the ops themselves
    return [(o, {k: copy.deepcopy(v) for k, v in vars(o).items() if plain(v) and keep(o, k)}) for o in objs]


def rewind(point):
    for o, d in point:
        vars(o).update(copy.deepcopy(d))


class Guest:
    """suspend_roundtrip's guest (scenarios/suspend.py): a root FIFO, child
    FIFO 1, task 1 with buffers A (a pattern), B, and P (private, written
    by a blit that rvg_device_finish_suspend writes back, W14), task 0 with a
    cursor glyph, and a connected display on port 0 whose cursor is shown."""

    def __init__(self, s):
        self.s = s
        g = self.g = DisplayGuest(s, pages=16)
        self.c = ChildFIFO(s, 1)
        self.c.setup_node()
        self.c.define()
        t = self.t = GPUTask(s, tid=1)
        self.a, self.a_va = t.buffer(contents=bytes(range(256)))
        self.b, self.b_va = t.buffer()
        self.p, self.p_va = t.buffer(flags=1)
        t.define()
        t.install()
        fillp = struct.pack("<IQQB3x", self.p, 0, 0x40, 0xEE)
        t.exec3([t.cmdbuf(seg(abi.ENC_BLIT, rec(306, fillp)))], resources=[(self.p, abi.RES_INVALIDATE_GUEST)])
        g.t0.write(8 * PAGE, struct.pack("<I", 0x80402010) * 4)
        g.connect(0, enabled=0xF)
        g.ack(0, 1)
        s.submit(abi.OP_DISPLAY_CURSOR_GLYPH, p_glyph(0, 8 * PAGE, 16, 8, 2, 2, 1, 1, 0))
        s.submit(abi.OP_DISPLAY_CURSOR_SHOW, p_show(0, 1))
        s.wait(cb="cursorShow")
        s.events("before")
        s.diff("before")

    def drive(self, label):
        """The workload after a resume: registers, a NOP on each FIFO, a
        blit in task 1, the private buffer from guest pages, the cursor."""
        s, t = self.s, self.t
        s.sweep(0x1000, 0x1040, label="%s: registers" % label)
        s.submit(abi.OP_NOP, label="%s: root NOP" % label)
        self.c.submit(abi.OP_NOP, label="%s: child NOP" % label)
        fill = struct.pack("<IQQB3x", self.b, 0, 0x40, 0xCD)
        copy_ = struct.pack("<IIQQQ", self.a, self.b, 0, 0x40, 0x40)
        t.exec3([t.cmdbuf(seg(abi.ENC_BLIT, rec(306, fill) + rec(301, copy_)))],
                resources=[(self.a, 5), (self.b, 5)])
        t.dump(self.b_va, 0x80, "%s: buffer B" % label)
        t.dump(self.p_va, 0x40, "%s: private buffer P" % label)
        s.api("cursorPosition", display=0, label=label)
        s.api("display_props", display=0, label=label)
        s.events(label)
        s.diff(label, gpa=s.pa(0), length=s._next_page * PAGE)


@scenario(setups=SAME + CROSS)
def fresh_resume_roundtrip():
    """suspend_roundtrip's guest suspended in one process and resumed in a
    new one, in every setup and every pair of different setups (a migration
    between hosts configured differently). The new process rebuilds the root
    FIFO at the blob's counts, child FIFO 1, tasks 0 and 1 and the display,
    replaying the cursor glyph and show, and raises one interrupt; P comes
    back from its guest pages, which rvg_device_finish_suspend wrote back. The
    resuming side's observations depend on its own setup only."""
    s = Scenario("fresh_resume_roundtrip", setups=SAME + CROSS, displays=[display(0)],
                 doc=fresh_resume_roundtrip.__doc__)
    g = Guest(s)
    suspend(s)
    s.diff("suspended")
    s.phase()
    resume(s)
    s.wait(cb="cursorShow", label="the restore shows the cursor")
    s.events("resumed")
    g.drive("resumed")
    return s


# ---- what a migration carries between setups ----

@scenario(setups=SAME + CROSS, host=True)
def fresh_resume_caps():
    """A blob with device info (the guest asked for key 45 first), resumed by a
    new process in every setup and every pair of setups: the capability check
    (device_info::check_capabilities, from
    DeviceCore::check_suspend_device_info) compares the source's host values
    with the target's, and what the setup changes in them decides whether a
    migration between differently configured hosts is refused
    (IncompatibleDevice, 4). Measured on an M3 Max: key 33's bits are 0xbdc in
    blocks, 0xbdf in map (large user and kernel tasks) and 0xbff in isolate
    (argument buffers too), and key 42 is 3 in isolate only; a target lacking a
    bit the source reported refuses (map to blocks, isolate to blocks, isolate
    to map), the other ways accept. A refused target is left
    LeavingSuspendEarly: only its registers are read. Host values, so keyed by
    host."""
    s = Scenario("fresh_resume_caps", setups=SAME + CROSS, host=True, doc=fresh_resume_caps.__doc__)
    s.bring_up()
    s.submit(abi.OP_GET_DEVICE_INFO, struct.pack("<III", 45, 64, s.pfn(s.alloc_page())))
    t = GPUTask(s, tid=1)
    b, b_va = t.buffer()
    t.define()
    t.install()
    s.events("before")
    suspend(s)
    s.phase()
    s.api("willResume", slot="s")
    s.sweep(0x1000, 0x1040, label="after willResume")
    s.events("end")
    return s


@scenario(setups=("blocks", "map", "isolate", ("blocks", "isolate", "map")))
def fresh_resume_twice():
    """One image (the blob and guest RAM) resumed by two processes in turn:
    phase 1 resumes it and drives the guest on; phase 2 loads the same
    image again, not phase 1's, and resumes it: its observations equal
    phase 1's (in the same setup), because nothing of the first resume
    reaches the image. Also as blocks, then isolate, then map."""
    s = Scenario("fresh_resume_twice", setups=("blocks", "map", "isolate", ("blocks", "isolate", "map")),
                 displays=[display(0)], doc=fresh_resume_twice.__doc__)
    g = Guest(s)
    suspend(s)
    point = rewind_point(s, s.fifo, g.c, g.t, g.t.ol, g.g.t0)
    s.phase()
    resume(s)
    s.wait(cb="cursorShow", label="the restore shows the cursor")
    s.events("resumed")
    g.drive("first resume")
    s.phase(save=False)
    rewind(point)
    resume(s)
    s.wait(cb="cursorShow", label="the restore shows the cursor")
    s.events("resumed")
    g.drive("second resume")
    return s


# ---- the target's guest RAM ----

def _task_far(s, first_page):
    """Task 1 over RAM pages from first_page on, with buffer A (a pattern)
    and B; returns (task, a, a_va, b, b_va)."""
    s._next_page = max(s._next_page, first_page)
    t = GPUTask(s, tid=1)
    a, a_va = t.buffer(contents=bytes(range(64)))
    b, b_va = t.buffer()
    t.define()
    t.install()
    return t, a, a_va, b, b_va


def _blit(t, a, b, byte, label, cmdbuf=None):
    """A fill of B then a copy of A into it, or (cmdbuf) the exec of a
    command buffer written before, which a target without those pages
    cannot be written again. Returns the command buffer."""
    if cmdbuf is None:
        fill = struct.pack("<IQQB3x", b, 0, 0x40, byte)
        copy_ = struct.pack("<IIQQQ", a, b, 0, 0x40, 0x40)
        cmdbuf = t.cmdbuf(seg(abi.ENC_BLIT, rec(306, fill) + rec(301, copy_)))
    t.exec3([cmdbuf], resources=[(a, 5), (b, 5)], fault="either", label=label)
    return cmdbuf


@scenario
def fresh_resume_ram():
    """A target whose guest RAM differs from the source's. The guest has a
    root FIFO and task 1, whose pages start at `far`; the image is loaded
    by guest address, and what the target lacks starts zeroed.
      - larger: 128 MiB where there were 64: the resume is the same as
        over the same RAM.
      - smaller: 512 KiB. The root page, FIFO and task root fit; task 1's
        pages (4 MiB in) do not, nor can the guest write there: the
        source's own blit, run again, faults the root FIFO in every setup.
      - split: the same 64 MiB as two 32 MiB ranges of the memory map
        (two host allocations), with task 1's pages straddling the seam."""
    out = []
    for name, ram, far in (("larger", [(RAM_BASE, 2 * RAM_SIZE)], 0x100),
                           ("smaller", [(RAM_BASE, 0x80000)], 0x100),
                           ("split", [(RAM_BASE, RAM_SIZE // 2), (RAM_BASE + RAM_SIZE // 2, RAM_SIZE // 2)],
                            RAM_SIZE // 2 // PAGE - 4)):
        s = Scenario("fresh_resume_ram_" + name, doc=fresh_resume_ram.__doc__)
        s.bring_up()
        t, a, a_va, b, b_va = _task_far(s, far)
        cb = _blit(t, a, b, 0x11, "source: the blit")
        t.dump(b_va, 0x40, "source: buffer B")
        s.events("source")
        suspend(s)
        s.phase(ram_ranges=ram)
        resume(s)
        s.wait(irqs=1, label="didResume's interrupt")
        s.events("resumed")
        s.sweep(0x1000, 0x1040, label="registers")
        s.submit(abi.OP_NOP, label="a NOP")
        if name == "smaller":
            _blit(t, a, b, 0, "target: the source's blit again", cmdbuf=cb)
        else:
            _blit(t, a, b, 0x22, "target: the blit")
            t.dump(b_va, 0x40, "target: buffer B")
        s.fault_state("target: after the blit")
        s.events("target")
        s.diff("target", gpa=s.pa(0), length=0x80000)
        out.append(s)
    return out


@scenario
def fresh_resume_ram_moved():
    """A target whose guest RAM is elsewhere (at 0x80000000): nothing of the
    image overlaps it, so the blob's root page does not map (in the blocks
    setup, task 0's map_memory of it returns false and the task is destroyed),
    rvg_device_did_resume has no root (no root node) and restores no FIFO
    (0x100c reads 0), task or display (the display's cursor position reads
    0xffff); it still raises its interrupt. And a target that lost only the
    first 32 KiB (the root page; the FIFO and the rest are there): the same,
    since everything hangs from the root."""
    out = []
    for name, ram in (("moved", [(0x80000000, RAM_SIZE)]), ("root_lost", [(RAM_BASE + 2 * PAGE, RAM_SIZE - 2 * PAGE)])):
        s = Scenario("fresh_resume_ram_" + name, displays=[display(0)], doc=fresh_resume_ram_moved.__doc__)
        g = Guest(s)
        suspend(s)
        # The interrupt's stamp snapshot reads the source's stamp page, which
        # the moved target lacks.
        s.phase(ram_ranges=ram, stamp_gpa=None)
        resume(s)
        s.wait(irqs=1, timeout_ms=500, label="an interrupt?")
        s.quiesce(100)
        s.events("resumed")
        s.sweep(0x1000, 0x1040, label="registers")
        s.api("display_props", display=0)
        out.append(s)
    return out


@scenario
def fresh_resume_blob_only():
    """A restore whose memory image was lost: the blob, over zeroed guest
    RAM. Every root word is 0: the root page maps (it is RAM) but names no
    task, child FIFO or display, so none comes back (the display's cursor
    position reads 0xffff); the root FIFO is recreated from the blob's
    counts over zeroed pages, and a NOP the guest writes at the next
    position completes."""
    s = Scenario("fresh_resume_blob_only", displays=[display(0)], doc=fresh_resume_blob_only.__doc__)
    g = Guest(s)
    suspend(s)
    s.phase(ram=False)
    resume(s)
    s.wait(irqs=1, label="didResume's interrupt")
    s.quiesce(100)
    s.events("resumed")
    s.sweep(0x1000, 0x1040, label="registers")
    s.api("display_props", display=0)
    s.submit(abi.OP_NOP, label="a NOP", timeout_ms=1000)
    s.events("end")
    s.diff("end", gpa=s.pa(0), length=s._next_page * PAGE)
    return s


# ---- the target's displays ----

@scenario
def fresh_resume_displays():
    """The target's displays, against the source's one display on port 0:
      - late: attached after rvg_device_will_resume and before
        rvg_device_did_resume: the same restore as attached before
        rvg_device_will_resume;
      - none: no display at all: the guest display is restored without one;
      - other: a display with another name, serial number and mode list:
        the restore's mode list differs from the page's;
      - two_ports: a device with 2 ports: the blob's port count (1)
        differs: InvalidDisplayPortCount (5), and nothing is resumed."""
    out = []
    for name in ("late", "none", "other", "two_ports"):
        s = Scenario("fresh_resume_displays_" + name, displays=[display(0)], doc=fresh_resume_displays.__doc__)
        g = Guest(s)
        suspend(s)
        if name == "late":
            s.phase(displays=None)
            s.api("willResume", slot="s")
            s.api("new_display", spec=display(0))
            s.api("didResume")
        elif name == "none":
            s.phase(displays=None)
            resume(s)
        elif name == "other":
            s.phase(displays=[display(0, serial=77, name="other", modes=((128, 64, 30), (64, 32, 60)))])
            resume(s)
        else:
            s.phase(descriptor={"port_count": 2}, displays=[display(1)])
            s.api("willResume", slot="s")
            s.sweep(0x1000, 0x1040, label="registers")
            out.append(s)
            continue
        s.wait(irqs=1, label="didResume's interrupt")
        s.quiesce(100)
        s.events("resumed")
        g.g.diff_page(0, "page")
        if name != "none":
            s.api("display_props", display=0)
        s.submit(abi.OP_NOP)
        s.events("end")
        out.append(s)
    return out


# ---- suspended with work in every state ----

@scenario
def fresh_resume_queued():
    """Commands written into the root FIFO's and child FIFO 1's rings but
    not rung when the host suspends: the device never saw them (the blob's
    written count is the last doorbell), and the target runs them when the
    guest rings, after the resume."""
    s = Scenario("fresh_resume_queued", doc=fresh_resume_queued.__doc__)
    f = s.bring_up()
    c = ChildFIFO(s, 1)
    c.setup_node()
    c.define()
    t = GPUTask(s, tid=1)
    b, b_va = t.buffer()
    t.define()
    t.install()
    s.submit(abi.OP_NOP)
    st_c = c.cmd(abi.OP_NOP)
    fill = struct.pack("<IQQB3x", b, 0, 0x40, 0x3C)
    st = f.cmd(abi.OP_EXEC_INDIRECT3, _exec_payload(t, fill, b))
    s.events("queued")
    suspend(s)
    s.phase()
    resume(s)
    s.wait(irqs=1)
    s.events("resumed")
    s.sweep(0x1000, 0x1040, label="registers")
    f.doorbell()
    f.wait_stamp(st, label="the queued exec")
    c.doorbell()
    c.wait_stamp(st_c, label="the queued child NOP")
    t.dump(b_va, 0x40, "buffer B")
    s.events("end")
    s.diff("end", gpa=s.pa(0), length=s._next_page * PAGE)
    return s


def _exec_payload(t, fill, b):
    """ExecIndirect3's payload (task.py exec3) for one blit segment that
    fills B, left for the caller to write and ring."""
    va, n = t.cmdbuf(seg(abi.ENC_BLIT, rec(306, fill)))
    return struct.pack("<III", t.tid, 1, 1) + struct.pack("<II", b, 5) + struct.pack("<QQ", va, n)


@scenario
def fresh_resume_blocked():
    """Work blocked when the host suspends: the root FIFO holds an exec
    whose barrier waits for child FIFO 1's stamp 1, and child 1's NOP is
    written but not rung. rvg_device_finish_suspend abandons the wait (the
    FIFOs are drained and released); the blob's read count is before the exec,
    so the target parses it again and waits again, and the exec runs once the
    guest rings child 1."""
    s = Scenario("fresh_resume_blocked", doc=fresh_resume_blocked.__doc__)
    f = s.bring_up()
    c = ChildFIFO(s, 1)
    c.setup_node()
    c.define()
    t = GPUTask(s, tid=1)
    b, b_va = t.buffer()
    t.define()
    t.install()
    st_c = c.cmd(abi.OP_NOP)
    fill = struct.pack("<IQQB3x", b, 0, 0x40, 0x5E)
    st = f.cmd(abi.OP_EXEC_INDIRECT3, _exec_payload(t, fill, b), barriers=[(c.stamp_index, st_c)])
    f.doorbell()
    s.wait(mmio=abi.R_FIFO_READ, mmio_ne=0, label="the root FIFO has begun")
    s.quiesce(100)
    s.r(abi.R_FIFO_READ, "the root FIFO's read count, waiting")
    s.events("blocked")
    suspend(s)
    t.dump(b_va, 0x40, "buffer B at the suspend")
    s.phase()
    resume(s)
    s.wait(irqs=1)
    s.quiesce(100)
    s.events("resumed")
    s.sweep(0x1000, 0x1040, label="registers")
    c.doorbell()
    f.wait_stamp(st, label="the blocked exec")
    t.dump(b_va, 0x40, "buffer B")
    s.events("end")
    s.diff("end", gpa=s.pa(0), length=s._next_page * PAGE)
    return s


@scenario
def fresh_resume_faulted():
    """suspend_resume_faults across processes: the root FIFO and child
    FIFO 1 faulted at suspend. The target's root FIFO resumes faulted (key
    19 carries its fault offset, 0x102c reads 1); child 1 is rebuilt from
    its node, whose fault offset the guest page keeps. 0x1024 and 0x1028
    resume them, and both run a NOP."""
    s = Scenario("fresh_resume_faulted", doc=fresh_resume_faulted.__doc__)
    s.bring_up()
    c = ChildFIFO(s, 1)
    c.setup_node()
    c.define()
    s.submit(99, fault=True)
    c.submit(99, fault=True)
    s.fault_state("faulted")
    s.events("faulted")
    suspend(s)
    s.phase()
    resume(s)
    s.wait(irqs=1)
    s.quiesce(100)
    s.events("resumed")
    s.fault_state("resumed")
    s.dump(c.node_gpa(), 0x14, "child node")
    s.resume_root()
    s.w(abi.R_RESUME_CHILD, 1)
    s.submit(abi.OP_NOP)
    c.submit(abi.OP_NOP)
    s.fault_state("after resuming both")
    s.events("end")
    return s


@scenario(setups=("isolate", ("isolate", "map"), ("isolate", "blocks")))
def fresh_resume_dead_task():
    """interplay_iso_suspend_with_dead_task across processes: task 1's
    service killed (an exec range past the task's end) and task 2 alive
    when the host suspends. The dead task's flush is answered at once
    and the blob is written; the target rebuilds both tasks from their
    root words, task 1 with a new service (or in the target's own
    process, in map and blocks), and both work. Isolate only at the
    source: elsewhere the death is the host's."""
    from scenarios.interplay_isolation import DEATH_PAST_END, REGION, Doomed, Healthy
    from pgtest.diff_util import fault_state, resume as resume_fifo
    s = Scenario("fresh_resume_dead_task", setups=("isolate", ("isolate", "map"), ("isolate", "blocks")),
                 doc=fresh_resume_dead_task.__doc__)
    s.bring_up()
    d = Doomed(s, 1)
    h = Healthy(s, 2)
    last = h.queue(0)
    s.doorbell()
    s.fifo.wait_stamp(last, label="task 2 alone")
    st = d.death(DEATH_PAST_END)
    s.doorbell()
    s.wait_fault(label="task 1's service dies")
    fault_state(s, s.fifo, "the death")
    resume_fifo(s, s.fifo)
    s.fifo.wait_stamp(st, label="the death's stamp")
    s.events("before the suspend")
    suspend(s)
    s.phase()
    resume(s)
    s.wait(irqs=1)
    s.quiesce(100)
    s.events("resumed")
    d.run(d.fill(0x55), label="task 1 after the resume")
    d.dump(d.buf_va, 0x40, "task 1's buffer")
    last = h.queue(1)
    s.doorbell()
    s.fifo.wait_stamp(last, label="task 2 after the resume")
    s.cmp(h.gpa(h.va(0)), h.gpa(h.va(1)), REGION, label="task 2: after the resume == before")
    d.delete()
    h.delete()
    s.events("end")
    return s


@scenario
def fresh_resume_paused():
    """A device paused when the host suspends (pause, then
    rvg_device_will_suspend and rvg_device_finish_suspend), with no display
    page: a NOP rung while paused does not run (pause stops the FIFOs), so the
    blob's read count (key 7) is before it and its written count (key 6) after.
    The pause is not in the blob: the target runs unpaused, runs the NOP at
    once, and unpause there, with no guest display, is harmless."""
    s = Scenario("fresh_resume_paused", doc=fresh_resume_paused.__doc__)
    f = s.bring_up()
    s.submit(abi.OP_NOP)
    s.api("pause")
    st = f.cmd(abi.OP_NOP)
    f.doorbell()
    f.wait_stamp(st, timeout_ms=500, label="a NOP while paused")
    s.quiesce(100)
    s.events("paused")
    suspend(s)
    s.phase()
    resume(s)
    s.wait(irqs=1)
    s.quiesce(100)
    s.events("resumed")
    s.sweep(0x1000, 0x1040, label="registers")
    f.wait_stamp(st, label="the NOP rung while paused")
    s.submit(abi.OP_NOP)
    s.api("unpause", label="on the target, never paused")
    s.quiesce(100)
    s.events("end")
    return s


@scenario
def fresh_resume_paused_display():
    """pause with a display page connected, then rvg_device_will_suspend and
    rvg_device_finish_suspend: rvg_device_finish_suspend pauses the guest
    display again, and a guest display's second pause asserts
    (suspend_host_asserts_pause_twice): the host dies in
    rvg_device_finish_suspend, so a paused VM with a display cannot be
    suspended through the C API."""
    s = Scenario("fresh_resume_paused_display", displays=[display(0)], doc=fresh_resume_paused_display.__doc__)
    g = DisplayGuest(s, task0=False)
    g.connect(0, enabled=0xF)
    g.ack(0, 1)
    s.events("connected")
    s.api("pause")
    suspend(s)
    return s


@scenario
def fresh_resume_mid_frame():
    """A frame in flight when the host suspends: the root FIFO's present (7, of
    a task 0 backing) waits on a barrier for child FIFO 1's stamp, whose NOP is
    written but not rung. The suspend abandons the wait; the target restores
    the display (a present-7 display restores through the mapped path only, so
    nothing is presented again, GuestDisplay::finish_restore), parses the
    present again, and presents it once the guest rings child 1."""
    s = Scenario("fresh_resume_mid_frame", displays=[display(0)], doc=fresh_resume_mid_frame.__doc__)
    g = DisplayGuest(s)
    r, va = g.backing(16, 8, pixels=pattern(16, 8))
    g.install()
    g.connect(0, enabled=0xF)
    g.ack(0, 1)
    c = ChildFIFO(s, 1)
    c.setup_node()
    c.define()
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, r))
    s.wait(cb="newFrame")
    st_c = c.cmd(abi.OP_NOP)
    st = s.fifo.cmd(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, r), barriers=[(c.stamp_index, st_c)])
    s.fifo.doorbell()
    s.wait(mmio=abi.R_FIFO_READ, mmio_ne=0, label="the root FIFO has begun")
    s.quiesce(100)
    s.events("the second present waits")
    suspend(s)
    g.diff_page(0, "suspended")
    s.phase()
    resume(s)
    s.wait(irqs=1)
    s.quiesce(100)
    s.events("resumed")
    s.api("present_counts", display=0)
    c.doorbell()
    s.fifo.wait_stamp(st, label="the held present")
    s.wait(cb="newFrame", label="its frame")
    s.events("presented")
    s.api("present_counts", display=0)
    g.diff_page(0, "end")
    return s


@scenario(host=True)
def fresh_resume_held_encode():
    """A frame the host holds (encoded, not committed) across the suspend:
    the host commits it after rvg_device_finish_suspend, and its pixels are the
    frame presented before; the target, whose display restores without a
    present (present 7), encodes the guest's next present."""
    s = Scenario("fresh_resume_held_encode", host=True, displays=[display(0)], doc=fresh_resume_held_encode.__doc__)
    g = DisplayGuest(s)
    r, va = g.backing(16, 8, pixels=pattern(16, 8, 3))
    g.install()
    g.connect(0, enabled=0xF)
    g.ack(0, 1)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, r))
    s.wait(cb="newFrame")
    s.encode(0, 16, 8, hold=True, label="held")
    suspend(s)
    s.encode_commit(0, label="the held frame, after finishSuspend")
    s.events("suspended")
    s.phase()
    resume(s)
    s.wait(irqs=1)
    s.quiesce(100)
    s.events("resumed")
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, r))
    s.wait(cb="newFrame")
    s.encode(0, 16, 8, label="the target's present")
    s.events("end")
    return s


@scenario(host=True)
def fresh_resume_efi():
    """suspend_resume_efi across processes, and on: EFI enabled at the
    suspend is replayed first by rvg_device_did_resume (keys 12-17), the
    target's EFI display presents the guest's framebuffer; a guest write and a
    trace fire present the new frame."""
    s = Scenario("fresh_resume_efi", host=True, displays=[display(0)], descriptor={"trace": True},
                 doc=fresh_resume_efi.__doc__)
    s.w(abi.R_VERSION, abi.MAX_VERSION)
    e = EFI(s)
    e.write_frame(16, 8, 64, seed=9)
    e.setup(16, 8, 64)
    s.wait(cb="newFrame")
    s.events("presented")
    suspend(s)
    s.phase()
    resume(s)
    s.wait(cb="newFrame", label="the restore presents")
    s.quiesce(100)
    s.events("resumed")
    e.regs("resumed")
    s.encode(0, 16, 8, label="after the resume")
    e.write_frame(16, 8, 64, seed=0x40)
    s.trace_fire(e.gpa(0x10), 4)
    s.wait(cb="newFrame", cb_count=2, label="the new frame")
    s.quiesce(100)
    s.events("a new frame")
    s.encode(0, 16, 8, label="the new frame")
    return s


@scenario(setups=("map", "isolate", ("map", "isolate"), ("isolate", "map")), host=True)
def fresh_resume_iosfc_live():
    """The IOSurface host device live when the host suspends: surface 1
    mapped and presented (8) on the display. In one process the restore
    finds the surface on the process's IOSurface host device
    (suspend_restore_mapped); in a new process that device is new and has
    no surfaces (the C API carries no IOSurface device state), so
    the restore's lookup of mapping 1 fails, and the display shows nothing
    until the guest starts its IOSurface ring again, maps surface 1 and
    presents it. Map and isolate: the mapper needs a memory map."""
    setups = ("map", "isolate", ("map", "isolate"), ("isolate", "map"))
    s = Scenario("fresh_resume_iosfc_live", setups=setups, host=True, iosfc=True, displays=[display(0)],
                 doc=fresh_resume_iosfc_live.__doc__)
    g = DisplayGuest(s)
    g.connect(0, enabled=0xF)
    g.ack(0, 1)
    d = IOSFC(s, v2=True)
    d.start()
    root, data = d.surface_pages(1)
    s.mem(s.pa(data), pattern(16, 8, 5))
    d.put_desc(1, desc_v2(root, 16, 8))
    d.submit((CMD_MAP, 1))
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1))
    s.wait(cb="newFrame")
    s.events("presented")
    s.diff("presented")
    suspend(s)
    s.diff("suspended")
    s.phase()
    resume(s)
    s.wait(irqs=1)
    s.quiesce(200)
    s.events("resumed")
    s.diff("resumed")
    for r in IREGS:
        s.iosfc_r(r, "the target's IOSurface registers")
    d.written = 0
    d.start()
    d.status(d.submit((CMD_MAP, 1)), "the guest maps surface 1 again")
    s.submit(abi.OP_DISPLAY_SWAP_MAPPING, p_swap(0, 1))
    s.wait(cb="newFrame")
    s.events("presented again")
    s.encode(0, 16, 8, label="surface 1")
    d.status(d.submit((CMD_TEST, 1)), "a test of surface 1")
    s.diff("after the test", gpa=s.pa(data), length=PAGE)
    return s


# ---- the binary version a failed resume leaves ----

@scenario
def fresh_resume_version_f5():
    """A rvg_device_will_resume refused at the capability check (a device-info
    key 1 of 0xffffffff: IncompatibleDevice, 4) has already set the binary
    version from the blob: on a device the guest never versioned, 0x1034 then
    reads 0xf5, and a later suspend's blob carries key 2 = 0xf5. A new process
    resumes that blob as a guest of version 0xf5. The control, with no failed
    resume first, carries 0."""
    out = []
    for name, fail in (("f5", True), ("f5_control", False)):
        s = Scenario("fresh_resume_version_" + name, doc=fresh_resume_version_f5.__doc__)
        s.r(abi.R_VERSION, "a fresh device")
        if fail:
            s.api("willResume", hex=blob(state(), [(1, 0xFFFFFFFF)]).hex(), label="refused at the capability check")
            s.r(abi.R_VERSION, "after the refusal")
        suspend(s)
        s.phase()
        resume(s)
        s.quiesce(100)
        s.events("resumed")
        s.sweep(0x1000, 0x1040, label="registers")
        out.append(s)
    return out


# ---- the blob, mutated ----

INFO_MAX = 45
# The blob's fields every host writes at the same offsets (docs/testing.md
# 6.24): the header (+0, +4), the registers section's tag and length (+8,
# +12) and its 20 registers (+16 to +92), the device-info section's tag,
# length and max key (+96 to +104), then the pairs from +108. Keys 1-13 are
# written on every host, so pairs 0-12 are too; the pairs after them, and
# with them the blob's length, depend on the host.
REGISTERS_AT, INFO_HEADER_AT, PAIRS_AT = 16, 96, 108
FIXED_PAIRS = 13
FIXED_LEN = PAIRS_AT + 8 * FIXED_PAIRS
REGISTER_KEYS = tuple(range(1, 20)) + (21,)
# The section tags, the device-info keys either side of the ones a resume
# reads (1-42, 44), the registers section's length, and the extremes.
VALUES = (0, 1, 2, 3, 42, 43, 44, 45, 80, 0xFFFFFFFF)


def _mutation_base(name, doc):
    """A device-written blob with device info (max key 45) from a brought-up
    device, in slot s."""
    s = Scenario(name, host=True, doc=doc)
    s.bring_up()
    s.submit(abi.OP_GET_DEVICE_INFO, struct.pack("<III", INFO_MAX, 64, s.pfn(s.alloc_page())))
    suspend(s)
    return s


def _offer(s, label, **kw):
    """One mutation on a fresh device over the same guest RAM: its answer,
    then the registers it leaves."""
    s.api("recreate", label=label)
    s.api("willResume", slot="s", label=label, **kw)
    s.sweep(0x1000, 0x1040, label=label)


def _fields(s, offsets):
    for off in offsets:
        for v in VALUES:
            _offer(s, "u32 +%d = %#x" % (off, v), patch=[[off, v]])


MUTATE_DOC = """A device-written blob (device info max key 45), mutated, each
    mutation offered to rvg_device_will_resume on a fresh device over the
    same guest: its answer (RvgResumeError 0-5, or accepted) and the
    registers the refusal or acceptance leaves (the binary version is set
    before the capability check). The parse checks the header, then the
    sections (their bounds, tags, lengths and device-info keys); then
    rvg_device_will_resume checks the binary version, the capabilities (host
    values: keyed by host), the port count and the connection types. Only
    the fields every host writes at the same offsets are mutated: the
    header, the registers, the device-info section's header and max key, and
    device-info pairs 0-12. Each u32 field is set to %s."""


@scenario(host=True)
def fresh_resume_mutate():
    """Every truncation of the blob's fixed part at a 4-byte boundary, and
    every u32 field of it set to each boundary value: grouped a scenario for
    the header, a register each, the device-info section's header, and six
    device-info pairs each."""
    doc = MUTATE_DOC % ", ".join(hex(v) for v in VALUES)
    out = []
    s = _mutation_base("fresh_resume_mutate_truncate",
                       doc + " Here: every truncation, 0 to %d bytes, then the blob as written." % (FIXED_LEN - 4))
    for cut in range(0, FIXED_LEN, 4):
        _offer(s, "truncate to %d" % cut, truncate=cut)
    _offer(s, "the blob as written")
    out.append(s)
    s = _mutation_base("fresh_resume_mutate_header",
                       doc + " Here: the magic, the version, and the registers section's tag and length"
                       " (+0 to +12).")
    _fields(s, range(0, REGISTERS_AT, 4))
    out.append(s)
    for i, k in enumerate(REGISTER_KEYS):
        s = _mutation_base("fresh_resume_mutate_state_%02d" % k,
                           doc + " Here: register %d, device-state key %d (+%d)." % (i, k, REGISTERS_AT + 4 * i))
        _fields(s, (REGISTERS_AT + 4 * i,))
        out.append(s)
    s = _mutation_base("fresh_resume_mutate_info_header",
                       doc + " Here: the device-info section's tag, length and max key (+%d to +%d)."
                       % (INFO_HEADER_AT, PAIRS_AT - 4))
    _fields(s, range(INFO_HEADER_AT, PAIRS_AT, 4))
    out.append(s)
    for first in range(0, FIXED_PAIRS, 6):
        last = min(first + 5, FIXED_PAIRS - 1)
        s = _mutation_base("fresh_resume_mutate_info_%02d_%02d" % (first, last),
                           doc + " Here: device-info pairs %d to %d (+%d to +%d)."
                           % (first, last, PAIRS_AT + 8 * first, PAIRS_AT + 7 + 8 * last))
        _fields(s, range(PAIRS_AT + 8 * first, PAIRS_AT + 8 * (last + 1), 4))
        out.append(s)
    return out
