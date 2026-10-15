"""Guest ABI versions across whole sessions (scenarios/interplay_versions.py):
what the binary version a guest settles on decides, restated from the
source so a scenario can say what each version's guest should see, and a
session (pgtest/session.py) that exercises every gated feature at once.

The version: set_binary_version (reims-vgpu-core's guest::abi::features)
ignores a request with bit 31 set, stores 1 and 236 with bit 31 set
(unsupported), keeps 0, clamps above 0xf5 to 0xf5, and steps an unknown
request down to the nearest valid version; the features are recomputed
from the result (features_for_version), a cumulative switch in
which each case enables its flags and falls through to the lower ones,
with three overrides after it (v == 40: no dual-plane textures; 7 <= v <
100: no shared textures; 7 <= v < 104: no sunburst display). Every write
that is not ignored sets maxFIFOCount to 32 before the switch, so 0 and
the unsupported versions still allow 31 child FIFOs; 61 and up allow 63.

GATES below is each flag's lowest version, and where the flag is read:
only the flags something reads are listed (the rest -- objectTables 2,
displayDoorbell 4, sunburst, synchronizeDiscardChildResources 30,
displayDimensionFloats 42, the 101-104 and 110-112 render flags,
supportsComputePassDescriptorDispatchType 200, supportsObjectUniqueIdentifier
201, supportsSharedStorageHeaps 203, supportsProtectionOptionsEnvelope 205,
supportsInsertCompressedTextureReinterpretationFlush 206, supportsCorrect2pXR10A8
209, supportsCPUBuffers 230, supportsProtectedTextureOptions 232,
supportsDispatchThreadsIndirect 233, supportsTextureDescriptor2 237 -- are
set and never read, so a guest below them sees nothing refused).

Module paths name reims-vgpu-core's items. Nothing here is imported from
the library's code.
"""
import struct

from . import abi
from .abi import PAGE
from .display_guest import (SS_ASLEEP, SS_COLOR_MATRICES, SS_ICC, floats, icc_profile, p_compositor, p_icc,
                            p_props, p_sleep, pattern)
from .fifo import ChildFIFO
from .fifo_util import probe
from .resources_build import (Depth32Float_Stencil8, RTask, b_fill_buf, buffer_rec, desc1, heap_rec,
                              shared_backing_rec, shared_texture_rec, simple_dim, texture_rec, dimension, mip)
from .session import FH, FW, Session
from .task import seg

VALID = set(abi.VALID_VERSIONS)

# Each flag something reads, by the lowest version whose case sets it
# (features_for_version), and the reader.
GATES = {
    "efiDisplay": 3,             # set_binary_version tears EFI down without it
    "sharedTextures": 5,         # the shared texture's create (resources::manager::texture),
                                 # device info key 19; off for 7-99
    "mapperIOSurfaces": 10,      # an in-process task's mapper surface lookup (tasks::local: an assert)
    "syncAndDiscard": 10,        # opcode 62
    "displayMapperSurface": 10,  # opcode 8
    "discardResources": 11,      # opcode 63
    "displaySleep": 21,          # opcode 9
    "RGhA": 22,                  # the guest display's present (display::guest_display::present): logs, allows
    "dualPlane": 31,             # types 12 and 18 (resources::manager::texture), key 12; off at 40
    "displaySetProperties": 40,  # opcode 10
    "metalHeaps": 43,            # opcode 64, keys 14-16
    "bufferFromIOSurface": 60,   # key 17, an in-process task's creation (tasks::local::new)
    "fifo64": 61,                # maxFIFOCount 64
    "s8one": 62,                 # s8ByteCountForD32S8 1, else 4 (resources::texture::pixel_format)
    "programmableSamplePositions": 105,  # key 21
    "rateMaps": 106,             # opcode 66
    "tileShaders": 107,          # keys 23-27
    "compositor": 108,           # opcode 11
    "exec3": 109,                # opcode 67
    "dynamicStride": 111,        # key 28 bit 1
    "mippedMultiplane": 113,     # the dual-plane blit texture and shared backing
                                 # (resources::texture::blit, resources::texture::shared)
    "cbJump": 202,               # key 36
    "heapTracking": 204,         # a detached heap (resources::heap::detach)
    "argumentBuffers": 207,      # opcode 68, the service's lists (gpu_task::handshake), read-only
                                 # buffers and heaps there (resources::buffer::detach,
                                 # resources::heap::detach)
    "icc": 208,                  # opcode 12
    "icb": 231,                  # the service's GPU-address registration (gpu_task::handshake)
    "infoIndirect": 235,         # opcode 69
    "connectionTypes": 238,      # the resume's check (device::lifecycle)
    "multipleDisplays": 245,     # 0x122c and 0x1234 (device::mmio)
}

UNSUPPORTED = 0x80000000


def effective(req, prev=0):
    """The version set_binary_version stores for a request, at feature
    level 0."""
    if req & 0x80000000:
        return prev
    if req in (1, 236):
        return UNSUPPORTED | req
    if req == 0 or req in VALID:
        return req
    if req > 0xF5:
        return 0xF5
    v = req
    while v and v not in VALID:
        v -= 1
    return v


def has(v, flag):
    """Whether version v (as stored) has the flag set."""
    if v is None or v & UNSUPPORTED or v not in VALID:
        return False
    if flag == "dualPlane" and v == 40:
        return False
    if flag == "sharedTextures" and 7 <= v < 100:
        return False
    return v >= GATES[flag]


def fifo_count(v):
    """maxFIFOCount: 0 before any version, 32 for any written one, 64 from
    61."""
    if v is None:
        return 0
    return 64 if has(v, "fifo64") else 32


def u32s(*v):
    return struct.pack("<%dI" % len(v), *v)


DS = desc1(fmt=Depth32Float_Stencil8, usage=3)
DS2 = desc1(fmt=Depth32Float_Stencil8, usage=3, mips=2)


def ds_dims(levels=1, s8=1, w=4, h=4):
    """Depth32Float_Stencil8's two planes: 4 bytes a texel at +0, s8 bytes
    at +0x100 (as scenarios/textures_dual_plane.py lays them out; the
    stencil texel is s8ByteCountForD32S8)."""
    def plane(bpp, base):
        mips, off, ww, hh = [], 0, w, h
        for _ in range(levels):
            mips.append(mip(off, ww * hh * bpp, ww * bpp, ww, hh))
            off += ww * hh * bpp
            ww, hh = max(ww // 2, 1), max(hh // 2, 1)
        return dimension(mips, base=base, block=(1, 1, 1, bpp))
    return [plane(4, 0), plane(s8, 0x100)]


def exec2_payload(tid, cmdbufs, resources=()):
    """ExecIndirect2 (55): {u32 taskID, u32 resourceCount, u32
    cmdBufferCount}, 24-byte resources {u32 ref, u8 invalidateHost, u8
    invalidateGuest, u8 synchronize, pad, u64, u64}, then {u64 VA, u64
    length} per command buffer."""
    p = struct.pack("<III", tid, len(resources), len(cmdbufs))
    p += b"".join(struct.pack("<IBBBxQQ", r, f & 1, (f >> 1) & 1, (f >> 2) & 1, 0, 0) for r, f in resources)
    p += b"".join(struct.pack("<QQ", va, n) for va, n in cmdbufs)
    return p


def exec3_payload(tid, cmdbufs, resources=()):
    """ExecIndirect3 (67) and InfoIndirect (69): 8-byte resources {u32 ref,
    u32 flags}."""
    p = struct.pack("<III", tid, len(resources), len(cmdbufs))
    p += b"".join(struct.pack("<II", r, f) for r, f in resources)
    p += b"".join(struct.pack("<QQ", va, n) for va, n in cmdbufs)
    return p


def outcome(s, opcode, payload, ok, label, fifo=None, **kw):
    """One command whose outcome the scenario states: ok True signals,
    False faults (then resumed with 0x1024), None either (where the host
    setups differ: the observation says which, and the FIFO is resumed
    either way; 0x1024 on a FIFO that is not faulted changes nothing)."""
    if ok is None:
        f = fifo or s.fifo
        assert f.fid == 0
        st = s.submit(opcode, payload, fault="either", label=label, **kw)
        s.fault_state(label)
        s.resume_root()
        f.wait_stamp(st, label=label + ": published")
        return st
    return probe(s, opcode, payload, "signal" if ok else "fault", label=label, fifo=fifo, **kw)


def heap_query(tid, reply_va):
    """HeapTextureSizeAndAlign (64): taskID@0 replyVO u64@4 replyLen u64@12
    payloadLen u32@20, then a serialized texture
    descriptor (2D RGBA8Unorm 256x256, private), as fifo_opcodes sends it."""
    word0 = 2 | (1 << 8) | (70 << 16)
    ser = struct.pack("<II", 22, 0) + struct.pack("<IIIIHHHHQ", word0, 256, 256, 1, 1, 1, 1, 0x20, 0)
    return struct.pack("<IQQI", tid, reply_va, 16, len(ser)) + ser


class ProbeTask(RTask):
    """Task `tid` with a shared probe buffer, a reply page for opcode 64,
    and one object of each version-gated kind (docs/testing.md, sections
    6.3, 6.11, 6.19): two Depth32Float_Stencil8 textures (type 12, one and
    two levels, the stencil texel the version's s8ByteCountForD32S8), a
    shared texture (type 16) over task 0's backing k17, a tracked private
    heap (13, flags 3), a read-only buffer (1, flags 2) and a read-only heap
    (13, flags 5). Each object is created by an exec that lists it beside
    the probe buffer and fills that exec's slot of it, so which creations
    fault reads back as zero slots."""

    SLOTS = 16

    def __init__(self, s, tid, v, k17, pages=64):
        super().__init__(s, tid=tid, pages=pages)
        self.v = v
        self.probe, self.probe_va = self.shared(PAGE)
        self.reply_va = self.area(PAGE, fill=0xA5)
        s8 = 1 if has(v, "s8one") else 4
        self.objects = [
            ("two-plane D32S8", self.add(abi.T_DUAL_PLANE_TEXTURE,
                                        texture_rec(PAGE, self.area(PAGE) // PAGE, ds_dims(s8=s8), DS))),
            ("two-plane D32S8, 2 levels", self.add(abi.T_DUAL_PLANE_TEXTURE, texture_rec(
                PAGE, self.area(PAGE) // PAGE, ds_dims(s8=s8, levels=2), DS2))),
            ("shared texture", self.add(abi.T_SHARED_TEXTURE, shared_texture_rec(k17))),
            ("tracked heap", self.add(abi.T_HEAP, heap_rec(self.area(4 * PAGE) // PAGE, 4 * PAGE, 3))),
            ("read-only buffer", self.add(abi.T_BUFFER, buffer_rec(PAGE, self.area(PAGE) // PAGE, 2))),
            ("read-only heap", self.add(abi.T_HEAP, heap_rec(self.area(4 * PAGE) // PAGE, 4 * PAGE, 5))),
        ]

    def fill_cb(self, slot):
        self.reset_cmdbufs()
        return self.cmdbuf(seg(abi.ENC_BLIT, b_fill_buf(self.probe, slot * 0x10, 0x10, 0x40 + slot)))

    def run55(self, slot, ok=True, extra=(), label=None):
        return outcome(self.s, abi.OP_EXEC_INDIRECT2,
                       exec2_payload(self.tid, [self.fill_cb(slot)], [(self.probe, 0)] + list(extra)), ok,
                       label or "task %d: 55, slot %d" % (self.tid, slot))

    def run67(self, slot, ok, opcode=abi.OP_EXEC_INDIRECT3, label=None):
        return outcome(self.s, opcode, exec3_payload(self.tid, [self.fill_cb(slot)], [(self.probe, 0)]), ok,
                       label or "task %d: %d, slot %d" % (self.tid, opcode, slot))

    def objects_probe(self, first_slot, oks=None, label=""):
        """Each gated object in its own exec (55, which every version has).
        oks: the outcome for each (None: either)."""
        for i, (name, ref) in enumerate(self.objects):
            ok = None if oks is None else oks[i]
            self.run55(first_slot + i, ok=ok, extra=[(ref, 0)], label="task %d: %s%s" % (self.tid, name, label))

    def dump_probe(self, label):
        self.dump(self.probe_va, 0x10 * self.SLOTS, "task %d probe: %s" % (self.tid, label))

    def root_dump(self, label):
        self.s.dump(self.s.pa(self.root), abi.TASK_ROOT.size, "task %d root: %s" % (self.tid, label))


class VGuest(Session):
    """A Session whose guest settles on a version and keeps track of it:
    two display ports (both with displays) unless told otherwise, and a
    kernel task that also holds a guest ICC profile and a shared texture
    backing (type 17) for the probe tasks' shared textures."""

    def __init__(self, name, doc, ports=2, setups=None, efi=False, iosfc=False, host=True, watchdog_s=120,
                 exclusive=False):
        kw = {} if setups is None else {"setups": setups}
        super().__init__(name, doc, ports=ports, efi=efi, iosfc=iosfc, host=host, watchdog_s=watchdog_s,
                         exclusive=exclusive, **kw)
        self.v = None
        self.probes = {}

    # ---- the version ----

    def set_version(self, req, label=None):
        """0x1034 = req, then what the guest reads back: 0x1034, 0x122c,
        0x1234."""
        s = self.s
        self.v = effective(req, self.v or 0)
        lab = label or "version 0x%x" % req
        s.w(abi.R_VERSION, req)
        s.r(abi.R_VERSION, lab)
        s.r(abi.R_PORT_COUNT, lab)
        s.r(abi.R_PORT_TYPES, lab)
        self.version = self.v

    def has(self, flag):
        return has(self.v, flag)

    def unsupported(self):
        return bool(self.v & UNSUPPORTED)

    # ---- the kernel task ----

    def kernel(self, pages=32, frames=2):
        """Task 0 as Session.kernel_task builds it (a glyph, `frames` BGRA
        backings, type 4), plus the ICC profile (pgtest.display_guest.
        icc_profile) and a 4 x 4 RGBA shared texture backing (type 17)."""
        t = self.t0 = RTask(self.s, tid=0, pages=pages)
        self.glyph_va = t.reserve(PAGE)
        t.write(self.glyph_va, struct.pack("<I", 0x80402010) * 4 + struct.pack("<I", 0xFF00FF00) * 4)
        for i in range(frames):
            va = t.reserve(PAGE)
            t.write(va, pattern(FW, FH, 0x30 * i + 1))
            ref = t.ol.add(abi.T_IOSURFACE_BACKING, struct.pack("<QIIB3x", PAGE, va // PAGE,
                                                                 struct.unpack(">I", b"BGRA")[0], 1)
                           + struct.pack("<IIII", 0, FW, FH, (FW * 4) | 4 << 24))
            self.backings.append((ref, va))
        prof = icc_profile(2.2)
        self.icc_va = t.reserve(PAGE)
        t.write(self.icc_va, prof)
        self.icc_len = len(prof)
        self.k17 = t.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, t.area(PAGE) // PAGE,
                                                                    [simple_dim(4, 4)], desc1(usage=3)))
        t.define(label="define task 0")
        t.install(label="task 0's object list")

    # ---- displays ----

    def display_ops(self, port=0, frame=1):
        """The display opcodes a version gates, on a connected port with a
        frame presented: sleep and wake (9, 21), a colour matrix (10, 40),
        compositor parameters (11, 108), the guest ICC profile (12, 208),
        each refused below its version (9 and 11 fault; 10 and 12 log and
        signal); then a present shows what reached
        the display."""
        s = self.s
        if self.has("displaySleep"):
            s.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(port, 1), label="sleep %d" % port)
            s.dump(self.ss(port, SS_ASLEEP), 4, "port %d asleep" % port)
            s.submit(abi.OP_DISPLAY_SLEEP_STATE, p_sleep(port, 0), label="wake %d" % port)
            self.expect("newFrame", label="woken %d: the current frame" % port)
        else:
            outcome(s, abi.OP_DISPLAY_SLEEP_STATE, p_sleep(port, 1), False, "sleep %d refused" % port)
        s.dump(self.ss(port, SS_ASLEEP), 4, "port %d awake" % port)
        s.submit(abi.OP_DISPLAY_SET_PROPERTIES,
                 p_props(port, [(0, floats(0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0))]), label="colour matrix %d" % port)
        s.dump(self.ss(port, SS_COLOR_MATRICES), 0x64, "port %d colour matrices" % port)
        outcome(s, abi.OP_DISPLAY_COMPOSITOR_PARAMETERS, p_compositor(port, 10, 20, 5, 2), self.has("compositor"),
                "compositor parameters %d" % port)
        s.submit(abi.OP_DISPLAY_SET_GUEST_ICC_PROFILE, p_icc(port, self.icc_len, self.icc_va, self.icc_len),
                 label="ICC profile %d" % port)
        s.dump(self.ss(port, SS_ICC), 0x10, "port %d ICC fields" % port)
        self.present(port, frame)

    # ---- child FIFOs ----

    def children_at_limit(self, label=""):
        """The highest child the version allows (maxFIFOCount - 1) is
        defined, the next one refused (create_fifo); returns the top
        child."""
        n = fifo_count(self.v)
        top = self.child(n - 1)
        over = ChildFIFO(self.s, n)
        over.setup_node()
        outcome(self.s, abi.OP_DEFINE_CHILD_FIFO, u32s(n), False, "define child %d%s" % (n, label))
        return top

    # ---- the probe task ----

    def probe_task(self, tid=1):
        t = ProbeTask(self.s, tid, self.v, self.k17)
        t.define(label="define task %d" % tid)
        t.install(label="task %d's object list" % tid)
        self.probes[tid] = t
        return t

    def task_ops(self, t, label=""):
        """The task opcodes a version gates, on task t: 55 always, 67 (109),
        69 (235), 62 (10) and 63 (11) with no references, 64 (43, its reply
        dumped), then each gated object (outcomes the setups decide), then
        68 (207) with the task's own list and a placement list of zeros,
        and a 55 after it."""
        s = self.s
        t.run55(0, label="task %d: 55%s" % (t.tid, label))
        t.run67(1, self.has("exec3"), label="task %d: 67%s" % (t.tid, label))
        t.run67(2, self.has("infoIndirect"), opcode=abi.OP_INFO_INDIRECT, label="task %d: 69%s" % (t.tid, label))
        outcome(s, abi.OP_SYNCHRONIZE_AND_DISCARD_RESOURCES, u32s(t.tid, 0), self.has("syncAndDiscard"),
                "task %d: 62%s" % (t.tid, label))
        outcome(s, abi.OP_DISCARD_RESOURCES, u32s(t.tid, 0), self.has("discardResources"),
                "task %d: 63%s" % (t.tid, label))
        outcome(s, abi.OP_HEAP_TEXTURE_SIZE_AND_ALIGN, heap_query(t.tid, t.reply_va), self.has("metalHeaps"),
                "task %d: 64%s" % (t.tid, label))
        t.dump(t.reply_va, 0x10, "task %d: 64's reply%s" % (t.tid, label))
        t.objects_probe(3, label=label)
        n = len(t.ol.entries)
        place = t.area(PAGE)
        outcome(s, abi.OP_SET_OBJECT_AND_PLACEMENT_LIST,
                struct.pack("<IQQII", t.tid, 0, place, t.ol.length(), 16 * n), self.has("argumentBuffers"),
                "task %d: 68%s" % (t.tid, label))
        t.run55(9, label="task %d: 55 after 68%s" % (t.tid, label))
        t.dump_probe("after the task opcodes" + label)
        t.root_dump("after the task opcodes" + label)

    # ---- teardown ----

    def finish(self, release=True):
        """Tasks, backings, display pages, task 0, children, then the host
        releases the device."""
        s = self.s
        for tid in sorted(self.probes):
            self.probes[tid].delete(label="delete task %d" % tid)
        for t in self.tasks:
            t.delete(label="delete task %d" % t.tid)
        if self.t0 is not None:
            self.delete_backings()
        for port in sorted(self.pages):
            self.disconnect(port)
        if self.t0 is not None:
            self.t0.delete(label="delete task 0")
        for fid in sorted(self.children):
            self.children[fid].delete(label="delete child %d" % fid)
        s.submit(abi.OP_NOP, label="last NOP")
        s.events("torn down")
        s.dump(s.pa(s.root_page), 0x10, "root page: task words")
        s.dump(s.pa(s.root_page, abi.ROOT_DISPLAY_WORD), 0x8, "root page: display words")
        s.fault_state("torn down")
        if release:
            s.api("release")
            s.events("released")


def resume_display(x, cursor_ports=(0,)):
    """After rvg_device_did_resume the restored guest displays replay each
    port's cursor glyph and show (GuestDisplay::finish_restore)."""
    for _ in cursor_ports:
        x.expect("cursorGlyph", label="glyph replayed")
        x.expect("cursorShow", label="show replayed")
    x.s.events("resumed")


def work(x, t, phase, fid=0, byte=0x5A, label=None):
    """A SessionTask's blit, compute and render work (pgtest/session.py) in
    one exec on FIFO fid: ExecIndirect3 where the version has it, else
    ExecIndirect2 with the same resources and command buffers (the same
    work)."""
    segments = t.blit_segments(phase, byte) + t.compute_segments(phase) + t.render_segments(phase)
    cbs = [t.cmdbuf(sg) for sg in segments]
    res = t.resources()
    if x.has("exec3"):
        op, payload = abi.OP_EXEC_INDIRECT3, exec3_payload(t.tid, cbs, res)
    else:
        op, payload = abi.OP_EXEC_INDIRECT2, exec2_payload(t.tid, cbs, res)
    f = x.fifo(fid)
    st = f.cmd(op, payload)
    f.doorbell()
    f.wait_stamp(st, label=label or "task %d phase %d (opcode %d) on FIFO %d" % (t.tid, phase, op, fid))
    t.reset_cmdbufs()
    return st
