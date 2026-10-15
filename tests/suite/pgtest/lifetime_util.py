"""Builders for the resource-lifetime interplay scenarios
(scenarios/interplay_lifetime.py): objects that live and die while work
that names them is queued, running or held on another FIFO, objects made
again under the same reference, and resources paged across many execs and
across the host's lifecycle calls.

The guest ABI is restated from byte offsets, as abi.py, task.py and the
builders beside it do: the object list entry (reims-vgpu-core's
guest::records::ObjectEntry), the exec payloads (55, 67), barriers, and the
buffer and texture records. Nothing here is taken from the library's
code.

What the scenarios lean on (read from the code, and locked by them):
- An object is made the first time a reference to it is resolved, from the
  object-list entry and the record as they are then, and cached until a
  delete (37, 40) or a discard of a kind that does not support discard
  (63 of an in-process heap buffer) forgets it. A forgotten object's
  record is read again at its next use; a live object's record and entry
  are never read again.
- The object list is read in place in every setup (tasks::object_lists),
  so a guest that rewrites an entry changes what the next creation reads
  without a new 51.
- A heap buffer and a texture placed in a heap hold the heap's Metal
  object, not the library's resource around it, and a view holds its
  base's.
"""
import struct

from . import abi
from .abi import PAGE
from .compute_objects import Blit as B, Compute as C
from .diff_util import BLIT, COMPUTE, Lab, exec_payload, pattern
from .fifo_util import child
from .task import seg
from .resources_build import (OPT_PRIVATE, RGBA8Unorm, U_READ, U_WRITE, desc1, ser_view_format, simple_dim,
                              texture_rec)

E3 = abi.OP_EXEC_INDIRECT3
IH, IG, SY = abi.RES_INVALIDATE_HOST, abi.RES_INVALIDATE_GUEST, abi.RES_SYNCHRONIZE
U_PFV = 0x10  # MTLTextureUsagePixelFormatView


# ---- outcomes that differ by setup ----

def either(s, opcode, payload, label, fifo=None):
    """One command on the root FIFO whose outcome the setups decide: waits
    for its stamp or a fault (the observation says which), reads the fault
    state, resumes (a no-op when nothing faulted: the stamp queue's
    resume_from_fault only logs), and waits for the stamp either path
    publishes."""
    f = fifo or s.fifo
    assert f.fid == 0, "either() is for the root FIFO"
    st = f.cmd(opcode, payload)
    f.doorbell()
    s.wait_any([dict(gpa=f.stamp_gpa(), ge=st), dict(mmio=abi.R_FAULT_MASK, mmio_eq=1)],
               label="%s: completed (0) or faulted (1)" % label)
    s.fault_state(label)
    s.resume_root()
    f.wait_stamp(st, label="%s: stamp" % label)
    return st


def run_either(t, segments, resources, label, opcode=E3):
    """An exec of t whose outcome the setups decide (either())."""
    cbs = t.place(segments)
    return either(t.s, opcode, exec_payload(t.tid, cbs, resources, opcode), label)


# ---- the object list, rewritten in place ----

def entry_gpa(t, ref):
    """Where entry `ref` of t's object list is in guest memory."""
    return t.gpa(t.ol.list_va + abi.OBJ_ENTRY.size * ref)


def rewrite_entry(t, ref, typ, record=None, va=None, length=None):
    """Rewrites entry `ref` of t's installed object list in guest memory:
    its type and a record. record: new bytes, placed after the records so
    far; or va and length: whatever is there. No command follows: the list
    is read in place."""
    if record is not None:
        record = bytes(record)
        va = t.ol._rec
        t.write(va, record)
        t.ol._rec = (va + len(record) + 15) // 16 * 16
        length = len(record)
    t.ol.entries[ref] = (typ, length, va)
    t.write(t.ol.list_va + abi.OBJ_ENTRY.size * ref, abi.OBJ_ENTRY.pack(typ | (length << 8), va))


# ---- objects ----

def tex2d(t, w=4, h=4, contents=None, usage=U_READ | U_WRITE, options=OPT_PRIVATE, fmt=RGBA8Unorm):
    """A normal texture (type 2) over a page of its own: {u64 length, u32
    page}, one dimension block, commandType 1 and a v1 descriptor.
    Returns (ref, va)."""
    va = t.reserve(PAGE)
    if contents is not None:
        t.write(va, contents)
    return t.add(abi.T_TEXTURE, texture_rec(PAGE, va // PAGE, [simple_dim(w, h)],
                                            desc1(fmt=fmt, w=w, h=h, usage=usage, options=options))), va


def view(t, base, fmt=RGBA8Unorm):
    """A pixel-format view of base (type 8, commandType 7)."""
    return t.add(abi.T_SERIALIZER_TEXTURE, ser_view_format(base, fmt))


def texels(w, h, seed=1):
    return bytes(((x * 16 + y * 3 + seed) & 0xFF) if c < 3 else 0xFF
                 for y in range(h) for x in range(w) for c in range(4))


# ---- records ----

def blit(*records):
    """One blit segment of the records."""
    return [seg(BLIT, b"".join(records))]


def tex_out(tex, buf, off, w=4, h=4):
    """Blit 302: a w x h RGBA8 texture copied out to buf at off."""
    return B.copy_texture_to_buffer(tex, buf, (0, 0, 0), (w, h, 1), off, 4 * w, 4 * w * h)


def tex_in(buf, tex, off, w=4, h=4):
    """Blit 300: buf at off copied into a w x h RGBA8 texture."""
    return B.copy_buffer_to_texture(buf, tex, off, 4 * w, 4 * w * h, (w, h, 1), (0, 0, 0))


def tex_fill(tex, colour, w=4, h=4):
    """Blit 321 FillTextureWithColor over the whole texture
    (resources_build.b_fill_tex_color)."""
    from .resources_build import b_fill_tex_color
    return b_fill_tex_color(tex, (w, h, 1), colour, RGBA8Unorm)


# ---- a guest with a gate ----

class Life:
    """The guest: bring-up, a gate (child FIFO 1: a barrier on its next
    stamp holds a root FIFO command for as long as the scenario likes, and
    a NOP on it releases the command), optionally a worker (child FIFO 2),
    and task `tid` as a Lab (diff_util: object list at VA 0, records at
    pages 1-3, command buffers at pages 4-7, data from page 8), unless
    task=False (the scenario brings its own, after the bring-up)."""

    def __init__(self, s, pages=96, worker=True, tid=1, task=True):
        self.s = s
        self.f = s.bring_up()
        self.gate = child(s, 1)
        self.worker = child(s, 2) if worker else None
        self.gate_value = 0
        self.t = Lab(s, tid=tid, pages=pages) if task else None

    def barrier(self, need=1):
        return [(self.gate.stamp_index, self.gate_value + need)]

    def hold(self, segments, resources, label, t=None, opcode=E3, ranges=None, need=1):
        """An exec of t on the root FIFO behind the gate (its barrier wants
        the gate's stamp `need` NOPs on); waits 300 ms to see that its stamp
        does not come. Returns the stamp."""
        t = t or self.t
        cbs = ranges if ranges is not None else t.place(segments)
        st = self.f.cmd(opcode, exec_payload(t.tid, cbs, resources, opcode), barriers=self.barrier(need))
        self.f.doorbell()
        self.f.wait_stamp(st, timeout_ms=300, label="%s: held by the gate" % label)
        return st

    def open_gate(self):
        self.gate_value += 1
        return self.gate.submit(abi.OP_NOP, label="the gate opens")

    def release(self, st, label):
        """Opens the gate and waits for the held exec's stamp or a fault;
        the fault state, a resume, and the stamp."""
        s = self.s
        self.open_gate()
        s.wait_any([dict(gpa=self.f.stamp_gpa(), ge=st), dict(mmio=abi.R_FAULT_MASK, mmio_eq=1)],
                   label="%s: completed (0) or faulted (1)" % label)
        s.fault_state(label)
        s.resume_root()
        self.f.wait_stamp(st, label="%s: stamp" % label)

    def release_last(self, need, label):
        """For an exec held with hold(need=need) whose release may kill the
        host: rings `need` NOPs on the gate in one doorbell and waits for a
        fault. The NOPs keep the gate busy well past the doorbell's MMIO
        write, so a death lands in the wait every run (as submit_last does,
        exec_objects.py). If the host lives, the fault state follows."""
        for _ in range(need):
            self.gate.cmd(abi.OP_NOP)
        self.gate_value += need
        self.gate.doorbell()
        self.s.wait_fault(label=label)
        self.s.fault_state(label)

    def work(self, opcode, payload, label):
        """A command on the worker FIFO, waited for."""
        return self.worker.submit(opcode, payload, label=label)


def delete_resource(tid, ref):
    """DeleteResource (37): {u32 taskID, u32 reference}."""
    return abi.OP_DELETE_RESOURCE, struct.pack("<II", tid, ref)


def delete_object(tid, operation, ref):
    """DeleteObject (40): {u32 taskID, u32 operationID, u32 length, u32
    reference}; the task gets the payload from operationID on."""
    return abi.OP_DELETE_OBJECT, struct.pack("<IIII", tid, operation, 12, ref)


def refs_payload(tid, refs):
    """{u32 taskID, u32 count} + count x u32 reference: 53, 62, 63."""
    return struct.pack("<II", tid, len(refs)) + b"".join(struct.pack("<I", r) for r in refs)


def invalidate_payload(tid, items):
    """52: {u32 taskID, u32 count}, count x {u32 reference, u8
    clearHostValid, setHostValid, clearGuestValid, setGuestValid}
    (tasks::ResourceInvalidation)."""
    return struct.pack("<II", tid, len(items)) + b"".join(struct.pack("<IBBBB", r, *f) for r, f in items)


__all__ = ["B", "C", "BLIT", "COMPUTE", "Lab", "Life", "pattern", "either", "run_either", "entry_gpa",
           "rewrite_entry", "tex2d", "view", "texels", "tex_out", "tex_in", "tex_fill", "delete_resource",
           "delete_object", "blit", "refs_payload", "invalidate_payload", "E3", "IH", "IG", "SY", "U_PFV"]
