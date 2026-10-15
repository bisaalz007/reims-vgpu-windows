"""Decoder paths the per-decoder scenarios leave (docs/testing.md,
sections 6.14-6.18). Most of what the four decoders still miss is not
reachable from a guest on this host: the compute and render decoders'
"not supported" refusals ask the host's Metal device
(supportsDynamicAttributeStride, supportsImageBlocks,
supportsCommandBufferJump), which supports all of them here."""
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.exec_objects import ObjTask, info_ref
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import seg


@scenario
def decoder_gap_info_reply_type0():
    """An info command (461, buffer host info) whose reply buffer names an
    object-list entry of type 0 with a nonzero length: the reply cursor is
    refused before any task cursor is asked for ("Attempting to fetch
    invalid object"), the decoder throws ("Failed to get cursor for buffer")
    and the exec faults, with nothing written. In-process too: the refusal
    comes before the in-process task's writable cursor, which aborts, would
    be reached. A type 10 entry as the reply fails the same way at the
    switch's default ("Invalid object type 10 for buffer cursor")."""
    s = Scenario("decoder_gap_info_reply_type0", doc=decoder_gap_info_reply_type0.__doc__)
    s.bring_up()
    t = ObjTask(s)
    data, data_va = t.buffer(PAGE, contents=bytes(range(256)))
    zva = t.reserve(PAGE)
    t.fill(zva, PAGE, 0x5A)
    zero = t.ol.add_raw(0, 16, zva)
    ten = t.ol.add_raw(10, 16, zva)
    t.define()
    t.install()
    for label, reply in (("reply into a type 0 entry", zero), ("reply into a type 10 entry", ten)):
        t.exec_segments([seg(abi.ENC_INFO, info_ref(461, data, reply, 0))], resources=[(data, 0)], fault=True,
                        label=label)
        s.fault_state(label)
        s.resume_root()
    t.dump(zva, 0x20, "the entries' record bytes: unchanged")
    s.events("end")
    return s
