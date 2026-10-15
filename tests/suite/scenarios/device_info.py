"""GetDeviceInfo (58): the host's capability pairs written into a guest
page (docs/testing.md, section 6.9).

device_info::get_device_info maps the destination page and calls
device_info::write_device_info, which writes {u32 key, u32 value} pairs while
key < maxKey and count < maxCount, in a fixed key order, with pairs gated by
the binary version's features, then the {0, 0} terminator if it still fits.
Every value reports the host GPU and setup, so these scenarios are host-keyed.
"""
import struct

from pgtest import abi
from pgtest.registry import scenario
from pgtest.scenario import Scenario

# Pairs of the order writeDeviceInfo writes: at most 45, so 0x170 bytes
# covers them and the terminator.
SPAN = 0x180
SENTINEL = 0xA5


def query(s, page, max_key, max_count, label, fault=False, dest=None):
    """Fills the destination with a sentinel so that every byte written
    shows, sends 58 {maxKey, maxCount, destinationPage}, and dumps the page's
    head."""
    s.fill(s.pa(page), SPAN, SENTINEL)
    s.submit(abi.OP_GET_DEVICE_INFO, struct.pack("<III", max_key, max_count, s.pfn(page) if dest is None else dest),
             fault=fault, label=label)
    if fault:
        s.fault_state(label)
        s.resume_root()
        s.fifo.wait_stamp(s.fifo.stamp)
    s.dump(s.pa(page), SPAN, label)


@scenario
def device_info_bounds():
    """maxKey and maxCount at their edges: maxCount 0 and maxKey 0 write
    nothing (not even the terminator); maxKey 1 writes only the terminator;
    one pair with or without room for the terminator; maxKey 5 count 2 (two
    pairs, no terminator); the full list with room to spare, exactly the
    pair count, and one short; maxKey 0xffffffff and maxCount 0xffffffff."""
    s = Scenario("device_info_bounds", host=True, doc=device_info_bounds.__doc__)
    s.bring_up()
    page = s.alloc_page()
    for mk, mc in ((64, 0), (0, 64), (1, 64), (2, 1), (2, 2), (5, 2), (64, 64), (45, 1000),
                   (0xFFFFFFFF, 0xFFFFFFFF), (0xFFFFFFFF, 1)):
        query(s, page, mk, mc, "maxKey %d, maxCount %d" % (mk, mc))
    # The pair count is host-reported; the exact-fit cases take it from the
    # full list: each count from 30 to 46 either fits the list or not.
    for mc in range(30, 47):
        query(s, page, 64, mc, "maxKey 64, maxCount %d" % mc)
    s.events("end of queries")
    return s


@scenario
def device_info_keys():
    """Every maxKey from 0 to 46 with room for everything: each key
    boundary, in writeDeviceInfo's order (1-13, 14-16, 17, 18, 19-28, 33,
    36, 29-32, 34, 41, 35, 37, 42, 44, 40), shows where the list stops."""
    s = Scenario("device_info_keys", host=True, doc=device_info_keys.__doc__)
    s.bring_up()
    page = s.alloc_page()
    for mk in range(0, 47):
        query(s, page, mk, 64, "maxKey %d" % mk)
    return s


@scenario
def device_info_versions():
    """The full list at binary versions on each side of every feature
    writeDeviceInfo checks: dualPlaneTextures (key 12: 31, not 40), metalHeaps
    (14-16: 43), bufferFromIOSurface (17: 60), supportsSharedTextures (19:
    5-6 and from 100), supportsProgrammableSamplePositions (21: 105),
    supportsTileShadersImageBlocks (23-27: 107), supportsDynamicAttributeStride
    (28 bit 1: 111), supportsCommandBufferJump (36: 202); and before any
    version, at 0, and at the unsupported 1."""
    s = Scenario("device_info_versions", host=True, doc=device_info_versions.__doc__)
    s.bring_up(version=None)
    page = s.alloc_page()
    query(s, page, 64, 64, "no version written")
    for v in (0, 1, 2, 4, 5, 6, 7, 30, 31, 40, 41, 42, 43, 60, 99, 100, 104, 105, 106, 107, 110, 111, 201, 202, 245):
        s.w(abi.R_VERSION, v)
        s.r(abi.R_VERSION)
        query(s, page, 64, 64, "version %d" % v)
    return s


@scenario
def device_info_refusals():
    """The destination page must map: page 0, a page past guest RAM and
    0xffffffff make device_info::get_device_info raise (the page is not
    mapped), and the command faults; a short payload (8 bytes) faults before
    anything is written; a longer one (16 bytes) is accepted."""
    s = Scenario("device_info_refusals", host=True, doc=device_info_refusals.__doc__)
    s.bring_up()
    page = s.alloc_page()
    past = (s.ram[0][0] + s.ram[0][1]) >> 14
    for dest in (0, past, 0xFFFFFFFF):
        query(s, page, 64, 64, "destination page 0x%x" % dest, fault=True, dest=dest)
    s.fill(s.pa(page), SPAN, SENTINEL)
    s.submit(abi.OP_GET_DEVICE_INFO, struct.pack("<II", 64, 64), fault=True, label="8-byte payload")
    s.fault_state("8-byte payload")
    s.resume_root()
    s.fifo.wait_stamp(s.fifo.stamp)
    s.fill(s.pa(page), SPAN, SENTINEL)
    s.submit(abi.OP_GET_DEVICE_INFO, struct.pack("<IIII", 3, 64, s.pfn(page), 0xFFFFFFFF), label="16-byte payload")
    s.dump(s.pa(page), SPAN, "16-byte payload")
    s.events("end")
    return s


@scenario
def device_info_descriptor():
    """Config fields that reach the device info: argument buffers (key 33
    bit 0x20, key 42) need isolation and are on unless the config disables
    them, so disabling them turns them off under isolation and enabling
    them changes nothing without it; a memory map sets key 33's large-task
    bits, which the blocks setup lacks. "protected" is the default config:
    it once set Apple's enableProtectedContent, which was never read and
    which the C API does not have."""
    out = []
    for suffix, desc in (("argbuf_off", {"enableArgumentBuffers": False}),
                         ("argbuf_on", {"enableArgumentBuffers": True}),
                         ("protected", {})):
        name = "device_info_descriptor_" + suffix
        s = Scenario(name, host=True, descriptor=desc, doc=device_info_descriptor.__doc__)
        s.bring_up()
        page = s.alloc_page()
        query(s, page, 64, 64, "full list")
        out.append(s)
    return out
