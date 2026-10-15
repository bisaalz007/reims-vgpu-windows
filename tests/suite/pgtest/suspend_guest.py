"""The suspend blob, as a scenario crafts one for rvg_device_will_resume
(docs/testing.md, section 6.24).

The format (reims-vgpu-core's guest::abi::suspend_blob: SuspendState
serializes and parses it): u32 magic 'RVGS' (0x53475652), u32 version 1,
then sections to the end of the blob, each a u32 tag, a u32 payload length
and the payload, each section exactly once, in either order:
- tag 1, the registers: 20 u32s (80 bytes);
- tag 2, the device info: an i32 max key, then at most 45 pairs {u32 key,
  u32 value}, each key one a resume reads (1-42 or 44).

A scenario names the registers by the device-state keys the normalizer
reports them under (pgtest/normalize.py): key k is register k - 1 for keys
1-19, and key 21 is register 19. A register a scenario does not give is 0.
"""
import struct

MAGIC = 0x53475652
VERSION = 1
TAG_REGISTERS, TAG_DEVICE_INFO = 1, 2
MAX_INFO_PAIRS = 45

# Device-state keys, each naming a register.
K_ROOT, K_VERSION, K_FIFO_BASE, K_FIFO_ENABLED, K_FIFO_LENGTH, K_FIFO_WRITTEN, K_FIFO_READ, K_FIFO_START = range(1, 9)
K_IRQ_DISPLAY, K_IRQ_STAMP, K_FAULT = 9, 10, 11
K_EFI_ENABLED, K_EFI_START, K_EFI_LENGTH, K_EFI_DEPTH, K_EFI_MODE, K_EFI_STRIDE = range(12, 18)
K_PORT_COUNT, K_FAULT_OFFSET, K_CONNECTION_TYPES = 18, 19, 21

# The key of each register, in the section's order.
REGISTER_KEYS = tuple(range(1, 20)) + (K_CONNECTION_TYPES,)


def pairs(kv):
    return b"".join(struct.pack("<II", k, v & 0xFFFFFFFF) for k, v in kv)


def state(version=0xF5, ports=1, **over):
    """A minimal device state: the binary version and the port count; the
    other registers only where given (name=value with the K_ names'
    suffix)."""
    kv = [(K_VERSION, version), (K_PORT_COUNT, ports)]
    for k, v in over.items():
        kv.append((globals()["K_" + k.upper()], v))
    return kv


def registers(state_kv):
    """The registers section's payload: each key's value in its register, a
    later key winning; the rest 0."""
    words = [0] * len(REGISTER_KEYS)
    for k, v in state_kv:
        words[REGISTER_KEYS.index(k)] = v & 0xFFFFFFFF
    return struct.pack("<%dI" % len(words), *words)


def device_info(info_kv=(), info_max=45):
    """The device-info section's payload: the max key, then the pairs."""
    return struct.pack("<i", info_max) + pairs(info_kv)


def header(magic=MAGIC, version=VERSION):
    return struct.pack("<II", magic, version)


def section(tag, payload, length=None):
    """A section: its tag, its length (the payload's unless given), the
    payload."""
    return struct.pack("<II", tag, len(payload) if length is None else length) + payload


def blob(state_kv, info_kv=(), info_max=45):
    """A well-formed blob: the header, the registers, the device info."""
    return (header() + section(TAG_REGISTERS, registers(state_kv))
            + section(TAG_DEVICE_INFO, device_info(info_kv, info_max)))
