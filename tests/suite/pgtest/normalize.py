"""Normalization: what of a run's log the expected file locks.

Only timing (t_*) and diagnostics (d_*: signals, messages, pointers) are
dropped; nothing guest-visible is masked (docs/testing.md, section 5.3). Two
things are rewritten:
- the suspend blob (blob_pgsd) is parsed into the device-state and
  device-info pairs it carries (section 6.24), so that a change of state
  reads as one;
- the process's end becomes one "outcome" record: completed, died (the host
  process; "after" is the op of the last observation before it, since
  a death on another thread races with the op that caused it; the signal is
  a diagnostic), watchdog (a hang, at op i: the op that never returned),
  timeout (the orchestrator killed it) or error (a scenario bug).
"""
import json
import struct


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not (k.startswith("t_") or k.startswith("d_"))}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


# The blob's format (docs/testing.md, section 6.24; pgtest/suspend_guest.py
# builds it): the magic 'RVGS', version 1, then the registers section (tag 1)
# and the device-info section (tag 2).
BLOB_MAGIC = 0x53475652
BLOB_VERSION = 1
# The device-state key each register is reported under, in the registers
# section's order.
REGISTER_KEYS = tuple(range(1, 20)) + (21,)


def parse_sections(b):
    """The magic, the version and the sections {tag: payload} of a blob, in
    order; ValueError where it is not a well-formed blob of ours."""
    if len(b) < 8:
        raise ValueError("short")
    magic, version = struct.unpack_from("<II", b, 0)
    if magic != BLOB_MAGIC or version != BLOB_VERSION:
        raise ValueError("foreign")
    sections, off = {}, 8
    while off < len(b):
        if off + 8 > len(b):
            raise ValueError("section header")
        tag, length = struct.unpack_from("<II", b, off)
        if tag in sections or off + 8 + length > len(b):
            raise ValueError("section")
        sections[tag] = b[off + 8:off + 8 + length]
        off += 8 + length
    return magic, version, sections


def blob_of(record):
    """The blob a parsed record describes, sections in the order the device
    writes them."""
    regs = dict((k, int(v, 16)) for k, v in record["state"])
    registers = struct.pack("<%dI" % len(REGISTER_KEYS), *(regs[k] for k in REGISTER_KEYS))
    info = struct.pack("<i", record["info_max_key"]) + b"".join(
        struct.pack("<II", k, int(v, 16)) for k, v in record["info"])
    return (struct.pack("<II", int(record["magic"], 16), record["version"])
            + struct.pack("<II", 1, len(registers)) + registers
            + struct.pack("<II", 2, len(info)) + info)


def parse_pgsd(hexstr):
    """The suspend blob (section 6.24), as the device-state and device-info
    pairs it carries: the registers as {key, value} pairs under the keys
    that name them (1-19 and 21), the device info as its max key and its
    pairs. The magic, the version and the length are the blob's own. The
    record is exact: a blob it does not describe byte for byte (a malformed
    or foreign one, or its sections in another order) is reported whole, as
    hex."""
    b = bytes.fromhex(hexstr)
    out = {"len": len(b)}
    try:
        magic, version, sections = parse_sections(b)
        registers, info = sections[1], sections[2]
        if len(sections) != 2 or len(registers) != 4 * len(REGISTER_KEYS) or len(info) < 4 or len(info) % 8 != 4:
            raise ValueError("sections")
    except (ValueError, KeyError):
        out["hex"] = hexstr
        return out
    words = struct.unpack("<%dI" % len(REGISTER_KEYS), registers)
    (info_max,) = struct.unpack_from("<i", info, 0)
    out.update(magic=hex(magic), version=version, info_max_key=info_max,
               state=[[k, hex(v)] for k, v in zip(REGISTER_KEYS, words)],
               info=[[k, hex(v)] for k, v in struct.iter_unpack("<II", info[4:])])
    if blob_of(out) != b:
        return {"len": len(b), "hex": hexstr}
    return out


def normalize(log_path, rc, seconds):
    obs = []
    crash = watchdog = err = None
    ended = False
    try:
        lines = open(log_path).read().splitlines()
    except FileNotFoundError:
        lines = []
    for line in lines:
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except ValueError:
            obs.append({"ev": "unparsable", "line": line[:200]})
            continue
        ev = o.get("ev")
        if ev == "crash":
            crash = o
            continue
        if ev == "watchdog":
            watchdog = o
            continue
        if ev == "scenario_error":
            err = o
        if ev == "end":
            ended = True
        if "blob_pgsd" in o:
            o["blob_pgsd"] = parse_pgsd(o["blob_pgsd"])
        obs.append(strip(o))
    if err or rc in (64, 65, 66, 67):
        out = {"ev": "outcome", "kind": "error", "why": (err or {}).get("why", "exit %s" % rc)}
    elif crash:
        last = [o.get("i") for o in obs if isinstance(o.get("i"), int)]
        out = {"ev": "outcome", "kind": "died", "after": last[-1] if last else None}
    elif watchdog:
        out = {"ev": "outcome", "kind": "watchdog", "i": watchdog["i"]}
    elif rc is None:
        out = {"ev": "outcome", "kind": "timeout"}
    elif rc == 0 and ended:
        out = {"ev": "outcome", "kind": "completed"}
    elif rc is not None and rc < 0:
        # Killed by a signal the handler did not see (SIGKILL): the host died.
        last = [o.get("i") for o in obs if isinstance(o.get("i"), int)]
        out = {"ev": "outcome", "kind": "died", "after": last[-1] if last else None}
    else:
        out = {"ev": "outcome", "kind": "error", "why": "exit %s without an end" % rc}
    obs.append(out)
    return obs
