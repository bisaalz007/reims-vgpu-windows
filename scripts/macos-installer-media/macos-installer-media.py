#!/usr/bin/env python3
"""Make a macOS installer image bootable under OVMF *and* mountable by XNU.

A downloaded "macOS ISO" is usually not ISO9660. The ones measured here are raw
disks carrying an Apple Partition Map with a single Apple_HFS partition, and
that shape is caught between the two ends of the boot:

  - EDK2's PartitionDxe reads MBR, GPT, El Torito and UDF, but not APM. Real Mac
    firmware reads APM; OVMF does not. So the partition never becomes a UEFI
    block device, the HFS+ driver never binds, and OpenCore's picker lists only
    its own ESP -- the installer is simply absent, with no error anywhere.

  - Handing the guest the bare partition instead (a `raw` view at the
    partition's offset) fixes that and breaks the other end. XNU roots by
    boot-uuid ("rooting via boot-uuid from /chosen"), and AppleFileSystemDriver
    publishes that UUID from a *partition* IOMedia. A disk with no partition map
    has nothing to match, so the kernel sits at "Still waiting for root device"
    with every disk detected and powered up.

A GPT satisfies both, because both ends read GPT. This writes one describing the
extent the APM already declared -- same start, same length -- into a qcow2
overlay, so no data is copied and the source image is never modified.

The output is a qcow2 whose backing file is the source image. Deleting it resets
any writes the installer made; the source is read only.
"""
import argparse
import os
import struct
import subprocess
import sys
import zlib

SECTOR = 512
# GPT reserves 34 sectors at each end: 1 protective MBR + 1 header + 32 entry
# sectors (128 entries of 128 bytes).
GPT_SECTORS = 34
# Room the overlay adds beyond the source so the backup GPT has somewhere to go.
# The source usually ends exactly where its last partition does.
PAD_SECTORS = 2048

# Apple HFS/HFS+ partition type GUID, 48465300-0000-11AA-AA11-00306543ECAC, in
# the mixed-endian form GPT stores.
TYPE_HFS = struct.pack("<IHH", 0x48465300, 0x0000, 0x11AA) + bytes.fromhex(
    "AA1100306543ECAC"
)


def die(msg):
    print(f"macos-installer-media: {msg}", file=sys.stderr)
    sys.exit(1)


def read_apm(path):
    """Return (start_lba, sectors, name) of the first Apple_HFS partition.

    The map is the source's own statement about itself, so the extent is read
    rather than guessed. A driver descriptor record signature of 'ER' at offset
    0 and 'PM' entries from offset 0x400 is the whole format we need.
    """
    with open(path, "rb") as f:
        if f.read(2) != b"ER":
            return None
        f.seek(0x400)
        for _ in range(64):
            entry = f.read(SECTOR)
            if len(entry) < SECTOR or entry[:2] != b"PM":
                break
            start, blocks = struct.unpack(">II", entry[8:16])
            name = entry[16:48].split(b"\0")[0].decode("ascii", "replace")
            ptype = entry[48:80].split(b"\0")[0].decode("ascii", "replace")
            if ptype == "Apple_HFS" and blocks:
                return start, blocks, name or "Install macOS"
    return None


def build_gpt(start_lba, part_sectors, total_sectors, label):
    """Return (head_bytes, tail_bytes, tail_offset) for a one-partition GPT."""
    last = total_sectors - 1

    # Fixed GUIDs. They only have to be stable across runs so the guest sees the
    # same disk every boot; a random one would make the volume look new each time.
    part_guid = struct.pack("<IHH", 0x52454D53, 0x0001, 0x4000) + bytes.fromhex(
        "9000000000000001"
    )
    disk_guid = struct.pack("<IHH", 0x52454D53, 0x0002, 0x4000) + bytes.fromhex(
        "9000000000000002"
    )

    entries = bytearray(128 * 128)
    entries[0:16] = TYPE_HFS
    entries[16:32] = part_guid
    entries[32:40] = struct.pack("<Q", start_lba)
    entries[40:48] = struct.pack("<Q", start_lba + part_sectors - 1)
    entries[48:56] = struct.pack("<Q", 0)
    entries[56:128] = label.encode("utf-16-le")[:72].ljust(72, b"\0")
    entries = bytes(entries)
    entries_crc = zlib.crc32(entries) & 0xFFFFFFFF

    def header(my_lba, alt_lba, entries_lba):
        h = bytearray(92)
        h[0:8] = b"EFI PART"
        h[8:12] = struct.pack("<I", 0x00010000)
        h[12:16] = struct.pack("<I", 92)
        h[24:32] = struct.pack("<Q", my_lba)
        h[32:40] = struct.pack("<Q", alt_lba)
        h[40:48] = struct.pack("<Q", GPT_SECTORS)
        h[48:56] = struct.pack("<Q", last - GPT_SECTORS + 1)
        h[56:72] = disk_guid
        h[72:80] = struct.pack("<Q", entries_lba)
        h[80:84] = struct.pack("<I", 128)
        h[84:88] = struct.pack("<I", 128)
        h[88:92] = struct.pack("<I", entries_crc)
        h[16:20] = struct.pack("<I", zlib.crc32(bytes(h)) & 0xFFFFFFFF)
        return bytes(h).ljust(SECTOR, b"\0")

    # Protective MBR, so anything that reads only an MBR sees the disk as
    # claimed rather than as unpartitioned and offers to initialise it.
    mbr = bytearray(SECTOR)
    mbr[446 + 4] = 0xEE
    mbr[446 + 8 : 446 + 12] = struct.pack("<I", 1)
    mbr[446 + 12 : 446 + 16] = struct.pack("<I", min(last, 0xFFFFFFFF))
    mbr[510:512] = b"\x55\xaa"

    head = bytes(mbr) + header(1, last, 2) + entries
    head = head.ljust(GPT_SECTORS * SECTOR, b"\0")

    # The backup is 33 sectors, not 34: it has no protective MBR. The entry array
    # sits at last-32 and the header goes at the very last LBA, because that is
    # where a reader looks for it when the primary is unreadable. Placing the
    # whole 33-sector block one sector earlier leaves the last LBA blank, and the
    # primary header's alternate-LBA pointer then names a sector with no header
    # in it.
    tail_lba = last - 32
    tail = entries + header(last, 1, tail_lba)
    assert len(tail) == 33 * SECTOR, len(tail)
    return head, tail, tail_lba * SECTOR


def main():
    ap = argparse.ArgumentParser(
        description="Wrap a macOS installer image in a GPT, without copying it."
    )
    ap.add_argument("source", help="installer image (the downloaded .iso/.img)")
    ap.add_argument("output", help="qcow2 overlay to create")
    ap.add_argument(
        "--qemu-img", default="qemu-img", help="qemu-img binary (default: on PATH)"
    )
    ap.add_argument(
        "--qemu-io", default="qemu-io", help="qemu-io binary (default: on PATH)"
    )
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = ap.parse_args()

    if not os.path.isfile(args.source):
        die(f"no such image: {args.source}")
    if os.path.exists(args.output):
        if not args.force:
            die(f"{args.output} exists (pass --force to rebuild it)")
        os.remove(args.output)

    size = os.path.getsize(args.source)
    if size % SECTOR:
        die(f"image size {size} is not a whole number of {SECTOR}-byte sectors")

    apm = read_apm(args.source)
    if apm is None:
        die(
            f"{args.source} has no Apple Partition Map with an Apple_HFS partition.\n"
            "                     If it is already GPT, attach it directly -- OVMF reads GPT.\n"
            "                     If it is ISO9660, attach it as CD media instead."
        )
    start_lba, part_sectors, label = apm

    src_sectors = size // SECTOR
    if start_lba + part_sectors > src_sectors:
        die(
            f"the partition map claims {start_lba + part_sectors} sectors but the "
            f"file holds {src_sectors}; the image is truncated"
        )

    total = src_sectors + PAD_SECTORS
    head, tail, tail_off = build_gpt(start_lba, part_sectors, total, label)

    # The overlay is larger than its backing so the backup GPT fits. Reads past
    # the backing return zeros, which is what those sectors are anyway.
    subprocess.run(
        [args.qemu_img, "create", "-f", "qcow2", "-F", "raw", "-b", os.path.abspath(args.source),
         args.output, str(total * SECTOR)],
        check=True, stdout=subprocess.DEVNULL,
    )

    head_tmp = args.output + ".gpthead"
    tail_tmp = args.output + ".gpttail"
    try:
        with open(head_tmp, "wb") as f:
            f.write(head)
        with open(tail_tmp, "wb") as f:
            f.write(tail)
        for src, off, length in ((head_tmp, 0, len(head)), (tail_tmp, tail_off, len(tail))):
            subprocess.run(
                [args.qemu_io, "-c", f"write -s {src} {off} {length}", args.output],
                check=True, stdout=subprocess.DEVNULL,
            )
    finally:
        for tmp in (head_tmp, tail_tmp):
            if os.path.exists(tmp):
                os.remove(tmp)

    print(
        f"macos-installer-media: {args.output}\n"
        f"  source     {args.source} ({src_sectors} sectors, untouched)\n"
        f"  partition  '{label}' lba {start_lba}..{start_lba + part_sectors - 1}\n"
        f"  gpt        primary @lba 1, backup @lba {tail_off // SECTOR}, disk {total} sectors"
    )


if __name__ == "__main__":
    main()
