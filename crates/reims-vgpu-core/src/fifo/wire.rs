//! What the guest writes into a FIFO's ring: the command header ([`FifoCommand`]), the barrier
//! records ([`StampWait`]), a command's payload ([`Payload`]) and the record arrays a command
//! hands on to a task or a guest display ([`Records`]).
//!
//! Every value here is guest data: it is read little-endian from a private copy the thread made
//! out of the ring, never through a reference into guest memory, and its arithmetic is spelt out.

#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use std::marker::PhantomData;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::num::{ToU64, ToUsize};

/// The command header every command starts with (12 bytes).
// name is ours: the image encodes the struct anonymously (`^{?=SSII}` in
// `v44@0:8^{?=SSII}16I24^v28Q36`); the field names follow what the thread does with each.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct FifoCommand {
    /// The opcode.
    pub command_id: u16,
    /// How many [`StampWait`] records follow the header.
    pub barrier_count: u16,
    /// The whole command's length, header included.
    pub length: u32,
    /// The stamp value the command completes.
    pub stamp_value: u32,
}

const _: () = assert!(size_of::<FifoCommand>() == 12);

/// One barrier: a stamp slot and the value it must reach before the command runs (8 bytes).
// name is ours: the method encodings (`-[_PGDevice waitStamp:waitingFIFO:]`,
// `-[PGFIFO barrierWait:barrier:]`) give an anonymous `{unsigned int; unsigned int}`; the roles
// come from the wait's log text ("waiting on %d:0x%x").
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct StampWait {
    /// The stamp slot.
    pub index: u32,
    /// The value it must reach.
    pub value: u32,
}

const _: () = assert!(size_of::<StampWait>() == 8);

/// The header's size.
pub const COMMAND_HEADER_SIZE: usize = 12;
/// The header's size, as the length checks compare it.
pub(crate) const HEADER_LEN: u32 = 12;
/// A barrier record's size.
pub(crate) const STAMP_WAIT_LEN: u64 = 8;
/// The most barriers a command may carry.
pub const MAX_BARRIERS: u16 = 64;
/// The barrier buffer: [`MAX_BARRIERS`] records.
pub(crate) const BARRIER_BYTES_MAX: usize = 64 * 8;

/// A command's payload: the bytes after its header and barriers, in the thread's buffer. Each
/// field is read at its offset after the size check that covers it (the records are
/// `#pragma pack(4)`, little-endian); a read past the end, which a size check always precludes,
/// reads zeros.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Payload<'a>(&'a [u8]);

impl<'a> Payload<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Payload<'a> {
        Payload(bytes)
    }

    /// The payload's size.
    pub(crate) fn len(self) -> usize {
        self.0.len()
    }

    /// The bytes.
    pub(crate) fn bytes(self) -> &'a [u8] {
        self.0
    }

    /// The payload from `off` on (a record list, the rest after a field).
    pub(crate) fn tail(self, off: usize) -> Payload<'a> {
        Payload(self.0.get(off..).unwrap_or(&[]))
    }

    /// `N` bytes at `off`.
    fn at<const N: usize>(self, off: usize) -> [u8; N] {
        let mut b = [0u8; N];
        if let Some(end) = off.checked_add(N)
            && let Some(s) = self.0.get(off..end)
        {
            b.copy_from_slice(s);
        }
        b
    }

    /// The byte at `off`.
    pub(crate) fn u8(self, off: usize) -> u8 {
        self.0.get(off).copied().unwrap_or(0)
    }

    /// The `u16` at `off`.
    pub(crate) fn u16(self, off: usize) -> u16 {
        u16::from_le_bytes(self.at(off))
    }

    /// The `u32` at `off`.
    pub(crate) fn u32(self, off: usize) -> u32 {
        u32::from_le_bytes(self.at(off))
    }

    /// The `u64` at `off` (4-byte aligned in the packed records).
    pub(crate) fn u64(self, off: usize) -> u64 {
        u64::from_le_bytes(self.at(off))
    }

    /// `count` records of `T` from `off`: what a size check has already covered.
    pub(crate) fn records<T: FromBytes + IntoBytes + Immutable + KnownLayout>(
        self,
        off: usize,
        count: u32,
    ) -> Records<'a, T> {
        Records::new(self.tail(off).0, count).unwrap_or_else(Records::empty)
    }
}

/// `count` records of `T`, read unaligned out of a command's payload as they cross into a task or
/// a guest display.
///
/// The payload has no alignment promise, so no `&[T]` is ever formed over it: `Records` keeps the
/// bytes and the count, and reads each record when the callee reads it. The bytes are the FIFO
/// thread's buffer, valid for the call only, which the borrow expresses; a callee that keeps
/// records past the call copies them ([`Records::to_vec`]).
#[derive(Debug, Clone, Copy)]
pub struct Records<'a, T> {
    bytes: &'a [u8],
    count: u32,
    _t: PhantomData<fn() -> T>,
}

impl<T> Records<'_, T> {
    /// How many records.
    pub fn count(&self) -> u32 {
        self.count
    }
}

impl<'a, T: FromBytes + IntoBytes + Immutable + KnownLayout> Records<'a, T> {
    /// The records at the start of `bytes`.
    ///
    /// `None` when `bytes` is shorter than `count` records: the FIFO checks each payload's size
    /// against its count before it calls a task or a guest display, so a caller that gets `None`
    /// has skipped that check, and must not call on.
    pub fn new(bytes: &'a [u8], count: u32) -> Option<Records<'a, T>> {
        let need = u64::from(count).wrapping_mul(Self::stride());
        if need > bytes.len().to_u64() {
            return None;
        }
        Some(Records {
            bytes,
            count,
            _t: PhantomData,
        })
    }

    /// No records (a count of 0 with an empty payload is legal: the callee is still called).
    pub fn empty() -> Records<'a, T> {
        Records {
            bytes: &[],
            count: 0,
            _t: PhantomData,
        }
    }

    /// One record's size.
    pub fn stride() -> u64 {
        size_of::<T>().to_u64()
    }

    /// The record at `index`, or `None` for `index >= count` (the callee's own index checks
    /// decide what an index out of its range does).
    pub fn get(&self, index: u32) -> Option<T> {
        if index >= self.count {
            return None;
        }
        let at = u64::from(index).wrapping_mul(Self::stride()).to_usize();
        let tail = self.bytes.get(at..)?;
        T::read_from_prefix(tail).ok().map(|(v, _)| v)
    }

    /// Every record, in index order.
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        (0..self.count).filter_map(move |i| self.get(i))
    }

    /// A copy of every record (for a callee that keeps them past the call).
    pub fn to_vec(&self) -> Vec<T> {
        self.iter().collect()
    }

    /// The bytes the records occupy: `count * size_of::<T>()` from the start (what the remote
    /// task sends on the wire).
    pub fn as_bytes(&self) -> &'a [u8] {
        let n = u64::from(self.count)
            .wrapping_mul(Self::stride())
            .to_usize();
        self.bytes.get(..n).unwrap_or(self.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guest::records::{CmdExecIndirectCmdBuffer, CmdExecIndirectResource3};

    /// The header: 12 bytes, little-endian, the opcode first
    #[test]
    fn fifo_command_layout() {
        let bytes = [
            0x3d, 0x00, 0x02, 0x00, 0x20, 0, 0, 0, 0x78, 0x56, 0x34, 0x12,
        ];
        let c = FifoCommand::read_from_bytes(&bytes).unwrap();
        assert_eq!(
            c,
            FifoCommand {
                command_id: 0x3d,
                barrier_count: 2,
                length: 0x20,
                stamp_value: 0x1234_5678,
            }
        );
        assert_eq!(c.as_bytes(), &bytes);
        assert_eq!(core::mem::offset_of!(FifoCommand, barrier_count), 2);
        assert_eq!(core::mem::offset_of!(FifoCommand, length), 4);
        assert_eq!(core::mem::offset_of!(FifoCommand, stamp_value), 8);
        assert_eq!(align_of::<FifoCommand>(), 4);
    }

    /// A barrier: 8 bytes, the slot then the value
    #[test]
    fn stamp_wait_layout() {
        let bytes = [3, 0, 0, 0, 0xff, 0xff, 0xff, 0x7f];
        let w = StampWait::read_from_bytes(&bytes).unwrap();
        assert_eq!(
            w,
            StampWait {
                index: 3,
                value: 0x7fff_ffff
            }
        );
        assert_eq!(w.as_bytes(), &bytes);
        assert_eq!(core::mem::offset_of!(StampWait, value), 4);
        assert_eq!(align_of::<StampWait>(), 4);
    }

    /// Fields read little-endian at their offsets; past the end reads zero
    #[test]
    fn payload_reads() {
        let p = Payload::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(p.u8(0), 1);
        assert_eq!(p.u16(1), 0x0302);
        assert_eq!(p.u32(4), 0x0807_0605);
        assert_eq!(p.u64(1), 0x0908_0706_0504_0302);
        assert_eq!(p.u32(7), 0);
        assert_eq!(p.u64(usize::MAX), 0);
        assert_eq!(p.tail(8).bytes(), &[9]);
        assert!(p.tail(10).bytes().is_empty());
    }

    /// Records::new refuses a payload shorter than its count
    #[test]
    fn new_checks_length() {
        let bytes = [0u8; 15];
        assert!(Records::<CmdExecIndirectCmdBuffer>::new(&bytes, 1).is_none());
        assert!(Records::<CmdExecIndirectCmdBuffer>::new(&bytes, 0).is_some());
        let bytes = [0u8; 16];
        assert!(Records::<CmdExecIndirectCmdBuffer>::new(&bytes, 1).is_some());
        // A huge count is refused, not wrapped.
        assert!(Records::<CmdExecIndirectCmdBuffer>::new(&bytes, u32::MAX).is_none());
    }

    /// Records::get reads unaligned records in order and stops at count
    #[test]
    fn get_reads_unaligned() {
        // One byte of misalignment in front of two 8-byte records.
        let mut buf = vec![0xAAu8];
        buf.extend_from_slice(&7u32.to_le_bytes());
        buf.extend_from_slice(&5u32.to_le_bytes());
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.push(0xBB); // trailing payload beyond the count
        let r = Records::<CmdExecIndirectResource3>::new(&buf[1..], 2).unwrap();
        assert_eq!(r.count(), 2);
        let a = r.get(0).unwrap();
        assert_eq!((a.reference, a.flags), (7, 5));
        let b = r.get(1).unwrap();
        assert_eq!((b.reference, b.flags), (9, 2));
        assert!(r.get(2).is_none());
        assert_eq!(r.to_vec().len(), 2);
        assert_eq!(
            r.iter().map(|x| x.reference).collect::<Vec<_>>(),
            vec![7, 9]
        );
        assert_eq!(r.as_bytes().len(), 16);
        assert_eq!(Records::<CmdExecIndirectResource3>::stride(), 8);
    }

    /// Records::empty has no records
    #[test]
    fn empty_has_none() {
        let r = Records::<u32>::empty();
        assert_eq!(r.count(), 0);
        assert!(r.get(0).is_none());
        assert!(r.as_bytes().is_empty());
    }
}
