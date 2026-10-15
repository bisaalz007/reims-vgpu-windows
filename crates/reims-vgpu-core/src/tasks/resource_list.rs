//! An exec's resource records, as one list whatever their width.

use crate::fifo::wire::Records;
use crate::guest::records::{CmdExecIndirectResource, CmdExecIndirectResource3};
use crate::num::Truncate;
use crate::raise::RaiseResult;
use crate::tasks::{DeathKind, RaiseKind, raise_caught};

/// An exec's resource records, 24-byte (opcode 55) or 8-byte (67, 69), each read as the 8-byte
/// form.
#[derive(Debug, Clone, Copy)]
pub enum ResourceList<'a> {
    /// The 24-byte records.
    V2(Records<'a, CmdExecIndirectResource>),
    /// The 8-byte records.
    V3(Records<'a, CmdExecIndirectResource3>),
}

/// The low bit of a flag byte: what storing a byte into a 1-bit bit-field keeps.
fn low_bit(b: u8) -> u32 {
    u32::from(b) & 1
}

/// The 8-byte form of a 24-byte record.
fn entry_from_v2(resource: &CmdExecIndirectResource) -> CmdExecIndirectResource3 {
    // Each flag byte is cut to its low bit (2 is clear, 3 is set), not `!= 0`.
    let flags = low_bit(resource.invalidate_host)
        | low_bit(resource.invalidate_guest).wrapping_shl(1)
        | low_bit(resource.synchronize).wrapping_shl(2);
    CmdExecIndirectResource3 {
        reference: resource.reference,
        flags,
    }
}

/// An 8-byte record with its flags cut to bits 0-2.
fn entry_from_v3(resource: &CmdExecIndirectResource3) -> CmdExecIndirectResource3 {
    CmdExecIndirectResource3 {
        reference: resource.reference,
        flags: resource.flags & 0b111,
    }
}

impl ResourceList<'_> {
    /// The record count.
    pub fn count(&self) -> u64 {
        match self {
            ResourceList::V2(r) => u64::from(r.count()),
            ResourceList::V3(r) => u64::from(r.count()),
        }
    }

    /// The record at `index`, read from the payload now. An index at or past the count raises
    /// (no caller reaches it: each loops below the count).
    pub fn get(&self, index: u64) -> RaiseResult<CmdExecIndirectResource3> {
        if index >= self.count() {
            return raise_caught(
                RaiseKind::ResourceIndex,
                format_args!(
                    "resource record {index} is past the list's {} records",
                    self.count()
                ),
            );
        }
        // `index < count <= u32::MAX`, so the cut keeps it.
        let at = index.low_u32();
        let entry = match self {
            ResourceList::V2(r) => r.get(at).map(|e| entry_from_v2(&e)),
            ResourceList::V3(r) => r.get(at).map(|e| entry_from_v3(&e)),
        };
        // The FIFO checked that the payload holds `count` records, so every index below it
        // reads.
        match entry {
            Some(e) => Ok(e),
            None => crate::fatal!(DeathKind::ResourceRecord),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::IntoBytes;

    fn v2(reference: u32, h: u8, g: u8, s: u8, field7: u8) -> CmdExecIndirectResource {
        CmdExecIndirectResource {
            reference,
            invalidate_host: h,
            invalidate_guest: g,
            synchronize: s,
            field7,
            field8: u64::MAX,
            field10: u64::MAX,
        }
    }

    mod get {
        use super::*;

        /// 24-byte records, in order
        #[test]
        fn v2_entries_in_order() {
            let recs = [v2(5, 1, 0, 0, 0xff), v2(6, 0, 1, 1, 0)];
            let bytes = recs.as_bytes();
            let l = ResourceList::V2(Records::new(bytes, 2).unwrap());
            assert_eq!(l.count(), 2);
            let e0 = l.get(0).unwrap();
            assert_eq!((e0.reference, e0.flags), (5, 0b001));
            let e1 = l.get(1).unwrap();
            assert_eq!((e1.reference, e1.flags), (6, 0b110));
            assert!(e1.invalidate_guest() && e1.synchronize() && !e1.invalidate_host());
        }

        /// A flag byte keeps only its low bit
        #[test]
        fn v2_flag_bytes_cut_to_low_bit() {
            let recs = [v2(1, 2, 3, 0xfe, 0), v2(2, 0xff, 0x80, 0x81, 0)];
            let l = ResourceList::V2(Records::new(recs.as_bytes(), 2).unwrap());
            assert_eq!(l.get(0).unwrap().flags, 0b010);
            assert_eq!(l.get(1).unwrap().flags, 0b101);
        }

        /// 8-byte records: bits 0-2 kept, 3-31 dropped
        #[test]
        fn v3_keeps_three_bits() {
            let recs = [
                CmdExecIndirectResource3 {
                    reference: 9,
                    flags: 0xffff_fff8 | 0b101,
                },
                CmdExecIndirectResource3 {
                    reference: 10,
                    flags: 0b1000,
                },
            ];
            let l = ResourceList::V3(Records::new(recs.as_bytes(), 2).unwrap());
            assert_eq!(l.count(), 2);
            let e = l.get(0).unwrap();
            assert_eq!((e.reference, e.flags), (9, 0b101));
            assert_eq!(l.get(1).unwrap().flags, 0);
        }

        /// `index >= count` raises
        #[test]
        fn out_of_range_throws() {
            let recs = [v2(1, 1, 1, 1, 0)];
            let l = ResourceList::V2(Records::new(recs.as_bytes(), 1).unwrap());
            let e = l.get(1).unwrap_err();
            assert!(matches!(e.origin(), crate::raise::Origin::Core(_)));
            assert_eq!(
                e.detail().unwrap_or_default(),
                "resource record 1 is past the list's 1 records"
            );
            assert_eq!(e.site().file, "tasks.resource.index");
            let e = l.get(u64::from(u32::MAX) + 1).unwrap_err();
            assert_eq!(
                e.detail().unwrap_or_default(),
                "resource record 4294967296 is past the list's 1 records"
            );
            let empty = ResourceList::V3(Records::empty());
            assert_eq!(empty.count(), 0);
            assert!(empty.get(0).is_err());
        }

        /// An empty list's first record raises, under the handler every caller has
        #[test]
        fn out_of_range_on_an_empty_list_raises() {
            let l = ResourceList::V3(Records::empty());
            let r = l.get(0).map(|_| ());
            assert_eq!(r.map_err(|r| r.id()), Err(RaiseKind::ResourceIndex.into()));
        }
    }
}
