//! Heaps: the guest's heap records (object type 13) and the kinds they make.
//!
//! [`new_heap_resource`] picks the kind: a task that detaches backings makes a [`detach`]
//! heap; otherwise a [`legacy`] heap, which must be private. Both make child buffers
//! ([`super::buffer::child`]) in themselves.

pub mod detach;
pub mod legacy;

use std::sync::Arc;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::guest::GuestPtr;
use crate::platform::Platform;
use crate::raise::RaiseResult;
use crate::resources::resource::{Resource, TaskRef};
use crate::resources::{DeathKind, RaiseKind, raise};

use detach::DetachHeapResource;
use legacy::LegacyHeapResource;

/// The guest's heap record: packed, 13 bytes read (the page at +0, the length at +4, the flags
/// at +12).
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    FromBytes,
    IntoBytes,
    Immutable,
    KnownLayout,
    Unaligned,
)]
#[repr(C, packed)]
pub struct HeapResourceDescriptor {
    /// The task VA page the heap starts at.
    pub page: [u8; 4],
    /// The heap's size.
    pub length: [u8; 8],
    /// Bit 0: private; bit 1: tracked; bit 2: read-only.
    pub flags: u8,
}

const _: () = assert!(size_of::<HeapResourceDescriptor>() == 13);

impl HeapResourceDescriptor {
    /// The first page.
    pub fn page(&self) -> u32 {
        u32::from_le_bytes(self.page)
    }
    /// The size.
    pub fn length(&self) -> u64 {
        u64::from_le_bytes(self.length)
    }
    /// Bit 0: private (a legacy heap must be).
    pub fn is_private(&self) -> bool {
        self.flags & 0x1 != 0
    }
    /// Bit 1: hazard-tracked.
    pub fn is_tracked(&self) -> bool {
        self.flags & 0x2 != 0
    }
    /// Bit 2: read-only.
    pub fn is_read_only(&self) -> bool {
        self.flags & 0x4 != 0
    }
}

/// A heap of `object_type` (13) from its record. +1, `None` (another type, or a refusal), or a
/// raise.
pub fn new_heap_resource<P: Platform>(
    task: TaskRef<P>,
    object_type: i32,
    descriptor: &[u8],
    placement: Option<GuestPtr>,
) -> RaiseResult<Option<Arc<Resource<P>>>> {
    if object_type == 13 {
        return new_normal_heap_resource(task, descriptor, placement);
    }
    Ok(None)
}

fn new_normal_heap_resource<P: Platform>(
    task: TaskRef<P>,
    descriptor: &[u8],
    placement: Option<GuestPtr>,
) -> RaiseResult<Option<Arc<Resource<P>>>> {
    // The caller passes 16: never taken.
    if descriptor.len() < 8 {
        return raise(RaiseKind::DescriptorLength);
    }
    // A torn-down task answers false to both questions.
    let t = task.upgrade();
    if t.as_ref()
        .is_some_and(|t| t.supports_resource_detach_backing())
    {
        let supports_ranges = t
            .as_ref()
            .is_some_and(|t| t.supports_heap_with_address_ranges());
        crate::fatal_unless!(supports_ranges, DeathKind::DetachHeapWithoutRanges);
        drop(t);
        return DetachHeapResource::new(task, descriptor, placement);
    }
    drop(t);
    LegacyHeapResource::new(task, descriptor, placement)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::resource::ResourceKind;
    use crate::resources::resource::test_support::{Env, gone_task, kind, logged, placement};
    use crate::testkit::capture::catch_death;
    use crate::testkit::platform::FakePlatform;
    use crate::testkit::script::Step;

    type Fp = FakePlatform;

    /// A 16-byte heap record.
    pub(super) fn heap_desc(page: u32, length: u64, flags: u8) -> Vec<u8> {
        let mut v = HeapResourceDescriptor {
            page: page.to_le_bytes(),
            length: length.to_le_bytes(),
            flags,
        }
        .as_bytes()
        .to_vec();
        v.extend_from_slice(&[0; 3]);
        v
    }

    /// Type 13 only; the task's detach support picks the kind
    #[test]
    fn type_13_only_and_the_task_picks_the_kind() {
        let env = Env::new();
        let mk = |ty| env.run(|| new_heap_resource(env.task(), ty, &heap_desc(1, 0x4000, 1), None));
        for ty in [0, 1, 14, 15] {
            assert!(mk(ty).unwrap().is_none());
        }
        let r = mk(13).unwrap().unwrap();
        assert!(matches!(r.kind, ResourceKind::DetachHeap(_)));
        env.p
            .script()
            .always("supports_resource_detach_backing", Step::value(false));
        let r = mk(13).unwrap().unwrap();
        assert!(matches!(r.kind, ResourceKind::LegacyHeap(_)));
    }

    /// A record shorter than 8 bytes raises
    #[test]
    fn a_short_descriptor_raises() {
        let env = Env::new();
        assert_eq!(
            kind(new_heap_resource(env.task(), 13, &[0; 7], None)),
            Some(RaiseKind::DescriptorLength)
        );
    }

    /// Detaching without address ranges is fatal
    #[test]
    fn detach_without_address_ranges_dies() {
        let env = Env::new();
        env.p
            .script()
            .push("supports_heap_with_address_ranges", Step::value(false));
        let d = catch_death(|| new_heap_resource(env.task(), 13, &heap_desc(1, 16, 1), None));
        assert_eq!(d.err(), Some(DeathKind::DetachHeapWithoutRanges.site()));
    }

    /// A torn-down task does not detach, so it gets a legacy heap (which refuses a placement),
    /// and Metal, asked of no device, never makes it
    #[test]
    fn a_gone_task_takes_the_legacy_kind() {
        let env = Env::new();
        let (_m, pl) = placement(0);
        let (r, lines) = logged(|| {
            env.run(|| new_heap_resource::<Fp>(gone_task(), 13, &heap_desc(1, 16, 1), Some(pl)))
        });
        assert!(r.unwrap().is_none());
        assert_eq!(lines, vec!["private heap refused: it takes no placement"]);
        let r = env.run(|| new_heap_resource::<Fp>(gone_task(), 13, &heap_desc(1, 16, 1), None));
        assert!(r.unwrap().is_none());
        assert!(env.p.call_log().calls().is_empty());
    }
}
