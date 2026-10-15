//! Buffers: the guest's buffer records (object types 1 and 15) and the kinds they make.
//!
//! [`new_buffer_resource`] picks the kind: a task that detaches backings makes a
//! [`detach`] buffer whatever the private bit says; otherwise the private bit picks
//! [`private`] or [`shared`]. Type 15 is a [`mapper_ref`] buffer. A heap makes [`child`]
//! buffers.

pub mod child;
pub mod detach;
pub mod mapper_ref;
pub mod private;
pub mod shared;

use std::sync::Arc;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::guest::GuestPtr;
use crate::platform::Platform;
use crate::raise::RaiseResult;
use crate::resources::resource::{Resource, TaskRef, read_descriptor};
use crate::resources::{DeathKind, RaiseKind, raise};
use crate::rt::Hold;

use detach::DetachBufferResource;
use mapper_ref::MapperRefBufferResource;
use private::PrivateBufferResource;
use shared::SharedBufferResource;

/// The guest's buffer record: 16 bytes, of which 13 are read.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct BufferResourceDescriptor {
    /// The buffer's length in bytes.
    pub length: u64,
    /// The task VA page it starts at.
    pub page: u32,
    /// Bit 0: private; bit 1: read-only.
    pub flags: u8,
    /// Padding nothing reads.
    pub unknown0d: [u8; 3],
}

const _: () = assert!(size_of::<BufferResourceDescriptor>() == 16);

impl BufferResourceDescriptor {
    /// Bit 0: a private buffer (ignored when the task detaches backings).
    pub fn is_private(&self) -> bool {
        self.flags & 0x1 != 0
    }
    /// Bit 1: read-only.
    pub fn is_read_only(&self) -> bool {
        self.flags & 0x2 != 0
    }
}

/// Raises unless the record is at least `minimum` bytes.
fn check_descriptor_length(length: usize, minimum: usize) -> RaiseResult<()> {
    if length < minimum {
        return raise(RaiseKind::DescriptorLength);
    }
    Ok(())
}

/// A buffer of `object_type` (1, or 15 for a mapper-ref buffer) from its record. `descriptor`
/// is the whole record. +1, `None` (another type, or a refusal), or a raise.
pub fn new_buffer_resource<P: Platform>(
    hold: Hold<'_>,
    task: TaskRef<P>,
    object_type: i32,
    descriptor: &[u8],
    placement: Option<GuestPtr>,
) -> RaiseResult<Option<Arc<Resource<P>>>> {
    match object_type {
        1 => new_normal_buffer(hold, task, descriptor, placement),
        15 => new_mapper_ref_buffer(task, descriptor, placement),
        _ => Ok(None),
    }
}

fn new_normal_buffer<P: Platform>(
    hold: Hold<'_>,
    task: TaskRef<P>,
    descriptor: &[u8],
    placement: Option<GuestPtr>,
) -> RaiseResult<Option<Arc<Resource<P>>>> {
    // 8, although the kinds read 13 bytes.
    check_descriptor_length(descriptor.len(), 8)?;
    let d: BufferResourceDescriptor = read_descriptor(descriptor);
    // A torn-down task answers false to both questions.
    let t = task.upgrade();
    if t.as_ref()
        .is_some_and(|t| t.supports_resource_detach_backing())
    {
        let supports_ranges = t
            .as_ref()
            .is_some_and(|t| t.supports_buffer_with_address_ranges());
        crate::fatal_unless!(supports_ranges, DeathKind::DetachBufferWithoutRanges);
        drop(t);
        return DetachBufferResource::new(task, descriptor, placement);
    }
    drop(t);
    if d.is_private() {
        return PrivateBufferResource::new(task, descriptor, placement);
    }
    SharedBufferResource::new(hold, task, descriptor, placement)
}

fn new_mapper_ref_buffer<P: Platform>(
    task: TaskRef<P>,
    descriptor: &[u8],
    placement: Option<GuestPtr>,
) -> RaiseResult<Option<Arc<Resource<P>>>> {
    // The mapper record's size.
    check_descriptor_length(descriptor.len(), 8)?;
    MapperRefBufferResource::new(task, descriptor, placement)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::resource::ResourceKind;
    use crate::resources::resource::test_support::{Env, gone_task, kind};
    use crate::testkit::capture::catch_death;
    use crate::testkit::platform::FakePlatform;
    use crate::testkit::script::Step;

    type Fp = FakePlatform;

    pub(super) fn desc(length: u64, page: u32, flags: u8) -> Vec<u8> {
        BufferResourceDescriptor {
            length,
            page,
            flags,
            unknown0d: [0; 3],
        }
        .as_bytes()
        .to_vec()
    }

    fn in_process(env: &Env) {
        env.p
            .script()
            .always("supports_resource_detach_backing", Step::value(false));
    }

    fn kind_name(r: &Resource<Fp>) -> &'static str {
        match r.kind {
            ResourceKind::SharedBuffer(_) => "shared",
            ResourceKind::PrivateBuffer(_) => "private",
            ResourceKind::DetachBuffer(_) => "detach",
            ResourceKind::MapperRefBuffer(_) => "mapper",
            _ => "other",
        }
    }

    /// The object type and the private bit pick the kind; other types make nothing
    #[test]
    fn object_types_pick_the_kind() {
        let env = Env::new();
        in_process(&env);
        let t = || env.task();
        let mk = |ty, d: &[u8]| env.run(|| new_buffer_resource(Hold::NONE, t(), ty, d, None));
        assert_eq!(
            kind_name(&mk(1, &desc(16, 1, 0)).unwrap().unwrap()),
            "shared"
        );
        assert_eq!(
            kind_name(&mk(1, &desc(16, 1, 1)).unwrap().unwrap()),
            "private"
        );
        assert_eq!(
            kind_name(&mk(15, &7u64.to_le_bytes()).unwrap().unwrap()),
            "mapper"
        );
        for ty in [0, 2, 13, 14, -1] {
            assert!(mk(ty, &desc(16, 1, 0)).unwrap().is_none(), "{ty}");
        }
    }

    /// A task that detaches backings gets a detach buffer whatever the private bit, after both
    /// questions in order
    #[test]
    fn detach_backing_takes_the_detach_kind_whatever_the_private_bit() {
        let env = Env::new();
        for flags in [0, 1] {
            let r = env
                .run(|| {
                    new_buffer_resource(Hold::NONE, env.task(), 1, &desc(0x4000, 1, flags), None)
                })
                .unwrap()
                .unwrap();
            assert_eq!(kind_name(&r), "detach");
        }
        let sels = env.rec.log().sels();
        assert!(
            env.rec.log().contains_in_order(&[
                "supports_resource_detach_backing",
                "supports_buffer_with_address_ranges"
            ]),
            "{sels:?}"
        );
    }

    /// Detaching without address ranges is fatal
    #[test]
    fn detach_without_address_ranges_dies() {
        let env = Env::new();
        env.p
            .script()
            .push("supports_buffer_with_address_ranges", Step::value(false));
        let d = catch_death(|| {
            env.run(|| new_buffer_resource(Hold::NONE, env.task(), 1, &desc(16, 1, 0), None))
        });
        assert_eq!(d.err(), Some(DeathKind::DetachBufferWithoutRanges.site()));
    }

    /// A record shorter than 8 bytes raises before the task is asked anything
    #[test]
    fn short_descriptors_raise_before_anything_is_asked() {
        let env = Env::new();
        let hold = Hold::NONE;
        for ty in [1, 15] {
            assert_eq!(
                kind(new_buffer_resource(hold, env.task(), ty, &[0; 7], None)),
                Some(RaiseKind::DescriptorLength)
            );
        }
        assert!(env.rec.log().is_empty());
        // Eight bytes pass (the bound is 8, not the 13 bytes read).
        in_process(&env);
        let r = env
            .run(|| new_buffer_resource(hold, env.task(), 1, &[0x10, 0, 0, 0, 0, 0, 0, 0], None));
        assert!(r.unwrap().is_some());
    }

    /// A torn-down task does not detach, so it gets a shared buffer, whose guest pages then
    /// raise
    #[test]
    fn a_gone_task_takes_the_shared_kind() {
        let hold = Hold::NONE;
        assert_eq!(
            kind(new_buffer_resource::<Fp>(
                hold,
                gone_task(),
                1,
                &desc(16, 1, 0),
                None
            )),
            Some(RaiseKind::NoGuestBuffer)
        );
    }
}
