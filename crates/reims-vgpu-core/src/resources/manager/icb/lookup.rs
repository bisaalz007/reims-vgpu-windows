//! The buffer lookups the walks make: a bind's buffer by reference or by GPU address, the bind
//! a written-back address becomes, and the walks' manager once the task's teardown has dropped
//! it ([`NoManager`]).

use std::marker::PhantomData;

use crate::platform::Platform;
use crate::platform::resource::Buffer;
use crate::raise::RaiseResult;
use crate::resources::manager::records::{IndirectCommandBufferBind, ResourceAndOffset};
use crate::rt::Hold;

use super::{DeathKind, IcbWalkOps};

/// The buffer at a GPU address: `floor` is the greatest registered address at or below it and
/// its reference, `existing` the cached resource of that reference (never made), `get_buffer`
/// the resource's buffer. Never a raise: a resource without a buffer is an error.
pub(crate) fn resource_and_offset_for_address<P: Platform, R>(
    hold: Hold<'_>,
    address: u64,
    floor: impl FnOnce(u64) -> Option<(u64, u32)>,
    existing: impl FnOnce(u32) -> Option<Option<R>>,
    get_buffer: impl FnOnce(&R) -> RaiseResult<Option<P::Buffer>>,
) -> Result<ResourceAndOffset<P>, String> {
    if address == 0 || address == u64::MAX {
        return Err(format!("GPU address {address:#x} is not valid"));
    }
    let Some((start, reference)) = floor(address) else {
        return Err(format!(
            "no buffer registered at or below GPU address {address:#x}"
        ));
    };
    let Some(resource) = existing(reference) else {
        return Err(format!(
            "resource {reference} at GPU address {address:#x} is not cached"
        ));
    };
    // The floor is at or below the address, so this holds.
    crate::fatal_unless!(address >= start, DeathKind::AddressBelowFloor);
    let offset = address.wrapping_sub(start);
    // The offset is taken before the buffer is asked for
    let buffer = match &resource {
        Some(r) => get_buffer(r),
        None => Ok(None),
    };
    let buffer = match buffer {
        Ok(buffer) => buffer,
        Err(raised) => {
            let e = format!("resource {reference} at GPU address {address:#x} is not a buffer");
            hold.land(raised);
            return Err(e);
        }
    };
    // A nil buffer has length 0.
    if offset >= buffer.as_ref().map_or(0, |b| b.length()) {
        return Err(format!(
            "GPU address {address:#x} is past the end of buffer {reference}"
        ));
    }
    Ok(ResourceAndOffset {
        buffer,
        buffer_reference: reference,
        offset,
    })
}

/// The buffer a guest bind names: by GPU address when its reference is all ones and its address
/// nonzero, otherwise by reference (made on a miss). A raise inside the lookup (a creation's, a
/// resource with no buffer) is caught here and becomes the error.
pub(crate) fn resource_and_offset_for_bind<P: Platform, M: IcbWalkOps<P>>(
    hold: Hold<'_>,
    m: &M,
    bind: &IndirectCommandBufferBind,
) -> Result<ResourceAndOffset<P>, String> {
    let reference = bind.buffer_reference;
    let gpu_address = bind.gpu_address;
    let offset = bind.offset;
    if reference == u32::MAX && gpu_address != 0 {
        return m.resource_and_offset_at(hold, gpu_address);
    }
    match m.get_buffer_object(hold, reference) {
        // The bind's offset is not checked
        Ok(Ok(buffer)) => Ok(ResourceAndOffset {
            buffer,
            buffer_reference: reference,
            offset,
        }),
        Ok(Err(e)) => Err(format!("no buffer for reference {reference}: {e}")),
        Err(raise) => {
            let e = format!(
                "no buffer for reference {reference}: {}",
                raise.detail().unwrap_or_default()
            );
            hold.land(raise);
            Err(e)
        }
    }
}

/// The bind a written-back GPU address becomes: its buffer's reference and offset, or all ones
/// and offset 0 when it does not resolve (the lookup's error is dropped).
pub(crate) fn fill_buffer_bind<P: Platform, M: IcbWalkOps<P>>(
    hold: Hold<'_>,
    m: &M,
    address: u64,
) -> IndirectCommandBufferBind {
    match m.resource_and_offset_at(hold, address) {
        Ok(r) => IndirectCommandBufferBind {
            buffer_reference: r.buffer_reference,
            gpu_address: address,
            offset: r.offset,
        },
        Err(_) => IndirectCommandBufferBind {
            buffer_reference: u32::MAX,
            gpu_address: address,
            offset: 0,
        },
    }
}

/// The walks' manager once the task's teardown has dropped it: a CPU prepare or synchronize of an
/// ICB still runs its walk, which answers as usual until it first touches the manager (reference 0
/// is nil, an all-ones pipeline identifier or a 0 or all-ones address is an error) and dies there.
pub(crate) struct NoManager<P>(PhantomData<fn() -> P>);

impl<P> NoManager<P> {
    /// The walks' manager with none behind it.
    pub(crate) const fn new() -> Self {
        NoManager(PhantomData)
    }

    /// A lookup by reference: 0 is nil before the manager is touched.
    fn get_object<T>(reference: u32) -> RaiseResult<Result<Option<T>, String>> {
        if reference == 0 {
            return Ok(Ok(None));
        }
        crate::fatal!(DeathKind::NoManager)
    }
}

impl<P: Platform> IcbWalkOps<P> for NoManager<P> {
    fn get_pipeline_ref(&self, identifier: u64) -> Result<u32, String> {
        if identifier == u64::MAX {
            return Err(format!("pipeline identifier {identifier:#x} is not valid"));
        }
        crate::fatal!(DeathKind::NoManager)
    }

    fn get_buffer_object(
        &self,
        _hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<P::Buffer>, String>> {
        Self::get_object(reference)
    }

    fn get_render_pipeline_object(
        &self,
        _hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<P::RenderPipeline>, String>> {
        Self::get_object(reference)
    }

    fn get_compute_pipeline_object(
        &self,
        _hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<P::ComputePipeline>, String>> {
        Self::get_object(reference)
    }

    fn resource_and_offset_at(
        &self,
        hold: Hold<'_>,
        address: u64,
    ) -> Result<ResourceAndOffset<P>, String> {
        // The address is checked before the registry is read
        resource_and_offset_for_address::<P, ()>(
            hold,
            address,
            |_| crate::fatal!(DeathKind::NoManager),
            |_| crate::fatal!(DeathKind::NoManager),
            |_| Ok(None),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;
    use crate::platform::Handle;
    use crate::testkit::capture::catch_death;
    use crate::testkit::{FakePlatform, Fk, kind};

    mod by_address {
        use super::*;

        fn run(
            address: u64,
            floor: Option<(u64, u32)>,
            existing: Option<Option<Option<Fk<kind::Buffer>>>>,
        ) -> Result<ResourceAndOffset<FakePlatform>, String> {
            resource_and_offset_for_address::<FakePlatform, Option<Fk<kind::Buffer>>>(
                Hold::NONE,
                address,
                |_| floor,
                |_| existing,
                |b| match b {
                    Some(b) => Ok(Some(b.clone_ref())),
                    None => Err(not_a_buffer()),
                },
            )
        }

        #[test]
        fn zero_and_all_ones_are_invalid_before_any_lookup() {
            let e = err(run(0, None, None));
            assert_eq!(e, "GPU address 0x0 is not valid");
            let e = err(run(u64::MAX, None, None));
            assert_eq!(e, "GPU address 0xffffffffffffffff is not valid");
        }

        #[test]
        fn no_floor_is_not_found() {
            let e = err(run(0x1000, None, None));
            assert_eq!(e, "no buffer registered at or below GPU address 0x1000");
        }

        #[test]
        fn a_missing_resource() {
            let e = err(run(0x1000, Some((0x1000, 3)), None));
            assert_eq!(e, "resource 3 at GPU address 0x1000 is not cached");
        }

        #[test]
        fn a_raise_in_get_buffer_is_not_a_buffer() {
            let e = err(run(0x1008, Some((0x1000, 3)), Some(Some(None))));
            assert_eq!(e, "resource 3 at GPU address 0x1008 is not a buffer");
        }

        #[test]
        fn a_nil_resource_is_a_nil_buffer_of_length_zero() {
            // A nil buffer's length is 0, so any offset is past its end.
            let e = err(run(0x1000, Some((0x1000, 3)), Some(None)));
            assert_eq!(e, "GPU address 0x1000 is past the end of buffer 3");
        }

        #[test]
        fn the_offset_must_be_inside_the_buffer() {
            let o = FakeOps::new();
            let b = o.buffer(0x10);
            let e = err(run(
                0x1010,
                Some((0x1000, 3)),
                Some(Some(Some(b.clone_ref()))),
            ));
            assert_eq!(e, "GPU address 0x1010 is past the end of buffer 3");
            let r = ok(run(
                0x100f,
                Some((0x1000, 3)),
                Some(Some(Some(b.clone_ref()))),
            ));
            assert_eq!(r.buffer.unwrap().id(), b.id());
            assert_eq!((r.buffer_reference, r.offset), (3, 0xf));
        }

        #[test]
        fn a_floor_above_the_address_dies() {
            let d = catch_death(|| run(0x1000, Some((0x2000, 3)), Some(None)).is_ok()).unwrap_err();
            assert_eq!(d, death(DeathKind::AddressBelowFloor));
        }
    }

    mod by_bind {
        use super::*;

        #[test]
        fn reference_all_ones_with_an_address_looks_up_the_address() {
            let mut o = FakeOps::new();
            let b = o.resident_buffer(4, 0x100);
            let r = ok(resource_and_offset_for_bind(
                Hold::NONE,
                &o,
                &bind(u32::MAX, b.gpu_address() + 0x20, 7),
            ));
            assert_eq!((r.buffer_reference, r.offset), (4, 0x20));
            assert_eq!(
                o.calls(),
                vec![format!(
                    "resource_and_offset_at({:#x})",
                    b.gpu_address() + 0x20
                )]
            );
        }

        #[test]
        fn reference_all_ones_without_an_address_is_a_reference_lookup() {
            let o = FakeOps::new();
            let e = err(resource_and_offset_for_bind(
                Hold::NONE,
                &o,
                &bind(u32::MAX, 0, 0),
            ));
            assert_eq!(e, "no buffer for reference 4294967295: Invalid reference");
            assert_eq!(o.calls(), vec!["get_buffer_object(4294967295)"]);
        }

        #[test]
        fn a_found_buffer_keeps_the_binds_offset_unchecked() {
            let mut o = FakeOps::new();
            let b = o.resident_buffer(4, 0x10);
            let r = ok(resource_and_offset_for_bind(
                Hold::NONE,
                &o,
                &bind(4, 0x1234, 0x9999),
            ));
            assert_eq!(r.buffer.unwrap().id(), b.id());
            assert_eq!((r.buffer_reference, r.offset), (4, 0x9999));
            assert_eq!(o.calls(), vec!["get_buffer_object(4)"]);
        }

        #[test]
        fn a_nil_buffer_is_found() {
            let mut o = FakeOps::new();
            o.buffers.push((4, Lookup::Found(None)));
            let r = ok(resource_and_offset_for_bind(Hold::NONE, &o, &bind(4, 0, 1)));
            assert!(r.buffer.is_none());
        }

        #[test]
        fn a_lookup_error_is_wrapped() {
            let mut o = FakeOps::new();
            o.buffers.push((4, Lookup::Missing("Object type mismatch")));
            let e = err(resource_and_offset_for_bind(Hold::NONE, &o, &bind(4, 0, 1)));
            assert_eq!(e, "no buffer for reference 4: Object type mismatch");
        }

        #[test]
        fn a_raise_in_the_lookup_becomes_the_error() {
            let mut o = FakeOps::new();
            o.buffers.push((4, Lookup::Raise("wrong kind")));
            let e = err(resource_and_offset_for_bind(Hold::NONE, &o, &bind(4, 0, 1)));
            assert!(e.starts_with("no buffer for reference 4: "), "{e}");
            assert!(e.contains("wrong kind"), "{e}");
        }
    }

    mod fill {
        use super::*;

        #[test]
        fn a_resolved_address_gets_its_reference_and_offset() {
            let mut o = FakeOps::new();
            let b = o.resident_buffer(9, 0x100);
            let a = b.gpu_address() + 0x30;
            let got = fill_buffer_bind(Hold::NONE, &o, a);
            assert_eq!(
                (got.buffer_reference, got.gpu_address, got.offset),
                (9, a, 0x30)
            );
        }

        #[test]
        fn an_unresolved_address_is_all_ones_and_offset_zero() {
            let o = FakeOps::new();
            let got = fill_buffer_bind(Hold::NONE, &o, 0x5000);
            assert_eq!(
                (got.buffer_reference, got.gpu_address, got.offset),
                (u32::MAX, 0x5000, 0)
            );
            let got = fill_buffer_bind(Hold::NONE, &o, 0);
            assert_eq!(
                (got.buffer_reference, got.gpu_address, got.offset),
                (u32::MAX, 0, 0)
            );
        }
    }

    /// The walks' manager after the task's teardown
    mod no_manager {
        use super::*;

        fn none() -> NoManager<FakePlatform> {
            NoManager::new()
        }

        #[test]
        fn what_answers_before_touching_the_manager_answers() {
            let m = none();
            assert!(
                m.get_buffer_object(Hold::NONE, 0)
                    .unwrap()
                    .unwrap()
                    .is_none()
            );
            assert!(
                m.get_render_pipeline_object(Hold::NONE, 0)
                    .unwrap()
                    .unwrap()
                    .is_none()
            );
            assert!(
                m.get_compute_pipeline_object(Hold::NONE, 0)
                    .unwrap()
                    .unwrap()
                    .is_none()
            );
            assert_eq!(
                m.get_pipeline_ref(u64::MAX),
                Err("pipeline identifier 0xffffffffffffffff is not valid".to_owned())
            );
            for a in [0, u64::MAX] {
                assert_eq!(
                    m.resource_and_offset_at(Hold::NONE, a).err(),
                    Some(format!("GPU address {a:#x} is not valid"))
                );
            }
        }

        #[test]
        fn the_first_use_of_the_manager_dies() {
            let m = none();
            let died = Some(death(DeathKind::NoManager));
            assert_eq!(
                catch_death(|| m.get_buffer_object(Hold::NONE, 1).is_ok()).err(),
                died
            );
            assert_eq!(
                catch_death(|| m.get_render_pipeline_object(Hold::NONE, 2).is_ok()).err(),
                died
            );
            assert_eq!(
                catch_death(|| m.get_compute_pipeline_object(Hold::NONE, 3).is_ok()).err(),
                died
            );
            assert_eq!(catch_death(|| m.get_pipeline_ref(7).is_ok()).err(), died);
            assert_eq!(
                catch_death(|| m.resource_and_offset_at(Hold::NONE, 0x1000).is_ok()).err(),
                died
            );
        }
    }
}
