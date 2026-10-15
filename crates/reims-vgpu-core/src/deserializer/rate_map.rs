//! The rasterization-rate-map reset: a rate map rebuilt from its serialized record (the reset
//! operation 0x32, a descriptor record, its layers' sample counts, then each layer's samples).

use crate::deserializer::records::{
    OperationRasterizationRateMapDescriptor, OperationRasterizationRateMapLayer,
};
use crate::deserializer::{DeathKind, Deserializer, RaiseKind};
use crate::guest::GuestPtr;
use crate::host::HostPtr;
use crate::memory::cursor::{Cursor, CursorRange};
use crate::num::{ToU64, ToUsize};
use crate::platform::Platform;
use crate::platform::rate_map::{
    RateMap as _, RateMapDescriptor as _, RateMapLayerDescriptor as _,
};
use crate::platform::types::MtlSize;
use crate::platform::values::RateMapDesc;
use crate::raise::{OnFault, Raise, RaiseResult};

/// The record's header: the operation, then 8 bytes the reset skips.
const HEADER_SIZE: u64 = 8;
/// Where the descriptor record starts.
const DESCRIPTOR_OFFSET: u64 = 12;
/// The one operation the reset acts on.
const RESET_OPERATION: u32 = 0x32;
/// `MTLMutabilityMutable`.
const MUTABLE: u64 = 1;

/// Copies `bytes` of samples from `cursor` into Metal's sample storage at `into`, answering
/// whether it could. The bound is checked first, then the storage (a null one fails even for no
/// samples), then the copy.
fn read_samples(cursor: &mut Cursor, bytes: u64, into: HostPtr) -> bool {
    if bytes > cursor.remaining() || into.is_null() {
        return false;
    }
    let mut samples = vec![0u8; bytes.to_usize()];
    if cursor.read_bytes(&mut samples).is_err() {
        return false;
    }
    GuestPtr::new(into).copy_from(0, &samples, DeathKind::SampleStorage);
    true
}

impl<P: Platform> Deserializer<P> {
    /// Resets `map` (`None` when the reference named none) from `data`, the map's serialized
    /// record. Raises in the caller's frame.
    ///
    /// The size is checked, then the mutability, then the operation.
    pub fn reset_rasterization_rate_map_with_serialized_data(
        &self,
        data: &[u8],
        map: Option<&P::RateMap>,
    ) -> RaiseResult<()> {
        let size = data.len().to_u64();
        if size < HEADER_SIZE {
            Raise::new(RaiseKind::RateMapRecordShort)
                .with_detail(format_args!(
                    "rate map record of {size} bytes is shorter than its {HEADER_SIZE}-byte header"
                ))
                .fatal();
        }
        // An absent map is refused as immutable.
        if map.map_or(0, |m| m.mutability()) != MUTABLE {
            Raise::new(RaiseKind::RateMapImmutable)
                .with_detail("the rate map is immutable")
                .fatal();
        }
        let operation = data
            .first_chunk::<4>()
            .map_or(0, |b| u32::from_le_bytes(*b));
        if operation != RESET_OPERATION {
            // Another operation is a success that does nothing.
            return Ok(());
        }
        // The cursor covers `[data + 12, data + size + 4)`, four bytes past the record. The copy
        // here is padded with four zero bytes, so lengths, successes and failures are the record's.
        // The bytes past the record read as zeros.
        let mut padded = Vec::with_capacity(data.len().saturating_add(4));
        padded.extend_from_slice(data);
        padded.extend_from_slice(&[0u8; 4]);
        let cursor = Cursor::with_ranges(vec![CursorRange {
            address: HostPtr::from_ptr(padded.as_mut_ptr()).wrapping_add(DESCRIPTOR_OFFSET),
            length: size.wrapping_sub(HEADER_SIZE),
        }]);
        let descriptor = self.rate_map_descriptor(cursor)?;
        if let Some(map) = map {
            map.reset(&descriptor);
        }
        drop(descriptor);
        drop(padded);
        Ok(())
    }

    /// The Metal descriptor the record describes: made with its screen size, and its
    /// mutability and minimum factor when the record is mutable; then each layer is made and set,
    /// then each layer's samples are read into Metal's storage.
    fn rate_map_descriptor(&self, mut cursor: Cursor) -> RaiseResult<P::RateMapDescriptor> {
        let record =
            cursor.read_or_raise::<OperationRasterizationRateMapDescriptor>(OnFault::Die)?;
        let mutable = u64::from(record.mutability) == MUTABLE;
        let descriptor = self.platform().rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(
                u64::from(record.screen_width),
                u64::from(record.screen_height),
                1,
            ),
            mutability: mutable.then_some(MUTABLE),
            min_factor: mutable.then_some(record.min_factor),
        });
        let layer_count = u64::from(record.layer_count);
        for index in 0..layer_count {
            let layer = cursor.read_or_raise::<OperationRasterizationRateMapLayer>(OnFault::Die)?;
            let Some(layer_descriptor) = self.platform().new_rate_map_layer(
                OnFault::Die,
                MtlSize::new(
                    u64::from(layer.horizontal_sample_count),
                    u64::from(layer.vertical_sample_count),
                    0,
                ),
            )?
            else {
                Raise::new(RaiseKind::RateMapLayer)
                    .with_detail("no rate map layer descriptor")
                    .fatal();
            };
            descriptor.set_layer(OnFault::Die, &layer_descriptor, index)?;
        }
        for index in 0..layer_count {
            let layer = descriptor.layer(OnFault::Die, index)?;
            // Metal's sample counts, not the record's, decide what is read; the vertical samples
            // only after the horizontal ones.
            let sample_count = crate::nil_send!(layer, sample_count());
            let read = read_samples(
                &mut cursor,
                sample_count.width.wrapping_mul(4),
                crate::nil_send!(layer, horizontal_sample_storage()),
            ) && read_samples(
                &mut cursor,
                sample_count.height.wrapping_mul(4),
                crate::nil_send!(layer, vertical_sample_storage()),
            );
            if !read {
                Raise::new(RaiseKind::RateMapSamples)
                    .with_detail("rate map samples not readable")
                    .fatal();
            }
        }
        crate::fatal_unless!(
            descriptor.layer_count() == layer_count,
            DeathKind::RateMapLayerCount
        );
        Ok(descriptor)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use zerocopy::IntoBytes;

    use super::*;
    use crate::deserializer::test_lookups::Lookups;
    use crate::platform::Handle;
    use crate::raise::Raise;
    use crate::raise::death::{DeathId, DeathKind as DeathClass, DeathSite};
    use crate::testkit::capture::catch_death;
    use crate::testkit::guest_ram::RawRegion;
    use crate::testkit::objects::Model;
    use crate::testkit::{Arg, FakePlatform, Fk, kind};

    type P = FakePlatform;

    struct Setup {
        p: Arc<FakePlatform>,
        d: Arc<Deserializer<P>>,
    }

    fn setup() -> Setup {
        let p = Arc::new(FakePlatform::new());
        let r = Arc::new(Lookups::new(&p));
        let d = Deserializer::new(p.clone(), Some(p.device()), r.lookup());
        p.call_log().clear();
        Setup { p, d }
    }

    /// The site a raise of `k` dies at.
    fn died_of(k: RaiseKind) -> DeathSite {
        Raise::new(k).site()
    }

    /// The serialized record: operation, 8 skipped bytes, the descriptor, then `tail`.
    fn reset_data(op: u32, desc: OperationRasterizationRateMapDescriptor, tail: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&op.to_le_bytes());
        v.extend_from_slice(&[0xee; 8]);
        v.extend_from_slice(desc.as_bytes());
        v.extend_from_slice(tail);
        v
    }

    fn desc(
        w: u16,
        h: u16,
        layers: u32,
        mutability: u32,
        min: f32,
    ) -> OperationRasterizationRateMapDescriptor {
        OperationRasterizationRateMapDescriptor {
            screen_width: w,
            screen_height: h,
            layer_count: layers,
            mutability,
            min_factor: min,
        }
    }

    fn floats(fs: &[f32]) -> Vec<u8> {
        fs.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    fn mutable_map(s: &Setup) -> Fk<kind::RateMap> {
        let map: Fk<kind::RateMap> = s.p.world().new_obj(Model::None);
        map.set_mutability(1);
        map
    }

    /// Runs `set` on the next rate-map layer Metal makes.
    fn next_layer(s: &Setup, set: impl FnOnce(&Fk<kind::RateMapLayerDescriptor>) + Send + 'static) {
        s.p.world().on_new::<kind::RateMapLayerDescriptor>(set);
    }

    fn read_floats(ptr: HostPtr, n: usize) -> Vec<f32> {
        let mut b = vec![0u8; n * 4];
        GuestPtr::new(ptr).copy_to(0, &mut b, DeathKind::SampleStorage);
        b.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn reset(s: &Setup, data: &[u8], map: Option<&Fk<kind::RateMap>>) -> RaiseResult<()> {
        s.d.reset_rasterization_rate_map_with_serialized_data(data, map)
    }

    /// The reset with no handler, as the GPU task's job runs it: the site its raise dies at.
    fn reset_dies(s: &Setup, data: &[u8], map: Option<&Fk<kind::RateMap>>) -> DeathSite {
        catch_death(|| {
            let _ =
                s.d.reset_rasterization_rate_map_with_serialized_data(data, map);
        })
        .unwrap_err()
    }

    #[test]
    fn a_short_record_is_refused_before_the_map_is_asked() {
        // The size, then the mutability (an immutable map loses to the short record), then the
        // operation
        let s = setup();
        let map: Fk<kind::RateMap> = s.p.world().new_obj(Model::None);
        assert_eq!(
            reset_dies(&s, &[0x32, 0, 0, 0, 0, 0, 0], Some(&map)),
            died_of(RaiseKind::RateMapRecordShort)
        );
        assert!(s.p.call_log().calls().is_empty());
    }

    #[test]
    fn an_immutable_or_absent_map_is_refused() {
        let s = setup();
        let map: Fk<kind::RateMap> = s.p.world().new_obj(Model::None);
        let data = reset_data(0x32, desc(1, 1, 0, 0, 0.0), &[]);
        for m in [Some(&map), None] {
            assert_eq!(
                reset_dies(&s, &data, m),
                died_of(RaiseKind::RateMapImmutable)
            );
        }
        assert!(s.p.call_log().calls().is_empty());
        // Mutability 2 is not mutable either.
        map.set_mutability(2);
        assert_eq!(
            reset_dies(&s, &data, Some(&map)),
            died_of(RaiseKind::RateMapImmutable)
        );
    }

    #[test]
    fn another_operation_succeeds_doing_nothing() {
        // Another operation succeeds with no action, after the size and mutability checks
        let s = setup();
        let map = mutable_map(&s);
        let data = reset_data(0x31, desc(1, 1, 0, 0, 0.0), &[]);
        reset(&s, &data, Some(&map)).unwrap();
        reset(&s, &[0; 8], Some(&map)).unwrap();
        assert!(s.p.call_log().calls().is_empty());
    }

    #[test]
    fn resets_with_every_layer_and_its_samples() {
        let s = setup();
        let map = mutable_map(&s);
        let mut tail = Vec::new();
        tail.extend_from_slice(&[2, 0, 3, 0]); // layer 0: 2 x 3
        tail.extend_from_slice(&[1, 0, 1, 0]); // layer 1: 1 x 1
        tail.extend(floats(&[0.5, 1.0]));
        tail.extend(floats(&[0.25, 0.75, 1.0]));
        tail.extend(floats(&[2.0]));
        tail.extend(floats(&[3.0]));
        let data = reset_data(0x32, desc(640, 480, 2, 1, 0.125), &tail);
        reset(&s, &data, Some(&map)).unwrap();
        let calls = s.p.call_log().calls();
        let sels: Vec<&str> = calls.iter().map(|c| c.sel).collect();
        assert_eq!(
            sels,
            vec![
                "initWithSampleCount:",
                "setLayer:atIndex:",
                "initWithSampleCount:",
                "setLayer:atIndex:",
                "resetUsingDescriptor:",
            ]
        );
        let descriptor = calls[1].obj;
        assert_eq!(calls[0].args, vec![Arg::val(&MtlSize::new(2, 3, 0))]);
        assert_eq!(calls[1].args[1], Arg::U64(0));
        assert_eq!(calls[2].args, vec![Arg::val(&MtlSize::new(1, 1, 0))]);
        assert_eq!(
            (calls[3].obj, &calls[3].args[1]),
            (descriptor, &Arg::U64(1))
        );
        let reset = &calls[4];
        assert_eq!(reset.obj, map.id());
        assert_eq!(reset.args, vec![Arg::Obj(Some(descriptor))]);
    }

    #[test]
    fn samples_are_copied_into_metals_storage() {
        let s = setup();
        let map = mutable_map(&s);
        let horizontal = RawRegion::new(8).unwrap();
        let vertical = RawRegion::new(12).unwrap();
        let (h, v) = (horizontal.addr(), vertical.addr());
        next_layer(&s, move |l| {
            l.set_horizontal_sample_storage(h);
            l.set_vertical_sample_storage(v);
        });
        let mut tail = vec![2, 0, 3, 0];
        tail.extend(floats(&[0.5, 1.0, 0.25, 0.75, 1.0]));
        let data = reset_data(0x32, desc(4, 4, 1, 0, 9.0), &tail);
        reset(&s, &data, Some(&map)).unwrap();
        assert_eq!(read_floats(horizontal.addr(), 2), vec![0.5, 1.0]);
        assert_eq!(read_floats(vertical.addr(), 3), vec![0.25, 0.75, 1.0]);
    }

    #[test]
    fn a_record_four_bytes_short_still_resets() {
        // The cursor runs four bytes past the record.
        let s = setup();
        let map = mutable_map(&s);
        let vertical = RawRegion::new(4).unwrap();
        let v = vertical.addr();
        next_layer(&s, move |l| l.set_vertical_sample_storage(v));
        let mut tail = vec![1, 0, 1, 0];
        tail.extend(floats(&[0.5])); // the vertical sample is missing
        let data = reset_data(0x32, desc(1, 1, 1, 0, 0.0), &tail);
        reset(&s, &data, Some(&map)).unwrap();
        assert_eq!(s.p.call_log().count("resetUsingDescriptor:"), 1);
        assert_eq!(read_floats(vertical.addr(), 1), vec![0.0]);
    }

    #[test]
    fn eight_bytes_short_fails_to_read_samples() {
        let s = setup();
        let map = mutable_map(&s);
        let data = reset_data(0x32, desc(1, 1, 1, 0, 0.0), &[1, 0, 1, 0]);
        assert_eq!(
            reset_dies(&s, &data, Some(&map)),
            died_of(RaiseKind::RateMapSamples)
        );
        assert_eq!(s.p.call_log().count("resetUsingDescriptor:"), 0);
    }

    #[test]
    fn a_failed_horizontal_read_skips_the_vertical_one() {
        // Short-circuit.
        let s = setup();
        let map = mutable_map(&s);
        // Metal reports 100 horizontal samples for a record that says 1.
        let vertical = RawRegion::new(4).unwrap();
        let v = vertical.addr();
        next_layer(&s, move |l| {
            l.set_sample_count(MtlSize::new(100, 1, 0));
            l.set_vertical_sample_storage(v);
        });
        let mut tail = vec![1, 0, 1, 0];
        tail.extend(floats(&[0.5, 0.5]));
        let data = reset_data(0x32, desc(1, 1, 1, 0, 0.0), &tail);
        assert_eq!(
            reset_dies(&s, &data, Some(&map)),
            died_of(RaiseKind::RateMapSamples)
        );
        // The vertical sample, which would read 0.5, is not copied.
        assert_eq!(read_floats(vertical.addr(), 1), vec![0.0]);
    }

    #[test]
    fn metals_sample_count_decides_what_is_read() {
        // Metal says 1 x 1 for a record of 2 x 2.
        let s = setup();
        let map = mutable_map(&s);
        let h = RawRegion::new(4).unwrap();
        let storage = h.addr();
        next_layer(&s, move |l| {
            l.set_sample_count(MtlSize::new(1, 1, 0));
            l.set_horizontal_sample_storage(storage);
        });
        let mut tail = vec![2, 0, 2, 0];
        tail.extend(floats(&[4.0, 5.0]));
        let data = reset_data(0x32, desc(1, 1, 1, 0, 0.0), &tail);
        reset(&s, &data, Some(&map)).unwrap();
        assert_eq!(read_floats(h.addr(), 1), vec![4.0]);
    }

    #[test]
    fn a_null_storage_fails_even_for_zero_samples() {
        let s = setup();
        let map = mutable_map(&s);
        next_layer(&s, |l| {
            l.set_sample_count(MtlSize::new(0, 0, 0));
            l.set_horizontal_sample_storage(HostPtr::NULL);
        });
        let data = reset_data(0x32, desc(1, 1, 1, 0, 0.0), &[1, 0, 1, 0, 0, 0, 0, 0]);
        assert_eq!(
            reset_dies(&s, &data, Some(&map)),
            died_of(RaiseKind::RateMapSamples)
        );
    }

    #[test]
    fn a_refused_layer_stops_before_the_next_layer_header() {
        let s = setup();
        let map = mutable_map(&s);
        let data = reset_data(0x32, desc(1, 1, 2, 0, 0.0), &[0, 0, 1, 0, 1, 0, 1, 0]);
        assert_eq!(
            reset_dies(&s, &data, Some(&map)),
            died_of(RaiseKind::RateMapLayer)
        );
        assert_eq!(s.p.call_log().count("initWithSampleCount:"), 1);
        assert_eq!(s.p.call_log().count("setLayer:atIndex:"), 0);
    }

    #[test]
    fn a_short_descriptor_fails_to_read() {
        let s = setup();
        let map = mutable_map(&s);
        // 12 + 8 bytes: the cursor holds 12, the descriptor needs 16.
        let mut data = vec![0x32, 0, 0, 0];
        data.extend_from_slice(&[0; 16]);
        let site = reset_dies(&s, &data, Some(&map));
        assert_eq!(
            (site.kind, site.file),
            (
                DeathClass::UnhandledRaise,
                crate::memory::RaiseKind::CursorReadFailed.name()
            )
        );
        assert!(s.p.call_log().calls().is_empty());
    }

    #[test]
    fn a_layer_count_mismatch_dies() {
        let s = setup();
        let map = mutable_map(&s);
        s.p.world()
            .on_new::<kind::RateMapDescriptor>(|d| d.set_layer_count(3));
        let data = reset_data(0x32, desc(1, 1, 0, 0, 0.0), &[]);
        let r = catch_death(|| {
            let _ = reset(&s, &data, Some(&map));
        });
        assert_eq!(r, Err(DeathId::from(DeathKind::RateMapLayerCount).site()));
        assert_eq!(r.map_err(|d| d.kind), Err(DeathClass::Assert));
    }

    #[test]
    fn a_raise_with_no_handler_is_the_service_death() {
        let s = setup();
        let r = catch_death(|| {
            let _ =
                s.d.reset_rasterization_rate_map_with_serialized_data(&[0; 4], None);
        });
        assert_eq!(
            r.map_err(|d| (d.kind, d.file)),
            Err((
                DeathClass::UnhandledRaise,
                "deserializer.rate_map.record_short"
            ))
        );
    }

    /// The descriptor `record` (with no layers) builds, and what it holds: its screen size, its
    /// mutability and its minimum factor.
    fn built(record: OperationRasterizationRateMapDescriptor) -> RateMapDesc {
        let s = setup();
        let mut bytes = record.as_bytes().to_vec();
        bytes.extend_from_slice(&[0; 4]);
        let cursor = Cursor::with_ranges(vec![CursorRange {
            address: HostPtr::from_ptr(bytes.as_mut_ptr()),
            length: 16,
        }]);
        let rmd = s.d.rate_map_descriptor(cursor).unwrap();
        assert!(s.p.call_log().calls().is_empty());
        let held = rmd.desc();
        drop(bytes);
        held
    }

    #[test]
    fn zero_layers_build_an_empty_descriptor() {
        assert_eq!(
            built(desc(3, 4, 0, 1, 0.5)),
            RateMapDesc {
                screen_size: MtlSize::new(3, 4, 1),
                mutability: Some(1),
                min_factor: Some(0.5),
            }
        );
        // Mutability 0: neither the mutability nor the minimum factor is set.
        assert_eq!(
            built(desc(3, 4, 0, 0, 0.5)),
            RateMapDesc {
                screen_size: MtlSize::new(3, 4, 1),
                mutability: None,
                min_factor: None,
            }
        );
    }
}
