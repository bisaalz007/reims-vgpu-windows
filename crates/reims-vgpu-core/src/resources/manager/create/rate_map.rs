//! Rasterization rate maps (object type 7, command 50): the record's layers are set on a new
//! map descriptor, then each layer's samples are read into Metal's storage. No placement is
//! touched.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use super::STATE_OBJECT_TYPE;
use crate::guest::GuestPtr;
use crate::guest::records::ObjectEntry;
use crate::host::HostPtr;
use crate::memory::cursor::{Cursor, CursorError};
use crate::num::ToUsize;
use crate::platform::Platform;
use crate::platform::device::GpuDevice;
use crate::platform::rate_map::{RateMapDescriptor, RateMapLayerDescriptor};
use crate::platform::types::MtlSize;
use crate::platform::values::RateMapDesc;
use crate::raise::{OnFault, RaiseResult};
use crate::resources::manager::DeathKind;
use crate::resources::manager::icb::ManagerOps;
use crate::resources::manager::records::OperationHeader;

/// The command type that makes a rate map from a serialized descriptor.
const NEW_RATE_MAP: u32 = 50;

/// `MTLMutabilityMutable`, the record's and Metal's.
const MUTABILITY_MUTABLE: u32 = 1;

/// The serialized map descriptor (16 bytes). `layer_count` is the metadata's name; the rest are
/// ours, after the properties they set.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
#[allow(missing_docs)]
pub(crate) struct RateMapRecord {
    pub screen_width: u16,
    pub screen_height: u16,
    pub layer_count: u32,
    /// 1 is mutable.
    pub mutability: u32,
    pub min_factor: f32,
}

/// One serialized layer (4 bytes): its sample counts.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
#[allow(missing_docs)]
pub(crate) struct RateMapLayerRecord {
    pub horizontal_sample_count: u16,
    pub vertical_sample_count: u16,
}

/// The command (0x14 bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub(crate) struct NewRateMap {
    /// Nothing reads it.
    pub field_00: u32,
    /// The map's descriptor.
    pub descriptor: RateMapRecord,
}

const _: () = assert!(size_of::<NewRateMap>() == 0x14);

/// Reads `bytes` of samples into `into`, Metal's storage for them: the length check first,
/// then the pointer's, then the copy. The counts are the guest's `u16`s, so `bytes` is at most
/// 0x3fffc.
fn read_samples(cursor: &mut Cursor, bytes: u64, into: HostPtr) -> Result<(), CursorError> {
    if bytes > cursor.remaining() {
        // The cursor's own length check refuses, with its message, and reads nothing
        return cursor.read_bytes(&mut vec![0u8; bytes.to_usize()]);
    }
    if into.is_null() {
        return Err("no sample storage to read into".to_owned());
    }
    let mut samples = vec![0u8; bytes.to_usize()];
    cursor.read_bytes(&mut samples)?;
    GuestPtr::new(into).copy_from(0, &samples, DeathKind::SampleStorage);
    Ok(())
}

/// A map descriptor of `record` and its layers, read from `cursor`.
///
/// Each layer descriptor goes into `layers`, which the caller holds past the map descriptor's
/// release, so the layers and their sample storage outlive it on every return.
fn new_descriptor<P: Platform, M: ManagerOps<P>>(
    m: &M,
    record: &RateMapRecord,
    cursor: &mut Cursor,
    layers: &mut Vec<P::RateMapLayerDescriptor>,
) -> RaiseResult<Result<P::RateMapDescriptor, String>> {
    // A mutable record's map is mutable and takes the record's minimum factor. Both are set with
    // the screen size, before the layers. Metal stores any factor without a raise (NaN, the
    // infinities, negatives, subnormals), and the maps it makes (parameter data, physical sizes,
    // coordinate maps) are the same whether the two are set before or after the layers (measured
    // on the M3 Max).
    let mutable = record.mutability == MUTABILITY_MUTABLE;
    let d = m.platform().rate_map_descriptor(&RateMapDesc {
        screen_size: MtlSize {
            width: u64::from(record.screen_width),
            height: u64::from(record.screen_height),
            depth: 1,
        },
        mutability: mutable.then_some(u64::from(MUTABILITY_MUTABLE)),
        min_factor: mutable.then_some(record.min_factor),
    });
    let layer_count = record.layer_count;
    for i in 0..layer_count {
        let layer = refuse!(
            cursor
                .read::<RateMapLayerRecord>()
                .map_err(|e| format!("rate map layer {i} not readable: {e}"))
        );
        // Depth 0
        let layer_descriptor = m.platform().new_rate_map_layer(
            OnFault::Return,
            MtlSize {
                width: u64::from(layer.horizontal_sample_count),
                height: u64::from(layer.vertical_sample_count),
                depth: 0,
            },
        )?;
        let Some(layer_descriptor) = layer_descriptor else {
            return Ok(Err(format!(
                "Metal made no descriptor for rate map layer {i}"
            )));
        };
        // Metal aborts at the third layer on the suite's host (render_ratemap_three_layers)
        let set = d.set_layer(OnFault::Return, &layer_descriptor, u64::from(i));
        layers.push(layer_descriptor);
        set?;
    }
    for i in 0..layer_count {
        let layer = d.layer(OnFault::Return, u64::from(i))?;
        let sample_count = layer
            .as_ref()
            .map_or_else(MtlSize::default, |l| l.sample_count());
        // Both sample arrays are read before either read is checked.
        let horizontal = layer
            .as_ref()
            .map_or(HostPtr::NULL, |l| l.horizontal_sample_storage());
        let width = read_samples(cursor, sample_count.width.wrapping_mul(4), horizontal);
        let vertical = layer
            .as_ref()
            .map_or(HostPtr::NULL, |l| l.vertical_sample_storage());
        let height = read_samples(cursor, sample_count.height.wrapping_mul(4), vertical);
        if let Err(e) = width {
            return Ok(Err(format!(
                "rate map layer {i}'s horizontal samples not readable: {e}"
            )));
        }
        if let Err(e) = height {
            return Ok(Err(format!(
                "rate map layer {i}'s vertical samples not readable: {e}"
            )));
        }
    }
    crate::fatal_unless!(
        d.layer_count() == u64::from(layer_count),
        DeathKind::RateMapLayerCount
    );
    Ok(Ok(d))
}

/// A rate map from `entry` and its record.
pub(crate) fn create<P: Platform, M: ManagerOps<P>>(
    m: &M,
    entry: &ObjectEntry,
    mut cursor: Cursor,
) -> RaiseResult<Result<P::RateMap, String>> {
    let object_type = entry.object_type();
    if object_type != STATE_OBJECT_TYPE {
        return Ok(Err(format!(
            "object of type {object_type} is not a rasterization rate map"
        )));
    }
    let header = refuse!(
        cursor
            .read::<OperationHeader>()
            .map_err(|e| format!("command header not readable: {e}"))
    );
    if header.command_type != NEW_RATE_MAP {
        let ct = header.command_type;
        return Ok(Err(format!("unknown command type {ct}")));
    }
    let command = refuse!(
        cursor
            .read::<NewRateMap>()
            .map_err(|e| format!("rate map command not readable: {e}"))
    );
    // The layers go after the map descriptor, on every return.
    let mut layers = Vec::new();
    let descriptor = refuse!(
        new_descriptor(m, &command.descriptor, &mut cursor, &mut layers)?
            .map_err(|e| format!("rate map descriptor not made: {e}"))
    );
    let map = m
        .device()
        .map_or(Ok(None), |d| d.new_rate_map(&descriptor))?;
    drop(descriptor);
    match map {
        Some(map) => Ok(Ok(map)),
        None => Ok(Err("Metal made no rasterization rate map".to_owned())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::manager::icb::testing::*;
    use crate::testkit::calllog::Arg;
    use crate::testkit::capture::catch_death;
    use crate::testkit::{FakePlatform, Fk, Step, kind};

    fn record(w: u16, h: u16, layers: u32, mutability: u32) -> RateMapRecord {
        RateMapRecord {
            screen_width: w,
            screen_height: h,
            layer_count: layers,
            mutability,
            min_factor: 0.5,
        }
    }

    fn floats(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Layer records, then each layer's horizontal and vertical samples.
    fn layers(counts: &[(u16, u16)]) -> Vec<u8> {
        let mut v = Vec::new();
        for &(h, w) in counts {
            v.extend(bytes(&RateMapLayerRecord {
                horizontal_sample_count: h,
                vertical_sample_count: w,
            }));
        }
        for (i, &(h, w)) in counts.iter().enumerate() {
            let base = i as f32 * 10.0;
            v.extend(floats(
                &(0..h).map(|k| base + f32::from(k)).collect::<Vec<_>>(),
            ));
            v.extend(floats(
                &(0..w)
                    .map(|k| base + 5.0 + f32::from(k))
                    .collect::<Vec<_>>(),
            ));
        }
        v
    }

    fn describe(
        o: &FakeOps,
        r: &RateMapRecord,
        data: &[u8],
    ) -> Result<Fk<kind::RateMapDescriptor>, String> {
        let mut c = cursor(data);
        new_descriptor::<FakePlatform, _>(o, r, &mut c, &mut Vec::new()).unwrap()
    }

    fn object(command_type: u32, r: &RateMapRecord, rest: &[u8]) -> Vec<u8> {
        let mut v = header(command_type);
        v.extend(bytes(&NewRateMap {
            field_00: 0,
            descriptor: *r,
        }));
        v.extend(rest);
        v
    }

    fn samples(ptr: HostPtr, n: u64) -> Vec<f32> {
        let mut b = vec![0u8; (n * 4) as usize];
        GuestPtr::new(ptr).copy_to(0, &mut b, DeathKind::SampleStorage);
        b.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    mod descriptor {
        use super::*;

        #[test]
        fn layers_are_set_then_their_samples_read_into_metals_storage() {
            let o = FakeOps::new();
            let d = describe(&o, &record(64, 32, 2, 0), &layers(&[(2, 3), (1, 1)])).unwrap();
            assert_eq!(
                o.log().calls_to("initWithSampleCount:")[0].args,
                vec![Arg::val(&MtlSize {
                    width: 2,
                    height: 3,
                    depth: 0
                })]
            );
            let layer = d.layer(OnFault::Die, 0).unwrap().unwrap();
            assert_eq!(
                samples(layer.horizontal_sample_storage(), 2),
                vec![0.0, 1.0]
            );
            assert_eq!(
                samples(layer.vertical_sample_storage(), 3),
                vec![5.0, 6.0, 7.0]
            );
            let layer = d.layer(OnFault::Die, 1).unwrap().unwrap();
            assert_eq!(samples(layer.horizontal_sample_storage(), 1), vec![10.0]);
            assert_eq!(samples(layer.vertical_sample_storage(), 1), vec![15.0]);
        }

        #[test]
        fn a_short_layer_record_fails_with_its_index() {
            let o = FakeOps::new();
            let data = layers(&[(1, 1)]);
            let e = describe(&o, &record(1, 1, 2, 0), &data[..4]).unwrap_err();
            assert!(e.starts_with("rate map layer 1 not readable: "), "{e}");
        }

        #[test]
        fn a_nil_layer_descriptor_fails() {
            let o = FakeOps::new();
            let e = describe(&o, &record(1, 1, 1, 0), &layers(&[(0, 1)])).unwrap_err();
            assert_eq!(e, "Metal made no descriptor for rate map layer 0");
        }

        /// Both sample arrays are read before either read is checked.
        #[test]
        fn both_sample_arrays_are_read_before_either_is_tested() {
            let o = FakeOps::new();
            let data = layers(&[(2, 2)]);
            let e = describe(&o, &record(1, 1, 1, 0), &data[..4 + 4]).unwrap_err();
            assert!(
                e.starts_with("rate map layer 0's horizontal samples not readable: "),
                "{e}"
            );
            let o = FakeOps::new();
            let e = describe(&o, &record(1, 1, 1, 0), &data[..4 + 8 + 4]).unwrap_err();
            assert!(
                e.starts_with("rate map layer 0's vertical samples not readable: "),
                "{e}"
            );
        }

        #[test]
        fn a_null_storage_is_refused() {
            let o = FakeOps::new();
            o.p.world().on_new::<kind::RateMapLayerDescriptor>(|l| {
                l.set_horizontal_sample_storage(HostPtr::NULL);
            });
            let e = describe(&o, &record(1, 1, 1, 0), &layers(&[(1, 1)])).unwrap_err();
            assert_eq!(
                e,
                "rate map layer 0's horizontal samples not readable: no sample storage to read into"
            );
        }

        #[test]
        fn a_layer_count_mismatch_dies() {
            let o = FakeOps::new();
            o.p.world()
                .on_new::<kind::RateMapDescriptor>(|d| d.set_layer_count(3));
            let data = layers(&[(1, 1)]);
            let d = catch_death(|| describe(&o, &record(1, 1, 1, 0), &data).is_ok()).unwrap_err();
            assert_eq!(d, DeathKind::RateMapLayerCount.site());
        }
    }

    mod creation {
        use super::*;

        fn make(o: &FakeOps, ty: u32, data: &[u8]) -> Result<Fk<kind::RateMap>, String> {
            create::<FakePlatform, _>(o, &entry(ty), cursor(data)).unwrap()
        }

        #[test]
        fn type_header_command_type_and_command_are_checked() {
            let o = FakeOps::new();
            let r = record(1, 1, 0, 0);
            let e = make(&o, 6, &object(50, &r, &[])).unwrap_err();
            assert_eq!(e, "object of type 6 is not a rasterization rate map");
            let e = make(&o, 7, &[0; 4]).unwrap_err();
            assert!(e.starts_with("command header not readable: "), "{e}");
            let e = make(&o, 7, &object(49, &r, &[])).unwrap_err();
            assert_eq!(e, "unknown command type 49");
            let e = make(&o, 7, &object(50, &r, &[])[..8 + 0x13]).unwrap_err();
            assert!(e.starts_with("rate map command not readable: "), "{e}");
            assert!(o.log().is_empty());
        }

        #[test]
        fn a_map_is_made_from_the_descriptor() {
            let o = FakeOps::new();
            let m = make(&o, 7, &object(50, &record(8, 4, 1, 1), &layers(&[(1, 1)]))).unwrap();
            assert_eq!(o.log().count("newRasterizationRateMapWithDescriptor:"), 1);
            assert_eq!(crate::platform::rate_map::RateMap::layer_count(&m), 1);
            assert!(o.calls().is_empty());
        }

        /// A mutable record makes a mutable map; any other value leaves Metal's default.
        #[test]
        fn a_mutable_record_makes_a_mutable_map() {
            use crate::platform::rate_map::RateMap;
            let o = FakeOps::new();
            let m = make(&o, 7, &object(50, &record(8, 4, 0, 1), &[])).unwrap();
            assert_eq!(m.mutability(), 1);
            let m = make(&o, 7, &object(50, &record(8, 4, 0, 2), &[])).unwrap();
            assert_eq!(m.mutability(), 0);
        }

        #[test]
        fn a_descriptor_error_is_wrapped() {
            let o = FakeOps::new();
            let e = make(&o, 7, &object(50, &record(8, 4, 1, 0), &layers(&[(0, 1)]))).unwrap_err();
            assert_eq!(
                e,
                "rate map descriptor not made: Metal made no descriptor for rate map layer 0"
            );
            assert_eq!(o.log().count("newRasterizationRateMapWithDescriptor:"), 0);
        }

        #[test]
        fn metals_nil_fails() {
            let o = FakeOps::new();
            o.p.script()
                .push("newRasterizationRateMapWithDescriptor:", Step::Nil);
            let e = make(&o, 7, &object(50, &record(8, 4, 0, 0), &[])).unwrap_err();
            assert_eq!(e, "Metal made no rasterization rate map");
        }
    }
}
