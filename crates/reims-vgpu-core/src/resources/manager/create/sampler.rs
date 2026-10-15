//! Sampler states (object type 7, command 3): the serialized descriptor's bit-fields make a
//! sampler descriptor, and the placement's index and ID are set, checked and written back.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use super::{STATE_OBJECT_TYPE, Slot};
use crate::guest::GuestPtr;
use crate::guest::records::ObjectEntry;
use crate::memory::cursor::Cursor;
use crate::platform::Platform;
use crate::platform::device::GpuDevice;
use crate::platform::resource::SamplerState;
use crate::platform::values::SamplerDesc;
use crate::raise::RaiseResult;
use crate::resources::manager::create::CreateEnv;
use crate::resources::manager::records::OperationHeader;

/// The command type that makes a sampler state from a serialized descriptor.
const NEW_SAMPLER_STATE: u32 = 3;

/// The serialized sampler descriptor: two words of bit-fields, then the two clamps. The
/// accessors read the bit-fields.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct SerializedSamplerDescriptor {
    /// `minFilter : 2` … `normalizedCoordinates : 1` (bits 0-31).
    bits0: u32,
    /// `supportArgumentBuffers : 1`, `reductionMode : 3` (bits 0-3).
    bits1: u32,
    lod_min_clamp: f32,
    lod_max_clamp: f32,
}

/// The `width` bits of `word` from bit `shift`, zero-extended.
fn bits(word: u32, shift: u32, width: u32) -> u64 {
    u64::from(word.wrapping_shr(shift) & 1_u32.wrapping_shl(width).wrapping_sub(1))
}

impl SerializedSamplerDescriptor {
    fn min_filter(&self) -> u64 {
        bits(self.bits0, 0, 2)
    }
    fn mag_filter(&self) -> u64 {
        bits(self.bits0, 2, 2)
    }
    fn lod_average(&self) -> bool {
        bits(self.bits0, 4, 1) != 0
    }
    fn compare_function(&self) -> u64 {
        bits(self.bits0, 5, 3)
    }
    fn s_address_mode(&self) -> u64 {
        bits(self.bits0, 8, 4)
    }
    fn t_address_mode(&self) -> u64 {
        bits(self.bits0, 12, 4)
    }
    fn r_address_mode(&self) -> u64 {
        bits(self.bits0, 16, 4)
    }
    fn border_color(&self) -> u64 {
        bits(self.bits0, 20, 4)
    }
    fn mip_filter(&self) -> u64 {
        bits(self.bits0, 24, 2)
    }
    fn max_anisotropy(&self) -> u64 {
        bits(self.bits0, 26, 5)
    }
    fn normalized_coordinates(&self) -> bool {
        bits(self.bits0, 31, 1) != 0
    }
    fn support_argument_buffers(&self) -> bool {
        bits(self.bits1, 0, 1) != 0
    }
    fn reduction_mode(&self) -> u64 {
        bits(self.bits1, 1, 3)
    }
}

/// The command (0x1c bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct NewSamplerState {
    field_00: u32,
    descriptor: SerializedSamplerDescriptor,
    field_14: [u8; 8],
}

const _: () = assert!(size_of::<NewSamplerState>() == 0x1c);

/// The sampler descriptor of `s`: every property set, the resource index when given.
fn descriptor(s: &SerializedSamplerDescriptor, resource_index: Option<u64>) -> SamplerDesc {
    SamplerDesc {
        min_filter: Some(s.min_filter()),
        mag_filter: Some(s.mag_filter()),
        lod_average: Some(s.lod_average()),
        compare_function: Some(s.compare_function()),
        s_address_mode: Some(s.s_address_mode()),
        t_address_mode: Some(s.t_address_mode()),
        r_address_mode: Some(s.r_address_mode()),
        mip_filter: Some(s.mip_filter()),
        max_anisotropy: Some(s.max_anisotropy()),
        normalized_coordinates: Some(s.normalized_coordinates()),
        support_argument_buffers: Some(s.support_argument_buffers()),
        reduction_mode: Some(s.reduction_mode()),
        lod_min_clamp: Some(s.lod_min_clamp),
        lod_max_clamp: Some(s.lod_max_clamp),
        border_color: Some(s.border_color()),
        resource_index,
    }
}

fn new_state<P: Platform, E: CreateEnv<P>>(
    env: &E,
    cursor: &mut Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<P::SamplerState, String>> {
    let command = refuse!(
        cursor
            .read::<NewSamplerState>()
            .map_err(|e| format!("sampler command not readable: {e}"))
    );
    let slot = placement.map(Slot);
    let (resource_index, resource_id) =
        slot.map_or((0, 0), |s| (s.resource_index(), s.resource_id()));
    // The index is set (not forced) when the ID is set, and the ID checked when the index is set.
    let descriptor = descriptor(
        &command.descriptor,
        (resource_id != 0).then_some(resource_index),
    );
    let state = env
        .device()
        .map_or(Ok(None), |d| d.new_sampler_state(&descriptor))?;
    let Some(state) = state else {
        return Ok(Err("Metal made no sampler state".to_owned()));
    };
    if resource_index != 0 && state.gpu_resource_id() != resource_id {
        // Metal is asked for the ID a second time, for the line, whether or not the line is
        // written.
        let got = state.gpu_resource_id();
        log::warn!(
            "sampler state placed at resource ID {got:#x}, not the guest's {resource_id:#x}"
        );
        drop(state);
        return Ok(Err(
            "sampler state not placed where the guest asked".to_owned()
        ));
    }
    // Written back only with argument-buffer support.
    if let Some(slot) = slot
        && command.descriptor.support_argument_buffers()
    {
        slot.set_resource_index(state.resource_index());
        slot.set_resource_id(state.gpu_resource_id());
    }
    Ok(Ok(state))
}

/// A sampler state from `entry` and its record.
pub(crate) fn create<P: Platform, E: CreateEnv<P>>(
    env: &E,
    entry: &ObjectEntry,
    mut cursor: Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<P::SamplerState, String>> {
    let object_type = entry.object_type();
    if object_type != STATE_OBJECT_TYPE {
        return Ok(Err(format!(
            "object of type {object_type} is not a sampler state"
        )));
    }
    let header = refuse!(
        cursor
            .read::<OperationHeader>()
            .map_err(|e| format!("command header not readable: {e}"))
    );
    match header.command_type {
        NEW_SAMPLER_STATE => new_state(env, &mut cursor, placement),
        ct => Ok(Err(format!("unknown command type {ct}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::manager::create::test_env::*;
    use crate::testkit::{Arg, Step};

    fn ok<T: std::fmt::Debug>(r: RaiseResult<Result<T, String>>) -> T {
        r.expect("no raise").expect("created")
    }

    fn err<T: std::fmt::Debug>(r: RaiseResult<Result<T, String>>) -> String {
        r.expect("no raise").expect_err("refused")
    }

    /// minFilter 1, magFilter 1, lodAverage, compare 7, s 2, t 3, r 4, border 2, mip 2,
    /// anisotropy 16, normalized; then supportArgumentBuffers and reduction 2.
    fn command(sab: bool) -> Bytes {
        let w0: u32 = 1
            | (1 << 2)
            | (1 << 4)
            | (7 << 5)
            | (2 << 8)
            | (3 << 12)
            | (4 << 16)
            | (2 << 20)
            | (2 << 24)
            | (16 << 26)
            | (1 << 31);
        let w1: u32 = u32::from(sab) | (2 << 1);
        Bytes::new()
            .header(3)
            .u32(0)
            .u32(w0)
            .u32(w1)
            .f32(0.5)
            .f32(8.0)
            .zeros(8)
    }

    /// The creator's value for `command(sab)`, with `resource_index`.
    fn sent(sab: bool, resource_index: Option<u64>) -> Arg {
        Arg::val(&SamplerDesc {
            min_filter: Some(1),
            mag_filter: Some(1),
            lod_average: Some(true),
            compare_function: Some(7),
            s_address_mode: Some(2),
            t_address_mode: Some(3),
            r_address_mode: Some(4),
            mip_filter: Some(2),
            max_anisotropy: Some(16),
            normalized_coordinates: Some(true),
            support_argument_buffers: Some(sab),
            reduction_mode: Some(2),
            lod_min_clamp: Some(0.5),
            lod_max_clamp: Some(8.0),
            border_color: Some(2),
            resource_index,
        })
    }

    /// The creator's value of its `n`th call.
    fn value(e: &TestEnv, n: usize) -> Arg {
        e.calls(NEW)[n].args[0].clone()
    }

    const NEW: &str = "newSamplerStateWithDescriptor:";

    #[test]
    fn the_descriptor_has_every_property_of_the_record() {
        let e = TestEnv::new();
        ok(create(&e, &entry(7), command(false).cursor(), None));
        assert_eq!(e.calls(NEW)[0].args, vec![sent(false, None)]);
    }

    #[test]
    fn refusals() {
        let e = TestEnv::new();
        assert_eq!(
            err(create(&e, &entry(1), command(false).cursor(), None)),
            "object of type 1 is not a sampler state"
        );
        assert!(
            err(create(&e, &entry(7), cur(&[3, 0, 0]), None))
                .starts_with("command header not readable: ")
        );
        assert_eq!(
            err(create(&e, &entry(7), Bytes::new().header(4).cursor(), None)),
            "unknown command type 4"
        );
        assert!(
            err(create(
                &e,
                &entry(7),
                cur(&command(false).0[..8 + 0x1b]),
                None
            ))
            .starts_with("sampler command not readable: ")
        );
        assert!(e.calls(NEW).is_empty());
        e.p.script().push(NEW, Step::Nil);
        assert_eq!(
            err(create(&e, &entry(7), command(false).cursor(), None)),
            "Metal made no sampler state"
        );
    }

    /// Makes the next sampler state Metal makes answer `gpuResourceID` with `id`.
    fn next_sampler_id(e: &TestEnv, id: u64) {
        e.p.world()
            .on_new::<crate::testkit::kind::SamplerState>(move |s| s.set_gpu_resource_id(id));
    }

    /// The index is set (not forced) when the ID is set, and verified when the index is set.
    /// Written back only with supportArgumentBuffers.
    #[test]
    fn placement_is_set_checked_and_written_back() {
        let e = TestEnv::new();
        // Index and ID: set, verified, written back.
        let (_m, g) = placement(5, 0x50);
        next_sampler_id(&e, 0x50);
        let s = ok(create(&e, &entry(7), command(true).cursor(), Some(g)));
        assert_eq!(value(&e, 0), sent(true, Some(5)));
        assert_eq!(read_placement(g), (s.resource_index(), s.gpu_resource_id()));
        // No supportArgumentBuffers: no write-back.
        let (_m, g) = placement(5, 0x50);
        next_sampler_id(&e, 0x50);
        ok(create(&e, &entry(7), command(false).cursor(), Some(g)));
        assert_eq!(read_placement(g), (5, 0x50));
        // ID set, index 0: index 0 is set, nothing verified.
        let (_m, g) = placement(0, 0x50);
        next_sampler_id(&e, 0x99);
        ok(create(&e, &entry(7), command(false).cursor(), Some(g)));
        assert_eq!(value(&e, 2), sent(false, Some(0)));
        // Index set, ID 0: not set, but verified against 0.
        let (_m, g) = placement(5, 0);
        next_sampler_id(&e, 0x7);
        let (r, lines) = logged(|| create(&e, &entry(7), command(true).cursor(), Some(g)));
        assert_eq!(err(r), "sampler state not placed where the guest asked");
        assert_eq!(value(&e, 3), sent(true, None));
        assert_eq!(read_placement(g), (5, 0));
        assert_eq!(
            lines,
            vec!["sampler state placed at resource ID 0x7, not the guest's 0x0"]
        );
    }
}
