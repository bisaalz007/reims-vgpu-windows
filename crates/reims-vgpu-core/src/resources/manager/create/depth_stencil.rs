//! Depth-stencil states (object type 7, command 4): the serialized descriptor makes a
//! depth-stencil descriptor with each face it flags, and the placement's index is zeroed.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use super::{STATE_OBJECT_TYPE, Slot};
use crate::guest::GuestPtr;
use crate::guest::records::ObjectEntry;
use crate::memory::cursor::Cursor;
use crate::platform::Platform;
use crate::platform::device::GpuDevice;
use crate::platform::values::{DepthStencilDesc, StencilDesc};
use crate::raise::RaiseResult;
use crate::resources::manager::create::CreateEnv;
use crate::resources::manager::records::OperationHeader;

/// The command type that makes a depth-stencil state from a serialized descriptor.
const NEW_DEPTH_STENCIL_STATE: u32 = 4;

/// The `width` bits of `word` from bit `shift`, zero-extended.
fn bits(word: u32, shift: u32, width: u32) -> u64 {
    u64::from(word.wrapping_shr(shift) & 1_u32.wrapping_shl(width).wrapping_sub(1))
}

/// A serialized stencil face.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct SerializedStencilDescriptor {
    /// `stencilCompareFunction : 3`, `stencilFailureOperation : 3`, `depthFailureOperation : 3`,
    /// `depthStencilPassOperation : 3`, 20 unnamed bits.
    bits: u32,
    read_mask: u32,
    write_mask: u32,
}

/// The serialized depth-stencil descriptor.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct SerializedDepthStencilDescriptor {
    /// `depthCompareFunction : 3`, `depthWriteEnabled : 1`, `hasFrontFaceStencil : 1`,
    /// `hasBackFaceStencil : 1`, 26 unnamed bits.
    bits: u32,
    front_face_stencil: SerializedStencilDescriptor,
    back_face_stencil: SerializedStencilDescriptor,
}

/// The command (0x20 bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct NewDepthStencilState {
    field_00: u32,
    descriptor: SerializedDepthStencilDescriptor,
}

const _: () = assert!(size_of::<NewDepthStencilState>() == 0x20);

/// The depth-stencil descriptor of `s`: its depth properties, then each face `s` flags.
fn descriptor(s: &SerializedDepthStencilDescriptor) -> DepthStencilDesc {
    DepthStencilDesc {
        depth_compare_function: Some(bits(s.bits, 0, 3)),
        depth_write_enabled: Some(bits(s.bits, 3, 1) != 0),
        front: (bits(s.bits, 4, 1) != 0).then(|| face(&s.front_face_stencil)),
        back: (bits(s.bits, 5, 1) != 0).then(|| face(&s.back_face_stencil)),
    }
}

/// A face's six properties.
fn face(s: &SerializedStencilDescriptor) -> StencilDesc {
    StencilDesc {
        stencil_compare_function: bits(s.bits, 0, 3),
        stencil_failure_operation: bits(s.bits, 3, 3),
        depth_failure_operation: bits(s.bits, 6, 3),
        depth_stencil_pass_operation: bits(s.bits, 9, 3),
        read_mask: s.read_mask,
        write_mask: s.write_mask,
    }
}

fn new_state<P: Platform, E: CreateEnv<P>>(
    env: &E,
    cursor: &mut Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<P::DepthStencilState, String>> {
    let command = refuse!(
        cursor
            .read::<NewDepthStencilState>()
            .map_err(|e| format!("depth-stencil command not readable: {e}"))
    );
    let descriptor = descriptor(&command.descriptor);
    let state = env
        .device()
        .map_or(Ok(None), |d| d.new_depth_stencil_state(&descriptor))?;
    let Some(state) = state else {
        return Ok(Err("Metal made no depth-stencil state".to_owned()));
    };
    // A depth-stencil state zeroes the index only.
    if let Some(p) = placement {
        Slot(p).set_resource_index(0);
    }
    Ok(Ok(state))
}

/// A depth-stencil state from `entry` and its record.
pub(crate) fn create<P: Platform, E: CreateEnv<P>>(
    env: &E,
    entry: &ObjectEntry,
    mut cursor: Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<P::DepthStencilState, String>> {
    let object_type = entry.object_type();
    if object_type != STATE_OBJECT_TYPE {
        return Ok(Err(format!(
            "object of type {object_type} is not a depth-stencil state"
        )));
    }
    let header = refuse!(
        cursor
            .read::<OperationHeader>()
            .map_err(|e| format!("command header not readable: {e}"))
    );
    match header.command_type {
        NEW_DEPTH_STENCIL_STATE => new_state(env, &mut cursor, placement),
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

    /// depth compare 5, write enabled, the faces as `faces` says; front
    /// {compare 1, fail 2, depthFail 3, pass 4, 0xf0, 0x0f}, back {7, 6, 5, 1, 0xaa, 0x55}.
    fn command(faces: u32) -> Bytes {
        let front: u32 = 1 | (2 << 3) | (3 << 6) | (4 << 9);
        let back: u32 = 7 | (6 << 3) | (5 << 6) | (1 << 9);
        Bytes::new()
            .header(4)
            .u32(0)
            .u32(5 | (1 << 3) | (faces << 4))
            .u32(front)
            .u32(0xf0)
            .u32(0x0f)
            .u32(back)
            .u32(0xaa)
            .u32(0x55)
    }

    const NEW: &str = "newDepthStencilStateWithDescriptor:";

    const FRONT: StencilDesc = StencilDesc {
        stencil_compare_function: 1,
        stencil_failure_operation: 2,
        depth_failure_operation: 3,
        depth_stencil_pass_operation: 4,
        read_mask: 0xf0,
        write_mask: 0x0f,
    };

    const BACK: StencilDesc = StencilDesc {
        stencil_compare_function: 7,
        stencil_failure_operation: 6,
        depth_failure_operation: 5,
        depth_stencil_pass_operation: 1,
        read_mask: 0xaa,
        write_mask: 0x55,
    };

    /// The creator's value for `command(_)` with these faces.
    fn sent(front: Option<StencilDesc>, back: Option<StencilDesc>) -> Vec<Arg> {
        vec![Arg::val(&DepthStencilDesc {
            depth_compare_function: Some(5),
            depth_write_enabled: Some(true),
            front,
            back,
        })]
    }

    #[test]
    fn the_descriptor_has_the_depth_properties_and_both_faces() {
        let e = TestEnv::new();
        ok(create(&e, &entry(7), command(3).cursor(), None));
        assert_eq!(e.calls(NEW)[0].args, sent(Some(FRONT), Some(BACK)));
    }

    #[test]
    fn a_face_is_sent_only_when_flagged() {
        let e = TestEnv::new();
        ok(create(&e, &entry(7), command(0).cursor(), None));
        assert_eq!(e.calls(NEW)[0].args, sent(None, None));
        ok(create(&e, &entry(7), command(2).cursor(), None));
        assert_eq!(e.calls(NEW)[1].args, sent(None, Some(BACK)));
        ok(create(&e, &entry(7), command(1).cursor(), None));
        assert_eq!(e.calls(NEW)[2].args, sent(Some(FRONT), None));
    }

    /// Depth-stencil zeroes field0 only.
    #[test]
    fn refusals_leave_the_placement_and_a_state_zeroes_its_index() {
        let e = TestEnv::new();
        assert_eq!(
            err(create(&e, &entry(8), command(0).cursor(), None)),
            "object of type 8 is not a depth-stencil state"
        );
        assert!(
            err(create(&e, &entry(7), cur(&[4]), None))
                .starts_with("command header not readable: ")
        );
        assert_eq!(
            err(create(&e, &entry(7), Bytes::new().header(3).cursor(), None)),
            "unknown command type 3"
        );
        assert!(
            err(create(&e, &entry(7), cur(&command(0).0[..0x27]), None))
                .starts_with("depth-stencil command not readable: ")
        );
        let (_m, g) = placement(4, 0x40);
        e.p.script().push(NEW, Step::Nil);
        assert_eq!(
            err(create(&e, &entry(7), command(0).cursor(), Some(g))),
            "Metal made no depth-stencil state"
        );
        assert_eq!(read_placement(g), (4, 0x40));
        ok(create(&e, &entry(7), command(0).cursor(), Some(g)));
        assert_eq!(read_placement(g), (0, 0x40));
    }
}
