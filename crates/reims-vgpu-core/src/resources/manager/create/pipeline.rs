//! Compute and render pipeline states (object type 7): commands 11 (compute), 14 (render) and
//! 51 (tile render). Each reads a serialized descriptor that Metal deserializes, calling the
//! manager's function resolver back for each function it names; the state is registered under
//! Metal's `uniqueIdentifier` for it (`every_form_registers_by_unique_identifier`). An indirect
//! command answers that identifier for the pipeline it was encoded with, and the ICB synchronize
//! walk writes back the reference registered under it, or 0 (`icb::synchronize`).

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use super::{STATE_OBJECT_TYPE, Slot};
use crate::guest::GuestPtr;
use crate::guest::records::ObjectEntry;
use crate::memory::cursor::Cursor;
use crate::num::ToUsize;
use crate::platform::Platform;
use crate::platform::descriptor::{ComputePipelineDescriptor, RenderPipelineDescriptor};
use crate::platform::device::GpuDevice;
use crate::platform::resource::{ComputePipeline, RenderPipeline};
use crate::raise::RaiseResult;
use crate::resources::manager::DeathKind;
use crate::resources::manager::icb::ManagerOps;
use crate::resources::manager::records::OperationHeader;
use crate::rt::Hold;

/// The command types.
const NEW_COMPUTE_PIPELINE_STATE: u32 = 11;
const NEW_RENDER_PIPELINE_STATE: u32 = 14;
const NEW_TILE_RENDER_PIPELINE_STATE: u32 = 51;

/// The command every pipeline creation opens with (8 bytes); the serialized descriptor follows.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub(crate) struct NewPipelineState {
    /// Nothing reads it.
    pub field_00: u32,
    /// The descriptor's length. // name is ours: the length the descriptor's read takes.
    pub descriptor_length: u32,
}

const _: () = assert!(size_of::<NewPipelineState>() == 8);

/// The command, the length check and the descriptor's bytes, as every form reads them first:
/// the `descriptor_length` bytes read, and the bytes the cursor had left after the command.
fn read_descriptor(cursor: &mut Cursor) -> Result<(Vec<u8>, u64), String> {
    let command = cursor
        .read::<NewPipelineState>()
        .map_err(|e| format!("pipeline command not readable: {e}"))?;
    let remaining = cursor.remaining();
    let length = u64::from(command.descriptor_length);
    if remaining < length {
        return Err(format!(
            "pipeline descriptor of {length:#x} bytes, more than the {remaining:#x} left"
        ));
    }
    let mut descriptor = vec![0u8; length.to_usize()];
    cursor
        .read_bytes(&mut descriptor)
        .map_err(|e| format!("pipeline descriptor not readable: {e}"))?;
    Ok((descriptor, remaining))
}

/// The placement's index, forced on the descriptor when the placement names an ID.
///
/// Compute and render force the index when the ID is set, and never check the ID Metal gives. The
/// index is read only then, the ID first.
fn forced_index(placement: Option<GuestPtr>) -> Option<u64> {
    let slot = Slot(placement?);
    (slot.resource_id() != 0).then(|| slot.resource_index())
}

/// Writes the state's index, then its ID, back to the placement.
fn write_back(placement: Option<GuestPtr>, index: impl FnOnce() -> u64, id: impl FnOnce() -> u64) {
    if let Some(p) = placement {
        let slot = Slot(p);
        slot.set_resource_index(index());
        slot.set_resource_id(id());
    }
}

fn new_compute_state<P: Platform, M: ManagerOps<P>>(
    hold: Hold<'_>,
    m: &M,
    cursor: &mut Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<P::ComputePipeline, String>> {
    let (mut descriptor, remaining) = refuse!(read_descriptor(cursor));
    // Metal gets the remaining bytes, of which only the descriptor's length were read (the rest
    // zero).
    descriptor.resize(remaining.to_usize(), 0);
    // Metal calls the resolver back, on this thread, while it deserializes
    let pipeline_descriptor = m.device().map_or(Ok(None), |d| {
        d.deserialize_compute_pipeline_descriptor(hold, &descriptor, &mut |hold, r| {
            m.function_with_function_ref(hold, r)
        })
    })?;
    if let Some(index) = forced_index(placement)
        && let Some(d) = &pipeline_descriptor
    {
        d.set_resource_index(index, true);
    }
    // A nil descriptor (an empty blob) goes to Metal as it is, which answers nil with an error
    let created = m.device().map_or(Ok(Ok(None)), |d| {
        d.new_compute_pipeline_state(pipeline_descriptor.as_ref(), 0)
    })?;
    let state = match created {
        Ok(Some(state)) => state,
        Ok(None) => {
            log::error!("Metal made no compute pipeline, and gave no error");
            crate::fatal!(DeathKind::NilPipelineWithoutError)
        }
        Err(error) => {
            log::warn!("Metal refused a compute pipeline: {error}");
            return Ok(Err(format!(
                "Metal refused a compute pipeline: {}",
                error.description
            )));
        }
    };
    // Always written back, index then ID.
    write_back(
        placement,
        || state.resource_index(),
        || state.gpu_resource_id(),
    );
    Ok(Ok(state))
}

fn new_render_state<P: Platform, M: ManagerOps<P>>(
    hold: Hold<'_>,
    m: &M,
    cursor: &mut Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<P::RenderPipeline, String>> {
    let (mut descriptor, remaining) = refuse!(read_descriptor(cursor));
    // As the compute form.
    descriptor.resize(remaining.to_usize(), 0);
    let pipeline_descriptor = m.device().map_or(Ok(None), |d| {
        d.deserialize_render_pipeline_descriptor(hold, &descriptor, &mut |hold, r| {
            m.function_with_function_ref(hold, r)
        })
    })?;
    if let Some(index) = forced_index(placement)
        && let Some(d) = &pipeline_descriptor
    {
        d.set_resource_index(index, true);
    }
    let created = m.device().map_or(Ok(Ok(None)), |d| {
        d.new_render_pipeline_state_from_descriptor(pipeline_descriptor.as_ref(), 0)
    })?;
    // No line on this form
    let state = match created {
        Ok(Some(state)) => state,
        Ok(None) => crate::fatal!(DeathKind::NilPipelineWithoutError),
        Err(error) => {
            return Ok(Err(format!(
                "Metal refused a render pipeline: {}",
                error.description
            )));
        }
    };
    // Always written back, index then ID.
    write_back(
        placement,
        || state.resource_index(),
        || state.gpu_resource_id(),
    );
    Ok(Ok(state))
}

/// The tile form takes no placement: nothing is forced and nothing written back.
fn new_tile_state<P: Platform, M: ManagerOps<P>>(
    hold: Hold<'_>,
    m: &M,
    cursor: &mut Cursor,
) -> RaiseResult<Result<P::RenderPipeline, String>> {
    // Metal gets the descriptor's length here, in a buffer of the remaining bytes that the platform
    // leaks whole.
    let (descriptor, remaining) = refuse!(read_descriptor(cursor));
    let pipeline_descriptor = m.device().map_or(Ok(None), |d| {
        d.deserialize_tile_render_pipeline_descriptor(
            hold,
            &descriptor,
            remaining,
            &mut |hold, r| m.function_with_function_ref(hold, r),
        )
    })?;
    // A nil descriptor goes to Metal as it is, as the render form's
    let created = m.device().map_or(Ok(Ok(None)), |d| {
        d.new_tile_render_pipeline_state(pipeline_descriptor.as_ref(), 0)
    })?;
    match created {
        Ok(Some(state)) => Ok(Ok(state)),
        Ok(None) => {
            log::error!("Metal made no tile render pipeline, and gave no error");
            crate::fatal!(DeathKind::NilPipelineWithoutError)
        }
        Err(error) => {
            log::warn!("Metal refused a tile render pipeline: {error}");
            Ok(Err(format!(
                "Metal refused a tile render pipeline: {}",
                error.description
            )))
        }
    }
}

/// Checks `entry`'s object type and reads the operation header: the command type.
fn command_type(entry: &ObjectEntry, cursor: &mut Cursor, what: &str) -> Result<u32, String> {
    let object_type = entry.object_type();
    if object_type != STATE_OBJECT_TYPE {
        return Err(format!("object of type {object_type} is not a {what}"));
    }
    let header = cursor
        .read::<OperationHeader>()
        .map_err(|e| format!("command header not readable: {e}"))?;
    Ok(header.command_type)
}

/// A compute pipeline state from `entry` and its record, registered for `reference`.
pub(crate) fn create_compute<P: Platform, M: ManagerOps<P>>(
    hold: Hold<'_>,
    m: &M,
    reference: u32,
    entry: &ObjectEntry,
    mut cursor: Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<P::ComputePipeline, String>> {
    let state = match refuse!(command_type(entry, &mut cursor, "compute pipeline state")) {
        NEW_COMPUTE_PIPELINE_STATE => refuse!(new_compute_state(hold, m, &mut cursor, placement)?),
        ct => return Ok(Err(format!("unknown command type {ct}"))),
    };
    // In every setup; the registration never fails
    if let Err(e) = m.register_pipeline(state.unique_identifier(), reference) {
        drop(state);
        return Ok(Err(e));
    }
    Ok(Ok(state))
}

/// A render pipeline state (render or tile) from `entry` and its record, registered for
/// `reference`.
pub(crate) fn create_render<P: Platform, M: ManagerOps<P>>(
    hold: Hold<'_>,
    m: &M,
    reference: u32,
    entry: &ObjectEntry,
    mut cursor: Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<P::RenderPipeline, String>> {
    let state = match refuse!(command_type(entry, &mut cursor, "render pipeline state")) {
        NEW_RENDER_PIPELINE_STATE => refuse!(new_render_state(hold, m, &mut cursor, placement)?),
        NEW_TILE_RENDER_PIPELINE_STATE => refuse!(new_tile_state(hold, m, &mut cursor)?),
        ct => return Ok(Err(format!("unknown command type {ct}"))),
    };
    if let Err(e) = m.register_pipeline(state.unique_identifier(), reference) {
        drop(state);
        return Ok(Err(e));
    }
    Ok(Ok(state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::Handle;
    use crate::platform::types::NsErrorText;
    use crate::resources::manager::icb::testing::*;
    use crate::testkit::calllog::Arg;
    use crate::testkit::capture::catch_death;
    use crate::testkit::impls::{NIL_DESCRIPTOR_ERROR, ResolveIds};
    use crate::testkit::{FakePlatform, Fk, Step, kind};

    const DESERIALIZE_COMPUTE: &str =
        "newComputePipelineDescriptorWithSerializedData:deserializationContext:";
    const NEW_COMPUTE: &str = "newComputePipelineStateWithDescriptor:options:reflection:error:";
    const DESERIALIZE_RENDER: &str =
        "newRenderPipelineDescriptorWithSerializedData:deserializationContext:";
    const DESERIALIZE_TILE: &str =
        "newTileRenderPipelineDescriptorWithSerializedData:deserializationContext:";
    const NEW_RENDER: &str = "newRenderPipelineStateWithDescriptor:options:reflection:error:";
    const NEW_TILE: &str = "newRenderPipelineStateWithTileDescriptor:options:reflection:error:";

    /// The header, the command and `descriptor`, then `tail` extra bytes.
    fn object(command_type: u32, length: u32, descriptor: &[u8], tail: usize) -> Vec<u8> {
        let mut v = header(command_type);
        v.extend(bytes(&NewPipelineState {
            field_00: 0,
            descriptor_length: length,
        }));
        v.extend(descriptor);
        v.extend(vec![0xaa; tail]);
        v
    }

    fn compute(
        o: &FakeOps,
        ty: u32,
        data: &[u8],
        placement: Option<GuestPtr>,
    ) -> Result<Fk<kind::ComputePipeline>, String> {
        create_compute::<FakePlatform, _>(Hold::NONE, o, 3, &entry(ty), cursor(data), placement)
            .unwrap()
    }

    fn render(
        o: &FakeOps,
        ty: u32,
        data: &[u8],
        placement: Option<GuestPtr>,
    ) -> Result<Fk<kind::RenderPipeline>, String> {
        create_render::<FakePlatform, _>(Hold::NONE, o, 3, &entry(ty), cursor(data), placement)
            .unwrap()
    }

    fn error(description: &str) -> Step {
        Step::Error(NsErrorText {
            domain: "MTLLibraryErrorDomain".to_owned(),
            code: 2,
            description: description.to_owned(),
        })
    }

    mod create {
        use super::*;

        #[test]
        fn type_header_and_command_type_are_checked() {
            let o = FakeOps::new();
            let e = compute(&o, 6, &object(11, 0, &[], 0), None).unwrap_err();
            assert_eq!(e, "object of type 6 is not a compute pipeline state");
            let e = compute(&o, 7, &[0; 4], None).unwrap_err();
            assert!(e.starts_with("command header not readable: "), "{e}");
            let e = compute(&o, 7, &object(14, 0, &[], 0), None).unwrap_err();
            assert_eq!(e, "unknown command type 14");
            let e = render(&o, 6, &object(14, 0, &[], 0), None).unwrap_err();
            assert_eq!(e, "object of type 6 is not a render pipeline state");
            let e = render(&o, 7, &[0; 4], None).unwrap_err();
            assert!(e.starts_with("command header not readable: "), "{e}");
            let e = render(&o, 7, &object(11, 0, &[], 0), None).unwrap_err();
            assert_eq!(e, "unknown command type 11");
            assert!(o.log().is_empty());
        }

        #[test]
        fn every_form_registers_by_unique_identifier() {
            let o = FakeOps::new();
            let s = compute(&o, 7, &object(11, 2, &[1, 2], 0), None).unwrap();
            assert_eq!(
                o.calls(),
                vec![format!("register_pipeline({}, 3)", s.unique_id_value())]
            );
            for ct in [14, 51] {
                let o = FakeOps::new();
                let s = render(&o, 7, &object(ct, 1, &[5], 0), None).unwrap();
                assert_eq!(
                    o.calls(),
                    vec![format!("register_pipeline({}, 3)", s.unique_id_value())]
                );
            }
        }

        #[test]
        fn a_registration_error_fails() {
            let mut o = FakeOps::new();
            o.register = Err("Duplicate pipeline".to_owned());
            let e = compute(&o, 7, &object(11, 2, &[1, 2], 0), None).unwrap_err();
            assert_eq!(e, "Duplicate pipeline");
            let e = render(&o, 7, &object(14, 0, &[], 0), None).unwrap_err();
            assert_eq!(e, "Duplicate pipeline");
        }

        #[test]
        fn short_commands_fail() {
            let o = FakeOps::new();
            let e = compute(&o, 7, &object(11, 0, &[], 0)[..8 + 7], None).unwrap_err();
            assert!(e.starts_with("pipeline command not readable: "), "{e}");
            let e = compute(&o, 7, &object(11, 4, &[1, 2, 3], 0), None).unwrap_err();
            assert_eq!(
                e,
                "pipeline descriptor of 0x4 bytes, more than the 0x3 left"
            );
            let e = render(&o, 7, &header(14), None).unwrap_err();
            assert!(e.starts_with("pipeline command not readable: "), "{e}");
            let e = render(&o, 7, &object(51, 4, &[1], 0), None).unwrap_err();
            assert_eq!(
                e,
                "pipeline descriptor of 0x4 bytes, more than the 0x1 left"
            );
            assert!(o.log().is_empty());
        }
    }

    mod compute_state {
        use super::*;

        fn run(
            o: &FakeOps,
            data: &[u8],
            placement: Option<GuestPtr>,
        ) -> Result<Fk<kind::ComputePipeline>, String> {
            let mut c = cursor(&data[8..]);
            new_compute_state::<FakePlatform, _>(Hold::NONE, o, &mut c, placement).unwrap()
        }

        /// Metal gets the remaining bytes, zero past the descriptor's length.
        #[test]
        fn metal_gets_the_remaining_bytes_zero_padded() {
            let o = FakeOps::new();
            run(&o, &object(11, 2, &[1, 2], 3), None).unwrap();
            let args = o.log().calls_to(DESERIALIZE_COMPUTE)[0].args.clone();
            assert_eq!(args[0], Arg::Bytes(vec![1, 2, 0, 0, 0]));
        }

        #[test]
        fn metal_resolves_functions_through_the_manager() {
            let mut o = FakeOps::new();
            let f = o.p.world().new_obj::<kind::Function>(Default::default());
            o.functions.push((4, f.clone_ref()));
            o.p.script()
                .push(DESERIALIZE_COMPUTE, Step::value(ResolveIds(vec![4, 5])));
            run(&o, &object(11, 0, &[], 0), None).unwrap();
            assert_eq!(
                o.calls(),
                vec![
                    "function_with_function_ref(4)",
                    "function_with_function_ref(5)"
                ]
            );
            assert_eq!(
                o.log().calls_to("deserializationContext")[0].args,
                vec![Arg::Obj(Some(f.id())), Arg::Obj(None)]
            );
        }

        /// A raise inside the resolver leaves Metal's deserialization and the creation; no
        /// state is made.
        #[test]
        fn a_raise_inside_the_resolver_unwinds_out_of_the_creation() {
            let mut o = FakeOps::new();
            o.function_throws.push(4);
            o.p.script()
                .push(DESERIALIZE_COMPUTE, Step::value(ResolveIds(vec![4, 5])));
            let data = object(11, 0, &[], 0);
            let hold = Hold::NONE;
            let mut c = cursor(&data[8..]);
            let t = new_compute_state::<FakePlatform, _>(hold, &o, &mut c, None)
                .err()
                .unwrap();
            assert_eq!(t.detail().unwrap_or_default(), "function lookup");
            assert_eq!(o.calls(), vec!["function_with_function_ref(4)"]);
            assert_eq!(o.log().count(NEW_COMPUTE), 0);
        }

        /// A nonzero ID forces the index, read again from the guest; nothing is checked, and index
        /// then ID are written back.
        #[test]
        fn a_nonzero_id_forces_the_reread_index_and_both_are_written_back() {
            let o = FakeOps::new();
            let (mem, p) = placement(12, 0x99);
            let s = run(&o, &object(11, 0, &[], 0), Some(p)).unwrap();
            assert_eq!(s.resource_index_value(), 12);
            assert_eq!(mem.read_u64(0), Some(12));
            assert_eq!(mem.read_u64(8), Some(s.gpu_resource_id_value()));
        }

        #[test]
        fn a_zero_id_forces_nothing_but_is_written_back() {
            let o = FakeOps::new();
            let (mem, p) = placement(12, 0);
            let s = run(&o, &object(11, 0, &[], 0), Some(p)).unwrap();
            assert_ne!(s.resource_index_value(), 12);
            assert_eq!(mem.read_u64(0), Some(s.resource_index_value()));
            assert_eq!(mem.read_u64(8), Some(s.gpu_resource_id_value()));
        }

        #[test]
        fn an_error_is_logged_and_returned() {
            let o = FakeOps::new();
            o.p.script().push(NEW_COMPUTE, error("no function"));
            let (r, lines) = logged(|| run(&o, &object(11, 0, &[], 0), None));
            assert_eq!(
                r.unwrap_err(),
                "Metal refused a compute pipeline: no function"
            );
            assert_eq!(lines, vec!["Metal refused a compute pipeline: no function"]);
        }

        #[test]
        fn nil_without_an_error_logs_and_dies() {
            let o = FakeOps::new();
            o.p.script().push(NEW_COMPUTE, Step::Nil);
            let data = object(11, 0, &[], 0);
            let (d, lines) = logged(|| catch_death(|| run(&o, &data, None).is_ok()));
            assert_eq!(d.unwrap_err(), DeathKind::NilPipelineWithoutError.site());
            assert_eq!(
                lines,
                vec!["Metal made no compute pipeline, and gave no error"]
            );
        }

        /// A nil descriptor goes to Metal unchecked, forces nothing, and Metal's error is
        /// logged and returned.
        #[test]
        fn a_nil_descriptor_is_sent_to_metal_and_its_error_logged_and_returned() {
            let o = FakeOps::new();
            o.p.script().push(DESERIALIZE_COMPUTE, Step::Nil);
            let (_mem, p) = placement(1, 1);
            let (r, lines) = logged(|| run(&o, &object(11, 0, &[], 0), Some(p)));
            assert_eq!(
                r.unwrap_err(),
                format!("Metal refused a compute pipeline: {NIL_DESCRIPTOR_ERROR}")
            );
            assert_eq!(o.log().count(NEW_COMPUTE), 1);
            assert_eq!(o.log().calls_to(NEW_COMPUTE)[0].args[0], Arg::Obj(None));
            assert_eq!(
                lines,
                vec![format!(
                    "Metal refused a compute pipeline: {NIL_DESCRIPTOR_ERROR}"
                )]
            );
        }
    }

    mod render_state {
        use super::*;

        fn run(
            o: &FakeOps,
            data: &[u8],
            placement: Option<GuestPtr>,
        ) -> Result<Fk<kind::RenderPipeline>, String> {
            let mut c = cursor(&data[8..]);
            new_render_state::<FakePlatform, _>(Hold::NONE, o, &mut c, placement).unwrap()
        }

        /// Metal gets the remaining bytes, zero past the descriptor's length.
        #[test]
        fn metal_gets_the_remaining_bytes_zero_padded_and_options_zero() {
            let o = FakeOps::new();
            run(&o, &object(14, 2, &[1, 2], 2), None).unwrap();
            assert_eq!(
                o.log().calls_to(DESERIALIZE_RENDER)[0].args[0],
                Arg::Bytes(vec![1, 2, 0, 0])
            );
            assert_eq!(o.log().calls_to(NEW_RENDER)[0].args[1], Arg::U64(0));
        }

        /// A nil descriptor goes to Metal unchecked, and its error is returned with no line.
        #[test]
        fn a_nil_descriptor_is_sent_to_metal_and_its_error_returned_unlogged() {
            let o = FakeOps::new();
            o.p.script().push(DESERIALIZE_RENDER, Step::Nil);
            let (_mem, p) = placement(1, 1);
            let (r, lines) = logged(|| run(&o, &object(14, 0, &[], 0), Some(p)));
            assert_eq!(
                r.unwrap_err(),
                format!("Metal refused a render pipeline: {NIL_DESCRIPTOR_ERROR}")
            );
            assert_eq!(o.log().calls_to(NEW_RENDER)[0].args[0], Arg::Obj(None));
            assert!(lines.is_empty());
        }

        /// As the compute form.
        #[test]
        fn placement_forces_on_a_nonzero_id_and_writes_back_index_then_id() {
            let o = FakeOps::new();
            let (mem, p) = placement(21, 0x5);
            let s = run(&o, &object(14, 0, &[], 0), Some(p)).unwrap();
            assert_eq!(s.resource_index_value(), 21);
            assert_eq!(mem.read_u64(0), Some(21));
            assert_eq!(mem.read_u64(8), Some(s.gpu_resource_id_value()));
        }

        #[test]
        fn a_zero_id_forces_nothing() {
            let o = FakeOps::new();
            let (_mem, p) = placement(21, 0);
            let s = run(&o, &object(14, 0, &[], 0), Some(p)).unwrap();
            assert_ne!(s.resource_index_value(), 21);
        }

        #[test]
        fn an_error_is_returned_without_a_line() {
            let o = FakeOps::new();
            o.p.script().push(NEW_RENDER, error("bad"));
            let (r, lines) = logged(|| run(&o, &object(14, 0, &[], 0), None));
            assert_eq!(r.unwrap_err(), "Metal refused a render pipeline: bad");
            assert!(lines.is_empty());
        }

        #[test]
        fn nil_without_an_error_dies_without_a_line() {
            let o = FakeOps::new();
            o.p.script().push(NEW_RENDER, Step::Nil);
            let data = object(14, 0, &[], 0);
            let (d, lines) = logged(|| catch_death(|| run(&o, &data, None).is_ok()));
            assert_eq!(d.unwrap_err(), DeathKind::NilPipelineWithoutError.site());
            assert!(lines.is_empty());
        }
    }

    mod tile_state {
        use super::*;

        fn run(o: &FakeOps, data: &[u8]) -> Result<Fk<kind::RenderPipeline>, String> {
            let mut c = cursor(&data[8..]);
            new_tile_state::<FakePlatform, _>(Hold::NONE, o, &mut c).unwrap()
        }

        /// Metal gets the descriptor's length, in a buffer of the remaining bytes, and no placement
        /// is used.
        #[test]
        fn metal_gets_exactly_the_descriptor_length_and_no_placement_is_used() {
            let o = FakeOps::new();
            let (mem, p) = placement(21, 0x5);
            render(&o, 7, &object(51, 2, &[1, 2], 3), Some(p)).unwrap();
            assert_eq!(
                o.log().calls_to(DESERIALIZE_TILE)[0].args[0],
                Arg::Bytes(vec![1, 2])
            );
            assert_eq!(o.log().calls_to(DESERIALIZE_TILE)[0].args[2], Arg::U64(5));
            assert_eq!(o.log().calls_to(NEW_TILE)[0].args[1], Arg::U64(0));
            assert_eq!(mem.read_u64(0), Some(21));
            assert_eq!(mem.read_u64(8), Some(5));
        }

        #[test]
        fn an_error_is_logged_and_returned() {
            let o = FakeOps::new();
            o.p.script().push(NEW_TILE, error("tile"));
            let (r, lines) = logged(|| run(&o, &object(51, 0, &[], 0)));
            assert_eq!(r.unwrap_err(), "Metal refused a tile render pipeline: tile");
            assert_eq!(lines, vec!["Metal refused a tile render pipeline: tile"]);
        }

        #[test]
        fn nil_without_an_error_logs_and_dies() {
            let o = FakeOps::new();
            o.p.script().push(NEW_TILE, Step::Nil);
            let data = object(51, 0, &[], 0);
            let (d, lines) = logged(|| catch_death(|| run(&o, &data).is_ok()));
            assert_eq!(d.unwrap_err(), DeathKind::NilPipelineWithoutError.site());
            assert_eq!(
                lines,
                vec!["Metal made no tile render pipeline, and gave no error"]
            );
        }

        #[test]
        fn a_nil_descriptor_is_sent_to_metal_and_its_error_logged_and_returned() {
            let o = FakeOps::new();
            o.p.script().push(DESERIALIZE_TILE, Step::Nil);
            let (r, lines) = logged(|| run(&o, &object(51, 0, &[], 0)));
            assert_eq!(
                r.unwrap_err(),
                format!("Metal refused a tile render pipeline: {NIL_DESCRIPTOR_ERROR}")
            );
            assert_eq!(o.log().count(NEW_TILE), 1);
            assert_eq!(o.log().calls_to(NEW_TILE)[0].args[0], Arg::Obj(None));
            assert_eq!(
                lines,
                vec![format!(
                    "Metal refused a tile render pipeline: {NIL_DESCRIPTOR_ERROR}"
                )]
            );
        }
    }
}
