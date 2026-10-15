//! The pass prefix, which builds a fresh segment's pass descriptor from the records before its
//! main loop, and the main-loop records about the pass itself: the texture barrier, heap and
//! resource residency, and threadgroup memory.
//!
//! - **The prefix** takes the records 0x1e-0x24 in any order and number until 0x1a, which
//!   describes the attachments and ends it; any other id raises after the records before it were
//!   applied (`tests::prefix_in_order_until_describe`, `tests::prefix_unexpected_command`).
//!   A prefix record's length wraps in 32 bits, so a length under 8 overruns the segment
//!   (`tests::prefix_short_length_wraps`).
//! - **Attachments** are set in order: the texture is looked up and set, then the resolve texture
//!   is looked up, then the resolve texture and the other properties are set
//!   (`tests::describe_depth_order`); a lookup that raises leaves what came before it set.
//! - The memoryless rule reads the actions back from Metal, and keeps MultisampleResolve and
//!   Unknown (`tests::describe_memoryless_rule`).
//! - The highest colour index with a texture is kept, never reset
//!   (`tests::describe_colors_and_max_index`).
//! - **Lists** (0x86, 0x1b, 0x87, 0x89) are checked against the rest of the record first, then
//!   each entry is read and looked up in turn, then the list is sent once
//!   (`tests::use_heaps`, `tests::use_heaps_too_many`).

use crate::deserializer::RaiseKind as Shared;
use crate::deserializer::records::{
    CmdCoordinate, CmdDescribeRenderPass, CmdHeader, CmdRenderThreadgroupMemoryLength,
    CmdRenderUseHeapsWithStages, CmdSetDefaultSampleCount, CmdSetImageBlockSampleLength,
    CmdSetProgrammableSamplePositions, CmdSetThreadGroupMemoryLength, CmdSetTileParameters,
    CmdSetVariableRasterizationRateMap, CmdUseHeaps, CmdUseResources, CmdUseResourcesWithStages,
};
use crate::deserializer::render::{RaiseKind, RenderDecoder, refs};
use crate::memory::cursor::Cursor;
use crate::num::ToUsize;
use crate::platform::Platform;
use crate::platform::blit::ResourceRef;
use crate::platform::descriptor::RenderPassDescriptor as _;
use crate::platform::render::RenderCmd;
use crate::platform::resource::GpuResource as _;
use crate::platform::types::{ClearColor, MtlSamplePosition};
use crate::platform::values::{
    AttachmentState, ColorAttachmentSet, DepthAttachmentSet, PassSet, StencilAttachmentSet,
};
use crate::raise::{OnFault, Raise, RaiseResult};
use crate::resources::resource::AnyResource;
use crate::rt::Hold;

/// `MTLStorageModeMemoryless`.
const MTL_STORAGE_MODE_MEMORYLESS: u64 = 3;
/// `MTLStoreActionDontCare`.
const MTL_STORE_ACTION_DONT_CARE: u64 = 0;
/// `MTLStoreActionMultisampleResolve`.
const MTL_STORE_ACTION_MULTISAMPLE_RESOLVE: u64 = 2;
/// `MTLStoreActionUnknown`.
const MTL_STORE_ACTION_UNKNOWN: u64 = 4;
/// `MTLLoadActionDontCare`.
const MTL_LOAD_ACTION_DONT_CARE: u64 = 0;
/// `MTLLoadActionClear`.
const MTL_LOAD_ACTION_CLEAR: u64 = 2;
/// `MTLBarrierScopeRenderTargets`.
const MTL_BARRIER_SCOPE_RENDER_TARGETS: u64 = 4;
/// `MTLRenderStageFragment`.
const MTL_RENDER_STAGE_FRAGMENT: u64 = 2;
/// `MTLRenderStageVertex`.
const MTL_RENDER_STAGE_VERTEX: u64 = 1;
/// The most programmable sample positions 0x20 takes.
const MAX_SAMPLE_POSITIONS: u32 = 8;
/// The colour attachments 0x1a describes.
const COLOR_ATTACHMENT_COUNT: u32 = 8;

/// The set of one attachment's properties after its texture, from its record `$a` and its
/// resolve texture `$resolve` (sent only when the record names one): the resolve texture, the
/// level, slice and plane fields widened from `u16`, the load and store actions and the store
/// action options, then the attachment's own, `$own`.
macro_rules! attachment_set {
    ($set:ident, $a:ident, $resolve:expr, { $($own:tt)* }) => {
        $set {
            resolve_texture: $resolve,
            level: Some(u64::from($a.level)),
            slice: Some(u64::from($a.slice)),
            depth_plane: Some(u64::from($a.depth_plane)),
            resolve_level: Some(u64::from($a.resolve_level)),
            resolve_slice: Some(u64::from($a.resolve_slice)),
            resolve_depth_plane: Some(u64::from($a.resolve_depth_plane)),
            load_action: Some(u64::from($a.load_action)),
            store_action: Some(u64::from($a.store_action)),
            store_action_options: Some(u64::from($a.store_action_options)),
            $($own)*
            ..$set::default()
        }
    };
}

/// One attachment of a pass.
#[derive(Debug, Clone, Copy)]
enum Attachment {
    /// The depth attachment.
    Depth,
    /// The stencil attachment.
    Stencil,
    /// The colour attachment at this index.
    Color(u64),
}

/// The attachment's load and store actions, as Metal reads them back.
fn actions<P: Platform>(
    pass: &P::RenderPassDescriptor,
    which: Attachment,
) -> RaiseResult<AttachmentState> {
    match which {
        Attachment::Depth => Ok(pass.depth_attachment_state()),
        Attachment::Stencil => Ok(pass.stencil_attachment_state()),
        Attachment::Color(index) => pass.color_attachment_state(index),
    }
}

/// Sets the attachment's load action and store action, each when given.
fn set_actions<P: Platform>(
    pass: &P::RenderPassDescriptor,
    which: Attachment,
    load_action: Option<u64>,
    store_action: Option<u64>,
) -> RaiseResult<()> {
    match which {
        Attachment::Depth => {
            pass.set_depth_attachment(&DepthAttachmentSet {
                load_action,
                store_action,
                ..DepthAttachmentSet::default()
            });
            Ok(())
        }
        Attachment::Stencil => {
            pass.set_stencil_attachment(&StencilAttachmentSet {
                load_action,
                store_action,
                ..StencilAttachmentSet::default()
            });
            Ok(())
        }
        Attachment::Color(index) => pass.set_color_attachment(
            OnFault::Return,
            index,
            &ColorAttachmentSet {
                load_action,
                store_action,
                ..ColorAttachmentSet::default()
            },
        ),
    }
}

/// The memoryless rule: over a memoryless texture (none reads as storage mode 0, not
/// memoryless), a store action other than DontCare, Unknown or MultisampleResolve becomes
/// DontCare, and then a load action other than DontCare or Clear becomes Clear, each action as
/// Metal reads it back. MultisampleResolve and Unknown are kept.
fn memoryless_rule<P: Platform>(
    pass: &P::RenderPassDescriptor,
    which: Attachment,
    texture: Option<&P::Texture>,
) -> RaiseResult<()> {
    if texture.map_or(0, |t| t.storage_mode()) != MTL_STORAGE_MODE_MEMORYLESS {
        return Ok(());
    }
    let store = actions::<P>(pass, which)?.store_action;
    if store != MTL_STORE_ACTION_DONT_CARE
        && store != MTL_STORE_ACTION_UNKNOWN
        && store != MTL_STORE_ACTION_MULTISAMPLE_RESOLVE
    {
        set_actions::<P>(pass, which, None, Some(MTL_STORE_ACTION_DONT_CARE))?;
    }
    let load = actions::<P>(pass, which)?.load_action;
    if load != MTL_LOAD_ACTION_DONT_CARE && load != MTL_LOAD_ACTION_CLEAR {
        set_actions::<P>(pass, which, Some(MTL_LOAD_ACTION_CLEAR), None)?;
    }
    Ok(())
}

/// The bytes a list of `count` references takes (a `u32` count: no overflow in 64 bits).
fn list_bytes(count: u32) -> u64 {
    u64::from(count).wrapping_mul(4)
}

/// A looked-up resource as the encoder takes it. The delegate answers no heap for a resource
/// reference; should it, the heap goes as none.
fn resource_ref<P: Platform>(r: &Option<AnyResource<P>>) -> Option<ResourceRef<'_, P>> {
    match r {
        Some(AnyResource::Buffer(b)) => Some(ResourceRef::Buffer(b)),
        Some(AnyResource::Texture(t)) => Some(ResourceRef::Texture(t)),
        Some(AnyResource::Icb(i)) => Some(ResourceRef::Icb(i)),
        Some(AnyResource::Heap(_)) | None => None,
    }
}

impl<P: Platform> RenderDecoder<P> {
    /// The pass prefix: a new pass descriptor, set by each prefix record in turn until 0x1a.
    pub(super) fn pass_prefix(
        &mut self,
        hold: Hold<'_>,
        segment: &mut Cursor,
    ) -> RaiseResult<P::RenderPassDescriptor> {
        let descriptor = self.deserializer().platform().new_render_pass_descriptor();
        loop {
            let header: CmdHeader = segment.read_or_raise(OnFault::Return)?;
            // 32-bit wrap, no underflow check: a length under 8 becomes at least 0xfffffff8 and
            // overruns the segment.
            let length = header.length.wrapping_sub(8);
            let mut record = match segment.sub_cursor(u64::from(length)) {
                Ok(record) => record,
                Err(error) => {
                    return Err(Raise::new(Shared::RecordOverrun)
                        .with_detail(format_args!("record not readable: {error}")));
                }
            };
            let c = &mut record;
            match header.command_id {
                0x1a => {
                    self.describe_render_pass(hold, c, &descriptor)?;
                    return Ok(descriptor);
                }
                0x1e => self.pass_default_raster_sample_count(c, &descriptor)?,
                0x1f => self.pass_default_color_sample_count(c, &descriptor)?,
                0x20 => self.pass_sample_positions(c, &descriptor)?,
                0x21 => self.pass_rate_map(hold, c, &descriptor)?,
                0x22 => self.pass_imageblock_sample_length(c, &descriptor)?,
                0x23 => self.pass_threadgroup_memory_length(c, &descriptor)?,
                0x24 => self.pass_tile_size(c, &descriptor)?,
                id => crate::raise!(
                    RaiseKind::UnexpectedPassCommand,
                    format_args!("command {id} is not a render pass descriptor command")
                ),
            }
        }
    }

    /// 0x1a: the depth, stencil and colour attachments (each only with a texture reference),
    /// the visibility result buffer, then the render target's extent.
    pub(super) fn describe_render_pass(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
        pass: &P::RenderPassDescriptor,
    ) -> RaiseResult<()> {
        let command: CmdDescribeRenderPass = c.read_or_raise(OnFault::Return)?;

        let depth = command.depth_attachment;
        if depth.texture_reference != 0 {
            let texture = self.texture(hold, depth.texture_reference)?;
            pass.set_depth_attachment(&DepthAttachmentSet {
                texture: Some(texture.as_ref()),
                ..DepthAttachmentSet::default()
            });
            let resolve = self.resolve_texture(hold, depth.resolve_texture_reference)?;
            pass.set_depth_attachment(&attachment_set!(
                DepthAttachmentSet,
                depth,
                resolve.as_ref().map(Option::as_ref),
                {
                    clear_depth: Some(depth.clear_depth),
                    depth_resolve_filter: Some(u64::from(depth.depth_resolve_filter)),
                }
            ));
            memoryless_rule::<P>(pass, Attachment::Depth, texture.as_ref())?;
        }

        let stencil = command.stencil_attachment;
        if stencil.texture_reference != 0 {
            let texture = self.texture(hold, stencil.texture_reference)?;
            pass.set_stencil_attachment(&StencilAttachmentSet {
                texture: Some(texture.as_ref()),
                ..StencilAttachmentSet::default()
            });
            let resolve = self.resolve_texture(hold, stencil.resolve_texture_reference)?;
            pass.set_stencil_attachment(&attachment_set!(
                StencilAttachmentSet,
                stencil,
                resolve.as_ref().map(Option::as_ref),
                {
                    clear_stencil: Some(stencil.clear_stencil),
                    stencil_resolve_filter: Some(u64::from(stencil.stencil_resolve_filter)),
                }
            ));
            memoryless_rule::<P>(pass, Attachment::Stencil, texture.as_ref())?;
        }

        for (i, color) in (0..COLOR_ATTACHMENT_COUNT).zip(command.color_attachments) {
            if color.texture_reference == 0 {
                continue;
            }
            let index = u64::from(i);
            let texture = self.texture(hold, color.texture_reference)?;
            pass.set_color_attachment(
                OnFault::Return,
                index,
                &ColorAttachmentSet {
                    texture: Some(texture.as_ref()),
                    ..ColorAttachmentSet::default()
                },
            )?;
            let resolve = self.resolve_texture(hold, color.resolve_texture_reference)?;
            pass.set_color_attachment(
                OnFault::Return,
                index,
                &attachment_set!(
                    ColorAttachmentSet,
                    color,
                    resolve.as_ref().map(Option::as_ref),
                    {
                        clear_color: Some(ClearColor {
                            red: color.clear_color_red,
                            green: color.clear_color_green,
                            blue: color.clear_color_blue,
                            alpha: color.clear_color_alpha,
                        }),
                    }
                ),
            )?;
            // The last colour index with a texture, not a count; never reset.
            self.max_color_attachment_index = i;
            memoryless_rule::<P>(pass, Attachment::Color(index), texture.as_ref())?;
        }

        let visibility = match command.visibility_result_buffer_reference {
            0 => None,
            reference => Some(self.buffer(hold, reference)?),
        };
        pass.set_pass(&PassSet {
            visibility_result_buffer: visibility.as_ref().map(Option::as_ref),
            render_target_array_length: Some(command.render_target_array_length),
            render_target_width: Some(command.render_target_width),
            render_target_height: Some(command.render_target_height),
            ..PassSet::default()
        });
        Ok(())
    }

    /// An attachment's resolve texture: `None` for reference 0, which sets none, else the
    /// lookup's answer (none included).
    fn resolve_texture(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Option<Option<P::Texture>>> {
        match reference {
            0 => Ok(None),
            reference => Ok(Some(self.texture(hold, reference)?)),
        }
    }

    /// 0x1e.
    pub(super) fn pass_default_raster_sample_count(
        &mut self,
        c: &mut Cursor,
        pass: &P::RenderPassDescriptor,
    ) -> RaiseResult<()> {
        let command: CmdSetDefaultSampleCount = c.read_or_raise(OnFault::Return)?;
        pass.set_pass(&PassSet {
            default_raster_sample_count: Some(u64::from(command.sample_count)),
            ..PassSet::default()
        });
        Ok(())
    }

    /// 0x1f.
    pub(super) fn pass_default_color_sample_count(
        &mut self,
        c: &mut Cursor,
        pass: &P::RenderPassDescriptor,
    ) -> RaiseResult<()> {
        let command: CmdSetDefaultSampleCount = c.read_or_raise(OnFault::Return)?;
        pass.set_pass(&PassSet {
            default_color_sample_count: Some(u64::from(command.sample_count)),
            ..PassSet::default()
        });
        Ok(())
    }

    /// 0x20: up to eight positions; a count of 0 still sends an empty list.
    pub(super) fn pass_sample_positions(
        &mut self,
        c: &mut Cursor,
        pass: &P::RenderPassDescriptor,
    ) -> RaiseResult<()> {
        let command: CmdSetProgrammableSamplePositions = c.read_or_raise(OnFault::Return)?;
        let count = command.count;
        if count > MAX_SAMPLE_POSITIONS {
            crate::raise!(
                RaiseKind::SamplePositionCount,
                format_args!("{count} sample positions, more than {MAX_SAMPLE_POSITIONS}")
            );
        }
        let mut positions = Vec::with_capacity(count.to_usize());
        for _ in 0..count {
            let coordinate: CmdCoordinate = c.read_or_raise(OnFault::Return)?;
            positions.push(MtlSamplePosition {
                x: coordinate.x,
                y: coordinate.y,
            });
        }
        pass.set_sample_positions(&positions)
    }

    /// 0x21: the rate map, none included.
    pub(super) fn pass_rate_map(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
        pass: &P::RenderPassDescriptor,
    ) -> RaiseResult<()> {
        let command: CmdSetVariableRasterizationRateMap = c.read_or_raise(OnFault::Return)?;
        let map = self
            .deserializer()
            .object_delegate()
            .get_rasterization_rate_map(hold, command.rasterization_rate_map_reference)?;
        pass.set_pass(&PassSet {
            rasterization_rate_map: Some(map.as_ref()),
            ..PassSet::default()
        });
        Ok(())
    }

    /// 0x22.
    pub(super) fn pass_imageblock_sample_length(
        &mut self,
        c: &mut Cursor,
        pass: &P::RenderPassDescriptor,
    ) -> RaiseResult<()> {
        let command: CmdSetImageBlockSampleLength = c.read_or_raise(OnFault::Return)?;
        pass.set_pass(&PassSet {
            imageblock_sample_length: Some(u64::from(command.imageblock_sample_length)),
            ..PassSet::default()
        });
        Ok(())
    }

    /// 0x23.
    pub(super) fn pass_threadgroup_memory_length(
        &mut self,
        c: &mut Cursor,
        pass: &P::RenderPassDescriptor,
    ) -> RaiseResult<()> {
        let command: CmdSetThreadGroupMemoryLength = c.read_or_raise(OnFault::Return)?;
        pass.set_pass(&PassSet {
            threadgroup_memory_length: Some(u64::from(command.threadgroup_memory_length)),
            ..PassSet::default()
        });
        Ok(())
    }

    /// 0x24: the tile width, then its height.
    pub(super) fn pass_tile_size(
        &mut self,
        c: &mut Cursor,
        pass: &P::RenderPassDescriptor,
    ) -> RaiseResult<()> {
        let command: CmdSetTileParameters = c.read_or_raise(OnFault::Return)?;
        pass.set_pass(&PassSet {
            tile_width: Some(u64::from(command.tile_width)),
            tile_height: Some(u64::from(command.tile_height)),
            ..PassSet::default()
        });
        Ok(())
    }

    /// 0x85: a render-target barrier from the fragment stage to the vertex stage. Reads nothing.
    pub(super) fn texture_barrier(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let _ = c;
        self.send(RenderCmd::MemoryBarrierWithScope {
            scope: MTL_BARRIER_SCOPE_RENDER_TARGETS,
            after: MTL_RENDER_STAGE_FRAGMENT,
            before: MTL_RENDER_STAGE_VERTEX,
        })
    }

    /// 0x86.
    pub(super) fn use_heaps(&mut self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdUseHeaps = c.read_or_raise(OnFault::Return)?;
        let heaps = self.heap_list(hold, c, command.count)?;
        self.send(RenderCmd::UseHeaps {
            heaps: &refs(&heaps),
        })
    }

    /// 0x1b.
    pub(super) fn use_heaps_with_stages(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdRenderUseHeapsWithStages = c.read_or_raise(OnFault::Return)?;
        let heaps = self.heap_list(hold, c, command.count)?;
        self.send(RenderCmd::UseHeapsStages {
            heaps: &refs(&heaps),
            stages: u64::from(command.stages),
        })
    }

    /// 0x87.
    pub(super) fn use_resources(&mut self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdUseResources = c.read_or_raise(OnFault::Return)?;
        let resources = self.resource_list(hold, c, command.count)?;
        let list: Vec<Option<ResourceRef<'_, P>>> = resources.iter().map(resource_ref).collect();
        self.send(RenderCmd::UseResources {
            resources: &list,
            usage: u64::from(command.usage),
        })
    }

    /// 0x89.
    pub(super) fn use_resources_with_stages(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdUseResourcesWithStages = c.read_or_raise(OnFault::Return)?;
        let resources = self.resource_list(hold, c, command.count)?;
        let list: Vec<Option<ResourceRef<'_, P>>> = resources.iter().map(resource_ref).collect();
        self.send(RenderCmd::UseResourcesStages {
            resources: &list,
            usage: u64::from(command.usage),
            stages: u64::from(command.stages),
        })
    }

    /// 0x9c: the index is bounded by the device's `maxComputeLocalMemorySizes`.
    pub(super) fn set_threadgroup_memory_length(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdRenderThreadgroupMemoryLength = c.read_or_raise(OnFault::Return)?;
        let index = u64::from(command.index);
        if index >= self.device_limit(|c| c.limits.max_compute_local_memory_sizes) {
            crate::raise!(
                RaiseKind::ThreadgroupMemoryIndex,
                format_args!("threadgroup memory index {index} is past the device's limit")
            );
        }
        self.send(RenderCmd::SetThreadgroupMemoryLength {
            length: command.length,
            offset: command.offset,
            index,
        })
    }

    /// `count` heap references, each looked up as it is read, once the rest of the record holds
    /// them all.
    fn heap_list(
        &self,
        hold: Hold<'_>,
        c: &mut Cursor,
        count: u32,
    ) -> RaiseResult<Vec<Option<P::Heap>>> {
        if c.remaining() < list_bytes(count) {
            crate::raise!(
                RaiseKind::ListLength,
                format_args!("reference count {count} runs past the record")
            );
        }
        let mut heaps = Vec::with_capacity(count.to_usize());
        for _ in 0..count {
            let reference: u32 = c.read_or_raise(OnFault::Return)?;
            heaps.push(
                self.deserializer()
                    .object_delegate()
                    .get_heap(hold, reference)?,
            );
        }
        Ok(heaps)
    }

    /// `count` resource references, each looked up as it is read, once the rest of the record
    /// holds them all.
    fn resource_list(
        &self,
        hold: Hold<'_>,
        c: &mut Cursor,
        count: u32,
    ) -> RaiseResult<Vec<Option<AnyResource<P>>>> {
        if c.remaining() < list_bytes(count) {
            crate::raise!(
                RaiseKind::ListLength,
                format_args!("reference count {count} runs past the record")
            );
        }
        let mut resources = Vec::with_capacity(count.to_usize());
        for _ in 0..count {
            let reference: u32 = c.read_or_raise(OnFault::Return)?;
            resources.push(
                self.deserializer()
                    .object_delegate()
                    .get_resource(hold, reference)?,
            );
        }
        Ok(resources)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::deserializer::records::{
        DescribeRenderPassColorAttachment, DescribeRenderPassDepthAttachment,
        DescribeRenderPassStencilAttachment,
    };
    use crate::deserializer::render::test_rig::*;
    use crate::platform::{Handle, ObjId};
    use crate::raise::death::DeathKind;
    use crate::testkit::FakePlatform;
    use crate::testkit::calllog::Arg;
    use crate::testkit::capture::catch_death;
    use crate::testkit::impls::descriptor::{AttachmentModel, PassModel};
    use crate::testkit::objects::{Fk, kind};
    use crate::testkit::script::Step;

    type Pass = <FakePlatform as Platform>::RenderPassDescriptor;

    fn describe(cmd: &CmdDescribeRenderPass) -> Vec<u8> {
        record(0x1a, &bytes(cmd))
    }

    /// Runs the prefix over `bytes`; returns the descriptor.
    fn prefix(hold: Hold<'_>, r: &mut Rig, b: &[u8]) -> RaiseResult<Pass> {
        let mut c = cursor(b);
        r.dec.pass_prefix(hold, &mut c)
    }

    fn texture(r: &Rig, reference: u32, storage_mode: u64) -> Fk<kind::Texture> {
        let t = r.provide::<kind::Texture>(reference);
        t.set_storage_mode(storage_mode);
        t
    }

    /// The ordered calls `obj` received, as (selector, args).
    fn sels_on(r: &Rig, obj: ObjId) -> Vec<(&'static str, Vec<Arg>)> {
        r.calls()
            .into_iter()
            .filter(|c| c.obj == obj)
            .map(|c| (c.sel, c.args))
            .collect()
    }

    /// The references the texture lookups were asked for, in order.
    fn texture_lookups(r: &Rig) -> Vec<Vec<Arg>> {
        r.calls()
            .into_iter()
            .filter(|c| c.sel == "get_texture")
            .map(|c| c.args)
            .collect()
    }

    /// The integers every attachment takes from its record, as `a` holds them: the level, slice
    /// and plane fields, then the load and store actions and the options.
    fn common(a: &AttachmentModel) -> [Option<u64>; 9] {
        [
            a.level,
            a.slice,
            a.depth_plane,
            a.resolve_level,
            a.resolve_slice,
            a.resolve_depth_plane,
            a.load_action,
            a.store_action,
            a.store_action_options,
        ]
    }

    /// A fresh descriptor, made outside the prefix.
    fn descriptor(r: &Rig) -> Pass {
        r.p.new_render_pass_descriptor()
    }

    /// Prefix records in any order and number, then 0x1a ends it; records may be longer than their
    /// struct.
    #[test]
    fn prefix_in_order_until_describe() {
        let hold = Hold::NONE;
        let mut r = Rig::new();
        let mut b = Vec::new();
        b.extend(record(0x1f, &5u32.to_le_bytes()));
        b.extend(record(0x1e, &[4, 0, 0, 0, 0xee, 0xee]));
        b.extend(record(0x22, &32u32.to_le_bytes()));
        b.extend(record(0x1e, &2u32.to_le_bytes()));
        b.extend(describe_empty());
        b.extend(record(0x85, &[]));
        let mut c = cursor(&b);
        let d = r.dec.pass_prefix(hold, &mut c).unwrap();
        assert_eq!(c.remaining(), 8, "the main loop's record is left");
        // The later 0x1e is the one the descriptor keeps.
        assert_eq!(
            d.pass(|p| [
                p.default_color_sample_count,
                p.default_raster_sample_count,
                p.imageblock_sample_length,
                p.render_target_array_length,
                p.render_target_width,
                p.render_target_height,
            ]),
            [5, 2, 32, 0, 0, 0].map(Some)
        );
        assert_eq!(d.pass(PassModel::visibility_result_buffer_id), None);
        assert!(r.sels().is_empty(), "nothing ordered, no lookup");
    }

    /// A prefix length under 8 wraps in 32 bits and fails the sub-cursor (a refused read).
    #[test]
    fn prefix_short_length_wraps() {
        let hold = Hold::NONE;
        let mut r = Rig::new();
        assert_raised(
            prefix(hold, &mut r, &header(0x1a, 4, &[0; 4])),
            Shared::RecordOverrun,
            "record not readable: a sub-cursor of 0xfffffffc bytes, with 0x4 left",
        );
        // An overrunning length fails the same way.
        let mut r = Rig::new();
        assert_raised(
            prefix(hold, &mut r, &header(0x1e, 16, &[0; 4])),
            Shared::RecordOverrun,
            "record not readable: a sub-cursor of 0x8 bytes, with 0x4 left",
        );
    }

    /// An empty segment fails at the first header read; a short record fails its own read.
    #[test]
    fn prefix_empty_and_short() {
        let hold = Hold::NONE;
        let mut r = Rig::new();
        assert_failed_read(prefix(hold, &mut r, &[]));
        let mut r = Rig::new();
        assert_failed_read(prefix(hold, &mut r, &record(0x1a, &[0; 0x247])));
        let mut r = Rig::new();
        assert_failed_read(prefix(hold, &mut r, &record(0x1e, &[0; 3])));
    }

    /// Any other id: "UnexpectedCommand", after the records before it were applied (the sample
    /// positions, which Metal takes at once).
    #[test]
    fn prefix_unexpected_command() {
        for id in [0x0, 0x1b, 0x1d, 0x25, 0x85] {
            let hold = Hold::NONE;
            let mut r = Rig::new();
            let mut b = record(0x20, &0u32.to_le_bytes());
            b.extend(record(id, &[0; 8]));
            assert_raised(
                prefix(hold, &mut r, &b),
                RaiseKind::UnexpectedPassCommand,
                &format!("command {id} is not a render pass descriptor command"),
            );
            assert_eq!(r.sels(), vec!["setSamplePositions:count:"]);
        }
    }

    fn depth_record(texture: u32, resolve: u32, load: u16, store: u16) -> CmdDescribeRenderPass {
        CmdDescribeRenderPass {
            depth_attachment: DescribeRenderPassDepthAttachment {
                texture_reference: texture,
                resolve_texture_reference: resolve,
                level: 1,
                slice: 2,
                depth_plane: 3,
                resolve_level: 4,
                resolve_slice: 5,
                resolve_depth_plane: 0xffff,
                load_action: load,
                store_action: store,
                store_action_options: 7,
                field_1a: 0xaaaa,
                clear_depth: 0.25,
                depth_resolve_filter: 2,
                field_26: 0xbbbb,
            },
            ..Default::default()
        }
    }

    /// The depth attachment: the texture looked up, then the resolve texture, and each property
    /// set from the record, widened from u16.
    #[test]
    fn describe_depth_order() {
        let hold = Hold::NONE;
        let mut r = Rig::new();
        let t = texture(&r, 9, 0);
        let rt = r.provide::<kind::Texture>(10);
        let d = prefix(hold, &mut r, &describe(&depth_record(9, 10, 1, 1))).unwrap();
        assert_eq!(
            texture_lookups(&r),
            vec![vec![Arg::U32(9)], vec![Arg::U32(10)]]
        );
        d.depth(|a| {
            assert_eq!(a.texture_id(), Some(t.id()));
            assert_eq!(a.resolve_texture_id(), Some(rt.id()));
            assert_eq!(common(a), [1, 2, 3, 4, 5, 0xffff, 1, 1, 7].map(Some));
            assert_eq!(a.clear_depth, Some(0.25));
            assert_eq!(a.depth_resolve_filter, Some(2));
        });
        // The stencil and colours are skipped (reference 0).
        assert_eq!(d.stencil(AttachmentModel::texture_id), None);
        assert!((0..8).all(|i| d.color(i, AttachmentModel::texture_id).is_none()));
    }

    /// A resolve texture's lookup that raises leaves the texture set on the attachment, and
    /// nothing after it; a resolve reference of 0 sets no resolve texture.
    #[test]
    fn describe_resolve_lookup_raises_after_the_texture_is_set() {
        let hold = Hold::NONE;
        let mut r = Rig::new();
        let t = texture(&r, 9, 0);
        r.p.script().push("get_texture", Step::Default);
        r.p.script().push(
            "get_texture",
            Step::raise("NSGenericException", "Failed to create Texture"),
        );
        let d = descriptor(&r);
        let cmd = bytes(&depth_record(9, 10, 1, 1));
        assert!(
            r.dec
                .describe_render_pass(hold, &mut cursor(&cmd), &d)
                .is_err()
        );
        d.depth(|a| {
            assert_eq!(a.texture_id(), Some(t.id()));
            assert!(a.resolve_texture.is_none());
            assert_eq!(common(a), [None; 9]);
        });
        // Reference 0: no resolve texture, the rest set.
        let mut r = Rig::new();
        texture(&r, 9, 0);
        let d = prefix(hold, &mut r, &describe(&depth_record(9, 0, 1, 1))).unwrap();
        d.depth(|a| {
            assert!(a.resolve_texture.is_none());
            assert_eq!(a.level, Some(1));
        });
        assert_eq!(texture_lookups(&r), vec![vec![Arg::U32(9)]]);
    }

    /// The actions a memoryless depth attachment ends with, `(load, store)`, from the record's.
    fn memoryless_depth(load: u16, store: u16) -> (u64, u64) {
        {
            let hold = Hold::NONE;
            let mut r = Rig::new();
            texture(&r, 9, MTL_STORAGE_MODE_MEMORYLESS);
            let d = prefix(hold, &mut r, &describe(&depth_record(9, 0, load, store))).unwrap();
            let actions = d.depth(|a| (a.load_action.unwrap(), a.store_action.unwrap()));
            drop(d);
            actions
        }
    }

    /// The memoryless rule keeps DontCare, Unknown and MultisampleResolve stores and DontCare and
    /// Clear loads; any other store becomes DontCare, any other load Clear.
    #[test]
    fn describe_memoryless_rule() {
        // Store 1 (Store) becomes DontCare; load 1 (Load) becomes Clear.
        assert_eq!(memoryless_depth(1, 1), (2, 0));
        // DontCare kept.
        assert_eq!(memoryless_depth(0, 0), (0, 0));
        // MultisampleResolve and Clear kept.
        assert_eq!(memoryless_depth(2, 2), (2, 2));
        // Unknown kept; StoreAndMultisampleResolve (3) becomes DontCare.
        assert_eq!(memoryless_depth(0, 4), (0, 4));
        assert_eq!(memoryless_depth(0, 3), (0, 0));
        // Not memoryless: the record's actions stay.
        {
            let hold = Hold::NONE;
            let mut r = Rig::new();
            texture(&r, 9, 0);
            let d = prefix(hold, &mut r, &describe(&depth_record(9, 0, 1, 1))).unwrap();
            assert_eq!(
                d.depth(|a| (a.load_action, a.store_action)),
                (Some(1), Some(1))
            );
        };
    }

    /// A memoryless depth attachment whose store action stays MultisampleResolve reaches Metal,
    /// which dies creating the encoder (testing.md 7.4; scripted here).
    #[test]
    fn describe_memoryless_resolve_dies_in_metal() {
        let hold = Hold::NONE;
        let mut r = Rig::new();
        texture(&r, 9, MTL_STORAGE_MODE_MEMORYLESS);
        r.p.script()
            .push("renderCommandEncoderWithDescriptor:", Step::crash());
        let seg = crate::deserializer::records::SegmentHeader::default();
        let b = describe(&depth_record(9, 0, 0, 2));
        let died = catch_death(move || {
            let _ = r.dec.decode(hold, &seg, cursor(&b));
        });
        assert_eq!(
            died,
            Err(crate::testkit::script::metal_crash_site(
                "renderCommandEncoderWithDescriptor:"
            ))
        );
    }

    /// The stencil attachment: clear stencil (u32) and its filter; the memoryless rule.
    #[test]
    fn describe_stencil() {
        let hold = Hold::NONE;
        let mut r = Rig::new();
        let t = texture(&r, 3, MTL_STORAGE_MODE_MEMORYLESS);
        let cmd = CmdDescribeRenderPass {
            stencil_attachment: DescribeRenderPassStencilAttachment {
                texture_reference: 3,
                load_action: 1,
                store_action: 1,
                clear_stencil: 0xdead_beef,
                stencil_resolve_filter: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let d = prefix(hold, &mut r, &describe(&cmd)).unwrap();
        d.stencil(|a| {
            assert_eq!(a.texture_id(), Some(t.id()));
            assert_eq!(a.clear_stencil, Some(0xdead_beef));
            assert_eq!(a.stencil_resolve_filter, Some(1));
            assert_eq!((a.load_action, a.store_action), (Some(2), Some(0)));
        });
        assert_eq!(d.depth(AttachmentModel::texture_id), None);
    }

    /// Colours with a texture only; `_maxColorAttachmentIndex` is the last such index, and a later
    /// pass with none leaves it.
    #[test]
    fn describe_colors_and_max_index() {
        let hold = Hold::NONE;
        let mut r = Rig::new();
        let t5 = texture(&r, 5, 0);
        let t6 = texture(&r, 6, MTL_STORAGE_MODE_MEMORYLESS);
        let mut cmd = CmdDescribeRenderPass::default();
        cmd.color_attachments[1] = DescribeRenderPassColorAttachment {
            texture_reference: 5,
            load_action: 1,
            clear_color_red: 0.5,
            clear_color_alpha: 1.0,
            ..Default::default()
        };
        cmd.color_attachments[6] = DescribeRenderPassColorAttachment {
            texture_reference: 6,
            load_action: 1,
            store_action: 1,
            ..Default::default()
        };
        let d = prefix(hold, &mut r, &describe(&cmd)).unwrap();
        assert_eq!(r.dec.max_color_attachment_index, 6);
        assert_eq!(
            texture_lookups(&r),
            vec![vec![Arg::U32(5)], vec![Arg::U32(6)]]
        );
        d.color(1, |c1| {
            assert_eq!(c1.texture_id(), Some(t5.id()));
            assert_eq!(
                c1.clear_color,
                Some(ClearColor {
                    red: 0.5,
                    green: 0.0,
                    blue: 0.0,
                    alpha: 1.0
                })
            );
            assert_eq!((c1.load_action, c1.store_action), (Some(1), Some(0)));
        });
        // Colour 6 is memoryless: store 1 → DontCare, load 1 → Clear.
        d.color(6, |c6| {
            assert_eq!(c6.texture_id(), Some(t6.id()));
            assert_eq!((c6.load_action, c6.store_action), (Some(2), Some(0)));
        });
        for i in [0, 2, 3, 4, 5, 7] {
            assert_eq!(d.color(i, common), [None; 9], "colour {i}");
        }
        // A later pass without colours leaves the index.
        prefix(hold, &mut r, &describe_empty()).unwrap();
        assert_eq!(r.dec.max_color_attachment_index, 6);
    }

    /// The visibility buffer (nullable lookup, only for a non-zero reference) and the render target
    /// extent, set unconditionally, last.
    #[test]
    fn describe_visibility_and_targets() {
        let hold = Hold::NONE;
        let mut r = Rig::new();
        let b = r.provide::<kind::Buffer>(0x77);
        let cmd = CmdDescribeRenderPass {
            visibility_result_buffer_reference: 0x77,
            render_target_array_length: 2,
            render_target_width: u64::MAX,
            render_target_height: 1 << 40,
            ..Default::default()
        };
        let d = prefix(hold, &mut r, &describe(&cmd)).unwrap();
        assert_eq!(r.sels(), vec!["get_buffer"]);
        assert_eq!(d.pass(PassModel::visibility_result_buffer_id), Some(b.id()));
        assert_eq!(
            d.pass(|p| [
                p.render_target_array_length,
                p.render_target_width,
                p.render_target_height
            ]),
            [Some(2), Some(u64::MAX), Some(1 << 40)]
        );
        // A lookup that raises stops the pass there.
        let mut r = Rig::new();
        r.p.script().push(
            "get_buffer",
            Step::raise("NSGenericException", "Failed to create Buffer"),
        );
        let d = descriptor(&r);
        assert!(
            r.dec
                .describe_render_pass(hold, &mut cursor(&bytes(&cmd)), &d)
                .is_err()
        );
        assert_eq!(d.pass(|p| p.render_target_array_length), None);
    }

    #[test]
    fn sample_positions() {
        let hold = Hold::NONE;
        // Two positions, bit for bit.
        let mut r = Rig::new();
        let mut p = 2u32.to_le_bytes().to_vec();
        for f in [0.25f32, -0.0, f32::MIN_POSITIVE, 0.75] {
            p.extend(f.to_le_bytes());
        }
        let mut b = record(0x20, &p);
        b.extend(describe_empty());
        let d = prefix(hold, &mut r, &b).unwrap();
        let on_d = sels_on(&r, d.id());
        let expect = [
            MtlSamplePosition { x: 0.25, y: -0.0 },
            MtlSamplePosition {
                x: f32::MIN_POSITIVE,
                y: 0.75,
            },
        ];
        assert_eq!(
            on_d[0],
            ("setSamplePositions:count:", vec![Arg::val(&expect[..])])
        );
        // Count 0: an empty list is still sent.
        let mut r = Rig::new();
        let mut b = record(0x20, &0u32.to_le_bytes());
        b.extend(describe_empty());
        let d = prefix(hold, &mut r, &b).unwrap();
        let empty: &[MtlSamplePosition] = &[];
        assert_eq!(
            sels_on(&r, d.id())[0],
            ("setSamplePositions:count:", vec![Arg::val(empty)])
        );
        // Count 9: refused before any coordinate is read.
        let mut r = Rig::new();
        assert_raised(
            prefix(hold, &mut r, &record(0x20, &9u32.to_le_bytes())),
            RaiseKind::SamplePositionCount,
            "9 sample positions, more than 8",
        );
        // Count 8 is allowed; a short list fails the read.
        let mut r = Rig::new();
        let mut p = 8u32.to_le_bytes().to_vec();
        p.extend([0u8; 8 * 7]);
        assert_failed_read(prefix(hold, &mut r, &record(0x20, &p)));
        assert!(!r.sels().contains(&"setSamplePositions:count:"));
        // A raise from Metal's setter propagates.
        let mut r = Rig::new();
        r.p.script().push(
            "setSamplePositions:count:",
            Step::raise("NSInvalidArgumentException", "x"),
        );
        assert!(prefix(hold, &mut r, &record(0x20, &0u32.to_le_bytes())).is_err());
    }

    #[test]
    fn texture_barrier() {
        let (mut r, e) = Rig::with_encoder();
        let mut c = cursor(&[1, 2, 3]);
        r.dec.texture_barrier(&mut c).unwrap();
        assert_eq!(c.remaining(), 3, "reads nothing");
        assert_eq!(
            sels_on(&r, e.id()),
            vec![(
                "memoryBarrierWithScope:afterStages:beforeStages:",
                vec![Arg::U64(4), Arg::U64(2), Arg::U64(1)]
            )]
        );
        // A nil encoder: a send to nil.
        r.dec.encoder = None;
        r.dec.texture_barrier(&mut c).unwrap();
        assert_eq!(r.calls().len(), 1);
    }

    fn list(head: &[u8], refs: &[u32]) -> Vec<u8> {
        let mut v = head.to_vec();
        for r in refs {
            v.extend(r.to_le_bytes());
        }
        v
    }

    /// A count past the record's remaining bytes raises before any lookup, a raise that names
    /// its own site.
    #[test]
    fn use_heaps_too_many() {
        let (mut r, _e) = Rig::with_encoder();
        let b = list(&3u32.to_le_bytes(), &[1, 2]);
        {
            {
                let hold = Hold::NONE;
                assert_raised(
                    r.dec.use_heaps(hold, &mut cursor(&b)),
                    RaiseKind::ListLength,
                    "reference count 3 runs past the record",
                );
            }
        }
        assert!(r.sels().is_empty());
        let raised = r.dec.use_heaps(Hold::NONE, &mut cursor(&b));
        assert_eq!(
            raised
                .map(|_| ())
                .map_err(|r| (r.site().kind, r.site().file)),
            Err((DeathKind::UnhandledRaise, "deserializer.render.list_length"))
        );
        // A short command record fails its read.
        {
            let hold = Hold::NONE;
            let (mut r, _e) = Rig::with_encoder();
            assert_failed_read(r.dec.use_heaps(hold, &mut cursor(&[1, 0])));
        };
    }

    /// A resource lookup per reference; buffers, textures and ICBs pass as themselves, nil as nil.
    #[test]
    fn use_resources() {
        let hold = Hold::NONE;
        let (mut r, e) = Rig::with_encoder();
        let b = r.provide::<kind::Buffer>(1);
        let t = r.provide::<kind::Texture>(2);
        let i = r.provide::<kind::Icb>(3);
        let head = bytes(&CmdUseResources { count: 4, usage: 3 });
        r.dec
            .use_resources(hold, &mut cursor(&list(&head, &[1, 2, 3, 99])))
            .unwrap();
        assert_eq!(r.p.call_log().count("get_resource"), 4);
        assert_eq!(
            sels_on(&r, e.id()),
            vec![(
                "useResources:count:usage:",
                vec![
                    Arg::Objs(vec![Some(b.id()), Some(t.id()), Some(i.id()), None]),
                    Arg::U64(3)
                ]
            )]
        );
        assert_raised(
            r.dec
                .use_resources(hold, &mut cursor(&list(&head, &[1, 2, 3]))),
            RaiseKind::ListLength,
            "reference count 4 runs past the record",
        );
    }

    #[test]
    fn use_resources_with_stages() {
        let hold = Hold::NONE;
        let (mut r, e) = Rig::with_encoder();
        let t = r.provide::<kind::Texture>(2);
        let head = bytes(&CmdUseResourcesWithStages {
            count: 1,
            usage: 2,
            stages: 6,
        });
        r.dec
            .use_resources_with_stages(hold, &mut cursor(&list(&head, &[2])))
            .unwrap();
        assert_eq!(
            sels_on(&r, e.id()),
            vec![(
                "useResources:count:usage:stages:",
                vec![Arg::Objs(vec![Some(t.id())]), Arg::U64(2), Arg::U64(6)]
            )]
        );
        assert_raised(
            r.dec
                .use_resources_with_stages(hold, &mut cursor(&list(&head, &[]))),
            RaiseKind::ListLength,
            "reference count 1 runs past the record",
        );
        // A heap the seam returns (the real lookup never does) goes as nil.
        let (mut r, e) = Rig::with_encoder();
        r.provide::<kind::Heap>(5);
        r.dec
            .use_resources_with_stages(hold, &mut cursor(&list(&head, &[5])))
            .unwrap();
        assert_eq!(sels_on(&r, e.id())[0].1[0], Arg::Objs(vec![None]));
    }

    #[test]
    fn set_render_threadgroup_memory_length() {
        let (mut r, e) = Rig::with_encoder();
        r.set_caps(|c| c.limits.max_compute_local_memory_sizes = 4);
        let rec = |index: u32| {
            bytes(&CmdRenderThreadgroupMemoryLength {
                length: 0x1_0000_0000,
                offset: 16,
                index,
            })
        };
        r.dec
            .set_threadgroup_memory_length(&mut cursor(&rec(3)))
            .unwrap();
        assert_eq!(
            sels_on(&r, e.id()),
            vec![(
                "setThreadgroupMemoryLength:offset:atIndex:",
                vec![Arg::U64(0x1_0000_0000), Arg::U64(16), Arg::U64(3)]
            )]
        );
        assert_raised(
            r.dec.set_threadgroup_memory_length(&mut cursor(&rec(4))),
            RaiseKind::ThreadgroupMemoryIndex,
            "threadgroup memory index 4 is past the device's limit",
        );
        assert_failed_read(r.dec.set_threadgroup_memory_length(&mut cursor(&[0; 0x13])));
        assert_eq!(
            r.p.call_log()
                .count("setThreadgroupMemoryLength:offset:atIndex:"),
            1
        );
    }

    /// Each reads a u32 and sets it widened; a short record fails.
    #[test]
    fn descriptor_scalars() {
        let mut r = Rig::new();
        let d = descriptor(&r);
        let v = u32::MAX.to_le_bytes();
        r.dec
            .pass_default_raster_sample_count(&mut cursor(&v), &d)
            .unwrap();
        r.dec
            .pass_default_color_sample_count(&mut cursor(&v), &d)
            .unwrap();
        r.dec
            .pass_imageblock_sample_length(&mut cursor(&v), &d)
            .unwrap();
        assert_eq!(
            d.pass(|p| [
                p.default_raster_sample_count,
                p.default_color_sample_count,
                p.imageblock_sample_length
            ]),
            [Some(0xffff_ffff); 3]
        );
        assert_failed_read(
            r.dec
                .pass_default_raster_sample_count(&mut cursor(&[0; 3]), &d),
        );
        assert_failed_read(
            r.dec
                .pass_default_color_sample_count(&mut cursor(&[0; 3]), &d),
        );
        assert_failed_read(
            r.dec
                .pass_imageblock_sample_length(&mut cursor(&[0; 3]), &d),
        );
    }

    #[test]
    fn pass_threadgroup_memory_length() {
        let at = bytes(&CmdSetThreadGroupMemoryLength {
            threadgroup_memory_length: 0xffff_ffff,
        });
        let mut r = Rig::new();
        let d = descriptor(&r);
        assert_failed_read(
            r.dec
                .pass_threadgroup_memory_length(&mut cursor(&at[..3]), &d),
        );
        assert_eq!(d.pass(|p| p.threadgroup_memory_length), None);
        r.dec
            .pass_threadgroup_memory_length(&mut cursor(&at), &d)
            .unwrap();
        assert_eq!(d.pass(|p| p.threadgroup_memory_length), Some(0xffff_ffff));
    }

    #[test]
    fn pass_tile_size() {
        let at = bytes(&CmdSetTileParameters {
            tile_width: 0xffff,
            tile_height: 16,
        });
        let mut r = Rig::new();
        let d = descriptor(&r);
        assert_failed_read(r.dec.pass_tile_size(&mut cursor(&at[..3]), &d));
        assert_eq!(d.pass(|p| p.tile_width), None);
        r.dec.pass_tile_size(&mut cursor(&at), &d).unwrap();
        assert_eq!(
            d.pass(|p| (p.tile_width, p.tile_height)),
            (Some(0xffff), Some(16))
        );
    }
}
