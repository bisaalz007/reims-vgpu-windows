//! `id<MTLComputeCommandEncoder>`: core's `ComputeEncoder`, whose commands (`ComputeCmd`) are
//! one exhaustive `match`, one typed call per variant, and its SPI control flow's `encodeEnd…`
//! calls, which answer a `BOOL`.

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBarrierScope, MTLCommandEncoder as _, MTLComputeCommandEncoder, MTLResourceUsage,
};
use reims_vgpu_core::num::ToUsize;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::compute::ComputeCmd;
use reims_vgpu_core::raise::{OnFault, RaiseResult};

use crate::gpu::{array, check_len, objects, resource};
use crate::msg::encode::ToMtl;
use crate::msg::{self, Policy};
use crate::platform::{MetalPlatform, Obj};
use crate::spi::methods::ComputeEncoderSpi;

/// `id<MTLComputeCommandEncoder>`.
pub type MetalComputeEncoder = Obj<ProtocolObject<dyn MTLComputeCommandEncoder>>;

impl reims_vgpu_core::platform::compute::ComputeEncoder<MetalPlatform> for MetalComputeEncoder {
    /// One call per command: a raise inside Metal is the caller's to catch. Every command but
    /// `EndEncoding` is encoded only under a handler (`Caught`); the resource flush also ends its
    /// encoder with none, so `EndEncoding` is `On(on)`, as the caller says. A `…:withRange:`
    /// command whose array is shorter than the range dies before the call ([`check_len`]). Raw
    /// guest values go to Metal unvalidated, which is what objc2-metal's `unsafe` marks: Metal
    /// checks them itself, and raises or faults the GPU on what it refuses; each C array has at
    /// least as many elements as its count or range.
    fn encode(&self, on: OnFault, cmd: ComputeCmd<'_, MetalPlatform>) -> RaiseResult<()> {
        let e = self.get();
        // SAFETY (every `unsafe` call below): as above.
        match cmd {
            ComputeCmd::DispatchThreadgroups {
                threadgroups_per_grid,
                threads_per_threadgroup,
            } => {
                let (grid, group) = (threadgroups_per_grid.mtl(), threads_per_threadgroup.mtl());
                msg::send(
                    "dispatchThreadgroups:threadsPerThreadgroup:",
                    Policy::Caught,
                    || e.dispatchThreadgroups_threadsPerThreadgroup(grid, group),
                )
            }
            ComputeCmd::DispatchThreadgroupsIndirect {
                indirect_buffer,
                indirect_buffer_offset,
                threads_per_threadgroup,
            } => {
                let (buffer, group) =
                    (indirect_buffer.map(Obj::get), threads_per_threadgroup.mtl());
                msg::send(
                    "dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:",
                    Policy::Caught,
                    || {
                        ComputeEncoderSpi::dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                        e,
                        buffer,
                        indirect_buffer_offset,
                        group,
                    )
                    },
                )
            }
            ComputeCmd::DispatchThreads {
                threads_per_grid,
                threads_per_threadgroup,
            } => {
                let (grid, group) = (threads_per_grid.mtl(), threads_per_threadgroup.mtl());
                msg::send(
                    "dispatchThreads:threadsPerThreadgroup:",
                    Policy::Caught,
                    || e.dispatchThreads_threadsPerThreadgroup(grid, group),
                )
            }
            ComputeCmd::DispatchThreadsIndirect {
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                let buffer = indirect_buffer.map(Obj::get);
                msg::send(
                    "dispatchThreadsWithIndirectBuffer:indirectBufferOffset:",
                    Policy::Caught,
                    || {
                        e.dispatchThreadsWithIndirectBuffer_indirectBufferOffset(
                            buffer,
                            indirect_buffer_offset,
                        )
                    },
                )
            }
            ComputeCmd::EncodeStartDoWhile => {
                msg::send("encodeStartDoWhile", Policy::Caught, || {
                    e.encodeStartDoWhile()
                })
            }
            ComputeCmd::EncodeStartElse => {
                msg::send("encodeStartElse", Policy::Caught, || e.encodeStartElse())
            }
            ComputeCmd::EncodeStartIf {
                buffer,
                offset,
                comparison,
                reference_value,
            } => {
                let buffer = buffer.map(Obj::get);
                msg::send(
                    "encodeStartIf:offset:comparison:referenceValue:",
                    Policy::Caught,
                    || {
                        e.encodeStartIf_offset_comparison_referenceValue(
                            buffer,
                            offset,
                            comparison,
                            reference_value,
                        )
                    },
                )
            }
            ComputeCmd::EncodeStartWhile {
                buffer,
                offset,
                comparison,
                reference_value,
            } => {
                let buffer = buffer.map(Obj::get);
                msg::send(
                    "encodeStartWhile:offset:comparison:referenceValue:",
                    Policy::Caught,
                    || {
                        e.encodeStartWhile_offset_comparison_referenceValue(
                            buffer,
                            offset,
                            comparison,
                            reference_value,
                        )
                    },
                )
            }
            ComputeCmd::EndEncoding => msg::send("endEncoding", Policy::On(on), || e.endEncoding()),
            ComputeCmd::ExecuteCommandsIndirect {
                indirect_commandbuffer,
                indirect_range_buffer,
                indirect_buffer_offset,
            } => {
                let (icb, buffer) = (indirect_commandbuffer.get(), indirect_range_buffer.get());
                let offset = indirect_buffer_offset.to_usize();
                msg::send(
                    "executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:",
                    Policy::Caught,
                    || unsafe {
                        e.executeCommandsInBuffer_indirectBuffer_indirectBufferOffset(
                            icb, buffer, offset,
                        )
                    },
                )
            }
            ComputeCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer,
                execution_range,
            } => {
                let (icb, range) = (indirect_command_buffer.get(), execution_range.mtl());
                msg::send(
                    "executeCommandsInBuffer:withRange:",
                    Policy::Caught,
                    || unsafe { e.executeCommandsInBuffer_withRange(icb, range) },
                )
            }
            ComputeCmd::CompressedTextureReinterpretationFlush => msg::send(
                "insertCompressedTextureReinterpretationFlush",
                Policy::Caught,
                || e.insertCompressedTextureReinterpretationFlush(),
            ),
            ComputeCmd::MemoryBarrierWithResources { resources } => {
                let resources = objects(resources.iter().map(|r| r.map(resource)));
                let (list, count) = (array(&resources), resources.len());
                msg::send(
                    "memoryBarrierWithResources:count:",
                    Policy::Caught,
                    || unsafe { e.memoryBarrierWithResources_count(list, count) },
                )
            }
            ComputeCmd::MemoryBarrierWithScope { scope } => {
                let scope = MTLBarrierScope(scope.to_usize());
                msg::send("memoryBarrierWithScope:", Policy::Caught, || {
                    e.memoryBarrierWithScope(scope)
                })
            }
            ComputeCmd::SetBufferOffset { offset, index } => {
                let (offset, index) = (offset.to_usize(), index.to_usize());
                msg::send("setBufferOffset:atIndex:", Policy::Caught, || unsafe {
                    e.setBufferOffset_atIndex(offset, index)
                })
            }
            ComputeCmd::SetBufferOffsetAttributeStride {
                offset,
                stride,
                index,
            } => {
                let (offset, stride, index) =
                    (offset.to_usize(), stride.to_usize(), index.to_usize());
                msg::send(
                    "setBufferOffset:attributeStride:atIndex:",
                    Policy::Caught,
                    || unsafe { e.setBufferOffset_attributeStride_atIndex(offset, stride, index) },
                )
            }
            ComputeCmd::SetBuffersAttributeStrides {
                buffers,
                offsets,
                strides,
                range,
            } => {
                let sel = "setBuffers:offsets:attributeStrides:withRange:";
                check_len(sel, strides.len(), range);
                check_len(sel, offsets.len(), range);
                check_len(sel, buffers.len(), range);
                let buffers = objects(buffers.iter().map(|o| o.map(Obj::get)));
                let (list, offsets, strides) = (array(&buffers), array(offsets), array(strides));
                let range = range.mtl();
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setBuffers_offsets_attributeStrides_withRange(list, offsets, strides, range)
                })
            }
            ComputeCmd::SetBuffers {
                buffers,
                offsets,
                range,
            } => {
                let sel = "setBuffers:offsets:withRange:";
                check_len(sel, offsets.len(), range);
                check_len(sel, buffers.len(), range);
                let buffers = objects(buffers.iter().map(|o| o.map(Obj::get)));
                let (list, offsets, range) = (array(&buffers), array(offsets), range.mtl());
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setBuffers_offsets_withRange(list, offsets, range)
                })
            }
            ComputeCmd::SetComputePipelineState { state } => {
                let state = state.map(Obj::get);
                msg::send("setComputePipelineState:", Policy::Caught, || {
                    ComputeEncoderSpi::setComputePipelineState(e, state)
                })
            }
            ComputeCmd::SetImageblockWidth { width, height } => {
                let (width, height) = (width.to_usize(), height.to_usize());
                msg::send("setImageblockWidth:height:", Policy::Caught, || {
                    e.setImageblockWidth_height(width, height)
                })
            }
            ComputeCmd::SetSamplerStatesLodClamps {
                samplers,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => {
                let sel = "setSamplerStates:lodMinClamps:lodMaxClamps:withRange:";
                check_len(sel, lod_max_clamps.len(), range);
                check_len(sel, lod_min_clamps.len(), range);
                check_len(sel, samplers.len(), range);
                let samplers = objects(samplers.iter().map(|o| o.map(Obj::get)));
                let (list, min, max) = (
                    array(&samplers),
                    array(lod_min_clamps),
                    array(lod_max_clamps),
                );
                let range = range.mtl();
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setSamplerStates_lodMinClamps_lodMaxClamps_withRange(list, min, max, range)
                })
            }
            ComputeCmd::SetSamplerStates { samplers, range } => {
                let sel = "setSamplerStates:withRange:";
                check_len(sel, samplers.len(), range);
                let samplers = objects(samplers.iter().map(|o| o.map(Obj::get)));
                let (list, range) = (array(&samplers), range.mtl());
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setSamplerStates_withRange(list, range)
                })
            }
            ComputeCmd::SetStageInRegion { region } => {
                let region = region.mtl();
                msg::send("setStageInRegion:", Policy::Caught, || {
                    e.setStageInRegion(region)
                })
            }
            ComputeCmd::SetStageInRegionIndirect {
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                let (buffer, offset) = (indirect_buffer.get(), indirect_buffer_offset.to_usize());
                msg::send(
                    "setStageInRegionWithIndirectBuffer:indirectBufferOffset:",
                    Policy::Caught,
                    || unsafe {
                        e.setStageInRegionWithIndirectBuffer_indirectBufferOffset(buffer, offset)
                    },
                )
            }
            ComputeCmd::SetTexture { texture, index } => {
                let (texture, index) = (texture.map(Obj::get), index.to_usize());
                msg::send("setTexture:atIndex:", Policy::Caught, || unsafe {
                    e.setTexture_atIndex(texture, index)
                })
            }
            ComputeCmd::SetTextures { textures, range } => {
                let sel = "setTextures:withRange:";
                check_len(sel, textures.len(), range);
                let textures = objects(textures.iter().map(|o| o.map(Obj::get)));
                let (list, range) = (array(&textures), range.mtl());
                msg::send(sel, Policy::Caught, || unsafe {
                    e.setTextures_withRange(list, range)
                })
            }
            ComputeCmd::SetThreadgroupMemoryLength { length, index } => {
                let (length, index) = (length.to_usize(), index.to_usize());
                msg::send(
                    "setThreadgroupMemoryLength:atIndex:",
                    Policy::Caught,
                    || unsafe { e.setThreadgroupMemoryLength_atIndex(length, index) },
                )
            }
            ComputeCmd::UpdateFence { fence } => {
                let fence = fence.get();
                msg::send("updateFence:", Policy::Caught, || e.updateFence(fence))
            }
            ComputeCmd::UseHeaps { heaps } => {
                let heaps = objects(heaps.iter().map(|o| o.map(Obj::get)));
                let (list, count) = (array(&heaps), heaps.len());
                msg::send("useHeaps:count:", Policy::Caught, || unsafe {
                    e.useHeaps_count(list, count)
                })
            }
            ComputeCmd::UseResources { resources, usage } => {
                let resources = objects(resources.iter().map(|r| r.map(resource)));
                let (list, count) = (array(&resources), resources.len());
                let usage = MTLResourceUsage(usage.to_usize());
                msg::send("useResources:count:usage:", Policy::Caught, || unsafe {
                    e.useResources_count_usage(list, count, usage)
                })
            }
            ComputeCmd::WaitForFence { fence } => {
                let fence = fence.get();
                msg::send("waitForFence:", Policy::Caught, || e.waitForFence(fence))
            }
        }
    }
    /// `encodeEndDoWhile:offset:comparison:referenceValue:` (SPI).
    fn encode_end_do_while(
        &self,
        buffer: Option<&<MetalPlatform as Platform>::Buffer>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    ) -> RaiseResult<bool> {
        let (e, buffer) = (self.get(), buffer.map(Obj::get));
        msg::send(
            "encodeEndDoWhile:offset:comparison:referenceValue:",
            Policy::Caught,
            || {
                e.encodeEndDoWhile_offset_comparison_referenceValue(
                    buffer,
                    offset,
                    comparison,
                    reference_value,
                )
            },
        )
    }
    /// `encodeEndIf` (SPI).
    fn encode_end_if(&self) -> RaiseResult<bool> {
        let e = self.get();
        msg::send("encodeEndIf", Policy::Caught, || e.encodeEndIf())
    }
    /// `encodeEndWhile` (SPI).
    fn encode_end_while(&self) -> RaiseResult<bool> {
        let e = self.get();
        msg::send("encodeEndWhile", Policy::Caught, || e.encodeEndWhile())
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::raise::OnFault;

    use reims_vgpu_core::platform::command::CommandBuffer as _;
    use reims_vgpu_core::platform::compute::*;
    use reims_vgpu_core::platform::resource::Buffer as _;
    use reims_vgpu_core::platform::types::*;
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::testkit::capture::catch_death;

    use reims_vgpu_core::platform::blit::ResourceRef;

    use crate::gpu::resource::testkit::*;

    /// Every ComputeEncoder method, pipeline first, endEncoding last
    #[test]
    fn compute_encoder_sends() {
        let k = kit();
        let cb = command_buffer(&k.queue);
        let x = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::SetComputePipelineState {
                state: Some(&k.compute_pipeline),
            },
        );
        assert!(r.is_ok(), "set_compute_pipeline_state: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::DispatchThreadgroups {
                threadgroups_per_grid: MtlSize::new(1, 1, 1),
                threads_per_threadgroup: MtlSize::new(1, 1, 1),
            },
        );
        assert!(r.is_ok(), "dispatch_threadgroups: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::DispatchThreadgroupsIndirect {
                indirect_buffer: Some(&k.buffer),
                indirect_buffer_offset: 0,
                threads_per_threadgroup: MtlSize::new(1, 1, 1),
            },
        );
        assert!(
            r.is_ok(),
            "dispatch_threadgroups_with_indirect_buffer: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(1, 1, 1),
                threads_per_threadgroup: MtlSize::new(1, 1, 1),
            },
        );
        assert!(r.is_ok(), "dispatch_threads: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::DispatchThreadsIndirect {
                indirect_buffer: Some(&k.buffer),
                indirect_buffer_offset: 0,
            },
        );
        assert!(r.is_ok(), "dispatch_threads_with_indirect_buffer: {r:?}");
        let r = x.encode_end_do_while(Some(&k.buffer), 0, 0, 0);
        assert!(r.is_ok(), "encode_end_do_while: {r:?}");
        let r = x.encode_end_if();
        assert!(r.is_ok(), "encode_end_if: {r:?}");
        let r = x.encode_end_while();
        assert!(r.is_ok(), "encode_end_while: {r:?}");
        let r = x.encode(OnFault::Return, ComputeCmd::EncodeStartDoWhile);
        assert!(r.is_ok(), "encode_start_do_while: {r:?}");
        let r = x.encode(OnFault::Return, ComputeCmd::EncodeStartElse);
        assert!(r.is_ok(), "encode_start_else: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::EncodeStartIf {
                buffer: Some(&k.buffer),
                offset: 0,
                comparison: 0,
                reference_value: 0,
            },
        );
        assert!(r.is_ok(), "encode_start_if: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::EncodeStartWhile {
                buffer: Some(&k.buffer),
                offset: 0,
                comparison: 0,
                reference_value: 0,
            },
        );
        assert!(r.is_ok(), "encode_start_while: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: &k.icb_compute,
                execution_range: NsRange::new(0, 1),
            },
        );
        assert!(r.is_ok(), "execute_commands_in_buffer: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::ExecuteCommandsIndirect {
                indirect_commandbuffer: &k.icb_compute,
                indirect_range_buffer: &k.buffer,
                indirect_buffer_offset: 0,
            },
        );
        assert!(
            r.is_ok(),
            "execute_commands_in_buffer_indirect_buffer_indirect_buffer_offset: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::CompressedTextureReinterpretationFlush,
        );
        assert!(
            r.is_ok(),
            "insert_compressed_texture_reinterpretation_flush: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::MemoryBarrierWithResources {
                resources: &[Some(ResourceRef::Buffer(&k.buffer))],
            },
        );
        assert!(r.is_ok(), "memory_barrier_with_resources: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::MemoryBarrierWithScope { scope: 0 },
        );
        assert!(r.is_ok(), "memory_barrier_with_scope: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::SetBufferOffset {
                offset: 0,
                index: 0,
            },
        );
        assert!(r.is_ok(), "set_buffer_offset: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::SetBufferOffsetAttributeStride {
                offset: 0,
                stride: 16,
                index: 0,
            },
        );
        assert!(r.is_ok(), "set_buffer_offset_attribute_stride: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::SetBuffers {
                buffers: &[Some(&k.buffer)],
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
        );
        assert!(r.is_ok(), "set_buffers: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::SetBuffersAttributeStrides {
                buffers: &[Some(&k.buffer)],
                offsets: &[0],
                strides: &[0],
                range: NsRange::new(0, 1),
            },
        );
        assert!(r.is_ok(), "set_buffers_attribute_strides: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::SetImageblockWidth {
                width: 1,
                height: 1,
            },
        );
        assert!(r.is_ok(), "set_imageblock_width: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::SetSamplerStates {
                samplers: &[Some(&k.sampler)],
                range: NsRange::new(0, 1),
            },
        );
        assert!(r.is_ok(), "set_sampler_states: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::SetSamplerStatesLodClamps {
                samplers: &[Some(&k.sampler)],
                lod_min_clamps: &[0.0],
                lod_max_clamps: &[0.0],
                range: NsRange::new(0, 1),
            },
        );
        assert!(
            r.is_ok(),
            "set_sampler_states_lod_min_clamps_lod_max_clamps: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::SetStageInRegion {
                region: MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(1, 1, 1)),
            },
        );
        assert!(r.is_ok(), "set_stage_in_region: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::SetStageInRegionIndirect {
                indirect_buffer: &k.buffer,
                indirect_buffer_offset: 0,
            },
        );
        assert!(r.is_ok(), "set_stage_in_region_with_indirect_buffer: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::SetTexture {
                texture: Some(&k.texture),
                index: 0,
            },
        );
        assert!(r.is_ok(), "set_texture: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::SetTextures {
                textures: &[Some(&k.texture)],
                range: NsRange::new(0, 1),
            },
        );
        assert!(r.is_ok(), "set_textures: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::SetThreadgroupMemoryLength {
                length: 16,
                index: 0,
            },
        );
        assert!(r.is_ok(), "set_threadgroup_memory_length: {r:?}");
        let r = x.encode(OnFault::Return, ComputeCmd::UpdateFence { fence: &k.fence });
        assert!(r.is_ok(), "update_fence: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::UseHeaps {
                heaps: &[Some(&k.heap)],
            },
        );
        assert!(r.is_ok(), "use_heaps: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::UseResources {
                resources: &[Some(ResourceRef::Buffer(&k.buffer))],
                usage: 1,
            },
        );
        assert!(r.is_ok(), "use_resources: {r:?}");
        let r = x.encode(
            OnFault::Return,
            ComputeCmd::WaitForFence { fence: &k.fence },
        );
        assert!(r.is_ok(), "wait_for_fence: {r:?}");
        let r = x.encode(OnFault::Return, ComputeCmd::EndEncoding);
        assert!(r.is_ok(), "end_encoding: {r:?}");
    }

    /// setComputePipelineState:, setBuffers:offsets:withRange: and
    /// dispatchThreads:threadsPerThreadgroup: run the kernel over the bound buffer
    #[test]
    fn dispatch_runs_the_kernel() {
        let k = kit();
        let out = buffer(&k.device, 64 * 4);
        let cb = command_buffer(&k.queue);
        let x = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        x.encode(
            OnFault::Return,
            ComputeCmd::SetComputePipelineState {
                state: Some(&k.compute_pipeline),
            },
        )
        .unwrap();
        x.encode(
            OnFault::Return,
            ComputeCmd::SetBuffers {
                buffers: &[Some(&out)],
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
        )
        .unwrap();
        x.encode(
            OnFault::Return,
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(64, 1, 1),
                threads_per_threadgroup: MtlSize::new(32, 1, 1),
            },
        )
        .unwrap();
        x.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        // SAFETY: a live shared buffer of 64 u32s.
        let o = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u32>(), 64) };
        assert_eq!(o[0], 0);
        assert_eq!(o[63], 63);
    }

    /// A `…:withRange:` send given an array shorter than the range dies at the send (OutOfBounds,
    /// file = the selector) and sends nothing
    #[test]
    fn short_arrays_die_before_the_send() {
        let k = kit();
        let cb = command_buffer(&k.queue);
        let x = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
        let site = catch_death(|| {
            let _ = x.encode(
                OnFault::Die,
                ComputeCmd::SetBuffers {
                    buffers: &[Some(&k.buffer)],
                    offsets: &[],
                    range: NsRange::new(0, 1),
                },
            );
        })
        .unwrap_err();
        assert_eq!(site.kind, DeathKind::OutOfBounds);
        assert_eq!((site.file, site.line), ("setBuffers:offsets:withRange:", 0));
        let site = catch_death(|| {
            let _ = x.encode(
                OnFault::Die,
                ComputeCmd::SetTextures {
                    textures: &[None],
                    range: NsRange::new(1, 2),
                },
            );
        })
        .unwrap_err();
        assert_eq!(site.file, "setTextures:withRange:");
        // A longer array is fine: Metal reads `range.length` entries.
        {
            x.encode(
                OnFault::Return,
                ComputeCmd::SetTextures {
                    textures: &[None, None, None],
                    range: NsRange::new(0, 2),
                },
            )
            .unwrap();
            x.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        }
    }
}
