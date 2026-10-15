//! `DeviceCaps`: everything core asks a device that does not change while the device lives,
//! read once per device by the platform ([`GpuDevice::caps`](super::device::GpuDevice::caps)).
//!
//! [`Limits`] and [`Flags`] hold one field per `NSUInteger` and `BOOL` property the device is
//! asked, named after its selector (`registryID` and `maxVertexAmplificationCount` are
//! [`DeviceCaps`]'s own fields). A query that can raise is kept as the `RaiseResult` it answered:
//! its accessor raises it again at the site that uses it, with the same identity and origin.

use super::types::{MtlSize, SizeAndAlign};
use crate::raise::{OnFault, RaiseResult};

/// The sample counts core asks `supportsTextureSampleCount:` about, in order.
pub const SAMPLE_COUNTS: [u64; 4] = [1, 2, 4, 8];
/// The primitive types core asks `supportsPrimitiveType:` about: `0..10`, raw.
pub const PRIMITIVE_TYPE_COUNT: u64 = 10;
/// The families core asks `supportsFamily:` about, in order: `MTLGPUFamilyApple5` to
/// `MTLGPUFamilyApple10`, then -1009, which no SDK family is.
pub const FAMILIES: [i64; 7] = [1005, 1006, 1007, 1008, 1009, 1010, -1009];
/// The modes core asks `supportsTextureWriteRoundingMode:` about, in order.
pub const WRITE_ROUNDING_MODES: [i64; 2] = [1, 2];
/// The storage mode of the heap-buffer size query: `MTLResourceStorageModePrivate`.
pub const HEAP_QUERY_OPTIONS: u64 = 0x20;
/// The pixel format of the linear-alignment query: `MTLPixelFormatBGRA8Unorm`.
pub const LINEAR_ALIGNMENT_FORMAT: u64 = 80;

/// Declares a struct with one `pub` field of `$ty` per property, named after its selector.
macro_rules! caps_struct {
    ($(#[$m:meta])* $name:ident: $ty:ty; $( $field:ident, )+) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        #[allow(missing_docs)] // each field is its selector's answer, named after it
        pub struct $name {
            $( pub $field: $ty, )+
        }
    };
}

caps_struct! {
    /// The device's `NSUInteger` properties, one field per selector.
    Limits: u64;
    device_linear_texture_alignment_bytes,
    iosurface_read_only_texture_alignment_bytes,
    iosurface_texture_alignment_bytes,
    linear_texture_alignment_bytes,
    max_compute_buffers,
    max_compute_local_memory_sizes,
    max_compute_samplers,
    max_compute_textures,
    max_compute_threadgroup_memory,
    max_compute_threadgroup_memory_alignment_bytes,
    max_fragment_buffers,
    max_fragment_samplers,
    max_fragment_textures,
    max_predicated_nesting_depth,
    max_rasterization_rate_layer_count,
    max_texture_height_2d,
    max_texture_layers,
    max_texture_width_2d,
    max_threadgroup_memory_length,
    max_tile_buffers,
    max_tile_samplers,
    max_tile_textures,
    max_total_compute_threads_per_threadgroup,
    max_vertex_buffers,
    max_vertex_samplers,
    max_vertex_textures,
    max_viewport_count,
}

caps_struct! {
    /// The device's `BOOL` properties, one field per selector.
    Flags: bool;
    are_raster_order_groups_supported,
    is_depth24_stencil8_pixel_format_supported,
    is_framebuffer_read_supported,
    is_large_mrt_supported,
    is_rgb10a2_gamma_supported,
    supports_buffer_with_address_ranges,
    supports_buffer_with_iosurface,
    supports_command_buffer_jump,
    supports_compute_compressed_texture_write,
    supports_conditional_load_store,
    supports_dynamic_attribute_stride,
    supports_extended_xr10_formats,
    supports_float16_bcubic_filtering,
    supports_heap_with_address_ranges,
    supports_image_blocks,
    supports_memory_order_atomics,
    supports_native_hardware_fp16,
    supports_non_square_tile_shaders,
    supports_programmable_sample_positions,
    supports_resource_detach_backing,
    supports_shared_texture_handles,
    supports_simd_reduction,
    supports_simd_shuffle_and_fill,
    supports_texture_2d_multisample_array,
    supports_tile_shaders,
}

/// Everything core asks a device that does not change while it lives.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceCaps {
    /// `name`; `None` for nil.
    pub name: Option<String>,
    /// `registryID`.
    pub registry_id: u64,
    /// `gpuCoreCount`; `None` when the device does not respond to it.
    pub gpu_core_count: Option<u64>,
    /// `maxThreadsPerThreadgroup`.
    pub max_threads_per_threadgroup: MtlSize,
    /// `maxVertexAmplificationCount`.
    pub max_vertex_amplification_count: u64,
    /// The other `NSUInteger` properties.
    pub limits: Limits,
    /// The `BOOL` properties.
    pub flags: Flags,
    /// `supportsTextureSampleCount:` of each of [`SAMPLE_COUNTS`].
    pub sample_counts: [RaiseResult<bool>; 4],
    /// `supportsPrimitiveType:` of `0..10`.
    pub primitive_types: [RaiseResult<bool>; 10],
    /// `supportsFamily:` of each of [`FAMILIES`].
    pub families: [RaiseResult<bool>; 7],
    /// `supportsTextureWriteRoundingMode:` of each of [`WRITE_ROUNDING_MODES`].
    pub write_rounding_modes: [RaiseResult<bool>; 2],
    /// `heapBufferSizeAndAlignWithLength:options:` of (`vm_page_size`, [`HEAP_QUERY_OPTIONS`]).
    pub heap_buffer_page_private: RaiseResult<SizeAndAlign>,
    /// `minimumLinearTextureAlignmentForPixelFormat:` of [`LINEAR_ALIGNMENT_FORMAT`].
    pub min_linear_alignment_bgra8: RaiseResult<u64>,
}

/// `r` again: its value, or its raise thrown here, as `on` says.
fn again<T: Clone>(on: OnFault, r: &RaiseResult<T>) -> RaiseResult<T> {
    match r {
        Ok(v) => Ok(v.clone()),
        Err(e) => on.throw(e.clone()),
    }
}

impl DeviceCaps {
    /// `heapBufferSizeAndAlignWithLength:options:` as read, its raise thrown again here as `on`
    /// says: the device info writer asks it both under a handler and with none.
    pub fn heap_buffer_page_private(&self, on: OnFault) -> RaiseResult<SizeAndAlign> {
        again(on, &self.heap_buffer_page_private)
    }

    /// `minimumLinearTextureAlignmentForPixelFormat:` as read, its raise thrown again here. Only
    /// the EFI display's registers, with no handler, ask it, so the raise dies here.
    pub fn min_linear_alignment_bgra8(&self) -> RaiseResult<u64> {
        again(OnFault::Die, &self.min_linear_alignment_bgra8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raise::Raise;
    use crate::testkit::impls::device::fake_caps;

    /// A raise kept in the caps is raised again where it is asked: the caller's under a handler,
    /// and a death with none (the alignment is asked only with none); a kept answer is the
    /// answer
    #[test]
    fn a_raise_is_kept_and_raised_again() {
        let raised = Raise::foreign("heapBufferSizeAndAlignWithLength:options:").with_detail("no");
        let caps = DeviceCaps {
            heap_buffer_page_private: Err(raised.clone()),
            min_linear_alignment_bgra8: Err(Raise::foreign(
                "minimumLinearTextureAlignmentForPixelFormat:",
            )
            .with_detail("refused")),
            ..fake_caps()
        };
        assert_eq!(caps.heap_buffer_page_private(OnFault::Return), Err(raised));
        assert_eq!(
            fake_caps().heap_buffer_page_private(OnFault::Die),
            Ok(SizeAndAlign {
                size: 0x4000,
                align: 0x4000
            })
        );
        let death = crate::testkit::capture::catch_death(|| {
            let _ = caps.min_linear_alignment_bgra8();
        });
        assert_eq!(
            death.map_err(|d| d.file),
            Err("minimumLinearTextureAlignmentForPixelFormat:")
        );
        let death = crate::testkit::capture::catch_death(|| {
            let _ = caps.heap_buffer_page_private(OnFault::Die);
        });
        assert!(death.is_err());
    }
}
