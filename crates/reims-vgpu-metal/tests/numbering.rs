//! Golden test: the numbers Metal hands out in a fresh process, through the crate's creators.
//!
//! GPU addresses, resource IDs and resource indices reach the guest, and Metal numbers its
//! objects in creation order: an allocation or a cache of the crate's own would move every number
//! after it. This binary holds one test, so it runs in a fresh process, and it creates a fixed
//! sequence of objects through `GpuDevice` and the resource traits, the heap, the sampler and the
//! texture descriptors from values as core sends them. The vector it reads is compared with
//! [`EXPECTED`], the vector recorded from this crate for the same sequence (before the values,
//! from descriptors made directly: the numbers did not move).
//!
//! Set `RVG_NUMBERING_PRINT=1` to print the vector instead of comparing it.

#![cfg(target_os = "macos")] // Metal is macOS's: the crate is empty elsewhere
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    missing_docs
)]

use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::resource::{Buffer, Heap, SamplerState, Texture};
use reims_vgpu_core::platform::values::{HeapDesc, OptionSet, SamplerDesc, TextureDesc};
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_metal::MetalPlatform;
use reims_vgpu_metal::gpu::desc::{MetalTextureDescriptor, texture_descriptor};
use reims_vgpu_metal::gpu::device::MetalDevice;

/// The recorded numbering of [`sequence`] in a fresh process: the addresses, IDs and indices the
/// guest would see for these objects. Recorded on `AppleM3Max-40c-25G229`, the same in three runs.
/// An OS update may move Metal's numbers, and a change to the crate's creators may move them on
/// purpose: then record it again with `RVG_NUMBERING_PRINT=1` and review the difference.
const EXPECTED: &[(&str, u64)] = &[
    ("buffer.gpuAddress", 0x1000001c500),
    ("private buffer.gpuAddress", 0x10000080000),
    ("texture.gpuResourceID", 0x1),
    ("texture.resourceIndex", 0x1),
    ("view.gpuResourceID", 0x2),
    ("view.resourceIndex", 0x2),
    ("heap.gpuAddress", 0x10000190000),
    ("heap buffer.gpuAddress", 0x10000190000),
    ("sampler.gpuResourceID", 0x2),
    ("sampler.resourceIndex", 0x2),
    ("shared texture.gpuResourceID", 0x3),
    ("shared texture.resourceIndex", 0x3),
    ("last buffer.gpuAddress", 0x1000001d500),
];

/// A 64x64 BGRA8Unorm private texture's descriptor, read by shaders and viewed in other formats.
fn private_texture() -> MetalTextureDescriptor {
    texture_descriptor(&TextureDesc {
        width: 64,
        height: 64,
        pixel_format: 80,
        usage: 0x11,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    })
}

/// The fixed sequence: every creator whose objects' numbers reach the guest, in one order, and
/// what each object's numbers read.
fn sequence(d: &MetalDevice) -> Vec<(&'static str, u64)> {
    let mut v = Vec::new();
    let _queue = d.new_command_queue().unwrap();
    let b1 = d
        .new_buffer_of_length(OnFault::Die, 4096, 0)
        .unwrap()
        .unwrap();
    v.push(("buffer.gpuAddress", b1.gpu_address()));
    let b2 = d
        .new_buffer_of_length(OnFault::Die, 1 << 20, 0x20)
        .unwrap()
        .unwrap();
    v.push(("private buffer.gpuAddress", b2.gpu_address()));
    let t = d
        .new_texture(OnFault::Die, &private_texture())
        .unwrap()
        .unwrap();
    v.push(("texture.gpuResourceID", t.gpu_resource_id()));
    v.push(("texture.resourceIndex", t.resource_index()));
    let view = t.new_texture_view(81).unwrap().unwrap();
    v.push(("view.gpuResourceID", view.gpu_resource_id()));
    v.push(("view.resourceIndex", view.resource_index()));
    // A private placement heap.
    let heap = d
        .new_heap(&HeapDesc {
            size: 1 << 20,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(2)],
            address_ranges: None,
            pinned_gpu_address: None,
        })
        .unwrap()
        .heap
        .unwrap();
    v.push(("heap.gpuAddress", heap.gpu_address()));
    let hb = heap.new_buffer_of_length(4096, 0x20, 0).unwrap().unwrap();
    v.push(("heap buffer.gpuAddress", hb.gpu_address()));
    let s = d
        .new_sampler_state(&SamplerDesc {
            support_argument_buffers: Some(true),
            ..SamplerDesc::default()
        })
        .unwrap()
        .unwrap();
    v.push(("sampler.gpuResourceID", s.gpu_resource_id()));
    v.push(("sampler.resourceIndex", s.resource_index()));
    let shared = d.new_shared_texture(&private_texture()).unwrap().unwrap();
    v.push(("shared texture.gpuResourceID", shared.gpu_resource_id()));
    v.push(("shared texture.resourceIndex", shared.resource_index()));
    let b3 = d
        .new_buffer_of_length(OnFault::Die, 4096, 0)
        .unwrap()
        .unwrap();
    v.push(("last buffer.gpuAddress", b3.gpu_address()));
    v
}

#[test]
fn a_fresh_process_numbers_as_recorded() {
    let d = MetalPlatform::system_default_device().expect("a Metal device");
    let v = sequence(&d);
    if std::env::var_os("RVG_NUMBERING_PRINT").is_some() {
        for (what, n) in &v {
            println!("    (\"{what}\", {n:#x}),");
        }
        return;
    }
    assert_eq!(v, EXPECTED);
}
