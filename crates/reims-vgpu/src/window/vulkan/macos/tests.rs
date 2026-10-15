//! The Vulkan platform's window sink on macOS: the swapchain's image of a frame is what the
//! display's compose draws into a texture of the window's size, with no copy between.

use std::ffi::c_void;
use std::sync::Arc;

use ash::vk;
use objc2::rc::Retained;
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_metal::MTLTextureUsage;
use objc2_quartz_core::CAMetalLayer;
use reims_vgpu_core::display::{GammaTableArgs, MINIMUM_TEXTURE_USAGE};
use reims_vgpu_core::num::ToU64 as _;
use reims_vgpu_core::platform::descriptor::TextureDescriptor as _;
use reims_vgpu_core::platform::device::GpuDevice as _;
use reims_vgpu_core::platform::resource::Texture as _;
use reims_vgpu_core::platform::types::{MtlOrigin, MtlRegion, MtlSize};
use reims_vgpu_core::platform::{Handle as _, HostRange, Platform};
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_core::rt::Hold;
use reims_vgpu_core::tasks::isolation::TaskThreads;
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::swapchain::Swapchain;
use reims_vgpu_window::{PixelRect, VulkanTarget};

use super::super::swapchain::tests::{draw_and_read_back, read_back};
use crate::PixelFormat;
use crate::display::no_catch;
use crate::entry;
use crate::platform::api_state;
use crate::window::{SinkDisplay, WindowFrame};
use crate::{Device, DeviceConfig, Display, DisplayConfig};

/// The platform's command queue.
type Queue = <VulkanPlatform as Platform>::CommandQueue;
/// Its command buffer.
type CommandBuffer = <VulkanPlatform as Platform>::CommandBuffer;
/// Its texture.
type Texture = <VulkanPlatform as Platform>::Texture;

/// `MTLResourceStorageModeShared`, with the default CPU cache and hazard tracking modes.
const SHARED: u64 = 0;

/// `MTLResourceStorageModePrivate`: a drawable's storage.
const PRIVATE: u64 = 0x20;

/// The frame queue of `device`'s Metal device.
fn frame_queue(device: &Device<VulkanPlatform>) -> Queue {
    entry::run(|_| {
        api_state(device.platform())
            .frame_queue(device.core().mtl_device_ref())
            .unwrap()
    })
}

/// A Vulkan device on a platform of its own, and a display on its port 0 that takes
/// presents: one with a new-frame callback.
fn presenting_display(serial: u32) -> (Device<VulkanPlatform>, Display<VulkanPlatform>) {
    let platform = Arc::new(VulkanPlatform::new(Box::new(TaskThreads::new())));
    let device = Device::with_platform(platform, DeviceConfig::default())
        .expect("a Vulkan device (MoltenVK)");
    let mut c = DisplayConfig::new(0, serial);
    c.callbacks.new_frame = Some(Arc::new(|| {}));
    let display = device.new_display(c).unwrap();
    (device, display)
}

/// Presents to `display` a frame of `(width, height)`, BGRA, of every colour channel a
/// gradient of its own, opaque: what the window draws next.
fn present_a_fixed_frame(
    device: &Device<VulkanPlatform>,
    display: &Display<VulkanPlatform>,
    (width, height): (u64, u64),
) {
    let bytes: Vec<u8> = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .flat_map(|(x, y)| {
            let cut = |v: u64| u8::try_from(v % 256).unwrap();
            [
                cut(x * 255 / (width - 1)),
                cut(y * 255 / (height - 1)),
                cut((x * 7 + y * 13) % 256),
                0xff,
            ]
        })
        .collect();
    entry::run(|_| {
        let format = u64::from(PixelFormat::Bgra8Unorm.metal());
        let td = no_catch(device.platform().texture_2d_descriptor(
            OnFault::Die,
            format,
            width,
            height,
            false,
        ));
        td.set_usage(MTLTextureUsage::ShaderRead.0.to_u64());
        td.set_resource_options(SHARED);
        let frame = no_catch(
            device
                .core()
                .mtl_device_ref()
                .new_texture(OnFault::Die, &td),
        )
        .unwrap();
        no_catch(frame.replace_region(
            MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(width, height, 1)),
            0,
            0,
            HostRange::of_slice(&bytes),
            width * 4,
            width * 4 * height,
        ));
        no_catch(display.core().present_frame(
            Hold::NONE,
            OnFault::Die,
            Some(frame),
            u32::from_be_bytes(*b"BGRA"),
            0,
            None,
            GammaTableArgs::default(),
            None,
            Box::new(|| {}),
        ));
    });
}

/// A `CAMetalLayer` of `(width, height)` pixels at a scale of 1, in no window: what a
/// swapchain over it sizes its images by.
fn a_layer((width, height): (u64, u64)) -> Retained<CAMetalLayer> {
    let layer = CAMetalLayer::new();
    layer.setBounds(CGRect::new(
        CGPoint::new(0.0, 0.0),
        CGSize::new(
            f64::from(u32::try_from(width).unwrap()),
            f64::from(u32::try_from(height).unwrap()),
        ),
    ));
    layer.setContentsScale(1.0);
    layer
}

/// A layer as the window's render thread gives it: a surface of it is made with
/// `VK_EXT_metal_surface` (as `reims_vgpu_window`'s own target makes it).
struct LayerTarget<'a>(&'a CAMetalLayer);

impl VulkanTarget for LayerTarget<'_> {
    fn create_surface(
        &self,
        get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
        instance: vk::Instance,
    ) -> Option<vk::SurfaceKHR> {
        // SAFETY: a live instance's look-up of an instance command by its name.
        let found =
            unsafe { get_instance_proc_addr(instance, c"vkCreateMetalSurfaceEXT".as_ptr()) }?;
        // SAFETY: the command the loader answers for the name.
        let create: vk::PFN_vkCreateMetalSurfaceEXT = unsafe { std::mem::transmute(found) };
        let layer: *const c_void = std::ptr::from_ref(self.0).cast();
        let info = vk::MetalSurfaceCreateInfoEXT::default().layer(layer);
        let mut surface = vk::SurfaceKHR::null();
        // SAFETY: a valid create info naming a live layer, which outlives the surface.
        let made = unsafe { create(instance, &info, std::ptr::null(), &mut surface) };
        (made == vk::Result::SUCCESS).then_some(surface)
    }
}

/// A frame drawn into a texture of the test's own, as a sink draws one into its target: the
/// command buffer it was encoded in, once committed.
struct Plain<'a> {
    /// The texture.
    texture: &'a Texture,
    /// The frame's command buffer.
    committed: Option<CommandBuffer>,
}

impl WindowFrame<VulkanPlatform> for &mut Plain<'_> {
    fn texture(&mut self) -> Option<&Texture> {
        Some(self.texture)
    }

    fn encoded(&mut self, command_buffer: &CommandBuffer) {
        self.committed = Some(command_buffer.clone_ref());
    }
}

/// What the display's compose draws of its current frame over `region` into a private BGRA
/// texture of `size`, a drawable's format and storage, read back.
fn composed(
    device: &Device<VulkanPlatform>,
    display: &Display<VulkanPlatform>,
    queue: &Queue,
    size: (u64, u64),
    region: PixelRect,
) -> Vec<u8> {
    let texture = entry::run(|_| {
        let format = u64::from(PixelFormat::Bgra8Unorm.metal());
        let td = no_catch(device.platform().texture_2d_descriptor(
            OnFault::Die,
            format,
            size.0,
            size.1,
            false,
        ));
        td.set_usage(MINIMUM_TEXTURE_USAGE);
        td.set_resource_options(PRIVATE);
        no_catch(
            device
                .core()
                .mtl_device_ref()
                .new_texture(OnFault::Die, &td),
        )
        .unwrap()
    });
    let sink = SinkDisplay::new(display.core().downgrade(), queue.clone_ref());
    let mut plain = Plain {
        texture: &texture,
        committed: None,
    };
    assert!(sink.encode(region, &mut plain), "the frame is drawn");
    assert!(plain.committed.is_some());
    read_back(queue, &texture, size)
}

/// The swapchain's image of a frame is, byte for byte, what the display's compose draws into a
/// texture of the window's size and format: a fixed frame scaled into a region of the window,
/// its rest cleared, so the same format, colour and scaling, with no copy between. (The staging
/// sink the swapchain replaced drew the same bytes: the comparison of the two passed before it
/// went.) The frame is one the display draws visibly, not all the clear
#[test]
fn the_swapchain_holds_the_composed_frame() {
    let (device, display) = presenting_display(0x5eed_7b70);
    present_a_fixed_frame(&device, &display, (64, 40));
    let size = (61, 37);
    let region = PixelRect {
        x: 4,
        y: 3,
        width: 50,
        height: 30,
    };
    let queue = frame_queue(&device);
    let theirs = composed(&device, &display, &queue, size, region);

    let layer = a_layer(size);
    let made = entry::run(|_| {
        // SAFETY: the target makes the surface on the instance it is given, and hands it over;
        // the layer outlives the swapchain.
        unsafe {
            Swapchain::over(&queue, |get_instance_proc_addr, instance| {
                LayerTarget(&layer).create_surface(get_instance_proc_addr, instance)
            })
        }
    });
    let mut swapchain = made.unwrap();
    let ours = draw_and_read_back(
        display.core().downgrade(),
        &queue,
        &mut swapchain,
        size,
        region,
    );
    entry::run(|_| drop(swapchain));
    drop(layer);

    let black = [0, 0, 0, 0xff];
    assert!(
        theirs.chunks(4).any(|p| p != black),
        "the frame shows: not all the clear"
    );
    assert_eq!(ours.len(), theirs.len());
    let first = (ours.chunks(4).zip(theirs.chunks(4)))
        .position(|(a, b)| a != b)
        .map(|i| {
            (
                i % 61,
                i / 61,
                &ours[i * 4..i * 4 + 4],
                &theirs[i * 4..i * 4 + 4],
            )
        });
    assert_eq!(
        first, None,
        "the first pixel that differs: (x, y, ours, theirs)"
    );
}
