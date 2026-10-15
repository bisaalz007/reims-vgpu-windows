//! Metal's C functions: `MTLPixelFormatGetInfoForDevice` and
//! `MTLGetTextureLevelInfoForDeviceWithOptions`. A call goes
//! through `msg::get`, which catches a raise there as it does in a method.
//!
//! The declarations take the widths core's calls pass.

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use reims_vgpu_core::platform::types::TextureLevelInfo;

use crate::msg::encode::RawPixelFormatInfo;

#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    /// `MTLPixelFormatGetInfoForDevice(device, pixelFormat)`. 0x38 bytes by value.
    pub fn MTLPixelFormatGetInfoForDevice(
        device: &ProtocolObject<dyn MTLDevice>,
        pixel_format: u64,
    ) -> RawPixelFormatInfo;
    /// `MTLGetTextureLevelInfoForDeviceWithOptions(device, pixelFormat, width, height, depth, arrayLength, options, flags)`.
    /// 0x68 bytes by value.
    pub fn MTLGetTextureLevelInfoForDeviceWithOptions(
        device: &ProtocolObject<dyn MTLDevice>,
        pixel_format: u64,
        width: u64,
        height: u64,
        depth: u64,
        array_length: u64,
        options: u64,
        flags: u64,
    ) -> TextureLevelInfo;
}
