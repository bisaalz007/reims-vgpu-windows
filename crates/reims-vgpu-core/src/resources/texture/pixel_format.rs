//! What a pixel format looks like to the guest: Metal's description, except for the XR10
//! formats 550 and 551 and the depth-stencil formats 260 and 261 as two planes, which core
//! describes itself.
//!
//! The description's layout has C linkage and no header naming it; its field names are ours,
//! after what [`guest_pixel_format_info`] stores and validation reads.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::guest::records::GuestFeatures;
use crate::nil_send;
use crate::num::Truncate;
use crate::platform::Platform;
use crate::platform::device::GpuDevice;

/// `MTLPixelFormatDepth32Float_Stencil8`.
pub const PIXEL_FORMAT_DEPTH32_FLOAT_STENCIL8: u64 = 260;
/// `MTLPixelFormatX32_Stencil8`.
pub const PIXEL_FORMAT_X32_STENCIL8: u64 = 261;
/// The first two-plane XR10 format the public headers do not name.
pub const PIXEL_FORMAT_550: u64 = 550;
/// The second (its sRGB twin, flags 0x83099f).
pub const PIXEL_FORMAT_551: u64 = 551;

/// `flags` bit 13: a PVRTC format.
pub const FLAG_PVRTC: u32 = 0x2000;
/// `flags` bit 23: a two-plane format.
pub const FLAG_TWO_PLANE: u32 = 0x80_0000;

/// `kind` 0: a format Metal describes, not compressed.
pub const KIND_NORMAL: u32 = 0;
/// `kind` 1: compressed, Metal's flags bit 10.
pub const KIND_COMPRESSED: u32 = 1;
/// `kind` 2: a two-plane format core describes itself.
pub const KIND_TWO_PLANE: u32 = 2;

/// Whether `pixel_format` is one of the two XR10 formats the compute kernels page.
pub(crate) fn is_xr10(pixel_format: u64) -> bool {
    pixel_format == PIXEL_FORMAT_550 || pixel_format == PIXEL_FORMAT_551
}

/// One plane's description, 0x20 bytes (names are ours: see the module doc).
///
/// `block_width` is read whole as a `u64` (the compressed block width a Metal format copies in),
/// and written as two bytes, a component count (+8) and an alignment (+9), for the formats core
/// describes itself.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct GuestPixelFormatPlaneInfo {
    /// Bytes per block (+0).
    pub block_bytes: u64,
    /// Block width (+8), or the component count (+8) and alignment (+9) as bytes.
    pub block_width: u64,
    /// Block height (+0x10).
    pub block_height: u64,
    /// Block depth (+0x18).
    pub block_depth: u64,
}

const _: () = assert!(size_of::<GuestPixelFormatPlaneInfo>() == 0x20);

impl GuestPixelFormatPlaneInfo {
    /// The component count: the byte at +8.
    pub fn component_count(&self) -> u8 {
        let [c, ..] = self.block_width.to_le_bytes();
        c
    }

    /// The alignment: the byte at +9.
    pub fn alignment(&self) -> u8 {
        let [_, a, ..] = self.block_width.to_le_bytes();
        a
    }

    /// Stores the component count and alignment bytes, leaving the rest of the word as it is
    /// (zero in every plane built here).
    fn set_component_count_and_alignment(&mut self, component_count: u8, alignment: u8) {
        let mut b = self.block_width.to_le_bytes();
        let [c, a, ..] = &mut b;
        *c = component_count;
        *a = alignment;
        self.block_width = u64::from_le_bytes(b);
    }
}

/// A pixel format's description, 0x50 bytes (names are ours: see the module doc).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct GuestPixelFormatInfo {
    /// Flags (+0): Metal's, or core's own for 550, 551, 260 and 261 (bit 13 PVRTC, bit 23 two
    /// planes).
    pub flags: u32,
    /// Kind (+4): 0 normal, 1 compressed, 2 a two-plane format described here.
    pub kind: u32,
    /// The linear texture alignment (+8): set only for 550 and 551; nothing reads it.
    pub linear_texture_alignment_bytes: u32,
    /// Padding (+0xc) nothing reads.
    pub unknown0c: [u8; 4],
    /// The planes (+0x10).
    pub planes: [GuestPixelFormatPlaneInfo; 2],
}

const _: () = assert!(size_of::<GuestPixelFormatInfo>() == 0x50);

impl GuestPixelFormatInfo {
    /// The 0x50 bytes (the padding as held).
    pub fn to_bytes(&self) -> [u8; 0x50] {
        let mut out = [0u8; 0x50];
        out.copy_from_slice(self.as_bytes());
        out
    }
}

/// Metal's description: its flags truncated to 32 bits, bit 10 of them as the kind, and its
/// four block words as plane 0.
fn info_from_metal<P: Platform>(
    device: Option<&P::Device>,
    pixel_format: u64,
) -> GuestPixelFormatInfo {
    // A Metal texture always has a device, so the nil arm (zeros) is never taken.
    let metal = nil_send!(device, pixel_format_info(pixel_format));
    let flags = metal.flags.low_u32();
    let mut info = GuestPixelFormatInfo::default();
    info.planes[0] = GuestPixelFormatPlaneInfo {
        block_bytes: metal.block_bytes,
        block_width: metal.block_width,
        block_height: metal.block_height,
        block_depth: metal.block_depth,
    };
    info.flags = flags;
    info.kind = flags.wrapping_shr(10) & 1;
    info
}

/// What `pixel_format` looks like to the guest: Metal's description, except for 550 and 551
/// (when the device supports extended XR10 formats; zero flags otherwise) and 260 and 261 (as
/// two planes when the guest takes dual-plane textures).
///
/// `device` is the texture's, asked unchecked. `features` is read only for 260 and 261.
#[expect(
    clippy::field_reassign_with_default,
    reason = "the description starts zeroed, then takes its fields one by one"
)]
pub fn guest_pixel_format_info<P: Platform>(
    device: Option<&P::Device>,
    pixel_format: u64,
    features: &GuestFeatures,
) -> GuestPixelFormatInfo {
    match pixel_format {
        PIXEL_FORMAT_550 | PIXEL_FORMAT_551 => {
            let mut info = GuestPixelFormatInfo::default();
            let caps = device.map(|d| d.caps());
            if !caps
                .as_ref()
                .is_some_and(|c| c.flags.supports_extended_xr10_formats)
            {
                info.flags = 0;
                return info;
            }
            info.flags = if pixel_format == PIXEL_FORMAT_551 {
                0x83_099f
            } else {
                0x83_019f
            };
            info.linear_texture_alignment_bytes = caps
                .map_or(0, |c| c.limits.linear_texture_alignment_bytes)
                .low_u32();
            info.planes[0].set_component_count_and_alignment(3, 4);
            info.planes[0].block_bytes = 4;
            info.planes[0].block_height = 4;
            info.planes[0].block_depth = 4;
            info.planes[1].set_component_count_and_alignment(1, 4);
            info.planes[1].block_bytes = 4;
            info.planes[1].block_height = 4;
            info.planes[1].block_depth = 4;
            info.kind = KIND_TWO_PLANE;
            info
        }
        PIXEL_FORMAT_DEPTH32_FLOAT_STENCIL8 | PIXEL_FORMAT_X32_STENCIL8 => {
            if features.dual_plane_textures {
                let mut info = GuestPixelFormatInfo::default();
                info.flags = if pixel_format == PIXEL_FORMAT_DEPTH32_FLOAT_STENCIL8 {
                    0x83_01ef
                } else {
                    0x83_01cf
                };
                info.planes[0].set_component_count_and_alignment(1, 4);
                info.planes[0].block_bytes = 4;
                info.planes[0].block_height = 4;
                info.planes[0].block_depth = 4;
                // The stencil plane's alignment byte and block bytes are both the guest's stencil
                // byte count (32 bits), narrowed to a byte and widened to 64 bits.
                let s8 = features.s8_byte_count_for_d32s8;
                info.planes[1].set_component_count_and_alignment(1, s8.low_u8());
                info.planes[1].block_bytes = u64::from(s8);
                info.planes[1].block_height = 1;
                info.planes[1].block_depth = 1;
                info.kind = KIND_TWO_PLANE;
                return info;
            }
            // Without dual-plane textures, D32S8 and X32S8 are Metal's description with the
            // two-plane bit cleared; the kind stays Metal's.
            let mut info = info_from_metal::<P>(device, pixel_format);
            info.flags &= !FLAG_TWO_PLANE;
            info
        }
        _ => info_from_metal::<P>(device, pixel_format),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::caps::DeviceCaps;
    use crate::platform::types::PixelFormatInfo;
    use crate::testkit::{FakePlatform, Step};

    fn features(dual: bool, s8: u32) -> GuestFeatures {
        GuestFeatures {
            dual_plane_textures: dual,
            s8_byte_count_for_d32s8: s8,
            ..GuestFeatures::default()
        }
    }

    fn metal(flags: u64, bb: u64, bw: u64, bh: u64, bd: u64) -> PixelFormatInfo {
        PixelFormatInfo {
            name: Some("MTLPixelFormatTest".into()),
            flags,
            cast_class: 7,
            block_bytes: bb,
            block_width: bw,
            block_height: bh,
            block_depth: bd,
        }
    }

    /// The device's capabilities, with extended XR10 formats as given and `alignment` as its
    /// linear texture alignment.
    fn set_caps(d: &<FakePlatform as Platform>::Device, xr10: bool, alignment: u64) {
        let mut caps = DeviceCaps::clone(&d.caps());
        caps.flags.supports_extended_xr10_formats = xr10;
        caps.limits.linear_texture_alignment_bytes = alignment;
        d.set_caps(caps);
    }

    fn plane(bb: u64, bw: u64, bh: u64, bd: u64) -> GuestPixelFormatPlaneInfo {
        GuestPixelFormatPlaneInfo {
            block_bytes: bb,
            block_width: bw,
            block_height: bh,
            block_depth: bd,
        }
    }

    mod get_info {
        use super::*;

        /// 550 and 551 on a device with extended XR10: core's two-plane description, the
        /// device's linear alignment as 32 bits; Metal's description not asked.
        #[test]
        fn xr10_supported() {
            for (fmt, flags) in [(550u64, 0x83_019fu32), (551, 0x83_099f)] {
                let p = FakePlatform::new();
                let d = p.device();
                set_caps(&d, true, 0x1_0000_0040);
                let info =
                    guest_pixel_format_info::<FakePlatform>(Some(&d), fmt, &features(false, 0));
                assert_eq!(info.flags, flags);
                assert_eq!(info.kind, KIND_TWO_PLANE);
                assert_eq!(info.linear_texture_alignment_bytes, 0x40);
                assert_eq!(info.planes[0], plane(4, 0x403, 4, 4));
                assert_eq!(info.planes[1], plane(4, 0x401, 4, 4));
                assert_eq!(info.planes[0].component_count(), 3);
                assert_eq!(info.planes[0].alignment(), 4);
                assert_eq!(info.unknown0c, [0; 4]);
                assert!(p.call_log().calls().is_empty());
            }
        }

        /// 550 and 551 without extended XR10: all zero, nothing else asked.
        #[test]
        fn xr10_unsupported() {
            for fmt in [550u64, 551] {
                let p = FakePlatform::new();
                let d = p.device();
                set_caps(&d, false, 0x40);
                let info =
                    guest_pixel_format_info::<FakePlatform>(Some(&d), fmt, &features(true, 4));
                assert_eq!(info, GuestPixelFormatInfo::default());
                assert!(p.call_log().calls().is_empty());
            }
        }

        /// 260 and 261 as two planes: plane 1 carries `s8ByteCountForD32S8` as its alignment
        /// byte (narrowed) and its blockBytes (widened); no platform call.
        #[test]
        fn depth_stencil_dual_plane() {
            for (fmt, flags) in [(260u64, 0x83_01efu32), (261, 0x83_01cf)] {
                for s8 in [4u32, 1, 0x1ff] {
                    let p = FakePlatform::new();
                    let d = p.device();
                    let info =
                        guest_pixel_format_info::<FakePlatform>(Some(&d), fmt, &features(true, s8));
                    assert_eq!(info.flags, flags);
                    assert_eq!(info.kind, KIND_TWO_PLANE);
                    assert_eq!(info.linear_texture_alignment_bytes, 0);
                    assert_eq!(info.planes[0], plane(4, 0x401, 4, 4));
                    let align = u64::from(s8 & 0xff);
                    assert_eq!(info.planes[1], plane(u64::from(s8), 1 | (align << 8), 1, 1));
                    assert!(p.call_log().is_empty());
                }
            }
        }

        /// 260 and 261 before `dualPlaneTextures`: Metal's info with bit 23 cleared and the kind
        /// from Metal's bit 10.
        #[test]
        fn depth_stencil_single_plane() {
            for fmt in [260u64, 261] {
                let p = FakePlatform::new();
                let d = p.device();
                p.script().push(
                    "MTLPixelFormatGetInfoForDevice()",
                    Step::value(metal(0x1_0080_0401, 8, 1, 1, 1)),
                );
                let info =
                    guest_pixel_format_info::<FakePlatform>(Some(&d), fmt, &features(false, 4));
                assert_eq!(info.flags, 0x0401);
                assert_eq!(info.kind, 1);
                assert_eq!(info.planes[0], plane(8, 1, 1, 1));
                assert_eq!(info.planes[1], GuestPixelFormatPlaneInfo::default());
                let calls = p.call_log().calls();
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].sel, "MTLPixelFormatGetInfoForDevice()");
                assert_eq!(calls[0].args[0].as_u64(), Some(fmt));
            }
        }

        /// Any other format: Metal's flags truncated to 32 bits, kind bit 10, plane 0 the four
        /// words at +0x18 whole, plane 1 zero; bit 23 kept.
        #[test]
        fn metal_formats() {
            let p = FakePlatform::new();
            let d = p.device();
            p.script().push(
                "MTLPixelFormatGetInfoForDevice()",
                Step::value(metal(0xffff_ffff_0080_2400, 16, 4, 4, 1)),
            );
            let info = guest_pixel_format_info::<FakePlatform>(Some(&d), 170, &features(true, 4));
            assert_eq!(info.flags, 0x0080_2400);
            assert_eq!(info.kind, 1);
            assert_eq!(info.linear_texture_alignment_bytes, 0);
            assert_eq!(info.planes[0], plane(16, 4, 4, 1));
            assert_eq!(info.planes[1], GuestPixelFormatPlaneInfo::default());

            p.script().push(
                "MTLPixelFormatGetInfoForDevice()",
                Step::value(metal(0x0703_019f, 4, 0x0104, 0, 0)),
            );
            let info = guest_pixel_format_info::<FakePlatform>(Some(&d), 80, &features(true, 4));
            assert_eq!(info.flags, 0x0703_019f);
            assert_eq!(info.kind, 0);
            assert_eq!(info.planes[0], plane(4, 0x0104, 0, 0));
            assert_eq!(
                p.call_log().sels(),
                vec![
                    "MTLPixelFormatGetInfoForDevice()",
                    "MTLPixelFormatGetInfoForDevice()"
                ]
            );
        }

        /// A nil device (never the case: a texture has its device) answers zeros for every send.
        #[test]
        fn nil_device() {
            let none: Option<&<FakePlatform as Platform>::Device> = None;
            let info = guest_pixel_format_info::<FakePlatform>(none, 550, &features(true, 4));
            assert_eq!(info, GuestPixelFormatInfo::default());
            let info = guest_pixel_format_info::<FakePlatform>(none, 70, &features(true, 4));
            assert_eq!(info, GuestPixelFormatInfo::default());
        }
    }

    /// The layout and its byte image
    #[test]
    fn to_bytes_layout() {
        let mut info = GuestPixelFormatInfo {
            flags: 0x0102_0304,
            kind: 2,
            linear_texture_alignment_bytes: 0x40,
            ..GuestPixelFormatInfo::default()
        };
        info.planes[1].set_component_count_and_alignment(1, 4);
        info.planes[1].block_depth = 9;
        let b = info.to_bytes();
        assert_eq!(&b[0..4], &[4, 3, 2, 1]);
        assert_eq!(b[4], 2);
        assert_eq!(b[8], 0x40);
        assert_eq!(&b[0x38..0x3a], &[1, 4]);
        assert_eq!(b[0x48], 9);
    }
}
