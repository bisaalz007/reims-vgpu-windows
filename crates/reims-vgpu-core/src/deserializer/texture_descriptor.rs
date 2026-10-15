//! The serialized texture descriptor, v1 and v2.
//!
//! The deserializer never interprets either descriptor: it hands the record to the delegate
//! (`textureDescriptorForDescriptor{,2}:`), whose `newTextureDescriptor` reads the bit-fields. The
//! accessors below are those bit-field reads: a C bit-field read zero-extends the field's bits, so
//! each returns the bits unchanged.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::num::Truncate;

/// `PGSerializedTextureDescriptor` (32 bytes). The first word holds `textureType : 4`,
/// `framebufferOnly : 1`, `isDrawable : 1`, `allowGPUOptimizedContents : 1`, `field0_7 : 1`,
/// `usage : 8`, `pixelFormat : 16`, read through the accessors. `field0_7` (bit 7) is never
/// read, so it has no accessor.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
#[allow(missing_docs)] // the guest ABI's field names
pub struct SerializedTextureDescriptor {
    pub bits: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mipmap_level_count: u16,
    pub sample_count: u16,
    pub array_length: u16,
    pub resource_options: u16,
    pub protection_options: u64,
}

const _: () = assert!(size_of::<SerializedTextureDescriptor>() == 32);

impl SerializedTextureDescriptor {
    /// `textureType`: bits 0-3 of the first word.
    pub fn texture_type(&self) -> u8 {
        (self.bits & 0xf).low_u8()
    }

    /// `framebufferOnly`: bit 4.
    pub const fn framebuffer_only(&self) -> bool {
        self.bits & 1u32.wrapping_shl(4) != 0
    }

    /// `isDrawable`: bit 5.
    pub const fn is_drawable(&self) -> bool {
        self.bits & 1u32.wrapping_shl(5) != 0
    }

    /// `allowGPUOptimizedContents`: bit 6.
    pub const fn allow_gpu_optimized_contents(&self) -> bool {
        self.bits & 1u32.wrapping_shl(6) != 0
    }

    /// `usage`: bits 8-15 (byte 1). Only eight bits, unlike descriptor 2's 32-bit `usage`.
    pub fn usage(&self) -> u8 {
        self.bits.wrapping_shr(8).low_u8()
    }

    /// `pixelFormat`: bits 16-31 (bytes 2-3).
    pub fn pixel_format(&self) -> u16 {
        self.bits.wrapping_shr(16).low_u16()
    }
}

/// `PGSerializedTextureDescriptor2`, packed, 0x27 bytes. The bit-fields fill bytes 0-2 only: byte 0
/// holds `textureType : 4`, `framebufferOnly : 1`, `isDrawable : 1`,
/// `allowGPUOptimizedContents : 1`, `writeSwizzleEnabled : 1`, and bytes 1-2 `pixelFormat : 16`,
/// read through the accessors; `usage` starts at +3.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
#[allow(missing_docs)] // the guest ABI's field names
pub struct SerializedTextureDescriptor2 {
    pub bits: [u8; 3],
    pub usage: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mipmap_level_count: u16,
    pub sample_count: u16,
    pub array_length: u16,
    pub resource_options: u16,
    pub protection_options: u64,
    pub swizzle_red: u8,
    pub swizzle_green: u8,
    pub swizzle_blue: u8,
    pub swizzle_alpha: u8,
}

// 's static_assert.
const _: () = assert!(size_of::<SerializedTextureDescriptor2>() == 0x27);

impl SerializedTextureDescriptor2 {
    /// Byte 0, which holds the flag bit-fields.
    const fn flags(&self) -> u8 {
        let [b0, _, _] = self.bits;
        b0
    }

    /// `textureType`: bits 0-3 of byte 0.
    pub const fn texture_type(&self) -> u8 {
        self.flags() & 0xf
    }

    /// `framebufferOnly`: bit 4 of byte 0.
    pub const fn framebuffer_only(&self) -> bool {
        self.flags() & 0x10 != 0
    }

    /// `isDrawable`: bit 5 of byte 0.
    pub const fn is_drawable(&self) -> bool {
        self.flags() & 0x20 != 0
    }

    /// `allowGPUOptimizedContents`: bit 6 of byte 0.
    pub const fn allow_gpu_optimized_contents(&self) -> bool {
        self.flags() & 0x40 != 0
    }

    /// `writeSwizzleEnabled`: bit 7 of byte 0.
    pub const fn write_swizzle_enabled(&self) -> bool {
        self.flags() & 0x80 != 0
    }

    /// `pixelFormat`: bytes 1-2, little-endian.
    pub const fn pixel_format(&self) -> u16 {
        let [_, lo, hi] = self.bits;
        u16::from_le_bytes([lo, hi])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    /// The v1 layout: 0x20 bytes, the ABI's offsets.
    #[test]
    fn descriptor_layout_matches_the_abi() {
        assert_eq!(size_of::<SerializedTextureDescriptor>(), 0x20);
        assert_eq!(align_of::<SerializedTextureDescriptor>(), 8);
        assert_eq!(offset_of!(SerializedTextureDescriptor, bits), 0x0);
        assert_eq!(offset_of!(SerializedTextureDescriptor, width), 0x4);
        assert_eq!(offset_of!(SerializedTextureDescriptor, height), 0x8);
        assert_eq!(offset_of!(SerializedTextureDescriptor, depth), 0xc);
        assert_eq!(
            offset_of!(SerializedTextureDescriptor, mipmap_level_count),
            0x10
        );
        assert_eq!(offset_of!(SerializedTextureDescriptor, sample_count), 0x12);
        assert_eq!(offset_of!(SerializedTextureDescriptor, array_length), 0x14);
        assert_eq!(
            offset_of!(SerializedTextureDescriptor, resource_options),
            0x16
        );
        assert_eq!(
            offset_of!(SerializedTextureDescriptor, protection_options),
            0x18
        );
    }

    /// The v2 layout: packed, 0x27 bytes, `usage` at +3.
    #[test]
    fn descriptor2_layout_matches_the_abi() {
        type D = SerializedTextureDescriptor2;
        assert_eq!(size_of::<D>(), 0x27);
        assert_eq!(offset_of!(D, bits), 0x0);
        assert_eq!(offset_of!(D, usage), 0x3);
        assert_eq!(offset_of!(D, width), 0x7);
        assert_eq!(offset_of!(D, height), 0xb);
        assert_eq!(offset_of!(D, depth), 0xf);
        assert_eq!(offset_of!(D, mipmap_level_count), 0x13);
        assert_eq!(offset_of!(D, sample_count), 0x15);
        assert_eq!(offset_of!(D, array_length), 0x17);
        assert_eq!(offset_of!(D, resource_options), 0x19);
        assert_eq!(offset_of!(D, protection_options), 0x1b);
        assert_eq!(offset_of!(D, swizzle_red), 0x23);
        assert_eq!(offset_of!(D, swizzle_green), 0x24);
        assert_eq!(offset_of!(D, swizzle_blue), 0x25);
        assert_eq!(offset_of!(D, swizzle_alpha), 0x26);
    }

    /// The v1 bytes a guest would write, little-endian.
    fn v1_bytes(word: u32) -> [u8; 0x20] {
        let mut b = [0u8; 0x20];
        b[0..4].copy_from_slice(&word.to_le_bytes());
        b[4..8].copy_from_slice(&640u32.to_le_bytes());
        b[8..12].copy_from_slice(&480u32.to_le_bytes());
        b[12..16].copy_from_slice(&1u32.to_le_bytes());
        b[0x10..0x12].copy_from_slice(&7u16.to_le_bytes());
        b[0x12..0x14].copy_from_slice(&4u16.to_le_bytes());
        b[0x14..0x16].copy_from_slice(&3u16.to_le_bytes());
        b[0x16..0x18].copy_from_slice(&0x20u16.to_le_bytes());
        b[0x18..0x20].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());
        b
    }

    /// every v1 bit-field and field decodes from guest bytes; bit 7 (`field0_7`) and
    /// the other bits do not leak into their neighbours.
    #[test]
    fn descriptor_fields_decode() {
        // textureType 2, framebufferOnly, allowGPUOptimizedContents, field0_7 set, isDrawable
        // clear; usage 0x17; pixelFormat 80 (BGRA8Unorm).
        let word = 0x2 | 0x10 | 0x40 | 0x80 | (0x17 << 8) | (80 << 16);
        let d = SerializedTextureDescriptor::read_from_bytes(&v1_bytes(word)).unwrap();
        assert_eq!(d.texture_type(), 2);
        assert!(d.framebuffer_only());
        assert!(!d.is_drawable());
        assert!(d.allow_gpu_optimized_contents());
        assert_eq!(d.usage(), 0x17);
        assert_eq!(d.pixel_format(), 80);
        assert_eq!((d.width, d.height, d.depth), (640, 480, 1));
        assert_eq!(d.mipmap_level_count, 7);
        assert_eq!(d.sample_count, 4);
        assert_eq!(d.array_length, 3);
        assert_eq!(d.resource_options, 0x20);
        assert_eq!(d.protection_options, 0x1122_3344_5566_7788);
    }

    /// the v1 bit-fields at their extremes: all ones and the single drawable bit.
    #[test]
    fn descriptor_bitfield_extremes() {
        let all = SerializedTextureDescriptor {
            bits: u32::MAX,
            ..Default::default()
        };
        assert_eq!(all.texture_type(), 0xf);
        assert!(all.framebuffer_only() && all.is_drawable() && all.allow_gpu_optimized_contents());
        assert_eq!(all.usage(), 0xff);
        assert_eq!(all.pixel_format(), 0xffff);
        let drawable = SerializedTextureDescriptor {
            bits: 0x20,
            ..Default::default()
        };
        assert!(drawable.is_drawable());
        assert!(!drawable.framebuffer_only() && !drawable.allow_gpu_optimized_contents());
        assert_eq!(
            (
                drawable.texture_type(),
                drawable.usage(),
                drawable.pixel_format()
            ),
            (0, 0, 0)
        );
    }

    /// A v1 descriptor needs all 0x20 bytes; one fewer fails (a record is read at its whole size).
    #[test]
    fn descriptor_short_read_fails() {
        let b = v1_bytes(0);
        assert!(SerializedTextureDescriptor::read_from_bytes(&b[..0x1f]).is_err());
        assert!(SerializedTextureDescriptor::read_from_bytes(&b).is_ok());
    }

    /// every v2 bit-field and field decodes from its packed guest bytes.
    #[test]
    fn descriptor2_fields_decode() {
        let mut b = [0u8; 0x27];
        // textureType 7, isDrawable, writeSwizzleEnabled; pixelFormat 0x0251.
        b[0] = 0x7 | 0x20 | 0x80;
        b[1..3].copy_from_slice(&0x0251u16.to_le_bytes());
        b[3..7].copy_from_slice(&0x8000_0013u32.to_le_bytes());
        b[7..0xb].copy_from_slice(&256u32.to_le_bytes());
        b[0xb..0xf].copy_from_slice(&128u32.to_le_bytes());
        b[0xf..0x13].copy_from_slice(&2u32.to_le_bytes());
        b[0x13..0x15].copy_from_slice(&9u16.to_le_bytes());
        b[0x15..0x17].copy_from_slice(&1u16.to_le_bytes());
        b[0x17..0x19].copy_from_slice(&6u16.to_le_bytes());
        b[0x19..0x1b].copy_from_slice(&0x10u16.to_le_bytes());
        b[0x1b..0x23].copy_from_slice(&0xdead_beef_0bad_f00du64.to_le_bytes());
        b[0x23..0x27].copy_from_slice(&[2, 3, 4, 9]);
        let d = SerializedTextureDescriptor2::read_from_bytes(&b).unwrap();
        assert_eq!(d.texture_type(), 7);
        assert!(!d.framebuffer_only());
        assert!(d.is_drawable());
        assert!(!d.allow_gpu_optimized_contents());
        assert!(d.write_swizzle_enabled());
        assert_eq!(d.pixel_format(), 0x0251);
        assert_eq!({ d.usage }, 0x8000_0013);
        assert_eq!(({ d.width }, { d.height }, { d.depth }), (256, 128, 2));
        assert_eq!({ d.mipmap_level_count }, 9);
        assert_eq!({ d.sample_count }, 1);
        assert_eq!({ d.array_length }, 6);
        assert_eq!({ d.resource_options }, 0x10);
        assert_eq!({ d.protection_options }, 0xdead_beef_0bad_f00d);
        // A channel is passed on raw; the resource manager maps >= 6 to One, not the record.
        assert_eq!(
            (
                d.swizzle_red,
                d.swizzle_green,
                d.swizzle_blue,
                d.swizzle_alpha
            ),
            (2, 3, 4, 9)
        );
        assert_eq!(d.as_bytes(), &b[..]);
    }

    /// the v2 flag bits one at a time, and all ones.
    #[test]
    fn descriptor2_flag_bits() {
        let with = |b0: u8| SerializedTextureDescriptor2 {
            bits: [b0, 0, 0],
            ..Default::default()
        };
        assert!(with(0x10).framebuffer_only());
        assert!(with(0x40).allow_gpu_optimized_contents());
        assert_eq!(with(0x10).texture_type(), 0);
        assert!(!with(0x0f).write_swizzle_enabled());
        let all = SerializedTextureDescriptor2 {
            bits: [0xff; 3],
            ..Default::default()
        };
        assert_eq!(all.texture_type(), 0xf);
        assert!(all.framebuffer_only() && all.is_drawable());
        assert!(all.allow_gpu_optimized_contents() && all.write_swizzle_enabled());
        assert_eq!(all.pixel_format(), 0xffff);
    }

    /// a v2 descriptor needs all 0x27 bytes; one fewer fails.
    #[test]
    fn descriptor2_short_read_fails() {
        let b = [0u8; 0x27];
        assert!(SerializedTextureDescriptor2::read_from_bytes(&b[..0x26]).is_err());
        assert!(SerializedTextureDescriptor2::read_from_bytes(&b).is_ok());
    }
}
