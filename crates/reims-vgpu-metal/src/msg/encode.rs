//! Value types at the Metal boundary.
//!
//! - Core's value types (`platform::types`) become objc2-metal's structs for a typed call
//!   ([`ToMtl`]) and come back from a typed getter ([`FromMtl`]), field for field.
//! - The private structs objc2-metal does not declare keep the runtime's encodings: [`Enc<T>`]
//!   wraps core's `ResourceAddressRange` behind the runtime's tag (D41), and
//!   [`RawPixelFormatInfo`] is the SPI's own layout. `tests/spi_encodings.rs` checks them against
//!   the runtime.

use std::ffi::c_char;

use objc2::encode::{Encode, Encoding, RefEncode};
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLClearColor, MTLOrigin, MTLRegion, MTLSamplePosition, MTLScissorRect, MTLSize,
    MTLSizeAndAlign, MTLTextureSwizzle, MTLTextureSwizzleChannels,
    MTLVertexAmplificationViewMapping, MTLViewport,
};
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::types::{
    ClearColor, MtlOrigin, MtlRegion, MtlSamplePosition, MtlScissorRect, MtlSize, MtlViewport,
    NsRange, PixelFormatInfo, ResourceAddressRange, SizeAndAlign, Swizzle,
    VertexAmplificationViewMapping,
};

/// A core value type as objc2-metal's struct: the same fields, `NSUInteger`s as `usize`
/// (lossless on the 64-bit target, D32).
pub trait ToMtl: Copy {
    /// objc2-metal's struct.
    type Mtl;
    /// The value, field for field.
    fn mtl(self) -> Self::Mtl;
}

/// objc2-metal's struct back as core's value type.
pub trait FromMtl<M>: Sized {
    /// The value, field for field.
    fn from_mtl(m: M) -> Self;
}

impl ToMtl for MtlSize {
    type Mtl = MTLSize;
    fn mtl(self) -> MTLSize {
        MTLSize {
            width: self.width.to_usize(),
            height: self.height.to_usize(),
            depth: self.depth.to_usize(),
        }
    }
}

impl FromMtl<MTLSize> for MtlSize {
    fn from_mtl(m: MTLSize) -> MtlSize {
        MtlSize::new(m.width.to_u64(), m.height.to_u64(), m.depth.to_u64())
    }
}

impl ToMtl for MtlOrigin {
    type Mtl = MTLOrigin;
    fn mtl(self) -> MTLOrigin {
        MTLOrigin {
            x: self.x.to_usize(),
            y: self.y.to_usize(),
            z: self.z.to_usize(),
        }
    }
}

impl ToMtl for MtlRegion {
    type Mtl = MTLRegion;
    fn mtl(self) -> MTLRegion {
        MTLRegion {
            origin: self.origin.mtl(),
            size: self.size.mtl(),
        }
    }
}

impl FromMtl<MTLRegion> for MtlRegion {
    fn from_mtl(m: MTLRegion) -> MtlRegion {
        MtlRegion {
            origin: MtlOrigin {
                x: m.origin.x.to_u64(),
                y: m.origin.y.to_u64(),
                z: m.origin.z.to_u64(),
            },
            size: MtlSize::from_mtl(m.size),
        }
    }
}

impl ToMtl for NsRange {
    type Mtl = NSRange;
    fn mtl(self) -> NSRange {
        NSRange::new(self.location.to_usize(), self.length.to_usize())
    }
}

impl FromMtl<MTLSizeAndAlign> for SizeAndAlign {
    fn from_mtl(m: MTLSizeAndAlign) -> SizeAndAlign {
        SizeAndAlign {
            size: m.size.to_u64(),
            align: m.align.to_u64(),
        }
    }
}

impl ToMtl for ClearColor {
    type Mtl = MTLClearColor;
    fn mtl(self) -> MTLClearColor {
        MTLClearColor {
            red: self.red,
            green: self.green,
            blue: self.blue,
            alpha: self.alpha,
        }
    }
}

impl FromMtl<MTLClearColor> for ClearColor {
    fn from_mtl(m: MTLClearColor) -> ClearColor {
        ClearColor {
            red: m.red,
            green: m.green,
            blue: m.blue,
            alpha: m.alpha,
        }
    }
}

impl ToMtl for Swizzle {
    type Mtl = MTLTextureSwizzleChannels;
    fn mtl(self) -> MTLTextureSwizzleChannels {
        MTLTextureSwizzleChannels {
            red: MTLTextureSwizzle(self.red),
            green: MTLTextureSwizzle(self.green),
            blue: MTLTextureSwizzle(self.blue),
            alpha: MTLTextureSwizzle(self.alpha),
        }
    }
}

impl ToMtl for MtlViewport {
    type Mtl = MTLViewport;
    fn mtl(self) -> MTLViewport {
        MTLViewport {
            originX: self.origin_x,
            originY: self.origin_y,
            width: self.width,
            height: self.height,
            znear: self.znear,
            zfar: self.zfar,
        }
    }
}

impl ToMtl for MtlScissorRect {
    type Mtl = MTLScissorRect;
    fn mtl(self) -> MTLScissorRect {
        MTLScissorRect {
            x: self.x.to_usize(),
            y: self.y.to_usize(),
            width: self.width.to_usize(),
            height: self.height.to_usize(),
        }
    }
}

impl ToMtl for MtlSamplePosition {
    type Mtl = MTLSamplePosition;
    fn mtl(self) -> MTLSamplePosition {
        MTLSamplePosition {
            x: self.x,
            y: self.y,
        }
    }
}

impl FromMtl<MTLSamplePosition> for MtlSamplePosition {
    fn from_mtl(m: MTLSamplePosition) -> MtlSamplePosition {
        MtlSamplePosition { x: m.x, y: m.y }
    }
}

impl ToMtl for VertexAmplificationViewMapping {
    type Mtl = MTLVertexAmplificationViewMapping;
    fn mtl(self) -> MTLVertexAmplificationViewMapping {
        MTLVertexAmplificationViewMapping {
            viewportArrayIndexOffset: self.viewport_array_index_offset,
            renderTargetArrayIndexOffset: self.render_target_array_index_offset,
        }
    }
}

/// Each element of `values` as objc2-metal's struct, for the calls that take a C array of them.
pub fn mtl_all<T: ToMtl>(values: &[T]) -> Vec<T::Mtl> {
    values.iter().map(|v| v.mtl()).collect()
}

/// One of core's private value types with the runtime's encoding. `#[repr(transparent)]`:
/// passing an `Enc<T>` by value or by pointer passes the `T` bit for bit.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(transparent)]
pub struct Enc<T>(pub T);

const Q: Encoding = u64::ENCODING;

// SAFETY: `ResourceAddressRange` is `#[repr(C)]` in core, two `u64`s; the runtime tags the range
// `MTLAddressRange` (`^{MTLAddressRange=QQ}` in `-[MTLResourceAddressRangeArray
// initWithRanges:count:]`), not the SPI's name (D41). `Enc` is transparent over it.
unsafe impl Encode for Enc<ResourceAddressRange> {
    const ENCODING: Encoding = Encoding::Struct("MTLAddressRange", &[Q, Q]);
}

// SAFETY: a pointer to `Enc<ResourceAddressRange>` is a pointer to the struct above.
unsafe impl RefEncode for Enc<ResourceAddressRange> {
    const ENCODING_REF: Encoding = Encoding::Pointer(&<Self as Encode>::ENCODING);
}

/// `MTLPixelFormatInfo` as the C function returns it: 0x38 bytes. Core's [`PixelFormatInfo`] owns
/// its name; this is the raw form with the C string pointer.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct RawPixelFormatInfo {
    /// `name` (+0x00): Metal's static C string, or NULL.
    pub name: *const c_char,
    /// `flags` (+0x08).
    pub flags: u64,
    /// `castClass` (+0x10).
    pub cast_class: u64,
    /// `blockBytes` (+0x18).
    pub block_bytes: u64,
    /// `blockWidth` (+0x20).
    pub block_width: u64,
    /// `blockHeight` (+0x28).
    pub block_height: u64,
    /// `blockDepth` (+0x30).
    pub block_depth: u64,
}

const _: () = assert!(size_of::<RawPixelFormatInfo>() == 0x38);

impl RawPixelFormatInfo {
    /// Core's copy: every word as Metal returned it, and the name copied out of Metal's string
    /// (`None` for NULL).
    pub fn to_info(&self) -> PixelFormatInfo {
        let name = if self.name.is_null() {
            None
        } else {
            // SAFETY: a non-NULL `name` is Metal's NUL-terminated static string for the format
            // (measured: `MTLPixelFormatBGRA8Unorm`).
            let s = unsafe { std::ffi::CStr::from_ptr(self.name) };
            Some(s.to_string_lossy().into_owned())
        };
        PixelFormatInfo {
            name,
            flags: self.flags,
            cast_class: self.cast_class,
            block_bytes: self.block_bytes,
            block_width: self.block_width,
            block_height: self.block_height,
            block_depth: self.block_depth,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The private structs spell the runtime's encoding strings (D41)
    #[test]
    fn encodings_match_the_runtime_strings() {
        let cases: &[(&Encoding, &str)] = &[
            (
                &Enc::<ResourceAddressRange>::ENCODING,
                "{MTLAddressRange=QQ}",
            ),
            (
                &Enc::<ResourceAddressRange>::ENCODING_REF,
                "^{MTLAddressRange=QQ}",
            ),
        ];
        for (e, s) in cases {
            assert!(e.equivalent_to_str(s), "{e} != {s}");
        }
        // The header's tag is not the runtime's: objc2 compares names behind one pointer.
        assert!(
            !Enc::<ResourceAddressRange>::ENCODING_REF
                .equivalent_to_str("^{MTLResourceAddressRange=QQ}")
        );
    }

    /// Enc is transparent (same size and bits as core's type)
    #[test]
    fn enc_is_transparent() {
        assert_eq!(size_of::<Enc<ResourceAddressRange>>(), 16);
        let r = Enc(ResourceAddressRange {
            address: 3,
            length: 4,
        });
        assert_eq!(r.0.length, 4);
    }

    /// Core's value types and objc2-metal's structs convert field for field, both ways
    #[test]
    fn conversions_keep_every_field() {
        let region = MtlRegion {
            origin: MtlOrigin { x: 1, y: 2, z: 3 },
            size: MtlSize::new(4, 5, 6),
        };
        let m = region.mtl();
        assert_eq!((m.origin.x, m.origin.y, m.origin.z), (1, 2, 3));
        assert_eq!((m.size.width, m.size.height, m.size.depth), (4, 5, 6));
        assert_eq!(MtlRegion::from_mtl(m), region);
        let r = NsRange::new(u64::MAX, 7).mtl();
        assert_eq!((r.location, r.length), (usize::MAX, 7));
        let s = Swizzle {
            red: 5,
            green: 4,
            blue: 3,
            alpha: 0xff,
        }
        .mtl();
        assert_eq!((s.red.0, s.green.0, s.blue.0, s.alpha.0), (5, 4, 3, 0xff));
        let v = MtlViewport {
            origin_x: 1.0,
            origin_y: 2.0,
            width: 3.0,
            height: 4.0,
            znear: 5.0,
            zfar: 6.0,
        }
        .mtl();
        assert_eq!((v.originX, v.zfar), (1.0, 6.0));
        let c = ClearColor {
            red: 0.1,
            green: 0.2,
            blue: 0.3,
            alpha: 0.4,
        };
        assert_eq!(ClearColor::from_mtl(c.mtl()), c);
        let a = mtl_all(&[VertexAmplificationViewMapping {
            viewport_array_index_offset: 1,
            render_target_array_index_offset: 2,
        }]);
        assert_eq!(
            (
                a[0].viewportArrayIndexOffset,
                a[0].renderTargetArrayIndexOffset
            ),
            (1, 2)
        );
    }

    /// RawPixelFormatInfo::to_info copies every word and the name (NULL -> None)
    #[test]
    fn raw_pixel_format_info_to_info() {
        let name = c"MTLPixelFormatBGRA8Unorm";
        let raw = RawPixelFormatInfo {
            name: name.as_ptr(),
            flags: 0x703019f,
            cast_class: 2,
            block_bytes: 4,
            block_width: 5,
            block_height: 6,
            block_depth: 7,
        };
        let i = raw.to_info();
        assert_eq!(i.name.as_deref(), Some("MTLPixelFormatBGRA8Unorm"));
        assert_eq!(
            (
                i.flags,
                i.cast_class,
                i.block_bytes,
                i.block_width,
                i.block_height,
                i.block_depth
            ),
            (0x703019f, 2, 4, 5, 6, 7)
        );
        let nul = RawPixelFormatInfo {
            name: std::ptr::null(),
            ..raw
        };
        assert_eq!(nul.to_info().name, None);
    }
}
