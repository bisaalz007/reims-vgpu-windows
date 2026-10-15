//! Golden test: the IOSurface retain count after each surface-taking constructor.
//!
//! A mapper unmap is `Busy` unless the surface's retain count is 1, so every retain a Metal object
//! makes of a surface is guest-visible. These tests record what Metal does with the count, which
//! the metal crate's handles rely on (a handle wraps exactly one reference; `retained()` is the
//! only way to add one):
//! - each surface-taking constructor retains the surface once for the object's life, and the
//!   release of the object gives the reference back;
//! - the `iosurface` getters return the surface at +0, so `Buffer::iosurface` and
//!   `Texture::iosurface` must add exactly one retain.
//!
//! The sends are made directly (test code): `MetalSurface` has no public constructor from an
//! existing `IOSurfaceRef`, so the metal crate's own surface-taking paths are checked here through
//! their Metal selectors, and through the crate where no surface is needed.

#![cfg(target_os = "macos")] // Metal is macOS's: the crate is empty elsewhere
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    missing_docs
)]

use std::ffi::c_void;

use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::{AnyClass, AnyObject};
use objc2_foundation::{NSDictionary, NSNumber, NSString};
use objc2_io_surface::IOSurfaceRef;
use objc2_metal::{MTLPixelFormat, MTLStorageMode, MTLTextureDescriptor, MTLTextureUsage};
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::resource::{Buffer, Texture};
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_metal::MetalPlatform;
use reims_vgpu_metal::gpu::device::MetalDevice;

unsafe extern "C" {
    fn CFGetRetainCount(cf: *const c_void) -> isize;
}

/// A 16x16 BGRA surface (`[[IOSurface alloc] initWithProperties:]`), the one reference this
/// test owns.
fn surface() -> Retained<AnyObject> {
    let keys = [
        NSString::from_str("IOSurfaceWidth"),
        NSString::from_str("IOSurfaceHeight"),
        NSString::from_str("IOSurfaceBytesPerElement"),
        NSString::from_str("IOSurfacePixelFormat"),
    ];
    let values = [
        NSNumber::new_u32(16),
        NSNumber::new_u32(16),
        NSNumber::new_u32(4),
        NSNumber::new_u32(u32::from_be_bytes(*b"BGRA")),
    ];
    let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
    let value_refs: Vec<&NSNumber> = values.iter().map(|v| &**v).collect();
    let props = NSDictionary::from_slices(&key_refs, &value_refs);
    let cls = AnyClass::get(c"IOSurface").unwrap();
    // SAFETY: `+alloc` then `-initWithProperties:` with a property dictionary; +1 or nil.
    unsafe {
        let a: *mut AnyObject = msg_send![cls, alloc];
        let s: *mut AnyObject = msg_send![a, initWithProperties: &*props];
        Retained::from_raw(s).unwrap()
    }
}

/// The surface as the `IOSurfaceRef` the SPI takes (toll-free bridged).
fn sref(s: &AnyObject) -> *const IOSurfaceRef {
    std::ptr::from_ref(s).cast()
}

fn count(s: &AnyObject) -> isize {
    // SAFETY: an IOSurface object is its IOSurfaceRef (toll-free bridged), a live CF object.
    unsafe { CFGetRetainCount(std::ptr::from_ref(s).cast()) }
}

fn dev() -> MetalDevice {
    MetalPlatform::system_default_device().unwrap()
}

/// newBufferWithIOSurface: retains the surface once while the buffer lives; -[MTLBuffer iosurface]
/// returns it at +0 (the one retain after the getter is the handle's)
#[test]
fn buffer_with_iosurface_retains_once() {
    let d = dev();
    let s = surface();
    let base = count(&s);
    // SAFETY: SPI `newBufferWithIOSurface:` (`@24@0:8^{__IOSurface=}16`), +1 or nil.
    let b: Option<Retained<AnyObject>> =
        unsafe { msg_send![d.get(), newBufferWithIOSurface: sref(&s)] };
    let b = b.unwrap();
    assert_eq!(count(&s), base + 1);
    // SAFETY: SPI `-iosurface` (`^{__IOSurface=}16@0:8`), +0.
    let got: *const IOSurfaceRef = unsafe { msg_send![&*b, iosurface] };
    assert_eq!(got, sref(&s));
    assert_eq!(count(&s), base + 1, "the getter returns +0");
    drop(b);
    assert_eq!(count(&s), base);
}

/// newTextureWithDescriptor:iosurface:plane: retains the surface once while the texture lives;
/// -[MTLTexture iosurface] returns it at +0
#[test]
fn texture_with_iosurface_retains_once() {
    let d = dev();
    let s = surface();
    let base = count(&s);
    let td = MTLTextureDescriptor::new();
    // SAFETY: plain property sets on a fresh descriptor.
    unsafe {
        td.setWidth(16);
        td.setHeight(16);
    }
    td.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    td.setStorageMode(MTLStorageMode::Shared);
    td.setUsage(MTLTextureUsage::ShaderRead);
    // SAFETY: `newTextureWithDescriptor:iosurface:plane:` with a live surface; +1 or nil.
    let t: Option<Retained<AnyObject>> = unsafe {
        msg_send![d.get(), newTextureWithDescriptor: &*td, iosurface: sref(&s), plane: 0usize]
    };
    let t = t.unwrap();
    assert_eq!(count(&s), base + 1);
    // SAFETY: SPI `-iosurface`, +0.
    let got: *const IOSurfaceRef = unsafe { msg_send![&*t, iosurface] };
    assert_eq!(got, sref(&s));
    assert_eq!(count(&s), base + 1, "the getter returns +0");
    drop(t);
    assert_eq!(count(&s), base);
}

/// Through the crate: `Buffer::iosurface` and `Texture::iosurface` on surface-backed objects return
/// the surface with one retain, and dropping it gives the reference back.
#[test]
fn crate_iosurface_getters_return_the_surface_plus_one() {
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{MTLBuffer, MTLTexture};
    let d = dev();
    let s = surface();
    let base = count(&s);
    // SAFETY: SPI `newBufferWithIOSurface:` (`@24@0:8^{__IOSurface=}16`), +1 or nil.
    let b: Option<Retained<ProtocolObject<dyn MTLBuffer>>> =
        unsafe { msg_send![d.get(), newBufferWithIOSurface: sref(&s)] };
    let b = reims_vgpu_metal::Obj::from_retained(b.unwrap());
    let td = MTLTextureDescriptor::new();
    // SAFETY: plain property sets on a fresh descriptor.
    unsafe {
        td.setWidth(16);
        td.setHeight(16);
    }
    td.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    td.setStorageMode(MTLStorageMode::Shared);
    td.setUsage(MTLTextureUsage::ShaderRead);
    // SAFETY: `newTextureWithDescriptor:iosurface:plane:` with a live surface; +1 or nil.
    let t: Option<Retained<ProtocolObject<dyn MTLTexture>>> = unsafe {
        msg_send![d.get(), newTextureWithDescriptor: &*td, iosurface: sref(&s), plane: 0usize]
    };
    let t = reims_vgpu_metal::Obj::from_retained(t.unwrap());
    assert_eq!(count(&s), base + 2);
    {
        let got = b.iosurface();
        assert!(got.is_some());
        assert_eq!(count(&s), base + 3);
    }
    assert_eq!(
        count(&s),
        base + 2,
        "the surface's drop gives the reference back"
    );
    {
        let got = t.iosurface();
        assert!(got.is_some());
        assert_eq!(count(&s), base + 3);
    }
    assert_eq!(
        count(&s),
        base + 2,
        "the surface's drop gives the reference back"
    );
}

/// Through the crate: a nil surface reaches Metal (nil buffer), and the iosurface getters of
/// objects with no surface are None, making no reference
#[test]
fn crate_paths_without_a_surface() {
    let d = dev();
    assert!(d.new_buffer_on_surface(None).unwrap().is_none());
    let b = d
        .new_buffer_of_length(OnFault::Return, 64, 0)
        .unwrap()
        .unwrap();
    assert!(b.iosurface().is_none());
    let td = MTLTextureDescriptor::new();
    // SAFETY: plain property sets on a fresh descriptor.
    unsafe {
        td.setWidth(4);
        td.setHeight(4);
    }
    td.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    let td = reims_vgpu_metal::Obj::from_retained(td);
    let t = d.new_texture(OnFault::Return, &td).unwrap().unwrap();
    assert!(t.iosurface().is_none());
    // The count helper itself: a fresh surface has exactly this test's reference.
    let s = surface();
    assert_eq!(count(&s), 1);
}
