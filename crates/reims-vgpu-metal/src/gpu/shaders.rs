//! Our own shaders: the display's present functions and the XR10 paging kernels, in Metal Shading
//! Language ([`SOURCE`], `shaders.metal`), compiled at run time with
//! `newLibraryWithSource:options:error:`.
//!
//! A library belongs to the device that compiled it, so the source is compiled once per Metal
//! device, the first time core asks for it (a device's blit pipelines, a display's pipeline cache,
//! a service task's blit pipelines), and every later ask answers the same library. Devices are
//! told apart by registry ID, which a host's wrapper of the system device (a proxy that forwards
//! every message to it) shares with the device it wraps. A compiled library stays for the life of
//! the process; one that fails to compile is not kept, and the next ask compiles again.
//!
//! What each function computes is written beside it in the source; the tests below check it on
//! known pixels.

use objc2_metal::{MTLCompileOptions, MTLDevice as _};
use reims_vgpu_core::platform::Handle;
use reims_vgpu_core::platform::types::NsErrorText;
use reims_vgpu_core::sync::Mutex;

use super::device::{MetalDevice, create_or_die};
use super::resource::MetalLibrary;
use crate::msg;
use crate::spi::methods::DeviceSpi;

/// The shaders' Metal Shading Language.
pub const SOURCE: &str = include_str!("shaders.metal");

/// The libraries compiled so far, each with its device's registry ID.
static COMPILED: Mutex<Vec<(u64, MetalLibrary)>> = Mutex::new(Vec::new());

/// The shader library of `device` (a new reference), compiled on the first call for its registry
/// ID. `Err` is the compiler's `NSError`; no path reaches it under a handler, so a raise inside
/// Metal dies at the send.
pub(crate) fn library(device: &MetalDevice) -> Result<Option<MetalLibrary>, NsErrorText> {
    let d = device.get();
    let id = msg::get("registryID", || d.registryID());
    // Held through the compile, so two first asks compile once.
    let mut compiled = COMPILED.lock();
    if let Some((_, library)) = compiled.iter().find(|(r, _)| *r == id) {
        return Ok(Some(library.clone_ref()));
    }
    let made = compile(device);
    if let Ok(Some(library)) = &made {
        compiled.push((id, library.clone_ref()));
    }
    made
}

/// [`SOURCE`] compiled for `device` with Metal's default options.
fn compile(device: &MetalDevice) -> Result<Option<MetalLibrary>, NsErrorText> {
    let source = msg::new_string(SOURCE);
    let options = msg::get("new", MTLCompileOptions::new);
    let d = device.get();
    // SAFETY: `create_or_die`'s out-parameter.
    create_or_die("newLibraryWithSource:options:error:", |error| unsafe {
        DeviceSpi::newLibraryWithSource_options_error(d, &source, Some(&options), error)
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::ptr::NonNull;

    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer as _, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCommandQueue as _,
        MTLComputeCommandEncoder as _, MTLDevice, MTLFunction, MTLLibrary as _, MTLLoadAction,
        MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLRenderCommandEncoder as _,
        MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLResourceOptions, MTLSize,
        MTLStorageMode, MTLStoreAction, MTLTexture, MTLTextureDescriptor, MTLTextureType,
        MTLTextureUsage,
    };

    use super::*;
    use crate::platform::MetalPlatform;

    type Tex = Retained<ProtocolObject<dyn MTLTexture>>;
    type Dev = ProtocolObject<dyn MTLDevice>;

    /// The display's and the blits' five entry points.
    const FUNCTIONS: [&str; 5] = [
        "BlitInRGB_2P_XR10_A8",
        "BlitOutRGB_2P_XR10_A8",
        "displayPresentVertex",
        "displayPresentFragment",
        "displayPresentFragmentWithGammaTableAndColorMatrix",
    ];

    /// The 2P XR10 A8 pixel format (not public), its plane views' formats, and the XR10
    /// kernels' threadgroup, as core's `resources/texture/compute.rs` uses them.
    const XR10_2P_A8: usize = 550;
    const PLANE0: usize = 554;
    const PLANE1: usize = 84;

    fn device() -> MetalDevice {
        MetalPlatform::system_default_device().expect("a Metal device")
    }

    fn shader_library(d: &MetalDevice) -> MetalLibrary {
        library(d).unwrap().unwrap()
    }

    fn function(lib: &MetalLibrary, name: &str) -> Retained<ProtocolObject<dyn MTLFunction>> {
        lib.get()
            .newFunctionWithName(&NSString::from_str(name))
            .unwrap_or_else(|| panic!("no function {name}"))
    }

    fn region(w: usize, h: usize) -> MTLRegion {
        MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: w,
                height: h,
                depth: 1,
            },
        }
    }

    fn texture(d: &Dev, format: MTLPixelFormat, w: usize, h: usize) -> Tex {
        let td = MTLTextureDescriptor::new();
        // SAFETY: plain property sets on a fresh descriptor.
        unsafe {
            td.setPixelFormat(format);
            td.setWidth(w);
            td.setHeight(h);
            td.setUsage(
                MTLTextureUsage::ShaderRead
                    | MTLTextureUsage::ShaderWrite
                    | MTLTextureUsage::RenderTarget,
            );
        }
        d.newTextureWithDescriptor(&td).unwrap()
    }

    /// Uploads `px` (rows of `row` bytes) into slice `slice` of `t`.
    fn upload<T>(t: &Tex, px: &[T], w: usize, h: usize, row: usize, slice: usize) {
        // SAFETY: `px` holds `h` rows of `row` bytes.
        unsafe {
            t.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                region(w, h),
                0,
                slice,
                NonNull::new(px.as_ptr().cast_mut().cast::<c_void>()).unwrap(),
                row,
                0,
            );
        }
    }

    fn download<T: Default + Clone>(t: &Tex, w: usize, h: usize, row: usize) -> Vec<T> {
        let mut out = vec![T::default(); w * h];
        // SAFETY: `out` holds `h` rows of `row` bytes.
        unsafe {
            t.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                NonNull::new(out.as_mut_ptr().cast::<c_void>()).unwrap(),
                row,
                region(w, h),
                0,
            );
        }
        out
    }

    /// Runs one XR10 kernel over `w` x `h`, its three textures at 0, 1 and 2.
    fn run_kernel(d: &MetalDevice, name: &str, textures: [&Tex; 3], w: usize, h: usize) {
        let lib = shader_library(d);
        let pipeline =
            MTLDevice::newComputePipelineStateWithFunction_error(d.get(), &function(&lib, name))
                .unwrap();
        let q = d.get().newCommandQueue().unwrap();
        let cb = q.commandBuffer().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pipeline);
        for (i, t) in textures.iter().enumerate() {
            // SAFETY: a live texture at an index the kernel declares.
            unsafe { enc.setTexture_atIndex(Some(t), i) };
        }
        let grid = MTLSize {
            width: w,
            height: h,
            depth: 1,
        };
        let group = MTLSize {
            width: 32,
            height: 32,
            depth: 1,
        };
        enc.dispatchThreads_threadsPerThreadgroup(grid, group);
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
    }

    /// The display's strip (core's `display/render.rs`).
    const POS: [f32; 16] = [
        -1.0, -1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
    ];
    const TEX: [f32; 8] = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];

    /// The fragment properties: `matrix` as rows, then the slopes at 0 and at 1.
    fn properties(matrix: [[f32; 4]; 3], at_zero: [f32; 3], at_one: [f32; 3]) -> [f32; 24] {
        let mut p = [0.0f32; 24];
        for (col, column) in p.chunks_mut(4).take(4).enumerate() {
            for (row, v) in column.iter_mut().take(3).enumerate() {
                *v = matrix[row][col];
            }
        }
        p[16..19].copy_from_slice(&at_zero);
        p[19..22].copy_from_slice(&at_one);
        p
    }

    const IDENTITY: [[f32; 4]; 3] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];

    /// A gamma pass's table (`n` entries a channel, the same ramp in each slice) and properties.
    struct Gamma {
        ramp: Vec<f32>,
        properties: [f32; 24],
    }

    /// Draws the present pass from `frame` into a BGRA8 target of the frame's size: the plain
    /// copy with `gamma` `None`, else the gamma shader. The target's pixels, as BGRA.
    fn present(d: &MetalDevice, frame: &Tex, gamma: Option<&Gamma>) -> Vec<[u8; 4]> {
        let (w, h) = (frame.width(), frame.height());
        let lib = shader_library(d);
        let dst = texture(d.get(), MTLPixelFormat::BGRA8Unorm, w, h);
        let fragment = match gamma {
            None => "displayPresentFragment",
            Some(_) => "displayPresentFragmentWithGammaTableAndColorMatrix",
        };
        let pd = MTLRenderPipelineDescriptor::new();
        pd.setVertexFunction(Some(&function(&lib, "displayPresentVertex")));
        pd.setFragmentFunction(Some(&function(&lib, fragment)));
        // SAFETY: attachment 0 exists.
        unsafe {
            pd.colorAttachments()
                .objectAtIndexedSubscript(0)
                .setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        }
        let pipeline = MTLDevice::newRenderPipelineStateWithDescriptor_error(d.get(), &pd).unwrap();
        let table = gamma.map(|g| {
            let td = MTLTextureDescriptor::new();
            // SAFETY: plain property sets on a fresh descriptor.
            unsafe {
                td.setTextureType(MTLTextureType::Type1DArray);
                td.setPixelFormat(MTLPixelFormat::R32Float);
                td.setWidth(g.ramp.len());
                td.setArrayLength(3);
                td.setUsage(MTLTextureUsage::ShaderRead);
            }
            let t = d.get().newTextureWithDescriptor(&td).unwrap();
            for slice in 0..3 {
                upload(&t, &g.ramp, g.ramp.len(), 1, g.ramp.len() * 4, slice);
            }
            t
        });
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        // SAFETY: attachment 0 of a fresh pass descriptor.
        unsafe {
            let a = pass.colorAttachments().objectAtIndexedSubscript(0);
            a.setTexture(Some(&dst));
            a.setLoadAction(MTLLoadAction::Clear);
            a.setStoreAction(MTLStoreAction::Store);
        }
        let q = d.get().newCommandQueue().unwrap();
        let cb = q.commandBuffer().unwrap();
        let enc = cb.renderCommandEncoderWithDescriptor(&pass).unwrap();
        enc.setRenderPipelineState(&pipeline);
        // SAFETY: the bytes outlive the encoder's copy; the indices are the shaders'.
        unsafe {
            enc.setVertexBytes_length_atIndex(
                NonNull::new(POS.as_ptr().cast_mut().cast()).unwrap(),
                size_of_val(&POS),
                0,
            );
            enc.setVertexBytes_length_atIndex(
                NonNull::new(TEX.as_ptr().cast_mut().cast()).unwrap(),
                size_of_val(&TEX),
                1,
            );
            enc.setFragmentTexture_atIndex(table.as_deref(), 1);
            enc.setFragmentTexture_atIndex(Some(frame), 0);
            if let Some(g) = gamma {
                enc.setFragmentBytes_length_atIndex(
                    NonNull::new(g.properties.as_ptr().cast_mut().cast()).unwrap(),
                    size_of_val(&g.properties),
                    0,
                );
            }
            enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::TriangleStrip, 0, 4);
        }
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        download(&dst, w, h, w * 4)
    }

    /// An RGBA8 frame of `w` x `h` holding `px`.
    fn rgba8_frame(d: &MetalDevice, px: &[[u8; 4]], w: usize, h: usize) -> Tex {
        let t = texture(d.get(), MTLPixelFormat::RGBA8Unorm, w, h);
        upload(&t, px, w, h, w * 4, 0);
        t
    }

    fn bgra(p: [u8; 4]) -> [u8; 4] {
        [p[2], p[1], p[0], p[3]]
    }

    /// The source compiles on this host's Metal, with the five entry points
    #[test]
    fn the_source_compiles_with_the_five_functions() {
        let lib = shader_library(&device());
        for f in FUNCTIONS {
            assert!(
                lib.get()
                    .newFunctionWithName(&NSString::from_str(f))
                    .is_some(),
                "missing {f}"
            );
        }
        assert_eq!(lib.get().functionNames().count(), FUNCTIONS.len());
    }

    /// A device compiles once: every later ask answers the same library
    #[test]
    fn compiled_once_per_device() {
        let d = device();
        let a = shader_library(&d);
        let b = shader_library(&device());
        assert_eq!(a.id(), b.id());
        let id = d.get().registryID();
        assert_eq!(COMPILED.lock().iter().filter(|(r, _)| *r == id).count(), 1);
    }

    /// The copy draws the frame upright, pixel for pixel, alpha included: neither premultiplied
    /// nor blended
    #[test]
    fn the_copy_passes_pixels_and_alpha_through() {
        let d = device();
        let (w, h) = (8, 4);
        let px: Vec<[u8; 4]> = (0..w * h)
            .map(|i| {
                let i = u8::try_from(i).unwrap();
                [
                    i.wrapping_mul(37),
                    200 - i,
                    i.wrapping_mul(5) + 3,
                    [0, 0x7f, 0xff, 0x80][usize::from(i % 4)],
                ]
            })
            .collect();
        let out = present(&d, &rgba8_frame(&d, &px, w, h), None);
        let out: Vec<[u8; 4]> = out.into_iter().map(bgra).collect();
        assert_eq!(out, px);
    }

    /// An RGBA16Float frame is converted to unorm8 by the target: clamped, rounded to nearest
    #[test]
    fn the_copy_clamps_extended_range_in_the_target() {
        let d = device();
        let t = texture(d.get(), MTLPixelFormat::RGBA16Float, 2, 1);
        // -0.5, 0.5, 1.75, 0.25 and 1.0, 0.0, 0.5, 0.75, as half floats.
        let px: [[u16; 4]; 2] = [
            [0xb800, 0x3800, 0x3f00, 0x3400],
            [0x3c00, 0, 0x3800, 0x3a00],
        ];
        upload(&t, &px, 2, 1, 16, 0);
        let out: Vec<[u8; 4]> = present(&d, &t, None).into_iter().map(bgra).collect();
        assert_eq!(out, vec![[0, 0x80, 0xff, 0x40], [0xff, 0, 0x80, 0xbf]]);
    }

    /// With identity ramps (256 entries) and the identity matrix, the gamma shader is the copy
    /// for colour, and every alpha is 0xff
    #[test]
    fn identity_gamma_is_the_copy_with_opaque_alpha() {
        let d = device();
        let (w, h) = (16, 16);
        let px: Vec<[u8; 4]> = (0..w * h)
            .map(|i| {
                let i = u8::try_from(i).unwrap();
                [i, 255 - i, i.wrapping_mul(7), i.wrapping_mul(3)]
            })
            .collect();
        let ramp: Vec<f32> = (0..256u16).map(|i| f32::from(i) / 255.0).collect();
        let gamma = Gamma {
            ramp,
            properties: properties(IDENTITY, [1.0; 3], [1.0; 3]),
        };
        let out = present(&d, &rgba8_frame(&d, &px, w, h), Some(&gamma));
        for (o, p) in out.into_iter().map(bgra).zip(&px) {
            assert_eq!(o, [p[0], p[1], p[2], 0xff]);
        }
    }

    /// A two-entry ramp is a threshold at 0.5, not an interpolation
    #[test]
    fn a_two_entry_table_is_a_threshold() {
        let d = device();
        let px = [[0x7f, 0x80, 0x00, 0x10], [0xff, 0x40, 0xc0, 0x20]];
        let gamma = Gamma {
            ramp: vec![0.25, 0.75],
            properties: properties(IDENTITY, [0.5; 3], [0.5; 3]),
        };
        let out: Vec<[u8; 4]> = present(&d, &rgba8_frame(&d, &px, 2, 1), Some(&gamma))
            .into_iter()
            .map(bgra)
            .collect();
        // 0.25 is 63.75, 0.75 191.25, each rounded to nearest.
        assert_eq!(
            out,
            vec![[0x40, 0xbf, 0x40, 0xff], [0xbf, 0x40, 0xbf, 0xff]]
        );
    }

    /// Row c of the matrix gives channel c from the looked-up colour and the fourth column
    #[test]
    fn the_matrix_follows_the_table() {
        let d = device();
        let px = [[0xff, 0x00, 0x80, 0xff], [0x00, 0xff, 0x00, 0x00]];
        let ramp: Vec<f32> = (0..256u16).map(|i| f32::from(i) / 255.0).collect();
        // Red and blue swapped; green is half of green plus 0.25.
        let matrix = [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0, 0.25],
            [1.0, 0.0, 0.0, 0.0],
        ];
        let gamma = Gamma {
            ramp,
            properties: properties(matrix, [1.0; 3], [1.0; 3]),
        };
        let out: Vec<[u8; 4]> = present(&d, &rgba8_frame(&d, &px, 2, 1), Some(&gamma))
            .into_iter()
            .map(bgra)
            .collect();
        // 0.25 is 63.75 and 0.75 191.25, each rounded to nearest.
        assert_eq!(
            out,
            vec![[0x80, 0x40, 0xff, 0xff], [0x00, 0xbf, 0x00, 0xff]]
        );
    }

    /// Outside [0, 1] the table's end entry is extended by its slope at that end
    #[test]
    fn out_of_range_components_follow_the_end_slopes() {
        let d = device();
        let t = texture(d.get(), MTLPixelFormat::RGBA16Float, 1, 1);
        // -0.25, 1.5, 0.5 as half floats.
        upload(&t, &[[0xb400u16, 0x3e00, 0x3800, 0x3c00]], 1, 1, 8, 0);
        let gamma = Gamma {
            ramp: vec![0.2, 0.4, 0.6],
            // Red's slope at 0 is 0.4 (-0.25 gives 0.1); green's at 1 is 0.2 (1.5 gives 0.7).
            properties: properties(IDENTITY, [0.4, 9.0, 9.0], [9.0, 0.2, 9.0]),
        };
        let out: Vec<[u8; 4]> = present(&d, &t, Some(&gamma))
            .into_iter()
            .map(bgra)
            .collect();
        // 0.1 is 25.5, 0.7 178.5, 0.4 102.
        assert_eq!(out, vec![[0x1a, 0xb3, 0x66, 0xff]]);
    }

    /// A buffer of `bytes` and two plane views over it, as core's compute layout makes them:
    /// colour at 0, alpha at `alpha_at`, each four bytes a texel.
    fn planes(
        d: &MetalDevice,
        bytes: &[u8],
        w: usize,
        h: usize,
        alpha_at: usize,
    ) -> (
        Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        Tex,
        Tex,
    ) {
        let b = d
            .get()
            .newBufferWithLength_options(bytes.len(), MTLResourceOptions::StorageModeShared)
            .unwrap();
        // SAFETY: the buffer holds `bytes.len()` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                b.contents().as_ptr().cast(),
                bytes.len(),
            );
        }
        let view = |format: usize, offset: usize| {
            let td = MTLTextureDescriptor::new();
            // SAFETY: plain property sets on a fresh descriptor.
            unsafe {
                td.setPixelFormat(MTLPixelFormat(format));
                td.setWidth(w);
                td.setHeight(h);
                td.setStorageMode(MTLStorageMode::Shared);
                td.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
            }
            b.newTextureWithDescriptor_offset_bytesPerRow(&td, offset, w * 4)
                .unwrap()
        };
        let color = view(PLANE0, 0);
        let alpha = view(PLANE1, alpha_at);
        (b, color, alpha)
    }

    fn contents(b: &ProtocolObject<dyn objc2_metal::MTLBuffer>, len: usize) -> Vec<u8> {
        // SAFETY: the buffer holds at least `len` bytes, and the GPU is done with it.
        unsafe { std::slice::from_raw_parts(b.contents().as_ptr().cast::<u8>(), len) }.to_vec()
    }

    /// Paging a 2P XR10 A8 texture in and out keeps each texel's 30 colour bits, sets the two
    /// bits above them, and takes alpha from the alpha plane's second byte into all four
    #[test]
    fn xr10_round_trip_keeps_colour_and_alpha() {
        let d = device();
        let (w, h) = (64, 4);
        let plane = w * h * 4;
        let color: Vec<u8> = (0..plane).map(|i| ((i * 37 + 11) & 0xff) as u8).collect();
        let alpha: Vec<u8> = color.iter().rev().copied().collect();
        let mut bytes = color.clone();
        bytes.extend_from_slice(&alpha);
        let (b, color_view, alpha_view) = planes(&d, &bytes, w, h, plane);
        let tex = texture(d.get(), MTLPixelFormat(XR10_2P_A8), w, h);
        run_kernel(
            &d,
            "BlitInRGB_2P_XR10_A8",
            [&color_view, &alpha_view, &tex],
            w,
            h,
        );
        // SAFETY: the buffer holds `bytes.len()` bytes; the kernel is done.
        unsafe { std::ptr::write_bytes(b.contents().as_ptr().cast::<u8>(), 0, bytes.len()) };
        run_kernel(
            &d,
            "BlitOutRGB_2P_XR10_A8",
            [&tex, &color_view, &alpha_view],
            w,
            h,
        );
        let out = contents(&b, bytes.len());
        for (i, (got, sent)) in out[..plane].chunks(4).zip(color.chunks(4)).enumerate() {
            let got = u32::from_le_bytes(got.try_into().unwrap());
            let sent = u32::from_le_bytes(sent.try_into().unwrap());
            assert_eq!(got, sent | 0xc000_0000, "colour texel {i}");
        }
        for (i, (got, sent)) in out[plane..].chunks(4).zip(alpha.chunks(4)).enumerate() {
            assert_eq!(got, [sent[1]; 4], "alpha texel {i}");
        }
    }
}
