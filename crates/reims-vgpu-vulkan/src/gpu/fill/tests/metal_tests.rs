//! The tests of `fill` that compare with Metal on this host: macOS only.

use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::resource::Buffer;
use reims_vgpu_core::platform::values::TextureDesc;
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_metal::MetalPlatform;

use super::*;

/// Component values that reach every code's edges: zeroes, ones, NaNs, infinities, the
/// integer types' bounds and past them, the half float's, denormals, halves between every
/// step of the narrow normalized codes, a ramp past [-1, 1] and scattered values.
fn values() -> Vec<f64> {
    let mut v = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        f64::NAN,
        -f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1e-40,
        -1e-40,
        1e-6,
        6.1e-5,
        3e-8,
        65504.0,
        65519.0,
        65520.0,
        70000.0,
        127.0,
        128.0,
        -128.0,
        -129.0,
        255.0,
        256.0,
        1023.0,
        1024.0,
        65535.0,
        65536.0,
        32767.0,
        32768.0,
        -32768.0,
        -32769.0,
        2_147_483_647.0,
        2_147_483_648.0,
        -2_147_483_649.0,
        4_294_967_295.0,
        4_294_967_296.0,
        1e20,
        0.1,
        0.2,
        0.3,
        0.7,
        0.998_039_2,
        1.5,
        2.5,
        -1.5,
        -2.5,
    ];
    for den in [1.0, 3.0, 15.0, 31.0, 63.0, 127.0, 255.0, 1023.0] {
        for k in 0..64 {
            v.push((f64::from(k) + 0.5) / den);
            v.push(-(f64::from(k) + 0.5) / den);
        }
    }
    for den in [32767.0, 65535.0] {
        for k in (0..32767).step_by(1021) {
            v.push((f64::from(k) + 0.5) / den);
            v.push(-(f64::from(k) + 0.5) / den);
        }
    }
    for i in -250..1250 {
        v.push(f64::from(i) / 1000.0);
    }
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    for _ in 0..600 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let unit = (seed >> 11) as f64 / (1u64 << 53) as f64;
        v.push(unit * 3.0 - 1.0);
        v.push((unit - 0.5) * 600.0);
        v.push((unit - 0.5) * 2f64.powi(34));
    }
    v
}

/// Every colour format this platform packs.
fn packed_formats() -> Vec<u64> {
    PixelFormat::ALL
        .iter()
        .filter(|f| pack(**f, ClearColor::default()).is_some())
        .map(|f| u64::from(f.as_raw()))
        .collect()
}

/// Colours of `v`, each value in each component.
fn colors_of(v: &[f64]) -> Vec<ClearColor> {
    (0..v.len())
        .map(|i| ClearColor {
            red: v[i],
            green: v[(i + 1) % v.len()],
            blue: v[(i + 2) % v.len()],
            alpha: v[(i + 3) % v.len()],
        })
        .collect()
}

/// A colour of each pixel format this platform packs, for many colours, is the bytes Metal
/// writes here: an RGBA32Uint texel of each, its 16 bytes the colour's repeated
#[test]
fn colors_are_the_metal_platforms_here() {
    assert_metals(&packed_formats(), &colors_of(&values()));
}

/// `edge`, the `f64`s next to it, the `f32` nearest it and those next to that
fn beside(edge: f64) -> [f64; 6] {
    let next = |x: f64, by: i64| f64::from_bits(x.to_bits().wrapping_add_signed(by));
    #[expect(clippy::cast_possible_truncation, reason = "the f32 nearest the edge")]
    let nearest = edge as f32;
    let next32 = |x: f32, by: i32| f64::from(f32::from_bits(x.to_bits().wrapping_add_signed(by)));
    [
        edge,
        next(edge, -1),
        next(edge, 1),
        next32(nearest, -1),
        f64::from(nearest),
        next32(nearest, 1),
    ]
}

/// Values on and beside every sRGB and extended-range code's least value, and past each end
#[test]
fn srgb_and_extended_range_edges_are_the_metal_platforms_here() {
    let mut v = vec![-1e-30, 1e-30, 0.999_999_9, 1.000_000_1, 1.5, -0.8];
    for &edge in srgb_edges() {
        v.extend(beside(edge));
    }
    for code in 0..=1024 {
        v.extend(beside((f64::from(code) - 384.5) / 510.0));
    }
    let formats = [11, 31, 71, 81, 552, 554];
    assert_metals(&formats, &colors_of(&v));
}

/// Values on and beside the halfway points of RG11B10Float's and RGB9E5Float's mantissas, at
/// every exponent and below the least normal, and past their largest values
#[test]
fn small_and_shared_exponent_float_edges_are_the_metal_platforms_here() {
    let mut v = vec![
        65024.0, 65279.0, 65280.0, 64512.0, 64767.0, 64768.0, 65408.0, 65472.0,
    ];
    v.extend([4e38, 3e38, 1e-40, -1e-40]);
    for mantissa in [5, 6] {
        for exponent in -21..=15 {
            let step = 2f64.powi(exponent.max(-14) - mantissa);
            let base = if exponent < -14 {
                0.0
            } else {
                2f64.powi(exponent)
            };
            for q in [0, 1, 2, 3, (1 << mantissa) - 2, (1 << mantissa) - 1] {
                v.extend(beside(base + (f64::from(q) + 0.5) * step));
            }
        }
    }
    let mut colors = colors_of(&v);
    for exponent in -16..=16 {
        // The largest component sets the exponent, whose step is 2^(exponent - 8).
        let largest = 1.5 * 2f64.powi(exponent);
        let step = 2f64.powi(exponent - 8);
        for m in [0, 1, 2, 255] {
            for c in beside((f64::from(m) + 0.5) * step) {
                colors.extend([
                    ClearColor {
                        red: largest,
                        green: c,
                        blue: 0.0,
                        alpha: 0.0,
                    },
                    ClearColor {
                        red: c,
                        green: 0.0,
                        blue: largest,
                        alpha: 0.0,
                    },
                ]);
            }
        }
        for m in [256, 300, 511] {
            for c in beside((f64::from(m) + 0.5) * step) {
                colors.push(ClearColor {
                    red: 0.0,
                    green: c,
                    blue: 0.0,
                    alpha: 0.0,
                });
            }
        }
    }
    assert_metals(&[92, 93], &colors);
}

/// NaNs of either sign, quiet and signalling, whose payloads set the bits a half, a float
/// and a bfloat keep and the bits below them, in every pixel format this platform packs; and
/// doubles on and beside the 8-, 16- and 32-bit integer types' edges (2^31 and 2^32 and
/// their negatives among them, and those strictly between 2^31 - 1 and 2^31 and between
/// 2^32 - 1 and 2^32), and doubles strictly between two integers inside each type's range,
/// far from its edges (2^24 + 0.5, 3e9 + 0.5 and their kin, of either sign), in the integer
/// formats
#[test]
fn nan_payloads_and_integer_edges_are_the_metal_platforms_here() {
    let sign = 1u64 << 63;
    let nans = [
        0x7ff8_0000_0000_0000u64,
        0x7ff8_0000_0000_0001,
        0x7ff8_0000_1000_0000,
        0x7ff8_0000_2000_0000,
        0x7ff8_0200_0000_0000,
        0x7ff8_0400_0000_0000,
        0x7ff8_1000_0000_0000,
        0x7ff8_2000_0000_0000,
        0x7ffa_5a5a_5a5a_5a5a,
        0x7ffc_0000_0000_0000,
        0x7fff_ffff_ffff_ffff,
        0x7ff0_0000_0000_0001,
        0x7ff0_0000_2000_0000,
        0x7ff0_0400_0000_0000,
        0x7ff0_2000_0000_0000,
        0x7ff4_0000_0000_0000,
        0x7ff5_a5a5_a5a5_a5a5,
        0x7ff7_ffff_ffff_ffff,
    ];
    let mut v: Vec<f64> = nans
        .iter()
        .flat_map(|&bits| [f64::from_bits(bits), f64::from_bits(bits | sign)])
        .collect();
    v.extend([0.0, 1.0, -1.0, 0.5]);
    assert_metals(&packed_formats(), &colors_of(&v));
    let mut v = Vec::new();
    for edge in [
        2f64.powi(31) - 1.0,
        2f64.powi(31),
        2f64.powi(32) - 1.0,
        2f64.powi(32),
        -(2f64.powi(31)),
        -(2f64.powi(31)) - 1.0,
        -(2f64.powi(32)),
        127.0,
        128.0,
        255.0,
        256.0,
        32767.0,
        32768.0,
        65535.0,
        65536.0,
        -128.0,
        -129.0,
        -32768.0,
        -32769.0,
        1023.0,
        1024.0,
    ] {
        v.extend(beside(edge));
        v.extend([-0.75, -0.5, -0.25, 0.25, 0.5].map(|by| edge + by));
    }
    for inside in [
        100.5,
        200.25,
        1000.75,
        30_000.5,
        65_534.5,
        123_456.5,
        16_777_216.5,
        33_554_433.5,
        1_000_000_000.25,
        1_073_741_824.5,
        2_147_483_648.5,
        3_000_000_000.5,
        4_000_000_000.75,
        4_294_967_294.5,
    ] {
        v.extend([inside, -inside]);
    }
    let integers = [13, 14, 23, 24, 53, 54, 73, 74, 91, 113, 114, 123, 124];
    assert_metals(&integers, &colors_of(&v));
}

/// Asserts that each of `colors`, in each pixel format of `formats`, is the bytes Metal
/// writes here, where there is a Metal device: the colours fill an RGBA32Uint texel each, of
/// a texture 4096 wide.
fn assert_metals(formats: &[u64], colors: &[ClearColor]) {
    let Some(d) = MetalPlatform::system_default_device() else {
        return;
    };
    let width = 4096u64;
    let height = (colors.len() as u64).div_ceil(width);
    let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&TextureDesc {
        pixel_format: 123,
        width,
        height,
        usage: 3,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    });
    let t = d.new_texture(OnFault::Return, &desc).unwrap().unwrap();
    let out = d
        .new_buffer_of_length(OnFault::Return, width * height * 16, 0)
        .unwrap()
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let mut wrong = Vec::new();
    for &format in formats {
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        for (i, color) in (0u64..).zip(colors) {
            let fill = BlitCmd::<MetalPlatform>::FillTextureColor {
                texture: &t,
                level: 0,
                slice: 0,
                region: MtlRegion::new(
                    MtlOrigin::new(i % width, i / width, 0),
                    MtlSize::new(1, 1, 1),
                ),
                color: *color,
                pixel_format: format,
            };
            e.encode(OnFault::Return, fill).unwrap();
        }
        let down = BlitCmd::<MetalPlatform>::CopyTextureToBufferOptions {
            source_texture: Some(&t),
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(width, height, 1),
            destination_buffer: Some(&out),
            destination_offset: 0,
            destination_bytes_per_row: width * 16,
            destination_bytes_per_image: width * height * 16,
            options: 0,
        };
        e.encode(OnFault::Return, down).unwrap();
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        let length = usize::try_from(width * height * 16).unwrap();
        // SAFETY: a live Shared buffer of `length` bytes, read after the GPU's writes.
        let metal = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), length) };
        for (color, metal) in colors.iter().zip(metal.chunks(16)) {
            let ours = texel(&color_bytes("test", *color, format), 16);
            if ours != metal {
                wrong.push(format!(
                    "{format} {color:?}: ours {ours:02x?} Metal {metal:02x?}"
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} differ, the first: {:#?}",
        wrong.len(),
        colors.len() * formats.len(),
        &wrong[..wrong.len().min(40)]
    );
}

/// What Metal writes for a colour in a format it packs none in, on this host: never the
/// colour's bytes, but those the fill before it packed, into another texture, whatever the
/// colour; this platform dies there
#[test]
fn metal_fills_a_colour_it_packs_none_in_with_an_earlier_fills_bytes() {
    for format in LEFTOVER_FORMATS {
        let raw = u64::from(format.as_raw());
        let site = catch_death(|| color_bytes("s", ClearColor::default(), raw)).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "s"),
            "{format:?}"
        );
    }
    let Some(d) = MetalPlatform::system_default_device() else {
        return;
    };
    let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&TextureDesc {
        pixel_format: 123,
        width: 1,
        height: 1,
        usage: 3,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    });
    let texture = || d.new_texture(OnFault::Return, &desc).unwrap().unwrap();
    let (earlier_texture, t) = (texture(), texture());
    let out = d
        .new_buffer_of_length(OnFault::Return, 16, 0)
        .unwrap()
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let one = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(1, 1, 1));
    let asked = ClearColor {
        red: 0.25,
        green: 0.5,
        blue: 0.75,
        alpha: 1.0,
    };
    for (red, format) in (7u8..).zip(LEFTOVER_FORMATS) {
        let earlier = ClearColor {
            red: f64::from(red),
            green: 9.0,
            blue: 3.0,
            alpha: 1.0,
        };
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        let fills = [
            (&earlier_texture, earlier, 123),
            (&t, asked, u64::from(format.as_raw())),
        ];
        for (texture, color, pixel_format) in fills {
            let fill = BlitCmd::<MetalPlatform>::FillTextureColor {
                texture,
                level: 0,
                slice: 0,
                region: one,
                color,
                pixel_format,
            };
            e.encode(OnFault::Return, fill).unwrap();
        }
        let down = BlitCmd::<MetalPlatform>::CopyTextureToBufferOptions {
            source_texture: Some(&t),
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(1, 1, 1),
            destination_buffer: Some(&out),
            destination_offset: 0,
            destination_bytes_per_row: 16,
            destination_bytes_per_image: 16,
            options: 0,
        };
        e.encode(OnFault::Return, down).unwrap();
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        // SAFETY: a live Shared buffer of 16 bytes, read after the GPU's writes.
        let metal = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), 16) };
        assert_eq!(
            metal,
            [red, 0, 0, 0, 9, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0],
            "{format:?}"
        );
    }
}
