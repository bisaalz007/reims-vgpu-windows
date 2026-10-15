//! The measurement harness of the YCbCr family: surfaces of the shapes the family's formats are
//! made over, filled with codes, and what a texture of any of the family's formats over one
//! reads, on Metal and on this platform, through the probe kernels (`ycbcr_probe.metal`), into
//! halves and into floats. The same probes run for every member: a surface's shape is its
//! OSType's ([`Shape`]), and the texture's format is a parameter of each probe ([`probes`]).
//! Floats show a texel's value at full precision; a kernel that reads into halves rounds it.
use reims_vgpu_core::platform::os::surface::{IoSurfaces as _, SurfacePlaneSpec, SurfaceSpec};

use crate::os::{OsSurface, OsSurfaces};

/// How a surface of two planes lays out its samples: plane 0 the luma, plane 1 Cb and Cr
/// interleaved, each sample a code in the top bits of `container` bytes, or, `packed`, three
/// codes to a little-endian 32-bit word, at its bits 0, 10 and 20 (bits 30 and 31 unread): luma
/// code `x` of a row in word `x / 3`, the chroma pair `x` as codes `2 x` (Cb) and `2 x + 1`
/// (Cr).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Shape {
    /// The surface's OSType.
    pub ostype: [u8; 4],
    /// The chroma plane's subsampling: the luma texels per chroma texel across and down, (2, 2)
    /// for 4:2:0, (2, 1) for 4:2:2 and (1, 1) for 4:4:4.
    pub subsampling: (u64, u64),
    /// The bits of a code.
    pub bits: u32,
    /// The bytes a sample is stored in: of a packed shape, the 4 bytes of a word of three.
    pub container: u64,
    /// Whether three codes share each 32-bit word.
    pub packed: bool,
}

impl Shape {
    /// The shape CoreVideo gives `ostype`, one of the two-plane OSTypes measured.
    pub(crate) fn of(ostype: &[u8; 4]) -> Shape {
        let (subsampling, bits, container) = match ostype {
            b"420v" | b"420f" => ((2, 2), 8, 1),
            b"422v" | b"422f" => ((2, 1), 8, 1),
            b"444v" | b"444f" => ((1, 1), 8, 1),
            b"x420" | b"xf20" => ((2, 2), 10, 2),
            b"x422" | b"xf22" => ((2, 1), 10, 2),
            b"x444" | b"xf44" => ((1, 1), 10, 2),
            b"p420" | b"pf20" => ((2, 2), 10, 4),
            b"p422" | b"pf22" => ((2, 1), 10, 4),
            b"p444" | b"pf44" => ((1, 1), 10, 4),
            other => panic!("no shape for {}", String::from_utf8_lossy(other)),
        };
        Shape {
            ostype: *ostype,
            subsampling,
            bits,
            container,
            packed: container == 4,
        }
    }

    /// The chroma plane's size for a luma plane of `size`.
    pub(crate) fn chroma_size(&self, (width, height): (u64, u64)) -> (u64, u64) {
        (
            width.div_ceil(self.subsampling.0),
            height.div_ceil(self.subsampling.1),
        )
    }

    /// The bytes of a row of `codes` codes: a sample each, or a word of each three, packed.
    pub(crate) fn row_bytes(&self, codes: u64) -> u64 {
        match self.packed {
            true => codes.div_ceil(3) * 4,
            false => codes * self.container,
        }
    }

    /// The spec of a host-only surface of this shape whose luma plane is `size`: rows of a
    /// multiple of 64 bytes and plane 1 at a multiple of 4096, as CoreVideo lays them out.
    /// IOSurface makes none of a surface's bytes per row of 0, so the surface's are plane 0's.
    /// A packed plane's element is a word of three luma codes (3 texels in 4 bytes), or two
    /// words of three chroma pairs (3 texels in 8 bytes).
    pub(crate) fn spec(&self, size: (u64, u64)) -> SurfaceSpec {
        let (width, height) = size;
        let (chroma_width, chroma_height) = self.chroma_size(size);
        let (element, element_width) = match self.packed {
            true => (4, 3),
            false => (self.container, 1),
        };
        let (luma_row, chroma_row) = (
            self.row_bytes(width).next_multiple_of(64),
            self.row_bytes(chroma_width * 2).next_multiple_of(64),
        );
        let luma_size = luma_row * height;
        let chroma_offset = luma_size.next_multiple_of(4096);
        let chroma_size = chroma_row * chroma_height;
        let plane =
            |offset, (width, height), bytes_per_row, bytes_per_element, size| SurfacePlaneSpec {
                base: offset,
                offset,
                width,
                height,
                size,
                bytes_per_row,
                bytes_per_element,
                element_width,
                element_height: 1,
                ..SurfacePlaneSpec::default()
            };
        SurfaceSpec {
            planes: Some(vec![
                plane(0, (width, height), luma_row, element, luma_size),
                plane(
                    chroma_offset,
                    (chroma_width, chroma_height),
                    chroma_row,
                    element * 2,
                    chroma_size,
                ),
            ]),
            pixel_format: u32::from_be_bytes(self.ostype),
            alloc_size: (chroma_offset + chroma_size).next_multiple_of(16384),
            width,
            height,
            bytes_per_element: element,
            bytes_per_row: luma_row,
            element_width,
            element_height: 1,
            ..SurfaceSpec::default()
        }
    }

    /// A host-only surface of this shape whose luma plane is `size` ([`Shape::spec`]).
    pub(crate) fn surface(&self, size: (u64, u64)) -> OsSurface {
        OsSurfaces::new().create(&self.spec(size)).unwrap()
    }
}

#[cfg(target_os = "macos")]
pub(crate) mod metal;
#[cfg(target_os = "macos")]
pub(crate) use metal::*;

/// The probes of a format's sampling over surfaces, each printing what Metal answers, read into
/// floats, for a texture of the format `RVG_YCBCR_FORMAT` names (505 by default) over a surface of
/// each OSType `RVG_YCBCR_OSTYPES` lists, comma-separated ('x420,xf20' by default): the
/// measurements a member of the family starts from. Each is run by hand:
/// `RVG_YCBCR_FORMAT=505 cargo nextest run -p reims-vgpu-vulkan --run-ignored only
/// -E 'test(/measure::probes/)' --no-capture`.
#[cfg(target_os = "macos")]
mod probes {
    use reims_vgpu_metal::MetalPlatform;

    use super::metal::{metals, set_value};
    use super::*;
    use crate::os::MATRIX_KEY;

    /// The format probed.
    fn format() -> u64 {
        std::env::var("RVG_YCBCR_FORMAT").map_or(505, |v| v.parse().unwrap())
    }

    /// The shapes of the OSTypes probed.
    fn shapes() -> Vec<Shape> {
        let list = std::env::var("RVG_YCBCR_OSTYPES").unwrap_or_else(|_| "x420,xf20".to_owned());
        list.split(',')
            .map(|t| Shape::of(t.as_bytes().try_into().unwrap()))
            .collect()
    }

    /// The largest code of `shape`.
    fn largest(shape: &Shape) -> u16 {
        u16::try_from((1u32 << shape.bits) - 1).unwrap()
    }

    /// The OSType's name.
    fn named(shape: &Shape) -> String {
        String::from_utf8_lossy(&shape.ostype).into_owned()
    }

    /// Metal's `read()` into floats of a texture of the format over `s` of `size`.
    fn floats(s: &OsSurface, size: (u64, u64)) -> Vec<[f32; 4]> {
        let (md, mt) = metals(format(), s, 0, size).unwrap();
        read_floats::<MetalPlatform>(&md, &mt, size)
    }

    /// What a texel reads where every luma code is Y and every chroma code (Cb, Cr): which channel
    /// carries which code, and any offset, in units of a code (the value times the largest code)
    /// and as the floats' bits.
    #[test]
    #[ignore = "a measurement, run by hand: prints Metal's answers"]
    fn constants() {
        for shape in shapes() {
            let top = largest(&shape);
            let third = top / 3;
            for (y, cb, cr) in [
                (0, 0, 0),
                (top, top, top),
                (third, 2 * third, top),
                (top / 2, 0, top / 4),
            ] {
                let size = (8, 8);
                let s = shape.surface(size);
                shape.fill_codes(&s, size, |_, _| y, |_, _| (cb, cr));
                let t = floats(&s, size)[0];
                let unit = f32::from(top);
                eprintln!(
                    "{} over {}: Y {y} Cb {cb} Cr {cr} reads ({:.6}, {:.6}, {:.6}, alpha {}) \
                     bits {:08x?}",
                    format(),
                    named(&shape),
                    t[0] * unit,
                    t[1] * unit,
                    t[2] * unit,
                    t[3],
                    t.map(f32::to_bits)
                );
            }
        }
    }

    /// Each code of the luma, then of Cb and Cr, alone: which channels it drives, and of the
    /// codes of each channel it drives, how many read as the float nearest the code over the
    /// largest code, and over the number of codes. The chroma's codes are laid so that the luma
    /// column read meets one code in every chroma texel a 2-tap filter reaches.
    #[test]
    #[ignore = "a measurement, run by hand: prints Metal's answers"]
    fn code_sweeps() {
        for shape in shapes() {
            let n = 1u64 << shape.bits;
            let (top, middle) = (n - 1, u16::try_from(n / 2).unwrap());
            let code = |c: u64| u16::try_from(c).unwrap();
            let size = (n, 2);
            let s = shape.surface(size);
            shape.fill_codes(&s, size, |x, _| code(x), |_, _| (middle, middle));
            let read = floats(&s, size);
            report(&shape, "Y", &|k| read[usize::try_from(k).unwrap()], &|k| k);
            let size = (4 * n, 2);
            let s = shape.surface(size);
            let across = shape.subsampling.0;
            shape.fill_codes(
                &s,
                size,
                |_, _| middle,
                |x, _| (code((x * across) >> 2), code(top - ((x * across) >> 2))),
            );
            let read = floats(&s, size);
            let at = |k: u64| read[usize::try_from(4 * k + 1).unwrap()];
            report(&shape, "Cb", &at, &|k| k);
            report(&shape, "Cr", &at, &|k| top - k);
        }
    }

    /// [`code_sweeps`]'s report of the code `what`, which is `code(k)` where texel `at(k)` reads.
    fn report(shape: &Shape, what: &str, at: &dyn Fn(u64) -> [f32; 4], code: &dyn Fn(u64) -> u64) {
        let n = 1u64 << shape.bits;
        let float = |c: u64| f32::from(u16::try_from(c).unwrap());
        for channel in 0..4 {
            let driven = (1..n).any(|k| at(k)[channel] != at(0)[channel]);
            if !driven {
                continue;
            }
            let over = |den: u64| {
                (0..n)
                    .filter(|&k| at(k)[channel] == float(code(k)) / float(den))
                    .count()
            };
            eprintln!(
                "{} over {}: {what} drives channel {channel}: {} of {n} codes read code / {}, {} \
                 read code / {n}",
                format(),
                named(shape),
                over(n - 1),
                n - 1,
                over(n)
            );
        }
    }

    /// The response to one chroma texel of Cb high and Cr low among Cb low and Cr high, at chroma
    /// (3, 3): each channel of the 12 by 12 luma texels around it, in units of a code.
    #[test]
    #[ignore = "a measurement, run by hand: prints Metal's answers"]
    fn chroma_siting() {
        for shape in shapes() {
            let top = largest(&shape);
            let (low, high, middle) = (top / 64, top - top / 64, top / 2);
            let size = (16, 16);
            let s = shape.surface(size);
            shape.fill_codes(
                &s,
                size,
                |_, _| middle,
                |x, y| {
                    if (x, y) == (3, 3) {
                        (high, low)
                    } else {
                        (low, high)
                    }
                },
            );
            let read = floats(&s, size);
            let unit = f32::from(top);
            for channel in 0..3 {
                eprintln!("{} over {}: channel {channel}", format(), named(&shape));
                for (y, texels) in read.chunks(16).take(12).enumerate() {
                    let row: Vec<String> = texels[..12]
                        .iter()
                        .map(|t| format!("{:9.3}", t[channel] * unit))
                        .collect();
                    eprintln!("  {y:2} {}", row.join(""));
                }
            }
        }
    }

    /// Whether the reads of the same random samples change with the OSType (each against the
    /// first listed), with `IOSurfaceYCbCrMatrix` set to BT.709 or BT.2020, or with the samples'
    /// low bits cleared.
    #[test]
    #[ignore = "a measurement, run by hand: prints Metal's answers"]
    fn independence() {
        let size = (33, 17);
        let read = |shape: &Shape, matrix: Option<&str>, clear: bool| {
            let s = shape.surface(size);
            if let Some(m) = matrix {
                set_value(&s, MATRIX_KEY, m);
            }
            let low = u32::try_from(shape.container * 8).unwrap() - shape.bits;
            let mask = if clear {
                !((1u16 << low) - 1)
            } else {
                u16::MAX
            };
            let mut next = noise(9);
            let (width, height) = size;
            let (chroma_width, chroma_height) = shape.chroma_size(size);
            let luma: Vec<u16> = (0..width * height).map(|_| next() & mask).collect();
            let chroma: Vec<(u16, u16)> = (0..chroma_width * chroma_height)
                .map(|_| (next() & mask, next() & mask))
                .collect();
            let at = |x: u64, y: u64, w: u64| usize::try_from(y * w + x).unwrap();
            shape.fill(
                &s,
                size,
                |x, y| luma[at(x, y, width)],
                |x, y| chroma[at(x, y, chroma_width)],
            );
            floats(&s, size)
        };
        let shapes = shapes();
        let first = &shapes[0];
        let base = read(first, None, false);
        let differ = |other: Vec<[f32; 4]>| base.iter().zip(&other).filter(|(a, b)| a != b).count();
        for shape in &shapes[1..] {
            let n = differ(read(shape, None, false));
            eprintln!(
                "{} over {}, against {}: {n} texels differ",
                format(),
                named(shape),
                named(first)
            );
        }
        for (what, matrix, clear) in [
            ("BT.709", Some("ITU_R_709_2"), false),
            ("BT.2020", Some("ITU_R_2020"), false),
            ("low bits cleared", None, true),
        ] {
            let n = differ(read(first, matrix, clear));
            eprintln!(
                "{} over {}, {what}: {n} texels differ",
                format(),
                named(first)
            );
        }
    }

    /// How each address mode (and each border colour) samples against `clamp_to_edge`, inside the
    /// texture and outside it, in normalized coordinates from 1 texel outside to 1 past the far
    /// edge: nearest and linear `sample()`, and `gather()` of red and of blue.
    #[test]
    #[ignore = "a measurement, run by hand: prints Metal's answers"]
    fn address_modes() {
        let f = format();
        let size = (40, 24);
        for shape in shapes() {
            let s = shape.surface(size);
            shape.fill_noise(&s, size, 7);
            let (md, mt) = metals(f, &s, 0, size).unwrap();
            let grid = ((size.0 + 2) * 3, (size.1 + 2) * 3);
            let inside = |i: u64| {
                let (x, y) = (i % grid.0, i / grid.0);
                (3..3 * (size.0 + 1)).contains(&x) && (3..3 * (size.1 + 1)).contains(&y)
            };
            for (read, filter) in [(0, 0), (0, 1), (1, 0), (2, 0)] {
                let bytes = probe((-0.9, -0.9), 1.0 / 3.0, grid, true, read);
                let sample = |address: u64, border: u64| {
                    let s = sampler::<MetalPlatform>(&md, (filter, address, true), border);
                    let read = super::probed_texels::<MetalPlatform>(
                        &md,
                        ("sample_texels_float", 16),
                        &mt,
                        grid,
                        Some(&s),
                        Some(&bytes),
                    );
                    float_texels(&read)
                };
                let edge = sample(0, 0);
                for (address, border) in [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (5, 1), (5, 2)] {
                    let (mut differ_in, mut differ_out) = (0, 0);
                    for (i, (a, b)) in (0u64..).zip(sample(address, border).iter().zip(&edge)) {
                        if a != b {
                            if inside(i) {
                                differ_in += 1;
                            } else {
                                differ_out += 1;
                            }
                        }
                    }
                    eprintln!(
                        "{f} over {}: read {read} filter {filter} address {address} border \
                         {border}: {differ_in} inside and {differ_out} outside differ from \
                         clamp_to_edge",
                        named(&shape)
                    );
                }
            }
        }
    }
}
