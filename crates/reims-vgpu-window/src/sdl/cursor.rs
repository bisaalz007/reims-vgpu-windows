//! The host's cursor over a window's image: the guest's glyph as SDL's colour cursor, sized as the
//! image shows it, or hidden while the guest hides its own. UI thread only.
//!
//! The glyph's RGBA bytes are straight alpha, as the guest's glyph comes, so a guest's cursor
//! looks as it does on macOS and in QEMU's own displays. SDL takes the cursor's image at the
//! screen's 100% scale, in points, and an alternate image for the screen's own scale, in pixels
//! (`SDL_AddSurfaceAlternateImage`, with `SDL_HINT_MOUSE_DPI_SCALE_CURSORS` on): at 190% a guest
//! one to one with the screen's pixels gets one glyph pixel per screen pixel. Each image is the
//! glyph resampled to its size by nearest pixel ([`resample`]), so a 1× guest's glyph keeps its
//! hard edges as the image scales it.

use sdl3_sys::everything::{
    SDL_AddSurfaceAlternateImage, SDL_CreateColorCursor, SDL_CreateSurface, SDL_Cursor,
    SDL_DestroyCursor, SDL_DestroySurface, SDL_PIXELFORMAT_RGBA32, SDL_Surface,
};

use crate::geometry;
use crate::handle::Glyph;

/// A cursor SDL made, destroyed with this.
#[derive(Debug)]
pub(crate) struct Cursor(*mut SDL_Cursor);

impl Cursor {
    /// SDL's cursor.
    pub(crate) fn raw(&self) -> *mut SDL_Cursor {
        self.0
    }
}

impl Drop for Cursor {
    fn drop(&mut self) {
        // SAFETY: the cursor is SDL's, made on the UI thread, where this runs; SDL falls back to
        // its default cursor where this one is set.
        unsafe { SDL_DestroyCursor(self.0) };
    }
}

/// The cursor built from a glyph, and what it was built from.
#[derive(Debug)]
struct Built {
    /// The glyph.
    glyph: Glyph,
    /// Points per guest pixel.
    points_per_pixel: f64,
    /// The screen's pixels per point.
    density: f64,
    /// The cursor.
    cursor: Cursor,
}

/// The cursor built for one window, built again when the glyph or a scale changes.
#[derive(Debug, Default)]
pub(crate) struct CursorCache {
    /// The last cursor built from the guest's glyph.
    built: Option<Built>,
}

impl CursorCache {
    /// The guest's `glyph` as a cursor drawn at `points_per_pixel` points per guest pixel, on a
    /// screen of `density` pixels per point; `None` (the host's arrow) before the guest has set
    /// a glyph, for one with no pixels, or where SDL makes none. An unchanged request answers the
    /// cursor it answered before.
    pub(crate) fn cursor(
        &mut self,
        glyph: Option<&Glyph>,
        points_per_pixel: f64,
        density: f64,
    ) -> Option<&Cursor> {
        let glyph = glyph?;
        let same = self.built.as_ref().is_some_and(|b| {
            b.points_per_pixel.to_bits() == points_per_pixel.to_bits()
                && b.density.to_bits() == density.to_bits()
                && b.glyph == *glyph
        });
        if !same {
            self.built = build(glyph, points_per_pixel, density).map(|cursor| Built {
                glyph: glyph.clone(),
                points_per_pixel,
                density,
                cursor,
            });
        }
        self.built.as_ref().map(|b| &b.cursor)
    }
}

/// `glyph` as a cursor drawn at `points_per_pixel` points per glyph pixel on a screen of
/// `density` (the module doc), or `None`.
fn build(glyph: &Glyph, points_per_pixel: f64, density: f64) -> Option<Cursor> {
    let size = geometry::cursor_size(
        (glyph.width, glyph.height),
        glyph.hot_spot,
        points_per_pixel,
    );
    let whole = |v: f64| v.round().max(1.0) as u32;
    let points = (whole(size.size.0), whole(size.size.1));
    let base = surface(glyph, points)?;
    let density = if density.is_finite() && density > 1.0 {
        density
    } else {
        1.0
    };
    if density > 1.0 {
        let pixels = (whole(size.size.0 * density), whole(size.size.1 * density));
        if let Some(alternate) = surface(glyph, pixels) {
            // SAFETY: both surfaces are live; SDL takes a reference to the alternate.
            unsafe {
                SDL_AddSurfaceAlternateImage(base, alternate);
                SDL_DestroySurface(alternate);
            }
        }
    }
    let hot = |v: f64, extent: u32| (v.round() as i32).clamp(0, extent.saturating_sub(1) as i32);
    // SAFETY: `base` is a live RGBA surface; SDL copies what it needs into the cursor.
    let cursor = unsafe {
        let cursor = SDL_CreateColorCursor(
            base,
            hot(size.hot_spot.0, points.0),
            hot(size.hot_spot.1, points.1),
        );
        SDL_DestroySurface(base);
        cursor
    };
    (!cursor.is_null()).then_some(Cursor(cursor))
}

/// An RGBA surface of `size` pixels holding `glyph` resampled to it, or `None` for a glyph with
/// no pixels, or where SDL makes no surface.
fn surface(glyph: &Glyph, size: (u32, u32)) -> Option<*mut SDL_Surface> {
    let pixels = resample(glyph, size)?;
    let (width, height) = (i32::try_from(size.0).ok()?, i32::try_from(size.1).ok()?);
    // SAFETY: a valid size and format.
    let surface = unsafe { SDL_CreateSurface(width, height, SDL_PIXELFORMAT_RGBA32) };
    if surface.is_null() {
        return None;
    }
    // SAFETY: the surface is SDL's new one, `height` rows of at least `width * 4` bytes, `pitch`
    // apart, which nothing else touches.
    unsafe {
        let s = &*surface;
        let pitch = usize::try_from(s.pitch).ok()?;
        let row = usize::try_from(size.0).ok()?.checked_mul(4)?;
        for (y, line) in pixels.chunks_exact(row).enumerate() {
            let at = s.pixels.cast::<u8>().add(y * pitch);
            std::ptr::copy_nonoverlapping(line.as_ptr(), at, row);
        }
    }
    Some(surface)
}

/// `glyph`'s pixels resampled to `size` by nearest pixel: `size.1` rows of `size.0 * 4` RGBA
/// bytes. A pixel the glyph's bytes do not cover is transparent. `None` for a glyph or a size
/// with no pixels.
pub(crate) fn resample(glyph: &Glyph, size: (u32, u32)) -> Option<Vec<u8>> {
    let (gw, gh) = (
        usize::try_from(glyph.width).ok()?,
        usize::try_from(glyph.height).ok()?,
    );
    let (w, h) = (usize::try_from(size.0).ok()?, usize::try_from(size.1).ok()?);
    if gw == 0 || gh == 0 || w == 0 || h == 0 {
        return None;
    }
    let mut out = vec![0u8; w.checked_mul(h)?.checked_mul(4)?];
    for (y, row) in out.chunks_exact_mut(w * 4).enumerate() {
        let sy = y * gh / h;
        for (x, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let sx = x * gw / w;
            let at = (sy * gw + sx) * 4;
            if let Some(src) = glyph.rgba.get(at..at + 4) {
                px.copy_from_slice(src);
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2×2 glyph: red, green / blue, clear.
    fn glyph() -> Glyph {
        Glyph {
            width: 2,
            height: 2,
            hot_spot: (0, 0),
            rgba: vec![
                255, 0, 0, 255, 0, 255, 0, 255, //
                0, 0, 255, 255, 0, 0, 0, 0,
            ],
        }
    }

    /// At its own size the glyph is its own pixels; doubled, each pixel is a 2×2 block
    #[test]
    fn a_glyph_is_resampled_by_nearest_pixel() {
        assert_eq!(resample(&glyph(), (2, 2)), Some(glyph().rgba));
        let doubled = resample(&glyph(), (4, 4)).unwrap();
        let px = |x: usize, y: usize| &doubled[(y * 4 + x) * 4..(y * 4 + x) * 4 + 4];
        assert_eq!(px(0, 0), &[255, 0, 0, 255]);
        assert_eq!(px(1, 1), &[255, 0, 0, 255]);
        assert_eq!(px(2, 0), &[0, 255, 0, 255]);
        assert_eq!(px(0, 3), &[0, 0, 255, 255]);
        assert_eq!(px(3, 3), &[0, 0, 0, 0]);
    }

    /// A glyph short of its bytes is transparent where they run out; one with no pixels is none
    #[test]
    fn a_short_or_empty_glyph() {
        let mut short = glyph();
        short.rgba.truncate(8);
        let out = resample(&short, (2, 2)).unwrap();
        assert_eq!(&out[8..], &[0; 8]);
        let empty = Glyph::default();
        assert_eq!(resample(&empty, (2, 2)), None);
        assert_eq!(resample(&glyph(), (0, 2)), None);
    }
}
