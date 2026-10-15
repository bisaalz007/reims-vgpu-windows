//! A display mode: its size and refresh rate, and the EDR and 10 bpc flags (off: no preference
//! is read).

use crate::display::DisplayCoord;

/// A display mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplayMode {
    /// The size in pixels.
    pub size_in_pixels: DisplayCoord,
    /// The refresh rate, what the SDK's `refreshRate` answers.
    pub refresh_rate_in_hz: f64,
    /// `supportsEDR`: off at construction.
    pub supports_edr: bool,
    /// `supports10bpc`: off at construction.
    pub supports_10bpc: bool,
}

impl DisplayMode {
    /// The size and rate stored unchecked, both flags off.
    pub fn new(size_in_pixels: DisplayCoord, refresh_rate_in_hz: f64) -> DisplayMode {
        DisplayMode {
            size_in_pixels,
            refresh_rate_in_hz,
            supports_edr: false,
            supports_10bpc: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: DisplayCoord = DisplayCoord { x: 1920, y: 1080 };

    /// Both flags off; size and rate stored as given.
    #[test]
    fn flags_off() {
        let m = DisplayMode::new(SIZE, 59.94);
        assert_eq!(m.size_in_pixels, SIZE);
        assert_eq!(m.refresh_rate_in_hz, 59.94);
        assert!(!m.supports_edr && !m.supports_10bpc);
    }

    /// Any double and any size are stored unchecked: zero, NaN, negative, the u16 extremes.
    #[test]
    fn values_unchecked() {
        let big = DisplayCoord { x: u16::MAX, y: 0 };
        let m = DisplayMode::new(big, -1.0);
        assert_eq!(m.size_in_pixels, big);
        assert_eq!(m.refresh_rate_in_hz, -1.0);
        let m = DisplayMode::new(big, f64::NAN);
        assert!(m.refresh_rate_in_hz.is_nan());
    }

    #[test]
    fn accessors_are_fields() {
        let mut m = DisplayMode::new(SIZE, 30.0);
        m.supports_edr = true;
        m.supports_10bpc = true;
        assert_eq!(
            m,
            DisplayMode {
                size_in_pixels: SIZE,
                refresh_rate_in_hz: 30.0,
                supports_edr: true,
                supports_10bpc: true
            }
        );
    }
}
