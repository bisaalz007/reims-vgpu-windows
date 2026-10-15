//! Whether the keyboard in use is ISO, from Carbon's HIToolbox: the key map swaps the two keys
//! an ISO keyboard has where an ANSI one has one (`keymap::usb_usage`).

/// `kKeyboardISO`, `'ISO '`: [`KBGetLayoutType`]'s answer for an ISO keyboard.
const KEYBOARD_ISO: u32 = u32::from_be_bytes(*b"ISO ");

#[link(name = "Carbon", kind = "framework")]
unsafe extern "C" {
    /// The type of the keyboard last used, as HIToolbox tracks it.
    fn LMGetKbdType() -> u8;
    /// The physical layout of keyboard type `keyboard_type`: `'ANSI'`, `'ISO '` or `'JIS '`.
    fn KBGetLayoutType(keyboard_type: i16) -> u32;
}

/// Whether the keyboard last typed on is ISO, as HIToolbox tracks it (the kind the Keyboard
/// Setup Assistant sets). It follows the keyboard in use, so ask at each key event.
///
/// Call it on the main thread, where key events arrive: Carbon documents `LMGetKbdType` as not
/// thread-safe. (Called from another thread it was seen to give the same answer, but nothing
/// promises that.)
pub(crate) fn is_iso_keyboard() -> bool {
    // SAFETY: both take and return plain integers, and read HIToolbox's own state.
    let layout = unsafe { KBGetLayoutType(i16::from(LMGetKbdType())) };
    layout == KEYBOARD_ISO
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The keyboard's kind is asked of HIToolbox without dying
    #[test]
    fn the_keyboard_kind_answers() {
        let _ = is_iso_keyboard();
    }
}
