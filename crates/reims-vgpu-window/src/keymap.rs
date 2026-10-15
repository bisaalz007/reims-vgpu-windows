//! macOS virtual key codes to USB HID keyboard usages (page 0x07).
//!
//! A macOS key code names a physical key, not the character its layout gives it, so a key
//! reaches the guest by position and the guest's own layout decides what it types. The usages
//! are the ones QEMU's key tables (keycodemapdb's OS-X and USB columns) give each `kVK_*` code
//! of HIToolbox, so each maps back to its key through QEMU's `qemu_input_map_usb_to_linux`.
//! The table covers every key a Mac keyboard has, and a PC keyboard's extra keys where macOS
//! gives them a code: letters, digits, punctuation, F1 to F20, the arrows, the navigation
//! block, the keypad, both sides' modifiers, Caps Lock, the volume keys, the menu key, JIS's
//! Eisu, Kana, Yen, Ro and keypad comma, and ISO's extra key.
//!
//! Two keys are special:
//! - The key macOS calls Help (`kVK_Help`, 0x72) is a PC keyboard's Insert (usage 0x49), which
//!   macOS reports as Help. It goes to the guest as 0x49, the key in that place, where a macOS
//!   guest reads Help again.
//! - Fn (`kVK_Function`, 0x3F) has no usage on the keyboard page; the guest's USB keyboard
//!   cannot carry it, so it maps to nothing.
//!
//! ISO keyboards: macOS reports the key left of 1 as `kVK_ISO_Section` (0x0A) and the key right
//! of left Shift as `kVK_ANSI_Grave` (0x32). Its layouts say so: German gives 0x0A `^` and
//! 0x32 `<`, British 0x0A `§` and 0x32 `` ` ``, the keys those places carry on those ISO
//! keyboards. A standard ISO keyboard sends those two places the other way round: usage 0x35
//! (Grave Accent) left of 1, 0x64 (Non-US Backslash) right of left Shift. So with an ISO
//! keyboard (as macOS's window asks HIToolbox) the two codes swap their usages, and each key reaches the
//! guest as it does from a real ISO keyboard. On ANSI and JIS keyboards 0x32 is the key left of
//! 1 and maps to 0x35; 0x0A, which they lack, maps to 0x64.

/// `kVK_ISO_Section`: on an ISO keyboard, the key left of 1.
const KVK_ISO_SECTION: u16 = 0x0A;
/// `kVK_ANSI_Grave`: the key left of 1 on ANSI; right of left Shift on ISO.
const KVK_ANSI_GRAVE: u16 = 0x32;
/// Keyboard Grave Accent and Tilde: the key left of 1.
const USAGE_GRAVE: u16 = 0x35;
/// Keyboard Non-US `\` and `|`: ISO's key right of left Shift.
const USAGE_NON_US_BACKSLASH: u16 = 0x64;

/// `(macOS key code, USB usage)`, by key code.
const PAIRS: &[(u16, u16)] = &[
    (0x00, 0x04), // A
    (0x01, 0x16), // S
    (0x02, 0x07), // D
    (0x03, 0x09), // F
    (0x04, 0x0B), // H
    (0x05, 0x0A), // G
    (0x06, 0x1D), // Z
    (0x07, 0x1B), // X
    (0x08, 0x06), // C
    (0x09, 0x19), // V
    (KVK_ISO_SECTION, USAGE_NON_US_BACKSLASH),
    (0x0B, 0x05), // B
    (0x0C, 0x14), // Q
    (0x0D, 0x1A), // W
    (0x0E, 0x08), // E
    (0x0F, 0x15), // R
    (0x10, 0x1C), // Y
    (0x11, 0x17), // T
    (0x12, 0x1E), // 1
    (0x13, 0x1F), // 2
    (0x14, 0x20), // 3
    (0x15, 0x21), // 4
    (0x16, 0x23), // 6
    (0x17, 0x22), // 5
    (0x18, 0x2E), // = +
    (0x19, 0x26), // 9
    (0x1A, 0x24), // 7
    (0x1B, 0x2D), // - _
    (0x1C, 0x25), // 8
    (0x1D, 0x27), // 0
    (0x1E, 0x30), // ] }
    (0x1F, 0x12), // O
    (0x20, 0x18), // U
    (0x21, 0x2F), // [ {
    (0x22, 0x0C), // I
    (0x23, 0x13), // P
    (0x24, 0x28), // Return
    (0x25, 0x0F), // L
    (0x26, 0x0D), // J
    (0x27, 0x34), // ' "
    (0x28, 0x0E), // K
    (0x29, 0x33), // ; :
    (0x2A, 0x31), // \ |
    (0x2B, 0x36), // , <
    (0x2C, 0x38), // / ?
    (0x2D, 0x11), // N
    (0x2E, 0x10), // M
    (0x2F, 0x37), // . >
    (0x30, 0x2B), // Tab
    (0x31, 0x2C), // Space
    (KVK_ANSI_GRAVE, USAGE_GRAVE),
    (0x33, 0x2A), // Delete (Backspace)
    (0x35, 0x29), // Escape
    (0x36, 0xE7), // right Command
    (0x37, 0xE3), // left Command
    (0x38, 0xE1), // left Shift
    (0x39, 0x39), // Caps Lock
    (0x3A, 0xE2), // left Option
    (0x3B, 0xE0), // left Control
    (0x3C, 0xE5), // right Shift
    (0x3D, 0xE6), // right Option
    (0x3E, 0xE4), // right Control
    (0x40, 0x6C), // F17
    (0x41, 0x63), // keypad .
    (0x43, 0x55), // keypad *
    (0x45, 0x57), // keypad +
    (0x47, 0x53), // keypad Clear (Num Lock)
    (0x48, 0x80), // Volume Up
    (0x49, 0x81), // Volume Down
    (0x4A, 0x7F), // Mute
    (0x4B, 0x54), // keypad /
    (0x4C, 0x58), // keypad Enter
    (0x4E, 0x56), // keypad -
    (0x4F, 0x6D), // F18
    (0x50, 0x6E), // F19
    (0x51, 0x67), // keypad =
    (0x52, 0x62), // keypad 0
    (0x53, 0x59), // keypad 1
    (0x54, 0x5A), // keypad 2
    (0x55, 0x5B), // keypad 3
    (0x56, 0x5C), // keypad 4
    (0x57, 0x5D), // keypad 5
    (0x58, 0x5E), // keypad 6
    (0x59, 0x5F), // keypad 7
    (0x5A, 0x6F), // F20
    (0x5B, 0x60), // keypad 8
    (0x5C, 0x61), // keypad 9
    (0x5D, 0x89), // JIS Yen
    (0x5E, 0x87), // JIS Ro (underscore)
    (0x5F, 0x8C), // JIS keypad comma
    (0x60, 0x3E), // F5
    (0x61, 0x3F), // F6
    (0x62, 0x40), // F7
    (0x63, 0x3C), // F3
    (0x64, 0x41), // F8
    (0x65, 0x42), // F9
    (0x66, 0x91), // JIS Eisu (Lang2)
    (0x67, 0x44), // F11
    (0x68, 0x90), // JIS Kana (Lang1)
    (0x69, 0x68), // F13
    (0x6A, 0x6B), // F16
    (0x6B, 0x69), // F14
    (0x6D, 0x43), // F10
    (0x6E, 0x65), // menu (Application)
    (0x6F, 0x45), // F12
    (0x71, 0x6A), // F15
    (0x72, 0x49), // Help: Insert's key
    (0x73, 0x4A), // Home
    (0x74, 0x4B), // Page Up
    (0x75, 0x4C), // Forward Delete
    (0x76, 0x3D), // F4
    (0x77, 0x4D), // End
    (0x78, 0x3B), // F2
    (0x79, 0x4E), // Page Down
    (0x7A, 0x3A), // F1
    (0x7B, 0x50), // Left
    (0x7C, 0x4F), // Right
    (0x7D, 0x51), // Down
    (0x7E, 0x52), // Up
];

/// Key codes run below this.
const CODES: usize = 0x80;

/// The usage of each key code, 0 for none (usage 0 is "no event").
const TABLE: [u16; CODES] = {
    let mut table = [0u16; CODES];
    let mut i = 0;
    while i < PAIRS.len() {
        let (code, usage) = PAIRS[i];
        table[code as usize] = usage;
        i += 1;
    }
    table
};

/// The USB HID keyboard usage (page 0x07) of the key macOS reports as `mac_keycode`
/// (`NSEvent.keyCode`), `iso` when the keyboard is ISO (the module doc says why it matters).
/// `None` for Fn and for a code no key has.
pub fn usb_usage(mac_keycode: u16, iso: bool) -> Option<u16> {
    let code = match mac_keycode {
        KVK_ISO_SECTION if iso => KVK_ANSI_GRAVE,
        KVK_ANSI_GRAVE if iso => KVK_ISO_SECTION,
        code => code,
    };
    TABLE
        .get(usize::from(code))
        .copied()
        .filter(|&usage| usage != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sample of keys, from every group, reaches the guest by position
    #[test]
    fn keys_map_by_position() {
        let cases: &[(u16, u16)] = &[
            (0x00, 0x04), // A
            (0x06, 0x1D), // Z: where Y is on a German keyboard
            (0x10, 0x1C), // Y
            (0x12, 0x1E), // 1
            (0x1D, 0x27), // 0
            (0x1B, 0x2D), // -
            (0x2A, 0x31), // backslash
            (0x24, 0x28), // Return
            (0x33, 0x2A), // Delete
            (0x75, 0x4C), // Forward Delete
            (0x35, 0x29), // Escape
            (0x31, 0x2C), // Space
            (0x7A, 0x3A), // F1
            (0x6F, 0x45), // F12
            (0x69, 0x68), // F13
            (0x5A, 0x6F), // F20
            (0x7B, 0x50), // Left
            (0x7E, 0x52), // Up
            (0x73, 0x4A), // Home
            (0x79, 0x4E), // Page Down
            (0x72, 0x49), // Help: Insert
            (0x52, 0x62), // keypad 0
            (0x4C, 0x58), // keypad Enter
            (0x47, 0x53), // keypad Clear
            (0x51, 0x67), // keypad =
            (0x3B, 0xE0), // left Control
            (0x38, 0xE1), // left Shift
            (0x3A, 0xE2), // left Option
            (0x37, 0xE3), // left Command
            (0x3E, 0xE4), // right Control
            (0x3C, 0xE5), // right Shift
            (0x3D, 0xE6), // right Option
            (0x36, 0xE7), // right Command
            (0x39, 0x39), // Caps Lock
            (0x66, 0x91), // Eisu
            (0x68, 0x90), // Kana
            (0x5D, 0x89), // Yen
            (0x5E, 0x87), // Ro
            (0x5F, 0x8C), // keypad comma
            (0x6E, 0x65), // menu
        ];
        for &(code, usage) in cases {
            assert_eq!(usb_usage(code, false), Some(usage), "key code {code:#04x}");
            if code != KVK_ISO_SECTION && code != KVK_ANSI_GRAVE {
                assert_eq!(usb_usage(code, true), Some(usage), "key code {code:#04x}");
            }
        }
        // Every key has its own usage.
        let mut usages: Vec<u16> = PAIRS.iter().map(|&(_, u)| u).collect();
        usages.sort_unstable();
        usages.dedup();
        assert_eq!(usages.len(), PAIRS.len());
        // The letters are the 26 usages from A.
        let letters = [
            0x00, 0x0B, 0x08, 0x02, 0x0E, 0x03, 0x05, 0x04, 0x22, 0x26, 0x28, 0x25, 0x2E, 0x2D,
            0x1F, 0x23, 0x0C, 0x0F, 0x01, 0x11, 0x20, 0x09, 0x0D, 0x07, 0x10, 0x06,
        ];
        for (i, code) in (0u16..).zip(letters) {
            assert_eq!(usb_usage(code, false), Some(0x04 + i));
        }
    }

    /// On ISO, the key left of 1 is Grave and the key right of left Shift Non-US Backslash;
    /// on ANSI the codes keep their own usages
    #[test]
    fn iso_swaps_section_and_grave() {
        // ISO: left of 1 (reported as Section), right of left Shift (reported as Grave).
        assert_eq!(usb_usage(KVK_ISO_SECTION, true), Some(USAGE_GRAVE));
        assert_eq!(
            usb_usage(KVK_ANSI_GRAVE, true),
            Some(USAGE_NON_US_BACKSLASH)
        );
        // ANSI and JIS: Grave is the key left of 1.
        assert_eq!(usb_usage(KVK_ANSI_GRAVE, false), Some(USAGE_GRAVE));
        assert_eq!(
            usb_usage(KVK_ISO_SECTION, false),
            Some(USAGE_NON_US_BACKSLASH)
        );
    }

    /// Fn and codes no key has map to nothing
    #[test]
    fn fn_and_unknown_codes_have_no_usage() {
        assert_eq!(usb_usage(0x3F, false), None);
        assert_eq!(usb_usage(0x3F, true), None);
        for code in [
            0x34,
            0x42,
            0x44,
            0x46,
            0x4D,
            0x6C,
            0x70,
            0x7F,
            0x80,
            0xFF,
            u16::MAX,
        ] {
            assert_eq!(usb_usage(code, false), None, "key code {code:#x}");
        }
    }
}
