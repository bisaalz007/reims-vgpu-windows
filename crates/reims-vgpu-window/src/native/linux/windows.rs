//! Windows language-key messages which SDL's ordinary key path may consume as IME keys.
//! Measured on an Apple JIS keyboard: Eisu is VK_FINAL/scan F1 and Kana is
//! VK_IME_ON/scan F2. Their Raw Input packets carry break/E1 flags even on press.

use std::ffi::c_void;

use sdl3_sys::everything::*;

/// The documented prefix of Win32 MSG; the callback does not read its trailing fields.
#[repr(C)]
struct MessagePrefix {
    window: *mut c_void,
    message: u32,
    wparam: usize,
    lparam: isize,
}

fn language_usage(vkey: usize, scan: u8) -> Option<u16> {
    match (vkey, scan) {
        (0x19 | 0x1a, 0x71 | 0xf1) => Some(0x91),
        (0x15 | 0x16, 0x72 | 0xf2) => Some(0x90),
        _ => None,
    }
}

unsafe extern "C" fn message_hook(_: *mut c_void, message: *mut MSG) -> bool {
    // SAFETY: SDL passes a mutable MSG on its UI thread; its documented prefix fits here.
    let Some(message) = (unsafe { message.cast::<MessagePrefix>().as_ref() }) else {
        return true;
    };
    if !matches!(message.message, 0x100 | 0x101 | 0x104 | 0x105) {
        return true;
    }
    let scan = ((message.lparam as usize >> 16) & 0xff) as u8;
    let Some(usage) = language_usage(message.wparam, scan) else {
        return true;
    };
    // SAFETY: SDL video is initialized and this callback runs on its UI thread.
    let focus = unsafe { SDL_GetKeyboardFocus() };
    if focus.is_null() {
        return true;
    }
    // SAFETY: focus is a live SDL window. Only that window's key messages are consumed.
    let window = unsafe {
        SDL_GetPointerProperty(
            SDL_GetWindowProperties(focus),
            SDL_PROP_WINDOW_WIN32_HWND_POINTER,
            std::ptr::null_mut(),
        )
    };
    if window != message.window {
        return true;
    }
    let down = matches!(message.message, 0x100 | 0x104);
    // SAFETY: initialize the keyboard variant and its tag together. SDL copies it.
    unsafe {
        let mut event = SDL_Event {
            key: SDL_KeyboardEvent {
                r#type: if down {
                    SDL_EVENT_KEY_DOWN
                } else {
                    SDL_EVENT_KEY_UP
                },
                timestamp: SDL_GetTicksNS(),
                windowID: SDL_GetWindowID(focus),
                scancode: SDL_Scancode::new(i32::from(usage)),
                down,
                repeat: down && (message.lparam as usize & (1 << 30)) != 0,
                raw: u16::from(scan),
                ..Default::default()
            },
        };
        SDL_PushEvent(&mut event);
    }
    false
}

pub(super) fn install_message_hook() {
    // SAFETY: called after SDL_Init on the UI thread; the callback has static lifetime.
    unsafe { SDL_SetWindowsMessageHook(Some(message_hook), std::ptr::null_mut()) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_messages_preserve_jis_and_leave_other_keys_alone() {
        assert_eq!(language_usage(0x1a, 0xf1), Some(0x91));
        assert_eq!(language_usage(0x16, 0xf2), Some(0x90));
        assert_eq!(language_usage(0x19, 0x71), Some(0x91));
        assert_eq!(language_usage(0x15, 0x72), Some(0x90));
        assert_eq!(language_usage(0x41, 0x1e), None);
        assert_eq!(language_usage(0x41, 0xf1), None);
        assert_eq!(language_usage(0x1a, 0x1e), None);
        assert_eq!(language_usage(0x5b, 0x5b), None);
    }
}
