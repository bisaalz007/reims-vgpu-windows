//! SDL on macOS: started on the main thread as the application launches, inside it, and its events
//! taken as they come.
//!
//! **The start** ([`start`]) runs on the main thread as the application launches (`app`), before
//! the windows posted so far open: Cocoa's video needs the main thread. SDL starts into the
//! application already there, the library's or the host's, and makes none of its own, nor a menu
//! or a delegate. Its settings, before its video starts: no signal handlers of its own (the
//! host's stay the host's), the screen saver allowed, cursors given for the screen's scale
//! (`crate::sdl::cursor`), no quit event when the last window closes (the windows' end is the
//! display's), the click that makes a window key AppKit's alone, and no handlers of SDL's on the
//! application's activation (AppKit's own key window rules apply). Where SDL's video does not
//! start, the reason is logged once and no SDL window opens.
//!
//! **Events.** SDL never runs a loop of its own, nor pumps AppKit's events: the application
//! dispatches them to SDL's windows and views, which queue what they mean for SDL. An event watch
//! ([`watch`]) posts a drain to the main run loop with the first event queued; the drain takes
//! every queued event, in order, to the shell (`crate::sdl::shell::event`). The run loop runs the
//! drain in its common modes, so a window resized live lays out as it resizes. Display events
//! are not taken: the application's screens notification lays the windows out (`app`).

use std::ffi::{CStr, c_void};
use std::sync::atomic::{AtomicBool, Ordering};

use objc2::MainThreadMarker;
use sdl3_sys::everything::*;

use crate::native::macos::main_thread::post_main;
use crate::sdl::{sdl_error, shell};

/// SDL's start has run: it runs once.
static STARTED: AtomicBool = AtomicBool::new(false);

/// A drain is posted and has not run yet.
static DRAIN_POSTED: AtomicBool = AtomicBool::new(false);

/// SDL's settings (the module doc).
const HINTS: [(&CStr, &CStr); 6] = [
    (c"SDL_NO_SIGNAL_HANDLERS", c"1"),
    (c"SDL_VIDEO_ALLOW_SCREENSAVER", c"1"),
    (c"SDL_MOUSE_DPI_SCALE_CURSORS", c"1"),
    (c"SDL_QUIT_ON_LAST_WINDOW_CLOSE", c"0"),
    (c"SDL_MOUSE_FOCUS_CLICKTHROUGH", c"0"),
    (c"SDL_MAC_REGISTER_ACTIVATION_HANDLERS", c"0"),
];

/// Starts SDL's video on the main thread, once (the module doc).
pub(crate) fn start(_mtm: MainThreadMarker) {
    if STARTED.swap(true, Ordering::AcqRel) {
        return;
    }
    for (name, value) in HINTS {
        // SAFETY: both are NUL-terminated strings SDL copies.
        unsafe { SDL_SetHint(name.as_ptr(), value.as_ptr()) };
    }
    // SAFETY: on the main thread, inside the application, which Cocoa's video takes.
    if !unsafe { SDL_Init(SDL_INIT_VIDEO) } {
        log::warn!(
            target: "reims_vgpu_window",
            "SDL's video does not start ({}): no window opens",
            sdl_error()
        );
        return;
    }
    // SAFETY: SDL's video runs; the watch takes no user data and lives for the process.
    if !unsafe { SDL_AddEventWatch(Some(watch), std::ptr::null_mut()) } {
        log::warn!(
            target: "reims_vgpu_window",
            "no watch of SDL's events ({}): the windows follow none of their changes",
            sdl_error()
        );
    }
    log::info!(
        target: "reims_vgpu_window",
        "windows open through SDL's cocoa driver"
    );
}

/// SDL queued an event: a drain is posted unless one is.
unsafe extern "C" fn watch(_userdata: *mut c_void, _event: *mut SDL_Event) -> bool {
    if !DRAIN_POSTED.swap(true, Ordering::AcqRel) {
        post_main(|_| drain());
    }
    true
}

/// Takes every event SDL queued, in order, to the shell.
fn drain() {
    DRAIN_POSTED.store(false, Ordering::Release);
    loop {
        // SAFETY: an all-zero event is a valid one for SDL to fill.
        let mut event: SDL_Event = unsafe { std::mem::zeroed() };
        // SAFETY: SDL's video runs on this thread; one event is taken into `event`.
        let taken = unsafe {
            SDL_PeepEvents(
                &mut event,
                1,
                SDL_GETEVENT,
                SDL_EVENT_FIRST.0,
                SDL_EVENT_LAST.0,
            )
        };
        if taken <= 0 {
            shell::batch_over();
            return;
        }
        shell::event(&event);
    }
}
