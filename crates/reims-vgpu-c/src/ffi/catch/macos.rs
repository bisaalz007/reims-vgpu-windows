//! macOS: objc2's `exception::catch`. An Objective-C exception the host throws, `@throw nil`
//! among them, stops at this frame. A Rust panic in `f` passes through unchanged, to the panic
//! hook.

use std::panic::AssertUnwindSafe;

/// Runs `f` inside `@try`: its value, or `Err(())` for an exception it threw.
///
/// The exception is never released: the caller ends the process at the call
/// (`DeathId::HostRaised`), and a release first would run the host's object's `dealloc`, host code,
/// before the death, which the library has never done.
pub(crate) fn catch<R>(f: impl FnOnce() -> R) -> Result<R, ()> {
    objc2::exception::catch(AssertUnwindSafe(f)).map_err(std::mem::forget)
}
