//! The log hook: the library's `log` lines sent to the host's function
//! (`reims_vgpu_core::logging`).

use std::ffi::{c_char, c_void};

use log::{Level, LevelFilter};
use reims_vgpu_core::logging;

use crate::ffi;

/// A log line's level: one of RVG_LOG_*.
pub type RvgLogLevel = u32;

// The constants below are RvgLogLevel values, typed `u32` so that cbindgen writes them with the
// header's other constants, ahead of its types.

/// No lines.
pub const RVG_LOG_OFF: u32 = 0;

#[expect(missing_docs, reason = "the name is the level")]
pub const RVG_LOG_ERROR: u32 = 1;

#[expect(missing_docs, reason = "the name is the level")]
pub const RVG_LOG_WARN: u32 = 2;

#[expect(missing_docs, reason = "the name is the level")]
pub const RVG_LOG_INFO: u32 = 3;

#[expect(missing_docs, reason = "the name is the level")]
pub const RVG_LOG_DEBUG: u32 = 4;

#[expect(missing_docs, reason = "the name is the level")]
pub const RVG_LOG_TRACE: u32 = 5;

/// One log line. `target` names the library module that wrote it; neither string is
/// NUL-terminated, and both are borrowed for the call. Runs on any library or host thread,
/// possibly on several at once, possibly under the library's locks: it must not call this
/// header's functions, and should return quickly.
pub type RvgLogFn = Option<
    unsafe extern "C-unwind" fn(
        ctx: *mut c_void,
        level: RvgLogLevel,
        target: *const c_char,
        target_length: usize,
        message: *const c_char,
        message_length: usize,
    ),
>;

/// The level filter of an `RvgLogLevel`; a value above RVG_LOG_TRACE is RVG_LOG_TRACE.
fn filter(level: RvgLogLevel) -> LevelFilter {
    match level {
        RVG_LOG_OFF => LevelFilter::Off,
        RVG_LOG_ERROR => LevelFilter::Error,
        RVG_LOG_WARN => LevelFilter::Warn,
        RVG_LOG_INFO => LevelFilter::Info,
        RVG_LOG_DEBUG => LevelFilter::Debug,
        _ => LevelFilter::Trace,
    }
}

/// The `RvgLogLevel` of a line's level.
fn level(level: Level) -> RvgLogLevel {
    match level {
        Level::Error => RVG_LOG_ERROR,
        Level::Warn => RVG_LOG_WARN,
        Level::Info => RVG_LOG_INFO,
        Level::Debug => RVG_LOG_DEBUG,
        Level::Trace => RVG_LOG_TRACE,
    }
}

/// The host's `ctx`, which it keeps valid for the rest of the process.
#[derive(Clone, Copy)]
struct Ctx(*mut c_void);

// SAFETY: rvg_log_set_handler's contract: `ctx` is usable from any thread, for the rest of the
// process.
unsafe impl Send for Ctx {}
// SAFETY: as above.
unsafe impl Sync for Ctx {}

/// Sends the library's log lines at `max_level` and below to `handler`, for the rest of the
/// process or until the next call; a `max_level` above RVG_LOG_TRACE is RVG_LOG_TRACE. NULL
/// sends them where REIMS_VGPU_LOG says (by default, nowhere). When REIMS_VGPU_LOG is set, its
/// filter picks the lines in place of `max_level`. In a process whose Rust code installed a
/// `log` crate logger first, the lines go there and no handler is kept. `ctx` must stay valid for
/// the rest of the process: a line under way on another thread may still reach the previous
/// handler. Whether logging is on never changes what the library does.
///
/// # Safety
/// `handler` is NULL or a function the host keeps callable, with `ctx`, from any thread for
/// the rest of the process.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_log_set_handler(
    handler: RvgLogFn,
    ctx: *mut c_void,
    max_level: RvgLogLevel,
) {
    ffi::export(|| {
        let Some(handler) = handler else {
            logging::clear_hook();
            return;
        };
        let ctx = Ctx(ctx);
        // A process with a `log` logger of its own keeps it: the lines go there, and the hook is
        // not kept.
        let _ = logging::set_hook(
            filter(max_level),
            Box::new(move |l: Level, target: &str, message: &str| {
                let ctx = ctx;
                // SAFETY: the host's function over its ctx (the contract above); both strings are
                // live for the call.
                ffi::host_call("log", || unsafe {
                    handler(
                        ctx.0,
                        level(l),
                        target.as_ptr().cast(),
                        target.len(),
                        message.as_ptr().cast(),
                        message.len(),
                    );
                });
            }),
        );
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every level maps both ways, and a level above trace is trace
    #[test]
    fn levels() {
        for (l, f) in [
            (RVG_LOG_OFF, LevelFilter::Off),
            (RVG_LOG_ERROR, LevelFilter::Error),
            (RVG_LOG_WARN, LevelFilter::Warn),
            (RVG_LOG_INFO, LevelFilter::Info),
            (RVG_LOG_DEBUG, LevelFilter::Debug),
            (RVG_LOG_TRACE, LevelFilter::Trace),
            (6, LevelFilter::Trace),
        ] {
            assert_eq!(filter(l), f);
            if let Some(level_of) = f.to_level() {
                assert_eq!(level(level_of), l.min(RVG_LOG_TRACE));
            }
        }
    }
}
