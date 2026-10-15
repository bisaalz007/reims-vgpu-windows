//! [`RvgTraceRangeHandler`]: the device's handler for one trace range, which `add_trace_range`
//! lends the host.

use reims_vgpu::TraceRangeHandler;

use crate::device::RvgPhysicalMemoryRange;
use crate::ffi;
use crate::handle;

/// The device's handler for one trace range: the host calls it when the guest writes the range.
pub struct RvgTraceRangeHandler {
    handler: TraceRangeHandler,
}

impl RvgTraceRangeHandler {
    /// A handle's value over the device's handler.
    pub(crate) fn new(handler: TraceRangeHandler) -> RvgTraceRangeHandler {
        RvgTraceRangeHandler { handler }
    }
}

/// Tells the device the guest wrote `dirty`, inside a range add_trace_range watches. Any thread,
/// except inside add_trace_range. NULL does nothing.
///
/// # Safety
/// `handler` is NULL or a live handler.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_trace_range_handler_invoke(
    handler: *mut RvgTraceRangeHandler,
    dirty: RvgPhysicalMemoryRange,
) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        if let Some(h) = unsafe { handle::get(handler) } {
            (h.handler)(dirty.into());
        }
    })
}

/// One more reference to `handler`; returns it.
///
/// # Safety
/// `handler` is NULL or a live handler.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_trace_range_handler_retain(
    handler: *mut RvgTraceRangeHandler,
) -> *mut RvgTraceRangeHandler {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { handle::retain(handler) }
    })
}

/// Drops one reference to `handler`.
///
/// # Safety
/// `handler` is NULL or a live handler whose reference the caller owns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_trace_range_handler_release(handler: *mut RvgTraceRangeHandler) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { handle::release(handler) }
    })
}
