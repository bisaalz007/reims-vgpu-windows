//! Validation and ownership of the host's memory backing, selected by OS.

#[cfg_attr(unix, path = "memory/unix.rs")]
#[cfg_attr(windows, path = "memory/windows.rs")]
mod native;

pub(super) use native::files;
#[cfg(all(unix, test))]
pub(super) use native::host_page_size;
