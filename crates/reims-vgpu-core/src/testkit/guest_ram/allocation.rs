//! Storage for the fake's raw regions, which the death harness can remap in place.

#[cfg(unix)]
pub(super) use std::alloc::{alloc_zeroed, dealloc};

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub(super) use windows::{alloc_zeroed, dealloc};

#[cfg(not(any(unix, windows)))]
compile_error!("testkit raw memory supports only Unix and Windows");
