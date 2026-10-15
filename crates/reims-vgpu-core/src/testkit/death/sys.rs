//! Native mapping, termination observation and file I/O for the death harness.

#[cfg_attr(target_os = "linux", path = "sys/linux.rs")]
#[cfg_attr(target_os = "macos", path = "sys/macos.rs")]
#[cfg(any(target_os = "linux", target_os = "macos"))]
mod raw;

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod unix;
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub use unix::*;

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
pub use windows::*;

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
compile_error!("the death harness supports only macOS, Linux and Windows");
