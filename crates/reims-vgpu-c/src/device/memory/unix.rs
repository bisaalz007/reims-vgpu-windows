//! Unix memory files: validate every range before duplicating any descriptor.

use super::super::{RvgMemoryRange, RvgMemoryRangeFile, RvgMemoryRangeSection};
use crate::status::{RVG_ERR_INVALID_ARGUMENT, RvgStatus};
use reims_vgpu::MemoryFile;
use std::mem::MaybeUninit;
use std::num::NonZeroU64;
use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::Arc;

pub(crate) unsafe fn files(
    ranges: &[RvgMemoryRange],
    files: *const RvgMemoryRangeFile,
    sections: *const RvgMemoryRangeSection,
) -> Result<Vec<Option<MemoryFile>>, RvgStatus> {
    if !sections.is_null() {
        log::warn!("device not created: memory sections are supported only on Windows");
        return Err(RVG_ERR_INVALID_ARGUMENT);
    }
    let files = if files.is_null() || ranges.is_empty() {
        &[]
    } else {
        // SAFETY: the caller supplied one entry per range.
        unsafe { std::slice::from_raw_parts(files, ranges.len()) }
    };
    let page = host_page_size();
    for (i, (r, f)) in ranges.iter().zip(files).enumerate() {
        if let Some(why) = file_refusal(r, f, page) {
            log::warn!("device not created: the file of memory range {i} {why}");
            return Err(RVG_ERR_INVALID_ARGUMENT);
        }
    }
    (0..ranges.len()).map(|i| match files.get(i) {
        Some(f) if f.fd >= 0 => match duplicate(f.fd) {
            Ok(fd) => Ok(Some(MemoryFile { fd: Arc::new(fd), offset: f.offset })),
            Err(e) => {
                log::warn!("device not created: the file of memory range {i}, fd {}, was not duplicated: {e}", f.fd);
                Err(RVG_ERR_INVALID_ARGUMENT)
            }
        },
        _ => Ok(None),
    }).collect()
}

/// Why the file entry `f` of range `r` is refused, as the end of a log line, or `None`: an `fd`
/// below -1, `flags` other than 0, and for an `fd` of 0 or more, a file `fstat` cannot read, one
/// holding fewer than `offset + physical_length` bytes, or an `offset` that is not a multiple of
/// `page`, in that order.
fn file_refusal(r: &RvgMemoryRange, f: &RvgMemoryRangeFile, page: NonZeroU64) -> Option<String> {
    if f.fd < -1 {
        return Some(format!("has fd {}, below -1", f.fd));
    }
    if f.flags != 0 {
        return Some(format!("has flags {:#x}, not 0", f.flags));
    }
    if f.fd == -1 {
        return None;
    }
    let size = match file_size(f.fd) {
        Ok(size) => size,
        Err(e) => return Some(format!("has fd {}, which fstat cannot read: {e}", f.fd)),
    };
    if f.offset
        .checked_add(r.physical_length)
        .is_none_or(|end| size < end)
    {
        return Some(format!(
            "holds {size:#x} bytes, fewer than its offset {:#x} and the range's {:#x}",
            f.offset, r.physical_length
        ));
    }
    if f.offset % page != 0 {
        return Some(format!(
            "has offset {:#x}, not a multiple of the page size {page:#x}",
            f.offset
        ));
    }
    None
}

/// The host's page size (`sysconf(_SC_PAGESIZE)`, which does not fail): 16 KiB on arm64 macOS,
/// 4 KiB on x86-64.
pub(crate) fn host_page_size() -> NonZeroU64 {
    // SAFETY: a query of a constant; it reads and writes no memory of the caller's.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(page)
        .ok()
        .and_then(NonZeroU64::new)
        .unwrap_or(NonZeroU64::MIN)
}

/// The size of the file `fd` names (`fstat`).
fn file_size(fd: i32) -> std::io::Result<u64> {
    let mut st = MaybeUninit::<libc::stat>::zeroed();
    // SAFETY: `st` is a writable `struct stat`; an `fd` that names no open file is an error the
    // call returns.
    if unsafe { libc::fstat(fd, st.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `fstat` succeeded, so it filled `st`; it was zeroed before.
    let st = unsafe { st.assume_init() };
    Ok(u64::try_from(st.st_size).unwrap_or(0))
}

/// A duplicate of `fd`, close-on-exec (`fcntl(F_DUPFD_CLOEXEC)`), which the map holds.
fn duplicate(fd: i32) -> std::io::Result<OwnedFd> {
    // SAFETY: `fcntl` reads and writes no memory of the caller's; an `fd` that names no open
    // file is an error the call returns.
    let new = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
    if new < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `new` is a descriptor this call opened, owned by nothing else.
    Ok(unsafe { OwnedFd::from_raw_fd(new) })
}
