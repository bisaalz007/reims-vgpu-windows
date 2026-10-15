//! Windows section handles: validate real writable views before duplicating any handle.

use std::ffi::c_void;
use std::io;
use std::os::windows::io::{FromRawHandle as _, OwnedHandle};
use std::sync::Arc;

use super::super::{RvgMemoryRange, RvgMemoryRangeFile, RvgMemoryRangeSection};
use crate::status::{RVG_ERR_INVALID_ARGUMENT, RvgStatus};
use reims_vgpu::MemoryFile;

#[repr(C)]
struct SystemInfo {
    architecture: u32,
    page_size: u32,
    minimum_address: *mut c_void,
    maximum_address: *mut c_void,
    processor_mask: usize,
    processor_count: u32,
    processor_type: u32,
    allocation_granularity: u32,
    processor_level: u16,
    processor_revision: u16,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetSystemInfo(info: *mut SystemInfo);
    fn GetCurrentProcess() -> *mut c_void;
    fn DuplicateHandle(
        source_process: *mut c_void,
        source: *mut c_void,
        target_process: *mut c_void,
        target: *mut *mut c_void,
        access: u32,
        inherit: i32,
        options: u32,
    ) -> i32;
    fn MapViewOfFile(
        section: *mut c_void,
        access: u32,
        high: u32,
        low: u32,
        len: usize,
    ) -> *mut c_void;
    fn UnmapViewOfFile(address: *const c_void) -> i32;
}

fn system_info() -> SystemInfo {
    // SAFETY: integers and raw pointers accept zero; GetSystemInfo fills this writable record.
    let mut info: SystemInfo = unsafe { std::mem::zeroed() };
    // SAFETY: a writable SYSTEM_INFO with its documented layout.
    unsafe { GetSystemInfo(&raw mut info) };
    info
}

fn handle(raw: u64) -> Option<*mut c_void> {
    usize::try_from(raw)
        .ok()
        .map(std::ptr::with_exposed_provenance_mut)
}

fn section_refusal(
    range: &RvgMemoryRange,
    section: &RvgMemoryRangeSection,
    info: &SystemInfo,
) -> Option<String> {
    if section.flags != 0 || section.reserved != 0 {
        return Some("has nonzero flags or reserved".into());
    }
    if section.handle == 0 {
        return None;
    }
    if section
        .offset
        .checked_add(range.physical_length)
        .is_none_or(|end| end > section.length)
    {
        return Some("does not contain offset + physical_length".into());
    }
    if info.page_size == 0 || !section.offset.is_multiple_of(u64::from(info.page_size)) {
        return Some("has an offset not aligned to the host page size".into());
    }
    if section.length == 0 || info.allocation_granularity == 0 {
        return Some("has no section bytes".into());
    }
    // Probe the end of the declared section: a fabricated length cannot hide an undersized handle.
    let last = section.length.saturating_sub(1);
    let granularity = u64::from(info.allocation_granularity);
    let off = last - last % granularity;
    let Ok(len) = usize::try_from(section.length - off) else {
        return Some("has a section length that does not fit the host address space".into());
    };
    let Some(raw) = handle(section.handle) else {
        return Some("has a handle that does not fit the host pointer".into());
    };
    // SAFETY: the OS validates this raw handle and requested range; it replaces no caller memory.
    let view = unsafe {
        MapViewOfFile(
            raw,
            2,
            u32::try_from(off >> 32).unwrap_or(0),
            u32::try_from(off & 0xffff_ffff).unwrap_or(0),
            len,
        )
    };
    if view.is_null() {
        return Some(format!(
            "cannot map its declared length with write access: {}",
            io::Error::last_os_error()
        ));
    }
    // SAFETY: release the temporary view just created, without taking the host's handle.
    if unsafe { UnmapViewOfFile(view) } == 0 {
        return Some(format!(
            "cannot unmap its validation view: {}",
            io::Error::last_os_error()
        ));
    }
    None
}

fn duplicate(raw: u64) -> io::Result<OwnedHandle> {
    let source = handle(raw).ok_or(io::ErrorKind::InvalidInput)?;
    // SAFETY: GetCurrentProcess returns this process's pseudo-handle.
    let process = unsafe { GetCurrentProcess() };
    let mut new = std::ptr::null_mut();
    // SAFETY: the OS validates the source and fills a new, noninheritable handle with the same access.
    if unsafe { DuplicateHandle(process, source, process, &raw mut new, 0, 0, 2) } == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the duplicate has one owner, returned to the memory map.
    Ok(unsafe { OwnedHandle::from_raw_handle(new) })
}

pub(crate) unsafe fn files(
    ranges: &[RvgMemoryRange],
    files: *const RvgMemoryRangeFile,
    sections: *const RvgMemoryRangeSection,
) -> Result<Vec<Option<MemoryFile>>, RvgStatus> {
    if !files.is_null() {
        log::warn!(
            "device not created: memory files are unsupported on Windows; use memory sections"
        );
        return Err(RVG_ERR_INVALID_ARGUMENT);
    }
    let sections = if sections.is_null() || ranges.is_empty() {
        &[]
    } else {
        // SAFETY: the caller supplied one descriptor per validated range.
        unsafe { std::slice::from_raw_parts(sections, ranges.len()) }
    };
    let info = system_info();
    for (i, (r, s)) in ranges.iter().zip(sections).enumerate() {
        if let Some(why) = section_refusal(r, s, &info) {
            log::warn!("device not created: the section of memory range {i} {why}");
            return Err(RVG_ERR_INVALID_ARGUMENT);
        }
    }
    (0..ranges.len()).map(|i| match sections.get(i) {
        Some(s) if s.handle != 0 => match duplicate(s.handle) {
            Ok(handle) => Ok(Some(MemoryFile { handle: Arc::new(handle), offset: s.offset })),
            Err(e) => {
                log::warn!("device not created: the section of memory range {i} was not duplicated: {e}");
                Err(RVG_ERR_INVALID_ARGUMENT)
            }
        },
        _ => Ok(None),
    }).collect()
}
