//! The Win32 memory and clock calls used by this platform.

use std::ffi::c_void;

pub(super) const PAGE_READWRITE: u32 = 4;
pub(super) const PAGE_READONLY: u32 = 2;
pub(super) const PAGE_NOACCESS: u32 = 1;
pub(super) const MEM_COMMIT: u32 = 0x1000;
pub(super) const MEM_RESERVE: u32 = 0x2000;
pub(super) const MEM_REPLACE_PLACEHOLDER: u32 = 0x4000;
pub(super) const MEM_RESERVE_PLACEHOLDER: u32 = 0x40000;
pub(super) const MEM_RELEASE: u32 = 0x8000;
pub(super) const MEM_PRESERVE_PLACEHOLDER: u32 = 2;
pub(super) const MEM_COALESCE_PLACEHOLDERS: u32 = 1;

#[repr(C)]
#[derive(Default)]
pub(super) struct SystemInfo {
    pub architecture: u32,
    pub page_size: u32,
    pub minimum: *mut c_void,
    pub maximum: *mut c_void,
    pub active_mask: usize,
    pub processor_count: u32,
    pub processor_type: u32,
    pub allocation_granularity: u32,
    pub processor_level: u16,
    pub processor_revision: u16,
}

#[repr(C)]
#[derive(Default)]
pub(super) struct MemoryInfo {
    pub base: *mut c_void,
    pub allocation_base: *mut c_void,
    pub allocation_protection: u32,
    pub partition_id: u16,
    pub region_size: usize,
    pub state: u32,
    pub protection: u32,
    pub kind: u32,
}

#[repr(C)]
#[derive(Default)]
pub(super) struct MemoryStatus {
    pub length: u32,
    pub load: u32,
    pub total_physical: u64,
    pub available_physical: u64,
    pub total_pagefile: u64,
    pub available_pagefile: u64,
    pub total_virtual: u64,
    pub available_virtual: u64,
    pub available_extended: u64,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    pub(super) fn GetSystemInfo(info: *mut SystemInfo);
    pub(super) fn GetCurrentProcess() -> *mut c_void;
    pub(super) fn GlobalMemoryStatusEx(info: *mut MemoryStatus) -> i32;
    pub(super) fn VirtualFree(address: *mut c_void, size: usize, flags: u32) -> i32;
    pub(super) fn VirtualProtect(
        address: *mut c_void,
        size: usize,
        protection: u32,
        old: *mut u32,
    ) -> i32;
    pub(super) fn VirtualQuery(address: *const c_void, info: *mut MemoryInfo, size: usize)
    -> usize;
    pub(super) fn CreateFileMappingW(
        file: *mut c_void,
        attributes: *const c_void,
        protection: u32,
        high: u32,
        low: u32,
        name: *const u16,
    ) -> *mut c_void;
}

#[link(name = "kernelbase")]
unsafe extern "system" {
    pub(super) fn QueryInterruptTimePrecise(time: *mut u64);
    pub(super) fn VirtualAlloc2(
        process: *mut c_void,
        address: *mut c_void,
        size: usize,
        flags: u32,
        protection: u32,
        parameters: *const c_void,
        count: u32,
    ) -> *mut c_void;
    pub(super) fn MapViewOfFile3(
        section: *mut c_void,
        process: *mut c_void,
        address: *mut c_void,
        offset: u64,
        size: usize,
        flags: u32,
        protection: u32,
        parameters: *const c_void,
        count: u32,
    ) -> *mut c_void;
    pub(super) fn UnmapViewOfFile2(process: *mut c_void, address: *mut c_void, flags: u32) -> i32;
}
