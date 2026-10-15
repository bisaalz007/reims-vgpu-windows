//! Shared section views over Windows placeholders. A remap aliases pages; it never copies them.

use super::api::*;
use crate::os::backing::{self as files, Backing, Piece, Registration};
use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::vm::{HostAddr, HostVm, KernReturn, VmProt};
use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal};
use reims_vgpu_core::sync::Mutex;
use std::collections::BTreeMap;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::sync::Arc;

const INVALID_ADDRESS: KernReturn = KernReturn(1);
const PROTECTION_FAILURE: KernReturn = KernReturn(2);
const NO_SPACE: KernReturn = KernReturn(3);
const INVALID_ARGUMENT: KernReturn = KernReturn(4);

#[derive(Clone, Debug)]
struct View {
    length: u64,
    backing: Backing,
    prot: i32,
}
static VIEWS: Mutex<BTreeMap<u64, View>> = Mutex::new(BTreeMap::new());

pub(super) fn page_size() -> u64 {
    static PAGE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *PAGE.get_or_init(|| {
        let mut info = SystemInfo::default();
        // SAFETY: a live SYSTEM_INFO output.
        unsafe { GetSystemInfo(&mut info) };
        u64::from(info.page_size)
    })
}

fn pages(address: HostAddr, length: u64) -> Option<(u64, u64)> {
    let page = page_size();
    let first = address.addr().to_u64() & !(page - 1);
    let end = address
        .addr()
        .to_u64()
        .checked_add(length)?
        .checked_next_multiple_of(page)?;
    Some((
        first,
        if length == 0 {
            0
        } else {
            end.checked_sub(first)?
        },
    ))
}

fn protection(prot: i32) -> u32 {
    if prot & 2 != 0 {
        PAGE_READWRITE
    } else if prot & 1 != 0 {
        PAGE_READONLY
    } else {
        PAGE_NOACCESS
    }
}
fn ptr(at: u64) -> *mut std::ffi::c_void {
    std::ptr::with_exposed_provenance_mut(at.to_usize())
}
fn error(call: &str, kr: KernReturn) -> KernReturn {
    log::error!(target: "reims_vgpu_vulkan::os", "{call}: {}", std::io::Error::last_os_error());
    kr
}
fn cannot_restore(call: &'static str) -> ! {
    error(call, INVALID_ADDRESS);
    fatal(DeathId::Platform {
        kind: DeathKind::Abort,
        call,
    })
}

fn section(length: u64) -> Result<Arc<OwnedHandle>, KernReturn> {
    let high = u32::try_from(length >> 32).map_err(|_| NO_SPACE)?;
    let low = u32::try_from(length & u64::from(u32::MAX)).map_err(|_| NO_SPACE)?;
    // SAFETY: an unnamed, pagefile-backed section; no user pointer is dereferenced.
    let handle = unsafe {
        CreateFileMappingW(
            std::ptr::with_exposed_provenance_mut(usize::MAX),
            std::ptr::null(),
            PAGE_READWRITE,
            high,
            low,
            std::ptr::null(),
        )
    };
    if handle.is_null() {
        return Err(error("CreateFileMappingW", NO_SPACE));
    }
    // SAFETY: this call owns the new section handle.
    Ok(Arc::new(unsafe { OwnedHandle::from_raw_handle(handle) }))
}

fn reserve(length: u64) -> Result<u64, KernReturn> {
    // SAFETY: reserves fresh address space, without committing physical memory.
    let base = unsafe {
        VirtualAlloc2(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            length.to_usize(),
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            PAGE_NOACCESS,
            std::ptr::null(),
            0,
        )
    };
    if base.is_null() {
        Err(error("VirtualAlloc2", NO_SPACE))
    } else {
        Ok(base.addr().to_u64())
    }
}

fn map(first: u64, length: u64, backing: &Backing, prot: i32) -> Result<(), KernReturn> {
    // SAFETY: the exact placeholder is replaced with a view of a retained section. This flag
    // permits page-aligned offsets and addresses, including the guest's 4 KiB pages.
    let mapped = unsafe {
        MapViewOfFile3(
            backing.file.as_raw_handle(),
            GetCurrentProcess(),
            ptr(first),
            backing.offset,
            length.to_usize(),
            MEM_REPLACE_PLACEHOLDER,
            protection(prot),
            std::ptr::null(),
            0,
        )
    };
    if mapped != ptr(first) {
        Err(error("MapViewOfFile3", NO_SPACE))
    } else {
        Ok(())
    }
}
fn unmap(first: u64) {
    // SAFETY: a view owned by this table, restored to its placeholder.
    if unsafe { UnmapViewOfFile2(GetCurrentProcess(), ptr(first), MEM_PRESERVE_PLACEHOLDER) } == 0 {
        cannot_restore("UnmapViewOfFile2");
    }
}
fn split_placeholder(first: u64, length: u64) -> Result<(), KernReturn> {
    // SAFETY: splits an existing placeholder; it does not free the address range.
    if unsafe {
        VirtualFree(
            ptr(first),
            length.to_usize(),
            MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        )
    } == 0
    {
        Err(error("VirtualFree split placeholder", INVALID_ADDRESS))
    } else {
        Ok(())
    }
}
fn coalesce(first: u64, length: u64) {
    // SAFETY: adjacent placeholders created while replacing this owned range.
    if unsafe {
        VirtualFree(
            ptr(first),
            length.to_usize(),
            MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS,
        )
    } == 0
    {
        cannot_restore("VirtualFree coalesce placeholders");
    }
}

/// Installs pieces in one placeholder. On failure restores one empty placeholder.
fn install(first: u64, length: u64, pieces: &[Piece]) -> Result<(), KernReturn> {
    let mut mapped = 0usize;
    let mut splits = 0usize;
    let result = (|| {
        for (i, piece) in pieces.iter().enumerate() {
            if i + 1 < pieces.len() {
                split_placeholder(first + piece.at, piece.length)?;
                splits += 1;
            }
            map(first + piece.at, piece.length, &piece.backing, piece.prot)?;
            mapped += 1;
        }
        Ok(())
    })();
    if result.is_err() {
        for piece in pieces.iter().take(mapped) {
            unmap(first + piece.at);
        }
        if splits != 0 {
            coalesce(first, length);
        }
    }
    result
}

fn pieces(first: u64, views: &[(u64, View)]) -> Vec<Piece> {
    views
        .iter()
        .map(|(at, view)| Piece {
            at: at - first,
            length: view.length,
            backing: view.backing.clone(),
            prot: view.prot,
        })
        .collect()
}

/// Splits a live view without changing any of its bytes or protections.
fn split(views: &mut BTreeMap<u64, View>, at: u64) -> Result<(), KernReturn> {
    let Some((&first, view)) = views.range(..=at).next_back() else {
        return Ok(());
    };
    if at == first || at >= first + view.length {
        return Ok(());
    }
    let old = view.clone();
    let head = View {
        length: at - first,
        ..old.clone()
    };
    let tail = View {
        length: first + old.length - at,
        backing: Backing {
            file: Arc::clone(&old.backing.file),
            offset: old.backing.offset + at - first,
        },
        prot: old.prot,
    };
    unmap(first);
    let cut = pieces(first, &[(first, head.clone()), (at, tail.clone())]);
    if let Err(kr) = install(first, old.length, &cut) {
        if install(first, old.length, &pieces(first, &[(first, old)])).is_err() {
            cannot_restore("restore split section view");
        }
        return Err(kr);
    }
    views.insert(first, head);
    views.insert(at, tail);
    Ok(())
}

fn replace(first: u64, length: u64, new: &[Piece]) -> Result<(), KernReturn> {
    let end = first.checked_add(length).ok_or(INVALID_ARGUMENT)?;
    let mut views = VIEWS.lock();
    let mut at = first;
    while at < end {
        let Some((&base, view)) = views.range(..=at).next_back() else {
            return Err(INVALID_ADDRESS);
        };
        if at >= base + view.length {
            return Err(INVALID_ADDRESS);
        }
        at = (base + view.length).min(end);
    }
    split(&mut views, first)?;
    split(&mut views, end)?;
    let old: Vec<_> = views
        .range(first..end)
        .map(|(&at, v)| (at, v.clone()))
        .collect();
    for (at, _) in &old {
        unmap(*at);
    }
    if old.len() > 1 {
        coalesce(first, length);
    }
    if let Err(kr) = install(first, length, new) {
        if install(first, length, &pieces(first, &old)).is_err() {
            cannot_restore("restore overwritten section views");
        }
        return Err(kr);
    }
    for (at, _) in old {
        views.remove(&at);
    }
    for piece in new {
        views.insert(
            first + piece.at,
            View {
                length: piece.length,
                backing: piece.backing.clone(),
                prot: piece.prot,
            },
        );
    }
    Ok(())
}

/// Maps the retained sections and records their views, including writable import aliases.
pub(super) fn map_parts(first: u64, length: u64, parts: &[Piece]) -> Result<(), KernReturn> {
    replace(first, length, parts)?;
    for piece in parts {
        files::record(
            first + piece.at,
            piece.length,
            piece.backing.clone(),
            piece.prot,
        );
    }
    Ok(())
}

fn allocate(length: u64) -> Result<HostAddr, KernReturn> {
    let backing = Backing {
        file: section(length)?,
        offset: 0,
    };
    let first = reserve(length)?;
    if let Err(kr) = map(first, length, &backing, 3) {
        // SAFETY: the placeholder was never mapped.
        unsafe { VirtualFree(ptr(first), 0, MEM_RELEASE) };
        return Err(kr);
    }
    VIEWS.lock().insert(
        first,
        View {
            length,
            backing: backing.clone(),
            prot: 3,
        },
    );
    files::record(first, length, backing, 3);
    Ok(HostPtr::from_addr(first.to_usize()))
}

/// A zero-filled range whose pages can be shared in other views.
#[cfg(test)]
pub(crate) fn allocate_aliasable(length: u64) -> Result<HostAddr, KernReturn> {
    WindowsVm::new().allocate_anywhere(length)
}

/// Memory owned by an IOSurface, which core cannot deallocate.
#[derive(Debug)]
pub(crate) struct OwnMemory {
    first: HostAddr,
    length: u64,
    registration: Option<Registration>,
}
impl OwnMemory {
    pub(crate) fn new(length: u64) -> Option<Self> {
        let (_, length) = pages(HostAddr::NULL, length)?;
        if length == 0 {
            return None;
        }
        let first = allocate(length).ok()?;
        let backing = files::pieces(first.addr().to_u64(), length)?
            .first()?
            .backing
            .clone();
        Some(Self {
            first,
            length,
            registration: Some(Registration::new(first.addr().to_u64(), length, backing)),
        })
    }
    pub(crate) fn first(&self) -> u64 {
        self.first.addr().to_u64()
    }
}
impl Drop for OwnMemory {
    fn drop(&mut self) {
        drop(self.registration.take());
        let _ = WindowsVm.deallocate(self.first, self.length);
    }
}

/// Mach VM's shared-view operations on Windows.
#[derive(Debug, Default)]
pub struct WindowsVm;
impl WindowsVm {
    /// The process's virtual memory service.
    pub fn new() -> Self {
        Self
    }
    /// Whether each byte is committed and readable, without dereferencing guest pointers.
    pub fn readable(&self, address: HostAddr, length: u64) -> bool {
        if files::readable(address.addr().to_u64(), length) == Some(false) {
            return false;
        }
        let Some(end) = address.addr().to_u64().checked_add(length) else {
            return false;
        };
        let mut at = address.addr().to_u64();
        while at < end {
            let mut info = MemoryInfo::default();
            // SAFETY: queries an address without reading it, into a live output.
            if unsafe { VirtualQuery(ptr(at), &mut info, size_of::<MemoryInfo>()) } == 0
                || info.state != MEM_COMMIT
                || info.protection & (PAGE_NOACCESS | 0x100) != 0
            {
                return false;
            }
            let next = info
                .base
                .addr()
                .to_u64()
                .saturating_add(info.region_size.to_u64());
            if next <= at {
                return false;
            }
            at = next;
        }
        true
    }
}

impl HostVm for WindowsVm {
    fn page_size(&self) -> u64 {
        page_size()
    }
    fn page_shift(&self) -> u32 {
        page_size().trailing_zeros()
    }
    fn allocate_anywhere(&self, length: u64) -> Result<HostAddr, KernReturn> {
        let (_, length) = pages(HostAddr::NULL, length).ok_or(NO_SPACE)?;
        if length == 0 {
            Ok(HostAddr::NULL)
        } else {
            allocate(length)
        }
    }
    fn allocate_fixed_overwrite(&self, address: HostAddr, length: u64) -> Result<(), KernReturn> {
        let (first, length) = pages(address, length).ok_or(INVALID_ARGUMENT)?;
        if length == 0 {
            return Ok(());
        }
        let backing = Backing {
            file: section(length)?,
            offset: 0,
        };
        replace(
            first,
            length,
            &[Piece {
                at: 0,
                length,
                backing: backing.clone(),
                prot: 3,
            }],
        )?;
        files::record(first, length, backing, 3);
        Ok(())
    }
    fn remap_shared_overwrite(
        &self,
        destination: HostAddr,
        source: HostAddr,
        length: u64,
    ) -> Result<HostAddr, KernReturn> {
        if length == 0 {
            return Ok(destination);
        }
        let offset = source.addr().to_u64() & (page_size() - 1);
        let (first, length) = pages(
            destination,
            length.checked_add(offset).ok_or(INVALID_ARGUMENT)?,
        )
        .ok_or(INVALID_ARGUMENT)?;
        let (source, _) = pages(source, 1).ok_or(INVALID_ADDRESS)?;
        let parts = files::pieces(source, length).ok_or(INVALID_ADDRESS)?;
        map_parts(first, length, &parts)?;
        Ok(destination)
    }
    fn protect_max(&self, address: HostAddr, length: u64, prot: VmProt) -> Result<(), KernReturn> {
        let (first, length) = pages(address, length).ok_or(INVALID_ARGUMENT)?;
        if length == 0 {
            return Ok(());
        }
        let end = first.checked_add(length).ok_or(INVALID_ARGUMENT)?;
        let mut views = VIEWS.lock();
        split(&mut views, first)?;
        split(&mut views, end)?;
        let mut at = first;
        for (&base, view) in views.range(first..end) {
            if base != at || prot.raw() & !view.prot != 0 {
                return Err(PROTECTION_FAILURE);
            }
            at += view.length;
        }
        if at != end {
            return Err(INVALID_ADDRESS);
        }
        for (&base, view) in views.range_mut(first..end) {
            let mut old = 0;
            // SAFETY: each view is owned and no access is made; only lowers its maximum.
            if unsafe {
                VirtualProtect(
                    ptr(base),
                    view.length.to_usize(),
                    protection(prot.raw()),
                    &mut old,
                )
            } == 0
            {
                return Err(error("VirtualProtect", PROTECTION_FAILURE));
            }
            view.prot = prot.raw();
        }
        files::protect(first, length, prot.raw());
        Ok(())
    }
    fn deallocate(&self, address: HostAddr, length: u64) -> KernReturn {
        let Some((first, length)) = pages(address, length) else {
            return INVALID_ARGUMENT;
        };
        if length == 0 {
            return KernReturn::SUCCESS;
        }
        if files::meets_registration(first, length) {
            return INVALID_ARGUMENT;
        }
        let end = first + length;
        let mut views = VIEWS.lock();
        if split(&mut views, first).is_err() || split(&mut views, end).is_err() {
            return INVALID_ADDRESS;
        }
        let bases: Vec<_> = views.range(first..end).map(|(&base, _)| base).collect();
        if bases.is_empty() {
            return INVALID_ADDRESS;
        }
        for base in bases {
            unmap(base);
            // SAFETY: releases the exact owned placeholder after its view is gone.
            if unsafe { VirtualFree(ptr(base), 0, MEM_RELEASE) } == 0 {
                cannot_restore("VirtualFree release placeholder");
            }
            views.remove(&base);
        }
        files::forget(first, length);
        KernReturn::SUCCESS
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub(crate) const PAGE: u64 = 4096;
    pub(crate) fn mapped(at: HostAddr) -> bool {
        WindowsVm.readable(at, 1)
    }
    pub(crate) fn writable(at: HostAddr, length: u64) -> bool {
        WindowsVm
            .protect_max(at, length, VmProt::READ | VmProt::WRITE)
            .is_ok()
    }

    #[test]
    fn aliases_are_shared_page_aligned_and_outlive_the_source() {
        let vm = WindowsVm;
        let source = vm.allocate_anywhere(3 * PAGE).unwrap();
        let target = vm.allocate_anywhere(3 * PAGE).unwrap();
        let middle = HostAddr::from_addr(source.addr() + PAGE.to_usize());
        // SAFETY: live writable allocations.
        unsafe { middle.as_ptr::<u8>().write(7) };
        vm.remap_shared_overwrite(target, middle, PAGE).unwrap();
        // SAFETY: the target shares the source's middle page.
        unsafe {
            assert_eq!(target.as_ptr::<u8>().read(), 7);
            target.as_ptr::<u8>().write(9);
            assert_eq!(middle.as_ptr::<u8>().read(), 9);
        }
        assert!(vm.deallocate(source, 3 * PAGE).is_success());
        // SAFETY: the alias independently keeps the section's page alive.
        unsafe {
            assert_eq!(target.as_ptr::<u8>().read(), 9);
        }
        assert!(vm.deallocate(target, 3 * PAGE).is_success());
        assert!(!mapped(target));
    }
    #[test]
    fn failed_remap_preserves_bytes_and_overwrite_only_zeros_its_page() {
        let vm = WindowsVm;
        let at = vm.allocate_anywhere(3 * PAGE).unwrap();
        // SAFETY: writable live allocation.
        unsafe {
            at.as_ptr::<u8>().write(4);
            at.as_ptr::<u8>().add(PAGE.to_usize()).write(5);
        }
        assert!(vm.remap_shared_overwrite(at, HostAddr::NULL, PAGE).is_err());
        vm.allocate_fixed_overwrite(HostAddr::from_addr(at.addr() + PAGE.to_usize()), PAGE)
            .unwrap();
        // SAFETY: same allocation, with a fresh middle page.
        unsafe {
            assert_eq!(at.as_ptr::<u8>().read(), 4);
            assert_eq!(at.as_ptr::<u8>().add(PAGE.to_usize()).read(), 0);
        }
        assert!(vm.deallocate(at, 3 * PAGE).is_success());
    }
}
