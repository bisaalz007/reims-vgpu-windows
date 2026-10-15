use std::os::windows::io::{AsRawHandle as _, FromRawHandle as _, OwnedHandle};

use super::*;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn CreateFileMappingW(
        file: *mut c_void,
        attrs: *const c_void,
        protection: u32,
        high: u32,
        low: u32,
        name: *const u16,
    ) -> *mut c_void;
    fn MapViewOfFile(
        section: *mut c_void,
        access: u32,
        high: u32,
        low: u32,
        len: usize,
    ) -> *mut c_void;
    fn UnmapViewOfFile(address: *const c_void) -> i32;
    fn GetHandleInformation(handle: *mut c_void, flags: *mut u32) -> i32;
}

fn section(length: u32) -> OwnedHandle {
    // SAFETY: a fresh pagefile-backed, writable section of the requested size.
    let raw = unsafe {
        CreateFileMappingW(
            std::ptr::with_exposed_provenance_mut(usize::MAX),
            std::ptr::null(),
            4,
            0,
            length,
            std::ptr::null(),
        )
    };
    assert!(!raw.is_null());
    // SAFETY: the new section has one owner.
    unsafe { OwnedHandle::from_raw_handle(raw) }
}

fn descriptor(section: &OwnedHandle, offset: u64, length: u64) -> RvgMemoryRangeSection {
    RvgMemoryRangeSection {
        handle: u64::try_from(section.as_raw_handle().addr()).unwrap(),
        offset,
        length,
        flags: 0,
        reserved: 0,
    }
}

/// Old file descriptors are refused; a 56-byte config reads no appended section pointer.
#[test]
fn windows_files_are_refused_and_56_byte_configs_ignore_sections() {
    let p = test_platform();
    let range = fake_range(&p, 0x4000);
    let map = RvgMemoryMap {
        ranges: &range,
        range_count: 1,
    };
    let file = RvgMemoryRangeFile {
        fd: -1,
        flags: 0,
        offset: 0,
    };
    // SAFETY: one live range and one live file entry.
    assert_eq!(
        unsafe { memory_map(&map, &file, std::ptr::null()) }.err(),
        Some(RVG_ERR_INVALID_ARGUMENT)
    );
    let rec = Rec::new();
    let table = only_destroy(&rec);
    let mut c = config(1, &table);
    c.memory_map = &map;
    c.memory_sections = std::ptr::dangling();
    c.struct_size = 56;
    let (status, device) = create(&c);
    assert_eq!(status, RVG_OK);
    // SAFETY: the device just created.
    unsafe { rvg_device_release(device) };
    assert_eq!(rec.destroys().len(), 1);
}

/// Invalid handles, arithmetic, sizes, alignment and reserved fields all reach real refusal paths.
#[test]
fn memory_section_refusals() {
    let p = test_platform();
    let range = fake_range(&p, 0x4000);
    let map = RvgMemoryMap {
        ranges: &range,
        range_count: 1,
    };
    let host = section(0x1_0000);
    let good = descriptor(&host, 0, 0x1_0000);
    let refused = |entry: RvgMemoryRangeSection| {
        // SAFETY: one live range and one live descriptor.
        assert_eq!(
            unsafe { memory_map(&map, std::ptr::null(), &entry) }.err(),
            Some(RVG_ERR_INVALID_ARGUMENT)
        );
    };
    refused(RvgMemoryRangeSection {
        handle: u64::MAX,
        ..good
    });
    refused(RvgMemoryRangeSection {
        length: 0x2000,
        ..good
    });
    refused(RvgMemoryRangeSection {
        offset: u64::MAX - 1,
        length: u64::MAX,
        ..good
    });
    refused(RvgMemoryRangeSection { offset: 1, ..good });
    refused(RvgMemoryRangeSection {
        length: 0x4_0000,
        ..good
    });
    refused(RvgMemoryRangeSection { flags: 1, ..good });
    refused(RvgMemoryRangeSection {
        reserved: 1,
        ..good
    });
    let file = RvgMemoryRangeFile {
        fd: -1,
        flags: 0,
        offset: 0,
    };
    // SAFETY: live matching entries; the combination is refused before reading them.
    assert_eq!(
        unsafe { memory_map(&map, &file, &good) }.err(),
        Some(RVG_ERR_INVALID_ARGUMENT)
    );
    // SAFETY: no map; sections without one are refused.
    assert_eq!(
        unsafe { memory_map(std::ptr::null(), std::ptr::null(), &good) }.err(),
        Some(RVG_ERR_INVALID_ARGUMENT)
    );
    // SAFETY: a live range without section backing is accepted.
    assert!(
        unsafe { memory_map(&map, std::ptr::null(), std::ptr::null()) }
            .unwrap()
            .unwrap()
            .files()[0]
            .is_none()
    );
}

/// The noninheritable duplicate retains the same shared pages after the host closes its handle.
#[test]
fn memory_sections_are_duplicated_and_alias_the_host_pages() {
    let host = section(0x2_0000);
    let entry = descriptor(&host, 0x1_0000, 0x2_0000);
    // SAFETY: a writable view of the host's section at an allocation-aligned offset.
    let view = unsafe { MapViewOfFile(host.as_raw_handle(), 2, 0, 0x1_0000, 0x4000) };
    assert!(!view.is_null());
    let range = RvgMemoryRange {
        physical_address: 0,
        physical_length: 0x4000,
        host_address: view,
    };
    let map = RvgMemoryMap {
        ranges: &range,
        range_count: 1,
    };
    // SAFETY: the range is really backed by this section and remains mapped until after the map drops.
    let m = unsafe { memory_map(&map, std::ptr::null(), &entry) }
        .unwrap()
        .unwrap();
    let backing = m.files()[0].as_ref().unwrap();
    assert_eq!(backing.offset, 0x1_0000);
    let duplicate = backing.handle.as_raw_handle();
    assert_ne!(duplicate, host.as_raw_handle());
    let mut flags = 0u32;
    // SAFETY: a query of the duplicate that the map owns.
    assert_ne!(
        unsafe { GetHandleInformation(duplicate, &raw mut flags) },
        0
    );
    assert_eq!(flags & 1, 0);
    drop(host);
    // SAFETY: the duplicate still names the same section after the original handle closes.
    let second = unsafe { MapViewOfFile(duplicate, 2, 0, 0x1_0000, 0x4000) };
    assert!(!second.is_null());
    // SAFETY: both writable views hold this byte and really alias the same section.
    unsafe {
        view.cast::<u8>().write(0xa5);
        assert_eq!(second.cast::<u8>().read(), 0xa5);
        second.cast::<u8>().write(0x5a);
        assert_eq!(view.cast::<u8>().read(), 0x5a);
    }
    let copy = m.clone();
    drop(m);
    // SAFETY: the last cloned map still owns the duplicate.
    assert_ne!(
        unsafe { GetHandleInformation(duplicate, &raw mut flags) },
        0
    );
    drop(copy);
    // SAFETY: querying a closed handle returns an error, without taking ownership.
    assert_eq!(
        unsafe { GetHandleInformation(duplicate, &raw mut flags) },
        0
    );
    // SAFETY: release the two views created above.
    unsafe {
        assert_ne!(UnmapViewOfFile(second), 0);
        assert_ne!(UnmapViewOfFile(view), 0);
    }
}
