use super::super::memory::host_page_size;
use super::*;
use std::mem::MaybeUninit;

/// A file of `len` bytes in the temporary directory, removed at its drop.
struct TempFile {
    path: std::path::PathBuf,
    file: std::fs::File,
}

impl TempFile {
    fn new(name: &str, len: u64) -> TempFile {
        let path =
            std::env::temp_dir().join(format!("rvg-memory-file-{name}-{}", std::process::id()));
        let file = std::fs::File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        file.set_len(len).unwrap();
        TempFile { path, file }
    }

    fn fd(&self) -> i32 {
        std::os::fd::AsRawFd::as_raw_fd(&self.file)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// The descriptors open in this process (nextest runs each test in a process of its own).
fn open_fds() -> usize {
    std::fs::read_dir("/dev/fd").unwrap().count()
}

/// Each file refusal in the header's order, the first range's first, each with its line and no
/// descriptor duplicated
#[test]
fn memory_file_refusals() {
    let p = test_platform();
    let page = host_page_size().get();
    let file = TempFile::new("refusals", 0x1_0000);
    let range = fake_range(&p, 0x8000);
    let ranges = [range, range];
    let map = RvgMemoryMap {
        ranges: ranges.as_ptr(),
        range_count: 2,
    };
    let ok = RvgMemoryRangeFile {
        fd: file.fd(),
        flags: 0,
        offset: 0x8000,
    };
    let none = RvgMemoryRangeFile {
        fd: -1,
        flags: 0,
        offset: 1,
    };
    let entry = |fd: i32, flags: u32, offset: u64| RvgMemoryRangeFile { fd, flags, offset };
    let logs = reims_vgpu_core::testkit::logs::capture();
    let fds = open_fds();
    let refused = |files: [RvgMemoryRangeFile; 2]| -> String {
        // SAFETY: a live map of two ranges, and two entries.
        let r = unsafe { memory_map(&map, files.as_ptr(), std::ptr::null()) };
        assert_eq!(r.err(), Some(RVG_ERR_INVALID_ARGUMENT));
        let lines = logs.take();
        assert_eq!(lines.len(), 1, "{lines:?}");
        lines[0].2.clone()
    };
    // Files without a map.
    // SAFETY: no map; the entry is live.
    let r = unsafe { memory_map(std::ptr::null(), &ok, std::ptr::null()) };
    assert_eq!(r.err(), Some(RVG_ERR_INVALID_ARGUMENT));
    assert_eq!(
        logs.take()[0].2,
        "device not created: memory files without a memory map"
    );
    // Each check, on range 1 after a range 0 that passes.
    assert_eq!(
        refused([ok, entry(-2, 1, 0x10)]),
        "device not created: the file of memory range 1 has fd -2, below -1"
    );
    assert_eq!(
        refused([ok, entry(i32::MAX, 1, 0x10)]),
        "device not created: the file of memory range 1 has flags 0x1, not 0"
    );
    assert_eq!(
        refused([ok, entry(-1, 2, 0)]),
        "device not created: the file of memory range 1 has flags 0x2, not 0"
    );
    let unreadable = refused([ok, entry(i32::MAX, 0, 0x10)]);
    assert!(
        unreadable.starts_with(
            "device not created: the file of memory range 1 has fd 2147483647, which fstat \
             cannot read: "
        ),
        "{unreadable}"
    );
    assert_eq!(
        refused([ok, entry(file.fd(), 0, 0x8010)]),
        "device not created: the file of memory range 1 holds 0x10000 bytes, fewer than its \
         offset 0x8010 and the range's 0x8000"
    );
    assert_eq!(
        refused([ok, entry(file.fd(), 0, u64::MAX - 1)]),
        "device not created: the file of memory range 1 holds 0x10000 bytes, fewer than its \
         offset 0xfffffffffffffffe and the range's 0x8000"
    );
    assert_eq!(
        refused([ok, entry(file.fd(), 0, 0x10)]),
        format!(
            "device not created: the file of memory range 1 has offset 0x10, not a multiple of \
             the page size {page:#x}"
        )
    );
    // The first range's refusal comes first.
    assert_eq!(
        refused([entry(file.fd(), 0, 0x10), entry(-2, 0, 0)]),
        format!(
            "device not created: the file of memory range 0 has offset 0x10, not a multiple of \
             the page size {page:#x}"
        )
    );
    // No refusal duplicated a descriptor; a range without a file takes no offset check.
    assert_eq!(open_fds(), fds);
    // SAFETY: a live map of two ranges, and two entries.
    let m = unsafe { memory_map(&map, [none, none].as_ptr(), std::ptr::null()) }.unwrap();
    assert!(m.unwrap().files().iter().all(Option::is_none));
    assert!(logs.take().is_empty());
}

/// The files the map takes: each a duplicate of the host's descriptor, close-on-exec, of the
/// same file, at the host's offset, in the ranges' order; none for -1 or NULL; the duplicates
/// close with the last map that holds them
#[test]
fn memory_files_are_duplicated() {
    let p = test_platform();
    let file = TempFile::new("duplicated", 0x2_0000);
    let range = fake_range(&p, 0x8000);
    let ranges = [range, range];
    let map = RvgMemoryMap {
        ranges: ranges.as_ptr(),
        range_count: 2,
    };
    let files = [
        RvgMemoryRangeFile {
            fd: -1,
            flags: 0,
            offset: 0,
        },
        RvgMemoryRangeFile {
            fd: file.fd(),
            flags: 0,
            offset: 0x1_0000,
        },
    ];
    let fds = open_fds();
    // SAFETY: a live map of two ranges, and two entries.
    let m = unsafe { memory_map(&map, files.as_ptr(), std::ptr::null()) }
        .unwrap()
        .unwrap();
    assert_eq!(open_fds(), fds + 1);
    assert!(m.files()[0].is_none());
    let f = m.files()[1].as_ref().unwrap();
    assert_eq!(f.offset, 0x1_0000);
    let dup = std::os::fd::AsRawFd::as_raw_fd(&*f.fd);
    assert_ne!(dup, file.fd());
    // SAFETY: a query of a descriptor the map holds open.
    let flags = unsafe { libc::fcntl(dup, libc::F_GETFD) };
    assert_eq!(flags & libc::FD_CLOEXEC, libc::FD_CLOEXEC);
    let ino = |fd: i32| {
        let mut st = MaybeUninit::<libc::stat>::zeroed();
        // SAFETY: `st` is a writable `struct stat`, and `fd` is open.
        assert_eq!(unsafe { libc::fstat(fd, st.as_mut_ptr()) }, 0);
        // SAFETY: filled by the call above.
        let st = unsafe { st.assume_init() };
        (st.st_dev, st.st_ino)
    };
    assert_eq!(ino(dup), ino(file.fd()));
    // NULL files: none.
    // SAFETY: a live map of two ranges.
    let bare = unsafe { memory_map(&map, std::ptr::null(), std::ptr::null()) }
        .unwrap()
        .unwrap();
    assert!(bare.files().iter().all(Option::is_none));
    let copy = m.clone();
    drop(m);
    assert_eq!(open_fds(), fds + 1);
    drop(copy);
    assert_eq!(open_fds(), fds);
}
