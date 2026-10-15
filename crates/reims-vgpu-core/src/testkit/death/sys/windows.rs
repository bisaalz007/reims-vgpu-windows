//! Windows file views and debugger events. Rust's fast-fail abort bypasses in-process handlers,
//! so the parent records the second-chance exception while every child thread is suspended.

use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::io::{self, ErrorKind};
use std::os::windows::fs::FileExt as WindowsFileExt;
use std::os::windows::io::{AsRawHandle as _, FromRawHandle as _, OwnedHandle};
use std::os::windows::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus};
use std::time::{Duration, Instant};

use super::super::{
    ENV_FILE, H_CURSOR, H_OVERFLOW, K_DEATH, NAME_MAX, REC_HEAD, SLOT, SLOTS, TRACE_OFF, put,
};
use crate::num::{ToU64, ToUsize, Truncate};

/// Rust std's `FAST_FAIL_FATAL_APP_EXIT`, observed as `STATUS_STACK_BUFFER_OVERRUN`.
pub const ABORT: i32 = 0xc000_0409u32.cast_signed();
/// A native breakpoint exception, without a POSIX signal.
pub const TRAP: i32 = 0x8000_0003u32.cast_signed();
/// A stack overflow's native Windows status, measured both with and without this harness.
pub const OVERFLOW: i32 = 0xc000_00fdu32.cast_signed();
#[cfg(test)]
pub const OVERFLOW_LINE: &str = "has overflowed its stack";

// The EXCEPTION_DEBUG_INFO is 160 bytes on the supported 64-bit Windows host.
#[repr(C)]
struct DebugEvent {
    code: u32,
    process: u32,
    thread: u32,
    info: [u64; 20],
}

const _: () = assert!(size_of::<DebugEvent>() == 176);

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
    fn MapViewOfFileEx(
        section: *mut c_void,
        access: u32,
        high: u32,
        low: u32,
        len: usize,
        addr: *mut c_void,
    ) -> *mut c_void;
    fn VirtualFree(addr: *mut c_void, len: usize, kind: u32) -> i32;
    fn CloseHandle(handle: *mut c_void) -> i32;
    fn GetCurrentThreadId() -> u32;
    fn WaitForDebugEvent(event: *mut DebugEvent, timeout: u32) -> i32;
    fn ContinueDebugEvent(process: u32, thread: u32, status: u32) -> i32;
    fn OpenThread(access: u32, inherit: i32, thread: u32) -> *mut c_void;
    fn GetThreadDescription(thread: *mut c_void, description: *mut *mut u16) -> i32;
    fn LocalFree(address: *mut c_void) -> *mut c_void;
    fn ExitProcess(code: u32) -> !;
    #[cfg(test)]
    fn RaiseException(code: u32, flags: u32, count: u32, args: *const usize);
}

/// Positioned file operations using std's Windows FileExt, including short reads and writes.
pub trait FileExt {
    /// Reads the whole buffer beginning at `off`, independent of the current file cursor.
    fn read_exact_at(&self, buf: &mut [u8], off: u64) -> io::Result<()>;
    /// Writes the whole buffer beginning at `off`, independent of the current file cursor.
    fn write_all_at(&self, buf: &[u8], off: u64) -> io::Result<()>;
}

impl FileExt for File {
    fn read_exact_at(&self, mut buf: &mut [u8], mut off: u64) -> io::Result<()> {
        while !buf.is_empty() {
            match self.seek_read(buf, off) {
                Ok(0) => return Err(ErrorKind::UnexpectedEof.into()),
                Ok(n) => {
                    buf = buf.get_mut(n..).unwrap_or_default();
                    off = off.checked_add(n.to_u64()).ok_or(ErrorKind::InvalidInput)?;
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    fn write_all_at(&self, mut buf: &[u8], mut off: u64) -> io::Result<()> {
        while !buf.is_empty() {
            match self.seek_write(buf, off) {
                Ok(0) => return Err(ErrorKind::WriteZero.into()),
                Ok(n) => {
                    buf = buf.get(n..).unwrap_or_default();
                    off = off.checked_add(n.to_u64()).ok_or(ErrorKind::InvalidInput)?;
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

pub fn this_thread() -> (u64, [u8; NAME_MAX], usize) {
    let thread = std::thread::current();
    let name = thread.name().unwrap_or_default().as_bytes();
    let len = name.len().min(NAME_MAX);
    let mut bytes = [0u8; NAME_MAX];
    put(&mut bytes, 0, name.get(..len).unwrap_or_default());
    // SAFETY: a query with no caller memory.
    (u64::from(unsafe { GetCurrentThreadId() }), bytes, len)
}

pub fn arm_handlers() {}

pub fn page_size() -> usize {
    // Windows file-view addresses have 64 KiB allocation granularity, not the 4 KiB guest page.
    0x1_0000
}

fn section(file: &File) -> OwnedHandle {
    // SAFETY: a writable file of fixed size; no caller-owned address is replaced.
    let handle = unsafe {
        CreateFileMappingW(
            file.as_raw_handle(),
            std::ptr::null(),
            4,
            0,
            0,
            std::ptr::null(),
        )
    };
    assert!(
        !handle.is_null(),
        "death harness: CreateFileMappingW: {}",
        io::Error::last_os_error()
    );
    // SAFETY: this newly created handle has one owner.
    unsafe { OwnedHandle::from_raw_handle(handle) }
}

pub fn map(file: &File, len: usize) -> *mut u8 {
    let section = section(file);
    // SAFETY: a shared view; the child keeps it until process exit.
    unsafe { MapViewOfFileEx(section.as_raw_handle(), 2, 0, 0, len, std::ptr::null_mut()) }.cast()
}

fn region_path(path: &Path, off: u64) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(format!(".{off:x}.ram"));
    PathBuf::from(name)
}

#[expect(
    clippy::unwrap_used,
    reason = "a failed native mapping fails the death harness"
)]
pub unsafe fn remap(_file: &File, start: usize, span: usize, off: u64, now: &[u8]) {
    // A mapped Windows file cannot grow: each shared allocation gets its own fixed-size file.
    let path = region_path(&PathBuf::from(std::env::var_os(ENV_FILE).unwrap()), off);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    file.set_len(span.to_u64()).unwrap();
    file.write_all_at(now, 0).unwrap();
    let section = section(&file);
    let addr = std::ptr::with_exposed_provenance_mut::<c_void>(start);
    // SAFETY: RawRegion owns this whole virtual allocation, shared before other threads touch it.
    let freed = unsafe { VirtualFree(addr, 0, 0x8000) };
    assert_ne!(freed, 0, "death harness: VirtualFree");
    // SAFETY: replace the same addresses with the file's shared bytes; all region pointers survive.
    let p = unsafe { MapViewOfFileEx(section.as_raw_handle(), 2, 0, 0, span, addr) };
    assert_eq!(
        p,
        addr,
        "death harness: cannot remap {start:#x}+{span:#x}: {}",
        io::Error::last_os_error()
    );
}

pub fn read_region(_file: &File, path: &Path, off: u64, delta: u64, len: u64) -> Option<Vec<u8>> {
    let mut v = vec![0u8; usize::try_from(len).ok()?];
    if !v.is_empty() {
        File::open(region_path(path, off))
            .ok()?
            .read_exact_at(&mut v, delta)
            .ok()?;
    }
    Some(v)
}

pub fn remove_region(path: &Path, off: u64) {
    let _ = std::fs::remove_file(region_path(path, off));
}

pub fn configure(command: &mut Command) {
    command.creation_flags(2); // DEBUG_ONLY_THIS_PROCESS; the creator consumes its debug events.
}

fn thread_name(tid: u32) -> String {
    // SAFETY: a query-only handle; an absent thread simply has no name.
    let handle = unsafe { OpenThread(0x0800, 0, tid) };
    if handle.is_null() {
        return String::new();
    }
    // SAFETY: a newly opened handle owned here.
    let handle = unsafe { OwnedHandle::from_raw_handle(handle) };
    let mut description = std::ptr::null_mut();
    // SAFETY: the output pointer is writable; a successful query allocates a UTF-16 string.
    if unsafe { GetThreadDescription(handle.as_raw_handle(), &raw mut description) } < 0
        || description.is_null()
    {
        return String::new();
    }
    let mut len = 0usize;
    // SAFETY: the successful query returned a NUL-terminated UTF-16 allocation.
    while unsafe { description.add(len).read() } != 0 {
        len = len.saturating_add(1);
    }
    // SAFETY: `len` is the length just measured inside that allocation.
    let name = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(description, len) });
    // SAFETY: GetThreadDescription's returned allocation is released with LocalFree.
    unsafe { LocalFree(description.cast()) };
    name
}

#[expect(
    clippy::unwrap_used,
    reason = "a failed trace write fails the death harness"
)]
fn mark(file: &File, tid: u32, code: u32) {
    let mut cursor = [0u8; 8];
    file.read_exact_at(&mut cursor, H_CURSOR.to_u64()).unwrap();
    let slot = u64::from_ne_bytes(cursor);
    if slot >= SLOTS.to_u64() {
        file.write_all_at(&1u32.to_ne_bytes(), H_OVERFLOW.to_u64())
            .unwrap();
        return;
    }
    let name = thread_name(tid);
    let name = name
        .as_bytes()
        .get(..name.len().min(NAME_MAX))
        .unwrap_or_default();
    let mut record = [0u8; SLOT];
    put(&mut record, 4, &[K_DEATH, name.len().low_u8()]);
    put(&mut record, 8, &code.to_ne_bytes());
    put(&mut record, 16, &u64::from(tid).to_ne_bytes());
    put(&mut record, REC_HEAD, name);
    put(&mut record, 0, &1u32.to_ne_bytes());
    // All child threads are suspended at a debug event; no child writer races this reservation.
    file.write_all_at(
        &record,
        TRACE_OFF
            .to_u64()
            .saturating_add(slot.saturating_mul(SLOT.to_u64())),
    )
    .unwrap();
    file.write_all_at(&slot.saturating_add(1).to_ne_bytes(), H_CURSOR.to_u64())
        .unwrap();
}

#[expect(
    clippy::unwrap_used,
    reason = "a failed native subprocess operation fails the death harness"
)]
pub fn wait(child: &mut Child, file: &File, timeout: Duration) -> (ExitStatus, bool, Option<i32>) {
    let started = Instant::now();
    let mut timed_out = false;
    let mut initial_breakpoint = true;
    let mut exception = None;
    loop {
        let mut event = DebugEvent {
            code: 0,
            process: 0,
            thread: 0,
            info: [0; 20],
        };
        // SAFETY: the event is writable, and this thread created the debugged child.
        if unsafe { WaitForDebugEvent(&raw mut event, 2) } == 0 {
            let e = io::Error::last_os_error();
            assert_eq!(
                e.raw_os_error(),
                Some(121),
                "death harness: WaitForDebugEvent: {e}"
            );
        } else {
            let mut status = 0x0001_0002;
            match event.code {
                1 => {
                    let code = event.info.first().copied().unwrap_or(0).low_u32();
                    let first = event.info.get(19).copied().unwrap_or(0).low_u32() != 0;
                    if initial_breakpoint && code == TRAP.cast_unsigned() && first {
                        initial_breakpoint = false;
                    } else {
                        status = 0x8001_0001; // Let first-chance handlers run; terminate second-chance.
                        if !first {
                            mark(file, event.thread, code);
                            exception = Some(code.cast_signed());
                        }
                    }
                }
                3 | 6 => {
                    let handle = event.info.first().copied().unwrap_or(0);
                    if handle != 0 {
                        // SAFETY: the create-process/load-dll event hands its file handle to us.
                        unsafe {
                            CloseHandle(std::ptr::with_exposed_provenance_mut(handle.to_usize()))
                        };
                    }
                }
                _ => {}
            }
            // SAFETY: resume the event just received; process/thread handles close at their exit.
            let continued = unsafe { ContinueDebugEvent(event.process, event.thread, status) };
            assert_ne!(
                continued,
                0,
                "death harness: ContinueDebugEvent: {}",
                io::Error::last_os_error()
            );
            if event.code == 5 {
                let exit = child.wait().unwrap();
                return (
                    exit,
                    timed_out,
                    exception.filter(|code| Some(*code) == exit.code()),
                );
            }
        }
        if !timed_out && started.elapsed() >= timeout {
            timed_out = true;
            child.kill().unwrap();
        }
    }
}

pub fn termination(status: ExitStatus) -> Option<i32> {
    status
        .code()
        .filter(|code| code.cast_unsigned() & 0x8000_0000 != 0)
}

pub fn signal(_status: ExitStatus) -> i32 {
    0
}

pub fn aborts_with_line(code: i32) -> bool {
    code == ABORT || code == OVERFLOW
}

pub fn exit(status: i32) -> ! {
    // SAFETY: deliberately ends this harness child, without running any Drop.
    unsafe { ExitProcess(status.cast_unsigned()) }
}

#[cfg(test)]
pub fn trap() -> ! {
    // SAFETY: the harness's deliberate, noncontinuable native exception.
    unsafe { RaiseException(TRAP.cast_unsigned(), 1, 0, std::ptr::null()) };
    panic!("native trap returned")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Positioned I/O starts at the requested offset and rejects a short file as UnexpectedEof.
    #[test]
    fn positioned_io_rejects_a_truncated_read() {
        let path = std::env::temp_dir().join(format!("rvg-positioned-{}", std::process::id()));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all_at(b"xyz", 7).unwrap();
        let mut bytes = [0u8; 3];
        file.read_exact_at(&mut bytes, 7).unwrap();
        assert_eq!(&bytes, b"xyz");
        let mut short = [0u8; 4];
        assert_eq!(
            file.read_exact_at(&mut short, 7).unwrap_err().kind(),
            ErrorKind::UnexpectedEof
        );
        assert_eq!(short.get(..3), Some(b"xyz".as_slice()));
        file.read_exact_at(&mut [], 0).unwrap();
        file.write_all_at(&[], 0).unwrap();
        drop(file);
        std::fs::remove_file(path).unwrap();
    }
}
