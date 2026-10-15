//! POSIX mapping, signal marking and subprocess waiting for the death harness.

use std::ffi::{c_char, c_int, c_void};
use std::fs::File;
use std::os::fd::AsRawFd as _;
use std::os::unix::process::ExitStatusExt as _;
use std::path::Path;
use std::process::{Child, Command, ExitStatus};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::super::{K_DEATH, NAME_MAX, append_here};
use super::raw as sys;
use crate::num::ToU64;
pub use std::os::unix::fs::FileExt;

pub const ABORT: i32 = sys::SIGABRT;
#[cfg(test)]
pub const TRAP: i32 = sys::SIGTRAP;
#[cfg(test)]
pub const OVERFLOW: i32 = sys::SIGABRT;
#[cfg(test)]
pub const OVERFLOW_LINE: &str = "fatal runtime error: stack overflow";

/// The calling thread's system id and name, without allocating (the signal handler calls it).
pub fn this_thread() -> (u64, [u8; NAME_MAX], usize) {
    let tid = sys::current_thread_id();
    let mut name = [0u8; NAME_MAX];
    // SAFETY: `pthread_self()` is the calling thread; the call writes within the buffer given.
    // For the calling thread it takes no lock (async-signal-safe in practice).
    unsafe {
        sys::pthread_getname_np(
            sys::pthread_self(),
            name.as_mut_ptr().cast::<c_char>(),
            NAME_MAX,
        );
    }
    let len = name.iter().position(|b| *b == 0).unwrap_or(NAME_MAX);
    (tid, name, len)
}

/// The actions the harness's handler replaced, by signal: the handler (`SIG_DFL` is 0) and its
/// flags. Stored before the handler is installed; the handler only reads them.
static PREV_HANDLER: [AtomicUsize; sys::NSIG] =
    [const { AtomicUsize::new(sys::SIG_DFL) }; sys::NSIG];
static PREV_FLAGS: [AtomicU32; sys::NSIG] = [const { AtomicU32::new(0) }; sys::NSIG];

/// The handler for the signals that end a child: it marks the death on the dying thread and
/// leaves the signal to do what it would have done without the harness.
///
/// - With no earlier handler (`SIG_DFL`), it appends the death marker, restores the default
///   action and re-raises: the process dies of the signal when the handler returns.
/// - With one (std installs its stack-overflow handler for `SIGSEGV` and `SIGBUS`, on an
///   alternate stack), it calls that handler first, as the signal would have. std's reports an
///   overflow and aborts, so the death is marked by the `SIGABRT` it raises; for any other fault
///   it restores the default action and returns. If it returns, the handler appends the marker
///   and returns, and the faulting instruction runs again and ends the process.
extern "C" fn on_fatal_signal(sig: c_int, info: *mut c_void, ctx: *mut c_void) {
    let i = usize::try_from(sig).unwrap_or(0);
    let prev = PREV_HANDLER
        .get(i)
        .map_or(sys::SIG_DFL, |h| h.load(Ordering::Acquire));
    let flags = PREV_FLAGS.get(i).map_or(0, |f| f.load(Ordering::Acquire));
    if prev == sys::SIG_DFL || prev == sys::SIG_IGN {
        append_here(K_DEATH, sig.cast_unsigned(), b"");
        let _ = set_action(sig, sys::Handler { raw: sys::SIG_DFL }, 0);
        // `raise` pends (the signal is blocked while its handler runs) and is delivered, with
        // the default action, when the handler returns.
        let _ = sys::raise(sig);
        return;
    }
    let chained = sys::Handler { raw: prev };
    // SAFETY: `prev` is the handler `sigaction` reported for `sig` (neither `SIG_DFL` nor
    // `SIG_IGN`), called the way its flags say the kernel would have called it.
    unsafe {
        if flags & sys::SA_SIGINFO.cast_unsigned() != 0 {
            (chained.action)(sig, info, ctx);
        } else {
            (chained.handler)(sig);
        }
    }
    append_here(K_DEATH, sig.cast_unsigned(), b"");
}

/// Sets `sig`'s action to `handler` (on the alternate stack, with `flags`), and returns the
/// action it replaced.
fn set_action(sig: c_int, handler: sys::Handler, flags: c_int) -> Option<sys::SigAction> {
    let act = sys::SigAction::new(handler, flags | sys::SA_ONSTACK);
    let mut old = sys::SigAction::new(sys::Handler { raw: sys::SIG_DFL }, 0);
    // SAFETY: both structs are the C library's `struct sigaction`, and live across the call.
    let r = unsafe { sys::sigaction(sig, &raw const act, &raw mut old) };
    (r == 0).then_some(old)
}

/// Installs [`on_fatal_signal`] for `sig`, keeping the action it replaces for the handler to
/// chain to. A signal the process ignores is left ignored.
fn arm(sig: c_int) {
    let mut cur = sys::SigAction::new(sys::Handler { raw: sys::SIG_DFL }, 0);
    // SAFETY: a query: `act` is null and `old` is a live `struct sigaction`.
    if unsafe { sys::sigaction(sig, std::ptr::null(), &raw mut cur) } != 0 {
        return;
    }
    // SAFETY: every bit pattern of the union is a valid `usize`.
    let raw = unsafe { cur.handler.raw };
    if raw == sys::SIG_IGN {
        return;
    }
    let i = usize::try_from(sig).unwrap_or(0);
    if let (Some(h), Some(f)) = (PREV_HANDLER.get(i), PREV_FLAGS.get(i)) {
        h.store(raw, Ordering::Release);
        f.store(cur.flags.cast_unsigned(), Ordering::Release);
    }
    let _ = set_action(
        sig,
        sys::Handler {
            action: on_fatal_signal,
        },
        sys::SA_SIGINFO,
    );
}

pub fn arm_handlers() {
    sys::widen_alternate_stack();
    for sig in [
        sys::SIGABRT,
        sys::SIGTRAP,
        sys::SIGILL,
        sys::SIGSEGV,
        sys::SIGBUS,
    ] {
        arm(sig);
    }
}

pub fn page_size() -> usize {
    usize::try_from(sys::getpagesize()).unwrap_or(0x4000)
}

pub fn map(file: &File, len: usize) -> *mut u8 {
    // SAFETY: a shared view of the file the parent sized; it lives until the child ends.
    unsafe {
        sys::mmap(
            std::ptr::null_mut(),
            len,
            sys::PROT_READ | sys::PROT_WRITE,
            sys::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    }
    .cast()
}

pub unsafe fn remap(file: &File, start: usize, span: usize, off: u64, now: &[u8]) {
    file.set_len(off.saturating_add(span.to_u64()))
        .unwrap_or_else(|e| panic!("death harness: cannot grow the file: {e}"));
    file.write_all_at(now, off)
        .unwrap_or_else(|e| panic!("death harness: cannot copy the region: {e}"));
    // SAFETY: the caller owns these pages and supplied their copied contents; replace in place.
    let p = unsafe {
        sys::mmap(
            std::ptr::with_exposed_provenance_mut::<c_void>(start),
            span,
            sys::PROT_READ | sys::PROT_WRITE,
            sys::MAP_SHARED | sys::MAP_FIXED,
            file.as_raw_fd(),
            off.cast_signed(),
        )
    };
    assert_eq!(
        p.addr(),
        start,
        "death harness: cannot remap {start:#x}+{span:#x}"
    );
}

pub fn configure(_command: &mut Command) {}

pub fn wait(child: &mut Child, _file: &File, timeout: Duration) -> (ExitStatus, bool, Option<i32>) {
    let started = Instant::now();
    loop {
        if let Some(status) = child
            .try_wait()
            .unwrap_or_else(|e| panic!("death harness: cannot wait for the child: {e}"))
        {
            return (status, false, status.signal());
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            return (
                child
                    .wait()
                    .unwrap_or_else(|e| panic!("death harness: cannot reap the child: {e}")),
                true,
                None,
            );
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

pub fn termination(status: ExitStatus) -> Option<i32> {
    status.signal()
}

pub fn signal(status: ExitStatus) -> i32 {
    status.signal().unwrap_or(0)
}

pub fn aborts_with_line(code: i32) -> bool {
    code == ABORT
}

pub fn exit(status: i32) -> ! {
    sys::_exit(status)
}

#[cfg(test)]
pub fn trap() -> ! {
    sys::raise(sys::SIGTRAP);
    panic!("SIGTRAP returned")
}

pub fn read_region(file: &File, _path: &Path, off: u64, delta: u64, len: u64) -> Option<Vec<u8>> {
    let mut v = vec![0u8; usize::try_from(len).ok()?];
    file.read_exact_at(&mut v, off.checked_add(delta)?).ok()?;
    Some(v)
}

pub fn remove_region(_path: &Path, _off: u64) {}
