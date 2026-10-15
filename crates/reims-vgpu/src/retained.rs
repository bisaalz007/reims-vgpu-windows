//! The reference the public device, display, IOSurface host device, frame and frame target types
//! hold.
//!
//! Every public entry point runs inside [`entry::run`] (6.5). A release is
//! one too: the last one runs core's whole teardown (the device's stop, the root page cleared, the
//! host's `destroyTask`, the display's unplug, the IOSurface queue's `dispatch_sync`) on the
//! caller's thread, so [`Ours`] drops its handle inside an entry of its own: marked as ours, so a
//! Rust panic in that teardown aborts at its op (D9) instead of unwinding into the host, and with
//! what the teardown autoreleases released before the release returns. A drop has no caller's
//! hold: the teardown runs under [`Hold::NONE`](reims_vgpu_core::rt::Hold::NONE).

use std::mem::ManuallyDrop;
use std::ops::Deref;

use crate::entry;

/// A counted reference (one of core's handles) whose release runs as an entry of its own.
pub(crate) struct Ours<T>(ManuallyDrop<T>);

impl<T> Ours<T> {
    /// Takes over `handle`'s reference.
    pub(crate) fn new(handle: T) -> Ours<T> {
        Ours(ManuallyDrop::new(handle))
    }
}

impl<T> Deref for Ours<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> Drop for Ours<T> {
    /// A release; the last one is the object's teardown, run as an entry.
    fn drop(&mut self) {
        let handle = &mut self.0;
        entry::run(|_| {
            // SAFETY: `handle` is dropped exactly once, here, and never touched again.
            unsafe { ManuallyDrop::drop(handle) };
        });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU8, Ordering};

    use reims_vgpu_core::rt::{enter_ours, is_our_thread};

    use super::*;

    /// Records, when dropped, whether its thread was marked as ours (1) or not (2).
    struct Probe(Arc<AtomicU8>);

    impl Drop for Probe {
        fn drop(&mut self) {
            self.0
                .store(if is_our_thread() { 1 } else { 2 }, Ordering::SeqCst);
        }
    }

    /// The last release runs marked as ours (the teardown of the device, the display and the
    /// IOSurface host device); an earlier one drops nothing; the mark is restored
    #[test]
    fn last_release_is_ours() {
        let seen = Arc::new(AtomicU8::new(0));
        let a = Ours::new(Arc::new(Probe(seen.clone())));
        let b = Ours::new(Arc::clone(&a));
        assert!(Arc::ptr_eq(&a, &b));
        drop(a);
        assert_eq!(seen.load(Ordering::SeqCst), 0);
        assert!(!is_our_thread());
        drop(b);
        assert_eq!(seen.load(Ordering::SeqCst), 1);
        assert!(!is_our_thread());
        // Nested inside an entry point, the caller's mark survives.
        let c = Ours::new(Arc::new(Probe(seen.clone())));
        let outer = enter_ours();
        drop(c);
        assert!(is_our_thread());
        drop(outer);
        assert!(!is_our_thread());
    }
}
