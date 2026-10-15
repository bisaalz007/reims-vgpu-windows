//! Opaque handles.
//!
//! Every `Rvg<Name> *` the API hands out is `Arc::into_raw` of an `Arc<Rvg<Name>>`: one
//! reference per pointer a create or a `_retain` returns. A release drops one, and the release
//! that drops the last one ends the object, which may block. Releasing NULL does nothing.

use std::sync::Arc;

/// A new handle holding one reference.
pub(crate) fn new<T>(value: T) -> *mut T {
    Arc::into_raw(Arc::new(value)).cast_mut()
}

/// A new handle lent to a host's callback for one call: this frame owns its one reference and
/// gives it back when the frame ends. A host that keeps the handle retains its own.
pub(crate) struct Lent<T>(*mut T);

impl<T> Lent<T> {
    /// Lends a new handle over `value`.
    pub(crate) fn new(value: T) -> Lent<T> {
        Lent(new(value))
    }

    /// The handle, live until `self` drops.
    pub(crate) fn as_ptr(&self) -> *mut T {
        self.0
    }
}

impl<T> Drop for Lent<T> {
    fn drop(&mut self) {
        // SAFETY: the one reference `new` made, given back once.
        unsafe { release(self.0) };
    }
}

/// One more reference; returns `p` (NULL stays NULL).
///
/// # Safety
/// `p` is NULL or a live handle of type `T` from this crate.
pub(crate) unsafe fn retain<T>(p: *mut T) -> *mut T {
    if !p.is_null() {
        // SAFETY: the caller's contract: `p` came from `Arc::into_raw` and is live.
        unsafe { Arc::increment_strong_count(p.cast_const()) };
    }
    p
}

/// One reference less; the last runs the value's drop (NULL does nothing).
///
/// # Safety
/// `p` is NULL or a live handle of type `T` from this crate, whose reference the caller owns.
pub(crate) unsafe fn release<T>(p: *mut T) {
    if !p.is_null() {
        // SAFETY: the caller's contract: this gives back one reference `Arc::into_raw` made.
        drop(unsafe { Arc::from_raw(p.cast_const()) });
    }
}

/// The value behind a borrowed handle; `None` for NULL.
///
/// # Safety
/// `p` is NULL or a live handle of type `T` from this crate, live for `'a`.
pub(crate) unsafe fn get<'a, T>(p: *const T) -> Option<&'a T> {
    // SAFETY: the caller's contract.
    unsafe { p.as_ref() }
}

/// Writes `value` through `out` when `out` is not NULL (an optional out-parameter).
///
/// # Safety
/// `out` is NULL or valid for a write of `T`.
pub(crate) unsafe fn write_out<T>(out: *mut T, value: T) {
    if !out.is_null() {
        // SAFETY: the caller's contract.
        unsafe { out.write(value) };
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    static DROPS: AtomicUsize = AtomicUsize::new(0);

    struct Counted(u32);
    impl Drop for Counted {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// One reference per pointer, the last release drops, NULL does nothing
    #[test]
    fn refcount() {
        let before = DROPS.load(Ordering::SeqCst);
        let p = new(Counted(7));
        // SAFETY: `p` is live throughout; each release gives back one reference.
        unsafe {
            assert_eq!(get(p).unwrap().0, 7);
            assert_eq!(retain(p), p);
            release(p);
            assert_eq!(DROPS.load(Ordering::SeqCst), before);
            release(p);
        }
        assert_eq!(DROPS.load(Ordering::SeqCst), before + 1);
        let null: *mut Counted = std::ptr::null_mut();
        // SAFETY: NULL is accepted by each.
        unsafe {
            assert!(retain(null).is_null());
            release(null);
            assert!(get(null).is_none());
        }
        {
            let lent = Lent::new(Counted(1));
            // SAFETY: the lent handle is live in this block.
            assert_eq!(unsafe { get(lent.as_ptr()) }.unwrap().0, 1);
        }
        assert_eq!(DROPS.load(Ordering::SeqCst), before + 2);
    }
}
