//! A callback table's `ctx` and its one `destroy`.
//!
//! Every closure the C layer builds from a table holds one `Arc` of its [`Table`], and so does
//! the handle of the object the table was given to ([`WithTable`]), which lets its reference go
//! before the object. `destroy(ctx)` so runs when the last closure goes: for a table with
//! functions, at the end of the object, where core drops the last of its closures; for one with
//! none, at the host's release of the object. A create that refuses runs it before it returns
//! ([`Table::end`]), even where core frees a closure later on another thread (a refused
//! display's new-frame closure, which its source's cancel frees on the display thread): that
//! closure is never called.

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ffi;

/// `destroy(ctx)`: the last call of a callback table (see the preamble).
pub type RvgDestroyFn = Option<unsafe extern "C-unwind" fn(ctx: *mut c_void)>;

/// One table's `ctx` and `destroy`.
pub(crate) struct Table {
    ctx: *mut c_void,
    destroy: RvgDestroyFn,
    ended: AtomicBool,
}

// SAFETY: the header's contract: `ctx` is usable from the threads each callback names, and
// `destroy` from any thread.
unsafe impl Send for Table {}
// SAFETY: as above.
unsafe impl Sync for Table {}

impl Table {
    /// A table over the host's `ctx` and `destroy`.
    pub(crate) fn new(ctx: *mut c_void, destroy: RvgDestroyFn) -> Arc<Table> {
        Arc::new(Table {
            ctx,
            destroy,
            ended: AtomicBool::new(false),
        })
    }

    /// The host's `ctx`.
    pub(crate) fn ctx(&self) -> *mut c_void {
        self.ctx
    }

    /// Runs `destroy(ctx)` now, unless it has run.
    pub(crate) fn end(&self) {
        if self.ended.swap(true, Ordering::AcqRel) {
            return;
        }
        if let Some(destroy) = self.destroy {
            let ctx = self.ctx;
            // SAFETY: the host's destroy over its own ctx, once.
            ffi::host_call("destroy", || unsafe { destroy(ctx) });
        }
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        self.end();
    }
}

/// An object and its callback table, as a handle holds them.
pub(crate) struct WithTable<T> {
    object: ManuallyDrop<T>,
    table: ManuallyDrop<Arc<Table>>,
}

impl<T> WithTable<T> {
    /// `object`, made with `table`'s closures.
    pub(crate) fn new(object: T, table: Arc<Table>) -> WithTable<T> {
        WithTable {
            object: ManuallyDrop::new(object),
            table: ManuallyDrop::new(table),
        }
    }
}

impl<T> Deref for WithTable<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.object
    }
}

impl<T> Drop for WithTable<T> {
    /// The table first, then the object: the object's closures, dropped at its end, are then the
    /// table's last references (module doc).
    fn drop(&mut self) {
        // SAFETY: each field is dropped exactly once, here, and never touched again.
        unsafe {
            ManuallyDrop::drop(&mut self.table);
            ManuallyDrop::drop(&mut self.object);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use super::*;

    static DESTROYS: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C-unwind" fn destroy(ctx: *mut c_void) {
        assert_eq!(ctx.addr(), 0x5a);
        DESTROYS.fetch_add(1, Ordering::SeqCst);
    }

    /// Once at the last reference, or once at `end`, never twice; no destroy is nothing
    #[test]
    fn once() {
        let ctx = std::ptr::without_provenance_mut(0x5a);
        let before = DESTROYS.load(Ordering::SeqCst);
        let t = Table::new(ctx, Some(destroy));
        let t2 = t.clone();
        drop(t);
        assert_eq!(DESTROYS.load(Ordering::SeqCst), before);
        drop(t2);
        assert_eq!(DESTROYS.load(Ordering::SeqCst), before + 1);
        let t = Table::new(ctx, Some(destroy));
        let t2 = t.clone();
        t.end();
        assert_eq!(DESTROYS.load(Ordering::SeqCst), before + 2);
        drop((t, t2));
        assert_eq!(DESTROYS.load(Ordering::SeqCst), before + 2);
        drop(Table::new(ctx, None));
    }

    /// An object holding a closure's reference to a table: records the destroys seen at its drop.
    struct Holds(Option<Arc<Table>>, Arc<AtomicUsize>);
    impl Drop for Holds {
        fn drop(&mut self) {
            let before = DESTROYS.load(Ordering::SeqCst);
            self.1.store(before, Ordering::SeqCst);
            drop(self.0.take());
        }
    }

    /// The handle's reference goes first, so the destroy runs at the object's drop, after it
    #[test]
    fn with_table_lets_the_table_go_first() {
        let ctx = std::ptr::without_provenance_mut(0x5a);
        let t = Table::new(ctx, Some(destroy));
        let seen = Arc::new(AtomicUsize::new(usize::MAX));
        let before = DESTROYS.load(Ordering::SeqCst);
        let w = WithTable::new(Holds(Some(t.clone()), seen.clone()), t);
        drop(w);
        assert_eq!(seen.load(Ordering::SeqCst), before);
        assert_eq!(DESTROYS.load(Ordering::SeqCst), before + 1);
    }
}
