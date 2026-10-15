//! [`StagedMutex`]: a mutex over fields some of which are also read, or written, without it.
//!
//! The device guards its registers and root objects with plain mutexes, yet reads several of
//! those fields without taking them: an MMIO read never waits behind a lock held across a host
//! callback, and a teardown reads the guest displays while another thread may hold the display
//! lock. Those fields live in atomics beside the mutex ([`Staged::Atomics`]), which the unlocked
//! readers read and write. A guard stages them:
//! - taking the lock pulls the atomics into the guarded copy;
//! - the holder reads and writes its copy only;
//! - the guard publishes each field it changed when it drops, or at [`StagedGuard::publish`]
//!   where a store must be seen before the lock is let go (a host call made under the lock
//!   reads the register it just wrote).
//!
//! So an unlocked reader sees either all of a guard's stores or none of them up to the publish
//! point, never one field new and another old (the root FIFO gone while `enabled` still reads
//! 1). Publishing store by store is exactly what this type exists to prevent.

use parking_lot::{Mutex, MutexGuard};

/// A lock's fields, some of which are also reached without the lock.
pub trait Staged {
    /// The fields reached without the lock, as atomics (or leaf cells held for one copy).
    type Atomics: Send + Sync;
    /// The staged fields as a guard found them, so it publishes only what it changed.
    type Snapshot: Copy;
    /// The atomics for a new lock over `self`.
    fn new_atomics(&self) -> Self::Atomics;
    /// Copies the atomics into the guarded fields (a guard is being taken).
    fn pull(&mut self, atomics: &Self::Atomics);
    /// The staged fields now.
    fn snapshot(&self) -> Self::Snapshot;
    /// Stores each staged field that differs from `since` into its atomic.
    fn publish(&self, atomics: &Self::Atomics, since: Self::Snapshot);
    /// The guard is going, after its last publish: drops what [`Staged::pull`] copied in for
    /// the holder alone (a reference whose owner is the atomic). Nothing by default.
    fn unlocking(&mut self) {}
}

/// A mutex over `T` whose staged fields are also readable without it (module doc).
pub struct StagedMutex<T: Staged> {
    inner: Mutex<T>,
    atomics: T::Atomics,
}

impl<T: Staged> StagedMutex<T> {
    /// A lock over `value`, its atomics made from it.
    pub fn new(value: T) -> Self {
        let atomics = value.new_atomics();
        StagedMutex {
            inner: Mutex::new(value),
            atomics,
        }
    }

    /// Takes the lock, staging the fields.
    pub fn lock(&self) -> StagedGuard<'_, T> {
        StagedGuard::new(self.inner.lock(), &self.atomics)
    }

    /// Takes the lock if it is free.
    pub fn try_lock(&self) -> Option<StagedGuard<'_, T>> {
        self.inner
            .try_lock()
            .map(|g| StagedGuard::new(g, &self.atomics))
    }

    /// The staged fields as readers without the lock see them.
    pub fn atomics(&self) -> &T::Atomics {
        &self.atomics
    }

    /// The guarded fields and the atomics, with no lock: the owner's teardown.
    pub fn get_mut(&mut self) -> (&mut T, &mut T::Atomics) {
        (self.inner.get_mut(), &mut self.atomics)
    }
}

/// A [`StagedMutex`] held.
pub struct StagedGuard<'a, T: Staged> {
    guard: MutexGuard<'a, T>,
    atomics: &'a T::Atomics,
    since: T::Snapshot,
}

impl<'a, T: Staged> StagedGuard<'a, T> {
    fn new(mut guard: MutexGuard<'a, T>, atomics: &'a T::Atomics) -> Self {
        guard.pull(atomics);
        let since = guard.snapshot();
        StagedGuard {
            guard,
            atomics,
            since,
        }
    }

    /// Makes the stores made so far visible to readers without the lock, now.
    pub fn publish(&mut self) {
        self.guard.publish(self.atomics, self.since);
        self.since = self.guard.snapshot();
    }
}

impl<T: Staged> std::ops::Deref for StagedGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T: Staged> std::ops::DerefMut for StagedGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}

impl<T: Staged> Drop for StagedGuard<'_, T> {
    fn drop(&mut self) {
        self.guard.publish(self.atomics, self.since);
        self.guard.unlocking();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Two fields readers see without the lock.
    struct Pair {
        a: u32,
        b: u32,
    }

    #[derive(Default)]
    struct PairAtomics {
        a: AtomicU32,
        b: AtomicU32,
    }

    impl Staged for Pair {
        type Atomics = PairAtomics;
        type Snapshot = (u32, u32);
        fn new_atomics(&self) -> PairAtomics {
            PairAtomics {
                a: AtomicU32::new(self.a),
                b: AtomicU32::new(self.b),
            }
        }
        fn pull(&mut self, at: &PairAtomics) {
            self.a = at.a.load(Ordering::Relaxed);
            self.b = at.b.load(Ordering::Relaxed);
        }
        fn snapshot(&self) -> (u32, u32) {
            (self.a, self.b)
        }
        fn publish(&self, at: &PairAtomics, since: (u32, u32)) {
            if self.a != since.0 {
                at.a.store(self.a, Ordering::Relaxed);
            }
            if self.b != since.1 {
                at.b.store(self.b, Ordering::Relaxed);
            }
        }
    }

    fn unlocked(m: &StagedMutex<Pair>) -> (u32, u32) {
        let at = m.atomics();
        (at.a.load(Ordering::Relaxed), at.b.load(Ordering::Relaxed))
    }

    /// The holder reads its own stores at once; readers without the lock see them only at the
    /// guard's drop, both fields together
    #[test]
    fn stores_publish_at_the_guard_drop_not_per_store() {
        let m = StagedMutex::new(Pair { a: 1, b: 2 });
        {
            let mut g = m.lock();
            g.a = 10;
            assert_eq!(g.a, 10);
            assert_eq!(unlocked(&m), (1, 2));
            g.b = 20;
            assert_eq!(unlocked(&m), (1, 2));
        }
        assert_eq!(unlocked(&m), (10, 20));
    }

    /// An explicit publish makes the stores so far visible before the lock is let go
    #[test]
    fn publish_makes_stores_visible_early() {
        let m = StagedMutex::new(Pair { a: 0, b: 0 });
        let mut g = m.lock();
        g.a = 5;
        g.publish();
        assert_eq!(unlocked(&m), (5, 0));
        g.b = 6;
        assert_eq!(unlocked(&m), (5, 0));
        drop(g);
        assert_eq!(unlocked(&m), (5, 6));
    }

    /// A guard pulls what an unlocked writer stored, and publishes back only the fields it
    /// changed, so it never undoes an unlocked store it did not see change
    #[test]
    fn a_guard_pulls_unlocked_stores_and_publishes_only_its_changes() {
        let m = StagedMutex::new(Pair { a: 0, b: 0 });
        m.atomics().a.store(7, Ordering::Relaxed);
        let mut g = m.lock();
        assert_eq!(g.a, 7);
        g.b = 1;
        // An unlocked store while the guard is held survives the guard's drop.
        m.atomics().a.store(8, Ordering::Relaxed);
        drop(g);
        assert_eq!(unlocked(&m), (8, 1));
    }
}
