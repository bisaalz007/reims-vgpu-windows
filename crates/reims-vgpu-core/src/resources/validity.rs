//! The validity model: the three flags every resource carries, and what each kind derives from
//! them.
//!
//! The flags are plain stores and loads with no ordering beyond the caller's locks: the FIFO and
//! Metal's completion threads read and write them unlocked. The derived answers are
//! [`ResourceClass`](super::resource::ResourceClass)'s defaults:
//! - needs sync: guest-invalid (asked first) and host-valid, each through the kind's own answer;
//! - needs prepare: host-invalid through the kind's answer, or the raw volatile flag;
//! - an eviction writes back when the raw flags (not the kind's answers) say guest-invalid and
//!   host-valid.
//!
//! A kind that derives host validity from its handle (the shared, mapper-ref and detach
//! buffers, the detach heap) leaves the host flag as dead state its setter still writes.

use std::sync::atomic::{AtomicBool, Ordering};

/// A resource's three flags: host-valid, guest-valid and volatile. All start false but
/// guest-valid, which each kind's initial answer sets.
#[derive(Debug, Default)]
pub struct ResourceState {
    host_valid: AtomicBool,
    guest_valid: AtomicBool,
    is_volatile: AtomicBool,
}

impl ResourceState {
    /// The raw host-valid flag (what an eviction and the paging methods read).
    pub fn host_valid(&self) -> bool {
        self.host_valid.load(Ordering::Relaxed)
    }
    /// Stores the host-valid flag.
    pub fn set_host_valid(&self, v: bool) {
        self.host_valid.store(v, Ordering::Relaxed)
    }
    /// The raw guest-valid flag.
    pub fn guest_valid(&self) -> bool {
        self.guest_valid.load(Ordering::Relaxed)
    }
    /// Stores the guest-valid flag.
    pub fn set_guest_valid(&self, v: bool) {
        self.guest_valid.store(v, Ordering::Relaxed)
    }
    /// The raw volatile flag: set by a discard that made the Metal object purgeable.
    pub fn is_volatile(&self) -> bool {
        self.is_volatile.load(Ordering::Relaxed)
    }
    /// Stores the volatile flag.
    pub fn set_is_volatile(&self, v: bool) {
        self.is_volatile.store(v, Ordering::Relaxed)
    }
}
