//! The host's opaque values: addresses, tasks, trace ranges and physical ranges.
//!
//! Core never dereferences a host address itself: `crate::guest` turns one back into a pointer
//! and owns every access.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// A host virtual address (`void *`): a task base, a no-copy buffer, `readMemory`'s
/// destination. Carries no provenance of its own: `crate::guest` turns it back into a pointer
/// (`with_exposed_provenance`). NULL is representable: the host passes it and core tests it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct HostPtr(usize);

impl HostPtr {
    /// `NULL`.
    pub const NULL: HostPtr = HostPtr(0);

    /// From an address.
    pub const fn from_addr(addr: usize) -> HostPtr {
        HostPtr(addr)
    }

    /// From a pointer, exposing its provenance (the metal and C crates' side).
    pub fn from_ptr<T>(p: *mut T) -> HostPtr {
        HostPtr(p.expose_provenance())
    }

    /// Back to a pointer with the exposed provenance. Creating it is safe; dereferencing it
    /// is `crate::guest`'s job.
    pub fn as_ptr<T>(self) -> *mut T {
        std::ptr::with_exposed_provenance_mut(self.0)
    }

    /// The address.
    pub const fn addr(self) -> usize {
        self.0
    }

    /// Whether it is `NULL`.
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    /// `self + off`, wrapping.
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "lossless on the 64-bit target, and `ToUsize` cannot be called in a const fn"
    )]
    pub const fn wrapping_add(self, off: u64) -> HostPtr {
        HostPtr(self.0.wrapping_add(off as usize))
    }
}

/// One range of guest physical memory, `{physicalAddress, physicalLength}`: the C API's
/// `RvgPhysicalMemoryRange`, which has its layout.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct PhysicalMemoryRange {
    /// `physicalAddress`.
    pub physical_address: u64,
    /// `physicalLength`.
    pub physical_length: u64,
}

/// The short name of [`PhysicalMemoryRange`].
pub type PhysRange = PhysicalMemoryRange;

const _: () = assert!(size_of::<PhysicalMemoryRange>() == 16);

/// `PGTask_t *`: the host's opaque task. NULL is representable: `createTask` answers NULL on
/// failure, and the device hands a NULL task to `destroyTask` after a failed root map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct HostTask(usize);

impl HostTask {
    /// `NULL`.
    pub const NULL: HostTask = HostTask(0);
    /// From the host's pointer value.
    pub const fn from_addr(addr: usize) -> HostTask {
        HostTask(addr)
    }
    /// The host's pointer value.
    pub const fn addr(self) -> usize {
        self.0
    }
    /// Whether it is `NULL`.
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }
}

/// `PGTraceRange_t *`: the host's opaque trace range; NULL when the host installs none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct HostTraceRange(usize);

impl HostTraceRange {
    /// `NULL`.
    pub const NULL: HostTraceRange = HostTraceRange(0);
    /// From the host's pointer value.
    pub const fn from_addr(addr: usize) -> HostTraceRange {
        HostTraceRange(addr)
    }
    /// The host's pointer value.
    pub const fn addr(self) -> usize {
        self.0
    }
    /// Whether it is `NULL`.
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// HostPtr round-trips and wraps
    #[test]
    fn host_ptr() {
        let mut x = 5u32;
        let raw = std::ptr::from_mut(&mut x);
        let p = HostPtr::from_ptr(raw);
        assert!(!p.is_null());
        assert_eq!(p.as_ptr::<u32>(), raw);
        assert!(HostPtr::NULL.is_null());
        assert_eq!(HostPtr::from_addr(usize::MAX).wrapping_add(2).addr(), 1);
        assert_eq!(HostPtr::default(), HostPtr::NULL);
    }

    /// The SDK layout of PGPhysicalMemoryRange_t
    #[test]
    fn phys_range_layout() {
        let bytes = [1u8, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0];
        let r = PhysicalMemoryRange::read_from_bytes(&bytes).ok();
        assert_eq!(
            r,
            Some(PhysRange {
                physical_address: 1,
                physical_length: 2
            })
        );
    }

    /// Opaque handles keep NULL
    #[test]
    fn opaque_handles() {
        assert!(HostTask::NULL.is_null());
        assert_eq!(HostTask::from_addr(8).addr(), 8);
        assert!(!HostTask::from_addr(8).is_null());
        assert!(HostTraceRange::NULL.is_null());
        assert_eq!(HostTraceRange::from_addr(9).addr(), 9);
        assert!(!HostTraceRange::from_addr(9).is_null());
    }
}
