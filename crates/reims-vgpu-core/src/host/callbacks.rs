//! The host's callbacks: the eight blocks of the device descriptor ([`HostBlocks`]) and the
//! device's copies of them ([`HostCallbacks`]).
//!
//! Every block is optional, and its absence is behaviour. `readMemory`, `addTraceRange` and
//! `removeTraceRange` are checked before a call: with no `readMemory` a physical read maps the
//! page through the root task, and with no trace range the EFI display presents on its timer
//! (`addTraceRange` without `removeTraceRange` refuses the device). The device calls the others
//! without a nil check, and a missing one is the device's `NilBlock` death at the call.
//!
//! The device's copies release the blocks in the order the host sees their `destroy` callbacks:
//! createTask, destroyTask, mapMemory, unmapMemory, readMemory, raiseInterrupt, addTraceRange,
//! removeTraceRange (`tests::the_copies_release_in_order`, and
//! the device's `tests::the_last_release_releases_the_blocks_in_order`). The descriptor's own
//! blocks release in [`HostBlocks`]' field order, which is where a refused device leaves the
//! last references.

use std::sync::Arc;

use crate::host::{HostPtr, HostTask, HostTraceRange, PhysicalMemoryRange};

/// The block the host calls with a dirtied range (the EFI display's present source). The host
/// may call it from any thread.
pub type TraceRangeHandler = Arc<dyn Fn(PhysicalMemoryRange) + Send + Sync>;

/// `raiseInterrupt(vector)`. Runs under the device's signal mutex: it must not re-enter.
pub type RaiseInterrupt = Arc<dyn Fn(u32) + Send + Sync>;

// A host block that raises ends the process at the call: the C ABI's and the testkit's
// callers die there with `HostRaised`, so every block answers its value and nothing else.

/// `createTask(vmSize, &baseAddress)`: `NULL` on failure; writes the base in place.
pub type CreateTask = Arc<dyn Fn(u64, &mut HostPtr) -> HostTask + Send + Sync>;
/// `destroyTask(task)`.
pub type DestroyTask = Arc<dyn Fn(HostTask) + Send + Sync>;
/// `mapMemory(task, rangeCount, virtualOffset, readOnly, ranges)`.
pub type MapMemory =
    Arc<dyn Fn(HostTask, u32, u64, bool, &[PhysicalMemoryRange]) -> bool + Send + Sync>;
/// `unmapMemory(task, virtualOffset, length)`.
pub type UnmapMemory = Arc<dyn Fn(HostTask, u64, u64) -> bool + Send + Sync>;
/// `readMemory(physicalAddress, length, dst)`.
pub type ReadMemory = Arc<dyn Fn(u64, u64, HostPtr) -> bool + Send + Sync>;
/// `addTraceRange(range, handler)`: `NULL` when the host installs none.
pub type AddTraceRange =
    Arc<dyn Fn(PhysicalMemoryRange, TraceRangeHandler) -> HostTraceRange + Send + Sync>;
/// `removeTraceRange(range)`.
pub type RemoveTraceRange = Arc<dyn Fn(HostTraceRange) + Send + Sync>;

/// The `PGDeviceDescriptor` host blocks, as the host hands them over. `None` is a nil block.
/// Its fields drop in declaration order.
#[derive(Clone, Default)]
pub struct HostBlocks {
    /// `raiseInterrupt`.
    pub raise_interrupt: Option<RaiseInterrupt>,
    /// `createTask`.
    pub create_task: Option<CreateTask>,
    /// `destroyTask`.
    pub destroy_task: Option<DestroyTask>,
    /// `mapMemory`.
    pub map_memory: Option<MapMemory>,
    /// `unmapMemory`.
    pub unmap_memory: Option<UnmapMemory>,
    /// `readMemory`.
    pub read_memory: Option<ReadMemory>,
    /// `addTraceRange`.
    pub add_trace_range: Option<AddTraceRange>,
    /// `removeTraceRange`.
    pub remove_trace_range: Option<RemoveTraceRange>,
}

impl std::fmt::Debug for HostBlocks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostBlocks")
            .field("raise_interrupt", &self.raise_interrupt.is_some())
            .field("create_task", &self.create_task.is_some())
            .field("destroy_task", &self.destroy_task.is_some())
            .field("map_memory", &self.map_memory.is_some())
            .field("unmap_memory", &self.unmap_memory.is_some())
            .field("read_memory", &self.read_memory.is_some())
            .field("add_trace_range", &self.add_trace_range.is_some())
            .field("remove_trace_range", &self.remove_trace_range.is_some())
            .finish()
    }
}

/// The device's copies of the host's blocks. The device fills them in at creation (the memory
/// blocks only without a memory map, the trace-range pair only when `addTraceRange` is given)
/// and keeps two references: its own and its mapping sink's, which is the last. When the host
/// has released its descriptor, the drop of the last copy is each block's last release, in the
/// module doc's order.
#[derive(Default)]
pub struct HostCallbacks {
    /// `createTask`.
    pub(crate) create_task: Option<CreateTask>,
    /// `destroyTask`.
    pub(crate) destroy_task: Option<DestroyTask>,
    /// `mapMemory`.
    pub(crate) map_memory: Option<MapMemory>,
    /// `unmapMemory`.
    pub(crate) unmap_memory: Option<UnmapMemory>,
    /// `readMemory`.
    pub(crate) read_memory: Option<ReadMemory>,
    /// `raiseInterrupt`.
    pub(crate) raise_interrupt: Option<RaiseInterrupt>,
    /// `addTraceRange`.
    pub(crate) add_trace_range: Option<AddTraceRange>,
    /// `removeTraceRange`.
    pub(crate) remove_trace_range: Option<RemoveTraceRange>,
}

impl HostCallbacks {
    /// Another reference to each of the same blocks.
    pub fn clone_ref(&self) -> HostCallbacks {
        HostCallbacks {
            create_task: self.create_task.clone(),
            destroy_task: self.destroy_task.clone(),
            map_memory: self.map_memory.clone(),
            unmap_memory: self.unmap_memory.clone(),
            read_memory: self.read_memory.clone(),
            raise_interrupt: self.raise_interrupt.clone(),
            add_trace_range: self.add_trace_range.clone(),
            remove_trace_range: self.remove_trace_range.clone(),
        }
    }
}

impl std::fmt::Debug for HostCallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostCallbacks")
            .field("create_task", &self.create_task.is_some())
            .field("destroy_task", &self.destroy_task.is_some())
            .field("map_memory", &self.map_memory.is_some())
            .field("unmap_memory", &self.unmap_memory.is_some())
            .field("read_memory", &self.read_memory.is_some())
            .field("raise_interrupt", &self.raise_interrupt.is_some())
            .field("add_trace_range", &self.add_trace_range.is_some())
            .field("remove_trace_range", &self.remove_trace_range.is_some())
            .finish()
    }
}

impl Drop for HostCallbacks {
    /// The blocks in the host's `destroy` order (module doc).
    fn drop(&mut self) {
        drop(self.create_task.take());
        drop(self.destroy_task.take());
        drop(self.map_memory.take());
        drop(self.unmap_memory.take());
        drop(self.read_memory.take());
        drop(self.raise_interrupt.take());
        drop(self.add_trace_range.take());
        drop(self.remove_trace_range.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A block's captured context: records its name when the block's last reference goes (the
    /// host's `destroy` at the last release).
    struct Tag(Arc<Mutex<Vec<&'static str>>>, &'static str);

    impl Drop for Tag {
        fn drop(&mut self) {
            self.0.lock().push(self.1);
        }
    }

    /// All eight blocks, each capturing a [`Tag`], declared in `HostBlocks`' field order.
    fn tagged(order: &Arc<Mutex<Vec<&'static str>>>) -> HostCallbacks {
        let tag = |name| Tag(order.clone(), name);
        let (raise, create, destroy) =
            (tag("raiseInterrupt"), tag("createTask"), tag("destroyTask"));
        let (map, unmap, read) = (tag("mapMemory"), tag("unmapMemory"), tag("readMemory"));
        let (add, remove) = (tag("addTraceRange"), tag("removeTraceRange"));
        HostCallbacks {
            raise_interrupt: Some(Arc::new(move |_| {
                let _ = &raise;
            })),
            create_task: Some(Arc::new(move |_, _| {
                let _ = &create;
                HostTask::NULL
            })),
            destroy_task: Some(Arc::new(move |_| {
                let _ = &destroy;
            })),
            map_memory: Some(Arc::new(move |_, _, _, _, _| {
                let _ = &map;
                false
            })),
            unmap_memory: Some(Arc::new(move |_, _, _| {
                let _ = &unmap;
                false
            })),
            read_memory: Some(Arc::new(move |_, _, _| {
                let _ = &read;
                false
            })),
            add_trace_range: Some(Arc::new(move |_, _| {
                let _ = &add;
                HostTraceRange::NULL
            })),
            remove_trace_range: Some(Arc::new(move |_| {
                let _ = &remove;
            })),
        }
    }

    /// The last copy releases createTask, destroyTask, mapMemory, unmapMemory, readMemory,
    /// raiseInterrupt, addTraceRange, removeTraceRange; an earlier copy's drop releases nothing
    #[test]
    fn the_copies_release_in_order() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let copies = tagged(&order);
        let other = copies.clone_ref();
        drop(copies);
        assert!(order.lock().is_empty());
        drop(other);
        assert_eq!(
            *order.lock(),
            [
                "createTask",
                "destroyTask",
                "mapMemory",
                "unmapMemory",
                "readMemory",
                "raiseInterrupt",
                "addTraceRange",
                "removeTraceRange",
            ]
        );
    }

    /// Blocks are optional and callable; Debug shows presence only
    #[test]
    fn blocks() {
        let n = Arc::new(AtomicU32::new(0));
        let n2 = n.clone();
        let b = HostBlocks {
            raise_interrupt: Some(Arc::new(move |v| {
                n2.store(v, Ordering::SeqCst);
            })),
            ..HostBlocks::default()
        };
        if let Some(f) = &b.raise_interrupt {
            f(3);
        }
        assert_eq!(n.load(Ordering::SeqCst), 3);
        let s = format!("{b:?}");
        assert!(s.contains("raise_interrupt: true"));
        assert!(s.contains("create_task: false"));
        let mut c = HostCallbacks::default();
        c.raise_interrupt = b.raise_interrupt.clone();
        assert!(format!("{c:?}").contains("raise_interrupt: true"));
    }
}
