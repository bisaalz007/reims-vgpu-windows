//! The host's side of the contract: the values it hands over (`types`), its callbacks
//! (`callbacks`) and its dispatch queue (`queue`).
//!
//! These are the consumer's (QEMU's) callbacks and opaque values, not platform APIs: the
//! platform is what core calls on Apple's frameworks, the host is who called core. Every item
//! is at `crate::host::…`.

mod callbacks;
mod queue;
mod types;

pub use callbacks::{
    AddTraceRange, CreateTask, DestroyTask, HostBlocks, HostCallbacks, MapMemory, RaiseInterrupt,
    ReadMemory, RemoveTraceRange, TraceRangeHandler, UnmapMemory,
};
pub use queue::HostQueue;
pub use types::{HostPtr, HostTask, HostTraceRange, PhysRange, PhysicalMemoryRange};
