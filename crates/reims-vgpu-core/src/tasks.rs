//! Guest tasks: the address spaces the guest's commands run in.
//!
//! A task is local (task 0 everywhere, and every task in the blocks and map setups) or remote (a
//! user task under isolation, run by its own service on threads of its own, [`isolation`]).
//! [`Task`] is the device's handle on one. Both kinds, and the service's own task, share the parts in
//! [`shared`]: the resource manager, the object lists ([`object_lists`]) and the deserializer,
//! with the operations on them the guest's commands reach. A command completes through a
//! [`Completion`] with an [`ExecStatus`].
//!
//! Ordering contracts (each named test pins it):
//! - **The task root's list fields** take the new values before any validation, and a refused
//!   or unmappable list zeroes them again (`object_lists` tests
//!   `end_at_task_length_refused_and_zeroed`, `old_list_unmappable_fails_after_setting_members`,
//!   `null_root_faults`).
//! - **List validation** is one `||` in a fixed order: overflow, the task's end (`>=`), the
//!   previous offsets' alignment, then the placement list's size (`overflow_refused`,
//!   `previous_offsets_alignment`, `required_placement_length`).
//! - **List bounds**: an entry index is checked against the object list's count, for the
//!   placement list too (`mapped_reads_live_and_checks_range`,
//!   `mapped_gives_address_bounded_by_object_count`).
//! - **Synchronize** makes one paging buffer even with nothing to encode: CPU paging first,
//!   then one blit and one compute encoder, the completion handler, the commit and the
//!   scheduling wait, then the discards, all under the task's unique lock
//!   (`shared` tests `an_empty_sync_still_commits_a_buffer`,
//!   `discard_runs_after_the_commit_and_the_completion_may_precede_it`).
//! - **Discard, invalidate and delete** walk their records in order; a raise leaves the later
//!   records untouched (`discards_or_deletes_in_record_order`, `flags_apply_in_order`).
//! - **Teardown** releases the deserializer, then the resource manager, then the object lists;
//!   a later use dies at the op (`later_uses_die_at_the_op`).
//!
//! [`Task`]'s commands answer `RaiseResult`. A remote task's own commands answer `()`: the work
//! they forward runs in the isolated task, and no raise comes back from it. The area raises its
//! [`RaiseKind`]s, and its handlers are the frames that take a raise's `Err`.

pub mod isolation;
pub mod launcher;
pub mod local;
pub mod mapping_task;
pub mod object_lists;
pub mod remote;
pub mod resource_list;
pub mod shared;
pub mod wire;

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::fifo::wire::Records;
use crate::guest::records::{
    CmdExecIndirectCmdBuffer, CmdExecIndirectResource, CmdExecIndirectResource3,
};
use crate::host::{HostPtr, HostTask};
use crate::platform::Platform;
use crate::platform::types::NsErrorText;
use crate::raise::RaiseId;
use crate::raise::RaiseResult;
use crate::raise::death::DeathSite;
use crate::raise::{OnFault, Raise};
use crate::rt::Hold;
use crate::tasks::local::LocalTask;
use crate::tasks::remote::RemoteTask;

/// What the device's `createTask` gives a task: the host's task and the base of its address
/// space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TaskInfo {
    /// The host's task: the `createTask` block's, or a `MappingTask`'s identity.
    pub task: HostTask,
    /// The base of the task's address space in the host.
    pub base_address: HostPtr,
}

/// A guest task as the device's task table holds it. Each variant is an `Arc`, so a clone is
/// another reference to the same task (the lookups hand out counted references).
pub enum Task<P: Platform> {
    /// In process: task 0 everywhere, every task in the blocks and map setups.
    Local(Arc<LocalTask<P>>),
    /// A user task under isolation, run by its own service.
    Remote(Arc<RemoteTask<P>>),
}

impl<P: Platform> Clone for Task<P> {
    fn clone(&self) -> Self {
        match self {
            Task::Local(t) => Task::Local(Arc::clone(t)),
            Task::Remote(t) => Task::Remote(Arc::clone(t)),
        }
    }
}

impl<P: Platform> fmt::Debug for Task<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Task::Local(_) => f.write_str("Task::Local"),
            Task::Remote(_) => f.write_str("Task::Remote"),
        }
    }
}

/// `$call` with `$t` bound to whichever task `$task` holds.
macro_rules! each {
    ($task:expr, $t:ident => $call:expr) => {
        match $task {
            Task::Local($t) => $call,
            Task::Remote($t) => $call,
        }
    };
}

/// The commands the device and the FIFO send a task. Each completes through its [`Completion`]
/// (see its contract): a command that raises before completing returns `Err` and does not call
/// it, and the FIFO then faults at the stamp value, where a completion whose status is not 1
/// faults at the offset.
impl<P: Platform> Task<P> {
    /// Tears the task down: the device's task delete (opcode 32) and its teardown of every task.
    pub fn teardown(&self) {
        each!(self, t => t.teardown())
    }

    /// Deletes the object a serialized delete names (opcode 40; `data` is the payload after its
    /// first word).
    pub fn delete_object_with_serialized_data(
        &self,
        hold: Hold<'_>,
        data: &[u8],
        done: Completion,
    ) -> RaiseResult<()> {
        match self {
            Task::Local(t) => t.delete_object_with_serialized_data(hold, data, done),
            Task::Remote(t) => {
                t.delete_object_with_serialized_data(data, done);
                Ok(())
            }
        }
    }

    /// Deletes a resource (opcodes 37 and 65).
    pub fn delete_resource(
        &self,
        hold: Hold<'_>,
        resource: u32,
        done: Completion,
    ) -> RaiseResult<()> {
        match self {
            Task::Local(t) => t.delete_resource(hold, resource, done),
            Task::Remote(t) => {
                t.delete_resource(resource, done);
                Ok(())
            }
        }
    }

    /// Discards resources (opcode 63).
    pub fn discard_resources(
        &self,
        hold: Hold<'_>,
        resources: Records<'_, ResourceReference>,
        done: Completion,
    ) -> RaiseResult<()> {
        match self {
            Task::Local(t) => t.discard_resources(hold, resources, done),
            Task::Remote(t) => {
                t.discard_resources(resources, done);
                Ok(())
            }
        }
    }

    /// A FIFO was deleted: the device tells every task.
    pub fn fifo_deleted(&self, stamp_index: u32, done: Completion) -> RaiseResult<()> {
        match self {
            Task::Local(t) => t.fifo_deleted(stamp_index, done),
            Task::Remote(t) => {
                t.fifo_deleted(stamp_index, done);
                Ok(())
            }
        }
    }

    /// Flushes the resources into guest memory: the device's suspend.
    pub fn flush_resources(&self, hold: Hold<'_>) -> RaiseResult<()> {
        match self {
            Task::Local(t) => t.flush_resources(hold),
            Task::Remote(t) => {
                t.flush_resources();
                Ok(())
            }
        }
    }

    /// Applies validity flags to resources (opcode 52).
    pub fn invalidate_resources(
        &self,
        resources: Records<'_, ResourceInvalidation>,
        done: Completion,
    ) -> RaiseResult<()> {
        match self {
            Task::Local(t) => t.invalidate_resources(resources, done),
            Task::Remote(t) => {
                t.invalidate_resources(resources, done);
                Ok(())
            }
        }
    }

    /// Maps task memory (opcode 57).
    pub fn map_memory_at_offset(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
        done: Completion,
    ) -> RaiseResult<()> {
        match self {
            Task::Local(t) => t.map_memory_at_offset(hold, offset, length, done),
            Task::Remote(t) => {
                t.map_memory_at_offset(offset, length, done);
                Ok(())
            }
        }
    }

    /// Discards the resource over replaced physical memory (opcode 60).
    pub fn replace_physical(&self, physical: u32, done: Completion) -> RaiseResult<()> {
        match self {
            Task::Local(t) => t.replace_physical(physical, done),
            Task::Remote(t) => {
                t.replace_physical(physical, done);
                Ok(())
            }
        }
    }

    /// Sets the object and placement lists (opcodes 51, with no placement list, and 68).
    pub fn set_object_list_offset(
        &self,
        hold: Hold<'_>,
        object_list_offset: u64,
        object_list_length: u32,
        placement_list_offset: u64,
        placement_list_length: u32,
        done: Completion,
    ) -> RaiseResult<()> {
        match self {
            Task::Local(t) => t.set_object_list_offset(
                hold,
                object_list_offset,
                object_list_length,
                placement_list_offset,
                placement_list_length,
                done,
            ),
            Task::Remote(t) => {
                t.set_object_list_offset(
                    object_list_offset,
                    object_list_length,
                    placement_list_offset,
                    placement_list_length,
                    done,
                );
                Ok(())
            }
        }
    }

    /// Synchronizes resources into guest memory, and with `discard` discards them (opcodes 53
    /// and 62).
    pub fn synchronize_resources(
        &self,
        hold: Hold<'_>,
        resources: Records<'_, ResourceReference>,
        discard: bool,
        done: Completion,
    ) -> RaiseResult<()> {
        match self {
            Task::Local(t) => t.synchronize_resources(hold, resources, discard, done),
            Task::Remote(t) => {
                t.synchronize_resources(resources, discard, done);
                Ok(())
            }
        }
    }

    /// Unmaps task memory (opcode 34).
    pub fn unmap_memory_at_offset(
        &self,
        offset: u64,
        length: u64,
        done: Completion,
    ) -> RaiseResult<()> {
        match self {
            Task::Local(t) => t.unmap_memory_at_offset(offset, length, done),
            Task::Remote(t) => {
                t.unmap_memory_at_offset(offset, length, done);
                Ok(())
            }
        }
    }

    /// Executes command buffers with 24-byte resource records (opcode 55). The FIFO has no
    /// handler around it: a raise escaping it is a death.
    pub fn exec_indirect2(
        &self,
        hold: Hold<'_>,
        args: ExecArgs<'_, P, CmdExecIndirectResource>,
        done: Completion,
    ) -> RaiseResult<()> {
        match self {
            Task::Local(t) => t.exec_indirect2(hold, args, done),
            Task::Remote(t) => {
                t.exec_indirect2(hold, args, done);
                Ok(())
            }
        }
    }

    /// Executes command buffers with 8-byte resource records (opcode 67). No handler around it.
    pub fn exec_indirect3(
        &self,
        hold: Hold<'_>,
        args: ExecArgs<'_, P, CmdExecIndirectResource3>,
        done: Completion,
    ) -> RaiseResult<()> {
        match self {
            Task::Local(t) => t.exec_indirect3(hold, args, done),
            Task::Remote(t) => {
                t.exec_indirect3(hold, args, done);
                Ok(())
            }
        }
    }

    /// Writes a compute pipeline's info into task memory (opcode 59).
    pub fn get_compute_info(
        &self,
        hold: Hold<'_>,
        pipeline_state_reference: u32,
        max_key: u32,
        count: u32,
        offset: u64,
        done: Completion,
    ) -> RaiseResult<()> {
        match self {
            Task::Local(t) => {
                t.get_compute_info(hold, pipeline_state_reference, max_key, count, offset, done)
            }
            Task::Remote(t) => {
                t.get_compute_info(pipeline_state_reference, max_key, count, offset, done);
                Ok(())
            }
        }
    }

    /// Writes a heap texture's size and alignment into task memory (opcode 64).
    pub fn heap_texture_size_and_align(
        &self,
        serializer_payload: &[u8],
        reply_virtual_offset: u64,
        reply_length: u64,
        done: Completion,
    ) -> RaiseResult<()> {
        match self {
            Task::Local(t) => t.heap_texture_size_and_align(
                serializer_payload,
                reply_virtual_offset,
                reply_length,
                done,
            ),
            Task::Remote(t) => {
                t.heap_texture_size_and_align(
                    serializer_payload,
                    reply_virtual_offset,
                    reply_length,
                    done,
                );
                Ok(())
            }
        }
    }

    /// Writes the info of command buffers into task memory (opcode 69). No handler around it.
    pub fn info_indirect(
        &self,
        hold: Hold<'_>,
        args: ExecArgs<'_, P, CmdExecIndirectResource3>,
        done: Completion,
    ) -> RaiseResult<()> {
        match self {
            Task::Local(t) => t.info_indirect(hold, args, done),
            Task::Remote(t) => {
                t.info_indirect(args, done);
                Ok(())
            }
        }
    }

    /// Resets a rasterization rate map (opcode 66).
    pub fn reset_rasterization_rate_map(&self, map: u32, done: Completion) -> RaiseResult<()> {
        match self {
            Task::Local(t) => t.reset_rasterization_rate_map(map, done),
            Task::Remote(t) => {
                t.reset_rasterization_rate_map(map, done);
                Ok(())
            }
        }
    }
}

/// The 8-byte records of an invalidation (opcode 52). The names are the SDK header's.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
#[allow(missing_docs)] // the SDK header's field names
pub struct ResourceInvalidation {
    pub reference: u32,
    pub clear_host_valid: u8,
    pub set_host_valid: u8,
    pub clear_guest_valid: u8,
    pub set_guest_valid: u8,
}

const _: () = assert!(size_of::<ResourceInvalidation>() == 8);

/// The 4-byte records of a synchronize or a discard (opcodes 53, 62, 63).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct ResourceReference {
    /// The resource's reference.
    pub reference: u32,
}

const _: () = assert!(size_of::<ResourceReference>() == 4);

/// The arguments of an exec (opcodes 55 and 67) or an info exec (69), less the completion. `R`
/// is the resource record: `CmdExecIndirectResource` (55) or `CmdExecIndirectResource3` (67, 69).
pub struct ExecArgs<'a, P: Platform, R> {
    /// The command-buffer records.
    pub cmd_buffers: Records<'a, CmdExecIndirectCmdBuffer>,
    /// The resource records.
    pub resources: Records<'a, R>,
    /// The FIFO's stamp index.
    pub stamp_index: u32,
    /// The FIFO's own shared event, `None` for nil: the FIFO makes it unchecked, so a nil event
    /// reaches the task, whose sends to it do nothing. A task that keeps it (the remote task's
    /// last FIFO event) takes another reference.
    ///
    /// The FIFO clones the handle out of its leaf lock before the call: no guard is held across
    /// an exec, which can block (the remote exec's wait for its GPU token).
    pub fifo_event: Option<&'a P::SharedEvent>,
    /// The FIFO's event value, which the task reads and advances, possibly later from another
    /// thread (the remote reply). Shared, so the task clones the `Arc` to keep it.
    pub fifo_event_value: &'a Arc<AtomicU64>,
}

impl<P: Platform, R> fmt::Debug for ExecArgs<'_, P, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecArgs")
            .field("cmd_buffers", &self.cmd_buffers.count())
            .field("resources", &self.resources.count())
            .field("stamp_index", &self.stamp_index)
            .finish_non_exhaustive()
    }
}

/// How a task command ended, as its completion reports it: 1 is success, and 2, 3 and 4 are
/// the failures the tasks produce.
///
/// A newtype, not an `enum`: the service's reply carries any `u32` across the wire unvalidated,
/// and every value stays representable (0, "not complete", hangs a waiter). The constants'
/// names are ours.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ExecStatus(pub u32);

impl ExecStatus {
    /// 0: a token not yet signalled.
    pub const NOT_COMPLETE: ExecStatus = ExecStatus(0);
    /// 1: success. The FIFO signals the stamp.
    pub const OK: ExecStatus = ExecStatus(1);
    /// 2: the task refused or failed the command (the FIFO faults at the offset).
    pub const FAIL: ExecStatus = ExecStatus(2);
    /// 3: a GPU or command-buffer error, or a timeout.
    pub const GPU_OR_TIMEOUT: ExecStatus = ExecStatus(3);
    /// 4: the isolated task died.
    pub const CRASHED: ExecStatus = ExecStatus(4);

    /// The raw value.
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Whether this is success: the only test the FIFO makes.
    pub const fn is_ok(self) -> bool {
        self.0 == 1
    }
}

/// A task command's completion: its status and, for a GPU failure, the command buffer's error
/// (the FIFO never reads it).
///
/// Called once on every normal path, synchronously inside the call or later on any thread. It
/// can be called twice (the stamp then asserts, a host death) or never (the stamp
/// stalls), so it is `Fn`, not `FnOnce`. A command that raises before completing does not call
/// it: the FIFO faults at the stamp value then, which differs from a completion whose status is
/// not 1 (the FIFO faults at the offset).
pub type Completion = Arc<dyn Fn(ExecStatus, Option<NsErrorText>) + Send + Sync>;

/// This area's raises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaiseKind {
    /// The resource manager could not make the resource a reference names.
    ResourceLookup,
    /// A guest read longer than the largest task (32 GiB).
    BufferLength,
    /// A resource index at or past an exec's resource count.
    ResourceIndex,
    /// The local task's.
    Local(local::RaiseKind),
}

impl RaiseKind {
    /// The raise's dotted id.
    pub const fn id(self) -> &'static str {
        match self {
            RaiseKind::ResourceLookup => "tasks.resource.lookup",
            RaiseKind::BufferLength => "tasks.buffer.length",
            RaiseKind::ResourceIndex => "tasks.resource.index",
            RaiseKind::Local(k) => k.id(),
        }
    }
}

impl fmt::Display for RaiseKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl From<RaiseKind> for RaiseId {
    fn from(k: RaiseKind) -> RaiseId {
        RaiseId::Tasks(k)
    }
}

/// An operation the shared task core leaves to the task kinds, which a kind that does not
/// provide it reaches: the process ends there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Unsupported {
    /// Reading an object-list entry through the task (the local task has none).
    GetObject,
    /// A writable cursor over task memory (the local task has none).
    WritableCursor,
    /// Address ranges of task pages (the local task reports no support for them).
    AddressRanges,
}

/// This area's deaths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathKind {
    /// A part the task's teardown released is used again.
    TornDown,
    /// The object lists are rebuilt for a task that is gone.
    TaskGone,
    /// An operation the task's kind does not provide.
    Unsupported(Unsupported),
    /// The task root is read or written through while NULL (task root 0).
    NullTaskRoot,
    /// An object-list entry is read at an address the lists never mapped.
    UnmappedEntry,
    /// The placement-list copy reaches an address that is not mapped.
    PlacementCopy,
    /// A resource record past the payload the FIFO checked.
    ResourceRecord,
    /// A mapping task whose size is not a whole number of pages.
    MappingSize,
    /// The local task's.
    Local(local::DeathKind),
    /// The remote tasks'.
    Remote(remote::DeathKind),
}

impl DeathKind {
    /// The death's dotted id.
    pub const fn id(self) -> &'static str {
        match self {
            DeathKind::TornDown => "tasks.torn_down",
            DeathKind::TaskGone => "tasks.task_gone",
            DeathKind::Unsupported(Unsupported::GetObject) => "tasks.unsupported.get_object",
            DeathKind::Unsupported(Unsupported::WritableCursor) => {
                "tasks.unsupported.writable_cursor"
            }
            DeathKind::Unsupported(Unsupported::AddressRanges) => {
                "tasks.unsupported.address_ranges"
            }
            DeathKind::NullTaskRoot => "tasks.root.null",
            DeathKind::UnmappedEntry => "tasks.object_list.unmapped_entry",
            DeathKind::PlacementCopy => "tasks.placement_list.copy",
            DeathKind::ResourceRecord => "tasks.resource.record",
            DeathKind::MappingSize => "tasks.mapping.size",
            DeathKind::Local(k) => k.id(),
            DeathKind::Remote(k) => k.id(),
        }
    }

    /// The site a death of this kind records (the kind of fault, and the id): what the guest-memory
    /// accessors take, and what `raise::death` reports.
    pub const fn site(self) -> DeathSite {
        use crate::raise::death::DeathKind as Fault;
        let kind = match self {
            DeathKind::Remote(k) => return k.site(),
            DeathKind::TornDown
            | DeathKind::TaskGone
            | DeathKind::NullTaskRoot
            | DeathKind::UnmappedEntry
            | DeathKind::PlacementCopy => Fault::NullDeref,
            DeathKind::Unsupported(_) => Fault::Abort,
            DeathKind::ResourceRecord => Fault::OutOfBounds,
            DeathKind::MappingSize => Fault::Assert,
            DeathKind::Local(k) => k.fault(),
        };
        DeathSite {
            kind,
            file: self.id(),
            line: 0,
        }
    }
}

impl fmt::Display for DeathKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl From<DeathKind> for crate::raise::DeathId {
    fn from(k: DeathKind) -> crate::raise::DeathId {
        crate::raise::DeathId::Tasks(k)
    }
}

/// Raises `kind` with `detail` in a function reached both ways ([`OnFault::throw`]): with
/// [`OnFault::Die`] the process dies here, before any `Drop` runs; with [`OnFault::Return`], the
/// raise is the caller's.
#[track_caller]
pub(crate) fn raise_on<T>(
    on: OnFault,
    kind: RaiseKind,
    detail: impl fmt::Display,
) -> RaiseResult<T> {
    on.throw(Raise::new(kind).with_detail(detail))
}

/// Raises `kind` with `detail` at a site no path reaches without a handler: `Err`, the caller's
/// to catch.
#[track_caller]
pub(crate) fn raise_caught<T>(kind: RaiseKind, detail: impl fmt::Display) -> RaiseResult<T> {
    Err(Raise::new(kind).with_detail(detail))
}

/// Raises `kind` with `detail` as [`raise_caught`] does, carrying `v` outward
/// ([`Raise::defer`]): `v` is released where the raise ends, after the catcher's report. Every
/// site is one no path reaches without a handler, so the raise is `Err`.
#[track_caller]
pub(crate) fn raise_carrying<T>(
    kind: RaiseKind,
    detail: impl fmt::Display,
    v: impl std::any::Any + Send,
) -> RaiseResult<T> {
    Err(Raise::new(kind).with_detail(detail).defer(v))
}

#[cfg(test)]
pub(crate) mod test_support {
    //! What the tests of the tasks and of the isolated tasks' service share.

    /// The lines the tasks area and the isolated tasks' service write while this lives, captured
    /// from the `log` crate (`testkit::logs`): it relies on nextest's one process per test, and
    /// one lives at a time.
    pub(crate) struct Lines(crate::testkit::logs::Capture);

    impl Lines {
        /// Starts capturing.
        pub(crate) fn capture() -> Lines {
            Lines(crate::testkit::logs::capture())
        }

        /// The two areas' messages since the capture began or the last `take`, in order.
        pub(crate) fn take(&self) -> Vec<String> {
            let below = |target: &str, area: &str| {
                target
                    .strip_prefix(area)
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with("::"))
            };
            self.0
                .take()
                .into_iter()
                .filter(|(_, target, _)| {
                    below(target, concat!(env!("CARGO_CRATE_NAME"), "::tasks"))
                        || below(target, concat!(env!("CARGO_CRATE_NAME"), "::gpu_task"))
                })
                .map(|(_, _, message)| message)
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raise::DeathId;
    use crate::raise::death::DeathKind as Fault;
    use crate::testkit::FakePlatform;
    use crate::testkit::capture::catch_death;
    use parking_lot::Mutex;

    type P = FakePlatform;

    /// ExecStatus keeps every u32 and tests only 1 as success
    #[test]
    fn exec_status_values() {
        assert_eq!(ExecStatus::NOT_COMPLETE.raw(), 0);
        assert_eq!(ExecStatus::OK.raw(), 1);
        assert_eq!(ExecStatus::FAIL.raw(), 2);
        assert_eq!(ExecStatus::GPU_OR_TIMEOUT.raw(), 3);
        assert_eq!(ExecStatus::CRASHED.raw(), 4);
        assert!(ExecStatus::OK.is_ok());
        assert!(!ExecStatus::FAIL.is_ok());
        assert!(!ExecStatus(0xdead_beef).is_ok());
        assert_eq!(ExecStatus(0xdead_beef).raw(), 0xdead_beef);
        assert_eq!(ExecStatus::default(), ExecStatus::NOT_COMPLETE);
    }

    /// A completion can be called twice, and both calls are seen
    #[test]
    fn completion_is_callable_twice() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        let done: Completion = Arc::new(move |st, e| s.lock().push((st, e.is_some())));
        done(ExecStatus::OK, None);
        done(ExecStatus::FAIL, Some(NsErrorText::default()));
        assert_eq!(
            *seen.lock(),
            vec![(ExecStatus::OK, false), (ExecStatus::FAIL, true)]
        );
    }

    /// A task info is zeroed by default
    #[test]
    fn a_task_info_is_zeroed_by_default() {
        let info = TaskInfo::default();
        assert_eq!(info.task, HostTask::default());
        assert_eq!(info.base_address, HostPtr::default());
    }

    /// The `Debug` form shows the record counts and the stamp index (diagnostics only).
    #[test]
    fn exec_args_debug() {
        let cmd = [0u8; 16];
        let res = [0u8; 8];
        let value = Arc::new(AtomicU64::new(0));
        let a: ExecArgs<'_, P, ResourceReference> = ExecArgs {
            cmd_buffers: Records::new(&cmd, 1).unwrap(),
            resources: Records::new(&res, 2).unwrap(),
            stamp_index: 3,
            fifo_event: None,
            fifo_event_value: &value,
        };
        assert_eq!(
            format!("{a:?}"),
            "ExecArgs { cmd_buffers: 1, resources: 2, stamp_index: 3, .. }"
        );
    }

    /// The command records have the guest's sizes and byte order
    #[test]
    fn record_layouts() {
        let b = [1u8, 0, 0, 0, 1, 0, 1, 0];
        let r = Records::<ResourceInvalidation>::new(&b, 1)
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(r.reference, 1);
        assert_eq!(
            (
                r.clear_host_valid,
                r.set_host_valid,
                r.clear_guest_valid,
                r.set_guest_valid
            ),
            (1, 0, 1, 0)
        );
        let r = Records::<ResourceReference>::new(&b, 2).unwrap();
        assert_eq!(
            r.to_vec(),
            vec![
                ResourceReference { reference: 1 },
                ResourceReference {
                    reference: 0x0001_0001
                }
            ]
        );
    }

    /// A raise with `Die` dies at the raise, with its kind's id
    #[test]
    fn a_raise_without_a_handler_dies() {
        let d = catch_death(|| raise_on::<()>(OnFault::Die, RaiseKind::ResourceIndex, "x"))
            .unwrap_err();
        assert_eq!(
            (d.kind, d.file, d.line),
            (Fault::UnhandledRaise, "tasks.resource.index", 0)
        );
    }

    /// A raise with `Return` comes back as `Err`, with its detail and its id
    #[test]
    fn a_raise_with_return_comes_back() {
        let r = raise_on::<u32>(
            OnFault::Return,
            RaiseKind::BufferLength,
            format_args!("n={}", 3),
        );
        let t = r.unwrap_err();
        assert_eq!(t.detail().unwrap_or_default(), "n=3");
        assert_eq!((t.site().file, t.site().line), ("tasks.buffer.length", 0));
    }

    /// Each death names its kind of fault and its id
    #[test]
    fn deaths_name_their_fault() {
        let cases = [
            (DeathKind::TornDown, Fault::NullDeref),
            (DeathKind::TaskGone, Fault::NullDeref),
            (DeathKind::NullTaskRoot, Fault::NullDeref),
            (DeathKind::UnmappedEntry, Fault::NullDeref),
            (DeathKind::PlacementCopy, Fault::NullDeref),
            (DeathKind::Unsupported(Unsupported::GetObject), Fault::Abort),
            (DeathKind::ResourceRecord, Fault::OutOfBounds),
            (DeathKind::MappingSize, Fault::Assert),
        ];
        for (k, fault) in cases {
            let d = catch_death(|| -> () { crate::fatal!(k) }).unwrap_err();
            assert_eq!(d, k.site());
            assert_eq!((d.kind, d.file), (fault, k.id()));
            assert_eq!(DeathId::from(k).to_string(), k.id());
        }
        assert_eq!(
            RaiseKind::ResourceLookup.to_string(),
            "tasks.resource.lookup"
        );
    }
}
