//! How the areas reach one another (architecture 9).
//!
//! Most areas call each other's concrete types: the FIFO, a task and a mapping call their device
//! through [`crate::device::DeviceCore`]'s methods, reached through the unretained links in
//! [`links`]; the device and the FIFO call a task through [`crate::tasks::Task`]'s methods; the
//! deserializer calls its manager's lookup
//! ([`crate::resources::manager::lookup::ManagerLookup`]). One `dyn` seam is left, because it has
//! two implementors: [`ResourceHost`] (`LocalTask` and `GpuTask`), what resources, the resource
//! manager and the object lists ask of their task. The isolated task's transport traits live in
//! `tasks::launcher`.
//!
//! Conventions every cross-area call follows:
//! - Handles cross by value, retained (`.clone_ref()` at the callee's return), never as borrows of
//!   the callee's fields: the caller may hold the callee through a `Weak`, so no reference into it
//!   can outlive the call. None of these retains is of an IOSurface, whose count is guest-visible
//!   (D6); surfaces cross only where one is handed over.
//! - A method that can raise returns `RaiseResult<T>` (D7). Returning `RaiseResult` never forces a
//!   catch: with no handler live, the raise dies at its own site.
//! - Records the guest wrote (payload arrays) cross as [`Records`](crate::fifo::wire::Records), a
//!   bounds-checked unaligned view of the FIFO's payload bytes, never as `&[T]` into them.

pub mod links;

pub use crate::resources::host::{CallBorrow, ResourceHost};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::Platform;

    /// Every seam is object-safe (D21): this only has to compile.
    #[allow(dead_code, clippy::too_many_arguments)]
    fn object_safe<P: Platform>(_: Option<&dyn ResourceHost<P>>) {}

    /// Each real type satisfies the seams it implements (architecture 9's pairings): these only
    /// have to compile.
    #[allow(dead_code, clippy::type_complexity)] // compile-only; the tuples list the seams
    mod pairings {
        use super::*;
        use crate::gpu_task::GpuTask;
        use crate::tasks::launcher::HostSide;
        use crate::tasks::local::LocalTask;
        use crate::tasks::remote::Endpoint;
        use std::sync::Arc;

        fn local_task<P: Platform>(t: Arc<LocalTask<P>>) -> Arc<dyn ResourceHost<P>> {
            t
        }

        fn remote_endpoint<P: Platform>(d: Arc<Endpoint<P>>) -> Arc<dyn HostSide<P>> {
            d
        }

        fn gpu_task<P: Platform>(t: Arc<GpuTask<P>>) -> Arc<dyn ResourceHost<P>> {
            t
        }
    }

    /// The seams are object-safe and upcast (compile-time checks)
    #[test]
    fn seams_are_object_safe() {
        // The functions above compiling is the test; nothing to run.
    }
}
