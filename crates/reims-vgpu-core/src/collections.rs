//! Containers named by the order they give.
//!
//! - [`BucketOrderMap`]: the task table. Iteration, drain and drop in ascending task ID, which
//!   orders every fan-out across tasks and the releases when the table empties.
//! - [`ListOrderMap`]: the resource caches. A hash map that can be walked and drained; nothing
//!   the guest or host sees depends on its order.
//! - [`FirstFitVa`]: the root task's virtual address space. First fit over a sorted free list,
//!   never reset, so a leaked range moves every later offset.
//! - [`LookupMap`]: maps that are only looked up. No iteration, and a fixed hasher, so its drop
//!   order is the same in every run.

pub mod bucket_order;
pub mod first_fit;
pub mod list_order;
pub mod lookup;

pub use bucket_order::BucketOrderMap;
pub use first_fit::{FirstFitVa, FreeRange};
pub use list_order::ListOrderMap;
pub use lookup::LookupMap;
