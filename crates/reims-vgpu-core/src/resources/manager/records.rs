//! The guest records the manager reads: the operation header every serialized creation opens
//! with, the buffer and heap records, and the placement slot a creation reads and writes back.

use std::mem::offset_of;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::guest::GuestPtr;
use crate::guest::records::ObjectPlacement;
use crate::platform::Platform;
use crate::resources::manager::DeathKind;

/// The header every serialized object creation command opens with (8 bytes).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct OperationHeader {
    /// The command's type. Name is ours: each kind's creator switches on it.
    pub command_type: u32,
    /// Nothing reads it.
    pub field4: u32,
}

/// An ICB bind: a buffer reference, a GPU address and an offset (packed, 0x14 bytes).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct IndirectCommandBufferBind {
    /// The buffer's reference.
    pub buffer_reference: u32,
    /// The GPU address the bind names.
    pub gpu_address: u64,
    /// The offset into the buffer.
    pub offset: u64,
}

/// A buffer found from a GPU address: the buffer, its reference and the offset into it.
pub struct ResourceAndOffset<P: Platform> {
    /// The buffer; `None` is nil.
    pub buffer: Option<P::Buffer>,
    /// Its reference.
    pub buffer_reference: u32,
    /// The offset into it.
    pub offset: u64,
}

/// A buffer's record (0x10 bytes): its length and first guest page, then a flags byte and
/// padding, which a buffer's creation passes on whole as its descriptor.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub(crate) struct ObjectBuffer {
    pub(crate) length: u64,
    pub(crate) page: u32,
    pub(crate) flags: u8,
    pub(crate) field_0d: [u8; 3],
}

/// A CPU buffer's record (packed, 0xc bytes): its length and first guest page.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
pub(crate) struct ObjectCpuBuffer {
    pub(crate) length: u64,
    pub(crate) page: u32,
}

/// A mapper-ref buffer's record: the mapping id.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub(crate) struct ObjectMapperRefBuffer {
    pub(crate) mapping_id: u64,
}

/// A heap's record (0x10 bytes), passed on whole to the heap's factory, which reads it.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub(crate) struct ObjectHeap {
    pub(crate) field_00: u64,
    pub(crate) field_08: u64,
}

/// A heap buffer's record (packed, 0x14 bytes): its heap's reference, then the offset and
/// length in the heap.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
pub(crate) struct ObjectHeapBuffer {
    pub(crate) heap_reference: u32,
    pub(crate) offset: u64,
    pub(crate) length: u64,
}

const _: () = {
    assert!(size_of::<OperationHeader>() == 8);
    assert!(size_of::<IndirectCommandBufferBind>() == 0x14);
    assert!(size_of::<ObjectBuffer>() == 0x10);
    assert!(size_of::<ObjectCpuBuffer>() == 0xc);
    assert!(size_of::<ObjectMapperRefBuffer>() == 8);
    assert!(size_of::<ObjectHeap>() == 0x10);
    assert!(size_of::<ObjectHeapBuffer>() == 0x14);
};

/// The little-endian `u32` at `at` of a guest command, if the command holds it.
pub(crate) fn word(command: &[u8], at: usize) -> Option<u32> {
    let bytes = command.get(at..)?.first_chunk::<4>()?;
    Some(u32::from_le_bytes(*bytes))
}

/// Where a placement slot's first word sits: the resource index an object is made at
/// (`ObjectPlacement::field0`).
pub(crate) const PLACEMENT_FIELD0: u64 = crate::num::const_u64(offset_of!(ObjectPlacement, field0));
/// Where a placement slot's second word sits: the GPU resource ID the object gets.
pub(crate) const PLACEMENT_RESOURCE_ID: u64 =
    crate::num::const_u64(offset_of!(ObjectPlacement, resource_id));

/// One load of a placement word. Each call is its own load: a guest storing into the slot
/// between two of them is seen by the second. Under test, a guest store may be pinned right
/// after it (`create::test_env::race_after_load`).
fn placement_load(slot: GuestPtr, offset: u64) -> u64 {
    let v = slot.read_u64(offset, DeathKind::NullPlacement);
    #[cfg(test)]
    crate::resources::manager::create::test_env::after_placement_load(slot, offset);
    v
}

/// The placement slot's first word, loaded now.
pub(crate) fn placement_field0(slot: GuestPtr) -> u64 {
    placement_load(slot, PLACEMENT_FIELD0)
}

/// The placement slot's GPU resource ID, loaded now.
pub(crate) fn placement_resource_id(slot: GuestPtr) -> u64 {
    placement_load(slot, PLACEMENT_RESOURCE_ID)
}

/// Stores the placement slot's first word.
pub(crate) fn set_placement_field0(slot: GuestPtr, v: u64) {
    slot.write_u64(PLACEMENT_FIELD0, v, DeathKind::NullPlacement);
}

/// Stores the placement slot's GPU resource ID.
pub(crate) fn set_placement_resource_id(slot: GuestPtr, v: u64) {
    slot.write_u64(PLACEMENT_RESOURCE_ID, v, DeathKind::NullPlacement);
}
