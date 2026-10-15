//! The guest's ICB records: the creation command with its serialized descriptor, the object
//! record the resource keeps, and the command records the walks page between guest memory and
//! Metal's indirect commands. Their layouts are the guest ABI's; the command records sit at the
//! offsets the ICB's [`IndirectCommandLayout`] names.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::platform::types::{MtlOrigin, MtlSize};
use crate::resources::icb::IndirectCommandLayout;

/// The object type of an indirect command buffer in the object list.
pub(crate) const ICB_OBJECT_TYPE: u8 = 20;

/// The creation command's type (after the object record): a new ICB with a descriptor.
pub(crate) const NEW_ICB_COMMAND: u32 = 54;

/// The serialized ICB descriptor (0x14 bytes). The one-bit fields sit in the `u16` at +0xe:
/// see [`SerializedIcbDescriptor::bit`].
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub(crate) struct SerializedIcbDescriptor {
    pub command_types: u32,
    pub max_vertex_buffer_bind_count: u8,
    pub max_fragment_buffer_bind_count: u8,
    pub max_kernel_buffer_bind_count: u8,
    pub max_object_buffer_bind_count: u8,
    pub max_mesh_buffer_bind_count: u8,
    pub max_kernel_threadgroup_memory_bind_count: u8,
    pub max_object_threadgroup_memory_bind_count: u8,
    pub max_scissor_rect_count: u8,
    pub max_viewport_count: u8,
    /// Feeds no setter.
    pub field_0d: u8,
    /// The one-bit fields, bit 0 (inherit pipeline state) to bit 14 (inherit blend color).
    pub bits: u16,
    pub allow_override_render_states: u32,
}

const _: () = assert!(size_of::<SerializedIcbDescriptor>() == 0x14);

impl SerializedIcbDescriptor {
    /// The one-bit field at bit `n` of the `u16` at +0xe.
    pub(crate) fn bit(&self, n: u32) -> bool {
        u32::from(self.bits).wrapping_shr(n) & 1 != 0
    }
}

/// The creation command: the descriptor, the command layout, the maximum command count and the
/// resource options (0x50 bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub(crate) struct NewIcbCommand {
    pub descriptor: SerializedIcbDescriptor,
    pub command_layout: IndirectCommandLayout,
    pub max_command_count: u32,
    pub options: u16,
    /// Tail padding, read with the rest.
    pub field_4e: [u8; 2],
}

const _: () = assert!(size_of::<NewIcbCommand>() == 0x50);

/// The ICB's object record (12 bytes, packed): the resource keeps these bytes as its
/// descriptor.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
pub(crate) struct ObjectIcb {
    pub length: u64,
    pub page: u32,
}

const _: () = assert!(size_of::<ObjectIcb>() == 12);

/// A bind's size in the command (a buffer reference, a GPU address and an offset, packed).
pub(crate) const BIND_SIZE: u64 = 0x14;
/// An attribute stride's and a threadgroup memory length's size in the command.
pub(crate) const WORD_SIZE: u64 = 8;

/// A draw (packed and 4-aligned: 0x24 bytes with 2 tail bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
pub(crate) struct DrawPrimitives {
    pub primitive_type: u16,
    pub vertex_start: u64,
    pub vertex_count: u64,
    pub instance_count: u64,
    pub base_instance: u64,
    /// The tail the 4-byte alignment adds.
    pub field_22: [u8; 2],
}

const _: () = assert!(size_of::<DrawPrimitives>() == 0x24);

/// An indexed draw (0x38 bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub(crate) struct DrawIndexedPrimitives {
    pub primitive_type: u16,
    pub index_type: u16,
    pub index_buffer_reference: u32,
    pub index_count: u64,
    pub index_buffer_gpu_address: u64,
    pub index_buffer_offset: u64,
    pub instance_count: u64,
    pub base_vertex: i64,
    pub base_instance: u64,
}

const _: () = assert!(size_of::<DrawIndexedPrimitives>() == 0x38);

/// A patch draw (packed and 4-aligned: 0x38 bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
pub(crate) struct DrawPatches {
    pub number_of_patch_control_points: u16,
    pub patch_start: u64,
    pub patch_count: u64,
    pub patch_index_buffer_reference: u32,
    pub patch_index_buffer_gpu_address: u64,
    pub patch_index_buffer_offset: u64,
    pub instance_count: u64,
    pub base_instance: u64,
    /// The tail the 4-byte alignment adds.
    pub field_36: [u8; 2],
}

const _: () = assert!(size_of::<DrawPatches>() == 0x38);

/// A patch draw's tessellation factor buffer (packed, 0x20 bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
pub(crate) struct TessellationFactor {
    pub buffer_reference: u32,
    pub buffer_gpu_address: u64,
    pub buffer_offset: u64,
    pub instance_stride: u64,
    pub scale: f32,
}

const _: () = assert!(size_of::<TessellationFactor>() == 0x20);

/// An indexed patch draw (packed and 4-aligned: 0x4c bytes).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
pub(crate) struct DrawIndexedPatches {
    pub number_of_patch_control_points: u16,
    pub patch_start: u64,
    pub patch_count: u64,
    pub patch_index_buffer_reference: u32,
    pub patch_index_buffer_gpu_address: u64,
    pub patch_index_buffer_offset: u64,
    pub control_point_index_buffer_reference: u32,
    pub control_point_index_buffer_gpu_address: u64,
    pub control_point_index_buffer_offset: u64,
    pub instance_count: u64,
    pub base_instance: u64,
    /// The tail the 4-byte alignment adds.
    pub field_4a: [u8; 2],
}

const _: () = assert!(size_of::<DrawIndexedPatches>() == 0x4c);

/// An `MTLSize` or an `MTLOrigin` as the records hold it: three `u64`s.
// name is ours: the records reuse one three-word layout for sizes and origins
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub(crate) struct Triple {
    pub a: u64,
    pub b: u64,
    pub c: u64,
}

impl Triple {
    /// The triple as a size.
    pub(crate) fn size(self) -> MtlSize {
        MtlSize {
            width: self.a,
            height: self.b,
            depth: self.c,
        }
    }

    /// The triple as an origin.
    pub(crate) fn origin(self) -> MtlOrigin {
        MtlOrigin {
            x: self.a,
            y: self.b,
            z: self.c,
        }
    }

    /// A size as a triple.
    pub(crate) fn of_size(s: MtlSize) -> Triple {
        Triple {
            a: s.width,
            b: s.height,
            c: s.depth,
        }
    }

    /// An origin as a triple.
    pub(crate) fn of_origin(o: MtlOrigin) -> Triple {
        Triple {
            a: o.x,
            b: o.y,
            c: o.z,
        }
    }
}

/// A concurrent dispatch (0x30 bytes): the grid (in threadgroups or in threads), then the
/// threads per threadgroup.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub(crate) struct ConcurrentDispatch {
    pub grid: Triple,
    pub threads_per_threadgroup: Triple,
}

/// A compute command's dispatch arguments (0x38 bytes): the stage-in region and the image
/// block size.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub(crate) struct Dispatch {
    pub stage_in_region_size: Triple,
    pub stage_in_region_origin: Triple,
    pub imageblock_width: u32,
    pub imageblock_height: u32,
}

const _: () = assert!(size_of::<Dispatch>() == 0x38);

/// The command types that make an ICB a render ICB: draw, indexed draw, the two patch draws
/// and the two mesh draws. Any other set pages as compute.
pub(crate) const RENDER_COMMAND_TYPES: u32 = 0x18f;

/// A render command's type (`MTLIndirectCommandType`) as the walks page it; the guest may write
/// any other value, which fails the walk after the command's barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RenderCommand {
    /// An empty slot: reset on prepare, only the barrier written back.
    Empty,
    /// A draw.
    Draw,
    /// An indexed draw.
    DrawIndexed,
    /// A patch draw.
    DrawPatches,
    /// An indexed patch draw.
    DrawIndexedPatches,
}

impl TryFrom<u32> for RenderCommand {
    type Error = u32;

    fn try_from(ty: u32) -> Result<Self, u32> {
        Ok(match ty {
            0 => RenderCommand::Empty,
            1 => RenderCommand::Draw,
            2 => RenderCommand::DrawIndexed,
            4 => RenderCommand::DrawPatches,
            8 => RenderCommand::DrawIndexedPatches,
            other => return Err(other),
        })
    }
}

/// A compute command's type as the walks page it; any other value fails the walk after the
/// command's barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ComputeCommand {
    /// An empty slot.
    Empty,
    /// A concurrent dispatch in threadgroups.
    DispatchThreadgroups,
    /// A concurrent dispatch in threads.
    DispatchThreads,
}

impl TryFrom<u32> for ComputeCommand {
    type Error = u32;

    fn try_from(ty: u32) -> Result<Self, u32> {
        Ok(match ty {
            0 => ComputeCommand::Empty,
            0x20 => ComputeCommand::DispatchThreadgroups,
            0x40 => ComputeCommand::DispatchThreads,
            other => return Err(other),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_types_convert_at_the_guest_edge() {
        assert_eq!(RenderCommand::try_from(0), Ok(RenderCommand::Empty));
        assert_eq!(
            RenderCommand::try_from(8),
            Ok(RenderCommand::DrawIndexedPatches)
        );
        assert_eq!(RenderCommand::try_from(0x10), Err(0x10));
        assert_eq!(
            ComputeCommand::try_from(0x40),
            Ok(ComputeCommand::DispatchThreads)
        );
        assert_eq!(ComputeCommand::try_from(1), Err(1));
    }

    #[test]
    fn descriptor_bits_read_the_u16_at_0xe() {
        let d = SerializedIcbDescriptor {
            bits: 0b0100_0000_0000_0101,
            ..Default::default()
        };
        assert!(d.bit(0) && !d.bit(1) && d.bit(2) && d.bit(14) && !d.bit(15));
        assert_eq!(d.as_bytes()[0xe..0x10], [0b101, 0b0100_0000]);
    }
}
