//! The guest's command records: every record a decoder reads and every reply it writes, as
//! `#[repr(C)]` (or packed) structs with const size asserts.
//!
//! Each record is the guest ABI's struct byte for byte, its tail padding included: a decoder
//! reads a record's whole `size_of` from its cursor, so a payload short only by the padding
//! still fails to read. The size includes padding; every padding byte is a field here. Padding is
//! `unknown_xx` (xx its offset), bytes read and ignored; an unread field whose meaning nothing
//! states is `field_xx`.
//!
//! Layout. A record the ABI lays out naturally is `#[repr(C)]`, with explicit padding arrays
//! where the C layout leaves gaps, so its size, field offsets and alignment are the ABI's. A
//! record the ABI declares packed (`packed, aligned(4)`, or under `#pragma pack(4)`) is
//! `#[repr(C, packed)]` with its gaps and tail padding spelled out, so its size and offsets are
//! the ABI's; its Rust alignment is 1 where the ABI's is 4, which nothing observes: records are
//! copied out of guest memory by value and replies are copied in by value. The tests check every
//! size and offset against a table of the guest ABI's layouts that Apple clang printed
//! (`sizeof`/`offsetof`, arm64).
//!
//! Field names are the guest ABI's, in snake case. Types drop the `PG` prefix; the anonymous
//! attachment layouts of the render pass record and the Metal struct copies (`*Record`) have
//! names that are ours, as their docs say.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::deserializer::texture_descriptor::{
    SerializedTextureDescriptor, SerializedTextureDescriptor2,
};
use crate::num::Truncate;
use crate::platform::types::{ClearColor, MtlCoordinate2D, MtlOrigin, MtlSize};

// ---------------------------------------------------------------------------------------------
// The Metal structs the records embed. `platform::types` keeps the by-value Metal types without
// zerocopy's derives; these are their wire copies (the names are ours: the Metal type plus
// `Record`), same fields, same layout, converted with `From`.

/// `MTLSize` as a record embeds it (`{?=QQQ}`, 0x18 bytes).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct MtlSizeRecord {
    /// `width`.
    pub width: u64,
    /// `height`.
    pub height: u64,
    /// `depth`.
    pub depth: u64,
}

const _: () = assert!(size_of::<MtlSizeRecord>() == size_of::<MtlSize>());

impl From<MtlSizeRecord> for MtlSize {
    fn from(r: MtlSizeRecord) -> MtlSize {
        MtlSize::new(r.width, r.height, r.depth)
    }
}

impl From<MtlSize> for MtlSizeRecord {
    fn from(s: MtlSize) -> MtlSizeRecord {
        MtlSizeRecord {
            width: s.width,
            height: s.height,
            depth: s.depth,
        }
    }
}

/// `MTLOrigin` as a record embeds it (`{?=QQQ}`, 0x18 bytes).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct MtlOriginRecord {
    /// `x`.
    pub x: u64,
    /// `y`.
    pub y: u64,
    /// `z`.
    pub z: u64,
}

const _: () = assert!(size_of::<MtlOriginRecord>() == size_of::<MtlOrigin>());

impl From<MtlOriginRecord> for MtlOrigin {
    fn from(r: MtlOriginRecord) -> MtlOrigin {
        MtlOrigin::new(r.x, r.y, r.z)
    }
}

impl From<MtlOrigin> for MtlOriginRecord {
    fn from(o: MtlOrigin) -> MtlOriginRecord {
        MtlOriginRecord {
            x: o.x,
            y: o.y,
            z: o.z,
        }
    }
}

/// `MTLClearColor` as a record embeds it (`{?=dddd}`, 0x20 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub struct ClearColorRecord {
    /// `red`.
    pub red: f64,
    /// `green`.
    pub green: f64,
    /// `blue`.
    pub blue: f64,
    /// `alpha`.
    pub alpha: f64,
}

const _: () = assert!(size_of::<ClearColorRecord>() == size_of::<ClearColor>());

impl From<ClearColorRecord> for ClearColor {
    fn from(r: ClearColorRecord) -> ClearColor {
        ClearColor {
            red: r.red,
            green: r.green,
            blue: r.blue,
            alpha: r.alpha,
        }
    }
}

impl From<ClearColor> for ClearColorRecord {
    fn from(c: ClearColor) -> ClearColorRecord {
        ClearColorRecord {
            red: c.red,
            green: c.green,
            blue: c.blue,
            alpha: c.alpha,
        }
    }
}

/// `MTLCoordinate2D` as a record embeds it (`{?=ff}`, 8 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub struct Coordinate2DRecord {
    /// `x`.
    pub x: f32,
    /// `y`.
    pub y: f32,
}

const _: () = assert!(size_of::<Coordinate2DRecord>() == size_of::<MtlCoordinate2D>());

impl From<Coordinate2DRecord> for MtlCoordinate2D {
    fn from(r: Coordinate2DRecord) -> MtlCoordinate2D {
        MtlCoordinate2D { x: r.x, y: r.y }
    }
}

impl From<MtlCoordinate2D> for Coordinate2DRecord {
    fn from(c: MtlCoordinate2D) -> Coordinate2DRecord {
        Coordinate2DRecord { x: c.x, y: c.y }
    }
}

/// `MTLResourceID` as a reply embeds it (`{MTLResourceID=Q}`, 8 bytes).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct ResourceIdRecord {
    /// `_impl`.
    pub impl_: u64,
}

// ---------------------------------------------------------------------------------------------
// The segment header.

/// `PGSerializerCommandSegmentHeader`: `{?=ICCC}`, read as 8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct SegmentHeader {
    /// `length`: `decodeSegments` truncates its cursor to it.
    pub length: u32,
    /// `encoderType`: 0 render, 1 compute, 2 blit, 4 info; 5 prefixes protection options.
    pub encoder_type: u8,
    /// `continuesPrevious`.
    pub continues_previous: u8,
    /// `continuedByNext`.
    pub continued_by_next: u8,
    /// Padding (the natural layout's).
    pub unknown07: u8,
}

const _: () = assert!(size_of::<SegmentHeader>() == 8);

// ---------------------------------------------------------------------------------------------
// The records.

/// `PGCmdFenceOp`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdFenceOp {
    /// `fenceReference`.
    pub fence_reference: u32,
}

const _: () = assert!(size_of::<CmdFenceOp>() == 0x4);

/// `PGCmdHeader`: 0x8 bytes.
///
/// The header every command record follows. The decoders switch on
/// commandID and take a sub-cursor over length - 8 bytes, so length is the
/// whole record's, with this header counted in.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdHeader {
    /// `commandID`.
    pub command_id: u32,
    /// `length`.
    pub length: u32,
}

const _: () = assert!(size_of::<CmdHeader>() == 0x8);

/// `PGCmdGenerateMipmaps`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdGenerateMipmaps {
    /// `textureReference`.
    pub texture_reference: u32,
}

const _: () = assert!(size_of::<CmdGenerateMipmaps>() == 0x4);

/// `PGCmdOptimize`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdOptimize {
    /// `textureReference`.
    pub texture_reference: u32,
}

const _: () = assert!(size_of::<CmdOptimize>() == 0x4);

/// `PGCmdOptimizeImage`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdOptimizeImage {
    /// `textureReference`.
    pub texture_reference: u32,
    /// `slice`.
    pub slice: u16,
    /// `level`.
    pub level: u16,
}

const _: () = assert!(size_of::<CmdOptimizeImage>() == 0x8);

/// `PGCmdSynchronize`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSynchronize {
    /// `resourceReference`.
    pub resource_reference: u32,
}

const _: () = assert!(size_of::<CmdSynchronize>() == 0x4);

/// `PGCmdSynchronizeImage`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSynchronizeImage {
    /// `textureReference`.
    pub texture_reference: u32,
    /// `slice`.
    pub slice: u16,
    /// `level`.
    pub level: u16,
}

const _: () = assert!(size_of::<CmdSynchronizeImage>() == 0x8);

/// `PGCmdCopyFromBufferToTexture`: 0x58 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdCopyFromBufferToTexture {
    /// `sourceBufferReference`.
    pub source_buffer_reference: u32,
    /// `destinationTextureReference`.
    pub destination_texture_reference: u32,
    /// `sourceOffset`.
    pub source_offset: u64,
    /// `sourceBytesPerRow`.
    pub source_bytes_per_row: u64,
    /// `sourceBytesPerImage`.
    pub source_bytes_per_image: u64,
    /// `sourceSize`.
    pub source_size: MtlSizeRecord,
    /// `destinationOrigin`.
    pub destination_origin: MtlOriginRecord,
    /// `destinationSlice`.
    pub destination_slice: u16,
    /// `destinationLevel`.
    pub destination_level: u16,
    /// `options`.
    pub options: u32,
}

const _: () = assert!(size_of::<CmdCopyFromBufferToTexture>() == 0x58);

/// `PGCmdCopyFromBufferToBuffer`: 0x20 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdCopyFromBufferToBuffer {
    /// `sourceBufferReference`.
    pub source_buffer_reference: u32,
    /// `destinationBufferReference`.
    pub destination_buffer_reference: u32,
    /// `sourceOffset`.
    pub source_offset: u64,
    /// `destinationOffset`.
    pub destination_offset: u64,
    /// `size`.
    pub size: u64,
}

const _: () = assert!(size_of::<CmdCopyFromBufferToBuffer>() == 0x20);

/// `PGCmdCopyFromTextureToBuffer`: 0x58 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdCopyFromTextureToBuffer {
    /// `sourceTextureReference`.
    pub source_texture_reference: u32,
    /// `destinationBufferReference`.
    pub destination_buffer_reference: u32,
    /// `sourceOrigin`.
    pub source_origin: MtlOriginRecord,
    /// `sourceSize`.
    pub source_size: MtlSizeRecord,
    /// `destinationOffset`.
    pub destination_offset: u64,
    /// `destinationBytesPerRow`.
    pub destination_bytes_per_row: u64,
    /// `destinationBytesPerImage`.
    pub destination_bytes_per_image: u64,
    /// `sourceSlice`.
    pub source_slice: u16,
    /// `sourceLevel`.
    pub source_level: u16,
    /// `options`.
    pub options: u16,
    /// Tail padding: the record's size counts it, so a record must carry it.
    pub unknown56: [u8; 2],
}

const _: () = assert!(size_of::<CmdCopyFromTextureToBuffer>() == 0x58);

/// `PGCmdCopyFromTextureToTexture`: 0x58 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdCopyFromTextureToTexture {
    /// `sourceTextureReference`.
    pub source_texture_reference: u32,
    /// `destinationTextureReference`.
    pub destination_texture_reference: u32,
    /// `sourceOrigin`.
    pub source_origin: MtlOriginRecord,
    /// `sourceSize`.
    pub source_size: MtlSizeRecord,
    /// `destinationOrigin`.
    pub destination_origin: MtlOriginRecord,
    /// `sourceSlice`.
    pub source_slice: u16,
    /// `sourceLevel`.
    pub source_level: u16,
    /// `destinationSlice`.
    pub destination_slice: u16,
    /// `destinationLevel`.
    pub destination_level: u16,
}

const _: () = assert!(size_of::<CmdCopyFromTextureToTexture>() == 0x58);

/// `PGCmdCopyFromTextureToTextureWithNumSliceLevel`: 0x14 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdCopyFromTextureToTextureWithNumSliceLevel {
    /// `sourceTextureReference`.
    pub source_texture_reference: u32,
    /// `destinationTextureReference`.
    pub destination_texture_reference: u32,
    /// `sourceSlice`.
    pub source_slice: u16,
    /// `sourceLevel`.
    pub source_level: u16,
    /// `destinationSlice`.
    pub destination_slice: u16,
    /// `destinationLevel`.
    pub destination_level: u16,
    /// `sliceCount`.
    pub slice_count: u16,
    /// `levelCount`.
    pub level_count: u16,
}

const _: () = assert!(size_of::<CmdCopyFromTextureToTextureWithNumSliceLevel>() == 0x14);

/// `PGCmdInvalidateCompressedTexture`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdInvalidateCompressedTexture {
    /// `textureReference`.
    pub texture_reference: u32,
}

const _: () = assert!(size_of::<CmdInvalidateCompressedTexture>() == 0x4);

/// `PGCmdInvalidateCompressedTextureImage`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdInvalidateCompressedTextureImage {
    /// `textureReference`.
    pub texture_reference: u32,
    /// `slice`.
    pub slice: u16,
    /// `level`.
    pub level: u16,
}

const _: () = assert!(size_of::<CmdInvalidateCompressedTextureImage>() == 0x8);

/// `PGCmdCopyIndirectCommandBuffer`: 0x20 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdCopyIndirectCommandBuffer {
    /// `sourceIndirectCommandBufferReference`.
    pub source_indirect_command_buffer_reference: u32,
    /// `destinationIndirectCommandBufferReference`.
    pub destination_indirect_command_buffer_reference: u32,
    /// `sourceRangeLocation`.
    pub source_range_location: u64,
    /// `sourceRangeLength`.
    pub source_range_length: u64,
    /// `destinationIndex`.
    pub destination_index: u64,
}

const _: () = assert!(size_of::<CmdCopyIndirectCommandBuffer>() == 0x20);

/// `PGCmdCopyFromTextureToTextureWithOptions`: 0x5c bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdCopyFromTextureToTextureWithOptions {
    /// `sourceTextureReference`.
    pub source_texture_reference: u32,
    /// `destinationTextureReference`.
    pub destination_texture_reference: u32,
    /// `sourceOrigin`.
    pub source_origin: MtlOriginRecord,
    /// `sourceSize`.
    pub source_size: MtlSizeRecord,
    /// `destinationOrigin`.
    pub destination_origin: MtlOriginRecord,
    /// `sourceSlice`.
    pub source_slice: u16,
    /// `sourceLevel`.
    pub source_level: u16,
    /// `destinationSlice`.
    pub destination_slice: u16,
    /// `destinationLevel`.
    pub destination_level: u16,
    /// `options`.
    pub options: u32,
}

const _: () = assert!(size_of::<CmdCopyFromTextureToTextureWithOptions>() == 0x5c);

/// `PGCmdFillBuffer`: 0x18 bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdFillBuffer {
    /// `bufferReference`.
    pub buffer_reference: u32,
    /// `rangeLocation`.
    pub range_location: u64,
    /// `rangeLength`.
    pub range_length: u64,
    /// `value`.
    pub value: u8,
    /// Tail padding: the record's size counts it, so a record must carry it.
    pub unknown15: [u8; 3],
}

const _: () = assert!(size_of::<CmdFillBuffer>() == 0x18);

/// `PGCmdFillBufferWithPattern`: 0x18 bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdFillBufferWithPattern {
    /// `bufferReference`.
    pub buffer_reference: u32,
    /// `rangeLocation`.
    pub range_location: u64,
    /// `rangeLength`.
    pub range_length: u64,
    /// `pattern`.
    pub pattern: u32,
}

const _: () = assert!(size_of::<CmdFillBufferWithPattern>() == 0x18);

/// `PGCmdFillTextureWithBytes`: 0x4c bytes, packed.
///
/// The region's size comes before its origin in both fill records.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdFillTextureWithBytes {
    /// `textureReference`.
    pub texture_reference: u32,
    /// `level`.
    pub level: u16,
    /// `slice`.
    pub slice: u16,
    /// `regionSize`.
    pub region_size: MtlSizeRecord,
    /// `regionOrigin`.
    pub region_origin: MtlOriginRecord,
    /// `bufferReference`.
    pub buffer_reference: u32,
    /// `bufferOffset`.
    pub buffer_offset: u64,
    /// `length`.
    pub length: u64,
}

const _: () = assert!(size_of::<CmdFillTextureWithBytes>() == 0x4c);

/// `PGCmdFillTextureWithColor`: 0x5c bytes, packed.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
pub struct CmdFillTextureWithColor {
    /// `textureReference`.
    pub texture_reference: u32,
    /// `level`.
    pub level: u16,
    /// `slice`.
    pub slice: u16,
    /// `regionSize`.
    pub region_size: MtlSizeRecord,
    /// `regionOrigin`.
    pub region_origin: MtlOriginRecord,
    /// `color`.
    pub color: ClearColorRecord,
    /// `pixelFormat`.
    pub pixel_format: u16,
    /// Tail padding: the record's size counts it, so a record must carry it.
    pub unknown5a: [u8; 2],
}

const _: () = assert!(size_of::<CmdFillTextureWithColor>() == 0x5c);

/// `PGCmdOptimizeResetIndirectBuffer`: 0x14 bytes, packed.
///
/// Read by both the reset and the optimize commands.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdOptimizeResetIndirectBuffer {
    /// `indirectCommandBufferReference`.
    pub indirect_command_buffer_reference: u32,
    /// `rangeLocation`.
    pub range_location: u64,
    /// `rangeLength`.
    pub range_length: u64,
}

const _: () = assert!(size_of::<CmdOptimizeResetIndirectBuffer>() == 0x14);

/// `PGCmdComputePipelineStateInfo`: 0x10 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdComputePipelineStateInfo {
    /// `pipelineStateReference`.
    pub pipeline_state_reference: u32,
    /// `replyBufferReference`.
    pub reply_buffer_reference: u32,
    /// `replyBufferOffset`.
    pub reply_buffer_offset: u64,
}

const _: () = assert!(size_of::<CmdComputePipelineStateInfo>() == 0x10);

/// `PGCmdHeapTextureSizeAndAlignWithDescriptor`: 0x2c bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdHeapTextureSizeAndAlignWithDescriptor {
    /// `descriptor`.
    pub descriptor: SerializedTextureDescriptor,
    /// `replyBufferReference`.
    pub reply_buffer_reference: u32,
    /// `replyBufferOffset`.
    pub reply_buffer_offset: u64,
}

const _: () = assert!(size_of::<CmdHeapTextureSizeAndAlignWithDescriptor>() == 0x2c);

/// `PGCmdHeapTextureSizeAndAlignWithDescriptor2`: 0x34 bytes, `#pragma pack(4)`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdHeapTextureSizeAndAlignWithDescriptor2 {
    /// `descriptor`.
    pub descriptor: SerializedTextureDescriptor2,
    /// Padding the layout leaves; read with the record, never interpreted.
    pub unknown27: [u8; 1],
    /// `replyBufferReference`.
    pub reply_buffer_reference: u32,
    /// `replyBufferOffset`.
    pub reply_buffer_offset: u64,
}

const _: () = assert!(size_of::<CmdHeapTextureSizeAndAlignWithDescriptor2>() == 0x34);

/// `PGCmdGetRasterizationRateMapInfo`: 0x18 bytes.
///
/// field10 is not read.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdGetRasterizationRateMapInfo {
    /// `rasterizationRateMapReference`.
    pub rasterization_rate_map_reference: u32,
    /// `replyBufferReference`.
    pub reply_buffer_reference: u32,
    /// `replyBufferOffset`.
    pub reply_buffer_offset: u64,
    /// `field10`.
    pub field_10: u64,
}

const _: () = assert!(size_of::<CmdGetRasterizationRateMapInfo>() == 0x18);

/// `PGCmdCopyRasterizationRateMapParameterBuffer`: 0x10 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdCopyRasterizationRateMapParameterBuffer {
    /// `rasterizationRateMapReference`.
    pub rasterization_rate_map_reference: u32,
    /// `bufferReference`.
    pub buffer_reference: u32,
    /// `bufferOffset`.
    pub buffer_offset: u64,
}

const _: () = assert!(size_of::<CmdCopyRasterizationRateMapParameterBuffer>() == 0x10);

/// `PGCmdMapCoordinates`: 0x1c bytes, packed.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
pub struct CmdMapCoordinates {
    /// `rasterizationRateMapReference`.
    pub rasterization_rate_map_reference: u32,
    /// `replyBufferReference`.
    pub reply_buffer_reference: u32,
    /// `replyBufferOffset`.
    pub reply_buffer_offset: u64,
    /// `layerIndex`.
    pub layer_index: u32,
    /// `coordinates`.
    pub coordinates: Coordinate2DRecord,
}

const _: () = assert!(size_of::<CmdMapCoordinates>() == 0x1c);

/// `PGCmdMapCoordinatesMultiple`: 0x20 bytes.
///
/// field10 is not read.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdMapCoordinatesMultiple {
    /// `rasterizationRateMapReference`.
    pub rasterization_rate_map_reference: u32,
    /// `replyBufferReference`.
    pub reply_buffer_reference: u32,
    /// `replyBufferOffset`.
    pub reply_buffer_offset: u64,
    /// `field10`.
    pub field_10: u64,
    /// `layerIndex`.
    pub layer_index: u32,
    /// `coordinateCount`.
    pub coordinate_count: u32,
}

const _: () = assert!(size_of::<CmdMapCoordinatesMultiple>() == 0x20);

/// `PGCmdRenderPipelineStateInfo`: 0x10 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdRenderPipelineStateInfo {
    /// `pipelineStateReference`.
    pub pipeline_state_reference: u32,
    /// `replyBufferReference`.
    pub reply_buffer_reference: u32,
    /// `replyBufferOffset`.
    pub reply_buffer_offset: u64,
}

const _: () = assert!(size_of::<CmdRenderPipelineStateInfo>() == 0x10);

/// `PGCmdPipelineStateImageBlockMemoryLength`: 0x28 bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdPipelineStateImageBlockMemoryLength {
    /// `pipelineStateReference`.
    pub pipeline_state_reference: u32,
    /// `imageblockDimensions`.
    pub imageblock_dimensions: MtlSizeRecord,
    /// `replyBufferReference`.
    pub reply_buffer_reference: u32,
    /// `replyBufferOffset`.
    pub reply_buffer_offset: u64,
}

const _: () = assert!(size_of::<CmdPipelineStateImageBlockMemoryLength>() == 0x28);

/// `PGCmdObjectUniqueIdentifier`: 0x14 bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdObjectUniqueIdentifier {
    /// `objectReference`.
    pub object_reference: u32,
    /// `objectType`.
    pub object_type: u32,
    /// `replyBufferReference`.
    pub reply_buffer_reference: u32,
    /// `replyBufferOffset`.
    pub reply_buffer_offset: u64,
}

const _: () = assert!(size_of::<CmdObjectUniqueIdentifier>() == 0x14);

/// `PGCmdBufferHostResourceInfo`: 0x10 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdBufferHostResourceInfo {
    /// `bufferReference`.
    pub buffer_reference: u32,
    /// `replyBufferReference`.
    pub reply_buffer_reference: u32,
    /// `replyBufferOffset`.
    pub reply_buffer_offset: u64,
}

const _: () = assert!(size_of::<CmdBufferHostResourceInfo>() == 0x10);

/// `PGCmdTextureHostResourceInfo`: 0x10 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdTextureHostResourceInfo {
    /// `textureReference`.
    pub texture_reference: u32,
    /// `replyBufferReference`.
    pub reply_buffer_reference: u32,
    /// `replyBufferOffset`.
    pub reply_buffer_offset: u64,
}

const _: () = assert!(size_of::<CmdTextureHostResourceInfo>() == 0x10);

/// `PGCmdHeapHostResourceInfo`: 0x10 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdHeapHostResourceInfo {
    /// `heapReference`.
    pub heap_reference: u32,
    /// `replyBufferReference`.
    pub reply_buffer_reference: u32,
    /// `replyBufferOffset`.
    pub reply_buffer_offset: u64,
}

const _: () = assert!(size_of::<CmdHeapHostResourceInfo>() == 0x10);

/// `PGCmdSamplerStateHostResourceInfo`: 0x10 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSamplerStateHostResourceInfo {
    /// `samplerStateReference`.
    pub sampler_state_reference: u32,
    /// `replyBufferReference`.
    pub reply_buffer_reference: u32,
    /// `replyBufferOffset`.
    pub reply_buffer_offset: u64,
}

const _: () = assert!(size_of::<CmdSamplerStateHostResourceInfo>() == 0x10);

/// `PGCmdICBHostResourceInfo`: 0x10 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdICBHostResourceInfo {
    /// `indirectCommandBufferReference`.
    pub indirect_command_buffer_reference: u32,
    /// `replyBufferReference`.
    pub reply_buffer_reference: u32,
    /// `replyBufferOffset`.
    pub reply_buffer_offset: u64,
}

const _: () = assert!(size_of::<CmdICBHostResourceInfo>() == 0x10);

/// `PGCmdObjectHostInfo`: 0x10 bytes.
///
/// Read for render and compute pipeline states and depth-stencil states.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdObjectHostInfo {
    /// `objectReference`.
    pub object_reference: u32,
    /// `replyBufferReference`.
    pub reply_buffer_reference: u32,
    /// `replyBufferOffset`.
    pub reply_buffer_offset: u64,
}

const _: () = assert!(size_of::<CmdObjectHostInfo>() == 0x10);

/// `PGReplyComputePipelineStateInfo`: 0x1c bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct ReplyComputePipelineStateInfo {
    /// `maxTotalThreadsPerThreadgroup`.
    pub max_total_threads_per_threadgroup: u64,
    /// `threadExecutionWidth`.
    pub thread_execution_width: u64,
    /// `staticThreadgroupMemoryLength`.
    pub static_threadgroup_memory_length: u64,
    /// `supportIndirectCommandBuffers`.
    pub support_indirect_command_buffers: u8,
    /// Tail padding: part of the reply written, content unspecified.
    pub unknown19: [u8; 3],
}

const _: () = assert!(size_of::<ReplyComputePipelineStateInfo>() == 0x1c);

/// `PGReplyHeapTextureSizeAndAlign`: 0x10 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct ReplyHeapTextureSizeAndAlign {
    /// `size`.
    pub size: u64,
    /// `align`.
    pub align: u64,
}

const _: () = assert!(size_of::<ReplyHeapTextureSizeAndAlign>() == 0x10);

/// `PGReplyRasterizationRateMapInfo`: 0x14 bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct ReplyRasterizationRateMapInfo {
    /// `parameterBufferSize`.
    pub parameter_buffer_size: u64,
    /// `parameterBufferAlign`.
    pub parameter_buffer_align: u64,
    /// `physicalGranularityWidth`.
    pub physical_granularity_width: u16,
    /// `physicalGranularityHeight`.
    pub physical_granularity_height: u16,
}

const _: () = assert!(size_of::<ReplyRasterizationRateMapInfo>() == 0x14);

/// `PGCmd2DSize16`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct Cmd2DSize16 {
    /// `width`.
    pub width: u16,
    /// `height`.
    pub height: u16,
}

const _: () = assert!(size_of::<Cmd2DSize16>() == 0x4);

/// `PGReplyRasterizationRateMapCoordinate`: 0x8 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub struct ReplyRasterizationRateMapCoordinate {
    /// `x`.
    pub x: f32,
    /// `y`.
    pub y: f32,
}

const _: () = assert!(size_of::<ReplyRasterizationRateMapCoordinate>() == 0x8);

/// `PGReplyRenderPipelineStateInfo`: 0xc bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct ReplyRenderPipelineStateInfo {
    /// `maxTotalThreadsPerThreadgroup`.
    pub max_total_threads_per_threadgroup: u32,
    /// `imageblockSampleLength`.
    pub imageblock_sample_length: u32,
    /// `threadgroupSizeMatchesTileSize`.
    pub threadgroup_size_matches_tile_size: u8,
    /// `supportIndirectCommandBuffers`.
    pub support_indirect_command_buffers: u8,
    /// Tail padding: part of the reply written, content unspecified.
    pub unknown0a: [u8; 2],
}

const _: () = assert!(size_of::<ReplyRenderPipelineStateInfo>() == 0xc);

/// `PGReplyPipelineStateImageBlockMemoryLength`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct ReplyPipelineStateImageBlockMemoryLength {
    /// `imageblockMemoryLength`.
    pub imageblock_memory_length: u32,
}

const _: () = assert!(size_of::<ReplyPipelineStateImageBlockMemoryLength>() == 0x4);

/// `PGReplyUniqueIdentifier_DEPRECATED`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct ReplyUniqueIdentifierDeprecated {
    /// `uniqueIdentifier`.
    pub unique_identifier: u64,
}

const _: () = assert!(size_of::<ReplyUniqueIdentifierDeprecated>() == 0x8);

/// `PGReplyBufferHostResourceInfo`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct ReplyBufferHostResourceInfo {
    /// `gpuAddress`.
    pub gpu_address: u64,
}

const _: () = assert!(size_of::<ReplyBufferHostResourceInfo>() == 0x8);

/// `PGReplyHostObjectInfo`: 0x10 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct ReplyHostObjectInfo {
    /// `gpuResourceID`.
    pub gpu_resource_id: ResourceIdRecord,
    /// `resourceIndex`.
    pub resource_index: u64,
}

const _: () = assert!(size_of::<ReplyHostObjectInfo>() == 0x10);

/// `PGCmdDispatchThreadgroups`: 0x30 bytes.
///
/// The dispatch sizes, as Metal's own struct: the decoder hands each field to
/// the encoder whole.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdDispatchThreadgroups {
    /// `threadgroupsPerGrid`.
    pub threadgroups_per_grid: MtlSizeRecord,
    /// `threadsPerThreadgroup`.
    pub threads_per_threadgroup: MtlSizeRecord,
}

const _: () = assert!(size_of::<CmdDispatchThreadgroups>() == 0x30);

/// `PGCmdDispatchThreads`: 0x30 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdDispatchThreads {
    /// `threadsPerGrid`.
    pub threads_per_grid: MtlSizeRecord,
    /// `threadsPerThreadgroup`.
    pub threads_per_threadgroup: MtlSizeRecord,
}

const _: () = assert!(size_of::<CmdDispatchThreads>() == 0x30);

/// `PGCmdSetComputeStageInRegion`: 0x30 bytes.
///
/// The size comes first and the origin second, the reverse of MTLRegion. The
/// decoder copies the size in before the origin, which is the order the
/// target's stores into the region have.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetComputeStageInRegion {
    /// `regionSize`.
    pub region_size: MtlSizeRecord,
    /// `regionOrigin`.
    pub region_origin: MtlOriginRecord,
}

const _: () = assert!(size_of::<CmdSetComputeStageInRegion>() == 0x30);

/// `PGCmdSetImageblockWidth`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetImageblockWidth {
    /// `width`.
    pub width: u32,
    /// `height`.
    pub height: u32,
}

const _: () = assert!(size_of::<CmdSetImageblockWidth>() == 0x8);

/// `PGCmdDispatchThreadgroupsIndirect`: 0x24 bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdDispatchThreadgroupsIndirect {
    /// `threadsPerThreadgroup`.
    pub threads_per_threadgroup: MtlSizeRecord,
    /// `indirectBufferOffset`.
    pub indirect_buffer_offset: u64,
    /// `indirectBufferReference`.
    pub indirect_buffer_reference: u32,
}

const _: () = assert!(size_of::<CmdDispatchThreadgroupsIndirect>() == 0x24);

/// `PGCmdDispatchThreadsIndirect`: 0xc bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdDispatchThreadsIndirect {
    /// `indirectBufferOffset`.
    pub indirect_buffer_offset: u64,
    /// `indirectBufferReference`.
    pub indirect_buffer_reference: u32,
}

const _: () = assert!(size_of::<CmdDispatchThreadsIndirect>() == 0xc);

/// `PGCmdSetComputeStageInRegionIndirect`: 0xc bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdSetComputeStageInRegionIndirect {
    /// `indirectBufferReference`.
    pub indirect_buffer_reference: u32,
    /// `indirectBufferOffset`.
    pub indirect_buffer_offset: u64,
}

const _: () = assert!(size_of::<CmdSetComputeStageInRegionIndirect>() == 0xc);

/// `PGCmdSetThreadgroupMemoryLength`: 0xc bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdSetThreadgroupMemoryLength {
    /// `length`.
    pub length: u64,
    /// `index`.
    pub index: u32,
}

const _: () = assert!(size_of::<CmdSetThreadgroupMemoryLength>() == 0xc);

/// `PGCmdMemoryBarrierResources`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdMemoryBarrierResources {
    /// `count`.
    pub count: u32,
}

const _: () = assert!(size_of::<CmdMemoryBarrierResources>() == 0x4);

/// `PGCmdEncodeEndDoWhile`: 0x14 bytes, packed.
///
/// The three conditional-jump records: a buffer reference, the offset of the
/// value in it, the comparison and the value compared against.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdEncodeEndDoWhile {
    /// `bufferReference`.
    pub buffer_reference: u32,
    /// `offset`.
    pub offset: u64,
    /// `comparison`.
    pub comparison: u32,
    /// `referenceValue`.
    pub reference_value: u32,
}

const _: () = assert!(size_of::<CmdEncodeEndDoWhile>() == 0x14);

/// `PGCmdEncodeStartWhile`: 0x14 bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdEncodeStartWhile {
    /// `bufferReference`.
    pub buffer_reference: u32,
    /// `offset`.
    pub offset: u64,
    /// `comparison`.
    pub comparison: u32,
    /// `referenceValue`.
    pub reference_value: u32,
}

const _: () = assert!(size_of::<CmdEncodeStartWhile>() == 0x14);

/// `PGCmdEncodeStartIf`: 0x14 bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdEncodeStartIf {
    /// `bufferReference`.
    pub buffer_reference: u32,
    /// `offset`.
    pub offset: u64,
    /// `comparison`.
    pub comparison: u32,
    /// `referenceValue`.
    pub reference_value: u32,
}

const _: () = assert!(size_of::<CmdEncodeStartIf>() == 0x14);

/// `PGCmdMemoryBarrierScope`: 0x4 bytes.
///
/// A 4-byte command word with 16 bits used: the compute encoder takes only a
/// scope, where PGCmdRenderMemoryBarrierScope spends the top half on stages.
/// The width is load-bearing -- a 2-byte struct arrives in a 64-bit register
/// and masks with `and x2, x0, #0xffff` where the target masks 32-bit.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdMemoryBarrierScope {
    /// The word holding the bit-field `uint32_t scope : 16` in its low 16 bits (the field name
    /// is ours; read it through [`CmdMemoryBarrierScope::scope`]).
    pub bits: u32,
}

const _: () = assert!(size_of::<CmdMemoryBarrierScope>() == 0x4);

impl CmdMemoryBarrierScope {
    /// `scope`: the low 16 bits of the word (`uint32_t scope : 16`); the high 16 are read with
    /// the record and ignored.
    pub fn scope(&self) -> u16 {
        self.bits.low_u16()
    }
}

/// `PGCmdSetComputePassDescriptorDispatchType`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetComputePassDescriptorDispatchType {
    /// `dispatchType`.
    pub dispatch_type: u32,
}

const _: () = assert!(size_of::<CmdSetComputePassDescriptorDispatchType>() == 0x4);

/// `PGCmdSetBuffers`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetBuffers {
    /// `rangeLocation`.
    pub range_location: u32,
    /// `rangeLength`.
    pub range_length: u32,
}

const _: () = assert!(size_of::<CmdSetBuffers>() == 0x8);

/// `PGCmdSetBufferEntry`: 0xc bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdSetBufferEntry {
    /// `bufferReference`.
    pub buffer_reference: u32,
    /// `offset`.
    pub offset: u64,
}

const _: () = assert!(size_of::<CmdSetBufferEntry>() == 0xc);

/// `PGCmdSetBufferOffset`: 0xc bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdSetBufferOffset {
    /// `index`.
    pub index: u32,
    /// `offset`.
    pub offset: u64,
}

const _: () = assert!(size_of::<CmdSetBufferOffset>() == 0xc);

/// `PGCmdSetBuffersWithStride`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetBuffersWithStride {
    /// `rangeLocation`.
    pub range_location: u32,
    /// `rangeLength`.
    pub range_length: u32,
}

const _: () = assert!(size_of::<CmdSetBuffersWithStride>() == 0x8);

/// `PGCmdSetBufferEntryWithStride`: 0x14 bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdSetBufferEntryWithStride {
    /// `bufferReference`.
    pub buffer_reference: u32,
    /// `offset`.
    pub offset: u64,
    /// `attributeStride`.
    pub attribute_stride: u64,
}

const _: () = assert!(size_of::<CmdSetBufferEntryWithStride>() == 0x14);

/// `PGCmdSetBufferOffsetWithStride`: 0x14 bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdSetBufferOffsetWithStride {
    /// `index`.
    pub index: u32,
    /// `offset`.
    pub offset: u64,
    /// `attributeStride`.
    pub attribute_stride: u64,
}

const _: () = assert!(size_of::<CmdSetBufferOffsetWithStride>() == 0x14);

/// `PGCmdSetSamplerStates`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetSamplerStates {
    /// `rangeLocation`.
    pub range_location: u32,
    /// `rangeLength`.
    pub range_length: u32,
}

const _: () = assert!(size_of::<CmdSetSamplerStates>() == 0x8);

/// `PGCmdSetSamplerStatesLODClamp`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetSamplerStatesLODClamp {
    /// `rangeLocation`.
    pub range_location: u32,
    /// `rangeLength`.
    pub range_length: u32,
}

const _: () = assert!(size_of::<CmdSetSamplerStatesLODClamp>() == 0x8);

/// `PGCmdSetSamplerStatesLODClampEntry`: 0xc bytes.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub struct CmdSetSamplerStatesLODClampEntry {
    /// `samplerStateReference`.
    pub sampler_state_reference: u32,
    /// `lodMinClamp`.
    pub lod_min_clamp: f32,
    /// `lodMaxClamp`.
    pub lod_max_clamp: f32,
}

const _: () = assert!(size_of::<CmdSetSamplerStatesLODClampEntry>() == 0xc);

/// `PGCmdSetTextures`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetTextures {
    /// `rangeLocation`.
    pub range_location: u32,
    /// `rangeLength`.
    pub range_length: u32,
}

const _: () = assert!(size_of::<CmdSetTextures>() == 0x8);

/// `PGCmdSetRenderPipelineState`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetRenderPipelineState {
    /// `pipelineStateReference`.
    pub pipeline_state_reference: u32,
}

const _: () = assert!(size_of::<CmdSetRenderPipelineState>() == 0x4);

/// `PGCmdUseHeaps`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdUseHeaps {
    /// `count`.
    pub count: u32,
}

const _: () = assert!(size_of::<CmdUseHeaps>() == 0x4);

/// `PGCmdUseResources`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdUseResources {
    /// `count`.
    pub count: u32,
    /// `usage`.
    pub usage: u32,
}

const _: () = assert!(size_of::<CmdUseResources>() == 0x8);

/// `PGCmdExecuteCommandsInBufferRanged`: 0x14 bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdExecuteCommandsInBufferRanged {
    /// `indirectCommandBufferReference`.
    pub indirect_command_buffer_reference: u32,
    /// `rangeLocation`.
    pub range_location: u64,
    /// `rangeLength`.
    pub range_length: u64,
}

const _: () = assert!(size_of::<CmdExecuteCommandsInBufferRanged>() == 0x14);

/// `PGCmdExecuteCommandsInBuffer`: 0x10 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdExecuteCommandsInBuffer {
    /// `indirectCommandBufferReference`.
    pub indirect_command_buffer_reference: u32,
    /// `indirectRangeBufferReference`.
    pub indirect_range_buffer_reference: u32,
    /// `indirectBufferOffset`.
    pub indirect_buffer_offset: u64,
}

const _: () = assert!(size_of::<CmdExecuteCommandsInBuffer>() == 0x10);

/// `PGCmdDrawPrimitives64`: 0x14 bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdDrawPrimitives64 {
    /// `primitiveType`.
    pub primitive_type: u32,
    /// `vertexStart`.
    pub vertex_start: u64,
    /// `vertexCount`.
    pub vertex_count: u64,
}

const _: () = assert!(size_of::<CmdDrawPrimitives64>() == 0x14);

/// `PGCmdDrawPrimitives16`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdDrawPrimitives16 {
    /// `primitiveType`.
    pub primitive_type: u32,
    /// `vertexStart`.
    pub vertex_start: u16,
    /// `vertexCount`.
    pub vertex_count: u16,
}

const _: () = assert!(size_of::<CmdDrawPrimitives16>() == 0x8);

/// `PGCmdDrawInstancedPrimitives64`: 0x1c bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdDrawInstancedPrimitives64 {
    /// `vertexStart`.
    pub vertex_start: u64,
    /// `vertexCount`.
    pub vertex_count: u64,
    /// `instanceCount`.
    pub instance_count: u64,
    /// `primitiveType`.
    pub primitive_type: u16,
    /// Tail padding: the record's size counts it, so a record must carry it.
    pub unknown1a: [u8; 2],
}

const _: () = assert!(size_of::<CmdDrawInstancedPrimitives64>() == 0x1c);

/// `PGCmdDrawInstancedPrimitives16`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdDrawInstancedPrimitives16 {
    /// `vertexStart`.
    pub vertex_start: u16,
    /// `vertexCount`.
    pub vertex_count: u16,
    /// `instanceCount`.
    pub instance_count: u16,
    /// `primitiveType`.
    pub primitive_type: u16,
}

const _: () = assert!(size_of::<CmdDrawInstancedPrimitives16>() == 0x8);

/// `PGCmdDrawInstancedBasePrimitives64`: 0x24 bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdDrawInstancedBasePrimitives64 {
    /// `vertexStart`.
    pub vertex_start: u64,
    /// `vertexCount`.
    pub vertex_count: u64,
    /// `instanceCount`.
    pub instance_count: u64,
    /// `baseInstance`.
    pub base_instance: u64,
    /// `primitiveType`.
    pub primitive_type: u16,
    /// Tail padding: the record's size counts it, so a record must carry it.
    pub unknown22: [u8; 2],
}

const _: () = assert!(size_of::<CmdDrawInstancedBasePrimitives64>() == 0x24);

/// `PGCmdDrawInstancedBasePrimitives16`: 0xc bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdDrawInstancedBasePrimitives16 {
    /// `vertexStart`.
    pub vertex_start: u16,
    /// `vertexCount`.
    pub vertex_count: u16,
    /// `instanceCount`.
    pub instance_count: u16,
    /// `baseInstance`.
    pub base_instance: u16,
    /// `primitiveType`.
    pub primitive_type: u16,
    /// Tail padding: the record's size counts it, so a record must carry it.
    pub unknown0a: [u8; 2],
}

const _: () = assert!(size_of::<CmdDrawInstancedBasePrimitives16>() == 0xc);

/// `PGCmdDrawIndexedPrimitives64`: 0x18 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdDrawIndexedPrimitives64 {
    /// `primitiveType`.
    pub primitive_type: u16,
    /// `indexType`.
    pub index_type: u16,
    /// `indexBufferReference`.
    pub index_buffer_reference: u32,
    /// `indexCount`.
    pub index_count: u64,
    /// `indexBufferOffset`.
    pub index_buffer_offset: u64,
}

const _: () = assert!(size_of::<CmdDrawIndexedPrimitives64>() == 0x18);

/// `PGCmdDrawIndexedPrimitives16`: 0xc bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdDrawIndexedPrimitives16 {
    /// `primitiveType`.
    pub primitive_type: u16,
    /// `indexType`.
    pub index_type: u16,
    /// `indexBufferReference`.
    pub index_buffer_reference: u32,
    /// `indexCount`.
    pub index_count: u16,
    /// `indexBufferOffset`.
    pub index_buffer_offset: u16,
}

const _: () = assert!(size_of::<CmdDrawIndexedPrimitives16>() == 0xc);

/// `PGCmdDrawIndexedInstancedPrimitives64`: 0x20 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdDrawIndexedInstancedPrimitives64 {
    /// `primitiveType`.
    pub primitive_type: u16,
    /// `indexType`.
    pub index_type: u16,
    /// `indexBufferReference`.
    pub index_buffer_reference: u32,
    /// `indexCount`.
    pub index_count: u64,
    /// `indexBufferOffset`.
    pub index_buffer_offset: u64,
    /// `instanceCount`.
    pub instance_count: u64,
}

const _: () = assert!(size_of::<CmdDrawIndexedInstancedPrimitives64>() == 0x20);

/// `PGCmdDrawIndexedInstancedPrimitives16`: 0x10 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdDrawIndexedInstancedPrimitives16 {
    /// `primitiveType`.
    pub primitive_type: u16,
    /// `indexType`.
    pub index_type: u16,
    /// `indexBufferReference`.
    pub index_buffer_reference: u32,
    /// `indexCount`.
    pub index_count: u16,
    /// `indexBufferOffset`.
    pub index_buffer_offset: u16,
    /// `instanceCount`.
    pub instance_count: u16,
    /// Tail padding: the record's size counts it, so a record must carry it.
    pub unknown0e: [u8; 2],
}

const _: () = assert!(size_of::<CmdDrawIndexedInstancedPrimitives16>() == 0x10);

/// `PGCmdDrawIndexedInstancedBasePrimitives64`: 0x30 bytes.
///
/// baseVertex is an NSInteger to Metal. Its signedness shows only in the
/// 16-bit pair: PGCmdDrawIndexedInstancedBasePrimitives16's is loaded
/// zero-extended (ldrh) and the _2 record's sign-extended (ldrsh). A 64-bit
/// load says neither, so the 64-bit pair repeats the 16-bit pair's choice.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdDrawIndexedInstancedBasePrimitives64 {
    /// `primitiveType`.
    pub primitive_type: u16,
    /// `indexType`.
    pub index_type: u16,
    /// `indexBufferReference`.
    pub index_buffer_reference: u32,
    /// `indexBufferOffset`.
    pub index_buffer_offset: u64,
    /// `indexCount`.
    pub index_count: u64,
    /// `instanceCount`.
    pub instance_count: u64,
    /// `baseVertex`.
    pub base_vertex: u64,
    /// `baseInstance`.
    pub base_instance: u64,
}

const _: () = assert!(size_of::<CmdDrawIndexedInstancedBasePrimitives64>() == 0x30);

/// `PGCmdDrawIndexedInstancedBasePrimitives16`: 0x14 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdDrawIndexedInstancedBasePrimitives16 {
    /// `primitiveType`.
    pub primitive_type: u16,
    /// `indexType`.
    pub index_type: u16,
    /// `indexBufferReference`.
    pub index_buffer_reference: u32,
    /// `indexBufferOffset`.
    pub index_buffer_offset: u16,
    /// `indexCount`.
    pub index_count: u16,
    /// `instanceCount`.
    pub instance_count: u16,
    /// `baseVertex`: unsigned, so the decoder zero-extends it (`ldrh`).
    // 0x0b zero-extends baseVertex where 0x1d sign-extends.
    pub base_vertex: u16,
    /// `baseInstance`.
    pub base_instance: u16,
    /// Tail padding: the record's size counts it, so a record must carry it.
    pub unknown12: [u8; 2],
}

const _: () = assert!(size_of::<CmdDrawIndexedInstancedBasePrimitives16>() == 0x14);

/// `PGCmdDrawIndexedInstancedBasePrimitives64_2`: 0x30 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdDrawIndexedInstancedBasePrimitives64_2 {
    /// `primitiveType`.
    pub primitive_type: u16,
    /// `indexType`.
    pub index_type: u16,
    /// `indexBufferReference`.
    pub index_buffer_reference: u32,
    /// `indexBufferOffset`.
    pub index_buffer_offset: u64,
    /// `indexCount`.
    pub index_count: u64,
    /// `instanceCount`.
    pub instance_count: u64,
    /// `baseVertex`.
    pub base_vertex: i64,
    /// `baseInstance`.
    pub base_instance: u64,
}

const _: () = assert!(size_of::<CmdDrawIndexedInstancedBasePrimitives64_2>() == 0x30);

/// `PGCmdDrawIndexedInstancedBasePrimitives16_2`: 0x14 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdDrawIndexedInstancedBasePrimitives16_2 {
    /// `primitiveType`.
    pub primitive_type: u16,
    /// `indexType`.
    pub index_type: u16,
    /// `indexBufferReference`.
    pub index_buffer_reference: u32,
    /// `indexBufferOffset`.
    pub index_buffer_offset: u16,
    /// `indexCount`.
    pub index_count: u16,
    /// `instanceCount`.
    pub instance_count: u16,
    /// `baseVertex`: signed, so the decoder sign-extends it (`ldrsh`).
    // 0x1d sign-extends baseVertex where 0x0b zero-extends.
    pub base_vertex: i16,
    /// `baseInstance`.
    pub base_instance: u16,
    /// Tail padding: the record's size counts it, so a record must carry it.
    pub unknown12: [u8; 2],
}

const _: () = assert!(size_of::<CmdDrawIndexedInstancedBasePrimitives16_2>() == 0x14);

/// `PGCmdDrawPatches64`: 0x30 bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdDrawPatches64 {
    /// `patchStart`.
    pub patch_start: u64,
    /// `patchCount`.
    pub patch_count: u64,
    /// `patchIndexBufferReference`.
    pub patch_index_buffer_reference: u32,
    /// `patchIndexBufferOffset`.
    pub patch_index_buffer_offset: u64,
    /// `instanceCount`.
    pub instance_count: u64,
    /// `baseInstance`.
    pub base_instance: u64,
    /// `numberOfPatchControlPoints`.
    pub number_of_patch_control_points: u16,
    /// Tail padding: the record's size counts it, so a record must carry it.
    pub unknown2e: [u8; 2],
}

const _: () = assert!(size_of::<CmdDrawPatches64>() == 0x30);

/// `PGCmdDrawPatches16`: 0x10 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdDrawPatches16 {
    /// `patchStart`.
    pub patch_start: u16,
    /// `patchCount`.
    pub patch_count: u16,
    /// `patchIndexBufferReference`.
    pub patch_index_buffer_reference: u32,
    /// `patchIndexBufferOffset`.
    pub patch_index_buffer_offset: u16,
    /// `instanceCount`.
    pub instance_count: u16,
    /// `baseInstance`.
    pub base_instance: u16,
    /// `numberOfPatchControlPoints`.
    pub number_of_patch_control_points: u16,
}

const _: () = assert!(size_of::<CmdDrawPatches16>() == 0x10);

/// `PGCmdDrawIndexedPatches64`: 0x3c bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdDrawIndexedPatches64 {
    /// `patchStart`.
    pub patch_start: u64,
    /// `patchCount`.
    pub patch_count: u64,
    /// `patchIndexBufferReference`.
    pub patch_index_buffer_reference: u32,
    /// `patchIndexBufferOffset`.
    pub patch_index_buffer_offset: u64,
    /// `controlPointIndexBufferReference`.
    pub control_point_index_buffer_reference: u32,
    /// `controlPointIndexBufferOffset`.
    pub control_point_index_buffer_offset: u64,
    /// `instanceCount`.
    pub instance_count: u64,
    /// `baseInstance`.
    pub base_instance: u64,
    /// `numberOfPatchControlPoints`.
    pub number_of_patch_control_points: u16,
    /// Tail padding: the record's size counts it, so a record must carry it.
    pub unknown3a: [u8; 2],
}

const _: () = assert!(size_of::<CmdDrawIndexedPatches64>() == 0x3c);

/// `PGCmdDrawIndexedPatches16`: 0x18 bytes, packed.
///
/// controlPointIndexBufferReference is 32 bits at 0xa.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdDrawIndexedPatches16 {
    /// `patchStart`.
    pub patch_start: u16,
    /// `patchCount`.
    pub patch_count: u16,
    /// `patchIndexBufferReference`.
    pub patch_index_buffer_reference: u32,
    /// `patchIndexBufferOffset`.
    pub patch_index_buffer_offset: u16,
    /// `controlPointIndexBufferReference`.
    pub control_point_index_buffer_reference: u32,
    /// `controlPointIndexBufferOffset`.
    pub control_point_index_buffer_offset: u16,
    /// `instanceCount`.
    pub instance_count: u16,
    /// `baseInstance`.
    pub base_instance: u16,
    /// `numberOfPatchControlPoints`.
    pub number_of_patch_control_points: u16,
    /// Tail padding: the record's size counts it, so a record must carry it.
    pub unknown16: [u8; 2],
}

const _: () = assert!(size_of::<CmdDrawIndexedPatches16>() == 0x18);

/// `PGCmdDrawPrimitivesIndirect`: 0x10 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdDrawPrimitivesIndirect {
    /// `indirectBufferOffset`.
    pub indirect_buffer_offset: u64,
    /// `indirectBufferReference`.
    pub indirect_buffer_reference: u32,
    /// `primitiveType`.
    pub primitive_type: u16,
    /// Tail padding: the record's size counts it, so a record must carry it.
    pub unknown0e: [u8; 2],
}

const _: () = assert!(size_of::<CmdDrawPrimitivesIndirect>() == 0x10);

/// `PGCmdDrawIndexedPrimitivesIndirect`: 0x1c bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdDrawIndexedPrimitivesIndirect {
    /// `primitiveType`.
    pub primitive_type: u16,
    /// `indexType`.
    pub index_type: u16,
    /// `indexBufferReference`.
    pub index_buffer_reference: u32,
    /// `indirectBufferReference`.
    pub indirect_buffer_reference: u32,
    /// `indexBufferOffset`.
    pub index_buffer_offset: u64,
    /// `indirectBufferOffset`.
    pub indirect_buffer_offset: u64,
}

const _: () = assert!(size_of::<CmdDrawIndexedPrimitivesIndirect>() == 0x1c);

/// `PGCmdDrawPatchesIndirect`: 0x1c bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdDrawPatchesIndirect {
    /// `patchIndexBufferReference`.
    pub patch_index_buffer_reference: u32,
    /// `indirectBufferReference`.
    pub indirect_buffer_reference: u32,
    /// `patchIndexBufferOffset`.
    pub patch_index_buffer_offset: u64,
    /// `indirectBufferOffset`.
    pub indirect_buffer_offset: u64,
    /// `numberOfPatchControlPoints`.
    pub number_of_patch_control_points: u16,
    /// Tail padding: the record's size counts it, so a record must carry it.
    pub unknown1a: [u8; 2],
}

const _: () = assert!(size_of::<CmdDrawPatchesIndirect>() == 0x1c);

/// `PGCmdDrawIndexedPatchesIndirect`: 0x28 bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdDrawIndexedPatchesIndirect {
    /// `patchIndexBufferReference`.
    pub patch_index_buffer_reference: u32,
    /// `controlPointIndexBufferReference`.
    pub control_point_index_buffer_reference: u32,
    /// `indirectBufferReference`.
    pub indirect_buffer_reference: u32,
    /// `patchIndexBufferOffset`.
    pub patch_index_buffer_offset: u64,
    /// `controlPointIndexBufferOffset`.
    pub control_point_index_buffer_offset: u64,
    /// `indirectBufferOffset`.
    pub indirect_buffer_offset: u64,
    /// `numberOfPatchControlPoints`.
    pub number_of_patch_control_points: u16,
    /// Tail padding: the record's size counts it, so a record must carry it.
    pub unknown26: [u8; 2],
}

const _: () = assert!(size_of::<CmdDrawIndexedPatchesIndirect>() == 0x28);

/// `PGCmdRenderUpdateFence`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdRenderUpdateFence {
    /// `fenceReference`.
    pub fence_reference: u32,
    /// `afterStages`.
    pub after_stages: u32,
}

const _: () = assert!(size_of::<CmdRenderUpdateFence>() == 0x8);

/// `PGCmdRenderWaitForFence`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdRenderWaitForFence {
    /// `fenceReference`.
    pub fence_reference: u32,
    /// `beforeStages`.
    pub before_stages: u32,
}

const _: () = assert!(size_of::<CmdRenderWaitForFence>() == 0x8);

/// `PGCmdSetImageBlockSampleLength`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetImageBlockSampleLength {
    /// `imageblockSampleLength`.
    pub imageblock_sample_length: u32,
}

const _: () = assert!(size_of::<CmdSetImageBlockSampleLength>() == 0x4);

/// `PGCmdSetThreadGroupMemoryLength`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetThreadGroupMemoryLength {
    /// `threadgroupMemoryLength`.
    pub threadgroup_memory_length: u32,
}

const _: () = assert!(size_of::<CmdSetThreadGroupMemoryLength>() == 0x4);

/// `PGCmdRenderMemoryBarrierScope`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdRenderMemoryBarrierScope {
    /// `scope`.
    pub scope: u16,
    /// `afterStages`.
    pub after_stages: u8,
    /// `beforeStages`.
    pub before_stages: u8,
}

const _: () = assert!(size_of::<CmdRenderMemoryBarrierScope>() == 0x4);

/// `PGCmdSetBlendColor`: 0x10 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub struct CmdSetBlendColor {
    /// `red`.
    pub red: f32,
    /// `green`.
    pub green: f32,
    /// `blue`.
    pub blue: f32,
    /// `alpha`.
    pub alpha: f32,
}

const _: () = assert!(size_of::<CmdSetBlendColor>() == 0x10);

/// `PGCmdSetDepthBias`: 0xc bytes.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub struct CmdSetDepthBias {
    /// `depthBias`.
    pub depth_bias: f32,
    /// `slopeScale`.
    pub slope_scale: f32,
    /// `clamp`.
    pub clamp: f32,
}

const _: () = assert!(size_of::<CmdSetDepthBias>() == 0xc);

/// `PGCmdSetFloatValue`: 0x4 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub struct CmdSetFloatValue {
    /// `value`.
    pub value: f32,
}

const _: () = assert!(size_of::<CmdSetFloatValue>() == 0x4);

/// `PGCmdSetStencilRef`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetStencilRef {
    /// `frontReferenceValue`.
    pub front_reference_value: u32,
    /// `backReferenceValue`.
    pub back_reference_value: u32,
}

const _: () = assert!(size_of::<CmdSetStencilRef>() == 0x8);

/// `PGCmdSetVisibiliyResultMode`: 0x10 bytes.
///
/// Spelled as the mangled symbol spells it, typo and all.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetVisibiliyResultMode {
    /// `offset`.
    pub offset: u64,
    /// `mode`.
    pub mode: u64,
}

const _: () = assert!(size_of::<CmdSetVisibiliyResultMode>() == 0x10);

/// `PGCmdSetProvokingVertexMode`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetProvokingVertexMode {
    /// `mode`.
    pub mode: u32,
}

const _: () = assert!(size_of::<CmdSetProvokingVertexMode>() == 0x4);

/// `PGCmdSetTransformFeedbackState`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetTransformFeedbackState {
    /// `state`.
    pub state: u32,
}

const _: () = assert!(size_of::<CmdSetTransformFeedbackState>() == 0x4);

/// `PGCmdSetViewportTransformEnabled`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetViewportTransformEnabled {
    /// `enabled`.
    pub enabled: u32,
}

const _: () = assert!(size_of::<CmdSetViewportTransformEnabled>() == 0x4);

/// `PGCmdSetTriangleFillModeFrontBack`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetTriangleFillModeFrontBack {
    /// `frontFillMode`.
    pub front_fill_mode: u16,
    /// `backFillMode`.
    pub back_fill_mode: u16,
}

const _: () = assert!(size_of::<CmdSetTriangleFillModeFrontBack>() == 0x4);

/// `PGCmdSetVertexAmplificationMode`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetVertexAmplificationMode {
    /// `mode`.
    pub mode: u32,
    /// `value`.
    pub value: u32,
}

const _: () = assert!(size_of::<CmdSetVertexAmplificationMode>() == 0x8);

/// `PGCmdSetDefaultSampleCount`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetDefaultSampleCount {
    /// `sampleCount`.
    pub sample_count: u32,
}

const _: () = assert!(size_of::<CmdSetDefaultSampleCount>() == 0x4);

/// `PGCmdSetTileParameters`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetTileParameters {
    /// `tileWidth`.
    pub tile_width: u16,
    /// `tileHeight`.
    pub tile_height: u16,
}

const _: () = assert!(size_of::<CmdSetTileParameters>() == 0x4);

/// `PGCmdSetValue`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetValue {
    /// `value`.
    pub value: u64,
}

const _: () = assert!(size_of::<CmdSetValue>() == 0x8);

/// `PGCmdSetLineWidth`: 0x4 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub struct CmdSetLineWidth {
    /// `lineWidth`.
    pub line_width: f32,
}

const _: () = assert!(size_of::<CmdSetLineWidth>() == 0x4);

/// `PGCmdSetAlphaTestReferenceValue`: 0x4 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub struct CmdSetAlphaTestReferenceValue {
    /// `referenceValue`.
    pub reference_value: f32,
}

const _: () = assert!(size_of::<CmdSetAlphaTestReferenceValue>() == 0x4);

/// `PGCmdSetPointSize`: 0x4 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub struct CmdSetPointSize {
    /// `pointSize`.
    pub point_size: f32,
}

const _: () = assert!(size_of::<CmdSetPointSize>() == 0x4);

/// `PGCmdSetStoreAction`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetStoreAction {
    /// `storeAction`.
    pub store_action: u32,
    /// `colorAttachmentIndex`.
    pub color_attachment_index: u32,
}

const _: () = assert!(size_of::<CmdSetStoreAction>() == 0x8);

/// `PGCmdSetStoreActionOption`: 0xc bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdSetStoreActionOption {
    /// `storeActionOptions`.
    pub store_action_options: u64,
    /// `colorAttachmentIndex`.
    pub color_attachment_index: u32,
}

const _: () = assert!(size_of::<CmdSetStoreActionOption>() == 0xc);

/// `PGCmdSetDepthStencilState`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetDepthStencilState {
    /// `depthStencilStateReference`.
    pub depth_stencil_state_reference: u32,
}

const _: () = assert!(size_of::<CmdSetDepthStencilState>() == 0x4);

/// `PGCmdSetScissorRect`: 0x20 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetScissorRect {
    /// `x`.
    pub x: u64,
    /// `y`.
    pub y: u64,
    /// `width`.
    pub width: u64,
    /// `height`.
    pub height: u64,
}

const _: () = assert!(size_of::<CmdSetScissorRect>() == 0x20);

/// `PGCmdSetScissorRects`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetScissorRects {
    /// `count`.
    pub count: u64,
}

const _: () = assert!(size_of::<CmdSetScissorRects>() == 0x8);

/// `PGCmdSetScissorRectEntry`: 0x20 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetScissorRectEntry {
    /// `x`.
    pub x: u64,
    /// `y`.
    pub y: u64,
    /// `width`.
    pub width: u64,
    /// `height`.
    pub height: u64,
}

const _: () = assert!(size_of::<CmdSetScissorRectEntry>() == 0x20);

/// `PGCmdSetTesselationFactorBuffer`: 0x14 bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdSetTesselationFactorBuffer {
    /// `bufferReference`.
    pub buffer_reference: u32,
    /// `offset`.
    pub offset: u64,
    /// `instanceStride`.
    pub instance_stride: u64,
}

const _: () = assert!(size_of::<CmdSetTesselationFactorBuffer>() == 0x14);

/// `PGCmdSetViewport`: 0x30 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub struct CmdSetViewport {
    /// `originX`.
    pub origin_x: f64,
    /// `originY`.
    pub origin_y: f64,
    /// `width`.
    pub width: f64,
    /// `height`.
    pub height: f64,
    /// `znear`.
    pub znear: f64,
    /// `zfar`.
    pub zfar: f64,
}

const _: () = assert!(size_of::<CmdSetViewport>() == 0x30);

/// `PGCmdSetViewports`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetViewports {
    /// `count`.
    pub count: u32,
}

const _: () = assert!(size_of::<CmdSetViewports>() == 0x4);

/// `PGCmdSetViewportEntry`: 0x30 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub struct CmdSetViewportEntry {
    /// `originX`.
    pub origin_x: f64,
    /// `originY`.
    pub origin_y: f64,
    /// `width`.
    pub width: f64,
    /// `height`.
    pub height: f64,
    /// `znear`.
    pub znear: f64,
    /// `zfar`.
    pub zfar: f64,
}

const _: () = assert!(size_of::<CmdSetViewportEntry>() == 0x30);

/// `PGCmdRenderUseHeapsWithStages`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdRenderUseHeapsWithStages {
    /// `count`.
    pub count: u32,
    /// `stages`.
    pub stages: u16,
    /// Tail padding: the record's size counts it, so a record must carry it.
    pub unknown06: [u8; 2],
}

const _: () = assert!(size_of::<CmdRenderUseHeapsWithStages>() == 0x8);

/// `PGCmdUseResourcesWithStages`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdUseResourcesWithStages {
    /// `count`.
    pub count: u32,
    /// `usage`.
    pub usage: u16,
    /// `stages`.
    pub stages: u16,
}

const _: () = assert!(size_of::<CmdUseResourcesWithStages>() == 0x8);

/// `PGCmdSetClipPlane`: 0x14 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub struct CmdSetClipPlane {
    /// `p1`.
    pub p1: f32,
    /// `p2`.
    pub p2: f32,
    /// `p3`.
    pub p3: f32,
    /// `p4`.
    pub p4: f32,
    /// `index`.
    pub index: u32,
}

const _: () = assert!(size_of::<CmdSetClipPlane>() == 0x14);

/// `PGCmdSetSamplerState`: 0x14 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub struct CmdSetSamplerState {
    /// `samplerStateReference`.
    pub sampler_state_reference: u32,
    /// `lodMinClamp`.
    pub lod_min_clamp: f32,
    /// `lodMaxClamp`.
    pub lod_max_clamp: f32,
    /// `lodBias`.
    pub lod_bias: f32,
    /// `index`.
    pub index: u32,
}

const _: () = assert!(size_of::<CmdSetSamplerState>() == 0x14);

/// `PGCmdSetPrimitiveRestartEnabledIndex`: 0x8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetPrimitiveRestartEnabledIndex {
    /// `enabled`.
    pub enabled: u32,
    /// `index`.
    pub index: u32,
}

const _: () = assert!(size_of::<CmdSetPrimitiveRestartEnabledIndex>() == 0x8);

/// `PGCmdSetColorResolveTexture`: 0x10 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetColorResolveTexture {
    /// `textureReference`.
    pub texture_reference: u32,
    /// `slice`.
    pub slice: u16,
    /// `depthPlane`.
    pub depth_plane: u16,
    /// `level`.
    pub level: u16,
    /// `yInvert`.
    pub y_invert: u16,
    /// `colorAttachmentIndex`.
    pub color_attachment_index: u16,
    /// Tail padding: the record's size counts it, so a record must carry it.
    pub unknown0e: [u8; 2],
}

const _: () = assert!(size_of::<CmdSetColorResolveTexture>() == 0x10);

/// `PGCmdSetDepthStencilResolveTexture`: 0xc bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetDepthStencilResolveTexture {
    /// `textureReference`.
    pub texture_reference: u32,
    /// `slice`.
    pub slice: u16,
    /// `depthPlane`.
    pub depth_plane: u16,
    /// `level`.
    pub level: u16,
    /// `yInvert`.
    pub y_invert: u16,
}

const _: () = assert!(size_of::<CmdSetDepthStencilResolveTexture>() == 0xc);

/// `PGCmdSetVertexAmplificationCount`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetVertexAmplificationCount {
    /// `count`.
    pub count: u32,
}

const _: () = assert!(size_of::<CmdSetVertexAmplificationCount>() == 0x4);

/// `PGVertexAmplificationViewMapping`: 0x8 bytes.
///
/// Not PGCmd-prefixed: the mangling of its readOrThrow names it so.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct VertexAmplificationViewMapping {
    /// `viewportArrayIndexOffset`.
    pub viewport_array_index_offset: u32,
    /// `renderTargetArrayIndexOffset`.
    pub render_target_array_index_offset: u32,
}

const _: () = assert!(size_of::<VertexAmplificationViewMapping>() == 0x8);

/// `PGCmdDispatchThreadsPerTile`: 0x18 bytes.
///
/// The tile dispatches hand Metal the record's `threadsPerTile` whole, as
/// the MTLSize it is.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdDispatchThreadsPerTile {
    /// `threadsPerTile`.
    pub threads_per_tile: MtlSizeRecord,
}

const _: () = assert!(size_of::<CmdDispatchThreadsPerTile>() == 0x18);

/// `PGCmdRenderThreadgroupMemoryLength`: 0x14 bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdRenderThreadgroupMemoryLength {
    /// `length`.
    pub length: u64,
    /// `offset`.
    pub offset: u64,
    /// `index`.
    pub index: u32,
}

const _: () = assert!(size_of::<CmdRenderThreadgroupMemoryLength>() == 0x14);

/// `PGCmdDispatchThreadsPerTileInRegionWithIndex`: 0x4c bytes, packed.
///
/// The plain in-region decoder reads this record too, and ignores
/// renderTargetArrayIndex. The region is a separate origin and size: the
/// decoders copy each in 24 bytes (16 then 8) into an MTLRegion to pass it.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdDispatchThreadsPerTileInRegionWithIndex {
    /// `threadsPerTile`.
    pub threads_per_tile: MtlSizeRecord,
    /// `regionOrigin`.
    pub region_origin: MtlOriginRecord,
    /// `regionSize`.
    pub region_size: MtlSizeRecord,
    /// `renderTargetArrayIndex`.
    pub render_target_array_index: u32,
}

const _: () = assert!(size_of::<CmdDispatchThreadsPerTileInRegionWithIndex>() == 0x4c);

/// `PGCmdSetProgrammableSamplePositions`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetProgrammableSamplePositions {
    /// `count`.
    pub count: u32,
}

const _: () = assert!(size_of::<CmdSetProgrammableSamplePositions>() == 0x4);

/// `PGCmdCoordinate`: 0x8 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub struct CmdCoordinate {
    /// `x`.
    pub x: f32,
    /// `y`.
    pub y: f32,
}

const _: () = assert!(size_of::<CmdCoordinate>() == 0x8);

/// `PGCmdSetVariableRasterizationRateMap`: 0x4 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct CmdSetVariableRasterizationRateMap {
    /// `rasterizationRateMapReference`.
    pub rasterization_rate_map_reference: u32,
}

const _: () = assert!(size_of::<CmdSetVariableRasterizationRateMap>() == 0x4);

/// `PGCmdDescribeRenderPass.depthAttachment`: 0x28 bytes, packed. The ABI leaves the struct
/// anonymous; the type name is ours, after the field that holds it.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
pub struct DescribeRenderPassDepthAttachment {
    /// `textureReference`.
    pub texture_reference: u32,
    /// `resolveTextureReference`.
    pub resolve_texture_reference: u32,
    /// `level`.
    pub level: u16,
    /// `slice`.
    pub slice: u16,
    /// `depthPlane`.
    pub depth_plane: u16,
    /// `resolveLevel`.
    pub resolve_level: u16,
    /// `resolveSlice`.
    pub resolve_slice: u16,
    /// `resolveDepthPlane`.
    pub resolve_depth_plane: u16,
    /// `loadAction`.
    pub load_action: u16,
    /// `storeAction`.
    pub store_action: u16,
    /// `storeActionOptions`.
    pub store_action_options: u16,
    /// `field1a` (not read).
    pub field_1a: u16,
    /// `clearDepth`.
    pub clear_depth: f64,
    /// `depthResolveFilter`.
    pub depth_resolve_filter: u16,
    /// `field26` (not read).
    pub field_26: u16,
}

const _: () = assert!(size_of::<DescribeRenderPassDepthAttachment>() == 0x28);

/// `PGCmdDescribeRenderPass.stencilAttachment`: 0x24 bytes, packed. The ABI leaves the struct
/// anonymous; the type name is ours, after the field that holds it.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct DescribeRenderPassStencilAttachment {
    /// `textureReference`.
    pub texture_reference: u32,
    /// `resolveTextureReference`.
    pub resolve_texture_reference: u32,
    /// `level`.
    pub level: u16,
    /// `slice`.
    pub slice: u16,
    /// `depthPlane`.
    pub depth_plane: u16,
    /// `resolveLevel`.
    pub resolve_level: u16,
    /// `resolveSlice`.
    pub resolve_slice: u16,
    /// `resolveDepthPlane`.
    pub resolve_depth_plane: u16,
    /// `loadAction`.
    pub load_action: u16,
    /// `storeAction`.
    pub store_action: u16,
    /// `storeActionOptions`.
    pub store_action_options: u16,
    /// `field1a` (not read).
    pub field_1a: u16,
    /// `clearStencil`.
    pub clear_stencil: u32,
    /// `stencilResolveFilter`.
    pub stencil_resolve_filter: u16,
    /// `field22` (not read).
    pub field_22: u16,
}

const _: () = assert!(size_of::<DescribeRenderPassStencilAttachment>() == 0x24);

/// `PGCmdDescribeRenderPass.colorAttachments[i]`: 0x3c bytes, packed. The ABI leaves the struct
/// anonymous; the type name is ours, after the field that holds it.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
pub struct DescribeRenderPassColorAttachment {
    /// `textureReference`.
    pub texture_reference: u32,
    /// `resolveTextureReference`.
    pub resolve_texture_reference: u32,
    /// `level`.
    pub level: u16,
    /// `slice`.
    pub slice: u16,
    /// `depthPlane`.
    pub depth_plane: u16,
    /// `resolveLevel`.
    pub resolve_level: u16,
    /// `resolveSlice`.
    pub resolve_slice: u16,
    /// `resolveDepthPlane`.
    pub resolve_depth_plane: u16,
    /// `loadAction`.
    pub load_action: u16,
    /// `storeAction`.
    pub store_action: u16,
    /// `storeActionOptions`.
    pub store_action_options: u16,
    /// `field1a` (not read).
    pub field_1a: u16,
    /// `clearColorRed`.
    pub clear_color_red: f64,
    /// `clearColorGreen`.
    pub clear_color_green: f64,
    /// `clearColorBlue`.
    pub clear_color_blue: f64,
    /// `clearColorAlpha`.
    pub clear_color_alpha: f64,
}

const _: () = assert!(size_of::<DescribeRenderPassColorAttachment>() == 0x3c);

/// `PGCmdDescribeRenderPass`: 0x248 bytes, packed.
///
/// A render pass, as -decodeRenderDescribeRenderPassWithCursor:descriptor:
/// applies it: the depth attachment, the stencil attachment, eight color
/// attachments, then the visibility result buffer and the render target
/// extent.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
pub struct CmdDescribeRenderPass {
    /// `depthAttachment`.
    pub depth_attachment: DescribeRenderPassDepthAttachment,
    /// `stencilAttachment`.
    pub stencil_attachment: DescribeRenderPassStencilAttachment,
    /// `colorAttachments[8]`.
    pub color_attachments: [DescribeRenderPassColorAttachment; 8],
    /// `visibilityResultBufferReference`.
    pub visibility_result_buffer_reference: u32,
    /// `renderTargetArrayLength`.
    pub render_target_array_length: u64,
    /// `renderTargetWidth`.
    pub render_target_width: u64,
    /// `renderTargetHeight`.
    pub render_target_height: u64,
}

const _: () = assert!(size_of::<CmdDescribeRenderPass>() == 0x248);

/// `PGCmdGetTileDimensions`: 0xc bytes, packed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct CmdGetTileDimensions {
    /// `replyBufferReference`.
    pub reply_buffer_reference: u32,
    /// `replyBufferOffset`.
    pub reply_buffer_offset: u64,
}

const _: () = assert!(size_of::<CmdGetTileDimensions>() == 0xc);

/// `PGReplyTileDimensions`: 0x4 bytes.
///
/// What -decodeGetTileDimensionsWithCursor: writes back: the encoder's tile
/// width and height, each stored as 16 bits (strh).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct ReplyTileDimensions {
    /// `tileWidth`.
    pub tile_width: u16,
    /// `tileHeight`.
    pub tile_height: u16,
}

const _: () = assert!(size_of::<ReplyTileDimensions>() == 0x4);

// ---------------------------------------------------------------------------------------------
// The rate-map reset's records (the serialized rate map, after its operation header).

/// The rate-map reset's descriptor record: 0x10 bytes, read after its operation header.
///
/// `layer_count` is the ABI's name; the others are ours, each named by the descriptor property
/// the reset sets from it.
#[derive(Debug, Clone, Copy, PartialEq, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub struct OperationRasterizationRateMapDescriptor {
    /// The screen width (name is ours: `setScreenSize:`'s width).
    pub screen_width: u16,
    /// The screen height (name is ours: `setScreenSize:`'s height).
    pub screen_height: u16,
    /// The number of layer records that follow.
    pub layer_count: u32,
    /// The mutability (name is ours: `setMutability:`; 1 is mutable).
    pub mutability: u32,
    /// The minimum factor (name is ours: `setMinFactor:`).
    pub min_factor: f32,
}

const _: () = assert!(size_of::<OperationRasterizationRateMapDescriptor>() == 0x10);

/// One layer's sample counts in the rate-map reset. Names are ours: the layer descriptor's
/// `initWithSampleCount:` width and height.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct OperationRasterizationRateMapLayer {
    /// The horizontal sample count.
    pub horizontal_sample_count: u16,
    /// The vertical sample count.
    pub vertical_sample_count: u16,
}

const _: () = assert!(size_of::<OperationRasterizationRateMapLayer>() == 4);

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    /// Asserts a record's size, its alignment when the ABI lays it out naturally (`Some`), and each
    /// field's offset, against the table clang printed for the guest ABI's records.
    macro_rules! check {
        ($t:ty, $size:expr, $align:expr; $($f:ident: $off:expr),* $(,)?) => {{
            assert_eq!(size_of::<$t>(), $size, "sizeof {}", stringify!($t));
            let align: Option<usize> = $align;
            if let Some(a) = align {
                assert_eq!(align_of::<$t>(), a, "alignof {}", stringify!($t));
            }
            $(
                assert_eq!(
                    offset_of!($t, $f),
                    $off,
                    "offsetof {}.{}",
                    stringify!($t),
                    stringify!($f)
                );
            )*
        }};
    }

    /// Every record's size (tail padding included), alignment and field offsets are the ABI's (clang's
    /// `sizeof`/`alignof`/`offsetof` over the guest ABI's records).
    #[test]
    fn every_record_matches_its_layout() {
        check!(SegmentHeader, 0x8, Some(4); length: 0x0, encoder_type: 0x4,
            continues_previous: 0x5, continued_by_next: 0x6);
        check!(CmdFenceOp, 0x4, Some(4); fence_reference: 0x0);
        check!(CmdHeader, 0x8, Some(4); command_id: 0x0, length: 0x4);
        check!(CmdGenerateMipmaps, 0x4, Some(4); texture_reference: 0x0);
        check!(CmdOptimize, 0x4, Some(4); texture_reference: 0x0);
        check!(CmdOptimizeImage, 0x8, Some(4); texture_reference: 0x0, slice: 0x4, level: 0x6);
        check!(CmdSynchronize, 0x4, Some(4); resource_reference: 0x0);
        check!(CmdSynchronizeImage, 0x8, Some(4); texture_reference: 0x0, slice: 0x4, level: 0x6);
        check!(CmdCopyFromBufferToTexture, 0x58, Some(8); source_buffer_reference: 0x0, destination_texture_reference: 0x4, source_offset: 0x8, source_bytes_per_row: 0x10, source_bytes_per_image: 0x18, source_size: 0x20, destination_origin: 0x38, destination_slice: 0x50, destination_level: 0x52, options: 0x54);
        check!(CmdCopyFromBufferToBuffer, 0x20, Some(8); source_buffer_reference: 0x0, destination_buffer_reference: 0x4, source_offset: 0x8, destination_offset: 0x10, size: 0x18);
        check!(CmdCopyFromTextureToBuffer, 0x58, Some(8); source_texture_reference: 0x0, destination_buffer_reference: 0x4, source_origin: 0x8, source_size: 0x20, destination_offset: 0x38, destination_bytes_per_row: 0x40, destination_bytes_per_image: 0x48, source_slice: 0x50, source_level: 0x52, options: 0x54);
        check!(CmdCopyFromTextureToTexture, 0x58, Some(8); source_texture_reference: 0x0, destination_texture_reference: 0x4, source_origin: 0x8, source_size: 0x20, destination_origin: 0x38, source_slice: 0x50, source_level: 0x52, destination_slice: 0x54, destination_level: 0x56);
        check!(CmdCopyFromTextureToTextureWithNumSliceLevel, 0x14, Some(4); source_texture_reference: 0x0, destination_texture_reference: 0x4, source_slice: 0x8, source_level: 0xa, destination_slice: 0xc, destination_level: 0xe, slice_count: 0x10, level_count: 0x12);
        check!(CmdInvalidateCompressedTexture, 0x4, Some(4); texture_reference: 0x0);
        check!(CmdInvalidateCompressedTextureImage, 0x8, Some(4); texture_reference: 0x0, slice: 0x4, level: 0x6);
        check!(CmdCopyIndirectCommandBuffer, 0x20, Some(8); source_indirect_command_buffer_reference: 0x0, destination_indirect_command_buffer_reference: 0x4, source_range_location: 0x8, source_range_length: 0x10, destination_index: 0x18);
        check!(CmdCopyFromTextureToTextureWithOptions, 0x5c, None; source_texture_reference: 0x0, destination_texture_reference: 0x4, source_origin: 0x8, source_size: 0x20, destination_origin: 0x38, source_slice: 0x50, source_level: 0x52, destination_slice: 0x54, destination_level: 0x56, options: 0x58);
        check!(CmdFillBuffer, 0x18, None; buffer_reference: 0x0, range_location: 0x4, range_length: 0xc, value: 0x14);
        check!(CmdFillBufferWithPattern, 0x18, None; buffer_reference: 0x0, range_location: 0x4, range_length: 0xc, pattern: 0x14);
        check!(CmdFillTextureWithBytes, 0x4c, None; texture_reference: 0x0, level: 0x4, slice: 0x6, region_size: 0x8, region_origin: 0x20, buffer_reference: 0x38, buffer_offset: 0x3c, length: 0x44);
        check!(CmdFillTextureWithColor, 0x5c, None; texture_reference: 0x0, level: 0x4, slice: 0x6, region_size: 0x8, region_origin: 0x20, color: 0x38, pixel_format: 0x58);
        check!(CmdOptimizeResetIndirectBuffer, 0x14, None; indirect_command_buffer_reference: 0x0, range_location: 0x4, range_length: 0xc);
        check!(CmdComputePipelineStateInfo, 0x10, Some(8); pipeline_state_reference: 0x0, reply_buffer_reference: 0x4, reply_buffer_offset: 0x8);
        check!(CmdHeapTextureSizeAndAlignWithDescriptor, 0x2c, None; descriptor: 0x0, reply_buffer_reference: 0x20, reply_buffer_offset: 0x24);
        check!(CmdHeapTextureSizeAndAlignWithDescriptor2, 0x34, None; descriptor: 0x0, reply_buffer_reference: 0x28, reply_buffer_offset: 0x2c);
        check!(CmdGetRasterizationRateMapInfo, 0x18, Some(8); rasterization_rate_map_reference: 0x0, reply_buffer_reference: 0x4, reply_buffer_offset: 0x8, field_10: 0x10);
        check!(CmdCopyRasterizationRateMapParameterBuffer, 0x10, Some(8); rasterization_rate_map_reference: 0x0, buffer_reference: 0x4, buffer_offset: 0x8);
        check!(CmdMapCoordinates, 0x1c, None; rasterization_rate_map_reference: 0x0, reply_buffer_reference: 0x4, reply_buffer_offset: 0x8, layer_index: 0x10, coordinates: 0x14);
        check!(CmdMapCoordinatesMultiple, 0x20, Some(8); rasterization_rate_map_reference: 0x0, reply_buffer_reference: 0x4, reply_buffer_offset: 0x8, field_10: 0x10, layer_index: 0x18, coordinate_count: 0x1c);
        check!(CmdRenderPipelineStateInfo, 0x10, Some(8); pipeline_state_reference: 0x0, reply_buffer_reference: 0x4, reply_buffer_offset: 0x8);
        check!(CmdPipelineStateImageBlockMemoryLength, 0x28, None; pipeline_state_reference: 0x0, imageblock_dimensions: 0x4, reply_buffer_reference: 0x1c, reply_buffer_offset: 0x20);
        check!(CmdObjectUniqueIdentifier, 0x14, None; object_reference: 0x0, object_type: 0x4, reply_buffer_reference: 0x8, reply_buffer_offset: 0xc);
        check!(CmdBufferHostResourceInfo, 0x10, Some(8); buffer_reference: 0x0, reply_buffer_reference: 0x4, reply_buffer_offset: 0x8);
        check!(CmdTextureHostResourceInfo, 0x10, Some(8); texture_reference: 0x0, reply_buffer_reference: 0x4, reply_buffer_offset: 0x8);
        check!(CmdHeapHostResourceInfo, 0x10, Some(8); heap_reference: 0x0, reply_buffer_reference: 0x4, reply_buffer_offset: 0x8);
        check!(CmdSamplerStateHostResourceInfo, 0x10, Some(8); sampler_state_reference: 0x0, reply_buffer_reference: 0x4, reply_buffer_offset: 0x8);
        check!(CmdICBHostResourceInfo, 0x10, Some(8); indirect_command_buffer_reference: 0x0, reply_buffer_reference: 0x4, reply_buffer_offset: 0x8);
        check!(CmdObjectHostInfo, 0x10, Some(8); object_reference: 0x0, reply_buffer_reference: 0x4, reply_buffer_offset: 0x8);
        check!(ReplyComputePipelineStateInfo, 0x1c, None; max_total_threads_per_threadgroup: 0x0, thread_execution_width: 0x8, static_threadgroup_memory_length: 0x10, support_indirect_command_buffers: 0x18);
        check!(ReplyHeapTextureSizeAndAlign, 0x10, Some(8); size: 0x0, align: 0x8);
        check!(ReplyRasterizationRateMapInfo, 0x14, None; parameter_buffer_size: 0x0, parameter_buffer_align: 0x8, physical_granularity_width: 0x10, physical_granularity_height: 0x12);
        check!(Cmd2DSize16, 0x4, Some(2); width: 0x0, height: 0x2);
        check!(ReplyRasterizationRateMapCoordinate, 0x8, Some(4); x: 0x0, y: 0x4);
        check!(ReplyRenderPipelineStateInfo, 0xc, Some(4); max_total_threads_per_threadgroup: 0x0, imageblock_sample_length: 0x4, threadgroup_size_matches_tile_size: 0x8, support_indirect_command_buffers: 0x9);
        check!(ReplyPipelineStateImageBlockMemoryLength, 0x4, Some(4); imageblock_memory_length: 0x0);
        check!(ReplyUniqueIdentifierDeprecated, 0x8, Some(8); unique_identifier: 0x0);
        check!(ReplyBufferHostResourceInfo, 0x8, Some(8); gpu_address: 0x0);
        check!(ReplyHostObjectInfo, 0x10, Some(8); gpu_resource_id: 0x0, resource_index: 0x8);
        check!(CmdDispatchThreadgroups, 0x30, Some(8); threadgroups_per_grid: 0x0, threads_per_threadgroup: 0x18);
        check!(CmdDispatchThreads, 0x30, Some(8); threads_per_grid: 0x0, threads_per_threadgroup: 0x18);
        check!(CmdSetComputeStageInRegion, 0x30, Some(8); region_size: 0x0, region_origin: 0x18);
        check!(CmdSetImageblockWidth, 0x8, Some(4); width: 0x0, height: 0x4);
        check!(CmdDispatchThreadgroupsIndirect, 0x24, None; threads_per_threadgroup: 0x0, indirect_buffer_offset: 0x18, indirect_buffer_reference: 0x20);
        check!(CmdDispatchThreadsIndirect, 0xc, None; indirect_buffer_offset: 0x0, indirect_buffer_reference: 0x8);
        check!(CmdSetComputeStageInRegionIndirect, 0xc, None; indirect_buffer_reference: 0x0, indirect_buffer_offset: 0x4);
        check!(CmdSetThreadgroupMemoryLength, 0xc, None; length: 0x0, index: 0x8);
        check!(CmdMemoryBarrierResources, 0x4, Some(4); count: 0x0);
        check!(CmdEncodeEndDoWhile, 0x14, None; buffer_reference: 0x0, offset: 0x4, comparison: 0xc, reference_value: 0x10);
        check!(CmdEncodeStartWhile, 0x14, None; buffer_reference: 0x0, offset: 0x4, comparison: 0xc, reference_value: 0x10);
        check!(CmdEncodeStartIf, 0x14, None; buffer_reference: 0x0, offset: 0x4, comparison: 0xc, reference_value: 0x10);
        check!(CmdMemoryBarrierScope, 0x4, Some(4); );
        check!(CmdSetComputePassDescriptorDispatchType, 0x4, Some(4); dispatch_type: 0x0);
        check!(CmdSetBuffers, 0x8, Some(4); range_location: 0x0, range_length: 0x4);
        check!(CmdSetBufferEntry, 0xc, None; buffer_reference: 0x0, offset: 0x4);
        check!(CmdSetBufferOffset, 0xc, None; index: 0x0, offset: 0x4);
        check!(CmdSetBuffersWithStride, 0x8, Some(4); range_location: 0x0, range_length: 0x4);
        check!(CmdSetBufferEntryWithStride, 0x14, None; buffer_reference: 0x0, offset: 0x4, attribute_stride: 0xc);
        check!(CmdSetBufferOffsetWithStride, 0x14, None; index: 0x0, offset: 0x4, attribute_stride: 0xc);
        check!(CmdSetSamplerStates, 0x8, Some(4); range_location: 0x0, range_length: 0x4);
        check!(CmdSetSamplerStatesLODClamp, 0x8, Some(4); range_location: 0x0, range_length: 0x4);
        check!(CmdSetSamplerStatesLODClampEntry, 0xc, Some(4); sampler_state_reference: 0x0, lod_min_clamp: 0x4, lod_max_clamp: 0x8);
        check!(CmdSetTextures, 0x8, Some(4); range_location: 0x0, range_length: 0x4);
        check!(CmdSetRenderPipelineState, 0x4, Some(4); pipeline_state_reference: 0x0);
        check!(CmdUseHeaps, 0x4, Some(4); count: 0x0);
        check!(CmdUseResources, 0x8, Some(4); count: 0x0, usage: 0x4);
        check!(CmdExecuteCommandsInBufferRanged, 0x14, None; indirect_command_buffer_reference: 0x0, range_location: 0x4, range_length: 0xc);
        check!(CmdExecuteCommandsInBuffer, 0x10, Some(8); indirect_command_buffer_reference: 0x0, indirect_range_buffer_reference: 0x4, indirect_buffer_offset: 0x8);
        check!(CmdDrawPrimitives64, 0x14, None; primitive_type: 0x0, vertex_start: 0x4, vertex_count: 0xc);
        check!(CmdDrawPrimitives16, 0x8, Some(4); primitive_type: 0x0, vertex_start: 0x4, vertex_count: 0x6);
        check!(CmdDrawInstancedPrimitives64, 0x1c, None; vertex_start: 0x0, vertex_count: 0x8, instance_count: 0x10, primitive_type: 0x18);
        check!(CmdDrawInstancedPrimitives16, 0x8, Some(2); vertex_start: 0x0, vertex_count: 0x2, instance_count: 0x4, primitive_type: 0x6);
        check!(CmdDrawInstancedBasePrimitives64, 0x24, None; vertex_start: 0x0, vertex_count: 0x8, instance_count: 0x10, base_instance: 0x18, primitive_type: 0x20);
        check!(CmdDrawInstancedBasePrimitives16, 0xc, None; vertex_start: 0x0, vertex_count: 0x2, instance_count: 0x4, base_instance: 0x6, primitive_type: 0x8);
        check!(CmdDrawIndexedPrimitives64, 0x18, Some(8); primitive_type: 0x0, index_type: 0x2, index_buffer_reference: 0x4, index_count: 0x8, index_buffer_offset: 0x10);
        check!(CmdDrawIndexedPrimitives16, 0xc, Some(4); primitive_type: 0x0, index_type: 0x2, index_buffer_reference: 0x4, index_count: 0x8, index_buffer_offset: 0xa);
        check!(CmdDrawIndexedInstancedPrimitives64, 0x20, Some(8); primitive_type: 0x0, index_type: 0x2, index_buffer_reference: 0x4, index_count: 0x8, index_buffer_offset: 0x10, instance_count: 0x18);
        check!(CmdDrawIndexedInstancedPrimitives16, 0x10, Some(4); primitive_type: 0x0, index_type: 0x2, index_buffer_reference: 0x4, index_count: 0x8, index_buffer_offset: 0xa, instance_count: 0xc);
        check!(CmdDrawIndexedInstancedBasePrimitives64, 0x30, Some(8); primitive_type: 0x0, index_type: 0x2, index_buffer_reference: 0x4, index_buffer_offset: 0x8, index_count: 0x10, instance_count: 0x18, base_vertex: 0x20, base_instance: 0x28);
        check!(CmdDrawIndexedInstancedBasePrimitives16, 0x14, Some(4); primitive_type: 0x0, index_type: 0x2, index_buffer_reference: 0x4, index_buffer_offset: 0x8, index_count: 0xa, instance_count: 0xc, base_vertex: 0xe, base_instance: 0x10);
        check!(CmdDrawIndexedInstancedBasePrimitives64_2, 0x30, Some(8); primitive_type: 0x0, index_type: 0x2, index_buffer_reference: 0x4, index_buffer_offset: 0x8, index_count: 0x10, instance_count: 0x18, base_vertex: 0x20, base_instance: 0x28);
        check!(CmdDrawIndexedInstancedBasePrimitives16_2, 0x14, Some(4); primitive_type: 0x0, index_type: 0x2, index_buffer_reference: 0x4, index_buffer_offset: 0x8, index_count: 0xa, instance_count: 0xc, base_vertex: 0xe, base_instance: 0x10);
        check!(CmdDrawPatches64, 0x30, None; patch_start: 0x0, patch_count: 0x8, patch_index_buffer_reference: 0x10, patch_index_buffer_offset: 0x14, instance_count: 0x1c, base_instance: 0x24, number_of_patch_control_points: 0x2c);
        check!(CmdDrawPatches16, 0x10, Some(4); patch_start: 0x0, patch_count: 0x2, patch_index_buffer_reference: 0x4, patch_index_buffer_offset: 0x8, instance_count: 0xa, base_instance: 0xc, number_of_patch_control_points: 0xe);
        check!(CmdDrawIndexedPatches64, 0x3c, None; patch_start: 0x0, patch_count: 0x8, patch_index_buffer_reference: 0x10, patch_index_buffer_offset: 0x14, control_point_index_buffer_reference: 0x1c, control_point_index_buffer_offset: 0x20, instance_count: 0x28, base_instance: 0x30, number_of_patch_control_points: 0x38);
        check!(CmdDrawIndexedPatches16, 0x18, None; patch_start: 0x0, patch_count: 0x2, patch_index_buffer_reference: 0x4, patch_index_buffer_offset: 0x8, control_point_index_buffer_reference: 0xa, control_point_index_buffer_offset: 0xe, instance_count: 0x10, base_instance: 0x12, number_of_patch_control_points: 0x14);
        check!(CmdDrawPrimitivesIndirect, 0x10, Some(8); indirect_buffer_offset: 0x0, indirect_buffer_reference: 0x8, primitive_type: 0xc);
        check!(CmdDrawIndexedPrimitivesIndirect, 0x1c, None; primitive_type: 0x0, index_type: 0x2, index_buffer_reference: 0x4, indirect_buffer_reference: 0x8, index_buffer_offset: 0xc, indirect_buffer_offset: 0x14);
        check!(CmdDrawPatchesIndirect, 0x1c, None; patch_index_buffer_reference: 0x0, indirect_buffer_reference: 0x4, patch_index_buffer_offset: 0x8, indirect_buffer_offset: 0x10, number_of_patch_control_points: 0x18);
        check!(CmdDrawIndexedPatchesIndirect, 0x28, None; patch_index_buffer_reference: 0x0, control_point_index_buffer_reference: 0x4, indirect_buffer_reference: 0x8, patch_index_buffer_offset: 0xc, control_point_index_buffer_offset: 0x14, indirect_buffer_offset: 0x1c, number_of_patch_control_points: 0x24);
        check!(CmdRenderUpdateFence, 0x8, Some(4); fence_reference: 0x0, after_stages: 0x4);
        check!(CmdRenderWaitForFence, 0x8, Some(4); fence_reference: 0x0, before_stages: 0x4);
        check!(CmdSetImageBlockSampleLength, 0x4, Some(4); imageblock_sample_length: 0x0);
        check!(CmdSetThreadGroupMemoryLength, 0x4, Some(4); threadgroup_memory_length: 0x0);
        check!(CmdRenderMemoryBarrierScope, 0x4, Some(2); scope: 0x0, after_stages: 0x2, before_stages: 0x3);
        check!(CmdSetBlendColor, 0x10, Some(4); red: 0x0, green: 0x4, blue: 0x8, alpha: 0xc);
        check!(CmdSetDepthBias, 0xc, Some(4); depth_bias: 0x0, slope_scale: 0x4, clamp: 0x8);
        check!(CmdSetFloatValue, 0x4, Some(4); value: 0x0);
        check!(CmdSetStencilRef, 0x8, Some(4); front_reference_value: 0x0, back_reference_value: 0x4);
        check!(CmdSetVisibiliyResultMode, 0x10, Some(8); offset: 0x0, mode: 0x8);
        check!(CmdSetProvokingVertexMode, 0x4, Some(4); mode: 0x0);
        check!(CmdSetTransformFeedbackState, 0x4, Some(4); state: 0x0);
        check!(CmdSetViewportTransformEnabled, 0x4, Some(4); enabled: 0x0);
        check!(CmdSetTriangleFillModeFrontBack, 0x4, Some(2); front_fill_mode: 0x0, back_fill_mode: 0x2);
        check!(CmdSetVertexAmplificationMode, 0x8, Some(4); mode: 0x0, value: 0x4);
        check!(CmdSetDefaultSampleCount, 0x4, Some(4); sample_count: 0x0);
        check!(CmdSetTileParameters, 0x4, Some(2); tile_width: 0x0, tile_height: 0x2);
        check!(CmdSetValue, 0x8, Some(8); value: 0x0);
        check!(CmdSetLineWidth, 0x4, Some(4); line_width: 0x0);
        check!(CmdSetAlphaTestReferenceValue, 0x4, Some(4); reference_value: 0x0);
        check!(CmdSetPointSize, 0x4, Some(4); point_size: 0x0);
        check!(CmdSetStoreAction, 0x8, Some(4); store_action: 0x0, color_attachment_index: 0x4);
        check!(CmdSetStoreActionOption, 0xc, None; store_action_options: 0x0, color_attachment_index: 0x8);
        check!(CmdSetDepthStencilState, 0x4, Some(4); depth_stencil_state_reference: 0x0);
        check!(CmdSetScissorRect, 0x20, Some(8); x: 0x0, y: 0x8, width: 0x10, height: 0x18);
        check!(CmdSetScissorRects, 0x8, Some(8); count: 0x0);
        check!(CmdSetScissorRectEntry, 0x20, Some(8); x: 0x0, y: 0x8, width: 0x10, height: 0x18);
        check!(CmdSetTesselationFactorBuffer, 0x14, None; buffer_reference: 0x0, offset: 0x4, instance_stride: 0xc);
        check!(CmdSetViewport, 0x30, Some(8); origin_x: 0x0, origin_y: 0x8, width: 0x10, height: 0x18, znear: 0x20, zfar: 0x28);
        check!(CmdSetViewports, 0x4, Some(4); count: 0x0);
        check!(CmdSetViewportEntry, 0x30, Some(8); origin_x: 0x0, origin_y: 0x8, width: 0x10, height: 0x18, znear: 0x20, zfar: 0x28);
        check!(CmdRenderUseHeapsWithStages, 0x8, Some(4); count: 0x0, stages: 0x4);
        check!(CmdUseResourcesWithStages, 0x8, Some(4); count: 0x0, usage: 0x4, stages: 0x6);
        check!(CmdSetClipPlane, 0x14, Some(4); p1: 0x0, p2: 0x4, p3: 0x8, p4: 0xc, index: 0x10);
        check!(CmdSetSamplerState, 0x14, Some(4); sampler_state_reference: 0x0, lod_min_clamp: 0x4, lod_max_clamp: 0x8, lod_bias: 0xc, index: 0x10);
        check!(CmdSetPrimitiveRestartEnabledIndex, 0x8, Some(4); enabled: 0x0, index: 0x4);
        check!(CmdSetColorResolveTexture, 0x10, Some(4); texture_reference: 0x0, slice: 0x4, depth_plane: 0x6, level: 0x8, y_invert: 0xa, color_attachment_index: 0xc);
        check!(CmdSetDepthStencilResolveTexture, 0xc, Some(4); texture_reference: 0x0, slice: 0x4, depth_plane: 0x6, level: 0x8, y_invert: 0xa);
        check!(CmdSetVertexAmplificationCount, 0x4, Some(4); count: 0x0);
        check!(VertexAmplificationViewMapping, 0x8, Some(4); viewport_array_index_offset: 0x0, render_target_array_index_offset: 0x4);
        check!(CmdDispatchThreadsPerTile, 0x18, Some(8); threads_per_tile: 0x0);
        check!(CmdRenderThreadgroupMemoryLength, 0x14, None; length: 0x0, offset: 0x8, index: 0x10);
        check!(CmdDispatchThreadsPerTileInRegionWithIndex, 0x4c, None; threads_per_tile: 0x0, region_origin: 0x18, region_size: 0x30, render_target_array_index: 0x48);
        check!(CmdSetProgrammableSamplePositions, 0x4, Some(4); count: 0x0);
        check!(CmdCoordinate, 0x8, Some(4); x: 0x0, y: 0x4);
        check!(CmdSetVariableRasterizationRateMap, 0x4, Some(4); rasterization_rate_map_reference: 0x0);
        check!(CmdGetTileDimensions, 0xc, None; reply_buffer_reference: 0x0, reply_buffer_offset: 0x4);
        check!(ReplyTileDimensions, 0x4, Some(2); tile_width: 0x0, tile_height: 0x2);
    }

    /// `PGCmdDescribeRenderPass`'s nested attachments sit where the render decoder loads them,
    /// and its tail fields at 0x22c-0x240.
    #[test]
    fn describe_render_pass_layout_matches_the_abi() {
        check!(DescribeRenderPassDepthAttachment, 0x28, None; texture_reference: 0x0,
            resolve_texture_reference: 0x4, level: 0x8, slice: 0xa, depth_plane: 0xc,
            resolve_level: 0xe, resolve_slice: 0x10, resolve_depth_plane: 0x12,
            load_action: 0x14, store_action: 0x16, store_action_options: 0x18, field_1a: 0x1a,
            clear_depth: 0x1c, depth_resolve_filter: 0x24, field_26: 0x26);
        check!(DescribeRenderPassStencilAttachment, 0x24, None; texture_reference: 0x0,
            store_action_options: 0x18, field_1a: 0x1a, clear_stencil: 0x1c,
            stencil_resolve_filter: 0x20, field_22: 0x22);
        check!(DescribeRenderPassColorAttachment, 0x3c, None; texture_reference: 0x0,
            field_1a: 0x1a, clear_color_red: 0x1c, clear_color_green: 0x24,
            clear_color_blue: 0x2c, clear_color_alpha: 0x34);
        check!(CmdDescribeRenderPass, 0x248, None; depth_attachment: 0x0,
            stencil_attachment: 0x28, color_attachments: 0x4c,
            visibility_result_buffer_reference: 0x22c, render_target_array_length: 0x230,
            render_target_width: 0x238, render_target_height: 0x240);
        // Color attachment i starts at 0x4c + i * 0x3c; the last one ends at 0x22c.
        let mut b = vec![0u8; 0x248];
        b[0x4c + 7 * 0x3c..0x4c + 7 * 0x3c + 4].copy_from_slice(&0xabcdu32.to_le_bytes());
        b[0x22c..0x230].copy_from_slice(&0x55u32.to_le_bytes());
        b[0x240..0x248].copy_from_slice(&1080u64.to_le_bytes());
        let pass = CmdDescribeRenderPass::read_from_bytes(&b).unwrap();
        let colors = pass.color_attachments;
        assert_eq!({ colors[7].texture_reference }, 0xabcd);
        assert_eq!({ colors[6].texture_reference }, 0);
        assert_eq!({ pass.visibility_result_buffer_reference }, 0x55);
        assert_eq!({ pass.render_target_height }, 1080);
    }

    /// A record whose `sizeof` counts tail padding needs every padding byte: one byte short fails
    /// to read, the padding is carried but not interpreted.
    #[test]
    fn short_by_tail_padding_fails_to_read() {
        let mut b = [0u8; 0x18];
        b[0..4].copy_from_slice(&3u32.to_le_bytes());
        b[4..0xc].copy_from_slice(&0x100u64.to_le_bytes());
        b[0xc..0x14].copy_from_slice(&0x40u64.to_le_bytes());
        b[0x14] = 0x7f;
        b[0x15..0x18].copy_from_slice(&[0xee; 3]);
        assert!(CmdFillBuffer::read_from_bytes(&b[..0x15]).is_err());
        assert!(CmdFillBuffer::read_from_bytes(&b[..0x17]).is_err());
        let r = CmdFillBuffer::read_from_bytes(&b).unwrap();
        assert_eq!(
            (
                { r.buffer_reference },
                { r.range_location },
                { r.range_length },
                r.value
            ),
            (3, 0x100, 0x40, 0x7f)
        );
        assert_eq!(r.unknown15, [0xee; 3]);
        // A naturally laid out record with tail padding behaves the same.
        assert!(CmdCopyFromTextureToBuffer::read_from_bytes(&[0u8; 0x56]).is_err());
        assert!(CmdCopyFromTextureToBuffer::read_from_bytes(&[0u8; 0x58]).is_ok());
        assert!(CmdDrawPrimitivesIndirect::read_from_bytes(&[0u8; 0xe]).is_err());
        assert!(CmdDrawPrimitivesIndirect::read_from_bytes(&[0u8; 0x10]).is_ok());
    }

    /// 0x0b's baseVertex is unsigned and 0x1d's signed: the same bytes widen differently.
    #[test]
    fn base_vertex_signedness_differs_between_0x0b_and_0x1d() {
        let mut b = [0u8; 0x14];
        b[0xe..0x10].copy_from_slice(&0xfffeu16.to_le_bytes());
        let unsigned = CmdDrawIndexedInstancedBasePrimitives16::read_from_bytes(&b).unwrap();
        let signed = CmdDrawIndexedInstancedBasePrimitives16_2::read_from_bytes(&b).unwrap();
        assert_eq!(u64::from(unsigned.base_vertex), 0xfffe);
        assert_eq!(i64::from(signed.base_vertex), -2);
        let mut b = [0u8; 0x30];
        b[0x20..0x28].copy_from_slice(&u64::MAX.to_le_bytes());
        let unsigned = CmdDrawIndexedInstancedBasePrimitives64::read_from_bytes(&b).unwrap();
        let signed = CmdDrawIndexedInstancedBasePrimitives64_2::read_from_bytes(&b).unwrap();
        assert_eq!(unsigned.base_vertex, u64::MAX);
        assert_eq!(signed.base_vertex, -1);
    }

    /// The records' odd offsets decode from guest bytes: 0x0f's 32-bit control-point reference at
    /// 0xa, 0x0a's index-buffer offset before its count, 0x84's offset before its mode.
    #[test]
    fn odd_offsets_decode() {
        let mut b = [0u8; 0x18];
        b[0xa..0xe].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        b[0xe..0x10].copy_from_slice(&0x9au16.to_le_bytes());
        let r = CmdDrawIndexedPatches16::read_from_bytes(&b).unwrap();
        assert_eq!({ r.control_point_index_buffer_reference }, 0x1234_5678);
        assert_eq!({ r.control_point_index_buffer_offset }, 0x9a);
        let mut b = [0u8; 0x30];
        b[8..0x10].copy_from_slice(&0x111u64.to_le_bytes());
        b[0x10..0x18].copy_from_slice(&0x222u64.to_le_bytes());
        let r = CmdDrawIndexedInstancedBasePrimitives64::read_from_bytes(&b).unwrap();
        assert_eq!((r.index_buffer_offset, r.index_count), (0x111, 0x222));
        let mut b = [0u8; 0x10];
        b[0..8].copy_from_slice(&0x40u64.to_le_bytes());
        b[8..0x10].copy_from_slice(&2u64.to_le_bytes());
        let r = CmdSetVisibiliyResultMode::read_from_bytes(&b).unwrap();
        assert_eq!((r.offset, r.mode), (0x40, 2));
    }

    /// `PGCmdMemoryBarrierScope` is a 4-byte word whose low 16 bits are the scope.
    #[test]
    fn memory_barrier_scope_is_the_low_16_bits() {
        let r = CmdMemoryBarrierScope::read_from_bytes(&0xdead_0004u32.to_le_bytes()).unwrap();
        assert_eq!(r.scope(), 4);
        assert_eq!(CmdMemoryBarrierScope { bits: 0xffff }.scope(), 0xffff);
        assert_eq!(CmdMemoryBarrierScope { bits: 0x1_0000 }.scope(), 0);
        assert!(CmdMemoryBarrierScope::read_from_bytes(&[0u8; 2]).is_err());
    }

    /// Info record 469 puts its reply reference at 0x28 and offset at 0x2c, after one pad byte
    /// that follows the 0x27-byte descriptor (`#pragma pack(4)`).
    #[test]
    fn heap_texture_size_and_align2_pack4_layout() {
        let mut b = [0u8; 0x34];
        b[3..7].copy_from_slice(&0x13u32.to_le_bytes());
        b[0x27] = 0xcc;
        b[0x28..0x2c].copy_from_slice(&9u32.to_le_bytes());
        b[0x2c..0x34].copy_from_slice(&0x1000u64.to_le_bytes());
        let r = CmdHeapTextureSizeAndAlignWithDescriptor2::read_from_bytes(&b).unwrap();
        let d = r.descriptor;
        assert_eq!({ d.usage }, 0x13);
        assert_eq!(r.unknown27, [0xcc]);
        assert_eq!(
            ({ r.reply_buffer_reference }, { r.reply_buffer_offset }),
            (9, 0x1000)
        );
    }

    /// Replies serialize to exactly their write size, their padding included (the padding bytes
    /// are part of the write).
    #[test]
    fn replies_serialize_at_their_sizes() {
        let r = ReplyComputePipelineStateInfo {
            max_total_threads_per_threadgroup: 1024,
            thread_execution_width: 32,
            static_threadgroup_memory_length: 0,
            support_indirect_command_buffers: 1,
            unknown19: [0; 3],
        };
        let bytes = r.as_bytes();
        assert_eq!(bytes.len(), 0x1c);
        assert_eq!(bytes[0..8], 1024u64.to_le_bytes());
        assert_eq!(bytes[8..0x10], 32u64.to_le_bytes());
        assert_eq!(bytes[0x18], 1);
        let r = ReplyRenderPipelineStateInfo {
            max_total_threads_per_threadgroup: 256,
            imageblock_sample_length: 8,
            threadgroup_size_matches_tile_size: 1,
            support_indirect_command_buffers: 0,
            unknown0a: [0; 2],
        };
        assert_eq!(r.as_bytes().len(), 0xc);
        assert_eq!(r.as_bytes()[8], 1);
        let r = ReplyRasterizationRateMapInfo {
            parameter_buffer_size: 0x100,
            parameter_buffer_align: 0x10,
            physical_granularity_width: 32,
            physical_granularity_height: 16,
        };
        assert_eq!(r.as_bytes().len(), 0x14);
        assert_eq!(r.as_bytes()[0x10..0x14], [32, 0, 16, 0]);
        let r = ReplyHostObjectInfo {
            gpu_resource_id: ResourceIdRecord { impl_: 0xfeed },
            resource_index: 3,
        };
        assert_eq!(r.as_bytes()[..8], 0xfeedu64.to_le_bytes());
        assert_eq!(r.as_bytes()[8..], 3u64.to_le_bytes());
        let r = ReplyTileDimensions {
            tile_width: 32,
            tile_height: 16,
        };
        assert_eq!(r.as_bytes(), &[32, 0, 16, 0]);
    }

    /// The segment header reads its padding byte and ignores it.
    #[test]
    fn segment_header_decodes() {
        let b = [0x20, 0, 0, 0, 5, 1, 0, 0xff];
        let h = SegmentHeader::read_from_bytes(&b).unwrap();
        assert_eq!(
            (
                h.length,
                h.encoder_type,
                h.continues_previous,
                h.continued_by_next
            ),
            (0x20, 5, 1, 0)
        );
        assert_eq!(h.unknown07, 0xff);
        assert!(SegmentHeader::read_from_bytes(&b[..7]).is_err());
    }

    /// The Metal struct copies convert to and from the platform types field for field.
    #[test]
    fn metal_records_convert() {
        let s = MtlSize::new(1, 2, 3);
        assert_eq!(MtlSize::from(MtlSizeRecord::from(s)), s);
        let o = MtlOrigin::new(4, 5, 6);
        assert_eq!(MtlOrigin::from(MtlOriginRecord::from(o)), o);
        let c = ClearColor {
            red: 0.25,
            green: 0.5,
            blue: 0.75,
            alpha: 1.0,
        };
        assert_eq!(ClearColor::from(ClearColorRecord::from(c)), c);
        let p = MtlCoordinate2D { x: 1.5, y: -2.0 };
        assert_eq!(MtlCoordinate2D::from(Coordinate2DRecord::from(p)), p);
        // A record's MTLRegion halves decode in the record's order: size, then origin.
        let mut b = [0u8; 0x30];
        for (i, v) in [7u64, 8, 9, 1, 2, 3].iter().enumerate() {
            b[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        let r = CmdSetComputeStageInRegion::read_from_bytes(&b).unwrap();
        assert_eq!(MtlSize::from(r.region_size), MtlSize::new(7, 8, 9));
        assert_eq!(MtlOrigin::from(r.region_origin), MtlOrigin::new(1, 2, 3));
        let r = CmdFillTextureWithColor::read_from_bytes(&[0u8; 0x5c]).unwrap();
        assert_eq!(ClearColor::from(r.color), ClearColor::default());
    }
}
