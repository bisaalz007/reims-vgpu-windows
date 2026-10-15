"""The guest ABI, restated from byte offsets (docs/testing.md, section 6).

Nothing here is imported from the implementation: each constant is
restated from its offsets, and names the reims-vgpu-core item that reads it,
so that a refactor that changes a struct leaves these unchanged and the
suite catches it.
"""
import struct

PAGE = 0x4000  # guest::records::APV_PAGE_SIZE

# ---- MMIO registers, main BAR (section 6.2) ----
# index = (uint16_t)((offset - 0x1000) / 4) for reads and writes alike
# (device::mmio::register_index).
R_FIFO_ENABLE = 0x1000
R_FIFO_LENGTH = 0x1004
R_FIFO_WRITTEN = 0x1008
R_FIFO_READ = 0x100C
R_FIFO_START = 0x1010
R_IRQ_DISPLAY = 0x1014
R_IRQ_STAMP = 0x1018
R_ROOT_PAGE = 0x101C
R_WAKE_CHILD = 0x1020
R_RESUME_ROOT = 0x1024
R_RESUME_CHILD = 0x1028
R_FAULT_MASK = 0x102C
R_FIFO_BASE = 0x1030
R_VERSION = 0x1034
R_TRACE_MARKER = 0x1038
R_FAULT_OFFSET = 0x103C
R_EFI_ENABLE = 0x1200
R_EFI_MODE_COUNT = 0x1204
R_EFI_MODE_SELECT = 0x1208
R_EFI_MODE_SELECTED = 0x120C
R_EFI_START = 0x1210
R_EFI_LENGTH = 0x1214
R_EFI_DEPTH = 0x1218
R_EFI_MODE = 0x121C
R_DISPLAY_DOORBELL = 0x1220
R_EFI_STRIDE_ALIGN = 0x1224
R_EFI_STRIDE = 0x1228
R_PORT_COUNT = 0x122C
R_PORT_TYPES = 0x1234

MAX_VERSION = 0xF5  # guest::abi::features::HIGHEST_BINARY_VERSION
VALID_VERSIONS = (
    list(range(2, 7)) + [10, 11, 20, 21, 22, 30, 31, 40, 41, 42, 43, 60, 61, 62]
    + list(range(100, 114)) + list(range(200, 210)) + [230, 231, 232, 233, 235]
    + [237, 238, 245])

# ---- the command header (section 6.6) ----
# fifo::wire::FifoCommand: u16 opcode, u16 barrierCount, u32 length (the
# whole command), u32 stamp value; 12 bytes. Barriers follow: {u32
# fifoIndex, u32 value} (fifo::wire::StampWait).
HDR = struct.Struct("<HHII")
assert HDR.size == 12
BARRIER = struct.Struct("<II")

# ---- opcodes: fifo::commands::Opcode ----
OP_DEBUG = 0
OP_DISPLAY_SET_SHARED_STATE_PAGE = 1
OP_DISPLAY_ACK = 2
OP_DISPLAY_CURSOR_GLYPH = 4
OP_DISPLAY_CURSOR_SHOW = 5
OP_DISPLAY_TRANSACTION2 = 6
OP_DISPLAY_TRANSACTION3 = 7
OP_DISPLAY_SWAP_MAPPING = 8
OP_DISPLAY_SLEEP_STATE = 9
OP_DISPLAY_SET_PROPERTIES = 10
OP_DISPLAY_COMPOSITOR_PARAMETERS = 11
OP_DISPLAY_SET_GUEST_ICC_PROFILE = 12
OP_NOP = 30
OP_DELETE_TASK = 32
OP_UNMAP_MEMORY = 34
OP_DELETE_RESOURCE = 37
OP_DELETE_OBJECT = 40
OP_DEFINE_CHILD_FIFO = 48
OP_DELETE_CHILD_FIFO = 49
OP_SET_OBJECT_LIST = 51
OP_INVALIDATE_RESOURCES = 52
OP_SYNCHRONIZE_RESOURCES = 53
OP_DELETE_IOSURFACE_BACKING2 = 54
OP_EXEC_INDIRECT2 = 55
OP_DEFINE_TASK2 = 56
OP_MAP_MEMORY2 = 57
OP_GET_DEVICE_INFO = 58
OP_GET_COMPUTE_INFO = 59
OP_REPLACE_PHYSICAL = 60
OP_DELAY = 61
OP_SYNCHRONIZE_AND_DISCARD_RESOURCES = 62
OP_DISCARD_RESOURCES = 63
OP_HEAP_TEXTURE_SIZE_AND_ALIGN = 64
OP_DELETE_SHARED_TEXTURE_BACKING = 65
OP_RESET_RASTERIZATION_RATE_MAP = 66
OP_EXEC_INDIRECT3 = 67
OP_SET_OBJECT_AND_PLACEMENT_LIST = 68
OP_INFO_INDIRECT = 69
DEPRECATED_OPCODES = [3, 31, 33, 35, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47, 50]

# ---- the root node page (section 6.4) ----
ROOT_TASK_WORD = 0x000      # + 4 * taskID
ROOT_CHILD_NODE = 0x400     # + 0x14 * (fifoID - 1)
ROOT_DISPLAY_WORD = 0x900   # + 4 * port
CHILD_NODE_SIZE = 0x14

# ---- tasks (section 6.10) ----
# guest::records::TaskRoot2, packed.
TASK_ROOT = struct.Struct("<IIQIQQI")
assert TASK_ROOT.size == 40
PTE_WRITABLE = 0x80000000  # bit 31 of a leaf, memory::page_table

# ---- objects (section 6.11) ----
# guest::records::ObjectEntry, packed: u32 {type:8, length:24}, u64 task VA
# of the record.
OBJ_ENTRY = struct.Struct("<IQ")
assert OBJ_ENTRY.size == 12
# guest::records::ObjectPlacement.
PLACEMENT = struct.Struct("<QQ")

T_BUFFER = 1
T_TEXTURE = 2
T_TEXTURE_MIPGEN = 3
T_IOSURFACE_BACKING = 4
T_BACKING_TEXTURE = 5
T_FUNCTION = 6
T_SERIALIZED = 7
T_SERIALIZER_TEXTURE = 8
T_SERIALIZER_TEXTURE9 = 9
T_MAPPER_TEXTURE = 11
T_DUAL_PLANE_TEXTURE = 12
T_HEAP = 13
T_HEAP_BUFFER = 14
T_MAPPER_BUFFER = 15
T_SHARED_TEXTURE = 16
T_SHARED_BACKING1 = 17
T_SHARED_BACKING2 = 18
T_CPU_BUFFER = 19
T_ICB = 20

# ---- segments (section 6.13) ----
# deserializer::records::SegmentHeader: u32 length (header included), u8
# encoderType, u8 continuesPrevious, u8 continuedByNext, u8 pad.
SEG_HDR = struct.Struct("<IBBBx")
ENC_RENDER, ENC_COMPUTE, ENC_BLIT, ENC_INFO, ENC_PROTECTION = 0, 1, 2, 4, 5
# deserializer::records::CmdHeader: u32 commandID, u32 length.
REC_HDR = struct.Struct("<II")

# Exec resource flags, guest::records::CmdExecIndirectResource3.
RES_INVALIDATE_HOST = 1
RES_INVALIDATE_GUEST = 2
RES_SYNCHRONIZE = 4
