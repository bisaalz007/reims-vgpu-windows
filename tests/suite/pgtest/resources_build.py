"""Builders for resource objects and textures, and the blit records and FIFO
commands that move their data (docs/testing.md, sections 4.10, 6.11, 6.14,
6.19, 6.20).

Layouts name the reims-vgpu-core module or item that reads them.
"""
import struct

from . import abi
from .abi import PAGE
from .task import GPUTask, rec, seg

# ---- pixel formats (MTLPixelFormat, Metal's public enum) ----
R8Unorm = 10
RG8Unorm = 30
R32Float = 55
RGBA8Unorm = 70
BGRA8Unorm = 80
RGBA16Float = 115
RGBA32Uint = 123
Depth32Float = 252
Depth32Float_Stencil8 = 260
X32_Stencil8 = 261
BC1_RGBA = 130
ASTC_4x4_LDR = 204
PVRTC_RGBA_4BPP = 162
XR10_2P_A8 = 550      # not public (resources::texture::pixel_format)
XR10_2P_A8_SRGB = 551

# MTLTextureType
TT_1D, TT_1DArray, TT_2D, TT_2DArray, TT_2DMS, TT_Cube, TT_CubeArray, TT_3D = 0, 1, 2, 3, 4, 5, 6, 7
TT_2DMSArray, TT_TextureBuffer = 8, 9

# MTLTextureUsage
U_READ, U_WRITE, U_RT = 1, 2, 4

# MTLResourceOptions: storage mode << 4
OPT_SHARED, OPT_MANAGED, OPT_PRIVATE, OPT_MEMORYLESS = 0x00, 0x10, 0x20, 0x30


# ---- object records (resources::manager::records) ----

def buffer_rec(length, page, flags=0):
    """Type 1: ObjectBuffer {u64 length, u32 page, u8 flags}, 16 bytes;
    flags bit 0 private, bit 1 read-only."""
    return struct.pack("<QIB3x", length, page, flags)


def cpu_buffer_rec(length, page):
    """Type 19: ObjectCpuBuffer, packed {u64 length, u32 page}."""
    return struct.pack("<QI", length, page)


def heap_rec(page, length, flags=0):
    """Type 13: packed {u32 page, u64 length, u8 flags}; bit 0 private, 1
    tracked, 2 read-only. A heap's creation reads 16 bytes (ObjectHeap)."""
    return struct.pack("<IQB3x", page, length, flags)


def heap_buffer_rec(heap_ref, offset, length):
    """Type 14: packed {u32 heapReference, u64 offset, u64 length}, 0x14
    (ObjectHeapBuffer)."""
    return struct.pack("<IQQ", heap_ref, offset, length)


def mapper_ref_rec(mapping_id):
    """Types 15 and 11's first field: {u64 mappingID} (ObjectMapperRefBuffer,
    and resources::manager::texture's ObjectMapperRefTexture)."""
    return struct.pack("<Q", mapping_id)


# ---- serialized texture descriptors (deserializer::texture_descriptor) ----

def desc1(fmt=RGBA8Unorm, w=4, h=4, d=1, tt=TT_2D, usage=U_READ | U_WRITE, mips=1, samples=1,
          array=1, options=OPT_PRIVATE, prot=0, fbonly=0, drawable=0, gpuopt=0):
    """v1, 32 bytes: u32 {textureType:4, framebufferOnly:1, isDrawable:1,
    allowGPUOptimizedContents:1, :1, usage:8, pixelFormat:16}, u32 width,
    height, depth, u16 mips, samples, arrayLength, resourceOptions, u64
    protectionOptions (SerializedTextureDescriptor)."""
    w0 = (tt & 0xF) | (fbonly & 1) << 4 | (drawable & 1) << 5 | (gpuopt & 1) << 6 | (usage & 0xFF) << 8 | (fmt & 0xFFFF) << 16
    return struct.pack("<IIIIHHHHQ", w0, w, h, d, mips, samples, array, options, prot)


def desc2(fmt=RGBA8Unorm, w=4, h=4, d=1, tt=TT_2D, usage=U_READ | U_WRITE, mips=1, samples=1,
          array=1, options=OPT_PRIVATE, prot=0, swizzle=(2, 3, 4, 5), wswz=0, fbonly=0, drawable=0, gpuopt=0):
    """v2, packed, 0x27 bytes: 3 bytes of bit-fields {textureType:4,
    framebufferOnly:1, isDrawable:1, allowGPUOptimizedContents:1,
    writeSwizzleEnabled:1, pixelFormat:16}, u32 usage @3, width @7, height
    @0xb, depth @0xf, u16 mips @0x13, samples, arrayLength, resourceOptions,
    u64 protection @0x1b, swizzle RGBA @0x23 (SerializedTextureDescriptor2; a
    channel >= 6 becomes One)."""
    b0 = (tt & 0xF) | (fbonly & 1) << 4 | (drawable & 1) << 5 | (gpuopt & 1) << 6 | (wswz & 1) << 7
    d2 = struct.pack("<BH", b0, fmt & 0xFFFF) + struct.pack("<IIIIHHHHQ", usage, w, h, d, mips, samples, array, options, prot)
    d2 += bytes(swizzle)
    assert len(d2) == 0x27
    return d2


def op_desc(d):
    """{u32 unread, descriptor}, padded to 4 bytes as the pack(4) operation
    structs are: 0x24 with a v1 descriptor, 0x2c with a v2."""
    body = struct.pack("<I", 0) + d
    return body + bytes(-len(body) % 4)


def op_hdr(command_type, unread=0):
    """OperationHeader {u32 commandType, u32 unread}
    (resources::manager::records)."""
    return struct.pack("<II", command_type, unread)


# ---- normal textures: dimension blocks (resources::texture::dimension) ----

def mip(offset, size, bpr, w, h=1, d=1):
    """TextureMipDimension, packed 0x24: u64 offset, size, bytesPerRow, u32
    width, height, depth."""
    return struct.pack("<QQQIII", offset, size, bpr, w, h, d)


def dimension(mips, array=1, base=0, slice_bytes=0, block=(1, 1, 1, 4), cube=0, compressed=0, levels=None):
    """TextureDimension, packed 0x18: u16 {mipLevels:14, isCube:1,
    isCompressed:1}, u16 arrayLength, u64 baseOffset @4, u64 bytesPerSlice
    @0xc, u8 blockWidth, blockHeight, blockDepth, blockBytes @0x14-0x17, then
    the mips. levels overrides the count field (default len(mips))."""
    n = len(mips) if levels is None else levels
    head = struct.pack("<HHQQBBBB", (n & 0x3FFF) | (cube & 1) << 14 | (compressed & 1) << 15, array, base,
                       slice_bytes, *block)
    return head + b"".join(mips)


def simple_dim(w, h, bpp=4, d=1, array=1, levels=1, cube=0, slice_bytes=0, base=0, bpr=None):
    """A packed mip chain for an uncompressed w x h x d texture: each level's
    rows tight (or bpr), levels one after another."""
    mips = []
    off = 0
    for _ in range(levels):
        r = bpr if bpr is not None else w * bpp
        size = r * h * d
        mips.append(mip(off, size, r, w, h, d))
        off += size
        w, h, d = max(w // 2, 1), max(h // 2, 1), max(d // 2, 1)
    return dimension(mips, array=array, base=base, slice_bytes=slice_bytes, block=(1, 1, 1, bpp), cube=cube)


def texture_rec(length, page, dims, desc=None, command_type=1, op=None):
    """Types 2, 3 and 12: ObjectTexture packed {u64 length, u32 page} (12),
    then the dimension block(s), the operation header, and OpWithDescriptor
    (2) {u32 unread, descriptor} (resources::manager::texture)."""
    if op is None:
        op = op_desc(desc if desc is not None else desc1())
    return struct.pack("<QI", length, page) + b"".join(dims) + op_hdr(command_type) + op


# ---- serializer textures (types 8, 9) ----

def ser_memoryless(desc=None, command_type=1):
    """commandType 1 or 52: {u32, descriptor}; must be memoryless."""
    return op_hdr(command_type) + op_desc(desc if desc is not None else desc1(options=OPT_MEMORYLESS, usage=U_RT))


def ser_view_format(base_ref, fmt):
    """commandType 7: pack(4) {u32, u32 base, u16 pixelFormat}, 0xc."""
    return op_hdr(7) + struct.pack("<IIH2x", 0, base_ref, fmt)


def ser_view_type(base_ref, fmt, tt, levels=(0, 1), slices=(0, 1), swizzle=None):
    """commandType 8 (0x2c) or with a swizzle 27 (0x30): pack(4) {u32, u32
    base, u16 pixelFormat, u16 textureType, NSRange levels @0xc, NSRange
    slices @0x1c, [u8 swizzle[4] @0x2c]}."""
    body = struct.pack("<IIHHQQQQ", 0, base_ref, fmt, tt, levels[0], levels[1], slices[0], slices[1])
    if swizzle is None:
        return op_hdr(8) + body
    return op_hdr(27) + body + bytes(swizzle)


def ser_on_buffer(buf_ref, offset, bpr, desc=None, v2=False):
    """commandType 9 (0x38) or 55 (0x40): natural layout {u32, u32 buffer, u64
    offset @8, u64 bytesPerRow @0x10, descriptor @0x18}."""
    if v2:
        d = desc if desc is not None else desc2(options=OPT_SHARED)
        body = struct.pack("<IIQQ", 0, buf_ref, offset, bpr) + d + b"\0"
        return op_hdr(55) + body
    d = desc if desc is not None else desc1(options=OPT_SHARED)
    return op_hdr(9) + struct.pack("<IIQQ", 0, buf_ref, offset, bpr) + d


def ser_in_heap(heap_ref, offset, desc=None, v2=False):
    """commandType 21 (0x34) or 56 (0x3c): pack(4) {u32, u32 heap,
    descriptor @8, u32 unread, u64 offset}."""
    if v2:
        d = desc if desc is not None else desc2()
        return op_hdr(56) + struct.pack("<II", 0, heap_ref) + d + b"\0" + struct.pack("<IQ", 0, offset)
    d = desc if desc is not None else desc1()
    return op_hdr(21) + struct.pack("<II", 0, heap_ref) + d + struct.pack("<IQ", 0, offset)


# ---- IOSurface backings and textures over them ----

def backing_rec(length, page, pixel_format, planes, count=None):
    """Type 4: pack(4) {u64 length, u32 page, u32 pixelFormat, u8 planeCount}
    (0x14), then planeCount x BackingPlane {u32 offset, width, height, u32
    bytesPerRow:24 | bytesPerElement:8} (resources::texture::backing)."""
    n = len(planes) if count is None else count
    out = struct.pack("<QIIB3x", length, page, pixel_format, n)
    for off, w, h, bpr, bpe in planes:
        out += struct.pack("<IIII", off, w, h, (bpr & 0xFFFFFF) | (bpe & 0xFF) << 24)
    return out


def iosurface_op(command_type=12, desc=None, plane=0, rotation=0):
    """commandType 12 (0x28: {u32, v1 desc @4, u16 plane @0x24}), 47 (with u16
    rotation @0x26) or 57 (0x30: {u32, v2 desc @4, u16 plane @0x2c, u16
    rotation @0x2e})."""
    if command_type == 57:
        d = desc if desc is not None else desc2(options=OPT_SHARED)
        return op_hdr(57) + struct.pack("<I", 0) + d + b"\0" + struct.pack("<HH", plane, rotation)
    d = desc if desc is not None else desc1(options=OPT_SHARED)
    if command_type == 47:
        return op_hdr(47) + struct.pack("<I", 0) + d + struct.pack("<HH", plane, rotation)
    return op_hdr(command_type) + struct.pack("<I", 0) + d + struct.pack("<HH", plane, 0)


def ref_texture_rec(backing_id, task_id=0, op=None):
    """Type 5: ObjectRefTexture {u32 backingID, u32 taskID}, then the
    header and an IOSurface op."""
    return struct.pack("<II", backing_id, task_id) + (op if op is not None else iosurface_op())


def shared_texture_rec(backing_id, command_type=49):
    """Type 16: {u32 sharedTextureBackingID} + header (49)."""
    return struct.pack("<I", backing_id) + op_hdr(command_type)


def shared_backing_rec(length, page, dims, desc=None, command_type=48):
    """Types 17 and 18: packed {u64 length, u32 page}, dimension(s), header
    (48 or 53), {u32, descriptor}."""
    if command_type == 53:
        d = desc if desc is not None else desc2()
    else:
        d = desc if desc is not None else desc1()
    return struct.pack("<QI", length, page) + b"".join(dims) + op_hdr(command_type) + op_desc(d)


# ---- blit records (deserializer::records, section 6.14) ----

def b_copy_buf_buf(src, dst, soff, doff, size):
    return rec(301, struct.pack("<IIQQQ", src, dst, soff, doff, size))


def b_fill_buf(buf, off, length, value):
    return rec(306, struct.pack("<IQQB3x", buf, off, length, value))


def b_copy_buf_tex(buf, tex, soff, bpr, bpi, size, origin=(0, 0, 0), slice_=0, level=0, options=0):
    """300: {u32 src, u32 dst, u64 srcOff, bpr, bpi, MTLSize, MTLOrigin, u16
    slice, u16 level, u32 options}, 0x58."""
    return rec(300, struct.pack("<IIQQQQQQQQQHHI", buf, tex, soff, bpr, bpi, *size, *origin, slice_, level, options))


def b_copy_tex_buf(tex, buf, doff, bpr, bpi, size, origin=(0, 0, 0), slice_=0, level=0, options=0):
    """302: {u32 tex, u32 buf, MTLOrigin, MTLSize, u64 dstOff, bpr, bpi, u16
    slice, level, options}, 0x58."""
    return rec(302, struct.pack("<IIQQQQQQQQQHHH2x", tex, buf, *origin, *size, doff, bpr, bpi, slice_, level, options))


def b_copy_tex_tex(src, dst, size, sorigin=(0, 0, 0), dorigin=(0, 0, 0), sslice=0, slevel=0, dslice=0, dlevel=0):
    """303: {u32 src, u32 dst, MTLOrigin, MTLSize, MTLOrigin, u16 x4}, 0x58."""
    return rec(303, struct.pack("<IIQQQQQQQQQHHHH", src, dst, *sorigin, *size, *dorigin, sslice, slevel, dslice, dlevel))


def b_gen_mipmaps(tex):
    return rec(307, struct.pack("<I", tex))


def b_sync_resource(ref):
    return rec(314, struct.pack("<I", ref))


def b_fill_tex_color(tex, size, color, fmt, origin=(0, 0, 0), level=0, slice_=0):
    """321: pack(4) {u32 tex, u16 level, u16 slice, MTLSize, MTLOrigin,
    MTLClearColor (4 x f64), u16 pixelFormat}, 0x5c."""
    return rec(321, struct.pack("<IHHQQQQQQddddH2x", tex, level, slice_, *size, *origin, *color, fmt))


# ---- FIFO commands on resources (fifo::commands::task) ----

def cmd_invalidate(s, tid, entries, **kw):
    """52: {u32 taskID, u32 count}, count x {u32 ref, u8 clearHostValid, u8
    setHostValid, u8 clearGuestValid, u8 setGuestValid}."""
    p = struct.pack("<II", tid, len(entries)) + b"".join(struct.pack("<IBBBB", *e) for e in entries)
    return s.submit(abi.OP_INVALIDATE_RESOURCES, p, **kw)


def _reflist(tid, refs):
    return struct.pack("<II", tid, len(refs)) + b"".join(struct.pack("<I", r) for r in refs)


def cmd_synchronize(s, tid, refs, **kw):
    """53: {u32 taskID, u32 count}, count x u32 ref."""
    return s.submit(abi.OP_SYNCHRONIZE_RESOURCES, _reflist(tid, refs), **kw)


def cmd_sync_discard(s, tid, refs, **kw):
    """62: as 53, then discards."""
    return s.submit(abi.OP_SYNCHRONIZE_AND_DISCARD_RESOURCES, _reflist(tid, refs), **kw)


def cmd_discard(s, tid, refs, **kw):
    """63: as 53."""
    return s.submit(abi.OP_DISCARD_RESOURCES, _reflist(tid, refs), **kw)


def cmd_replace_physical(s, tid, ref, **kw):
    """60: {u32 taskID, u32 reference}."""
    return s.submit(abi.OP_REPLACE_PHYSICAL, struct.pack("<II", tid, ref), **kw)


def cmd_delete_resource(s, tid, ref, **kw):
    """37: {u32 taskID, u32 reference}."""
    return s.submit(abi.OP_DELETE_RESOURCE, struct.pack("<II", tid, ref), **kw)


def cmd_delete_object(s, tid, operation, ref, length=12, extra=b"", **kw):
    """40: {u32 taskID, u32 operationID, u32 length, ...}; the task receives
    the payload from operationID on, and words[2] is the reference."""
    return s.submit(abi.OP_DELETE_OBJECT, struct.pack("<IIII", tid, operation, length, ref) + extra, **kw)


def cmd_delete_backing(s, ref, tid=0, **kw):
    """54: {u32 backingReference, u32 taskID (must be 0)}."""
    return s.submit(abi.OP_DELETE_IOSURFACE_BACKING2, struct.pack("<II", ref, tid), **kw)


def cmd_delete_shared_backing(s, ref, **kw):
    """65: {u32 reference} on the kernel task."""
    return s.submit(abi.OP_DELETE_SHARED_TEXTURE_BACKING, struct.pack("<I", ref), **kw)


# ---- helpers ----

def fault_and_resume(s, label):
    """Waits for the fault, records the fault state, resumes the root FIFO."""
    s.wait_fault(label=label)
    s.fault_state(label)
    s.resume_root()


class RTask(GPUTask):
    """A GPUTask with helpers for resource scenarios. Records go in pages 1-3
    and command buffers in pages 4-7 (GPUTask); data pages from 8."""

    def add(self, typ, record):
        return self.ol.add(typ, record)

    def area(self, length=PAGE, contents=None, fill=None):
        """Reserves task pages for a resource's bytes; returns the VA."""
        va = self.reserve(max(length, 1))
        if contents is not None:
            self.write(va, contents)
        elif fill is not None:
            self.fill(va, length, fill)
        return va

    def shared(self, length=PAGE, contents=None, fill=None):
        """A shared buffer (type 1, flags 0); returns (ref, va)."""
        va = self.area(length, contents, fill)
        return self.add(abi.T_BUFFER, buffer_rec(length, va // PAGE)), va

    def blit(self, *records, resources=(), fault=False, label=None, opcode=None):
        """One blit segment in one command buffer, executed with opcode 67,
        or 55 (ExecIndirect2) when the scenario's version (s.pg_version) is
        below 109, where 67 does not exist."""
        self.reset_cmdbufs()
        cb = self.cmdbuf(seg(abi.ENC_BLIT, b"".join(records)))
        if opcode is None:
            opcode = abi.OP_EXEC_INDIRECT3 if getattr(self.s, "pg_version", 245) >= 109 else abi.OP_EXEC_INDIRECT2
        if opcode == abi.OP_EXEC_INDIRECT2:
            res = [(r, f & 1, (f >> 1) & 1, (f >> 2) & 1) for r, f in resources]
            self.exec2([cb], resources=res, fault=fault, label=label)
            if fault:
                self.s.fault_state(label)
                self.s.resume_root()
            return
        if fault:
            self.exec3([cb], resources=resources, opcode=opcode, fault=True, label=label)
            self.s.fault_state(label)
            self.s.resume_root()
        else:
            self.exec3([cb], resources=resources, opcode=opcode, label=label)
