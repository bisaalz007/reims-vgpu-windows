// The tests' kernels that write a slice of a kernel's explicit imageblock into a texture
// (`air.write_imageblock_slice_to_texture_2d`), as macOS 26's glass blur writes its mip texture,
// of usage ShaderRead | RenderTarget | BlockWritesOnly (0x10005). Hand-written: no guest data.
#include <metal_stdlib>
using namespace metal;

// One texel of the imageblock.
struct Px {
    half4 c;
};

// Each thread puts a colour of its position in the imageblock, and thread (0, 0) writes the
// threadgroup's slice of it into dst at the threadgroup's origin, at level `lod`, read from a
// buffer, so not a constant.
kernel void slice_write(imageblock<Px, imageblock_layout_explicit> ib,
                        texture2d<half, access::write> dst [[texture(0)]],
                        constant uint &lod [[buffer(0)]],
                        ushort2 lid [[thread_position_in_threadgroup]],
                        ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tgs [[threads_per_threadgroup]]) {
    threadgroup_imageblock Px *p = ib.data(lid);
    p->c = half4(half(lid.x + 1) / 16.0h, half(lid.y + 1) / 16.0h, 0.5h, 1.0h);
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    if (lid.x == 0 && lid.y == 0) {
        dst.write(ib.slice(p->c), tg * tgs, lod);
    }
}

// As `slice_write`, at level 0, a constant.
kernel void slice_write_l0(imageblock<Px, imageblock_layout_explicit> ib,
                           texture2d<half, access::write> dst [[texture(0)]],
                           ushort2 lid [[thread_position_in_threadgroup]],
                           ushort2 tg [[threadgroup_position_in_grid]],
                           ushort2 tgs [[threads_per_threadgroup]]) {
    threadgroup_imageblock Px *p = ib.data(lid);
    p->c = half4(half(lid.x + 1) / 16.0h, half(lid.y + 1) / 16.0h, 0.5h, 1.0h);
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    if (lid.x == 0 && lid.y == 0) {
        dst.write(ib.slice(p->c), tg * tgs);
    }
}
