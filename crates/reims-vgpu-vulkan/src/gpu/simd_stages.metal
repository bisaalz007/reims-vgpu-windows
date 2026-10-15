// The tests' functions that use SIMD-group operations (`encode_render.rs`), of the classes the
// guest's device-info key 33 answers for: `simd_sum`, a reduction (0x40), and
// `simd_shuffle_and_fill_down` of an explicit width, a modulo of 8, a shuffle and fill (0x100).
// Each runs over a lane value: a kernel thread's position, a fragment's pixel index in a 16 by 16
// target, a vertex's ID. `simd_kernel` writes both results of each thread to buffer 0;
// `sum_fragment` and `fill_fragment` write one as colour, its low byte as red and the next as
// green. The vertex functions place the corners of one triangle covering the target and pass
// `value_fragment`, which writes it as colour as those do, a value: `plain_vertex` its vertex ID,
// `sum_vertex` and `fill_vertex` their results.
#include <metal_stdlib>
using namespace metal;

static uint fill(uint v) {
    return simd_shuffle_and_fill_down(v, v ^ 0xffu, ushort(3), ushort(8));
}

kernel void simd_kernel(device uint2 *out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    uint v = i * 7u + 1u;
    out[i] = uint2(simd_sum(v), fill(v));
}

struct Lane {
    float4 position [[position]];
    uint value [[flat]];
};

static Lane corner(uint vid, uint value) {
    const float2 p[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return Lane{float4(p[vid % 3], 0, 1), value};
}

vertex Lane plain_vertex(uint vid [[vertex_id]]) {
    return corner(vid, vid);
}

vertex Lane sum_vertex(uint vid [[vertex_id]]) {
    return corner(vid, simd_sum(vid));
}

vertex Lane fill_vertex(uint vid [[vertex_id]]) {
    return corner(vid, fill(vid));
}

static float4 colour(uint v) {
    return float4(float(v & 0xffu), float((v >> 8) & 0xffu), 0, 255) / 255.0;
}

static uint pixel(Lane in) {
    uint2 g = uint2(in.position.xy);
    return g.y * 16u + g.x;
}

fragment float4 value_fragment(Lane in [[stage_in]]) {
    return colour(in.value);
}

fragment float4 sum_fragment(Lane in [[stage_in]]) {
    return colour(simd_sum(pixel(in)));
}

fragment float4 fill_fragment(Lane in [[stage_in]]) {
    return colour(fill(pixel(in)));
}
