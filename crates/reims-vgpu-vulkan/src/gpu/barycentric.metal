// The tests' render functions that read `[[barycentric_coord]]` (`encode_render.rs`), over the
// vertices of `rdec_flat_v`'s buffers: positions from buffer 0 and colours from buffer 1, by
// `vertex_id`. The fragment functions write the weights as red, green and blue; the flat one
// writes its flat colour's blue in place of the third weight.
#include <metal_stdlib>
using namespace metal;

struct Smooth {
    float4 position [[position]];
    float4 color;
};

struct Flat {
    float4 position [[position]];
    float4 color [[flat]];
};

vertex Smooth smooth_vertex(uint vid [[vertex_id]], const device float4 *positions [[buffer(0)]],
                            const device float4 *colors [[buffer(1)]]) {
    return Smooth{positions[vid], colors[vid]};
}

vertex Flat flat_vertex(uint vid [[vertex_id]], const device float4 *positions [[buffer(0)]],
                        const device float4 *colors [[buffer(1)]]) {
    return Flat{positions[vid], colors[vid]};
}

fragment float4 smooth_weights(Smooth in [[stage_in]], float3 weights [[barycentric_coord]]) {
    return float4(weights, 1);
}

fragment float4 flat_weights(Flat in [[stage_in]], float3 weights [[barycentric_coord]]) {
    return float4(weights.xy, in.color.z, 1);
}
