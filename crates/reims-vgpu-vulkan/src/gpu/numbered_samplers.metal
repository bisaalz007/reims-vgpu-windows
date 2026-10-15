// The tests' functions whose pipeline states take sampler IDs as Metal's creation numbers them
// (`states.rs`, `sampler_ids_follow_pipeline_holders_as_metals`): a vertex function that uses no
// texture, a fragment function that reads texture 0, and kernels that read texture 0, sample it
// through a constexpr sampler, and write it. build.rs compiles it into numbered_samplers.metallib.
#include <metal_stdlib>
using namespace metal;

struct NumberedVertex {
    float4 position [[position]];
};

vertex NumberedVertex numbered_vertex(uint vid [[vertex_id]]) {
    NumberedVertex out;
    out.position = float4(float(vid & 1u), float(vid >> 1), 0.0f, 1.0f);
    return out;
}

fragment float4 numbered_fragment_read(NumberedVertex in [[stage_in]],
                                       texture2d<float, access::read> t [[texture(0)]]) {
    return t.read(uint2(in.position.xy));
}

kernel void numbered_read(texture2d<float, access::read> t [[texture(0)]],
                          device float4 *out [[buffer(0)]],
                          uint i [[thread_position_in_grid]]) {
    out[i] = t.read(uint2(i, 0));
}

constexpr sampler linear_edge(coord::normalized, address::clamp_to_edge, filter::linear);

kernel void numbered_sample(texture2d<float> t [[texture(0)]],
                            device float4 *out [[buffer(0)]],
                            uint i [[thread_position_in_grid]]) {
    out[i] = t.sample(linear_edge, float2(float(i) / 16.0f, 0.5f));
}

kernel void numbered_write(texture2d<float, access::write> t [[texture(0)]],
                           uint i [[thread_position_in_grid]]) {
    t.write(float4(1.0f), uint2(i, 0));
}
