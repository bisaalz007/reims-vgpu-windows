// The tests' kernel whose pipeline state numbers five constexpr samplers, each its own state
// (`states.rs`, `pipeline_samplers_grow_metals_sampler_mempool`): five sampler IDs taken past the
// sampler states'. build.rs compiles it into five_samplers.metallib.
#include <metal_stdlib>
using namespace metal;

constexpr sampler five_a(coord::normalized, address::clamp_to_edge, filter::linear);
constexpr sampler five_b(coord::normalized, address::repeat, filter::linear);
constexpr sampler five_c(coord::normalized, address::mirrored_repeat, filter::linear);
constexpr sampler five_d(coord::normalized, address::clamp_to_edge, filter::nearest);
constexpr sampler five_e(coord::normalized, address::repeat, filter::nearest);

kernel void five_samplers(texture2d<float> t [[texture(0)]],
                          device float4 *out [[buffer(0)]],
                          uint i [[thread_position_in_grid]]) {
    float2 at = float2(float(i) / 16.0f, 0.5f);
    out[i] = t.sample(five_a, at) + t.sample(five_b, at) + t.sample(five_c, at) +
             t.sample(five_d, at) + t.sample(five_e, at);
}
