// The tests' functions that import `ext` from a dynamic library (`states::tests`), which the
// build links against three libraries, each naming another install name: one that exports it,
// one whose file at its install name exports something else, and one with no file there.
#include <metal_stdlib>
using namespace metal;

extern int ext(int x);

struct Vertex {
    float4 position [[position]];
};

struct Tile {
    float4 color [[color(0)]];
};

kernel void k_import(device int *o [[buffer(0)]]) {
    o[0] = ext(o[0]);
}

vertex Vertex v_import(uint i [[vertex_id]]) {
    float2 q = float2((i << 1) & 2, i & 2);
    Vertex v;
    v.position = float4(q * 2 - 1, 0, 1);
    return v;
}

fragment float4 f_import(Vertex v [[stage_in]], constant int &k [[buffer(0)]]) {
    return float4(float(ext(k)));
}

kernel void t_import(imageblock<Tile, imageblock_layout_implicit> block,
                     constant int &k [[buffer(0)]],
                     ushort2 lid [[thread_position_in_threadgroup]]) {
    Tile t;
    t.color = float4(float(ext(k)));
    block.write(t, lid);
}
