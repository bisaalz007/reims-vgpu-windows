// The tests' functions that Metal's compiler refuses on a device without Apple10
// (`states::tests`): render state setters on an indirect render command in every stage, and a
// fragment function's read of a program-scope `thread_index_in_quadgroup`, each beside a function
// Metal makes.
#include <metal_stdlib>
using namespace metal;

struct Commands {
    command_buffer commands [[id(0)]];
    depth_stencil_state depth_stencil [[id(1)]];
};

struct Vertex {
    float4 position [[position]];
};

struct Tile {
    float4 color [[color(0)]];
};

kernel void k_cull(device Commands &a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    render_command c(a.commands, i);
    c.set_cull_mode(cull_mode::back);
}

kernel void k_depth_stencil(device Commands &a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    render_command c(a.commands, i);
    c.set_depth_stencil_state(a.depth_stencil);
}

kernel void k_draw(device Commands &a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    render_command c(a.commands, i);
    c.draw_primitives(primitive_type::triangle, 0, 3, 1, 0);
}

vertex Vertex v_plain(uint i [[vertex_id]]) {
    float2 q = float2((i << 1) & 2, i & 2);
    Vertex v;
    v.position = float4(q * 2 - 1, 0, 1);
    return v;
}

vertex Vertex v_cull(device Commands &a [[buffer(0)]], uint i [[vertex_id]]) {
    render_command c(a.commands, i);
    c.set_cull_mode(cull_mode::back);
    Vertex v;
    v.position = float4(0, 0, 0, 1);
    return v;
}

fragment float4 f_plain(Vertex v [[stage_in]]) {
    return float4(1);
}

fragment float4 f_cull(device Commands &a [[buffer(0)]], Vertex v [[stage_in]]) {
    render_command c(a.commands, 0);
    c.set_cull_mode(cull_mode::back);
    return float4(1);
}

fragment float4 f_draw(device Commands &a [[buffer(0)]], Vertex v [[stage_in]]) {
    render_command c(a.commands, 0);
    c.draw_primitives(primitive_type::triangle, 0, 3, 1, 0);
    return float4(1);
}

uint quad_lane [[thread_index_in_quadgroup]];
ushort quad_lane16 [[thread_index_in_quadgroup]];
uint simd_width [[threads_per_simdgroup]];

fragment float4 f_quad_lane(Vertex v [[stage_in]]) {
    return float4(float(quad_lane));
}

fragment float4 f_quad_lane16(Vertex v [[stage_in]]) {
    return float4(float(quad_lane16));
}

fragment float4 f_simd_width(Vertex v [[stage_in]]) {
    return float4(float(simd_width));
}

kernel void t_cull(imageblock<Tile, imageblock_layout_implicit> block,
                   device Commands &a [[buffer(0)]],
                   ushort2 lid [[thread_position_in_threadgroup]]) {
    render_command c(a.commands, 0);
    c.set_cull_mode(cull_mode::back);
}

kernel void t_draw(imageblock<Tile, imageblock_layout_implicit> block,
                   device Commands &a [[buffer(0)]],
                   ushort2 lid [[thread_position_in_threadgroup]]) {
    render_command c(a.commands, 0);
    c.draw_primitives(primitive_type::triangle, 0, 3, 1, 0);
}
