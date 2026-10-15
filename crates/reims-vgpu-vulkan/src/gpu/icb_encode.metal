// The tests' kernels that encode indirect commands (`encode_compute.rs`, `icb.rs`): each takes
// an indirect command buffer through an argument buffer. `reset_commands` resets its commands,
// one a thread; `set_compute_state` and `set_render_state` encode its command 0 with the
// pipeline state the argument buffer holds, which no setter gave the command. The vertex and
// fragment functions and the empty kernel are the states' own.
#include <metal_stdlib>
using namespace metal;

struct Commands {
    command_buffer commands [[id(0)]];
};

kernel void reset_commands(device Commands &held [[buffer(0)]],
                           uint index [[thread_position_in_grid]]) {
    compute_command(held.commands, index).reset();
}

struct ComputeState {
    command_buffer commands [[id(0)]];
    compute_pipeline_state state [[id(1)]];
};

kernel void set_compute_state(device ComputeState &held [[buffer(0)]]) {
    compute_command command(held.commands, 0);
    command.set_compute_pipeline_state(held.state);
    command.concurrent_dispatch_threadgroups(uint3(1, 1, 1), uint3(1, 1, 1));
}

struct RenderState {
    command_buffer commands [[id(0)]];
    render_pipeline_state state [[id(1)]];
};

kernel void set_render_state(device RenderState &held [[buffer(0)]]) {
    render_command command(held.commands, 0);
    command.set_render_pipeline_state(held.state);
    command.draw_primitives(primitive_type::triangle, 0, 3, 1, 0);
}

kernel void nothing() {}

// Writes 1 into buffer 0's first word: a state's program whose run shows.
kernel void mark(device uint *out [[buffer(0)]]) {
    out[0] = 1;
}

vertex float4 a_vertex() {
    return float4(0.0, 0.0, 0.0, 1.0);
}

fragment float4 a_colour() {
    return float4(1.0);
}
