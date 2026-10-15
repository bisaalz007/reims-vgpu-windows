// The tests' kernels that read, sample, gather and write a texture of one of Apple's RGBX formats
// (gpu/rgbx.rs), which build.rs compiles into rgbx.metallib. Each thread takes its texel at `g`
// of a 16 by 16 grid, and element `g.y * 16 + g.x` of buffer 0, so the tests compare Metal's
// texture with this platform's bit for bit.
#include <metal_stdlib>
using namespace metal;

// The grid's side, in texels.
constant uint SIDE = 16;

// The thread's element of buffer 0.
static uint at(uint2 g) {
    return g.y * SIDE + g.x;
}

// Whether the writers write the texel at `g`: two of every three, so a texel left alone keeps
// its bytes.
static bool written(uint2 g) {
    return (g.x + g.y) % 3 != 0;
}

// read() at `g` plus the offset in buffer 1: past the texture's 16 by 16 where it is 8 or more.
#define READ(name, type, out_type)                                                                 \
    kernel void name(texture2d<type, access::read> t [[texture(0)]],                              \
                     device out_type *out [[buffer(0)]],                                           \
                     constant uint2 &offset [[buffer(1)]],                                         \
                     uint2 g [[thread_position_in_grid]]) {                                        \
        out[at(g)] = out_type(t.read(g + offset));                                                 \
    }
READ(rgbx_read_float, float, float4)
READ(rgbx_read_half, half, half4)
READ(rgbx_read_uint, uint, uint4)
READ(rgbx_read_int, int, int4)

// sample() through sampler 0 at the normalized coordinates in buffer 1.
#define SAMPLE(name, type, out_type)                                                               \
    kernel void name(texture2d<type> t [[texture(0)]],                                            \
                     sampler s [[sampler(0)]],                                                     \
                     device out_type *out [[buffer(0)]],                                           \
                     device const float2 *where [[buffer(1)]],                                     \
                     uint2 g [[thread_position_in_grid]]) {                                        \
        out[at(g)] = out_type(t.sample(s, where[at(g)]));                                          \
    }
SAMPLE(rgbx_sample_float, float, float4)
SAMPLE(rgbx_sample_half, half, half4)
// As integers: an integer format's own samples, and the samples of a format of another class
// (rgbx.rs's cross_class_samples_are_metals).
SAMPLE(rgbx_sample_uint, uint, uint4)
SAMPLE(rgbx_sample_int, int, int4)

// gather() of red, green, blue and alpha through sampler 0 at the normalized coordinates in
// buffer 1: four elements a thread.
kernel void rgbx_gather(texture2d<float> t [[texture(0)]],
                        sampler s [[sampler(0)]],
                        device float4 *out [[buffer(0)]],
                        device const float2 *where [[buffer(1)]],
                        uint2 g [[thread_position_in_grid]]) {
    uint i = at(g);
    float2 c = where[i];
    out[4 * i] = t.gather(s, c, int2(0), component::x);
    out[4 * i + 1] = t.gather(s, c, int2(0), component::y);
    out[4 * i + 2] = t.gather(s, c, int2(0), component::z);
    out[4 * i + 3] = t.gather(s, c, int2(0), component::w);
}

// write() of the thread's element of buffer 0 where `written`.
#define WRITE(name, type, in_type)                                                                 \
    kernel void name(texture2d<type, access::write> t [[texture(0)]],                             \
                     device const in_type *values [[buffer(0)]],                                   \
                     uint2 g [[thread_position_in_grid]]) {                                        \
        if (written(g)) {                                                                          \
            t.write(values[at(g)], g);                                                             \
        }                                                                                          \
    }
WRITE(rgbx_write_float, float, float4)
WRITE(rgbx_write_half, half, half4)
WRITE(rgbx_write_uint, uint, uint4)
WRITE(rgbx_write_int, int, int4)

// write() of the thread's element of buffer 0 at every texel (the sRGB sweeps).
#define STORE(name, type, in_type)                                                                 \
    kernel void name(texture2d<type, access::write> t [[texture(0)]],                             \
                     device const in_type *values [[buffer(0)]],                                   \
                     uint2 g [[thread_position_in_grid]]) {                                        \
        t.write(values[at(g)], g);                                                                 \
    }
STORE(rgbx_store_float, float, float4)
STORE(rgbx_store_half, half, half4)

// read() of the texel into buffer 0, then write() of its components reversed, changed by
// `change`, where `written`.
#define READ_WRITE(name, type, vec, change)                                                        \
    kernel void name(texture2d<type, access::read_write> t [[texture(0)]],                        \
                     device vec *out [[buffer(0)]],                                                \
                     uint2 g [[thread_position_in_grid]]) {                                        \
        vec v = t.read(g);                                                                         \
        out[at(g)] = v;                                                                            \
        if (written(g)) {                                                                          \
            t.write(v.wzyx change, g);                                                             \
        }                                                                                          \
    }
READ_WRITE(rgbx_read_write_float, float, float4, *0.5f + 0.25f)
READ_WRITE(rgbx_read_write_half, half, half4, *0.5h + 0.25h)
READ_WRITE(rgbx_read_write_uint, uint, uint4, +3u)
READ_WRITE(rgbx_read_write_int, int, int4, -3)

// read() of texel `at(g)` of a texture buffer.
#define READ_BUFFER(name, type, out_type)                                                          \
    kernel void name(texture_buffer<type, access::read> t [[texture(0)]],                         \
                     device out_type *out [[buffer(0)]],                                           \
                     uint2 g [[thread_position_in_grid]]) {                                        \
        out[at(g)] = t.read(at(g));                                                                \
    }
READ_BUFFER(rgbx_read_buffer_float, float, float4)
READ_BUFFER(rgbx_read_buffer_uint, uint, uint4)
READ_BUFFER(rgbx_read_buffer_int, int, int4)

// read() of each of the four samples of a 4-sample texture at `g`, one after another.
kernel void rgbx_read_samples(texture2d_ms<float, access::read> t [[texture(0)]],
                              device float4 *out [[buffer(0)]],
                              uint2 g [[thread_position_in_grid]]) {
    for (uint s = 0; s < 4; s++) {
        out[at(g) * 4 + s] = t.read(g, s);
    }
}

// What an argument buffer holds of a texture a kernel reads as `type`: its resource ID, which
// the CPU writes; and the kernel's read() at `g` of it.
#define HELD_READ(holder, name, type, out_type)                                                    \
    struct holder {                                                                                \
        texture2d<type, access::read> t [[id(0)]];                                                 \
    };                                                                                             \
    kernel void name(device out_type *out [[buffer(0)]],                                           \
                     constant holder &held [[buffer(1)]],                                          \
                     uint2 g [[thread_position_in_grid]]) {                                        \
        out[at(g)] = held.t.read(g);                                                               \
    }
HELD_READ(ReadFloat, rgbx_read_held_float, float, float4)
HELD_READ(ReadHalf, rgbx_read_held_half, half, half4)
HELD_READ(ReadUint, rgbx_read_held_uint, uint, uint4)
HELD_READ(ReadInt, rgbx_read_held_int, int, int4)

// The width, height and level count of the texture an argument buffer holds, as a float
// texture.
kernel void rgbx_query_held(device uint4 *out [[buffer(0)]],
                            constant ReadFloat &held [[buffer(1)]],
                            uint2 g [[thread_position_in_grid]]) {
    out[at(g)] = uint4(held.t.get_width(), held.t.get_height(), held.t.get_num_mip_levels(), 0);
}

// What an argument buffer holds of a texture a kernel samples as floats: its resource ID.
struct SampleFloat {
    texture2d<float> t [[id(0)]];
};

// sample() of the texture an argument buffer holds, through sampler 0, at the normalized
// coordinates in buffer 2.
kernel void rgbx_sample_held(device float4 *out [[buffer(0)]],
                             constant SampleFloat &held [[buffer(1)]],
                             device const float2 *where [[buffer(2)]],
                             sampler s [[sampler(0)]],
                             uint2 g [[thread_position_in_grid]]) {
    out[at(g)] = held.t.sample(s, where[at(g)]);
}

// gather() of red of the texture an argument buffer holds, through sampler 0, at the normalized
// coordinates in buffer 2.
kernel void rgbx_gather_held(device float4 *out [[buffer(0)]],
                             constant SampleFloat &held [[buffer(1)]],
                             device const float2 *where [[buffer(2)]],
                             sampler s [[sampler(0)]],
                             uint2 g [[thread_position_in_grid]]) {
    out[at(g)] = held.t.gather(s, where[at(g)], int2(0), component::x);
}

// What an argument buffer holds of a texture a kernel writes.
#define HELD(holder, name, type, in_type)                                                          \
    struct holder {                                                                                \
        texture2d<type, access::write> t [[id(0)]];                                                \
    };                                                                                             \
    kernel void name(device const in_type *values [[buffer(0)]],                                   \
                     constant holder &held [[buffer(1)]],                                          \
                     uint2 g [[thread_position_in_grid]]) {                                        \
        held.t.write(values[at(g)], g);                                                            \
    }
HELD(WriteFloat, rgbx_write_held_float, float, float4)
HELD(WriteUint, rgbx_write_held_uint, uint, uint4)
HELD(WriteInt, rgbx_write_held_int, int, int4)
