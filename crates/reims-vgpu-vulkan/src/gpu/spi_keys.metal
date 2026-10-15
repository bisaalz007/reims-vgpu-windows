// The tests' render functions for Metal's private render keys (`serialized/draws.rs`): a
// triangle over the whole target at depth `z` (vertex buffer 0), its clip distances, none, one
// or two, cutting off the target's left half and its top half, and a fragment of `colour`
// (fragment buffer 0), over the whole target or its right half.
#include <metal_stdlib>
using namespace metal;

constant float2 corners[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};

struct Covered {
    float4 position [[position]];
};

struct Clipped1 {
    float4 position [[position]];
    float clip [[clip_distance]] [1];
};

struct Clipped2 {
    float4 position [[position]];
    float clip [[clip_distance]] [2];
};

vertex Covered cover(uint vid [[vertex_id]], constant float &z [[buffer(0)]]) {
    return Covered{float4(corners[vid], z, 1)};
}

vertex Clipped1 cover_clip1(uint vid [[vertex_id]], constant float &z [[buffer(0)]]) {
    Clipped1 out;
    out.position = float4(corners[vid], z, 1);
    out.clip[0] = out.position.x;
    return out;
}

vertex Clipped2 cover_clip2(uint vid [[vertex_id]], constant float &z [[buffer(0)]]) {
    Clipped2 out;
    out.position = float4(corners[vid], z, 1);
    out.clip[0] = out.position.x;
    out.clip[1] = -out.position.y;
    return out;
}

fragment float4 paint(constant float4 &colour [[buffer(0)]]) {
    return colour;
}

fragment float4 paint_right(float4 p [[position]], constant float4 &colour [[buffer(0)]]) {
    if (p.x < 4) {
        discard_fragment();
    }
    return colour;
}
