// The tests' kernel that keeps the GPU busy (`tests/new_images.rs`): one thread runs c[0] rounds
// of 4096 steps of busy work, about 60 µs a round on the M3 Max, then writes the work's result to
// c[2] and 0x600d to c[1], so the CPU can tell whether it has ended.
#include <metal_stdlib>
using namespace metal;

kernel void spin(device uint *c [[buffer(0)]]) {
    uint rounds = c[0];
    uint a = 7;
    for (uint n = 0; n < rounds; n++) {
        for (uint j = 0; j < 4096; j++) {
            a = a * 1664525u + 1013904223u;
        }
    }
    c[2] = a;
    c[1] = 0x600du;
}
