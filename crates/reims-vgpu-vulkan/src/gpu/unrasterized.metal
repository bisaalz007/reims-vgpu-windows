// A vertex function that returns nothing, the one a render state with rasterization off takes
// (Metal refuses one that returns a value there): the tests' states with rasterization off.
#include <metal_stdlib>
using namespace metal;

vertex void void_vertex(uint vid [[vertex_id]]) {
}
