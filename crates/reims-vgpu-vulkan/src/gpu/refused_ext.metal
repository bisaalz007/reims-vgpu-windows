// A dynamic library exporting `ext`, which `refused_import.metal` imports (`states::tests`).
#include <metal_stdlib>
using namespace metal;

int ext(int x) {
    return x + 1;
}
