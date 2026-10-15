// A dynamic library exporting `other` and no `ext` (`states::tests`).
#include <metal_stdlib>
using namespace metal;

int other(int x) {
    return x + 2;
}
