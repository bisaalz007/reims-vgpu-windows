// A fragment function that samples texture 0 through sampler 0 and writes colour 1 alone: the
// tests' state that supports indirect command buffers whose sample reaches a colour no
// attachment may write.
#include <metal_stdlib>
using namespace metal;

struct Color1 {
    float4 colour [[color(1)]];
};

fragment Color1 ftex_color1(float4 p [[position]], texture2d<float> t [[texture(0)]],
                            sampler s [[sampler(0)]]) {
    return {t.sample(s, p.xy / 16.0)};
}
