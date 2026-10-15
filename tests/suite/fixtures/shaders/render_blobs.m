// Serializes the render suite's pipeline descriptors with the host's Metal
// (-[MTLDevice serializeRenderPipelineDescriptor:] and the tile variant, the
// method docs/testing.md section 6.18 describes), and checks that each one
// deserializes and builds. Each function is an NSProxy whose -functionRef is
// a sentinel (vertex 0xa1a1a1a1, fragment 0xb2b2b2b2, tile 0xc3c3c3c3) that
// tests/suite/pgtest/render_lib.py replaces with the function's object-list
// reference. It prints one "name hex" line per blob; the output is pasted
// into render_lib.py's BLOBS. Not part of any build:
//   clang -fobjc-arc -framework Metal -framework Foundation render_blobs.m -o /tmp/render_blobs
//   (cd tests/suite/fixtures/shaders && /tmp/render_blobs)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

@interface RefFn : NSProxy
@property(nonatomic, strong) id target;
@property(nonatomic) uint32_t ref;
@end
@implementation RefFn
- (uint32_t)functionRef { return self.ref; }
- (NSMethodSignature *)methodSignatureForSelector:(SEL)s { return [self.target methodSignatureForSelector:s]; }
- (void)forwardInvocation:(NSInvocation *)inv { [inv invokeWithTarget:self.target]; }
- (BOOL)respondsToSelector:(SEL)s { return s == @selector(functionRef) || [self.target respondsToSelector:s]; }
@end
@interface Ctx : NSObject
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, id<MTLFunction>> *fns;
@end
@implementation Ctx
- (id<MTLFunction>)functionWithFunctionRef:(uint64_t)ref { return self.fns[@(ref)]; }
@end

static id<MTLDevice> dev;
static Ctx *ctx;
static id fn(NSString *name, uint32_t ref) {
    NSData *b = [NSData dataWithContentsOfFile:[name stringByAppendingString:@".metallib"]];
    dispatch_data_t d = dispatch_data_create(b.bytes, b.length, nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError *e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithData:d error:&e];
    if (!lib) { fprintf(stderr, "%s: %s\n", name.UTF8String, e.description.UTF8String); exit(1); }
    id<MTLFunction> f = [lib newFunctionWithName:lib.functionNames[0]];
    ctx.fns[@(ref)] = f;
    RefFn *p = [RefFn alloc];
    p.target = f;
    p.ref = ref;
    return p;
}
static void out(const char *name, dispatch_data_t data) {
    const void *b;
    size_t n;
    dispatch_data_t m = dispatch_data_create_map(data, &b, &n);
    (void)m;
    printf("%s ", name);
    for (size_t i = 0; i < n; i++) printf("%02x", ((const uint8_t *)b)[i]);
    printf("\n");
}
static void render(const char *name, MTLRenderPipelineDescriptor *d) {
    dispatch_data_t data = [(id)dev performSelector:@selector(serializeRenderPipelineDescriptor:) withObject:d];
    MTLRenderPipelineDescriptor *back = [(id)dev performSelector:@selector(newRenderPipelineDescriptorWithSerializedData:deserializationContext:)
                                                      withObject:data withObject:ctx];
    NSError *e = nil;
    id ps = [dev newRenderPipelineStateWithDescriptor:back error:&e];
    if (!ps) { fprintf(stderr, "%s: %s\n", name, e.description.UTF8String); exit(1); }
    out(name, data);
}
static void tile(const char *name, MTLTileRenderPipelineDescriptor *d) {
    dispatch_data_t data = [(id)dev performSelector:@selector(serializeTileRenderPipelineDescriptor:) withObject:d];
    MTLTileRenderPipelineDescriptor *back = [(id)dev performSelector:@selector(newTileRenderPipelineDescriptorWithSerializedData:deserializationContext:)
                                                          withObject:data withObject:ctx];
    NSError *e = nil;
    id ps = [dev newRenderPipelineStateWithTileDescriptor:back options:0 reflection:nil error:&e];
    if (!ps) { fprintf(stderr, "%s: %s\n", name, e.description.UTF8String); exit(1); }
    out(name, data);
}

int main() {
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        ctx = [Ctx new];
        ctx.fns = [NSMutableDictionary dictionary];
        const uint32_t V = 0xa1a1a1a1, F = 0xb2b2b2b2, T = 0xc3c3c3c3;
        MTLRenderPipelineDescriptor *d;
#define NEW(v, f) d = [MTLRenderPipelineDescriptor new]; d.vertexFunction = fn(@v, V); \
        if (f) d.fragmentFunction = fn(@"" f, F); d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        NEW("vert", "frag"); render("basic", d);
        NEW("vert", "render_fbuf"); render("fbuf", d);
        d.colorAttachments[0].blendingEnabled = YES;
        d.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorBlendColor;
        d.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
        d.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorBlendAlpha;
        d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorZero;
        render("fbuf_blend", d);
        NEW("render_vbuf", "render_color"); render("vbuf", d);
        NEW("vert", "render_ftex"); render("ftex", d);
        NEW("render_vtex", "render_color"); render("vtex", d);
        NEW("render_vdesc", "render_color");
        MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
        vd.attributes[0].format = MTLVertexFormatFloat2;
        vd.attributes[0].offset = 0;
        vd.attributes[0].bufferIndex = 0;
        vd.attributes[1].format = MTLVertexFormatUChar4Normalized;
        vd.attributes[1].offset = 8;
        vd.attributes[1].bufferIndex = 0;
        vd.layouts[0].stride = 12;
        d.vertexDescriptor = vd;
        render("vdesc", d);
        vd.layouts[0].stride = MTLBufferLayoutStrideDynamic;
        render("vdesc_dyn", d);
        NEW("render_depth", "render_color");
        d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
        d.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
        render("depth", d);
        NEW("vert", "frag"); d.rasterSampleCount = 4; render("msaa4", d);
        NEW("vert", "frag"); d.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA8Unorm; render("two", d);
        NEW("vert", "frag"); d.maxVertexAmplificationCount = 2; render("amp2", d);
        NEW("render_tess", "render_color");
        d.tessellationPartitionMode = MTLTessellationPartitionModeInteger;
        d.maxTessellationFactor = 16;
        render("tess", d);
        d.tessellationControlPointIndexType = MTLTessellationControlPointIndexTypeUInt16;
        render("tess_idx", d);
        NEW("render_vbuf", "render_color"); d.rasterSampleCount = 4; render("vbuf_msaa4", d);
        d = [MTLRenderPipelineDescriptor new]; d.vertexFunction = fn(@"vert", V); render("nocolor", d);
        d.rasterSampleCount = 4; render("nocolor4", d);
        d = [MTLRenderPipelineDescriptor new]; d.vertexFunction = fn(@"render_vbuf", V); render("vbuf_nocolor", d);
        d.rasterSampleCount = 4; render("vbuf_nocolor4", d);
        MTLTileRenderPipelineDescriptor *t = [MTLTileRenderPipelineDescriptor new];
        t.tileFunction = fn(@"tile", T);
        t.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        t.threadgroupSizeMatchesTileSize = YES;
        tile("tile", t);
        t = [MTLTileRenderPipelineDescriptor new];
        t.tileFunction = fn(@"render_tilebuf", T);
        t.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        t.threadgroupSizeMatchesTileSize = YES;
        tile("tilebuf", t);
    }
}
