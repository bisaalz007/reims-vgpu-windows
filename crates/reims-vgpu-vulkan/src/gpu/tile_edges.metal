// The tests' tile functions over a colour attachment's implicit imageblock, colour 0 read as four
// floats (`encode_render`'s `tile_dispatch_cells_past_the_attachment_are_metals`): each thread
// is a cell. `tile_edges_read` writes its cell into its element of buffer 0, and
// `tile_edges_write` writes its element of buffer 0 into its cell. A tile is 16 by 16 pixels and
// its threadgroup 16 by 16 threads, over an attachment one tile across, so cell (x, y) of the
// pass's tiles is element 16 y + x.
#include <metal_stdlib>
using namespace metal;

struct Cell {
    float4 color [[color(0)]];
};

static uint element(uint2 tile, ushort2 cell) {
    return (tile.y * 16u + uint(cell.y)) * 16u + uint(cell.x);
}

kernel void tile_edges_read(imageblock<Cell, imageblock_layout_implicit> block,
                            device float4 *out [[buffer(0)]],
                            ushort2 cell [[thread_position_in_threadgroup]],
                            uint2 tile [[threadgroup_position_in_grid]]) {
    out[element(tile, cell)] = block.read(cell).color;
}

kernel void tile_edges_write(imageblock<Cell, imageblock_layout_implicit> block,
                             device const float4 *in [[buffer(0)]],
                             ushort2 cell [[thread_position_in_threadgroup]],
                             uint2 tile [[threadgroup_position_in_grid]]) {
    Cell c;
    c.color = in[element(tile, cell)];
    block.write(c, cell);
}
