"""Tile shading in the render decoder: tile pipelines (type 7, commandType
51), 0x9b and 0xa2/0xa3 tile dispatches, tile buffers, samplers and textures
(0x9d-0xa1), 0x9c threadgroup memory, and 0xa4 GetTileDimensions, whose reply
only the service can write (docs/testing.md, sections 6.16, 6.17 and 7.4)."""
import struct

from pgtest import abi
from pgtest import render_lib as R
from pgtest.registry import scenario
from pgtest.scenario import Scenario
from pgtest.task import rec

W = H = 8
NPIX = W * H * 4


def half4(*v):
    return struct.pack("<4e", *v)


class TileTask:
    def __init__(self, s):
        t = self.t = R.RenderTask(s)
        tile = t.function("shaders/tile.metallib")
        tilebuf = t.function("shaders/render_tilebuf.metallib")
        self.p_tile = t.pipeline("tile", t=tile)
        self.p_tilebuf = t.pipeline("tile", t=tilebuf)
        self.tex, _, self.va = t.target(W, H)
        self.halfs, _ = t.buffer(contents=half4(1.0, 0.0, 1.0, 1.0) + half4(0.0, 0.0, 1.0, 1.0))
        self.reply, self.reply_va = t.buffer()
        self.smp = t.sampler()

    def pass_(self, tile=(16, 16)):
        pre = [R.tile_size(*tile)] if tile else []
        return pre + [R.describe_pass({0: dict(tex=self.tex, clear=(1.0, 0.0, 0.0, 1.0))}, width=W, height=H)]

    def start(self):
        self.t.define()
        self.t.install()


def case(s, tt, label, records, fault=False, dump=True, tile=(16, 16)):
    t = tt.t
    if dump and not fault:
        t.fill(tt.va, NPIX, 0x11)
    t.render(tt.pass_(tile) + records, fault=fault, label=label)
    if fault:
        s.fault_state(label)
        s.resume_root()
    elif dump:
        t.dump(tt.va, NPIX, label)


@scenario
def render_tile_dispatch():
    """Tile dispatches over a red-cleared 8x8 target with 16x16 tiles
    (0x24): 0x9b with tile.metal's pipeline writes green; 0x9d SetTileBuffers
    and 0x9e SetTileBufferOffset feed render_tilebuf.metal's colour (magenta,
    then blue at offset 8); 0xa2 and 0xa3 dispatch in a one-tile region;
    0x9c, 0x9f, 0xa0 and 0xa1 set state. A dispatch before a pipeline
    throws "Attempting to dispatch without pipeline set"."""
    s = Scenario("render_tile_dispatch", doc=render_tile_dispatch.__doc__)
    s.bring_up()
    tt = TileTask(s)
    tt.start()
    p = R.pipeline_state
    case(s, tt, "0x9b tile.metal", [p(tt.p_tile), R.dispatch_tile(16, 16)])
    case(s, tt, "0x9d tile buffer: magenta", [p(tt.p_tilebuf), R.buffers(0x9D, 0, [(tt.halfs, 0)]),
                                             R.dispatch_tile(16, 16)])
    case(s, tt, "0x9e offset 8: blue", [p(tt.p_tilebuf), R.buffers(0x9D, 0, [(tt.halfs, 0)]),
                                       R.buffer_offset(0x9E, 0, 8), R.dispatch_tile(16, 16)])
    case(s, tt, "0xa2 region (0, 0) 1x1", [p(tt.p_tile), R.dispatch_tile_region(0xA2, (16, 16, 1), (0, 0, 0),
                                                                                (1, 1, 1), 5)])
    case(s, tt, "0xa3 region (0, 0) 1x1, layer 0", [p(tt.p_tile), R.dispatch_tile_region(0xA3, (16, 16, 1), (0, 0, 0),
                                                                                        (1, 1, 1), 0)])
    case(s, tt, "0x9c, 0x9f, 0xa0, 0xa1", [p(tt.p_tile), R.tg_memory(0, 0, 0), R.refs_range(0x9F, 0, [tt.smp]),
                                           R.samplers_lod(0xA0, 1, [(tt.smp, 0.0, 1.0)]),
                                           R.refs_range(0xA1, 0, [tt.tex]), R.dispatch_tile(16, 16)])
    case(s, tt, "0x9b no pipeline", [R.dispatch_tile(16, 16)], fault=True)
    case(s, tt, "0xa2 no pipeline", [R.dispatch_tile_region(0xA2, (16, 16, 1), (0, 0, 0), (1, 1, 1))], fault=True)
    case(s, tt, "0xa3 no pipeline", [R.dispatch_tile_region(0xA3, (16, 16, 1), (0, 0, 0), (1, 1, 1))], fault=True)
    case(s, tt, "0x9b short", [rec(0x9B, bytes(8))], fault=True)
    tt.t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def render_tile_dimensions():
    """0xa4 GetTileDimensions in the service: it writes {u16 tileWidth, u16
    tileHeight} of the encoder at the reply buffer's offset; 16x16 and 32x32
    as 0x24 set them, then the host's default. A reply buffer ref 0 has no
    cursor ("Failed to get cursor for buffer"); an offset at the buffer's end
    fails to write. Isolate only: in-process the reply aborts the host
    (render_tile_dimensions_host). host: the default tile size is the
    GPU's."""
    s = Scenario("render_tile_dimensions", setups=("isolate",), host=True, doc=render_tile_dimensions.__doc__)
    s.bring_up()
    tt = TileTask(s)
    tt.start()
    tt.t.fill(tt.reply_va, 16, 0xEE)
    case(s, tt, "16x16 @0", [R.tile_dims(tt.reply, 0)], dump=False)
    case(s, tt, "32x32 @6", [R.tile_dims(tt.reply, 6)], dump=False, tile=(32, 32))
    case(s, tt, "default @12", [R.tile_dims(tt.reply, 12)], dump=False, tile=None)
    tt.t.dump(tt.reply_va, 16, "replies")
    case(s, tt, "reply buffer 0", [R.tile_dims(0, 0)], fault=True)
    case(s, tt, "offset at the end", [R.tile_dims(tt.reply, abi.PAGE)], fault=True)
    case(s, tt, "offset 2 short of the end", [R.tile_dims(tt.reply, abi.PAGE - 2)], fault=True)
    case(s, tt, "reply buffer a texture", [R.tile_dims(tt.tex, 0)], fault=True)
    tt.t.delete()
    s.events("end")
    return s


@scenario(setups=("blocks", "map"))
def render_tile_dimensions_host():
    """0xa4 GetTileDimensions in an in-process task: an in-process task has
    no cursor for the reply, and asking for one aborts (section 7.4), so the
    host dies at the exec. Not isolate: there the service writes it
    (render_tile_dimensions)."""
    s = Scenario("render_tile_dimensions_host", setups=("blocks", "map"), doc=render_tile_dimensions_host.__doc__)
    s.bring_up()
    tt = TileTask(s)
    tt.start()
    case(s, tt, "0xa4", [R.tile_dims(tt.reply, 0)], fault=True)
    return s
