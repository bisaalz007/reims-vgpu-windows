"""Pixel formats as the guest meets them: a normal texture (type 2) of each
public MTLPixelFormat, whose dimension block states the format's block
size and bytes as Metal documents them, each listed in its own exec
(docs/testing.md, section 6.19).

The texture dimension check compares the block against the device's
pixel-format table, which is Metal's own but for the depth-stencil and XR10
two-plane formats; Metal then makes the texture, or not, and the prepare
pages it in, a PVRTC texture with MTLBlitOptionRowLinearPVRTC. Whether each
exec runs is the host's Metal: the scenario is host-keyed.
Depth24Unorm_Stencil8 (255) and X24_Stencil8 (262), which this GPU lacks, are
left out: Metal aborts making them."""
from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.resources_build import desc1, dimension, mip, texture_rec
from scenarios.resources import creation_case

# (pixel format, block width, block height, block bytes, compressed)
FORMATS = [
    (1, 1, 1, 1, 0), (10, 1, 1, 1, 0), (11, 1, 1, 1, 0), (12, 1, 1, 1, 0), (13, 1, 1, 1, 0), (14, 1, 1, 1, 0),
    (20, 1, 1, 2, 0), (22, 1, 1, 2, 0), (23, 1, 1, 2, 0), (24, 1, 1, 2, 0), (25, 1, 1, 2, 0),
    (30, 1, 1, 2, 0), (31, 1, 1, 2, 0), (32, 1, 1, 2, 0), (33, 1, 1, 2, 0), (34, 1, 1, 2, 0),
    (40, 1, 1, 2, 0), (41, 1, 1, 2, 0), (42, 1, 1, 2, 0), (43, 1, 1, 2, 0),
    (53, 1, 1, 4, 0), (54, 1, 1, 4, 0), (55, 1, 1, 4, 0), (60, 1, 1, 4, 0), (62, 1, 1, 4, 0), (63, 1, 1, 4, 0),
    (64, 1, 1, 4, 0), (65, 1, 1, 4, 0), (70, 1, 1, 4, 0), (71, 1, 1, 4, 0), (72, 1, 1, 4, 0), (73, 1, 1, 4, 0),
    (74, 1, 1, 4, 0), (80, 1, 1, 4, 0), (81, 1, 1, 4, 0), (90, 1, 1, 4, 0), (91, 1, 1, 4, 0), (92, 1, 1, 4, 0),
    (93, 1, 1, 4, 0), (94, 1, 1, 4, 0), (554, 1, 1, 4, 0), (555, 1, 1, 4, 0),
    (103, 1, 1, 8, 0), (104, 1, 1, 8, 0), (105, 1, 1, 8, 0), (110, 1, 1, 8, 0), (112, 1, 1, 8, 0),
    (113, 1, 1, 8, 0), (114, 1, 1, 8, 0), (115, 1, 1, 8, 0), (552, 1, 1, 8, 0), (553, 1, 1, 8, 0),
    (123, 1, 1, 16, 0), (124, 1, 1, 16, 0), (125, 1, 1, 16, 0),
    (130, 4, 4, 8, 1), (131, 4, 4, 8, 1), (132, 4, 4, 16, 1), (133, 4, 4, 16, 1), (134, 4, 4, 16, 1),
    (135, 4, 4, 16, 1), (140, 4, 4, 8, 1), (141, 4, 4, 8, 1), (142, 4, 4, 16, 1), (143, 4, 4, 16, 1),
    (150, 4, 4, 16, 1), (151, 4, 4, 16, 1), (152, 4, 4, 16, 1), (153, 4, 4, 16, 1),
    (160, 8, 4, 8, 1), (161, 8, 4, 8, 1), (162, 4, 4, 8, 1), (163, 4, 4, 8, 1),
    (164, 8, 4, 8, 1), (165, 8, 4, 8, 1), (166, 4, 4, 8, 1), (167, 4, 4, 8, 1),
    (170, 4, 4, 8, 1), (172, 4, 4, 8, 1), (174, 4, 4, 16, 1), (176, 4, 4, 16, 1), (178, 4, 4, 16, 1),
    (179, 4, 4, 16, 1), (180, 4, 4, 8, 1), (181, 4, 4, 8, 1), (182, 4, 4, 8, 1), (183, 4, 4, 8, 1),
    (186, 4, 4, 16, 1), (204, 4, 4, 16, 1), (222, 4, 4, 16, 1),
    (240, 2, 1, 4, 0), (241, 2, 1, 4, 0),
    (250, 1, 1, 2, 0), (252, 1, 1, 4, 0), (253, 1, 1, 1, 0),
]

# Formats whose exec faults on this host: validateTextureDimension refuses
# the block, or Metal makes no texture. Measured on the reference host.
FAULTS = {240, 241}


def _record(fmt, bw, bh, bb, comp):
    w, h = 16, 16
    bpr = (w // bw) * bb
    size = bpr * (h // bh)
    dim = dimension([mip(0, size, bpr, w, h)], block=(bw, bh, 1, bb), compressed=comp)
    return texture_rec(PAGE, 8, [dim], desc1(fmt=fmt, w=w, h=h, usage=1))


@scenario(host=True)
def textures_formats():
    """A 16x16 texture of each public pixel format (FORMATS), one exec each;
    the probe buffer's slot for a format is written only if its exec ran.
    FAULTS lists the formats whose exec faults here."""
    cases = [(abi.T_TEXTURE, _record(*f), None, f[0] in FAULTS) for f in FORMATS]
    return creation_case("textures_formats", textures_formats.__doc__, cases, host=True, probe_len=4)
