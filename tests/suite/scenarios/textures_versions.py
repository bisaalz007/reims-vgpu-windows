"""The binary version's feature gates on resources and textures, each tested
below and at the version that turns it on (docs/testing.md, section 5.6).

- dualPlaneTextures: from 31, but not 40 (types 12 and 18);
- s8ByteCountForD32S8: 4 below 62, 1 from it (the stencil plane's texel);
- supportsSharedTextures: off from 7 to 99 (type 16);
- supportsMultiplaneMippedBlitTextures: from 113 (mipped types 12 and 18);
- supportsHeapHazardTracking: from 204 (tracked heaps);
- supportsArgumentBuffers: from 207 (read-only buffers and heaps in the
  service)."""
from pgtest import abi
from pgtest.abi import PAGE
from pgtest.registry import scenario
from pgtest.resources_build import (Depth32Float_Stencil8, buffer_rec, desc1, heap_rec, shared_backing_rec,
                                    shared_texture_rec, simple_dim, texture_rec)
from pgtest.scenario import Scenario
from scenarios.resources import creation_case
from scenarios.textures_dual_plane import ds_dims

T12 = abi.T_DUAL_PLANE_TEXTURE
DS = desc1(fmt=Depth32Float_Stencil8, usage=3)
DS2 = desc1(fmt=Depth32Float_Stencil8, usage=3, mips=2)


def _records(version):
    s8 = 1 if version >= 62 else 4
    dual = 31 <= version and version != 40
    mipped = version >= 113
    tracked = version >= 204
    ro_service = version >= 207
    return [
        (T12, texture_rec(PAGE, 8, ds_dims(s8=s8), DS), None, not dual),
        (T12, texture_rec(PAGE, 9, ds_dims(s8=s8, levels=2), DS2), None, not (dual and mipped)),
        (abi.T_TEXTURE, texture_rec(PAGE, 10, [simple_dim(4, 4, bpp=8)], DS), None, None),
        (abi.T_HEAP, heap_rec(12, 4 * PAGE, 3), None, not tracked),
        (abi.T_BUFFER, buffer_rec(PAGE, 11, 2), None, {"inproc": False, "service": not ro_service}),
        (abi.T_HEAP, heap_rec(16, 4 * PAGE, 5), None, {"inproc": False, "service": not ro_service}),
    ]


def _version_case(version, doc):
    recs = _records(version)
    # The one-plane D32S8 at versions without two-plane textures is locked
    # separately: whether it is made is Metal's pixel-format info.
    return creation_case("textures_version_%d" % version, doc, recs, version=version)


@scenario
def textures_versions():
    """The gates at versions 30, 40, 60, 62, 112, 113, 203, 204, 206 and 207:
    a two-plane Depth32Float_Stencil8 texture (type 12, one and two levels,
    with the stencil texel size the version's), a tracked private heap, and a
    read-only buffer and heap (both refused in the service below 207, the
    buffer with "Cannot use unsupported readonly buffers"). Each object is
    listed in its own exec (scenarios.resources.creation_case)."""
    out = []
    for v in (30, 40, 60, 62, 112, 113, 203, 204, 206, 207):
        out += _version_case(v, textures_versions.__doc__)
    return out


@scenario
def textures_versions_shared():
    """Shared textures below version 100 (from 7): a type 16 texture is
    refused ("Creation of reference shared texture unsupported on device"),
    where at 100 it is made; a type 18 backing at version 40 (no two-plane
    textures) and a two-level one at 112 (no mipped two-plane blits: "Multi
    plane mipped blit shared texture backing is unsupported")."""
    from scenarios.textures_shared import kernel_and_user, two_task_probe
    out = []
    for v in (40, 62, 100, 112):
        s = Scenario("textures_versions_shared_%d" % v, doc=textures_versions_shared.__doc__)
        s.bring_up(version=v)
        s.pg_version = v
        k, t = kernel_and_user(s)
        s8 = 1 if v >= 62 else 4
        k17 = k.add(abi.T_SHARED_BACKING1, shared_backing_rec(PAGE, 8, [simple_dim(4, 4)], desc1(usage=3)))
        k18 = k.add(abi.T_SHARED_BACKING2, shared_backing_rec(PAGE, 9, ds_dims(s8=s8), DS))
        k18m = k.add(abi.T_SHARED_BACKING2, shared_backing_rec(PAGE, 10, ds_dims(s8=s8, levels=2), DS2))
        shared = not (7 <= v < 100)
        dual = v != 40
        cases = [(t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(k17)), not shared),
                 (t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(k18)), not (shared and dual)),
                 (t.add(abi.T_SHARED_TEXTURE, shared_texture_rec(k18m)), not (shared and dual and v >= 113))]
        out.append(two_task_probe(s, k, t, cases))
    return out


@scenario(host=True)
def textures_version_single_plane_depth_stencil():
    """At version 40 (no two-plane textures) Depth32Float_Stencil8 and
    X32_Stencil8 are described by Metal's pixel-format info with bit 23 of
    its flags cleared, so a type 2 texture of either is one plane whose texel
    size is Metal's: records with 8, 4 and 5 bytes a texel, each listed in
    its own exec (opcode 55). Depth32Float_Stencil8 passes with 8 bytes a
    texel; every X32_Stencil8 record faults. Host-keyed: the outcome is
    Metal's pixel-format table."""
    from pgtest.resources_build import X32_Stencil8
    cases = []
    for fmt in (Depth32Float_Stencil8, X32_Stencil8):
        d = desc1(fmt=fmt, usage=3)
        for bpp, faults in SINGLE_PLANE_DS[fmt]:
            cases.append((abi.T_TEXTURE, texture_rec(PAGE, 8, [simple_dim(4, 4, bpp=bpp)], d), None, faults))
    return creation_case("textures_version_single_plane_depth_stencil",
                         textures_version_single_plane_depth_stencil.__doc__, cases, version=40, host=True)


# (bytes a texel, whether the exec faults), as this host's Metal describes
# each format.
SINGLE_PLANE_DS = {
    Depth32Float_Stencil8: [(8, False), (4, True), (5, True)],
    261: [(8, True), (4, True), (5, True)],
}
