"""Gate C, reverse direction -- brick C0' (docs/2026-06-24_gpu-3d-vertex-model-exploration.md):

The index-based [H]-config detector (topology_csr.find_small_triangles_csr /
h_neighbourhood_csr) is the H->I analogue of the forward C0 detector. It must find the
triangular reconnection sites on a MUTATED PaddedMesh with NO TF handles, and every HCfgIdx
it emits must be surgery-ready -- drive a clean h_to_i_csr that restores the pre-triangle
[I] topology. (No TF-handle oracle to compare against here, because the triangles only exist
AFTER device-side i_to_h surgery; instead the gate is the strong self-consistent one:
detect-then-reverse-everything == identity, by body-anchored fingerprint.)
"""
from .. import topology as topo
from ..gpu import csr_mesh as cm
from ..gpu import reconnect_csr as rcsr
from ..gpu import schedule_csr as sched
from ..gpu import topology_csr as tcsr
from ..gpu.device_mesh import PaddedMesh
from . import helpers as H


def test_h_detector_finds_the_triangle_from_one_i_to_h(vsolver):
    """One I->H makes one triangle; the [H] detector finds exactly it, structurally valid,
    and reversing via the DETECTED config restores the original [I] fingerprint."""
    _tf, _tfv, stype, btype = vsolver
    cfg_in = H.build_minimal_i_config(stype, btype, center=(12., 24., 48.), edge=0.5)
    m0 = cm.extract_csr(cfg_in["bodies"])
    fp0 = cm.fingerprint(m0)

    pm = PaddedMesh.from_csr(m0)
    cfg = tcsr.find_short_edges_csr(pm, threshold=1.0)[0][2]
    hcfg_made = rcsr.i_to_h_csr(pm, cfg, dl_th=0.5)
    assert pm.check_consistency() == []

    tris = tcsr.find_small_triangles_csr(pm, threshold=1.0)
    assert len(tris) == 1, f"expected exactly one [H] site after one I->H, got {len(tris)}"
    tri_s, hcfg = tris[0]
    assert tri_s == hcfg_made.triangle, "detector found a different surface than the new triangle"
    # structurally well-formed (mirror the forward detector's shape checks)
    assert len(hcfg.tri_verts) == 3 and len(hcfg.arms) == 3
    assert len(hcfg.side_cells) == 3
    assert len(hcfg.top_faces) == 3 and len(hcfg.bottom_faces) == 3
    assert hcfg.cap_top != hcfg.cap_bot
    assert set(hcfg.tri_verts) == set(hcfg_made.tri_verts)

    # reverse via the DETECTED (not the i_to_h-returned) config -> must restore [I]
    rcsr.h_to_i_csr(pm, hcfg, dl_th=0.5)
    assert pm.check_consistency() == []
    assert cm.fingerprint(pm.to_csr()) == fp0, "detected-config H->I did not restore the [I] topology"


def test_h_detector_pure_i_mesh_has_no_triangles(vsolver):
    """A freshly-built Kelvin block (truncated-octahedron faces: squares + hexagons, never
    triangles) has no [H] sites -- the detector must return empty before any I->H."""
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 56., 24.))
    pm = PaddedMesh.from_csr(cm.extract_csr(bodies))
    assert tcsr.find_small_triangles_csr(pm, threshold=1.0) == [], \
        "found a triangular [H] site in a fresh [I]-only Kelvin block"


def test_h_detector_finds_capcap_sites_and_reverses_them(vsolver):
    """Apply a conflict-free batch of N I->H on a Kelvin block, then the [H] detector must
    find AT LEAST the N canonical cap-cap triangles, and reversing those (via the DETECTED
    configs, matched by surface index) restores the original [I] fingerprint -- a
    detect-driven full round-trip at scale, no TF handles.

    NOTE (the reverse-direction cascade): an I->H can collapse a quad side-face
    [outer_top, v10, v11, outer_bot] into a triangle [outer_top, tri_k, outer_bot] -- a
    genuine (transient) [H] site the detector also reports (so len(tris) >= len(batch), not
    ==). Those side-collapse triangles SHARE the new tri vertex with their cap-cap triangle,
    so reversing ALL detected triangles would double-touch it; we reverse only the cap-cap
    sites (mutually disjoint across an independent batch). Reversing a cap-cap triangle
    re-expands its collapsed side-faces back to quads, so the whole neighbourhood restores.
    This mirrors the forward C1 cascade (an I->H seeds new short edges); production relaxes
    the mesh between steps."""
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(40., 56., 8.))
    m0 = cm.extract_csr(bodies)
    fp0 = cm.fingerprint(m0)

    sites = tcsr.find_short_edges_csr(PaddedMesh.from_csr(m0), threshold=1.0)
    sites = [s for s in sites if sched.i_to_h_veto_csr(PaddedMesh.from_csr(m0), s[2]) is None]
    batch = sched.independent_set(sites)
    assert len(batch) >= 2, f"need a multi-reconnection batch, got {len(batch)}"

    # bump allocator never reclaims: +3 verts/I->H forward, +2 verts/H->I reverse = 5/op
    hv, hs = 5 * len(batch) + 16, len(batch) + 16
    pm = PaddedMesh.from_csr(m0, v_headroom=hv, s_headroom=hs)
    # apply the batch, collecting each I->H's cap-cap triangle (the canonical sites)
    capcap = [rcsr.i_to_h_csr(pm, cfg, dl_th=0.3) for (_v10, _v11, cfg) in batch]
    assert pm.check_consistency() == []
    capcap_surfs = {h.triangle for h in capcap}
    assert len(capcap_surfs) == len(batch), "cap-cap triangles not distinct"

    tris = tcsr.find_small_triangles_csr(pm, threshold=1.0)
    detected = {s: h for s, h in tris}
    assert capcap_surfs <= set(detected), "detector missed a canonical cap-cap triangle"
    assert len(tris) >= len(batch), f"fewer triangles than reconnections: {len(tris)} < {len(batch)}"

    # reverse only the cap-cap sites, via the DETECTED configs (proves they're surgery-ready)
    for s in capcap_surfs:
        rcsr.h_to_i_csr(pm, detected[s], dl_th=0.3)
    assert pm.check_consistency() == []
    assert cm.fingerprint(pm.to_csr()) == fp0, \
        "detected-config H->I of the cap-cap sites did not restore the [I] topology"
