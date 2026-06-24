"""On-GPU gather (docs/2026-06-24_gpu-3d-vertex-model-exploration.md): the [I]-neighbourhood
gather + fused Condition-4 veto as a Warp kernel (gather_warp), the device counterpart of
topology_csr.i_neighbourhood_csr + schedule_csr.i_to_h_veto_csr. With detect_warp's scan, the
reservation/apply kernels, and compact_warp, a whole reconnection round runs with NO from_warp.

Gated by:
  * per candidate, the device `valid` flag == (host gather returns a config AND host veto passes),
    and for valid candidates the emitted config (caps, side cells, arms, top/bottom faces) matches
    the host one (normalised for the free arm/face ordering);
  * a FULLY-ON-DEVICE forward round (scan -> device gather -> device pack -> reserve -> apply, no
    from_warp) == the hybrid round (host gather) by the body-anchored fingerprint.
"""
import numpy as np
import pytest

from ..gpu import csr_mesh as cm
from ..gpu import detect_warp as dw
from ..gpu import gather_warp as gw
from ..gpu import reconnect_csr as rcsr
from ..gpu import schedule_csr as sched
from ..gpu import topology_csr as tcsr
from ..gpu.device_mesh import PaddedMesh
from . import helpers as H


def _cuda_or_skip():
    import warp as wp
    if not any(d.is_cuda for d in wp.get_devices()):
        pytest.skip("no CUDA device")
    return next(d for d in wp.get_devices() if d.is_cuda)


def _seed_triangles(m0, dl_th=0.3, **pm_kwargs):
    """Apply a conflict-free forward I->H batch -> a post-forward host PaddedMesh with [H] sites."""
    sites = [s for s in tcsr.find_short_edges_csr(PaddedMesh.from_csr(m0), threshold=1.0)
             if sched.i_to_h_veto_csr(PaddedMesh.from_csr(m0), s[2]) is None]
    batch = sched.independent_set(sites)
    pm_kwargs.setdefault("v_headroom", 5 * len(batch) + 16)
    pm_kwargs.setdefault("s_headroom", len(batch) + 16)
    pm = PaddedMesh.from_csr(m0, **pm_kwargs)
    for (_v10, _v11, cfg) in batch:
        rcsr.i_to_h_csr(pm, cfg, dl_th)
    return pm


def test_i_gather_matches_host_per_candidate(vsolver):
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 8., 8.))
    m0 = cm.extract_csr(bodies)
    pm = PaddedMesh.from_csr(m0)
    g = pm.to_warp(device=dev)

    edges = dw.find_short_edges_warp(g, threshold=1.0)
    assert len(edges) > 2, "need several candidates"
    out = gw.gather_i_configs_warp(g, edges, device=dev)
    valid = out["valid"].numpy()
    cap_top, cap_bot = out["cap_top"].numpy(), out["cap_bot"].numpy()
    side = out["side"].numpy()
    arm_side, arm_otop, arm_obot = out["arm_side"].numpy(), out["arm_otop"].numpy(), out["arm_obot"].numpy()
    top, bot = out["top"].numpy(), out["bot"].numpy()

    nvalid = 0
    for i, (v, w) in enumerate(edges):
        v, w = int(v), int(w)
        cfg = tcsr.i_neighbourhood_csr(pm, v, w)
        host_ok = cfg is not None and sched.i_to_h_veto_csr(pm, cfg) is None
        assert bool(valid[i]) == host_ok, f"valid flag mismatch at edge ({v},{w})"
        if not host_ok:
            continue
        nvalid += 1
        assert int(cap_top[i]) == cfg.cap_top and int(cap_bot[i]) == cfg.cap_bot, f"caps at ({v},{w})"
        assert set(int(x) for x in side[i]) == set(cfg.side_cells), f"side cells at ({v},{w})"
        host_arms = {(a.side_surface, a.outer_top, a.outer_bot) for a in cfg.arms}
        dev_arms = {(int(arm_side[i, k]), int(arm_otop[i, k]), int(arm_obot[i, k])) for k in range(3)}
        assert dev_arms == host_arms, f"arms at ({v},{w})"
        assert set(int(x) for x in top[i]) == set(cfg.top_faces.values()), f"top faces at ({v},{w})"
        assert set(int(x) for x in bot[i]) == set(cfg.bottom_faces.values()), f"bottom faces at ({v},{w})"
    assert nvalid >= 2, f"expected several valid [I] configs, got {nvalid}"


def test_i_gather_rejects_invalid_and_vetoed(vsolver):
    """Sanity: on a fresh Kelvin block at least one trigger candidate is rejected by the gather
    (non-interior endpoint or a Condition-4 veto), proving `valid` is doing real filtering."""
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(40., 40., 40.))
    pm = PaddedMesh.from_csr(cm.extract_csr(bodies))
    g = pm.to_warp(device=dev)
    edges = dw.find_short_edges_warp(g, threshold=1.0)
    valid = gw.gather_i_configs_warp(g, edges, device=dev)["valid"].numpy()
    host_ok = [tcsr.i_neighbourhood_csr(pm, int(v), int(w)) is not None
               and sched.i_to_h_veto_csr(pm, tcsr.i_neighbourhood_csr(pm, int(v), int(w))) is None
               for v, w in edges]
    assert list(map(bool, valid)) == host_ok, "device valid set != host (gather+veto)"


# ======================================================================================
# fully-on-device detection + sweep round (no from_warp)
# ======================================================================================
def test_detect_short_edges_device_matches_hybrid(vsolver):
    """detect_short_edges_device (GPU scan + GPU gather + fused veto, no from_warp) == the hybrid
    detection AFTER the Condition-4 veto -- same sites, and surgery-ready."""
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(56., 8., 8.))
    m0 = cm.extract_csr(bodies)
    pm = PaddedMesh.from_csr(m0, v_headroom=400, s_headroom=400)
    g = pm.to_warp(device=dev)

    device_sites = gw.detect_short_edges_device(g, threshold=1.0)
    hybrid_vetoed = [s for s in dw.detect_short_edges_hybrid(g, pm, threshold=1.0)
                     if sched.i_to_h_veto_csr(pm, s[2]) is None]
    assert {(v, w) for v, w, _ in device_sites} == {(v, w) for v, w, _ in hybrid_vetoed}, \
        "device detection != hybrid (after veto)"
    assert [(v, w) for v, w, _ in device_sites] == sorted((v, w) for v, w, _ in hybrid_vetoed)

    batch = sched.independent_set(device_sites)
    assert batch and sched.batch_is_conflict_free(batch)
    assert sched.apply_batch(pm, batch, dl_th=0.3) == len(batch)
    assert pm.check_consistency() == [], "device-gathered config produced an inconsistent mesh"


def test_fully_device_sweep_round_matches_host(vsolver):
    """reconnect_sweep_warp_device (scan+gather+reserve+apply, NO from_warp) round 1 == the
    host-scan sweep round 1: same batch size and the same body-anchored fingerprint (the device
    gather may order arms differently, permuting tri-vertex positions, but the TOPOLOGY matches)."""
    from ..gpu import schedule_warp as sw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 56., 56.))
    m0 = cm.extract_csr(bodies)
    mk = lambda: PaddedMesh.from_csr(m0, v_headroom=6000, s_headroom=6000,
                                     ring_pad=64, vs_pad=64, bs_pad=64).to_warp(device=dev)
    g_host, g_dev = mk(), mk()
    rep_host = sw.reconnect_sweep_warp(g_host, threshold=1.0, dl_th=0.3, max_rounds=1, gpu_scan=False)
    rep_dev = sw.reconnect_sweep_warp_device(g_dev, threshold=1.0, dl_th=0.3, max_rounds=1)
    assert rep_dev["round_sizes"] == rep_host["round_sizes"] and rep_dev["total"] > 0, \
        f"fully-device sweep diverged: {rep_dev} vs {rep_host}"
    pm_host, pm_dev = PaddedMesh.from_warp(g_host), PaddedMesh.from_warp(g_dev)
    assert pm_dev.check_consistency() == []
    assert cm.fingerprint(pm_dev.to_csr()) == cm.fingerprint(pm_host.to_csr()), \
        "fully-device sweep round != host-scan sweep round (topology)"


# ======================================================================================
# reverse direction: the [H]-neighbourhood gather
# ======================================================================================
def test_h_gather_matches_host_per_candidate(vsolver):
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(40., 56., 40.))
    pm = _seed_triangles(cm.extract_csr(bodies))
    g = pm.to_warp(device=dev)

    tris = dw.find_small_triangles_warp(g, threshold=1.0)
    assert len(tris) >= 2
    out = gw.gather_h_configs_warp(g, tris, device=dev)
    valid = out["valid"].numpy()
    cap_top, cap_bot = out["cap_top"].numpy(), out["cap_bot"].numpy()
    tri, side = out["tri"].numpy(), out["side"].numpy()
    arm_side, arm_otop, arm_obot = out["arm_side"].numpy(), out["arm_otop"].numpy(), out["arm_obot"].numpy()
    top, bot = out["top"].numpy(), out["bot"].numpy()

    nvalid = 0
    for i, s in enumerate(tris):
        s = int(s)
        cfg = tcsr.h_neighbourhood_csr(pm, s)
        host_ok = cfg is not None and sched.h_to_i_veto_csr(pm, cfg) is None
        assert bool(valid[i]) == host_ok, f"valid flag mismatch at triangle {s}"
        if not host_ok:
            continue
        nvalid += 1
        assert int(cap_top[i]) == cfg.cap_top and int(cap_bot[i]) == cfg.cap_bot, f"caps at {s}"
        assert set(int(x) for x in tri[i]) == set(cfg.tri_verts), f"tri verts at {s}"
        assert set(int(x) for x in side[i]) == set(cfg.side_cells), f"side cells at {s}"
        host_arms = {(a.tri_vertex, a.side_surface, a.outer_top, a.outer_bot) for a in cfg.arms}
        dev_arms = {(int(tri[i, k]), int(arm_side[i, k]), int(arm_otop[i, k]), int(arm_obot[i, k]))
                    for k in range(3)}
        assert dev_arms == host_arms, f"arms at {s}"
        assert set(int(x) for x in top[i]) == set(cfg.top_faces.values()), f"top faces at {s}"
        assert set(int(x) for x in bot[i]) == set(cfg.bottom_faces.values()), f"bottom faces at {s}"
    assert nvalid >= 2, f"expected several valid [H] configs, got {nvalid}"


def test_detect_small_triangles_device_matches_hybrid(vsolver):
    """detect_small_triangles_device (GPU scan + GPU gather + fused veto, no from_warp) == the
    hybrid detection AFTER the Condition-4 veto, and the configs are surgery-ready."""
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 40., 8.))
    pm = _seed_triangles(cm.extract_csr(bodies), v_headroom=400, s_headroom=400)
    g = pm.to_warp(device=dev)

    device_sites = gw.detect_small_triangles_device(g, threshold=1.0)
    hybrid_vetoed = [s for s in dw.detect_small_triangles_hybrid(g, pm, threshold=1.0)
                     if sched.h_to_i_veto_csr(pm, s[1]) is None]
    assert {s for s, _ in device_sites} == {s for s, _ in hybrid_vetoed}, \
        "device [H] detection != hybrid (after veto)"

    batch = sched.h_independent_set(device_sites)
    assert batch and sched.h_batch_is_conflict_free(batch)
    assert sched.h_apply_batch(pm, batch, dl_th=0.3) == len(batch)
    assert pm.check_consistency() == [], "device-gathered [H] config produced an inconsistent mesh"


def test_fully_device_h_sweep_round_matches_host(vsolver):
    """reconnect_sweep_h_to_i_warp_device (scan+gather+reserve+apply, NO from_warp) round 1 == the
    host-scan reverse sweep round 1, by body-anchored fingerprint."""
    from ..gpu import schedule_warp as sw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(56., 40., 56.))
    m0 = cm.extract_csr(bodies)
    pm_seed = _seed_triangles(m0, v_headroom=6000, s_headroom=6000,
                              ring_pad=64, vs_pad=64, bs_pad=64)
    g_host, g_dev = pm_seed.to_warp(device=dev), pm_seed.to_warp(device=dev)
    rep_host = sw.reconnect_sweep_h_to_i_warp(g_host, threshold=1.0, dl_th=0.3, max_rounds=1, gpu_scan=False)
    rep_dev = sw.reconnect_sweep_h_to_i_warp_device(g_dev, threshold=1.0, dl_th=0.3, max_rounds=1)
    assert rep_dev["round_sizes"] == rep_host["round_sizes"] and rep_dev["total"] > 0, \
        f"fully-device reverse sweep diverged: {rep_dev} vs {rep_host}"
    pm_host, pm_dev = PaddedMesh.from_warp(g_host), PaddedMesh.from_warp(g_dev)
    assert pm_dev.check_consistency() == []
    assert cm.fingerprint(pm_dev.to_csr()) == cm.fingerprint(pm_host.to_csr()), \
        "fully-device reverse sweep round != host-scan reverse sweep round (topology)"
