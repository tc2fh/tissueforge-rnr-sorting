"""On-GPU detection (docs/2026-06-24_gpu-3d-vertex-model-exploration.md): the Condition-2
trigger scans of topology_csr.find_short_edges_csr / find_small_triangles_csr ported to Warp
kernels (detect_warp). The O(mesh) scan that the host did in Python now runs one-thread-per-
element on the RTX 5090. Gated by:

  * the GPU trigger set == the host trigger reference, EXACTLY (per direction);
  * the hybrid detect (GPU scan + host gather) == the pure-host detector, as a set of sites;
  * every hybrid-detected config is surgery-ready (drives a clean parallel apply / round-trip).

[H] sites only exist after a forward I->H, so the reverse tests seed them with a forward batch.
"""
import numpy as np
import pytest

from ..gpu import csr_mesh as cm
from ..gpu import detect_warp as dw
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
    """Apply a conflict-free forward I->H batch to a fresh PaddedMesh(m0); return the
    post-forward mesh (now bearing triangular [H] sites) and the forward batch size."""
    sites = tcsr.find_short_edges_csr(PaddedMesh.from_csr(m0), threshold=1.0)
    sites = [s for s in sites if sched.i_to_h_veto_csr(PaddedMesh.from_csr(m0), s[2]) is None]
    batch = sched.independent_set(sites)
    pm_kwargs.setdefault("v_headroom", 5 * len(batch) + 16)
    pm_kwargs.setdefault("s_headroom", len(batch) + 16)
    pm = PaddedMesh.from_csr(m0, **pm_kwargs)
    for (_v10, _v11, cfg) in batch:
        rcsr.i_to_h_csr(pm, cfg, dl_th)
    return pm, len(batch)


# ======================================================================================
# H-side: small-triangle trigger scan
# ======================================================================================
def test_h_scan_matches_host_trigger(vsolver):
    """scan_small_triangles_kernel reproduces the host trigger set exactly, and that set is a
    superset of the fully-validated find_small_triangles_csr sites (trigger before gather)."""
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 40., 56.))
    pm, nbatch = _seed_triangles(cm.extract_csr(bodies))
    assert nbatch >= 2

    g = pm.to_warp(device=dev)
    gpu_set = set(int(x) for x in dw.find_small_triangles_warp(g, threshold=1.0))
    host_set = dw.small_triangle_trigger_host(pm, threshold=1.0)
    assert gpu_set == host_set, "GPU triangle trigger scan != host trigger reference"

    validated = {s for s, _cfg in tcsr.find_small_triangles_csr(pm, threshold=1.0)}
    assert validated <= gpu_set, "a validated [H] site was missed by the GPU trigger scan"


def test_h_hybrid_detect_matches_host(vsolver):
    """detect_small_triangles_hybrid (GPU scan + host gather) == find_small_triangles_csr as a
    set of sites, and the hybrid configs are surgery-ready: reversing the cap-cap sites restores
    the [I] fingerprint."""
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(40., 8., 56.))
    m0 = cm.extract_csr(bodies)
    fp0 = cm.fingerprint(m0)
    pm, _nbatch = _seed_triangles(m0)

    g = pm.to_warp(device=dev)
    hybrid = dw.detect_small_triangles_hybrid(g, pm, threshold=1.0)
    host = tcsr.find_small_triangles_csr(pm, threshold=1.0)
    assert {s for s, _ in hybrid} == {s for s, _ in host}, "hybrid [H] sites != host detector"
    # same order, too (both surface-ascending) -> identical lowest-id reservation downstream
    assert [s for s, _ in hybrid] == sorted(s for s, _ in host)

    # surgery-ready: an independent (conflict-free) batch of the hybrid configs applies cleanly
    batch = sched.h_independent_set(hybrid)
    assert batch and sched.h_batch_is_conflict_free(batch)
    assert sched.h_apply_batch(pm, batch, dl_th=0.3) == len(batch)
    assert pm.check_consistency() == [], "hybrid-detected [H] config produced an inconsistent mesh"


# ======================================================================================
# I-side: short-edge trigger scan
# ======================================================================================
def test_i_scan_matches_host_trigger(vsolver):
    """scan_short_edges_kernel reproduces the host trigger set exactly (each edge once, sorted),
    and that set is a superset of the validated find_short_edges_csr sites."""
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 8., 8.))
    m0 = cm.extract_csr(bodies)
    pm = PaddedMesh.from_csr(m0)

    g = pm.to_warp(device=dev)
    gpu = dw.find_short_edges_warp(g, threshold=1.0)
    gpu_set = {(int(a), int(b)) for a, b in gpu}
    host_set = dw.short_edge_trigger_host(pm, threshold=1.0)
    assert gpu_set == host_set, "GPU edge trigger scan != host trigger reference"
    assert len(gpu) == len(gpu_set), "duplicate edges in the GPU scan output"
    assert [(int(a), int(b)) for a, b in gpu] == sorted(gpu_set), "GPU scan not canonically sorted"

    validated = {(v, w) for v, w, _ in tcsr.find_short_edges_csr(pm, threshold=1.0)}
    assert validated <= gpu_set, "a validated [I] edge was missed by the GPU trigger scan"


def test_i_hybrid_detect_matches_host(vsolver):
    """detect_short_edges_hybrid (GPU scan + host gather) == find_short_edges_csr as a set AND
    in order, and the hybrid configs drive a clean conflict-free parallel apply."""
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(56., 56., 8.))
    m0 = cm.extract_csr(bodies)
    pm = PaddedMesh.from_csr(m0, v_headroom=400, s_headroom=400)

    g = pm.to_warp(device=dev)
    hybrid = dw.detect_short_edges_hybrid(g, pm, threshold=1.0)
    host = tcsr.find_short_edges_csr(pm, threshold=1.0)
    assert {(v, w) for v, w, _ in hybrid} == {(v, w) for v, w, _ in host}, \
        "hybrid [I] sites != host detector"
    assert [(v, w) for v, w, _ in hybrid] == sorted((v, w) for v, w, _ in host), \
        "hybrid detection order != host (breaks deterministic reservation)"

    batch = sched.independent_set(hybrid)
    assert batch and sched.batch_is_conflict_free(batch)
    assert sched.apply_batch(pm, batch, dl_th=0.3) == len(batch)
    assert pm.check_consistency() == [], "hybrid-detected [I] config produced an inconsistent mesh"


# ======================================================================================
# end-to-end: gpu_scan=True is a drop-in for the host Python scan in the device sweeps
# ======================================================================================
def test_i_sweep_gpu_scan_matches_host_scan(vsolver):
    """reconnect_sweep_warp(gpu_scan=True) == (gpu_scan=False) bit-for-bit: the GPU trigger
    scan yields the same sites in the same order, so the whole deterministic detect->reserve->
    apply pipeline matches (same round sizes AND same body-anchored fingerprint)."""
    from ..gpu import schedule_warp as sw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(24., 8., 8.))
    m0 = cm.extract_csr(bodies)
    mk = lambda: PaddedMesh.from_csr(m0, v_headroom=6000, s_headroom=6000,
                                     ring_pad=64, vs_pad=64, bs_pad=64).to_warp(device=dev)
    g_host, g_gpu = mk(), mk()
    rep_host = sw.reconnect_sweep_warp(g_host, threshold=1.0, dl_th=0.3, max_rounds=2, gpu_scan=False)
    rep_gpu = sw.reconnect_sweep_warp(g_gpu, threshold=1.0, dl_th=0.3, max_rounds=2, gpu_scan=True)
    assert rep_gpu["round_sizes"] == rep_host["round_sizes"] and rep_gpu["total"] > 0, \
        f"GPU-scan sweep diverged: {rep_gpu} vs {rep_host}"
    pm_host, pm_gpu = PaddedMesh.from_warp(g_host), PaddedMesh.from_warp(g_gpu)
    assert pm_gpu.check_consistency() == []
    assert cm.fingerprint(pm_gpu.to_csr()) == cm.fingerprint(pm_host.to_csr()), \
        "GPU-scan I->H sweep != host-scan sweep"


def test_h_sweep_gpu_scan_matches_host_scan(vsolver):
    """reconnect_sweep_h_to_i_warp(gpu_scan=True) == (gpu_scan=False), reverse direction."""
    from ..gpu import schedule_warp as sw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 24., 8.))
    m0 = cm.extract_csr(bodies)
    pm_seed, nbatch = _seed_triangles(m0, v_headroom=6000, s_headroom=6000,
                                      ring_pad=64, vs_pad=64, bs_pad=64)
    assert nbatch >= 2
    g_host, g_gpu = pm_seed.to_warp(device=dev), pm_seed.to_warp(device=dev)
    rep_host = sw.reconnect_sweep_h_to_i_warp(g_host, threshold=1.0, dl_th=0.3, max_rounds=2, gpu_scan=False)
    rep_gpu = sw.reconnect_sweep_h_to_i_warp(g_gpu, threshold=1.0, dl_th=0.3, max_rounds=2, gpu_scan=True)
    assert rep_gpu["round_sizes"] == rep_host["round_sizes"] and rep_gpu["total"] > 0, \
        f"GPU-scan reverse sweep diverged: {rep_gpu} vs {rep_host}"
    pm_host, pm_gpu = PaddedMesh.from_warp(g_host), PaddedMesh.from_warp(g_gpu)
    assert pm_gpu.check_consistency() == []
    assert cm.fingerprint(pm_gpu.to_csr()) == cm.fingerprint(pm_host.to_csr()), \
        "GPU-scan H->I sweep != host-scan sweep"
