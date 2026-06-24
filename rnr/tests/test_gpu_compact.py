"""Gate D: stream-compaction of dead vertex/surface slots (docs/2026-06-24_gpu-3d-vertex-
model-exploration.md). The bump allocator never reclaims (+3 verts/+1 surf per I->H, +2
verts/-1 surf per H->I), so the high-water counters n_v_used/n_s_used only grow. Compaction
renumbers the LIVE elements into a contiguous prefix and resets the counters, so arrays stay
bounded over many reconnection passes. Gated by:

  * compaction preserves the body-anchored fingerprint (only slot labels change);
  * the high-water counters drop to the live counts; a second compact is a no-op;
  * over many forward-then-reverse + compact passes the counters stay bounded (return to the
    original mesh size), where WITHOUT compaction they would grow ~5*batch each pass.

The device compaction (compact_warp) must match the host PaddedMesh.compact() by fingerprint.
"""
import numpy as np
import pytest

from ..gpu import csr_mesh as cm
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


def _forward_batch(pm):
    """A conflict-free forward I->H batch on pm; returns the cap-cap HCfgIdx list (to reverse)."""
    sites = [s for s in tcsr.find_short_edges_csr(pm, threshold=1.0)
             if sched.i_to_h_veto_csr(pm, s[2]) is None]
    batch = sched.independent_set(sites)
    return [rcsr.i_to_h_csr(pm, cfg, dl_th=0.3) for (_v10, _v11, cfg) in batch]


# ======================================================================================
# host reference: PaddedMesh.compact()
# ======================================================================================
def test_host_compact_preserves_fingerprint_and_shrinks(vsolver):
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 8., 40.))
    m0 = cm.extract_csr(bodies)
    fp0 = cm.fingerprint(m0)

    # forward then reverse -> topology restored to [I], but the arrays are full of dead slots.
    pm = PaddedMesh.from_csr(m0, v_headroom=400, s_headroom=400)
    capcap = _forward_batch(pm)
    assert len(capcap) >= 2
    for h in capcap:
        rcsr.h_to_i_csr(pm, h, dl_th=0.3)
    assert cm.fingerprint(pm.to_csr()) == fp0, "round-trip did not restore the topology"

    n_v_before, n_s_before = pm.n_v_used, pm.n_s_used
    live_v = int(pm.vert_alive[:pm.n_v_used].sum())
    live_s = int(pm.surf_alive[:pm.n_s_used].sum())
    assert n_v_before > live_v and n_s_before > live_s, "expected dead slots before compaction"

    rep = pm.compact()
    assert pm.n_v_used == live_v == m0.nv and pm.n_s_used == live_s == m0.ns
    assert rep == dict(vfreed=n_v_before - live_v, sfreed=n_s_before - live_s)
    assert pm.check_consistency() == [], "compaction produced an inconsistent mesh"
    assert cm.fingerprint(pm.to_csr()) == fp0, "compaction changed the topology"

    assert pm.compact() == dict(vfreed=0, sfreed=0), "second compact should be a no-op"


def test_host_compact_keeps_bounds_over_many_passes(vsolver):
    """Repeated forward-batch + full-reverse + compact: the high-water counters return to the
    original mesh size every pass (bounded), where without compaction they would grow each pass."""
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(40., 8., 40.))
    m0 = cm.extract_csr(bodies)
    fp0 = cm.fingerprint(m0)
    pm = PaddedMesh.from_csr(m0, v_headroom=400, s_headroom=400)

    for _ in range(4):
        capcap = _forward_batch(pm)
        assert len(capcap) >= 1
        for h in capcap:
            rcsr.h_to_i_csr(pm, h, dl_th=0.3)
        pm.compact()
        assert pm.n_v_used == m0.nv and pm.n_s_used == m0.ns, "counters not bounded by compaction"
        assert pm.check_consistency() == []
        assert cm.fingerprint(pm.to_csr()) == fp0, "a pass changed the topology"


# ======================================================================================
# device: compact_warp matches the host reference + restores bounds after device surgery
# ======================================================================================
def _make_dead(m0):
    """Forward batch then full reverse on a host PaddedMesh -> topology restored, arrays full
    of dead slots. Deterministic, so two calls give bit-identical pre-compaction state."""
    pm = PaddedMesh.from_csr(m0, v_headroom=400, s_headroom=400)
    capcap = _forward_batch(pm)
    for h in capcap:
        rcsr.h_to_i_csr(pm, h, dl_th=0.3)
    return pm, len(capcap)


def test_device_compact_matches_host(vsolver):
    """compact_warp on the device SoA == PaddedMesh.compact() -- same counters, same fingerprint,
    and (both renumber live slots in ascending old-slot order) slot-for-slot."""
    from ..gpu import compact_warp as cw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 40., 40.))
    m0 = cm.extract_csr(bodies)
    fp0 = cm.fingerprint(m0)

    pm_host, nbatch = _make_dead(m0)
    assert nbatch >= 2
    pm_dev, _ = _make_dead(m0)                    # identical pre-compaction state (deterministic)

    g = pm_dev.to_warp(device=dev)
    cw.compact_warp(g)
    pm_back = PaddedMesh.from_warp(g)
    pm_host.compact()

    assert pm_back.n_v_used == pm_host.n_v_used == m0.nv
    assert pm_back.n_s_used == pm_host.n_s_used == m0.ns
    assert pm_back.check_consistency() == [], "device compaction produced an inconsistent mesh"
    assert cm.fingerprint(pm_back.to_csr()) == cm.fingerprint(pm_host.to_csr()) == fp0
    # slot-for-slot (deterministic ascending renumber on both sides)
    np.testing.assert_array_equal(pm_back.vert_alive, pm_host.vert_alive)
    np.testing.assert_array_equal(pm_back.surf_alive, pm_host.surf_alive)
    np.testing.assert_allclose(pm_back.vert_pos[:m0.nv], pm_host.vert_pos[:m0.nv])
    np.testing.assert_array_equal(pm_back.s2v, pm_host.s2v)
    np.testing.assert_array_equal(pm_back.s2b, pm_host.s2b)
    np.testing.assert_array_equal(pm_back.v2s_len, pm_host.v2s_len)


def test_device_compact_after_device_roundtrip_restores_bounds(vsolver):
    """A full forward+reverse batch ON THE DEVICE leaves dead slots; compact_warp reclaims them,
    restoring the high-water counters to the original mesh size with the topology preserved."""
    from ..gpu import compact_warp as cw
    from ..gpu import reconnect_warp as rw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(40., 40., 8.))
    m0 = cm.extract_csr(bodies)
    fp0 = cm.fingerprint(m0)

    pm0 = PaddedMesh.from_csr(m0)
    batch = sched.independent_set([s for s in tcsr.find_short_edges_csr(pm0, threshold=1.0)
                                   if sched.i_to_h_veto_csr(pm0, s[2]) is None])
    assert len(batch) >= 2
    g = PaddedMesh.from_csr(m0, v_headroom=400, s_headroom=400).to_warp(device=dev)
    hcfgs = rw.apply_i_to_h_batch_warp(g, batch, dl_th=0.3)
    rw.apply_h_to_i_batch_warp(g, [(h.triangle, h) for h in hcfgs], dl_th=0.3)

    pm_pre = PaddedMesh.from_warp(g)
    live_v = int(pm_pre.vert_alive[:pm_pre.n_v_used].sum())
    assert pm_pre.n_v_used > live_v, "expected dead slots after the device round-trip"

    cw.compact_warp(g)
    pm_post = PaddedMesh.from_warp(g)
    assert pm_post.n_v_used == live_v == m0.nv and pm_post.n_s_used == m0.ns
    assert pm_post.check_consistency() == []
    assert cm.fingerprint(pm_post.to_csr()) == fp0, "device compaction changed the topology"
