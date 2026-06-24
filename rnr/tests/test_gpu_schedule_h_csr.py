"""Gate C reverse direction -- brick C1' (host reference): the conflict-free parallel-H->I
scheduler on the PaddedMesh (docs/2026-06-24_gpu-3d-vertex-model-exploration.md). The mirror
of test_gpu_schedule_csr.py for the reverse reconnection (a small triangular face collapses
back to a short edge):

  * h_independent_set returns a conflict-free (mutually H-footprint-disjoint) batch;
  * an independent reverse batch applies ORDER-INDEPENDENTLY (same body-anchored fingerprint,
    forwards and backwards) -- and a forward batch then its reverse restores the [I] topology;
  * the Condition-4 veto fires on a double cap-cap contact ([beta]);
  * the iterated reverse sweep runs consistently, and a pure-[I] mesh has no [H] sites.

[H] sites only exist AFTER a forward I->H, so every test seeds them with a forward batch first.
"""
import numpy as np

from ..gpu import csr_mesh as cm
from ..gpu import reconnect_csr as rcsr
from ..gpu import schedule_csr as sched
from ..gpu import topology_csr as tcsr
from ..gpu.device_mesh import PaddedMesh
from . import helpers as H


def _seed_triangles(m0, dl_th=0.3, **pm_kwargs):
    """Apply a conflict-free forward I->H batch to a fresh PaddedMesh(m0); return
    (pm, batch, capcap): pm is the post-forward mesh (now bearing triangular faces), batch the
    forward [I] candidates, capcap the cap-cap HCfgIdx (the canonical, mutually-disjoint reverse
    sites). Deterministic (fixed batch order), so two calls give bit-identical slot layouts.
    Default headroom covers fwd(+3) + a full rev(+2) = 5 verts/op (matches the detector test)."""
    sites = tcsr.find_short_edges_csr(PaddedMesh.from_csr(m0), threshold=1.0)
    sites = [s for s in sites if sched.i_to_h_veto_csr(PaddedMesh.from_csr(m0), s[2]) is None]
    batch = sched.independent_set(sites)
    if "v_headroom" not in pm_kwargs:
        pm_kwargs["v_headroom"] = 5 * len(batch) + 16
    if "s_headroom" not in pm_kwargs:
        pm_kwargs["s_headroom"] = len(batch) + 16
    pm = PaddedMesh.from_csr(m0, **pm_kwargs)
    capcap = [rcsr.i_to_h_csr(pm, cfg, dl_th) for (_v10, _v11, cfg) in batch]
    return pm, batch, capcap


def test_h_independent_set_is_conflict_free(vsolver):
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 8., 56.))
    m0 = cm.extract_csr(bodies)
    pm, batch, _capcap = _seed_triangles(m0)
    assert len(batch) >= 2, f"need a multi-reconnection forward batch, got {len(batch)}"

    sites = tcsr.find_small_triangles_csr(pm, threshold=1.0)
    assert sites, "no [H] sites after the forward batch"
    h_batch = sched.h_independent_set(sites)
    assert h_batch, "H independent set empty"
    assert sched.h_batch_is_conflict_free(h_batch), "h_independent_set returned a conflicting batch"
    assert len(h_batch) <= len(sites)


def test_h_to_i_veto_fires_on_double_cap_contact(vsolver):
    """h_to_i_veto_csr must pass the legal cap-cap triangle (the caps share ONLY it) and veto
    once a SECOND cap_top<->cap_bot contact face exists (Okuda 4(iii)/[beta])."""
    _tf, _tfv, stype, btype = vsolver
    cfg_in = H.build_minimal_i_config(stype, btype, center=(56., 24., 24.), edge=0.5)
    pm = PaddedMesh.from_csr(cm.extract_csr(cfg_in["bodies"]), v_headroom=64, s_headroom=64)
    cfg = tcsr.find_short_edges_csr(pm, threshold=1.0)[0][2]
    hcfg = rcsr.i_to_h_csr(pm, cfg, dl_th=0.5)
    assert sched.h_to_i_veto_csr(pm, hcfg) is None, "legal H config wrongly vetoed"

    # inject a direct second cap_top<->cap_bot contact face (3 fresh verts + a surface).
    tri = [pm.alloc_vertex(np.array([100. + i, 0., 0.])) for i in range(3)]
    F = pm.alloc_surface()
    pm.set_ring(F, tri)
    pm.attach_body(F, hcfg.cap_top)
    pm.attach_body(F, hcfg.cap_bot)
    reason = sched.h_to_i_veto_csr(pm, hcfg)
    assert reason is not None and "cap" in reason.lower(), \
        f"double cap contact must be vetoed, got {reason!r}"


def test_h_batch_apply_is_order_independent(vsolver):
    """An independent reverse (cap-cap) batch applies to the SAME body-anchored topology in
    any order (the parallel-safety guarantee for H->I), and forward-then-reverse restores the
    original [I] fingerprint. Slots differ by order (bump allocator), so only the slot-invariant
    fingerprint matches."""
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(56., 8., 8.))
    m0 = cm.extract_csr(bodies)
    fp0 = cm.fingerprint(m0)

    # _seed_triangles is deterministic -> both calls yield identical post-forward slot layouts,
    # so the two reverse batches are element-identical; we apply one forwards, one reversed.
    pm_fwd, batch, capcap_f = _seed_triangles(m0)
    rev_fwd = [(h.triangle, h) for h in capcap_f]
    assert sched.h_batch_is_conflict_free(rev_fwd), "cap-cap reverse batch not conflict-free"
    assert sched.h_apply_batch(pm_fwd, rev_fwd, dl_th=0.3) == len(batch)

    pm_rev, _b2, capcap_r = _seed_triangles(m0)
    rev_rev = [(h.triangle, h) for h in capcap_r]
    assert sched.h_apply_batch(pm_rev, list(reversed(rev_rev)), dl_th=0.3) == len(batch)

    assert pm_fwd.check_consistency() == [] and pm_rev.check_consistency() == []
    fp_fwd, fp_rev = cm.fingerprint(pm_fwd.to_csr()), cm.fingerprint(pm_rev.to_csr())
    assert fp_fwd == fp_rev, "reverse batch apply is order-DEPENDENT (independence violated)"
    assert fp_fwd == fp0, "forward-then-reverse did not restore the [I] topology"


def test_reverse_sweep_runs_consistently(vsolver):
    """The iterated reverse (reservation) sweep reconnects >=1 [H] site per round and keeps the
    mesh consistent. Generous pads: the forward batch already grew faces/valence, and each H->I
    inserts a vertex into the side faces (mirrors the forward sweep's bounded-rounds test)."""
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 56., 8.))
    m0 = cm.extract_csr(bodies)
    pm, _batch, _capcap = _seed_triangles(m0, v_headroom=4000, s_headroom=4000,
                                          ring_pad=64, vs_pad=64, bs_pad=64)
    report = sched.reconnect_sweep_h_reserve_host(pm, threshold=1.0, dl_th=0.3, max_rounds=3)
    assert report["rounds"] >= 1 and report["total"] >= 1, report
    assert all(s > 0 for s in report["round_sizes"]), "a round applied an empty batch"
    assert pm.check_consistency() == [], "mesh inconsistent during the reverse sweep"


def test_reverse_sweep_pure_i_mesh_is_noop(vsolver):
    """A freshly-built Kelvin block (no triangular faces) has no [H] sites -- the reverse sweep
    must be a 0-round no-op."""
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(24., 56., 24.))
    pm = PaddedMesh.from_csr(cm.extract_csr(bodies))
    report = sched.reconnect_sweep_h_to_i(pm, threshold=1.0, dl_th=0.3, max_rounds=4)
    assert report["rounds"] == 0 and report["total"] == 0, \
        "pure [I] mesh should have no [H] sites to reverse"
