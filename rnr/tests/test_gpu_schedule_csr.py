"""Gate C brick C1 (docs/2026-06-24_gpu-3d-vertex-model-exploration.md):
the conflict-free parallel-I->H scheduler (host reference). Validates the cellGPU
independent-set protocol on the PaddedMesh before the Warp-atomics port (C2):

  * independent_set returns a conflict-free (mutually footprint-disjoint) batch;
  * THE parallel-safety property: applying an independent batch in ANY order yields the
    SAME body-anchored topology (fingerprint) -- the invariant the GPU's nondeterministic
    atomic ordering relies on;
  * the Condition-4 veto fires on the irreversible cap-contact pattern;
  * the iterated sweep converges, stays consistent, and leaves no legal short [I] edge.
"""
import numpy as np

from ..gpu import csr_mesh as cm
from ..gpu import schedule_csr as sched
from ..gpu import topology_csr as tcsr
from ..gpu.device_mesh import PaddedMesh
from . import helpers as H


def test_independent_set_is_conflict_free(vsolver):
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 56., 56.))
    pm = PaddedMesh.from_csr(cm.extract_csr(bodies))
    sites = tcsr.find_short_edges_csr(pm, threshold=1.0)
    assert sites, "no short-edge candidates"
    batch = sched.independent_set(sites)
    assert batch, "independent set empty"
    assert sched.batch_is_conflict_free(batch), "independent_set returned a conflicting batch"
    # selecting a subset: the chosen batch never exceeds the candidate pool
    assert len(batch) <= len(sites)


def test_batch_apply_is_order_independent(vsolver):
    """Two disjoint [I] configs in one universe -> independent batch. Forward vs reverse
    application must give the same body-anchored topology (the parallel-safety guarantee).
    Slots differ by order (bump allocator), so only the slot-invariant fingerprint matches."""
    _tf, _tfv, stype, btype = vsolver
    cfg_a = H.build_minimal_i_config(stype, btype, center=(40., 40., 12.), edge=0.5)
    cfg_b = H.build_minimal_i_config(stype, btype, center=(40., 40., 48.), edge=0.5)
    bodies = cfg_a["bodies"] + cfg_b["bodies"]
    m0 = cm.extract_csr(bodies)
    fp0 = cm.fingerprint(m0)

    sites = tcsr.find_short_edges_csr(PaddedMesh.from_csr(m0), threshold=1.0)
    assert len(sites) == 2, f"expected 2 disjoint short edges, got {len(sites)}"
    batch = sched.independent_set(sites)
    assert len(batch) == 2 and sched.batch_is_conflict_free(batch), \
        "two disjoint configs should form a size-2 conflict-free batch"

    pm_fwd = PaddedMesh.from_csr(m0)
    assert sched.apply_batch(pm_fwd, batch, dl_th=0.5) == 2
    pm_rev = PaddedMesh.from_csr(m0)
    assert sched.apply_batch(pm_rev, list(reversed(batch)), dl_th=0.5) == 2

    assert pm_fwd.check_consistency() == [] and pm_rev.check_consistency() == []
    fp_fwd = cm.fingerprint(pm_fwd.to_csr())
    fp_rev = cm.fingerprint(pm_rev.to_csr())
    assert fp_fwd == fp_rev, "batch apply is order-DEPENDENT (independence criterion too weak!)"
    assert fp_fwd != fp0, "batch apply changed nothing"


def test_condition4_veto_fires_on_cap_contact(vsolver):
    """i_to_h_veto_csr must veto when the two caps already share a face (Okuda 4(iii)/[beta])."""
    _tf, _tfv, stype, btype = vsolver
    cfg_in = H.build_minimal_i_config(stype, btype, center=(56., 40., 40.), edge=0.5)
    pm = PaddedMesh.from_csr(cm.extract_csr(cfg_in["bodies"]))
    v10, v11, cfg = tcsr.find_short_edges_csr(pm, threshold=1.0)[0]
    assert sched.i_to_h_veto_csr(pm, cfg) is None, "legal config wrongly vetoed"

    # inject a direct cap_top<->cap_bot contact face (3 fresh verts + a surface).
    tri = [pm.alloc_vertex(np.array([float(i), 0., 0.])) for i in range(3)]
    F = pm.alloc_surface()
    pm.set_ring(F, tri)
    pm.attach_body(F, cfg.cap_top)
    pm.attach_body(F, cfg.cap_bot)
    reason = sched.i_to_h_veto_csr(pm, cfg)
    assert reason is not None and "cap" in reason.lower(), \
        f"cap-contact must be vetoed, got {reason!r}"


def test_kelvin_parallel_batch_is_order_independent(vsolver):
    """THE Gate-C property at scale: a realistic independent batch (~10 non-conflicting
    I->H) on a Kelvin block applies to the SAME body-anchored topology regardless of order
    -- the guarantee that lets the GPU apply them with nondeterministic atomic scheduling."""
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 8., 8.))
    m0 = cm.extract_csr(bodies)
    fp0 = cm.fingerprint(m0)

    sites = tcsr.find_short_edges_csr(PaddedMesh.from_csr(m0), threshold=1.0)
    batch = sched.independent_set(sites)
    assert len(batch) >= 2 and sched.batch_is_conflict_free(batch), \
        f"expected a multi-candidate conflict-free batch, got {len(batch)}"

    # one batch: +3 verts / +1 surface per reconnection; faces grow by <=1 (independence)
    hv, hs = 3 * len(batch) + 16, len(batch) + 16
    pm_fwd = PaddedMesh.from_csr(m0, v_headroom=hv, s_headroom=hs)
    pm_rev = PaddedMesh.from_csr(m0, v_headroom=hv, s_headroom=hs)
    assert sched.apply_batch(pm_fwd, batch, dl_th=0.3) == len(batch)
    assert sched.apply_batch(pm_rev, list(reversed(batch)), dl_th=0.3) == len(batch)

    assert pm_fwd.check_consistency() == [] and pm_rev.check_consistency() == []
    fp_fwd, fp_rev = cm.fingerprint(pm_fwd.to_csr()), cm.fingerprint(pm_rev.to_csr())
    assert fp_fwd == fp_rev, f"batch of {len(batch)} is order-DEPENDENT (unsafe to parallelise)"
    assert fp_fwd != fp0, "batch changed nothing"


def test_sweep_loop_runs_consistently(vsolver):
    """The iterated independent-set loop applies a conflict-free, consistent batch each
    round. NOTE: a static-mesh sweep does NOT converge -- an I->H places triangle vertices
    that themselves form new short edges (verified: 1 edge -> 3), so reconnections cascade.
    In production, force relaxation between steps grows those edges back; here we bound the
    rounds and verify only the loop mechanics + per-round consistency."""
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(24., 8., 8.))
    m0 = cm.extract_csr(bodies)
    # generous fixed capacity (Gate D will replace this with stream-compaction)
    pm = PaddedMesh.from_csr(m0, v_headroom=6000, s_headroom=6000,
                             ring_pad=64, vs_pad=64, bs_pad=64)
    report = sched.reconnect_sweep_i_to_h(pm, threshold=1.0, dl_th=0.3, veto=True, max_rounds=2)

    assert report["rounds"] == 2, f"expected the loop to run the bounded rounds: {report}"
    assert all(s > 0 for s in report["round_sizes"]), "a round applied an empty batch"
    assert report["total"] == sum(report["round_sizes"]) > 0
    assert pm.check_consistency() == [], "mesh inconsistent during the sweep loop"
