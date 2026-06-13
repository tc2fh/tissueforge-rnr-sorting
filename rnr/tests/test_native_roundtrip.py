"""Phase-C gate: native C++ RNR surgery is reversible.

These tests mirror test_roundtrip.py, but drive the C++ MeshQuality debug entry points
that force exactly one reconnection. They deliberately bypass the full doQuality scan and
the stock degenerate-collapse passes; the vsolver fixture keeps mesh.quality disabled.

The final test is the Phase-P3 gate for the periodic minimum-image slice: it runs the SAME
native round-trip with mesh.periodic_geometry=True on a config whose short edge straddles a
box face, proving the I->H/H->I mutate path (placement, volume, topology walk) is
periodic-correct across a boundary. See rnr/PORTING_NOTES.md (periodic geometry).
"""
import numpy as np
import pytest

from .. import reconnect as rc
from .. import topology as topo
from . import helpers as H

import tissue_forge as tf
from tissue_forge.models.vertex import solver as tfv


def _quality(dl_th):
    q = tfv.Quality()
    if not hasattr(q, "force_reconnect_i_to_h"):
        pytest.skip("native RNR Phase-C force entry points absent -- rebuild with `pixi run build-tf`")
    q.reconnect_length = dl_th
    q.reconnect_hysteresis = 0.0
    return q


def _surface_by_id(bodies, sid):
    for b in bodies:
        for s in b.getSurfaces():
            if s.id == sid:
                return s
    return None


def test_native_minimal_roundtrip(vsolver):
    _tf, _tfv, stype, btype = vsolver
    edge = 0.5
    cfg_in = H.build_minimal_i_config(stype, btype, center=(8., 24., 40.), edge=edge)
    bodies = cfg_in["bodies"]

    cfg = topo.i_neighbourhood(cfg_in["v10"], cfg_in["v11"])
    assert cfg is not None
    snap = H.snapshot_i(cfg, bodies)

    q = _quality(edge)
    res = q.force_reconnect_i_to_h(cfg_in["v10"].id, cfg_in["v11"].id)
    assert res["ok"], res["reason"]

    T = _surface_by_id(bodies, res["new_surface_id"])
    assert T is not None and T.validate()
    assert {b.id for b in T.getBodies()} == snap.cap_ids
    assert all(b.volume > 0 for b in bodies), "native I->H produced a non-positive body volume"

    hcfg = topo.h_neighbourhood(T)
    assert hcfg is not None
    assert set(hcfg.side_cell_ids) == snap.side_cell_ids
    assert {hcfg.cap_top_id, hcfg.cap_bot_id} == snap.cap_ids

    res2 = q.force_reconnect_h_to_i(T.id)
    assert res2["ok"], res2["reason"]

    assert len(H.vertex_ids(bodies)) == snap.n_verts
    assert len(H.surface_ids(bodies)) == snap.n_surfs

    allv = H.all_vertices(bodies)
    nv10, nv11 = allv[res2["new_vertex_ids"][0]], allv[res2["new_vertex_ids"][1]]
    cfg2 = topo.i_neighbourhood(nv10, nv11) or topo.i_neighbourhood(nv11, nv10)
    assert cfg2 is not None, "recovered edge is not a valid [I] config"
    assert set(cfg2.side_cell_ids) == snap.side_cell_ids
    assert {cfg2.cap_top_id, cfg2.cap_bot_id} == snap.cap_ids
    assert len(cfg2.cap_top.find_interface(cfg2.cap_bot)) == 0

    assert H.max_outer_drift(snap, bodies) < 1e-9
    drift = H.edge_drift(snap, rc._np(nv10.position), rc._np(nv11.position))
    assert drift < 1e-5, f"minimal native round-trip edge drift {drift} not near-exact"


def test_native_kelvin_roundtrip(vsolver):
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(40., 8., 40.))

    sites = topo.find_short_edges(bodies, threshold=1.0)
    assert sites, "no interior [I] short-edge sites in the Kelvin block"
    sites.sort(key=lambda t: (min(t[2].v10_id, t[2].v11_id), max(t[2].v10_id, t[2].v11_id)))
    v10, v11, cfg = sites[0]
    dl_th = cfg.length
    snap = H.snapshot_i(cfg, bodies)

    q = _quality(dl_th)
    res = q.force_reconnect_i_to_h(v10.id, v11.id)
    assert res["ok"], res["reason"]
    T = _surface_by_id(bodies, res["new_surface_id"])
    assert T is not None and T.validate()

    hcfg = topo.h_neighbourhood(T)
    assert hcfg is not None
    assert set(hcfg.side_cell_ids) == snap.side_cell_ids

    res2 = q.force_reconnect_h_to_i(T.id)
    assert res2["ok"], res2["reason"]

    assert len(H.vertex_ids(bodies)) == snap.n_verts
    assert len(H.surface_ids(bodies)) == snap.n_surfs

    allv = H.all_vertices(bodies)
    nv10, nv11 = allv[res2["new_vertex_ids"][0]], allv[res2["new_vertex_ids"][1]]
    cfg2 = topo.i_neighbourhood(nv10, nv11) or topo.i_neighbourhood(nv11, nv10)
    assert cfg2 is not None
    assert set(cfg2.side_cell_ids) == snap.side_cell_ids
    assert {cfg2.cap_top_id, cfg2.cap_bot_id} == snap.cap_ids

    assert H.max_outer_drift(snap, bodies) < 1e-9
    drift = H.edge_drift(snap, rc._np(nv10.position), rc._np(nv11.position))
    assert drift < dl_th, f"Kelvin native round-trip edge drift {drift} exceeds dl_th {dl_th}"


def test_native_force_i_to_h_veto_leaves_mesh_untouched(vsolver):
    _tf, _tfv, stype, btype = vsolver
    cfg_in = H.build_minimal_i_config(stype, btype, center=(40., 40., 8.), edge=0.5)
    bodies = cfg_in["bodies"]
    cap_top, cap_bot = cfg_in["cap_top"], cfg_in["cap_bot"]

    pos = 0.5 * (rc._np(cap_top.centroid) + rc._np(cap_bot.centroid))
    extra = [tfv.Vertex.create(tf.FVector3(*(pos + d)))
             for d in ([0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1])]
    F = stype(vertices=extra)
    rc._attach_body(F, cap_top)
    rc._attach_body(F, cap_bot)

    n_verts, n_surfs = len(H.vertex_ids(bodies)), len(H.surface_ids(bodies))

    q = _quality(0.5)
    res = q.force_reconnect_i_to_h(cfg_in["v10"].id, cfg_in["v11"].id)
    assert res["ok"] is False and "caps" in res["reason"].lower()
    assert len(H.vertex_ids(bodies)) == n_verts
    assert len(H.surface_ids(bodies)) == n_surfs


# ======================================================================================
# Phase P3 gate: the native round-trip is periodic-correct across a box boundary.
# Same surgery as test_native_minimal_roundtrip, but mesh.periodic_geometry=True and the
# short edge straddles the z face -> the I->H/H->I placement, the body volumes, and the
# recovered geometry must all be measured under the minimum-image convention.
# ======================================================================================
def _periodic_edge_drift(snap, p10_new, p11_new, box):
    """Round-trip edge drift under minimum image (recovered endpoints may land in a
    different periodic image than the originals, so a plain coordinate diff would read
    ~box across the wall). Min over the two endpoint labellings (the pair may swap)."""
    def mi(a, b):
        d = a - b
        return float(np.linalg.norm(d - box * np.round(d / box)))
    same = max(mi(p10_new, snap.p10), mi(p11_new, snap.p11))
    swap = max(mi(p10_new, snap.p11), mi(p11_new, snap.p10))
    return min(same, swap)


def test_native_periodic_roundtrip_across_boundary(vsolver):
    _tf, _tfv, stype, btype = vsolver
    mesh = tfv.MeshSolver.get().get_mesh()
    if not hasattr(mesh, "periodic_geometry"):
        pytest.skip("periodic geometry slice absent -- rebuild with `pixi run build-tf`")
    box = np.array([tf.Universe.dim[0], tf.Universe.dim[1], tf.Universe.dim[2]])
    edge = 0.5

    prev = mesh.periodic_geometry
    mesh.periodic_geometry = True
    try:
        # Centre the [I] short edge ON the z=0 face: v10 at z=+edge/2, v11 at z=-edge/2
        # wraps to ~box-edge/2, so the edge's two endpoints sit in different z-images.
        cfg_in = H.build_minimal_i_config(stype, btype, center=(36., 24., 0.), edge=edge)
        bodies = cfg_in["bodies"]
        tfv.MeshSolver.get().position_changed()

        # test setup must genuinely straddle the boundary (else this proves nothing)
        zsep = abs(cfg_in["v10"].position[2] - cfg_in["v11"].position[2])
        assert zsep > box[2] / 2, f"short edge does not straddle z=0 (raw |dz|={zsep})"

        cfg = topo.i_neighbourhood(cfg_in["v10"], cfg_in["v11"])
        assert cfg is not None, "straddling [I] config not recognised by the topology walk"
        snap = H.snapshot_i(cfg, bodies)

        q = _quality(edge)
        res = q.force_reconnect_i_to_h(cfg_in["v10"].id, cfg_in["v11"].id)
        assert res["ok"], res["reason"]

        T = _surface_by_id(bodies, res["new_surface_id"])
        assert T is not None and T.validate()
        assert {b.id for b in T.getBodies()} == snap.cap_ids
        # min-image volumes: a straddling cell must stay positive (box-spanning would be huge/wrong)
        assert all(b.volume > 0 for b in bodies), "periodic I->H produced a non-positive volume"
        assert all(b.volume < 100.0 for b in bodies), "a straddling body measured box-spanning (not min-image)"

        hcfg = topo.h_neighbourhood(T)
        assert hcfg is not None
        assert set(hcfg.side_cell_ids) == snap.side_cell_ids
        assert {hcfg.cap_top_id, hcfg.cap_bot_id} == snap.cap_ids

        res2 = q.force_reconnect_h_to_i(T.id)
        assert res2["ok"], res2["reason"]

        # topology returns to the original [I] (counts + adjacency are image-independent)
        assert len(H.vertex_ids(bodies)) == snap.n_verts
        assert len(H.surface_ids(bodies)) == snap.n_surfs

        allv = H.all_vertices(bodies)
        nv10, nv11 = allv[res2["new_vertex_ids"][0]], allv[res2["new_vertex_ids"][1]]
        cfg2 = topo.i_neighbourhood(nv10, nv11) or topo.i_neighbourhood(nv11, nv10)
        assert cfg2 is not None, "recovered edge is not a valid [I] config"
        assert set(cfg2.side_cell_ids) == snap.side_cell_ids
        assert {cfg2.cap_top_id, cfg2.cap_bot_id} == snap.cap_ids
        assert len(cfg2.cap_top.find_interface(cfg2.cap_bot)) == 0

        # geometry returns to original under minimum image (outer verts never moved; the
        # recovered short edge matches to float32 min-image roundoff, ~1e-6 per PORTING §6g)
        assert H.max_outer_drift(snap, bodies) < 1e-9
        drift = _periodic_edge_drift(snap, rc._np(nv10.position), rc._np(nv11.position), box)
        assert drift < 1e-4, f"periodic native round-trip edge drift {drift} not near-exact"
    finally:
        mesh.periodic_geometry = prev
