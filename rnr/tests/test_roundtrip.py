"""THE PHASE-1 GATE: the Okuda I<->H reconnection is reversible, and the Condition-4
guards veto the irreversible patterns.

Reversibility is the entire point of RNR (CLAUDE.md). The round-trip restores:
  * TOPOLOGY exactly  -- same vertex/surface counts; the recovered short edge is again a
    valid [I] config between the SAME 5 cells; the caps are no longer in contact;
  * GEOMETRY within O(Delta_l_th) (Okuda Eqs. 5-7) -- all OTHER vertices are preserved
    byte-for-byte (never moved), and the recovered edge endpoints land within
    Delta_l_th of the originals (exactly so when the original edge length == Delta_l_th,
    as in the hand-built minimal config).

Run: pixi run test
"""
import numpy as np
import pytest

from .. import conditions as cond
from .. import reconnect as rc
from .. import topology as topo
from . import helpers as H

import tissue_forge as tf
from tissue_forge.models.vertex import solver as tfv


# ======================================================================================
# round-trip: hand-built minimal config (the strict gate -- exact geometric restoration)
# ======================================================================================
def test_minimal_config_is_canonical_I(vsolver):
    _tf, _tfv, stype, btype = vsolver
    cfg_in = H.build_minimal_i_config(stype, btype, center=(8., 8., 8.))
    v10, v11 = cfg_in["v10"], cfg_in["v11"]

    # the short edge endpoints are interior (4 cells each); caps not yet touching.
    assert len(v10.getBodies()) == 4 and len(v11.getBodies()) == 4
    cfg = topo.i_neighbourhood(v10, v11)
    assert cfg is not None, "minimal config is not a recognised [I] neighbourhood"
    assert len(cfg.side_cell_ids) == 3
    assert len(cfg.cap_top.find_interface(cfg.cap_bot)) == 0, "caps already touch"
    assert cond.i_to_h_veto(cfg) is None, "legal config should not be vetoed"


def test_minimal_roundtrip(vsolver):
    _tf, _tfv, stype, btype = vsolver
    cfg_in = H.build_minimal_i_config(stype, btype, center=(8., 24., 8.), edge=0.5)
    bodies = cfg_in["bodies"]
    edge = 0.5

    cfg = topo.i_neighbourhood(cfg_in["v10"], cfg_in["v11"])
    assert cfg is not None
    snap = H.snapshot_i(cfg, bodies)

    # I -> H
    res = rc.i_to_h(cfg, dl_th=edge, stype=stype)
    assert res.ok, res.reason
    T = res.new_surface
    assert T.validate()
    assert {b.id for b in T.getBodies()} == snap.cap_ids, "triangle must be the caps' new contact"
    assert all(b.volume > 0 for b in bodies), "I->H produced a non-positive body volume"

    hcfg = topo.h_neighbourhood(T)
    assert hcfg is not None, "I->H produced a triangle H->I can't parse"
    assert set(hcfg.side_cell_ids) == snap.side_cell_ids
    assert {hcfg.cap_top_id, hcfg.cap_bot_id} == snap.cap_ids

    # H -> I
    res2 = rc.h_to_i(hcfg, dl_th=edge)
    assert res2.ok, res2.reason

    # topology restored exactly
    assert len(H.vertex_ids(bodies)) == snap.n_verts
    assert len(H.surface_ids(bodies)) == snap.n_surfs
    allv = H.all_vertices(bodies)
    nv10, nv11 = allv[res2.new_vertex_ids[0]], allv[res2.new_vertex_ids[1]]
    cfg2 = topo.i_neighbourhood(nv10, nv11) or topo.i_neighbourhood(nv11, nv10)
    assert cfg2 is not None, "recovered edge is not a valid [I] config"
    assert set(cfg2.side_cell_ids) == snap.side_cell_ids
    assert {cfg2.cap_top_id, cfg2.cap_bot_id} == snap.cap_ids
    # caps separated again
    assert len(cfg2.cap_top.find_interface(cfg2.cap_bot)) == 0

    # geometry: outer verts byte-identical; edge verts recovered EXACTLY (edge==dl_th).
    assert H.max_outer_drift(snap, bodies) < 1e-9
    drift = H.edge_drift(snap, rc._np(nv10.position), rc._np(nv11.position))
    assert drift < 1e-6, f"minimal round-trip edge drift {drift} not ~exact"


# ======================================================================================
# round-trip: real Kelvin interior edge (realistic mid-scale smoke test)
# ======================================================================================
def test_kelvin_roundtrip(vsolver):
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(40., 40., 40.))

    sites = topo.find_short_edges(bodies, threshold=1.0)
    assert sites, "no interior [I] short-edge sites in the Kelvin block"
    # deterministic pick: smallest edge endpoints.
    sites.sort(key=lambda t: (min(t[2].v10_id, t[2].v11_id), max(t[2].v10_id, t[2].v11_id)))
    v10, v11, cfg = sites[0]
    dl_th = cfg.length            # near-exact round-trip
    snap = H.snapshot_i(cfg, bodies)

    res = rc.i_to_h(cfg, dl_th=dl_th, stype=stype)
    assert res.ok, res.reason
    T = res.new_surface
    assert T.validate()
    hcfg = topo.h_neighbourhood(T)
    assert hcfg is not None
    assert set(hcfg.side_cell_ids) == snap.side_cell_ids

    res2 = rc.h_to_i(hcfg, dl_th=dl_th)
    assert res2.ok, res2.reason

    assert len(H.vertex_ids(bodies)) == snap.n_verts
    assert len(H.surface_ids(bodies)) == snap.n_surfs
    allv = H.all_vertices(bodies)
    nv10, nv11 = allv[res2.new_vertex_ids[0]], allv[res2.new_vertex_ids[1]]
    cfg2 = topo.i_neighbourhood(nv10, nv11) or topo.i_neighbourhood(nv11, nv10)
    assert cfg2 is not None
    assert set(cfg2.side_cell_ids) == snap.side_cell_ids
    assert {cfg2.cap_top_id, cfg2.cap_bot_id} == snap.cap_ids

    assert H.max_outer_drift(snap, bodies) < 1e-9
    drift = H.edge_drift(snap, rc._np(nv10.position), rc._np(nv11.position))
    assert drift < dl_th, f"Kelvin round-trip edge drift {drift} exceeds dl_th {dl_th}"


# ======================================================================================
# Condition-4 veto tests (Okuda Figs. 6 / 9: double edges, double trigonal faces)
# ======================================================================================
def test_4ii_detects_double_edge(vsolver):
    """4(ii): two faces sharing two SEPARATE edges -> the [alpha] double-edge pattern."""
    _tf, _tfv, stype, btype = vsolver
    C = np.array([8., 40., 8.])
    P = lambda dx, dy, dz: tfv.Vertex.create(tf.FVector3(*(C + [dx, dy, dz])))
    a0, a1, a2, a3 = P(0, 0, 0), P(1, 0, 0), P(2, 1, 0), P(0, 1, 0)
    x0, x1 = P(0.5, -1, 0), P(1.5, -1, 0)
    y0, y1 = P(0.5, 2, 0.2), P(1.5, 2, 0.2)
    # two hexagons sharing edge (a0,a1) AND edge (a2,a3) -- two disjoint shared edges.
    fa = stype(vertices=[a0, a1, x1, a2, a3, x0])
    fb = stype(vertices=[a1, a0, y0, a3, a2, y1])
    assert fa.num_shared_contiguous_vertex_sets(fb) >= 2
    assert cond.faces_share_multiple_edges(fa, fb) is True

    # control: two quads sharing only ONE edge (a0,a1) -> not flagged.
    b0, b1 = P(0, 5, 0), P(1, 5, 0)
    qc = stype(vertices=[a0, a1, b1, b0])
    assert cond.faces_share_multiple_edges(fa, qc) is False


def test_4iii_detects_double_trigonal_face(vsolver):
    """4(iii): two cells sharing >= 2 faces -> the [beta] double trigonal-face pattern.

    Start from the minimal config (wedge0 & wedge1 share exactly ONE face), inject a
    second shared face between them, and confirm the guard flips.
    """
    _tf, _tfv, stype, btype = vsolver
    cfg_in = H.build_minimal_i_config(stype, btype, center=(24., 40., 8.))
    w0, w1 = cfg_in["wedges"][0], cfg_in["wedges"][1]
    assert cond.cells_share_multiple_faces(w0, w1) is False  # exactly one side face

    # inject a second shared face between the same two cells.
    pos = rc._np(w0.centroid)
    extra = [tfv.Vertex.create(tf.FVector3(*(pos + d)))
             for d in ([0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1])]
    F = stype(vertices=extra)
    rc._attach_body(F, w0)
    rc._attach_body(F, w1)
    assert len(w0.find_interface(w1)) >= 2
    assert cond.cells_share_multiple_faces(w0, w1) is True


def test_i_to_h_refuses_when_caps_already_touch(vsolver):
    """The I->H guard must VETO (and leave the mesh untouched) if the two caps already
    share a face -- adding the triangle would make a double trigonal face [beta]."""
    _tf, _tfv, stype, btype = vsolver
    cfg_in = H.build_minimal_i_config(stype, btype, center=(40., 8., 8.))
    bodies = cfg_in["bodies"]
    cap_top, cap_bot = cfg_in["cap_top"], cfg_in["cap_bot"]

    cfg = topo.i_neighbourhood(cfg_in["v10"], cfg_in["v11"])
    assert cfg is not None and cond.i_to_h_veto(cfg) is None  # legal before injection

    # inject a direct cap_top<->cap_bot contact face.
    pos = 0.5 * (rc._np(cap_top.centroid) + rc._np(cap_bot.centroid))
    extra = [tfv.Vertex.create(tf.FVector3(*(pos + d)))
             for d in ([0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1])]
    F = stype(vertices=extra)
    rc._attach_body(F, cap_top)
    rc._attach_body(F, cap_bot)

    n_verts, n_surfs = len(H.vertex_ids(bodies)), len(H.surface_ids(bodies))
    cfg = topo.i_neighbourhood(cfg_in["v10"], cfg_in["v11"])
    assert cfg is not None
    assert cond.i_to_h_veto(cfg) is not None, "caps-touch must be vetoed"

    res = rc.i_to_h(cfg, dl_th=0.5, stype=stype)
    assert res.ok is False and "caps" in res.reason.lower()
    # mesh untouched by the refused operation.
    assert len(H.vertex_ids(bodies)) == n_verts
    assert len(H.surface_ids(bodies)) == n_surfs
