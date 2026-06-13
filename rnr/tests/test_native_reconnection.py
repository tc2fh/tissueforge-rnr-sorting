"""THE PHASE-B GATE (native C++ port): the native RNR neighborhood walk + Okuda
Condition-4 guards agree with the validated Python prototype (the oracle).

Phase B ports rnr/topology.py (the I/H neighborhood walk) + rnr/conditions.py (the
Condition-4 vetoes) into TissueForge's C++ engine, exposed for testing through three
read-only diagnostic entry points on the MeshQuality object
(`analyze_i_reconnection`, `analyze_h_reconnection`, `find_reconnection_candidates`).
The mutate half (surgery + Okuda Appendix-1 placement) is Phase C, so these tests check
identification + vetoes only -- they never change the mesh.

This mirrors the Python Phase-1 gate's identification/veto half (test_roundtrip.py's
`test_minimal_config_is_canonical_I`, `test_4ii_*`, `test_4iii_*`, caps-touch), but runs
the assertions against the NATIVE engine, cross-checked edge-for-edge against the Python
oracle on the same meshes. The build must include the native port (`pixi run build-tf`);
if the diagnostic methods are missing the tests skip with a clear message.

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


# --------------------------------------------------------------------------------------
# a fresh, detached MeshQuality whose diagnostics read the global mesh
# --------------------------------------------------------------------------------------
def _quality():
    """A standalone MeshQuality (not attached to the mesh; the vsolver fixture keeps the
    mesh's own quality disabled). Its read-only diagnostics operate on the global mesh via
    Mesh::get(). Skips the whole module if the native Phase-B port isn't in this build."""
    q = tfv.Quality()
    if not hasattr(q, "analyze_i_reconnection"):
        pytest.skip("native RNR Phase-B diagnostics absent -- rebuild with `pixi run build-tf`")
    return q


# ======================================================================================
# canonical [I] identification (native walk == Python oracle)
# ======================================================================================
def test_native_identifies_canonical_I(vsolver):
    _tf, _tfv, stype, btype = vsolver
    cfg_in = H.build_minimal_i_config(stype, btype, center=(24., 8., 24.))
    v10, v11 = cfg_in["v10"], cfg_in["v11"]

    # the Python oracle's view (the spec we must reproduce).
    ocfg = topo.i_neighbourhood(v10, v11)
    assert ocfg is not None and cond.i_to_h_veto(ocfg) is None

    q = _quality()
    res = q.analyze_i_reconnection(v10.id, v11.id)

    assert res["valid"] is True, "native walk failed to recognise the canonical [I] config"
    assert res["kind"] == "I"
    assert {res["v10_id"], res["v11_id"]} == {v10.id, v11.id}
    assert set(res["side_cell_ids"]) == set(ocfg.side_cell_ids), "native side cells != oracle"
    assert {res["cap_top_id"], res["cap_bot_id"]} == {ocfg.cap_top_id, ocfg.cap_bot_id}
    assert res["legal"] is True and res["veto_reason"] == "", "legal config must not be vetoed"
    # geometry: the native (periodic-correct) length matches the oracle's plain length here.
    assert abs(res["length"] - ocfg.length) < 1e-5


def test_native_rejects_non_interior_edge(vsolver):
    """A boundary/free-surface vertex pair is not a reconnection site (endpoints not 4-cell)."""
    _tf, _tfv, stype, btype = vsolver
    cfg_in = H.build_minimal_i_config(stype, btype, center=(24., 8., 40.))
    # an outer (cap-base) vertex is on the free surface -> fewer than 4 cells.
    outer = cfg_in["top"][0]
    v11 = cfg_in["v11"]
    assert len(outer.getBodies()) != 4
    q = _quality()
    res = q.analyze_i_reconnection(outer.id, v11.id)
    assert res["valid"] is False


# ======================================================================================
# Condition-4 vetoes for I->H (native i_to_h_veto)
# ======================================================================================
def test_native_i_veto_when_caps_touch(vsolver):
    """[beta] 4(iii): the two caps already sharing a face must veto I->H (adding the
    triangle would make a double trigonal face). Mirrors test_i_to_h_refuses_when_caps_
    already_touch, but checks the NATIVE veto."""
    _tf, _tfv, stype, btype = vsolver
    cfg_in = H.build_minimal_i_config(stype, btype, center=(40., 24., 24.))
    v10, v11 = cfg_in["v10"], cfg_in["v11"]
    cap_top, cap_bot = cfg_in["cap_top"], cfg_in["cap_bot"]

    q = _quality()
    assert q.analyze_i_reconnection(v10.id, v11.id)["legal"] is True  # legal before injection

    # inject a direct cap_top<->cap_bot contact face (3 fresh verts).
    pos = 0.5 * (rc._np(cap_top.centroid) + rc._np(cap_bot.centroid))
    extra = [tfv.Vertex.create(tf.FVector3(*(pos + d)))
             for d in ([0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1])]
    F = stype(vertices=extra)
    rc._attach_body(F, cap_top)
    rc._attach_body(F, cap_bot)

    res = q.analyze_i_reconnection(v10.id, v11.id)
    assert res["valid"] is True          # still a structural [I] config
    assert res["legal"] is False and "caps" in res["veto_reason"].lower()


def test_native_i_veto_when_side_cells_share_two_faces(vsolver):
    """4(iii): two side cells already sharing >=2 faces must veto I->H. Mirrors
    test_4iii_detects_double_trigonal_face but exercises the composite native veto."""
    _tf, _tfv, stype, btype = vsolver
    cfg_in = H.build_minimal_i_config(stype, btype, center=(8., 24., 24.))
    v10, v11 = cfg_in["v10"], cfg_in["v11"]
    w0, w1 = cfg_in["wedges"][0], cfg_in["wedges"][1]

    q = _quality()
    assert q.analyze_i_reconnection(v10.id, v11.id)["legal"] is True

    # inject a second shared face between two SIDE cells (they already share one side face).
    pos = rc._np(w0.centroid)
    extra = [tfv.Vertex.create(tf.FVector3(*(pos + d)))
             for d in ([0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1])]
    F = stype(vertices=extra)
    rc._attach_body(F, w0)
    rc._attach_body(F, w1)
    assert len(w0.find_interface(w1)) >= 2

    res = q.analyze_i_reconnection(v10.id, v11.id)
    assert res["legal"] is False and "share >=2 faces" in res["veto_reason"]


# ======================================================================================
# [H] identification + H->I veto (native walk, on a Python-built H state)
# ======================================================================================
def test_native_identifies_H_and_veto(vsolver):
    """Build the [H] state with the validated Python I->H, then check the NATIVE H walk
    parses it like the oracle (legal), and that a second cap-cap face vetoes H->I ([beta])."""
    _tf, _tfv, stype, btype = vsolver
    cfg_in = H.build_minimal_i_config(stype, btype, center=(24., 40., 24.), edge=0.5)
    cfg = topo.i_neighbourhood(cfg_in["v10"], cfg_in["v11"])
    assert cfg is not None

    res = rc.i_to_h(cfg, dl_th=0.5, stype=stype)   # oracle mutate: edge -> triangle
    assert res.ok, res.reason
    T = res.new_surface
    assert T.validate()

    ohcfg = topo.h_neighbourhood(T)                # oracle's view of the H state
    assert ohcfg is not None

    q = _quality()
    hres = q.analyze_h_reconnection(T.id)
    assert hres["valid"] is True, "native H walk failed on a Python-built triangle"
    assert hres["kind"] == "H" and hres["triangle_id"] == T.id
    assert set(hres["side_cell_ids"]) == set(ohcfg.side_cell_ids)
    assert {hres["cap_top_id"], hres["cap_bot_id"]} == {ohcfg.cap_top_id, ohcfg.cap_bot_id}
    assert hres["legal"] is True and hres["veto_reason"] == ""  # caps share exactly the triangle

    # inject a SECOND cap-cap face -> the caps now share 2 faces -> H->I vetoed.
    cap_top = next(b for b in T.getBodies() if b.id == ohcfg.cap_top_id)
    cap_bot = next(b for b in T.getBodies() if b.id == ohcfg.cap_bot_id)
    pos = 0.5 * (rc._np(cap_top.centroid) + rc._np(cap_bot.centroid))
    extra = [tfv.Vertex.create(tf.FVector3(*(pos + d)))
             for d in ([0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1])]
    F = stype(vertices=extra)
    rc._attach_body(F, cap_top)
    rc._attach_body(F, cap_bot)

    hres2 = q.analyze_h_reconnection(T.id)
    assert hres2["valid"] is True
    assert hres2["legal"] is False and "caps share >=2 faces" in hres2["veto_reason"]


# ======================================================================================
# the scan (Condition-2 trigger) == the Python oracle, on a real Kelvin block
# ======================================================================================
def test_native_scan_matches_oracle_on_kelvin(vsolver):
    """find_reconnection_candidates (the same scanners doQuality uses) discovers exactly the
    interior short-edge [I] sites the Python oracle find_short_edges does, on a Kelvin block."""
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 40., 40.))
    threshold = 1.0

    oracle = {(min(c.v10_id, c.v11_id), max(c.v10_id, c.v11_id))
              for _v, _w, c in topo.find_short_edges(bodies, threshold)}
    assert oracle, "no interior [I] short-edge sites in the Kelvin block"

    block_vids = H.vertex_ids(bodies)
    q = _quality()
    q.reconnect_length = threshold
    cands = q.find_reconnection_candidates()

    # the scan is whole-mesh (the session universe holds every test's mesh); scope to this block.
    native = {(min(c["v10_id"], c["v11_id"]), max(c["v10_id"], c["v11_id"]))
              for c in cands if c["kind"] == "I" and c["v10_id"] in block_vids}

    assert native == oracle, (
        f"native scan != oracle\n  only native: {sorted(native - oracle)}"
        f"\n  only oracle: {sorted(oracle - native)}")
    # a clean Kelvin interior has no irreversible patterns: every block candidate is legal.
    for c in cands:
        if c["kind"] == "I" and c["v10_id"] in block_vids:
            assert c["legal"] is True, f"unexpected veto {c['veto_reason']} on Kelvin edge"
