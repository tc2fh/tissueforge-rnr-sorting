"""Phase-D gate: native RNR runs through MeshQuality.doQuality().

The Phase-B/C tests use diagnostics and force entry points. These tests exercise the live
MeshQuality scheduler path: scan -> ReconnectionOperation::check/prep -> implement(), with
the known-crashy stock degenerate-collapse passes disabled so native RNR is isolated.
"""
import pytest

from .. import topology as topo
from . import helpers as H

from tissue_forge.models.vertex import solver as tfv


def _quality_for_live_rnr(stype, bodies, dl_th, hysteresis=0.0):
    q = tfv.Quality()
    if not hasattr(q, "stock_quality_operations"):
        pytest.skip("native RNR Phase-D quality isolation knob absent -- rebuild with `pixi run build-tf`")
    q.stock_quality_operations = False
    q.reconnect_length = dl_th
    q.reconnect_hysteresis = hysteresis
    q.reconnect_energy_gate = False
    q.collision_2d = False

    # Tests share one TF universe. Scope the surface-triggered reconnection pass to this
    # fixture's surfaces so doQuality cannot mutate meshes left by earlier tests.
    keep = H.surface_ids(bodies)
    for s in list(stype.instances):
        if s.id not in keep:
            q.exclude_surface(s.id)
    return q


def _surface_by_id(bodies, sid):
    for b in bodies:
        for s in b.getSurfaces():
            if s.id == sid:
                return s
    return None


def test_native_doquality_i_to_h_on_minimal_config(vsolver):
    _tf, tfv_mod, stype, btype = vsolver
    edge = 0.5
    cfg_in = H.build_minimal_i_config(stype, btype, center=(8., 8., 8.), edge=edge)
    bodies = cfg_in["bodies"]

    cfg = topo.i_neighbourhood(cfg_in["v10"], cfg_in["v11"])
    assert cfg is not None
    assert len(cfg.cap_top.find_interface(cfg.cap_bot)) == 0

    verts0 = H.vertex_ids(bodies)
    surfs0 = H.surface_ids(bodies)

    q = _quality_for_live_rnr(stype, bodies, dl_th=edge * 1.01)
    mesh = tfv_mod.MeshSolver.get().get_mesh()
    mesh.quality = q
    try:
        q.do_quality()
    finally:
        mesh.quality = None

    assert len(H.vertex_ids(bodies)) == len(verts0) + 1
    assert len(H.surface_ids(bodies)) == len(surfs0) + 1

    # Exactly one new surface appeared (the cap-cap triangle): compare id sets before/after.
    new_surfs = H.surface_ids(bodies) - surfs0
    assert len(new_surfs) == 1
    # The new cap-cap triangle is easiest to identify by topology: it is the only cap interface.
    cap_iface = cfg.cap_top.find_interface(cfg.cap_bot)
    assert len(cap_iface) == 1
    T = cap_iface[0]
    assert T.id in new_surfs
    assert T.validate()
    hcfg = topo.h_neighbourhood(T)
    assert hcfg is not None, "doQuality I->H did not produce a valid [H] triangle"
    assert set(hcfg.side_cell_ids) == set(cfg.side_cell_ids)
    assert all(b.volume > 0 for b in bodies)


def test_native_doquality_reconnect_length_zero_is_disabled(vsolver):
    _tf, tfv_mod, stype, btype = vsolver
    cfg_in = H.build_minimal_i_config(stype, btype, center=(8., 8., 24.), edge=0.5)
    bodies = cfg_in["bodies"]

    n_verts0 = len(H.vertex_ids(bodies))
    n_surfs0 = len(H.surface_ids(bodies))

    q = _quality_for_live_rnr(stype, bodies, dl_th=0.0)
    mesh = tfv_mod.MeshSolver.get().get_mesh()
    mesh.quality = q
    try:
        q.do_quality()
    finally:
        mesh.quality = None

    assert len(H.vertex_ids(bodies)) == n_verts0
    assert len(H.surface_ids(bodies)) == n_surfs0


def test_native_doquality_reconnect_interval_throttles(vsolver):
    """The reconnect_interval knob (the oracle's dtr) gates the reconnection pass to every
    interval-th do_quality() call. We isolate the GATE deterministically: the internal call
    counter advances on EVERY do_quality() (independently of reconnect_length), so a single
    reconnect_length=0 priming call leaves the counter odd; with interval=2 the next call is a
    SKIP (counter % 2 != 0) even though an I->H trigger is live, and the call after that is DUE
    and fires it. This proves the pass is throttled, not merely slow."""
    _tf, tfv_mod, stype, btype = vsolver
    edge = 0.5
    cfg_in = H.build_minimal_i_config(stype, btype, center=(24., 8., 24.), edge=edge)
    bodies = cfg_in["bodies"]
    n_verts0 = len(H.vertex_ids(bodies))

    q = _quality_for_live_rnr(stype, bodies, dl_th=edge * 1.01)
    if not hasattr(q, "reconnect_interval"):
        pytest.skip("native reconnect_interval knob absent -- rebuild with `pixi run build-tf`")

    # getter/setter round-trip + the <1 => 1 clamp.
    assert q.reconnect_interval == 1           # default = every step
    q.reconnect_interval = 2
    assert q.reconnect_interval == 2
    q.reconnect_interval = 0
    assert q.reconnect_interval == 1           # values < 1 are clamped to 1
    q.reconnect_interval = 2

    mesh = tfv_mod.MeshSolver.get().get_mesh()
    mesh.quality = q
    try:
        # Priming call: reconnect_length=0 makes the (due) pass a no-op, but the call counter
        # still advances 0 -> 1. The pending I->H trigger is left untouched.
        q.reconnect_length = 0.0
        q.do_quality()
        assert len(H.vertex_ids(bodies)) == n_verts0

        # Counter is now 1: with the trigger enabled, this call is a SKIP (1 % 2 != 0).
        q.reconnect_length = edge * 1.01
        q.do_quality()
        assert len(H.vertex_ids(bodies)) == n_verts0, \
            "interval=2 did not throttle: the pass ran on a skip step"

        # Counter is now 2: this call is DUE (2 % 2 == 0) and fires the pending I->H (+1 vertex).
        q.do_quality()
        assert len(H.vertex_ids(bodies)) == n_verts0 + 1, \
            "interval=2 never fired the pending reconnection on a due step"
        assert all(b.volume > 0 for b in bodies)
    finally:
        mesh.quality = None


def test_native_doquality_kelvin_smoke_stock_ops_disabled(vsolver):
    _tf, tfv_mod, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(40., 40., 8.))

    sites = topo.find_short_edges(bodies, threshold=1.0)
    assert sites, "no interior [I] short-edge sites in the Kelvin block"
    sites.sort(key=lambda t: (t[2].length, min(t[2].v10_id, t[2].v11_id), max(t[2].v10_id, t[2].v11_id)))
    v10, v11, cfg = sites[0]
    trigger_sid = min(a.side_surface.id for a in cfg.arms)

    q = _quality_for_live_rnr(stype, bodies, dl_th=cfg.length * 1.01, hysteresis=0.2)
    for s in list(stype.instances):
        if s.id != trigger_sid:
            q.exclude_surface(s.id)

    n_verts0 = len(H.vertex_ids(bodies))
    n_surfs0 = len(H.surface_ids(bodies))

    mesh = tfv_mod.MeshSolver.get().get_mesh()
    mesh.quality = q
    try:
        q.do_quality()
    finally:
        mesh.quality = None

    assert len(H.vertex_ids(bodies)) == n_verts0 + 1
    assert len(H.surface_ids(bodies)) == n_surfs0 + 1
    assert all(b.volume > 0 for b in bodies), "isolated native doQuality pass inverted a Kelvin body"
    assert all(b.validate() is not False for b in bodies)
