"""Regression tests for TissueForge periodic vertex-mesh geometry (minimum-image).

The native engine grew a `mesh.periodic_geometry` flag (Codex's periodic slice on
`feat/native-rnr-reconnection`). When ON, every vertex-mesh geometry computation
(surface centroid/area/normal, body centroid/volume, the actor force gradients, and the
RNR Condition-2 edge-length trigger) is taken under the minimum-image convention over
the `Universe::dim()` box, so a cell that straddles a periodic boundary is measured by
its SHORT image rather than the long box-spanning coordinates.

These tests are the gate for that slice. They prove, on a boundary-straddling unit cube
and a boundary-straddling Okuda [I] config:
  1. area & volume use the short image (and the flag is load-bearing: OFF -> box-spanning);
  2. the actor FORCES (Volume + SurfaceArea gradients) on a straddling cell equal those on
     the identical interior cell to float32 precision -- the dynamics are periodic-correct,
     not just the energies;
  3. the native RNR edge-length (the reconnection trigger) is the minimum-image length
     near a boundary.

NB the default boundary conditions are already PERIODIC_FULL, so the particle integrator
wraps positions itself; periodicity of the *geometry* (this flag) is the missing piece.
See rnr/PORTING_NOTES.md (periodic geometry) for the design + the wrapping caveat.

Run: pixi run test
"""
import contextlib

import numpy as np
import pytest

import tissue_forge as tf
from tissue_forge.models.vertex import solver as tfv

from . import helpers as H
from .. import geometry as G


# --------------------------------------------------------------------------------------
# a closed unit cube (8 shared vertices, 6 quad faces, 1 body), built either centred in
# the interior or with its x coordinates split across the x=0 boundary (0.5 / box-0.5).
# --------------------------------------------------------------------------------------
_CORNERS = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def _cidx(sx, sy, sz):
    return ((sx > 0) << 2) | ((sy > 0) << 1) | (sz > 0)


# face cyclic orderings; this winding yields a positive body volume (verified)
_FACES = [
    [_cidx(-1, -1, -1), _cidx(-1, 1, -1), _cidx(-1, 1, 1), _cidx(-1, -1, 1)],  # x=-h
    [_cidx(1, -1, -1), _cidx(1, 1, -1), _cidx(1, 1, 1), _cidx(1, -1, 1)],      # x=+h
    [_cidx(-1, -1, -1), _cidx(1, -1, -1), _cidx(1, -1, 1), _cidx(-1, -1, 1)],  # y=-h
    [_cidx(-1, 1, -1), _cidx(1, 1, -1), _cidx(1, 1, 1), _cidx(-1, 1, 1)],      # y=+h
    [_cidx(-1, -1, -1), _cidx(1, -1, -1), _cidx(1, 1, -1), _cidx(-1, 1, -1)],  # z=-h
    [_cidx(-1, -1, 1), _cidx(1, -1, 1), _cidx(1, 1, 1), _cidx(-1, 1, 1)],      # z=+h
]


def _build_cube(stype, btype, xfn, yz=(30.0, 30.0), h=0.5):
    """Unit cube. `xfn(sx)` returns the absolute x of the -/+ x corners, so the same
    builder makes an interior cube (xfn = 30 +/- h) or a boundary-straddling one
    (xfn -> {h, box-h}). y,z are centred at `yz`."""
    verts = [tfv.Vertex.create(tf.FVector3(xfn(sx), yz[0] + sy * h, yz[1] + sz * h))
             for (sx, sy, sz) in _CORNERS]
    surfs = [stype(vertices=[verts[i] for i in f]) for f in _FACES]
    return btype(surfs), verts


@contextlib.contextmanager
def _periodic(mesh, on):
    """Set mesh.periodic_geometry for the duration, then restore (the shared session mesh
    must be left in its historical finite-cluster mode for the other test modules)."""
    prev = mesh.periodic_geometry
    mesh.periodic_geometry = on
    try:
        yield
    finally:
        mesh.periodic_geometry = prev


def _arr(fv):
    return np.array([fv[0], fv[1], fv[2]])


# ======================================================================================
# P0: the flag exists and is settable (kept from the scaffold)
# ======================================================================================
def test_mesh_periodic_geometry_flag_is_settable(vsolver):
    _tf, _tfv, _stype, _btype = vsolver
    mesh = tfv.MeshSolver.get().get_mesh()
    assert hasattr(mesh, "periodic_geometry")
    with _periodic(mesh, False):
        assert mesh.periodic_geometry is False
    with _periodic(mesh, True):
        assert mesh.periodic_geometry is True
    assert mesh.periodic_geometry is False  # restored


# ======================================================================================
# 1. area & volume of a boundary-straddling cell use the SHORT periodic image
# ======================================================================================
def test_straddling_cell_area_volume_use_short_image(vsolver):
    _tf, _tfv, stype, btype = vsolver
    mesh = tfv.MeshSolver.get().get_mesh()
    box = tf.Universe.dim[0]
    h = 0.5

    with _periodic(mesh, True):
        interior, _ = _build_cube(stype, btype, lambda sx: 30.0 + sx * h, yz=(30.0, 30.0))
        # explicit split coords {h, box-h}: the cube spans the x=0 wall by exactly 2h
        straddle, _ = _build_cube(stype, btype,
                                  lambda sx: h if sx > 0 else box - h, yz=(20.0, 30.0))
        tfv.MeshSolver.get().position_changed()

        a_int, v_int = interior.area, interior.volume
        a_str, v_str = straddle.area, straddle.volume

    # the straddling cube must read like a unit cube, matching the interior one
    assert v_int == pytest.approx(1.0, abs=1e-4)
    assert a_int == pytest.approx(6.0, abs=1e-4)
    assert v_str == pytest.approx(v_int, abs=1e-4), "straddling volume != interior volume"
    assert a_str == pytest.approx(a_int, abs=1e-4), "straddling area != interior area"

    # negative control: the SAME split coordinates with the flag OFF give the long,
    # box-spanning measurement -> the flag is load-bearing, not a no-op.
    with _periodic(mesh, False):
        bad, _ = _build_cube(stype, btype, lambda sx: h if sx > 0 else box - h, yz=(40.0, 30.0))
        tfv.MeshSolver.get().position_changed()
        a_bad, v_bad = bad.area, bad.volume

    assert v_bad > 10.0, f"flag OFF should give box-spanning volume, got {v_bad}"
    assert a_bad > 50.0, f"flag OFF should give box-spanning area, got {a_bad}"


# ======================================================================================
# 2. the actor FORCES on a straddling cell equal the interior cell's (periodic-correct
#    gradients, not just energies). float32 roundoff from the box subtraction is ~1e-6.
# ======================================================================================
def test_periodic_forces_match_interior(vsolver):
    _tf, _tfv, stype, btype = vsolver
    mesh = tfv.MeshSolver.get().get_mesh()
    box = tf.Universe.dim[0]
    h = 0.5

    with _periodic(mesh, True):
        interior, vi = _build_cube(stype, btype, lambda sx: 30.0 + sx * h, yz=(30.0, 10.0))
        straddle, vs = _build_cube(stype, btype,
                                   lambda sx: h if sx > 0 else box - h, yz=(20.0, 10.0))
        tfv.MeshSolver.get().position_changed()

        bi = mesh.get_body(interior.id)
        bs = mesh.get_body(straddle.id)

        # VolumeConstraint gradient (off-rest-volume so the force is non-zero)
        vc = tfv.VolumeConstraint(1.0, 0.6)
        worst_v = max(
            np.linalg.norm(_arr(vc.force(bi, mesh.get_vertex(vi[n].id)))
                           - _arr(vc.force(bs, mesh.get_vertex(vs[n].id))))
            for n in range(8))

        # SurfaceAreaConstraint gradient on the matched +x faces
        sac = tfv.SurfaceAreaConstraint(1.0, 0.5)
        si = mesh.get_surface(list(interior.getSurfaces())[1].id)
        ss = mesh.get_surface(list(straddle.getSurfaces())[1].id)
        si_v = list(list(interior.getSurfaces())[1].getVertices())
        ss_v = list(list(straddle.getSurfaces())[1].getVertices())
        worst_a = max(
            np.linalg.norm(_arr(sac.force(si, mesh.get_vertex(a.id)))
                           - _arr(sac.force(ss, mesh.get_vertex(b.id))))
            for a, b in zip(si_v, ss_v))

    # forces must agree to float32 precision; the only difference is the box subtraction.
    assert worst_v < 1e-4, f"volume-constraint force differs across boundary: {worst_v}"
    assert worst_a < 1e-4, f"area-constraint force differs across boundary: {worst_a}"

    # NB body-`Adhesion` (the heterotypic-tension driver) is NOT separately tested here: its
    # `Adhesion_force_Body` area-gradient loop is line-identical to `SurfaceAreaConstraint`'s
    # (same `meshPositionNear(v, scent)` unwrap, same cross-product), differing only by the
    # scalar coefficient and the het type-pair gate -- so this Δ=0 result covers its geometry
    # path too. (A dedicated Adhesion test needs a 2-type pair sharing a straddling interface.)


# ======================================================================================
# 3. the native RNR edge-length (Condition-2 trigger) is the minimum-image length near a
#    boundary -> a short interior edge stays "short" even when it straddles the wall.
# ======================================================================================
def test_rnr_edge_length_uses_minimum_image_across_boundary(vsolver):
    _tf, _tfv, stype, btype = vsolver
    mesh = tfv.MeshSolver.get().get_mesh()
    box = tf.Universe.dim[2]

    q = tfv.Quality()
    if not hasattr(q, "analyze_i_reconnection"):
        pytest.skip("native RNR diagnostics absent -- rebuild with `pixi run build-tf`")

    with _periodic(mesh, True):
        # reference [I] config in the interior: known short edge of length `edge`
        edge = 0.5
        ref = H.build_minimal_i_config(stype, btype, center=(24.0, 24.0, 24.0), edge=edge)
        r_ref = q.analyze_i_reconnection(ref["v10"].id, ref["v11"].id)
        assert r_ref["valid"] and r_ref["legal"]
        assert r_ref["length"] == pytest.approx(edge, abs=1e-4)

        # the SAME config centred on z=0: v11 (z=-edge/2) wraps to ~box-edge/2, so the
        # short edge's two endpoints sit in different z-images.
        strd = H.build_minimal_i_config(stype, btype, center=(36.0, 24.0, 0.0), edge=edge)
        assert abs(strd["v10"].position[2] - strd["v11"].position[2]) > box / 2, \
            "test setup: the short edge must straddle the z boundary in raw coordinates"

        r_strd = q.analyze_i_reconnection(strd["v10"].id, strd["v11"].id)
        assert r_strd["valid"], "native walk failed on a boundary-straddling [I] config"
        assert r_strd["legal"], "boundary-straddling clean [I] config must not be vetoed"
        # minimum-image: the trigger length is the short edge, not the ~box-edge long image
        assert r_strd["length"] == pytest.approx(edge, abs=1e-4), \
            f"RNR edge-length is not minimum-image across the boundary: {r_strd['length']}"


# ======================================================================================
# P4. the periodic mesh GENERATOR (rnr/geometry.build_periodic_voronoi): a space-filling
# Kelvin foam in the periodic box with NO free surface -- every face interior (b1/b2 both
# set), body adjacency wrapping across the box faces. This is the initial packing for the
# bulk sorting run (PORTING_NOTES §6g "Still TODO ... a periodic Voronoi initial packing").
#
# The foam MUST fill the WHOLE universe box: the engine min-images at Universe::dim(), so a
# straddling cell is only measured by its short image when the foam wall IS a universe wall
# (a sub-box silently gives box-spanning straddling cells -- the bug this gate would catch).
# ======================================================================================
def _arr3(p):
    return np.array([p[0], p[1], p[2]])


def _pack_surfaces(bodies):
    """Unique surfaces over the returned bodies, keyed by id (scoped to this pack)."""
    surfs = {}
    for b in bodies:
        for s in b.getSurfaces():
            surfs[s.id] = s
    return surfs


def test_periodic_voronoi_pack_is_space_filling_closed_and_wraps(vsolver):
    _tf, _tfv, stype, btype = vsolver
    mesh = tfv.MeshSolver.get().get_mesh()

    # foam fills the entire universe box (required -- see module/function docstrings)
    L = float(tf.Universe.dim[0])
    assert tf.Universe.dim[1] == tf.Universe.dim[2] == tf.Universe.dim[0], "cubic box assumed"
    box = [[0.0, L], [0.0, L], [0.0, L]]
    n = 3                                            # 2*n^3 = 54 Kelvin cells; n>=3 (no self-faces)
    seeds = G.periodic_bcc_seeds(n, box)

    with _periodic(mesh, True):
        bodies, seedarr, stats = G.build_periodic_voronoi(seeds, box, btype, stype)
        tfv.MeshSolver.get().position_changed()

        vols = np.array([b.volume for b in bodies])
        boxvol = L ** 3
        surfs = _pack_surfaces(bodies)
        nbodies_per_surf = [len(s.getBodies()) for s in surfs.values()]

        # straddling bodies: raw (wrapped) vertex span > half box on some axis
        id2i = {b.id: i for i, b in enumerate(bodies)}
        straddling = []
        for i, b in enumerate(bodies):
            vp = np.array([_arr3(v.position) for v in b.getVertices()])
            if np.any(vp.max(0) - vp.min(0) > L / 2):
                straddling.append(i)

        # wrap faces: a shared surface whose two bodies' raw seeds are > half box apart
        n_wrap = 0
        for s in surfs.values():
            bs = list(s.getBodies())
            if len(bs) != 2:
                continue
            i, j = id2i[bs[0].id], id2i[bs[1].id]
            if np.any(np.abs(seedarr[i] - seedarr[j]) > L / 2):
                n_wrap += 1

    # (a) SPACE-FILLING: sum of min-image body volumes == box volume
    assert vols.sum() == pytest.approx(boxvol, abs=1e-4 * boxvol), \
        f"pack not space-filling: Σvol={vols.sum()} vs boxvol={boxvol}"
    # every cell is a finite, positive, min-image-small Kelvin cell (no box-spanning blowup)
    assert vols.min() > 0, f"non-positive body volume in pack: min={vols.min()}"
    assert vols.max() < 0.5 * boxvol, f"box-spanning body volume (min-image broken): {vols.max()}"

    # (b) NO FREE SURFACE: every surface is shared by exactly two bodies (b1 & b2 both set)
    assert set(nbodies_per_surf) == {2}, \
        f"free/under-shared surfaces present: bodies-per-surface = {set(nbodies_per_surf)}"

    # (c) ADJACENCY WRAPS: at least one genuine wrap face, and EVERY straddling body has a
    #     positive, min-image-small volume (the straddle is measured by its short image).
    assert n_wrap > 0, "no wrap faces -- body adjacency does not cross the periodic box"
    assert straddling, "no straddling bodies -- the foam does not actually use the boundary"
    for i in straddling:
        assert 0 < vols[i] < 0.5 * boxvol, \
            f"straddling body {i} volume not min-image-small: {vols[i]}"

    # the generator's own bookkeeping must agree with the engine-measured topology
    assert stats["n_self_faces"] == 0
    assert stats["n_surfaces"] == len(surfs)
    assert stats["n_wrap_faces"] == n_wrap


def test_periodic_voronoi_rejects_sub_box(vsolver):
    """The footgun guard: a foam box smaller than the universe must be refused (its walls
    are not periodic walls to the engine, so straddling cells would be box-spanning)."""
    _tf, _tfv, stype, btype = vsolver
    L = float(tf.Universe.dim[0])
    sub = [[L / 4, 3 * L / 4]] * 3                    # a centred half-size sub-box
    seeds = G.periodic_bcc_seeds(3, sub)
    with pytest.raises(ValueError, match="universe box"):
        G.build_periodic_voronoi(seeds, sub, btype, stype)
