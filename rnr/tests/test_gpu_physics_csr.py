"""Gate E, host reference (geometry half): `rnr/gpu/physics_csr.py` reproduces TissueForge's
per-surface and per-body geometry EXACTLY on a CSR extracted from a live TF mesh.

This is the foundation the force kernels stand on: the volume/area/adhesion forces all read
the cached surface centroid/area/normal + body volume/area/orientSign, so if the geometry
re-derivation matches TF, the forces have the right inputs. We build a periodic Kelvin foam
(the production initial packing), extract it to the CSR/PaddedMesh, run `compute_geometry`,
and compare to TF's own `b.volume / b.area / b.centroid / s.area / s.getCentroid` under the
same minimum-image (`mesh.periodic_geometry=True`).

Run: pixi run python -m pytest rnr/tests/test_gpu_physics_csr.py -q
"""
import contextlib

import numpy as np
import pytest

import tissue_forge as tf
from tissue_forge.models.vertex import solver as tfv

from tissue_forge.models.vertex.solver.mesh_types import BodyTypeSpec, SurfaceTypeSpec

from .. import geometry as G
from ..gpu.csr_mesh import extract_csr, id_maps
from ..gpu.device_mesh import PaddedMesh
from ..gpu import physics_csr as P


@contextlib.contextmanager
def _periodic(mesh, on):
    prev = mesh.periodic_geometry
    mesh.periodic_geometry = on
    try:
        yield
    finally:
        mesh.periodic_geometry = prev


def _arr(p):
    return np.array([p[0], p[1], p[2]])


def _build_foam(stype, btype, n=3):
    """A space-filling periodic Kelvin foam filling the whole universe box (2*n^3 cells)."""
    L = float(tf.Universe.dim[0])
    box = [[0.0, L], [0.0, L], [0.0, L]]
    seeds = G.periodic_bcc_seeds(n, box)
    bodies, _seedarr, _stats = G.build_periodic_voronoi(seeds, box, btype, stype)
    tfv.MeshSolver.get().position_changed()
    return bodies, np.array([L, L, L])


def test_geometry_matches_tf(vsolver):
    """Host compute_geometry == TF geometry on a periodic foam, to float32 round-off.

    TF's FloatP_t is single precision, and the periodic min-image is a (large) box
    subtraction, so the achievable agreement is ~1e-5 relative -- the same float32 band the
    periodic-forces regression (test_periodic_geometry.py) uses."""
    _tf, _tfv, stype, btype = vsolver
    mesh = tfv.MeshSolver.get().get_mesh()

    with _periodic(mesh, True):
        bodies, box = _build_foam(stype, btype, n=3)
        csr = extract_csr(bodies)
        pm = PaddedMesh.from_csr(csr)
        geom = P.compute_geometry(pm, box)

        vi, si, bi = id_maps(bodies)

        # ---- per-body: volume, area (extensive ~4000/~1500 -> RELATIVE), centroid (~box, abs) ----
        vol_rel = area_rel = cent_err = 0.0
        for b in bodies:
            k = bi[b.id]
            vol_rel = max(vol_rel, abs(geom.bvol[k] - b.volume) / abs(b.volume))
            area_rel = max(area_rel, abs(geom.barea[k] - b.area) / abs(b.area))
            # centroid is image-dependent: compare under min-image
            dc = P.minimg(geom.bcent[k] - _arr(b.centroid), box)
            cent_err = max(cent_err, np.linalg.norm(dc))

        # ---- per-surface: area, centroid ----
        surfs = {s.id: s for b in bodies for s in b.getSurfaces()}
        sarea_rel = scent_err = 0.0
        for sid, s in surfs.items():
            k = si[sid]
            sarea_rel = max(sarea_rel, abs(geom.sarea[k] - s.area) / abs(s.area))
            dc = P.minimg(geom.scent[k] - _arr(s.getCentroid()), box)
            scent_err = max(scent_err, np.linalg.norm(dc))

    # geometry must reproduce TF to single-precision round-off (TF's FloatP_t is float32;
    # the foam sits at box-centre coords ~30, so absolute round-off on the extensive V~4000
    # is ~5e-4 == float32 EPS relative). A wrong formula would be off by O(1), not O(1e-6).
    print(f"\n[E0 geometry] vol_rel={vol_rel:.2e} area_rel={area_rel:.2e} "
          f"|Δcent|={cent_err:.2e} s.area_rel={sarea_rel:.2e} |Δs.cent|={scent_err:.2e}")
    assert vol_rel < 1e-5, f"body volume mismatch vs TF (rel): {vol_rel}"
    assert area_rel < 1e-5, f"body area mismatch vs TF (rel): {area_rel}"
    assert cent_err < 1e-3, f"body centroid mismatch vs TF: {cent_err}"
    assert sarea_rel < 1e-5, f"surface area mismatch vs TF (rel): {sarea_rel}"
    assert scent_err < 1e-3, f"surface centroid mismatch vs TF: {scent_err}"


def test_geometry_volume_is_positive_and_space_filling(vsolver):
    """Sanity: the host geometry sees a valid foam -- all volumes positive, summing to the
    box volume (the same space-filling property TF reports)."""
    _tf, _tfv, stype, btype = vsolver
    mesh = tfv.MeshSolver.get().get_mesh()
    with _periodic(mesh, True):
        bodies, box = _build_foam(stype, btype, n=3)
        csr = extract_csr(bodies)
        pm = PaddedMesh.from_csr(csr)
        geom = P.compute_geometry(pm, box)
    boxvol = float(box[0] * box[1] * box[2])
    assert geom.bvol.min() > 0, f"non-positive host volume: {geom.bvol.min()}"
    assert geom.bvol.sum() == pytest.approx(boxvol, rel=1e-4), \
        f"host geometry not space-filling: Σvol={geom.bvol.sum()} vs {boxvol}"
    assert np.all(geom.borient == 1.0), "unexpected everted cell (orientSign=-1) in a clean foam"


# ======================================================================================
# E1: per-vertex FORCES match TF's directly-callable actor forces (the conservative half:
# volume + surface-area + adhesion). TF exposes `actor.force(body, vertex) -> FVector3`, so
# we sum the SAME actors TF's VertexForce sums (per incident body) and compare per vertex.
# Each component is checked separately so a wrong gradient is pinpointed. The TF mesh-hygiene
# regularizers (Flat/Convex) are simply not included -- they are not part of this port.
# ======================================================================================

# distinct type names so we never collide with another module's A/B types in the shared
# session universe.
_KV, _V0, _KA, _A0, _SIGMA = 10.0, 0.6, 1.0, 5.6, 0.5


class _PhysIface(SurfaceTypeSpec):
    pass


class _PhysA(BodyTypeSpec):
    volume_lam = _KV; volume_val = _V0
    surface_area_lam = _KA; surface_area_val = _A0
    adhesion = {"_PhysA": 0.0, "_PhysB": _SIGMA}


class _PhysB(BodyTypeSpec):
    volume_lam = _KV; volume_val = _V0
    surface_area_lam = _KA; surface_area_val = _A0
    adhesion = {"_PhysA": _SIGMA, "_PhysB": 0.0}


def _build_two_type_foam(n=3, seed_motility=True, jitter=0.10, rng_seed=3, ic="demixed"):
    """A periodic foam split z-low=A / z-high=B (guarantees an A-B heterotypic interface),
    with the production actors bound. Returns everything the force comparison needs.

    The seeds are JITTERED (default 10% of the lattice spacing) to make the foam
    POLYDISPERSE. This is load-bearing for the force gate: a monodisperse Kelvin foam has
    near-zero NET volume/area force at every vertex (Σ_cell dV/dx = 0 by space-filling, and
    the area gradients cancel by Kelvin symmetry), so with v0 far from the cell volume TF's
    float32 per-cell forces (~1e6) cancel to pure round-off noise (~0.1) -- a meaningless
    comparison. Polydispersity + v0/a0 set to the mean (see _run_force_case) gives genuine,
    well-conditioned O(1) net forces where host fp64 and TF fp32 agree to float32 precision."""
    L = float(tf.Universe.dim[0])
    box_list = [[0.0, L], [0.0, L], [0.0, L]]
    stype, btA, btB = _PhysIface.get(), _PhysA.get(), _PhysB.get()
    seeds = np.array(G.periodic_bcc_seeds(n, box_list))
    if jitter:
        rng = np.random.default_rng(rng_seed)
        seeds = (seeds + rng.normal(scale=jitter * (L / n), size=seeds.shape)) % L
    bodies, _sd, _st = G.build_periodic_voronoi(seeds.tolist(), box_list, btA, stype)
    b_is_B = set()
    if ic == "mixed":
        # Fig 1E initial condition: random A/B (high het-contact fraction; can demix)
        mrng = np.random.default_rng(rng_seed + 100)
        for b in bodies:
            if mrng.random() < 0.5:
                b.become(btB)
                b_is_B.add(b.id)
    else:  # "demixed": z-split slab (Fig 1F initial condition)
        half = L / 2.0
        for b in bodies:
            if b.centroid[2] >= half:
                b.become(btB)
                b_is_B.add(b.id)
    result = BodyTypeSpec.bind_adhesion([_PhysA, _PhysB])
    adh = result["_PhysA"]["_PhysB"]
    if seed_motility:
        tfv.MeshSolver.set_motility(0.1, 1.0, 12345)   # seed random-on-S^2 directors
    tfv.MeshSolver.get().position_changed()
    return bodies, np.array([L, L, L]), btA, btB, b_is_B, adh


def _tf_force_per_vertex(bodies, mesh, actors):
    """{vertex id -> summed force} of `actors` over each vertex's incident bodies, exactly
    as TF's VertexForce accumulates them (for b in v.getBodies(): for a: f += a.force(b,v))."""
    # actor.force wants the underlying Body*/Vertex* (mesh.get_body/get_vertex), not the
    # BodyHandle returned by the type constructor (test_periodic_geometry fetches the same way).
    vid2bodyids = {}
    for b in bodies:
        for v in b.getVertices():
            vid2bodyids.setdefault(v.id, set()).add(b.id)
    out = {}
    for vid, bids in vid2bodyids.items():
        vh = mesh.get_vertex(vid)
        f = np.zeros(3)
        for bid in bids:
            bh = mesh.get_body(bid)
            for a in actors:
                fa = a.force(bh, vh)
                f += np.array([fa[0], fa[1], fa[2]])
        out[vid] = f
    return out


def _worst_rel(host_f, tf_force, vi):
    """max over vertices of ||host - tf|| / ||tf||, restricted to vertices where the TF
    force is well above the float32 noise floor (so near-zero forces don't blow up the ratio)."""
    mags = np.array([np.linalg.norm(f) for f in tf_force.values()])
    floor = max(1e-6, 1e-7 * mags.max())
    worst = 0.0
    for vid, ftf in tf_force.items():
        m = np.linalg.norm(ftf)
        if m < floor:
            continue
        rel = np.linalg.norm(host_f[vi[vid]] - ftf) / m
        worst = max(worst, rel)
    return worst


def _run_force_case(vsolver, kv, ka, sigma, label):
    """Compare the host force (only the requested components enabled) to the matching TF
    actors summed per vertex. Returns the worst per-vertex relative error."""
    mesh = tfv.MeshSolver.get().get_mesh()
    with _periodic(mesh, True):
        bodies, box, btA, btB, b_is_B, adh = _build_two_type_foam(n=3)
        csr = extract_csr(bodies)
        pm = PaddedMesh.from_csr(csr)
        vi, si, bi = id_maps(bodies)

        # v0/a0 at the MEAN cell volume/area -> the volume/area constraint coefficient is the
        # per-cell DEVIATION (well-conditioned), not a uniform ~kv*V0 offset that would cancel.
        v0 = float(np.mean([b.volume for b in bodies]))
        a0 = float(np.mean([b.area for b in bodies]))

        state = P.phys_state_from_tf(bodies, lambda b: 1 if b.id in b_is_B else 0)
        params = P.PhysParams(box=box, kv=kv, v0=v0, ka=ka, a0=a0, sigma=sigma, v_active=0.0)
        geom = P.compute_geometry(pm, box)
        host_f = P.forces(pm, geom, params, state)

        actors = []
        if kv:
            actors.append(tfv.VolumeConstraint(kv, v0))
        if ka:
            actors.append(tfv.SurfaceAreaConstraint(ka, a0))
        if sigma:
            actors.append(adh)
        tf_force = _tf_force_per_vertex(bodies, mesh, actors)
        worst = _worst_rel(host_f, tf_force, vi)
    print(f"\n[E1 {label}] worst per-vertex rel err vs TF = {worst:.2e}")
    return worst


def test_force_volume_matches_tf(vsolver):
    """VolumeConstraint gradient: host == TF, per vertex, to float32."""
    assert _run_force_case(vsolver, _KV, 0.0, 0.0, "volume") < 1e-4


def test_force_surface_area_matches_tf(vsolver):
    """SurfaceAreaConstraint (body variant) gradient: host == TF."""
    assert _run_force_case(vsolver, 0.0, _KA, 0.0, "area") < 1e-4


def test_force_adhesion_matches_tf(vsolver):
    """Adhesion (heterotypic interfacial tension) gradient: host == TF. Only the A-B faces
    contribute; this is the σ that drives sorting."""
    assert _run_force_case(vsolver, 0.0, 0.0, _SIGMA, "adhesion") < 1e-4


def test_force_combined_matches_tf(vsolver):
    """All three conservative actors together: host == TF summed force per vertex."""
    assert _run_force_case(vsolver, _KV, _KA, _SIGMA, "combined") < 1e-4


def test_force_active_drive(vsolver):
    """The active self-propulsion force on each vertex is v0*<incident-cell directors>
    (the formula the native drive adds in VertexForce; its tie to TF displacement is owned by
    probe_native_calibration). Check the host's active component matches that mean directly."""
    mesh = tfv.MeshSolver.get().get_mesh()
    V0_ACT = 0.1
    with _periodic(mesh, True):
        bodies, box, btA, btB, b_is_B, adh = _build_two_type_foam(n=3)
        csr = extract_csr(bodies)
        pm = PaddedMesh.from_csr(csr)
        vi, si, bi = id_maps(bodies)
        state = P.phys_state_from_tf(bodies, lambda b: 1 if b.id in b_is_B else 0)

        # isolate the active drive (conservative coeffs zeroed) so there is no large force to
        # cancel against -- the active force IS the whole force here.
        act = P.PhysParams(box=box, kv=0.0, v0=_V0, ka=0.0, a0=_A0, sigma=0.0, v_active=V0_ACT)
        geom = P.compute_geometry(pm, box)
        f1 = P.forces(pm, geom, act, state)

        # independent reference: v0 * mean of incident body directors, per live vertex
        worst = 0.0
        for v in range(pm.n_v_used):
            if not pm.vert_alive[v]:
                continue
            bs = P.incident_bodies(pm, v)
            ref = V0_ACT * state.body_director[bs].mean(axis=0) if bs else np.zeros(3)
            worst = max(worst, np.linalg.norm(f1[v] - ref))
    print(f"\n[E1 active] worst |active - v0*<n>| = {worst:.2e}")
    assert worst < 1e-12, f"active drive component wrong: {worst}"
