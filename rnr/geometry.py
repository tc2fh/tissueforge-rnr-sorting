"""Voronoi -> TissueForge finite-cluster vertex-mesh construction (Phase 0).

A 3D Voronoi tessellation maps almost 1:1 onto TissueForge's vertex model:
  Voronoi cell    -> Body (b1/b2 set on shared faces)
  Voronoi face    -> Surface (planar, which satisfies TF's flat-surface rule)
  Voronoi vertex  -> Vertex
The one non-obvious step is that pyvoro reports each cell with its OWN local vertex
list and each shared face TWICE (once per adjacent cell). We dedup vertices and
faces GLOBALLY so each interior face becomes a SINGLE TF Surface referenced by both
neighbouring bodies (so Surface.b1/b2 are both set). Box-wall faces (pyvoro
adjacent_cell < 0) belong to one body only and form the cluster's free outer
boundary -- this is the "finite cluster" boundary (the vertex layer has no periodic
support; see PORTING_NOTES.md).

Validated against pyvoro: built bodies' volumes match pyvoro cell volumes to ~1e-5,
the mesh is space-filling (sum of body volumes == bounding-box volume), interior
faces == 2-body surfaces, wall faces == 1-body surfaces, and TF body adjacency
matches the pyvoro interior-face count.

GOTCHAS (all bite headless macOS runs; see PORTING_NOTES.md):
  * tf.init(windowless=True) is REQUIRED headless or tf.init() blocks on GL/window
    context creation.
  * tf.init() is a singleton -- calling it twice in one process hangs. One init per
    process.
  * pyvoro-mmalahe returns overlapping full-box cells when its `dispersion` arg is
    smaller than the seed spacing. We default dispersion to the largest box edge so
    it is always >= spacing -> a correct space-filling tessellation.
  * Handle identity: BodyHandle.id (not Python id()) is the stable mesh identifier;
    `connected_bodies` etc. return fresh wrapper objects each call.
"""
import itertools
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pyvoro
import tissue_forge as tf
from tissue_forge.models.vertex import solver as tfv


Box = Sequence[Tuple[float, float]]   # [[xlo,xhi],[ylo,yhi],[zlo,zhi]]


def random_seeds(n: int, box: Box, rng_seed: int = 0) -> List[List[float]]:
    """`n` uniformly-random seed points inside `box`. Reproducible via rng_seed."""
    rng = np.random.default_rng(rng_seed)
    lo = np.array([b[0] for b in box], dtype=float)
    hi = np.array([b[1] for b in box], dtype=float)
    return (rng.random((n, 3)) * (hi - lo) + lo).tolist()


def bcc_seeds(n_per_axis: int, box: Box, jitter: float = 0.0, rng_seed: int = 0) -> List[List[float]]:
    """Seeds on a BCC lattice filling `box` (≈cubic box assumed). The Voronoi cells of
    a BCC lattice are tetrakaidecahedra — Kelvin cells: 8 hexagons + 6 squares, equal
    volume — i.e. the monodisperse "ideal foam" of Okuda et al. 2013, Fig. 7a.

    Yields 2*n_per_axis**3 seeds. Built through `build_voronoi_cluster` (which clips to
    the box), the INTERIOR cells come out as perfect Kelvin cells; the boundary cells are
    clipped to the box and form the cluster's free surface (TissueForge's vertex layer
    has no periodic boundary, so we use a finite block rather than the paper's periodic
    box). `jitter` (fraction of the lattice edge) optionally perturbs seeds for testing
    robustness to non-ideal packing; default 0 = the exact lattice.
    """
    lo = np.array([b[0] for b in box], dtype=float)
    hi = np.array([b[1] for b in box], dtype=float)
    a = (hi - lo) / n_per_axis                      # cubic cell edge per axis
    rng = np.random.default_rng(rng_seed)
    pts = []
    for i in range(n_per_axis):
        for j in range(n_per_axis):
            for k in range(n_per_axis):
                base = lo + (np.array([i, j, k]) + 0.5) * a      # cell-centred sublattice
                pts.append(base)
                bc = base + 0.5 * a                              # body-centre sublattice (BCC offset)
                if np.all(bc < hi):
                    pts.append(bc)
    pts = np.array(pts)
    if jitter:
        pts = pts + rng.normal(scale=jitter * float(np.mean(a)), size=pts.shape)
    return pts.tolist()


def build_voronoi_cluster(points, limits: Box, btype, stype,
                          dispersion: Optional[float] = None,
                          key_decimals: int = 6):
    """Build a finite-cluster TF vertex mesh from a pyvoro tessellation of `points`
    clipped to `limits`.

    All bodies are created as `btype` (assign a second type afterwards with
    BodyHandle.become(...)). `stype` is the surface type for every face.

    Returns (bodies, cells, stats) where `bodies[i]` corresponds to `points[i]`,
    `cells` is the raw pyvoro output, and `stats` has n_internal / n_wall /
    n_vertices counts.
    """
    if dispersion is None:
        # pyvoro-mmalahe gotcha: dispersion must be >= seed spacing or it returns
        # overlapping full-box cells. Largest box edge is always safe.
        dispersion = max(hi - lo for lo, hi in limits)
    cells = pyvoro.compute_voronoi(points, list(limits), dispersion)

    # --- global vertex dedup (pyvoro gives each cell its own vertex list) ---
    gpos: List[list] = []
    gkey = {}
    cell_l2g: List[List[int]] = []

    def vkey(p):
        return (round(p[0], key_decimals), round(p[1], key_decimals), round(p[2], key_decimals))

    for c in cells:
        l2g = []
        for v in c['vertices']:
            k = vkey(v)
            gi = gkey.get(k)
            if gi is None:
                gi = len(gpos)
                gkey[k] = gi
                gpos.append(v)
            l2g.append(gi)
        cell_l2g.append(l2g)

    vhandles = [tfv.Vertex.create(tf.FVector3(*p)) for p in gpos]

    # --- dedup faces; collect each body's surfaces (interior faces shared) ---
    face_surf = {}                       # (min,max) cell-pair -> shared SurfaceHandle
    cell_surfs: List[list] = [[] for _ in cells]
    n_internal = n_wall = 0
    for ci, c in enumerate(cells):
        l2g = cell_l2g[ci]
        for f in c['faces']:
            a = f['adjacent_cell']
            verts = [vhandles[l2g[lv]] for lv in f['vertices']]
            if a >= 0:                                   # interior: shared face
                key = (min(ci, a), max(ci, a))
                surf = face_surf.get(key)
                if surf is None:
                    surf = stype(vertices=verts)
                    face_surf[key] = surf
                    n_internal += 1
                cell_surfs[ci].append(surf)
            else:                                        # box wall: free outer face
                cell_surfs[ci].append(stype(vertices=verts))
                n_wall += 1

    bodies = [btype(cell_surfs[ci]) for ci in range(len(cells))]
    stats = dict(n_internal=n_internal, n_wall=n_wall, n_vertices=len(gpos))
    return bodies, cells, stats


# ======================================================================================
# Periodic (minimum-image) packing -- the P4 generator (PORTING_NOTES §6g, periodic mesh)
# ======================================================================================
#
# `build_voronoi_cluster` above makes a FINITE block: box-wall faces are 1-body free
# surfaces and the cluster has an outer boundary. For the paper's / 3DVertVor's regime we
# need a SPACE-FILLING periodic foam with NO free surface -- every face interior (b1/b2
# both set) and body adjacency that WRAPS across the box faces -- to run with the engine's
# `mesh.periodic_geometry=True` minimum-image geometry (PORTING_NOTES §6g).
#
# Two things make the periodic build different from the finite one:
#   1. Vertex dedup must CANONICALIZE into the box. A face shared across a box wall is
#      reported by its two cells with vertex coordinates that differ by a box vector (one
#      cell sees x≈lo⁺, the other x≈hi⁻ → raw image x≈lo-ε). Raw-coordinate dedup (the
#      finite path's `vkey`) would NOT merge those into one TF Vertex, so the wrap face
#      would never become a single shared Surface. We min-image each vertex into [lo,hi)
#      before keying (and create the TF vertex at that wrapped position).
#   2. Faces are deduped by their VERTEX SET (frozenset of global vertex ids), NOT by a
#      (min,max) cell-pair key. The cell-pair key breaks if a cell touches the same
#      neighbour twice -- directly AND through the wrap -- which is geometrically possible
#      in a small box (a cell adjacent to its own periodic image). A Voronoi face is the
#      unique set of vertices equidistant to two seeds, so the vertex set is a perfect,
#      multiplicity-robust face key. (We additionally veto i==j self-faces, which require
#      n≥3 per axis to avoid; see the build below.)
#
# Implementation route: GHOST-TILING (route B in the kickoff). pyvoro-mmalahe DOES accept
# `periodic=[True,True,True]`, but on this build it returns garbage for a real lattice
# (box-spanning overlapping cells + spurious wall faces) at most `dispersion` values --
# only dispersions that don't align with the lattice happen to work, which is too fragile
# to rely on. So instead we replicate the seeds across a 3×3×3 supercell, run the existing
# NON-periodic Voronoi, keep only the central-image cells (which are then fully surrounded
# by ghosts -> zero wall faces), and remap each central face's `adjacent_cell` back to its
# central index. This reuses the validated finite code path and is robust.
#
# `dispersion` gotcha (verified 2026-06-01): voro++ block-size aliasing makes pyvoro return
# overlapping cells when `dispersion` divides the (highly regular) lattice spacing. We force
# a SINGLE brute-force block by defaulting dispersion to the full enlarged-box edge -- exact
# Voronoi, no block-boundary degeneracy. This is O(N²) in the 27·N ghost seeds; fine for a
# one-time mesh build at the sizes we use (N≈O(10²)). For very large N, ghost only the
# near-boundary seeds instead (left as a future optimization; correctness first).


def periodic_bcc_seeds(n_per_axis: int, box: Box) -> List[List[float]]:
    """Seeds of a PERIODIC BCC lattice filling `box` -- 2·n³ points, all wrapped into
    [lo,hi), none dropped. Under periodicity the Voronoi cells are identical Kelvin
    tetrakaidecahedra (Okuda Fig. 7a, the paper's N≈1728 ideal-foam regime).

    Differs from `bcc_seeds` (which is for the FINITE block and DROPS the body-centred
    points that fall on/over the upper wall): here the body-centred sublattice is wrapped
    back into the box (`%L`) rather than clipped, so the lattice is seamless across every
    face -- the correct input for `build_periodic_voronoi`.
    """
    lo = np.array([b[0] for b in box], dtype=float)
    hi = np.array([b[1] for b in box], dtype=float)
    L = hi - lo
    a = L / n_per_axis                                  # cubic sublattice edge per axis
    pts = []
    for i in range(n_per_axis):
        for j in range(n_per_axis):
            for k in range(n_per_axis):
                corner = lo + np.array([i, j, k]) * a               # corner sublattice
                body = lo + (np.array([i, j, k]) + 0.5) * a         # body-centred (+½a)
                pts.append(corner.tolist())
                pts.append((lo + (body - lo) % L).tolist())         # wrap, do NOT drop
    return pts


def build_periodic_voronoi(points, box: Box, btype, stype,
                           dispersion: Optional[float] = None,
                           key_decimals: int = 6):
    """Build a SPACE-FILLING periodic TF vertex mesh (no free surface) from `points`,
    via voro++'s NATIVE periodic mode + minimum-image vertex dedup. Intended to be used with
    the engine's `mesh.periodic_geometry=True` so straddling cells measure by their short
    image (PORTING_NOTES §6g).

    All bodies are created as `btype`; `stype` is the surface type for every (interior)
    face. Returns (bodies, seeds, stats) where `bodies[i]` corresponds to `points[i]`,
    `seeds` is `np.asarray(points)` (so callers can map a body back to its raw seed for
    wrap detection), and `stats` has n_surfaces / n_vertices / n_wrap_faces /
    n_self_faces counts. Raises if any cell is adjacent to its own periodic image
    (n<3 per axis); use a larger box.

    CRITICAL — the foam box MUST be the universe box `[[0,dim_x],[0,dim_y],[0,dim_z]]`.
    The engine's minimum-image (`mesh.periodic_geometry`) wraps displacements at
    `Universe::dim()` (tf_mesh_metrics.cpp `meshPeriodicBox` -> `Universe::dim()`), so a
    cell straddling a foam-box wall is only measured by its short image when that wall IS
    a universe wall. Building the foam in a SUB-box of the universe silently yields
    box-spanning volumes/centroids for the straddling cells (the wall isn't a periodic
    wall to the engine). We assert the match here to prevent that footgun.
    """
    seeds = np.asarray(points, dtype=float)
    n = len(seeds)
    lo = np.array([b[0] for b in box], dtype=float)
    hi = np.array([b[1] for b in box], dtype=float)
    L = hi - lo

    # The foam's periodic box must equal the universe box (see the CRITICAL note above):
    # minimumImage is over Universe::dim(), so the foam period must be the universe period
    # and the origin must be 0 (the integrator's PERIODIC_FULL BC wraps p->x into [0,dim)).
    dim = np.array([tf.Universe.dim[0], tf.Universe.dim[1], tf.Universe.dim[2]], dtype=float)
    if not (np.allclose(lo, 0.0, atol=1e-6) and np.allclose(L, dim, atol=1e-4)):
        raise ValueError(
            f"periodic foam box {list(map(list, box))} must equal the universe box "
            f"[[0,{dim[0]}],[0,{dim[1]}],[0,{dim[2]}]]: the engine min-images at "
            "Universe::dim(), so a sub-box gives box-spanning straddling cells.")

    # --- native periodic Voronoi: NO 3×3×3 ghost tiling (voro++ wraps internally) -----------
    # The prior build ghost-tiled (27× the points) AND ran voro++ in a SINGLE brute-force block
    # (dispersion = whole box) -> O(N^2), profiled at 96% of the build (~28 s at n=8). voro++'s
    # NATIVE periodic mode computes the SAME diagram (cell adjacencies verified bit-for-bit
    # identical) directly on the N seeds with proper spatial blocking -> ~O(N), ~500× faster
    # (n=8: 27 s -> 0.05 s; n=10 / 2000 cells: 0.09 s). `dispersion` (mean seed spacing) only
    # sizes the block grid -- its exact value affects speed, not the result.
    box_l = [[float(lo[d]), float(hi[d])] for d in range(3)]
    if dispersion is None:
        dispersion = float((np.prod(L) / max(n, 1)) ** (1.0 / 3.0))
    cells = pyvoro.compute_voronoi([list(p) for p in seeds], box_l, dispersion,
                                   periodic=[True, True, True])

    # --- min-image vertex canonicalization + global dedup ------------------------------
    def vkey(v):
        # fold into [lo,hi); round; refold a value that rounded up to exactly hi back to lo
        # so the two images of a boundary vertex (x≈hi⁻ and x≈lo⁻→hi⁻) collapse to one key.
        u = (np.asarray(v, dtype=float) - lo) % L
        key = []
        for x, Lx in zip(u, L):
            r = round(float(x), key_decimals)
            if r >= round(float(Lx), key_decimals):
                r -= float(Lx)
            key.append(r)
        return tuple(key)

    gpos: List[np.ndarray] = []
    gkey = {}
    cell_l2g: List[List[int]] = []        # per cell: local vertex idx -> global idx
    for i in range(n):
        l2g = []
        for v in cells[i]['vertices']:
            k = vkey(v)
            gi = gkey.get(k)
            if gi is None:
                gi = len(gpos)
                gkey[k] = gi
                gpos.append(lo + (np.asarray(v, dtype=float) - lo) % L)   # wrapped into box
            l2g.append(gi)
        cell_l2g.append(l2g)

    # BATCH-create all vertices in ONE call. Per-element `Vertex.create` goes through
    # Mesh::allocateVertex, which exhausts a 100-slot free-list every TFMESHINV_INCR=100 verts
    # and then calls Mesh::incrementVertices -- a realloc that COPIES the vertex vector and walks
    # every surface + vertex to fix up the raw pointers (O(N) each). N/100 such reallocs => O(N^2)
    # (profiled: 12.7 s / 35% of the n=20 build, exponent 2.08). The batch overload
    # `Vertex.create(list[FVector3])` -> Mesh::allocateVertices -> a SINGLE incrementVertices(N)
    # -> ~O(N). Bit-identical to the loop: allocateVertices pulls free ids 0,1,2,... in input
    # order, so vhandles[i] has id i and position gpos[i] exactly as before.
    vhandles = list(tfv.Vertex.create([tf.FVector3(*map(float, p)) for p in gpos]))

    # --- dedup faces by vertex SET; build shared surfaces + per-body surface lists -----
    # periodic mode: f['adjacent_cell'] is the neighbour SEED index directly (no ghost remap),
    # and every face is interior (no box wall) so it is always >= 0.
    face_surf = {}                              # frozenset(global vids) -> SurfaceHandle
    face_pair = {}                              # frozenset(global vids) -> (i, j) seed pair
    cell_surfs: List[list] = [[] for _ in range(n)]
    n_self_faces = 0
    for i in range(n):
        l2g = cell_l2g[i]
        for f in cells[i]['faces']:
            j = f['adjacent_cell']               # neighbour seed index (periodic; no remap)
            if j == i:
                n_self_faces += 1                # cell adjacent to its own image (n<3 box)
            gids = [l2g[lv] for lv in f['vertices']]
            key = frozenset(gids)
            surf = face_surf.get(key)
            if surf is None:
                surf = stype(vertices=[vhandles[g] for g in gids])
                face_surf[key] = surf
                face_pair[key] = (i, j)
            cell_surfs[i].append(surf)
    if n_self_faces:
        raise ValueError(
            f"{n_self_faces} self-adjacent faces (a cell shares a face with its own "
            "periodic image): the box is too small (need n≥3 per axis).")

    bodies = [btype(cell_surfs[i]) for i in range(n)]

    # wrap-face count: a shared face whose two seeds are >½ box apart on some axis (raw)
    # -> the adjacency genuinely crosses a periodic boundary (min-image, not box-spanning).
    n_wrap_faces = sum(1 for (i, j) in face_pair.values()
                       if np.any(np.abs(seeds[i] - seeds[j]) > L / 2))
    stats = dict(n_surfaces=len(face_surf), n_vertices=len(gpos),
                 n_self_faces=n_self_faces, n_wrap_faces=n_wrap_faces)
    return bodies, seeds, stats
