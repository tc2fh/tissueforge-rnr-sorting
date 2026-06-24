"""Gate A of the GPU port: flatten TissueForge's pointer-graph vertex mesh into an
index-based CSR / Structure-of-Arrays representation, and prove it round-trips the TF
topology exactly.

This is Stage 0 of docs/2026-06-24_gpu-3d-vertex-model-exploration.md -- the foundational
GPU-friendly data layout everything else rests on. TF stores connectivity as a heap
pointer-graph (`Vertex` holds `vector<Surface*>`, `Surface` holds `vector<Vertex*>` +
`b1,b2`, `Body` holds `vector<Surface*>`) which is GPU-hostile: a kernel cannot chase
host pointers. Here we flatten it, once, into compact integer-indexed arrays -- the 3D
generalisation of cellGPU's SoA, but *ragged* (CSR offsets + flat index arrays) because
3D meshes have no fixed valence (interior vertices are 4-cell, faces are variable-size
polygons, cells are variable-size polyhedra). id<->index maps let us push results back.

No GPU is required to BUILD the CSR (extraction is host-side numpy); `to_warp()` uploads
it to the device for the later gates. The round-trip verifier is the Gate-A test: read TF
-> CSR -> read back == original, plus internal transpose-consistency of the CSR itself.

CSR conventions
---------------
Entities are compact-indexed 0..n-1 by ascending TF id (deterministic). For a ragged map
A->B we store `a2b_off` (len nA+1, prefix sums) and `a2b_idx` (len = total incidences),
so the B-indices for entity `a` are `a2b_idx[a2b_off[a] : a2b_off[a+1]]`.

    vert_pos   (nv,3) f64   vertex positions
    s2v_off/idx            surface -> ORDERED ring of vertex indices (winding preserved)
    s2b        (ns,2) i32   surface -> up to 2 incident body indices (getBodies() order;
                            -1 pads a boundary/free face with a single body)
    v2s_off/idx           vertex  -> incident surface indices
    b2s_off/idx           body    -> bounding surface indices
    *_alive   (n,) bool    liveness (all True at extraction; Gate B/D toggle these)
    *_id      (n,) i64     index -> TF id  (round-trip handle back to the pointer graph)
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..reconnect import _np  # FVector3 -> np.array([p[0],p[1],p[2]])


# --------------------------------------------------------------------------------------
# the CSR container
# --------------------------------------------------------------------------------------
@dataclass
class CSRMesh:
    """Index-based CSR/SoA snapshot of a TF vertex mesh (host-side numpy)."""
    nv: int
    ns: int
    nb: int
    # positions + index->id maps
    vert_pos: np.ndarray   # (nv,3) f64
    vert_id: np.ndarray    # (nv,)  i64
    surf_id: np.ndarray    # (ns,)  i64
    body_id: np.ndarray    # (nb,)  i64
    # connectivity (CSR)
    s2v_off: np.ndarray    # (ns+1,) i32
    s2v_idx: np.ndarray    # (Esv,)  i32  (ordered ring)
    s2b: np.ndarray        # (ns,2)  i32  (-1 pad)
    v2s_off: np.ndarray    # (nv+1,) i32
    v2s_idx: np.ndarray    # (Evs,)  i32
    b2s_off: np.ndarray    # (nb+1,) i32
    b2s_idx: np.ndarray    # (Ebs,)  i32
    # liveness
    vert_alive: np.ndarray  # (nv,) bool
    surf_alive: np.ndarray  # (ns,) bool
    body_alive: np.ndarray  # (nb,) bool

    # --- convenience accessors (host-side; the GPU kernels index the raw arrays) ------
    def surf_verts(self, s: int) -> np.ndarray:
        """Ordered vertex indices of surface s."""
        return self.s2v_idx[self.s2v_off[s]:self.s2v_off[s + 1]]

    def vert_surfs(self, v: int) -> np.ndarray:
        return self.v2s_idx[self.v2s_off[v]:self.v2s_off[v + 1]]

    def body_surfs(self, b: int) -> np.ndarray:
        return self.b2s_idx[self.b2s_off[b]:self.b2s_off[b + 1]]

    def to_warp(self, device=None) -> Dict[str, object]:
        """Upload the SoA to a Warp device (default: first CUDA device, else CPU).

        Returns a dict name -> wp.array. Positions go up as an (nv,3) f64 array; index
        arrays as int32; liveness as int32 (Warp has no bool array dtype we rely on).
        This is the host->device handoff the Gate-B kernels consume.
        """
        import warp as wp
        if device is None:
            cuda = [d for d in wp.get_devices() if d.is_cuda]
            device = cuda[0] if cuda else "cpu"
        f = lambda a, dt: wp.array(np.ascontiguousarray(a), dtype=dt, device=device)
        return dict(
            device=device,
            vert_pos=f(self.vert_pos, wp.vec3d),
            s2v_off=f(self.s2v_off, wp.int32), s2v_idx=f(self.s2v_idx, wp.int32),
            s2b=f(self.s2b, wp.int32),
            v2s_off=f(self.v2s_off, wp.int32), v2s_idx=f(self.v2s_idx, wp.int32),
            b2s_off=f(self.b2s_off, wp.int32), b2s_idx=f(self.b2s_idx, wp.int32),
            vert_alive=f(self.vert_alive.astype(np.int32), wp.int32),
            surf_alive=f(self.surf_alive.astype(np.int32), wp.int32),
            body_alive=f(self.body_alive.astype(np.int32), wp.int32),
        )


# --------------------------------------------------------------------------------------
# extraction:  TF pointer-graph  ->  CSR
# --------------------------------------------------------------------------------------
def _csr_from_rows(rows: List[List[int]], n: int) -> Tuple[np.ndarray, np.ndarray]:
    off = np.zeros(n + 1, np.int32)
    for i, r in enumerate(rows):
        off[i + 1] = off[i] + len(r)
    total = int(off[-1])
    idx = np.empty(total, np.int32)
    k = 0
    for r in rows:
        for x in r:
            idx[k] = x
            k += 1
    return off, idx


def _enumerate(bodies) -> Tuple[Dict[int, object], Dict[int, object], Dict[int, object]]:
    """Collect the unique vertices/surfaces/bodies reachable from a body list, by id.

    Walks bodies -> their surfaces & vertices, and each surface -> its ring vertices
    (defensive: guarantees every vertex named in a ring is enumerated). Mirrors the
    test-helper convention (helpers.all_vertices/surface_ids) of scoping to a body list,
    never the global *.instances which would mix coexisting meshes in the shared universe.
    """
    vset: Dict[int, object] = {}
    sset: Dict[int, object] = {}
    bset: Dict[int, object] = {}
    for b in bodies:
        bset.setdefault(b.id, b)
        for v in b.getVertices():
            vset.setdefault(v.id, v)
        for s in b.getSurfaces():
            sset.setdefault(s.id, s)
    for s in list(sset.values()):
        for v in s.vertices:
            vset.setdefault(v.id, v)
    return vset, sset, bset


def id_maps(bodies) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """The TF id -> compact CSR index maps for `bodies` (vertex, surface, body).

    Identical enumeration to extract_csr (same `_enumerate` + ascending-id sort), so the
    indices these return line up 1:1 with a CSRMesh / PaddedMesh built from the same
    bodies. Used by reconnect_csr.iconfig_to_indices to translate a TF-handle IConfig
    into the index world the GPU surgery operates in.
    """
    vset, sset, bset = _enumerate(bodies)
    vi = {vid: i for i, vid in enumerate(sorted(vset))}
    si = {sid: i for i, sid in enumerate(sorted(sset))}
    bi = {bid: i for i, bid in enumerate(sorted(bset))}
    return vi, si, bi


def extract_csr(bodies) -> CSRMesh:
    """Flatten the TF mesh reachable from `bodies` into a CSRMesh (host numpy)."""
    vset, sset, bset = _enumerate(bodies)
    vids = sorted(vset)
    sids = sorted(sset)
    bids = sorted(bset)
    vi = {vid: i for i, vid in enumerate(vids)}
    si = {sid: i for i, sid in enumerate(sids)}
    bi = {bid: i for i, bid in enumerate(bids)}
    nv, ns, nb = len(vids), len(sids), len(bids)

    vert_pos = np.empty((nv, 3), np.float64)
    for vid in vids:
        vert_pos[vi[vid]] = _np(vset[vid].position)

    # surface -> ordered ring (CSR) + surface -> bodies (pad to 2)
    s2v_rows: List[List[int]] = []
    s2b = np.full((ns, 2), -1, np.int32)
    for sid in sids:
        s = sset[sid]
        s2v_rows.append([vi[v.id] for v in s.vertices])
        bs = [b for b in s.getBodies() if b.id in bi]
        for k, b in enumerate(bs[:2]):
            s2b[si[sid], k] = bi[b.id]
    s2v_off, s2v_idx = _csr_from_rows(s2v_rows, ns)

    # vertex -> incident surfaces (CSR), body -> bounding surfaces (CSR)
    v2s_rows = [[si[s.id] for s in vset[vid].getSurfaces() if s.id in si] for vid in vids]
    v2s_off, v2s_idx = _csr_from_rows(v2s_rows, nv)
    b2s_rows = [[si[s.id] for s in bset[bid].getSurfaces() if s.id in si] for bid in bids]
    b2s_off, b2s_idx = _csr_from_rows(b2s_rows, nb)

    return CSRMesh(
        nv=nv, ns=ns, nb=nb,
        vert_pos=vert_pos,
        vert_id=np.array(vids, np.int64),
        surf_id=np.array(sids, np.int64),
        body_id=np.array(bids, np.int64),
        s2v_off=s2v_off, s2v_idx=s2v_idx, s2b=s2b,
        v2s_off=v2s_off, v2s_idx=v2s_idx,
        b2s_off=b2s_off, b2s_idx=b2s_idx,
        vert_alive=np.ones(nv, bool),
        surf_alive=np.ones(ns, bool),
        body_alive=np.ones(nb, bool),
    )


# --------------------------------------------------------------------------------------
# the Gate-A round-trip:  read TF  ==  CSR read back, + CSR internal consistency
# --------------------------------------------------------------------------------------
def _read_tf_topology(bodies) -> dict:
    """Ground-truth id-keyed topology read straight from the TF pointer graph."""
    vset, sset, bset = _enumerate(bodies)
    sids = set(sset)
    bids = set(bset)
    return dict(
        pos={vid: _np(v.position) for vid, v in vset.items()},
        ring={sid: [v.id for v in s.vertices] for sid, s in sset.items()},
        sbody={sid: frozenset(b.id for b in s.getBodies() if b.id in bids)
               for sid, s in sset.items()},
        vsurf={vid: frozenset(s.id for s in v.getSurfaces() if s.id in sids)
               for vid, v in vset.items()},
        bsurf={bid: frozenset(s.id for s in b.getSurfaces() if s.id in sids)
               for bid, b in bset.items()},
    )


def _csr_to_topology(m: CSRMesh) -> dict:
    """Reconstruct the same id-keyed topology from the CSR arrays (index -> id)."""
    vid = m.vert_id
    sid = m.surf_id
    bid = m.body_id
    pos = {int(vid[i]): m.vert_pos[i].copy() for i in range(m.nv)}
    ring = {int(sid[s]): [int(vid[i]) for i in m.surf_verts(s)] for s in range(m.ns)}
    sbody = {int(sid[s]): frozenset(int(bid[b]) for b in m.s2b[s] if b >= 0)
             for s in range(m.ns)}
    vsurf = {int(vid[v]): frozenset(int(sid[s]) for s in m.vert_surfs(v))
             for v in range(m.nv)}
    bsurf = {int(bid[b]): frozenset(int(sid[s]) for s in m.body_surfs(b))
             for b in range(m.nb)}
    return dict(pos=pos, ring=ring, sbody=sbody, vsurf=vsurf, bsurf=bsurf)


def check_internal_consistency(m: CSRMesh) -> List[str]:
    """CSR is a valid mesh rep on its own terms: offsets monotonic, indices in range,
    and the inverse maps are exact transposes of the forward maps."""
    p: List[str] = []
    # offsets monotonic non-decreasing, final == idx length
    for name, off, idx, n in [("s2v", m.s2v_off, m.s2v_idx, m.ns),
                              ("v2s", m.v2s_off, m.v2s_idx, m.nv),
                              ("b2s", m.b2s_off, m.b2s_idx, m.nb)]:
        if off.shape[0] != n + 1:
            p.append(f"{name}_off length {off.shape[0]} != n+1 ({n+1})")
        if np.any(np.diff(off) < 0):
            p.append(f"{name}_off not monotonic")
        if int(off[-1]) != idx.shape[0]:
            p.append(f"{name}_off[-1]={int(off[-1])} != len({name}_idx)={idx.shape[0]}")
    if m.s2v_idx.size and (m.s2v_idx.min() < 0 or m.s2v_idx.max() >= m.nv):
        p.append("s2v_idx out of vertex range")
    if m.v2s_idx.size and (m.v2s_idx.min() < 0 or m.v2s_idx.max() >= m.ns):
        p.append("v2s_idx out of surface range")
    if m.b2s_idx.size and (m.b2s_idx.min() < 0 or m.b2s_idx.max() >= m.ns):
        p.append("b2s_idx out of surface range")
    if m.s2b.size and (m.s2b.max() >= m.nb or m.s2b.min() < -1):
        p.append("s2b out of body range")

    # transpose:  s in v2s[v]  <=>  v in s2v[s]
    sv = {(s, int(v)) for s in range(m.ns) for v in m.surf_verts(s)}
    vs = {(int(s), v) for v in range(m.nv) for s in m.vert_surfs(v)}
    if sv != vs:
        p.append(f"s2v/v2s not transpose (sv-only={len(sv-vs)}, vs-only={len(vs-sv)})")
    # transpose:  b in s2b[s]  <=>  s in b2s[b]
    sb = {(s, int(b)) for s in range(m.ns) for b in m.s2b[s] if b >= 0}
    bs = {(int(s), b) for b in range(m.nb) for s in m.body_surfs(b)}
    if sb != bs:
        p.append(f"s2b/b2s not transpose (sb-only={len(sb-bs)}, bs-only={len(bs-sb)})")
    return p


def verify_roundtrip(m: CSRMesh, bodies, pos_atol: float = 0.0) -> dict:
    """Gate A: CSR must round-trip the TF mesh exactly + be internally consistent.

    Returns a report dict {ok, problems, counts}. `problems` is empty iff every check
    passed. Positions are compared exactly by default (we copied the same floats);
    pass pos_atol>0 to allow slack.
    """
    problems: List[str] = []
    tf = _read_tf_topology(bodies)
    cs = _csr_to_topology(m)

    if set(tf["pos"]) != set(cs["pos"]):
        problems.append("vertex id set mismatch")
    if set(tf["ring"]) != set(cs["ring"]):
        problems.append("surface id set mismatch")
    if set(tf["bsurf"]) != set(cs["bsurf"]):
        problems.append("body id set mismatch")

    for vid, p_tf in tf["pos"].items():
        p_cs = cs["pos"].get(vid)
        if p_cs is None or not np.allclose(p_tf, p_cs, atol=pos_atol, rtol=0.0):
            problems.append(f"position mismatch at vertex {vid}")
            break

    # surface rings: exact ordered equality (winding preserved 1:1)
    for sid, r_tf in tf["ring"].items():
        if cs["ring"].get(sid) != r_tf:
            problems.append(f"ring mismatch at surface {sid}: tf={r_tf} csr={cs['ring'].get(sid)}")
            break
    # surface->bodies, vertex->surfaces, body->surfaces: set equality
    for key, label in [("sbody", "surface->bodies"),
                       ("vsurf", "vertex->surfaces"),
                       ("bsurf", "body->surfaces")]:
        for oid, set_tf in tf[key].items():
            if cs[key].get(oid) != set_tf:
                problems.append(f"{label} mismatch at id {oid}")
                break

    problems += check_internal_consistency(m)

    return dict(
        ok=(len(problems) == 0),
        problems=problems,
        counts=dict(nv=m.nv, ns=m.ns, nb=m.nb,
                    n_incid_sv=int(m.s2v_off[-1]),
                    n_incid_vs=int(m.v2s_off[-1]),
                    n_incid_bs=int(m.b2s_off[-1])),
    )


def summary(m: CSRMesh) -> str:
    valence = np.diff(m.v2s_off)           # surfaces per vertex
    poly = np.diff(m.s2v_off)              # vertices per surface
    faces = np.diff(m.b2s_off)             # surfaces per body
    interior = int(np.sum((m.s2b >= 0).all(axis=1)))
    return (f"CSRMesh: {m.nv} verts, {m.ns} surfs ({interior} interior/2-body), {m.nb} bodies\n"
            f"  vertex valence (surfs/vert): min={valence.min()} max={valence.max()} mean={valence.mean():.2f}\n"
            f"  face size (verts/surf):      min={poly.min()} max={poly.max()} mean={poly.mean():.2f}\n"
            f"  cell size (surfs/body):      min={faces.min()} max={faces.max()} mean={faces.mean():.2f}")


# --------------------------------------------------------------------------------------
# body-anchored, slot-invariant fingerprint  (the Gate-B round-trip gate)
# --------------------------------------------------------------------------------------
def fingerprint(m: CSRMesh) -> tuple:
    """A topology fingerprint keyed on BODY indices -- invariant to vertex/surface slot
    relabelling, so it survives the count-changing I<->H surgery + to_csr() compaction.

    Bodies are never created/destroyed by I<->H and keep their identity/index across a
    round-trip (extract_csr and PaddedMesh.to_csr both index bodies 0..nb-1 in the same
    ascending-id order), so they are the stable anchor. Vertices and surfaces, by
    contrast, are freed/reallocated to fresh slots, so array equality is meaningless
    post-surgery -- but the body-set each one touches is preserved iff the topology is.

      vertex fp = sorted tuple of incident body indices         (via v2s -> s2b)
      face   fp = (sorted body indices, sorted multiset of its ring verts' fps)
      mesh   fp = (sorted multiset of vertex fps, sorted multiset of face fps)

    Equal fingerprints  <=>  same body-anchored connectivity  (the round-trip invariant).
    All-live assumed (true for extract_csr and the compacted to_csr output).
    """
    vkey = []
    for v in range(m.nv):
        bset = set()
        for s in m.vert_surfs(v):
            for b in m.s2b[int(s)]:
                if b >= 0:
                    bset.add(int(b))
        vkey.append(tuple(sorted(bset)))
    ffp = []
    for s in range(m.ns):
        sb = tuple(sorted(int(b) for b in m.s2b[s] if b >= 0))
        vs = tuple(sorted(vkey[int(v)] for v in m.surf_verts(s)))
        ffp.append((sb, vs))
    return (tuple(sorted(vkey)), tuple(sorted(ffp)))
