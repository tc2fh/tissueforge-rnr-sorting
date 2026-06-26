"""Gate C, on-GPU detection (docs/2026-06-24_gpu-3d-vertex-model-exploration.md, the post-
scheduler "next" step): the Condition-2 TRIGGER SCANS of topology_csr.find_short_edges_csr /
find_small_triangles_csr, ported to parallel Warp kernels.

WHY: each round of reconnect_sweep_*_warp currently does PaddedMesh.from_warp(g) + a host
PYTHON scan over every vertex / surface (O(mesh)) to find reconnection sites. That Python loop
is the round's scaling bottleneck. Here the O(mesh) scan runs one-thread-per-element on the
device; only the (small, O(candidates)) neighbourhood gather + veto remain on the host -- a
"hybrid" detect that keeps the expensive part on the GPU.

The split mirrors the host: a cheap STRUCTURAL+LENGTH trigger (this module) enumerates candidate
primitives (short edges / small triangles), then the gather (topology_csr.i_neighbourhood_csr /
h_neighbourhood_csr) validates each into a surgery-ready config. The kernels prune with cheap
checks the host gather would also reject (e.g. a triangle must border exactly 2 bodies), so the
trigger set is matched EXACTLY by a host reference here (small_triangle_trigger_host /
short_edge_trigger_host) and the hybrid output equals the pure-host detector by construction.

Compaction is an atomic-append (nondeterministic order); the hybrid wrappers SORT the small
candidate list by canonical key (surface index / (v10,v11)) so the downstream lowest-id-wins
reservation stays deterministic + host-matchable, exactly as before.
"""
from typing import List, Optional, Set, Tuple

import numpy as np

import warp as wp

from .device_mesh import PaddedMesh
from .reconnect_csr import HCfgIdx, ICfgIdx
from .reconnect_warp import d_ring_pos
from .topology_csr import (connected_vertices, edge_length, h_neighbourhood_csr,
                           i_neighbourhood_csr, vert_bodies)

wp.init()


# ======================================================================================
# H-side: small-triangle trigger scan (one thread per surface, single emit)
# ======================================================================================
@wp.kernel
def scan_small_triangles_kernel(
        vert_pos: wp.array(dtype=wp.vec3d), surf_alive: wp.array(dtype=wp.int32),
        s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
        s2b: wp.array2d(dtype=wp.int32), threshold: wp.float64,
        out_tris: wp.array(dtype=wp.int32), out_count: wp.array(dtype=wp.int32)):
    """Flag triangular faces (alive, exactly 3 verts, bordering exactly 2 bodies) whose MAX
    edge < threshold -- Condition 2 for the reverse direction triggers on the max edge (NOT
    the min; Honda's wrong 'condition H'). Append the surface index via an atomic counter."""
    s = wp.tid()
    if surf_alive[s] == 0:
        return
    if s2v_len[s] != 3:
        return
    nb = int(0)                          # a reverse site is an interior face: 2 bordering cells
    if s2b[s, 0] >= 0:
        nb += 1
    if s2b[s, 1] >= 0:
        nb += 1
    if nb != 2:
        return
    a = s2v[s, 0]
    b = s2v[s, 1]
    c = s2v[s, 2]
    e0 = wp.length(vert_pos[a] - vert_pos[b])
    e1 = wp.length(vert_pos[b] - vert_pos[c])
    e2 = wp.length(vert_pos[c] - vert_pos[a])
    mx = wp.max(e0, wp.max(e1, e2))
    if mx < threshold:
        idx = wp.atomic_add(out_count, 0, 1)
        out_tris[idx] = s


def find_small_triangles_warp(g: dict, threshold: float) -> np.ndarray:
    """GPU parallel Condition-2 trigger scan for [H] sites. Returns the (sorted) compact array
    of candidate triangular-face indices, read straight from the device SoA `g` -- the O(mesh)
    work that find_small_triangles_csr did in Python, now one thread per surface on the GPU."""
    dev = g["device"]
    n_s = int(g["n_used"].numpy()[1])               # tiny readback (1 int, not O(mesh))
    if n_s == 0:
        return np.zeros(0, np.int32)
    out = wp.zeros(n_s, dtype=wp.int32, device=dev)
    count = wp.zeros(1, dtype=wp.int32, device=dev)
    wp.launch(scan_small_triangles_kernel, dim=n_s, device=dev, inputs=[
        g["vert_pos"], g["surf_alive"], g["s2v"], g["s2v_len"], g["s2b"],
        wp.float64(threshold), out, count])
    wp.synchronize_device(dev)
    k = int(count.numpy()[0])
    return np.sort(out.numpy()[:k])                 # canonical (surface-ascending) order


def small_triangle_trigger_host(pm: PaddedMesh, threshold: float) -> Set[int]:
    """Host reference for scan_small_triangles_kernel: the exact same cheap trigger (alive,
    triangular, 2-bordering-cells, max edge < threshold), as a set of surface indices. The
    GPU scan must reproduce this set; the h_neighbourhood gather then validates each."""
    out: Set[int] = set()
    for s in range(pm.n_s_used):
        if not pm.surf_alive[s] or int(pm.s2v_len[s]) != 3:
            continue
        if int((pm.s2b[s] >= 0).sum()) != 2:
            continue
        a, b, c = (int(x) for x in pm.s2v[s, :3])
        mx = max(edge_length(pm, a, b), edge_length(pm, b, c), edge_length(pm, c, a))
        if mx < threshold:
            out.add(s)
    return out


def detect_small_triangles_hybrid(g: dict, pm: PaddedMesh, threshold: float
                                  ) -> List[Tuple[int, HCfgIdx]]:
    """GPU trigger scan + host h_neighbourhood gather on just the candidates -- the same
    (triangle_idx, HCfgIdx) list as topology_csr.find_small_triangles_csr, but the O(mesh)
    scan runs on the GPU. `pm` is the host mirror (PaddedMesh.from_warp(g)) used only for the
    O(candidates) gather. Surface-ascending order (deterministic reservation downstream)."""
    out = []
    for s in find_small_triangles_warp(g, threshold):
        cfg = h_neighbourhood_csr(pm, int(s))
        if cfg is not None:
            out.append((int(s), cfg))
    return out


# ======================================================================================
# I-side: short-edge trigger scan (one thread per vertex; per-thread dedup, no scratch)
# ======================================================================================
@wp.func
def d_vert_body_count(v2s: wp.array2d(dtype=wp.int32), v2s_len: wp.array(dtype=wp.int32),
                      s2b: wp.array2d(dtype=wp.int32), v: int) -> int:
    """Number of DISTINCT bodies incident to vertex v (host vert_bodies, as a count). O(k^2)
    over v's <=8 incident surfaces x 2 bodies -- a body counts once, at its first occurrence."""
    L = v2s_len[v]
    nb = wp.int32(0)
    for i in range(L):
        s = v2s[v, i]
        for slot in range(2):
            b = s2b[s, slot]
            if b >= 0:
                first = wp.int32(1)
                for i2 in range(L):
                    for slot2 in range(2):
                        if (i2 < i) or (i2 == i and slot2 < slot):
                            if s2b[v2s[v, i2], slot2] == b:
                                first = wp.int32(0)
                if first == 1:
                    nb += 1
    return nb


@wp.func
def d_ring_neighbor(s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
                    s: int, v: int, side: int) -> int:
    """v's ring neighbour in surface s: side==0 -> previous, side==1 -> next (cyclic)."""
    Ls = s2v_len[s]
    pi = d_ring_pos(s2v, s2v_len, s, v)
    if side == 0:
        return s2v[s, (pi - 1 + Ls) % Ls]
    return s2v[s, (pi + 1) % Ls]


@wp.kernel
def scan_short_edges_kernel(
        vert_pos: wp.array(dtype=wp.vec3d), vert_alive: wp.array(dtype=wp.int32),
        surf_alive: wp.array(dtype=wp.int32),
        s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
        threshold: wp.float64,
        out_v10: wp.array(dtype=wp.int32), out_v11: wp.array(dtype=wp.int32),
        out_count: wp.array(dtype=wp.int32)):
    """One thread per SURFACE. Emit each implicit edge (consecutive ring pair) with len<threshold,
    canonicalized smaller-endpoint-first -- the cheap LENGTH half of the host find_short_edges
    trigger. Implicit edges ARE consecutive ring pairs, so this finds the identical edge set the
    per-vertex walk did, but reads s2v[s,:] CONTIGUOUSLY and needs no d_ring_pos ring-search.

    WHY per-surface (profiled): the old per-vertex scan was the dominant per-step cost (~2.1 ms in
    the dynamic loop even at quiescence) -- cheap warm (0.03 ms) but ~10x cold-cache-penalized,
    because each (vertex, incident-surface, side) did a scattered d_ring_neighbor -> d_ring_pos ring
    walk with no structural early-out. The per-surface form has the SAME contiguous access pattern
    as compute_geometry_warp (which costs ~0.29 ms), so it is ~7x cheaper cold. Each edge is shared
    by two surfaces -> emitted twice; the interior (4-cell) half is applied SEPARATELY
    (filter_interior_short_edges_kernel, on the smaller endpoint); find_short_edges_warp dedups +
    sorts on the host. Output (interior-filtered, deduped, sorted) is BYTE-IDENTICAL -> no gate
    change. Output cap n_s*MAX_RING bounds the (<= one-per-ring-edge-per-surface) emission.

    NB no minimum-image in the length: a sub-Lth edge is tiny (both endpoints same box image), so
    the raw distance is exact for short edges -- matching the prior per-vertex scan."""
    s = wp.tid()
    if surf_alive[s] == 0:
        return
    L = s2v_len[s]
    for i in range(L):
        a = s2v[s, i]
        b = s2v[s, (i + 1) % L]
        v = a                                            # canonical: smaller endpoint = v10
        w = b
        if b < a:
            v = b
            w = a
        if vert_alive[v] == 0 or vert_alive[w] == 0:
            continue
        if wp.length(vert_pos[v] - vert_pos[w]) < threshold:
            idx = wp.atomic_add(out_count, 0, 1)
            out_v10[idx] = v
            out_v11[idx] = w


@wp.kernel
def filter_interior_short_edges_kernel(
        cand_v10: wp.array(dtype=wp.int32),
        v2s: wp.array2d(dtype=wp.int32), v2s_len: wp.array(dtype=wp.int32),
        s2b: wp.array2d(dtype=wp.int32), keep: wp.array(dtype=wp.int32)):
    """Per emitted candidate edge: keep iff its SMALLER endpoint (v10) is an interior 4-cell
    vertex -- the find_short_edges interior filter, applied to the (few) candidates so the
    O(L^2) d_vert_body_count stays OUT of the per-vertex scan kernel (occupancy). Mirrors the
    host short_edge_trigger_host, which filters on the smaller endpoint's interiorness only."""
    i = wp.tid()
    if d_vert_body_count(v2s, v2s_len, s2b, cand_v10[i]) == 4:
        keep[i] = wp.int32(1)
    else:
        keep[i] = wp.int32(0)


def _ensure_detect_buf(g: dict, cap: int) -> dict:
    """Get-or-grow the reusable scan buffers (out_v10/out_v11/keep sized >= cap + the 1-int count),
    stashed on g['_detect_buf']. Per-call `wp.zeros(cap)` of the two cap=n_s*MAX_RING (~4.2M-int)
    emit arrays was 59% of the early-phase detect cost (scratchpad/prof_detect.py); the scan writes
    only [0,count) and the host reads only [:k], so stale tail entries are never consumed -> reuse
    is bit-identical, needing only a count reset. Grows x2 (mirrors _ensure_gather_buf)."""
    buf = g.get("_detect_buf")
    if buf is not None and buf["cap"] >= cap:
        return buf
    dev = g["device"]
    newcap = max(cap, 256, (buf["cap"] * 2 if buf is not None else 0))
    z = lambda: wp.zeros(newcap, dtype=wp.int32, device=dev)
    buf = dict(cap=newcap, out_v10=z(), out_v11=z(), keep=z(),
               count=wp.zeros(1, dtype=wp.int32, device=dev))
    g["_detect_buf"] = buf
    return buf


def find_short_edges_warp(g: dict, threshold: float) -> np.ndarray:
    """GPU parallel Condition-2 trigger scan for [I] sites. Returns an (M,2) int32 array of
    candidate short edges (v10, v11) sorted by (v10, v11), interior-filtered + deduped, read
    straight from the device SoA -- the O(mesh) work of find_short_edges_csr, now one thread per
    SURFACE (cheap length trigger) + a per-candidate interior filter on the GPU."""
    dev = g["device"]
    n_s = int(g["n_used"].numpy()[1])               # tiny readback (live surfaces; per-surface scan)
    if n_s == 0:
        return np.zeros((0, 2), np.int32)
    cap = n_s * g["MAX_RING"]                        # safe bound: <= one emit per ring edge per surface
    buf = _ensure_detect_buf(g, cap)                # REUSED buffers (no per-call cap-sized alloc)
    out_v10, out_v11, keep, count = buf["out_v10"], buf["out_v11"], buf["keep"], buf["count"]
    count.zero_()                                   # reset the atomic emit counter (the only reset needed)
    wp.launch(scan_short_edges_kernel, dim=n_s, device=dev, inputs=[     # cheap per-surface length trigger
        g["vert_pos"], g["vert_alive"], g["surf_alive"], g["s2v"], g["s2v_len"],
        wp.float64(threshold), out_v10, out_v11, count])
    wp.synchronize_device(dev)
    k = int(count.numpy()[0])
    if k == 0:
        return np.zeros((0, 2), np.int32)
    wp.launch(filter_interior_short_edges_kernel, dim=k, device=dev,    # interior filter, off-hot-path
              inputs=[out_v10, g["v2s"], g["v2s_len"], g["s2b"], keep])
    wp.synchronize_device(dev)
    mask = keep[:k].numpy().astype(bool)
    v10 = out_v10[:k].numpy()[mask]                  # slice ON DEVICE first -> copy only k, not cap
    v11 = out_v11[:k].numpy()[mask]
    if v10.size == 0:
        return np.zeros((0, 2), np.int32)
    # np.unique(axis=0) both DEDUPS (the scan emits an edge once per incident face) and sorts rows
    # lexicographically -> the exact canonical (v10,v11)-ascending interior-filtered deduped set the
    # host trigger (find_short_edges_csr) produces. (Dedup + interior filter moved off the scan.)
    return np.unique(np.stack([v10, v11], axis=1).astype(np.int32), axis=0)


def short_edge_trigger_host(pm: PaddedMesh, threshold: float) -> Set[Tuple[int, int]]:
    """Host reference for scan_short_edges_kernel: the exact find_short_edges trigger WITHOUT
    the i_neighbourhood gather -- edges (v, w), v<w, both alive, v interior (4-cell), w a
    ring-neighbour of v, len < threshold. The GPU scan must reproduce this set."""
    out: Set[Tuple[int, int]] = set()
    for v in range(pm.n_v_used):
        if not pm.vert_alive[v] or len(vert_bodies(pm, v)) != 4:
            continue
        for w in connected_vertices(pm, v):
            if w <= v or not pm.vert_alive[w]:
                continue
            if edge_length(pm, v, w) < threshold:
                out.add((v, w))
    return out


def detect_short_edges_hybrid(g: dict, pm: PaddedMesh, threshold: float
                              ) -> List[Tuple[int, int, ICfgIdx]]:
    """GPU trigger scan + host i_neighbourhood gather on just the candidates -- the same
    (v10, v11, ICfgIdx) list as topology_csr.find_short_edges_csr, but the O(mesh) scan runs
    on the GPU. `pm` is the host mirror used only for the O(candidates) gather; (v10,v11)
    -ascending order keeps the downstream reservation deterministic."""
    out = []
    for v, w in find_short_edges_warp(g, threshold):
        cfg = i_neighbourhood_csr(pm, int(v), int(w))
        if cfg is not None:
            out.append((int(v), int(w), cfg))
    return out
