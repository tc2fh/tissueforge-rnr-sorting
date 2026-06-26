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
from warp.utils import array_scan, radix_sort_pairs

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


_TRI_BUF_START = 8192   # grown to >= n_s on first use; small triangles are rare so k is tiny


def _ensure_tri_buf(g: dict, cap: int) -> dict:
    """Get-or-grow the reusable [H] scan+sort buffer (stashed on g['_tri_buf']). `keys` doubles as
    the scan emit array AND the radix scratch, so it is sized 2*cap (radix_sort_pairs needs 2*count);
    `values` is the required int32 companion. Sized >= n_s -- the scan emits at most one entry per
    surface, so it never overflows and needs no bounds guard (unlike the I-side, whose emit can
    exceed n_s) -- grown x2. The H emit k is tiny (small triangles are rare), so the persistent
    footprint is ~n_s ints, negligible vs compact's double-buffer."""
    buf = g.get("_tri_buf")
    if buf is not None and buf["cap"] >= cap:
        return buf
    dev = g["device"]
    newcap = max(cap, _TRI_BUF_START, (buf["cap"] * 2 if buf is not None else 0))
    buf = dict(cap=newcap, keys=wp.zeros(2 * newcap, dtype=wp.int32, device=dev),
               values=wp.zeros(2 * newcap, dtype=wp.int32, device=dev),
               count=wp.zeros(1, dtype=wp.int32, device=dev))
    g["_tri_buf"] = buf
    return buf


def find_small_triangles_device(g: dict, threshold: float):
    """FULLY DEVICE-RESIDENT [H] trigger detect. Returns (c_tris, M): c_tris is a DEVICE int32
    array whose first M entries are the canonical (surface-ascending) small-triangle candidate
    indices -- the SAME set AND order as find_small_triangles_warp's host np.sort, but the candidate
    list never leaves the device (only the scalar count M is read back, to size the downstream
    gather/reserve/apply launches). SIMPLER than the I-side: scan_small_triangles_kernel emits one
    thread per surface so there are NO duplicates -> a plain device sort (radix_sort_pairs on the
    int32 surface indices) reproduces np.sort; no dedup pass. Consumed by gather_h_configs_warp_device
    with no h2d re-upload."""
    dev = g["device"]
    n_s = int(g["n_used"].numpy()[1])
    buf = _ensure_tri_buf(g, n_s)        # >= n_s so the <=1-emit-per-surface scan never overflows
    if n_s == 0:
        return buf["keys"], 0
    buf["count"].zero_()
    wp.launch(scan_small_triangles_kernel, dim=n_s, device=dev, inputs=[
        g["vert_pos"], g["surf_alive"], g["s2v"], g["s2v_len"], g["s2b"],
        wp.float64(threshold), buf["keys"], buf["count"]])
    wp.synchronize_device(dev)
    k = int(buf["count"].numpy()[0])
    if k == 0:
        return buf["keys"], 0
    radix_sort_pairs(buf["keys"], buf["values"], k)   # surface-ascending == host np.sort(out[:k])
    return buf["keys"], k


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
            if idx < out_v10.shape[0]:           # bounds guard: count stays exact on overflow so the
                out_v10[idx] = v                 # caller (find_short_edges_warp) grows the buffer to
                out_v11[idx] = w                 # fit k and rescans -> small persistent buffer, no OOB.


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


# --------------------------------------------------------------------------------------
# device-resident dedup+lex-sort: reproduce np.unique(axis=0) ON THE GPU so the candidate
# list never round-trips to host (find_short_edges_device). Pack (v10,v11) into one int64
# key v10*STRIDE+v11 (STRIDE > any vertex slot) -> radix_sort_pairs gives lex-ascending order
# == np.unique(axis=0); a mark-first + array_scan + scatter then dedups deterministically.
# --------------------------------------------------------------------------------------
_KEY_STRIDE = wp.constant(wp.int64(1 << 32))     # > any vertex slot index (cap_v << 2^32)
_SENTINEL_KEY = wp.constant(wp.int64(1 << 62))   # filtered-out edges -> sort to the end, never emitted


@wp.kernel
def build_short_edge_keys_kernel(
        out_v10: wp.array(dtype=wp.int32), out_v11: wp.array(dtype=wp.int32),
        keep: wp.array(dtype=wp.int32),
        keys: wp.array(dtype=wp.int64), values: wp.array(dtype=wp.int32)):
    """Pack each raw candidate edge into a sortable int64 key = v10*STRIDE + v11 (v10<v11
    already canonical from the scan); interior-FILTERED-OUT edges (keep==0) get _SENTINEL_KEY
    so the lex sort drops them to the tail where mark_first never emits them. `values[i]=i`
    rides along the stable radix sort so the scatter can read the unpermuted (out_v10,out_v11)."""
    i = wp.tid()
    values[i] = i
    if keep[i] == 1:
        keys[i] = wp.int64(out_v10[i]) * _KEY_STRIDE + wp.int64(out_v11[i])
    else:
        keys[i] = _SENTINEL_KEY


@wp.kernel
def mark_first_kernel(keys: wp.array(dtype=wp.int64), is_first: wp.array(dtype=wp.int32)):
    """After the lex sort, flag the FIRST occurrence of each distinct (non-sentinel) key --
    the dedup half of np.unique. Equal rows are adjacent post-sort, so one flag per distinct
    edge; sentinels (filtered-out) are never flagged."""
    j = wp.tid()
    kj = keys[j]
    if kj == _SENTINEL_KEY:
        is_first[j] = 0
    elif j == 0:
        is_first[j] = 1
    elif keys[j - 1] != kj:
        is_first[j] = 1
    else:
        is_first[j] = 0


@wp.kernel
def scatter_unique_kernel(
        values: wp.array(dtype=wp.int32), is_first: wp.array(dtype=wp.int32),
        out_pos: wp.array(dtype=wp.int32),
        out_v10: wp.array(dtype=wp.int32), out_v11: wp.array(dtype=wp.int32),
        c_v10: wp.array(dtype=wp.int32), c_v11: wp.array(dtype=wp.int32)):
    """Scatter each flagged distinct edge to its lex-rank slot (inclusive-scan position - 1),
    reading the original (v10,v11) via the sort-permuted index `values[j]`. Result c_v10/c_v11
    [0,M) is the (v10,v11)-ascending deduped set -- byte-equal to np.unique(axis=0)."""
    j = wp.tid()
    if is_first[j] == 1:
        idx = out_pos[j] - 1
        orig = values[j]
        c_v10[idx] = out_v10[orig]
        c_v11[idx] = out_v11[orig]


_DETECT_BUF_START = 8192   # the actual emit k is ~150 (max seen ~222); start small, grow on overflow


def _ensure_detect_buf(g: dict, cap: int) -> dict:
    """Get-or-grow the reusable scan buffers (out_v10/out_v11/keep sized >= cap + the 1-int count),
    stashed on g['_detect_buf']. Per-call `wp.zeros(...)` of the emit arrays was 59% of the
    early-phase detect cost (scratchpad/prof_detect.py); the scan writes only [0,count) and the host
    reads only [:k], so stale tail entries are never consumed -> reuse is bit-identical, needing only
    a count reset. Sized to the ~150-entry ACTUAL emit (not the n_s*MAX_RING ~4.2M worst case): starts
    at _DETECT_BUF_START and grows x2 on overflow (find_short_edges_warp rescans), keeping the
    persistent per-sim footprint at ~KB not ~50 MB (matters under heavy concurrency)."""
    buf = g.get("_detect_buf")
    if buf is not None and buf["cap"] >= cap:
        return buf
    dev = g["device"]
    newcap = max(cap, _DETECT_BUF_START, (buf["cap"] * 2 if buf is not None else 0))
    z = lambda: wp.zeros(newcap, dtype=wp.int32, device=dev)
    # keys/values feed radix_sort_pairs, which needs 2*count slots of scratch (count<=cap).
    buf = dict(cap=newcap, out_v10=z(), out_v11=z(), keep=z(),
               count=wp.zeros(1, dtype=wp.int32, device=dev),
               keys=wp.zeros(2 * newcap, dtype=wp.int64, device=dev),
               values=wp.zeros(2 * newcap, dtype=wp.int32, device=dev),
               is_first=z(), out_pos=z(), cand_v10=z(), cand_v11=z())
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

    def _scan(buf):                                 # reset count, emit short edges into buf; return k
        buf["count"].zero_()
        wp.launch(scan_short_edges_kernel, dim=n_s, device=dev, inputs=[  # cheap per-surface length trigger
            g["vert_pos"], g["vert_alive"], g["surf_alive"], g["s2v"], g["s2v_len"],
            wp.float64(threshold), buf["out_v10"], buf["out_v11"], buf["count"]])
        wp.synchronize_device(dev)
        return int(buf["count"].numpy()[0])

    buf = _ensure_detect_buf(g, 0)                  # small REUSED buffer (no per-call cap-sized alloc)
    k = _scan(buf)
    if k > buf["cap"]:                              # buffer overflowed (the kernel bounds-guards writes)
        buf = _ensure_detect_buf(g, k)             # -> grow to fit the exact count + rescan (rare)
        k = _scan(buf)
    out_v10, out_v11, keep = buf["out_v10"], buf["out_v11"], buf["keep"]
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


def find_short_edges_device(g: dict, threshold: float):
    """FULLY DEVICE-RESIDENT [I] trigger detect. Returns (cand_v10, cand_v11, M): cand_v10/
    cand_v11 are DEVICE int32 arrays whose first M entries are the canonical (v10,v11)-ascending
    interior-filtered deduped short edges -- the SAME set AND order as find_short_edges_warp's
    host np.unique(axis=0), but the candidate list NEVER leaves the device (only the scalar count
    M is read back, to size the downstream gather/reserve/apply launches).

    Removes, per I->H round vs find_short_edges_warp: the keep[:k] + out_v10/out_v11[:k] d2h
    reads and the host np.unique; the gather then consumes these device arrays directly
    (gather_i_configs_warp_device) so the h2d candidate re-upload is gone too. Net per round:
    ~3 serializing host syncs -> 1 (n_s, k, M readbacks; the scan still needs k for the launch
    dims + overflow-grow). Bit-identical: the dedup is np.unique reproduced on-device (int64
    lex-sort of v10*STRIDE+v11, mark-first + array_scan + scatter)."""
    dev = g["device"]
    n_s = int(g["n_used"].numpy()[1])               # live surfaces -> per-surface scan dim
    buf = _ensure_detect_buf(g, 0)
    if n_s == 0:
        return buf["cand_v10"], buf["cand_v11"], 0

    def _scan(b):                                   # reset count, emit short edges; return raw k
        b["count"].zero_()
        wp.launch(scan_short_edges_kernel, dim=n_s, device=dev, inputs=[
            g["vert_pos"], g["vert_alive"], g["surf_alive"], g["s2v"], g["s2v_len"],
            wp.float64(threshold), b["out_v10"], b["out_v11"], b["count"]])
        wp.synchronize_device(dev)
        return int(b["count"].numpy()[0])

    k = _scan(buf)
    if k > buf["cap"]:                              # bounds-guarded scan overflowed -> grow + rescan
        buf = _ensure_detect_buf(g, k)
        k = _scan(buf)
    if k == 0:
        return buf["cand_v10"], buf["cand_v11"], 0

    out_v10, out_v11, keep = buf["out_v10"], buf["out_v11"], buf["keep"]
    keys, values, is_first = buf["keys"], buf["values"], buf["is_first"]
    out_pos, cand_v10, cand_v11 = buf["out_pos"], buf["cand_v10"], buf["cand_v11"]

    wp.launch(filter_interior_short_edges_kernel, dim=k, device=dev,   # keep stays on device
              inputs=[out_v10, g["v2s"], g["v2s_len"], g["s2b"], keep])
    wp.launch(build_short_edge_keys_kernel, dim=k, device=dev,         # pack (v10,v11) -> int64 key
              inputs=[out_v10, out_v11, keep, keys, values])
    radix_sort_pairs(keys, values, k)                                 # lex-ascending == np.unique order
    wp.launch(mark_first_kernel, dim=k, device=dev, inputs=[keys, is_first])   # dedup-adjacent flag
    array_scan(is_first[:k], out_pos[:k], inclusive=True)             # flag -> output slot (inclusive)
    M = int(out_pos[k - 1:k].numpy()[0])             # distinct count = last inclusive-scan value
    if M == 0:                                       # every candidate filtered out
        return cand_v10, cand_v11, 0
    wp.launch(scatter_unique_kernel, dim=k, device=dev,
              inputs=[values, is_first, out_pos, out_v10, out_v11, cand_v10, cand_v11])
    return cand_v10, cand_v11, M


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
