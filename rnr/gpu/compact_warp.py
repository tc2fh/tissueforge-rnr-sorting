"""Gate D (device): stream-compaction of dead vertex/surface slots on the GPU SoA -- the
device counterpart of PaddedMesh.compact() (docs/2026-06-24_gpu-3d-vertex-model-exploration.md).

The bump allocator never reclaims (deaths only set alive=0), so n_v_used/n_s_used grow with
every reconnection. Compaction renumbers the LIVE elements into a contiguous prefix [0, n_live)
in ascending old-slot order and resets the high-water counters, so the arrays stay bounded over
long runs -- cellGPU's grow-then-compact, in 3D.

Device algorithm (no O(mesh) host work):
  1. exclusive prefix-sum of the alive flags (wp.utils.array_scan) -> for each live old slot,
     scan[old] IS its new index (count of live slots before it). This is the remap; it is
     deterministic + ascending, so it matches the host np.where-based compact slot-for-slot.
  2. SCATTER live rows into fresh arrays through the remap (verts via vmap=scan_v, surfaces via
     smap=scan_s); s2v holds vertex idx -> vmap, v2s/b2s hold surface idx -> smap, s2b holds
     body idx -> unchanged (bodies are stable under I<->H).
  3. set n_used from the scan tail on-device; swap the new arrays into `g` in place.

The fresh arrays are pre-filled (-1 pad / 0 alive / 0 len), so the reclaimed tail is clean
without an extra pass. Matches the host reference by the body-anchored fingerprint AND, since
both renumber in ascending old-slot order, slot-for-slot.
"""
import numpy as np

import warp as wp

wp.init()


@wp.kernel
def _scatter_verts_kernel(
        vert_alive: wp.array(dtype=wp.int32), scan_v: wp.array(dtype=wp.int32),
        scan_s: wp.array(dtype=wp.int32),
        vert_pos: wp.array(dtype=wp.vec3d), v2s: wp.array2d(dtype=wp.int32),
        v2s_len: wp.array(dtype=wp.int32),
        vert_pos_n: wp.array(dtype=wp.vec3d), vert_alive_n: wp.array(dtype=wp.int32),
        v2s_n: wp.array2d(dtype=wp.int32), v2s_len_n: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if vert_alive[i] == 0:
        return
    nw = scan_v[i]                              # new slot = #live before i (ascending)
    vert_pos_n[nw] = vert_pos[i]
    vert_alive_n[nw] = 1
    L = v2s_len[i]
    for k in range(L):
        v2s_n[nw, k] = scan_s[v2s[i, k]]        # v2s holds SURFACE indices -> smap
    v2s_len_n[nw] = L


@wp.kernel
def _scatter_surfs_kernel(
        surf_alive: wp.array(dtype=wp.int32), scan_v: wp.array(dtype=wp.int32),
        scan_s: wp.array(dtype=wp.int32),
        s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
        s2b: wp.array2d(dtype=wp.int32),
        surf_alive_n: wp.array(dtype=wp.int32), s2v_n: wp.array2d(dtype=wp.int32),
        s2v_len_n: wp.array(dtype=wp.int32), s2b_n: wp.array2d(dtype=wp.int32)):
    i = wp.tid()
    if surf_alive[i] == 0:
        return
    nw = scan_s[i]
    surf_alive_n[nw] = 1
    L = s2v_len[i]
    for k in range(L):
        s2v_n[nw, k] = scan_v[s2v[i, k]]        # s2v holds VERTEX indices -> vmap
    s2v_len_n[nw] = L
    s2b_n[nw, 0] = s2b[i, 0]                     # s2b holds BODY indices -> unchanged
    s2b_n[nw, 1] = s2b[i, 1]


@wp.kernel
def _relabel_b2s_kernel(
        scan_s: wp.array(dtype=wp.int32), b2s: wp.array2d(dtype=wp.int32),
        b2s_len: wp.array(dtype=wp.int32),
        b2s_n: wp.array2d(dtype=wp.int32), b2s_len_n: wp.array(dtype=wp.int32)):
    b = wp.tid()
    L = b2s_len[b]
    for k in range(L):
        b2s_n[b, k] = scan_s[b2s[b, k]]          # bodies stable; only their surface refs remap
    b2s_len_n[b] = L


@wp.kernel
def _set_nused_kernel(scan_v: wp.array(dtype=wp.int32), vert_alive: wp.array(dtype=wp.int32),
                      scan_s: wp.array(dtype=wp.int32), surf_alive: wp.array(dtype=wp.int32),
                      cap_v: int, cap_s: int, n_used_n: wp.array(dtype=wp.int32)):
    # exclusive-scan tail + last flag = total live count (the dead tail contributes 0)
    n_used_n[0] = scan_v[cap_v - 1] + vert_alive[cap_v - 1]
    n_used_n[1] = scan_s[cap_s - 1] + surf_alive[cap_s - 1]


def compact_warp(g: dict) -> dict:
    """Compact dead vertex/surface slots in the device SoA `g` IN PLACE (the arrays in `g` are
    replaced with fresh, compacted ones; cap_v/cap_s/MAX_*/nb are unchanged). Returns `g`.
    Matches PaddedMesh.compact() by fingerprint and slot-for-slot."""
    dev = g["device"]
    cap_v, cap_s, nb = g["cap_v"], g["cap_s"], g["nb"]
    mr, mvs, mbs = g["MAX_RING"], g["MAX_VS"], g["MAX_BS"]

    scan_v = wp.zeros(cap_v, dtype=wp.int32, device=dev)
    scan_s = wp.zeros(cap_s, dtype=wp.int32, device=dev)
    wp.utils.array_scan(g["vert_alive"], scan_v, False)     # exclusive: scan[i] = #live < i
    wp.utils.array_scan(g["surf_alive"], scan_s, False)

    full = lambda shape: wp.array(np.full(shape, -1, np.int32), dtype=wp.int32, device=dev)
    vert_pos_n = wp.zeros(cap_v, dtype=wp.vec3d, device=dev)
    vert_alive_n = wp.zeros(cap_v, dtype=wp.int32, device=dev)
    v2s_n = full((cap_v, mvs))
    v2s_len_n = wp.zeros(cap_v, dtype=wp.int32, device=dev)
    surf_alive_n = wp.zeros(cap_s, dtype=wp.int32, device=dev)
    s2v_n = full((cap_s, mr))
    s2v_len_n = wp.zeros(cap_s, dtype=wp.int32, device=dev)
    s2b_n = full((cap_s, 2))
    b2s_n = full((nb, mbs))
    b2s_len_n = wp.zeros(nb, dtype=wp.int32, device=dev)
    n_used_n = wp.zeros(2, dtype=wp.int32, device=dev)

    wp.launch(_scatter_verts_kernel, dim=cap_v, device=dev, inputs=[
        g["vert_alive"], scan_v, scan_s, g["vert_pos"], g["v2s"], g["v2s_len"],
        vert_pos_n, vert_alive_n, v2s_n, v2s_len_n])
    wp.launch(_scatter_surfs_kernel, dim=cap_s, device=dev, inputs=[
        g["surf_alive"], scan_v, scan_s, g["s2v"], g["s2v_len"], g["s2b"],
        surf_alive_n, s2v_n, s2v_len_n, s2b_n])
    wp.launch(_relabel_b2s_kernel, dim=nb, device=dev, inputs=[
        scan_s, g["b2s"], g["b2s_len"], b2s_n, b2s_len_n])
    wp.launch(_set_nused_kernel, dim=1, device=dev, inputs=[
        scan_v, g["vert_alive"], scan_s, g["surf_alive"], cap_v, cap_s, n_used_n])
    wp.synchronize_device(dev)

    g["vert_pos"], g["vert_alive"] = vert_pos_n, vert_alive_n
    g["v2s"], g["v2s_len"] = v2s_n, v2s_len_n
    g["surf_alive"], g["s2v"], g["s2v_len"], g["s2b"] = surf_alive_n, s2v_n, s2v_len_n, s2b_n
    g["b2s"], g["b2s_len"] = b2s_n, b2s_len_n
    g["n_used"] = n_used_n
    return g
