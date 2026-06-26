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


def _alloc_compact_alt(g: dict) -> dict:
    """The alternate (double-buffer) set of compacted arrays, allocated ONCE and reused. Compaction
    ping-pongs between this set and `g`'s live set, so no per-call allocation (the old per-step
    `wp.zeros`/`np.full(-1)`+h2d was 91% of compact's cost -- ~3.0 of 3.5 ms at n=16)."""
    dev = g["device"]
    cap_v, cap_s, nb = g["cap_v"], g["cap_s"], g["nb"]
    mr, mvs, mbs = g["MAX_RING"], g["MAX_VS"], g["MAX_BS"]
    return dict(
        vert_pos=wp.zeros(cap_v, dtype=wp.vec3d, device=dev),
        vert_alive=wp.zeros(cap_v, dtype=wp.int32, device=dev),
        v2s=wp.zeros((cap_v, mvs), dtype=wp.int32, device=dev),
        v2s_len=wp.zeros(cap_v, dtype=wp.int32, device=dev),
        surf_alive=wp.zeros(cap_s, dtype=wp.int32, device=dev),
        s2v=wp.zeros((cap_s, mr), dtype=wp.int32, device=dev),
        s2v_len=wp.zeros(cap_s, dtype=wp.int32, device=dev),
        s2b=wp.zeros((cap_s, 2), dtype=wp.int32, device=dev),
        b2s=wp.zeros((nb, mbs), dtype=wp.int32, device=dev),
        b2s_len=wp.zeros(nb, dtype=wp.int32, device=dev),
        n_used=wp.zeros(2, dtype=wp.int32, device=dev),
    )


_SWAPPED = ("vert_pos", "vert_alive", "v2s", "v2s_len", "surf_alive", "s2v", "s2v_len",
            "s2b", "b2s", "b2s_len", "n_used")


def compact_warp(g: dict) -> dict:
    """Compact dead vertex/surface slots in the device SoA `g` IN PLACE (the arrays in `g` are
    replaced with fresh, compacted ones; cap_v/cap_s/MAX_*/nb are unchanged). Returns `g`.
    Matches PaddedMesh.compact() by fingerprint and slot-for-slot.

    Double-buffered: the scratch arrays + the prefix-scan buffers are allocated ONCE (lazily) and
    reused, reset on-device (`fill_(-1)`/`zero_()`) instead of rebuilt on the host every call.
    Bit-identical to the alloc-every-call version (same scan + scatter; dead slots reset to the
    same -1/0 pad). No `wp.synchronize` -- same-stream ordering makes the pointer swap safe, and
    the only caller that reads `n_used` (engine.forward_step) syncs there; dropping the barrier
    also lets concurrent sims overlap."""
    dev = g["device"]
    cap_v, cap_s, nb = g["cap_v"], g["cap_s"], g["nb"]
    if "_compact_alt" not in g:
        g["_compact_alt"] = _alloc_compact_alt(g)
        g["_compact_scan_v"] = wp.zeros(cap_v, dtype=wp.int32, device=dev)
        g["_compact_scan_s"] = wp.zeros(cap_s, dtype=wp.int32, device=dev)
    dst = g["_compact_alt"]
    scan_v, scan_s = g["_compact_scan_v"], g["_compact_scan_s"]

    # reset the reused scratch to the pad state (device-side; replaces per-call np.full(-1)+h2d).
    # -1 for the index/ref arrays (dead/empty entries), 0 for pos/alive/len/n_used.
    dst["vert_pos"].zero_(); dst["vert_alive"].zero_()
    dst["v2s"].fill_(-1); dst["v2s_len"].zero_()
    dst["surf_alive"].zero_(); dst["s2v"].fill_(-1); dst["s2v_len"].zero_(); dst["s2b"].fill_(-1)
    dst["b2s"].fill_(-1); dst["b2s_len"].zero_(); dst["n_used"].zero_()

    wp.utils.array_scan(g["vert_alive"], scan_v, False)     # exclusive: scan[i] = #live < i
    wp.utils.array_scan(g["surf_alive"], scan_s, False)

    wp.launch(_scatter_verts_kernel, dim=cap_v, device=dev, inputs=[
        g["vert_alive"], scan_v, scan_s, g["vert_pos"], g["v2s"], g["v2s_len"],
        dst["vert_pos"], dst["vert_alive"], dst["v2s"], dst["v2s_len"]])
    wp.launch(_scatter_surfs_kernel, dim=cap_s, device=dev, inputs=[
        g["surf_alive"], scan_v, scan_s, g["s2v"], g["s2v_len"], g["s2b"],
        dst["surf_alive"], dst["s2v"], dst["s2v_len"], dst["s2b"]])
    wp.launch(_relabel_b2s_kernel, dim=nb, device=dev, inputs=[
        scan_s, g["b2s"], g["b2s_len"], dst["b2s"], dst["b2s_len"]])
    wp.launch(_set_nused_kernel, dim=1, device=dev, inputs=[
        scan_v, g["vert_alive"], scan_s, g["surf_alive"], cap_v, cap_s, dst["n_used"]])

    # ping-pong: the compacted scratch becomes g's live set; g's old set becomes the next scratch
    # (same shapes). Pure pointer swap -- ordered on the stream after the scatters above.
    old = {k: g[k] for k in _SWAPPED}
    for k in _SWAPPED:
        g[k] = dst[k]
    g["_compact_alt"] = old
    return g
