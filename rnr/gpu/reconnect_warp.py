"""Gate B3 of the GPU port (docs/2026-06-24_gpu-3d-vertex-model-exploration.md):
the count-CHANGING I<->H surgery as a NVIDIA Warp kernel on the device, plus host
wrappers. This is the host-reference algorithm of reconnect_csr.py (Gate B2) lifted onto
the GPU -- the proof that the parallel-slot-allocator + Okuda placement + ragged-ring
surgery actually run as device code.

Single op per launch (`dim=1`): there is exactly one I<->H here, so the bump allocator's
`wp.atomic_add` is uncontended -- it is the SAME primitive the Gate-C independent-set
scheduler will run with many threads, validated here in isolation first (build the
make-or-break op before the scheduler, mirroring the CPU-RNR methodology).

Device functions (`d_*`) mirror PaddedMesh's primitives 1:1 (both-sides adjacency upkeep),
so this is a faithful translation, not a redesign -- the same discipline that lets the
eventual CUDA-in-fork port be a translation of this.

PRECISION (risk #2 in the design doc): placement runs in **fp64** (`wp.vec3d`). The
integer topology surgery is precision-independent, so it matches the host reference
bit-for-bit; fp64 placement reproduces the CPU oracle's Okuda formula to round-off
(~1e-12), preserving I<->H reversibility. `probe_placement_precision` measures the fp32
alternative on-device so the choice is data-backed (fp32 drifts ~1e-6, fine for the
gate tolerance but NOT bit-reversible -> fp64 is the right call for the RNR path).
"""
import numpy as np

import warp as wp

from .reconnect_csr import HArmIdx, HCfgIdx, ICfgIdx

wp.init()


# ======================================================================================
# device helpers -- mirror PaddedMesh primitives (both-sides adjacency upkeep)
# ======================================================================================
@wp.func
def d_ring_pos(s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
               s: int, v: int) -> int:
    L = s2v_len[s]
    for i in range(L):
        if s2v[s, i] == v:
            return i
    return -1


@wp.func
def d_ring_insert_after(s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
                        s: int, new_v: int, after_v: int):
    L = s2v_len[s]
    i = d_ring_pos(s2v, s2v_len, s, after_v)
    j = L
    while j > i + 1:                 # shift [i+1, L) right by one to open a slot at i+1
        s2v[s, j] = s2v[s, j - 1]
        j -= 1
    s2v[s, i + 1] = new_v
    s2v_len[s] = L + 1


@wp.func
def d_ring_drop(s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
                s: int, v: int):
    L = s2v_len[s]
    i = d_ring_pos(s2v, s2v_len, s, v)
    for j in range(i, L - 1):
        s2v[s, j] = s2v[s, j + 1]
    s2v[s, L - 1] = -1
    s2v_len[s] = L - 1


@wp.func
def d_v2s_add(v2s: wp.array2d(dtype=wp.int32), v2s_len: wp.array(dtype=wp.int32),
              v: int, s: int):
    L = v2s_len[v]
    for i in range(L):
        if v2s[v, i] == s:
            return
    v2s[v, L] = s
    v2s_len[v] = L + 1


@wp.func
def d_v2s_remove(v2s: wp.array2d(dtype=wp.int32), v2s_len: wp.array(dtype=wp.int32),
                 v: int, s: int):
    L = v2s_len[v]
    idx = wp.int32(-1)          # dynamic (mutated in the loop); Warp needs the typed init
    for i in range(L):
        if v2s[v, i] == s and idx < 0:
            idx = i
    if idx < 0:
        return
    for j in range(idx, L - 1):
        v2s[v, j] = v2s[v, j + 1]
    v2s[v, L - 1] = -1
    v2s_len[v] = L - 1


@wp.func
def d_replace_v(s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
                v2s: wp.array2d(dtype=wp.int32), v2s_len: wp.array(dtype=wp.int32),
                s: int, old_v: int, new_v: int):
    i = d_ring_pos(s2v, s2v_len, s, old_v)
    s2v[s, i] = new_v
    d_v2s_add(v2s, v2s_len, new_v, s)
    d_v2s_remove(v2s, v2s_len, old_v, s)


@wp.func
def d_insert_between(s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
                     v2s: wp.array2d(dtype=wp.int32), v2s_len: wp.array(dtype=wp.int32),
                     s: int, new_v: int, v1: int, v2: int):
    i1 = d_ring_pos(s2v, s2v_len, s, v1)
    i2 = d_ring_pos(s2v, s2v_len, s, v2)
    L = s2v_len[s]
    after = v1
    if ((i1 + 1) % L) != i2:
        after = v2
    d_ring_insert_after(s2v, s2v_len, s, new_v, after)
    d_v2s_add(v2s, v2s_len, new_v, s)


@wp.func
def d_drop_v(s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
             v2s: wp.array2d(dtype=wp.int32), v2s_len: wp.array(dtype=wp.int32),
             s: int, v: int):
    d_ring_drop(s2v, s2v_len, s, v)
    d_v2s_remove(v2s, v2s_len, v, s)


@wp.func
def d_attach_body(s2b: wp.array2d(dtype=wp.int32), b2s: wp.array2d(dtype=wp.int32),
                  b2s_len: wp.array(dtype=wp.int32), s: int, b: int):
    if s2b[s, 0] == b or s2b[s, 1] == b:
        return
    k = 0
    if s2b[s, 0] >= 0:
        k = 1
    s2b[s, k] = b
    L = b2s_len[b]
    b2s[b, L] = s
    b2s_len[b] = L + 1


@wp.func
def d_detach_body(s2b: wp.array2d(dtype=wp.int32), b2s: wp.array2d(dtype=wp.int32),
                  b2s_len: wp.array(dtype=wp.int32), s: int, b: int):
    for k in range(2):
        if s2b[s, k] == b:
            s2b[s, k] = -1
    L = b2s_len[b]
    idx = wp.int32(-1)          # dynamic (mutated in the loop); Warp needs the typed init
    for i in range(L):
        if b2s[b, i] == s and idx < 0:
            idx = i
    if idx >= 0:
        for j in range(idx, L - 1):
            b2s[b, j] = b2s[b, j + 1]
        b2s[b, L - 1] = -1
        b2s_len[b] = L - 1


@wp.func
def d_find3(a: wp.array(dtype=wp.int32), val: int) -> int:
    for k in range(3):
        if a[k] == val:
            return k
    return -1


@wp.func
def safe_unit(a: wp.vec3d) -> wp.vec3d:
    n = wp.length(a)
    if n > wp.float64(0.0):
        return a / n
    return a


# ======================================================================================
# I -> H kernel  (mirror reconnect_csr.i_to_h_csr; placement fp64)
# ======================================================================================
@wp.kernel
def i_to_h_kernel(
        vert_pos: wp.array(dtype=wp.vec3d), vert_alive: wp.array(dtype=wp.int32),
        v2s: wp.array2d(dtype=wp.int32), v2s_len: wp.array(dtype=wp.int32),
        surf_alive: wp.array(dtype=wp.int32),
        s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
        s2b: wp.array2d(dtype=wp.int32),
        b2s: wp.array2d(dtype=wp.int32), b2s_len: wp.array(dtype=wp.int32),
        n_used: wp.array(dtype=wp.int32),
        v10: int, v11: int, cap_top: int, cap_bot: int, dl_th: wp.float64,
        arm_side: wp.array(dtype=wp.int32), arm_otop: wp.array(dtype=wp.int32),
        arm_obot: wp.array(dtype=wp.int32),
        top_faces: wp.array(dtype=wp.int32), bot_faces: wp.array(dtype=wp.int32),
        out_tri: wp.array(dtype=wp.int32), out_T: wp.array(dtype=wp.int32)):

    # ---- bump-allocate 3 triangle vertices + 1 surface (uncontended at dim=1) ----------
    v0 = wp.atomic_add(n_used, 0, 3)
    sT = wp.atomic_add(n_used, 1, 1)
    tri0 = v0
    tri1 = v0 + 1
    tri2 = v0 + 2

    # ---- Okuda Appendix-1 placement (fp64) ---------------------------------------------
    p10 = vert_pos[v10]
    p11 = vert_pos[v11]
    r0 = wp.float64(0.5) * (p10 + p11)
    uT = safe_unit(p10 - p11)
    w0 = wp.float64(0.5) * (safe_unit(vert_pos[arm_otop[0]] - r0) + safe_unit(vert_pos[arm_obot[0]] - r0))
    vp0 = w0 - wp.dot(w0, uT) * uT
    w1 = wp.float64(0.5) * (safe_unit(vert_pos[arm_otop[1]] - r0) + safe_unit(vert_pos[arm_obot[1]] - r0))
    vp1 = w1 - wp.dot(w1, uT) * uT
    w2 = wp.float64(0.5) * (safe_unit(vert_pos[arm_otop[2]] - r0) + safe_unit(vert_pos[arm_obot[2]] - r0))
    vp2 = w2 - wp.dot(w2, uT) * uT
    lmax = wp.max(wp.length(vp0 - vp1), wp.max(wp.length(vp0 - vp2), wp.length(vp1 - vp2)))
    if lmax == wp.float64(0.0):
        lmax = wp.float64(1.0)
    sc = dl_th / lmax
    vert_pos[tri0] = r0 + sc * vp0
    vert_pos[tri1] = r0 + sc * vp1
    vert_pos[tri2] = r0 + sc * vp2
    vert_alive[tri0] = 1
    v2s_len[tri0] = 0
    vert_alive[tri1] = 1
    v2s_len[tri1] = 0
    vert_alive[tri2] = 1
    v2s_len[tri2] = 0

    # ---- (1) SIDE faces:  [.., otop, v10, v11, obot, ..] -> [.., otop, tri_k, obot, ..] --
    for k in range(3):
        s = arm_side[k]
        d_replace_v(s2v, s2v_len, v2s, v2s_len, s, v10, v0 + k)
        d_drop_v(s2v, s2v_len, v2s, v2s_len, s, v11)

    # ---- (2) TOP faces: v10 -> triangle edge (tri_prev, tri_next) -----------------------
    for t in range(3):
        face = top_faces[t]
        Lf = s2v_len[face]
        pi = d_ring_pos(s2v, s2v_len, face, v10)
        prev_v = s2v[face, (pi - 1 + Lf) % Lf]
        next_v = s2v[face, (pi + 1) % Lf]
        kp = d_find3(arm_otop, prev_v)
        kn = d_find3(arm_otop, next_v)
        d_replace_v(s2v, s2v_len, v2s, v2s_len, face, v10, v0 + kp)
        d_insert_between(s2v, s2v_len, v2s, v2s_len, face, v0 + kn, v0 + kp, next_v)

    # ---- (3) BOTTOM faces: mirror with v11 / arm_obot ----------------------------------
    for t in range(3):
        face = bot_faces[t]
        Lf = s2v_len[face]
        pi = d_ring_pos(s2v, s2v_len, face, v11)
        prev_v = s2v[face, (pi - 1 + Lf) % Lf]
        next_v = s2v[face, (pi + 1) % Lf]
        kp = d_find3(arm_obot, prev_v)
        kn = d_find3(arm_obot, next_v)
        d_replace_v(s2v, s2v_len, v2s, v2s_len, face, v11, v0 + kp)
        d_insert_between(s2v, s2v_len, v2s, v2s_len, face, v0 + kn, v0 + kp, next_v)

    # ---- (4) the new triangular face (winding [tri0, tri1, tri2]), caps' new contact ----
    surf_alive[sT] = 1
    s2v_len[sT] = 3
    s2v[sT, 0] = tri0
    s2v[sT, 1] = tri1
    s2v[sT, 2] = tri2
    s2b[sT, 0] = -1
    s2b[sT, 1] = -1
    d_v2s_add(v2s, v2s_len, tri0, sT)
    d_v2s_add(v2s, v2s_len, tri1, sT)
    d_v2s_add(v2s, v2s_len, tri2, sT)
    d_attach_body(s2b, b2s, b2s_len, sT, cap_top)
    d_attach_body(s2b, b2s, b2s_len, sT, cap_bot)

    # ---- (5) destroy the orphaned edge vertices ----------------------------------------
    vert_alive[v10] = 0
    v2s_len[v10] = 0
    vert_alive[v11] = 0
    v2s_len[v11] = 0

    out_tri[0] = tri0
    out_tri[1] = tri1
    out_tri[2] = tri2
    out_T[0] = sT


# ======================================================================================
# H -> I kernel  (mirror reconnect_csr.h_to_i_csr; placement fp64)
# ======================================================================================
@wp.kernel
def h_to_i_kernel(
        vert_pos: wp.array(dtype=wp.vec3d), vert_alive: wp.array(dtype=wp.int32),
        v2s: wp.array2d(dtype=wp.int32), v2s_len: wp.array(dtype=wp.int32),
        surf_alive: wp.array(dtype=wp.int32),
        s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
        s2b: wp.array2d(dtype=wp.int32),
        b2s: wp.array2d(dtype=wp.int32), b2s_len: wp.array(dtype=wp.int32),
        n_used: wp.array(dtype=wp.int32),
        triangle: int, cap_top: int, cap_bot: int, dl_th: wp.float64,
        tri_verts: wp.array(dtype=wp.int32),
        arm_side: wp.array(dtype=wp.int32), arm_otop: wp.array(dtype=wp.int32),
        arm_obot: wp.array(dtype=wp.int32),
        top_faces: wp.array(dtype=wp.int32), bot_faces: wp.array(dtype=wp.int32),
        out_nv: wp.array(dtype=wp.int32)):

    # ---- Okuda Eqs. 42-45 placement (fp64) ---------------------------------------------
    p0 = vert_pos[tri_verts[0]]
    p1 = vert_pos[tri_verts[1]]
    p2 = vert_pos[tri_verts[2]]
    r0 = (p0 + p1 + p2) / wp.float64(3.0)
    nrm = safe_unit(wp.cross(p1 - p0, p2 - p0))
    tm = (vert_pos[arm_otop[0]] + vert_pos[arm_otop[1]] + vert_pos[arm_otop[2]]) / wp.float64(3.0)
    if wp.dot(tm - r0, nrm) < wp.float64(0.0):
        nrm = -nrm
    half = wp.float64(0.5) * dl_th

    v0 = wp.atomic_add(n_used, 0, 2)
    nv10 = v0
    nv11 = v0 + 1
    vert_pos[nv10] = r0 + half * nrm
    vert_pos[nv11] = r0 - half * nrm
    vert_alive[nv10] = 1
    v2s_len[nv10] = 0
    vert_alive[nv11] = 1
    v2s_len[nv11] = 0

    # ---- (1) SIDE faces: [.., otop, tri_k, obot, ..] -> [.., otop, nv10, nv11, obot, ..] -
    for k in range(3):
        s = arm_side[k]
        d_replace_v(s2v, s2v_len, v2s, v2s_len, s, tri_verts[k], nv10)
        d_insert_between(s2v, s2v_len, v2s, v2s_len, s, nv11, nv10, arm_obot[k])

    # ---- (2) TOP faces: triangle edge -> single nv10 -----------------------------------
    for t in range(3):
        face = top_faces[t]
        first = wp.int32(-1)    # dynamic (mutated in the loop); Warp needs the typed init
        second = wp.int32(-1)
        L = s2v_len[face]
        for i in range(L):
            x = s2v[face, i]
            if x == tri_verts[0] or x == tri_verts[1] or x == tri_verts[2]:
                if first < 0:
                    first = x
                elif second < 0:
                    second = x
        d_replace_v(s2v, s2v_len, v2s, v2s_len, face, first, nv10)
        d_drop_v(s2v, s2v_len, v2s, v2s_len, face, second)

    # ---- (3) BOTTOM faces: mirror -> single nv11 ---------------------------------------
    for t in range(3):
        face = bot_faces[t]
        first = wp.int32(-1)    # dynamic (mutated in the loop); Warp needs the typed init
        second = wp.int32(-1)
        L = s2v_len[face]
        for i in range(L):
            x = s2v[face, i]
            if x == tri_verts[0] or x == tri_verts[1] or x == tri_verts[2]:
                if first < 0:
                    first = x
                elif second < 0:
                    second = x
        d_replace_v(s2v, s2v_len, v2s, v2s_len, face, first, nv11)
        d_drop_v(s2v, s2v_len, v2s, v2s_len, face, second)

    # ---- (4) detach + destroy the triangle and its orphaned vertices -------------------
    d_detach_body(s2b, b2s, b2s_len, triangle, cap_top)
    d_detach_body(s2b, b2s, b2s_len, triangle, cap_bot)
    surf_alive[triangle] = 0
    s2v_len[triangle] = 0
    for k in range(3):
        vert_alive[tri_verts[k]] = 0
        v2s_len[tri_verts[k]] = 0

    out_nv[0] = nv10
    out_nv[1] = nv11


# ======================================================================================
# host wrappers
# ======================================================================================
def _ai32(vals, dev):
    return wp.array(np.array(vals, np.int32), dtype=wp.int32, device=dev)


def i_to_h_warp(g: dict, cfg: ICfgIdx, dl_th: float) -> HCfgIdx:
    """Run one I->H on the device SoA `g` (PaddedMesh.to_warp output), in place. Returns
    the post-state HCfgIdx (in indices) for the inverse, exactly like i_to_h_csr."""
    dev = g["device"]
    arm_side = _ai32([a.side_surface for a in cfg.arms], dev)
    arm_otop = _ai32([a.outer_top for a in cfg.arms], dev)
    arm_obot = _ai32([a.outer_bot for a in cfg.arms], dev)
    topf = _ai32(list(cfg.top_faces.values()), dev)
    botf = _ai32(list(cfg.bottom_faces.values()), dev)
    out_tri = wp.zeros(3, dtype=wp.int32, device=dev)
    out_T = wp.zeros(1, dtype=wp.int32, device=dev)
    wp.launch(i_to_h_kernel, dim=1, device=dev, inputs=[
        g["vert_pos"], g["vert_alive"], g["v2s"], g["v2s_len"], g["surf_alive"],
        g["s2v"], g["s2v_len"], g["s2b"], g["b2s"], g["b2s_len"], g["n_used"],
        int(cfg.v10), int(cfg.v11), int(cfg.cap_top), int(cfg.cap_bot), float(dl_th),
        arm_side, arm_otop, arm_obot, topf, botf, out_tri, out_T])
    wp.synchronize_device(dev)
    tri = [int(x) for x in out_tri.numpy()]
    T = int(out_T.numpy()[0])
    arms = [HArmIdx(tri_vertex=tri[k], side_surface=cfg.arms[k].side_surface,
                    outer_top=cfg.arms[k].outer_top, outer_bot=cfg.arms[k].outer_bot)
            for k in range(3)]
    return HCfgIdx(triangle=T, tri_verts=tri, cap_top=cfg.cap_top, cap_bot=cfg.cap_bot,
                   side_cells=list(cfg.side_cells), arms=arms,
                   top_faces=dict(cfg.top_faces), bottom_faces=dict(cfg.bottom_faces))


def h_to_i_warp(g: dict, cfg: HCfgIdx, dl_th: float):
    """Run one H->I on the device SoA `g`, in place. Returns (nv10, nv11) slot indices."""
    dev = g["device"]
    tri_verts = _ai32(cfg.tri_verts, dev)
    arm_side = _ai32([a.side_surface for a in cfg.arms], dev)
    arm_otop = _ai32([a.outer_top for a in cfg.arms], dev)
    arm_obot = _ai32([a.outer_bot for a in cfg.arms], dev)
    topf = _ai32(list(cfg.top_faces.values()), dev)
    botf = _ai32(list(cfg.bottom_faces.values()), dev)
    out_nv = wp.zeros(2, dtype=wp.int32, device=dev)
    wp.launch(h_to_i_kernel, dim=1, device=dev, inputs=[
        g["vert_pos"], g["vert_alive"], g["v2s"], g["v2s_len"], g["surf_alive"],
        g["s2v"], g["s2v_len"], g["s2b"], g["b2s"], g["b2s_len"], g["n_used"],
        int(cfg.triangle), int(cfg.cap_top), int(cfg.cap_bot), float(dl_th),
        tri_verts, arm_side, arm_otop, arm_obot, topf, botf, out_nv])
    wp.synchronize_device(dev)
    nv = out_nv.numpy()
    return int(nv[0]), int(nv[1])


# ======================================================================================
# C2b: PARALLEL count-changing apply -- dim=N winners each run I->H simultaneously
# ======================================================================================
@wp.kernel
def i_to_h_batch_kernel(
        vert_pos: wp.array(dtype=wp.vec3d), vert_alive: wp.array(dtype=wp.int32),
        v2s: wp.array2d(dtype=wp.int32), v2s_len: wp.array(dtype=wp.int32),
        surf_alive: wp.array(dtype=wp.int32),
        s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
        s2b: wp.array2d(dtype=wp.int32),
        b2s: wp.array2d(dtype=wp.int32), b2s_len: wp.array(dtype=wp.int32),
        n_used: wp.array(dtype=wp.int32), dl_th: wp.float64,
        c_v10: wp.array(dtype=wp.int32), c_v11: wp.array(dtype=wp.int32),
        c_cap_top: wp.array(dtype=wp.int32), c_cap_bot: wp.array(dtype=wp.int32),
        c_arm_side: wp.array2d(dtype=wp.int32), c_arm_otop: wp.array2d(dtype=wp.int32),
        c_arm_obot: wp.array2d(dtype=wp.int32),
        c_top: wp.array2d(dtype=wp.int32), c_bot: wp.array2d(dtype=wp.int32),
        out_tri: wp.array2d(dtype=wp.int32), out_T: wp.array(dtype=wp.int32)):
    """One thread per winning candidate. Footprints are disjoint (the reservation
    guarantees it), so concurrent surgeries never touch a shared existing element; births
    use the shared atomic bump counter, so each thread gets distinct fresh slots. This is
    the i_to_h_kernel body, indexed per-candidate by tid."""
    i = wp.tid()
    v10 = c_v10[i]
    v11 = c_v11[i]
    cap_top = c_cap_top[i]
    cap_bot = c_cap_bot[i]

    # ---- bump-allocate 3 tri verts + 1 surface (atomic; concurrent threads get distinct) -
    v0 = wp.atomic_add(n_used, 0, 3)
    sT = wp.atomic_add(n_used, 1, 1)

    # ---- placement (fp64) --------------------------------------------------------------
    p10 = vert_pos[v10]
    p11 = vert_pos[v11]
    r0 = wp.float64(0.5) * (p10 + p11)
    uT = safe_unit(p10 - p11)
    w0 = wp.float64(0.5) * (safe_unit(vert_pos[c_arm_otop[i, 0]] - r0) + safe_unit(vert_pos[c_arm_obot[i, 0]] - r0))
    vp0 = w0 - wp.dot(w0, uT) * uT
    w1 = wp.float64(0.5) * (safe_unit(vert_pos[c_arm_otop[i, 1]] - r0) + safe_unit(vert_pos[c_arm_obot[i, 1]] - r0))
    vp1 = w1 - wp.dot(w1, uT) * uT
    w2 = wp.float64(0.5) * (safe_unit(vert_pos[c_arm_otop[i, 2]] - r0) + safe_unit(vert_pos[c_arm_obot[i, 2]] - r0))
    vp2 = w2 - wp.dot(w2, uT) * uT
    lmax = wp.max(wp.length(vp0 - vp1), wp.max(wp.length(vp0 - vp2), wp.length(vp1 - vp2)))
    if lmax == wp.float64(0.0):
        lmax = wp.float64(1.0)
    sc = dl_th / lmax
    vert_pos[v0] = r0 + sc * vp0
    vert_pos[v0 + 1] = r0 + sc * vp1
    vert_pos[v0 + 2] = r0 + sc * vp2
    vert_alive[v0] = 1
    v2s_len[v0] = 0
    vert_alive[v0 + 1] = 1
    v2s_len[v0 + 1] = 0
    vert_alive[v0 + 2] = 1
    v2s_len[v0 + 2] = 0

    # ---- (1) SIDE faces ----------------------------------------------------------------
    for k in range(3):
        s = c_arm_side[i, k]
        d_replace_v(s2v, s2v_len, v2s, v2s_len, s, v10, v0 + k)
        d_drop_v(s2v, s2v_len, v2s, v2s_len, s, v11)

    # ---- (2) TOP faces: v10 -> triangle edge -------------------------------------------
    for t in range(3):
        face = c_top[i, t]
        Lf = s2v_len[face]
        pi = d_ring_pos(s2v, s2v_len, face, v10)
        prev_v = s2v[face, (pi - 1 + Lf) % Lf]
        next_v = s2v[face, (pi + 1) % Lf]
        kp = wp.int32(-1)
        kn = wp.int32(-1)
        for kk in range(3):
            if c_arm_otop[i, kk] == prev_v:
                kp = kk
            if c_arm_otop[i, kk] == next_v:
                kn = kk
        d_replace_v(s2v, s2v_len, v2s, v2s_len, face, v10, v0 + kp)
        d_insert_between(s2v, s2v_len, v2s, v2s_len, face, v0 + kn, v0 + kp, next_v)

    # ---- (3) BOTTOM faces: mirror with v11 / arm_obot ----------------------------------
    for t in range(3):
        face = c_bot[i, t]
        Lf = s2v_len[face]
        pi = d_ring_pos(s2v, s2v_len, face, v11)
        prev_v = s2v[face, (pi - 1 + Lf) % Lf]
        next_v = s2v[face, (pi + 1) % Lf]
        kp = wp.int32(-1)
        kn = wp.int32(-1)
        for kk in range(3):
            if c_arm_obot[i, kk] == prev_v:
                kp = kk
            if c_arm_obot[i, kk] == next_v:
                kn = kk
        d_replace_v(s2v, s2v_len, v2s, v2s_len, face, v11, v0 + kp)
        d_insert_between(s2v, s2v_len, v2s, v2s_len, face, v0 + kn, v0 + kp, next_v)

    # ---- (4) new triangular face -------------------------------------------------------
    surf_alive[sT] = 1
    s2v_len[sT] = 3
    s2v[sT, 0] = v0
    s2v[sT, 1] = v0 + 1
    s2v[sT, 2] = v0 + 2
    s2b[sT, 0] = -1
    s2b[sT, 1] = -1
    d_v2s_add(v2s, v2s_len, v0, sT)
    d_v2s_add(v2s, v2s_len, v0 + 1, sT)
    d_v2s_add(v2s, v2s_len, v0 + 2, sT)
    d_attach_body(s2b, b2s, b2s_len, sT, cap_top)
    d_attach_body(s2b, b2s, b2s_len, sT, cap_bot)

    # ---- (5) destroy orphaned edge verts -----------------------------------------------
    vert_alive[v10] = 0
    v2s_len[v10] = 0
    vert_alive[v11] = 0
    v2s_len[v11] = 0

    out_tri[i, 0] = v0
    out_tri[i, 1] = v0 + 1
    out_tri[i, 2] = v0 + 2
    out_T[i] = sT


def apply_i_to_h_batch_warp(g: dict, batch, dl_th: float):
    """Apply a conflict-free batch of I->H on the device SoA `g`, ALL IN PARALLEL (dim=N).
    `batch` is a list of (v10, v11, ICfgIdx) (e.g. from the reservation). Returns the list
    of post-state HCfgIdx (one per candidate). Disjoint footprints => race-free."""
    n = len(batch)
    if n == 0:
        return []
    dev = g["device"]
    cfgs = [b[2] for b in batch]
    col = lambda fn: wp.array(np.array([fn(c) for c in cfgs], np.int32), dtype=wp.int32, device=dev)
    mat = lambda fn: wp.array(np.array([fn(c) for c in cfgs], np.int32), dtype=wp.int32, device=dev)
    c_v10 = col(lambda c: c.v10)
    c_v11 = col(lambda c: c.v11)
    c_cap_top = col(lambda c: c.cap_top)
    c_cap_bot = col(lambda c: c.cap_bot)
    c_arm_side = mat(lambda c: [a.side_surface for a in c.arms])
    c_arm_otop = mat(lambda c: [a.outer_top for a in c.arms])
    c_arm_obot = mat(lambda c: [a.outer_bot for a in c.arms])
    c_top = mat(lambda c: list(c.top_faces.values()))
    c_bot = mat(lambda c: list(c.bottom_faces.values()))
    out_tri = wp.zeros((n, 3), dtype=wp.int32, device=dev)
    out_T = wp.zeros(n, dtype=wp.int32, device=dev)
    wp.launch(i_to_h_batch_kernel, dim=n, device=dev, inputs=[
        g["vert_pos"], g["vert_alive"], g["v2s"], g["v2s_len"], g["surf_alive"],
        g["s2v"], g["s2v_len"], g["s2b"], g["b2s"], g["b2s_len"], g["n_used"], float(dl_th),
        c_v10, c_v11, c_cap_top, c_cap_bot, c_arm_side, c_arm_otop, c_arm_obot,
        c_top, c_bot, out_tri, out_T])
    wp.synchronize_device(dev)
    tri = out_tri.numpy()
    T = out_T.numpy()
    hcfgs = []
    for i, c in enumerate(cfgs):
        arms = [HArmIdx(tri_vertex=int(tri[i, k]), side_surface=c.arms[k].side_surface,
                        outer_top=c.arms[k].outer_top, outer_bot=c.arms[k].outer_bot)
                for k in range(3)]
        hcfgs.append(HCfgIdx(triangle=int(T[i]), tri_verts=[int(tri[i, 0]), int(tri[i, 1]), int(tri[i, 2])],
                             cap_top=c.cap_top, cap_bot=c.cap_bot, side_cells=list(c.side_cells),
                             arms=arms, top_faces=dict(c.top_faces), bottom_faces=dict(c.bottom_faces)))
    return hcfgs


# ======================================================================================
# precision probe (risk #2): fp32 vs fp64 placement on-device, vs the numpy fp64 oracle
# ======================================================================================
@wp.kernel
def _place_itoh_f64(P: wp.array(dtype=wp.vec3d), dl_th: wp.float64,
                    out: wp.array(dtype=wp.vec3d)):
    p10 = P[0]
    p11 = P[1]
    r0 = wp.float64(0.5) * (p10 + p11)
    uT = safe_unit(p10 - p11)
    w0 = wp.float64(0.5) * (safe_unit(P[2] - r0) + safe_unit(P[5] - r0))
    vp0 = w0 - wp.dot(w0, uT) * uT
    w1 = wp.float64(0.5) * (safe_unit(P[3] - r0) + safe_unit(P[6] - r0))
    vp1 = w1 - wp.dot(w1, uT) * uT
    w2 = wp.float64(0.5) * (safe_unit(P[4] - r0) + safe_unit(P[7] - r0))
    vp2 = w2 - wp.dot(w2, uT) * uT
    lmax = wp.max(wp.length(vp0 - vp1), wp.max(wp.length(vp0 - vp2), wp.length(vp1 - vp2)))
    if lmax == wp.float64(0.0):
        lmax = wp.float64(1.0)
    sc = dl_th / lmax
    out[0] = r0 + sc * vp0
    out[1] = r0 + sc * vp1
    out[2] = r0 + sc * vp2


@wp.func
def safe_unit_f32(a: wp.vec3) -> wp.vec3:
    n = wp.length(a)
    if n > wp.float32(0.0):
        return a / n
    return a


@wp.kernel
def _place_itoh_f32(P: wp.array(dtype=wp.vec3), dl_th: wp.float32,
                    out: wp.array(dtype=wp.vec3)):
    p10 = P[0]
    p11 = P[1]
    r0 = wp.float32(0.5) * (p10 + p11)
    uT = safe_unit_f32(p10 - p11)
    w0 = wp.float32(0.5) * (safe_unit_f32(P[2] - r0) + safe_unit_f32(P[5] - r0))
    vp0 = w0 - wp.dot(w0, uT) * uT
    w1 = wp.float32(0.5) * (safe_unit_f32(P[3] - r0) + safe_unit_f32(P[6] - r0))
    vp1 = w1 - wp.dot(w1, uT) * uT
    w2 = wp.float32(0.5) * (safe_unit_f32(P[4] - r0) + safe_unit_f32(P[7] - r0))
    vp2 = w2 - wp.dot(w2, uT) * uT
    lmax = wp.max(wp.length(vp0 - vp1), wp.max(wp.length(vp0 - vp2), wp.length(vp1 - vp2)))
    if lmax == wp.float32(0.0):
        lmax = wp.float32(1.0)
    sc = dl_th / lmax
    out[0] = r0 + sc * vp0
    out[1] = r0 + sc * vp1
    out[2] = r0 + sc * vp2


def probe_placement_precision(p10, p11, outer_tops, outer_bots, dl_th, device=None):
    """Compute the I->H triangle placement on-device in fp64 and fp32; return both plus
    the numpy fp64 oracle (reconnect.place_i_to_h_xyz). Lets the test report the fp32 drift
    that motivates the fp64 choice for the reversible RNR path."""
    from ..reconnect import place_i_to_h_xyz
    if device is None:
        cuda = [d for d in wp.get_devices() if d.is_cuda]
        device = cuda[0] if cuda else "cpu"
    P = np.array([p10, p11, outer_tops[0], outer_tops[1], outer_tops[2],
                  outer_bots[0], outer_bots[1], outer_bots[2]], dtype=np.float64)
    P64 = wp.array(P, dtype=wp.vec3d, device=device)
    out64 = wp.zeros(3, dtype=wp.vec3d, device=device)
    wp.launch(_place_itoh_f64, dim=1, inputs=[P64, float(dl_th), out64], device=device)
    P32 = wp.array(P.astype(np.float32), dtype=wp.vec3, device=device)
    out32 = wp.zeros(3, dtype=wp.vec3, device=device)
    wp.launch(_place_itoh_f32, dim=1, inputs=[P32, float(dl_th), out32], device=device)
    wp.synchronize_device(device)
    oracle = np.array(place_i_to_h_xyz(np.array(p10), np.array(p11),
                                       [np.array(t) for t in outer_tops],
                                       [np.array(b) for b in outer_bots], dl_th))
    return dict(oracle=oracle,
                gpu_f64=out64.numpy().reshape(3, 3),
                gpu_f32=out32.numpy().reshape(3, 3).astype(np.float64))
