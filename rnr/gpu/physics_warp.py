"""Gate E (Stage-1 physics), GPU KERNELS: the geometry + four-force model of physics_csr.py
as NVIDIA Warp kernels on the device SoA (the `g` dict from PaddedMesh.to_warp).

Gated against the host reference (physics_csr.py), which is itself validated against TF
(test_gpu_physics_csr.py: geometry to float32-eps, all four forces to float32). Precision is
fp64 throughout (wp.vec3d / wp.float64) -- matches the RNR path and makes the GPU==host gate
tight (~1e-10), so a kernel bug shows up far below the float32 floor of the TF comparison.

Three kernels:
  * surface_geom_kernel (1 thread/surface): centroid, area, unnormalized normal.
  * body_geom_kernel    (1 thread/body):    volume, area, centroid, orientSign.
  * force_kernel        (1 thread/vertex):  volume + area + adhesion + active force.

The per-body force sum is restructured as a per-(surface, body) sum: a body `src` "defines"
surface `s` iff src in s2b[s], so iterating the vertex's surfaces and, for each, its <=2
incident bodies enumerates exactly the same (src, s) pairs the host's `for src: for s` loop
does -- no body de-duplication needed for the conservative forces. Only the active drive
(mean over DISTINCT incident bodies) needs the small O(valence^2) first-occurrence dedup.
"""
import numpy as np
import warp as wp

from .physics_csr import PhysParams, PhysState


# --------------------------------------------------------------------------------------
# device helpers (periodic minimum image + the shared area-gradient inner loop)
# --------------------------------------------------------------------------------------
@wp.func
def d_minimg(d: wp.vec3d, box: wp.vec3d) -> wp.vec3d:
    """Minimum-image displacement (tf_mesh_metrics.cpp:59-71). A non-positive box component
    leaves that axis unwrapped. Matches physics_csr.minimg (d - L*round(d/L))."""
    x = d[0]
    y = d[1]
    z = d[2]
    zero = wp.float64(0.0)
    if box[0] > zero:
        x = x - box[0] * wp.round(x / box[0])
    if box[1] > zero:
        y = y - box[1] * wp.round(y / box[1])
    if box[2] > zero:
        z = z - box[2] * wp.round(z / box[2])
    return wp.vec3d(x, y, z)


@wp.func
def d_area_grad(s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
                vert_pos: wp.array(dtype=wp.vec3d),
                s: wp.int32, v: wp.int32, sc: wp.vec3d, box: wp.vec3d) -> wp.vec3d:
    """dA_s/dx_v: the area gradient of surface s w.r.t. vertex v -- the inner loop shared by
    SurfaceAreaConstraint and Adhesion (physics_csr._area_gradient_of_surface)."""
    L = s2v_len[s]
    g_tot = wp.vec3d(0.0, 0.0, 0.0)
    for i in range(L):
        vc = s2v[s, i]
        vn = s2v[s, (i + 1) % L]
        posvc = sc + d_minimg(vert_pos[vc] - sc, box)
        posvn = sc + d_minimg(vert_pos[vn] - sc, box)
        tn = wp.cross(posvc - sc, posvn - sc)
        ln = wp.length(tn)
        if ln > wp.float64(0.0):
            g = (posvc - posvn) / wp.float64(L)
            if vc == v:
                g = g + (posvn - sc)
            elif vn == v:
                g = g - (posvc - sc)
            g_tot = g_tot + wp.cross(tn / ln, g)
    return g_tot


# --------------------------------------------------------------------------------------
# geometry kernels
# --------------------------------------------------------------------------------------
@wp.kernel
def surface_geom_kernel(vert_pos: wp.array(dtype=wp.vec3d),
                        s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
                        surf_alive: wp.array(dtype=wp.int32), box: wp.vec3d,
                        scent_out: wp.array(dtype=wp.vec3d),
                        sarea_out: wp.array(dtype=wp.float64),
                        snorm_out: wp.array(dtype=wp.vec3d)):
    s = wp.tid()
    if surf_alive[s] == 0:
        return
    L = s2v_len[s]
    if L == 0:
        return
    origin = vert_pos[s2v[s, 0]]
    cen = wp.vec3d(0.0, 0.0, 0.0)
    for i in range(L):
        cen = cen + d_minimg(vert_pos[s2v[s, i]] - origin, box)
    cen = origin + cen / wp.float64(L)
    nrm = wp.vec3d(0.0, 0.0, 0.0)
    ar = wp.float64(0.0)
    for i in range(L):
        posc = cen + d_minimg(vert_pos[s2v[s, i]] - cen, box)
        posp = cen + d_minimg(vert_pos[s2v[s, (i + 1) % L]] - cen, box)
        tn = wp.cross(posc - cen, posp - cen)
        nrm = nrm + tn
        ar = ar + wp.length(tn)
    scent_out[s] = cen
    snorm_out[s] = nrm
    sarea_out[s] = ar * wp.float64(0.5)


@wp.kernel
def body_geom_kernel(b2s: wp.array2d(dtype=wp.int32), b2s_len: wp.array(dtype=wp.int32),
                     s2b: wp.array2d(dtype=wp.int32), body_alive: wp.array(dtype=wp.int32),
                     scent: wp.array(dtype=wp.vec3d), sarea: wp.array(dtype=wp.float64),
                     snorm: wp.array(dtype=wp.vec3d), box: wp.vec3d,
                     bvol_out: wp.array(dtype=wp.float64), barea_out: wp.array(dtype=wp.float64),
                     bcent_out: wp.array(dtype=wp.vec3d), borient_out: wp.array(dtype=wp.float64)):
    b = wp.tid()
    if body_alive[b] == 0:
        return
    L = b2s_len[b]
    if L == 0:
        return
    origin = scent[b2s[b, 0]]
    atot = wp.float64(0.0)
    cen = wp.vec3d(0.0, 0.0, 0.0)
    for i in range(L):
        s = b2s[b, i]
        a = sarea[s]
        atot = atot + a
        cen = cen + (origin + d_minimg(scent[s] - origin, box)) * a
    cen = cen / atot
    vol = wp.float64(0.0)
    for i in range(L):
        s = b2s[b, i]
        sc = cen + d_minimg(scent[s] - cen, box)
        sense = wp.float64(1.0)
        if s2b[s, 0] != b:
            sense = wp.float64(-1.0)
        vol = vol + wp.dot(snorm[s], sc) * sense / wp.float64(6.0)
    orient = wp.float64(1.0)
    if vol < wp.float64(0.0):
        orient = wp.float64(-1.0)
        vol = -vol
    bvol_out[b] = vol
    barea_out[b] = atot
    bcent_out[b] = cen
    borient_out[b] = orient


# --------------------------------------------------------------------------------------
# force kernel (per vertex)
# --------------------------------------------------------------------------------------
@wp.kernel
def force_kernel(vert_pos: wp.array(dtype=wp.vec3d), vert_alive: wp.array(dtype=wp.int32),
                 s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
                 s2b: wp.array2d(dtype=wp.int32),
                 v2s: wp.array2d(dtype=wp.int32), v2s_len: wp.array(dtype=wp.int32),
                 scent: wp.array(dtype=wp.vec3d), snorm: wp.array(dtype=wp.vec3d),
                 bvol: wp.array(dtype=wp.float64), barea: wp.array(dtype=wp.float64),
                 borient: wp.array(dtype=wp.float64),
                 body_type: wp.array(dtype=wp.int32), body_director: wp.array(dtype=wp.vec3d),
                 box: wp.vec3d, kv: wp.float64, v0: wp.float64, ka: wp.float64, a0: wp.float64,
                 sigma: wp.float64, v_active: wp.float64,
                 fout: wp.array(dtype=wp.vec3d)):
    v = wp.tid()
    if vert_alive[v] == 0:
        return
    posv = vert_pos[v]
    fv = wp.vec3d(0.0, 0.0, 0.0)
    nval = v2s_len[v]

    # ---- conservative forces: per (surface of v, its <=2 bodies) ----
    for a in range(nval):
        s = v2s[v, a]
        L = s2v_len[s]
        # ring index of v in s (v is guaranteed present -> s in v2s[v])
        iv = wp.int32(0)
        for j in range(L):
            if s2v[s, j] == v:
                iv = j
        sc = scent[s]
        ag = d_area_grad(s2v, s2v_len, vert_pos, s, v, sc, box)   # shared area gradient
        # volume term needs the ring neighbours of v (vp = next, vn = prev) and snorm/L
        vp = s2v[s, (iv + 1) % L]
        vn = s2v[s, (iv - 1 + L) % L]
        scent_near = posv + d_minimg(sc - posv, box)
        edge = d_minimg(vert_pos[vn] - vert_pos[vp], box)
        vterm = wp.cross(scent_near, edge) + snorm[s] / wp.float64(L)

        for slot in range(2):
            src = s2b[s, slot]
            if src < 0:
                continue
            sense = wp.float64(1.0)
            if slot == 1:
                sense = wp.float64(-1.0)
            if kv != wp.float64(0.0):
                cv = borient[src] * kv * (v0 - bvol[src]) / wp.float64(3.0)
                fv = fv + vterm * (sense * cv)
            if ka != wp.float64(0.0):
                fv = fv + ag * (ka * (barea[src] - a0))
            if sigma != wp.float64(0.0):
                other = s2b[s, 1 - slot]
                if other >= 0:
                    if body_type[src] != body_type[other]:
                        fv = fv + ag * (wp.float64(0.25) * sigma)

    # ---- active drive: v_active * mean over DISTINCT incident bodies ----
    if v_active > wp.float64(0.0):
        dirsum = wp.vec3d(0.0, 0.0, 0.0)
        cnt = wp.int32(0)
        for a in range(nval):
            s = v2s[v, a]
            for slot in range(2):
                b = s2b[s, slot]
                if b >= 0:
                    seen = wp.int32(0)
                    for a2 in range(a + 1):
                        s2 = v2s[v, a2]
                        for slot2 in range(2):
                            if (a2 < a or slot2 < slot) and s2b[s2, slot2] == b:
                                seen = wp.int32(1)
                    if seen == 0:
                        dirsum = dirsum + body_director[b]
                        cnt = cnt + 1
        if cnt > 0:
            fv = fv + dirsum * (v_active / wp.float64(cnt))

    fout[v] = fv


# --------------------------------------------------------------------------------------
# director rotational diffusion + overdamped integrator
# --------------------------------------------------------------------------------------
@wp.func
def d_wrapbox(p: wp.vec3d, box: wp.vec3d) -> wp.vec3d:
    """Wrap a position into [0, L) per axis (the engine's PERIODIC_FULL particle BC). A
    non-positive box component leaves that axis unwrapped (finite cluster)."""
    x = p[0]
    y = p[1]
    z = p[2]
    if box[0] > wp.float64(0.0):
        x = x - box[0] * wp.floor(x / box[0])
    if box[1] > wp.float64(0.0):
        y = y - box[1] * wp.floor(y / box[1])
    if box[2] > wp.float64(0.0):
        z = z - box[2] * wp.floor(z / box[2])
    return wp.vec3d(x, y, z)


@wp.kernel
def integrate_kernel(vert_alive: wp.array(dtype=wp.int32), force: wp.array(dtype=wp.vec3d),
                     dt: wp.float64, box: wp.vec3d, vert_pos: wp.array(dtype=wp.vec3d)):
    """Overdamped forward Euler x += dt*f (mobility mu=1, density-0 vertex => unit mass),
    then wrap into the periodic box. In-place: each thread touches only its own vertex."""
    v = wp.tid()
    if vert_alive[v] == 0:
        return
    vert_pos[v] = d_wrapbox(vert_pos[v] + force[v] * dt, box)


@wp.kernel
def director_update_kernel(body_alive: wp.array(dtype=wp.int32), rot_std: wp.float64,
                           seed: wp.int32, step_dev: wp.array(dtype=wp.int32), nb: wp.int32,
                           body_director: wp.array(dtype=wp.vec3d)):
    """Active-Brownian rotational diffusion (tfMeshSolver.cpp:543-554):
    n <- normalize(n + sqrt(2*Dr*dt)*(xi - n)), xi ~ uniform on S^2. An unset director
    (|n|^2 ~ 0) is lazily seeded random-on-S^2. Statistical, NOT bit-matched to TF's mt19937;
    the validated quantity is the decay rate (2/3)*Dr (probe_native_calibration Part B).

    The per-step RNG key offset key0 = step*nb is read from the DEVICE scalar step_dev[0] (not a
    baked host int) so a CUDA-graph-captured launch varies its noise per replay: the host bumps
    step_dev (set_director_step) between replays, OUTSIDE the capture region. Byte-identical to the
    old host-key0 path for eager use -- step_dev[0]*nb is the same int32 as the old wp.int32(step*nb)
    (two's-complement product), so rand_init(seed, key0+b) is unchanged."""
    b = wp.tid()
    if body_alive[b] == 0:
        return
    n = body_director[b]
    st = wp.rand_init(seed, step_dev[0] * nb + b)
    if wp.dot(n, n) < wp.float64(1e-12):
        n = wp.vec3d(wp.float64(wp.randn(st)), wp.float64(wp.randn(st)), wp.float64(wp.randn(st)))
        nl = wp.length(n)
        if nl > wp.float64(1e-9):
            n = n / nl
    xi = wp.vec3d(wp.float64(wp.randn(st)), wp.float64(wp.randn(st)), wp.float64(wp.randn(st)))
    xl = wp.length(xi)
    if xl > wp.float64(1e-9):
        xi = xi / xl
    n = n + (xi - n) * rot_std
    body_director[b] = wp.normalize(n)


def integrate_warp(g: dict, force: wp.array, dt: float) -> None:
    """One overdamped step on the device (mutates g['vert_pos'] in place)."""
    wp.launch(integrate_kernel, dim=g["cap_v"], device=g["device"],
              inputs=[g["vert_alive"], force, wp.float64(dt), g["box"], g["vert_pos"]])


def set_director_step(g: dict, step: int) -> None:
    """Write the current integration step into g['_step_dev'] (a 1-int device scalar; lazily
    allocated). MUST run OUTSIDE any CUDA-graph capture region: a captured director launch reads
    step_dev each replay, so bumping it here is what makes the per-step RNG key vary across replays
    (the launch bakes seed/rot_std/nb, which are constant per sim). For eager stepping this is just
    the host providing `step` -- byte-identical to the old baked key0 = step*nb."""
    if "_step_dev" not in g:
        g["_step_dev"] = wp.zeros(1, dtype=wp.int32, device=g["device"])
    g["_step_dev"].fill_(int(step))


def _launch_director_update(g: dict, phys: dict, dr: float, dt: float, seed: int) -> None:
    """Launch the director kernel (reads g['_step_dev'] for the per-step key). CAPTURE-SAFE: the
    only per-step-varying input is the device scalar step_dev, so this launch goes inside a captured
    step graph while set_director_step runs between replays. Requires set_director_step to have run
    at least once (it allocates _step_dev)."""
    rot_std = float(np.sqrt(2.0 * dr * dt))
    wp.launch(director_update_kernel, dim=g["nb"], device=g["device"],
              inputs=[g["body_alive"], wp.float64(rot_std), wp.int32(int(seed)),
                      g["_step_dev"], wp.int32(int(g["nb"])), phys["body_director"]])


def director_update_warp(g: dict, phys: dict, dr: float, dt: float, seed: int, step: int) -> None:
    """Evolve every body's director by one rotational-diffusion step (mutates
    phys['body_director'] in place). `step` makes the per-body RNG key unique across steps; it is
    routed through a device scalar (g['_step_dev']) so the same launch can be CUDA-graph-captured and
    replayed with a per-step-varying seed (set_director_step bumps step_dev between replays). Eager
    callers get the old behavior byte-for-byte."""
    set_director_step(g, step)
    _launch_director_update(g, phys, dr, dt, seed)


# --------------------------------------------------------------------------------------
# python drivers
# --------------------------------------------------------------------------------------
def _box_vec(box):
    b = np.asarray(box, dtype=np.float64)
    return wp.vec3d(float(b[0]), float(b[1]), float(b[2]))


def upload_phys(state: PhysState, device) -> dict:
    """Upload per-body physics state (type + director) to the device."""
    bdir = np.ascontiguousarray(state.body_director, dtype=np.float64)
    return dict(
        body_type=wp.array(np.ascontiguousarray(state.body_type, dtype=np.int32),
                           dtype=wp.int32, device=device),
        body_director=wp.array(bdir, dtype=wp.vec3d, device=device),
    )


def _ensure_step_buffers(g: dict) -> None:
    """Lazily allocate the per-step geometry + force output buffers ONCE on `g` (reused every step,
    zeroed in place). The mempool already makes per-call wp.zeros free (measured 0 latency), so this
    is NOT a speed change -- it is the CUDA-GRAPH-CAPTURE PREREQUISITE: graph capture forbids memory
    allocation inside the captured region, so the static step path must own its scratch up front
    (docs/2026-06-26_cuda-graph-experiment-scope.md, P1). Surface buffers are shared by
    compute_geometry_warp and compute_surface_geom_warp (the latter runs in orient, after the former
    is consumed)."""
    if "_geo_scent" in g:
        return
    dev, cap_s, nb, cap_v = g["device"], g["cap_s"], g["nb"], g["cap_v"]
    g["_geo_scent"] = wp.zeros(cap_s, dtype=wp.vec3d, device=dev)
    g["_geo_sarea"] = wp.zeros(cap_s, dtype=wp.float64, device=dev)
    g["_geo_snorm"] = wp.zeros(cap_s, dtype=wp.vec3d, device=dev)
    g["_geo_bvol"] = wp.zeros(nb, dtype=wp.float64, device=dev)
    g["_geo_barea"] = wp.zeros(nb, dtype=wp.float64, device=dev)
    g["_geo_bcent"] = wp.zeros(nb, dtype=wp.vec3d, device=dev)
    g["_geo_borient"] = wp.zeros(nb, dtype=wp.float64, device=dev)
    g["_force"] = wp.zeros(cap_v, dtype=wp.vec3d, device=dev)


def compute_geometry_warp(g: dict) -> dict:
    """Launch the geometry kernels; return device arrays scent/sarea/snorm/bvol/barea/
    bcent/borient. Box is read from g['box'] (a wp.vec3d), set by physics_state_to_warp.
    Writes into persistent buffers on `g` (zeroed in place -> byte-identical to the old per-call
    wp.zeros; alloc-free for graph capture). The kernels skip dead slots, so the zero_ keeps their
    pad at 0 exactly as wp.zeros did."""
    _ensure_step_buffers(g)
    dev, cap_s, nb, box = g["device"], g["cap_s"], g["nb"], g["box"]
    scent, sarea, snorm = g["_geo_scent"], g["_geo_sarea"], g["_geo_snorm"]
    bvol, barea, bcent, borient = g["_geo_bvol"], g["_geo_barea"], g["_geo_bcent"], g["_geo_borient"]
    scent.zero_(); sarea.zero_(); snorm.zero_()
    wp.launch(surface_geom_kernel, dim=cap_s, device=dev,
              inputs=[g["vert_pos"], g["s2v"], g["s2v_len"], g["surf_alive"], box],
              outputs=[scent, sarea, snorm])
    bvol.zero_(); barea.zero_(); bcent.zero_(); borient.zero_()
    wp.launch(body_geom_kernel, dim=nb, device=dev,
              inputs=[g["b2s"], g["b2s_len"], g["s2b"], g["body_alive"],
                      scent, sarea, snorm, box],
              outputs=[bvol, barea, bcent, borient])
    return dict(scent=scent, sarea=sarea, snorm=snorm,
                bvol=bvol, barea=barea, bcent=bcent, borient=borient)


def compute_surface_geom_warp(g: dict) -> dict:
    """Surface-only geometry (centroid/area/UNNORMALIZED normal); runs ONLY surface_geom_kernel,
    skipping the body kernel. For callers that need just snorm (orient_repair_warp). The returned
    snorm is byte-identical to compute_geometry_warp's (same kernel, same inputs); the body kernel
    never feeds back into the surface kernel, so dropping it is arithmetic-preserving. Reuses the
    shared persistent surface buffers (alloc-free for capture)."""
    _ensure_step_buffers(g)
    dev, cap_s, box = g["device"], g["cap_s"], g["box"]
    scent, sarea, snorm = g["_geo_scent"], g["_geo_sarea"], g["_geo_snorm"]
    scent.zero_(); sarea.zero_(); snorm.zero_()
    wp.launch(surface_geom_kernel, dim=cap_s, device=dev,
              inputs=[g["vert_pos"], g["s2v"], g["s2v_len"], g["surf_alive"], box],
              outputs=[scent, sarea, snorm])
    return dict(scent=scent, sarea=sarea, snorm=snorm)


def compute_forces_warp(g: dict, geom_w: dict, params: PhysParams, phys: dict) -> wp.array:
    """Launch the force kernel; return a device (cap_v,) vec3d force array (a persistent buffer on
    `g`, zeroed in place -> byte-identical to the old per-call wp.zeros; alloc-free for capture)."""
    _ensure_step_buffers(g)
    dev = g["device"]
    f = g["_force"]
    f.zero_()
    wp.launch(force_kernel, dim=g["cap_v"], device=dev,
              inputs=[g["vert_pos"], g["vert_alive"], g["s2v"], g["s2v_len"], g["s2b"],
                      g["v2s"], g["v2s_len"], geom_w["scent"], geom_w["snorm"],
                      geom_w["bvol"], geom_w["barea"], geom_w["borient"],
                      phys["body_type"], phys["body_director"], g["box"],
                      wp.float64(params.kv), wp.float64(params.v0),
                      wp.float64(params.ka), wp.float64(params.a0),
                      wp.float64(params.sigma), wp.float64(params.v_active)],
              outputs=[f])
    return f


def attach_box(g: dict, box) -> dict:
    """Stash the periodic box (wp.vec3d) on the device dict so the kernels can read it."""
    g["box"] = _box_vec(box)
    return g
