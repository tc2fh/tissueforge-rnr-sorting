"""Long-run (100k-step) stability validation of the GPU 3D vertex engine.

THE faithfulness gate for the GPU sort. The Gate-E tests run at most 600 steps; a sort is
only "faithful" if the engine runs a full production-length trajectory (100k steps, the
Fig 1E/1F run length) without the mesh degrading. This harness drives `engine.forward_step`
in the FAITHFUL production regime (K_V=10, K_A=1, V0~1, A0~5.6, sigma in {0.1,0.2,0.5},
Lth=1e-3, active v0=0.1, Dr=1, reconnect every round(0.01/dt) steps -- sort_periodic_oracle.py)
and periodically AUDITS the mesh:

    * both-sides adjacency consistent      (PaddedMesh.check_consistency)
    * vertex positions all finite          (no NaN/Inf blow-up)
    * every cell volume strictly positive  (no inverted/everted cell)
    * bump-allocated slots stay bounded     (Gate-D compaction keeps nv/ns < capacity)
    * the demixing order parameter          (het-contact fraction: does it stay demixed?)

It reuses `_setup_unit_foam` from the Gate-E engine test VERBATIM, so the foam it stresses is
the exact one the test suite validates (host==TF / GPU==host / deterministic trajectory).

If the mesh degrades, the prime suspect is the two mesh-hygiene regularizers the port
intentionally omitted (they are not the sorting physics): FlatSurfaceConstraint
(force = mass/dt*lambda*(d.n_hat)n_hat, i.e. displacement lambda*(d.n_hat)n_hat) and
ConvexPolygonConstraint, both lambda=0.1, auto-bound by TF on every SurfaceType. Re-add them
as two per-surface force kernels in physics_warp and re-run.

Usage (faithful defaults; everything overridable):
    pixi run gpu-stability                              # n=4, 100k steps, sigma=0.5, demixed IC
    pixi run gpu-stability --steps 5000                 # quick smoke / per-step timing
    pixi run gpu-stability --sigma 0.2 --ic mixed       # Fig-1E mixed-start variant
    pixi run gpu-stability --steps 100000 --check-every 5000 --csv rnr/exports/gpu_stability.csv
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import warp as wp

import tissue_forge as tf
from tissue_forge.models.vertex import solver as tfv

from rnr.gpu import engine as E
from rnr.gpu import physics_csr as P
from rnr.gpu.device_mesh import PaddedMesh


def _audit(g, box, body_type):
    """One O(mesh) health snapshot off the device. Returns a dict of the audited signals.

    A single `from_warp` serves both the topology audit and the geometry/metric so the long
    loop only pays one host copy per check."""
    pm = PaddedMesh.from_warp(g)
    problems = pm.check_consistency()
    geom = P.compute_geometry(pm, box)
    vol = geom.bvol[:pm.nb]
    pos = pm.vert_pos[:pm.n_v_used]
    het, total = E.het_contact_fraction(pm, body_type)
    return dict(
        problems=problems,
        nv=pm.n_v_used, ns=pm.n_s_used,
        vol_min=float(vol.min()), vol_max=float(vol.max()),
        finite=bool(np.all(np.isfinite(pos))),
        het=(het / total if total else 0.0),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4, help="BCC foam size (2*n^3 cells); 4 -> 128")
    ap.add_argument("--steps", type=int, default=100_000, help="forward steps (Fig 1E/1F length)")
    ap.add_argument("--sigma", type=float, default=0.5, help="heterotypic tension (Fig1: 0.1/0.2/0.5)")
    ap.add_argument("--v-active", type=float, default=0.1, help="active self-propulsion speed (=KT)")
    ap.add_argument("--dt", type=float, default=5e-3, help="time step (oracle 1e-3..1e-2)")
    ap.add_argument("--dr", type=float, default=1.0, help="director rotational diffusion")
    ap.add_argument("--lth", type=float, default=1e-3, help="reconnection trigger + Okuda placement gap")
    ap.add_argument("--interval", type=int, default=0, help="reconnect cadence in steps (0 -> round(0.01/dt))")
    ap.add_argument("--ic", choices=["demixed", "mixed"], default="demixed",
                    help="demixed = Fig-1F slab (stays demixed?); mixed = Fig-1E random (demixes?)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--check-every", type=int, default=5000, help="audit cadence in steps")
    ap.add_argument("--vol-factor", type=float, default=20.0,
                    help="degeneration bound: fail if any cell volume leaves [v0/f, f*v0] "
                         "(a stiff K_V=10 foam holds V near V0; f=20 flags a ballooned/collapsed cell)")
    ap.add_argument("--headroom", type=int, default=4000, help="vertex/surface slot capacity")
    ap.add_argument("--rebuild-foam", action="store_true",
                    help="ignore any cached foam and rebuild it de novo (then overwrite the cache)")
    ap.add_argument("--csv", default="", help="optional path to dump the audit timeline")
    ap.add_argument("--stop-on-fail", action="store_true", default=True,
                    help="halt at the first hard failure (default on)")
    ap.add_argument("--no-stop-on-fail", dest="stop_on_fail", action="store_false")
    ap.add_argument("--reconnect", action="store_true", default=True,
                    help="run RNR reconnection each interval (default on)")
    ap.add_argument("--no-reconnect", dest="reconnect", action="store_false",
                    help="force+integrate only -- isolates the integrator from the RNR pipeline")
    args = ap.parse_args()

    interval = args.interval if args.interval > 0 else max(1, round(0.01 / args.dt))

    wp.init()
    cuda = [d for d in wp.get_devices() if d.is_cuda]
    if not cuda:
        print("FATAL: no CUDA device -- the GPU stability run needs the RTX 5090.")
        sys.exit(2)
    dev = cuda[0]

    # Build-once / load-many: the unit-cell foam is cached to disk (foam_cache). On a cache HIT
    # we load the saved geometry and skip the TF foam builder -- AND TF init -- entirely; on a
    # MISS we init TF and build de novo (then save). The build half is the VERBATIM Gate-E setup
    # (TF foam -> CSR -> unit scale); upload_unit_foam is the shared device half.
    from rnr.gpu.foam_cache import load_or_build

    def _build_host():
        # only runs on a cache miss -- defer the (slow) TF init here so a hit needs no TF at all.
        # mirrors conftest.vsolver / gpu_csr_demo so the foam is identical to the validated one
        tf.init(windowless=True, dim=[60., 60., 60.], cutoff=5.0, dt=0.001)
        tfv.init()
        tfv.MeshSolver.get().get_mesh().quality = None
        from rnr.tests.test_gpu_engine import _build_unit_foam_host
        return _build_unit_foam_host(n=args.n, headroom=args.headroom, ic=args.ic)

    g, phys, body_type, box, v0, a0 = load_or_build(
        dev, n=args.n, ic=args.ic, headroom=args.headroom,
        build_host_fn=_build_host, rebuild=args.rebuild_foam)
    params = P.PhysParams(box=box, kv=10.0, v0=v0, ka=1.0, a0=a0,
                          sigma=args.sigma, v_active=args.v_active)
    cap_v = int(g["cap_v"])
    cap_s = int(g["cap_s"])

    print(f"[gpu-stability] dev={dev} n={args.n} cells={body_type.shape[0]} "
          f"verts(start)={int(g['n_used'].numpy()[0])} cap_v={cap_v} cap_s={cap_s}")
    print(f"[gpu-stability] FAITHFUL params: kv=10 v0={v0:.3f} ka=1 a0={a0:.3f} sigma={args.sigma} "
          f"v_active={args.v_active} dt={args.dt} dr={args.dr} Lth={args.lth} interval={interval} "
          f"ic={args.ic} steps={args.steps}")

    start = _audit(g, box, body_type)
    print(f"[gpu-stability] step 0: het={start['het']:.4f} vol[{start['vol_min']:.3f},"
          f"{start['vol_max']:.3f}] nv={start['nv']} ns={start['ns']} "
          f"consistent={not start['problems']}")

    rows = [dict(step=0, het=start["het"], nv=start["nv"], ns=start["ns"],
                 vol_min=start["vol_min"], vol_max=start["vol_max"],
                 recon_i=0, recon_h=0, nv_max=start["nv"], n_problems=len(start["problems"]),
                 sec=0.0)]

    recon_i = recon_h = 0
    nv_max = start["nv"]
    failed_at = None
    vol_hi = args.vol_factor * v0
    vol_lo = v0 / args.vol_factor
    recon_hist = []          # cumulative (I+H) at each checkpoint -- to spot a topological freeze
    t0 = time.perf_counter()

    for step in range(args.steps):
        rep = E.forward_step(g, phys, params, args.dt, args.dr, seed=args.seed, step=step,
                             threshold=args.lth, dl_th=args.lth, reconnect=args.reconnect,
                             interval=interval, compact=True, max_rounds=8)
        recon_i += rep["i"]
        recon_h += rep["h"]
        nv_max = max(nv_max, rep["nv"])

        # cheap per-step guards (no host copy): slot capacity is the only thing a single step
        # can blow without an audit -- catch it immediately rather than at the next checkpoint.
        if rep["nv"] >= cap_v or rep["ns"] >= cap_s:
            failed_at = step
            print(f"\n[FAIL] step {step}: slot capacity exhausted nv={rep['nv']}/{cap_v} "
                  f"ns={rep['ns']}/{cap_s} -- compaction is not bounding slots.")
            if args.stop_on_fail:
                break

        if (step + 1) % args.check_every == 0 or step == args.steps - 1:
            sec = time.perf_counter() - t0
            a = _audit(g, box, body_type)
            recon_hist.append(recon_i + recon_h)
            degen = a["vol_max"] > vol_hi or a["vol_min"] < vol_lo
            bad = bool(a["problems"]) or not a["finite"] or a["vol_min"] <= 0.0 or degen
            tag = "FAIL" if bad else "ok"
            print(f"[{tag}] step {step+1:>7} ({sec:6.1f}s, {1e3*sec/(step+1):.2f} ms/step) "
                  f"het={a['het']:.4f} vol[{a['vol_min']:.3f},{a['vol_max']:.3f}] "
                  f"nv={a['nv']} ns={a['ns']} nv_max={nv_max} "
                  f"recon I/H={recon_i}/{recon_h} problems={len(a['problems'])}")
            rows.append(dict(step=step + 1, het=a["het"], nv=a["nv"], ns=a["ns"],
                             vol_min=a["vol_min"], vol_max=a["vol_max"],
                             recon_i=recon_i, recon_h=recon_h, nv_max=nv_max,
                             n_problems=len(a["problems"]), sec=sec))
            if bad:
                failed_at = step
                if a["problems"]:
                    print("        first problems:", a["problems"][:5])
                if not a["finite"]:
                    print("        non-finite vertex positions (force blow-up)")
                if a["vol_min"] <= 0.0:
                    print(f"        a cell inverted: min volume = {a['vol_min']:.3e}")
                if degen:
                    print(f"        cell-volume DEGENERATION: vol[{a['vol_min']:.3e},"
                          f"{a['vol_max']:.3e}] left healthy band [{vol_lo:.3g},{vol_hi:.3g}] "
                          f"(a cell ballooned/collapsed -- the mesh is geometrically broken "
                          f"even though adjacency is still self-consistent)")
                if args.stop_on_fail:
                    break

    total_sec = time.perf_counter() - t0
    done_steps = (failed_at + 1) if failed_at is not None else args.steps
    final = _audit(g, box, body_type)

    final_ok = (failed_at is None and not final["problems"] and final["finite"]
                and vol_lo <= final["vol_min"] and final["vol_max"] <= vol_hi)
    # a healthy sort keeps reconnecting; a flat cumulative over the back half = topological freeze
    frozen = len(recon_hist) >= 3 and recon_hist[-1] == recon_hist[len(recon_hist) // 2]

    print("\n" + "=" * 78)
    if final_ok and not frozen:
        print(f"VERDICT: STABLE -- {args.steps} steps, mesh valid throughout. "
              f"A faithful GPU sort runs to completion.")
    elif final_ok and frozen:
        print(f"VERDICT: STABLE BUT FROZEN -- mesh stayed valid + volume-bounded, but "
              f"reconnections stalled (no topology change over the back half). The sort did "
              f"not progress; suspect the trigger/active-drive scale, not a blow-up.")
    else:
        print(f"VERDICT: DEGRADED at step ~{failed_at} (ran {done_steps}/{args.steps}). "
              f"Cell volume left [{vol_lo:.3g},{vol_hi:.3g}] (ballooned/collapsed cell). "
              f"Likely fixes: smaller dt, or re-add Flat/Convex regularizers (module docstring).")
    print(f"  het: {start['het']:.4f} -> {final['het']:.4f}   ({args.ic} IC)")
    print(f"  reconnections: I->H={recon_i}  H->I={recon_h}   nv_max={nv_max}/{cap_v}"
          + ("   [FROZEN after early steps]" if frozen else ""))
    print(f"  final volume range: [{final['vol_min']:.3f}, {final['vol_max']:.3f}]  "
          f"healthy band [{vol_lo:.3g}, {vol_hi:.3g}]")
    print(f"  wall: {total_sec:.1f}s  ({1e3*total_sec/max(1,done_steps):.2f} ms/step)")
    print("=" * 78)

    if args.csv:
        import csv as _csv
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  timeline -> {args.csv}")

    sys.exit(1 if failed_at is not None else 0)


if __name__ == "__main__":
    main()
