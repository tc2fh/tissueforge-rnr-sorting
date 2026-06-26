"""GPU scalability sweep of the Warp 3D-vertex cell-sorting engine (rnr/gpu).

Measures how the GPU engine scales, two ways:

  STUDY 1 (serial, no concurrency): one sim at a time, foam size n = 8,12,...,48
    (cells = 2*n^3 -> 1,024 .. 221,184). Per n: ms/step, #cells, GPU util, VRAM.

  STUDY 2 (concurrency): for n = 16,24,32, run K independent sims at once
    (K = 1,2,4,6,8,12,24). Per (n,K): per-sim ms/step (under contention),
    AGGREGATE throughput (sum of per-sim steps/s), GPU util, total VRAM.

WHY a timed window, not the full 100k production length: ms/step, util and VRAM are
RATES -- they converge within a few hundred steps. The full 100k only adds long-run
drift (as cells coalesce, reconnections thin out and steps speed up) and, for the
24-way concurrency point, would run for hours. So each config does `--warmup` steps
(kernel JIT + initial orient-heal + reconnection ramp) then TIMES `--timed` steps with
a CUDA sync around the window. Pure `engine.forward_step` -- no per-step audit (the
het-fraction metric is a Python loop over every surface, which dominates wall-clock at
scale and is NOT the GPU step cost; see gpu_stability.py's contaminated timing).

HEADROOM SCALES WITH MESH SIZE. The bump-allocated vertex/surface slots
(cap = n_used + headroom) must hold one parallel reconnection round's births, which
grow with mesh size. The fixed default (4000) overflows above ~n=32 -> CUDA error 700
(illegal memory access). We size headroom = max(4000, 0.10*n_verts), which clears n=48.

CONCURRENCY MEASUREMENT. Each (n,K) launches K worker subprocesses. A file barrier in a
scratch dir makes all K start their TIMED window simultaneously (after every worker has
warmed up), so each worker's self-timed ms/step is measured under full K-way contention.
Aggregate throughput = sum over workers of (1000/ms_per_step). The orchestrator samples
nvidia-smi (global VRAM + SM util) across the window. A VRAM guard skips any (n,K) that
would exceed ~93% of the card (K independent CUDA contexts each carry ~mesh+context MiB).

Usage:
  pixi run python rnr/scripts/gpu_scalability.py                 # both studies, defaults
  pixi run python rnr/scripts/gpu_scalability.py --study 1       # serial sweep only
  pixi run python rnr/scripts/gpu_scalability.py --study 2       # concurrency only
  pixi run python rnr/scripts/gpu_scalability.py --plot-only     # re-plot existing CSVs
  # tuning: --warmup 300 --timed 2000 --dt 0.01 --ic mixed
  #         --study1-ns 8,12,...,48  --study2-ns 16,24,32  --study2-conc 1,2,4,6,8,12,24
Outputs (rnr/exports/):
  gpu_scalability_study1.csv / _study2.csv   + gpu_scalability_study1.png / _study2.png
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
EXPORTS = os.path.join(ROOT, "rnr", "exports")
SCRATCH = "/tmp/claude-1000/-home-tien-Work-SegoLab-VertexModeling/e3586485-f566-410b-955f-e10757735218/scratchpad"

# measured: n_verts ~= 6.1 * cells across the whole range (n=8..48), very tight.
NV_PER_CELL = 6.1
GPU_TOTAL_MIB = 32607          # RTX 5090
VRAM_BASE_MIB = 2810           # system/desktop floor before a sim allocates (measured)
VRAM_GUARD_FRAC = 0.93         # skip an (n,K) whose predicted footprint exceeds this


def cells_of(n):
    return 2 * n ** 3


def est_nv(n):
    return int(NV_PER_CELL * cells_of(n))


def headroom_for(n):
    """Slots above n_used. Must hold one parallel reconnection round's births (grows with
    mesh size); the fixed 4000 overflows past ~n=32. 10% of n_verts clears n=48."""
    return max(4000, int(0.10 * est_nv(n)))


# ============================================================ WORKER (one sim) ====
def run_worker(args):
    """One process: load/scale the foam, warm up, (barrier), TIME a window of pure
    forward_step, print one machine-readable WORKER_RESULT json line."""
    import warp as wp
    from rnr.gpu import engine as E
    from rnr.gpu import physics_csr as P
    from rnr.gpu.foam_cache import load_or_build

    wp.init()
    cuda = [d for d in wp.get_devices() if d.is_cuda]
    if not cuda:
        print("WORKER_RESULT " + json.dumps(dict(error="no CUDA device")), flush=True)
        sys.exit(2)
    dev = cuda[0]
    headroom = headroom_for(args.n)
    interval = max(1, round(0.01 / args.dt))

    def _build_host():
        # cache miss only: defer the (slow) TF foam build here so a hit needs no TF at all
        import tissue_forge as tf
        from tissue_forge.models.vertex import solver as tfv
        tf.init(windowless=True, dim=[60., 60., 60.], cutoff=5.0, dt=0.001)
        tfv.init()
        tfv.MeshSolver.get().get_mesh().quality = None
        from rnr.tests.test_gpu_engine import _build_unit_foam_host
        return _build_unit_foam_host(n=args.n, headroom=headroom, ic=args.ic)

    g, phys, body_type, box, v0, a0 = load_or_build(
        dev, n=args.n, ic=args.ic, headroom=headroom, build_host_fn=_build_host)
    params = P.PhysParams(box=box, kv=10.0, v0=v0, ka=1.0, a0=a0, sigma=0.5, v_active=0.1)
    ncells = int(body_type.shape[0])
    nv0 = int(g["n_used"].numpy()[0])

    if args.captured:
        # CUDA-graph-captured step (the productionized path): NO per-step host readback -- pure async
        # graph replay. Construction warms up + captures internally; step() ignores its arg (uses an
        # internal counter). This is what lets us test whether capture changes concurrency scaling.
        from rnr.gpu.capture_warp import CapturedStep
        cs = CapturedStep(g, phys, params, args.dt, 1.0, seed=7, threshold=1e-3, dl_th=1e-3,
                          reconnect=True, interval=interval)
        _ctr = [cs.next_step]

        def step(_):
            cs.step(_ctr[0]); _ctr[0] += 1
    else:
        def step(s):
            E.forward_step(g, phys, params, args.dt, 1.0, seed=7, step=s, threshold=1e-3,
                           dl_th=1e-3, reconnect=True, interval=interval, compact=True, max_rounds=8)

    for s in range(args.warmup):                                   # JIT + heal + ramp
        step(s)
    wp.synchronize_device(dev)

    if args.barrier_dir:                                           # release together under contention
        open(os.path.join(args.barrier_dir, f"ready_{args.worker_id}"), "w").close()
        t_wait = time.time()
        while len(glob.glob(os.path.join(args.barrier_dir, "ready_*"))) < args.barrier_n:
            if time.time() - t_wait > 240:
                break
            time.sleep(0.05)

    wp.synchronize_device(dev)
    t0 = time.perf_counter()
    for s in range(args.timed):
        step(args.warmup + s)
    wp.synchronize_device(dev)
    el = time.perf_counter() - t0

    print("WORKER_RESULT " + json.dumps(dict(
        n=args.n, cells=ncells, verts=nv0, headroom=headroom, timed=args.timed,
        ms_per_step=1e3 * el / args.timed, steps_per_s=args.timed / el,
        worker_id=args.worker_id)), flush=True)


# =================================================== ORCHESTRATOR (sampling) ====
def _smi():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"]).decode().strip().splitlines()
    mem, util = out[0].split(", ")
    return int(mem), int(util)


def _sampler(stop_evt, samples):
    while not stop_evt.is_set():
        try:
            samples.append((time.time(),) + _smi())
        except Exception:
            pass
        time.sleep(0.2)


def predict_footprint_mib(n, measured):
    """Per-sim VRAM footprint (context+mesh). Prefer the measured K=1 value for this n;
    else a linear-in-cells fit of the study-1 probe (0.59 + 8.4e-6*cells, GiB)."""
    if n in measured:
        return measured[n]
    return 1024.0 * (0.59 + 8.4e-6 * cells_of(n))


def _prebuild_foam(n, args):
    """Ensure foam size n is cached on disk BEFORE a concurrent group launches -- else K
    workers would each fire a TF build at once. No-op (fast) if the cache already exists."""
    from rnr.gpu.foam_cache import cache_path
    if cache_path(n, args.ic).exists():
        return
    print(f"   [prebuild] n={n}: foam cache missing -> building once (TF) ...", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--n", str(n), "--ic", args.ic,
           "--dt", str(args.dt), "--warmup", "1", "--timed", "1"]
    subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_config(n, K, args, measured_footprint):
    """Launch K workers for foam size n, time them under a shared barrier, sample nvidia-smi.
    Returns a result row dict (or a 'skipped' row if it would exceed VRAM)."""
    if K > 1:
        _prebuild_foam(n, args)
    foot = predict_footprint_mib(n, measured_footprint)
    predicted = K * foot + VRAM_BASE_MIB
    if predicted > VRAM_GUARD_FRAC * GPU_TOTAL_MIB:
        print(f"   [skip] n={n} K={K}: predicted {predicted:.0f} MiB > "
              f"{VRAM_GUARD_FRAC:.0%} of {GPU_TOTAL_MIB} MiB (per-sim ~{foot:.0f})", flush=True)
        return dict(n=n, cells=cells_of(n), K=K, skipped=1,
                    reason=f"predicted {predicted:.0f}MiB VRAM")

    bdir = os.path.join(SCRATCH, f"barrier_n{n}_K{K}")
    shutil.rmtree(bdir, ignore_errors=True)
    os.makedirs(bdir, exist_ok=True)
    logdir = os.path.join(EXPORTS, "scalability_logs")
    os.makedirs(logdir, exist_ok=True)

    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    procs, logs = [], []
    for i in range(K):
        cmd = [sys.executable, os.path.abspath(__file__), "--worker",
               "--n", str(n), "--ic", args.ic, "--dt", str(args.dt),
               "--warmup", str(args.warmup), "--timed", str(args.timed),
               "--barrier-dir", bdir, "--barrier-n", str(K), "--worker-id", str(i)]
        if args.captured:
            cmd.append("--captured")
        lf = open(os.path.join(logdir, f"n{n}_K{K}_w{i}.log"), "w")
        procs.append(subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env))
        logs.append(lf.name)

    samples, stop = [], threading.Event()
    th = threading.Thread(target=_sampler, args=(stop, samples), daemon=True)
    th.start()
    t_launch = time.time()

    # detect barrier release: when all K workers have warmed up (ready files present)
    t_release = None
    while any(p.poll() is None for p in procs):
        if t_release is None and len(glob.glob(os.path.join(bdir, "ready_*"))) >= K:
            t_release = time.time()
        time.sleep(0.1)
    for p in procs:
        p.wait()
    stop.set()
    th.join()

    # parse worker results
    results = []
    for lp in logs:
        with open(lp) as fh:
            for line in fh:
                if line.startswith("WORKER_RESULT "):
                    results.append(json.loads(line[len("WORKER_RESULT "):]))
    ok = [r for r in results if "error" not in r]
    if len(ok) < K:
        print(f"   [warn] n={n} K={K}: only {len(ok)}/{K} workers returned a result", flush=True)
    if not ok:
        return dict(n=n, cells=cells_of(n), K=K, skipped=1, reason="all workers failed")

    # util/VRAM over the timed window [release, last-done]; fall back to whole run if no release seen
    t_lo = t_release if t_release else t_launch
    win = [s for s in samples if s[0] >= t_lo] or samples
    vram_peak = max(s[1] for s in win)
    utils = [s[2] for s in win]
    util_mean = sum(utils) / len(utils)
    util_peak = max(utils)

    ms_each = [r["ms_per_step"] for r in ok]
    per_sim_ms = sum(ms_each) / len(ms_each)
    agg_steps_s = sum(r["steps_per_s"] for r in ok)        # total sim-steps/s across all K
    row = dict(
        n=n, cells=cells_of(n), verts=ok[0]["verts"], headroom=ok[0]["headroom"], K=K,
        per_sim_ms_per_step=round(per_sim_ms, 3),
        per_sim_steps_per_s=round(1e3 / per_sim_ms, 2),
        agg_steps_per_s=round(agg_steps_s, 2),
        speedup_vs_1=0.0,                                  # filled by caller (needs K=1 row)
        vram_peak_mib=vram_peak, vram_per_sim_mib=round((vram_peak - VRAM_BASE_MIB) / K, 1),
        util_mean_pct=round(util_mean, 1), util_peak_pct=util_peak,
        n_workers_ok=len(ok), skipped=0)
    shutil.rmtree(bdir, ignore_errors=True)
    print(f"   n={n:>2} K={K:>2}: per-sim {per_sim_ms:6.2f} ms/step | agg {agg_steps_s:8.1f} steps/s "
          f"| VRAM {vram_peak:>6} MiB | util {util_mean:4.1f}% (pk {util_peak}%) "
          f"| {len(ok)}/{K} ok", flush=True)
    return row


# ================================================================== STUDIES ====
def study1(args, measured_footprint):
    ns = [int(x) for x in args.study1_ns.split(",")]
    print(f"\n=== STUDY 1: serial sweep (K=1), n={ns} "
          f"cells={[cells_of(n) for n in ns]} ===", flush=True)
    rows = []
    for n in ns:
        r = run_config(n, 1, args, measured_footprint)
        rows.append(r)
        if not r.get("skipped"):
            measured_footprint[n] = r["vram_peak_mib"] - VRAM_BASE_MIB   # feed study 2's guard
    _write_csv(os.path.join(EXPORTS, "gpu_scalability_study1.csv"), rows)
    return rows


def study2(args, measured_footprint):
    ns = [int(x) for x in args.study2_ns.split(",")]
    Ks = [int(x) for x in args.study2_conc.split(",")]
    print(f"\n=== STUDY 2: concurrency sweep, n={ns}, K={Ks} ===", flush=True)
    rows = []
    for n in ns:
        base = None
        for K in Ks:
            r = run_config(n, K, args, measured_footprint)
            if not r.get("skipped"):
                if K == 1 or base is None:
                    base = r["per_sim_steps_per_s"]
                    measured_footprint.setdefault(n, r["vram_peak_mib"] - VRAM_BASE_MIB)
                r["speedup_vs_1"] = round(r["agg_steps_per_s"] / base, 2) if base else 0.0
            rows.append(r)
    _write_csv(os.path.join(EXPORTS, f"gpu_scalability_study2{args.out_suffix}.csv"), rows)
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    head = [k for k in ["n", "cells", "verts", "K", "per_sim_ms_per_step", "per_sim_steps_per_s",
                        "agg_steps_per_s", "speedup_vs_1", "vram_peak_mib", "vram_per_sim_mib",
                        "util_mean_pct", "util_peak_pct", "headroom", "n_workers_ok",
                        "skipped", "reason"] if k in keys]
    head += [k for k in keys if k not in head]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=head)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"   -> {path}", flush=True)


# ================================================================== PLOTTING ====
def plot_all():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p1 = os.path.join(EXPORTS, "gpu_scalability_study1.csv")
    p2 = os.path.join(EXPORTS, "gpu_scalability_study2.csv")

    if os.path.exists(p1):
        rows = [r for r in csv.DictReader(open(p1)) if r.get("skipped") in ("0", "", None)]
        if rows:
            cells = [int(r["cells"]) for r in rows]
            ms = [float(r["per_sim_ms_per_step"]) for r in rows]
            sps = [float(r["per_sim_steps_per_s"]) for r in rows]
            vram = [float(r["vram_peak_mib"]) - VRAM_BASE_MIB for r in rows]
            util = [float(r["util_mean_pct"]) for r in rows]
            upk = [float(r["util_peak_pct"]) for r in rows]
            fig, ax = plt.subplots(2, 2, figsize=(13, 9))
            ax[0, 0].loglog(cells, ms, "o-"); ax[0, 0].set_title("per-step time vs scale")
            ax[0, 0].set_xlabel("cells"); ax[0, 0].set_ylabel("ms / step"); ax[0, 0].grid(True, which="both", alpha=.3)
            ax[0, 1].loglog(cells, sps, "o-", color="C1"); ax[0, 1].set_title("throughput vs scale")
            ax[0, 1].set_xlabel("cells"); ax[0, 1].set_ylabel("steps / s"); ax[0, 1].grid(True, which="both", alpha=.3)
            ax[1, 0].plot(cells, vram, "o-", color="C2"); ax[1, 0].set_title("VRAM (sim footprint) vs scale")
            ax[1, 0].set_xlabel("cells"); ax[1, 0].set_ylabel("MiB"); ax[1, 0].grid(True, alpha=.3)
            ax[1, 1].plot(cells, util, "o-", label="mean"); ax[1, 1].plot(cells, upk, "s--", label="peak", alpha=.6)
            ax[1, 1].set_title("GPU utilization vs scale"); ax[1, 1].set_xlabel("cells")
            ax[1, 1].set_ylabel("SM util %"); ax[1, 1].legend(); ax[1, 1].grid(True, alpha=.3)
            fig.suptitle("GPU 3D-vertex engine — Study 1: serial scalability (one sim)", fontsize=13)
            fig.tight_layout()
            out = os.path.join(EXPORTS, "gpu_scalability_study1.png")
            fig.savefig(out, dpi=120); plt.close(fig)
            print(f"   -> {out}")

    if os.path.exists(p2):
        rows = [r for r in csv.DictReader(open(p2)) if r.get("skipped") in ("0", "", None)]
        if rows:
            ns = sorted({int(r["n"]) for r in rows})
            fig, ax = plt.subplots(2, 2, figsize=(13, 9))
            for n in ns:
                rr = sorted([r for r in rows if int(r["n"]) == n], key=lambda r: int(r["K"]))
                K = [int(r["K"]) for r in rr]
                agg = [float(r["agg_steps_per_s"]) for r in rr]
                psm = [float(r["per_sim_ms_per_step"]) for r in rr]
                spd = [float(r["speedup_vs_1"]) for r in rr]
                vram = [float(r["vram_peak_mib"]) for r in rr]
                lab = f"n={n} ({cells_of(n)} cells)"
                ax[0, 0].plot(K, agg, "o-", label=lab)
                ax[0, 1].plot(K, spd, "o-", label=lab)
                ax[1, 0].plot(K, psm, "o-", label=lab)
                ax[1, 1].plot(K, vram, "o-", label=lab)
            ax[0, 0].set_title("aggregate throughput vs concurrency")
            ax[0, 0].set_xlabel("K (concurrent sims)"); ax[0, 0].set_ylabel("total steps / s")
            ax[0, 1].set_title("aggregate speedup vs K=1"); ax[0, 1].set_xlabel("K"); ax[0, 1].set_ylabel("× K=1")
            ax[0, 1].plot([1, max(int(r["K"]) for r in rows)], [1, max(int(r["K"]) for r in rows)],
                          "k:", alpha=.4, label="ideal (linear)")
            ax[1, 0].set_title("per-sim ms/step (contention) vs K"); ax[1, 0].set_xlabel("K"); ax[1, 0].set_ylabel("ms/step")
            ax[1, 1].set_title("total VRAM vs K"); ax[1, 1].set_xlabel("K"); ax[1, 1].set_ylabel("MiB")
            ax[1, 1].axhline(GPU_TOTAL_MIB, color="r", ls="--", alpha=.5, label="card total")
            for a in ax.flat:
                a.grid(True, alpha=.3); a.legend(fontsize=8)
            fig.suptitle("GPU 3D-vertex engine — Study 2: concurrency scalability", fontsize=13)
            fig.tight_layout()
            out = os.path.join(EXPORTS, "gpu_scalability_study2.png")
            fig.savefig(out, dpi=120); plt.close(fig)
            print(f"   -> {out}")


# ====================================================================== MAIN ====
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--captured", action="store_true",
                    help="drive with the CUDA-graph CapturedStep (no per-step host sync) "
                         "instead of eager forward_step -- to test concurrency under capture")
    ap.add_argument("--out-suffix", default="", help="suffix for study2 CSV (e.g. _captured)")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--ic", default="mixed", help="foam IC (timing is IC-independent; mixed has caches)")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=300, help="untimed steps (JIT/heal/ramp)")
    ap.add_argument("--timed", type=int, default=2000, help="timed-window steps")
    ap.add_argument("--barrier-dir", default="")
    ap.add_argument("--barrier-n", type=int, default=1)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--study", choices=["1", "2", "both"], default="both")
    ap.add_argument("--study1-ns", default="8,12,16,20,24,28,32,36,40,44,48")
    ap.add_argument("--study2-ns", default="16,24,32")
    ap.add_argument("--study2-conc", default="1,2,4,6,8,12,24")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if args.worker:
        return run_worker(args)

    os.makedirs(EXPORTS, exist_ok=True)
    if args.plot_only:
        return plot_all()

    t0 = time.time()
    measured = {}                # n -> per-sim VRAM footprint MiB (study 1 feeds study 2 guard)
    if args.study in ("1", "both"):
        study1(args, measured)
    if args.study in ("2", "both"):
        study2(args, measured)
    plot_all()
    print(f"\n[gpu-scalability] ALL DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
