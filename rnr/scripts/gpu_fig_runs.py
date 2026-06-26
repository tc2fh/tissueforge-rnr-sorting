"""Orchestrate the GPU ensemble for Manning2024 Fig 1E + 1F at PAPER SCALE (N=2000).

Runs gpu_stability.py for every (sigma, IC, seed) with a concurrency cap, writing one timeline
CSV per run (the per-checkpoint `het`, from which the plotter forms DP = 1 - 2*het). The GPU
engine underutilizes the RTX 5090 at N=2000 (host-bound), so several runs go concurrently for a
~3x aggregate speedup; thread pools are capped (BLAS=1) to avoid host contention (memory
tf-threading-for-sweeps).

Replicate convention: a COMMON cached foam per IC (geometry + type assignment fixed, rng_seed=3);
the seeds vary the active-drive (director) noise realization -> independent sort trajectories from
a shared start. This spends the compute budget on length, not redundant ~10-min foam rebuilds.

Fig 1E: sigma in {0.1, 0.2, 0.5}, IC=mixed (DP~0 -> rises, sigma-ordered).
Fig 1F: sigma=0.5, IC=demixed (starts at DP_max -> holds); paired with the sigma=0.5 mixed runs.

Usage:
  pixi run python rnr/scripts/gpu_fig_runs.py [STEPS] [CONC] [SEEDS] [DT] [--captured] [--force]
  defaults: STEPS=400000 CONC=6 SEEDS=7,8,9 DT=0.01   (~3h for 12 runs)
  --captured: drive each sim with the CUDA-graph-captured step (byte-identical, less host overhead).
Resumable: a run whose CSV already reaches STEPS is skipped (use --force to rerun all).
CSVs: rnr/exports/gpu_sort_n10_S{sigma}_{ic}_dt{dt}_seed{seed}.csv
"""
import csv
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXPORTS = os.path.join(ROOT, "rnr", "exports")
GPU_STAB = os.path.join(HERE, "gpu_stability.py")

N = 10
SIGMAS = [0.1, 0.2, 0.5]
FORCE = "--force" in sys.argv
# Drive each per-sim subprocess with the CUDA-graph-captured step (gpu_stability --captured):
# byte-identical, removes the per-step host overhead -> each concurrent sim runs faster + contends
# the host less (capture_warp.CapturedStep / docs/2026-06-26_cuda-graph-experiment-scope.md).
CAPTURED = "--captured" in sys.argv
argv = [a for a in sys.argv[1:] if not a.startswith("--")]
STEPS = int(argv[0]) if len(argv) > 0 else 400_000
CONC = int(argv[1]) if len(argv) > 1 else 6
SEEDS = [int(s) for s in argv[2].split(",")] if len(argv) > 2 else [7, 8, 9]
DT = argv[3] if len(argv) > 3 else "0.01"
CHECK_EVERY = max(1000, STEPS // 80)     # ~80 points along each curve


def csv_path(sigma, ic, seed):
    return os.path.join(EXPORTS, f"gpu_sort_n{N}_S{sigma:g}_{ic}_dt{DT}_seed{seed}.csv")


def is_complete(path):
    """True if `path` exists and its last row reached STEPS (so we can resume/skip)."""
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            last = None
            for last in csv.DictReader(f):
                pass
        return last is not None and int(float(last["step"])) >= STEPS
    except Exception:
        return False


def build_jobs():
    jobs = []
    for sigma in SIGMAS:                                  # Fig 1E: mixed
        for seed in SEEDS:
            jobs.append((sigma, "mixed", seed))
    for seed in SEEDS:                                    # Fig 1F: demixed at sigma=0.5
        jobs.append((0.5, "demixed", seed))
    return jobs


def cmd_for(sigma, ic, seed):
    out = csv_path(sigma, ic, seed)
    cmd = [sys.executable, GPU_STAB, "--n", str(N), "--steps", str(STEPS), "--dt", DT,
           "--ic", ic, "--sigma", str(sigma), "--seed", str(seed),
           "--check-every", str(CHECK_EVERY), "--csv", out]
    if CAPTURED:
        cmd.append("--captured")
    return cmd, out


def main():
    jobs = build_jobs()
    todo = [j for j in jobs if FORCE or not is_complete(csv_path(*j))]
    done_already = len(jobs) - len(todo)
    print(f"[fig-runs] N={N} (2000 cells) steps={STEPS} (t={int(STEPS) * float(DT):g}) "
          f"dt={DT} conc={CONC} seeds={SEEDS}")
    print(f"[fig-runs] {len(jobs)} jobs, {done_already} already complete, {len(todo)} to run")
    if not todo:
        print("[fig-runs] nothing to do.")
        return

    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    logdir = os.path.join(EXPORTS, "fig_run_logs")
    os.makedirs(logdir, exist_ok=True)
    t0 = time.perf_counter()
    running = {}      # proc -> (job, logfile_handle, start_t)
    queue = list(todo)
    n_done = 0

    def launch(job):
        cmd, out = cmd_for(*job)
        sigma, ic, seed = job
        lf = open(os.path.join(logdir, f"S{sigma:g}_{ic}_seed{seed}.log"), "w")
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        running[p] = (job, lf, time.perf_counter())
        print(f"[fig-runs] +launch sigma={sigma:g} {ic} seed={seed} (pid {p.pid}); "
              f"{len(running)} running, {len(queue)} queued", flush=True)

    while queue and len(running) < CONC:
        launch(queue.pop(0))
    while running:
        time.sleep(5.0)
        for p in list(running):
            if p.poll() is None:
                continue
            job, lf, st = running.pop(p)
            lf.close()
            n_done += 1
            sigma, ic, seed = job
            ok = (p.returncode == 0) and is_complete(csv_path(*job))
            dt_min = (time.perf_counter() - st) / 60.0
            print(f"[fig-runs] -done   sigma={sigma:g} {ic} seed={seed} rc={p.returncode} "
                  f"{'OK' if ok else 'CHECK'} ({dt_min:.1f} min)  [{n_done}/{len(todo)}]", flush=True)
            if queue:
                launch(queue.pop(0))
    print(f"[fig-runs] ALL DONE in {(time.perf_counter() - t0) / 60.0:.1f} min", flush=True)


if __name__ == "__main__":
    main()
