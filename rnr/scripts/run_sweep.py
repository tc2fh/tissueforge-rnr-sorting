"""Larger-scale Manning2024 Fig 1E/1F sweep at a chosen system size M (this run: M=8, N=512).

WHY (2026-06-23). The committed figures are M=6 / N=216 / t=100. At that scale the paper's EXACT
count-based DP/DP_max stays ~0 for the MIXED IC (Fig 1E): neighbour-count demixing needs DOMAIN
formation, which is finite-size + run-length limited (PORTING_NOTES §6l/§6m; paper N>=512, t~10000).
This sweep scales BOTH knobs -- N=512 (M=8) and t into the thousands -- to test whether the mixed-IC
count-DP starts to RESOLVE once domains can form, i.e. whether the remaining gap is purely scale.

THREADING (measured 2026-06-23, Ryzen 9 9950X3D, linux-64) -- the load-bearing fix:
  * `tf.init(threads=1)` does NOT cap TF's thread pool on this build: every process spawns ~128
    threads and, unpinned, actively uses ~8 cores. Running 18 unpinned jobs drove loadavg to ~188
    on 32 logical cores -> total thrash, ZERO progress. (The macOS-era "no thread scaling, 1
    thread/job" note in tf-threading-for-sweeps does not transfer to linux-64.)
  * TF *does* scale with cores here, but SUBLINEARLY (per-step 1c~0.041s, 4c~0.027s, 8c~0.018s), so
    aggregate throughput is maximised by 1 core/job x many jobs -- PROVIDED each job is hard-pinned
    so its pool can't sprawl. We therefore `taskset -c <core>` each job to a single distinct logical
    core (a free-core pool, never two concurrent jobs on one core) AND force single-threaded BLAS
    (OMP/OPENBLAS/MKL/NUMEXPR=1) so numpy in the metric/build doesn't add its own pool.
  * Memory: steady-state ~0.58 GB/job; the one-time Voronoi build spikes ~3.3 GB for several s, so
    we still STAGGER launches (builds never coincide -> no OOM).

dt LEVER (PORTING_NOTES dt study, 2026-06-23): a larger dt is ~ (dt/1e-3)x fewer steps for the same
physical time. The native active model is STABLE up to dt=1e-2 (per-step motion = 1.0*Lth) and keeps
~75% of the reconnection rate at dt=5e-3 (vs ~44% at 1e-2). So dt=5e-3 is the speed/fidelity sweet
spot for a desktop PROBE (5x cheaper, mild fidelity cost); the HPC run can drop to 1e-3/2e-3 for full
fidelity. INTERVAL is auto-set to round(0.01/dt) to hold the oracle reconnection cadence dtr=10*dt.

Pipeline: Phase 0 DP_max(M); Phase 1 the 2*|SIGMAS|*|SEEDS| sims (mixed+demixed) at NATIVE motility;
Phase 2 fig1e+fig1f, copied to *_M{M}.* so the M=6 set is never clobbered.

Run (background): pixi run python rnr/scripts/run_sweep.py [NSTEPS] [M] [MAXPAR] [STAGGER] [MODEL] [DT]
  e.g. NSTEPS=400000 DT=5e-3 -> t=2000 at N=512; ~5 h/job x (18 jobs / 16 cores) ~ overnight.
"""
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXPORTS = os.path.join(ROOT, "rnr", "exports")
LOGDIR = os.path.join(EXPORTS, "sweep_logs")
MASTER = os.path.join(EXPORTS, "sweep_run.log")
PY = sys.executable

NSTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000    # t = NSTEPS*dt = 2000 at dt=5e-3
M = int(sys.argv[2]) if len(sys.argv) > 2 else 8               # N = M^3 = 512
MAXPAR = int(sys.argv[3]) if len(sys.argv) > 3 else 16         # 1 job per physical core (this host=16)
STAGGER = float(sys.argv[4]) if len(sys.argv) > 4 else 12.0    # s between launches (offset build spikes)
MODEL = sys.argv[5] if len(sys.argv) > 5 else "native"         # native engine motility (production)
DT = sys.argv[6] if len(sys.argv) > 6 else "5e-3"              # timestep; INTERVAL derived below
DT_F = float(DT)
DT_TAG = f"{DT_F:g}"                                           # CSV/fig tag form (5e-3 -> "0.005")
INTERVAL = max(1, round(0.01 / DT_F))                          # hold oracle cadence dtr=10*dt
KT, LTH, CUT = "0.1", "1e-3", "1.9"                            # KT reused as active speed v0
SIGMAS = [0.1, 0.2, 0.5]
SEEDS = [7, 8, 9]
ICS = ["mixed", "demixed"]
FIG_STEMS = ["fig1e_demixing_native", "fig1f_stability_native"]
# Single-threaded BLAS so numpy (metric/build) doesn't spawn its own per-process pool on top of TF's.
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "NUMEXPR_NUM_THREADS": "1", "TF_THREADS": "1"}

os.makedirs(LOGDIR, exist_ok=True)
_t0 = time.time()


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def mlog(msg):
    line = f"[{stamp()} +{(time.time() - _t0) / 60:7.1f}m] {msg}"
    print(line, flush=True)
    with open(MASTER, "a") as fh:
        fh.write(line + "\n")


def oracle_cmd(sigma, seed, ic):
    # sort_periodic_oracle.py: MODE M SIGMA KT LTH DT CUT NSTEPS SEED CLAMP IC NOISE_MODEL INTERVAL
    return [PY, os.path.join(HERE, "sort_periodic_oracle.py"), "sort", str(M), str(sigma),
            KT, LTH, DT, CUT, str(NSTEPS), str(seed), "0", ic, MODEL, str(INTERVAL)]


def run_pool(joblist, stagger, pin):
    """Run joblist with <=MAXPAR concurrent children, each new start delayed `stagger` s (so the
    memory-heavy Voronoi builds don't coincide). If `pin`, each job is taskset-pinned to a distinct
    free logical core (a core pool; reclaimed on finish) so TF's ~128-thread pool can't sprawl and
    thrash. Failure-tolerant: a crashed job is logged + skipped. Returns [(name, ok, rc)]."""
    results, running, idx = [], [], 0
    free_cores = list(range(MAXPAR)) if pin else [None] * MAXPAR
    last_launch = 0.0
    while idx < len(joblist) or running:
        if (len(running) < MAXPAR and idx < len(joblist) and free_cores
                and (time.time() - last_launch) >= stagger):
            name, cmd, out = joblist[idx]; idx += 1
            core = free_cores.pop(0)
            wrapped = (["taskset", "-c", str(core)] + cmd) if core is not None else cmd
            fh = open(os.path.join(LOGDIR, f"{name}.log"), "w")
            mlog(f"START {name}  ({idx}/{len(joblist)})" + (f" core={core}" if core is not None else ""))
            proc = subprocess.Popen(wrapped, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                    env={**os.environ, **THREAD_ENV})
            running.append((name, proc, fh, out, core))
            last_launch = time.time()
        time.sleep(2)
        still = []
        for name, proc, fh, out, core in running:
            rc = proc.poll()
            if rc is None:
                still.append((name, proc, fh, out, core)); continue
            fh.close()
            if core is not None:
                free_cores.append(core)
            ok = (rc == 0) and (out is None or os.path.exists(out))
            mlog(f"{'OK   ' if ok else 'FAIL '} {name} (rc={rc}"
                 + ("" if out is None or os.path.exists(out) else ", output missing") + ")")
            results.append((name, ok, rc))
        running = still
    return results


# ---- preserve the committed M=6 native figures before this M run can touch the bare names ----
for stem in FIG_STEMS:
    for ext in ("png", "csv"):
        bare, m6 = os.path.join(EXPORTS, f"{stem}.{ext}"), os.path.join(EXPORTS, f"{stem}_M6.{ext}")
        if os.path.exists(bare) and not os.path.exists(m6):
            shutil.copy2(bare, m6)
            mlog(f"backed up existing {stem}.{ext} -> {stem}_M6.{ext}")

mlog(f"=== SWEEP START [{MODEL}]: M={M} N={M ** 3} NSTEPS={NSTEPS} dt={DT} (t={NSTEPS * DT_F:g}) "
     f"INTERVAL={INTERVAL} MAXPAR={MAXPAR} pinned=1core/job STAGGER={STAGGER}s ===")

# ---- Phase 0: DP_max(M) (the figure normalizer) -- quick, run alone (pinned to core 0) ----
mlog(f"PHASE 0: compute DP_max(M={M})")
with open(os.path.join(LOGDIR, "dpmax.log"), "w") as fh:
    rc = subprocess.call(["taskset", "-c", "0", PY, os.path.join(HERE, "compute_dpmax.py"), str(M), "7"],
                         cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, **THREAD_ENV})
mlog(f"PHASE 0 DONE: compute_dpmax rc={rc}")

# ---- Phase 1: the sim wave (pinned, staggered) ----
jobs = []
for ic in ICS:
    for sigma in SIGMAS:
        for seed in SEEDS:
            suff = f"_{MODEL}" + ("_demixed" if ic == "demixed" else "")
            csv = os.path.join(EXPORTS, f"sort_oracle_M{M}_S{sigma:g}_KT0.1_L0.001_dt{DT_TAG}"
                                        f"_cut1.9_seed{seed}{suff}.csv")
            jobs.append((f"sim_M{M}_S{sigma:g}_seed{seed}_{ic}", oracle_cmd(sigma, seed, ic), csv))

mlog(f"PHASE 1: {len(jobs)} sims (pinned, staggered {STAGGER}s)")
sim_results = run_pool(jobs, STAGGER, pin=True)
n_ok = sum(1 for _, ok, _ in sim_results if ok)
mlog(f"PHASE 1 DONE: {n_ok}/{len(sim_results)} sims OK")

# ---- Phase 2: figures for this M (tolerant; uses whatever CSVs survived) ----
fig_jobs = [
    (f"fig1e_{MODEL}", [PY, os.path.join(HERE, "fig1e_demixing.py"), str(M), DT_TAG, MODEL],
     os.path.join(EXPORTS, f"fig1e_demixing_{MODEL}.png")),
    (f"fig1f_{MODEL}", [PY, os.path.join(HERE, "fig1f_stability.py"), str(M), DT_TAG, MODEL],
     os.path.join(EXPORTS, f"fig1f_stability_{MODEL}.png")),
]
fig_results = run_pool(fig_jobs, 0.0, pin=False)

# ---- tag the produced native figures by M so they sit alongside the M=6 set ----
for stem in FIG_STEMS:
    for ext in ("png", "csv"):
        bare, mtag = os.path.join(EXPORTS, f"{stem}.{ext}"), os.path.join(EXPORTS, f"{stem}_M{M}.{ext}")
        if os.path.exists(bare):
            shutil.copy2(bare, mtag)

# ---- summary ----
mlog("=" * 72)
mlog("FINAL SUMMARY")
for name, ok, rc in sim_results + fig_results:
    mlog(f"  {'OK  ' if ok else 'FAIL'}  {name}")
mlog(f"Figures (M={M}, dt={DT_TAG}) in {EXPORTS}:")
for stem in FIG_STEMS:
    p = os.path.join(EXPORTS, f"{stem}_M{M}.png")
    mlog(f"  {'[ok] ' if os.path.exists(p) else '[--] '}{stem}_M{M}.png")
mlog(f"=== SWEEP DONE in {(time.time() - _t0) / 60:.1f} min ===")
