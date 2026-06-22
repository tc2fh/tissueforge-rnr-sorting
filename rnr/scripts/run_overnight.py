"""Overnight batch: the FULL active-motility ensemble + figures + a demixing video.

Runs, with bounded concurrency and tolerant of individual failures:
  Phase 1 -- 18 sims: sigma in {0.1,0.2,0.5} x seed in {7,8,9} x IC in {mixed,demixed},
             M=6, 100k steps, FAITHFUL active-motility noise (clamp-free), via
             sort_periodic_oracle.py -> rnr/exports/sort_oracle_..._active[_demixed].csv
             + one video run (video_periodic_active.py) -> sort_active_demixing.gif
  Phase 2 -- fig1e (mixed, MODEL=active) + fig1f (demixed vs mixed, MODEL=active)
             -> fig1e_demixing_active.{png,csv}, fig1f_stability_active.{png,csv}

Each job logs to rnr/exports/overnight_logs/<name>.log; a master timeline + final summary go to
rnr/exports/overnight_run.log (tail -f that to watch). A crashed sim is recorded and skipped;
the ensemble averages over whatever seeds survived.

Run (in background):  pixi run python rnr/scripts/run_overnight.py [MAXPAR] [NSTEPS]
                      defaults MAXPAR=3 NSTEPS=100000
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXPORTS = os.path.join(ROOT, "rnr", "exports")
LOGDIR = os.path.join(EXPORTS, "overnight_logs")
MASTER = os.path.join(EXPORTS, "overnight_run.log")
PY = sys.executable

# THROUGHPUT (2026-06-21): TissueForge's engine spins a thread pool sized to ALL host cores by
# default (Simulator.threads = n_cpu), but on these small M^3-cell vertex runs it shows ~0 parallel
# scaling -- benchmarked 1 vs 16 threads = 32.6s vs 33.4s for 3000 steps (noise). So per-job threads
# are wasted and only oversubscribe; throughput is maximised by 1 thread/job + many jobs at once.
# We therefore pin TF_THREADS=1 per child (honoured by sort_periodic_oracle.py / video_*.py tf.init)
# and run ALL sims concurrently. 18 sims + 1 video = 19 jobs <= 32 cores -> one wave (~18 min for
# 100k steps), vs the old MAXPAR=3 / 32-thread-pool config's ~6 waves (~100 min). ~5-6x faster.
MAXPAR = int(sys.argv[1]) if len(sys.argv) > 1 else 19
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
# Active-drive model for the whole sweep + figures: "active" = Python add_noise_active (the prior
# canonical figs); "native" = the C++ engine drive (MeshSolver.set_motility), ~3x faster and
# statistically equivalent. native writes …_native[_demixed].csv and fig…_native.png so it never
# clobbers the active set -- the two can be compared side by side (compare_active_native.py).
MODEL = sys.argv[3] if len(sys.argv) > 3 else "active"
TF_THREADS = os.environ.get("TF_THREADS", "1")   # 1 = throughput-optimal (no per-job scaling at M=6)
M, DT, LTH, CUT = 6, "1e-3", "1e-3", "1.9"
SIGMAS = [0.1, 0.2, 0.5]
SEEDS = [7, 8, 9]
ICS = ["mixed", "demixed"]
VIDEO_STEPS = min(40000, NSTEPS)   # video run is shorter (illustrative; ~50 frames)

os.makedirs(LOGDIR, exist_ok=True)
_t0 = time.time()


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def mlog(msg):
    line = f"[{stamp()} +{(time.time() - _t0) / 60:6.1f}m] {msg}"
    print(line, flush=True)
    with open(MASTER, "a") as fh:
        fh.write(line + "\n")


def oracle_cmd(sigma, seed, ic):
    # sort_periodic_oracle.py: MODE M SIGMA KT LTH DT CUT NSTEPS SEED CLAMP IC NOISE_MODEL
    return [PY, os.path.join(HERE, "sort_periodic_oracle.py"), "sort", str(M), str(sigma),
            "0.1", LTH, DT, CUT, str(NSTEPS), str(seed), "0", ic, MODEL]


def video_cmd():
    # video_periodic_active.py: N_STEPS SIGMA V0 M SEED CAPTURE_EVERY
    return [PY, os.path.join(HERE, "video_periodic_active.py"), str(VIDEO_STEPS), "0.5", "0.1",
            str(M), "7", "800"]


# ------- build the job list (name, cmd, expected output file or None) -------
jobs = []
for ic in ICS:
    for sigma in SIGMAS:
        for seed in SEEDS:
            suff = f"_{MODEL}" + ("_demixed" if ic == "demixed" else "")
            csv = os.path.join(
                EXPORTS, f"sort_oracle_M{M}_S{sigma:g}_KT0.1_L0.001_dt0.001_cut1.9_seed{seed}{suff}.csv")
            jobs.append((f"sim_S{sigma:g}_seed{seed}_{ic}", oracle_cmd(sigma, seed, ic), csv))
# The demixing video is the Python-active illustration only; skip it for native (CSV comparison is
# what proves equivalence, and the active run already produced sort_active_demixing.gif).
if MODEL == "active":
    jobs.append(("video_S0.5", video_cmd(), os.path.join(EXPORTS, "sort_active_demixing.gif")))


def run_pool(joblist):
    """Run joblist with <=MAXPAR concurrent subprocesses; return list of (name, ok, rc)."""
    results = []
    running = []   # (name, proc, fh, out)
    idx = 0
    while idx < len(joblist) or running:
        while len(running) < MAXPAR and idx < len(joblist):
            name, cmd, out = joblist[idx]; idx += 1
            fh = open(os.path.join(LOGDIR, f"{name}.log"), "w")
            mlog(f"START {name}")
            child_env = {**os.environ, "TF_THREADS": TF_THREADS}   # 1 thread/job (throughput)
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=child_env)
            running.append((name, proc, fh, out))
        time.sleep(5)
        still = []
        for name, proc, fh, out in running:
            rc = proc.poll()
            if rc is None:
                still.append((name, proc, fh, out)); continue
            fh.close()
            ok = (rc == 0) and (out is None or os.path.exists(out))
            mlog(f"{'OK   ' if ok else 'FAIL '} {name} (rc={rc}"
                 + ("" if out is None or os.path.exists(out) else ", output missing") + ")")
            results.append((name, ok, rc))
        running = still
    return results


mlog(f"=== OVERNIGHT START [{MODEL}]: {len(jobs)} jobs, MAXPAR={MAXPAR}, "
     f"TF_THREADS={TF_THREADS}/job, NSTEPS={NSTEPS}, M={M} ===")
sim_results = run_pool(jobs)
n_ok = sum(1 for _, ok, _ in sim_results if ok)
mlog(f"=== PHASE 1 DONE: {n_ok}/{len(sim_results)} jobs OK ===")

# ------- Phase 2: figures (tolerant; need the CSVs from phase 1) -------
fig_jobs = [
    (f"fig1e_{MODEL}", [PY, os.path.join(HERE, "fig1e_demixing.py"), str(M), "0.001", MODEL],
     os.path.join(EXPORTS, f"fig1e_demixing_{MODEL}.png")),
    (f"fig1f_{MODEL}", [PY, os.path.join(HERE, "fig1f_stability.py"), str(M), "0.001", MODEL],
     os.path.join(EXPORTS, f"fig1f_stability_{MODEL}.png")),
]
fig_results = run_pool(fig_jobs)

# ------- final summary -------
mlog("=" * 70)
mlog("FINAL SUMMARY")
for name, ok, rc in sim_results + fig_results:
    mlog(f"  {'OK  ' if ok else 'FAIL'}  {name}")
mlog(f"Outputs in {EXPORTS}:")
_outs = [f"fig1e_demixing_{MODEL}.png", f"fig1f_stability_{MODEL}.png"]
if MODEL == "active":
    _outs.append("sort_active_demixing.gif")
for f in _outs:
    p = os.path.join(EXPORTS, f)
    mlog(f"  {'[ok] ' if os.path.exists(p) else '[--] '}{f}")
mlog(f"=== OVERNIGHT DONE in {(time.time() - _t0) / 60:.1f} min ===")
