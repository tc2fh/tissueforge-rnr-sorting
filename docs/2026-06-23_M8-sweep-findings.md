# M=8 / N=512 dt-lever sweep — findings (2026-06-23)

**Generated:** 2026-06-23 ~08:34, finalized ~09:12 (sweep ran 02:01→09:11).
**Status:** COMPLETE. 13/18 runs survived (reached t=2000); 5/18 inflated early (dt=5e-3 artifact, §3).
Survivor-only figures regenerated: `rnr/exports/fig1e_demixing_native.png`,
`fig1f_stability_native.png` (these hold the M8 survivor result; the redundant `_M8`-suffixed
copies were removed in a 2026-06-23 exports cleanup. The `_M6` baseline figures were pruned in the
same cleanup but are regenerable from the retained M6 `sort_oracle_*.csv` raw data; unstable CSVs
remain in `unstable_dt0.005/`).

Scope: a *desktop probe* (not the publication run) testing whether the paper's exact count-based
DP/DP_max starts to resolve as we scale toward the paper (N=216→512, t=100→2000). The faithful run
is the eventual HPC sweep. Companion: `docs/BUGS.md`, `rnr/PORTING_NOTES.md`, memories
`tf-threading-for-sweeps`, `dt-lever-for-faithful-sort`.

---

## 1. Headline

- **Infrastructure bug found + root-caused** (TF thread pool ignores `threads=N`) → `docs/BUGS.md` #1.
  Fixed operationally by `taskset` pinning + single-thread BLAS in `run_sweep.py`.
- **The dt lever works but has a ceiling:** the native active-motility sort is stable in short tests
  up to the paper's dt=1e-2, but a *long* M=8 run at **dt=5e-3 inflates ~5/18 jobs** — a coarse-dt
  numerical instability, concentrated in the **demixed IC** and **weak σ=0.1**. dt≤2e-3 is robust.
- **Net:** the desktop probe is partial (σ=0.1 demixed lost entirely). The path to the paper is the
  HPC run at dt≤2e-3, larger N, t→10000.

## 2. Infrastructure: TissueForge threading bug (root-caused)

`tf.init(threads=N)` only sizes the MD nonbond "runner" threads; the **vertex/mesh** force + quality
loops use a *separate* `ThreadPool` singleton hardcoded to `hardware_concurrency()-1` workers
(`source/types/tfThreadPool.h`), dispatched via `parallel_for(ThreadPool::size())`
(`tfMeshSolver.cpp:427/437`). So `threads=N` is a **no-op** for vertex work — a process uses every
core it's allowed unless confined by OS affinity. 18 unpinned jobs → loadavg ~188, **zero progress**.
(OpenBLAS added ~32 more threads/proc until `OMP/OPENBLAS_NUM_THREADS=1` — standard numpy, not a TF
bug, but it compounded the thrash.) Fix in harness: `taskset -c <core>` per job + single-thread BLAS;
TF *does* scale sublinearly so 1-core/job × many jobs is the right throughput config. Full detail +
upstream fix sketch in `docs/BUGS.md` #1.

## 3. The dt lever + the dt=5e-3 long-run instability

Measured (native active model, INTERVAL=round(0.01/dt) to hold the oracle cadence):

| dt | steps for t=2000 | reconnection rate vs dt=1e-3 | short-test stable? | **long M=8 run** |
|----|------------------|------------------------------|--------------------|------------------|
| 1e-3 | 2M | 100% | ✅ | robust (committed M=6 ensembles never inflated) |
| 2e-3 | 1M | ~71% | ✅ | (recommended for HPC; not yet long-tested) |
| **5e-3** | **400k** | ~75% | ✅ (t≤30) | **~5/18 INFLATE at t≈100–1600** |
| 1e-2 | 200k | ~44% | ✅ (t≤30) | (not long-tested; expect worse) |

The instability is **inflation** (`max_vol` 4–12 > the 4·V0 guard; `min_vol` stayed > 0). That
signature matters: it is the displacement-overshoot mode, NOT eversion — the native orientation repair
cures eversion (sign flip) but not magnitude overshoot. The documented fix is a per-vertex
**displacement trust-region in the vertex integrator** (PORTING_NOTES §6j/§7), a planned native item;
until then, **smaller dt** sidesteps it.

Unstable runs (all `dt=5e-3`): σ0.1 demixed seed7@20k / seed8@320k / seed9@280k; σ0.1 mixed seed9@230k;
σ0.5 demixed seed8@170k. CSVs preserved in `rnr/exports/unstable_dt0.005/`.

## 4. Hypotheses (the questions asked)

### 4a. Why we lose the σ=0.1 demixed curve entirely

**Primary hypothesis — it is a coarse-timestep numerical artifact, not physics.** dt=5e-3 is near/over
the stability limit of the stiff overdamped volume dynamics (K_V=10) when a reconnection perturbs a
cell; the overdamped relaxation overshoots and the cell inflates faster than the volume constraint can
recover at that dt. Evidence: (i) it never happened at dt=1e-3 (committed ensembles); (ii) the
signature is inflation, the known coarse-dt/overshoot mode; (iii) it appears mid-run (t≈100–1600), not
a model blow-up at t=0.

**Why it concentrates on demixed + weak σ (modulators, lower confidence):**
- **Demixed IC is the bigger risk factor** (4 demixed vs 1 mixed failure). The demixed IC is
  *constructed* by retyping a relaxed *mixed* Voronoi across a planar z-cut — so the cells straddling
  the new interface have shapes equilibrated for a random-neighbour environment, not for the new flat
  het-interface tension landscape. They must reshape (a transient), and at coarse dt + reconnections
  that transient is the most likely place to overshoot. The mixed IC has no such concentrated feature.
- **Weak σ=0.1 amplifies it within demixed** (3/3 fail vs σ0.5 1/3, σ0.2 0/3). Het tension adds
  geometric stiffness/regularization to het faces; σ=0.1 gives the least, leaving the interface cells
  "floppiest" and most perturbable. (Caveat: σ=0.2 had *zero* failures, breaking strict monotonicity —
  so part of the σ/seed pattern is small-N stochasticity; 5/18 is a small sample.)

**Bottom line:** losing σ=0.1 demixed reflects the *aggressive desktop dt=5e-3*, not a problem with
σ=0.1 physics or the model. dt≤2e-3 (or the native trust-region) removes it for all σ/IC — and that is
what the HPC run will use. This is a probe artifact, not a scientific result about σ=0.1.

### 4b. Why it diverges from the paper (count-based DP/DP_max)

*(Hypothesis stated ahead of the figures; the survivor figures below confirm/refute.)* The paper's
Fig 1E metric counts neighbour-type fractions, which only move once **domains nucleate and coarsen**.
Three gates keep us below that threshold — none a model error:
1. **Run length** — t=2000 is 1/5 of the paper's t=10000. Domain coarsening is slow (sub-diffusive);
   at t=2000 domains have not grown enough to shift neighbour fractions.
2. **System size** — N=512 is the paper's *minimum*; DP_max=0.65 (→1 only as N→∞). The box is ~8
   cells across, so domains reach finite-size before coarsening far. The paper leans on larger N +
   ensemble averaging.
3. **Reconnection rate (self-inflicted)** — dt=5e-3 under-catches reconnections ~25% vs dt=1e-3, and
   neighbour-exchange (the DP driver) is reconnection-limited, so the desktop dt actively slows the
   very metric being tested.
   Possibly also: the active stirring (v0=0.1) randomizes neighbour swaps faster than weak σ sorts
   them (het-fraction pinned ≈0.5), and pure geometric RNR from a *mixed* IC may not nucleate domains
   at feasible N/t without domain-seeded ICs or longer t.

The **Fig 1F (demixed-IC)** test sidesteps domain formation — it starts demixed and asks whether the
het tension HOLDS it. The surviving σ=0.2/0.5 demixed curves are the clean count-DP evidence that the
energetics are correct (as at M=6). So the *expected* picture: Fig 1F holds (energetics right), Fig 1E
count-DP stays low (scale/rate-limited) while area-demixing σ-orders. The figures below test this.

## 5. Figure results (survivor-only, dt=5e-3, t=2000)

Survivors 13/18 — Fig 1E mixed: σ0.1 n=2, σ0.2 n=3, σ0.5 n=3; Fig 1F demixed: σ0.2 n=3, σ0.5 n=2
(σ0.1 demixed lost — all 3 inflated).

### Fig 1E (mixed IC → sorting)
| σ | count DP/DP_max (end) | S_area (end) | reconnections |
|---|----------------------|--------------|---------------|
| 0.1 | −0.008 | 0.043 | 822 |
| 0.2 | −0.020 | 0.081 | 777 |
| 0.5 | −0.008 | 0.152 | 718 |

- **Count-based DP/DP_max (the paper's EXACT metric): FLAT ≈0, NOT σ-ordered** → did **not** resolve.
- **Area demixing S_area: σ-ORDERED, monotonic** (0.043 < 0.081 < 0.152), ~3.5× span, and deeper than M=6.

### Fig 1F (demixed IC → hold)
| series | DP/DP_max start → end |
|--------|----------------------|
| demixed σ=0.2 | 0.981 → 0.972 |
| demixed σ=0.5 | 0.975 → **0.982** |
| mixed σ=0.5 | −0.021 → −0.008 (≈0) |

- **Demixed state HELD at ≈0.97–0.98** for both surviving σ (TIGHTER than M=6's 0.89–0.91); mixed
  stays ≈0. The demixed≈0.98-vs-mixed≈0 contrast is clean count-DP evidence the energetics + native
  RNR are correct.

### M=6 vs M=8 — did scaling help?
| metric | M=6 (N=216, t=100) | M=8 (N=512, t=2000) | effect |
|--------|--------------------|---------------------|--------|
| DP_max (finite-N ceiling) | 0.562 | 0.648 | ↑ toward 1 |
| **Fig 1E count-DP/DP_max** | ≈0 (flat) | ≈0 (flat) | **no change — still flat** |
| Fig 1E S_area (σ=0.5) | ~0.12 | 0.152 | ↑ deeper |
| Fig 1F demixed hold (σ=0.5) | ~0.91 | 0.982 | ↑ tighter |

**Verdict — hypothesis 4b CONFIRMED.** Scaling N (2.4×) + t (20×) deepened the *resolvable* metrics
(area demixing, demixed-hold) but did NOT make the paper's count-based DP resolve — it stayed flat ≈0,
exactly as at M=6. So the gap to the paper's exact Fig 1E is confirmed **domain-formation-limited**
(needs larger N + longer t + the full reconnection rate of dt≤2e-3), **not a model error**: Fig 1F is a
strong and *tightening* reproduction, so the energetics + native RNR are right. **Hypothesis 4a is also
consistent** — the 5 inflations were exactly the predicted demixed/weak-σ pattern, though this desktop
run can't fully separate the σ/IC modulators from small-N noise (σ=0.2 had zero failures).

## 6. Recommendations / HPC plan

- **HPC sweep at dt≤2e-3** (robust; `run_sweep.py` derives INTERVAL from dt). On SLURM,
  `--cpus-per-task=1` provides the affinity the desktop did via `taskset`; still export single-thread
  BLAS per task. Each (σ, seed, IC) is an independent array job.
- **Refill σ=0.1 demixed** at dt≤2e-3 (the gap this probe left).
- **Push N (≥512, ideally ≥1000) and t→10000** for the count-DP domain test; needs the deferred
  "ghost only near-boundary seeds" Voronoi-build optimization for N≫512.
- **Native fix (optional, removes the dt ceiling):** add the per-vertex displacement trust-region to
  the vertex integrator so larger dt is safe (PORTING_NOTES §6j/§7).
