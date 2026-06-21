---
name: clamp-enables-reconnection-not-just-stability
description: "RESOLVED 2026-06-11 by [[active-motility-not-thermal-noise]] — the OBSERVATIONS here hold (unclamped THERMAL noise starves native I↔H: ~1 recon/3000 vs ~24 clamped, M=4) but the CONCLUSION ('clamp needed / no clamp-free path') is WRONG: the clamp was a band-aid over the WRONG NOISE MODEL. The 3DVertVor oracle does NOT use thermal √dt noise (its thermal line Run.cpp:1344 is commented out); it uses ACTIVE motility dt·v0 (sub-Lth/step), so reconnection is caught with NO clamp. Faithful fix = swap the noise model, not clamp it. The '√dt kicks 14–45× Lth blow edges past the trigger' mechanism is real and is exactly why thermal-without-clamp fails; active motility avoids it by construction."
metadata:
  node_type: memory
  type: project
  originSessionId: native-rnr-reconnection
---

**RESOLUTION (2026-06-11, next session — supersedes the "clamp is load-bearing / unreachable
clamp-free" conclusion below; the rate observations stay valid).** The clamp was a band-aid over a
WRONG NOISE MODEL. The 3DVertVor/Manning oracle does NOT use thermal Brownian noise — its per-vertex
thermal line `Run.cpp:1344` (`cR·ndist`, √dt) is COMMENTED OUT; it advances by `dt·motility`
(`:1345`), ACTIVE self-propulsion that scales as **dt, not √dt**, so per-step displacement ≤ dt·v0 ≈
0.1×Lth — *below* the trigger, caught with NO clamp. Our harness substituted thermal `tf.Force.random`
(§6f calibrated long-time diffusion, not per-step displacement) → 14–45× Lth/step → starvation →
needed the clamp. Faithful fix = use active motility (`probe_active_motility.py`,
`sort_periodic_oracle.py NOISE_MODEL=active` default): rate restored clamp-free (M=4 35, M=6 141, vs
starved 1), STABLE, σ-ordered demixing holds. A v0=0 control gives 38 — the reconnections are
DETERMINISTIC-relaxation-driven; noise's job is to NOT SABOTAGE the trigger, which thermal does and
active doesn't. Line 37–39 below ("noise applied so the deterministic tension dominates near
collapsing edges") was the right intuition. See [[active-motility-not-thermal-noise]], PORTING_NOTES
§6n, gate `test_clampfree_reconnection.py`. NOTE: the existing Fig 1E/1F RESULTS below were generated
with the thermal+clamp departure; they reproduce the same trend under the active model (being re-run).

While starting the Manning2024 Fig 1E reproduction (the kickoff prompt
`docs/fig1e_reproduction_kickoff_prompt.md`, which instructed "clamp=0, the native repair
provides stability now"), a reconnection-rate diagnostic (probe_periodic_sort.py, σ=0.5, M=4,
3000 steps) showed the kickoff's premise is **incomplete**:

| dt | clamp | recon / 3000 steps |
|---|---|---|
| 0.001 | 0 (kickoff "faithful") | ~1 |
| 0.001 | 0.4 (§6j) | ~24 |
| 0.005 | 0.4 | ~11 |
| 0.01 | 0.4 | ~3 (plateaus) |
| 0.01 | 0 | ~0 (through t=600) |
| 0.001 | 0, interval=1 | ~3 |

**Mechanism.** A native I↔H places two vertices Lth=1e-3 apart; one Euler-Maruyama thermal
kick is DISP_STD=sqrt(2μ·kT·dt) = 0.0141 (dt=1e-3) … 0.0447 (dt=1e-2), i.e. 14–45× Lth. With
clamp=0 the kick throws a collapsing edge back above the Lth trigger before doQuality (runs
every reconnect_interval=10 steps) can catch it below threshold ⇒ reconnection starves ⇒ no
neighbour exchange ⇒ no sorting (DP is neighbour-count based). The §6j clamp (cap noise at
0.4×nn-dist) binds ONLY on near-degenerate short edges, letting them persist below Lth so the
T1 fires. So the clamp does **two** jobs: (1) prevent the post-reconnection eversion (§6j) AND
(2) **enable reconnection at all**. The §6k orientation repair only fixes (1) the eversion —
it is orthogonal to (2). Hence clamp=0 + repair ON is stable but **frozen** (the §6k "faithful
stable 20000 steps" claim is true but the tissue isn't sorting — it's the jammed control).

**Consequence for faithfulness.** A fully clamp-free faithful sort isn't reachable with the
current Python-pre-step-noise harness: interval=1 (check every step) barely helps (~3), because
the edge is above Lth at essentially every check. Catching reconnections without the clamp would
need the trigger to use a noise-free / time-averaged edge length, or noise applied so the
deterministic tension dominates near collapsing edges — an engine change, not done. For now the
Fig 1E runs use **clamp=0.4 (flagged departure) + repair ON + dt=0.001 + interval=10**. This is
the §6j sorting setup; the repair adds the faithful eversion fix on top.

**dt.** Larger dt starves reconnection (noise ∝ √dt) AND runs doQuality fewer times per unit
physical time, so dt=0.01 (the paper's value) is unusable here despite being volume-stable.
dt=0.001 gives the best reconnection rate (per step AND per unit t). Reaching the paper's
t≈2000–4000 at dt=1e-3 needs 2–4e6 steps, so we compare **trend over a feasible window /
fraction-of-run** (the project's validated approach), not absolute physical time.

**Fig 1E metric.** DP (Sahu Eq.2, ref[4] arXiv 2102.05397) = ⟨2(N_s/N_t − 1/2)⟩ = **−our
demixing_index** exactly (documented sign flip). DP_max (segregated config, finite-N) computed
by `compute_dpmax.py`: **DP_max(M=6)=0.56** (not 1.0; Sahu SI: DP_max ≈ 1 − O(N^−1/3)). Paper
plots DP/DP_max. Early M=6 data: AREA-based het fraction is cleanly σ-ordered (0.5>0.2>0.1 drop)
within a few k steps; COUNT-based DP lags (areas shrink before neighbours are lost) and is noisy
⇒ ensemble-average over seeds (paper does too).

**RESULT (3×3 ensemble, M=6, 100k steps, clamp=0.4+repair, dt=1e-3; rnr/exports/fig1e_demixing.png).**
All 9 STABLE (~540 reconnections each, no eversion/inflation — repair holds at scale). **AREA
demixing reproduces the σ-ordered Fig 1E trend**: normalized S_area = 0.022 / 0.037 / 0.090 for
σ=0.1/0.2/0.5 (monotonic, ORDERED). **Count-based DP/DP_max (paper's exact metric) does NOT
resolve** — all within ±0.01 of 0, not ordered (noise-dominated): with energy-gate-OFF geometric
RNR the total neighbour count GROWS (foam churns, ~1687→2219 contacts) so het FRACTION stays ≈0.5;
count-DP needs DOMAIN formation ⇒ larger N (paper N≥512, DP_max→1) + longer t (paper 10000 vs our
100). Verdict: TREND match (area) + stable engine, NOT absolute count-DP — a system-size/run-length
gap, not a correctness gap. OOM note: ≤3–4 parallel 216-cell runs (9 at once kills ~3).

**Fig 1F (DONE, the clean count-DP win).** sort_periodic_oracle.py 11th arg `demixed` seeds a
segregated z-slab (=DP_max config); `pixi run fig1f` → fig1f_stability.png. Initialized DEMIXED, the
tissue **STAYS demixed for every σ**: DP/DP_max 0.93→0.86 (σ=0.1), 0.93→0.86 (σ=0.2), 0.97→0.92
(σ=0.5) over 100k steps — vs mixed-IC ≈0. This SIDESTEPS the 1E count-DP limit (starts past domain
nucleation) and is direct evidence the energetics+RNR are correct (demixed = stable minimum the het
tension holds, even at weak σ). So 1E trend (area) + 1F count-DP stability both reproduced; only
mixed→demixed count-DP convergence is still out of reach at N=216/t=100. See
[[native-orientation-repair-faithful]], [[periodic-sort-noise-overshoot-fixed]],
[[oracle-comparison-ceiling-physical]].
