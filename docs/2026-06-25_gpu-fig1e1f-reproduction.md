# GPU Fig 1E/1F reproduction at paper scale + the σ=0.5 timestep-arrest finding (2026-06-25)

Reproduces Manning2024 (Lawson-Keister, Zhang, Nazari, Fagotto, **Manning**, *PLoS Comp Biol* 2024,
`journal.pcbi.1011724`) **Fig 1E + 1F** in the **GPU** 3D-vertex engine at **paper scale (N=2000)**,
now feasible on the RTX 5090 (foam caching + concurrent host-bound runs).

## The figures (paper definitions)

- **Fig 1E** (p.7): 3D vertex, **mixed** IC (DP≈0); demixing parameter vs time for σ ∈ {0.1,0.2,0.5};
  speed + magnitude increase with σ.
- **Fig 1F**: at fixed σ, **demixed**-start stays sorted (energetically preferred) vs mixed-start rises.
- **DP = ⟨2(N_s/N_t − ½)⟩** (Sahu Eq. 2) **= 1 − 2·het_frac**, normalized by **DP_max** (segregated
  finite-N ceiling). Energy Eq. (3) = the repo's exact physics (VolumeConstraint + SurfaceArea +
  Adhesion-as-σ on heterotypic faces). Paper N=1728, S₀=5.6.

## Pipeline (new scripts, all GPU; no TF on the hot path)

- `rnr/scripts/gpu_dpmax.py` — DP_max via a planar 50/50 cut on the cached foam → **DP_max = 0.789**
  for N=2000 (matches the paper's 1 − O(N^−⅓) finite-N ceiling).
- `rnr/scripts/gpu_fig_runs.py` — concurrency-capped ensemble runner (resumable), CSVs
  `gpu_sort_n10_S{σ}_{ic}_dt{dt}_seed{seed}.csv`.
- `rnr/scripts/gpu_fig1e1f.py` — plotter (DP=1−2·het, normalized, mean±SE, paper palette).
- `rnr/scripts/gpu_video_cells.py` — GPU-driven whole-cell-by-type turntable video.

**Run config:** N=2000 (n=10 cached foam), dt=0.01 (interval=1), **400k steps (t=4000)**, 3 seeds
{7,8,9}, **6-concurrent** (GPU ~1% utilized → host-bound; 6-way ≈ 2.8×, BLAS-capped). 12 runs in
**~3.3h**. Replicates = independent active-drive (director) noise on a common cached foam.

## Results

**Fig 1F — faithfully reproduced.** Demixed-IC σ=0.5 holds at **DP/DP_max ≈ 1.0** for all t=4000
(the demixed state is energetically preferred + stable). `rnr/exports/fig1f_gpu.png`.

**Fig 1E — σ=0.1/0.2 reproduce; σ=0.5 deviates at dt=0.01** (`rnr/exports/fig1e_gpu.png`):

| σ | DP/DP_max @ t=4000 | reconnections (I+H) | regime |
|---|---|---|---|
| 0.1 | 0.59 | 435,471 | fluid, coarsens |
| 0.2 | **0.64** | 198,646 | fluid, coarsens (highest) |
| 0.5 | 0.16 | 15,645 | **kinetically arrested** |

σ=0.1/0.2 demix **strongly** with the correct σ-ordering — this **resolves the long-standing
finite-N limitation**: the old desktop CPU runs (N=216–512) had a flat count-DP near 0
(memory `m8-count-dp-still-scale-limited`). Paper-scale N=2000 + long t on the GPU fixed it.

## THE FINDING: the σ=0.5 arrest is a TIMESTEP artifact (not physics, not a bug)

At high σ the foam stiffens; at dt=0.01 the integrator over-damps the small relaxations that drive
edges below L_th, so the RNR/T1 trigger rarely fires → neighbor-exchange freezes → count-DP plateaus.
Confirmed by a targeted re-run at **dt=2e-3** (interval=5), σ=0.5, 3 seeds, to t=1000:

| σ=0.5 @ t=1000 | reconnections | DP | DP/DP_max |
|---|---|---|---|
| dt=0.01 | 11,138 | 0.111 | 0.14 (arrested) |
| **dt=2e-3** | **~500,000** (≈48×) | **0.55** | **0.70** |

All 3 seeds agree (DP 0.546–0.564). Volumes healthy in both. At dt=2e-3, σ=0.5 reconnects ~48× more,
demixes 5× more, and at DP/DP_max≈0.70 by t=1000 becomes the **fastest/highest — recovering the
paper's σ-ordering**. This confirms `dt-lever-for-faithful-sort` + `m8-count-dp-still-scale-limited`:
the count-DP for the stiffest tension needs **dt ≤ 2e-3**. The low-σ cases stay fluid at dt=0.01.

**Implication for a fully-faithful Fig 1E:** run ALL σ at dt=2e-3 (the dt=0.01 figure is annotated
with the σ=0.5 arrest). Scoped but not run here (budget); the σ=0.5 dt=2e-3 CSVs
(`gpu_sort_n10_S0.5_mixed_dt0.002_seed{7,8,9}.csv`) are the proof + a starting point.

## Video

`rnr/exports/gpu_cells_sort_mixed.mp4` — GPU-driven, **100k steps, ~2000 cells, σ=0.5, mixed, NO
clip plane**, turntable at **0.75°/frame** (¼ of the native clipmz script's 3°/frame), 201 frames,
20 fps. NB: σ=0.5/dt=0.01 is the *arrested* regime (DP→0.11 by 100k), so it shows local clustering,
not full coarsening; a dt=2e-3 video would demix more visibly.

## Reproduce

```
pixi run python rnr/scripts/gpu_dpmax.py 10 mixed                  # DP_max (needs cached n=10 foam)
pixi run python rnr/scripts/gpu_fig_runs.py 400000 6 7,8,9 0.01    # ensemble (~3.3h)
pixi run python rnr/scripts/gpu_fig1e1f.py 10 0.01                 # render Fig 1E/1F
pixi run python rnr/scripts/gpu_video_cells.py 100000 0.5 mixed 500 7   # video (~35 min)
# faithful σ=0.5 (dt=2e-3): pixi run python rnr/scripts/gpu_fig_runs.py 500000 6 7,8,9 0.002
```
Foam is cached (`rnr/exports/foam_cache/foam_n10_{mixed,demixed}_*.npz`, gitignored); first build ~20min.
</content>
