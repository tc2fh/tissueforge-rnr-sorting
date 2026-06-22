"""Reproduce Manning2024 Fig 1E: normalized demixing DP/DP_max vs time, sigma in {0.1,0.2,0.5}.

THE TARGET (Manning2024 PLOS Comp Biol, Fig 1E, p.7): a 3D VERTEX tissue of two cell types,
initialized MIXED (DP ~ 0), driven by heterotypic interfacial tension sigma. DP/DP_max vs time
rises MONOTONICALLY and is ORDERED by sigma (0.5 fastest/highest, 0.1 slowest). DP is from ref [4]
Sahu/Schwarz/Manning Eq. 2:

    DP = <2*(N_s/N_t - 1/2)>     N_s = homotypic neighbours, N_t = total neighbours, <.> over cells.

Our metrics.demixing_index is D = <2*(het_frac - 1/2)> = -DP (documented sign flip), so DP = -D.
DP_max (segregated-config value for our N; <1 at finite N) is from compute_dpmax.py (dpmax.json).

ENSEMBLE: the count-based DP is noisy at our system size, so (like the paper) we average over
several random seeds and show mean +/- standard error. CSVs are grouped by sigma over seeds; all
seeds share the same checkpoint grid so we average row-by-row.

Two metrics (count-based DP is the faithful Fig-1E one; area is a faster-responding cross-check
that the metrics.py docstring flags as shape-sensitive, not pure neighbour-exchange sorting):
  * panel 1 -- DP/DP_max = -<D>/DP_max  vs time           (faithful Fig 1E reproduction)
  * panel 2 -- area demixing score S_area = 1 - <hetA(t)>/<hetA(0)>

LICENSE: 3DVertVor/tvm GPL -> oracle ONLY. DP RE-DERIVED from Sahu Eq. 2, not copied.

Usage:  pixi run python rnr/scripts/fig1e_demixing.py [M] [DT]    (defaults M=6 dt=0.001)
"""
import csv
import glob
import json
import math
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXPORTS = os.path.join(ROOT, "rnr", "exports")

M = int(sys.argv[1]) if len(sys.argv) > 1 else 6
DT = sys.argv[2] if len(sys.argv) > 2 else "0.001"
# noise model whose CSVs to plot: "active" (FAITHFUL clamp-free, …_seed{N}_active.csv) is the
# default; "thermal" plots the legacy clamp=0.4 CSVs (…_seed{N}.csv). See PORTING_NOTES §6n.
MODEL = sys.argv[3] if len(sys.argv) > 3 else "active"
_SUF = {"active": "_active", "native": "_native"}.get(MODEL, "")  # native = engine drive (…_native.csv)
_NOISE_LBL = {"active": "active motility, clamp-free",
              "native": "native engine motility, clamp-free"}.get(MODEL, "clamp=0.4 + native repair")
COLORS = {0.1: "#e89bd0", 0.2: "#8d4bd6", 0.5: "#2e1378"}  # light pink -> dark (paper palette)


def load_csv(path):
    col = None
    with open(path) as f:
        rd = csv.DictReader(f)
        col = {k: [] for k in (rd.fieldnames or [])}
        for row in rd:
            for k in col:
                col[k].append(float(row[k]))
    return col


def mean_se(rows):
    """rows: list of equal-length sequences -> (mean[], se[]) column-wise."""
    n = len(rows)
    L = min(len(r) for r in rows)
    mean, se = [], []
    for j in range(L):
        vals = [r[j] for r in rows]
        m = sum(vals) / n
        var = sum((v - m) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
        mean.append(m)
        se.append(math.sqrt(var / n) if n > 1 else 0.0)
    return mean, se


def main():
    dp_max = None
    p = os.path.join(EXPORTS, "dpmax.json")
    if os.path.exists(p):
        dp_max = json.load(open(p)).get(str(M), {}).get("DP_max")

    # group CSVs by sigma over seeds
    pat = os.path.join(EXPORTS, f"sort_oracle_M{M}_S*_KT0.1_L0.001_dt{DT}_cut1.9_seed*.csv")
    by_sigma = {}
    for path in sorted(glob.glob(pat)):
        base = os.path.basename(path)
        if not re.search(rf"_seed\d+{_SUF}\.csv$", base):
            continue  # mixed-IC of the chosen MODEL (skip demixed + the other noise model)
        m = re.search(rf"_M{M}_S([0-9.]+)_KT", base)
        if not m:
            continue
        by_sigma.setdefault(float(m.group(1)), []).append(path)
    if not by_sigma:
        print(f"NO CSVs match {pat}")
        return

    dtf = float(DT)
    ylab = r"$DP / DP_{max}$" if dp_max else "DP (= -demixing_index)"
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    summary = []
    for sig in sorted(by_sigma):
        paths = by_sigma[sig]
        cols = [load_csv(pp) for pp in paths]
        L = min(len(c["step"]) for c in cols)
        t = [s * dtf for s in cols[0]["step"][:L]]
        D_mean, D_se = mean_se([c["D"][:L] for c in cols])
        hetA_mean, _ = mean_se([c["het_area"][:L] for c in cols])
        recon_mean, _ = mean_se([c["recon"][:L] for c in cols])
        col = COLORS.get(sig, "k")
        # ---- panel 1: DP/DP_max (faithful, count-based) ----
        DP = [-d for d in D_mean]
        if dp_max:
            y = [v / dp_max for v in DP]
            yse = [s / dp_max for s in D_se]
        else:
            y, yse = DP, D_se
        ax[0].plot(t, y, "-", color=col, lw=2, label=f"$\\sigma$={sig:g} (n={len(paths)})")
        ax[0].fill_between(t, [a - b for a, b in zip(y, yse)], [a + b for a, b in zip(y, yse)],
                           color=col, alpha=0.18, lw=0)
        # ---- panel 2: area demixing score ----
        S_area = [1.0 - h / hetA_mean[0] if hetA_mean[0] else 0.0 for h in hetA_mean]
        ax[1].plot(t, S_area, "-", color=col, lw=2, label=f"$\\sigma$={sig:g}")
        summary.append((sig, len(paths), -D_mean[0], -D_mean[-1],
                        y[-1], S_area[-1], recon_mean[-1], t[-1]))

    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel(r"time  $t = \mathrm{step}\cdot dt$")
    ax[0].set_ylabel(ylab)
    ttl = "Paper's exact metric: count-based $DP/DP_{max}$ (Sahu Eq.2)"
    if dp_max:
        ttl += f"\n$DP_{{max}}$={dp_max:.3f} (finite-N ceiling; paper $\\to$1). Flat here:"
    ttl += "\nneighbour-count sorting is reconnection/finite-N limited (N=216, t=100)"
    ax[0].set_title(ttl, fontsize=9)
    ax[0].legend(title="het. tension", fontsize=9, loc="lower left")
    ax[0].grid(alpha=0.3)

    ax[1].set_xlabel(r"time  $t = \mathrm{step}\cdot dt$")
    ax[1].set_ylabel(r"area demixing  $S_{area}=1-\langle hetA(t)\rangle/\langle hetA(0)\rangle$")
    ax[1].set_title("Interface-area demixing: reproduces the $\\sigma$-ordered Fig 1E trend",
                    fontsize=9)
    ax[1].legend(title="het. tension", fontsize=9, loc="upper left")
    ax[1].grid(alpha=0.3)
    ax[1].annotate("increasing\ninterfacial tension", xy=(0.96, 0.5), xycoords="axes fraction",
                   ha="right", va="center", fontsize=9, color="0.3")

    t_end = max(s[7] for s in summary)
    fig.suptitle(f"Manning2024 Fig 1E in TissueForge 3D vertex (M={M}, N={M**3}, {_NOISE_LBL}, "
                 f"dt={DT}, interval=10)  —  trend match, not absolute-DP "
                 f"(our t$\\leq${int(t_end)} vs paper t=10000)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _figtag = "" if MODEL == "thermal" else f"_{MODEL}"  # active fig kept distinct from legacy thermal
    png = os.path.join(EXPORTS, f"fig1e_demixing{_figtag}.png")
    fig.savefig(png, dpi=140)

    out = os.path.join(EXPORTS, f"fig1e_demixing{_figtag}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma", "n_seeds", "DP_0", "DP_end", "DP/DPmax_end",
                    "S_area_end", "recon_end", "t_end"])
        for r in summary:
            w.writerow([f"{x:.4g}" for x in r])

    print("=" * 78)
    print(f"Fig 1E reproduction  (M={M}, N={M**3}, dt={DT}, {_NOISE_LBL}; ensemble mean)")
    print(f"DP_max (segregated, finite-N) = {dp_max}")
    print(f"{'sigma':>6} {'seeds':>5} {'DP_0':>8} {'DP_end':>8} {'DP/DPmax':>9} "
          f"{'S_area':>8} {'recon':>7} {'t_end':>7}")
    for sig, ns, dp0, dpe, dpn, sa, rc, te in summary:
        print(f"{sig:>6g} {ns:>5d} {dp0:>8.4f} {dpe:>8.4f} {dpn:>9.4f} "
              f"{sa:>8.4f} {rc:>7.0f} {te:>7.1f}")
    sigs = sorted(by_sigma)
    dpe = {s: summary[i][3] for i, s in enumerate(sigs)}
    sae = {s: summary[i][5] for i, s in enumerate(sigs)}
    ord_dp = all(dpe[sigs[i]] <= dpe[sigs[i + 1]] for i in range(len(sigs) - 1))
    ord_sa = all(sae[sigs[i]] <= sae[sigs[i + 1]] for i in range(len(sigs) - 1))
    print("-" * 78)
    print(f"sigma-ordering of final DP   (expect 0.1<0.2<0.5): "
          f"{[round(dpe[s],4) for s in sigs]} -> {'ORDERED' if ord_dp else 'NOT clean'}")
    print(f"sigma-ordering of final S_area (expect 0.1<0.2<0.5): "
          f"{[round(sae[s],4) for s in sigs]} -> {'ORDERED' if ord_sa else 'NOT clean'}")
    print(f"saved: {png}\nsaved: {out}")


if __name__ == "__main__":
    main()
