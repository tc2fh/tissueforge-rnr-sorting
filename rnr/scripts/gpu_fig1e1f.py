"""Plot Manning2024 Fig 1E + 1F from the GPU ensemble CSVs (paper scale, N=2000).

Reads the per-run timelines written by gpu_fig_runs.py (gpu_sort_n10_S*_*_dt*_seed*.csv), forms
the demixing parameter DP = 1 - 2*het (Sahu Eq. 2; het = engine het_contact_fraction), normalizes
by DP_max (gpu_dpmax.json), and averages over seeds (mean +/- standard error).

  Fig 1E (fig1e_gpu.png): DP/DP_max vs t for sigma in {0.1,0.2,0.5}, MIXED IC -- the demixing
    rises and is ordered by interfacial tension (0.5 fastest/highest, 0.1 slowest).
  Fig 1F (fig1f_gpu.png): at sigma=0.5, DEMIXED IC (starts ~DP_max, stays high: the demixed state
    is energetically preferred) vs MIXED IC (rises toward it).

Usage:  pixi run python rnr/scripts/gpu_fig1e1f.py [N] [DT]   (defaults n=10, dt=0.01)
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
EXPORTS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "rnr", "exports")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
DT = sys.argv[2] if len(sys.argv) > 2 else "0.01"
DTF = float(DT)
COLORS = {0.1: "#e89bd0", 0.2: "#8d4bd6", 0.5: "#2e1378"}   # paper palette: light pink -> dark


def load(path):
    with open(path) as f:
        rd = csv.DictReader(f)
        col = {k: [] for k in (rd.fieldnames or [])}
        for row in rd:
            for k in col:
                col[k].append(float(row[k]))
    return col


def mean_se(rows):
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


def collect(ic):
    """{sigma: [csv paths]} for the given IC."""
    by = {}
    for path in sorted(glob.glob(os.path.join(EXPORTS, f"gpu_sort_n{N}_S*_{ic}_dt{DT}_seed*.csv"))):
        m = re.search(rf"_S([0-9.]+)_{ic}_", os.path.basename(path))
        if m:
            by.setdefault(float(m.group(1)), []).append(path)
    return by


def dp_series(paths, dp_max):
    """(t, DP/DP_max mean, SE) averaged over seeds; DP = 1 - 2*het."""
    cols = [load(p) for p in paths]
    L = min(len(c["step"]) for c in cols)
    t = [s * DTF for s in cols[0]["step"][:L]]
    dp_rows = [[1.0 - 2.0 * h for h in c["het"][:L]] for c in cols]
    m, se = mean_se(dp_rows)
    y = [v / dp_max for v in m]
    yse = [s / dp_max for s in se]
    return t, y, yse


def main():
    dpj = os.path.join(EXPORTS, "gpu_dpmax.json")
    dp_max = json.load(open(dpj)).get(str(N), {}).get("DP_max") if os.path.exists(dpj) else None
    if not dp_max:
        print(f"FATAL: no DP_max for n={N} in {dpj} (run gpu_dpmax.py).")
        sys.exit(2)
    mixed, demixed = collect("mixed"), collect("demixed")
    if not mixed:
        print("FATAL: no mixed-IC CSVs found (run gpu_fig_runs.py).")
        sys.exit(2)

    # ---------------- Fig 1E: demixing vs time, sigma-ordered (mixed IC) ----------------
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    e_summary = []
    for sig in sorted(mixed):
        t, y, yse = dp_series(mixed[sig], dp_max)
        c = COLORS.get(sig, "k")
        ax.plot(t, y, "-", color=c, lw=2.2, label=f"$\\sigma$={sig:g}  (n={len(mixed[sig])})")
        ax.fill_between(t, [a - b for a, b in zip(y, yse)], [a + b for a, b in zip(y, yse)],
                        color=c, alpha=0.18, lw=0)
        e_summary.append((sig, len(mixed[sig]), y[0], y[-1], t[-1]))
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"time  $t = \mathrm{step}\cdot dt$")
    ax.set_ylabel(r"$DP / DP_{max}$")
    ax.set_title("Manning2024 Fig 1E — 3D vertex demixing (GPU, N=2000)\n"
                 f"mixed IC; $DP=1-2\\,$het (Sahu Eq.2), $DP_{{max}}$={dp_max:.3f}", fontsize=10)
    ax.legend(title="het. interfacial tension", fontsize=9, loc="center right")
    ax.grid(alpha=0.3)
    # honest note: sigma=0.5 kinetically ARRESTS (stiff interfaces -> the RNR trigger rarely fires
    # -> neighbour-exchange stalls), so the count-DP ordering is 0.1<0.2 but 0.5 plateaus LOW.
    arrested = len(e_summary) >= 3 and e_summary[-1][3] < e_summary[-2][3]
    if arrested:
        ax.annotate("$\\sigma$=0.5 kinetically arrests\n(stiff foam: reconnections stall,\n"
                    "neighbour-count sorting freezes)", xy=(0.5, 0.20), xycoords="axes fraction",
                    ha="center", va="center", fontsize=8.5, color="#2e1378",
                    bbox=dict(boxstyle="round", fc="white", ec="#2e1378", alpha=0.85))
    fig.tight_layout()
    e_png = os.path.join(EXPORTS, "fig1e_gpu.png")
    fig.savefig(e_png, dpi=150)
    plt.close(fig)

    # ---------------- Fig 1F: demixed state is stable (sigma=0.5) -----------------------
    sig0 = 0.5
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    f_summary = []
    if sig0 in demixed:
        t, y, yse = dp_series(demixed[sig0], dp_max)
        ax.plot(t, y, "-", color="#b3007a", lw=2.4, label=f"initially DEMIXED ($\\sigma$={sig0:g})")
        ax.fill_between(t, [a - b for a, b in zip(y, yse)], [a + b for a, b in zip(y, yse)],
                        color="#b3007a", alpha=0.18, lw=0)
        f_summary.append(("demixed", y[0], y[-1], len(demixed[sig0])))
    if sig0 in mixed:
        t, y, yse = dp_series(mixed[sig0], dp_max)
        ax.plot(t, y, "--", color="#e89bd0", lw=2.4, label=f"initially MIXED ($\\sigma$={sig0:g})")
        ax.fill_between(t, [a - b for a, b in zip(y, yse)], [a + b for a, b in zip(y, yse)],
                        color="#e89bd0", alpha=0.18, lw=0)
        f_summary.append(("mixed", y[0], y[-1], len(mixed[sig0])))
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel(r"time  $t = \mathrm{step}\cdot dt$")
    ax.set_ylabel(r"$DP / DP_{max}$")
    ax.set_title(f"Manning2024 Fig 1F — demixed state is energetically preferred (GPU, N=2000)\n"
                 f"$\\sigma$={sig0:g}: demixed stays high; mixed rises toward it", fontsize=10)
    ax.legend(fontsize=10, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f_png = os.path.join(EXPORTS, "fig1f_gpu.png")
    fig.savefig(f_png, dpi=150)
    plt.close(fig)

    # ---------------- summary CSV + stdout ----------------
    out = os.path.join(EXPORTS, "fig1e1f_gpu_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fig", "series", "n_seeds", "DPnorm_0", "DPnorm_end", "t_end"])
        for sig, ns, y0, ye, te in e_summary:
            w.writerow(["1E", f"sigma={sig:g} mixed", ns, f"{y0:.4f}", f"{ye:.4f}", f"{te:g}"])
        for nm, y0, ye, ns in f_summary:
            w.writerow(["1F", f"sigma={sig0:g} {nm}", ns, f"{y0:.4f}", f"{ye:.4f}", ""])

    print("=" * 72)
    print(f"Fig 1E/1F (GPU, N=2000, dt={DT}); DP_max={dp_max:.4f}")
    print(f"{'fig':>3} {'series':>22} {'seeds':>5} {'DPn_0':>7} {'DPn_end':>8} {'t_end':>7}")
    for sig, ns, y0, ye, te in e_summary:
        print(f"{'1E':>3} {f'sigma={sig:g} mixed':>22} {ns:>5} {y0:>7.3f} {ye:>8.3f} {te:>7g}")
    for nm, y0, ye, ns in f_summary:
        print(f"{'1F':>3} {f'sigma={sig0:g} {nm}':>22} {ns:>5} {y0:>7.3f} {ye:>8.3f} {'':>7}")
    sigs = sorted(mixed)
    ends = {s: e_summary[i][3] for i, s in enumerate(sigs)}
    ordered = all(ends[sigs[i]] <= ends[sigs[i + 1]] for i in range(len(sigs) - 1))
    print("-" * 72)
    print(f"sigma-ordering of final DP/DP_max (expect 0.1<0.2<0.5): "
          f"{[round(ends[s], 3) for s in sigs]} -> {'ORDERED' if ordered else 'NOT clean'}")
    print(f"saved: {e_png}\nsaved: {f_png}\nsaved: {out}")


if __name__ == "__main__":
    main()
