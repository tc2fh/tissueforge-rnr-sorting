"""Gate: the FAITHFUL active-motility noise restores the reconnection rate with NO clamp.

This locks in the 2026-06-11 finding (PORTING_NOTES §6n). The 3DVertVor/Manning oracle does not
use thermal Brownian noise -- its per-vertex thermal line (Run.cpp:1344, `cR*ndist`, scaling as
sqrt(dt)) is COMMENTED OUT; it advances by `dt*motility` (Run.cpp:1345), an ACTIVE self-propulsion
that scales as dt. With v0~temperature=0.1 and dt=1e-3 the per-step displacement is ~0.1*Lth, well
below the reconnect trigger, so a collapsing edge persists below Lth and is caught WITHOUT any
clamp.

Our harness had substituted thermal `tf.Force.random`/Euler-Maruyama noise (DISP_STD = sqrt(2*mu*
kT*dt) = 14-45x Lth), which blows freshly-collapsing edges back above the trigger and STARVES
reconnection -- the symptom that forced the §6j noise clamp. The two tests below are the
load-bearing contrast:
  * active motility, NO clamp        -> reconnection rate HEALTHY (and stable)
  * thermal noise,  NO clamp         -> reconnection STARVED (the bug the active model fixes)

If the first ever drops to the starved rate, the active model has regressed to the thermal failure
mode; if the second ever reports a healthy rate, the contrast (and §6n's premise) is no longer real.

Run: pixi run test
"""
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ACTIVE = os.path.join(_ROOT, "rnr", "scripts", "probe_active_motility.py")
_THERMAL = os.path.join(_ROOT, "rnr", "scripts", "probe_periodic_sort.py")


def _run(script, args):
    proc = subprocess.run(
        [sys.executable, script, *[str(a) for a in args]],
        cwd=_ROOT, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, f"probe crashed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def _recon_verdict(stdout):
    """Parse 'VERDICT ... recon=N STABLE/UNSTABLE' -> (int recon, str verdict)."""
    line = next(ln for ln in stdout.splitlines() if ln.startswith("VERDICT"))
    toks = dict(t.split("=") for t in line.split() if "=" in t)
    return int(toks["recon"]), line.strip().split()[-1]


def test_active_motility_restores_reconnection_rate_without_clamp():
    """Faithful active model (clamp-free): the reconnection rate is restored to the healthy
    clamped-thermal level (~35/3000 at M=4, sigma=0.5) and the sort stays stable. Threshold is
    set well below the measured ~35 to tolerate run-to-run variation, but far above the starved
    ~1 of unclamped thermal noise."""
    # args: STEPS SIGMA V0 LTH DT CUT DR M SEED INTERVAL
    out = _run(_ACTIVE, [3000, 0.5, 0.1, 1e-3, 1e-3, 1.9, 1.0, 4, 7, 10])
    recon, verdict = _recon_verdict(out)
    assert verdict == "STABLE", f"clamp-free active-motility sort is UNSTABLE:\n{out}"
    assert recon >= 10, (
        f"active motility starved reconnection (recon={recon}); the faithful model should keep "
        f"reconnection healthy (~35) with no clamp:\n{out}")


def test_thermal_noise_without_clamp_starves_reconnection():
    """Load-bearing contrast (PORTING_NOTES §6l/§6n): the thermal Brownian departure WITHOUT the
    clamp starves reconnection -- one sqrt(dt) kick (~14*Lth) blows each freshly-collapsing edge
    back above the trigger before doQuality can catch it. This is exactly what the active model
    fixes. (Repair is ON by default, so the run is stable -- but FROZEN, not sorting.)"""
    # args: STEPS SIGMA KT LTH DT CUT CLAMP M SEED INTERVAL
    out = _run(_THERMAL, [3000, 0.5, 0.1, 1e-3, 1e-3, 1.9, 0.0, 4, 7, 10])
    recon, _verdict = _recon_verdict(out)
    assert recon <= 3, (
        f"unclamped thermal noise was expected to STARVE reconnection (~1/3000) but got "
        f"recon={recon}; the §6n premise (thermal sqrt(dt) noise sabotages the trigger) may no "
        f"longer hold:\n{out}")
