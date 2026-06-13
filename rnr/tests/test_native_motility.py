"""Gate: the NATIVE (C++ engine) active-motility drive reproduces the Python-active model.

Phase 3 (PORTING_NOTES §6o) moved the active self-propulsion -- a per-cell director with
rotational diffusion + a per-vertex active force v0*<incident-cell directors> -- out of the Python
harness (sort_periodic_oracle.py::add_noise_active) and INTO the C++ engine (Body::director +
MeshSolver::preStepStart director evolution + the active term in VertexForce, set via
MeshSolver.set_motility). These tests lock in that the native drive is faithful:

  * NATIVE rate (clamp-free): the same reconnection rate as the Python-active model (~35/3000 at
    M=4, sigma=0.5), STABLE -- i.e. the engine drive behaves exactly like the validated harness
    drive (PORTING_NOTES §6n), with NO Python per-step injection.
  * CALIBRATION: the per-step vertex displacement is exactly dt*v0*<director> (confirms the
    overdamped mobility mu=1 and the force scaling), and the director rotational-diffusion rate
    scales with Dr (confirms rotStd = sqrt(2*Dr*dt) is wired correctly).

If the native rate ever drops to the starved (~1) level, the engine drive has regressed; if the
calibration fails, the force scaling or Dr wiring is wrong. Companion to
test_clampfree_reconnection.py (which gates the Python-active model).

Run: pixi run test
"""
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROBE = os.path.join(_ROOT, "rnr", "scripts", "probe_active_motility.py")
_CALIB = os.path.join(_ROOT, "rnr", "scripts", "probe_native_calibration.py")


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


def test_native_motility_matches_active_rate():
    """The native C++ active drive (MODEL=native, no Python injection) restores the same
    clamp-free reconnection rate as the Python-active model (~35/3000 at M=4, sigma=0.5) and stays
    stable. Threshold is set well below the measured ~37 to tolerate run-to-run variation, but far
    above the starved ~1 of unclamped thermal noise -- and capped to catch a gross scaling error."""
    # args: STEPS SIGMA V0 LTH DT CUT DR M SEED INTERVAL MODEL
    out = _run(_PROBE, [3000, 0.5, 0.1, 1e-3, 1e-3, 1.9, 1.0, 4, 7, 10, "native"])
    recon, verdict = _recon_verdict(out)
    assert verdict == "STABLE", f"native active-motility sort is UNSTABLE:\n{out}"
    assert 10 <= recon <= 80, (
        f"native drive reconnection rate (recon={recon}) is outside the faithful band; it should "
        f"match the Python-active ~35/3000 (PORTING_NOTES §6n/§6o):\n{out}")


def test_native_calibration_displacement_and_dr():
    """Calibration (PORTING_NOTES §6o): the native per-vertex displacement equals dt*v0*<director>
    (overdamped mu=1 + correct force scaling) and the director rotational-diffusion rate scales
    with Dr (rotStd = sqrt(2*Dr*dt))."""
    out = _run(_CALIB, [4, 7, 1000])
    verdict = next(ln for ln in out.splitlines() if ln.startswith("CALIBRATION VERDICT"))
    assert "PASS" in verdict, f"native motility calibration FAILED:\n{out}"
