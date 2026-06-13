"""Dynamic gate: a periodic Kelvin-foam substrate must INTEGRATE stably.

The static periodic-geometry tests (test_periodic_geometry.py) only check rest-geometry
*values* on hand-built configs. They passed for months while a real bug made the periodic
bulk blow up under *dynamics*: `FlatSurfaceConstraint` (a default surface actor, lam=0.1,
auto-bound to every SurfaceType) computed its planarity force from a RAW `centroid - vertex`
difference. For a surface that wraps a periodic box wall that difference is ~box-sized, so
wall vertices got a spurious ~1/dt force that inverted a cell within a few hundred steps --
even at sigma=0 with no reconnection and no noise. See PORTING_NOTES.md (periodic geometry)
and docs/periodic_substrate_engine_bug.md.

The fix makes FlatSurfaceConstraint use the minimum-image displacement (no-op when
mesh.periodic_geometry is off). This gate locks it in: integrate a space-filling periodic
foam at sigma=0 and assert no cell inverts / inflates. It runs the probe in a SUBPROCESS
because tf.init() is a one-per-process singleton and the shared-session universe would
otherwise be stepped underneath the other test modules.

Run: pixi run test
"""
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROBE = os.path.join(_ROOT, "rnr", "scripts", "probe_periodic_substrate.py")
_SORT_PROBE = os.path.join(_ROOT, "rnr", "scripts", "probe_periodic_sort.py")


def _run_probe(steps, sigma, dt, kT):
    """Run the substrate probe in a fresh process; return its stdout."""
    proc = subprocess.run(
        [sys.executable, _PROBE, str(steps), str(sigma), str(dt), str(kT)],
        cwd=_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"probe crashed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def _run_sort_probe(steps, clamp, sigma=0.5, kT=0.1, lth=1e-3, dt=1e-3, cut=1.9, extra_env=None):
    """Run the reconnection+noise sort probe in a fresh process; return its stdout.

    `extra_env` is merged over the inherited environment -- used to set
    TF_VERTEX_NO_VOLUME_REPAIR=1 (disable the native orientation repair) so a test can
    show the repair is what keeps the unclamped noisy sort stable.
    """
    env = {**os.environ, **(extra_env or {})}
    proc = subprocess.run(
        [sys.executable, _SORT_PROBE, str(steps), str(sigma), str(kT), str(lth),
         str(dt), str(cut), str(clamp)],
        cwd=_ROOT, capture_output=True, text=True, timeout=600, env=env,
    )
    assert proc.returncode == 0, f"sort probe crashed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def _verdict(stdout):
    """Parse the probe's final 'VERDICT ... worst_min=.. worst_max=.. STABLE/UNSTABLE' line."""
    line = next(ln for ln in stdout.splitlines() if ln.startswith("VERDICT"))
    toks = dict(t.split("=") for t in line.split() if "=" in t)
    return float(toks["worst_min"]), float(toks["worst_max"]), line.strip().split()[-1]


@pytest.mark.parametrize("sigma", [0.0, 1.0])
def test_periodic_substrate_integrates_stably(sigma):
    """A space-filling periodic foam must not invert (min_vol>0) or inflate (max_vol<3*V0)
    over thousands of steps. sigma=0 is the pure-substrate gate; sigma=1 adds heterotypic
    tension (cells deform but must stay valid)."""
    V0 = 4.0
    out = _run_probe(steps=3000, sigma=sigma, dt=1e-4, kT=0.0)
    worst_min, worst_max, verdict = _verdict(out)
    assert verdict == "STABLE", f"sigma={sigma} periodic substrate is UNSTABLE:\n{out}"
    assert worst_min > 0.0, f"a cell inverted (min_vol={worst_min}) at sigma={sigma}"
    assert worst_max < 3.0 * V0, f"a cell inflated (max_vol={worst_max}) at sigma={sigma}"


def test_periodic_substrate_stable_with_thermal_noise():
    """The fix lets a LARGE cutoff be used, so the engine thermal noise stays alive AND the
    foam is stable -- the central point (the old small-cutoff 'workaround' froze the mesh
    and killed the noise). Integrate with kT=0.1 white noise and require no inversion."""
    V0 = 4.0
    out = _run_probe(steps=3000, sigma=1.0, dt=1e-4, kT=0.1)
    worst_min, worst_max, verdict = _verdict(out)
    assert verdict == "STABLE", f"noisy periodic substrate is UNSTABLE:\n{out}"
    assert worst_min > 0.0, f"a cell inverted under thermal noise (min_vol={worst_min})"
    assert worst_max < 3.0 * V0, f"a cell inflated under thermal noise (max_vol={worst_max})"


def test_periodic_sort_stable_with_reconnection_and_noise():
    """The session's target gate: a periodic two-type sort with native I<->H reconnection ON
    AND thermal noise must stay VALID (no cell inverts/inflates) through many reconnections.

    This is the combination the substrate gate cannot exercise. The blow-up it locks out: a
    reconnection places two vertices Lth=1e-3 apart, then one thermal kick (DISP_STD ~14*Lth)
    throws one past its neighbour, everting the cell. The fix is the per-vertex noise
    trust-region clamp (clamp=0.4); see probe_periodic_sort.py / sort_periodic_oracle.py."""
    V0 = 1.0
    out = _run_sort_probe(steps=2000, clamp=0.4)
    worst_min, worst_max, verdict = _verdict(out)
    assert verdict == "STABLE", f"clamped periodic sort is UNSTABLE:\n{out}"
    assert worst_min > 0.0, f"a cell inverted during the sort (min_vol={worst_min})"
    assert worst_max < 4.0 * V0, f"a cell inflated during the sort (max_vol={worst_max})"


def test_periodic_sort_stable_with_native_volume_repair():
    """The faithful unblock: with the NATIVE orientation repair on (default) the noisy
    reconnecting sort stays valid WITHOUT the Python noise clamp (clamp=0.0). This is the
    paper's own mechanism (3DVertVor Cell.cpp:216-221, oracle stabilizer #3): a transient
    eversion has its signed volume abs'd and its winding parity flipped so the
    VolumeConstraint force stays restoring and the cell recovers, instead of the force
    running the cell away into inflation. It supersedes the per-vertex noise trust-region
    clamp the prototype used before. (worst_min>0 is automatic here -- the repair reports
    |volume| -- so the load-bearing discriminator is no inflation, worst_max<3*V0, plus the
    counter-test below that the same run blows up when the repair is disabled.)"""
    V0 = 1.0
    out = _run_sort_probe(steps=1000, clamp=0.0)
    worst_min, worst_max, verdict = _verdict(out)
    assert verdict == "STABLE", f"native-repaired unclamped sort is UNSTABLE:\n{out}"
    assert worst_max < 3.0 * V0, f"a cell inflated despite the repair (max_vol={worst_max})"
    assert worst_min > 0.0, f"volume should be reported positive under the repair (got {worst_min})"


def test_periodic_sort_unrepaired_unclamped_noise_inverts_a_cell():
    """Load-bearing counter-test: disable the native orientation repair
    (TF_VERTEX_NO_VOLUME_REPAIR=1) AND the noise clamp (clamp=0.0) and the SAME sort everts a
    cell within a few hundred steps -- proving the repair (not some incidental stability) is
    what holds the noisy reconnecting geometry together. If this ever passes as STABLE, the
    repair has become a no-op and the gate above is vacuous."""
    out = _run_sort_probe(steps=1000, clamp=0.0, extra_env={"TF_VERTEX_NO_VOLUME_REPAIR": "1"})
    worst_min, _worst_max, verdict = _verdict(out)
    assert verdict == "UNSTABLE", (
        f"unrepaired unclamped sort was expected to evert a cell but stayed stable "
        f"(worst_min={worst_min}); the repair may have become a no-op:\n{out}")
    assert worst_min <= 0.0, f"expected an eversion (min_vol<=0) but worst_min={worst_min}"
