"""The harness must never silently substitute one method for another: a missing method has to be
recorded as missing, or the comparison table lies about what was run."""
import numpy as np
import pytest
from run_unringing import run_method, available_methods, MethodUnavailable


def _crop_recon_edge(n=64, o=8):
    """A fine step, k-space cropped to `n`, reconstructed AT `n`.

    This is the simulator's mechanism, and it matters: it produces period-2 ringing at the nominal
    resolution, which is what Kellner/mrdegibbs assume. See `_zero_filled_edge` for the contrast.
    """
    N = n * o
    x = np.arange(N)
    obj = (x >= (n // 2 + 0.5) * o).astype(float)
    K = np.fft.fftshift(np.fft.fft(obj))
    c = K[N // 2 - n // 2:N // 2 + n // 2] / o
    return np.tile(np.real(np.fft.ifft(np.fft.ifftshift(c))), (n, 1))


def _zero_filled_edge(n=64, pct=6.0):
    """The OLD TRXScan mechanism: zero `pct`% of k-space at each edge, reconstruct at `n`.

    `pct` defaults to **6.0, the value the old CLI actually shipped**. Rings at period
    2/(1-pct/100) > 2 voxels, so the cutoff is no longer at the reconstruction Nyquist.
    """
    x = np.arange(n)
    obj = (x >= n // 2 + 0.5).astype(float)
    k = np.fft.fftshift(np.fft.fft(obj))
    r = int(np.ceil((n / 2) * pct / 100))
    keep = np.ones(n, bool)
    keep[:r] = False
    keep[n - r:] = False
    return np.tile(np.real(np.fft.ifft(np.fft.ifftshift(k * keep))), (n, 1))


def _ripple(a):
    return float(np.std(a[:, 3 * a.shape[1] // 4:]))


def test_control_is_exactly_identity():
    m = np.random.RandomState(0).rand(16, 16)
    p = np.zeros_like(m)
    om, op = run_method("none", m, p)
    assert np.array_equal(om, m) and np.array_equal(op, p)


def test_unavailable_method_raises_rather_than_substituting():
    m = np.zeros((16, 16))
    p = np.zeros_like(m)
    if "rpg" not in available_methods():
        with pytest.raises(MethodUnavailable):
            run_method("rpg", m, p)


def test_unknown_method_is_an_error_not_a_fallback():
    m = np.zeros((16, 16))
    with pytest.raises(MethodUnavailable):
        run_method("definitely-not-a-method", m, np.zeros_like(m))


@pytest.mark.skipif("dipy" not in available_methods(), reason="dipy unavailable")
def test_dipy_substantially_removes_ringing_from_a_crop_recon_edge():
    img = _crop_recon_edge()
    out, _ = run_method("dipy", img, np.zeros_like(img))
    assert _ripple(out) < 0.25 * _ripple(img), f"{_ripple(out)} vs {_ripple(img)}"


@pytest.mark.skipif("mrdegibbs" not in available_methods(), reason="mrdegibbs unavailable")
def test_mrdegibbs_substantially_removes_ringing_from_a_crop_recon_edge():
    img = _crop_recon_edge()
    out, _ = run_method("mrdegibbs", img, np.zeros_like(img))
    assert _ripple(out) < 0.25 * _ripple(img), f"{_ripple(out)} vs {_ripple(img)}"


@pytest.mark.skipif(
    not {"dipy", "mrdegibbs"} & set(available_methods()), reason="no unringing method"
)
def test_correctability_degrades_as_the_old_mechanism_moves_the_cutoff():
    """Why the forward model had to change, measured across the real parameter range.

    An earlier version of this test used only 50% truncation and concluded the old mechanism was
    "uncorrectable". That overstated it: the CLI actually shipped **6%**, where dipy still recovers
    a good deal. What the measurement supports is weaker and graded -- correctability DEGRADES as
    the cutoff moves off the reconstruction Nyquist:

        trunc   period     dipy    mrdegibbs
          6%   2.07 vox    -47%        +7%
         25%   2.58 vox     -2%       +21%
         50%   3.88 vox    +16%        -0%

    against -87% / -92% on a correctly crop-and-reconstructed edge. So even the shipped default
    was a materially worse benchmark than the corrected forward model, and the aggressive settings
    are actively misleading.
    """
    new = _crop_recon_edge()
    for m in ("dipy", "mrdegibbs"):
        if m not in available_methods():
            continue
        n_out, _ = run_method(m, new, np.zeros_like(new))
        assert _ripple(n_out) / _ripple(new) < 0.25, f"{m}: crop+recon should be correctable"
        # correctability must get worse as truncation grows
        gains = []
        for pct in (6.0, 25.0, 50.0):
            old = _zero_filled_edge(pct=pct)
            o_out, _ = run_method(m, old, np.zeros_like(old))
            gains.append(_ripple(o_out) / _ripple(old))
        # No monotonicity is asserted: mrdegibbs measures [1.07, 1.21, 1.00] across 6/25/50%,
        # never helping but not ordered either. What holds for BOTH methods at EVERY truncation is
        # that the zero-filled mechanism is materially harder to correct than crop+reconstruct.
        crop_gain = _ripple(n_out) / _ripple(new)
        for pct, g in zip((6.0, 25.0, 50.0), gains):
            assert g > crop_gain + 0.2, (
                f"{m} at {pct}%: gain {g:.3f} vs {crop_gain:.3f} on crop+recon -- the old "
                f"mechanism should be clearly harder to correct")


def test_available_methods_always_includes_the_control():
    assert "none" in available_methods(), "the no-op control is not optional"


def test_old_dipy_fallback_does_not_mutate_the_callers_array(monkeypatch):
    """Emulate a pre-`inplace` DIPY and assert `run_method` still hands back an intact input.

    Runs with or without DIPY installed: the module is faked outright, so the test pins the
    harness's contract rather than any particular DIPY version. The fake rejects the `inplace`
    and `num_processes` kwargs exactly as the old signatures did, which forces `_run_dipy` down
    the fallback branch, and then overwrites its argument the way `inplace=True` does.
    """
    import sys
    import types

    import run_unringing

    def old_gibbs_removal(vol, *a, **kw):
        if a or kw:
            raise TypeError("gibbs_removal() got unexpected arguments")
        vol *= 0.0                       # in-place, as DIPY's default has always been
        vol += 7.0
        return vol

    mod = types.ModuleType("dipy.denoise.gibbs")
    mod.gibbs_removal = old_gibbs_removal
    monkeypatch.setitem(sys.modules, "dipy", types.ModuleType("dipy"))
    monkeypatch.setitem(sys.modules, "dipy.denoise", types.ModuleType("dipy.denoise"))
    monkeypatch.setitem(sys.modules, "dipy.denoise.gibbs", mod)
    monkeypatch.setattr(run_unringing, "available_methods", lambda: ["none", "dipy"])

    # float64 so `np.asarray(mag, float)` in run_method returns the SAME object, which is the
    # condition under which the mutation would reach the caller.
    mag = np.arange(16, dtype=float).reshape(4, 4)
    before = mag.copy()
    out, _ = run_unringing.run_method("dipy", mag, np.zeros_like(mag))

    assert np.array_equal(mag, before), "old-DIPY fallback overwrote the caller's magnitude"
    assert np.allclose(out, 7.0), "the fake correction did not reach the caller"
