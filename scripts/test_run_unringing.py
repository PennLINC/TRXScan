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


def _zero_filled_edge(n=64):
    """The OLD TRXScan mechanism: crop k-space, then zero-fill back to `n`.

    Rings at period 2/(1-r) > 2 voxels, so the cutoff is no longer at the reconstruction Nyquist.
    Retained only to document why that made the simulator unusable as a benchmark.
    """
    x = np.arange(n)
    obj = (x >= n // 2 + 0.5).astype(float)
    k = np.fft.fftshift(np.fft.fft(obj))
    keep = np.zeros(n, bool)
    keep[n // 4:3 * n // 4] = True
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
def test_the_old_zero_filled_mechanism_is_not_correctable():
    """Why the forward model had to change, measured rather than argued.

    Zero-filled truncation moves the Fourier cutoff off the reconstruction Nyquist, so the ripple
    period is no longer 2 voxels and the sub-voxel-shift assumption these tools rest on does not
    hold. mrdegibbs achieves essentially nothing on it and dipy makes it measurably worse -- while
    both remove ~90% from the crop-and-reconstruct edge.
    """
    old, new = _zero_filled_edge(), _crop_recon_edge()
    for m in ("dipy", "mrdegibbs"):
        if m not in available_methods():
            continue
        o_out, _ = run_method(m, old, np.zeros_like(old))
        n_out, _ = run_method(m, new, np.zeros_like(new))
        old_gain = _ripple(o_out) / _ripple(old)
        new_gain = _ripple(n_out) / _ripple(new)
        assert new_gain < 0.25, f"{m}: expected a large reduction on crop+recon, got {new_gain:.3f}"
        assert old_gain > 0.9, (
            f"{m}: zero-filled ringing should be largely uncorrectable, got {old_gain:.3f}"
        )


def test_available_methods_always_includes_the_control():
    assert "none" in available_methods(), "the no-op control is not optional"
