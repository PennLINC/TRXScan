"""Calibration statistics for the object-phase model (spec 4.2).

Term 2 (pre-readout background) is TUNED as an effective benchmark parameter, not fitted: observed
phase mixes magnetization, coil and combination, scanner conventions and possibly reconstruction
filtering, and only some of that precedes Fourier encoding. Term 3 (diffusion) IS fitted, but only
with the thermal-noise contribution modelled -- otherwise falling SNR at high b biases p upward.
"""
import numpy as np
import pytest
from calibrate_phase import (
    siemens_phase_to_radians, phase_spatial_stats, fit_b_dependence,
    circular_sd, WRAP_CEILING,
)


def test_siemens_rescaling_maps_the_integer_range_onto_pi():
    raw = np.array([-4096.0, -2048.0, 0.0, 2048.0, 4095.0])
    rad = siemens_phase_to_radians(raw)
    assert abs(rad[0] + np.pi) < 1e-9
    assert abs(rad[2]) < 1e-12
    assert abs(rad[1] + np.pi / 2) < 1e-9
    assert rad[-1] < np.pi and rad[-1] > np.pi - 0.01


def test_rescaling_refuses_data_that_is_already_radians():
    # Guard against double-scaling: a volume already in radians must not be silently rescaled.
    with pytest.raises(ValueError):
        siemens_phase_to_radians(np.linspace(-np.pi, np.pi, 100))


def test_stats_recover_a_known_synthetic_field():
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    true_grad = 0.05
    ph = np.angle(np.exp(1j * (true_grad * xx)))       # wrapped linear ramp
    mag = np.ones((n, n))
    s = phase_spatial_stats(ph[None], mag[None], (mag > 0.5)[None])
    # grad_rms is the RMS gradient MAGNITUDE, so a pure-x ramp of slope g reads as g.
    assert abs(s["grad_rms"] - true_grad) < 0.01, s
    assert s["wrap_density"] > 0, "a ramp spanning many cycles must show wraps"


def test_flat_phase_has_no_wraps_and_zero_gradient():
    n = 32
    ph = np.zeros((1, n, n))
    mag = np.ones((1, n, n))
    s = phase_spatial_stats(ph, mag, mag > 0.5)
    assert s["grad_rms"] < 1e-9 and s["wrap_density"] == 0


def test_correlation_length_grows_with_smoothness():
    n = 96
    yy, xx = np.mgrid[0:n, 0:n]
    mag = np.ones((1, n, n))
    smooth = phase_spatial_stats(np.angle(np.exp(1j * 0.01 * xx))[None], mag, mag > 0.5)
    rough = phase_spatial_stats(np.angle(np.exp(1j * 0.30 * xx))[None], mag, mag > 0.5)
    assert smooth["corr_length_vox"] > rough["corr_length_vox"], (smooth, rough)


def test_snr_correction_removes_the_upward_bias():
    rng = np.random.RandomState(0)
    bvals = np.array([1000, 2000, 3000, 4000, 5000], float)
    p_true, a_true = 0.5, 0.002
    sig = a_true * bvals ** p_true
    # SNR must fall hard enough that the thermal floor actually dominates at high b -- otherwise
    # there is no bias for the correction to remove and the test asserts nothing.
    snr = 40.0 * np.exp(-bvals / 1500.0)
    obs = np.sqrt(sig ** 2 + (1.0 / snr) ** 2)        # thermal phase noise adds in quadrature
    shots = [rng.normal(0, s, 4000) for s in obs]
    out = fit_b_dependence(shots, bvals, snr)
    assert abs(out["p"] - p_true) < 0.12, out
    assert out["p_naive"] > out["p"] + 0.05, f"naive fit should be biased upward: {out}"
    assert out["snr_corrected"] is True


def test_linear_sd_saturates_where_circular_sd_does_not():
    """Why the fit uses circular statistics: wrapped phase has a hard linear-SD ceiling."""
    rng = np.random.RandomState(1)
    tight = rng.normal(0, 0.5, 200000)
    wide = rng.normal(0, 4.0, 200000)          # true spread well beyond pi
    wrap = lambda a: np.angle(np.exp(1j * a))
    assert np.std(wrap(tight)) < 0.9 * WRAP_CEILING
    assert np.std(wrap(wide)) > 0.97 * WRAP_CEILING, "a wide spread must saturate the linear SD"
    # circular SD keeps ordering the two correctly
    assert circular_sd(wrap(wide)) > 2.0 * circular_sd(wrap(tight))


def test_fit_reports_saturation_so_a_bad_shell_is_visible():
    rng = np.random.RandomState(2)
    bvals = np.array([1000.0, 2000.0, 3000.0])
    shots = [np.angle(np.exp(1j * rng.normal(0, s, 20000))) for s in (0.4, 1.0, 6.0)]
    out = fit_b_dependence(shots, bvals, np.array([50.0, 40.0, 30.0]))
    assert out["statistic"] == "circular"
    assert out["saturation"][-1] > 0.9, "the saturated shell must be flagged"
    assert out["saturation"][0] < 0.5
