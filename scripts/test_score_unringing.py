"""The benchmark's value is the DECOMPOSITION of error, not a single distance.

A method that merely blurs suppresses ringing perfectly while destroying resolution. If the suite
reported one number, that failure mode would be indistinguishable from a genuine unringing. These
tests pin the separation.
"""
import numpy as np
import pytest
from score_unringing import score


def _edge(n=64, ring=0.0, blur=0.0):
    x = np.arange(n)
    img = (x >= n // 2).astype(float)
    if ring:
        # (-1)**x, not sin(pi*x): the latter is identically zero on an integer grid,
        # which silently made the 'ringing' image equal to the reference.
        img = img + ring * ((-1.0) ** x) * np.exp(-np.abs(x - n // 2) / 6.0)
    if blur:
        k = np.exp(-0.5 * (np.arange(-9, 10) / blur) ** 2)
        k /= k.sum()
        img = np.convolve(img, k, mode="same")
    return np.tile(img, (n, 1))


def test_blurring_and_unringing_are_distinguishable():
    ref = _edge()
    rung = _edge(ring=0.09)      # ringing, but sharp
    blurred = _edge(blur=2.0)    # no ringing, but soft
    z = np.zeros_like(ref)
    s_ring = score(rung, z, ref, z)
    s_blur = score(blurred, z, ref, z)
    assert s_blur["oscillatory_residual"] < s_ring["oscillatory_residual"], (
        "the blurred image must win on oscillation")
    assert s_blur["edge_sharpness"] < s_ring["edge_sharpness"], (
        "and must lose on sharpness -- otherwise the two are indistinguishable")


def test_perfect_reconstruction_scores_zero_error_and_unit_sharpness():
    ref = _edge()
    z = np.zeros_like(ref)
    s = score(ref, z, ref, z)
    assert s["oscillatory_residual"] < 1e-12
    assert s["complex_rmse"] < 1e-12
    assert abs(s["edge_sharpness"] - 1.0) < 1e-9, "sharpness is normalized to the reference"
    assert abs(s["edge_location_bias"]) < 1e-9


def test_phase_error_ignores_near_zero_magnitude():
    n = 32
    ref_mag = np.zeros((n, n))
    ref_mag[:, n // 2:] = 1.0
    ref_ph = np.zeros((n, n))
    est_ph = ref_ph.copy()
    est_ph[:, : n // 2] = 3.0            # garbage phase where magnitude is zero
    s = score(ref_mag, est_ph, ref_mag, ref_ph)
    assert s["phase_rmse_masked"] < 1e-9, "phase error must be magnitude-masked"


def test_phase_error_is_detected_where_magnitude_is_real():
    n = 32
    ref_mag = np.ones((n, n))
    ref_ph = np.zeros((n, n))
    est_ph = np.full((n, n), 0.4)
    s = score(ref_mag, est_ph, ref_mag, ref_ph)
    assert abs(s["phase_rmse_masked"] - 0.4) < 1e-6, s["phase_rmse_masked"]


def test_phase_error_wraps_correctly():
    # A 2pi offset is the same phase; a naive difference would report 2pi of error.
    n = 16
    mag = np.ones((n, n))
    s = score(mag, np.full((n, n), np.pi - 0.05), mag, np.full((n, n), -np.pi + 0.05))
    assert s["phase_rmse_masked"] < 0.15, f"wrapped difference expected ~0.1, got {s}"


def test_edge_location_bias_detects_a_shift():
    ref = _edge(n=64)
    shifted = np.roll(ref, 2, axis=1)
    z = np.zeros_like(ref)
    s = score(shifted, z, ref, z)
    assert abs(s["edge_location_bias"] - 2.0) < 0.25, s["edge_location_bias"]
