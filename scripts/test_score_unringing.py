"""The benchmark's value is the DECOMPOSITION of error, not a single distance.

A method that merely blurs suppresses ringing perfectly while destroying resolution. If the suite
reported one number, that failure mode would be indistinguishable from a genuine unringing. These
tests pin the separation.
"""
import numpy as np
import pytest
from score_unringing import score, residual_alignment


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


def _complex_edge(n=64, ring=0.0):
    x = np.arange(n)
    img = (x >= n // 2 + 0.5).astype(float)
    if ring:
        img = img + ring * ((-1.0) ** x) * np.exp(-np.abs(x - n // 2) / 6.0)
    return np.tile(img, (n, 1)).astype(complex)


def test_residual_alignment_spans_zero_to_one():
    ref = _complex_edge()
    ctl = _complex_edge(ring=0.09)
    assert abs(residual_alignment(ctl, ref, ctl) - 1.0) < 1e-9, "control removed nothing"
    assert abs(residual_alignment(ref, ref, ctl)) < 1e-9, "perfect recovery"
    half = ref + 0.5 * (ctl - ref)
    assert abs(residual_alignment(half, ref, ctl) - 0.5) < 1e-6, "half removed"


def test_residual_alignment_is_invariant_to_global_rotation():
    """The reason this metric exists alongside the Nyquist projection: complex-aware methods must
    not be scored differently just because the object carries a different global phase."""
    ref, ctl = _complex_edge(), _complex_edge(ring=0.09)
    base = residual_alignment(ref + 0.4 * (ctl - ref), ref, ctl)
    for a in (0.3, 1.1, 2.7):
        r = np.exp(1j * a)
        rot = residual_alignment((ref + 0.4 * (ctl - ref)) * r, ref * r, ctl * r)
        assert abs(rot - base) < 1e-9, f"alignment moved under rotation {a}: {rot} vs {base}"


def test_residual_alignment_flags_amplification():
    ref, ctl = _complex_edge(), _complex_edge(ring=0.09)
    worse = ref + 1.5 * (ctl - ref)
    assert residual_alignment(worse, ref, ctl) > 1.0


def _complex_ringing_edge(n=64):
    """A step whose complex Gibbs sidelobes alternate on BOTH sides of the edge."""
    x = np.arange(n)
    ref = (x >= n // 2 + 0.5).astype(float).astype(complex)
    lobes = 0.09 * ((-1.0) ** x) * np.exp(-np.abs(x - n // 2) / 6.0)
    return np.tile(ref, (n, 1)), np.tile(ref + lobes, (n, 1))


def test_dark_side_ringing_is_scored_not_discarded():
    """The exact round-4 defect: bright-side-only masking discarded half the artifact.

    The independent cross-check cannot protect against this -- both scorers used the bright-only
    interpretation and AGREED WITH EACH OTHER WHILE BOTH WERE WRONG. This pins the physical
    property instead: a method that removes only the bright-side sidelobes has NOT removed the
    artifact, and the primary complex score must say so.
    """
    from score_unringing import residual_energy_ratio, PHASE_ENCODE_AXIS
    ref, ctl = _complex_ringing_edge()
    n = ref.shape[-1]
    bright = np.abs(ref) > 0.5 * np.abs(ref).max()

    # Fix only where the reference is bright; leave the dark side exactly as acquired.
    est = ctl.copy()
    est[bright] = ref[bright]

    full = residual_energy_ratio(est, ref, ctl, axis=PHASE_ENCODE_AXIS)
    bright_only = residual_energy_ratio(est, ref, ctl, axis=PHASE_ENCODE_AXIS, bright=0.5)

    assert bright_only < 0.05, (
        f"a bright-side-only fix should look perfect to the bright-only metric, got {bright_only:.3f}")
    assert full > 0.5, (
        f"the primary metric must still see the untouched dark-side ringing, got {full:.3f}")


def test_dark_side_alternation_survives_the_complex_reconstruction():
    """Why the bright-only mask was wrong: magnitude rectifies, complex does not."""
    ref, ctl = _complex_ringing_edge()
    row = ref.shape[0] // 2
    n = ref.shape[-1]
    dark = slice(n // 2 - 9, n // 2 - 2)
    mag_resid = (np.abs(ctl) - np.abs(ref))[row, dark]
    cpx_resid = (ctl - ref).real[row, dark]
    assert np.all(mag_resid > 0), "magnitude rectifies the dark side"
    assert np.any(np.diff(np.sign(cpx_resid)) != 0), "complex must retain the alternation"


def test_worst_edge_sharpness_catches_a_one_sided_blur():
    """The failure mode that motivated per-edge sharpness.

    `_sharpness` takes the profile MAXIMUM gradient, so blurring one boundary of a box while
    leaving the other perfectly sharp leaves that maximum untouched -- the whole-profile ratio
    still looks healthy. Rule (c) would have passed a method that destroyed half the resolution.
    """
    from score_unringing import _sharpness, _sharpness_per_edge

    n = 64
    q = n // 4
    ax = np.arange(n)
    cov = np.clip(np.minimum(ax + 1.0, 3 * q + 0.5) - np.maximum(ax, q + 0.5), 0.0, 1.0)
    ref = np.tile(cov, (n, 1))

    # Blur ONLY the left boundary; leave the right one bit-identical to the reference.
    est = ref.copy()
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.2) ** 2)
    k /= k.sum()
    left = slice(0, n // 2)
    for r in range(n):
        est[r, left] = np.convolve(ref[r], k, mode="same")[left]

    whole = _sharpness(est) / _sharpness(ref)
    per = _sharpness_per_edge(est, ref)

    assert len(per) == 2, f"expected two edges, got {per}"
    assert whole > 0.9, (
        f"whole-profile sharpness should look fine -- that is the blind spot -- got {whole:.3f}")
    assert min(per) < 0.6, f"the blurred edge must be detected: per-edge {per}"
    assert max(per) > 0.9, f"the untouched edge must stay sharp: per-edge {per}"
