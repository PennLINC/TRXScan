"""Compare simulated and real data by how mrdegibbs BEHAVES on them (spec 4.2).

This is a plausibility comparison, not a source of theoretical constants: the analytic covariance
stays derived from the simulator's own reconstruction operator.
"""
import numpy as np
from behavioural_validation import spectral_profile, compare_profiles


def test_profile_localises_nyquist_ripple_in_the_top_bin():
    n = 64
    x = np.arange(n)
    nyq = np.tile((-1.0) ** x, (n, 1))
    prof = spectral_profile(nyq)
    assert int(np.argmax(prof)) >= len(prof) - 2, prof


def test_comparison_flags_a_frequency_mismatch():
    n = 64
    x = np.arange(n)
    at_nyquist = np.tile((-1.0) ** x, (n, 1))
    slower = np.tile(np.cos(2 * np.pi * x / 4.0), (n, 1))     # period 4, not 2
    ok, detail = compare_profiles(spectral_profile(at_nyquist), spectral_profile(slower))
    assert not ok, detail


def test_comparison_accepts_matching_profiles():
    n = 64
    x = np.arange(n)
    a = np.tile((-1.0) ** x, (n, 1))
    b = np.tile((-1.0) ** x * 0.5, (n, 1))     # same frequency, different amplitude
    ok, detail = compare_profiles(spectral_profile(a), spectral_profile(b))
    assert ok, detail
