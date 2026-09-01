"""Two independent implementations must agree, or one of them is wrong."""
import numpy as np
import pytest
from score_crosscheck import compare_scorers, naive_alignment, naive_oscillatory


def _edge(n=64, ring=0.0, phase=0.0, blur=0.0):
    x = np.arange(n)
    img = (x >= n // 2 + 0.5).astype(float)
    if ring:
        img = img + ring * ((-1.0) ** x) * np.exp(-np.abs(x - n // 2) / 6.0)
    if blur:
        k = np.exp(-0.5 * (np.arange(-9, 10) / blur) ** 2)
        k /= k.sum()
        img = np.convolve(img, k, mode="same")
    return np.tile(img, (n, 1)).astype(complex) * np.exp(1j * phase)


@pytest.mark.parametrize("phase", [0.0, 0.7, 2.5])
@pytest.mark.parametrize("frac", [0.0, 0.5, 1.0])
def test_implementations_agree_across_phase_and_removal(phase, frac):
    ref = _edge(phase=phase)
    ctl = _edge(ring=0.09, phase=phase)
    est = ref + frac * (ctl - ref)
    ok, detail = compare_scorers(est, ref, ctl)
    assert ok, detail


def test_implementations_agree_on_a_blurring_method():
    ref, ctl = _edge(), _edge(ring=0.09)
    ok, detail = compare_scorers(_edge(blur=2.0), ref, ctl)
    assert ok, detail


def test_naive_alignment_matches_its_definition_exactly():
    ref, ctl = _edge(), _edge(ring=0.09)
    assert abs(naive_alignment(ctl, ref, ctl) - 1.0) < 1e-12
    assert abs(naive_alignment(ref, ref, ctl)) < 1e-12


def test_naive_oscillatory_is_zero_for_perfect_recovery():
    ref = _edge()
    assert naive_oscillatory(ref, ref) < 1e-12
