"""The acceptance criterion is physical consistency, not a ranking.

Rankings between unringing methods are recorded, never asserted: the suite's job is to show that
varying PF, phase, noise and windowing changes behaviour in interpretable ways.
"""
import pytest
from acceptance import check_consistency


def _row(method, pf, phase, osc, sharp=1.0, phase_err=0.0, noisy=False, window="None"):
    return {"method": method, "pf": pf, "phase": phase, "noisy": noisy, "window": window,
            "oscillatory_residual": osc, "edge_sharpness": sharp,
            "phase_rmse_masked": phase_err}


def test_partial_fourier_trend_is_reported_not_asserted():
    """The PF axis is a known limitation: a Nyquist-projection metric cannot see PF ringing.

    Rows whose only anomaly is a PF trend must NOT fail acceptance, in either direction.
    """
    for a, b in ((0.05, 0.01), (0.01, 0.05)):
        rows = [_row("none", 1.0, "nophase", 0.09), _row("none", 0.75, "nophase", 0.11),
                _row("x", 1.0, "nophase", a, sharp=0.95),
                _row("x", 0.75, "nophase", b, sharp=0.95)]
        ok, reasons = check_consistency(rows)
        assert ok, (a, b, reasons)


def test_flags_a_method_that_does_not_beat_the_control():
    rows = [_row("none", 1.0, "nophase", 0.09), _row("x", 1.0, "nophase", 0.12)]
    ok, reasons = check_consistency(rows)
    assert not ok and any("control" in r.lower() for r in reasons), reasons


def test_flags_a_method_that_only_blurs():
    rows = [_row("none", 1.0, "nophase", 0.09, sharp=1.0),
            _row("x", 1.0, "nophase", 0.01, sharp=0.3)]
    ok, reasons = check_consistency(rows)
    assert not ok and any("sharp" in r.lower() for r in reasons), reasons


def test_accepts_a_physically_consistent_result():
    rows = [
        _row("none", 1.0, "nophase", 0.090), _row("none", 0.75, "nophase", 0.110),
        _row("x", 1.0, "nophase", 0.010, sharp=0.95),
        _row("x", 0.75, "nophase", 0.030, sharp=0.94),
        _row("none", 1.0, "ramp", 0.090), _row("x", 1.0, "ramp", 0.012, sharp=0.95),
    ]
    ok, reasons = check_consistency(rows)
    assert ok, reasons


def test_does_not_assert_a_ranking_between_methods():
    # Two methods, different quality, both physically consistent: that is not a failure.
    rows = [
        _row("none", 1.0, "nophase", 0.09), _row("none", 0.75, "nophase", 0.11),
        _row("a", 1.0, "nophase", 0.010, sharp=0.98), _row("a", 0.75, "nophase", 0.030, sharp=0.97),
        _row("b", 1.0, "nophase", 0.050, sharp=0.99), _row("b", 0.75, "nophase", 0.070, sharp=0.98),
    ]
    ok, reasons = check_consistency(rows)
    assert ok, reasons


def test_control_is_matched_on_window_not_just_pf_and_phase():
    """Apodized data has little ringing left, so its control residual is far lower.

    Comparing an unapodized method row against an apodized control makes every method look worse
    than doing nothing. Each must be compared within its own window.
    """
    rows = [
        _row("none", 1.0, "nophase", 0.020, window="None"),
        _row("none", 1.0, "nophase", 0.002, window="Hann"),
        _row("x", 1.0, "nophase", 0.005, sharp=0.95, window="None"),   # beats its own control
        _row("x", 1.0, "nophase", 0.001, sharp=0.95, window="Hann"),   # also beats its own
    ]
    ok, reasons = check_consistency(rows)
    assert ok, reasons


def test_conditions_with_no_ringing_left_are_exempt_from_beating_the_control():
    """Apodized data has almost no ringing, so a method can only perturb it.

    Penalising that would mark correct behaviour as a failure. The exemption is relative to the
    median control residual, so it self-calibrates rather than hard-coding a magnitude.
    """
    rows = [
        _row("none", 1.0, "nophase", 0.020, window="None"),
        _row("none", 0.75, "nophase", 0.020, window="None"),
        _row("none", 1.0, "nophase", 0.001, window="Hann"),
        _row("x", 1.0, "nophase", 0.005, sharp=0.95, window="None"),
        _row("x", 0.75, "nophase", 0.008, sharp=0.95, window="None"),
        _row("x", 1.0, "nophase", 0.002, sharp=0.95, window="Hann"),   # worse, but exempt
    ]
    ok, reasons = check_consistency(rows)
    assert ok, reasons
