"""A deliberately naive second scorer, for cross-checking `score_unringing`.

Rationale: the primary scorer accumulated six independent defects during development -- wrong
axis, edge included in the projection window, magnitude-rectification blindness, a real-only
complex projection, a rolling-mean high-pass that ranked blur as more oscillatory than ringing,
and a whole-profile centroid that reported a +2 voxel shift as -14. Every one produced
confident-looking output while the suite ran clean.

So this module recomputes the same quantities the dumbest way that is still correct, sharing no
code with the primary implementation. It is slower and cruder. If the two disagree materially on
analytic fixtures, at least one is wrong and CI should fail.
"""
import numpy as np

__all__ = ["naive_oscillatory", "naive_alignment", "compare_scorers"]


def _bright_sidelobe_indices(ref_profile, guard=2, half_width=10, bright=0.5):
    """Indices beside the edge where the reference is bright. Written independently."""
    g = [abs(ref_profile[i + 1] - ref_profile[i]) for i in range(len(ref_profile) - 1)]
    peak = max(range(len(g)), key=lambda i: g[i])
    hi = max(ref_profile)
    out = []
    for i in range(len(ref_profile)):
        if abs(i - peak) <= guard or abs(i - peak) > half_width:
            continue
        if ref_profile[i] > bright * hi:
            out.append(i)
    return out


def naive_oscillatory(est, ref):
    """Nyquist amplitude by explicit Python loops over rows. `est`/`ref` are complex 2D arrays."""
    est, ref = np.asarray(est), np.asarray(ref)
    axis = 0 if np.abs(np.diff(np.abs(ref), axis=0)).mean() > \
        np.abs(np.diff(np.abs(ref), axis=1)).mean() else 1
    if axis == 0:
        est, ref = est.T, ref.T
    tot, rows = 0.0, 0
    for r in range(est.shape[0]):
        rp = list(np.abs(ref[r]))
        idx = _bright_sidelobe_indices(rp)
        if not idx:
            continue
        acc = 0j
        for i in idx:
            acc += (est[r][i] - ref[r][i]) * ((-1.0) ** i)
        tot += abs(acc / len(idx)) ** 2
        rows += 1
    return float(np.sqrt(tot / rows)) if rows else float("nan")


def naive_alignment(est, ref, control):
    """`|<Rm, R0>| / ||R0||^2` by explicit loops."""
    est, ref, control = (np.asarray(a) for a in (est, ref, control))
    axis = 0 if np.abs(np.diff(np.abs(ref), axis=0)).mean() > \
        np.abs(np.diff(np.abs(ref), axis=1)).mean() else 1
    if axis == 0:
        est, ref, control = est.T, ref.T, control.T
    num, den = 0j, 0.0
    for r in range(est.shape[0]):
        idx = _bright_sidelobe_indices(list(np.abs(ref[r])))
        for i in idx:
            rm = est[r][i] - ref[r][i]
            r0 = control[r][i] - ref[r][i]
            num += rm * np.conj(r0)
            den += abs(r0) ** 2
    return float(abs(num) / den) if den > 0 else float("nan")


def compare_scorers(est, ref, control, rtol=0.02, atol=1e-6):
    """Run both implementations; return (agree, detail)."""
    from score_unringing import residual_alignment, score
    s = score(np.abs(est), np.angle(est), np.abs(ref), np.angle(ref),
              control_mag=np.abs(control), control_phase=np.angle(control))
    pairs = [
        ("oscillatory_residual", s["oscillatory_residual"], naive_oscillatory(est, ref)),
        ("residual_alignment", s["residual_alignment"], naive_alignment(est, ref, control)),
    ]
    bad = []
    for name, a, b in pairs:
        if np.isnan(a) and np.isnan(b):
            continue
        if not np.isclose(a, b, rtol=rtol, atol=atol):
            bad.append(f"{name}: primary {a:.6g} vs naive {b:.6g}")
    return (not bad), "; ".join(bad)
