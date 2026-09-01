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

__all__ = ["naive_oscillatory", "naive_alignment", "naive_energy_ratio",
           "naive_artifact_norm", "compare_scorers"]


def _peak_indices(grad, thr_frac=0.35):
    """One index per PHYSICAL edge: contiguous above-threshold runs collapsed to their centroid.

    Independent of the primary, same definition. A half-voxel edge yields two adjacent
    above-threshold gradient samples, and treating each as its own peak reported the benchmark
    box's two edges per axis as four.
    """
    if not grad or max(grad) <= 0:
        return []
    thr = thr_frac * max(grad)
    out, i, n = [], 0, len(grad)
    while i < n:
        if grad[i] < thr:
            i += 1
            continue
        j = i
        while j + 1 < n and grad[j + 1] >= thr:
            j += 1
        w = grad[i:j + 1]
        tot = sum(w)
        out.append(int(round(sum(k * w[k - i] for k in range(i, j + 1)) / tot)))
        i = j + 1
    return out or [max(range(n), key=lambda i: grad[i])]


def _bright_sidelobe_indices(ref_profile, guard=2, half_width=10, bright=0.0, peaks=None):
    """Indices beside EVERY significant edge where the reference is bright.

    Written independently of the primary, but to the same definition -- including scoring both of
    a box's edges rather than only the strongest, and (with the default `bright = 0.0`) both the
    bright and dark sides, since the complex residual preserves alternation on both. When the primary was extended to both edges and
    this was not, the two disagreed by ~4% on the partial-Fourier box, which is exactly what this
    cross-check exists to surface.
    """
    n = len(ref_profile)
    if peaks is None:
        peaks = _peak_indices([abs(ref_profile[i + 1] - ref_profile[i]) for i in range(n - 1)])
    if not peaks:
        return []
    # Row must cross an edge to participate; brightness is a separate, optional restriction.
    if max(abs(ref_profile[i + 1] - ref_profile[i]) for i in range(n - 1)) <= 1e-12:
        return []
    hi = max(ref_profile)
    out = []
    for i in range(n):
        near = [p for p in peaks if abs(i - p) <= half_width]
        if not near:
            continue
        if any(abs(i - p) <= guard for p in peaks):
            continue
        if bright <= 0.0 or ref_profile[i] > bright * hi:
            out.append(i)
    return out


def naive_oscillatory(est, ref, axis=None):
    """Nyquist amplitude by explicit Python loops over rows. `est`/`ref` are complex 2D arrays.

    `axis` must be given for any phantom whose gradients tie -- the benchmark's symmetric box does,
    and this implementation's tie-break historically chose axis 1 while the primary chose axis 0.
    That disagreement was invisible while the fixtures were all one-dimensional.
    """
    est, ref = np.asarray(est), np.asarray(ref)
    if axis is None:
        axis = 0 if np.abs(np.diff(np.abs(ref), axis=0)).mean() > \
            np.abs(np.diff(np.abs(ref), axis=1)).mean() else 1
    if axis == 0:
        est, ref = est.T, ref.T
    gsum = list(np.abs(np.diff(np.abs(ref), axis=-1)).sum(axis=0))
    peaks = _peak_indices(gsum)
    tot, rows = 0.0, 0
    for r in range(est.shape[0]):
        rp = list(np.abs(ref[r]))
        idx = _bright_sidelobe_indices(rp, peaks=peaks)
        if not idx:
            continue
        acc = 0j
        for i in idx:
            acc += (est[r][i] - ref[r][i]) * ((-1.0) ** i)
        tot += abs(acc / len(idx)) ** 2
        rows += 1
    return float(np.sqrt(tot / rows)) if rows else float("nan")


def naive_alignment(est, ref, control, axis=None):
    """`|<Rm, R0>| / ||R0||^2` by explicit loops. See `naive_oscillatory` on `axis`."""
    est, ref, control = (np.asarray(a) for a in (est, ref, control))
    if axis is None:
        axis = 0 if np.abs(np.diff(np.abs(ref), axis=0)).mean() > \
            np.abs(np.diff(np.abs(ref), axis=1)).mean() else 1
    if axis == 0:
        est, ref, control = est.T, ref.T, control.T
    peaks = _peak_indices(list(np.abs(np.diff(np.abs(ref), axis=-1)).sum(axis=0)))
    num, den = 0j, 0.0
    for r in range(est.shape[0]):
        idx = _bright_sidelobe_indices(list(np.abs(ref[r])), peaks=peaks)
        for i in idx:
            rm = est[r][i] - ref[r][i]
            r0 = control[r][i] - ref[r][i]
            num += rm * np.conj(r0)
            den += abs(r0) ** 2
    return float(abs(num) / den) if den > 0 else float("nan")


def naive_energy_ratio(est, ref, control, axis=None):
    """`||Rm|| / ||R0||` by explicit loops. See `naive_oscillatory` on `axis`."""
    est, ref, control = (np.asarray(a) for a in (est, ref, control))
    if axis is None:
        axis = 0 if np.abs(np.diff(np.abs(ref), axis=0)).mean() > \
            np.abs(np.diff(np.abs(ref), axis=1)).mean() else 1
    if axis == 0:
        est, ref, control = est.T, ref.T, control.T
    peaks = _peak_indices(list(np.abs(np.diff(np.abs(ref), axis=-1)).sum(axis=0)))
    num, den = 0.0, 0.0
    for r in range(est.shape[0]):
        for i in _bright_sidelobe_indices(list(np.abs(ref[r])), peaks=peaks):
            num += abs(est[r][i] - ref[r][i]) ** 2
            den += abs(control[r][i] - ref[r][i]) ** 2
    return float(np.sqrt(num) / np.sqrt(den)) if den > 0 else float("nan")


def naive_artifact_norm(ref, control, axis=None):
    """RMS `|R0|` over the sidelobe region, by explicit loops."""
    ref, control = np.asarray(ref), np.asarray(control)
    if axis is None:
        axis = 0 if np.abs(np.diff(np.abs(ref), axis=0)).mean() > \
            np.abs(np.diff(np.abs(ref), axis=1)).mean() else 1
    if axis == 0:
        ref, control = ref.T, control.T
    peaks = _peak_indices(list(np.abs(np.diff(np.abs(ref), axis=-1)).sum(axis=0)))
    tot, n = 0.0, 0
    for r in range(ref.shape[0]):
        for i in _bright_sidelobe_indices(list(np.abs(ref[r])), peaks=peaks):
            tot += abs(control[r][i] - ref[r][i]) ** 2
            n += 1
    return float(np.sqrt(tot / n)) if n else float("nan")


def compare_scorers(est, ref, control, rtol=0.02, atol=1e-6, axis=None):
    """Run both implementations; return (agree, detail).

    Pass `axis` explicitly for phantoms whose gradients tie -- otherwise the two implementations
    may silently score different physics and still "agree" on a one-dimensional fixture.
    """
    from score_unringing import residual_alignment, score
    s = score(np.abs(est), np.angle(est), np.abs(ref), np.angle(ref),
              control_mag=np.abs(control), control_phase=np.angle(control), axis=axis)
    key = {0: "residual_alignment_ro", 1: "residual_alignment_pe"}.get(axis, "residual_alignment")
    sfx = {0: "_ro", 1: "_pe"}.get(axis, "")
    pairs = [
        ("oscillatory_residual", s["oscillatory_residual"], naive_oscillatory(est, ref, axis)),
        (key, s[key], naive_alignment(est, ref, control, axis)),
    ]
    # The two quantities that now drive PF acceptance need independent implementations too --
    # exercising them only through synthetic acceptance rows tests the branching, not the maths.
    if sfx:
        pairs += [
            (f"residual_energy{sfx}", s[f"residual_energy{sfx}"],
             naive_energy_ratio(est, ref, control, axis)),
            (f"artifact_norm{sfx}", s[f"artifact_norm{sfx}"],
             naive_artifact_norm(ref, control, axis)),
        ]
    bad = []
    for name, a, b in pairs:
        if np.isnan(a) and np.isnan(b):
            continue
        if not np.isclose(a, b, rtol=rtol, atol=atol):
            bad.append(f"{name}: primary {a:.6g} vs naive {b:.6g}")
    return (not bad), "; ".join(bad)
