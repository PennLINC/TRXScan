"""Error decomposition for Gibbs-unringing benchmarks (spec 3.5.5).

No unringing method can reach `object-nominal` exactly: the acquired data genuinely lacks the
truncated frequencies, so any un-ringed image is an inference, and an ideal Gibbs-remover returns a
band-limited estimate differing from the box-integrated object by the difference between a sinc and
a box PSF. Reporting one distance would therefore be misleading *and* would hide the failure mode
that matters most -- a method that suppresses ringing by blurring.

So this module returns a decomposition. A blurrer scores well on `oscillatory_residual` and badly
on `edge_sharpness`; a genuine unringer does well on both.
"""
import numpy as np

__all__ = ["score", "residual_alignment", "residual_energy_ratio", "artifact_norm",
           "dominant_axis", "READOUT_AXIS", "PHASE_ENCODE_AXIS"]


def dominant_axis(ref):
    """The axis along which `ref` actually varies most.

    Every metric here measures gradients and Nyquist projections along one axis. Assuming the last
    one is not safe: NIfTI arrays from the simulator carry x (the readout direction) as axis 0, and
    scoring along the wrong axis silently returns near-zero for everything because the object is
    constant there.
    """
    a = np.asarray(ref, float)
    best, ax = -1.0, a.ndim - 1
    for k in range(a.ndim):
        if a.shape[k] < 2:
            continue
        v = float(np.abs(np.diff(a, axis=k)).mean())
        if v > best + 1e-12:            # strict, so an exact tie keeps the FIRST axis
            best, ax = v, k
    return ax


# Axis convention for simulator output: 0 = readout (x), 1 = phase-encode (y).
# Partial Fourier undersamples ky, so anything PF-related must be scored on AXIS 1. Auto-selection
# cannot be trusted for that: the benchmark phantom is a symmetric box, its two gradient means tie
# exactly, and the tie-break silently decides which physics you measure.
READOUT_AXIS, PHASE_ENCODE_AXIS = 0, 1


def _to_last(a, axis):
    return np.moveaxis(np.asarray(a, float), axis, -1)


def _edge_mask(ref, frac=0.25):
    """Voxels near a structural edge of `ref`, where ringing lives."""
    gx = np.abs(np.gradient(ref, axis=-1))
    gy = np.abs(np.gradient(ref, axis=-2))
    g = np.hypot(gx, gy)
    thr = frac * g.max() if g.max() > 0 else np.inf
    m = g > thr
    # dilate by 3 voxels along the readout axis so the sidelobes are included, not just the step
    out = m.copy()
    for s in range(1, 4):
        out |= np.roll(m, s, axis=-1) | np.roll(m, -s, axis=-1)
    return out


def _nyquist_amplitude(diff, ref, half_width=10, guard=2, bright=0.0):
    """Amplitude of the voxel-alternating component of `diff`, beside `ref`'s edge.

    The projection onto (-1)**x, i.e. the discrete Nyquist frequency. Gibbs ringing lands almost
    entirely there; smooth errors (blur, scaling, bias) project to nearly zero.

    Two exclusions, both learned the hard way on real fixtures:

    - **Guard band around the edge.** The reference is box-integrated while the estimate is
      band-limited, so the transition voxels differ by the sinc-vs-box PSF regardless of ringing,
      and an unringer makes that transition SOFTER, enlarging the difference exactly there.
      Including the edge ranked both real methods worse than doing nothing.
    - **`bright` defaults to 0.0: BOTH sides of the edge are scored.** An earlier version masked to
      `ref > 0.5*max(ref)` because "these are magnitude images", where negative ringing is rectified
      and the alternation destroyed. That justification stopped being true once `score()` began
      reconstructing complex data: a negative lobe stored as positive magnitude with phase ~pi comes
      back as a negative complex value. Measured on a real fixture, the dark-side residual runs
      `++++++++` in magnitude but `+-+-+-+-` in complex, amplitudes 0.091 / 0.050 / 0.034 -- a full
      Gibbs sidelobe train. Masking it away discarded half the artifact and would have scored a
      method that fixes only the bright side as perfect.

      Set `bright > 0` to restrict to the bright side; `score()` reports that as a secondary
      `*_bright` variant for comparison with magnitude-only methods.
    """
    dp = _profiles(diff)
    dr = np.abs(np.diff(_profiles(ref), axis=-1))
    if dr.size == 0 or dr.max() <= 0:
        return 0.0
    # Sidelobes beside EVERY significant edge, with a guard band excluding the transition voxels
    # themselves: the reference is box-integrated while the estimate is band-limited, so those
    # differ by the sinc-vs-box PSF regardless of ringing, and an unringer makes the transition
    # SOFTER, enlarging the difference exactly there. Including the edge ranked both real methods
    # worse than doing nothing despite each cutting plateau ripple 76-88%.
    idx = _sidelobe_indices(dr.sum(axis=0), dp.shape[-1], half_width, guard)
    if idx.size == 0:
        return 0.0
    rp = _profiles(np.abs(ref))
    rmax = float(rp.max()) if rp.size else 0.0
    if rmax <= 0:
        return 0.0
    # A row participates if it CROSSES AN EDGE, not if it is bright: with bright = 0 the two
    # differ, and a `> 0.0` test silently drops exact zeros -- the dark side we now want.
    crosses = np.abs(np.diff(rp, axis=-1)).max(axis=-1) > 1e-12
    keep = (np.ones_like(rp[:, idx], bool) if bright <= 0.0 else rp[:, idx] > bright * rmax)
    keep = keep & crosses[:, None]
    w = dp[:, idx] * keep
    n = keep.sum(axis=-1)
    good = n > 0
    if not good.any():
        return 0.0
    alt = (-1.0) ** idx
    proj = (w[good] * alt).sum(axis=-1) / n[good]
    # |.|^2 so the measure is invariant to a global complex rotation of object and residual.
    return float(np.sqrt((np.abs(proj) ** 2).mean()))



def _sidelobe_indices(grad_profile, n, half_width, guard):
    """Sidelobe indices around EVERY significant edge, not just the strongest.

    A box has two edges per axis. Scoring only `argmax` was fine under full Fourier, where symmetry
    makes them equivalent, but a one-sided partial-Fourier transfer function is asymmetric and the
    two phase-encode edges need not have equivalent sidelobes -- so measuring one of them measures
    half the artifact.
    """
    g = np.asarray(grad_profile, float)
    if g.size == 0 or g.max() <= 0:
        return np.arange(0)
    peaks = []
    thr = 0.35 * g.max()
    for i in range(g.size):
        if g[i] < thr:
            continue
        lo = max(0, i - 2)
        hi = min(g.size, i + 3)
        if g[i] >= g[lo:hi].max():          # local maximum above threshold
            peaks.append(i)
    if not peaks:
        peaks = [int(np.argmax(g))]
    keep = np.zeros(n, bool)
    for p in peaks:
        lo, hi = max(0, p - half_width), min(n, p + half_width + 1)
        w = np.arange(lo, hi)
        keep[w[np.abs(w - p) > guard]] = True
    for p in peaks:                          # guard bands win over neighbouring windows
        keep[max(0, p - guard):min(n, p + guard + 1)] = False
    return np.nonzero(keep)[0]

def _profiles(a):
    """Rows of a 2D image, or of every slice of a 3D stack."""
    return a.reshape(-1, a.shape[-1])


def _sharpness(a):
    """Mean peak absolute gradient across profiles: the resolution proxy."""
    d = np.abs(np.diff(_profiles(a), axis=-1))
    return float(d.max(axis=-1).mean()) if d.size else 0.0


def _edge_shifts(est, ref, half_width=8):
    """Per-edge sub-voxel displacements, in profile order.

    Reported per edge because a one-sided partial-Fourier transfer function is asymmetric: a box's
    two edges on the same axis need not show the same apparent bias. Measured under contiguous 6/8
    they run [+0.25, +0.49, -0.49, -0.25].
    """
    de = np.abs(np.diff(_profiles(est), axis=-1))
    dr = np.abs(np.diff(_profiles(ref), axis=-1))
    if dr.size == 0 or dr.max() <= 0:
        return []
    x = np.arange(dr.shape[-1]) + 0.5
    summed = dr.sum(axis=0)
    thr = 0.35 * summed.max()
    peaks = [i for i in range(summed.size)
             if summed[i] >= thr
             and summed[i] >= summed[max(0, i - 2):min(summed.size, i + 3)].max()]
    if not peaks:
        peaks = [int(np.argmax(summed))]
    out = []
    for p in peaks:
        lo, hi = max(0, p - half_width), min(dr.shape[-1], p + half_width + 1)
        e, r, xx = de[:, lo:hi], dr[:, lo:hi], x[lo:hi]
        we, wr = e.sum(axis=-1), r.sum(axis=-1)
        good = (we > 1e-12) & (wr > 1e-12)
        if not good.any():
            out.append(float("nan"))
            continue
        ce = (e[good] * xx).sum(axis=-1) / we[good]
        cr = (r[good] * xx).sum(axis=-1) / wr[good]
        out.append(float((ce - cr).mean()))
    return out


def _edge_shift(est, ref, half_width=8):
    """Mean sub-voxel edge displacement over ALL edges. See `_edge_shifts` for the per-edge detail.

    Windowed around each REFERENCE edge, not taken over the whole profile: a whole-profile centroid
    is pulled arbitrarily far by any second structural edge -- including a periodic image's
    wrap-around -- which once reported a +2 voxel shift as -14.
    """
    vals = [v for v in _edge_shifts(est, ref, half_width) if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def residual_alignment(est, ref, control, axis=None, half_width=10, guard=2, bright=0.0):
    """How much of the CONTROL artifact survives in a method's residual.

    Frequency-agnostic, complex-valued, and therefore partial-Fourier-aware, which the Nyquist
    projection is not: PF ringing arises from two k-space intervals and does not sit at Nyquist,
    so a Nyquist projection simply stops seeing it as PF grows more aggressive.

    The simulator supplies the exact artifact, so its frequency need not be known a priori. With
    `R0 = control - ref` (the uncorrected complex residual) and `Rm = est - ref`:

        alignment = |<Rm, R0>| / ||R0||^2

    1.0 means the method removed nothing and >1 that it amplified the artifact. **0.0 does NOT mean
    the artifact is gone** -- only that the residual is ORTHOGONAL to the original template, which a
    method can achieve by shifting or reshaping the pattern while preserving its energy. That is why
    `residual_energy_ratio` exists; read the two together. All inputs are complex arrays.
    """
    est, ref, control = (np.asarray(a) for a in (est, ref, control))
    if axis is None:
        axis = dominant_axis(np.abs(ref))
    est, ref, control = (np.moveaxis(a, axis, -1) for a in (est, ref, control))
    r0 = _profiles(control - ref)
    rm = _profiles(est - ref)
    rp = _profiles(np.abs(ref))
    dr = np.abs(np.diff(rp, axis=-1))
    if dr.size == 0 or dr.max() <= 0:
        return float("nan")
    idx = _sidelobe_indices(dr.sum(axis=0), r0.shape[-1], half_width, guard)
    rmax = float(rp.max())
    if idx.size == 0 or rmax <= 0:
        return float("nan")
    crosses = np.abs(np.diff(rp, axis=-1)).max(axis=-1) > 1e-12
    keep = (np.ones_like(rp[:, idx], bool) if bright <= 0.0
            else rp[:, idx] > bright * rmax) & crosses[:, None]
    a, b = rm[:, idx] * keep, r0[:, idx] * keep
    denom = float(np.sum(np.abs(b) ** 2))
    if denom <= 0:
        return float("nan")
    return float(abs(np.sum(a * np.conj(b))) / denom)


def residual_energy_ratio(est, ref, control, axis=None, half_width=10, guard=2, bright=0.0):
    """`||Rm|| / ||R0||` over the same sidelobe region as [`residual_alignment`].

    Alignment alone is a PROJECTION: zero means the residual is orthogonal to the original artifact
    template, not that it is gone. A method could preserve substantial residual energy while
    shifting or reshaping the pattern and still score near zero. Reporting both makes that much
    harder to hit by accident -- energy says how much error remains at all, alignment says how much
    of the original artifact pattern remains.
    """
    est, ref, control = (np.asarray(a) for a in (est, ref, control))
    if axis is None:
        axis = dominant_axis(np.abs(ref))
    est, ref, control = (np.moveaxis(a, axis, -1) for a in (est, ref, control))
    rp = _profiles(np.abs(ref))
    dr = np.abs(np.diff(rp, axis=-1))
    if dr.size == 0 or dr.max() <= 0:
        return float("nan")
    idx = _sidelobe_indices(dr.sum(axis=0), rp.shape[-1], half_width, guard)
    rmax = float(rp.max())
    if idx.size == 0 or rmax <= 0:
        return float("nan")
    crosses = np.abs(np.diff(rp, axis=-1)).max(axis=-1) > 1e-12
    keep = (np.ones_like(rp[:, idx], bool) if bright <= 0.0
            else rp[:, idx] > bright * rmax) & crosses[:, None]
    rm = _profiles(est - ref)[:, idx] * keep
    r0 = _profiles(control - ref)[:, idx] * keep
    den = float(np.sqrt(np.sum(np.abs(r0) ** 2)))
    if den <= 0:
        return float("nan")
    return float(np.sqrt(np.sum(np.abs(rm) ** 2)) / den)


def artifact_norm(ref, control, axis, half_width=10, guard=2, bright=0.0):
    """`||R0||` -- the ABSOLUTE size of the uncorrected artifact in the sidelobe region.

    Needed as an acceptance gate. `residual_energy_ratio` cannot serve: for the control itself
    `Rm == R0`, so its ratio is identically 1 no matter how little artifact there is, and a gate
    built on it never skips anything. This is the quantity that actually distinguishes "apodized,
    almost nothing to remove" from "unapodized, plenty to remove".
    """
    ref, control = np.moveaxis(np.asarray(ref), axis, -1), np.moveaxis(np.asarray(control), axis, -1)
    rp = _profiles(np.abs(ref))
    dr = np.abs(np.diff(rp, axis=-1))
    if dr.size == 0 or dr.max() <= 0:
        return float("nan")
    idx = _sidelobe_indices(dr.sum(axis=0), rp.shape[-1], half_width, guard)
    rmax = float(rp.max())
    if idx.size == 0 or rmax <= 0:
        return float("nan")
    crosses = np.abs(np.diff(rp, axis=-1)).max(axis=-1) > 1e-12
    keep = (np.ones_like(rp[:, idx], bool) if bright <= 0.0
            else rp[:, idx] > bright * rmax) & crosses[:, None]
    r0 = _profiles(control - ref)[:, idx] * keep
    n = int(keep.sum())
    return float(np.sqrt(np.sum(np.abs(r0) ** 2) / n)) if n else float("nan")


def score(est_mag, est_phase, ref_mag, ref_phase, mag_threshold=0.1, axis=None,
          control_mag=None, control_phase=None):
    """Decompose the error of `est` against the artifact-free reference `ref`.

    All inputs are magnitude/phase pairs of identical shape. `mag_threshold` is a fraction of
    `max|ref|` below which phase is treated as meaningless.
    """
    est_mag = np.asarray(est_mag, float)
    ref_mag = np.asarray(ref_mag, float)
    est_phase = np.asarray(est_phase, float)
    ref_phase = np.asarray(ref_phase, float)
    if est_mag.shape != ref_mag.shape:
        raise ValueError(f"shape mismatch: {est_mag.shape} vs {ref_mag.shape}")
    # Keep the ORIGINAL orientation for the per-axis metrics below. Reusing the reoriented arrays
    # while leaving `control_*` untouched compared a transposed estimate against an untransposed
    # control, and the control's own alignment -- which is 1.0 by construction -- came out 0.24
    # whenever the auto-axis resolved to 0.
    est_mag_in, est_phase_in = est_mag, est_phase
    ref_mag_in, ref_phase_in = ref_mag, ref_phase

    # Orient so the varying axis is last, since every metric below works along axis -1.
    if axis is None:
        axis = dominant_axis(ref_mag)
    est_mag, ref_mag = _to_last(est_mag, axis), _to_last(ref_mag, axis)
    est_phase, ref_phase = _to_last(est_phase, axis), _to_last(ref_phase, axis)

    ze = est_mag * np.exp(1j * est_phase)
    zr = ref_mag * np.exp(1j * ref_phase)
    diff = ze - zr

    # Edge metrics need a gradient along the scoring axis. Determine that once, up front.
    ref_sharp = _sharpness(ref_mag)
    edge_valid = ref_sharp > 0

    # --- oscillatory residual: the NYQUIST-alternating part of the error, near edges ---
    #
    # Ringing's defining signature is that it alternates sign every voxel (verified exactly for a
    # rectangular window, robust across sub-voxel edge position). So project the error onto the
    # alternating sequence rather than high-pass filtering it: a rolling-mean high-pass cannot
    # remove a sharp edge transition, and measured a Gaussian blur as MORE oscillatory than real
    # ringing -- exactly backwards.
    # COMPLEX projection, not the real part. Taking np.real(diff) here made the primary ringing
    # score depend on the global phase: a residual rotated into the imaginary channel scored
    # better without any less complex Gibbs error. That is precisely the phase dependence this
    # simulator was rebuilt to represent correctly.
    oscillatory = _nyquist_amplitude(diff, ref_mag) if edge_valid else float("nan")

    # --- resolution and geometry ---
    # No gradient along the scoring axis means the edge metrics are undefined -- either the
    # reference genuinely has no edge (phase-only scoring on a uniform magnitude) or the wrong axis
    # was chosen. Report them as NaN and flag it, rather than returning a plausible-looking zero.
    sharpness = float(_sharpness(est_mag) / ref_sharp) if edge_valid else float("nan")
    bias = _edge_shift(est_mag, ref_mag) if edge_valid else float("nan")

    # --- magnitude-masked circular phase error ---
    thr = mag_threshold * (np.abs(zr).max() if np.abs(zr).max() > 0 else 1.0)
    keep = np.abs(zr) >= thr
    if keep.any():
        dphi = np.angle(ze[keep] * np.conj(zr[keep]))
        phase_rmse = float(np.sqrt((dphi ** 2).mean()))
    else:
        phase_rmse = np.nan

    # PF-aware, frequency-agnostic companions to the Nyquist projection. Reported PER AXIS:
    # readout and phase-encode are different physics under partial Fourier, and collapsing them
    # onto one auto-selected axis silently measured readout for a symmetric phantom.
    per_axis = {}
    if control_mag is not None and control_phase is not None:
        cm = np.asarray(control_mag, float)
        cp = np.asarray(control_phase, float)
        zc_raw = cm * np.exp(1j * cp)
        ze_raw = np.asarray(est_mag_in, float) * np.exp(1j * np.asarray(est_phase_in, float))
        zr_raw = np.asarray(ref_mag_in, float) * np.exp(1j * np.asarray(ref_phase_in, float))
        for name, ax in (("ro", READOUT_AXIS), ("pe", PHASE_ENCODE_AXIS)):
            if zr_raw.ndim <= ax:
                continue
            per_axis[f"residual_alignment_{name}"] = residual_alignment(
                ze_raw, zr_raw, zc_raw, axis=ax)
            per_axis[f"residual_energy_{name}"] = residual_energy_ratio(
                ze_raw, zr_raw, zc_raw, axis=ax)
            per_axis[f"artifact_norm_{name}"] = artifact_norm(zr_raw, zc_raw, axis=ax)
            # Secondary, bright-side only: comparable with magnitude-only methods, which cannot act
            # on the dark side at all. Never the primary score.
            per_axis[f"residual_alignment_{name}_bright"] = residual_alignment(
                ze_raw, zr_raw, zc_raw, axis=ax, bright=0.5)
            em = np.moveaxis(np.abs(np.asarray(est_mag_in, float)), ax, -1)
            rm_ = np.moveaxis(np.abs(np.asarray(ref_mag_in, float)), ax, -1)
            rs = _sharpness(rm_)
            per_axis[f"edge_sharpness_{name}"] = (
                float(_sharpness(em) / rs) if rs > 0 else float("nan"))
            per_axis[f"edge_location_bias_{name}"] = (
                _edge_shift(em, rm_) if rs > 0 else float("nan"))
            per_axis[f"edge_location_bias_{name}_per_edge"] = (
                _edge_shifts(em, rm_) if rs > 0 else [])
        # Unsuffixed key keeps the auto-selected axis, for 1D fixtures and general use. The
        # _ro / _pe pair is what the benchmark must consult, because auto-selection is undefined
        # on a symmetric phantom.
        alignment = residual_alignment(ze_raw, zr_raw, zc_raw, axis=axis)
    else:
        alignment = float("nan")

    return {
        "edge_metrics_valid": bool(edge_valid),
        "residual_alignment": alignment,
        **per_axis,
        "oscillatory_residual": oscillatory,
        "edge_location_bias": float(bias),
        "edge_sharpness": sharpness,
        "complex_rmse": float(np.sqrt((np.abs(diff) ** 2).mean())),
        "magnitude_rmse": float(np.sqrt(((est_mag - ref_mag) ** 2).mean())),
        "phase_rmse_masked": phase_rmse,
    }
