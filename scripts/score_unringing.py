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

__all__ = ["score"]


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
        if v > best:
            best, ax = v, k
    return ax


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


def _nyquist_amplitude(diff, ref, half_width=10, guard=2, bright=0.5):
    """Amplitude of the voxel-alternating component of `diff`, beside `ref`'s edge.

    The projection onto (-1)**x, i.e. the discrete Nyquist frequency. Gibbs ringing lands almost
    entirely there; smooth errors (blur, scaling, bias) project to nearly zero.

    Two exclusions, both learned the hard way on real fixtures:

    - **Guard band around the edge.** The reference is box-integrated while the estimate is
      band-limited, so the transition voxels differ by the sinc-vs-box PSF regardless of ringing,
      and an unringer makes that transition SOFTER, enlarging the difference exactly there.
      Including the edge ranked both real methods worse than doing nothing.
    - **Dark voxels.** These are MAGNITUDE images, so ringing that goes negative is rectified to
      positive and the sign alternation is destroyed: beside a bright edge the deviations run
      +0.092, -0.050, +0.034 (alternating) but on the dark side +0.092, +0.050, +0.034 (not).
      Restrict to `ref > bright * max(ref)`, where the alternation survives.
    """
    dp = _profiles(diff)
    dr = np.abs(np.diff(_profiles(ref), axis=-1))
    if dr.size == 0 or dr.max() <= 0:
        return 0.0
    peak = int(np.argmax(dr.sum(axis=0)))
    lo, hi = max(0, peak - half_width), min(dp.shape[-1], peak + half_width + 1)
    idx = np.arange(lo, hi)
    # EXCLUDE a guard band around the edge itself. The reference is box-integrated while the
    # estimate is band-limited, so the transition voxels differ by the sinc-vs-box PSF regardless
    # of ringing -- and an unringer makes that transition SOFTER, enlarging the difference there.
    # Including the edge made the metric rank both real methods worse than doing nothing, even
    # though both cut the plateau ripple by 76-88%. Ringing lives in the sidelobes; measure there.
    idx = idx[np.abs(idx - peak) > guard]
    if idx.size == 0:
        return 0.0
    rp = _profiles(np.abs(ref))
    rmax = float(rp.max()) if rp.size else 0.0
    if rmax <= 0:
        return 0.0
    keep = rp[:, idx] > bright * rmax
    w = dp[:, idx] * keep
    n = keep.sum(axis=-1)
    good = n > 0
    if not good.any():
        return 0.0
    alt = (-1.0) ** idx
    proj = (w[good] * alt).sum(axis=-1) / n[good]
    # |.|^2 so the measure is invariant to a global complex rotation of object and residual.
    return float(np.sqrt((np.abs(proj) ** 2).mean()))


def _profiles(a):
    """Rows of a 2D image, or of every slice of a 3D stack."""
    return a.reshape(-1, a.shape[-1])


def _sharpness(a):
    """Mean peak absolute gradient across profiles: the resolution proxy."""
    d = np.abs(np.diff(_profiles(a), axis=-1))
    return float(d.max(axis=-1).mean()) if d.size else 0.0


def _edge_shift(est, ref, half_width=8):
    """Sub-voxel edge displacement of `est` relative to `ref`.

    The centroid is taken in a window around the REFERENCE edge rather than over the whole
    profile. A whole-profile centroid is not robust: any second structural edge -- including the
    wrap-around edge of a periodic image -- pulls it arbitrarily far, which made a +2 voxel shift
    report as -14.
    """
    de = np.abs(np.diff(_profiles(est), axis=-1))
    dr = np.abs(np.diff(_profiles(ref), axis=-1))
    if dr.size == 0 or dr.max() <= 0:
        return np.nan
    x = np.arange(dr.shape[-1]) + 0.5
    # window from the reference's strongest gradient, shared by both so the comparison is fair
    peak = int(np.argmax(dr.sum(axis=0)))
    lo, hi = max(0, peak - half_width), min(dr.shape[-1], peak + half_width + 1)
    de, dr, x = de[:, lo:hi], dr[:, lo:hi], x[lo:hi]
    we, wr = de.sum(axis=-1), dr.sum(axis=-1)
    good = (we > 1e-12) & (wr > 1e-12)
    if not good.any():
        return np.nan
    ce = (de[good] * x).sum(axis=-1) / we[good]
    cr = (dr[good] * x).sum(axis=-1) / wr[good]
    return float((ce - cr).mean())


def score(est_mag, est_phase, ref_mag, ref_phase, mag_threshold=0.1, axis=None):
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

    return {
        "edge_metrics_valid": bool(edge_valid),
        "oscillatory_residual": oscillatory,
        "edge_location_bias": float(bias),
        "edge_sharpness": sharpness,
        "complex_rmse": float(np.sqrt((np.abs(diff) ** 2).mean())),
        "magnitude_rmse": float(np.sqrt(((est_mag - ref_mag) ** 2).mean())),
        "phase_rmse_masked": phase_rmse,
    }
