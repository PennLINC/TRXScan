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


def _nyquist_amplitude(diff, ref, half_width=8):
    """Amplitude of the voxel-alternating component of `diff`, in a window around `ref`'s edge.

    This is the projection onto (-1)**x, i.e. the discrete Nyquist frequency. Gibbs ringing lands
    almost entirely here; smooth errors (blur, scaling, bias) project to nearly zero.
    """
    dp = _profiles(diff)
    dr = np.abs(np.diff(_profiles(ref), axis=-1))
    if dr.size == 0 or dr.max() <= 0:
        return 0.0
    peak = int(np.argmax(dr.sum(axis=0)))
    lo, hi = max(0, peak - half_width), min(dp.shape[-1], peak + half_width + 1)
    w = dp[:, lo:hi]
    if w.shape[-1] == 0:
        return 0.0
    alt = (-1.0) ** np.arange(w.shape[-1])
    proj = (w * alt).mean(axis=-1)
    return float(np.sqrt((proj ** 2).mean()))


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


def score(est_mag, est_phase, ref_mag, ref_phase, mag_threshold=0.1):
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

    ze = est_mag * np.exp(1j * est_phase)
    zr = ref_mag * np.exp(1j * ref_phase)
    diff = ze - zr

    # --- oscillatory residual: the NYQUIST-alternating part of the error, near edges ---
    #
    # Ringing's defining signature is that it alternates sign every voxel (verified exactly for a
    # rectangular window, robust across sub-voxel edge position). So project the error onto the
    # alternating sequence rather than high-pass filtering it: a rolling-mean high-pass cannot
    # remove a sharp edge transition, and measured a Gaussian blur as MORE oscillatory than real
    # ringing -- exactly backwards.
    oscillatory = _nyquist_amplitude(np.real(diff), ref_mag)

    # --- resolution and geometry ---
    ref_sharp = _sharpness(ref_mag)
    sharpness = float(_sharpness(est_mag) / ref_sharp) if ref_sharp > 0 else np.nan
    bias = _edge_shift(est_mag, ref_mag)

    # --- magnitude-masked circular phase error ---
    thr = mag_threshold * (np.abs(zr).max() if np.abs(zr).max() > 0 else 1.0)
    keep = np.abs(zr) >= thr
    if keep.any():
        dphi = np.angle(ze[keep] * np.conj(zr[keep]))
        phase_rmse = float(np.sqrt((dphi ** 2).mean()))
    else:
        phase_rmse = np.nan

    return {
        "oscillatory_residual": oscillatory,
        "edge_location_bias": float(bias),
        "edge_sharpness": sharpness,
        "complex_rmse": float(np.sqrt((np.abs(diff) ** 2).mean())),
        "magnitude_rmse": float(np.sqrt(((est_mag - ref_mag) ** 2).mean())),
        "phase_rmse_masked": phase_rmse,
    }
