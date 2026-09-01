"""Fit and tune the object-phase model against real complex DWI (spec 4.2).

Two standards, deliberately different:

- **Term 2, pre-readout background phase: TUNED.** Reconstructed phase is an inseparable mixture of
  magnetization phase, coil phase and combination, scanner phase conventions and possibly
  reconstruction filtering. Only some of that precedes Fourier encoding, so these statistics are
  effective benchmark targets, not a recovered physical distribution.
- **Term 3, diffusion phase: FITTED**, but only with the thermal contribution modelled. Magnitude
  SNR falls with b, so measured phase variance rises even if motion-induced phase does not; a naive
  fit absorbs that and biases `p` upward.
"""
import numpy as np
from scipy.optimize import curve_fit

__all__ = ["siemens_phase_to_radians", "phase_spatial_stats", "fit_b_dependence"]

SIEMENS_HALF_RANGE = 4096.0


def siemens_phase_to_radians(raw):
    """Convert Siemens integer phase (-4096..4095) to radians.

    NIBS stores phase as uint16 with `scl_slope` NaN and `Units: arbitrary`, so nibabel returns raw
    integers. Skipping this rescale would inflate every phase statistic by ~1300x.
    """
    raw = np.asarray(raw, float)
    span = np.nanmax(np.abs(raw))
    if span <= 2 * np.pi + 1e-6:
        raise ValueError(
            f"data already looks like radians (max |value| = {span:.3f}); refusing to rescale twice"
        )
    return raw * np.pi / SIEMENS_HALF_RANGE


def _wrapped_diff(a, axis):
    d = np.diff(a, axis=axis)
    return np.angle(np.exp(1j * d))


def phase_spatial_stats(phase, mag, mask):
    """Spatial statistics of a phase volume stack, in radians.

    Returns `grad_rms` (radians per voxel), `corr_length_vox`, and `wrap_density` (fraction of
    neighbouring pairs whose raw difference exceeds pi).
    """
    phase = np.asarray(phase, float)
    mask = np.asarray(mask, bool)

    gx = _wrapped_diff(phase, -1)
    gy = _wrapped_diff(phase, -2)
    mx = mask[..., :, 1:] & mask[..., :, :-1]
    my = mask[..., 1:, :] & mask[..., :-1, :]
    # RMS of the gradient MAGNITUDE, not of the pooled per-axis components: pooling makes a ramp
    # along one axis read as g/sqrt(2), since the orthogonal component is identically zero.
    gxc, gyc = gx[..., :-1, :], gy[..., :, :-1]
    mc = mx[..., :-1, :] & my[..., :, :-1]
    grad_rms = float(np.sqrt((gxc[mc] ** 2 + gyc[mc] ** 2).mean())) if mc.any() else 0.0

    raw_x = np.diff(phase, axis=-1)
    wrap_density = float((np.abs(raw_x[mx]) > np.pi).mean()) if mx.any() else 0.0

    # Correlation length: the lag at which the RMS wrapped phase difference first exceeds 1 rad.
    #
    # NOT the autocorrelation of exp(i*phase): a deterministic ramp has |autocorr| identically 1 at
    # every lag, so that measure pinned at its cap and could not distinguish smooth from rough.
    max_lag = min(64, phase.shape[-1] - 1)
    corr_length = float(max_lag)
    for k in range(1, max_lag + 1):
        d = np.angle(np.exp(1j * (phase[..., k:] - phase[..., :-k])))
        mk = mask[..., k:] & mask[..., :-k]
        if not mk.any():
            continue
        if np.sqrt((d[mk] ** 2).mean()) > 1.0:
            corr_length = float(k)
            break

    return {"grad_rms": grad_rms, "corr_length_vox": corr_length, "wrap_density": wrap_density}


def fit_b_dependence(shots, bvals, snr):
    """Fit `sigma_phi(b) = a * b**p`, correcting for thermal phase noise.

    `shots[i]` are per-shot phase samples at `bvals[i]`; `snr[i]` is the magnitude SNR there. The
    observed SD is modelled as `sqrt((a*b**p)^2 + (1/snr)^2)`, so the thermal floor is fitted
    rather than absorbed into `p`. `p_naive` is the uncorrected fit, reported so the size of the
    bias stays visible.
    """
    bvals = np.asarray(bvals, float)
    snr = np.asarray(snr, float)
    obs = np.array([float(np.std(s)) for s in shots])

    lg = np.polyfit(np.log(bvals), np.log(obs), 1)
    p_naive, a_naive = float(lg[0]), float(np.exp(lg[1]))

    def model(b, a, p):
        return np.sqrt((a * b ** p) ** 2 + (1.0 / np.interp(b, bvals, snr)) ** 2)

    popt, _ = curve_fit(model, bvals, obs, p0=[a_naive, p_naive],
                        bounds=([0.0, 0.0], [np.inf, 3.0]), maxfev=20000)
    return {
        "c_q": float(popt[0]),
        "p": float(popt[1]),
        "a_naive": a_naive,
        "p_naive": p_naive,
        "snr_corrected": True,
    }
