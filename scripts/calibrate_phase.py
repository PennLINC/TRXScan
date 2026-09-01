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

__all__ = [
    "siemens_phase_to_radians", "phase_spatial_stats", "fit_b_dependence",
    "circular_sd", "WRAP_CEILING",
]

# Linear SD of a uniformly-wrapped phase. Any measurement approaching this is saturated and says
# nothing about the underlying spread.
WRAP_CEILING = np.pi / np.sqrt(3.0)

SIEMENS_HALF_RANGE = 4096.0


def siemens_phase_to_radians(raw):
    """Convert Siemens integer phase to radians, accepting either stored representation.

    **Verified against NIBS sub-60515** (`_part-phase_dwi.nii.gz`):

    - header `datatype` 512 (**uint16**), `bitpix` 16, `scl_slope` and `scl_inter` both **NaN**
    - `dataobj.get_unscaled()` -> uint16 in **[0, 4095]**, one full turn
    - `np.asanyarray(dataobj)` -> float in **[-4096, 4094]**; nibabel applies slope 2, inter -4096

    Both map one turn onto 2*pi, but by different formulas, and feeding the wrong one produces
    silently wrong numbers rather than an error:

        signed   v in [-4096, 4095]:  rad = v * pi / 4096
        unsigned u in [0, 4095]:      rad = u * 2*pi / 4096 - pi

    The representation is detected from the sign of the minimum. An earlier version assumed the
    signed form unconditionally. That happens to be what `np.asanyarray` returns, so the published
    calibration constants are unaffected -- but a caller reading with `get_unscaled()` would have
    got a half-scaled, pi-offset field with no warning.
    """
    raw = np.asarray(raw, float)
    lo, hi = float(np.nanmin(raw)), float(np.nanmax(raw))
    if max(abs(lo), abs(hi)) <= 2 * np.pi + 1e-6:
        raise ValueError(
            f"data already looks like radians (max |value| = {max(abs(lo), abs(hi)):.3f}); "
            "refusing to rescale twice"
        )
    if lo < 0:
        return raw * np.pi / SIEMENS_HALF_RANGE                      # signed store
    # 4096 counts span 2*pi here, so the per-count rate is 2*pi/4096 -- twice the signed rate,
    # where 8192 counts span the same turn. Using the signed rate gives a half-scaled field.
    return raw * (2.0 * np.pi) / SIEMENS_HALF_RANGE - np.pi          # unsigned store


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


def circular_sd(samples):
    """Circular SD, sqrt(-2 ln R), where R is the mean resultant length.

    Linear `np.std` is wrong for wrapped phase: it saturates at `WRAP_CEILING` = pi/sqrt(3) once
    the phase is uniformly distributed, so shells whose true spread exceeds ~pi all report the same
    number and the fitted exponent is biased DOWNWARD. Measured on NIBS, b=3000 sat at 99.4% of the
    ceiling, and the linear fit gave p = 0.257 against a circular fit of ~0.43.
    """
    z = np.exp(1j * np.asarray(samples, float))
    R = float(np.abs(z.mean()))
    if R <= 0:
        return np.inf
    return float(np.sqrt(-2.0 * np.log(R)))


def saturation_fraction(samples):
    """How close a shell's LINEAR SD sits to the wrapped-phase ceiling. >0.9 means unusable."""
    return float(np.std(np.asarray(samples, float)) / WRAP_CEILING)


def fit_b_dependence(shots, bvals, snr, statistic="circular"):
    """Fit `sigma_phi(b) = a * b**p`, correcting for thermal phase noise.

    `shots[i]` are per-shot phase samples at `bvals[i]`; `snr[i]` is the magnitude SNR there. The
    observed SD is modelled as `sqrt((a*b**p)^2 + (1/snr)^2)`, so the thermal floor is fitted
    rather than absorbed into `p`. `p_naive` is the uncorrected fit, reported so the size of the
    bias stays visible.
    """
    bvals = np.asarray(bvals, float)
    snr = np.asarray(snr, float)
    if statistic == "circular":
        obs = np.array([circular_sd(s) for s in shots])
    elif statistic == "linear":
        obs = np.array([float(np.std(s)) for s in shots])
    else:
        raise ValueError(f"unknown statistic {statistic!r}")

    lg = np.polyfit(np.log(bvals), np.log(obs), 1)
    p_naive, a_naive = float(lg[0]), float(np.exp(lg[1]))

    # np.interp requires ascending x; sort the observations rather than assuming the caller did.
    order = np.argsort(bvals)
    b_sorted, snr_sorted = bvals[order], snr[order]

    def model(b, a, p):
        return np.sqrt((a * b ** p) ** 2 + (1.0 / np.interp(b, b_sorted, snr_sorted)) ** 2)

    popt, _ = curve_fit(model, bvals, obs, p0=[a_naive, p_naive],
                        bounds=([0.0, 0.0], [np.inf, 3.0]), maxfev=20000)
    return {
        "c_q": float(popt[0]),
        "p": float(popt[1]),
        "a_naive": a_naive,
        "p_naive": p_naive,
        "snr_corrected": True,
        "statistic": statistic,
        "saturation": [saturation_fraction(s) for s in shots],
    }
