"""Behavioural comparison of simulated vs real data (spec 4.2).

The highest-value real-data check, because it tests the nominal-Nyquist cutoff using a tool that
assumes it: run `mrdegibbs` on both, and compare the spatial-frequency profile of what it removed.
If the simulator's ringing sat at the wrong frequency -- as the pre-phase-1 mechanism's did, at
period 3.88 voxels -- the removed-energy profiles would peak in different bins.

This informs plausibility only. Theoretical constants come from the simulator's own reconstruction
operator, never from reconstructed real data whose scanner recon, GRAPPA kernel, coil combination
and filtering are unknown.
"""
import numpy as np

__all__ = ["spectral_profile", "compare_profiles"]


def spectral_profile(diff, bins=16):
    """Radially-binned power spectrum of a 2D difference image, normalized to unit sum."""
    d = np.asarray(diff, float)
    d = d - d.mean()
    F = np.fft.fftshift(np.fft.fft2(d))
    P = np.abs(F) ** 2
    ny, nx = P.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot((yy - ny / 2) / (ny / 2), (xx - nx / 2) / (nx / 2))
    # Normalize the radius to the AXIS edge (|k| = 1), not the corner: an axis-aligned Nyquist
    # ripple is the signature we care about, and dividing by sqrt(2) would put it in bin 11 of 16
    # instead of the top bin. Corner frequencies clip into the top bin, which is correct.
    idx = np.clip((r * bins).astype(int), 0, bins - 1)
    prof = np.array([P[idx == k].sum() for k in range(bins)])
    s = prof.sum()
    return prof / s if s > 0 else prof


def compare_profiles(a, b, peak_tol=1, centroid_tol=0.12):
    """Do two removed-energy profiles describe ringing at the same spatial frequency?"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pa, pb = int(np.argmax(a)), int(np.argmax(b))
    k = np.arange(len(a))
    ca = float((a * k).sum() / max(a.sum(), 1e-30)) / len(a)
    cb = float((b * k).sum() / max(b.sum(), 1e-30)) / len(b)
    reasons = []
    if abs(pa - pb) > peak_tol:
        reasons.append(f"peak bin {pa} vs {pb} (tol {peak_tol})")
    if abs(ca - cb) > centroid_tol:
        reasons.append(f"normalized centroid {ca:.3f} vs {cb:.3f} (tol {centroid_tol})")
    return (not reasons), "; ".join(reasons)
