#!/usr/bin/env python3
"""Measure the SENSE g-factor of TRXScan's coil model against a physical loop array.

Reimplements `coil_sensitivity` (src/kspace.rs) in numpy so the encoding power of the shipped
model can be compared to a physically motivated one without running the simulator. Needs numpy
only.

    python3 tools/coil_gfactor.py
"""
import numpy as np

NX = NY = 140      # NIBS/HBCD acquisition matrix
VOX = 1.7          # mm isotropic

X, Y = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
XM, YM = (X - NX / 2) * VOX, (Y - NY / 2) * VOX     # mm from image centre


def shipped(n):
    """src/kspace.rs `coil_sensitivity`: ring at 0.6*max(nx,ny), Gaussian sigma 0.9*max(nx,ny)."""
    out = []
    for c in range(n):
        cx, cy = NX / 2, NY / 2
        r = 0.6 * max(NX, NY)
        t = 2 * np.pi * c / n
        px, py = cx + r * np.cos(t), cy + r * np.sin(t)
        d2 = (X - px) ** 2 + (Y - py) ** 2
        s = 0.9 * max(NX, NY)
        out.append((np.exp(-d2 / (2 * s * s)) + 0.15).astype(complex))
    return np.stack(out)


def loops(n, phase=True, radius_mm=110.0, loop_mm=55.0):
    """Circular elements on a cylinder: loop response a^2/(d^2+a^2)^{3/2}, optional B1- phase."""
    out = []
    for c in range(n):
        t = 2 * np.pi * c / n
        px, py = radius_mm * np.cos(t), radius_mm * np.sin(t)
        d = np.sqrt((XM - px) ** 2 + (YM - py) ** 2)
        s = (loop_mm ** 2 / (d ** 2 + loop_mm ** 2) ** 1.5).astype(complex)
        if phase:
            s = s * np.exp(1j * np.arctan2(YM - py, XM - px))
        out.append(s)
    return np.stack(out)


def gfactor(S, R, stride=7):
    """Mean and p99 SENSE g over voxels, Psi = I.  g = sqrt[(A^H A)^-1_rr (A^H A)_rr]."""
    g, step = [], NY // R
    for x in range(0, NX, stride):
        for y0 in range(0, step, stride):
            A = np.stack([S[:, x, (y0 + k * step) % NY] for k in range(R)], axis=1)
            AhA = A.conj().T @ A
            try:
                inv = np.linalg.inv(AhA)
            except np.linalg.LinAlgError:
                continue
            for r in range(R):
                v = np.sqrt(np.real(inv[r, r] * AhA[r, r]))
                if np.isfinite(v):
                    g.append(v)
    g = np.array(g)
    return g.mean(), np.percentile(g, 99)


def main():
    models = []
    for n in (8, 16, 32):
        models.append((f"shipped (Gaussian ring), n={n}", shipped(n)))
    for n in (8, 16, 32):
        models.append((f"loops, magnitude only, n={n}", loops(n, phase=False)))
    for n in (8, 16, 32):
        models.append((f"loops, magnitude + phase, n={n}", loops(n, phase=True)))

    print(f"{'model':38s} {'range':>8s} {'adj corr':>9s} "
          f"{'g(R=2) mean/p99':>17s} {'g(R=3) mean/p99':>17s}")
    for label, S in models:
        amp = np.abs(S)
        F = S.reshape(S.shape[0], -1)
        F = F - F.mean(1, keepdims=True)
        F = F / np.linalg.norm(F, axis=1, keepdims=True)
        adj = abs(np.vdot(F[0], F[1]))
        m2, p2 = gfactor(S, 2)
        m3, p3 = gfactor(S, 3)
        print(f"{label:38s} {amp.max() / amp.min():7.1f}x {adj:9.3f} "
              f"{m2:8.2f}/{p2:<8.2f} {m3:8.2f}/{p3:<8.2f}")


if __name__ == "__main__":
    main()
