#!/usr/bin/env python
"""Independent B0 fieldmap (Hz) + R2*/T2* from a multi-echo GRE (magnitude + phase), and comparison
against another field on a common grid.

Subcommands
  fit          complex 5-echo fit in MEGRE native space:
                 (a) coarse field from the spatially 3D-unwrapped echo-1/echo-2 phase difference,
                 (b) temporal unwrapping of every echo with the coarse field and a magnitude^2-weighted
                     linear fit of phase vs TE (two passes), f = slope / 2pi;
                 (c) R2* from a magnitude^2-weighted log-linear fit of |S| vs TE.
  extrapolate  nearest-value extrapolation of a masked field + Gaussian smoothing outside the mask
               (whole-FOV output, so a later resampling has no edge to interpolate against).
  finish       taper a resampled field to 0 beyond N voxels outside a grid mask; optionally apply a
               sign and an offset (from `compare`); R2* -> T2* with clamping.
  regcheck     PNG: edges of a resampled moving image over the fixed image; optional field overlay.
  compare      field A (test) vs field B (reference) inside a mask: sign (Pearson r), offset
               (median), r / RMS / percentiles whole-brain and per region (orbitofrontal, temporal
               poles, rest), phase-encode Jacobian maps + pile-up/stretch fractions, PNG montages.

The `fit` sign is the raw one (d(phase)/dt / 2pi with the phase as stored); the physical sign is
decided in `compare` against the reference field and applied in `finish`.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi

TWO_PI = 2.0 * np.pi


# ----------------------------------------------------------------------------------------- helpers
def load(path):
    im = nib.load(str(path))
    return im, np.asanyarray(im.dataobj).astype(np.float32)


def save(path, data, like, dtype=np.float32):
    out = nib.Nifti1Image(np.asarray(data, dtype=dtype), like.affine)
    out.header.set_xyzt_units("mm")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(path))
    return str(path)


def wrap(x):
    return (x + np.pi) % TWO_PI - np.pi


def q(x, ps=(5, 50, 95)):
    x = np.asarray(x, dtype=np.float64)
    return [float(v) for v in np.percentile(x, ps)] if x.size else [float("nan")] * len(ps)


def pearson(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def dilate(mask, vox):
    return ndi.binary_dilation(mask, iterations=int(vox)) if vox > 0 else mask.copy()


def taper_weights(mask, plateau_vox, taper_vox):
    """1 inside `mask` dilated by plateau_vox, linear ramp to 0 over the next taper_vox, 0 beyond."""
    d = ndi.distance_transform_edt(~mask)               # voxel distance from the mask
    w = np.clip(1.0 - (d - plateau_vox) / max(taper_vox, 1e-6), 0.0, 1.0)
    return w.astype(np.float32)


def plt_init():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def sl(vol, axis, idx):
    """2D slice with anterior/superior up regardless of the array axis orientation (caller passes
    volumes already in a known axis order; we only transpose for display)."""
    s = np.take(vol, idx, axis=axis)
    return np.rot90(s)


# --------------------------------------------------------------------------------------------- fit
def unwrap3d(phase, mask):
    from skimage.restoration import unwrap_phase
    arr = np.ma.array(phase.astype(np.float64), mask=~mask)
    unw = np.asarray(unwrap_phase(arr, wrap_around=False)).astype(np.float32)
    # unwrap_phase returns values equal to the input modulo 2pi (up to a global multiple); pin
    # the global multiple so that the brain-median correction is zero (|dTE field| small there).
    k = np.round(np.median((unw[mask] - phase[mask]) / TWO_PI))
    unw = unw - TWO_PI * k
    unw[~mask] = phase[~mask]
    return unw


def cmd_fit(a):
    te = np.asarray(a.te, dtype=np.float64) * 1e-3                      # ms -> s
    n = len(te)
    assert len(a.mag) == n and len(a.phase) == n, "one magnitude and one phase per TE"
    like, m0 = load(a.mag[0])
    mag = np.stack([load(p)[1] for p in a.mag], -1)
    ph = np.stack([load(p)[1] for p in a.phase], -1) * a.phase_scale    # -> radians
    if a.mask:
        mask = load(a.mask)[1] > 0.5
    else:
        mask = m0 > a.mask_frac * np.percentile(m0, 99)
        mask = ndi.binary_fill_holes(mask)
    if a.mag_floor_frac > 0:                                            # drop signal voids
        mask &= m0 > a.mag_floor_frac * np.percentile(m0[mask], 99)
    print(f"[fit] TE(ms) {a.te}  voxels in mask {int(mask.sum())}")

    cx = mag * np.exp(1j * ph)
    # (a) coarse field: echo-2 x conj(echo-1), spatially unwrapped in 3D inside the mask
    dte = te[1] - te[0]
    d12 = np.angle(cx[..., 1] * np.conj(cx[..., 0])).astype(np.float32)
    d12u = unwrap3d(d12, mask)
    f_coarse = d12u / (TWO_PI * dte)
    n_wrapped = int((np.abs(d12u[mask] - d12[mask]) > np.pi).sum())
    print(f"[fit] coarse field: dTE {dte*1e3:.2f} ms (+-{0.5/dte:.1f} Hz per wrap); "
          f"{n_wrapped} in-mask voxels changed by spatial unwrapping; "
          f"coarse range in mask {f_coarse[mask].min():.0f}..{f_coarse[mask].max():.0f} Hz")

    # (b) temporal unwrapping + weighted linear fit, two passes
    w = mag.astype(np.float64) ** 2
    w[~mask] = 0
    f = f_coarse.astype(np.float64)
    for it in range(2):
        phi0 = wrap(ph[..., 0] - TWO_PI * f * te[0])
        pred = phi0[..., None] + TWO_PI * f[..., None] * te[None, None, None, :]
        phu = pred + wrap(ph - pred)                                    # temporally unwrapped
        sw = w.sum(-1) + 1e-12
        t_bar = (w * te).sum(-1) / sw
        p_bar = (w * phu).sum(-1) / sw
        cov = (w * (te - t_bar[..., None]) * (phu - p_bar[..., None])).sum(-1)
        var = (w * (te - t_bar[..., None]) ** 2).sum(-1) + 1e-30
        slope = cov / var
        f = slope / TWO_PI
        f[~mask] = 0
    icpt = p_bar - slope * t_bar
    resid = phu - (icpt[..., None] + slope[..., None] * te)
    rms = np.sqrt((w * resid ** 2).sum(-1) / sw)                         # weighted, rad
    rms[~mask] = 0
    # per-echo unweighted residual within mask, to expose odd/even (bipolar) offsets
    per_echo = [float(np.median(resid[..., i][mask])) for i in range(n)]
    per_echo_rms = [float(np.sqrt(np.mean(resid[..., i][mask] ** 2))) for i in range(n)]
    print(f"[fit] fit residual (weighted RMS, rad): median {np.median(rms[mask]):.4f} "
          f"p95 {np.percentile(rms[mask], 95):.4f};  per-echo median residual {np.round(per_echo, 4)}")
    print(f"[fit] field range in mask {f[mask].min():.0f}..{f[mask].max():.0f} Hz, "
          f"median {np.median(f[mask]):.1f}, |f - f_coarse| median {np.median(np.abs(f - f_coarse)[mask]):.2f} Hz")

    # (c) R2*: weighted log-linear
    lm = np.log(np.maximum(mag, 1e-3)).astype(np.float64)
    l_bar = (w * lm).sum(-1) / sw
    r2s = -((w * (te - t_bar[..., None]) * (lm - l_bar[..., None])).sum(-1) / var)
    lo, hi = 1.0 / (a.t2s_clamp[1] * 1e-3), 1.0 / (a.t2s_clamp[0] * 1e-3)
    r2s = np.clip(r2s, lo, hi); r2s[~mask] = 0
    t2s = np.zeros_like(r2s); t2s[mask] = 1e3 / r2s[mask]
    print(f"[fit] T2* (ms) in mask: median {np.median(t2s[mask]):.1f}  IQR {q(t2s[mask], (25, 75))}")

    P = a.out_prefix
    out = {
        "fieldmap_raw": save(P + "fieldmap_raw.nii.gz", f, like),
        "fieldmap_coarse": save(P + "fieldmap_coarse.nii.gz", f_coarse, like),
        "fit_rms_rad": save(P + "fit_rms.nii.gz", rms, like),
        "R2star": save(P + "R2star.nii.gz", r2s, like),
        "T2star": save(P + "T2star.nii.gz", t2s, like),
        "mask": save(P + "fitmask.nii.gz", mask, like, np.uint8),
    }
    summary = {
        "te_ms": list(map(float, a.te)), "phase_scale_rad_per_unit": a.phase_scale,
        "mask_voxels": int(mask.sum()), "coarse_dte_ms": float(dte * 1e3),
        "coarse_unaliased_hz": float(0.5 / dte), "coarse_spatial_unwrap_changed_voxels": n_wrapped,
        "field_raw_hz": {"min": float(f[mask].min()), "max": float(f[mask].max()),
                         "p5_50_95": q(f[mask])},
        "fit_rms_rad": {"median": float(np.median(rms[mask])), "p95": float(np.percentile(rms[mask], 95))},
        "per_echo_median_residual_rad": per_echo, "per_echo_rms_residual_rad": per_echo_rms,
        "coarse_vs_final_abs_diff_hz_median": float(np.median(np.abs(f - f_coarse)[mask])),
        "coarse_vs_final_abs_diff_hz_p99": float(np.percentile(np.abs(f - f_coarse)[mask], 99)),
        "t2star_ms": {"median": float(np.median(t2s[mask])), "p5_50_95": q(t2s[mask]),
                      "clamp": list(map(float, a.t2s_clamp))},
        "weights": "magnitude^2 per echo (phase and log-magnitude fits)", "outputs": out,
    }
    if a.fit_json:
        Path(a.fit_json).write_text(json.dumps(summary, indent=1))
    if a.qc_png:
        fit_qc_png(a.qc_png, m0, d12, d12u, f, rms, t2s, mask)
    print(json.dumps({k: v for k, v in summary.items() if k != "outputs"}, indent=1))


def fit_qc_png(png, m0, d12, d12u, f, rms, t2s, mask):
    plt = plt_init()
    nz = m0.shape[2]
    zs = [int(nz * 0.30), int(nz * 0.45), int(nz * 0.60)]
    rows = [("echo-1 |S|", m0, dict(cmap="gray", vmin=0, vmax=np.percentile(m0[mask], 99))),
            ("wrapped d(phase) e2-e1 (rad)", d12, dict(cmap="twilight", vmin=-np.pi, vmax=np.pi)),
            ("unwrapped d(phase) (rad)", d12u, dict(cmap="RdBu_r", vmin=-3 * np.pi, vmax=3 * np.pi)),
            ("field, raw sign (Hz)", f, dict(cmap="RdBu_r", vmin=-150, vmax=150)),
            ("fit RMS residual (rad)", rms, dict(cmap="magma", vmin=0, vmax=0.5)),
            ("T2* (ms)", t2s, dict(cmap="viridis", vmin=0, vmax=100))]
    fig, ax = plt.subplots(len(rows), len(zs), figsize=(3.2 * len(zs), 2.6 * len(rows)))
    for r, (name, vol, kw) in enumerate(rows):
        for c, z in enumerate(zs):
            im = ax[r, c].imshow(sl(np.where(mask | (r == 0), vol, np.nan), 2, z), **kw)
            ax[r, c].axis("off")
            if c == 0:
                ax[r, c].set_title(name, fontsize=9, loc="left")
        fig.colorbar(im, ax=ax[r, :].tolist(), fraction=0.02, pad=0.01)
    fig.suptitle("MEGRE fit QC (native RAS, axial slices)", fontsize=10)
    fig.savefig(png, dpi=110, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------------------------- extrapolate
def cmd_extrapolate(a):
    like, f = load(a.field)
    mask = load(a.mask)[1] > 0.5
    if a.erode:
        mask = ndi.binary_erosion(mask, iterations=a.erode)  # drop the noisiest rim before extrapolating
    _, idx = ndi.distance_transform_edt(~mask, return_indices=True)
    nn = f[tuple(idx)]                                       # nearest in-mask value everywhere
    sm = ndi.gaussian_filter(nn, a.sigma)
    out = np.where(mask, f, sm).astype(np.float32)
    save(a.out, out, like)
    print(f"[extrapolate] nearest-value extrapolation + Gaussian sigma {a.sigma} vox outside the mask "
          f"(erode {a.erode}); whole-FOV output {a.out}")


# ------------------------------------------------------------------------------------------ finish
def cmd_finish(a):
    like, f = load(a.field)
    mask = load(a.mask)[1] > 0.5
    w = taper_weights(mask, a.plateau, a.taper)
    sign, offset = a.sign, a.offset
    if a.correction_json:
        c = json.load(open(a.correction_json))
        sign, offset = c["sign"], c["offset_hz"]
    out = w * (sign * f - offset)
    save(a.out, out, like)
    print(f"[finish] sign {sign:+d}, offset {offset:.3f} Hz subtracted, weight = 1 within mask + "
          f"{a.plateau} vox, taper to 0 over {a.taper} vox -> {a.out}; nonzero voxels {(out != 0).sum()}")
    if a.r2s and a.out_t2s:
        _, r2 = load(a.r2s)
        cov = load(a.coverage)[1] if a.coverage else np.ones_like(r2)
        lo, hi = 1.0 / (a.t2s_clamp[1] * 1e-3), 1.0 / (a.t2s_clamp[0] * 1e-3)
        ok = (cov > 0.5) & (r2 > 0)
        t2 = np.zeros_like(r2); t2[ok] = 1e3 / np.clip(r2[ok], lo, hi)
        save(a.out_t2s, t2, like)
        print(f"[finish] T2* = 1/R2* where coverage > 0.5 -> {a.out_t2s}")


# ---------------------------------------------------------------------------------------- regcheck
def edges(img2d, thr_q=(70, 98)):
    g = np.hypot(*np.gradient(img2d.astype(np.float64)))
    lo, hi = np.percentile(g[np.isfinite(g)], thr_q)
    return np.ma.masked_where(g < lo, np.clip((g - lo) / (hi - lo + 1e-9), 0, 1))


def cmd_regcheck(a):
    plt = plt_init()
    fixed_im, fixed = load(a.fixed)
    mov = load(a.moving)[1]
    if mov.shape != fixed.shape:
        sys.exit(f"[regcheck] moving {mov.shape} not on the fixed grid {fixed.shape}")
    field = load(a.field)[1] if a.field else None
    mask = load(a.mask)[1] > 0.5 if a.mask else fixed > np.percentile(fixed, 60)
    # display in canonical RAS array order (same reorientation for every volume on the grid)
    orn = nib.io_orientation(fixed_im.affine)
    canon = lambda v: nib.apply_orientation(v, orn)
    fixed, mov, mask = canon(fixed), canon(mov), canon(mask)
    field = canon(field) if field is not None else None
    def view(vol, axis, idx):
        # RAS array: axis 0 sagittal (cols = A, rows = S), 1 coronal (cols = R, rows = S), 2 axial (cols = R, rows = A)
        s = np.take(vol, idx, axis=axis).T
        return s[::-1]                                   # up = S (sag/cor) or A (axial)
    idxs = [int(np.mean(np.nonzero(mask.any(axis=tuple(j for j in range(3) if j != i)))[0])) for i in range(3)]
    zoom = np.abs(np.asarray(nib.io_orientation(fixed_im.affine))[:, 0].astype(int))  # canonical axis -> array axis
    vox = np.asarray(fixed_im.header.get_zooms()[:3])[np.argsort(zoom)]                 # mm per canonical axis
    aspect = lambda axis: vox[[i for i in range(3) if i != axis][1]] / vox[[i for i in range(3) if i != axis][0]]
    nrow = 2 if field is not None else 1
    fig, ax = plt.subplots(nrow, 3, figsize=(13, 4.3 * nrow), squeeze=False)
    vmax = np.percentile(fixed[mask], 99.5)
    for c, axis in enumerate((0, 1, 2)):
        i = idxs[axis]
        ax[0, c].imshow(view(fixed, axis, i), cmap="gray", vmin=0, vmax=vmax, aspect=aspect(axis))
        ax[0, c].imshow(edges(view(mov, axis, i)), cmap="autumn", alpha=0.9, vmin=0, vmax=1, aspect=aspect(axis))
        ax[0, c].set_title(f"moving edges on fixed  ({'xyz'[axis]}={i})", fontsize=8)
        ax[0, c].axis("off")
        if field is not None:
            ax[1, c].imshow(view(fixed, axis, i), cmap="gray", vmin=0, vmax=vmax, aspect=aspect(axis))
            fv = np.where(mask, field, np.nan)
            im = ax[1, c].imshow(view(fv, axis, i), cmap="RdBu_r", vmin=-a.vlim, vmax=a.vlim, alpha=0.6, aspect=aspect(axis))
            ax[1, c].set_title(f"field (+-{a.vlim:g} Hz) in mask", fontsize=8); ax[1, c].axis("off")
    if field is not None:
        fig.colorbar(im, ax=ax[1, :].tolist(), fraction=0.02, pad=0.01, label="Hz")
    fig.suptitle((a.title or "registration check") + f"\nmoving {Path(a.moving).name}   fixed {Path(a.fixed).name}", fontsize=9)
    fig.savefig(a.png, dpi=110, bbox_inches="tight"); plt.close(fig)
    print("[regcheck] wrote", a.png)


# ----------------------------------------------------------------------------------------- compare
def regions_from_grid(im, mask, aseg):
    """Orbitofrontal/inferior-frontal, temporal poles, rest — on an LPS (or any) grid, using world
    coordinates so the definitions are axis-orientation independent. Returns dict name->bool array
    and a description dict."""
    aff = im.affine
    ijk = np.indices(mask.shape).reshape(3, -1).T
    xyz = (aff[:3, :3] @ ijk.T).T + aff[:3, 3]
    X = xyz.reshape(mask.shape + (3,))
    # NIfTI world coordinates are RAS whatever the array axis order: +x right, +y anterior, +z superior
    ant, sup, lat = X[..., 1], X[..., 2], -X[..., 0]           # lat: +left
    a_lo, a_hi = ant[mask].min(), ant[mask].max()
    s_lo, s_hi = sup[mask].min(), sup[mask].max()
    a_frac = (ant - a_lo) / (a_hi - a_lo)                       # 1 = most anterior
    s_frac = (sup - s_lo) / (s_hi - s_lo)                       # 1 = most superior
    x_mid = float(np.median(lat[mask]))
    desc = {"anterior_extent_mm": [float(a_lo), float(a_hi)], "superior_extent_mm": [float(s_lo), float(s_hi)],
            "midline_lat_mm": x_mid}
    temporal = np.zeros_like(mask)
    if aseg is not None:
        for lab, side in ((18, "left"), (54, "right")):
            sel = aseg == lab
            if sel.sum() == 0:
                continue
            ca, cs, cl = ant[sel].mean(), sup[sel].mean(), lat[sel].mean()
            box = (ant >= ca - 5) & (ant <= ca + 32) & (sup >= cs - 22) & (sup <= cs + 10)
            lat_ok = (lat - x_mid) * np.sign(cl - x_mid) >= abs(cl - x_mid) - 8
            temporal |= box & lat_ok
            desc[f"amygdala_{side}_centroid_ant_sup_lat_mm"] = [float(ca), float(cs), float(cl)]
    temporal &= mask
    orbitofrontal = mask & (a_frac > 2.0 / 3.0) & (s_frac < 0.5) & ~temporal
    rest = mask & ~orbitofrontal & ~temporal
    # tighter sub-regions of the OF box (not disjoint from it): the inferior part (lowest 30% of the
    # brain height, i.e. gyrus rectus / OF cortex above the sinuses) and the OF cortical surface
    # (within 2 voxels of the inferior/anterior brain-mask boundary)
    edge_d = ndi.distance_transform_edt(mask)
    of_inferior = orbitofrontal & (s_frac < 0.30)
    of_surface = orbitofrontal & (edge_d <= 2.0) & (s_frac < 0.40)
    desc["orbitofrontal_inferior"] = "OF box AND lowest 30% of the brain-mask superior extent"
    desc["orbitofrontal_surface"] = "OF box AND within 2 voxels of the brain-mask boundary AND lowest 40% of the height"
    desc["orbitofrontal"] = "anterior third of the brain-mask extent (world anterior axis) AND lower half " \
                            "of the superior extent, minus the temporal-pole boxes"
    desc["temporal_poles"] = "per hemisphere: box anterior of the aseg amygdala centroid (-5..+32 mm " \
                             "anterior, -22..+10 mm superior), at least (|amygdala lateral offset| - 8 mm) " \
                             "from the midline, within the brain mask"
    return {"whole_brain": mask, "orbitofrontal": orbitofrontal, "temporal_poles": temporal, "rest": rest,
            "orbitofrontal_inferior": of_inferior, "orbitofrontal_surface": of_surface}, desc, ant


def pe_jacobian(field, im, trt, sigma=0.0):
    """J = 1 + TRT * d(field)/d(anterior) in voxel units along the anterior direction, i.e. the
    Jacobian of the PA (blips toward anterior, +field -> anterior shift) displacement map. J < 1
    compresses (pile-up), J > 1 stretches. Returns J and the anterior-derivative (Hz/voxel)."""
    codes = nib.aff2axcodes(im.affine)
    f = ndi.gaussian_filter(field, sigma) if sigma > 0 else field
    dfdj = np.gradient(f.astype(np.float64), axis=1)            # per voxel along +j
    dfd_ant = -dfdj if codes[1] == "P" else dfdj
    return 1.0 + trt * dfd_ant, dfd_ant


def cmd_compare(a):
    plt = plt_init()
    im, A = load(a.test)                    # MEGRE (raw sign)
    _, B = load(a.ref)                      # DRBUDDI (reference sign + offset)
    mask = load(a.mask)[1] > 0.5
    aseg = load(a.aseg)[1].astype(int) if a.aseg else None
    if a.erode_mask:
        mask = ndi.binary_erosion(mask, iterations=a.erode_mask)
    R, rdesc, ant = regions_from_grid(im, mask, aseg)
    # (a) sign
    r_raw = pearson(A[mask], B[mask])
    sign = 1 if r_raw >= 0 else -1
    As = sign * A
    # (b) offset
    offset = float(np.median(As[mask] - B[mask]))
    Ac = As - offset
    D = Ac - B
    # (c) per-region stats
    def stats(sel):
        d = D[sel]
        return {"n": int(sel.sum()), "r": pearson(Ac[sel], B[sel]),
                "rms_hz": float(np.sqrt(np.mean(d ** 2))), "median_diff_hz": float(np.median(d)),
                "diff_p5_50_95_hz": q(d), "megre_p5_50_95_hz": q(Ac[sel]), "drbuddi_p5_50_95_hz": q(B[sel]),
                "slope_megre_on_drbuddi": float(np.polyfit(B[sel], Ac[sel], 1)[0]) if sel.sum() > 3 else None}
    metrics = {"sign_determination": {"pearson_r_raw_vs_ref": r_raw, "sign_applied_to_megre_raw": sign,
               "raw_convention_implied": ("stored Siemens phase INCREASES with +field (phase = +2*pi*f*TE) -- "
                                          "same sign as the reference" if sign > 0 else
                                          "stored Siemens phase DECREASES with +field (phase = -2*pi*f*TE) -- "
                                          "the raw field had to be negated to match the reference")},
               "offset": {"median_megre_minus_ref_hz_after_sign": offset,
                          "note": "subtracted from the MEGRE field so both are on the reference's centre frequency"},
               "regions": {k: stats(v) for k, v in R.items()}, "region_definitions": rdesc,
               "trt_s": a.trt}
    # (d) PE Jacobian
    jac = {}
    for sig in (0.0, a.jac_sigma):
        JA, gA = pe_jacobian(Ac, im, a.trt, sig)
        JB, gB = pe_jacobian(B, im, a.trt, sig)
        key = f"sigma_{sig:g}vox"
        jac[key] = {}
        for name, sel in R.items():
            jac[key][name] = {
                "megre": {"J_p5_50_95": q(JA[sel]), "frac_J_lt_0.7_pileup": float((JA[sel] < 0.7).mean()),
                          "frac_J_gt_1.3_stretch": float((JA[sel] > 1.3).mean()),
                          "frac_J_lt_1": float((JA[sel] < 1).mean()),
                          "dfield_dant_hz_per_vox_p5_50_95": q(gA[sel])},
                "drbuddi": {"J_p5_50_95": q(JB[sel]), "frac_J_lt_0.7_pileup": float((JB[sel] < 0.7).mean()),
                            "frac_J_gt_1.3_stretch": float((JB[sel] > 1.3).mean()),
                            "frac_J_lt_1": float((JB[sel] < 1).mean()),
                            "dfield_dant_hz_per_vox_p5_50_95": q(gB[sel])},
                "r_J": pearson(JA[sel], JB[sel]),
                "mean_J_megre": float(JA[sel].mean()), "mean_J_drbuddi": float(JB[sel].mean()),
                "frac_megre_J_lt_drbuddi_J": float((JA[sel] < JB[sel]).mean()),
                "r_dfield_dant": pearson(gA[sel], gB[sel]),
                "frac_megre_pileup_where_drbuddi_stretch":
                    float(((JA[sel] < 0.7) & (JB[sel] > 1.3)).sum() / max((JB[sel] > 1.3).sum(), 1)),
                "frac_megre_stretch_where_drbuddi_pileup":
                    float(((JA[sel] > 1.3) & (JB[sel] < 0.7)).sum() / max((JB[sel] < 0.7).sum(), 1)),
                "frac_sign_disagree_J_minus_1":
                    float((np.sign(JA[sel] - 1) != np.sign(JB[sel] - 1)).mean()),
            }
        if sig == a.jac_sigma:
            JA_s, JB_s = JA, JB
    metrics["pe_jacobian"] = {"definition": "J = 1 + TRT * d(field)/d(anterior), derivative per voxel along the "
                                            "world-anterior direction (= -j on this LPS grid); PA blips: +field "
                                            "-> anterior displacement; J<1 pile-up, J>1 stretch",
                              "jac_smoothing_sigma_vox": a.jac_sigma, "by_smoothing": jac}
    # write corrected field + Jacobians + regions
    P = a.out_prefix
    outs = {"megre_corrected_unmasked": save(P + "megre_signoffset.nii.gz", Ac, im),
            "diff": save(P + "diff_megre_minus_drbuddi.nii.gz", np.where(mask, D, 0), im),
            "J_megre": save(P + "J_megre.nii.gz", np.where(mask, JA_s, 0), im),
            "J_drbuddi": save(P + "J_drbuddi.nii.gz", np.where(mask, JB_s, 0), im),
            "regions": save(P + "regions.nii.gz",
                            R["orbitofrontal"] * 1 + R["temporal_poles"] * 2 + R["rest"] * 3, im, np.uint8),
            "correction_json": P + "correction.json"}
    Path(P + "correction.json").write_text(json.dumps({"sign": sign, "offset_hz": offset}, indent=1))
    metrics["outputs"] = outs
    Path(a.metrics).write_text(json.dumps(metrics, indent=1))
    # figures
    compare_pngs(a, im, Ac, B, D, JA_s, JB_s, R, mask, ant, plt)
    print(json.dumps({k: metrics[k] for k in ("sign_determination", "offset")}, indent=1))
    for k, v in metrics["regions"].items():
        print(f"[compare] {k:14s} n {v['n']:7d} r {v['r']:.3f} RMS {v['rms_hz']:.1f} Hz  diff p5/50/95 {np.round(v['diff_p5_50_95_hz'], 1)}")
    for k, v in jac[f"sigma_{a.jac_sigma:g}vox"].items():
        print(f"[jacobian s={a.jac_sigma}] {k:14s} MEGRE pile {v['megre']['frac_J_lt_0.7_pileup']:.3f} stretch {v['megre']['frac_J_gt_1.3_stretch']:.3f} | "
              f"DRBUDDI pile {v['drbuddi']['frac_J_lt_0.7_pileup']:.3f} stretch {v['drbuddi']['frac_J_gt_1.3_stretch']:.3f} | r_J {v['r_J']:.3f} | "
              f"meanJ {v['mean_J_megre']:.2f}/{v['mean_J_drbuddi']:.2f} | MEGRE pile where DRB stretch {v['frac_megre_pileup_where_drbuddi_stretch']:.2f}, "
              f"MEGRE stretch where DRB pile {v['frac_megre_stretch_where_drbuddi_pileup']:.2f}")


def compare_pngs(a, im, Ac, B, D, JA, JB, R, mask, ant, plt):
    # display everything in canonical RAS array order: axial = A up / subject L on image left,
    # sagittal = S up / A on the right
    orn = nib.io_orientation(im.affine)
    canon = lambda v: nib.apply_orientation(v, orn)
    Ac, B, D, JA, JB, mask = map(canon, (Ac, B, D, JA, JB, mask))
    R = {k: canon(v) for k, v in R.items()}
    nanm = lambda v: np.where(mask, v, np.nan)
    ax_view = lambda vol, z: vol[:, :, z].T[::-1]
    sag = lambda vol, x: vol[x, :, :].T[::-1]
    # 4 axial slices: two through the orbitofrontal region, two higher
    zof = np.nonzero(R["orbitofrontal"].any(axis=(0, 1)))[0]
    zall = np.nonzero(mask.any(axis=(0, 1)))[0]
    zs = sorted(set([int(np.percentile(zof, 20)), int(np.percentile(zof, 55)),
                     int(np.percentile(zall, 50)), int(np.percentile(zall, 75))]))
    vl = a.vlim
    fig, ax = plt.subplots(len(zs), 4, figsize=(15, 3.6 * len(zs)))
    for r, z in enumerate(zs):
        for c, (name, vol, kw) in enumerate((("DRBUDDI (Hz)", B, dict(cmap="RdBu_r", vmin=-vl, vmax=vl)),
                                             ("MEGRE, sign/offset matched (Hz)", Ac, dict(cmap="RdBu_r", vmin=-vl, vmax=vl)),
                                             ("MEGRE - DRBUDDI (Hz)", D, dict(cmap="RdBu_r", vmin=-vl / 2, vmax=vl / 2)),
                                             ("regions (1 OF, 2 temporal, 3 rest)", R["orbitofrontal"] * 1.0 + R["temporal_poles"] * 2 + R["rest"] * 3,
                                              dict(cmap="Set1", vmin=0.5, vmax=9.5)))):
            h = ax[r, c].imshow(ax_view(nanm(vol), z), **kw)
            ax[r, c].set_title(f"{name}  z={z}", fontsize=8); ax[r, c].axis("off")
            if r == 0 and c < 3:
                fig.colorbar(h, ax=ax[:, c].tolist(), fraction=0.02, pad=0.01)
    fig.suptitle("Field comparison on the scanner acq grid (axial, anterior up, subject left on image left)", fontsize=10)
    fig.savefig(a.report_dir + "/field_comparison_axial.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(len(zs), 3, figsize=(11.5, 3.6 * len(zs)))
    for r, z in enumerate(zs):
        for c, (name, vol) in enumerate((("J DRBUDDI (PA)", JB), ("J MEGRE (PA)", JA), ("J MEGRE - J DRBUDDI", JA - JB))):
            kw = dict(cmap="PuOr", vmin=0, vmax=2) if c < 2 else dict(cmap="RdBu_r", vmin=-1, vmax=1)
            h = ax[r, c].imshow(ax_view(nanm(vol), z), **kw)
            ax[r, c].set_title(f"{name}  z={z}", fontsize=8); ax[r, c].axis("off")
            if r == 0:
                fig.colorbar(h, ax=ax[:, c].tolist(), fraction=0.02, pad=0.01)
    fig.suptitle(f"PE Jacobian J = 1 + TRT d(field)/d(anterior)  (smoothing sigma {a.jac_sigma} vox; <1 pile-up, >1 stretch; anterior up)", fontsize=10)
    fig.savefig(a.report_dir + "/jacobian_axial.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # sagittal through the orbitofrontal region: field + Jacobian, both estimates
    xs = np.nonzero(R["orbitofrontal"].any(axis=(1, 2)))[0]
    xsel = [int(np.percentile(xs, 30)), int(np.percentile(xs, 50)), int(np.percentile(xs, 70))]
    fig, ax = plt.subplots(len(xsel), 4, figsize=(15, 3.4 * len(xsel)))
    for r, x in enumerate(xsel):
        for c, (name, vol, kw) in enumerate((("DRBUDDI (Hz)", B, dict(cmap="RdBu_r", vmin=-vl, vmax=vl)),
                                             ("MEGRE (Hz)", Ac, dict(cmap="RdBu_r", vmin=-vl, vmax=vl)),
                                             ("J DRBUDDI", JB, dict(cmap="PuOr", vmin=0, vmax=2)),
                                             ("J MEGRE", JA, dict(cmap="PuOr", vmin=0, vmax=2)))):
            h = ax[r, c].imshow(sag(nanm(vol), x), **kw)
            ax[r, c].set_title(f"{name}  x={x} (anterior right)", fontsize=8); ax[r, c].axis("off")
            if r == 0:
                fig.colorbar(h, ax=ax[:, c].tolist(), fraction=0.02, pad=0.01)
    fig.savefig(a.report_dir + "/field_jacobian_sagittal.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    ant = canon(ant)

    # scatter + anterior profile through the orbitofrontal region
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    sel = R["whole_brain"]
    idx = np.random.default_rng(0).choice(np.flatnonzero(sel), min(40000, int(sel.sum())), replace=False)
    ax[0].scatter(B.ravel()[idx], Ac.ravel()[idx], s=1, alpha=0.2)
    ax[0].plot([-vl, vl], [-vl, vl], "k--", lw=0.8); ax[0].set_xlabel("DRBUDDI (Hz)"); ax[0].set_ylabel("MEGRE (Hz)")
    ax[0].set_title("whole brain", fontsize=9)
    sel = R["orbitofrontal"]
    ax[1].scatter(B[sel], Ac[sel], s=1, alpha=0.3, c="C3")
    ax[1].plot([-vl, vl], [-vl, vl], "k--", lw=0.8); ax[1].set_xlabel("DRBUDDI (Hz)"); ax[1].set_ylabel("MEGRE (Hz)")
    ax[1].set_title("orbitofrontal / inferior frontal", fontsize=9)
    # mean field vs anterior coordinate inside the OF region, binned per mm
    bins = np.arange(np.floor(ant[sel].min()), np.ceil(ant[sel].max()) + 1.7, 1.7)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    for vol, lab, col in ((B, "DRBUDDI", "C0"), (Ac, "MEGRE", "C3")):
        m = [np.mean(vol[sel][(ant[sel] >= bins[i]) & (ant[sel] < bins[i + 1])]) if ((ant[sel] >= bins[i]) & (ant[sel] < bins[i + 1])).any() else np.nan for i in range(len(ctr))]
        ax[2].plot(ctr, m, "-o", ms=3, label=lab, color=col)
    ax[2].set_xlabel("anterior coordinate (mm)"); ax[2].set_ylabel("mean field in OF region (Hz)"); ax[2].legend()
    ax[2].set_title("OF field profile vs anterior position (slope -> PA Jacobian)", fontsize=9)
    fig.savefig(a.report_dir + "/scatter_profile.png", dpi=110, bbox_inches="tight"); plt.close(fig)


# -------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("fit", help="complex multi-echo field + R2* fit in native space")
    p.add_argument("--mag", nargs="+", required=True); p.add_argument("--phase", nargs="+", required=True)
    p.add_argument("--te", nargs="+", type=float, required=True, help="ms")
    p.add_argument("--phase-scale", type=float, default=np.pi / 4096, help="radians per stored unit")
    p.add_argument("--mask"); p.add_argument("--mask-frac", type=float, default=0.15, help="fallback: |S1| > frac*p99")
    p.add_argument("--mag-floor-frac", type=float, default=0.0, help="also drop voxels with |S1| below frac*p99 (signal voids)")
    p.add_argument("--t2s-clamp", nargs=2, type=float, default=[2.0, 200.0], help="ms")
    p.add_argument("--out-prefix", required=True); p.add_argument("--fit-json"); p.add_argument("--qc-png")
    p.set_defaults(fn=cmd_fit)

    p = sp.add_parser("extrapolate"); p.add_argument("--field", required=True); p.add_argument("--mask", required=True)
    p.add_argument("--sigma", type=float, default=3.0, help="Gaussian sigma (vox) applied outside the mask")
    p.add_argument("--erode", type=int, default=0); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_extrapolate)

    p = sp.add_parser("finish"); p.add_argument("--field", required=True); p.add_argument("--mask", required=True)
    p.add_argument("--plateau", type=float, default=5, help="voxels beyond the mask kept at full weight")
    p.add_argument("--taper", type=float, default=3, help="voxels over which the weight ramps to 0")
    p.add_argument("--sign", type=int, default=1); p.add_argument("--offset", type=float, default=0.0)
    p.add_argument("--correction-json", help="{sign, offset_hz} written by compare")
    p.add_argument("--out", required=True)
    p.add_argument("--r2s"); p.add_argument("--coverage"); p.add_argument("--out-t2s")
    p.add_argument("--t2s-clamp", nargs=2, type=float, default=[2.0, 200.0])
    p.set_defaults(fn=cmd_finish)

    p = sp.add_parser("regcheck"); p.add_argument("--fixed", required=True); p.add_argument("--moving", required=True)
    p.add_argument("--field"); p.add_argument("--mask"); p.add_argument("--vlim", type=float, default=150)
    p.add_argument("--title"); p.add_argument("--png", required=True)
    p.set_defaults(fn=cmd_regcheck)

    p = sp.add_parser("compare"); p.add_argument("--test", required=True, help="MEGRE field, raw sign, on the grid")
    p.add_argument("--ref", required=True, help="reference field (DRBUDDI) on the same grid")
    p.add_argument("--mask", required=True); p.add_argument("--aseg"); p.add_argument("--erode-mask", type=int, default=0)
    p.add_argument("--trt", type=float, required=True, help="TotalReadoutTime (s) of the DWI")
    p.add_argument("--jac-sigma", type=float, default=1.0, help="Gaussian smoothing (vox) before the Jacobian derivative")
    p.add_argument("--vlim", type=float, default=150)
    p.add_argument("--out-prefix", required=True); p.add_argument("--metrics", required=True)
    p.add_argument("--report-dir", required=True)
    p.set_defaults(fn=cmd_compare)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
