#!/usr/bin/env python3
"""Voxelwise T2/S0 from a multi-echo spin-echo EPI series, and a predicted diffusion b=0.

Subcommands (driven by scripts/mese_pipeline.sh, but each is usable on its own):

  fit          monoexponential S(TE) = S0 exp(-TE/T2) per voxel (NLLS: vectorised Gauss-Newton on
               (S0, R2), initialised from a signal-weighted log-linear fit). Echoes acquired at a
               longer TR are first rescaled to the reference TR with a per-voxel T1-saturation
               factor from a tissue label image (nominal T1). Writes S0 (bias-corrected and raw),
               T2 (ms, clamped), R2 (1/ms) and the fit mask in the echoes' own space.
  finish-maps  turn resampled S0 / R2 (+ coverage) images into masked S0 and T2 maps; optional
               overlay PNG of the T2 map on a reference (WM probseg) for registration checks.
  predict      b0 at the diffusion TE/TR from the S0/T2 maps (map prediction) and from tissue-
               constant values (baseline), compared voxelwise against the real mean b0; writes
               metrics.json and the figures.
  sdc-qc       before/after montage for the topup correction.

Nominal T1 (s): WM 0.83, GM 1.33, CSF 4.3 (3T literature values, used for TR corrections only).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

T1_NOMINAL = {"WM": 0.83, "GM": 1.33, "CSF": 4.3}
T2_BASELINE = {"WM": 70.0, "GM": 100.0, "CSF": 2000.0}
T2_CLAMP = (5.0, 3000.0)
TISSUES = ("WM", "GM", "CSF")


def sat(tr, t1):
    """Spin-echo T1 saturation factor 1 - exp(-TR/T1) (TR and T1 in the same unit)."""
    return 1.0 - np.exp(-tr / t1)


def load_f32(path):
    im = nib.load(str(path))
    return im, np.asanyarray(im.dataobj).astype(np.float32)


def save(path, data, like, dtype=np.float32):
    out = nib.Nifti1Image(np.asarray(data, dtype=dtype), like.affine)
    out.header.set_xyzt_units("mm")
    nib.save(out, str(path))


def quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    q = np.percentile(x, [25, 50, 75])
    return {"n": int(x.size), "median": float(q[1]), "iqr": [float(q[0]), float(q[2])],
            "mean": float(x.mean())}


# ----------------------------------------------------------------------------- fit

def t2_fit(echoes, te, n_iter=25):
    """NLLS monoexponential fit. echoes: (n_te, nvox) float64, te: (n_te,) ms.

    Returns S0, R2 (1/ms), rmse. Gauss-Newton from a signal-weighted log-linear start; R2 is kept
    inside [1/T2max, 1/T2min] during iteration and S0 non-negative.
    """
    S = np.clip(echoes, 1e-3, None)
    w = S  # weight ~ signal (first-order noise weighting of ln S)
    te = te[:, None]
    lnS = np.log(S)
    W = w.sum(0); Wt = (w * te).sum(0); Wtt = (w * te * te).sum(0)
    Wy = (w * lnS).sum(0); Wty = (w * te * lnS).sum(0)
    det = W * Wtt - Wt * Wt
    slope = (W * Wty - Wt * Wy) / det
    icpt = (Wy - slope * Wt) / W
    r2 = np.clip(-slope, 1.0 / T2_CLAMP[1], 1.0 / T2_CLAMP[0])
    s0 = np.clip(np.exp(icpt), 1e-3, None)
    for _ in range(n_iter):
        e = np.exp(-te * r2)
        m = s0 * e
        r = echoes - m
        j0 = e; j1 = -s0 * te * e
        a = (j0 * j0).sum(0); b = (j0 * j1).sum(0); d = (j1 * j1).sum(0)
        g0 = (j0 * r).sum(0); g1 = (j1 * r).sum(0)
        lam = 1e-3 * (a + d)  # light Levenberg damping for the flat (CSF) direction
        a = a + lam; d = d + lam
        det = a * d - b * b
        ds0 = (d * g0 - b * g1) / det
        dr2 = (a * g1 - b * g0) / det
        s0 = np.clip(s0 + ds0, 1e-3, None)
        r2 = np.clip(r2 + dr2, 1.0 / T2_CLAMP[1], 1.0 / T2_CLAMP[0])
    m = s0 * np.exp(-te * r2)
    rmse = np.sqrt(((echoes - m) ** 2).mean(0))
    return s0, r2, rmse


def cmd_fit(a):
    ims = [load_f32(p) for p in a.echoes]
    like = ims[0][0]
    data = np.stack([x for _, x in ims], 0)  # (n_te, X, Y, Z)
    te = np.asarray(a.te, float); tr = np.asarray(a.tr, float)
    tr_ref = tr[0] if a.tr_ref is None else a.tr_ref
    e1 = data[0]

    # fit mask: echo-1 intensity (fraction of its 99th percentile inside the brain mask, if any),
    # all echoes positive, optionally restricted to a dilated brain mask.
    bm = None
    if a.brain_mask:
        bm = load_f32(a.brain_mask)[1] > 0.5
        from scipy.ndimage import binary_dilation
        bm_d = binary_dilation(bm, iterations=a.brain_mask_dilate)
        p99 = np.percentile(e1[bm], 99)
    else:
        bm_d = np.ones(e1.shape, bool)
        p99 = np.percentile(e1[e1 > 0], 99)
    fitmask = (e1 > a.mask_frac * p99) & (data > 0).all(0) & bm_d
    print(f"[fit] echo-1 p99 {p99:.1f}, threshold {a.mask_frac * p99:.1f}, "
          f"fit voxels {int(fitmask.sum())}")

    # per-voxel T1 from the label image -> TR correction factors for echoes not at tr_ref
    t1 = np.full(e1.shape, T1_NOMINAL["GM"], np.float32)  # unlabeled: GM (middle value)
    if a.dseg:
        dseg = np.rint(load_f32(a.dseg)[1]).astype(int)
        for t, lab in zip(TISSUES, a.dseg_labels):
            t1[dseg == lab] = T1_NOMINAL[t]
    factors = []
    for k in range(len(te)):
        f = sat(tr_ref, t1) / sat(tr[k], t1) if tr[k] != tr_ref else np.ones_like(t1)
        factors.append(f)
        if tr[k] != tr_ref:
            print(f"[fit] echo {k + 1}: TR {tr[k]} -> {tr_ref} s, factor range "
                  f"{f[fitmask].min():.4f}..{f[fitmask].max():.4f}")
    factors = np.stack(factors, 0)
    corrected = data * factors

    idx = np.where(fitmask)
    Y = corrected[:, idx[0], idx[1], idx[2]].astype(np.float64)
    s0, r2, rmse = t2_fit(Y, te)
    t2 = 1.0 / r2

    S0 = np.zeros(e1.shape, np.float32); T2 = np.zeros_like(S0); R2 = np.zeros_like(S0)
    RMSE = np.zeros_like(S0)
    S0[idx] = s0; T2[idx] = t2; R2[idx] = r2; RMSE[idx] = rmse / np.maximum(s0, 1e-3)
    S0raw = S0.copy()
    if a.bias:
        bias = load_f32(a.bias)[1]
        ref = bm if bm is not None else fitmask
        bias = bias / np.median(bias[ref & (bias > 0)])
        S0 = np.where(bias > 0, S0 / np.maximum(bias, 1e-3), S0).astype(np.float32)
        print(f"[fit] bias field (normalised to brain median 1): "
              f"range in fit mask {bias[fitmask].min():.3f}..{bias[fitmask].max():.3f}")

    pre = a.out_prefix
    save(f"{pre}S0_mese.nii.gz", S0, like); save(f"{pre}S0raw_mese.nii.gz", S0raw, like)
    save(f"{pre}T2_mese.nii.gz", T2, like); save(f"{pre}R2_mese.nii.gz", R2, like)
    save(f"{pre}fitmask_mese.nii.gz", fitmask.astype(np.float32), like)
    save(f"{pre}fitrmse_mese.nii.gz", RMSE, like)

    # summary + sensitivity to the echo-4 TR correction (single WM factor / none)
    out = {"model": "S0*exp(-TE/T2), NLLS Gauss-Newton (25 it) from signal-weighted log-linear",
           "te_ms": a.te, "tr_s": a.tr, "tr_ref_s": float(tr_ref), "t1_nominal_s": T1_NOMINAL,
           "t2_clamp_ms": T2_CLAMP, "mask_frac_of_p99": a.mask_frac, "fit_voxels": int(fitmask.sum()),
           "bias_corrected_S0": bool(a.bias), "tissue": {}, "sensitivity_tr_correction": {}}
    if a.dseg:
        for t, lab in zip(TISSUES, a.dseg_labels):
            sel = fitmask & (dseg == lab)
            out["tissue"][t] = {"T2_ms": quantiles(T2[sel]), "S0": quantiles(S0[sel]),
                                "S0raw": quantiles(S0raw[sel]),
                                "norm_rmse": quantiles(RMSE[sel])}
        wm, gm, csf = (out["tissue"][t]["S0"]["median"] for t in TISSUES)
        out["S0_ratio_GM_WM"] = gm / wm; out["S0_ratio_CSF_WM"] = csf / wm
        # sensitivity: refit with (i) no TR correction, (ii) a single WM factor everywhere
        for name, fac in [("none", np.ones_like(t1)),
                          ("single_WM_factor", np.full_like(t1, sat(tr_ref, T1_NOMINAL["WM"])
                                                            / sat(tr[-1], T1_NOMINAL["WM"])))]:
            F = factors.copy(); F[-1] = fac
            Y2 = (data * F)[:, idx[0], idx[1], idx[2]].astype(np.float64)
            _, r2b, _ = t2_fit(Y2, te)
            T2b = np.zeros_like(T2); T2b[idx] = 1.0 / r2b
            out["sensitivity_tr_correction"][name] = {
                t: {"T2_median_ms": float(np.median(T2b[fitmask & (dseg == lab)])),
                    "T2_median_change_pct": float(100 * (np.median(T2b[fitmask & (dseg == lab)])
                                                         / np.median(T2[fitmask & (dseg == lab)]) - 1))}
                for t, lab in zip(TISSUES, a.dseg_labels)}
    if a.fit_json:
        Path(a.fit_json).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["tissue"], indent=1))


# ----------------------------------------------------------------------------- finish-maps

def overlay_png(img, ref, png, title, window, zooms=(1, 1, 1)):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sh = img.shape
    zooms = np.asarray(zooms, float)
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    cuts = [(0, [0.35, 0.5, 0.65]), (1, [0.4, 0.5, 0.6]), (2, [0.35, 0.5, 0.65])]
    for row, (ax_i, fr) in enumerate(cuts):
        for col, f in enumerate(fr):
            k = int(sh[ax_i] * f)
            sl = np.take(img, k, axis=ax_i).T; rf = np.take(ref, k, axis=ax_i).T
            ax = axs[row, col]
            other = [k for k in range(3) if k != ax_i]  # (cols, rows) after .T
            ax.imshow(sl, cmap="gray", vmin=window[0], vmax=window[1], origin="lower",
                      aspect=zooms[other[1]] / zooms[other[0]])
            ax.contour(rf, levels=[0.5], colors=["r"], linewidths=0.6)
            ax.set_title(f"axis {ax_i} idx {k}"); ax.axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)


def cmd_finish_maps(a):
    like, s0 = load_f32(a.s0); r2 = load_f32(a.r2)[1]
    cov = load_f32(a.coverage)[1] if a.coverage else np.ones_like(s0)
    valid = (cov > a.min_coverage) & (r2 > 0) & (s0 > 0)
    t2 = np.zeros_like(s0)
    t2[valid] = np.clip(1.0 / r2[valid], *T2_CLAMP)
    s0 = np.where(valid, s0, 0).astype(np.float32)
    save(a.out_t2, t2, like); save(a.out_s0, s0, like)
    if a.s0raw and a.out_s0raw:
        s0r = load_f32(a.s0raw)[1]; save(a.out_s0raw, np.where(valid, s0r, 0), like)
    print(f"[finish-maps] {a.out_t2}: valid {int(valid.sum())} voxels, "
          f"T2 median {np.median(t2[valid]):.1f} ms")
    if a.overlay and a.overlay_png:
        ref = load_f32(a.overlay)[1]
        if ref.shape != t2.shape:
            sys.exit(f"overlay reference {a.overlay} shape {ref.shape} != map {t2.shape}")
        overlay_png(t2, ref, a.overlay_png, f"T2 map (40-150 ms) with {Path(a.overlay).name} = 0.5 contour",
                    (40, 150), like.header.get_zooms()[:3])


# ----------------------------------------------------------------------------- predict

def resample_to(img, target_img, order=1):
    from nibabel.processing import resample_from_to
    r = resample_from_to(img, (target_img.shape[:3], target_img.affine), order=order)
    return r.get_fdata(dtype=np.float32)


def pearson(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if x.size < 3:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compare(pred, real, masks, scale_mask):
    """Scale pred by the WM median of real/pred, then per-mask r and pred/real quantiles."""
    k = float(np.median(real[scale_mask] / pred[scale_mask]))
    p = pred * k
    out = {"scale_from_WM_median": k, "regions": {}}
    # low-frequency part of log(pred/real) over the brain (Gaussian sigma 6 voxels ~ 10 mm):
    # a smooth residual field points at bias-field / coil-profile mismatch rather than tissue error
    from scipy.ndimage import gaussian_filter
    bm = masks["brain"]
    lr = np.clip(np.where(bm, np.log(np.maximum(p, 1e-3) / np.maximum(real, 1e-3)), 0.0), -1, 1)
    num = gaussian_filter(lr * bm, 6); den = gaussian_filter(bm.astype(float), 6)
    low = np.where(den > 0.2, num / np.maximum(den, 1e-6), 0)
    pct = np.percentile(np.exp(low[bm]), [5, 50, 95])
    out["lowfreq_ratio_field_p5_p50_p95"] = [float(x) for x in pct]
    out["highfreq_rms_log_ratio"] = float(np.sqrt(np.mean((lr[bm] - low[bm]) ** 2)))
    for name, m in masks.items():
        ratio = p[m] / real[m]
        out["regions"][name] = {"pearson_r": pearson(p[m], real[m]), "ratio_pred_over_real": quantiles(ratio),
                                "median_abs_pct_err": float(100 * np.median(np.abs(ratio - 1))),
                                "rms_log_ratio": float(np.sqrt(np.mean(np.log(ratio) ** 2)))}
    return p, out


def cmd_predict(a):
    dwi_im = nib.load(a.dwi)
    bval = np.loadtxt(a.bval).ravel()
    b0_idx = np.where(bval < a.b0_thresh)[0]
    real = np.mean(np.stack([np.asanyarray(dwi_im.dataobj[..., int(i)]).astype(np.float32)
                             for i in b0_idx], -1), -1)
    like = nib.Nifti1Image(real, dwi_im.affine)
    print(f"[predict] real b0 = mean of {len(b0_idx)} volumes with b < {a.b0_thresh}")

    t2_im, t2 = load_f32(a.t2); s0 = load_f32(a.s0)[1]
    if t2.shape != real.shape or not np.allclose(t2_im.affine, dwi_im.affine, atol=1e-3):
        sys.exit("T2/S0 maps must be on the DWI grid")
    dmask = load_f32(a.dwi_mask)[1] > 0.5
    probs = {}
    for t, p in zip(TISSUES, a.probseg):
        probs[t] = np.clip(resample_to(nib.load(p), dwi_im, 1), 0, None)
    psum = sum(probs.values())
    frac = {t: np.where(psum > 0, probs[t] / np.maximum(psum, 1e-6), 0) for t in TISSUES}

    # T1-saturation ratio TR_dwi vs TR_ref, tissue-fraction weighted; unlabeled -> GM
    sat_dwi = sum(frac[t] * sat(a.tr, T1_NOMINAL[t]) for t in TISSUES)
    sat_ref = sum(frac[t] * sat(a.tr_ref, T1_NOMINAL[t]) for t in TISSUES)
    unl = psum <= 0
    sat_dwi[unl] = sat(a.tr, T1_NOMINAL["GM"]); sat_ref[unl] = sat(a.tr_ref, T1_NOMINAL["GM"])
    t1ratio = sat_dwi / sat_ref

    valid = (t2 > 0) & (s0 > 0)
    pred = np.where(valid, s0 * np.exp(-a.te / np.maximum(t2, 1e-3)) * t1ratio, 0).astype(np.float32)

    # pure-tissue masks: probseg > 0.9 (on the DWI grid), eroded 1 voxel, inside brain and valid
    from scipy.ndimage import binary_erosion
    brain = dmask & valid & (real > 0)
    pure = {t: binary_erosion(probs[t] > a.pure_thresh, iterations=1) & brain for t in TISSUES}
    masks = {"brain": brain, **pure}

    # baseline: per-tissue S0 = class mean of the fitted S0 in pure voxels; fixed T2 per tissue
    s0_class = {t: float(s0[pure[t]].mean()) for t in TISSUES}
    base = np.zeros_like(pred)
    for t in TISSUES:
        base += frac[t] * s0_class[t] * np.exp(-a.te / T2_BASELINE[t]) * sat(a.tr, T1_NOMINAL[t]) \
                / sat(a.tr_ref, T1_NOMINAL[t])
    base[unl] = 0
    base = np.where(valid & (psum > 0), base, 0).astype(np.float32)
    brain_b = brain & (base > 0)
    masks_b = {"brain": brain_b, **{t: pure[t] & brain_b for t in TISSUES}}

    pred_s, m_map = compare(pred, real, masks, pure["WM"])
    base_s, m_base = compare(base, real, masks_b, pure["WM"])

    metrics = {
        "inputs": {"t2": a.t2, "s0": a.s0, "dwi": a.dwi, "n_b0": int(len(b0_idx)), "b0_thresh": a.b0_thresh},
        "prediction": {"te_ms": a.te, "tr_s": a.tr, "tr_ref_s": a.tr_ref, "t1_nominal_s": T1_NOMINAL,
                       "formula": "S0*exp(-TE/T2)*(1-exp(-TR/T1))/(1-exp(-TR_ref/T1)), T1 mixed by probseg"},
        "baseline": {"t2_ms": T2_BASELINE, "s0_class_mean_pure": s0_class,
                     "formula": "sum_t f_t*S0_t*exp(-TE/T2_t)*sat_t(TR)/sat_t(TR_ref)"},
        "pure_tissue_def": f"probseg > {a.pure_thresh} on the DWI grid, eroded 1 voxel",
        "mask_sizes": {k: int(v.sum()) for k, v in masks.items()},
        "map": m_map, "baseline_metrics": m_base,
        "T2_ms_pure": {t: quantiles(t2[pure[t]]) for t in TISSUES},
        "S0_pure": {t: quantiles(s0[pure[t]]) for t in TISSUES},
        "S0_ratio_GM_WM": float(np.median(s0[pure["GM"]]) / np.median(s0[pure["WM"]])),
        "S0_ratio_CSF_WM": float(np.median(s0[pure["CSF"]]) / np.median(s0[pure["WM"]])),
        "real_b0_contrast": {"GM_over_WM": float(np.median(real[pure["GM"]]) / np.median(real[pure["WM"]])),
                             "CSF_over_WM": float(np.median(real[pure["CSF"]]) / np.median(real[pure["WM"]]))},
        "pred_b0_contrast": {"GM_over_WM": float(np.median(pred_s[pure["GM"]]) / np.median(pred_s[pure["WM"]])),
                             "CSF_over_WM": float(np.median(pred_s[pure["CSF"]]) / np.median(pred_s[pure["WM"]]))},
        "baseline_b0_contrast": {"GM_over_WM": float(np.median(base_s[pure["GM"]]) / np.median(base_s[pure["WM"]])),
                                 "CSF_over_WM": float(np.median(base_s[pure["CSF"]]) / np.median(base_s[pure["WM"]]))},
    }
    # partial-volume-weighted per-tissue view (all brain voxels weighted by fraction), which
    # does not depend on the pure-voxel selection
    for name, pr, mk in [("map", pred_s, brain), ("baseline", base_s, brain_b)]:
        metrics[f"{name}_fraction_weighted_ratio"] = {
            t: float(np.sum(frac[t][mk] * (pr[mk] / real[mk])) / np.sum(frac[t][mk])) for t in TISSUES}

    save(a.out_pred, pred_s, like); save(a.out_baseline, base_s, like)
    if a.out_realb0:
        save(a.out_realb0, real, like)
    rep = Path(a.report_dir); rep.mkdir(parents=True, exist_ok=True)
    (rep / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({"map": m_map, "baseline": m_base}, indent=1))

    make_figures(rep, real, pred_s, base_s, t2, brain, pure, a)


def make_figures(rep, real, pred, base, t2, brain, pure, a):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    zs = np.where(brain.any((0, 1)))[0]
    slices = np.linspace(zs[0] + 0.15 * len(zs), zs[-1] - 0.15 * len(zs), 6).astype(int)
    vmax = np.percentile(real[brain], 99)
    cols = [("real b0", real, "gray", (0, vmax)), ("pred b0 (T2/S0 map)", pred, "gray", (0, vmax)),
            ("pred b0 (baseline consts)", base, "gray", (0, vmax)), ("T2 map [40-150 ms]", t2, "viridis", (40, 150)),
            ("map/real [0.7-1.3]", np.where(brain, pred / np.maximum(real, 1e-3), np.nan), "RdBu_r", (0.7, 1.3)),
            ("baseline/real [0.7-1.3]", np.where(brain, base / np.maximum(real, 1e-3), np.nan), "RdBu_r", (0.7, 1.3))]
    fig, axs = plt.subplots(len(slices), len(cols), figsize=(2.6 * len(cols), 2.6 * len(slices)))
    for i, z in enumerate(slices):
        for j, (name, img, cm, (lo, hi)) in enumerate(cols):
            ax = axs[i, j]
            ax.imshow(img[:, :, z].T, cmap=cm, vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
            ax.axis("off")
            if i == 0:
                ax.set_title(name, fontsize=9)
            if j == 0:
                ax.text(2, 2, f"z={z}", color="w", fontsize=8)
    fig.tight_layout(); fig.savefig(rep / "b0_montage.png", dpi=110); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    lim = (0, vmax * 1.2)
    for ax, (name, p) in zip(axs, [("T2/S0 map", pred), ("baseline constants", base)]):
        m = brain & (p > 0)
        ax.hist2d(real[m], p[m], bins=150, range=[lim, lim], norm=LogNorm(), cmap="magma")
        ax.plot(lim, lim, "c--", lw=0.8)
        for t, c in zip(TISSUES, ["#1f77b4", "#2ca02c", "#d62728"]):
            mm = pure[t] & m
            ax.scatter(real[mm][::7], p[mm][::7], s=1.5, c=c, alpha=0.35, label=f"pure {t}")
        ax.set_xlabel("real mean b0"); ax.set_ylabel(f"predicted b0 ({name})")
        ax.set_title(f"{name}: r = {pearson(real[m], p[m]):.3f}"); ax.legend(markerscale=6, fontsize=8)
    fig.tight_layout(); fig.savefig(rep / "b0_scatter.png", dpi=110); plt.close(fig)

    if a.t2w and a.reg_check:
        t2w_im, t2w = load_f32(a.t2w); mov = load_f32(a.reg_check)[1]
        if mov.shape != t2w.shape:
            print("[predict] registration-check image is not on the T2w grid; skipping overlay")
            return
        bm = load_f32(a.brain_mask)[1] > 0.5 if a.brain_mask else t2w > 0
        from scipy.ndimage import sobel, gaussian_filter
        g = gaussian_filter(mov, 1.0)
        edge = np.sqrt(sum(sobel(g, axis=i) ** 2 for i in range(3)))
        thr = np.percentile(edge[bm], 85)
        fig, axs = plt.subplots(3, 4, figsize=(16, 12))
        for row, ax_i in enumerate([2, 1, 0]):
            idxs = np.where(bm.any(tuple(k for k in range(3) if k != ax_i)))[0]
            picks = np.linspace(idxs[0] + 0.2 * len(idxs), idxs[-1] - 0.2 * len(idxs), 4).astype(int)
            for col, k in enumerate(picks):
                ax = axs[row, col]
                bg = np.take(t2w, k, axis=ax_i).T; ed = np.take(edge, k, axis=ax_i).T
                ax.imshow(bg, cmap="gray", vmin=0, vmax=np.percentile(t2w[bm], 99), origin="lower")
                ax.contour(ed, levels=[thr], colors=["r"], linewidths=0.5)
                ax.axis("off"); ax.set_title(f"axis {ax_i} idx {k}", fontsize=9)
        fig.suptitle("registration check: SDC-corrected MESE echo-3 edges (red) over ACPC preproc T2w")
        fig.tight_layout(); fig.savefig(rep / "registration_check.png", dpi=100); plt.close(fig)


# ----------------------------------------------------------------------------- sdc-qc

def cmd_sdc_qc(a):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ims = [load_f32(p)[1] for p in [a.ap_raw, a.pa_raw, a.ap_sdc, a.pa_sdc]]
    names = ["AP raw", "PA raw", "AP corrected", "PA corrected"]
    zs = [int(ims[0].shape[2] * f) for f in (0.35, 0.5, 0.65)]
    vmax = np.percentile(ims[0], 99.5)
    fig, axs = plt.subplots(len(zs), 6, figsize=(18, 3.2 * len(zs)))
    for i, z in enumerate(zs):
        for j, (n, im) in enumerate(zip(names, ims)):
            axs[i, j].imshow(im[:, :, z].T, cmap="gray", vmin=0, vmax=vmax, origin="lower"); axs[i, j].axis("off")
            axs[i, j].set_title(n if i == 0 else "", fontsize=9)
        d0 = (ims[0][:, :, z] - ims[1][:, :, z]).T; d1 = (ims[2][:, :, z] - ims[3][:, :, z]).T
        axs[i, 4].imshow(d0, cmap="RdBu_r", vmin=-vmax / 2, vmax=vmax / 2, origin="lower"); axs[i, 4].axis("off")
        axs[i, 5].imshow(d1, cmap="RdBu_r", vmin=-vmax / 2, vmax=vmax / 2, origin="lower"); axs[i, 5].axis("off")
        if i == 0:
            axs[i, 4].set_title("AP-PA raw", fontsize=9); axs[i, 5].set_title("AP-PA corrected", fontsize=9)
    fig.suptitle("topup: MESE echo-1 AP/PA before and after correction (axial)")
    fig.tight_layout(); fig.savefig(a.png, dpi=100); plt.close(fig)
    m = ims[0] > np.percentile(ims[0], 90)
    print(f"[sdc-qc] AP/PA echo-1 correlation raw {pearson(ims[0][m], ims[1][m]):.4f} -> "
          f"corrected {pearson(ims[2][m], ims[3][m]):.4f}")


# ----------------------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    f = sp.add_parser("fit")
    f.add_argument("--echoes", nargs="+", required=True)
    f.add_argument("--te", nargs="+", type=float, required=True, help="ms, one per echo")
    f.add_argument("--tr", nargs="+", type=float, required=True, help="s, one per echo")
    f.add_argument("--tr-ref", type=float, default=None, help="TR the S0 map refers to (default: echo 1's)")
    f.add_argument("--dseg", help="tissue label image in the echoes' space")
    f.add_argument("--dseg-labels", nargs=3, type=int, default=[3, 2, 1], metavar=("WM", "GM", "CSF"))
    f.add_argument("--brain-mask", help="brain mask in the echoes' space (restricts the fit, dilated)")
    f.add_argument("--brain-mask-dilate", type=int, default=2)
    f.add_argument("--bias", help="multiplicative receive bias field (N4) to divide out of S0")
    f.add_argument("--mask-frac", type=float, default=0.12, help="echo-1 threshold as fraction of p99")
    f.add_argument("--out-prefix", required=True)
    f.add_argument("--fit-json")
    f.set_defaults(func=cmd_fit)

    g = sp.add_parser("finish-maps")
    g.add_argument("--s0", required=True); g.add_argument("--r2", required=True)
    g.add_argument("--s0raw"); g.add_argument("--coverage")
    g.add_argument("--min-coverage", type=float, default=0.5)
    g.add_argument("--out-t2", required=True); g.add_argument("--out-s0", required=True)
    g.add_argument("--out-s0raw")
    g.add_argument("--overlay", help="reference image on the same grid (e.g. WM probseg) for a QC PNG")
    g.add_argument("--overlay-png")
    g.set_defaults(func=cmd_finish_maps)

    p = sp.add_parser("predict")
    p.add_argument("--t2", required=True); p.add_argument("--s0", required=True)
    p.add_argument("--dwi", required=True); p.add_argument("--bval", required=True)
    p.add_argument("--dwi-mask", required=True)
    p.add_argument("--probseg", nargs=3, required=True, metavar=("WM", "GM", "CSF"))
    p.add_argument("--te", type=float, default=88.0); p.add_argument("--tr", type=float, default=4.8)
    p.add_argument("--tr-ref", type=float, default=5.71)
    p.add_argument("--b0-thresh", type=float, default=50.0)
    p.add_argument("--pure-thresh", type=float, default=0.9)
    p.add_argument("--out-pred", required=True); p.add_argument("--out-baseline", required=True)
    p.add_argument("--out-realb0")
    p.add_argument("--report-dir", required=True)
    p.add_argument("--t2w"); p.add_argument("--reg-check", help="moving image resampled onto the T2w grid")
    p.add_argument("--brain-mask")
    p.set_defaults(func=cmd_predict)

    q = sp.add_parser("sdc-qc")
    for k in ["ap-raw", "pa-raw", "ap-sdc", "pa-sdc", "png"]:
        q.add_argument(f"--{k}", required=True)
    q.set_defaults(func=cmd_sdc_qc)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
