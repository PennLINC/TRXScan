#!/usr/bin/env python3
"""Compare TRXScan simulated complex DWI against a real acquisition of the same subject.

Both must share a world frame (e.g. simulate in the scanner frame with the DRBUDDI field).
The real data is resampled onto the simulation's acquisition grid, put on the sim intensity
scale by one global WM-b0 factor, then compared per shell and per tissue in magnitude AND phase.

Outputs a JSON of metrics and, with --png, montages + a self-contained HTML report.

Needs: nibabel, numpy, scipy, matplotlib. Phase conversion reuses scripts/calibrate_phase.py.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, nibabel as nb
from nibabel.processing import resample_from_to

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrate_phase import siemens_phase_to_radians, WRAP_CEILING


def load4d(p):
    im = nb.load(str(p)); return im, np.asanyarray(im.get_fdata(dtype=np.float32))


def shell_index(bval, centers=(0, 500, 1000, 2000, 3000), tol=150):
    bval = np.asarray(bval); out = {}
    for c in centers:
        idx = np.where(np.abs(bval - c) <= tol)[0]
        if len(idx): out[c] = idx
    return out


def circ_sd(ph):
    """Circular SD (rad) of a 1-D array of phases."""
    if len(ph) == 0: return np.nan
    R = np.abs(np.mean(np.exp(1j * ph)))
    return float(np.sqrt(-2.0 * np.log(max(R, 1e-12))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--real-dir', required=True)
    ap.add_argument('--sim-dir', required=True)
    ap.add_argument('--grid-dir', required=True, help='scanner acq grid dir with wm/gm/csf/mask.nii.gz')
    ap.add_argument('--pe', default='PA', choices=['PA', 'AP'])
    ap.add_argument('--real-stem', default='sub-60501_ses-01_acq-HBCD75_rec-norm_dir-{pe}_run-01_part-{part}_dwi')
    ap.add_argument('--sim-stem', default='sub-60501_ses-01_dir-{pe}_part-{part}_dwi')
    ap.add_argument('--out', required=True)
    ap.add_argument('--png', action='store_true')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)

    realm = Path(a.real_dir) / (a.real_stem.format(pe=a.pe, part='mag') + '.nii.gz')
    realp = Path(a.real_dir) / (a.real_stem.format(pe=a.pe, part='phase') + '.nii.gz')
    realbval = Path(a.real_dir) / (a.real_stem.format(pe=a.pe, part='mag') + '.bval')
    simm = Path(a.sim_dir) / (a.sim_stem.format(pe=a.pe, part='mag') + '.nii.gz')
    simp = Path(a.sim_dir) / (a.sim_stem.format(pe=a.pe, part='phase') + '.nii.gz')
    simbval = Path(a.sim_dir) / (a.sim_stem.format(pe=a.pe, part='') .replace('part-_', '') + '.bval')
    if not simbval.exists():
        simbval = Path(a.sim_dir) / (a.sim_stem.format(pe=a.pe, part='mag').replace('_part-mag_dwi', '_dwi') + '.bval')

    sim_im, sim_mag = load4d(simm)
    _, sim_ph = load4d(simp)
    sbval = np.loadtxt(simbval)

    # tissue maps + mask on the sim acq grid
    G = Path(a.grid_dir)
    wm = nb.load(str(G / 'wm.nii.gz')).get_fdata().astype(np.float32)
    gm = nb.load(str(G / 'gm.nii.gz')).get_fdata().astype(np.float32)
    csf = nb.load(str(G / 'csf.nii.gz')).get_fdata().astype(np.float32)
    mask = nb.load(str(G / 'mask.nii.gz')).get_fdata() > 0.5
    assert wm.shape == sim_mag.shape[:3], (wm.shape, sim_mag.shape)

    # real onto sim grid
    rim = nb.load(str(realm)); rdata = rim.get_fdata(dtype=np.float32)
    rbval = np.loadtxt(realbval)
    real_mag = np.stack([resample_from_to(nb.Nifti1Image(rdata[..., i], rim.affine),
                                          (sim_mag.shape[:3], sim_im.affine), order=1).get_fdata()
                         for i in range(rdata.shape[3])], -1).astype(np.float32)
    rpim = nb.load(str(realp)); rpdata = rpim.get_fdata(dtype=np.float32)
    real_ph_rad_native = siemens_phase_to_radians(rpdata)  # convert on native grid then resample cos/sin
    # resample phase via cos/sin to avoid wrap artifacts
    def resample_phase(ph):
        c = np.stack([resample_from_to(nb.Nifti1Image(np.cos(ph[..., i]), rpim.affine),
                                       (sim_mag.shape[:3], sim_im.affine), order=1).get_fdata() for i in range(ph.shape[3])], -1)
        s = np.stack([resample_from_to(nb.Nifti1Image(np.sin(ph[..., i]), rpim.affine),
                                       (sim_mag.shape[:3], sim_im.affine), order=1).get_fdata() for i in range(ph.shape[3])], -1)
        return np.arctan2(s, c).astype(np.float32)
    real_ph = resample_phase(real_ph_rad_native)

    rsh = shell_index(rbval); ssh = shell_index(sbval)
    shells = sorted(set(rsh) & set(ssh))

    # global scale: WM-b0 median ratio
    wmmask = mask & (wm > 0.7)
    r_b0 = real_mag[..., rsh[0]].mean(-1); s_b0 = sim_mag[..., ssh[0]].mean(-1)
    scale = float(np.median(r_b0[wmmask]) / max(np.median(s_b0[wmmask]), 1e-9))
    sim_mag_s = sim_mag * scale

    tis = {'WM': mask & (wm > 0.7), 'GM': mask & (gm > 0.7), 'CSF': mask & (csf > 0.7)}
    metrics = {'pe': a.pe, 'global_scale_sim_to_real': scale, 'shells': shells,
               'n_tissue_vox': {k: int(v.sum()) for k, v in tis.items()}, 'per_tissue': {}, 'phase': {}}

    # voxelwise b0 agreement (real vs scaled sim), whole brain and per tissue: Pearson r and the
    # median absolute percentage error. Tissue-mean metrics cannot see within-tissue structure
    # (a per-voxel T2/S0 map changes r, not the class means).
    s_b0s = s_b0 * scale
    vox = {}
    for tname, tm in [('brain', mask & (r_b0 > 0))] + list(tis.items()):
        rr, ss = r_b0[tm].astype(np.float64), s_b0s[tm].astype(np.float64)
        ok = rr > 0
        rr, ss = rr[ok], ss[ok]
        r = float(np.corrcoef(rr, ss)[0, 1]) if rr.size > 2 else float('nan')
        vox[tname] = {'n': int(rr.size), 'pearson_r': r,
                      'mdape_pct': float(np.median(np.abs(ss / rr - 1.0)) * 100.0),
                      'ratio_iqr': [float(np.percentile(ss / rr, 25)), float(np.percentile(ss / rr, 75))]}
    metrics['b0_voxelwise'] = vox

    # per-tissue b0 level (real vs scaled sim) and S(b)/S0 decay
    for tname, tm in tis.items():
        d = {'real_b0_mean': float(r_b0[tm].mean()), 'sim_b0_mean': float((s_b0 * scale)[tm].mean()),
             'real_S_over_S0': {}, 'sim_S_over_S0': {}}
        r0 = r_b0[tm].mean(); s0 = s_b0[tm].mean()
        for b in shells:
            if b == 0: continue
            rb = real_mag[..., rsh[b]].mean(-1)[tm].mean()
            sb = sim_mag[..., ssh[b]].mean(-1)[tm].mean()
            d['real_S_over_S0'][b] = float(rb / max(r0, 1e-9))
            d['sim_S_over_S0'][b] = float(sb / max(s0, 1e-9))
        metrics['per_tissue'][tname] = d

    # background noise floor (corner cube) + SNR in WM per shell
    corner = np.zeros(sim_mag.shape[:3], bool); corner[:12, :12, :] = True; corner &= ~mask
    rn = float(real_mag[corner][..., None].std()) if corner.sum() else np.nan
    sig_r = {int(b): float(real_mag[..., rsh[b]].mean(-1)[tis['WM']].mean()) for b in shells}
    sig_s = {int(b): float(sim_mag_s[..., ssh[b]].mean(-1)[tis['WM']].mean()) for b in shells}
    # sim background std
    sn = float(sim_mag_s[corner].std()) if corner.sum() else np.nan
    metrics['noise'] = {'real_bg_std': rn, 'sim_bg_std': sn,
                        'real_WM_SNR': {b: sig_r[b] / rn if rn else np.nan for b in sig_r},
                        'sim_WM_SNR': {b: sig_s[b] / sn if sn else np.nan for b in sig_s}}

    # phase: circular SD per shell in WM (real vs sim)
    for b in shells:
        if b == 0: continue
        rp = np.concatenate([real_ph[..., i][tis['WM']] for i in rsh[b]])
        sp = np.concatenate([sim_ph[..., i][tis['WM']] for i in ssh[b]])
        metrics['phase'][int(b)] = {'real_circ_sd': circ_sd(rp), 'sim_circ_sd': circ_sd(sp),
                                    'wrap_ceiling': float(WRAP_CEILING)}
    # phase spatial gradient RMS (b0, WM) — simple central-diff in-plane
    def grad_rms(ph3, m):
        gx = np.angle(np.exp(1j * (ph3[1:, :, :] - ph3[:-1, :, :])))
        gy = np.angle(np.exp(1j * (ph3[:, 1:, :] - ph3[:, :-1, :])))
        mm = m[1:, :, :] & m[:-1, :, :]
        my = m[:, 1:, :] & m[:, :-1, :]
        vals = np.concatenate([np.abs(gx[mm]), np.abs(gy[my])])
        return float(np.sqrt(np.mean(vals**2))) if len(vals) else np.nan
    metrics['phase']['b0_grad_rms_WM'] = {'real': grad_rms(real_ph[..., rsh[0][0]], tis['WM']),
                                          'sim': grad_rms(sim_ph[..., ssh[0][0]], tis['WM'])}

    with open(a.out + '.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=float)
    print('wrote', a.out + '.json')
    print(json.dumps({'scale': round(scale, 4),
                      'WM': {'real_b0': round(metrics['per_tissue']['WM']['real_b0_mean'], 1),
                             'sim_b0': round(metrics['per_tissue']['WM']['sim_b0_mean'], 1)},
                      'S/S0@b1000': {t: (round(metrics['per_tissue'][t]['real_S_over_S0'].get(1000, np.nan), 3),
                                         round(metrics['per_tissue'][t]['sim_S_over_S0'].get(1000, np.nan), 3))
                                     for t in tis},
                      'phase_circ_sd': {b: (round(metrics['phase'][b]['real_circ_sd'], 2),
                                            round(metrics['phase'][b]['sim_circ_sd'], 2)) for b in metrics['phase'] if isinstance(b, int)},
                      'bg_std': (round(metrics['noise']['real_bg_std'], 1), round(metrics['noise']['sim_bg_std'], 1))}, indent=2))

    if a.png:
        make_pngs(a.out, real_mag, sim_mag_s, real_ph, sim_ph, rsh, ssh, shells, mask)


def make_pngs(out, real_mag, sim_mag_s, real_ph, sim_ph, rsh, ssh, shells, mask):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    z = mask.sum((0, 1)).argmax()
    bset = [b for b in (0, 1000, 3000) if b in shells]
    fig, ax = plt.subplots(4, len(bset), figsize=(4 * len(bset), 14))
    for j, b in enumerate(bset):
        rm = real_mag[..., rsh[b]].mean(-1)[:, :, z].T
        sm = sim_mag_s[..., ssh[b]].mean(-1)[:, :, z].T
        vmax = np.percentile(rm[rm > 0], 99) if (rm > 0).any() else 1
        rp = real_ph[..., rsh[b][0]][:, :, z].T
        sp = sim_ph[..., ssh[b][0]][:, :, z].T
        for i, (im, ttl, cmap, vlim) in enumerate([
                (rm, f'real mag b{b}', 'gray', (0, vmax)),
                (sm, f'sim mag b{b}', 'gray', (0, vmax)),
                (rp, f'real phase b{b}', 'twilight', (-np.pi, np.pi)),
                (sp, f'sim phase b{b}', 'twilight', (-np.pi, np.pi))]):
            a_ = ax[i, j]; a_.imshow(im, cmap=cmap, origin='lower', vmin=vlim[0], vmax=vlim[1])
            a_.set_title(ttl, fontsize=9); a_.axis('off')
    plt.tight_layout(); plt.savefig(out + '_montage.png', dpi=85); plt.close()
    print('wrote', out + '_montage.png')


if __name__ == '__main__':
    main()
