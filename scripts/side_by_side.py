#!/usr/bin/env python3
"""Side-by-side real vs simulated DWI (same subject, same scanner-frame grid): magnitude at
b=0/1000/3000 in axial, coronal and sagittal views, phase, and a tight-window background panel
(ghost/noise), all on matched windows. Real data is resampled onto the sim grid and put on the sim
scale by the WM-b0 median ratio, like compare_real_sim.py.

usage: side_by_side.py --real-dir D --sim-dir D --grid-dir D --pe PA --sim-stem 'x_dir-{pe}_part-{part}_dwi' --out PREFIX
"""
import argparse
from pathlib import Path
import numpy as np, nibabel as nb
from nibabel.processing import resample_from_to
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).parent))
from calibrate_phase import siemens_phase_to_radians  # noqa: E402


def shell_mean(mag, bval, b, tol=150):
    sel = np.abs(bval - b) < tol
    return mag[..., sel].mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real-dir', required=True); ap.add_argument('--sim-dir', required=True)
    ap.add_argument('--grid-dir', required=True); ap.add_argument('--pe', default='PA')
    ap.add_argument('--real-stem', required=True, help='BIDS stem of the real DWI with {pe}/{part} placeholders')
    ap.add_argument('--sim-stem', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--label', default='sim')
    a = ap.parse_args()
    R = Path(a.real_dir); S = Path(a.sim_dir)
    rm = nb.load(str(R / (a.real_stem.format(pe=a.pe, part='mag') + '.nii.gz')))
    rp = nb.load(str(R / (a.real_stem.format(pe=a.pe, part='phase') + '.nii.gz')))
    rb = np.loadtxt(str(R / (a.real_stem.format(pe=a.pe, part='mag') + '.bval')))
    sm = nb.load(str(S / (a.sim_stem.format(pe=a.pe, part='mag') + '.nii.gz')))
    sp = nb.load(str(S / (a.sim_stem.format(pe=a.pe, part='phase') + '.nii.gz')))
    sbp = S / (a.sim_stem.format(pe=a.pe, part='mag').replace('_part-mag_dwi', '_dwi') + '.bval')
    sb = np.loadtxt(str(sbp))
    smag = sm.get_fdata(dtype=np.float32); sph = sp.get_fdata(dtype=np.float32)
    rmag = rm.get_fdata(dtype=np.float32); rph = siemens_phase_to_radians(rp.get_fdata(dtype=np.float32))
    tgt = (smag.shape[:3], sm.affine)
    rs = lambda v, im: resample_from_to(nb.Nifti1Image(v, im.affine), tgt, order=1).get_fdata()
    # grid files are LPS, the output LAS: always go through the affines, never reuse indices
    mask = resample_from_to(nb.load(str(Path(a.grid_dir) / 'mask.nii.gz')), tgt, order=0).get_fdata() > 0.5
    wm = resample_from_to(nb.load(str(Path(a.grid_dir) / 'wm.nii.gz')), tgt, order=1).get_fdata() > 0.7
    shells = [0, 1000, 3000]
    real = {b: rs(shell_mean(rmag, rb, b), rm) for b in shells}
    sim = {b: shell_mean(smag, sb, b) for b in shells}
    scale = np.median(real[0][mask & wm]) / max(np.median(sim[0][mask & wm]), 1e-9)
    sim = {b: v * scale for b, v in sim.items()}
    # phase: one representative volume per shell (first of the shell), resampled via cos/sin
    def phase_vol(ph, bval, b):
        i = int(np.where(np.abs(bval - b) < 150)[0][0]); return ph[..., i]
    rphase = {b: np.arctan2(rs(np.sin(phase_vol(rph, rb, b)), rp), rs(np.cos(phase_vol(rph, rb, b)), rp)) for b in shells}
    sphase = {b: phase_vol(sph, sb, b) for b in shells}

    nx, ny, nz = smag.shape[:3]
    zs = [int(nz * f) for f in (0.35, 0.5, 0.65)]
    win = {b: np.percentile(real[b][mask], 99.5) for b in shells}

    # Figure 1: magnitude, axial x3 + coronal + sagittal, real/sim pairs per shell
    views = [('axial', lambda v, z=z: v[:, :, z].T) for z in zs] + \
            [('coronal', lambda v: v[:, ny // 2, :].T), ('sagittal', lambda v: v[nx // 2, :, :].T)]
    fig, ax = plt.subplots(len(views), 2 * len(shells), figsize=(2.6 * 2 * len(shells), 2.8 * len(views)))
    for r, (vn, cut) in enumerate(views):
        for c, b in enumerate(shells):
            for k, (name, vol) in enumerate((('real', real[b]), (a.label, sim[b]))):
                a_ = ax[r, 2 * c + k]
                a_.imshow(cut(vol), cmap='gray', origin='lower', vmin=0, vmax=win[b], aspect='equal')
                a_.set_title(f'{name} b={b} {vn}', fontsize=8); a_.axis('off')
    plt.suptitle(f'{a.pe}: real vs {a.label} — magnitude, shared window per shell (99.5th pct of real)', fontsize=10)
    plt.tight_layout(); plt.savefig(a.out + '_magnitude.png', dpi=90); plt.close()

    # Figure 2: phase (axial) real/sim per shell, plus tight-window background (ghost/noise) for b=0 and b=3000
    fig, ax = plt.subplots(2, 2 * len(shells), figsize=(2.6 * 2 * len(shells), 6))
    z = zs[1]
    for c, b in enumerate(shells):
        for k, (name, vol) in enumerate((('real', rphase[b]), (a.label, sphase[b]))):
            a_ = ax[0, 2 * c + k]
            a_.imshow(np.where(mask[:, :, z], vol[:, :, z], np.nan).T, cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
            a_.set_title(f'{name} phase b={b}', fontsize=8); a_.axis('off')
        for k, (name, vol) in enumerate((('real', real[b]), (a.label, sim[b]))):
            a_ = ax[1, 2 * c + k]
            a_.imshow(vol[:, :, z].T, cmap='gray', origin='lower', vmin=0, vmax=0.06 * win[b])
            a_.set_title(f'{name} b={b} ×16 window (ghost/noise)', fontsize=8); a_.axis('off')
    plt.tight_layout(); plt.savefig(a.out + '_phase_background.png', dpi=90); plt.close()

    # Figure 3: intensity profiles through the centre (b0, b1000, b3000) along x and y, inside the
    # brain mask only (the real image carries scalp the simulated object does not have)
    fig, ax = plt.subplots(2, len(shells), figsize=(4 * len(shells), 6))
    bm = lambda v: np.where(mask, v, np.nan)
    for c, b in enumerate(shells):
        ax[0, c].plot(bm(real[b])[:, ny // 2, z], 'k', lw=1, label='real'); ax[0, c].plot(bm(sim[b])[:, ny // 2, z], 'r', lw=1, label=a.label)
        ax[0, c].set_title(f'b={b}: profile along x (row y={ny//2}, z={z}), brain only', fontsize=8)
        ax[1, c].plot(bm(real[b])[nx // 2, :, z], 'k', lw=1); ax[1, c].plot(bm(sim[b])[nx // 2, :, z], 'r', lw=1)
        ax[1, c].set_title(f'b={b}: profile along y (PE) (col x={nx//2}, z={z}), brain only', fontsize=8)
    ax[0, 0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(a.out + '_profiles.png', dpi=90); plt.close()
    print('wrote', a.out + '_{magnitude,phase_background,profiles}.png; scale sim→real', round(scale, 2))


if __name__ == '__main__':
    main()
