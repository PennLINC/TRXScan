#!/usr/bin/env python3
"""Prepare a TRXScan acquisition grid from the bundled anatomical-grid inputs.

Resamples the WM/GM/CSF probsegs and the fieldmap onto an isotropic acquisition grid
(default 1.7 mm, HBCD-like), cropped to the tissue bounding box and padded along the
phase-encode (y) axis so EPI distortion cannot wrap around the FOV. Optionally writes a
geometric myelination map (EDT depth x posterior->anterior ramp) for the infant preset.

`--voxel` is in millimetres and is resolved against the source affine's zooms, so any
anatomical grid works; the bundled inputs happen to be 1 mm isotropic, but nothing here
assumes it. `--pad` remains in TARGET voxels, which is what the EPI wrap-around argument
is actually about.

Needs: python3 with nibabel and scipy.

Example (from the bundle root):
    python3 scripts/prepare_acquisition_grid.py \
        --anat-dir sub-0001a/anat --prefix sub-0001a_space-ACPC --out work --voxel 1.7
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anat-dir", required=True, type=Path,
                    help="directory holding <prefix>_label-{WM,GM,CSF}_probseg.nii.gz and "
                         "<prefix>_desc-atlas_fieldmap.nii.gz")
    ap.add_argument("--prefix", required=True, help="e.g. sub-0001a_space-ACPC")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--voxel", type=float, default=1.7,
                    help="acquisition voxel size, mm isotropic. Resolved against the SOURCE "
                         "affine's zooms, so it means millimetres for any input grid, not only "
                         "for a 1 mm anatomical.")
    ap.add_argument("--oversample", type=int, default=1, metavar="N",
                    help="also write a simulation grid at voxel/N in-plane (same FOV, same "
                         "slice thickness) into <out>/sim. N must be a positive integer: the "
                         "k-space crop assumes an integer matrix ratio. The slice direction "
                         "is never oversampled - it is not Fourier-encoded in 2D EPI. "
                         "N=4 is the recommended value for Gibbs-ringing work; 1 (default) "
                         "writes no simulation grid.")
    ap.add_argument("--pad", type=int, nargs=3, default=[4, 14, 2], metavar=("PX", "PY", "PZ"),
                    help="FOV padding in voxels per side; keep PY generous (PE axis) to avoid "
                         "EPI wrap-around")
    ap.add_argument("--myelin", action="store_true",
                    help="also write myelin.nii.gz (depth-based myelination proxy, 0..~0.7) "
                         "for use with the infant preset")
    a = ap.parse_args()
    if a.oversample < 1:
        ap.error("--oversample must be a positive integer")
    a.out.mkdir(parents=True, exist_ok=True)

    probs = {t: nib.load(a.anat_dir / f"{a.prefix}_label-{t}_probseg.nii.gz")
             for t in ["WM", "GM", "CSF"]}
    tissue = sum(p.get_fdata(dtype=np.float32) for p in probs.values())
    VOX, PAD = a.voxel, np.array(a.pad)
    nz = np.argwhere(tissue > 0.3)
    lo, hi = nz.min(0), nz.max(0) + 1
    A = probs["WM"].affine
    # `--voxel` is millimetres, so the index-space scale factor is VOX / (source zoom) PER AXIS.
    # Hardcoding `diag([VOX] * 3)` made the whole geometry -- target spacing, matrix size and the
    # `(VOX - 1) / 2` centring term alike -- silently assume a 1 mm isotropic source, which the
    # bundled anatomicals happen to be. On anything else the output was VOX * zoom mm per voxel
    # while the header claimed otherwise.
    zooms = np.asarray(probs["WM"].header.get_zooms()[:3], float)
    if not np.all(zooms > 0):
        raise SystemExit(f"source has a non-positive voxel size {tuple(zooms)}")
    step = VOX / zooms                       # target voxels expressed in source voxel indices
    shape = tuple(int(np.ceil(s / k)) + 2 * p for s, k, p in zip(hi - lo, step, PAD))
    M = np.eye(4)
    M[:3, :3] = np.diag(step)
    # Half-voxel centring: a target voxel spans `step` source voxels, so its centre sits
    # `(step - 1) / 2` source-voxel indices past the first source voxel it covers.
    M[:3, 3] = lo + (step - 1) / 2.0 - step * PAD
    affine = A @ M
    target = (shape, affine)
    got = np.linalg.norm(affine[:3, :3], axis=0)
    if not np.allclose(got, VOX, rtol=1e-6, atol=1e-6):
        raise SystemExit(
            f"internal error: target spacing {tuple(np.round(got, 6))} mm != requested {VOX} mm "
            f"(source zooms {tuple(zooms)}); a sheared source affine is not supported"
        )

    for t, key in [("WM", "wm"), ("GM", "gm"), ("CSF", "csf")]:
        r = resample_from_to(probs[t], target, order=1)
        nib.save(nib.Nifti1Image(np.clip(r.get_fdata(dtype=np.float32), 0, None), affine),
                 a.out / f"{key}.nii.gz")
    w, g, c = [nib.load(a.out / f"{k}.nii.gz").get_fdata() for k in ["wm", "gm", "csf"]]
    mask = ((w + g + c) > 0.3).astype(np.float32)
    nib.save(nib.Nifti1Image(mask, affine), a.out / "mask.nii.gz")

    fmap_src = a.anat_dir / f"{a.prefix}_desc-atlas_fieldmap.nii.gz"
    r = resample_from_to(nib.load(fmap_src), target, order=1)
    nib.save(nib.Nifti1Image(r.get_fdata(dtype=np.float32), affine), a.out / "fmap_hz.nii.gz")

    if a.myelin:
        from scipy.ndimage import distance_transform_edt
        edt = distance_transform_edt(mask > 0) * VOX
        depth = np.clip(edt / 20.0, 0, 1)
        jj = np.arange(shape[1])
        yw = (affine @ np.stack([np.zeros_like(jj), jj, np.zeros_like(jj),
                                 np.ones_like(jj)]))[1]
        ramp = 0.65 + 0.35 * (1.0 - (yw - yw.min()) / (yw.max() - yw.min()))
        my = (0.7 * depth * ramp[None, :, None]).astype(np.float32) * (mask > 0)
        nib.save(nib.Nifti1Image(my, affine), a.out / "myelin.nii.gz")

    if a.oversample > 1:
        n = a.oversample
        sim_dir = a.out / "sim"
        sim_dir.mkdir(parents=True, exist_ok=True)
        # Refine in-plane only, preserving the FOV: dims *= n, in-plane zooms /= n.
        sim_shape = (shape[0] * n, shape[1] * n, shape[2])
        sim_affine = affine.copy()
        sim_affine[:, 0] /= n
        sim_affine[:, 1] /= n
        # Keep the FOV edge fixed: the first sim voxel centre moves inward by half the
        # difference between the coarse and the fine voxel size on each refined axis.
        sim_affine[:3, 3] = (affine[:3, 3]
                             - 0.5 * affine[:3, 0] * (1 - 1.0 / n)
                             - 0.5 * affine[:3, 1] * (1 - 1.0 / n))
        sim_target = (sim_shape, sim_affine)
        # Resample from the ANATOMICAL source, never from the acquisition grid written above.
        # The source carries frequencies well above the acquisition Nyquist, and those are
        # exactly what generates Gibbs ringing when the acquired band is selected. Upsampling the
        # already-downsampled maps adds no k-space content, so cropping back would return them
        # unchanged and produce no ringing at all (spec 3.1, "On the 1 mm source").
        sim = {}
        for t, key in [("WM", "wm"), ("GM", "gm"), ("CSF", "csf")]:
            r = resample_from_to(probs[t], sim_target, order=1)
            sim[key] = np.clip(r.get_fdata(dtype=np.float32), 0, None)
            nib.save(nib.Nifti1Image(sim[key], sim_affine), sim_dir / f"{key}.nii.gz")
        sim_mask = ((sim["wm"] + sim["gm"] + sim["csf"]) > 0.3).astype(np.float32)
        nib.save(nib.Nifti1Image(sim_mask, sim_affine), sim_dir / "mask.nii.gz")
        r = resample_from_to(nib.load(fmap_src), sim_target, order=1)
        nib.save(nib.Nifti1Image(r.get_fdata(dtype=np.float32), sim_affine),
                 sim_dir / "fmap_hz.nii.gz")
        print(f"sim grid {sim_shape} @ {VOX / n:.4f} mm in-plane x {VOX} mm slice, "
              f"brain voxels {int(sim_mask.sum())} -> {sim_dir}/")

    print(f"grid {shape} @ {VOX} mm, brain voxels {int(mask.sum())} -> {a.out}/")


if __name__ == "__main__":
    main()
