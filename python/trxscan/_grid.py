"""Grid the anatomical phantom onto an acquisition grid and the oversampled simulation grid.

A line-for-line port of TRXScan's ``scripts/prepare_acquisition_grid.py``: the tissue bounding
box (``w + g + c > 0.3``) at the requested voxel size, padded per side (generously along the
phase-encode ``j`` axis so EPI distortion cannot wrap), half-voxel centring, and the simulation
grid resampled from the ANATOMICAL source (never from the acquisition grid: the source carries
the frequencies above the acquisition Nyquist that Gibbs ringing comes from).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from . import _core
from ._arrays import flat_f32

if TYPE_CHECKING:  # pragma: no cover
    from .phantom import Object, Phantom


def _resample(img: nib.Nifti1Image, target: tuple[tuple[int, int, int], np.ndarray], clip: bool) -> np.ndarray:
    r = resample_from_to(img, target, order=1)
    a = r.get_fdata(dtype=np.float32)
    return np.clip(a, 0, None) if clip else a


def grid_phantom(
    phantom: "Phantom", voxel_mm: tuple[float, float, float], oversample: int, pad: tuple[int, int, int],
    matrix: tuple[int, int, int] | None = None, target: tuple[tuple[int, int, int], np.ndarray] | None = None,
    oblique_deg: tuple[float, float, float] | None = None,
) -> "Object":
    """``matrix`` fixes the acquired matrix instead of bounding box + ``pad``: the tissue
    bounding box is centred in it (the FOV of a real scan). An axis the object overruns is
    cropped, centred, with a warning (a scanner would wrap it). ``target`` ``(shape, affine)``
    uses that exact acquisition grid instead (the grid of another run, so a moved subject is
    sampled inside the same field of view). ``oblique_deg`` rotates the grid about its own
    centre (an oblique acquisition; the object does not move)."""
    from .phantom import Object

    if oversample < 1:
        raise ValueError("oversample must be a positive integer")
    probs = {"wm": phantom.wm, "gm": phantom.gm, "csf": phantom.csf}
    src = probs["wm"]
    tissue = sum(np.asanyarray(p.dataobj, dtype=np.float32) for p in probs.values())
    VOX = np.asarray(voxel_mm, dtype=np.float64)
    PAD = np.asarray(pad, dtype=np.int64)
    nz_ = np.argwhere(tissue > 0.3)
    if nz_.size == 0:
        raise ValueError("tissue fractions are empty (nothing above 0.3)")
    lo, hi = nz_.min(0), nz_.max(0) + 1
    A = np.asarray(src.affine, dtype=np.float64)
    zooms = np.asarray(src.header.get_zooms()[:3], dtype=np.float64)
    if not np.all(zooms > 0):
        raise ValueError(f"source has a non-positive voxel size {tuple(zooms)}")
    directions = A[:3, :3] / zooms
    gram = directions.T @ directions
    if not np.allclose(gram, np.eye(3), atol=1e-6):
        raise ValueError("source affine is sheared; an isotropic target voxel is not well defined")
    step = VOX / zooms
    bbox = np.array([int(np.ceil(s / k)) for s, k in zip(hi - lo, step)])
    if matrix is None:
        shape = tuple(int(b + 2 * p) for b, p in zip(bbox, PAD))
        pad_lo = PAD.astype(np.float64)
    else:
        shape = tuple(int(v) for v in matrix)
        over = [i for i in range(3) if bbox[i] > shape[i]]
        if over:
            import warnings

            warnings.warn(f"object spans {tuple(bbox)} voxels but the matrix is {shape}: cropping axis {over}", stacklevel=3)
        pad_lo = (np.asarray(shape) - bbox) / 2.0
    if target is not None:
        shape = tuple(int(v) for v in target[0])
        affine = np.asarray(target[1], dtype=np.float64).reshape(4, 4)
    else:
        M = np.eye(4)
        M[:3, :3] = np.diag(step)
        M[:3, 3] = lo + (step - 1) / 2.0 - step * pad_lo
        affine = A @ M
    if oblique_deg is not None and np.any(np.asarray(oblique_deg) != 0):
        from .bids import rigid_offset

        centre = affine @ np.append((np.asarray(shape) - 1) / 2.0, 1.0)
        affine = rigid_offset((0.0, 0.0, 0.0), oblique_deg, centre[:3]) @ affine
    got = np.linalg.norm(affine[:3, :3], axis=0)
    if not np.allclose(got, VOX, rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"internal error: target spacing {got} != requested {VOX}")
    target = (shape, affine)
    acq = {k: _resample(p, target, True) for k, p in probs.items()}
    mask = ((acq["wm"] + acq["gm"] + acq["csf"]) > 0.3).astype(np.float32)
    fmap_img = phantom.fieldmap
    fmap = _resample(fmap_img, target, False) if fmap_img is not None else np.zeros(shape, np.float32)

    if oversample > 1:
        sim_affine_flat, sim_shape = _core.hires_grid(affine, shape, oversample)
        sim_affine = sim_affine_flat.reshape(4, 4)
        sim_target = (sim_shape, sim_affine)
        sim = {k: _resample(p, sim_target, True) for k, p in probs.items()}
        sim_mask = ((sim["wm"] + sim["gm"] + sim["csf"]) > 0.3).astype(np.float32)
        sim_fmap = _resample(fmap_img, sim_target, False) if fmap_img is not None else np.zeros(sim_shape, np.float32)
    else:
        sim_affine, sim_shape = affine, shape
        sim, sim_mask, sim_fmap = acq, mask, fmap

    # Non-brain head for the GRE magnitude: the subject's T1w outside the (soft) brain, in units
    # of the brain's own T1w level, so scalp/skull/neck ride along with the tissue mixture.
    sim_head = None
    if phantom.t1w is not None:
        t1 = _resample(phantom.t1w, (tuple(sim_shape), sim_affine), True)
        brain = np.clip(sim["wm"] + sim["gm"] + sim["csf"], 0.0, 1.0)
        inside = t1[brain > 0.5]
        level = float(np.median(inside)) if inside.size else 0.0
        if level > 0:
            sim_head = np.clip(t1 / level, 0.0, None) * (1.0 - brain)

    return Object(
        dims=shape, affine=affine, oversample=oversample, sim_dims=tuple(sim_shape), sim_affine=sim_affine,
        wm=flat_f32(acq["wm"]), gm=flat_f32(acq["gm"]), csf=flat_f32(acq["csf"]), mask=flat_f32(mask),
        sim_wm=flat_f32(sim["wm"]), sim_gm=flat_f32(sim["gm"]), sim_csf=flat_f32(sim["csf"]), sim_mask=flat_f32(sim_mask),
        sim_fmap=flat_f32(sim_fmap), fmap=flat_f32(fmap), streamlines=phantom.streamlines, fibers=None,
        sim_head=None if sim_head is None else flat_f32(sim_head), voxel_mm=tuple(float(v) for v in VOX), name=phantom.name,
    )
