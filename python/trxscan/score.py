"""Scoring a pipeline's output against the simulation's ground truth: the generic half of a
truth-scored test (the pipeline-specific half, which transform maps its space to the
simulation's, stays with the pipeline). Everything takes NIfTI images or arrays already in one
frame; :func:`resample_like` brings a truth image onto an estimate's grid."""

from __future__ import annotations

from typing import Any

import nibabel as nib
import numpy as np


def _arr(x: Any) -> np.ndarray:
    return np.asanyarray(x.dataobj if hasattr(x, "dataobj") else x, dtype=np.float64)


def resample_like(img: nib.Nifti1Image, ref: nib.Nifti1Image, order: int = 1) -> np.ndarray:
    """``img`` resampled onto ``ref``'s grid (trilinear by default), as an array. Vector images
    (a trailing dimension of 3) are resampled per component."""
    from nibabel.processing import resample_from_to

    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        out = np.stack([resample_from_to(nib.Nifti1Image(data[..., c], img.affine), (ref.shape[:3], ref.affine), order=order).get_fdata() for c in range(data.shape[-1])], axis=-1)
        return out
    return resample_from_to(img, (ref.shape[:3], ref.affine), order=order).get_fdata()


def pe_displacement(fieldmap: nib.Nifti1Image, sidecar: dict[str, Any]) -> nib.Nifti1Image:
    """The susceptibility displacement a fieldmap (Hz, on the acquisition grid) implies for an
    acquisition with the sidecar's ``TotalReadoutTime`` and ``PhaseEncodingDirection``:
    apparent-minus-true RAS vectors in mm, three volumes (what
    :attr:`trxscan.Simulation.displacement` writes)."""
    ped = sidecar["PhaseEncodingDirection"]
    axis = {"i": 0, "j": 1, "k": 2}[ped[0]]
    sign = -1.0 if ped.endswith("-") else 1.0
    shift_vox = sign * _arr(fieldmap) * float(sidecar["TotalReadoutTime"])
    disp = shift_vox[..., None] * np.asarray(fieldmap.affine)[:3, axis]
    return nib.Nifti1Image(disp.astype(np.float32), fieldmap.affine)


def compare_displacement(estimate: Any, truth: Any, mask: Any = None, axis: Any = None) -> dict[str, float]:
    """Compare an estimated displacement field with the true one (both ``(..., 3)`` vector
    arrays or images in the same frame and units). Vectors are projected on ``axis`` (a unit
    vector; default: the truth's dominant direction) and compared inside ``mask``. Returns
    ``corr``, ``slope`` (estimate / truth, least squares through the origin), ``rms_residual``,
    ``rms_truth`` (the error of doing nothing), ``p99_estimate`` and ``p99_truth`` (mm)."""
    e, t = _arr(estimate).reshape(-1, 3), _arr(truth).reshape(-1, 3)
    if e.shape != t.shape:
        raise ValueError(f"estimate {e.shape} and truth {t.shape} differ; resample first")
    m = np.ones(e.shape[0], bool) if mask is None else (_arr(mask).reshape(-1) > 0)
    m &= np.isfinite(e).all(1) & np.isfinite(t).all(1)
    if axis is None:
        u, s, vt = np.linalg.svd(t[m], full_matrices=False)
        axis = vt[0]
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    pe, pt = e[m] @ axis, t[m] @ axis
    denom = float(pt @ pt)
    return {
        "corr": float(np.corrcoef(pe, pt)[0, 1]) if pe.std() > 0 and pt.std() > 0 else float("nan"),
        "slope": float(pe @ pt / denom) if denom > 0 else float("nan"),
        "rms_residual": float(np.sqrt(np.mean((pe - pt) ** 2))),
        "rms_truth": float(np.sqrt(np.mean(pt ** 2))),
        "p99_estimate": float(np.percentile(np.abs(pe), 99)),
        "p99_truth": float(np.percentile(np.abs(pt), 99)),
        "n": int(m.sum()),
    }


def image_similarity(a: Any, b: Any, mask: Any = None) -> float:
    """Pearson correlation of two images inside ``mask`` (the corrected image against the
    clean truth, say; compare with the uncorrected image's value)."""
    x, y = _arr(a).reshape(-1), _arr(b).reshape(-1)
    m = np.ones(x.size, bool) if mask is None else (_arr(mask).reshape(-1) > 0)
    m &= np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1])


def angular_error(estimate: Any, truth: Any, mask: Any = None) -> np.ndarray:
    """Per-voxel angle (degrees) between the truth's first peak and the nearest estimated
    peak. Both are ``(..., 3k)`` peak images (``desc-truthpeaks``: peak k in volumes
    3k..3k+2, scaled by mass; zero = no peak) or ``(..., 3)`` single directions; sign is
    ignored. Voxels without a truth peak (or outside ``mask``) are NaN."""
    e, t = _arr(estimate), _arr(truth)
    shape = t.shape[:-1]
    e, t = e.reshape(-1, e.shape[-1] // 3, 3), t.reshape(-1, t.shape[-1] // 3, 3)
    t0 = t[:, 0]
    nt = np.linalg.norm(t0, axis=1)
    ne = np.linalg.norm(e, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.abs(np.einsum("nkc,nc->nk", e, t0) / (ne * nt[:, None]))
    cos = np.where(ne > 0, cos, np.nan)
    best = np.nanmax(np.where(np.isnan(cos), -1.0, cos), axis=1)
    deg = np.degrees(np.arccos(np.clip(best, 0.0, 1.0)))
    deg[(nt == 0) | np.all(ne == 0, axis=1)] = np.nan
    if mask is not None:
        deg[_arr(mask).reshape(-1) <= 0] = np.nan
    return deg.reshape(shape)


# ─── ITK transforms ─────────────────────────────────────────────────────────

LPS = np.diag([-1.0, -1.0, 1.0])


def _euler_zxy(ax: float, ay: float, az: float) -> np.ndarray:
    """ITK ``Euler3DTransform`` with ``ComputeZYX`` off: ``R = Rz(az) Rx(ax) Ry(ay)``."""
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]); Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def _itk_matrix(name: str, params: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    p, c = np.asarray(params, dtype=np.float64).ravel(), np.asarray(fixed, dtype=np.float64).ravel()[:3]
    if name.startswith("Euler3D") or p.size == 6:
        A, t = _euler_zxy(*p[:3]), p[3:6]
    elif p.size == 12:
        A, t = p[:9].reshape(3, 3), p[9:12]
    elif p.size == 3:
        A, t = np.eye(3), p[:3]
    else:
        raise ValueError(f"unsupported ITK transform {name!r} with {p.size} parameters")
    M = np.eye(4)
    M[:3, :3] = A
    M[:3, 3] = t + c - A @ c
    return M


def read_itk_transform(path: Any) -> np.ndarray:
    """An ITK transform file as a 4x4 matrix in LPS mm: ``.txt`` (Insight text), ``.mat``
    (MATLAB, as ANTs/qsiprep write) or ``.h5`` composite (needs ``h5py``). Euler3D and affine
    transforms with a fixed centre are supported; a composite's stages are multiplied in order.
    The matrix maps output (fixed) points to input (moving) points, ITK's resampling
    convention, so ``antsApplyTransforms -t`` with it pulls the moving image onto the fixed
    grid. Use :func:`lps_to_ras` for an RAS point map."""
    from pathlib import Path

    path = Path(path)
    if path.suffix == ".txt":
        text = path.read_text()
        kinds = [ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("Transform:")]
        params = [np.array(ln.split(":", 1)[1].split(), float) for ln in text.splitlines() if ln.startswith("Parameters:")]
        fixed = [np.array(ln.split(":", 1)[1].split(), float) for ln in text.splitlines() if ln.startswith("FixedParameters:")]
        M = np.eye(4)
        for k, p, f in zip(kinds, params, fixed):
            M = M @ _itk_matrix(k, p, f)
        return M
    if path.suffix == ".mat":
        from scipy.io import loadmat

        m = loadmat(str(path))
        key = next(k for k in m if not k.startswith("__") and k != "fixed")
        return _itk_matrix(key, m[key], m.get("fixed", np.zeros(3)))
    if path.suffix == ".h5":
        import h5py

        M = np.eye(4)
        with h5py.File(path) as h:
            for g in sorted(k for k in h["TransformGroup"] if k != "0"):
                grp = h["TransformGroup"][g]
                kind = grp["TransformType"][()][0].decode() if grp["TransformType"].dtype.kind in "OS" else "Affine"
                M = M @ _itk_matrix(kind, grp["TransformParameters"][()], grp["TransformFixedParameters"][()])
        return M
    raise ValueError(f"unknown ITK transform format {path.suffix!r}")


def lps_to_ras(M: np.ndarray) -> np.ndarray:
    """Re-express a 4x4 point map from LPS to RAS coordinates (or back; it is an involution)."""
    F = np.diag([-1.0, -1.0, 1.0, 1.0])
    return F @ np.asarray(M, dtype=np.float64) @ F


def rigid_error(estimate: np.ndarray, truth: np.ndarray, center: Any = (0.0, 0.0, 0.0)) -> dict[str, float]:
    """How far a rigid/affine point map is from the true one: the rotation angle of
    ``estimate @ inv(truth)`` and the displacement of ``center`` between the two maps (mm).
    Both maps in the same frame (e.g. :func:`read_itk_transform` outputs, LPS)."""
    E, T = np.asarray(estimate, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    dA = E[:3, :3] @ np.linalg.inv(T[:3, :3])
    ang = float(np.degrees(np.arccos(np.clip((np.trace(dA) - 1) / 2, -1, 1))))
    c = np.append(np.asarray(center, dtype=np.float64), 1.0)
    return {"rotation_deg": ang, "translation_mm": float(np.linalg.norm((E @ c)[:3] - (T @ c)[:3]))}
