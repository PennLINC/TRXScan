"""Array conventions at the Rust boundary.

TRXScan stores a 3-D volume as ``x + nx*(y + ny*z)``: the memory order of an F-contiguous
``(nx, ny, nz)`` array, which is exactly what nibabel hands out. A 4-D series is
``(x + nx*(y + ny*z))*ngrad + g``: the memory order of a C-contiguous ``(nz, ny, nx, ngrad)``
array. Per-slice k-space is ``kx + nx*ky``: a C-contiguous ``(ny, nx)`` array, phase-encode axis
first, which is also the book's convention. The binding therefore exchanges *flat* 1-D buffers
and this module does the (zero-copy) reshaping on the Python side.
"""

from __future__ import annotations

from typing import Any

import nibabel as nib
import numpy as np


def flat_f32(a: Any) -> np.ndarray:
    """A 3-D image (nibabel image or array) as the flat float32 buffer Rust expects."""
    if hasattr(a, "dataobj"):
        a = np.asanyarray(a.dataobj)
    a = np.asarray(a)
    if a.ndim == 4 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got shape {a.shape}")
    return np.ascontiguousarray(np.asfortranarray(a, dtype=np.float32).ravel(order="F"))


def flat_series_f32(a: np.ndarray) -> np.ndarray:
    """A 4-D ``(nx, ny, nz, ngrad)`` array as the flat interleaved buffer Rust expects."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 4:
        raise ValueError(f"expected a 4-D series, got shape {a.shape}")
    return np.ascontiguousarray(a.transpose(2, 1, 0, 3).ravel(order="C"))


def volume(flat: np.ndarray, dims: tuple[int, int, int]) -> np.ndarray:
    """Flat Rust buffer -> ``(nx, ny, nz)`` F-order view."""
    return np.asarray(flat).reshape(tuple(dims), order="F")


def series(flat: np.ndarray, dims: tuple[int, int, int], ngrad: int) -> np.ndarray:
    """Flat interleaved Rust buffer -> ``(nx, ny, nz, ngrad)`` view (no copy)."""
    nx, ny, nz = dims
    return np.asarray(flat).reshape((nz, ny, nx, ngrad)).transpose(2, 1, 0, 3)


def vectors(flat: np.ndarray, dims: tuple[int, int, int], k: int) -> np.ndarray:
    """Flat ``vox*k + c`` buffer -> ``(nx, ny, nz, k)`` view."""
    return series(flat, dims, k)


def complex_pairs(flat: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Flat float32 ``[re, im]`` pairs -> complex64 array of ``shape`` (no copy)."""
    return np.asarray(flat, dtype=np.float32).view(np.complex64).reshape(shape)


def affine4(a: Any) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 16:
        a = a.reshape(4, 4)
    if a.shape != (4, 4):
        raise ValueError(f"affine must be 4x4, got {a.shape}")
    return np.ascontiguousarray(a)


def nifti(data: np.ndarray, affine: np.ndarray, *, dtype=None) -> nib.Nifti1Image:
    """A Nifti1Image with qform and sform both set to ``affine`` (code 1, scanner), matching
    TRXScan's writer (``io::header_for_grid``)."""
    if dtype is not None:
        data = np.asarray(data, dtype=dtype)
    img = nib.Nifti1Image(data, affine)
    img.header.set_qform(affine, code=1)
    img.header.set_sform(affine, code=1)
    img.header.set_xyzt_units("mm")
    return img


def as_gtab(gtab: Any) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    """Duck-type a dipy ``GradientTable`` (or a ``(bvals, bvecs)`` pair) into
    ``(bvals (n,), bvecs (n, 3), big_delta, small_delta)``."""
    if hasattr(gtab, "bvals") and hasattr(gtab, "bvecs"):
        bvals = np.asarray(gtab.bvals, dtype=np.float64)
        bvecs = np.asarray(gtab.bvecs, dtype=np.float64)
        bd = getattr(gtab, "big_delta", None)
        sd = getattr(gtab, "small_delta", None)
    else:
        bvals, bvecs = gtab
        bvals = np.asarray(bvals, dtype=np.float64)
        bvecs = np.asarray(bvecs, dtype=np.float64)
        bd = sd = None
    bvals = bvals.reshape(-1)
    if bvecs.shape == (3, bvals.size) and bvals.size != 3:
        bvecs = bvecs.T
    if bvecs.shape != (bvals.size, 3):
        raise ValueError(f"bvecs must be (n, 3) matching {bvals.size} bvals, got {bvecs.shape}")
    return bvals, np.ascontiguousarray(bvecs), (None if bd is None else float(bd)), (None if sd is None else float(sd))


def b_max_of(bvals: np.ndarray) -> float:
    """The scheme's reference b-value (its maximum), with a b0-only scheme mapped to 1000 so
    the Fiberfox gradient scaling ``sqrt(b_i / b_max)`` stays defined (every gradient is zero
    then, so the value does not matter)."""
    b = float(np.max(bvals)) if np.size(bvals) else 0.0
    return b if b > 0.0 else 1000.0
