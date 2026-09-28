"""Synthetic objects: small scenes that go through the same signal and k-space stages as real
anatomy, for dipy.sims-style pedagogy with real artifacts. Each returns an
:class:`~trxscan.phantom.Object`."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import _core
from .phantom import Fibers, Object

FiberSpec = Sequence[tuple[Sequence[float], float]]


def _block_mean(sim: np.ndarray, o: int) -> np.ndarray:
    """In-plane block average of an ``(snx, sny, nz)`` array by ``o``."""
    snx, sny, nz = sim.shape
    return sim.reshape(snx // o, o, sny // o, o, nz).mean(axis=(1, 3))


def _fibers_where(wm: np.ndarray, fibers: FiberSpec | None, region: np.ndarray | None = None) -> Fibers | None:
    if not fibers:
        return None
    where = wm > 0 if region is None else (wm > 0) & region
    vox = np.flatnonzero(where.ravel(order="F")).astype(np.uint32)
    dirs, weight, voxel = [], [], []
    for d, w in fibers:
        d = np.asarray(d, dtype=np.float64)
        d = d / max(np.linalg.norm(d), 1e-12)
        voxel.append(vox)
        dirs.append(np.tile(d, (vox.size, 1)))
        weight.append(np.full(vox.size, float(w)))
    return Fibers(np.concatenate(voxel), np.concatenate(dirs), np.concatenate(weight))


def from_arrays(
    sim_wm: np.ndarray,
    sim_gm: np.ndarray | None = None,
    sim_csf: np.ndarray | None = None,
    *,
    oversample: int = 2,
    voxel_mm: float | Sequence[float] = 2.0,
    fibers: FiberSpec | Fibers | None = None,
    fieldmap: np.ndarray | None = None,
    affine: np.ndarray | None = None,
    name: str = "synthetic",
) -> Object:
    """An object from compartment-fraction arrays on the SIMULATION grid ``(nx*o, ny*o, nz)``.
    ``fibers`` is a :class:`~trxscan.phantom.Fibers` or a list of ``(direction, weight)`` pairs
    applied to every voxel with ``wm > 0``. ``fieldmap`` (Hz) is on the simulation grid. The
    acquisition-grid fractions are the in-plane block means. The default affine centres the
    FOV on the origin with the given voxel size."""
    o = int(oversample)
    sim_wm = np.asarray(sim_wm, dtype=np.float32)
    if sim_wm.ndim == 2:
        sim_wm = sim_wm[:, :, None]
    snx, sny, nz = sim_wm.shape
    if snx % o or sny % o:
        raise ValueError(f"simulation grid {sim_wm.shape} is not a multiple of oversample {o} in-plane")
    nx, ny = snx // o, sny // o
    sim_gm = np.zeros_like(sim_wm) if sim_gm is None else np.asarray(sim_gm, np.float32).reshape(sim_wm.shape)
    sim_csf = np.zeros_like(sim_wm) if sim_csf is None else np.asarray(sim_csf, np.float32).reshape(sim_wm.shape)
    vox = (float(voxel_mm),) * 3 if np.isscalar(voxel_mm) else tuple(float(v) for v in voxel_mm)  # type: ignore[arg-type]
    if affine is None:
        affine = np.diag([vox[0], vox[1], vox[2], 1.0])
        affine[:3, 3] = -np.array([(nx - 1) / 2 * vox[0], (ny - 1) / 2 * vox[1], (nz - 1) / 2 * vox[2]])
    affine = np.asarray(affine, dtype=np.float64)
    sim_affine_flat, sim_dims = _core.hires_grid(affine, (nx, ny, nz), o)
    sim_mask = ((sim_wm + sim_gm + sim_csf) > 0.3).astype(np.float32)
    wm, gm, csf = _block_mean(sim_wm, o), _block_mean(sim_gm, o), _block_mean(sim_csf, o)
    mask = ((wm + gm + csf) > 0.3).astype(np.float32)
    fmap = np.zeros(sim_wm.shape, np.float32) if fieldmap is None else np.asarray(fieldmap, np.float32).reshape(sim_wm.shape)
    fib = fibers if isinstance(fibers, Fibers) else _fibers_where(sim_wm, fibers)
    F = lambda a: np.ascontiguousarray(np.asfortranarray(a).ravel(order="F"))  # noqa: E731
    return Object(
        dims=(nx, ny, nz), affine=affine, oversample=o, sim_dims=tuple(int(v) for v in sim_dims), sim_affine=sim_affine_flat.reshape(4, 4),
        wm=F(wm), gm=F(gm), csf=F(csf), mask=F(mask), sim_wm=F(sim_wm), sim_gm=F(sim_gm), sim_csf=F(sim_csf), sim_mask=F(sim_mask),
        sim_fmap=F(fmap), fmap=F(_block_mean(fmap, o)), fibers=fib, voxel_mm=vox, name=name,
    )


def box(
    size_vox: float | tuple[float, float] = 8.0,
    *,
    matrix: int = 32,
    edge: float = 0.3,
    oversample: int = 2,
    voxel_mm: float = 2.0,
    nz: int = 1,
    fibers: FiberSpec | None = (((1.0, 0.0, 0.0), 1.0),),
    gm: float = 0.0,
    csf: float = 0.0,
    fieldmap: np.ndarray | None = None,
) -> Object:
    """A rectangle of ``size_vox`` acquired voxels centred in a ``matrix`` x ``matrix`` slice,
    with its edges offset by ``edge`` of a voxel so partial-volume edges (and hence ringing)
    are real (``kspace::box_hires``). The box is white matter with the given fibres; ``gm`` /
    ``csf`` are fractions inside the box (WM takes the rest)."""
    o = int(oversample)
    n = int(matrix)
    sx, sy = (size_vox, size_vox) if np.isscalar(size_vox) else size_vox  # type: ignore[misc]
    x0 = (n / 2 - sx / 2 + edge) * o
    x1 = (n / 2 + sx / 2 + edge) * o
    y0 = (n / 2 - sy / 2 + edge) * o
    y1 = (n / 2 + sy / 2 + edge) * o
    occ = np.asarray(_core.box_hires(n * o, n * o, x0, x1, y0, y1), np.float32).reshape(n * o, n * o, order="C").T  # kx + nx*ky -> (x, y)
    occ = np.repeat(occ[:, :, None], nz, axis=2)
    wm = occ * (1.0 - gm - csf)
    return from_arrays(wm, occ * gm, occ * csf, oversample=o, voxel_mm=voxel_mm, fibers=fibers, fieldmap=fieldmap, name="box")


def disc(
    radius_vox: float = 8.0,
    *,
    matrix: int = 32,
    oversample: int = 2,
    voxel_mm: float = 2.0,
    nz: int = 1,
    fibers: FiberSpec | None = (((1.0, 0.0, 0.0), 1.0),),
    gm: float = 0.0,
    csf: float = 0.0,
    center: tuple[float, float] = (0.3, 0.2),
    fieldmap: np.ndarray | None = None,
) -> Object:
    """A disc of ``radius_vox`` acquired voxels (sub-voxel ``center`` offset), anti-aliased on
    the simulation grid; a curved edge rings in both axes."""
    o = int(oversample)
    n = int(matrix)
    s = n * o
    yy, xx = np.mgrid[0:s, 0:s]
    cx, cy = (n / 2 + center[0]) * o, (n / 2 + center[1]) * o
    r = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy) / o
    occ = np.clip(radius_vox - r + 0.5 / o, 0, 1.0 / o) * o
    occ = occ.T.astype(np.float32)
    occ = np.repeat(occ[:, :, None], nz, axis=2)
    wm = occ * (1.0 - gm - csf)
    return from_arrays(wm, occ * gm, occ * csf, oversample=o, voxel_mm=voxel_mm, fibers=fibers, fieldmap=fieldmap, name="disc")


def crossing(
    angle_deg: float = 90.0,
    *,
    matrix: int = 32,
    width_vox: float = 6.0,
    oversample: int = 2,
    voxel_mm: float = 2.0,
    nz: int = 1,
    weights: tuple[float, float] = (0.5, 0.5),
) -> Object:
    """Two straight fibre bands crossing at ``angle_deg`` in the slice centre: single-fibre
    voxels along each arm, a two-fibre mixture where they overlap."""
    o = int(oversample)
    n = int(matrix)
    s = n * o
    yy, xx = np.mgrid[0:s, 0:s]
    c = s / 2
    d1 = np.array([1.0, 0.0])
    a = np.radians(angle_deg)
    d2 = np.array([np.cos(a), np.sin(a)])
    half = width_vox * o / 2

    def band(d: np.ndarray) -> np.ndarray:
        nrm = np.array([-d[1], d[0]])
        dist = np.abs((xx + 0.5 - c) * nrm[0] + (yy + 0.5 - c) * nrm[1])
        return (dist <= half).T.astype(np.float32)

    b1, b2 = band(d1), band(d2)
    wm = np.clip(b1 + b2, 0, 1)
    wm = np.repeat(wm[:, :, None], nz, axis=2)
    reg1 = np.repeat((b1 > 0)[:, :, None], nz, axis=2)
    reg2 = np.repeat((b2 > 0)[:, :, None], nz, axis=2)
    f1 = _fibers_where(wm, [((1.0, 0.0, 0.0), weights[0])], reg1)
    f2 = _fibers_where(wm, [((float(d2[0]), float(d2[1]), 0.0), weights[1])], reg2)
    assert f1 is not None and f2 is not None
    fib = Fibers(np.concatenate([f1.voxel, f2.voxel]), np.concatenate([f1.dirs, f2.dirs]), np.concatenate([f1.weight, f2.weight]))
    return from_arrays(wm, oversample=o, voxel_mm=voxel_mm, fibers=fib, name="crossing")


def fill(obj: Object, voxel: Any) -> Object:
    """Give every WM voxel of ``obj`` the mixture of a :class:`~trxscan.voxel.Voxel` (its fibre
    directions and weights, and its GM/CSF fractions scaled by the object's occupancy)."""
    import dataclasses

    from ._arrays import volume

    occ = volume(obj.sim_wm + obj.sim_gm + obj.sim_csf, obj.sim_dims)
    wm = occ * voxel.wm
    fib = _fibers_where(wm, voxel.fibers)
    F = lambda a: np.ascontiguousarray(np.asfortranarray(a).ravel(order="F"))  # noqa: E731
    o = obj.oversample
    return dataclasses.replace(
        obj, sim_wm=F(wm), sim_gm=F(occ * voxel.gm), sim_csf=F(occ * voxel.csf), fibers=fib,
        wm=F(_block_mean(wm, o)), gm=F(_block_mean(occ * voxel.gm, o)), csf=F(_block_mean(occ * voxel.csf, o)),
    )
