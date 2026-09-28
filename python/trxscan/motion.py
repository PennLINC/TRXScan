"""Head motion as one rigid pose per volume: :class:`Motion`.

Poses are absolute (each volume relative to the un-moved head), about the FOV centre, with
rotations composed as Rz*Ry*Rx in **degrees** internally (``Pose::to_matrix``). Loaders from
real preprocessing output convert from their own conventions and say so.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import _core

Kind = Literal["trajectory", "random", "linear"]


@dataclass(frozen=True, eq=False)
class Motion:
    """A per-volume rigid motion specification, resolved against a gradient table at
    simulation time (:meth:`poses_for`)."""

    kind: Kind
    #: ``(n, 6)`` ``[tx, ty, tz mm, rx, ry, rz deg]`` for ``"trajectory"``; unused otherwise.
    poses: np.ndarray | None = None
    trans_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    volumes: tuple[int, ...] | None = None
    seed: int = 0
    source: str = ""

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_arrays(cls, trans_mm: Any, rot: Any, *, rot_units: Literal["deg", "rad"] = "deg", source: str = "arrays") -> "Motion":
        """From ``(n, 3)`` translations (mm) and ``(n, 3)`` rotations, or one ``(n, 6)`` array
        as ``trans_mm`` with ``rot=None``."""
        t = np.asarray(trans_mm, dtype=np.float64)
        if rot is None:
            if t.ndim != 2 or t.shape[1] != 6:
                raise ValueError("expected an (n, 6) array when rot is None")
            r = t[:, 3:6]
            t = t[:, 0:3]
        else:
            r = np.asarray(rot, dtype=np.float64)
        if t.shape != r.shape or t.ndim != 2 or t.shape[1] != 3:
            raise ValueError(f"trans_mm and rot must both be (n, 3); got {t.shape} and {r.shape}")
        if rot_units == "rad":
            r = np.degrees(r)
        return cls(kind="trajectory", poses=np.ascontiguousarray(np.hstack([t, r])), source=source)

    @classmethod
    def from_confounds(cls, path: str | Path) -> "Motion":
        """From a qsiprep / fMRIPrep-style confounds TSV: columns ``trans_x/y/z`` (mm) and
        ``rot_x/y/z`` (**radians**), one row per volume; ``n/a`` -> 0. Exactly what the CLI's
        ``--motion`` reads (``motion::load_motion_tsv``)."""
        path = Path(path)
        with open(path) as f:
            header = f.readline().rstrip("\n").split("\t")
            rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
        try:
            idx = [header.index(c) for c in ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")]
        except ValueError as e:
            raise ValueError(f"{path}: missing motion column ({e})") from e

        def val(s: str) -> float:
            try:
                return float(s)
            except ValueError:
                return 0.0

        arr = np.array([[val(r[i]) if i < len(r) else 0.0 for i in idx] for r in rows], dtype=np.float64)
        if arr.size == 0:
            raise ValueError(f"{path}: no rows")
        return cls.from_arrays(arr[:, :3], arr[:, 3:], rot_units="rad", source=str(path))

    @classmethod
    def from_eddy(cls, path: str | Path) -> "Motion":
        """From FSL eddy's ``.eddy_parameters`` (columns 1-3 translations mm, 4-6 rotations
        radians, then eddy-current terms) — the movement part only."""
        arr = np.loadtxt(path, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 6:
            raise ValueError("eddy_parameters must have at least 6 columns")
        return cls.from_arrays(arr[:, :3], arr[:, 3:6], rot_units="rad", source=str(path))

    @classmethod
    def from_affines(cls, affines: Any, center: tuple[float, float, float] | None = None, source: str = "affines") -> "Motion":
        """From one 4x4 world (RAS mm) rigid transform per volume, e.g. dipy/ANTs registration
        results mapping the moved head onto the reference. Each matrix is read as
        ``p' = R (p - c) + c + t`` about ``center`` ``c`` (the FOV centre when ``None``: pass the
        object's centre for exact reproduction), with Euler angles extracted for ``R = Rz Ry Rx``.
        """
        mats = np.asarray(affines, dtype=np.float64)
        if mats.ndim == 2:
            mats = mats[None]
        if mats.shape[1:] != (4, 4):
            raise ValueError("affines must be (n, 4, 4)")
        c = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64)
        poses = np.zeros((mats.shape[0], 6))
        for i, m in enumerate(mats):
            R = m[:3, :3]
            if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
                raise ValueError(f"affine {i} is not rigid (R R^T != I)")
            # R = Rz Ry Rx  ->  ry = -asin(R[2,0]); rx = atan2(R[2,1], R[2,2]); rz = atan2(R[1,0], R[0,0])
            ry = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
            rx = np.arctan2(R[2, 1], R[2, 2])
            rz = np.arctan2(R[1, 0], R[0, 0])
            t = m[:3, 3] - c + R @ c
            poses[i, :3] = t
            poses[i, 3:] = np.degrees([rx, ry, rz])
        return cls(kind="trajectory", poses=poses, source=source)

    @classmethod
    def random(cls, trans_mm: tuple[float, float, float], rot_deg: tuple[float, float, float], volumes: Any = None, *, seed: int = 0) -> "Motion":
        """Independent random impulses (uniform in +-amplitude per axis) on the listed volumes
        (all DWI volumes when ``None``), returning to baseline in between (Fiberfox's random
        mode)."""
        return cls(kind="random", trans_mm=tuple(trans_mm), rot_deg=tuple(rot_deg), volumes=None if volumes is None else tuple(int(v) for v in volumes), seed=seed, source="random")

    @classmethod
    def linear(cls, trans_mm: tuple[float, float, float], rot_deg: tuple[float, float, float], volumes: Any = None) -> "Motion":
        """Monotonic drift reaching the given end-of-scan totals (Fiberfox's linear mode)."""
        return cls(kind="linear", trans_mm=tuple(trans_mm), rot_deg=tuple(rot_deg), volumes=None if volumes is None else tuple(int(v) for v in volumes), source="linear")

    # -- resolution ----------------------------------------------------------

    @property
    def n_vol(self) -> int | None:
        return None if self.poses is None else int(self.poses.shape[0])

    def poses_for(self, n_vol: int, *, seed: int | None = None) -> np.ndarray:
        """``(n_vol, 6)`` absolute poses for a scheme of ``n_vol`` volumes. A trajectory longer
        than the scheme is truncated; a shorter one raises (use :meth:`resample`)."""
        if self.kind == "trajectory":
            assert self.poses is not None
            if self.poses.shape[0] < n_vol:
                raise ValueError(f"motion trace has {self.poses.shape[0]} volumes but the scheme has {n_vol}; use Motion.resample(n_vol) to interpolate")
            return np.ascontiguousarray(self.poses[:n_vol])
        vols = list(self.volumes) if self.volumes is not None else None
        flat = _core.resolve_poses(self.kind, n_vol, self.seed if seed is None else seed, tuple(self.trans_mm), tuple(self.rot_deg), vols, None)
        return flat.reshape(n_vol, 6)

    def resample(self, n_vol: int) -> "Motion":
        """Linearly interpolate a trajectory to ``n_vol`` volumes over the same scan time."""
        if self.kind != "trajectory":
            return dataclasses.replace(self)
        assert self.poses is not None
        n = self.poses.shape[0]
        if n == n_vol:
            return self
        src = np.linspace(0.0, 1.0, n)
        dst = np.linspace(0.0, 1.0, n_vol)
        out = np.column_stack([np.interp(dst, src, self.poses[:, c]) for c in range(6)])
        return dataclasses.replace(self, poses=out, source=f"{self.source} (resampled {n}->{n_vol})")

    def matrices(self, n_vol: int, center: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        """``(n_vol, 4, 4)`` world transforms (``Pose::to_matrix``) about ``center``."""
        flat = _core.poses_to_matrices(np.ascontiguousarray(self.poses_for(n_vol)), tuple(float(c) for c in center))
        return flat.reshape(n_vol, 4, 4)

    def framewise_displacement(self, n_vol: int | None = None, radius_mm: float = 50.0) -> np.ndarray:
        """Power-style FD per volume: sum of absolute translation and rotation (converted to arc
        length at ``radius_mm``) differences between consecutive volumes; first volume 0."""
        p = self.poses_for(n_vol if n_vol is not None else (self.n_vol or 0))
        d = np.diff(p, axis=0, prepend=p[:1])
        return np.abs(d[:, :3]).sum(axis=1) + np.abs(np.radians(d[:, 3:])).sum(axis=1) * radius_mm

    def plot(self, n_vol: int | None = None, ax: Any = None):
        """Plot translations and rotations per volume (matplotlib)."""
        import matplotlib.pyplot as plt

        p = self.poses_for(n_vol if n_vol is not None else (self.n_vol or 0))
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(9, 3))
        for i, lab in enumerate("xyz"):
            ax[0].plot(p[:, i], label=f"trans {lab} (mm)")
            ax[1].plot(p[:, 3 + i], label=f"rot {lab} (deg)")
        for a in ax:
            a.set_xlabel("volume")
            a.legend(fontsize=8)
        ax[0].set_title(self.source or self.kind)
        return ax
