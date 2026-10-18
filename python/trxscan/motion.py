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
SliceOrder = Literal["interleaved", "sequential", "descending", "odd-first", "protocol"]


@dataclass(frozen=True, eq=False)
class ShotEvents:
    """Within-volume (per-shot) motion and dropout (P9). A shot is the slices excited together:
    one slice single-band, ``Protocol.mb`` slices multiband. Independent of :class:`Motion`
    (``Artifacts.motion``): asking for shot events never switches the signal stage to the
    per-volume re-simulation.

    ``slice_order`` is ``"interleaved"`` (even groups then odd, what multiband dropout has always
    used), ``"sequential"``, ``"descending"``, ``"odd-first"``, or ``"protocol"`` (the protocol's
    ``SliceTiming``, :meth:`Protocol.slice_timing_for`). Events are drawn at ``rate`` per
    diffusion-weighted volume (the CLI's ``--dropout-rate`` rule: a 1-3 mm jump, b=0 volumes
    exempt), or given as ``events``: a list of dicts (``volume``, ``shot``, ``severity``,
    ``jump_mm``, ``jump_deg``) or the path of an events TSV (``Simulation.motion_events``, or
    aslscan's ``desc-motionEvents_gt.tsv``). A dropped shot is attenuated
    ``1 - severity * b / b_max``; the jump persists to the end of its volume.

    ``spin_history`` adds the tissue's spin history (P9): tissue carried through the slice
    direction into a slice not yet excited is excited twice and is dark the second time; carried
    out of one it is skipped and is bright in the next volume. It needs ``Protocol.t1_s`` and
    ``Protocol.tr_s``, the whole slice stack (no ``chunk``, ``slices`` or ``z_mm``), no
    ``Protocol.head``, and events that jump or an ``Artifacts.motion``. The correction is
    :attr:`Simulation.spin_history`.
    """

    slice_order: SliceOrder = "interleaved"
    rate: float = 0.0
    events: Any = None
    spin_history: bool = False

    def __post_init__(self) -> None:
        if self.slice_order not in ("interleaved", "sequential", "descending", "odd-first", "protocol"):
            raise ValueError(f"slice_order {self.slice_order!r}: expected interleaved, sequential, descending, odd-first or protocol")
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError("rate is a probability in [0, 1]")
        if self.events is not None and self.rate > 0.0:
            raise ValueError("give events or a rate, not both")

    def offsets(self, nz: int, mb: int, tr_s: float, protocol: Any = None) -> list[float]:
        """Each of the ``nz`` slices' excitation offset (s) in a repetition of ``tr_s``."""
        if self.slice_order == "protocol":
            if protocol is None or protocol.tr_s is None:
                raise ValueError("slice_order='protocol' needs Protocol.tr_s (the slice times are checked against it)")
            st = protocol.slice_timing_for(nz)
            if st is None:
                raise ValueError("slice_order='protocol' needs a protocol with SliceTiming for every slice (or tr_s and mb)")
            return [float(t) for t in st]
        return list(_core.shot_offsets(int(nz), int(mb), self.slice_order, float(tr_s)))


@dataclass(frozen=True, eq=False)
class Motion:
    """A per-volume rigid motion specification, resolved against a gradient table at
    simulation time (:meth:`poses_for`).

    A **timed** trajectory (P10, ``times`` set) holds samples on the run's clock (volume ``v``
    starts at ``v * Protocol.tr_s``) instead of one row per volume, usually faster than the
    repetition. Each volume's pose is the trace's at its first shot, which the re-simulation uses;
    with ``sampling="shot"`` each shot also takes the trace's pose at its own time, the motion
    beyond its volume's pose resampled per shot (not re-simulated: the fibre-gradient angle follows
    the volume's pose). The trace must cover every time it is sampled at, and is never
    extrapolated."""

    kind: Kind
    #: ``(n, 6)`` ``[tx, ty, tz mm, rx, ry, rz deg]`` for ``"trajectory"``; unused otherwise.
    poses: np.ndarray | None = None
    trans_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    volumes: tuple[int, ...] | None = None
    seed: int = 0
    source: str = ""
    #: P10: ``(n,)`` sample times (s) of a timed trajectory, strictly increasing; ``None`` when
    #: the rows are volumes.
    times: np.ndarray | None = None
    #: P10: ``"volume"`` (one pose per volume) or ``"shot"`` (each shot at its own time).
    sampling: Literal["volume", "shot"] = "volume"

    def __post_init__(self) -> None:
        if self.sampling not in ("volume", "shot"):
            raise ValueError(f"sampling {self.sampling!r}: expected 'volume' or 'shot'")
        if self.times is not None:
            if self.kind != "trajectory" or self.poses is None:
                raise ValueError("times belong to a trajectory")
            t = np.asarray(self.times, dtype=np.float64)
            if t.ndim != 1 or t.size != self.poses.shape[0]:
                raise ValueError(f"times: {t.size} values for {self.poses.shape[0]} poses")
            if t.size < 2 or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
                raise ValueError("times: at least two, finite and strictly increasing")
        if self.sampling == "shot" and self.kind != "trajectory":
            raise ValueError("sampling='shot' needs a trajectory (random poses are impulses, not a path)")

    @property
    def is_timed(self) -> bool:
        return self.times is not None

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
    def timed(cls, times: Any, trans_mm: Any, rot: Any, *, rot_units: Literal["deg", "rad"] = "deg",
                    sampling: Literal["volume", "shot"] = "volume", source: str = "timed") -> "Motion":
        """A timed trajectory (P10) from arrays: ``(n,)`` sample times (s on the run's clock) and
        ``(n, 3)`` translations (mm) and rotations."""
        m = cls.from_arrays(trans_mm, rot, rot_units=rot_units, source=source)
        return dataclasses.replace(m, times=np.ascontiguousarray(np.asarray(times, dtype=np.float64)), sampling=sampling)

    @classmethod
    def from_confounds(cls, path: str | Path, *, sampling: Literal["volume", "shot"] = "volume") -> "Motion":
        """From a qsiprep / fMRIPrep-style confounds TSV: columns ``trans_x/y/z`` (mm) and
        ``rot_x/y/z`` (**radians**), one row per volume; ``n/a`` -> 0. Exactly what the CLI's
        ``--motion`` reads (``motion::load_motion_tsv``).

        P10: a ``time`` column, or a sidecar at the same path with ``.json`` holding
        ``SamplingFrequency`` (and optionally ``StartTime``), makes it a timed trace, read strictly
        (``motion::load_motion_trace``); a sidecar without ``SamplingFrequency`` leaves it as
        above."""
        path = Path(path)
        side = path.with_suffix(".json")
        hz, start = None, 0.0
        if side.is_file():
            import json

            try:
                meta = json.loads(side.read_text())
            except json.JSONDecodeError as e:
                raise ValueError(f"{side}: {e} (it decides whether the trace is timed, so it must be valid JSON)") from e
            if "SamplingFrequency" in meta:
                hz, start = float(meta["SamplingFrequency"]), float(meta.get("StartTime", 0.0))
        with open(path) as f:
            header = f.readline().rstrip("\n").split("\t")
        if hz is not None or "time" in [h.strip() for h in header]:
            times, flat = _core.load_motion_trace(str(path), hz, start)
            poses = np.asarray(flat, dtype=np.float64).reshape(-1, 6)
            return cls(kind="trajectory", poses=np.ascontiguousarray(poses), times=np.asarray(times), sampling=sampling,
                       source=str(path))
        if sampling == "shot":
            m = cls.from_confounds(path)
            return dataclasses.replace(m, sampling="shot")
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
        """The number of volumes of an untimed trajectory (``None`` for a timed one, whose rows are
        samples)."""
        return None if self.poses is None or self.times is not None else int(self.poses.shape[0])

    def _trace_at(self, query: Any) -> np.ndarray:
        """A timed trace's ``(m, 6)`` poses at ``query`` times (``Trace::pose_at``), refused unless
        the trace covers every one."""
        assert self.times is not None and self.poses is not None
        q = np.ascontiguousarray(np.asarray(query, dtype=np.float64).ravel())
        flat = _core.sample_poses(np.asarray(self.times, dtype=np.float64), np.ascontiguousarray(self.poses), q,
                                  f"motion trace {self.source or 'timed'}")
        return np.asarray(flat).reshape(-1, 6)

    def poses_for(self, n_vol: int, *, seed: int | None = None, tr_s: float | None = None,
                  offsets: Any = None) -> np.ndarray:
        """``(n_vol, 6)`` absolute poses for a scheme of ``n_vol`` volumes. A trajectory longer
        than the scheme is truncated; a shorter one raises (use :meth:`resample`).

        A timed trace (P10) gives each volume its pose at its first shot, ``v * tr_s`` plus the
        earliest of ``offsets`` (the slices' excitation offsets; 0 when not given), and needs
        ``tr_s``."""
        if self.times is not None:
            if tr_s is None:
                raise ValueError("a timed motion trace needs tr_s (Protocol.tr_s) to place the volumes on its clock")
            first = float(np.min(offsets)) if offsets is not None and len(offsets) else 0.0
            return self._trace_at(np.arange(n_vol) * float(tr_s) + first)
        if self.kind == "trajectory":
            assert self.poses is not None
            if self.poses.shape[0] < n_vol:
                raise ValueError(f"motion trace has {self.poses.shape[0]} volumes but the scheme has {n_vol}; use Motion.resample(n_vol) to interpolate")
            return np.ascontiguousarray(self.poses[:n_vol])
        vols = list(self.volumes) if self.volumes is not None else None
        flat = _core.resolve_poses(self.kind, n_vol, self.seed if seed is None else seed, tuple(self.trans_mm), tuple(self.rot_deg), vols, None)
        return flat.reshape(n_vol, 6)

    def shot_poses_for(self, n_vol: int, shot_times: Any, *, tr_s: float, offsets: Any = None) -> np.ndarray:
        """P10: ``(n_vol * n_shots, 6)`` poses at each shot's time, volume-major (``shot_times``
        ``(n_vol, n_shots)`` seconds on the run's clock): a timed trace's, refused unless it covers
        them; an untimed trajectory's volume poses placed at each volume's first shot and
        interpolated between them (held past the last)."""
        st = np.asarray(shot_times, dtype=np.float64)
        if self.times is not None:
            return self._trace_at(st)
        nodes = np.arange(n_vol) * float(tr_s) + (float(np.min(offsets)) if offsets is not None and len(offsets) else 0.0)
        flat = _core.sample_poses(nodes, np.ascontiguousarray(self.poses_for(n_vol)), np.ascontiguousarray(st.ravel()), None)
        return np.asarray(flat).reshape(-1, 6)

    def resample(self, n_vol: int) -> "Motion":
        """Linearly interpolate a trajectory to ``n_vol`` volumes over the same scan time."""
        if self.times is not None:
            raise ValueError("a timed motion trace is sampled at the run's own times; resampling it by volume count has no meaning")
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

    def matrices(self, n_vol: int, center: tuple[float, float, float] = (0.0, 0.0, 0.0), *,
                 tr_s: float | None = None, offsets: Any = None) -> np.ndarray:
        """``(n_vol, 4, 4)`` world transforms (``Pose::to_matrix``) about ``center``; a timed trace
        needs ``tr_s`` (its volumes' poses, :meth:`poses_for`)."""
        poses = self.poses_for(n_vol, tr_s=tr_s, offsets=offsets)
        flat = _core.poses_to_matrices(np.ascontiguousarray(poses), tuple(float(c) for c in center))
        return flat.reshape(n_vol, 4, 4)

    def framewise_displacement(self, n_vol: int | None = None, radius_mm: float = 50.0, *,
                               tr_s: float | None = None, offsets: Any = None) -> np.ndarray:
        """Power-style FD per volume: sum of absolute translation and rotation (converted to arc
        length at ``radius_mm``) differences between consecutive volumes; first volume 0. A timed
        trace needs ``n_vol`` and ``tr_s`` (its volumes' poses)."""
        if self.times is not None and n_vol is None:
            raise ValueError("a timed motion trace needs n_vol and tr_s for a per-volume displacement")
        p = self.poses_for(n_vol if n_vol is not None else (self.n_vol or 0), tr_s=tr_s, offsets=offsets)
        d = np.diff(p, axis=0, prepend=p[:1])
        return np.abs(d[:, :3]).sum(axis=1) + np.abs(np.radians(d[:, 3:])).sum(axis=1) * radius_mm

    def plot(self, n_vol: int | None = None, ax: Any = None, *, tr_s: float | None = None, offsets: Any = None):
        """Plot translations and rotations per volume (matplotlib). A timed trace without ``tr_s``
        is drawn against its own times, its samples as they are."""
        import matplotlib.pyplot as plt

        if self.times is not None and tr_s is None:
            x, p, xlabel = np.asarray(self.times), np.asarray(self.poses), "time (s)"
        else:
            p = self.poses_for(n_vol if n_vol is not None else (self.n_vol or 0), tr_s=tr_s, offsets=offsets)
            x, xlabel = np.arange(p.shape[0]), "volume"
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(9, 3))
        for i, lab in enumerate("xyz"):
            ax[0].plot(x, p[:, i], label=f"trans {lab} (mm)")
            ax[1].plot(x, p[:, 3 + i], label=f"rot {lab} (deg)")
        for a in ax:
            a.set_xlabel(xlabel)
            a.legend(fontsize=8)
        ax[0].set_title(self.source or self.kind)
        return ax
