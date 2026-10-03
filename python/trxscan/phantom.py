"""The object being scanned: :class:`Phantom` (real anatomy: tissue maps, streamlines,
fieldmap, measured motion) and :class:`Object` (the phantom, or a synthetic scene, gridded for
one protocol: the thing :meth:`Object.simulate` actually runs on)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import nibabel as nib
import numpy as np

from . import _core
from ._arrays import affine4, flat_f32, nifti, volume
from .motion import Motion
from .protocol import Protocol, Tissue

if TYPE_CHECKING:  # pragma: no cover
    from .artifacts import Artifacts, Gre
    from .simulate import Simulation


@dataclass(frozen=True, eq=False)
class Streamlines:
    """A tractogram as CSR arrays: ``positions`` ``(n, 3)`` world RAS mm, ``offsets``
    ``(n_streamlines + 1,)`` uint32, optional per-streamline ``weights`` (SIFT2)."""

    positions: np.ndarray
    offsets: np.ndarray
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", np.ascontiguousarray(self.positions, dtype=np.float64))
        object.__setattr__(self, "offsets", np.ascontiguousarray(self.offsets, dtype=np.uint32))
        if self.weights is not None:
            object.__setattr__(self, "weights", np.ascontiguousarray(self.weights, dtype=np.float32))
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must be (n, 3)")
        if self.offsets.ndim != 1 or self.offsets.size == 0 or int(self.offsets[-1]) != self.positions.shape[0]:
            raise ValueError("offsets must be CSR with n_streamlines + 1 entries ending at len(positions)")
        if self.weights is not None and self.weights.shape != (self.n,):
            raise ValueError("weights must have one entry per streamline")

    @property
    def n(self) -> int:
        return int(self.offsets.size - 1)

    @classmethod
    def load(cls, path: str | Path, weights: str | None = "sift2_weights") -> "Streamlines":
        """Load a TRX (via ``trx-python``), or TRK/TCK/VTK (via nibabel / dipy) tractogram.
        For TRX, ``weights`` names a per-streamline field to use as SIFT2 weights (ignored if
        absent)."""
        path = Path(path)
        if path.suffix == ".trx" or path.is_dir():
            try:
                from trx.trx_file_memmap import load as trx_load
            except ImportError as e:  # pragma: no cover
                raise ImportError("loading .trx needs trx-python: pip install trxscan[trx]") from e
            t = trx_load(str(path))
            try:
                # trx-python hands out memory maps: copy everything before closing the file, or
                # an array whose dtype already matches would be left pointing at unmapped memory
                pos = np.array(t.streamlines._data, dtype=np.float64, copy=True)
                off = np.array(t.streamlines._offsets, dtype=np.int64, copy=True)
                if off.size == 0 or off[-1] != pos.shape[0]:
                    off = np.append(off, pos.shape[0])
                w = None
                if weights and weights in t.data_per_streamline:
                    w = np.array(t.data_per_streamline[weights], dtype=np.float32, copy=True).reshape(-1)
            finally:
                t.close()
            return cls(pos, off, w)
        # nibabel handles trk/tck; positions come out in RAS mm
        tg = nib.streamlines.load(str(path))
        sl = tg.streamlines
        pos = np.asarray(sl.get_data(), dtype=np.float64)
        off = np.asarray(sl._offsets, dtype=np.int64)
        off = np.append(off, pos.shape[0])
        return cls(pos, off, None)

    def subsample(self, n: int, *, seed: int = 0) -> tuple["Streamlines", dict[str, Any]]:
        pos, off, w, stats = _core.subsample_streamlines(self.positions, self.offsets, self.weights, int(n), int(seed))
        return Streamlines(pos.reshape(-1, 3), off, w), stats

    def crop_k(self, affine: np.ndarray, k0: float, k1: float) -> "Streamlines":
        """Keep streamlines with at least one point whose continuous voxel ``k`` coordinate under
        ``affine`` lies in ``[k0, k1)``. Weights are kept, not renormalised."""
        inv = np.linalg.inv(affine4(affine))
        k = self.positions @ inv[2, :3] + inv[2, 3]
        inside = (k >= k0) & (k < k1)
        starts = self.offsets[:-1].astype(np.int64)
        counts = np.diff(self.offsets.astype(np.int64))
        keep_stream = np.zeros(self.n, dtype=bool)
        nz = np.flatnonzero(inside)
        if nz.size:
            owner = np.searchsorted(starts, nz, side="right") - 1
            keep_stream[np.unique(owner)] = True
        if keep_stream.all():
            return self
        idx = np.flatnonzero(keep_stream)
        if idx.size == 0:
            return Streamlines(np.zeros((0, 3)), np.zeros(1, dtype=np.uint32), None if self.weights is None else np.zeros(0, np.float32))
        pieces = [self.positions[starts[i] : starts[i] + counts[i]] for i in idx]
        pos = np.concatenate(pieces)
        off = np.concatenate([[0], np.cumsum(counts[idx])]).astype(np.uint32)
        w = None if self.weights is None else self.weights[idx]
        return Streamlines(pos, off, w)


@dataclass(frozen=True, eq=False)
class Fibers:
    """Explicit fibre orientations on the simulation grid: entry ``i`` deposits ``weight[i]``
    of orientation mass along ``dirs[i]`` in flat sim-grid voxel ``voxel[i]``."""

    voxel: np.ndarray
    dirs: np.ndarray
    weight: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "voxel", np.ascontiguousarray(self.voxel, dtype=np.uint32).reshape(-1))
        object.__setattr__(self, "dirs", np.ascontiguousarray(self.dirs, dtype=np.float64).reshape(-1, 3))
        object.__setattr__(self, "weight", np.ascontiguousarray(self.weight, dtype=np.float64).reshape(-1))
        if not (self.voxel.size == self.dirs.shape[0] == self.weight.size):
            raise ValueError("voxel, dirs and weight must have the same length")


# ─── Object ─────────────────────────────────────────────────────────────────


def _sim_affine(affine: np.ndarray, dims: tuple[int, int, int], o: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    a, d = _core.hires_grid(affine4(affine), tuple(int(v) for v in dims), int(o))
    return a.reshape(4, 4), tuple(int(v) for v in d)


@dataclass(frozen=True, eq=False)
class Object:
    """A gridded object: compartment fractions on the acquisition grid and on the ``oversample``
    times finer simulation grid, the fieldmap (Hz), and its fibre content as streamlines or an
    explicit fibre list. Built by :meth:`Phantom.grid` or the synthetic constructors in
    :mod:`trxscan.objects`. Arrays are flat float32 in TRXScan's F-order (see
    :mod:`trxscan._arrays`); use :meth:`image` for NIfTI views.
    """

    dims: tuple[int, int, int]
    affine: np.ndarray
    oversample: int
    sim_dims: tuple[int, int, int]
    sim_affine: np.ndarray
    wm: np.ndarray
    gm: np.ndarray
    csf: np.ndarray
    mask: np.ndarray
    sim_wm: np.ndarray
    sim_gm: np.ndarray
    sim_csf: np.ndarray
    sim_mask: np.ndarray
    sim_fmap: np.ndarray
    fmap: np.ndarray | None = None
    streamlines: Streamlines | None = None
    fibers: Fibers | None = None
    myelin: np.ndarray | None = None
    sim_head: np.ndarray | None = None
    z_offset: int = 0
    nz_full: int | None = None
    voxel_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "affine", affine4(self.affine))
        object.__setattr__(self, "sim_affine", affine4(self.sim_affine))
        object.__setattr__(self, "dims", tuple(int(v) for v in self.dims))
        object.__setattr__(self, "sim_dims", tuple(int(v) for v in self.sim_dims))
        n, ns = int(np.prod(self.dims)), int(np.prod(self.sim_dims))
        for name in ("wm", "gm", "csf", "mask"):
            v = np.ascontiguousarray(getattr(self, name), dtype=np.float32).reshape(-1)
            if v.size != n:
                raise ValueError(f"{name}: expected {n} voxels for dims {self.dims}, got {v.size}")
            object.__setattr__(self, name, v)
        for name in ("sim_wm", "sim_gm", "sim_csf", "sim_mask", "sim_fmap"):
            v = np.ascontiguousarray(getattr(self, name), dtype=np.float32).reshape(-1)
            if v.size != ns:
                raise ValueError(f"{name}: expected {ns} voxels for sim_dims {self.sim_dims}, got {v.size}")
            object.__setattr__(self, name, v)
        if self.fmap is not None:
            object.__setattr__(self, "fmap", np.ascontiguousarray(self.fmap, dtype=np.float32).reshape(-1))
        if self.myelin is not None:
            m = np.ascontiguousarray(self.myelin, dtype=np.float32).reshape(-1)
            if m.size != ns:
                raise ValueError("myelin must be on the simulation grid")
            object.__setattr__(self, "myelin", m)
        if self.sim_head is not None:
            h = np.ascontiguousarray(self.sim_head, dtype=np.float32).reshape(-1)
            if h.size != ns:
                raise ValueError("sim_head must be on the simulation grid")
            object.__setattr__(self, "sim_head", h)
        if self.nz_full is None:
            object.__setattr__(self, "nz_full", self.dims[2])
        o = self.oversample
        if self.sim_dims != (self.dims[0] * o, self.dims[1] * o, self.dims[2]):
            raise ValueError(f"sim_dims {self.sim_dims} is not {o}x the acquisition grid {self.dims} in-plane")

    # -- geometry ------------------------------------------------------------

    @property
    def nz(self) -> int:
        return self.dims[2]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.dims

    def z_of(self, z_mm: float) -> int:
        """The local slice index whose centre is nearest world ``z_mm`` (third RAS axis)."""
        inv = np.linalg.inv(self.affine)
        centre_xy = self.affine @ np.array([(self.dims[0] - 1) / 2, (self.dims[1] - 1) / 2, 0.0, 1.0])
        p = np.array([centre_xy[0], centre_xy[1], z_mm, 1.0])
        k = (inv @ p)[2]
        return int(np.clip(np.rint(k), 0, self.dims[2] - 1))

    def slab(self, z0: int, z1: int) -> "Object":
        """Local slices ``z0:z1`` as a new Object (streamlines untouched; fibres filtered)."""
        z0, z1 = int(z0), int(z1)
        if not 0 <= z0 < z1 <= self.dims[2]:
            raise ValueError(f"slab {z0}:{z1} out of range for {self.dims[2]} slices")
        if (z0, z1) == (0, self.dims[2]):
            return self
        nx, ny, nz = self.dims
        snx, sny, _ = self.sim_dims
        per, sper = nx * ny, snx * sny
        cut = lambda v: v[z0 * per : z1 * per]  # noqa: E731
        scut = lambda v: v[z0 * sper : z1 * sper]  # noqa: E731
        aff = self.affine.copy()
        aff[:3, 3] += self.affine[:3, 2] * z0
        saff = self.sim_affine.copy()
        saff[:3, 3] += self.sim_affine[:3, 2] * z0
        fibers = None
        if self.fibers is not None:
            zf = self.fibers.voxel // sper
            keep = (zf >= z0) & (zf < z1)
            fibers = Fibers(self.fibers.voxel[keep] - z0 * sper, self.fibers.dirs[keep], self.fibers.weight[keep])
        return dataclasses.replace(
            self, dims=(nx, ny, z1 - z0), sim_dims=(snx, sny, z1 - z0), affine=aff, sim_affine=saff,
            wm=cut(self.wm), gm=cut(self.gm), csf=cut(self.csf), mask=cut(self.mask),
            sim_wm=scut(self.sim_wm), sim_gm=scut(self.sim_gm), sim_csf=scut(self.sim_csf), sim_mask=scut(self.sim_mask),
            sim_fmap=scut(self.sim_fmap), fmap=None if self.fmap is None else cut(self.fmap),
            myelin=None if self.myelin is None else scut(self.myelin), sim_head=None if self.sim_head is None else scut(self.sim_head),
            fibers=fibers, z_offset=self.z_offset + z0,
        )

    def image(self, name: str) -> nib.Nifti1Image:
        """A NIfTI view of ``wm``/``gm``/``csf``/``mask``/``fmap`` (acquisition grid) or
        ``sim_wm``/... /``sim_fmap``/``myelin`` (simulation grid)."""
        if name.startswith("sim_") or name == "myelin":
            v = getattr(self, name)
            if v is None:
                raise ValueError(f"{name} is not set")
            return nifti(volume(v, self.sim_dims), self.sim_affine)
        v = getattr(self, name)
        if v is None:
            raise ValueError(f"{name} is not set")
        return nifti(volume(v, self.dims), self.affine)

    def with_myelin(self, myelin: Any) -> "Object":
        """Attach a per-voxel myelination map (0..1) on the simulation grid (lerps WM toward the
        adult endpoint in signal and truth)."""
        m = flat_f32(myelin) if hasattr(myelin, "dataobj") or np.ndim(myelin) == 3 else np.asarray(myelin, np.float32).reshape(-1)
        return dataclasses.replace(self, myelin=m)

    # -- simulation ----------------------------------------------------------

    def simulate(self, gtab: Any, protocol: Protocol = Protocol.DEFAULT, artifacts: "Artifacts | None" = None, **kw: Any) -> "Simulation":
        """Simulate this object under ``gtab`` (a dipy ``GradientTable`` or ``(bvals, bvecs)``),
        ``protocol`` and ``artifacts``. See :func:`trxscan.simulate.run` for the keyword
        options (``slices=``, ``context=``, ``kspace=``, ``gre=``, ``truth_peaks=``,
        ``progress=``)."""
        from .artifacts import Artifacts
        from .simulate import run

        return run(self, gtab, protocol, Artifacts() if artifacts is None else artifacts, **kw)

    def microstructure(self, protocol: Protocol = Protocol.DEFAULT, gtab: Any = None, **kw: Any) -> dict[str, nib.Nifti1Image]:
        """The 27 closed-form ground-truth maps of this object on the acquisition grid. See
        :func:`trxscan.simulate.microstructure`."""
        from .simulate import microstructure

        return microstructure(self, protocol, gtab, **kw)


# ─── Phantom ────────────────────────────────────────────────────────────────


def _load_img(x: Any) -> nib.Nifti1Image:
    if isinstance(x, (str, Path)):
        return nib.load(str(x))
    return x


@dataclass(frozen=True, eq=False)
class Phantom:
    """Real anatomy to simulate: WM/GM/CSF fraction maps and a brain mask on the anatomical
    grid, a tractogram in the same world (RAS mm) frame, an off-resonance fieldmap (Hz), the
    subject's measured head-motion traces (``motion["AP"]``, a :class:`Motion`) and,
    optionally, its T1w/T2w images for the ``anat/`` folder of a simulated dataset.

    Build with :meth:`load` (the hosted phantoms), :meth:`from_files`, or directly from
    images and a :class:`Streamlines`. Grid it for a protocol with :meth:`grid`, or simply call
    :meth:`simulate`.
    """

    wm: nib.Nifti1Image
    gm: nib.Nifti1Image
    csf: nib.Nifti1Image
    streamlines: Streamlines
    mask: nib.Nifti1Image | None = None
    fieldmap: nib.Nifti1Image | None = None
    motion: dict[str, Motion] = field(default_factory=dict)
    #: The subject's anatomical images in the same world frame (written to BIDS ``anat/`` by
    #: :class:`~trxscan.bids.Dataset`; nothing in the simulation reads them).
    t1w: nib.Nifti1Image | None = None
    t2w: nib.Nifti1Image | None = None
    name: str = ""
    path: Path | None = None
    _grids: dict[Any, Object] = field(default_factory=dict, repr=False)

    # -- constructors --------------------------------------------------------

    @classmethod
    def load(cls, name: str = "sub-60501", **kw: Any) -> "Phantom":
        """A hosted phantom, downloaded once into the pooch cache (or read from
        ``$TRXSCAN_DATA/<name>``). See :mod:`trxscan.data`."""
        from .data import load_phantom

        return load_phantom(name, **kw)

    @classmethod
    def from_files(
        cls, *, wm: Any, gm: Any, csf: Any, streamlines: Any, mask: Any = None, fieldmap: Any = None,
        weights: str | np.ndarray | None = "sift2_weights", motion: dict[str, Any] | None = None,
        t1w: Any = None, t2w: Any = None, name: str = "",
    ) -> "Phantom":
        """From NIfTI paths/images and a tractogram path (TRX/TRK/TCK) or :class:`Streamlines`.
        ``weights`` is a TRX per-streamline field name or an explicit array; ``motion`` maps
        labels to confounds TSV paths or :class:`Motion` objects; ``t1w``/``t2w`` are the
        subject's anatomical images (same world frame)."""
        if isinstance(streamlines, Streamlines):
            sl = streamlines
            if isinstance(weights, np.ndarray):
                sl = Streamlines(sl.positions, sl.offsets, weights)
        else:
            sl = Streamlines.load(streamlines, weights if isinstance(weights, str) else None)
            if isinstance(weights, np.ndarray):
                sl = Streamlines(sl.positions, sl.offsets, weights)
        mot: dict[str, Motion] = {}
        for k, v in (motion or {}).items():
            mot[k] = v if isinstance(v, Motion) else Motion.from_confounds(v)
        return cls(
            wm=_load_img(wm), gm=_load_img(gm), csf=_load_img(csf), streamlines=sl,
            mask=None if mask is None else _load_img(mask), fieldmap=None if fieldmap is None else _load_img(fieldmap),
            motion=mot, t1w=None if t1w is None else _load_img(t1w), t2w=None if t2w is None else _load_img(t2w), name=name,
        )

    # -- derived -------------------------------------------------------------

    @property
    def n_streamlines(self) -> int:
        return self.streamlines.n

    @property
    def anatomical_affine(self) -> np.ndarray:
        return np.asarray(self.wm.affine, dtype=np.float64)

    def moved(self, T: Any, name: str | None = None) -> "Phantom":
        """The same subject after a rigid movement ``T`` (4x4 world RAS, or
        ``(tx, ty, tz, rx, ry, rz)`` in mm and degrees about the anatomy's centre): every
        image's affine is pre-multiplied by ``T`` and the streamlines are moved with it, so the
        head (and its field) sits at ``T·x``. Grid it ``like=`` another run's object to put the
        moved head inside that run's fixed field of view."""
        from .bids import _offset_of, offset_image

        T = _offset_of(T, self.wm)
        if T is None or np.allclose(T, np.eye(4)):
            return self
        pos = self.streamlines.positions @ T[:3, :3].T + T[:3, 3]
        sl = Streamlines(pos, self.streamlines.offsets, self.streamlines.weights)
        mv = lambda img: None if img is None else offset_image(img, T)  # noqa: E731
        return dataclasses.replace(
            self, wm=mv(self.wm), gm=mv(self.gm), csf=mv(self.csf), mask=mv(self.mask), fieldmap=mv(self.fieldmap),
            t1w=mv(self.t1w), t2w=mv(self.t2w), streamlines=sl, name=name if name is not None else (f"{self.name}+moved" if self.name else ""), _grids={},
        )

    def subsample(self, n: int, *, seed: int = 0) -> "Phantom":
        """Keep ``n`` streamlines sampled proportionally to weight (deterministic in ``seed``);
        survivors get uniform weights. The only place subsampling happens, so every
        simulation and every truth map from the returned phantom describe the same subset."""
        sl, _ = self.streamlines.subsample(n, seed=seed)
        return dataclasses.replace(self, streamlines=sl, name=f"{self.name}[{n}@{seed}]" if self.name else "", _grids={})

    def grid(
        self, protocol: Protocol | None = None, *, voxel_mm: Any = None, oversample: int | None = None, pad: Any = None,
        matrix: Any = None, like: "Object | None" = None,
    ) -> Object:
        """The phantom resampled onto the acquisition grid (and the finer simulation grid) a
        protocol implies: its own bounding box plus ``pad``, centred in a fixed ``matrix``
        (``Protocol.matrix``), or exactly the grid of another run's object (``like``). Cached
        per ``(voxel_mm, oversample, pad, matrix, like)``."""
        from ._grid import grid_phantom

        p = protocol or Protocol.DEFAULT
        vox = p.voxel if voxel_mm is None else voxel_mm
        vox = (float(vox),) * 3 if np.isscalar(vox) else tuple(float(v) for v in vox)
        o = int(p.oversample if oversample is None else oversample)
        pd = tuple(int(v) for v in (p.pad if pad is None else pad))
        mt = p.matrix if matrix is None else tuple(int(v) for v in matrix)
        target = None if like is None else (tuple(like.dims), np.asarray(like.affine))
        key = (vox, o, pd, mt, None if like is None else (target[0], tuple(np.round(target[1].ravel(), 9))), p.oblique_deg)
        obj = self._grids.get(key)
        if obj is None:
            obj = grid_phantom(self, vox, o, pd, mt, target, p.oblique_deg)
            self._grids[key] = obj
        return obj

    def simulate(self, gtab: Any, protocol: Protocol = Protocol.DEFAULT, artifacts: "Artifacts | None" = None, **kw: Any) -> "Simulation":
        """``self.grid(protocol).simulate(gtab, protocol, artifacts, **kw)``."""
        return self.grid(protocol).simulate(gtab, protocol, artifacts, **kw)

    def microstructure(self, protocol: Protocol = Protocol.DEFAULT, gtab: Any = None, **kw: Any) -> dict[str, nib.Nifti1Image]:
        """``self.grid(protocol).microstructure(protocol, gtab, **kw)``."""
        return self.grid(protocol).microstructure(protocol, gtab, **kw)
