"""Single-voxel simulation in the style of ``dipy.sims.voxel``, with TRXScan's compartment
presets, per-compartment T2, the closed-form ground truth of the voxel, and noise through the
real coil combine."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from . import _core
from ._arrays import as_gtab, b_max_of
from .motion import Motion
from .protocol import Protocol, Tissue, _as_tissue

FiberSpec = Sequence[tuple[Sequence[float], float]]


@dataclass(frozen=True)
class VoxelSignal:
    """Clean signal of one voxel: ``total`` ``(n_vol,)`` and the per-compartment series."""

    total: np.ndarray
    fiber: np.ndarray
    gm: np.ndarray
    csf: np.ndarray
    bvals: np.ndarray
    bvecs: np.ndarray
    te_ms: float | None

    @property
    def compartments(self) -> dict[str, np.ndarray]:
        return {"fiber": self.fiber, "gm": self.gm, "csf": self.csf}


class VoxelTruth(dict):
    """The 27 closed-form scalars of a voxel, as a dict with attribute access
    (``truth.fa``, ``truth.mk``, ``truth.rtop``, ``truth.icvf`` ...)."""

    def __getattr__(self, name: str) -> float:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


@dataclass(frozen=True)
class Voxel:
    """One voxel's tissue mixture: fibres as ``(direction, weight)`` pairs plus WM/GM/CSF
    fractions (normalised) and a tissue preset::

        v = Voxel(fibers=[((1, 0, 0), 0.6), ((0, 1, 0), 0.4)], wm=0.8, gm=0.15, csf=0.05)
        v.signal(gtab, te_ms=88).total        # what dipy.sims.multi_tensor gives, with T2 and presets
        v.truth().fa, v.truth().mk            # what a fit *should* recover
        v.noise(v.signal(gtab).total, 0.02, coils=8, accel=2)
    """

    fibers: FiberSpec = (((1.0, 0.0, 0.0), 1.0),)
    wm: float = 1.0
    gm: float = 0.0
    csf: float = 0.0
    tissue: Tissue | str = "adult"
    kappa: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tissue", _as_tissue(self.tissue))
        tot = self.wm + self.gm + self.csf
        if tot <= 0:
            raise ValueError("fractions must sum to a positive value")
        object.__setattr__(self, "wm", self.wm / tot)
        object.__setattr__(self, "gm", self.gm / tot)
        object.__setattr__(self, "csf", self.csf / tot)
        object.__setattr__(self, "fibers", tuple((tuple(float(x) for x in d), float(w)) for d, w in self.fibers))

    def replace(self, **kw: Any) -> "Voxel":
        return dataclasses.replace(self, **kw)

    # -- API -----------------------------------------------------------------

    def _dirs_weights(self, dirs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        d = np.array([f[0] for f in self.fibers], dtype=np.float64).reshape(-1, 3) if dirs is None else np.asarray(dirs, np.float64).reshape(-1, 3)
        w = np.array([f[1] for f in self.fibers], dtype=np.float64)
        return np.ascontiguousarray(d), w

    def signal(self, gtab: Any, *, te_ms: float | None = None, motion: Motion | None = None, tissue_s0: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> VoxelSignal:
        """The clean per-compartment signal for ``gtab`` (S/S0 per compartment times its
        fraction, exact fibre directions, no hemisphere quantisation; with ``te_ms`` each
        compartment also decays as ``exp(-TE/T2)``). With ``motion``, the fibres are rotated by
        each volume's pose before that volume is evaluated: the b-vector rotation problem in one
        voxel."""
        bvals, bvecs, _, _ = as_gtab(gtab)
        b_max = b_max_of(bvals)
        params = self.tissue.to_params(b_max)
        d, w = self._dirs_weights()
        if motion is None:
            fib, gm, csf = (np.asarray(v, dtype=np.float64) for v in _core.voxel_signal(bvals, bvecs, d, w, self.wm, self.gm, self.csf, params))
        else:
            mats = motion.matrices(bvals.size)
            fib, gm, csf = (np.zeros(bvals.size) for _ in range(3))
            for g in range(bvals.size):
                R = mats[g, :3, :3]
                a, b, c = _core.voxel_signal(bvals, bvecs, np.ascontiguousarray(d @ R.T), w, self.wm, self.gm, self.csf, params)
                fib[g], gm[g], csf[g] = float(a[g]), float(b[g]), float(c[g])
        t2 = self.tissue.t2_ms
        if te_ms is not None:
            fib = fib * np.exp(-te_ms / t2[0])
            gm = gm * np.exp(-te_ms / t2[1])
            csf = csf * np.exp(-te_ms / t2[2])
        fib, gm, csf = fib * tissue_s0[0], gm * tissue_s0[1], csf * tissue_s0[2]
        return VoxelSignal(total=fib + gm + csf, fiber=fib, gm=gm, csf=csf, bvals=bvals, bvecs=bvecs, te_ms=te_ms)

    def truth(self, *, big_delta: float | None = None, small_delta: float | None = None, tau: float | None = None, b_max: float = 1000.0) -> VoxelTruth:
        """The 27 closed-form microstructure scalars of this voxel (FORCE; dipy-validated),
        with exact fibre directions. ``big_delta``/``small_delta`` (s) put the MAP-MRI scalars
        in physical units."""
        if tau is None and big_delta is not None and small_delta is not None:
            tau = big_delta - small_delta / 3.0
        d, w = self._dirs_weights()
        vals = _core.voxel_truth(d, w, self.wm, self.gm, self.csf, self.tissue.to_params(b_max), tau, None, None, None)
        return VoxelTruth({name: float(v) for name, v in zip(_core.scalar_names(), vals)})

    @staticmethod
    def noise(signal: np.ndarray, sigma: float, *, coils: int = 1, accel: int = 1, seed: int = 0, matrix: int = 16) -> np.ndarray:
        """Magnitude noise through the simulator's own reception chain: the clean ``signal``
        fills a uniform disc on a small ``matrix``, k-space noise of per-component variance
        ``sigma**2`` is added per coil, GRAPPA (``accel``) and the Roemer combine run, and the
        centre voxel's magnitude series is returned. Rician at one coil, non-central chi with
        the GRAPPA g-factor at more."""
        from .objects import disc

        s = np.asarray(signal, dtype=np.float64).reshape(-1)
        n = s.size
        obj = disc(radius_vox=matrix * 0.35, matrix=matrix, oversample=1, fibers=None)
        nvox = int(np.prod(obj.sim_dims))
        img = np.repeat(obj.sim_wm.astype(np.float32)[:, None], n, axis=1) * s.astype(np.float32)[None, :]
        comp = _core.Compartments.from_images(obj.sim_dims, n, [np.ascontiguousarray(img.reshape(-1))], [1e9])
        bvals = np.zeros(n)
        bvecs = np.zeros((n, 3))
        acq = {**Protocol.DEFAULT.acquisition(obj.dims[1]), "noise_variance": float(sigma) ** 2, "n_coils": int(coils), "accel": int(accel), "acs_lines": 8, "do_relaxation": False, "do_distortions": False, "signal_scale": 1.0, "seed": int(seed)}
        out = _core.simulate_acquisition(comp, np.zeros(nvox, np.float32), obj.dims, bvals, bvecs, acq, "none", int(seed), None, None, None, None, None, (False, False, False), None)
        nx, ny, _ = obj.dims
        centre = (nx // 2) + nx * (ny // 2)
        return np.asarray(out["mag"]).reshape(nvox, n)[centre]
