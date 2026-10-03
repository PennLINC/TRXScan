"""What goes wrong in an acquisition: :class:`Artifacts`, everything off by default."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .motion import Motion

Window = str | tuple[str, float] | tuple[str, float, float] | None


@dataclass(frozen=True)
class Gre:
    """Synthetic dual-echo GRE fieldmap of the same object (Siemens conventions), requested
    with ``Object.simulate(..., gre=Gre())``.

    The magnitudes are spoiled-GRE steady states: per compartment
    ``sin(flip) (1 - E1) / (1 - cos(flip) E1)``, ``E1 = exp(-tr_s / T1)``, times the T2 decay at
    each echo, so with the defaults (TR 0.5 s, FA 60 deg, adult 3 T T1 of 830/1330/4000 ms for
    fiber/GM/CSF) the image carries the WM > GM >> CSF contrast of a real fieldmap magnitude.
    ``tr_s=float("inf")`` with ``flip_deg=90`` gives pure proton density (the tissue-fraction
    sum). ``snr`` is the SNR of the brightest compartment.

    Realism of a scanner fieldmap magnitude (modelled on a Prisma ``B0map``, TR 0.52 s, FA 45):
    ``pd`` proton densities flatten the brain to its real near-uniform look; ``bias`` is the
    periphery-bright receive profile of a head array (centre ``1-bias``, corners ``1+bias``);
    ``ringing`` reconstructs the acquisition matrix by Fourier truncation of the fine grid
    (Gibbs ringing at edges) instead of box averaging; ``head`` adds the non-brain head
    (scalp, skull, neck) from the phantom's T1w outside the brain, ``head_level`` times as
    bright as WM, with ``t2_head_ms``. A phantom without a T1w (or a synthetic object) has no
    head."""

    te_s: tuple[float, float] = (4.92e-3, 7.38e-3)
    snr: float = 50.0
    res_mm: float | None = None
    snr_vol_exp: float = 1.0
    output: Literal["phasediff", "phase"] = "phasediff"
    rx_phase_rad: float = 6.0
    b0_field: str = "b0gre"
    tr_s: float = 0.5
    flip_deg: float = 60.0
    t1_ms: tuple[float, float, float] = (830.0, 1330.0, 4000.0)
    pd: tuple[float, float, float] = (0.7, 0.85, 1.0)
    bias: float = 0.3
    ringing: bool = True
    head: bool = True
    head_level: float = 1.0
    t2_head_ms: float = 70.0

    def replace(self, **kw: Any) -> "Gre":
        return dataclasses.replace(self, **kw)


@dataclass(frozen=True, eq=False)
class Artifacts:
    """Artifact specification. ``Artifacts()`` is the clean reference; every field is off by
    default. Immutable; vary with :meth:`replace`.

    k-space stage: ``noise`` (complex k-space noise variance -> Rician magnitude noise),
    ``noise_map`` (per-voxel noise SD image on the acquisition grid; writes the truth sigma),
    ``eddy``/``eddy_quad`` (linear/quadratic eddy-current geometric distortion), ``eddy_phase``
    (eddy phase ramp in the complex output), ``ghost`` (Nyquist ghost kx offset), ``spikes`` +
    ``spike_amplitude`` (random k-space spikes per slice), ``window`` (reconstruction
    apodization: ``"hann"``, ``("tukey", alpha)``, ``("fermi", radius, width)``),
    ``distortion``/``relaxation`` (fieldmap and T2 effects on/off).

    Signal stage: ``motion`` (a :class:`~trxscan.motion.Motion`; the faithful per-volume
    re-simulation), ``dropout`` (within-volume multiband dropout probability per DWI volume;
    needs ``Protocol.mb > 1``), ``gnl`` (gradient nonlinearity: ``"whole-body-80"``,
    ``"connectom-300"``, the path of a Siemens ``.grad`` file, or its text) with
    ``gnl_scale``/``gnl_warp``/``gnl_encoding``/``gnl_jacobian`` and ``isocenter`` (world mm).

    ``te_per_volume`` overrides the protocol's TE per volume; ``seed`` fixes the noise, spike,
    dropout and shot-phase realisation.
    """

    noise: float = 0.0
    noise_map: Any = None
    eddy: float = 0.0
    eddy_quad: float = 0.0
    eddy_phase: float = 0.0
    eddy_tau_ms: float = 70.0
    ghost: float = 0.0
    spikes: int = 0
    spike_amplitude: float = 1.0
    window: Window = None
    distortion: bool = True
    relaxation: bool = True
    motion: Motion | None = None
    dropout: float = 0.0
    gnl: str | None = None
    gnl_scale: float = 1.0
    gnl_warp: bool = True
    gnl_encoding: bool = True
    gnl_jacobian: bool = True
    isocenter: tuple[float, float, float] | None = None
    te_per_volume: Any = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.motion is not None and not isinstance(self.motion, Motion):
            raise TypeError("motion must be a trxscan.Motion (see Motion.from_confounds, Motion.random, ...)")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout is a probability in [0, 1]")
        if self.window is not None and not isinstance(self.window, (str, tuple)):
            raise TypeError("window must be None, 'hann', ('tukey', a) or ('fermi', r, w)")

    def replace(self, **kw: Any) -> "Artifacts":
        return dataclasses.replace(self, **kw)

    @property
    def clean(self) -> bool:
        """True when nothing is switched on (the reference acquisition)."""
        return self == Artifacts()

    def __eq__(self, other: object) -> bool:  # arrays make the generated __eq__ unusable
        if not isinstance(other, Artifacts):
            return NotImplemented
        a, b = dataclasses.asdict(self), dataclasses.asdict(other)
        for k in ("noise_map", "te_per_volume", "motion"):
            x, y = a.pop(k), b.pop(k)
            if (x is None) != (y is None):
                return False
            if x is not None and not np.array_equal(np.asarray(x, dtype=object), np.asarray(y, dtype=object)):
                return False
        return a == b

    def acquisition(self) -> dict[str, Any]:
        """The ``kspace::Acquisition`` fields this spec sets."""
        d: dict[str, Any] = {
            "noise_variance": float(self.noise),
            "eddy_strength": float(self.eddy),
            "eddy_quad": float(self.eddy_quad),
            "eddy_phase": float(self.eddy_phase),
            "eddy_tau": float(self.eddy_tau_ms),
            "ghost_offset": float(self.ghost),
            "n_spikes": int(self.spikes),
            "spike_amplitude": float(self.spike_amplitude),
            "do_distortions": bool(self.distortion),
            "do_relaxation": bool(self.relaxation),
            "seed": int(self.seed),
        }
        if self.window is not None:
            d["window"] = self.window
        return d

    @property
    def needs_context(self) -> bool:
        """Whether this spec pulls signal from neighbouring slices (motion, dropout jumps, GNL
        warp), so a slice simulation needs a slab of context around it."""
        return self.motion is not None or self.dropout > 0.0 or (self.gnl is not None and self.gnl_warp)
