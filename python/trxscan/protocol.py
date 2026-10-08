"""Scanner settings (:class:`Protocol`), tissue presets (:class:`Tissue`) and the EPI readout
timing they imply (:class:`EpiReadout`). All immutable; presets are instances."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from . import _core

# ─── tissue ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Tissue:
    """Compartment model parameters (the simulator's ``CompartmentParams`` minus ``b_value``,
    which comes from the gradient table). Build from a preset and override fields::

        Tissue.ADULT.replace(t2_fiber=75.0)

    Presets: ``Tissue.NEONATAL`` (Fiberfox legacy), ``Tissue.ADULT`` (3 T literature values),
    ``Tissue.INFANT`` (unmyelinated white matter). Diffusivities are in mm^2/s, T2s in ms.
    """

    name: str
    fiber_radius_mm: float
    intra_frac: float
    extra_frac: float
    d_intra: float
    d_extra: tuple[float, float, float]
    d_gm: float
    d_csf: float
    t2_fiber: float
    t2_gm: float
    t2_csf: float
    gm_restricted_frac: float
    d_soma: float

    NEONATAL: ClassVar["Tissue"]
    ADULT: ClassVar["Tissue"]
    INFANT: ClassVar["Tissue"]

    @classmethod
    def from_preset(cls, name: str) -> "Tissue":
        d = _core.tissue_preset(name)
        d.pop("b_value", None)
        return cls(name=name, **d)

    def replace(self, **kw: Any) -> "Tissue":
        if kw and "name" not in kw:
            kw["name"] = f"{self.name}*"
        return dataclasses.replace(self, **kw)

    @property
    def t2_ms(self) -> tuple[float, float, float]:
        """Per-compartment T2 ``(fiber, gm, csf)`` in ms."""
        return (self.t2_fiber, self.t2_gm, self.t2_csf)

    def to_params(self, b_max: float, diff_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> dict[str, Any]:
        """The dict the Rust core takes; ``diff_scale`` mirrors the CLI's ``--diff-scale wm,gm,csf``."""
        sw, sg, sc = diff_scale
        d = dataclasses.asdict(self)
        d.pop("name")
        d["b_value"] = float(b_max)
        d["d_intra"] = self.d_intra * sw
        d["d_extra"] = tuple(v * sw for v in self.d_extra)
        d["d_gm"] = self.d_gm * sg
        d["d_soma"] = self.d_soma * sg
        d["d_csf"] = self.d_csf * sc
        return d


Tissue.NEONATAL = Tissue.from_preset("neonatal")
Tissue.ADULT = Tissue.from_preset("adult")
Tissue.INFANT = Tissue.from_preset("infant")


def _as_tissue(t: Any) -> Tissue:
    if isinstance(t, Tissue):
        return t
    if isinstance(t, str):
        return Tissue.from_preset(t)
    raise TypeError(f"tissue must be a Tissue or a preset name, got {type(t).__name__}")


# ─── readout timing ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EpiReadout:
    """The single-shot EPI readout an acquisition implies on an ``nx`` x ``ny`` matrix: the
    simulator's own per-line timing, for annotating k-space and pulse-sequence figures."""

    nx: int
    ny: int
    #: ky index of each line in acquisition order (``ny`` entries).
    ky_order: np.ndarray
    #: +1/-1 readout direction per line in acquisition order.
    kx_direction: np.ndarray
    #: Per ky line (indexed by ky): ms from the echo (k-space centre), since RF, since readout start.
    t_ms: np.ndarray
    t_rf_ms: np.ndarray
    t_read_ms: np.ndarray
    #: Every sample: ``(kx, ky, t_ms)`` in acquisition order, ``(nx*ny, 3)``.
    ticks: np.ndarray
    #: Which samples are acquired, ``(ny, nx)`` bool.
    mask: np.ndarray
    t_line_ms: float
    t_echo_ms: float
    dt_ms: float

    @property
    def acquired_lines(self) -> np.ndarray:
        """ky indices actually acquired, in acquisition order."""
        return np.array([ky for ky in self.ky_order if self.mask[ky, 0]])

    @property
    def total_readout_ms(self) -> float:
        """First to last acquired line centre (what BIDS TotalReadoutTime approximates)."""
        lines = self.acquired_lines
        return float(self.t_read_ms[lines[-1]] - self.t_read_ms[lines[0]]) if lines.size else 0.0

    @property
    def time_to_center_ms(self) -> float:
        """From the first acquired line to ky = ny // 2 (sets the minimum TE contribution)."""
        lines = self.acquired_lines
        return float(self.t_read_ms[self.ny // 2] - self.t_read_ms[lines[0]]) if lines.size else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "nx": self.nx, "ny": self.ny, "ky_order": self.ky_order.tolist(), "kx_direction": self.kx_direction.tolist(),
            "t_ms": self.t_ms.tolist(), "t_rf_ms": self.t_rf_ms.tolist(), "t_read_ms": self.t_read_ms.tolist(),
            "ticks": self.ticks.tolist(), "mask": self.mask.astype(int).tolist(),
            "t_line_ms": self.t_line_ms, "t_echo_ms": self.t_echo_ms, "dt_ms": self.dt_ms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EpiReadout":
        return cls(
            nx=d["nx"], ny=d["ny"], ky_order=np.asarray(d["ky_order"]), kx_direction=np.asarray(d["kx_direction"]),
            t_ms=np.asarray(d["t_ms"]), t_rf_ms=np.asarray(d["t_rf_ms"]), t_read_ms=np.asarray(d["t_read_ms"]),
            ticks=np.asarray(d["ticks"]), mask=np.asarray(d["mask"], dtype=bool),
            t_line_ms=d["t_line_ms"], t_echo_ms=d["t_echo_ms"], dt_ms=d["dt_ms"],
        )


def _readout_from_acquisition(nx: int, ny: int, acq: dict[str, Any]) -> EpiReadout:
    t = _core.epi_timing(nx, ny, acq)
    kx, ky, tt = _core.epi_trajectory(nx, ny, acq)
    ticks = np.stack([kx.astype(np.int64), ky.astype(np.int64), tt], axis=1)
    # readout direction of each line: sign of the kx step within the line
    order = t["order"].astype(np.int64)
    kx_dir = np.ones(ny, dtype=np.int64)
    for i in range(ny):
        seg = kx[i * nx : (i + 1) * nx]
        if seg.size > 1 and seg[1] < seg[0]:
            kx_dir[i] = -1
    mask = _core.sampling_mask(nx, ny, acq).reshape(ny, nx).astype(bool)
    return EpiReadout(
        nx=nx, ny=ny, ky_order=order, kx_direction=kx_dir, t_ms=t["t_ms"], t_rf_ms=t["t_rf_ms"],
        t_read_ms=t["t_read_ms"], ticks=ticks, mask=mask, t_line_ms=float(t["t_line"]),
        t_echo_ms=float(t["t_echo"]), dt_ms=float(t["dt"]),
    )


# ─── protocol ───────────────────────────────────────────────────────────────

_PE = {"AP": False, "j-": False, "PA": True, "j": True}
_GNL_TAGS = ("ND", "DIS2D", "DIS3D")

#: Descriptive sidecar keys :meth:`Protocol.from_bids` carries over into ``metadata`` (and so
#: into the simulated run's sidecar): what scanner and sequence the run stands in for. Timing,
#: matrix and acceleration keys are never copied; they are derived from the simulation.
SIDECAR_PASSTHROUGH: tuple[str, ...] = (
    "Manufacturer", "ManufacturersModelName", "MagneticFieldStrength", "ImagingFrequency",
    "ReceiveCoilName", "ReceiveCoilActiveElements", "CoilCombinationMethod", "MatrixCoilMode",
    "PulseSequenceType", "PulseSequenceDetails", "ScanningSequence", "SequenceVariant", "SequenceName",
    "ScanOptions", "MRAcquisitionType", "DiffusionScheme",
    "BodyPartExamined", "PatientPosition", "ProtocolName", "SeriesDescription", "InstitutionName",
    "InstitutionAddress", "StationName", "SoftwareVersions", "PixelBandwidth",
    "NonlinearGradientCorrection", "DwellTime",
)


def simulation_stamp() -> dict[str, Any]:
    """The keys every simulated sidecar carries so a reader can tell it from a scan."""
    from . import __version__

    return {"SimulationSoftware": "TRXScan", "SimulationSoftwareVersion": __version__,
            "ConversionSoftware": "trxscan", "ConversionSoftwareVersion": __version__}


def siemens_slice_timing(nz: int, mb: int, tr_s: float) -> tuple[float, ...]:
    """Idealised Siemens/CMRR interleaved multiband slice timing: ``nz // mb`` shots of
    ``tr_s / n_shots`` each; slice ``z`` is excited in the shot of its in-group position
    ``z % n_shots``; an odd number of shots runs even positions first (0, 2, ..., 1, 3, ...),
    an even number odd first. Matches dcm2niix's ``SliceTiming`` for the CMRR diffusion
    sequence up to the sequence's dead time at the end of the TR."""
    n_shots = nz // mb
    if n_shots < 1 or nz % mb:
        raise ValueError(f"{nz} slices do not split into multiband groups of {mb}")
    first = 0 if n_shots % 2 else 1
    order = list(range(first, n_shots, 2)) + list(range(1 - first, n_shots, 2))
    shot_of = {pos: k for k, pos in enumerate(order)}
    dt = tr_s / n_shots
    return tuple(shot_of[z % n_shots] * dt for z in range(nz))
_PF_MODES = ("scanner", "contiguous", "fiberfox")
_PHASE_MODELS = ("hbcd", "none")


@dataclass(frozen=True)
class Protocol:
    """Scanner settings in physical units. Immutable; presets are instances
    (``Protocol.HBCD``, ``Protocol.DEFAULT``); vary with :meth:`replace`::

        Protocol.HBCD.replace(voxel_mm=2.5, coils=8, accel=2)

    The phase-encode axis is the grid's ``j`` axis. ``pe`` is ``"AP"`` (``"j-"``, the forward
    scan: a positive off-resonance field displaces signal toward -j) or ``"PA"`` (``"j"``, the
    reversed polarity of a blip-up/blip-down pair), named in the frame of the file you get:
    the object's native voxel axes, or LAS with ``fsl_orientation`` (then ``"j-"`` always
    means a posterior shift, whatever the phantom's native axes, and the sidecar's
    ``PhaseEncodingDirection`` equals ``pe``).

    ``readout_ms`` is the phase-encode train duration (BIDS ``TotalReadoutTime`` in ms); the
    per-line time is derived from it and the matrix at simulation time. Give
    ``echo_spacing_ms`` instead to fix the line time and let the train length follow the matrix.

    ``partial_fourier`` < 1 behaves like a scanner by default (``pf_mode="scanner"``): the train
    starts late and skips its first lines, so the k-space centre is reached sooner and the
    readout is shorter; ``te_ms`` is still what you set, so lower it to bank the TE gain.
    ``pf_mode="contiguous"`` drops the *last* lines instead with unchanged timing (the legacy
    TRXScan rule) and ``"fiberfox"`` is the ported Fiberfox rule.

    ``matrix`` fixes the acquired matrix ``(nx, ny, nz)``: the object is centred in it (the
    FOV of a real scan) instead of being gridded to its own bounding box plus ``pad``.
    ``oblique_deg`` ``(rx, ry, rz)`` tilts the acquisition grid about its centre (degrees about
    the RAS axes, applied x then y then z): an oblique acquisition, whose written header
    carries the rotation while the object stays where it is in the scanner.

    **Recorded, not modeled.** ``tr_s``, ``flip_angle_deg``, ``field_strength_t``,
    ``slice_timing`` and ``metadata`` go into the BIDS sidecar so a simulated run documents
    the scan it stands in for, but the signal model has no TR, T1 or flip-angle term and the
    acquisition has no slice clock. ``slice_timing`` (seconds per slice, one entry per slice
    of the full volume) defaults to a Siemens-style interleaved multiband order derived from
    ``tr_s``, the slice count and ``mb``. ``metadata`` is extra sidecar keys written verbatim
    (``Manufacturer``, ``PulseSequenceDetails``, ...); :meth:`from_bids` fills it from a real
    sidecar's descriptive keys. ``gnl_tag`` is the Siemens gradient-nonlinearity tag written
    into ``ImageType`` (``"ND"`` = not corrected, ``"DIS2D"``/``"DIS3D"`` = the scanner claims
    to have corrected in-plane / in 3-D); it changes the sidecar only, so pairing ``"DIS3D"``
    with ``Artifacts(gnl=...)`` deliberately mislabels a warped image for tests of a
    pipeline's tag handling.

    :meth:`replace` keeps the field of view: changing ``voxel_mm`` on a protocol that has a
    ``matrix`` rescales the matrix (``Protocol.from_bids(...).replace(voxel_mm=3.0)`` is the
    same scan at 3 mm, not a 3 mm scan with the 1.7 mm matrix). Pass ``matrix`` too to
    override that. Presets that pin ``readout_ms`` (``HBCD``: 91.7 ms) keep it under
    ``replace(voxel_mm=...)``, so the echo spacing, not the train length, follows the matrix;
    the sidecar reports whichever results.
    """

    voxel_mm: float | tuple[float, float, float] = 2.5
    oversample: int = 2
    pad: tuple[int, int, int] = (4, 14, 2)
    te_ms: float = 90.0
    readout_ms: float | None = None
    echo_spacing_ms: float | None = None
    t_inhom_ms: float = 50.0
    partial_fourier: float = 1.0
    pf_mode: Literal["scanner", "contiguous", "fiberfox"] = "scanner"
    acs_lines: int = 24
    coils: int = 1
    accel: int = 1
    mb: int = 1
    pe: Literal["AP", "PA", "j", "j-"] = "AP"
    tissue: Tissue = field(default_factory=lambda: Tissue.ADULT)
    kappa: float | None = None
    tissue_s0: tuple[float, float, float] = (1.0, 1.0, 1.0)
    diff_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    phase_model: Literal["hbcd", "none"] = "hbcd"
    fsl_orientation: bool = False
    signal_scale: float = 100.0
    matrix: tuple[int, int, int] | None = None
    oblique_deg: tuple[float, float, float] | None = None
    tr_s: float | None = None
    flip_angle_deg: float | None = None
    field_strength_t: float | None = 3.0
    slice_timing: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    gnl_tag: Literal["ND", "DIS2D", "DIS3D"] = "ND"
    name: str = ""

    HBCD: ClassVar["Protocol"]
    DEFAULT: ClassVar["Protocol"]

    def __post_init__(self) -> None:
        if self.pe not in _PE:
            raise ValueError(f"pe must be one of {sorted(_PE)}, got {self.pe!r}")
        if self.pf_mode not in _PF_MODES:
            raise ValueError(f"pf_mode must be one of {_PF_MODES}, got {self.pf_mode!r}")
        if self.phase_model not in _PHASE_MODELS:
            raise ValueError(f"phase_model must be one of {_PHASE_MODELS}, got {self.phase_model!r}")
        if not 0.5 <= self.partial_fourier <= 1.0:
            raise ValueError("partial_fourier must be in [0.5, 1]")
        if self.oversample < 1 or self.coils < 1 or self.accel < 1 or self.mb < 1:
            raise ValueError("oversample, coils, accel and mb must be >= 1")
        if self.readout_ms is not None and self.echo_spacing_ms is not None:
            raise ValueError("give readout_ms or echo_spacing_ms, not both")
        if self.matrix is not None:
            m = tuple(int(v) for v in self.matrix)
            if len(m) != 3 or min(m) < 1:
                raise ValueError(f"matrix must be three positive ints, got {self.matrix!r}")
            object.__setattr__(self, "matrix", m)
        if self.tr_s is not None and self.tr_s <= 0:
            raise ValueError("tr_s must be positive")
        if self.oblique_deg is not None:
            o = tuple(float(v) for v in self.oblique_deg)
            if len(o) != 3:
                raise ValueError("oblique_deg must be three angles (rx, ry, rz)")
            object.__setattr__(self, "oblique_deg", o)
        if self.slice_timing is not None:
            object.__setattr__(self, "slice_timing", tuple(float(v) for v in self.slice_timing))
        if self.gnl_tag not in _GNL_TAGS:
            raise ValueError(f"gnl_tag must be one of {_GNL_TAGS}, got {self.gnl_tag!r}")
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "tissue", _as_tissue(self.tissue))

    def replace(self, **kw: Any) -> "Protocol":
        if kw and "name" not in kw and self.name:
            kw["name"] = f"{self.name}*"
        if "voxel_mm" in kw and "matrix" not in kw and self.matrix is not None:
            new = kw["voxel_mm"]
            new = (float(new),) * 3 if np.isscalar(new) else tuple(float(v) for v in new)
            kw["matrix"] = tuple(max(1, int(round(n * o / v))) for n, o, v in zip(self.matrix, self.voxel, new))
        return dataclasses.replace(self, **kw)

    # -- derived -------------------------------------------------------------

    @property
    def fov_mm(self) -> tuple[float, float, float] | None:
        """The field of view ``matrix * voxel`` in mm, or None without a fixed matrix."""
        return None if self.matrix is None else tuple(n * v for n, v in zip(self.matrix, self.voxel))

    @property
    def voxel(self) -> tuple[float, float, float]:
        v = self.voxel_mm
        return (float(v), float(v), float(v)) if np.isscalar(v) else tuple(float(x) for x in v)  # type: ignore[arg-type]

    @property
    def reverse_phase(self) -> bool:
        return _PE[self.pe]

    def t_line_ms(self, ny: int) -> float:
        """ms per phase-encode line on an ``ny``-line matrix."""
        if self.readout_ms is not None:
            return self.readout_ms / max(ny, 1)
        if self.echo_spacing_ms is not None:
            return self.echo_spacing_ms
        return 1.0  # kspace::default_acquisition().t_line

    def acquisition(self, ny: int) -> dict[str, Any]:
        """The ``kspace::Acquisition`` fields this protocol fixes (artifact knobs come from
        :class:`~trxscan.artifacts.Artifacts`)."""
        return {
            "t_line": self.t_line_ms(ny),
            "t_echo": float(self.te_ms),
            "t_inhom": float(self.t_inhom_ms),
            "signal_scale": float(self.signal_scale),
            "reverse_phase": self.reverse_phase,
            "partial_fourier": float(self.partial_fourier),
            "pf_mode": self.pf_mode,
            "n_coils": int(self.coils),
            "accel": int(self.accel),
            "acs_lines": int(self.acs_lines),
        }

    def readout(self, nx: int, ny: int) -> EpiReadout:
        """The EPI readout timing on an ``nx`` x ``ny`` matrix."""
        return _readout_from_acquisition(nx, ny, self.acquisition(ny))

    def slice_timing_for(self, nz: int) -> tuple[float, ...] | None:
        """Seconds from the volume's start to each of ``nz`` slices (BIDS ``SliceTiming``):
        the explicit ``slice_timing`` when it has ``nz`` entries, else the Siemens interleaved
        multiband order derived from ``tr_s`` and ``mb`` (``nz`` must divide into ``mb``
        slice groups), else None."""
        if self.slice_timing is not None:
            return self.slice_timing if len(self.slice_timing) == nz else None
        if self.tr_s is None or nz < 1 or nz % self.mb:
            return None
        return siemens_slice_timing(nz, self.mb, self.tr_s)

    def sidecar(self, ny: int, nz: int | None = None, nx: int | None = None) -> dict[str, Any]:
        """The protocol part of the BIDS DWI sidecar for an ``nx`` x ``ny`` x ``nz`` acquisition
        (``PhaseEncodingDirection`` is added at write time from the written frame). Timing keys
        are the simulated readout; the recorded-only fields (TR, flip angle, slice timing,
        ``metadata``) are included when set."""
        t_line = self.t_line_ms(ny)
        trt = round(t_line * ny / 1000.0, 6)
        ees = round(trt / max(ny - 1, 1), 8)
        n_lines = int(round(ny * self.partial_fourier))
        side: dict[str, Any] = {"Manufacturer": "TRXScan"}
        side.update(self.metadata)
        side.update({
            "PulseSequenceType": "Single-shot spin-echo EPI",
            "ScanningSequence": "EP",
            "MRAcquisitionType": "2D",
            "TotalReadoutTime": trt,
            "EffectiveEchoSpacing": ees,
            "BandwidthPerPixelPhaseEncode": round(1.0 / (ees * ny), 4) if ees > 0 else None,
            "EchoTime": round(self.te_ms / 1000.0, 4),
            "PartialFourier": self.partial_fourier,
            "ParallelReductionFactorInPlane": self.accel,
            "MultibandAccelerationFactor": self.mb,
            "ReconMatrixPE": int(ny),
            "AcquisitionMatrixPE": int(ny),
            "PhaseEncodingSteps": n_lines,
            "EchoTrainLength": n_lines,
            "SliceThickness": self.voxel[2],
            "SpacingBetweenSlices": self.voxel[2],
        })
        side["ImageType"] = ["ORIGINAL", "PRIMARY", "DIFFUSION", "NONE"] + (["MB"] if self.mb > 1 else []) + [self.gnl_tag]
        if self.partial_fourier < 1.0:
            side["PartialFourierDirection"] = "PHASE"
        if self.accel > 1:
            side["ParallelAcquisitionTechnique"] = "GRAPPA"
        if self.tr_s is not None:
            side["RepetitionTime"] = float(self.tr_s)
        if self.flip_angle_deg is not None:
            side["FlipAngle"] = float(self.flip_angle_deg)
        if self.field_strength_t is not None:
            side["MagneticFieldStrength"] = float(self.field_strength_t)
        if nz is not None:
            st = self.slice_timing_for(nz)
            if st is not None:
                side["SliceTiming"] = [round(v, 6) for v in st]
        side.update(simulation_stamp())
        return {k: v for k, v in side.items() if v is not None}

    # -- from a real acquisition ---------------------------------------------

    @classmethod
    def from_bids(cls, json_path: str | Path, nifti_path: str | Path | None = None, **overrides: Any) -> "Protocol":
        """A protocol from a BIDS DWI sidecar (and the NIfTI header for the voxel size).

        Mapping: ``EchoTime`` -> ``te_ms``; ``TotalReadoutTime`` (else ``EffectiveEchoSpacing``
        x (``ReconMatrixPE`` or the header's PE dimension)) -> ``readout_ms``;
        ``PhaseEncodingDirection`` -> ``pe``; ``PartialFourier`` -> ``partial_fourier``;
        ``ParallelReductionFactorInPlane`` -> ``accel``; ``MultibandAccelerationFactor`` ->
        ``mb``; ``pixdim`` -> ``voxel_mm``; the header's shape -> ``matrix`` (so the simulated
        FOV matches); an LAS header -> ``fsl_orientation``. Recorded only: ``RepetitionTime``
        -> ``tr_s``, ``FlipAngle`` -> ``flip_angle_deg``, ``MagneticFieldStrength`` ->
        ``field_strength_t``, ``SliceTiming`` -> ``slice_timing``, and the scanner/sequence
        descriptors in :data:`SIDECAR_PASSTHROUGH` -> ``metadata``. Keys that are absent keep
        :attr:`DEFAULT`'s values and are listed in one warning. Nothing is inferred about
        artifacts.
        """
        import warnings

        json_path = Path(json_path)
        with open(json_path) as f:
            side = json.load(f)
        kw: dict[str, Any] = {}
        missing: list[str] = []
        if "EchoTime" in side:
            kw["te_ms"] = float(side["EchoTime"]) * 1000.0
        else:
            missing.append("EchoTime")
        ny = None
        if nifti_path is None:
            cand = json_path.with_suffix("").with_suffix(".nii.gz")
            if not cand.exists():
                cand = json_path.with_suffix(".nii")
            nifti_path = cand if cand.exists() else None
        if nifti_path is not None:
            import nibabel as nib

            img = nib.load(str(nifti_path))
            zooms = img.header.get_zooms()[:3]
            kw["voxel_mm"] = tuple(round(float(z), 6) for z in zooms)  # float32 pixdims -> clean mm
            kw["matrix"] = tuple(int(v) for v in img.shape[:3])
            kw["fsl_orientation"] = nib.aff2axcodes(img.affine) == ("L", "A", "S")
            ny = int(img.shape[1])
        else:
            missing.append("voxel size (no NIfTI found)")
        if "TotalReadoutTime" in side:
            kw["readout_ms"] = float(side["TotalReadoutTime"]) * 1000.0
        elif "EffectiveEchoSpacing" in side:
            n_pe = side.get("ReconMatrixPE", ny)
            if n_pe is not None:
                kw["readout_ms"] = float(side["EffectiveEchoSpacing"]) * 1000.0 * int(n_pe)
            else:
                kw["echo_spacing_ms"] = float(side["EffectiveEchoSpacing"]) * 1000.0
        else:
            missing.append("TotalReadoutTime/EffectiveEchoSpacing")
        ped = side.get("PhaseEncodingDirection")
        if ped in ("j", "j-"):
            kw["pe"] = ped
        elif ped is None:
            missing.append("PhaseEncodingDirection")
        else:
            warnings.warn(f"PhaseEncodingDirection {ped!r} is not along j; the simulator encodes along the grid's j axis, keeping pe='AP'", stacklevel=2)
        if "PartialFourier" in side:
            pf = float(side["PartialFourier"])
            kw["partial_fourier"] = pf if pf <= 1.0 else pf / 8.0  # some converters write 6 for 6/8
        else:
            missing.append("PartialFourier")
        if "ParallelReductionFactorInPlane" in side:
            kw["accel"] = int(round(float(side["ParallelReductionFactorInPlane"])))
        if "MultibandAccelerationFactor" in side:
            kw["mb"] = int(round(float(side["MultibandAccelerationFactor"])))
        if "RepetitionTime" in side:
            kw["tr_s"] = float(side["RepetitionTime"])
        if "FlipAngle" in side:
            kw["flip_angle_deg"] = float(side["FlipAngle"])
        if "MagneticFieldStrength" in side:
            kw["field_strength_t"] = float(side["MagneticFieldStrength"])
        if isinstance(side.get("SliceTiming"), list):
            kw["slice_timing"] = tuple(float(v) for v in side["SliceTiming"])
        tags = [t for t in side.get("ImageType", []) if t in _GNL_TAGS] if isinstance(side.get("ImageType"), list) else []
        if tags:
            kw["gnl_tag"] = tags[-1]
        kw["metadata"] = {k: side[k] for k in SIDECAR_PASSTHROUGH if k in side}
        if missing:
            warnings.warn(f"{json_path.name}: no {', '.join(missing)}; using Protocol.DEFAULT values for those", stacklevel=2)
        kw["name"] = json_path.name.replace(".json", "")
        kw.update(overrides)
        return cls.DEFAULT.replace(**kw)


Protocol.DEFAULT = Protocol(name="default")
# The CLI's shipping protocol (`kspace::hbcd_acquisition`): TE 88 ms, PE train pinned to HBCD's
# TotalReadoutTime 91.7 ms, 6/8 scanner-style partial Fourier, 24 ACS lines, 1.7 mm. The CLI also
# adds a subtle Nyquist ghost (0.015), which is an artifact here: `Artifacts(ghost=0.015)`.
# `readout_ms` is pinned, so `HBCD.replace(voxel_mm=3.0)` keeps the 91.7 ms train and shortens
# the echo spacing; give `readout_ms=None, echo_spacing_ms=...` to let the train follow the matrix.
Protocol.HBCD = Protocol(voxel_mm=1.7, te_ms=88.0, readout_ms=91.7, partial_fourier=0.75, acs_lines=24, name="hbcd")
