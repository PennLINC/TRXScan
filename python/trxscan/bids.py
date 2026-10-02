"""BIDS in and out.

* :func:`write_bids` — everything the ``trxscan`` CLI writes for one run under a stem.
* :class:`Dataset` — a BIDS dataset being assembled: ``dataset_description.json``,
  ``participants.tsv``, ``sub-*/[ses-*/]{anat,dwi,fmap}`` with entity-ordered names, the
  ground truth under ``derivatives/trxscan``; :meth:`Dataset.mirror` builds one that mimics a
  real dataset run for run.
* :class:`BidsDwi` — a real acquisition's gradient table and :class:`Protocol` from its files.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import nibabel as nib
import numpy as np

from .protocol import SIDECAR_PASSTHROUGH, Protocol, simulation_stamp

if TYPE_CHECKING:  # pragma: no cover
    from .artifacts import Artifacts
    from .phantom import Phantom
    from .simulate import Simulation


def write_bval_bvec(stem: str | Path, bvals: np.ndarray, bvecs: np.ndarray) -> tuple[Path, Path]:
    """FSL layout in TRXScan's own text format (``io::write_bval_bvec``): one row of b-values,
    three rows of components, ``{:.6}``-style formatting."""
    stem = Path(stem)
    bval_p, bvec_p = Path(f"{stem}.bval"), Path(f"{stem}.bvec")
    bval_p.write_text(" ".join(f"{b:g}" for b in np.asarray(bvals).reshape(-1)) + "\n")
    bv = np.asarray(bvecs).reshape(-1, 3)
    bvec_p.write_text("\n".join(" ".join(f"{v:.6f}" for v in bv[:, c]) for c in range(3)) + "\n")
    return bval_p, bvec_p


def read_bval_bvec(bval: str | Path, bvec: str | Path) -> tuple[np.ndarray, np.ndarray]:
    bvals = np.loadtxt(bval, dtype=np.float64).reshape(-1)
    bvecs = np.loadtxt(bvec, dtype=np.float64)
    if bvecs.shape == (3, bvals.size):
        bvecs = bvecs.T
    return bvals, bvecs.reshape(-1, 3)


def _json(d: dict[str, Any]) -> str:
    return json.dumps(d, indent=2) + "\n"


def write_bids(sim: "Simulation", prefix: str | Path) -> list[Path]:
    """Write ``<prefix>_part-mag_dwi.nii.gz``, ``_part-phase_dwi.nii.gz`` (+ JSON sidecars),
    ``_dwi.bval``/``.bvec``, and whatever ground truth the run produced: ``_desc-noise_sigma``,
    ``_desc-truth_peaks``, ``_desc-gnl_{coeff.grad,disp,invdisp,graddev}``,
    ``_desc-dropout_slices.tsv`` and the GRE fieldmap files — the same names and sidecar keys as
    the CLI."""
    prefix = str(prefix)
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []

    def p(suffix: str) -> Path:
        q = Path(f"{prefix}{suffix}")
        out.append(q)
        return q

    sim.magnitude.to_filename(str(p("_part-mag_dwi.nii.gz")))
    sim.phase.to_filename(str(p("_part-phase_dwi.nii.gz")))
    write_bval_bvec(f"{prefix}_dwi", sim.bvals, sim.bvecs_fsl)
    out += [Path(f"{prefix}_dwi.bval"), Path(f"{prefix}_dwi.bvec")]
    common = dict(sim.sidecar)
    p("_part-mag_dwi.json").write_text(_json({**common, "ImageComparison": "magnitude"}))
    p("_part-phase_dwi.json").write_text(_json({**common, "ImageComparison": "phase", "Units": "rad"}))
    if sim.noise_sigma is not None:
        sim.noise_sigma.to_filename(str(p("_desc-noise_sigma.nii.gz")))
    if sim.truth_peaks is not None:
        sim.truth_peaks.to_filename(str(p("_desc-truth_peaks.nii.gz")))
        p("_desc-truth_peaks.json").write_text(_json({
            "Description": "Ground-truth fibre orientations: up to 3 peaks of the orientation mixture per voxel, volumes 3k..3k+2 = peak k as a unit vector in world RAS scaled by its mass fraction (0 = no peak). Object in its true (gradient-nonlinearity-free) frame."
        }))
    if sim.gnl is not None:
        p("_desc-gnl_coeff.grad").write_text(sim.gnl.coeff_text)
        sim.gnl.disp.to_filename(str(p("_desc-gnl_disp.nii.gz")))
        sim.gnl.invdisp.to_filename(str(p("_desc-gnl_invdisp.nii.gz")))
        sim.gnl.graddev.to_filename(str(p("_desc-gnl_graddev.nii.gz")))
        p("_desc-gnl_graddev.json").write_text(_json({
            "Description": "Ground-truth gradient deviation: 9 volumes, HCP/FSL layout; read row-major into T, the applied gradient in this image's voxel axes is T.T @ g; identity included.",
            "GradientNonlinearity": sim.gnl.spec, "GradientNonlinearityScale": sim.gnl.scale,
            "SpatialWarpApplied": sim.gnl.warp, "EncodingDeviationApplied": sim.gnl.encoding, "JacobianModulation": sim.gnl.jacobian,
        }))
        p("_desc-gnl_disp.json").write_text(_json({"Description": "Ground-truth gradient-nonlinearity displacement d(r) = phi(r) - r at each voxel centre r, three volumes = RAS x,y,z in mm (apparent minus true position). Apply to points/streamlines to warp true -> apparent."}))
        p("_desc-gnl_invdisp.json").write_text(_json({"Description": "Inverse gradient-nonlinearity displacement phi^-1(x) - x at each voxel centre x, three volumes = RAS x,y,z in mm (true minus apparent position). Resample images through it to pull apparent <- true."}))
    if sim.dropout:
        p("_desc-dropout_slices.tsv").write_text(sim.dropout_table())
    if sim.gre is not None:
        g = sim.gre
        gp = f"{prefix}_gre"
        te = g.te_s
        for i, img in enumerate((g.magnitude1, g.magnitude2), start=1):
            img.to_filename(str(p(f"_gre_magnitude{i}.nii.gz")))
            Path(f"{gp}_magnitude{i}.json").write_text(_json({"EchoTime": te[i - 1], "B0FieldIdentifier": g.b0_field}))
            out.append(Path(f"{gp}_magnitude{i}.json"))
        for name, img in g.phase.items():
            img.to_filename(str(p(f"_gre_{name}.nii.gz")))
            meta: dict[str, Any] = {"B0FieldIdentifier": g.b0_field, "Units": "arbitrary"}
            if name == "phasediff":
                meta.update({"EchoTime1": te[0], "EchoTime2": te[1]})
            else:
                meta["EchoTime"] = te[0] if name == "phase1" else te[1]
            Path(f"{gp}_{name}.json").write_text(_json(meta))
            out.append(Path(f"{gp}_{name}.json"))
    return out


@dataclass(frozen=True)
class BidsDwi:
    """A real diffusion acquisition read from its BIDS files: the gradient table (dipy
    ``GradientTable`` when dipy is installed, else a ``(bvals, bvecs)`` pair), the
    :class:`Protocol` its sidecar and header imply, and the raw sidecar."""

    gtab: Any
    protocol: Protocol
    sidecar: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: str | Path, **protocol_overrides: Any) -> "BidsDwi":
        """From the DWI NIfTI (``sub-01_dwi.nii.gz``) or its JSON; the ``.bval``/``.bvec``/
        ``.json`` siblings are found by BIDS naming."""
        path = Path(path)
        name = path.name
        for ext in (".nii.gz", ".nii", ".json"):
            if name.endswith(ext):
                stem = name[: -len(ext)]
                break
        else:
            raise ValueError("expected a .nii.gz/.nii/.json path")
        base = path.parent / stem
        json_p = Path(f"{base}.json")
        nii = Path(f"{base}.nii.gz") if Path(f"{base}.nii.gz").exists() else (Path(f"{base}.nii") if Path(f"{base}.nii").exists() else None)
        bval_p, bvec_p = Path(f"{base}.bval"), Path(f"{base}.bvec")
        if not json_p.exists():
            raise FileNotFoundError(json_p)
        if not (bval_p.exists() and bvec_p.exists()):
            raise FileNotFoundError(f"{base}.bval/.bvec")
        with open(json_p) as f:
            side = json.load(f)
        bvals, bvecs = read_bval_bvec(bval_p, bvec_p)
        try:
            from dipy.core.gradients import gradient_table

            gtab = gradient_table(bvals, bvecs=bvecs)
        except ImportError:  # pragma: no cover
            gtab = (bvals, bvecs)
        proto = Protocol.from_bids(json_p, nii, **protocol_overrides)
        return cls(gtab=gtab, protocol=proto, sidecar=side, path=path)

    @classmethod
    def load_epi(cls, path: str | Path, **protocol_overrides: Any) -> "BidsDwi":
        """A PEPOLAR ``_epi`` fieldmap (or any b=0 EPI without bval/bvec): the gradient table
        is all-zero with one entry per volume."""
        path = Path(path)
        ents, suf, ext = split_name(path)
        base = path.parent / path.name[: -len(ext)]
        json_p = Path(f"{base}.json")
        if not json_p.exists():
            raise FileNotFoundError(json_p)
        with open(json_p) as f:
            side = json.load(f)
        img = nib.load(str(path))
        n = int(img.shape[3]) if img.ndim == 4 else 1
        proto = Protocol.from_bids(json_p, path, **protocol_overrides)
        return cls(gtab=_b0_gtab(n), protocol=proto, sidecar=side, path=path)

    @property
    def n_vol(self) -> int:
        g = self.gtab
        return int(g[0].size) if isinstance(g, tuple) else int(g.bvals.size)


# ─── names ──────────────────────────────────────────────────────────────────

#: BIDS entity order (schema 1.9) for composing file names.
ENTITY_ORDER: tuple[str, ...] = (
    "sub", "ses", "sample", "task", "tracksys", "acq", "nuc", "voi", "ce", "trc", "stain", "rec", "dir", "run",
    "mod", "echo", "flip", "inv", "mt", "part", "proc", "hemi", "space", "split", "recording", "chunk", "seg",
    "res", "den", "label", "from", "to", "mode", "desc",
)
_EXTS = (".nii.gz", ".nii", ".json", ".bval", ".bvec", ".tsv", ".grad")


def split_name(name: str | Path) -> tuple[dict[str, str], str, str]:
    """``"sub-01_ses-a_dir-AP_part-mag_dwi.nii.gz"`` -> ``({"sub": "01", "ses": "a", "dir": "AP",
    "part": "mag"}, "dwi", ".nii.gz")``."""
    name = Path(name).name
    ext = next((e for e in _EXTS if name.endswith(e)), "")
    stem = name[: len(name) - len(ext)] if ext else name
    parts = stem.split("_")
    suffix = parts[-1]
    ents: dict[str, str] = {}
    for kv in parts[:-1]:
        k, _, v = kv.partition("-")
        ents[k] = v
    return ents, suffix, ext


def bids_name(entities: dict[str, Any], suffix: str, ext: str = "") -> str:
    """Compose a BIDS file name with entities in schema order (``None`` values are skipped)."""
    keys = [k for k in ENTITY_ORDER if entities.get(k) is not None]
    extra = [k for k in entities if k not in ENTITY_ORDER and entities[k] is not None]
    parts = [f"{k}-{entities[k]}" for k in keys + extra]
    return "_".join(parts + [suffix]) + ext


def _dir_of(protocol: Protocol) -> str:
    return "PA" if protocol.reverse_phase else "AP"


# ─── anatomical images ──────────────────────────────────────────────────────

#: Relative compartment intensities of the synthetic tissue-contrast images.
ANAT_CONTRAST: dict[str, tuple[float, float, float]] = {"T1w": (1.0, 0.72, 0.20), "T2w": (0.55, 0.75, 1.0)}


def synthesize_anat(phantom: "Phantom", suffix: str = "T1w", scale: float = 1000.0) -> nib.Nifti1Image:
    """A synthetic ``T1w``/``T2w``-like image on the phantom's anatomical grid: the WM/GM/CSF
    fractions weighted by :data:`ANAT_CONTRAST` (no relaxometry, no noise, no bias field).
    The stand-in when a phantom has no real anatomical image."""
    if suffix not in ANAT_CONTRAST:
        raise ValueError(f"no synthetic contrast for {suffix!r}; known: {sorted(ANAT_CONTRAST)}")
    w = ANAT_CONTRAST[suffix]
    frac = [np.asanyarray(img.dataobj, dtype=np.float32) for img in (phantom.wm, phantom.gm, phantom.csf)]
    data = scale * (w[0] * frac[0] + w[1] * frac[1] + w[2] * frac[2])
    if phantom.mask is not None:
        data = data * (np.asanyarray(phantom.mask.dataobj, dtype=np.float32) > 0)
    img = nib.Nifti1Image(data.astype(np.float32), phantom.anatomical_affine)
    img.header.set_qform(phantom.anatomical_affine, code=1)
    img.header.set_sform(phantom.anatomical_affine, code=1)
    img.header.set_xyzt_units("mm")
    return img


def phantom_anat(phantom: "Phantom", suffix: str, mode: str = "auto") -> tuple[nib.Nifti1Image, str]:
    """The phantom's ``T1w``/``T2w`` image and a provenance note. ``mode``: ``"phantom"`` (the
    subject's image, error if absent), ``"synthetic"`` (always :func:`synthesize_anat`), or
    ``"auto"`` (the subject's image when present, else synthetic)."""
    attr = {"T1w": "t1w", "T2w": "t2w"}.get(suffix)
    img = getattr(phantom, attr) if attr else None
    if mode == "phantom" or (mode == "auto" and img is not None):
        if img is None:
            raise ValueError(f"phantom {phantom.name!r} has no {suffix} image")
        return img, f"Subject {suffix} image of the phantom ({phantom.name}) in the simulation's world frame."
    if mode in ("synthetic", "auto"):
        w = ANAT_CONTRAST[suffix]
        return synthesize_anat(phantom, suffix), (
            f"Synthetic {suffix}-like tissue-contrast image: WM/GM/CSF fractions weighted {w[0]}/{w[1]}/{w[2]}; "
            "no relaxometry, noise or bias field."
        )
    raise ValueError("mode must be 'auto', 'phantom' or 'synthetic'")


def match_geometry(img: nib.Nifti1Image, zooms: Sequence[float] | None = None, axcodes: Sequence[str] | None = None) -> nib.Nifti1Image:
    """Resample ``img`` to ``zooms`` (mm, trilinear) and/or reorder its axes to ``axcodes``
    (e.g. ``("L", "A", "S")``), keeping its world frame."""
    from nibabel.processing import resample_to_output

    out = img
    if zooms is not None and not np.allclose(out.header.get_zooms()[:3], zooms, rtol=1e-4):
        out = resample_to_output(out, voxel_sizes=tuple(float(z) for z in zooms), order=1)
    if axcodes is not None and nib.aff2axcodes(out.affine) != tuple(axcodes):
        cur = nib.io_orientation(out.affine)
        want = nib.orientations.axcodes2ornt(tuple(axcodes))
        out = out.as_reoriented(nib.orientations.ornt_transform(cur, want))
    out = nib.Nifti1Image(np.asanyarray(out.dataobj, dtype=np.float32), out.affine)
    out.header.set_qform(out.affine, code=1)
    out.header.set_sform(out.affine, code=1)
    out.header.set_xyzt_units("mm")
    return out


# keys of a real anatomical sidecar worth carrying into the stand-in's sidecar
_ANAT_PASSTHROUGH = SIDECAR_PASSTHROUGH + ("EchoTime", "RepetitionTime", "InversionTime", "FlipAngle", "EchoTrainLength")


def itk_displacement(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """A 4-D RAS displacement image (three volumes, mm) as the ITK/ANTs displacement-field
    layout: 5-D ``(nx, ny, nz, 1, 3)`` float32, LPS components, intent ``vector``."""
    d = np.asanyarray(img.dataobj, dtype=np.float32)
    if d.ndim != 4 or d.shape[-1] != 3:
        raise ValueError(f"expected a (nx, ny, nz, 3) displacement image, got {d.shape}")
    lps = d * np.array([-1.0, -1.0, 1.0], np.float32)
    out = nib.Nifti1Image(lps[:, :, :, None, :], img.affine)
    out.header.set_intent("vector")
    out.header.set_qform(img.affine, code=1)
    out.header.set_sform(img.affine, code=1)
    out.header.set_xyzt_units("mm")
    return out


def rigid_offset(trans_mm: Sequence[float] = (0.0, 0.0, 0.0), rot_deg: Sequence[float] = (0.0, 0.0, 0.0), center: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """A 4x4 world (RAS mm) rigid transform: rotate by ``rot_deg`` about the RAS axes (x, then
    y, then z) around ``center``, then translate by ``trans_mm``."""
    rx, ry, rz = np.radians(np.asarray(rot_deg, dtype=np.float64))
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    R = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    c = np.asarray(center, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c - R @ c + np.asarray(trans_mm, dtype=np.float64)
    return T


def offset_image(img: nib.Nifti1Image, T: np.ndarray, mode: str = "resample", phase: str | None = None) -> nib.Nifti1Image:
    """``img`` as seen after the object moved by ``T`` (world RAS). ``mode="header"`` keeps the
    voxels and pre-multiplies the affine (exact, but a pipeline that drops header obliquity
    never sees the movement); ``mode="resample"`` (default) keeps the original grid and
    resamples the moved object into it (trilinear), which is what a scanner would record.
    ``phase`` resamples a wrapped phase image through its complex representation:
    ``"siemens"`` for int16 0..4095 phase difference images, ``"rad"`` for radians."""
    T = np.asarray(T, dtype=np.float64)
    if mode == "header":
        out = nib.Nifti1Image(np.asanyarray(img.dataobj), T @ img.affine, img.header)
        out.header.set_qform(out.affine, code=1)
        out.header.set_sform(out.affine, code=1)
        return out
    if mode != "resample":
        raise ValueError("mode must be 'resample' or 'header'")
    from nibabel.processing import resample_from_to

    moved = nib.Nifti1Image(np.asanyarray(img.dataobj), T @ img.affine)
    target = (img.shape[:3], img.affine)
    if phase is None:
        data = resample_from_to(moved, target, order=1).get_fdata().astype(np.float32)
    else:
        raw = np.asanyarray(img.dataobj, dtype=np.float64)
        rad = raw / 4096.0 * 2 * np.pi - np.pi if phase == "siemens" else raw
        re = resample_from_to(nib.Nifti1Image(np.cos(rad).astype(np.float32), T @ img.affine), target, order=1).get_fdata()
        im = resample_from_to(nib.Nifti1Image(np.sin(rad).astype(np.float32), T @ img.affine), target, order=1).get_fdata()
        ang = np.arctan2(im, re)
        data = np.round((ang + np.pi) / (2 * np.pi) * 4096.0).clip(0, 4095).astype(np.int16) if phase == "siemens" else ang.astype(np.float32)
    out = nib.Nifti1Image(data, img.affine)
    out.header.set_qform(img.affine, code=1)
    out.header.set_sform(img.affine, code=1)
    out.header.set_xyzt_units("mm")
    return out


def itk_transform_text(T_ras: np.ndarray) -> str:
    """An ITK text transform (``AffineTransform_double_3_3``, LPS) that maps a point of the
    un-moved frame to the same tissue in the frame moved by ``T_ras``: the transform
    ``antsApplyTransforms`` needs to resample the moved image back onto the un-moved grid."""
    LPS = np.diag([-1.0, -1.0, 1.0])
    A = LPS @ T_ras[:3, :3] @ LPS
    t = LPS @ T_ras[:3, 3]
    params = " ".join(f"{v:.10g}" for v in np.concatenate([A.ravel(), t]))
    return f"#Insight Transform File V1.0\n#Transform 0\nTransform: AffineTransform_double_3_3\nParameters: {params}\nFixedParameters: 0 0 0\n"


def _offset_of(offset: Any, img: nib.Nifti1Image) -> np.ndarray | None:
    """``offset`` as a 4x4: None, a 4x4, or six numbers ``(tx, ty, tz, rx, ry, rz)`` (mm, deg)
    about the image's volume centre."""
    if offset is None:
        return None
    o = np.asarray(offset, dtype=np.float64)
    if o.shape == (4, 4):
        return o
    if o.size != 6:
        raise ValueError("offset must be a 4x4 world transform or (tx, ty, tz, rx, ry, rz)")
    centre = img.affine @ np.append((np.asarray(img.shape[:3]) - 1) / 2.0, 1.0)
    return rigid_offset(o[:3], o[3:], centre[:3])


# ─── Dataset ────────────────────────────────────────────────────────────────


@dataclass(eq=False)
class Dataset:
    """A BIDS dataset being written at ``root``. Creating one writes ``dataset_description.json``
    (and ``README`` if absent); :meth:`add_dwi`, :meth:`add_anat` and :meth:`add_participant`
    append to it, keeping ``participants.tsv`` and the ``_sessions.tsv`` files current. Raw
    folders hold only what a scanner would have produced; the simulation's ground truth
    (noise sigma, truth peaks, GNL fields, dropped shots) goes to ``derivatives/trxscan``
    with the same stems.

    Reopening an existing root keeps its participants and sessions.
    """

    root: Path
    name: str = "TRXScan simulation"
    authors: tuple[str, ...] = ()
    derivatives: str = "trxscan"
    description: dict[str, Any] = field(default_factory=dict)
    participants: dict[str, dict[str, Any]] = field(default_factory=dict)
    sessions: dict[str, set[str]] = field(default_factory=dict)
    written: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.participants = dict(self.participants)
        self.sessions = {k: set(v) for k, v in self.sessions.items()}
        ptsv = self.root / "participants.tsv"
        if ptsv.exists():
            rows = [line.rstrip("\n").split("\t") for line in ptsv.read_text().splitlines() if line.strip()]
            if rows:
                head = rows[0]
                for r in rows[1:]:
                    d = dict(zip(head, r))
                    self.participants.setdefault(d.pop("participant_id"), d)
        for sdir in self.root.glob("sub-*"):
            if sdir.is_dir():
                self.sessions.setdefault(sdir.name, set()).update(d.name for d in sdir.glob("ses-*") if d.is_dir())
        desc = {
            "Name": self.name, "BIDSVersion": "1.9.0", "DatasetType": "raw",
            "GeneratedBy": [{"Name": "TRXScan", "Version": simulation_stamp()["SimulationSoftwareVersion"],
                             "Description": "Simulated diffusion MRI from a tractogram phantom", "CodeURL": "https://github.com/PennLINC/TRXScan"}],
        }
        if self.authors:
            desc["Authors"] = list(self.authors)
        desc.update(self.description)
        self._write_json(self.root / "dataset_description.json", desc)
        readme = self.root / "README"
        if not readme.exists():
            readme.write_text(
                f"{self.name}\n\nSimulated with TRXScan (https://github.com/PennLINC/TRXScan). Every image is synthetic; the\n"
                "sidecars record the protocol the simulation stands in for (SimulationSoftware marks them).\n"
                f"Ground truth lives in derivatives/{self.derivatives}.\n"
            )
        dd = self.deriv_root / "dataset_description.json"
        if not dd.exists():
            self.deriv_root.mkdir(parents=True, exist_ok=True)
            self._write_json(dd, {
                "Name": f"{self.name}: ground truth", "BIDSVersion": "1.9.0", "DatasetType": "derivative",
                "GeneratedBy": desc["GeneratedBy"],
            })
        self._write_participants()

    # -- layout --------------------------------------------------------------

    @property
    def deriv_root(self) -> Path:
        return self.root / "derivatives" / self.derivatives

    @staticmethod
    def _label(sub: str | None, prefix: str) -> str | None:
        if sub is None:
            return None
        s = str(sub)
        return s[len(prefix) + 1 :] if s.startswith(prefix + "-") else s

    def folder(self, sub: str, ses: str | None, datatype: str, *, derivative: bool = False) -> Path:
        """``root/sub-X/[ses-Y/]datatype`` (created), registering the subject and session."""
        sub_l, ses_l = self._label(sub, "sub"), self._label(ses, "ses")
        base = (self.deriv_root if derivative else self.root) / f"sub-{sub_l}"
        if ses_l:
            base = base / f"ses-{ses_l}"
        base = base / datatype
        base.mkdir(parents=True, exist_ok=True)
        self.add_participant(sub_l)
        if ses_l:
            self.add_session(sub_l, ses_l)
        return base

    def add_participant(self, sub: str, **columns: Any) -> None:
        """Register ``sub`` (label or ``sub-``label) with optional ``participants.tsv`` columns."""
        sub_l = f"sub-{self._label(sub, 'sub')}"
        row = self.participants.setdefault(sub_l, {})
        row.update(columns)
        self._write_participants()

    def add_session(self, sub: str, ses: str) -> None:
        sub_l, ses_l = f"sub-{self._label(sub, 'sub')}", f"ses-{self._label(ses, 'ses')}"
        self.sessions.setdefault(sub_l, set()).add(ses_l)
        sdir = self.root / sub_l
        sdir.mkdir(parents=True, exist_ok=True)
        rows = sorted(self.sessions[sub_l])
        (sdir / f"{sub_l}_sessions.tsv").write_text("session_id\n" + "".join(f"{r}\n" for r in rows))

    def _write_participants(self) -> None:
        cols: list[str] = []
        for row in self.participants.values():
            cols += [c for c in row if c not in cols]
        lines = ["\t".join(["participant_id"] + cols)]
        for sub in sorted(self.participants):
            row = self.participants[sub]
            lines.append("\t".join([sub] + [str(row.get(c, "n/a")) for c in cols]))
        (self.root / "participants.tsv").write_text("\n".join(lines) + "\n")

    def _write_json(self, path: Path, d: dict[str, Any]) -> Path:
        path.write_text(_json(d))
        self.written.append(path)
        return path

    def _write_img(self, path: Path, img: nib.Nifti1Image) -> Path:
        img.to_filename(str(path))
        self.written.append(path)
        return path

    def uri(self, path: Path) -> str:
        """The ``bids::`` URI of a file in this dataset (for ``IntendedFor``)."""
        return "bids::" + path.relative_to(self.root).as_posix()

    # -- runs ----------------------------------------------------------------

    def add_dwi(
        self, sim: "Simulation", sub: str, ses: str | None = None, *, suffix: str = "dwi",
        parts: Sequence[str] | None = ("mag", "phase"), truth: bool = True, b0_field: str | None = None,
        b0_identifier: str | None = None, intended_for: Sequence[str | Path] | None = None,
        fmap_entities: dict[str, Any] | None = None, fmap_offset: Any = None, dwi_offset: Any = None, **entities: Any,
    ) -> list[Path]:
        """Write a simulated run. ``suffix`` is ``"dwi"`` (+ bval/bvec), ``"sbref"`` or ``"epi"``
        (a PEPOLAR fieldmap: goes to ``fmap/``, magnitude only, with ``IntendedFor`` =
        ``intended_for``). ``parts`` ``("mag", "phase")`` writes complex data as ``part-mag`` /
        ``part-phase`` (one bval/bvec pair without the ``part`` entity serves both); ``None``
        or ``("mag",)`` writes the magnitude alone without a ``part`` entity. ``dir`` defaults to the protocol's phase-encode polarity; ``dir=None`` leaves it out. A GRE fieldmap
        simulated with the run is written to ``fmap/`` (``fmap_entities`` names it) and
        ``b0_field`` stamps ``B0FieldSource`` on a run whose fieldmap another run carries;
        ``b0_identifier`` also stamps ``B0FieldIdentifier``, for a DWI that is itself a source
        of a PEPOLAR estimation (a reverse-PE pair needs both members under one identifier).
        ``fmap_offset`` moves the subject between the DWI and the fieldmap written here (the
        GRE carried by the run, or this run itself when ``suffix="epi"``); ``dwi_offset`` moves
        this DWI run itself relative to the dataset's other runs (the bvecs stay in the image
        axes, so a pipeline aligning runs must rotate them); see :meth:`add_anat`.
        Ground truth goes to the derivatives tree when ``truth``. Returns the raw files written."""
        ent: dict[str, Any] = {"sub": self._label(sub, "sub"), "ses": self._label(ses, "ses")}
        if "dir" not in entities:  # default the polarity entity; an explicit dir=None leaves it out
            entities["dir"] = _dir_of(sim.protocol)
        ent.update({k: v for k, v in entities.items() if v is not None})
        is_fmap = suffix == "epi"
        datatype = "fmap" if is_fmap else "dwi"
        out_dir = self.folder(sub, ses, datatype)
        side = dict(sim.sidecar)
        side.pop("ImageComparison", None)
        if b0_field is not None:
            side["B0FieldSource"] = b0_field
        if b0_identifier is not None:
            side["B0FieldIdentifier"] = b0_identifier
        if is_fmap:
            side.pop("B0FieldSource", None)
            if intended_for:
                side["IntendedFor"] = [str(p) if str(p).startswith("bids::") else self.uri(Path(p)) for p in intended_for]
            if b0_field is not None:
                side["B0FieldIdentifier"] = b0_field
        written: list[Path] = []
        use_parts = [p for p in (parts or ("mag",)) if p in ("mag", "phase")]
        complex_out = len(use_parts) > 1 and not is_fmap
        mag_stem: Path | None = None
        T_epi = _offset_of(fmap_offset, sim.magnitude) if is_fmap else _offset_of(dwi_offset, sim.magnitude)
        for part in use_parts if complex_out else ["mag"]:
            e = {**ent, "part": part if complex_out else None}
            stem = out_dir / bids_name(e, suffix)
            img = sim.magnitude if part == "mag" else sim.phase
            if T_epi is not None:
                img = offset_image(img, T_epi, "resample")
            written.append(self._write_img(Path(f"{stem}.nii.gz"), img))
            meta = dict(side)
            if part == "phase":
                meta["Units"] = "rad"
            written.append(self._write_json(Path(f"{stem}.json"), meta))
            if part == "mag":
                mag_stem = stem
        assert mag_stem is not None
        if suffix == "dwi":
            # no `part` entity: by inheritance one bval/bvec pair serves part-mag and part-phase
            bval_p, bvec_p = write_bval_bvec(out_dir / bids_name({**ent, "part": None}, suffix), sim.bvals, sim.bvecs_fsl)
            written += [bval_p, bvec_p]
            self.written += [bval_p, bvec_p]
        if T_epi is not None:
            written += self.add_offset_truth(sub, ses, datatype, "epi" if is_fmap else (f"dir{ent['dir']}" if ent.get("dir") else "run"), T_epi,
                                             **{k: v for k, v in ent.items() if k not in ("sub", "ses")})
        if sim.gre is not None and not is_fmap:
            written += self.add_gre(sim, sub, ses, intended_for=[written[0]], offset=fmap_offset, **(fmap_entities or {k: ent[k] for k in ("run",) if ent.get(k)}))
        if truth:
            self._add_truth(sim, sub, ses, {**ent, "part": None}, suffix)
        return written

    def add_gre(self, sim: "Simulation", sub: str, ses: str | None = None, *, intended_for: Sequence[str | Path] | None = None, offset: Any = None, offset_mode: str = "resample", **entities: Any) -> list[Path]:
        """Write the GRE fieldmap simulated with ``sim`` (``magnitude1``/``magnitude2`` +
        ``phasediff`` or ``phase1``/``phase2``) to ``fmap/`` with ``B0FieldIdentifier`` and
        ``IntendedFor``. ``offset`` moves the subject between the DWI and the fieldmap (see
        :meth:`add_anat`); the field moves with the head."""
        g = sim.gre
        if g is None:
            raise ValueError("the simulation carries no GRE fieldmap (run with gre=Gre())")
        ent: dict[str, Any] = {"sub": self._label(sub, "sub"), "ses": self._label(ses, "ses")}
        ent.update({k: v for k, v in entities.items() if v is not None})
        out_dir = self.folder(sub, ses, "fmap")
        T = _offset_of(offset, g.magnitude1)
        mv = (lambda im, ph=None: offset_image(im, T, offset_mode, ph)) if T is not None else (lambda im, ph=None: im)
        base: dict[str, Any] = {"B0FieldIdentifier": g.b0_field, **simulation_stamp()}
        for k in ("Manufacturer", "ManufacturersModelName", "MagneticFieldStrength"):
            if k in sim.sidecar:
                base[k] = sim.sidecar[k]
        if intended_for:
            base["IntendedFor"] = [str(p) if str(p).startswith("bids::") else self.uri(Path(p)) for p in intended_for]
        te = g.te_s
        written: list[Path] = []
        for i, img in enumerate((g.magnitude1, g.magnitude2), start=1):
            stem = out_dir / bids_name(ent, f"magnitude{i}")
            written.append(self._write_img(Path(f"{stem}.nii.gz"), mv(img)))
            written.append(self._write_json(Path(f"{stem}.json"), {**base, "EchoTime": te[i - 1]}))
        for name, img in g.phase.items():
            stem = out_dir / bids_name(ent, name)
            meta: dict[str, Any] = {**base, "Units": "arbitrary"}
            if name == "phasediff":
                meta.update({"EchoTime1": te[0], "EchoTime2": te[1]})
            else:
                meta["EchoTime"] = te[0] if name == "phase1" else te[1]
            written.append(self._write_img(Path(f"{stem}.nii.gz"), mv(img, "siemens")))
            written.append(self._write_json(Path(f"{stem}.json"), meta))
        if T is not None:
            written += self.add_offset_truth(sub, ses, "fmap", "fmap", T, **{k: v for k, v in ent.items() if k not in ("sub", "ses")})
        return written

    def _add_truth(self, sim: "Simulation", sub: str, ses: str | None, ent: dict[str, Any], suffix: str) -> list[Path]:
        has_field = sim.fieldmap is not None and bool(np.any(np.asanyarray(sim.fieldmap.dataobj)))
        if sim.noise_sigma is None and sim.truth_peaks is None and sim.gnl is None and not sim.dropout and sim.clean_b0 is None and not has_field:
            return []
        out_dir = self.folder(sub, ses, "dwi", derivative=True)
        written: list[Path] = []
        stamp = simulation_stamp()

        def put(desc: str, img: nib.Nifti1Image | None = None, meta: dict[str, Any] | None = None, text: str | None = None, ext: str = ".nii.gz") -> None:
            stem = out_dir / bids_name({**ent, "desc": desc}, suffix)
            if img is not None:
                written.append(self._write_img(Path(f"{stem}{ext}"), img))
            if text is not None:
                p = Path(f"{stem}{ext}")
                p.write_text(text)
                written.append(p)
                self.written.append(p)
            if meta is not None:
                written.append(self._write_json(Path(f"{stem}.json"), {**meta, **stamp}))

        if sim.clean_b0 is not None:
            put("cleanb0", sim.clean_b0, {"Description": "b=0 volume of the same object and protocol with every geometric and noise artifact off (no susceptibility distortion, gradient nonlinearity, eddy, ghost, spikes, noise, motion or dropout): what a corrected b=0 should converge to."})
        if has_field:
            put("fieldmap", sim.fieldmap, {"Units": "Hz", "Description": "Off-resonance field the acquisition applied, on the acquisition grid (a +field shifts signal toward +PhaseEncodingDirection)."})
            disp = sim.displacement
            if disp is not None:
                put("displacement", disp, {"Units": "mm", "Description": "Susceptibility displacement applied: three volumes = RAS x,y,z of where tissue at each true position appears (apparent minus true), fieldmap x TotalReadoutTime along the phase-encode axis."})
        if sim.noise_sigma is not None:
            put("noisesigma", sim.noise_sigma, {"Description": "Per-voxel magnitude noise SD applied by the simulation."})
        if sim.truth_peaks is not None:
            put("truthpeaks", sim.truth_peaks, {"Description": "Ground-truth fibre orientations: up to 3 peaks of the orientation mixture per voxel, volumes 3k..3k+2 = peak k as a unit vector in world RAS scaled by its mass fraction (0 = no peak). Object in its true (gradient-nonlinearity-free) frame."})
        if sim.gnl is not None:
            put("gnlcoeff", text=sim.gnl.coeff_text, ext=".grad")
            put("gnldisp", sim.gnl.disp, {"Description": "Ground-truth gradient-nonlinearity displacement d(r) = phi(r) - r at each voxel centre r, three volumes = RAS x,y,z in mm (apparent minus true position)."})
            put("gnlinvdisp", sim.gnl.invdisp, {"Description": "Inverse gradient-nonlinearity displacement phi^-1(x) - x at each voxel centre x, three volumes = RAS x,y,z in mm (true minus apparent position)."})
            put("gnldispitk", itk_displacement(sim.gnl.disp), {"Description": "The gradient-nonlinearity displacement d(r) = phi(r) - r as an ITK displacement field (5-D vector image, LPS mm, NIFTI_INTENT_VECTOR): at each output (true) point the vector to the input (apparent) point, i.e. the correction warp to pull the apparent image into the true frame."})
            put("gnlgraddev", sim.gnl.graddev, {
                "Description": "Ground-truth gradient deviation: 9 volumes, HCP/FSL layout; read row-major into T, the applied gradient in this image's voxel axes is T.T @ g; identity included.",
                "GradientNonlinearity": sim.gnl.spec, "GradientNonlinearityScale": sim.gnl.scale,
                "SpatialWarpApplied": sim.gnl.warp, "EncodingDeviationApplied": sim.gnl.encoding, "JacobianModulation": sim.gnl.jacobian,
            })
        if sim.dropout:
            put("dropout", text=sim.dropout_table(), ext=".tsv", meta={"Description": "Multiband shots dropped by within-volume motion: volume, shot, slices, attenuation."})
        return written

    def add_anat(
        self, img: nib.Nifti1Image, sub: str, ses: str | None = None, *, suffix: str = "T1w",
        sidecar: dict[str, Any] | None = None, offset: Any = None, offset_mode: str = "resample", **entities: Any,
    ) -> list[Path]:
        """Write an anatomical image to ``anat/`` with its sidecar (``sidecar`` keys plus the
        simulation stamp). ``offset`` (``(tx, ty, tz, rx, ry, rz)`` in mm and degrees, or a
        4x4 world transform) moves the subject between this scan and the DWI, by resampling
        the moved head into the scan's grid (``offset_mode="resample"``) or by rotating the
        header (``"header"``); the truth transform is written to the derivatives (see
        :meth:`add_offset_truth`)."""
        ent: dict[str, Any] = {"sub": self._label(sub, "sub"), "ses": self._label(ses, "ses")}
        ent.update({k: v for k, v in entities.items() if v is not None})
        out_dir = self.folder(sub, ses, "anat")
        stem = out_dir / bids_name(ent, suffix)
        meta = {"Manufacturer": "TRXScan", **(sidecar or {}), **simulation_stamp()}
        T = _offset_of(offset, img)
        files = []
        if T is not None:
            img = offset_image(img, T, offset_mode)
            files += self.add_offset_truth(sub, ses, "anat", suffix, T, **{k: v for k, v in ent.items() if k not in ("sub", "ses")})
        return [self._write_img(Path(f"{stem}.nii.gz"), img), self._write_json(Path(f"{stem}.json"), meta)] + files

    def add_offset_truth(self, sub: str, ses: str | None, datatype: str, from_label: str, T: np.ndarray, **entities: Any) -> list[Path]:
        """Record a between-scan movement ``T`` (world RAS, 4x4) as ground truth: an ITK text
        transform ``from-<label>_to-dwi`` that resamples the moved image onto the DWI frame,
        plus its JSON (the 4x4, and translation/rotation magnitudes)."""
        out_dir = self.folder(sub, ses, datatype, derivative=True)
        ent = {"sub": self._label(sub, "sub"), "ses": self._label(ses, "ses"), **entities, "from": from_label, "to": "dwi", "mode": "image", "desc": "truth"}
        stem = out_dir / bids_name(ent, "xfm")
        p = Path(f"{stem}.txt")
        p.write_text(itk_transform_text(T))
        self.written.append(p)
        R = T[:3, :3]
        meta = {
            "Description": f"Rigid movement of the subject between the DWI and this {from_label}: the ITK transform maps DWI-frame points to the same tissue in the {from_label} frame (what antsApplyTransforms needs to pull the {from_label} onto the DWI grid). WorldTransformRAS is the 4x4 applied to the {from_label} affine.",
            "WorldTransformRAS": T.tolist(), "RotationDeg": float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), "TranslationMm": float(np.linalg.norm(T[:3, 3])),
            **simulation_stamp(),
        }
        return [p, self._write_json(Path(f"{stem}.json"), meta)]

    def add_phantom_anat(
        self, phantom: "Phantom", sub: str, ses: str | None = None, *, suffixes: Sequence[str] = ("T1w", "T2w"),
        mode: str = "auto", zooms: Sequence[float] | None = None, axcodes: Sequence[str] | None = None,
        sidecar: dict[str, Any] | None = None, offset: Any = None, offset_mode: str = "resample", **entities: Any,
    ) -> list[Path]:
        """Write the phantom's T1w/T2w (or synthetic stand-ins, see :func:`phantom_anat`),
        resampled to ``zooms`` / reordered to ``axcodes`` when given, moved by ``offset``
        (see :meth:`add_anat`) when given."""
        out: list[Path] = []
        for suffix in suffixes:
            img, note = phantom_anat(phantom, suffix, mode)
            img = match_geometry(img, zooms, axcodes)
            meta = {"Description": note, **(sidecar or {})}
            out += self.add_anat(img, sub, ses, suffix=suffix, sidecar=meta, offset=offset, offset_mode=offset_mode, **entities)
        return out

    # -- checks --------------------------------------------------------------

    def validate(self, *args: str) -> tuple[bool, str]:
        """Run the ``bids-validator`` executable on the dataset if one is on PATH. Returns
        ``(ok, output)``; ``ok`` is None-like False with a note when no validator is found."""
        exe = shutil.which("bids-validator")
        if exe is None:
            return False, "bids-validator not found on PATH"
        r = subprocess.run([exe, str(self.root), *args], capture_output=True, text=True)
        return r.returncode == 0, (r.stdout + r.stderr)

    # -- mirroring a real dataset --------------------------------------------

    @classmethod
    def mirror(
        cls, source: str | Path, phantom: Any, out: str | Path, *, subjects: Iterable[str] | None = None,
        sessions: Iterable[str] | None = None, artifacts: "Artifacts | None" = None,
        protocol: Callable[[Protocol], Protocol] | None = None, anat: str | None = "auto", fmap: bool = True,
        sbref: bool = True, truth: bool = True, slices: Any = "all", kspace: bool = False, chunk: int | None = 8,
        log: Callable[[str], None] | None = print, name: str | None = None, **simulate_kw: Any,
    ) -> "Dataset":
        """Simulate every DWI run of a real BIDS dataset and write the result as a dataset of the
        same shape at ``out``: the same subjects, sessions, entities and file names; per run,
        the gradient table from its bval/bvec and a :class:`Protocol` from its sidecar and
        header (voxel size, matrix, TE, readout, PE polarity, partial Fourier, GRAPPA, MB, and
        the recorded-only TR/flip angle/slice timing/scanner descriptors). ``sbref`` runs are
        one b=0 volume under the same protocol; ``fmap/`` ``_epi`` runs are b=0 volumes with
        their own protocol, GRE (``phasediff``/``phase1``+``phase2``) fieldmaps are synthesised
        with the session's first DWI run at the source's echo times and resolution; other
        fieldmap types are skipped with a note. Anatomical images come from the phantom (or
        are synthetic, see :func:`phantom_anat`; ``anat=None`` skips them), resampled to the
        source's voxel size and axis order. No demographics are copied: ``participants.tsv``
        lists the ids only.

        ``phantom`` is a :class:`~trxscan.phantom.Phantom`, a ``{subject: Phantom}`` mapping
        or a callable of the subject label. ``protocol`` adjusts each derived protocol
        (``lambda p: p.replace(tissue="infant", coils=32)``). Remaining keywords go to
        :meth:`Object.simulate` (``context=``, ``progress=``, ...). ``slices`` restricts every
        run to a slab (for a quick look); the sidecars then describe the slab. ``chunk`` runs
        each volume ``chunk`` slices at a time so memory stays at a slab's worth.
        """
        from .artifacts import Artifacts, Gre
        from .phantom import Phantom

        src = Path(source)
        if not src.is_dir():
            raise FileNotFoundError(src)
        say = log or (lambda _s: None)
        arts = artifacts if artifacts is not None else Artifacts()
        src_desc = {}
        if (src / "dataset_description.json").exists():
            with open(src / "dataset_description.json") as f:
                src_desc = json.load(f)
        simulate_kw = {"chunk": chunk, **simulate_kw}
        ds = cls(Path(out), name=name or f"TRXScan simulation of {src_desc.get('Name', src.name)}",
                 description={"SourceDatasets": [{"URL": src.resolve().as_uri(), "Name": src_desc.get("Name", src.name)}]})
        want_subs = None if subjects is None else {f"sub-{cls._label(s, 'sub')}" for s in subjects}
        want_ses = None if sessions is None else {f"ses-{cls._label(s, 'ses')}" for s in sessions}
        subs = sorted(d.name for d in src.glob("sub-*") if d.is_dir() and (want_subs is None or d.name in want_subs))
        if not subs:
            raise ValueError(f"no subjects to mirror in {src}")

        def phantom_of(sub: str) -> Phantom:
            if isinstance(phantom, Phantom):
                return phantom
            if isinstance(phantom, dict):
                return phantom.get(sub) or phantom[cls._label(sub, "sub")]
            return phantom(sub)

        for sub in subs:
            ph = phantom_of(sub)
            ses_dirs = sorted(d.name for d in (src / sub).glob("ses-*") if d.is_dir() and (want_ses is None or d.name in want_ses))
            for ses in ses_dirs or [None]:
                base = src / sub / ses if ses else src / sub
                _mirror_session(ds, base, sub, ses, ph, arts, protocol, anat, fmap, sbref, truth, slices, kspace, say, simulate_kw, Gre)
            ds.add_participant(sub)
        return ds


def _runs_in(folder: Path, suffix: str) -> dict[str, dict[str | None, Path]]:
    """``{stem-without-part: {part: nifti}}`` for the ``_<suffix>.nii[.gz]`` files in a folder."""
    groups: dict[str, dict[str | None, Path]] = {}
    for p in sorted(list(folder.glob(f"*_{suffix}.nii.gz")) + list(folder.glob(f"*_{suffix}.nii"))):
        if p.name.startswith("."):  # macOS AppleDouble `._*` siblings and other dot-files
            continue
        ents, suf, _ = split_name(p)
        part = ents.pop("part", None)
        key = bids_name(ents, suf)
        groups.setdefault(key, {})[part] = p
    return groups


def _sidecar_of(nii: Path) -> dict[str, Any]:
    ents, suf, ext = split_name(nii)
    j = nii.with_name(nii.name[: -len(ext)] + ".json")
    if not j.exists():
        return {}
    with open(j) as f:
        return json.load(f)


def _mirror_session(ds, base, sub, ses, ph, arts, protocol, anat, fmap, sbref, truth, slices, kspace, say, simulate_kw, Gre) -> None:
    tag = f"{sub}" + (f"/{ses}" if ses else "")
    proto_fix = protocol or (lambda p: p)
    # -- anat
    if anat and (base / "anat").is_dir():
        for suffix in ("T1w", "T2w"):
            for key, parts in _runs_in(base / "anat", suffix).items():
                nii = parts.get(None) or parts.get("mag")
                if nii is None:
                    continue
                ents, _, _ = split_name(key)
                ents.pop("sub", None); ents.pop("ses", None)
                side = _sidecar_of(nii)
                hdr = nib.load(str(nii))
                meta = {k: side[k] for k in _ANAT_PASSTHROUGH if k in side}
                say(f"{tag}: anat {key}")
                ds.add_phantom_anat(ph, sub, ses, suffixes=(suffix,), mode=anat, zooms=hdr.header.get_zooms()[:3],
                                    axcodes=nib.aff2axcodes(hdr.affine), sidecar=meta, **ents)
    # -- dwi runs
    dwi_dir = base / "dwi"
    runs = _runs_in(dwi_dir, "dwi") if dwi_dir.is_dir() else {}
    gre_specs = _gre_specs(base / "fmap") if fmap and (base / "fmap").is_dir() else []
    first_dwi_files: list[Path] = []
    b0_field = gre_specs[0][1].b0_field if gre_specs else None
    for i, (key, parts) in enumerate(runs.items()):
        nii = parts.get("mag") or parts.get(None)
        if nii is None:
            continue
        real = BidsDwi.load(nii)
        proto = proto_fix(real.protocol)
        ents, _, _ = split_name(key)
        ents.pop("sub", None); ents.pop("ses", None)
        if not proto.fsl_orientation:
            codes = nib.aff2axcodes(nib.load(str(nii)).affine)
            warnings.warn(f"{nii.name}: source axes are {''.join(codes)}; the simulation is written in the phantom's native axes (only LAS is matched)", stacklevel=3)
        use_parts = ("mag", "phase") if "phase" in parts else ("mag",)
        dir_ent = ents.get("dir")
        gre_kw = {}
        if i == 0 and gre_specs:
            gre_kw["gre"] = gre_specs[0][1]
        say(f"{tag}: dwi {key} ({real.gtab[0].size if isinstance(real.gtab, tuple) else real.gtab.bvals.size} volumes, {proto.voxel[0]:g} mm, matrix {proto.matrix})")
        sim = ph.simulate(real.gtab, proto, arts, slices=slices, kspace=kspace, truth_peaks=truth, **gre_kw, **simulate_kw)
        files = ds.add_dwi(sim, sub, ses, parts=use_parts, truth=truth, b0_field=real.sidecar.get("B0FieldSource", b0_field),
                           b0_identifier=real.sidecar.get("B0FieldIdentifier"), fmap_entities=gre_specs[0][0] if gre_specs else None, dir=ents.pop("dir", None), **ents)
        first_dwi_files.append(files[0])
        if sbref:
            sb = _runs_in(dwi_dir, "sbref").get(key.replace("_dwi", "_sbref"))
            if sb:
                say(f"{tag}: sbref for {key}")
                sim_sb = ph.simulate(_b0_gtab(1), proto, arts, slices=slices, kspace=False, truth_peaks=False, **simulate_kw)
                ds.add_dwi(sim_sb, sub, ses, suffix="sbref", parts=("mag", "phase") if "phase" in sb else ("mag",), truth=False, b0_field=b0_field, dir=dir_ent, **ents)
    # -- PEPOLAR fieldmaps
    if fmap and (base / "fmap").is_dir():
        for key, parts in _runs_in(base / "fmap", "epi").items():
            nii = parts.get(None) or parts.get("mag")
            if nii is None:
                continue
            real = BidsDwi.load_epi(nii)
            proto = proto_fix(real.protocol)
            ents, _, _ = split_name(key)
            ents.pop("sub", None); ents.pop("ses", None)
            n = real.n_vol
            say(f"{tag}: fmap epi {key} ({n} b=0 volumes)")
            sim = ph.simulate(_b0_gtab(n), proto, arts, slices=slices, kspace=False, truth_peaks=False, **simulate_kw)
            # the source's IntendedFor, matched by file name to what was written; else every DWI
            wanted = {Path(str(p)).name for p in (real.sidecar.get("IntendedFor") or [])}
            targets = [p for p in first_dwi_files if p.name in wanted] or first_dwi_files
            ds.add_dwi(sim, sub, ses, suffix="epi", parts=("mag",), truth=False, intended_for=targets,
                       b0_field=real.sidecar.get("B0FieldIdentifier"), dir=ents.pop("dir", None), **ents)
        skipped = sorted({split_name(p)[1] for p in (base / "fmap").glob("*.nii*") if not p.name.startswith(".")} - {"epi", "magnitude1", "magnitude2", "phasediff", "phase1", "phase2"})
        if skipped:
            say(f"{tag}: fmap types not simulated: {', '.join(skipped)}")


def _gre_specs(fmap_dir: Path) -> list[tuple[dict[str, Any], Any]]:
    """``[(entities, Gre)]`` for the GRE fieldmaps in a source ``fmap/`` folder."""
    from .artifacts import Gre

    out: list[tuple[dict[str, Any], Gre]] = []
    seen: set[str] = set()
    for p in sorted(fmap_dir.glob("*_phasediff.nii*")) + sorted(fmap_dir.glob("*_phase1.nii*")):
        if p.name.startswith("."):
            continue
        ents, suf, _ = split_name(p)
        ents.pop("sub", None); ents.pop("ses", None)
        key = bids_name(ents, "gre")
        if key in seen:
            continue
        seen.add(key)
        side = _sidecar_of(p)
        hdr = nib.load(str(p))
        res = float(np.mean(hdr.header.get_zooms()[:3]))
        if suf == "phasediff":
            te = (float(side.get("EchoTime1", 4.92e-3)), float(side.get("EchoTime2", 7.38e-3)))
            output = "phasediff"
        else:
            p2 = p.with_name(p.name.replace("_phase1", "_phase2"))
            te2 = float(_sidecar_of(p2).get("EchoTime", 7.38e-3)) if p2.exists() else 7.38e-3
            te = (float(side.get("EchoTime", 4.92e-3)), te2)
            output = "phase"
        out.append((ents, Gre(te_s=te, res_mm=res, output=output, b0_field=str(side.get("B0FieldIdentifier", "b0gre")))))
    return out


def _b0_gtab(n: int) -> Any:
    bvals = np.zeros(n)
    bvecs = np.zeros((n, 3))
    try:
        from dipy.core.gradients import gradient_table

        return gradient_table(bvals, bvecs=bvecs)
    except ImportError:  # pragma: no cover
        return (bvals, bvecs)
