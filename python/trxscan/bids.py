"""BIDS in and out: :func:`write_bids` (everything the ``trxscan`` CLI writes for a run) and
:class:`BidsDwi` (a real acquisition's gradient table and protocol from its sidecar)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .protocol import Protocol

if TYPE_CHECKING:  # pragma: no cover
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
