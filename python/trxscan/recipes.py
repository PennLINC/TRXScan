"""Named fixture generators: a small, fully specified BIDS dataset from one call (or the
``trxscan-fixture`` command), for pipeline tests that want known inputs and known truth
without a source dataset on disk. Each recipe writes a :class:`~trxscan.bids.Dataset` and a
``derivatives/trxscan/recipe.json`` whose :meth:`Recipe.digest` (the recipe, its parameters
and the trxscan version) is the cache key for the generated data::

    ds = trxscan.recipes.rpe_series("out/rpe", voxel=3.0, ndirs=16)
    $ trxscan-fixture rpe_series --out out/rpe --set voxel=3 --set ndirs=16

Recipes default to the hosted ``sub-60501`` phantom (``$TRXSCAN_DATA`` or the pooch cache;
``trxscan-fetch`` prewarms it) subsampled to 400k streamlines, 1 coil, no acceleration, LAS
output like a converted scan, and the truth files on by default.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .artifacts import Artifacts, Gre
from .bids import Dataset
from .motion import Motion
from .phantom import Phantom
from .protocol import Protocol, simulation_stamp

RECIPES: dict[str, Callable[..., Dataset]] = {}


@dataclass(frozen=True)
class Recipe:
    """What was generated: the recipe name, its parameters and the trxscan version."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return simulation_stamp()["SimulationSoftwareVersion"]

    def to_dict(self) -> dict[str, Any]:
        return {"Name": self.name, "Parameters": _jsonable(self.params), "SimulationSoftware": "TRXScan", "SimulationSoftwareVersion": self.version}

    def digest(self) -> str:
        """16 hex chars identifying this recipe + parameters + version (a cache key)."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    def write(self, ds: Dataset) -> Path:
        p = ds.deriv_root / "recipe.json"
        p.write_text(json.dumps({**self.to_dict(), "Digest": self.digest()}, indent=2) + "\n")
        return p


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, Phantom):
        return v.name or "<phantom>"
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, slice):
        return [v.start, v.stop, v.step]
    return v


def recipe(fn: Callable[..., Dataset]) -> Callable[..., Dataset]:
    RECIPES[fn.__name__] = fn
    return fn


# ─── building blocks ────────────────────────────────────────────────────────


def hemisphere_directions(n: int) -> np.ndarray:
    """``n`` unit vectors spread over a hemisphere (golden-section spiral, z >= 0):
    deterministic, no dipy needed."""
    i = np.arange(n) + 0.5
    z = i / n  # 0..1: one hemisphere
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def scheme(shells: Sequence[tuple[float, int]] = ((1000.0, 16),), nb0: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """``(bvals, bvecs)``: ``nb0`` b=0 volumes first, then each ``(bval, ndirs)`` shell."""
    bvals = [np.zeros(nb0)]
    bvecs = [np.zeros((nb0, 3))]
    for b, n in shells:
        bvals.append(np.full(n, float(b)))
        bvecs.append(hemisphere_directions(int(n)))
    return np.concatenate(bvals), np.concatenate(bvecs)


def load_phantom(phantom: str | Phantom, subsample: int | None, seed: int) -> Phantom:
    ph = phantom if isinstance(phantom, Phantom) else Phantom.load(phantom)
    if subsample and subsample < ph.n_streamlines:
        ph = ph.subsample(int(subsample), seed=seed)
    return ph


#: Echo spacing the recipes pin (ms per phase-encode line): HBCD's 91.7 ms train over 140 lines.
#: The readout then follows the matrix (a 3 mm scan gets a 3 mm readout), unlike the HBCD preset
#: itself, which pins the 91.7 ms train and would give coarse voxels unrealistic distortion.
ECHO_SPACING_MS = 91.7 / 140


def _protocol(voxel: float, coils: int, accel: int, mb: int, tr_s: float | None, pe: str, pad: Sequence[int] = (4, 14, 2), oversample: int = 2,
              echo_spacing_ms: float = ECHO_SPACING_MS, acq_rotation: Sequence[float] | None = None, **kw: Any) -> Protocol:
    return Protocol.HBCD.replace(voxel_mm=voxel, coils=coils, accel=accel, mb=mb, tr_s=tr_s, pe=pe, fsl_orientation=True,
                                 pad=tuple(int(v) for v in pad), oversample=int(oversample), readout_ms=None, echo_spacing_ms=float(echo_spacing_ms),
                                 oblique_deg=None if acq_rotation is None else tuple(float(v) for v in acq_rotation), **kw)


def _artifacts(noise: float, seed: int, gnl: str | None, gnl_scale: float, isocenter: Sequence[float] | None = None, **kw: Any) -> Artifacts:
    return Artifacts(noise=noise, seed=seed, gnl=gnl, gnl_scale=gnl_scale, isocenter=None if isocenter is None else tuple(float(v) for v in isocenter), **kw)


def _new(out: str | Path, name: str, params: dict[str, Any]) -> tuple[Dataset, Recipe]:
    rec = Recipe(name, params)
    ds = Dataset(out, name=f"TRXScan fixture {name}", description={"Recipe": rec.to_dict()})
    return ds, rec


# ─── recipes ────────────────────────────────────────────────────────────────


@recipe
def rpe_series(
    out: str | Path, *, phantom: str | Phantom = "sub-60501", subject: str = "trxscan", voxel: float = 3.0, ndirs: int = 16,
    nb0: int = 2, bval: float = 1000.0, pe: Sequence[str] = ("AP", "PA"), subsample: int | None = 400_000, noise: float = 0.0,
    coils: int = 1, accel: int = 1, mb: int = 1, tr_s: float | None = 4.0, gnl: str | None = None, gnl_scale: float = 1.0,
    gnl_tag: str = "ND", pa_offset: Sequence[float] | None = None, parts: Sequence[str] | None = None, anat: bool = True, truth: bool = True, slices: Any = "all",
    pad: Sequence[int] = (4, 14, 2), oversample: int = 2, echo_spacing_ms: float = ECHO_SPACING_MS, chunk: int | None = 8, anat_offset: Sequence[float] | None = None, acq_rotation: Sequence[float] | None = None,
    isocenter: Sequence[float] | None = None, seed: int = 0,
) -> Dataset:
    """A blip-up/blip-down pair: one single-shell DWI series per phase-encode polarity in
    ``pe`` with the phantom's measured fieldmap (``dir-AP``/``dir-PA``), magnitude only by
    default (``parts=("mag", "phase")`` for complex). ``gnl`` adds gradient nonlinearity
    (``"whole-body-80"``, ``"connectom-300"`` or a coefficient file) and writes its truth;
    ``gnl_tag`` sets the ``ImageType`` claim. ``pa_offset`` moves the subject between the two
    polarities: every run after the first is re-simulated with the moved head inside the first
    run's field of view (``Phantom.moved`` + ``grid(like=)``), and the movement is written as a
    truth transform, so run alignment and the bvec rotation it implies can be scored. ``pad`` (voxels per side, the phase-encode axis
    generous so distortion cannot wrap) and ``oversample`` size the grid: ``pad=(4, 6, 2),
    oversample=1`` is the lean variant for pipeline tests. ``echo_spacing_ms`` fixes the line time,
    so ``TotalReadoutTime`` follows the matrix. ``chunk`` simulates ``chunk`` slices
    at a time (memory stays at a slab's worth; the result is the whole-volume result).
    ``anat_offset`` / ``fmap_offset`` ``(tx, ty, tz, rx, ry, rz)`` (mm, degrees) move the subject
    between the DWI and the anatomical / fieldmap scans (the moved head is resampled into the
    scan's own grid, so the movement is in the voxels, not the header); the truth transform is
    written as an ITK text file in the derivatives, so registration can be scored against a
    non-identity. ``acq_rotation`` ``(rx, ry, rz)`` tilts the DWI acquisition grid about its
    centre (an oblique acquisition: the header carries the rotation, the head does not move).
    ``isocenter`` (world mm) places the scanner isocenter, which matters with ``gnl``. Truth: clean b=0, fieldmap,
    displacement, fibre peaks, GNL fields."""
    params = {k: v for k, v in locals().items() if k != "out"}
    ds, rec = _new(out, "rpe_series", params)
    ph = load_phantom(phantom, subsample, seed)
    bvals, bvecs = scheme(((bval, ndirs),), nb0)
    first_obj = None
    for i, d in enumerate(pe):
        proto = _protocol(voxel, coils, accel, mb, tr_s, d, pad, oversample, echo_spacing_ms, acq_rotation, gnl_tag=gnl_tag)
        if i > 0 and pa_offset is not None:
            # the subject moved between the runs: the moved head inside the FIRST run's field of view
            obj = ph.moved(pa_offset).grid(proto, like=first_obj)
            sim = obj.simulate((bvals, bvecs), proto, _artifacts(noise, seed, gnl, gnl_scale, isocenter), slices=slices, kspace=False, chunk=chunk, truth_peaks=truth, clean_b0=truth)
            files = ds.add_dwi(sim, subject, parts=parts, truth=truth, dir=d)
            from .bids import _offset_of
            ds.add_offset_truth(subject, None, "dwi", f"dir{d}", _offset_of(pa_offset, ph.wm), dir=d)
        else:
            obj = ph.grid(proto)
            first_obj = first_obj or obj
            sim = obj.simulate((bvals, bvecs), proto, _artifacts(noise, seed, gnl, gnl_scale, isocenter), slices=slices, kspace=False, chunk=chunk, truth_peaks=truth, clean_b0=truth)
            ds.add_dwi(sim, subject, parts=parts, truth=truth, dir=d)
    if anat:
        ds.add_phantom_anat(ph, subject, offset=anat_offset)
    rec.write(ds)
    return ds


@recipe
def epi_fieldmap(
    out: str | Path, *, phantom: str | Phantom = "sub-60501", subject: str = "trxscan", voxel: float = 3.0, ndirs: int = 16,
    nb0: int = 2, bval: float = 1000.0, pe: str = "AP", nb0_epi: int = 2, fmap_offset: Sequence[float] | None = None, subsample: int | None = 400_000, noise: float = 0.0,
    coils: int = 1, accel: int = 1, mb: int = 1, tr_s: float | None = 4.0, gnl: str | None = None, gnl_scale: float = 1.0,
    parts: Sequence[str] | None = None, anat: bool = True, truth: bool = True, slices: Any = "all", pad: Sequence[int] = (4, 14, 2), oversample: int = 2, echo_spacing_ms: float = ECHO_SPACING_MS, chunk: int | None = 8, anat_offset: Sequence[float] | None = None, acq_rotation: Sequence[float] | None = None,
    isocenter: Sequence[float] | None = None, seed: int = 0,
) -> Dataset:
    """One DWI series plus a PEPOLAR ``fmap/_epi`` of ``nb0_epi`` b=0 volumes with the opposite
    polarity, linked by ``IntendedFor`` and ``B0FieldIdentifier``."""
    params = {k: v for k, v in locals().items() if k != "out"}
    ds, rec = _new(out, "epi_fieldmap", params)
    ph = load_phantom(phantom, subsample, seed)
    bvals, bvecs = scheme(((bval, ndirs),), nb0)
    rev = "PA" if pe == "AP" else "AP"
    arts = _artifacts(noise, seed, gnl, gnl_scale, isocenter)
    sim = ph.simulate((bvals, bvecs), _protocol(voxel, coils, accel, mb, tr_s, pe, pad, oversample, echo_spacing_ms, acq_rotation), arts, slices=slices, kspace=False, chunk=chunk, truth_peaks=truth, clean_b0=truth)
    files = ds.add_dwi(sim, subject, parts=parts, truth=truth, dir=pe, b0_field="pepolar", b0_identifier="pepolar")
    epi = ph.simulate(scheme((), nb0_epi), _protocol(voxel, coils, accel, mb, tr_s, rev, pad, oversample, echo_spacing_ms, acq_rotation), arts, slices=slices, kspace=False, chunk=chunk, truth_peaks=False, clean_b0=False)
    ds.add_dwi(epi, subject, suffix="epi", parts=("mag",), truth=False, intended_for=[files[0]], b0_field="pepolar", dir=rev, fmap_offset=fmap_offset)
    if anat:
        ds.add_phantom_anat(ph, subject, offset=anat_offset)
    rec.write(ds)
    return ds


@recipe
def phasediff_fieldmap(
    out: str | Path, *, phantom: str | Phantom = "sub-60501", subject: str = "trxscan", voxel: float = 3.0, ndirs: int = 16,
    nb0: int = 2, bval: float = 1000.0, pe: str = "AP", te_s: Sequence[float] = (4.92e-3, 7.38e-3), gre_res_mm: float | None = None,
    gre_snr: float = 50.0, output: str = "phasediff", fmap_offset: Sequence[float] | None = None, subsample: int | None = 400_000, noise: float = 0.0, coils: int = 1,
    accel: int = 1, mb: int = 1, tr_s: float | None = 4.0, parts: Sequence[str] | None = None, anat: bool = True,
    truth: bool = True, slices: Any = "all", pad: Sequence[int] = (4, 14, 2), oversample: int = 2, echo_spacing_ms: float = ECHO_SPACING_MS, chunk: int | None = 8, anat_offset: Sequence[float] | None = None, acq_rotation: Sequence[float] | None = None,
    isocenter: Sequence[float] | None = None, seed: int = 0,
) -> Dataset:
    """One DWI series plus a synthetic dual-echo GRE fieldmap (``phasediff`` or
    ``phase1``/``phase2`` with ``output="phase"``) of the same object in ``fmap/``."""
    params = {k: v for k, v in locals().items() if k != "out"}
    ds, rec = _new(out, "phasediff_fieldmap", params)
    ph = load_phantom(phantom, subsample, seed)
    bvals, bvecs = scheme(((bval, ndirs),), nb0)
    gre = Gre(te_s=tuple(float(t) for t in te_s), snr=gre_snr, res_mm=gre_res_mm, output=output)
    sim = ph.simulate((bvals, bvecs), _protocol(voxel, coils, accel, mb, tr_s, pe, pad, oversample, echo_spacing_ms, acq_rotation), _artifacts(noise, seed, None, 1.0, isocenter), slices=slices, kspace=False, chunk=chunk, truth_peaks=truth, clean_b0=truth, gre=gre)
    ds.add_dwi(sim, subject, parts=parts, truth=truth, dir=pe, fmap_entities={"acq": "gre"}, fmap_offset=fmap_offset)
    if anat:
        ds.add_phantom_anat(ph, subject, offset=anat_offset)
    rec.write(ds)
    return ds


@recipe
def multishell(
    out: str | Path, *, phantom: str | Phantom = "sub-60501", subject: str = "trxscan", voxel: float = 3.0,
    shells: Sequence[Sequence[float]] = ((1000.0, 16), (2000.0, 32)), nb0: int = 4, pe: Sequence[str] = ("AP",),
    subsample: int | None = 400_000, noise: float = 0.0, coils: int = 1, accel: int = 1, mb: int = 1, tr_s: float | None = 4.0,
    parts: Sequence[str] | None = None, anat: bool = True, truth: bool = True, slices: Any = "all", pad: Sequence[int] = (4, 14, 2), oversample: int = 2, echo_spacing_ms: float = ECHO_SPACING_MS, chunk: int | None = 8, anat_offset: Sequence[float] | None = None, acq_rotation: Sequence[float] | None = None,
    isocenter: Sequence[float] | None = None, seed: int = 0,
) -> Dataset:
    """A multi-shell series (``shells`` = ``[(bval, ndirs), ...]``) per polarity in ``pe``,
    for reconstruction tests scored against the fibre peaks and clean b=0."""
    params = {k: v for k, v in locals().items() if k != "out"}
    ds, rec = _new(out, "multishell", params)
    ph = load_phantom(phantom, subsample, seed)
    bvals, bvecs = scheme([(float(b), int(n)) for b, n in shells], nb0)
    for d in pe:
        sim = ph.simulate((bvals, bvecs), _protocol(voxel, coils, accel, mb, tr_s, d, pad, oversample, echo_spacing_ms, acq_rotation), _artifacts(noise, seed, None, 1.0, isocenter), slices=slices, kspace=False, chunk=chunk, truth_peaks=truth, clean_b0=truth)
        ds.add_dwi(sim, subject, parts=parts, truth=truth, dir=d)
    if anat:
        ds.add_phantom_anat(ph, subject, offset=anat_offset)
    rec.write(ds)
    return ds


@recipe
def motion(
    out: str | Path, *, phantom: str | Phantom = "sub-60501", subject: str = "trxscan", voxel: float = 3.0, ndirs: int = 16,
    nb0: int = 2, bval: float = 1000.0, pe: str = "AP", trace: str = "AP", scale: float = 1.0, dropout: float = 0.0, mb: int = 3,
    subsample: int | None = 400_000, noise: float = 0.0, coils: int = 1, accel: int = 1, tr_s: float | None = 4.0,
    parts: Sequence[str] | None = None, anat: bool = True, slices: Any = "all", pad: Sequence[int] = (4, 14, 2), oversample: int = 2, echo_spacing_ms: float = ECHO_SPACING_MS, chunk: int | None = 8, anat_offset: Sequence[float] | None = None, acq_rotation: Sequence[float] | None = None,
    isocenter: Sequence[float] | None = None, seed: int = 0,
) -> Dataset:
    """One DWI series acquired under the phantom's measured head-motion trace ``trace``
    (``phantom.motion[trace]``, scaled by ``scale``), re-simulated per volume so fibre-gradient
    angles move with the head; ``dropout`` adds multiband shot dropout (needs ``mb > 1``). The
    applied motion is written as ``desc-motion_timeseries.tsv`` next to the truth."""
    params = {k: v for k, v in locals().items() if k != "out"}
    ds, rec = _new(out, "motion", params)
    ph = load_phantom(phantom, subsample, seed)
    if trace not in ph.motion:
        raise ValueError(f"phantom {ph.name!r} has no motion trace {trace!r} (has {sorted(ph.motion)})")
    bvals, bvecs = scheme(((bval, ndirs),), nb0)
    mot = scaled_motion(ph.motion[trace], scale, n_vol=bvals.size)
    arts = Artifacts(noise=noise, seed=seed, motion=mot, dropout=dropout, isocenter=None if isocenter is None else tuple(float(v) for v in isocenter))
    sim = ph.simulate((bvals, bvecs), _protocol(voxel, coils, accel, mb, tr_s, pe, pad, oversample, echo_spacing_ms, acq_rotation), arts, slices=slices, kspace=False, chunk=chunk, truth_peaks=False, clean_b0=True)
    ds.add_dwi(sim, subject, parts=parts, truth=True, dir=pe)
    stem = ds.folder(subject, None, "dwi", derivative=True) / f"sub-{subject}_dir-{pe}_desc-motion"
    Path(f"{stem}_timeseries.tsv").write_text(motion_tsv(mot, sim.n_vol, seed))
    Path(f"{stem}_timeseries.json").write_text(json.dumps({
        "Description": "Head pose applied to each volume (the object moved before the acquisition): translations in mm and rotations in radians about the RAS axes, the qsiprep/fMRIPrep confounds convention Motion.from_confounds reads.",
        **simulation_stamp()}, indent=2) + "\n")
    if anat:
        ds.add_phantom_anat(ph, subject, offset=anat_offset)
    rec.write(ds)
    return ds


def scaled_motion(mot: Motion, scale: float, n_vol: int) -> Motion:
    """``mot`` with every translation and rotation multiplied by ``scale``, resampled to
    ``n_vol`` volumes when it is a shorter trajectory."""
    if mot.kind == "trajectory":
        assert mot.poses is not None
        if mot.poses.shape[0] < n_vol:
            mot = mot.resample(n_vol)
        if scale == 1.0:
            return mot
        return Motion.from_arrays(mot.poses[:, :3] * scale, mot.poses[:, 3:] * scale, source=f"{mot.source} x{scale:g}")
    if scale == 1.0:
        return mot
    return dataclasses.replace(mot, trans_mm=tuple(v * scale for v in mot.trans_mm), rot_deg=tuple(v * scale for v in mot.rot_deg))


def motion_tsv(mot: Motion, n_vol: int, seed: int) -> str:
    """The applied poses as a confounds-style TSV (``trans_x..rot_z``, mm and radians)."""
    poses = np.asarray(mot.poses_for(n_vol, seed=seed), dtype=float).reshape(n_vol, 6)
    poses = np.hstack([poses[:, :3], np.radians(poses[:, 3:])])
    cols = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    return "\t".join(cols) + "\n" + "\n".join("\t".join(f"{v:.6f}" for v in row) for row in poses) + "\n"


# ─── command line ───────────────────────────────────────────────────────────


def _parse_value(v: str) -> Any:
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="trxscan-fixture", description="Generate a BIDS test fixture from a named TRXScan recipe.")
    ap.add_argument("recipe", nargs="?", choices=sorted(RECIPES), help="recipe name")
    ap.add_argument("--out", type=Path, help="dataset root to write")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="recipe parameter (JSON values: --set ndirs=16 --set pe='[\"AP\"]')")
    ap.add_argument("--list", action="store_true", help="list recipes and exit")
    ap.add_argument("--digest", action="store_true", help="print the cache digest for the recipe + parameters and exit (no simulation)")
    a = ap.parse_args(argv)
    if a.list or a.recipe is None:
        for name, fn in sorted(RECIPES.items()):
            print(f"{name}: {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0
    params = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        if not _:
            ap.error(f"--set expects KEY=VALUE, got {kv!r}")
        params[k] = _parse_value(v)
    if a.digest:
        print(Recipe(a.recipe, params).digest())
        return 0
    if a.out is None:
        ap.error("--out is required")
    ds = RECIPES[a.recipe](a.out, **params)
    print(f"{ds.root}  digest {Recipe(a.recipe, params).digest()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
