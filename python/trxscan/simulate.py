"""The orchestration behind :meth:`Object.simulate` and :meth:`Object.microstructure`, and
the :class:`Simulation` result. Mirrors the ``trxscan`` binary's ``main()`` step for step."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import nibabel as nib
import numpy as np

from . import _core
from ._arrays import affine4, as_gtab, b_max_of, complex_pairs, flat_f32, nifti, series, vectors, volume
from .artifacts import Artifacts, Gre
from .kspace import KSpace
from .protocol import EpiReadout, Protocol, _readout_from_acquisition

if TYPE_CHECKING:  # pragma: no cover
    from .phantom import Object


class TrxscanError(RuntimeError):
    """Raised when the simulation core fails (wraps a Rust panic or a core error)."""


Progress = Callable[[str, int, int], None]


@dataclass(eq=False)
class GreResult:
    """A synthesized GRE fieldmap: ``magnitude1``/``magnitude2`` and either ``phasediff`` or
    ``phase1``/``phase2`` (Siemens int16 0..4095) as NIfTI images, plus the sidecar fields."""

    magnitude1: nib.Nifti1Image
    magnitude2: nib.Nifti1Image
    phase: dict[str, nib.Nifti1Image]
    te_s: tuple[float, float]
    output: str
    b0_field: str
    sigma: float
    stamped: bool


@dataclass(eq=False)
class GnlResult:
    """Gradient-nonlinearity ground truth on the written grid: the Siemens-format coefficient
    text, forward/inverse displacement (RAS mm, 3 volumes) and the 9-volume graddev image."""

    coeff_text: str
    disp: nib.Nifti1Image
    invdisp: nib.Nifti1Image
    graddev: nib.Nifti1Image
    spec: str
    scale: float
    warp: bool
    encoding: bool
    jacobian: bool


@dataclass(eq=False)
class DroppedShot:
    volume: int
    shot: int
    slices: tuple[int, ...]
    attenuation: float


@dataclass(eq=False)
class Simulation:
    """The result of :meth:`Object.simulate`: the complex image on the acquired slices (as
    ``Nifti1Image`` views with the slices' true affine), the gradient table in both frames, the
    BIDS sidecar, and whatever k-space and ground truth was requested."""

    re: np.ndarray
    im: np.ndarray
    mag: np.ndarray
    ph: np.ndarray
    dims: tuple[int, int, int]
    affine: np.ndarray
    slices: np.ndarray
    bvals: np.ndarray
    bvecs: np.ndarray
    bvecs_fsl: np.ndarray
    protocol: Protocol
    artifacts: Artifacts
    sidecar: dict[str, Any]
    readout: EpiReadout
    kspace: KSpace | None = None
    truth_peaks: nib.Nifti1Image | None = None
    gre: GreResult | None = None
    gnl: GnlResult | None = None
    dropout: list[DroppedShot] = field(default_factory=list)
    noise_sigma: nib.Nifti1Image | None = None
    mixture: Any = None
    compartments: Any = None
    object: Any = None
    timing: dict[str, float] = field(default_factory=dict)

    # -- images --------------------------------------------------------------

    @property
    def n_vol(self) -> int:
        return int(self.bvals.size)

    def _image(self, flat: np.ndarray) -> nib.Nifti1Image:
        img = nifti(series(flat, self.dims, self.n_vol), self.affine)
        if self.protocol.tr_s is not None:
            img.header.set_zooms(tuple(img.header.get_zooms()[:3]) + (float(self.protocol.tr_s),))
            img.header.set_xyzt_units("mm", "sec")
        return img

    @property
    def magnitude(self) -> nib.Nifti1Image:
        return self._image(self.mag)

    @property
    def phase(self) -> nib.Nifti1Image:
        return self._image(self.ph)

    @property
    def real(self) -> nib.Nifti1Image:
        return self._image(self.re)

    @property
    def imag(self) -> nib.Nifti1Image:
        return self._image(self.im)

    @property
    def complex(self) -> np.ndarray:
        """``(nx, ny, nz, n_vol)`` complex64."""
        return series(self.re, self.dims, self.n_vol) + 1j * series(self.im, self.dims, self.n_vol)

    def series(self, voxel: tuple[int, int, int] | tuple[int, int], kind: str = "magnitude") -> np.ndarray:
        """One voxel's series across volumes (``kind``: magnitude, phase, complex)."""
        x, y = voxel[0], voxel[1]
        z = voxel[2] if len(voxel) > 2 else 0
        if kind == "magnitude":
            return series(self.mag, self.dims, self.n_vol)[x, y, z]
        if kind == "phase":
            return series(self.ph, self.dims, self.n_vol)[x, y, z]
        if kind == "complex":
            return self.complex[x, y, z]
        raise ValueError("kind must be magnitude, phase or complex")

    def dropout_table(self) -> str:
        """The ``_desc-dropout_slices.tsv`` text the CLI writes."""
        out = "volume\tshot\tslices\tattenuation\n"
        for d in self.dropout:
            out += f"{d.volume}\t{d.shot}\t{','.join(str(s) for s in d.slices)}\t{d.attenuation:.3f}\n"
        return out

    def to_bids(self, prefix: str | Path) -> list[Path]:
        """Write everything the CLI writes for ``--out <prefix>``. See :func:`trxscan.bids.write_bids`."""
        from .bids import write_bids

        return write_bids(self, prefix)


# ─── helpers ────────────────────────────────────────────────────────────────


def _resolve_slices(slices: Any, z_mm: float | None, obj: "Object") -> np.ndarray:
    nz = obj.dims[2]
    if z_mm is not None:
        if slices != "all":
            raise ValueError("give slices or z_mm, not both")
        return np.array([obj.z_of(z_mm)])
    if isinstance(slices, str):
        if slices != "all":
            raise ValueError("slices must be 'all', an int, a slice/range or a contiguous sequence")
        return np.arange(nz)
    if isinstance(slices, (int, np.integer)):
        z = int(slices)
        if z < 0:
            z += nz
        if not 0 <= z < nz:
            raise ValueError(f"slice {slices} out of range for {nz} slices")
        return np.array([z])
    if isinstance(slices, slice):
        return np.arange(nz)[slices]
    arr = np.array(sorted({int(s) for s in slices}))
    if arr.size == 0 or arr[0] < 0 or arr[-1] >= nz:
        raise ValueError(f"slices {slices} out of range for {nz} slices")
    if not np.all(np.diff(arr) == 1):
        raise ValueError("slices must be contiguous (an image needs one affine); run separate simulations otherwise")
    return arr


def _coef_from(gnl: str, scale: float) -> tuple[_core.GradCoef, str]:
    p = Path(gnl)
    if p.exists():
        coef = _core.GradCoef.parse_siemens(p.read_text())
        spec = str(p)
    elif "\n" in gnl:
        coef = _core.GradCoef.parse_siemens(gnl)
        spec = "<text>"
    else:
        coef = _core.GradCoef.from_spec(gnl)
        spec = gnl
    if scale != 1.0:
        coef = coef.scaled(scale)
    return coef, spec


def _flat_or_none(img: Any, dims: tuple[int, int, int], what: str) -> np.ndarray | None:
    if img is None:
        return None
    v = flat_f32(img) if (hasattr(img, "dataobj") or np.ndim(img) == 3) else np.asarray(img, np.float32).reshape(-1)
    if v.size != int(np.prod(dims)):
        raise ValueError(f"{what} has {v.size} voxels, expected {int(np.prod(dims))} for grid {dims}")
    return v


def _voxel_to_world(affine: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    return ijk @ affine[:3, :3].T + affine[:3, 3]


# ─── run ────────────────────────────────────────────────────────────────────


def run(
    obj: "Object",
    gtab: Any,
    protocol: Protocol = Protocol.DEFAULT,
    artifacts: Artifacts = Artifacts(),
    *,
    slices: Any = "all",
    z_mm: float | None = None,
    context: int | None = None,
    kspace: bool | None = None,
    kspace_capture: Sequence[str] = ("acquired", "reconstructed"),
    gre: Gre | None = None,
    truth_peaks: bool = False,
    progress: Progress | None = None,
) -> Simulation:
    """Simulate ``obj`` under ``gtab``/``protocol``/``artifacts`` on the requested slices.

    ``slices`` is ``"all"``, one local slice index, a ``slice``/``range``, or a contiguous
    sequence; ``z_mm`` picks the slice nearest a world z. ``context`` is how many slices each
    side are simulated as context for motion, dropout jumps and the GNL warp (auto: 3 when
    needed, else 0). ``kspace`` captures per-coil k-space for the requested slices (default: when
    at most 8 slices are requested); ``kspace_capture`` picks ``"acquired"``,
    ``"reconstructed"``, ``"coil_images"``. ``gre`` also synthesizes a GRE fieldmap;
    ``truth_peaks`` writes up to three ground-truth fibre peaks per voxel. ``progress`` is
    called as ``progress(stage, done, total)``.
    """
    t_start = time.perf_counter()
    timing: dict[str, float] = {}
    bvals, bvecs, _, _ = as_gtab(gtab)
    ngrad = int(bvals.size)
    b_max = b_max_of(bvals)
    nx, ny, nz = obj.dims
    o = obj.oversample
    say = progress or (lambda *_: None)

    sel = _resolve_slices(slices, z_mm, obj)
    ctx = int(context) if context is not None else (3 if artifacts.needs_context else 0)
    z0, z1 = max(0, int(sel[0]) - ctx), min(nz, int(sel[-1]) + ctx + 1)
    sub = obj.slab(z0, z1)
    local = [int(z) - z0 for z in sel]
    global_z = [obj.z_offset + int(z) for z in sel]
    nz_full = int(obj.nz_full or nz)
    sub_global_z = [obj.z_offset + z for z in range(z0, z1)]
    if kspace is None:
        kspace = len(sel) <= 8

    # Streamlines: keep only those entering the slab (+ margin), cheap and exact for rasterization.
    positions = offsets = weights = None
    if sub.streamlines is not None:
        sl = sub.streamlines
        if (z0, z1) != (0, nz):
            margin = 5.0 / max(sub.voxel_mm[2], 1e-6)
            sl = sl.crop_k(obj.affine, z0 - margin, z1 + margin)
        positions, offsets, weights = sl.positions, sl.offsets, sl.weights

    acq_affine = sub.affine.copy()
    sim_affine = sub.sim_affine.copy()
    if artifacts.isocenter is not None:
        iso = np.asarray(artifacts.isocenter, dtype=np.float64)
        acq_affine[:3, 3] -= iso
        sim_affine[:3, 3] -= iso
        if positions is not None:
            positions = positions - iso

    # Gradient nonlinearity on the slab's simulation grid.
    gnl_field = None
    coef = None
    gnl_spec = ""
    if artifacts.gnl is not None:
        if artifacts.motion is not None:
            raise ValueError("gnl is not supported together with motion (the per-segment motion path has no mixture)")
        coef, gnl_spec = _coef_from(artifacts.gnl, artifacts.gnl_scale)
        t = time.perf_counter()
        gnl_field = _core.GnlField.on_grid(coef, sub.sim_dims, sim_affine)
        timing["gnl_field"] = time.perf_counter() - t

    params = protocol.tissue.to_params(b_max, protocol.diff_scale)
    seed = int(artifacts.seed)

    # ---- signal stage ----
    t = time.perf_counter()
    say("signal", 0, 1)
    mixture = None
    peaks_flat = None
    if artifacts.motion is not None:
        if truth_peaks:
            raise ValueError("truth_peaks is not available together with motion (no orientation mixture)")
        if sub.myelin is not None:
            raise ValueError("a myelin map is not supported together with motion")
        if positions is None:
            raise ValueError("the motion path needs streamlines; synthetic fibre objects cannot move (yet)")
        if weights is not None or protocol.kappa is not None:
            warnings.warn("weights and kappa are ignored in motion mode (per-segment re-simulation)", stacklevel=2)
        poses = artifacts.motion.poses_for(ngrad, seed=seed)
        comp = _core.signal_moving(
            sub.sim_dims, sub.sim_wm, sub.sim_gm, sub.sim_csf, sub.sim_mask, sim_affine,
            positions, offsets, bvals, bvecs, params, np.ascontiguousarray(poses),
        )
    else:
        if sub.fibers is not None:
            mixture = _core.mixture_from_fibers(
                sub.sim_dims, sub.fibers.voxel, sub.fibers.dirs, sub.fibers.weight,
                sub.sim_wm, sub.sim_gm, sub.sim_csf, sub.sim_mask, params, protocol.kappa,
            )
        elif positions is not None:
            mixture = _core.build_mixture(
                sub.sim_dims, sub.sim_wm, sub.sim_gm, sub.sim_csf, sub.sim_mask, sim_affine,
                positions, offsets, weights, params, protocol.kappa, sub.myelin,
            )
        else:
            raise ValueError("the object has neither streamlines nor fibres")
        if sub.myelin is not None and sub.fibers is not None:
            mixture = mixture.with_myelin(sub.myelin)
        if truth_peaks:
            peaks_flat = mixture.truth_peaks(o, 3)
        comp = _core.signal_from_mixture(mixture, bvals, bvecs, gnl_field if artifacts.gnl_encoding else None)
    comp.apply_s0(tuple(float(v) for v in protocol.tissue_s0))
    timing["signal"] = time.perf_counter() - t
    say("signal", 1, 1)

    # ---- multiband dropout ----
    dropped: list[DroppedShot] = []
    if protocol.mb > 1 and artifacts.dropout > 0.0:
        n_shots = max(nz_full // protocol.mb, 1)
        events = _core.dropout_events(bvals, n_shots, float(artifacts.dropout), _core.dropout_seed(seed))
        gt = comp.apply_multiband_motion(sim_affine, int(protocol.mb), True, bvals, b_max, events, sub_global_z, nz_full)
        dropped = [DroppedShot(int(d["volume"]), int(d["shot"]), tuple(int(s) for s in d["slices"]), float(d["attenuation"])) for d in gt]

    # ---- GNL spatial warp ----
    fmap_sim = sub.sim_fmap
    if gnl_field is not None and artifacts.gnl_warp:
        t = time.perf_counter()
        comp.warp_gnl(gnl_field, bool(artifacts.gnl_jacobian))
        fmap_sim = gnl_field.warp_volume(fmap_sim, False)
        timing["gnl_warp"] = time.perf_counter() - t

    # ---- GRE fieldmap ----
    gre_result = None
    if gre is not None:
        t = time.perf_counter()
        g = _core.gre_synthesize(
            sub.sim_dims, sim_affine, sub.dims, acq_affine, sub.sim_wm, sub.sim_gm, sub.sim_csf,
            tuple(float(v) for v in protocol.tissue_s0), list(comp.t2), fmap_sim, float(protocol.signal_scale), seed,
            te_s=tuple(gre.te_s), snr=gre.snr, res_mm=gre.res_mm, snr_vol_exp=gre.snr_vol_exp, output=gre.output,
            rx_phase_rad=gre.rx_phase_rad, b0_field=gre.b0_field,
            warp=gnl_field if (gnl_field is not None and artifacts.gnl_warp) else None, warp_modulate=bool(artifacts.gnl_jacobian),
        )
        gdims = tuple(int(v) for v in g["dims"])
        gaff = g["affine"].reshape(4, 4)
        # On the acquisition grid (no res_mm) keep only the requested slices, like the DWI.
        gsel = local if (gdims == tuple(sub.dims) and local != list(range(z1 - z0))) else None

        def gvol(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            v = volume(flat, gdims)
            if gsel is None:
                return v, gaff
            a = gaff.copy()
            a[:3, 3] += gaff[:3, 2] * gsel[0]
            return np.ascontiguousarray(v[:, :, gsel[0] : gsel[-1] + 1]), a

        phases = {}
        names = ["phasediff"] if g["output"] == "phasediff" else ["phase1", "phase2"]
        for name, arr in zip(names, g["phase"]):
            v, a = gvol(arr)
            phases[name] = nifti(v, a, dtype=np.int16)
        m1, a1 = gvol(g["magnitude1"])
        m2, a2 = gvol(g["magnitude2"])
        gre_result = GreResult(
            magnitude1=nifti(m1, a1), magnitude2=nifti(m2, a2),
            phase=phases, te_s=tuple(g["te_s"]), output=g["output"], b0_field=g["b0_field"], sigma=float(g["sigma"]), stamped=bool(g["stamped"]),
        )
        timing["gre"] = time.perf_counter() - t

    # ---- acquisition on the requested slices ----
    all_local = list(range(z1 - z0))
    comp_sel = comp if local == all_local else comp.select_slices(local)
    snx, sny, _ = sub.sim_dims
    fmap_sel = fmap_sim if local == all_local else np.concatenate([fmap_sim[z * snx * sny : (z + 1) * snx * sny] for z in local])
    nsl = len(local)
    noise_sigma = _flat_or_none(artifacts.noise_map, obj.dims, "noise_map") if artifacts.noise_map is not None else None
    if noise_sigma is not None:
        per = nx * ny
        noise_sigma = np.concatenate([noise_sigma[int(z) * per : (int(z) + 1) * per] for z in sel])
    acq = {**protocol.acquisition(ny), **artifacts.acquisition()}
    # `pe` names the phase-encode polarity in the WRITTEN frame (BIDS "j-"/"j" of the file the
    # caller gets). With fsl_orientation the written frame is LAS; if that reorientation flips
    # the native j axis (an LPS phantom), the simulation's native polarity is the opposite.
    pe_flip = protocol.fsl_orientation and _core.out_ped(_core.reorient_to_las(obj.affine, obj.dims), 1, -1) == "j"
    acq["reverse_phase"] = bool(protocol.reverse_phase) != pe_flip
    te_pv = None
    if artifacts.te_per_volume is not None:
        te_pv = [float(v) for v in np.asarray(artifacts.te_per_volume).reshape(-1)]
        if len(te_pv) != ngrad:
            raise ValueError(f"te_per_volume has {len(te_pv)} entries, the scheme {ngrad} volumes")
    cap_names = set(kspace_capture)
    bad = cap_names - {"acquired", "reconstructed", "coil_images"}
    if bad:
        raise ValueError(f"unknown kspace_capture entries {sorted(bad)}")
    capture = ("acquired" in cap_names, "reconstructed" in cap_names, "coil_images" in cap_names)

    t = time.perf_counter()
    say("acquire", 0, ngrad)
    prog = (lambda done, total: say("acquire", done, total)) if progress is not None else None
    try:
        out = _core.simulate_acquisition(
            comp_sel, fmap_sel, (nx, ny, nsl), bvals, bvecs, acq, protocol.phase_model, seed, noise_sigma, te_pv,
            global_z, nz_full, list(range(nsl)) if kspace else None, capture, prog,
        )
    except BaseException as e:  # pyo3 panics are BaseExceptions
        if type(e).__name__ == "PanicException":
            raise TrxscanError(str(e)) from None
        raise
    timing["acquire"] = time.perf_counter() - t

    # ---- output frame ----
    out_dims = (nx, ny, nsl)
    out_affine = obj.affine.copy()
    out_affine[:3, 3] += obj.affine[:3, 2] * int(sel[0])
    if artifacts.isocenter is not None:
        out_affine[:3, 3] -= np.asarray(artifacts.isocenter, dtype=np.float64)
    reo = _core.reorient_to_las(out_affine, out_dims) if protocol.fsl_orientation else _core.reorient_identity(out_dims)
    in_pe_sign = 1 if acq["reverse_phase"] else -1
    ped = _core.out_ped(reo, 1, in_pe_sign)
    re_, im_, mag_, ph_ = out["re"], out["im"], out["mag"], out["phase"]
    if protocol.fsl_orientation:
        re_ = _core.apply_reorient(re_, ngrad, reo)
        im_ = _core.apply_reorient(im_, ngrad, reo)
        mag_ = _core.apply_reorient(mag_, ngrad, reo)
        ph_ = _core.apply_reorient(ph_, ngrad, reo)
        out_affine = _core.reorient_affine(out_affine, reo).reshape(4, 4)
        out_dims = tuple(int(v) for v in reo["out_dims"])
    bvecs_fsl = _core.fsl_bvecs(np.ascontiguousarray(bvecs), out_affine).reshape(-1, 3)
    sidecar = {**protocol.sidecar(ny, nz=nz_full, nx=nx), "PhaseEncodingDirection": ped}
    # EffectiveEchoSpacing follows the CLI: TotalReadoutTime / (n_PE - 1) along the WRITTEN PE axis.
    pe_axis = {"i": 0, "j": 1, "k": 2}[ped[0]]
    sidecar["EffectiveEchoSpacing"] = round(sidecar["TotalReadoutTime"] / max(out_dims[pe_axis] - 1, 1), 8)
    if "SliceTiming" in sidecar:
        # the written image holds `global_z` of the full volume's slices
        sidecar["SliceTiming"] = [sidecar["SliceTiming"][z] for z in global_z]
    if gre_result is not None:
        sidecar["B0FieldSource"] = gre_result.b0_field

    readout = _readout_from_acquisition(nx, ny, acq)
    ks = None
    if out["kspace"] is not None:
        k = out["kspace"]
        nc = int(k["n_coils"])
        shp5 = (ngrad, nsl, nc, ny, nx)
        ks = KSpace(
            acquired=None if k["acquired"] is None else complex_pairs(k["acquired"], shp5),
            reconstructed=None if k["reconstructed"] is None else complex_pairs(k["reconstructed"], shp5),
            coil_images=None if k["coil_images"] is None else complex_pairs(k["coil_images"], shp5),
            combined=complex_pairs(k["combined"], (ngrad, nsl, ny, nx)),
            sensitivities=np.asarray(k["sensitivities"], dtype=np.float32).reshape(nc, ny, nx),
            mask=np.asarray(k["mask"]).reshape(ny, nx).astype(bool),
            slices=np.asarray(global_z), bvals=bvals, bvecs=bvecs, readout=readout,
            meta={"seed": seed, "oversample": o, "accel": protocol.accel, "acs_lines": protocol.acs_lines, "voxel_mm": list(obj.voxel_mm), "affine": out_affine.tolist(), "trxscan": _core.core_version()},
        )

    tp_img = None
    if peaks_flat is not None:
        per = nx * ny * 9
        pk = np.concatenate([peaks_flat[z * per : (z + 1) * per] for z in local])
        if protocol.fsl_orientation:
            pk = _core.apply_reorient(pk, 9, reo)
        tp_img = nifti(vectors(pk, out_dims, 9), out_affine)

    gnl_result = None
    if coef is not None:
        field_out = _core.GnlField.on_grid(coef, out_dims, out_affine)
        disp = vectors(field_out.disp(), out_dims, 3)
        src = vectors(field_out.src_vox(), out_dims, 3)
        ijk = np.stack(np.meshgrid(*[np.arange(n) for n in out_dims], indexing="ij"), axis=-1).astype(np.float64)
        inv = _voxel_to_world(out_affine, src.astype(np.float64)) - _voxel_to_world(out_affine, ijk)
        gnl_result = GnlResult(
            coeff_text=coef.to_siemens(), disp=nifti(disp, out_affine), invdisp=nifti(inv.astype(np.float32), out_affine),
            graddev=nifti(vectors(field_out.graddev(out_affine), out_dims, 9), out_affine), spec=gnl_spec,
            scale=float(artifacts.gnl_scale), warp=bool(artifacts.gnl_warp), encoding=bool(artifacts.gnl_encoding), jacobian=bool(artifacts.gnl_jacobian),
        )

    ns_img = None
    if noise_sigma is not None:
        ns = _core.apply_reorient(noise_sigma, 1, reo) if protocol.fsl_orientation else noise_sigma
        ns_img = nifti(volume(ns, out_dims), out_affine)

    timing["total"] = time.perf_counter() - t_start
    return Simulation(
        re=re_, im=im_, mag=mag_, ph=ph_, dims=out_dims, affine=out_affine, slices=np.asarray(global_z),
        bvals=bvals, bvecs=bvecs, bvecs_fsl=bvecs_fsl, protocol=protocol, artifacts=artifacts, sidecar=sidecar,
        readout=readout, kspace=ks, truth_peaks=tp_img, gre=gre_result, gnl=gnl_result, dropout=dropped,
        noise_sigma=ns_img, mixture=mixture, compartments=comp, object=sub, timing=timing,
    )


# ─── microstructure ─────────────────────────────────────────────────────────


def microstructure(
    obj: "Object",
    protocol: Protocol = Protocol.DEFAULT,
    gtab: Any = None,
    *,
    slices: Any = "all",
    big_delta: float | None = None,
    small_delta: float | None = None,
    tau: float | None = None,
    d_perp_floor: float | None = None,
    ng_floor: float | None = None,
    ng_top_k: int | None = None,
) -> dict[str, nib.Nifti1Image]:
    """The 27 closed-form ground-truth maps (``trxscan-microstructure``) on the acquisition
    grid of ``obj``, keyed by name (``fa``, ``md``, ..., ``icvf``, ``odi``, ``isovf``).

    ``tau = big_delta - small_delta / 3`` puts the MAP-MRI maps in physical units; it is taken
    from the dipy gradient table's timing when ``gtab`` carries it, from ``big_delta`` /
    ``small_delta``, or from ``tau``; otherwise dipy's normalised-units default applies.
    """
    bd = sd = None
    b_max = 1000.0
    if gtab is not None:
        bvals, _, bd, sd = as_gtab(gtab)
        b_max = b_max_of(bvals)
    if big_delta is not None:
        bd = big_delta
    if small_delta is not None:
        sd = small_delta
    if tau is None and bd is not None and sd is not None:
        tau = float(bd) - float(sd) / 3.0
    sel = _resolve_slices(slices, None, obj)
    sub = obj.slab(int(sel[0]), int(sel[-1]) + 1)
    params = protocol.tissue.to_params(b_max, protocol.diff_scale)
    if sub.fibers is not None:
        # fibres live on the simulation grid; map them to acquisition voxels
        o = sub.oversample
        nx, ny, _ = sub.dims
        snx, sny, _ = sub.sim_dims
        v = sub.fibers.voxel.astype(np.int64)
        x, y, z = v % snx, (v // snx) % sny, v // (snx * sny)
        acq_vox = (x // o) + nx * ((y // o) + ny * z)
        mix = _core.mixture_from_fibers(sub.dims, acq_vox.astype(np.uint32), sub.fibers.dirs, sub.fibers.weight, sub.wm, sub.gm, sub.csf, sub.mask, params, protocol.kappa)
    elif sub.streamlines is not None:
        sl = sub.streamlines.crop_k(obj.affine, int(sel[0]) - 1.0, int(sel[-1]) + 2.0) if sub.dims[2] != obj.dims[2] else sub.streamlines
        mix = _core.build_mixture(sub.dims, sub.wm, sub.gm, sub.csf, sub.mask, sub.affine, sl.positions, sl.offsets, sl.weights, params, protocol.kappa, None)
    else:
        raise ValueError("the object has neither streamlines nor fibres")
    maps = mix.scalars(tau, d_perp_floor, ng_floor, ng_top_k)
    return {name: nifti(volume(m, sub.dims), sub.affine) for name, m in zip(_core.scalar_names(), maps)}
