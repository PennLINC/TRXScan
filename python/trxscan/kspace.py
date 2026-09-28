"""Raw k-space from a simulation, as numpy ``complex64`` in the book's ``(..., ky, kx)`` layout."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .protocol import EpiReadout


@dataclass(eq=False)
class KSpace:
    """Per-slice, per-coil k-space captured during :meth:`Object.simulate`.

    Axes are ``[vol, slice, coil, ky, kx]`` for the per-coil arrays and ``[vol, slice, ky, kx]``
    for ``combined``; ``ky`` is the phase-encode axis, centred at ``ny // 2``.
    """

    #: As acquired: after spikes and k-space noise, BEFORE GRAPPA, unwindowed; un-acquired lines
    #: (partial Fourier, undersampling) are exactly zero. ``None`` if not captured.
    acquired: np.ndarray | None
    #: As reconstructed: after GRAPPA and the reconstruction window (what is inverse-transformed).
    reconstructed: np.ndarray | None
    #: Per-coil complex images before the Roemer combine.
    coil_images: np.ndarray | None
    #: Roemer-combined complex image per (vol, slice), before any image-space noise.
    combined: np.ndarray
    #: Real coil sensitivities on the acquired grid, ``(coil, y, x)``.
    sensitivities: np.ndarray
    #: Which samples were acquired, ``(ky, kx)`` bool (partial Fourier + undersampling).
    mask: np.ndarray
    #: Full-FOV slice index of each captured slice.
    slices: np.ndarray
    bvals: np.ndarray
    bvecs: np.ndarray
    readout: "EpiReadout | None" = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_vol(self) -> int:
        return self.combined.shape[0]

    @property
    def n_coils(self) -> int:
        return self.sensitivities.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        """``(ny, nx)``."""
        return self.mask.shape

    @staticmethod
    def recon_reference(kspace: np.ndarray, sensitivities: np.ndarray) -> np.ndarray:
        """The simulator's own reconstruction of per-coil ``[..., coil, ky, kx]`` k-space, for
        checking a Python reconstruction against ``combined``.

        TRXScan's inverse transform is the centred sum ``sum_k K(k) exp(-i 2 pi k.x / N)`` with
        no ``1/N`` (the forward carries it). The **negative** exponent makes it numpy's *forward*
        transform on centred data: ``fftshift(fft2(ifftshift(K)))``; ``dwibook.kspace.ifft2c``
        differs by the exponent sign (a spatial flip) and the orthonormal scale. The Roemer
        combine is then ``sum_c img_c s_c / sum_c s_c^2`` with the real sensitivities.
        """
        k = np.asarray(kspace)
        img = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(k, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
        s = np.asarray(sensitivities, dtype=np.float64)
        num = (img * s).sum(axis=-3)
        den = (s * s).sum(axis=0)
        return num / np.maximum(den, 1e-12)

    def save_npz(self, path: str | Path) -> Path:
        """Arrays plus JSON-encoded metadata in one ``.npz``."""
        path = Path(path)
        payload: dict[str, Any] = {
            "combined": self.combined,
            "sensitivities": self.sensitivities,
            "mask": self.mask,
            "slices": self.slices,
            "bvals": self.bvals,
            "bvecs": self.bvecs,
            "meta": np.array(json.dumps(_jsonable(self.meta))),
        }
        for name in ("acquired", "reconstructed", "coil_images"):
            v = getattr(self, name)
            if v is not None:
                payload[name] = v
        if self.readout is not None:
            payload["readout"] = np.array(json.dumps(self.readout.to_dict()))
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load_npz(cls, path: str | Path) -> "KSpace":
        from .protocol import EpiReadout

        with np.load(path, allow_pickle=False) as z:
            get = lambda k: z[k] if k in z.files else None  # noqa: E731
            readout = EpiReadout.from_dict(json.loads(str(z["readout"]))) if "readout" in z.files else None
            return cls(
                acquired=get("acquired"),
                reconstructed=get("reconstructed"),
                coil_images=get("coil_images"),
                combined=z["combined"],
                sensitivities=z["sensitivities"],
                mask=z["mask"].astype(bool),
                slices=z["slices"],
                bvals=z["bvals"],
                bvecs=z["bvecs"],
                readout=readout,
                meta=json.loads(str(z["meta"])) if "meta" in z.files else {},
            )

    def to_ismrmrd(self, path: str | Path) -> Path:
        """Write the acquired per-coil k-space as an ISMRMRD/MRD HDF5 dataset (needs the
        optional ``ismrmrd`` package: ``pip install trxscan[ismrmrd]``).

        One ``Acquisition`` per (volume, slice, ky line) with data ``(coil, nx)``; volumes map to
        ``idx.repetition``, slices to ``idx.slice``, the ky index to
        ``idx.kspace_encode_step_1``; lines inside the ACS band are flagged
        ``ACQ_IS_PARALLEL_CALIBRATION``. Un-acquired lines are not written.
        """
        try:
            import ismrmrd
            from ismrmrd import xsd
        except ImportError as e:  # pragma: no cover
            raise ImportError("KSpace.to_ismrmrd needs the 'ismrmrd' package: pip install trxscan[ismrmrd]") from e
        if self.acquired is None:
            raise ValueError("to_ismrmrd needs the 'acquired' k-space; simulate with kspace_capture including 'acquired'")
        path = Path(path)
        if path.exists():
            path.unlink()
        n_vol, n_sl, n_coil, ny, nx = self.acquired.shape
        acs = int(self.meta.get("acs_lines", 0))
        accel = int(self.meta.get("accel", 1))
        ys = ny // 2
        dset = ismrmrd.Dataset(str(path), "dataset", create_if_needed=True)
        header = xsd.ismrmrdHeader()
        exp = xsd.experimentalConditionsType()
        exp.H1resonanceFrequency_Hz = 123_200_000
        header.experimentalConditions = exp
        enc = xsd.encodingType()
        for space_name in ("encodedSpace", "reconSpace"):
            space = xsd.encodingSpaceType()
            mat = xsd.matrixSizeType()
            mat.x, mat.y, mat.z = nx, ny, 1
            fov = xsd.fieldOfViewMm()
            vx = self.meta.get("voxel_mm", [1.0, 1.0, 1.0])
            fov.x, fov.y, fov.z = float(vx[0]) * nx, float(vx[1]) * ny, float(vx[2])
            space.matrixSize = mat
            space.fieldOfView_mm = fov
            setattr(enc, space_name, space)
        limits = xsd.encodingLimitsType()
        e1 = xsd.limitType()
        e1.minimum, e1.maximum, e1.center = 0, ny - 1, ys
        limits.kspace_encoding_step_1 = e1
        sl = xsd.limitType()
        sl.minimum, sl.maximum, sl.center = 0, n_sl - 1, 0
        limits.slice = sl
        rep = xsd.limitType()
        rep.minimum, rep.maximum, rep.center = 0, n_vol - 1, 0
        limits.repetition = rep
        enc.encodingLimits = limits
        enc.trajectory = xsd.trajectoryType.EPI
        if accel > 1:
            pi = xsd.parallelImagingType()
            af = xsd.accelerationFactorType()
            af.kspace_encoding_step_1, af.kspace_encoding_step_2 = accel, 1
            pi.accelerationFactor = af
            pi.calibrationMode = xsd.calibrationModeType.EMBEDDED
            enc.parallelImaging = pi
        header.encoding.append(enc)
        dset.write_xml_header(xsd.ToXML(header))
        scan = 0
        for v in range(n_vol):
            for s in range(n_sl):
                for ky in range(ny):
                    if not self.mask[ky, 0]:
                        continue
                    acq = ismrmrd.Acquisition()
                    acq.resize(nx, n_coil)
                    acq.scan_counter = scan
                    acq.idx.kspace_encode_step_1 = ky
                    acq.idx.slice = s
                    acq.idx.repetition = v
                    acq.center_sample = nx // 2
                    acq.available_channels = n_coil
                    acq.active_channels = n_coil
                    if accel > 1 and abs(ky - ys) <= acs // 2:
                        acq.setFlag(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)
                    if ky == self.mask[:, 0].nonzero()[0][0]:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_SLICE)
                    if ky == self.mask[:, 0].nonzero()[0][-1]:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                    acq.data[:] = self.acquired[v, s, :, ky, :]
                    dset.append_acquisition(acq)
                    scan += 1
        dset.close()
        return path


def _jsonable(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.floating,)):
        return float(d)
    if isinstance(d, (np.bool_,)):
        return bool(d)
    return d
