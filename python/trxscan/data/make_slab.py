"""Derive the small ``slab`` bundle from the full phantom, using only the public API::

    python -m trxscan.data.make_slab --out slab/ --voxel 2.5 --slices 24:30 --subsample 50000 --seed 1

Writes the slab's tissue maps, mask and fieldmap on the ANATOMICAL grid restricted to the
world-z range of the chosen acquisition slices (plus margin), the cropped, subsampled
tractogram (TRX, with SIFT2 weights), the motion trace and the example scheme, and a
``slab.json`` provenance record. The slab is itself a regression fixture for the gridding.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from .. import Phantom, Protocol, __version__
from . import scheme_files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--phantom", default="sub-60501")
    ap.add_argument("--voxel", type=float, default=2.5)
    ap.add_argument("--slices", default="24:30", help="acquisition-slice range z0:z1 at --voxel")
    ap.add_argument("--subsample", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--margin-mm", type=float, default=10.0)
    a = ap.parse_args(argv)
    z0, z1 = (int(v) for v in a.slices.split(":"))
    ph = Phantom.load(a.phantom)
    proto = Protocol.DEFAULT.replace(voxel_mm=a.voxel, oversample=1)
    obj = ph.grid(proto)
    # world-z range of the requested slices (+ margin)
    lo = obj.affine @ np.array([0, 0, z0, 1.0])
    hi = obj.affine @ np.array([0, 0, z1 - 1, 1.0])
    zlo, zhi = min(lo[2], hi[2]) - a.margin_mm, max(lo[2], hi[2]) + a.margin_mm
    a.out.mkdir(parents=True, exist_ok=True)

    def crop(img: nib.Nifti1Image) -> nib.Nifti1Image:
        inv = np.linalg.inv(img.affine)
        ks = [(inv @ np.array([0, 0, z, 1.0]))[2] for z in (zlo, zhi)]
        k0, k1 = int(np.floor(min(ks))), int(np.ceil(max(ks))) + 1
        k0, k1 = max(k0, 0), min(k1, img.shape[2])
        data = np.asanyarray(img.dataobj)[:, :, k0:k1]
        aff = img.affine.copy()
        aff[:3, 3] += img.affine[:3, 2] * k0
        return nib.Nifti1Image(np.asarray(data, dtype=np.float32), aff)

    for key, suffix in (("wm", "label-WM_probseg"), ("gm", "label-GM_probseg"), ("csf", "label-CSF_probseg"), ("mask", "desc-brain_mask"), ("fieldmap", "fieldmap")):
        img = getattr(ph, key)
        if img is None:
            continue
        crop(img).to_filename(str(a.out / f"slab_{suffix}.nii.gz"))
    sub = ph.subsample(a.subsample, seed=a.seed)
    sl = sub.streamlines.crop_k(obj.affine, z0 - a.margin_mm / a.voxel, z1 + a.margin_mm / a.voxel)
    try:
        from trx.trx_file_memmap import TrxFile  # noqa: F401

        from ..viz import write_trx
    except ImportError as e:  # pragma: no cover
        raise SystemExit("make_slab needs trx-python (pip install trxscan[trx])") from e
    write_trx(sl, a.out / "slab_desc-actsift2_tracks.trx", reference=ph.wm)
    if "AP" in ph.motion:
        src = ph.motion["AP"].source
        if src and Path(src).exists():
            shutil.copy(src, a.out / "slab_dir-AP_desc-confounds_timeseries.tsv")
    bval, bvec = scheme_files(a.phantom, "AP")
    shutil.copy(bval, a.out / "hbcd75_ap.bval")
    shutil.copy(bvec, a.out / "hbcd75_ap.bvec")
    (a.out / "slab.json").write_text(json.dumps({
        "source": a.phantom, "voxel_mm": a.voxel, "slices": [z0, z1], "subsample": a.subsample, "seed": a.seed,
        "margin_mm": a.margin_mm, "n_streamlines": sl.n, "trxscan": __version__,
    }, indent=2) + "\n")
    print(f"wrote slab bundle to {a.out} ({sl.n} streamlines)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
