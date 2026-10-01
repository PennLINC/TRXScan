"""Hosted phantoms: :func:`load_phantom` fetches the NIBS subject ``sub-60501`` (or the small
``slab`` bundle) from the ``PennLINC/trxscan-phantoms`` GitHub release into the pooch cache,
verifying checksums against the registries shipped in this package.

Resolution order: ``$TRXSCAN_DATA/<name>/`` when it exists (a local copy, e.g. the original
kit), else the pooch cache (``pooch.os_cache("trxscan")``). ``TRXSCAN_OFFLINE=1`` makes any
download attempt fail immediately.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..motion import Motion
from ..phantom import Phantom

REPO = "PennLINC/trxscan-phantoms"

#: Files a bundle may lack (skipped when absent locally or missing from the release registry).
OPTIONAL_FILES = ("t1w", "t2w", "motion_AP", "motion_PA")

#: bundle name -> (release tag, registry file, layout)
BUNDLES: dict[str, dict[str, Any]] = {
    "sub-60501": {
        "tag": "sub-60501-v1",
        "registry": "registry_sub-60501.txt",
        "prefix": "sub-60501_ses-01_space-ACPC",
        "files": {
            "wm": "{p}_label-WM_probseg.nii.gz",
            "gm": "{p}_label-GM_probseg.nii.gz",
            "csf": "{p}_label-CSF_probseg.nii.gz",
            "mask": "{p}_desc-brain_mask.nii.gz",
            "fieldmap": "{p}_fieldmap.nii.gz",
            "streamlines": "{p}_desc-actsift2_tracks.trx",
            "motion_AP": "sub-60501_ses-01_dir-AP_desc-confounds_timeseries.tsv",
            "motion_PA": "sub-60501_ses-01_dir-PA_desc-confounds_timeseries.tsv",
            "bval_AP": "hbcd75_ap.bval", "bvec_AP": "hbcd75_ap.bvec", "bval_PA": "hbcd75_pa.bval", "bvec_PA": "hbcd75_pa.bvec",
            "t1w": "{p}_desc-preproc_T1w.nii.gz", "t2w": "{p}_desc-preproc_T2w.nii.gz",
        },
        # the original kit's nested layout, accepted under $TRXSCAN_DATA
        "kit_layout": {
            "t1w": "sub-60501/ses-01/anat/{p}_desc-preproc_T1w.nii.gz",
            "t2w": "sub-60501/ses-01/anat/{p}_desc-preproc_T2w.nii.gz",
            "wm": "sub-60501/ses-01/anat/{p}_label-WM_probseg.nii.gz",
            "gm": "sub-60501/ses-01/anat/{p}_label-GM_probseg.nii.gz",
            "csf": "sub-60501/ses-01/anat/{p}_label-CSF_probseg.nii.gz",
            "mask": "sub-60501/ses-01/anat/{p}_desc-brain_mask.nii.gz",
            "fieldmap": "sub-60501/ses-01/anat/{p}_fieldmap.nii.gz",
            "streamlines": "sub-60501/ses-01/tract/{p}_desc-actsift2_tracks.trx",
            "motion_AP": "sub-60501/ses-01/dwi/sub-60501_ses-01_dir-AP_desc-confounds_timeseries.tsv",
            "motion_PA": "sub-60501/ses-01/dwi/sub-60501_ses-01_dir-PA_desc-confounds_timeseries.tsv",
            "bval_AP": "scheme/hbcd75_ap.bval", "bvec_AP": "scheme/hbcd75_ap.bvec", "bval_PA": "scheme/hbcd75_pa.bval", "bvec_PA": "scheme/hbcd75_pa.bvec",
        },
    },
    "slab": {
        "tag": "slab-v1",
        "registry": "registry_slab.txt",
        "prefix": "slab",
        "files": {
            "wm": "slab_label-WM_probseg.nii.gz",
            "gm": "slab_label-GM_probseg.nii.gz",
            "csf": "slab_label-CSF_probseg.nii.gz",
            "mask": "slab_desc-brain_mask.nii.gz",
            "fieldmap": "slab_fieldmap.nii.gz",
            "streamlines": "slab_desc-actsift2_tracks.trx",
            "motion_AP": "slab_dir-AP_desc-confounds_timeseries.tsv",
            "bval_AP": "hbcd75_ap.bval", "bvec_AP": "hbcd75_ap.bvec",
        },
    },
}


def registry(name: str) -> dict[str, str]:
    """The ``{file: sha256}`` registry shipped for bundle ``name`` (empty until released)."""
    path = Path(__file__).with_name(BUNDLES[name]["registry"])
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f, digest = line.split()
        out[f] = digest
    return out


def data_dir(name: str) -> Path | None:
    """``$TRXSCAN_DATA/<name>`` if it exists, else None."""
    root = os.environ.get("TRXSCAN_DATA")
    if not root:
        return None
    p = Path(root) / name
    return p if p.is_dir() else None


def _resolve(name: str) -> dict[str, Path]:
    """File paths of a bundle, plus ``"root"``: the directory holding it."""
    b = BUNDLES[name]
    local = data_dir(name)
    if local is not None:
        for layout_key in ("kit_layout", "files"):
            layout = b.get(layout_key)
            if not layout:
                continue
            paths = {k: local / v.format(p=b["prefix"]) for k, v in layout.items()}
            if paths["wm"].exists():
                paths["root"] = local
                return paths
        raise FileNotFoundError(f"$TRXSCAN_DATA/{name} exists but has neither the kit nor the release layout")
    if os.environ.get("TRXSCAN_OFFLINE"):
        raise RuntimeError(f"TRXSCAN_OFFLINE is set and {name!r} is not in $TRXSCAN_DATA")
    reg = registry(name)
    if not reg:
        raise RuntimeError(f"no download registry for {name!r} yet: set TRXSCAN_DATA to a local copy")
    import pooch

    fetcher = pooch.create(
        path=pooch.os_cache("trxscan") / name,
        base_url=f"https://github.com/{REPO}/releases/download/{b['tag']}/",
        registry=reg,
    )
    paths = {k: Path(fetcher.fetch(v.format(p=b["prefix"]))) for k, v in b["files"].items()
             if k not in OPTIONAL_FILES or v.format(p=b["prefix"]) in reg}
    paths["root"] = Path(fetcher.abspath)
    return paths


def load_phantom(name: str = "sub-60501", *, weights: str | None = "sift2_weights") -> Phantom:
    """A hosted phantom as a :class:`~trxscan.phantom.Phantom`. Its example acquisition
    files (``hbcd75_{ap,pa}.bval/.bvec``) are reachable through ``phantom.path`` and
    :func:`scheme_files`."""
    if name not in BUNDLES:
        raise KeyError(f"unknown phantom {name!r}; known: {sorted(BUNDLES)}")
    paths = _resolve(name)
    motion = {k.split("_", 1)[1]: Motion.from_confounds(v) for k, v in paths.items() if k.startswith("motion_") and v.exists()}
    root = paths["root"]
    anat = {k: paths[k] for k in ("t1w", "t2w") if k in paths and paths[k].exists()}
    ph = Phantom.from_files(
        wm=paths["wm"], gm=paths["gm"], csf=paths["csf"], mask=paths.get("mask"), fieldmap=paths.get("fieldmap"),
        streamlines=paths["streamlines"], weights=weights, motion=motion, name=name, **anat,
    )
    return Phantom(**{**{f.name: getattr(ph, f.name) for f in ph.__dataclass_fields__.values() if f.name != "_grids"}, "path": root})


def scheme_files(name: str = "sub-60501", pe: str = "AP") -> tuple[Path, Path]:
    """``(bval, bvec)`` paths of the example acquisition that ships with a bundle (feed to
    dipy's ``read_bvals_bvecs``/``gradient_table``)."""
    paths = _resolve(name)
    return paths[f"bval_{pe}"], paths[f"bvec_{pe}"]
