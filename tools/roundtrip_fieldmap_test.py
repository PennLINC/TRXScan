#!/usr/bin/env python
"""Round-trip test: does TRXScan's EPI distortion obey the FSL / dcm2niix convention?

Every real dataset reaches a pipeline via dcm2niix, which writes NIfTIs in a fixed
orientation (radiological, det<0) with a BIDS `PhaseEncodingDirection` that refers to
the data axes *as stored*. FSL (topup/eddy/fugue) and qsiprep assume exactly that.
A simulator that writes a *different* orientation, or distorts opposite to the sign its
sidecar declares, produces undefined behaviour downstream — silently, because the
magnitude of the distortion is still right.

This test pins both halves of the contract, using only the sidecar metadata a real
pipeline reads. It targets `trxscan --fsl-orientation`, which resolves two defects that
the native-grid output has:

  (1) SIGN.  TRXScan's k-space forward model (`phi = +fmap*t`, high-ky-first readout,
      -i recon: `build_coil_kspace` in src/kspace.rs, `line_times`, src/readout.rs) displaces a
      +Hz field toward -j. FSL's convention (eddy `--field=+F`, acqp "0 1 0") is +field
      -> +j. The written PhaseEncodingDirection now describes this honestly (`orient::
      Reorient::out_ped`): native output is labelled "j-", and --fsl-orientation flips
      the j-axis so the distortion becomes +j and the label is "j".

  (2) ORIENTATION.  The native output inherits the LPS+ / det=+1 (neurological) grid of
      the reference tissue maps. Real dcm2niix DWIs are LAS / det<0 (radiological), +j ->
      Anterior. --fsl-orientation reorients the volume + affine + bvecs to LAS (a
      world-space no-op) so eddy/topup/fugue and qsiprep consume it without a
      conform-to-canonical step silently reversing the PE axis.

Both are pure I/O (the Fiberfox physics is untouched). Reorienting to LAS fixes BOTH at
once: because SDC is defined in the data-array frame, flipping the j-axis turns the -j
distortion into +j while making the orientation radiological.

Design notes:
  * The reference forward distortion `fsl_convention_forward` *is* the convention we are
    standardising on (documented + self-tested), so the test needs no FSL install.
  * `--real-ref <dwi.nii.gz>` grounds the orientation expectation on an actual dcm2niix
    file (default expectation: radiological, PE axis along A-P, matching MBP's LAS DWIs).
  * The direction check compares TRXScan's +field distortion against the FSL-convention
    +field vs -field references by correlation — robust to the Jacobian/blur differences
    a k-space model has over a pixel-shift model (right sign vs flipped sign is a large,
    stable gap), so it does not gate on tight RMSE.

Run (needs the trxscan binary on PATH + the reference bundle, $TRXSCAN_REFERENCE or
~/projects/trxscan-reference; skips cleanly otherwise):
    python tools/roundtrip_fieldmap_test.py --bundle <reference-bundle> \
        --trxscan target/release/trxscan
Or under pytest:  pytest tools/roundtrip_fieldmap_test.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
    from nibabel.orientations import aff2axcodes
except ImportError:  # pragma: no cover - environment guard
    nib = None

try:
    from scipy.ndimage import map_coordinates
except ImportError:  # pragma: no cover
    map_coordinates = None

DEFAULT_BUNDLE = Path(os.environ.get("TRXSCAN_REFERENCE", Path.home() / "projects/trxscan-reference"))

# ----- BIDS PhaseEncodingDirection -> (data-array axis, sign) -------------------------
# The axis is the voxel/data-array index (i=0, j=1, k=2), NOT a world direction; the sign
# is +1 for "j" and -1 for "j-", matching FSL eddy's acqp rows (j -> "0 1 0", j- -> "0 -1 0").
_PED = {"i": (0, +1), "i-": (0, -1), "j": (1, +1), "j-": (1, -1), "k": (2, +1), "k-": (2, -1)}


def ped_axis_sign(ped: str) -> tuple[int, int]:
    ped = ped.replace("+", "")
    if ped not in _PED:
        raise ValueError(f"unrecognised PhaseEncodingDirection {ped!r}")
    return _PED[ped]


def fsl_convention_forward(vol: np.ndarray, field_hz: np.ndarray, trt: float, ped: str,
                           jacobian: bool = True) -> np.ndarray:
    """Distort `vol` by an off-resonance `field_hz` per the FSL / BIDS convention.

    Displacement along the PE data-axis is d = field_hz * TotalReadoutTime * sign(ped)
    voxels, and a POSITIVE field pushes signal toward +PE (this is the direction eddy
    `--field=+F` removes). Implemented as a pull-resample (dist[p] = vol[p - d]); with
    `jacobian`, intensity is modulated by (1 + d d/dp) so signal piles up where it
    compresses — the same effect a faithful k-space model produces.
    """
    if map_coordinates is None:
        raise RuntimeError("scipy is required for the reference warp")
    axis, sign = ped_axis_sign(ped)
    d = field_hz.astype(np.float64) * trt * sign  # voxels, per-voxel
    grid = np.indices(vol.shape, dtype=np.float64)
    grid[axis] = grid[axis] - d  # pull from p-d  => +field moves signal to +PE
    out = map_coordinates(vol.astype(np.float64), grid, order=1, mode="constant", cval=0.0)
    if jacobian:
        dd = np.gradient(d, axis=axis)
        out = out * np.clip(1.0 + dd, 0.0, None)
    return out.astype(np.float32)


def _pe_profile(vol: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    other = tuple(i for i in range(3) if i != axis)
    return (vol * mask).sum(axis=other)


def estimate_pe_shift(truth: np.ndarray, dist: np.ndarray, mask: np.ndarray, axis: int) -> float:
    """Signed shift (voxels) of `dist` relative to `truth` along data-axis `axis`, from the
    peak of the 1-D cross-correlation of the masked marginal profiles (parabolic-refined).
    Convention: cc(lag) = Σ truth[j]·dist[j+lag], so a value of +s means `dist` is `truth`
    translated toward +axis by s voxels. Robust to the 2-D structure a centroid dilutes."""
    a = _pe_profile(truth, mask, axis).astype(float); a -= a.mean()
    b = _pe_profile(dist, mask, axis).astype(float); b -= b.mean()
    n = len(a); max_lag = n // 2
    cc = {}
    for lag in range(-max_lag, max_lag + 1):
        cc[lag] = float(np.sum(a[:n - lag] * b[lag:]) if lag >= 0 else np.sum(a[-lag:] * b[:n + lag]))
    best = max(cc, key=cc.get)
    lag = float(best)
    if best - 1 in cc and best + 1 in cc:  # parabolic subpixel refine
        y0, y1, y2 = cc[best - 1], cc[best], cc[best + 1]
        d = y0 - 2 * y1 + y2
        if d != 0:
            lag = best + 0.5 * (y0 - y2) / d
    return lag


# ----- orientation contract -----------------------------------------------------------

def orientation_facts(path: Path) -> dict:
    img = nib.load(str(path))
    aff = img.affine
    det = float(np.linalg.det(aff[:3, :3]))
    ax = aff2axcodes(aff)
    return {"axcodes": ax, "det": det, "radiological": det < 0, "pe_pole": ax[1]}


def assert_dcm2niix_orientation(path: Path, real_ref: Path | None = None) -> list[str]:
    """Return a list of contract violations ([] == passes)."""
    got = orientation_facts(path)
    fails = []
    # (a) handedness: real dcm2niix DWIs are radiological (det<0). TRXScan's LPS+ det=+1 fails.
    if not got["radiological"]:
        fails.append(f"orientation is NEUROLOGICAL (det={got['det']:+.2f}); "
                     f"dcm2niix writes RADIOLOGICAL (det<0). axcodes={got['axcodes']}")
    # (b) PE-axis anatomical sense must match a real reference (default: +j -> Anterior, as MBP LAS).
    want_pole = orientation_facts(real_ref)["pe_pole"] if real_ref else "A"
    if got["pe_pole"] != want_pole:
        fails.append(f"phase-encode (j) axis points +j->{got['pe_pole']}; "
                     f"real dcm2niix data has +j->{want_pole}. A 'j' label is not portable across these.")
    return fails


# ----- driving the simulator ----------------------------------------------------------

def _first_b0_index(bval_path: Path) -> int:
    bvals = np.loadtxt(str(bval_path)).ravel()
    idx = np.where(bvals < 50)[0]
    return int(idx[0]) if idx.size else 0


def run_trxscan(trxscan: str, grid: Path, streamlines: Path, bval: Path, bvec: Path,
                fmap: Path, out_stem: Path, extra: list[str] | None = None) -> Path:
    # Pin the legacy (--oversample 1) path: it takes the acquisition-grid --fmap directly (no
    # --sim-* grids needed) and is fast. The orientation + distortion-sign convention this test
    # checks is applied post-simulation and is identical on the oversampled path.
    cmd = [trxscan,
           "--wm", str(grid / "wm.nii.gz"), "--gm", str(grid / "gm.nii.gz"),
           "--csf", str(grid / "csf.nii.gz"), "--mask", str(grid / "mask.nii.gz"),
           "--streamlines", str(streamlines), "--bval", str(bval), "--bvec", str(bvec),
           "--oversample", "1", "--fmap", str(fmap), "-o", str(out_stem)]
    cmd += extra or []
    subprocess.run(cmd, check=True, capture_output=True)
    return out_stem.with_name(out_stem.name + "_part-mag_dwi.nii.gz")


def prepare_grid(bundle: Path, out: Path, voxel: float = 3.0) -> None:
    repo = Path(__file__).resolve().parent.parent
    subprocess.run([sys.executable, str(repo / "scripts/prepare_acquisition_grid.py"),
                    "--anat-dir", str(bundle / "sub-0001a/anat"),
                    "--prefix", "sub-0001a_space-ACPC", "--out", str(out),
                    "--voxel", str(voxel)], check=True, capture_output=True)


def run_roundtrip(bundle: Path, trxscan: str, work: Path, voxel: float = 3.0,
                  field_scale: float | None = None, real_ref: Path | None = None,
                  fsl_orientation: bool = True) -> dict:
    """Full round trip. Returns a report dict with orientation + direction results.

    `fsl_orientation` runs trxscan with --fsl-orientation (the mode eddy/topup consume): both
    contracts should pass. Set it False to inspect the native-grid output instead."""
    extra = ["--fsl-orientation"] if fsl_orientation else []
    grid = work / "grid"
    prepare_grid(bundle, grid, voxel)
    streamlines = bundle / "sub-0001a/tract/sub-0001a_space-ACPC_desc-actsift2_tracks.trx"
    bval, bvec = bundle / "scheme/hbcd_ap.bval", bundle / "scheme/hbcd_ap.bvec"
    grid_img = nib.load(str(grid / "fmap_hz.nii.gz"))

    # Drive the simulator with UNIFORM +f0 and -f0 off-resonance fields: each translates the whole
    # image rigidly along the PE axis, in opposite directions. Measuring the +f image RELATIVE to
    # the -f image doubles the signal and cancels any bias, giving an unambiguous signed direction
    # (a spatially-varying field's k-space pile-up/blur is too structure-dependent to judge by
    # correlation). f0 aims for a few-voxel shift.
    trt_guess = float(voxel) * grid_img.shape[1] / 1000.0
    f0 = float(field_scale) if field_scale is not None else 40.0  # Hz
    for tag, val in [("pos", f0), ("neg", -f0)]:
        nib.save(nib.Nifti1Image(np.full(grid_img.shape, val, np.float32), grid_img.affine, grid_img.header),
                 grid / f"fmap_{tag}.nii.gz")

    pos_mag = run_trxscan(trxscan, grid, streamlines, bval, bvec, grid / "fmap_pos.nii.gz", work / "pos", extra)
    neg_mag = run_trxscan(trxscan, grid, streamlines, bval, bvec, grid / "fmap_neg.nii.gz", work / "neg", extra)

    # --- orientation contract (reads what trxscan actually wrote) ---
    orient_fails = assert_dcm2niix_orientation(pos_mag, real_ref)

    # --- signed-direction contract (measured on trxscan's own output frame) ---
    b0 = _first_b0_index(bval)
    sidecar = _read_sidecar(pos_mag)
    ped = sidecar.get("PhaseEncodingDirection", "j")
    trt = float(sidecar.get("TotalReadoutTime", trt_guess))
    axis, ped_sign = ped_axis_sign(ped)
    pos = np.asarray(nib.load(str(pos_mag)).dataobj, dtype=np.float32)[..., b0]
    neg = np.asarray(nib.load(str(neg_mag)).dataobj, dtype=np.float32)[..., b0]
    # brain mask derived from the output itself (frame-agnostic; grid/mask.nii.gz is in the grid
    # frame, which differs from the written frame under --fsl-orientation).
    thr = 0.05 * float(max(pos.max(), neg.max()))
    mask = (pos > thr) | (neg > thr)

    # shift of the +field image relative to the -field image, along the declared PE axis.
    # FSL convention: +field -> +axis, -field -> -axis, so +f sits at +2*shift vs -f for PED "j".
    dj = estimate_pe_shift(neg, pos, mask, axis)
    direction_ok = (dj * ped_sign) > 0.5

    return {"orientation_fails": orient_fails, "fsl_orientation": fsl_orientation,
            "ped": ped, "trt": trt, "f0_hz": f0, "pe_axis": axis, "ped_sign": ped_sign,
            "measured_shift_vox": dj, "expected_shift_vox": 2.0 * f0 * trt * ped_sign,
            "direction_ok": direction_ok}


def _read_sidecar(mag_path: Path) -> dict:
    import json
    j = mag_path.with_name(mag_path.name.replace("_dwi.nii.gz", "_dwi.json"))
    try:
        return json.loads(j.read_text())
    except Exception:
        return {}


# ----- pytest entry points (skip cleanly when deps/inputs are absent) -----------------

def _resolve_trxscan(trxscan: str | None) -> str | None:
    """Return a runnable trxscan path (from PATH or an explicit file), else None."""
    if not trxscan:
        return None
    if shutil.which(trxscan):
        return shutil.which(trxscan)
    p = Path(trxscan)
    return str(p) if p.is_file() and os.access(p, os.X_OK) else None


def _skip_reason(bundle: Path, trxscan: str | None) -> str | None:
    if nib is None or map_coordinates is None:
        return "needs nibabel + scipy"
    if _resolve_trxscan(trxscan) is None:
        return "trxscan binary not found (build with: cargo build --release --features cli,par)"
    if not (bundle / "sub-0001a/anat").is_dir():
        return f"reference bundle not found at {bundle} (set TRXSCAN_REFERENCE)"
    return None


def _blob(shape, center_j, sigma=2.5):
    zz, jj, kk = np.indices(shape, dtype=np.float64)
    g = np.exp(-((jj - center_j) ** 2) / (2 * sigma ** 2)
               - ((zz - shape[0] / 2) ** 2 + (kk - shape[2] / 2) ** 2) / (2 * (sigma * 1.5) ** 2))
    return g.astype(np.float32)


def _centroid_j(vol):
    jj = np.arange(vol.shape[1])
    w = vol.sum(axis=(0, 2))
    return float((jj * w).sum() / w.sum())


def test_reference_forward_is_self_consistent():
    """Harness check (no trxscan). Uses a single blob and tracks its centroid, so it
    asserts two things unambiguously: (a) the forward inverts under a flipped-sign unwarp
    (round-trip identity), and (b) the convention direction is +field -> +j for PED "j"
    (the direction eddy `--field=+F` removes). This is the operator the test judges
    TRXScan against, so it must be sound and direction-correct itself."""
    if nib is None or map_coordinates is None:
        import pytest; pytest.skip("needs nibabel + scipy")
    shape = (20, 40, 16)
    vol = _blob(shape, center_j=20)
    mask = np.ones(shape, bool)
    field = np.full(shape, 30.0, np.float32)  # uniform +30 Hz
    trt = 0.06  # -> shift = 30*0.06 = 1.8 voxels
    warped = fsl_convention_forward(vol, field, trt, "j", jacobian=False)
    recovered = fsl_convention_forward(warped, -field, trt, "j", jacobian=False)
    c0, cw, cr = _centroid_j(vol), _centroid_j(warped), _centroid_j(recovered)
    # (b) +field pushes the blob toward +j (FSL convention for "j")
    assert cw - c0 > 1.0, f"+field must move signal toward +j, got dj={cw - c0:+.2f}"
    assert abs((cw - c0) - field.flat[0] * trt) < 0.4, "shift magnitude must be field*TRT"
    # (a) round-trip returns the blob to where it started
    assert abs(cr - c0) < 0.3, f"round-trip must recover centroid, off by {cr - c0:+.2f}"
    # (c) the shift estimator itself is signed correctly: a +j warp reads as +, a -j warp as -.
    assert estimate_pe_shift(vol, warped, mask, 1) > 1.0
    warped_neg = fsl_convention_forward(vol, field, trt, "j-", jacobian=False)
    assert estimate_pe_shift(vol, warped_neg, mask, 1) < -1.0


def test_trxscan_fieldmap_roundtrip():
    trxscan = shutil.which("trxscan")
    reason = _skip_reason(DEFAULT_BUNDLE, trxscan)
    if reason:
        import pytest; pytest.skip(reason)
    with tempfile.TemporaryDirectory() as td:
        rep = run_roundtrip(DEFAULT_BUNDLE, trxscan, Path(td), fsl_orientation=True)
    assert not rep["orientation_fails"], "orientation contract: " + "; ".join(rep["orientation_fails"])
    assert rep["direction_ok"], (
        f"distortion direction disagrees with declared PED {rep['ped']!r}: the +field-vs-(-field) "
        f"image shifted {rep['measured_shift_vox']:+.2f} vox along axis {rep['pe_axis']} "
        f"(FSL convention needs this to match the sign of {rep['expected_shift_vox']:+.2f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--trxscan", default=shutil.which("trxscan"))
    ap.add_argument("--voxel", type=float, default=3.0)
    ap.add_argument("--real-ref", type=Path, default=None,
                    help="a real dcm2niix DWI to ground the orientation expectation")
    ap.add_argument("--keep", action="store_true", help="keep the work dir")
    ap.add_argument("--native", action="store_true",
                    help="test the native-grid output instead of --fsl-orientation")
    a = ap.parse_args()

    reason = _skip_reason(a.bundle, a.trxscan)
    if reason:
        print(f"SKIP: {reason}")
        return 0

    trxscan = _resolve_trxscan(a.trxscan)
    work = Path(tempfile.mkdtemp(prefix="trxscan_roundtrip_"))
    try:
        rep = run_roundtrip(a.bundle, trxscan, work, a.voxel, real_ref=a.real_ref,
                            fsl_orientation=not a.native)
    finally:
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)

    mode = "native grid" if a.native else "--fsl-orientation"
    print(f"\n=== TRXScan fieldmap round-trip ({mode}) ===")
    print(f"declared PhaseEncodingDirection : {rep['ped']}   TotalReadoutTime={rep['trt']:.5f}s")
    print("\n(1) ORIENTATION (must match dcm2niix / FSL: radiological, +j->Anterior):")
    if rep["orientation_fails"]:
        for f in rep["orientation_fails"]:
            print("    FAIL:", f)
    else:
        print("    PASS")
    print(f"\n(2) DISTORTION DIRECTION (+field must move signal toward +PED; +/-{rep['f0_hz']:.0f} Hz differential):")
    print(f"    (+f) vs (-f) shift = {rep['measured_shift_vox']:+.2f} vox along axis {rep['pe_axis']} "
          f"('{rep['ped']}')  -- the sign is what matters")
    print(f"    FSL convention expects the sign of {rep['expected_shift_vox']:+.2f}")
    print("    PASS" if rep["direction_ok"] else
          "    FAIL: TRXScan distorts opposite to its declared PED (SDC tools would double the distortion)")
    ok = not rep["orientation_fails"] and rep["direction_ok"]
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
