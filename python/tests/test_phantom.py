"""Tests on the full NIBS phantom: run with TRXSCAN_RUN_PHANTOM=1 and TRXSCAN_DATA pointing at
the kit (or after the data release)."""

import os
import subprocess
import time

import numpy as np
import pytest

import trxscan as ts
from trxscan.data import BUNDLES  # noqa: F401

pytestmark = pytest.mark.phantom


@pytest.fixture(scope="module")
def hbcd(phantom):
    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs

    bval, bvec = ts.data.scheme_files("sub-60501", "AP")
    return gradient_table(*read_bvals_bvecs(str(bval), str(bvec)))


def test_load_and_grid(phantom):
    assert phantom.n_streamlines == 1_000_000
    assert phantom.streamlines.weights is not None
    assert "AP" in phantom.motion and phantom.motion["AP"].n_vol == 76
    obj = phantom.grid(ts.Protocol.HBCD.replace(voxel_mm=2.5))
    assert obj.oversample == 2 and obj.sim_dims[:2] == (2 * obj.dims[0], 2 * obj.dims[1])


def test_one_slice_all_volumes_under_ten_seconds(phantom, hbcd):
    proto = ts.Protocol.HBCD.replace(voxel_mm=2.5, coils=8, accel=2)
    t = time.perf_counter()
    sim = phantom.simulate(hbcd, proto, ts.Artifacts(noise=2e-4, ghost=0.015), z_mm=10.0)
    assert time.perf_counter() - t < 60
    assert sim.magnitude.shape[3] == 76 and sim.kspace.acquired.shape[:3] == (76, 1, 8)
    rec = ts.KSpace.recon_reference(sim.kspace.reconstructed[:, 0], sim.kspace.sensitivities)
    assert np.abs(rec - sim.kspace.combined[:, 0]).max() < 1e-5 * np.abs(sim.kspace.combined).max()


def test_subsample_is_shared_by_truth_and_data(phantom, hbcd):
    sub = phantom.subsample(20000, seed=3)
    proto = ts.Protocol.HBCD.replace(voxel_mm=3.0, oversample=1)
    z = sub.grid(proto).z_of(10.0)
    a = sub.simulate(hbcd, proto, ts.Artifacts(), slices=z, kspace=False)
    b = sub.microstructure(proto, hbcd, slices=z)
    assert a.mixture is not None and b["fa"].shape == a.magnitude.shape[:3]


@pytest.mark.skipif(not os.environ.get("TRXSCAN_CLI"), reason="set TRXSCAN_CLI to the trxscan binary")
def test_bids_parity_with_the_cli(phantom, hbcd, tmp_path):
    import json

    import nibabel as nib
    from dipy.core.gradients import gradient_table

    gtab = gradient_table(hbcd.bvals[:6], bvecs=hbcd.bvecs[:6])
    proto = ts.Protocol.HBCD.replace(voxel_mm=3.0, coils=4, accel=2)
    obj = phantom.grid(proto)
    S = tmp_path
    for name in ("wm", "gm", "csf", "mask"):
        obj.image(name).to_filename(str(S / f"{name}.nii.gz"))
        obj.image("sim_" + name).to_filename(str(S / f"sim_{name}.nii.gz"))
    obj.image("sim_fmap").to_filename(str(S / "sim_fmap.nii.gz"))
    np.savetxt(S / "dwi.bval", gtab.bvals[None], fmt="%g")
    np.savetxt(S / "dwi.bvec", gtab.bvecs.T, fmt="%.8f")
    z = obj.z_of(10.0)
    art = ts.Artifacts(noise=2e-4, ghost=0.015, eddy=0.02, eddy_phase=1.7e-5, seed=3)
    sim = obj.simulate(gtab, proto, art, slices=z, kspace=False, truth_peaks=True)
    sim.to_bids(S / "py")
    trx = next(p for p in phantom.path.rglob("*.trx"))
    cmd = [os.environ["TRXSCAN_CLI"], "--wm", S / "wm.nii.gz", "--gm", S / "gm.nii.gz", "--csf", S / "csf.nii.gz", "--mask", S / "mask.nii.gz",
           "--sim-wm", S / "sim_wm.nii.gz", "--sim-gm", S / "sim_gm.nii.gz", "--sim-csf", S / "sim_csf.nii.gz", "--sim-mask", S / "sim_mask.nii.gz", "--sim-fmap", S / "sim_fmap.nii.gz",
           "--streamlines", trx, "--weights", "sift2_weights", "--bval", S / "dwi.bval", "--bvec", S / "dwi.bvec", "--oversample", "2", "--params", "adult",
           "--coils", "4", "--accel", "2", "--noise", "2e-4", "--eddy", "0.02", "--eddy-phase", "1.7e-5", "--seed", "3", "--truth-peaks", "--out", S / "cli"]
    subprocess.run([str(c) for c in cmd], check=True, capture_output=True)
    for part in ("mag", "phase"):
        c = np.asarray(nib.load(S / f"cli_part-{part}_dwi.nii.gz").dataobj)[:, :, z, :]
        p = np.asarray(nib.load(S / f"py_part-{part}_dwi.nii.gz").dataobj)[:, :, 0, :]
        assert np.array_equal(c, p), part
    # the Python sidecar is a superset (recorded-only keys, simulation stamp); the CLI's keys agree
    cli_side, py_side = json.load(open(S / "cli_part-mag_dwi.json")), json.load(open(S / "py_part-mag_dwi.json"))
    assert {k: cli_side[k] for k in cli_side if k != "Manufacturer"} == {k: py_side[k] for k in cli_side if k != "Manufacturer"}
    assert np.abs(np.loadtxt(S / "cli_dwi.bvec") - np.loadtxt(S / "py_dwi.bvec")).max() <= 1e-6
