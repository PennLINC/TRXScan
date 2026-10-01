"""The BIDS dataset writer and the mirror of a real dataset, on a tiny synthetic phantom."""

import json

import nibabel as nib
import numpy as np
import pytest

import trxscan as ts


@pytest.fixture(scope="module")
def tiny_phantom():
    """A 1 mm LPS anatomical grid (like the ACPC kit): a WM slab with GM/CSF rims, two
    straight fibre bundles, a T1w image."""
    nx, ny, nz = 30, 34, 10
    wm = np.zeros((nx, ny, nz), np.float32)
    gm = np.zeros_like(wm)
    csf = np.zeros_like(wm)
    wm[8:22, 10:24, 2:8] = 1.0
    gm[6:8, 8:26, 2:8] = 1.0
    gm[22:24, 8:26, 2:8] = 1.0
    csf[8:22, 8:10, 2:8] = 1.0
    csf[8:22, 24:26, 2:8] = 1.0
    aff = np.diag([-1.0, -1.0, 1.0, 1.0])
    aff[:3, 3] = [nx / 2, ny / 2, -nz / 2]
    img = lambda a: nib.Nifti1Image(a, aff)  # noqa: E731
    # fibres: lines along world x through the slab, and a few along y
    pts, offs = [], [0]
    for y in np.linspace(-6, 6, 7):
        for z in np.linspace(-2, 2, 3):
            line = np.stack([np.linspace(-6, 6, 25), np.full(25, y), np.full(25, z)], 1)
            pts.append(line); offs.append(offs[-1] + 25)
    for x in np.linspace(-4, 4, 5):
        line = np.stack([np.full(25, x), np.linspace(-6, 6, 25), np.zeros(25)], 1)
        pts.append(line); offs.append(offs[-1] + 25)
    sl = ts.Streamlines(np.concatenate(pts), np.array(offs, np.uint32), None)
    t1 = img((1000 * wm + 700 * gm + 200 * csf).astype(np.float32))
    return ts.Phantom.from_files(wm=img(wm), gm=img(gm), csf=img(csf), mask=img(((wm + gm + csf) > 0).astype(np.float32)),
                                 streamlines=sl, t1w=t1, name="tiny")


def _las_affine(shape, vox):
    aff = np.diag([-vox, vox, vox, 1.0])
    aff[:3, 3] = [shape[0] * vox / 2, -shape[1] * vox / 2, -shape[2] * vox / 2]
    return aff


def _write_source(root, *, complex_parts=True):
    """A fake raw dataset: one subject/session, AP+PA complex DWI with sbref, a T1w, a GRE
    phasediff fieldmap and a PEPOLAR epi. Images are zeros; only headers and sidecars matter."""
    sub, ses = "sub-01", "ses-a"
    base = root / sub / ses
    for d in ("anat", "dwi", "fmap"):
        (base / d).mkdir(parents=True)
    (root / "dataset_description.json").write_text(json.dumps({"Name": "FakeStudy", "BIDSVersion": "1.9.0"}))
    (root / "participants.tsv").write_text("participant_id\tage\tsex\nsub-01\t31\tF\n")
    shape, vox = (24, 28, 6), 2.0
    dwi_side = {
        "EchoTime": 0.088, "RepetitionTime": 4.8, "FlipAngle": 78, "TotalReadoutTime": 0.0917, "PartialFourier": 0.75,
        "ParallelReductionFactorInPlane": 2, "MultibandAccelerationFactor": 3, "MagneticFieldStrength": 3,
        "Manufacturer": "Siemens", "ManufacturersModelName": "Prisma", "PulseSequenceDetails": "%CustomerSeq%\\cmrr_mbep2d_diff",
        "ReceiveCoilName": "HeadNeck_64", "SliceTiming": [0.0, 2.4, 0.0, 2.4, 0.0, 2.4],
        "ConversionSoftware": "dcm2niix", "SeriesNumber": 7,
    }
    bvals = np.array([0, 1000, 1000, 1000])
    bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    for d, ped in (("AP", "j-"), ("PA", "j")):
        stem = f"{sub}_{ses}_acq-test_dir-{d}_run-01"
        parts = ("part-mag_", "part-phase_") if complex_parts else ("",)
        for part in parts:
            for suf in ("dwi", "sbref"):
                n = len(bvals) if suf == "dwi" else 1
                nib.Nifti1Image(np.zeros(shape + (n,), np.float32), _las_affine(shape, vox)).to_filename(str(base / "dwi" / f"{stem}_{part}{suf}.nii.gz"))
                (base / "dwi" / f"{stem}_{part}{suf}.json").write_text(json.dumps({**dwi_side, "PhaseEncodingDirection": ped}))
        np.savetxt(base / "dwi" / f"{stem}_{parts[0]}dwi.bval", bvals[None], fmt="%g")
        np.savetxt(base / "dwi" / f"{stem}_{parts[0]}dwi.bvec", bvecs.T, fmt="%.6f")
    t1_shape = (40, 44, 12)
    nib.Nifti1Image(np.zeros(t1_shape, np.float32), _las_affine(t1_shape, 1.5)).to_filename(str(base / "anat" / f"{sub}_{ses}_acq-MPRAGE_T1w.nii.gz"))
    (base / "anat" / f"{sub}_{ses}_acq-MPRAGE_T1w.json").write_text(json.dumps({"EchoTime": 0.00281, "RepetitionTime": 2.4, "InversionTime": 0.9, "FlipAngle": 8, "Manufacturer": "Siemens", "SeriesNumber": 3}))
    f_shape = (12, 14, 6)
    for name, meta in (("magnitude1", {"EchoTime": 0.00492}), ("magnitude2", {"EchoTime": 0.00738}), ("phasediff", {"EchoTime1": 0.00492, "EchoTime2": 0.00738, "B0FieldIdentifier": "gre0"})):
        nib.Nifti1Image(np.zeros(f_shape, np.float32), _las_affine(f_shape, 4.0)).to_filename(str(base / "fmap" / f"{sub}_{ses}_acq-gre_{name}.nii.gz"))
        (base / "fmap" / f"{sub}_{ses}_acq-gre_{name}.json").write_text(json.dumps(meta))
    nib.Nifti1Image(np.zeros(shape + (2,), np.float32), _las_affine(shape, vox)).to_filename(str(base / "fmap" / f"{sub}_{ses}_dir-PA_epi.nii.gz"))
    (base / "fmap" / f"{sub}_{ses}_dir-PA_epi.json").write_text(json.dumps({**dwi_side, "PhaseEncodingDirection": "j", "B0FieldIdentifier": "pepolar", "IntendedFor": ["bids::sub-01/ses-a/dwi/x.nii.gz"]}))
    nib.Nifti1Image(np.zeros(f_shape, np.float32), _las_affine(f_shape, 4.0)).to_filename(str(base / "fmap" / f"{sub}_{ses}_acq-famp_TB1TFL.nii.gz"))
    return root


def test_names_round_trip():
    ents, suf, ext = ts.bids.split_name("sub-01_ses-a_acq-x_dir-AP_run-01_part-mag_dwi.nii.gz")
    assert suf == "dwi" and ext == ".nii.gz" and ents["part"] == "mag"
    assert ts.bids.bids_name(ents, suf, ext) == "sub-01_ses-a_acq-x_dir-AP_run-01_part-mag_dwi.nii.gz"
    assert ts.bids.bids_name({"dir": "AP", "sub": "1", "ses": None}, "dwi") == "sub-1_dir-AP_dwi"


def test_recorded_fields_and_slice_timing():
    st = ts.protocol.siemens_slice_timing(87, 3, 4.8)
    dt = 4.8 / 29
    assert [round(st[z] / dt) for z in (0, 1, 2, 29, 30)] == [0, 15, 1, 0, 15]  # the CMRR order dcm2niix reports
    p = ts.Protocol.HBCD.replace(tr_s=4.8, flip_angle_deg=78, mb=3, metadata={"Manufacturer": "Siemens", "ReceiveCoilName": "HeadNeck_64"})
    side = p.sidecar(140, nz=87)
    assert side["RepetitionTime"] == 4.8 and side["FlipAngle"] == 78 and side["Manufacturer"] == "Siemens"
    assert side["PhaseEncodingSteps"] == 105 and side["EchoTrainLength"] == 105 and side["ReconMatrixPE"] == 140
    assert side["BandwidthPerPixelPhaseEncode"] == pytest.approx(10.83, abs=0.01)
    assert len(side["SliceTiming"]) == 87 and side["SimulationSoftware"] == "TRXScan"
    assert "SliceTiming" not in p.sidecar(140, nz=88)  # 88 slices do not split into MB-3 groups
    assert ts.Protocol.HBCD.replace(slice_timing=(0.0, 1.0)).slice_timing_for(2) == (0.0, 1.0)
    with pytest.raises(ValueError):
        ts.Protocol(matrix=(0, 1, 1))


def test_from_bids_reads_recorded_fields_and_matrix(tmp_path):
    src = _write_source(tmp_path / "src")
    real = ts.BidsDwi.load(src / "sub-01/ses-a/dwi/sub-01_ses-a_acq-test_dir-AP_run-01_part-mag_dwi.nii.gz")
    p = real.protocol
    assert p.tr_s == 4.8 and p.flip_angle_deg == 78 and p.mb == 3 and p.accel == 2 and p.pe == "j-"
    assert p.matrix == (24, 28, 6) and p.voxel == (2.0, 2.0, 2.0) and p.fsl_orientation
    assert p.slice_timing == (0.0, 2.4, 0.0, 2.4, 0.0, 2.4)
    assert p.metadata["ManufacturersModelName"] == "Prisma" and "SeriesNumber" not in p.metadata and "ConversionSoftware" not in p.metadata
    epi = ts.BidsDwi.load_epi(src / "sub-01/ses-a/fmap/sub-01_ses-a_dir-PA_epi.nii.gz")
    assert epi.n_vol == 2 and epi.protocol.pe == "j"


def test_grid_centres_the_object_in_a_fixed_matrix(tiny_phantom):
    proto = ts.Protocol.DEFAULT.replace(voxel_mm=2.0, oversample=1, matrix=(24, 28, 6))
    obj = tiny_phantom.grid(proto)
    assert obj.dims == (24, 28, 6)
    assert np.allclose(np.linalg.norm(obj.affine[:3, :3], axis=0), 2.0)
    m = obj.mask.reshape(6, 28, 24)  # (z, y, x) view of the flat F-order array
    occ = np.argwhere(m > 0)
    lo, hi = occ.min(0), occ.max(0)
    for ax, n in zip(range(3), (6, 28, 24)):
        assert abs(lo[ax] - (n - 1 - hi[ax])) <= 1, f"axis {ax} not centred: {lo[ax]} vs {n - 1 - hi[ax]}"
    with pytest.warns(UserWarning):
        tiny_phantom.grid(proto.replace(matrix=(4, 28, 6)))


def test_dataset_writer_on_a_box(gtab6, tmp_path):
    box = ts.objects.box(8, matrix=24, oversample=1, nz=3)
    proto = ts.Protocol.HBCD.replace(voxel_mm=2.0, oversample=1, mb=3, tr_s=3.0, flip_angle_deg=90)
    sim = box.simulate(gtab6, proto, ts.Artifacts(noise_map=np.full((24, 24, 3), 0.01, np.float32), dropout=1.0, seed=1), kspace=False, truth_peaks=True, gre=ts.Gre())
    ds = ts.Dataset(tmp_path / "ds", name="box", authors=("A. Tester",))
    files = ds.add_dwi(sim, "01", run="01", acq="box")
    names = {p.name for p in files}
    assert names == {
        "sub-01_acq-box_dir-AP_run-01_part-mag_dwi.nii.gz", "sub-01_acq-box_dir-AP_run-01_part-mag_dwi.json",
        "sub-01_acq-box_dir-AP_run-01_part-phase_dwi.nii.gz", "sub-01_acq-box_dir-AP_run-01_part-phase_dwi.json",
        "sub-01_acq-box_dir-AP_run-01_dwi.bval", "sub-01_acq-box_dir-AP_run-01_dwi.bvec",
        "sub-01_run-01_magnitude1.nii.gz", "sub-01_run-01_magnitude1.json", "sub-01_run-01_magnitude2.nii.gz", "sub-01_run-01_magnitude2.json",
        "sub-01_run-01_phasediff.nii.gz", "sub-01_run-01_phasediff.json",
    }
    side = json.load(open(tmp_path / "ds/sub-01/dwi/sub-01_acq-box_dir-AP_run-01_part-mag_dwi.json"))
    assert side["RepetitionTime"] == 3.0 and side["FlipAngle"] == 90 and len(side["SliceTiming"]) == 3
    assert side["B0FieldSource"] == "b0gre" and "ImageComparison" not in side
    phase = json.load(open(tmp_path / "ds/sub-01/dwi/sub-01_acq-box_dir-AP_run-01_part-phase_dwi.json"))
    assert phase["Units"] == "rad"
    fm = json.load(open(tmp_path / "ds/sub-01/fmap/sub-01_run-01_phasediff.json"))
    assert fm["B0FieldIdentifier"] == "b0gre" and fm["IntendedFor"] == ["bids::sub-01/dwi/sub-01_acq-box_dir-AP_run-01_part-mag_dwi.nii.gz"]
    assert nib.load(tmp_path / "ds/sub-01/dwi/sub-01_acq-box_dir-AP_run-01_part-mag_dwi.nii.gz").header.get_zooms()[3] == 3.0
    deriv = tmp_path / "ds/derivatives/trxscan/sub-01/dwi"
    assert {p.name for p in deriv.iterdir()} >= {"sub-01_acq-box_dir-AP_run-01_desc-truthpeaks_dwi.nii.gz", "sub-01_acq-box_dir-AP_run-01_desc-noisesigma_dwi.nii.gz", "sub-01_acq-box_dir-AP_run-01_desc-dropout_dwi.tsv"}
    assert not any(p.name.startswith("sub-01_acq-box_dir-AP_run-01_desc") for p in (tmp_path / "ds/sub-01/dwi").iterdir())
    desc = json.load(open(tmp_path / "ds/dataset_description.json"))
    assert desc["Name"] == "box" and desc["Authors"] == ["A. Tester"] and desc["GeneratedBy"][0]["Name"] == "TRXScan"
    assert (tmp_path / "ds/participants.tsv").read_text() == "participant_id\nsub-01\n"
    # reopening keeps the participants; magnitude-only output drops the part entity
    ds2 = ts.Dataset(tmp_path / "ds")
    ds2.add_dwi(sim, "02", parts=None, truth=False)
    assert (tmp_path / "ds/sub-02/dwi/sub-02_dir-AP_dwi.nii.gz").exists() and not (tmp_path / "ds/sub-02/dwi/sub-02_dir-AP_part-mag_dwi.nii.gz").exists()
    assert (tmp_path / "ds/participants.tsv").read_text() == "participant_id\nsub-01\nsub-02\n"


def test_mirror_reproduces_the_source_layout(tiny_phantom, tmp_path):
    src = _write_source(tmp_path / "src")
    seen = []
    ds = ts.Dataset.mirror(src, tiny_phantom, tmp_path / "out", artifacts=ts.Artifacts(noise=1e-4, seed=2), log=seen.append,
                           protocol=lambda p: p.replace(oversample=1, coils=2))
    out = tmp_path / "out"
    dwi = out / "sub-01/ses-a/dwi"
    expect = {f"sub-01_ses-a_acq-test_dir-{d}_run-01_{p}{s}" for d in ("AP", "PA") for p in ("part-mag_", "part-phase_") for s in ("dwi.nii.gz", "dwi.json", "sbref.nii.gz", "sbref.json")}
    expect |= {f"sub-01_ses-a_acq-test_dir-{d}_run-01_dwi.{e}" for d in ("AP", "PA") for e in ("bval", "bvec")}
    assert {p.name for p in dwi.iterdir()} == expect
    img = nib.load(dwi / "sub-01_ses-a_acq-test_dir-AP_run-01_part-mag_dwi.nii.gz")
    assert img.shape == (24, 28, 6, 4) and nib.aff2axcodes(img.affine) == ("L", "A", "S")
    assert img.header.get_zooms() == (2.0, 2.0, 2.0, 4.8)
    assert np.isfinite(img.get_fdata()).all() and img.get_fdata().max() > 0
    side = json.load(open(dwi / "sub-01_ses-a_acq-test_dir-AP_run-01_part-mag_dwi.json"))
    pa = json.load(open(dwi / "sub-01_ses-a_acq-test_dir-PA_run-01_part-mag_dwi.json"))
    assert side["PhaseEncodingDirection"] == "j-" and pa["PhaseEncodingDirection"] == "j"
    assert side["RepetitionTime"] == 4.8 and side["FlipAngle"] == 78 and side["MultibandAccelerationFactor"] == 3
    assert side["Manufacturer"] == "Siemens" and side["ManufacturersModelName"] == "Prisma" and side["SimulationSoftware"] == "TRXScan"
    assert side["ConversionSoftware"] == "trxscan" and "SeriesNumber" not in side
    assert side["SliceTiming"] == [0.0, 2.4, 0.0, 2.4, 0.0, 2.4] and side["TotalReadoutTime"] == pytest.approx(0.0917)
    assert side["ReconMatrixPE"] == 28 and side["B0FieldSource"] == "gre0"
    assert np.loadtxt(dwi / "sub-01_ses-a_acq-test_dir-AP_run-01_dwi.bval").tolist() == [0, 1000, 1000, 1000]
    sb = nib.load(dwi / "sub-01_ses-a_acq-test_dir-AP_run-01_part-mag_sbref.nii.gz")
    assert sb.shape == (24, 28, 6, 1)
    # anat: the phantom's T1w at the source's voxel size and axes
    t1 = nib.load(out / "sub-01/ses-a/anat/sub-01_ses-a_acq-MPRAGE_T1w.nii.gz")
    assert t1.header.get_zooms()[:3] == (1.5, 1.5, 1.5) and nib.aff2axcodes(t1.affine) == ("L", "A", "S")
    t1j = json.load(open(out / "sub-01/ses-a/anat/sub-01_ses-a_acq-MPRAGE_T1w.json"))
    assert t1j["InversionTime"] == 0.9 and t1j["Description"].startswith("Subject T1w") and "SeriesNumber" not in t1j
    # fmap: the GRE at the source's echo times and resolution, the epi with the source's IntendedFor rewritten
    fmap = out / "sub-01/ses-a/fmap"
    assert {p.name for p in fmap.iterdir()} == {
        "sub-01_ses-a_acq-gre_magnitude1.nii.gz", "sub-01_ses-a_acq-gre_magnitude1.json", "sub-01_ses-a_acq-gre_magnitude2.nii.gz", "sub-01_ses-a_acq-gre_magnitude2.json",
        "sub-01_ses-a_acq-gre_phasediff.nii.gz", "sub-01_ses-a_acq-gre_phasediff.json", "sub-01_ses-a_dir-PA_epi.nii.gz", "sub-01_ses-a_dir-PA_epi.json",
    }
    pd = json.load(open(fmap / "sub-01_ses-a_acq-gre_phasediff.json"))
    assert pd["B0FieldIdentifier"] == "gre0" and pd["EchoTime1"] == 0.00492 and pd["EchoTime2"] == 0.00738
    assert nib.load(fmap / "sub-01_ses-a_acq-gre_phasediff.nii.gz").header.get_zooms()[0] == pytest.approx(4.0)
    epi = nib.load(fmap / "sub-01_ses-a_dir-PA_epi.nii.gz")
    epij = json.load(open(fmap / "sub-01_ses-a_dir-PA_epi.json"))
    assert epi.shape == (24, 28, 6, 2) and epij["PhaseEncodingDirection"] == "j" and epij["B0FieldIdentifier"] == "pepolar"
    assert epij["IntendedFor"] == ["bids::sub-01/ses-a/dwi/sub-01_ses-a_acq-test_dir-AP_run-01_part-mag_dwi.nii.gz", "bids::sub-01/ses-a/dwi/sub-01_ses-a_acq-test_dir-PA_run-01_part-mag_dwi.nii.gz"]
    assert any("TB1TFL" in line for line in seen)
    # dataset level: no demographics copied, source recorded, sessions listed, truth in derivatives
    assert (out / "participants.tsv").read_text() == "participant_id\nsub-01\n"
    assert (out / "sub-01/sub-01_sessions.tsv").read_text() == "session_id\nses-a\n"
    desc = json.load(open(out / "dataset_description.json"))
    assert desc["SourceDatasets"][0]["Name"] == "FakeStudy"
    assert (out / "derivatives/trxscan/sub-01/ses-a/dwi/sub-01_ses-a_acq-test_dir-AP_run-01_desc-truthpeaks_dwi.nii.gz").exists()
    ok, text = ds.validate("--ignoreWarnings")
    if "not found" not in text:
        assert ok, text


def test_mirror_magnitude_only_and_synthetic_anat(tiny_phantom, tmp_path):
    src = _write_source(tmp_path / "src", complex_parts=False)
    ph = ts.Phantom(wm=tiny_phantom.wm, gm=tiny_phantom.gm, csf=tiny_phantom.csf, mask=tiny_phantom.mask, streamlines=tiny_phantom.streamlines)
    ts.Dataset.mirror(src, {"sub-01": ph}, tmp_path / "out", fmap=False, sbref=False, truth=False, log=None, protocol=lambda p: p.replace(oversample=1))
    dwi = tmp_path / "out/sub-01/ses-a/dwi"
    assert {p.name for p in dwi.iterdir()} == {f"sub-01_ses-a_acq-test_dir-{d}_run-01_dwi.{e}" for d in ("AP", "PA") for e in ("nii.gz", "json", "bval", "bvec")}
    t1j = json.load(open(tmp_path / "out/sub-01/ses-a/anat/sub-01_ses-a_acq-MPRAGE_T1w.json"))
    assert t1j["Description"].startswith("Synthetic T1w")
    assert not (tmp_path / "out/derivatives/trxscan/sub-01").exists()
    with pytest.raises(ValueError):
        ts.bids.phantom_anat(ph, "T1w", "phantom")
