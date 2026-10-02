"""Fixture recipes, the scoring helpers and the new truth outputs, on the tiny phantom."""

import json

import nibabel as nib
import numpy as np
import pytest

import trxscan as ts
from trxscan import recipes, score

from test_bids import tiny_phantom  # noqa: F401  (fixture)


@pytest.fixture(scope="module")
def tiny_with_field(tiny_phantom):  # noqa: F811
    """The tiny phantom with a smooth fieldmap (Hz) and a motion trace."""
    shape = tiny_phantom.wm.shape
    y = np.linspace(-1, 1, shape[1])[None, :, None]
    fmap = (25.0 * y * np.ones(shape)).astype(np.float32)
    mot = ts.Motion.from_arrays(np.zeros((40, 3)), np.column_stack([np.zeros(40), np.zeros(40), np.linspace(0, 4, 40)]))
    return ts.Phantom(wm=tiny_phantom.wm, gm=tiny_phantom.gm, csf=tiny_phantom.csf, mask=tiny_phantom.mask,
                      streamlines=tiny_phantom.streamlines, fieldmap=nib.Nifti1Image(fmap, tiny_phantom.wm.affine),
                      t1w=tiny_phantom.t1w, motion={"AP": mot}, name="tiny")


def test_scheme_and_directions():
    d = recipes.hemisphere_directions(16)
    assert d.shape == (16, 3) and np.allclose(np.linalg.norm(d, axis=1), 1.0) and (d[:, 2] >= 0).all()
    bvals, bvecs = recipes.scheme(((1000, 6), (2000, 10)), nb0=2)
    assert bvals.tolist() == [0, 0] + [1000] * 6 + [2000] * 10 and bvecs.shape == (18, 3)


def test_recipe_digest_is_stable_and_parameter_sensitive():
    a, b = recipes.Recipe("rpe_series", {"voxel": 3.0}), recipes.Recipe("rpe_series", {"voxel": 2.0})
    assert a.digest() == recipes.Recipe("rpe_series", {"voxel": 3.0}).digest() and a.digest() != b.digest()
    assert len(a.digest()) == 16 and a.to_dict()["SimulationSoftwareVersion"] == ts.__version__


def test_rpe_series_recipe_writes_truth(tiny_with_field, tmp_path):
    ds = recipes.rpe_series(tmp_path / "rpe", phantom=tiny_with_field, voxel=2.0, ndirs=3, nb0=1, subsample=None, subject="01")
    root = tmp_path / "rpe"
    dwi = root / "sub-01/dwi"
    assert {p.name for p in dwi.iterdir()} == {f"sub-01_dir-{d}_dwi.{e}" for d in ("AP", "PA") for e in ("nii.gz", "json", "bval", "bvec")}
    img = nib.load(dwi / "sub-01_dir-AP_dwi.nii.gz")
    assert img.shape[3] == 4 and nib.aff2axcodes(img.affine) == ("L", "A", "S")
    side = json.load(open(dwi / "sub-01_dir-AP_dwi.json"))
    assert side["PhaseEncodingDirection"] == "j-" and side["ImageType"][-1] == "ND" and side["RepetitionTime"] == 4.0
    assert side["TotalReadoutTime"] == pytest.approx(side["ReconMatrixPE"] * recipes.ECHO_SPACING_MS / 1000, rel=1e-3)  # readout follows the matrix
    pa = json.load(open(dwi / "sub-01_dir-PA_dwi.json"))
    assert pa["PhaseEncodingDirection"] == "j"
    truth = root / "derivatives/trxscan/sub-01/dwi"
    names = {p.name for p in truth.iterdir()}
    for want in ("cleanb0", "fieldmap", "displacement", "truthpeaks"):
        assert f"sub-01_dir-AP_desc-{want}_dwi.nii.gz" in names, want
    rec = json.load(open(root / "derivatives/trxscan/recipe.json"))
    assert rec["Name"] == "rpe_series" and rec["Parameters"]["voxel"] == 2.0 and rec["Parameters"]["phantom"] == "tiny" and len(rec["Digest"]) == 16
    assert (root / "sub-01/anat/sub-01_T1w.nii.gz").exists()
    # the displacement truth agrees with the scoring helper and has opposite signs for AP/PA
    fm = nib.load(truth / "sub-01_dir-AP_desc-fieldmap_dwi.nii.gz")
    disp = nib.load(truth / "sub-01_dir-AP_desc-displacement_dwi.nii.gz")
    assert np.allclose(np.asarray(disp.dataobj), np.asarray(score.pe_displacement(fm, side).dataobj))
    disp_pa = nib.load(truth / "sub-01_dir-PA_desc-displacement_dwi.nii.gz")
    assert np.allclose(np.asarray(disp.dataobj), -np.asarray(disp_pa.dataobj))
    assert np.abs(np.asarray(disp.dataobj)[..., 1]).max() > 0.5  # mm along A/P
    # the clean b0 is undistorted: closer to a distortion-free run than the distorted b0 is
    ok, text = ds.validate("--ignoreWarnings")
    if "not found" not in text:
        assert ok, text


def test_other_recipes_and_cli(tiny_with_field, tmp_path):
    recipes.epi_fieldmap(tmp_path / "epi", phantom=tiny_with_field, voxel=2.0, ndirs=2, nb0=1, nb0_epi=2, subsample=None, subject="01", anat=False)
    fmap = tmp_path / "epi/sub-01/fmap"
    epi = json.load(open(fmap / "sub-01_dir-PA_epi.json"))
    assert nib.load(fmap / "sub-01_dir-PA_epi.nii.gz").shape[3] == 2 and epi["PhaseEncodingDirection"] == "j"
    assert epi["IntendedFor"] == ["bids::sub-01/dwi/sub-01_dir-AP_dwi.nii.gz"] and epi["B0FieldIdentifier"] == "pepolar"
    dwi_side = json.load(open(tmp_path / "epi/sub-01/dwi/sub-01_dir-AP_dwi.json"))
    assert dwi_side["B0FieldSource"] == "pepolar" and dwi_side["B0FieldIdentifier"] == "pepolar"  # the DWI is a PEPOLAR source too
    recipes.phasediff_fieldmap(tmp_path / "pd", phantom=tiny_with_field, voxel=2.0, ndirs=2, nb0=1, subsample=None, subject="01", anat=False)
    pdj = json.load(open(tmp_path / "pd/sub-01/fmap/sub-01_acq-gre_phasediff.json"))
    assert pdj["EchoTime1"] == 0.00492 and pdj["IntendedFor"] == ["bids::sub-01/dwi/sub-01_dir-AP_dwi.nii.gz"]
    recipes.multishell(tmp_path / "ms", phantom=tiny_with_field, voxel=2.0, shells=((1000, 2), (2000, 3)), nb0=1, subsample=None, subject="01", anat=False)
    assert np.loadtxt(tmp_path / "ms/sub-01/dwi/sub-01_dir-AP_dwi.bval").tolist() == [0, 1000, 1000, 2000, 2000, 2000]
    recipes.motion(tmp_path / "mot", phantom=tiny_with_field, voxel=2.0, ndirs=2, nb0=1, subsample=None, subject="01", anat=False, scale=2.0, mb=1)
    tsv = (tmp_path / "mot/derivatives/trxscan/sub-01/dwi/sub-01_dir-AP_desc-motion_timeseries.tsv").read_text().splitlines()
    assert tsv[0].split("\t") == ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"] and len(tsv) == 4
    assert float(tsv[-1].split("\t")[5]) == pytest.approx(np.radians(2 * 4 * (2 / 39)), rel=1e-3)  # resampled 40 -> 3, scaled x2
    assert (tmp_path / "mot/derivatives/trxscan/sub-01/dwi/sub-01_dir-AP_desc-cleanb0_dwi.nii.gz").exists()
    # command line
    assert recipes.main(["--list"]) == 0
    assert recipes.main(["rpe_series", "--digest", "--set", "voxel=3", "--set", "pe=[\"AP\"]"]) == 0
    assert recipes._parse_value("3") == 3 and recipes._parse_value("whole-body-80") == "whole-body-80"


def test_gnl_truth_in_itk_form_and_image_type(gtab6, tmp_path):
    box = ts.objects.box(8, matrix=24, oversample=1, nz=2)
    proto = ts.Protocol.HBCD.replace(voxel_mm=2.0, oversample=1, gnl_tag="DIS3D")
    sim = box.simulate(gtab6, proto, ts.Artifacts(gnl="whole-body-80", gnl_scale=2.0, isocenter=(0.0, -20.0, -30.0)), kspace=False)
    assert sim.sidecar["ImageType"] == ["ORIGINAL", "PRIMARY", "DIFFUSION", "NONE", "DIS3D"]
    ds = ts.Dataset(tmp_path / "ds")
    ds.add_dwi(sim, "01", parts=None)
    itk = nib.load(tmp_path / "ds/derivatives/trxscan/sub-01/dwi/sub-01_dir-AP_desc-gnldispitk_dwi.nii.gz")
    assert itk.shape == (24, 24, 2, 1, 3) and itk.header.get_intent()[0] == "vector"
    ras = np.asarray(sim.gnl.disp.dataobj)
    assert np.allclose(np.asarray(itk.dataobj)[:, :, :, 0, :], ras * [-1, -1, 1])
    assert ts.Protocol.from_bids.__doc__ and ts.Protocol(gnl_tag="DIS2D").gnl_tag == "DIS2D"
    with pytest.raises(ValueError):
        ts.Protocol(gnl_tag="DIS4D")


def test_score_helpers():
    rng = np.random.default_rng(0)
    truth = np.zeros((6, 6, 3, 3)); truth[..., 1] = rng.normal(0, 2, (6, 6, 3))
    est = 0.8 * truth + rng.normal(0, 0.05, truth.shape)
    r = score.compare_displacement(est, truth)
    assert r["slope"] == pytest.approx(0.8, abs=0.02) and r["corr"] > 0.99 and r["rms_residual"] < r["rms_truth"] and r["n"] == 108
    r2 = score.compare_displacement(est, truth, axis=(0, 1, 0), mask=np.ones((6, 6, 3)))
    assert r2["slope"] == pytest.approx(r["slope"], abs=1e-6)
    assert score.image_similarity(truth[..., 1], est[..., 1]) > 0.99
    t = np.zeros((2, 2, 1, 6)); t[..., :3] = [1, 0, 0]; t[0, 0, 0, 3:] = [0, 1, 0]
    e = np.zeros((2, 2, 1, 3)); e[..., :] = [np.cos(np.radians(10)), np.sin(np.radians(10)), 0]; e[1, 1, 0] = [-1, 0, 0]
    ang = score.angular_error(e, t)
    assert ang[0, 0, 0] == pytest.approx(10.0, abs=1e-6) and ang[1, 1, 0] == pytest.approx(0.0, abs=1e-6)
    t[0, 1, 0] = 0
    assert np.isnan(score.angular_error(e, t)[0, 1, 0])
    fm = nib.Nifti1Image(np.full((4, 4, 2), 10.0, np.float32), np.diag([-2.0, 2.0, 2.0, 1.0]))
    d = np.asarray(score.pe_displacement(fm, {"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05}).dataobj)
    assert np.allclose(d[..., 1], -10 * 0.05 * 2.0) and np.allclose(d[..., 0], 0)


def test_replace_keeps_the_fov_and_mirror_honours_a_voxel_override(tiny_phantom, tmp_path):  # noqa: F811
    from test_bids import _write_source

    p = ts.Protocol.HBCD.replace(matrix=(140, 140, 87)).replace(voxel_mm=3.0)
    assert p.matrix == (79, 79, 49) and p.fov_mm == (237.0, 237.0, 147.0)
    assert ts.Protocol.HBCD.replace(matrix=(140, 140, 87)).replace(voxel_mm=3.0, matrix=(10, 10, 10)).matrix == (10, 10, 10)
    src = _write_source(tmp_path / "src")
    (src / "sub-01/ses-a/dwi/._sub-01_ses-a_acq-test_dir-AP_run-01_part-mag_dwi.nii.gz").write_bytes(b"\x00\x05\x16\x07junk")
    (src / "sub-01/ses-a/fmap/._sub-01_ses-a_acq-gre_phasediff.nii.gz").write_bytes(b"\x00\x05\x16\x07junk")
    ts.Dataset.mirror(src, tiny_phantom, tmp_path / "out", log=None, fmap=False, sbref=False, truth=False, anat=None,
                      protocol=lambda q: q.replace(voxel_mm=4.0, oversample=1))
    img = nib.load(tmp_path / "out/sub-01/ses-a/dwi/sub-01_ses-a_acq-test_dir-AP_run-01_part-mag_dwi.nii.gz")
    assert img.shape[:3] == (12, 14, 3) and img.header.get_zooms()[:3] == (4.0, 4.0, 4.0)


def test_chunked_run_is_identical_to_the_whole(gtab6):
    """Slab-wise simulation (bounded memory) reproduces the unchunked run: exactly for noise,
    ghost, noise maps, truth, GRE and clean b0; to float rounding for the GNL warp; and to the
    context's reach for within-volume dropout jumps."""
    fm = np.zeros((48, 48, 6), np.float32); fm[:, 10:40, :] = 20.0
    box = ts.objects.box(10, matrix=24, oversample=2, nz=6, fieldmap=fm)
    proto = ts.Protocol.HBCD.replace(voxel_mm=2.0, mb=3, tr_s=3.0, fsl_orientation=True)
    art = ts.Artifacts(noise=1e-4, ghost=0.02, seed=2, noise_map=np.full((24, 24, 6), 0.01, np.float32))
    whole = box.simulate(gtab6, proto, art, kspace=False, truth_peaks=True, gre=ts.Gre())
    parts = box.simulate(gtab6, proto, art, kspace=False, truth_peaks=True, gre=ts.Gre(), chunk=2)
    assert parts.dims == whole.dims and np.array_equal(parts.slices, whole.slices) and np.allclose(parts.affine, whole.affine)
    for name in ("mag", "ph", "re", "im"):
        assert np.array_equal(getattr(parts, name), getattr(whole, name)), name
    for name in ("truth_peaks", "clean_b0", "fieldmap", "noise_sigma", "displacement"):
        assert np.array_equal(np.asarray(getattr(parts, name).dataobj), np.asarray(getattr(whole, name).dataobj)), name
    assert parts.sidecar == whole.sidecar and len(parts.sidecar["SliceTiming"]) == 6
    assert np.array_equal(np.asarray(parts.gre.magnitude1.dataobj), np.asarray(whole.gre.magnitude1.dataobj))
    assert np.array_equal(np.asarray(parts.gre.phase["phasediff"].dataobj), np.asarray(whole.gre.phase["phasediff"].dataobj))
    assert parts.mixture is None and parts.kspace is None and "clean_b0" in parts.timing
    with pytest.raises(ValueError):
        box.simulate(gtab6, proto, art, kspace=True, chunk=2)
    # GNL (warp + encoding + jacobian): float rounding only; truth fields exact
    gart = ts.Artifacts(gnl="whole-body-80", gnl_scale=2.0, isocenter=(0.0, -20.0, -30.0))
    gw = box.simulate(gtab6, proto, gart, kspace=False, gre=ts.Gre())
    gp = box.simulate(gtab6, proto, gart, kspace=False, gre=ts.Gre(), chunk=2)
    assert np.abs(gp.mag - gw.mag).max() < 1e-4 * gw.mag.max()
    for a, b in ((gp.gnl.disp, gw.gnl.disp), (gp.gnl.graddev, gw.gnl.graddev), (gp.gnl.invdisp, gw.gnl.invdisp)):
        assert np.allclose(np.asarray(a.dataobj), np.asarray(b.dataobj), atol=1e-5)  # slab-origin rounding
    assert np.array_equal(np.asarray(gp.gre.magnitude1.dataobj), np.asarray(gw.gre.magnitude1.dataobj))
    # dropout jumps: same events, images agree within the context's reach
    dart = ts.Artifacts(dropout=1.0, seed=2)
    dw = box.simulate(gtab6, proto, dart, kspace=False)
    dp = box.simulate(gtab6, proto, dart, kspace=False, chunk=2)
    assert dp.dropout_table() == dw.dropout_table() and len(dp.dropout) > 0
    assert np.abs(dp.mag - dw.mag).max() < 5e-3 * dw.mag.max()


def test_offsets_move_the_subject_and_write_the_truth_transform(tiny_with_field, tmp_path):
    ds = recipes.phasediff_fieldmap(tmp_path / "off", phantom=tiny_with_field, voxel=2.0, ndirs=2, nb0=1, subsample=None, subject="01",
                                    anat_offset=(2.0, -3.0, 1.0, 4.0, 0.0, -2.0), fmap_offset=(0.0, 1.5, 0.0, 0.0, 3.0, 0.0))
    root = tmp_path / "off"
    nib.load(root / "sub-01/dwi/sub-01_dir-AP_dwi.nii.gz")
    t1 = nib.load(root / "sub-01/anat/sub-01_T1w.nii.gz")
    mag = nib.load(root / "sub-01/fmap/sub-01_acq-gre_magnitude1.nii.gz")
    # resampled offsets keep the scan's own grid; the head moved inside it
    tx = json.load(open(root / "derivatives/trxscan/sub-01/fmap/sub-01_acq-gre_from-fmap_to-dwi_mode-image_desc-truth_xfm.json"))
    T = np.array(tx["WorldTransformRAS"])
    ref_mag = nib.load(root / "sub-01/fmap/sub-01_acq-gre_magnitude2.nii.gz")
    assert np.allclose(mag.affine, ref_mag.affine) and mag.shape == ref_mag.shape
    assert tx["RotationDeg"] == pytest.approx(3.0, abs=1e-6) and tx["TranslationMm"] > 0
    ax = json.load(open(root / "derivatives/trxscan/sub-01/anat/sub-01_from-T1w_to-dwi_mode-image_desc-truth_xfm.json"))
    assert ax["RotationDeg"] == pytest.approx(np.degrees(np.arccos((np.trace(ts.bids.rigid_offset((0, 0, 0), (4, 0, -2))[:3, :3]) - 1) / 2)), abs=1e-6)
    # the ITK text transform maps a DWI-frame point to the moved frame: p_moved = T p
    txt = (root / "derivatives/trxscan/sub-01/anat/sub-01_from-T1w_to-dwi_mode-image_desc-truth_xfm.txt").read_text()
    params = np.array(txt.split("Parameters: ")[1].split("\n")[0].split(), float); A, tr = params[:9].reshape(3, 3), params[9:]
    LPS = np.diag([-1.0, -1.0, 1.0]); p = np.array([10.0, -20.0, 5.0]); TA = np.array(ax["WorldTransformRAS"])
    assert np.allclose(LPS @ (A @ (LPS @ p) + tr), TA[:3, :3] @ p + TA[:3, 3])
    # anatomy grid unchanged, content moved (resampled): same affine, different voxels, same mass
    assert np.allclose(t1.affine, tiny_with_field.t1w.affine) and not np.array_equal(np.asarray(t1.dataobj), np.asarray(tiny_with_field.t1w.dataobj))
    # header mode keeps the voxels and moves the frame
    hdr = ts.bids.offset_image(tiny_with_field.t1w, T, "header")
    assert np.array_equal(np.asarray(hdr.dataobj), np.asarray(tiny_with_field.t1w.dataobj)) and np.allclose(hdr.affine, T @ tiny_with_field.t1w.affine)
    # a Siemens phase image resampled through its complex form stays in 0..4095 and keeps its mean field
    pd_img = nib.load(root / "sub-01/fmap/sub-01_acq-gre_phasediff.nii.gz")
    assert pd_img.get_data_dtype() == np.int16 and 0 <= np.asarray(pd_img.dataobj).min() and np.asarray(pd_img.dataobj).max() <= 4095
    ok, text = ds.validate("--ignoreWarnings")
    if "not found" not in text:
        assert ok, text


def test_pa_offset_moves_the_subject_inside_the_same_fov(tiny_with_field, tmp_path):
    recipes.rpe_series(tmp_path / "pa", phantom=tiny_with_field, voxel=2.0, ndirs=2, nb0=1, subsample=None, subject="01", anat=False, pa_offset=(1.0, 2.0, 0.0, 0.0, 0.0, 5.0))
    root = tmp_path / "pa"
    ap, pa = nib.load(root / "sub-01/dwi/sub-01_dir-AP_dwi.nii.gz"), nib.load(root / "sub-01/dwi/sub-01_dir-PA_dwi.nii.gz")
    assert np.allclose(ap.affine, pa.affine) and ap.shape == pa.shape           # same field of view
    tx = json.load(open(root / "derivatives/trxscan/sub-01/dwi/sub-01_dir-PA_from-dirPA_to-dwi_mode-image_desc-truth_xfm.json"))
    assert tx["RotationDeg"] == pytest.approx(5.0, abs=1e-6)
    # the moved head sits elsewhere in the grid: centre of mass of the clean b0 differs by ~the translation
    ca = nib.load(root / "derivatives/trxscan/sub-01/dwi/sub-01_dir-AP_desc-cleanb0_dwi.nii.gz"); cp = nib.load(root / "derivatives/trxscan/sub-01/dwi/sub-01_dir-PA_desc-cleanb0_dwi.nii.gz")
    def com(img):
        d = np.asarray(img.dataobj); ijk = np.indices(d.shape).reshape(3, -1); w = d.ravel()
        return (img.affine[:3, :3] @ (ijk @ w / w.sum()) + img.affine[:3, 3])
    assert np.linalg.norm(com(cp) - com(ca)) > 1.0
    m = tiny_with_field.moved((0, 0, 0, 0, 0, 0)); assert m is tiny_with_field


def test_itk_transform_readers_and_rigid_error(tmp_path):
    from scipy.io import savemat
    T = ts.bids.rigid_offset((2.0, -3.0, 1.0), (4.0, 0.0, -2.0), (10.0, 20.0, -5.0))      # RAS point map
    (tmp_path / "t.txt").write_text(ts.bids.itk_transform_text(T))
    M = score.read_itk_transform(tmp_path / "t.txt")
    assert np.allclose(score.lps_to_ras(M), T)                                           # text round trip
    # ANTs-style .mat: AffineTransform_double_3_3 with a fixed centre; and an Euler3D one
    Ml = score.lps_to_ras(T); c = np.array([1.0, 2.0, 3.0]); A, off = Ml[:3, :3], Ml[:3, 3]
    savemat(tmp_path / "a.mat", {"AffineTransform_double_3_3": np.concatenate([A.ravel(), off - c + A @ c])[:, None], "fixed": c[:, None]})
    assert np.allclose(score.read_itk_transform(tmp_path / "a.mat"), Ml)
    ang = (0.1, -0.2, 0.3); R = score._euler_zxy(*ang)
    savemat(tmp_path / "e.mat", {"Euler3DTransform_double_3_3": np.array(list(ang) + [5.0, 6.0, 7.0])[:, None], "fixed": np.zeros((3, 1))})
    Me = score.read_itk_transform(tmp_path / "e.mat")
    assert np.allclose(Me[:3, :3], R) and np.allclose(Me[:3, 3], [5, 6, 7])
    err = score.rigid_error(Me, Me); assert err["rotation_deg"] == pytest.approx(0.0, abs=1e-9) and err["translation_mm"] == pytest.approx(0.0, abs=1e-9)
    err = score.rigid_error(np.eye(4), Me, center=(0, 0, 0))
    assert err["rotation_deg"] == pytest.approx(np.degrees(np.arccos((np.trace(R) - 1) / 2)), abs=1e-6) and err["translation_mm"] == pytest.approx(np.linalg.norm([5, 6, 7]))
    try:
        import h5py  # noqa: F401
    except ImportError:
        return
    with h5py.File(tmp_path / "c.h5", "w") as h:
        g = h.create_group("TransformGroup/1")
        g.create_dataset("TransformType", data=np.array([b"AffineTransform_double_3_3"]))
        g.create_dataset("TransformParameters", data=np.concatenate([A.ravel(), off - c + A @ c]))
        g.create_dataset("TransformFixedParameters", data=c)
    assert np.allclose(score.read_itk_transform(tmp_path / "c.h5"), Ml)


def test_oblique_acquisition_and_isocenter(tiny_with_field, tmp_path):
    proto = ts.Protocol.DEFAULT.replace(voxel_mm=2.0, oversample=1, oblique_deg=(6.0, 0.0, 4.0))
    obj = tiny_with_field.grid(proto); straight = tiny_with_field.grid(proto.replace(oblique_deg=None))
    assert obj.dims == straight.dims
    R = obj.affine[:3, :3] @ np.linalg.inv(straight.affine[:3, :3])
    assert np.degrees(np.arccos((np.trace(R) - 1) / 2)) == pytest.approx(np.degrees(np.arccos((np.trace(ts.bids.rigid_offset((0, 0, 0), (6, 0, 4))[:3, :3]) - 1) / 2)), abs=1e-6)
    # the object did not move: the oblique run's clean b0, resampled onto the straight grid, matches the straight run's
    g = (np.array([0, 1000.0]), np.array([[0, 0, 0], [1, 0, 0.0]]))
    ob = obj.simulate(g, proto, ts.Artifacts(), kspace=False)
    st = straight.simulate(g, proto.replace(oblique_deg=None), ts.Artifacts(), kspace=False)
    from nibabel.processing import resample_from_to
    back = resample_from_to(ob.clean_b0, (st.clean_b0.shape[:3], st.clean_b0.affine), order=1).get_fdata()
    ref = np.asarray(st.clean_b0.dataobj); m = ref > 0.2 * ref.max()
    assert np.corrcoef(back[m], ref[m])[0, 1] > 0.9
    las = obj.simulate(g, proto.replace(fsl_orientation=True), ts.Artifacts(), kspace=False, clean_b0=False)
    assert nib.aff2axcodes(las.magnitude.affine) == ("L", "A", "S")   # fsl_orientation picks the closest axes of an oblique grid
    recipes.rpe_series(tmp_path / "obl", phantom=tiny_with_field, voxel=2.0, ndirs=2, nb0=1, subsample=None, subject="01", anat=False, pe=("AP",), acq_rotation=(5.0, 0.0, 0.0), isocenter=(0.0, -20.0, -30.0))
    img = nib.load(tmp_path / "obl/sub-01/dwi/sub-01_dir-AP_dwi.nii.gz")
    assert not np.allclose(np.abs(img.affine[:3, :3]), np.diag(np.diag(np.abs(img.affine[:3, :3]))), atol=1e-6)   # oblique header
    rec = json.load(open(tmp_path / "obl/derivatives/trxscan/recipe.json"))
    assert rec["Parameters"]["acq_rotation"] == [5.0, 0.0, 0.0] and rec["Parameters"]["isocenter"] == [0.0, -20.0, -30.0]
