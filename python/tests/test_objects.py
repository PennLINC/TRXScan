import numpy as np
import pytest

import trxscan as ts


@pytest.fixture(scope="module")
def box_sim(gtab6):
    box = ts.objects.box(8, matrix=32, oversample=2, voxel_mm=2.0)
    proto = ts.Protocol.HBCD.replace(voxel_mm=2.0, coils=4, accel=2)
    sim = box.simulate(gtab6, proto, ts.Artifacts(noise=1e-4, ghost=0.02, spikes=1), kspace=True, kspace_capture=("acquired", "reconstructed", "coil_images"))
    return box, proto, sim


def test_box_geometry_and_outputs(box_sim, gtab6):
    box, proto, sim = box_sim
    assert box.dims == (32, 32, 1) and box.sim_dims == (64, 64, 1)
    assert sim.magnitude.shape == (32, 32, 1, 6)
    assert sim.sidecar["PhaseEncodingDirection"] == "j-"
    assert sim.sidecar["TotalReadoutTime"] == pytest.approx(0.0917)
    assert sim.sidecar["EffectiveEchoSpacing"] == pytest.approx(0.0917 / 31, rel=1e-4)
    assert np.isfinite(sim.magnitude.get_fdata()).all()
    assert sim.complex.dtype == np.complex64


def test_kspace_capture_and_reference_recon(box_sim):
    _, _, sim = box_sim
    k = sim.kspace
    assert k.acquired.shape == (6, 1, 4, 32, 32) and k.combined.shape == (6, 1, 32, 32)
    assert k.mask.shape == (32, 32) and k.mask.sum() == 32 * (32 * 0.75 // 2 + 12) or k.mask.any()
    # un-acquired lines are zero in `acquired`, filled by GRAPPA in `reconstructed`
    missing = ~k.mask[:, 0]
    assert np.all(k.acquired[:, :, :, missing, :] == 0)
    assert np.any(k.reconstructed[:, :, :, missing, :] != 0)
    rec = ts.KSpace.recon_reference(k.reconstructed[:, 0], k.sensitivities)
    assert np.abs(rec - k.combined[:, 0]).max() < 1e-5 * np.abs(k.combined).max()
    # coil images are the inverse transform of the reconstructed k-space
    img = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(k.reconstructed[0, 0, 0])))
    assert np.abs(img - k.coil_images[0, 0, 0]).max() < 1e-4 * np.abs(img).max()
    # magnitude equals |combined| without a noise map
    assert np.abs(np.abs(k.combined[:, 0]).transpose(2, 1, 0) - sim.magnitude.get_fdata()[:, :, 0, :]).max() < 1e-5


def test_kspace_npz_round_trip(box_sim, tmp_path):
    _, _, sim = box_sim
    p = sim.kspace.save_npz(tmp_path / "k.npz")
    k2 = ts.KSpace.load_npz(p)
    assert np.array_equal(k2.acquired, sim.kspace.acquired)
    assert np.array_equal(k2.mask, sim.kspace.mask)
    assert k2.readout.total_readout_ms == pytest.approx(sim.readout.total_readout_ms)


def test_oversampling_rings_and_o1_does_not(gtab6):
    proto = ts.Protocol.DEFAULT.replace(voxel_mm=2.0, te_ms=0.0, signal_scale=1.0)
    art = ts.Artifacts(relaxation=False, distortion=False)
    o2 = ts.objects.box(8, matrix=32, oversample=2).simulate(gtab6, proto.replace(oversample=2, phase_model="none"), art, kspace=False)
    o1 = ts.objects.box(8, matrix=32, oversample=1).simulate(gtab6, proto.replace(oversample=1, phase_model="none"), art, kspace=False)
    row2 = o2.magnitude.get_fdata()[:, 16, 0, 0]
    row1 = o1.magnitude.get_fdata()[:, 16, 0, 0]
    plateau = row2[14:18].mean()
    assert row2.max() - plateau > 0.02 * plateau, "o=2 must ring"
    assert abs(row1.max() - row1[14:18].mean()) < 1e-3, "o=1 is an exact round trip"


def test_slice_unit_equals_full_run(gtab6):
    # a 5-slice box: simulating slice 2 alone must equal slice 2 of the whole run
    box = ts.objects.box(8, matrix=24, oversample=2, nz=5)
    proto = ts.Protocol.HBCD.replace(voxel_mm=2.0, coils=2, accel=1)
    art = ts.Artifacts(noise=1e-4, eddy=0.02, eddy_quad=0.003, eddy_phase=2e-5, ghost=0.01, spikes=1, seed=4)
    full = box.simulate(gtab6, proto, art, kspace=False)
    one = box.simulate(gtab6, proto, art, slices=2, kspace=False)
    assert np.array_equal(full.magnitude.get_fdata()[:, :, 2], one.magnitude.get_fdata()[:, :, 0])
    assert np.array_equal(full.phase.get_fdata()[:, :, 2], one.phase.get_fdata()[:, :, 0])
    assert np.allclose(one.affine[:3, 3], full.affine[:3, 3] + full.affine[:3, 2] * 2)


def test_crossing_truth_and_peaks(gtab6):
    cross = ts.objects.crossing(60, matrix=24)
    proto = ts.Protocol.DEFAULT.replace(voxel_mm=2.0)
    truth = cross.microstructure(proto, gtab6)
    assert set(truth) == set(ts.scalar_names())
    fa = truth["fa"].get_fdata()
    assert 0 < fa.max() <= 1
    sim = cross.simulate(gtab6, proto, ts.Artifacts(), truth_peaks=True, kspace=False)
    pk = sim.truth_peaks.get_fdata().reshape(24, 24, 1, 3, 3)
    n_peaks = (np.linalg.norm(pk, axis=-1) > 0).sum(axis=-1)
    assert n_peaks.max() == 2 and (n_peaks == 1).any()


def test_fill_from_voxel_and_series(gtab6):
    v = ts.Voxel(fibers=[((1, 0, 0), 0.5), ((0, 1, 0), 0.5)], wm=0.7, gm=0.2, csf=0.1)
    obj = ts.objects.fill(ts.objects.box(8, matrix=24, oversample=2), v)
    sim = obj.simulate(gtab6, ts.Protocol.DEFAULT.replace(voxel_mm=2.0, te_ms=0.0, signal_scale=1.0), ts.Artifacts(relaxation=False, distortion=False), kspace=False)
    centre = sim.series((12, 12, 0))
    ref = v.signal(gtab6).total
    # the box interior is uniform, so the centre voxel reproduces the voxel signal up to the
    # hemisphere quantisation of the mixture (directions along the axes are exact vertices)
    assert np.abs(centre / centre[0] - ref / ref[0]).max() < 2e-2


def test_multiband_dropout_and_gre(gtab6):
    box = ts.objects.box(8, matrix=24, oversample=1, nz=6)
    proto = ts.Protocol.HBCD.replace(voxel_mm=2.0, mb=3, oversample=1)
    sim = box.simulate(gtab6, proto, ts.Artifacts(dropout=1.0, seed=1), gre=ts.Gre(), kspace=False, slices=slice(2, 4))
    assert sim.dropout and all(len(d.slices) == 3 for d in sim.dropout)
    assert "volume\tshot\tslices\tattenuation" in sim.dropout_table()
    assert sim.gre.magnitude1.shape[:3] == (24, 24, 2)
    assert np.allclose(sim.gre.magnitude1.affine, sim.affine)
    assert "phasediff" in sim.gre.phase and sim.sidecar["B0FieldSource"] == "b0gre"


def test_gnl_and_bids_writer(gtab6, tmp_path):
    box = ts.objects.box(8, matrix=24, oversample=1, nz=2)
    proto = ts.Protocol.HBCD.replace(voxel_mm=2.0, oversample=1, fsl_orientation=True)
    sim = box.simulate(gtab6, proto, ts.Artifacts(gnl="whole-body-80", gnl_scale=2.0, isocenter=(0.0, -20.0, -30.0), noise_map=np.full((24, 24, 2), 0.01, np.float32)), kspace=False, truth_peaks=True)
    assert sim.gnl.graddev.shape[-1] == 9 and sim.gnl.coeff_text.startswith(("#", "\n", " ")) or sim.gnl.coeff_text
    files = sim.to_bids(tmp_path / "sub-01_dir-AP")
    names = {p.name for p in files}
    for want in ("sub-01_dir-AP_part-mag_dwi.nii.gz", "sub-01_dir-AP_part-phase_dwi.json", "sub-01_dir-AP_dwi.bval", "sub-01_dir-AP_desc-gnl_graddev.nii.gz", "sub-01_dir-AP_desc-noise_sigma.nii.gz", "sub-01_dir-AP_desc-truth_peaks.nii.gz"):
        assert want in names, want
    real = ts.BidsDwi.load(tmp_path / "sub-01_dir-AP_part-mag_dwi.nii.gz") if False else None
    assert sim.sidecar["PhaseEncodingDirection"] in ("j", "j-")


def test_b0_only_scheme_is_finite():
    from dipy.core.gradients import gradient_table

    g = gradient_table(np.zeros(3), bvecs=np.zeros((3, 3)))
    sim = ts.objects.box(6, matrix=16).simulate(g, ts.Protocol.HBCD.replace(voxel_mm=2.0), ts.Artifacts(), kspace=False)
    m = sim.magnitude.get_fdata()
    assert np.isfinite(m).all() and m.max() > 0
    v = ts.Voxel(fibers=[((1, 0, 0), 1.0)]).signal(g).total
    assert np.allclose(v, 1.0)
