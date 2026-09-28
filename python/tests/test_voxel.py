import numpy as np
import pytest

import trxscan as ts


def test_single_stick_matches_dipy_single_tensor(gtab6):
    from dipy.sims.voxel import single_tensor

    d = ts.Tissue.ADULT.d_intra
    v = ts.Voxel(fibers=[((1, 0, 0), 1.0)], tissue=ts.Tissue.ADULT.replace(intra_frac=1.0, extra_frac=0.0))
    ref = single_tensor(gtab6, S0=1.0, evals=(d, 0, 0), evecs=np.eye(3))
    assert np.abs(v.signal(gtab6).total - ref).max() < 1e-8


def test_arbitrary_direction_is_exact_not_quantised(gtab6):
    from dipy.sims.voxel import single_tensor

    d = ts.Tissue.ADULT.d_intra
    direction = np.array([0.3, -0.5, 0.812])
    direction /= np.linalg.norm(direction)
    evecs = np.linalg.qr(np.column_stack([direction, np.eye(3)[:, :2]]))[0]
    evecs[:, 0] *= np.sign(evecs[:, 0] @ direction)
    v = ts.Voxel(fibers=[(direction, 1.0)], tissue=ts.Tissue.ADULT.replace(intra_frac=1.0, extra_frac=0.0))
    ref = single_tensor(gtab6, S0=1.0, evals=(d, 0, 0), evecs=evecs)
    assert np.abs(v.signal(gtab6).total - ref).max() < 1e-8


def test_rotated_fibers_equal_rotated_bvecs(gtab6):
    from dipy.core.gradients import gradient_table

    v = ts.Voxel(fibers=[((1, 0, 0), 0.6), ((0, 1, 0), 0.4)], wm=0.8, gm=0.15, csf=0.05)
    m = ts.Motion.from_arrays(np.zeros((6, 3)), np.tile([[10.0, -20.0, 35.0]], (6, 1)))
    R = m.matrices(6)[0, :3, :3]
    rotated = gradient_table(gtab6.bvals, bvecs=(gtab6.bvecs @ R))
    a = v.signal(gtab6, motion=m).total
    b = v.signal(rotated).total
    assert np.abs(a - b).max() < 1e-10


def test_truth_matches_dipy_for_one_tensor():
    from dipy.reconst.dti import fractional_anisotropy, mean_diffusivity

    t = ts.Tissue.ADULT
    v = ts.Voxel(fibers=[((1, 0, 0), 1.0)], tissue=t.replace(intra_frac=0.0, extra_frac=1.0))
    tr = v.truth()
    evals = np.array(t.d_extra)
    assert tr.fa == pytest.approx(float(fractional_anisotropy(evals)), rel=1e-6)
    assert tr.md == pytest.approx(float(mean_diffusivity(evals)), rel=1e-6)
    assert tr.ad == pytest.approx(evals[0], rel=1e-6)
    assert tr.icvf == 0.0 and tr.isovf == 0.0
    assert set(tr) == set(ts.scalar_names())


def test_te_weighting_and_fractions(gtab6):
    v = ts.Voxel(fibers=[((1, 0, 0), 1.0)], wm=0.5, gm=0.3, csf=0.2)
    s0 = v.signal(gtab6)
    s88 = v.signal(gtab6, te_ms=88.0)
    t2 = ts.Tissue.ADULT.t2_ms
    assert np.allclose(s88.fiber, s0.fiber * np.exp(-88.0 / t2[0]))
    assert np.allclose(s88.csf, s0.csf * np.exp(-88.0 / t2[2]))
    assert s0.total[0] == pytest.approx(1.0)


def test_noise_is_rician_at_one_coil():
    sig = np.full(400, 0.05)
    n = ts.Voxel.noise(sig, 0.05, coils=1, seed=3)
    assert n.shape == (400,) and (n > 0).all()
    assert n.mean() > 0.05  # Rician bias at SNR 1
    n8 = ts.Voxel.noise(sig, 0.05, coils=8, accel=2, seed=3)
    assert n8.std() > 0 and not np.allclose(n8, n)
