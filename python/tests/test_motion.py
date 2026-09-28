import numpy as np
import pytest

import trxscan as ts


def test_from_confounds_converts_radians(tmp_path):
    tsv = tmp_path / "c.tsv"
    tsv.write_text("framewise_displacement\ttrans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\nn/a\t1\t2\t3\t0.1\t0\t-0.2\n0.5\t0\t0\t0\t0\t0\t0\n")
    m = ts.Motion.from_confounds(tsv)
    p = m.poses_for(2)
    assert p.shape == (2, 6)
    assert p[0, :3].tolist() == [1, 2, 3]
    assert p[0, 3] == pytest.approx(np.degrees(0.1)) and p[0, 5] == pytest.approx(np.degrees(-0.2))
    with pytest.raises(ValueError):
        m.poses_for(3)
    assert m.resample(5).poses_for(5).shape == (5, 6)


def test_affine_round_trip():
    m = ts.Motion.from_arrays(np.array([[1.0, -2.0, 0.5]]), np.array([[10.0, -20.0, 35.0]]))
    c = (3.0, -4.0, 5.0)
    mats = m.matrices(1, center=c)
    back = ts.Motion.from_affines(mats, center=c)
    assert np.allclose(back.poses_for(1), m.poses_for(1), atol=1e-9)


def test_generators_are_deterministic():
    a = ts.Motion.random((2, 2, 1), (3, 3, 3), seed=7).poses_for(10)
    b = ts.Motion.random((2, 2, 1), (3, 3, 3), seed=7).poses_for(10)
    assert np.array_equal(a, b) and np.abs(a).max() > 0
    lin = ts.Motion.linear((3, 3, 2), (3, 2, 2)).poses_for(4)
    assert np.allclose(lin[-1], [3, 3, 2, 3, 2, 2])
    fd = ts.Motion.linear((3, 3, 2), (3, 2, 2)).framewise_displacement(4)
    assert fd[0] == 0 and (fd[1:] > 0).all()


def test_motion_path_on_a_synthetic_object_requires_streamlines(gtab6):
    box = ts.objects.box(8, matrix=16)
    with pytest.raises(ValueError):
        box.simulate(gtab6, ts.Protocol.DEFAULT, ts.Artifacts(motion=ts.Motion.random((1, 1, 1), (1, 1, 1))), kspace=False)
