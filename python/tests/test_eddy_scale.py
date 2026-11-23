"""The eddy-current drive follows the gradient's strength, ``sqrt(bval)``, unless
``Artifacts.eddy_scale`` gives a scale per volume (``bvals`` is TRXScan 0.3.0's drive)."""

import numpy as np
import pytest

import trxscan as ts

EDDY = dict(eddy=0.5, eddy_quad=0.05, eddy_phase=1e-3)


@pytest.fixture(scope="module")
def run(gtab6):
    box = ts.objects.box(8, matrix=24, oversample=2, voxel_mm=2.0)
    proto = ts.Protocol.HBCD.replace(voxel_mm=2.0)

    def simulate(**kw):
        sim = box.simulate(gtab6, proto, ts.Artifacts(**kw), kspace=False, clean_b0=False)
        return sim.magnitude.get_fdata(), sim.phase.get_fdata()

    return simulate


def test_the_default_scale_is_the_square_root_of_the_b_value(run, gtab6):
    default = run(**EDDY)
    explicit = run(**EDDY, eddy_scale=np.sqrt(gtab6.bvals))
    assert np.array_equal(default[0], explicit[0])
    assert np.array_equal(default[1], explicit[1])


def test_the_b_value_as_the_scale_is_another_drive(run, gtab6):
    # the drive of TRXScan 0.3.0 and earlier, larger by sqrt(bval)
    default = run(**EDDY)
    linear = run(**EDDY, eddy_scale=gtab6.bvals)
    assert not np.allclose(default[0][..., 1:], linear[0][..., 1:])


def test_the_scale_multiplies_the_drive(run, gtab6):
    # twice the scale is twice the strength, for the linear term alone
    scale = np.sqrt(gtab6.bvals)
    twice = run(eddy=0.5, eddy_scale=2 * scale)
    stronger = run(eddy=1.0, eddy_scale=scale)
    assert np.allclose(twice[0], stronger[0], rtol=1e-5, atol=1e-6 * stronger[0].max())


def test_b0_keeps_no_drive_whatever_its_scale(run, gtab6):
    scale = np.sqrt(gtab6.bvals)
    scale[0] = 100.0
    assert np.array_equal(run(**EDDY)[0][..., 0], run(**EDDY, eddy_scale=scale)[0][..., 0])


def test_a_scale_of_the_wrong_length_is_refused(run):
    with pytest.raises(ValueError, match="eddy_scale has 2 entries"):
        run(**EDDY, eddy_scale=[1.0, 2.0])


def test_artifacts_compare_their_scales():
    assert ts.Artifacts(eddy_scale=[1.0, 2.0]) == ts.Artifacts(eddy_scale=np.array([1.0, 2.0]))
    assert ts.Artifacts(eddy_scale=[1.0, 2.0]) != ts.Artifacts(eddy_scale=[1.0, 3.0])
    assert ts.Artifacts(eddy_scale=[1.0, 2.0]) != ts.Artifacts()
    assert not ts.Artifacts(eddy_scale=[1.0]).clean
