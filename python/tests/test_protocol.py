import json

import numpy as np
import pytest

import trxscan as ts
from trxscan import _core


def test_tissue_presets_come_from_rust():
    for name in ("neonatal", "adult", "infant"):
        d = _core.tissue_preset(name)
        t = ts.Tissue.from_preset(name)
        assert t.d_intra == d["d_intra"] and t.t2_fiber == d["t2_fiber"]
    assert ts.Tissue.ADULT.replace(t2_fiber=75.0).t2_fiber == 75.0
    assert ts.Tissue.ADULT.to_params(3000.0)["b_value"] == 3000.0
    assert ts.Tissue.ADULT.to_params(1000.0, diff_scale=(2.0, 1.0, 1.0))["d_intra"] == 2 * ts.Tissue.ADULT.d_intra


def test_hbcd_preset_matches_the_cli_acquisition():
    for ny in (32, 96, 128):
        want = _core.acquisition_hbcd(ny)
        got = _core.acquisition_defaults(ts.Protocol.HBCD.acquisition(ny))
        for k in ("t_line", "t_echo", "t_inhom", "partial_fourier", "pf_mode", "acs_lines", "signal_scale"):
            assert got[k] == pytest.approx(want[k]), k
    # DEFAULT + Artifacts() is exactly kspace::default_acquisition()
    d = _core.acquisition_defaults({**ts.Protocol.DEFAULT.acquisition(64), **ts.Artifacts().acquisition()})
    assert d == _core.acquisition_defaults({})


def test_pe_and_replace():
    assert not ts.Protocol.DEFAULT.reverse_phase
    assert ts.Protocol.DEFAULT.replace(pe="PA").reverse_phase
    assert ts.Protocol.DEFAULT.replace(pe="j").reverse_phase
    with pytest.raises(ValueError):
        ts.Protocol(pe="k")
    with pytest.raises(ValueError):
        ts.Protocol(readout_ms=90, echo_spacing_ms=0.9)
    p = ts.Protocol.HBCD.replace(voxel_mm=2.5)
    assert p.voxel == (2.5, 2.5, 2.5) and p.name == "hbcd*"


def test_readout_is_a_permutation_and_pf_shortens_time_to_center():
    r = ts.Protocol.HBCD.readout(24, 32)
    assert sorted(r.ky_order.tolist()) == list(range(32))
    assert r.ticks.shape == (24 * 32, 3)
    assert np.all(np.diff(r.ticks[:, 2]) > 0)
    full = ts.Protocol.HBCD.replace(partial_fourier=1.0).readout(24, 32)
    # scanner-style partial Fourier skips the START of the train: the centre is reached a
    # quarter of the train sooner and the readout is shorter; the legacy rule keeps the timing
    assert r.time_to_center_ms == pytest.approx(full.time_to_center_ms - 8 * r.t_line_ms, abs=r.dt_ms + 1e-9)
    assert r.total_readout_ms < full.total_readout_ms
    assert r.acquired_lines.size == round(32 * 0.75)
    legacy = ts.Protocol.HBCD.replace(pf_mode="contiguous").readout(24, 32)
    assert legacy.time_to_center_ms == pytest.approx(full.time_to_center_ms)
    assert set(legacy.acquired_lines.tolist()) != set(r.acquired_lines.tolist())
    # 31 line times, give or take one sample: odd lines are read backwards, so their
    # centre sample sits one dt later than on even lines
    assert abs(full.total_readout_ms - 91.7 * 31 / 32) <= full.dt_ms + 1e-9


def test_sidecar_and_from_bids_round_trip(tmp_path):
    p = ts.Protocol.HBCD.replace(voxel_mm=2.0, coils=8, accel=2, mb=3, pe="PA")
    side = {"PhaseEncodingDirection": "j", **p.sidecar(96)}
    (tmp_path / "dwi.json").write_text(json.dumps(side))
    import nibabel as nib

    nib.Nifti1Image(np.zeros((10, 96, 4), np.float32), np.diag([2.0, 2.0, 2.0, 1.0])).to_filename(str(tmp_path / "dwi.nii.gz"))
    q = ts.Protocol.from_bids(tmp_path / "dwi.json")
    assert q.te_ms == pytest.approx(p.te_ms)
    assert q.readout_ms == pytest.approx(p.readout_ms)
    assert q.partial_fourier == p.partial_fourier and q.accel == 2 and q.mb == 3 and q.pe == "j"
    assert q.voxel == (2.0, 2.0, 2.0)
    with pytest.warns(UserWarning):
        (tmp_path / "bare.json").write_text("{}")
        ts.Protocol.from_bids(tmp_path / "bare.json")


def test_applied_flip_angles_weight_the_signal():
    import math

    nominal = ts.Protocol.HBCD.replace(applied_flip_deg=(90.0, 180.0))
    assert nominal.rf_weights == (1.0, 1.0, 1.0) and nominal.s0 == (1.0, 1.0, 1.0)
    assert ts.Protocol.HBCD.rf_weights == (1.0, 1.0, 1.0)
    # fully relaxed: the pathway share sin(a) sin^2(b/2), the same for every compartment
    p = ts.Protocol.HBCD.replace(applied_flip_deg=(63.0, 126.0), tissue_s0=(1.0, 0.8, 0.5))
    want = math.sin(math.radians(63.0)) * math.sin(math.radians(63.0)) ** 2
    assert p.rf_weights == pytest.approx((want,) * 3)
    assert p.s0 == pytest.approx((want, 0.8 * want, 0.5 * want))
    # at TR 4.2 s the T1 loop saturates CSF (T1 4 s) and the clipped pair saturates it a little more
    q = p.replace(tr_s=4.2)
    w, rel = q.rf_weights, q.rf_relative_weights
    assert w[2] < w[0] < want and rel[2] < rel[0] < 1.0
    assert q.replace(applied_flip_deg=(90.0, 180.0)).rf_relative_weights == pytest.approx((1.0, 1.0, 1.0))
    assert _core.spin_echo_weights(63.0, 126.0, 4.2, 0.088, (830.0, 1330.0, 4000.0)) == pytest.approx(w)
    side = q.sidecar(96, nz=6)
    assert side["AppliedFlipAngle"] == 63.0 and side["AppliedRefocusingFlipAngle"] == 126.0 and side["RepetitionTime"] == 4.2
    assert "AppliedFlipAngle" not in ts.Protocol.HBCD.sidecar(96)
    with pytest.raises(ValueError):
        ts.Protocol(applied_flip_deg=(200.0, 180.0))
    with pytest.raises(ValueError):
        ts.Protocol(applied_flip_deg=(90.0, 180.0), tr_s=0.05, te_ms=88.0)
    with pytest.raises(ValueError):
        ts.Protocol(t1_ms=(1.0, 2.0))
