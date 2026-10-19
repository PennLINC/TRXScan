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


# ---- P9: per-shot events ----


def test_shot_events_validation():
    with pytest.raises(ValueError):
        ts.ShotEvents(slice_order="random")
    with pytest.raises(ValueError):
        ts.ShotEvents(rate=0.2, events=[])
    with pytest.raises(ValueError):
        ts.Artifacts(dropout=0.2, shot_events=ts.ShotEvents(rate=0.1))
    assert ts.Artifacts(shot_events=ts.ShotEvents(rate=0.1)).needs_context


def test_slice_orders():
    from trxscan import _core

    sched = lambda order, mb=1: _core.slice_schedule_from_times(_core.shot_offsets(6, mb, order, 3.0))
    assert sched("sequential") == [[0], [1], [2], [3], [4], [5]]
    assert sched("descending") == [[5], [4], [3], [2], [1], [0]]
    assert sched("interleaved") == [[0], [2], [4], [1], [3], [5]]
    assert sched("odd-first") == [[1], [3], [5], [0], [2], [4]]
    assert sched("interleaved", 2) == [[0, 3], [2, 5], [1, 4]]


def _shot_run(gtab6, events, *, nz=6, order="sequential", **kw):
    box = ts.objects.box(8, matrix=16, nz=nz)
    art = ts.Artifacts(shot_events=ts.ShotEvents(slice_order=order, events=events))
    return box.simulate(gtab6, ts.Protocol.DEFAULT, art, kspace=False, clean_b0=False, **kw)


def test_shot_events_drop_one_single_band_slice_on_a_synthetic_object(gtab6, tmp_path):
    # a synthetic fibre object: the re-simulation path would refuse it, ShotEvents does not select it
    ev = [{"volume": 1, "shot": 2, "severity": 0.5, "jump_mm": (0.0, 0.0, 0.0), "jump_deg": (0.0, 0.0, 0.0)}]
    sim = _shot_run(gtab6, ev)
    clean = ts.objects.box(8, matrix=16, nz=6).simulate(gtab6, ts.Protocol.DEFAULT, ts.Artifacts(), kspace=False, clean_b0=False)
    assert [(d.volume, d.shot, d.slices) for d in sim.dropout] == [(1, 2, (2,))]
    # volume 1 (b = 1000 of b_max 2000), slice 2: 1 - 0.5 * 1000 / 2000; elsewhere unchanged
    img_a = np.asarray(clean.magnitude.dataobj, dtype=float)
    img_b = np.asarray(sim.magnitude.dataobj, dtype=float)
    m = img_a[..., 2, 1] > 0.1 * img_a.max()
    assert np.allclose(img_b[..., 2, 1][m] / img_a[..., 2, 1][m], 0.75, rtol=1e-4)
    keep = np.ones(img_a.shape, bool)
    keep[..., 2, 1] = False
    assert np.allclose(img_b[keep], img_a[keep], rtol=1e-5, atol=1e-6 * img_a.max())
    # the events table replays the run
    path = tmp_path / "ev.tsv"
    path.write_text(sim.motion_events)
    again = _shot_run(gtab6, str(path))
    assert np.array_equal(np.asarray(again.magnitude.dataobj), img_b)


def test_shot_events_on_slabs_use_the_full_stack(gtab6):
    ev = [{"volume": 2, "shot": 5, "severity": 0.0, "jump_mm": (0.6, 0.0, 0.0), "jump_deg": (0.0, 0.0, 3.0)}]
    full = _shot_run(gtab6, ev, nz=12)
    chunked = _shot_run(gtab6, ev, nz=12, chunk=4)
    a = np.asarray(full.magnitude.dataobj, dtype=float)
    b = np.asarray(chunked.magnitude.dataobj, dtype=float)
    assert np.allclose(a, b, rtol=1e-5, atol=1e-6 * a.max())
    assert not np.allclose(a[..., 5, 2], _shot_run(gtab6, None, nz=12).magnitude.dataobj[..., 5, 2]), "the jump moves slice 5"


def test_spin_history_refusals(gtab6):
    box = ts.objects.box(8, matrix=16, nz=6)
    ev = [{"volume": 1, "shot": 2, "severity": 0.0, "jump_mm": (0.0, 0.0, 0.5), "jump_deg": (0.0, 0.0, 0.0)}]
    art = ts.Artifacts(shot_events=ts.ShotEvents(slice_order="sequential", events=ev, spin_history=True))
    with pytest.raises(ValueError, match="t1_s"):
        box.simulate(gtab6, ts.Protocol.DEFAULT, art, kspace=False, clean_b0=False)
    proto = ts.Protocol.DEFAULT.replace(t1_s=(0.8, 1.3, 4.0), tr_s=4.0)
    with pytest.raises(ValueError, match="whole slice stack"):
        box.simulate(gtab6, proto, art, kspace=False, clean_b0=False, chunk=2)
    still = ts.Artifacts(shot_events=ts.ShotEvents(slice_order="sequential", events=[{**ev[0], "jump_mm": (0.0, 0.0, 0.0)}], spin_history=True))
    with pytest.raises(ValueError, match="history of"):
        box.simulate(gtab6, proto, still, kspace=False, clean_b0=False)


def test_spin_history_is_signal_weighted_and_independent_of_b(gtab6):
    from dipy.core.gradients import gradient_table

    box = ts.objects.box(8, matrix=16, nz=8)
    proto = ts.Protocol.DEFAULT.replace(t1_s=(0.8, 1.3, 4.0), tr_s=4.0)
    ev = [{"volume": 1, "shot": 3, "severity": 0.0, "jump_mm": (0.0, 0.0, 0.5), "jump_deg": (0.0, 0.0, 0.0)}]
    art = ts.Artifacts(shot_events=ts.ShotEvents(slice_order="sequential", events=ev, spin_history=True))
    half = gradient_table(gtab6.bvals * 0.5, bvecs=gtab6.bvecs)

    def rel(gtab):
        sim = box.simulate(gtab, proto, art, kspace=False, clean_b0=False)
        clean = box.simulate(gtab, proto, ts.Artifacts(), kspace=False, clean_b0=False)
        sh = np.asarray(sim.spin_history.dataobj, dtype=float)
        mag = np.asarray(clean.magnitude.dataobj, dtype=float)
        m = mag[..., 3, 1] > 0.2 * mag[..., 3, 1].max()
        assert np.abs(sh[..., :3, 1]).max() < 1e-6 * np.abs(sh).max(), "nothing before the event shot"
        return np.median(sh[..., 3, 1][m] / mag[..., 3, 1][m])

    a, b = rel(gtab6), rel(half)
    assert a < 0, "the event slice is darker"
    assert abs(a - b) <= 1e-4 * abs(a), f"{a} vs {b}"


def test_shot_events_refuse_duplicates_and_state_their_timing(gtab6):
    e = {"volume": 1, "shot": 2, "severity": 0.5, "jump_mm": (0.0, 0.0, 0.0), "jump_deg": (0.0, 0.0, 0.0)}
    with pytest.raises(ValueError, match="second event"):
        _shot_run(gtab6, [e, e])
    box = ts.objects.box(8, matrix=16, nz=6)
    proto = ts.Protocol.DEFAULT.replace(tr_s=3.0)
    sim = box.simulate(gtab6, proto, ts.Artifacts(shot_events=ts.ShotEvents(slice_order="sequential", events=[e])),
                       kspace=False, clean_b0=False)
    assert sim.sidecar["SliceTiming"] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
