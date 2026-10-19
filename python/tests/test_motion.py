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


# ---- P10: timed traces ----


def test_timed_traces_are_read_strictly_and_untimed_ones_as_before(tmp_path):
    from trxscan import _core

    h = "trans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n"
    # untimed: unchanged, n/a and words as zeros, truncation, short-trace refusal
    u = tmp_path / "u.tsv"
    u.write_text(h + "1\tn/a\tabc\t0\t0\t0\n0\t0\t0\t0\t0\t0\n0\t0\t1\t0\t0\t0\n")
    m = ts.Motion.from_confounds(u)
    assert not m.is_timed and m.n_vol == 3
    assert m.poses_for(2)[0, :3].tolist() == [1, 0, 0]
    with pytest.raises(ValueError, match="volumes"):
        m.poses_for(4)
    (tmp_path / "u.json").write_text('{"trans_x": {"Units": "mm"}}')
    assert not ts.Motion.from_confounds(u).is_timed, "a confounds JSON leaves it untimed"
    # a time column: timed, read by the same reader as the CLI's
    t = tmp_path / "t.tsv"
    t.write_text("time\t" + h + "0\t0\t0\t0\t0\t0\t0\n10\t0\t0\t2\t0\t0\t0.1\n")
    m = ts.Motion.from_confounds(t)
    times, flat = _core.load_motion_trace(str(t))
    assert m.is_timed and m.n_vol is None
    assert np.array_equal(m.times, times) and np.array_equal(m.poses, np.asarray(flat).reshape(-1, 6))
    with pytest.raises(ValueError, match="tr_s"):
        m.poses_for(3)
    p = m.poses_for(3, tr_s=4.0)
    assert np.allclose(p[:, 2], [0.0, 0.8, 1.6]) and p[2, 5] == pytest.approx(np.degrees(0.1) * 0.8)
    with pytest.raises(ValueError, match="never extrapolated"):
        m.poses_for(4, tr_s=4.0)
    with pytest.raises(ValueError, match="resampling"):
        m.resample(5)
    with pytest.raises(ValueError):
        (tmp_path / "bad.tsv").write_text("time\t" + h + "0\t0\t0\t0\t0\t0\t0\n1\tabc\t0\t0\t0\t0\t0\n")
        ts.Motion.from_confounds(tmp_path / "bad.tsv")
    # a sidecar SamplingFrequency
    s = tmp_path / "s.tsv"
    s.write_text(h + "0\t0\t0\t0\t0\t0\n0\t0\t1\t0\t0\t0\n0\t0\t2\t0\t0\t0\n")
    (tmp_path / "s.json").write_text('{"SamplingFrequency": 0.5, "StartTime": -1}')
    m = ts.Motion.from_confounds(s)
    assert m.times.tolist() == [-1.0, 1.0, 3.0]
    # the utilities: per volume with tr_s; a timed trace's displacement needs it
    assert m.matrices(2, tr_s=1.0).shape == (2, 4, 4)
    assert m.framewise_displacement(2, tr_s=1.0)[1] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        m.framewise_displacement()
    # Motion.timed from arrays
    mt = ts.Motion.timed([0.0, 1.0], [[0, 0, 0], [0, 0, 1]], [[0, 0, 0], [0, 0, 0]], sampling="shot")
    assert mt.sampling == "shot" and mt.poses_for(2, tr_s=0.5)[1, 2] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        ts.Motion.timed([0.0, 0.0], [[0, 0, 0], [0, 0, 1]], [[0, 0, 0], [0, 0, 0]])


def test_shot_pose_bindings_check_their_shapes(gtab6):
    box = ts.objects.box(8, matrix=16, nz=6)
    from trxscan import _core

    offsets = _core.shot_offsets(6, 1, "sequential", 3.0)
    vp = np.zeros((int(gtab6.bvals.size), 6))
    ok = np.zeros((int(gtab6.bvals.size) * 6, 6))
    tsv = _core.motion_shots_tsv(np.asarray(box.sim_affine), tuple(box.sim_dims), offsets, 3.0, int(gtab6.bvals.size), [],
                                 0, vp, ok)
    assert tsv.splitlines()[0].startswith("volume\tshot\tslices\ttime") and len(tsv.splitlines()) == 1 + vp.shape[0] * 6
    with pytest.raises(ValueError, match="volume_poses"):
        _core.motion_shots_tsv(np.asarray(box.sim_affine), tuple(box.sim_dims), offsets, 3.0, int(gtab6.bvals.size), [], 0, None, ok)
    with pytest.raises(ValueError, match="shot_poses"):
        _core.motion_shots_tsv(np.asarray(box.sim_affine), tuple(box.sim_dims), offsets, 3.0, int(gtab6.bvals.size), [], 0, vp, ok[:-1])


def test_shot_motion_bindings_validate_volume_poses(gtab6):
    # implementation review: the bindings that apply the motion check volume_poses too, with a
    # ValueError rather than an index panic
    from trxscan import _core
    from trxscan._arrays import as_gtab, b_max_of

    box = ts.objects.box(8, matrix=16, nz=6)
    bvals, bvecs, _, _ = as_gtab(gtab6)
    b_max = b_max_of(bvals)
    params = ts.Protocol.DEFAULT.tissue.to_params(b_max, ts.Protocol.DEFAULT.diff_scale)
    mix = _core.mixture_from_fibers(box.sim_dims, box.fibers.voxel, box.fibers.dirs, box.fibers.weight,
                                    box.sim_wm, box.sim_gm, box.sim_csf, box.sim_mask, params, None)
    comp = _core.signal_from_mixture(mix, bvals, bvecs, None)
    offsets = _core.shot_offsets(6, 1, "sequential", 3.0)
    n = int(bvals.size)
    good = np.zeros((n * 6, 6))
    for bad in (np.zeros((n, 5)), np.full((n, 6), np.nan)):
        with pytest.raises(ValueError, match="volume_poses"):
            comp.apply_shot_motion(np.asarray(box.sim_affine), offsets, 3.0, bvals, b_max, [], None, None, bad, 1, shot_poses=good)
    with pytest.raises(ValueError, match="volume_poses"):
        comp.apply_shot_motion(np.asarray(box.sim_affine), offsets, 3.0, bvals, b_max, [], None, None, None, 1, shot_poses=good)
    with pytest.raises(ValueError, match="shot_poses"):
        comp.apply_shot_motion(np.asarray(box.sim_affine), offsets, 3.0, bvals, b_max, [], None, None, np.zeros((n, 6)), 1,
                               shot_poses=good[:-1])
    # sample_poses refuses what a timed trace cannot answer
    with pytest.raises(ValueError):
        _core.sample_poses(np.array([]), np.zeros((0, 6)), [0.0], "trace")
    with pytest.raises(ValueError, match="not finite"):
        _core.sample_poses(np.array([0.0, 1.0]), np.zeros((2, 6)), [np.nan], "trace")
    m = ts.Motion.timed([0.0, 10.0], [[0, 0, 0], [0, 0, 1]], [[0, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError, match="positive tr_s"):
        m.poses_for(2, tr_s=0.0)


def test_shot_sampling_works_chunked_and_refuses_dropout(gtab6):
    # per-shot sampling on slabs: the re-simulation and the shot plan both rotate about the whole
    # stack's centre, so a chunked run equals the unchunked one; Artifacts.dropout is refused
    # rather than silently dropped
    obj = _streamline_box(nz=12)
    proto = ts.Protocol.DEFAULT.replace(tr_s=3.0)
    n = int(gtab6.bvals.size)
    m = ts.Motion.timed([0.0, 3.0 * n], [[0, 0, 0], [0, 1.0, 0.3]], [[0, 0, 0], [0, 0, 5.0]])
    assert m.sampling == "shot", "a timed trace is sampled per shot by default"
    full = obj.simulate(gtab6, proto, ts.Artifacts(motion=m), kspace=False, clean_b0=False)
    chunked = obj.simulate(gtab6, proto, ts.Artifacts(motion=m), kspace=False, clean_b0=False, chunk=4)
    a = np.asarray(full.magnitude.dataobj, dtype=float)
    b = np.asarray(chunked.magnitude.dataobj, dtype=float)
    assert np.allclose(a, b, rtol=1e-5, atol=1e-6 * a.max())
    assert full.motion_shots == chunked.motion_shots
    with pytest.raises(ValueError, match="ShotEvents"):
        obj.simulate(gtab6, proto.replace(mb=2), ts.Artifacts(motion=m, dropout=0.5), kspace=False, clean_b0=False)


def test_a_protocol_slice_timing_offset_places_the_reference_pose(gtab6, monkeypatch):
    # implementation review: a protocol SliceTiming that starts at 0.2 s puts each volume's first
    # shot, and so its re-simulated pose, there (final review: checked on the poses the
    # re-simulation receives, not only on the shots table)
    import importlib

    sim_mod = importlib.import_module("trxscan.simulate")
    seen = []
    original = sim_mod._core.signal_moving

    def spy(*args, **kw):
        seen.append(np.asarray(args[-1], dtype=float).copy())
        return original(*args, **kw)

    monkeypatch.setattr(sim_mod._core, "signal_moving", spy)
    obj = _streamline_box(nz=6)
    proto = ts.Protocol.DEFAULT.replace(tr_s=3.0, slice_timing=(0.2, 0.7, 1.2, 1.7, 2.2, 2.7))
    n = int(gtab6.bvals.size)
    ramp = ts.Motion.timed([0.0, 3.0 * n + 1], [[0, 0, 0], [0, 0, 3.0 * n + 1]], [[0, 0, 0], [0, 0, 0]], sampling="shot")
    sim = obj.simulate(gtab6, proto, ts.Artifacts(motion=ramp, shot_events=ts.ShotEvents(slice_order="protocol")),
                       kspace=False, clean_b0=False)
    first = [l.split("\t") for l in sim.motion_shots.splitlines()[1:] if l.split("\t")[1] == "0"]
    for r in first:
        v, t, z = int(r[0]), float(r[3]), float(r[6])
        assert t == pytest.approx(3.0 * v + 0.2) and z == pytest.approx(t, abs=1e-9)
    assert len(seen) == 1, "one re-simulation"
    assert np.allclose(seen[0][:, 2], 3.0 * np.arange(n) + 0.2, atol=1e-9), seen[0][:, 2]


def _streamline_box(nz=6):
    """A box with straight streamlines along x through it (the re-simulation path needs them)."""
    import dataclasses

    box = ts.objects.box(8, matrix=16, nz=nz)
    aff = np.asarray(box.affine, dtype=float)
    nx, ny, _ = box.dims
    pts, offs = [], [0]
    for j in np.linspace(2, ny - 3, 5):
        for k in np.linspace(0.5, nz - 1.5, 4):
            line = [aff @ np.array([x, j, k, 1.0]) for x in np.linspace(0, nx - 1, 24)]
            pts.extend(p[:3] for p in line)
            offs.append(len(pts))
    sl = ts.Streamlines(np.array(pts), np.array(offs))
    return dataclasses.replace(box, streamlines=sl, fibers=None)


def test_rotating_motion_on_slabs_uses_the_full_stack(gtab6):
    # the re-simulation of a slab rotates about the full stack's centre, so a chunked run (or a
    # slices= run) of a rotating head is the unchunked one: the slabs of a volume move alike
    obj = _streamline_box(nz=12)
    n = int(gtab6.bvals.size)
    rot = np.tile([3.0, -2.5, 1.5], (n, 1)) * np.arange(n)[:, None] / n
    m = ts.Motion.from_arrays(np.tile([0.2, -0.1, 0.15], (n, 1)), rot)
    run = lambda **kw: np.asarray(obj.simulate(gtab6, ts.Protocol.DEFAULT, ts.Artifacts(motion=m), kspace=False,  # noqa: E731
                                               clean_b0=False, **kw).magnitude.dataobj, dtype=float)
    full = run()
    chunked = run(chunk=4)
    # away from the slab edges (the slabs' context reaches 3 slices)
    inner = [z for z in range(12) if z % 4 not in (0, 3)]
    assert np.allclose(chunked[..., inner, :], full[..., inner, :], rtol=1e-5, atol=1e-6 * full.max())
    part = run(slices=[5, 6])
    assert np.allclose(part, full[..., 5:7, :], rtol=1e-5, atol=1e-6 * full.max())
    still = np.asarray(obj.simulate(gtab6, ts.Protocol.DEFAULT, ts.Artifacts(), kspace=False, clean_b0=False).magnitude.dataobj)
    assert not np.allclose(full[..., -1], still[..., -1], rtol=1e-3), "the last volume is rotated"
    from trxscan import _core

    with pytest.raises(ValueError, match="nz_full"):
        _core.signal_moving(obj.sim_dims, obj.sim_wm, obj.sim_gm, obj.sim_csf, obj.sim_mask, np.asarray(obj.sim_affine),
                            obj.streamlines.positions, obj.streamlines.offsets, gtab6.bvals, gtab6.bvecs,
                            ts.Protocol.DEFAULT.tissue.to_params(float(gtab6.bvals.max()), ts.Protocol.DEFAULT.diff_scale),
                            np.zeros((n, 6)), z_offset=2, nz_full=12)


def test_a_timed_trace_sampled_per_shot_writes_each_shots_pose(gtab6):
    obj = _streamline_box()
    proto = ts.Protocol.DEFAULT.replace(tr_s=3.0)
    n = int(gtab6.bvals.size)
    ramp = ts.Motion.timed([0.0, 3.0 * n], [[0, 0, 0], [0, 0, 2.0]], [[0, 0, 0], [0, 0, 0]], sampling="shot")
    sim = obj.simulate(gtab6, proto, ts.Artifacts(motion=ramp, shot_events=ts.ShotEvents(slice_order="sequential")),
                       kspace=False, clean_b0=False)
    rows = [l.split("\t") for l in sim.motion_shots.splitlines()[1:]]
    assert len(rows) == n * 6
    for r in rows:
        v, s, t, z = int(r[0]), int(r[1]), float(r[3]), float(r[6])
        assert t == pytest.approx(3.0 * v + 0.5 * s)
        assert z == pytest.approx(2.0 * t / (3.0 * n), abs=1e-9), "the shot's pose is the trace at its time"
    # one pose per volume, asked for: the re-simulation alone, no shots table
    once = ts.Motion.timed([0.0, 3.0 * n], [[0, 0, 0], [0, 0, 2.0]], [[0, 0, 0], [0, 0, 0]], sampling="volume")
    vol = obj.simulate(gtab6, proto, ts.Artifacts(motion=once), kspace=False, clean_b0=False)
    assert vol.motion_shots is None
    # an untimed trajectory stays one pose per volume
    assert ts.Motion.from_arrays(np.zeros((2, 3)), np.zeros((2, 3))).sampling == "volume"
    with pytest.raises(ValueError, match="tr_s"):
        obj.simulate(gtab6, ts.Protocol.DEFAULT, ts.Artifacts(motion=ramp), kspace=False, clean_b0=False)


def test_shot_events_refuse_duplicates_and_state_their_timing(gtab6):
    e = {"volume": 1, "shot": 2, "severity": 0.5, "jump_mm": (0.0, 0.0, 0.0), "jump_deg": (0.0, 0.0, 0.0)}
    with pytest.raises(ValueError, match="second event"):
        _shot_run(gtab6, [e, e])
    box = ts.objects.box(8, matrix=16, nz=6)
    proto = ts.Protocol.DEFAULT.replace(tr_s=3.0)
    sim = box.simulate(gtab6, proto, ts.Artifacts(shot_events=ts.ShotEvents(slice_order="sequential", events=[e])),
                       kspace=False, clean_b0=False)
    assert sim.sidecar["SliceTiming"] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
