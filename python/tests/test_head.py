"""The DWI's non-brain head (``Protocol.head``) from the phantom's T2w."""

import nibabel as nib
import numpy as np
import pytest
from nibabel.processing import resample_from_to

import trxscan as ts

#: T2w / T1w intensities per region of the synthetic head (air is 0 in both).
T2W = {"wm": 300.0, "gm": 450.0, "csf": 1000.0, "muscle": 250.0, "eye": 1600.0, "fat": 900.0}
T1W = {"wm": 1000.0, "gm": 700.0, "csf": 200.0, "muscle": 500.0, "eye": 150.0, "fat": 1600.0}


@pytest.fixture(scope="module")
def headed():
    """A 1 mm LPS brain slab inside a head: muscle all round, a fat layer on its outer edge and a
    fluid "eye" in front, all within the brain's slices; a T1w and a T2w over everything."""
    nx, ny, nz = 40, 44, 10
    lab = {k: np.zeros((nx, ny, nz), np.float32) for k in T2W}
    lab["wm"][14:26, 15:29, 2:8] = 1
    lab["gm"][12:14, 13:31, 2:8] = lab["gm"][26:28, 13:31, 2:8] = 1
    lab["csf"][14:26, 13:15, 2:8] = lab["csf"][14:26, 29:31, 2:8] = 1
    brain = lab["wm"] + lab["gm"] + lab["csf"]
    head = np.zeros_like(brain)
    head[5:35, 5:39, 2:8] = 1
    lab["fat"][5:35, 5:39, 2:8] = 1
    lab["fat"][7:33, 7:37, 2:8] = 0
    lab["eye"][17:23, 33:37, 3:7] = 1
    lab["muscle"] = head * (1 - brain) * (1 - lab["fat"]) * (1 - lab["eye"])
    aff = np.diag([-1.0, -1.0, 1.0, 1.0])
    aff[:3, 3] = [nx / 2, ny / 2, -nz / 2]
    img = lambda a: nib.Nifti1Image(a.astype(np.float32), aff)  # noqa: E731
    pts, offs = [], [0]
    for y in np.linspace(-5, 5, 6):
        line = np.stack([np.linspace(-5, 5, 21), np.full(21, y), np.zeros(21)], 1)
        pts.append(line)
        offs.append(offs[-1] + 21)
    sl = ts.Streamlines(np.concatenate(pts), np.array(offs, np.uint32), None)
    t2 = sum(T2W[k] * lab[k] for k in lab)
    t1 = sum(T1W[k] * lab[k] for k in lab)
    ph = ts.Phantom.from_files(wm=img(lab["wm"]), gm=img(lab["gm"]), csf=img(lab["csf"]), mask=img(brain > 0), streamlines=sl,
                               t1w=img(t1), t2w=img(t2), name="headed")
    return ph, {k: img(v) for k, v in lab.items()}


PROTO = ts.Protocol.DEFAULT.replace(voxel_mm=2.0, oversample=1, pad=(1, 3, 0))
GTAB = (np.array([0.0, 1000.0]), np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))


def _region(lab, sim, k):
    """Voxels of region ``k`` (most of the voxel) on the simulation's output grid."""
    out = sim.magnitude
    return resample_from_to(lab[k], (out.shape[:3], out.affine), order=1).get_fdata() > 0.75


def test_head_needs_a_t2w(headed):
    ph, _ = headed
    bare = ts.Phantom.from_files(wm=ph.wm, gm=ph.gm, csf=ph.csf, mask=ph.mask, streamlines=ph.streamlines, t1w=ph.t1w)
    with pytest.raises(ValueError, match="T2w"):
        bare.grid(PROTO.replace(head=True))


def test_head_widens_the_grid_in_plane_only(headed):
    ph, _ = headed
    bare, wide = ph.grid(PROTO), ph.grid(PROTO.replace(head=True))
    assert wide.dims[0] > bare.dims[0] and wide.dims[1] > bare.dims[1]
    assert wide.dims[2] == bare.dims[2]
    assert bare.sim_t2w is None and wide.sim_t2w is not None and wide.sim_fat is not None


def test_head_b0_has_bright_fluid_and_dark_tissue(headed):
    ph, lab = headed
    sim = ph.simulate(GTAB, PROTO.replace(head=True), ts.Artifacts(), clean_b0=True)
    b0 = sim.clean_b0.get_fdata()
    level = {k: float(np.median(b0[_region(lab, sim, k)])) for k in ("csf", "muscle", "eye", "fat")}
    head = sum(np.asarray(lab[k].dataobj) for k in lab)
    air = b0[resample_from_to(nib.Nifti1Image(head, lab["wm"].affine), (b0.shape, sim.clean_b0.affine), order=0).get_fdata() == 0]
    # fluid as bright as CSF; muscle (T2 35 ms at TE 90) and saturated fat nearly dark
    assert level["eye"] == pytest.approx(level["csf"], rel=0.2), level
    assert 0 < level["muscle"] < 0.15 * level["csf"], level
    assert 0 < level["fat"] < 0.15 * level["csf"], level
    # the air holds only the reconstruction's ringing off the head's edge
    assert np.abs(air).mean() < 0.01 * level["csf"]
    # without the head, nothing outside the brain
    bare = ph.simulate(GTAB, PROTO, ts.Artifacts(), clean_b0=True)
    assert bare.clean_b0.shape != sim.clean_b0.shape or float(np.median(bare.clean_b0.get_fdata()[_region(lab, sim, "eye")])) == 0.0


def test_head_fluid_attenuates_more_than_tissue(headed):
    ph, lab = headed
    sim = ph.simulate(GTAB, PROTO.replace(head=True), ts.Artifacts(), clean_b0=False)
    mag = sim.magnitude.get_fdata()
    ratio = {k: float(np.median(mag[..., 1][_region(lab, sim, k)] / mag[..., 0][_region(lab, sim, k)])) for k in ("eye", "muscle")}
    assert ratio["eye"] < ratio["muscle"] < 1.0, ratio


def test_head_slabs_match_the_whole_run(headed):
    ph, _ = headed
    proto = PROTO.replace(head=True)
    whole = ph.simulate(GTAB, proto, ts.Artifacts(), clean_b0=True)
    slabs = ph.simulate(GTAB, proto, ts.Artifacts(), clean_b0=True, chunk=2)
    np.testing.assert_allclose(slabs.magnitude.get_fdata(), whole.magnitude.get_fdata(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(slabs.clean_b0.get_fdata(), whole.clean_b0.get_fdata(), rtol=1e-5, atol=1e-5)


def test_head_is_refused_with_motion(headed):
    ph, _ = headed
    mot = ts.Motion.from_arrays(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="motion"):
        ph.simulate(GTAB, PROTO.replace(head=True), ts.Artifacts(motion=mot))


#: Two eyes in the air in front of the synthetic head (world RAS mm; the grid is LPS, 1 mm).
EYES = ((-4.0, -21.0, 0.5), (4.0, -21.0, 0.5))


def _at(img, xyz):
    ijk = np.round(np.linalg.solve(img.affine, np.append(xyz, 1.0))[:3]).astype(int)
    return float(np.asarray(img.dataobj)[tuple(ijk)])


def test_eyes_are_fluid_outside_a_defaced_head(headed):
    ph, lab = headed
    eyed = ph.with_eyes(EYES, radius_mm=3.0)
    sim = eyed.simulate(GTAB, PROTO.replace(head=True), ts.Artifacts(), clean_b0=True)
    b0 = sim.clean_b0.get_fdata()
    csf = float(np.median(b0[_region(lab, sim, "csf")]))
    for centre in EYES:
        assert _at(sim.clean_b0, centre) == pytest.approx(csf, rel=0.25)
    without = ph.simulate(GTAB, PROTO.replace(head=True), ts.Artifacts(), clean_b0=True)
    assert _at(without.clean_b0, EYES[0]) < 0.05 * csf
    # the in-plane field of view grows to take them in
    assert eyed.grid(PROTO.replace(head=True)).dims[1] > ph.grid(PROTO.replace(head=True)).dims[1]


def test_eyes_move_with_the_head(headed):
    ph, _ = headed
    moved = ph.with_eyes(EYES, radius_mm=3.0).moved((2.0, -1.0, 0.0, 0.0, 0.0, 90.0))
    T = ts.bids._offset_of((2.0, -1.0, 0.0, 0.0, 0.0, 90.0), ph.wm)
    for (got, _), centre in zip(moved.eyes, EYES, strict=True):
        np.testing.assert_allclose(got, T[:3, :3] @ np.asarray(centre) + T[:3, 3], atol=1e-6)


def test_receive_bias_brightens_the_periphery_and_spares_the_clean_b0(headed):
    ph, lab = headed
    proto = PROTO.replace(head=True)
    flat = ph.simulate(GTAB, proto, ts.Artifacts(), clean_b0=True)
    shaded = ph.simulate(GTAB, proto, ts.Artifacts(receive_bias=0.3), clean_b0=True)
    f0 = flat.magnitude.get_fdata()[..., 0]
    ratio = shaded.magnitude.get_fdata()[..., 0] / np.maximum(f0, 1e-6)
    nx, ny, _ = ratio.shape
    u = (np.arange(nx) - (nx - 1) / 2) / ((nx - 1) / 2)
    w = (np.arange(ny) - (ny - 1) / 2) / ((ny - 1) / 2)
    radius = np.sqrt(u[:, None] ** 2 + w[None, :] ** 2)[..., None] * np.ones(ratio.shape)
    signal = f0 > 0.2 * np.percentile(f0, 99)
    assert np.median(ratio[signal & (radius < 0.3)]) < np.median(ratio[signal & (radius > 0.5)])
    np.testing.assert_allclose(shaded.clean_b0.get_fdata(), flat.clean_b0.get_fdata())
