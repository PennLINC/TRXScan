import os

import numpy as np
import pytest

import trxscan as ts

odx = pytest.importorskip("odx")


@pytest.fixture(scope="module")
def cross_sim(gtab6):
    cross = ts.objects.crossing(60, matrix=16)
    return cross.simulate(gtab6, ts.Protocol.DEFAULT.replace(voxel_mm=2.0), ts.Artifacts(), kspace=False)


def test_hemisphere_matches_odf8(cross_sim):
    from trxscan.viz import _match_hemisphere

    src = np.asarray(cross_sim.mixture.sphere_vertices()).reshape(-1, 3)
    dst, faces = odx.spheres.dsistudio_odf8()
    idx, sign, worst = _match_hemisphere(src, np.asarray(dst, dtype=np.float64))
    assert src.shape == (321, 3) and np.asarray(dst).shape == (321, 3)
    # different tessellations of the same size: nearest-vertex resampling within a few degrees
    assert worst < 8.0
    assert len(set(idx.tolist())) > 250


def test_trx_round_trip_copies_out_of_the_memmap(tmp_path):
    from trxscan.phantom import Streamlines
    from trxscan.viz import write_trx

    rng = np.random.default_rng(0)
    pos = rng.normal(size=(3000, 3)) * 10
    off = np.arange(0, 3001, 10, dtype=np.uint32)
    sl = Streamlines(pos, off, np.full(300, 20.5, np.float32))  # float32 weights: the no-cast path
    back = Streamlines.load(write_trx(sl, tmp_path / "t.trx"))
    assert back.n == 300 and np.abs(back.positions - pos).max() < 1e-3
    assert np.isfinite(back.weights).all() and np.allclose(back.weights, 20.5)


def test_mixture_to_odx_round_trip(cross_sim, tmp_path):
    import warnings

    from trxscan.viz import mixture_to_odx

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        o = mixture_to_odx(cross_sim.mixture, cross_sim.object.sim_dims, cross_sim.object.sim_affine)
    path = tmp_path / "fod.odx"
    o.save(str(path))
    assert path.exists() and path.stat().st_size > 0
    back = odx.load(str(path))
    assert tuple(back.dimensions) == tuple(cross_sim.object.sim_dims)


@pytest.mark.skipif(not (os.environ.get("TRXVIZ_CLI") or __import__("shutil").which("trxviz-cli")), reason="no trxviz-cli")
def test_render_smoke(cross_sim, tmp_path):
    from trxscan.viz import mixture_to_odx, render

    o = mixture_to_odx(cross_sim.mixture, cross_sim.object.sim_dims, cross_sim.object.sim_affine)
    o.save(str(tmp_path / "fod.odx"))
    png = render(odx=tmp_path / "fod.odx", width=200, height=150, env={"LIBGL_ALWAYS_SOFTWARE": "1"})
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
