"""Checks --oversample emits a sim grid that is an exact integer refinement of the acq grid."""
import subprocess, sys, pathlib

import pytest

# Imported lazily: nibabel is not part of the self-contained CI environment, and a module-level
# import fails at COLLECTION -- before any in-test skip guard can run.
nib = pytest.importorskip("nibabel")


def test_sim_grid_is_integer_refinement(tmp_path):
    here = pathlib.Path(__file__).parent
    # The truth-data bundle is a sibling of the TRXScan repo, not part of it.
    root = here.parent.parent / "data" / "trxscan_truth_data"
    if not (root / "sub-0001a" / "anat").is_dir():
        import pytest
        pytest.skip(f"truth-data bundle not found at {root}", allow_module_level=False)
    subprocess.run([sys.executable, str(here / "prepare_acquisition_grid.py"),
                    "--anat-dir", str(root / "sub-0001a" / "anat"),
                    "--prefix", "sub-0001a_space-ACPC",
                    "--out", str(tmp_path), "--voxel", "1.7", "--oversample", "2"],
                   check=True)
    acq = nib.load(tmp_path / "wm.nii.gz")
    sim = nib.load(tmp_path / "sim" / "wm.nii.gz")
    # in-plane refined by exactly 2, slice direction untouched (spec 3.1)
    assert sim.shape[0] == acq.shape[0] * 2
    assert sim.shape[1] == acq.shape[1] * 2
    assert sim.shape[2] == acq.shape[2]
    # same FOV
    for i in (0, 1):
        assert abs(sim.shape[i] * sim.header.get_zooms()[i]
                   - acq.shape[i] * acq.header.get_zooms()[i]) < 1e-3


def _synthetic_source(tmp_path, zoom):
    """Write a probseg/fieldmap set on a `zoom` mm isotropic grid.

    Deliberately not 1 mm: `--voxel` used to be applied as an index-space scale factor, which
    silently meant `--voxel * zoom` millimetres. On the bundled 1 mm anatomicals the two are the
    same number, so no test using them could see the difference.
    """
    import numpy as np
    d = tmp_path / "anat"
    d.mkdir(parents=True, exist_ok=True)
    # Fixed 96 mm FOV holding a fixed 32 mm ball, so the PHYSICAL object is identical across
    # source spacings and only the sampling changes. A ball defined in voxels would grow with the
    # spacing and the extent check below would be measuring the fixture, not the script.
    n = int(round(96.0 / zoom))
    aff = np.diag([zoom, zoom, zoom, 1.0])
    xx, yy, zz = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    c = (n - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2) * zoom
    ball = (r < 16.0).astype(np.float32)
    for t, v in [("WM", ball), ("GM", ball * 0.0), ("CSF", ball * 0.0)]:
        nib.save(nib.Nifti1Image(v, aff), d / f"synth_label-{t}_probseg.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((n, n, n), np.float32), aff),
             d / "synth_desc-atlas_fieldmap.nii.gz")
    return d


@pytest.mark.parametrize("zoom", [0.8, 1.0, 2.0])
def test_voxel_is_millimetres_regardless_of_source_spacing(tmp_path, zoom):
    import numpy as np
    here = pathlib.Path(__file__).parent
    src = _synthetic_source(tmp_path / f"z{zoom}", zoom)
    out = tmp_path / f"out{zoom}"
    subprocess.run([sys.executable, str(here / "prepare_acquisition_grid.py"),
                    "--anat-dir", str(src), "--prefix", "synth",
                    "--out", str(out), "--voxel", "1.7"], check=True)
    img = nib.load(out / "wm.nii.gz")
    # The header must say 1.7 mm AND the affine must mean it. Before the fix this was
    # `1.7 * zoom` mm for a source that was not 1 mm isotropic.
    assert np.allclose(img.header.get_zooms()[:3], 1.7, atol=1e-5), img.header.get_zooms()
    assert np.allclose(np.linalg.norm(img.affine[:3, :3], axis=0), 1.7, atol=1e-5), img.affine

    # And the physical FOV must not depend on the source spacing: 32 mm of ball plus the default
    # padding of [4, 14, 2] TARGET voxels per side, within a target voxel of bounding-box rounding.
    expected = 32.0 + 2 * np.array([4, 14, 2]) * 1.7
    extent = np.array(img.shape) * 1.7
    assert np.all(np.abs(extent - expected) <= 2 * 1.7), (extent, expected)
