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
