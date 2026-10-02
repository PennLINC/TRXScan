"""TRXScan's FSL bvec convention against dcm2niix on real oblique acquisitions.

The fixture holds, for one Siemens Prisma DTI protocol acquired orthogonal and tilted (pitch,
roll, yaw, two axes), the per-volume DICOM gradient direction (patient LPS frame), dcm2niix's
reference affine and its FSL ``.bvec`` (from neurolabusc/dcm_qa_dti). The simulator's
gradients are world RAS vectors and ``_core.fsl_bvecs`` writes them in FSL's image-axes
convention; the two must agree exactly."""

import json
from pathlib import Path

import numpy as np

from trxscan import _core

LPS = np.diag([-1.0, -1.0, 1.0])


def _angle(a, b):
    c = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def test_fsl_bvecs_match_dcm2niix_on_oblique_siemens_dti():
    fx = json.load(open(Path(__file__).parent / "data" / "dcm_qa_dti_bvecs.json"))
    tilts = {}
    for name, s in fx["series"].items():
        affine = np.array(s["affine"]); bvals = np.array(s["bvals"]); g_lps = np.array(s["grad_lps"]); ref = np.array(s["dcm2niix_bvec"])
        dwi = bvals > 50
        got = np.asarray(_core.fsl_bvecs(np.ascontiguousarray(g_lps @ LPS.T), np.ascontiguousarray(affine))).reshape(-1, 3)
        ang = _angle(got[dwi], ref[dwi])
        assert ang.max() < 0.01, (name, ang.max())           # same direction AND sign, every volume
        R = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
        tilts[name] = np.degrees(np.arccos(abs(R[2, 2])))
    assert max(tilts.values()) > 25.0                          # the fixture really is oblique
    # positive determinant (what the simulator writes with fsl_orientation off): FSL's rule is
    # that mirroring the voxel x axis leaves the bvec text unchanged
    s = fx["series"]["dti_nopf_x2_2axis"]; affine = np.array(s["affine"]); ref = np.array(s["dcm2niix_bvec"]); bvals = np.array(s["bvals"])
    F = np.eye(4); F[0, 0] = -1; F[0, 3] = s["shape"][0] - 1
    flipped = affine @ F
    assert np.linalg.det(flipped[:3, :3]) > 0 > np.linalg.det(affine[:3, :3])
    g_ras = ref @ (affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)).T
    got = np.asarray(_core.fsl_bvecs(np.ascontiguousarray(g_ras), np.ascontiguousarray(flipped))).reshape(-1, 3)
    assert _angle(got[bvals > 50], ref[bvals > 50]).max() < 0.01
