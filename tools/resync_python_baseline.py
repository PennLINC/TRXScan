#!/usr/bin/env python3
"""The re-sync baseline of the Python bindings (mrsim-acq docs/plans/2026-10-08-trxscan-resync.md, Task 1).

Calls the acquisition surface of ``trxscan._core`` on small deterministic inputs and writes one SHA-256 per
returned array or value, so the re-pointed bindings can be checked bit for bit against main's.

    python tools/resync_python_baseline.py <output-file>

Build the extension first (``maturin develop --release`` in ``python/``).
"""
import hashlib
import sys

import numpy as np
from trxscan import _core as c


def digest(x):
    """SHA-256 of an array (dtype, shape and bytes) or of a value's repr, recursively for containers."""
    if isinstance(x, np.ndarray):
        h = hashlib.sha256(str((x.dtype.str, x.shape)).encode())
        h.update(np.ascontiguousarray(x).tobytes())
        return h.hexdigest()
    if isinstance(x, dict):
        return {k: digest(v) for k, v in sorted(x.items())}
    if isinstance(x, (list, tuple)):
        return [digest(v) for v in x]
    return hashlib.sha256(repr(x).encode()).hexdigest()


def flatten(prefix, d, out):
    if isinstance(d, dict):
        for k, v in d.items():
            flatten(f"{prefix}.{k}", v, out)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            flatten(f"{prefix}[{i}]", v, out)
    else:
        out.append((prefix, d))


NX, NY, NZ, NGRAD, O = 16, 16, 4, 3, 2
SIM = (NX * O, NY * O, NZ)
BVALS = np.array([0.0, 1000.0, 2000.0])
BVECS = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def compartments(nz=NZ):
    rng = np.random.default_rng(20261008)
    n = SIM[0] * SIM[1] * nz * NGRAD
    imgs = [rng.random(n, dtype=np.float32) for _ in range(3)]
    return c.Compartments.from_images((SIM[0], SIM[1], nz), NGRAD, imgs, [80.0, 90.0, 2000.0])


def fieldmap(nz=NZ):
    rng = np.random.default_rng(7)
    return (rng.standard_normal(SIM[0] * SIM[1] * nz) * 5.0).astype(np.float32)


def main(path):
    r = {}
    pf = {"partial_fourier": 0.75}
    # the defaults: an empty dict pins Acquisition::default() (pf_mode among them), hbcd its preset
    r["defaults"] = c.acquisition_defaults({})
    r["defaults_pf"] = c.acquisition_defaults(pf)
    r["hbcd"] = c.acquisition_hbcd(NY)
    for mode in [None, "scanner", "contiguous", "fiberfox"]:
        a = dict(pf, **({"pf_mode": mode} if mode else {}))
        tag = mode or "default"
        r[f"mask_{tag}"] = c.sampling_mask(NX, NY, a)
        r[f"timing_{tag}"] = c.epi_timing(NX, NY, a)
        r[f"trajectory_{tag}"] = c.epi_trajectory(NX, NY, a)
    r["dropout_seed"] = c.dropout_seed(20261008)
    events = c.dropout_events(BVALS, 4, 0.5, c.dropout_seed(20261008))
    r["dropout_events"] = events
    r["hires_grid"] = c.hires_grid(np.eye(4), (NX, NY, NZ), O)
    acq_dims = (NX, NY, NZ)
    cases = {
        "default": dict(acquisition={}),
        "pf_eddy_noise": dict(acquisition=dict(pf, eddy_strength=0.03, eddy_phase=0.02, noise_variance=0.01), seed=5),
        "capture": dict(acquisition=dict(pf), kspace_slices=[1], capture=(True, True, True)),
        "te_per_volume": dict(acquisition={}, te_per_volume=[60.0, 70.0, 80.0]),
        "noise_map": dict(acquisition={}, noise_sigma=np.linspace(0.0, 1.0, NX * NY * NZ, dtype=np.float32), seed=3),
        "phase_none": dict(acquisition=dict(pf, pf_mode="contiguous"), phase_model="none"),
        # several slices in a non-sorted order, several coils (the capture's order and layouts)
        "capture_multi": dict(acquisition=dict(pf, n_coils=2, accel=2, acs_lines=6), kspace_slices=[3, 0, 2],
                              capture=(True, True, True), seed=9),
    }
    for name, kw in cases.items():
        r[f"acq_{name}"] = c.simulate_acquisition(compartments(), fieldmap(), acq_dims, BVALS, BVECS, **kw)
    # a slab: slices 1-2 of 4
    r["acq_slab"] = c.simulate_acquisition(compartments(2), fieldmap(2), (NX, NY, 2), BVALS, BVECS,
                                           acquisition=dict(pf), slice_z=[1, 2], nz_full=4)
    # multiband motion with dropout, whole volume and a slab
    for name, kw, nz in [("mb", {}, NZ), ("mb_slab", dict(slice_z=[1, 2], nz_full=4), 2)]:
        cm = compartments(nz)
        dropped = cm.apply_multiband_motion(np.eye(4), 2, True, BVALS, 2000.0, events, **kw)
        r[name] = dict(dropped=dropped, images=cm.images())
    # a nominal b_max below jittered b-values
    jitter = np.array([0.0, 1005.0, 995.0])
    ev_j = c.dropout_events(jitter, 4, 1.0, c.dropout_seed(7))
    cm = compartments()
    r["mb_jitter"] = dict(dropped=cm.apply_multiband_motion(np.eye(4), 2, True, jitter, 1000.0, ev_j), images=cm.images())
    flat = []
    flatten("", digest(r), flat)
    with open(path, "w") as f:
        for k, v in flat:
            f.write(f"{v}  {k[1:]}\n")
    print(f"{len(flat)} hashes -> {path}")


if __name__ == "__main__":
    main(sys.argv[1])
