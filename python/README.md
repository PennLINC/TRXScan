# trxscan

Python package for [TRXScan](https://github.com/PennLINC/TRXScan), a diffusion-MRI simulator
written in Rust. It simulates single-shot EPI acquisitions of a tractogram-and-tissue phantom,
with susceptibility distortion, T2/T2* decay, eddy currents, Nyquist ghosts, partial Fourier,
Gibbs ringing, k-space spikes, multi-coil reception with GRAPPA, k-space noise, head motion,
multiband dropout, gradient nonlinearity and a synthetic GRE fieldmap, and it computes the
ground truth of the same object.

```bash
pip install trxscan                  # binary wheels for Linux, macOS and Windows
pip install "trxscan[dipy,trx,viz]"  # dipy gradient tables, TRX streamlines, ODX export for TRXViz
```

## Overview

```python
import trxscan as ts
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs

phantom = ts.Phantom.load("sub-60501")           # one subject: tissue maps, 1M streamlines, measured fieldmap, motion traces
gtab = gradient_table(*read_bvals_bvecs("my.bval", "my.bvec"))   # any dipy GradientTable
proto = ts.Protocol.HBCD.replace(voxel_mm=2.5, coils=8, accel=2)  # scanner settings; presets are instances
art = ts.Artifacts(noise=2e-4, ghost=0.015)                       # artifacts are off unless set

sim = phantom.simulate(gtab, proto, art, slices=40)   # one axial slice, every volume
sim.magnitude          # nibabel Nifti1Image (nx, ny, 1, n_vol) with the slice's affine
sim.phase              # radians
sim.kspace.acquired    # (n_vol, 1, coils, ky, kx) complex64 as acquired (pre-GRAPPA); .reconstructed, .combined, .mask
sim.readout            # EPI line timing: ky order, t_ms per line, total_readout_ms
sim.to_bids("out/sub-60501_dir-AP")   # the same files the trxscan CLI writes

truth = phantom.microstructure(proto, gtab, slices=40)   # 27 ground-truth maps (fa, md, mk, rtop, icvf, odi, ...)
```

The unit of simulation is a slice. The acquisition stage runs per slice, and the signal stage
rasterizes only the streamlines that cross the requested slices, so one slice of the full
phantom under a 76-volume scheme with 8 coils and GRAPPA takes a few seconds. Pass
`slices="all"` or a range for a volume.

### Head motion and real acquisition parameters

```python
motion = ts.Motion.from_confounds("sub-01_dir-AP_desc-confounds_timeseries.tsv")  # qsiprep/eddy
moving = phantom.simulate(gtab, proto.replace(mb=3), ts.Artifacts(motion=motion, dropout=0.1), slices=40)
moving.dropout          # which shots dropped, and by how much

real = ts.BidsDwi.load("sub-01/dwi/sub-01_dir-AP_dwi.nii.gz")   # sidecar + header -> gtab and Protocol
sim = phantom.simulate(real.gtab, real.protocol, ts.Artifacts(noise=1e-4), slices=40)
```

### Single voxels

The `Voxel` API follows `dipy.sims`: one voxel, a gradient table, a signal. It adds the tissue
presets with per-compartment T2, the closed-form ground truth of the voxel, and the option of
placing the voxel inside a small object and running it through the acquisition stage.

```python
v = ts.Voxel(fibers=[((1, 0, 0), 0.6), ((0, 1, 0), 0.4)], wm=0.8, gm=0.15, csf=0.05, tissue="adult")
v.signal(gtab, te_ms=88).total        # per-compartment T2; a single stick matches dipy's single_tensor
v.truth().fa, v.truth().mk            # what a fit should recover (FORCE closed forms, tested against dipy)
v.signal(gtab, motion=motion)         # fibres rotated per volume
ts.Voxel.noise(v.signal(gtab).total, 0.02, coils=8, accel=2)   # magnitude noise through the coil combine

obj = ts.objects.fill(ts.objects.box(8, matrix=32), v)          # the voxel inside a small object
acq = obj.simulate(gtab, ts.Protocol.HBCD.replace(voxel_mm=2.0, coils=8, accel=2), ts.Artifacts(noise=2e-4, ghost=0.02))
acq.magnitude, acq.series((16, 16))                              # Gibbs ringing, partial-Fourier blur, ghost and GRAPPA noise on a box
```

### Conventions

* Volumes come back as `nibabel.Nifti1Image` with qform and sform set; k-space as
  `complex64` with the phase-encode axis first (`[..., ky, kx]`).
* Results are deterministic for a given `Artifacts(seed=...)`. A slice simulated alone is
  identical to that slice of the full run, and `to_bids` writes the same files as the CLI.
* `Artifacts()` is the clean reference. `Protocol.DEFAULT` is the simulator's `Acquisition`
  default; `Protocol.HBCD` is the HBCD-like protocol the CLI uses.
* Partial Fourier behaves like a scanner: the train skips its first lines and reaches the
  k-space centre sooner (`Protocol.readout(...)` shows the timing). Lower `te_ms` yourself to
  take the TE gain. `pf_mode="contiguous"` restores the older tail-dropping rule.
* Set `TRXSCAN_DATA=/path` to use a local copy of the phantom bundles instead of downloading.

### Building from source

```bash
pip install maturin
cd python && maturin develop --release     # in a virtualenv
```

The extension depends only on the std-only simulation core (`kspace,par` features). File I/O
is done in Python with nibabel and trx-python, so no HDF5 or cmake is involved.
