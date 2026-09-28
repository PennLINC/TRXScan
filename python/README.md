# trxscan

Python bindings for [TRXScan](https://github.com/PennLINC/TRXScan), a headless diffusion-MRI
simulator (MITK Fiberfox physics in Rust) with realistic single-shot EPI artifacts:
susceptibility distortion, T2/T2* decay, eddy currents, Nyquist ghosts, partial Fourier, Gibbs
ringing, k-space spikes, multi-coil reception with GRAPPA, k-space noise, head motion and
multiband dropout, gradient nonlinearity, synthetic GRE fieldmaps, and the closed-form ground
truth of the same object.

```bash
pip install trxscan                 # wheels for Linux, macOS, Windows; no compiler, no system libs
pip install "trxscan[dipy,trx,viz]"  # dipy gradient tables, TRX streamlines, ODX export for TRXViz
```

## Four nouns, one verb

```python
import trxscan as ts
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs

phantom = ts.Phantom.load("sub-60501")           # real anatomy: tissue maps, 1M streamlines, measured fieldmap, motion traces
gtab = gradient_table(*read_bvals_bvecs("my.bval", "my.bvec"))   # your scheme, in dipy's object
proto = ts.Protocol.HBCD.replace(voxel_mm=2.5, coils=8, accel=2)  # scanner settings; presets are instances
art = ts.Artifacts(noise=2e-4, ghost=0.015)                       # everything off by default

sim = phantom.simulate(gtab, proto, art, slices=40)   # one axial slice, every volume: a few seconds
sim.magnitude          # nibabel Nifti1Image (nx, ny, 1, n_vol) with the slice's true affine
sim.phase              # radians
sim.kspace.acquired    # (n_vol, 1, coils, ky, kx) complex64 as acquired (pre-GRAPPA); .reconstructed, .combined, .mask
sim.readout            # EPI line timing: ky order, t_ms per line, total_readout_ms
sim.to_bids("out/sub-60501_dir-AP")   # the same files the trxscan CLI writes

truth = phantom.microstructure(proto, gtab, slices=40)   # 27 ground-truth maps (fa, md, mk, rtop, icvf, odi, ...)
```

The unit of simulation is a slice: the acquisition stage is exactly per slice, and the signal
stage rasterizes only what crosses it, so a single slice of the full phantom under a 76-volume
scheme with 8 coils and GRAPPA takes about four seconds. Ask for `slices="all"` (or a range)
for a volume.

### Real head motion, real acquisitions

```python
motion = ts.Motion.from_confounds("sub-01_dir-AP_desc-confounds_timeseries.tsv")  # qsiprep/eddy
moving = phantom.simulate(gtab, proto.replace(mb=3), ts.Artifacts(motion=motion, dropout=0.1), slices=40)
moving.dropout          # which shots dropped, and by how much

real = ts.BidsDwi.load("sub-01/dwi/sub-01_dir-AP_dwi.nii.gz")   # sidecar + header -> gtab and Protocol
sim = phantom.simulate(real.gtab, real.protocol, ts.Artifacts(noise=1e-4), slices=40)
```

### Single voxels, like `dipy.sims`, with real artifacts

```python
v = ts.Voxel(fibers=[((1, 0, 0), 0.6), ((0, 1, 0), 0.4)], wm=0.8, gm=0.15, csf=0.05, tissue="adult")
v.signal(gtab, te_ms=88).total        # per-compartment T2, exact directions (matches dipy's single_tensor)
v.truth().fa, v.truth().mk            # what a fit should recover (FORCE closed forms, dipy-validated)
v.signal(gtab, motion=motion)         # fibres rotated per volume: the b-vector rotation problem in one voxel
ts.Voxel.noise(v.signal(gtab).total, 0.02, coils=8, accel=2)   # magnitude noise through the real coil combine

obj = ts.objects.fill(ts.objects.box(8, matrix=32), v)          # the voxel inside a small object
acq = obj.simulate(gtab, ts.Protocol.HBCD.replace(voxel_mm=2.0, coils=8, accel=2), ts.Artifacts(noise=2e-4, ghost=0.02))
acq.magnitude, acq.series((16, 16))                              # Gibbs ringing, PF blur, ghost, GRAPPA noise on a box
```

### Conventions

* Volumes come back as `nibabel.Nifti1Image` with qform and sform set; k-space as
  `complex64` with the phase-encode axis first (`[..., ky, kx]`).
* Everything is deterministic in `Artifacts(seed=...)`; a slice simulated alone is
  bit-identical to that slice of the full run, and `to_bids` reproduces the CLI's files.
* `Artifacts()` is the clean reference. `Protocol.DEFAULT` is the simulator's `Acquisition`
  default; `Protocol.HBCD` is the HBCD-like protocol the CLI ships.
* Partial Fourier behaves like a scanner: the train skips its first lines and reaches the
  k-space centre sooner (`Protocol.readout(...)` shows it); lower `te_ms` yourself to bank the
  TE gain. `pf_mode="contiguous"` restores the legacy tail-dropping rule.
* Set `TRXSCAN_DATA=/path` to use a local copy of the phantom bundles instead of downloading.

### Building from source

```bash
pip install maturin
cd python && maturin develop --release     # in a virtualenv
```

The extension depends only on the std-only simulation core (`kspace,par` features); file
I/O is done in Python (nibabel, trx-python), so no HDF5 or cmake is involved.
