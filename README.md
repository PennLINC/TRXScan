# TRXScan

A headless diffusion-MRI simulator written in Rust, used from Python (`pip install trxscan`)
or from the command line. It takes a
tractogram, tissue volume-fraction maps and a gradient scheme, and writes a complex-valued 4D
DWI with the artifacts of a single-shot EPI acquisition, together with the ground truth for
each artifact that has one.

The signal and k-space models originate in [MITK Fiberfox](https://github.com/MIC-DKFZ/MITK-Diffusion)
(Neher et al. 2014; see [Origins](#origins)). TRXScan reimplements those parts as a Rust crate
with no system dependencies and adds an acquisition model that works per slice rather than per
volume: within-volume (multiband) motion, GRAPPA, complex magnitude and phase output,
scanner-style partial Fourier, gradient nonlinearity, a synthetic GRE fieldmap of the same
object, and analytic microstructure ground truth computed from the same per-voxel mixture the
signal is generated from.

**Input:** a tractogram (TRX/TRK/TCK/VTK), tissue volume-fraction maps, a gradient scheme
(a dipy `GradientTable` or FSL `.bval`/`.bvec`), and optionally a fieldmap.
**Output:** a complex 4D DWI as nibabel images or BIDS files (`part-mag` / `part-phase` with
`.bval`/`.bvec` and JSON sidecars), the per-coil k-space and readout timing, and ground truth:
dropped slices, the noise-level map, the gradient-nonlinearity field, fibre orientations, and
27 microstructure scalars.

```
streamlines + tissue maps + scheme
        │  rasterize (3D-DDA, path lengths) → per-voxel orientation histogram
        ▼
   per-voxel, per-gradient clean signal   ── Stick / Tensor / Ball, kept per compartment
        │  (+ gradient nonlinearity: per-voxel Jᵀg and the spatial warp)
        │  per-slice k-space (the per-line sum, FFT/NUFFT-accelerated)
        ▼
   distortion · T2* · eddy · Nyquist ghost · partial Fourier · Gibbs · spikes
   · multi-coil · GRAPPA · k-space noise · object phase
        │
        ▼            (+ motion: per volume, or per multiband shot)
   BIDS complex 4D DWI  (mag + phase, .bval/.bvec/.json)  + ground truth
```

`trxscan-microstructure` derives analytic microstructure maps from the same per-voxel mixture
the simulator generates its signal from, so the DWI and its ground truth describe one object.

## Why this exists

Fiberfox is a GUI-driven simulator built into the MITK-Diffusion stack (ITK, VTK, DCMTK, CTK,
Qt). We wanted a simulator that runs headless, in batch and from notebooks, and an acquisition
model that could be extended per slice. The simulation math lifted out of MITK is a few
thousand lines; TRXScan started as a port of it and has since been extended in the ways listed
above. The ported parts keep references into the MITK source so the two can be compared.

## Capabilities

| Area | What it does |
|---|---|
| Gradient scheme | FSL `.bval`/`.bvec`, shell detection, b-value carried in the gradient norm |
| Rasterization | 3D-DDA segment → (voxel, path length) on the target (oblique) grid |
| Signal models | Stick, Tensor (cylindrically symmetric), Ball; GM as a two-ball (free + soma) mixture |
| Signal stage | per-compartment clean signal from a per-voxel orientation histogram with SIFT2 weights and Watson κ; per-voxel myelination; per-volume re-simulation under motion |
| Acquisition stage (k-space) | EPI distortion, T2*, eddy (linear + quadratic geometric, plus an object-phase ramp), Nyquist ghost, partial Fourier, Gibbs ringing from an oversampled object, spikes, multi-coil Roemer combine, GRAPPA, k-space noise, per-voxel noise map, calibrated object phase, complex output |
| Gradient nonlinearity | synthetic whole-body / Connectom coefficient sets (or a Siemens `.grad` file): per-voxel diffusion-encoding deviation `Jᵀg` and the spatial warp with Jacobian modulation; writes the coefficient file, forward/inverse displacement fields and graddev |
| GRE fieldmap | dual-echo `magnitude1/2` + `phasediff` (or `phase1/2`) from the same off-resonance field, Siemens integer phase, optional coarser resolution with intravoxel dephasing, `B0FieldIdentifier` linkage |
| Motion | random / linear / trajectory poses, qsiprep-TSV traces, multiband schedule, within-volume dropout with ground truth |
| Ground truth | dropped-slice TSV, noise-SD map, GNL field and graddev, up to three fibre peaks per voxel, and 27 microstructure maps (DTI/DKI, QTI, MAP-MRI, NG/PA, GFA/QA, NODDI-style; tested against dipy) |
| I/O | TRX/TRK/TCK/VTK load (with SIFT2 weights from TRX `dps`), NIfTI tissue maps, BIDS complex 4D write in the native or FSL/dcm2niix (LAS) orientation with FSL-convention bvecs |

## Scope and limitations

- **k-space is computed as the per-line sum, not approximated.** The sum is organised for
  speed (static factors hoisted, the affine-in-ky phase advanced by memoised rotors), and the
  `kspace` feature runs it through `rustfft` plus a type-1 NUFFT for the fieldmap term, which
  gives the same numbers about five times faster. The literal sum is kept as a test oracle.
- **The CLI protocol is HBCD-like and set in code.** `Acquisition::hbcd` (TE 88 ms, 6/8 partial
  Fourier, 24 ACS lines, a small Nyquist ghost) is the library default and the flags override
  its artifact settings. There is no config file (the `config` feature is a stub). The Python
  package exposes the protocol as a `Protocol` object instead.
- **Motion mode uses the per-segment signal stage.** With a `--motion` trace, SIFT2 weights
  and the Watson kernel are ignored, and `--myelin`, `--truth-peaks` and `--gnl` are refused.
  The no-motion path and `trxscan-microstructure` share the orientation-histogram path.
- **`Tensor` handles the cylindrically symmetric case only** (`d2 == d3`).
- **The microstructure closed forms are tested against dipy** (30 checked-in fixture mixtures,
  agreement to 1e-8). The k-space stage has physics-behaviour and self-consistency tests but
  no numerical comparison against Fiberfox.

## Python

The `trxscan` package on PyPI wraps the same simulator and is the recommended way to use it
from notebooks and scripts:

```bash
pip install "trxscan[dipy,trx,viz]"    # binary wheels for Linux, macOS and Windows; no Rust toolchain needed
```

```python
import trxscan as ts
from dipy.core.gradients import gradient_table

phantom = ts.Phantom.load("sub-60501")                             # downloaded and cached on first use
gtab = gradient_table(bvals, bvecs)                                 # any scheme
proto = ts.Protocol.HBCD.replace(voxel_mm=2.5, coils=8, accel=2)
sim = phantom.simulate(gtab, proto, ts.Artifacts(noise=2e-4), slices=40)
sim.magnitude, sim.phase          # nibabel images of that slice, every volume
sim.kspace.acquired, sim.readout  # per-coil k-space and EPI line timing
sim.to_bids("out/sub-60501_dir-AP")
truth = phantom.microstructure(proto, gtab, slices=40)             # 27 ground-truth maps
```

Simulation runs per slice, so one slice under a full scheme takes seconds and a slice
simulated alone is identical to that slice of the CLI's full-volume run. `Motion` loads real
head-motion traces (qsiprep or eddy confounds, or registration affines), `BidsDwi` builds a
gradient table and `Protocol` from a real acquisition's sidecar, and `Voxel` gives
dipy.sims-style single-voxel signals with their closed-form ground truth. The full API is
described in [`python/README.md`](python/README.md).

## Build

The default build is pure std: every external dependency is optional and off, so the
simulation core compiles and its tests run offline with no system libraries.

```bash
cargo test
```

(About 115 tests. `--features cli` adds the I/O tests; `--features kspace` runs the k-space
oracle over the FFT/NUFFT path.)

The binaries need NIfTI and TRX I/O, which lives behind the `cli` feature:

```bash
cargo build --release --features cli,par,kspace
```

`cli` pulls in `clap` and the I/O stack; `par` turns on rayon (streamline groups in the signal
stage, volumes in the acquisition stage); `kspace` is the FFT/NUFFT k-space path. The first
build with `cli`/`io` compiles the I/O stack from source (`trx-rs` → `itk-transforms-rs` →
`hdf5-metno-src`, which builds HDF5), so it needs network access and `cmake` and takes a while.
`trx-rs` and `odx-rs` are git-pinned in [`Cargo.toml`](Cargo.toml); no sibling checkout is
needed. The crate is a cargo workspace whose second member, [`python/`](python/), is the pyo3
extension behind the `trxscan` PyPI package. It depends on the std-only core with `kspace,par`,
so the wheel has no C dependencies.

## Command line

Three binaries, all with named arguments; run any with `--help` for the full list:
`trxscan` (the simulator), `trxscan-microstructure` (ground-truth scalar maps) and
`trxscan-benchmark` (the Gibbs-ringing factor grid that `scripts/acceptance.py` scores).

### Preparing the grids

The object is simulated on a grid finer than the acquisition matrix (`--oversample`, default 2),
which is where Gibbs ringing comes from. [`scripts/prepare_acquisition_grid.py`](scripts/prepare_acquisition_grid.py)
resamples anatomical-grid probsegs (and the fieldmap) onto both grids:

```bash
python3 scripts/prepare_acquisition_grid.py \
  --anat-dir sub-01/anat --prefix sub-01_space-ACPC --voxel 1.7 --oversample 2 --out grid
# -> grid/{wm,gm,csf,mask,fmap_hz}.nii.gz and grid/sim/{...} at 0.85 mm in-plane
```

### `trxscan`: simulate a 4D DWI

```bash
trxscan \
  --wm grid/wm.nii.gz --gm grid/gm.nii.gz --csf grid/csf.nii.gz --mask grid/mask.nii.gz \
  --sim-wm grid/sim/wm.nii.gz --sim-gm grid/sim/gm.nii.gz --sim-csf grid/sim/csf.nii.gz \
  --sim-mask grid/sim/mask.nii.gz --sim-fmap grid/sim/fmap_hz.nii.gz \
  --streamlines tracts.trx --weights sift2_weights --bval dwi.bval --bvec dwi.bvec \
  --fsl-orientation --out out/sub-01_dir-AP_run-01
```

Writes `<out>_part-mag_dwi.nii.gz`, `_part-phase_dwi.nii.gz`, `_dwi.bval`, `_dwi.bvec`, and JSON
sidecars (`PhaseEncodingDirection`, `TotalReadoutTime`, `EchoTime`, …). Options, by artifact
(defaults in brackets):

| Flag | Default | Meaning |
|---|---|---|
| `--oversample <N>` | 2 | simulation-grid refinement (1 = object on the reconstruction matrix: no Gibbs ringing; takes `--fmap` instead of `--sim-fmap`) |
| `--phase-model <m>` | hbcd | object phase: `hbcd` (calibrated background ramp + per-shot diffusion phase) or `none` |
| `--reverse-pe` | off | flip phase-encode polarity (the AP/PA pair for topup/DRBUDDI) |
| `--fsl-orientation` | off | write radiological LAS with FSL-convention bvecs, as dcm2niix would |
| `--pf-mode <m>` | scanner | partial-Fourier rule: `scanner` (skips the first lines of the train, so the centre is reached sooner), `contiguous` (drops the last lines, timing unchanged) or `fiberfox` |
| `--noise <var>` | 0 | complex k-space noise variance → Rician magnitude |
| `--noise-map <nii>` | — | per-voxel noise SD on the acquisition grid (image-space, shared by mag and phase); writes `<out>_desc-noise_sigma.nii.gz` as ground truth |
| `--eddy <s>` / `--eddy-quad <s>` | 0 | linear / quadratic eddy-current geometric distortion (b0 exempt) |
| `--eddy-phase <s>` | 0 | direction- and b-dependent eddy phase ramp on the reconstructed phase (about 1.7e-5 matches a 3T HBCD scan) |
| `--accel <R>` | 1 | GRAPPA acceleration (24 ACS lines) |
| `--coils <n>` | 1 | receiver coils (ring-arranged sensitivities, Roemer combine) |
| `--motion <tsv>` | — | qsiprep/eddy confounds TSV (`trans_x/y/z` mm, `rot_x/y/z` rad); per-volume re-simulation |
| `--mb <f>` / `--dropout-rate <p>` | 1 / 0 | multiband factor and per-volume within-volume dropout probability; writes `<out>_desc-dropout_slices.tsv` |
| `--gnl <preset\|file>` | — | gradient nonlinearity: `whole-body-80`, `connectom-300` or a Siemens `.grad`; `--gnl-scale`, `--isocenter x,y,z`, `--gnl-no-warp`, `--gnl-no-encoding`, `--gnl-no-jacobian-modulation`, `--gnl-info` |
| `--gre-out <prefix>` | — | also synthesize a dual-echo GRE fieldmap; `--gre-snr` [50], `--gre-res <mm>`, `--gre-output phasediff\|phase`, `--gre-rx-phase`, `--gre-snr-vol-exp`, `--gre-b0field` [b0gre], `--gre-tr` [0.5 s] / `--gre-flip` [60°] / `--gre-t1` [830,1330,4000 ms] / `--gre-pd` [0.7,0.85,1.0] (spoiled-GRE steady state × proton density: the nearly flat brain of a real fieldmap magnitude), `--gre-bias` [0.3] (periphery-bright receive profile), `--gre-no-ringing` (box averaging instead of the Fourier-truncation Gibbs ringing). The Python `Gre(head=True)` also adds the non-brain head from the phantom's T1w. |
| `--truth-peaks` | off | write up to three ground-truth fibre peaks per voxel, `<out>_desc-truth_peaks.nii.gz` (9 volumes) |
| `--weights <spec>` | — | SIFT2 weights: a TRX `dps` name, or an MRtrix `tcksift2` text file |
| `--kappa <κ>` | — | Watson dispersion applied to the orientation histogram |
| `--params <preset>` | neonatal | compartment preset: `neonatal`, `adult`, or `infant` |
| `--myelin <nii>` | — | per-voxel myelination (0..1); interpolates WM toward the `adult` endpoint |
| `--tissue-s0 wm,gm,csf` / `--diff-scale wm,gm,csf` | 1,1,1 | per-compartment b0 amplitude / diffusivity factors, for matching a real scan's levels (signal only; the ground truth keeps the preset) |
| `--seed <n>` / `--subsample <N>` | 0 / — | noise/dropout realization; keep N streamlines sampled ∝ weight |

With `--gnl` it also writes `<out>_desc-gnl_coeff.grad` (for `qsiprep --gradient-file`),
`_desc-gnl_disp` (φ(r)−r: warps points/streamlines true → apparent), `_desc-gnl_invdisp`
(φ⁻¹(x)−x: pulls images apparent ← true) and `_desc-gnl_graddev` (HCP layout, identity
included), all RAS mm on the written grid. With `--gre-out` the fieldmap files carry
`B0FieldIdentifier` and the DWI the matching `B0FieldSource`, so qsiprep links them.

Passing `--motion` switches the signal stage to the per-volume path: for each volume the
streamlines and tissue maps are rigidly transformed by that volume's pose and the signal is
re-rasterized and re-simulated, so fibre–gradient angles change as they would in a moving head
(roughly n_volumes times the rasterization cost, parallel over volumes with `par`).

### `trxscan-microstructure`: analytic ground-truth maps

No acquisition stage and no noise: the maps describe the mixture the simulator generates its
signal from.

```bash
trxscan-microstructure \
  --wm wm.nii.gz --gm gm.nii.gz --csf csf.nii.gz --mask mask.nii.gz \
  --streamlines tracts.trx --out out/sub-01 \
  --kappa 15 --weights sift2_weights --big-delta 0.030 --small-delta 0.010
```

`--subsample <N> --seed <n>` work here too and select the same subset as `trxscan` given the
same values, so ground truth and simulated data describe the same phantom. Sampling is
probability-proportional-to-weight rather than top-N by weight (which would strip the
over-tracked bundles); survivors get uniform weights so the density stays unbiased.

Writes 27 maps as `<out>_<name>.nii.gz`: `fa md rd ad ak rk mk mkt kfa micro_fa coherence k_bulk
k_shear rtop rtap rtpp msd qiv ng ngpar ngperp pa gfa qa icvf odi isovf`. `--big-delta` /
`--small-delta` (seconds) set `tau = Δ − δ/3` so the MAP-MRI maps come out in physical units;
omit them for dipy's normalized-units default (voxel-to-voxel contrast only).

### Library entry points

The binaries are thin: the simulation is `kspace::simulate_acquisition(&SimulationInput, &Acquisition)`
on the compartments from `compartments::generate_mixture` → `signal_from_mixture_gnl`, the GRE
is `gre::synthesize(&GreObject, &GreParams)`, and `Acquisition::hbcd(ny)` is the CLI protocol.
This is the surface the Python package wraps.

## Conventions

- **The b-value lives in the gradient norm.** Signal models take `b_value = b_max` and a gradient
  scaled to `unit_bvec · sqrt(b_i / b_max)`; feed them `GradientScheme::fiberfox_gradients()`.
- **4D layout is voxel-major interleaved:** `(x + nx*(y + ny*z)) * ngrad + g`.
- **Compartments stay separate through the signal stage** so k-space can apply per-compartment T2.
- **Streamlines and tissue maps must share a world (RAS mm) frame.** Nothing re-registers them.
- **A positive off-resonance field displaces signal toward −j on the forward scan** (the sign
  convention inherited from Fiberfox); the written `PhaseEncodingDirection` always describes the
  baked-in distortion, and `--fsl-orientation` makes it the `j` that FSL/dcm2niix expect.
  [`tools/roundtrip_fieldmap_test.py`](tools/roundtrip_fieldmap_test.py) checks this against the
  FSL convention end to end.

## Sibling crates

| Crate | Used for |
|---|---|
| [`trx-rs`](https://github.com/tee-ar-ex/trx-rs) (git-pinned) | load TRX/TRK/TCK/VTK streamlines and per-streamline SIFT2 weights from TRX `dps` |
| [`odx-rs`](https://github.com/PennLINC/odx-rs) (git-pinned) | *planned:* SH/sphere math and ground-truth ODX export (behind the `odx` feature; not wired up yet) |
| [`nifti`](https://crates.io/crates/nifti) 0.17 | 3D/4D NIfTI read and write, affine |

## Origins

TRXScan began as a port of the simulation code in MITK-Diffusion's Fiberfox. The parts that
came from there are the rasterizer, the Stick/Tensor/Ball signal models with the
b-in-gradient-norm encoding, the per-slice k-space model with its distortion, relaxation, eddy,
ghost, partial-Fourier, spike and noise terms, and the between-volume motion generators. Those
modules carry `file:line` references into the MITK source in their doc comments, and
[`docs/FINDINGS.md`](docs/FINDINGS.md) describes how that code decomposes.

The rest was written for TRXScan: the orientation-histogram signal stage with SIFT2 weights and
dispersion, multiband motion and dropout, GRAPPA and the multi-coil combine, complex output and
the object-phase model, scanner-style partial Fourier, the FFT/NUFFT k-space path, gradient
nonlinearity, the GRE fieldmap, the microstructure ground truth, and the Python package. The
design notes in [`docs/`](docs/) cover multiband motion and GRAPPA (`FEATURES.md`), gradient
nonlinearity (`GNL.md`) and the ground-truth microstructure scalars (`FORCE.md`).

The microstructure closed forms are a port of DIPY's `_force_moments` and are tested against it.

## Citing

If you use TRXScan, please also cite Fiberfox, whose simulation model it builds on:

> Neher PF, Laun FB, Stieltjes B, Maier-Hein KH. *Fiberfox: Facilitating the creation of realistic
> white matter software phantoms.* Magnetic Resonance in Medicine. 2014;72(5):1460–1470.

The ground-truth microstructure scalars come from the FORCE closed forms in
[DIPY](https://dipy.org) (Garyfallidis E, et al. *Dipy, a library for the analysis of diffusion
MRI data.* Frontiers in Neuroinformatics. 2014;8:8).

## License

TRXScan's own code is dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE),
at your option. The ported portions remain under their upstreams' BSD-3-Clause terms, whose notices
are retained verbatim in [`LICENSE-MITK`](LICENSE-MITK) (MITK-Diffusion / Fiberfox) and
[`LICENSE-DIPY`](LICENSE-DIPY) (DIPY).
