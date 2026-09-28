# TRXScan

A from-scratch, headless **diffusion-MRI simulator in Rust**. It reimplements the simulation math
of [MITK Fiberfox](https://github.com/MIC-DKFZ/MITK-Diffusion) outside the MITK/ITK/VTK/Qt stack,
and extends it with what Fiberfox can't do: **within-volume (multiband) motion**, **GRAPPA**,
**complex (magnitude + phase) output**, **gradient nonlinearity**, and a **synthetic GRE fieldmap**
of the same object.

**Input:** a tractogram (TRX/TRK/TCK/VTK) + tissue volume-fraction maps + an FSL gradient scheme
(+ a fieldmap).
**Output:** a BIDS complex 4D DWI (`part-mag` / `part-phase`) + `.bval`/`.bvec`, with realistic
acquisition artifacts and motion — plus the ground truth for each artifact that has one (dropped
slices, noise level, gradient-nonlinearity field, fibre orientations).

```
streamlines + tissue maps + scheme
        │  rasterize (exact 3D-DDA, path lengths) → per-voxel orientation histogram
        ▼
   per-voxel, per-gradient clean signal   ── Stick / Tensor / Ball, kept per-compartment
        │  (+ gradient nonlinearity: per-voxel Jᵀg and the spatial warp)
        │  per-slice k-space (exact sum, FFT/NUFFT-accelerated)
        ▼
   distortion · T2* · eddy · Nyquist ghost · partial Fourier · Gibbs · spikes
   · multi-coil · GRAPPA · k-space noise · object phase
        │
        ▼            (+ motion: per volume, or per multiband shot)
   BIDS complex 4D DWI  (mag + phase, .bval/.bvec/.json)  + ground truth
```

A companion tool derives **analytic ground-truth microstructure maps** from the same per-voxel
mixture the simulator builds its signal from, so the answer key and the DWI can't disagree.

## Why this exists

Fiberfox is a complete, GUI-driven phantom simulator built into the full MITK-Diffusion stack
(ITK + VTK + DCMTK + CTK + Qt). For headless, scriptable, large-batch simulation we wanted just its
simulation *math*, which — lifted out of that infrastructure — is about 2–2.5k lines, running as a
small Rust crate with no system dependencies. TRXScan is that reimplementation, plus the
extensions Fiberfox's volume-level acquisition model doesn't express. It's a complement to
Fiberfox, not a replacement — Fiberfox remains the reference we validate against.

## Capabilities

| Area | What it does |
|---|---|
| Gradient scheme | FSL `.bval`/`.bvec`, shell detection, Fiberfox gradient encoding |
| Rasterization | exact 3D-DDA segment → (voxel, path length) on the target (oblique) grid |
| Signal models | Stick, Tensor (cylindrically symmetric), Ball; GM as a two-ball (free + soma) mixture |
| Signal stage | per-compartment clean signal from a per-voxel orientation histogram with SIFT2 weights + Watson κ; per-voxel myelination; motion-aware re-simulation |
| Acquisition stage (k-space) | EPI distortion, T2*, eddy (linear + quadratic geometric, plus an object-phase ramp), Nyquist ghost, partial Fourier, intrinsic Gibbs ringing (oversampled object), spikes, multi-coil Roemer combine, GRAPPA, k-space noise, per-voxel noise map, calibrated object phase, complex output |
| Gradient nonlinearity | synthetic whole-body / Connectom coefficient sets (or a Siemens `.grad` file): per-voxel diffusion-encoding deviation `Jᵀg` **and** the spatial warp with Jacobian modulation; writes the coefficient file, forward/inverse displacement fields and graddev |
| GRE fieldmap | dual-echo `magnitude1/2` + `phasediff` (or `phase1/2`) from the same off-resonance field, Siemens integer phase, optional coarser resolution with intravoxel dephasing, `B0FieldIdentifier` linkage |
| Motion | random / linear / trajectory poses, qsiprep-TSV traces, multiband schedule, within-volume dropout + ground truth |
| Ground truth | dropped-slice TSV, noise-SD map, GNL field + graddev, up to three fibre peaks per voxel, and 27 FORCE microstructure maps (DTI/DKI, QTI, MAP-MRI, NG/PA, GFA/QA, NODDI-style; dipy-validated) |
| I/O | TRX/TRK/TCK/VTK load (+ SIFT2 weights from TRX `dps`), NIfTI tissue maps, BIDS complex 4D write in the native or FSL/dcm2niix (LAS) orientation with FSL-convention bvecs |

## Scope & limitations

- **k-space is the exact per-line sum, not an approximation of it.** There is no FFT-convention
  ambiguity while validating against Fiberfox. It is *organised* for speed (static factors hoisted,
  the affine-in-ky phase advanced by memoised rotors), and the `kspace` feature runs the same sum
  through `rustfft` plus a type-1 NUFFT for the fieldmap term — identical numbers, ~5× faster. The
  literal sum survives as a test oracle.
- **The protocol is HBCD-like and set in code.** `Acquisition::hbcd` (TE 88 ms, 6/8 partial
  Fourier, 24 ACS lines, a subtle Nyquist ghost) is the library default; the flags override its
  artifact knobs. There is no config file yet (the `config` feature is a stub).
- **Motion mode uses the per-segment signal stage**, so with a `--motion` trace SIFT2 weights and
  the Watson kernel are ignored, and `--myelin`, `--truth-peaks` and `--gnl` are refused. The
  default (no-motion) path and `trxscan-microstructure` share the orientation-mixture path.
- **`Tensor` handles the cylindrically-symmetric case only** (`d2 == d3`).
- **The microstructure closed forms are validated against dipy** (30 checked-in fixture mixtures,
  ≤1e-8). The k-space stage has physics-behaviour and self-consistency tests, but no oracle diff
  against Fiberfox yet.

## Build

The default build is **pure std**: every external dependency is optional and off, so the
simulation core compiles and its tests run offline with no system libraries.

```bash
cargo test
```

(~115 tests. `--features cli` adds the I/O tests; `--features kspace` runs the k-space oracle over
the FFT/NUFFT path.)

The binaries need NIfTI + TRX I/O, which lives behind the `cli` feature:

```bash
cargo build --release --features cli,par,kspace
```

`cli` pulls in `clap` and the I/O stack; `par` turns on rayon (streamline groups in the signal
stage, volumes in the acquisition stage); `kspace` is the fast k-space path. The **first** build
with `cli`/`io` compiles the I/O stack from source — `trx-rs` → `itk-transforms-rs` →
`hdf5-metno-src`, which builds HDF5 — so it needs network access and `cmake` and takes a while.
`trx-rs` and `odx-rs` are git-pinned in [`Cargo.toml`](Cargo.toml); no sibling checkout is needed.
The crate is a cargo workspace whose second member, [`python/`](python/), is the pyo3 extension
behind the `trxscan` PyPI package (it depends on the std-only core with `kspace,par`, so the wheel
has no C dependencies; see `python/README.md`).

## Python

The [`trxscan`](python/README.md) package (`pip install trxscan`) wraps the same simulator for
notebooks: `Phantom.load("sub-60501").simulate(gtab, Protocol.HBCD, Artifacts(noise=2e-4),
slices=40)` returns nibabel images, per-coil k-space and readout timing for one slice in a few
seconds, bit-identical to the CLI's output for that slice; `Voxel(...)` gives dipy.sims-style
single-voxel signals with the closed-form ground truth. See `python/README.md`.

## Usage

Three binaries, all with named arguments; run any with `--help` for the full list:
`trxscan` (the simulator), `trxscan-microstructure` (ground-truth scalar maps) and
`trxscan-benchmark` (the scoreable Gibbs-ringing factor grid; `scripts/acceptance.py` scores it).

### Preparing the grids

The object is simulated on a grid finer than the acquisition matrix (`--oversample`, default 2),
which is what makes Gibbs ringing intrinsic. [`scripts/prepare_acquisition_grid.py`](scripts/prepare_acquisition_grid.py)
resamples anatomical-grid probsegs (+ fieldmap) onto both grids:

```bash
python3 scripts/prepare_acquisition_grid.py \
  --anat-dir sub-01/anat --prefix sub-01_space-ACPC --voxel 1.7 --oversample 2 --out grid
# -> grid/{wm,gm,csf,mask,fmap_hz}.nii.gz and grid/sim/{...} at 0.85 mm in-plane
```

### `trxscan` — simulate a 4D DWI

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
| `--pf-mode <m>` | scanner | partial-Fourier rule: `scanner` (skips the first lines of the train, so the centre is reached sooner), `contiguous` (legacy: drops the last lines, timing unchanged) or `fiberfox` |
| `--noise <var>` | 0 | complex k-space noise variance → Rician magnitude |
| `--noise-map <nii>` | — | per-voxel noise SD on the acquisition grid (image-space, shared by mag and phase); writes `<out>_desc-noise_sigma.nii.gz` as ground truth |
| `--eddy <s>` / `--eddy-quad <s>` | 0 | linear / quadratic eddy-current geometric distortion (b0 exempt) |
| `--eddy-phase <s>` | 0 | direction- and b-dependent eddy phase ramp on the reconstructed phase (~1.7e-5 matches a 3T HBCD scan) |
| `--accel <R>` | 1 | GRAPPA acceleration (24 ACS lines) |
| `--coils <n>` | 1 | receiver coils (ring-arranged sensitivities, Roemer combine) |
| `--motion <tsv>` | — | qsiprep/eddy confounds TSV (`trans_x/y/z` mm, `rot_x/y/z` rad); faithful per-volume re-simulation |
| `--mb <f>` / `--dropout-rate <p>` | 1 / 0 | multiband factor and per-volume within-volume dropout probability; writes `<out>_desc-dropout_slices.tsv` |
| `--gnl <preset\|file>` | — | gradient nonlinearity: `whole-body-80`, `connectom-300` or a Siemens `.grad`; `--gnl-scale`, `--isocenter x,y,z`, `--gnl-no-warp`, `--gnl-no-encoding`, `--gnl-no-jacobian-modulation`, `--gnl-info` |
| `--gre-out <prefix>` | — | also synthesize a dual-echo GRE fieldmap; `--gre-snr` [50], `--gre-res <mm>`, `--gre-output phasediff\|phase`, `--gre-rx-phase`, `--gre-snr-vol-exp`, `--gre-b0field` [b0gre] |
| `--truth-peaks` | off | write up to three ground-truth fibre peaks per voxel, `<out>_desc-truth_peaks.nii.gz` (9 volumes) |
| `--weights <spec>` | — | SIFT2 weights: a TRX `dps` name, or an MRtrix `tcksift2` text file |
| `--kappa <κ>` | — | Watson dispersion applied to the orientation histogram |
| `--params <preset>` | neonatal | compartment preset: `neonatal`, `adult`, or `infant` |
| `--myelin <nii>` | — | per-voxel myelination (0..1); lerps WM toward the `adult` endpoint |
| `--tissue-s0 wm,gm,csf` / `--diff-scale wm,gm,csf` | 1,1,1 | per-compartment b0 amplitude / diffusivity factors, for matching a real scan's levels (signal only; the ground truth keeps the preset) |
| `--seed <n>` / `--subsample <N>` | 0 / — | noise/dropout realization; keep N streamlines sampled ∝ weight |

With `--gnl` it also writes `<out>_desc-gnl_coeff.grad` (for `qsiprep --gradient-file`),
`_desc-gnl_disp` (φ(r)−r: warps points/streamlines true → apparent), `_desc-gnl_invdisp`
(φ⁻¹(x)−x: pulls images apparent ← true) and `_desc-gnl_graddev` (HCP layout, identity
included), all RAS mm on the written grid. With `--gre-out` the fieldmap files carry
`B0FieldIdentifier` and the DWI the matching `B0FieldSource`, so qsiprep links them.

Passing `--motion` switches the signal stage to the **faithful** path: for each volume the
streamlines and tissue maps are rigidly transformed by that volume's pose and the signal is
re-rasterized and re-simulated, so fiber–gradient angles change correctly (roughly ×n_volumes the
rasterization, parallel over volumes with `par`).

### `trxscan-microstructure` — analytic ground-truth maps

No acquisition, no noise — the answer key for the mixture the simulator itself uses.

```bash
trxscan-microstructure \
  --wm wm.nii.gz --gm gm.nii.gz --csf csf.nii.gz --mask mask.nii.gz \
  --streamlines tracts.trx --out out/sub-01 \
  --kappa 15 --weights sift2_weights --big-delta 0.030 --small-delta 0.010
```

`--subsample <N> --seed <n>` work here too and select the **identical** subset as `trxscan`
given the same values, so ground truth and simulated data always describe the same phantom.
Sampling is probability-proportional-to-weight (never top-N by weight, which would gut the
over-tracked bundles); survivors get uniform weights so the density stays unbiased.

Writes 27 maps as `<out>_<name>.nii.gz` — `fa md rd ad ak rk mk mkt kfa micro_fa coherence k_bulk
k_shear rtop rtap rtpp msd qiv ng ngpar ngperp pa gfa qa icvf odi isovf`. `--big-delta` /
`--small-delta` (seconds) set `tau = Δ − δ/3` so the MAP-MRI maps come out in physical units; omit
them for dipy's normalized-units default (voxel-to-voxel contrast only).

### Library entry points

The binaries are thin: the simulation is `kspace::simulate_acquisition(&SimulationInput, &Acquisition)`
on the compartments from `compartments::generate_mixture` → `signal_from_mixture_gnl`, the GRE
is `gre::synthesize(&GreObject, &GreParams)`, and `Acquisition::hbcd(ny)` is the protocol —
the surface a Python binding wraps.

## Conventions worth knowing

- **The b-value lives in the gradient norm.** Signal models take `b_value = b_max` and a gradient
  scaled to `unit_bvec · sqrt(b_i / b_max)`; feed them `GradientScheme::fiberfox_gradients()`.
- **4D layout is voxel-major interleaved:** `(x + nx*(y + ny*z)) * ngrad + g`.
- **Compartments stay separate through the signal stage** so k-space can apply per-compartment T2.
- **Streamlines and tissue maps must share a world (RAS mm) frame.** Nothing re-registers for you.
- **A positive off-resonance field displaces signal toward −j on the forward scan** (the Fiberfox
  k-space convention); the written `PhaseEncodingDirection` always describes the baked-in
  distortion, and `--fsl-orientation` makes it the `j` that FSL/dcm2niix expect.
  [`tools/roundtrip_fieldmap_test.py`](tools/roundtrip_fieldmap_test.py) pins this against the
  FSL convention end-to-end.

## Sibling crates

| Crate | Used for |
|---|---|
| [`trx-rs`](https://github.com/tee-ar-ex/trx-rs) (git-pinned) | load TRX/TRK/TCK/VTK streamlines + per-streamline SIFT2 weights from TRX `dps` |
| [`odx-rs`](https://github.com/PennLINC/odx-rs) (git-pinned) | *planned:* SH/sphere math and ground-truth ODX export (behind the `odx` feature; not wired up yet) |
| [`nifti`](https://crates.io/crates/nifti) 0.17 | 3D/4D NIfTI read + write, affine |

## Provenance

TRXScan is a port: nearly every module's doc comment carries a `file:line` anchor into the
MITK-Diffusion source it reimplements — read that anchor before changing the physics. The design
notes in [`docs/`](docs/) cover the Fiberfox decomposition (`FINDINGS.md`), the multiband-motion
and GRAPPA extensions (`FEATURES.md`), gradient nonlinearity (`GNL.md`) and the ground-truth
microstructure scalars (`FORCE.md`).

## Acknowledgments

TRXScan stands on the shoulders of **[MITK Fiberfox](https://github.com/MIC-DKFZ/MITK-Diffusion)**,
the diffusion-MRI phantom simulator developed by the German Cancer Research Center (DKFZ), Division
of Medical Image Computing. Fiberfox is the reference implementation this project reimplements and
the gold standard we validate against; TRXScan's physics is a faithful, function-by-function port of
it. We're deeply grateful for that work — if you use TRXScan, please also credit Fiberfox:

> Neher PF, Laun FB, Stieltjes B, Maier-Hein KH. *Fiberfox: Facilitating the creation of realistic
> white matter software phantoms.* Magnetic Resonance in Medicine. 2014;72(5):1460–1470.

The closed-form microstructure scalars are ported from **[DIPY](https://dipy.org)**, which also
serves as their validation oracle — thanks to the DIPY developers.

## License

TRXScan's own code is dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE),
at your option. The ported portions remain under their upstreams' BSD-3-Clause terms, whose notices
are retained verbatim in [`LICENSE-MITK`](LICENSE-MITK) (MITK-Diffusion / Fiberfox) and
[`LICENSE-DIPY`](LICENSE-DIPY) (DIPY).
