# TRXScan

A from-scratch, headless **diffusion-MRI simulator in Rust**. It reimplements the simulation math
of [MITK Fiberfox](https://github.com/MIC-DKFZ/MITK-Diffusion) outside the MITK/ITK/VTK/Qt stack,
and extends it with three things Fiberfox can't do: **within-volume (multiband) motion**,
**GRAPPA**, and **complex (magnitude + phase) output**.

**Input:** a tractogram (TRX/TRK/TCK/VTK) + tissue volume-fraction maps + an FSL gradient scheme
(+ a fieldmap).
**Output:** a BIDS complex 4D DWI (`part-mag` / `part-phase`) + `.bval`/`.bvec`, with realistic
acquisition artifacts and motion — plus dropped-slice ground truth when multiband motion is on.

```
streamlines + tissue maps + scheme
        │  rasterize (exact 3D-DDA, path lengths)
        ▼
   per-voxel, per-gradient clean signal   ── Stick / Tensor / Ball, kept per-compartment
        │  per-slice k-space (direct DFT)
        ▼
   distortion · T2* · eddy · Nyquist ghost · partial Fourier · Gibbs · spikes
   · multi-coil · GRAPPA · k-space noise
        │
        ▼            (+ motion: per volume, or per multiband shot)
   BIDS complex 4D DWI  (mag + phase, .bval/.bvec/.json)
```

A companion tool derives **analytic ground-truth microstructure maps** from the same per-voxel
mixture the simulator builds its signal from, so the answer key and the DWI can't disagree.

## Why this exists

Fiberfox is a complete, GUI-driven phantom simulator built into the full MITK-Diffusion stack
(ITK + VTK + DCMTK + CTK + Qt). For headless, scriptable, large-batch simulation we wanted just its
simulation *math*, which — lifted out of that infrastructure — is about 2–2.5k lines, running as a
small Rust crate with no system dependencies. TRXScan is that reimplementation, plus a few
extensions (within-volume/multiband motion, GRAPPA, complex output) that Fiberfox's volume-level
acquisition model doesn't express. It's a complement to Fiberfox, not a replacement — Fiberfox
remains the reference we validate against.

## Capabilities

| Area | What it does |
|---|---|
| Gradient scheme | FSL `.bval`/`.bvec`, shell detection, Fiberfox gradient encoding |
| Rasterization | exact 3D-DDA segment → (voxel, path length) on the target (oblique) grid |
| Signal models | Stick, Tensor (cylindrically symmetric), Ball |
| Signal stage | per-compartment clean signal; per-voxel orientation-mixture path with SIFT2 weights + Watson κ; motion-aware re-simulation |
| Acquisition stage (k-space) | EPI distortion, T2*, eddy (linear + quadratic), Nyquist ghost, partial Fourier, Gibbs ringing, spikes, multi-coil Roemer combine, GRAPPA, k-space noise, complex output |
| Motion | random / linear / trajectory poses, qsiprep-TSV traces, multiband schedule, within-volume dropout + ground truth |
| Microstructure ground truth | FORCE closed forms — DTI/DKI (exact MK via Carlson), QTI (µFA, k_bulk/k_shear), MAP-MRI, NG/PA, GFA/QA, NODDI-style ICVF/ODI/ISOVF: 27 maps, dipy-validated |
| I/O | TRX/TRK/TCK/VTK load (+ SIFT2 weights from TRX `dps`), NIfTI tissue maps, BIDS complex 4D write, scalar-map write |

## Scope & limitations

- **k-space is an exact direct DFT, not an FFT** — O(N³) per slice, deliberately, so there is no
  FFT-convention ambiguity while validating against Fiberfox. A faster time-segmented FFT path is
  designed but unwritten.
- **The acquisition protocol is not yet configurable from a file.** Inputs are `clap` flags, but
  the protocol itself (TE, partial Fourier, ghosting, …) is a hard-coded literal in
  [`src/bin/trxscan.rs`](src/bin/trxscan.rs) (an HBCD-like protocol). A TOML config is planned
  (the `config` feature).
- **Motion mode uses the per-segment signal stage**, so SIFT2 weights and the Watson kernel are ignored
  when a `--motion` trace is given; the default (no-motion) path and `trxscan-microstructure` share
  the orientation-mixture path.
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

(49 tests. Building with `--features cli` adds the two I/O tests → 51.)

The binaries need NIfTI + TRX I/O, which lives behind the `cli` feature:

```bash
cargo build --release --features cli,par
```

`cli` pulls in `clap` and the I/O stack; `par` turns on rayon (streamline groups in the signal
stage, volumes in the acquisition stage). The **first** build compiles the I/O stack from source — `trx-rs` →
`itk-transforms-rs` (a git dependency) → `hdf5-metno-src`, which builds HDF5 — so it needs network
access and `cmake` and takes a while. Lay the sibling crates out next to this one:

```
~/projects/
  ├── TRXScan/      (this crate)
  ├── odx-rs/
  └── rust/trx-rs/
```

The path dependencies in [`Cargo.toml`](Cargo.toml) point at those locations.

## Usage

Both binaries take named arguments; run either with `--help` for the full list.

### `trxscan` — simulate a 4D DWI

```bash
trxscan \
  --wm wm.nii.gz --gm gm.nii.gz --csf csf.nii.gz --mask mask.nii.gz \
  --streamlines tracts.trx --bval dwi.bval --bvec dwi.bvec --fmap fmap_hz.nii.gz \
  --out out/sub-01_dir-AP_run-01
```

Writes `<out>_part-mag_dwi.nii.gz`, `_part-phase_dwi.nii.gz`, `_dwi.bval`, `_dwi.bvec`, and JSON
sidecars. Options add realism (defaults in brackets):

| Flag | Default | Meaning |
|---|---|---|
| `--reverse-pe` | off | flip phase-encode polarity (the AP/PA pair for topup/DRBUDDI) |
| `--noise <var>` | 0 | complex k-space noise variance → Rician magnitude |
| `--eddy <s>` / `--eddy-quad <s>` | 0 | linear / quadratic eddy-current strength (b0 exempt) |
| `--accel <R>` | 1 | GRAPPA acceleration |
| `--coils <n>` | 1 | receiver coils (ring-arranged sensitivities) |
| `--motion <tsv>` | — | qsiprep/eddy confounds TSV (`trans_x/y/z` mm, `rot_x/y/z` rad) |
| `--mb <f>` | 1 | multiband factor |
| `--dropout-rate <p>` | 0 | per-DWI-volume probability of a within-volume dropout event |
| `--weights <spec>` | — | SIFT2 weights: a TRX `dps` name, or an MRtrix `tcksift2` text file |
| `--kappa <κ>` | — | Watson dispersion applied to the orientation histogram |
| `--params <preset>` | neonatal | compartment preset: `neonatal`, `adult`, or `infant` |
| `--myelin <nii>` | — | per-voxel myelination (0..1); lerps WM toward the `adult` endpoint |
| `--seed <n>` | 0 | noise/dropout realization + `--subsample` draw; 0 reproduces the historical output |
| `--subsample <N>` | — | keep N streamlines, sampled ∝ SIFT2 weight (survivors re-weighted uniform) |

With `--mb > 1` and `--dropout-rate > 0` it also writes `<out>_desc-dropout_slices.tsv` — the
dropped-shot ground truth for scoring `eddy --repol` or SHORELine outlier detection.

Passing `--motion` switches the signal stage to the **faithful** path: for each volume the streamlines and
tissue maps are rigidly transformed by that volume's pose and the signal is re-rasterized and
re-simulated, so fiber–gradient angles change correctly (roughly ×n_volumes the rasterization,
parallel over volumes with `par`).

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

## Conventions worth knowing

- **The b-value lives in the gradient norm.** Signal models take `b_value = b_max` and a gradient
  scaled to `unit_bvec · sqrt(b_i / b_max)`; feed them `GradientScheme::fiberfox_gradients()`.
- **4D layout is voxel-major interleaved:** `(x + nx*(y + ny*z)) * ngrad + g`.
- **Compartments stay separate through the signal stage** so k-space can apply per-compartment T2.
- **Streamlines and tissue maps must share a world (RAS mm) frame.** Nothing re-registers for you.

## Sibling crates

| Crate | Used for |
|---|---|
| [`trx-rs`](https://github.com/tee-ar-ex/trx-rs) (`../rust/trx-rs`) | load TRX/TRK/TCK/VTK streamlines + per-streamline SIFT2 weights from TRX `dps` |
| [`odx-rs`](https://github.com/PennLINC/odx-rs) (`../odx-rs`) | *planned:* SH/sphere math and ground-truth ODX export (behind the `odx` feature; not wired up yet) |
| [`nifti`](https://crates.io/crates/nifti) 0.17 | 3D/4D NIfTI read + write, affine |

## Provenance

TRXScan is a port: nearly every module's doc comment carries a `file:line` anchor into the
MITK-Diffusion source it reimplements — read that anchor before changing the physics.

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
