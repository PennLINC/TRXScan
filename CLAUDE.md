# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What this is

A headless diffusion-MRI simulator in Rust with a Python package. The signal and k-space
models were ported from MITK Fiberfox; the acquisition model has been extended since with
within-volume/multiband motion, GRAPPA, complex magnitude+phase output, scanner-style partial
Fourier, gradient nonlinearity, a synthetic GRE fieldmap and analytic microstructure ground
truth. Input: tractogram + tissue volume-fraction maps + FSL bval/bvec (+ fieldmap). Output:
BIDS complex 4D DWI (`part-mag`/`part-phase`) + bval/bvec (+ ground-truth maps). A companion
binary emits the microstructure maps from the same per-voxel mixture. The Python package
(`python/`) wraps the library surface (structs in, structs out) for a "dMRI and its artifacts"
Jupyter book, so keep that surface the thing that grows, not the CLI.

## Commands

```bash
cargo test
```

The default build is **feature-less and pure std** — ~115 unit tests, no network, no system libs
(`--features cli` — which implies `io` — adds the io tests; `--features kspace` runs the same
k-space oracle over the FFT/NUFFT path). Every module except `io`/`config` compiles here, and
the tests live as `#[cfg(test)] mod tests` at the bottom of each module; `tests/` holds the dipy
fixture data and one CLI-level integration test (`benchmark_cli.rs`), nothing else. Keep it that
way.

The `microstructure` fixture test reads `tests/fixtures/force_moments.txt`, regenerated (only when
the closed forms or sampling change) with:

```bash
python tools/gen_force_fixtures.py   # run inside the crash_force dipy-fork env
```

That conda env has the crash_force dipy fork installed in dev mode — the oracle the Rust closed
forms are diffed against at ≤1e-8.

```bash
cargo test kspace::
```

Filters by substring (module path or test name), e.g. `cargo test grappa_unfolds` for one test.

```bash
cargo build --release --features cli,par
```

Builds the three binaries (`trxscan`, `trxscan-microstructure`, `trxscan-benchmark`; add
`,kspace` for the ~5× faster FFT/NUFFT k-space path). First time is slow and needs
network + cmake: `cli` → `io` → `trx-rs` → `itk-transforms-rs` (git dep) → `hdf5-metno-src`
(builds HDF5 from source).

```bash
cargo clippy --all-targets
```

~16 pre-existing style warnings in the lib. CI (`.github/workflows/ci.yml`) runs the default,
`cli`, `cli,par` and `kspace` test matrices plus `pytest scripts`; clippy is advisory there.

**Do not run `cargo fmt` repo-wide.** The source is hand-formatted (compact one-line blocks, ~100
col); `cargo fmt` rewrites ~2300 lines. There is no rustfmt.toml.

### Running the binaries

All binaries use a `clap` CLI with named flags (`required-features = ["cli"]`); run any with
`--help` for the full list. `trxscan` simulates a 4D DWI (streamlines + tissue + scheme + fieldmap
→ complex BIDS output, plus optional GNL / GRE-fieldmap / truth-peak / noise-map ground truth);
`trxscan-microstructure` writes ground-truth scalar maps (no acquisition); `trxscan-benchmark`
emits the scoreable Gibbs-ringing factor grid (`scripts/acceptance.py` scores it). Required inputs
are `--wm/--gm/--csf/--mask/--streamlines` (+ `--bval/--bvec` and `--sim-*` or `--fmap` for
`trxscan`) and `-o/--out`; the rest are optional knobs with defaults. See the README for examples.

## Architecture

Two stages (the split Fiberfox also uses):

```
Signal   streamlines + tissue maps + scheme → clean per-voxel, per-gradient signal
         raster (exact 3D-DDA) → signal (Stick/Tensor/Ball) → compartments
Acquire  per-slice k-space: distortion · T2* · eddy · ghost · PF · ringing · spikes · coils · GRAPPA · noise
         readout (EPI trajectory/timing) → kspace
Motion   cuts across both: poses per volume, or per multiband slice-group
```

**The acquisition stage lives in [mrsim-acq](https://github.com/PennLINC/mrsim-acq)**, shared with
aslscan: `mat`, `orient`, `analytic`, `readout`, `phase`, `nufft`, `noise` and `config` are its
modules re-exported; `kspace` and `motion` re-export its modules and add TRXScan's diffusion-shaped
surface (`SimulationInput` by bvals/bvecs, `default_acquisition` and `hbcd_acquisition` — never
`Acquisition::default()`, which is mrsim-acq's Fiberfox-partial-Fourier default — and the dropout
by b-value); `raster::Grid` is mrsim-acq's, its geometry the `GridRaster` trait; `io` re-exports
its NIfTI volume I/O. Changes to the forward model go to mrsim-acq, gated there (its tests, aslscan's
regress, and this repo's `tools/run_resync_baseline.sh` / `tools/resync_python_baseline.py`
against `tests/fixtures/resync_baseline`).

Module map (`src/`): `scheme` (bval/bvec, shells), `raster` (segment→voxel path lengths),
`signal` (compartment responses), `compartments` (signal-stage assembly — both the original
per-segment path and the histogram-first `generate_mixture`/`signal_from_mixture` path),
`sphere` (icosphere hemisphere), `mixture` (the per-voxel orientation-histogram type),
`microstructure` (FORCE closed-form ground-truth scalars, dipy-fixture-validated), `truth`
(ground-truth fibre peaks per acquisition voxel), `readout`, `phase` (object phase model),
`kspace` (all of the acquisition stage; `kspace::hbcd_acquisition` is the shipping protocol,
`kspace::default_acquisition` the default, `SimulationInput` + `simulate_acquisition` the entry
point), `nufft` (its gridded y-transform,
feature `kspace`), `gnl` (gradient nonlinearity: coefficients, field, warp, graddev), `gre`
(synthetic dual-echo GRE fieldmap), `motion`, `orient` (FSL/LAS reorientation + FSL bvecs), `mat`
(std-only 3×3 helpers), `analytic` (Fourier test oracles), `benchmark` (Gibbs factor grid),
`io` (feature-gated), plus `noise`/`config` which are still `todo!()` stubs.

Three binaries (all `--features cli`): `trxscan` (full simulation), `trxscan-microstructure`
(ground-truth scalar maps from tractogram + tissue — no acquisition; supports Watson κ dispersion
and TRX dps SIFT2 weights) and `trxscan-benchmark`. The first two take a `--params` preset —
`neonatal` (default; Fiberfox's ffp values, T2s nearly cancel at TE 88 → weak low-b GM/WM
contrast), `adult` (literature diffusivities; T2s 68/76 ms from a 3T EPI relaxometry fit, which
the literature 70/100 overpredicted GM with), or `infant` (unmyelinated-WM diffusivities) — and an
optional per-voxel `--myelin` map (0..1; lerps
the WM compartment toward the `adult` endpoint in both signal and ground truth). GM is a **two-ball
mixture** on the mixture path (`gm_restricted_frac`/`d_soma` per preset) so GM keeps signal at high
b. The default (no-motion) `trxscan` path and `trxscan-microstructure` share the histogram-first
mixture; **motion mode still uses the per-segment `generate_compartments_moving`**, where GM is a
single ball, SIFT2 weights / κ are ignored (with a note) and `--myelin`, `--truth-peaks`, `--gnl`
are refused (hard errors, never silently dropped). Sub-flags require their parent (`--gre-*`
needs `--gre-out`, `--gnl-*` needs `--gnl`) via clap `requires`; keep new knobs on that pattern.

### Conventions to respect

- **Partial Fourier skips the START of the EPI train by default** (`PartialFourierMode::Scanner`:
  the centre is reached `ny(1-pf)` lines sooner, the eddy-decay clock starts at the first acquired
  line, `t_echo` stays the caller's number). `Contiguous` (drop the END, timing unchanged) and
  `FiberfoxCompatible` are kept for reproducing older runs; `--pf-mode` selects.
- **b-value lives in the gradient norm.** Fiberfox's signal models take `b_value = b_max` and a
  gradient scaled to `unit_bvec · sqrt(b_i / b_max)`, so `|g|²` carries the per-volume b-value.
  Always feed models `GradientScheme::fiberfox_gradients()`, never raw bvecs.
- **4D layout is voxel-major interleaved**: `(x + nx*(y + ny*z)) * ngrad + g`. Single slices inside
  `kspace` are `x + nx*y`. `io::write_4d` is the only place this is transposed to `[x,y,z,g]`.
- **Compartments stay separate** through the signal stage (`Compartments { images, t2 }` — fiber/GM/CSF) so
  `kspace` can apply per-compartment T2 relaxation. `Compartments::mixed()` collapses them.
- **Streamlines and tissue maps must share a world (RAS mm) frame.** The rasterizer maps world → the
  DWI grid's own (possibly oblique) affine; nothing re-registers for you.
- **`kspace` is the O(N³) per-line sum, not an approximation of it** — there is no FFT-convention
  ambiguity to track. It is *organised* for speed without changing a term: static
  factors hoisted, the affine-in-ky phase (fieldmap, eddy shear, y-DFT kernel) advanced by
  memoised per-voxel rotors, the eddy polynomial factored per axis, and the per-line x-DFT plus
  the 2-D reconstruction done by `rustfft` (feature `kspace`) or twiddle-table sums (default).
  The literal per-line sum lives on in the test module as the oracle
  (`restructured_forward_matches_the_literal_sum`, 1e-10); keep any further speed-up pinned to it.
  Under `kspace` the fieldmap y-sum is a type-1 NUFFT (`src/nufft.rs`, ES kernel, ~1e-13) —
  the fieldmap read as a source warp `y − sny·τ·fmap` — so the whole forward is O(N log N) except
  the legacy `--eddy` polynomial (non-affine time profile), which keeps the O(N³) rotor path.
  Production Stage B: ~145 s → 8 s; the default std-only build (twiddle tables, rotors) ~40 s.
- **Two motion paths, different fidelity.** `compartments::generate_compartments_moving`
  re-transforms streamlines per volume and re-simulates, so fiber–gradient angles change;
  `motion::apply_motion` only resamples the finished images and misses the directional
  effect. `motion::apply_multiband_motion` layers within-volume shot jumps + b-scaled dropout on top
  and returns the dropped-slice ground truth.
- **Rayon accumulation in `compartments` is deliberately chunked, not `fold`ed** — see the comment
  in `generate_compartments`. Using rayon's `fold` allocates an `nvox*ngrad` buffer per adaptive
  split (tens of GB). Don't "simplify" it. Relatedly, `generate_mixture` is deliberately **serial**
  (its comment explains: parallel chunk buffers would be `n_groups × nvox×nvert`); the parallel win
  lives in `signal_from_mixture` / `field_scalars`, which go wide over voxels.
- **The histogram-first path is the ground-truth path.** `generate_mixture` → `signal_from_mixture`
  + `microstructure::field_scalars` derive signal and truth from one object (docs/FORCE.md §3). The
  fallback contract matters: WM voxels with no streamline support must be treated as an isotropic
  Gaussian at `md_fallback()` by *both* consumers.
- **`microstructure` is a port of dipy's closed forms — don't "fix" its conventions.** Branch thresholds,
  kurtosis clips ([-3/7, 3] for MK; [-3/7, 10] for AK/RK/MKT), Carlson errtols, and guard values
  are dipy's, on purpose; the fixture diff enforces them. Change behaviour only together with
  regenerated fixtures.

### Feature flags

`io`, `config`, and `par` gate code via `#[cfg]`; `cli` gates the binaries (`required-features`, and
implies `io`; it also puts `clap::ValueEnum` on `gre::GreOutput`). `odx` is declared in
`Cargo.toml` for work that isn't written yet. The **`kspace` module is always compiled and needs
no feature**; the `kspace` feature only swaps the x-stage and reconstruction to `rustfft` and the
fieldmap y-sum to the NUFFT (identical numbers, ~5× faster than the default build). `par` (rayon) parallelizes over streamline groups
in the signal stage and over volumes in the acquisition stage. `trx-rs`, `odx-rs` and `mrsim-acq`
are git-pinned in `Cargo.toml` (no sibling checkout; for local mrsim-acq work, a `[patch]` in
`.cargo/config.toml`, see the README). TRXScan's `kspace`, `io` and `par` features forward to
mrsim-acq's. The crate is a workspace; `python/` is the pyo3 extension
(`trxscan._core`, std-only core + `kspace,par`, built with maturin) behind the `trxscan` PyPI package.

## Python bindings (`python/`)

`python/` is a workspace member: a pyo3 `cdylib` (`trxscan._core`, `python/src/lib.rs`) over the
std-only core with `kspace,par`, plus the object-oriented Python layer in `python/trxscan/`
(`Phantom`/`Object`, `Protocol`, `Artifacts`, `Motion`, `Voxel`, `Simulation`, `KSpace`,
`BidsDwi`). Conventions: arrays cross the boundary **flat** (3-D as `x + nx*(y + ny*z)` = an
F-contiguous `(nx, ny, nz)` array; 4-D as `(x + nx*(y + ny*z))*ngrad + g` = a C-contiguous
`(nz, ny, nx, ngrad)` array; k-space `kx + nx*ky` = C-order `(ny, nx)`); complex data are
`float32` pairs viewed as `complex64`; every heavy call releases the GIL. The Rust entry points
the binding needs are `kspace::simulate_acquisition_complex` (complex output, k-space capture,
per-volume TE, `slice_z`/`nz_full` for slab runs), `compartments::mixture_from_fibers`,
`streamlines::subsample_streamlines`, `motion::{dropout_events, apply_multiband_motion_slab}`.

```bash
cd python && maturin develop --release && pytest tests -m "not phantom"   # in a virtualenv
TRXSCAN_RUN_PHANTOM=1 TRXSCAN_DATA=/path TRXSCAN_CLI=target/release/trxscan pytest python/tests
```

`tests/test_phantom.py::test_bids_parity_with_the_cli` pins Python-vs-CLI bit-identity on the
real phantom (the Python sidecar is a superset of the CLI's: it adds the recorded-only
`RepetitionTime`/`FlipAngle`/`SliceTiming`/scanner descriptors and a `SimulationSoftware`
stamp). BIDS output lives in `python/trxscan/bids.py`: `Dataset` (dataset-level files,
entity-ordered names, `fmap/` for GRE/PEPOLAR, ground truth under `derivatives/trxscan`, never
in the raw tree) and `Dataset.mirror(source, phantom, out)`, which simulates every DWI/sbref/epi
run of a real BIDS dataset under `Protocol.from_bids` (voxel + `matrix` from the header, timing
and acceleration from the sidecar; `Protocol.metadata` carries the `SIDECAR_PASSTHROUGH`
descriptors, never conversion/series keys or demographics). `Protocol.pe` names the polarity in
the *written* frame: under `fsl_orientation` an LPS phantom's native polarity is flipped so
`"j-"` is always a posterior shift. `Phantom.t1w`/`t2w` are pass-through anatomy for `anat/`
(synthetic tissue contrast when absent). `Protocol.replace(voxel_mm=...)` on a protocol with a
`matrix` keeps the FOV (rescales the matrix). Every `run` also acquires a `clean_b0` (reusing
the orientation mixture; `clean_b0=False` skips it) and exposes the applied `fieldmap` and
`displacement`; `Dataset` writes them, the GNL field in ITK form and `ImageType` (from
`Protocol.gnl_tag`) as ground truth. `python/trxscan/recipes.py` (named fixture generators,
`trxscan-fixture`/`trxscan-fetch` console scripts, `recipe.json` digests for CI caches) and
`python/trxscan/score.py` (generic truth comparisons) exist for qsiprep's CI; keep the
pipeline-specific transform handling out of them. **Memory is the dense orientation histogram**
(`MixtureField.odf`, sim voxels × 321 vertices, several GB at 2 mm whole-brain), not the 4-D
output; `run(..., chunk=N)` simulates N-slice slabs and stitches them (identical results,
except dropout jumps/GNL warp within the context's reach), dropping each slab's mixture as it
goes. Recipes and `mirror` default to `chunk=8`. A sparse or per-slab histogram in Rust would
be the next step if that is not enough. Between-scan movement truths: `Dataset.add_anat/add_gre/add_dwi`
take `offset` (resampled into the scan's grid by default; `offset_mode="header"` rotates the
header, which qsiprep's anatomical `Conform(deoblique_header=True)` silently discards),
`Phantom.moved(T)` + `Phantom.grid(like=obj)` put a moved head inside another run's FOV, and
`Protocol.oblique_deg` tilts the acquisition grid; truths go to `derivatives/trxscan` as ITK
text transforms, readable with `score.read_itk_transform`. ITK conventions live in Python
there on purpose: `itk-transforms-rs` would drag a static HDF5 build into the wheel. Use `CARGO_TARGET_DIR=python/target` for wheel builds so they do not block on the
CLI build's HDF5 compile. Release: tag `X.Y.Z` (no `v`) matching `[workspace.package].version`
(`.github/workflows/release.yml` builds manylinux/musllinux/macOS/Windows wheels and publishes).

## Origins and reference anchors

The modules ported from Fiberfox (`raster`, `signal`, `scheme`, the original per-segment path in
`compartments`, the between-volume generators in `motion`, and the distortion / relaxation / eddy
/ ghost / partial-Fourier / spike / noise terms in `kspace`) carry `file:line` anchors into the
MITK-Diffusion source in their doc comments (a local checkout of
github.com/MIC-DKFZ/MITK-Diffusion; e.g.
`Modules/MriSimulation/Algorithms/itkKspaceImageFilter.cpp:452`). Read the anchor before changing
that physics, and keep the anchors accurate. Everything else (the histogram signal stage,
multiband motion, GRAPPA, complex output, object phase, scanner partial Fourier, the FFT/NUFFT
path, GNL, GRE, microstructure, the Python package) is TRXScan's own and is documented by its
tests and `docs/`. `docs/FINDINGS.md` decomposes the Fiberfox source; `docs/FEATURES.md` is the
design note for multiband motion and GRAPPA; `docs/GNL.md` for gradient nonlinearity;
`docs/FORCE.md` covers the ground-truth microstructure scalars (FORCE closed forms, dipy as the
fixture oracle) and sourcing compartment fractions, fibre density (SIFT2 / AFD), and fixel
dispersion from CONSH.

## Not implemented (don't assume from the module list)

`noise.rs` and `config.rs` are `todo!()` stubs — k-space noise lives inline in `kspace.rs`, so
`noise.rs` is dead code, and there is no TOML config (the protocol is `kspace::hbcd_acquisition` in the
library, overridden field-by-field from the flags in `src/bin/trxscan.rs`). ODX ground-truth
export, GNL together with `--motion`, and any numerical comparison against Fiberfox are unwritten. Per-fixel
κ from CONSH fixels (vs the current global κ) and the disp(κ) LUT are future work
(docs/FORCE.md §4c). The one-subject realism-tuning pipeline (MESE/MEGRE relaxometry fits, the
HTML report) was moved out of this repo into the nibs reference kit; `scripts/` keeps only the
generic real-vs-sim comparison and fitting helpers (see `scripts/README.md`).
