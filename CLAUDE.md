# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What this is

A headless diffusion-MRI simulator: the MITK Fiberfox simulation math re-implemented in Rust on
`trx-rs`/`odx-rs`, plus three things Fiberfox can't do (within-volume/multiband motion, GRAPPA,
complex magnitude+phase output). Input: tractogram + tissue volume-fraction maps + FSL bval/bvec
(+ fieldmap). Output: BIDS complex 4D DWI (`part-mag`/`part-phase`) + bval/bvec. A companion
binary emits analytic ground-truth microstructure maps from the same per-voxel mixture.

## Commands

```bash
cargo test
```

The default build is **feature-less and pure std** — ~110 unit tests, no network, no system libs
(`--features cli` — which implies `io` — adds the io tests). Every module except `io`/`config`
compiles here, and the tests live as `#[cfg(test)] mod tests` at the bottom of each module;
`tests/` holds only the dipy fixture data, not test code. Keep it that way.

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

Builds the two binaries (`trxscan`, `trxscan-microstructure`). First time is slow and needs
network + cmake: `cli` → `io` → `trx-rs` → `itk-transforms-rs` (git dep) → `hdf5-metno-src`
(builds HDF5 from source).

```bash
cargo clippy --all-targets
```

~6 pre-existing warnings in the lib; there is no CI and no `deny(warnings)`.

**Do not run `cargo fmt` repo-wide.** The source is hand-formatted (compact one-line blocks, ~100
col); `cargo fmt` rewrites ~2300 lines. There is no rustfmt.toml.

### Running the binaries

Both binaries use a `clap` CLI with named flags (`required-features = ["cli"]`); run either with
`--help` for the full list. `trxscan` simulates a 4D DWI (streamlines + tissue + scheme + fieldmap
→ complex BIDS output); `trxscan-microstructure` writes ground-truth scalar maps (no acquisition).
Required inputs are `--wm/--gm/--csf/--mask/--streamlines` (+ `--bval/--bvec/--fmap` for `trxscan`)
and `-o/--out`; the rest are optional knobs with defaults. See the README for examples.

## Architecture

Two stages, mirroring Fiberfox's own split:

```
Signal   streamlines + tissue maps + scheme → clean per-voxel, per-gradient signal
         raster (exact 3D-DDA) → signal (Stick/Tensor/Ball) → compartments
Acquire  per-slice k-space: distortion · T2* · eddy · ghost · PF · ringing · spikes · coils · GRAPPA · noise
         readout (EPI trajectory/timing) → kspace
Motion   cuts across both: poses per volume, or per multiband slice-group
```

Module map (`src/`): `scheme` (bval/bvec, shells), `raster` (segment→voxel path lengths),
`signal` (compartment responses), `compartments` (signal-stage assembly — both the original
per-segment path and the histogram-first `generate_mixture`/`signal_from_mixture` path),
`sphere` (icosphere hemisphere), `mixture` (the per-voxel orientation-histogram type),
`microstructure` (FORCE closed-form ground-truth scalars, dipy-fixture-validated), `readout`,
`kspace` (all of the acquisition stage), `nufft` (its gridded y-transform, feature `kspace`), `motion`, `mat` (std-only 3×3 helpers), `io` (feature-gated), plus
`noise`/`config` which are still `todo!()` stubs.

Two binaries (both `--features cli`): `trxscan` (full simulation) and `trxscan-microstructure`
(ground-truth scalar maps from tractogram + tissue — no acquisition; supports Watson κ dispersion
and TRX dps SIFT2 weights). Both take a `--params` preset — `neonatal` (default; Fiberfox ffp
legacy, T2s nearly cancel at TE 88 → weak low-b GM/WM contrast), `adult` (3T literature values),
or `infant` (unmyelinated-WM diffusivities) — and an optional per-voxel `--myelin` map (0..1; lerps
the WM compartment toward the `adult` endpoint in both signal and ground truth). GM is a **two-ball
mixture** on the mixture path (`gm_restricted_frac`/`d_soma` per preset) so GM keeps signal at high
b. The default (no-motion) `trxscan` path and `trxscan-microstructure` share the histogram-first
mixture; **motion mode still uses the per-segment `generate_compartments_moving`**, where GM is a
single ball and SIFT2 weights / κ / myelin are ignored.

### Conventions to respect

- **b-value lives in the gradient norm.** Fiberfox's signal models take `b_value = b_max` and a
  gradient scaled to `unit_bvec · sqrt(b_i / b_max)`, so `|g|²` carries the per-volume b-value.
  Always feed models `GradientScheme::fiberfox_gradients()`, never raw bvecs.
- **4D layout is voxel-major interleaved**: `(x + nx*(y + ny*z)) * ngrad + g`. Single slices inside
  `kspace` are `x + nx*y`. `io::write_4d` is the only place this is transposed to `[x,y,z,g]`.
- **Compartments stay separate** through the signal stage (`Compartments { images, t2 }` — fiber/GM/CSF) so
  `kspace` can apply per-compartment T2 relaxation. `Compartments::mixed()` collapses them.
- **Streamlines and tissue maps must share a world (RAS mm) frame.** The rasterizer maps world → the
  DWI grid's own (possibly oblique) affine; nothing re-registers for you.
- **`kspace` is the exact O(N³) sum, not an approximation of it** — no FFT-convention ambiguity
  while validating against Fiberfox. It is *organised* for speed without changing a term: static
  factors hoisted, the affine-in-ky phase (fieldmap, eddy shear, y-DFT kernel) advanced by
  memoised per-voxel rotors, the eddy polynomial factored per axis, and the per-line x-DFT plus
  the 2-D reconstruction done by `rustfft` (feature `kspace`) or twiddle-table sums (default).
  The literal per-line sum lives on in the test module as the oracle
  (`restructured_forward_matches_the_literal_sum`, 1e-10); keep any further speed-up pinned to it.
  Under `kspace` the fieldmap y-sum is a type-1 NUFFT (`src/nufft.rs`, ES kernel, ~1e-13) —
  the fieldmap read as a source warp `y − sny·τ·fmap` — so the whole forward is O(N log N) except
  the legacy `--eddy` polynomial (non-affine time profile), which keeps the O(N³) rotor path.
  Production Stage B: ~145 s → 8 s; the default std-only build (twiddle tables, rotors) ~40 s.
- **Two motion paths, very different fidelity.** `compartments::generate_compartments_moving` is the
  faithful one (re-transforms streamlines per volume and re-simulates, so fiber–gradient angles
  change); `motion::apply_motion` only resamples the finished images and misses the directional
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
- **`microstructure` is a faithful dipy port — don't "fix" its conventions.** Branch thresholds,
  kurtosis clips ([-3/7, 3] for MK; [-3/7, 10] for AK/RK/MKT), Carlson errtols, and guard values
  are dipy's, on purpose; the fixture diff enforces them. Change behaviour only together with
  regenerated fixtures.

### Feature flags

`io`, `config`, and `par` gate code via `#[cfg]`; `cli` gates the binaries (`required-features`, and
implies `io`). `kspace` and `odx` are declared in `Cargo.toml` for work that isn't written yet —
notably the **`kspace` module is always compiled and needs no feature**; the `kspace` feature only
swaps the x-stage and reconstruction to `rustfft` and the fieldmap y-sum to the NUFFT (identical
numbers, ~5× faster than the default build). `par` (rayon) parallelizes over streamline groups
in the signal stage and over volumes in the acquisition stage. Path deps assume sibling checkouts: `../rust/trx-rs` and
`../odx-rs`.

## Fiberfox is the oracle

Nearly every module's doc comment carries a `file:line` anchor into the MITK-Diffusion source,
checked out at `~/projects/MITK-Diffusion` (e.g.
`Modules/MriSimulation/Algorithms/itkKspaceImageFilter.cpp:452`). Read the anchor before changing
physics. `docs/FINDINGS.md` decomposes that source; `docs/FEATURES.md` designs the
multiband-motion and GRAPPA extensions; `docs/FORCE.md` covers the ground-truth microstructure
scalars (FORCE closed forms, dipy as the fixture oracle) and sourcing compartment fractions, fibre
density (SIFT2 / AFD), and fixel dispersion from CONSH.

## Not implemented (don't assume from the module list)

`noise.rs` and `config.rs` are `todo!()` stubs — k-space noise lives inline in `kspace.rs`, so
`noise.rs` is dead code, and there is no TOML config (acquisition params are a hard-coded
`Acquisition` literal in `src/bin/trxscan.rs`). ODX ground-truth export, and any
oracle diff against Fiberfox are unwritten. Per-fixel κ from CONSH fixels (vs the current global κ)
and the disp(κ) LUT are future work (docs/FORCE.md §4c).
