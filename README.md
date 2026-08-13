# TRXScan

A from-scratch, headless **diffusion-MRI simulator in Rust** — the simulation math
of [MITK Fiberfox](https://github.com/MIC-DKFZ/MITK-Diffusion) lifted out of the
MITK/ITK/VTK/Qt stack, rebuilt on your existing Rust crates, and extended with the
two things Fiberfox can't do: **within-volume (multiband) motion** and **GRAPPA**.

Input: a tractogram (TRX/TRK) + tissue volume-fraction maps + a gradient scheme.
Output: a 4D DWI NIfTI (+ bval/bvec), with realistic acquisition artifacts and
motion — and, optionally, a ground-truth ODX of the per-voxel fiber orientations.

```
streamlines + tissue maps + scheme
        │  rasterize (exact 3D-DDA, path lengths)
        ▼
   per-voxel, per-gradient clean signal   ── Stick / Tensor / Ball
        │  per-slice k-space (FFT)
        ▼
   distortion · T2*/eddy · coils · partial-Fourier · ringing · spikes · noise
        │
        ▼            (+ motion applied to fibers per volume / per slice-group)
   4D DWI NIfTI  (+ ground-truth ODX)
```

## Why this exists

Extending Fiberfox itself means owning a MITK-Diffusion superbuild (ITK+VTK+DCMTK+CTK+Qt,
multi-hour, ~30 GB) and rebasing a fork forever. The *simulation math*, once separated
from that infrastructure, is ~2–2.5k lines. This crate is that separation.


## How it leverages your existing crates

| Your crate | What TRXScan uses it for | Notes |
|---|---|---|
| [`trx-rs`](https://github.com/tee-ar-ex/trx-rs) (`../rust/trx-rs`) | Load TRX/TRK streamlines; **apply rigid head motion** to streamlines (`apply_transform_in_place`, rayon-parallel) | Lightweight; same loader TRXViz uses; motion chain composes with your ITK/H5 warps |
| [`odx-rs`](https://github.com/PennLINC/odx-rs) (`../odx-rs`) | SH/sphere math for the optional `todf` fast-mode; **ground-truth ODX export** of per-voxel fiber ODFs | `nalgebra`/`ndarray`; pulls hdf5 (feature-gate it) |
| [`nifti`](https://crates.io/crates/nifti) 0.17 | 3D/4D NIfTI read+write, affine | Same version your crates pin |

**Rasterizer note.** TRXViz's `orientation_field.rs` attributes a segment to its *midpoint
voxel* (a direction histogram) and its grid is an isotropic axis-aligned bbox with no oblique
affine — neither is faithful to Fiberfox, which needs exact segment→voxel path lengths on the
*target DWI grid*. TRXScan implements that exact 3D-DDA itself (`src/raster.rs`).

## Build

Prereqs: a recent stable Rust (`rustup update`). Lay the repos out as siblings:

```
~/projects/
  ├── TRXScan/      (this crate)
  ├── odx-rs/
  ├── TRXViz/
  └── rust/trx-rs/
```

The path dependencies in [`Cargo.toml`](Cargo.toml) already point at those locations. Then:

```bash
cd ~/projects/TRXScan
cargo test          # runs the implemented `scheme` module tests
cargo build --release
```

The whole of **Stage A is implemented and unit-tested** (21 tests, offline, no deps): `scheme`
(bval/bvec + shells), `signal` (Stick, Tensor, Ball), `raster` (exact 3D-DDA), `compartments`
(streamlines + tissue → clean 4D signal), plus `readout` (single-shot EPI), `motion` (pose
matrices + Random/Linear resolution), and `mat` (helpers). The remaining modules — `kspace` (the
FFT acquisition stage), `noise`, `io`, `config` — are specced stubs, each with its Fiberfox
`file:line` anchor; fill them in following [`docs/PORT-PLAN.md`](docs/PORT-PLAN.md).

## Where to read next

- [`docs/FINDINGS.md`](docs/FINDINGS.md) — what Fiberfox's simulator *is*, decomposed, with `file:line` anchors into the MITK source.
- [`docs/PORT-PLAN.md`](docs/PORT-PLAN.md) — module-by-module port plan, reuse boundary, validation-against-Fiberfox strategy, milestones.
- [`docs/FEATURES.md`](docs/FEATURES.md) — designs for within-volume/multiband motion and GRAPPA.
- [`docs/FORCE.md`](docs/FORCE.md) — assessment of dipy FORCE (ground-truth microstructure scalars).

## Status

Stage A (streamlines → clean signal) implemented and tested; Stage B (`kspace`) + `io` are the
remaining work to a running simulator. See milestones in [`docs/PORT-PLAN.md`](docs/PORT-PLAN.md).
