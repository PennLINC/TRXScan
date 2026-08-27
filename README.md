# TRXScan

A from-scratch, headless **diffusion-MRI simulator in Rust** — the simulation math
of [MITK Fiberfox](https://github.com/MIC-DKFZ/MITK-Diffusion) lifted out of the
MITK/ITK/VTK/Qt stack, rebuilt on PennLINC Rust crates, and extended with 
**within-volume (multiband) motion** and **GRAPPA**.

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

## Build

Prereqs: a recent stable Rust (`rustup update`). Lay the repos out as siblings:

```
~/projects/
  ├── TRXScan/      (this crate)
  ├── odx-rs/
  ├── TRXViz/
  └── rust/trx-rs/
```
