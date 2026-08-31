//! # TRXScan
//!
//! A headless diffusion-MRI simulator: the MITK Fiberfox simulation math, rebuilt in Rust on
//! `trx-rs` / `odx-rs` and extended with within-volume (multiband) motion, GRAPPA, and complex
//! (magnitude + phase) output.
//!
//! The pipeline runs in two stages, mirroring Fiberfox's own split, with head motion cutting
//! across both:
//!
//! ```text
//! streamlines + tissue maps + scheme
//!   → raster (exact 3D-DDA) → signal (Stick/Tensor/Ball) → compartments   [signal stage → clean]
//!   → readout (EPI) → kspace (distortion/T2*/eddy/coils/PF/ringing/spikes/GRAPPA/noise)  [acquisition]
//!   (+ motion applied to the fibers per volume, or per multiband slice-group)
//! → BIDS complex 4D DWI (magnitude + phase, .bval/.bvec)
//! ```
//!
//! A parallel path builds the per-voxel orientation [`mixture`] once and derives both the clean
//! signal and closed-form ground-truth [`microstructure`] scalars from it, so they cannot disagree.
//!
//! ## Build shape
//! Every external dependency is optional; the default build is pure std, so `cargo test` exercises
//! the simulation core offline. The feature flags (`io`, `kspace`, `config`, `odx`, `cli`, `par`)
//! pull in the crates a given capability needs — see the README.

/// A direction or point in 3D. The pure-std core uses a plain `[f64; 3]`; the feature-gated I/O and
/// k-space paths convert to `nalgebra::Vector3<f64>` where they need heavier linear algebra.
pub type Vec3 = [f64; 3];

/// std-only 3×3 / vector helpers, keeping the pure-math core dependency-free and testable offline.
pub mod mat;

/// Analytic Fourier references used as test oracles (spec 4.1).
pub mod analytic;

// --- signal stage ---
pub mod scheme;
pub mod raster;
pub mod signal;
pub mod compartments;

// --- ground truth: orientation mixture + closed-form scalars ---
pub mod sphere;
pub mod mixture;
pub mod microstructure;

// --- acquisition stage ---
pub mod readout;
pub mod kspace;
pub mod noise;

// --- cross-cutting ---
pub mod motion;

// --- feature-gated I/O + config (reference their deps) ---
#[cfg(feature = "io")]
pub mod io;
#[cfg(feature = "config")]
pub mod config;
