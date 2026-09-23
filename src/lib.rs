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

// The acquisition stage lives in `mrsim-acq`; re-exported so `crate::` paths keep resolving.
pub use mrsim_acq::{analytic, kspace, mat, motion, noise, orient, phase, readout, Vec3};
// `nufft.rs` is `#![cfg(feature = "kspace")]` inside mrsim-acq, so the re-export must be too.
#[cfg(feature = "kspace")]
pub use mrsim_acq::nufft;

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
/// Scoreable benchmark outputs (spec 3.5).
pub mod benchmark;

// --- cross-cutting ---
/// Gradient nonlinearity: coefficient model, field cache, image warp, graddev export.
pub mod gnl;
pub mod truth;

// --- feature-gated I/O + config (reference their deps) ---
#[cfg(feature = "io")]
pub mod io;
#[cfg(feature = "config")]
pub use mrsim_acq::config;
