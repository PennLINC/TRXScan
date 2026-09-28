//! # TRXScan
//!
//! A headless diffusion-MRI simulator. The signal and k-space models were ported from MITK
//! Fiberfox; the acquisition model adds within-volume (multiband) motion, GRAPPA, complex
//! (magnitude + phase) output, scanner-style partial Fourier, gradient nonlinearity and a
//! synthetic GRE fieldmap, and a histogram-based signal stage supplies closed-form ground truth.
//!
//! The pipeline runs in two stages, with head motion cutting across both:
//!
//! ```text
//! streamlines + tissue maps + scheme
//!   → raster (3D-DDA) → signal (Stick/Tensor/Ball) → compartments   [signal stage → clean]
//!   → readout (EPI) → kspace (distortion/T2*/eddy/coils/PF/ringing/spikes/GRAPPA/noise)  [acquisition]
//!   (+ motion applied to the fibers per volume, or per multiband slice-group)
//! → BIDS complex 4D DWI (magnitude + phase, .bval/.bvec)
//! ```
//!
//! The default path builds the per-voxel orientation [`mixture`] once and derives both the clean
//! signal and the closed-form ground-truth [`microstructure`] scalars from it.
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
/// std-only voxel-axis reorientation to the FSL/dcm2niix (radiological LAS) convention.
pub mod orient;

/// Analytic Fourier references used as test oracles (spec 4.1).
pub mod analytic;

// --- signal stage ---
pub mod scheme;
pub mod raster;
pub mod signal;
pub mod compartments;
/// Weighted streamline subsampling (std-only; shared by both binaries and the bindings).
pub mod streamlines;

// --- ground truth: orientation mixture + closed-form scalars ---
pub mod sphere;
pub mod mixture;
pub mod microstructure;

// --- acquisition stage ---
/// Object phase model (spec 3.2).
pub mod phase;
pub mod readout;
pub mod kspace;
/// Type-1 NUFFT (feature `kspace`): the fieldmap y-sum of the forward model as one gridded FFT.
pub mod nufft;
/// Scoreable benchmark outputs (spec 3.5).
pub mod benchmark;
pub mod noise;

// --- cross-cutting ---
pub mod motion;
/// Gradient nonlinearity: coefficient model, field cache, image warp, graddev export.
pub mod gnl;
/// Dual-echo GRE fieldmap synthesis from the same object (Siemens integer-phase conventions).
pub mod gre;
/// Ground-truth fibre orientations (peaks of the orientation mixture) per acquisition voxel.
pub mod truth;

// --- feature-gated I/O + config (reference their deps) ---
#[cfg(feature = "io")]
pub mod io;
#[cfg(feature = "config")]
pub mod config;
