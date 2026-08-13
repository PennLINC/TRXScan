//! # TRXScan
//!
//! A headless diffusion-MRI simulator: the MITK Fiberfox simulation math, rebuilt in Rust on
//! `trx-rs` / `odx-rs`, extended with within-volume (multiband) motion and GRAPPA.
//!
//! See `docs/FINDINGS.md` for the Fiberfox decomposition (with `file:line` anchors), and
//! `docs/PORT-PLAN.md` for the module-by-module plan this file mirrors.
//!
//! ## Pipeline
//! ```text
//! streamlines + tissue maps + scheme
//!   → raster (exact 3D-DDA)  → signal (Stick/Tensor/Ball)  → compartments   [Stage A: clean 4D DWI]
//!   → readout (EPI) → kspace (FFT: distortion/T2*/eddy/coils/PF/ringing/spikes) → noise  [Stage B]
//!   (+ motion applied to fibers per volume / per slice-group)
//! → 4D DWI NIfTI (+ ground-truth ODX)
//! ```
//!
//! ## Build shape
//! Every external dep is optional; the default build is pure std so `cargo test` runs the
//! implemented [`scheme`] module offline. Turn on features (`io`, `kspace`, `config`, `odx`,
//! `cli`) as you implement each stage.

/// A direction / point in 3D. Stubs use a plain array; real implementations should switch to
/// `nalgebra::Vector3<f64>` (behind the `io`/`kspace` features) for the linear algebra.
pub type Vec3 = [f64; 3];

/// std-only 3×3 / vector helpers (keeps the pure-math core dependency-free + testable)
pub mod mat;

// --- Stage A: signal ---
pub mod scheme; // IMPLEMENTED
pub mod raster;
pub mod signal;
pub mod compartments;

// --- Stage B: acquisition ---
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
