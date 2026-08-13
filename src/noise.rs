//! Noise models. Port of `SignalModels/mitkRicianNoiseModel`, `mitkChiSquareNoiseModel`, and the
//! complex-Gaussian used by the k-space stage.
//!
//! Fiberfox adds noise in **k-space, per coil**, with variance scaled by partial Fourier and
//! `1/(kx·ky)` (`itkKspaceImageFilter.cpp:80`) — so for GRAPPA the g-factor amplification emerges
//! naturally from reconstructing undersampled noisy multi-coil data (see `docs/FEATURES.md`).
//! Implement complex-Gaussian in k-space (physically correct); Rician/χ² are the magnitude-image
//! equivalents for quick tests. Use `rand`/`rand_distr` behind the `kspace` feature.

/// Complex-Gaussian k-space noise: add `N(0, var/2)` to each of real & imag, per coil.
pub fn add_complex_gaussian(_kspace_re: &mut [f32], _kspace_im: &mut [f32], _variance: f64, _seed: u64) {
    todo!("per-sample complex Gaussian; variance per itkKspaceImageFilter.cpp:80")
}
