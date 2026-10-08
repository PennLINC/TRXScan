//! The acquisition stage. The k-space forward model and reconstruction are `mrsim-acq`'s
//! ([`mrsim_acq::kspace`], shared with aslscan), re-exported here whole; this module adds TRXScan's
//! diffusion-shaped surface on top:
//!
//! - [`default_acquisition`] and [`hbcd_acquisition`]: TRXScan's defaults. `Acquisition` is
//!   mrsim-acq's type, whose `Default` keeps Fiberfox-compatible partial Fourier (aslscan relies on
//!   it); TRXScan's default is [`PartialFourierMode::Scanner`]. Use these, never
//!   `Acquisition::default()`.
//! - [`SimulationInput`], [`simulate_acquisition`] and [`simulate_acquisition_complex`]: the entry
//!   point by b-values and gradient directions. mrsim-acq takes the sequence-neutral drives
//!   instead; the translation is P0's (`mrsim-acq/docs/plans/2026-09-21-p0-mrsim-acq-extraction.md`):
//!   the eddy drive is `bvec * bval` where `|bval| > 1e-9` (none for b0), the prep-phase drive is
//!   `(bval, bvec)` for every volume, and each compartment's T2 is a uniform `T2Volume`.
//!
//! One deliberate difference from the pre-mrsim-acq stage: a readout that starts before its
//! excitation (`t_echo + t <= 0` on an acquired line) is refused (mrsim-acq's
//! `validate_acquisition_timing`) instead of being simulated with signal growth.

pub use mrsim_acq::kspace::*;

use crate::phase::PhaseModel;

/// TRXScan's default acquisition: mrsim-acq's [`Acquisition::default`] with scanner-style partial
/// Fourier (the train starts late; see [`PartialFourierMode::Scanner`]).
pub fn default_acquisition() -> Acquisition {
    Acquisition { pf_mode: PartialFourierMode::Scanner, ..Acquisition::default() }
}

/// The HBCD-like protocol the `trxscan` binary ships: TE 88 ms, 6/8 scanner-style partial
/// Fourier, 24 ACS lines, a subtle residual Nyquist ghost, no spikes, unapodized. The PE-train
/// duration is pinned to the HBCD TotalReadoutTime (0.0917 s) for ANY matrix: distortion
/// shift = fmap · ny · t_line, so `t_line = 91.7 ms / ny`. Everything else (noise, eddy,
/// coils, GRAPPA, seed) is left at [`default_acquisition`] and set by the caller.
pub fn hbcd_acquisition(ny: usize) -> Acquisition {
    Acquisition {
        t_line: 91.7 / ny.max(1) as f64,
        t_echo: 88.0,
        t_inhom: 50.0,
        partial_fourier: 0.75,
        pf_mode: PartialFourierMode::Scanner,
        ghost_offset: 0.015,
        ..default_acquisition()
    }
}

/// The acquisition's inputs (the simulation grid is `[nx*o, ny*o, nz]`; `images` and `fmap` live
/// there, the output on the acquired matrix).
#[derive(Clone, Copy)]
pub struct SimulationInput<'a> {
    /// Simulation grid `[nx*o, ny*o, nz]`; `images` and `fmap` live here.
    pub sim_dims: [usize; 3],
    /// Acquired matrix `[nx, ny, nz]`; the output lives here. `o` is derived and must divide both
    /// in-plane axes. `o = 1` is allowed: the transforms are then an exact round trip, so the
    /// output has no intrinsic Gibbs ringing.
    pub acq_dims: [usize; 3],
    pub ngrad: usize,
    /// Per-compartment clean signal, each `(x + snx*(y + sny*z))*ngrad + g`.
    pub images: &'a [Vec<f32>],
    /// Per-compartment T2 (ms).
    pub t2: &'a [f32],
    /// Off-resonance field (Hz).
    pub fmap: &'a [f32],
    pub bvals: &'a [f64],
    /// Unit gradient directions (world RAS).
    pub bvecs: &'a [[f64; 3]],
    pub phase: &'a PhaseModel,
    /// Noise / shot-phase realisation; mixed into every per-slice seed.
    pub seed: u64,
    /// Optional per-voxel per-component noise SD on the ACQUIRED grid (`x + nx*(y + ny*z)`). When
    /// given, complex Gaussian noise of this SD is added to the reconstructed complex image, before
    /// the magnitude/phase split — so magnitude and phase share the SAME noise realization, and the
    /// written SD map is the exact ground truth for a denoiser's estimated noise level. This is the
    /// image-space, spatially-varying counterpart to `Acquisition::noise_variance` (uniform, k-space).
    pub noise_sigma: Option<&'a [f32]>,
}

/// The acquisition: every (volume, slice) of the object through the forward model — a finer
/// object, the nominal k-space band, reconstruction at the acquisition matrix (spec 3.1), with
/// the object phase model applied before encoding (spec 3.2). Returns `(magnitude, phase)` 4D
/// arrays in `(x + nx*(y + ny*z))*ngrad + g` layout (phase in radians) — the complex pair BIDS
/// `part-mag`/`part-phase` and complex denoisers need. Parallel over volumes with `par`.
///
/// This is [`simulate_acquisition_complex`] with default [`AcquisitionOptions`], followed by the
/// same `f32` magnitude/phase arithmetic; the two are bit-identical.
pub fn simulate_acquisition(inp: &SimulationInput, acq: &Acquisition) -> (Vec<f32>, Vec<f32>) {
    let out = simulate_acquisition_complex(inp, acq, &AcquisitionOptions::default());
    let n = out.re.len();
    let (mut mag, mut ph) = (vec![0.0f32; n], vec![0.0f32; n]);
    for i in 0..n {
        let (re, im) = (out.re[i], out.im[i]);
        mag[i] = (re * re + im * im).sqrt();
        ph[i] = im.atan2(re);
    }
    (mag, ph)
}

/// [`simulate_acquisition`] returning the complex image (real/imaginary) and, on request, the
/// k-space of selected slices, per-volume echo times, slab-aware slice indexing and a progress
/// callback. See [`AcquisitionOptions`]. mrsim-acq's
/// [`mrsim_acq::kspace::simulate_acquisition_complex`] with the diffusion drives (module docs).
pub fn simulate_acquisition_complex(
    inp: &SimulationInput,
    acq: &Acquisition,
    opts: &AcquisitionOptions,
) -> AcquisitionOutput {
    let SimulationInput { sim_dims, acq_dims, ngrad, images, t2, fmap, bvals, bvecs, phase, seed, noise_sigma } = *inp;
    // the first `ngrad` entries, as main read them (`bvals[g]`, `bvecs[g]`)
    let (eddy_drive, prep_drive) = diffusion_drives(&bvals[..ngrad], &bvecs[..ngrad]);
    let t2: Vec<T2Volume> = t2.iter().map(|&v| T2Volume::Uniform(v)).collect();
    let general = AcquisitionInput {
        sim_dims, acq_dims, n_volumes: ngrad, images, t2: &t2, fmap, t_inhom: None, eddy_drive: &eddy_drive,
        prep_drive: &prep_drive, phase, seed, noise_sigma,
    };
    mrsim_acq::kspace::simulate_acquisition_complex(&general, acq, opts)
}

/// The per-volume drives of a diffusion scheme (module docs): the eddy drive `bvec * bval`, or
/// `None` where `|bval| <= 1e-9` (the decision is made on the b-value, never on the vector), and
/// the prep-phase drive `(bval, bvec)` exactly as read for every volume, b0 included (the prep
/// phase's own magnitude guard disables those).
#[allow(clippy::type_complexity)]
pub fn diffusion_drives(bvals: &[f64], bvecs: &[[f64; 3]]) -> (Vec<Option<[f64; 3]>>, Vec<Option<(f64, [f64; 3])>>) {
    let eddy = bvals.iter().zip(bvecs)
        .map(|(&b, v)| if b.abs() > 1e-9 { Some([v[0] * b, v[1] * b, v[2] * b]) } else { None })
        .collect();
    let prep = bvals.iter().zip(bvecs).map(|(&b, &v)| Some((b, v))).collect();
    (eddy, prep)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TRXScan's defaults, field by field, are main's before the move to mrsim-acq (the
    /// `Acquisition::default()` and `Acquisition::hbcd` of TRXScan `acb7506`).
    #[test]
    fn the_defaults_are_trxscans() {
        let d = default_acquisition();
        assert_eq!((d.t_line, d.t_echo, d.t_inhom, d.signal_scale), (1.0, 90.0, 50.0, 100.0));
        assert_eq!((d.reverse_phase, d.do_distortions, d.do_relaxation), (false, true, true));
        assert_eq!((d.noise_variance, d.partial_fourier, d.pf_mode), (0.0, 1.0, PartialFourierMode::Scanner));
        assert_eq!((d.ghost_offset, d.eddy_strength, d.eddy_quad, d.eddy_phase, d.eddy_tau), (0.0, 0.0, 0.0, 0.0, 70.0));
        assert_eq!((d.n_spikes, d.spike_amplitude, d.window), (0, 1.0, KspaceWindow::None));
        assert_eq!((d.n_coils, d.accel, d.acs_lines, d.seed), (1, 1, 24, 0));
        assert_eq!(d.echo, EchoFormation::Spin);
        let h = hbcd_acquisition(140);
        assert_eq!((h.t_line, h.t_echo, h.t_inhom), (91.7 / 140.0, 88.0, 50.0));
        assert_eq!((h.partial_fourier, h.pf_mode, h.ghost_offset), (0.75, PartialFourierMode::Scanner, 0.015));
        assert_eq!((h.acs_lines, h.signal_scale, h.n_coils, h.seed), (24, 100.0, 1, 0));
        assert_eq!(hbcd_acquisition(0).t_line, 91.7);
    }

    /// The drives: eddy only where the b-value is nonzero, the prep drive for every volume.
    #[test]
    fn diffusion_drives_follow_the_b_value() {
        let (e, p) = diffusion_drives(&[0.0, 1000.0, 1e-12], &[[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 1.0, 0.0]]);
        assert_eq!(e, vec![None, Some([600.0, 800.0, 0.0]), None]);
        assert_eq!(p, vec![Some((0.0, [1.0, 0.0, 0.0])), Some((1000.0, [0.6, 0.8, 0.0])), Some((1e-12, [0.0, 1.0, 0.0]))]);
    }
}
