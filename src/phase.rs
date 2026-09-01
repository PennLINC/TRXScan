//! Object phase (spec 3.2). Three additive terms, all applied to the object on the SIMULATION
//! grid *before* the finite Fourier acquisition.
//!
//! **No term is derived from the fieldmap.** TRXScan simulates spin-echo EPI DWI (`kspace.rs`
//! applies `exp(-|t|/t_inhom)` centred on the echo, and `readout.rs` defines
//! `time_from_rf = t_echo + time_from_max_echo`). Static off-resonance is refocused at the spin
//! echo and survives only as readout-time-dependent phase, which `kspace` already models as
//! geometric distortion. A `2*PI*fmap*TE` term would double-count B0 and impose gradient-echo
//! physics on a spin-echo sequence.

/// Deterministic Gaussian source, seeded per shot. Mirrors the SplitMix64 generator in `kspace`.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Smooth pre-readout object phase: one low-order 3D polynomial, sampled per slice.
///
/// Named for what it is. This multiplies the object *before* Fourier encoding, and
/// `F_trunc{f * exp(i*phi)} != exp(i*phi) * F_trunc{f}`, so it may stand in only for phase that
/// genuinely exists before encoding. A scanner phase convention applied *after* reconstruction
/// rotates an already-reconstructed ringing pattern and belongs in a separate transform.
///
/// It is **not** a substitute for coil-specific complex sensitivities: a real multi-coil model has
/// a different `theta_c(r)` per coil, and one common multiplier cannot reproduce the relative coil
/// phases that GRAPPA and coil combination depend on.
///
/// Coefficient order: `1, x, y, z, x^2, y^2, z^2, xy, xz, yz`, in voxel units from the FOV centre.
#[derive(Debug, Clone, Copy)]
pub struct BackgroundPhase {
    pub coeffs: [f64; 10],
}

impl BackgroundPhase {
    pub fn at(&self, x: f64, y: f64, z: f64) -> f64 {
        let c = &self.coeffs;
        c[0] + c[1] * x + c[2] * y + c[3] * z
            + c[4] * x * x + c[5] * y * y + c[6] * z * z
            + c[7] * x * y + c[8] * x * z + c[9] * y * z
    }
}

/// One shot's realised motion and effective q-vector.
#[derive(Debug, Clone, Copy)]
pub struct ShotPhase {
    /// Effective q-vector `c_q * sqrt(b) * bvec_unit`. See [`DiffusionPhase`] on why "effective".
    pub q_eff: [f64; 3],
    /// Translation drawn for this shot (voxel units).
    pub dx: [f64; 3],
    /// Rotation vector drawn for this shot (radians), about the FOV centre.
    pub rot: [f64; 3],
}

impl ShotPhase {
    /// Phase at position `r` (voxel units from the FOV centre): `q_eff . (dx + rot x r)`.
    /// Constant plus linear in `r` by construction.
    pub fn at(&self, r: [f64; 3]) -> f64 {
        let c = [
            self.rot[1] * r[2] - self.rot[2] * r[1],
            self.rot[2] * r[0] - self.rot[0] * r[2],
            self.rot[0] * r[1] - self.rot[1] * r[0],
        ];
        (0..3).map(|i| self.q_eff[i] * (self.dx[i] + c[i])).sum()
    }
}

/// Motion-induced diffusion phase, `phi = q_eff . u(r)`.
///
/// `q_eff = c_q * sqrt(b) * bvec_unit` is an **effective** q-vector, not the physical one: in PGSE
/// `b ~ q^2 (Delta - delta/3)`, so `sqrt(b) * bvec` is proportional to `q` only under fixed, known
/// timing, and TRXScan has no `delta`, `Delta` or waveform parameters. `c_q` absorbs the timing and
/// the radian/cycle convention, and is calibrated. If waveform parameters are added later this
/// becomes a physical q-vector without changing this interface.
#[derive(Debug, Clone, Copy)]
pub struct DiffusionPhase {
    pub c_q: f64,
    /// SD of the per-shot translation (voxel units).
    pub sigma_dx: f64,
    /// SD of the per-shot rotation (radians).
    pub sigma_rot: f64,
}

impl DiffusionPhase {
    pub fn shot(&self, bval: f64, bvec: [f64; 3], volume: usize, slice_group: usize, seed: u64) -> ShotPhase {
        let n = (bvec[0] * bvec[0] + bvec[1] * bvec[1] + bvec[2] * bvec[2]).sqrt();
        // b = 0 has no diffusion encoding, so no motion-induced phase. Also avoids a degenerate
        // unit vector when bvec is the zero vector.
        if bval <= 0.0 || n < 1e-12 {
            return ShotPhase { q_eff: [0.0; 3], dx: [0.0; 3], rot: [0.0; 3] };
        }
        let s = self.c_q * bval.sqrt() / n;
        let q_eff = [bvec[0] * s, bvec[1] * s, bvec[2] * s];
        let mut rng = Rng(
            seed ^ (volume as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (slice_group as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9),
        );
        let mut draw = |sd: f64| [rng.gauss() * sd, rng.gauss() * sd, rng.gauss() * sd];
        let dx = draw(self.sigma_dx);
        let rot = draw(self.sigma_rot);
        ShotPhase { q_eff, dx, rot }
    }
}

/// The complete object-phase model.
#[derive(Debug, Clone, Copy)]
pub struct PhaseModel {
    /// Global phase. A gauge choice and test control, never calibrated.
    pub global: f64,
    pub background: BackgroundPhase,
    pub diffusion: DiffusionPhase,
}

impl PhaseModel {
    /// Phase at `r` (voxel units from the FOV centre) for one shot.
    pub fn at(&self, r: [f64; 3], shot: &ShotPhase) -> f64 {
        self.global + self.background.at(r[0], r[1], r[2]) + shot.at(r)
    }

    /// A preset tuned to the HBCD-protocol NIBS data (spec 4.2).
    ///
    /// Calibrated 2026-09-01 from NIBS sub-60515 ses-01 dir-AP run-01, central 5 slices
    /// (`scripts/calibration_nibs.json`), with the two terms held to different standards.
    ///
    /// **Background is TUNED, not fitted.** Its gradient matches the measured 0.164 rad/voxel,
    /// pinned by a test. Reconstructed phase mixes magnetization, coil and combination, scanner
    /// conventions and possibly reconstruction filtering, and only some of that precedes Fourier
    /// encoding, so this is an effective benchmark parameter, never a recovered physical field.
    ///
    /// **The b-exponent is NOT a calibrated parameter.** [`DiffusionPhase::shot`] forms
    /// `q_eff = c_q * sqrt(b) * bvec`, so `p = 0.5` is fixed by construction -- it follows from
    /// modelling phase as `q . dx` with `b ~ q^2` at fixed timing. Fitting `p` is therefore a
    /// *validation* of that modelling choice, not a way to set it. Those fits bracket 0.5 without
    /// pinning it:
    ///
    /// | source | estimator | fitted `p` |
    /// |---|---|---|
    /// | NIBS, shelled | within-volume spatial | 0.445 |
    /// | NIBS, shelled | across-volume constant | ~0.66 |
    /// | ds006131, per-volume | within-volume spatial | 0.270 |
    /// | theory, bulk translation at fixed timing | -- | 0.5 |
    ///
    /// The estimator moves `p` more than the dataset does, and for good reason: the model's shot
    /// phase is a constant plus a linear term, and a within-volume spatial SD is blind to the
    /// constant while an across-volume SD sees only it. ds006131 is additionally non-shelled
    /// CS-DSI, so direction and |q| are confounded and no b has enough volumes for a stable
    /// across-volume estimate. Treat 0.27-0.66 as the honest spread around a theoretically fixed
    /// 0.5.
    ///
    /// **`sigma_rot` is HEURISTIC, not calibrated.** Only the translation amplitude below is
    /// fitted. Separating the two needs two different estimators -- a within-volume phase-gradient
    /// statistic identifies the rotation term, an across-volume global-phase statistic identifies
    /// the translation term -- and only the latter has been done. Treat `sigma_rot = 2e-3` as a
    /// placeholder that produces plausible spatial-linear phase, not a measured quantity.
    ///
    /// **Only the translation amplitude is calibrated:** `c_q * sigma_dx = 0.0425`, fitted with `p` held at
    /// 0.5 (circular statistics, thermal floor modelled as `1/SNR` in quadrature). It reproduces
    /// the measured phase SD to under 1% -- 1.344 vs 1.348 rad at b=1000, 2.328 vs 2.311 at
    /// b=3000. An earlier version wrongly used the amplitude from a `p`-free fit, which
    /// over-predicted phase by 51%. Only the product is constrained, so the split between `c_q`
    /// and `sigma_dx` is a convention.
    pub fn hbcd_like() -> Self {
        PhaseModel {
            global: 0.0,
            // hypot(0.13, 0.10) = 0.164 rad/voxel, the measured background gradient
            background: BackgroundPhase {
                coeffs: [0.0, 0.13, 0.10, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            diffusion: DiffusionPhase { c_q: 0.0425, sigma_dx: 1.0, sigma_rot: 2.0e-3 },
        }
    }

    /// A zero model: real-valued object. Useful for isolating non-phase behaviour in tests.
    pub fn none() -> Self {
        PhaseModel {
            global: 0.0,
            background: BackgroundPhase { coeffs: [0.0; 10] },
            diffusion: DiffusionPhase { c_q: 0.0, sigma_dx: 0.0, sigma_rot: 0.0 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dp() -> DiffusionPhase {
        DiffusionPhase { c_q: 1e-3, sigma_dx: 0.5, sigma_rot: 0.0 }
    }

    #[test]
    fn b_zero_gives_exactly_zero_phase() {
        let s = dp().shot(0.0, [1.0, 0.0, 0.0], 3, 1, 42);
        assert_eq!(s.at([10.0, -4.0, 2.0]), 0.0);
        assert_eq!(s.q_eff, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn phase_reverses_sign_with_the_gradient() {
        let (a, b) = (dp().shot(1000.0, [0.0, 1.0, 0.0], 3, 1, 42),
                      dp().shot(1000.0, [0.0, -1.0, 0.0], 3, 1, 42));
        let r = [1.0, 2.0, 3.0];
        assert!((a.at(r) + b.at(r)).abs() < 1e-12, "{} vs {}", a.at(r), b.at(r));
    }

    #[test]
    fn phase_scales_as_sqrt_b_not_linearly() {
        // Same shot => same dx; only |q_eff| changes. 4x b must give 2x phase, not 4x.
        let (a, b) = (dp().shot(1000.0, [1.0, 0.0, 0.0], 3, 1, 42),
                      dp().shot(4000.0, [1.0, 0.0, 0.0], 3, 1, 42));
        let r = [0.0, 0.0, 0.0];
        let ratio = b.at(r) / a.at(r);
        assert!((ratio - 2.0).abs() < 1e-9, "sqrt(b) scaling expected, got ratio {ratio}");
    }

    #[test]
    fn rotation_makes_phase_linear_in_position() {
        let d = DiffusionPhase { c_q: 1e-3, sigma_dx: 0.0, sigma_rot: 1e-3 };
        let s = d.shot(2000.0, [1.0, 0.0, 0.0], 1, 0, 7);
        // linear field => midpoint value equals the mean of the endpoints
        let (p, q) = ([0.0, -8.0, 0.0], [0.0, 8.0, 0.0]);
        let mid = [0.0, 0.0, 0.0];
        assert!((s.at(mid) - 0.5 * (s.at(p) + s.at(q))).abs() < 1e-12);
    }

    #[test]
    fn different_shots_draw_different_motion() {
        let (a, b) = (dp().shot(1000.0, [1.0, 0.0, 0.0], 3, 1, 42),
                      dp().shot(1000.0, [1.0, 0.0, 0.0], 4, 1, 42));
        assert!(a.dx != b.dx, "per-volume draws must differ");
    }

    #[test]
    fn background_field_is_smooth_and_reproducible() {
        let bg = BackgroundPhase { coeffs: [0.3, 0.01, -0.02, 0.005, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0] };
        assert_eq!(bg.at(1.0, 2.0, 3.0), bg.at(1.0, 2.0, 3.0));
        let step = (bg.at(1.1, 2.0, 3.0) - bg.at(1.0, 2.0, 3.0)).abs();
        assert!(step < 0.05, "background phase must vary slowly, got {step} per 0.1 voxel");
    }

    #[test]
    fn hbcd_like_preset_is_nondegenerate_and_sqrt_b_dependent() {
        let m = PhaseModel::hbcd_like();
        // b = 0 must still be phase-free.
        let s0 = m.diffusion.shot(0.0, [1.0, 0.0, 0.0], 1, 0, 5);
        assert_eq!(s0.at([3.0, 1.0, 0.0]), 0.0);
        // a real shell gives non-trivial phase, scaling as sqrt(b)
        let a = m.diffusion.shot(1000.0, [1.0, 0.0, 0.0], 1, 0, 5);
        let b = m.diffusion.shot(4000.0, [1.0, 0.0, 0.0], 1, 0, 5);
        let r = [2.0, -1.0, 0.0];
        assert!(a.at(r).abs() > 1e-6, "calibrated model should give real phase");
        assert!((b.at(r) / a.at(r) - 2.0).abs() < 1e-9, "sqrt(b) scaling");
        // the background field must actually vary across the FOV
        let d = (m.background.at(8.0, 0.0, 0.0) - m.background.at(-8.0, 0.0, 0.0)).abs();
        assert!(d > 1e-3, "background phase should vary, got {d}");
        // the calibrated amplitude must reproduce the MEASURED phase SD at real b-values.
        // sigma_phi(b) = c_q * sqrt(b) * sigma_dx, with the fitted product 0.0425.
        for (bval, measured) in [(1000.0_f64, 1.348_f64), (3000.0, 2.311)] {
            let pred = m.diffusion.c_q * bval.sqrt() * m.diffusion.sigma_dx;
            assert!(
                (pred - measured).abs() / measured < 0.05,
                "b={bval}: predicted sigma_phi {pred:.3} vs measured {measured:.3}"
            );
        }
        // and its in-plane gradient must match the measured 0.164 rad/voxel
        let gx = m.background.at(1.0, 0.0, 0.0) - m.background.at(0.0, 0.0, 0.0);
        let gy = m.background.at(0.0, 1.0, 0.0) - m.background.at(0.0, 0.0, 0.0);
        let g = (gx * gx + gy * gy).sqrt();
        assert!((g - 0.164).abs() < 0.005, "background gradient {g}, expected ~0.164 rad/voxel");
    }
}
