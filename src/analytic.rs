//! Analytic references for finite-Fourier acquisition, used as test oracles.
//!
//! These are closed-form results, independent of the simulator, so a disagreement between
//! [`truncated_step_profile`] and `kspace::simulate_slice` always indicts the simulator.
//!
//! Convention matches `kspace.rs`: `n` centred coefficients `k = -n/2 ..= n/2-1` (asymmetric by
//! one sample for even `n`, as real even-matrix Cartesian acquisitions are), reconstructed at the
//! `n` voxel centres `(j+0.5)/n` over a unit FOV.

use std::f64::consts::TAU;

/// Reconstructed profile of a continuous unit step: `f(x) = 0` for `x < x0`, `1` for `x >= x0`,
/// on periodic `[0,1)`, truncated to `n` centred Fourier coefficients, sampled at voxel centres.
///
/// The Fourier coefficients of the step are, for `k != 0`,
/// `c_k = (1 - exp(-i*TAU*k*x0)) / (-i*TAU*k)`, and `c_0 = 1 - x0`.
pub fn truncated_step_profile(n: usize, x0: f64) -> Vec<f64> {
    let half = (n / 2) as i64;
    (0..n)
        .map(|j| {
            let x = (j as f64 + 0.5) / n as f64;
            let mut acc = 1.0 - x0; // k = 0 term; real
            for k in -half..half {
                if k == 0 {
                    continue;
                }
                let kf = k as f64;
                let w = TAU * kf;
                // numerator 1 - exp(-i*w*x0) = (1 - cos(w*x0)) + i*sin(w*x0)
                let (s, c) = (w * x0).sin_cos();
                let (nr, ni) = (1.0 - c, s);
                // divide by -i*w  =>  multiply by i/w
                let (cr, ci) = (-ni / w, nr / w);
                // multiply by exp(i*w*x) and keep the real part
                let (s2, c2) = (w * x).sin_cos();
                acc += cr * c2 - ci * s2;
            }
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Max value above 1.0 among samples on the bright side of the edge.
    fn overshoot(prof: &[f64], n: usize) -> f64 {
        prof[n / 2 + 1..].iter().cloned().fold(f64::MIN, f64::max) - 1.0
    }

    #[test]
    fn overshoot_depends_on_subvoxel_edge_position() {
        let n = 64;
        // Edge at a voxel boundary: the sampled overshoot is far below the Gibbs constant.
        let at_boundary = overshoot(&truncated_step_profile(n, 32.0 / n as f64), n);
        assert!(
            (at_boundary - 0.0119).abs() < 0.002,
            "boundary offset overshoot {at_boundary:.4}, expected ~0.0119"
        );
        // Edge at a voxel centre: the sampled overshoot reaches the ~8.95% Gibbs constant.
        let at_centre = overshoot(&truncated_step_profile(n, 32.5 / n as f64), n);
        assert!(
            (at_centre - 0.0893).abs() < 0.002,
            "centre offset overshoot {at_centre:.4}, expected ~0.0893"
        );
        // This 7.5x spread is exactly why a fixed-overshoot test is invalid (spec 4.1.1).
        assert!(at_centre > 5.0 * at_boundary);
    }

    #[test]
    fn sidelobes_alternate_sign_regardless_of_offset() {
        let n = 64;
        for &x0 in &[32.0 / 64.0, 32.5 / 64.0] {
            let p = truncated_step_profile(n, x0);
            for j in 0..8 {
                let a = p[n / 2 + 1 + j] - 1.0;
                let b = p[n / 2 + 2 + j] - 1.0;
                assert!(a * b < 0.0, "lobes {j} and {} share a sign at x0={x0}", j + 1);
            }
        }
    }

    #[test]
    fn far_from_the_edge_the_profile_is_flat() {
        let n = 128;
        let p = truncated_step_profile(n, 64.0 / n as f64);
        // The step is periodic on [0,1), so it has TWO edges: the rising one at x0 = 0.5 and the
        // falling wrap-around at x = 0 == 1. The flat plateaus are therefore at the quarter
        // points, midway between them; p[1] and p[n-2] sit 1.5 voxels from the wrap edge and
        // carry its +-1.19% first sidelobe (the very amplitude
        // `overshoot_depends_on_subvoxel_edge_position` pins).
        assert!((p[3 * n / 4] - 1.0).abs() < 0.01, "bright plateau {}", p[3 * n / 4]);
        assert!(p[n / 4].abs() < 0.01, "dark plateau {}", p[n / 4]);
    }
}

