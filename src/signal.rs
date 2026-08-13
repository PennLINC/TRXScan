//! Diffusion signal models — the per-gradient response of a compartment.
//!
//! Port of `Modules/MriSimulation/SignalModels/*`. The b-value is carried in the gradient norm
//! (see [`crate::scheme`]): pass Fiberfox-encoded gradients (`unit_bvec * sqrt(b/b_max)`) and set
//! `b_value = b_max`.
//!
//! [`Stick`] is implemented and tested (it's the load-bearing intra-axonal model). [`Tensor`] and
//! [`Ball`] carry their algorithm in the doc comment; fill them in per `docs/PORT-PLAN.md`.

use crate::Vec3;

#[inline]
fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// A fiber compartment: response depends on the angle between gradient and fiber tangent.
pub trait FiberSignalModel {
    /// Normalized signal `S/S0` for one gradient and one (unit) fiber direction.
    fn simulate(&self, gradient: Vec3, fiber_dir: Vec3) -> f64;
}

/// A direction-independent (isotropic) compartment: GM / CSF free water.
pub trait IsotropicSignalModel {
    fn simulate(&self, gradient: Vec3) -> f64;
}

/// Intra-axonal stick: `S = exp(-b_value · d · (f·g)^2)`.
/// Port of `mitkStickModel.cpp::SimulateMeasurement`.
#[derive(Debug, Clone, Copy)]
pub struct Stick {
    /// baseline b-value `m_BValue` (= scheme `b_max`)
    pub b_value: f64,
    /// axial diffusivity (mm^2/s), e.g. 0.0012
    pub diffusivity: f64,
}

impl FiberSignalModel for Stick {
    fn simulate(&self, gradient: Vec3, fiber_dir: Vec3) -> f64 {
        let norm2 = dot(gradient, gradient);
        if norm2 < 1e-8 {
            return 1.0; // b0
        }
        let d = dot(fiber_dir, gradient);
        (-self.b_value * self.diffusivity * d * d).exp()
    }
}

/// Anisotropic tensor with principal axis along the fiber. For the cylindrically-symmetric case
/// `d2 == d3` (the standard DWI tensor), `D = d2·I + (d1−d2)·f̂f̂ᵀ`, so
/// `gᵀDg = d2·|g|² + (d1−d2)·(f̂·g)²` and `S = exp(-b_value · gᵀDg)` — exact and frame-unambiguous
/// (no roll degeneracy). Equivalent to `mitkTensorModel.cpp::SimulateMeasurement` for `d2==d3`.
///
/// TODO (`d2 != d3`): match Fiberfox's kernel-frame quaternion rotation
/// (`axis = kernel_dir × fiber_dir`, `angle = acos(kernel_dir·fiber_dir)`, `D = Rᵀ D_kernel R`)
/// — the perpendicular-axis roll matters only there. Use [`crate::mat::axis_angle`].
#[derive(Debug, Clone, Copy)]
pub struct Tensor {
    /// baseline b-value `m_BValue` (= scheme `b_max`)
    pub b_value: f64,
    /// eigenvalues (axial d1, radial d2, radial d3), e.g. (0.0012, 0.0003, 0.0003)
    pub eigenvalues: (f64, f64, f64),
}

impl FiberSignalModel for Tensor {
    fn simulate(&self, gradient: Vec3, fiber_dir: Vec3) -> f64 {
        let g2 = dot(gradient, gradient);
        if g2 < 1e-8 {
            return 1.0; // b0
        }
        let f = crate::mat::normalize(fiber_dir);
        let (d1, d2, _d3) = self.eigenvalues; // cylindrical: d3 assumed == d2
        let fg = dot(f, gradient);
        let gt_d_g = d2 * g2 + (d1 - d2) * fg * fg;
        (-self.b_value * gt_d_g).exp()
    }
}

/// Isotropic ball (GM/CSF free water): `S = exp(-b_value · |g|² · d_iso)`.
/// Faithful port of `mitkBallModel.cpp::SimulateMeasurement` (`bVal = |g|²`).
#[derive(Debug, Clone, Copy)]
pub struct Ball {
    pub b_value: f64,
    /// isotropic diffusivity, e.g. 0.001 (GM) or 0.003 (CSF)
    pub diffusivity: f64,
}

impl IsotropicSignalModel for Ball {
    fn simulate(&self, gradient: Vec3) -> f64 {
        let g2 = dot(gradient, gradient);
        if g2 < 1e-8 {
            return 1.0;
        }
        (-self.b_value * g2 * self.diffusivity).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stick_attenuates_along_fiber() {
        // gradient encoded for b=1000 with b_max=1000 → |g|=1 along x
        let m = Stick { b_value: 1000.0, diffusivity: 0.0012 };
        let fiber = [1.0, 0.0, 0.0];
        let g_par = [1.0, 0.0, 0.0]; // parallel: strong attenuation
        let g_perp = [0.0, 1.0, 0.0]; // perpendicular: none
        let s_par = m.simulate(g_par, fiber);
        let s_perp = m.simulate(g_perp, fiber);
        assert!(s_par < 0.35, "parallel should attenuate: {s_par}");
        assert!((s_perp - 1.0).abs() < 1e-9, "perpendicular unattenuated: {s_perp}");
        // b0
        assert!((m.simulate([0.0, 0.0, 0.0], fiber) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn tensor_anisotropy_axial_gt_radial() {
        let m = Tensor { b_value: 1000.0, eigenvalues: (0.0012, 0.0003, 0.0003) };
        let fiber = [0.0, 1.0, 0.0];
        let s_par = m.simulate([0.0, 1.0, 0.0], fiber); // g ∥ fiber → axial d1
        let s_perp = m.simulate([1.0, 0.0, 0.0], fiber); // g ⊥ fiber → radial d2
        assert!(s_par < s_perp, "axial should attenuate more: {s_par} vs {s_perp}");
        // exact: parallel exp(-b·d1), perpendicular exp(-b·d2)
        assert!((s_par - (-1000.0f64 * 0.0012).exp()).abs() < 1e-9);
        assert!((s_perp - (-1000.0f64 * 0.0003).exp()).abs() < 1e-9);
    }

    #[test]
    fn ball_is_isotropic_and_b_dependent() {
        let m = Ball { b_value: 1000.0, diffusivity: 0.003 };
        // direction-independent for a fixed |g|
        let a = m.simulate([1.0, 0.0, 0.0]);
        let b = m.simulate([0.0, 0.0, 1.0]);
        assert!((a - b).abs() < 1e-12);
        assert!((a - (-1000.0f64 * 1.0 * 0.003).exp()).abs() < 1e-9);
        // higher b (|g|²=2) attenuates more
        let hi = m.simulate([2.0f64.sqrt(), 0.0, 0.0]);
        assert!(hi < a);
        assert!((m.simulate([0.0, 0.0, 0.0]) - 1.0).abs() < 1e-12);
    }
}
