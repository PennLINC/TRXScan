//! Gradient nonlinearity (GNL): the scanner's gradient coils are not perfectly linear, so a spin at
//! true position `r` is encoded at `φ(r) = r + d(r)` and experiences the gradient `J(r)ᵀ g` with
//! `J = ∂φ/∂x`. See `docs/GNL.md` for the physics and the frame conventions.
//!
//! Coefficient model: the Siemens solid-harmonic convention as read by TORTOISE
//! (`tools/gradnonlin/mk_displacementMaps.h`, `GradWarpDispAtPoint`) and HCP's `gradunwarp`:
//!
//! ```text
//!   d_a(r) = R0 · Σ_{l≥3,m} [ α_a(l,m) cos(mϕ) + β_a(l,m) sin(mϕ) ] · (r/R0)^l · N_lm · P_l^m(cos θ)
//!   N_lm   = (−1)^m √((2l+1)(l−m)! / (2(l+m)!))  (m > 0),  N_l0 = 1
//! ```
//!
//! evaluated in the vendor's scanner frame: for Siemens, `p_scanner = S · (p_ras − isocentre)`
//! with `S = diag(−1, +1, −1)` (TORTOISE's `ras_to_lai · lps_to_ras`), and the displacement maps
//! back with the same `S`. The linear `l = 1` term is the nominal field and is never part of `d`.
//!
//! The associated Legendre routine is the Numerical Recipes recursion TORTOISE uses (`plgndr`),
//! Condon–Shortley phase included; `N_lm`'s `(−1)^m` cancels it, exactly as in that code.
//!
//! Pure std. Nothing here reads or writes NIfTI; `io` adapts.

use crate::mat::{self, Mat3};
use crate::raster::Grid;
use crate::Vec3;

/// Which gradient coil a term belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    fn letter(self) -> char {
        match self {
            Axis::X => 'x',
            Axis::Y => 'y',
            Axis::Z => 'z',
        }
    }
}

/// One solid-harmonic coefficient. `sine` selects the `sin(mϕ)` basis (`β`), else `cos(mϕ)` (`α`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GnlTerm {
    pub axis: Axis,
    pub l: u32,
    pub m: u32,
    pub sine: bool,
    pub coef: f64,
}

/// Scanner vendor: selects the scanner frame the coefficients are defined in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vendor {
    /// Coefficients in Siemens LAI: `p_scanner = diag(−1, 1, −1) · p_ras`.
    Siemens,
    /// Coefficients directly in RAS (TORTOISE's `lps2ras` path).
    GeOrPhilips,
}

/// A gradient-coil coefficient set.
#[derive(Debug, Clone, PartialEq)]
pub struct GradCoef {
    /// Reference radius in millimetres (Siemens files state it in metres).
    pub r0_mm: f64,
    pub vendor: Vendor,
    pub terms: Vec<GnlTerm>,
}

/// Synthetic coefficient sets calibrated to a literature envelope (docs/GNL.md §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GnlPreset {
    /// Whole-body 80 mT/m class (Prisma-like): ~1–4 mm, 3–8 % gradient deviation at 10 cm.
    WholeBody80,
    /// Connectom-class: the same shape at ~2.5× the amplitude.
    Connectom300,
}

impl GradCoef {
    /// A calibrated synthetic set. Odd `l` only; x/z use cosine terms, y uses sine terms, z has
    /// `m = 0` only; `l = 3` dominates and is negative (the field rolls off at large radius).
    pub fn preset(p: GnlPreset) -> Self {
        let scale = match p {
            GnlPreset::WholeBody80 => 1.0,
            GnlPreset::Connectom300 => 2.5,
        };
        let t = |axis: Axis, l: u32, m: u32, coef: f64| GnlTerm {
            axis,
            l,
            m,
            sine: axis == Axis::Y,
            coef: coef * scale,
        };
        GradCoef {
            r0_mm: 250.0,
            vendor: Vendor::Siemens,
            terms: vec![
                t(Axis::X, 3, 1, -0.08),
                t(Axis::X, 3, 3, 0.006),
                t(Axis::X, 5, 1, 0.010),
                t(Axis::X, 5, 5, -0.001),
                t(Axis::X, 7, 1, -0.001),
                t(Axis::Y, 3, 1, -0.08),
                t(Axis::Y, 3, 3, 0.006),
                t(Axis::Y, 5, 1, 0.010),
                t(Axis::Y, 5, 5, -0.001),
                t(Axis::Y, 7, 1, -0.001),
                t(Axis::Z, 3, 0, -0.10),
                t(Axis::Z, 5, 0, 0.015),
                t(Axis::Z, 7, 0, -0.002),
            ],
        }
    }

    /// Multiply every nonlinear term by `s` (the severity knob).
    /// A coefficient set from a CLI-style spec: a preset name (`whole-body-80`, `connectom-300`)
    /// or the path of a Siemens `.grad` coefficient file.
    pub fn from_spec(spec: &str) -> Result<Self, String> {
        Ok(match spec {
            "whole-body-80" => GradCoef::preset(GnlPreset::WholeBody80),
            "connectom-300" => GradCoef::preset(GnlPreset::Connectom300),
            path => {
                let text = std::fs::read_to_string(path)
                    .map_err(|e| format!("cannot read coefficient file {path}: {e}"))?;
                GradCoef::parse_siemens(&text).map_err(|e| format!("{path}: {e}"))?
            }
        })
    }

    pub fn scale_nonlinear(&mut self, s: f64) {
        for t in &mut self.terms {
            t.coef *= s;
        }
    }

    /// Parse a Siemens `.grad` file with the grammar TORTOISE's `read_Siemens_format` uses:
    /// a line containing `= R0` gives the reference radius in metres (read from columns 1–5);
    /// coefficient lines look like `  1 A( 3, 1)  -0.080000 x`, with the axis letter last. Any
    /// line for the y coil is a sine term, every x/z line a cosine term, regardless of `A`/`B`
    /// (that reader keys on the axis, not the letter). `l = 1` lines are the nominal field and
    /// are dropped.
    pub fn parse_siemens(text: &str) -> Result<Self, String> {
        let mut r0_mm = 250.0;
        let mut terms = Vec::new();
        for line in text.lines() {
            if line.contains("= R0") {
                let s = line.get(1..6).unwrap_or("").trim();
                let r0_m: f64 = s.parse().map_err(|_| format!("bad R0 line: {line:?}"))?;
                r0_mm = r0_m * 1000.0;
                continue;
            }
            let Some(open) = line.find('(') else { continue };
            if open < 3 || open >= 10 {
                continue;
            }
            let (Some(comma), Some(close)) = (line.find(','), line.find(')')) else { continue };
            if comma < open || close < comma {
                continue;
            }
            let l: u32 = line[open + 1..comma].trim().parse().map_err(|_| format!("bad l in {line:?}"))?;
            let m_raw: i32 = line[comma + 1..close].trim().parse().map_err(|_| format!("bad m in {line:?}"))?;
            let rest = line[close + 1..].trim_end();
            let axis = match rest.chars().last() {
                Some('x') => Axis::X,
                Some('y') => Axis::Y,
                Some('z') => Axis::Z,
                _ => continue,
            };
            let coef: f64 = rest[..rest.len() - 1]
                .trim()
                .parse()
                .map_err(|_| format!("bad coefficient in {line:?}"))?;
            if l == 1 {
                continue;
            }
            terms.push(GnlTerm { axis, l, m: m_raw.unsigned_abs(), sine: axis == Axis::Y, coef });
        }
        if terms.is_empty() {
            return Err("no coefficient lines found".into());
        }
        Ok(GradCoef { r0_mm, vendor: Vendor::Siemens, terms })
    }

    /// Write the Siemens `.grad` grammar qsiprep and TORTOISE read (see [`Self::parse_siemens`]).
    /// Only `A(l,m)` lines are emitted; the y coil's sine basis is implied by its axis.
    pub fn write_siemens(&self) -> String {
        let mut out = String::new();
        out.push_str(" Synthetic gradient coefficients - TRXScan GNL\n");
        out.push_str(&format!(" {:.3} = R0\n\n", self.r0_mm / 1000.0));
        for (i, t) in self.terms.iter().enumerate() {
            out.push_str(&format!(
                "{:>3} A({:>2},{:>2}) {:.6} {}\n",
                i + 1,
                t.l,
                t.m,
                t.coef,
                t.axis.letter()
            ));
        }
        out.push('\n');
        out
    }

    fn scanner_sign(&self) -> Vec3 {
        match self.vendor {
            Vendor::Siemens => [-1.0, 1.0, -1.0],
            Vendor::GeOrPhilips => [1.0, 1.0, 1.0],
        }
    }

    /// Displacement `d = φ(p) − p` in the scanner frame, millimetres, from the nonlinear terms.
    pub fn displacement_scanner(&self, p_mm: Vec3) -> Vec3 {
        let r = mat::norm(p_mm);
        if r < 1e-9 {
            return [0.0; 3];
        }
        let cos_theta = (p_mm[2] / r).clamp(-1.0, 1.0);
        let phi = p_mm[1].atan2(p_mm[0]);
        let ratio = r / self.r0_mm;
        let mut d = [0.0f64; 3];
        for t in &self.terms {
            let f = ratio.powi(t.l as i32);
            let ang = if t.sine { (t.m as f64 * phi).sin() } else { (t.m as f64 * phi).cos() };
            let p = normfact(t.l, t.m) * plgndr(t.l, t.m, cos_theta);
            let contrib = self.r0_mm * t.coef * f * p * ang;
            match t.axis {
                Axis::X => d[0] += contrib,
                Axis::Y => d[1] += contrib,
                Axis::Z => d[2] += contrib,
            }
        }
        d
    }

    /// Displacement in RAS millimetres at a RAS point, for a scanner isocentre at `isocenter`.
    pub fn displacement_ras(&self, p_ras: Vec3, isocenter: Vec3) -> Vec3 {
        let s = self.scanner_sign();
        let q = [
            s[0] * (p_ras[0] - isocenter[0]),
            s[1] * (p_ras[1] - isocenter[1]),
            s[2] * (p_ras[2] - isocenter[2]),
        ];
        let d = self.displacement_scanner(q);
        [s[0] * d[0], s[1] * d[1], s[2] * d[2]]
    }

    /// `J = ∂φ/∂x` (identity plus the displacement Jacobian) in RAS axes at a RAS point, by
    /// central differences of `d` (`eps` mm). `d` is a low-order polynomial, so this is accurate
    /// to ~1e-6 at `eps = 0.5`.
    pub fn jacobian_ras(&self, p_ras: Vec3, isocenter: Vec3, eps: f64) -> Mat3 {
        let mut j = [[0.0f64; 3]; 3];
        for c in 0..3 {
            let mut pp = p_ras;
            let mut pm = p_ras;
            pp[c] += eps;
            pm[c] -= eps;
            let dp = self.displacement_ras(pp, isocenter);
            let dm = self.displacement_ras(pm, isocenter);
            for r in 0..3 {
                j[r][c] = (dp[r] - dm[r]) / (2.0 * eps);
            }
            j[c][c] += 1.0;
        }
        j
    }
}

/// `N_lm`: `(−1)^m √((2l+1)(l−m)!/(2(l+m)!))` for `m > 0`, 1 for `m = 0` (as TORTOISE).
fn normfact(l: u32, m: u32) -> f64 {
    if m == 0 {
        return 1.0;
    }
    let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
    sign * (((2 * l + 1) as f64) * factorial(l - m) / (2.0 * factorial(l + m))).sqrt()
}

fn factorial(n: u32) -> f64 {
    (1..=n).fold(1.0f64, |acc, k| acc * k as f64)
}

/// Associated Legendre `P_l^m(x)`, Numerical Recipes recursion (Condon–Shortley phase included).
fn plgndr(l: u32, m: u32, x: f64) -> f64 {
    debug_assert!(m <= l);
    let mut pmm = 1.0f64;
    if m > 0 {
        let somx2 = (1.0 - x * x).max(0.0).sqrt();
        let mut fact = 1.0;
        for _ in 1..=m {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }
    if l == m {
        return pmm;
    }
    let mut pmmp1 = x * (2 * m + 1) as f64 * pmm;
    if l == m + 1 {
        return pmmp1;
    }
    let mut pll = 0.0;
    for lm in (m + 2)..=l {
        pll = (x * (2 * lm - 1) as f64 * pmmp1 - (lm + m - 1) as f64 * pmm) / (lm - m) as f64;
        pmm = pmmp1;
        pmmp1 = pll;
    }
    pll
}

/// Envelope statistics of a field over the voxels within `radius_mm` of the isocentre.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Envelope {
    pub n_voxels: usize,
    /// max |d| (mm)
    pub max_disp_mm: f64,
    /// max over axes and voxels of `| |Jᵀ e_i| − 1 |` (relative gradient amplitude deviation)
    pub max_gradient_dev: f64,
    /// max over axes and voxels of the angle between `e_i` and `Jᵀ e_i` (degrees)
    pub max_angle_deg: f64,
}

/// The field cached on a [`Grid`]: per voxel, the displacement (RAS mm), the Jacobian of `φ`
/// (RAS axes), and the backward map needed to warp images.
#[derive(Debug, Clone)]
pub struct GnlField {
    pub dims: [usize; 3],
    /// `d(x)` at each voxel centre `x`, RAS mm, layout `x + nx*(y + ny*z)`.
    pub disp: Vec<[f32; 3]>,
    /// `J = ∂φ/∂x` at each voxel centre, row-major RAS.
    pub jac: Vec<[[f32; 3]; 3]>,
    /// Continuous voxel coordinate of `φ⁻¹(x)` for each voxel centre `x`: where a spin that
    /// appears at `x` truly sits.
    pub src_vox: Vec<[f32; 3]>,
    /// `1 / |det J(φ⁻¹(x))|`, the density factor for [`GnlField::warp_volume`].
    pub inv_det: Vec<f32>,
}

impl GnlField {
    /// Evaluate the field at every voxel centre of `grid`, with the scanner isocentre at the
    /// world origin (translate the grid's affine — and the streamlines — to put it there).
    pub fn on_grid(coef: &GradCoef, grid: &Grid) -> Self {
        let isocenter: Vec3 = [0.0; 3];
        let [nx, ny, nz] = grid.dims;
        let nvox = nx * ny * nz;
        let a = &grid.voxel_to_world;
        let lin: Mat3 = [
            [a[0][0], a[0][1], a[0][2]],
            [a[1][0], a[1][1], a[1][2]],
            [a[2][0], a[2][1], a[2][2]],
        ];
        let origin = [a[0][3], a[1][3], a[2][3]];
        let inv = mat::inverse3(&lin).expect("grid affine is singular");
        let vox_size = mat::norm([lin[0][0], lin[1][0], lin[2][0]]).max(1e-6);
        let eps = 0.25 * vox_size;
        let mut disp = Vec::with_capacity(nvox);
        let mut jac = Vec::with_capacity(nvox);
        let mut src_vox = Vec::with_capacity(nvox);
        let mut inv_det = Vec::with_capacity(nvox);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let v = [x as f64, y as f64, z as f64];
                    let p = mat::matvec(&lin, v);
                    let p = [p[0] + origin[0], p[1] + origin[1], p[2] + origin[2]];
                    let d = coef.displacement_ras(p, isocenter);
                    let j = coef.jacobian_ras(p, isocenter, eps);
                    disp.push([d[0] as f32, d[1] as f32, d[2] as f32]);
                    jac.push([
                        [j[0][0] as f32, j[0][1] as f32, j[0][2] as f32],
                        [j[1][0] as f32, j[1][1] as f32, j[1][2] as f32],
                        [j[2][0] as f32, j[2][1] as f32, j[2][2] as f32],
                    ]);
                    // backward map: solve r = p − d(r) by fixed-point iteration (d is small and
                    // smooth; 8 steps take it far below 1e-3 mm for any realistic field).
                    let mut r = p;
                    for _ in 0..8 {
                        let dr = coef.displacement_ras(r, isocenter);
                        r = [p[0] - dr[0], p[1] - dr[1], p[2] - dr[2]];
                    }
                    let jr = coef.jacobian_ras(r, isocenter, eps);
                    let det = det3(&jr).abs().max(1e-6);
                    let rv = mat::matvec(&inv, mat::sub(r, origin));
                    src_vox.push([rv[0] as f32, rv[1] as f32, rv[2] as f32]);
                    inv_det.push((1.0 / det) as f32);
                }
            }
        }
        GnlField { dims: grid.dims, disp, jac, src_vox, inv_det }
    }

    /// Envelope over voxels within `radius_mm` of the isocentre (the world origin).
    pub fn envelope(&self, grid: &Grid, radius_mm: f64) -> Envelope {
        let [nx, ny, nz] = self.dims;
        let a = &grid.voxel_to_world;
        let mut e = Envelope { n_voxels: 0, max_disp_mm: 0.0, max_gradient_dev: 0.0, max_angle_deg: 0.0 };
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let (fx, fy, fz) = (x as f64, y as f64, z as f64);
                    let p = [
                        a[0][0] * fx + a[0][1] * fy + a[0][2] * fz + a[0][3],
                        a[1][0] * fx + a[1][1] * fy + a[1][2] * fz + a[1][3],
                        a[2][0] * fx + a[2][1] * fy + a[2][2] * fz + a[2][3],
                    ];
                    if mat::norm(p) > radius_mm {
                        continue;
                    }
                    let i = x + nx * (y + ny * z);
                    e.n_voxels += 1;
                    let d = self.disp[i];
                    e.max_disp_mm = e.max_disp_mm.max(mat::norm([d[0] as f64, d[1] as f64, d[2] as f64]));
                    let j = self.jac[i];
                    for axis in 0..3 {
                        // Jᵀ e_axis = row `axis` of J
                        let g = [j[axis][0] as f64, j[axis][1] as f64, j[axis][2] as f64];
                        let n = mat::norm(g);
                        e.max_gradient_dev = e.max_gradient_dev.max((n - 1.0).abs());
                        let cos = (g[axis] / n).clamp(-1.0, 1.0);
                        e.max_angle_deg = e.max_angle_deg.max(cos.acos().to_degrees());
                    }
                }
            }
        }
        e
    }

    /// The effective encoded gradient at voxel `vox`: `Jᵀ g`.
    #[inline]
    pub fn effective_gradient(&self, vox: usize, g: Vec3) -> Vec3 {
        let j = &self.jac[vox];
        [
            j[0][0] as f64 * g[0] + j[1][0] as f64 * g[1] + j[2][0] as f64 * g[2],
            j[0][1] as f64 * g[0] + j[1][1] as f64 * g[1] + j[2][1] as f64 * g[2],
            j[0][2] as f64 * g[0] + j[1][2] as f64 * g[1] + j[2][2] as f64 * g[2],
        ]
    }

    /// Warp one 3-D volume (layout `x + nx*(y + ny*z)`) from the true frame to the acquired
    /// frame: `out(x) = src(φ⁻¹(x))`, times `1/|det J|` when `modulate` (a density, e.g. a
    /// signal image) and without it for a pointwise field (e.g. the off-resonance map).
    pub fn warp_volume(&self, src: &[f32], modulate: bool) -> Vec<f32> {
        let n = self.dims[0] * self.dims[1] * self.dims[2];
        assert_eq!(src.len(), n, "volume is not on the field's grid");
        let mut out = vec![0.0f32; n];
        for (i, o) in out.iter_mut().enumerate() {
            let s = self.src_vox[i];
            let v = crate::motion::trilinear(src, self.dims, [s[0] as f64, s[1] as f64, s[2] as f64]);
            *o = if modulate { v * self.inv_det[i] } else { v };
        }
        out
    }

    /// Warp a 4-D image in layout `(x + nx*(y + ny*z))*ngrad + g`, volume by volume.
    pub fn warp_4d(&self, src: &[f32], ngrad: usize, modulate: bool) -> Vec<f32> {
        let n = self.dims[0] * self.dims[1] * self.dims[2];
        assert_eq!(src.len(), n * ngrad);
        let mut out = vec![0.0f32; n * ngrad];
        let mut vol = vec![0.0f32; n];
        for g in 0..ngrad {
            for v in 0..n {
                vol[v] = src[v * ngrad + g];
            }
            let w = self.warp_volume(&vol, modulate);
            for v in 0..n {
                out[v * ngrad + g] = w[v];
            }
        }
        out
    }

    /// The 9-component gradient-deviation image in the HCP/FSL layout: reading the nine values
    /// of a voxel row-major gives `J` expressed in the grid's own voxel axes (`J_ijk = R⁻¹ J R`,
    /// `R` the column-normalised affine rotation), identity included. Layout
    /// `(x + nx*(y + ny*z))*9 + 3*i + j`. The effective gradient in voxel axes is `Jᵀ g`.
    pub fn graddev_volumes(&self, grid: &Grid) -> Vec<f32> {
        let a = &grid.voxel_to_world;
        let mut r: Mat3 = [[0.0; 3]; 3];
        for c in 0..3 {
            let n = mat::norm([a[0][c], a[1][c], a[2][c]]).max(1e-12);
            for row in 0..3 {
                r[row][c] = a[row][c] / n;
            }
        }
        let rinv = mat::inverse3(&r).expect("affine rotation is singular");
        let n = self.dims[0] * self.dims[1] * self.dims[2];
        let mut out = vec![0.0f32; n * 9];
        for v in 0..n {
            let j = self.jac[v];
            let jf: Mat3 = [
                [j[0][0] as f64, j[0][1] as f64, j[0][2] as f64],
                [j[1][0] as f64, j[1][1] as f64, j[1][2] as f64],
                [j[2][0] as f64, j[2][1] as f64, j[2][2] as f64],
            ];
            let jijk = mat::matmul(&mat::matmul(&rinv, &jf), &r);
            for i in 0..3 {
                for k in 0..3 {
                    out[v * 9 + 3 * i + k] = jijk[i][k] as f32;
                }
            }
        }
        out
    }
}

fn det3(m: &Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_mm(dims: [usize; 3], vox: f64, origin: Vec3) -> Grid {
        Grid {
            dims,
            voxel_to_world: [
                [vox, 0.0, 0.0, origin[0]],
                [0.0, vox, 0.0, origin[1]],
                [0.0, 0.0, vox, origin[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    #[test]
    fn siemens_grammar_round_trips() {
        let c = GradCoef::preset(GnlPreset::WholeBody80);
        let text = c.write_siemens();
        assert!(text.contains(" 0.250 = R0"));
        let back = GradCoef::parse_siemens(&text).unwrap();
        assert_eq!(back.terms.len(), c.terms.len());
        for (a, b) in back.terms.iter().zip(&c.terms) {
            assert_eq!((a.axis, a.l, a.m, a.sine), (b.axis, b.l, b.m, b.sine));
            assert!((a.coef - b.coef).abs() < 1e-9);
        }
        assert!((back.r0_mm - 250.0).abs() < 1e-9);
        // TORTOISE's reader quirks: '(' at column >= 3, axis letter last, R0 in columns 1-5.
        for line in text.lines().filter(|l| l.contains(" A(")) {
            assert!(line.find('(').unwrap() >= 3);
            assert!(matches!(line.chars().last(), Some('x' | 'y' | 'z')));
        }
    }

    #[test]
    fn linear_only_set_has_no_displacement() {
        let c = GradCoef { r0_mm: 250.0, vendor: Vendor::Siemens, terms: vec![] };
        let d = c.displacement_ras([30.0, -20.0, 50.0], [0.0; 3]);
        assert_eq!(d, [0.0; 3]);
        let j = c.jacobian_ras([30.0, -20.0, 50.0], [0.0; 3], 0.5);
        for r in 0..3 {
            for k in 0..3 {
                assert!((j[r][k] - if r == k { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn odd_l_sets_are_antisymmetric() {
        let c = GradCoef::preset(GnlPreset::WholeBody80);
        for p in [[40.0, 10.0, -30.0], [-70.0, 55.0, 20.0], [5.0, -90.0, 60.0]] {
            let d = c.displacement_ras(p, [0.0; 3]);
            let dm = c.displacement_ras([-p[0], -p[1], -p[2]], [0.0; 3]);
            for k in 0..3 {
                assert!((d[k] + dm[k]).abs() < 1e-9, "axis {k}: {} vs {}", d[k], dm[k]);
            }
        }
    }

    #[test]
    fn jacobian_matches_closed_form_for_a30() {
        // z-coil A(3,0): d_z = c·R0·(r/R0)^3·P_3(cosθ), P_3(x) = (5x^3 − 3x)/2, so in the
        // scanner frame d_z = c/(2R0^2)·(2z^3 − 3z(x^2 + y^2)).
        let coef = -0.1;
        let r0 = 250.0;
        let c = GradCoef {
            r0_mm: r0,
            vendor: Vendor::GeOrPhilips, // identity frame keeps the algebra readable
            terms: vec![GnlTerm { axis: Axis::Z, l: 3, m: 0, sine: false, coef }],
        };
        let (x, y, z) = (30.0, -40.0, 60.0);
        let d = c.displacement_scanner([x, y, z]);
        let expect = coef / (2.0 * r0 * r0) * (2.0 * z * z * z - 3.0 * z * (x * x + y * y));
        assert!((d[2] - expect).abs() < 1e-9, "{} vs {expect}", d[2]);
        assert!(d[0].abs() < 1e-12 && d[1].abs() < 1e-12);
        let j = c.jacobian_ras([x, y, z], [0.0; 3], 0.5);
        let k = coef / (2.0 * r0 * r0);
        let want = [k * (-6.0 * z * x), k * (-6.0 * z * y), k * (6.0 * z * z - 3.0 * (x * x + y * y))];
        for (col, w) in want.iter().enumerate() {
            let got = j[2][col] - if col == 2 { 1.0 } else { 0.0 };
            assert!((got - w).abs() < 1e-6, "col {col}: {got} vs {w}");
        }
    }

    #[test]
    fn preset_envelope_is_realistic_at_10cm() {
        let c = GradCoef::preset(GnlPreset::WholeBody80);
        // 4 mm grid spanning ±120 mm around the isocentre.
        let g = grid_mm([61, 61, 61], 4.0, [-120.0, -120.0, -120.0]);
        let f = GnlField::on_grid(&c, &g);
        let inner = f.envelope(&g, 20.0);
        let e = f.envelope(&g, 100.0);
        assert!(inner.max_disp_mm < 0.1 && inner.max_gradient_dev < 0.005 && inner.max_angle_deg < 0.2,
            "near isocentre: {inner:?}");
        assert!(e.max_disp_mm >= 1.0 && e.max_disp_mm <= 4.0, "displacement {e:?}");
        assert!(e.max_gradient_dev >= 0.03 && e.max_gradient_dev <= 0.08, "gradient {e:?}");
        assert!(e.max_angle_deg >= 1.0 && e.max_angle_deg <= 4.0, "angle {e:?}");
    }

    #[test]
    fn backward_map_inverts_the_forward_map() {
        let c = GradCoef::preset(GnlPreset::WholeBody80);
        let g = grid_mm([31, 31, 31], 6.0, [-90.0, -90.0, -90.0]);
        let f = GnlField::on_grid(&c, &g);
        let [nx, ny, _] = g.dims;
        for (x, y, z) in [(3usize, 5usize, 7usize), (28, 2, 15), (15, 15, 15), (1, 29, 29)] {
            let i = x + nx * (y + ny * z);
            let s = f.src_vox[i];
            let r = [-90.0 + 6.0 * s[0] as f64, -90.0 + 6.0 * s[1] as f64, -90.0 + 6.0 * s[2] as f64];
            let d = c.displacement_ras(r, [0.0; 3]);
            let back = [r[0] + d[0], r[1] + d[1], r[2] + d[2]];
            let p = [-90.0 + 6.0 * x as f64, -90.0 + 6.0 * y as f64, -90.0 + 6.0 * z as f64];
            assert!(mat::norm(mat::sub(back, p)) < 1e-3, "voxel {x},{y},{z}: {back:?} vs {p:?}");
        }
    }

    #[test]
    fn warping_a_point_source_moves_it_by_the_displacement() {
        let c = GradCoef::preset(GnlPreset::Connectom300);
        let g = grid_mm([61, 61, 61], 4.0, [-120.0, -120.0, -120.0]);
        let f = GnlField::on_grid(&c, &g);
        let [nx, ny, nz] = g.dims;
        // A smooth blob at the periphery; its intensity-weighted centroid should move by ≈ d.
        let centre = [80.0, 0.0, 80.0];
        let mut src = vec![0.0f32; nx * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let p = [-120.0 + 4.0 * x as f64, -120.0 + 4.0 * y as f64, -120.0 + 4.0 * z as f64];
                    let r2 = mat::dot(mat::sub(p, centre), mat::sub(p, centre));
                    src[x + nx * (y + ny * z)] = (-r2 / (2.0 * 8.0 * 8.0)).exp() as f32;
                }
            }
        }
        let out = f.warp_volume(&src, true);
        let centroid = |img: &[f32]| {
            let (mut sw, mut sx) = (0.0f64, [0.0f64; 3]);
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let w = img[x + nx * (y + ny * z)] as f64;
                        sw += w;
                        sx[0] += w * (-120.0 + 4.0 * x as f64);
                        sx[1] += w * (-120.0 + 4.0 * y as f64);
                        sx[2] += w * (-120.0 + 4.0 * z as f64);
                    }
                }
            }
            ([sx[0] / sw, sx[1] / sw, sx[2] / sw], sw)
        };
        let (c_in, sum_in) = centroid(&src);
        let (c_out, sum_out) = centroid(&out);
        let d = c.displacement_ras(centre, [0.0; 3]);
        assert!(mat::norm(d) > 1.5, "test needs a visible displacement, got {d:?}");
        let moved = mat::sub(c_out, c_in);
        assert!(mat::norm(mat::sub(moved, d)) < 0.25 * mat::norm(d), "moved {moved:?} vs d {d:?}");
        // Jacobian modulation conserves mass.
        assert!((sum_out - sum_in).abs() < 0.02 * sum_in, "{sum_out} vs {sum_in}");
    }

    #[test]
    fn graddev_is_identity_for_a_linear_coil() {
        let c = GradCoef { r0_mm: 250.0, vendor: Vendor::Siemens, terms: vec![] };
        let g = grid_mm([3, 3, 3], 2.0, [0.0; 3]);
        let f = GnlField::on_grid(&c, &g);
        let gd = f.graddev_volumes(&g);
        for v in 0..27 {
            for i in 0..3 {
                for k in 0..3 {
                    let want = if i == k { 1.0 } else { 0.0 };
                    assert!((gd[v * 9 + 3 * i + k] - want).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn graddev_components_follow_the_voxel_axes() {
        // On an LPS-ordered grid the x and y voxel axes are the negatives of RAS x and y, so
        // J_ijk = R⁻¹ J R flips the sign of the xz, zx, yz and zy entries.
        let c = GradCoef::preset(GnlPreset::Connectom300);
        let ras = grid_mm([3, 3, 3], 2.0, [58.0, -2.0, 38.0]);
        let mut lps = ras.clone();
        lps.voxel_to_world[0][0] = -2.0;
        lps.voxel_to_world[1][1] = -2.0;
        lps.voxel_to_world[0][3] = 62.0; // voxel 2 of LPS sits where voxel 0 of RAS sits
        lps.voxel_to_world[1][3] = 2.0;
        let fr = GnlField::on_grid(&c, &ras);
        let fl = GnlField::on_grid(&c, &lps);
        let gr = fr.graddev_volumes(&ras);
        let gl = fl.graddev_volumes(&lps);
        // voxel (0,0,1) in RAS == voxel (2,2,1) in LPS (same world point)
        let vr = 0 + 3 * (0 + 3 * 1);
        let vl = 2 + 3 * (2 + 3 * 1);
        let sign = |i: usize, k: usize| if (i == 2) != (k == 2) { -1.0 } else { 1.0 };
        for i in 0..3 {
            for k in 0..3 {
                let a = gr[vr * 9 + 3 * i + k] as f64;
                let b = gl[vl * 9 + 3 * i + k] as f64;
                assert!((a - sign(i, k) * b).abs() < 1e-5, "entry {i}{k}: {a} vs {b}");
            }
        }
        assert!((gr[vr * 9 + 2] as f64).abs() > 1e-4, "test needs a nonzero xz entry");
    }
}
