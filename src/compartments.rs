//! Stage A assembly: streamlines + tissue maps → clean per-voxel, per-gradient signal.
//!
//! Port of the accumulation + normalization in `itkTractsToDWIImageFilter.cpp:1108–1230`.
//!
//! Per streamline segment: rasterize to `(voxel, length)` ([`crate::raster`]); accumulate the
//! fiber response `length · seg_area · (f_intra·stick + f_extra·tensor)` and the intra-axonal
//! volume. Then per masked voxel, normalize the fiber response by its intra-axonal volume and mix
//! with the isotropic GM/CSF compartments by tissue fraction:
//! `S = wm·fiber̄ + gm·ball_gm + csf·ball_csf`.
//!
//! **Design note (FORCE).** The per-voxel result this builds *is* a Gaussian multi-tensor mixture
//! — the same object dipy's FORCE (`dipy.sims.force`) samples parametrically. The difference is
//! that here the fiber orientations come from the **tractogram**, not a Watson/Bingham prior, so
//! there's no library coverage gap (FORCE FINDINGS §1) and the geometry is ground truth. Retaining
//! this mixture (orientations + weights + diffusivities per voxel) is the natural hook to later
//! emit FORCE's closed-form ground-truth scalars (DKI/MAP-MRI/QTI/RISH). See docs.

use crate::mat;
use crate::raster::Grid;
use crate::scheme::GradientScheme;
use crate::signal::{Ball, FiberSignalModel, IsotropicSignalModel, Stick, Tensor};
use std::f64::consts::PI;

/// Per-voxel tissue volume fractions (0..1) on the acquisition grid, flat `x + nx*(y + ny*z)`.
pub struct TissueFractions {
    pub dims: [usize; 3],
    pub wm: Vec<f32>,
    pub gm: Vec<f32>,
    pub csf: Vec<f32>,
    pub mask: Vec<u8>,
}

/// Compartment model parameters. Defaults match Fiberfox / the PennBBL fiberfox-wrapper.
#[derive(Debug, Clone, Copy)]
pub struct CompartmentParams {
    pub b_value: f64,        // = scheme b_max
    pub fiber_radius_mm: f64,
    pub intra_frac: f64,     // intra/extra split of WM (0.55 / 0.45)
    pub extra_frac: f64,
    pub d_intra: f64,               // intra-axonal stick axial diffusivity
    pub d_extra: (f64, f64, f64),   // extra-axonal tensor eigenvalues
    pub d_gm: f64,                  // GM ball
    pub d_csf: f64,                 // CSF ball
    pub t2_fiber: f32,              // T2 (ms) of the fiber (intra+extra) compartment
    pub t2_gm: f32,
    pub t2_csf: f32,
}

impl Default for CompartmentParams {
    fn default() -> Self {
        // Diffusivities match the Fiberfox ffp used for the reference dataset (neonatal-ish).
        CompartmentParams {
            b_value: 1000.0,
            fiber_radius_mm: 1.0,
            intra_frac: 0.55,
            extra_frac: 0.45,
            d_intra: 0.0015,
            d_extra: (0.0015, 0.0006, 0.0006),
            d_gm: 0.0012,
            d_csf: 0.003,
            t2_fiber: 180.0,
            t2_gm: 220.0,
            t2_csf: 2500.0,
        }
    }
}

/// Clean 4D signal, layout `(x + nx*(y + ny*z)) * ngrad + g`.
pub struct CleanDwi {
    pub dims: [usize; 3],
    pub ngrad: usize,
    pub data: Vec<f32>,
}

/// Per-compartment clean signals kept separate (so Stage B can apply per-compartment T2), each in
/// the same layout as [`CleanDwi`]. `t2[i]` (ms) is the T2 of `images[i]`.
pub struct Compartments {
    pub dims: [usize; 3],
    pub ngrad: usize,
    pub images: Vec<Vec<f32>>,
    pub t2: Vec<f32>,
}

impl Compartments {
    /// Collapse to the mixed S/S₀ signal (compartment sum), ignoring T2.
    pub fn mixed(&self) -> CleanDwi {
        let n = self.images.first().map_or(0, |v| v.len());
        let mut data = vec![0.0f32; n];
        for img in &self.images {
            for (d, v) in data.iter_mut().zip(img) {
                *d += v;
            }
        }
        CleanDwi { dims: self.dims, ngrad: self.ngrad, data }
    }
}

/// Assemble Stage A as a mixed S/S₀ signal (compartment sum, no T2).
pub fn generate_clean_signal(
    grid: &Grid,
    positions: &[[f64; 3]],
    offsets: &[u32],
    tissue: &TissueFractions,
    scheme: &GradientScheme,
    params: &CompartmentParams,
) -> CleanDwi {
    generate_compartments(grid, positions, offsets, tissue, scheme, params).mixed()
}

/// Assemble Stage A keeping the fiber / GM / CSF compartments separate (for per-tissue T2 in
/// Stage B). `positions` is flat world-mm points; `offsets` is CSR (len = n_streamlines+1).
pub fn generate_compartments(
    grid: &Grid,
    positions: &[[f64; 3]],
    offsets: &[u32],
    tissue: &TissueFractions,
    scheme: &GradientScheme,
    params: &CompartmentParams,
) -> Compartments {
    let [nx, ny, nz] = grid.dims;
    let nvox = nx * ny * nz;
    let ngrad = scheme.len();
    let flat = |v: [usize; 3]| v[0] + nx * (v[1] + ny * v[2]);

    let grads = scheme.fiberfox_gradients();
    let stick = Stick { b_value: params.b_value, diffusivity: params.d_intra };
    let extra = Tensor { b_value: params.b_value, eigenvalues: params.d_extra };
    let gm = Ball { b_value: params.b_value, diffusivity: params.d_gm };
    let csf = Ball { b_value: params.b_value, diffusivity: params.d_csf };
    let seg_area = PI * params.fiber_radius_mm * params.fiber_radius_mm;

    let n_streamlines = offsets.len().saturating_sub(1);

    // Accumulate one streamline's segments into the (fiber, intra_vol) buffers. Read-only over the
    // shared inputs, so it's safe to run per-streamline in parallel (each thread owns its buffers).
    let process = |s: usize, fiber: &mut [f64], intra_vol: &mut [f64]| {
        let (lo, hi) = (offsets[s] as usize, offsets[s + 1] as usize);
        for j in lo..hi.saturating_sub(1) {
            let (a, b) = (positions[j], positions[j + 1]);
            let dir = mat::normalize(mat::sub(b, a));
            if mat::norm(dir) < 0.5 {
                continue;
            }
            let hits = grid.intersect_segment(a, b);
            if hits.is_empty() {
                continue;
            }
            // fiber response per gradient for this segment (compute once, distribute to voxels)
            let resp: Vec<f64> = (0..ngrad)
                .map(|g| {
                    params.intra_frac * stick.simulate(grads[g], dir)
                        + params.extra_frac * extra.simulate(grads[g], dir)
                })
                .collect();
            for h in &hits {
                let vf = flat(h.voxel);
                let w = h.length * seg_area;
                let base = vf * ngrad;
                for g in 0..ngrad {
                    fiber[base + g] += w * resp[g];
                }
                intra_vol[vf] += w;
            }
        }
    };

    #[cfg(feature = "par")]
    let (fiber, intra_vol) = {
        use rayon::prelude::*;
        // Process GROUPS of streamlines, not one per task: a single streamline is far too little
        // work, so per-streamline rayon overhead dwarfs it. Split into a FIXED, small number of
        // big chunks (≈2×threads) and `map` each to its own accumulation buffer — this pins the
        // number of big (nvox·ngrad) buffers to the chunk count. (Do NOT use `fold` here: rayon
        // adaptively splits the chunked iterator into hundreds of runs, calling the buffer-
        // allocating init for each → tens of GB and slower than serial.) `reduce_with` sums the
        // buffers with no identity allocation.
        let n_groups = (rayon::current_num_threads() * 2).max(1);
        let chunk = n_streamlines.div_ceil(n_groups).max(1);
        (0..n_streamlines)
            .into_par_iter()
            .chunks(chunk)
            .map(|group| {
                let mut fiber = vec![0.0f64; nvox * ngrad];
                let mut intra_vol = vec![0.0f64; nvox];
                for s in group {
                    process(s, &mut fiber, &mut intra_vol);
                }
                (fiber, intra_vol)
            })
            .reduce_with(|mut a, b| {
                a.0.iter_mut().zip(b.0.iter()).for_each(|(x, y)| *x += y);
                a.1.iter_mut().zip(b.1.iter()).for_each(|(x, y)| *x += y);
                a
            })
            .unwrap_or_else(|| (vec![0.0f64; nvox * ngrad], vec![0.0f64; nvox]))
    };
    #[cfg(not(feature = "par"))]
    let (fiber, intra_vol) = {
        let mut fiber = vec![0.0f64; nvox * ngrad];
        let mut intra_vol = vec![0.0f64; nvox];
        for s in 0..n_streamlines {
            process(s, &mut fiber, &mut intra_vol);
        }
        (fiber, intra_vol)
    };

    // hindered fallback for WM voxels with no streamline (isotropic, mean diffusivity)
    let md = (params.d_extra.0 + params.d_extra.1 + params.d_extra.2) / 3.0;

    let mut fiber_img = vec![0.0f32; nvox * ngrad];
    let mut gm_img = vec![0.0f32; nvox * ngrad];
    let mut csf_img = vec![0.0f32; nvox * ngrad];
    for vox in 0..nvox {
        if tissue.mask[vox] == 0 {
            continue;
        }
        let (mut wf, mut gf, mut cf) =
            (tissue.wm[vox] as f64, tissue.gm[vox] as f64, tissue.csf[vox] as f64);
        let tot = wf + gf + cf;
        if tot <= 1e-9 {
            continue;
        }
        wf /= tot;
        gf /= tot;
        cf /= tot;
        let iv = intra_vol[vox];
        let base = vox * ngrad;
        for g in 0..ngrad {
            let fiber_resp = if iv > 1e-12 {
                fiber[base + g] / iv
            } else {
                let g2 = mat::dot(grads[g], grads[g]);
                (-params.b_value * g2 * md).exp() // no fibers → hindered isotropic
            };
            fiber_img[base + g] = (wf * fiber_resp) as f32;
            gm_img[base + g] = (gf * gm.simulate(grads[g])) as f32;
            csf_img[base + g] = (cf * csf.simulate(grads[g])) as f32;
        }
    }

    Compartments {
        dims: grid.dims,
        ngrad,
        images: vec![fiber_img, gm_img, csf_img],
        t2: vec![params.t2_fiber, params.t2_gm, params.t2_csf],
    }
}

/// Motion-aware Stage A — the faithful port of Fiberfox's `SimulateMotion`: for each volume, the
/// streamlines are rigidly transformed by that volume's pose (so segment directions pick up the
/// rotation → the fiber–gradient angle changes correctly), the tissue maps are resampled by the
/// same pose, and the volume's signal is **re-rasterized and re-simulated** from the moved head.
/// Rotation is about the FOV centre; the fieldmap is left in scanner space for Stage B.
///
/// Cost ≈ ×n_volumes the rasterization (parallel over volumes with `par`) — that's the price of
/// doing it right, and it captures both the geometric and directional motion effects that the
/// cheap resample ([`crate::motion::apply_motion`]) misses.
pub fn generate_compartments_moving(
    grid: &Grid,
    positions: &[[f64; 3]],
    offsets: &[u32],
    tissue: &TissueFractions,
    scheme: &GradientScheme,
    params: &CompartmentParams,
    poses: &[crate::motion::Pose],
) -> Compartments {
    let [nx, ny, nz] = grid.dims;
    let nvox = nx * ny * nz;
    let ngrad = scheme.len();
    let flat = |v: [usize; 3]| v[0] + nx * (v[1] + ny * v[2]);
    let grads = scheme.fiberfox_gradients();
    let stick = Stick { b_value: params.b_value, diffusivity: params.d_intra };
    let extra = Tensor { b_value: params.b_value, eigenvalues: params.d_extra };
    let gm = Ball { b_value: params.b_value, diffusivity: params.d_gm };
    let csf = Ball { b_value: params.b_value, diffusivity: params.d_csf };
    let seg_area = PI * params.fiber_radius_mm * params.fiber_radius_mm;
    let md = (params.d_extra.0 + params.d_extra.1 + params.d_extra.2) / 3.0;
    let n_streamlines = offsets.len().saturating_sub(1);
    let center = crate::motion::fov_center(grid.dims, grid.voxel_to_world);
    let mask_f: Vec<f32> = tissue.mask.iter().map(|&m| m as f32).collect();

    let per_vol = |g: usize| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let pose = poses.get(g).copied().unwrap_or(crate::motion::Pose::IDENTITY);
        let m = pose.to_matrix(center); // world-space rigid transform for the streamlines
        let mv = |p: [f64; 3]| {
            [
                m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
                m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
                m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3],
            ]
        };
        let grad = grads[g];
        // tissue moves with the head
        let (mwm, mgm, mcsf, mmask) = (
            crate::motion::resample_by_pose(&tissue.wm, grid.dims, grid.voxel_to_world, pose),
            crate::motion::resample_by_pose(&tissue.gm, grid.dims, grid.voxel_to_world, pose),
            crate::motion::resample_by_pose(&tissue.csf, grid.dims, grid.voxel_to_world, pose),
            crate::motion::resample_by_pose(&mask_f, grid.dims, grid.voxel_to_world, pose),
        );
        // re-rasterize the moved streamlines, accumulate the fiber signal for this gradient
        let mut fiber = vec![0.0f64; nvox];
        let mut ivol = vec![0.0f64; nvox];
        for s in 0..n_streamlines {
            let (lo, hi) = (offsets[s] as usize, offsets[s + 1] as usize);
            for j in lo..hi.saturating_sub(1) {
                let (a, b) = (mv(positions[j]), mv(positions[j + 1]));
                let dir = mat::normalize(mat::sub(b, a));
                if mat::norm(dir) < 0.5 {
                    continue;
                }
                let hits = grid.intersect_segment(a, b);
                let resp = params.intra_frac * stick.simulate(grad, dir)
                    + params.extra_frac * extra.simulate(grad, dir);
                for h in &hits {
                    let vf = flat(h.voxel);
                    let w = h.length * seg_area;
                    fiber[vf] += w * resp;
                    ivol[vf] += w;
                }
            }
        }
        // combine with the moved tissue fractions
        let (mut fib, mut gmi, mut csfi) = (vec![0.0f32; nvox], vec![0.0f32; nvox], vec![0.0f32; nvox]);
        for vox in 0..nvox {
            if mmask[vox] <= 0.5 {
                continue;
            }
            let (mut wf, mut gf, mut cf) = (mwm[vox] as f64, mgm[vox] as f64, mcsf[vox] as f64);
            let tot = wf + gf + cf;
            if tot <= 1e-9 {
                continue;
            }
            wf /= tot;
            gf /= tot;
            cf /= tot;
            let fiber_resp = if ivol[vox] > 1e-12 {
                fiber[vox] / ivol[vox]
            } else {
                (-params.b_value * mat::dot(grad, grad) * md).exp()
            };
            fib[vox] = (wf * fiber_resp) as f32;
            gmi[vox] = (gf * gm.simulate(grad)) as f32;
            csfi[vox] = (cf * csf.simulate(grad)) as f32;
        }
        (fib, gmi, csfi)
    };

    #[cfg(feature = "par")]
    let vols: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = {
        use rayon::prelude::*;
        (0..ngrad).into_par_iter().map(per_vol).collect()
    };
    #[cfg(not(feature = "par"))]
    let vols: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..ngrad).map(per_vol).collect();

    let (mut fiber_img, mut gm_img, mut csf_img) =
        (vec![0.0f32; nvox * ngrad], vec![0.0f32; nvox * ngrad], vec![0.0f32; nvox * ngrad]);
    for (g, (f, gmv, cs)) in vols.iter().enumerate() {
        for vox in 0..nvox {
            fiber_img[vox * ngrad + g] = f[vox];
            gm_img[vox * ngrad + g] = gmv[vox];
            csf_img[vox * ngrad + g] = cs[vox];
        }
    }
    Compartments {
        dims: grid.dims,
        ngrad,
        images: vec![fiber_img, gm_img, csf_img],
        t2: vec![params.t2_fiber, params.t2_gm, params.t2_csf],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_1mm(n: usize) -> Grid {
        Grid {
            dims: [n, n, n],
            voxel_to_world: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    #[test]
    fn single_fiber_signal_is_anisotropic_and_b0_unattenuated() {
        let n = 6;
        let grid = grid_1mm(n);
        // one straight streamline along x at y=z=2 (voxel centers at 2.5)
        let positions: Vec<[f64; 3]> = vec![[0.2, 2.5, 2.5], [5.8, 2.5, 2.5]];
        let offsets = vec![0u32, 2];
        // WM=1 along the fiber row, else 0
        let (mut wm, gm, csf, mut mask) =
            (vec![0f32; n * n * n], vec![0f32; n * n * n], vec![0f32; n * n * n], vec![0u8; n * n * n]);
        let flat = |x: usize, y: usize, z: usize| x + n * (y + n * z);
        for x in 0..n {
            wm[flat(x, 2, 2)] = 1.0;
            mask[flat(x, 2, 2)] = 1;
        }
        let tissue = TissueFractions { dims: [n, n, n], wm, gm, csf, mask };
        // scheme: b0, b1000 ∥x, b1000 ⊥x(along y)
        let scheme = GradientScheme::from_str("0 1000 1000", "0 1 0\n0 0 1\n0 0 0").unwrap();
        let params = CompartmentParams { b_value: 1000.0, ..Default::default() };

        let dwi = generate_clean_signal(&grid, &positions, &offsets, &tissue, &scheme, &params);
        let v = flat(3, 2, 2) * dwi.ngrad; // a voxel on the fiber
        let (s_b0, s_par, s_perp) = (dwi.data[v], dwi.data[v + 1], dwi.data[v + 2]);
        assert!((s_b0 - 1.0).abs() < 1e-4, "b0 should be ~1, got {s_b0}");
        assert!(s_par < s_perp, "∥-fiber should attenuate more: {s_par} vs {s_perp}");
        assert!(s_par > 0.0 && s_perp <= 1.0 + 1e-4);
        // a voxel off the fiber (no WM, no signal)
        let off = flat(3, 4, 4) * dwi.ngrad;
        assert_eq!(dwi.data[off], 0.0);
    }
}
