//! Signal-stage assembly: streamlines + tissue maps → clean per-voxel, per-gradient signal.
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
//! there's no library coverage gap and the geometry is ground truth. Retaining this mixture
//! (orientations + weights + diffusivities per voxel) is the natural hook to later emit FORCE's
//! closed-form ground-truth scalars (DKI/MAP-MRI/QTI/RISH).
//!
//! **Histogram-first path.** [`generate_mixture`] + [`signal_from_mixture`]
//! re-express that same accumulation with the marginalization *deferred*: segments land in a
//! per-voxel orientation histogram ([`crate::mixture::MixtureField`]) and the signal is computed
//! **from** the histogram, so the signal-stage DWI and the closed-form ground truth of
//! [`crate::microstructure`] derive from one object and cannot disagree by construction. Deferring
//! also buys the two things the per-segment path cannot express: per-streamline (SIFT2) weights
//! and a dispersion kernel. Both paths expect the caller to set `params.b_value = scheme.b_max`
//! (what the binaries do) — the per-volume b-value rides in the gradient norm, see
//! [`crate::scheme`].

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
    /// fraction of the GM compartment that is a **restricted** (soma/neurite) ball — the
    /// non-Gaussian-looking signal real GM keeps at high b; a second Gaussian, so every
    /// closed form stays exact (FORCE's soma slot). Order-of-magnitude values, not fitted.
    pub gm_restricted_frac: f64,
    /// diffusivity of the restricted GM ball (mm^2/s)
    pub d_soma: f64,
}

impl Default for CompartmentParams {
    fn default() -> Self {
        // Diffusivities match the Fiberfox ffp used for the reference dataset (neonatal-ish).
        // NOTE the T2s: at TE ~88 ms, 180 vs 220 ms nearly cancel — neonatal data has weak
        // GM/WM contrast at low b, and so does this parameter set. For adult subjects use
        // [`CompartmentParams::adult`].
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
            gm_restricted_frac: 0.15,
            d_soma: 0.0003,
        }
    }
}

impl CompartmentParams {
    /// Adult 3T parameter set. T2s from a 4-echo spin-echo EPI relaxometry fit (TE 15–100 ms) on
    /// a 3T reference subject at 1.7 mm: WM 67.7 [63–72] ms, cortical GM 75.7 [71–81] ms. The
    /// literature values first used here (Stanisz
    /// 2005 / Wansapura 1999: 70 / 100 ms) overpredicted GM at TE 88 by 30 % — the real b0
    /// GM/WM ratio is 1.16, not 1.5 — because the EPI-measured T2 of cortex, with its partial
    /// volume at 1.7 mm, is well below the pure-tissue literature number. CSF is not measurable
    /// with TE ≤ 100 ms and keeps the literature 2000 ms. Diffusivities:
    /// intra-axonal axial 1.7e-3 (standard-model range), extra-axonal (1.7, 0.6, 0.6)e-3 —
    /// the 0.55/0.45 mixture lands WM MD ≈ 0.75e-3 — and GM ball 0.85e-3, adult cortical MD.
    pub fn adult() -> Self {
        CompartmentParams {
            b_value: 1000.0,
            fiber_radius_mm: 1.0,
            intra_frac: 0.55,
            extra_frac: 0.45,
            d_intra: 0.0017,
            d_extra: (0.0017, 0.0006, 0.0006),
            d_gm: 0.00085,
            d_csf: 0.003,
            t2_fiber: 68.0,
            t2_gm: 76.0,
            t2_csf: 2000.0,
            gm_restricted_frac: 0.20,
            d_soma: 0.0003,
        }
    }

    /// Infant (~3–6 mo) parameter set: largely **unmyelinated** WM — high diffusivity and low
    /// anisotropy (infant DTI literature: WM MD ≈ 1.2–1.4e-3, FA low outside the PLIC). Lower
    /// intra-axonal fraction (0.35) and high extra-axonal radial diffusivity give a mixture
    /// WM MD ≈ 1.15e-3; GM ball 1.3e-3. T2s keep the neonatal ffp values — infant WM/GM T2s
    /// are long and *similar*, so the weak b0 GM/WM contrast this produces is itself
    /// age-appropriate. (A myelination *gradient* — PLIC/splenium restricted, periphery free —
    /// needs per-voxel parameter maps; this is the best a global set can do.)
    pub fn infant() -> Self {
        CompartmentParams {
            b_value: 1000.0,
            fiber_radius_mm: 1.0,
            intra_frac: 0.35,
            extra_frac: 0.65,
            d_intra: 0.0019,
            d_extra: (0.0019, 0.0012, 0.0012),
            d_gm: 0.0013,
            d_csf: 0.003,
            t2_fiber: 180.0,
            t2_gm: 220.0,
            t2_csf: 2500.0,
            gm_restricted_frac: 0.25,
            d_soma: 0.0003,
        }
    }
}

/// Clean 4D signal, layout `(x + nx*(y + ny*z)) * ngrad + g`.
pub struct CleanDwi {
    pub dims: [usize; 3],
    pub ngrad: usize,
    pub data: Vec<f32>,
}

/// Per-compartment clean signals kept separate (so the acquisition stage can apply per-compartment T2), each in
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

    /// Scale each compartment image by a per-compartment amplitude `[fiber, gm, csf]`
    /// (proton density x T1 saturation). Lets a caller match a real acquisition's tissue b0
    /// levels without touching the physics. No-op for an all-ones factor.
    /// The compartment images restricted to the given local slices (in that order), as a new
    /// `Compartments` on an `[nx, ny, slices.len()]` grid.
    pub fn select_slices(&self, slices: &[usize]) -> Compartments {
        let [nx, ny, nz] = self.dims;
        let per = nx * ny * self.ngrad;
        let images = self
            .images
            .iter()
            .map(|img| {
                let mut out = Vec::with_capacity(per * slices.len());
                for &z in slices {
                    assert!(z < nz, "slice {z} out of range for {nz} slices");
                    out.extend_from_slice(&img[z * per..(z + 1) * per]);
                }
                out
            })
            .collect();
        Compartments { dims: [nx, ny, slices.len()], ngrad: self.ngrad, images, t2: self.t2.clone() }
    }

    pub fn apply_s0(&mut self, s0: [f32; 3]) {
        for (img, &a) in self.images.iter_mut().zip(s0.iter()) {
            if (a - 1.0).abs() > f32::EPSILON {
                for v in img.iter_mut() {
                    *v *= a;
                }
            }
        }
    }
}

/// Assemble the signal stage as a mixed S/S₀ signal (compartment sum, no T2).
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

/// Assemble the signal stage keeping the fiber / GM / CSF compartments separate (for per-tissue T2
/// in the acquisition stage). `positions` is flat world-mm points; `offsets` is CSR (len = n_streamlines+1).
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

/// Rasterize the streamlines into the **per-voxel orientation histogram** — the primary signal-stage
/// object — instead of marginalizing to a signal at accumulation time.
///
/// The segment loop is the one in [`generate_compartments`]: same iteration, same
/// `mat::normalize` degenerate-direction skip, same [`Grid::intersect_segment`] hits. What changes
/// is where the weight goes. For streamline `s` with weight `w_s` (SIFT2 / Fiberfox's dropped
/// `fiberWeight`; `weights == None` ⇒ every `w_s = 1`), each voxel hit of
/// each segment contributes `w = w_s · length · π·r²` to that voxel's histogram row:
///
/// - `kappa == None` — all of `w` on [`crate::sphere::HemiSphere::nearest`] (plain binning);
/// - `kappa == Some(k)` — spread by a sign-free Watson kernel `K_v = exp(k·((v·d̂)² − 1))`,
///   normalized so `Σ_v K_v = 1`. That is the fixel dispersion — the
///   orientation spread the curvature-regularized tractogram smooths away — and because it is
///   normalized, the row **sum** is independent of `k`. Subtracting 1 in the exponent (rather than
///   scaling by `exp(k)` afterwards) keeps it ≤ 0, so large `k` underflows to 0 instead of
///   overflowing to inf.
///
/// Tissue fractions are normalized per voxel exactly as [`generate_compartments`] does (mask
/// check, `tot ≤ 1e-9` skip, `wm+gm+csf = 1`); voxels that fail either check are left at 0.
/// [`crate::mixture::MixtureField::fallback`] marks masked WM voxels whose row is empty — the
/// hindered-isotropic case, which [`signal_from_mixture`] and the closed forms must both treat as
/// an isotropic Gaussian at `md`, or truth and signal disagree where the tractogram is sparse.
pub fn generate_mixture(
    grid: &Grid,
    positions: &[[f64; 3]],
    offsets: &[u32],
    weights: Option<&[f32]>,
    tissue: &TissueFractions,
    params: &CompartmentParams,
    kappa: Option<f64>,
    sphere: crate::sphere::HemiSphere,
) -> crate::mixture::MixtureField {
    let [nx, ny, nz] = grid.dims;
    let nvox = nx * ny * nz;
    let nvert = sphere.len();
    let flat = |v: [usize; 3]| v[0] + nx * (v[1] + ny * v[2]);
    let seg_area = PI * params.fiber_radius_mm * params.fiber_radius_mm;
    let n_streamlines = offsets.len().saturating_sub(1);

    // SERIAL on purpose — do NOT copy the chunked-buffer pattern from `generate_compartments`.
    // This accumulation is geometry-only (no per-gradient inner loop), so there is little for
    // rayon to amortize, and a parallel version would need n_groups × (nvox·nvert) f64 buffers:
    // gigabytes at 321 vertices on any real grid. The parallel win moves to
    // `signal_from_mixture`, which is the O(nvox·nvert·ngrad) half and writes disjoint chunks.
    let mut hist = vec![0.0f64; nvox * nvert];
    let mut kern: Vec<(usize, f64)> = Vec::with_capacity(nvert);
    let (mut ts_buf, mut hits) = (Vec::with_capacity(16), Vec::with_capacity(16));
    for s in 0..n_streamlines {
        let w_s = weights.map_or(1.0, |w| w.get(s).copied().unwrap_or(1.0) as f64);
        if w_s == 0.0 {
            continue;
        }
        let (lo, hi) = (offsets[s] as usize, offsets[s + 1] as usize);
        for j in lo..hi.saturating_sub(1) {
            let (a, b) = (positions[j], positions[j + 1]);
            let dir = mat::normalize(mat::sub(b, a));
            if mat::norm(dir) < 0.5 {
                continue;
            }
            grid.intersect_segment_into(a, b, &mut ts_buf, &mut hits);
            if hits.is_empty() {
                continue;
            }
            // orientation kernel for this segment — identical for every voxel it crosses
            kern.clear();
            match kappa {
                None => kern.push((sphere.nearest(dir), 1.0)),
                Some(k) => {
                    let mut sum = 0.0;
                    for (i, v) in sphere.verts.iter().enumerate() {
                        let d = mat::dot(*v, dir);
                        let kv = (k * (d * d - 1.0)).exp();
                        if kv < 1e-8 {
                            continue; // relative to K_max = 1 (an exactly aligned vertex)
                        }
                        kern.push((i, kv));
                        sum += kv;
                    }
                    if sum > 0.0 {
                        for e in kern.iter_mut() {
                            e.1 /= sum;
                        }
                    } else {
                        // κ so large that even the nearest vertex underflowed → plain binning
                        kern.push((sphere.nearest(dir), 1.0));
                    }
                }
            }
            for h in &hits {
                let base = flat(h.voxel) * nvert;
                let w = w_s * h.length * seg_area;
                for &(i, kv) in &kern {
                    hist[base + i] += w * kv;
                }
            }
        }
    }

    let (mut wm, mut gm, mut csf) = (vec![0.0f32; nvox], vec![0.0f32; nvox], vec![0.0f32; nvox]);
    let mut fallback = vec![0u8; nvox];
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
        wm[vox] = wf as f32;
        gm[vox] = gf as f32;
        csf[vox] = cf as f32;
        let row: f64 = hist[vox * nvert..(vox + 1) * nvert].iter().sum();
        fallback[vox] = (wf > 0.0 && row < 1e-12) as u8;
    }

    crate::mixture::MixtureField {
        dims: grid.dims,
        sphere,
        odf: hist.iter().map(|&x| x as f32).collect(),
        wm,
        gm,
        csf,
        fallback,
        params: *params,
        myelin: None,
    }
}

/// Build a [`crate::mixture::MixtureField`] from an explicit fibre list instead of rasterized
/// streamlines: entry `i` deposits `weight[i]` of orientation mass along `dir[i]` into voxel
/// `voxel[i]` (flat `x + nx*(y + ny*z)`), through the same nearest-vertex / Watson-κ kernel as
/// [`generate_mixture`]. Tissue normalisation and the hindered fallback are identical, so a
/// synthetic object (a box with one crossing, a single voxel) goes through exactly the signal
/// and ground-truth code paths real anatomy does.
pub fn mixture_from_fibers(
    dims: [usize; 3],
    voxel: &[u32],
    dir: &[[f64; 3]],
    weight: &[f64],
    tissue: &TissueFractions,
    params: &CompartmentParams,
    kappa: Option<f64>,
    sphere: crate::sphere::HemiSphere,
) -> crate::mixture::MixtureField {
    assert_eq!(voxel.len(), dir.len());
    assert_eq!(voxel.len(), weight.len());
    assert_eq!(tissue.dims, dims, "tissue fractions are not on the requested grid");
    let [nx, ny, nz] = dims;
    let nvox = nx * ny * nz;
    let nvert = sphere.len();
    let mut hist = vec![0.0f64; nvox * nvert];
    let mut kern: Vec<(usize, f64)> = Vec::with_capacity(nvert);
    for ((&v, d), &w) in voxel.iter().zip(dir).zip(weight) {
        let v = v as usize;
        assert!(v < nvox, "fibre voxel index {v} out of range for {dims:?}");
        let d = mat::normalize(*d);
        if mat::norm(d) < 0.5 || w == 0.0 {
            continue;
        }
        kern.clear();
        match kappa {
            None => kern.push((sphere.nearest(d), 1.0)),
            Some(k) => {
                let mut sum = 0.0;
                for (i, vert) in sphere.verts.iter().enumerate() {
                    let c = mat::dot(*vert, d);
                    let kv = (k * (c * c - 1.0)).exp();
                    if kv < 1e-8 {
                        continue;
                    }
                    kern.push((i, kv));
                    sum += kv;
                }
                if sum > 0.0 {
                    for e in kern.iter_mut() {
                        e.1 /= sum;
                    }
                } else {
                    kern.push((sphere.nearest(d), 1.0));
                }
            }
        }
        for &(i, kv) in &kern {
            hist[v * nvert + i] += w * kv;
        }
    }
    let (mut wm, mut gm, mut csf) = (vec![0.0f32; nvox], vec![0.0f32; nvox], vec![0.0f32; nvox]);
    let mut fallback = vec![0u8; nvox];
    for vox in 0..nvox {
        if tissue.mask[vox] == 0 {
            continue;
        }
        let (mut wf, mut gf, mut cf) = (tissue.wm[vox] as f64, tissue.gm[vox] as f64, tissue.csf[vox] as f64);
        let tot = wf + gf + cf;
        if tot <= 1e-9 {
            continue;
        }
        wf /= tot;
        gf /= tot;
        cf /= tot;
        wm[vox] = wf as f32;
        gm[vox] = gf as f32;
        csf[vox] = cf as f32;
        let row: f64 = hist[vox * nvert..(vox + 1) * nvert].iter().sum();
        fallback[vox] = (wf > 0.0 && row < 1e-12) as u8;
    }
    crate::mixture::MixtureField {
        dims,
        sphere,
        odf: hist.iter().map(|&x| x as f32).collect(),
        wm,
        gm,
        csf,
        fallback,
        params: *params,
        myelin: None,
    }
}

/// [`signal_from_mixture`] with a per-voxel gradient-nonlinearity field: every voxel sees the
/// gradient the coils actually produced there, `J(x)ᵀ g` ([`crate::gnl::GnlField`]), so both the
/// b-vector and (through the norm) the b-value deviate per voxel. The global vertex×gradient
/// response table cannot be shared any more, so each voxel evaluates its nonzero histogram bins
/// directly — the same shape as the myelin branch of [`signal_from_mixture`]. A `None` field
/// reproduces [`signal_from_mixture`] bit for bit through the same code path.
pub fn signal_from_mixture_gnl(
    mix: &crate::mixture::MixtureField,
    scheme: &GradientScheme,
    field: Option<&crate::gnl::GnlField>,
) -> Compartments {
    let Some(field) = field else { return signal_from_mixture(mix, scheme) };
    assert_eq!(field.dims, mix.dims, "GNL field is not on the mixture grid");
    let ngrad = scheme.len();
    let nvox = mix.nvox();
    let p = mix.params;
    let t2 = vec![p.t2_fiber, p.t2_gm, p.t2_csf];
    if ngrad == 0 {
        return Compartments { dims: mix.dims, ngrad, images: vec![Vec::new(); 3], t2 };
    }
    let grads = scheme.fiberfox_gradients();
    let gm = Ball { b_value: p.b_value, diffusivity: p.d_gm };
    let csf = Ball { b_value: p.b_value, diffusivity: p.d_csf };
    let soma = Ball { b_value: p.b_value, diffusivity: p.d_soma };
    let adult = CompartmentParams::adult();
    let md = mix.md_fallback();

    let voxel = |vox: usize, fib: &mut [f32], gmo: &mut [f32], cso: &mut [f32]| {
        if !mix.is_masked(vox) {
            return;
        }
        let (wf, gf, cf) = (mix.wm[vox] as f64, mix.gm[vox] as f64, mix.csf[vox] as f64);
        let row = mix.odf_row(vox);
        let nz: Vec<(usize, f64)> =
            row.iter().enumerate().filter(|(_, &w)| w != 0.0).map(|(v, &w)| (v, w as f64)).collect();
        let tot: f64 = nz.iter().map(|&(_, w)| w).sum();
        let m = mix.myelin.as_ref().map(|mm| mm[vox] as f64).unwrap_or(0.0);
        let li = |a: f64, b: f64| a + (b - a) * m;
        let stick_m = Stick { b_value: p.b_value, diffusivity: li(p.d_intra, adult.d_intra) };
        let extra_m = Tensor {
            b_value: p.b_value,
            eigenvalues: (
                li(p.d_extra.0, adult.d_extra.0),
                li(p.d_extra.1, adult.d_extra.1),
                li(p.d_extra.2, adult.d_extra.2),
            ),
        };
        let (fi, fe) = (li(p.intra_frac, adult.intra_frac), li(p.extra_frac, adult.extra_frac));
        let md_m = if m <= 1e-3 {
            md
        } else {
            (extra_m.eigenvalues.0 + extra_m.eigenvalues.1 + extra_m.eigenvalues.2) / 3.0
        };
        for g in 0..ngrad {
            let gv = field.effective_gradient(vox, grads[g]);
            let fiber_resp = if tot > 1e-12 {
                nz.iter()
                    .map(|&(v, w)| {
                        let vert = mix.sphere.verts[v];
                        w * (fi * stick_m.simulate(gv, vert) + fe * extra_m.simulate(gv, vert))
                    })
                    .sum::<f64>()
                    / tot
            } else {
                (-p.b_value * mat::dot(gv, gv) * md_m).exp()
            };
            fib[g] = (wf * fiber_resp) as f32;
            gmo[g] = (gf * ((1.0 - p.gm_restricted_frac) * gm.simulate(gv)
                + p.gm_restricted_frac * soma.simulate(gv))) as f32;
            cso[g] = (cf * csf.simulate(gv)) as f32;
        }
    };

    let mut fiber_img = vec![0.0f32; nvox * ngrad];
    let mut gm_img = vec![0.0f32; nvox * ngrad];
    let mut csf_img = vec![0.0f32; nvox * ngrad];
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        fiber_img
            .par_chunks_mut(ngrad)
            .zip(gm_img.par_chunks_mut(ngrad))
            .zip(csf_img.par_chunks_mut(ngrad))
            .enumerate()
            .for_each(|(vox, ((f, g), c))| voxel(vox, f, g, c));
    }
    #[cfg(not(feature = "par"))]
    {
        let it = fiber_img.chunks_mut(ngrad).zip(gm_img.chunks_mut(ngrad)).zip(csf_img.chunks_mut(ngrad));
        for (vox, ((f, g), c)) in it.enumerate() {
            voxel(vox, f, g, c);
        }
    }
    Compartments { dims: mix.dims, ngrad, images: vec![fiber_img, gm_img, csf_img], t2 }
}

/// Evaluate the clean signal **from** a [`crate::mixture::MixtureField`] — the second half
/// of the histogram-first path. Returns the same
/// fiber/GM/CSF [`Compartments`] as [`generate_compartments`], so the acquisition stage is unaffected.
///
/// Per voxel, `S_fiber(g) = wm · Σ_v odf[v]·(f_intra·stick(g,v) + f_extra·zeppelin(g,v)) / Σ_v
/// odf[v]`, with the vertex responses precomputed once into an `nvert × ngrad` table (the cost
/// shift `O(hits·ngrad) → O(hits) + O(nvox·nvert·ngrad)` of the histogram-first path). An empty row falls back
/// to the hindered isotropic `exp(-b·|g|²·md)` of [`generate_compartments`] — the same formula,
/// flagged by [`crate::mixture::MixtureField::fallback`].
///
/// As everywhere in the signal stage, the caller is expected to have built the mixture with
/// `params.b_value = scheme.b_max` (what the binaries pass): the per-volume b-value rides in the
/// gradient norm, so `scheme` here must be the *same* scheme that set `b_max`.
pub fn signal_from_mixture(mix: &crate::mixture::MixtureField, scheme: &GradientScheme) -> Compartments {
    let ngrad = scheme.len();
    let nvox = mix.nvox();
    let p = mix.params;
    let t2 = vec![p.t2_fiber, p.t2_gm, p.t2_csf];
    if ngrad == 0 {
        return Compartments { dims: mix.dims, ngrad, images: vec![Vec::new(); 3], t2 };
    }
    let grads = scheme.fiberfox_gradients();
    let stick = Stick { b_value: p.b_value, diffusivity: p.d_intra };
    let extra = Tensor { b_value: p.b_value, eigenvalues: p.d_extra };
    let gm = Ball { b_value: p.b_value, diffusivity: p.d_gm };
    let csf = Ball { b_value: p.b_value, diffusivity: p.d_csf };
    let soma = Ball { b_value: p.b_value, diffusivity: p.d_soma };

    // response table: resp[v*ngrad + g] = the WM fiber response of a segment along vertex v
    let mut resp = vec![0.0f64; mix.nvert() * ngrad];
    for (v, vert) in mix.sphere.verts.iter().enumerate() {
        for g in 0..ngrad {
            resp[v * ngrad + g] =
                p.intra_frac * stick.simulate(grads[g], *vert) + p.extra_frac * extra.simulate(grads[g], *vert);
        }
    }
    // GM = (1−f_r)·free ball + f_r·restricted (soma) ball — keeps real GM's high-b signal
    let gm_resp: Vec<f64> = (0..ngrad)
        .map(|g| {
            (1.0 - p.gm_restricted_frac) * gm.simulate(grads[g])
                + p.gm_restricted_frac * soma.simulate(grads[g])
        })
        .collect();
    let csf_resp: Vec<f64> = (0..ngrad).map(|g| csf.simulate(grads[g])).collect();
    let md = mix.md_fallback();
    let iso: Vec<f64> =
        (0..ngrad).map(|g| (-p.b_value * mat::dot(grads[g], grads[g]) * md).exp()).collect();

    // one voxel → its ngrad-long slice of each compartment image (read-only over everything else).
    // With a myelin map, the WM compartment's parameters are lerped per voxel between `mix.params`
    // (unmyelinated) and `CompartmentParams::adult()` (myelinated) — the global response table
    // only serves the m ≈ 0 fast path; myelinated voxels evaluate their nonzero bins directly.
    let adult = CompartmentParams::adult();
    let voxel = |vox: usize, fib: &mut [f32], gmo: &mut [f32], cso: &mut [f32]| {
        if !mix.is_masked(vox) {
            return;
        }
        let (wf, gf, cf) = (mix.wm[vox] as f64, mix.gm[vox] as f64, mix.csf[vox] as f64);
        let row = mix.odf_row(vox);
        let nz: Vec<(usize, f64)> =
            row.iter().enumerate().filter(|(_, &w)| w != 0.0).map(|(v, &w)| (v, w as f64)).collect();
        let tot: f64 = nz.iter().map(|&(_, w)| w).sum();
        let m = mix.myelin.as_ref().map(|mm| mm[vox] as f64).unwrap_or(0.0);
        if m <= 1e-3 {
            for g in 0..ngrad {
                let fiber_resp = if tot > 1e-12 {
                    nz.iter().map(|&(v, w)| w * resp[v * ngrad + g]).sum::<f64>() / tot
                } else {
                    iso[g] // no streamline support → hindered isotropic
                };
                fib[g] = (wf * fiber_resp) as f32;
                gmo[g] = (gf * gm_resp[g]) as f32;
                cso[g] = (cf * csf_resp[g]) as f32;
            }
        } else {
            let li = |a: f64, b: f64| a + (b - a) * m;
            let stick_m = Stick { b_value: p.b_value, diffusivity: li(p.d_intra, adult.d_intra) };
            let extra_m = Tensor {
                b_value: p.b_value,
                eigenvalues: (
                    li(p.d_extra.0, adult.d_extra.0),
                    li(p.d_extra.1, adult.d_extra.1),
                    li(p.d_extra.2, adult.d_extra.2),
                ),
            };
            let (fi, fe) = (li(p.intra_frac, adult.intra_frac), li(p.extra_frac, adult.extra_frac));
            let md_m = (extra_m.eigenvalues.0 + extra_m.eigenvalues.1 + extra_m.eigenvalues.2) / 3.0;
            for g in 0..ngrad {
                let fiber_resp = if tot > 1e-12 {
                    nz.iter()
                        .map(|&(v, w)| {
                            let vert = mix.sphere.verts[v];
                            w * (fi * stick_m.simulate(grads[g], vert)
                                + fe * extra_m.simulate(grads[g], vert))
                        })
                        .sum::<f64>()
                        / tot
                } else {
                    (-p.b_value * mat::dot(grads[g], grads[g]) * md_m).exp()
                };
                fib[g] = (wf * fiber_resp) as f32;
                gmo[g] = (gf * gm_resp[g]) as f32;
                cso[g] = (cf * csf_resp[g]) as f32;
            }
        }
    };

    let mut fiber_img = vec![0.0f32; nvox * ngrad];
    let mut gm_img = vec![0.0f32; nvox * ngrad];
    let mut csf_img = vec![0.0f32; nvox * ngrad];
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        // Disjoint per-voxel chunks of the three outputs. Unlike the segment accumulation there is
        // nothing to duplicate — `resp` and `mix` are shared read-only — so plain `par_chunks_mut`
        // is right here and the buffer-per-chunk dance of `generate_compartments` is not needed.
        fiber_img
            .par_chunks_mut(ngrad)
            .zip(gm_img.par_chunks_mut(ngrad))
            .zip(csf_img.par_chunks_mut(ngrad))
            .enumerate()
            .for_each(|(vox, ((f, g), c))| voxel(vox, f, g, c));
    }
    #[cfg(not(feature = "par"))]
    {
        let it = fiber_img.chunks_mut(ngrad).zip(gm_img.chunks_mut(ngrad)).zip(csf_img.chunks_mut(ngrad));
        for (vox, ((f, g), c)) in it.enumerate() {
            voxel(vox, f, g, c);
        }
    }

    Compartments { dims: mix.dims, ngrad, images: vec![fiber_img, gm_img, csf_img], t2 }
}

/// Motion-aware signal stage — the faithful port of Fiberfox's `SimulateMotion`: for each volume, the
/// streamlines are rigidly transformed by that volume's pose (so segment directions pick up the
/// rotation → the fiber–gradient angle changes correctly), the tissue maps are resampled by the
/// same pose, and the volume's signal is **re-rasterized and re-simulated** from the moved head.
/// Rotation is about the FOV centre; the fieldmap is left in scanner space for the acquisition stage.
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
        let (mut ts_buf, mut hits) = (Vec::with_capacity(16), Vec::with_capacity(16));
        for s in 0..n_streamlines {
            let (lo, hi) = (offsets[s] as usize, offsets[s + 1] as usize);
            for j in lo..hi.saturating_sub(1) {
                let (a, b) = (mv(positions[j]), mv(positions[j + 1]));
                let dir = mat::normalize(mat::sub(b, a));
                if mat::norm(dir) < 0.5 {
                    continue;
                }
                grid.intersect_segment_into(a, b, &mut ts_buf, &mut hits);
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
    use crate::mixture::MixtureField;
    use crate::sphere::HemiSphere;

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

    // --- histogram-first path ---

    const N: usize = 6;
    /// b0, b1000 ∥x (∥ fiber), b1000 ∥y (⊥ fiber).
    const BVAL: &str = "0 1000 1000";
    const BVEC: &str = "0 1 0\n0 0 1\n0 0 0";

    #[inline]
    fn fl(x: usize, y: usize, z: usize) -> usize {
        x + N * (y + N * z)
    }

    /// The single-fiber fixture: 6³ 1 mm grid, one straight x-aligned streamline through the
    /// y=z=2 voxel row, WM=1 + mask=1 along it. `extra_wm` adds WM voxels with no streamline.
    fn fiber_fixture(extra_wm: &[[usize; 3]]) -> (Grid, Vec<[f64; 3]>, Vec<u32>, TissueFractions) {
        let nv = N * N * N;
        let (mut wm, gm, csf, mut mask) = (vec![0f32; nv], vec![0f32; nv], vec![0f32; nv], vec![0u8; nv]);
        for x in 0..N {
            wm[fl(x, 2, 2)] = 1.0;
            mask[fl(x, 2, 2)] = 1;
        }
        for v in extra_wm {
            wm[fl(v[0], v[1], v[2])] = 1.0;
            mask[fl(v[0], v[1], v[2])] = 1;
        }
        let tissue = TissueFractions { dims: [N, N, N], wm, gm, csf, mask };
        (grid_1mm(N), vec![[0.2, 2.5, 2.5], [5.8, 2.5, 2.5]], vec![0u32, 2], tissue)
    }

    #[test]
    fn mixture_single_fiber_matches_per_segment_path() {
        let (grid, positions, offsets, tissue) = fiber_fixture(&[]);
        let scheme = GradientScheme::from_str(BVAL, BVEC).unwrap();
        let params = CompartmentParams { b_value: 1000.0, ..Default::default() };

        let want = generate_clean_signal(&grid, &positions, &offsets, &tissue, &scheme, &params);
        let mix =
            generate_mixture(&grid, &positions, &offsets, None, &tissue, &params, None, HemiSphere::icosphere(3));
        let got = signal_from_mixture(&mix, &scheme).mixed();

        // the fiber is ∥x, which IS an icosphere vertex, so binning is exact here
        let v = fl(3, 2, 2) * got.ngrad;
        for g in 0..3 {
            let (a, b) = (got.data[v + g], want.data[v + g]);
            assert!((a - b).abs() < 1e-3, "grad {g}: from-mixture {a} vs per-segment {b}");
        }
        let (s_b0, s_par, s_perp) = (got.data[v], got.data[v + 1], got.data[v + 2]);
        assert!((s_b0 - 1.0).abs() < 1e-4, "b0 should be ~1, got {s_b0}");
        assert!(s_par < s_perp, "∥-fiber should attenuate more: {s_par} vs {s_perp}");
        assert_eq!(got.data[fl(3, 4, 4) * got.ngrad], 0.0, "off-fiber voxel stays empty");
    }

    #[test]
    fn streamline_weights_scale_the_histogram() {
        let (grid, positions, _, tissue) = fiber_fixture(&[]);
        let params = CompartmentParams::default();
        let mix = |pos: &[[f64; 3]], off: &[u32], w: &[f32]| {
            generate_mixture(&grid, pos, off, Some(w), &tissue, &params, None, HemiSphere::icosphere(3))
        };
        // two identical streamlines at weight 1 == one streamline at weight 2
        let twice = [positions[0], positions[1], positions[0], positions[1]];
        let a = mix(&twice, &[0, 2, 4], &[1.0, 1.0]);
        let b = mix(&positions, &[0, 2], &[2.0]);
        assert!(a.odf.iter().any(|&x| x > 0.0), "histogram must not be empty");
        for (i, (x, y)) in a.odf.iter().zip(b.odf.iter()).enumerate() {
            assert!((x - y).abs() <= 1e-6 * x.abs().max(1.0), "bin {i}: {x} vs {y}");
        }
        // weight 0 contributes nothing at all — the WM voxel then has no streamline support
        let z = mix(&positions, &[0, 2], &[0.0]);
        assert!(z.odf.iter().all(|&x| x == 0.0), "zero-weight streamline must not accumulate");
        assert_eq!(z.fallback[fl(3, 2, 2)], 1);
    }

    #[test]
    fn watson_kappa_spreads_orientation_weight() {
        let (grid, positions, offsets, tissue) = fiber_fixture(&[]);
        let scheme = GradientScheme::from_str(BVAL, BVEC).unwrap();
        let params = CompartmentParams { b_value: 1000.0, ..Default::default() };
        let vox = fl(3, 2, 2);
        let mix = |k: Option<f64>| {
            generate_mixture(&grid, &positions, &offsets, None, &tissue, &params, k, HemiSphere::icosphere(3))
        };
        let sig = |m: &MixtureField| {
            let d = signal_from_mixture(m, &scheme).mixed();
            [d.data[vox * 3], d.data[vox * 3 + 1], d.data[vox * 3 + 2]]
        };
        let nonzero = |m: &MixtureField| m.odf_row(vox).iter().filter(|&&x| x > 0.0).count();
        let total = |m: &MixtureField| m.odf_row(vox).iter().map(|&x| x as f64).sum::<f64>();

        let (sharp, disp) = (mix(None), mix(Some(20.0)));
        assert_eq!(nonzero(&sharp), 1, "nearest-vertex binning is a delta");
        assert!(nonzero(&disp) > nonzero(&sharp), "κ=20 must spread, got {} bins", nonzero(&disp));
        // the kernel is normalized, so the row sum is κ-independent (1e-6, not 1e-9: `odf` is f32)
        let (t0, t1) = (total(&sharp), total(&disp));
        assert!((t1 - t0).abs() <= 1e-6 * t0, "row sum must not change with κ: {t1} vs {t0}");

        let (a, b) = (sig(&sharp), sig(&disp));
        assert!(b[2] - b[1] < a[2] - a[1], "dispersion lowers anisotropy: {b:?} vs {a:?}");
        // κ → ∞ collapses back onto the aligned vertex
        let c = sig(&mix(Some(1e6)));
        for g in 0..3 {
            assert!((c[g] - a[g]).abs() < 1e-6, "grad {g}: κ=1e6 gives {} vs binned {}", c[g], a[g]);
        }
    }

    #[test]
    fn fallback_flags_wm_voxels_without_streamlines() {
        let (grid, positions, offsets, tissue) = fiber_fixture(&[[3, 4, 4]]);
        let scheme = GradientScheme::from_str(BVAL, BVEC).unwrap();
        let params = CompartmentParams { b_value: 1000.0, ..Default::default() };
        let mix =
            generate_mixture(&grid, &positions, &offsets, None, &tissue, &params, None, HemiSphere::icosphere(3));
        let (on, off) = (fl(3, 2, 2), fl(3, 4, 4));
        assert_eq!(mix.fallback[on], 0, "the fiber voxel has streamline support");
        assert_eq!(mix.fallback[off], 1, "WM with no streamline must be flagged");

        let d = signal_from_mixture(&mix, &scheme).mixed();
        let s = [d.data[off * 3], d.data[off * 3 + 1], d.data[off * 3 + 2]];
        let md = (params.d_extra.0 + params.d_extra.1 + params.d_extra.2) / 3.0;
        let want = (-params.b_value * md).exp() as f32; // |g|² = 1 on the b=1000 shell
        assert!((s[0] - 1.0).abs() < 1e-6, "b0 unattenuated: {}", s[0]);
        assert!((s[1] - s[2]).abs() < 1e-7, "hindered fallback is isotropic: {} vs {}", s[1], s[2]);
        assert!(s[1] < 1.0 && (s[1] - want).abs() < 1e-6, "exp(-b·|g|²·md): {} vs {want}", s[1]);
    }
    #[test]
    fn an_identity_gnl_field_reproduces_signal_from_mixture() {
        use crate::gnl::{GnlField, GradCoef, Vendor};
        let (grid, positions, offsets, tissue) = fiber_fixture(&[[3, 4, 4]]);
        let scheme = GradientScheme::from_str(BVAL, BVEC).unwrap();
        let params = CompartmentParams { b_value: 1000.0, ..Default::default() };
        let mix =
            generate_mixture(&grid, &positions, &offsets, None, &tissue, &params, None, HemiSphere::icosphere(3));
        // no nonlinear terms: d ≡ 0, J ≡ I
        let identity = GnlField::on_grid(&GradCoef { r0_mm: 250.0, vendor: Vendor::Siemens, terms: Vec::new() }, &grid);
        let a = signal_from_mixture(&mix, &scheme);
        let b = signal_from_mixture_gnl(&mix, &scheme, Some(&identity));
        assert_eq!(a.t2, b.t2);
        for (c, (x, y)) in a.images.iter().zip(&b.images).enumerate() {
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                assert!((p - q).abs() <= 1e-6 * p.abs().max(1e-6), "compartment {c} sample {i}: {p} vs {q}");
            }
        }
    }

    #[test]
    fn apply_s0_scales_each_compartment_independently() {
        let mut c = Compartments {
            dims: [1, 1, 1],
            ngrad: 1,
            images: vec![vec![2.0f32], vec![3.0f32], vec![5.0f32]],
            t2: vec![68.0, 76.0, 2000.0],
        };
        c.apply_s0([1.0, 0.5, 0.0]);
        assert_eq!(c.images[0][0], 2.0); // fiber unchanged (factor 1.0)
        assert_eq!(c.images[1][0], 1.5); // gm halved
        assert_eq!(c.images[2][0], 0.0); // csf zeroed
        // the mixed signal is the sum of the scaled compartments
        assert_eq!(c.mixed().data[0], 3.5);
    }

    #[test]
    fn mixture_from_fibers_matches_a_rasterized_single_segment() {
        // One streamline segment along +x through one voxel vs. an explicit fibre with the same
        // orientation mass: identical histograms, fractions and fallback flags.
        use crate::sphere::HemiSphere;
        let dims = [3usize, 3, 1];
        let grid = Grid { dims, voxel_to_world: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] };
        let nvox = 9;
        let tissue = TissueFractions {
            dims, wm: vec![0.7; nvox], gm: vec![0.2; nvox], csf: vec![0.1; nvox], mask: vec![1; nvox],
        };
        let params = CompartmentParams { b_value: 1000.0, ..CompartmentParams::adult() };
        // segment from (1.0, 1.5, 0.5) to (2.0, 1.5, 0.5): 1 mm inside voxel (1,1,0)
        let positions = vec![[1.0, 1.5, 0.5], [2.0, 1.5, 0.5]];
        let offsets = vec![0u32, 2];
        let a = generate_mixture(&grid, &positions, &offsets, None, &tissue, &params, None, HemiSphere::icosphere(3));
        let seg_area = PI * params.fiber_radius_mm * params.fiber_radius_mm;
        let b = mixture_from_fibers(dims, &[4], &[[1.0, 0.0, 0.0]], &[1.0 * seg_area], &tissue, &params, None, HemiSphere::icosphere(3));
        assert_eq!(a.odf, b.odf);
        assert_eq!(a.wm, b.wm);
        assert_eq!(a.fallback, b.fallback);
        assert_eq!(b.fallback.iter().filter(|&&f| f == 1).count(), 8, "every WM voxel without a fibre falls back");
        let sa = signal_from_mixture(&a, &GradientScheme::from_str("0 1000 1000", "0 1 0\n0 0 1\n0 0 0").unwrap());
        let sb = signal_from_mixture(&b, &GradientScheme::from_str("0 1000 1000", "0 1 0\n0 0 1\n0 0 0").unwrap());
        assert_eq!(sa.images, sb.images);
    }
}
