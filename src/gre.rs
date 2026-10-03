//! Dual-echo gradient-echo (GRE) fieldmap synthesis from the same object the DWI is simulated
//! from: per-echo magnitudes as T2-weighted compartment sums at TE1/TE2, phase from the
//! off-resonance field (`2π·f·TE`), written the way a Siemens scanner stores phase (integers
//! 0..4095 over −π..π) so qsiprep's and FSL's fieldmap routes consume it unchanged. Optional
//! complex noise on both echoes makes the magnitudes Rician and the background phase wrap
//! uniformly, which the min/max-based Siemens phase conversion downstream relies on.
//!
//! Pure std. `io::write_gre_fieldmap` writes the result as BIDS `magnitude1/2` + `phasediff`
//! (or `phase1/2`) with `B0FieldIdentifier` sidecars.

use crate::gnl::GnlField;
use crate::kspace::Rng;
use crate::raster::Grid;

/// What phase representation the GRE fieldmap carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum GreOutput {
    /// Single phase-difference image (`phasediff`): the receiver phase cancels, so it is smooth
    /// (few wraps) — the compact BIDS GRE fieldmap.
    Phasediff,
    /// Both individual echo phases (`phase1`/`phase2`): each carries the smooth receiver phase φ₀
    /// plus 2π·f·TE, so they show the fringe wrapping a raw GRE phase image has (what PRELUDE
    /// unwraps). φ₀ cancels when the consumer differences them, so the field is identical.
    Phase,
}

/// The GRE protocol and output choices.
#[derive(Debug, Clone)]
pub struct GreParams {
    /// Echo times (s). Default: the Siemens `gre_field_mapping` 4.92 / 7.38 ms.
    pub te_s: [f64; 2],
    /// Tissue SNR of the echoes (complex Gaussian noise on both; 0 = noiseless). Real
    /// phase-difference images span their whole integer range because the background phase is
    /// uniformly random, and qsiprep's Siemens phase conversion relies on that (it maps the
    /// image's min/max onto −π..π); a noiseless phasediff would get stretched, so at 0 the two
    /// corner voxels are stamped to the extremes instead.
    pub snr: f64,
    /// Output resolution (isotropic mm). `None`: the DWI acquisition grid. With a value the GRE is
    /// generated at that resolution by complex-averaging the fine-grid signal onto the coarser
    /// grid, so steep-field voxels dephase within the voxel (the intravoxel dephasing a real
    /// low-resolution fieldmap shows) instead of sampling the field pointwise.
    pub res_mm: Option<f64>,
    /// How the SNR scales with voxel volume relative to the DWI acquisition voxel: SNR ∝ V^exp,
    /// i.e. σ ∝ (V_dwi / V_gre)^exp. 1.0 = fixed total scan time and FOV (the textbook relation);
    /// 0.5 = fixed number of averages; 0 = constant σ. `snr` is the SNR at the DWI voxel, so the
    /// default grid is unaffected.
    pub snr_vol_exp: f64,
    pub output: GreOutput,
    /// Peak amplitude (rad) of the smooth receiver/transmit phase φ₀ added to the individual echo
    /// phases (`GreOutput::Phase`). Larger → more fringes. Cancels in the phase difference.
    pub rx_phase_rad: f64,
    /// BIDS `B0FieldIdentifier` written to every sidecar; the DWI carries it as `B0FieldSource`.
    pub b0_field: String,
    /// Repetition time (s) of the spoiled GRE. With `flip_deg` and `t1_ms` it sets the T1
    /// steady state `sin α (1 − E1) / (1 − cos α · E1)`, `E1 = exp(−TR/T1)`, per compartment —
    /// the WM > GM ≫ CSF contrast of a real fieldmap magnitude (Siemens `gre_field_mapping`:
    /// TR ≈ 0.4–0.7 s, FA 60°). Without it the echoes at TE ≈ 5 ms collapse to the tissue
    /// fraction sum (a flat head), which gives magnitude→EPI registrations nothing to lock onto.
    pub tr_s: f64,
    /// Flip angle (degrees).
    pub flip_deg: f64,
    /// Per-compartment T1 (ms) `[fiber, gm, csf]`; default adult 3 T (Wansapura 1999: WM 830,
    /// GM 1330; CSF ~4000).
    pub t1_ms: [f32; 3],
}

impl Default for GreParams {
    fn default() -> Self {
        GreParams {
            te_s: [4.92e-3, 7.38e-3],
            snr: 50.0,
            res_mm: None,
            snr_vol_exp: 1.0,
            output: GreOutput::Phasediff,
            rx_phase_rad: 6.0,
            b0_field: "b0gre".to_string(),
            tr_s: 0.5,
            flip_deg: 60.0,
            t1_ms: [830.0, 1330.0, 4000.0],
        }
    }
}

/// Spoiled-GRE steady-state factor per compartment: `sin α (1 − E1) / (1 − cos α · E1)` with
/// `E1 = exp(−TR/T1)`. `tr_s = ∞` with `flip_deg = 90` gives 1 = pure proton density.
pub fn steady_state(p: &GreParams) -> [f64; 3] {
    let a = p.flip_deg.to_radians();
    let mut out = [1.0f64; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let e1 = (-(p.tr_s * 1000.0) / p.t1_ms[c] as f64).exp();
        *o = a.sin() * (1.0 - e1) / (1.0 - a.cos() * e1);
    }
    out
}

/// The object the GRE is synthesized from, on the signal (simulation) grid.
pub struct GreObject<'a> {
    /// Signal grid: `o×` finer in-plane than `acq_grid`.
    pub sig_grid: &'a Grid,
    /// The DWI acquisition grid (the default output grid).
    pub acq_grid: &'a Grid,
    /// Tissue fractions `[fiber, gm, csf]` on `sig_grid`.
    pub fractions: [&'a [f32]; 3],
    /// Per-compartment amplitude `[fiber, gm, csf]` (the DWI's `--tissue-s0`).
    pub s0: [f32; 3],
    /// Per-compartment T2 (ms).
    pub t2_ms: &'a [f32],
    /// Off-resonance field (Hz) on `sig_grid`, already in the apparent (GNL-warped) frame.
    pub fmap_hz: &'a [f32],
    /// The gradient-nonlinearity warp the DWI saw, if any: `(field on sig_grid, Jacobian
    /// modulation)`.
    pub warp: Option<(&'a GnlField, bool)>,
    /// The DWI acquisition's nominal signal scale (`Acquisition::signal_scale`).
    pub signal_scale: f64,
    pub seed: u64,
}

/// A synthesized GRE fieldmap.
#[derive(Debug, Clone)]
pub struct GreFieldmap {
    pub grid: Grid,
    /// `magnitude1`, `magnitude2` (layout `x + nx*(y + ny*z)`).
    pub magnitude: [Vec<f32>; 2],
    /// `[phasediff]` or `[phase1, phase2]`, Siemens integers 0..4095.
    pub phase: Vec<Vec<i16>>,
    pub te_s: [f64; 2],
    pub output: GreOutput,
    pub b0_field: String,
    /// Per-component noise SD applied to the echoes.
    pub sigma: f64,
    /// Whether the corner voxels of a noiseless phasediff were stamped to the range extremes.
    pub stamped: bool,
}

/// Synthesize the GRE from the object. Magnitudes come from the tissue fractions with the
/// compartment T1 steady state (`steady_state`) and T2 decay (and the same GNL warp as the DWI);
/// the phase from the (already warped) field.
pub fn synthesize(obj: &GreObject, p: &GreParams) -> GreFieldmap {
    let sig = obj.sig_grid;
    let o = (sig.dims[0] / obj.acq_grid.dims[0].max(1)).max(1);
    let [te1, te2] = p.te_s;
    let pi = std::f64::consts::PI;
    let tau = std::f64::consts::TAU;
    let nvox_sig = sig.dims.iter().product::<usize>();
    assert_eq!(obj.fmap_hz.len(), nvox_sig, "fieldmap is not on the signal grid");
    assert_eq!(obj.t2_ms.len(), 3, "three compartment T2s");

    // per-echo magnitude on the fine signal grid (compartment T1 steady state × T2 decay + the
    // same GNL warp)
    let ss = steady_state(p);
    let mut fine_mag: Vec<Vec<f32>> = Vec::new();
    for te in [te1, te2] {
        let mut m = vec![0.0f32; nvox_sig];
        for (v, mv) in m.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for (c, frac) in obj.fractions.iter().enumerate() {
                acc += frac[v] as f64 * obj.s0[c] as f64 * ss[c] * (-(te * 1000.0) / obj.t2_ms[c] as f64).exp();
            }
            *mv = (acc * obj.signal_scale) as f32;
        }
        if let Some((field, modulate)) = obj.warp {
            m = field.warp_volume(&m, modulate);
        }
        fine_mag.push(m);
    }
    // With `Phase` output, each echo carries a smooth receiver phase φ₀ on top of the field term
    // (2π·f·TE), producing the fringe wrapping of a raw GRE phase image. φ₀ cancels in the phase
    // difference, so the recovered field is unchanged.
    let phase_mode = p.output == GreOutput::Phase;
    let phi0 = if phase_mode { smooth_rx_phase(sig, p.rx_phase_rad) } else { Vec::new() };
    let phase_of = |v: usize, te: f64| -> f64 {
        (if phase_mode { phi0[v] as f64 } else { 0.0 }) + tau * obj.fmap_hz[v] as f64 * te
    };
    // Pre-noise complex echoes (real, imag) on the GRE grid.
    let (gre_grid, s1re, s1im, s2re, s2im) = if let Some(res) = p.res_mm {
        // Independent isotropic grid: box-average the fine COMPLEX signal, each echo carrying
        // its own phase 2π·f·TE, so a coarse voxel spanning a steep gradient loses magnitude
        // and gets an averaged phase — the intravoxel dephasing a real low-res fieldmap shows.
        let tg = target_grid(sig, res);
        let nt = tg.dims.iter().product::<usize>();
        let m = &sig.voxel_to_world;
        let sp = |c: usize| (m[0][c] * m[0][c] + m[1][c] * m[1][c] + m[2][c] * m[2][c]).sqrt();
        let ratio = [sp(0) / res, sp(1) / res, sp(2) / res];
        let (mut a1r, mut a1i, mut a2r, mut a2i, mut cnt) =
            (vec![0.0f64; nt], vec![0.0f64; nt], vec![0.0f64; nt], vec![0.0f64; nt], vec![0.0f64; nt]);
        let (sx, sy, sz) = (sig.dims[0], sig.dims[1], sig.dims[2]);
        let map = |idx: usize, r: f64, n: usize| (((idx as f64 + 0.5) * r - 0.5).round() as isize).clamp(0, n as isize - 1) as usize;
        for k in 0..sz {
            for j in 0..sy {
                for i in 0..sx {
                    let vf = i + sx * (j + sy * k);
                    let vt = map(i, ratio[0], tg.dims[0]) + tg.dims[0] * (map(j, ratio[1], tg.dims[1]) + tg.dims[1] * map(k, ratio[2], tg.dims[2]));
                    let (p1, p2) = (phase_of(vf, te1), phase_of(vf, te2));
                    a1r[vt] += fine_mag[0][vf] as f64 * p1.cos();
                    a1i[vt] += fine_mag[0][vf] as f64 * p1.sin();
                    a2r[vt] += fine_mag[1][vf] as f64 * p2.cos();
                    a2i[vt] += fine_mag[1][vf] as f64 * p2.sin();
                    cnt[vt] += 1.0;
                }
            }
        }
        let norm = |a: &[f64]| -> Vec<f32> { a.iter().zip(&cnt).map(|(&x, &n)| if n > 0.0 { (x / n) as f32 } else { 0.0 }).collect() };
        (tg, norm(&a1r), norm(&a1i), norm(&a2r), norm(&a2i))
    } else if phase_mode {
        // Phase mode on the DWI acquisition grid: build each echo's full complex signal
        // (φ₀ + 2π·f·TE) on the fine grid, then in-plane block-average the real/imag parts
        // (complex intravoxel dephasing), same grid as the default phasediff output.
        let build = |mag: &[f32], te: f64| -> (Vec<f32>, Vec<f32>) {
            let (mut re, mut im) = (vec![0.0f32; nvox_sig], vec![0.0f32; nvox_sig]);
            for v in 0..nvox_sig {
                let ph = phase_of(v, te);
                re[v] = (mag[v] as f64 * ph.cos()) as f32;
                im[v] = (mag[v] as f64 * ph.sin()) as f32;
            }
            (downsample_inplane(&re, sig.dims, o), downsample_inplane(&im, sig.dims, o))
        };
        let (r1, i1) = build(&fine_mag[0], te1);
        let (r2, i2) = build(&fine_mag[1], te2);
        (obj.acq_grid.clone(), r1, i1, r2, i2)
    } else {
        // Default: the DWI acquisition grid. Echo 1 carries no phase and echo 2 the whole phase
        // difference (2π·f·ΔTE) evaluated on the downsampled field.
        let m1 = downsample_inplane(&fine_mag[0], sig.dims, o);
        let m2 = downsample_inplane(&fine_mag[1], sig.dims, o);
        let fmap_acq = downsample_inplane(obj.fmap_hz, sig.dims, o);
        let n = m1.len();
        let (mut s2re, mut s2im) = (vec![0.0f32; n], vec![0.0f32; n]);
        for v in 0..n {
            let phi = tau * fmap_acq[v] as f64 * (te2 - te1);
            s2re[v] = (m2[v] as f64 * phi.cos()) as f32;
            s2im[v] = (m2[v] as f64 * phi.sin()) as f32;
        }
        (obj.acq_grid.clone(), m1, vec![0.0f32; n], s2re, s2im)
    };
    // Complex noise on each echo: the magnitudes become Rician and the phase difference wraps
    // uniformly wherever there is no signal. The per-voxel signal is a mean of the fine grid
    // (resolution-independent), so cross-resolution SNR realism comes from scaling σ: a larger
    // GRE voxel collects proportionally more signal, SNR ∝ V^exp ⇒ σ ∝ (V_dwi / V_gre)^exp,
    // referenced to the DWI acquisition voxel (V_gre == V_dwi ⇒ scale 1). `snr` is the SNR of
    // the brightest compartment at its steady state, so the T1 weighting does not change it.
    let ss_max = ss.iter().cloned().fold(0.0f64, f64::max).max(f64::MIN_POSITIVE);
    let sigma = if p.snr > 0.0 {
        let det3 = |m: &[[f64; 4]; 4]| (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])).abs();
        let scale = (det3(&obj.acq_grid.voxel_to_world) / det3(&gre_grid.voxel_to_world)).powf(p.snr_vol_exp);
        obj.signal_scale * ss_max / p.snr * scale
    } else {
        0.0
    };
    let mut rng = Rng(0x6E7E_F1E1_D000_0000 ^ obj.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let n = s1re.len();
    let (mut mag1, mut mag2) = (vec![0.0f32; n], vec![0.0f32; n]);
    // Siemens integer phase: [−π, π] → [0, 4095]; both routes qsiprep reads use this scale.
    let siemens = |ph: f64| -> i16 { ((ph + pi) / tau * 4096.0).round().clamp(0.0, 4095.0) as i16 };
    let mut phase: Vec<Vec<i16>> = if phase_mode { vec![Vec::with_capacity(n), Vec::with_capacity(n)] } else { vec![Vec::with_capacity(n)] };
    for v in 0..n {
        let s1 = (s1re[v] as f64 + sigma * rng.gauss(), s1im[v] as f64 + sigma * rng.gauss());
        let s2 = (s2re[v] as f64 + sigma * rng.gauss(), s2im[v] as f64 + sigma * rng.gauss());
        mag1[v] = (s1.0 * s1.0 + s1.1 * s1.1).sqrt() as f32;
        mag2[v] = (s2.0 * s2.0 + s2.1 * s2.1).sqrt() as f32;
        if phase_mode {
            phase[0].push(siemens(s1.1.atan2(s1.0)));
            phase[1].push(siemens(s2.1.atan2(s2.0)));
        } else {
            // arg(s2 · conj(s1)) is already wrapped to (−π, π]
            let dphi = (s2.1 * s1.0 - s2.0 * s1.1).atan2(s2.0 * s1.0 + s2.1 * s1.1);
            phase[0].push(siemens(dphi));
        }
    }
    // Only the phasediff route relies on the image spanning a full turn (min/max rescale); the
    // two-phase route uses a fixed scale, so no stamping is needed.
    let stamped = if phase_mode { false } else { stamp_phasediff_range(&mut phase[0]) };
    GreFieldmap {
        grid: gre_grid,
        magnitude: [mag1, mag2],
        phase,
        te_s: p.te_s,
        output: p.output,
        b0_field: p.b0_field.clone(),
        sigma,
        stamped,
    }
}

/// An isotropic `res`-mm grid covering `sig`'s field of view, sharing its world orientation
/// (direction cosines) and FOV corner. Only the voxel spacing changes, so the output stays in the
/// same world frame the DWI/field live in — and, because the axes are shared, a fine voxel maps to
/// a coarse voxel by a per-axis spacing ratio alone (no affine inversion needed).
pub fn target_grid(sig: &Grid, res: f64) -> Grid {
    let m = &sig.voxel_to_world;
    let mut tv = [[0.0f64; 4]; 4];
    tv[3][3] = 1.0;
    let mut spacing = [0.0f64; 3];
    for c in 0..3 {
        let s = (m[0][c] * m[0][c] + m[1][c] * m[1][c] + m[2][c] * m[2][c]).sqrt();
        spacing[c] = s;
        for r in 0..3 {
            tv[r][c] = m[r][c] / s * res; // unit cosine * new spacing
        }
    }
    let dims = [
        ((spacing[0] * sig.dims[0] as f64 / res).round() as usize).max(1),
        ((spacing[1] * sig.dims[1] as f64 / res).round() as usize).max(1),
        ((spacing[2] * sig.dims[2] as f64 / res).round() as usize).max(1),
    ];
    // Preserve the FOV corner (world coord of voxel index (-0.5,-0.5,-0.5)).
    for r in 0..3 {
        let corner = -0.5 * (m[r][0] + m[r][1] + m[r][2]) + m[r][3];
        tv[r][3] = corner + 0.5 * (tv[r][0] + tv[r][1] + tv[r][2]);
    }
    Grid { dims, voxel_to_world: tv }
}

/// A smooth low-order receiver/transmit phase field (radians) on `sig`, peak ~`amp`. A real GRE's
/// per-echo phase is this plus the field term `2π·f·TE`; φ₀ (not the field) drives most of the
/// fringe wrapping in individual phase images, and cancels exactly in the phase difference.
pub fn smooth_rx_phase(sig: &Grid, amp: f64) -> Vec<f32> {
    let [nx, ny, nz] = sig.dims;
    let mut v = vec![0.0f32; nx * ny * nz];
    let norm = |a: usize, n: usize| 2.0 * a as f64 / (n.max(2) - 1) as f64 - 1.0; // -> [-1, 1]
    for k in 0..nz {
        let w = norm(k, nz);
        for j in 0..ny {
            let vv = norm(j, ny);
            for i in 0..nx {
                let u = norm(i, nx);
                // a smooth ramp + curvature + a couple of cross terms (unit-ish scale)
                let g = u + 0.7 * vv + 0.4 * w + 0.5 * (u * u - 0.5) + 0.4 * (vv * vv - 0.5) + 0.6 * u * vv + 0.3 * vv * w;
                v[i + nx * (j + ny * k)] = (amp * g) as f32;
            }
        }
    }
    v
}

/// Block-average an in-plane oversampled volume (`sim = [nx*o, ny*o, nz]`) down to `[nx, ny, nz]`.
pub fn downsample_inplane(v: &[f32], sim: [usize; 3], o: usize) -> Vec<f32> {
    if o <= 1 {
        return v.to_vec();
    }
    let [snx, sny, nz] = sim;
    let (nx, ny) = (snx / o, sny / o);
    let mut out = vec![0.0f32; nx * ny * nz];
    let norm = 1.0 / (o * o) as f32;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let mut acc = 0.0f32;
                for dy in 0..o {
                    for dx in 0..o {
                        acc += v[(x * o + dx) + snx * ((y * o + dy) + sny * z)];
                    }
                }
                out[x + nx * (y + ny * z)] = acc * norm;
            }
        }
    }
    out
}

/// Make a Siemens-style phase-difference image span its whole 0..4095 range.
///
/// Consumers that recover radians from the image's own min/max (qsiprep's `siemens2rads`,
/// which assumes scanner data whose noisy background wraps uniformly) stretch anything
/// narrower. With noise on the echoes the range is full anyway; without it, two background
/// voxels — the first and the last, always outside the object — are stamped to the extremes.
pub fn stamp_phasediff_range(phasediff: &mut [i16]) -> bool {
    if phasediff.len() < 2 {
        return false;
    }
    let (min, max) = phasediff.iter().fold((i16::MAX, i16::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    if min <= 0 && max >= 4095 {
        return false;
    }
    phasediff[0] = 0;
    let last = phasediff.len() - 1;
    phasediff[last] = 4095;
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(dims: [usize; 3], mm: f64) -> Grid {
        Grid { dims, voxel_to_world: [[mm, 0.0, 0.0, 0.0], [0.0, mm, 0.0, 0.0], [0.0, 0.0, mm, 0.0], [0.0, 0.0, 0.0, 1.0]] }
    }

    #[test]
    fn a_partial_range_gets_the_extremes_stamped_on_the_corner_voxels() {
        let mut v = vec![2048i16; 27];
        v[13] = 3140;
        assert!(stamp_phasediff_range(&mut v));
        assert_eq!((v[0], v[26]), (0, 4095));
        assert_eq!(v[13], 3140);
        assert_eq!(v.iter().filter(|&&x| x == 2048).count(), 24);
    }

    #[test]
    fn a_full_range_is_left_alone() {
        let mut v = vec![2048i16; 27];
        v[1] = 0;
        v[2] = 4095;
        let before = v.clone();
        assert!(!stamp_phasediff_range(&mut v));
        assert_eq!(v, before);
    }

    #[test]
    fn noiseless_phasediff_encodes_two_pi_f_delta_te_in_siemens_integers() {
        // one voxel column of pure WM at 30 Hz; o = 2 in-plane
        let acq = grid([2, 2, 1], 2.0);
        let sig = grid([4, 4, 1], 1.0);
        let n = 16;
        let wm = vec![1.0f32; n];
        let zero = vec![0.0f32; n];
        let fmap = vec![30.0f32; n];
        let obj = GreObject {
            sig_grid: &sig, acq_grid: &acq, fractions: [&wm, &zero, &zero], s0: [1.0; 3],
            t2_ms: &[70.0, 100.0, 2000.0], fmap_hz: &fmap, warp: None, signal_scale: 100.0, seed: 0,
        };
        let p = GreParams { snr: 0.0, tr_s: f64::INFINITY, ..Default::default() };
        let g = synthesize(&obj, &p);
        assert_eq!(g.grid.dims, [2, 2, 1]);
        assert_eq!(g.phase.len(), 1);
        let want = std::f64::consts::TAU * 30.0 * (p.te_s[1] - p.te_s[0]);
        let code = ((want + std::f64::consts::PI) / std::f64::consts::TAU * 4096.0).round() as i16;
        // interior voxels carry the field; the two corners are stamped (noiseless)
        assert_eq!(g.phase[0][1], code);
        assert_eq!(g.phase[0][2], code);
        assert!(g.stamped && g.phase[0][0] == 0 && g.phase[0][3] == 4095);
        assert!(g.phase[0].iter().all(|&v| (0..=4095).contains(&v)));
        // magnitude2 < magnitude1 by exp(−ΔTE/T2)
        let ratio = g.magnitude[1][1] / g.magnitude[0][1];
        let want_ratio = (-(p.te_s[1] - p.te_s[0]) * 1000.0 / 70.0).exp() as f32;
        assert!((ratio - want_ratio).abs() < 1e-5, "{ratio} vs {want_ratio}");
    }

    #[test]
    fn the_steady_state_gives_a_t1_weighted_magnitude() {
        // pure WM / GM / CSF voxels in one row; defaults TR 0.5 s, FA 60°
        let acq = grid([3, 1, 1], 1.0);
        let sig = grid([3, 1, 1], 1.0);
        let wm = vec![1.0f32, 0.0, 0.0];
        let gm = vec![0.0f32, 1.0, 0.0];
        let csf = vec![0.0f32, 0.0, 1.0];
        let fmap = vec![0.0f32; 3];
        let obj = GreObject {
            sig_grid: &sig, acq_grid: &acq, fractions: [&wm, &gm, &csf], s0: [1.0; 3],
            t2_ms: &[70.0, 100.0, 2000.0], fmap_hz: &fmap, warp: None, signal_scale: 100.0, seed: 0,
        };
        let p = GreParams { snr: 0.0, ..Default::default() };
        let ss = steady_state(&p);
        let a = 60f64.to_radians();
        let e1 = (-500.0f64 / 830.0).exp();
        assert!((ss[0] - a.sin() * (1.0 - e1) / (1.0 - a.cos() * e1)).abs() < 1e-12);
        // WM > GM > CSF, and CSF far below WM (a real fieldmap magnitude)
        assert!(ss[0] > ss[1] && ss[1] > ss[2] && ss[2] < 0.4 * ss[0], "{ss:?}");
        let g = synthesize(&obj, &p);
        let m = &g.magnitude[0];
        let want = |c: usize| (100.0 * ss[c] * (-(p.te_s[0] * 1000.0) / [70.0, 100.0, 2000.0][c]).exp()) as f32;
        for c in 0..3 {
            assert!((m[c] - want(c)).abs() < 1e-3 * want(c), "compartment {c}: {} vs {}", m[c], want(c));
        }
        // TR = ∞ at 90° is proton density (the pre-T1 behaviour)
        let pd = steady_state(&GreParams { tr_s: f64::INFINITY, flip_deg: 90.0, ..Default::default() });
        assert!(pd.iter().all(|&v| (v - 1.0).abs() < 1e-12), "{pd:?}");
    }

    #[test]
    fn the_two_phase_route_differences_back_to_the_same_field() {
        let acq = grid([4, 4, 1], 2.0);
        let sig = grid([4, 4, 1], 2.0);
        let n = 16;
        let wm = vec![1.0f32; n];
        let zero = vec![0.0f32; n];
        let fmap: Vec<f32> = (0..n).map(|i| 5.0 * i as f32).collect();
        let obj = GreObject {
            sig_grid: &sig, acq_grid: &acq, fractions: [&wm, &zero, &zero], s0: [1.0; 3],
            t2_ms: &[70.0, 100.0, 2000.0], fmap_hz: &fmap, warp: None, signal_scale: 100.0, seed: 0,
        };
        let p = GreParams { snr: 0.0, output: GreOutput::Phase, tr_s: f64::INFINITY, ..Default::default() };
        let g = synthesize(&obj, &p);
        assert_eq!(g.phase.len(), 2);
        let tau = std::f64::consts::TAU;
        for v in 0..n {
            let to_rad = |c: i16| c as f64 / 4096.0 * tau - std::f64::consts::PI;
            let d = to_rad(g.phase[1][v]) - to_rad(g.phase[0][v]);
            let d = (d + std::f64::consts::PI).rem_euclid(tau) - std::f64::consts::PI;
            let want = tau * fmap[v] as f64 * (p.te_s[1] - p.te_s[0]);
            let want = (want + std::f64::consts::PI).rem_euclid(tau) - std::f64::consts::PI;
            assert!((d - want).abs() < 2.0 * tau / 4096.0, "voxel {v}: {d} vs {want}");
        }
    }

    #[test]
    fn a_coarser_output_grid_keeps_the_fov() {
        let sig = grid([8, 8, 4], 1.0);
        let tg = target_grid(&sig, 2.0);
        assert_eq!(tg.dims, [4, 4, 2]);
        // corner of the FOV is preserved: voxel (-0.5) maps to world -0.5 on both grids
        for r in 0..3 {
            let fine = -0.5 * sig.voxel_to_world[r][r] + sig.voxel_to_world[r][3];
            let coarse = -0.5 * tg.voxel_to_world[r][r] + tg.voxel_to_world[r][3];
            assert!((fine - coarse).abs() < 1e-12);
        }
    }
}
