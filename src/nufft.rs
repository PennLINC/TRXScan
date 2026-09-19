//! One-dimensional type-1 NUFFT on a periodic interval — nonuniform sources, uniform integer
//! frequencies:
//!
//! ```text
//! F[k] = Σ_j w_j · exp(+i·2π·k·v_j / n),     k = k_lo .. k_lo + n_out,   v_j ∈ ℝ (period n)
//! ```
//!
//! This is the y-sum of the k-space forward model once the fieldmap is read as geometry: with the
//! readout time affine in the PE line, `fmap(r)·t(ky)` is a displacement of the source along y,
//! `v = (y − c) + sny·τ·fmap`, and the per-line sum over voxels becomes one transform of the
//! displaced object (`kspace.rs`). Gridding after Greengard & Lee (2004) with the "exponential of
//! semicircle" kernel of Barnett, Magland & af Klinteberg (2019) at oversampling σ = 2: each source
//! is spread onto `w` fine cells, one FFT (`rustfft`), and the kernel's own transform is divided
//! out. The accuracy is set by `w` (≈ 10^{−(w−1)}); it is not an expansion whose cost grows with
//! the fieldmap range. Several weight vectors sharing the same positions are transformed
//! together, so the kernel is evaluated once per source. Pinned against the literal sum by
//! `matches_the_direct_sum`.
#![cfg(feature = "kspace")]

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::TAU;
use std::sync::Arc;

pub struct Nufft1 {
    n: usize,
    nf: usize,
    w: usize,
    beta: f64,
    k_lo: i64,
    n_out: usize,
    fft: Arc<dyn Fft<f64>>,
    scratch: Vec<Complex<f64>>,
    /// 1/ψ̂(k) for the `n_out` output frequencies.
    corr: Vec<f64>,
    /// Kernel values of the current source, reused across weight vectors.
    kern: Vec<f64>,
    grid: Vec<Vec<Complex<f64>>>,
}

impl Nufft1 {
    /// Period `n` (cells), outputs at integer frequencies `k_lo .. k_lo + n_out` (which must lie
    /// within `[-n/2, n/2]`), tolerance `tol` (relative to the total weight; 1e-12 → w = 13).
    pub fn new(n: usize, k_lo: i64, n_out: usize, tol: f64) -> Nufft1 {
        assert!(n_out > 0 && k_lo >= -(n as i64) / 2 && k_lo + n_out as i64 - 1 <= (n as i64) / 2,
            "output band must sit inside the coarse Nyquist band");
        let w = ((-tol.log10()).ceil() as usize + 1).clamp(2, 16);
        let beta = 2.30 * w as f64;
        let nf = 2 * n;
        let fft = FftPlanner::<f64>::new().plan_fft_inverse(nf);
        let scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        // ψ̂(k) = Σ_m ψ(m) e^{i2π k m/nf} over the kernel's support (real, symmetric).
        let half = w as f64 / 2.0;
        let psi = |z: f64| if z.abs() < 1.0 { (beta * ((1.0 - z * z).sqrt() - 1.0)).exp() } else { 0.0 };
        let mut corr = Vec::with_capacity(n_out);
        for j in 0..n_out {
            let k = (k_lo + j as i64) as f64;
            let mut s = psi(0.0);
            let mut m = 1.0;
            while m < half {
                s += 2.0 * psi(m / half) * (TAU * k * m / nf as f64).cos();
                m += 1.0;
            }
            corr.push(1.0 / s);
        }
        Nufft1 { n, nf, w, beta, k_lo, n_out, fft, scratch, corr, kern: vec![0.0; w], grid: Vec::new() }
    }

    /// `pos[j]` in cells (any real; periodic modulo `n`). `weights`: any number of `(re, im)`
    /// vectors of the same length as `pos`. `out[q]` receives the `n_out` complex outputs of
    /// weight vector `q` as `(re, im)`.
    pub fn run(&mut self, pos: &[f64], weights: &[(&[f64], &[f64])], out: &mut [(Vec<f64>, Vec<f64>)]) {
        let nw = weights.len();
        assert_eq!(out.len(), nw);
        let (nf, w, half) = (self.nf, self.w, self.w as f64 / 2.0);
        self.grid.resize_with(nw, Vec::new);
        for g in self.grid.iter_mut() {
            g.clear();
            g.resize(nf, Complex::new(0.0, 0.0));
        }
        let beta = self.beta;
        for (j, &v) in pos.iter().enumerate() {
            // fine-grid position, wrapped into [0, nf)
            let p = (2.0 * v).rem_euclid(nf as f64);
            let m0 = (p - half).floor() as i64 + 1;
            for i in 0..w {
                let z = (m0 as f64 + i as f64 - p) / half;
                self.kern[i] = if z.abs() < 1.0 { (beta * ((1.0 - z * z).sqrt() - 1.0)).exp() } else { 0.0 };
            }
            for (q, &(re, im)) in weights.iter().enumerate() {
                let (wr, wi) = (re[j], im[j]);
                let g = &mut self.grid[q];
                for i in 0..w {
                    let m = (m0 + i as i64).rem_euclid(nf as i64) as usize;
                    g[m].re += wr * self.kern[i];
                    g[m].im += wi * self.kern[i];
                }
            }
        }
        for (q, g) in self.grid.iter_mut().enumerate() {
            self.fft.process_with_scratch(g, &mut self.scratch);
            let (ore, oim) = &mut out[q];
            ore.clear();
            oim.clear();
            for j in 0..self.n_out {
                let bin = (self.k_lo + j as i64).rem_euclid(nf as i64) as usize;
                ore.push(g[bin].re * self.corr[j]);
                oim.push(g[bin].im * self.corr[j]);
            }
        }
    }

    pub fn n(&self) -> usize { self.n }
    pub fn kernel_width(&self) -> usize { self.w }
}

/// Reference: the literal sum, O(n_src · n_out).
#[cfg(test)]
fn direct(n: usize, k_lo: i64, n_out: usize, pos: &[f64], re: &[f64], im: &[f64]) -> Vec<(f64, f64)> {
    (0..n_out)
        .map(|j| {
            let k = (k_lo + j as i64) as f64;
            let (mut ar, mut ai) = (0.0, 0.0);
            for (jj, &v) in pos.iter().enumerate() {
                let ph = TAU * k * v / n as f64;
                let (c, s) = (ph.cos(), ph.sin());
                ar += re[jj] * c - im[jj] * s;
                ai += re[jj] * s + im[jj] * c;
            }
            (ar, ai)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kspace::Rng;

    #[test]
    fn matches_the_direct_sum() {
        let mut rng = Rng(42);
        for &(n, k_lo, n_out, tol) in &[(48usize, -12i64, 24usize, 1e-12), (140, -35, 70, 1e-12), (16, -8, 16, 1e-12), (64, -16, 32, 1e-8)] {
            let nsrc = 3 * n;
            let pos: Vec<f64> = (0..nsrc).map(|_| (rng.unit() - 0.5) * 3.0 * n as f64).collect();
            let re: Vec<f64> = (0..nsrc).map(|_| rng.unit() - 0.5).collect();
            let im: Vec<f64> = (0..nsrc).map(|_| rng.unit() - 0.5).collect();
            let re2: Vec<f64> = (0..nsrc).map(|_| rng.unit()).collect();
            let im2: Vec<f64> = vec![0.0; nsrc];
            let mut nufft = Nufft1::new(n, k_lo, n_out, tol);
            let mut out = vec![(Vec::new(), Vec::new()), (Vec::new(), Vec::new())];
            nufft.run(&pos, &[(&re, &im), (&re2, &im2)], &mut out);
            for (q, (r, i)) in [(&re, &im), (&re2, &im2)].iter().enumerate() {
                let want = direct(n, k_lo, n_out, &pos, r, i);
                let total: f64 = r.iter().zip(i.iter()).map(|(a, b)| a.hypot(*b)).sum();
                let worst = want.iter().zip(out[q].0.iter().zip(&out[q].1))
                    .map(|(&(wr, wi), (&or, &oi))| (wr - or).hypot(wi - oi)).fold(0.0, f64::max);
                assert!(worst <= 10.0 * tol * total, "n={n} q={q}: max err {worst:.3e} vs {:.3e} allowed (total {total:.3e})", 10.0 * tol * total);
            }
        }
    }

    #[test]
    fn integer_positions_reduce_to_the_dft() {
        // Sources on the grid: the NUFFT must reproduce a plain DFT of the periodic sequence.
        let n = 32;
        let pos: Vec<f64> = (0..n).map(|y| y as f64).collect();
        let re: Vec<f64> = (0..n).map(|y| ((y as f64) * 0.3).sin()).collect();
        let im = vec![0.0; n];
        let mut nufft = Nufft1::new(n, -16, 32, 1e-12);
        let mut out = vec![(Vec::new(), Vec::new())];
        nufft.run(&pos, &[(&re, &im)], &mut out);
        let want = direct(n, -16, 32, &pos, &re, &im);
        for (j, &(wr, wi)) in want.iter().enumerate() {
            assert!((wr - out[0].0[j]).abs() < 1e-10 && (wi - out[0].1[j]).abs() < 1e-10, "k index {j}");
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::kspace::Rng;

    /// Timing only: one slice-column's worth of work at production size (sny=256, ny=128, 6 weight
    /// vectors), split into spreading and FFT. Run with `--ignored --nocapture`.
    #[test]
    #[ignore = "timing micro-benchmark; run explicitly with --ignored"]
    fn bench_column() {
        let mut rng = Rng(7);
        let (n, n_out, nw, cols) = (256usize, 128usize, 6usize, 170usize);
        let pos: Vec<f64> = (0..n).map(|y| y as f64 - 128.0 + 30.0 * (rng.unit() - 0.5)).collect();
        let ws: Vec<(Vec<f64>, Vec<f64>)> = (0..nw).map(|_| ((0..n).map(|_| rng.unit()).collect(), (0..n).map(|_| rng.unit()).collect())).collect();
        let weights: Vec<(&[f64], &[f64])> = ws.iter().map(|(a, b)| (a.as_slice(), b.as_slice())).collect();
        let mut nufft = Nufft1::new(n, -64, n_out, 1e-13);
        let mut out: Vec<(Vec<f64>, Vec<f64>)> = (0..nw).map(|_| (Vec::new(), Vec::new())).collect();
        let t = std::time::Instant::now();
        for _ in 0..cols { nufft.run(&pos, &weights, &mut out); }
        let per_slice = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("NUFFT y-stage per slice ({cols} columns, w={}): {per_slice:.2} ms", nufft.kernel_width());
    }
}
