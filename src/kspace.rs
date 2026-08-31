//! Acquisition stage — per-slice k-space: EPI geometric distortion, T2* relaxation, eddy
//! currents, Nyquist ghosting, partial Fourier, Gibbs ringing, spikes, multi-coil combine, and
//! GRAPPA. Faithful port of the core DFT in `Algorithms/itkKspaceImageFilter.cpp:452`:
//!
//! ```text
//! kspace[kx,ky] = (1/N) Σ_{x,y} f(x,y)·exp( i·2π·( kx·x + ky·y + φ ) ),   φ = fmap(x,y)·t(ky)
//! image[x,y]    =        Σ_{kx,ky} kspace[kx,ky]·exp( -i·2π·( kx·x + ky·y ) )
//! ```
//! `f = Σ_c comp_c · exp(-tRf/T2_c − |t|/tInhom) · signal_scale`. The `φ = fmap·t(ky)` term (the
//! readout time increases with the PE line) is what warps EPI along the phase-encode axis — the
//! same physics that makes AP/PA reverse-PE pairs distort oppositely (the DRBUDDI/topup target).
//!
//! The transform is a direct sum (O(N³) via 1D factoring), std-only and exact — no FFT-convention
//! ambiguity, so it stays directly comparable against Fiberfox. A faster time-segmented FFT path
//! (via `rustfft`, behind the `kspace` feature) is not yet written.

use crate::readout::{Readout, SingleShotEpi};
use std::f64::consts::TAU;

#[derive(Clone, Copy)]
struct C {
    re: f64,
    im: f64,
}
impl C {
    const ZERO: C = C { re: 0.0, im: 0.0 };
    #[inline]
    fn cis(theta: f64) -> C {
        C { re: theta.cos(), im: theta.sin() }
    }
    #[inline]
    fn add(self, o: C) -> C {
        C { re: self.re + o.re, im: self.im + o.im }
    }
    #[inline]
    fn mul(self, o: C) -> C {
        C { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
    #[inline]
    fn scale(self, s: f64) -> C {
        C { re: self.re * s, im: self.im * s }
    }
    #[inline]
    fn abs(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

/// Deterministic Gaussian source (SplitMix64 + Box–Muller), std-only so k-space stays dep-free.
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
    /// standard normal
    fn gauss(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

/// Acquisition parameters for the k-space stage.
#[derive(Debug, Clone)]
pub struct Acquisition {
    pub t_line: f64,      // ms per PE line
    pub t_echo: f64,      // ms
    pub t_inhom: f64,     // ms, T2* inhomogeneity time
    pub signal_scale: f64,
    pub reverse_phase: bool,
    pub do_distortions: bool,
    pub do_relaxation: bool,
    pub noise_variance: f64,  // complex k-space noise variance (0 = off); Rician magnitude in image
    pub partial_fourier: f64, // fraction of PE lines acquired (1.0 = full); skips low-ky lines
    pub ghost_offset: f64,    // Nyquist ghost: kx offset (±) on odd/even PE lines (0 = off)
    pub eddy_strength: f64,   // linear (in-plane) eddy-current phase scale (0 = off)
    pub eddy_quad: f64,       // quadratic (x²,y²,z²) eddy-current phase scale (0 = off)
    pub eddy_tau: f64,        // eddy-current decay time (ms)
    pub n_spikes: usize,      // random k-space spikes per slice (0 = off) → herringbone
    pub spike_amplitude: f64, // spike magnitude as a fraction of the peak k-space sample
    pub zero_ringing: f64,    // Gibbs: % of k-space half-extent to zero at the edges (0 = off)
    pub n_coils: usize,       // receiver coils (1 = uniform single coil); ring-arranged sensitivities
    pub accel: usize,         // GRAPPA acceleration R (1 = fully sampled); undersamples PE lines
    pub acs_lines: usize,     // GRAPPA autocalibration lines (fully-sampled central PE band)
    pub seed: u64,            // mixed into every derived per-slice seed; 0 reproduces legacy output
}

/// Spatial sensitivity of `coil` (of `n_coils` arranged in a ring) at pixel `(x,y)`. Uniform for a
/// single coil; otherwise a Gaussian falloff from the coil position (distinct per coil — the
/// spatial diversity GRAPPA/SENSE need). Not normalized; RSS combine handles the overall scale.
fn coil_sensitivity(coil: usize, n_coils: usize, x: usize, y: usize, nx: usize, ny: usize) -> f64 {
    if n_coils <= 1 {
        return 1.0;
    }
    let (cx, cy) = (nx as f64 / 2.0, ny as f64 / 2.0);
    let r = 0.6 * nx.max(ny) as f64;
    let ang = TAU * coil as f64 / n_coils as f64;
    let (px, py) = (cx + r * ang.cos(), cy + r * ang.sin());
    let d2 = (x as f64 - px).powi(2) + (y as f64 - py).powi(2);
    let sigma = 0.9 * nx.max(ny) as f64;
    (-d2 / (2.0 * sigma * sigma)).exp() + 0.15
}

impl Default for Acquisition {
    fn default() -> Self {
        Acquisition {
            t_line: 1.0,
            t_echo: 90.0,
            t_inhom: 50.0,
            signal_scale: 100.0,
            reverse_phase: false,
            do_distortions: true,
            do_relaxation: true,
            noise_variance: 0.0,
            partial_fourier: 1.0,
            ghost_offset: 0.0,
            eddy_strength: 0.0,
            eddy_quad: 0.0,
            eddy_tau: 70.0,
            n_spikes: 0,
            spike_amplitude: 1.0,
            zero_ringing: 0.0,
            n_coils: 1,
            accel: 1,
            acs_lines: 24,
            seed: 0,
        }
    }
}

impl Acquisition {
    /// Defaults for a given acquired matrix. `Default::default()` is retained for callers that
    /// do not care about the matrix.
    pub fn default_for(_nx: usize, _ny: usize) -> Self {
        Acquisition::default()
    }
}

/// Per-PE-line readout times (ms): `t` from max echo, `tRf` from RF, `tRead` from the last
/// diffusion gradient (drives eddy-current decay).
fn line_times(epi: &SingleShotEpi) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (nx, ny) = (epi.kx_max, epi.ky_max);
    let xs = nx / 2;
    let (mut t, mut trf, mut tread) = (vec![0.0; ny], vec![0.0; ny], vec![0.0; ny]);
    for tick in 0..nx * ny {
        let (kx, ky) = epi.kspace_index(tick);
        if kx == xs {
            t[ky] = epi.time_from_max_echo(tick);
            trf[ky] = epi.time_from_rf(tick);
            tread[ky] = epi.time_from_last_diffusion_gradient(tick);
        }
    }
    (t, trf, tread)
}

/// Everything one slice's forward model needs. The simulation grid (`sim`) and the acquired
/// matrix (`acq_matrix`) are distinct: the object lives on the finer grid, and only the central
/// `acq_matrix` block of its k-space is evaluated (spec 3.1).
pub struct SliceInput<'a> {
    /// Per-compartment images on the SIM grid, layout `x + snx*y`.
    pub compartments: &'a [&'a [f32]],
    pub t2: &'a [f32],
    /// Off-resonance field (Hz) on the SIM grid.
    pub fmap: &'a [f32],
    /// Pre-readout object phase (radians) on the SIM grid. Ignored until Task 6.
    pub phase0: Option<&'a [f64]>,
    /// `[snx, sny]` — simulation grid, in-plane.
    pub sim: [usize; 2],
    /// `[nx, ny]` — acquired matrix.
    pub acq_matrix: [usize; 2],
    pub z: usize,
    pub nz: usize,
    /// Unit diffusion-gradient direction. Kept separate from `bval` so the phase model can form
    /// an effective q-vector without recovering it from a scaled product (spec 3.2).
    pub bvec: [f64; 3],
    pub bval: f64,
    pub slice_seed: u64,
}

/// Step edge at continuous position `edge` (sim-voxel units) with exact fractional occupancy in
/// the boundary voxel. This is the same partial-volume representation the path-length rasterizer
/// produces for real anatomy, so it is a production code path, not a test-only fixture.
pub fn step_hires(snx: usize, sny: usize, edge: f64) -> Vec<f32> {
    let mut v = vec![0.0f32; snx * sny];
    for x in 0..snx {
        let (lo, hi) = (x as f64, x as f64 + 1.0);
        let frac = if hi <= edge {
            0.0
        } else if lo >= edge {
            1.0
        } else {
            hi - edge
        };
        for y in 0..sny {
            v[x + snx * y] = frac as f32;
        }
    }
    v
}

/// Build the acquired k-space for one coil, complete with the ringing mask, spikes and thermal
/// noise, but before GRAPPA, reconstruction and coil combination. Shared by [`simulate_slice`] and
/// [`simulate_slice_kspace`] so tests can inspect the coefficients before reconstruction.
fn build_coil_kspace(inp: &SliceInput, acq: &Acquisition, coil: usize, ncoils: usize) -> Vec<C> {
    let [snx, sny] = inp.sim;
    let [nx, ny] = inp.acq_matrix;
    assert!(
        snx % nx == 0 && sny % ny == 0,
        "sim grid must be an integer multiple of the acquired matrix"
    );
    let (z, nz) = (inp.z, inp.nz);
    let (compartments, t2, fmap) = (inp.compartments, inp.t2, inp.fmap);
    let epi = SingleShotEpi {
        kx_max: nx,
        ky_max: ny,
        t_line: acq.t_line,
        t_echo: acq.t_echo,
        reverse_phase: acq.reverse_phase,
    };
    let (t_ms, trf_ms, tread_ms) = line_times(&epi);
    let gradient = [inp.bvec[0] * inp.bval, inp.bvec[1] * inp.bval, inp.bvec[2] * inp.bval];
    // eddy currents affect diffusion-weighted volumes only (b0 gradient ≈ 0)
    let do_eddy = acq.eddy_strength != 0.0 && inp.bval.abs() > 1e-9;
    // acquired-matrix centres (k-space indexing) and sim-grid centres (image indexing)
    let (xs, ys, zs) = (nx / 2, ny / 2, nz / 2);
    let (sxs, sys) = (snx / 2, sny / 2);
    let (ox, oy) = (snx / nx, sny / ny); // in-plane oversampling factors
    // Half-cell alignment. Sim cell `x` covers [x, x+1) in sim units, so its centre is at x+0.5;
    // acquired cell `X` covers o cells and is centred at o*X + o/2. Aligning sample index `x` with
    // `o*X` — as a bare index substitution does — therefore misregisters the object against the
    // reconstruction grid by (o-1)/2 sim cells, i.e. (o-1)/(2o) of an ACQUIRED voxel: 0.44 voxels
    // at o=8. Since this whole model turns on sub-voxel edge position, that shift is fatal and the
    // acquired k-space is measured from the acquired grid's centre instead. Exactly zero at o=1.
    let (xoff, yoff) = ((ox as f64 - 1.0) / 2.0, (oy as f64 - 1.0) / 2.0);
    let at = |x: usize, y: usize| x + snx * y; // SIM-grid image index
    let kat = |kx: usize, ky: usize| kx + nx * ky; // acquired k-space / acquired-image index

    let accel = acq.accel.max(1);
    let acs_half = (acq.acs_lines / 2) as i64;
    // acquired PE line? every R-th line (phase-aligned to k-space centre) plus the central ACS band.
    let acquired = |ky: usize| -> bool {
        accel <= 1 || (ky as i64 - ys as i64).abs() <= acs_half || ky % accel == ys % accel
    };
    // ---- forward: build k-space, factored as  Σ_x e^{..kx x}[ Σ_y mod(x,y) e^{..ky y} ] ----
    // mod(x,y) depends on the PE line ky (through φ and relaxation), so the y-sum is recomputed
    // per ky, but that keeps the whole build at O(N³).
    let mut kspace = vec![C::ZERO; nx * ny];
    let n_inv = 1.0 / (snx * sny) as f64;
    for kyi in 0..ny {
        // partial Fourier: don't acquire the "later" PE lines (low-ky, or high-ky when reversed);
        // those k-space rows stay zero. Mirrors itkKspaceImageFilter.cpp:322.
        if acq.partial_fourier < 1.0 {
            let skip = if acq.reverse_phase {
                kyi as f64 > (ny as f64 * acq.partial_fourier).ceil()
            } else {
                (kyi as f64) < (ny as f64 * (1.0 - acq.partial_fourier)).floor()
                    && (kyi > 0 || ny % 2 == 1)
            };
            if skip {
                continue;
            }
        }
        if !acquired(kyi) {
            continue; // undersampled PE line — GRAPPA fills it from the ACS-calibrated kernel
        }
        let t = t_ms[kyi] / 1000.0; // seconds
        let trf = trf_ms[kyi];
        // Divide by the SIM extent: the loop still runs over the ny ACQUIRED lines, but each sits
        // at absolute sim index sys - ys + kyi, whose normalized frequency is (kyi - ys)/sny.
        let ky_norm = (kyi as f64 - ys as f64) / sny as f64;
        // Nyquist (N/2) ghost: alternating readout-line kx offset (gradient-delay mismatch).
        let ghost_shift = if kyi % 2 == 1 { -acq.ghost_offset } else { acq.ghost_offset };
        // eddy-current decay for this PE line: exp(-tRead/τ)·t (itkKspaceImageFilter.cpp:354)
        let eddy_decay = if do_eddy { (-tread_ms[kyi] / acq.eddy_tau).exp() * t } else { 0.0 };

        // modulated image for this PE line
        let mut modimg = vec![C::ZERO; snx * sny];
        for y in 0..sny {
            for x in 0..snx {
                let mut f_real = 0.0f64;
                for (c, comp) in compartments.iter().enumerate() {
                    let mut v = comp[at(x, y)] as f64;
                    if acq.do_relaxation {
                        v *= (-(trf as f64) / t2[c] as f64 - t.abs() * 1000.0 / acq.t_inhom).exp();
                    }
                    f_real += v;
                }
                // sensitivity is evaluated on the sim grid; the profile is scale-invariant, so
                // the same physical field is sampled at any oversampling factor
                f_real *= acq.signal_scale * coil_sensitivity(coil, ncoils, x, y, snx, sny);
                let mut phi = if acq.do_distortions { fmap[at(x, y)] as f64 * t } else { 0.0 };
                if do_eddy {
                    // gradient-dependent field growing through the readout: linear (g·pos) plus a
                    // quadratic (g·pos²) term — the polynomial forms eddy/TORTOISE fit.
                    // centre on the sim grid, then express in ACQUIRED voxel units so that
                    // eddy_strength/eddy_quad keep their meaning independent of oversampling
                    let (xc, yc, zc) = (
                        (x as f64 - sxs as f64 - xoff) / ox as f64,
                        (y as f64 - sys as f64 - yoff) / oy as f64,
                        z as f64 - zs as f64,
                    );
                    let lin = gradient[0] * xc + gradient[1] * yc + gradient[2] * zc;
                    let quad =
                        gradient[0] * xc * xc + gradient[1] * yc * yc + gradient[2] * zc * zc;
                    phi += (acq.eddy_strength * lin + acq.eddy_quad * quad) * eddy_decay;
                }
                modimg[at(x, y)] = C::cis(TAU * phi).scale(f_real);
            }
        }
        // inner y-sum over the SIM grid → g(x), then an x-DFT evaluated only at acquired kx
        let mut g = vec![C::ZERO; snx];
        for x in 0..snx {
            let mut acc = C::ZERO;
            for y in 0..sny {
                let ph = C::cis(TAU * ky_norm * (y as f64 - sys as f64 - yoff));
                acc = acc.add(modimg[at(x, y)].mul(ph));
            }
            g[x] = acc;
        }
        for kxi in 0..nx {
            let kx_norm = (kxi as f64 - xs as f64 + ghost_shift) / snx as f64;
            let mut acc = C::ZERO;
            for x in 0..snx {
                acc = acc.add(g[x].mul(C::cis(TAU * kx_norm * (x as f64 - sxs as f64 - xoff))));
            }
            kspace[kat(kxi, kyi)] = acc.scale(n_inv);
        }
    }

    // Gibbs ringing: zero the outer high-frequency k-space lines (truncation → ringing at edges).
    // Mirrors itkKspaceImageFilter.cpp:334 (an alternative to acquiring a smaller matrix).
    if acq.zero_ringing > 0.0 {
        let rx = ((nx as f64 / 2.0) * acq.zero_ringing / 100.0).ceil() as usize;
        let ry = ((ny as f64 / 2.0) * acq.zero_ringing / 100.0).ceil() as usize;
        for ky in 0..ny {
            for kx in 0..nx {
                if kx < rx || ky < ry || kx + rx >= nx || ky + ry >= ny {
                    kspace[kat(kx, ky)] = C::ZERO;
                }
            }
        }
    }

    // Spikes: overwrite random k-space points with a fraction of the peak sample → herringbone
    // ripple in the image (itkKspaceImageFilter.cpp:502).
    if acq.n_spikes > 0 {
        let (mut peak, mut peak_mag) = (C::ZERO, 0.0);
        for k in &kspace {
            let m = k.abs();
            if m > peak_mag {
                peak_mag = m;
                peak = *k;
            }
        }
        let spike = peak.scale(acq.spike_amplitude);
        let mut rng = Rng(inp.slice_seed ^ 0xA5A5_1234_5678_9ABC);
        for _ in 0..acq.n_spikes {
            let kx = (rng.next_u64() as usize) % nx;
            let ky = (rng.next_u64() as usize) % ny;
            kspace[kat(kx, ky)] = spike;
        }
    }

    // complex k-space noise → Rician magnitude in image; image-space SD ≈ sqrt(noise_variance).
    if acq.noise_variance > 0.0 {
        let mut rng = Rng((inp.slice_seed ^ (coil as u64).wrapping_mul(0x9E37_79B9)) | 1);
        let sigma = (acq.noise_variance / (nx * ny) as f64).sqrt();
        for k in kspace.iter_mut() {
            k.re += rng.gauss() * sigma;
            k.im += rng.gauss() * sigma;
        }
    }

    kspace
}

/// The acquired k-space of the first coil, before reconstruction. Layout `kx + nx*ky`.
/// Used by the convergence tests (spec 4.1.6); not part of the simulation path.
pub fn simulate_slice_kspace(inp: &SliceInput, acq: &Acquisition) -> Vec<(f64, f64)> {
    build_coil_kspace(inp, acq, 0, acq.n_coils.max(1))
        .into_iter()
        .map(|c| (c.re, c.im))
        .collect()
}

/// Simulate one slice: compartment images on the SIM grid (each `snx*sny`, layout `x + snx*y`) →
/// complex image on the ACQUIRED matrix (`nx*ny`). `t2` is the per-compartment T2 (ms); `fmap` is
/// the off-resonance field (Hz), same sim-grid layout. Only the central `nx*ny` block of the sim
/// grid's k-space is evaluated, so truncation to the nominal band happens during the forward
/// transform rather than by discarding a computed k-space (spec 3.1).
pub fn simulate_slice(inp: &SliceInput, acq: &Acquisition) -> Vec<(f32, f32)> {
    let [nx, ny] = inp.acq_matrix;
    let (xs, ys) = (nx / 2, ny / 2);
    let kat = |kx: usize, ky: usize| kx + nx * ky; // acquired k-space / acquired-image index

    // Build each coil's k-space (undersampled for GRAPPA when accel>1), reconstruct, then combine.
    let ncoils = acq.n_coils.max(1);
    let accel = acq.accel.max(1);
    let mut coil_kspace: Vec<Vec<C>> = Vec::with_capacity(ncoils);
    for coil in 0..ncoils {
        coil_kspace.push(build_coil_kspace(inp, acq, coil, ncoils));
    }

    // GRAPPA: fill the un-acquired PE lines with a kernel calibrated on the ACS band across coils.
    if accel > 1 {
        grappa_reconstruct(&mut coil_kspace, nx, ny, ys, accel, acq.acs_lines);
    }

    // inverse each coil, then phase-preserving Roemer combine with the known sensitivities:
    //   combined = Σ_c image_c · sens_c / Σ_c sens_c²  (real sensitivities here)
    let mut wsum = vec![C::ZERO; nx * ny];
    let mut ssum = vec![0.0f64; nx * ny];
    for (coil, ks) in coil_kspace.iter().enumerate() {
        let img = inverse_2d(ks, nx, ny, xs, ys);
        for y in 0..ny {
            for x in 0..nx {
                let s = coil_sensitivity(coil, ncoils, x, y, nx, ny);
                let i = kat(x, y);
                wsum[i] = wsum[i].add(img[i].scale(s));
                ssum[i] += s * s;
            }
        }
    }
    (0..nx * ny)
        .map(|i| {
            let s = ssum[i].max(1e-12);
            ((wsum[i].re / s) as f32, (wsum[i].im / s) as f32)
        })
        .collect()
}

/// Inverse 2D DFT (separable direct sums): complex k-space → complex image, centred convention.
fn inverse_2d(kspace: &[C], nx: usize, ny: usize, xs: usize, ys: usize) -> Vec<C> {
    let at = |x: usize, y: usize| x + nx * y;
    let mut h = vec![C::ZERO; nx * ny];
    for kxi in 0..nx {
        for y in 0..ny {
            let mut acc = C::ZERO;
            for kyi in 0..ny {
                let ky_norm = (kyi as f64 - ys as f64) / ny as f64;
                acc = acc.add(kspace[at(kxi, kyi)].mul(C::cis(-TAU * ky_norm * (y as f64 - ys as f64))));
            }
            h[at(kxi, y)] = acc;
        }
    }
    let mut out = vec![C::ZERO; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            let mut acc = C::ZERO;
            for kxi in 0..nx {
                let kx_norm = (kxi as f64 - xs as f64) / nx as f64;
                acc = acc.add(h[at(kxi, y)].mul(C::cis(-TAU * kx_norm * (x as f64 - xs as f64))));
            }
            out[at(x, y)] = acc;
        }
    }
    out
}

/// Solve the Hermitian normal-equations system `H x = b` (complex, N×N) by Gaussian elimination
/// with partial pivoting. `H = AᴴA` is positive-definite after Tikhonov regularization.
fn solve_hermitian(mut h: Vec<Vec<C>>, mut b: Vec<C>) -> Vec<C> {
    let n = b.len();
    let recip = |d: C| {
        let m2 = d.re * d.re + d.im * d.im;
        if m2 < 1e-18 { C::ZERO } else { C { re: d.re / m2, im: -d.im / m2 } }
    };
    for col in 0..n {
        let (mut piv, mut best) = (col, h[col][col].abs());
        for r in (col + 1)..n {
            let m = h[r][col].abs();
            if m > best { best = m; piv = r; }
        }
        h.swap(col, piv);
        b.swap(col, piv);
        let dinv = recip(h[col][col]);
        for r in (col + 1)..n {
            let f = h[r][col].mul(dinv);
            for c in col..n {
                let s = f.mul(h[col][c]);
                h[r][c] = h[r][c].add(s.scale(-1.0));
            }
            let sb = f.mul(b[col]);
            b[r] = b[r].add(sb.scale(-1.0));
        }
    }
    let mut x = vec![C::ZERO; n];
    for col in (0..n).rev() {
        let mut s = b[col];
        for c in (col + 1)..n {
            s = s.add(h[col][c].mul(x[c]).scale(-1.0));
        }
        x[col] = s.mul(recip(h[col][col]));
    }
    x
}

/// GRAPPA: fill un-acquired PE lines in-place. Calibrates a kernel (2 PE source lines spaced `R`,
/// × 3 readout points, × all coils) on the fully-sampled central ACS band, then applies it to
/// synthesize each missing line per coil. The g-factor noise amplification emerges automatically
/// from reconstructing undersampled noisy multi-coil data.
fn grappa_reconstruct(coil: &mut [Vec<C>], nx: usize, ny: usize, ys: usize, accel: usize, acs_lines: usize) {
    let nc = coil.len();
    if nc == 0 || accel <= 1 {
        return;
    }
    let at = |x: usize, y: usize| x + nx * y;
    let dkx: [i64; 3] = [-1, 0, 1];
    let nfeat = 2 * dkx.len() * nc;
    let sidx = |line: usize, ki: usize, ci: usize| (line * dkx.len() + ki) * nc + ci;
    let acs_half = (acs_lines / 2) as i64;
    let (acs_lo, acs_hi) = ((ys as i64 - acs_half).max(0) as usize, ((ys as i64 + acs_half) as usize).min(ny - 1));

    let feature = |coil: &[Vec<C>], s0: usize, s1: usize, kx: usize| -> Vec<C> {
        let mut f = vec![C::ZERO; nfeat];
        for (li, &sl) in [s0, s1].iter().enumerate() {
            for (ki, &dk) in dkx.iter().enumerate() {
                let kxs = (kx as i64 + dk).rem_euclid(nx as i64) as usize;
                for ci in 0..nc {
                    f[sidx(li, ki, ci)] = coil[ci][at(kxs, sl)];
                }
            }
        }
        f
    };

    for d in 1..accel {
        // calibrate: accumulate AᴴA (shared) and Aᴴb (per output coil) over ACS positions
        let mut aha = vec![vec![C::ZERO; nfeat]; nfeat];
        let mut ahb = vec![vec![C::ZERO; nfeat]; nc];
        for b in acs_lo..acs_hi {
            let (s0, s1, tgt_ky) = (b, b + accel, b + d);
            if s1 > acs_hi || tgt_ky > acs_hi {
                continue;
            }
            for kx in 0..nx {
                let f = feature(coil, s0, s1, kx);
                for i in 0..nfeat {
                    let fic = C { re: f[i].re, im: -f[i].im }; // conj
                    for j in 0..nfeat {
                        aha[i][j] = aha[i][j].add(fic.mul(f[j]));
                    }
                    for co in 0..nc {
                        ahb[co][i] = ahb[co][i].add(fic.mul(coil[co][at(kx, tgt_ky)]));
                    }
                }
            }
        }
        for (i, row) in aha.iter_mut().enumerate() {
            row[i] = row[i].add(C { re: 1e-4, im: 0.0 }); // Tikhonov
        }
        let weights: Vec<Vec<C>> = (0..nc).map(|co| solve_hermitian(aha.clone(), ahb[co].clone())).collect();

        // synthesize every missing line at this offset (outside the ACS)
        for ky in 0..ny {
            if (ky as i64 - ys as i64).rem_euclid(accel as i64) != d as i64 {
                continue;
            }
            if (ky as i64 - ys as i64).abs() <= acs_half || ky < d || ky + accel - d >= ny {
                continue;
            }
            let (s0, s1) = (ky - d, ky - d + accel);
            for kx in 0..nx {
                let f = feature(coil, s0, s1, kx);
                for co in 0..nc {
                    let mut v = C::ZERO;
                    for i in 0..nfeat {
                        v = v.add(weights[co][i].mul(f[i]));
                    }
                    coil[co][at(kx, ky)] = v;
                }
            }
        }
    }
}

/// Run the k-space acquisition over a whole 4D clean-signal volume: every (volume, slice) through
/// [`simulate_slice`]. `clean` is the clean signal in `(x+nx*(y+ny*z))*ngrad + g` layout; `fmap`
/// is the off-resonance field (Hz) on the same grid. v1 treats the mixed signal as one compartment
/// with an effective T2 (`t2_eff`, ms) — per-tissue T2 for realistic b0 contrast is a refinement.
/// Returns `(magnitude, phase)` 4D arrays (phase in radians, atan2), same layout — the complex pair
/// needed for BIDS `part-mag`/`part-phase` and complex denoisers. Parallel over volumes with `par`.
pub fn simulate_acquisition(
    dims: [usize; 3],
    ngrad: usize,
    images: &[Vec<f32>],
    t2: &[f32],
    fmap: &[f32],
    acq: &Acquisition,
    gradients: &[[f64; 3]],
) -> (Vec<f32>, Vec<f32>) {
    let [nx, ny, nz] = dims;
    let nvox = nx * ny * nz;
    let ncomp = images.len();
    // returns (magnitude, phase) for one volume
    let per_vol = |g: usize| -> (Vec<f32>, Vec<f32>) {
        let (mut mag, mut phase) = (vec![0.0f32; nvox], vec![0.0f32; nvox]);
        let mut cslices = vec![vec![0.0f32; nx * ny]; ncomp];
        let mut fslice = vec![0.0f32; nx * ny];
        // split the scaled gradient into a unit direction and a magnitude; the forward model
        // recombines them, so this preserves the previous behaviour exactly
        let gr = gradients[g];
        let bval = (gr[0] * gr[0] + gr[1] * gr[1] + gr[2] * gr[2]).sqrt();
        let bvec =
            if bval > 1e-12 { [gr[0] / bval, gr[1] / bval, gr[2] / bval] } else { [0.0; 3] };
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let vox = x + nx * (y + ny * z);
                    for (c, img) in images.iter().enumerate() {
                        cslices[c][x + nx * y] = img[vox * ngrad + g];
                    }
                    fslice[x + nx * y] = fmap[vox];
                }
            }
            let refs: Vec<&[f32]> = cslices.iter().map(|v| v.as_slice()).collect();
            let seed = (g as u64).wrapping_mul(0x100_0001).wrapping_add(z as u64).wrapping_mul(0x9E37)
                .wrapping_add(acq.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let inp = SliceInput {
                compartments: &refs,
                t2,
                fmap: &fslice,
                phase0: None,
                // v1 acquisition path: the object is already on the acquired matrix (o = 1).
                sim: [nx, ny],
                acq_matrix: [nx, ny],
                z,
                nz,
                bvec,
                bval,
                slice_seed: seed,
            };
            let out = simulate_slice(&inp, acq);
            for y in 0..ny {
                for x in 0..nx {
                    let (re, im) = out[x + nx * y];
                    let vox = x + nx * (y + ny * z);
                    mag[vox] = (re * re + im * im).sqrt();
                    phase[vox] = im.atan2(re);
                }
            }
        }
        (mag, phase)
    };

    #[cfg(feature = "par")]
    let vols: Vec<(Vec<f32>, Vec<f32>)> = {
        use rayon::prelude::*;
        (0..ngrad).into_par_iter().map(per_vol).collect()
    };
    #[cfg(not(feature = "par"))]
    let vols: Vec<(Vec<f32>, Vec<f32>)> = (0..ngrad).map(per_vol).collect();

    let (mut magd, mut phased) = (vec![0.0f32; nvox * ngrad], vec![0.0f32; nvox * ngrad]);
    for (g, (m, p)) in vols.iter().enumerate() {
        for vox in 0..nvox {
            magd[vox * ngrad + g] = m[vox];
            phased[vox * ngrad + g] = p[vox];
        }
    }
    (magd, phased)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mag(v: &[(f32, f32)]) -> Vec<f32> {
        v.iter().map(|&(r, i)| (r * r + i * i).sqrt()).collect()
    }

    /// The pre-existing single-compartment call shape, on a `SliceInput` whose simulation grid
    /// equals the acquired matrix (`o = 1`, `phase0: None`) — i.e. the old behaviour exactly.
    #[allow(clippy::too_many_arguments)]
    fn slice1(
        img: &[f32],
        t2: &[f32],
        fmap: &[f32],
        nx: usize,
        ny: usize,
        z: usize,
        nz: usize,
        acq: &Acquisition,
        gradient: [f64; 3],
        slice_seed: u64,
    ) -> Vec<(f32, f32)> {
        let bval = (gradient[0].powi(2) + gradient[1].powi(2) + gradient[2].powi(2)).sqrt();
        let bvec = if bval > 1e-12 {
            [gradient[0] / bval, gradient[1] / bval, gradient[2] / bval]
        } else {
            [0.0; 3]
        };
        let comps: [&[f32]; 1] = [img];
        simulate_slice(
            &SliceInput {
                compartments: &comps,
                t2,
                fmap,
                phase0: None,
                sim: [nx, ny],
                acq_matrix: [nx, ny],
                z,
                nz,
                bvec,
                bval,
                slice_seed,
            },
            acq,
        )
    }

    fn phantom(nx: usize, ny: usize) -> Vec<f32> {
        // a bright square in the middle
        let mut v = vec![0.0f32; nx * ny];
        for y in ny / 3..2 * ny / 3 {
            for x in nx / 3..2 * nx / 3 {
                v[x + nx * y] = 1.0;
            }
        }
        v
    }

    #[test]
    fn roundtrip_recovers_image_when_no_distortion() {
        let (nx, ny) = (16, 16);
        let img = phantom(nx, ny);
        let fmap = vec![0.0f32; nx * ny];
        let acq = Acquisition {
            signal_scale: 1.0,
            do_distortions: false,
            do_relaxation: false,
            ..Default::default()
        };
        let out = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &acq, [0.0, 0.0, 0.0], 0));
        let err: f32 = img.iter().zip(&out).map(|(a, b)| (a - b).abs()).sum::<f32>() / (nx * ny) as f32;
        assert!(err < 1e-4, "roundtrip mean abs err {err}");
    }

    #[test]
    fn uniform_fieldmap_shifts_along_phase_encode() {
        // A uniform off-resonance shifts the whole image along PE (y) by ~fmap·N_pe·t_line pixels.
        let (nx, ny) = (24, 24);
        let img = phantom(nx, ny);
        let f0 = 60.0f32; // Hz off-resonance
        let fmap = vec![f0; nx * ny];
        // predicted PE shift ≈ f0 · t_line[s] · N_pe = 60 · 0.003 · 24 ≈ 4.3 px
        let acq = Acquisition {
            t_line: 3.0, // ms
            signal_scale: 1.0,
            do_distortions: true,
            do_relaxation: false,
            reverse_phase: false,
            ..Default::default()
        };
        let out = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &acq, [0.0, 0.0, 0.0], 0));
        // centroid of the bright region should move in y vs the undistorted image
        let cy = |v: &[f32]| {
            let (mut sw, mut sy) = (0.0f64, 0.0f64);
            for y in 0..ny {
                for x in 0..nx {
                    let w = v[x + nx * y] as f64;
                    sw += w;
                    sy += w * y as f64;
                }
            }
            sy / sw.max(1e-9)
        };
        let acq0 = Acquisition { do_distortions: false, ..acq.clone() };
        let base = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &acq0, [0.0, 0.0, 0.0], 0));
        let shift = cy(&out) - cy(&base);
        assert!(shift.abs() > 1.5, "expected a clear PE shift (~4px), got {shift}");
        // reverse phase-encode flips the distortion direction
        let acq_rev = Acquisition { reverse_phase: true, ..acq.clone() };
        let rev = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &acq_rev, [0.0, 0.0, 0.0], 0));
        let shift_rev = cy(&rev) - cy(&base);
        assert!(shift * shift_rev < 0.0, "AP/PA should distort oppositely: {shift} vs {shift_rev}");
    }

    #[test]
    fn partial_fourier_changes_the_image_but_preserves_brain() {
        let (nx, ny) = (24, 24);
        let img = phantom(nx, ny);
        let fmap = vec![0.0f32; nx * ny];
        let full = Acquisition { signal_scale: 1.0, do_distortions: false, do_relaxation: false, ..Default::default() };
        let pf = Acquisition { partial_fourier: 0.6, ..full.clone() };
        let a = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &full, [0.0, 0.0, 0.0], 0));
        let b = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &pf, [0.0, 0.0, 0.0], 0));
        let diff: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f32>() / (nx * ny) as f32;
        assert!(diff > 1e-3, "partial Fourier should alter the image, diff {diff}");
        // most of the signal energy is still there (PF keeps the central k-space)
        let (ea, eb): (f32, f32) = (a.iter().sum(), b.iter().sum());
        assert!((eb / ea - 1.0).abs() < 0.5, "PF shouldn't destroy the brain: {ea} vs {eb}");
    }

    #[test]
    fn nyquist_ghost_leaks_signal_into_background() {
        // off-centre source so its N/2 ghost lands in otherwise-empty background
        let (nx, ny) = (24, 24);
        let mut img = vec![0.0f32; nx * ny];
        for y in 2..6 {
            for x in 10..14 {
                img[x + nx * y] = 1.0;
            }
        }
        let fmap = vec![0.0f32; nx * ny];
        let base = Acquisition { signal_scale: 1.0, do_distortions: false, do_relaxation: false, ..Default::default() };
        let ghost = Acquisition { ghost_offset: 0.5, ..base.clone() };
        let a = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &base, [0.0, 0.0, 0.0], 0));
        let b = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &ghost, [0.0, 0.0, 0.0], 0));
        // background region ~half-FOV away in the PE(y) direction from the source
        let bg = |v: &[f32]| {
            let mut s = 0.0f32;
            for y in 14..18 {
                for x in 10..14 {
                    s += v[x + nx * y];
                }
            }
            s
        };
        assert!(bg(&b) > bg(&a) + 0.1, "ghost should add background signal: {} vs {}", bg(&a), bg(&b));
    }

    #[test]
    fn eddy_currents_affect_dwi_but_not_b0() {
        let (nx, ny) = (24, 24);
        let img = phantom(nx, ny);
        let fmap = vec![0.0f32; nx * ny];
        let base = Acquisition { signal_scale: 1.0, do_distortions: false, do_relaxation: false, ..Default::default() };
        let eddy = Acquisition { eddy_strength: 5.0, ..base.clone() };
        // DWI volume (gradient along x): eddy shears the image
        let grad = [1.0, 0.0, 0.0];
        let a = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &base, grad, 0));
        let b = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &eddy, grad, 0));
        let diff: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f32>() / (nx * ny) as f32;
        assert!(diff > 1e-3, "eddy should shear the DWI, diff {diff}");
        // b0 (zero gradient): eddy must have no effect
        let z0 = [0.0, 0.0, 0.0];
        let a0 = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &base, z0, 0));
        let b0 = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &eddy, z0, 0));
        let diff0: f32 = a0.iter().zip(&b0).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff0 < 1e-6, "eddy must not touch b0, diff {diff0}");
    }

    #[test]
    fn spikes_ripple_into_the_background() {
        let (nx, ny) = (24, 24);
        let img = phantom(nx, ny);
        let fmap = vec![0.0f32; nx * ny];
        let base = Acquisition { signal_scale: 1.0, do_distortions: false, do_relaxation: false, ..Default::default() };
        let spiky = Acquisition { n_spikes: 3, spike_amplitude: 1.0, ..base.clone() };
        let a = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &base, [0.0, 0.0, 0.0], 0));
        let b = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &spiky, [0.0, 0.0, 0.0], 0));
        let corner = |v: &[f32]| {
            let mut s = 0.0f32;
            for y in 0..3 {
                for x in 0..3 {
                    s += v[x + nx * y];
                }
            }
            s
        };
        assert!(corner(&b) > corner(&a) + 0.1, "spikes should ripple into background: {} vs {}", corner(&a), corner(&b));
    }

    #[test]
    fn gibbs_ringing_changes_the_image() {
        let (nx, ny) = (24, 24);
        let img = phantom(nx, ny);
        let fmap = vec![0.0f32; nx * ny];
        let base = Acquisition { signal_scale: 1.0, do_distortions: false, do_relaxation: false, ..Default::default() };
        let ring = Acquisition { zero_ringing: 25.0, ..base.clone() };
        let a = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &base, [0.0, 0.0, 0.0], 0));
        let b = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &ring, [0.0, 0.0, 0.0], 0));
        let diff: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f32>() / (nx * ny) as f32;
        assert!(diff > 1e-3, "Gibbs zeroing should change the image, diff {diff}");
    }

    #[test]
    fn multicoil_rss_recovers_the_brain() {
        let (nx, ny) = (24, 24);
        let img = phantom(nx, ny);
        let fmap = vec![0.0f32; nx * ny];
        let acq = Acquisition { n_coils: 4, signal_scale: 1.0, do_distortions: false, do_relaxation: false, ..Default::default() };
        let out = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &acq, [0.0, 0.0, 0.0], 0));
        let center = out[12 + nx * 12];
        let corner = out[1 + nx * 1];
        assert!(center > 0.1 && center > corner, "RSS should recover the brain: center {center} corner {corner}");
    }

    #[test]
    fn grappa_unfolds_undersampled_multicoil_data() {
        let (nx, ny) = (32, 32);
        let img = phantom(nx, ny);
        let fmap = vec![0.0f32; nx * ny];
        let full = Acquisition { n_coils: 8, signal_scale: 1.0, do_distortions: false, do_relaxation: false, ..Default::default() };
        let accel = Acquisition { accel: 2, acs_lines: 16, ..full.clone() };
        let a = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &full, [0.0, 0.0, 0.0], 0));
        let g = mag(&slice1(&img, &[100.0], &fmap, nx, ny, 0, 1, &accel, [0.0, 0.0, 0.0], 0));
        // noise-free: GRAPPA should recover the fully-sampled image closely (aliasing unfolded)
        let num: f32 = a.iter().zip(&g).map(|(x, y)| (x - y).abs()).sum();
        let den: f32 = a.iter().sum::<f32>().max(1e-6);
        let rel = num / den;
        assert!(rel < 0.15, "GRAPPA should reconstruct the image, rel err {rel}");
    }
    use crate::analytic::truncated_step_profile;

    /// Acquisition with every artifact off: pure finite-Fourier acquisition.
    fn clean(nx: usize, ny: usize) -> Acquisition {
        Acquisition {
            signal_scale: 1.0,
            do_distortions: false,
            do_relaxation: false,
            noise_variance: 0.0,
            partial_fourier: 1.0,
            ghost_offset: 0.0,
            eddy_strength: 0.0,
            n_coils: 1,
            accel: 1,
            ..Acquisition::default_for(nx, ny)
        }
    }

    #[test]
    fn crop_reproduces_the_analytic_profile_across_subvoxel_offsets() {
        let (nx, ny, o) = (32usize, 32usize, 8usize);
        let (snx, sny) = (nx * o, ny * o);
        let fmap = vec![0.0f32; snx * sny];
        let acq = clean(nx, ny);

        for i in 0..16 {
            let delta = i as f64 / 16.0;
            // Edge at (nx/2 + delta) acquired voxels, expressed in sim-voxel units.
            let edge = (nx as f64 / 2.0 + delta) * o as f64;
            let img = step_hires(snx, sny, edge);
            let comps: [&[f32]; 1] = [&img];
            let inp = SliceInput {
                compartments: &comps,
                t2: &[100.0],
                fmap: &fmap,
                phase0: None,
                sim: [snx, sny],
                acq_matrix: [nx, ny],
                z: 0,
                nz: 1,
                bvec: [0.0, 0.0, 0.0],
                bval: 0.0,
                slice_seed: 0,
            };
            let out = simulate_slice(&inp, &acq);

            let expect = truncated_step_profile(nx, (nx as f64 / 2.0 + delta) / nx as f64);
            let row = ny / 2;
            let worst = (0..nx)
                .map(|x| (out[x + nx * row].0 as f64 - expect[x]).abs())
                .fold(0.0f64, f64::max);
            assert!(worst < 5e-3, "offset {delta}: max profile deviation {worst:.4e}");
        }
    }

    #[test]
    fn ringing_is_intrinsic_without_any_ringing_parameter() {
        // A sub-voxel-positioned edge must now ring: the old exact-DFT round-trip is gone.
        let (nx, ny, o) = (32usize, 32usize, 8usize);
        let (snx, sny) = (nx * o, ny * o);
        let fmap = vec![0.0f32; snx * sny];
        let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
        let comps: [&[f32]; 1] = [&img];
        let inp = SliceInput {
            compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: None,
            sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1,
            bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 0,
        };
        let out = simulate_slice(&inp, &clean(nx, ny));
        let row = ny / 2;
        let peak = (nx / 2 + 1..nx).map(|x| out[x + nx * row].0 as f64).fold(f64::MIN, f64::max);
        assert!(peak > 1.05, "expected intrinsic overshoot, got peak {peak:.4}");
    }

    #[test]
    fn acquired_band_converges_with_simulation_resolution() {
        // Spec 4.1.6: adequacy is convergence of the ACQUIRED coefficients, not smoothness of the
        // object. A sharp edge is deliberately not band-limited; that is not a defect.
        let (nx, ny) = (16usize, 16usize);
        let acq = clean(nx, ny);
        let k_at = |o: usize| -> Vec<(f64, f64)> {
            let (snx, sny) = (nx * o, ny * o);
            let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.37) * o as f64);
            let fmap = vec![0.0f32; snx * sny];
            let comps: [&[f32]; 1] = [&img];
            simulate_slice_kspace(
                &SliceInput {
                    compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: None,
                    sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1,
                    bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 0,
                },
                &acq,
            )
        };
        let rel = |a: &[(f64, f64)], b: &[(f64, f64)]| {
            let num: f64 =
                a.iter().zip(b).map(|(p, q)| (p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sum();
            let den: f64 = b.iter().map(|q| q.0 * q.0 + q.1 * q.1).sum::<f64>().max(1e-30);
            (num / den).sqrt()
        };
        let (k2, k4, k8) = (k_at(2), k_at(4), k_at(8));
        let (e2, e4) = (rel(&k2, &k4), rel(&k4, &k8));
        assert!(e4 < e2, "error must shrink with o: e(2->4)={e2:.4e}, e(4->8)={e4:.4e}");
        // Report o_min for the production default (spec 4.1.10): the smallest o under tolerance.
        let eps = 1e-3;
        let o_min = if e2 < eps { 2 } else if e4 < eps { 4 } else { 8 };
        println!("o_min at eps={eps}: {o_min}  (e2={e2:.3e}, e4={e4:.3e})");
        assert!(e4 < 1e-2, "o=4 should be within 1% of o=8: {e4:.4e}");
    }
}
