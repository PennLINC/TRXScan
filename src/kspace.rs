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
//! The transform is exact — the literal O(N³) sum, not an approximation of it — so it stays
//! directly comparable against Fiberfox. It is organised for speed without changing a term: the
//! per-line factors are split into static (hoisted), affine-in-ky (advanced by per-voxel rotors),
//! per-line scalars (relaxation) and per-axis separable (the eddy polynomial), leaving an inner
//! loop of complex multiply-adds; the x-DFT of each line and the 2-D reconstruction are FFTs
//! (`rustfft`, behind the `kspace` feature) or twiddle-table sums (default, std-only) — same
//! numbers either way (`restructured_forward_matches_the_literal_sum`,
//! `fft_inverse_matches_the_direct_dft`). Under `kspace` the y-sum with the fieldmap is also
//! O(N log N): read as geometry it is a source warp `y − sny·τ·fmap`, evaluated by a type-1
//! NUFFT (`nufft.rs`, ~1e-13); only the legacy gradient-model eddy polynomial, whose time profile
//! is not affine in ky, keeps the O(N³) rotor path.

use crate::phase::{PhaseModel, ShotPhase};
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
    #[cfg_attr(feature = "kspace", allow(dead_code))]
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
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    /// standard normal
    pub fn gauss(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

/// How partial Fourier drops phase-encode lines.
///
/// These differ by more than an off-by-one, and the difference matters for anything evaluating
/// PF-aware unringing (RPG and friends), so it is an explicit choice rather than a hidden quirk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PartialFourierMode {
    /// What a scanner does, and the default: the train **starts late** and skips the first
    /// `ny - round(ny*pf)` lines it would have acquired, so the k-space centre is reached
    /// `ny*(1-pf)` line times sooner (the reason partial Fourier exists: a shorter minimum TE)
    /// and the readout is that much shorter. Exactly `round(ny*pf)` consecutive lines are kept:
    /// the centre plus one full side, the side acquired *after* the centre. `t_echo` is still
    /// the caller's number; lower it to bank the TE gain. Time since the readout start (the
    /// eddy-current decay clock) counts from the first acquired line.
    #[default]
    Scanner,
    /// Legacy TRXScan rule: the same `round(ny*pf)` consecutive lines, but dropped from the END
    /// of the train, so timing is unchanged (the centre is reached as late as at full Fourier)
    /// and the kept side is the one acquired *before* the centre. Kept for reproducing older
    /// runs; not what a scanner does.
    Contiguous,
    /// Fiberfox's rule, ported as-is. On an even matrix it additionally **preserves line zero**,
    /// because that line's Nyquist conjugate is absent. The consequence is that nominal "6/8" is
    /// not 75%: at `ny = 32` it keeps 25 of 32 lines, **78.1%**, and leaves one isolated line
    /// disconnected from the acquired block. Faithful to the reference implementation, kept for
    /// validation against it.
    FiberfoxCompatible,
}

/// How many lines at the start of the EPI train a [`PartialFourierMode::Scanner`] acquisition
/// skips (0 for the other modes and at full Fourier).
pub fn pf_skipped_lines(ny: usize, acq: &Acquisition) -> usize {
    if acq.partial_fourier < 1.0 && acq.pf_mode == PartialFourierMode::Scanner {
        ny - ((ny as f64 * acq.partial_fourier).round() as usize).min(ny)
    } else {
        0
    }
}

/// Scanner-side reconstruction apodization. Distinct transfer functions with distinct PSFs, so
/// they are named rather than hidden behind one ambiguous scalar.
///
/// `None` is the default and is what the algorithm-validation fixture uses: Kellner/`mrdegibbs`
/// assume an unapodized rectangular window. Note this is the default *window*; TRXScan's shipping
/// acquisition default remains 6/8 partial Fourier, and the full-Fourier benchmark fixture sets
/// `partial_fourier = 1.0` explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KspaceWindow {
    None,
    Tukey { alpha: f64 },
    Hann,
    Fermi { radius: f64, width: f64 },
}

impl KspaceWindow {
    /// Window value at normalized frequency `(kx, ky)`, each in `[-0.5, 0.5]`.
    ///
    /// **These are RADIAL windows**, `r = 2*hypot(kx, ky)`, not the more common separable
    /// Cartesian form `W(kx)*W(ky)`. Radial is isotropic in-plane, which suits a benchmark whose
    /// edges are not axis-aligned; a scanner reproducing a specific vendor filter may need the
    /// separable convention instead.
    pub fn at(&self, kx: f64, ky: f64) -> f64 {
        let r = 2.0 * (kx * kx + ky * ky).sqrt(); // 0 at centre, 1 at the band edge
        match *self {
            KspaceWindow::None => 1.0,
            KspaceWindow::Hann => {
                if r >= 1.0 { 0.0 } else { 0.5 * (1.0 + (std::f64::consts::PI * r).cos()) }
            }
            KspaceWindow::Tukey { alpha } => {
                let a = alpha.clamp(0.0, 1.0);
                if r <= 1.0 - a {
                    1.0
                } else if r >= 1.0 {
                    0.0
                } else {
                    0.5 * (1.0 + (std::f64::consts::PI * (r - (1.0 - a)) / a.max(1e-12)).cos())
                }
            }
            KspaceWindow::Fermi { radius, width } => {
                1.0 / (1.0 + ((r - radius) / width.max(1e-12)).exp())
            }
        }
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
    /// Per-component variance of the reconstructed complex image at **full sampling, single coil,
    /// pre-combination**: `Var(Re n) = Var(Im n) = noise_variance`, so `E[|n|^2] = 2*noise_variance`.
    /// The per-k-space-sample variance is derived from the reconstruction normalization; the
    /// combined multi-coil variance emerges from the coil model as `V / sum_c s_c^2`.
    pub noise_variance: f64,
    pub partial_fourier: f64, // fraction of PE lines acquired (1.0 = full); skips low-ky lines
    /// Which PF line-dropping rule to use. See [`PartialFourierMode`]: the default is NOT exactly
    /// `partial_fourier` of the lines on an even matrix.
    pub pf_mode: PartialFourierMode,
    pub ghost_offset: f64,    // Nyquist ghost: kx offset (±) on odd/even PE lines (0 = off)
    pub eddy_strength: f64,   // linear (in-plane) eddy-current phase scale (0 = off)
    pub eddy_quad: f64,       // quadratic (x²,y²,z²) eddy-current phase scale (0 = off)
    pub eddy_phase: f64,      // eddy OBJECT-phase ramp: rad per unit (bvec·bval) per acquired voxel;
                              // direction- and b-dependent, imprints reconstructed phase (0 = off)
    pub eddy_tau: f64,        // eddy-current decay time (ms)
    pub n_spikes: usize,      // random k-space spikes per slice (0 = off) → herringbone
    pub spike_amplitude: f64, // spike magnitude as a fraction of the peak k-space sample
    pub window: KspaceWindow, // reconstruction apodization; None = unapodized rectangular
    pub n_coils: usize,       // receiver coils (1 = uniform single coil); ring-arranged sensitivities
    pub accel: usize,         // GRAPPA acceleration R (1 = fully sampled); undersamples PE lines
    pub acs_lines: usize,     // GRAPPA autocalibration lines (fully-sampled central PE band)
    pub seed: u64,            // mixed into every derived per-slice seed; 0 reproduces legacy output
}

/// Spatial sensitivity of `coil` (of `n_coils` arranged in a ring) at continuous position
/// `(x, y)` in ACQUIRED-voxel units, on an `nx` x `ny` acquired matrix. Uniform for a single coil;
/// otherwise a Gaussian falloff from the coil position (distinct per coil — the spatial diversity
/// GRAPPA/SENSE need). Not normalized; the Roemer combine handles the overall scale.
///
/// The coordinates are continuous and acquired-voxel-based on purpose. This function is called
/// from two places that walk DIFFERENT grids — the forward model walks the oversampled simulation
/// grid, the Roemer combine walks the acquired matrix — and it must describe the same physical
/// field to both, or the combine divides by sensitivities the signal was never multiplied by.
/// Taking `(x: usize, nx: usize)` and centring on `nx/2` did not: at oversampling `o` the forward
/// model's field was displaced by `(o-1)/(2o)` acquired voxels relative to the combine's, because
/// the sim cells composing an acquired voxel are centred on it (`xoff = (o-1)/2`) rather than
/// starting at it. Sub-voxel and negligible for these very smooth Gaussians — and exactly zero for
/// the canonical Gibbs benchmark, which forces `n_coils = 1` — but wrong, and it would stop being
/// negligible the moment anyone supplied a sharper sensitivity map.
fn coil_sensitivity(coil: usize, n_coils: usize, x: f64, y: f64, nx: usize, ny: usize) -> f64 {
    if n_coils <= 1 {
        return 1.0;
    }
    let (cx, cy) = (nx as f64 / 2.0, ny as f64 / 2.0);
    let r = 0.6 * nx.max(ny) as f64;
    let ang = TAU * coil as f64 / n_coils as f64;
    let (px, py) = (cx + r * ang.cos(), cy + r * ang.sin());
    let d2 = (x - px).powi(2) + (y - py).powi(2);
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
            pf_mode: PartialFourierMode::Scanner,
            ghost_offset: 0.0,
            eddy_strength: 0.0,
            eddy_quad: 0.0,
            eddy_phase: 0.0,
            eddy_tau: 70.0,
            n_spikes: 0,
            spike_amplitude: 1.0,
            window: KspaceWindow::None,
            n_coils: 1,
            accel: 1,
            acs_lines: 24,
            seed: 0,
        }
    }
}

impl Acquisition {
    /// The HBCD-like protocol the `trxscan` binary ships: TE 88 ms, 6/8 scanner-style partial
    /// Fourier, 24 ACS lines, a subtle residual Nyquist ghost, no spikes, unapodized. The PE-train
    /// duration is pinned to the HBCD TotalReadoutTime (0.0917 s) for ANY matrix: distortion
    /// shift = fmap · ny · t_line, so `t_line = 91.7 ms / ny`. Everything else (noise, eddy,
    /// coils, GRAPPA, seed) is left at the [`Default`] and set by the caller.
    pub fn hbcd(ny: usize) -> Self {
        Acquisition {
            t_line: 91.7 / ny.max(1) as f64,
            t_echo: 88.0,
            t_inhom: 50.0,
            partial_fourier: 0.75,
            pf_mode: PartialFourierMode::Scanner,
            ghost_offset: 0.015,
            ..Default::default()
        }
    }
}

/// Per-PE-line readout times (ms): `t` from max echo, `tRf` from RF, `tRead` from the last
/// diffusion gradient (drives eddy-current decay).
fn line_times_for(epi: &SingleShotEpi, acq: &Acquisition) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (t, trf, mut tread) = line_times(epi);
    // Scanner-style partial Fourier: the readout physically starts `skip` lines into the nominal
    // train, so the eddy-decay clock (time since the readout began) starts there.
    let offset = pf_skipped_lines(epi.ky_max, acq) as f64 * epi.t_line;
    if offset > 0.0 {
        for v in tread.iter_mut() {
            *v -= offset;
        }
    }
    (t, trf, tread)
}

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

/// The per-line timing of the single-shot EPI readout an [`Acquisition`] implies on an
/// `nx` x `ny` matrix: what the forward model uses for relaxation, T2* decay and eddy decay.
/// Public so front ends can annotate k-space figures with the simulator's own timing.
#[derive(Debug, Clone, PartialEq)]
pub struct EpiTiming {
    /// ms from the k-space centre (echo) at the centre of each ky line, indexed by ky.
    pub t_ms: Vec<f64>,
    /// ms since the RF pulse, indexed by ky (`t_echo + t_ms`).
    pub t_rf_ms: Vec<f64>,
    /// ms since the readout started (drives eddy-current decay), indexed by ky. With
    /// scanner-style partial Fourier the clock starts at the first *acquired* line, so the
    /// skipped lines carry negative values.
    pub t_read_ms: Vec<f64>,
    /// ky lines in the order they are acquired (`ny` entries).
    pub order: Vec<usize>,
    /// ms per PE line.
    pub t_line: f64,
    /// Echo time (ms).
    pub t_echo: f64,
    /// ms per k-space sample (`t_line / nx`).
    pub dt: f64,
}

/// Readout timing per ky line for `acq` on an `nx` x `ny` matrix. See [`EpiTiming`].
pub fn epi_timing(nx: usize, ny: usize, acq: &Acquisition) -> EpiTiming {
    let epi = SingleShotEpi {
        kx_max: nx, ky_max: ny, t_line: acq.t_line, t_echo: acq.t_echo, reverse_phase: acq.reverse_phase,
    };
    let (t_ms, t_rf_ms, t_read_ms) = line_times_for(&epi, acq);
    let order = (0..ny).map(|line| epi.kspace_index(line * nx).1).collect();
    EpiTiming { t_ms, t_rf_ms, t_read_ms, order, t_line: acq.t_line, t_echo: acq.t_echo, dt: epi.dt() }
}

/// The full k-space trajectory: `(kx, ky, ms from the echo)` for every sample tick, in
/// acquisition order (`nx * ny` entries). Partial Fourier / GRAPPA line skipping is a property
/// of [`sampling_mask`], not of the trajectory.
pub fn epi_trajectory(nx: usize, ny: usize, acq: &Acquisition) -> Vec<(usize, usize, f64)> {
    let epi = SingleShotEpi {
        kx_max: nx, ky_max: ny, t_line: acq.t_line, t_echo: acq.t_echo, reverse_phase: acq.reverse_phase,
    };
    (0..nx * ny)
        .map(|tick| {
            let (kx, ky) = epi.kspace_index(tick);
            (kx, ky, epi.time_from_max_echo(tick))
        })
        .collect()
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
    /// Pre-readout object phase (radians) on the SIM grid: the object phase model, added to
    /// every voxel before encoding (`None` = real-valued object).
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

/// A rectangle with sub-voxel-positioned edges on all four sides, with exact fractional occupancy
/// in every boundary voxel.
///
/// Unlike [`step_hires`], this has edges along BOTH axes, so an artifact that acts on only one --
/// partial Fourier, which zero-fills phase-encode lines -- actually shows up. A readout-only step
/// leaves the whole PF axis of a factor grid inert.
pub fn box_hires(snx: usize, sny: usize, x0: f64, x1: f64, y0: f64, y1: f64) -> Vec<f32> {
    let cover = |lo: f64, hi: f64, a: f64, b: f64| -> f64 {
        (hi.min(b) - lo.max(a)).clamp(0.0, 1.0)
    };
    let mut v = vec![0.0f32; snx * sny];
    for y in 0..sny {
        let fy = cover(y as f64, y as f64 + 1.0, y0, y1);
        if fy <= 0.0 {
            continue;
        }
        for x in 0..snx {
            let fx = cover(x as f64, x as f64 + 1.0, x0, x1);
            v[x + snx * y] = (fx * fy) as f32;
        }
    }
    v
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

/// Which k-space samples are actually acquired: partial Fourier plus GRAPPA undersampling.
/// Layout `kx + nx*ky`.
///
/// This is the single source of truth for both the forward build and the noise, so signal and
/// noise can never disagree about what was sampled. The original defect (section 1, finding 2.5)
/// was exactly that disagreement: noise was added after the line skip, populating k-space that
/// was never acquired.
pub fn sampling_mask(nx: usize, ny: usize, acq: &Acquisition) -> Vec<bool> {
    let ys = ny / 2;
    let accel = acq.accel.max(1);
    let acs_half = (acq.acs_lines / 2) as i64;
    let mut m = vec![false; nx * ny];
    // Scanner mode: exactly round(ny*pf) consecutive lines; the FIRST lines of the train are
    // skipped. The forward train starts at high ky (`SingleShotEpi::kspace_index`), so the
    // skipped block is the high-ky end; reversed polarity starts at low ky and skips there.
    // Contiguous (legacy) mode keeps the same count but drops the LAST lines instead.
    let keep_n = (ny as f64 * acq.partial_fourier).round() as usize;
    for kyi in 0..ny {
        if acq.partial_fourier < 1.0 && acq.pf_mode == PartialFourierMode::Scanner {
            let skip = if acq.reverse_phase { kyi < ny - keep_n } else { kyi >= keep_n };
            if skip {
                continue;
            }
        } else if acq.partial_fourier < 1.0 && acq.pf_mode == PartialFourierMode::Contiguous {
            let skip = if acq.reverse_phase { kyi >= keep_n } else { kyi < ny - keep_n };
            if skip {
                continue;
            }
        } else if acq.partial_fourier < 1.0 {
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
        let acquired = accel <= 1
            || (kyi as i64 - ys as i64).abs() <= acs_half
            || kyi % accel == ys % accel;
        if !acquired {
            continue;
        }
        for kxi in 0..nx {
            m[kxi + nx * kyi] = true;
        }
    }
    m
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
    let (t_ms, trf_ms, tread_ms) = line_times_for(&epi, acq);
    let gradient = [inp.bvec[0] * inp.bval, inp.bvec[1] * inp.bval, inp.bvec[2] * inp.bval];
    // eddy currents affect diffusion-weighted volumes only (b0 gradient ≈ 0)
    let do_eddy = acq.eddy_strength != 0.0 && inp.bval.abs() > 1e-9;
    let do_eddy_phase = acq.eddy_phase != 0.0 && inp.bval.abs() > 1e-9;
    // acquired-matrix centres (k-space indexing) and sim-grid centres (image indexing)
    // Centred k-space indexing: the acquired band is [-n/2, n/2-1], asymmetric about k=0 by one
    // sample. This is deliberate, not an off-by-one -- real even-matrix Cartesian acquisitions
    // cover exactly this range. It gives a real object a small deterministic imaginary component,
    // which is NOT object phase; see `phase.rs` for that. Pinned by
    // `even_matrix_window_asymmetry_is_intentional`.
    let (xs, ys, zs) = (nx / 2, ny / 2, nz / 2);
    let (ox, oy) = (snx / nx, sny / ny); // in-plane oversampling factors
    // The sim-grid centre is the IMAGE of the acquired centre, o*(n/2), not snx/2: for an odd
    // acquired matrix those differ by one sim cell, i.e. (1/o) of an acquired voxel in-plane.
    let (sxs, sys) = (ox * xs, oy * ys);
    // Half-cell alignment. Sim cell `x` covers [x, x+1) in sim units, so its centre is at x+0.5;
    // acquired cell `X` covers o cells and is centred at o*X + o/2. Aligning sample index `x` with
    // `o*X` — as a bare index substitution does — therefore misregisters the object against the
    // reconstruction grid by (o-1)/2 sim cells, i.e. (o-1)/(2o) of an ACQUIRED voxel: 0.44 voxels
    // at o=8. Since this whole model turns on sub-voxel edge position, that shift is fatal and the
    // acquired k-space is measured from the acquired grid's centre instead. Exactly zero at o=1.
    let (xoff, yoff) = ((ox as f64 - 1.0) / 2.0, (oy as f64 - 1.0) / 2.0);
    let at = |x: usize, y: usize| x + snx * y; // SIM-grid image index
    let nvox = snx * sny;

    // Which samples are acquired (partial Fourier + GRAPPA undersampling). Single source of
    // truth, shared with the noise below so signal and noise cannot disagree.
    let mask = sampling_mask(nx, ny, acq);

    // ---- forward: k[kx,ky] = (1/N) Σ_x e^{i2π kx x} Σ_y mod_ky(x,y) e^{i2π ky y} ----
    //
    // The same O(N³) sum Fiberfox evaluates, but organised by how each factor of mod_ky depends
    // on the PE line, so that the inner loop is pure complex arithmetic with no transcendentals:
    //
    //   static      amp(r)·e^{iφ0(r)}: signal_scale, coil sensitivity, object phase, the eddy
    //               object-phase ramp — evaluated once per slice and coil;
    //   affine      e^{i2π(rate(r)·t(ky) + ky_norm·(y−c))}: the fieldmap (rate = fmap) and the
    //               replayed linear eddy shear (rate = a·pos/TRT) together with the y-DFT kernel.
    //               `t` is affine in the line index per line parity (`readout.rs`), so this
    //               factor advances between acquired lines by a per-voxel ROTOR that depends only
    //               on the step (Δky, parity) — a complex multiply per voxel per line. The rotors
    //               are memoised per step and checked against the actual Δt, so an exotic
    //               readout timing degrades to the closed form rather than to a wrong answer;
    //   per line    the compartment relaxations e^{-tRf/T2_c − |t|/tInhom}: one scalar per
    //               compartment per line;
    //   separable   the gradient-model eddy polynomial × exp(-tRead/τ)·t: e^{iθ_x(x)}·e^{iθ_y(y)}
    //               ·e^{iθ_z}, i.e. snx + sny cis per line, not snx·sny.
    //
    // Round-off of the rotor recurrence is ~ny·ε; the state is re-anchored to the closed form
    // every `REANCHOR` acquired lines. Pinned against the literal per-line sum by
    // `restructured_forward_matches_the_literal_sum`.
    const REANCHOR: usize = 32;
    let zc = z as f64 - zs as f64;
    let zc_c = z as f64 - (nz as f64 - 1.0) / 2.0;
    let mut amp = vec![0.0f64; nvox];
    let mut phi0 = vec![0.0f64; nvox];
    let mut rate = vec![0.0f64; nvox];
    for y in 0..sny {
        for x in 0..snx {
            let i = at(x, y);
            // TWO coordinate frames, named apart on purpose. Both are in acquired-voxel
            // UNITS and both carry the half-cell registration (`xoff = (o-1)/2`), but they
            // have DIFFERENT ORIGINS, and a quantity evaluated in the wrong one is wrong by
            // half a FOV rather than by a sub-voxel amount:
            //
            //   xa — ABSOLUTE on the acquired grid, 0 .. nx-1, averaging to exactly `v` over
            //        the o sim cells of acquired voxel `v`. This is the frame the Roemer
            //        combine walks (`x in 0..nx`), so anything the combine must agree with —
            //        `coil_sensitivity`, whose coil ring is centred on nx/2 — uses it.
            //   xc — CENTRED on the acquired image, -nx/2 .. nx/2-1. The eddy polynomial is
            //        an expansion about the image centre and needs this one.
            //
            // Conflating them is not hypothetical: an earlier revision passed `xc` to
            // `coil_sensitivity`, which put the forward model's sensitivity field half a FOV
            // from the combine's at EVERY oversampling factor, o = 1 included (28% of peak).
            // Pinned end-to-end by
            // `multicoil_roemer_reproduces_the_single_coil_image_at_every_oversampling`.
            let (xa, ya) = ((x as f64 - xoff) / ox as f64, (y as f64 - yoff) / oy as f64);
            amp[i] = acq.signal_scale * coil_sensitivity(coil, ncoils, xa, ya, nx, ny);
            let (xc, yc) = (
                (x as f64 - sxs as f64 - xoff) / ox as f64,
                (y as f64 - sys as f64 - yoff) / oy as f64,
            );
            // Pre-readout object phase, already in radians (outside the TAU factor).
            let mut p0 = inp.phase0.map_or(0.0, |p| p[i]);
            if do_eddy_phase {
                // Eddy OBJECT-phase ramp: constant across the readout (NOT ∝ ky), so it
                // imprints the reconstructed object phase rather than distorting geometry.
                // Direction- and b-dependent (∝ gradient = bvec·bval), it reproduces the
                // per-volume phase-ramp variation real DWI shows (∝ gradient direction).
                // z centred on the volume's geometric centre (nz−1)/2; the eddy polynomial
                // below keeps Fiberfox's k-space convention `zc = z − nz/2`.
                p0 += acq.eddy_phase * (gradient[0] * xc + gradient[1] * yc + gradient[2] * zc_c);
            }
            phi0[i] = p0;
            rate[i] = if acq.do_distortions { fmap[i] as f64 } else { 0.0 };
        }
    }
    // Centred, half-cell-registered sim-grid coordinates in ACQUIRED-voxel units, per axis.
    let xc_of = |x: usize| (x as f64 - sxs as f64 - xoff) / ox as f64;
    let yc_of = |y: usize| (y as f64 - sys as f64 - yoff) / oy as f64;
    let ky_norm_of = |kyi: usize| (kyi as f64 - ys as f64) / sny as f64;

    // Rotor state: amp·e^{i(φ0 + 2π(rate·t + ky_norm·(y−c)))} for the current line. Allocated
    // only when the rotor path runs (no NUFFT rows), below.
    let (mut st_re, mut st_im): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    let set_state = |kyi: usize, st_re: &mut [f64], st_im: &mut [f64]| {
        let t = t_ms[kyi] / 1000.0;
        let kyn = ky_norm_of(kyi);
        for y in 0..sny {
            let yy = y as f64 - sys as f64 - yoff;
            for x in 0..snx {
                let i = at(x, y);
                let ph = phi0[i] + TAU * (rate[i] * t + kyn * yy);
                st_re[i] = amp[i] * ph.cos();
                st_im[i] = amp[i] * ph.sin();
            }
        }
    };
    // Memoised rotors, keyed by (Δky, parity of the source line); each remembers the (Δt, Δky_norm)
    // it was built for so a mismatching step falls back to the closed form.
    struct Rotor { key: (usize, usize), dt: f64, dkyn: f64, re: Vec<f64>, im: Vec<f64> }
    let mut rotors: Vec<Rotor> = Vec::new();

    // Compartment relaxation weights for this line.
    let mut rel = vec![1.0f64; compartments.len()];
    // Separable gradient-model eddy factors.
    let (mut ex_re, mut ex_im) = (vec![1.0f64; snx], vec![0.0f64; snx]);
    let (mut ey_re, mut ey_im) = (vec![1.0f64; sny], vec![0.0f64; sny]);
    // y-sum result and x-stage inputs.
    let (mut g_re, mut g_im) = (vec![0.0f64; snx], vec![0.0f64; snx]);
    let mut xstage = XStage::new(snx, nx, xs, sxs, xoff, acq.ghost_offset);

    // ---- NUFFT y-stage (feature `kspace`): the fieldmap as geometry ----
    // With t affine in the line index per parity, t(ky) = τ·ky + t0 + δ·[ky odd], the phase
    // 2π(rate·t + ky_norm·(y−c)) is 2π(ky−ys)·v/sny + const, with the SOURCE displaced to
    // v = (y − c) + sny·τ·rate — every voxel contributes from its distorted PE position. The
    // per-line y-sum then collapses to one type-1 NUFFT per column (per compartment, since the
    // relaxation is an output-side scalar; per parity, since δ adds a static phase to the odd
    // lines). The gradient-model eddy polynomial has a non-affine time profile and keeps the
    // rotor path. The timing model is checked, not assumed.
    #[cfg(feature = "kspace")]
    let nufft_rows: Option<Vec<(Vec<f64>, Vec<f64>)>> = (!do_eddy && ny >= 3).then(|| {
        let tau = (t_ms[2] - t_ms[0]) / 2.0;
        let t0 = t_ms[0];
        let delta = t_ms[1] - (tau + t0);
        let tmax = t_ms.iter().fold(0.0f64, |m, t| m.max(t.abs())).max(1e-300);
        let affine = (0..ny).all(|k| {
            let pred = tau * k as f64 + t0 + if k % 2 == 1 { delta } else { 0.0 };
            (t_ms[k] - pred).abs() <= 1e-9 * tmax
        });
        if !affine {
            return None;
        }
        let (tau_s, delta_s) = (tau / 1000.0, delta / 1000.0);
        let t_centre = (tau * ys as f64 + t0) / 1000.0; // even-line time at the centre line
        let ncomp = compartments.len();
        let mut nufft = crate::nufft::Nufft1::new(sny, -(ys as i64), ny, 1e-13);
        // per-column buffers: positions, then (compartment, parity) weight vectors
        let mut pos = vec![0.0f64; sny];
        let mut wre = vec![vec![0.0f64; sny]; 2 * ncomp];
        let mut wim = vec![vec![0.0f64; sny]; 2 * ncomp];
        let mut out: Vec<(Vec<f64>, Vec<f64>)> = (0..2 * ncomp).map(|_| (Vec::new(), Vec::new())).collect();
        // rows[c] = (re, im) of the y-summed line, layout x + snx*kyi
        let mut rows: Vec<(Vec<f64>, Vec<f64>)> = (0..ncomp).map(|_| (vec![0.0; snx * ny], vec![0.0; snx * ny])).collect();
        for x in 0..snx {
            for y in 0..sny {
                let i = at(x, y);
                pos[y] = (y as f64 - sys as f64 - yoff) + sny as f64 * tau_s * rate[i];
                let ph = phi0[i] + TAU * rate[i] * t_centre;
                let (c0, s0) = (ph.cos(), ph.sin());
                let phd = TAU * rate[i] * delta_s;
                let (cd, sd) = (phd.cos(), phd.sin());
                // odd-line phase = even phase + δ term
                let (c1, s1) = (c0 * cd - s0 * sd, c0 * sd + s0 * cd);
                for (c, comp) in compartments.iter().enumerate() {
                    let a = amp[i] * comp[i] as f64;
                    wre[2 * c][y] = a * c0;
                    wim[2 * c][y] = a * s0;
                    wre[2 * c + 1][y] = a * c1;
                    wim[2 * c + 1][y] = a * s1;
                }
            }
            let weights: Vec<(&[f64], &[f64])> = (0..2 * ncomp).map(|q| (wre[q].as_slice(), wim[q].as_slice())).collect();
            nufft.run(&pos, &weights, &mut out);
            for c in 0..ncomp {
                for kyi in 0..ny {
                    let q = 2 * c + (kyi % 2);
                    rows[c].0[x + snx * kyi] = out[q].0[kyi];
                    rows[c].1[x + snx * kyi] = out[q].1[kyi];
                }
            }
        }
        Some(rows)
    }).flatten();
    #[cfg(not(feature = "kspace"))]
    let nufft_rows: Option<Vec<(Vec<f64>, Vec<f64>)>> = None;
    if nufft_rows.is_none() {
        st_re = vec![0.0f64; nvox];
        st_im = vec![0.0f64; nvox];
    }

    let mut kspace = vec![C::ZERO; nx * ny];
    let mut prev: Option<usize> = None;
    let mut since_anchor = 0usize;
    for kyi in 0..ny {
        // Not acquired (partial Fourier, or GRAPPA undersampling): this k-space row stays
        // zero and, critically, receives no noise either.
        if !mask[nx * kyi] {
            continue;
        }
        let t = t_ms[kyi] / 1000.0; // seconds
        let trf = trf_ms[kyi];
        // Advance the affine phase to this line: closed form on the first line and every
        // REANCHOR lines, otherwise one rotor multiply per voxel.
        if nufft_rows.is_none() {
            match prev {
                Some(p) if since_anchor < REANCHOR => {
                    let key = (kyi - p, p % 2);
                    let (dt, dkyn) = (t - t_ms[p] / 1000.0, ky_norm_of(kyi) - ky_norm_of(p));
                    let idx = rotors.iter().position(|r| r.key == key).unwrap_or_else(|| {
                        let (mut re, mut im) = (vec![0.0f64; nvox], vec![0.0f64; nvox]);
                        for y in 0..sny {
                            let yy = y as f64 - sys as f64 - yoff;
                            for x in 0..snx {
                                let i = at(x, y);
                                let ph = TAU * (rate[i] * dt + dkyn * yy);
                                re[i] = ph.cos();
                                im[i] = ph.sin();
                            }
                        }
                        rotors.push(Rotor { key, dt, dkyn, re, im });
                        rotors.len() - 1
                    });
                    let r = &rotors[idx];
                    if (r.dt - dt).abs() <= 1e-12 * dt.abs().max(1e-300) && r.dkyn == dkyn {
                        for i in 0..nvox {
                            let (a, b) = (st_re[i], st_im[i]);
                            st_re[i] = a * r.re[i] - b * r.im[i];
                            st_im[i] = a * r.im[i] + b * r.re[i];
                        }
                        since_anchor += 1;
                    } else {
                        set_state(kyi, &mut st_re, &mut st_im);
                        since_anchor = 0;
                    }
                }
                _ => {
                    set_state(kyi, &mut st_re, &mut st_im);
                    since_anchor = 0;
                }
            }
            prev = Some(kyi);
        }

        for (c, w) in rel.iter_mut().enumerate() {
            *w = if acq.do_relaxation {
                (-trf / t2[c] as f64 - t.abs() * 1000.0 / acq.t_inhom).exp()
            } else {
                1.0
            };
        }
        // Gradient-model eddy field growing through the readout: linear (g·pos) plus a
        // quadratic (g·pos²) term — the polynomial eddy/TORTOISE fit — times
        // exp(-tRead/τ)·t (itkKspaceImageFilter.cpp:354), so it grows with ky → geometric
        // DISTORTION. The polynomial is a sum over axes, so its phase factor is separable.
        let mut ez = (1.0f64, 0.0f64);
        if do_eddy {
            let eddy_decay = (-tread_ms[kyi] / acq.eddy_tau).exp() * t;
            for x in 0..snx {
                let xc = xc_of(x);
                let ph = TAU * (acq.eddy_strength * gradient[0] * xc + acq.eddy_quad * gradient[0] * xc * xc) * eddy_decay;
                ex_re[x] = ph.cos();
                ex_im[x] = ph.sin();
            }
            for y in 0..sny {
                let yc = yc_of(y);
                let ph = TAU * (acq.eddy_strength * gradient[1] * yc + acq.eddy_quad * gradient[1] * yc * yc) * eddy_decay;
                ey_re[y] = ph.cos();
                ey_im[y] = ph.sin();
            }
            let ph = TAU * (acq.eddy_strength * gradient[2] * zc + acq.eddy_quad * gradient[2] * zc * zc) * eddy_decay;
            ez = (ph.cos(), ph.sin());
        }

        // inner y-sum over the SIM grid → g(x): Σ_y (Σ_c rel_c comp_c) · state · e_y
        g_re.iter_mut().for_each(|v| *v = 0.0);
        g_im.iter_mut().for_each(|v| *v = 0.0);
        if let Some(rows) = &nufft_rows {
            for (c, (rre, rim)) in rows.iter().enumerate() {
                let row = &rre[snx * kyi..snx * (kyi + 1)];
                let rowi = &rim[snx * kyi..snx * (kyi + 1)];
                for x in 0..snx {
                    g_re[x] += rel[c] * row[x];
                    g_im[x] += rel[c] * rowi[x];
                }
            }
        } else {
            for y in 0..sny {
                let row = snx * y;
                let (eyr, eyi) = (ey_re[y], ey_im[y]);
                for x in 0..snx {
                    let i = row + x;
                    let mut w = 0.0f64;
                    for (c, comp) in compartments.iter().enumerate() {
                        w += rel[c] * comp[i] as f64;
                    }
                    let (sr, si) = (st_re[i], st_im[i]);
                    let (mr, mi) = if do_eddy { (sr * eyr - si * eyi, sr * eyi + si * eyr) } else { (sr, si) };
                    g_re[x] += w * mr;
                    g_im[x] += w * mi;
                }
            }
        }
        if do_eddy {
            for x in 0..snx {
                let (a, b) = (g_re[x], g_im[x]);
                let (fr, fi) = (ex_re[x] * ez.0 - ex_im[x] * ez.1, ex_re[x] * ez.1 + ex_im[x] * ez.0);
                g_re[x] = a * fr - b * fi;
                g_im[x] = a * fi + b * fr;
            }
        }
        // x-DFT evaluated only at the acquired kx band, with the Nyquist (N/2) ghost: an
        // alternating readout-line kx offset (gradient-delay mismatch).
        let ghost_shift = if kyi % 2 == 1 { -acq.ghost_offset } else { acq.ghost_offset };
        xstage.run(&g_re, &g_im, ghost_shift, &mut kspace[nx * kyi..nx * (kyi + 1)]);
    }
    let n_inv = 1.0 / nvox as f64;
    for k in kspace.iter_mut() {
        k.re *= n_inv;
        k.im *= n_inv;
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
        // Spikes land only on ACQUIRED samples. Choosing arbitrary matrix coordinates could put
        // a spike on a partial-Fourier or GRAPPA-omitted line, which the scanner never read.
        let acquired: Vec<usize> = (0..nx * ny).filter(|&i| mask[i]).collect();
        if !acquired.is_empty() {
            let mut rng = Rng(inp.slice_seed ^ 0xA5A5_1234_5678_9ABC);
            for _ in 0..acq.n_spikes {
                let pick = acquired[(rng.next_u64() as usize) % acquired.len()];
                kspace[pick] = spike;
            }
        }
    }

    // Complex k-space noise, on ACQUIRED samples only.
    //
    // `noise_variance` is the PER-COMPONENT variance of the reconstructed complex image under
    // full sampling, single-coil, pre-combination: Var(Re n) = Var(Im n) = noise_variance, so
    // E[|n|^2] = 2*noise_variance. The per-sample variance follows from the reconstruction
    // normalization, which uses the ACQUIRED matrix, never the simulation grid.
    //
    // Masking alone produces the sqrt(f) scaling; there is deliberately NO additional
    // sampled-fraction factor, which would double-count it and give SD proportional to f.
    if acq.noise_variance > 0.0 {
        let mut rng = Rng((inp.slice_seed ^ (coil as u64).wrapping_mul(0x9E37_79B9)) | 1);
        let sigma = (acq.noise_variance / (nx * ny) as f64).sqrt();
        for (i, k) in kspace.iter_mut().enumerate() {
            if mask[i] {
                k.re += rng.gauss() * sigma;
                k.im += rng.gauss() * sigma;
            }
        }
    }

    kspace
}


/// The x-stage of the forward model: one PE line's y-summed row `g[x]` (sim grid, `snx` points)
/// → its `nx` acquired kx samples, `kx_norm = (kxi − xs + ghost)/snx`, measured from the acquired
/// grid's centre `sxs + xoff`. The Nyquist-ghost offset is an off-grid frequency shift, i.e. a
/// linear phase on `g`; the centre offset is a linear phase on the output; between them sits a
/// plain length-`snx` e^{+i2π} transform (rustfft's unnormalised inverse), of which the bins
/// `(kxi − xs) mod snx` are kept. Without the `kspace` feature the identical sum is evaluated
/// from precomputed twiddle tables (one per ghost polarity).
struct XStage {
    snx: usize,
    nx: usize,
    #[cfg(feature = "kspace")]
    xs: usize,
    #[cfg(feature = "kspace")]
    fft: std::sync::Arc<dyn rustfft::Fft<f64>>,
    #[cfg(feature = "kspace")]
    buf: Vec<rustfft::num_complex::Complex<f64>>,
    #[cfg(feature = "kspace")]
    scratch: Vec<rustfft::num_complex::Complex<f64>>,
    /// Input ghost ramps `e^{i2π(±ghost)(x − sxs − xoff)/snx}` for [+ghost, −ghost].
    #[cfg(feature = "kspace")]
    ramp: [Vec<(f64, f64)>; 2],
    /// Output centre phase `e^{−i2π m (sxs + xoff)/snx}`, m = kxi − xs.
    #[cfg(feature = "kspace")]
    centre: Vec<(f64, f64)>,
    /// Twiddles `e^{i2π (m ± ghost)(x − sxs − xoff)/snx}`, layout `x + snx*kxi`, per polarity.
    #[cfg(not(feature = "kspace"))]
    tw: [Vec<(f64, f64)>; 2],
}

impl XStage {
    fn new(snx: usize, nx: usize, xs: usize, sxs: usize, xoff: f64, ghost: f64) -> XStage {
        let c = sxs as f64 + xoff;
        #[cfg(feature = "kspace")]
        {
            let fft = rustfft::FftPlanner::<f64>::new().plan_fft_inverse(snx);
            let scratch = vec![rustfft::num_complex::Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
            let ramp = [ghost, -ghost].map(|gh| {
                (0..snx).map(|x| { let p = TAU * gh * (x as f64 - c) / snx as f64; (p.cos(), p.sin()) }).collect()
            });
            let centre = (0..nx)
                .map(|kxi| { let p = -TAU * (kxi as f64 - xs as f64) * c / snx as f64; (p.cos(), p.sin()) })
                .collect();
            XStage { snx, nx, xs, fft, buf: vec![rustfft::num_complex::Complex::new(0.0, 0.0); snx], scratch, ramp, centre }
        }
        #[cfg(not(feature = "kspace"))]
        {
            let tw = [ghost, -ghost].map(|gh| {
                let mut t = Vec::with_capacity(snx * nx);
                for kxi in 0..nx {
                    let kx_norm = (kxi as f64 - xs as f64 + gh) / snx as f64;
                    for x in 0..snx {
                        let p = TAU * kx_norm * (x as f64 - c);
                        t.push((p.cos(), p.sin()));
                    }
                }
                t
            });
            XStage { snx, nx, tw }
        }
    }

    /// `ghost_shift` is `±ghost` for this line's parity; `out` receives the nx samples (unnormalised).
    fn run(&mut self, g_re: &[f64], g_im: &[f64], ghost_shift: f64, out: &mut [C]) {
        let pol = if ghost_shift < 0.0 { 1 } else { 0 };
        #[cfg(feature = "kspace")]
        {
            for x in 0..self.snx {
                let (rr, ri) = self.ramp[pol][x];
                self.buf[x] = rustfft::num_complex::Complex::new(g_re[x] * rr - g_im[x] * ri, g_re[x] * ri + g_im[x] * rr);
            }
            self.fft.process_with_scratch(&mut self.buf, &mut self.scratch);
            for kxi in 0..self.nx {
                let m = kxi as i64 - self.xs as i64;
                let bin = m.rem_euclid(self.snx as i64) as usize;
                let (cr, ci) = self.centre[kxi];
                let v = self.buf[bin];
                out[kxi] = C { re: v.re * cr - v.im * ci, im: v.re * ci + v.im * cr };
            }
        }
        #[cfg(not(feature = "kspace"))]
        {
            let tw = &self.tw[pol];
            for kxi in 0..self.nx {
                let row = &tw[self.snx * kxi..self.snx * (kxi + 1)];
                let (mut ar, mut ai) = (0.0f64, 0.0f64);
                for x in 0..self.snx {
                    let (tr, ti) = row[x];
                    ar += g_re[x] * tr - g_im[x] * ti;
                    ai += g_re[x] * ti + g_im[x] * tr;
                }
                out[kxi] = C { re: ar, im: ai };
            }
        }
    }
}

/// The acquired k-space of the first coil, before reconstruction. Layout `kx + nx*ky`.
/// Used by the convergence tests (spec 4.1.6); not part of the simulation path.
pub fn simulate_slice_kspace(inp: &SliceInput, acq: &Acquisition) -> Vec<(f64, f64)> {
    build_coil_kspace(inp, acq, 0, acq.n_coils.max(1))
        .into_iter()
        .map(|c| (c.re, c.im))
        .collect()
}

/// Sample a [`PhaseModel`] onto one slice of the simulation grid.
///
/// Positions are in **acquired-voxel units** from the FOV centre, so the same coefficients describe
/// the same physical field at any oversampling factor `o`. The half-cell offset matches the forward
/// transform's registration convention (see [`simulate_slice`]).
pub fn phase_slice(
    model: &PhaseModel,
    shot: &ShotPhase,
    snx: usize,
    sny: usize,
    o: usize,
    z: usize,
    nz: usize,
) -> Vec<f64> {
    let (sxs, sys) = (snx as f64 / 2.0, sny as f64 / 2.0);
    let off = (o as f64 - 1.0) / 2.0;
    let zc = z as f64 - nz as f64 / 2.0;
    let mut v = vec![0.0f64; snx * sny];
    for y in 0..sny {
        for x in 0..snx {
            let r = [
                (x as f64 - sxs - off) / o as f64,
                (y as f64 - sys - off) / o as f64,
                zc,
            ];
            v[x + snx * y] = model.at(r, shot);
        }
    }
    v
}

/// Which intermediates [`simulate_slice_full`] should retain besides the combined image.
/// Everything off is what the production path uses; each flag costs one `n_coils * nx * ny`
/// complex buffer per slice.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SliceCapture {
    /// Per-coil k-space as acquired: after spikes and k-space noise, BEFORE GRAPPA, unwindowed;
    /// un-acquired lines (partial Fourier, undersampling) are exactly zero.
    pub acquired: bool,
    /// Per-coil k-space as reconstructed: after GRAPPA and the reconstruction window — what is
    /// inverse-transformed.
    pub reconstructed: bool,
    /// Per-coil complex images, before the Roemer combine.
    pub coil_images: bool,
}

impl SliceCapture {
    pub const ALL: SliceCapture = SliceCapture { acquired: true, reconstructed: true, coil_images: true };
}

/// One slice's reconstruction with the requested intermediates. Complex values are `[re, im]`
/// pairs (the memory layout of `numpy.complex128`); all 2-D buffers are `kx + nx*ky` /
/// `x + nx*y`.
#[derive(Debug, Clone)]
pub struct SliceRecon {
    pub nx: usize,
    pub ny: usize,
    pub n_coils: usize,
    /// Which k-space samples were acquired (= [`sampling_mask`]).
    pub mask: Vec<bool>,
    /// Per coil; see [`SliceCapture::acquired`].
    pub acquired: Option<Vec<Vec<[f64; 2]>>>,
    /// Per coil; see [`SliceCapture::reconstructed`].
    pub reconstructed: Option<Vec<Vec<[f64; 2]>>>,
    /// Per coil; see [`SliceCapture::coil_images`].
    pub coil_images: Option<Vec<Vec<[f64; 2]>>>,
    /// Per coil, the real sensitivity on the acquired grid (what the combine divides by).
    pub sensitivities: Vec<Vec<f64>>,
    /// The Roemer-combined complex image, before the `f32` cast [`simulate_slice`] applies.
    pub combined: Vec<[f64; 2]>,
}

/// Simulate one slice: compartment images on the SIM grid (each `snx*sny`, layout `x + snx*y`) →
/// complex image on the ACQUIRED matrix (`nx*ny`). `t2` is the per-compartment T2 (ms); `fmap` is
/// the off-resonance field (Hz), same sim-grid layout. Only the central `nx*ny` block of the sim
/// grid's k-space is evaluated, so truncation to the nominal band happens during the forward
/// transform rather than by discarding a computed k-space (spec 3.1).
pub fn simulate_slice(inp: &SliceInput, acq: &Acquisition) -> Vec<(f32, f32)> {
    simulate_slice_full(inp, acq, SliceCapture::default())
        .combined
        .iter()
        .map(|c| (c[0] as f32, c[1] as f32))
        .collect()
}

/// [`simulate_slice`] with the intermediates selected by `cap` retained. The combined image is
/// identical to `simulate_slice`'s before its `f32` cast; capturing changes no arithmetic.
pub fn simulate_slice_full(inp: &SliceInput, acq: &Acquisition, cap: SliceCapture) -> SliceRecon {
    let [nx, ny] = inp.acq_matrix;
    let (xs, ys) = (nx / 2, ny / 2);
    let kat = |kx: usize, ky: usize| kx + nx * ky; // acquired k-space / acquired-image index
    let pairs = |v: &[C]| -> Vec<[f64; 2]> { v.iter().map(|c| [c.re, c.im]).collect() };

    // Build each coil's k-space (undersampled for GRAPPA when accel>1), reconstruct, then combine.
    let ncoils = acq.n_coils.max(1);
    let accel = acq.accel.max(1);
    let mut coil_kspace: Vec<Vec<C>> = Vec::with_capacity(ncoils);
    for coil in 0..ncoils {
        coil_kspace.push(build_coil_kspace(inp, acq, coil, ncoils));
    }
    let acquired = cap.acquired.then(|| coil_kspace.iter().map(|k| pairs(k)).collect());

    // GRAPPA: fill the un-acquired PE lines with a kernel calibrated on the ACS band across coils.
    if accel > 1 {
        grappa_reconstruct(&mut coil_kspace, nx, ny, ys, accel, acq.acs_lines);
    }

    // Reconstruction window: after GRAPPA, before the inverse transform, so it acts on
    // originally-acquired and GRAPPA-synthesized lines alike -- and on the noise those lines
    // already carry: K_filtered = W(k) * [K_signal(k) + n(k)].
    //
    // The position matters. Replacing the old zero_ringing block in situ would have kept the
    // order signal -> band limitation -> noise, recreating the very defect this work removes:
    // filtered signal combined with unfiltered noise (section 1, finding 2.5).
    if acq.window != KspaceWindow::None {
        for ks in coil_kspace.iter_mut() {
            for kyi in 0..ny {
                for kxi in 0..nx {
                    let kx = (kxi as f64 - xs as f64) / nx as f64;
                    let ky = (kyi as f64 - ys as f64) / ny as f64;
                    let w = acq.window.at(kx, ky);
                    let k = &mut ks[kxi + nx * kyi];
                    k.re *= w;
                    k.im *= w;
                }
            }
        }
    }
    let reconstructed = cap.reconstructed.then(|| coil_kspace.iter().map(|k| pairs(k)).collect());

    // inverse each coil, then phase-preserving Roemer combine with the known sensitivities:
    //   combined = Σ_c image_c · sens_c / Σ_c sens_c²  (real sensitivities here)
    let sensitivities: Vec<Vec<f64>> = (0..ncoils)
        .map(|coil| {
            (0..nx * ny)
                .map(|i| coil_sensitivity(coil, ncoils, (i % nx) as f64, (i / nx) as f64, nx, ny))
                .collect()
        })
        .collect();
    let mut wsum = vec![C::ZERO; nx * ny];
    let mut ssum = vec![0.0f64; nx * ny];
    let mut coil_images: Option<Vec<Vec<[f64; 2]>>> = cap.coil_images.then(Vec::new);
    for (coil, ks) in coil_kspace.iter().enumerate() {
        #[cfg(feature = "kspace")]
        let img = inverse_2d_fft(ks, nx, ny, xs, ys);
        #[cfg(not(feature = "kspace"))]
        let img = inverse_2d(ks, nx, ny, xs, ys);
        for y in 0..ny {
            for x in 0..nx {
                let i = kat(x, y);
                let s = sensitivities[coil][i];
                wsum[i] = wsum[i].add(img[i].scale(s));
                ssum[i] += s * s;
            }
        }
        if let Some(ci) = coil_images.as_mut() {
            ci.push(pairs(&img));
        }
    }
    let combined = (0..nx * ny)
        .map(|i| {
            let s = ssum[i].max(1e-12);
            [wsum[i].re / s, wsum[i].im / s]
        })
        .collect();
    SliceRecon {
        nx,
        ny,
        n_coils: ncoils,
        mask: sampling_mask(nx, ny, acq),
        acquired,
        reconstructed,
        coil_images,
        sensitivities,
        combined,
    }
}

/// Exact FFT reconstruction (feature `kspace`, backed by the well-tested `rustfft` crate).
/// Bit-for-bit equivalent to [`inverse_2d`] up to floating-point round-off: the centred convention
/// (`xs = nx/2`, `ys = ny/2`) is a fftshift, realised as linear-phase modulations of the input and
/// output plus a constant phase, so only the O(N log N) transform is delegated to rustfft.
#[cfg(feature = "kspace")]
fn inverse_2d_fft(kspace: &[C], nx: usize, ny: usize, xs: usize, ys: usize) -> Vec<C> {
    use rustfft::{num_complex::Complex, FftPlanner};
    let mut planner = FftPlanner::<f64>::new();
    let fx = planner.plan_fft_forward(nx);
    let fy = planner.plan_fft_forward(ny);
    let (nxf, nyf, xsf, ysf) = (nx as f64, ny as f64, xs as f64, ys as f64);
    // 1) pre-modulate: K'[kx,ky] = K · cis(2π(kx·xs/nx + ky·ys/ny))
    let mut buf: Vec<Complex<f64>> = (0..nx * ny)
        .map(|i| {
            let (kx, ky) = ((i % nx) as f64, (i / nx) as f64);
            let ph = TAU * (kx * xsf / nxf + ky * ysf / nyf);
            let (c, sn) = (ph.cos(), ph.sin());
            Complex::new(kspace[i].re * c - kspace[i].im * sn, kspace[i].re * sn + kspace[i].im * c)
        })
        .collect();
    // 2) forward FFT along x (rows are contiguous), then along y (strided columns)
    for ky in 0..ny {
        fx.process(&mut buf[ky * nx..(ky + 1) * nx]);
    }
    let mut col = vec![Complex::new(0.0, 0.0); ny];
    for kx in 0..nx {
        for y in 0..ny {
            col[y] = buf[y * nx + kx];
        }
        fy.process(&mut col);
        for y in 0..ny {
            buf[y * nx + kx] = col[y];
        }
    }
    // 3) post-modulate: out = F · cis(2π(x·xs/nx + y·ys/ny)) · cis(-2π(xs²/nx + ys²/ny))
    let const_ph = -TAU * (xsf * xsf / nxf + ysf * ysf / nyf);
    (0..nx * ny)
        .map(|i| {
            let (x, y) = ((i % nx) as f64, (i / nx) as f64);
            let ph = TAU * (x * xsf / nxf + y * ysf / nyf) + const_ph;
            let (c, sn) = (ph.cos(), ph.sin());
            C { re: buf[i].re * c - buf[i].im * sn, im: buf[i].re * sn + buf[i].im * c }
        })
        .collect()
}

/// Inverse 2D DFT (separable direct sums): complex k-space → complex image, centred convention.
#[cfg(any(not(feature = "kspace"), test))]
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

/// Everything [`simulate_acquisition`] needs besides the [`Acquisition`] protocol: the object
/// (per-compartment images, their T2s and the off-resonance field, all on the **simulation**
/// grid), the diffusion scheme, the object phase model and the optional per-voxel noise map.
/// One shape for the CLI and for bindings.
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

/// Options for [`simulate_acquisition_complex`] beyond the object and protocol. `Default` is
/// exactly what [`simulate_acquisition`] does.
#[derive(Clone, Copy, Default)]
pub struct AcquisitionOptions<'a> {
    /// Full-FOV z index of each local slice (length `acq_dims[2]`), so a slab or a single slice
    /// sees the same eddy z-terms, object phase and per-slice seeds as the full-volume run.
    /// `None` = `0..nz`.
    pub slice_z: Option<&'a [usize]>,
    /// Full-FOV slice count when `slice_z` is given. `None` = `acq_dims[2]`.
    pub nz_full: Option<usize>,
    /// LOCAL slice indices whose k-space is retained (in this order). `None` = none.
    pub kspace_slices: Option<&'a [usize]>,
    /// Which per-coil intermediates to keep for those slices.
    pub capture: SliceCapture,
    /// Per-volume echo time (ms), length `ngrad`; `None` = `acq.t_echo` for every volume.
    pub t_echo_per_volume: Option<&'a [f64]>,
    /// Called after each volume completes with `(done, total)`. Volumes run in parallel under
    /// `par`, so it must be `Sync`; completion order is not volume order.
    pub progress: Option<&'a (dyn Fn(usize, usize) + Sync)>,
}

/// K-space and per-coil intermediates for the captured slices. Complex arrays are `[re, im]`
/// pairs in C order over the axes named; `nsl = slices.len()`.
#[derive(Debug, Clone)]
pub struct KspaceCapture {
    /// The local slice indices captured, in array order.
    pub slices: Vec<usize>,
    pub nx: usize,
    pub ny: usize,
    pub n_coils: usize,
    /// `[ky, kx]`, shared by every slice and volume (= [`sampling_mask`]).
    pub mask: Vec<bool>,
    /// `[coil, y, x]` real coil sensitivities on the acquired grid.
    pub sensitivities: Vec<f32>,
    /// `[vol, slice, coil, ky, kx]`; see [`SliceCapture::acquired`].
    pub acquired: Option<Vec<[f32; 2]>>,
    /// `[vol, slice, coil, ky, kx]`; see [`SliceCapture::reconstructed`].
    pub reconstructed: Option<Vec<[f32; 2]>>,
    /// `[vol, slice, coil, y, x]`; see [`SliceCapture::coil_images`].
    pub coil_images: Option<Vec<[f32; 2]>>,
    /// `[vol, slice, y, x]`: the Roemer-combined complex image before any image-space noise.
    pub combined: Vec<[f32; 2]>,
}

/// The complex acquisition output: real and imaginary parts in the 4-D layout
/// `(x + nx*(y + ny*z))*ngrad + g`, after image-space noise, plus the optional k-space capture.
#[derive(Debug, Clone)]
pub struct AcquisitionOutput {
    pub re: Vec<f32>,
    pub im: Vec<f32>,
    pub kspace: Option<KspaceCapture>,
}

/// The acquisition: every (volume, slice) of the object through [`simulate_slice`] — a finer
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
/// callback. See [`AcquisitionOptions`].
pub fn simulate_acquisition_complex(
    inp: &SimulationInput,
    acq: &Acquisition,
    opts: &AcquisitionOptions,
) -> AcquisitionOutput {
    let SimulationInput { sim_dims, acq_dims, ngrad, images, t2, fmap, bvals, bvecs, phase, seed, noise_sigma } = *inp;
    let [snx, sny, nz] = sim_dims;
    let [nx, ny, nzo] = acq_dims;
    assert_eq!(nz, nzo, "slice count must match; z is never oversampled");
    assert!(snx % nx == 0 && sny % ny == 0, "sim grid must be an integer multiple of the acquired matrix");
    let o = snx / nx;
    assert_eq!(o, sny / ny, "oversampling must match on both axes");
    let (nvox_sim, nvox_acq) = (snx * sny * nz, nx * ny * nz);
    let ncomp = images.len();
    for im in images {
        assert_eq!(im.len(), nvox_sim * ngrad, "compartment image is not on the simulation grid");
    }
    assert_eq!(fmap.len(), nvox_sim, "fieldmap is not on the simulation grid");
    if let Some(sz) = opts.slice_z {
        assert_eq!(sz.len(), nz, "slice_z must name every local slice");
    }
    if let Some(t) = opts.t_echo_per_volume {
        assert_eq!(t.len(), ngrad, "t_echo_per_volume must have one entry per volume");
    }
    let nz_full = opts.nz_full.unwrap_or(nz);
    let global_z = |zl: usize| opts.slice_z.map_or(zl, |s| s[zl]);
    let captured: Vec<usize> = opts.kspace_slices.map(|s| s.to_vec()).unwrap_or_default();
    for &zl in &captured {
        assert!(zl < nz, "kspace_slices index {zl} out of range for {nz} slices");
    }
    let done = std::sync::atomic::AtomicUsize::new(0);

    let per_vol = |g: usize| -> (Vec<f32>, Vec<f32>, Vec<SliceRecon>) {
        let acq_owned;
        let acq_g: &Acquisition = match opts.t_echo_per_volume {
            Some(t) => {
                acq_owned = Acquisition { t_echo: t[g], ..acq.clone() };
                &acq_owned
            }
            None => acq,
        };
        let (mut re_out, mut im_out) = (vec![0.0f32; nvox_acq], vec![0.0f32; nvox_acq]);
        let mut recons = Vec::with_capacity(captured.len());
        let mut cslices = vec![vec![0.0f32; snx * sny]; ncomp];
        let mut fslice = vec![0.0f32; snx * sny];
        for zl in 0..nz {
            let z = global_z(zl);
            for y in 0..sny {
                for x in 0..snx {
                    let vox = x + snx * (y + sny * zl);
                    for (c, img) in images.iter().enumerate() {
                        cslices[c][x + snx * y] = img[vox * ngrad + g];
                    }
                    fslice[x + snx * y] = fmap[vox];
                }
            }
            let refs: Vec<&[f32]> = cslices.iter().map(|v| v.as_slice()).collect();
            let shot = phase.diffusion.shot(bvals[g], bvecs[g], g, z, seed);
            let phi = phase_slice(phase, &shot, snx, sny, o, z, nz_full);
            let slice_seed = (g as u64)
                .wrapping_mul(0x100_0001)
                .wrapping_add(z as u64)
                .wrapping_mul(0x9E37)
                ^ seed;
            let want = captured.iter().any(|&c| c == zl);
            let cap = if want { opts.capture } else { SliceCapture::default() };
            let out = simulate_slice_full(
                &SliceInput {
                    compartments: &refs,
                    t2,
                    fmap: &fslice,
                    phase0: Some(&phi),
                    sim: [snx, sny],
                    acq_matrix: [nx, ny],
                    z,
                    nz: nz_full,
                    bvec: bvecs[g],
                    bval: bvals[g],
                    slice_seed,
                },
                acq_g,
                cap,
            );
            for y in 0..ny {
                for x in 0..nx {
                    let c = out.combined[x + nx * y];
                    let (mut re, mut im) = (c[0] as f32, c[1] as f32);
                    let vox = x + nx * (y + ny * zl);
                    if let Some(ns) = noise_sigma {
                        let sd = ns[vox] as f64;
                        if sd > 0.0 {
                            // deterministic per (volume, FULL-FOV voxel); parallel-safe (per_vol is over g)
                            let vox_g = x + nx * (y + ny * z);
                            let mut rng = Rng(
                                seed ^ (g as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                    ^ (vox_g as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9) | 1,
                            );
                            re += (rng.gauss() * sd) as f32;
                            im += (rng.gauss() * sd) as f32;
                        }
                    }
                    re_out[vox] = re;
                    im_out[vox] = im;
                }
            }
            if want {
                recons.push(out);
            }
        }
        if let Some(p) = opts.progress {
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            p(n, ngrad);
        }
        (re_out, im_out, recons)
    };

    #[cfg(feature = "par")]
    let vols: Vec<(Vec<f32>, Vec<f32>, Vec<SliceRecon>)> = {
        use rayon::prelude::*;
        (0..ngrad).into_par_iter().map(per_vol).collect()
    };
    #[cfg(not(feature = "par"))]
    let vols: Vec<(Vec<f32>, Vec<f32>, Vec<SliceRecon>)> = (0..ngrad).map(per_vol).collect();

    let (mut red, mut imd) = (vec![0.0f32; nvox_acq * ngrad], vec![0.0f32; nvox_acq * ngrad]);
    for (g, (r, i, _)) in vols.iter().enumerate() {
        for vox in 0..nvox_acq {
            red[vox * ngrad + g] = r[vox];
            imd[vox * ngrad + g] = i[vox];
        }
    }

    let kspace = if captured.is_empty() {
        None
    } else {
        // The captured slices are stored in `captured` order (per_vol pushes in zl order over a
        // filtered loop, so re-sort them to the requested order here).
        let nsl = captured.len();
        let ncoils = acq.n_coils.max(1);
        let per_slice = nx * ny;
        let f = |v: &[f64; 2]| [v[0] as f32, v[1] as f32];
        let mut acquired = opts.capture.acquired.then(|| Vec::with_capacity(ngrad * nsl * ncoils * per_slice));
        let mut reconstructed = opts.capture.reconstructed.then(|| Vec::with_capacity(ngrad * nsl * ncoils * per_slice));
        let mut coil_images = opts.capture.coil_images.then(|| Vec::with_capacity(ngrad * nsl * ncoils * per_slice));
        let mut combined = Vec::with_capacity(ngrad * nsl * per_slice);
        let mut sorted_local: Vec<usize> = captured.clone();
        sorted_local.sort_unstable();
        let pos_in_run = |zl: usize| sorted_local.iter().position(|&s| s == zl).unwrap();
        for (_, _, recons) in &vols {
            for &zl in &captured {
                let r = &recons[pos_in_run(zl)];
                if let Some(a) = acquired.as_mut() {
                    for coil in r.acquired.as_ref().unwrap() {
                        a.extend(coil.iter().map(f));
                    }
                }
                if let Some(a) = reconstructed.as_mut() {
                    for coil in r.reconstructed.as_ref().unwrap() {
                        a.extend(coil.iter().map(f));
                    }
                }
                if let Some(a) = coil_images.as_mut() {
                    for coil in r.coil_images.as_ref().unwrap() {
                        a.extend(coil.iter().map(f));
                    }
                }
                combined.extend(r.combined.iter().map(f));
            }
        }
        let first = &vols[0].2[0];
        let sensitivities = first.sensitivities.iter().flat_map(|c| c.iter().map(|&v| v as f32)).collect();
        Some(KspaceCapture {
            slices: captured,
            nx,
            ny,
            n_coils: ncoils,
            mask: first.mask.clone(),
            sensitivities,
            acquired,
            reconstructed,
            coil_images,
            combined,
        })
    };
    AcquisitionOutput { re: red, im: imd, kspace }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The literal per-line sum the production forward was restructured from: every
    /// factor re-evaluated for every (voxel, line). Kept verbatim as the oracle for
    /// `restructured_forward_matches_the_literal_sum`; not a code path.
    #[allow(clippy::needless_range_loop)]
    fn reference_coil_kspace(inp: &SliceInput, acq: &Acquisition, coil: usize, ncoils: usize) -> Vec<C> {
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
        let (t_ms, trf_ms, tread_ms) = line_times_for(&epi, acq);
        let gradient = [inp.bvec[0] * inp.bval, inp.bvec[1] * inp.bval, inp.bvec[2] * inp.bval];
        // eddy currents affect diffusion-weighted volumes only (b0 gradient ≈ 0)
        let do_eddy = acq.eddy_strength != 0.0 && inp.bval.abs() > 1e-9;
        let do_eddy_phase = acq.eddy_phase != 0.0 && inp.bval.abs() > 1e-9;
        // acquired-matrix centres (k-space indexing) and sim-grid centres (image indexing)
        // Centred k-space indexing: the acquired band is [-n/2, n/2-1], asymmetric about k=0 by one
        // sample. This is deliberate, not an off-by-one -- real even-matrix Cartesian acquisitions
        // cover exactly this range. It gives a real object a small deterministic imaginary component,
        // which is NOT object phase; see `phase.rs` for that. Pinned by
        // `even_matrix_window_asymmetry_is_intentional`.
        let (xs, ys, zs) = (nx / 2, ny / 2, nz / 2);
        let (ox, oy) = (snx / nx, sny / ny); // in-plane oversampling factors
        // The sim-grid centre is the IMAGE of the acquired centre, o*(n/2), not snx/2: for an odd
        // acquired matrix those differ by one sim cell, i.e. (1/o) of an acquired voxel in-plane.
        let (sxs, sys) = (ox * xs, oy * ys);
        // Half-cell alignment. Sim cell `x` covers [x, x+1) in sim units, so its centre is at x+0.5;
        // acquired cell `X` covers o cells and is centred at o*X + o/2. Aligning sample index `x` with
        // `o*X` — as a bare index substitution does — therefore misregisters the object against the
        // reconstruction grid by (o-1)/2 sim cells, i.e. (o-1)/(2o) of an ACQUIRED voxel: 0.44 voxels
        // at o=8. Since this whole model turns on sub-voxel edge position, that shift is fatal and the
        // acquired k-space is measured from the acquired grid's centre instead. Exactly zero at o=1.
        let (xoff, yoff) = ((ox as f64 - 1.0) / 2.0, (oy as f64 - 1.0) / 2.0);
        let at = |x: usize, y: usize| x + snx * y; // SIM-grid image index
        let kat = |kx: usize, ky: usize| kx + nx * ky; // acquired k-space / acquired-image index

        // Which samples are acquired (partial Fourier + GRAPPA undersampling). Single source of
        // truth, shared with the noise below so signal and noise cannot disagree.
        let mask = sampling_mask(nx, ny, acq);
        // ---- forward: build k-space, factored as  Σ_x e^{..kx x}[ Σ_y mod(x,y) e^{..ky y} ] ----
        // mod(x,y) depends on the PE line ky (through φ and relaxation), so the y-sum is recomputed
        // per ky, but that keeps the whole build at O(N³).
        let mut kspace = vec![C::ZERO; nx * ny];
        let n_inv = 1.0 / (snx * sny) as f64;
        for kyi in 0..ny {
            // Not acquired (partial Fourier, or GRAPPA undersampling): this k-space row stays
            // zero and, critically, receives no noise either.
            if !mask[nx * kyi] {
                continue;
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
                    // TWO coordinate frames, named apart on purpose. Both are in acquired-voxel
                    // UNITS and both carry the half-cell registration (`xoff = (o-1)/2`), but they
                    // have DIFFERENT ORIGINS, and a quantity evaluated in the wrong one is wrong by
                    // half a FOV rather than by a sub-voxel amount:
                    //
                    //   xa — ABSOLUTE on the acquired grid, 0 .. nx-1, averaging to exactly `v` over
                    //        the o sim cells of acquired voxel `v`. This is the frame the Roemer
                    //        combine walks (`x in 0..nx`), so anything the combine must agree with —
                    //        `coil_sensitivity`, whose coil ring is centred on nx/2 — uses it.
                    //   xc — CENTRED on the acquired image, -nx/2 .. nx/2-1. The eddy polynomial is
                    //        an expansion about the image centre and needs this one.
                    //
                    // Conflating them is not hypothetical: an earlier revision passed `xc` to
                    // `coil_sensitivity`, which put the forward model's sensitivity field half a FOV
                    // from the combine's at EVERY oversampling factor, o = 1 included (28% of peak).
                    // Pinned end-to-end by
                    // `multicoil_roemer_reproduces_the_single_coil_image_at_every_oversampling`.
                    let (xa, ya) = (
                        (x as f64 - xoff) / ox as f64,
                        (y as f64 - yoff) / oy as f64,
                    );
                    f_real *= acq.signal_scale * coil_sensitivity(coil, ncoils, xa, ya, nx, ny);
                    let mut phi = if acq.do_distortions { fmap[at(x, y)] as f64 * t } else { 0.0 };
                    // Pre-readout object phase, already in radians, so it is added outside the TAU
                    // factor that scales the distortion/eddy term.
                    let mut phi0 = inp.phase0.map_or(0.0, |p| p[at(x, y)]);
                    if do_eddy || do_eddy_phase {
                        // centre on the sim grid, then express in ACQUIRED voxel units so that the
                        // eddy scales keep their meaning independent of oversampling
                        let (xc, yc, zc) = (
                            (x as f64 - sxs as f64 - xoff) / ox as f64,
                            (y as f64 - sys as f64 - yoff) / oy as f64,
                            z as f64 - zs as f64,
                        );
                        if do_eddy {
                            // gradient-dependent field growing through the readout: linear (g·pos)
                            // plus a quadratic (g·pos²) term — the polynomial eddy/TORTOISE fit. This
                            // grows with the readout time (eddy_decay ∝ ky) → geometric DISTORTION.
                            let lin = gradient[0] * xc + gradient[1] * yc + gradient[2] * zc;
                            let quad =
                                gradient[0] * xc * xc + gradient[1] * yc * yc + gradient[2] * zc * zc;
                            phi += (acq.eddy_strength * lin + acq.eddy_quad * quad) * eddy_decay;
                        }
                        if do_eddy_phase {
                            // Eddy OBJECT-phase ramp: constant across the readout (NOT ∝ ky), so it
                            // imprints the reconstructed object phase rather than distorting geometry.
                            // Direction- and b-dependent (∝ gradient = bvec·bval), it reproduces the
                            // per-volume phase-ramp variation real DWI shows (∝ gradient direction).
                            // z centred on the volume (zc above is slice-index-from-start, not centred).
                            let zc_c = z as f64 - (inp.nz as f64 - 1.0) / 2.0;
                            phi0 += acq.eddy_phase
                                * (gradient[0] * xc + gradient[1] * yc + gradient[2] * zc_c);
                        }
                    }
                    modimg[at(x, y)] = C::cis(TAU * phi + phi0).scale(f_real);
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
        kspace
    }

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

    /// END-TO-END: a Roemer combine of N coils must reproduce the single-coil image, at every
    /// oversampling factor.
    ///
    /// This is the test that actually exercises the production coordinate transform. The
    /// helper-level test below checks the arithmetic of the transform the forward model OUGHT to
    /// use; only this one goes through `build_coil_kspace` and `simulate_slice`, so only this one
    /// can catch the forward model passing a coordinate in the wrong frame.
    ///
    /// It exists because that is exactly what happened: the coil call was given the eddy
    /// polynomial's CENTRED coordinate (`(x - sxs - xoff)/o`, spanning [-nx/2, nx/2)) while the
    /// combine passed absolute indices (0..nx) into a function whose coil ring is centred on
    /// nx/2. A half-FOV origin error, present even at o = 1, which every existing multi-coil test
    /// missed because they only assert broad properties like "centre brighter than corner".
    ///
    /// The combine is exact only for sensitivities band-limited within the acquired band; these
    /// Gaussians (sigma = 0.9 max(nx,ny)) are very smooth but not exactly so, hence a tolerance
    /// rather than equality. Measured: 0.0 at o = 1 (the two grids coincide, so the combine
    /// divides out exactly), 1.1e-3 at o = 2 and 1.0e-3 at o = 4 — against 0.28 unregistered.
    #[test]
    fn multicoil_roemer_reproduces_the_single_coil_image_at_every_oversampling() {
        let (nx, ny) = (32usize, 32usize);
        for o in [1usize, 2, 4] {
            let (snx, sny) = (nx * o, ny * o);
            // Edges off the voxel grid so the object is not accidentally band-limited.
            let q = nx as f64 / 4.0;
            let img = box_hires(
                snx, sny,
                (q + 0.5) * o as f64, (3.0 * q + 0.5) * o as f64,
                (q + 0.5) * o as f64, (3.0 * q + 0.5) * o as f64,
            );
            let fmap = vec![0.0f32; snx * sny];
            let comps: [&[f32]; 1] = [&img];
            let inp = step_input(&comps, &fmap, None, snx, sny, nx, ny);

            let one = simulate_slice(&inp, &Acquisition { n_coils: 1, ..clean() });
            let many = simulate_slice(&inp, &Acquisition { n_coils: 6, ..clean() });

            let peak = one.iter().map(|&(r, i)| (r * r + i * i).sqrt()).fold(0.0f32, f32::max);
            assert!(peak > 0.5, "o={o}: degenerate reference, peak {peak}");
            let worst = one
                .iter()
                .zip(&many)
                .map(|(&(ar, ai), &(br, bi))| ((ar - br).powi(2) + (ai - bi).powi(2)).sqrt())
                .fold(0.0f32, f32::max);
            assert!(
                worst / peak < 0.02,
                "o={o}: 6-coil Roemer combine differs from single coil by {:.4} of peak \
                 -- the forward model and the combine are not sampling one sensitivity field",
                worst / peak
            );
        }
    }

    /// The SUB-VOXEL half of the registration, at helper level.
    ///
    /// Scope, stated explicitly because getting this wrong is what let a half-FOV origin error
    /// ship: this test evaluates `coil_sensitivity` directly on the coordinate expression
    /// `(v*o + i - xoff)/o`. It therefore pins that expression's arithmetic — that the `o` sim
    /// cells of acquired voxel `v` are centred ON `v`, so their sensitivities average to the
    /// value the combine uses there — and NOT that the forward model passes that expression.
    /// A test that evaluates the transform it wishes the caller used cannot catch the caller
    /// using a different one. `multicoil_roemer_reproduces_the_single_coil_image_at_every_
    /// oversampling` above covers that, through the production call path.
    ///
    /// The sub-voxel error this one is about: the original code passed sim indices against
    /// `(snx, sny)`, centring the ring on `snx/2` = `nx/2 - (o-1)/(2o)` in acquired-voxel terms,
    /// a quarter voxel from the combine's `nx/2` at o = 2.
    #[test]
    fn coil_sensitivity_is_registered_identically_on_both_grids() {
        let (nx, ny, ncoils) = (32usize, 32usize, 4usize);
        for o in [1usize, 2, 3, 4, 8] {
            let (xoff, yoff) = ((o as f64 - 1.0) / 2.0, (o as f64 - 1.0) / 2.0);
            for coil in 0..ncoils {
                for v in [0usize, 1, 7, nx / 2, nx - 1] {
                    // Mean over the sim cells of acquired voxel (v, v), in acquired-voxel units.
                    let mut acc = 0.0;
                    for i in 0..o {
                        for j in 0..o {
                            let xc = (v * o + i) as f64 / o as f64 - xoff / o as f64;
                            let yc = (v * o + j) as f64 / o as f64 - yoff / o as f64;
                            acc += coil_sensitivity(coil, ncoils, xc, yc, nx, ny);
                        }
                    }
                    let sim_mean = acc / (o * o) as f64;
                    let combined = coil_sensitivity(coil, ncoils, v as f64, v as f64, nx, ny);
                    // A Gaussian is not linear, so the block mean is not EXACTLY the centre
                    // value. Its curvature over one acquired voxel puts the residual at most
                    // 7.9e-5 across this sweep; the tolerance sits an order of magnitude above
                    // that and two below the unregistered error asserted just below, so the test
                    // separates the two rather than merely accepting the current numbers.
                    assert!(
                        (sim_mean - combined).abs() < 1e-3,
                        "o={o} coil={coil} v={v}: sim-grid mean {sim_mean} vs combine {combined}"
                    );

                    // The OLD convention, reconstructed here so the test cannot pass vacuously:
                    // sim index against `(snx, sny)`. Every length in `coil_sensitivity` scales
                    // with the matrix, so that is exactly this field evaluated at `x / o` — i.e.
                    // displaced by `xoff / o = (o-1)/(2o)` acquired voxels. Worst case over this
                    // sweep is 1.15e-2, 146x the registered residual.
                    let mut old = 0.0;
                    for i in 0..o {
                        for j in 0..o {
                            old += coil_sensitivity(
                                coil, ncoils,
                                (v * o + i) as f64 / o as f64,
                                (v * o + j) as f64 / o as f64,
                                nx, ny,
                            );
                        }
                    }
                    let old_mean = old / (o * o) as f64;
                    if o > 1 && v != nx / 2 {
                        assert!(
                            (old_mean - combined).abs() > (sim_mean - combined).abs(),
                            "o={o} coil={coil} v={v}: unregistered sampling was no worse \
                             ({old_mean} vs registered {sim_mean}, combine {combined})"
                        );
                    }
                }
            }
        }
    }

    /// The half-cell registration itself, stated as arithmetic: the sim cells of acquired voxel
    /// `v` have centroid exactly `v` in acquired-voxel units, at every oversampling factor. This
    /// is the invariant every acquired-grid quantity evaluated on the sim grid depends on.
    #[test]
    fn sim_cells_of_an_acquired_voxel_are_centred_on_it() {
        for o in [1usize, 2, 3, 4, 8, 16] {
            let xoff = (o as f64 - 1.0) / 2.0;
            for v in 0..5usize {
                let mean: f64 = (0..o)
                    .map(|i| ((v * o + i) as f64 - xoff) / o as f64)
                    .sum::<f64>()
                    / o as f64;
                assert!((mean - v as f64).abs() < 1e-12, "o={o} v={v}: centroid {mean}");
            }
        }
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
    fn clean() -> Acquisition {
        Acquisition {
            signal_scale: 1.0,
            do_distortions: false,
            do_relaxation: false,
            noise_variance: 0.0,
            partial_fourier: 1.0,
            pf_mode: PartialFourierMode::Scanner,
            ghost_offset: 0.0,
            eddy_strength: 0.0,
            n_coils: 1,
            accel: 1,
            ..Acquisition::default()
        }
    }

    /// Build a step-edge SliceInput on the sim grid, optionally with an object phase field.
    fn step_input<'a>(
        comps: &'a [&'a [f32]],
        fmap: &'a [f32],
        phase0: Option<&'a [f64]>,
        snx: usize, sny: usize, nx: usize, ny: usize,
    ) -> SliceInput<'a> {
        SliceInput {
            compartments: comps, t2: &[100.0], fmap, phase0,
            sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1,
            bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 0
        }
    }

    #[test]
    fn noise_lands_only_on_acquired_samples() {
        // The original defect (finding 2.5): noise populated k-space that was never acquired,
        // so the signal was band-limited while the noise stayed white to the full Nyquist.
        let (nx, ny) = (24usize, 24usize);
        let acq = Acquisition { partial_fourier: 0.75, noise_variance: 1.0, ..clean() };
        let mask = sampling_mask(nx, ny, &acq);
        assert!(mask.iter().any(|b| !b), "pf=0.75 must leave some samples unacquired");
        let empty = vec![0.0f32; nx * ny];
        let fmap = vec![0.0f32; nx * ny];
        let comps: [&[f32]; 1] = [&empty];
        let k = simulate_slice_kspace(
            &SliceInput {
                compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: None,
                sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
                bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 9
            },
            &acq,
        );
        for i in 0..nx * ny {
            if !mask[i] {
                assert_eq!((k[i].0, k[i].1), (0.0, 0.0), "unacquired sample {i} carries noise");
            }
        }
    }

    #[test]
    fn noise_sd_scales_as_sqrt_sampled_fraction() {
        // Scoped: GRAPPA disabled, no window, identical per-sample thermal variance (spec 4.1.8).
        // `noise_variance` is the PER-COMPONENT variance of the reconstructed image at full
        // sampling, so the full-sampling SD is sqrt(noise_variance) = 1.0 here.
        let (nx, ny) = (32usize, 32usize);
        let sd_at = |pf: f64| -> f64 {
            let acq = Acquisition { partial_fourier: pf, noise_variance: 1.0, ..clean() };
            let empty = vec![0.0f32; nx * ny];
            let fmap = vec![0.0f32; nx * ny];
            let comps: [&[f32]; 1] = [&empty];
            let out = simulate_slice(
                &SliceInput {
                    compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: None,
                    sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
                    bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 3
                },
                &acq,
            );
            let v: f64 = out.iter().map(|p| (p.0 as f64).powi(2)).sum::<f64>() / (nx * ny) as f64;
            v.sqrt()
        };
        let full = sd_at(1.0);
        assert!((full - 1.0).abs() < 0.15, "full-sampling per-component SD {full:.3}, expected ~1.0");
        let frac = |pf: f64| {
            let acq = Acquisition { partial_fourier: pf, ..clean() };
            let m = sampling_mask(nx, ny, &acq);
            m.iter().filter(|b| **b).count() as f64 / (nx * ny) as f64
        };
        for pf in [0.75, 0.5] {
            let (sd, f) = (sd_at(pf), frac(pf));
            let ratio = sd / full;
            assert!(
                (ratio - f.sqrt()).abs() < 0.1,
                "pf={pf}: SD ratio {ratio:.3}, expected sqrt(sampled fraction {f:.3}) = {:.3}",
                f.sqrt()
            );
        }
    }

    #[test]
    fn noise_covariance_matches_the_sampling_mask() {
        // Spec 4.1.7a, scoped to GRAPPA disabled and window None: for white noise on mask M, the
        // reconstructed image noise autocovariance is the inverse DFT of M, up to scale.
        let (nx, ny) = (16usize, 16usize);
        let acq = Acquisition { partial_fourier: 0.75, noise_variance: 1.0, ..clean() };
        let mask = sampling_mask(nx, ny, &acq);
        let empty = vec![0.0f32; nx * ny];
        let fmap = vec![0.0f32; nx * ny];
        let (trials, lags) = (240usize, 4usize);
        let mut meas = vec![0.0f64; lags];
        for t in 0..trials {
            let comps: [&[f32]; 1] = [&empty];
            let out = simulate_slice(
                &SliceInput {
                    compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: None,
                    sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
                    bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 1000 + t as u64
                },
                &acq,
            );
            // Lag along the PHASE-ENCODE axis. Partial Fourier undersamples ky only, so a
            // readout-direction lag cannot see it: the skipped lines contribute uniformly to
            // every kx lag and cancel when normalizing by lag 0. Measuring along kx made this
            // test pass even with the mask guard removed.
            for (d, m) in meas.iter_mut().enumerate() {
                let mut acc = 0.0;
                for y in 0..ny - d {
                    for x in 0..nx {
                        acc += out[x + nx * y].0 as f64 * out[x + nx * (y + d)].0 as f64;
                    }
                }
                *m += acc / (nx * (ny - d)) as f64;
            }
        }
        for m in meas.iter_mut() {
            *m /= trials as f64;
        }
        // Prediction: inverse DFT of the mask along ky.
        let mut pred = vec![0.0f64; lags];
        for (d, p) in pred.iter_mut().enumerate() {
            let mut acc = 0.0;
            for kyi in 0..ny {
                for kxi in 0..nx {
                    if mask[kxi + nx * kyi] {
                        let ky = (kyi as f64 - (ny / 2) as f64) / ny as f64;
                        acc += (TAU * ky * d as f64).cos();
                    }
                }
            }
            *p = acc;
        }
        for d in 1..lags {
            let (a, b) = (meas[d] / meas[0], pred[d] / pred[0]);
            assert!((a - b).abs() < 0.08, "lag {d}: measured {a:.3}, mask prediction {b:.3}");
        }
    }

    #[test]
    fn partial_fourier_ringing_is_asymmetric_about_the_edge() {
        // Zero-filled PF (no homodyne/POCS) rings asymmetrically along PE, unlike the symmetric
        // full-Fourier case. 6/8 is the shipping default, so this is the primary configuration.
        let (nx, ny, o) = (32usize, 32usize, 8usize);
        let (snx, sny) = (nx * o, ny * o);
        let edge = (ny as f64 / 2.0 + 0.5) * o as f64;
        let mut img = vec![0.0f32; snx * sny];
        for y in 0..sny {
            let f = if (y as f64 + 1.0) <= edge {
                0.0
            } else if y as f64 >= edge {
                1.0
            } else {
                y as f64 + 1.0 - edge
            };
            for x in 0..snx {
                img[x + snx * y] = f as f32;
            }
        }
        let fmap = vec![0.0f32; snx * sny];
        let comps: [&[f32]; 1] = [&img];
        let run = |pf: f64| {
            simulate_slice(
                &SliceInput {
                    compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: None,
                    sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1,
                    bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 0
                },
                &Acquisition { partial_fourier: pf, ..clean() },
            )
        };
        let col = nx / 2;
        let asym = |v: &Vec<(f32, f32)>| {
            let below: f64 = (1..6).map(|d| (v[col + nx * (ny / 2 - d)].0 as f64).powi(2)).sum();
            let above: f64 = (1..6).map(|d| (v[col + nx * (ny / 2 + d)].0 as f64 - 1.0).powi(2)).sum();
            (above / below.max(1e-12)).ln().abs()
        };
        let full = run(1.0);
        for pf in [0.75, 0.875] {
            let p = run(pf);
            assert!(
                asym(&p) > asym(&full),
                "pf={pf} should ring more asymmetrically than full Fourier: {} vs {}",
                asym(&p), asym(&full)
            );
        }
        let e_full: f32 = full.iter().map(|v| v.0.abs()).sum();
        let e_pf: f32 = run(0.75).iter().map(|v| v.0.abs()).sum();
        assert!((e_pf / e_full - 1.0).abs() < 0.5, "PF energy {e_pf} vs full {e_full}");
    }

    #[test]
    fn even_matrix_window_asymmetry_is_intentional() {
        // The acquired band is [-n/2, n/2-1]: asymmetric about k=0 by one sample, exactly as real
        // even-matrix Cartesian acquisitions are. Retained deliberately (spec 3.4), so pin it.
        let n = 32i64;
        let (lo, hi) = (-(n / 2), n / 2 - 1);
        assert_eq!((lo, hi), (-16, 15));
        assert_eq!((hi - lo + 1) as usize, n as usize, "the band must hold exactly n samples");
        assert_eq!(lo.abs() - hi.abs(), 1, "one extra negative-frequency sample, by convention");

        // Consequence: a real object acquires a small imaginary component. It is deterministic and
        // is NOT realistic object phase -- that is what `phase.rs` supplies.
        let (nx, ny, o) = (16usize, 16usize, 4usize);
        let (snx, sny) = (nx * o, ny * o);
        let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
        let fmap = vec![0.0f32; snx * sny];
        let comps: [&[f32]; 1] = [&img];
        let out = simulate_slice(&step_input(&comps, &fmap, None, snx, sny, nx, ny), &clean());
        let mr = out.iter().map(|p| p.0.abs()).fold(0.0f32, f32::max);
        let mi = out.iter().map(|p| p.1.abs()).fold(0.0f32, f32::max);
        let residual = (mi / mr) as f64;
        assert!(residual < 0.05, "asymmetry residual should stay small: {residual}");
        assert!(residual > 1e-4, "a real object should still show the asymmetry: {residual}");
    }

    #[test]
    fn window_filters_signal_and_noise_together() {
        // The original defect (finding 2.5) was band-limited signal with unfiltered noise.
        // A reconstruction window must multiply both, so noise-only data must be suppressed too.
        let (nx, ny) = (32usize, 32usize);
        let empty = vec![0.0f32; nx * ny];
        let fmap = vec![0.0f32; nx * ny];
        let sd = |w: KspaceWindow| -> f64 {
            let comps: [&[f32]; 1] = [&empty];
            let out = simulate_slice(
                &SliceInput {
                    compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: None,
                    sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
                    bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 5
                },
                &Acquisition { noise_variance: 1.0, window: w, ..clean() },
            );
            (out.iter().map(|p| (p.0 as f64).powi(2)).sum::<f64>() / (nx * ny) as f64).sqrt()
        };
        let unwindowed = sd(KspaceWindow::None);
        let hann = sd(KspaceWindow::Hann);
        assert!(hann < 0.8 * unwindowed, "a window must attenuate noise too: {hann:.3} vs {unwindowed:.3}");
    }

    #[test]
    fn window_reduces_ringing_below_the_unapodized_case() {
        let (nx, ny, o) = (32usize, 32usize, 8usize);
        let (snx, sny) = (nx * o, ny * o);
        let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
        let fmap = vec![0.0f32; snx * sny];
        let comps: [&[f32]; 1] = [&img];
        let peak = |w: KspaceWindow| {
            let out = simulate_slice(
                &SliceInput {
                    compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: None,
                    sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1,
                    bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 0
                },
                &Acquisition { window: w, ..clean() },
            );
            (nx / 2 + 1..nx).map(|x| out[x + nx * (ny / 2)].0 as f64).fold(f64::MIN, f64::max)
        };
        let (plain, hann) = (peak(KspaceWindow::None), peak(KspaceWindow::Hann));
        assert!(hann < plain, "apodization must reduce overshoot: {hann:.4} vs {plain:.4}");
        assert!(plain > 1.05, "unapodized case should show real Gibbs overshoot: {plain:.4}");
    }

    #[test]
    fn global_phase_rotation_is_exact() {
        // Rotating the object by exp(i*alpha) must rotate the reconstructed image by exactly
        // exp(i*alpha) and leave its magnitude untouched. A gauge invariance no phase-histogram
        // statistic can substitute for (spec 4.1.5).
        let (nx, ny, o) = (16usize, 16usize, 4usize);
        let (snx, sny) = (nx * o, ny * o);
        let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
        let fmap = vec![0.0f32; snx * sny];
        let comps: [&[f32]; 1] = [&img];
        let acq = clean();
        let base = simulate_slice(&step_input(&comps, &fmap, None, snx, sny, nx, ny), &acq);
        let alpha = 0.7f64;
        let field = vec![alpha; snx * sny];
        let rot = simulate_slice(&step_input(&comps, &fmap, Some(&field), snx, sny, nx, ny), &acq);
        let (sa, ca) = alpha.sin_cos();
        for i in 0..nx * ny {
            let (re, im) = (base[i].0 as f64, base[i].1 as f64);
            let (er, ei) = (re * ca - im * sa, re * sa + im * ca);
            assert!((rot[i].0 as f64 - er).abs() < 1e-6, "re mismatch at {i}");
            assert!((rot[i].1 as f64 - ei).abs() < 1e-6, "im mismatch at {i}");
            let m0 = (re * re + im * im).sqrt();
            let m1 = ((rot[i].0 as f64).powi(2) + (rot[i].1 as f64).powi(2)).sqrt();
            assert!((m0 - m1).abs() < 1e-6, "magnitude changed at {i}");
        }
    }

    #[test]
    fn object_phase_puts_ringing_in_both_channels() {
        // The original defect: with a real object the image is real to machine precision, so all
        // ringing sits in Re and part-phase is degenerate. With object phase it must not.
        let (nx, ny, o) = (16usize, 16usize, 4usize);
        let (snx, sny) = (nx * o, ny * o);
        let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
        let fmap = vec![0.0f32; snx * sny];
        let comps: [&[f32]; 1] = [&img];
        let acq = clean();
        let ratio = |v: &Vec<(f32, f32)>| {
            let mr = v.iter().map(|p| p.0.abs()).fold(0.0f32, f32::max);
            let mi = v.iter().map(|p| p.1.abs()).fold(0.0f32, f32::max);
            (mi / mr) as f64
        };
        // Without object phase the image is real APART FROM the deliberate even-N band
        // asymmetry (spec 3.4), whose residual `even_matrix_window_asymmetry_is_intentional`
        // bounds at 0.05. Use that same bound here: a tighter one would contradict it.
        let none = simulate_slice(&step_input(&comps, &fmap, None, snx, sny, nx, ny), &acq);
        let r_none = ratio(&none);
        assert!(r_none < 0.05, "no-phase Im/Re should be only the even-N residual: {r_none}");
        let ramp: Vec<f64> = (0..snx * sny)
            .map(|i| 0.9 * ((i % snx) as f64 - snx as f64 / 2.0) / snx as f64)
            .collect();
        let withph = simulate_slice(&step_input(&comps, &fmap, Some(&ramp), snx, sny, nx, ny), &acq);
        let r_ph = ratio(&withph);
        assert!(r_ph > 0.1, "phase should move energy into Im: {r_ph}");
        // And the effect must dominate the asymmetry residual, not merely exceed a threshold.
        assert!(r_ph > 5.0 * r_none, "object phase {r_ph} vs even-N residual {r_none}");
    }

    #[test]
    fn crop_reproduces_the_analytic_profile_across_subvoxel_offsets() {
        let (nx, ny, o) = (32usize, 32usize, 8usize);
        let (snx, sny) = (nx * o, ny * o);
        let fmap = vec![0.0f32; snx * sny];
        let acq = clean();

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
            bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 0
        };
        let out = simulate_slice(&inp, &clean());
        let row = ny / 2;
        let peak = (nx / 2 + 1..nx).map(|x| out[x + nx * row].0 as f64).fold(f64::MIN, f64::max);
        assert!(peak > 1.05, "expected intrinsic overshoot, got peak {peak:.4}");
    }

    #[test]
    fn acquired_band_converges_with_simulation_resolution() {
        // Spec 4.1.6: adequacy is convergence of the ACQUIRED coefficients, not smoothness of the
        // object. A sharp edge is deliberately not band-limited; that is not a defect.
        let (nx, ny) = (16usize, 16usize);
        let acq = clean();
        let k_at = |o: usize| -> Vec<(f64, f64)> {
            let (snx, sny) = (nx * o, ny * o);
            let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.37) * o as f64);
            let fmap = vec![0.0f32; snx * sny];
            let comps: [&[f32]; 1] = [&img];
            simulate_slice_kspace(
                &SliceInput {
                    compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: None,
                    sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1,
                    bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 0
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

    #[test]
    fn production_path_produces_intrinsic_ringing_and_complex_phase() {
        // Regression guard for the defect this test exists because of: the shipped CLI once ran
        // at o=1, where the forward/inverse transforms are an exact round trip, so it produced
        // +0.0000% overshoot and no object phase.
        use crate::phase::PhaseModel;
        let (nx, ny, nz, o) = (32usize, 32usize, 1usize, 4usize);
        let (snx, sny) = (nx * o, ny * o);
        let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
        let fmap = vec![0.0f32; snx * sny * nz];
        let acq = Acquisition {
            signal_scale: 1.0, do_distortions: false, do_relaxation: false, ..Default::default()
        };
        let model = PhaseModel::hbcd_like();
        let images = [img];
        let (mag, ph) = simulate_acquisition(
            &SimulationInput {
                sim_dims: [snx, sny, nz], acq_dims: [nx, ny, nz], ngrad: 1, images: &images,
                t2: &[100.0], fmap: &fmap, bvals: &[1000.0], bvecs: &[[1.0, 0.0, 0.0]],
                phase: &model, seed: 7, noise_sigma: None,
            },
            &acq,
        );
        let row: Vec<f64> = (nx / 2 + 1..nx).map(|x| mag[(x + nx * (ny / 2)) * 1] as f64).collect();
        let over = row.iter().cloned().fold(f64::MIN, f64::max) - 1.0;
        assert!(over > 0.05, "production path must ring intrinsically, got {:+.4}%", over * 100.0);

        // and the object must actually be complex: phase varies across the bright region
        let bright: Vec<f64> = (nx / 2 + 2..nx - 2)
            .map(|x| ph[(x + nx * (ny / 2)) * 1] as f64)
            .collect();
        let spread = bright.iter().cloned().fold(f64::MIN, f64::max)
            - bright.iter().cloned().fold(f64::MAX, f64::min);
        assert!(spread > 0.1, "object phase should vary across the image, spread {spread}");
    }

    #[test]
    fn o1_is_an_exact_round_trip_with_no_ringing() {
        // Pins WHY `--oversample 1` cannot show Gibbs ringing: with the object on the
        // reconstruction matrix the forward and inverse transforms cancel exactly.
        let (nx, ny, nz) = (32usize, 32usize, 1usize);
        let mut img = vec![0.0f32; nx * ny * nz];
        for y in 0..ny {
            for x in nx / 2..nx {
                img[x + nx * y] = 1.0;
            }
        }
        let acq = Acquisition {
            signal_scale: 1.0, do_distortions: false, do_relaxation: false, ..Default::default()
        };
        let images = [img];
        let fmap = vec![0.0f32; nx * ny * nz];
        let phase = PhaseModel::none();
        let (mag, _) = simulate_acquisition(
            &SimulationInput {
                sim_dims: [nx, ny, nz], acq_dims: [nx, ny, nz], ngrad: 1, images: &images,
                t2: &[100.0], fmap: &fmap, bvals: &[0.0], bvecs: &[[0.0, 0.0, 0.0]],
                phase: &phase, seed: 0, noise_sigma: None,
            },
            &acq,
        );
        let over = (nx / 2 + 1..nx)
            .map(|x| mag[x + nx * (ny / 2)] as f64)
            .fold(f64::MIN, f64::max) - 1.0;
        assert!(over.abs() < 1e-4, "o = 1 is an exact round trip by construction: {over}");
    }

    #[test]
    fn partial_fourier_keeps_the_same_line_count_in_both_pe_polarities() {
        // AP and PA must acquire equally many PE lines, or a reverse-PE pair is not comparable --
        // which is the whole point of simulating one (topup / DRBUDDI).
        for (ny, pf) in [(32usize, 0.75f64), (32, 0.875), (64, 0.75), (64, 0.875), (30, 0.75)] {
            let mk = |rev: bool| {
                let a = Acquisition { partial_fourier: pf, reverse_phase: rev, ..Default::default() };
                let m = sampling_mask(16, ny, &a);
                (0..ny).filter(|&ky| m[16 * ky]).count()
            };
            let (f, r) = (mk(false), mk(true));
            let want = (ny as f64 * pf).round() as usize;
            assert_eq!(f, r, "ny={ny} pf={pf}: forward keeps {f} lines, reverse {r}");
            assert!(
                (f as i64 - want as i64).abs() <= 1,
                "ny={ny} pf={pf}: kept {f} lines, expected about {want}"
            );
        }
    }

    #[test]
    fn spikes_land_only_on_acquired_samples() {
        let (nx, ny) = (24usize, 24usize);
        let acq = Acquisition {
            signal_scale: 1.0, do_distortions: false, do_relaxation: false,
            partial_fourier: 0.75, n_spikes: 40, spike_amplitude: 1.0, ..Default::default()
        };
        let mask = sampling_mask(nx, ny, &acq);
        let img = vec![1.0f32; nx * ny];
        let comps: [&[f32]; 1] = [&img];
        let k = simulate_slice_kspace(
            &SliceInput {
                compartments: &comps, t2: &[100.0], fmap: &vec![0.0f32; nx * ny], phase0: None,
                sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
                bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 5
            },
            &acq,
        );
        for i in 0..nx * ny {
            if !mask[i] {
                assert_eq!((k[i].0, k[i].1), (0.0, 0.0), "spike landed on unacquired sample {i}");
            }
        }
    }

    #[test]
    fn contiguous_partial_fourier_keeps_exactly_the_nominal_fraction() {
        // The default Fiberfox rule does NOT: on an even matrix it preserves line zero because
        // that line's Nyquist conjugate is absent, so nominal 6/8 is 25/32 = 78.1%, with one
        // isolated line detached from the acquired block. Contiguous mode is what a scanner
        // produces and what a PF-aware unringing benchmark should default to comparing against.
        for &(ny, pf) in &[(32usize, 0.75f64), (32, 0.875), (64, 0.75), (140, 0.75)] {
            let count = |mode, rev| {
                let a = Acquisition {
                    partial_fourier: pf, pf_mode: mode, reverse_phase: rev, ..Default::default()
                };
                let m = sampling_mask(8, ny, &a);
                (0..ny).filter(|&k| m[8 * k]).count()
            };
            let want = (ny as f64 * pf).round() as usize;
            for rev in [false, true] {
                assert_eq!(
                    count(PartialFourierMode::Contiguous, rev), want,
                    "ny={ny} pf={pf} reverse={rev}: contiguous must keep exactly {want} lines"
                );
            }
            // and the two modes must actually differ on an even matrix
            if ny % 2 == 0 {
                assert!(
                    count(PartialFourierMode::FiberfoxCompatible, false) > want,
                    "ny={ny} pf={pf}: the Fiberfox rule should keep MORE than {want}"
                );
            }
        }
    }

    #[test]
    fn contiguous_partial_fourier_lines_are_actually_contiguous() {
        let a = Acquisition {
            partial_fourier: 0.75, pf_mode: PartialFourierMode::Contiguous, ..Default::default()
        };
        let ny = 32;
        let m = sampling_mask(8, ny, &a);
        let kept: Vec<usize> = (0..ny).filter(|&k| m[8 * k]).collect();
        assert!(!kept.is_empty());
        assert_eq!(
            kept.last().unwrap() - kept[0] + 1, kept.len(),
            "contiguous mode must leave no gaps, got {kept:?}"
        );
    }
    /// The production forward is the literal per-line sum reorganised (static factors hoisted,
    /// the affine phase advanced by memoised rotors, the eddy polynomial factored per axis, the
    /// x-DFT done by FFT / twiddle table). Every effect, alone and combined, must reproduce the
    /// literal sum to round-off — including the rotor re-anchoring (ny > REANCHOR), the odd-line
    /// dwell offset, GRAPPA strides, partial Fourier in both modes and polarities, odd matrices
    /// and o = 1 as well as o = 2.
    #[test]
    fn restructured_forward_matches_the_literal_sum() {
        let mut rng = Rng(0xC0FFEE);
        let unit = |r: &mut Rng| r.unit();
        for &(nx, ny, o) in &[(17usize, 70usize, 2usize), (16, 40, 1), (8, 9, 3)] {
            let (snx, sny) = (nx * o, ny * o);
            let n = snx * sny;
            let mut comps: Vec<Vec<f32>> = Vec::new();
            for c in 0..3 {
                comps.push((0..n).map(|i| {
                    let (x, y) = ((i % snx) as f64 / snx as f64, (i / snx) as f64 / sny as f64);
                    let blob = (-(((x - 0.5).powi(2) + (y - 0.45).powi(2)) / (0.03 * (c + 1) as f64))).exp();
                    (blob + 0.05 * unit(&mut rng)) as f32
                }).collect());
            }
            let comp_refs: Vec<&[f32]> = comps.iter().map(|v| v.as_slice()).collect();
            let t2 = [70.0f32, 100.0, 2000.0];
            // fieldmap with a large range (±300 Hz over a 40–70 ms readout: ±20 cycles at the edge)
            let fmap: Vec<f32> = (0..n).map(|i| {
                let (x, y) = ((i % snx) as f64 / snx as f64, (i / snx) as f64 / sny as f64);
                (300.0 * ((3.0 * x).sin() * (2.0 * y + 1.0).cos()) + 20.0 * unit(&mut rng)) as f32
            }).collect();
            let phase0: Vec<f64> = (0..n).map(|_| TAU * unit(&mut rng)).collect();
            let full = Acquisition {
                do_distortions: true, do_relaxation: true, signal_scale: 100.0,
                ..Acquisition::default()
            };
            let cases: Vec<(&str, Acquisition, [f64; 3], f64, usize, usize)> = vec![
                ("clean", Acquisition { do_distortions: false, do_relaxation: false, ..full.clone() }, [0.0; 3], 0.0, 0, 1),
                ("distortion+relaxation", full.clone(), [0.0; 3], 0.0, 0, 1),
                ("reverse", Acquisition { reverse_phase: true, ..full.clone() }, [0.0; 3], 0.0, 0, 1),
                ("ghost", Acquisition { ghost_offset: 0.015, ..full.clone() }, [0.0; 3], 0.0, 0, 1),
                ("eddy-poly", Acquisition { eddy_strength: 3.0, eddy_quad: 0.4, eddy_tau: 70.0, ..full.clone() },
                    [0.3, -0.8, 0.5], 1.0, 0, 1),
                ("eddy-phase", Acquisition { eddy_phase: 0.2, ..full.clone() }, [0.6, 0.6, 0.5], 1.0, 0, 1),
                ("pf-fiberfox", Acquisition { partial_fourier: 0.75, ..full.clone() }, [0.0; 3], 0.0, 0, 1),
                ("pf-contiguous-reverse", Acquisition { partial_fourier: 0.75, pf_mode: PartialFourierMode::Contiguous,
                    reverse_phase: true, ..full.clone() }, [0.0; 3], 0.0, 0, 1),
                ("grappa-coils", Acquisition { accel: 2, acs_lines: 6, n_coils: 4, ..full.clone() }, [0.0; 3], 0.0, 2, 4),
                ("grappa3", Acquisition { accel: 3, acs_lines: 8, n_coils: 4, ..full.clone() }, [0.0; 3], 0.0, 1, 4),
                ("everything", Acquisition { ghost_offset: 0.02, eddy_strength: 2.0, eddy_quad: 0.3, eddy_phase: 0.1,
                    partial_fourier: 0.8, accel: 2, acs_lines: 8, n_coils: 3, ..full.clone() },
                    [0.5, 0.5, 0.7], 1.0, 0, 3),
            ];
            for (name, acq, bvec, bval, coil, ncoils) in cases {
                let inp = SliceInput {
                    compartments: &comp_refs, t2: &t2, fmap: &fmap, phase0: Some(&phase0),
                    sim: [snx, sny], acq_matrix: [nx, ny], z: 3, nz: 9, bvec, bval, slice_seed: 0,
                };
                let a = build_coil_kspace(&inp, &acq, coil, ncoils);
                let b = reference_coil_kspace(&inp, &acq, coil, ncoils);
                let peak = b.iter().map(|k| k.abs()).fold(0.0, f64::max);
                let worst = a.iter().zip(&b).map(|(p, q)| (p.re - q.re).hypot(p.im - q.im)).fold(0.0, f64::max);
                assert!(peak > 0.0, "{name}: empty reference");
                assert!(worst <= 1e-10 * peak, "{name} @ {nx}x{ny} o={o}: max |Δ| {worst:.3e} vs peak {peak:.3e}");
            }
        }
    }

    #[cfg(feature = "kspace")]
    #[test]
    fn fft_inverse_matches_the_direct_dft() {
        // random k-space of a typical acquired size; the FFT recon must equal the direct DFT.
        let (nx, ny) = (84, 100);
        let mut rng = Rng(0xC0FFEE);
        let ks: Vec<C> = (0..nx * ny).map(|_| C { re: rng.gauss(), im: rng.gauss() }).collect();
        let (xs, ys) = (nx / 2, ny / 2);
        let direct = inverse_2d(&ks, nx, ny, xs, ys);
        let fft = inverse_2d_fft(&ks, nx, ny, xs, ys);
        let mut max = 0.0f64;
        for (a, b) in direct.iter().zip(&fft) {
            max = max.max((a.re - b.re).abs()).max((a.im - b.im).abs());
        }
        // amplitudes are O(nx*ny) after an unnormalized inverse, so a ~1e-8 relative tolerance
        assert!(max < 1e-6 * (nx * ny) as f64, "FFT vs direct DFT max abs diff {max}");
    }
    #[cfg(feature = "kspace")]
    #[test]
    #[ignore = "timing micro-benchmark; run explicitly with --ignored"]
    fn bench_inverse_direct_vs_fft() {
        // Load-independent micro-benchmark of the reconstruction transform at the acquired size.
        let (nx, ny) = (85usize, 128usize);
        let mut rng = Rng(1);
        let ks: Vec<C> = (0..nx * ny).map(|_| C { re: rng.gauss(), im: rng.gauss() }).collect();
        let (xs, ys) = (nx / 2, ny / 2);
        let n = 3000;
        let mut sink = 0.0f64;
        let t0 = std::time::Instant::now();
        for _ in 0..n { sink += inverse_2d(&ks, nx, ny, xs, ys)[0].re; }
        let td = t0.elapsed();
        let t1 = std::time::Instant::now();
        for _ in 0..n { sink += inverse_2d_fft(&ks, nx, ny, xs, ys)[0].re; }
        let tf = t1.elapsed();
        println!(
            "inverse {nx}x{ny}: direct {:.2} us/call, fft {:.2} us/call, speedup {:.1}x (sink {sink:.1})",
            td.as_micros() as f64 / n as f64, tf.as_micros() as f64 / n as f64,
            td.as_secs_f64() / tf.as_secs_f64());
    }

    /// The oversampled k-space stage must not move the object: a blob centred on acquired voxel
    /// (X0, Y0) -- i.e. at sim index o*X0 + (o-1)/2 -- must come back centred on (X0, Y0). Even
    /// and odd acquired matrices: with an odd matrix the sim grid's `snx/2` is not the image of
    /// the acquired centre `nx/2`, which used to shift the object by a fraction of a voxel.
    #[test]
    fn oversampled_reconstruction_keeps_the_object_in_place() {
        fn centroid(o: usize, nx: usize, ny: usize, x0: f64, y0: f64) -> (f64, f64) {
            let (snx, sny) = (nx * o, ny * o);
            let off = (o as f64 - 1.0) / 2.0;
            let (sx0, sy0) = (o as f64 * x0 + off, o as f64 * y0 + off);
            let mut img = vec![0.0f32; snx * sny];
            for y in 0..sny {
                for x in 0..snx {
                    let d2 = (x as f64 - sx0).powi(2) + (y as f64 - sy0).powi(2);
                    img[x + snx * y] = (-d2 / (2.0 * (1.5 * o as f64).powi(2))).exp() as f32;
                }
            }
            let comps: [&[f32]; 1] = [&img];
            let fmap = vec![0.0f32; snx * sny];
            let acq = Acquisition { do_distortions: false, do_relaxation: false, ..Acquisition::default() };
            let out = simulate_slice(
                &SliceInput { compartments: &comps, t2: &[80.0], fmap: &fmap, phase0: None, sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1, bvec: [1.0, 0.0, 0.0], bval: 0.0, slice_seed: 0 },
                &acq,
            );
            let (mut sx, mut sy, mut sw) = (0.0, 0.0, 0.0);
            for y in 0..ny {
                for x in 0..nx {
                    let (re, im) = out[x + nx * y];
                    let m = ((re * re + im * im) as f64).sqrt();
                    sx += m * x as f64; sy += m * y as f64; sw += m;
                }
            }
            (sx / sw, sy / sw)
        }
        for (nx, ny) in [(24usize, 20usize), (25, 21), (107, 151)] {
            for o in [1usize, 2, 4] {
                if nx * o > 200 { continue; }
                let (cx, cy) = centroid(o, nx, ny, 9.0, 7.0);
                assert!((cx - 9.0).abs() < 0.05 && (cy - 7.0).abs() < 0.05, "{nx}x{ny} o={o}: centroid ({cx:.3}, {cy:.3})");
            }
        }
    }

    // ---- complex / k-space entry point (Python-binding surface) ----

    /// A 3-slice, 2-compartment object at o=2 with every in-plane artifact on, distinct per slice.
    fn artifact_fixture() -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Acquisition, [usize; 3], [usize; 3]) {
        let (nx, ny, nz, o) = (16usize, 16usize, 3usize, 2usize);
        let (snx, sny) = (nx * o, ny * o);
        let ngrad = 2;
        let mut fiber = vec![0.0f32; snx * sny * nz * ngrad];
        let mut gm = vec![0.0f32; snx * sny * nz * ngrad];
        let mut fmap = vec![0.0f32; snx * sny * nz];
        for z in 0..nz {
            let sh = z as f64 * 3.0;
            let a = box_hires(snx, sny, 6.0 + sh, 20.0 + sh, 8.0, 24.0 - sh);
            let b = box_hires(snx, sny, 12.0, 26.0, 4.0 + sh, 18.0);
            for y in 0..sny {
                for x in 0..snx {
                    let i = x + snx * y;
                    let vox = x + snx * (y + sny * z);
                    for g in 0..ngrad {
                        let att = if g == 0 { 1.0 } else { 0.6 };
                        fiber[vox * ngrad + g] = a[i] * att;
                        gm[vox * ngrad + g] = 0.5 * b[i] * att;
                    }
                    fmap[vox] = 3.0 * ((x as f64 - 16.0) / 16.0) as f32 + z as f32;
                }
            }
        }
        let acq = Acquisition {
            signal_scale: 1.0,
            noise_variance: 1e-4,
            partial_fourier: 0.75,
            pf_mode: PartialFourierMode::Contiguous,
            ghost_offset: 0.01,
            eddy_strength: 0.01,
            eddy_quad: 0.002,
            eddy_phase: 2e-5,
            n_spikes: 1,
            spike_amplitude: 0.2,
            n_coils: 4,
            accel: 2,
            acs_lines: 8,
            seed: 11,
            ..Default::default()
        };
        (vec![fiber, gm], vec![80.0, 60.0], fmap, acq, [snx, sny, nz], [nx, ny, nz])
    }

    fn fixture_input<'a>(
        images: &'a [Vec<f32>], t2: &'a [f32], fmap: &'a [f32], phase: &'a PhaseModel, noise_sigma: Option<&'a [f32]>,
        sim: [usize; 3], acq: [usize; 3],
    ) -> SimulationInput<'a> {
        SimulationInput {
            sim_dims: sim, acq_dims: acq, ngrad: 2, images, t2, fmap,
            bvals: &[0.0, 1000.0], bvecs: &[[0.0, 0.0, 0.0], [0.6, 0.8, 0.0]],
            phase, seed: 5, noise_sigma,
        }
    }

    #[test]
    fn simulate_acquisition_is_bit_identical_to_the_per_slice_loop() {
        // The literal loop the acquisition used to be (simulate_slice per (g, z), then f32
        // mag/phase) must match the wrapper over simulate_acquisition_complex sample for sample,
        // with the image-space noise map on so that path is covered too.
        let (images, t2, fmap, acq, sim, acqd) = artifact_fixture();
        let [snx, sny, nz] = sim;
        let [nx, ny, _] = acqd;
        let o = snx / nx;
        let phase = PhaseModel::hbcd_like();
        let ns: Vec<f32> = (0..nx * ny * nz).map(|v| 0.01 * ((v % 7) as f32)).collect();
        let inp = fixture_input(&images, &t2, &fmap, &phase, Some(&ns), sim, acqd);
        let (mag, ph) = simulate_acquisition(&inp, &acq);

        let ngrad = 2;
        for g in 0..ngrad {
            for z in 0..nz {
                let mut cs = vec![vec![0.0f32; snx * sny]; images.len()];
                let mut fs = vec![0.0f32; snx * sny];
                for y in 0..sny {
                    for x in 0..snx {
                        let vox = x + snx * (y + sny * z);
                        for (c, img) in images.iter().enumerate() {
                            cs[c][x + snx * y] = img[vox * ngrad + g];
                        }
                        fs[x + snx * y] = fmap[vox];
                    }
                }
                let refs: Vec<&[f32]> = cs.iter().map(|v| v.as_slice()).collect();
                let shot = phase.diffusion.shot(inp.bvals[g], inp.bvecs[g], g, z, inp.seed);
                let phi = phase_slice(&phase, &shot, snx, sny, o, z, nz);
                let slice_seed = (g as u64).wrapping_mul(0x100_0001).wrapping_add(z as u64).wrapping_mul(0x9E37) ^ inp.seed;
                let out = simulate_slice(
                    &SliceInput {
                        compartments: &refs, t2: &t2, fmap: &fs, phase0: Some(&phi), sim: [snx, sny],
                        acq_matrix: [nx, ny], z, nz, bvec: inp.bvecs[g], bval: inp.bvals[g], slice_seed,
                    },
                    &acq,
                );
                for y in 0..ny {
                    for x in 0..nx {
                        let (mut re, mut im) = out[x + nx * y];
                        let vox = x + nx * (y + ny * z);
                        let sd = ns[vox] as f64;
                        if sd > 0.0 {
                            let mut rng = Rng(
                                inp.seed ^ (g as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                    ^ (vox as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9) | 1,
                            );
                            re += (rng.gauss() * sd) as f32;
                            im += (rng.gauss() * sd) as f32;
                        }
                        let i = vox * ngrad + g;
                        assert_eq!(mag[i].to_bits(), (re * re + im * im).sqrt().to_bits(), "mag differs at g={g} z={z} x={x} y={y}");
                        assert_eq!(ph[i].to_bits(), im.atan2(re).to_bits(), "phase differs at g={g} z={z} x={x} y={y}");
                    }
                }
            }
        }
    }

    #[test]
    fn capturing_kspace_changes_no_arithmetic() {
        let (images, t2, fmap, acq, sim, acqd) = artifact_fixture();
        let phase = PhaseModel::hbcd_like();
        let inp = fixture_input(&images, &t2, &fmap, &phase, None, sim, acqd);
        let plain = simulate_acquisition_complex(&inp, &acq, &AcquisitionOptions::default());
        let slices = [1usize, 0];
        let cap = simulate_acquisition_complex(
            &inp, &acq,
            &AcquisitionOptions { kspace_slices: Some(&slices), capture: SliceCapture::ALL, ..Default::default() },
        );
        assert_eq!(plain.re, cap.re);
        assert_eq!(plain.im, cap.im);
        let k = cap.kspace.expect("k-space requested");
        assert_eq!(k.slices, vec![1, 0]);
        let (nx, ny, ncoils, ngrad, nsl) = (k.nx, k.ny, k.n_coils, 2, 2);
        assert_eq!(k.combined.len(), ngrad * nsl * nx * ny);
        assert_eq!(k.acquired.as_ref().unwrap().len(), ngrad * nsl * ncoils * nx * ny);
        // `combined` for a captured slice is the pre-noise complex image of that slice, in order.
        for g in 0..ngrad {
            for (si, &zl) in k.slices.iter().enumerate() {
                for y in 0..ny {
                    for x in 0..nx {
                        let c = k.combined[((g * nsl + si) * ny + y) * nx + x];
                        let vox = x + nx * (y + ny * zl);
                        assert_eq!(c[0], plain.re[vox * ngrad + g]);
                        assert_eq!(c[1], plain.im[vox * ngrad + g]);
                    }
                }
            }
        }
    }

    #[test]
    fn captured_reconstructed_kspace_inverts_to_combined() {
        let (images, t2, fmap, acq, sim, acqd) = artifact_fixture();
        let phase = PhaseModel::hbcd_like();
        let inp = fixture_input(&images, &t2, &fmap, &phase, None, sim, acqd);
        let slices = [2usize];
        let out = simulate_acquisition_complex(
            &inp, &acq,
            &AcquisitionOptions { kspace_slices: Some(&slices), capture: SliceCapture::ALL, ..Default::default() },
        );
        let k = out.kspace.unwrap();
        let (nx, ny, nc) = (k.nx, k.ny, k.n_coils);
        let (xs, ys) = (nx / 2, ny / 2);
        let rec = k.reconstructed.as_ref().unwrap();
        let cimg = k.coil_images.as_ref().unwrap();
        for g in 0..2 {
            let mut wsum = vec![C::ZERO; nx * ny];
            let mut ssum = vec![0.0f64; nx * ny];
            for coil in 0..nc {
                let base = ((g * 1 + 0) * nc + coil) * nx * ny;
                let ks: Vec<C> = rec[base..base + nx * ny].iter().map(|v| C { re: v[0] as f64, im: v[1] as f64 }).collect();
                let img = inverse_2d(&ks, nx, ny, xs, ys);
                for i in 0..nx * ny {
                    let ci = cimg[base + i];
                    assert!((img[i].re - ci[0] as f64).abs() < 1e-4 && (img[i].im - ci[1] as f64).abs() < 1e-4, "coil image mismatch");
                    let s = k.sensitivities[coil * nx * ny + i] as f64;
                    wsum[i] = wsum[i].add(img[i].scale(s));
                    ssum[i] += s * s;
                }
            }
            for i in 0..nx * ny {
                let c = k.combined[g * nx * ny + i];
                let (re, im) = (wsum[i].re / ssum[i].max(1e-12), wsum[i].im / ssum[i].max(1e-12));
                assert!((re - c[0] as f64).abs() < 1e-4 && (im - c[1] as f64).abs() < 1e-4, "combined mismatch at g={g} i={i}: ({re},{im}) vs {c:?}");
            }
        }
    }

    #[test]
    fn acquired_kspace_is_pre_grappa_and_mask_is_the_sampling_mask() {
        let (images, t2, fmap, acq, sim, acqd) = artifact_fixture();
        let phase = PhaseModel::none();
        let inp = fixture_input(&images, &t2, &fmap, &phase, None, sim, acqd);
        let slices = [0usize];
        let out = simulate_acquisition_complex(
            &inp, &acq,
            &AcquisitionOptions { kspace_slices: Some(&slices), capture: SliceCapture::ALL, ..Default::default() },
        );
        let k = out.kspace.unwrap();
        assert_eq!(k.mask, sampling_mask(k.nx, k.ny, &acq));
        let (nx, ny, nc) = (k.nx, k.ny, k.n_coils);
        let acqd_k = k.acquired.as_ref().unwrap();
        let rec = k.reconstructed.as_ref().unwrap();
        let mut filled = 0;
        for coil in 0..nc {
            for ky in 0..ny {
                for kx in 0..nx {
                    let i = coil * nx * ny + kx + nx * ky;
                    if !k.mask[kx + nx * ky] {
                        assert_eq!(acqd_k[i], [0.0, 0.0], "un-acquired sample must be zero in `acquired`");
                        if rec[i] != [0.0, 0.0] {
                            filled += 1;
                        }
                    }
                }
            }
        }
        assert!(filled > 0, "GRAPPA must have synthesised un-acquired lines in `reconstructed`");
    }

    #[test]
    fn per_volume_te_equals_separate_runs() {
        let (images, t2, fmap, acq, sim, acqd) = artifact_fixture();
        let phase = PhaseModel::hbcd_like();
        let inp = fixture_input(&images, &t2, &fmap, &phase, None, sim, acqd);
        let tes = [70.0, 110.0];
        let both = simulate_acquisition_complex(
            &inp, &acq, &AcquisitionOptions { t_echo_per_volume: Some(&tes), ..Default::default() },
        );
        for (g, te) in tes.iter().enumerate() {
            let one = simulate_acquisition_complex(&inp, &Acquisition { t_echo: *te, ..acq.clone() }, &AcquisitionOptions::default());
            let nvox = acqd[0] * acqd[1] * acqd[2];
            for vox in 0..nvox {
                assert_eq!(both.re[vox * 2 + g].to_bits(), one.re[vox * 2 + g].to_bits());
                assert_eq!(both.im[vox * 2 + g].to_bits(), one.im[vox * 2 + g].to_bits());
            }
        }
        let same = simulate_acquisition_complex(&inp, &acq, &AcquisitionOptions::default());
        assert_ne!(same.re, both.re, "a different TE must change the relaxation weighting");
    }

    #[test]
    fn a_single_slice_run_is_bit_identical_to_its_slice_of_the_full_run() {
        // The unit of simulation: with slice_z / nz_full the eddy z-terms, object phase, per-slice
        // seeds and the noise-map RNG all see the slice's full-FOV position.
        let (images, t2, fmap, acq, sim, acqd) = artifact_fixture();
        let [snx, sny, nz] = sim;
        let [nx, ny, _] = acqd;
        let phase = PhaseModel::hbcd_like();
        let ns: Vec<f32> = (0..nx * ny * nz).map(|v| 0.02 * ((v % 5) as f32)).collect();
        let inp = fixture_input(&images, &t2, &fmap, &phase, Some(&ns), sim, acqd);
        let full = simulate_acquisition_complex(&inp, &acq, &AcquisitionOptions::default());

        let zl = 1usize;
        let ngrad = 2;
        let sub_images: Vec<Vec<f32>> = images
            .iter()
            .map(|img| img[snx * sny * zl * ngrad..snx * sny * (zl + 1) * ngrad].to_vec())
            .collect();
        let sub_fmap = fmap[snx * sny * zl..snx * sny * (zl + 1)].to_vec();
        let sub_ns = ns[nx * ny * zl..nx * ny * (zl + 1)].to_vec();
        let sub = fixture_input(&sub_images, &t2, &sub_fmap, &phase, Some(&sub_ns), [snx, sny, 1], [nx, ny, 1]);
        let z_of = [zl];
        let one = simulate_acquisition_complex(
            &sub, &acq, &AcquisitionOptions { slice_z: Some(&z_of), nz_full: Some(nz), ..Default::default() },
        );
        for y in 0..ny {
            for x in 0..nx {
                for g in 0..ngrad {
                    let vf = (x + nx * (y + ny * zl)) * ngrad + g;
                    let vs = (x + nx * y) * ngrad + g;
                    assert_eq!(full.re[vf].to_bits(), one.re[vs].to_bits(), "re differs at x={x} y={y} g={g}");
                    assert_eq!(full.im[vf].to_bits(), one.im[vs].to_bits(), "im differs at x={x} y={y} g={g}");
                }
            }
        }
        // and without the z bookkeeping it is NOT the same slice (the z-terms are live in this fixture)
        let naive = simulate_acquisition_complex(&sub, &acq, &AcquisitionOptions::default());
        assert_ne!(naive.re, one.re);
    }

    #[test]
    fn epi_timing_agrees_with_line_times_and_the_trajectory() {
        for reverse in [false, true] {
            let acq = Acquisition { t_line: 0.8, t_echo: 75.0, reverse_phase: reverse, ..Default::default() };
            let (nx, ny) = (12usize, 10usize);
            let t = epi_timing(nx, ny, &acq);
            let epi = SingleShotEpi { kx_max: nx, ky_max: ny, t_line: 0.8, t_echo: 75.0, reverse_phase: reverse };
            let (a, b, c) = line_times_for(&epi, &acq);
            assert_eq!((&t.t_ms, &t.t_rf_ms, &t.t_read_ms), (&a, &b, &c));
            let mut seen = t.order.clone();
            seen.sort_unstable();
            assert_eq!(seen, (0..ny).collect::<Vec<_>>(), "order must visit every line once");
            for w in t.order.windows(2) {
                assert!(t.t_read_ms[w[1]] > t.t_read_ms[w[0]], "readout time must increase along the acquisition order");
            }
            for ky in 0..ny {
                assert!((t.t_rf_ms[ky] - (t.t_echo + t.t_ms[ky])).abs() < 1e-12);
            }
            let traj = epi_trajectory(nx, ny, &acq);
            assert_eq!(traj.len(), nx * ny);
            assert!(traj.windows(2).all(|w| w[1].2 > w[0].2));
            assert_eq!(traj[0].1, t.order[0]);
        }
    }

    #[test]
    fn scanner_partial_fourier_skips_the_first_lines_and_reaches_the_centre_sooner() {
        let (nx, ny, pf) = (16usize, 32usize, 0.75f64);
        for reverse in [false, true] {
            let acq = Acquisition { partial_fourier: pf, reverse_phase: reverse, t_line: 2.0, ..Default::default() };
            assert_eq!(acq.pf_mode, PartialFourierMode::Scanner, "scanner rule is the default");
            let full = Acquisition { partial_fourier: 1.0, reverse_phase: reverse, t_line: 2.0, ..Default::default() };
            let m = sampling_mask(nx, ny, &acq);
            let kept: Vec<usize> = (0..ny).filter(|&ky| m[nx * ky]).collect();
            assert_eq!(kept.len(), (ny as f64 * pf).round() as usize);
            assert!(kept.windows(2).all(|w| w[1] == w[0] + 1), "kept lines are contiguous");
            assert!(kept.contains(&(ny / 2)), "the centre line is kept");
            // the skipped lines are the FIRST ones in acquisition order
            let order = epi_timing(nx, ny, &acq).order;
            let skip = pf_skipped_lines(ny, &acq);
            assert_eq!(skip, ny / 4);
            assert!(order[..skip].iter().all(|ky| !kept.contains(ky)), "first {skip} lines skipped: {:?}", &order[..skip]);
            assert!(order[skip..].iter().all(|ky| kept.contains(ky)));
            // and the readout clock starts at the first acquired line: the centre is reached
            // `skip` line times sooner than at full Fourier, the echo-relative times unchanged
            let (tp, tf) = (epi_timing(nx, ny, &acq), epi_timing(nx, ny, &full));
            assert_eq!(tp.t_ms, tf.t_ms);
            let c = ny / 2;
            assert!((tp.t_read_ms[c] - (tf.t_read_ms[c] - skip as f64 * 2.0)).abs() < 1e-9);
            let first = order[skip];
            assert!(tp.t_read_ms[first] > 0.0 && tp.t_read_ms[first] < 2.0, "first acquired line reads at the start of its own train");
            // legacy contiguous mode: same count, dropped from the END, timing unchanged
            let legacy = Acquisition { pf_mode: PartialFourierMode::Contiguous, ..acq.clone() };
            let ml = sampling_mask(nx, ny, &legacy);
            let kept_l: Vec<usize> = (0..ny).filter(|&ky| ml[nx * ky]).collect();
            assert_eq!(kept_l.len(), kept.len());
            assert!(order[ny - skip..].iter().all(|ky| !kept_l.contains(ky)), "legacy drops the last lines");
            assert_eq!(epi_timing(nx, ny, &legacy).t_read_ms, tf.t_read_ms);
        }
    }
}
