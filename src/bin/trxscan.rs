//! End-to-end simulator: streamlines + tissue + scheme + fieldmap → BIDS complex 4D DWI.
//! Signal stage (per-voxel mixture → clean signal) → acquisition stage (EPI distortion, T2*, eddy,
//! partial Fourier, Gibbs ringing, spikes, multi-coil, GRAPPA, noise), with optional head motion,
//! gradient nonlinearity and a matching synthetic GRE fieldmap.

use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use std::time::Instant;

use trxscan::compartments::{
    generate_compartments_moving, generate_mixture, signal_from_mixture_gnl, CompartmentParams,
};
use trxscan::gnl::{GnlField, GradCoef};
use trxscan::gre::{self, GreObject, GreOutput, GreParams};
use trxscan::io;
use trxscan::kspace::{simulate_acquisition, Acquisition, PartialFourierMode, SimulationInput};
use trxscan::motion;
use trxscan::orient::Reorient;
use trxscan::phase::PhaseModel;
use trxscan::raster::Grid;
use trxscan::scheme::GradientScheme;
use trxscan::sphere::HemiSphere;

/// Compartment parameter preset (diffusivities + T2 relaxation times).
#[derive(Copy, Clone, Debug, ValueEnum)]
enum Preset {
    /// Fiberfox ffp legacy values (weak GM/WM contrast at low b by design)
    Neonatal,
    /// Adult 3T values: literature diffusivities, T2s from an EPI relaxometry fit
    Adult,
    /// Unmyelinated-WM diffusivities (pair with a myelin map)
    Infant,
}

impl Preset {
    fn params(self) -> CompartmentParams {
        match self {
            Preset::Neonatal => CompartmentParams::default(),
            Preset::Adult => CompartmentParams::adult(),
            Preset::Infant => CompartmentParams::infant(),
        }
    }
}

impl std::fmt::Display for Preset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_possible_value().expect("no skipped variants").get_name().fmt(f)
    }
}

/// Simulate a realistic complex 4D diffusion-MRI acquisition from a tractogram and tissue maps.
#[derive(Parser)]
#[command(name = "trxscan", version)]
struct Cli {
    /// White-matter volume-fraction map (NIfTI)
    #[arg(long, value_name = "NII")]
    wm: PathBuf,
    /// Grey-matter volume-fraction map (NIfTI)
    #[arg(long, value_name = "NII")]
    gm: PathBuf,
    /// CSF volume-fraction map (NIfTI)
    #[arg(long, value_name = "NII")]
    csf: PathBuf,
    /// Brain mask (NIfTI)
    #[arg(long, value_name = "NII")]
    mask: PathBuf,
    /// Streamlines (TRX / TRK / TCK / VTK)
    #[arg(long, value_name = "TRACT")]
    streamlines: PathBuf,
    /// FSL b-values
    #[arg(long, value_name = "BVAL")]
    bval: PathBuf,
    /// FSL b-vectors
    #[arg(long, value_name = "BVEC")]
    bvec: PathBuf,
    /// Off-resonance fieldmap in Hz on the ACQUISITION grid. Used only with `--oversample 1`;
    /// the default oversampled path takes `--sim-fmap` (on the simulation grid) instead.
    #[arg(long, value_name = "NII")]
    fmap: Option<PathBuf>,
    /// Output BIDS stem, e.g. out/sub-01_dir-AP_run-01
    #[arg(short, long, value_name = "PREFIX")]
    out: String,

    /// In-plane oversampling factor for the simulation grid.
    ///
    /// This is what makes Gibbs ringing INTRINSIC to the acquisition: the object is simulated
    /// finer than the acquisition matrix and only the nominal k-space band is acquired. With
    /// `--oversample 1` the object sits on the reconstruction matrix, the transforms are an exact
    /// round trip, and the output has NO ringing.
    ///
    /// Requires `--sim-*` inputs at `voxel/N`, from
    /// `scripts/prepare_acquisition_grid.py --oversample N`. Upsampling the acquisition-grid maps
    /// instead would add no k-space content and produce no ringing.
    /// **Default 2, not 4.** o=4 is the accuracy target (residual ~4.5% of the artifact against
    /// ~15.6% at o=2) but is not viable on ordinary hardware. In-plane voxels scale as o^2, and on
    /// the default no-motion path the dominant allocation is a dense nvox x 321 f64 orientation
    /// histogram plus an f32 ODF copy: for an HBCD-sized grid that bound is ~26 GB at o=2 and
    /// ~104 GB at o=4. A measured o=2 run peaked at 11.5 GB -- well under its bound, because the
    /// histogram commits lazily, but that is data-dependent. Treat 16 GB as tight, not safe.
    #[arg(long, value_name = "N", default_value_t = 2)]
    oversample: usize,
    /// Simulation-grid white-matter map (from prepare_acquisition_grid.py --oversample)
    #[arg(long, value_name = "NII")]
    sim_wm: Option<PathBuf>,
    /// Simulation-grid grey-matter map
    #[arg(long, value_name = "NII")]
    sim_gm: Option<PathBuf>,
    /// Simulation-grid CSF map
    #[arg(long, value_name = "NII")]
    sim_csf: Option<PathBuf>,
    /// Simulation-grid brain mask
    #[arg(long, value_name = "NII")]
    sim_mask: Option<PathBuf>,
    /// Simulation-grid fieldmap (Hz)
    #[arg(long, value_name = "NII")]
    sim_fmap: Option<PathBuf>,
    /// Object phase model: "hbcd" (calibrated), or "none" for a real-valued object
    #[arg(long, value_name = "MODEL", default_value = "hbcd")]
    phase_model: String,
    /// Partial-Fourier line-dropping rule: "contiguous" (scanner-like, exactly round(ny*pf)
    /// consecutive lines) or "fiberfox" (the ported rule, which preserves line zero on even
    /// matrices and so keeps ~78% at a nominal 6/8).
    #[arg(long, value_name = "MODE", default_value = "contiguous")]
    pf_mode: String,

    /// Flip phase-encode polarity (the AP/PA pair for topup / DRBUDDI)
    #[arg(long)]
    reverse_pe: bool,
    /// Write the DWI in FSL / dcm2niix orientation (radiological LAS) instead of the input grid's
    /// orientation. eddy/topup/fugue and qsiprep then consume it directly, with the
    /// PhaseEncodingDirection matching the baked-in distortion (no reorientation surprises).
    #[arg(long)]
    fsl_orientation: bool,
    /// Noise level: the PER-COMPONENT variance of the reconstructed complex image at full
    /// sampling, single coil, pre-combination -- so Var(Re) = Var(Im) = this, and E[|n|^2] is
    /// twice it. The per-k-space-sample variance is derived from the reconstruction
    /// normalization; multi-coil combination lowers the final variance by sum_c s_c^2.
    #[arg(long, default_value_t = 0.0, value_name = "VAR")]
    noise: f64,
    /// Per-voxel noise level (SD) map on the ACQUISITION grid: spatially-varying complex Gaussian
    /// noise added to the reconstructed image, so magnitude and phase share one realization. Writes
    /// the SD map as <out>_desc-noise_sigma.nii.gz (ground truth for a denoiser's noise estimate).
    #[arg(long, value_name = "NII")]
    noise_map: Option<PathBuf>,
    /// Linear eddy-current strength (DWI volumes only; b0 exempt)
    #[arg(long, default_value_t = 0.0, value_name = "S")]
    eddy: f64,
    /// Quadratic eddy-current strength
    #[arg(long, default_value_t = 0.0, value_name = "S")]
    eddy_quad: f64,
    /// Eddy-current OBJECT-phase ramp strength (rad per unit bvec·bval per acquired voxel). Unlike
    /// --eddy (which distorts geometry), this imprints a direction- and b-dependent ramp on the
    /// reconstructed phase, reproducing the per-volume phase variation real DWI shows (~1.7e-5
    /// matches a 3T HBCD-protocol scan). DWI volumes only.
    #[arg(long, default_value_t = 0.0, value_name = "S")]
    eddy_phase: f64,
    /// GRAPPA acceleration factor R
    #[arg(long, default_value_t = 1, value_name = "R")]
    accel: usize,
    /// Receiver coils (ring-arranged sensitivities)
    #[arg(long, default_value_t = 1, value_name = "N")]
    coils: usize,
    /// Multiband factor
    #[arg(long, default_value_t = 1, value_name = "MB")]
    mb: usize,

    /// Head-motion trace: a qsiprep/eddy confounds TSV (trans_x/y/z mm, rot_x/y/z rad). Switches
    /// the signal stage to the faithful per-volume re-simulation path (no SIFT2 weights, Watson
    /// kappa, myelin, GNL or truth peaks there).
    #[arg(long, value_name = "TSV")]
    motion: Option<PathBuf>,
    /// Per-DWI-volume probability of a within-volume dropout event (needs --mb > 1)
    #[arg(long, default_value_t = 0.0, value_name = "P")]
    dropout_rate: f64,

    /// SIFT2 weights: a TRX data-per-streamline array name, or an MRtrix tcksift2 text file
    #[arg(long, value_name = "SPEC")]
    weights: Option<String>,
    /// Watson dispersion concentration κ for the orientation histogram
    #[arg(long, value_name = "KAPPA")]
    kappa: Option<f64>,
    /// Compartment parameter preset
    #[arg(long, value_enum, default_value_t = Preset::Neonatal)]
    params: Preset,
    /// Per-voxel myelination map (0..1): lerps the WM compartment toward the adult endpoint
    #[arg(long, value_name = "NII")]
    myelin: Option<PathBuf>,
    /// Per-compartment signal amplitude "wm,gm,csf" (proton density x T1 saturation): scales the
    /// fiber/GM/CSF b0 levels to match a real acquisition. Default: no change. CSF < 1 mimics
    /// TR/T1 saturation (long-T1 CSF is not fully relaxed at a finite TR).
    #[arg(long, value_name = "WM,GM,CSF")]
    tissue_s0: Option<String>,
    /// Scale each compartment's diffusivities "wm,gm,csf" to tune the S(b)/S0 decay to a real
    /// acquisition (WM intra+extra + soma, GM ball, CSF ball). Default: no change.
    #[arg(long, value_name = "WM,GM,CSF")]
    diff_scale: Option<String>,

    /// Gradient nonlinearity: a preset ("whole-body-80", "connectom-300") or a Siemens `.grad`
    /// coefficient file. Adds the spatial encoding warp AND the per-voxel diffusion-encoding
    /// deviation; writes the coefficient file, the displacement fields and the graddev image
    /// next to the DWI. See docs/GNL.md.
    #[arg(long, value_name = "PRESET|FILE")]
    gnl: Option<String>,
    /// Multiply the nonlinear coefficients (l >= 3) by this factor (severity knob).
    #[arg(long, default_value_t = 1.0, value_name = "S", requires = "gnl")]
    gnl_scale: f64,
    /// Scanner isocentre, world RAS mm as "x,y,z". Default: the world origin. Every written
    /// header is translated so this point becomes the origin, which is where TORTOISE's
    /// coefficient evaluation puts the isocentre.
    #[arg(long, value_name = "X,Y,Z")]
    isocenter: Option<String>,
    /// GNL: skip the spatial warp (diffusion-encoding deviation only).
    #[arg(long, requires = "gnl")]
    gnl_no_warp: bool,
    /// GNL: skip the diffusion-encoding deviation (spatial warp only).
    #[arg(long, requires = "gnl")]
    gnl_no_encoding: bool,
    /// GNL: do not modulate warped intensities by 1/|det J|.
    #[arg(long, requires = "gnl")]
    gnl_no_jacobian_modulation: bool,
    /// GNL: print the field envelope at 2/5/8/10/12 cm from the isocentre and exit.
    #[arg(long, requires = "gnl")]
    gnl_info: bool,

    /// Also synthesize a dual-echo GRE fieldmap (magnitude1/magnitude2 + phasediff or
    /// phase1/phase2, Siemens conventions, with sidecars) from the same off-resonance field, at
    /// this BIDS stem. Sees the same GNL warp and --tissue-s0 as the DWI.
    #[arg(long, value_name = "PREFIX")]
    gre_out: Option<String>,
    /// Tissue SNR of the synthetic GRE echoes (complex Gaussian noise on both echoes; 0 =
    /// noiseless, with the phasediff's corner voxels stamped to 0/4095 so its range stays full
    /// for min/max-based Siemens phase conversion).
    #[arg(long, default_value_t = 50.0, requires = "gre_out")]
    gre_snr: f64,
    /// GRE fieldmap output resolution (isotropic mm). Default: the DWI acquisition grid. With a
    /// value, the GRE is generated at that resolution by complex-averaging the fine-grid signal
    /// onto the coarser grid (intravoxel dephasing) instead of sampling the field pointwise.
    #[arg(long, value_name = "MM", requires = "gre_out")]
    gre_res: Option<f64>,
    /// B0 field identifier label for the GRE fieldmap (BIDS `B0FieldIdentifier`, which replaces
    /// the deprecated `IntendedFor`). The DWI written in the same run carries the matching
    /// `B0FieldSource`, so qsiprep links them automatically.
    #[arg(long, default_value = "b0gre", value_name = "LABEL", requires = "gre_out")]
    gre_b0field: String,
    /// How the GRE SNR scales with voxel volume relative to the DWI acquisition resolution:
    /// SNR ∝ V^exp, i.e. σ ∝ (V_dwi / V_gre)^exp. 1.0 (default) = fixed scan time and FOV;
    /// 0.5 = fixed number of averages; 0 = constant σ at all resolutions.
    #[arg(long, default_value_t = 1.0, value_name = "EXP", requires = "gre_out")]
    gre_snr_vol_exp: f64,
    /// GRE phase representation: `phasediff` (default, smooth single image) or `phase` (the two
    /// individual echo phases `phase1`/`phase2`, which show the receiver-phase fringe wrapping of a
    /// raw GRE and go through qsiprep's two-phase route).
    #[arg(long, value_enum, default_value_t = GreOutput::Phasediff, requires = "gre_out")]
    gre_output: GreOutput,
    /// Peak amplitude (rad) of the smooth receiver/transmit phase φ₀ added to the individual echo
    /// phases (`--gre-output phase`). Larger → more fringes. Cancels in the phase difference.
    #[arg(long, default_value_t = 6.0, value_name = "RAD", requires = "gre_out")]
    gre_rx_phase: f64,

    /// Also write the ground-truth fibre orientations per acquisition voxel: up to three peaks
    /// of the orientation mixture (aggregated over the oversampled cells, refined to sub-bin
    /// accuracy), each a unit vector in world RAS scaled by its mass fraction, as a 9-volume
    /// NIfTI `<out>_desc-truth_peaks.nii.gz`. Unaffected by the GNL warp: the mixture is the
    /// object in its true frame.
    #[arg(long)]
    truth_peaks: bool,

    /// Global random seed: selects the noise/dropout realization and drives --subsample.
    /// The default (0) reproduces the historical output for identical inputs.
    #[arg(long, default_value_t = 0, value_name = "SEED")]
    seed: u64,
    /// Keep only N streamlines, sampled with probability proportional to the SIFT2 weight
    /// (uniform without weights). The same N and --seed select the identical subset in
    /// trxscan-microstructure, so simulated data and ground truth describe the same phantom.
    #[arg(long, value_name = "N")]
    subsample: Option<usize>,
}

/// Deterministically pick DWI shots to corrupt: each DWI volume gets a dropout event with
/// probability `rate`, at a pseudo-random shot, with severity 0.6–1.0 and a ~1–3 mm bulk jump.
fn gen_dropout_events(bvals: &[f64], n_shots: usize, rate: f64, seed: u64) -> Vec<motion::MotionEvent> {
    let mix = |z0: u64| {
        let mut z = z0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut evs = Vec::new();
    for (g, &b) in bvals.iter().enumerate() {
        if b < 50.0 {
            continue; // b0s don't dropout
        }
        let h = mix(seed ^ (g as u64).wrapping_mul(0x0100_0001));
        if (h % 100_000) as f64 / 100_000.0 >= rate {
            continue;
        }
        let shot = (h >> 20) as usize % n_shots.max(1);
        let severity = 0.6 + ((h >> 33) % 40) as f32 / 100.0;
        let j = 1.0 + ((h >> 45) % 20) as f64 / 10.0;
        evs.push(motion::MotionEvent {
            volume: g,
            shot,
            severity,
            jump_mm: [0.3 * j, j, 0.2 * j],
            jump_deg: [0.5, 0.3, 0.2],
        });
    }
    evs
}

/// Parse a "a,b,c" triple.
fn parse3(spec: &str, name: &str) -> Result<[f64; 3], String> {
    let v: Vec<f64> = spec.split(',').map(|t| t.trim().parse::<f64>())
        .collect::<Result<_, _>>().map_err(|_| format!("{name} expects three comma-separated numbers, got {spec:?}"))?;
    if v.len() != 3 {
        return Err(format!("{name} expects three comma-separated values, got {spec:?}"));
    }
    Ok([v[0], v[1], v[2]])
}

/// World position (RAS mm) of a continuous voxel coordinate on `grid`.
fn voxel_to_world(grid: &Grid, p: [f64; 3]) -> [f64; 3] {
    let w = &grid.voxel_to_world;
    [
        w[0][0] * p[0] + w[0][1] * p[1] + w[0][2] * p[2] + w[0][3],
        w[1][0] * p[0] + w[1][1] * p[1] + w[1][2] * p[2] + w[1][3],
        w[2][0] * p[0] + w[2][1] * p[1] + w[2][2] * p[2] + w[2][3],
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let kappa = cli.kappa.filter(|k| *k > 0.0);

    let (tissue, mut grid) = io::load_tissue(&cli.wm, &cli.gm, &cli.csf, &cli.mask)?;

    // The signal stage runs on ONE grid, chosen here. With --oversample > 1 that is the finer
    // simulation grid; the acquisition grid still defines the output matrix. Building the nominal
    // `comp` and then discarding it would silently drop motion, SIFT2 weights, kappa, myelin and
    // dropout -- and waste a full Stage A -- which is exactly what an earlier version did.
    let need_sim = |o: &Option<PathBuf>, n: &str| -> Result<PathBuf, String> {
        o.clone().ok_or_else(|| format!(
            "--oversample {} requires --{n}; generate the simulation grid with \
             scripts/prepare_acquisition_grid.py --oversample {}. Upsampling the \
             acquisition-grid maps would add no k-space content and produce no ringing.",
            cli.oversample, cli.oversample))
    };
    let (sig_tissue, sig_grid, sig_fmap_path) = if cli.oversample > 1 {
        let (st, sg) = io::load_tissue(
            &need_sim(&cli.sim_wm, "sim-wm")?, &need_sim(&cli.sim_gm, "sim-gm")?,
            &need_sim(&cli.sim_csf, "sim-csf")?, &need_sim(&cli.sim_mask, "sim-mask")?)?;
        let o = cli.oversample;
        if sg.dims != [grid.dims[0] * o, grid.dims[1] * o, grid.dims[2]] {
            return Err(format!(
                "simulation grid {:?} is not {o}x the acquisition grid {:?} in-plane \
                 (z is never oversampled)", sg.dims, grid.dims).into());
        }
        (st, sg, need_sim(&cli.sim_fmap, "sim-fmap")?)
    } else {
        eprintln!(
            "WARNING: --oversample 1 puts the object on the reconstruction matrix, so the \
             transforms are an exact round trip and the output has NO Gibbs ringing. Use \
             --oversample 2 (default) or 4 for a realistic acquisition.");
        (tissue, grid.clone(), cli.fmap.clone().ok_or(
            "--oversample 1 requires --fmap, the acquisition-grid fieldmap. \
             The default oversampled path takes --sim-fmap instead and ignores --fmap.")?)
    };
    let (mut positions, mut offsets, mut weights) =
        io::load_streamlines_spec(&cli.streamlines, cli.weights.as_deref())?;
    if let Some(n) = cli.subsample {
        (positions, offsets, weights) =
            io::subsample_streamlines(positions, offsets, weights, n, cli.seed);
    }
    let mut scheme = GradientScheme::from_fsl(&cli.bval, &cli.bvec)?;
    // ONE fieldmap load, on whichever grid the signal stage runs on -- `--sim-fmap` when
    // oversampling, `--fmap` at o = 1.
    let (mut fmap, fgrid) = io::load_volume(&sig_fmap_path)?;
    if fgrid.dims != sig_grid.dims {
        return Err(format!("fieldmap grid {:?} != signal grid {:?}",
                           fgrid.dims, sig_grid.dims).into());
    }

    // Scanner isocentre: translate the world frame so it sits at the origin. Streamlines live in
    // world mm and are rasterised through the grid affines, so they move with the grids.
    let mut sig_grid = sig_grid;
    if let Some(spec) = &cli.isocenter {
        let v = parse3(spec, "--isocenter")?;
        for r in 0..3 {
            grid.voxel_to_world[r][3] -= v[r];
            sig_grid.voxel_to_world[r][3] -= v[r];
        }
        for p in positions.iter_mut() {
            for r in 0..3 {
                p[r] -= v[r];
            }
        }
        println!("isocenter: world point {:?} mm moved to the origin", v);
    }

    // Gradient nonlinearity: the coefficient set and its field on the signal grid.
    let gnl: Option<(GradCoef, GnlField)> = if let Some(spec) = &cli.gnl {
        let mut coef = GradCoef::from_spec(spec).map_err(|e| format!("--gnl: {e}"))?;
        coef.scale_nonlinear(cli.gnl_scale);
        let t = Instant::now();
        let field_sig = GnlField::on_grid(&coef, &sig_grid);
        println!("GNL: {} ({} terms, scale {}), field built in {:?}",
            spec, coef.terms.len(), cli.gnl_scale, t.elapsed());
        for radius in [20.0, 50.0, 80.0, 100.0, 120.0] {
            let e = field_sig.envelope(&sig_grid, radius);
            println!("  within {radius:>5.0} mm: {:>7} voxels  max |d| {:.2} mm  max gradient dev {:.2} %  max angle {:.2} deg",
                e.n_voxels, e.max_disp_mm, 100.0 * e.max_gradient_dev, e.max_angle_deg);
        }
        if cli.gnl_info {
            return Ok(());
        }
        Some((coef, field_sig))
    } else {
        None
    };
    let gnl_encoding = gnl.as_ref().filter(|_| !cli.gnl_no_encoding).map(|(_, f)| f);
    let gnl_warp = gnl.as_ref().filter(|_| !cli.gnl_no_warp).map(|(_, f)| (f, !cli.gnl_no_jacobian_modulation));
    println!("acquisition grid {:?}  signal grid {:?}  {} streamlines  {} volumes  shells {:?}",
        grid.dims, sig_grid.dims, offsets.len().saturating_sub(1), scheme.len(),
        scheme.shells(50.0));

    // Pre-flight memory estimate, BEFORE the signal stage allocates anything. An earlier version
    // computed this immediately before Stage B -- i.e. after generate_mixture had already
    // allocated -- so an allocation failure happened before the warning could ever print.
    //
    // The dominant allocation differs by path, and each is modelled from what the code actually
    // does rather than from a guess:
    //
    //   no motion: generate_mixture's dense nvox * 321 f64 orientation histogram + its f32 ODF
    //              copy, plus the f32 compartment images.
    //   motion:    generate_compartments_moving has NO nvox*ngrad f64 accumulator. It works per
    //              volume (four nvox f32 resampled tissue arrays, two nvox f64, three nvox f32
    //              outputs), COLLECTS the three f32 outputs for every volume into `vols`
    //              (3*nvox*ngrad*4 bytes), and then allocates three more nvox*ngrad f32 arrays and
    //              copies into them. Both are live during that copy, so the floor is
    //              24 * nvox * ngrad bytes -- not the 20 an earlier version estimated -- plus a
    //              per-worker allowance, which `par` multiplies.
    {
        let nvox = sig_grid.dims.iter().product::<usize>() as f64;
        let ngrad = scheme.len() as f64;
        let images_gb = nvox * ngrad * 3.0 * 4.0 / 1e9;
        // Rayon's actual pool size when it is running the show, since RAYON_NUM_THREADS may
        // differ from the machine's parallelism; available_parallelism is only the fallback.
        let workers = if cfg!(feature = "par") {
            std::env::var("RAYON_NUM_THREADS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
                .unwrap_or(1) as f64
        } else {
            1.0
        };
        let (dominant_gb, what) = if cli.motion.is_some() {
            // collected per-volume outputs + final arrays, both live during the copy, plus each
            // in-flight worker's own nvox f32/f64 working set.
            let per_worker = nvox * (4.0 * 4.0 + 2.0 * 8.0 + 3.0 * 4.0) / 1e9;
            (nvox * ngrad * 24.0 / 1e9 + workers * per_worker,
             "collected per-volume outputs + final images, plus per-worker arrays")
        } else {
            (nvox * 321.0 * 12.0 / 1e9, "dense orientation histogram + ODF")
        };
        // The motion path's figure already includes the images; the mixture path's does not.
        let bound = if cli.motion.is_some() { dominant_gb } else { dominant_gb + images_gb };
        if bound > 8.0 {
            // The caveat differs by path and must not be copied across: only the mixture path's
            // dense histogram commits lazily.
            let caveat = if cli.motion.is_some() {
                "These arrays are densely written, so committed memory tracks the bound closely -- \
                 unlike the no-motion histogram, this does not benefit from sparse page commitment."
            } else {
                "Committed memory is DATA-DEPENDENT and usually lower: the histogram faults in \
                 only where streamlines deposit, and an o=2 run of this size measured 11.5 GB \
                 against a 25.9 GB bound. Treat 16 GB as tight, not safe."
            };
            eprintln!(
                "WARNING: signal stage upper bound about {bound:.1} GB on grid {:?} \
                 ({dominant_gb:.1} GB {what}; compartment images {images_gb:.1} GB; {} volumes).\n\
                 \x20        {caveat} In-plane memory scales as o^2.",
                sig_grid.dims, scheme.len());
        }
    }

    let mut base = cli.params.params();
    if let Some(spec) = &cli.diff_scale {
        let s = parse3(spec, "--diff-scale")?;
        base.d_intra *= s[0];
        base.d_extra = (base.d_extra.0 * s[0], base.d_extra.1 * s[0], base.d_extra.2 * s[0]);
        base.d_gm *= s[1];
        base.d_soma *= s[1];
        base.d_csf *= s[2];
        println!("diffusivity scale: wm x{} gm x{} csf x{}", s[0], s[1], s[2]);
    }
    let tissue_s0: [f32; 3] = match &cli.tissue_s0 {
        Some(spec) => {
            let s = parse3(spec, "--tissue-s0")?;
            println!("tissue s0: fiber {} gm {} csf {}", s[0], s[1], s[2]);
            [s[0] as f32, s[1] as f32, s[2] as f32]
        }
        None => [1.0; 3],
    };
    println!("compartment params: {}  T2 fiber/gm/csf {}/{}/{} ms",
        cli.params, base.t2_fiber, base.t2_gm, base.t2_csf);
    let params = CompartmentParams { b_value: scheme.b_max, ..base };
    let t = Instant::now();
    // With a motion trace, use the faithful path: per volume, transform the streamlines + tissue
    // by that volume's pose and re-simulate (Fiberfox-style). Otherwise the histogram-first
    // mixture path — SIFT2 weights + Watson κ dispersion, and the same
    // mixture `trxscan-microstructure` computes ground truth from.
    let mut truth_peaks: Option<Vec<f32>> = None;
    let mut comp = if let Some(tsv) = &cli.motion {
        // The per-segment path has no orientation mixture, so everything that lives on the
        // mixture is unavailable. Refuse rather than silently ignore.
        if weights.is_some() || kappa.is_some() {
            eprintln!("note: weights/kappa are ignored in motion mode (per-segment re-simulation)");
        }
        if cli.myelin.is_some() {
            return Err("--myelin is not supported together with --motion (per-segment path)".into());
        }
        if cli.truth_peaks {
            return Err("--truth-peaks is not available together with --motion (no orientation mixture)".into());
        }
        if gnl.is_some() {
            return Err("--gnl is not supported together with --motion yet".into());
        }
        let poses = motion::load_motion_tsv(tsv)?;
        let moved = poses.iter().filter(|p| **p != motion::Pose::IDENTITY).count();
        println!("Motion (faithful re-simulation): {} poses, {} moved, from {}",
            poses.len(), moved, tsv.display());
        generate_compartments_moving(
            &sig_grid, &positions, &offsets, &sig_tissue, &scheme, &params, &poses)
    } else {
        let mut mix = generate_mixture(
            &sig_grid, &positions, &offsets, weights.as_deref(), &sig_tissue, &params, kappa,
            HemiSphere::icosphere(3),
        );
        if let Some(mp) = &cli.myelin {
            let (my, mgrid) = io::load_volume(mp)?;
            if mgrid.dims != sig_grid.dims {
                return Err(format!(
                    "myelin grid {:?} != signal grid {:?}. With --oversample > 1 the myelin map \
                     must be on the SIMULATION grid.", mgrid.dims, sig_grid.dims).into());
            }
            println!("myelin map: {} (per-voxel lerp to adult endpoint)", mp.display());
            mix.myelin = Some(my);
        }
        if cli.truth_peaks {
            truth_peaks = Some(trxscan::truth::truth_peaks(&mix, cli.oversample.max(1), 3));
        }
        let fb = mix.fallback.iter().filter(|&&f| f == 1).count();
        println!(
            "Mixture: {} fallback WM voxels{}{}",
            fb,
            if weights.is_some() { ", SIFT2-weighted" } else { "" },
            kappa.map(|k| format!(", Watson kappa {k}")).unwrap_or_default()
        );
        signal_from_mixture_gnl(&mix, &scheme, gnl_encoding)
    };
    println!("Signal stage (per-compartment signal): {:?}  T2 {:?}", t.elapsed(), comp.t2);
    comp.apply_s0(tissue_s0);

    // Multiband within-volume motion + slice dropout: inject synthetic bulk-motion events on random
    // DWI shots and write the dropped-slice ground truth (for scoring eddy --repol / SHORELine).
    if cli.mb > 1 && cli.dropout_rate > 0.0 {
        let n_shots = (sig_grid.dims[2] / cli.mb).max(1);
        let events = gen_dropout_events(&scheme.bvals, n_shots, cli.dropout_rate,
            0xB10C_5EED ^ cli.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let gt = motion::apply_multiband_motion(
            &mut comp.images, sig_grid.dims, comp.ngrad, sig_grid.voxel_to_world,
            cli.mb, true, &scheme.bvals, scheme.b_max, &events);
        let mut tsv = String::from("volume\tshot\tslices\tattenuation\n");
        for d in &gt {
            let sl = d.slices.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",");
            tsv.push_str(&format!("{}\t{}\t{}\t{:.3}\n", d.volume, d.shot, sl, d.attenuation));
        }
        std::fs::write(format!("{}_desc-dropout_slices.tsv", cli.out), tsv)?;
        println!("Multiband dropout: mb={}, {} shots dropped -> {}_desc-dropout_slices.tsv",
            cli.mb, gt.len(), cli.out);
    }

    // Gradient nonlinearity, spatial part: the scanner encodes tissue at φ(r), so every
    // compartment image and the off-resonance field move to the apparent frame before k-space
    // encoding (kspace looks the field up at the voxel it is encoding).
    if let Some((field_sig, modulate)) = gnl_warp {
        let t = Instant::now();
        for img in comp.images.iter_mut() {
            *img = field_sig.warp_4d(img, comp.ngrad, modulate);
        }
        fmap = field_sig.warp_volume(&fmap, false);
        println!("GNL spatial warp of {} compartment images + fieldmap: {:?}", comp.images.len(), t.elapsed());
    }

    // The HBCD-like protocol; the flags override its artifact knobs.
    let acq = Acquisition {
        signal_scale: 100.0,
        reverse_phase: cli.reverse_pe,
        noise_variance: cli.noise,
        pf_mode: match cli.pf_mode.as_str() {
            "contiguous" => PartialFourierMode::Contiguous,
            "fiberfox" => PartialFourierMode::FiberfoxCompatible,
            o => return Err(format!("unknown --pf-mode {o:?}; expected contiguous or fiberfox").into()),
        },
        eddy_strength: cli.eddy,
        eddy_quad: cli.eddy_quad,
        eddy_phase: cli.eddy_phase,
        n_coils: cli.coils,
        accel: cli.accel,
        seed: cli.seed,
        ..Acquisition::hbcd(grid.dims[1])
    };

    // Dual-echo GRE fieldmap from the same object: magnitudes from the tissue mixture with the
    // compartment T2s, phase from the (already warped) off-resonance field.
    let gre_fieldmap = if let Some(prefix) = &cli.gre_out {
        let p = GreParams {
            snr: cli.gre_snr,
            res_mm: cli.gre_res,
            snr_vol_exp: cli.gre_snr_vol_exp,
            output: cli.gre_output,
            rx_phase_rad: cli.gre_rx_phase,
            b0_field: cli.gre_b0field.clone(),
            ..Default::default()
        };
        let g = gre::synthesize(&GreObject {
            sig_grid: &sig_grid,
            acq_grid: &grid,
            fractions: [&sig_tissue.wm, &sig_tissue.gm, &sig_tissue.csf],
            s0: tissue_s0,
            t2_ms: &comp.t2,
            fmap_hz: &fmap,
            warp: gnl_warp,
            signal_scale: acq.signal_scale,
            seed: cli.seed,
        }, &p);
        println!(
            "GRE fieldmap: TE {:.2}/{:.2} ms, {}, {}, tissue SNR {} (sigma {:.3}){}",
            p.te_s[0] * 1e3, p.te_s[1] * 1e3,
            match p.output {
                GreOutput::Phase => format!("phase1/phase2 (rx-phase {:.1} rad)", p.rx_phase_rad),
                GreOutput::Phasediff => "phasediff".to_string(),
            },
            match p.res_mm {
                Some(r) => format!("{r} mm iso {:?} (complex intravoxel averaging)", g.grid.dims),
                None => "DWI acquisition grid".to_string(),
            },
            p.snr, g.sigma,
            if g.stamped { "; corner voxels stamped to 0/4095 so the phasediff spans its full range" } else { "" }
        );
        Some((prefix.clone(), g))
    } else {
        None
    };

    // Optional spatially-varying noise level map (acquisition grid, per-voxel per-component SD).
    let noise_sigma: Option<Vec<f32>> = if let Some(nmp) = &cli.noise_map {
        let (nm, ng) = io::load_volume(nmp)?;
        if ng.dims != grid.dims {
            return Err(format!(
                "--noise-map grid {:?} != acquisition grid {:?}", ng.dims, grid.dims).into());
        }
        println!("noise level map: {} (spatially-varying complex noise; ground truth written)",
            nmp.display());
        Some(nm)
    } else {
        None
    };

    let phase_model = match cli.phase_model.as_str() {
        "hbcd" => PhaseModel::hbcd_like(),
        "none" => PhaseModel::none(),
        other => return Err(format!("unknown --phase-model {other:?}; expected hbcd or none").into()),
    };

    let t = Instant::now();
    println!("Stage B (o={}: {}object phase '{}')",
        cli.oversample, if cli.oversample > 1 { "intrinsic Gibbs + " } else { "" }, cli.phase_model);
    let (mut mag, mut phase) = simulate_acquisition(
        &SimulationInput {
            sim_dims: sig_grid.dims,
            acq_dims: grid.dims,
            ngrad: comp.ngrad,
            images: &comp.images,
            t2: &comp.t2,
            fmap: &fmap,
            bvals: &scheme.bvals,
            bvecs: &scheme.bvecs,
            phase: &phase_model,
            seed: cli.seed,
            noise_sigma: noise_sigma.as_deref(),
        },
        &acq,
    );
    println!("Stage B: {:?}", t.elapsed());

    // Resolve the output orientation + PhaseEncodingDirection. PE is the grid y/j axis; a positive
    // off-resonance field displaces signal toward −grid-j on the forward scan (the Fiberfox k-space
    // convention this port preserves), +grid-j with --reverse-pe. `Reorient` carries that physics
    // into whatever frame we write, so the PED always describes the baked-in distortion — LAS
    // (radiological, dcm2niix/FSL) with --fsl-orientation, else the native grid frame.
    let total_readout_time = acq.t_line * grid.dims[1] as f64 / 1000.0; // PE-axis, before reindex
    let in_pe_sign = if cli.reverse_pe { 1 } else { -1 };
    let reo = if cli.fsl_orientation {
        Reorient::to_las(&grid.voxel_to_world, grid.dims)
    } else {
        Reorient::identity(grid.dims)
    };
    let phase_encoding_direction = reo.out_ped(1, in_pe_sign);
    if cli.fsl_orientation {
        mag = reo.apply_volume(&mag, comp.ngrad);
        phase = reo.apply_volume(&phase, comp.ngrad);
        grid.voxel_to_world = reo.apply_affine(&grid.voxel_to_world);
        grid.dims = reo.out_dims;
        println!("Reoriented output to FSL/dcm2niix (LAS), PhaseEncodingDirection={phase_encoding_direction}");
    }
    // The scheme's directions are world (RAS) -- that is how the signal encoded them and how
    // the truth peaks are written -- and FSL bvecs are voxel-frame with the determinant rule.
    for b in scheme.bvecs.iter_mut() {
        *b = Reorient::fsl_bvec(*b, &grid.voxel_to_world);
    }

    let sidecar = io::SidecarInfo {
        phase_encoding_direction,
        total_readout_time,
        echo_time: acq.t_echo / 1000.0,
        partial_fourier: acq.partial_fourier,
        accel: cli.accel,
        mb: cli.mb,
        // When a GRE fieldmap is written this run, tag the DWI as its B0FieldSource.
        b0_field_source: gre_fieldmap.as_ref().map(|(_, g)| g.b0_field.clone()),
    };

    io::write_complex_dwi(&cli.out, grid.dims, comp.ngrad, &mag, &phase, &grid, &scheme, &sidecar)?;
    println!("wrote BIDS {}_part-{{mag,phase}}_dwi.nii.gz (+bval/bvec/json)", cli.out);
    if let Some(ns) = &noise_sigma {
        let ns_out = if cli.fsl_orientation { reo.apply_volume(ns, 1) } else { ns.clone() };
        io::write_4d(&PathBuf::from(format!("{}_desc-noise_sigma.nii.gz", cli.out)),
            grid.dims, 1, &ns_out, &grid)?;
        println!("wrote noise-level ground truth {}_desc-noise_sigma.nii.gz", cli.out);
    }

    if let Some((prefix, g)) = &gre_fieldmap {
        io::write_gre_fieldmap(prefix, g, cli.fsl_orientation)?;
        match g.output {
            GreOutput::Phase => println!("wrote GRE fieldmap {prefix}_magnitude{{1,2}}/_phase{{1,2}} (+json)"),
            GreOutput::Phasediff => println!("wrote GRE fieldmap {prefix}_magnitude{{1,2}}/_phasediff (+json)"),
        }
    }

    if let Some(pk) = truth_peaks.as_ref() {
        let vol = reo.apply_volume(pk, 9);
        io::write_4d(&PathBuf::from(format!("{}_desc-truth_peaks.nii.gz", cli.out)), grid.dims, 9, &vol, &grid)?;
        std::fs::write(format!("{}_desc-truth_peaks.json", cli.out),
            "{\n  \"Description\": \"Ground-truth fibre orientations: up to 3 peaks of the orientation mixture per voxel, volumes 3k..3k+2 = peak k as a unit vector in world RAS scaled by its mass fraction (0 = no peak). Object in its true (gradient-nonlinearity-free) frame.\"\n}\n")?;
        println!("wrote truth peaks {}_desc-truth_peaks.nii.gz", cli.out);
    }
    if let Some((coef, _)) = gnl.as_ref() {
        std::fs::write(format!("{}_desc-gnl_coeff.grad", cli.out), coef.write_siemens())?;
        // Truth on the written (possibly reoriented) grid: the field is world-defined, so it is
        // simply re-evaluated there. Forward displacement d(r) = φ(r) − r warps points/streamlines
        // (true → apparent); the inverse φ⁻¹(x) − x pulls images (apparent ← true).
        let field_out = GnlField::on_grid(coef, &grid);
        let nvox = grid.dims.iter().product::<usize>();
        let (nx, ny) = (grid.dims[0], grid.dims[1]);
        let mut disp = vec![0.0f32; nvox * 3];
        let mut inv = vec![0.0f32; nvox * 3];
        for v in 0..nvox {
            disp[v * 3..v * 3 + 3].copy_from_slice(&field_out.disp[v]);
            let sv = field_out.src_vox[v];
            let src = voxel_to_world(&grid, [sv[0] as f64, sv[1] as f64, sv[2] as f64]);
            let here = voxel_to_world(&grid, [(v % nx) as f64, ((v / nx) % ny) as f64, (v / (nx * ny)) as f64]);
            for c in 0..3 {
                inv[v * 3 + c] = (src[c] - here[c]) as f32;
            }
        }
        io::write_4d(&PathBuf::from(format!("{}_desc-gnl_disp.nii.gz", cli.out)), grid.dims, 3, &disp, &grid)?;
        io::write_4d(&PathBuf::from(format!("{}_desc-gnl_invdisp.nii.gz", cli.out)), grid.dims, 3, &inv, &grid)?;
        let graddev = field_out.graddev_volumes(&grid);
        io::write_4d(&PathBuf::from(format!("{}_desc-gnl_graddev.nii.gz", cli.out)), grid.dims, 9, &graddev, &grid)?;
        std::fs::write(format!("{}_desc-gnl_graddev.json", cli.out), format!(
            "{{\n  \"Description\": \"Ground-truth gradient deviation: 9 volumes, HCP/FSL layout; read row-major into T, the applied gradient in this image's voxel axes is T.T @ g; identity included.\",\n  \"GradientNonlinearity\": \"{}\",\n  \"GradientNonlinearityScale\": {},\n  \"SpatialWarpApplied\": {},\n  \"EncodingDeviationApplied\": {},\n  \"JacobianModulation\": {}\n}}\n",
            cli.gnl.as_deref().unwrap_or(""), cli.gnl_scale, !cli.gnl_no_warp, !cli.gnl_no_encoding, !cli.gnl_no_jacobian_modulation))?;
        std::fs::write(format!("{}_desc-gnl_disp.json", cli.out),
            "{\n  \"Description\": \"Ground-truth gradient-nonlinearity displacement d(r) = phi(r) - r at each voxel centre r, three volumes = RAS x,y,z in mm (apparent minus true position). Apply to points/streamlines to warp true -> apparent.\"\n}\n")?;
        std::fs::write(format!("{}_desc-gnl_invdisp.json", cli.out),
            "{\n  \"Description\": \"Inverse gradient-nonlinearity displacement phi^-1(x) - x at each voxel centre x, three volumes = RAS x,y,z in mm (true minus apparent position). Resample images through it to pull apparent <- true.\"\n}\n")?;
        println!("wrote GNL truth {}_desc-gnl_{{coeff.grad,disp,invdisp,graddev}}", cli.out);
    }
    Ok(())
}
