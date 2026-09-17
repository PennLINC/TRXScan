//! End-to-end simulator: streamlines + tissue + scheme + fieldmap → BIDS complex 4D DWI.
//! Signal stage (per-voxel mixture → clean signal) → acquisition stage (EPI distortion, T2*, eddy,
//! partial Fourier, Gibbs ringing, spikes, multi-coil, GRAPPA, noise), with optional head motion.

use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use std::time::Instant;
use trxscan::compartments::{
    generate_compartments_moving, generate_mixture, signal_from_mixture_gnl, CompartmentParams,
};
use trxscan::gnl::{GnlField, GnlPreset, GradCoef};
use trxscan::io;
use trxscan::kspace::{
    simulate_acquisition_legacy, simulate_acquisition_oversampled, Acquisition, KspaceWindow,
    PartialFourierMode,
};
use trxscan::phase::PhaseModel;
use trxscan::motion;
use trxscan::orient::Reorient;
use trxscan::scheme::GradientScheme;
use trxscan::sphere::HemiSphere;

/// Compartment parameter preset (diffusivities + T2 relaxation times).
#[derive(Copy, Clone, Debug, ValueEnum)]
enum Preset {
    /// Fiberfox ffp legacy values (weak GM/WM contrast at low b by design)
    Neonatal,
    /// 3T literature T2s / diffusivities
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
    /// Off-resonance fieldmap in Hz, on the ACQUISITION grid. Required only for the legacy
    /// `--oversample 1` path; the default oversampled path takes `--sim-fmap` instead.
    ///
    /// It used to be unconditionally required, which meant a default run could fail on a missing
    /// or mis-gridded file whose values never reached the output.
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
    /// round trip, and the output has NO ringing and no object phase (the legacy path).
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
    /// Linear eddy-current strength (DWI volumes only; b0 exempt)
    #[arg(long, default_value_t = 0.0, value_name = "S")]
    eddy: f64,
    /// Quadratic eddy-current strength
    #[arg(long, default_value_t = 0.0, value_name = "S")]
    eddy_quad: f64,
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
    /// the signal stage to the faithful per-volume re-simulation path.
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

    /// Gradient nonlinearity: a preset ("whole-body-80", "connectom-300") or a Siemens `.grad`
    /// coefficient file. Adds the spatial encoding warp AND the per-voxel diffusion-encoding
    /// deviation; writes the coefficient file, the displacement field and the graddev image
    /// next to the DWI. See docs/GNL.md.
    #[arg(long, value_name = "PRESET|FILE")]
    gnl: Option<String>,
    /// Multiply the nonlinear coefficients (l >= 3) by this factor (severity knob).
    #[arg(long, default_value_t = 1.0, value_name = "S")]
    gnl_scale: f64,
    /// Scanner isocentre, world RAS mm as "x,y,z". Default: the world origin. Every written
    /// header is translated so this point becomes the origin, which is where TORTOISE's
    /// coefficient evaluation puts the isocentre.
    #[arg(long, value_name = "X,Y,Z")]
    isocenter: Option<String>,
    /// GNL: skip the spatial warp (diffusion-encoding deviation only).
    #[arg(long)]
    gnl_no_warp: bool,
    /// GNL: skip the diffusion-encoding deviation (spatial warp only).
    #[arg(long)]
    gnl_no_encoding: bool,
    /// GNL: do not modulate warped intensities by 1/|det J|.
    #[arg(long)]
    gnl_no_jacobian_modulation: bool,
    /// GNL: print the field envelope at 2/5/8/10/12 cm from the isocentre and exit.
    #[arg(long)]
    gnl_info: bool,
    /// Also synthesize a dual-echo GRE fieldmap (magnitude1/magnitude2/phasediff + sidecars,
    /// Siemens conventions) from the same off-resonance field, at this BIDS stem. Sees the same
    /// GNL warp as the DWI.
    #[arg(long, value_name = "PREFIX")]
    gre_out: Option<String>,
    /// Tissue SNR of the synthetic GRE echoes (complex Gaussian noise on both echoes; 0 =
    /// noiseless). Real phase-difference images span their whole integer range because the
    /// background phase is uniformly random, and qsiprep's Siemens phase conversion relies on
    /// that (it maps the image's min/max onto -pi..pi); a noiseless phasediff gets stretched.
    #[arg(long, default_value_t = 50.0)]
    gre_snr: f64,
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
        // `--fmap` is required HERE and only here: the legacy path is the only one that uses it.
        (tissue, grid.clone(), cli.fmap.clone().ok_or(
            "--oversample 1 (the legacy path) requires --fmap, the acquisition-grid fieldmap. \
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
    // oversampling, `--fmap` on the legacy path. Loading and grid-checking `--fmap` here as well
    // made a default run depend on a file whose values could not influence its output.
    let (mut fmap, fgrid) = io::load_volume(&sig_fmap_path)?;
    if fgrid.dims != sig_grid.dims {
        return Err(format!("fieldmap grid {:?} != signal grid {:?}",
                           fgrid.dims, sig_grid.dims).into());
    }

    // Scanner isocentre: translate the world frame so it sits at the origin. Streamlines live in
    // world mm and are rasterised through the grid affines, so they move with the grids.
    let mut sig_grid = sig_grid;
    if let Some(spec) = &cli.isocenter {
        let v: Vec<f64> = spec.split(',').map(|t| t.trim().parse::<f64>()).collect::<Result<_, _>>()
            .map_err(|_| format!("--isocenter expects x,y,z in mm, got {spec:?}"))?;
        if v.len() != 3 {
            return Err(format!("--isocenter expects three values, got {spec:?}").into());
        }
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

    // Gradient nonlinearity: the coefficient set and its field on both grids.
    let gnl: Option<(GradCoef, GnlField, GnlField)> = if let Some(spec) = &cli.gnl {
        let mut coef = match spec.as_str() {
            "whole-body-80" => GradCoef::preset(GnlPreset::WholeBody80),
            "connectom-300" => GradCoef::preset(GnlPreset::Connectom300),
            path => GradCoef::parse_siemens(&std::fs::read_to_string(path)?)
                .map_err(|e| format!("--gnl {path}: {e}"))?,
        };
        coef.scale_nonlinear(cli.gnl_scale);
        let t = Instant::now();
        let field_sig = GnlField::on_grid(&coef, &sig_grid, [0.0; 3]);
        let field_acq = GnlField::on_grid(&coef, &grid, [0.0; 3]);
        println!("GNL: {} ({} terms, scale {}), fields built in {:?}",
            spec, coef.terms.len(), cli.gnl_scale, t.elapsed());
        for radius in [20.0, 50.0, 80.0, 100.0, 120.0] {
            let e = field_sig.envelope(&sig_grid, radius);
            println!("  within {radius:>5.0} mm: {:>7} voxels  max |d| {:.2} mm  max gradient dev {:.2} %  max angle {:.2} deg",
                e.n_voxels, e.max_disp_mm, 100.0 * e.max_gradient_dev, e.max_angle_deg);
        }
        if cli.gnl_info {
            return Ok(());
        }
        Some((coef, field_sig, field_acq))
    } else {
        None
    };
    let gnl_encoding = gnl.as_ref().filter(|_| !cli.gnl_no_encoding).map(|(_, f, _)| f);
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

    let base = cli.params.params();
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
        if weights.is_some() || kappa.is_some() {
            eprintln!("note: weights/kappa are ignored in motion mode (per-segment re-simulation)");
        }
        let poses = motion::load_motion_tsv(tsv)?;
        let moved = poses.iter().filter(|p| **p != motion::Pose::IDENTITY).count();
        println!("Motion (faithful re-simulation): {} poses, {} moved, from {}",
            poses.len(), moved, tsv.display());
        if gnl.is_some() {
            return Err("--gnl is not supported together with --motion yet".into());
        }
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
            let o = if cli.oversample > 1 { cli.oversample } else { 1 };
            let (pk, _) = trxscan::truth::truth_peaks(&mix, o, 3);
            truth_peaks = Some(pk);
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

    let mut gre_outputs: Option<(Vec<Vec<f32>>, Vec<i16>, f64, f64, String)> = None;

    // Gradient nonlinearity, spatial part: the scanner encodes tissue at φ(r), so every
    // compartment image and the off-resonance field move to the apparent frame before k-space
    // encoding (kspace looks the field up at the voxel it is encoding).
    if let Some((_, field_sig, _)) = gnl.as_ref().filter(|_| !cli.gnl_no_warp) {
        let t = Instant::now();
        let modulate = !cli.gnl_no_jacobian_modulation;
        for img in comp.images.iter_mut() {
            *img = field_sig.warp_4d(img, comp.ngrad, modulate);
        }
        fmap = field_sig.warp_volume(&fmap, false);
        println!("GNL spatial warp of {} compartment images + fieldmap: {:?}", comp.images.len(), t.elapsed());
    }

    // Dual-echo GRE fieldmap from the same object: magnitudes from the tissue mixture with the
    // compartment T2s, phase difference from the (already warped) off-resonance field.
    if let Some(gre_prefix) = &cli.gre_out {
        let o = if cli.oversample > 1 { cli.oversample } else { 1 };
        let (te1, te2) = (4.92e-3, 7.38e-3); // s, the Siemens gre_field_mapping defaults
        let nvox_sig = sig_grid.dims.iter().product::<usize>();
        let fractions = [&sig_tissue.wm, &sig_tissue.gm, &sig_tissue.csf];
        let mut mags = Vec::new();
        for te in [te1, te2] {
            let mut m = vec![0.0f32; nvox_sig];
            for v in 0..nvox_sig {
                let mut acc = 0.0f64;
                for (c, frac) in fractions.iter().enumerate() {
                    acc += frac[v] as f64 * (-(te * 1000.0) / comp.t2[c] as f64).exp();
                }
                m[v] = (acc * acq_signal_scale()) as f32;
            }
            if let Some((_, field_sig, _)) = gnl.as_ref().filter(|_| !cli.gnl_no_warp) {
                m = field_sig.warp_volume(&m, !cli.gnl_no_jacobian_modulation);
            }
            mags.push(downsample_inplane(&m, sig_grid.dims, o));
        }
        let fmap_acq = downsample_inplane(&fmap, sig_grid.dims, o);
        // Complex noise on each echo: the magnitudes become Rician and the phase difference
        // wraps uniformly wherever there is no signal.
        let sigma = if cli.gre_snr > 0.0 { acq_signal_scale() / cli.gre_snr } else { 0.0 };
        let mut rng = trxscan::kspace::Rng(0x6E7E_F1E1_D000_0000 ^ cli.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut phasediff = Vec::with_capacity(fmap_acq.len());
        for v in 0..fmap_acq.len() {
            let phi = 2.0 * std::f64::consts::PI * fmap_acq[v] as f64 * (te2 - te1);
            let (m1, m2) = (mags[0][v] as f64, mags[1][v] as f64);
            let s1 = (m1 + sigma * rng.gauss(), sigma * rng.gauss());
            let s2 = (m2 * phi.cos() + sigma * rng.gauss(), m2 * phi.sin() + sigma * rng.gauss());
            // arg(s2 * conj(s1)) is already wrapped to (-pi, pi]
            let dphi = (s2.1 * s1.0 - s2.0 * s1.1).atan2(s2.0 * s1.0 + s2.1 * s1.1);
            mags[0][v] = (s1.0 * s1.0 + s1.1 * s1.1).sqrt() as f32;
            mags[1][v] = (s2.0 * s2.0 + s2.1 * s2.1).sqrt() as f32;
            phasediff.push(((dphi + std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * 4096.0).round().clamp(0.0, 4095.0) as i16);
        }
        let stamped = trxscan::gnl::stamp_phasediff_range(&mut phasediff);
        println!(
            "GRE fieldmap: TE {:.2}/{:.2} ms, tissue SNR {} (sigma {:.3}){}",
            te1 * 1e3, te2 * 1e3, cli.gre_snr, sigma,
            if stamped { "; corner voxels stamped to 0/4095 so the phasediff spans its full range" } else { "" }
        );
        gre_outputs = Some((mags, phasediff, te1, te2, gre_prefix.clone()));
    }

    // HBCD-like acquisition, per-compartment T2, optional Rician noise. The PE-train duration
    // is pinned to the HBCD protocol's TotalReadoutTime (0.0917 s) for ANY matrix:
    // distortion shift = fmap · ny · t_line, so t_line = 91.7 ms / ny.
    let acq = Acquisition {
        t_line: 91.7 / grid.dims[1] as f64,
        t_echo: 88.0,
        t_inhom: 50.0,
        signal_scale: 100.0,
        reverse_phase: cli.reverse_pe,
        do_distortions: true,
        do_relaxation: true,
        noise_variance: cli.noise,
        partial_fourier: 0.75,  // HBCD
        // Defaults to CONTIGUOUS: this CLI describes itself as HBCD-like, and a scanner produces
        // contiguous PF. The Fiberfox rule keeps ~78% at a nominal 6/8 because it preserves line
        // zero on even matrices; it stays available via --pf-mode fiberfox.
        pf_mode: match cli.pf_mode.as_str() {
            "contiguous" => PartialFourierMode::Contiguous,
            "fiberfox" => PartialFourierMode::FiberfoxCompatible,
            o => return Err(format!("unknown --pf-mode {o:?}; expected contiguous or fiberfox").into()),
        },
        ghost_offset: 0.015,    // subtle residual Nyquist ghost
        eddy_strength: cli.eddy,
        eddy_quad: cli.eddy_quad,
        eddy_tau: 70.0,
        n_spikes: 0,            // spikes are rare/aggressive; left off (available)
        spike_amplitude: 1.0,
        window: KspaceWindow::None,  // unapodized; ringing comes from the crop when --oversample > 1
        n_coils: cli.coils,
        accel: cli.accel,
        acs_lines: 24,
        seed: cli.seed,
    };
    // per-volume eddy gradient = unit bvec × b-value (b0 → zero → no eddy)
    let gradients: Vec<[f64; 3]> = (0..scheme.len())
        .map(|g| {
            let (d, b) = (scheme.bvecs[g], scheme.bvals[g]);
            [d[0] * b, d[1] * b, d[2] * b]
        })
        .collect();
    let t = Instant::now();

    let phase_model = match cli.phase_model.as_str() {
        "hbcd" => PhaseModel::hbcd_like(),
        "none" => PhaseModel::none(),
        other => return Err(format!("unknown --phase-model {other:?}; expected hbcd or none").into()),
    };

    let (mut mag, mut phase) = if cli.oversample > 1 {
        // `comp` was already built on the simulation grid above, so motion, SIFT2 weights, kappa,
        // myelin and dropout all apply here exactly as they do on the nominal path.
        println!("Stage B (oversampled o={}: intrinsic Gibbs + object phase '{}')",
                 cli.oversample, cli.phase_model);
        simulate_acquisition_oversampled(
            sig_grid.dims, grid.dims, comp.ngrad, &comp.images, &comp.t2,
            &fmap, &acq, &scheme.bvals, &scheme.bvecs, &phase_model, cli.seed)
    } else {
        eprintln!(
            "WARNING: --oversample 1 uses the legacy path. The object sits on the reconstruction \
             matrix, so the transforms are an exact round trip: the output will contain NO Gibbs \
             ringing and no object phase. Use --oversample 2 (default) or 4 for a realistic \
             acquisition.");
        simulate_acquisition_legacy(
            grid.dims, comp.ngrad, &comp.images, &comp.t2, &fmap, &acq, &gradients)
    };
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
    };

    io::write_complex_dwi(&cli.out, grid.dims, comp.ngrad, &mag, &phase, &grid, &scheme, &sidecar)?;
    println!("wrote BIDS {}_part-{{mag,phase}}_dwi.nii.gz (+bval/bvec/json)", cli.out);

    if let Some((mags, phasediff, te1, te2, prefix)) = gre_outputs.take() {
        let p = |s: &str| PathBuf::from(format!("{prefix}{s}"));
        for (k, m) in mags.iter().enumerate() {
            let vol = reo.apply_volume(m, 1);
            io::write_3d(&p(&format!("_magnitude{}.nii.gz", k + 1)), grid.dims, &vol, &grid)?;
            let te = if k == 0 { te1 } else { te2 };
            std::fs::write(p(&format!("_magnitude{}.json", k + 1)), format!(
                "{{\n  \"Manufacturer\": \"TRXScan\",\n  \"EchoTime\": {te:.5},\n  \"ImageType\": [\"ORIGINAL\", \"PRIMARY\", \"M\", \"ND\"]\n}}\n"))?;
        }
        let ph_f: Vec<f32> = phasediff.iter().map(|&v| v as f32).collect();
        let ph_reo: Vec<i16> = reo.apply_volume(&ph_f, 1).iter().map(|&v| v as i16).collect();
        io::write_3d_i16(&p("_phasediff.nii.gz"), grid.dims, &ph_reo, &grid)?;
        std::fs::write(p("_phasediff.json"), format!(
            "{{\n  \"Manufacturer\": \"TRXScan\",\n  \"EchoTime1\": {te1:.5},\n  \"EchoTime2\": {te2:.5},\n  \"ImageType\": [\"ORIGINAL\", \"PRIMARY\", \"P\", \"ND\", \"PHASE\"],\n  \"IntendedFor\": []\n}}\n"))?;
        println!("wrote GRE fieldmap {prefix}_magnitude{{1,2}}/_phasediff (+json)");
    }

    if let Some(pk) = truth_peaks.as_ref() {
        let vol = reo.apply_volume(pk, 9);
        io::write_4d(&PathBuf::from(format!("{}_desc-truth_peaks.nii.gz", cli.out)), grid.dims, 9, &vol, &grid)?;
        std::fs::write(format!("{}_desc-truth_peaks.json", cli.out),
            "{\n  \"Description\": \"Ground-truth fibre orientations: up to 3 peaks of the orientation mixture per voxel, volumes 3k..3k+2 = peak k as a unit vector in world RAS scaled by its mass fraction (0 = no peak). Object in its true (gradient-nonlinearity-free) frame.\"\n}\n")?;
        println!("wrote truth peaks {}_desc-truth_peaks.nii.gz", cli.out);
    }
    if let Some((coef, _, _)) = gnl.as_ref() {
        std::fs::write(format!("{}_desc-gnl_coeff.grad", cli.out), coef.write_siemens())?;
        // Truth on the written (possibly reoriented) grid: the field is world-defined, so it is
        // simply re-evaluated there.
        let field_out = GnlField::on_grid(coef, &grid, [0.0; 3]);
        let nvox = grid.dims.iter().product::<usize>();
        let mut disp = vec![0.0f32; nvox * 3];
        for v in 0..nvox {
            disp[v * 3..v * 3 + 3].copy_from_slice(&field_out.disp[v]);
        }
        io::write_4d(&PathBuf::from(format!("{}_desc-gnl_disp.nii.gz", cli.out)), grid.dims, 3, &disp, &grid)?;
        let graddev = field_out.graddev_volumes(&grid);
        io::write_4d(&PathBuf::from(format!("{}_desc-gnl_graddev.nii.gz", cli.out)), grid.dims, 9, &graddev, &grid)?;
        std::fs::write(format!("{}_desc-gnl_graddev.json", cli.out), format!(
            "{{\n  \"Description\": \"Ground-truth gradient deviation: 9 volumes, HCP/FSL layout; read row-major into T, the applied gradient in this image's voxel axes is T.T @ g; identity included.\",\n  \"GradientNonlinearity\": \"{}\",\n  \"GradientNonlinearityScale\": {},\n  \"SpatialWarpApplied\": {},\n  \"EncodingDeviationApplied\": {},\n  \"JacobianModulation\": {}\n}}\n",
            cli.gnl.as_deref().unwrap_or(""), cli.gnl_scale, !cli.gnl_no_warp, !cli.gnl_no_encoding, !cli.gnl_no_jacobian_modulation))?;
        std::fs::write(format!("{}_desc-gnl_disp.json", cli.out),
            "{\n  \"Description\": \"Ground-truth gradient-nonlinearity displacement d(r) = phi(r) - r at each voxel centre r, three volumes = RAS x,y,z in mm (apparent minus true position).\"\n}\n")?;
        println!("wrote GNL truth {}_desc-gnl_{{coeff.grad,disp,graddev}}", cli.out);
    }
    Ok(())
}

/// The acquisition's nominal signal scale (matches `Acquisition::signal_scale` in `main`).
fn acq_signal_scale() -> f64 {
    100.0
}

/// Block-average an in-plane oversampled volume (`sim = [nx*o, ny*o, nz]`) down to `[nx, ny, nz]`.
fn downsample_inplane(v: &[f32], sim: [usize; 3], o: usize) -> Vec<f32> {
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
