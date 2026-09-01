//! End-to-end simulator: streamlines + tissue + scheme + fieldmap → BIDS complex 4D DWI.
//! Signal stage (per-voxel mixture → clean signal) → acquisition stage (EPI distortion, T2*, eddy,
//! partial Fourier, Gibbs ringing, spikes, multi-coil, GRAPPA, noise), with optional head motion.

use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use std::time::Instant;
use trxscan::compartments::{
    generate_compartments_moving, generate_mixture, signal_from_mixture, CompartmentParams,
};
use trxscan::io;
use trxscan::kspace::{
    simulate_acquisition_legacy, simulate_acquisition_oversampled, Acquisition, KspaceWindow,
    PartialFourierMode,
};
use trxscan::phase::PhaseModel;
use trxscan::motion;
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
    /// Off-resonance fieldmap in Hz (NIfTI, same grid as the tissue maps)
    #[arg(long, value_name = "NII")]
    fmap: PathBuf,
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
    /// ~15.6% at o=2) but is not memory-safe on ordinary hardware: in-plane voxels scale as o^2,
    /// a measured end-to-end o=4 run peaked at 13.0 GB single-threaded, and under `--features par`
    /// the signal stage's f64 accumulator alone is 16.1 GB per buffer with a pair live during the
    /// reduction. Use o=4 with >=32 GB, or wait for z-slab streaming.
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
    /// Complex k-space noise variance (→ Rician magnitude)
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

    let (tissue, grid) = io::load_tissue(&cli.wm, &cli.gm, &cli.csf, &cli.mask)?;

    // The signal stage runs on ONE grid, chosen here. With --oversample > 1 that is the finer
    // simulation grid; the acquisition grid still defines the output matrix. Building the nominal
    // `comp` and then discarding it would silently drop motion, SIFT2 weights, kappa, myelin and
    // dropout -- and waste a full Stage A -- which is exactly what an earlier version did.
    let (sig_tissue, sig_grid, sig_fmap_path) = if cli.oversample > 1 {
        let need = |o: &Option<PathBuf>, n: &str| -> Result<PathBuf, String> {
            o.clone().ok_or_else(|| format!(
                "--oversample {} requires --{n}; generate the simulation grid with \
                 scripts/prepare_acquisition_grid.py --oversample {}. Upsampling the \
                 acquisition-grid maps would add no k-space content and produce no ringing.",
                cli.oversample, cli.oversample))
        };
        let (st, sg) = io::load_tissue(
            &need(&cli.sim_wm, "sim-wm")?, &need(&cli.sim_gm, "sim-gm")?,
            &need(&cli.sim_csf, "sim-csf")?, &need(&cli.sim_mask, "sim-mask")?)?;
        let o = cli.oversample;
        if sg.dims != [grid.dims[0] * o, grid.dims[1] * o, grid.dims[2]] {
            return Err(format!(
                "simulation grid {:?} is not {o}x the acquisition grid {:?} in-plane \
                 (z is never oversampled)", sg.dims, grid.dims).into());
        }
        (st, sg, need(&cli.sim_fmap, "sim-fmap")?)
    } else {
        (tissue, grid.clone(), cli.fmap.clone())
    };
    let (mut positions, mut offsets, mut weights) =
        io::load_streamlines_spec(&cli.streamlines, cli.weights.as_deref())?;
    if let Some(n) = cli.subsample {
        (positions, offsets, weights) =
            io::subsample_streamlines(positions, offsets, weights, n, cli.seed);
    }
    let scheme = GradientScheme::from_fsl(&cli.bval, &cli.bvec)?;
    let (fmap, fgrid) = io::load_volume(&cli.fmap)?;
    if fgrid.dims != grid.dims {
        return Err(format!("fieldmap grid {:?} != tissue grid {:?}", fgrid.dims, grid.dims).into());
    }
    println!("acquisition grid {:?}  signal grid {:?}  {} streamlines  {} volumes  shells {:?}",
        grid.dims, sig_grid.dims, offsets.len().saturating_sub(1), scheme.len(),
        scheme.shells(50.0));

    let base = cli.params.params();
    println!("compartment params: {}  T2 fiber/gm/csf {}/{}/{} ms",
        cli.params, base.t2_fiber, base.t2_gm, base.t2_csf);
    let params = CompartmentParams { b_value: scheme.b_max, ..base };
    let t = Instant::now();
    // With a motion trace, use the faithful path: per volume, transform the streamlines + tissue
    // by that volume's pose and re-simulate (Fiberfox-style). Otherwise the histogram-first
    // mixture path — SIFT2 weights + Watson κ dispersion, and the same
    // mixture `trxscan-microstructure` computes ground truth from.
    let mut comp = if let Some(tsv) = &cli.motion {
        if weights.is_some() || kappa.is_some() {
            eprintln!("note: weights/kappa are ignored in motion mode (per-segment re-simulation)");
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
        let fb = mix.fallback.iter().filter(|&&f| f == 1).count();
        println!(
            "Mixture: {} fallback WM voxels{}{}",
            fb,
            if weights.is_some() { ", SIFT2-weighted" } else { "" },
            kappa.map(|k| format!(", Watson kappa {k}")).unwrap_or_default()
        );
        signal_from_mixture(&mix, &scheme)
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

    let (mag, phase) = if cli.oversample > 1 {
        // `comp` was already built on the simulation grid above, so motion, SIFT2 weights, kappa,
        // myelin and dropout all apply here exactly as they do on the nominal path.
        let (sim_fmap, sim_fgrid) = io::load_volume(&sig_fmap_path)?;
        if sim_fgrid.dims != sig_grid.dims {
            return Err(format!("sim fieldmap grid {:?} != simulation grid {:?}",
                               sim_fgrid.dims, sig_grid.dims).into());
        }
        // Pre-flight memory estimate. The DOMINANT allocation on the default no-motion path is
        // generate_mixture's dense orientation histogram, nvox * 321 * 8 bytes of f64, plus its
        // f32 ODF copy -- not the compartment images. An earlier version estimated only the
        // latter and so predicted 6.0 GB for a run that measured 11.5 GB.
        //
        // Measured, full 1 M-streamline tractogram, 107x151x104 acquisition grid, 75 volumes:
        //   o=2  ->  11.46 GB peak RSS
        // The arithmetic upper bound is higher (~26 GB at o=2) because the histogram is allocated
        // densely but committed lazily: only voxels a streamline actually touches fault in.
        let sim_vox = sig_grid.dims.iter().product::<usize>() as f64;
        let nvert = 321.0;                       // HemiSphere::icosphere(3)
        let hist_gb = sim_vox * nvert * 12.0 / 1e9;   // f64 histogram + f32 ODF, upper bound
        let images_gb = sim_vox * scheme.len() as f64 * 3.0 * 4.0 / 1e9;
        let est_gb = hist_gb + images_gb;
        if est_gb > 8.0 {
            eprintln!(
                "WARNING: signal stage upper bound about {est_gb:.1} GB on grid {:?} \
                 ({:.1} GB dense orientation histogram + ODF, {:.1} GB compartment images, \
                 {} volumes). Committed memory is lower -- an o=2 run of this size measured \
                 11.5 GB -- because the histogram faults in only where streamlines deposit. \
                 In-plane memory scales as o^2.",
                sig_grid.dims, hist_gb, images_gb, scheme.len());
        }
        println!("Stage B (oversampled o={}: intrinsic Gibbs + object phase '{}')",
                 cli.oversample, cli.phase_model);
        simulate_acquisition_oversampled(
            sig_grid.dims, grid.dims, comp.ngrad, &comp.images, &comp.t2,
            &sim_fmap, &acq, &scheme.bvals, &scheme.bvecs, &phase_model, cli.seed)
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

    let sidecar = io::SidecarInfo {
        reverse_pe: cli.reverse_pe,
        total_readout_time: acq.t_line * grid.dims[1] as f64 / 1000.0,
        echo_time: acq.t_echo / 1000.0,
        partial_fourier: acq.partial_fourier,
        accel: cli.accel,
        mb: cli.mb,
    };

    io::write_complex_dwi(&cli.out, grid.dims, comp.ngrad, &mag, &phase, &grid, &scheme, &sidecar)?;
    println!("wrote BIDS {}_part-{{mag,phase}}_dwi.nii.gz (+bval/bvec/json)", cli.out);
    Ok(())
}
