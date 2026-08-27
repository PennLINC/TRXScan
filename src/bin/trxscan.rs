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
use trxscan::kspace::{simulate_acquisition, Acquisition};
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
    println!("grid {:?}  {} streamlines  {} volumes  shells {:?}", grid.dims,
        offsets.len().saturating_sub(1), scheme.len(), scheme.shells(50.0));

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
        generate_compartments_moving(&grid, &positions, &offsets, &tissue, &scheme, &params, &poses)
    } else {
        let mut mix = generate_mixture(
            &grid, &positions, &offsets, weights.as_deref(), &tissue, &params, kappa,
            HemiSphere::icosphere(3),
        );
        if let Some(mp) = &cli.myelin {
            let (my, mgrid) = io::load_volume(mp)?;
            if mgrid.dims != grid.dims {
                return Err(format!("myelin grid {:?} != tissue grid {:?}", mgrid.dims, grid.dims).into());
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
        let n_shots = (grid.dims[2] / cli.mb).max(1);
        let events = gen_dropout_events(&scheme.bvals, n_shots, cli.dropout_rate,
            0xB10C_5EED ^ cli.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let gt = motion::apply_multiband_motion(
            &mut comp.images, grid.dims, comp.ngrad, grid.voxel_to_world,
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
        ghost_offset: 0.015,    // subtle residual Nyquist ghost
        eddy_strength: cli.eddy,
        eddy_quad: cli.eddy_quad,
        eddy_tau: 70.0,
        n_spikes: 0,            // spikes are rare/aggressive; left off (available)
        spike_amplitude: 1.0,
        zero_ringing: 6.0,      // mild Gibbs
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
    let (mag, phase) =
        simulate_acquisition(grid.dims, comp.ngrad, &comp.images, &comp.t2, &fmap, &acq, &gradients);
    println!("Acquisition stage (distortion+T2*{}{}{}): {:?}",
        if cli.noise > 0.0 { "+noise" } else { "" },
        if cli.eddy > 0.0 { "+eddy" } else { "" },
        if cli.accel > 1 { "+GRAPPA" } else { "" },
        t.elapsed());

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
