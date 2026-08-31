//! `trxscan-microstructure` — ground-truth microstructure scalar maps from a tractogram + tissue
//! maps, with no acquisition and no noise. Builds the per-voxel orientation mixture
//! (`generate_mixture`) and evaluates the FORCE closed forms: DTI/DKI, QTI/µFA,
//! MAP-MRI, NG/PA, GFA/QA, plus NODDI-style ICVF/ODI/ISOVF — 27 maps.
//!
//! The maps come from the same mixture the simulator evaluates its signal from, so they are the
//! exact analytic answer key for a noise-free, artifact-free signal. Run with
//! `--help` for the full argument list.

use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use std::time::Instant;
use trxscan::compartments::{generate_mixture, CompartmentParams};
use trxscan::io;
use trxscan::microstructure::{field_scalars, ScalarConfig, SCALAR_NAMES};
use trxscan::sphere::HemiSphere;

/// Compartment parameter preset (diffusivities + T2 relaxation times).
#[derive(Copy, Clone, Debug, ValueEnum)]
enum Preset {
    /// Fiberfox ffp legacy values
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

/// Ground-truth microstructure scalar maps from a tractogram + tissue maps (no acquisition).
#[derive(Parser)]
#[command(name = "trxscan-microstructure", version)]
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
    /// Output prefix (writes <prefix>_<map>.nii.gz for each scalar)
    #[arg(short, long, value_name = "PREFIX")]
    out: String,

    /// Watson dispersion concentration κ for the orientation histogram
    /// (omitted or 0 = nearest-vertex binning, no added dispersion)
    #[arg(long, value_name = "KAPPA")]
    kappa: Option<f64>,
    /// SIFT2 weights: a TRX data-per-streamline array name, or an MRtrix tcksift2 text file
    #[arg(long, value_name = "SPEC")]
    weights: Option<String>,
    /// Diffusion Δ in seconds. With --small-delta, sets tau = Δ − δ/3 so MAP-MRI maps come out in
    /// physical mm units; omitted → dipy's normalized-units default (contrast only)
    #[arg(long, value_name = "SEC")]
    big_delta: Option<f64>,
    /// Diffusion δ in seconds (see --big-delta)
    #[arg(long, value_name = "SEC")]
    small_delta: Option<f64>,
    /// Compartment parameter preset
    #[arg(long, value_enum, default_value_t = Preset::Neonatal)]
    params: Preset,
    /// Per-voxel myelination map (0..1): lerps the WM compartment toward the adult endpoint
    #[arg(long, value_name = "NII")]
    myelin: Option<PathBuf>,

    /// Random seed for --subsample; match trxscan's --seed so ground truth and simulation
    /// describe the same phantom
    #[arg(long, default_value_t = 0, value_name = "SEED")]
    seed: u64,
    /// Keep only N streamlines, sampled with probability proportional to the SIFT2 weight
    /// (uniform without weights); identical N and --seed select the identical subset in trxscan
    #[arg(long, value_name = "N")]
    subsample: Option<usize>,
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
    let n_streamlines = offsets.len().saturating_sub(1);
    println!(
        "grid {:?}  {} streamlines{}  kappa {:?}",
        grid.dims,
        n_streamlines,
        if weights.is_some() { " (weighted)" } else { "" },
        kappa
    );

    let params = cli.params.params();
    println!("compartment params: {}", cli.params);
    let t = Instant::now();
    let mut mix = generate_mixture(
        &grid, &positions, &offsets, weights.as_deref(), &tissue, &params, kappa,
        HemiSphere::icosphere(3),
    );
    if let Some(mp) = &cli.myelin {
        let (my, mgrid) = io::load_volume(mp)?;
        if mgrid.dims != grid.dims {
            return Err(format!("myelin grid {:?} != tissue grid {:?}", mgrid.dims, grid.dims).into());
        }
        println!("myelin map: {}", mp.display());
        mix.myelin = Some(my);
    }
    let brain = (0..mix.nvox()).filter(|&v| mix.is_masked(v)).count();
    let fb = mix.fallback.iter().filter(|&&f| f == 1).count();
    println!(
        "mixture: {} brain voxels, {} WM-without-streamlines fallback  [{:?}]",
        brain, fb, t.elapsed()
    );

    let mut cfg = ScalarConfig::default();
    match (cli.big_delta, cli.small_delta) {
        (Some(bd), Some(sd)) => {
            cfg.tau = bd - sd / 3.0;
            println!("tau = {:.6} s (physical units, from Δ={bd}, δ={sd})", cfg.tau);
        }
        _ => println!("tau = 1/(4π²) (dipy default — normalized units, contrast only)"),
    }

    let t = Instant::now();
    let maps = field_scalars(&mix, &cfg);
    println!("closed forms: {} maps  [{:?}]", maps.len(), t.elapsed());

    io::write_scalar_maps(&cli.out, grid.dims, &SCALAR_NAMES, &maps, &grid)?;
    println!("wrote {}_{{{}}}.nii.gz", cli.out, SCALAR_NAMES.join(","));
    Ok(())
}
