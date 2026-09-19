//! Emit a synthetic gradient-nonlinearity field on a reference grid, WITHOUT simulating a DWI.
//!
//! Writes the coefficient file (for `qsiprep --gradient-file`), the forward displacement
//! `d_fwd(r) = phi(r) - r` (warp true -> apparent; apply to STREAMLINES/points), the inverse
//! displacement `d_inv(x) = phi^-1(x) - x` (pull apparent <- true; resample tissue IMAGES), and
//! the graddev tensor. All three volumes are RAS mm on the reference grid; a small Python step
//! converts the displacements to ITK 5-D (ANTs) / h5 (trxrs) form. See docs/GNL.md.
use std::path::PathBuf;
use clap::Parser;
use trxscan::gnl::{GnlField, GnlPreset, GradCoef};
use trxscan::io;
use trxscan::raster::Grid;

#[derive(Parser, Debug)]
#[command(about = "Synthetic gradient-nonlinearity field export (no acquisition)")]
struct Cli {
    /// Preset ("whole-body-80", "connectom-300") or a Siemens `.grad` coefficient file.
    #[arg(long, value_name = "PRESET|FILE")]
    gnl: String,
    /// Multiply the nonlinear coefficients (l >= 3) by this severity factor.
    #[arg(long, default_value_t = 1.0, value_name = "S")]
    gnl_scale: f64,
    /// Scanner isocentre, world RAS mm "x,y,z" (default: world origin).
    #[arg(long, value_name = "X,Y,Z")]
    isocenter: Option<String>,
    /// Reference NIfTI whose grid (dims + affine) the field is sampled on.
    #[arg(long, value_name = "NII")]
    reference: PathBuf,
    /// Output prefix.
    #[arg(short, long, value_name = "PREFIX")]
    out: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let (_data, grid): (Vec<f32>, Grid) = io::load_volume(&cli.reference)?;
    let iso = match &cli.isocenter {
        Some(spec) => {
            let v: Vec<f64> = spec.split(',').map(|t| t.trim().parse::<f64>())
                .collect::<Result<_, _>>().map_err(|_| format!("--isocenter x,y,z, got {spec:?}"))?;
            if v.len() != 3 { return Err(format!("--isocenter needs three values, got {spec:?}").into()); }
            [v[0], v[1], v[2]]
        }
        None => [0.0; 3],
    };
    // isocentre handling matches trxscan: evaluate the field about the origin, so translate the
    // grid so `iso` becomes the origin (streamlines/tissue must be translated the same way, or use
    // --isocenter 0,0,0 when the reference frame already has the isocentre at the origin).
    let mut grid = grid;
    for r in 0..3 { grid.voxel_to_world[r][3] -= iso[r]; }

    let mut coef = if cli.gnl == "whole-body-80" {
        GradCoef::preset(GnlPreset::WholeBody80)
    } else if cli.gnl == "connectom-300" {
        GradCoef::preset(GnlPreset::Connectom300)
    } else {
        GradCoef::parse_siemens(&std::fs::read_to_string(&cli.gnl)
            .map_err(|e| format!("cannot read coefficient file {}: {e}", cli.gnl))?)?
    };
    coef.scale_nonlinear(cli.gnl_scale);

    let field = GnlField::on_grid(&coef, &grid, [0.0; 3]);
    let e = field.envelope(&grid, 100.0);
    println!("GNL field on {:?}: within 100mm  max |d|={:.2} mm  |Jᵀĝ|-1={:.1}%  angle={:.2}°",
        grid.dims, e.max_disp_mm, 100.0 * e.max_gradient_dev, e.max_angle_deg);

    std::fs::write(format!("{}_coeff.grad", cli.out), coef.write_siemens())?;

    let nvox = grid.dims[0] * grid.dims[1] * grid.dims[2];
    let (nx, ny) = (grid.dims[0], grid.dims[1]);
    let w = &grid.voxel_to_world;
    // world position of voxel-centre linear index v
    let world = |v: usize| -> [f64; 3] {
        let (x, y, z) = (v % nx, (v / nx) % ny, v / (nx * ny));
        let (xf, yf, zf) = (x as f64, y as f64, z as f64);
        [w[0][0]*xf + w[0][1]*yf + w[0][2]*zf + w[0][3],
         w[1][0]*xf + w[1][1]*yf + w[1][2]*zf + w[1][3],
         w[2][0]*xf + w[2][1]*yf + w[2][2]*zf + w[2][3]]
    };
    // src_vox (voxel coords of phi^-1(x)) -> world
    let vox_to_world = |p: [f32; 3]| -> [f64; 3] {
        let (xf, yf, zf) = (p[0] as f64, p[1] as f64, p[2] as f64);
        [w[0][0]*xf + w[0][1]*yf + w[0][2]*zf + w[0][3],
         w[1][0]*xf + w[1][1]*yf + w[1][2]*zf + w[1][3],
         w[2][0]*xf + w[2][1]*yf + w[2][2]*zf + w[2][3]]
    };
    let mut fwd = vec![0.0f32; nvox * 3];
    let mut inv = vec![0.0f32; nvox * 3];
    for v in 0..nvox {
        fwd[v*3..v*3+3].copy_from_slice(&field.disp[v]);   // phi(r)-r in RAS
        let sw = vox_to_world(field.src_vox[v]);            // phi^-1(x) in world
        let x = world(v);
        for c in 0..3 { inv[v*3 + c] = (sw[c] - x[c]) as f32; }
    }
    io::write_4d(&PathBuf::from(format!("{}_desc-fwd_disp.nii.gz", cli.out)), grid.dims, 3, &fwd, &grid)?;
    io::write_4d(&PathBuf::from(format!("{}_desc-inv_disp.nii.gz", cli.out)), grid.dims, 3, &inv, &grid)?;
    let graddev = field.graddev_volumes(&grid);
    io::write_4d(&PathBuf::from(format!("{}_desc-gnl_graddev.nii.gz", cli.out)), grid.dims, 9, &graddev, &grid)?;
    println!("wrote {out}_coeff.grad, {out}_desc-{{fwd,inv}}_disp.nii.gz (RAS mm, 3 vols), {out}_desc-gnl_graddev.nii.gz (9 vols)",
        out = cli.out);
    Ok(())
}
