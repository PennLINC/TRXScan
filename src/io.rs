//! I/O (feature `io`): streamlines in, tissue maps in, 4D DWI out.
//!
//! - **Streamlines**: `trx-rs` `read_tractogram` (TRX/TRK/TCK/VTK) → positions + CSR offsets.
//! - **Tissue maps**: `nifti` 0.17 → f32 volume + voxel→world affine (sform, else qform quaternion,
//!   mirroring `TRXViz/trxviz-core/src/data/nifti_data.rs`).
//! - **4D DWI out**: NIfTI-1 with the acquisition affine + FSL `.bval`/`.bvec`.
//!
//! Streamlines and tissue maps must share the same **world (RAS mm)** frame (they do when the TRK
//! is world-space and the NIfTI affine is scanner/RAS — the qsiprep-testdata pipeline's convention).

use crate::compartments::{CleanDwi, TissueFractions};
use crate::raster::Grid;
use crate::scheme::GradientScheme;

use nalgebra::Matrix4;
use ndarray::{Array4, Ix3};
use nifti::writer::WriterOptions;
use nifti::{IntoNdArray, NiftiHeader, NiftiObject, ReaderOptions, XForm};
use std::error::Error;
use std::path::{Path, PathBuf};

type R<T> = Result<T, Box<dyn Error>>;

/// Load a tractogram (TRX/TRK/TCK/VTK) as CSR: flat world-mm positions + per-streamline offsets.
pub fn load_streamlines(path: &Path) -> R<(Vec<[f64; 3]>, Vec<u32>)> {
    let t = trx_rs::read_tractogram(path, &trx_rs::ConversionOptions::default())?;
    let pos = t.positions().iter().map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect();
    Ok((pos, t.offsets().to_vec()))
}

/// voxel→world affine from a NIfTI header: sform if active, else qform quaternion.
fn affine_from_header(h: &NiftiHeader) -> [[f64; 4]; 4] {
    if h.sform_code > 0 {
        let (sx, sy, sz) = (h.srow_x, h.srow_y, h.srow_z);
        [
            [sx[0] as f64, sx[1] as f64, sx[2] as f64, sx[3] as f64],
            [sy[0] as f64, sy[1] as f64, sy[2] as f64, sy[3] as f64],
            [sz[0] as f64, sz[1] as f64, sz[2] as f64, sz[3] as f64],
            [0.0, 0.0, 0.0, 1.0],
        ]
    } else {
        let qfac = if h.pixdim[0] < 0.0 { -1.0 } else { 1.0 };
        quatern_to_mat44(
            h.quatern_b as f64, h.quatern_c as f64, h.quatern_d as f64,
            h.quatern_x as f64, h.quatern_y as f64, h.quatern_z as f64,
            h.pixdim[1] as f64, h.pixdim[2] as f64, h.pixdim[3] as f64, qfac,
        )
    }
}

/// NIfTI-1 `quatern_to_mat44` (voxel→world from the qform quaternion + pixdims).
fn quatern_to_mat44(
    b: f64, c: f64, d: f64, qx: f64, qy: f64, qz: f64, dx: f64, dy: f64, dz: f64, qfac: f64,
) -> [[f64; 4]; 4] {
    let mut a = 1.0 - (b * b + c * c + d * d);
    let (mut b, mut c, mut d) = (b, c, d);
    if a < 1e-7 {
        let n = (b * b + c * c + d * d).sqrt();
        b /= n;
        c /= n;
        d /= n;
        a = 0.0;
    } else {
        a = a.sqrt();
    }
    let dz = dz * qfac;
    [
        [(a * a + b * b - c * c - d * d) * dx, 2.0 * (b * c - a * d) * dy, 2.0 * (b * d + a * c) * dz, qx],
        [2.0 * (b * c + a * d) * dx, (a * a + c * c - b * b - d * d) * dy, 2.0 * (c * d - a * b) * dz, qy],
        [2.0 * (b * d - a * c) * dx, 2.0 * (c * d + a * b) * dy, (a * a + d * d - c * c - b * b) * dz, qz],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Read a 3D NIfTI scalar volume as a flat `x + nx*(y + ny*z)` f32 buffer + its grid.
pub fn load_volume(path: &Path) -> R<(Vec<f32>, Grid)> {
    let obj = ReaderOptions::new().read_file(path)?;
    let header = obj.header().clone();
    let dims = [header.dim[1] as usize, header.dim[2] as usize, header.dim[3] as usize];
    let aff = affine_from_header(&header);
    let arr = obj.into_volume().into_ndarray::<f32>()?.into_dimensionality::<Ix3>()?;
    let [nx, ny, nz] = dims;
    let mut flat = vec![0.0f32; nx * ny * nz];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                flat[x + nx * (y + ny * z)] = arr[[x, y, z]];
            }
        }
    }
    Ok((flat, Grid { dims, voxel_to_world: aff }))
}

/// Load WM/GM/CSF fraction maps + a brain mask onto a shared grid (dims/affine must match).
pub fn load_tissue(wm: &Path, gm: &Path, csf: &Path, mask: &Path) -> R<(TissueFractions, Grid)> {
    let (wm, grid) = load_volume(wm)?;
    let (gm, g2) = load_volume(gm)?;
    let (csf, g3) = load_volume(csf)?;
    let (maskf, g4) = load_volume(mask)?;
    for g in [&g2, &g3, &g4] {
        if g.dims != grid.dims {
            return Err(format!("tissue map grid mismatch: {:?} vs {:?}", g.dims, grid.dims).into());
        }
    }
    let mask = maskf.iter().map(|&m| (m > 0.5) as u8).collect();
    let dims = grid.dims;
    Ok((TissueFractions { dims, wm, gm, csf, mask }, grid))
}

fn header_for_grid(v: [[f64; 4]; 4]) -> NiftiHeader {
    let mut h = NiftiHeader::default();
    let col_norm = |c: usize| (v[0][c].powi(2) + v[1][c].powi(2) + v[2][c].powi(2)).sqrt() as f32;
    h.pixdim = [1.0, col_norm(0), col_norm(1), col_norm(2), 0.0, 0.0, 0.0, 0.0];
    h.xyzt_units = 2; // mm
    let m = Matrix4::<f64>::new(
        v[0][0], v[0][1], v[0][2], v[0][3],
        v[1][0], v[1][1], v[1][2], v[1][3],
        v[2][0], v[2][1], v[2][2], v[2][3],
        v[3][0], v[3][1], v[3][2], v[3][3],
    );
    h.set_qform(&m, XForm::ScannerAnat);
    h.set_sform(&m, XForm::ScannerAnat);
    h
}

/// Write a 4D `[x,y,z,g]` (layout `(x+nx*(y+ny*z))*ngrad+g`) as NIfTI-1 with the given affine.
fn write_4d(path: &Path, dims: [usize; 3], ngrad: usize, data: &[f32], grid: &Grid) -> R<()> {
    let [nx, ny, nz] = dims;
    let arr = Array4::from_shape_fn((nx, ny, nz, ngrad), |(x, y, z, g)| {
        data[(x + nx * (y + ny * z)) * ngrad + g]
    });
    let hdr = header_for_grid(grid.voxel_to_world);
    WriterOptions::new(path).reference_header(&hdr).write_nifti(&arr)?;
    Ok(())
}

fn write_bval_bvec(bval: &Path, bvec: &Path, scheme: &GradientScheme) -> R<()> {
    let line = scheme.bvals.iter().map(|b| format!("{b}")).collect::<Vec<_>>().join(" ");
    std::fs::write(bval, format!("{line}\n"))?;
    let mut v = String::new();
    for axis in 0..3 {
        let row = scheme.bvecs.iter().map(|d| format!("{:.6}", d[axis])).collect::<Vec<_>>().join(" ");
        v.push_str(&row);
        v.push('\n');
    }
    std::fs::write(bvec, v)?;
    Ok(())
}

/// Write a magnitude 4D DWI series + FSL `.bval`/`.bvec`.
pub fn write_dwi(out_prefix: &Path, dwi: &CleanDwi, grid: &Grid, scheme: &GradientScheme) -> R<()> {
    write_4d(&out_prefix.with_extension("nii.gz"), grid.dims, dwi.ngrad, &dwi.data, grid)?;
    write_bval_bvec(&out_prefix.with_extension("bval"), &out_prefix.with_extension("bvec"), scheme)
}

/// Write a **complex** DWI as BIDS `part-mag` / `part-phase` NIfTIs (phase in radians) + shared
/// `.bval`/`.bvec` + JSON sidecars — the layout `dwidenoise`/`dwidenoise2` consume for complex
/// denoising. `out_prefix` is a BIDS stem, e.g. `.../sub-01_ses-V02_dir-AP_run-01`.
pub fn write_complex_dwi(
    out_prefix: &str,
    dims: [usize; 3],
    ngrad: usize,
    mag: &[f32],
    phase: &[f32],
    grid: &Grid,
    scheme: &GradientScheme,
) -> R<()> {
    let p = |s: &str| PathBuf::from(format!("{out_prefix}{s}"));
    write_4d(&p("_part-mag_dwi.nii.gz"), dims, ngrad, mag, grid)?;
    write_4d(&p("_part-phase_dwi.nii.gz"), dims, ngrad, phase, grid)?;
    write_bval_bvec(&p("_dwi.bval"), &p("_dwi.bvec"), scheme)?;
    std::fs::write(p("_part-mag_dwi.json"),
        "{\n  \"Manufacturer\": \"TRXScan\",\n  \"ImageComparison\": \"magnitude\"\n}\n")?;
    std::fs::write(p("_part-phase_dwi.json"),
        "{\n  \"Manufacturer\": \"TRXScan\",\n  \"ImageComparison\": \"phase\",\n  \"Units\": \"rad\"\n}\n")?;
    Ok(())
}
