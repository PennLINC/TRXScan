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
    let (pos, offsets, _) = load_streamlines_weighted(path, None)?;
    Ok((pos, offsets))
}

/// As [`load_streamlines`], plus an optional **per-streamline weight** array taken from the
/// tractogram's data-per-streamline (dps) fields — SIFT2 weights, or the `fiberWeight` this port
/// dropped from Fiberfox. Feed the result to
/// [`crate::compartments::generate_mixture`].
///
/// `dps_name == None` ⇒ `Ok((.., None))`, i.e. every streamline weighs 1: a TRK with no
/// properties, or a TRX with no dps, simply loads unweighted. `Some(name)` is an assertion that
/// the field is there — a missing name (or a length that disagrees with the streamline count) is
/// an error, not a silent fallback.
pub fn load_streamlines_weighted(
    path: &Path,
    dps_name: Option<&str>,
) -> R<(Vec<[f64; 3]>, Vec<u32>, Option<Vec<f32>>)> {
    let t = trx_rs::read_tractogram(path, &trx_rs::ConversionOptions::default())?;
    let pos: Vec<[f64; 3]> =
        t.positions().iter().map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect();
    let offsets = t.offsets().to_vec();
    let n = offsets.len().saturating_sub(1);
    let weights = match dps_name {
        None => None,
        Some(name) => {
            let arr = t.dps_arrays().get(name).ok_or_else(|| {
                let mut have: Vec<&str> = t.dps_names().collect();
                have.sort_unstable();
                let have = if have.is_empty() { "(none)".to_string() } else { have.join(", ") };
                format!("dps field {name:?} not in {}; available dps: {have}", path.display())
            })?;
            let w = dps_as_f32(arr, name)?;
            if w.len() != n {
                return Err(format!(
                    "dps field {name:?} has {} values but the tractogram has {n} streamlines",
                    w.len()
                )
                .into());
            }
            Some(w)
        }
    };
    Ok((pos, offsets, weights))
}

/// Resolve a CLI weight spec: `None` ⇒ unweighted; a spec naming an **existing file** ⇒ MRtrix
/// text weights ([`read_weights_txt`], count-checked); anything else ⇒ a TRX dps field name
/// ([`load_streamlines_weighted`]). The one loader both binaries share.
pub fn load_streamlines_spec(
    path: &Path,
    weight_spec: Option<&str>,
) -> R<(Vec<[f64; 3]>, Vec<u32>, Option<Vec<f32>>)> {
    let txt = weight_spec.filter(|w| Path::new(w).exists());
    let (pos, offsets, mut weights) =
        load_streamlines_weighted(path, weight_spec.filter(|_| txt.is_none()))?;
    if let Some(t) = txt {
        let w = read_weights_txt(Path::new(t))?;
        let n = offsets.len().saturating_sub(1);
        if w.len() != n {
            return Err(format!(
                "weights file {t} has {} values but the tractogram has {n} streamlines",
                w.len()
            )
            .into());
        }
        weights = Some(w);
    }
    Ok((pos, offsets, weights))
}

/// Read MRtrix-style per-streamline weights from a text file (`tcksift2` output): lines starting
/// with `#` are comments, everything else is whitespace-separated floats, one weight per
/// streamline in tractogram order. The TCK + weights.txt pair is the MRtrix-ecosystem equivalent
/// of a TRX dps field; validate the count against the tractogram at the call site.
pub fn read_weights_txt(path: &Path) -> R<Vec<f32>> {
    let text = std::fs::read_to_string(path)?;
    let mut w = Vec::new();
    for line in text.lines() {
        if line.trim_start().starts_with('#') {
            continue;
        }
        for tok in line.split_whitespace() {
            w.push(tok.parse::<f32>().map_err(|_| {
                format!("weights file {}: bad float {tok:?}", path.display())
            })?);
        }
    }
    if w.is_empty() {
        return Err(format!("weights file {} has no values", path.display()).into());
    }
    Ok(w)
}

/// One scalar dps array → per-streamline f32.
///
/// Read through the raw bytes rather than `DataArray::cast_slice`: [`trx_rs::Tractogram`] holds
/// dps in an owned `Vec<u8>`, which carries no alignment guarantee for f32/f64, and bytemuck
/// *panics* on a misaligned cast. TRX stores little-endian, which is also what `cast_slice` would
/// have assumed.
fn dps_as_f32(arr: &trx_rs::DataArray, name: &str) -> R<Vec<f32>> {
    use trx_rs::DType;
    if arr.ncols() != 1 {
        return Err(format!("dps field {name:?} has {} columns; expected a scalar", arr.ncols()).into());
    }
    let b = arr.as_bytes();
    Ok(match arr.dtype() {
        DType::Float16 => b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        DType::Float32 => {
            b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
        DType::Float64 => b
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        other => {
            return Err(format!("dps field {name:?} has dtype {other}; expected a float field").into())
        }
    })
}

/// IEEE-754 binary16 → f32. TRX's f16 arrays are common and `half` is not a dependency here
/// (trx-rs doesn't re-export it), so decode the bit pattern directly.
fn f16_to_f32(h: u16) -> f32 {
    let (sign, exp, mant) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let bits = match exp {
        0 if mant == 0 => sign << 31,                                     // ±0
        0 => {
            // subnormal: renormalize so the leading 1 becomes implicit
            let shift = mant.leading_zeros() - 21; // 1..=10
            (sign << 31) | ((113 - shift) << 23) | ((mant << (13 + shift)) & 0x7f_ffff)
        }
        0x1f => (sign << 31) | 0x7f80_0000 | (mant << 13),                // ±inf / NaN
        _ => (sign << 31) | ((exp + 112) << 23) | (mant << 13),           // bias 15 → 127
    };
    f32::from_bits(bits)
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

/// Write one 3D scalar volume (layout `x + nx*(y + ny*z)`) as NIfTI-1 with the given affine.
fn write_3d(path: &Path, dims: [usize; 3], data: &[f32], grid: &Grid) -> R<()> {
    let [nx, ny, nz] = dims;
    let arr =
        ndarray::Array3::from_shape_fn((nx, ny, nz), |(x, y, z)| data[x + nx * (y + ny * z)]);
    let hdr = header_for_grid(grid.voxel_to_world);
    WriterOptions::new(path).reference_header(&hdr).write_nifti(&arr)?;
    Ok(())
}

/// Write named 3D scalar maps as `<out_prefix>_<name>.nii.gz` siblings (the cs-odf
/// `--microstructure-nifti` convention). `maps[i]` pairs with `names[i]`.
pub fn write_scalar_maps(
    out_prefix: &str,
    dims: [usize; 3],
    names: &[&str],
    maps: &[Vec<f32>],
    grid: &Grid,
) -> R<()> {
    for (name, map) in names.iter().zip(maps) {
        write_3d(&PathBuf::from(format!("{out_prefix}_{name}.nii.gz")), dims, map, grid)?;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_decode_matches_known_bit_patterns() {
        for (bits, want) in [
            (0x0000u16, 0.0f32), (0x8000, -0.0), (0x3c00, 1.0), (0xbc00, -1.0),
            (0x3555, 0.333_251_95), (0x4900, 10.0), (0x7bff, 65504.0), (0x0400, 6.103_515_6e-5),
            (0x0001, 5.960_464_5e-8), (0x03ff, 6.097_555_2e-5), // smallest / largest subnormal
        ] {
            let got = f16_to_f32(bits);
            assert!((got - want).abs() <= 1e-12 * want.abs().max(1e-7), "{bits:#06x}: {got} vs {want}");
        }
        assert!(f16_to_f32(0x7c00).is_infinite() && f16_to_f32(0x7c00) > 0.0);
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    /// Round-trip an in-memory tractogram with a dps array through a temp TRX and read the
    /// weights back. No network, no fixture file.
    #[test]
    fn dps_weights_round_trip_through_a_trx() {
        use trx_rs::dtype::DType;
        use trx_rs::mmap_backing::vec_to_bytes;
        use trx_rs::{DataArray, Tractogram};

        let mut t = Tractogram::new();
        t.push_streamline(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).unwrap();
        t.push_streamline(&[[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]).unwrap();
        t.insert_dps("weights", DataArray::owned_bytes(vec_to_bytes(vec![2.5f32, 0.5]), 1, DType::Float32));
        t.insert_dps("w64", DataArray::owned_bytes(vec_to_bytes(vec![1.25f64, 4.0]), 1, DType::Float64));

        let dir = std::env::temp_dir().join(format!("trxscan_dps_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weighted.trx");
        trx_rs::write_tractogram(&path, &t, &trx_rs::ConversionOptions::default()).unwrap();

        let (pos, offsets, w) = load_streamlines_weighted(&path, Some("weights")).unwrap();
        assert_eq!(pos.len(), 5);
        assert_eq!(offsets, vec![0, 2, 5]);
        assert_eq!(w.unwrap(), vec![2.5f32, 0.5]);
        assert_eq!(load_streamlines_weighted(&path, Some("w64")).unwrap().2.unwrap(), vec![1.25f32, 4.0]);
        // no dps requested → unweighted, and the plain loader still agrees
        assert!(load_streamlines_weighted(&path, None).unwrap().2.is_none());
        assert_eq!(load_streamlines(&path).unwrap().1, vec![0, 2, 5]);
        // a missing field is an error that lists what is actually there
        let e = load_streamlines_weighted(&path, Some("nope")).unwrap_err().to_string();
        assert!(e.contains("nope") && e.contains("weights"), "unhelpful error: {e}");

        std::fs::remove_dir_all(&dir).ok();
    }
}
