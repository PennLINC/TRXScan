//! I/O (feature `io`): streamlines in, tissue maps in, 4D DWI out.
//!
//! - **Streamlines**: `trx-rs` `read_tractogram` (TRX/TRK/TCK/VTK) → positions + CSR offsets.
//! - **Tissue maps**: `nifti` 0.17 → f32 volume + voxel→world affine (sform, else qform quaternion,
//!   mirroring `TRXViz/trxviz-core/src/data/nifti_data.rs`).
//! - **4D DWI out**: NIfTI-1 with the acquisition affine (`mrsim_acq::io::write_complex_4d`)
//!   + FSL `.bval`/`.bvec`.
//!
//! Streamlines and tissue maps must share the same **world (RAS mm)** frame (they do when the TRK
//! is world-space and the NIfTI affine is scanner/RAS — the qsiprep-testdata pipeline's convention).

use crate::compartments::{CleanDwi, TissueFractions};
use crate::raster::Grid;
use crate::scheme::GradientScheme;

use std::error::Error;
use std::path::{Path, PathBuf};

// NIfTI volume read and array write live in mrsim-acq (the acquisition stage shared with aslscan);
// re-exported so `io::` paths inside and outside this crate keep resolving.
pub use mrsim_acq::io::{hires_grid, load_volume, write_3d, write_3d_i16, write_4d};

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

/// Write the four benchmark images as BIDS-style magnitude/phase pairs.
///
/// `object-hires` uses [`hires_grid`]; the other three use the acquisition grid. Emitting the
/// clean/noisy pair separately is deliberate: their difference is noise alone, which is what
/// separates residual Gibbs from noise amplification when scoring.
pub fn write_benchmark(
    out_prefix: &Path,
    dims: [usize; 3],
    o: usize,
    slices: &[crate::benchmark::BenchmarkSlice],
    grid: &Grid,
) -> R<()> {
    let [nx, ny, nz] = dims;
    assert_eq!(slices.len(), nz, "expected one BenchmarkSlice per slice");
    let hg = hires_grid(grid, o);
    let (hnx, hny) = (nx * o, ny * o);

    // (name, per-slice complex accessor as (re, im) f32, dims, grid)
    let write_pair = |name: &str,
                          data: Vec<(f32, f32)>,
                          d: [usize; 3],
                          g: &Grid|
     -> R<()> {
        let n = d[0] * d[1] * d[2];
        assert_eq!(data.len(), n);
        let mag: Vec<f32> = data.iter().map(|p| (p.0 * p.0 + p.1 * p.1).sqrt()).collect();
        let ph: Vec<f32> = data.iter().map(|p| p.1.atan2(p.0)).collect();
        let mp = out_prefix.with_file_name(format!(
            "{}_desc-{name}_part-mag_dwi.nii.gz",
            out_prefix.file_name().unwrap().to_string_lossy()
        ));
        let pp = out_prefix.with_file_name(format!(
            "{}_desc-{name}_part-phase_dwi.nii.gz",
            out_prefix.file_name().unwrap().to_string_lossy()
        ));
        write_4d(&mp, d, 1, &mag, g)?;
        write_4d(&pp, d, 1, &ph, g)?;
        Ok(())
    };

    let gather_f64 = |pick: &dyn Fn(&crate::benchmark::BenchmarkSlice) -> &Vec<(f64, f64)>,
                      w: usize,
                      h: usize|
     -> Vec<(f32, f32)> {
        let mut v = vec![(0.0f32, 0.0f32); w * h * nz];
        for (z, s) in slices.iter().enumerate() {
            let src = pick(s);
            for i in 0..w * h {
                v[i + w * h * z] = (src[i].0 as f32, src[i].1 as f32);
            }
        }
        v
    };
    let gather_f32 = |pick: &dyn Fn(&crate::benchmark::BenchmarkSlice) -> &Vec<(f32, f32)>|
     -> Vec<(f32, f32)> {
        let mut v = vec![(0.0f32, 0.0f32); nx * ny * nz];
        for (z, s) in slices.iter().enumerate() {
            let src = pick(s);
            for i in 0..nx * ny {
                v[i + nx * ny * z] = src[i];
            }
        }
        v
    };

    write_pair("objecthires", gather_f64(&|s| &s.object_hires, hnx, hny), [hnx, hny, nz], &hg)?;
    write_pair("objectnominal", gather_f64(&|s| &s.object_nominal, nx, ny), dims, grid)?;
    write_pair("acquiredclean", gather_f32(&|s| &s.acquired_clean), dims, grid)?;
    write_pair("acquirednoisy", gather_f32(&|s| &s.acquired_noisy), dims, grid)?;
    Ok(())
}

/// Write a synthesized GRE fieldmap as BIDS `<prefix>_magnitude{1,2}` + `_phasediff` (or
/// `_phase{1,2}`) NIfTIs with sidecars carrying the echo times and `B0FieldIdentifier`. With
/// `fsl_orientation` the volumes are written radiological LAS on the GRE's own grid (which may
/// differ from the DWI grid when a resolution was requested).
pub fn write_gre_fieldmap(prefix: &str, gre: &crate::gre::GreFieldmap, fsl_orientation: bool) -> R<()> {
    use crate::gre::GreOutput;
    use crate::orient::Reorient;
    let greo = if fsl_orientation {
        Reorient::to_las(&gre.grid.voxel_to_world, gre.grid.dims)
    } else {
        Reorient::identity(gre.grid.dims)
    };
    let mut wgrid = gre.grid.clone();
    wgrid.voxel_to_world = greo.apply_affine(&gre.grid.voxel_to_world);
    wgrid.dims = greo.out_dims;
    let b0id = &gre.b0_field;
    let p = |s: &str| PathBuf::from(format!("{prefix}{s}"));
    for (k, m) in gre.magnitude.iter().enumerate() {
        let vol = greo.apply_volume(m, 1);
        write_3d(&p(&format!("_magnitude{}.nii.gz", k + 1)), wgrid.dims, &vol, &wgrid)?;
        let te = gre.te_s[k];
        std::fs::write(p(&format!("_magnitude{}.json", k + 1)), format!(
            "{{\n  \"Manufacturer\": \"TRXScan\",\n  \"EchoTime\": {te:.5},\n  \"B0FieldIdentifier\": \"{b0id}\",\n  \"ImageType\": [\"ORIGINAL\", \"PRIMARY\", \"M\", \"ND\"]\n}}\n"))?;
    }
    let write_phase = |suffix: &str, data: &[i16], json: String| -> R<()> {
        let ph_f: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let ph_reo: Vec<i16> = greo.apply_volume(&ph_f, 1).iter().map(|&v| v as i16).collect();
        write_3d_i16(&p(&format!("_{suffix}.nii.gz")), wgrid.dims, &ph_reo, &wgrid)?;
        std::fs::write(p(&format!("_{suffix}.json")), json)?;
        Ok(())
    };
    match gre.output {
        GreOutput::Phase => {
            for (k, ph) in gre.phase.iter().enumerate() {
                let te = gre.te_s[k];
                write_phase(&format!("phase{}", k + 1), ph, format!(
                    "{{\n  \"Manufacturer\": \"TRXScan\",\n  \"EchoTime\": {te:.5},\n  \"B0FieldIdentifier\": \"{b0id}\",\n  \"ImageType\": [\"ORIGINAL\", \"PRIMARY\", \"P\", \"ND\", \"PHASE\"]\n}}\n"))?;
            }
        }
        GreOutput::Phasediff => {
            let [te1, te2] = gre.te_s;
            write_phase("phasediff", &gre.phase[0], format!(
                "{{\n  \"Manufacturer\": \"TRXScan\",\n  \"EchoTime1\": {te1:.5},\n  \"EchoTime2\": {te2:.5},\n  \"B0FieldIdentifier\": \"{b0id}\",\n  \"ImageType\": [\"ORIGINAL\", \"PRIMARY\", \"P\", \"ND\", \"PHASE\"]\n}}\n"))?;
        }
    }
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
/// Acquisition facts the BIDS JSON sidecars record. A plain struct (not
/// `kspace::Acquisition`) so `io` stays usable without the `kspace` feature.
pub struct SidecarInfo {
    /// BIDS PhaseEncodingDirection ("j", "j-", …) already resolved for the *written* voxel frame
    /// (see `orient` + the caller): a +off-resonance field displaces signal toward +this-axis.
    pub phase_encoding_direction: String,
    pub total_readout_time: f64, // s
    pub echo_time: f64,          // s
    pub partial_fourier: f64,
    pub accel: usize,
    pub mb: usize,
    /// Modern BIDS B0 linkage: when a fieldmap is written for this DWI, its
    /// `B0FieldIdentifier` label is recorded here so the DWI carries the matching
    /// `B0FieldSource` (the replacement for the deprecated `IntendedFor`).
    pub b0_field_source: Option<String>,
}

pub fn write_complex_dwi(
    out_prefix: &str,
    dims: [usize; 3],
    ngrad: usize,
    mag: &[f32],
    phase: &[f32],
    grid: &Grid,
    scheme: &GradientScheme,
    info: &SidecarInfo,
) -> R<()> {
    let shared = mrsim_acq::io::SidecarInfo {
        manufacturer: "TRXScan".to_string(),
        phase_encoding_direction: info.phase_encoding_direction.clone(),
        total_readout_time: info.total_readout_time,
        echo_time: info.echo_time,
        partial_fourier: info.partial_fourier,
        accel: info.accel,
        mb: info.mb,
        repetition_time_s: None,
        b0_field_source: info.b0_field_source.clone(),
    };
    mrsim_acq::io::write_complex_4d(out_prefix, "dwi", dims, ngrad, mag, phase, grid, &shared)?;
    let p = |s: &str| PathBuf::from(format!("{out_prefix}{s}"));
    write_bval_bvec(&p("_dwi.bval"), &p("_dwi.bvec"), scheme)
}

/// Keep `n` streamlines sampled proportionally to weight; see
/// [`crate::streamlines::subsample_streamlines`]. This wrapper prints the summary the binaries
/// have always printed.
pub fn subsample_streamlines(
    positions: Vec<[f64; 3]>,
    offsets: Vec<u32>,
    weights: Option<Vec<f32>>,
    n: usize,
    seed: u64,
) -> (Vec<[f64; 3]>, Vec<u32>, Option<Vec<f32>>) {
    let (p, o, w, stats) = crate::streamlines::subsample_streamlines(positions, offsets, weights, n, seed);
    println!("{}", stats.summary(seed));
    (p, o, w)
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

    #[test]
    fn hires_affine_preserves_the_fov() {
        // The hires volume must cover the same FOV as the nominal one, or the two references
        // cannot be compared voxel-for-voxel after block reduction.
        let g = Grid {
            dims: [8, 8, 2],
            voxel_to_world: [
                [1.7, 0.0, 0.0, -10.0],
                [0.0, 1.7, 0.0, -12.0],
                [0.0, 0.0, 1.7, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        let o = 4;
        let h = hires_grid(&g, o);
        assert_eq!(h.dims, [32, 32, 2]);
        for ax in 0..2 {
            let nom = g.dims[ax] as f64 * g.voxel_to_world[ax][ax];
            let hi = h.dims[ax] as f64 * h.voxel_to_world[ax][ax];
            assert!((nom - hi).abs() < 1e-9, "axis {ax}: FOV {nom} vs {hi}");
        }
        assert_eq!(h.voxel_to_world[2][2], g.voxel_to_world[2][2], "z must not be refined");
    }

    #[test]
    fn hires_grid_is_identity_at_o_equals_one() {
        let g = Grid {
            dims: [5, 7, 3],
            voxel_to_world: [
                [2.0, 0.0, 0.0, 1.0],
                [0.0, 2.0, 0.0, 2.0],
                [0.0, 0.0, 3.0, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        let h = hires_grid(&g, 1);
        assert_eq!(h.dims, g.dims);
        for r in 0..4 {
            for c in 0..4 {
                assert!((h.voxel_to_world[r][c] - g.voxel_to_world[r][c]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn hires_corners_coincide_with_the_nominal_grid() {
        // Same check the prep script's geometry satisfies: the outer FOV corners must land in the
        // same world position, so the two grids describe one physical volume.
        let g = Grid {
            dims: [6, 9, 2],
            voxel_to_world: [
                [1.7, 0.0, 0.0, -5.0],
                [0.0, 1.7, 0.0, -7.0],
                [0.0, 0.0, 1.7, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        for &o in &[2usize, 3, 4, 8] {
            let h = hires_grid(&g, o);
            // corner of voxel (0,0): left edge = origin - half a voxel along each in-plane axis
            for r in 0..3 {
                let nom = g.voxel_to_world[r][3]
                    - 0.5 * g.voxel_to_world[r][0]
                    - 0.5 * g.voxel_to_world[r][1];
                let hi = h.voxel_to_world[r][3]
                    - 0.5 * h.voxel_to_world[r][0]
                    - 0.5 * h.voxel_to_world[r][1];
                assert!((nom - hi).abs() < 1e-9, "o={o} row {r}: {nom} vs {hi}");
            }
        }
    }
}
