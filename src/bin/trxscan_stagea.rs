//! Stage-A profiling harness: streamlines + tissue maps + scheme → clean 4D DWI.
//!
//! Usage:
//!   sp-stage-a <wm> <gm> <csf> <mask> <streamlines.trk|.trx> <bval> <bvec> <out_prefix>
//!
//! Prints timing and a few sanity stats so we can profile and diff against Fiberfox.

use std::path::Path;
use std::time::Instant;
use trxscan::compartments::{generate_clean_signal, CompartmentParams};
use trxscan::io;
use trxscan::scheme::GradientScheme;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 9 {
        eprintln!("usage: sp-stage-a <wm> <gm> <csf> <mask> <streamlines> <bval> <bvec> <out_prefix>");
        std::process::exit(2);
    }
    let p = |i: usize| Path::new(&a[i]);

    let t = Instant::now();
    let (tissue, grid) = io::load_tissue(p(1), p(2), p(3), p(4))?;
    let brain = tissue.mask.iter().filter(|&&m| m > 0).count();
    println!("tissue: grid {:?}  brain voxels {}  [{:?}]", grid.dims, brain, t.elapsed());

    let t = Instant::now();
    let (positions, offsets) = io::load_streamlines(p(5))?;
    let n_streamlines = offsets.len().saturating_sub(1);
    println!("streamlines: {} lines, {} points  [{:?}]", n_streamlines, positions.len(), t.elapsed());

    let scheme = GradientScheme::from_fsl(p(6), p(7))?;
    println!("scheme: {} volumes, shells {:?}", scheme.len(), scheme.shells(50.0));

    let params = CompartmentParams { b_value: scheme.b_max, ..Default::default() };
    let t = Instant::now();
    let dwi = generate_clean_signal(&grid, &positions, &offsets, &tissue, &scheme, &params);
    let dt = t.elapsed();

    let nz = dwi.data.iter().filter(|&&v| v > 0.0).count();
    let b0_mean: f64 = {
        // mean of the first b0 volume over brain voxels
        let g0 = (0..scheme.len()).find(|&g| scheme.is_b0(g)).unwrap_or(0);
        let (mut s, mut n) = (0.0, 0usize);
        for vox in 0..grid.dims.iter().product() {
            if tissue.mask[vox] > 0 {
                s += dwi.data[vox * dwi.ngrad + g0] as f64;
                n += 1;
            }
        }
        if n > 0 { s / n as f64 } else { 0.0 }
    };
    println!(
        "SIGNAL: shape {:?}x{}  nonzero {}  brain-b0 mean {:.3}  [{:?}]",
        dwi.dims, dwi.ngrad, nz, b0_mean, dt
    );

    let t = Instant::now();
    io::write_dwi(Path::new(&a[8]), &dwi, &grid, &scheme)?;
    println!("wrote {}.nii.gz (+bval/bvec)  [{:?}]", a[8], t.elapsed());
    Ok(())
}
