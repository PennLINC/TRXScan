//! The oversampled k-space stage must not move the object: a blob centred on acquired voxel
//! (X0, Y0) -- i.e. at sim index o*X0 + (o-1)/2 -- must come back centred on (X0, Y0).
use trxscan::kspace::{simulate_slice, Acquisition, SliceInput, T2Slice};

fn centroid(o: usize, nx: usize, ny: usize, x0: f64, y0: f64) -> (f64, f64) {
    let (snx, sny) = (nx * o, ny * o);
    let off = (o as f64 - 1.0) / 2.0;
    let (sx0, sy0) = (o as f64 * x0 + off, o as f64 * y0 + off);
    let mut img = vec![0.0f32; snx * sny];
    for y in 0..sny {
        for x in 0..snx {
            let d2 = (x as f64 - sx0).powi(2) + (y as f64 - sy0).powi(2);
            img[x + snx * y] = (-d2 / (2.0 * (1.5 * o as f64).powi(2))).exp() as f32;
        }
    }
    let comps: [&[f32]; 1] = [&img];
    let fmap = vec![0.0f32; snx * sny];
    let acq = Acquisition { do_distortions: false, do_relaxation: false, ..Acquisition::default() };
    let out = simulate_slice(
        &SliceInput { compartments: &comps, t2: &[T2Slice::Uniform(80.0)], t_inhom: None, fmap: &fmap, phase0: None, sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1, eddy_drive: None, prep_drive: None, slice_seed: 0, eddy_lin: None },
        &acq,
    );
    let (mut sx, mut sy, mut sw) = (0.0, 0.0, 0.0);
    for y in 0..ny {
        for x in 0..nx {
            let (re, im) = out[x + nx * y];
            let m = ((re * re + im * im) as f64).sqrt();
            sx += m * x as f64; sy += m * y as f64; sw += m;
        }
    }
    (sx / sw, sy / sw)
}

#[test]
fn oversampled_reconstruction_keeps_the_object_in_place() {
    // Even and odd acquired matrices: with an odd matrix the sim grid's `snx/2` is not the image
    // of the acquired centre `nx/2`, which used to shift the object by (o-1)/o... of a voxel.
    for (nx, ny) in [(24usize, 20usize), (25, 21), (107, 151)] {
        for o in [1usize, 2, 4] {
            if nx * o > 200 { continue; }
            let (cx, cy) = centroid(o, nx, ny, 9.0, 7.0);
            println!("{nx}x{ny} o={o}: centroid ({cx:.3}, {cy:.3}) expected (9, 7)");
            assert!((cx - 9.0).abs() < 0.05 && (cy - 7.0).abs() < 0.05, "{nx}x{ny} o={o}: centroid ({cx:.3}, {cy:.3})");
        }
    }
}
