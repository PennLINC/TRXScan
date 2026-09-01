//! Emit Gibbs-benchmark fixture sets across the acceptance-suite factor grid (spec 5.2).
//!
//! Each grid point produces the four co-registered images from ONE object and phase realization,
//! in the canonical Gibbs-only configuration: distortion, T2*, eddy, ghosts, spikes, multi-coil
//! and GRAPPA are all off, so that `unringed - object-nominal` is unringing error and not
//! geometric error. Partial Fourier, noise and windowing stay on as legitimate factors.

use std::path::PathBuf;
use trxscan::benchmark::{
    factor_grid, gibbs_benchmark_acquisition, phase_model_for, produce_slice, PhaseKind,
};
use trxscan::io;
use trxscan::kspace::{box_hires, Acquisition, PartialFourierMode};
use trxscan::raster::Grid;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!(
            "usage: trxscan-benchmark <out_dir> [matrix=64] [oversample=4] [slices=1] [only=<label>]"
        );
        std::process::exit(2);
    }
    let out = PathBuf::from(&a[1]);
    let n: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let o: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let nz: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let only = a.get(5).cloned();

    let (snx, sny) = (n * o, n * o);
    // A centred rectangle with sub-voxel-positioned edges on ALL FOUR sides. Edges along both
    // axes matter: partial Fourier acts only along phase-encode, so a readout-only step would
    // leave the whole PF axis of the factor grid inert. The +0.5 offsets put the first sample
    // near the continuous overshoot maximum.
    let q = n as f64 / 4.0;
    let img = box_hires(
        snx, sny,
        (q + 0.5) * o as f64, (3.0 * q + 0.5) * o as f64,
        (q + 0.5) * o as f64, (3.0 * q + 0.5) * o as f64,
    );
    let fmap = vec![0.0f32; snx * sny];
    let grid = Grid {
        dims: [n, n, nz],
        voxel_to_world: [
            [1.7, 0.0, 0.0, 0.0],
            [0.0, 1.7, 0.0, 0.0],
            [0.0, 0.0, 1.7, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    let grid_points = factor_grid();
    let mut written = 0usize;
    for p in &grid_points {
        if let Some(f) = &only {
            if &p.label != f {
                continue;
            }
        }
        let dir = out.join(&p.label);
        std::fs::create_dir_all(&dir)?;

        let acq = gibbs_benchmark_acquisition(&Acquisition {
            signal_scale: 1.0,
            partial_fourier: p.partial_fourier,
            // Scanner-like CONTIGUOUS partial Fourier, not the Fiberfox default. The fixtures are
            // named pf68/pf78 and the whole point of the PF axis is to evaluate PF-aware
            // reconstruction, so they must actually be 6/8 and 7/8 rather than the Fiberfox rule's
            // 78.1%/90.6% with a detached line-zero.
            pf_mode: PartialFourierMode::Contiguous,
            window: p.window,
            // Per-component image-space variance at full sampling; ~30 dB against a unit step.
            // One nonzero level for every fixture; produce_slice emits the clean/noisy pair.
            noise_variance: 1.0e-3,
            ..Acquisition::default()
        });
        let model = phase_model_for(p.phase);
        // b and direction only matter for the Diffusion condition; fixed elsewhere for determinism.
        let (bval, bvec) = if p.phase == PhaseKind::Diffusion {
            (2000.0, [1.0, 0.0, 0.0])
        } else {
            (0.0, [0.0, 0.0, 0.0])
        };

        let comps: [&[f32]; 1] = [&img];
        let slices: Vec<_> = (0..nz)
            .map(|z| {
                let shot = model.diffusion.shot(bval, bvec, 0, z, 0xB0A7);
                produce_slice(
                    &comps, &[100.0], &fmap, &model, &shot,
                    [snx, sny], [n, n], z, nz, &acq, bvec, bval,
                    (z as u64).wrapping_mul(0x9E37) ^ 0x51E,
                )
            })
            .collect();

        io::write_benchmark(&dir.join("bench"), [n, n, nz], o, &slices, &grid)?;
        std::fs::write(
            dir.join("factors.json"),
            format!(
                "{{\n  \"label\": \"{}\",\n  \"partial_fourier\": {},\n  \"phase\": \"{:?}\",\n  \
                 \"window\": \"{:?}\",\n  \"pf_mode\": \"Contiguous\",\n  \"matrix\": {},\n  \"oversample\": {},\n  \
                 \"slices\": {},\n  \"bval\": {}\n}}\n",
                p.label, p.partial_fourier, p.phase, p.window, n, o, nz, bval
            ),
        )?;
        written += 1;
    }
    println!("wrote {written} fixture set(s) under {}", out.display());
    Ok(())
}
