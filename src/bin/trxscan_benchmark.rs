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
use trxscan::kspace::{box_hires, default_acquisition, Acquisition, PartialFourierMode, T2Slice};
use trxscan::phase::ShotPhase;
use trxscan::raster::Grid;

const USAGE: &str = "usage: trxscan-benchmark <out_dir> \
[matrix=<n>] [oversample=<n>] [slices=<n>] [only=<label>]\n\
  defaults: matrix=64 oversample=4 slices=1, all fixture sets\n\
  arguments after <out_dir> are key=value and may appear in any order";

fn die(msg: &str) -> ! {
    eprintln!("trxscan-benchmark: {msg}\n{USAGE}");
    std::process::exit(2);
}

/// Parse `key=value` arguments.
///
/// The previous version took bare positionals while the usage string documented `key=value`, and
/// swallowed parse failures with `.parse().ok().unwrap_or(default)`. Following the documented
/// interface therefore ran a DIFFERENT configuration than requested -- `matrix=128` became 64,
/// `oversample=8` became 4 -- and printed nothing about it. For a tool whose only job is emitting
/// carefully controlled scientific fixtures, silently substituting the defaults is the worst
/// available failure mode, so every malformed input is now fatal.
fn parse_args(a: &[String]) -> (PathBuf, usize, usize, usize, Option<String>) {
    if a.len() < 2 || a[1] == "-h" || a[1] == "--help" {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    let out = PathBuf::from(&a[1]);
    let (mut n, mut o, mut nz, mut only) = (64usize, 4usize, 1usize, None);
    for arg in &a[2..] {
        let Some((key, val)) = arg.split_once('=') else {
            die(&format!("expected key=value, got {arg:?}"));
        };
        let num = |what: &str| -> usize {
            match val.parse::<usize>() {
                Ok(v) if v > 0 => v,
                _ => die(&format!("{what} must be a positive integer, got {val:?}")),
            }
        };
        match key {
            "matrix" => n = num("matrix"),
            "oversample" => o = num("oversample"),
            "slices" => nz = num("slices"),
            "only" => only = Some(val.to_string()),
            _ => die(&format!("unknown argument {key:?}")),
        }
    }
    (out, n, o, nz, only)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let (out, n, o, nz, only) = parse_args(&a);

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
            ..default_acquisition()
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
                let shot = model.prep.map_or(
                    ShotPhase { q_eff: [0.0; 3], dx: [0.0; 3], rot: [0.0; 3] },
                    |p| p.shot(bval, bvec, 0, z, 0xB0A7));
                produce_slice(
                    &comps, &[T2Slice::Uniform(100.0)], &fmap, &model, &shot,
                    [snx, sny], [n, n], z, nz, &acq,
                    if bval.abs() > 1e-9 { Some([bvec[0] * bval, bvec[1] * bval, bvec[2] * bval]) } else { None },
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
    // A filter that matches nothing used to print "wrote 0 fixture set(s)" and exit 0, so a
    // typo in `only=` was indistinguishable from a successful run in any script that checks
    // only the exit status.
    if written == 0 {
        let labels: Vec<&str> = grid_points.iter().map(|p| p.label.as_str()).collect();
        die(&format!(
            "only={:?} matched none of the {} fixture labels: {}",
            only.unwrap_or_default(),
            labels.len(),
            labels.join(", ")
        ));
    }
    println!("wrote {written} fixture set(s) under {}", out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_args;

    fn args(rest: &[&str]) -> Vec<String> {
        std::iter::once("trxscan-benchmark".to_string())
            .chain(std::iter::once("/tmp/out".to_string()))
            .chain(rest.iter().map(|s| s.to_string()))
            .collect()
    }

    #[test]
    fn documented_key_value_syntax_is_the_syntax_that_is_accepted() {
        let (out, n, o, nz, only) =
            parse_args(&args(&["matrix=128", "oversample=8", "slices=3", "only=pf68_diff_nowin"]));
        assert_eq!(out.to_str(), Some("/tmp/out"));
        assert_eq!((n, o, nz), (128, 8, 3));
        assert_eq!(only.as_deref(), Some("pf68_diff_nowin"));
    }

    #[test]
    fn order_does_not_matter_and_omitted_keys_keep_their_defaults() {
        let (_, n, o, nz, only) = parse_args(&args(&["slices=2", "matrix=32"]));
        assert_eq!((n, o, nz), (32, 4, 2));
        assert!(only.is_none());
    }
}
