//! Rigid head motion: mrsim-acq's poses, multiband slice schedules and resampling
//! ([`mrsim_acq::motion`], shared with aslscan), re-exported here whole, with TRXScan's
//! diffusion-specific dropout on top: [`dropout_seed`] and [`dropout_events`] (b0 volumes never
//! drop out), and [`apply_multiband_motion`] / [`apply_multiband_motion_slab`] by b-values, which
//! build mrsim-acq's [`DropoutLaw`] with [`diffusion_dropout_law`].

pub use mrsim_acq::motion::*;

/// The seed [`dropout_events`] is fed by the `trxscan` binary for `--seed <s>`: a fixed salt
/// mixed with the run seed, so dropout realisations are decoupled from the noise realisation.
/// Exposed so other front ends (the Python bindings) reproduce the CLI's dropout table exactly.
pub fn dropout_seed(run_seed: u64) -> u64 {
    0xB10C_5EED ^ run_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Deterministically pick DWI shots to corrupt: each DWI volume (b >= 50) gets a dropout event with
/// probability `rate`, at a pseudo-random shot, with severity 0.6–1.0 and a ~1–3 mm bulk jump.
pub fn dropout_events(bvals: &[f64], n_shots: usize, rate: f64, seed: u64) -> Vec<MotionEvent> {
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
        evs.push(MotionEvent {
            volume: g,
            shot,
            severity,
            jump_mm: [0.3 * j, j, 0.2 * j],
            jump_deg: [0.5, 0.3, 0.2],
        });
    }
    evs
}

/// The diffusion dropout law for `ngrad` volumes: a dropped shot is attenuated
/// `1 − severity·(b/b_max)`; a volume with `b < 50` (a b0, or one missing from `bvals`) is exempt,
/// and so is everything when `b_max <= 0`. mrsim-acq's [`DropoutLaw::Scaled`] normalises by the
/// largest drive, so `b_max` is appended after the volumes' drives, which makes it that largest
/// one; a `b_max` below the largest b-value has no such law and is refused.
pub fn diffusion_dropout_law(bvals: &[f64], b_max: f64, ngrad: usize) -> DropoutLaw {
    let mut drive: Vec<f64> = (0..ngrad).map(|g| bvals.get(g).copied().unwrap_or(0.0)).collect();
    if b_max > 0.0 {
        let top = drive.iter().cloned().fold(0.0f64, f64::max);
        assert!(b_max >= top, "b_max {b_max} is below the largest b-value {top}");
        drive.push(b_max);
    } else {
        drive.iter_mut().for_each(|d| *d = 0.0);
    }
    DropoutLaw::Scaled { drive, floor: 50.0 }
}

/// Apply multiband within-volume motion + slice dropout to the per-compartment signal in place.
/// For each event: (1) the head jumps for that shot and the ones after it within the volume (a
/// per-shot 3D resample of the affected slices — the within-volume wobble), and (2) the shot's
/// slices' diffusion signal is attenuated `1 − severity·(b/b_max)` (b0 exempt; higher b drops
/// harder) — the phenomenological dropout `eddy --repol` targets. Returns the dropped-shot ground
/// truth. Runs after volume-level motion, before k-space.
#[allow(clippy::too_many_arguments)]
pub fn apply_multiband_motion(
    images: &mut [Vec<f32>],
    dims: [usize; 3],
    ngrad: usize,
    v2w: [[f64; 4]; 4],
    mb: usize,
    interleaved: bool,
    bvals: &[f64],
    b_max: f64,
    events: &[MotionEvent],
) -> Vec<DroppedShot> {
    apply_multiband_motion_slab(images, dims, ngrad, v2w, mb, interleaved, bvals, b_max, events, None, None)
}

/// [`apply_multiband_motion`] on a slab of a larger volume: `slice_z` gives the full-FOV z index
/// of each local slice and `nz_full` the full slice count, so the multiband shot schedule (and
/// therefore which shots touch which slices) is the full volume's. Shots whose slices lie outside
/// the slab are skipped; the returned [`DroppedShot::slices`] are full-FOV indices. The geometric
/// jump resamples within the slab, so keep a few slices of context around the ones of interest.
#[allow(clippy::too_many_arguments)]
pub fn apply_multiband_motion_slab(
    images: &mut [Vec<f32>],
    dims: [usize; 3],
    ngrad: usize,
    v2w: [[f64; 4]; 4],
    mb: usize,
    interleaved: bool,
    bvals: &[f64],
    b_max: f64,
    events: &[MotionEvent],
    slice_z: Option<&[usize]>,
    nz_full: Option<usize>,
) -> Vec<DroppedShot> {
    let law = diffusion_dropout_law(bvals, b_max, ngrad);
    mrsim_acq::motion::apply_multiband_motion_slab(images, dims, ngrad, v2w, mb, interleaved, &law, events, slice_z, nz_full)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dropout_events_are_deterministic_and_skip_b0() {
        let bvals: Vec<f64> = vec![0.0, 1000.0, 1000.0, 2000.0, 0.0, 3000.0, 1000.0, 2000.0, 3000.0, 1000.0];
        let a = dropout_events(&bvals, 20, 0.5, dropout_seed(0));
        let b = dropout_events(&bvals, 20, 0.5, dropout_seed(0));
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!((x.volume, x.shot, x.severity, x.jump_mm, x.jump_deg), (y.volume, y.shot, y.severity, y.jump_mm, y.jump_deg));
        }
        assert!(a.iter().all(|e| bvals[e.volume] >= 50.0));
        assert!(a.iter().all(|e| e.shot < 20 && (0.6..=1.0).contains(&e.severity)));
        let all = dropout_events(&bvals, 20, 1.0, dropout_seed(0));
        assert_eq!(all.len(), 8, "rate 1 drops every DWI volume");
        assert!(dropout_events(&bvals, 20, 0.0, dropout_seed(0)).is_empty());
    }

    /// The law is main's formula, bit for bit: `1 − severity·(b/b_max)` in `f32`, b < 50 and
    /// volumes missing from `bvals` exempt, nothing attenuated when `b_max <= 0`, including a
    /// `b_max` above the largest b-value (the Python bindings' b0-only 1000).
    #[test]
    fn the_dropout_law_is_mains_formula() {
        let main = |bvals: &[f64], b_max: f64, g: usize, sev: f32| -> f32 {
            let is_b0 = bvals.get(g).copied().unwrap_or(0.0) < 50.0;
            if is_b0 || b_max <= 0.0 { 1.0 } else { 1.0 - sev * (bvals[g] / b_max) as f32 }
        };
        let cases: [(&[f64], f64); 4] =
            [(&[0.0, 1000.0, 2000.0, 30.0], 2000.0), (&[0.0, 700.0, 1300.0], 3000.0), (&[0.0, 0.0], 1000.0), (&[0.0, 1000.0], 0.0)];
        for (bvals, b_max) in cases {
            let ngrad = bvals.len() + 1;
            let law = diffusion_dropout_law(bvals, b_max, ngrad);
            for g in 0..ngrad {
                for sev in [0.6f32, 0.83, 1.0] {
                    assert_eq!(law.attenuation(g, sev).to_bits(), main(bvals, b_max, g, sev).to_bits(), "{bvals:?} {b_max} g={g}");
                }
            }
        }
        let r = std::panic::catch_unwind(|| diffusion_dropout_law(&[0.0, 2000.0], 1000.0, 2));
        assert!(r.is_err(), "b_max below the largest b-value");
    }
}
