//! End-to-end simulator: streamlines + tissue + scheme + fieldmap → distorted 4D DWI.
//! Stage A (clean signal) → Stage B v1 (EPI distortion + T2*).
//!
//! Usage:
//!   sp-sim <wm> <gm> <csf> <mask> <streamlines> <bval> <bvec> <fmap_hz> <out_prefix> [reverse_pe]

use std::path::Path;
use std::time::Instant;
use trxscan::compartments::{generate_compartments, generate_compartments_moving, CompartmentParams};
use trxscan::io;
use trxscan::kspace::{simulate_acquisition, Acquisition};
use trxscan::motion;
use trxscan::scheme::GradientScheme;

/// Deterministically pick DWI shots to corrupt: each DWI volume gets a dropout event with
/// probability `rate`, at a pseudo-random shot, with severity 0.6–1.0 and a ~1–3 mm bulk jump.
fn gen_dropout_events(bvals: &[f64], n_shots: usize, rate: f64, seed: u64) -> Vec<motion::MotionEvent> {
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
        evs.push(motion::MotionEvent {
            volume: g,
            shot,
            severity,
            jump_mm: [0.3 * j, j, 0.2 * j],
            jump_deg: [0.5, 0.3, 0.2],
        });
    }
    evs
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 10 {
        eprintln!("usage: sp-sim <wm> <gm> <csf> <mask> <streamlines> <bval> <bvec> <fmap_hz> <out_prefix> [reverse_pe] [noise] [eddy] [accel] [n_coils] [motion.tsv] [mb] [dropout_rate] [eddy_quad]");
        std::process::exit(2);
    }
    let p = |i: usize| Path::new(&a[i]);
    let reverse_pe = a.get(10).map(|s| s == "1" || s == "true").unwrap_or(false);
    let noise_variance = a.get(11).and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
    let eddy_strength = a.get(12).and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
    let accel = a.get(13).and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
    let n_coils = a.get(14).and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
    let motion_tsv = a.get(15).filter(|s| !s.is_empty() && *s != "-").cloned();
    let mb = a.get(16).and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
    let dropout_rate = a.get(17).and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
    let eddy_quad = a.get(18).and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);

    let (tissue, grid) = io::load_tissue(p(1), p(2), p(3), p(4))?;
    let (positions, offsets) = io::load_streamlines(p(5))?;
    let scheme = GradientScheme::from_fsl(p(6), p(7))?;
    let (fmap, fgrid) = io::load_volume(p(8))?;
    if fgrid.dims != grid.dims {
        return Err(format!("fieldmap grid {:?} != tissue grid {:?}", fgrid.dims, grid.dims).into());
    }
    println!("grid {:?}  {} streamlines  {} volumes  shells {:?}", grid.dims,
        offsets.len().saturating_sub(1), scheme.len(), scheme.shells(50.0));

    let params = CompartmentParams { b_value: scheme.b_max, ..Default::default() };
    let t = Instant::now();
    // With a motion trace, use the faithful path: per volume, transform the streamlines + tissue
    // by that volume's pose and re-simulate (Fiberfox-style). Otherwise the single-pose Stage A.
    let mut comp = if let Some(tsv) = &motion_tsv {
        let poses = motion::load_motion_tsv(Path::new(tsv))?;
        let moved = poses.iter().filter(|p| **p != motion::Pose::IDENTITY).count();
        println!("Motion (faithful re-simulation): {} poses, {} moved, from {}", poses.len(), moved, tsv);
        generate_compartments_moving(&grid, &positions, &offsets, &tissue, &scheme, &params, &poses)
    } else {
        generate_compartments(&grid, &positions, &offsets, &tissue, &scheme, &params)
    };
    println!("Stage A (per-compartment signal): {:?}  T2 {:?}", t.elapsed(), comp.t2);

    // Multiband within-volume motion + slice dropout: inject synthetic bulk-motion events on random
    // DWI shots and write the dropped-slice ground truth (for scoring eddy --repol / SHORELine).
    if mb > 1 && dropout_rate > 0.0 {
        let n_shots = (grid.dims[2] / mb).max(1);
        let events = gen_dropout_events(&scheme.bvals, n_shots, dropout_rate, 0xB10C_5EED);
        let gt = motion::apply_multiband_motion(
            &mut comp.images, grid.dims, comp.ngrad, grid.voxel_to_world,
            mb, true, &scheme.bvals, scheme.b_max, &events);
        let mut tsv = String::from("volume\tshot\tslices\tattenuation\n");
        for d in &gt {
            let sl = d.slices.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",");
            tsv.push_str(&format!("{}\t{}\t{}\t{:.3}\n", d.volume, d.shot, sl, d.attenuation));
        }
        std::fs::write(format!("{}_desc-dropout_slices.tsv", a[9]), tsv)?;
        println!("Multiband dropout: mb={mb}, {} shots dropped -> {}_desc-dropout_slices.tsv", gt.len(), a[9]);
    }

    // HBCD-like acquisition (tLine 0.4 ms, TE 88 ms), per-compartment T2, optional Rician noise.
    let acq = Acquisition {
        t_line: 0.4,
        t_echo: 88.0,
        t_inhom: 50.0,
        signal_scale: 100.0,
        reverse_phase: reverse_pe,
        do_distortions: true,
        do_relaxation: true,
        noise_variance,
        partial_fourier: 0.75,  // HBCD
        ghost_offset: 0.015,    // subtle residual Nyquist ghost
        eddy_strength,
        eddy_quad,
        eddy_tau: 70.0,
        n_spikes: 0,            // spikes are rare/aggressive; left off (available)
        spike_amplitude: 1.0,
        zero_ringing: 6.0,      // mild Gibbs
        n_coils,
        accel,
        acs_lines: 24,
    };
    // per-volume eddy gradient = unit bvec × b-value (b0 → zero → no eddy)
    let gradients: Vec<[f64; 3]> = (0..scheme.len())
        .map(|g| {
            let (d, b) = (scheme.bvecs[g], scheme.bvals[g]);
            [d[0] * b, d[1] * b, d[2] * b]
        })
        .collect();
    let t = Instant::now();
    let (mag, phase) =
        simulate_acquisition(grid.dims, comp.ngrad, &comp.images, &comp.t2, &fmap, &acq, &gradients);
    println!("Stage B (distortion+T2*{}{}{}): {:?}",
        if noise_variance > 0.0 { "+noise" } else { "" },
        if eddy_strength > 0.0 { "+eddy" } else { "" },
        if accel > 1 { "+GRAPPA" } else { "" },
        t.elapsed());

    io::write_complex_dwi(&a[9], grid.dims, comp.ngrad, &mag, &phase, &grid, &scheme)?;
    println!("wrote BIDS {}_part-{{mag,phase}}_dwi.nii.gz (+bval/bvec/json)", a[9]);
    Ok(())
}
