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

/// The diffusion dropout law: a dropped shot is attenuated `1 − severity·(b/b_max)`; a volume with
/// `b < 50` (a b0, or one missing from `bvals`) is exempt, and so is everything when `b_max <= 0`
/// (mrsim-acq's [`DropoutLaw::ScaledTo`], `b_max` the caller's).
pub fn diffusion_dropout_law(bvals: &[f64], b_max: f64) -> DropoutLaw {
    DropoutLaw::ScaledTo { drive: bvals.to_vec(), max: b_max, floor: 50.0 }
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
    let law = diffusion_dropout_law(bvals, b_max);
    mrsim_acq::motion::apply_multiband_motion_slab(images, dims, ngrad, v2w, mb, interleaved, &law, events, slice_z, nz_full)
}

// ---- spin history (P9, part C3) ----

/// The spin-history correction of each compartment (`[compartment][vox * ngrad + v]` on `grid`),
/// from the compartments' signal before any per-shot motion. A diffusion series is spin echo at
/// 90 degrees with every volume prepared alike, so the reference is `1 - exp(-TR/T1)`: each
/// compartment's `T1` is uniform (`t1_s`, seconds, one per compartment). The engine
/// (`mrsim_acq::spin_history`) tracks every voxel the compartment has signal in, as exact pieces
/// along z, and is stepped through every shot of `plan` with its `history` transform; per volume its
/// weight is the compartment's signal over the reference magnetization, so the correction is in
/// signal units, weighted by the signal actually there.
///
/// Under the re-simulation path (`plan` built with volume poses) the images already sit at each
/// volume's pose, so the weights are read from them moved back by its inverse (trilinear), into
/// the head frame the engine's history is kept in.
pub fn spin_history_corrections(images: &[Vec<f32>], grid: &mrsim_acq::grid::Grid, ngrad: usize, t1_s: &[f64], tr_s: f64,
                                plan: &ShotPlan) -> Result<Vec<Vec<f32>>, String> {
    use mrsim_acq::spin_history::{Slabs, Source, SpinHistory, NO_CLASS};
    if t1_s.len() != images.len() {
        return Err(format!("spin history needs one T1 per compartment: {} T1s for {} compartments", t1_s.len(), images.len()));
    }
    if let Some(bad) = t1_s.iter().find(|t| !(t.is_finite() && **t > 0.0)) {
        return Err(format!("spin history: T1 {bad} s is not positive"));
    }
    if !(tr_s.is_finite() && tr_s > 0.0) {
        return Err(format!("spin history: TR {tr_s} s is not positive"));
    }
    let [nx, ny, nz] = grid.dims;
    let (nvox, slab) = (nx * ny * nz, nx * ny);
    let mut out = Vec::with_capacity(images.len());
    let all: Vec<usize> = (0..nz).collect();
    let nv = ngrad.min(plan.times.len());
    for (img, &t1) in images.iter().zip(t1_s) {
        let r = 1.0 - (-tr_s / t1).exp();
        // each volume's signal in the head frame (re-simulated images carry their volume pose)
        let head: Vec<Vec<f32>> = (0..nv).map(|v| {
            let vol: Vec<f32> = (0..nvox).map(|i| img[i * ngrad + v]).collect();
            match plan.volume.get(v) {
                Some(p) if *p != IDENTITY_MATRIX => resample_slices_by_matrix(&vol, grid.dims, grid.voxel_to_world, rigid_inverse(p), &all),
                _ => vol,
            }
        }).collect();
        let class: Vec<u8> = (0..nvox).map(|i| if head.iter().any(|h| h[i] != 0.0) { 0 } else { NO_CLASS }).collect();
        let t1v = vec![t1 as f32; nvox];
        let zeros = vec![0.0f32; nvox];
        let mut h = SpinHistory::new(
            Source { dims: grid.dims, v2w: grid.voxel_to_world, t1: &t1v, weight: &zeros, class: &class, n_classes: 1 },
            Slabs { dims: grid.dims, v2w: grid.voxel_to_world },
            Vec::new(),
        );
        let prime: Vec<(f64, &[usize])> = plan.schedule.iter().enumerate().map(|(s, sl)| (plan.times[0][s] - tr_s, sl.as_slice())).collect();
        h.prime(&prime, plan.history[0][0]);
        let mut corr = vec![0.0f32; nvox * ngrad];
        let reference = move |t1: f64| 1.0 - (-tr_s / t1).exp();
        for (v, hv) in head.iter().enumerate() {
            let weight: Vec<f32> = hv.iter().map(|x| (*x as f64 / r) as f32).collect();
            for (s, slices) in plan.schedule.iter().enumerate() {
                for term in h.excite(plan.times[v][s], plan.history[v][s], slices, &reference, Some(&weight)) {
                    for (cell, x) in term.num[0].iter().enumerate() {
                        corr[(term.slice * slab + cell) * ngrad + v] += *x;
                    }
                }
            }
        }
        out.push(corr);
    }
    Ok(out)
}

/// [`apply_shot_motion`] with spin history: the corrections from the signal before the motion,
/// the per-shot motion and dropout, then each compartment's correction added at its shot's slices
/// times that shot's dropout attenuation. Returns the dropped-shot truth and the correction summed
/// over compartments (`vox * ngrad + v` on this grid). The whole slice stack only.
#[allow(clippy::too_many_arguments)]
pub fn apply_shot_motion_with_spin_history(
    images: &mut [Vec<f32>],
    grid: &mrsim_acq::grid::Grid,
    ngrad: usize,
    plan: &ShotPlan,
    bvals: &[f64],
    b_max: f64,
    t1_s: &[f64],
    tr_s: f64,
) -> Result<(Vec<DroppedShot>, Vec<f32>), String> {
    let corr = spin_history_corrections(images, grid, ngrad, t1_s, tr_s, plan)?;
    let dropped = apply_shot_motion(images, grid.dims, ngrad, grid.voxel_to_world, plan, None, bvals, b_max);
    let law = diffusion_dropout_law(bvals, b_max);
    let slab = grid.dims[0] * grid.dims[1];
    let mut sum = vec![0.0f32; corr.first().map_or(0, |c| c.len())];
    for v in 0..ngrad {
        for (s, slices) in plan.schedule.iter().enumerate() {
            let att: f32 = plan.events.iter().filter(|e| e.volume == v && e.shot == s).map(|e| law.attenuation(v, e.severity)).product();
            for &z in slices {
                for xy in 0..slab {
                    let i = (z * slab + xy) * ngrad + v;
                    for (img, c) in images.iter_mut().zip(&corr) {
                        img[i] += c[i] * att;
                        sum[i] += c[i];
                    }
                }
            }
        }
    }
    Ok((dropped, sum))
}

// ---- per-shot motion (P9, part C): any slice order, single band, events from a file ----

/// The order in which a volume's shots (groups of `mb` slices `{g, g + nz/mb, ...}`) are excited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum SliceOrder {
    /// Even groups, then odd: the order `apply_multiband_motion` has always used.
    Interleaved,
    Sequential,
    Descending,
    /// Odd groups, then even.
    OddFirst,
}

/// Each slice's excitation offset (s) within a repetition of `tr_s` for `nz` slices at multiband
/// `mb` in `order`: the shots are spread evenly, shot `s` at `s * tr_s / n_shots`, every slice of a
/// shot sharing its time. `Interleaved` and `Sequential` give the shots of
/// [`slice_schedule`]`(nz, mb, ..)` in the same order (through [`slice_schedule_from_times`]).
/// `nz` must be a multiple of `mb`: every shot excites exactly `mb` slices.
pub fn shot_offsets(nz: usize, mb: usize, order: SliceOrder, tr_s: f64) -> Result<Vec<f64>, String> {
    let mb = mb.max(1);
    if nz == 0 || nz % mb != 0 {
        return Err(format!("{nz} slices are not a multiple of the multiband factor {mb}: a shot would excite a different number of slices"));
    }
    let n_groups = nz / mb;
    let groups: Vec<usize> = match order {
        SliceOrder::Sequential => (0..n_groups).collect(),
        SliceOrder::Descending => (0..n_groups).rev().collect(),
        SliceOrder::Interleaved => (0..n_groups).step_by(2).chain((1..n_groups).step_by(2)).collect(),
        SliceOrder::OddFirst => (1..n_groups).step_by(2).chain((0..n_groups).step_by(2)).collect(),
    };
    let mut offsets = vec![0.0f64; nz];
    for (s, &g) in groups.iter().enumerate() {
        let t = s as f64 * tr_s / n_groups as f64;
        for j in 0..mb {
            if let Some(o) = offsets.get_mut(g + j * n_groups) {
                *o = t;
            }
        }
    }
    Ok(offsets)
}

/// Check per-slice offsets before a plan is built from them: a finite positive repetition, every
/// offset finite and inside `[0, tr_s)`, and each distinct offset shared by exactly `mb` slices
/// (one at single band), so a shot is the slices the protocol excites together.
pub fn check_shot_timing(offsets: &[f64], mb: usize, tr_s: f64) -> Result<(), String> {
    if !(tr_s.is_finite() && tr_s > 0.0) {
        return Err(format!("the repetition time {tr_s} s is not positive"));
    }
    if let Some(bad) = offsets.iter().find(|t| !(t.is_finite() && **t >= 0.0 && **t < tr_s)) {
        return Err(format!("slice offset {bad} s is not inside the repetition [0, {tr_s}) s"));
    }
    let mb = mb.max(1);
    for shot in slice_schedule_from_times(offsets) {
        if shot.len() != mb {
            return Err(format!("slices {shot:?} share an excitation time, but the multiband factor is {mb}"));
        }
    }
    Ok(())
}

/// The shots of a run (full slice stack): the schedule, each shot's time, and per shot two
/// world-space transforms about the full stack's field-of-view centre: `geometry` (what the
/// images are resampled by) and `history` (the head's actual pose, for spin history). Without
/// per-volume poses both are the shot's cumulative jump `J`. With them (the `--motion` /
/// `Artifacts.motion` path, whose images the re-simulation already put at the volume pose `P`),
/// `geometry` is `J` and `history` is `J . P`.
#[derive(Debug, Clone)]
pub struct ShotPlan {
    pub schedule: Vec<Vec<usize>>,
    /// `[volume][shot]`, seconds on the series clock (`v * tr_s + offset`).
    pub times: Vec<Vec<f64>>,
    pub geometry: Vec<Vec<[[f64; 4]; 4]>>,
    pub history: Vec<Vec<[[f64; 4]; 4]>>,
    /// `[volume]`: the volume pose `P` (the identity without volume poses), which re-simulated
    /// images already carry.
    pub volume: Vec<[[f64; 4]; 4]>,
    pub events: Vec<MotionEvent>,
}

/// The inverse of a rigid world-space transform: `[R^T | -R^T t]`.
fn rigid_inverse(m: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = IDENTITY_MATRIX;
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = m[j][i];
        }
        out[i][3] = -(0..3).map(|k| m[k][i] * m[k][3]).sum::<f64>();
    }
    out
}

fn matmul4(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut c = [[0.0; 4]; 4];
    for (i, row) in c.iter_mut().enumerate() {
        for (j, x) in row.iter_mut().enumerate() {
            *x = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

impl ShotPlan {
    /// `offsets` per slice of the full stack (seconds), `full_grid` the full stack's grid (its
    /// centre is the rotation centre).
    pub fn new(offsets: &[f64], n_volumes: usize, tr_s: f64, volume_poses: Option<&[Pose]>, events: Vec<MotionEvent>,
               full_grid: &mrsim_acq::grid::Grid) -> ShotPlan {
        let schedule = slice_schedule_from_times(offsets);
        let c = fov_center(full_grid.dims, full_grid.voxel_to_world);
        let mat = |p: Pose| if p == Pose::IDENTITY { IDENTITY_MATRIX } else { p.to_matrix(c) };
        let n_shots = schedule.len();
        let mut times = vec![vec![0.0; n_shots]; n_volumes];
        let mut geometry = vec![vec![IDENTITY_MATRIX; n_shots]; n_volumes];
        let mut history = vec![vec![IDENTITY_MATRIX; n_shots]; n_volumes];
        let mut volume = vec![IDENTITY_MATRIX; n_volumes];
        for v in 0..n_volumes {
            let mut cum = Pose::IDENTITY;
            let pm = mat(volume_poses.and_then(|p| p.get(v).copied()).unwrap_or(Pose::IDENTITY));
            volume[v] = pm;
            for (s, slices) in schedule.iter().enumerate() {
                for e in events.iter().filter(|e| e.volume == v && e.shot == s) {
                    for i in 0..3 {
                        cum.trans_mm[i] += e.jump_mm[i];
                        cum.rot_deg[i] += e.jump_deg[i];
                    }
                }
                times[v][s] = v as f64 * tr_s + offsets[slices[0]];
                let jm = mat(cum);
                geometry[v][s] = jm;
                history[v][s] = if pm == IDENTITY_MATRIX { jm } else if jm == IDENTITY_MATRIX { pm } else { matmul4(&jm, &pm) };
            }
        }
        ShotPlan { schedule, times, geometry, history, volume, events }
    }
}

/// Read an events TSV in aslscan's `desc-motionEvents_gt.tsv` format: `volume`, `shot`, `slices`
/// (ignored), `attenuation` (read as `1 - severity`, the dropout law's input; the b-value law
/// applies after reading), `jump_x/y/z` (mm), `jump_rx/ry/rz` (radians). Every cell present and
/// finite, `volume < n_volumes`, `shot < n_shots`, `attenuation` in `[0, 1]`, one row per volume and
/// shot.
pub fn load_motion_events(path: &std::path::Path, n_volumes: usize, n_shots: usize) -> Result<Vec<MotionEvent>, String> {
    const COLS: [&str; 9] = ["volume", "shot", "attenuation", "jump_x", "jump_y", "jump_z", "jump_rx", "jump_ry", "jump_rz"];
    let what = format!("motion events {}", path.display());
    let text = std::fs::read_to_string(path).map_err(|e| format!("{what}: {e}"))?;
    let mut lines = text.lines().map(|l| l.trim_end_matches('\r')).filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines.next().ok_or_else(|| format!("{what}: empty"))?.split('\t').collect();
    let idx = COLS.iter()
        .map(|c| header.iter().position(|h| h.trim() == *c).ok_or_else(|| format!("{what}: column {c:?} missing")))
        .collect::<Result<Vec<usize>, String>>()?;
    // `slices` must be there (the format's columns), though the schedule decides the slices
    if !header.iter().any(|h| h.trim() == "slices") {
        return Err(format!("{what}: column \"slices\" missing"));
    }
    let mut events: Vec<MotionEvent> = Vec::new();
    for (i, line) in lines.enumerate() {
        let cells: Vec<&str> = line.split('\t').collect();
        let mut v = [0.0f64; 9];
        for (k, (&c, name)) in idx.iter().zip(COLS).enumerate() {
            let cell = cells.get(c).map(|s| s.trim()).ok_or_else(|| format!("{what} row {i}: {name} missing"))?;
            v[k] = cell.parse::<f64>().ok().filter(|x| x.is_finite())
                .ok_or_else(|| format!("{what} row {i}: {name} = {cell:?} is not a finite number"))?;
        }
        let index = |x: f64, name: &str, limit: usize| -> Result<usize, String> {
            if x >= 0.0 && x.fract() == 0.0 && (x as usize) < limit { Ok(x as usize) } else {
                Err(format!("{what} row {i}: {name} {x} is not an index below {limit}"))
            }
        };
        let (volume, shot) = (index(v[0], "volume", n_volumes)?, index(v[1], "shot", n_shots)?);
        if !(0.0..=1.0).contains(&v[2]) {
            return Err(format!("{what} row {i}: attenuation {} is not in [0, 1]", v[2]));
        }
        if events.iter().any(|e| e.volume == volume && e.shot == shot) {
            return Err(format!("{what} row {i}: a second event for volume {volume}, shot {shot}"));
        }
        events.push(MotionEvent {
            volume, shot, severity: (1.0 - v[2]) as f32,
            jump_mm: [v[3], v[4], v[5]],
            jump_deg: [v[6].to_degrees(), v[7].to_degrees(), v[8].to_degrees()],
        });
    }
    Ok(events)
}

/// Events in the format [`load_motion_events`] reads: raw severities (`attenuation = 1 - severity`,
/// before any b-value scaling), full precision; `slices` from the schedule.
pub fn write_motion_events(events: &[MotionEvent], schedule: &[Vec<usize>]) -> String {
    let mut s = String::from("volume\tshot\tslices\tattenuation\tjump_x\tjump_y\tjump_z\tjump_rx\tjump_ry\tjump_rz\n");
    for e in events {
        let sl = schedule.get(e.shot).map(|x| x.iter().map(|z| z.to_string()).collect::<Vec<_>>().join(",")).unwrap_or_default();
        s.push_str(&format!("{}\t{}\t{sl}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n", e.volume, e.shot, 1.0 - e.severity,
                            e.jump_mm[0], e.jump_mm[1], e.jump_mm[2],
                            e.jump_deg[0].to_radians(), e.jump_deg[1].to_radians(), e.jump_deg[2].to_radians()));
    }
    s
}

/// Per-shot motion and dropout on the per-compartment signal, in place: each shot's slices are
/// resampled by its `geometry` transform and attenuated `1 - severity (b/b_max)` per event (b0
/// exempt). `slice_z` / `nz_full` place a slab in the full stack as for
/// [`apply_multiband_motion_slab`]: the plan's schedule is the full stack's, mapped to the slab's
/// local slices, with every shot kept (a shot outside the slab is empty). Returns the dropped-shot
/// truth, slices in full-stack indices.
#[allow(clippy::too_many_arguments)]
pub fn apply_shot_motion(
    images: &mut [Vec<f32>],
    dims: [usize; 3],
    ngrad: usize,
    v2w: [[f64; 4]; 4],
    plan: &ShotPlan,
    slice_z: Option<&[usize]>,
    bvals: &[f64],
    b_max: f64,
) -> Vec<DroppedShot> {
    let law = diffusion_dropout_law(bvals, b_max);
    let local_of = |zg: usize| -> Option<usize> {
        match slice_z {
            None => (zg < dims[2]).then_some(zg),
            Some(sz) => sz.iter().position(|&z| z == zg),
        }
    };
    let schedule: Vec<Vec<usize>> = plan.schedule.iter().map(|s| s.iter().filter_map(|&z| local_of(z)).collect()).collect();
    let n_shots = schedule.len();
    let attenuation: Vec<Vec<Vec<f32>>> = (0..ngrad).map(|g| {
        (0..n_shots).map(|s| plan.events.iter().filter(|e| e.volume == g && e.shot == s).map(|e| law.attenuation(g, e.severity)).collect()).collect()
    }).collect();
    let geometry: Vec<Vec<[[f64; 4]; 4]>> = (0..ngrad).map(|g| plan.geometry.get(g).cloned().unwrap_or_else(|| vec![IDENTITY_MATRIX; n_shots])).collect();
    apply_shot_transforms(images, dims, ngrad, v2w, &schedule, &geometry, &attenuation);
    plan.events.iter().filter(|e| e.volume < ngrad).map(|e| DroppedShot {
        volume: e.volume, shot: e.shot, slices: plan.schedule[e.shot].clone(), attenuation: law.attenuation(e.volume, e.severity),
    }).collect()
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

    #[test]
    fn slice_orders_give_their_schedules() {
        for (nz, mb) in [(12usize, 1usize), (12, 2), (12, 3), (9, 3)] {
            let sched = |o: SliceOrder| slice_schedule_from_times(&shot_offsets(nz, mb, o, 2.0).unwrap());
            assert_eq!(sched(SliceOrder::Interleaved), slice_schedule(nz, mb, true), "nz {nz} mb {mb}");
            assert_eq!(sched(SliceOrder::Sequential), slice_schedule(nz, mb, false));
            let mut rev = slice_schedule(nz, mb, false);
            rev.reverse();
            assert_eq!(sched(SliceOrder::Descending), rev);
            let odd: Vec<Vec<usize>> = sched(SliceOrder::OddFirst);
            assert_eq!(odd[0][0], 1, "odd-first starts with group 1");
        }
        let o = shot_offsets(6, 1, SliceOrder::Sequential, 3.0).unwrap();
        assert!((o[5] - 2.5).abs() < 1e-12, "shots spread evenly over the repetition");
    }

    #[test]
    fn events_file_round_trips_and_is_validated() {
        let sched: Vec<Vec<usize>> = (0..4).map(|z| vec![z]).collect();
        let ev = vec![
            MotionEvent { volume: 1, shot: 2, severity: 0.75, jump_mm: [0.1, -0.4, 1.25], jump_deg: [0.0, 2.0, -1.5] },
            MotionEvent { volume: 3, shot: 0, severity: 0.0, jump_mm: [0.0; 3], jump_deg: [0.0; 3] },
        ];
        let dir = std::env::temp_dir().join(format!("trxscan-events-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ev.tsv");
        std::fs::write(&path, write_motion_events(&ev, &sched)).unwrap();
        let back = load_motion_events(&path, 4, 4).unwrap();
        assert_eq!(back.len(), 2);
        for (a, b) in ev.iter().zip(&back) {
            assert_eq!((a.volume, a.shot), (b.volume, b.shot));
            assert!((a.severity - b.severity).abs() < 1e-6);
            for i in 0..3 {
                assert!((a.jump_mm[i] - b.jump_mm[i]).abs() < 1e-12 && (a.jump_deg[i] - b.jump_deg[i]).abs() < 1e-9);
            }
        }
        assert!(load_motion_events(&path, 3, 4).unwrap_err().contains("volume 3"));
        let no_slices = write_motion_events(&ev, &sched).replacen("\tslices", "", 1)
            .lines().enumerate().map(|(i, l)| if i == 0 { l.to_string() } else {
                let mut c: Vec<&str> = l.split('\t').collect();
                c.remove(2);
                c.join("\t")
            }).collect::<Vec<_>>().join("\n");
        std::fs::write(&path, no_slices).unwrap();
        assert!(load_motion_events(&path, 4, 4).unwrap_err().contains("\"slices\""));
        let mut dup = write_motion_events(&ev, &sched);
        dup.push_str("1\t2\t2\t1\t0\t0\t0\t0\t0\t0\n");
        std::fs::write(&path, dup).unwrap();
        assert!(load_motion_events(&path, 4, 4).unwrap_err().contains("second event"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// With no volume poses, the per-shot path on the interleaved order equals the multiband path to
    /// rounding, b-value law included; single band drops one slice per event.
    #[test]
    fn shot_motion_matches_multiband_motion() {
        let (nx, ny, nz, ngrad) = (6usize, 5usize, 6usize, 3usize);
        let mut v2w = IDENTITY_MATRIX;
        v2w[0][0] = 1.7;
        v2w[1][1] = 1.7;
        v2w[2][2] = 2.0;
        let grid = mrsim_acq::grid::Grid { dims: [nx, ny, nz], voxel_to_world: v2w };
        let mk = || vec![(0..nx * ny * nz * ngrad).map(|i| 1.0 + ((i * 7) % 11) as f32 * 0.1).collect::<Vec<f32>>()];
        let bvals = [0.0, 1000.0, 2000.0];
        let events = vec![
            MotionEvent { volume: 1, shot: 1, severity: 0.8, jump_mm: [0.6, 0.0, 0.4], jump_deg: [0.0, 0.0, 2.0] },
            MotionEvent { volume: 2, shot: 2, severity: 0.7, jump_mm: [0.0, 0.5, 0.0], jump_deg: [1.0, 0.0, 0.0] },
        ];
        let mut old = mk();
        let gt_old = apply_multiband_motion(&mut old, [nx, ny, nz], ngrad, v2w, 2, true, &bvals, 2000.0, &events);
        let plan = ShotPlan::new(&shot_offsets(nz, 2, SliceOrder::Interleaved, 3.0).unwrap(), ngrad, 3.0, None, events.clone(), &grid);
        let mut new = mk();
        let gt_new = apply_shot_motion(&mut new, [nx, ny, nz], ngrad, v2w, &plan, None, &bvals, 2000.0);
        let peak = mk()[0].iter().fold(0.0f32, |m, x| m.max(*x));
        for i in 0..old[0].len() {
            assert!((old[0][i] - new[0][i]).abs() <= 1e-6 * peak, "{i}: {} vs {}", old[0][i], new[0][i]);
        }
        assert_eq!(gt_old.len(), gt_new.len());
        for (a, b) in gt_old.iter().zip(&gt_new) {
            assert_eq!((a.volume, a.shot, &a.slices, a.attenuation.to_bits()), (b.volume, b.shot, &b.slices, b.attenuation.to_bits()));
        }
        // single band: one slice per event
        let plan1 = ShotPlan::new(&shot_offsets(nz, 1, SliceOrder::Sequential, 3.0).unwrap(), ngrad, 3.0, None, events, &grid);
        let mut img = mk();
        let gt1 = apply_shot_motion(&mut img, [nx, ny, nz], ngrad, v2w, &plan1, None, &bvals, 2000.0);
        assert!(gt1.iter().all(|d| d.slices.len() == 1));
    }

    /// With volume poses the geometry is the jump alone and the history the jump after the pose.
    #[test]
    fn shot_plan_splits_geometry_and_history() {
        let grid = mrsim_acq::grid::Grid { dims: [4, 4, 4], voxel_to_world: IDENTITY_MATRIX };
        let p = Pose { rot_deg: [0.0, 2.0, 0.0], trans_mm: [0.0, 0.0, 1.0] };
        let e = MotionEvent { volume: 0, shot: 1, severity: 0.0, jump_mm: [0.0, 0.0, 0.5], jump_deg: [0.0; 3] };
        let plan = ShotPlan::new(&shot_offsets(4, 1, SliceOrder::Sequential, 2.0).unwrap(), 1, 2.0, Some(&[p]), vec![e], &grid);
        let c = fov_center(grid.dims, grid.voxel_to_world);
        let j = Pose { rot_deg: [0.0; 3], trans_mm: [0.0, 0.0, 0.5] };
        assert_eq!(plan.geometry[0][0], IDENTITY_MATRIX);
        assert_eq!(plan.history[0][0], p.to_matrix(c));
        assert_eq!(plan.geometry[0][1], j.to_matrix(c));
        assert_eq!(plan.history[0][1], matmul4(&j.to_matrix(c), &p.to_matrix(c)));
        assert!((plan.times[0][3] - 1.5).abs() < 1e-12);
    }

    /// A uniform two-compartment object on 2 mm slices, sequential single band, TR 4 s: a jump of a
    /// quarter slice toward the next slab at shot 3 of volume 1 corrects slice 3 of volume 1 by
    /// `-0.25 S (1 - e_dt / e_TR)` per compartment, `S` its signal and `T1` its own; nothing in the
    /// slices before; a still head gets nothing. The relative correction does not depend on the
    /// diffusion signal, and the dropped shot's attenuation scales its correction.
    #[test]
    fn spin_history_meets_the_closed_form_per_compartment() {
        let (nx, ny, nz, ngrad) = (4usize, 4usize, 8usize, 3usize);
        let mut v2w = IDENTITY_MATRIX;
        v2w[2][2] = 2.0;
        let grid = mrsim_acq::grid::Grid { dims: [nx, ny, nz], voxel_to_world: v2w };
        let nvox = nx * ny * nz;
        let tr = 4.0;
        let t1 = [0.9, 1.6];
        // compartment signals per volume (a b=0 and two diffusion-weighted volumes)
        let sig = [[1.0f32, 0.6, 0.3], [0.5, 0.45, 0.4]];
        let mk = || -> Vec<Vec<f32>> { (0..2).map(|c| (0..nvox * ngrad).map(|i| sig[c][i % ngrad]).collect()).collect() };
        let offsets = shot_offsets(nz, 1, SliceOrder::Sequential, tr).unwrap();
        let jump = MotionEvent { volume: 1, shot: 3, severity: 0.0, jump_mm: [0.0, 0.0, 0.5], jump_deg: [0.0; 3] };
        let still = ShotPlan::new(&offsets, ngrad, tr, None, Vec::new(), &grid);
        let corr = spin_history_corrections(&mk(), &grid, ngrad, &t1, tr, &still).unwrap();
        assert!(corr.iter().flatten().all(|x| x.abs() < 1e-6), "a still head");
        let plan = ShotPlan::new(&offsets, ngrad, tr, None, vec![jump], &grid);
        let corr = spin_history_corrections(&mk(), &grid, ngrad, &t1, tr, &plan).unwrap();
        let dt = tr / nz as f64;
        let at = |c: usize, x: usize, y: usize, z: usize, v: usize| corr[c][(x + nx * (y + ny * z)) * ngrad + v] as f64;
        for c in 0..2 {
            let (e_tr, e_dt) = (1.0 - (-tr / t1[c]).exp(), 1.0 - (-dt / t1[c]).exp());
            let want = -0.25 * (1.0 - e_dt / e_tr);
            for (x, y) in [(0, 0), (2, 1), (3, 3)] {
                assert!((at(c, x, y, 3, 1) / sig[c][1] as f64 - want).abs() < 1e-5, "compartment {c}: {} vs {want}", at(c, x, y, 3, 1) / sig[c][1] as f64);
                for z in 0..3 {
                    assert!(at(c, x, y, z, 1).abs() < 1e-7, "slice {z} before the jump");
                }
            }
        }
        // b-value independence: the same events with every volume's signal scaled gives the same
        // relative correction
        let scaled: Vec<Vec<f32>> = mk().into_iter().map(|c| c.into_iter().map(|x| x * 0.37).collect()).collect();
        let corr2 = spin_history_corrections(&scaled, &grid, ngrad, &t1, tr, &plan).unwrap();
        for c in 0..2 {
            for i in 0..nvox * ngrad {
                assert!((corr2[c][i] - 0.37 * corr[c][i]).abs() <= 1e-6, "{c} {i}");
            }
        }
        // applied: the correction is added at the shot's slices, times its attenuation
        let ev = MotionEvent { severity: 0.5, ..jump };
        let plan = ShotPlan::new(&offsets, ngrad, tr, None, vec![ev], &grid);
        let bvals = [0.0, 1000.0, 2000.0];
        let mut img = mk();
        let (dropped, sum) = apply_shot_motion_with_spin_history(&mut img, &grid, ngrad, &plan, &bvals, 2000.0, &t1, tr).unwrap();
        assert_eq!(dropped.len(), 1);
        let i = (1 + nx * (1 + ny * 3)) * ngrad + 1;
        let att = 1.0 - 0.5 * (1000.0 / 2000.0);
        assert!((sum[i] as f64 - (corr[0][i] + corr[1][i]) as f64).abs() < 1e-6);
        let base = mk();
        let mut moved = base.clone();
        apply_shot_motion(&mut moved, grid.dims, ngrad, v2w, &plan, None, &bvals, 2000.0);
        for c in 0..2 {
            assert!((img[c][i] - (moved[c][i] + corr[c][i] * att as f32)).abs() < 1e-6);
        }
        assert!(spin_history_corrections(&mk(), &grid, ngrad, &[1.0], tr, &plan).unwrap_err().contains("one T1 per compartment"));
    }

    /// Review fix: on the re-simulation path the images already carry the volume pose, so the
    /// weights are read back in the head frame and the correction lands on the tissue: signal at
    /// scanner x = 4 under a +2 mm x volume pose is corrected at x = 4, nothing at x = 6.
    #[test]
    fn spin_history_weights_undo_the_volume_pose() {
        let (nx, ny, nz, ngrad) = (10usize, 2usize, 8usize, 2usize);
        let mut v2w = IDENTITY_MATRIX;
        v2w[2][2] = 2.0;
        let grid = mrsim_acq::grid::Grid { dims: [nx, ny, nz], voxel_to_world: v2w };
        let nvox = nx * ny * nz;
        let img: Vec<f32> = (0..nvox * ngrad).map(|i| if (i / ngrad) % nx == 4 { 1.0 } else { 0.0 }).collect();
        let poses = [Pose { trans_mm: [2.0, 0.0, 0.0], ..Pose::IDENTITY }; 2];
        let jump = MotionEvent { volume: 1, shot: 3, severity: 0.0, jump_mm: [0.0, 0.0, 0.5], jump_deg: [0.0; 3] };
        let offsets = shot_offsets(nz, 1, SliceOrder::Sequential, 4.0).unwrap();
        let plan = ShotPlan::new(&offsets, ngrad, 4.0, Some(&poses), vec![jump], &grid);
        let corr = spin_history_corrections(&[img], &grid, ngrad, &[1.33], 4.0, &plan).unwrap();
        let at = |x: usize, z: usize| corr[0][(x + nx * ny * z) * ngrad + 1];
        assert!(at(4, 3) < -1e-3, "the tissue at x = 4 is corrected: {}", at(4, 3));
        assert!(at(6, 3).abs() < 1e-7 && at(2, 3).abs() < 1e-7, "nothing where there is no tissue: {} {}", at(6, 3), at(2, 3));
        assert!(shot_offsets(5, 2, SliceOrder::Sequential, 3.0).is_err(), "5 slices at multiband 2");
    }

    /// Final-review fix: shot timing is checked before a plan is built.
    #[test]
    fn shot_timing_is_checked() {
        let ok = shot_offsets(6, 2, SliceOrder::Interleaved, 3.0).unwrap();
        assert!(check_shot_timing(&ok, 2, 3.0).is_ok());
        assert!(check_shot_timing(&ok, 2, 0.0).unwrap_err().contains("not positive"));
        assert!(check_shot_timing(&ok, 1, 3.0).unwrap_err().contains("multiband factor is 1"));
        assert!(check_shot_timing(&[0.0, 0.5, 3.5], 1, 3.0).unwrap_err().contains("inside the repetition"));
        assert!(check_shot_timing(&[0.0, f64::NAN], 1, 3.0).is_err());
    }

    /// The law is main's formula, bit for bit: `1 − severity·(b/b_max)` in `f32`, b < 50 and
    /// volumes missing from `bvals` exempt, nothing attenuated when `b_max <= 0`, for any `b_max`
    /// (above the largest b-value, as the Python bindings' b0-only 1000, or below it, as a nominal
    /// b_max with jittered b-values).
    #[test]
    fn the_dropout_law_is_mains_formula() {
        let main = |bvals: &[f64], b_max: f64, g: usize, sev: f32| -> f32 {
            let is_b0 = bvals.get(g).copied().unwrap_or(0.0) < 50.0;
            if is_b0 || b_max <= 0.0 { 1.0 } else { 1.0 - sev * (bvals[g] / b_max) as f32 }
        };
        let cases: [(&[f64], f64); 5] = [(&[0.0, 1000.0, 2000.0, 30.0], 2000.0), (&[0.0, 700.0, 1300.0], 3000.0),
                                          (&[0.0, 0.0], 1000.0), (&[0.0, 1000.0], 0.0), (&[0.0, 1005.0, 995.0], 1000.0)];
        for (bvals, b_max) in cases {
            let ngrad = bvals.len() + 1;
            let law = diffusion_dropout_law(bvals, b_max);
            for g in 0..ngrad {
                for sev in [0.6f32, 0.83, 1.0] {
                    assert_eq!(law.attenuation(g, sev).to_bits(), main(bvals, b_max, g, sev).to_bits(), "{bvals:?} {b_max} g={g}");
                }
            }
        }
    }
}
