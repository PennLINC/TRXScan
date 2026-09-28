//! Head motion: rigid transforms applied to the fibers (and mask) per volume — or, for the new
//! within-volume feature, per slice-group.
//!
//! Port of `SimulateMotion` (`itkTractsToDWIImageFilter.cpp:1456`). Modes:
//! - **Random** (`:1486`): each listed volume gets an independent uniform draw in `[-amp,+amp]`
//!   per axis; fibers+mask reset to baseline between moved volumes (`:1461`) → intermittent motion.
//! - **Linear** (`:1508`): constant increment `amp / n_moved` per listed volume; accumulates and
//!   persists → monotonic drift.
//! - **Trajectory** (new): an explicit per-volume (or per slice-group) 6-DOF pose — the thing
//!   Fiberfox can't take from its CLI. Enables drift+sneeze, real motion traces, and multiband
//!   within-volume motion.
//!
//! The applied per-unit transform must also be handed to [`crate::kspace`] so the fieldmap warp
//! tracks the moved head (Fiberfox: `SetTranslation`/`SetRotationMatrix`, `:270`).
//!
//! ## Applying the pose — reuse `trx-rs`, don't hand-roll the point loop
//! `trx-rs` already transforms streamlines: `apply_transform_in_place(&mut Tractogram, &TransformChain)`
//! (`../rust/trx-rs/src/transform.rs:26`, rayon-parallel with its `parallel` feature). A `TransformChain`
//! is built from `itk_transforms_rs::{Affine3, TransformChain}` via `chain.push_affine(Affine3::from_matrix(m))`
//! (`transform.rs:100`). So this module only needs to **compute** the per-unit pose matrix `m`
//! (`nalgebra::Matrix4<f64>`); trx-rs applies it — and the same chain composes with the ITK/H5
//! registration warps you already use.
//!
//! Poses are **absolute**, so apply each to a *fresh copy of the baseline streamlines* (clone →
//! `apply_transform`), mirroring Fiberfox's reset-to-baseline between moved volumes (`:1461`);
//! do not accumulate in place.

use crate::Vec3;

#[derive(Debug, Clone)]
pub enum MotionMode {
    Off,
    /// per-axis amplitude (mm) + (deg), impulses that return to baseline
    Random { trans_mm: Vec3, rot_deg: Vec3, volumes: Vec<usize> },
    /// per-axis end-of-scan totals (mm) + (deg), monotonic drift
    Linear { trans_mm: Vec3, rot_deg: Vec3, volumes: Vec<usize> },
    /// explicit pose per acquisition unit (volume, or slice-group when within_volume)
    Trajectory { poses: Vec<Pose> },
}

/// A rigid pose: rotation (deg, per axis) + translation (mm).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Pose {
    pub rot_deg: Vec3,
    pub trans_mm: Vec3,
}

impl Pose {
    pub const IDENTITY: Pose = Pose { rot_deg: [0.0; 3], trans_mm: [0.0; 3] };

    /// 4×4 rigid transform (world mm) about `center`, faithful to
    /// `FiberBundle::TransformFibers` (`mitkFiberBundle.cpp:1632`): `p' = R·(p−c) + c + t`, with
    /// `R = Rz·Ry·Rx` in degrees. Hand this to trx-rs via `Affine3::from_matrix` (real impl builds
    /// a `nalgebra::Matrix4`).
    pub fn to_matrix(&self, center: crate::Vec3) -> [[f64; 4]; 4] {
        let r = crate::mat::rotation_zyx_deg(self.rot_deg[0], self.rot_deg[1], self.rot_deg[2]);
        // translation part = c − R·c + t   (so that center maps to center + t)
        let rc = crate::mat::matvec(&r, center);
        let tx = center[0] - rc[0] + self.trans_mm[0];
        let ty = center[1] - rc[1] + self.trans_mm[1];
        let tz = center[2] - rc[2] + self.trans_mm[2];
        [
            [r[0][0], r[0][1], r[0][2], tx],
            [r[1][0], r[1][1], r[1][2], ty],
            [r[2][0], r[2][1], r[2][2], tz],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
}

/// Deterministic, seedable PRNG (SplitMix64) so motion is reproducible without pulling `rand` into
/// the default build. (Fiberfox uses a Mersenne Twister; we only need *our own* reproducibility.)
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// uniform in [0,1)
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// uniform in [-amp, amp]  (matches Fiberfox `GetVariateWithClosedRange(2·amp) − amp`)
    fn signed(&mut self, amp: f64) -> f64 {
        2.0 * amp * self.unit() - amp
    }
}

fn as_flags(volumes: &[usize], n_units: usize) -> Vec<bool> {
    let mut f = vec![false; n_units];
    for &v in volumes {
        if v < n_units {
            f[v] = true;
        }
    }
    f
}

/// Resolve a motion mode into the per-unit **absolute** pose sequence — the ground truth to apply
/// (clone-baseline-then-transform) and to write to a motion TSV. Mirrors `SimulateMotion`
/// (`itkTractsToDWIImageFilter.cpp:1456`).
pub fn resolve_poses(mode: &MotionMode, n_units: usize, seed: u64) -> Vec<Pose> {
    let mut poses = vec![Pose::IDENTITY; n_units];
    match mode {
        MotionMode::Off => {}
        MotionMode::Random { trans_mm, rot_deg, volumes } => {
            // independent draw per listed volume; unlisted stay at baseline (reset-to-baseline).
            let mut rng = SplitMix64(seed.wrapping_add(0x1234_5678));
            for &v in volumes {
                if v >= n_units {
                    continue;
                }
                poses[v] = Pose {
                    rot_deg: [rng.signed(rot_deg[0]), rng.signed(rot_deg[1]), rng.signed(rot_deg[2])],
                    trans_mm: [rng.signed(trans_mm[0]), rng.signed(trans_mm[1]), rng.signed(trans_mm[2])],
                };
            }
        }
        MotionMode::Linear { trans_mm, rot_deg, volumes } => {
            // constant increment per listed volume; accumulates and persists (monotonic drift).
            let n = volumes.len().max(1) as f64;
            let flags = as_flags(volumes, n_units);
            let mut counter = 0.0_f64;
            for v in 0..n_units {
                if flags[v] {
                    counter += 1.0;
                }
                let k = counter;
                poses[v] = Pose {
                    rot_deg: [rot_deg[0] / n * k, rot_deg[1] / n * k, rot_deg[2] / n * k],
                    trans_mm: [trans_mm[0] / n * k, trans_mm[1] / n * k, trans_mm[2] / n * k],
                };
            }
        }
        MotionMode::Trajectory { poses: p } => {
            for (i, slot) in poses.iter_mut().enumerate() {
                if let Some(x) = p.get(i) {
                    *slot = *x;
                }
            }
        }
    }
    poses
}

/// Parse a qsiprep/`eddy`-style confounds TSV — columns `trans_x/y/z` (mm) and `rot_x/y/z`
/// (**radians**), one row per volume — into an absolute per-volume [`Pose`] trajectory (rotations
/// converted to the degrees `Pose` uses). `n/a` cells map to 0. This is the "real motion trace" path.
pub fn load_motion_tsv(path: &std::path::Path) -> Result<Vec<Pose>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut lines = text.lines();
    let header = lines.next().ok_or("empty TSV")?;
    let cols: Vec<&str> = header.split('\t').collect();
    let find = |name: &str| {
        cols.iter().position(|&c| c == name).ok_or_else(|| format!("column '{name}' missing"))
    };
    let (tx, ty, tz) = (find("trans_x")?, find("trans_y")?, find("trans_z")?);
    let (rx, ry, rz) = (find("rot_x")?, find("rot_y")?, find("rot_z")?);
    let deg = 180.0 / std::f64::consts::PI;
    let mut poses = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let v = |i: usize| f.get(i).and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(0.0);
        poses.push(Pose {
            trans_mm: [v(tx), v(ty), v(tz)],
            rot_deg: [v(rx) * deg, v(ry) * deg, v(rz) * deg],
        });
    }
    Ok(poses)
}

#[inline]
pub(crate) fn trilinear(vol: &[f32], dims: [usize; 3], p: Vec3) -> f32 {
    let [nx, ny, nz] = dims;
    let (x, y, z) = (p[0], p[1], p[2]);
    if x < 0.0 || y < 0.0 || z < 0.0 || x > (nx - 1) as f64 || y > (ny - 1) as f64 || z > (nz - 1) as f64 {
        return 0.0;
    }
    let (x0, y0, z0) = (x.floor() as usize, y.floor() as usize, z.floor() as usize);
    let (x1, y1, z1) = ((x0 + 1).min(nx - 1), (y0 + 1).min(ny - 1), (z0 + 1).min(nz - 1));
    let (fx, fy, fz) = (x - x0 as f64, y - y0 as f64, z - z0 as f64);
    let at = |xi: usize, yi: usize, zi: usize| vol[xi + nx * (yi + ny * zi)] as f64;
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let c00 = lerp(at(x0, y0, z0), at(x1, y0, z0), fx);
    let c10 = lerp(at(x0, y1, z0), at(x1, y1, z0), fx);
    let c01 = lerp(at(x0, y0, z1), at(x1, y0, z1), fx);
    let c11 = lerp(at(x0, y1, z1), at(x1, y1, z1), fx);
    lerp(lerp(c00, c10, fy), lerp(c01, c11, fy), fz) as f32
}

/// The FOV centre in world coordinates (the rotation centre for a pose on this grid).
pub fn fov_center(dims: [usize; 3], v2w: [[f64; 4]; 4]) -> Vec3 {
    use crate::mat;
    let [nx, ny, nz] = dims;
    let lin = linear3(v2w);
    let c = mat::matvec(&lin, [(nx as f64 - 1.0) / 2.0, (ny as f64 - 1.0) / 2.0, (nz as f64 - 1.0) / 2.0]);
    [c[0] + v2w[0][3], c[1] + v2w[1][3], c[2] + v2w[2][3]]
}

fn linear3(v2w: [[f64; 4]; 4]) -> crate::mat::Mat3 {
    [
        [v2w[0][0], v2w[0][1], v2w[0][2]],
        [v2w[1][0], v2w[1][1], v2w[1][2]],
        [v2w[2][0], v2w[2][1], v2w[2][2]],
    ]
}

/// Resample one 3D scalar volume by an absolute rigid pose (head moves by `p' = R(p−c)+c+t`,
/// rotation about the FOV centre). Backward-maps each output voxel and trilinearly samples `src`.
pub fn resample_by_pose(src: &[f32], dims: [usize; 3], v2w: [[f64; 4]; 4], pose: Pose) -> Vec<f32> {
    use crate::mat;
    let [nx, ny, nz] = dims;
    let nvox = nx * ny * nz;
    if pose == Pose::IDENTITY {
        return src.to_vec();
    }
    let lin = linear3(v2w);
    let origin = [v2w[0][3], v2w[1][3], v2w[2][3]];
    let linv = match mat::inverse3(&lin) {
        Some(m) => m,
        None => return src.to_vec(),
    };
    let c = fov_center(dims, v2w);
    let rt = mat::transpose(&mat::rotation_zyx_deg(pose.rot_deg[0], pose.rot_deg[1], pose.rot_deg[2]));
    let t = pose.trans_mm;
    let mut out = vec![0.0f32; nvox];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let w = mat::matvec(&lin, [x as f64, y as f64, z as f64]);
                let wv = [w[0] + origin[0], w[1] + origin[1], w[2] + origin[2]];
                let d = [wv[0] - c[0] - t[0], wv[1] - c[1] - t[1], wv[2] - c[2] - t[2]];
                let pr = mat::matvec(&rt, d);
                let hv = mat::matvec(&linv, [pr[0] + c[0] - origin[0], pr[1] + c[1] - origin[1], pr[2] + c[2] - origin[2]]);
                out[x + nx * (y + ny * z)] = trilinear(src, dims, hv);
            }
        }
    }
    out
}

/// Fast (approximate) motion: resample each volume of each compartment image by its pose. This
/// moves the head geometrically but does NOT rotate the fiber orientations relative to the
/// gradients — use [`crate::compartments::generate_compartments_moving`] for the faithful
/// (re-simulate-from-moved-streamlines) version. Fieldmap stays fixed (scanner space).
pub fn apply_motion(images: &mut [Vec<f32>], dims: [usize; 3], ngrad: usize, v2w: [[f64; 4]; 4], poses: &[Pose]) {
    let nvox = dims[0] * dims[1] * dims[2];
    for g in 0..ngrad {
        let pose = poses.get(g).copied().unwrap_or(Pose::IDENTITY);
        if pose == Pose::IDENTITY {
            continue;
        }
        for img in images.iter_mut() {
            let src: Vec<f32> = (0..nvox).map(|v| img[v * ngrad + g]).collect();
            let moved = resample_by_pose(&src, dims, v2w, pose);
            for v in 0..nvox {
                img[v * ngrad + g] = moved[v];
            }
        }
    }
}

/// Multiband slice schedule: acquisition order of shots → the anatomical slices excited in each.
/// `mb` slices (separated by `nz/mb`) fire simultaneously per shot; `n_shots = nz/mb` shots per
/// volume, acquired sequentially or interleaved (even groups then odd).
pub fn slice_schedule(nz: usize, mb: usize, interleaved: bool) -> Vec<Vec<usize>> {
    let mb = mb.max(1);
    let n_groups = nz / mb; // simultaneously-excited group count = shots per volume
    if n_groups == 0 {
        return vec![(0..nz).collect()];
    }
    let group_slices = |gi: usize| -> Vec<usize> { (0..mb).map(|j| gi + j * n_groups).filter(|&z| z < nz).collect() };
    let order: Vec<usize> = if interleaved {
        (0..n_groups).step_by(2).chain((1..n_groups).step_by(2)).collect()
    } else {
        (0..n_groups).collect()
    };
    order.into_iter().map(group_slices).collect()
}

/// A synthetic within-volume bulk-motion event: at `shot` of `volume` the head jumps (geometric)
/// and the diffusion signal of that shot's slices drops out (`severity` 0..1).
#[derive(Debug, Clone, Copy)]
pub struct MotionEvent {
    pub volume: usize,
    pub shot: usize,
    pub severity: f32,
    pub jump_mm: [f64; 3],
    pub jump_deg: [f64; 3],
}

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

/// Ground truth of a dropped shot (for scoring `eddy --repol` / SHORELine outlier detection).
#[derive(Debug, Clone)]
pub struct DroppedShot {
    pub volume: usize,
    pub shot: usize,
    pub slices: Vec<usize>,
    pub attenuation: f32,
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
    let [nx, ny, nz] = dims;
    let nvox = nx * ny * nz;
    let nz_full = nz_full.unwrap_or(nz);
    if let Some(sz) = slice_z {
        assert_eq!(sz.len(), nz, "slice_z must name every local slice");
    }
    // full-FOV schedule, then each shot's slices mapped to LOCAL indices (dropping any outside)
    let schedule_full = slice_schedule(nz_full, mb, interleaved);
    let local_of = |zg: usize| -> Option<usize> {
        match slice_z {
            None => (zg < nz).then_some(zg),
            Some(sz) => sz.iter().position(|&z| z == zg),
        }
    };
    let schedule: Vec<Vec<usize>> =
        schedule_full.iter().map(|shot| shot.iter().filter_map(|&z| local_of(z)).collect()).collect();
    let n_shots = schedule.len();
    let at3 = |x: usize, y: usize, z: usize| x + nx * (y + ny * z);
    let mut gt = Vec::new();

    for g in 0..ngrad {
        let evs: Vec<&MotionEvent> = events.iter().filter(|e| e.volume == g && e.shot < n_shots).collect();
        if evs.is_empty() {
            continue;
        }
        // within-volume cumulative pose per shot (each event's jump persists for later shots)
        let mut shot_pose = vec![Pose::IDENTITY; n_shots];
        let mut cum = Pose::IDENTITY;
        for (k, sp) in shot_pose.iter_mut().enumerate() {
            for e in &evs {
                if e.shot == k {
                    for i in 0..3 {
                        cum.trans_mm[i] += e.jump_mm[i];
                        cum.rot_deg[i] += e.jump_deg[i];
                    }
                }
            }
            *sp = cum;
        }
        // geometric: resample each moved shot's slices from the volume's pre-motion signal
        if shot_pose.iter().any(|p| *p != Pose::IDENTITY) {
            for img in images.iter_mut() {
                let orig: Vec<f32> = (0..nvox).map(|v| img[v * ngrad + g]).collect();
                for (k, &pose) in shot_pose.iter().enumerate() {
                    if pose == Pose::IDENTITY {
                        continue;
                    }
                    let rs = resample_by_pose(&orig, dims, v2w, pose);
                    for &z in &schedule[k] {
                        for y in 0..ny {
                            for x in 0..nx {
                                img[at3(x, y, z) * ngrad + g] = rs[at3(x, y, z)];
                            }
                        }
                    }
                }
            }
        }
        // dropout: attenuate each event shot's slices (DWI only, b-scaled)
        let is_b0 = bvals.get(g).copied().unwrap_or(0.0) < 50.0;
        for e in &evs {
            let atten = if is_b0 || b_max <= 0.0 {
                1.0
            } else {
                1.0 - e.severity * (bvals[g] / b_max) as f32
            };
            for img in images.iter_mut() {
                for &z in &schedule[e.shot] {
                    for y in 0..ny {
                        for x in 0..nx {
                            img[at3(x, y, z) * ngrad + g] *= atten;
                        }
                    }
                }
            }
            gt.push(DroppedShot { volume: g, shot: e.shot, slices: schedule_full[e.shot].clone(), attenuation: atten });
        }
    }
    gt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiband_slab_matches_the_full_volume_where_no_jump_crosses_the_edge() {
        // Pure dropout (zero jump) on a 2-slice slab of a 6-slice volume, mb=2: the slab sees the
        // full volume's shot schedule, so the same slices attenuate by the same factor.
        let (nx, ny, nz, ngrad) = (4usize, 3usize, 6usize, 2usize);
        let v2w = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let mk = || vec![(0..nx * ny * nz * ngrad).map(|i| 1.0 + (i % 13) as f32).collect::<Vec<f32>>()];
        let bvals = [0.0, 2000.0];
        let events = [MotionEvent { volume: 1, shot: 1, severity: 0.8, jump_mm: [0.0; 3], jump_deg: [0.0; 3] }];
        let mut full = mk();
        let gt_full = apply_multiband_motion(&mut full, [nx, ny, nz], ngrad, v2w, 2, true, &bvals, 2000.0, &events);
        // slab = slices 4..6 (local 0,1 = global 4,5)
        let mut slab = vec![full[0][..0].to_vec()];
        let orig = mk();
        slab[0] = orig[0][nx * ny * 4 * ngrad..nx * ny * 6 * ngrad].to_vec();
        let gt_slab = apply_multiband_motion_slab(
            &mut slab, [nx, ny, 2], ngrad, v2w, 2, true, &bvals, 2000.0, &events, Some(&[4, 5]), Some(nz),
        );
        assert_eq!(slab[0], full[0][nx * ny * 4 * ngrad..nx * ny * 6 * ngrad].to_vec());
        assert_eq!(gt_full.len(), 1);
        assert_eq!(gt_full[0].slices, gt_slab[0].slices, "dropped slices are reported in full-FOV indices");
        assert!(gt_full[0].slices.contains(&4) || gt_full[0].slices.contains(&5) || gt_full[0].slices.iter().all(|&z| z < 4));
    }

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
    fn identity_pose_is_the_identity_matrix() {
        let m = Pose::IDENTITY.to_matrix([10.0, -5.0, 3.0]);
        for i in 0..3 {
            for j in 0..4 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((m[i][j] - want).abs() < 1e-9, "{i},{j}={}", m[i][j]);
            }
        }
    }

    #[test]
    fn rotation_leaves_center_fixed_plus_translation() {
        let center = [4.0, 2.0, -1.0];
        let p = Pose { rot_deg: [0.0, 0.0, 30.0], trans_mm: [1.0, 0.0, 0.0] };
        let m = p.to_matrix(center);
        // apply to the center point: should land at center + translation
        let c = [
            m[0][0] * center[0] + m[0][1] * center[1] + m[0][2] * center[2] + m[0][3],
            m[1][0] * center[0] + m[1][1] * center[1] + m[1][2] * center[2] + m[1][3],
            m[2][0] * center[0] + m[2][1] * center[1] + m[2][2] * center[2] + m[2][3],
        ];
        assert!((c[0] - (center[0] + 1.0)).abs() < 1e-9);
        assert!((c[1] - center[1]).abs() < 1e-9);
        assert!((c[2] - center[2]).abs() < 1e-9);
    }

    #[test]
    fn linear_drift_ramps_to_amplitude() {
        // all 74 non-baseline volumes listed → smooth ramp 0 → amp (matches the 3 mm Fiberfox run)
        let n = 75;
        let volumes: Vec<usize> = (1..n).collect();
        let poses = resolve_poses(
            &MotionMode::Linear { trans_mm: [3.0, 3.0, 2.0], rot_deg: [3.0, 2.0, 2.0], volumes },
            n,
            0,
        );
        assert_eq!(poses[0].trans_mm, [0.0, 0.0, 0.0]);
        let last = poses[n - 1];
        assert!((last.trans_mm[0] - 3.0).abs() < 1e-9, "{:?}", last.trans_mm);
        assert!((last.rot_deg[2] - 2.0).abs() < 1e-9);
        // monotonic
        for v in 1..n {
            assert!(poses[v].trans_mm[0] >= poses[v - 1].trans_mm[0] - 1e-12);
        }
    }

    #[test]
    fn random_moves_only_listed_within_bounds_and_is_deterministic() {
        let mode = MotionMode::Random {
            trans_mm: [2.5, 2.5, 2.5],
            rot_deg: [2.0, 2.0, 2.0],
            volumes: vec![4, 8, 13],
        };
        let a = resolve_poses(&mode, 20, 42);
        let b = resolve_poses(&mode, 20, 42);
        for v in 0..20 {
            // determinism
            assert_eq!(a[v].trans_mm, b[v].trans_mm);
            if [4, 8, 13].contains(&v) {
                for i in 0..3 {
                    assert!(a[v].trans_mm[i].abs() <= 2.5 + 1e-9);
                    assert!(a[v].rot_deg[i].abs() <= 2.0 + 1e-9);
                }
            } else {
                assert_eq!(a[v].trans_mm, [0.0, 0.0, 0.0], "unlisted vol {v} moved");
                assert_eq!(a[v].rot_deg, [0.0, 0.0, 0.0]);
            }
        }
    }

    #[test]
    fn load_tsv_reads_trans_mm_and_rot_rad_to_deg() {
        let dir = std::env::temp_dir();
        let path = dir.join("sp_motion_test.tsv");
        std::fs::write(&path,
            "framewise_displacement\ttrans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n\
             n/a\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\n\
             0.7\t-0.5\t0.25\t0.1\t0.0174533\t0.0\t0.0\n").unwrap();
        let poses = load_motion_tsv(&path).unwrap();
        assert_eq!(poses.len(), 2);
        assert_eq!(poses[0], Pose::IDENTITY);
        assert!((poses[1].trans_mm[0] - (-0.5)).abs() < 1e-9);
        assert!((poses[1].rot_deg[0] - 1.0).abs() < 1e-3); // 0.0174533 rad ≈ 1°
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn apply_motion_translation_shifts_the_head() {
        let (n, ng) = (9usize, 1usize);
        let dims = [n, n, n];
        let id = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let mut img = vec![0.0f32; n * n * n]; // one compartment, ngrad=1
        let at = |x: usize, y: usize, z: usize| x + n * (y + n * z);
        img[at(4, 4, 4)] = 1.0; // bright voxel at centre
        let mut images = vec![img];
        // +2 mm in x on an identity grid → head moves +2 voxels in x
        let poses = vec![Pose { trans_mm: [2.0, 0.0, 0.0], rot_deg: [0.0, 0.0, 0.0] }];
        apply_motion(&mut images, dims, ng, id, &poses);
        assert!(images[0][at(6, 4, 4)] > 0.8, "head should shift to x=6: {}", images[0][at(6, 4, 4)]);
        assert!(images[0][at(4, 4, 4)] < 0.2, "original position should empty out");
    }

    #[test]
    fn multiband_dropout_darkens_dwi_shot_not_b0() {
        let (nx, ny, nz) = (4, 4, 4);
        let dims = [nx, ny, nz];
        let (ngrad, nvox) = (2usize, nx * ny * nz);
        let id = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let mut images = vec![vec![1.0f32; nvox * ngrad]]; // one compartment, uniform
        let bvals = [0.0, 1000.0]; // vol 0 = b0, vol 1 = DWI
        let ev = MotionEvent { volume: 1, shot: 0, severity: 1.0, jump_mm: [0.0; 3], jump_deg: [0.0; 3] };
        let gt = apply_multiband_motion(&mut images, dims, ngrad, id, 2, false, &bvals, 1000.0, &[ev]);
        let at = |x: usize, y: usize, z: usize, g: usize| (x + nx * (y + ny * z)) * ngrad + g;
        // mb=2, nz=4 → shot 0 = slices {0, 2}: both drop in the DWI volume
        assert!(images[0][at(1, 1, 0, 1)] < 0.01, "DWI shot slice should drop");
        assert!(images[0][at(1, 1, 2, 1)] < 0.01, "MB-partner slice drops together");
        assert!((images[0][at(1, 1, 1, 1)] - 1.0).abs() < 0.01, "non-event slice kept");
        assert!((images[0][at(1, 1, 0, 0)] - 1.0).abs() < 1e-6, "b0 exempt from dropout");
        assert_eq!(gt.len(), 1);
        assert_eq!(gt[0].slices, vec![0, 2]);
    }
}
