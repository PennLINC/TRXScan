//! Ground-truth fibre orientations from the orientation mixture, per acquisition voxel.
//!
//! The signal stage encodes each voxel's histogram row (weights on the hemisphere vertices, in
//! world RAS) as a sum of sticks, so the peaks of that row are the orientations a reconstruction
//! is trying to recover. This aggregates the `o×o` simulation cells of every acquisition voxel,
//! finds the local maxima on the vertex graph and refines each by the mass-weighted mean of its
//! neighbourhood, which recovers sub-bin directions (the level-3 icosphere bins are ~9° apart).

use crate::mixture::MixtureField;

/// Neighbourhood on the hemisphere: vertices within this angle (sign-free) of a peak vertex.
const NEIGHBOUR_DEG: f64 = 14.0;
/// Two peaks closer than this are one peak.
const MIN_SEPARATION_DEG: f64 = 25.0;
/// Peaks carrying less than this fraction of the voxel's mass are dropped.
const MIN_FRACTION: f64 = 0.05;

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Up to `npeaks` peaks of one histogram row: `(unit direction in world RAS, mass fraction)`,
/// strongest first.
pub fn row_peaks(verts: &[[f64; 3]], row: &[f64], npeaks: usize) -> Vec<([f64; 3], f64)> {
    let total: f64 = row.iter().sum();
    if total <= 0.0 {
        return Vec::new();
    }
    let cos_n = NEIGHBOUR_DEG.to_radians().cos();
    let cos_sep = MIN_SEPARATION_DEG.to_radians().cos();
    // local maxima: no neighbour strictly higher (ties broken by index so a plateau yields one)
    let mut maxima: Vec<(usize, f64)> = Vec::new();
    for i in 0..verts.len() {
        if row[i] <= 0.0 {
            continue;
        }
        let mut is_max = true;
        for j in 0..verts.len() {
            if j != i && dot(verts[i], verts[j]).abs() > cos_n && (row[j] > row[i] || (row[j] == row[i] && j < i)) {
                is_max = false;
                break;
            }
        }
        if is_max {
            maxima.push((i, row[i]));
        }
    }
    maxima.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut peaks: Vec<([f64; 3], f64)> = Vec::new();
    for (i, _) in maxima {
        if peaks.len() == npeaks {
            break;
        }
        // refine: mass-weighted mean of the neighbourhood, sign-aligned with the peak vertex
        let (mut acc, mut mass) = ([0.0f64; 3], 0.0f64);
        for j in 0..verts.len() {
            let d = dot(verts[i], verts[j]);
            if d.abs() > cos_n && row[j] > 0.0 {
                let s = if d < 0.0 { -1.0 } else { 1.0 };
                for c in 0..3 {
                    acc[c] += s * verts[j][c] * row[j];
                }
                mass += row[j];
            }
        }
        let norm = (acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]).sqrt();
        if norm <= 0.0 || mass / total < MIN_FRACTION {
            continue;
        }
        let dir = [acc[0] / norm, acc[1] / norm, acc[2] / norm];
        if peaks.iter().any(|(p, _)| dot(*p, dir).abs() > cos_sep) {
            continue;
        }
        peaks.push((dir, 0.0));
    }
    // Mass fraction: every vertex's weight goes to its nearest peak, so the fractions of a
    // voxel's peaks sum to one.
    for j in 0..verts.len() {
        if row[j] <= 0.0 {
            continue;
        }
        if let Some(k) = (0..peaks.len()).max_by(|&a, &b| {
            dot(peaks[a].0, verts[j]).abs().partial_cmp(&dot(peaks[b].0, verts[j]).abs()).unwrap()
        }) {
            peaks[k].1 += row[j] / total;
        }
    }
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    peaks
}

/// Peaks for every acquisition voxel, layout `vox * 3 * npeaks + 3 * k + c` (the `write_4d`
/// layout with `3 * npeaks` volumes): each peak is its unit direction scaled by its mass
/// fraction, zeros where there are fewer peaks. `o` is the in-plane oversampling of `mix`.
pub fn truth_peaks(mix: &MixtureField, o: usize, npeaks: usize) -> (Vec<f32>, [usize; 3]) {
    let [snx, sny, nz] = mix.dims;
    let (nx, ny) = (snx / o, sny / o);
    let nvert = mix.nvert();
    let verts = &mix.sphere.verts;
    let mut out = vec![0.0f32; nx * ny * nz * 3 * npeaks];
    let mut row = vec![0.0f64; nvert];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                row.iter_mut().for_each(|r| *r = 0.0);
                for dy in 0..o {
                    for dx in 0..o {
                        let sv = (x * o + dx) + snx * ((y * o + dy) + sny * z);
                        for (r, &w) in row.iter_mut().zip(mix.odf_row(sv)) {
                            *r += w as f64;
                        }
                    }
                }
                let vox = x + nx * (y + ny * z);
                for (k, (dir, frac)) in row_peaks(verts, &row, npeaks).into_iter().enumerate() {
                    for c in 0..3 {
                        out[(vox * npeaks + k) * 3 + c] = (dir[c] * frac) as f32;
                    }
                }
            }
        }
    }
    (out, [nx, ny, nz])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sphere::HemiSphere;

    fn watson_row(sphere: &HemiSphere, dirs: &[([f64; 3], f64)], kappa: f64) -> Vec<f64> {
        sphere
            .verts
            .iter()
            .map(|v| dirs.iter().map(|(d, w)| w * (kappa * (dot(*v, *d).powi(2) - 1.0)).exp()).sum())
            .collect()
    }

    fn unit(v: [f64; 3]) -> [f64; 3] {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    }

    fn angle_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
        dot(a, b).abs().min(1.0).acos().to_degrees()
    }

    #[test]
    fn a_single_off_vertex_direction_is_recovered_to_under_a_degree() {
        let s = HemiSphere::icosphere(3);
        let d = unit([0.3, 0.8, 0.52]);
        let row = watson_row(&s, &[(d, 1.0)], 20.0);
        let peaks = row_peaks(&s.verts, &row, 3);
        assert_eq!(peaks.len(), 1);
        assert!(angle_deg(peaks[0].0, d) < 1.0, "angle {}", angle_deg(peaks[0].0, d));
        assert!((peaks[0].1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_sixty_degree_crossing_gives_two_peaks_in_weight_order() {
        let s = HemiSphere::icosphere(3);
        let d1 = unit([1.0, 0.0, 0.0]);
        let d2 = unit([0.5, 0.866, 0.0]);
        let row = watson_row(&s, &[(d1, 2.0), (d2, 1.0)], 20.0);
        let peaks = row_peaks(&s.verts, &row, 3);
        assert_eq!(peaks.len(), 2);
        assert!(angle_deg(peaks[0].0, d1) < 1.5 && angle_deg(peaks[1].0, d2) < 1.5);
        assert!(peaks[0].1 > peaks[1].1);
        assert!((peaks[0].1 + peaks[1].1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn an_empty_row_has_no_peaks() {
        let s = HemiSphere::icosphere(3);
        assert!(row_peaks(&s.verts, &vec![0.0; s.len()], 3).is_empty());
    }
}
