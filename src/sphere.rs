//! Hemisphere direction set for the per-voxel orientation histogram.
//!
//! A subdivided icosahedron, folded antipodally to a hemisphere: fibre orientations are
//! sign-free (`v` ≡ `−v`), and every consumer of the histogram — the `vv^T`/`vv^T⊗vv^T`
//! moments in [`crate::microstructure`], the stick/zeppelin responses in
//! [`crate::compartments`] — is antipodally invariant, so half the vertices carry all the
//! information. Pure std, deterministic, no external sphere files.
//!
//! Level 3 (642 full-sphere vertices → 321 hemisphere) keeps the binning quantization
//! (max ≈5.3° to the nearest vertex) well under the angular scale of the tensor responses,
//! and far under any realistic dispersion kernel width.

/// A unit hemisphere direction set. `verts` are unit vectors, one per antipodal pair,
/// canonicalized to `z > 0` (ties: `y > 0`, then `x > 0`).
#[derive(Debug, Clone)]
pub struct HemiSphere {
    pub verts: Vec<[f64; 3]>,
}

impl HemiSphere {
    /// Build by subdividing an icosahedron `level` times (level 3 → 321 hemisphere verts).
    pub fn icosphere(level: usize) -> HemiSphere {
        let (mut verts, mut faces) = icosahedron();
        for _ in 0..level {
            subdivide(&mut verts, &mut faces);
        }
        // fold antipodally: keep one canonical representative per ± pair
        let mut hemi: Vec<[f64; 3]> = Vec::with_capacity(verts.len() / 2);
        for v in verts {
            let c = canonical(v);
            if !hemi.iter().any(|h| close(*h, c)) {
                hemi.push(c);
            }
        }
        HemiSphere { verts: hemi }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.verts.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.verts.is_empty()
    }

    /// Index of the vertex closest to `dir` (sign-free: maximizes `|v·dir|`).
    /// `dir` need not be normalized; a zero vector returns 0.
    pub fn nearest(&self, dir: [f64; 3]) -> usize {
        let (mut best, mut best_d) = (0usize, -1.0f64);
        for (i, v) in self.verts.iter().enumerate() {
            let d = (v[0] * dir[0] + v[1] * dir[1] + v[2] * dir[2]).abs();
            if d > best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }
}

fn close(a: [f64; 3], b: [f64; 3]) -> bool {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    d0 * d0 + d1 * d1 + d2 * d2 < 1e-16
}

/// Canonical representative of the antipodal pair {v, −v}.
fn canonical(v: [f64; 3]) -> [f64; 3] {
    let eps = 1e-12;
    let flip = if v[2].abs() > eps {
        v[2] < 0.0
    } else if v[1].abs() > eps {
        v[1] < 0.0
    } else {
        v[0] < 0.0
    };
    if flip {
        [-v[0], -v[1], -v[2]]
    } else {
        v
    }
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

/// The 12 vertices / 20 faces of a unit icosahedron.
fn icosahedron() -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let verts: Vec<[f64; 3]> = [
        [-1.0, phi, 0.0], [1.0, phi, 0.0], [-1.0, -phi, 0.0], [1.0, -phi, 0.0],
        [0.0, -1.0, phi], [0.0, 1.0, phi], [0.0, -1.0, -phi], [0.0, 1.0, -phi],
        [phi, 0.0, -1.0], [phi, 0.0, 1.0], [-phi, 0.0, -1.0], [-phi, 0.0, 1.0],
    ]
    .into_iter()
    .map(normalize)
    .collect();
    let faces = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    (verts, faces)
}

/// One 4-to-1 face subdivision with edge-midpoint sharing.
fn subdivide(verts: &mut Vec<[f64; 3]>, faces: &mut Vec<[usize; 3]>) {
    use std::collections::HashMap;
    let mut midpoint: HashMap<(usize, usize), usize> = HashMap::new();
    let mut mid = |a: usize, b: usize, verts: &mut Vec<[f64; 3]>| -> usize {
        let key = (a.min(b), a.max(b));
        if let Some(&i) = midpoint.get(&key) {
            return i;
        }
        let (va, vb) = (verts[a], verts[b]);
        let m = normalize([(va[0] + vb[0]) / 2.0, (va[1] + vb[1]) / 2.0, (va[2] + vb[2]) / 2.0]);
        verts.push(m);
        let i = verts.len() - 1;
        midpoint.insert(key, i);
        i
    };
    let mut out = Vec::with_capacity(faces.len() * 4);
    for &[a, b, c] in faces.iter() {
        let ab = mid(a, b, verts);
        let bc = mid(b, c, verts);
        let ca = mid(c, a, verts);
        out.push([a, ab, ca]);
        out.push([b, bc, ab]);
        out.push([c, ca, bc]);
        out.push([ab, bc, ca]);
    }
    *faces = out;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level3_has_321_hemisphere_verts_all_unit_all_canonical() {
        let s = HemiSphere::icosphere(3);
        assert_eq!(s.len(), 321, "642 full-sphere verts fold to 321");
        for v in &s.verts {
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((n - 1.0).abs() < 1e-12);
            assert!(canonical(*v) == *v, "vertex not canonical: {v:?}");
        }
        // no antipodal duplicates
        for i in 0..s.len() {
            for j in (i + 1)..s.len() {
                let d = s.verts[i][0] * s.verts[j][0]
                    + s.verts[i][1] * s.verts[j][1]
                    + s.verts[i][2] * s.verts[j][2];
                assert!(d.abs() < 1.0 - 1e-9, "verts {i},{j} are (anti)parallel");
            }
        }
    }

    #[test]
    fn nearest_returns_self_for_vertex_directions_both_signs() {
        let s = HemiSphere::icosphere(2);
        for (i, v) in s.verts.iter().enumerate() {
            assert_eq!(s.nearest(*v), i);
            assert_eq!(s.nearest([-v[0], -v[1], -v[2]]), i, "sign-free lookup");
        }
    }

    #[test]
    fn quantization_error_is_small_at_level3() {
        let s = HemiSphere::icosphere(3);
        // worst-case angle to nearest vertex over a deterministic direction sweep
        let mut worst = 0.0f64;
        for a in 0..40 {
            for b in 0..40 {
                let (t, p) = (a as f64 * 0.0785, b as f64 * 0.157);
                let d = [t.sin() * p.cos(), t.sin() * p.sin(), t.cos()];
                let v = s.verts[s.nearest(d)];
                let dot = (v[0] * d[0] + v[1] * d[1] + v[2] * d[2]).abs().min(1.0);
                worst = worst.max(dot.acos().to_degrees());
            }
        }
        assert!(worst < 6.0, "worst-case quantization {worst}°"); // measured ≈5.3°
    }
}
