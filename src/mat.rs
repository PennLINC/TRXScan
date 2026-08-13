//! Tiny std-only 3×3 / vector helpers so the pure-math core (motion, signal) stays in the
//! default (dependency-free) build and unit-tests offline. Real hot paths can switch to `nalgebra`
//! later; the semantics here mirror `mitk::imv::GetRotationMatrixItk`
//! (`Modules/DiffusionImage/mitkDiffusionImageHelperFunctions.h:151`).

use crate::Vec3;

/// Row-major 3×3 matrix.
pub type Mat3 = [[f64; 3]; 3];

pub const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

#[inline]
pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
#[inline]
pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}
/// Normalize; returns `[0,0,0]` for a (near-)zero vector.
pub fn normalize(a: Vec3) -> Vec3 {
    let n = norm(a);
    if n < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        [a[0] / n, a[1] / n, a[2] / n]
    }
}

pub fn matvec(m: &Mat3, v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

pub fn matmul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    o
}

pub fn transpose(a: &Mat3) -> Mat3 {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[j][i];
        }
    }
    o
}

/// Rotation matrix from Euler angles in **degrees**, composed `Rz·Ry·Rx` — a faithful port of
/// `GetRotationMatrixItk` (rotate about X, then Y, then Z).
pub fn rotation_zyx_deg(rx: f64, ry: f64, rz: f64) -> Mat3 {
    let (rx, ry, rz) = (rx.to_radians(), ry.to_radians(), rz.to_radians());
    let rot_x = [[1.0, 0.0, 0.0], [0.0, rx.cos(), -rx.sin()], [0.0, rx.sin(), rx.cos()]];
    let rot_y = [[ry.cos(), 0.0, ry.sin()], [0.0, 1.0, 0.0], [-ry.sin(), 0.0, ry.cos()]];
    let rot_z = [[rz.cos(), -rz.sin(), 0.0], [rz.sin(), rz.cos(), 0.0], [0.0, 0.0, 1.0]];
    matmul(&rot_z, &matmul(&rot_y, &rot_x))
}

/// Inverse of a 3×3 matrix (adjugate / determinant). `None` if singular.
pub fn inverse3(m: &Mat3) -> Option<Mat3> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-18 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

/// Rodrigues rotation of `angle` (radians) about a unit `axis`.
pub fn axis_angle(axis: Vec3, angle: f64) -> Mat3 {
    let a = normalize(axis);
    if norm(a) < 0.5 {
        return IDENTITY; // degenerate axis
    }
    let (c, s) = (angle.cos(), angle.sin());
    let t = 1.0 - c;
    let (x, y, z) = (a[0], a[1], a[2]);
    [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_90_about_z_maps_x_to_y() {
        let r = rotation_zyx_deg(0.0, 0.0, 90.0);
        let v = matvec(&r, [1.0, 0.0, 0.0]);
        assert!((v[0]).abs() < 1e-9 && (v[1] - 1.0).abs() < 1e-9 && v[2].abs() < 1e-9, "{v:?}");
    }

    #[test]
    fn rotation_is_orthonormal() {
        let r = rotation_zyx_deg(11.0, -23.0, 42.0);
        let rt_r = matmul(&transpose(&r), &r);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((rt_r[i][j] - want).abs() < 1e-9, "RᵀR not identity at {i},{j}");
            }
        }
    }

    #[test]
    fn axis_angle_matches_euler_for_z() {
        let a = axis_angle([0.0, 0.0, 1.0], 90f64.to_radians());
        let e = rotation_zyx_deg(0.0, 0.0, 90.0);
        for i in 0..3 {
            for j in 0..3 {
                assert!((a[i][j] - e[i][j]).abs() < 1e-9);
            }
        }
    }
}
