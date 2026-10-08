//! Exact streamline rasterization: each segment → the voxels it crosses, with sub-voxel path
//! lengths (in **world mm**), on the target DWI grid (arbitrary/oblique affine).
//!
//! Port of `IntersectImage` (`Algorithms/itkTractsToDWIImageFilter.cpp:1105`). Unlike TRXViz's
//! `BoundaryContactField` (midpoint voxel, isotropic axis-aligned grid), this is a proper
//! parametric voxel traversal bound to the acquisition grid.
//!
//! Method: map both endpoints into continuous voxel coordinates with `world_to_voxel`; the segment
//! is `p(t) = a_v + t·(b_v − a_v)`, `t ∈ [0,1]`. Collect every integer-boundary crossing `t`, and
//! between consecutive crossings the segment lies in one voxel; the physical length there is
//! `(t₁ − t₀) · |b − a|_world` (so oblique/anisotropic grids are handled by the world metric).

use crate::mat;
use crate::Vec3;

/// The acquisition voxel grid: dimensions + voxel→world (RAS mm) affine (may be oblique).
/// mrsim-acq's; the rasterisation geometry is [`GridRaster`].
pub use mrsim_acq::grid::Grid;

/// One intersection of a streamline segment with a voxel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentHit {
    pub voxel: [usize; 3],
    /// path length of the segment inside this voxel (world mm)
    pub length: f64,
}

/// Rasterisation geometry for [`Grid`]. `Grid` lives in mrsim-acq, so these cannot be inherent
/// methods; the trait keeps call sites in method syntax (`use crate::raster::GridRaster`).
pub trait GridRaster {
    /// The simulation grid: in-plane refined by `o`, same FOV, slice direction untouched.
    ///
    /// The origin shifts by half the difference between the coarse and fine voxel sizes on each
    /// refined axis, matching the half-cell registration the forward transform uses and the
    /// geometry `scripts/prepare_acquisition_grid.py` writes. Identity at `o = 1`.
    fn hires(&self, o: usize) -> Grid;
    /// The linear (3×3) part and origin of the voxel→world affine.
    fn linear_and_origin(&self) -> (mat::Mat3, Vec3);
    /// World point → continuous voxel coordinate. Returns `None` if the affine is singular.
    fn world_to_voxel(&self, p: Vec3) -> Option<Vec3>;
    /// Walk one segment `a → b` (world mm); return each voxel it passes through with the length of
    /// the intersection (world mm). Voxels outside the grid are dropped.
    fn intersect_segment(&self, a: Vec3, b: Vec3) -> Vec<SegmentHit>;
    /// [`GridRaster::intersect_segment`] writing into caller-owned buffers (`ts` is scratch, `hits`
    /// is cleared and filled), so a loop over millions of segments allocates nothing per segment.
    /// Same hits, same order, same lengths.
    fn intersect_segment_into(&self, a: Vec3, b: Vec3, ts: &mut Vec<f64>, hits: &mut Vec<SegmentHit>);
}

impl GridRaster for Grid {
    fn hires(&self, o: usize) -> Grid {
        assert!(o > 0, "oversampling factor must be positive");
        let f = o as f64;
        let mut m = self.voxel_to_world;
        for r in 0..3 {
            for c in 0..2 {
                m[r][c] /= f;
            }
        }
        for r in 0..3 {
            m[r][3] = self.voxel_to_world[r][3]
                - 0.5 * self.voxel_to_world[r][0] * (1.0 - 1.0 / f)
                - 0.5 * self.voxel_to_world[r][1] * (1.0 - 1.0 / f);
        }
        Grid { dims: [self.dims[0] * o, self.dims[1] * o, self.dims[2]], voxel_to_world: m }
    }

    fn linear_and_origin(&self) -> (mat::Mat3, Vec3) {
        let m = &self.voxel_to_world;
        (
            [[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]],
            [m[0][3], m[1][3], m[2][3]],
        )
    }

    fn world_to_voxel(&self, p: Vec3) -> Option<Vec3> {
        let (lin, origin) = self.linear_and_origin();
        let inv = mat::inverse3(&lin)?;
        Some(mat::matvec(&inv, mat::sub(p, origin)))
    }

    fn intersect_segment(&self, a: Vec3, b: Vec3) -> Vec<SegmentHit> {
        let mut ts = Vec::new();
        let mut hits = Vec::new();
        self.intersect_segment_into(a, b, &mut ts, &mut hits);
        hits
    }

    fn intersect_segment_into(&self, a: Vec3, b: Vec3, ts: &mut Vec<f64>, hits: &mut Vec<SegmentHit>) {
        hits.clear();
        let (av, bv) = match (self.world_to_voxel(a), self.world_to_voxel(b)) {
            (Some(x), Some(y)) => (x, y),
            _ => return,
        };
        let world_len = mat::norm(mat::sub(b, a));
        if world_len < 1e-12 {
            return;
        }
        // Exact early-out: a segment whose two endpoints lie beyond the same face of the grid on
        // any axis cannot enter it (the grid is convex). Skips the boundary walk and its
        // allocations for the bulk of a tractogram when the grid is a slab of the head.
        for i in 0..3 {
            let n = self.dims[i] as f64;
            if (av[i] < 0.0 && bv[i] < 0.0) || (av[i] >= n && bv[i] >= n) {
                return;
            }
        }
        let dv = [bv[0] - av[0], bv[1] - av[1], bv[2] - av[2]];

        // all t in (0,1) where the ray crosses an integer voxel boundary, on any axis
        ts.clear();
        ts.push(0.0);
        ts.push(1.0);
        for i in 0..3 {
            if dv[i].abs() <= 1e-12 {
                continue;
            }
            let (lo, hi) = (av[i].min(bv[i]), av[i].max(bv[i]));
            let mut k = lo.floor() as i64 + 1;
            while (k as f64) < hi {
                let t = ((k as f64) - av[i]) / dv[i];
                if t > 1e-12 && t < 1.0 - 1e-12 {
                    ts.push(t);
                }
                k += 1;
            }
        }
        ts.sort_by(|x, y| x.partial_cmp(y).unwrap());

        for w in ts.windows(2) {
            let (t0, t1) = (w[0], w[1]);
            if t1 - t0 <= 1e-12 {
                continue;
            }
            let tm = 0.5 * (t0 + t1);
            let vx = (av[0] + tm * dv[0]).floor();
            let vy = (av[1] + tm * dv[1]).floor();
            let vz = (av[2] + tm * dv[2]).floor();
            if vx < 0.0 || vy < 0.0 || vz < 0.0 {
                continue;
            }
            let (vx, vy, vz) = (vx as usize, vy as usize, vz as usize);
            if vx >= self.dims[0] || vy >= self.dims[1] || vz >= self.dims[2] {
                continue;
            }
            hits.push(SegmentHit { voxel: [vx, vy, vz], length: (t1 - t0) * world_len });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1 mm isotropic identity grid, 10³.
    fn grid_1mm() -> Grid {
        Grid {
            dims: [10, 10, 10],
            voxel_to_world: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    #[test]
    fn axis_aligned_segment_lengths_sum_to_segment_length() {
        let g = grid_1mm();
        // from (0.5, 0.5, 0.5) to (3.5, 0.5, 0.5): crosses voxels x=0,1,2,3 along x
        let hits = g.intersect_segment([0.5, 0.5, 0.5], [3.5, 0.5, 0.5]);
        let total: f64 = hits.iter().map(|h| h.length).sum();
        assert!((total - 3.0).abs() < 1e-9, "total {total}");
        assert_eq!(hits.len(), 4);
        assert_eq!(hits[0].voxel, [0, 0, 0]);
        assert!((hits[0].length - 0.5).abs() < 1e-9); // 0.5→1.0
        assert!((hits[1].length - 1.0).abs() < 1e-9); // full voxel
        assert!((hits[3].length - 0.5).abs() < 1e-9); // 3.0→3.5
    }

    #[test]
    fn total_length_equals_euclidean_for_oblique_ray() {
        let g = grid_1mm();
        let a = [0.3, 0.2, 0.1];
        let b = [4.7, 3.3, 2.9];
        let hits = g.intersect_segment(a, b);
        let total: f64 = hits.iter().map(|h| h.length).sum();
        let euclid = mat::norm(mat::sub(b, a));
        assert!((total - euclid).abs() < 1e-9, "sum {total} vs {euclid}");
    }

    #[test]
    fn respects_anisotropic_voxels() {
        // 2 mm voxels along x: a 4 mm world segment covers 2 voxels (2 mm each), not 4.
        let mut g = grid_1mm();
        g.voxel_to_world[0][0] = 2.0;
        let hits = g.intersect_segment([0.0, 0.5, 0.5], [4.0, 0.5, 0.5]); // world 0→4 mm
        let total: f64 = hits.iter().map(|h| h.length).sum();
        assert!((total - 4.0).abs() < 1e-9, "total {total}"); // length is the world metric
        assert_eq!(hits.len(), 2, "should span 2 voxels of 2 mm");
        assert_eq!(hits[0].voxel, [0, 0, 0]);
        assert_eq!(hits[1].voxel, [1, 0, 0]);
        assert!((hits[0].length - 2.0).abs() < 1e-9);
    }

    #[test]
    fn out_of_bounds_dropped() {
        let g = grid_1mm();
        let hits = g.intersect_segment([-5.0, 0.5, 0.5], [-1.0, 0.5, 0.5]);
        assert!(hits.is_empty());
    }
}
