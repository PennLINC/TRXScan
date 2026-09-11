//! Voxel-axis reorientation to the FSL / dcm2niix convention (radiological LAS).
//!
//! Real datasets reach FSL (topup/eddy/fugue) and qsiprep via dcm2niix, which stores NIfTIs
//! radiologically (det<0) with a BIDS `PhaseEncodingDirection` that indexes the data axes *as
//! stored*. TRXScan otherwise inherits the tissue-map orientation of its inputs (an atlas grid is
//! typically LPS+/neurological), so its `j`-labelled output distorts opposite to what FSL expects
//! and flips again under any conform-to-canonical step. This module re-stores a finished volume in
//! LAS — a world-space no-op (same image, reindexed) that, because EPI SDC is defined in the
//! data-array frame, makes the declared PE axis and its distortion sign agree with FSL.
//!
//! Pure std (no `io` feature): the array/affine/bvec/PED maths is unit-tested here; `io` only
//! applies it when writing.

/// A signed-permutation reorientation of voxel axes: output axis `o` is fed by input axis `src[o]`,
/// reversed when `flip[o]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reorient {
    pub src: [usize; 3],
    pub flip: [bool; 3],
    pub in_dims: [usize; 3],
    pub out_dims: [usize; 3],
}

impl Reorient {
    /// The no-op reorientation (native output): axes and signs unchanged.
    pub fn identity(dims: [usize; 3]) -> Reorient {
        Reorient { src: [0, 1, 2], flip: [false, false, false], in_dims: dims, out_dims: dims }
    }

    /// Reorient an (axis-aligned) voxel→world affine to LAS — radiological, `+i→L, +j→A, +k→S`,
    /// the dcm2niix/FSL standard. Assumes the affine's 3×3 is a signed permutation (true for every
    /// standard scanner/template orientation; oblique affines are not the target here).
    pub fn to_las(affine: &[[f64; 4]; 4], dims: [usize; 3]) -> Reorient {
        // For each INPUT axis, which world axis it maps to (RAS+: 0=R/x, 1=A/y, 2=S/z) and its sign.
        let mut in_world = [0usize; 3];
        let mut in_sign = [1i32; 3];
        for a in 0..3 {
            let (mut w, mut best) = (0usize, 0.0f64);
            for j in 0..3 {
                if affine[j][a].abs() > best {
                    best = affine[j][a].abs();
                    w = j;
                }
            }
            in_world[a] = w;
            in_sign[a] = if affine[w][a] < 0.0 { -1 } else { 1 };
        }
        // Desired OUTPUT: axis 0→L(world0,−), axis 1→A(world1,+), axis 2→S(world2,+).
        let want_world = [0usize, 1, 2];
        let want_sign = [-1i32, 1, 1];
        let mut src = [0usize; 3];
        let mut flip = [false; 3];
        let mut out_dims = [0usize; 3];
        for o in 0..3 {
            let a = (0..3).find(|&a| in_world[a] == want_world[o])
                .expect("affine 3x3 is not a permutation of the world axes");
            src[o] = a;
            flip[o] = in_sign[a] != want_sign[o];
            out_dims[o] = dims[a];
        }
        Reorient { src, flip, in_dims: dims, out_dims }
    }

    fn is_noop(&self) -> bool {
        self.src == [0, 1, 2] && self.flip == [false, false, false]
    }

    /// Voxel→world affine after reorientation (world coordinates of every voxel are preserved).
    pub fn apply_affine(&self, a: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
        // v_in = S·v_out + t_v, with S[in][out] = ±1 and t_v[in] = (out_dims-1) on flipped axes.
        let mut s = [[0.0f64; 3]; 3];
        let mut tv = [0.0f64; 3];
        for o in 0..3 {
            let sign = if self.flip[o] { -1.0 } else { 1.0 };
            s[self.src[o]][o] = sign;
            if self.flip[o] {
                tv[self.src[o]] = (self.out_dims[o] - 1) as f64;
            }
        }
        let mut out = [[0.0f64; 4]; 4];
        out[3][3] = 1.0;
        for r in 0..3 {
            // linear part: A_out3 = A_in3 · S
            for o in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += a[r][k] * s[k][o];
                }
                out[r][o] = acc;
            }
            // translation: b_out = A_in3 · t_v + b_in
            let mut acc = a[r][3];
            for k in 0..3 {
                acc += a[r][k] * tv[k];
            }
            out[r][3] = acc;
        }
        out
    }

    /// Remap a 4D volume in layout `(x + nx*(y + ny*z))*ngrad + g` into the reoriented grid.
    pub fn apply_volume(&self, data: &[f32], ngrad: usize) -> Vec<f32> {
        if self.is_noop() {
            return data.to_vec();
        }
        let [nxi, nyi, _nzi] = self.in_dims;
        let [nxo, nyo, nzo] = self.out_dims;
        let mut out = vec![0.0f32; nxo * nyo * nzo * ngrad];
        let mut oc = [0usize; 3];
        for oz in 0..nzo {
            oc[2] = oz;
            for oy in 0..nyo {
                oc[1] = oy;
                for ox in 0..nxo {
                    oc[0] = ox;
                    // input coord per input axis a = src[o]
                    let mut ic = [0usize; 3];
                    for o in 0..3 {
                        let c = if self.flip[o] { self.out_dims[o] - 1 - oc[o] } else { oc[o] };
                        ic[self.src[o]] = c;
                    }
                    let in_base = (ic[0] + nxi * (ic[1] + nyi * ic[2])) * ngrad;
                    let out_base = (ox + nxo * (oy + nyo * oz)) * ngrad;
                    out[out_base..out_base + ngrad]
                        .copy_from_slice(&data[in_base..in_base + ngrad]);
                }
            }
        }
        out
    }

    /// Transform a bvec (image/voxel-frame gradient direction) consistently with the axis remap:
    /// reorder to the output axes and negate the flipped ones.
    pub fn apply_bvec(&self, b: [f64; 3]) -> [f64; 3] {
        let mut out = [0.0f64; 3];
        for o in 0..3 {
            out[o] = if self.flip[o] { -b[self.src[o]] } else { b[self.src[o]] };
        }
        out
    }

    /// BIDS `PhaseEncodingDirection` in the OUTPUT frame, given the PE data-axis and the signed
    /// direction a +off-resonance field displaces signal in the INPUT (grid) frame
    /// (`in_pe_sign = -1` means toward −axis). This carries the true acquisition physics through
    /// the reorientation so the written label matches what SDC tools must undo.
    pub fn out_ped(&self, in_pe_axis: usize, in_pe_sign: i32) -> String {
        let o = (0..3).find(|&o| self.src[o] == in_pe_axis).expect("pe axis not in permutation");
        let out_sign = in_pe_sign * if self.flip[o] { -1 } else { 1 };
        let axis = ['i', 'j', 'k'][o];
        if out_sign < 0 { format!("{axis}-") } else { axis.to_string() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LPS: [[f64; 4]; 4] =
        [[-1.0, 0.0, 0.0, 96.0], [0.0, -1.0, 0.0, 96.0], [0.0, 0.0, 1.0, -78.0], [0.0, 0.0, 0.0, 1.0]];

    #[test]
    fn lps_to_las_is_a_j_flip() {
        let r = Reorient::to_las(&LPS, [4, 5, 6]);
        assert_eq!(r.src, [0, 1, 2]);
        assert_eq!(r.flip, [false, true, false]);
        assert_eq!(r.out_dims, [4, 5, 6]);
    }

    #[test]
    fn to_las_affine_is_radiological_and_preserves_world() {
        let r = Reorient::to_las(&LPS, [4, 5, 6]);
        let a = r.apply_affine(&LPS);
        // LAS: diag(-1,+1,+1); det<0 (radiological)
        assert!((a[0][0] + 1.0).abs() < 1e-9 && (a[1][1] - 1.0).abs() < 1e-9 && (a[2][2] - 1.0).abs() < 1e-9);
        let det = a[0][0] * a[1][1] * a[2][2];
        assert!(det < 0.0, "must be radiological, got det {det}");
        // world of input voxel (2,1,3) equals world of its reoriented voxel (2,3,3)
        let w_in = [LPS[0][0] * 2.0 + LPS[0][3], LPS[1][1] * 1.0 + LPS[1][3], LPS[2][2] * 3.0 + LPS[2][3]];
        let w_out = [a[0][0] * 2.0 + a[0][3], a[1][1] * 3.0 + a[1][3], a[2][2] * 3.0 + a[2][3]];
        for k in 0..3 {
            assert!((w_in[k] - w_out[k]).abs() < 1e-9, "world not preserved on axis {k}");
        }
    }

    #[test]
    fn volume_reindex_flips_j() {
        // ngrad=1, dims 2x3x1; mark input (0,0,0). After a j-flip it lands at output (0, ny-1, 0).
        let r = Reorient::to_las(&LPS, [2, 3, 1]);
        let mut data = vec![0.0f32; 2 * 3 * 1];
        data[0 + 2 * (0 + 3 * 0)] = 1.0; // input (x=0,y=0,z=0)
        let out = r.apply_volume(&data, 1);
        assert_eq!(out[0 + 2 * (2 + 3 * 0)], 1.0); // output (0, ny-1=2, 0)
        assert_eq!(out.iter().filter(|&&v| v != 0.0).count(), 1);
    }

    #[test]
    fn bvec_flips_the_j_component() {
        let r = Reorient::to_las(&LPS, [4, 5, 6]);
        assert_eq!(r.apply_bvec([0.1, 0.2, 0.3]), [0.1, -0.2, 0.3]);
    }

    #[test]
    fn ped_reflects_physics_and_reorientation() {
        // TRXScan physics: +field displaces toward −grid-j  => in_pe_axis=1, in_pe_sign=-1 (fwd).
        let las = Reorient::to_las(&LPS, [4, 5, 6]); // flips j
        assert_eq!(las.out_ped(1, -1), "j"); // forward: −grid-j, flipped → +out-j = "j"
        assert_eq!(las.out_ped(1, 1), "j-"); // reverse-PE
        // native (no reorient) keeps the grid frame: the honest label is "j-" for the forward scan
        let id = Reorient::identity([4, 5, 6]);
        assert_eq!(id.out_ped(1, -1), "j-");
        assert_eq!(id.out_ped(1, 1), "j");
    }

    #[test]
    fn transposed_orientation_permutes() {
        // an affine where +i→A(y), +j→S(z), +k→R(x): to_las must permute, not just flip.
        let aff = [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let r = Reorient::to_las(&aff, [3, 4, 5]);
        // output L(world0/x) is fed by input axis 2; A(world1/y) by input 0; S(world2/z) by input 1
        assert_eq!(r.src, [2, 0, 1]);
        let a = r.apply_affine(&aff);
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        assert!(det < 0.0, "radiological after reorient");
    }
}
