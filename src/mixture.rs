//! The per-voxel Gaussian mixture — the shared object between the signal stage and the ground-truth
//! closed forms.
//!
//! [`MixtureField`] is what the rasterizer *actually knows* before it marginalizes: for every
//! voxel, the path-length-weighted orientation histogram of the streamline segments crossing it
//! (on [`crate::sphere::HemiSphere`] vertices), plus the normalized tissue fractions and the
//! compartment parameters. From it:
//!
//! - [`crate::compartments::signal_from_mixture`] evaluates the clean signal, and
//! - [`crate::microstructure`] evaluates the FORCE-style closed-form scalars,
//!
//! so signal and ground truth derive from one object and cannot disagree by construction.
//!
//! Layouts match the rest of the crate: 3D flat index `x + nx*(y + ny*z)`; the histogram is
//! `vox * nvert + v`. Histogram weights are **unnormalized** accumulations of
//! `w_s · length · seg_area` (per-streamline weight × world-mm path length in the voxel ×
//! fibre cross-section); consumers normalize per voxel. The per-voxel fibre *mixture* uses only
//! the row's relative weights — the absolute scale cancels, as it does in the signal stage.

use crate::compartments::CompartmentParams;
use crate::sphere::HemiSphere;

/// Per-voxel orientation histogram + normalized tissue fractions + compartment parameters.
pub struct MixtureField {
    pub dims: [usize; 3],
    pub sphere: HemiSphere,
    /// Orientation weights, `vox * nvert + v`, unnormalized (see module docs). All-zero rows in
    /// WM voxels mean "no streamline support" — see [`MixtureField::fallback`].
    pub odf: Vec<f32>,
    /// Normalized WM fraction per voxel (`wm + gm + csf = 1` inside the mask, all 0 outside).
    pub wm: Vec<f32>,
    /// Normalized GM fraction per voxel.
    pub gm: Vec<f32>,
    /// Normalized CSF fraction per voxel.
    pub csf: Vec<f32>,
    /// 1 where the voxel is masked, WM > 0, and the histogram row is (near-)empty. Both the
    /// signal and the scalars must treat that WM fraction as an **isotropic** Gaussian at
    /// [`MixtureField::md_fallback`] — the hindered fallback of `compartments.rs` — or truth
    /// and signal silently disagree exactly where the tractogram is sparse.
    pub fallback: Vec<u8>,
    pub params: CompartmentParams,
    /// Optional per-voxel **myelination fraction** (0..1): 0 = this field's own `params`
    /// (the unmyelinated endpoint), 1 = [`CompartmentParams::adult`] (the myelinated endpoint).
    /// Consumers lerp `intra_frac`/`d_intra`/`d_extra` per voxel — the geometric
    /// myelin-gradient variant of the tier-2 parameter maps. GM/CSF are
    /// untouched. Set after construction; `None` ⇒ uniform `params` everywhere.
    pub myelin: Option<Vec<f32>>,
}

impl MixtureField {
    #[inline]
    pub fn nvox(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }
    #[inline]
    pub fn nvert(&self) -> usize {
        self.sphere.len()
    }
    /// The voxel's histogram row.
    #[inline]
    pub fn odf_row(&self, vox: usize) -> &[f32] {
        let n = self.nvert();
        &self.odf[vox * n..(vox + 1) * n]
    }
    /// Inside the brain mask (some tissue present)?
    #[inline]
    pub fn is_masked(&self, vox: usize) -> bool {
        self.wm[vox] + self.gm[vox] + self.csf[vox] > 0.0
    }
    /// Mean diffusivity of the extra-axonal tensor — the diffusivity of the isotropic hindered
    /// fallback used for WM voxels with no streamline support (`compartments.rs`).
    #[inline]
    pub fn md_fallback(&self) -> f64 {
        let (d1, d2, d3) = self.params.d_extra;
        (d1 + d2 + d3) / 3.0
    }
}
