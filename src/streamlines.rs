//! Streamline-set utilities that need no I/O: weighted subsampling.
//!
//! Lives in the std-only core so the CLI, `trxscan-microstructure` and the Python bindings all
//! select the **identical** subset for the same `(n, seed)` — the guarantee that ground truth and
//! simulated data describe the same phantom.

/// What [`subsample_streamlines`] did, for logging.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubsampleStats {
    /// Streamlines kept (`min(n, total)`).
    pub kept: usize,
    /// Streamlines in the input.
    pub total: usize,
    /// Fraction of vertices kept.
    pub vertex_fraction: f64,
    /// Fraction of the input weight mass kept (1.0 when unweighted).
    pub weight_fraction: f64,
}

impl SubsampleStats {
    /// The one-line summary the binaries print.
    pub fn summary(&self, seed: u64) -> String {
        if self.kept >= self.total {
            format!("subsample: {} >= {} streamlines, keeping all", self.kept, self.total)
        } else {
            format!(
                "subsample: kept {}/{} streamlines (seed {seed}), {:.1}% of vertices, {:.1}% of weight -> uniform",
                self.kept, self.total, 100.0 * self.vertex_fraction, 100.0 * self.weight_fraction,
            )
        }
    }
}

/// Keep `n` streamlines, sampled without replacement with probability proportional to the SIFT2
/// weight (uniform when unweighted): Efraimidis-Spirakis exponential keys from a SplitMix64 hashed
/// per streamline index, so the same `n` and `seed` select the same subset in every caller.
/// Survivors get uniform weights (total/n), keeping the weighted density unbiased in expectation.
/// Selecting the top-n *by weight* instead would gut over-tracked bundles: SIFT2 weights are
/// tight around 1 and high weight means under-tracked, not important.
///
/// `n >= total` keeps everything unchanged (weights included).
pub fn subsample_streamlines(
    positions: Vec<[f64; 3]>,
    offsets: Vec<u32>,
    weights: Option<Vec<f32>>,
    n: usize,
    seed: u64,
) -> (Vec<[f64; 3]>, Vec<u32>, Option<Vec<f32>>, SubsampleStats) {
    let total = offsets.len().saturating_sub(1);
    if n >= total {
        let stats = SubsampleStats { kept: total, total, vertex_fraction: 1.0, weight_fraction: 1.0 };
        return (positions, offsets, weights, stats);
    }
    let mut keys: Vec<(f64, u32)> = (0..total as u32)
        .map(|i| {
            let mut z = (seed ^ (i as u64).wrapping_mul(0xA24B_AED4_963E_E407))
                .wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64; // in (0,1)
            let w = weights.as_ref().map_or(1.0, |w| (w[i as usize] as f64).max(1e-12));
            (u.ln() / w, i)
        })
        .collect();
    keys.select_nth_unstable_by(n - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut idx: Vec<u32> = keys[..n].iter().map(|k| k.1).collect();
    idx.sort_unstable();

    let total_w: f64 = weights.as_ref().map_or(total as f64, |w| w.iter().map(|&x| x as f64).sum());
    let kept_w: f64 = weights
        .as_ref()
        .map_or(n as f64, |w| idx.iter().map(|&i| w[i as usize] as f64).sum());
    let mut new_pos = Vec::new();
    let mut new_off = Vec::with_capacity(n + 1);
    new_off.push(0u32);
    for &i in &idx {
        let (s0, e0) = (offsets[i as usize] as usize, offsets[i as usize + 1] as usize);
        new_pos.extend_from_slice(&positions[s0..e0]);
        new_off.push(new_pos.len() as u32);
    }
    let stats = SubsampleStats {
        kept: n,
        total,
        vertex_fraction: new_pos.len() as f64 / positions.len().max(1) as f64,
        weight_fraction: kept_w / total_w,
    };
    let new_weights = weights.map(|_| vec![(total_w / n as f64) as f32; n]);
    (new_pos, new_off, new_weights, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy(n: usize) -> (Vec<[f64; 3]>, Vec<u32>) {
        let mut pos = Vec::new();
        let mut off = vec![0u32];
        for s in 0..n {
            for k in 0..3 {
                pos.push([s as f64, k as f64, 0.0]);
            }
            off.push(pos.len() as u32);
        }
        (pos, off)
    }

    #[test]
    fn subsample_is_deterministic_by_seed_and_n() {
        let (pos, off) = toy(50);
        let a = subsample_streamlines(pos.clone(), off.clone(), None, 10, 7);
        let b = subsample_streamlines(pos.clone(), off.clone(), None, 10, 7);
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
        assert_eq!(a.3, b.3);
        let c = subsample_streamlines(pos, off, None, 10, 8);
        assert_ne!(a.0, c.0, "a different seed must select a different subset");
        assert_eq!(a.3.kept, 10);
        assert_eq!(a.3.total, 50);
        assert!((a.3.vertex_fraction - 0.2).abs() < 1e-12);
    }

    #[test]
    fn keeping_everything_is_the_identity() {
        let (pos, off) = toy(5);
        let w = Some(vec![0.5f32; 5]);
        let (p, o, ww, st) = subsample_streamlines(pos.clone(), off.clone(), w.clone(), 99, 0);
        assert_eq!((p, o, ww), (pos, off, w));
        assert_eq!(st.kept, 5);
        assert_eq!(st.weight_fraction, 1.0);
    }

    #[test]
    fn survivors_get_uniform_weights_preserving_total_mass() {
        let (pos, off) = toy(40);
        let w: Vec<f32> = (0..40).map(|i| 0.5 + (i % 4) as f32 * 0.5).collect();
        let total: f64 = w.iter().map(|&x| x as f64).sum();
        let (_, o, ww, st) = subsample_streamlines(pos, off, Some(w), 8, 3);
        let ww = ww.unwrap();
        assert_eq!(o.len(), 9);
        assert_eq!(ww.len(), 8);
        let kept: f64 = ww.iter().map(|&x| x as f64).sum();
        assert!((kept - total).abs() < 1e-3);
        assert!(st.weight_fraction > 0.0 && st.weight_fraction < 1.0);
    }
}
