//! Gradient scheme: FSL `.bval`/`.bvec` reading, shell detection, and the Fiberfox
//! gradient encoding.
//!
//! **Fiberfox convention (important):** the diffusion signal models encode the b-value in the
//! *gradient norm*. From `SignalModels/mitkStickModel.cpp`, `signal = exp(-B0 * d * (f·g)^2)`
//! where `B0` is the baseline b-value (`m_BValue`) and `g` is the stored gradient. To make a
//! volume's effective b-value equal `b_i`, the stored gradient must be
//! `g_i = unit_bvec_i * sqrt(b_i / b_max)`, with `B0 = b_max`; then `|g_i|^2 = b_i / b_max`
//! and the effective b-value is `b_max * (b_i / b_max) = b_i`. (This is the `b2q=True` path in
//! PennBBL's `fiberfox-wrapper/simulate_scheme.py`.)
//!
//! This module is dependency-free (pure std).

use std::fmt;
use std::path::Path;

/// Default b-value (s/mm^2) below which a volume is treated as a b0.
pub const B0_THRESHOLD: f64 = 50.0;

#[derive(Debug)]
pub enum SchemeError {
    Io(std::io::Error),
    /// bvec token count was not 3 × (number of b-values).
    Shape { n_bvals: usize, n_bvec_tokens: usize },
    /// A token failed to parse as a float.
    Parse(String),
    Empty,
}

impl fmt::Display for SchemeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchemeError::Io(e) => write!(f, "io error: {e}"),
            SchemeError::Shape { n_bvals, n_bvec_tokens } => write!(
                f,
                "bvec has {n_bvec_tokens} tokens, expected 3 × {n_bvals} = {}",
                3 * n_bvals
            ),
            SchemeError::Parse(t) => write!(f, "could not parse number: {t:?}"),
            SchemeError::Empty => write!(f, "empty bval file"),
        }
    }
}
impl std::error::Error for SchemeError {}
impl From<std::io::Error> for SchemeError {
    fn from(e: std::io::Error) -> Self {
        SchemeError::Io(e)
    }
}

/// A diffusion gradient scheme: per-volume b-values and unit gradient directions.
#[derive(Debug, Clone)]
pub struct GradientScheme {
    /// b-values as read (s/mm^2), one per volume.
    pub bvals: Vec<f64>,
    /// Unit gradient directions; b0 volumes are `[0,0,0]`.
    pub bvecs: Vec<[f64; 3]>,
    /// Reference (maximum) b-value used as Fiberfox's `m_BValue`.
    pub b_max: f64,
}

impl GradientScheme {
    /// Read an FSL `.bval` + `.bvec` pair.
    pub fn from_fsl(bval_path: impl AsRef<Path>, bvec_path: impl AsRef<Path>) -> Result<Self, SchemeError> {
        let bval = std::fs::read_to_string(bval_path)?;
        let bvec = std::fs::read_to_string(bvec_path)?;
        Self::from_str(&bval, &bvec)
    }

    /// Parse from in-memory FSL-format strings. `.bval` is whitespace-separated b-values;
    /// `.bvec` is 3 rows (x, y, z) each with one entry per volume (row-major, whitespace-separated).
    pub fn from_str(bval: &str, bvec: &str) -> Result<Self, SchemeError> {
        let bvals = parse_floats(bval)?;
        if bvals.is_empty() {
            return Err(SchemeError::Empty);
        }
        let n = bvals.len();
        let toks = parse_floats(bvec)?;
        if toks.len() != 3 * n {
            return Err(SchemeError::Shape { n_bvals: n, n_bvec_tokens: toks.len() });
        }
        // FSL bvec is row-major: [x_0..x_{n-1}, y_0..y_{n-1}, z_0..z_{n-1}].
        let mut bvecs = Vec::with_capacity(n);
        for i in 0..n {
            let raw = [toks[i], toks[n + i], toks[2 * n + i]];
            let norm = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
            bvecs.push(if norm < 1e-6 {
                [0.0, 0.0, 0.0]
            } else {
                [raw[0] / norm, raw[1] / norm, raw[2] / norm]
            });
        }
        let b_max = bvals.iter().cloned().fold(0.0_f64, f64::max);
        Ok(GradientScheme { bvals, bvecs, b_max })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.bvals.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bvals.is_empty()
    }

    /// Whether volume `i` is a b0 (b-value below [`B0_THRESHOLD`]).
    #[inline]
    pub fn is_b0(&self, i: usize) -> bool {
        self.bvals[i] < B0_THRESHOLD
    }

    /// Distinct shells as `(rounded_bvalue, count)`, sorted ascending. `round_to` is the bin
    /// width in s/mm^2 (e.g. 50 collapses 995 and 1005 to 1000).
    pub fn shells(&self, round_to: f64) -> Vec<(i64, usize)> {
        let mut counts: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
        for &b in &self.bvals {
            let key = (b / round_to).round() as i64 * round_to as i64;
            *counts.entry(key).or_insert(0) += 1;
        }
        counts.into_iter().collect()
    }

    /// The Fiberfox-encoded gradient for volume `i`: `unit_bvec * sqrt(b_i / b_max)`
    /// (b0 → `[0,0,0]`). Feed these, plus `b_max` as the baseline b-value, to the signal models.
    pub fn fiberfox_gradient(&self, i: usize) -> [f64; 3] {
        if self.b_max <= 0.0 || self.is_b0(i) {
            return [0.0, 0.0, 0.0];
        }
        let s = (self.bvals[i] / self.b_max).sqrt();
        let g = self.bvecs[i];
        [g[0] * s, g[1] * s, g[2] * s]
    }

    /// All Fiberfox-encoded gradients.
    pub fn fiberfox_gradients(&self) -> Vec<[f64; 3]> {
        (0..self.len()).map(|i| self.fiberfox_gradient(i)).collect()
    }
}

fn parse_floats(s: &str) -> Result<Vec<f64>, SchemeError> {
    s.split_whitespace()
        .map(|t| t.parse::<f64>().map_err(|_| SchemeError::Parse(t.to_string())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // 5 volumes: b0, then two b=1000 and two b=3000 in orthogonal-ish directions.
    const BVAL: &str = "0 1000 1000 3000 3000";
    const BVEC: &str = "0 1 0 0 1\n\
                        0 0 1 0 0\n\
                        0 0 0 1 0";

    #[test]
    fn parses_and_normalizes() {
        let s = GradientScheme::from_str(BVAL, BVEC).unwrap();
        assert_eq!(s.len(), 5);
        assert_eq!(s.b_max, 3000.0);
        assert!(s.is_b0(0) && !s.is_b0(1));
        // b0 gradient is zeroed
        assert_eq!(s.bvecs[0], [0.0, 0.0, 0.0]);
        // unit norm on DWI volumes
        for i in 1..5 {
            let g = s.bvecs[i];
            let n = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
            assert!((n - 1.0).abs() < 1e-9, "vol {i} not unit: {n}");
        }
    }

    #[test]
    fn fiberfox_norm_encodes_bvalue() {
        let s = GradientScheme::from_str(BVAL, BVEC).unwrap();
        for i in 0..s.len() {
            let g = s.fiberfox_gradient(i);
            let norm2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
            // |g|^2 == b_i / b_max  ⇒  effective b = b_max * |g|^2 = b_i
            let expected = s.bvals[i] / s.b_max;
            assert!((norm2 - expected).abs() < 1e-9, "vol {i}: |g|^2={norm2} expected {expected}");
        }
    }

    #[test]
    fn shell_detection() {
        let s = GradientScheme::from_str(BVAL, BVEC).unwrap();
        assert_eq!(s.shells(50.0), vec![(0, 1), (1000, 2), (3000, 2)]);
    }

    #[test]
    fn rejects_bad_shape() {
        // 3 b-values need 9 bvec tokens; give 6.
        assert!(matches!(
            GradientScheme::from_str("0 1000 2000", "0 1 0\n0 0 0"),
            Err(SchemeError::Shape { .. })
        ));
    }
}
