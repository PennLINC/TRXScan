//! Readout schemes: `tick → (kx, ky)` k-space trajectory and per-tick timing.
//!
//! [`SingleShotEpi`] is a faithful port of `Sequences/mitkSingleShotEpi.h` (implemented + tested).
//! Fast/Conventional spin-echo (`mitkFastSpinEcho.h`, `mitkConventionalSpinEcho.h`) can follow the
//! same trait later. The `AcquisitionType` interface is purely in-plane — slice timing for
//! multiband lives in `motion`/`kspace` (see `docs/FEATURES.md`).

/// In-plane readout trajectory + timing. `tick` runs `0..kx_max*ky_max` in acquisition order.
pub trait Readout {
    fn kspace_index(&self, tick: usize) -> (usize, usize);
    /// ms from the maximum-echo (k-space centre) time.
    fn time_from_max_echo(&self, tick: usize) -> f64;
    /// ms since readout start (used for eddy-current decay).
    fn time_from_last_diffusion_gradient(&self, tick: usize) -> f64;
    /// ms since the RF pulse (used for T2/T2* relaxation).
    fn time_from_rf(&self, tick: usize) -> f64;
}

/// Single-shot EPI: one echo, max intensity at k-space centre, zig-zag ("snake") trajectory.
#[derive(Debug, Clone, Copy)]
pub struct SingleShotEpi {
    pub kx_max: usize,
    pub ky_max: usize,
    /// time to read one PE line (ms)
    pub t_line: f64,
    /// echo time (ms)
    pub t_echo: f64,
    pub reverse_phase: bool,
}

impl SingleShotEpi {
    /// time to read one k-space sample (ms)
    #[inline]
    pub fn dt(&self) -> f64 {
        self.t_line / self.kx_max as f64
    }
    #[inline]
    fn half_read_time(&self) -> f64 {
        (self.kx_max * self.ky_max) as f64 * self.dt() / 2.0
    }
}

impl Readout for SingleShotEpi {
    fn kspace_index(&self, tick: usize) -> (usize, usize) {
        let mut x = tick % self.kx_max;
        let mut y = tick / self.kx_max;
        if !self.reverse_phase {
            y = self.ky_max - 1 - y; // start at the maximum k-space line
            if y % 2 == 1 {
                x = self.kx_max - x - 1; // reverse frequency-encode direction on odd lines
            }
        } else if y % 2 == 1 {
            x = self.kx_max - x - 1;
        }
        (x, y)
    }

    fn time_from_max_echo(&self, tick: usize) -> f64 {
        self.dt() * (tick as f64 + 0.5) - self.half_read_time()
    }
    fn time_from_last_diffusion_gradient(&self, tick: usize) -> f64 {
        tick as f64 * self.dt() + self.dt() / 2.0
    }
    fn time_from_rf(&self, tick: usize) -> f64 {
        self.t_echo + self.time_from_max_echo(tick)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_is_a_permutation() {
        // every (kx,ky) cell is visited exactly once, both phase polarities.
        for reverse in [false, true] {
            let epi = SingleShotEpi { kx_max: 6, ky_max: 5, t_line: 1.0, t_echo: 90.0, reverse_phase: reverse };
            let mut seen = vec![false; epi.kx_max * epi.ky_max];
            for tick in 0..epi.kx_max * epi.ky_max {
                let (x, y) = epi.kspace_index(tick);
                assert!(x < epi.kx_max && y < epi.ky_max);
                let flat = y * epi.kx_max + x;
                assert!(!seen[flat], "cell ({x},{y}) visited twice");
                seen[flat] = true;
            }
            assert!(seen.iter().all(|&v| v), "trajectory missed a cell");
        }
    }

    #[test]
    fn echo_time_is_monotonic() {
        let epi = SingleShotEpi { kx_max: 8, ky_max: 8, t_line: 1.0, t_echo: 90.0, reverse_phase: false };
        let n = epi.kx_max * epi.ky_max;
        for tick in 1..n {
            assert!(epi.time_from_max_echo(tick) > epi.time_from_max_echo(tick - 1));
        }
    }
}
