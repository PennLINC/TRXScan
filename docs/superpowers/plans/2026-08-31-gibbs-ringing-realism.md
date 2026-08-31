# Gibbs Ringing Realism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace TRXScan's hand-injected k-space truncation with a physically correct finite-acquisition forward model that produces intrinsic Gibbs ringing at the nominal resolution in genuinely complex-valued data, with a scoreable artifact-free reference.

**Architecture:** Simulate the object on a finer in-plane grid, give it a real complex phase, then evaluate *only* the k-space samples the scanner actually acquires — cropping during the forward transform rather than computing a large k-space and discarding it. Reconstruction stays at the nominal matrix, so ringing arises from the acquisition itself rather than from an added filter. A block-mean reduction of the same complex realization provides the artifact-free scoring target.

**Tech Stack:** Rust 2021, pure `std` (no features required for any code in this plan). Tests are inline `#[cfg(test)] mod tests` blocks, run with `cargo test`. Complex arithmetic is hand-rolled `f64` pairs, matching the existing `kspace.rs` style.

**Spec:** `docs/superpowers/specs/2026-08-31-gibbs-ringing-realism-design.md` (revision 6, frozen)

**Scope:** Spec phases 0-7. Phases 8-10 (real-data calibration, the unringing-method acceptance suite, FFT/streaming optimization) are a separate plan; they depend on external datasets and on this plan being complete.

## Global Constraints

- **Pure std.** Every module touched here must compile with `cargo build` (no features). Do not add dependencies. `rustfft` is explicitly out of scope (spec 2, non-goals).
- **Direct sums stay the oracle.** Do not replace the O(N^3) DFT with an FFT. Spec `kspace.rs:12-14` documents the deliberate avoidance of FFT-convention ambiguity.
- **In-plane oversampling only.** Never oversample z; the slice direction is not Fourier-encoded in 2D EPI (spec 3.1).
- **Motion disabled** for every task in this plan (spec 6). Do not modify `motion.rs` or the `generate_compartments_moving` path.
- **Noise convention:** `noise_variance` is the **per-component** variance of the reconstructed complex image under full sampling, single-coil, pre-combination. `Var(Re n) = Var(Im n) = noise_variance`, so `E[|n|^2] = 2 * noise_variance`. Per-k-space-sample variance is `noise_variance / (nx*ny)` (spec 3.3).
- **`object-nominal` is a block MEAN**, `(1/o^2) * sum_j z_j`, never a sum. A sum scales intensities by `o^2` (spec 3.5.1).
- **No band-limit requirement** is imposed on the object or phase model on either grid. Adequacy is established by acquired-band convergence only (spec 3.2, 4.1.6).
- **`q_eff` is an effective q-vector:** `q_eff = c_q * sqrt(b) * bvec_unit`. TRXScan has no `delta`/`Delta`/waveform parameters; `c_q` absorbs the diffusion-timing and radian/cycle conventions. Never call it the physical q-vector (spec 3.2).
- **Integer oversampling ratio** `o = snx/nx = sny/ny`, enforced by assertion. This is an implementation restriction for voxel-subdivision clarity and parity safety, not a Fourier requirement (spec 3.1).
- Commit after every task. Run `cargo test` before each commit.

## Working Tree State (read before Task 1)

This branch was merged with `main` after the spec was written. Three things will look wrong to a
fresh reader and are **not** yours to fix:

1. **One test already fails, for an environment reason.**
   `microstructure::tests::matches_dipy_closed_forms_on_fixtures` panics with
   `run tools/gen_force_fixtures.py first: No such file or directory`. It arrived from `main`
   (commit `668dfea`) and needs a fixture-generation script that has not been run here. Baseline is
   **48 passed, 1 failed**.
   - Do not fix it, do not delete it, do not run the fixture generator.
   - Where a step says "Run: `cargo test`", the pass criterion is *no new failures beyond that one*.
   - Prefer the scoped form while iterating: `cargo test --lib analytic`, `--lib kspace`,
     `--lib phase`, `--lib benchmark`.

2. **Every tracked file shows as modified, with no content change.** The working tree is CRLF, the
   committed blobs are LF, and `core.autocrlf=false`, so `git status` reports all 24 files as
   modified. This predates the work and is deliberately left alone.
   - **Never `git add -A` or `git add .`** — it would commit thousands of lines of line-ending
     churn into `PennLINC/TRXScan`. Add only the exact paths each task's commit step names.

3. **New modules exist that this plan does not touch:** `src/sphere.rs`, `src/mixture.rs`,
   `src/microstructure.rs`, and `src/bin/trxscan_microstructure.rs`. They are independent of the
   k-space forward model. The only file outside `src/kspace.rs` that reads `zero_ringing` is
   `src/bin/trxscan.rs:261`, so Task 12's removal has exactly one call site to update.

All `file:line` citations in this plan and in the spec were re-verified against the merged tree.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/analytic.rs` (new) | Analytic truncated-Fourier-series references used as test oracles. No simulator dependencies. |
| `src/phase.rs` (new) | `PhaseModel`: global, pre-readout 3D background field, per-shot diffusion phase from `q_eff`. Pure math, no k-space knowledge. |
| `src/benchmark.rs` (new) | Block-mean reduction and the benchmark output bundle. Knows about grids, not about acquisition. |
| `src/kspace.rs` (modify) | The forward model: sim-grid crop, sampling mask, window, noise. The only file that knows the acquisition. |
| `src/lib.rs` (modify) | Declare the three new modules. |
| `data/trxscan_truth_data/scripts/prepare_acquisition_grid.py` (modify) | `--oversample N` to emit sim-grid tissue maps and fieldmap. |

`src/compartments.rs`, `src/raster.rs` and `src/readout.rs` are **not modified**: Stage A already accepts an arbitrary `Grid`, so oversampling is achieved by passing a finer one, and `SingleShotEpi` correctly stays on the acquired matrix.

---

### Task 1: Analytic reference oracle

**Files:**
- Create: `src/analytic.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `analytic::truncated_step_profile(n: usize, x0: f64) -> Vec<f64>` — the reconstructed profile of a continuous unit step located at `x0` (in units of the unit FOV, so `x0 in [0,1)`), truncated to `n` centred Fourier coefficients and sampled at the `n` voxel centres `(j+0.5)/n`.

- [ ] **Step 1: Write the failing test**

Add to `src/analytic.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Max value above 1.0 among samples on the bright side of the edge.
    fn overshoot(prof: &[f64], n: usize) -> f64 {
        prof[n / 2 + 1..].iter().cloned().fold(f64::MIN, f64::max) - 1.0
    }

    #[test]
    fn overshoot_depends_on_subvoxel_edge_position() {
        let n = 64;
        // Edge at a voxel boundary: the sampled overshoot is far below the Gibbs constant.
        let at_boundary = overshoot(&truncated_step_profile(n, 32.0 / n as f64), n);
        assert!(
            (at_boundary - 0.0119).abs() < 0.002,
            "boundary offset overshoot {at_boundary:.4}, expected ~0.0119"
        );
        // Edge at a voxel centre: the sampled overshoot reaches the ~8.95% Gibbs constant.
        let at_centre = overshoot(&truncated_step_profile(n, 32.5 / n as f64), n);
        assert!(
            (at_centre - 0.0893).abs() < 0.002,
            "centre offset overshoot {at_centre:.4}, expected ~0.0893"
        );
        // This 7.5x spread is exactly why a fixed-overshoot test is invalid (spec 4.1.1).
        assert!(at_centre > 5.0 * at_boundary);
    }

    #[test]
    fn sidelobes_alternate_sign_regardless_of_offset() {
        let n = 64;
        for &x0 in &[32.0 / 64.0, 32.5 / 64.0] {
            let p = truncated_step_profile(n, x0);
            for j in 0..8 {
                let a = p[n / 2 + 1 + j] - 1.0;
                let b = p[n / 2 + 2 + j] - 1.0;
                assert!(a * b < 0.0, "lobes {j} and {} share a sign at x0={x0}", j + 1);
            }
        }
    }

    #[test]
    fn far_from_the_edge_the_profile_is_flat() {
        let n = 128;
        let p = truncated_step_profile(n, 64.0 / n as f64);
        assert!((p[n - 2] - 1.0).abs() < 0.01, "bright plateau {}", p[n - 2]);
        assert!(p[1].abs() < 0.01, "dark plateau {}", p[1]);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib analytic`
Expected: FAIL — `cannot find function truncated_step_profile` / `file not found for module analytic`.

- [ ] **Step 3: Write minimal implementation**

Put this at the top of `src/analytic.rs`, above the test module:

```rust
//! Analytic references for finite-Fourier acquisition, used as test oracles.
//!
//! These are closed-form results, independent of the simulator, so a disagreement between
//! [`truncated_step_profile`] and `kspace::simulate_slice` always indicts the simulator.
//!
//! Convention matches `kspace.rs`: `n` centred coefficients `k = -n/2 ..= n/2-1` (asymmetric by
//! one sample for even `n`, as real even-matrix Cartesian acquisitions are), reconstructed at the
//! `n` voxel centres `(j+0.5)/n` over a unit FOV.

use std::f64::consts::TAU;

/// Reconstructed profile of a continuous unit step: `f(x) = 0` for `x < x0`, `1` for `x >= x0`,
/// on periodic `[0,1)`, truncated to `n` centred Fourier coefficients, sampled at voxel centres.
///
/// The Fourier coefficients of the step are, for `k != 0`,
/// `c_k = (1 - exp(-i*TAU*k*x0)) / (-i*TAU*k)`, and `c_0 = 1 - x0`.
pub fn truncated_step_profile(n: usize, x0: f64) -> Vec<f64> {
    let half = (n / 2) as i64;
    (0..n)
        .map(|j| {
            let x = (j as f64 + 0.5) / n as f64;
            let mut acc = 1.0 - x0; // k = 0 term; real
            for k in -half..half {
                if k == 0 {
                    continue;
                }
                let kf = k as f64;
                let w = TAU * kf;
                // numerator 1 - exp(-i*w*x0) = (1 - cos(w*x0)) + i*sin(w*x0)
                let (s, c) = (w * x0).sin_cos();
                let (nr, ni) = (1.0 - c, s);
                // divide by -i*w  =>  multiply by i/w
                let (cr, ci) = (-ni / w, nr / w);
                // multiply by exp(i*w*x) and keep the real part
                let (s2, c2) = (w * x).sin_cos();
                acc += cr * c2 - ci * s2;
            }
            acc
        })
        .collect()
}
```

Add to `src/lib.rs`, immediately after the `pub mod mat;` line:

```rust
/// Analytic Fourier references used as test oracles (spec 4.1).
pub mod analytic;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib analytic`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/analytic.rs src/lib.rs
git commit -m "feat(analytic): truncated Fourier step-response oracle"
```

---

### Task 2: Sim-grid crop in the forward transform

**Files:**
- Modify: `src/kspace.rs` (the `simulate_slice` signature and its forward loops)
- Test: `src/kspace.rs` (inline `mod tests`)

**Interfaces:**
- Consumes: `analytic::truncated_step_profile` from Task 1.
- Produces:
  - `pub struct SliceInput<'a>` with fields `compartments: &'a [&'a [f32]]`, `t2: &'a [f32]`, `fmap: &'a [f32]`, `phase0: Option<&'a [f64]>`, `sim: [usize; 2]`, `acq_matrix: [usize; 2]`, `z: usize`, `nz: usize`, `bvec: [f64; 3]`, `bval: f64`, `slice_seed: u64`.
  - `pub fn simulate_slice(inp: &SliceInput, acq: &Acquisition) -> Vec<(f32, f32)>` returning `nx*ny` complex samples.
  - `pub fn step_hires(snx: usize, sny: usize, edge: f64) -> Vec<f32>` (test helper, `pub` so later tasks reuse it).

`phase0` is accepted but ignored until Task 6; pass `None` everywhere in this task.

- [ ] **Step 1: Write the failing test**

Add to the `mod tests` block in `src/kspace.rs`:

```rust
use crate::analytic::truncated_step_profile;

/// Acquisition with every artifact off: pure finite-Fourier acquisition.
fn clean(nx: usize, ny: usize) -> Acquisition {
    Acquisition {
        signal_scale: 1.0,
        do_distortions: false,
        do_relaxation: false,
        noise_variance: 0.0,
        partial_fourier: 1.0,
        ghost_offset: 0.0,
        eddy_strength: 0.0,
        n_coils: 1,
        accel: 1,
        ..Acquisition::default_for(nx, ny)
    }
}

#[test]
fn crop_reproduces_the_analytic_profile_across_subvoxel_offsets() {
    let (nx, ny, o) = (32usize, 32usize, 8usize);
    let (snx, sny) = (nx * o, ny * o);
    let fmap = vec![0.0f32; snx * sny];
    let acq = clean(nx, ny);

    for i in 0..16 {
        let delta = i as f64 / 16.0;
        // Edge at (nx/2 + delta) acquired voxels, expressed in sim-voxel units.
        let edge = (nx as f64 / 2.0 + delta) * o as f64;
        let img = step_hires(snx, sny, edge);
        let comps: [&[f32]; 1] = [&img];
        let inp = SliceInput {
            compartments: &comps,
            t2: &[100.0],
            fmap: &fmap,
            phase0: None,
            sim: [snx, sny],
            acq_matrix: [nx, ny],
            z: 0,
            nz: 1,
            bvec: [0.0, 0.0, 0.0],
            bval: 0.0,
            slice_seed: 0,
        };
        let out = simulate_slice(&inp, &acq);

        let expect = truncated_step_profile(nx, (nx as f64 / 2.0 + delta) / nx as f64);
        let row = ny / 2;
        let worst = (0..nx)
            .map(|x| (out[x + nx * row].0 as f64 - expect[x]).abs())
            .fold(0.0f64, f64::max);
        assert!(worst < 5e-3, "offset {delta}: max profile deviation {worst:.4e}");
    }
}

#[test]
fn ringing_is_intrinsic_without_any_ringing_parameter() {
    // A sub-voxel-positioned edge must now ring: the old exact-DFT round-trip is gone.
    let (nx, ny, o) = (32usize, 32usize, 8usize);
    let (snx, sny) = (nx * o, ny * o);
    let fmap = vec![0.0f32; snx * sny];
    let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
    let comps: [&[f32]; 1] = [&img];
    let inp = SliceInput {
        compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: None,
        sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1,
        bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 0,
    };
    let out = simulate_slice(&inp, &clean(nx, ny));
    let row = ny / 2;
    let peak = (nx / 2 + 1..nx).map(|x| out[x + nx * row].0 as f64).fold(f64::MIN, f64::max);
    assert!(peak > 1.05, "expected intrinsic overshoot, got peak {peak:.4}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib kspace`
Expected: FAIL — `cannot find struct SliceInput`, `no function default_for`, `cannot find function step_hires`.

- [ ] **Step 3: Write minimal implementation**

In `src/kspace.rs`, add the input struct above `simulate_slice`:

```rust
/// Everything one slice's forward model needs. The simulation grid (`sim`) and the acquired
/// matrix (`acq_matrix`) are distinct: the object lives on the finer grid, and only the central
/// `acq_matrix` block of its k-space is evaluated (spec 3.1).
pub struct SliceInput<'a> {
    /// Per-compartment images on the SIM grid, layout `x + snx*y`.
    pub compartments: &'a [&'a [f32]],
    pub t2: &'a [f32],
    /// Off-resonance field (Hz) on the SIM grid.
    pub fmap: &'a [f32],
    /// Pre-readout object phase (radians) on the SIM grid. Ignored until Task 6.
    pub phase0: Option<&'a [f64]>,
    /// `[snx, sny]` — simulation grid, in-plane.
    pub sim: [usize; 2],
    /// `[nx, ny]` — acquired matrix.
    pub acq_matrix: [usize; 2],
    pub z: usize,
    pub nz: usize,
    /// Unit diffusion-gradient direction. Kept separate from `bval` so the phase model can form
    /// an effective q-vector without recovering it from a scaled product (spec 3.2).
    pub bvec: [f64; 3],
    pub bval: f64,
    pub slice_seed: u64,
}

/// Step edge at continuous position `edge` (sim-voxel units) with exact fractional occupancy in
/// the boundary voxel. This is the same partial-volume representation the path-length rasterizer
/// produces for real anatomy, so it is a production code path, not a test-only fixture.
pub fn step_hires(snx: usize, sny: usize, edge: f64) -> Vec<f32> {
    let mut v = vec![0.0f32; snx * sny];
    for x in 0..snx {
        let (lo, hi) = (x as f64, x as f64 + 1.0);
        let frac = if hi <= edge {
            0.0
        } else if lo >= edge {
            1.0
        } else {
            hi - edge
        };
        for y in 0..sny {
            v[x + snx * y] = frac as f32;
        }
    }
    v
}
```

Add a convenience constructor next to `impl Default for Acquisition`:

```rust
impl Acquisition {
    /// Defaults for a given acquired matrix. `Default::default()` is retained for callers that
    /// do not care about the matrix.
    pub fn default_for(_nx: usize, _ny: usize) -> Self {
        Acquisition::default()
    }
}
```

Now change `simulate_slice`. Replace its signature and the head of its body:

```rust
pub fn simulate_slice(inp: &SliceInput, acq: &Acquisition) -> Vec<(f32, f32)> {
    let [snx, sny] = inp.sim;
    let [nx, ny] = inp.acq_matrix;
    assert!(snx % nx == 0 && sny % ny == 0, "sim grid must be an integer multiple of the acquired matrix");
    let (z, nz) = (inp.z, inp.nz);
    let (compartments, t2, fmap) = (inp.compartments, inp.t2, inp.fmap);
    let epi = SingleShotEpi {
        kx_max: nx,
        ky_max: ny,
        t_line: acq.t_line,
        t_echo: acq.t_echo,
        reverse_phase: acq.reverse_phase,
    };
    let (t_ms, trf_ms, tread_ms) = line_times(&epi);
    let gradient = [inp.bvec[0] * inp.bval, inp.bvec[1] * inp.bval, inp.bvec[2] * inp.bval];
    let do_eddy = acq.eddy_strength != 0.0 && inp.bval.abs() > 1e-9;
    // acquired-matrix centres (k-space indexing) and sim-grid centres (image indexing)
    let (xs, ys, zs) = (nx / 2, ny / 2, nz / 2);
    let (sxs, sys) = (snx / 2, sny / 2);
    let at = |x: usize, y: usize| x + snx * y;
```

Then, inside the `kyi` loop, change the normalizer and the modulated-image extents:

```rust
        // Divide by the SIM extent: the loop still runs over the ny ACQUIRED lines, but each sits
        // at absolute sim index sys - ys + kyi, whose normalized frequency is (kyi - ys)/sny.
        let ky_norm = (kyi as f64 - ys as f64) / sny as f64;
```

Replace `let mut modimg = vec![C::ZERO; nx * ny];` with `let mut modimg = vec![C::ZERO; snx * sny];`, and change both inner loops from `for y in 0..ny` / `for x in 0..nx` to `for y in 0..sny` / `for x in 0..snx`. In the eddy block, centre on the sim grid:

```rust
                    let (xc, yc, zc) =
                        (x as f64 - sxs as f64, y as f64 - sys as f64, z as f64 - zs as f64);
```

Replace the y-sum and x-DFT blocks with:

```rust
        // inner y-sum over the SIM grid -> g(x), then an x-DFT evaluated only at acquired kx
        let mut g = vec![C::ZERO; snx];
        for x in 0..snx {
            let mut acc = C::ZERO;
            for y in 0..sny {
                acc = acc.add(modimg[at(x, y)].mul(C::cis(TAU * ky_norm * (y as f64 - sys as f64))));
            }
            g[x] = acc;
        }
        for kxi in 0..nx {
            let kx_norm = (kxi as f64 - xs as f64 + ghost_shift) / snx as f64;
            let mut acc = C::ZERO;
            for x in 0..snx {
                acc = acc.add(g[x].mul(C::cis(TAU * kx_norm * (x as f64 - sxs as f64))));
            }
            kspace[kxi + nx * kyi] = acc.scale(n_inv);
        }
```

and set `let n_inv = 1.0 / (snx * sny) as f64;` where `n_inv` is currently defined. The k-space buffer stays `vec![C::ZERO; nx * ny]` and `inverse_2d` is unchanged — reconstruction is at the acquired matrix.

Finally, update the two internal callers so the crate still builds: in `simulate_acquisition`, replace the `simulate_slice(&refs, ...)` call with a `SliceInput` carrying `sim: [nx, ny]` and `acq_matrix: [nx, ny]` (i.e. `o = 1`, preserving current behaviour), splitting the existing `gradients[g]` into `bvec`/`bval` via `let b = norm(gradients[g]); let bvec = if b > 1e-12 { [gradients[g][0]/b, ...] } else { [0.0;3] };`. Update every existing test in `mod tests` to build a `SliceInput` with `sim == acq_matrix` and `phase0: None`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS — the two new tests plus all pre-existing `kspace` tests (which now exercise `o = 1`).

- [ ] **Step 5: Commit**

```bash
git add src/kspace.rs
git commit -m "feat(kspace): crop to the acquired band during the forward transform"
```

---

### Task 3: Acquired-band convergence and oversampling selection

**Files:**
- Modify: `src/kspace.rs`
- Test: `src/kspace.rs`

**Interfaces:**
- Consumes: `SliceInput`, `simulate_slice` from Task 2.
- Produces: `pub fn simulate_slice_kspace(inp: &SliceInput, acq: &Acquisition) -> Vec<(f64, f64)>` — the acquired `nx*ny` k-space, layout `kx + nx*ky`, before the inverse transform.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn acquired_band_converges_with_simulation_resolution() {
    // Spec 4.1.6: adequacy is convergence of the ACQUIRED coefficients, not smoothness of the
    // object. A sharp edge is deliberately not band-limited; that is not a defect.
    let (nx, ny) = (16usize, 16usize);
    let acq = clean(nx, ny);
    let k_at = |o: usize| -> Vec<(f64, f64)> {
        let (snx, sny) = (nx * o, ny * o);
        let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.37) * o as f64);
        let fmap = vec![0.0f32; snx * sny];
        let comps: [&[f32]; 1] = [&img];
        simulate_slice_kspace(
            &SliceInput {
                compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: None,
                sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1,
                bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 0,
            },
            &acq,
        )
    };
    let rel = |a: &[(f64, f64)], b: &[(f64, f64)]| {
        let num: f64 = a.iter().zip(b).map(|(p, q)| (p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sum();
        let den: f64 = b.iter().map(|q| q.0 * q.0 + q.1 * q.1).sum::<f64>().max(1e-30);
        (num / den).sqrt()
    };
    let (k2, k4, k8) = (k_at(2), k_at(4), k_at(8));
    let (e2, e4) = (rel(&k2, &k4), rel(&k4, &k8));
    assert!(e4 < e2, "error must shrink with o: e(2->4)={e2:.4e}, e(4->8)={e4:.4e}");
    // Report o_min for the production default (spec 4.1.10): the smallest o under tolerance.
    let eps = 1e-3;
    let o_min = if e2 < eps { 2 } else if e4 < eps { 4 } else { 8 };
    println!("o_min at eps={eps}: {o_min}  (e2={e2:.3e}, e4={e4:.3e})");
    assert!(e4 < 1e-2, "o=4 should be within 1% of o=8: {e4:.4e}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib acquired_band_converges`
Expected: FAIL — `cannot find function simulate_slice_kspace`.

- [ ] **Step 3: Write minimal implementation**

Refactor `simulate_slice` so the k-space build is reusable. Extract everything from the start of the coil loop through the noise block into:

```rust
/// Build the acquired k-space for one coil. Shared by [`simulate_slice`] and
/// [`simulate_slice_kspace`] so tests can inspect coefficients before reconstruction.
fn build_coil_kspace(inp: &SliceInput, acq: &Acquisition, coil: usize, ncoils: usize) -> Vec<C> {
    // ... existing per-coil body, unchanged apart from taking `inp` ...
}
```

`simulate_slice` then loops `for coil in 0..ncoils { coil_kspace.push(build_coil_kspace(inp, acq, coil, ncoils)); }` and proceeds as before. Add the public accessor:

```rust
/// The acquired k-space of the first coil, before reconstruction. Layout `kx + nx*ky`.
/// Used by the convergence tests (spec 4.1.6); not part of the simulation path.
pub fn simulate_slice_kspace(inp: &SliceInput, acq: &Acquisition) -> Vec<(f64, f64)> {
    build_coil_kspace(inp, acq, 0, acq.n_coils.max(1))
        .into_iter()
        .map(|c| (c.re, c.im))
        .collect()
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS. Note the printed `o_min` line — it determines the production default (spec 4.1.10). Record it in the commit message.

- [ ] **Step 5: Commit**

```bash
git add src/kspace.rs
git commit -m "feat(kspace): expose acquired k-space; add convergence test for oversampling"
```

---

### Task 4: `--oversample` in the acquisition-grid prep script

**Files:**
- Modify: `data/trxscan_truth_data/scripts/prepare_acquisition_grid.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: a `work/sim/` directory of WM/GM/CSF/mask/fieldmap NIfTIs at `voxel/N` over the same FOV as `work/`, with matrix dims exactly `N` times the acquisition dims.

- [ ] **Step 1: Write the failing test**

Create `data/trxscan_truth_data/scripts/test_oversample.py`:

```python
"""Checks --oversample emits a sim grid that is an exact integer refinement of the acq grid."""
import subprocess, sys, pathlib
import nibabel as nib

def test_sim_grid_is_integer_refinement(tmp_path):
    here = pathlib.Path(__file__).parent
    root = here.parent
    subprocess.run([sys.executable, str(here / "prepare_acquisition_grid.py"),
                    "--anat-dir", str(root / "sub-0001a" / "anat"),
                    "--prefix", "sub-0001a_space-ACPC",
                    "--out", str(tmp_path), "--voxel", "1.7", "--oversample", "2"],
                   check=True)
    acq = nib.load(tmp_path / "wm.nii.gz")
    sim = nib.load(tmp_path / "sim" / "wm.nii.gz")
    # in-plane refined by exactly 2, slice direction untouched (spec 3.1)
    assert sim.shape[0] == acq.shape[0] * 2
    assert sim.shape[1] == acq.shape[1] * 2
    assert sim.shape[2] == acq.shape[2]
    # same FOV
    for i in (0, 1):
        assert abs(sim.shape[i] * sim.header.get_zooms()[i]
                   - acq.shape[i] * acq.header.get_zooms()[i]) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n <env> python -m pytest data/trxscan_truth_data/scripts/test_oversample.py -v`
Expected: FAIL — `unrecognized arguments: --oversample`.

- [ ] **Step 3: Write minimal implementation**

Add the argument next to `--voxel`:

```python
    ap.add_argument("--oversample", type=int, default=1, metavar="N",
                    help="also write a simulation grid at voxel/N in-plane (same FOV, same slice "
                         "thickness) into <out>/sim. N must be a positive integer: the k-space "
                         "crop assumes an integer matrix ratio. Slice direction is never "
                         "oversampled - it is not Fourier-encoded in 2D EPI.")
```

After the existing resampling writes its outputs, add:

```python
    if a.oversample > 1:
        n = a.oversample
        sim_dir = a.out / "sim"
        sim_dir.mkdir(parents=True, exist_ok=True)
        # Refine in-plane only, preserving FOV: dims *= n, in-plane zooms /= n.
        sim_shape = (acq_shape[0] * n, acq_shape[1] * n, acq_shape[2])
        sim_affine = acq_affine.copy()
        sim_affine[:, 0] /= n
        sim_affine[:, 1] /= n
        # Keep the FOV corner fixed: the first sim voxel centre moves inward by half the
        # difference between the coarse and fine voxel sizes on each refined axis.
        sim_affine[:3, 3] = (acq_affine[:3, 3]
                             - 0.5 * acq_affine[:3, 0] * (1 - 1.0 / n)
                             - 0.5 * acq_affine[:3, 1] * (1 - 1.0 / n))
        sim_ref = nib.Nifti1Image(np.zeros(sim_shape, dtype=np.float32), sim_affine)
        for name, img in outputs.items():
            nib.save(resample_from_to(img, sim_ref, order=1), sim_dir / f"{name}.nii.gz")
        print(f"wrote simulation grid {sim_shape} at {a.voxel / n:.3f} mm in-plane -> {sim_dir}")
```

where `outputs` is the dict of source anatomical-grid images the script already builds before writing the acquisition grid, `acq_shape`/`acq_affine` are the acquisition grid it computed, and `resample_from_to` is already imported. **Resample from the 1 mm anatomical source, not from the 1.7 mm acquisition grid** — upsampling the coarse maps adds no k-space content and would produce no ringing (spec 3.1).

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n <env> python -m pytest data/trxscan_truth_data/scripts/test_oversample.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data/trxscan_truth_data/scripts/
git commit -m "feat(data): --oversample emits an integer-refined simulation grid"
```

---

### Task 5: Phase model

**Files:**
- Create: `src/phase.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub struct BackgroundPhase { pub coeffs: [f64; 10] }` with `pub fn at(&self, x: f64, y: f64, z: f64) -> f64`.
  - `pub struct DiffusionPhase { pub c_q: f64, pub sigma_dx: f64, pub sigma_rot: f64 }` with `pub fn shot(&self, bval: f64, bvec: [f64; 3], volume: usize, slice_group: usize, seed: u64) -> ShotPhase`.
  - `pub struct ShotPhase { pub q_eff: [f64; 3], pub dx: [f64; 3], pub rot: [f64; 3] }` with `pub fn at(&self, r: [f64; 3]) -> f64`.
  - `pub struct PhaseModel { pub global: f64, pub background: BackgroundPhase, pub diffusion: DiffusionPhase }`.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn dp() -> DiffusionPhase {
        DiffusionPhase { c_q: 1e-3, sigma_dx: 0.5, sigma_rot: 0.0 }
    }

    #[test]
    fn b_zero_gives_exactly_zero_phase() {
        let s = dp().shot(0.0, [1.0, 0.0, 0.0], 3, 1, 42);
        assert_eq!(s.at([10.0, -4.0, 2.0]), 0.0);
        assert_eq!(s.q_eff, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn phase_reverses_sign_with_the_gradient() {
        let (a, b) = (dp().shot(1000.0, [0.0, 1.0, 0.0], 3, 1, 42),
                      dp().shot(1000.0, [0.0, -1.0, 0.0], 3, 1, 42));
        let r = [1.0, 2.0, 3.0];
        assert!((a.at(r) + b.at(r)).abs() < 1e-12, "{} vs {}", a.at(r), b.at(r));
    }

    #[test]
    fn phase_scales_as_sqrt_b_not_linearly() {
        // Same shot => same dx; only |q_eff| changes. 4x b must give 2x phase, not 4x.
        let (a, b) = (dp().shot(1000.0, [1.0, 0.0, 0.0], 3, 1, 42),
                      dp().shot(4000.0, [1.0, 0.0, 0.0], 3, 1, 42));
        let r = [0.0, 0.0, 0.0];
        let ratio = b.at(r) / a.at(r);
        assert!((ratio - 2.0).abs() < 1e-9, "sqrt(b) scaling expected, got ratio {ratio}");
    }

    #[test]
    fn rotation_makes_phase_linear_in_position() {
        let d = DiffusionPhase { c_q: 1e-3, sigma_dx: 0.0, sigma_rot: 1e-3 };
        let s = d.shot(2000.0, [1.0, 0.0, 0.0], 1, 0, 7);
        // linear field => midpoint value equals the mean of the endpoints
        let (p, q) = ([0.0, -8.0, 0.0], [0.0, 8.0, 0.0]);
        let mid = [0.0, 0.0, 0.0];
        assert!((s.at(mid) - 0.5 * (s.at(p) + s.at(q))).abs() < 1e-12);
    }

    #[test]
    fn different_shots_draw_different_motion() {
        let (a, b) = (dp().shot(1000.0, [1.0, 0.0, 0.0], 3, 1, 42),
                      dp().shot(1000.0, [1.0, 0.0, 0.0], 4, 1, 42));
        assert!(a.dx != b.dx, "per-volume draws must differ");
    }

    #[test]
    fn background_field_is_smooth_and_reproducible() {
        let bg = BackgroundPhase { coeffs: [0.3, 0.01, -0.02, 0.005, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0] };
        assert_eq!(bg.at(1.0, 2.0, 3.0), bg.at(1.0, 2.0, 3.0));
        let step = (bg.at(1.1, 2.0, 3.0) - bg.at(1.0, 2.0, 3.0)).abs();
        assert!(step < 0.05, "background phase must vary slowly, got {step} per 0.1 voxel");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib phase`
Expected: FAIL — `file not found for module phase`.

- [ ] **Step 3: Write minimal implementation**

`src/phase.rs`:

```rust
//! Object phase (spec 3.2). Three additive terms, all applied to the object on the SIMULATION
//! grid *before* the finite Fourier acquisition.
//!
//! **No term is derived from the fieldmap.** TRXScan simulates spin-echo EPI DWI (`kspace.rs`
//! applies `exp(-|t|/t_inhom)` centred on the echo, and `readout.rs` defines
//! `time_from_rf = t_echo + time_from_max_echo`). Static off-resonance is refocused at the spin
//! echo and survives only as readout-time-dependent phase, which `kspace` already models as
//! geometric distortion. A `2*PI*fmap*TE` term would double-count B0 and impose gradient-echo
//! physics on a spin-echo sequence.

/// Deterministic Gaussian source, seeded per shot. Mirrors the SplitMix64 generator in `kspace`.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Smooth pre-readout object phase: one low-order 3D polynomial, sampled per slice.
///
/// Named for what it is. This multiplies the object *before* Fourier encoding, and
/// `F_trunc{f * exp(i*phi)} != exp(i*phi) * F_trunc{f}`, so it may stand in only for phase that
/// genuinely exists before encoding. A scanner phase convention applied *after* reconstruction
/// rotates an already-reconstructed ringing pattern and belongs in a separate transform.
///
/// It is **not** a substitute for coil-specific complex sensitivities: a real multi-coil model has
/// a different `theta_c(r)` per coil, and one common multiplier cannot reproduce the relative coil
/// phases that GRAPPA and coil combination depend on.
///
/// Coefficient order: `1, x, y, z, x^2, y^2, z^2, xy, xz, yz`, in voxel units from the FOV centre.
#[derive(Debug, Clone, Copy)]
pub struct BackgroundPhase {
    pub coeffs: [f64; 10],
}

impl BackgroundPhase {
    pub fn at(&self, x: f64, y: f64, z: f64) -> f64 {
        let c = &self.coeffs;
        c[0] + c[1] * x + c[2] * y + c[3] * z
            + c[4] * x * x + c[5] * y * y + c[6] * z * z
            + c[7] * x * y + c[8] * x * z + c[9] * y * z
    }
}

/// One shot's realised motion and effective q-vector.
#[derive(Debug, Clone, Copy)]
pub struct ShotPhase {
    /// Effective q-vector `c_q * sqrt(b) * bvec_unit`. See [`DiffusionPhase`] on why "effective".
    pub q_eff: [f64; 3],
    /// Translation drawn for this shot (voxel units).
    pub dx: [f64; 3],
    /// Rotation vector drawn for this shot (radians), about the FOV centre.
    pub rot: [f64; 3],
}

impl ShotPhase {
    /// Phase at position `r` (voxel units from the FOV centre): `q_eff . (dx + rot x r)`.
    /// Constant plus linear in `r` by construction.
    pub fn at(&self, r: [f64; 3]) -> f64 {
        let c = [
            self.rot[1] * r[2] - self.rot[2] * r[1],
            self.rot[2] * r[0] - self.rot[0] * r[2],
            self.rot[0] * r[1] - self.rot[1] * r[0],
        ];
        (0..3).map(|i| self.q_eff[i] * (self.dx[i] + c[i])).sum()
    }
}

/// Motion-induced diffusion phase, `phi = q_eff . u(r)`.
///
/// `q_eff = c_q * sqrt(b) * bvec_unit` is an **effective** q-vector, not the physical one: in PGSE
/// `b ~ q^2 (Delta - delta/3)`, so `sqrt(b) * bvec` is proportional to `q` only under fixed, known
/// timing, and TRXScan has no `delta`, `Delta` or waveform parameters. `c_q` absorbs the timing and
/// the radian/cycle convention, and is calibrated. If waveform parameters are added later this
/// becomes a physical q-vector without changing this interface.
#[derive(Debug, Clone, Copy)]
pub struct DiffusionPhase {
    pub c_q: f64,
    /// SD of the per-shot translation (voxel units).
    pub sigma_dx: f64,
    /// SD of the per-shot rotation (radians).
    pub sigma_rot: f64,
}

impl DiffusionPhase {
    pub fn shot(&self, bval: f64, bvec: [f64; 3], volume: usize, slice_group: usize, seed: u64) -> ShotPhase {
        let n = (bvec[0] * bvec[0] + bvec[1] * bvec[1] + bvec[2] * bvec[2]).sqrt();
        // b = 0 has no diffusion encoding, so no motion-induced phase. Also avoids a degenerate
        // unit vector when bvec is the zero vector.
        if bval <= 0.0 || n < 1e-12 {
            return ShotPhase { q_eff: [0.0; 3], dx: [0.0; 3], rot: [0.0; 3] };
        }
        let s = self.c_q * bval.sqrt() / n;
        let q_eff = [bvec[0] * s, bvec[1] * s, bvec[2] * s];
        let mut rng = Rng(
            seed ^ (volume as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (slice_group as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9),
        );
        let mut draw = |sd: f64| [rng.gauss() * sd, rng.gauss() * sd, rng.gauss() * sd];
        let dx = draw(self.sigma_dx);
        let rot = draw(self.sigma_rot);
        ShotPhase { q_eff, dx, rot }
    }
}

/// The complete object-phase model.
#[derive(Debug, Clone, Copy)]
pub struct PhaseModel {
    /// Global phase. A gauge choice and test control, never calibrated.
    pub global: f64,
    pub background: BackgroundPhase,
    pub diffusion: DiffusionPhase,
}

impl PhaseModel {
    /// Phase at `r` (voxel units from the FOV centre) for one shot.
    pub fn at(&self, r: [f64; 3], shot: &ShotPhase) -> f64 {
        self.global + self.background.at(r[0], r[1], r[2]) + shot.at(r)
    }

    /// A zero model: real-valued object. Useful for isolating non-phase behaviour in tests.
    pub fn none() -> Self {
        PhaseModel {
            global: 0.0,
            background: BackgroundPhase { coeffs: [0.0; 10] },
            diffusion: DiffusionPhase { c_q: 0.0, sigma_dx: 0.0, sigma_rot: 0.0 },
        }
    }
}
```

Add to `src/lib.rs` after `pub mod signal;`:

```rust
/// Object phase model (spec 3.2).
pub mod phase;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib phase`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add src/phase.rs src/lib.rs
git commit -m "feat(phase): object phase model with effective q-vector diffusion phase"
```

---

### Task 6: Wire phase into the forward model

**Files:**
- Modify: `src/kspace.rs`
- Test: `src/kspace.rs`

**Interfaces:**
- Consumes: `SliceInput.phase0` (Task 2), `phase::PhaseModel` (Task 5).
- Produces: `simulate_slice` honouring `phase0`; `pub fn phase_slice(model: &PhaseModel, shot: &ShotPhase, snx: usize, sny: usize, z: usize, nz: usize) -> Vec<f64>` building the per-slice sim-grid phase field.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn global_phase_rotation_is_exact() {
    let (nx, ny, o) = (16usize, 16usize, 4usize);
    let (snx, sny) = (nx * o, ny * o);
    let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
    let fmap = vec![0.0f32; snx * sny];
    let comps: [&[f32]; 1] = [&img];
    let run = |ph: Option<&[f64]>| {
        simulate_slice(
            &SliceInput {
                compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: ph,
                sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1,
                bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 0,
            },
            &clean(nx, ny),
        )
    };
    let alpha = 0.7f64;
    let base = run(None);
    let rotated = run(Some(&vec![alpha; snx * sny]));
    let (ca, sa) = alpha.sin_cos();
    let (sa, ca) = (ca, sa); // sin_cos returns (sin, cos)
    for i in 0..nx * ny {
        let (re, im) = (base[i].0 as f64, base[i].1 as f64);
        let (er, ei) = (re * ca - im * sa, re * sa + im * ca);
        assert!((rotated[i].0 as f64 - er).abs() < 1e-6, "re mismatch at {i}");
        assert!((rotated[i].1 as f64 - ei).abs() < 1e-6, "im mismatch at {i}");
        let (m0, m1) = ((re * re + im * im).sqrt(),
                        ((rotated[i].0 as f64).powi(2) + (rotated[i].1 as f64).powi(2)).sqrt());
        assert!((m0 - m1).abs() < 1e-6, "magnitude changed at {i}");
    }
}

#[test]
fn object_phase_puts_ringing_in_both_channels() {
    // With no object phase the image is real to machine precision; with phase it is not.
    let (nx, ny, o) = (16usize, 16usize, 4usize);
    let (snx, sny) = (nx * o, ny * o);
    let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
    let fmap = vec![0.0f32; snx * sny];
    let comps: [&[f32]; 1] = [&img];
    let run = |ph: Option<&[f64]>| {
        simulate_slice(
            &SliceInput {
                compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: ph,
                sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1,
                bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 0,
            },
            &clean(nx, ny),
        )
    };
    let ratio = |v: &Vec<(f32, f32)>| {
        let mr = v.iter().map(|p| p.0.abs()).fold(0.0f32, f32::max);
        let mi = v.iter().map(|p| p.1.abs()).fold(0.0f32, f32::max);
        (mi / mr) as f64
    };
    assert!(ratio(&run(None)) < 1e-2, "no-phase object should be essentially real");
    let ramp: Vec<f64> = (0..snx * sny)
        .map(|i| 0.9 * ((i % snx) as f64 - snx as f64 / 2.0) / snx as f64)
        .collect();
    assert!(ratio(&run(Some(&ramp))) > 0.1, "phase should move energy into the imaginary channel");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib kspace`
Expected: FAIL — `object_phase_puts_ringing_in_both_channels` fails because `phase0` is ignored, so both ratios are identical.

- [ ] **Step 3: Write minimal implementation**

In `build_coil_kspace`, change the modulated-image line to add the object phase:

```rust
                let phi0 = inp.phase0.map_or(0.0, |p| p[at(x, y)]);
                modimg[at(x, y)] = C::cis(TAU * phi + phi0).scale(f_real);
```

(`phi` is the existing distortion/eddy term, which carries its own `TAU` factor; `phi0` is already in radians, so it is added outside the `TAU` multiplication.)

Add the field builder near `simulate_slice`:

```rust
use crate::phase::{PhaseModel, ShotPhase};

/// Sample the phase model onto one slice of the simulation grid. Positions are voxel units from
/// the FOV centre, so the same coefficients mean the same field at any oversampling factor.
pub fn phase_slice(
    model: &PhaseModel,
    shot: &ShotPhase,
    snx: usize,
    sny: usize,
    o: usize,
    z: usize,
    nz: usize,
) -> Vec<f64> {
    let (sxs, sys, zs) = (snx as f64 / 2.0, sny as f64 / 2.0, nz as f64 / 2.0);
    let zc = z as f64 - zs;
    let mut v = vec![0.0f64; snx * sny];
    for y in 0..sny {
        for x in 0..snx {
            // convert sim-voxel indices to ACQUIRED voxel units so coefficients are o-invariant
            let r = [(x as f64 - sxs) / o as f64, (y as f64 - sys) / o as f64, zc];
            v[x + snx * y] = model.at(r, shot);
        }
    }
    v
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS, including both new tests and everything prior.

- [ ] **Step 5: Commit**

```bash
git add src/kspace.rs
git commit -m "feat(kspace): apply pre-readout object phase before the acquisition"
```

---

### Task 7: Benchmark outputs and the block mean

**Files:**
- Create: `src/benchmark.rs`
- Modify: `src/lib.rs`
- Test: `src/benchmark.rs`

**Interfaces:**
- Consumes: `SliceInput`, `simulate_slice` (Task 2), `phase_slice` (Task 6).
- Produces:
  - `pub fn block_mean_complex(hires: &[(f64, f64)], snx: usize, sny: usize, o: usize) -> Vec<(f64, f64)>`.
  - `pub struct BenchmarkSlice { pub object_hires: Vec<(f64, f64)>, pub object_nominal: Vec<(f64, f64)>, pub acquired_clean: Vec<(f32, f32)>, pub acquired_noisy: Vec<(f32, f32)> }`.
  - `pub fn gibbs_benchmark_acquisition(base: &Acquisition) -> Acquisition` returning the canonical Gibbs-only configuration.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_mean_preserves_scale() {
        // THE hazard this function exists to prevent: a sum would multiply intensities by o^2.
        let (nx, ny, o) = (4usize, 4usize, 3usize);
        let (snx, sny) = (nx * o, ny * o);
        let hires = vec![(2.0f64, -1.0f64); snx * sny];
        let out = block_mean_complex(&hires, snx, sny, o);
        assert_eq!(out.len(), nx * ny);
        for (re, im) in out {
            assert!((re - 2.0).abs() < 1e-12, "block mean rescaled: got {re}, expected 2.0");
            assert!((im + 1.0).abs() < 1e-12, "block mean rescaled: got {im}, expected -1.0");
        }
    }

    #[test]
    fn block_mean_carries_intravoxel_dephasing() {
        // Two hires samples with opposite phase must cancel, not add in magnitude.
        let (o, snx, sny) = (2usize, 2usize, 2usize);
        let hires = vec![(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (-1.0, 0.0)];
        let out = block_mean_complex(&hires, snx, sny, o);
        assert_eq!(out.len(), 1);
        assert!(out[0].0.abs() < 1e-12, "dephasing must cancel, got {}", out[0].0);
    }

    #[test]
    fn canonical_benchmark_mode_disables_confounding_artifacts() {
        let a = gibbs_benchmark_acquisition(&Acquisition::default());
        assert!(!a.do_distortions, "EPI distortion confounds edge location");
        assert!(!a.do_relaxation, "T2* readout decay is a k-space filter");
        assert_eq!(a.eddy_strength, 0.0);
        assert_eq!(a.ghost_offset, 0.0);
        assert_eq!(a.accel, 1, "GRAPPA is a separate reconstruction operator (spec 3.5.3)");
        assert_eq!(a.n_coils, 1);
        assert_eq!(a.n_spikes, 0);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib benchmark`
Expected: FAIL — `file not found for module benchmark`.

- [ ] **Step 3: Write minimal implementation**

`src/benchmark.rs`:

```rust
//! Scoreable benchmark outputs (spec 3.5).
//!
//! Evaluating an unringing method needs an artifact-free target *and* a configuration in which the
//! score means what it claims. This module provides both.

use crate::kspace::Acquisition;

/// Reduce a simulation-grid complex field onto the acquisition grid by the **complex block mean**
/// over each `o x o` cell.
///
/// This is a mean, `(1/o^2) * sum_j z_j`, never a sum: a sum would scale every intensity by `o^2`.
///
/// It averages the *complex* field, so a voxel whose sub-voxel phase varies loses signal. That
/// intravoxel dephasing is physical and belongs in the reference.
pub fn block_mean_complex(hires: &[(f64, f64)], snx: usize, sny: usize, o: usize) -> Vec<(f64, f64)> {
    assert!(o > 0 && snx % o == 0 && sny % o == 0, "sim grid must be an integer multiple of o");
    let (nx, ny) = (snx / o, sny / o);
    let inv = 1.0 / (o * o) as f64;
    let mut out = vec![(0.0f64, 0.0f64); nx * ny];
    for y in 0..sny {
        for x in 0..snx {
            let s = hires[x + snx * y];
            let d = &mut out[(x / o) + nx * (y / o)];
            d.0 += s.0;
            d.1 += s.1;
        }
    }
    for d in out.iter_mut() {
        d.0 *= inv;
        d.1 *= inv;
    }
    out
}

/// The four co-registered images one benchmark slice produces.
pub struct BenchmarkSlice {
    /// The simulated complex object on the simulation grid, before any acquisition. Diagnostic.
    pub object_hires: Vec<(f64, f64)>,
    /// `object_hires` block-averaged onto the acquisition grid. **The scoring target.**
    ///
    /// Derived from the same array that produced the acquired images, never regenerated by a
    /// second Stage A run at `o = 1`: that would lose intravoxel dephasing and would be a
    /// different random realization besides.
    pub object_nominal: Vec<(f64, f64)>,
    /// Finite-band reconstruction, no noise.
    pub acquired_clean: Vec<(f32, f32)>,
    /// Finite-band reconstruction, same object and phase realization, with noise.
    pub acquired_noisy: Vec<(f32, f32)>,
}

/// The canonical Gibbs-only configuration: everything that would contaminate
/// `unringed - object_nominal` with non-Gibbs error is disabled.
///
/// GRAPPA is disabled on principle, not on measurement. It survives at R = 2 empirically, but it
/// is a separate reconstruction operator and its own error would contribute to the score.
/// Partial Fourier, noise and windows are left as the caller set them: they are legitimate
/// experimental factors within the Gibbs benchmark.
pub fn gibbs_benchmark_acquisition(base: &Acquisition) -> Acquisition {
    Acquisition {
        do_distortions: false,
        do_relaxation: false,
        eddy_strength: 0.0,
        eddy_quad: 0.0,
        ghost_offset: 0.0,
        n_spikes: 0,
        accel: 1,
        n_coils: 1,
        ..base.clone()
    }
}
```

Add to `src/lib.rs` after `pub mod kspace;`:

```rust
/// Scoreable benchmark outputs (spec 3.5).
pub mod benchmark;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib benchmark`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/benchmark.rs src/lib.rs
git commit -m "feat(benchmark): complex block mean and canonical Gibbs-only fixture"
```

---

### Task 8: Sampling mask and noise convention

**Files:**
- Modify: `src/kspace.rs`
- Test: `src/kspace.rs`

**Interfaces:**
- Consumes: `SliceInput` (Task 2).
- Produces: `pub fn sampling_mask(nx: usize, ny: usize, acq: &Acquisition) -> Vec<bool>` — `true` where a k-space sample is acquired, layout `kx + nx*ky`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn noise_lands_only_on_acquired_samples() {
    let (nx, ny) = (24usize, 24usize);
    let acq = Acquisition { partial_fourier: 0.75, noise_variance: 1.0, ..clean(nx, ny) };
    let mask = sampling_mask(nx, ny, &acq);
    let empty = vec![0.0f32; nx * ny];
    let comps: [&[f32]; 1] = [&empty];
    let fmap = vec![0.0f32; nx * ny];
    let k = simulate_slice_kspace(
        &SliceInput {
            compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: None,
            sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
            bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 9,
        },
        &acq,
    );
    for i in 0..nx * ny {
        if !mask[i] {
            assert_eq!((k[i].0, k[i].1), (0.0, 0.0), "unacquired sample {i} carries noise");
        }
    }
}

#[test]
fn noise_sd_scales_as_sqrt_sampled_fraction() {
    // Scoped: GRAPPA off, no window, identical per-sample variance (spec 4.1.8).
    let (nx, ny) = (32usize, 32usize);
    let sd_at = |pf: f64| -> f64 {
        let acq = Acquisition { partial_fourier: pf, noise_variance: 1.0, ..clean(nx, ny) };
        let empty = vec![0.0f32; nx * ny];
        let comps: [&[f32]; 1] = [&empty];
        let fmap = vec![0.0f32; nx * ny];
        let out = simulate_slice(
            &SliceInput {
                compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: None,
                sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
                bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 3,
            },
            &acq,
        );
        // per-component SD, matching the noise_variance convention
        let v: f64 = out.iter().map(|p| (p.0 as f64).powi(2)).sum::<f64>() / (nx * ny) as f64;
        v.sqrt()
    };
    let (full, pf) = (sd_at(1.0), sd_at(0.5));
    assert!((full - 1.0).abs() < 0.15, "full-sampling per-component SD {full:.3}, expected ~1.0");
    let ratio = pf / full;
    assert!((ratio - 0.5f64.sqrt()).abs() < 0.1, "expected sqrt(0.5)=0.707 scaling, got {ratio:.3}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib kspace`
Expected: FAIL — `cannot find function sampling_mask`; and `noise_lands_only_on_acquired_samples` fails because noise is currently added to every sample.

- [ ] **Step 3: Write minimal implementation**

Add above `build_coil_kspace`:

```rust
/// Which k-space samples are actually acquired: partial Fourier plus GRAPPA undersampling.
/// Layout `kx + nx*ky`. This is the single source of truth for both the forward build and the
/// noise, so signal and noise can never disagree about what was sampled (spec 3.3).
pub fn sampling_mask(nx: usize, ny: usize, acq: &Acquisition) -> Vec<bool> {
    let (xs_unused, ys) = (nx / 2, ny / 2);
    let _ = xs_unused;
    let accel = acq.accel.max(1);
    let acs_half = (acq.acs_lines / 2) as i64;
    let mut m = vec![false; nx * ny];
    for kyi in 0..ny {
        if acq.partial_fourier < 1.0 {
            let skip = if acq.reverse_phase {
                kyi as f64 > (ny as f64 * acq.partial_fourier).ceil()
            } else {
                (kyi as f64) < (ny as f64 * (1.0 - acq.partial_fourier)).floor()
                    && (kyi > 0 || ny % 2 == 1)
            };
            if skip {
                continue;
            }
        }
        let acquired = accel <= 1
            || (kyi as i64 - ys as i64).abs() <= acs_half
            || kyi % accel == ys % accel;
        if !acquired {
            continue;
        }
        for kxi in 0..nx {
            m[kxi + nx * kyi] = true;
        }
    }
    m
}
```

In `build_coil_kspace`, replace the two `continue` guards at the top of the `kyi` loop with a single mask lookup, and replace the noise block:

```rust
    // complex k-space noise, on ACQUIRED samples only.
    //
    // `noise_variance` is the per-component variance of the reconstructed complex image under
    // full sampling, single-coil, pre-combination: Var(Re n) = Var(Im n) = noise_variance, so
    // E[|n|^2] = 2*noise_variance. The per-sample variance follows from the reconstruction
    // normalization. Masking alone produces the sqrt(f) scaling; there is deliberately no
    // additional sampled-fraction factor, which would double-count it.
    if acq.noise_variance > 0.0 {
        let mut rng = Rng((inp.slice_seed ^ (coil as u64).wrapping_mul(0x9E37_79B9)) | 1);
        let sigma = (acq.noise_variance / (nx * ny) as f64).sqrt();
        for (i, k) in kspace.iter_mut().enumerate() {
            if mask[i] {
                k.re += rng.gauss() * sigma;
                k.im += rng.gauss() * sigma;
            }
        }
    }
```

with `let mask = sampling_mask(nx, ny, acq);` computed once at the top of the function.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kspace.rs
git commit -m "fix(kspace): confine noise to the sampling mask under an explicit variance convention"
```

---

### Task 9: Noise covariance matches the sampling mask

**Files:**
- Modify: `src/kspace.rs`
- Test: `src/kspace.rs`

**Interfaces:**
- Consumes: `sampling_mask` (Task 8), `simulate_slice` (Task 2).
- Produces: no new public API; test only.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn noise_covariance_matches_the_sampling_mask() {
    // Spec 4.1.7a, scoped to GRAPPA disabled and window None: for white noise on mask M, the
    // reconstructed image noise autocovariance is the inverse DFT of M, up to scale.
    let (nx, ny) = (16usize, 16usize);
    let acq = Acquisition { partial_fourier: 0.75, noise_variance: 1.0, ..clean(nx, ny) };
    let mask = sampling_mask(nx, ny, &acq);
    let empty = vec![0.0f32; nx * ny];
    let fmap = vec![0.0f32; nx * ny];

    // Monte Carlo the lag-(dx,0) autocovariance along the readout axis.
    let trials = 240;
    let lags = 4usize;
    let mut meas = vec![0.0f64; lags];
    for t in 0..trials {
        let comps: [&[f32]; 1] = [&empty];
        let out = simulate_slice(
            &SliceInput {
                compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: None,
                sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
                bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 1000 + t as u64,
            },
            &acq,
        );
        for (d, m) in meas.iter_mut().enumerate() {
            let mut s = 0.0;
            for y in 0..ny {
                for x in 0..nx - d {
                    s += out[x + nx * y].0 as f64 * out[x + d + nx * y].0 as f64;
                }
            }
            *m += s / ((nx - d) * ny) as f64;
        }
    }
    for m in meas.iter_mut() { *m /= trials as f64; }

    // Prediction: inverse DFT of the mask along kx, summed over the acquired ky lines.
    let mut pred = vec![0.0f64; lags];
    for (d, p) in pred.iter_mut().enumerate() {
        let mut s = 0.0;
        for kyi in 0..ny {
            for kxi in 0..nx {
                if mask[kxi + nx * kyi] {
                    let kx = (kxi as f64 - (nx / 2) as f64) / nx as f64;
                    s += (TAU * kx * d as f64).cos();
                }
            }
        }
        *p = s;
    }
    // Compare shapes, normalizing out the common scale.
    for d in 1..lags {
        let (a, b) = (meas[d] / meas[0], pred[d] / pred[0]);
        assert!((a - b).abs() < 0.08, "lag {d}: measured {a:.3}, mask prediction {b:.3}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib noise_covariance`
Expected: FAIL if any part of Task 8 is incomplete; if Task 8 is correct this test may pass immediately, which is an acceptable outcome for a characterization test — confirm it fails when you temporarily revert the mask guard in the noise block.

- [ ] **Step 3: Write minimal implementation**

No production change is required; Task 8's mask makes the prediction hold. If the test fails, the defect is in `sampling_mask` or in the noise block, not in the test. Verify by asserting `mask.iter().filter(|b| **b).count()` matches `partial_fourier * nx * ny` to within one PE line.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kspace.rs
git commit -m "test(kspace): noise autocovariance matches the inverse DFT of the sampling mask"
```

---

### Task 10: Partial-Fourier ringing

**Files:**
- Modify: `src/kspace.rs`
- Test: `src/kspace.rs`

**Interfaces:**
- Consumes: `step_hires`, `SliceInput`, `simulate_slice` (Task 2).
- Produces: no new public API; tests only.

6/8 is the shipping default (`src/bin/trxscan.rs:254`), so this is the primary configuration, not an edge case.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn partial_fourier_ringing_is_asymmetric_about_the_edge() {
    // Zero-filled PF (no homodyne/POCS) blurs and rings asymmetrically along PE, unlike the
    // symmetric full-Fourier case. RPG exists because of this.
    let (nx, ny, o) = (32usize, 32usize, 8usize);
    let (snx, sny) = (nx * o, ny * o);
    // Edge along the PHASE-ENCODE axis so PF acts on it.
    let mut img = vec![0.0f32; snx * sny];
    let edge = (ny as f64 / 2.0 + 0.5) * o as f64;
    for y in 0..sny {
        let f = if (y as f64 + 1.0) <= edge { 0.0 }
                else if y as f64 >= edge { 1.0 }
                else { y as f64 + 1.0 - edge };
        for x in 0..snx { img[x + snx * y] = f as f32; }
    }
    let fmap = vec![0.0f32; snx * sny];
    let comps: [&[f32]; 1] = [&img];
    let run = |pf: f64| {
        simulate_slice(
            &SliceInput {
                compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: None,
                sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1,
                bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 0,
            },
            &Acquisition { partial_fourier: pf, ..clean(nx, ny) },
        )
    };
    let col = nx / 2;
    let asym = |v: &Vec<(f32, f32)>| {
        // compare ripple energy just above vs just below the edge
        let below: f64 = (1..6).map(|d| (v[col + nx * (ny / 2 - d)].0 as f64).powi(2)).sum();
        let above: f64 = (1..6).map(|d| (v[col + nx * (ny / 2 + d)].0 as f64 - 1.0).powi(2)).sum();
        (above / below.max(1e-12)).ln().abs()
    };
    let full = run(1.0);
    for pf in [0.75, 0.875] {
        let p = run(pf);
        assert!(asym(&p) > asym(&full),
            "pf={pf} should ring more asymmetrically than full Fourier: {} vs {}",
            asym(&p), asym(&full));
    }
    // PF must not destroy the object.
    let e_full: f32 = full.iter().map(|v| v.0.abs()).sum();
    let e_pf: f32 = run(0.75).iter().map(|v| v.0.abs()).sum();
    assert!((e_pf / e_full - 1.0).abs() < 0.5, "PF energy {e_pf} vs full {e_full}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib partial_fourier_ringing`
Expected: PASS or FAIL depending on the existing PF implementation. If it passes immediately, it is a characterization test locking in correct behaviour — keep it. If it fails, the PF line-skip logic in `sampling_mask` disagrees with the old inline logic; reconcile them, keeping `sampling_mask` as the single source of truth.

- [ ] **Step 3: Write minimal implementation**

No production change expected. If reconciliation is needed, change only `sampling_mask` and delete any residual inline PF logic from `build_coil_kspace`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kspace.rs
git commit -m "test(kspace): partial-Fourier ringing is asymmetric at 6/8 and 7/8"
```

---

### Task 11: Even-matrix window asymmetry

**Files:**
- Modify: `src/kspace.rs`
- Test: `src/kspace.rs`

**Interfaces:**
- Consumes: `simulate_slice_kspace` (Task 3).
- Produces: no new public API; a documented invariant plus its test.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn even_matrix_window_asymmetry_is_intentional() {
    // The acquired band is [-n/2, n/2-1]: asymmetric about k=0 by one sample, exactly as real
    // even-matrix Cartesian acquisitions are. Retained deliberately (spec 3.4), so pin it.
    let n = 32i64;
    let (lo, hi) = (-(n / 2), n / 2 - 1);
    assert_eq!(lo, -16);
    assert_eq!(hi, 15);
    assert_eq!((hi - lo + 1) as usize, n as usize, "the band must hold exactly n samples");
    assert_eq!(lo.abs() - hi.abs(), 1, "one extra negative-frequency sample, by convention");

    // Consequence: a real object acquires a small imaginary component. It is deterministic and
    // is NOT realistic object phase (that is what `phase.rs` is for).
    let (nx, ny, o) = (16usize, 16usize, 4usize);
    let (snx, sny) = (nx * o, ny * o);
    let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
    let fmap = vec![0.0f32; snx * sny];
    let comps: [&[f32]; 1] = [&img];
    let out = simulate_slice(
        &SliceInput {
            compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: None,
            sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1,
            bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 0,
        },
        &clean(nx, ny),
    );
    let mr = out.iter().map(|p| p.0.abs()).fold(0.0f32, f32::max);
    let mi = out.iter().map(|p| p.1.abs()).fold(0.0f32, f32::max);
    assert!(mi / mr < 0.05, "asymmetry residual should stay small: {}", mi / mr);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib even_matrix`
Expected: FAIL — test not yet present. (It passes on first run once added; this task's deliverable is the pinned invariant and its documentation.)

- [ ] **Step 3: Write minimal implementation**

Add the explanatory comment above the `xs`/`ys` definitions in `build_coil_kspace`:

```rust
    // Centred k-space indexing: the acquired band is [-n/2, n/2-1], asymmetric about k=0 by one
    // sample. This is deliberate, not an off-by-one: real even-matrix Cartesian acquisitions cover
    // exactly this range. It gives a real object a small deterministic imaginary component, which
    // is NOT object phase - see `phase.rs` for that. Pinned by
    // `even_matrix_window_asymmetry_is_intentional`.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kspace.rs
git commit -m "docs(kspace): pin the even-matrix band asymmetry as intentional"
```

---

### Task 12: Reconstruction windows

**Files:**
- Modify: `src/kspace.rs`
- Test: `src/kspace.rs`

**Interfaces:**
- Consumes: `simulate_slice` (Task 2), `sampling_mask` (Task 8).
- Produces:
  - `pub enum KspaceWindow { None, Tukey { alpha: f64 }, Hann, Fermi { radius: f64, width: f64 } }` with `pub fn at(&self, kx: f64, ky: f64) -> f64` taking normalized radii.
  - `Acquisition.window: KspaceWindow` replacing `Acquisition.zero_ringing: f64`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn window_filters_signal_and_noise_together() {
    // The original defect (spec finding 2.5) was a band-limited signal with unfiltered noise.
    // A reconstruction window must multiply both, so noise-only data must be suppressed too.
    let (nx, ny) = (32usize, 32usize);
    let empty = vec![0.0f32; nx * ny];
    let fmap = vec![0.0f32; nx * ny];
    let sd = |w: KspaceWindow| -> f64 {
        let comps: [&[f32]; 1] = [&empty];
        let out = simulate_slice(
            &SliceInput {
                compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: None,
                sim: [nx, ny], acq_matrix: [nx, ny], z: 0, nz: 1,
                bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 5,
            },
            &Acquisition { noise_variance: 1.0, window: w, ..clean(nx, ny) },
        );
        (out.iter().map(|p| (p.0 as f64).powi(2)).sum::<f64>() / (nx * ny) as f64).sqrt()
    };
    let unwindowed = sd(KspaceWindow::None);
    let hann = sd(KspaceWindow::Hann);
    assert!(hann < 0.8 * unwindowed,
        "a window must attenuate noise too: {hann:.3} vs {unwindowed:.3}");
}

#[test]
fn window_reduces_ringing_below_the_unapodized_case() {
    let (nx, ny, o) = (32usize, 32usize, 8usize);
    let (snx, sny) = (nx * o, ny * o);
    let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
    let fmap = vec![0.0f32; snx * sny];
    let comps: [&[f32]; 1] = [&img];
    let peak = |w: KspaceWindow| {
        let out = simulate_slice(
            &SliceInput {
                compartments: &comps, t2: &[100.0], fmap: &fmap, phase0: None,
                sim: [snx, sny], acq_matrix: [nx, ny], z: 0, nz: 1,
                bvec: [0.0, 0.0, 0.0], bval: 0.0, slice_seed: 0,
            },
            &Acquisition { window: w, ..clean(nx, ny) },
        );
        (nx / 2 + 1..nx).map(|x| out[x + nx * (ny / 2)].0 as f64).fold(f64::MIN, f64::max)
    };
    assert!(peak(KspaceWindow::Hann) < peak(KspaceWindow::None),
        "apodization must reduce overshoot");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib window`
Expected: FAIL — `cannot find type KspaceWindow`, `no field window on Acquisition`.

- [ ] **Step 3: Write minimal implementation**

Add the enum near `Acquisition`:

```rust
/// Scanner-side reconstruction apodization. Distinct transfer functions with distinct PSFs, so
/// they are named rather than hidden behind one ambiguous scalar.
///
/// `None` is the default and is what the algorithm-validation fixture uses: `mrdegibbs`/Kellner
/// assume an unapodized rectangular window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KspaceWindow {
    None,
    Tukey { alpha: f64 },
    Hann,
    Fermi { radius: f64, width: f64 },
}

impl KspaceWindow {
    /// Window value at normalized frequency `(kx, ky)`, each in `[-0.5, 0.5]`.
    pub fn at(&self, kx: f64, ky: f64) -> f64 {
        let r = 2.0 * (kx * kx + ky * ky).sqrt(); // 0 at centre, 1 at the band edge
        match *self {
            KspaceWindow::None => 1.0,
            KspaceWindow::Hann => {
                if r >= 1.0 { 0.0 } else { 0.5 * (1.0 + (std::f64::consts::PI * r).cos()) }
            }
            KspaceWindow::Tukey { alpha } => {
                let a = alpha.clamp(0.0, 1.0);
                if r <= 1.0 - a {
                    1.0
                } else if r >= 1.0 {
                    0.0
                } else {
                    0.5 * (1.0 + (std::f64::consts::PI * (r - (1.0 - a)) / a.max(1e-12)).cos())
                }
            }
            KspaceWindow::Fermi { radius, width } => {
                1.0 / (1.0 + ((r - radius) / width.max(1e-12)).exp())
            }
        }
    }
}
```

Replace the `zero_ringing: f64` field with `pub window: KspaceWindow`, set `window: KspaceWindow::None` in `Default`, and **delete the entire `if acq.zero_ringing > 0.0 { ... }` block** from `build_coil_kspace` — truncation is now the crop, not a filter.

Apply the window in `simulate_slice`, **after** GRAPPA and before `inverse_2d`:

```rust
    // Reconstruction window: after GRAPPA, before the inverse transform, so it acts on
    // originally-acquired and GRAPPA-synthesized lines alike, and on the noise those lines
    // already carry: K_filtered = W(k) * [K_signal(k) + n(k)].
    if acq.window != KspaceWindow::None {
        for ks in coil_kspace.iter_mut() {
            for kyi in 0..ny {
                for kxi in 0..nx {
                    let kx = (kxi as f64 - xs as f64) / nx as f64;
                    let ky = (kyi as f64 - ys as f64) / ny as f64;
                    let w = acq.window.at(kx, ky);
                    let k = &mut ks[kxi + nx * kyi];
                    k.re *= w;
                    k.im *= w;
                }
            }
        }
    }
```

Update `src/bin/trxscan.rs:261`, replacing `zero_ringing: 6.0,` with `window: KspaceWindow::None,` and adding `KspaceWindow` to the `use trxscan::kspace::{...}` import. Delete the now-obsolete `gibbs_ringing_changes_the_image` test.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test && cargo build --release`
Expected: PASS; the binary still builds.

- [ ] **Step 5: Commit**

```bash
git add src/kspace.rs src/bin/trxscan.rs
git commit -m "feat(kspace): replace zero_ringing with named reconstruction windows"
```

---

## Self-Review

**Spec coverage.** 3.1 two grids → Tasks 2, 4. 3.1 windows → Task 12. 3.2 phase → Tasks 5, 6. 3.3 noise → Task 8. 3.4 asymmetry → Task 11. 3.5 benchmark outputs → Task 7. 4.1.1 sub-voxel profile → Task 2. 4.1.2 sign alternation → Task 1. 4.1.3 intrinsic ringing → Task 2. 4.1.4 residual rotation → Task 6. 4.1.5 controlled fields → Tasks 5, 6. 4.1.6 acquired-band convergence → Task 3. 4.1.7 covariance → Task 9. 4.1.8 sqrt(f) → Task 8. 4.1.9 PF → Task 10. 4.1.10 `o_min` → Task 3. 4.1.11 asymmetry → Task 11.

**Known gap, deliberate:** the `BenchmarkSlice` *producer* (wiring Stage A on the sim grid through `simulate_slice` twice, clean and noisy, and emitting NIfTIs) is not a task here. It needs `io.rs` changes behind the `io` feature and is the natural first task of the phases 8-10 plan, where the acceptance suite consumes it. Task 7 delivers the type and the block mean that producer will use, both fully tested.

**Type consistency.** `SliceInput` field names are identical in Tasks 2, 3, 6, 8, 9, 10, 11, 12. `simulate_slice(&inp, &acq)` argument order is constant. `block_mean_complex(hires, snx, sny, o)` matches its single definition. `KspaceWindow::at(kx, ky)` takes normalized frequencies everywhere. `phase_slice` takes `o` so coefficients stay oversampling-invariant, consistent with `PhaseModel::at` taking acquired-voxel units.
