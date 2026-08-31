# Gibbs Benchmark and Calibration Implementation Plan (spec phases 8-10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the corrected forward model into a working benchmark — emit scoreable outputs, run real unringing methods against them, and calibrate the phase model against real complex DWI.

**Architecture:** Phases 0-7 made the simulator physically correct. This plan makes it *useful*: a producer that emits the four benchmark images, a scoring module that decomposes error rather than reporting one distance, a harness that runs `mrdegibbs`/DIPY/RPG across a factor grid, and calibration scripts that fit the phase model's free parameters against NIBS and ds006131.

**Tech Stack:** Rust 2021 (`io` feature: `nifti` 0.17, `ndarray`) for the producer; Python in the `linc311` micromamba env (nibabel 5.3.2, scipy 1.16.3, DIPY 1.12.1) for scoring harness and calibration; MRtrix3 CLI (`mrdegibbs`, `dwidenoise`) from `/usr/bin`.

**Spec:** `docs/superpowers/specs/2026-08-31-gibbs-ringing-realism-design.md` (revision 6, frozen). Sections 3.5, 4.2, 5.2 own this plan's requirements.

**Predecessor:** `docs/superpowers/plans/2026-08-31-gibbs-ringing-realism.md` (phases 0-7, complete, 12/12 tasks landed).

## Global Constraints

- **`noise_variance`** is the per-component variance of the reconstructed complex image at full sampling, single-coil, pre-combination. `E[|n|^2] = 2*noise_variance`.
- **`object-nominal` is a complex block MEAN**, `(1/o^2) * sum_j z_j`, derived from the *same* hires array that produced the acquired images. Never regenerate it by re-running Stage A at `o = 1`.
- **Oversampling default `o = 4`** (spec 3.1, measured). `o` is a parameter everywhere; do not hardcode it.
- **Scoring runs in the canonical Gibbs fixture** (`benchmark::gibbs_benchmark_acquisition`): distortion, T2*, eddy, ghosts, spikes, multi-coil and GRAPPA all off. Any run that re-enables them must not report raw distance to `object-nominal` as unringing error.
- **Phase error must be magnitude-masked or magnitude-weighted.** Use `dphi = arg(z_est * conj(z_ref))` with a threshold on `|z_ref|`. Never average raw phase difference over all voxels.
- **Correctness vs calibration vs behavioural** stay separate (spec 4.2). Nothing in this plan may make a phases-0-7 CI test depend on real data.
- Python via `micromamba run -n linc311 ...`. Do not install packages or create environments.
- Commit after every task. Run `cargo test` before each Rust commit.

## Verified Environment (checked at plan time)

| Dependency | Status |
|---|---|
| `mrdegibbs`, `dwidenoise`, `mrconvert` | `/usr/bin` — present (MRtrix3) |
| DIPY `gibbs_removal` | 1.12.1 in `linc311` — present |
| **RPG (PF-aware unringing)** | **NOT installed** — Task 6's PF arm is gated on it |
| NIBS dataset | `/mnt/c/Users/tsalo/Documents/datasets/nibs` — 274 G, 908 `part-phase` files |
| ds006131 | `/mnt/c/Users/tsalo/Documents/datasets/ds006131` — 28 G, 4852 `part-phase` files |
| `cargo check --features io` | clean |

**NIBS matches the simulator's shipping defaults**, which is why spec 4.2 makes it the primary source for sampling-mask-dependent behaviour: `EchoTime 0.088` vs `t_echo: 88.0`; `PartialFourier 0.75` vs `partial_fourier: 0.75`; `MultibandAccelerationFactor 3`; `PhaseEncodingDirection j-`; `AcquisitionMatrixPE 140`.

## Working Tree State (read before Task 1)

Carried forward from the predecessor plan, still true:

1. **`microstructure::tests::matches_dipy_closed_forms_on_fixtures` already fails** — it needs `tools/gen_force_fixtures.py`, which is absent. Baseline is **73 passed / 1 failed**. Check the FAILURE count, not the pass count. Do not fix it.
2. **All pre-existing tracked files show as modified** (CRLF working tree vs LF blobs, `core.autocrlf=false`). **Never `git add -A` or `git add .`.** Normalize any pre-existing file to LF before staging it:
   ```bash
   python3 -c "import io,sys;p=sys.argv[1];b=io.open(p,'rb').read();io.open(p,'wb').write(b.replace(b'\r\n',b'\n'))" <file>
   ```
   Then confirm with `git diff --numstat --cached` that staged counts reflect only real change. New files are LF already.
3. `scripts/__pycache__/` is gitignored; do not stage bytecode.

## Ordering, and why it differs from the spec

Spec 5 sequences these as 8 → 9 → 10. This plan runs **producer → 9 → 8 → 10**, for three measured reasons:

- Phase 9's acceptance suite **needs** the benchmark producer, which phases 0-7 deliberately deferred. It is the true first task.
- Phase 9 is fully unblocked (tools present, uses simulated data only); phase 8 depends on 274 G of real data and slower iteration.
- Phase 10 is **deprioritized**: measured wall time is 1.6 h for a full `o = 4` acquisition single-threaded, already parallel over volumes. The FFT path is a convenience, not a prerequisite.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/benchmark.rs` (modify) | Add the producer that fills `BenchmarkSlice`; already owns `block_mean_complex` and the fixture. |
| `src/io.rs` (modify) | Add a complex-pair NIfTI writer for the four benchmark images. |
| `src/bin/trxscan_benchmark.rs` (new) | CLI that emits a benchmark fixture set across the factor grid. |
| `scripts/score_unringing.py` (new) | Error decomposition: oscillatory residual, edge bias, sharpness, complex/mag/phase. |
| `scripts/run_unringing.py` (new) | Harness: `mrdegibbs`, DIPY, RPG (gated), no-op control. |
| `scripts/calibrate_phase.py` (new) | Phase 8: NIBS spatial statistics, ds006131 b-dependence with SNR correction. |
| `src/phase.rs` (modify) | Add the `HBCDLike` preset built from fitted constants. |

---

### Task 1: Benchmark producer

**Files:**
- Modify: `src/benchmark.rs`
- Test: `src/benchmark.rs`

**Interfaces:**
- Consumes: `kspace::{SliceInput, simulate_slice, phase_slice, Acquisition}`, `phase::{PhaseModel, ShotPhase}`, `benchmark::{block_mean_complex, gibbs_benchmark_acquisition, BenchmarkSlice}`.
- Produces: `pub fn produce_slice(comps: &[&[f32]], t2: &[f32], fmap: &[f32], model: &PhaseModel, shot: &ShotPhase, sim: [usize; 2], acq_matrix: [usize; 2], z: usize, nz: usize, acq: &Acquisition, bvec: [f64; 3], bval: f64, seed: u64) -> BenchmarkSlice`.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn producer_derives_nominal_from_the_same_realization() {
        use crate::kspace::{step_hires, Acquisition};
        use crate::phase::{DiffusionPhase, PhaseModel};
        let (nx, ny, o) = (16usize, 16usize, 4usize);
        let (snx, sny) = (nx * o, ny * o);
        let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
        let fmap = vec![0.0f32; snx * sny];
        let comps: [&[f32]; 1] = [&img];
        let model = PhaseModel { global: 0.3, ..PhaseModel::none() };
        let shot = DiffusionPhase { c_q: 0.0, sigma_dx: 0.0, sigma_rot: 0.0 }
            .shot(0.0, [0.0, 0.0, 0.0], 0, 0, 1);
        let acq = gibbs_benchmark_acquisition(&Acquisition {
            signal_scale: 1.0, noise_variance: 1.0, ..Acquisition::default()
        });
        let b = produce_slice(&comps, &[100.0], &fmap, &model, &shot,
                              [snx, sny], [nx, ny], 0, 1, &acq, [0.0; 3], 0.0, 7);

        assert_eq!(b.object_hires.len(), snx * sny);
        assert_eq!(b.object_nominal.len(), nx * ny);
        assert_eq!(b.acquired_clean.len(), nx * ny);
        assert_eq!(b.acquired_noisy.len(), nx * ny);

        // object_nominal must be exactly the block mean of THIS object_hires, not a re-run.
        let expect = block_mean_complex(&b.object_hires, snx, sny, o);
        for i in 0..nx * ny {
            assert!((b.object_nominal[i].0 - expect[i].0).abs() < 1e-12);
            assert!((b.object_nominal[i].1 - expect[i].1).abs() < 1e-12);
        }
        // The global phase must be present in the object, not silently dropped.
        let m = b.object_hires.iter().map(|z| z.1.abs()).fold(0.0f64, f64::max);
        assert!(m > 0.1, "global phase 0.3 rad should give a real imaginary part, got {m}");
    }

    #[test]
    fn clean_and_noisy_share_one_realization() {
        use crate::kspace::{step_hires, Acquisition};
        use crate::phase::{DiffusionPhase, PhaseModel};
        let (nx, ny, o) = (16usize, 16usize, 4usize);
        let (snx, sny) = (nx * o, ny * o);
        let img = step_hires(snx, sny, (nx as f64 / 2.0 + 0.5) * o as f64);
        let fmap = vec![0.0f32; snx * sny];
        let comps: [&[f32]; 1] = [&img];
        let shot = DiffusionPhase { c_q: 0.0, sigma_dx: 0.0, sigma_rot: 0.0 }
            .shot(0.0, [0.0, 0.0, 0.0], 0, 0, 1);
        let acq = gibbs_benchmark_acquisition(&Acquisition {
            signal_scale: 1.0, noise_variance: 4.0, ..Acquisition::default()
        });
        let b = produce_slice(&comps, &[100.0], &fmap, &PhaseModel::none(), &shot,
                              [snx, sny], [nx, ny], 0, 1, &acq, [0.0; 3], 0.0, 11);
        // Same object => the difference is noise alone, and it must be non-trivial.
        let d: f64 = (0..nx * ny)
            .map(|i| (b.acquired_noisy[i].0 - b.acquired_clean[i].0) as f64)
            .map(|v| v * v)
            .sum::<f64>()
            / (nx * ny) as f64;
        assert!(d.sqrt() > 0.5, "noisy and clean should differ by the noise: {}", d.sqrt());
        // ...but the underlying signal must be identical: clean is noise-free.
        let e: f64 = b.acquired_clean.iter().map(|p| (p.0 as f64).powi(2)).sum();
        assert!(e > 1.0, "clean image should carry the object, got energy {e}");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib benchmark`
Expected: FAIL with `cannot find function produce_slice`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/benchmark.rs`:

```rust
use crate::kspace::{phase_slice, simulate_slice, SliceInput};
use crate::phase::{PhaseModel, ShotPhase};

/// Produce the four co-registered benchmark images for one slice.
///
/// `object_nominal` is the block mean of the SAME `object_hires` array that produced the acquired
/// images, and `acquired_clean` / `acquired_noisy` share one object and phase realization. Both
/// properties are the point of this function: regenerating any of them independently would give a
/// different random draw and, for the nominal reference, would also lose intravoxel dephasing.
#[allow(clippy::too_many_arguments)]
pub fn produce_slice(
    comps: &[&[f32]],
    t2: &[f32],
    fmap: &[f32],
    model: &PhaseModel,
    shot: &ShotPhase,
    sim: [usize; 2],
    acq_matrix: [usize; 2],
    z: usize,
    nz: usize,
    acq: &Acquisition,
    bvec: [f64; 3],
    bval: f64,
    seed: u64,
) -> BenchmarkSlice {
    let [snx, sny] = sim;
    let [nx, ny] = acq_matrix;
    assert!(snx % nx == 0 && sny % ny == 0, "sim grid must be an integer multiple");
    let o = snx / nx;
    assert_eq!(o, sny / ny, "oversampling must match on both axes");

    let phi = phase_slice(model, shot, snx, sny, o, z, nz);

    // The complex object, exactly as the acquisition sees it.
    let object_hires: Vec<(f64, f64)> = (0..snx * sny)
        .map(|i| {
            let amp: f64 = comps.iter().map(|c| c[i] as f64).sum::<f64>() * acq.signal_scale;
            let (s, c) = phi[i].sin_cos();
            (amp * c, amp * s)
        })
        .collect();
    let object_nominal = block_mean_complex(&object_hires, snx, sny, o);

    let mk = |noise: f64| -> Vec<(f32, f32)> {
        let a = Acquisition { noise_variance: noise, ..acq.clone() };
        simulate_slice(
            &SliceInput {
                compartments: comps,
                t2,
                fmap,
                phase0: Some(&phi),
                sim,
                acq_matrix,
                z,
                nz,
                bvec,
                bval,
                slice_seed: seed,
            },
            &a,
        )
    };
    let acquired_clean = mk(0.0);
    let acquired_noisy = mk(acq.noise_variance);

    BenchmarkSlice { object_hires, object_nominal, acquired_clean, acquired_noisy }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS, 75 passed / 1 failed (the known environment failure).

- [ ] **Step 5: Commit**

```bash
python3 -c "import io,sys;p=sys.argv[1];b=io.open(p,'rb').read();io.open(p,'wb').write(b.replace(b'\r\n',b'\n'))" src/benchmark.rs
git add src/benchmark.rs
git commit -m "feat(benchmark): producer emitting the four scoreable images"
```

---

### Task 2: Complex benchmark NIfTI writer

**Files:**
- Modify: `src/io.rs`
- Test: `src/io.rs`

**Interfaces:**
- Consumes: `raster::Grid`, `benchmark::BenchmarkSlice`.
- Produces: `pub fn write_benchmark(out_prefix: &Path, dims: [usize; 3], o: usize, slices: &[BenchmarkSlice], grid: &Grid) -> R<()>` writing `<prefix>_desc-{objecthires,objectnominal,acquiredclean,acquirednoisy}_part-{mag,phase}_dwi.nii.gz`.

The hires volume has in-plane dims `o *` the nominal ones and needs its own affine with the in-plane columns divided by `o` and the half-cell origin shift, matching `scripts/prepare_acquisition_grid.py`.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hires_affine_preserves_the_fov() {
        // The hires volume must cover the same FOV as the nominal one, or the two references
        // cannot be compared voxel-for-voxel after block reduction.
        let g = Grid {
            dims: [8, 8, 2],
            voxel_to_world: [
                [1.7, 0.0, 0.0, -10.0],
                [0.0, 1.7, 0.0, -12.0],
                [0.0, 0.0, 1.7, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        let o = 4;
        let h = hires_grid(&g, o);
        assert_eq!(h.dims, [32, 32, 2]);
        for ax in 0..2 {
            let nom = g.dims[ax] as f64 * g.voxel_to_world[ax][ax];
            let hi = h.dims[ax] as f64 * h.voxel_to_world[ax][ax];
            assert!((nom - hi).abs() < 1e-9, "axis {ax}: FOV {nom} vs {hi}");
        }
        assert_eq!(h.voxel_to_world[2][2], g.voxel_to_world[2][2], "z must not be refined");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --features io --lib io`
Expected: FAIL with `cannot find function hires_grid`.

- [ ] **Step 3: Write minimal implementation**

```rust
/// The simulation grid's `Grid`: in-plane refined by `o`, same FOV, slice direction untouched.
///
/// The origin shifts by half the difference between the coarse and fine voxel sizes on each
/// refined axis, matching the half-cell registration the forward transform uses and the geometry
/// `scripts/prepare_acquisition_grid.py` writes.
pub fn hires_grid(g: &Grid, o: usize) -> Grid {
    let f = o as f64;
    let mut m = g.voxel_to_world;
    for r in 0..3 {
        for c in 0..2 {
            m[r][c] /= f;
        }
    }
    for r in 0..3 {
        m[r][3] = g.voxel_to_world[r][3]
            - 0.5 * g.voxel_to_world[r][0] * (1.0 - 1.0 / f)
            - 0.5 * g.voxel_to_world[r][1] * (1.0 - 1.0 / f);
    }
    Grid { dims: [g.dims[0] * o, g.dims[1] * o, g.dims[2]], voxel_to_world: m }
}
```

Then `write_benchmark` assembles each of the four images into magnitude/phase 4D arrays and reuses the existing `write_4d` helper, using `hires_grid(grid, o)` for the hires pair and `grid` for the other three.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --features io`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
python3 -c "import io,sys;p=sys.argv[1];b=io.open(p,'rb').read();io.open(p,'wb').write(b.replace(b'\r\n',b'\n'))" src/io.rs
git add src/io.rs
git commit -m "feat(io): write the four benchmark images with a correct hires affine"
```

---

### Task 3: Benchmark fixture CLI

**Files:**
- Create: `src/bin/trxscan_benchmark.rs`
- Modify: `Cargo.toml` (register the bin with `required-features = ["cli"]`)

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: a binary emitting one fixture set per point on the factor grid: `{full-Fourier, 6/8, 7/8} x {no phase, constant, ramp, diffusion} x {noiseless, noisy} x {window None, Hann}`, each a directory of the Task 2 outputs plus a `factors.json` recording the point.

- [ ] **Step 1: Write the failing test**

Integration test at `tests/benchmark_cli.rs`:

```rust
#[test]
fn factor_grid_is_complete_and_labelled() {
    // The grid is the acceptance suite's input; a missing arm silently weakens every conclusion.
    let g = trxscan::benchmark::factor_grid();
    assert_eq!(g.len(), 3 * 4 * 2 * 2, "expected 48 grid points, got {}", g.len());
    let pf: std::collections::BTreeSet<_> =
        g.iter().map(|p| format!("{:.3}", p.partial_fourier)).collect();
    assert_eq!(pf.len(), 3, "expected full/6-8/7-8, got {pf:?}");
    for p in &g {
        assert!(!p.label.is_empty(), "every grid point must be labelled for the results table");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --features cli --test benchmark_cli`
Expected: FAIL with `cannot find function factor_grid`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/benchmark.rs`:

```rust
/// One point on the acceptance-suite factor grid (spec 5.2).
#[derive(Debug, Clone)]
pub struct FactorPoint {
    pub label: String,
    pub partial_fourier: f64,
    pub phase: PhaseKind,
    pub noisy: bool,
    pub window: crate::kspace::KspaceWindow,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhaseKind { None, Constant, Ramp, Diffusion }

/// The full factor grid: 3 PF x 4 phase x 2 noise x 2 window = 48 points.
pub fn factor_grid() -> Vec<FactorPoint> {
    use crate::kspace::KspaceWindow;
    let mut v = Vec::new();
    for (pfl, pf) in [("full", 1.0), ("pf68", 0.75), ("pf78", 0.875)] {
        for (phl, ph) in [("nophase", PhaseKind::None), ("const", PhaseKind::Constant),
                          ("ramp", PhaseKind::Ramp), ("diff", PhaseKind::Diffusion)] {
            for (nl, noisy) in [("clean", false), ("noisy", true)] {
                for (wl, w) in [("nowin", KspaceWindow::None), ("hann", KspaceWindow::Hann)] {
                    v.push(FactorPoint {
                        label: format!("{pfl}_{phl}_{nl}_{wl}"),
                        partial_fourier: pf,
                        phase: ph,
                        noisy,
                        window: w,
                    });
                }
            }
        }
    }
    v
}
```

The binary iterates `factor_grid()`, builds the phantom, calls `produce_slice` per slice, and writes via `write_benchmark`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --features cli`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/benchmark.rs src/bin/trxscan_benchmark.rs Cargo.toml tests/benchmark_cli.rs
git commit -m "feat(benchmark): factor grid and fixture-emitting CLI"
```

---

### Task 4: Scoring — error decomposition

**Files:**
- Create: `scripts/score_unringing.py`
- Test: `scripts/test_score_unringing.py`

**Interfaces:**
- Produces: `score(est_mag, est_phase, ref_mag, ref_phase, mask=None) -> dict` with keys `oscillatory_residual`, `edge_location_bias`, `edge_sharpness`, `complex_rmse`, `magnitude_rmse`, `phase_rmse_masked`.

A single distance is explicitly not the deliverable: a method that merely blurs must score well on `oscillatory_residual` and badly on `edge_sharpness`, and the suite's value is that separation.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from score_unringing import score

def _edge(n=64, ring=0.0, blur=0.0):
    x = np.arange(n)
    img = (x >= n // 2).astype(float)
    if ring:
        img = img + ring * np.sin(np.pi * x) * np.exp(-np.abs(x - n // 2) / 6.0)
    if blur:
        k = np.exp(-0.5 * ((np.arange(-9, 10)) / blur) ** 2); k /= k.sum()
        img = np.convolve(img, k, mode="same")
    return np.tile(img, (n, 1))

def test_blurring_and_unringing_are_distinguishable():
    ref = _edge()
    rung = _edge(ring=0.09)                 # ringing, sharp
    blurred = _edge(blur=2.0)               # no ringing, soft
    z = np.zeros_like(ref)
    s_ring = score(rung, z, ref, z)
    s_blur = score(blurred, z, ref, z)
    # The blurred image wins on oscillation and loses on sharpness. If one number were reported,
    # these two failure modes would be indistinguishable.
    assert s_blur["oscillatory_residual"] < s_ring["oscillatory_residual"]
    assert s_blur["edge_sharpness"] < s_ring["edge_sharpness"]

def test_phase_error_ignores_near_zero_magnitude():
    n = 32
    ref_mag = np.zeros((n, n)); ref_mag[:, n // 2:] = 1.0
    ref_ph = np.zeros((n, n))
    est_ph = ref_ph.copy()
    est_ph[:, : n // 2] = 3.0            # garbage phase where magnitude is zero
    s = score(ref_mag, est_ph, ref_mag, ref_ph)
    assert s["phase_rmse_masked"] < 1e-9, "phase error must be magnitude-masked"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n linc311 python -m pytest scripts/test_score_unringing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'score_unringing'`.

- [ ] **Step 3: Write minimal implementation**

`scripts/score_unringing.py` implementing:
- `oscillatory_residual`: RMS of the band-passed difference near edges (difference minus its local mean), restricted to a dilated edge mask from `ref`.
- `edge_location_bias`: sub-voxel shift of the estimate's edge relative to `ref`, from the centroid of `|d/dx|` along each profile crossing the edge.
- `edge_sharpness`: mean `max|d/dx|` across edge profiles, normalized by `ref`'s. A blurrer scores below 1.
- `complex_rmse` / `magnitude_rmse`: direct.
- `phase_rmse_masked`: `angle(z_est * conj(z_ref))` weighted by `|z_ref|`, with voxels below `0.1 * max|z_ref|` excluded.

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n linc311 python -m pytest scripts/test_score_unringing.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/score_unringing.py scripts/test_score_unringing.py
git commit -m "feat(scripts): error decomposition separating unringing from blurring"
```

---

### Task 5: Unringing harness

**Files:**
- Create: `scripts/run_unringing.py`
- Test: `scripts/test_run_unringing.py`

**Interfaces:**
- Produces: `run_method(name, mag, phase) -> (mag, phase)` for `"none"` (control), `"mrdegibbs"`, `"dipy"`, `"rpg"`.

`"none"` is a required arm: without a no-unringing control the suite cannot tell whether a method helped or merely changed the image.

**RPG is not installed** (verified at plan time). `run_method("rpg", ...)` must raise a clear `MethodUnavailable` that the runner records as skipped — never silently fall back to another method, which would corrupt the comparison.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np, pytest
from run_unringing import run_method, MethodUnavailable, available_methods

def test_control_is_identity():
    m = np.random.RandomState(0).rand(16, 16); p = np.zeros_like(m)
    om, op = run_method("none", m, p)
    assert np.allclose(om, m) and np.allclose(op, p)

def test_unavailable_method_raises_rather_than_substituting():
    m = np.zeros((16, 16)); p = np.zeros_like(m)
    if "rpg" not in available_methods():
        with pytest.raises(MethodUnavailable):
            run_method("rpg", m, p)

def test_dipy_reduces_ringing_on_a_truncated_edge():
    n = 64
    x = np.arange(n)
    obj = (x >= n // 2).astype(float)
    k = np.fft.fftshift(np.fft.fft(obj))
    keep = np.zeros(n, bool); keep[n // 4 : 3 * n // 4] = True
    trunc = np.real(np.fft.ifft(np.fft.ifftshift(k * keep)))
    img = np.tile(trunc, (n, 1))
    out, _ = run_method("dipy", img, np.zeros_like(img))
    ripple = lambda a: np.std(a[:, 3 * n // 4 :])
    assert ripple(out) < ripple(img), "dipy gibbs_removal should reduce the ripple"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n linc311 python -m pytest scripts/test_run_unringing.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

`scripts/run_unringing.py`: `available_methods()` probes `shutil.which("mrdegibbs")`, imports DIPY, and checks for RPG; `run_method` dispatches, shelling out to `mrdegibbs` via a temporary NIfTI for the MRtrix arm and calling `dipy.denoise.gibbs.gibbs_removal` directly for the DIPY arm. Raise `MethodUnavailable` for anything absent.

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n linc311 python -m pytest scripts/test_run_unringing.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_unringing.py scripts/test_run_unringing.py
git commit -m "feat(scripts): unringing harness with an explicit no-op control"
```

---

### Task 6: Acceptance runner

**Files:**
- Create: `scripts/acceptance.py`
- Test: `scripts/test_acceptance.py`

**Interfaces:**
- Consumes: Tasks 3-5.
- Produces: a results table (one row per grid point x method) plus the acceptance check.

**The acceptance criterion is not a ranking.** It is that varying PF, phase, noise and windowing produces *interpretable and physically consistent* changes in method behaviour. Concretely, assert: (a) every method beats the `none` control on `oscillatory_residual` under full Fourier with no phase; (b) magnitude-only methods degrade on `phase_rmse_masked` relative to the control when object phase is present; (c) all methods do worse on 6/8 than on full Fourier. Record, do not assert, relative rankings.

- [ ] **Step 1: Write the failing test**

```python
from acceptance import check_consistency

def test_consistency_rules_catch_an_implausible_result():
    # A method that "improves" as partial Fourier gets more aggressive is not physical.
    rows = [
        {"method": "x", "pf": 1.0,  "phase": "nophase", "oscillatory_residual": 0.05},
        {"method": "x", "pf": 0.75, "phase": "nophase", "oscillatory_residual": 0.01},
        {"method": "none", "pf": 1.0, "phase": "nophase", "oscillatory_residual": 0.09},
    ]
    ok, reasons = check_consistency(rows)
    assert not ok
    assert any("partial fourier" in r.lower() for r in reasons)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n linc311 python -m pytest scripts/test_acceptance.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

`scripts/acceptance.py` with `check_consistency(rows) -> (bool, list[str])` implementing rules (a)-(c), and a `main()` that walks the fixture directories, runs each method, scores against `object-nominal`, and writes `results.csv` plus a Markdown summary.

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n linc311 python -m pytest scripts/test_acceptance.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/acceptance.py scripts/test_acceptance.py
git commit -m "feat(scripts): acceptance runner with physical-consistency rules"
```

---

### Task 7: Phase 8a — NIBS spatial phase statistics

**Files:**
- Create: `scripts/calibrate_phase.py`
- Test: `scripts/test_calibrate_phase.py`

**Interfaces:**
- Produces: `phase_spatial_stats(phase_vol, mag_vol, mask) -> dict` with `grad_rms`, `corr_length_vox`, `wrap_density`; and a CLI writing `scripts/calibration/nibs_phase.json`.

Data: `/mnt/c/Users/tsalo/Documents/datasets/nibs`, `*_part-phase_dwi.nii.gz` with matching `part-mag`. Start with the b=0 volumes, where SNR is highest and no diffusion phase is present, so this isolates term 2.

**Term 2 is tuned, not fitted** (spec 4.2). Reconstructed phase mixes magnetization phase, coil phase and combination, scanner conventions and possibly reconstruction filtering; only some of that precedes Fourier encoding. Record these statistics as *effective benchmark targets*, and say so in the output JSON.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from calibrate_phase import phase_spatial_stats

def test_stats_recover_a_known_synthetic_field():
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    true_grad = 0.05
    ph = np.angle(np.exp(1j * (true_grad * xx)))     # wrapped linear ramp
    mag = np.ones((n, n)); mask = mag > 0.5
    s = phase_spatial_stats(ph[None], mag[None], mask[None])
    assert abs(s["grad_rms"] - true_grad) < 0.01, s
    assert s["wrap_density"] > 0, "a ramp spanning many cycles must show wraps"

def test_flat_phase_has_no_wraps_and_zero_gradient():
    n = 32
    ph = np.zeros((1, n, n)); mag = np.ones((1, n, n)); mask = mag > 0.5
    s = phase_spatial_stats(ph, mag, mask)
    assert s["grad_rms"] < 1e-9 and s["wrap_density"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n linc311 python -m pytest scripts/test_calibrate_phase.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

Gradient of the *unwrapped local* phase via `np.angle(np.exp(1j*(ph[...,1:]-ph[...,:-1])))`; correlation length from the autocorrelation of the demeaned complex unit-phase field; wrap density as the fraction of neighbouring pairs whose phase difference exceeds pi.

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n linc311 python -m pytest scripts/test_calibrate_phase.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_phase.py scripts/test_calibrate_phase.py
git commit -m "feat(scripts): NIBS spatial phase statistics for term 2 tuning"
```

---

### Task 8: Phase 8b — b-dependence with SNR correction

**Files:**
- Modify: `scripts/calibrate_phase.py`
- Test: `scripts/test_calibrate_phase.py`

**Interfaces:**
- Produces: `fit_b_dependence(shots, bvals, snr) -> dict` with `c_q`, `p`, `p_naive`, `snr_corrected: True`.

Data: ds006131 (DSI, wide b range, the reason spec 4.2 assigns it this role).

**The SNR correction is the point of this task.** Magnitude SNR falls with b, so measured phase variance rises from thermal noise alone; fitting `Var(phi) ~ b^p` to raw wrapped phase absorbs that and biases `p` upward. Model observed variance as `Var_signal(b) + 1/SNR(b)^2` and fit both terms, reporting `p_naive` alongside `p` so the size of the bias is visible.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from calibrate_phase import fit_b_dependence

def test_snr_correction_removes_the_upward_bias():
    rng = np.random.RandomState(0)
    bvals = np.array([1000, 2000, 3000, 4000, 5000], float)
    p_true, a = 0.5, 0.02
    sig = a * bvals ** p_true                     # true motion phase SD
    snr = 40.0 * np.exp(-bvals / 3000.0)          # SNR falls with b
    obs = np.sqrt(sig ** 2 + (1.0 / snr) ** 2)    # thermal phase noise adds in quadrature
    shots = [rng.normal(0, s, 4000) for s in obs]
    out = fit_b_dependence(shots, bvals, snr)
    assert abs(out["p"] - p_true) < 0.12, out
    assert out["p_naive"] > out["p"] + 0.05, f"naive fit should be biased upward: {out}"
    assert out["snr_corrected"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n linc311 python -m pytest scripts/test_calibrate_phase.py::test_snr_correction_removes_the_upward_bias -v`
Expected: FAIL — `cannot import name 'fit_b_dependence'`.

- [ ] **Step 3: Write minimal implementation**

`p_naive` from a log-log least squares on raw SD; `p` and `c_q` from a `scipy.optimize.curve_fit` of `sqrt((a*b**p)**2 + (1/snr)**2)` against the observed SD.

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n linc311 python -m pytest scripts/test_calibrate_phase.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_phase.py scripts/test_calibrate_phase.py
git commit -m "feat(scripts): SNR-corrected b-dependence fit for the diffusion phase"
```

---

### Task 9: `PhaseModel::hbcd_like` preset

**Files:**
- Modify: `src/phase.rs`
- Create: `src/phase_presets.rs` (constants only, generated by Task 7-8 output)
- Test: `src/phase.rs`

**Interfaces:**
- Produces: `pub fn hbcd_like() -> PhaseModel` on `PhaseModel`.

Constants come from `scripts/calibration/*.json`. Commit them as data with the fitting procedure recorded alongside (spec 4.2 requires the chosen SNR-handling method be recorded with the constants).

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn hbcd_like_preset_is_nondegenerate_and_b_dependent() {
        let m = PhaseModel::hbcd_like();
        // b = 0 must still be phase-free.
        let s0 = m.diffusion.shot(0.0, [1.0, 0.0, 0.0], 1, 0, 5);
        assert_eq!(s0.at([3.0, 1.0, 0.0]), 0.0);
        // and a real shell must produce a non-trivial, sqrt(b)-scaling phase
        let a = m.diffusion.shot(1000.0, [1.0, 0.0, 0.0], 1, 0, 5);
        let b = m.diffusion.shot(4000.0, [1.0, 0.0, 0.0], 1, 0, 5);
        let r = [2.0, -1.0, 0.0];
        assert!(a.at(r).abs() > 1e-6, "calibrated model should give real phase");
        assert!((b.at(r) / a.at(r) - 2.0).abs() < 1e-9, "sqrt(b) scaling");
        // the background field must vary across the FOV
        assert!((m.background.at(8.0, 0.0, 0.0) - m.background.at(-8.0, 0.0, 0.0)).abs() > 1e-3);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib hbcd_like`
Expected: FAIL — `no function hbcd_like`.

- [ ] **Step 3: Write minimal implementation**

`hbcd_like()` returning a `PhaseModel` built from the committed constants, with a doc comment recording the source datasets, the fit date, and the SNR-handling method used.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
python3 -c "import io,sys;p=sys.argv[1];b=io.open(p,'rb').read();io.open(p,'wb').write(b.replace(b'\r\n',b'\n'))" src/phase.rs
git add src/phase.rs src/phase_presets.rs scripts/calibration/
git commit -m "feat(phase): HBCDLike preset from NIBS and ds006131 calibration"
```

---

### Task 10: Phase 8c — behavioural validation against real data

**Files:**
- Create: `scripts/behavioural_validation.py`
- Test: `scripts/test_behavioural_validation.py`

**Interfaces:**
- Produces: a report comparing simulated vs real data under `mrdegibbs` and `dwidenoise`: variance removed, the spatial-frequency profile of (before - after), and `dwidenoise`'s noise-level estimate with the autocorrelation of its residual.

This is the highest-value real-data check because it tests the nominal-Nyquist cutoff *behaviourally*, using a tool that assumes it. It is a plausibility comparison, **not** a source of theoretical constants — the analytic covariance stays derived from the simulator's own reconstruction operator (spec 4.2).

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from behavioural_validation import spectral_profile, compare_profiles

def test_profile_localises_ringing_at_nyquist():
    n = 64
    x = np.arange(n)
    nyq = np.tile(((-1.0) ** x), (n, 1))          # pure Nyquist ripple
    prof = spectral_profile(nyq)
    assert np.argmax(prof) >= len(prof) - 2, "Nyquist ripple must peak in the top bin"

def test_comparison_flags_a_frequency_mismatch():
    n = 64
    x = np.arange(n)
    at_nyquist = np.tile((-1.0) ** x, (n, 1))
    slower = np.tile(np.cos(2 * np.pi * x / 3.0), (n, 1))   # period 3, not 2
    ok, detail = compare_profiles(spectral_profile(at_nyquist), spectral_profile(slower))
    assert not ok, detail
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n linc311 python -m pytest scripts/test_behavioural_validation.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

`spectral_profile` returns the radially-binned power spectrum of the (before - after) difference; `compare_profiles` compares peak bin and centroid within a tolerance and returns a reason on mismatch.

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n linc311 python -m pytest scripts/test_behavioural_validation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/behavioural_validation.py scripts/test_behavioural_validation.py
git commit -m "feat(scripts): behavioural validation of the ringing frequency vs real data"
```

---

### Task 11: Phase 10 — parallelism and memory, measurement-gated

**Files:**
- Modify: `src/kspace.rs` (only if the measurement demands it)
- Create: `benches/` or an example that records the measurement

**This task is gated on its own measurement and may correctly end in no code change.**

Measured at plan time (release, 108x152 in-plane, single-threaded): 124 ms/slice at `o=1`, 252 ms at `o=2`, **750 ms at `o=4`** — 1.6 h for a 104-slice, 75-volume acquisition, and `simulate_acquisition` is already parallel over volumes under the `par` feature. The FFT path is therefore a convenience, not a prerequisite.

- [ ] **Step 1: Measure before changing anything**

Record: wall time for a full `o=4` acquisition with `--features par`, and peak RSS. Compare against the 4x memory estimate in spec 6.

- [ ] **Step 2: Decide, and record the decision**

If parallel wall time is acceptable and peak RSS fits, **write the measurement into spec 6 and stop.** Do not implement the FFT path or z-slab streaming speculatively — spec 6 defers both explicitly until measurement demands them.

- [ ] **Step 3: Only if the measurement demands it**

Implement z-slab streaming first (Stage B is already per-slice and z is not oversampled, so it is the smaller change), and only then the `rustfft` path — which must be validated against the direct sums, since those are the comparison oracle.

- [ ] **Step 4: Verify**

Run: `cargo test --features par` and confirm the analytic profile tests from phases 0-7 still pass unchanged.

- [ ] **Step 5: Commit**

```bash
git add -- <only the files you actually changed>
git commit -m "perf(kspace): <measured outcome, or 'record measurement, no change needed'>"
```

---

## Self-Review

**Spec coverage.** 3.5 producer -> Tasks 1-3. 3.5.5 scoring decomposition + magnitude-masked phase -> Task 4. 5.2 acceptance suite and its factor grid -> Tasks 3, 5, 6. 4.2 phase statistics (term 2, tuned) -> Task 7. 4.2 b-dependence with SNR handling (term 3, fitted) -> Task 8. 4.2 named preset -> Task 9. 4.2 behavioural validation -> Task 10. Spec 6 optimization, measurement-gated -> Task 11.

**Known gaps, deliberate:**
- **RPG is not installed**, so Task 5's PF arm raises `MethodUnavailable` and Task 6 records it as skipped. The 6/8 and 7/8 grid arms still run for the other methods; only the PF-*aware* comparison is missing. Installing or vendoring RPG would close it.
- `--oversample N --myelin` still writes no `myelin.nii.gz` (carried from the predecessor plan). Only matters if the myelin preset is used for benchmark phantoms.
- Task 11 may correctly produce no code.

**Type consistency.** `produce_slice` returns `BenchmarkSlice` as defined in phases 0-7 and used by Task 2's writer. `FactorPoint.window` is `kspace::KspaceWindow`, matching Task 12 of the predecessor plan. `score()`'s return keys are consumed verbatim by `check_consistency` in Task 6 and by Task 10's report. `phase_spatial_stats` and `fit_b_dependence` both write into `scripts/calibration/`, read by Task 9.
