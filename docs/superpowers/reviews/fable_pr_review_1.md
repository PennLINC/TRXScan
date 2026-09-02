## Review: `gibbs-ringing-realism-spec` (55 commits, 53 files, ~13,300 insertions vs `main`)

**Verdict: the branch is in strong shape and I'd approve it with two medium-severity fixes.** The core physics is correct — I verified the trickiest math by hand and against the analytic oracle — and all test suites pass locally: 85 + 2 Rust tests (default features), the `cli`-feature tests, and 64 Python tests (4 environment-dependent skips). Nothing I found invalidates the science; the two medium findings are about CI coverage and a mutation hazard in a script fallback path.

One housekeeping note first: your working tree has 18 modified files that are **pure line-ending churn** (exactly 3,248 insertions and 3,248 deletions, CRLF↔LF). Discard them or add a `.gitattributes` before committing anything, or they'll pollute the next commit.

### Core physics (reviewed by hand) — sound

The rewrite of `simulate_slice` around `SliceInput` (oversampled sim grid, truncation at nominal Nyquist during the forward DFT) is correct in the places most likely to be wrong:

- **Half-cell registration** (`src/kspace.rs`, the `xoff = (o-1)/2` logic): the sim cells composing an acquired voxel are symmetric about that voxel's centre, so the object is registered against the reconstruction grid at any oversampling factor. This is pinned by `crop_reproduces_the_analytic_profile_across_subvoxel_offsets`, which checks 16 sub-voxel edge positions against the independent closed-form oracle in `src/analytic.rs` to 5e-3 — exactly the right kind of test for this feature.
- **Acquired-band frequencies** are correctly cycles/FOV (`(kyi - ys)/sny` with sim-grid coordinates), and the noise normalization stays consistent with the unnormalized inverse DFT (empirically pinned by `noise_sd_scales_as_sqrt_sampled_fraction`).
- **`sampling_mask` as single source of truth** for signal, noise, and spikes genuinely fixes the old "band-limited signal, white noise" defect, and `noise_covariance_matches_the_sampling_mask` measures the lag along the phase-encode axis specifically so the test can't pass vacuously.
- `src/phase.rs` is honest about its epistemics (calibrated vs tuned vs heuristic parameters, documented in `hbcd_like()`), and the seed wiring is consistent: the CLI passes the same `cli.seed` into both `Acquisition::seed` and `simulate_acquisition_oversampled`.

### Findings

**Medium 1 — CI never executes the feature-gated tests.** `.github/workflows/ci.yml:39-50` runs `cargo test` with default features only, then merely `cargo check --features cli`. Since `default = []` excludes the `io` module, every unit test in `src/io.rs` — including the `hires_grid` geometry tests that pin the benchmark affine this branch depends on — runs in no CI step. The workflow's own comment identifies this exact blind spot but closes it only for compilation. One-line fix: `cargo test --features cli` (verified passing locally).

**Medium 2 — old-DIPY fallback can corrupt the caller's array.** In `scripts/run_unringing.py`, `run_method` does `mag = np.asarray(mag, float)` (no copy for float input) and the fallback at line 89 calls `gibbs_removal(mag)` with no kwargs — and older DIPY versions default to `inplace=True`. On such environments the caller's magnitude array is silently mutated, which would skew any scorer comparing "before" and "after". Fix: pass `mag.copy()` in the fallback branch.

**Low severity:**

- `src/microstructure.rs:1397` — the DIPY-oracle test early-returns because `tests/fixtures/force_moments.txt` is untracked, and the regeneration script it names (`tools/gen_force_fixtures.py`) is also untracked. The test is permanently green and decorative; either commit the fixture or the generator.
- `src/bin/trxscan_benchmark.rs:25-59` — numeric positionals use `.parse().ok().unwrap_or(default)`, so typos silently become defaults; and the usage string's `[only=<label>]` invites syntax the code doesn't parse, so a filter that matches nothing writes zero fixtures and exits 0. `written == 0` with a filter should be a hard error.
- `src/bin/trxscan.rs:73, 430-442` — `--fmap` is required but has no effect on the default oversampled path (only `--sim-fmap` feeds the model); either make it optional-unless-legacy or document it.
- `Acquisition::default_for(_nx, _ny)` in `src/kspace.rs` ignores both arguments — a placeholder that does nothing; drop it or implement it.
- `scripts/acceptance.py` rule (b) checks the window as a string inconsistently with how other rules match it; `scripts/prepare_acquisition_grid.py` `--voxel` is in source-voxel units, not mm, without saying so; `acceptance._median` returns the upper-middle element for even-length input.
- Minor, no action needed: the forward model samples `coil_sensitivity` on the sim grid without the half-cell offset used elsewhere, a sub-acquired-voxel misregistration versus the Roemer combine — negligible for the smooth Gaussian profiles used, but worth a comment if anyone ever adds sharper sensitivity maps.

Details behind the script and auxiliary-Rust findings are in the reports from [Python scripts review](ac2d08cd-0996-4fb4-a71f-4d38d5c521f5) and [Rust benchmark/IO review](a1ba9aba-3e1f-4d4c-929f-10b8012d14de) — the latter also verified benchmark determinism empirically (two runs, nine outputs byte-identical) and decoded the NIfTI headers (correct affine, dtype, and scaling).
