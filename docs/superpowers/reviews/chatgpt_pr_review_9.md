I think the third agent’s review is strong, and it found a few real issues that my previous review missed.  After checking each finding against the current PR head (`eac04449be43163e1b03cb0b0e0a622f5030fa95`), I would **revise my previous “approve as-is” recommendation**: the physics and benchmark design remain sound, but I would make a small set of fixes before merging.

The current PR is mergeable and no longer a draft, and its latest CI run is green.   A green run does not contradict the CI finding below, because the missing feature-gated tests simply are not part of that run.

| Third-agent finding                                                                         | My assessment                                   |                                          Severity |
| ------------------------------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------: |
| CI does not execute feature-gated `io`/`cli` tests                                          | **Confirmed**                                   |                     **Medium / fix before merge** |
| Old-DIPY fallback can mutate caller array                                                   | **Confirmed**                                   |                     **Medium / fix before merge** |
| `trxscan-benchmark` silently accepts malformed arguments / broken documented `only=` syntax | **Confirmed, and understated**                  |                     **Medium / fix before merge** |
| DIPY/FORCE oracle fixture and generator absent                                              | **Confirmed**                                   |                                               Low |
| `--fmap` required but unused for default oversampled output                                 | **Confirmed**                                   |                                               Low |
| `Acquisition::default_for(nx, ny)` ignores arguments                                        | **Confirmed**                                   |                                               Low |
| Acceptance window-string inconsistency                                                      | **Technically true, little current impact**     |                                          Very low |
| `_median` is not actually a median for even N                                               | **Confirmed and currently active**              |                                     Low, easy fix |
| `prepare_acquisition_grid --voxel` assumes source voxels are 1 mm                           | **Confirmed**                                   | Low for intended data; more important generically |
| Coil-sensitivity half-cell offset mismatch                                                  | **Confirmed, negligible for current benchmark** |                                          Very low |
| Line-ending churn in working tree                                                           | **Not a PR-code finding**                       |                                      Housekeeping |

### 1. CI coverage is definitely incomplete

This is exactly as the third agent says. `default = []`; `cli` implies `io`; and `src/lib.rs` only compiles `io` under `#[cfg(feature = "io")]`.

Yet CI does:

* `cargo test` with default features;
* `cargo check --features cli` for the binaries;
* `cargo check --features cli,par` for the production parallel path.

It never does `cargo test --features cli`.

So feature-gated `io.rs` tests can regress while CI remains green. This is especially unfortunate because the simulation/acquisition-grid affine is scientifically important to this PR.

**I would require `cargo test --features cli` before merge.** If runtime is acceptable, `cargo test --features cli,par` would be even better, but the former closes the identified hole.

### 2. The DIPY mutation finding is real

Current code is:

```python
mag = np.asarray(mag, float)
...
try:
    return gibbs_removal(mag, inplace=False, num_processes=1)
except TypeError:
    return gibbs_removal(mag)
```

For a float ndarray, `np.asarray(..., float)` normally returns the same backing array. The fallback therefore passes caller-owned memory into DIPY.

DIPY 1.12 still documents `inplace=True` as the default. Older DIPY history is even more relevant: the `inplace` option was introduced specifically after reports that `gibbs_removal` overwrote the input. ([docs.dipy.org][1])

In the current default method order this is less catastrophic than the review wording might suggest because DIPY normally comes after the other available methods. But `run_suite()` reuses `acq_m` across methods, so a custom ordering, another subsequently added method, or direct use of `run_method()` can absolutely observe the mutation.

The fix is trivial and correct:

```python
return gibbs_removal(mag.copy())
```

in the fallback. I would also add a monkeypatched test that emulates an old mutating DIPY implementation and verifies that `run_method()` preserves its input.

### 3. I would promote the benchmark-CLI issue from low to medium

The third agent understates this one.

The usage says:

```text
trxscan-benchmark <out_dir>
  [matrix=64] [oversample=4] [slices=1] [only=<label>]
```

but the code actually does:

```rust
a[2].parse::<usize>()
a[3].parse::<usize>()
a[4].parse::<usize>()
let only = a.get(5).cloned();
```

That means a user following the documented interface gets:

* `matrix=128` → parse failure → silently **64**;
* `oversample=8` → parse failure → silently **4**;
* `slices=3` → parse failure → silently **1**;
* `only=pf68_diff_nowin` → compared literally against `pf68_diff_nowin`, matches nothing.

And a no-match filter ends with:

```text
wrote 0 fixture set(s)
```

and success.

For a benchmark intended to generate carefully controlled scientific fixtures, silently running a different configuration than the user requested is exactly the sort of failure mode worth eliminating.

I would fix this before merge, preferably by using Clap rather than hand-parsing. At minimum:

* invalid numeric inputs must error;
* documented syntax must match accepted syntax;
* an explicitly supplied filter that matches zero fixtures must error.

### 4. The missing FORCE/DIPY fixture is also real

Both:

* `tests/fixtures/force_moments.txt`
* `tools/gen_force_fixtures.py`

are absent from the current PR tree.

Meanwhile the supposedly oracle-backed test explicitly skips when the fixture is absent.

So on a clean checkout that particular DIPY-oracle comparison is not a test at all. I agree with the third agent that calling it “decorative” is fair in CI.

I would not block this Gibbs PR on it because microstructure is peripheral to the Gibbs work, but I would prefer to commit at least the generator, and ideally the deterministic fixture as well.

### 5. `--fmap` really is redundant on the normal path

`--fmap` is a required `PathBuf`, while `--sim-fmap` is optional at parsing time. With `oversample > 1`, the code requires `--sim-fmap` and Stage B loads that. `--fmap` is still separately loaded and grid-validated, but its values do not affect the oversampled acquisition.

The acquisition-grid `fmap` is only actually supplied to the legacy `oversample=1` acquisition.

So a default run can fail because a required file is missing or invalid even though that file would never influence its output.

I agree with the suggested cleanup: make acquisition-grid `--fmap` required only for the legacy path, while `--sim-fmap` is required for the normal oversampled path. Low severity.

### 6. `default_for()` is indeed a no-op

The code literally contains:

```rust
pub fn default_for(_nx: usize, _ny: usize) -> Self {
    Acquisition::default()
}
```

This is harmless today but misleading API design. I would remove it unless matrix-dependent defaults are imminent.

### 7. The acceptance findings need to be separated

The **window-string inconsistency** is real but basically harmless for the generated benchmark. The fixture writer serializes `KspaceWindow::None` as `"None"`, which is handled. So differences among `None`, `"None"`, and `"none"` mostly affect hand-created rows or future callers.

The **median finding is more substantive than that**.

`_median()` implements:

```python
v[len(v) // 2]
```

not the conventional median for even-sized populations.

And the actual factor grid has **four phase conditions** and three PF levels.  Therefore:

* rule (a)'s full-Fourier/no-window control population has 4 entries;
* rule (b)'s no-window PF-aware control population has 12.

Both are even. So this is not merely a theoretical future problem: the current “median” gates are actually using the upper middle sample.

I doubt it changes the current PASS, but because the fix is trivial, I would change this to `statistics.median()` and rerun the acceptance suite.

### 8. The `--voxel` units criticism is correct, with an important scope qualification

The CLI says:

> acquisition voxel size, mm iso

but the geometry is built as:

```python
M[:3, :3] = np.diag([VOX] * 3)
affine = A @ M
```

and the output shape is derived from source **voxel indices** divided by `VOX`.

Therefore `VOX=1.7` means 1.7 mm only when the source anatomical grid is 1 mm isotropic.

For the bundled data this is apparently exactly the assumption—the script itself describes the bundled anatomical inputs and the design repeatedly identifies them as 1-mm anatomicals. So it does not invalidate the current benchmark preparation.

But if the script is supposed to accept arbitrary anatomical NIfTIs, its units are wrong. Either:

* implement the target spacing in physical millimetres using the source affine/zooms; or
* explicitly state and enforce “input must be 1-mm isotropic.”

I favor the former.

### 9. The coil-sensitivity observation is technically correct but non-blocking

The main forward model correctly shifts spatial coordinates by the half-cell offset:

```rust
x - sxs - xoff
y - sys - yoff
```

while `coil_sensitivity()` simply sees `x`, `y`, `snx`, `sny` and centers its Gaussian at `snx/2, sny/2`.

So at `o > 1` its physical field is shifted by `(o-1)/2` simulation cells relative to the half-cell-registered object.

I agree with the third agent that this is negligible for the extremely smooth coil sensitivities currently used. More importantly, the **canonical Gibbs benchmark forces `n_coils=1`**, so the sensitivity is uniform and this has exactly no impact on the benchmark whose validity we have been reviewing.

It is a worthwhile comment/fix for future multi-coil realism, not a merge blocker.

### Line-ending note

I would not count the line-ending observation as a PR finding. The third agent is describing its local working tree, not committed branch contents.  The PR description itself currently mentions a similar set of **unstaged** line-ending-only modifications.

A `.gitattributes` is sensible housekeeping, but there is no line-ending churn in the PR that needs reviewing unless those files get staged later.

## On its physics assessment

I agree with the third agent's positive assessment of the core physics.  It independently focused on the same areas that drove the earlier review rounds—half-cell registration, acquired-band frequency convention, mask-consistent noise, phase modeling—and reached the same conclusion: **none of these new findings undermine the Gibbs simulation or the scientific conclusions of the benchmark.**

That substantially increases my confidence in that part of the PR rather than decreasing it.

## Revised merge recommendation

I would now make **four small changes before merge**:

1. add `cargo test --features cli` to CI;
2. copy `mag` in the old-DIPY fallback;
3. fix the `trxscan-benchmark` CLI so documented arguments cannot silently become defaults or zero outputs;
4. replace the hand-written `_median()` with a real median and rerun acceptance.

The first three are the ones I care about most. None requires redesigning or revalidating the Gibbs forward model.

Everything else in the third-agent review can reasonably be follow-up cleanup.

So I would characterize the third review as: **scientifically reassuring, technically useful, and mostly correct; it found two real medium issues I missed, plus a benchmark CLI issue that I think deserves medium rather than low severity.**

[1]: https://docs.dipy.org/stable/reference/dipy.denoise.html?utm_source=chatgpt.com "denoise — dipy 1.12.0 documentation"
