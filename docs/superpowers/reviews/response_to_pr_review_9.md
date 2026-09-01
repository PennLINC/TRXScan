# Response to review round 8-9 (Fable review + ChatGPT cross-review)

Two reviews this round: `fable_pr_review_1.md`, an independent pass over the branch, and
`chatgpt_pr_review_9.md`, which checked each of its findings against the PR head and revised its
own earlier "approve as-is" to "four small changes before merge".

**All four are done, plus every low-severity item except one, which is named and scoped out
below.** Commit `24d81c6`. Rust: 94 tests under `--features cli`, 87 under default features, 72
Python tests. `ACCEPTANCE: PASS` on a regenerated 24-set factor grid.

I'll take the two medium findings first, since they are the ones neither of the previous eight
review rounds caught, then the rest in the order the cross-review ranked them.

---

## 1. CI never executed the feature-gated tests — confirmed, fixed

Both reviewers are right, and this is the finding I'm most glad arrived. `default = []`,
`cli` implies `io`, and `src/lib.rs` compiles the `io` module only under `#[cfg(feature = "io")]`.
The workflow ran `cargo test` (default features) and then `cargo check --features cli`. So every
unit test in `src/io.rs` executed in **no CI step at all** — including the `hires_grid` geometry
tests that pin the benchmark affine, which is one of the more scientifically load-bearing things
on this branch.

Worse, the workflow's own comment identified exactly this blind spot and then closed it only for
compilation. I wrote that comment. It described the right problem and shipped the wrong remedy.

```yaml
- name: Test the feature-gated code and the production binaries
  run: cargo test --features cli --all-targets --verbose
- name: Test the parallel path
  run: cargo test --features cli,par --all-targets --verbose
```

I took the cross-review's stronger option rather than its minimum: both feature sets now *run*
rather than *check*. `--all-targets` still builds the binaries, so it subsumes the
`cargo check --bin` steps it replaces rather than sitting alongside them. Locally both pass; the
`par` run adds about two seconds.

## 2. Old-DIPY fallback could mutate the caller's array — confirmed, fixed, and pinned

Confirmed exactly as described. The chain is:

- `run_method` does `mag = np.asarray(mag, float)`, which does **not** copy a float64 ndarray;
- `_run_dipy` falls through to `gibbs_removal(mag)` on a `TypeError`;
- `gibbs_removal` defaults to `inplace=True`, and the `inplace` keyword exists only on versions
  new enough to take the branch *above* the fallback — so on precisely the versions that reach
  the fallback, the input is overwritten;
- `run_suite` reuses one `acq_m` across every method in the loop.

The cross-review's qualification is fair — DIPY currently runs after the other methods in the
default order, so the blast radius today is small. But the ordering is not a guarantee, and
`run_method` is a public entry point. Now `gibbs_removal(mag.copy())`, with a comment explaining
that the copy is load-bearing rather than defensive, so nobody optimises it away later.

I also took the suggested test, and made it environment-independent: it fakes the whole
`dipy.denoise.gibbs` module with an implementation that rejects the modern kwargs (forcing the
fallback branch) and then overwrites its argument. It runs whether or not DIPY is installed —
important, because the CI Python environment deliberately has neither DIPY nor nibabel.

Verified it is a real test rather than a decorative one: reverting the `.copy()` makes it fail
with *"old-DIPY fallback overwrote the caller's magnitude"*, and restoring it makes it pass.

## 3. `trxscan-benchmark` argument parsing — agreed with the promotion to medium

The cross-review is right that Fable understated this, and right about why: a tool whose entire
job is emitting carefully controlled scientific fixtures must never quietly emit a different
configuration than the one requested. The documented interface was `key=value`; the code read
bare positionals and swallowed failures with `.parse().ok().unwrap_or(default)`. Anyone following
the usage string got `matrix=128` → 64, `oversample=8` → 4, `slices=3` → 1, and an `only=` filter
compared against the literal string `"only=pf68_diff_nowin"`, matching nothing.

Rewritten as a real `key=value` parser. I did not reach for Clap: the binary has four arguments
and no dependency on it today, and the failure mode here was permissiveness, not hand-rolling.
What changed is that **every malformed input is now fatal**:

```
$ trxscan-benchmark out matrix=oops
trxscan-benchmark: matrix must be a positive integer, got "oops"
$ trxscan-benchmark out 128
trxscan-benchmark: expected key=value, got "128"
$ trxscan-benchmark out bogus=1
trxscan-benchmark: unknown argument "bogus"
$ trxscan-benchmark out only=nope
trxscan-benchmark: only="nope" matched none of the 24 fixture labels: full_nophase_nowin, ...
```

All four exit 2. Zero is rejected along with garbage — `matrix=0` is not a fixture set. Two unit
tests pin that the documented syntax is the accepted syntax, including that order does not matter
and that omitted keys keep their defaults, so the usage string and the parser cannot drift apart
again silently.

## 4. `_median` is not a median — confirmed, fixed, and re-measured

Confirmed, and the cross-review is right that this is currently active rather than theoretical:
the factor grid has three PF levels and four phase conditions, so rule (a)'s control population
has 4 members and rule (b)'s has 12. Both even. `v[len(v) // 2]` was returning the upper-middle
sample on every run this branch has ever done.

Now `statistics.median`. Reran the suite and measured the difference rather than assuming it:

| gate | population | old `v[n//2]` | true median | floor shift | rows exempted |
|---|---|---|---|---|---|
| rule (a) `oscillatory_residual` | n = 4 | 0.0193325 | 0.0192638 | 0.0048331 → 0.0048160 (−0.36%) | 24 → 24 |
| rule (b) `artifact_norm_pe` | n = 12 | 0.0297406 | 0.0297406 | unchanged | 0 → 0 |

Rule (b) is unchanged because its two middle values are identical. Rule (a)'s floor moves by
0.36%, exempting the same set of rows. The full acceptance table is byte-identical to the round-5
one and the verdict is unchanged:

```
ACCEPTANCE: PASS (clean unapodized rows; PF asserted on the PE axis; RPG absent)
```

The cross-review guessed it wouldn't change the PASS. It didn't — but that is now a measurement
rather than a guess, which is the point of rerunning it.

---

## Low-severity items

### `--fmap` required but unused on the default path — fixed

Correct, and slightly worse than described: `--fmap` was not merely loaded, it was *grid-validated*
against the acquisition grid. So a default run could abort on a mis-gridded fieldmap whose values
could never reach its output.

`fmap` is now `Option<PathBuf>` and required only when `--oversample 1`. The two branches
collapsed into one load against whichever grid the signal stage actually runs on — `--sim-fmap`
when oversampling, `--fmap` on the legacy path — which removed a duplicated load and a duplicated
grid check as a side effect. The legacy path gets its own error message rather than inheriting the
oversampling one, which would otherwise have told a legacy user to go generate a simulation grid:

```
$ trxscan --oversample 1 ... (no --fmap)
Error: "--oversample 1 (the legacy path) requires --fmap, the acquisition-grid fieldmap.
        The default oversampled path takes --sim-fmap instead and ignores --fmap."
```

### `Acquisition::default_for(_nx, _ny)` — removed

Removed rather than implemented; there are no matrix-dependent defaults pending. Its only caller
was the test helper `clean(nx, ny)`, which — I noticed while removing it — had acquired the
identical defect: it took a matrix and ignored it. That lost its arguments too, across all fifteen
call sites.

### Acceptance window-string inconsistency — fixed

Real, and the cross-review's "little current impact" is accurate, but the inconsistency was the
kind that gets expensive later: rule (b) spelled the test `r.get("window") not in (None, "None")`
while rules (a) and (c) spelled it `str(r.get("window")) not in ("None", "none")`. The two forms
disagree on a lowercase `"none"`, which would have put such a row *inside* rule (b) and *outside*
(a) and (c) simultaneously. There is now one `_unapodized(row)` predicate, used by all three rules
and by the out-of-domain report, with the discrepancy documented in its docstring.

### `prepare_acquisition_grid --voxel` units — fixed properly

I took the cross-review's preferred option (implement physical millimetres) rather than its
fallback (document and enforce a 1 mm input), because the enforcement version would have made the
script strictly less useful than its help text already promises.

`--voxel` is now resolved against the source affine's zooms **per axis**, so the index-space step
is `VOX / zoom_i`; matrix size, the affine scale and the half-voxel centring term all derive from
that step rather than from `VOX` directly. Anisotropic sources work; a sheared source affine is
rejected explicitly rather than silently mis-scaled, via a post-construction assertion that the
realised column norms equal the requested spacing.

New test drives 0.8 / 1.0 / 2.0 mm synthetic sources holding a fixed 32 mm ball in a fixed 96 mm
FOV, and asserts both that the header says 1.7 mm and that the affine means it, plus that the
physical FOV is invariant to the source spacing. Confirmed non-vacuous: restoring the old
`diag([VOX] * 3)` fails the 0.8 and 2.0 mm cases (the guard fires with *"target spacing (3.4, 3.4,
3.4) mm != requested 1.7 mm"*), and the 1.0 mm case still passes — which is exactly why no test
using the bundled anatomicals could ever have caught this.

### Coil-sensitivity half-cell offset — fixed rather than commented

Both reviewers called this negligible and non-blocking, and both are right: the sensitivities are
very smooth Gaussians, and the canonical Gibbs benchmark forces `n_coils = 1`, so its impact on
anything under review is exactly zero. I fixed it anyway, because the fix is smaller than the
comment explaining why it was left.

The root cause is worth stating precisely, since it is the same class of error as the half-cell
registration bug that drove several earlier rounds: `coil_sensitivity` took `(x: usize, nx: usize)`
and centred on `nx / 2`, and was called with **sim** indices against `(snx, sny)` in the forward
model and **acquired** indices against `(nx, ny)` in the Roemer combine. Every length in the
function scales with the matrix, so the forward model's field was that same field evaluated at
`x / o` — displaced by `xoff / o = (o-1)/(2o)` acquired voxels from the combine's. The combine was
dividing by sensitivities the signal was never multiplied by.

It now takes continuous coordinates in acquired-voxel units, and the forward model passes the same
half-cell-registered `(xc, yc)` it already computed for the eddy polynomial — which also removed a
duplicated coordinate transform. Two tests: one that the `o` sim cells of acquired voxel `v` have
centroid exactly `v` at every oversampling factor (the invariant), and one that their mean
sensitivity matches the combine's value there. The second reconstructs the *old* convention inline
and asserts it is measurably worse, so the test cannot pass vacuously: worst residual 7.9e-5
registered versus 1.15e-2 unregistered, a factor of 146.

Regenerated the full 24-set fixture grid after this change and diffed it against the pre-change
run: **byte-identical**, as expected from `n_coils = 1`.

### Line-ending churn — `.gitattributes` added

I agree with the cross-review that this was not a PR finding: Fable was describing its working
tree, and nothing in the branch's committed contents has line-ending churn. Adding the
`.gitattributes` anyway, since the churn was real and recurring, and it bit this branch twice
during earlier rounds (two commits swept 3,248 lines of CRLF rewrites and had to be reset).

One deliberate omission: a dozen review documents under `docs/superpowers/` were themselves
committed with CRLF blobs, and `git add --renormalize` would convert them. That would add roughly
3,400 lines of pure line-ending diff to a branch that reviewers have already flagged as large, for
prose nothing reads programmatically. They will normalise on their own the next time one is
edited. The rationale is in the `.gitattributes` comment so the next person doesn't wonder.

---

## Not fixed: the FORCE/DIPY moment oracle

The one finding I'm deliberately leaving. Both reviewers confirmed it and the cross-review said it
shouldn't block this PR; I agree, and want to be explicit about the scope rather than quiet about
it.

`matches_dipy_closed_forms_on_fixtures` in `src/microstructure.rs` needs
`tests/fixtures/force_moments.txt`, and neither that file nor the `tools/gen_force_fixtures.py`
its old panic message named has ever been tracked in this repository — the message pointed at a
script that was not there. What this branch changed is that the test used to **fail on every fresh
clone**, which is what blocked adding CI at all; it now skips explicitly and prints why.

So Fable's "permanently green and decorative" is a fair description of the current state, and I'm
not claiming otherwise. But restoring the oracle means reconstructing a DIPY-derived fixture set
for the microstructure module, which this branch does not touch and which shares no code with the
Gibbs work. That belongs in a microstructure change where someone can validate the fixtures
against DIPY properly, not bolted onto a k-space PR. It is now recorded in spec section 6 alongside
the other deferred items, with the explicit note that the test must not be counted as coverage
until then.

---

## Verification

```
cargo test                                   87 passed
cargo test --features cli --all-targets      94 passed
cargo test --features cli,par --all-targets  94 passed
python -m pytest scripts -q                  72 passed
clippy --features cli --all-targets          45 warnings, unchanged from before this commit
acceptance.py <24-set grid>                  PASS
```

Clippy is compared against the pre-commit count deliberately: it is advisory in CI because most
findings are pre-existing in modules this branch doesn't touch, so "no new warnings" is the honest
claim, not "clean".

## On the cross-review

Two things I'd flag back.

The cross-review's confidence note — that an independent reviewer converging on the same areas
(half-cell registration, acquired-band frequency convention, mask-consistent noise, phase
modelling) and reaching the same conclusion raises rather than lowers confidence in the physics —
is the right reading, and it's worth saying that neither review found anything that touches the
forward model or the benchmark's conclusions. Every fix in this round is CI configuration, script
hygiene, or CLI ergonomics.

The exception is the coil-sensitivity registration, which *is* forward-model code. It changed no
number under review, because the benchmark is single-coil. But it is the third instance on this
branch of the same underlying mistake — evaluating an acquired-grid quantity at sim-grid
coordinates without the half-cell offset — after the object registration itself and the eddy
polynomial. That pattern is worth naming: any quantity defined on the acquisition matrix and
sampled on the simulation grid needs `(x - sxs - xoff) / o`, and the codebase now routes all three
through one such computation per sim cell rather than deriving it separately each time.
