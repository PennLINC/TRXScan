# Response to PR review round 2

**Reviewed head:** `31b389db4cf7b8eb95613e0e33b0e3c6e7b434c4`
**Response head:** `b189ca7`
**Review:** `docs/superpowers/reviews/chatgpt_pr_review_2.md`

**All eight review items and both CI failures are addressed.** Every fix below was verified by
measurement, not by inspection — several of the review's points turned out to be understated once
measured, and three additional defects surfaced while fixing them.

**State:** 83 Rust tests, 56 Python tests, `cargo test` exit 0. Both CI jobs verified locally in the
exact environments they run in (bare `numpy/scipy/pytest` venv for Python; sibling crates cloned for
Rust). 17 pre-existing CRLF-only files remain unstaged, as throughout.

---

## CI failures

| Job | Cause | Fix | Verification |
|---|---|---|---|
| Rust: `Build` | Cargo resolves **optional** path dependencies too, so `../rust/trx-rs` and `../odx-rs` must exist even for the pure-std default build that uses neither | CI clones both (confirmed public) rather than stubbing, keeping the dependency graph honest | Reproduced locally by hiding `../odx-rs`; identical error |
| Python: `Test` | `test_oversample.py` imported `nibabel` at module level, before any in-test skip guard could run | `pytest.importorskip` | Bare `numpy/scipy/pytest` venv: 52 passed, 4 skipped, exit 0 |

**nibabel is deliberately still not installed in CI.** The scorer, its independent cross-check and
the calibration maths must stand on numpy/scipy alone, and the run now proves they do.

**A defect surfaced from that decision.** Testing in a bare venv exposed that
`available_methods()` reported `mrdegibbs` usable on `shutil.which` alone — but the adapter shells
out through a temporary NIfTI and therefore needs nibabel. In an environment with the binary but not
the library it advertised availability and then crashed rather than skipping. Availability now
requires both.

---

## Review items

### 1. Blocker — the oversampled path silently discarded features. **Accepted, fixed, verified.**

Confirmed exactly as described, and the review's framing is right that it was worse than
unsupported: the CLI accepted `--motion`, `--weights`, `--kappa`, `--myelin` and `--dropout-rate`,
printed messages implying they applied, and then discarded the `comp` they had shaped.

The signal grid is now selected **once**, before Stage A, and the existing branching runs on it —
so motion, mixture (weights + κ), myelin and dropout all reach the oversampled output through the
same code path as the nominal one. This also removes the wasted nominal Stage A the review noted
under item 7.

**Verified end-to-end**, not by inspection: `--kappa 8 --oversample 4` prints
`Mixture: 5373738 fallback WM voxels, Watson kappa 8`. 5.37 M is the **simulation** grid's
brain-voxel count (prep reported 5,422,698); the acquisition grid has 338,921. κ is demonstrably
acting on the finer grid.

### 2. Blocker — PF scored on the wrong axis. **Accepted, and worse than stated.**

Confirmed: the benchmark's symmetric box has gradient means equal to 12 significant figures, and
`dominant_axis()` kept axis 0 (readout) while PF acts on ky (axis 1).

**The review could not see how bad this was.** The independent scorer broke the same tie the
*opposite* way (axis 1). Two implementations were silently measuring different physics, and no test
caught it because every cross-check fixture was one-dimensional.

Fixes: explicit `READOUT_AXIS` / `PHASE_ENCODE_AXIS` convention; metrics reported per axis
(`_ro` / `_pe`); both scorers take an explicit `axis`; a 2D-box cross-check runs on **both** axes;
and a test asserts the tie exists so nobody "simplifies" the tie-break later.

### 3. `residual_alignment` is a projection, not "fraction surviving". **Accepted.**

Added `residual_energy_ratio` over the same sidelobe region, reported alongside. Docstring corrected:
zero means the residual is *orthogonal to the artifact template*, not gone.

### 4. The PF rule was gated by the PF-blind metric. **Accepted, and the proposed gate needed one more step.**

The review is right that gating on `oscillatory_residual` let the PF-aware rule be skipped exactly
where PF had moved the artifact off Nyquist.

The suggested replacement — the norm of R₀ in the PE sidelobe — is correct, and the reason it has to
be a **norm rather than a ratio** is sharper than it first appears: for the control, `Rm ≡ R0`, so
its energy *ratio* is identically 1 no matter how little artifact exists. A ratio-based gate can
never skip anything. `artifact_norm` is therefore an absolute quantity.

Also corrected: the rule's comment claimed "must remove some of the artifact" while the code only
failed on `a > 1`. It now states what it enforces — must not amplify — plus a companion check on
residual energy growth.

Acceptance tests now cover the PF rule, the gate, and the out-of-domain exemption; the review
correctly noted none existed.

### 5. Noisy rows conflated noise removal with Gibbs removal. **Accepted.**

The clean acquisition is now the artifact template for **both** rows, so a pure denoiser cannot
lower alignment without removing Gibbs. Summary rows are additionally grouped by window and noise as
well as pf — averaging them made the PF trend look cleaner than the data supports.

### 6. The PF mask is a Fiberfox quirk, not exact 6/8. **Accepted, measured, both modes now offered.**

Verified:

| `ny` | nominal | kept (fwd/rev) | actual |
|---|---|---|---|
| 32 | 0.750 | 25 / 25 | **78.12%** |
| 32 | 0.875 | 29 / 29 | 90.62% |
| 64 | 0.750 | 49 / 49 | 76.56% |
| 140 | 0.750 | 106 / 106 | 75.71% |

Added `PartialFourierMode { FiberfoxCompatible, Contiguous }`. `FiberfoxCompatible` remains the
default so shipped behaviour is unchanged; `Contiguous` keeps exactly `round(ny*pf)` consecutive
lines — what a scanner produces, and the more relevant condition now the object is genuinely complex
with no Hermitian symmetry to exploit. Tests pin the exact count in both polarities and that
`Contiguous` leaves no gaps.

### 7. `o=4` default vs. 13 GB. **Accepted — and the parallel case is worse than measured.**

The review's `par` concern checks out. The fiber accumulator is **f64**, so one buffer is
`nvox·ngrad·8` = **16.1 GB** at `o=4`, with a pair live during `reduce_with`; `compartments.rs`
already warns against `fold` for exactly this reason. **The measured 13.04 GB peak was without
`par`** — the parallel path needs upwards of 32 GB.

There is no setting that is both accurate and memory-safe on ordinary hardware until slab streaming
lands. **Default lowered to `o=2`.** `o=4` remains the accuracy target (~4.5% residual vs ~15.6%)
and the benchmark fixtures still use it, their matrices being small. Spec §6 records the f64
accumulator figure and names parallel-accumulation redesign as the companion fix to slab streaming.

### 8. CI red. **Fixed** — see above.

### Remaining caveat — `sigma_rot`. **Accepted.**

Now explicitly labelled heuristic. Separating translation from rotation needs two different
estimators (across-volume global phase for the constant term, within-volume phase gradients for the
spatial-linear term) and only the former has been done.

---

## Defects found while fixing, that the review did not raise

1. **`score()` compared a transposed estimate against an untransposed control.** It reused arrays
   already reoriented by `_to_last` while leaving `control_*` alone. The control's own alignment —
   **exactly 1.0 by construction** — came out 0.24 whenever the auto-axis resolved to 0. Caught by
   noticing an impossible value rather than a failing test. All 24 clean control rows are now
   exactly 1.0, and that invariant is worth checking precisely because it cannot be anything else.

2. **A test I wrote asserted something measurably false.** It claimed PF "shows up more on
   phase-encode". Measured: zero-filled PF trades ringing for blur along PE, the blur sits at the
   edge which the sidelobe guard excludes, so PF makes **readout** worse on this metric
   (`ro 0.0828` vs `pe 0.0757`). Rewritten around what is demonstrable — a symmetric box has exactly
   equal artifact on both axes under full Fourier, and PF destroys that equality.

3. **The `mrdegibbs` availability bug** described under CI above.

---

## One judgment call, flagged for the reviewer

`mrdegibbs` **amplifies the phase-encode artifact ~25%** on Hann-windowed input (mean PE alignment
1.251); dipy is near-neutral (0.981).

I scoped acceptance rule (b) to unapodized input and report the apodized behaviour as an explicit
out-of-domain observation rather than failing the suite. The reasoning: Kellner's sub-voxel shift
assumes an unapodized rectangular window, and MRtrix itself recommends disabling scanner filtering
for best `mrdegibbs` performance — so this is a real result about the method outside its documented
domain, not a simulator defect. Failing the whole suite because a method misbehaves where it is not
designed to work would make the gate less useful, not more.

**If you would rather that be a hard failure, it is a one-line change** (drop the `window` guard in
rule b). I did not want to make that call silently.

---

## Not addressed, and why

| Item | Status |
|---|---|
| **RPG** | Still not installed, so the PF-*aware method* comparison remains unavailable. The PF *axis* is now measured via `residual_alignment_pe`. The PR does not claim the method comparison is complete. |
| **Slab-streamed Stage A / parallel accumulation redesign** | The real fix for item 7. Recorded in spec §6 as warranted rather than deferred; `o=2` is the interim answer. |
| **Clippy as a gate** | Still advisory. 34 findings, almost all pre-existing in `microstructure`/`mixture`/`sphere`, which this branch does not touch. Making it a gate is a separate cleanup, not something to smuggle into this PR. |
| **`--oversample N --myelin`** | Still writes no `myelin.nii.gz` on the sim grid; `distance_transform_edt` needs anisotropic `sampling`. Recorded in the phases 0-7 plan. |
