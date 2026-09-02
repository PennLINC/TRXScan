I reviewed the attached round-6 response and the current PR head, `90dfdc9f1112b0b85fc55e6214b6a7adbc943934`. The PR is still draft/open and mergeable.  The response says all six findings were addressed and reports a fresh external-method acceptance run with MRtrix 3.0.4 and DIPY 1.12.1.

## Verdict: I would now approve the scientific implementation

I do **not** see another merge-blocking problem in the forward model or in the primary benchmark scorer. The major problems uncovered across the previous reviews—production-path wiring, PF sampling, phase/noise realism, complex scoring, dark-side masking, edge duplication, Nyquist cancellation, and asymmetric blurring—are now addressed coherently.

The exact current head also has a successful GitHub Actions run, including Python tests, Rust tests, production CLI compilation, the parallel path, and Clippy.

I found **two small issues I would still fix**, plus one worthwhile test addition. I would treat these as minor cleanup rather than another reason to reopen the architecture.

### 1. The independent scorer still has the rounding-parity bug

This is the one actual code defect I found.

The primary scorer now correctly converts a fractional physical-edge centroid using:

`floor(c + 0.5)`

in `sidelobe_runs()`, `_sidelobe_indices()`, `_sharpness_per_edge()`, and `_edge_shifts()`.

But the supposedly independent scorer still does:

`int(round(...))`

inside `_peak_indices()`.

So the response's statement that the parity fix was applied “throughout” is not quite true.

It does **not affect the default 64×64 benchmark**, because the two particular half-integer centroids happen to round in the desired direction. But it can affect another valid matrix size. For example, a centroid of `14.5` becomes `14` under Python's ties-to-even `round()`, while the primary scorer maps it to `15`.

I would change the naive scorer to the same explicit half-up convention and add a parity regression using, for example, both `n=60` and `n=64`. That is a one-line implementation fix plus a small test.

Importantly, this is a defect in the **cross-check**, not in the primary scores or the acceptance results reported for the default benchmark.

### 2. The specification still contains contradictions the response says were removed

The response says the spec now cleanly distinguishes the memory-driven shipped `o=2` from the accuracy-driven `o=4` target and states that memory warrants z-slab streaming.  Much of the updated risk section does now say exactly that.

But two stale statements remain.

The convergence-test section still says:

> “The production default becomes `o_min`”

even though production actually ships `o=2` because the accuracy-selected `o=4` is not affordable.

More conspicuously, the runtime section says:

> “The phase-10 FFT path and z-slab streaming are therefore not implemented … the measurement does not [demand them].”

while the memory section immediately above says z-slab streaming is **warranted**, and the decision log now says:

> “z-slab streaming — Warranted, not deferred.”

I would make the intended distinction explicit:

* runtime does **not** justify an FFT rewrite;
* memory **does** justify streaming/sparse storage;
* shipped default remains `o=2` until that memory work permits the accuracy target `o=4`.

Because this document is serving as the historical/authoritative design record, it is worth making internally consistent before merging.

### 3. Add one semantic test for the new worst-edge sharpness rule

The implementation is good: `_sharpness_per_edge()` now computes ratios separately around each physical boundary, `score()` exposes the per-edge values and worst edge, and acceptance rule (c) uses the worst edge rather than the whole-profile maximum.

I do not see a targeted test for the exact failure mode that motivated it: **blur one edge only, leave the other perfectly sharp, and verify that the old whole-profile sharpness looks good while `edge_sharpness_*_worst` detects the blur.**

Given how many scorer defects in this PR passed plausible-looking tests, I would add that test. I would not block merge on it if the two items above are fixed.

## The round-6 Nyquist fix looks correct

The response's correction to my diagnosis is persuasive. The critical cancellation is not merely between the box's rising and falling boundaries; Gibbs is antisymmetric across the **two sides of each individual boundary**, so combining dark-side and bright-side signed projections before taking magnitude can cancel even on a single edge.

The implementation now does the right thing: `sidelobe_runs()` creates one run per side per physical edge, `_nyquist_amplitude()` projects each independently, takes the complex magnitude, and then aggregates the squared magnitudes.  The independent implementation was changed to the same per-side mathematical definition.

More importantly, there is now a **semantic regression test** that explicitly verifies:

* two physical edges;
* four edge-side runs;
* nonzero artifact on every run;
* the old union projection cancels;
* the corrected score remains nonzero.

That closes the blocker from my previous review.

## The half-voxel fixture is now real rather than vacuous

The Python `_box()` helper now actually mirrors the benchmark geometry: `(q + 0.5)` boundaries and a block-mean nominal reference. It explicitly produces the partial-volume transition needed to exercise the gradient plateau, and another test asserts that a ~0.5 voxel really exists in the reference.

So my previous concern that the regression test could pass without reproducing the bug is closed.

## The acceptance rerun is now meaningful

The fresh results are substantially more convincing than the earlier `PASS`, because the full-Fourier no-op control no longer collapses to approximately zero:

| Method    |    PF | Nyquist | PE alignment | PE energy | Worst PE sharpness |
| --------- | ----: | ------: | -----------: | --------: | -----------------: |
| none      | 1.000 | 0.01945 |        1.000 |     1.000 |              1.221 |
| DIPY      | 1.000 | 0.01195 |        0.204 |     0.683 |              1.055 |
| mrdegibbs | 1.000 | 0.01216 |        0.379 |     0.494 |              1.079 |
| DIPY      | 0.750 | 0.01029 |        0.480 |     0.822 |              0.988 |
| mrdegibbs | 0.750 | 0.01025 |        0.607 |     0.720 |              0.904 |

Those are the values reported by the agent after rerunning with DIPY 1.12.1 and MRtrix 3.0.4.

The behavior is internally sensible: both methods reduce the full-Fourier oscillatory component; the PE artifact metrics worsen as PF becomes more aggressive; neither method achieves that reduction by violating the current sharpness floor. The alignment-versus-energy distinction also remains informative rather than redundant.

I agree with continuing **not** to encode the observed DIPY-vs-MRtrix ranking as an acceptance requirement.

## Production memory handling also looks resolved

The motion and no-motion caveats are now actually branched. The motion path warns that its arrays are densely written and should track the calculated bound, whereas the no-motion path explains why the dense histogram's committed memory can be lower because only touched pages are faulted.

That accurately reflects the two different allocation behaviors and fixes the misleading warning from the previous state.

## Final recommendation

I would **not issue another substantive `REQUEST_CHANGES` review** on this PR.

The primary implementation is now in a state I would approve. Before merging, I would make these small edits:

1. replace the remaining `round()` in the independent scorer with explicit half-up conversion and test an opposite-parity matrix;
2. reconcile the remaining `o_min` / `o=2` and z-slab-streaming contradictions in the spec;
3. preferably add the asymmetric one-edge-blur test for worst-edge sharpness.

After those, I do not think another full review cycle is warranted. The remaining explicitly deferred items—RPG integration, memory optimization needed for `o=4`, calibrated `sigma_rot`, and sim-grid myelin generation—are appropriately separable follow-up work rather than reasons to hold this Gibbs-realism PR.
