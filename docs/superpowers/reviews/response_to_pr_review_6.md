# Response to PR review round 6

**Reviewed head:** `a62964de1eaa1865af28e5cc7fda4a5b568afdba`
**Response head:** `373c127`
**Review:** `docs/superpowers/reviews/chatgpt_pr_review_6.md`

All six items addressed. **83 Rust tests, 65 Python tests**, all six CI steps verified locally at
exit 0.

---

## 1. Nyquist cancellation — confirmed, and the mechanism is worse than diagnosed

**The review is right that the metric was cancelling, and right about the fix. The cause is
different and more severe.**

The review attributed the cancellation to the box's two opposing edges. Measured, it is between the
**two sides of a single edge**:

| region, one edge, guard ±2 | signed projection |
|---|---|
| dark side (i = 6..13) | **+0.01936** |
| bright side (i = 19..26) | **−0.01936** |
| both summed | **−0.00000** |

Gibbs is **antisymmetric about an edge** — it overshoots one side and undershoots the other — so
projecting both onto one global `(-1)^x` and summing annihilates it. This would cancel on a
**single-edge phantom**, so it is not a property of the box's symmetry at all.

**This was self-inflicted in round 4.** Including both sides was the correct fix for the dark-side
masking defect; summing them into one signed projection was not. `oscillatory_residual` has been
~0 for the benchmark box ever since, which means acceptance rule (a) has been comparing noise.

Fixed as the review proposed: project **each side of each physical edge separately**, take the
magnitude, then aggregate as RMS. `sidelobe_runs()` returns one run per side per edge — four for a
box axis. The uncorrected box now scores **0.01920** where it scored ~0. The same change was made
independently in `naive_oscillatory()`, since both implementations shared the defect.

The semantic test the review asked for asserts all three properties: two physical edges per axis,
four sidelobe runs, every run carrying nonzero artifact, and per-side aggregation surviving where
the union projection cancels.

**The review is also right that the previous `ACCEPTANCE: PASS` was not conclusive.** Rerun below.

## 2. The edge-clustering regression test was vacuous — confirmed

The `_box()` helper placed the box on voxel boundaries and point-sampled with `obj[::o, ::o]`. A
voxel-aligned edge has a single nonzero gradient sample, so the old local-maximum detector would
have found two edges anyway. **The test could not fail.**

The fixture now matches `trxscan-benchmark`: edges at `(q + 0.5) * o`, reference by **block mean**,
reproducing the `0, 0.5, 1` partial-volume transition that caused the plateau in the first place. A
guard test asserts the reference actually contains a `0.5` edge voxel, so the fixture cannot
silently regress to a form that proves nothing.

## 3. Rounding parity — confirmed, with a concrete demonstration

`int(round(c))` is round-half-to-even. Demonstrated on real centroids:

```
centroids [2.5, 7.5]  ->  round():     [2, 8]   same fractional part, opposite direction
                      ->  floor(c+.5): [3, 8]   deterministic half-up
```

Since `trxscan-benchmark` accepts arbitrary matrix sizes, which way a centroid lands was
parity-dependent. Now `floor(c + 0.5)` throughout.

## 4. Edge sharpness one-edge blind spot — confirmed

`_sharpness` takes the profile **maximum** gradient, so blurring one boundary of a box hid behind
the other — the same structural pattern as the Nyquist issue. Added `_sharpness_per_edge`; the
score exposes `edge_sharpness_{ro,pe}_per_edge` and `_worst`, and **acceptance rule (c) now uses the
worst edge**, so a method cannot buy ringing reduction by blurring any single boundary.

## 5. Motion warning text — confirmed

The arithmetic was branched but the caveat was not: it explained histogram page sparsity even under
`--motion`, which has no histogram. Now branched too — the motion path's arrays are densely written
and do **not** benefit from sparse commitment, so its committed memory tracks the bound closely.
Worker count now prefers `RAYON_NUM_THREADS` over `available_parallelism`.

## 6. Spec contradictions — confirmed

- No longer says Stage A "remains unmeasured end-to-end" immediately after recording the 11.46 GB
  measurement.
- No longer says z-slab streaming is unwarranted immediately after saying memory warrants it.
  **Runtime and memory reach opposite conclusions** — the FFT path is a convenience, streaming is
  required — and both are now stated as such rather than one silently overwriting the other.
- Decision log distinguishes the **memory-driven shipped `o = 2`** from the **accuracy-driven
  `o = 4` target**, and records sparse/masked orientation storage as warranted.
- Acceptance docstring typo fixed.

---

## Acceptance rerun with the corrected metric

Clean, unapodized. **mrdegibbs 3.0.4, DIPY 1.12.1** — neither is installed in CI, so this table is
not independently reproduced by the workflow and is recorded here for that reason.

| method | pf | oscill (per-side Nyquist) | align_pe | energy_pe | sharp_pe worst |
|---|---|---|---|---|---|
| none | 1.000 | 0.01945 | 1.000 | 1.000 | 1.221 |
| none | 0.875 | 0.01345 | 1.000 | 1.000 | 1.181 |
| none | 0.750 | 0.01518 | 1.000 | 1.000 | 1.041 |
| dipy | 1.000 | 0.01195 | 0.204 | 0.683 | 1.055 |
| dipy | 0.875 | 0.00918 | 0.244 | 0.767 | 1.020 |
| dipy | 0.750 | 0.01029 | 0.480 | 0.822 | 0.988 |
| mrdegibbs | 1.000 | 0.01216 | 0.379 | 0.494 | 1.079 |
| mrdegibbs | 0.875 | 0.00923 | 0.433 | 0.557 | 1.165 |
| mrdegibbs | 0.750 | 0.01025 | 0.607 | 0.720 | 0.904 |

`ACCEPTANCE: PASS`.

**Both methods now beat the control on `oscillatory_residual`** — 0.012 against 0.019 at full
Fourier — which was not measurable at all before the fix, since the control itself scored ~0. The
PE-axis alignment and energy figures are unchanged from round 5, as expected: that cancellation
never affected them.

---

## Still open, unchanged

- **RPG** — no adapter; the PF *axis* is asserted, the PF-*aware method* comparison is not.
- **Slab streaming / sparse orientation storage** — the blocker on raising the default above
  `o = 2`; for the motion path, writing volumes directly into the final arrays would remove the
  24-byte duplication.
- **Calibrated `sigma_rot`** — labelled heuristic.
- **Sim-grid myelin map** — not emitted by the prep script; the CLI errors rather than mismatching.
- **Clippy as a gate** — advisory.
