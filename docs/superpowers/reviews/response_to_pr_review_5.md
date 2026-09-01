# Response to PR review round 5

**Reviewed head:** `9c2c452b3cab91dd62ed9cfb0f31e470ac71a5fc`
**Response head:** `60dfa47`
**Review:** `docs/superpowers/reviews/chatgpt_pr_review_5.md`

All five items are addressed. **83 Rust tests, 63 Python tests.** All six CI steps verified locally
at exit 0. 17 pre-existing CRLF-only files remain unstaged.

---

## 1. The motion pre-flight modelled a data structure that does not exist

**Confirmed.** `generate_compartments_moving` has **no** `nvox × ngrad` f64 accumulator. Reading the
actual lifecycle: it works per volume (four `nvox` f32 resampled tissue arrays, two `nvox` f64,
three `nvox` f32 outputs), **collects** the three f32 outputs for every volume into `vols`, then
allocates three *more* `nvox × ngrad` f32 arrays and copies into them. Both are live during the copy.

| | bytes per voxel-volume | at o=2, HBCD-sized |
|---|---|---|
| old estimate (f64 accumulator + images) | 20 | 10.1 GB |
| **actual (collected + final, both live)** | **24** | **12.1 GB** |

A 20% understatement of a floor it also mis-described, with no allowance for concurrent per-worker
arrays. Now modelled from the real lifecycle, with a per-worker term that `par` multiplies. The
spec inherited the same wrong statement and is corrected alongside.

The review's longer-term suggestion — write each volume directly into the final arrays and drop the
duplication — is recorded in the spec as the fix that would actually remove the 24-byte floor.

## 2. The edge detector split one physical edge into two

**Confirmed, and the review diagnosed it from my own numbers.** The `[+0.25, +0.49, -0.49, -0.25]`
I reported as "four PE edges" was the tell: a rectangle has **two** boundaries per axis.

The benchmark deliberately places edges at half-voxel positions, so a block-averaged profile reads
`0, 0.5, 1` and its gradient reads `0.5, 0.5` — two adjacent samples that **both** satisfy
`>= local max`. Verified directly: a box profile yields detected peaks `[2, 3, 8, 9]` where it
should yield two.

This was not confined to the bias metric. `_sidelobe_indices` uses the same definition, so **every
artifact mask** had widened and duplicated guard regions around the phantom double-peaks.

Fixed as proposed: contiguous above-threshold runs are collapsed to their gradient-weighted
centroid, in both the primary and the independent scorer. The same box profile now yields
`[2.5, 8.5]` — two edges at the correct half-voxel positions. A regression test asserts **exactly
two edges along RO and two along PE** on the actual benchmark box, in both implementations.

## 3. The scalar edge bias cancelled real geometric error

**Confirmed.** Signed mean of `[+0.25, +0.49, -0.49, -0.25]` is ~0, so `edge_location_bias`
reported "no bias" while every detected edge had moved substantially.

The review's distinction is the right one and both quantities are now reported:

- `edge_location_bias_{ro,pe}` — **RMS displacement**, non-cancelling, the geometric-fidelity
  number the error decomposition actually wants;
- `edge_location_shift_{ro,pe}_signed` — signed mean, i.e. global translation, which *can*
  legitimately cancel;
- `edge_location_bias_{ro,pe}_per_edge` — the signed per-edge values, retained.

## 4. The dark-side fix had no targeted semantic test

**The review's reasoning here is the most important point in this round**, and it is right in a way
that generalises: the independent cross-check **could not** have protected against the round-4
defect, because both implementations used the bright-only interpretation and **agreed with each
other while both were wrong**. Agreement proves consistency, never correctness.

Added an explicit fixture that pins the physical property instead. It builds a complex ringing
control, constructs a method result that removes **only the bright-side sidelobes** and leaves the
dark side untouched, then asserts:

- the bright-only secondary metric scores **< 0.05** — it looks perfect;
- the primary both-side metric scores **> 0.5** — it is not.

A second test pins the underlying reason: the dark-side magnitude residual is strictly positive
(rectified) while the complex residual alternates in sign.

## 5. Reran the real-method acceptance suite

Done, and it mattered — **the definition change moved the numbers materially.** Clean, unapodized,
phase-encode axis, `alignment / energy`:

| method | pf = 1.000 | pf = 0.875 | pf = 0.750 |
|---|---|---|---|
| none (control) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| **dipy** | **0.204 / 0.683** | 0.244 / 0.767 | 0.480 / 0.822 |
| **mrdegibbs** | **0.379 / 0.494** | 0.433 / 0.557 | 0.607 / 0.720 |

`ACCEPTANCE: PASS` (clean unapodized rows; PF asserted on the PE axis; RPG absent).

Three things worth noting:

- Both methods degrade **monotonically** as partial Fourier grows more aggressive, which is the
  physical expectation the withdrawn round-2 rule was reaching for and could not measure.
- **dipy scores far better than under bright-only scoring** (0.204 vs 0.605 at full Fourier). That
  is the point of item 2 in the last round: it was being penalised for dark-side ringing the old
  mask could not see.
- **mrdegibbs leaves less total residual energy but more of it aligned with the original artifact**
  than dipy. That is exactly the distinction the alignment/energy pair exists to expose, and neither
  metric alone would show it.

No ranking is asserted; these are recorded so the next reviewer can see the methods behave sensibly
under the new definitions.

---

## Documentation cleanup

- Spec no longer says `o = 2` "fits a 16 GB machine". It now matches the CLI: **tight and
  data-dependent**, because 11.46 GB is one tractogram and one mask against a 25.9 GB bound, and a
  denser tractogram moves it toward the bound.
- `acceptance.py` docstring typo fixed — it had dropped the metric names after "Partial Fourier is
  now covered by".

---

## Still open, unchanged

- **RPG** — no adapter; `available_methods()` never reports it. The PF *axis* is asserted; the
  PF-*aware method* comparison is not.
- **Slab streaming / sparse orientation storage** — the blocker on raising the default above
  `o = 2`. For the motion path specifically, writing volumes directly into the final arrays would
  remove the 24-byte duplication.
- **Calibrated `sigma_rot`** — labelled heuristic.
- **Sim-grid myelin map** — prep script does not emit one; the CLI errors rather than mismatching.
- **Clippy as a gate** — advisory; findings are almost all in untouched modules.
