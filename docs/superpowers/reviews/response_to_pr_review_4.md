# Response to PR review round 4

**Reviewed head:** `222dfb91100af390c5260ebb884b521345ad4f1d`
**Response head:** `70685e0`
**Review:** `docs/superpowers/reviews/chatgpt_pr_review_4.md`

All three items and every cleanup are addressed. **83 Rust tests, 60 Python tests.** All six CI
steps verified locally at exit 0, including the binary and `par` checks added last round. 17
pre-existing CRLF-only files remain unstaged.

---

## 1. The pre-flight ran after the allocation it protects

**Confirmed.** `generate_mixture` and `signal_from_mixture` executed first; the estimate printed
immediately before Stage B. An allocation failure therefore occurred **before the warning could ever
appear**, which made it useless for the one job it had.

Moved to before Stage A. It now distinguishes the two paths, whose dominant allocations differ, as
the review asked:

| path | dominant allocation |
|---|---|
| no motion (default) | dense `nvox × 321` f64 orientation histogram + f32 ODF copy |
| motion | `nvox × ngrad` f64 accumulator (×2 live under `--features par`) |

**The stale CLI help was the more dangerous half.** It still cited the 13.04 GB `o=4` figure and
recommended `o=4` with ≥32 GB — both from the signal path that no longer runs. The current `o=4`
bound is ~104 GB, so that advice would have sent someone straight into an OOM. Removed.

I also took the review's point on wording. `o = 2` measured 11.46 GB against a 25.9 GB arithmetic
bound, and the gap is sparse page commitment for **one** tractogram and mask. The help and spec now
describe 16 GB as **tight and data-dependent**, not safe.

## 2. The complex metrics discarded the entire dark-side ringing pattern

**This was the most valuable finding of the round, and the reasoning is exactly right.**

All four metrics masked to `ref > 0.5 * max(ref)`. The justification in `_nyquist_amplitude` was
that negative ringing is rectified because "these are MAGNITUDE images". That stopped being true the
moment `score()` began reconstructing `mag * exp(i*phase)` — and I never revisited the mask.

Measured on a real fixture, across one edge of the benchmark box:

| | dark-side sign pattern |
|---|---|
| magnitude residual | `++++++++` (rectified, no alternation) |
| **complex residual** | `+-+-+-+-`, amplitudes 0.091 / 0.050 / 0.034 |

A full Gibbs sidelobe train, discarded. The review's hypothetical is the correct reading: **a method
that perfectly corrected the bright side of every edge and left the dark side untouched would have
scored zero alignment and zero residual energy** — perfect — in a benchmark whose entire purpose is
complex-aware unringing.

Fixed as the review proposed, keeping the two concepts separate:

- **complex Gibbs metrics** (`_nyquist_amplitude`, `residual_alignment`, `residual_energy_ratio`,
  `artifact_norm`) now score **both sides**, excluding only the transition guard band;
- **phase RMSE** keeps its magnitude threshold, which remains correct — phase genuinely is
  meaningless where magnitude is zero;
- bright-only is retained as a secondary `*_bright` variant, for comparison against magnitude-only
  methods that cannot act on the dark side at all.

**A subtlety that fell out of the fix.** With `bright = 0`, a `> bright * rmax` test still drops
exact zeros — which *are* the dark side. Row inclusion now depends on whether a row **crosses an
edge**, not on brightness. Both scorers were updated together, and the cross-check confirms they
still agree on all four metrics, both axes, on the contiguous-PF box.

## 3. Clean-only was claimed but implemented only in rule (b)

**Confirmed.** Rules (a) and (c) still evaluated noisy rows, contradicting the round-3 response's
stated disposition. All three Gibbs-removal rules now skip them; the paired noisy rows remain for
noise-amplification and robustness reporting.

**The gating populations were also wrong**, as the review notes. The `(a)` floor pooled controls from
PF, noisy and Hann conditions although the rule applies only to clean full-Fourier unapodized rows;
`pe_floor` pooled noisy and Hann controls that rule (b) then excluded. Each floor is now computed
from the population its own rule sees.

---

## Cleanups

| Item | Disposition |
|---|---|
| Spec internally contradictory: §3.1 said `o=4` default, §6 said `o=2`; §6's runtime table said streaming was not warranted immediately after the memory text said it was | Reconciled. `o = 2` is the shipped default set by **memory**; `o = 4` is the **accuracy target**, memory-limited. Runtime and memory reach *opposite* conclusions about phase 10 — the FFT path is a convenience, z-slab streaming and sparse orientation storage are warranted — and both now say so explicitly rather than one silently overwriting the other. |
| `residual_alignment` docstring still claimed 0.0 means "entirely gone" | Corrected: 0.0 means orthogonal to the template, which is why `residual_energy_ratio` exists. (I had reported this fixed in round 3; it was not — the edit targeted text that had already changed.) |
| Acceptance banner comment still said the PF axis was "descriptive only" | Corrected: PF now carries an asserted PE-axis rule. What the banner still excludes is stated instead — apodized input, noisy rows, and the absence of any PF-aware method. |
| `--noise` help described k-space variance | Now states the implemented convention: per-component variance of the reconstructed complex image at full sampling, single coil, pre-combination. |
| `_edge_shift` used only the strongest edge | Aggregates all edges and exposes them per edge. Under contiguous 6/8 the four PE edges measure `[+0.25, +0.49, -0.49, -0.25]` — the asymmetry a one-sided transfer function produces, invisible to a single-edge measurement. |

---

## A near-miss worth reporting

While rewriting `_edge_shift` I sliced from its definition to `def score(`, which silently deleted
`residual_alignment`, `residual_energy_ratio` and `artifact_norm` along the way. An `ImportError`
caught it; I restored from git, re-applied the both-sides fix deliberately, and redid the edge change
against the *next function boundary* rather than a distant landmark.

Worth stating because the file had just been given a subtle, high-value correction, and a
coarser-grained edit came close to destroying three functions while the tests it would have broken
were the ones I was about to rely on.

---

## Still open, unchanged from round 3

- **RPG.** No adapter; `available_methods()` correctly never reports it. The PF *axis* is asserted;
  the PF-*aware method* comparison is not, and the PR does not claim otherwise.
- **Slab streaming / sparse orientation storage.** The real fix for item 1 and the blocker on
  raising the default above `o = 2`.
- **Calibrated `sigma_rot`.** Labelled heuristic; separating it from translation needs a second
  estimator.
- **Sim-grid myelin map.** Prep script does not emit one; the CLI errors rather than silently
  mismatching.
- **Clippy as a gate.** Advisory; findings are almost all in modules this branch does not touch.
