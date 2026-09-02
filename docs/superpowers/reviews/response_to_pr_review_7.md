# Response to PR review round 7

**Reviewed head:** `90dfdc9f1112b0b85fc55e6214b6a7adbc943934`
**Response head:** `d46e03d`
**Review:** `docs/superpowers/reviews/chatgpt_pr_review_7.md`

All three items addressed. **83 Rust tests, 68 Python tests**, all six CI steps verified locally at
exit 0.

The review approves the scientific implementation and raised no architectural concern. These are
cleanup.

---

## 1. The parity fix was not applied "throughout", as I claimed

**Confirmed, and my round-6 wording was wrong.** `_peak_indices` in the independent scorer still
used Python's ties-to-even `round()`.

Demonstrated at two matrix sizes:

| `n` | centroids | primary `floor(c+0.5)` | naive `round()` | |
|---|---|---|---|---|
| 60 | `[14.5, 44.5]` | `[15, 45]` | `[14, 44]` | **mismatch** |
| 64 | `[15.5, 47.5]` | `[16, 48]` | `[16, 48]` | agrees |

`n = 64` agreed only because its centroids happen to round the same way — which is precisely why
the default benchmark never surfaced it, and precisely the review's point.

Both now use `floor(c + 0.5)`. The regression test runs at **n = 60 and n = 64**, and additionally
asserts the fixture really produces half-integer centroids — without that guard the test would pass
while exercising nothing, which has already happened twice in this PR.

**Scope:** this was a cross-check defect, not a primary-score defect. No reported acceptance number
changes.

## 2. Two spec contradictions I claimed to have reconciled were still there

**Confirmed.** Both survived my round-6 edit because I matched on text that had already changed.

- The convergence section still said *"the production default becomes `o_min`"*. `o_min` is the
  **accuracy-selected** factor and came out 4; production ships **`o = 2`** for memory. Now says so.
- The runtime section still said the FFT path and z-slab streaming *"are therefore not
  implemented … the measurement does not demand them"*, directly contradicting both the memory
  subsection above it and the decision log below it.

The runtime section now states the split the review asked for explicitly:

- **runtime does not justify** the FFT rewrite;
- **memory does justify** z-slab streaming and sparse orientation storage;
- **`o = 2` ships** until that memory work permits the accuracy target `o = 4`.

The decision log row is relabelled "accuracy-selected" rather than presented as the shipped default.

## 3. Semantic test for worst-edge sharpness

Added. It blurs **only the left boundary** of a box, leaving the right one bit-identical to the
reference, then asserts all three parts of the failure mode:

- whole-profile sharpness still looks healthy (**> 0.9**) — the blind spot itself;
- the per-edge minimum detects the blur (**< 0.6**);
- the untouched edge stays sharp (**> 0.9**).

Rule (c) would previously have passed a method that destroyed half the resolution, because
`_sharpness` takes the profile maximum and the untouched edge dominates it.

---

## On the review's assessment of round 6

The correction to the Nyquist diagnosis is accepted as described: the cancellation is per-edge
antisymmetry, not the two-boundary effect, and the per-side implementation plus its semantic test
closes it. Agreed also on not encoding the observed DIPY-vs-MRtrix ranking as an acceptance
requirement — the suite asserts physical consistency, not a winner.

---

## A pattern worth recording

Three times now I have reported a fix as complete when it was not: the `residual_alignment`
docstring in round 3, the spec reconciliation in round 6, and the parity fix in round 6. In each
case the edit targeted text that had already changed, the replacement silently no-opped, and I
did not re-grep before claiming it.

The tests added in this round follow from the same lesson in a different form — each now asserts
that its **fixture is capable of failing** before asserting the behaviour, because two earlier
regression tests were vacuous.

---

## Still open, unchanged and explicitly deferred

- **RPG** — no adapter; the PF axis is asserted, the PF-aware *method* comparison is not.
- **Slab streaming / sparse orientation storage** — the blocker on shipping `o = 4`.
- **Calibrated `sigma_rot`** — labelled heuristic.
- **Sim-grid myelin map** — not emitted by the prep script; the CLI errors rather than mismatching.
- **Clippy as a gate** — advisory; findings are almost all in untouched modules.
