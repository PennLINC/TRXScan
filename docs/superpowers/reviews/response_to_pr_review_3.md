# Response to PR review round 3

**Reviewed head:** `e6be119f99d5d484cc6ecfa424612933492a8409`
**Response head:** `7a4ed26`
**Review:** `docs/superpowers/reviews/chatgpt_pr_review_3.md`

All three blockers and all remaining items are addressed. **83 Rust tests, 60 Python tests**; every
CI step verified locally at exit 0, including the three binary checks that did not previously exist.

---

## Blocker 1 — the memory analysis described a path that no longer runs

**Confirmed, and the review is right about the cause.** Fixing the round-2 blocker moved the default
no-motion path from `generate_compartments` to `generate_mixture`, which allocates a dense
`nvox * 321 * 8` f64 orientation histogram plus an f32 ODF copy (`compartments.rs:364`,
`HemiSphere::icosphere(3)` → `nvert = 321`, verified). The 13.04 GB figure measured the old path.

**Remeasured on the current path**, full 1 M-streamline tractogram, 107×151×104 acquisition grid,
75 volumes:

| `o` | sim voxels | histogram + ODF (bound) | compartment images | **measured peak RSS** |
|---|---|---|---|---|
| 1 | 1.68 M | 6.5 GB | 1.5 GB | — |
| **2 (default)** | 6.72 M | 25.9 GB | 6.0 GB | **11.46 GB** |
| 4 | 26.89 M | 103.6 GB | 24.2 GB | not attempted |

**The review's ~26 GB arithmetic is the correct upper bound; measured is 11.46 GB.** The gap is lazy
zero-page commit — the histogram is allocated densely but faulted in only where streamlines deposit,
781,325 of 6.72 M voxels here. The bound is what a fully-covered volume would cost; 11.46 GB is what
this tractogram actually costs.

So `o = 2` fits a 16 GB machine, with little headroom. `o = 4` was not attempted: a 104 GB bound is
beyond ordinary hardware even sparsely covered. **The default stays `o = 2`.**

**The review's sharpest sub-point was the pre-flight.** It estimated only the compartment images —
6.0 GB for a run that measured 11.5 GB — so it would not have fired at all. It now includes the
histogram term and reports both components.

Spec §6 records the remeasurement and upgrades sparse/masked orientation storage to a warranted
mitigation beside z-slab streaming.

## Blocker 2 — the scanner-like PF mode was never used

**Confirmed.** `trxscan-benchmark` inherited `Acquisition::default()`, so fixtures named `pf68` and
`pf78` were Fiberfox masks at 78.1% and 90.6%. The mode existed and nothing selected it.

Fixed: the benchmark sets `PartialFourierMode::Contiguous` and records `pf_mode` in `factors.json`;
the CLI gains `--pf-mode`, defaulting to `contiguous` on the reasoning the review gives — this
binary describes itself as HBCD-like and a scanner produces contiguous PF. `fiberfox` stays
selectable. Fixtures regenerated.

## Blocker 3 — green CI never compiled the production binary

**Confirmed, and it is the most uncomfortable of the three.** `default = []`, every bin has
`required-features = ["cli"]`, so `cargo build` and `cargo test` compiled **no binary at all** —
verified: the default build produces no `target/debug/trxscan`. Both major regressions in this
review history were in exactly that wiring.

CI now runs `cargo check --features cli` for `trxscan` and `trxscan-benchmark`, plus
`--features cli,par` for `trxscan`, since the parallel signal stage differs materially.

---

## Remaining items

| # | Item | Disposition |
|---|---|---|
| 4 | Rule (a) asserted a PF-invalid metric on PF rows | Scoped to full Fourier. |
| 4 | Rule (c) used one auto-selected axis | Now checks **both**; `edge_sharpness` and `edge_location_bias` are per axis. This matters because zero-filled PF itself trades ringing for PE-axis blur — a PE-only blurrer was previously invisible. A test covers it. |
| 5 | Noisy rows unsuitable as an amplification gate | Gibbs-removal rules run on **clean rows only**. The arbitrary `1.5×` cushion is gone. |
| 6 | New PF metrics not independently cross-checked | `naive_energy_ratio` and `naive_artifact_norm` added, compared on the 2D box and on a contiguous-PF box, both axes. |
| 7 | Only one of two box edges measured | All four metrics now score **every** significant edge. A one-sided PF transfer function is asymmetric, so the two PE edges need not match and `argmax` alone measured half the artifact. |
| Judgment call | Rules (a) and (c) could still fail a Hann row | **The review was right and the test was lying.** `test_apodized_rows_are_out_of_domain_and_never_fail` promised what only rule (b) delivered — and rule (a) was in fact failing Hann rows once scoped. All three method-performance rules now skip apodized input. |
| Cleanup | `residual_alignment` docstring | Corrected: 0.0 means orthogonal to the template, not gone. |
| Cleanup | `available_methods()` could report `rpg` | Now never does — there is no adapter, so it could only ever raise. |
| Cleanup | sim-grid myelin map | Still not produced by the prep script; the CLI errors rather than silently mismatching. Recorded, unfixed. |

---

## What the cross-check caught, immediately

Extending the naive scorer (item 6) paid for itself within minutes. After making all four metrics
score both edges, the two implementations diverged on the PF box — `0.00892` vs `0.00753`. Cause:
**the both-edges change had not been applied to `_nyquist_amplitude`**, so the primary was still
scoring a single `argmax` edge while the naive scored two. Nothing else would have surfaced that;
the acceptance suite would have kept reporting a plausible number.

That is the second time an independent reimplementation has caught a defect in the primary scorer,
which is the argument for keeping it.

---

## Process note

I twice ran `git add src/` in this session and swept the pre-existing CRLF-only files into a commit,
both times catching it in the diffstat and resetting. The 17 line-ending-only files remain unstaged.
Worth stating plainly since a reviewer looking at commit history should know the discipline slipped,
even though no churn reached `main`.

---

## Still open

- **RPG.** No adapter; `available_methods()` now correctly never reports it. The PF *axis* is
  measured, the PF-*aware method* comparison is not, and the PR does not claim otherwise.
- **Slab streaming / sparse orientation storage.** The real fix for blocker 1. `o = 2` at 11.46 GB
  is the interim answer, not a comfortable one.
- **Clippy as a gate.** Still advisory; findings are almost all in modules this branch does not
  touch.
