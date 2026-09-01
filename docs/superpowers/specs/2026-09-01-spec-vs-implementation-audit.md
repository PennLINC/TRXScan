# Spec vs. implementation: audit for external review

**Date:** 2026-09-01
**Reviewer brief:** every place the implementation departs from the frozen spec, plus every defect
found by executing rather than reviewing. Written for a reviewer who has not seen the session.

**Documents**
- Spec: `docs/superpowers/specs/2026-08-31-gibbs-ringing-realism-design.md` (revision 6, frozen)
- Assessment: `docs/gibbs-ringing-assessment.md` (revision 2)
- Plan A (phases 0-7): `docs/superpowers/plans/2026-08-31-gibbs-ringing-realism.md` — 12/12 done
- Plan B (phases 8-10): `docs/superpowers/plans/2026-08-31-gibbs-benchmark-and-calibration.md` — 11/11 done

**State:** 76 Rust tests (81 with `--features io`), 32 Python tests, 2 integration tests. One
pre-existing failure inherited from `main` (`microstructure::tests::matches_dipy_closed_forms_on_fixtures`,
needs an absent `tools/gen_force_fixtures.py`). Branch `gibbs-ringing-realism-spec`, 34 commits,
unpushed.

**How to read this.** Sections 1-2 are places the shipped code deliberately differs from the frozen
spec — those need a reviewer's judgement. Section 3 is spec text that was *wrong* and had to be
corrected. Section 4 is defects the tests caught. Section 5 is what is not implemented. Section 6
is the empirical findings that justify the work.

---

## 1. Deliberate departures from the frozen spec

These are decisions taken during implementation that contradict spec text. Each has a rationale;
none were silent.

| # | Spec says | Implementation does | Why | Risk if wrong |
|---|---|---|---|---|
| 1.1 | Sequencing 8 → 9 → 10 (spec §5) | producer → 9 → 8 → 10 (Plan B) | Phase 9 needs the benchmark producer that phases 0-7 deferred; phase 9 is unblocked while phase 8 needs 274 GB of real data; phase 10 was deprioritized by measurement | Low. Ordering only. |
| 1.2 | ds006131 owns the diffusion-phase b-dependence (§4.2) | `PhaseModel::hbcd_like()` uses the **NIBS** fit | The two datasets disagree (see 6.4). The preset models the HBCD protocol and NIBS *is* that protocol (TE 88 ms, 6/8 PF, MB 3); ds006131 is a different sequence | **Medium.** A reviewer may prefer the wider b range over the protocol match. Both numbers are recorded. |
| 1.3 | `o = 2` used as the worked cost example (§3.1) | Production default `o = 4` | Convergence study: `o=2` profile error is 15.6% of the ~9% artifact being measured | Low. Measured, tabulated in §3.1. |
| 1.4 | Acceptance rule "aggressive PF must not make unringing easier" (§5.2) | **Rule withdrawn**, PF trend tabulated not asserted | The oscillatory metric is a Nyquist projection; PF ringing arises from two k-space intervals and lies elsewhere (the reason RPG exists). Both real methods violate every form of the rule *because of the metric's blind spot* | **High — needs review.** This removes the only rule covering a whole factor axis. See 5.1. |
| 1.5 | Scripts live in the truth-data bundle | Scripts moved to `TRXScan/scripts/` | The bundle is a sibling directory and not a git repo, so the spec's location was uncommittable. The bundle keeps its copy, which its own README references | Low. |
| 1.6 | Plan A Task 6 test consolidation | `slice1()` shim added in `kspace` tests rather than expanding 19 call sites | Behaviour-preserving; all pre-existing assertions untouched | Low. |

## 2. Interface and semantics decisions the spec left open

| # | Item | Decision | Note for reviewer |
|---|---|---|---|
| 2.1 | Complex noise variance convention | **Per-component**: `Var(Re) = Var(Im) = noise_variance`, so `E[\|n\|²] = 2·noise_variance` | Chosen to match existing code (`kspace.rs` draws `sigma` into `re` and `im` independently) so the CLI `noise` argument keeps its meaning. The alternative (`noise_variance = E[\|n\|²]`) would silently halve it. |
| 2.2 | Multi-coil noise semantics | Single-coil **pre-combination** image variance; combined variance emerges from the coil model as `V / Σ_c s_c²` | Verified against the Roemer path. |
| 2.3 | `c_q` vs `sigma_dx` split | `c_q = 0.0425`, `sigma_dx = 1.0` | Only the **product** is constrained by data. The split is a convention and is documented as such. |
| 2.4 | Scoring axis | Auto-detected as the axis of largest mean gradient | Simulator NIfTIs carry readout as axis 0, not −1. Explicit `axis=` override available. |
| 2.5 | Fixture phantom | 2D box (`box_hires`), not a 1D step | A readout-only step leaves the PF axis inert (see 4.11). |

---

## 3. Spec text that was WRONG and has been corrected

These were errors in the frozen spec, found by implementing it. All are fixed in the spec, but a
reviewer should confirm the corrections.

### 3.1 Half-cell registration in the core transform — **most serious**

Spec §3.1's implementation snippet aligned grid **indices**. An index denotes a cell's left edge,
and the two grids have different cell widths, so cell **centres** were misaligned by `(o−1)/(2o)`
acquired voxels — **0.44 voxels at o = 8**, which swamps the ringing being measured.

Caught by `crop_reproduces_the_analytic_profile_across_subvoxel_offsets` at **3.7e-1 against a
5e-3 tolerance** (75× breach). Correction: `xoff = (o−1)/2`, satisfying `u = (x+0.5)/o − 0.5 − xs`
exactly and identically zero at `o = 1`. Post-fix deviation 8.3e-4.

*A "does the image change?" test — which is what the codebase had — would not have caught this.*

### 3.2 The `truth-nominal` derivation (caught in review round 3, before implementation)

Revision 3 claimed the nominal reference was free by re-running Stage A at `o = 1`. False once a
spatial phase model exists: `(1/N)Σ f_j e^{iφ_j}` carries intravoxel dephasing that `f̄·e^{iφ_coarse}`
does not, and an independent run is a different random realization. Now block-reduced from the same
array. Renamed `object-hires` / `object-nominal` / `acquired`.

### 3.3 The cost argument for `o = 8`

Spec claimed `o = 8` "would in practice require the phase-10 FFT path first." Measured: `o = 4` is
714 ms/slice, 1.55 h for a 104-slice 75-volume acquisition single-threaded, **0.19 h on 8 cores**;
`o = 8` extrapolates to ~1.7 h on 8 cores. Not prohibitive. `o = 4` stands on **accuracy** grounds
alone. Phase 10 demoted to a convenience.

### 3.4 An unjustified tolerance drove the wrong default

Plan A wrote `eps = 1e-3` for the convergence criterion with no justification. Taken literally it
selects `o = 8` (~45× cost). Anchored to the ~9% artifact the benchmark exists to measure, `o = 4`
(4.5% residual, 12×) is the defensible answer. **The number was arbitrary and it took running the
study to notice.**

### 3.5 Assessment claims that did not survive

- "1.55–10.04% overshoot proves the amplitude is wrong" — **withdrawn**. A *correct* implementation
  spans +0.29% to +8.93% across sub-voxel edge positions. The load-bearing criticism is the
  **cutoff frequency**, not amplitude.
- "No real MRI is Gibbs-free" — softened. Scanner filtering and non-step edges both moderate it.

---

## 4. Defects found by executing, not reviewing

Five rounds of external review converged on prose. Every one of these was found by running code.

| # | Where | Defect | How it presented |
|---|---|---|---|
| 4.1 | Plan A T1 | `far_from_the_edge_the_profile_is_flat` sampled `p[1]`, `p[n-2]` — 1.5 voxels from the periodic step's **wrap-around** edge, on its ±1.19% sidelobe | Test contradicted its neighbour and failed on correct code |
| 4.2 | Plan A T1 | Module declaration deferred to step 3, so step 2 could not fail: an undeclared module isn't compiled and the name filter matched an unrelated test | **False green light** |
| 4.3 | Plan A T2 | Plan said redefine `at` for the sim grid, but `at` also indexes `nx·ny` buffers (ringing block, spikes, Roemer combine) | Would corrupt or panic in 3 places |
| 4.4 | Plan A T2 | Eddy centring on raw sim indices | `eddy_strength` would scale as `o` (linear) and `o²` (quadratic) — latent until oversampling met eddy currents |
| 4.5 | Plan A T2 | `coil_sensitivity` called with `(nx, ny)` inside the sim loop | Coil ring at 1/o of the FOV |
| 4.6 | Plan A T4 | Plan pointed at a path **outside the repo**, in a directory that is not a git repo; variable names in the snippet were invented | Task uncommittable as written |
| 4.7 | Plan A T4 | The obvious reconstruction of the snippet uses `w, g, c`, which are **re-loaded from the already-downsampled output** | Would silently produce the null upsample the spec warns against — a script that does nothing |
| 4.8 | Plan A T6 | "Essentially real" threshold 1e-2 vs spec §3.4's 0.05 bound for the *same quantity* | Two thresholds disagreeing |
| 4.9 | Plan A T9 | Noise covariance measured along **readout**, but PF undersamples **phase-encode** | **Vacuous** — passed with the defect reintroduced |
| 4.10 | Plan B T4 | Oscillatory metric high-passed with a rolling mean, which can't remove a sharp edge | Rated a Gaussian blur as **more** oscillatory than real ringing — inverted the benchmark's central conclusion |
| 4.11 | Plan B T4 | `edge_location_bias` centroid over the whole profile | Reported a **+2 voxel shift as −14** |
| 4.12 | Plan B T4 | Test fixture built ringing as `sin(π·x)` on an integer grid — **identically zero** | **Vacuous** — the "ringing" image was byte-identical to the reference |
| 4.13 | Plan B T5 | Test phantom used the *old* zero-filled mechanism | The tools cannot correct it (see 6.1) |
| 4.14 | Plan B T7 | `grad_rms` pooled per-axis gradients | A pure-x ramp of slope g read as `g/√2` |
| 4.15 | Plan B T7 | `corr_length` used \|autocorr\| of `exp(iφ)`, which is **identically 1 at every lag** for a ramp | Pinned at its cap; could not tell smooth from rough |
| 4.16 | Plan B T8 | SNR-bias test fixture too weak — thermal noise never dominated | Test asserted nothing |
| 4.17 | Plan B T9 | **Preset amplitude taken from a `p`-free fit and used in a model that forces `p = 0.5`** | **51% over-prediction** of phase (2.030 vs 1.348 rad measured at b=1000) |
| 4.18 | Plan B T10 | Radial binning normalized to the FFT **corner** | Axis-aligned Nyquist landed in bin 11 of 16 instead of the top |
| 4.19 | Suite | Every scorer helper worked on `axis=-1`; simulator NIfTIs carry readout on **axis 0** | Measured gradients where the object is constant — near-zero for everything, while appearing to run |
| 4.20 | Suite | Oscillatory window included the edge voxel; reference is box-integrated, estimate band-limited, and an unringer makes the transition *softer* | **Ranked both real methods worse than doing nothing**, despite both cutting ripple 76-88% |
| 4.21 | Suite | Metric blind to **magnitude rectification** — dark-side ringing goes negative and returns as positive magnitude | Alternation destroyed on half the profile |
| 4.22 | Suite | `check_consistency` matched controls on `pf`/`phase` but **not `window`** | Compared unapodized method rows against apodized controls |
| 4.23 | Suite | Fixture phantom was a readout-only step | **The entire PF axis of the 48-point grid was inert** (control residual flat at 0.0106/0.0107/0.0104) |

### 4.24 Two of my own claims were wrong and were corrected

- **"ds006131 has 0 fetched files."** `find -type f` tests the **symlink**, not its target, so it
  returns 0 for healthy git-annex links. Compounded by spot-checking `sub-20828`, one of the 48
  genuinely unfetched subjects. **2 of 50 are fetched** (`sub-20188`, `sub-24053`).
- **"`o = 8` needs the FFT path."** See 3.3.

---

## 5. Not implemented

| # | Item | Status | Blocking? |
|---|---|---|---|
| 5.1 | **RPG (PF-aware unringing)** | Not installed. `run_method("rpg")` raises `MethodUnavailable` and never substitutes | **Yes for the PF axis.** Combined with 1.4, the suite currently makes *no* validated claim about partial Fourier. Highest-value gap. |
| 5.2 | `Tukey` and `Fermi` windows | Implemented and unit-tested, but the factor grid exercises only `None` and `Hann` | No |
| 5.3 | Post-reconstruction phase transform (spec §3.2) | Not implemented. Only pre-readout object phase exists | No — spec offers it as optional |
| 5.4 | Complex coil sensitivities | Deferred by design (spec §6). `coil_sensitivity -> f64`, real Roemer combine | No, but §3.2 term 2 must not be described as coil phase until it lands |
| 5.5 | z-slab streaming and the `rustfft` path | Correctly not implemented — Plan B Task 11 is measurement-gated and the measurement says no | No |
| 5.6 | **Stage A 4× memory** | **Unmeasured end-to-end.** Task 11 measured Stage B only (1.1 MB/slice). Spec §6 estimates 1.4 GB → 5.5 GB for Stage A | **Possibly.** Needs a full pipeline run against sim-grid maps. |
| 5.7 | `--oversample N --myelin` | Writes no `myelin.nii.gz`. Not a plain resample: `distance_transform_edt` needs `sampling=(VOX/n, VOX/n, VOX)` on the anisotropic sim grid | No |
| 5.8 | `object_hires` consumption | Emitted but diagnostic only; scoring uses `object_nominal` | No — matches spec §3.5 |
| 5.9 | Motion under oversampling | Disabled for all benchmark work (spec §6). Fibers are safe (world-space re-rasterization); **tissue is not** (`resample_by_pose` interpolates) | No, but the rotation test in spec §6 has not been run |

---

## 6. Empirical findings

### 6.1 The old mechanism was uncorrectable — the headline result

Real tools, both mechanisms, three independent measurements agreeing:

| phantom | ripple period | DIPY | `mrdegibbs` | where `mrdegibbs` finds removed energy |
|---|---|---|---|---|
| Zero-filled truncation (**shipped**) | 3.88 vox | **+16% worse** | −0% | bin 7/16 |
| Crop + reconstruct (**built**) | 2.14 vox | **−87%** | **−92%** | bin 15/16 (Nyquist) |

The shipped mechanism was not merely inaccurate — it was **uncorrectable by the methods it would
have been used to benchmark**, and actively misled one. Pinned as
`test_the_old_zero_filled_mechanism_is_not_correctable`.

### 6.2 Sub-voxel dependence invalidates fixed-overshoot tests

A *correct* rectangular truncation at the nominal matrix, sampled at voxel centres:

| offset | 0.000 | 0.125 | 0.250 | 0.375 | **0.500** | 0.625 | 0.750 | 0.875 | 0.938 |
|---|---|---|---|---|---|---|---|---|---|
| overshoot | +1.19% | +3.88% | +6.38% | +8.22% | **+8.93%** | +8.09% | +5.36% | +1.02% | +0.29% |

A 30-fold spread. Sign alternation, by contrast, is **exact and offset-invariant** (`+-+-+-+-+-` at
both 0.0 and 0.5) — which is why it survived as the pinned invariant.

### 6.3 Wrapped phase saturates; circular statistics are mandatory

Linear SD of wrapped phase ceilings at `π/√3 = 1.814`. NIBS shells sat at **62 / 80 / 97 / 99%** of
it; ds006131 at **87-100%**. Linear fits: p = 0.257 (NIBS), **0.080** (ds006131). Circular SD
`√(−2 ln R)` is accurate to σ ≈ 3.0 with its own ceiling near `√(ln N) ≈ 3.4` — verified by
recovering known σ through wrapping. All shells in both datasets sit below 3.0, so both fits are in
the valid regime.

### 6.4 `p` is fixed by construction; only the amplitude is calibrated

`DiffusionPhase` forms `q_eff = c_q·√b·bvec`, so **p = 0.5 is hardcoded** — it follows from
`φ = q·Δx` with `b ∝ q²`. Fitting `p` is a *validation* of that choice, never a way to set it.

| source | estimator | fitted `p` |
|---|---|---|
| NIBS, shelled | within-volume spatial | 0.445 |
| NIBS, shelled | across-volume constant | ~0.66 |
| ds006131, per-volume | within-volume spatial | 0.270 |
| theory (bulk translation, fixed timing) | — | **0.5** |

**The estimator moves `p` more than the dataset does**, and should: the shot phase is a constant
plus a linear term; a within-volume spatial SD is blind to the constant, an across-volume SD sees
only it. ds006131 is additionally **non-shelled CS-DSI**, so direction and \|q\| are confounded and
no b has enough volumes for a stable across-volume estimate. Treat **0.27-0.66** as the honest
spread around a theoretically fixed 0.5.

### 6.5 Acceptance suite result

288 rows over 48 fixtures, **ACCEPTANCE: PASS**.

| method | oscillatory residual | edge sharpness |
|---|---|---|
| none (control) | 0.0104 | 1.00 |
| `mrdegibbs` | 0.0025 (−76%) | 0.94 |
| DIPY | 0.0015 (−86%) | 0.94 |

Apodization independently checks out: Hann cuts control ringing 0.0191 → 0.0020 (~90%). With the
2D phantom, phase error rises **0.008 → 0.018 → 0.038** as PF grows more aggressive, as zero-filled
PF should.

### 6.6 NIBS matches the shipping configuration

`EchoTime 0.088` vs `t_echo: 88.0`; `PartialFourier 0.75` vs `partial_fourier: 0.75`;
`MultibandAccelerationFactor 3`. Confirms the protocol-match reasoning in spec §4.2 rather than
assuming it. Phase is **uint16, `scl_slope` NaN, `Units: arbitrary`** — skipping the `π/4096`
rescale inflates every statistic ~1300×, so the converter refuses input already in radians.
Measured background gradient **0.164 rad/voxel**.

---

## 7. Questions for the reviewer

1. **1.4 / 5.1 — the PF axis.** Withdrawing the rule was the honest call given a Nyquist-only
   metric, but it leaves a whole factor axis unasserted. Is a PF-aware metric worth building before
   RPG is available, or should the axis stay reported-only until it is?
2. **1.2 — NIBS over ds006131 for the preset.** Protocol match beat wider b range. Defensible?
3. **6.4 — the estimator gap.** Should the calibration measure the constant and linear shot-phase
   terms *separately* (across-volume and within-volume respectively) rather than conflating them?
4. **5.6 — Stage A memory.** Should a full-pipeline memory run gate anything before this is used in
   anger at `o = 4`?
5. **4.19-4.23.** Four independent defects in the scoring path, each of which made the suite report
   confidently wrong numbers while running clean. Is the current metric trustworthy, or does it
   want an independent reimplementation to cross-check?
