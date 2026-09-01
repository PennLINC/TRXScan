I reviewed the current PR head, `31b389db4cf7b8eb95613e0e33b0e3c6e7b434c4`. The three blockers from my previous review were genuinely fixed: the normal CLI now has an oversampled acquisition path and calibrated phase model, the duplicated noise factor is gone, and the Nyquist ringing metric is complex-valued.

I would **still request changes before merge**, though. The remaining problems are now more concentrated.

## 1. New merge blocker: the oversampled production path silently discards several existing TRXScan features

The new CLI first constructs `comp` using the existing signal-stage logic. That correctly handles:

* `--motion` via `generate_compartments_moving`;
* SIFT2 `--weights`;
* `--kappa`;
* `--myelin`;
* multiband dropout modifications.

But when `--oversample > 1`—now the default—the actual output does **not use that `comp`**. It loads the simulation-grid tissue and calls:

```rust
let sim_comp = generate_compartments(...)
```

from scratch, then feeds `sim_comp` to `simulate_acquisition_oversampled`.

`generate_compartments` has no weight, dispersion, myelin, or motion arguments. The codebase explicitly documents `generate_mixture` as the path required for SIFT2 weights and κ dispersion.

So under the new default path:

* `--weights` is silently ignored;
* `--kappa` is silently ignored;
* `--myelin` is silently ignored;
* `--motion` is silently ignored;
* `--dropout-rate` modifies the discarded acquisition-grid `comp`, so the output does not contain the dropout even though its ground-truth TSV is written.

That is worse than merely not supporting these options: the CLI accepts them and can print messages implying they were applied.

### Required fix

The simulation-grid signal stage needs to follow the **same branching logic** as the nominal-grid signal stage.

For example:

* motion → `generate_compartments_moving(&sim_grid, ...)`;
* weights/kappa → `generate_mixture(&sim_grid, ..., weights, ..., kappa)` → `signal_from_mixture`;
* myelin → a simulation-grid myelin map applied to that mixture;
* dropout → operate on `sim_comp.images`, not the discarded nominal `comp`.

Until all of these are supported, the CLI should at minimum **error explicitly** for unsupported combinations rather than silently ignore them.

I would add end-to-end regression tests proving that each of these options changes the oversampled output.

---

## 2. The new PF-aware metric still usually scores the wrong spatial axis

I like the idea behind `residual_alignment`, but the implementation is not yet genuinely PF-aware for the actual benchmark phantom.

The benchmark is deliberately a **symmetric 2D box** so there are edges in both x and y.  Partial Fourier acts specifically along **ky / y**.

But `score()` calls `dominant_axis(ref_mag)` and evaluates all edge metrics along that **one** axis. `dominant_axis()` breaks an exact tie by keeping the first axis encountered—axis 0.

For the no-phase and constant-phase square phantom, x and y gradients are symmetric. So the metric selects **x/readout**, while PF modifies **y/phase-encode**.

Consequently the reported:

> `pf=1 → 7/8 → 6/8`

`residual_alignment` trend can be driven mostly by how the methods behave around the ordinary x-direction Gibbs edge under a PF-modulated 2D image, rather than directly measuring the PF ringing around the y edge.

There is a particularly useful clue in the new independent scorer: its tie rule does the **opposite**—it chooses axis 1 when the gradient means tie.  Yet the cross-check tests use only one-dimensional edge fixtures, so this disagreement is never exposed.

### Required fix

I would stop auto-selecting one axis for the 2D benchmark.

At minimum report:

* readout-edge metrics;
* phase-encode-edge metrics;

separately.

For PF acceptance, use the **known PE axis explicitly**.

An even better general `residual_alignment` implementation would use a 2D edge/sidelobe mask rather than collapsing the problem into profiles along one selected axis.

Add a cross-check using the **actual 2D box fixture at PF=0.75**. That test should fail with the current primary-versus-naive tie behavior and will protect the intended PF axis.

---

## 3. `residual_alignment` is a projection, not “the fraction of artifact surviving”

The definition

$$
A=
\frac{|\langle R_m,R_0\rangle|}
{\|R_0\|^2}
$$

is useful, but the interpretation currently given in the docstring is too strong:

> 1 = removed nothing; 0 = artifact entirely gone.

Zero only means that the remaining residual is **orthogonal to the original artifact template**. A method could preserve substantial residual energy while spatially shifting or otherwise changing its pattern and obtain a small alignment.

I would keep the projection, but pair it with:

$$
E=
\frac{\|R_m\|_\Omega}
{\|R_0\|_\Omega},
$$

over the same sidelobe region.

Then you have:

* **residual energy ratio**: how much error remains at all;
* **template alignment**: how much of the original artifact pattern remains.

That is substantially harder for an algorithm to “game” accidentally.

---

## 4. The PF acceptance rule is still gated by the PF-blind metric

There is a logical inconsistency in `check_consistency()`.

The new rule correctly uses `residual_alignment`, but it decides whether there is enough artifact to test using:

```python
ctrl[0]["oscillatory_residual"] < floor
```

But `oscillatory_residual` is exactly the Nyquist metric that the code says becomes blind as PF grows more aggressive.

So the new PF-aware rule can be **skipped because the old PF-blind metric says there is little artifact**.

The gate should instead use a PF-aware quantity, ideally the norm of the clean control artifact \(R_0\) in the relevant PE sidelobe mask.

There is another mismatch: the comment says:

> “a method must remove some of the control artifact”

but the code only fails when:

```python
a > 1.0
```

A method with `a == 1`—no reduction at all—passes.

That can be fine if the rule is really just **“must not amplify PF ringing”**, but then the comment and acceptance claim should say that. Right now the implementation and stated criterion differ.

Also, the acceptance unit tests have not been updated to include `residual_alignment` at all, so none of this new logic is directly tested.

---

## 5. Noisy `residual_alignment` currently conflates noise removal with Gibbs removal

For each row, `run_suite()` passes the input acquisition itself as the artifact template:

```python
control_mag=acq_m
control_phase=acq_p
```

For a noisy row, therefore,

$$
R_0 =
I_{\rm acquired,noisy}-I_{\rm ref}
$$

contains both Gibbs **and thermal noise**.

A method that only reduces noise can lower `residual_alignment` without removing any Gibbs.

That works against the point of emitting paired `acquired-clean` and `acquired-noisy` images.

For measuring Gibbs removal, I would use:

$$
R_0 =
I_{\rm acquired,clean}-I_{\rm ref}
$$

as the artifact template **for both clean and noisy method runs**.

Or, more simply, use the artifact-removal acceptance metrics only on clean runs and use the paired noisy runs strictly for robustness/noise-amplification analyses.

The current summary also averages clean/noisy and Hann/unwindowed cases together by `(method, pf)`, which makes the reported PF numbers harder to interpret than they appear.

---

## 6. The PF sampling mask is still a Fiberfox-specific quirk, not exact 6/8 or 7/8 sampling

The new AP/PA symmetry test is useful, and the spikes are now correctly restricted to acquired samples.

However, look closely at `sampling_mask()`.

For a 32-line matrix at PF=0.75:

* reverse PE keeps indices `0..24` → **25 lines**;
* forward PE keeps index `0` plus `8..31` → also **25 lines**.

So nominal “6/8” is actually:

$$
25/32=78.125\%.
$$

And one polarity has a disconnected isolated Nyquist line.

The test explicitly allows a one-line discrepancy from `round(ny*pf)`, so it verifies polarity symmetry but not actual 6/8 acquisition.

This is not an accidental invention in TRXScan: Fiberfox itself contains the same special case, explicitly preserving line zero on even matrices because the opposite Nyquist sample is absent.

So this is a **design decision**, not necessarily a porting bug.

But for your use case—evaluating RPG and other PF-aware unringing algorithms—I would seriously consider separating:

* `FiberfoxCompatible` PF sampling;
* conventional **contiguous zero-filled 6/8 or 7/8** sampling.

The latter is probably the more relevant benchmark of scanner-like reconstructed PF data, especially now that the object is genuinely complex and no longer has simple Hermitian symmetry.

I would not label the existing mask simply “6/8” without documenting this behavior.

---

## 7. `o=4` is still the production default even though the new measurement says it is not safely a production default

The new memory measurement was very useful. The PR now reports about **13.04 GB peak RSS** for the realistic `o=4` run and explicitly says this can OOM a 16-GB machine.

Yet the CLI still defaults to:

```text
--oversample 4
```

and merely prints a warning before proceeding.

There is an additional issue that makes the parallel case worse. `generate_compartments` under the `par` feature creates roughly `2 × thread_count` independent accumulation buffers, each containing an `nvox × ngrad` **f64** fiber array.

At the reported `o=4` HBCD-sized grid, one such logical buffer is already on the order of **16 GB**. Replicating it across Rayon groups is not viable on ordinary hardware.

So the previously attractive “use 8 cores and the runtime becomes manageable” story collides with the Stage-A memory implementation.

### Recommendation

Before treating `o=4` as the ordinary CLI default, I would do one of:

1. implement z-slab/streamed Stage A;
2. redesign parallel accumulation so it does not replicate full-volume buffers;
3. temporarily default the general simulator to `o=2`, while keeping `o=4` as the benchmark/high-accuracy option.

The existing nominal-grid `comp` is also constructed and retained before `sim_comp` is built, even though its output is discarded on the oversampled path. Refactoring issue #1 will eliminate that unnecessary work and memory too.

---

## 8. CI exists now, but the current head is red

This was a good addition, but the current workflow run on this exact head completed with **both jobs failing**:

* Rust failed at **Build (default features)**;
* Python failed at **Test**.

The workflow claims to be self-contained.  There are at least two obvious inconsistencies with that claim:

* `Cargo.toml` still contains optional sibling **path dependencies** (`../rust/trx-rs`, `../odx-rs`), which a clean Actions checkout does not contain.
* the Python job installs `numpy scipy pytest`, but `test_oversample.py` imports `nibabel` at module import time, before it can skip for the absent truth bundle.

So CI needs to be green before merge.

I would also remove `continue-on-error: true` from Clippy once the basic workflow is functioning; otherwise it is informational rather than a gate.

---

# Things that are now satisfactorily fixed

Several changes from the last round look good:

* The main CLI really does route to `simulate_acquisition_oversampled` by default and explicitly retains a documented legacy path.
* The benchmark grid is correctly reduced from 48 to 24 fixtures, with a clean/noisy pair emitted by each.
* `PhaseKind::Diffusion` now uses the calibrated HBCD-like model rather than the drastically weaker arbitrary phase constants.
* The complex Nyquist metric is now globally phase-rotation invariant.
* The Siemens stored/scaled representation is now explicitly documented and tested in both forms, and provenance is checked into the calibration JSON.
* The previous “old mechanism was uncorrectable” overstatement was corrected to a graded 6%/25%/50% result.
* Spikes now respect the sampling mask.
* `np.interp` now sorts its b-values first.
* The independent scorer is a good idea and worth retaining.

## Remaining calibration caveat

`HBCDLike` still calibrates `c_q × sigma_dx`, while `sigma_rot = 2e-3` remains essentially a chosen value rather than a separately fitted spatial-linear phase parameter. The code documentation is honest that only the translation amplitude is calibrated, but I would explicitly label `sigma_rot` as heuristic until the within-volume gradient calibration is done.

RPG also remains absent, so the benchmark still lacks the most important PF-specific external method. I don't think that has to block the forward-model PR, provided the PR does not claim that the PF-aware **method comparison** is complete.

# Revised verdict

The central Fourier/phase implementation is now in much better shape, and the three blockers I previously identified were actually resolved.

I would still **request changes**, primarily for:

1. **silent loss of motion/weights/κ/myelin/dropout on the new default oversampled production path;**
2. **PF scoring along a single auto-selected axis, which misses the actual PE axis for the symmetric benchmark;**
3. **the PF metric's projection/noise/gating problems;**
4. **the unresolved high-memory `o=4` production default, especially under `par`;**
5. **currently failing CI.**

The first is the most serious product regression. The second and third are the most serious threats to the scientific validity of the PF benchmark. Once those are fixed, I think the PR will be much closer to something I would approve.
