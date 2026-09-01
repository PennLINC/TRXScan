I reviewed the audit and the current PR #4 head (`3ca5da61bee018982b8c7e9c21949fa49d69aeff`) directly. The PR is still a draft and now contains 35 commits across 34 changed files.   I did not execute the branch locally, so I am treating the audit's runtime/test results as reported results and independently reviewing whether the code supports them.

## Overall verdict

**I would request changes before merging.**

The low-level forward-model work is considerably better than the original TRXScan implementation. In particular, the PR correctly implements the half-cell registration correction, evaluates the finite acquisition from a higher-resolution object, makes signal and noise share one sampling mask, applies reconstruction windows to both signal and noise, derives `object_nominal` from the same high-resolution complex realization, and separates an effective pre-readout phase from fieldmap-induced EPI distortion. Those are all substantive improvements, and several of the implementation-time departures documented in the audit are good decisions.

However, I found **three major problems not identified in the audit**, the first of which means that the normal TRXScan executable still does **not** accomplish the original goal.

---

# Merge blockers

## 1. The normal `trxscan` path does not use the new Gibbs/complex forward model

This is the largest issue.

The ordinary `trxscan` executable still calls:

`simulate_acquisition(grid.dims, ..., &comp.images, ..., &fmap, ...)`

on the ordinary acquisition-resolution grid.

Inside `simulate_acquisition`, the new `SliceInput` is explicitly constructed as:

* `phase0: None`
* `sim: [nx, ny]`
* `acq_matrix: [nx, ny]`

with the comment:

> “v1 acquisition path: the object is already on the acquired matrix (o = 1).”

This matters because PR #4 simultaneously removes the old `zero_ringing` injection. Therefore, for ordinary TRXScan runs:

1. the object is still defined on the reconstruction matrix;
2. the forward/inverse DFT remains essentially an exact round trip;
3. the new intrinsic finite-band Gibbs mechanism never occurs;
4. `phase0=None`, so the new object-phase model is not used either.

In other words, **the new correct simulation exists in the low-level API and in `trxscan-benchmark`, but not in the main simulator the user actually runs**.

That contradicts the spec's production design, which calls for the simulation grid to be distinct from the acquisition matrix, with the production oversampling factor set to `o=4`.

It also makes this comment in `trxscan.rs` misleading:

> `window: KspaceWindow::None, // unapodized; ringing is now intrinsic to the acquisition`

because ringing is **not** intrinsic on that `o=1` code path.

### Required fix

I would keep the old `simulate_acquisition` as an explicitly legacy/compatibility path if needed, but add a real high-level oversampled acquisition path and make the main CLI use it:

**native/high-resolution inputs → simulation grid (`o=4`) → phase model → finite acquired band → target-resolution output.**

The CLI also needs an explicit phase-model choice or a defensible default such as the calibrated HBCD-like model.

Until this is done, I would not describe PR #4 as fixing Gibbs ringing in **TRXScan itself**. It currently fixes a new benchmark subsystem.

---

## 2. The benchmark noise factor is duplicated and partly mislabeled

The factor grid has:

$$
3\ {\rm PF}\times4\ {\rm phase}\times2\ {\rm noise}\times2\ {\rm window}=48
$$

points, and `FactorPoint` contains `noisy: bool`.

For a `p.noisy=false` fixture, the producer sets:

```text
noise_variance = 0
```

while for `p.noisy=true` it uses `1e-3`.

But **every fixture already emits both**:

* `acquired-clean`
* `acquired-noisy`

Then `acceptance.py` independently loops over:

```python
for noisy in (False, True):
```

and labels those rows according to that loop.

Consequently:

* a `_clean_` fixture has zero noise, so its `acquired-noisy` image is also clean;
* that second image is nevertheless recorded as `noisy=True`;
* a `_noisy_` fixture produces both a clean image and a genuinely noisy image;
* therefore the suite duplicates every clean condition.

This also means `check_consistency()` can have **multiple candidate controls** for the same `pf/phase/noisy/window` combination, one of which is a mislabeled noise-free result.

The audit's **288-row acceptance result** is therefore partly a consequence of this duplication, not 288 unique experimental conditions.

### Required fix

Drop `noisy: bool` from the fixture-factor grid.

Use:

$$
3\ {\rm PF}\times4\ {\rm phase}\times2\ {\rm window}=24
$$

fixtures, and let every fixture emit the paired clean/noisy acquisitions from a single configured nonzero noise level.

If different **noise amplitudes** are scientifically interesting, then make *noise level* the factor, but still emit clean/noisy pairs at each nonzero level.

I would rerun the acceptance suite after that correction before trusting its reported PASS.

---

## 3. The main ringing metric throws away the imaginary residual

This is particularly important given that the point of this work is **complex-aware** Gibbs correction.

`score()` correctly reconstructs complex values:

```python
ze = est_mag * exp(1j * est_phase)
zr = ref_mag * exp(1j * ref_phase)
diff = ze - zr
```

but then the primary ringing score is:

```python
oscillatory = _nyquist_amplitude(np.real(diff), ref_mag)
```

So the metric only measures the **real projection** of the complex Gibbs residual.

That means a simple phase rotation can alter the score without altering the amount of complex Gibbs error. A residual rotated predominantly into the imaginary channel can look artificially better.

This is exactly the kind of phase-dependence the simulator was changed to handle correctly.

### Required fix

Make the Nyquist projection complex-valued:

$$
c =
\frac{1}{N}\sum_x
R(x)(-1)^x
$$

and score

$$
\sqrt{\operatorname{mean}|c|^2}.
$$

Do not cast `diff` to its real component.

That makes the metric invariant to a global complex rotation and substantially more appropriate for comparing complex-aware methods.

I would add a test that globally rotating **both the object and its ringing residual** leaves `oscillatory_residual` unchanged.

---

# Important issues

## 4. I agree with withdrawing the partial-Fourier monotonicity rule—but PF is not validated yet

The audit is correct here.

The rule:

> “more aggressive PF must make unringing worse”

was too strong, and the current Nyquist projection cannot measure PF ringing adequately because PF changes the spectral structure. Withdrawing that assertion was preferable to asserting something the metric cannot see.

However, the consequence is substantial:

**the benchmark currently makes no validated claim about the PF axis**, despite 6/8 PF being TRXScan's shipping configuration.

RPG is also not implemented in the method harness. `run_method("rpg")` ultimately raises `MethodUnavailable`. The harness is honest about this, which is good, but it leaves the most relevant PF-aware method absent.

I would not block the *forward-model* fix on RPG, but I would change any summary from:

> “ACCEPTANCE: PASS”

to something explicitly narrower, such as:

> “Full-Fourier/Nyquist-component acceptance passed; partial-Fourier conditions are descriptive only.”

### A PF-aware metric is worth implementing before RPG

Because the simulator provides the exact control residual, you do not need to know PF's ringing frequency a priori.

For each fixture define the complex control residual:

$$
R_0 =
I_{\rm acquired,clean} -
I_{\rm object,nominal}.
$$

Then measure how much of a method's residual remains aligned with that control artifact:

$$
\frac{|\langle R_{\rm method},R_0\rangle|}
     {\|R_0\|^2},
$$

within an appropriate edge/sidelobe region.

That is:

* complex-valued;
* frequency-agnostic;
* naturally PF-aware;
* applicable to full Fourier too.

I would retain the Nyquist metric as a useful specialized measure for full Fourier and add the residual-template metric as the general one.

---

## 5. The calibrated Siemens phase conversion needs verification before trusting `HBCDLike`

The calibration code says:

> NIBS stores phase as `uint16`

while `siemens_phase_to_radians()` assumes the numerical values represent the signed range:

$$
-4096,\ldots,4095
$$

and simply computes:

$$
\phi = raw\frac{\pi}{4096}.
$$

Those statements need reconciliation. A raw `uint16` array cannot directly contain negative integers unless a particular offset, signed reinterpretation, or conversion convention is being applied.

The unit test only exercises synthetic signed values, not an actual NIBS raw range.

Because phase **scale**, not merely phase offset, determines:

* the reported 0.164 rad/voxel background gradient;
* circular SD;
* the fitted diffusion-phase amplitude;

this needs to be pinned from an actual NIBS image.

Before accepting:

```rust
c_q = 0.0425
```

and the HBCD-like background coefficients, I would record in the calibration JSON:

* actual stored dtype;
* observed raw min/max;
* conversion formula;
* confirmation using scanner/converter metadata.

The HBCD preset itself is otherwise well documented.

---

## 6. The benchmark's “Diffusion” phase condition does not use the calibrated HBCD-like phase

This surprised me.

The checked-in HBCD-like model uses:

```text
c_q = 0.0425
sigma_dx = 1.0
```

But the acceptance factor grid's `PhaseKind::Diffusion` uses:

```text
c_q = 0.002
sigma_dx = 0.4
```

plus an arbitrary background ramp.

At \(b=2000\), the diffusion-translation amplitude scales roughly as:

$$
0.002\times0.4\times\sqrt{2000}
\approx 0.036
$$

versus approximately

$$
0.0425\times1.0\times\sqrt{2000}
\approx 1.90
$$

for the HBCD-like model.

That's over a **50× difference**.

So the acceptance suite's “Diffusion” condition is not testing anything close to the calibrated realistic phase regime.

Once the phase scaling itself is verified, I would either:

* make `PhaseKind::Diffusion` use `PhaseModel::hbcd_like()`, or
* add a separate `HBCDLike` phase factor.

---

## 7. The audit overstates the “shipped mechanism was uncorrectable” result

The conceptual conclusion is still valuable, but the empirical headline should be corrected.

The audit describes the old **shipped** mechanism as having period ≈3.88 voxels and being essentially uncorrectable.

But the test that demonstrates this constructs the old mechanism by retaining only:

```python
keep[n // 4 : 3 * n // 4] = True
```

i.e. **50% of k-space**.

The actual previous CLI default was the much milder `zero_ringing=6%`, whose measured ripple period was about 2.13 voxels.

So the test establishes:

> aggressive instances of the old zero-fill-after-truncation mechanism are badly incompatible with Kellner-style methods.

It does **not** directly establish that the actual prior 6% default was “uncorrectable.”

I would add the real old 6% default to the test table, plus perhaps 25% and 50% stress cases, and soften the audit headline.

The load-bearing theoretical criticism remains valid regardless: the old parameter moves the cutoff away from the nominal Nyquist.

---

## 8. There appears to be an AP/PA partial-Fourier line-count asymmetry

From the sampling-mask logic in the PR, forward and reverse PE use different endpoint comparisons.

For `ny=32`, PF=0.75:

* one direction appears to keep 24 lines;
* the reverse branch appears capable of keeping 25 because the cutoff is inclusive.

I would add a very simple test:

```text
count(mask(reverse=false, PF=0.75))
==
count(mask(reverse=true, PF=0.75))
==
24
```

and equivalent 7/8 cases.

This is not central to the canonical benchmark, but it matters for realistic AP/PA DWI simulation.

---

## 9. Spikes can be injected into k-space samples that were never acquired

Signal and thermal noise now correctly share the acquisition mask. That's a good fix.

But spikes are generated afterward by choosing arbitrary matrix coordinates, regardless of that mask. A spike can therefore populate a PF-omitted or GRAPPA-omitted sample.

For physical consistency, random spike locations should be selected from the mask's `true` indices.

This does not affect the canonical Gibbs benchmark because spikes are disabled there.

---

## 10. Stage A memory should be measured before the regular CLI defaults to `o=4`

The audit correctly flags that only Stage B memory was measured, while the original estimate for Stage A was roughly:

$$
1.4\,{\rm GB}\rightarrow5.5\,{\rm GB}
$$

at the oversampled grid.

Because my biggest requested change is to make the **normal TRXScan pipeline** actually use `o=4`, this becomes important.

I would make one realistic HBCD-sized end-to-end `o=4` run a release criterion for that integration. If peak RSS is unreasonable, implement the already-designed slab/streaming mitigation.

---

# Review of the deliberate departures in the audit

My judgment on the audit's six departures is:

| Departure                                    | Judgment                                                                                                                                               |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Change phase sequencing                      | **Fine.** Implementation dependency is a better reason than preserving arbitrary spec order.                                                           |
| Use NIBS rather than ds006131 for `HBCDLike` | **Defensible, and I prefer it**, because the preset explicitly represents HBCD. Use ds006131 for scaling validation rather than owning HBCD amplitude. |
| Production `o=4`                             | **Supported by the reported convergence study.**                                                                                                       |
| Withdraw PF monotonicity rule                | **Correct decision**, but PF is consequently not accepted/validated yet.                                                                               |
| Move scripts into repository                 | **Correct.**                                                                                                                                           |
| Add test shim rather than rewrite 19 calls   | **Fine.**                                                                                                                                              |



I would therefore preserve most of the agent's implementation-time deviations.

---

# Answers to the audit's five reviewer questions

### 1. Should you build a PF-aware metric before RPG is available?

**Yes.**

Do not reinstate the monotonic-hardness rule. Build a frequency-agnostic complex residual/template metric first. Then RPG becomes the high-value external validation of that PF axis rather than a prerequisite for defining a sensible metric.

### 2. NIBS or ds006131 for `HBCDLike`?

**NIBS.**

A preset named `HBCDLike` should prioritize protocol match. The wider CS-DSI b range is valuable for checking whether the assumed \(\sqrt b\) behavior remains plausible over a wider range, but its acquisition differences make it a weaker source for the amplitude of an HBCD-specific preset.

This is conditional on verifying the NIBS phase-value conversion.

### 3. Should constant and spatial-linear diffusion phase be calibrated separately?

**Yes.**

The implemented model itself makes the distinction:

$$
\phi(\mathbf r)
=
\mathbf q_{\rm eff}\cdot
[\Delta\mathbf x + \boldsymbol\omega\times\mathbf r].
$$

The translation term is spatially constant; rotation produces a spatial linear field.

A within-volume spatial statistic cannot identify the constant term, while an across-volume global-phase statistic cannot identify the spatial gradient. The audit's 0.27–0.66 exponent spread is therefore partly an **estimator-definition problem**, not merely uncertainty in one scalar exponent.

I would fit:

* `c_q * sigma_dx` from volume/shot-to-volume global phase variation after removal of smooth spatial phase;
* `c_q * sigma_rot` from the distribution of within-volume phase gradients.

### 4. Should Stage A memory be a gate?

**Yes, before enabling `o=4` in ordinary TRXScan.**

It doesn't need to block the standalone benchmark binary, but it should block claiming that `o=4` is production-ready for full TRXScan acquisitions.

### 5. Is the current scorer trustworthy, or should it get an independent implementation?

**It needs an independent cross-check.**

The audit documents an unusually large number of independently plausible scoring bugs that all produced confident-looking output.  My direct review then found two more: duplicated noise conditions and a real-only Gibbs metric.

I would implement a second, simpler scorer based on known complex control residuals and compare both implementations on the exact same analytic fixtures. If the two disagree materially, CI should fail.

---

# Smaller cleanup items

A few things are worth fixing but would not independently block the PR:

* `Acquisition.noise_variance` still has the old field comment calling it “complex k-space noise variance,” while its implemented semantics are now **single-coil, pre-combination, full-sampling image-space per-component variance**.
* `KspaceWindow::Hann` and `Tukey` are implemented as **radial** windows. That's a legitimate choice, but it is not the usual separable Cartesian Hann/Tukey convention and should be documented explicitly.
* `fit_b_dependence()` calls `np.interp(b, bvals, snr)` but does not sort `bvals`; the helper should sort the observations internally before using `np.interp`.
* The PR currently has no review discussion and I found no reported status checks on the head. Given how many implementation defects were found only by execution, I would add CI for the self-contained Rust/Python suites rather than relying on the local audit alone. The PR itself is currently a draft.

## Recommended disposition

I would **request changes**, with this minimum merge checklist:

1. **Integrate the oversampled two-grid + object-phase model into normal `trxscan`.**
2. **Remove the duplicated clean/noisy factor dimension and rerun acceptance.**
3. **Make the oscillatory residual metric complex-valued.**
4. **Verify and document the actual NIBS stored-phase conversion, then regenerate calibration constants if needed.**
5. **Add a PF-aware complex residual metric; keep PF acceptance explicitly incomplete until then.**
6. Test AP/PA PF mask symmetry and mask k-space spikes.
7. Run an end-to-end `o=4` memory measurement.

After those, I think the underlying finite-Fourier implementation itself is on a strong footing. The half-cell registration correction in particular looks like an important implementation discovery, and the audit is right that it would have been very easy for a superficial “does the image change?” test to miss it.

The main gap is now less about the Fourier physics and more about **making sure the production path actually uses it, and that the benchmark measures complex/PF behavior without quietly collapsing back to a real/full-Fourier proxy.**
