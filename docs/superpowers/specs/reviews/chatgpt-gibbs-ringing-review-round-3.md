This iteration is **very close**, but I still see two conceptual issues that I would fix before implementation, plus several smaller specification problems. The assessment is now much more internally consistent with the design than the previous version.  The design has also incorporated essentially all of the previous review points.

My current verdict would be:

**Architecture approved, but I would not freeze the spec yet. Two benchmark-reference issues remain important enough to address first.**

## 1. `truth-nominal` is not actually “free” from Stage A at `o=1`

This is the most important remaining problem.

The design now defines:

* `truth-hires = f e^{i\phi}` on the simulation grid;
* `acquired` = finite-band reconstruction;
* `truth-nominal` = exact voxel-integrated complex object on the nominal grid.

I strongly agree with having all three.

But the claim that `truth-nominal` is exactly what Stage A already computes at `o=1` is generally false **once the new spatial phase model exists**.

Suppose a nominal voxel contains high-resolution samples with different phase:

$$
z_j=f_j e^{i\phi_j}.
$$

The desired reference is something proportional to

$$
\frac{1}{N}\sum_j f_j e^{i\phi_j}.
$$

Running Stage A at `o=1` gives you the coarse signal amplitude and then, under the proposed phase architecture, essentially applies one coarse phase value:

$$
\bar f\,e^{i\phi_{\rm coarse}}.
$$

Those are not equivalent when phase varies within the voxel. The first naturally includes intravoxel dephasing; the second generally does not.

This is particularly important because the design explicitly says:

> “Integrate the complex object, not the magnitude.”

That statement is correct, but the proposed implementation does not guarantee it.

### Better definition

Make the high-resolution complex object the single source of truth:

$$
Z_{\rm hires}=f_{\rm hires}e^{i\phi_{\rm hires}}.
$$

Then derive `truth-nominal` directly from it by spatial integration/block reduction over the corresponding \(o\times o\) simulation cells.

Because the design already imposes an integer simulation/acquisition ratio, this is cheap.

In other words:

**Do not independently regenerate the nominal truth at `o=1`. Downsample/integrate the exact same complex realization used for acquisition.**

That also guarantees that random diffusion-phase coefficients, tissue geometry, fractional occupancies, etc. are identical between the two branches.

The assessment repeats the same “Stage A at `o=1` is free truth” claim in R5, so it needs the same correction.

---

## 2. The scoring reference can be confounded by all the other simulated acquisition artifacts

This is the other issue I would consider a blocker.

`truth-nominal` is the pre-acquisition complex object. But TRXScan can also simulate:

* EPI distortion;
* T2/T2* evolution during readout;
* eddy-current effects;
* Nyquist ghosts;
* GRAPPA;
* motion;
* etc.

If `acquired` contains these effects but `truth-nominal` does not, then

$$
\text{unringed} - \text{truth-nominal}
$$

contains much more than Gibbs-removal error.

For example, geometric displacement from the fieldmap will look like huge edge-location error even if the unringing method is perfect.

The spec should therefore explicitly define a **Gibbs benchmark mode** in which unrelated effects are disabled or held out.

I would make the primary scoreable fixtures:

* oversampled object;
* complex phase;
* finite Fourier acquisition;
* optional PF;
* optional k-space noise;
* optional reconstruction window;

with **distortion, eddy currents, ghosts and motion disabled**.

Then later robustness experiments can turn those other effects back on, but they should not use raw distance to `truth-nominal` as though all errors were Gibbs errors.

That distinction would make §3.5 much stronger.

---

# 3. I would rename `truth-nominal`

The document itself correctly acknowledges:

> “No unringing method can be expected to reach `truth-nominal` exactly.”

because the acquired image is sinc-band-limited whereas this target represents box-integrated object values.

That makes `truth-nominal` a useful **object-space reference**, but calling it Gibbs-free “truth” risks implying that it is the physically expected result of an ideal Fourier reconstruction.

I would consider names such as:

* `object-nominal`;
* `reference-nominal`;
* `truth-box`.

Then:

* `truth-hires` = underlying simulated object;
* `reference-nominal` = object integrated onto nominal voxels;
* `acquired` = MRI acquisition/reconstruction.

This makes the distinction clearer when evaluating smoothing versus ringing removal.

---

## 4. The simulation-grid sampling-adequacy test should be changed

The current criterion is:

> spectral energy in the top 10% of simulation-grid Nyquist must be negligible.

I don't think that is a robust correctness criterion.

A sharp tissue boundary is intentionally **not band-limited**. Its Fourier coefficients decay slowly. Therefore a perfectly reasonable high-resolution representation of a discontinuous object can continue to have appreciable energy close to its simulation Nyquist.

You do not actually care whether the high-resolution object has negligible high-frequency energy.

You care whether the **acquired Fourier coefficients have converged** as the simulation grid becomes finer.

I would replace test 7 with something like:

### `acquired_band_converges_with_simulation_resolution`

For \(o=2,4,8,\ldots\), compare the central \(N_x\times N_y\) Fourier coefficients that constitute the actual acquisition:

$$
K_o(k_x,k_y)
$$

against the higher-resolution result.

Require

$$
\frac{\|K_o-K_{2o}\|}{\|K_{2o}\|}
$$

to be below a predefined tolerance within the **acquired band**.

That directly tests the numerical quantity the simulator needs to get right and does not require the underlying object to be artificially band-limited.

This would also provide a much better basis for selecting the production oversampling factor.

---

## 5. Test 11 still partially contradicts the new philosophy about choosing `o`

The design now correctly says:

> “The default `o` is an output of this study, not an input.”

Excellent.

But test 11 still says:

> “the `o=4` profile sits within tolerance of the `o -> infinity` limit.”

That presupposes that `o=4` must be adequate.

I would instead define an acceptable error threshold and choose:

$$
o_{\min}=\min\{o:\text{error}(o)<\epsilon\}.
$$

Then the production default becomes `o_min`, perhaps with a safety margin.

The test should check convergence, not prescribe beforehand which oversampling level passes.

---

## 6. There is a default-mode contradiction around full Fourier versus 6/8 PF

§3.1 calls:

> “Algorithm-validation mode (the default): unapodized rectangular **full-Fourier** acquisition.”

But elsewhere the spec emphasizes that the actual CLI default is:

> `partial_fourier: 0.75`

i.e. 6/8 PF.

Those cannot both describe the default configuration.

I think the intended distinction is:

* `KspaceWindow::None` is the default **window**;
* existing CLI acquisition defaults can still use 6/8 PF;
* the **full-Fourier benchmark fixture** explicitly sets `partial_fourier=1`.

I would say exactly that. There is no need to change TRXScan's shipping PF default just to get a clean full-Fourier benchmark.

---

## 7. The complex-noise convention needs one more definition

The image-space versus k-space variance contradiction is now fixed.

One small ambiguity remains: what exactly does **complex variance** mean?

For

$$
n=n_R+i n_I,
$$

does `noise_variance = σ²` mean

$$
E|n|^2=\sigma^2
$$

so that

$$
Var(n_R)=Var(n_I)=\sigma^2/2,
$$

or does it mean each component has variance \(\sigma^2\), giving \(E|n|^2=2\sigma^2\)?

Either convention is valid, but this factor of two propagates directly into SNR and calibration.

I would explicitly define `noise_variance` as, for example:

> `noise_variance = E[|n|²]` for the reconstructed complex image under full sampling, with independent real and imaginary components each having variance `noise_variance / 2`.

Then all analytic tests become unambiguous.

---

# 8. The diffusion phase model could use the q-vector more directly

The change from linear-in-\(b\) to

$$
\sigma_\phi(b)=a b^p
$$

with default \(p=0.5\) is a significant improvement.

But TRXScan already knows the diffusion gradient directions. If the physical motivation is

$$
\phi = \mathbf q\cdot\Delta\mathbf x,
$$

then a still cleaner model would make the random translational component explicitly dependent on the **q-vector**, not only its magnitude.

For example, draw a small motion vector \(\Delta\mathbf x\) for a shot and compute a phase offset from

$$
\mathbf q\cdot\Delta\mathbf x.
$$

That automatically gives:

* approximately \(\sqrt b\) scaling under fixed timing;
* dependence on gradient direction;
* sign reversal for reversed diffusion gradients.

Rotational motion can motivate the spatial linear phase terms.

I do **not** consider this necessary for the Gibbs implementation itself. The proposed \(a b^p\) model is adequate for producing nondegenerate complex images. But if the intent of phase 8 is quantitative calibration against real DWI, a q-vector formulation is more physically interpretable.

---

## 9. The background phase should probably be 3D-smooth

The reconstruction/background phase is currently described as a low-order **2D** field generated per slice.

For the Gibbs problem, that is enough because the acquisition is 2D.

For realistic complex DWI, however, independently generated 2D phase fields can create artificial discontinuities in z.

I would generate one smooth low-order **3D field**, then sample its 2D slices. Keep the diffusion-shot phase separate and allowed to vary between slice/MB groups.

That gives you:

* slowly varying structural/background phase in all three dimensions;
* acquisition-dependent shot phase where discontinuities are actually plausible.

Again, this is an enhancement rather than an implementation blocker.

---

# 10. Calibration of diffusion-phase versus b needs to account for phase noise

The proposed use of high-b CS-DSI to estimate \(p\) makes conceptual sense.

But apparent phase variation increases automatically when magnitude SNR drops.

At high b-values, the magnitude signal is weaker, so measured phase becomes less precise even if the underlying physiological/motion phase does not change.

Therefore fitting something like

$$
Var(\phi)\sim b^p
$$

directly from wrapped reconstructed phase risks interpreting **thermal phase uncertainty** as diffusion-induced phase.

The calibration procedure should either:

* estimate/remove the phase-noise contribution;
* restrict fitting to sufficiently high-SNR voxels;
* use phase estimates after an appropriate complex denoising/phase-estimation procedure;
* or model observed phase variance as signal-phase variance + noise-dependent phase variance.

Otherwise \(p\) is likely to be biased upward.

---

## 11. Reconstructed real data should not determine the theoretical sampling-mask covariance

Relatedly, I would soften this statement:

> “NIBS/HBCD for sampling-mask-dependent constants.”

If the NIBS data are reconstructed magnitude+phase images rather than raw k-space, their noise covariance also reflects:

* scanner reconstruction;
* GRAPPA kernel;
* coil combination;
* possible k-space filtering;
* interpolation;
* any proprietary reconstruction behavior.

So they are excellent for **behavioural realism comparisons**, but they cannot provide a clean empirical ground truth for covariance attributable solely to the PF/GRAPPA sampling mask unless those reconstruction details are known.

The analytic simulator covariance should remain derived from its explicit reconstruction operator. Real reconstructed data should be a plausibility comparison, not the source of theoretical mask constants.

This fits nicely with the correctness/calibration/behavioural separation that revision 3 already introduced.

---

# 12. Phase-error scoring needs a magnitude mask or weighting

The proposed error decomposition includes separate magnitude and phase error.

For complex data, phase becomes essentially meaningless where magnitude approaches zero.

So a future metric should not average raw phase difference over all voxels.

Use either:

$$
\Delta\phi = \arg(z_{\rm estimate}z_{\rm truth}^*)
$$

with a truth-magnitude threshold, or magnitude-weight the circular phase error.

This is a small detail now but will matter once the benchmark starts producing quantitative rankings.

---

## 13. A paired clean/noisy acquired output would be useful

Since one stated goal is evaluation under noise, I would consider emitting both:

* `acquired-clean`;
* `acquired-noisy`;

from the same simulated object and phase realization.

The document currently defines one `acquired` image that is optionally noisy.

Having the clean pair available makes it possible to distinguish:

* residual Gibbs;
* noise amplification;
* interaction between denoising and unringing.

Given the relatively low cost of keeping both products, this would make the benchmark considerably more informative.

---

# 14. Minor cleanup

There are still a few document-level leftovers:

* The header says **“Status: revision 2”** while immediately describing **Revision 3**.
* §6 says FFT is “phase 6”; the sequence now puts optimization in **phase 10**.
* §5.1 mentions optimization/direct evaluation in **phase 9**, but phase 9 is now the acceptance suite; optimization is phase 10.
* §6 says streaming is “built in phase 9 if it bites,” whereas the decision log correctly places it in phase 10.
* The goals still say “correct period (2 voxels) and amplitude (~8.95% first overshoot),” which is less careful than the new analytic-profile formulation. I would change that goal to **“the analytic nominal-band rectangular-truncation profile across subvoxel edge positions.”**

The updated **assessment** is otherwise in good shape. Its recommendations now actually track the current design, and the rewritten §2.2 correctly makes cutoff frequency—not a naive 9% sampled-overshoot criterion—the load-bearing criticism.

## Final assessment

Compared with the previous revision, I think the hard physics and forward-model problems are essentially solved.

I would classify the remaining issues this way:

**Before implementation**

1. derive `truth-nominal` from the **same high-resolution complex realization**, not Stage A independently at `o=1`;
2. define a scoreable Gibbs benchmark configuration that does not conflate Gibbs error with distortion/eddy/ghost/motion error;
3. replace the top-of-simulation-band energy criterion with **convergence of the acquired k-space coefficients**;
4. make oversampling-factor selection tolerance-driven rather than requiring `o=4` to pass;
5. resolve the full-Fourier-default versus 6/8-PF-default wording;
6. define complex noise variance precisely.

**Can comfortably follow during implementation/calibration**

* q-vector-aware diffusion phase;
* 3D-smooth background phase;
* SNR-aware estimation of diffusion-phase parameters;
* magnitude-weighted phase scoring;
* paired clean/noisy outputs.

With those first six changes, I would consider the specification ready to turn into a detailed implementation plan.
