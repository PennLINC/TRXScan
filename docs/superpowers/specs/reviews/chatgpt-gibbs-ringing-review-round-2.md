The revised design is **substantially better**. It correctly incorporates nearly all of the major points from my previous review, and the central architecture now looks sound. I would be comfortable moving toward an implementation plan **after another cleanup pass**, because there are still several important inconsistencies—some are simply stale text, but a few affect the proposed validation strategy.

The biggest problem is that the **design was revised much more completely than the assessment**. Several recommendations in the revised assessment still prescribe approaches that the revised design explicitly rejects.

## 1. The major conceptual problems from revision 1 are fixed

The revised design handles the most important issues correctly. In particular, it now:

* makes oversample → finite acquisition band → target reconstruction mandatory rather than optional;
* removes the inappropriate `2π * fmap * TE` image-phase term for spin-echo DWI;
* treats object phase above the acquisition Nyquist as legitimate content rather than something that must be suppressed;
* replaces the fixed ~9% voxel-sampled overshoot test with comparisons across subvoxel edge positions;
* adds first-class 6/8 and 7/8 partial-Fourier tests;
* separates unapodized algorithm-validation mode from optional scanner-realism windows;
* replaces phase-histogram heuristics with deterministic complex-valued invariants;
* acknowledges that an integer oversampling ratio is an implementation choice rather than a Fourier requirement;
* correctly describes the nominal sinc PSF as inherent to the finite acquisition rather than claiming “no blur.”

Those are substantial improvements. The overall simulation architecture in §§3.1–3.2 is now one I think is defensible for generating complex dMRI data with genuine finite-Fourier Gibbs ringing.

## 2. The assessment still contains the old recommendations

This is the most obvious thing I would fix before calling the assessment final.

The revised assessment's **R2** still recommends:

> “a smooth spatially-varying background phase ... or derived from the existing fieldmap as `phi_0 = 2*pi*fmap*TE`”

and then says R2 should be done even if R1 is deferred.

Both statements directly contradict the revised design, which correctly removes the fieldmap-derived phase and explicitly states that R1 is **not optional**.

Likewise, assessment **R3** still tells the implementation to scale noise variance by the sampled fraction. That is exactly the double-counting problem revision 2 says it corrected.

Assessment **R5** still asks for:

* overshoot approximately 9%;
* period `2.0 +/- 0.05`;
* increasing image-noise autocorrelation.

Those are the tests the revised design explicitly replaced with the subvoxel-profile family and analytic covariance test.

So §§3/R1–R5 of the assessment need to be rewritten from the revised design rather than lightly edited.

### The assessment's opening of §2.2 also still contradicts its own caveat

It currently begins:

> “with the first overshoot in the voxel immediately adjacent to the edge, at ~9% ... regardless of matrix size”

and then immediately explains that a correctly simulated acquisition produces sampled overshoots from roughly 0.29% to 8.93% depending on subvoxel edge position.

The second statement is the useful one. I would change the opening to something like:

> The continuous rectangularly truncated step response approaches the ~8.95% Gibbs overshoot, but the amplitudes observed at discrete voxel centers depend strongly on subvoxel edge position. The invariant relevant here is that the Fourier cutoff occurs at the nominal acquisition bandwidth.

That would make §2.2 internally consistent.

## 3. The design still carries a few stale revision-1 statements

There are several outright contradictions in the **decision log**.

It says:

* "`zero_ringing` → renamed to `apodization`"
* "Background phase source → derived from the existing fieldmap"
* FFT rewrite → optional phase 6

but the body now says:

* `zero_ringing` is replaced with `window: KspaceWindow`;
* the fieldmap-derived static phase was explicitly removed;
* optimization is phase 9.

Those rows should simply be regenerated from the current design.

There is another smaller stale cross-reference in §3.2:

> “Parameters for terms 1-3 are set by the phase-7 calibration”

Calibration is now **phase 8**, not phase 7.

And I would question whether the global phase parameter needs empirical calibration at all. It is essentially a gauge choice/test control. The empirically interesting parameters are the spatial background field and diffusion-dependent phase variation.

## 4. The noise convention is improved, but still not completely self-consistent

Revision 2 fixes the original double-scaling error, but there is still a definitional problem.

The design states:

> `noise_variance` is the variance of the complex Gaussian noise on each acquired k-space sample.

Then shortly afterward it derives the result using:

> `sigma^2 = noise_variance / (nx*ny)`

Those are not the same definition.

If `noise_variance` literally means **per-k-space-sample variance**, then each sampled point should have variance `noise_variance`. The reconstructed image variance will then depend on the normalization convention of the inverse transform.

If instead `noise_variance` is intended to mean **the full-sampling image-space variance**, then dividing the k-space variance by the appropriate normalization factor makes sense.

I would choose one of these explicitly. For example:

> `noise_variance` is the desired complex image-space variance under a fully sampled acquisition. The per-k-space-sample variance is derived from the DFT normalization.

or:

> `noise_variance` is the variance per acquired complex k-space sample. Image-space variance is derived from the reconstruction normalization.

Either is fine. Right now the prose chooses the second and the equation appears to choose the first.

### The covariance test also needs its scope specified

The statement

> image-noise autocovariance equals the inverse DFT of the sampling mask

is an excellent analytic test **for zero-filled linear reconstruction of independent k-space noise**.

It is not the final covariance after arbitrary GRAPPA reconstruction and coil combination. GRAPPA introduces correlations through its reconstruction operator, and the multi-coil combine changes covariance again.

So I would split this into:

1. **PF / simple undersampling with GRAPPA disabled:** analytic covariance = transform of the mask.
2. **GRAPPA enabled:** predicted covariance from the actual linear GRAPPA operator, or a Monte Carlo/reference implementation comparison.

That will prevent a mathematically correct simple-mask test from accidentally being applied to the wrong stage.

## 5. The complex phase model is now reasonable, but I would change how diffusion phase scales with b

The revised phase decomposition is much better:

* global phase;
* smooth combined-image background/reconstruction phase;
* volume/slice-dependent diffusion phase.

I agree with deferring complex coil sensitivities. They would improve scanner realism, but they are not necessary to establish that a reconstructed complex image can contain physically sensible Gibbs ringing in both channels.

I would, however, change:

> amplitude scaling with b-value

to something less prescriptive.

For simple translational motion under diffusion encoding, phase is fundamentally related to the diffusion wavevector and motion, while **b is quadratic in gradient/q amplitude** under fixed timing. So a linear relationship between phase amplitude and `b` is not the most natural physics model. It could grossly exaggerate phase at b=5000 relative to b=1000.

I would instead parameterize something like:

$$
\sigma_\phi(b) = a\,b^p
$$

with `p` configurable/calibrated, or operate directly from q-vector magnitude where possible. A \(\sqrt b\)-like dependence is a more natural starting point under simplified fixed-timing assumptions, but empirical calibration can determine whether something else is appropriate.

The important thing is to avoid baking **linear-in-b** into the architecture.

## 6. The subvoxel edge tests need a clearer relationship to the oversampling factor

The new analytic-profile test is much better than the old ~9% assertion, but there is an implementation issue worth resolving before coding.

The test proposes **16 different subvoxel offsets**, while the default simulation oversampling is only `o = 2`.

If the simulated step is represented as hard-valued cells on a 2× grid, you only have a handful of distinct edge positions within an acquired voxel. You cannot faithfully represent 16 offsets simply by shifting the edge between simulation-grid samples.

This is solvable in several ways:

* analytically integrate the step's fractional occupancy within simulation voxels;
* allow the test phantom to evaluate the continuous object directly rather than through the ordinary rasterizer;
* use a much larger `o` for the analytic correctness test and reserve `o=2` for the convergence/production test.

I favor the last two together.

For example:

**Oracle test:** continuous analytic step → exact Fourier-series solution.

**High-resolution implementation test:** `o=16` or `o=32`, all 16 offsets → extremely close to oracle.

**Production approximation test:** compare `o=2`, `o=4`, `o=8` and quantify convergence.

That would also tell you something very useful: **how much error does choosing `o=2` actually introduce?**

Right now the document assumes `o=2` is a good production choice and then adds a convergence test. I would make the result of that convergence experiment the criterion for whether `o=2` remains the default.

## 7. Be precise about the “sign alternates every voxel” test

I think this is fine as a specialized analytic invariant, but I would rename it to encode its scope.

Something like:

`axis_aligned_step_rectangular_full_fourier_sidelobes_alternate_sign`

The current name sounds as though *all Gibbs ringing* must alternate exactly voxel-by-voxel. That is not true once you introduce arbitrary edge orientation, 2D interactions, PF, apodization, etc.

For the intended 1D axis-aligned full-Fourier rectangular-window oracle, the test is useful.

## 8. The design still uses phase-distribution claims it says are not validity criteria

Revision 2 correctly removed:

* “Im/Re should be order 1”
* “phase should fill all eight bins”
* mean absolute wrapped phase as a correctness measure.

But the **problem table** still characterizes “Real MRI” as:

> `max|Im|/max|Re|`: Order 1
> mean |phase|: Full range, wrapped

Those statements are no longer necessary and are stronger than the design needs.

Similarly, the assessment still says:

> “In real DWI the background phase ... is full-range and wrapped”

as though that is a universal requirement.

I would replace these with something more robust:

| Property           | TRXScan today        | Required benchmark behavior                         |
| ------------------ | -------------------- | --------------------------------------------------- |
| Object phase       | Essentially absent   | Controlled, spatially varying complex phase         |
| Re/Im Gibbs energy | Almost entirely real | Rotates correctly according to imposed object phase |
| Inter-volume phase | Essentially absent   | Optional controlled diffusion-dependent variation   |

That expresses the actual failure without pretending there is one universal histogram of real DWI phase.

## 9. There is one larger missing component: define the **benchmark ground truth**

This is the biggest substantive addition I would make now.

The spec does an excellent job defining how to produce **ringing-corrupted complex data**, but your actual goal is not merely to simulate Gibbs—it is to **evaluate Gibbs-unringing methods**.

For that, TRXScan should expose an explicit artifact-free reference.

The design should define at least three related images:

1. **Continuous/high-resolution complex truth**
   The oversampled complex object before finite acquisition.

2. **Acquired ringing image**
   Finite rectangular/PF k-space → target reconstruction.

3. **Target-resolution Gibbs-free reference**
   A clearly defined target against which unringing output can be scored.

The third needs careful definition. You should not simply call the oversampled image “truth” and compare it voxel-for-voxel to an \(N\times N\) unringed result.

One reasonable benchmark target is the **exact voxel-integrated/downsampled complex object** at the nominal grid, without Fourier truncation ringing. That allows you to separately measure:

* residual oscillatory Gibbs error;
* edge-location bias;
* edge sharpness/resolution changes;
* complex-value error;
* magnitude error;
* phase error.

This becomes particularly valuable for distinguishing a method that genuinely suppresses Gibbs while preserving detail from one that simply smooths the image.

I would consider this a goal-level requirement, not just something for a later benchmarking document.

## 10. Add explicit end-to-end tests of the intended Gibbs-removal algorithms

Relatedly, the real-data calibration currently uses `mrdegibbs` behavior as a realism check, which is useful.

But once the basic simulator tests pass, I would add an **offline acceptance suite** covering the actual methods this work is intended to benchmark:

* conventional magnitude `mrdegibbs`/Kellner;
* RPG for PF;
* each proposed complex-aware method;
* a no-unringing control.

Test across:

* full Fourier;
* 6/8 PF;
* 7/8 PF;
* constant phase;
* spatially varying phase;
* diffusion-dependent phase;
* noiseless and noisy data;
* optional scanner windowing.

You do **not** need the methods to achieve predetermined rankings. The acceptance criterion is that changes in phase, PF, noise, etc. produce interpretable and physically consistent changes in their behavior.

That is the best demonstration that TRXScan has become a useful benchmark rather than merely a more sophisticated simulator.

## 11. One subtle concern about in-plane-only oversampling and motion

For a stationary 2D EPI acquisition, oversampling only the two Fourier-encoded axes is exactly the natural thing to do.

There is a complication if TRXScan applies arbitrary **3D head rotations before slice formation**. A rotation can move fine through-plane anatomical structure into the acquired in-plane directions. If the object is only finely sampled in x/y but remains coarse along z, the rotated slice is no longer necessarily an adequate sampling of the underlying continuous anatomy.

I would not make this a blocker for the Gibbs work. Instead:

* validate the Gibbs benchmark initially with motion disabled;
* document that in-plane oversampling is sufficient for the stationary acquisition;
* add an offline test with rotations to determine whether z-resolution materially affects the high-frequency content.

If it does, the better solution may be sampling each moved slice directly from the native 1-mm 3D source rather than globally oversampling z and paying a huge memory penalty.

## 12. Real-data calibration should be kept separate from correctness

I like the revised calibration quantities—phase-gradient RMS, structure functions, wrap density and inter-volume phase changes are much better than histogram occupancy.

But this sentence remains problematic:

> “Its role is to supply constants the analytic tests assert against”

The analytic correctness tests should ideally **not depend on empirical real-data calibration**.

I would separate:

**Correctness tests:** mathematical invariants and analytic references. These gate CI.

**Realism calibration:** estimates parameters for a named realistic preset, e.g. `PhaseModel::HBCDLike`. These can be checked into the repository but should not determine whether the Fourier implementation is mathematically correct.

**Behavioural validation:** compare simulator and real data with `mrdegibbs`, noise estimators, phase statistics, etc.

That separation will make future changes much easier to reason about.

---

# My revised verdict

The **design is now fundamentally sound**.  I no longer see a conceptual objection to the central strategy of:

**higher-resolution object → controlled complex phase → finite nominal acquisition band/PF mask → sampled k-space noise → reconstruction.**

I would therefore change my assessment from the previous **“needs substantive revision before implementation”** to:

**“Architecture approved; specification cleanup and validation details needed before implementation.”**

The remaining items I would treat as blockers are relatively focused:

1. reconcile the noise-variance definition with its equation;
2. rewrite the stale recommendations in the assessment;
3. clean the stale decision-log entries and phase references;
4. define how 16 subvoxel offsets are represented relative to `o=2`;
5. make the analytic noise-covariance test explicitly pre-GRAPPA or model the GRAPPA operator;
6. avoid hardcoding diffusion-phase amplitude as linear in `b`;
7. **define and output the Gibbs-free nominal-resolution benchmark truth.**

That last item is the main thing the agent has not yet addressed. Once it is present, the design will not merely generate much more realistic complex Gibbs ringing—it will provide the ingredients needed to quantitatively determine whether the complex-aware unringing methods actually improve reconstruction.
