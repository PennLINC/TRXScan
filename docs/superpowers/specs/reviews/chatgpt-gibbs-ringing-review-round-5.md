This revision is now **implementation-ready in its core architecture**. The four issues I considered pre-freeze in the previous round have all been addressed correctly: acquired-band convergence replaces the bad simulation-Nyquist criterion; the complex-rotation test replaces the incompatible Re/Im energy test; reconstruction windows now filter signal and noise together; and GRAPPA is explicitly excluded from the canonical Gibbs-only fixture.

I would **not reopen the architecture**. I do, however, see a handful of specification details worth cleaning up before or while writing the implementation plan. Two are substantive modeling ambiguities; the rest are documentation cleanup.

## 1. The biggest remaining issue: `q = q_scale * sqrt(b) * bvec` is not really “the q-vector”

The move to using gradient direction is an improvement, but the spec now slightly overstates its physical interpretation.

It says:

> `phi_shot = q . dx, q = q_scale * sqrt(b) * bvec`

and calls this modeling “from the q-vector directly.”

But \(b\) and `bvec` alone do **not** determine the physical q-vector unless diffusion timing is fixed and known. In simplified PGSE,

$$
b \propto q^2(\Delta-\delta/3),
$$

so

$$
q \propto
\sqrt{\frac{b}{\Delta-\delta/3}}.
$$

Thus `sqrt(b) * bvec` is only proportional to q under a fixed-timing assumption. The undefined `q_scale` is absorbing diffusion timing and the radian/cycle convention.

That is perfectly acceptable for TRXScan if timing is fixed or unavailable, but I would call it an **effective q-vector** or **q-like phase-encoding vector**, unless TRXScan actually knows \(\delta\), \(\Delta\), and the gradient waveform.

For example:

$$
\mathbf q_{\rm eff}
=
c_q\sqrt b\,\hat{\mathbf g},
\qquad
\phi_{\rm shot}
=
\mathbf q_{\rm eff}\cdot\Delta\mathbf x.
$$

Then state explicitly that \(c_q\) is calibrated and absorbs the diffusion-timing convention.

If TRXScan later gains waveform/timing parameters, that can become a physical q-vector without changing the phase-model interface.

This is not a Gibbs-model blocker, but I would fix the terminology before implementation because otherwise the code may accidentally encode units nobody can later interpret.

---

## 2. `noise_variance` is still ambiguous once multiple coils are enabled

The factor-of-two ambiguity is now fixed nicely:

$$
\mathrm{Var}(\Re n)
=
\mathrm{Var}(\Im n)
=
\texttt{noise\_variance}.
$$



However, the spec says this is the variance of the **reconstructed complex image**, then derives each k-space sample's variance simply as

$$
\sigma_k^2=
\frac{\texttt{noise\_variance}}{N_xN_y}.
$$

That derivation is straightforward for one coil. With multiple noisy coils followed by the Roemer combination, the final noise variance also depends on the coil-combination weights/sensitivities.

So one of two definitions should be chosen:

**Option A — simplest:** `noise_variance` means the full-sampling per-component variance of the **single-coil reconstructed image before coil combination**. Multi-coil output variance then emerges from the coil model.

**Option B — preserve final-image semantics:** `noise_variance` is the requested final combined-image variance, and the per-coil k-space variance is scaled according to the coil-combination operator.

I favor **A**, especially since the canonical Gibbs fixture now explicitly uses a single/uniform coil. It makes the noise model local and mathematically transparent.

The implementation plan should at least investigate what the existing CLI `noise` value means in the multi-coil path before claiming that the new convention preserves current behavior.

---

## 3. Specify the reconstruction-window position relative to GRAPPA

The spec now correctly establishes:

$$
K_{\rm filtered}
=
W(K_{\rm signal}+n).
$$

Excellent.

But it still says:

> “The placement relative to GRAPPA is a documented modelling choice”

without actually making that choice.

Because revision 5 calls itself **frozen**, I would settle it.

For a generic *reconstruction* apodization window, my default would be:

$$
\text{acquire noisy undersampled k-space}
\rightarrow
\text{GRAPPA reconstruct}
\rightarrow
W(k)
\rightarrow
\text{inverse DFT}.
$$

That way the reconstruction window acts on both originally acquired and GRAPPA-synthesized lines.

If you specifically want to simulate scanner filtering occurring before parallel reconstruction, that could later be another explicit mode, but the generic `KspaceWindow` should have one deterministic location.

This has no effect on the canonical benchmark because GRAPPA is disabled there, but it matters for reproducibility of the acquisition-realism benchmark.

---

## 4. The GRAPPA covariance wording should be corrected

Test 8b currently says:

> “GRAPPA's kernel introduces correlation and the coil combine changes the covariance again, so **no closed form applies**. Compare against the covariance predicted by the actual linear GRAPPA operator…”



The latter sentence actually contradicts the former.

With fixed GRAPPA weights, the reconstruction is linear. If the noise vector has covariance \(\Sigma\) and reconstruction/combination is represented by \(A\),

$$
\Sigma_{\rm out}
=
A\Sigma A^H.
$$

That **is** an exact analytic expression.

What is no longer valid is the simple

$$
\mathcal F^{-1}\{M\}
$$

mask identity.

I would change this to:

> With GRAPPA enabled, the simple mask-only covariance identity no longer applies. Predict covariance from the explicit linear GRAPPA + coil-combination operator \(A\Sigma A^H\), with Monte Carlo as an independent validation.

That is both stronger and more precise.

---

## 5. There is still a stale sentence in the pre-readout phase section

The rewritten phase distinction is much better. In particular, the spec now correctly says that the pre-readout term may stand in only for phase that exists **before Fourier truncation**, while post-reconstruction phase belongs in a different transform.

But immediately after that discussion, the text still says:

> “This stands in for receive-chain, RF and **reconstruction phase in the combined image**.”

That is exactly the broader interpretation the preceding paragraphs just rejected.

I would delete that sentence.

Similarly, the later wording:

> “until then this term is named … a combined-image background phase”

should become simply **pre-readout object phase** everywhere.

Revision 5 has found the right distinction; a couple of revision-4 labels just survived inside the prose.

---

## 6. A common object phase is not really an approximation of multi-coil receive phase

Closely related: §3.2 says the smooth pre-readout object phase can stand in for:

> “complex receive-coil sensitivity phase”

and calls it a scalar approximation of the deferred complex-coil model.

For the canonical single-coil fixture, this is fine.

But in a genuine multi-coil model, every coil has a different complex sensitivity

$$
S_c(\mathbf r)=|S_c(\mathbf r)|e^{i\theta_c(\mathbf r)}.
$$

One common \(e^{i\phi(\mathbf r)}\) multiplier cannot reproduce those relative coil phases, which matter to GRAPPA and coil combination.

I would phrase the scope narrowly:

> The common pre-readout phase supplies a controlled complex object for Gibbs benchmarking. In single-coil simulations it may also mimic a smooth receive phase. It is not a substitute for coil-specific complex sensitivities in multi-coil realism simulations.

That matches the decision to defer complex coils without overclaiming what the current phase model does.

---

## 7. Real-data calibration of the pre-readout phase remains non-identifiable

The design now explicitly warns not to equate all observed combined-image phase with the pre-readout object phase. Good.

But §4.2 still says the measured real-data phase statistics:

> “set the amplitudes of the 3.2 terms 2 and 3”

and the division of labor says either real dataset can provide “background-phase spatial statistics.”

For term 3—diffusion-dependent phase—that is sensible after accounting for thermal phase uncertainty.

For term 2, reconstructed phase is an inseparable mixture of things such as:

* magnetization phase;
* coil phase/combination;
* scanner phase conventions/corrections;
* possibly reconstruction filtering.

So I would distinguish:

**Term 3:** quantitatively fitted where identifiable.

**Term 2:** tuned so simulated final phase has realistic spatial scales/amplitudes, but treated as an **effective benchmark parameter**, not a recovered physical pre-readout phase distribution.

That distinction is already implicit in §3.2; §4.2 should use the same language.

---

## 8. `object-nominal` should explicitly be a block **mean**, not merely “integrated”

The math in §3.5.2 correctly uses

$$
\frac1N\sum_j z_j,
$$

which is what I think you want given the simulator's intensity normalization.

Elsewhere it says “integrated” and “block-reducing.” I would call this a **complex block average / voxel average** explicitly.

Otherwise an implementation could accidentally use a sum and multiply intensities by \(o^2\).

This is a small implementation-specification detail, but an easy source of a catastrophic scale error.

---

## 9. Minor test-suite cleanup

The consolidated phase test is correct and is much stronger than the old `cos²/sin²` assertion.

The numbering now jumps:

1, 2, 3, **4**, **6**, 7, …

because tests 4 and 5 were consolidated.

I would renumber them before writing the implementation plan, particularly because other sections refer to “test 11,” “test 12,” etc.

Otherwise those references will become tedious to maintain.

I'd also define the “ringing residual” used by the phase-rotation test explicitly. For example,

$$
R_\phi
=
I_{\rm acquired,\phi}
-
I_{\rm object\text{-}nominal,\phi}
$$

in the noiseless canonical fixture.

Then assert

$$
R_\phi=e^{i\phi}R_0.
$$

That prevents an implementer from choosing some other residual definition.

---

## 10. Two remaining document leftovers

The decision-log preamble still says:

> “Regenerated from the current body at **revision 3**”

although the document is now revision 5.

And the problem table still says:

> Ringing present in real MRI: **Always**

even though both the assessment and previous revisions correctly softened that claim. The assessment says scanner filtering and real edge structure can suppress visible ringing.

I would replace “Always” with something like:

> Finite-band sinc PSF intrinsic; visible ringing depends on object and reconstruction filtering.

Likewise, the opening problem sentence still says the current artifact has “wrong and non-monotonic amplitude.” Since the assessment now carefully explains that sampled amplitude alone is not the load-bearing problem, “nonphysical parameter-dependent amplitude behavior” would be more precise.

---

# Assessment document

The revised assessment is now in good shape. In particular, turning §3 into a **pointer index** rather than repeatedly paraphrasing the changing design was a good structural fix; it should prevent the drift that occurred in the previous rounds.

Its core conclusion also remains correct:

* TRXScan currently needs an explicit extra low-pass operation to create ringing;
* that moves the Fourier cutoff below the nominal acquisition bandwidth;
* the underlying simulated object lacks meaningful complex phase;
* noise is incorrectly populated in non-acquired k-space;
* the existing test does not constrain any of those properties.

I don't see a new substantive problem in the assessment itself.

## Verdict

I agree with the new status:

> **“revision 5 — frozen pending implementation plan.”**

I would no longer send this through another architecture-review cycle. The core physics and benchmark design have converged.

Before implementation, I would make only these targeted edits:

1. call `sqrt(b)*bvec` an **effective q-vector** unless physical diffusion timing is available;
2. resolve the multi-coil meaning of `noise_variance`;
3. choose the reconstruction-window order relative to GRAPPA;
4. replace “no closed form” for GRAPPA covariance with the explicit linear-operator covariance;
5. remove the remaining “combined-image/reconstruction phase” wording from the pre-readout phase term;
6. clarify that the common object phase is not a substitute for coil-specific phase in multi-coil simulations;
7. explicitly make `object-nominal` a complex block **mean**;
8. clean the test numbering and small stale prose.

Those are implementation-semantics/wording issues rather than architectural blockers. With them either patched into the spec or made explicit in the implementation plan, I would proceed.
