This is the strongest iteration so far. The major architectural problems are now resolved, and the revised assessment is much better aligned with the spec. I would now describe the design as **nearly implementation-ready**, but I found a few remaining issues—two of them substantive enough that I would fix them before freezing the specification.

The revised design correctly fixes the nominal reference, isolates a Gibbs-specific benchmark mode, makes oversampling selection tolerance-driven, defines complex noise variance unambiguously, uses q-vector-aware diffusion phase, introduces 3D-smooth background phase, and separates theoretical correctness from empirical realism.  The assessment also now reflects most of those decisions.

## 1. There is still a stale, directly contradictory phase-sampling criterion

This is the clearest remaining internal contradiction.

In §3.2, the design still says:

> the spectral energy of `f * exp(i*phi0)` within the top 10% of the simulation-grid Nyquist band must be negligible

and points to test 4.1.7.

But §4.1.7 now correctly says that **this exact criterion is wrong** because sharp tissue boundaries are deliberately non-band-limited, and replaces it with convergence of the Fourier coefficients **inside the acquired band**.

The §3.2 paragraph should therefore be rewritten completely. I would make it say, essentially:

> The phase/object model is not required to be band-limited on the simulation grid. Sampling adequacy is established by convergence of the acquired Fourier coefficients as the simulation resolution increases (§4.1.7).

The decision log still has the related stale entry:

> Phase band-limit | Sim-grid sampling-adequacy condition

That should also become something like:

> Simulation-grid adequacy | Convergence of acquired-band Fourier coefficients

This is mostly a document consistency problem, but because §3.2 could otherwise guide an implementer toward reintroducing the rejected test, I would fix it before coding.

## 2. The `ringing_splits_across_re_im` test conflicts with the intentional even-N asymmetry

This is the most important new technical issue I see.

Test 4 currently says that for constant phase \(\phi\), ringing energy should split between the real and imaginary channels according to

$$
\cos^2(\phi),\quad \sin^2(\phi).
$$

That relationship assumes the unrotated ringing pattern is purely real.

But the design deliberately retains the even-matrix Fourier asymmetry

$$
[-N/2,\ldots,N/2-1],
$$

and explicitly states that this produces a small imaginary component even for an otherwise real object. Test 12 then pins that behavior as intentional.

Those two requirements are mathematically in tension. If the reference image is

$$
z_0(x)=a(x)+ib(x),
$$

then a global phase gives

$$
z_\phi(x)=e^{i\phi}z_0(x),
$$

not a real function whose energy simply divides as \(\cos^2\phi\) and \(\sin^2\phi\). The pre-existing \(b(x)\) term matters.

I would remove the \(\cos^2/\sin^2\) criterion and test instead:

$$
z_\phi=e^{i\phi}z_0
$$

for the **ringing residual itself**. For example, construct the φ=0 result, construct φ=π/4 and π/2 results, and verify that their complex ringing residuals equal an exact global rotation of the φ=0 residual.

That is stronger, works with the intentional even-N asymmetry, and is consistent with the existing `global_phase_rotation_is_exact` test. Tests 4 and 5 could probably be consolidated.

Alternatively, if you specifically want the pedagogical \(\cos^2/\sin^2\) test, perform it using an odd-N/conjugate-symmetric analytic fixture where the zero-phase PSF is genuinely real.

## 3. The optional reconstruction window needs an explicit position relative to noise

This has not yet been specified, and it matters because the original TRXScan bug already involved applying a signal-band limitation without giving noise the same treatment.

The design replaces `zero_ringing` with a **reconstruction window**—Tukey, Hann, Fermi, etc.—but does not explicitly say where that operator occurs relative to adding thermal noise.

If the implementation simply replaces the current `zero_ringing` code in situ, the sequence would apparently remain approximately:

$$
\text{signal}\rightarrow\text{window}\rightarrow\text{noise}.
$$

For a reconstruction/apodization window, that would be wrong. The acquired k-space already contains thermal noise, so multiplying by \(W(k)\) should multiply **both signal and noise**:

$$
K_{\rm filtered}(k)
=
W(k)\,[K_{\rm signal}(k)+n(k)].
$$

Otherwise scanner-realism mode recreates a version of the mismatch the redesign is intended to eliminate: filtered signal combined with unfiltered noise.

The spec should establish an explicit operator order. The exact placement relative to GRAPPA can be a documented modeling choice, but windowing must affect whichever noise has already entered at that stage.

Consequently, the analytic noise tests should explicitly state `window=None`; with a window,

$$
C(\Delta x)\propto
\mathcal F^{-1}\{|M(k)W(k)|^2\},
$$

rather than simply the inverse transform of \(M\).

I would consider this a pre-implementation requirement for §3.1/§3.3.

## 4. The clean Gibbs benchmark should explicitly address GRAPPA

The new Gibbs benchmark mode correctly disables distortion, eddy currents, ghosts, motion and T2/T2* readout effects.

It does not say what happens to GRAPPA.

If the purpose of the baseline fixture is:

> isolate error caused by finite Fourier truncation,

then GRAPPA reconstruction is another independent reconstruction operator. Even though the assessment found that the existing edge profile survives GRAPPA to about four decimal places, that empirical observation should not make it conceptually part of the canonical Gibbs-only benchmark.

I would define two levels:

**Canonical Gibbs benchmark:** no parallel acceleration/GRAPPA; ideally simple or uniform coil configuration; full Fourier or specified PF; optional noise/window.

**Acquisition-realism/robustness benchmark:** restore GRAPPA and realistic multi-coil behavior.

This also makes the score against `object-nominal` easier to interpret. With GRAPPA active, reconstruction error can contribute to the difference independently of Gibbs removal.

It does not mean GRAPPA should be removed from the overall benchmark suite—quite the opposite. It should be an explicit experimental factor rather than silently part of the “Gibbs-only” condition.

## 5. Test 9 also needs to be scoped away from GRAPPA and windowing

The current test:

> `noise_sd_scales_as_sqrt_sampled_fraction`

is correct for the simple masked zero-filled reconstruction under the stated normalization.

It is **not generally true after GRAPPA reconstruction**, where interpolation and g-factor effects change the noise variance, nor after a reconstruction window.

So I would state that test 9 is run with:

`GRAPPA disabled; KspaceWindow::None; identical per-sample thermal variance`.

Then its expected \(\sqrt f\) behavior is clean.

GRAPPA/window cases belong under covariance/operator-specific tests rather than that scalar law.

## 6. The pre-acquisition “background/reconstruction phase” still mixes physically different phase sources

This is the one remaining physics issue I would think carefully about rather than simply edit mechanically.

The design now generates one smooth 3D field and multiplies the object by it **before Fourier acquisition**, but describes this field as standing in for:

> receive-chain, RF and reconstruction phase in the combined image.

Those sources are not all equivalent.

A phase that exists in the transverse magnetization before/readout during acquisition genuinely modifies the complex object being Fourier truncated. Complex receive-coil sensitivity also multiplies the object before Fourier encoding.

But a phase correction or phase convention introduced **after reconstruction** does not alter which spatial frequencies were truncated. It rotates an already reconstructed Gibbs pattern.

That distinction matters specifically for your project because:

$$
\mathcal F_{\rm truncate}
\{f(x)e^{i\phi(x)}\}
$$

is generally not equivalent to

$$
e^{i\phi(x)}
\mathcal F_{\rm truncate}\{f(x)\}.
$$

So I would rename term 2 to something like **smooth pre-readout object phase** and only claim that it approximates physical phase sources that genuinely exist before finite Fourier encoding.

If you also want realistic combined-image/reconstruction phase, add a separately controlled **post-reconstruction phase transform**.

Complex coil sensitivities would ultimately be the more physically complete treatment of receive phase, as the design already recognizes.

I don't think this invalidates the current Gibbs benchmark; the deliberately imposed pre-acquisition phase is perfectly useful for testing complex-aware unringing. I mainly would not calibrate its parameters directly against all observed combined-image phase and then claim those statistics have a one-to-one physical interpretation.

## 7. The q-vector model is an improvement, but the implementation interface should be explicit

I like the move from \(a b^p\) to

$$
\phi_{\rm shot}=\mathbf q\cdot\Delta\mathbf x.
$$

It gives the desired \(\sqrt b\) scaling, direction dependence and sign reversal naturally.

One implementation detail should be nailed down in the plan: the document says the existing `gradients[g]` is constructed as approximately `bvec * bval`. That is **not itself the q-vector**.

So either pass `bval` and unit `bvec` separately into the phase model, which is cleanest, or explicitly recover them:

$$
b=\|\mathbf g_{\rm stored}\|,
\qquad
\hat g=\frac{\mathbf g_{\rm stored}}{\|\mathbf g_{\rm stored}\|},
\qquad
\mathbf q=q_{\rm scale}\sqrt b\,\hat g.
$$

I would prefer the separate B-value/b-vector interface because it makes dimensional semantics obvious.

## 8. The assessment has a few stale revision-3 statements

The assessment is now much better, but its recommendations no longer fully summarize the authoritative revision-4 design.

Most noticeably:

* It says the authoritative design is **revision 3** rather than revision 4.
* R2 still describes diffusion phase using `sigma_phi(b) = a*b^p`; the design now prefers explicit \(\mathbf q\cdot\Delta x\).
* R2 calls the phase field simply low-order background/reconstruction phase and does not mention that it is now explicitly 3D.
* R3 says `noise_variance` is the “complex image-space variance,” whereas the design now very specifically defines it as the **per-component** variance:

  $$
  Var(Re\,n)=Var(Im\,n)=\text{noise_variance}.
  $$
* R5 does not mention the new paired `acquired-clean` / `acquired-noisy` outputs.

These don't undermine the assessment's core verdict, but since the document explicitly says its recommendation section summarizes the authoritative spec, I would regenerate that section one more time.

There is also an awkward phrase in §2.4:

> “The small non-zero residual with ringing on is **not physics** — it comes from the one-sample asymmetry…”

But the same documents correctly decide that this even-N asymmetry is present in real Cartesian acquisitions and should be retained.

I would change “not physics” to:

> “is not realistic object phase; it is a deterministic consequence of the even-N Fourier sampling convention.”

That reconciles the two statements.

## 9. One more stale contradiction remains in the real-data section

The detailed NIBS paragraph now correctly says reconstructed NIBS magnitude+phase data:

> “cannot supply theoretical mask constants”

and should only be used for behavioral plausibility.

But the concluding “Division of labour” paragraph still says:

> “NIBS/HBCD for sampling-mask-dependent constants”

which contradicts the paragraph immediately above it.

That should become:

> **NIBS/HBCD for behavioral comparison of sampling-mask-dependent effects**; ds006131 for diffusion-phase dependence; either for background-phase spatial statistics.

## 10. A smaller stale default remains at the start of §3.1

The simulation-grid definition still says:

> `Default o = 2`

while the rest of revision 4 explicitly makes the production default:

$$
o_{\min}=\min\{o:\mathrm{error}(o)<\epsilon\}.
$$



Change the first statement to something like:

> Candidate starting value `o=2`; production default is selected by the convergence criterion in §4.1.11.

The cost and memory tables can still report `o=2` as a concrete example.

---

## Verdict

I would now call the **core forward-model architecture approved**.

The important pieces are in the right form:

$$
\text{higher-resolution complex object}
\rightarrow
\text{finite nominal k-space acquisition}
\rightarrow
\text{PF/noise/reconstruction}
\rightarrow
\text{complex reconstructed DWI},
$$

with the object-space nominal reference derived from **the exact same realization**.

Before turning the spec into the implementation plan, I would fix four things I consider genuinely important:

1. remove the stale simulation-Nyquist-energy criterion and use acquired-band convergence everywhere;
2. replace the incompatible \(\cos^2/\sin^2\) Re/Im ringing test with a complex-rotation test;
3. explicitly place reconstruction windows relative to noise so they filter noise as well as signal;
4. define whether the canonical Gibbs-only fixture disables GRAPPA, rather than leaving it implicit.

Everything else above is either clarification or cleanup. Once those four are resolved, I would be comfortable treating the specification as frozen and proceeding to implementation planning.
