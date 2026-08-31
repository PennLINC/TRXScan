# Gibbs ringing realism: design

**Date:** 2026-08-31
**Status:** revision 6 — **frozen**. Architecture converged at revision 5 and was not reopened;
revision 6 applies implementation-semantics and wording fixes only (section 12).
**Revision 6** applies the fifth review round; see section 12. No architectural change: effective
q-vector naming, single-coil noise semantics, window position relative to GRAPPA, the GRAPPA
covariance closed form, `object-nominal` as a block mean, and test renumbering.

**Revision 5** incorporated a fourth review round; see section 11. Substantive changes: the stale
simulation-Nyquist energy criterion is removed from 3.2; the Re/Im split test is replaced by a
complex-rotation test compatible with the even-N asymmetry; reconstruction windows are placed
explicitly after noise so they filter both; the canonical Gibbs fixture disables GRAPPA.

**Revision 4** incorporated a third review round; see section 10. Substantive changes: the nominal
reference is derived from the same high-resolution complex realization rather than regenerated
(3.5); a Gibbs benchmark mode isolates ringing error from distortion/eddy/ghost/motion (3.5);
sampling adequacy is retested as acquired-band convergence (4.1.7); oversampling selection is
tolerance-driven; the complex noise variance convention is pinned to a component convention.

**Revision 3** incorporated a second review round; see section 9. Substantive changes: the noise
variance convention is made self-consistent; the benchmark ground-truth outputs are promoted to a
goal-level requirement (3.5); diffusion phase is parameterized `a*b^p` rather than linear in b; the
sub-voxel offset tests are decoupled from the production oversampling factor; the noise-covariance
test is scoped pre-GRAPPA; correctness, calibration and behavioural validation are separated.

**Revision 2** incorporated the first external review (`reviews/chatgpt-gibbs-ringing-review.md`). Substantive
changes: the `2*PI*fmap*TE` phase term is removed as inappropriate for spin echo; the phase
band-limit constraint is inverted; the noise variance convention is fixed; the fixed-overshoot test
is replaced by a sub-voxel edge family; partial-Fourier ringing gets dedicated tests; R1 is no
longer optional; phases are reordered R1-first. See section 8 for the full disposition.
**Scope:** `src/kspace.rs`, new `src/phase.rs`, `src/compartments.rs` (grid plumbing),
`src/bin/trxscan.rs`, `data/trxscan_truth_data/scripts/prepare_acquisition_grid.py`

## 1. Problem

An audit of the k-space stage (see `docs/gibbs-ringing-assessment.md`) found that
TRXScan's Gibbs ringing uses the correct physical mechanism — k-space truncation — but produces an
artifact that is wrong in period, wrong and non-monotonic in amplitude, accompanied by spurious
resolution loss, and entirely absent from the imaginary channel. Measured on 64x64:

| Property | TRXScan today | Real MRI |
|---|---|---|
| Ringing present by default | No (DFT round-trip exact to 3e-15) | Finite-band sinc PSF intrinsic; visible ringing depends on object structure and reconstruction filtering |
| Ripple period | 2.13 px at the CLI default, 2.67 px at 25% | Exactly 2 voxels |
| Fourier cutoff | Moved off the nominal Nyquist by 6-25% | At the nominal Nyquist |
| First overshoot | 1.55%-10.04%, reverses above 25% | +0.29%..+8.93% by sub-voxel offset (see 4.1.1) |
| Effective resolution | 1.07x-1.33x nominal voxel | Nominal |
| Object phase | Essentially absent (max\|Im\|/max\|Re\| = 4.1e-16) | Controlled, spatially varying complex phase |
| Re/Im ringing energy | Almost entirely real | Rotates correctly with the imposed object phase |
| Inter-volume phase | Essentially absent (mean \|phase\| 0.19 rad) | Controlled diffusion-dependent variation |
| Noise autocorrelation under truncation | Unchanged (white at all truncations) | Correlated |

Root causes: (a) the object is defined on the reconstruction grid, so the forward/inverse DFT pair
is an exact identity and no ringing can arise; (b) the compartment signal is real and non-negative
(`kspace.rs:232-234`) and the only phase terms are ky-dependent (`kspace.rs:238-246`), so they warp
geometry rather than imprint image phase; (c) noise is added after truncation and after the
partial-Fourier line skip (`kspace.rs:302-309`), populating samples that were never acquired.

## 2. Goals and non-goals

**Goals**

- Gibbs ringing that is intrinsic to the forward model, at the nominal resolution, matching the
  analytic nominal-band rectangular-truncation profile across sub-voxel edge positions.
- Complex-valued output in which ringing distributes across real and imaginary channels according
  to a physically motivated object phase, so `part-phase` is usable.
- Noise consistent with the k-space sampling mask.
- Quantitative tests that pin all of the above, backed by constants calibrated against real
  complex DWI.
- **Scoreable benchmark outputs.** The goal is not to simulate Gibbs but to *evaluate Gibbs-removal
  methods*, which requires an explicit artifact-free reference alongside the corrupted image. See
  3.5; this is a goal-level requirement, not a later benchmarking concern.

**Non-goals**

- Replacing the direct-sum DFT with FFT. Deferred to an optional later phase; the direct sums are
  the comparison oracle and `kspace.rs:12-14` documents the deliberate avoidance of FFT-convention
  ambiguity.
- Non-Cartesian trajectories, ramp sampling, or through-slice (3D) encoding.
- Homodyne/POCS partial-Fourier reconstruction. Zero-filled PF stays as-is; only its noise handling
  changes.

## 3. Design

### 3.1 Two grids (R1)

Introduce a **simulation grid** distinct from the **acquisition matrix**.

- Simulation grid: `Grid { dims: [o*nx, o*ny, nz] }` with the in-plane columns of `voxel_to_world`
  divided by `o`. **The production default is `o = 4`**, set by the convergence study (4.1.10) and
  the cost/accuracy tradeoff below, not assumed. Cost and memory figures below use `o = 2` only
  because it is the smallest non-trivial case; see the measured table for the real figures. In-plane only: the slice direction is not Fourier-encoded in 2D
  EPI, so there is nothing to truncate along z and oversampling it would only cost.
- Acquisition matrix: `(nx, ny)` — same FOV, same voxel size, what is written out.

Because the sim grid covers the same FOV with finer voxels, the k-space sample spacing is unchanged
and k_max is `o` times larger. Acquiring the central `nx x ny` block of the `o*nx x o*ny` k-space is
therefore exactly acquisition at the nominal matrix over the nominal FOV.

**Implementation.** The acquired k-space sample `(kxi, kyi)` sits at absolute sim index
`kxi_s = sxs - xs + kxi` where `sxs = snx/2`, `xs = nx/2`. Its normalized frequency is
`(kxi_s - sxs)/snx = (kxi - xs)/snx`. So the forward loops keep running over the `ny` acquired lines
and the `nx` acquired readout points, and only the normalizers change:

```rust
// kspace.rs:216, 254 today
let ky_norm = (kyi as f64 - ys as f64) / ny as f64;
let kx_norm = (kxi as f64 - xs as f64 + ghost_shift) / nx as f64;
// R1: divide by the SIM extent; the loop bounds are unchanged
let ky_norm = (kyi as f64 - ys as f64) / sny as f64;
let kx_norm = (kxi as f64 - xs as f64 + ghost_shift) / snx as f64;
```

with the inner `x`/`y` sums ranging over `0..snx` / `0..sny`, centring on `sxs`/`sys`, and
`n_inv = 1.0 / (snx * sny)`. `inverse_2d` is unchanged: it reconstructs the acquired `nx x ny`
k-space onto the `nx x ny` grid, which is correct because the recon FOV equals the acquisition FOV.

Consistency check: with `u = (y_s - sys)/o` the acquired-centred coordinate,
`ky_norm_s * (y_s - sys) = ((kyi - ys)/sny) * o*u = ((kyi - ys)/ny) * u`, i.e. exactly the
acquired-matrix DFT evaluated on the finer object. DC is preserved: a uniform object gives
`kspace[0] = 1` and reconstructs to 1.

`SingleShotEpi` stays on the acquired matrix. `t(ky)`, `tRf`, `tRead` are properties of the readout,
not of the object, so `line_times` and the eddy/distortion timing are untouched.

**Cost.** Per slice, counting complex operations with `N = nx = ny`:

| Term | Today | R1 at `o` | R1 at `o=2` |
|---|---|---|---|
| modimg build | N^3 | o^2 N^3 | 4 N^3 |
| y-sum | N^3 | o^2 N^3 | 4 N^3 |
| x-DFT | N^3 | o N^3 | 2 N^3 |
| **Total** | **3 N^3** | **(2o^2 + o) N^3** | **10 N^3 (3.3x)** |

Computing the full `o*nx x o*ny` k-space and cropping afterwards would be `24 N^3` at `o=2` (8x), so
cropping during the transform saves 2.4x against the naive route. The net cost against today is
`(2o^2 + o)/3`.

**Measured cost and accuracy (phase 1).** The convergence study rejected the `o = 2` example this
section originally used. Acquired-band relative error: e(2->4) = 8.5e-3, e(4->8) = 4.0e-3,
e(8->16) = 4.0e-4. Image-domain profile deviation against the analytic oracle, judged against the
~8.95% Gibbs overshoot the benchmark exists to measure:

| `o` | cost vs today | profile error | as a fraction of the artifact |
|---|---|---|---|
| 2 | 3.3x | ~1.4e-2 | 15.6% - too coarse to benchmark against |
| **4 (default)** | **12x** | **~4e-3** | **4.5%** |
| 8 | 45x | 1.8e-3 | 2.0% |

`o = 4` is the default **on accuracy grounds**: the residual sits well below the effect being
measured. A strict 1e-3 acquired-band tolerance would select `o = 8`; that tolerance was never
justified against the artifact amplitude, and the table above replaces it.

**Measured wall time (release, 108x152 in-plane, single-threaded):** 124 ms/slice at `o = 1`,
252 ms at `o = 2`, 750 ms at `o = 4`. For a 104-slice, 75-volume acquisition that is 0.3 h, 0.5 h
and **1.6 h** respectively, and the stage is already parallel over volumes under the `par` feature.
An earlier draft claimed `o = 8` "would in practice require the phase-10 FFT path first"; the
measurement does not support that -- `o = 8` extrapolates to ~13.6 h single-threaded, roughly 1.7 h
on 8 cores, which is tolerable. The FFT path is therefore a convenience, not a prerequisite, and
phase 10 is deprioritized accordingly. `o = 4` stands on its accuracy justification alone.

**Input path.** `data/trxscan_truth_data/sub-0001a/anat/` holds 1 mm isotropic WM/GM/CSF probsegs
and fieldmap on a 193x229x193 ACPC grid, and `scripts/prepare_acquisition_grid.py` already
downsamples them to a 1.7 mm acquisition grid. R1 adds `--oversample N` to that script, emitting a
second set of maps at `voxel/N` over the same FOV. Stage A consumes the sim-grid maps; the fieldmap
also moves to the sim grid, which is more correct since distortion is a continuous-object phenomenon.

**On the integer ratio.** An integer ratio is *not* mathematically required. Both grids cover the
same FOV, so the Fourier sample spacing is `1/FOV` for both and the acquired samples are a subset of
the simulated ones for any `snx >= nx`. The integer restriction is an implementation choice, adopted
because (a) it makes each acquired voxel an exact union of sim voxels, which keeps the partial-volume
rasterization interpretable, and (b) it makes the centring offset `sxs - xs` parity-safe. It is
documented as a restriction, not a necessity.

**On resolution.** Cropping to the acquired band necessarily imposes a sinc PSF relative to the finer
object — that is not a defect, it is the *correct nominal acquisition PSF*. The distinction from
today's behaviour is that today discards a further 6-25% of an already target-resolution matrix,
producing resolution loss *below* nominal. The claim to make is "correct nominal PSF", never "no
blur".

**On the 1 mm source.** Resampling the 1 mm anatomicals onto a 0.85 mm simulation lattice does not
invent frequencies above the 1 mm source bandwidth, but the 1 mm data already contains frequencies
well above the 1.7 mm acquisition Nyquist. Those are precisely what generates Gibbs when the 1.7 mm
band is selected, so the resampling is legitimate. What is *not* legitimate is upsampling the
1.7 mm maps, which have no content above the acquisition Nyquist to recover.

Explicitly rejected: interpolating the existing 1.7 mm maps up to 0.85 mm. Sinc interpolation adds
no k-space content, so cropping back returns the original image exactly and produces zero ringing.
Smoother interpolants produce *less* ringing than the truth. The finer object must come from the
finer source.

**`zero_ringing` is removed and replaced by an explicit window.** Two modes are separated:

- **Algorithm-validation mode:** unapodized rectangular full-Fourier acquisition. The cleanest known
  forward model and the exact scenario Kellner/`mrdegibbs` assumes; MRtrix recommends disabling
  scanner filtering for best `mrdegibbs` performance.

  **This is a benchmark *fixture*, not the CLI default.** Two different defaults were previously
  conflated. To be precise: `KspaceWindow::None` is the default **window**; TRXScan's shipping
  acquisition default remains 6/8 partial Fourier (`partial_fourier: 0.75`,
  `src/bin/trxscan.rs:254`); and the full-Fourier benchmark fixture explicitly sets
  `partial_fourier = 1.0`. There is no reason to change the shipping PF default to obtain a clean
  full-Fourier benchmark.
- **Scanner-realism mode (opt-in):** a named reconstruction window with explicit parameters,
  `enum KspaceWindow { None, Tukey { alpha }, Hann, Fermi { radius, width } }`. These are different
  transfer functions with different PSFs and must not hide behind one ambiguous `f64`.

**Operator order — the window filters noise as well as signal.** The window is applied *after* noise
has entered the k-space:

```text
K_filtered(k) = W(k) * [ K_signal(k) + n(k) ]
```

This is stated explicitly because replacing the `zero_ringing` code in situ would preserve the
existing order — signal, then band limitation, then noise — which is precisely the defect this
redesign exists to remove (section 1, finding 2.5): filtered signal combined with unfiltered noise.
A reconstruction window acts on acquired k-space, which already contains thermal noise, so it must
multiply both.

**Position relative to GRAPPA — settled, not left open.** The generic `KspaceWindow` has one
deterministic location:

```text
acquire noisy undersampled k-space -> GRAPPA reconstruct -> W(k) -> inverse DFT
```

so the window acts on originally-acquired and GRAPPA-synthesized lines alike, which is what a
*reconstruction* apodization does. Scanner filtering applied *before* parallel reconstruction is a
physically different thing and, if wanted later, becomes a separate explicit mode rather than an
ambiguity in this one. This has no effect on the canonical benchmark, where GRAPPA is disabled, but
it makes the acquisition-realism benchmark reproducible.

Consequence for the noise tests: with a window active the image noise autocovariance is
`F^-1{ |M(k) W(k)|^2 }`, not `F^-1{M}`. Tests 4.1.7a and 4.1.8 therefore specify `window = None`
(see those tests).

The struct field `zero_ringing: f64` is replaced by `window: KspaceWindow`, defaulting to `None`.
There is no compatibility surface to preserve: the value is hardcoded at `src/bin/trxscan.rs:261`
and is not a CLI argument (usage string, `src/bin/trxscan.rs:186`).

Window strength is **not** calibrated from real tissue-edge ringing amplitude — that measurement is
confounded by edge orientation, partial volume, reconstruction filtering and sub-voxel position. If
it is calibrated at all, the input must be a phantom or a raw full-Fourier k-space dataset.

### 3.2 Object phase (R2), new `src/phase.rs`

A `PhaseModel` produces a per-slice `phi0: Vec<f64>` on the **simulation** grid, applied at
`kspace.rs:246`:

```rust
modimg[at(x, y)] = C::cis(TAU * phi + phi0[at(x, y)]).scale(f_real);
```

**Removed: `2*PI * fmap * TE`.** TRXScan simulates spin-echo EPI DWI, confirmed by the T2'
term `exp(-|t| / t_inhom)` centred on the echo (`kspace.rs:230`) and
`time_from_rf = t_echo + time_from_max_echo` (`readout.rs:64`). Static off-resonance is *refocused*
at the spin echo; it survives only as readout-time-dependent phase, which the existing
`phi = fmap * t(ky)` term already models as distortion. Adding `2*PI*fmap*TE` would double-count B0
and impose gradient-echo phase on a spin-echo sequence. The fieldmap contributes to distortion only.

Three terms, all independent of the fieldmap:

1. **Global phase** `phi_g`: a single controlled constant. Trivial but not invalid, and the handle
   for the global-rotation invariant test (4.1.5).
2. **Smooth pre-readout object phase**: a smooth low-order **3D** field (low-order polynomial in
   x, y, z), seeded per subject, constant across volumes, from which each slice's 2D field is
   sampled. Generating independent 2D fields per slice would create artificial discontinuities along
   z that no physical background phase has; one 3D field costs nothing more and avoids them.

   **The name is deliberate.** This term multiplies the object *before* Fourier encoding, and
   `F_trunc{f * exp(i*phi)}` is not `exp(i*phi) * F_trunc{f}` — so it must only stand in for phase
   sources that genuinely exist before finite Fourier encoding: transverse magnetization phase and,
   in single-coil simulations, a smooth receive phase.

   **Scope, stated narrowly:** the common pre-readout phase supplies a controlled complex object for
   Gibbs benchmarking. It is **not** a substitute for coil-specific complex sensitivities in
   multi-coil realism simulations — a genuine multi-coil model has
   `S_c(r) = |S_c(r)| * exp(i*theta_c(r))` with a *different* `theta_c` per coil, and one common
   `exp(i*phi(r))` multiplier cannot reproduce the relative coil phases that GRAPPA and coil
   combination depend on. That remains the deferred work in section 6.

   A phase convention or correction introduced by the scanner *after* reconstruction does **not**
   change which spatial frequencies were truncated — it rotates an already-reconstructed ringing
   pattern. If that is wanted, it belongs in a separately controlled **post-reconstruction phase
   transform**, not here. Consequently this term's parameters must not be calibrated against all
   observed combined-image phase and then claimed to have a one-to-one physical interpretation
   (4.2).

   Deliberately **not** modelled as per-coil phase. `coil_sensitivity` returns `f64` and the Roemer
   combine treats sensitivities as real (`kspace.rs:97`, `kspace.rs:320-335`). Giving coils complex
   sensitivities is a change to the coil and GRAPPA model, not to the phase model, and is not needed
   for a Gibbs benchmark. Recorded as a deferred follow-up in section 6; until then this term is
   named for what it is — a *pre-readout object phase*, not a coil phase.
3. **Diffusion-encoding phase**: a low-order random phase (constant + linear in x and y) drawn per
   volume and per slice-group, identically zero at b = 0 — gated exactly as the eddy term at
   `kspace.rs:177-178`. Motion during the diffusion gradients is the established source of
   substantial shot-to-shot DWI phase.

   **Modelled from the q-vector directly, not from `b` alone.** Draw a small random translation
   `dx` per shot and compute the phase offset as

   ```text
   phi_shot = q_eff . dx,      q_eff = c_q * sqrt(b) * bvec_unit
   ```

   **`q_eff` is an *effective* q-vector, not the physical one.** In PGSE
   `b ~ q^2 (Delta - delta/3)`, so `q ~ sqrt(b / (Delta - delta/3))` and `sqrt(b) * bvec` is
   proportional to `q` only under fixed, known timing. TRXScan has no `delta`, `Delta` or gradient
   waveform parameters anywhere in `signal.rs` or `compartments.rs`, so the calibrated constant `c_q`
   absorbs both the diffusion-timing convention and the radian/cycle convention. Naming it `q_eff`
   keeps the units interpretable; if TRXScan later gains waveform parameters this becomes a physical
   q-vector without changing the phase-model interface.

   **Interface.** `gradients[g]` as currently built is `bvec * bval` (`src/bin/trxscan.rs:268-273`),
   which is *not* the q-vector. The phase model therefore takes **`bval` and the unit `bvec`
   separately** rather than recovering them as `b = ||g||`, `ghat = g/||g||` — the separate interface
   makes the dimensional semantics explicit and avoids the degenerate `b = 0` case, where
   `g = [0,0,0]` leaves `ghat` undefined. (b = 0 is already gated to zero phase regardless.)

   This is preferred over the scalar `sigma_phi(b) = a * b^p` form because the direction information
   is already available, so the vector form costs nothing extra and yields three things the scalar
   form cannot: approximate `sqrt(b)` scaling under fixed timing, dependence on gradient
   *direction*, and sign reversal under reversed diffusion gradients. Rotational motion motivates the
   spatial linear phase terms.

   Linear-in-`b` is explicitly rejected: `phi = q . dx` with `b ~ q^2 (Delta - delta/3)` at fixed
   timing gives `phi ~ sqrt(b)`, so a linear model would grossly exaggerate phase at b = 5000
   relative to b = 1000. The scalar `a * b^p` form is retained only as a fallback knob for cases
   where no gradient vector is available.

**Sampling adequacy — no band-limit requirement.** The object and phase model are **not** required to
be band-limited, on either grid. Revision 1 required `exp(i*phi0)` to be band-limited to the
*acquired* matrix, which was backwards: frequencies above the acquired band are exactly what a finite
acquisition truncates, and suppressing them would make complex ringing artificially easy to remove.
Revision 3 then over-corrected into requiring negligible energy near the *simulation-grid* Nyquist,
which is also wrong: a sharp tissue boundary is deliberately non-band-limited and its coefficients
decay slowly, so that criterion would demand an artificially smooth object.

Adequacy is established empirically instead, by **convergence of the acquired-band Fourier
coefficients as the simulation resolution increases** (test 4.1.6). No band-limit condition is
imposed on the model anywhere.

Terms 2 and 3 are calibrated (4.2). Term 1 is **not** calibrated — a global phase is a gauge choice
and a test control, not an empirical quantity.

### 3.3 Sampling mask and noise (R3)

**Variance convention.** `noise_variance` is defined as **the per-component variance of the
reconstructed complex image under a fully sampled acquisition**:

```text
Var(Re n) = Var(Im n) = noise_variance        =>   E[|n|^2] = 2 * noise_variance
```

The per-k-space-sample, per-component variance is *derived* from it via the reconstruction
normalization: `sigma^2 = noise_variance / (nx*ny)`.

**Multi-coil semantics: `noise_variance` is the single-coil, pre-combination image variance.** The
final combined variance then emerges from the coil model rather than being imposed. This matches the
existing code: `kspace.rs:305-309` uses the same `sigma` for every coil, and the Roemer combine
(`kspace.rs:320-335`) forms `sum_c img_c * s_c / sum_c s_c^2`, so for independent per-coil noise of
per-component variance `V` the combined variance is

```text
Var_combined = V * sum_c s_c^2 / (sum_c s_c^2)^2 = V / sum_c s_c^2
```

i.e. today's `noise` argument already means per-coil pre-combination variance, not final-image
variance. Defining it that way preserves current behaviour and keeps the noise model local and
analytically transparent. It is also sufficient for the canonical Gibbs fixture, which uses a single
or uniform coil (3.5.3).

The factor of two is stated explicitly because it propagates into every SNR figure and calibration
constant. This component convention is chosen over `noise_variance = E[|n|^2]` because it is what
the code already implements — `kspace.rs:305-309` draws `sigma` independently into `re` and `im`,
giving per-component image variance `noise_variance` — so the CLI's `noise` argument keeps its
current meaning. The measured full-sampling figure of 1.0243 in section 1 is a *per-component* SD,
consistent with this definition.

Revision 2 stated the definition one way ("variance per acquired k-space sample") and then used the
equation for the other, which was self-contradictory. This revision picks the image-space definition
because it preserves the existing meaning of the CLI's `noise` argument (arg 11,
`src/bin/trxscan.rs`) — image SD is `sqrt(noise_variance)` at full sampling today — and because it is
the more directly interpretable knob. Every other scaling is derived, never asserted.

Build a per-slice `sampled: Vec<bool>` of length `nx*ny` from the partial-Fourier line skip and the
GRAPPA undersampling pattern (the two `continue` branches at `kspace.rs:198-212`), then add noise
**only** where `sampled[i]`.

Revision 1 additionally scaled the variance by the sampled fraction. That was a double-count. With
the convention above and an unnormalized inverse sum, image noise variance is
`(#sampled) * sigma^2 = f * noise_variance`, so **masking alone already yields SD proportional to
sqrt(f)** — consistent with the measured 1.0243 at f = 1. Multiplying the per-sample variance by `f`
as well would give SD proportional to `f`. The mask is the whole mechanism.

GRAPPA continues to overwrite its synthesized lines afterwards, so g-factor noise amplification and
correlation emerge from reconstructing noisy undersampled multi-coil data rather than from any
added factor.

### 3.4 Even-matrix window asymmetry (R4)

The acquired band is `[-ny/2, ny/2 - 1]`, asymmetric about k=0 by one sample. Real even-matrix
Cartesian acquisitions have exactly this asymmetry, so it is kept. The change is documentation plus
a test that pins the behaviour, converting an accident into a decision. Today it is the *only*
source of imaginary signal in the image (measured Im/Re 0.7-1.4%); after R2 it is a negligible
component of a physically motivated phase.

### 3.5 Benchmark reference outputs (goal-level)

Evaluating an unringing method requires something to score against, and a configuration in which the
score means what it claims to.

#### 3.5.1 The three images

Per volume, co-registered:

1. **`object-hires`** — the simulated complex object `f * exp(i*phi)` on the simulation grid, before
   any finite acquisition. The single source of truth.
2. **`object-nominal`** — `object-hires` reduced onto the acquisition grid by the **complex block
   mean** over each `o x o` cell: `(1/o^2) * sum_j z_j`, *not* a sum. The scoring target. Stated
   explicitly because a sum would scale intensities by `o^2` — a silent, catastrophic scale error.
3. **`acquired`** — the finite-band (optionally PF-masked, optionally noisy) reconstruction at the
   nominal matrix. What a de-Gibbs method consumes.

Renamed from revision 3's `truth-hires` / `truth-nominal`. Calling image 2 "truth" implied it is the
physically expected output of an ideal Fourier reconstruction, which it is not — it is box-integrated,
while any Fourier reconstruction is sinc-band-limited. Naming both object images consistently keeps
the distinction from `acquired` clear.

#### 3.5.2 `object-nominal` is derived, never regenerated

**Revision 3 claimed `object-nominal` was free by re-running Stage A at `o = 1`. That is wrong once
a spatial phase model exists.** For hires samples `z_j = f_j * exp(i*phi_j)` inside one nominal
voxel, the reference is `(1/N) * sum_j f_j * exp(i*phi_j)`, which carries intravoxel dephasing.
Re-running Stage A at `o = 1` yields a coarse amplitude with a single coarse phase,
`f_bar * exp(i*phi_coarse)`, which has no dephasing and is not the same quantity. The amplitude half
of the old claim was right; the phase half was not.

There is a second and arguably stronger reason: an independent `o = 1` run is a *different
realization*. Random diffusion-phase draws, fractional occupancies and tissue geometry would all
differ from the branch that produced `acquired`, so the two would not be comparable even if the
dephasing issue were solved.

Therefore: **take the complex block mean of the same array used for the acquisition.** The integer
simulation/acquisition ratio (3.1) makes this a cheap `o x o` average, and it guarantees both branches
share one realization by construction. Integrate the complex object, never the magnitude — averaging
`f * exp(i*phi)` over a voxel with varying phase loses signal, and that intravoxel dephasing is
physical and belongs in the reference.

#### 3.5.3 Gibbs benchmark mode

`object-nominal` is a *pre-acquisition* object. If `acquired` also carries EPI distortion, T2/T2*
readout evolution, eddy currents, Nyquist ghosts, GRAPPA or motion, then
`unringed - object-nominal` contains far more than unringing error — fieldmap displacement alone
would register as large edge-location error even for a perfect method.

The scoreable fixtures therefore run at one of two explicit levels:

**Canonical Gibbs benchmark** — isolates error attributable to finite Fourier truncation:

| Enabled | Disabled |
|---|---|
| oversampled object; complex phase; finite Fourier acquisition | EPI distortion |
| optional partial Fourier | eddy currents |
| optional k-space noise | Nyquist ghosts |
| optional reconstruction window | motion |
| single or uniform coil | T2/T2* readout evolution |
| | **GRAPPA / parallel acceleration** |

**Acquisition-realism benchmark** — restores GRAPPA and realistic multi-coil behaviour, plus any of
the above, as explicit experimental factors.

GRAPPA is disabled in the canonical fixture on principle, not on measurement. Section 2's finding
that the edge profile survives R = 2 to roughly four decimal places is an empirical observation on
one phantom; it does not make a separate reconstruction operator conceptually part of a Gibbs-only
condition, and with GRAPPA active its own reconstruction error contributes to the distance from
`object-nominal` independently of unringing. GRAPPA belongs in the suite as a factor, never silently
inside the baseline.

Robustness experiments may re-enable any of these, but they must **not** score raw distance to
`object-nominal` as though every error were a Gibbs error. Of the disabled effects, T2/T2* readout
decay is the natural first one to restore, because it acts as a k-space-domain filter — the same
class of thing as apodization — whereas distortion and motion are geometric and confound edge
location directly.

#### 3.5.4 Paired clean and noisy outputs

Emit `acquired-clean` and `acquired-noisy` from the same object and phase realization rather than a
single optionally-noisy image. The pair is nearly free and is what makes it possible to separate
residual Gibbs from noise amplification, and to study the interaction between denoising and
unringing — which is the central question for complex-aware methods.

#### 3.5.5 Scoring

No unringing method can be expected to reach `object-nominal` exactly: the acquired data genuinely
lacks those frequencies, so any un-ringed image is an inference, and an ideal Gibbs-remover returns a
band-limited estimate differing from the box-integrated object by the difference between a sinc and a
box PSF. The benchmark's value is the **decomposition** of the error, not a single distance:

| Metric | Distinguishes |
|---|---|
| Residual oscillatory error near edges | Gibbs actually suppressed |
| Edge-location bias | Geometric fidelity |
| Edge sharpness / effective resolution | A genuine unringer from a smoother |
| Complex, magnitude and phase error, reported separately | Whether phase survives the method |

**Phase error must be magnitude-masked or magnitude-weighted.** Phase is meaningless where magnitude
approaches zero, so raw phase difference must never be averaged over all voxels. Use
`dphi = arg(z_est * conj(z_ref))` with a threshold on `|z_ref|`, or weight the circular error by
`|z_ref|`.

A method that merely blurs will score well on oscillatory residual and badly on edge sharpness. That
separation is the point.

## 4. Testing

### 4.1 Analytic tests (fast, CI)

The reference is an analytic truncated-Fourier-series implementation of a step edge, built in
phase 0 and used as the oracle for everything below.

1. **`gibbs_profile_matches_analytic_reference_over_subvoxel_offsets`.** The single fixed-overshoot
   test from revision 1 was invalid. For a *correct* rectangular truncation at the nominal matrix,
   the sampled overshoot depends strongly on where the edge sits in the voxel — measured over 16
   offsets at N = 64:

   | offset | 0.000 | 0.125 | 0.250 | 0.375 | 0.500 | 0.625 | 0.750 | 0.875 | 0.938 |
   |---|---|---|---|---|---|---|---|---|---|
   | overshoot | +1.19% | +3.88% | +6.38% | +8.22% | **+8.93%** | +8.09% | +5.36% | +1.02% | +0.29% |

   The continuous Gibbs constant (+8.95%) is reached only near offset 0.5. A test asserting
   "overshoot in [8.0%, 10.0%]" would fail on correct code at most offsets. Replacement: compare the
   **entire 1D profile** to the analytic reference at each of 16 offsets. This is also the dependence
   Kellner's sub-voxel-shift method exploits, so getting it right is what makes the data a valid
   `mrdegibbs` benchmark.

   **Relating offsets to the oversampling factor.** 16 offsets cannot be represented by shifting a
   hard-valued edge on an `o = 2` grid — that grid admits only 2 distinct edge positions per acquired
   voxel. The test is therefore split three ways:

   | Tier | Setup | Asserts |
   |---|---|---|
   | Oracle | Continuous analytic step -> exact Fourier series | Reference itself is correct |
   | Implementation | `o = 16` or `32`, all 16 offsets | Simulator matches the oracle to tight tolerance |
   | Production | `o = 2, 4, 8`, convergence study | Quantifies the error `o = 2` actually introduces |

   Arbitrary offsets are represented on any grid by giving the boundary simulation voxel a
   **fractional occupancy** value — which the exact path-length rasterizer already produces for real
   anatomy, so this is the production code path, not a test-only fixture.

   **The default `o` is an output of this study, not an input.** Revision 2 assumed `o = 2` and added
   a convergence test. Instead, the convergence result *determines* whether `o = 2` remains the
   default. If the `o = 2` profile error is a significant fraction of the ringing being benchmarked,
   the default rises.
2. **`axis_aligned_step_rectangular_full_fourier_sidelobes_alternate_sign`.** The name encodes the
   scope deliberately: this invariant holds for a 1D axis-aligned step under a rectangular
   full-Fourier window, and is *not* a claim that all Gibbs ringing alternates voxel-by-voxel — that
   fails under arbitrary edge orientation, 2D interactions, PF and apodization. Verified robust to
   sub-voxel position: the sign pattern
   past the edge is `+-+-+-+-+-` at both offset 0.0 and 0.5, while the amplitude varies 30-fold over
   that same range. Sign alternation is exact for rectangular truncation, not "roughly two-voxel", so
   it is a sound invariant to pin independently of amplitude.
3. **`ringing_is_intrinsic`.** For a sub-voxel-positioned edge the round-trip is no longer exact;
   today's 3e-15 becomes order 1e-1.
4. **`ringing_residual_rotates_with_object_phase`** (consolidates revision 4's tests 4 and 5).
   Revision 4 asserted that ringing energy splits as `cos^2(phi)` / `sin^2(phi)` between the real and
   imaginary channels. **That test was incompatible with this design's own test 11.** It assumes the
   zero-phase ringing pattern is purely real, whereas the deliberately retained even-N asymmetry
   (3.4) gives `z_0 = a + i*b` with `b` non-zero — measured at `max|Im|/max|Re| = 1.5e-2` in section
   2. For `z_phi = exp(i*phi) * z_0` the real-channel energy is `|a cos(phi) - b sin(phi)|^2`, which
   departs from `cos^2(phi)` by roughly `2*(b/a)*tan(phi)` — about 3% at `phi = pi/4`, exceeding the
   2% tolerance the test itself specified. The test would have failed on correct code.

   Replacement, which is both stronger and compatible with the asymmetry. **The residual is defined
   explicitly**, so no implementer picks a different one:

   ```text
   R_phi = acquired_phi - object_nominal_phi        (noiseless canonical fixture, 3.5.3)
   assert  R_phi == exp(i*phi) * R_0                 elementwise, to tight tolerance
   ```

   evaluated at `phi = 0, pi/4, pi/2`.

   together with the object-level invariant that rotating the object by `exp(i*alpha)` rotates the
   reconstructed image by exactly `exp(i*alpha)` and leaves its magnitude unchanged. If a pedagogical
   `cos^2`/`sin^2` demonstration is wanted, it must use an odd-N or conjugate-symmetric fixture whose
   zero-phase PSF is genuinely real.
5. **`controlled_phase_fields_reconstruct_correctly`.** Deterministic cases: constant phase; a linear
   phase ramp; a low-order curved field; a per-volume diffusion-phase perturbation. Each compared to
   its analytic expectation.
6. **`acquired_band_converges_with_simulation_resolution`.** Revision 3 required the spectral energy
   of `f * exp(i*phi0)` near the simulation-grid Nyquist to be negligible. That criterion was wrong:
   a sharp tissue boundary is *intentionally* not band-limited and its Fourier coefficients decay
   slowly, so a perfectly good high-resolution representation of a discontinuous object retains
   appreciable energy near its own Nyquist. Demanding otherwise would require an artificially smooth
   object.

   What actually matters is whether the **acquired** coefficients have converged. For
   `o = 2, 4, 8, ...` compare the central `nx x ny` coefficients that constitute the acquisition:

   ```text
   || K_o - K_2o ||  /  || K_2o ||   <  tolerance,   within the acquired band
   ```

   This tests the numerical quantity the simulator must get right, imposes no artificial smoothness,
   and provides the basis for selecting the production oversampling factor (test 11).
7. **`noise_covariance_matches_sampling_mask`** — scoped explicitly, because the analytic identity
   holds only for zero-filled *linear* reconstruction of independent k-space noise:
   - **7a, GRAPPA disabled, `window = None` (PF and simple undersampling):** the reconstructed image
     noise autocovariance equals the inverse DFT of the mask `M` up to scale. Asserted analytically.
     With a window active the prediction becomes `F^-1{ |M(k) W(k)|^2 }` (3.1), which is a separate
     case rather than a violation.
   - **7b, GRAPPA enabled:** the simple mask-only identity no longer applies, but a closed form does.
     With fixed GRAPPA weights the whole reconstruction and coil combination is a linear operator
     `A`, so for input noise covariance `Sigma` the output is exactly

     ```text
     Sigma_out = A Sigma A^H
     ```

     Predict from that explicit operator and use Monte Carlo as independent validation. (Revision 5
     said "no closed form applies" and then asked for the operator prediction in the next sentence —
     self-contradictory; linearity is precisely what makes the operator form available.)

   Splitting these prevents a mathematically correct simple-mask identity from being applied to a
   stage where it does not hold. A lag-1 coefficient is used nowhere: it is not guaranteed monotonic
   across arbitrary masks and GRAPPA kernels.
8. **`noise_sd_scales_as_sqrt_sampled_fraction`.** Follows from the 3.3 convention with masking as the
   only mechanism; within 5%. **Scoped: GRAPPA disabled, `window = None`, identical per-sample
   thermal variance.** The scalar `sqrt(f)` law does not survive GRAPPA, where interpolation and
   g-factor effects change the noise variance, nor a reconstruction window; those cases belong to the
   operator-specific covariance test 7b, not to this scalar law.
9. **`partial_fourier_ringing`.** Dedicated 6/8 and 7/8 tests. PF changes the ringing structure —
    RPG (Lee et al., MRM 2021) exists precisely because PF ringing arises from two intervals and needs
    the sub-voxel shift applied twice. This is not an edge case here: `partial_fourier: 0.75` is the
    CLI default (`src/bin/trxscan.rs:254`), i.e. **the shipping configuration is 6/8 PF**, so a
    full-Fourier-only suite would validate a configuration nobody runs. Assert the asymmetric PF PSF
    against an analytic reference and its interaction with complex phase.
10. **`oversampling_converges`.** Revision 3 asserted that "the `o=4` profile sits within tolerance
    of the `o -> infinity` limit", which presupposed the answer. The test instead defines an
    acceptable error `epsilon` and reports

    ```text
    o_min = min { o : error(o) < epsilon }
    ```

    from the 4.1.6 convergence criterion and the 4.1.1 profile comparison. The **production default
    becomes `o_min`**, optionally with a safety margin. The test checks convergence; it does not
    prescribe in advance which oversampling level passes.
11. **`even_matrix_window_asymmetry_is_intentional`.** Pins R4.

Removed from revision 1: `gibbs_overshoot_matches_theory` (invalid, see test 1),
`phase_is_broadly_distributed` (gauge-dependent — mean absolute wrapped phase depends on an arbitrary
global phase origin, and the "no bin above 40%" threshold had no physical basis; a constant global
phase is a perfectly valid complex image), and the lag-1 monotonicity criterion (see test 8).

### 4.2 Real-data calibration (offline, not CI)

`scripts/calibrate_against_real.py`. **Three activities are kept separate**; revision 2 coupled the
first two by saying calibration "supplies constants the analytic tests assert against", which would
make mathematical correctness depend on empirical measurement:

| Activity | Gates CI? | Depends on real data? |
|---|---|---|
| **Correctness** — mathematical invariants and analytic references (4.1) | Yes | No |
| **Realism calibration** — fits parameters for a named preset, `PhaseModel::HBCDLike` | No | Yes |
| **Behavioural validation** — simulator vs. real under `mrdegibbs`, noise estimators, phase stats | No | Yes |

Calibrated presets are checked into the repository as data. They never determine whether the Fourier
implementation is correct. Real data offers no observable un-truncated ground truth, so it cannot
serve the correctness role at all.

Measured quantities are **circular/spatial**, not marginal-histogram: local phase-gradient RMS, phase
structure functions and correlation lengths, wrap density, and inter-volume phase differences after
removing a global phase offset.

**The two phase terms are calibrated to different standards, because term 2 is not identifiable.**

- **Term 3 (diffusion phase)** is *quantitatively fitted* where identifiable, subject to the
  phase-noise handling below. Inter-volume phase differences isolate it reasonably well.
- **Term 2 (pre-readout object phase)** is *tuned* so that the simulated final phase has realistic
  spatial scales and amplitudes, and is treated as an **effective benchmark parameter, not a
  recovered physical pre-readout distribution**. Reconstructed phase is an inseparable mixture of
  magnetization phase, coil phase and combination, scanner phase conventions and corrections, and
  possibly reconstruction filtering; only some of that precedes Fourier encoding (3.2). Fitting term
  2 to all observed combined-image phase and claiming a physical interpretation would be
  unidentifiable.

- **Noise structure** from background voxels: compare against the 4.1.8 mask prediction.
- **`mrdegibbs` round-trip**: DIPY `gibbs_removal` or MRtrix `mrdegibbs` on real and simulated data;
  compare variance removed and the spatial-frequency profile of (before - after). The highest-value
  behavioural check, because it tests the nominal-Nyquist cutoff using a tool that assumes it.
- **`dwidenoise` (MPPCA)**: compare estimated noise level and residual structure.

**Datasets (resolved).** Revision 1 named HBCD/HCP-D on assumption; the reviewer correctly noted the
public documentation does not evidence `part-phase` DWI there. Two verified complex sources are
available. **Both are valid for every measurement in this section** — they differ in protocol, not in
reconstruction class.

- **NIBS local dataset, HBCD protocol.** Matches the simulator's shipping configuration directly:
  6/8 partial Fourier (`partial_fourier: 0.75`, `src/bin/trxscan.rs:254`), multiband, GRAPPA. Use it
  for **behavioural comparison** of sampling-mask-dependent quantities — PF ringing structure
  (4.1.9) and noise plausibility.

  **It cannot supply theoretical mask constants.** These are reconstructed magnitude+phase images,
  so their noise covariance also reflects the scanner reconstruction, its GRAPPA kernel, coil
  combination, any k-space filtering, and interpolation — none of which are known. The analytic
  covariance of 4.1.8 must therefore be derived from the simulator's own explicit reconstruction
  operator; real data is a plausibility check on it, never its source. This is exactly the
  correctness/calibration/behavioural separation above.
- **ds006131, CS-DSI.** The compressed sensing in CS-DSI is applied in **q-space**: the dense
  Cartesian DSI grid is subsampled and sparsity is exploited when reconstructing the diffusion
  propagator, per voxel, downstream at the modelling stage. The individual diffusion-weighted volumes
  are conventional EPI images with a conventional linear reconstruction, so they are fully valid for
  noise-covariance plausibility, `mrdegibbs` round-trip, and phase measurements. Its DSI q-space
  scheme reaches higher b than the HBCD protocol, which makes it the **better** source for the
  b-dependence of the diffusion-phase term (3.2 term 3), where a wide b range is an asset.

  **Fitting the b-dependence must account for phase noise.** Magnitude SNR falls with b, so measured
  phase variance rises from thermal noise alone even if the underlying motion-induced phase does not
  change; fitting `Var(phi) ~ b^p` to raw wrapped phase would absorb that and bias `p` upward. The
  procedure must do at least one of: restrict the fit to sufficiently high-SNR voxels; model observed
  variance as signal-phase variance plus a noise-dependent term and fit both; or estimate phase after
  an appropriate complex denoising step. Whichever is chosen must be recorded with the fitted
  constants.

Revision 2 initially restricted ds006131 on the grounds that CS reconstruction is nonlinear. That was
an error — it conflated k-space CS (spatial undersampling with a nonlinear solver, which would break
linearity) with q-space CS (which does not touch image reconstruction). The restriction is withdrawn.

Division of labour: **NIBS/HBCD for behavioural comparison of sampling-mask-dependent effects;
ds006131 for the b-dependence of diffusion phase; either for background-phase spatial statistics.**
No theoretical constant is taken from reconstructed data. Acquisition parameters for both
(TE, resolution, PF, acceleration) are to be read from the data at the start of phase 8 rather than
assumed, since TE and resolution differ between a DSI protocol and HBCD and both feed the phase model.

## 5. Sequencing

Reordered from revision 1 to put the forward model first: the analytic oracle and the correct
acquisition model are prerequisites for testing everything else, and R1 is no longer optional
(see 5.1).

| Phase | Content | Depends on |
|---|---|---|
| 0 | Analytic 1D/2D reference implementation; fix Fourier conventions; sub-voxel edge test harness | — |
| 1 | **R1** oversample -> acquire nominal band -> reconstruct at target; `--oversample` in the prep script | 0 |
| 2 | **R2** controlled complex object phase (no fieldmap-derived term) | 0 |
| 3 | Full-Fourier complex Gibbs tests and benchmark fixtures | 1, 2 |
| 4 | Partial-Fourier sampling plus dedicated 6/8 and 7/8 tests | 3 |
| 5 | **R3** sampling-aware complex k-space noise with the explicit variance convention | 1 |
| 6 | **R4** even-matrix asymmetry documentation + pinning test | 1 |
| 7 | Optional reconstruction windows (Tukey/Hann/Fermi) | 3 |
| 8 | Calibration against a verified complex-DWI dataset; emit `PhaseModel::HBCDLike` | 2, 5 |
| 9 | Offline acceptance suite over real unringing methods (see 5.2) | 3, 4, 8 |
| 10 | *(optional)* FFT and/or z-slab streaming optimization | 1 |

Motion is **disabled** for phases 1-4. See section 6 for why, and phase 9 for where it returns.

### 5.1 R1 is not optional

Revision 1 offered phase 2 as a stopping point if R1's cost proved unattractive. That is withdrawn.
Stopping after the phase model would fix the degenerate complex channel but leave the incorrect
Gibbs-generation mechanism in place, which does not meet the stated goal of a valid complex Gibbs
benchmark. If R1's ~3.3x runtime or 4x memory is prohibitive, the answer is to **optimize it** —
z-slab streaming, FFTs, or direct evaluation from the native higher-resolution source (phase 10) —
not to retain the current truncation model.

The narrower claim that does survive: after phase 2 the complex channel becomes usable for purposes
that do not depend on the ringing forward model, such as complex denoising fixtures. That is a
statement about partial utility, not a stopping point.

### 5.2 Offline acceptance suite

Once the simulator's own tests pass, the question shifts from "is the simulator correct" to "is it a
useful benchmark". Phase 9 runs the methods this work exists to evaluate:

- conventional magnitude `mrdegibbs` / Kellner;
- RPG for partial Fourier;
- each complex-aware method under consideration;
- a no-unringing control.

across a grid of: full Fourier, 6/8 PF, 7/8 PF; constant phase, spatially varying phase,
diffusion-dependent phase; noiseless and noisy; with and without a reconstruction window.

**The acceptance criterion is not a predetermined ranking.** It is that varying phase, PF, noise and
windowing produces *interpretable and physically consistent* changes in method behaviour, scored with
the 3.5 error decomposition. That is what demonstrates TRXScan has become a benchmark rather than a
more elaborate simulator.

## 6. Risks

- **Stage A memory, 4x.** `comp.images` is `nvox * ngrad` per compartment; a 1.7 mm HBCD grid at
  ~100 volumes and 3 compartments is order 1.4 GB today, ~5.5 GB at `o=2`. Mitigation is z-slab
  streaming, which is natural because Stage B is already per-slice and z is not oversampled. Measured
  at the start of phase 1; built in phase 10 if it bites. Per 5.1, `--oversample 1` is a debugging
  aid, not a shipping configuration.
- **Runtime, ~3.3x** in the k-space stage. Acceptable given the stage is already parallel over
  volumes; phase 10 (FFT) is the answer if it is not.
- **Motion path: cost.** `apply_multiband_motion` and `generate_compartments_moving` operate on
  `comp.images` at the grid dims and inherit both the 4x memory and the 4x rasterization cost. No
  logic change needed; cost only.
- **Motion path: 3D rotation vs. in-plane-only oversampling.** In-plane oversampling is exactly right
  for a *stationary* 2D EPI acquisition. Under an arbitrary 3D head rotation applied before slice
  formation, through-plane structure rotates into the acquired in-plane directions; if the object
  stays coarse along z, the rotated slice is no longer an adequate sampling of the underlying
  anatomy.

  This splits cleanly by compartment. **Fibers are safe**: streamlines are transformed in *world
  space* and re-rasterized from continuous geometry each pose (`compartments.rs:621-628`), so the
  fiber compartment is exact at any rotation. **Tissue is not**: it goes through
  `resample_by_pose` (`compartments.rs:630-635`), discrete volume interpolation on the grid.

  Therefore: validate the Gibbs benchmark with motion disabled (phases 1-4); document that in-plane
  oversampling is sufficient for the stationary acquisition; add an offline rotation test in phase 9
  to measure whether z-resolution materially affects in-plane high-frequency content. If it does, the
  fix is targeted — resample tissue for each moved slice directly from the native 1 mm anatomy with
  the composed transform — not a global z-oversample with its memory penalty.
- **Phase model plausibility.** The 3.2 terms are structurally motivated but their amplitudes are
  guesses until phase 8, and unresolved if no complex dataset is reachable (4.2). Until then the
  phase channel is *structurally* correct (controlled spatial structure, volume-dependent diffusion
  phase) but not *quantitatively* calibrated, and must be described that way.
- **Deferred: complex coil sensitivities.** `coil_sensitivity -> f64` with a real Roemer combine
  (`kspace.rs:97`, `kspace.rs:320-335`). Real per-coil phase would make the multi-coil and GRAPPA
  simulation more faithful, but it is a coil-model change, not a phase-model change, and is out of
  scope here. Until it lands, 3.2 term 2 must be described as *pre-readout object phase* and never as
  coil phase.
- **Integer ratio.** An implementation restriction, not a mathematical one (see 3.1). The prep script
  enforces it for voxel-subdivision clarity and parity safety; relaxing it later is legitimate.

## 7. Decisions log

Regenerated from the current body at revision 5. Earlier revisions left rows describing superseded
choices; this table is rewritten wholesale whenever the body changes rather than patched.

| Decision | Choice | Rationale |
|---|---|---|
| R1 mechanism | Crop during the forward transform | 3.3x rather than 8x; keeps direct sums as oracle |
| FFT rewrite | Deferred to optional phase 10 | Preserves the comparison oracle during validation |
| Oversampling axes | In-plane only | Slice direction is not Fourier-encoded in 2D EPI |
| Oversample factor | `o_min = min{o : error(o) < eps}` from the convergence study | Tolerance-driven; rev. 3 still presupposed `o=4` would pass (4.1.10) |
| Finer object source | Native 1 mm anat maps via `--oversample` | Interpolating the 1.7 mm maps is a mathematical no-op |
| Integer sim/acq ratio | Implementation restriction, not a requirement | Voxel-subdivision clarity and parity safety (3.1) |
| `zero_ringing` | **Replaced by `window: KspaceWindow`**, default `None` | Tukey/Hann/Fermi are different PSFs; benchmark wants none |
| Static phase from fieldmap | **Removed** | Spin echo refocuses static off-resonance; would double-count B0 |
| Background phase source | Standalone low-order field, unrelated to the fieldmap | The fieldmap contributes to distortion only |
| Diffusion phase | `phi = q . dx` from a per-shot random translation | Gradient vectors are already plumbed in; gives sqrt(b) scaling, direction dependence and sign reversal free |
| Background phase field | One low-order **3D** field, sliced; named *pre-readout object phase* | Independent 2D fields create z discontinuities; only pre-encoding sources belong here |
| Post-reconstruction phase | Separate transform if wanted; not folded into term 2 | It rotates an already-reconstructed pattern; it does not change what was truncated |
| q-vector interface | `bval` and unit `bvec` passed separately | `gradients[g] = bvec*bval` is not q; avoids the degenerate `b = 0` case |
| q naming | `q_eff = c_q*sqrt(b)*bvec`, an *effective* q-vector | No `delta`/`Delta` anywhere in the codebase; `c_q` absorbs timing and angle conventions |
| Noise, multi-coil | Single-coil pre-combination variance; combined emerges from the coil model | Matches the existing Roemer path, where combined variance is `V / sum_c s_c^2` |
| Window vs. GRAPPA | After GRAPPA, before the inverse DFT | A reconstruction window acts on synthesized lines too; one deterministic location |
| GRAPPA covariance | `Sigma_out = A Sigma A^H` from the explicit linear operator | Fixed weights make the reconstruction linear, so a closed form does exist |
| `object-nominal` reduction | Complex block **mean**, not sum | A sum would scale intensities by `o^2` |
| Term 2 calibration | Tuned as an effective benchmark parameter, not fitted | Combined-image phase is an inseparable mixture; term 2 is not identifiable |
| Window vs. noise order | `W(k) * [K_signal + n]` — window filters both | In-situ replacement would recreate the filtered-signal/unfiltered-noise defect |
| GRAPPA in canonical fixture | **Disabled**; an explicit factor in the realism fixture | A separate reconstruction operator should not sit silently inside a Gibbs-only baseline |
| Re/Im ringing test | Complex rotation of the residual, not `cos^2`/`sin^2` | The retained even-N asymmetry makes the zero-phase pattern non-real |
| Global phase | Not calibrated | A gauge choice and test control, not an empirical quantity |
| Band-limit requirements | **None, on either grid** | Adequacy is established by acquired-band coefficient convergence (4.1.7) |
| Coil phase | Deferred; term named background/reconstruction phase | Coils are real-valued today; complex sensitivities are a coil-model change |
| Noise variance | **Per-component** image variance at full sampling; `E[\|n\|^2] = 2*noise_variance` | Matches `kspace.rs:305-309`, so the CLI `noise` argument keeps its meaning |
| Noise covariance test | Analytic pre-GRAPPA; operator/Monte-Carlo with GRAPPA | The IDFT-of-mask identity does not survive GRAPPA or coil combine |
| Overshoot test | Sub-voxel edge family vs. analytic profile, three tiers | A correct implementation spans +0.29%..+8.93% by offset alone |
| Sign-alternation test | Kept, renamed to encode its scope | Exact for a 1D axis-aligned step under a rectangular window only |
| Phase histogram test | **Removed** | Gauge-dependent; a constant global phase is a valid complex image |
| Partial Fourier | First-class, with 6/8 and 7/8 tests | 6/8 is the shipping default (`src/bin/trxscan.rs:254`) |
| Nominal reference | `object-nominal` = block-reduction of the **same** `object-hires` array | Regenerating at `o=1` loses intravoxel dephasing and is a different realization (3.5.2) |
| Reference naming | `object-hires` / `object-nominal` / `acquired` | "truth-nominal" implied an ideal Fourier result; it is box-integrated |
| Benchmark configuration | Explicit Gibbs mode: distortion, eddy, ghosts, motion, T2* off | Otherwise `unringed - reference` is not a Gibbs error (3.5.3) |
| Acquired outputs | Paired `acquired-clean` and `acquired-noisy` | Separates residual Gibbs from noise amplification (3.5.4) |
| Phase-error scoring | Magnitude-masked or magnitude-weighted circular error | Phase is meaningless where magnitude approaches zero (3.5.5) |
| Scoring | Error *decomposition*, not a single distance | No unringer can reach `object-nominal`; separates recovery from smoothing |
| Correctness gate | Analytic/synthetic only | Real data has no observable un-truncated ground truth |
| Calibration | Produces named presets; never gates CI | Keeps mathematical correctness independent of empirical fits |
| Calibration sources | NIBS/HBCD for behavioural comparison; ds006131 for b-dependence | Reconstructed data cannot supply theoretical mask constants (4.2) |
| Fitting `p` | Must model or exclude thermal phase noise | Magnitude SNR falls with b; naive fits bias `p` upward |
| Motion during phases 1-4 | Disabled | 3D rotation vs. in-plane-only oversampling is unresolved until phase 9 |
| R1 optional? | **No** | Stopping after R2 leaves the wrong Gibbs mechanism (5.1) |
| z-slab streaming | Deferred to phase 10, measured in phase 1 | YAGNI until the measurement says otherwise |

## 8. Disposition of the external review

Source: `reviews/chatgpt-gibbs-ringing-review.md`. Verification notes are in the session transcript; the two
numeric checks are reproduced in 4.1.1 and 3.3.

| # | Point | Disposition |
|---|---|---|
| 1 | `2*PI*fmap*TE` inappropriate for spin echo | **Accepted.** Term removed (3.2). Spin-echo confirmed at `kspace.rs:230`, `readout.rs:64`. |
| 2 | Coil phase should be per-coil | **Partially accepted.** Term renamed to background/reconstruction phase; complex coil sensitivities deferred as a coil-model change (3.2, section 6). |
| 3 | Phase band-limit requirement is backwards | **Accepted.** Inverted to a simulation-grid sampling-adequacy condition (3.2). |
| 4 | Noise scaling double-counts | **Accepted.** Verified analytically; variance convention fixed, mask is the only mechanism (3.3). |
| 5 | Lag-1 autocorrelation not a valid CI criterion | **Accepted and sharpened.** Replaced by an exact assertion that noise autocovariance equals the IDFT of the sampling mask (4.1.7). |
| 6 | Replace fixed 9% test with a sub-voxel edge family | **Accepted.** Verified numerically: a correct implementation spans +0.29% to +8.93% across offsets (4.1.1). |
| 7 | R1 is not optional | **Accepted.** Stopping point withdrawn (5.1); phases reordered R1-first. |
| 8 | Dedicated partial-Fourier tests | **Accepted and strengthened.** 6/8 is the shipping default, so this is the primary case, not an edge case (4.1.9). |
| — | Integer ratio not mathematically required | **Accepted.** Restated as an implementation restriction with its actual justification (3.1). |
| — | "No blur" is wrong | **Accepted.** Restated as the correct nominal acquisition PSF (3.1). |
| — | Soften "real MRI always has Gibbs" | **Accepted** for the assessment document; scanner filtering and non-step edges both moderate it. The finding that an exact DFT identity yields *zero* ringing is unaffected. |
| — | Phase histogram / Im-Re-order-1 not validity criteria | **Accepted.** Tests removed, replaced by controlled fields and the global-rotation invariant (4.1.5-6). |
| — | Apodization: separate modes, named windows | **Accepted** (3.1). |
| — | HBCD/HCP-D may not release `part-phase` | **Accepted and resolved.** The reviewer was right that revision 1 assumed rather than verified. Replaced with two named sources, NIBS/HBCD and ds006131 CS-DSI, both valid. A revision-2 restriction on ds006131 (claiming CS implies nonlinear image reconstruction) was itself wrong and has been withdrawn: CS-DSI's compressed sensing is in q-space, not k-space. See 4.2. |
| — | "Period is roughly two voxels" | **Partially accepted.** The full-profile test is better, but sign alternation is *exact* and robust across sub-voxel offsets (verified `+-+-+-+-+-` at offsets 0.0 and 0.5 while amplitude varies 30-fold), so it is kept as an independent invariant (4.1.2) rather than softened. |

## 9. Disposition of the second review round

Source: `reviews/chatgpt-gibbs-ringing-review-round-2.md`. Verdict received: "architecture approved;
specification cleanup and validation details needed before implementation." All seven blockers are
accepted.

| # | Point | Disposition |
|---|---|---|
| 1 | Noise-variance definition contradicts its own equation | **Accepted.** Resolved to the image-space definition, per-sample derived, with the CLI-compatibility rationale (3.3). |
| 2 | Assessment still carries revision-1 recommendations | **Accepted.** `gibbs-ringing-assessment.md` section 3 regenerated from this design; section 2.2 opening and 2.4 phase claims corrected. |
| 3 | Stale decision-log rows and phase cross-references | **Accepted.** Log regenerated wholesale (section 7); "phase-7 calibration" corrected to phase 8. |
| 4 | 16 sub-voxel offsets cannot be represented at `o=2` | **Accepted.** Three-tier structure (oracle / high-`o` implementation / production convergence), fractional-occupancy representation, and `o` default demoted to an output of the convergence study (4.1.1). |
| 5 | Covariance identity does not survive GRAPPA | **Accepted.** Split into 8a analytic pre-GRAPPA and 8b operator/Monte-Carlo (4.1.7). |
| 6 | Do not hardcode diffusion phase linear in b | **Accepted.** `sigma_phi(b) = a*b^p`, default `p = 0.5`, derived from `phi = q.dx` with `b ~ q^2` at fixed timing (3.2). |
| 7 | Define the Gibbs-free nominal-resolution benchmark truth | **Accepted and promoted to a goal.** New section 3.5, three outputs, plus the caveat that no unringer can reach the nominal reference, so scoring is an error decomposition. *(Revision 3 additionally claimed the nominal reference was free from Stage A at `o=1`; that claim was refuted in round 3 and is superseded by 3.5.2. The names used here were `truth-hires`/`truth-nominal`, since renamed.)* |
| — | Global phase needs no calibration | **Accepted** (3.2). |
| — | Rename the sign-alternation test to encode scope | **Accepted** (4.1.2). |
| — | Problem table still asserts "Im/Re order 1", "full range, wrapped" | **Accepted.** Replaced with requirement-framed rows (section 1). |
| — | Add an offline acceptance suite over real unringing methods | **Accepted.** New phase 9 and section 5.2. |
| — | 3D rotation vs. in-plane-only oversampling | **Accepted and sharpened.** Fibers are safe (world-space transform + re-rasterization, `compartments.rs:621-628`); only tissue degrades (`resample_by_pose`, `compartments.rs:630-635`), so the fix is targeted rather than a global z-oversample (section 6). |
| — | Separate correctness from calibration | **Accepted.** Three-way split; calibration no longer supplies constants that correctness tests assert against (4.2). |

## 10. Disposition of the third review round

Source: `reviews/chatgpt-gibbs-ringing-review-round-3.md`. Verdict received: "architecture approved, but I
would not freeze the spec yet." All six pre-implementation items are accepted, as are the five
follow-on items, four of which are adopted now rather than deferred because they cost nothing extra.

| # | Point | Disposition |
|---|---|---|
| 1 | `object-nominal` is not free from Stage A at `o=1` | **Accepted — this was a real error.** Once phase varies within a voxel, `(1/N) sum f_j exp(i phi_j)` is not `f_bar exp(i phi_coarse)`. Now derived by block-reducing the same `object-hires` array, which also guarantees a shared realization (3.5.2). The amplitude half of the old claim was right; the phase half was not. Assessment R5 corrected to match. |
| 2 | Scoring confounded by distortion/eddy/ghost/motion | **Accepted.** Explicit Gibbs benchmark mode (3.5.3), with a note that T2/T2* is the natural first effect to restore since it is a k-space filter rather than a geometric distortion. |
| 3 | Rename `truth-nominal` | **Accepted, extended.** Renamed the whole triple to `object-hires` / `object-nominal` / `acquired` so both object images are named consistently. |
| 4 | Top-of-sim-band energy is a bad criterion | **Accepted.** A sharp boundary is intentionally not band-limited. Replaced by acquired-band convergence `\|K_o - K_2o\| / \|K_2o\|` (4.1.6). |
| 5 | Test 11 presupposes `o=4` passes | **Accepted.** Now tolerance-driven: `o_min = min{o : error(o) < eps}` (4.1.10). |
| 6 | Full-Fourier vs 6/8 PF default contradiction | **Accepted.** `KspaceWindow::None` is the default window; 6/8 PF remains the shipping acquisition default; full-Fourier is a benchmark fixture setting `partial_fourier = 1.0` (3.1). |
| 7 | Complex variance ambiguous by a factor of 2 | **Accepted, with a different convention than suggested.** Pinned to the *component* convention — `Var(Re) = Var(Im) = noise_variance`, `E[\|n\|^2] = 2*noise_variance` — because `kspace.rs:305-309` draws `sigma` into each component independently. Adopting `noise_variance = E[\|n\|^2]` would have silently halved the CLI argument's meaning, contradicting the stated rationale for the image-space definition (3.3). |
| 8 | q-vector-aware diffusion phase | **Accepted now, not deferred.** `gradients[g]` is already plumbed into `simulate_slice` (`src/bin/trxscan.rs:268-273`), so `phi = q . dx` costs nothing beyond the scalar form and subsumes it (3.2). |
| 9 | 3D-smooth background phase | **Accepted now.** One low-order 3D field, sliced; avoids artificial z discontinuities at no extra cost (3.2). |
| 10 | SNR-aware fitting of the b-dependence | **Accepted.** Three permitted procedures, with the chosen one recorded alongside the constants (4.2). |
| 11 | Reconstructed data cannot supply mask covariance constants | **Accepted.** NIBS demoted to behavioural comparison; the analytic covariance derives from the simulator's own reconstruction operator (4.2). |
| 12 | Magnitude-masked phase scoring | **Accepted** (3.5.5). |
| 13 | Paired clean/noisy outputs | **Accepted** (3.5.4). |
| 14 | Document leftovers (status line, phase numbers, goal wording) | **Accepted.** Status now revision 4; phase references corrected to 10 for optimization; the goal restated as the analytic profile rather than "period 2 / ~8.95%". |

## 11. Disposition of the fourth review round

Source: `reviews/chatgpt-gibbs-ringing-review-round-4.md`. Verdict received: "core forward-model architecture
approved"; four items flagged as pre-freeze. All ten are accepted.

| # | Point | Disposition |
|---|---|---|
| 1 | Stale simulation-Nyquist energy criterion in 3.2 | **Accepted.** 3.2 rewritten to impose no band-limit on either grid; adequacy is 4.1.6 convergence alone. Decision-log row replaced. Flagged as important because the stale text could have led an implementer to reinstate the rejected test. |
| 2 | `cos^2`/`sin^2` test conflicts with the even-N asymmetry | **Accepted — the test would have failed on correct code.** With `b/a = 1.5e-2` measured in section 2, the real-channel energy departs from `cos^2(phi)` by about 3% at `phi = pi/4`, against the test's own 2% tolerance. Replaced by an exact complex-rotation assertion on the ringing residual, consolidating revision 4's tests 4 and 5 (4.1.4). |
| 3 | Window must filter noise as well as signal | **Accepted.** Operator order pinned to `W(k)*[K_signal + n]` (3.1), explicitly because in-situ replacement of the `zero_ringing` code would have recreated the original filtered-signal/unfiltered-noise defect. Covariance prediction under a window stated. |
| 4 | Canonical Gibbs fixture must decide GRAPPA | **Accepted.** Split into canonical (GRAPPA off, simple coil) and acquisition-realism levels (3.5.3). Disabled on principle: the measured survival at R = 2 is one phantom's empirical result and does not make a separate reconstruction operator part of a Gibbs-only condition. |
| 5 | Scope test 9 away from GRAPPA and windowing | **Accepted** (4.1.8); 8a likewise scoped to `window = None` (4.1.7). |
| 6 | Background phase mixes pre- and post-encoding sources | **Accepted.** Renamed *smooth pre-readout object phase*, restricted to pre-encoding sources, with a separate post-reconstruction transform if wanted (3.2). Noted that this term is the scalar approximation of the deferred complex coil sensitivities rather than an independent effect. |
| 7 | q-vector interface must not assume `gradients[g]` is q | **Accepted.** `bval` and unit `bvec` passed separately, which also avoids the degenerate `b = 0` case where `ghat` is undefined (3.2). |
| 8 | Assessment carries stale revision-3 statements | **Accepted, with a structural fix.** This is the second round in which the assessment drifted behind the spec. Its recommendation section is reduced to a pointer table indexed to spec sections rather than a prose summary that must be kept in sync. Section 2.4's "not physics" reworded. |
| 9 | "Division of labour" contradicts the NIBS paragraph above it | **Accepted** (4.2). |
| 10 | Stale `Default o = 2` in 3.1 | **Accepted** (3.1); `o = 2` retained only as the concrete example in the cost and memory figures. |

## 12. Disposition of the fifth review round

Source: `reviews/chatgpt-gibbs-ringing-review-round-5.md`. Verdict received: architecture converged, status
"frozen" agreed, remaining items are implementation semantics and wording. All ten accepted; none
reopened the architecture.

| # | Point | Disposition |
|---|---|---|
| 1 | `sqrt(b)*bvec` is not the physical q-vector | **Accepted.** Renamed `q_eff` with `c_q` absorbing diffusion timing and angle convention. Verified there are no `delta`/`Delta`/waveform parameters in `signal.rs` or `compartments.rs`, so the effective form is the only honest one (3.2). |
| 2 | `noise_variance` ambiguous once multiple coils are enabled | **Accepted, Option A.** Verified against the code: the same `sigma` is used for every coil (`kspace.rs:305-309`) and the Roemer combine gives `Var_combined = V / sum_c s_c^2` (`kspace.rs:320-335`), so today's argument already means single-coil pre-combination variance. Option A both preserves behaviour and keeps the model local (3.3). |
| 3 | Window position relative to GRAPPA left unchosen | **Accepted.** Settled as `... -> GRAPPA -> W(k) -> inverse DFT`, with pre-GRAPPA scanner filtering deferred to a separate explicit mode (3.1). |
| 4 | "No closed form" for GRAPPA covariance is self-contradictory | **Accepted.** Fixed weights make the reconstruction linear, so `Sigma_out = A Sigma A^H` is exact; what fails is only the mask-only identity (4.1.7b). |
| 5 | Stale "combined-image / reconstruction phase" wording survived | **Accepted.** Removed from 3.2 and from the section 6 deferral note; *pre-readout object phase* used throughout. |
| 6 | A common object phase is not multi-coil receive phase | **Accepted.** Scope narrowed explicitly: one `exp(i*phi(r))` cannot reproduce per-coil `theta_c(r)`, which GRAPPA and coil combination depend on (3.2). |
| 7 | Term 2 is not identifiable from reconstructed phase | **Accepted.** 4.2 now calibrates term 3 quantitatively and *tunes* term 2 as an effective benchmark parameter, matching the language already in 3.2. |
| 8 | `object-nominal` must be a block mean, not a sum | **Accepted.** Stated as `(1/o^2) * sum_j z_j` in both places, with the `o^2` scale-error hazard called out (3.5.1, 3.5.2). |
| 9 | Test numbering jumps; residual undefined | **Accepted.** Renumbered 1-11 with all cross-references updated; the residual is now defined as `R_phi = acquired_phi - object_nominal_phi` in the noiseless canonical fixture (4.1.4). |
| 10 | Decision-log preamble, "Always", amplitude wording | **Accepted.** Preamble now says revision 5 and states the table is rewritten wholesale; "Always" replaced with the sinc-PSF formulation; the opening restated around the cutoff rather than amplitude. |
