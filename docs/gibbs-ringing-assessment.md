# TRXScan: Gibbs ringing realism assessment

**Question:** Does TRXScan simulate Gibbs ringing in a realistic manner that would reflect
properly in complex-valued simulated data?

> **Revision 2 (2026-08-31):** corrected after external review. Section 2.2's amplitude argument was
> overstated — see the sub-voxel caveat there. The load-bearing criticism is the cutoff frequency,
> not the overshoot amplitude. Section 2.1's "no real MRI is" is softened. All measurements stand;
> two inferences drawn from them did not.

**Verdict:** The ringing *mechanism* is physically correct, but the artifact it produces is not
realistic at the nominal resolution, and it does **not** reflect properly in the complex-valued
output — the reconstructed image is real-valued to machine precision, so all ringing lands in the
real channel and the `part-phase` output is degenerate.

Assessed against commit as cloned in `TRXScan/`. All numbers below were measured by building an
isolated copy of the crate and driving `kspace::simulate_slice` directly (64x64, single coil,
`do_distortions`/`do_relaxation` off unless stated).

---

## 1. What the implementation does

`TRXScan/src/kspace.rs:266-278` — a hard boxcar zeroing of the outer `zero_ringing`% of k-space in
both in-plane directions:

```rust
if acq.zero_ringing > 0.0 {
    let rx = ((nx as f64 / 2.0) * acq.zero_ringing / 100.0).ceil() as usize;
    let ry = ((ny as f64 / 2.0) * acq.zero_ringing / 100.0).ceil() as usize;
    // zero kx < rx || ky < ry || kx + rx >= nx || ky + ry >= ny
}
```

Position in the chain: inside the per-coil loop, **after** the distortion/T2*/eddy-modulated
forward DFT, **before** spikes, noise, GRAPPA and the inverse DFT. The CLI ships
`zero_ringing: 6.0` (`src/bin/trxscan.rs:122`) and writes BIDS `part-mag` + `part-phase`
(`src/bin/trxscan.rs:143`). This mirrors Fiberfox's `itkKspaceImageFilter.cpp:334`.

### What is right

- Truncation is the correct physical **cause** of Gibbs ringing — not a post-hoc convolution or
  an image-domain filter.
- The ordering in the acquisition chain is correct: ringing is applied to the already-distorted,
  relaxed, coil-weighted k-space, per coil, so the ringing follows the EPI geometric distortion
  and combines coherently across coils.
- No apodization (pure boxcar) — worst-case ringing, the conservative choice.
- 2D only, which is correct for 2D EPI (no through-slice Fourier encoding, so no partition-
  direction ringing).
- Deterministic and identical across slices and volumes.
- Survives GRAPPA: at R=2 / 8 coils / 16 ACS lines the edge profile is identical to the
  fully-sampled truncated profile to ~4 decimal places. GRAPPA does not meaningfully re-fill the
  zeroed high-k lines.

---

## 2. Findings

### 2.1 There is no intrinsic Gibbs ringing

Round-trip error with `zero_ringing: 0` is **3.0e-15** (max |recon.re - object|). The object is
defined on the reconstruction grid, so the forward DFT followed by the inverse DFT is an exact
identity. There is no continuous object being finitely sampled, so no ringing can arise on its own.

Consequence: all Gibbs must be hand-injected, and `Acquisition::default()` (`zero_ringing: 0.0`)
produces *exactly* Gibbs-free images. Real acquisitions vary in how much ringing they show — scanner
k-space filtering suppresses it, and real tissue edges are not perfect step functions, so "real MRI
always has strong Gibbs" would be too strong a claim. But zero ringing from an exact DFT identity is
not a filtered acquisition; it is the absence of a finite-acquisition model. Simulated data generated
with the defaults behaves as if it were already de-Gibbsed.

### 2.2 The Fourier cutoff is not at the nominal acquisition bandwidth

The continuous rectangularly-truncated step response approaches the ~8.95% Gibbs overshoot, but the
amplitudes observed at discrete voxel centres depend strongly on sub-voxel edge position — see the
caveat below. The invariant that matters here is different and simpler: **for a real acquisition the
Fourier cutoff sits at the nominal acquisition bandwidth**, and the sidelobes of a 1D axis-aligned
step under that rectangular window alternate sign voxel by voxel. Zeroing r% of k-space moves the
cutoff, giving a ripple period of `2 / (1 - r/100)` voxels instead.

Measured on a grid-aligned step edge (64x64):

| `zero_ringing` | lines cut/side | kept matrix | overshoot | ripple period | eff. voxel |
|---:|---:|---:|---:|---:|---:|
| 1%    | 1  | 62/64 | +1.55%  | 2.06 px | 1.03x |
| 3%    | 1  | 62/64 | +1.55%  | 2.06 px | 1.03x |
| **6% (CLI default)** | **2** | **60/64** | **+3.09%** | **2.13 px** | **1.07x** |
| 10%   | 4  | 56/64 | +5.95%  | 2.29 px | 1.14x |
| 12.5% | 4  | 56/64 | +5.95%  | 2.29 px | 1.14x |
| 25%   | 8  | 48/64 | +10.04% | 2.67 px | 1.33x |
| 50%   | 16 | 32/64 | +6.88%  | 4.00 px | 2.00x |

**Sub-voxel caveat (added in revision 2).** The overshoot column above cannot on its own establish
that the amplitude is wrong. For a *correct* rectangular truncation at the nominal matrix, the
sampled overshoot depends strongly on where the edge falls within a voxel — computed from the
analytic truncated Fourier series at N = 64, it ranges from **+0.29% to +8.93%** across sub-voxel
offsets, reaching the 8.95% Gibbs constant only near offset 0.5. That dependence is exactly what
Kellner's sub-voxel-shift method exploits. So a measured 1.55% is not by itself evidence of a defect.

What the table *does* establish is points 1 and 3 below. The load-bearing criticism is **the cutoff
frequency**: zeroing moves the Fourier cutoff off the nominal acquisition Nyquist, changing the PSF
and the spatial scale of the ringing relative to the voxel grid. That is unambiguously wrong for a
benchmark, independent of any amplitude argument.

Three problems visible here:

1. **Wrong period.** At the CLI default the ripple period is 2.13 px, not 2.0. At 25% it is
   2.67 px. Sub-voxel-shift de-Gibbs methods (Kellner et al. — MRtrix `mrdegibbs`, DIPY
   `gibbs_removal`) assume truncation at the image matrix, i.e. period-2 ringing. They will not
   cleanly remove ringing at these periods, so the data is not a valid benchmark for them.
2. **Non-monotonic amplitude.** Overshoot ranges 1.55%-10.04% and *decreases* from 25% to 50%,
   because the ripple period beats against the voxel grid. Note the caveat above: a correct
   implementation also varies with sub-voxel edge position, so non-monotonicity in the *parameter*
   is the anomaly here, not the range itself. The parameter should not be a knob whose effect
   reverses.
3. **Parameter quantization.** `ceil()` makes 1% and 3% produce byte-identical output, as do 10%
   and 12.5%. The knob is much coarser than it appears.

### 2.3 It is a resolution loss as well as ringing

Because the reconstruction grid stays at the nominal matrix while the sampled extent shrinks, the
image is blurred in addition to ringing: effective voxel 1.07x nominal at the 6% default, 1.33x at
25%. The voxel size written into the NIfTI header therefore overstates the true resolution of the
simulated data.

### 2.4 The complex-valued output is degenerate (the core issue)

The compartment signal is assembled as a **real, non-negative** sum (`src/kspace.rs:230-232`), and
the only phase terms in the model — the fieldmap and the eddy-current polynomial — are applied as
`phi = fmap * t(ky)` (`src/kspace.rs:236-244`), i.e. **ky-dependent**. A ky-dependent phase warps
geometry along the phase-encode axis; it does not imprint a static image-domain phase. There is no
object phase model anywhere in the pipeline.

Measured:

| condition | max\|Im\| / max\|Re\| |
|---|---:|
| `zero_ringing: 0`, real off-centre object | **4.1e-16** |
| `zero_ringing: 6`  | 6.9e-3 |
| `zero_ringing: 25` | 1.5e-2 |

The small non-zero residual with ringing on **is not realistic object phase**; it is a deterministic
consequence of the even-N Fourier sampling convention — the one-sample asymmetry of the truncation
window (kept kx range is [-24, +23] about k=0, not [-24, +24]). That asymmetry is present in real
even-matrix Cartesian acquisitions and is deliberately retained (see R4); what it is not is a
substitute for an object phase model.

With the full artifact chain enabled (distortion + relaxation + eddy + Nyquist ghost + ringing,
with a realistic linear fieldmap), the phase inside the object has **mean |phase| = 0.19 rad**, and
96% of bright voxels fall into 2 of 8 phase bins spanning -pi..pi.

**Consequences:**

- Gibbs ringing appears essentially only in the **real** channel; the imaginary channel is ~zero.
- In real DWI the background phase (B0/shim/coil, plus shot-to-shot diffusion motion phase) is
  non-trivial and spatially varying, so ringing lobes rotate through the complex plane and split
  across Re/Im differently across the image. (The requirement is *controlled, spatially varying
  phase* — not any particular phase histogram. A constant global phase is a perfectly valid complex
  image; what makes the current output unusable is that there is effectively no object phase at all.)
- Any consumer of `part-phase` or the complex pair sees the wrong thing: complex NORDIC / MPPCA,
  phase-based de-Gibbs, phase-corrected real averaging, background-phase removal, phase unwrapping.
- The **magnitude** channel is the only defensible output of the current model.

One exception worth noting: zero-filled partial Fourier is the single mechanism in the code that
produces meaningful imaginary signal (measured Im/Re = 0.16 at `partial_fourier: 0.6`). There is no
homodyne or POCS reconstruction, so that is a reconstruction artifact, not object phase.

### 2.5 Noise is not band-limited to the truncated k-space

Noise is added **after** the ringing zeroing and after the partial-Fourier line skipping
(`src/kspace.rs:300-307`), so k-space samples that were never acquired are filled with pure noise.

Measured (noise-only object, `noise_variance: 1.0`):

| condition | image noise SD | lag-1 autocorrelation |
|---|---:|---:|
| `zero_ringing: 0`  | 1.0243 | -0.047 (white) |
| `zero_ringing: 50` | 1.0243 | -0.047 (white) |
| `partial_fourier: 1.0` | 1.0141 | — |
| `partial_fourier: 0.6` | 1.0141 | — |

Physically, a truncated acquisition carries no noise beyond the sampled band, so image noise should
be **spatially correlated** (smoothed by the same PSF as the signal) and total variance should drop
with the sampled fraction. `src/noise.rs:5` even documents that Fiberfox scales noise variance by
partial Fourier — TRXScan does not.

The resulting data has band-limited signal combined with white full-Nyquist noise, a combination
that cannot occur in real acquisitions, and precisely the mismatch that noise-level estimators
(MPPCA/NORDIC) and Gibbs-correction methods key on.

### 2.6 Grid-aligned edges

The `raster` stage produces partial-volume edges, so there is *some* sub-voxel edge positioning.
But the object is still discrete on the recon grid, so the ringing phase relative to the voxel grid
is not driven by true sub-voxel edge placement the way it is in real acquisitions.

### 2.7 The existing test does not constrain any of this

`gibbs_ringing_changes_the_image` (`src/kspace.rs:714-724`) asserts only that the mean absolute
image difference exceeds 1e-3. It checks neither ripple period, nor overshoot amplitude, nor the
complex channels — so every issue above passes the test suite.

---

## 3. Recommendations

The design spec is authoritative: `superpowers/specs/2026-08-31-gibbs-ringing-realism-design.md`.

This section is deliberately a **pointer index, not a summary**. Earlier revisions restated the
design here in prose and drifted behind it twice as the spec was revised; the table below carries
only the one-line intent plus the section that owns the detail, so there is nothing to fall out of
sync.

| # | Intent | Owned by |
|---|---|---|
| R1 | Oversample the object in-plane, acquire only the nominal k-space band, reconstruct at the target matrix. Mandatory, not optional. | spec 3.1, 5.1 |
| R2 | Give the object a real complex phase: global, smooth pre-readout 3D field, and per-shot diffusion phase from `q . dx`. No fieldmap-derived static term — the sequence is spin-echo. | spec 3.2 |
| R3 | Add noise only at sampled k-space locations, under an explicit per-component variance convention. No extra sampled-fraction factor. | spec 3.3 |
| R4 | Keep the even-N window asymmetry; document and pin it rather than "fixing" it. | spec 3.4 |
| R5 | Emit `object-hires` / `object-nominal` / `acquired-clean` / `acquired-noisy`, score in an explicit Gibbs benchmark mode, and test against analytic references rather than fixed overshoot or period constants. | spec 3.5, 4.1 |

Three points that earlier revisions of this document got wrong, recorded so they are not
reintroduced:

- **No fieldmap-derived static phase.** TRXScan is spin-echo (`kspace.rs:229`, `readout.rs:63`);
  static off-resonance is refocused at TE and survives only as the readout-time phase already
  modelled as distortion. A `2*pi*fmap*TE` term would double-count B0.
- **No sampled-fraction variance factor.** Masking alone produces the `sqrt(f)` scaling; an
  additional factor double-counts it.
- **The nominal reference is derived, never regenerated.** Block-reduce the same high-resolution
  complex array that produced the acquired image. Re-running Stage A at `o = 1` loses intravoxel
  dephasing and is a different random realization besides.

## 4. Summary table

| Aspect | Status |
|---|---|
| Ringing arises from k-space truncation (correct cause) | Yes |
| Correct position in the acquisition chain | Yes |
| Ringing present by default / intrinsic to the forward model | **No** (exact DFT round-trip) |
| Fourier cutoff at the nominal acquisition Nyquist | **No** (cutoff moved; PSF scale wrong) |
| Sidelobes alternate sign every voxel (1D axis-aligned step) | **No** (2.13 px period at CLI default) |
| Overshoot monotonic in the parameter | **No** (1.6%-10%, reverses above 25%) |
| Preserves nominal resolution | **No** (1.07x-1.33x effective voxel) |
| Ringing distributed realistically across Re/Im | **No** (image is real to ~1e-16) |
| `part-phase` output realistic | **No** (mean \|phase\| 0.19 rad) |
| Noise band-limited to sampled k-space | **No** (white at all truncations) |
| Correct for 2D EPI (no through-slice ringing) | Yes |
| Survives GRAPPA / multi-coil combine | Yes |
| Covered by tests | **No** (change-only assertion) |
