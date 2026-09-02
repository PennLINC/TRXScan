I agree with the agent’s **main diagnosis**, but I would **not implement the design unchanged**. The assessment correctly identifies the central problem—TRXScan’s present complex output is not a valid benchmark for complex-aware Gibbs removal—but several quantitative claims in the assessment are overstated, and the proposed design has a few substantive physics and testing problems.

## Overall verdict

The assessment is strongest on the software architecture. TRXScan defines its underlying diffusion signal on essentially the same discrete grid on which it reconstructs, so without additional truncation the DFT round trip has no opportunity to generate ordinary finite-acquisition Gibbs ringing. It also lacks a meaningful spatial image-phase model, leaving the reconstructed complex image effectively real. Those two observations are sufficient to conclude that the current output is inappropriate for benchmarking complex-valued Gibbs correction.

The design's central solution—**construct a finer complex object, acquire only the target-resolution k-space band, and reconstruct at the target matrix**—is the right direction. It is also conceptually consistent with how Fiberfox has been described: generate a higher-resolution object and crop its k-space to the desired acquisition matrix. ([ResearchGate][1]) The separate addition of physically plausible image phase is also necessary.

Where I disagree is mostly in the claimed quantitative properties of Gibbs ringing, the proposed phase model, and the noise tests.

## Review of the assessment

The claim that the present `zero_ringing` behavior is really **additional low-pass truncation** rather than simulation of the natural finite acquisition at the nominal matrix is correct. It both changes the PSF and lowers the effective spatial bandwidth. The conclusion that this is a poor benchmark for `mrdegibbs`-type algorithms at the nominal resolution is therefore well-founded.

However, I would rewrite the discussion of the “9% overshoot” and “exactly 2 voxel period.” The ~8.95% Gibbs constant describes the limiting **continuous reconstructed step response** of rectangular Fourier truncation. What you actually observe at discrete voxel samples depends strongly on where the edge lies relative to the reconstruction grid. That dependence is in fact central to the Kellner method: the paper explicitly motivates subvoxel shifting because ringing severity depends on how the sinc oscillation is sampled relative to the edge. ([PubMed][2])

So measurements such as “1.55% overshoot” do **not by themselves prove** that the simulated Gibbs amplitude is physically wrong. They can reflect the discrete sampling phase of the edge. The more defensible criticism is that the current operator changes the Fourier cutoff from the nominal acquisition Nyquist frequency, thereby changing the PSF and spatial scale of the ringing relative to the voxel grid. That is unquestionably a problem for a benchmark.

Similarly, “real Gibbs has exactly a two-voxel period” should be stated more carefully. For a rectangular full-Fourier acquisition at the target sampling bandwidth, successive sinc sidelobes alternate sign on the voxel scale; same-sign lobes recur on roughly a two-voxel scale. But a single “period = 2.000 voxels” statistic is not a complete characterization of the truncated step response.

I would also soften “real MRI always has Gibbs ringing.” A finite rectangular Fourier window has a sinc PSF, but scanners commonly apply k-space/image-space filters, and real tissue edges are not perfect step functions. MRtrix specifically warns that scanner filtering changes the ringing pattern and may leave residuals after `mrdegibbs`. ([GitHub][3])

The complex-data diagnosis, by contrast, is compelling. If the underlying object signal is real and the only substantial phase evolution is acquisition-time-dependent EPI phase, then that phase primarily encodes distortion rather than supplying a general static complex image phase. A complex benchmark needs a nontrivial complex object before the finite Fourier acquisition.

But I would not make “phase spans the full \(-\pi,\pi\) range” or “Im/Re is order 1” a validity requirement. Complex reconstructed DWI phase depends on coil combination, phase referencing, scanner reconstruction and acquisition. A constant global phase of 0.3 rad is already a perfectly valid complex image, albeit a trivial one. For testing complex-aware algorithms, what matters much more is having **controlled spatial phase structure and volume-dependent diffusion phase**, not forcing the wrapped phase histogram to be uniform.

## The most important problems in the proposed design

I would require the following changes before implementation:

1. **Remove `phi_bg = 2π·fieldmap·TE` as the default static phase model.** This is essentially a gradient-echo phase formula. TRXScan is simulating spin-echo EPI DWI. Static off-resonance is refocused at the spin echo; during the EPI readout it still causes line-dependent phase accrual and therefore distortion/blurring, which TRXScan already appears to model. Simply adding \(\exp(i2\pi\Delta f\,TE)\) to the object would double-count B0 effects and generate the wrong kind of phase. Spin-echo refocusing specifically suppresses the static off-resonance phase that would otherwise accumulate through TE, while off-resonance during EPI readout still produces distortion. ([PMC][4])

2. **Keep an object-phase model, but decompose it differently.** I would use a controlled global phase; a smooth subject/reconstruction phase field; and a diffusion-encoding-induced phase term varying by volume/slice. Motion during diffusion gradients is well established as a source of substantial diffusion-image phase. ([PMC][5]) If TRXScan explicitly models individual receive coils with complex sensitivities, coil phase should be coil-specific rather than folded into one common “coil/shim” polynomial. If it only aims to output a final combined complex image, call this a reconstruction/background phase term instead.

3. **Do not require `exp(iφ)` to be band-limited to the acquisition matrix.** That restriction is backwards. The high-resolution **complex object** may legitimately contain spatial frequencies above the acquired band; those frequencies are precisely what the finite acquisition is supposed to truncate. The requirement should instead be that the simulation grid adequately samples the continuous complex object—for example, negligible spectral energy close to the *simulation-grid* Nyquist limit. Otherwise you risk making complex ringing artificially easy.

4. **Fix the noise-model contradiction.** The design says both “add noise only at sampled locations” and “scale variance by sampled fraction,” but its proposed test expects image-noise SD to scale as \(\sqrt{f}\). With a fixed reconstruction normalization, adding independent equal-variance noise to only \(f\) of the k-space samples already produces the \(\sqrt{f}\) scaling. Multiplying each sample's variance by \(f\) as well would generally drive the SD toward \(f\), double-counting the sampling reduction. Decide what `noise_variance` means—preferably variance **per acquired complex k-space sample**—and derive every scaling from that definition. For GRAPPA, let the reconstruction generate the expected noise amplification/correlation rather than adding an extra heuristic factor.

5. **Do not make “lag-1 autocorrelation increases monotonically with undersampling” a general CI requirement.** Zero-filled PF creates correlated/anisotropic image noise, and GRAPPA produces structured correlated noise, but one particular lag-1 coefficient is not guaranteed to behave monotonically across arbitrary masks and GRAPPA kernels. Test the covariance or spectral structure against a known linear reconstruction instead.

6. **Replace the single 9%-overshoot test with a family of analytic edge tests over subvoxel positions.** A step at one arbitrarily defined “voxel boundary” will not necessarily place a reconstructed sample at the continuous 8.95% maximum. Kellner's whole method relies on this dependence on subvoxel edge position. ([Bishtref][6]) I would evaluate the entire 1D profile against an analytic truncated Fourier-series reference for perhaps 8–16 edge offsets in \([0,1)\) voxel. Then test the expected alternating sidelobes, energy, extrema and convergence with oversampling.

7. **Do not treat R1 as optional for this project.** The plan currently says Phase 2—adding object phase—is a reasonable stopping point if oversampling is too expensive.  That would fix the degenerate complex channel but leave you with the current incorrect Gibbs-generation mechanism. For your stated goal, **both R1 and R2 are required**. If R1 is too expensive, the solution is to optimize it—streaming, FFTs, direct evaluation from the native higher-resolution source—not to retain the current truncation model.

8. **Add explicit partial-Fourier validation.** The design says zero-filled PF remains supported, but virtually all of the analytic Gibbs tests are for the full-Fourier case. RPG exists precisely because PF changes the ringing structure; its authors describe PF ringing as arising from two intervals and apply the subvoxel-shift correction twice. ([GitHub][7]) You should have dedicated 6/8 and perhaps 7/8 tests verifying the asymmetric PF point-spread function and its interaction with complex phase.

## Oversample-and-crop itself

I strongly support this part.

The proposed approach of evaluating the object on a finer in-plane grid while acquiring only the target \(N_x\times N_y\) Fourier coefficients is much closer to a genuine MRI forward model.  It also naturally produces both the nominal-resolution full-Fourier Gibbs pattern and, when PF is subsequently applied, the additional asymmetric PF ringing.

There are two statements I would change, though.

First, an **integer simulation/acquisition matrix ratio is not mathematically required**. If both grids cover exactly the same FOV, Fourier-series sample spacing is \(1/\mathrm{FOV}\), independent of matrix size. An integer oversampling factor is a perfectly reasonable implementation restriction, but it should be documented as such rather than described as necessary for an “exact” crop.

Second, I would not say oversample-and-crop produces Gibbs “with no blur.” Finite acquisition bandwidth necessarily imposes a sinc PSF relative to the higher-resolution underlying object. The important distinction is that this is the **correct nominal acquisition PSF**, rather than the additional resolution loss produced by throwing away another 6–25% of an already target-resolution k-space matrix.

The discussion of source anatomy is mostly good. Resampling a 1.7-mm image upward cannot manufacture missing spatial frequencies. But resampling an actually finer **1-mm source** to a convenient 0.85-mm simulation lattice is different: although the interpolation does not invent frequencies above the 1-mm source bandwidth, the 1-mm data already contain frequencies above the Nyquist limit of the intended 1.7-mm acquisition. Those can therefore legitimately generate Gibbs when the target 1.7-mm Fourier band is selected.

## Phase testing should be much more controlled

I would remove `phase_is_broadly_distributed` from CI. Mean absolute wrapped phase is dependent on an arbitrary global phase origin, and the “no histogram bin >40%” threshold has no physical basis.

A much better deterministic test suite would include constant phase \(0\), \(\pi/4\), and \(\pi/2\); a smooth linear phase ramp; a low-order curved phase field; and a per-volume diffusion phase perturbation. For constant \(\phi\), the proposed \(\cos^2\phi/\sin^2\phi\) Re/Im ringing-energy test is excellent. You can also test a very strong invariant: globally rotating the object by \(e^{i\alpha}\) must rotate the reconstructed complex image by exactly \(e^{i\alpha}\), while leaving its magnitude unchanged.

For realism calibration, use circular/spatial quantities such as local phase-gradient RMS, phase structure functions or correlation lengths, wrap density, and inter-volume phase differences after removal of a global phase offset. Those are much more informative than the marginal wrapped-phase histogram.

Also, I would not currently hard-code HBCD/HCP-D as the calibration datasets. The HBCD release documentation I can find describes released raw DWI files, b-values, b-vectors and SBRefs, but does not document `part-phase` DWI files. ([HBCD Study Docs][8]) HCP-D similarly documents unprocessed diffusion data but I did not find evidence in the release documentation that reconstructed complex/phase DWI is supplied. ([Human Connectome Project][9]) The design should say “a verified complex-DWI dataset” until that has been established.

## Apodization

I would separate **algorithm-validation mode** from **scanner-realism mode**.

For the core Gibbs benchmark, the default should be an unapodized rectangular full-Fourier acquisition. That gives the cleanest known forward model and is exactly the scenario against which Kellner/`mrdegibbs` assumptions can be tested. MRtrix explicitly recommends disabling scanner filtering for optimal `mrdegibbs` performance. ([GitHub][3])

Then optionally support named reconstruction windows—Tukey, Hann, Fermi, scanner-derived filters—with explicit parameters. I would not have one ambiguous `apodization: f64` whose interpretation is “Hanning/Fermi.” Those are different transfer functions and produce different PSFs.

Calibration against real tissue-edge “ringing amplitude” is also highly confounded by edge orientation, partial volume, reconstruction filtering and subvoxel position. A phantom or raw/full-Fourier k-space dataset would be much better for estimating scanner apodization.

## How I would reorder the implementation

| Phase | What I would do                                                                                     |
| ----- | --------------------------------------------------------------------------------------------------- |
| 0     | Analytic 1D/2D phantom/reference implementation; define Fourier conventions and subvoxel edge tests |
| 1     | **Oversample → finite acquisition band → target reconstruction**                                    |
| 2     | **Controlled complex object phase**, excluding `2π·fieldmap·TE`                                     |
| 3     | Full-Fourier complex Gibbs tests and benchmark fixtures                                             |
| 4     | PF sampling plus dedicated 6/8 and 7/8 theoretical tests                                            |
| 5     | Sampling-aware complex k-space noise with an explicitly defined variance convention                 |
| 6     | Optional reconstruction filters/apodization                                                         |
| 7     | Calibration against verified complex DWI                                                            |
| 8     | FFT/streaming optimization as necessary                                                             |

That sequencing gets to the thing you actually need—a **valid complex Gibbs benchmark—as early as possible**.

### Bottom line

The agent found the important TRXScan problem correctly, and **oversample-and-crop plus a genuine object-phase model is the right architecture**.

I would nevertheless change the design's status from **“approved” to “needs revision before implementation.”** The blockers are the spin-echo-inappropriate `2π·fieldmap·TE` phase term, the inconsistent noise scaling, the use of a fixed ~9% discrete-voxel overshoot as ground truth, the phase-histogram validity criterion, and the idea that stopping after phase modeling without fixing the Gibbs forward model would still meet the project's goal. The rest is mostly refinement rather than a change in direction.

[1]: https://www.researchgate.net/publication/264162038_Evaluation_of_neuronal_fiber_tractography?utm_source=chatgpt.com "(PDF) Evaluation of neuronal fiber tractography"
[2]: https://pubmed.ncbi.nlm.nih.gov/26745823/?utm_source=chatgpt.com "Gibbs-ringing artifact removal based on local subvoxel-shifts."
[3]: https://github.com/MRtrix3/mrtrix3/blob/master/cmd/mrdegibbs.cpp?utm_source=chatgpt.com "mrtrix3/cmd/mrdegibbs.cpp at master · MRtrix3/mrtrix3 · GitHub"
[4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC2819607/?utm_source=chatgpt.com "Efficient Correction of Inhomogeneous Static Magnetic Field-Induced Distortion in Echo Planar Imaging - PMC"
[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3320700/?utm_source=chatgpt.com "A Model for the Correction of Motion-Induced Phase Errors in Multi-Shot DW-MRI of the Head: Are Cardiac-Motion-Induced Phase Errors Reproducible from Beat to Beat? - PMC"
[6]: https://bishtref.com/articles/10.1002/mrm.26054?utm_source=chatgpt.com "Gibbs‐ringing artifact removal based on local subvoxel‐shifts (2015)"
[7]: https://github.com/NYU-DiffusionMRI/rpg-degibbs?utm_source=chatgpt.com "GitHub - NYU-DiffusionMRI/rpg-degibbs: Removal of Partial-Fourier induced Gibbs-ringing (RPG) in Lee et al., Magn Reson Med 2021 · GitHub"
[8]: https://docs.hbcdstudy.org/r1.1/instruments/mri/dmri/?utm_source=chatgpt.com "Diffusion MRI - HBCD Data Release Docs"
[9]: https://www.humanconnectome.org/study/hcp-lifespan-development?utm_source=chatgpt.com "HCP Development - Connectome - Publications"
