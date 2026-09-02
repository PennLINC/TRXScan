## Summary

TRXScan's simulated DWI had no usable Gibbs ringing and no usable image phase. This PR replaces the
mechanism that produced them with a physically correct finite-acquisition forward model, and adds a
benchmark that can be used to evaluate unringing methods against known ground truth.

**The headline result.** The old and new mechanisms were run through the actual tools people use:

| phantom | ripple period | DIPY | `mrdegibbs` |
|---|---|---|---|
| zero-filled truncation (**old**) | 3.88 vox | **+16% worse** | −0% |
| crop + reconstruct (**new**) | 2.14 vox | **−87%** | **−92%** |

The old mechanism was not merely inaccurate — it was **uncorrectable by the methods it would have
been used to benchmark**, and actively misled one of them.

## What was wrong

Measured on the previous implementation:

- **Ringing was not intrinsic.** The forward and inverse transforms were an exact round trip
  (round-trip error 3e-15), so all ringing had to be injected by a `zero_ringing` parameter.
- **That parameter moved the Fourier cutoff** off the nominal acquisition bandwidth, giving a
  ripple period of 2.13 voxels at the shipped default instead of 2, and costing 1.07–1.33× in
  effective resolution.
- **The complex output was degenerate.** `max|Im| / max|Re|` was 4e-16 for a real object: the
  compartment signal was real and the only phase terms were `ky`-dependent, i.e. geometric warp
  rather than image phase. `part-phase` was unusable.
- **Noise was added after the k-space line skip**, so unacquired samples carried noise and image
  noise stayed white at every truncation.
- **The only test asserted that the image changed.**

## What changed

**Forward model.** The object is simulated on a finer in-plane grid and only the nominal k-space
band is evaluated — cropping *during* the forward transform rather than computing a large k-space
and discarding it (~3.3× cost, not 8×). Reconstruction stays at the acquisition matrix, so ringing
arises from the acquisition itself. Overshoot is now 8.95%, the Gibbs constant, with no ringing
parameter anywhere; `zero_ringing` is replaced by named reconstruction windows
(`None`/`Tukey`/`Hann`/`Fermi`).

**Object phase.** A global term, a smooth pre-readout 3D field, and a per-shot diffusion phase from
an effective q-vector `φ = q·Δx`. Deliberately **no** fieldmap-derived static term: the sequence is
spin-echo, so static off-resonance is refocused at TE and survives only as the readout-time phase
already modelled as distortion.

**Noise** is confined to the sampling mask under an explicit per-component variance convention, so
signal and noise can no longer disagree about what was acquired.

**Partial Fourier** is now an explicit choice. The ported Fiberfox rule preserves line zero on even
matrices, so a nominal 6/8 actually keeps 78.1% of lines; `PartialFourierMode::Contiguous` gives
scanner-like contiguous sampling and is the default for the HBCD-like CLI and the benchmark.

**Benchmark.** `trxscan-benchmark` emits `object-hires`, `object-nominal` (the scoring target) and
paired `acquired-clean`/`acquired-noisy` across a 24-point factor grid, plus a Python scorer that
decomposes error rather than reporting one distance — oscillatory residual, PF-aware residual
alignment and energy, per-edge sharpness and location bias, magnitude-masked phase error.

**Calibration.** The phase model's amplitude is fitted against real complex DWI (NIBS HBCD-protocol;
ds006131 CS-DSI as an independent check), using circular statistics with a thermal-noise term.

## What this is verified against

- **Analytic ground truth.** A truncated-Fourier-series oracle; the simulator matches its profile
  across 16 sub-voxel edge offsets. This matters because a *correct* implementation's sampled
  overshoot spans +0.29% to +8.93% depending on where the edge falls in the voxel — a fixed
  overshoot test would fail on correct code.
- **Real tools.** `mrdegibbs` 3.0.4 and DIPY 1.12.1 across the factor grid, with a no-op control.
- **Real data.** NIBS acquisition parameters match the shipping configuration exactly (TE 88 ms,
  6/8 PF, MB 3); measured background phase gradient 0.164 rad/voxel.
- **An independent second scorer** that shares no code with the primary and is compared on the same
  analytic fixtures. It has caught two defects the primary's own tests missed.

Current acceptance run (clean, unapodized; `none` is the no-op control):

| method | PF | Nyquist | PE alignment | PE energy | worst PE sharpness |
|---|---|---|---|---|---|
| none | 1.000 | 0.01945 | 1.000 | 1.000 | 1.221 |
| DIPY | 1.000 | 0.01195 | 0.204 | 0.683 | 1.055 |
| `mrdegibbs` | 1.000 | 0.01216 | 0.379 | 0.494 | 1.079 |
| DIPY | 0.750 | 0.01029 | 0.480 | 0.822 | 0.988 |
| `mrdegibbs` | 0.750 | 0.01025 | 0.607 | 0.720 | 0.904 |

Both methods beat the control and degrade as partial Fourier grows more aggressive. No ranking is
asserted — the suite checks physical consistency, not a winner.

## Using it

```bash
# generate a simulation grid at voxel/N alongside the acquisition grid
python scripts/prepare_acquisition_grid.py --anat-dir <anat> --prefix <p> --out work \
    --voxel 1.7 --oversample 2

# simulate; --oversample defaults to 2, --phase-model to the calibrated hbcd preset
trxscan --wm work/wm.nii.gz ... --sim-wm work/sim/wm.nii.gz ... --oversample 2
```

`--oversample 1` selects a documented legacy path that produces **no** ringing and no object phase,
and warns accordingly.

## Things worth knowing before merging

- **`--oversample` defaults to 2, not 4.** `o = 4` is the accuracy target (residual 4.5% of the
  artifact vs 15.6% at `o = 2`), but on the default no-motion path the dominant allocation is a
  dense `nvox × 321` f64 orientation histogram: an HBCD-sized `o = 2` run **measured 11.46 GB**
  against a 25.9 GB bound, and `o = 4`'s bound is ~104 GB. 16 GB is tight and data-dependent, not
  safe. z-slab streaming or sparse orientation storage is the blocker on raising the default.
- **RPG is not integrated**, so the partial-Fourier *axis* is measured but no PF-*aware method* is
  compared. This PR does not claim otherwise.
- **`sigma_rot` is heuristic**, not calibrated — only the translation amplitude is fitted.
- `--myelin` has no simulation-grid map yet; the CLI errors rather than silently mismatching.
- The `microstructure` module's DIPY moment oracle is a **skip, not a pass** — its fixture has never
  been tracked here. Out of scope for this PR (that module shares no code with the Gibbs work) and
  recorded as deferred in the spec.

## Testing

95 Rust tests under `--features cli`, 88 under default features, 74 Python tests.

CI **runs** three configurations rather than compiling them: `cargo test` (default, pure std),
`cargo test --features cli --all-targets`, and `cargo test --features cli,par --all-targets`. That
matters more than it sounds: `default = []` and `src/lib.rs` gates the whole `io` module on a
feature the default build does not enable, so before round 9 every test in `src/io.rs` — including
the geometry tests pinning the benchmark affine — ran in no CI step at all, and the binaries were
only type-checked. Both of the most serious bugs found in review lived in exactly that gap.

The Python suite is self-contained: nibabel, MRtrix and DIPY are deliberately absent from CI, and
tests needing them skip. Clippy is advisory, not a gate — most findings are pre-existing in modules
this branch does not touch.

One test in `microstructure` previously failed for every fresh clone (its fixture and generator have
never been tracked); it now skips explicitly when the fixture is absent, which is what made green CI
possible. It provides no coverage until someone restores the fixture, and is recorded that way in
the spec rather than counted above.

## Review history

Ten review rounds are recorded under `docs/superpowers/reviews/`, each with a point-by-point
response — nine from one reviewer, one independent pass plus a cross-review of it. The two rounds
worth reading if you read none of the others are round 9 (CI was not running the feature-gated
tests; the benchmark CLI silently substituted defaults for the arguments its own usage string
documented) and round 10 (a coil-sensitivity cleanup in round 9 mixed centred and absolute
coordinates, a half-FOV error in the multi-coil forward model — caught by review, fixed, and now
pinned by an end-to-end test that runs the production path rather than re-deriving the coordinate
it wishes the caller used).

Design rationale and measurements are in
`docs/superpowers/specs/2026-08-31-gibbs-ringing-realism-design.md`; the original assessment of the
old behaviour is in `docs/gibbs-ringing-assessment.md`.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01K6WMQX3cMWPPsYUMXh46tD
