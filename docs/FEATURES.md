# Design note: multiband motion and GRAPPA

The first two extensions of the k-space stage beyond the ported Fiberfox model (later ones,
gradient nonlinearity and the GRE fieldmap, have their own note in `GNL.md`). Fiberfox's
acquisition model is volume-level and single-image, so neither exists there; TRXScan's per-slice
acquisition stage is where they attach.

**Status: the design note below predates the implementation. What was built differs in three
places:**

- *Multiband motion* is the "cheaper first cut" of §1: `motion::apply_multiband_motion` keeps the
  whole-volume signal and resamples the finished images per slice-group, layering within-volume
  shot jumps and b-scaled dropout on top (with the dropped-slice ground truth). The
  per-volume re-simulation (`compartments::generate_compartments_moving`) is volume-level only.
- *GRAPPA* (§2) is as designed except the coil combination: it is a Roemer combine with the
  known sensitivities, not RSS (`kspace.rs`, `simulate_slice`).
- The *TOML config surface* was never written: `config.rs` is a `todo!()` stub and the protocol is
  `kspace::hbcd_acquisition` in the library plus `clap` flags in `src/bin/trxscan.rs`.

## 1. Within-volume motion under multiband

**Where the Fiberfox model stops.** Signal is generated once per volume at one pose
(`itkTractsToDWIImageFilter.cpp:1018`); all slices of that volume share it. The `AcquisitionType`
interface has **no slice-dimension timing** (`Sequences/mitkAcquisitionType.h:41`) — slices are just
looped. So motion can only change between volumes.

**The design.**

1. **Slice schedule.** A multiband model: `mb` simultaneously-excited slices (separated by
   `n_slices/mb`), acquired in an interleaved order over the TR. Produces, per volume, a list of
   *slice-groups* each with an acquisition time `t_group` → a head pose `P(volume, group)` sampled
   from the motion trajectory.
2. **Per-slice-group signal.** For slice-level motion the slice's *content* must be
   generated at its group's pose, not the volume pose. Restructure the signal stage to
   `volume → slice-group → pose → paint only that group's slices`. Cost ≈ ×(`n_slices/mb`) the fiber
   loop per volume (bounded, `rayon`-parallel). A cheaper first cut: keep whole-volume signal but
   index the k-space **distortion** pose per slice-group (Fiberfox already takes a per-call
   `SetTranslation`/`SetRotationMatrix`, `itkTractsToDWIImageFilter.cpp:270`), which captures the
   motion×distortion interaction without regenerating signal.
3. **Artifact.** This reproduces the dominant DWI motion signature: slice/volume misalignment and
   through-plane dropout, correctly coupled to EPI distortion.

**Explicitly out of scope (first cut):** true *spin-history* — re-exciting tissue that moved between
excitations (slice cross-talk) is a Bloch-level problem, not geometric, and much harder. Flag it as
future work.

**Since then (P9, shared with aslscan through mrsim-acq).**

- *Per-shot motion* (`motion::ShotPlan`, `motion::apply_shot_motion`): a shot is the slices excited
  together, from per-slice offsets (any order: `--slice-order`, or a protocol's `SliceTiming`), so
  single-band runs have within-volume events too. Each shot is resampled once by its transform,
  about the whole stack's centre (slabs match the full run). Events can come from a file
  (`--motion-events`, `ShotEvents(events=...)`) in aslscan's `desc-motionEvents_gt.tsv` format, and a
  run writes its own as `<out>_desc-motionEvents.tsv` to replay it.
- *Spin history* (`--spin-history`, `ShotEvents(spin_history=True)`): the longitudinal
  magnetization of each compartment is stepped through every slice excitation at the shot's pose
  (`mrsim_acq::spin_history`: a box slice profile as thick as the slice spacing, each voxel's z
  extent kept as exact pieces), and the difference from the closed form `1 - exp(-TR/T1)` is added,
  weighted by the compartment's signal, after the geometry; a still head is unchanged. It needs a
  `T1` per compartment and the repetition time, and the whole slice stack.
- *Still not modeled:* real slice profiles and slice gaps (cross-talk without motion), the voxel's
  tilt under rotation inside the box overlap, and spin history of the head compartments
  (`Protocol.head` is refused). Under the re-simulation path the spin-history weights are read
  from the re-simulated images moved back by each volume's pose (trilinear), into the head frame.

## 2. GRAPPA (parallel imaging)

**What the ported model already provides.** Fiberfox synthesizes a **per-coil complex forward model** — the coil loop
(`itkTractsToDWIImageFilter.cpp:247`), per-coil sensitivity that moves with the excited slice
(`itkKspaceImageFilter.cpp:241`), per-coil real/imag output — and it already **skips PE lines** for
partial Fourier (`itkKspaceImageFilter.cpp:322`). The whole substrate exists in the `kspace` port.

**The design.**

1. **Undersample.** Add `accel R` + `acs_lines` to the PE-line skip: acquire every `R`-th `ky` line
   plus a central ACS band. Mirrors the partial-Fourier skip condition (~10 lines).
2. **Noise before recon.** Keep noise injection per coil in the undersampled k-space (already there).
3. **GRAPPA reconstruction.** Fit kernel weights from the ACS lines (per coil), synthesize the missing
   lines, then coil-combine (RSS). New numeric code, well-specified. `nalgebra` for the weight solve.
4. **g-factor.** Because undersampled, noisy multi-coil data are reconstructed, the
   spatially-varying noise amplification (g-factor) follows from the reconstruction and is not
   modelled separately.

## Config surface (sketch)

```toml
[motion]
mode = "linear"            # off | random | linear | trajectory
trans_mm = [3.0, 3.0, 2.0]
rot_deg  = [3.0, 2.0, 2.0]
within_volume = true       # apply per slice-group instead of per volume

[acquisition]
multiband = 3
slice_order = "interleaved" # sequential | interleaved
partial_fourier = 0.75

[parallel_imaging]
accel = 2                   # GRAPPA R
acs_lines = 24
```
