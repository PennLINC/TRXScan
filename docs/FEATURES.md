# Extensions: multiband motion and GRAPPA

The two capabilities TRXScan adds beyond Fiberfox. Both extend the k-space stage; Fiberfox can't
do either because its acquisition model is volume-level and single-image, whereas in a clean Rust
design they fall out naturally.

## 1. Within-volume motion under multiband

**Why Fiberfox can't.** Signal is generated once per volume at one pose
(`itkTractsToDWIImageFilter.cpp:1018`); all slices of that volume share it. The `AcquisitionType`
interface has **no slice-dimension timing** (`Sequences/mitkAcquisitionType.h:41`) — slices are just
looped. So motion can only change between volumes.

**The design.**

1. **Slice schedule.** A multiband model: `mb` simultaneously-excited slices (separated by
   `n_slices/mb`), acquired in an interleaved order over the TR. Produces, per volume, a list of
   *slice-groups* each with an acquisition time `t_group` → a head pose `P(volume, group)` sampled
   from the motion trajectory.
2. **Per-slice-group signal.** For faithful slice-level motion the slice's *content* must be
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

## 2. GRAPPA (parallel imaging)

**Big head start.** Fiberfox already synthesizes a **per-coil complex forward model** — the coil loop
(`itkTractsToDWIImageFilter.cpp:247`), per-coil sensitivity that moves with the excited slice
(`itkKspaceImageFilter.cpp:241`), per-coil real/imag output — and it already **skips PE lines** for
partial Fourier (`itkKspaceImageFilter.cpp:322`). The whole substrate exists in the `kspace` port.

**The design.**

1. **Undersample.** Add `accel R` + `acs_lines` to the PE-line skip: acquire every `R`-th `ky` line
   plus a central ACS band. Mirrors the partial-Fourier skip condition (~10 lines).
2. **Noise before recon.** Keep noise injection per coil in the undersampled k-space (already there).
3. **GRAPPA reconstruction.** Fit kernel weights from the ACS lines (per coil), synthesize the missing
   lines, then coil-combine (RSS). New numeric code, well-specified. `nalgebra` for the weight solve.
4. **g-factor for free.** Because you reconstruct genuinely undersampled *noisy* multi-coil data, the
   spatially-varying noise amplification (g-factor) emerges from the physics — no need to model it.

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
