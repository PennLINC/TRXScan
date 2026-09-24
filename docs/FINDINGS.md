# The Fiberfox simulator, decomposed

All `file:line` anchors are into the **MITK-Diffusion** source tree
(`Modules/MriSimulation/…`, from the 2024.11.26 release). Once you strip MITK/ITK/VTK/Qt,
the simulator is two transforms plus motion, and the code that matters is small.

## The pipeline in two stages

```
Signal       tractogram + tissue maps  ─────►  clean 4D DWI (magnitude, no artifacts)
Acquisition  clean DWI  ──per slice──►  realistic DWI (distortion, relaxation, coils, noise)
Motion       rigid-transforms the fibers (+mask) before signal generation; the fieldmap moves with acquisition
```

- **Signal stage** — `Algorithms/itkTractsToDWIImageFilter.cpp`, volume loop at `:1015`.
- **Acquisition stage** — same file, k-space loop at `:193`, nested **volume → slice `z` → coil `c`**,
  each slice handed to `Algorithms/itkKspaceImageFilter.cpp`.

## The signal stage — signal generation

Per volume `g` (`itkTractsToDWIImageFilter.cpp:1015`):

1. `SimulateMotion(g)` (`:1018`, `:1456`) poses the whole fiber bundle once (see Motion).
2. OpenMP loop over fibers (`:1050`). For each fiber segment `j`:
   - `seg_volume = fiberWeight · π · r²` (`:1073`).
   - **`IntersectImage(spacing, start, end, …)`** (`:1105`) → `Vec<(voxel_index, path_length)>` —
     the exact segment/voxel intersection with sub-voxel lengths. **This is the rasterizer TRXScan reimplements** (`src/raster.rs`).
   - For each fiber compartment `k`:
     `signal_add = model_k.SimulateMeasurement(g, dir) · seg_volume` (`:1110`), then for each
     intersected voxel `voxel[k][g] += length · signal_add` (`:1122`), accumulating the
     intra-axonal volume per voxel (`:1129`) for normalization.
3. After the fiber loop (`:1150`+): density-correct fiber compartments to the voxel volume, then
   fill the **non-fiber** signal (extra-axonal, GM, CSF) via `SimulateExtraAxonalSignal` using
   the tissue volume-fraction maps, and normalize the fractions to sum to 1 per voxel.

Output: `m_CompartmentImages` — one 4D `(x,y,z,g)` image per compartment, summed downstream.

### Signal models (`SignalModels/`)

The b-value is encoded in the **gradient norm** (|g|² relative to the baseline `m_BValue`), so
these are pure per-direction responses:

- **Stick** (intra-axonal), `mitkStickModel.cpp`:
  `S = exp(−m_BValue · d · (f̂·g)²)`  — `f̂` fiber tangent, `d` axial diffusivity.
- **Tensor**, `mitkTensorModel.cpp`: rotate a kernel tensor `D` (eigenvalues `d1,d2,d3`) onto the
  fiber direction via a quaternion, then `S = exp(−m_BValue · gᵀ D g)`.
- **Ball** (isotropic non-fiber; CSF/GM free water), `mitkBallModel.cpp`: `S = exp(−m_BValue · d_iso)`.
- Also present but out of scope for a first cut: `AstroStick`, `Dot`, `RawSh` (SH-based).

Each also has a `SimulateMeasurement(dir)` batch form returning the whole gradient vector.

## The acquisition stage — k-space (`itkKspaceImageFilter.cpp`)

Per (volume, slice, coil), the slice's compartment signal is turned into a distorted image:

- **Readout as a manual DFT** (`:303`+). Iterates k-space points; for each, sums over **all**
  image pixels applying a per-pixel phase — O(N⁴) per slice. This is Fiberfox's real bottleneck.
  TRXScan keeps the exact per-line sum but organises it for speed (`src/kspace.rs`): static
  factors hoisted, the affine-in-ky phase advanced by memoised rotors, and — under the `kspace`
  feature — the x-DFT / reconstruction done by rustfft and the fieldmap y-sum by a type-1 NUFFT
  (`src/nufft.rs`), so the whole forward is O(N log N) except the non-affine eddy polynomial. The
  literal sum survives as the test oracle (`restructured_forward_matches_the_literal_sum`).
- **`tick`** (`:305`) = readout time index; `m_ReadoutScheme->GetActualKspaceIndex(tick)` (`:308`)
  maps it to `(kx,ky)` per the trajectory (EPI zig-zag).
- **Off-resonance distortion** (`:435`): `phi += fmap[pixel] · t(ky)` where `t` is the readout time
  at that PE line. This is what warps EPI along the PE axis — and, because the fieldmap stays in
  scanner space while the head moves, it's the physically-correct motion×distortion coupling.
- **T2*/T1 relaxation** (`:369`): `f_real *= T1relax · exp(−tRf/T2 − |t|/tInhom)`.
- **Eddy currents** (`:428`): `phi += (g·pos) · eddyDecay`, `eddyDecay` from `:354`.
- **Coil sensitivity** (`:241`, `:420`): per-coil spatial weighting; coil ring moves with the slice.
- **Partial Fourier** (`:322`): **skips selected PE (`ky`) lines** (sets them zero, `continue`).
  *This exact mechanism is where GRAPPA undersampling slots in.*
- **Gibbs ringing** (`:334`): zeroes high-frequency k-space corners.
- **Ghosting** (`:378`): Nyquist ghost via alternating `kx` offset on odd/even lines.
- **Spikes** (`:506`): random k-space spikes.
- **Noise** (`:80`): variance scaled by partial Fourier and `1/(kx·ky)`, added in k-space, per coil.
- Then `itkDftImageFilter` transforms back to image space (`:300`); per-coil real/imag are combined
  into the magnitude DWI (`:315`+).

### Readout schemes (`Sequences/`)

Small (~60 lines each). `mitkSingleShotEpi.h`:
- `dt = tLine / kxMax`; `half_read_time = kxMax·kyMax·dt/2`.
- `GetActualKspaceIndex(tick)`: EPI snake — start at max `ky`, reverse `kx` every other line.
- `GetTimeFromMaxEcho / GetTimeFromLastDiffusionGradient / GetTimeFromRf`: linear in `tick·dt`.

The `AcquisitionType` interface (`Sequences/mitkAcquisitionType.h:41`) is **purely in-plane** —
there is *no* slice-dimension timing. Multiband slice scheduling is genuinely new code (see
[`FEATURES.md`](FEATURES.md)). `mitkFastSpinEcho.h`, `mitkConventionalSpinEcho.h` are the other two.

## Motion (`itkTractsToDWIImageFilter.cpp:1456`)

Per volume `g`, from a global amplitude + a `motionvolumes` list + a `randomMotion` flag:

- **Random** (`:1486`): each listed volume gets an independent uniform draw in `[−amp,+amp]` per
  axis; **resets fibers+mask to baseline between moved volumes** (`:1461`) → intermittent motion
  that returns to baseline.
- **Linear** (`:1508`): constant increment `amp / N_moved` per listed volume, accumulates and
  persists → monotonic drift.
- The per-volume transforms are stored (`m_Rotations`, `m_Translations`, `:1521`) and passed to the
  k-space filter (`SetTranslation`/`SetRotationMatrix`, `itkTractsToDWIImageFilter.cpp:270`) so the
  fieldmap warp tracks the moved head.

A load-order gotcha in the original: `motionvolumes` is parsed against the gradient count *at ffp
load time*, before the `-t` template sets the real scheme — so with no gradients loaded it silently
moves **all** volumes. TRXScan designs this away (the scheme is known up-front).

## What TRXScan omits

MITK/ITK/VTK/Qt entirely (→ `ndarray` + `nalgebra` + TRXScan's own streamline type); the Qt
Fiberfox GUI plugin; the `.ffp` XML parameter format and its load-order bugs (→ a Rust config
struct); procedural fiber *generation* / spline fitting (TRXScan consumes external tractograms);
DICOM and the format zoo; the unused signal models. What remains is the physics above.
