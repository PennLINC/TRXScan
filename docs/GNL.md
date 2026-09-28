# Gradient nonlinearity (GNL) in TRXScan — design note

Status: **implemented** (`src/gnl.rs`, the `--gnl*` flags of `trxscan`, `compartments::
signal_from_mixture_gnl`; unit tests 1–7 of §5 are in-module). This note is the design it was
built from; where the code differs, the code wins:

- `GnlField::on_grid(coef, grid)` takes no isocentre — the field is evaluated about the world
  origin and `trxscan --isocenter` translates the grids and streamlines instead. `envelope(grid,
  radius)` likewise.
- Outputs: `<out>_desc-gnl_coeff.grad`, `_desc-gnl_disp` (φ(r)−r, warps points/streamlines
  true → apparent), `_desc-gnl_invdisp` (φ⁻¹(x)−x, pulls images apparent ← true) and
  `_desc-gnl_graddev`, all RAS mm on the written grid, plus JSON sidecars — there is no
  separate export binary.
- The GRE synthesizer §7 asked for is `src/gre.rs` / `--gre-out` (Siemens integer phase,
  `phasediff` or `phase1/2`, optional coarser `--gre-res`, `B0FieldIdentifier` linkage).
- `--gnl` with `--motion` is refused (the per-segment path has no per-voxel gradient).
- The cross-tool checks 8–10 of §5 were run by hand (§7); `tools/validate_gnl.sh` was never
  written.

Companion to `docs/FEATURES.md` (motion/GRAPPA) and `docs/FORCE.md` (ground truth). Written
2026-09-09 to close the validation loop for qsiprep's `--gradient-file` (gradwarp +
`*_space-ACPC_graddev.nii.gz`) and `odx graddev`.

## 1. What GNL is, and what the simulator has to reproduce

A gradient coil `a ∈ {x, y, z}` does not produce a perfectly linear field. Its z-field at
scanner position `r` (metres from isocentre) is a solid-harmonic expansion

```
φ_a(r) = R0 · Σ_{l,m} [ α_a(l,m) cos(mϕ) + β_a(l,m) sin(mϕ) ] · (r/R0)^l · N_lm · P_l^m(cos θ)
N_lm   = (−1)^m √( (2l+1)(l−m)! / (2 (l+m)!) )   for m > 0,   N_l0 = 1
```

with the `l = 1` term being the nominal linear field (`α_x(1,1) = β_y(1,1) = α_z(1,0) = 1`,
so `φ_a(r) = r_a` for a perfect coil). This is the Siemens convention as read by
TORTOISE (`GradWarpDispAtPoint`,
`src/tools/CreateGradientNonlinearityBMatrix/CreateGradientNonlinearityBMatrix.cxx:131-322` in
the TORTOISE source) and by HCP's `gradunwarp` (`siemens_B`, BSD-3). Write `φ(r) = r + d(r)`; `d` is the
displacement field. Two things happen to a spin at true position `r`:

1. **Spatial encoding error.** The scanner assigns it the position `φ(r)`. The acquired image is
   `I_dist(r') = I_true(r) / |det ∇φ(r)|` with `r' = φ(r)` — a smooth 3-D warp (through-slice
   too) plus an intensity Jacobian. Images look compressed toward isocentre; gradwarp undoes it.
2. **Diffusion encoding error.** The gradient it experiences is
   `G_eff(r) = Σ_a g_a ∇φ_a(r) = J(r)ᵀ g`, `J_ij = ∂φ_i/∂x_j`. Direction *and* magnitude change,
   so both b-vector and b-value deviate per voxel. qsiprep's graddev stores `J` row-major
   (HCP layout, `vol[3i+j] = J_ij`, identity included) in the output image's voxel frame — see
   the `graddev` module docs in `odx-rs` for the consumer side.

Both effects derive from one object, `φ`. TRXScan must simulate both, from one synthetic
coefficient set that it also *writes out* in Siemens `.grad` format, so that qsiprep and
TORTOISE consume the very same field. That closes the loop: truth field vs TORTOISE's field
(unit-level), and truth orientations vs corrected reconstruction (end-to-end).

### What the existing code provides

- **b lives in the gradient norm** (`src/scheme.rs:123-137`, `fiberfox_gradient`). Replacing
  the encoded gradient `g` by `J(x)ᵀ g` per voxel gives the full b-matrix deviation
  `exp(−b_max · gᵀ J D Jᵀ g)` with zero changes to `Stick`/`Tensor`/`Ball`.
- **Ground truth is scheme-free**: `field_scalars` reads the mixture, not the gradients
  (`docs/FORCE.md` §3). The GNL stage cannot contaminate the answer key.
- The myelin branch of `signal_from_mixture` (`src/compartments.rs:526-556`) is already the
  "abandon the global response table, evaluate this voxel's nonzero bins" pattern that a
  per-voxel gradient needs; `docs/FORCE.md` §4d sizes it (~1e9 exp per brain, seconds under
  rayon).

## 2. Coefficient sets without proprietary files

Vendor `.grad`/`gw_coils.dat` files cannot ship. The synthetic sets are built from three
constraints that every whole-body gradient set satisfies, plus a calibrated *envelope* rather
than copied numbers:

**Symmetry (structural realism).** Whole-body coils are antisymmetric about isocentre, so only
odd `l` appear. The transverse coils have odd `m` only: x uses cosine terms `A(l, m)`, y uses
sine terms `B(l, m)` (TORTOISE encodes the y-sine terms as negative `m`). The z coil is
axisymmetric: `m = 0` only. `l = 3` dominates; `l = 5` is smaller with opposite sign (it
flattens the roll-off); `l = 7` is a small residual. The `l = 3` coefficients are **negative**:
the field rolls off below linear at large `r`, which gives the familiar compression toward
isocentre and `|G_eff| < |g|` over most of the periphery.

**Envelope (magnitude realism), at `r = 10 cm` from isocentre (adult brain edge), for a
whole-body 80 mT/m class system (Prisma-like):**

| quantity | target at 10 cm | at 5 cm (infant, HBCD) |
|---|---|---|
| displacement `|d|` | 1–4 mm | ≈ ×1/8 (l = 3 goes as r³) |
| relative gradient amplitude deviation `|J ᵀ ĝ| − 1` | 3–8 % | ≈ ×1/4 (goes as r²) |
| b-value deviation (= 2× the above) | 6–15 % | 1–4 % |
| direction deviation `∠(ĝ, Jᵀĝ)` | 1–4 ° max | < 1 ° |
| all of the above within 2 cm of isocentre | < 0.1 mm, < 0.5 %, < 0.2 ° | — |

These are order-of-magnitude anchors from the literature on whole-body 3 T systems (Bammer
et al. 2003 MRM 50:560 — b and direction errors growing away from isocentre; Mesri et al.
2020 NeuroImage 205 — in-brain b deviations of several percent, angular deviations of a few
degrees; Sotiropoulos et al. 2013 NeuroImage 80:125 and Rudrapatna et al. 2021 for the
Connectom 300 mT/m system, where everything is 2–4× larger). They are targets to calibrate
to, not claims about any vendor file. Note the last column: at HBCD's infant head size the
bias is barely above reconstruction noise, which is *why* the simulator needs an explicit
severity knob.

**Starting set (`GnlPreset::WholeBody80`), R0 = 0.25 m, to be calibrated by the envelope
test in §5:**

```
x: A(3,1) −0.08   A(3,3) +0.006   A(5,1) +0.010   A(5,5) −0.001   A(7,1) −0.001
y: B(3,1) −0.08   B(3,3) +0.006   B(5,1) +0.010   B(5,5) −0.001   B(7,1) −0.001
z: A(3,0) −0.10   A(5,0) +0.015   A(7,0) −0.002
```

Back-of-envelope at 10 cm (`r/R0 = 0.4`): displacement `≈ R0·c·0.4³·O(1) ≈ 250·0.08·0.064
≈ 1.3 mm`, gradient deviation `≈ c·3·0.4² ≈ 4 %`, i.e. inside the envelope; the test tunes the
exact values. Second preset `Connectom300` = the same shape scaled ≈ 2.5×. A third preset
`HeadInsert` (short linear region, larger `l = 5`) is optional. qsiprep's own synthetic
fixture (`~/projects/qsiprep/qsiprep/tests/gradient_fixtures.py`, `DEFAULT_TERMS`) is the
same file grammar and can be loaded as-is for parity tests, but it was not calibrated for
realism.

**Severity knob.** `--gnl-scale s` multiplies every nonlinear term (`l ≥ 3`) by `s`. Sweeping
`s ∈ {0, 1, 3, 10}` on one phantom gives a dose-response curve for any correction under test,
and `s = 3` is what makes an HBCD-sized head show a measurable effect.

**Isocentre.** The field is defined about the scanner isocentre, which TORTOISE takes to be
the NIfTI world origin. TRXScan evaluates the field at `p_world − isocenter` and, when
`--isocenter` is given, **translates the output affines so that point becomes the world
origin** (streamline/tissue inputs are untouched; only the written headers move). Off-centre
placement (an infant positioned 4 cm superior of isocentre, say) is then a one-flag experiment.

**Vendor frame.** TORTOISE evaluates Siemens coefficients at `lps2lai · p_LPS`
(`CreateGradientNonlinearityBMatrix.cxx:341-348, 461-473`), i.e. `(−x, +y, −z)` of the RAS
point, and conjugates the resulting matrix back the same way; for GE/Philips it uses
`lps2ras`, which makes the scanner frame *equal* to RAS. Implement `scanner_from_ras` /
`ras_from_scanner` as that diagonal (`S = diag(−1, 1, −1)` Siemens, `I` otherwise) and
conjugate `J_ras = S · J_scanner · S`. Do not trust this paragraph: §5's cross-tool test
pins it numerically.

## 3. Implementation

### 3.1 `src/gnl.rs` (new, pure std, ~450 lines)

```rust
pub enum Axis { X, Y, Z }
pub struct GnlTerm { pub axis: Axis, pub l: u8, pub m: u8, pub sine: bool, pub coef: f64 }
pub enum Vendor { Siemens, GeOrPhilips }
pub struct GradCoef { pub r0_mm: f64, pub vendor: Vendor, pub terms: Vec<GnlTerm> }

impl GradCoef {
    pub fn preset(p: GnlPreset) -> Self;
    pub fn scale_nonlinear(&mut self, s: f64);
    /// Siemens `.grad` grammar exactly as TORTOISE reads it (gradcal.cxx:99-183) and
    /// qsiprep writes it: `N A(l,m) coef x`, axis letter last on the line, `R0` line in metres.
    pub fn parse_siemens(text: &str) -> Result<Self, String>;
    pub fn write_siemens(&self) -> String;
    /// Displacement d(r) in scanner mm: the l ≥ 3 part of φ (port of gradunwarp `siemens_B`;
    /// associated Legendre by recurrence, no scipy). Adds LICENSE-GRADUNWARP (BSD-3).
    pub fn displacement_scanner(&self, p_mm: Vec3) -> Vec3;
    pub fn displacement_ras(&self, p_ras: Vec3, isocenter: Vec3) -> Vec3;
    /// J = ∂φ/∂x by central differences of φ = p + d (ε = 0.5 mm; φ is a low-order
    /// polynomial, so this is exact to ~1e-6 and avoids porting TORTOISE's `dshdx`).
    pub fn jacobian_ras(&self, p_ras: Vec3, isocenter: Vec3) -> Mat3;
}

/// Per-voxel cache on the simulation grid.
pub struct GnlField { pub dims: [usize; 3], pub disp: Vec<Vec3>, pub jac: Vec<Mat3> }
impl GnlField {
    pub fn on_grid(coef: &GradCoef, grid: &Grid, isocenter: Vec3) -> Self;
    /// max |d|, max ||Jᵀĝ|−1|, max ∠(ĝ, Jᵀĝ) within a radius — the envelope test + `--gnl-info`.
    pub fn envelope(&self, grid: &Grid, isocenter: Vec3, radius_mm: f64) -> Envelope;
    /// HCP/FSL layout, identity included, components in the grid's own (i,j,k) frame:
    /// J_ijk = Rᵀ J_ras R with R the grid's column-normalized rotation. 9 values per voxel.
    pub fn graddev_volumes(&self, grid: &Grid) -> Vec<f32>;
}

/// I_dist(r') = I_true(φ⁻¹(r')) / |det J|: backward map by fixed-point iteration
/// r ← r' − d(r) (4 steps suffice for |d| ≲ 5 mm), trilinear sample (`motion::trilinear`
/// becomes `pub(crate)`), optional Jacobian modulation.
pub fn warp_volume(src: &[f32], grid: &Grid, field: &GnlField, modulate: bool) -> Vec<f32>;
```

### 3.2 Signal stage — `src/compartments.rs`

- `signal_from_mixture(mix, scheme)` gains a sibling
  `signal_from_mixture_gnl(mix, scheme, field: &GnlField)`; the existing function stays
  bit-identical. Inside the voxel closure (`compartments.rs:505-557`): build
  `gv[g] = matvec(&transpose(&field.jac[vox]), grads[g])` once per voxel, take the
  "evaluate nonzero bins directly" branch unconditionally (the myelin lerp folds in — refactor
  that branch into `fn wm_response(bins, gv, stick, extra)` so there is one copy), and evaluate
  `gm_resp`/`csf_resp`/`iso` per voxel from `gv` instead of the global tables at
  `compartments.rs:489-498`. Cost: ~20 bins × ngrad × 2 exp per voxel.
- `generate_compartments_moving` (`compartments.rs:595-681`): the per-volume `grad` at `:629`
  becomes per-voxel `Jᵀ_vox grad` inside the voxel combine loop (`:661-681`). The field is
  sampled at the *grid* voxel, which is correct: streamlines were moved into scanner space and
  rasterized there, and GNL is fixed in scanner space.
- The legacy per-segment `generate_compartments` is not extended (error if GNL is requested on
  a path that uses it). It computes responses per segment, not per voxel.

### 3.3 Spatial stage — `src/bin/trxscan.rs`

Order in `main` after the signal stage: multiband motion (`:225-240`, the object moves in the
scanner) → **GNL warp of every compartment image and of the fieldmap, per volume** (new) →
`simulate_acquisition` (`:274`). Each of the three compartment images is warped separately with
the same `φ`, so the per-compartment T2 in `kspace` still applies. The fieldmap must be warped
too: `kspace` looks the off-resonance up at the voxel it is encoding (`φ = fmap(x,y)·t(ky)`,
`src/kspace.rs:6,454`), and ΔB is a property of the tissue, so on the apparent grid it reads
`ΔB(φ⁻¹(x))`, exactly what a GRE fieldmap of the same object records. The two distortions then
compose (encoding error in 3-D, then PE-axis shift), which is the physical order to first
approximation. The eddy gradient array at `:267-273` is left alone (a scanner-level
model, not a per-voxel one).

CLI (no TOML yet, per `config.rs` being a stub):

```
--gnl <whole-body-80 | connectom-300 | path/to/coeff.grad>
--gnl-scale <f>                (default 1.0; multiplies l ≥ 3 terms)
--isocenter <x,y,z>            (world RAS mm; default 0,0,0; shifts output affines)
--gnl-no-warp                  (encoding deviation only — isolates the graddev effect)
--gnl-no-encoding              (spatial warp only — isolates the gradwarp effect)
--gnl-no-jacobian-modulation
--gnl-info                     (print the envelope at 2/5/8/10/12 cm and exit)
```

`--gnl-no-warp` / `--gnl-no-encoding` are the important ones for testing: the first produces
the dataset on which `odx graddev` alone should recover truth; the second is what qsiprep's
spatial gradwarp alone should undo.

### 3.4 Outputs — `src/io.rs`

Next to the existing `<out>_part-mag_dwi.nii.gz` family:

- `<out>_desc-gnl_coeff.grad` — the synthetic coefficient file, in the grammar TORTOISE parses.
  This is what goes to `qsiprep --gradient-file`.
- `<out>_desc-gnl_graddev.nii.gz` — truth `J` (9 volumes, HCP layout, identity included, on the
  *undistorted* grid, grid voxel frame). Comparable to qsiprep's graddev after accounting for
  the ACPC rotation, and directly to TORTOISE's output on the raw b0.
- `<out>_desc-gnl_disp.nii.gz` — `d(r)` in RAS mm, 3 volumes (or ITK 5-D vector layout so
  `CreateNonlinearityDisplacementMap` output diffs against it directly).
- JSON sidecar keys (edit the `format!` at `io.rs:345-351`): `GradientNonlinearityPreset`,
  `GradientNonlinearityScale`, `GradientCoefficientFile`, `IsocenterRAS`.
- `write_4d` (`io.rs:259`, private) becomes `pub(crate)` or gets a `write_components` sibling.

`trxscan-microstructure` is untouched: its maps are the undistorted truth, which is exactly what
a gradwarp-corrected reconstruction should be scored against (`docs/FORCE.md` §3, "ground truth
for the signal stage only").

## 4. Interactions and known approximations

- **Motion × GNL.** In the per-volume re-simulation path the field is sampled where the tissue *is*
  during each volume (scanner grid), so subject motion changes the bias in head coordinates —
  the effect Rudrapatna et al. describe. In the default path with multiband intra-volume
  motion, the encoding deviation is evaluated at the unmoved position and only the images move;
  document as an approximation (error ≈ field gradient × motion amplitude, sub-0.1 ° for mm-scale
  motion).
- **Slice selection** is also GNL-affected; it is covered by the 3-D warp (through-slice
  component), not modelled in `kspace`.
- **Intensity modulation** `1/|det J|` is a few percent at the periphery and identical across
  volumes, so it cancels in `S/S0`; it matters only for absolute-intensity comparisons with
  qsiprep's gradwarp output (TORTOISE's resampling does not apply it — leave it on for physics,
  off to match).
- **Vendor coverage**: Siemens grammar and frame first; GE/Philips is a different file grammar
  (`read_GE_format`) but the same physics — the frame switch is one diagonal.

## 5. Validation

Unit tests, in-module and std-only as `CLAUDE.md` requires:

1. `.grad` write → parse round trip, and the exact TORTOISE grammar quirks qsiprep's
   `test_gradient_fixtures.py` asserts (axis letter last, `R0` in columns 1–5, `(` at column ≥ 3).
2. A linear-only set gives `d ≡ 0`, `J ≡ I`; odd-`l` sets satisfy `d(−r) = −d(r)`.
3. Finite-difference `J` against the closed form for `A(3,0)` (`φ_z = z + c·R0·(r/R0)³·P_3(cos θ)`
   differentiates by hand) to 1e-6.
4. The preset envelope at 10 cm lands inside the table in §2 and below the near-isocentre
   bounds — this is the "realistic ballpark" test, and it fails loudly if someone retunes a
   coefficient carelessly.
5. `warp_volume`: fixed-point inverse converges (`|φ(r) − r'| < 1e-3 mm`); a constant image
   with modulation preserves its sum; a point source moves by `d`.
6. Identity field ⇒ `signal_from_mixture_gnl` bit-identical to `signal_from_mixture`.
7. Uniform-`J` field with `J = R` (rotation) ⇒ signal equals the baseline simulated with bvecs
   rotated by `R` — the same convention test `odx-rs/tests/graddev.rs` uses from the other side.

Cross-tool, scripted in the `qsiprep:unstable` container (the test that pins SH
normalization and frame — everything above could be self-consistently wrong):

8. `CreateNonlinearityDisplacementMap coeff.grad b0.nii.gz field.nii 0` vs
   `_desc-gnl_disp`: max |Δ| < 0.05 mm. `CreateGradientNonlinearityBMatrix -f b0 -g coeff.grad`
   vs `_desc-gnl_graddev`: max |Δ| < 1e-3. If either fails, the bug is in §2's frame
   paragraph or the Legendre normalization, and nothing downstream is meaningful until fixed.
9. qsiprep with `--gradient-file <out>_desc-gnl_coeff.grad` on the full simulation: its
   gradwarp-corrected DWI should match the `--gnl-no-warp` simulation to sub-voxel accuracy
   (same seed), and its graddev should match truth after the ACPC rotation.

End to end:

10. Reconstruct the corrected qsiprep output (DSI Studio GQI and MRtrix CSD), run
    `odx graddev`, and `odx compare` against the reconstruction of the `--gnl 0` simulation with
    the same seed: `mean_match_angle_deg` should fall from ≈ the field's median rotation to
    ≈ 0, stratified by distance from isocentre; `--gnl-scale` sweeps give the dose-response.
    Once the ODX ground-truth export (`docs/FORCE.md`, ODX design) exists, compare against the
    mixture's own peaks instead and drop the reference reconstruction.

## 6. Sizing

| piece | new/changed lines | notes |
|---|---|---|
| `src/gnl.rs` | ~450 | parser/writer, Legendre, field cache, warp, envelope |
| `src/compartments.rs` | ~80 | `signal_from_mixture_gnl` + shared `wm_response`, moving path |
| `src/bin/trxscan.rs` | ~100 | flags, isocentre affine shift, stage wiring, output writes |
| `src/io.rs` | ~40 | `write_4d` visibility, sidecar keys |
| tests | ~250 | items 1–7 above |
| `tools/validate_gnl.sh` | ~80 | items 8–9 in the container |

About three days of implementation plus a day of cross-tool validation. The single largest
risk is a silent frame or normalization mismatch with TORTOISE; item 8 exists so that risk is
retired before anything end-to-end is interpreted.

## 7. Experiments run against this implementation (2026-09)

Two loops were closed by hand; the scripts live outside this repo (qsiprep test-data outputs),
only the conclusions are recorded here.

**GDC × SDC frame consistency.** One object with a strong B0 field *and* GNL, both fieldmap
types emitted for it (`--gre-out`, pepolar pair), every qsiprep SDC path run with
`--gradient-file`; truth is the un-warped, undistorted simulation. Three findings that changed
the code:

- qsiprep's Siemens phase conversion maps the phasediff image's *min/max* onto −π..π, so a
  noiseless phasediff spanning a fraction of 0–4095 gets stretched (the first run produced a
  registered field 3× the truth). Hence `--gre-snr` noise on the echoes by default and the
  corner-voxel stamping at `--gre-snr 0`.
- On an **odd** acquired matrix the oversampled k-space stage placed the image half an acquired
  voxel off in-plane (`snx/2` is not `o·(nx/2)`). Fixed in `kspace.rs`, pinned by
  `oversampled_reconstruction_keeps_the_object_in_place`.
- The written `.bvec` was the scheme's RAS-world numbers, not FSL's voxel-frame convention with
  its determinant rule, so under FSL/MRtrix import every orientation came out one axis flip
  wrong. `Reorient::fsl_bvec` now converts for whatever grid is written.

**Streamline round trip.** Tissue *and* streamlines warped by a synthetic whole-body field
(`_desc-gnl_disp` on the tractogram, `_desc-gnl_invdisp` on the images), simulated, run through
qsiprep with the matching `coeff.grad`, and scored by the SyN b0 residual against the
uncorrected reference: the gradient-file correction halved the median geometric residual
(0.60 → 0.30 mm) and cut the 95th percentile to a third (1.76 → 0.68 mm). Tractography-level
metrics (TDI correlation) could not resolve the effect at whole-body-80 strength — it sits at
the CSD/partial-volume noise floor — so score GNL recovery geometrically, not by tracking.
