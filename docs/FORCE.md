# FORCE ground truth, and where the compartments come from

This covers the analytic ground-truth microstructure scalars TRXScan derives from the per-voxel
orientation mixture (the FORCE closed forms), and where the compartment fractions, fibre density,
and fixel dispersion come from. The dipy FORCE fork is the fixture oracle; CONSH is the
fibre-composition source.

**Implemented (§1–3).** `sphere.rs` + `mixture.rs` define the shared contracts;
`compartments::generate_mixture` / `signal_from_mixture` build the histogram-first signal stage
(per-streamline SIFT2 weights, global-κ Watson kernel); `microstructure.rs` is the full
closed-form port, validated against dipy at ≤1e-8 on 30 fixture mixtures
(`tools/gen_force_fixtures.py`, `tests/fixtures/force_moments.txt`); and `trxscan-microstructure`
writes 27 scalar NIfTIs — the 24 closed forms plus NODDI-style `icvf`/`odi`/`isovf` read directly
off the mixture (`odi` by Watson moment-matching of the orientation histogram). The `trxscan`
simulator (no-motion path) runs the same mixture, so signal and ground truth come from one object.

**Open (§4).** Per-fixel κ from CONSH fixels + the disp(κ) LUT (κ is currently one global value),
AFD-based fixel reweighting, the soma compartment in the default parameter set, and weights in the
motion re-simulation path.

## What FORCE is

- **`dipy.sims.force`** — a *forward simulator*: `generate_force_simulations` builds a large
  **library** of per-voxel signals from a parametric tissue model
  (`_force_core.create_mixed_signal` + `_multi_tensor_omp.multi_tensor`).
- **`dipy.reconst.force`** — a *reconstructor*: `FORCEModel` matches measured voxels to that
  library (argmax / posterior).
- **`dipy.sims._force_moments`** — the part that matters to us: **closed-form microstructure
  scalars computed analytically from a Gaussian compartment mixture**. No fitting, exact.

The forward model is a Gaussian multi-tensor mixture: 1–3 fibre populations with Bingham/Watson
dispersion, isotropic GM / free-water / optional soma-dot compartments, fractions and
diffusivities from priors.

---

## 1. Is TRXScan's signal model just ball-and-stick?

No. Per WM voxel it is **stick + zeppelin per fibre segment, plus two balls**
([`src/signal.rs`](../src/signal.rs), [`src/compartments.rs`](../src/compartments.rs)):

| Compartment | Model | Default |
|---|---|---|
| intra-axonal | `Stick`, `exp(-b·d·(f̂·g)²)` | `d_intra = 1.5e-3`, weight `intra_frac = 0.55` |
| extra-axonal | `Tensor` (zeppelin) | `d_extra = (1.5, 0.6, 0.6)e-3`, weight `extra_frac = 0.45` |
| GM | `Ball` | `d_gm = 1.2e-3` |
| CSF | `Ball` | `d_csf = 3.0e-3` |

Every one of these is a Gaussian compartment, which is the precondition for FORCE's closed forms.

Gaps against FORCE's compartment set: no **soma/dot** compartment (FORCE's `soma_frac`/`soma_d`,
worth borrowing — it keeps the closed forms exact), and `Tensor` currently implements only the
cylindrically-symmetric case `d2 == d3` (the `d2 != d3` kernel-frame rotation is a TODO in
`signal.rs`). Neither blocks the work below.

## 2. Multiple fibre populations: modelled correctly, then thrown away

**The signal is right.** `generate_compartments` accumulates *per segment*, not per voxel
([`compartments.rs:136`](../src/compartments.rs)):

```
fiber[vox][g] += length · seg_area · (intra_frac·stick(g, dir) + extra_frac·tensor(g, dir))
intra_vol[vox] += length · seg_area
...
fiber_resp(g) = fiber[vox][g] / intra_vol[vox]
```

That is a **path-length-weighted average over every segment orientation crossing the voxel**. A
90° crossing gives the correct two-population signal; fanning gives genuine dispersion; a
three-way crossing needs no special case. There is no orientation prior, no peak-count limit, and
no dispersion parameter — the geometry *is* the model. This differs from FORCE's parametric
sampling and covers cases its library does not (dense WM at high dispersion is absent from the
parametric library).

**The representation is lossy.** The gradient sum happens *at accumulation time*: only
`S(g)` and the scalar `intra_vol` survive. The set of (direction, weight) pairs — the mixture
itself — is never stored. So the per-voxel mixture cannot be recovered from a finished
signal-stage run, and FORCE's closed forms cannot be applied to it. That is the one thing that
has to change, and it is small.

## 3. FORCE scalars from the rasterized segments

FORCE's closed forms do **not** want a parametric dispersion model. They want orientation weights
on a sphere — which is precisely what the rasterizer already computes and then marginalizes away.
The API (`dipy/sims/_force_moments.py`):

```python
moments_from_odfs(intra_odf, extra_odf, verts, d_par, d_perp,
                  gm_frac, gm_d, csf_frac, csf_d, soma_frac=0.0, soma_d=1e-3)
    -> D_app (3,3), C (3,3,3,3)
```

`intra_odf` / `extra_odf` are per-vertex weights on `verts`, each summing to that compartment's
volume fraction. Downstream, all in closed form:

| Call | Gives |
|---|---|
| `dki_params_from_moments(D_app, C)` → `dki_scalars_from_params` | FA, MD, RD, AD, AK, RK, MK, MKT, KFA |
| `qti_indices_from_moments(D_app, C)` | µFA, coherence, `k_bulk` / `k_shear` |
| `mapmri_closed_form_indices(..., tau)` | RTOP, RTAP, RTPP, MSD, QIV |
| `ng_pa_from_odfs(...)` | NG, NGpar, NGperp, PA |
| `gqi_indices_from_odfs(odf, wm_frac)` | GFA, QA |

### The change to `compartments`

One extra accumulation next to `intra_vol[vox] += w`, into a per-voxel sphere histogram:

```rust
// v = index of the nearest vertex to normalize(b - a), antipodally folded
odf[vox * nvert + v] += length * seg_area;
```

Then export per masked voxel, with the fractions already computed in the normalization loop:

- `intra_odf = wf · intra_frac · odf/Σodf`, `extra_odf = wf · extra_frac · odf/Σodf`
- `gm_frac = gf`, `csf_frac = cf`, `d_par = d_intra`, `d_perp = d_extra.1`

`wf + gf + cf` is already normalized to 1 in `compartments.rs`, which matters: the from-odf
helpers **require** the mixture to sum to 1 per voxel (`_check_mixture_weights`, atol 1e-2) —
unlike `dki_params_from_tensor_distribution`, they do not renormalize for you.

### Three things to get right

1. **The no-streamline fallback needs its own mixture entry.** WM voxels with `intra_vol ≈ 0`
   fall back to a hindered isotropic signal `exp(-b·|g|²·MD)` ([`compartments.rs:232`](../src/compartments.rs)).
   That has no orientation content, so it must be exported as an isotropic Gaussian at `md`, not
   as a stick+zeppelin with an empty ODF — otherwise the scalars silently disagree with the
   signal exactly where the tractogram is sparse.
2. **Sphere resolution is a quantization error**, not a modelling choice. A 724- or
   1000-vertex sphere puts the binning error well below the tensor's angular scale. Reuse
   odx-rs's sphere so the export lines up with `cs-odf` / TRXViz conventions.
3. **These are ground truth for the signal stage only** — the noise-free, artifact-free mixture.
   That is the point: they are the answer key against which a reconstruction of the acquisition
   output is
   scored. Don't present them as ground truth *for the distorted DWI*.

### Shape of the deliverable

Make the histogram the **primary** object, not a side product: `generate_compartments` (behind a
flag, or a sibling function) accumulates segments → per-voxel orientation histogram (per-streamline
weights and the dispersion kernel of §4 both enter here), then computes the WM signal *from* the
histogram — `S(g) = Σ_v odf[v] · (intra_frac·stick(g, v) + extra_frac·zeppelin(g, v))` — then
exports it. Signal and ground truth derive from one object, so they cannot disagree by
construction. Cost shifts from O(hits·ngrad) to O(hits) + O(nvox_wm·nvert·ngrad) — comparable for
dense tractograms, and strictly better once a dispersion kernel enters (evaluating a kernel per
segment per gradient would be far more expensive). Memory is `nvox·nvert` floats: ~90 MB f32 at 362
hemisphere vertices on a 64³ grid (fold antipodally — `vv^T` and every downstream moment are
antipodally invariant).

The closed forms then turn the histogram into ground-truth NIfTIs — in Rust, per the sizing
below. Two scalar families need more than the mixture: MAP-MRI indices need `tau` from Δ/δ
(dipy's default gives normalized-units values — contrast only), and the scheme-dependent ones
(RSI, RISH) are fits, not closed forms — run them on the clean signal (which *is* the
mixture evaluated on the scheme) with existing tools.

### Porting the closed forms to Rust (sizing)

Porting the closed forms to Rust (rather than calling dipy) is bounded and dependency-free,
with dipy as the fixture oracle. Two facts make it tractable:

- The mixture is Gaussian, so ~90 % of the math is elementary linear algebra — contractions,
  3×3 determinants, one symmetric 3×3 eigendecomposition. The **only special function in the
  entire set** is Carlson's `R_F`/`R_D` (exact MK), and dipy implements both itself by the
  duplication theorem (`carlson_rf`/`carlson_rd`, `dipy/reconst/dki.py:77,151`, ~40 lines each)
  — no scipy dependency to replicate.
- dipy is BSD-3, so its source can be read and ported directly — none of the MRtrix-style
  clean-room constraint.

The port surface is `_force_moments.py` (937 lines including docstrings) plus the specific dipy
scalar functions it calls (`dti` scalars, `dki` analytic AK/RK/MK, `qti.QtiFit` invariants,
`odf.gfa`):

| Piece | Math | Difficulty |
|---|---|---|
| moments `D_app`, `C` | weighted `vv^T` / `vv^T⊗vv^T` sums | trivial (store the 15 unique symmetric components per vertex) |
| FA / MD / RD / AD | 3×3 symmetric eigh | trivial (analytic; extend `mat.rs`) |
| MKT, KFA | linear contractions of `W` | trivial |
| AK | `W` along `e1` in the eigenframe | easy |
| RK | Tabesh `G1/G2`, algebraic | easy–moderate (degenerate-λ branches) |
| MK | Tabesh `F1/F2` via Carlson `R_F`/`R_D` | moderate — the one special function |
| QTI: µFA, C_c, k_bulk, k_shear | Voigt 6×6 contractions | easy (√2 convention care) |
| RTOP / RTAP / RTPP / MSD / QIV | Gaussian marginals: dets + axis projections | easy (τ/unit care; `d_perp` floor for sticks) |
| NG / NGpar / NGperp | pairwise Gaussian-product inner products (3×3 dets) | moderate (O(k²) pairs — keep FORCE's top-k pruning) |
| PA | same + a 1-D solve for the isotropic reference scale | moderate |
| GFA / QA | ODF sample statistics | trivial |
| RISH / RSI | **not closed forms** — SH fit / NNLS on a scheme | skip; run on the emitted DWI |

Roughly 1–1.2 k LOC of pure-std Rust (no new dependencies, offline-testable — the crate's
default-build philosophy): a few focused days, versus an afternoon for the Python companion.
What the Rust route buys: **the interchange format stops being load-bearing.** Scalars stream
per voxel in-memory straight off the signal-stage histogram, one binary emits DWI + ground truth,
and the histogram export survives as an optional debug/reuse artifact rather than a required
pipeline stage. Python's role shrinks to one-time fixture generation.

Validation, fixtures first: a small script samples random mixtures (weights, directions,
diffusivities, fractions), runs `_force_moments` on them once, and checks inputs + outputs into
`tests/` as JSON; `cargo test` asserts ≤1e-8 relative, offline. The fiddly 20 % is exactly what
those diffs catch: dipy's conventions (the 15-element `kt` ordering, Voigt √2 factors, τ and 2π
q-space conventions, mm vs µm) and the degenerate-eigenvalue branches in MK/RK. Add physics
limits as native tests (isotropic mixture → MK = µFA = NG = 0; single stick+zeppelin → Tabesh's
single-tensor values; a 90° crossing → FA collapses while µFA holds), and cross-check Carlson MK
against numerical spherical averaging of directional kurtosis on a dense sphere.

---

## 4. Compartment fractions from CONSH

There are two different uses, worth not conflating.

Today the WM/GM/CSF fractions are three externally supplied NIfTIs, and the intra/extra split
inside WM is a global constant (0.55/0.45). CONSH fits all three tissues jointly from the DWI
itself.

### (a) Three-tissue amplitudes → per-voxel compartment fractions

CONSH builds each isotropic column as `Y₀₀·√(4π)·r₀`, so a compartment filling the voxel reads
**≈0.2821**. Divide `c_gm`, `c_csf`, and the WM
`l=0` term by 0.2821 and you have fractions of S0 — a drop-in replacement for the three tissue
maps, derived from a real subject instead of assumed. `consh --odx` already bundles WM SH + GM +
CSF + mask + peaks.

### (b) Quantitative AFD → per-fixel density weights

`consh --quantitative` multiplies the fitted coefficients by each voxel's b=0 signal, restoring
the property that fODF amplitude ∝ intra-cellular volume — the Apparent Fibre Density / SIFT2
property that per-voxel S/S₀ normalization destroys (Smith 2022; Dhollander 2021). Per-fixel peak
amplitude is then an **empirical apparent fibre density per population**, which is the same
quantity TRXScan currently approximates with `Σ length · π r²` from the tractogram.

That comparison exposes a real weakness in the current model. Because `fiber_resp =
fiber/intra_vol`, anything that scales *every* segment weight equally cancels exactly:
`fiber_radius_mm` does nothing, and neither does a global change in tractogram seeding density.
The absolute WM signal weight comes entirely from the supplied WM fraction map. What survives is
only the **relative** weighting between orientations *within* a voxel — and that is precisely
where tractography bias lives: if two bundles cross and one was seeded 10× denser, its
orientation gets 10× the weight, for reasons that have nothing to do with tissue. So the
tractogram contributes fixel-level density that is arbitrary in scale and biased in ratio, and
nothing in the pipeline currently corrects it.

This is the bias SIFT exists to remove, run in reverse: SIFT filters streamlines until their
density matches the FOD's fibre density, while the rasterizer takes *unfiltered* streamline
density and defines fibre density with it. The simplest fix is per-streamline weights, which
Fiberfox has (`fiberWeight`, `itkTractsToDWIImageFilter.cpp:1055–1073`) and the port had
dropped. Three ways in, cheapest first:

1. **Consume SIFT2 weights.** Accept per-streamline weights and accumulate
   `w_s · length · seg_area`. SIFT2's defining property is that `Σ w_s·length` is proportional
   to fibre volume per fixel — exactly the corrected density the rasterizer wants, consumed
   directly. `trx-rs` already loads TRX `dps` arrays (`TrxFile::dps(name)`), so
   `io::load_streamlines` just needs to surface one, defaulting to `w_s = 1`.
2. **Calibrate.** Regress the rasterizer's per-fixel `Σ length·area` against CONSH's per-fixel
   AFD on a real subject. The scatter — not the slope, which is unidentifiable — is the
   interesting part: it measures how far the tractogram's fixel density departs from the data's,
   per bundle, and gives a correction.
3. **Weight per fixel.** Match rasterized fixels to CONSH fixels by angle, then set each
   population's weight from the CONSH AFD rather than the path-length sum. The phantom then
   reproduces a real subject's fibre-density map while keeping the tractogram's geometry — which
   is exactly the ground truth FBA and SIFT2 validation lack.

**Caveats.** AFD is relative: it needs `mtnormalise` (hence `--quantitative` conflicting with
`--no-normalize`) and, cross-subject, a group-average response via `--read-responses-from`. And
AFD is a *density weight*, not a compartment fraction — it aggregates everything with high-b
signal in that fixel, so it does not directly give `intra_frac`. The intra/extra split still
needs a model or NODDI-style priors.

### (c) Fixel dispersion → the orientation spread streamlines can't provide

The third thing the FOD measures and the tractogram doesn't. Streamlines are smooth by
construction — curvature penalties, step-size regularization — so the within-voxel spread of
segment directions badly understates true axonal dispersion (histology puts ~18° in even the
"coherent" corpus callosum, Ronen 2014; NODDI ODI runs 0.2–0.5 across normal WM). A
segments-only phantom is unrealistically coherent: FA and µFA too high, the dispersion-sensitive
ground truth (kurtosis, NG) skewed, and reconstruction validation too easy — every method looks
sharp against a delta-function phantom.

Per-fixel dispersion is estimable from the FOD lobes: `fod2fixel`'s `-disp` is lobe integral /
peak amplitude, and odx-rs's FMLS segmentation computes both per lobe (`Lobe { integral,
peak_value, .. }`, exposed as `Lobe::dispersion()`). cs-odf and consh
now emit that ratio as a per-fixel `dispersion` field in the ODX (dpf), next to
direction/amplitude/QA — so a real subject's fODF gives us the Watson-κ source directly, by
inverting a `disp(κ)` LUT at the working lmax.

Substitution mechanics:

1. Tabulate `disp(κ)` once for a Watson distribution at the working lmax: sample Watson(κ) onto
   the sphere, project to SH, run FMLS, take integral/peak. Monotonic in κ → invert per fixel.
2. In the histogram accumulation, spread each segment's weight with its matched fixel's kernel
   instead of nearest-vertex binning: `odf[vox][v] += w · Watson_κ(v · dir)`. Segment→fixel
   matching by angle against that voxel's CONSH fixel table; unmatched segments fall back to a
   global (or per-bundle) κ prior.
3. Nothing downstream changes. Dispersed weights flow through `moments_from_odfs` untouched —
   this is FORCE's own construction (its library entries are Bingham-dispersed compartments
   sampled onto verts: `dispersion_lut` / `bingham_to_sf` in `dipy/sims/force.py`). And because
   the signal is computed from the same histogram (§3), the dispersion is *in the DWI*, not just
   in the answer key.

Caveats:

- **Measured disp = true dispersion ⊕ method blur.** The SH bandlimit alone gives a delta
  function a finite lobe width at lmax 8, and CONSH's `l ≥ 2` ridge *deliberately* broadens
  lobes for peak stability. Calibrate the floor by pushing known-κ Watsons through the identical
  pipeline (same lmax, same ridge settings) and inverting on the κ scale; for the estimation
  run, prefer a low-slack `--wm-ridge-anchor` so damping stays minimal where the data support
  sharpness.
- **Bounded circularity.** The phantom's dispersion truth inherits the estimator's blur — fine
  when the goal is realism, weaker when the goal is scoring dispersion *recovery* itself. For
  that, NODDI / Bingham-NODDI ODI is an independent, model-based source (voxel-level rather
  than per-fixel), or sweep κ parametrically and let CONSH's estimate be the thing under test.

### (d) Per-voxel parameter maps — spatially varying "integrity" (tier-2 design)

*Status:* not built. What exists instead are global signal-only knobs — `--tissue-s0` (per-
compartment amplitude) and `--diff-scale` (per-compartment diffusivity factor) in `trxscan` —
which scale the **simulated signal only**: `trxscan-microstructure` and `field_scalars` do not see
them, so the ground truth stays the preset's. Use them for matching a real scan's levels, never
for a phantom whose truth must reflect them.

With global `CompartmentParams`, the composition-driven scalars (µFA, MD, MSD, RTOP, QIV,
k_bulk) are constant wherever wf ≈ 1: the model contains exactly one white matter. That is a
*feature* for validation — a built-in null: any structure a fitted method shows in those maps
within deep WM is measurable artifact — but realism wants per-voxel parameters. The design:

- **The math is already general.** `VoxelMixture` takes `d_par`/`d_perp`/iso per call, and the
  fixtures vary them per case — the closed forms need no new validation. Only
  `field_scalars` flattens to global constants today.
- **Plumbing**: a `Param { Const(f64) | Map(Vec<f32>) }` enum with `at(vox)`; `MixtureField`
  carries a `ParamField` of them; the fallback diffusivity becomes per-voxel. ~20 lines in the
  scalar path.
- **Packaging**: the dormant `config` feature — a TOML where every compartment parameter is
  scalar-or-NIfTI-path (`intra_frac = "icvf.nii.gz"`).
- **Sources**: NODDI ICVF (qsirecon) for the intra/extra split; SMI / single-fibre DTI for
  diffusivities; CONSH amplitudes for fractions (§4a); or — most self-consistent — FORCE
  posterior maps from real fits (matched model family: stick + zeppelin + balls).
- **Cost lives in the signal path only**: per-voxel parameters break `signal_from_mixture`'s
  global vertex×gradient response table. Fix: evaluate over the voxel's **nonzero histogram
  bins** only (~20/voxel) — ~1e9 exp for a whole brain, seconds under rayon. No LUT needed.
- **The histogram is parameter-independent**, so one rasterization (the expensive step) can be
  swept over many parameter settings — including a flat-null run and a varying run of the same
  geometry.

**Tier 3 (per-bundle / along-tract, sized honestly):** parameters as TRX dps/dpg (or dpp for
along-tract) so integrity varies along bundles — the tractometry/FBA ground truth. The closed
forms already accept heterogeneous per-voxel compartment lists; the *accumulator* is what needs
generalizing (P parameter classes → P histograms wants the sparse-histogram refactor, or per-bin
sufficient statistics Σw·d, Σw·d²). Do tier 2 first; it forces the right shapes.

### The loop this closes

`cs-odf` already estimates RTOP/RTAP/RTPP/MSD/QIV/NG per voxel from real data. So:

```
CONSH (real subject)  ──fractions + AFD + κ──►  TRXScan phantom
                                                 │
                        FORCE closed forms ◄─────┤ (per-voxel mixture)
                                 │               │
                            ground truth         └──► simulated DWI ──► CONSH / cs-odf
                                 │                                          │
                                 └──────────── compare ─────────────────────┘
```

Simulate from a real subject's tissue composition, reconstruct the simulated DWI with the same
tooling used on real data, and score it against scalars that were computed analytically rather
than fitted.

## Summary

Keep streamline-driven signal generation — the crossing/branching geometry, and *which
population is which bundle*, only the tractogram knows. Borrow FORCE for **ground-truth
scalars** (retain the orientation histogram; closed forms ported to Rust, dipy as the fixture
oracle) and its
**soma/dot compartment**. Borrow from the FOD everything the tractogram can't supply:
**compartment fractions** (CONSH tissue amplitudes / 0.2821), **fibre density** (SIFT2 weights
or quantitative AFD — undoing the seeding bias the rasterizer otherwise bakes into the phantom),
and **fixel dispersion** (FMLS lobe integral/peak — restoring the orientation spread streamlines
smooth away). The division of labour: the tractogram supplies what the FOD can't measure, the
FOD supplies what the tractogram can't represent.
