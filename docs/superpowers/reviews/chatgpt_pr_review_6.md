I reviewed the agent’s round-5 response and the current PR head, `a62964de1eaa1865af28e5cc7fda4a5b568afdba`. The PR remains open, draft, and mergeable.  The response reports that all five prior items were addressed and that the real-method acceptance suite was rerun with the new both-side metrics.

The round-5 changes are mostly correct, and CI on the **exact current head** is fully green, including the production binaries and `cli,par` path.  I nevertheless found **one important new scorer bug** that I would fix before merge, plus two narrower regression-test issues.

## What is now satisfactorily fixed

The motion-path memory estimate now describes the implementation much more faithfully. The CLI recognizes that `generate_compartments_moving()` collects three f32 volumes for every gradient and later allocates another three full 4-D arrays, giving the `24 × nvox × ngrad` floor, plus a worker allowance.  That matches the actual lifecycle in `generate_compartments_moving()`.

The non-cancelling geometric metric is also implemented as requested: `_edge_shift()` is now an RMS of the per-edge signed displacements, while `_edge_shift_signed()` retains the global signed mean.  The score output exposes RMS, signed shift, and per-edge values independently.

The targeted dark-side test is a real improvement. It constructs exactly the semantic failure mode we discussed—a method that corrects only the bright-side sidelobes—and verifies that the bright-only score looks near-perfect while the primary both-side residual remains large.  This is much stronger than relying on two scorers that could share the same conceptual error.

The reported new real-method results are also encouraging. The agent reports both DIPY and `mrdegibbs` degrading monotonically as PF becomes more aggressive under the new PE-axis alignment/energy metrics, with the no-op control fixed at 1.0.  Those PF results look much more interpretable than the earlier metric behavior.

# 1. Merge blocker: the full-Fourier Nyquist metric cancels the two box edges against each other

This is the most important finding.

`_nyquist_amplitude()` now gets `_sidelobe_indices()` covering **all significant edges**, then computes one projection:

```python
proj = (w[good] * alt).sum(axis=-1) / n[good]
```

across the union of those sidelobes.

But the benchmark object is a centered rectangle with **two opposing edges along each axis**. At the default matrix of 64, the edges are separated by 32 acquired voxels.

For full Fourier:

* the rising edge and falling edge have opposite-polarity Gibbs residuals;
* their separation is even, so the global `(-1)^x` Nyquist sequence has the same parity at the two edges;
* therefore their signed Nyquist projections tend to be **opposite**.

Summing them before taking the magnitude can make a strongly ringing symmetric box score nearly zero.

That defeats the purpose of the full-Fourier acceptance rule, because rule (a) still uses `oscillatory_residual` to decide whether a method beats its no-op control.  In particular, the no-phase and constant-global-phase fixtures are the conditions most susceptible to this symmetry cancellation.

### Required fix

Compute the Nyquist projection **separately for each physical edge**, take its complex magnitude, and only then aggregate across edges/profiles, e.g. RMS:

$$
G =
\sqrt{
\frac{1}{N_{\rm edges}}
\sum_e |c_e|^2
}.
$$

Do not concatenate both edge windows into one signed projection.

The same change must be made independently in `naive_oscillatory()`; it currently performs the same union-and-sum operation, so the two implementations can agree while both cancel the artifact.

I would add a semantic test on the symmetric full-Fourier box asserting that:

* the uncorrected box has a clearly nonzero Nyquist artifact score;
* its two individual edge projections are both nonzero;
* combining the edges does not cancel them.

This issue means I would **not yet treat the reported overall `ACCEPTANCE: PASS` as conclusive**, although the reported PF alignment/energy results remain useful.

# 2. The new edge-clustering regression test does not actually reproduce the bug it claims to pin

The response says the new test uses “the actual benchmark box” and verifies that its half-voxel edges produce two physical edges rather than four.

But the Python `_box()` helper does something different.

It creates:

```python
obj[q:3*q, q:3*q] = 1
```

and for the reference returns:

```python
obj[::o, ::o]
```

That is a **voxel-aligned, point-sampled** box.

The actual Rust benchmark deliberately places the boundaries at:

```text
(q + 0.5) * o
(3q + 0.5) * o
```

and `object_nominal` is a **block mean**, producing the `0, 0.5, 1` partial-volume transition that caused the double-gradient plateau in the first place.

So the new test:

> `test_the_box_has_exactly_two_edges_per_axis`

does **not** reproduce the failure mode. A voxel-aligned sharp edge already has only one nonzero gradient sample, so the old local-maximum detector would also have found two edges.

### Required fix

Make the test fixture match `trxscan-benchmark`:

* same `+0.5` acquired-voxel edge locations;
* same high-resolution construction;
* complex/block **mean**, not `obj[::o]`.

Then assert both:

* exactly two physical edges per axis;
* the expected fractional centroid locations.

This is a small change, but it is important because this regression test currently gives false confidence.

# 3. Fractional edge centroids are immediately thrown away with Python `round()`

`physical_edges()` now correctly returns floating-point weighted centroids, but both `_sidelobe_indices()` and `_edge_shifts()` immediately do:

```python
int(round(c))
```

Python uses **round-half-to-even**.

For the default 64×64 benchmark, the half-voxel edges happen to map to indices for which this lands where intended. But for another valid matrix size or an edge whose partial-volume voxel has the opposite parity, a centroid such as `2.5` becomes `2`, whereas the partial-volume voxel is `3`.

`trxscan-benchmark` accepts arbitrary matrix sizes, so this is not purely hypothetical.

I would either keep the centroid as a float throughout the window-distance calculations or use an explicitly defined coordinate conversion rather than Python's `round()`. If integer mapping is genuinely intended, `floor(c + 0.5)` at least has deterministic half-up semantics.

A regression test with an odd-parity partial-volume voxel would pin this.

## 4. Edge sharpness still has a one-edge blind spot

This is less severe than the Nyquist cancellation, but it is the same structural pattern.

`_sharpness()` returns the **maximum** absolute gradient in each profile.  A box profile contains two physical edges.

Therefore a method can substantially blur one edge while leaving the other sharp, and the profile's peak gradient remains dominated by the untouched edge. Rule (c), which is intended to prevent a method from “winning by blurring,” can miss that asymmetric failure.

Now that the scorer explicitly acknowledges one-sided PF asymmetry and has machinery for physical edges, I would calculate edge sharpness per physical edge too. For acceptance, either use the worst edge or require all edges to remain above the floor.

I regard this as **important**, though if you want to keep the merge gate narrow, I would rank it below the Nyquist cancellation and the ineffective regression test.

# 5. Motion memory accounting is now structurally correct, but the warning text still assumes the no-motion histogram path

For the motion path, the arithmetic is now conservative and much better.

But the same warning message then says:

> “Committed memory is DATA-DEPENDENT and usually lower — the histogram faults in only where streamlines deposit…”

even when `--motion` selected the path that has **no histogram**.

That explanation only applies to the no-motion `generate_mixture` path.

For motion, the large final f32 arrays are not governed by histogram page sparsity in the same way. I would branch the explanatory text just as the arithmetic is already branched.

Also, when the `par` feature is enabled, `available_parallelism()` is being used as the worker count. A more exact estimate would use Rayon's actual pool size, particularly if `RAYON_NUM_THREADS` is configured. This is a minor operational improvement, not a merge blocker.

# 6. The design document still contains stale contradictions

The round-5 response correctly fixed the “16 GB is safe” wording.  The spec now says 16 GB is tight and data-dependent.

But several older statements remain.

Within the same section, the memory discussion says z-slab streaming and sparse storage are now **warranted**, then the runtime paragraph says:

> “z-slab streaming … [is] not implemented … the measurement does not [demand it].”

It also says Stage A memory “remains unmeasured end-to-end,” immediately after documenting the 11.46-GB current-path Stage-A measurement.

The decision log likewise still says:

* oversampling factor is tolerance-driven, rather than distinguishing the **memory-driven shipped `o=2`** from the **accuracy-driven `o=4` target**;
* z-slab streaming is deferred because of YAGNI, although the body now explicitly says memory warrants it.

Because this document is being retained as the authoritative design record, I would clean those up before declaring the PR finished.

There is also a tiny typo in the acceptance module docstring:

> ``residual_energy_pe`**, , a frequency-agnostic…`

## Assessment of the reported acceptance rerun

The new reported results themselves look sensible:

| Method              |    FF | 7/8 PF | 6/8 PF |
| ------------------- | ----: | -----: | -----: |
| DIPY alignment      | 0.204 |  0.244 |  0.480 |
| DIPY energy         | 0.683 |  0.767 |  0.822 |
| mrdegibbs alignment | 0.379 |  0.433 |  0.607 |
| mrdegibbs energy    | 0.494 |  0.557 |  0.720 |

Those values and the monotonic PF degradation are reported in the agent's response.

I think the **PE-axis residual alignment/energy portion** of that result is meaningful. My concern is specifically that the separate full-Fourier `oscillatory_residual` acceptance criterion can cancel across the two opposite box edges.

It would also be useful to record the exact DIPY and MRtrix versions used for that acceptance run, since those external tools are not installed in CI and the numerical table is therefore not independently reproduced by the current workflow.

## Verdict

I would still choose **REQUEST CHANGES**, but the reason is now very narrow.

The round-5 response successfully closes the five issues from my previous review. The core TRXScan forward model, complex phase, PF acquisition, noise model, production integration, and PF-aware residual metrics are not the problem. CI is green on the exact current head.

Before approval I would fix:

1. **Nyquist scoring per physical edge, before aggregation**—this is the actual blocker.
2. Make the edge-clustering test use the **real half-voxel/block-mean benchmark reference**.
3. Remove parity-sensitive `round()` of fractional edge centroids.
4. Preferably make sharpness per-edge as well.
5. Clean the remaining memory/spec wording.

Once the Nyquist cancellation is fixed and covered by a semantic box test, I would expect this PR to be ready for approval rather than needing another architectural review.
