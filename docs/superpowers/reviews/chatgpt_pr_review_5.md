I reviewed the agent's response and the current PR head, `9c2c452b3cab91dd62ed9cfb0f31e470ac71a5fc`. The response says all three round-4 items and cleanups were addressed.  The current GitHub CI is also fully green, including Python tests, Rust tests, both CLI binaries, the `cli,par` build, and Clippy.

This is now **very close to approval**, and the three things I asked for last time are mostly fixed correctly. I still found two substantive issues plus two validation/documentation items that I would fix before merging.

## What is now closed

The memory warning really does run **before Stage A** now, and the no-motion estimate uses the dense `nvox × 321` histogram + f32 ODF rather than the obsolete compartment-only estimate. The CLI help also now correctly says that 16 GB is tight/data-dependent and removes the dangerous old “o=4 needs ~13 GB” advice.

The dark-side reasoning is also correct. The primary complex residual metrics now include both sides of the edge, while phase RMSE remains magnitude-masked; this is the right conceptual separation. `residual_alignment` also finally documents correctly that zero means orthogonality, not necessarily zero residual.

And the acceptance policy is internally much cleaner: rules (a), (b), and (c) now all use clean rows; the full-Fourier rule is limited to full Fourier; Hann input is out-of-domain; and the negligible-artifact floors are calculated from the populations to which their rules actually apply.

So those previous blockers are closed.

# 1. The motion-path memory pre-flight is still modeling the wrong implementation

This is the clearest remaining production bug.

The new warning says that with motion the dominant allocation is:

> `nvox × ngrad` f64 signal accumulator, “×2 live under --features par”

and its arithmetic uses:

$$
8\,n_{\rm vox}n_{\rm grad}
+
12\,n_{\rm vox}n_{\rm grad}
=
20\,n_{\rm vox}n_{\rm grad}\ {\rm bytes}.
$$

But `generate_compartments_moving()` does **not** contain such an `nvox × ngrad` f64 accumulator.

For each gradient/volume it actually allocates:

* four `nvox` f32 resampled tissue/mask arrays;
* two `nvox` f64 arrays, `fiber` and `ivol`;
* three `nvox` f32 output arrays.

It then **collects the three f32 arrays for every volume** into `vols`. After all volumes have been generated, it allocates another three complete `nvox × ngrad` f32 output arrays and copies the collected volume data into them.

Therefore, during the final copy alone you have approximately:

$$
2\times
3\times n_{\rm vox}n_{\rm grad}\times4
=
24\,n_{\rm vox}n_{\rm grad}\ {\rm bytes},
$$

before counting tissue, streamlines, masks, and other resident memory.

Under `par`, you additionally have multiple concurrently active per-volume working sets. But there is **not** a pair of full `nvox × ngrad` f64 accumulators.

So the current pre-flight:

* describes the wrong data structure;
* underestimates even the unavoidable `vols + final images` peak by about 20% (`24` vs `20` bytes per voxel-volume);
* does not account for concurrent per-volume working arrays.

### Fix

Model the actual lifecycle of `generate_compartments_moving`.

At minimum:

$$
M_{\rm collected+final}
\approx
24\,n_{\rm vox}n_{\rm grad}
$$

plus a serial or parallel worker allowance.

Longer term, the moving path could avoid this duplication by writing each volume directly into the final arrays, but the immediate requirement is simply that the pre-flight describe and estimate the code that actually runs.

The design document has inherited the same incorrect statement about the motion path and should be corrected with it.

---

# 2. The new “all edges” detector is counting partial-volume gradient plateaus as multiple physical edges

This one matters to the benchmark metrics.

The benchmark object is a rectangle: there are four physical sides total, hence **two physical edges along any one scoring axis**.

But the response says that under 6/8 PF the PE-edge biases are:

> `[+0.25, +0.49, -0.49, -0.25]`

and calls those four PE edges.

That is a clue that the peak detector is splitting each physical edge in two.

The code identifies an edge wherever a gradient sample is above threshold and:

```python
summed[i] >= local_window.max()
```

For this benchmark the edges are deliberately placed at half-voxel positions. After block averaging, a profile around a physical edge is approximately:

$$
0,\ 0.5,\ 1,
$$

so its discrete gradient contains:

$$
0.5,\ 0.5.
$$

Both adjacent samples satisfy `>= local max`.

Thus one physical boundary becomes two “peaks.” That explains why a rectangle with two PE boundaries yields four reported PE-edge shifts.

This affects more than `_edge_shifts()`: `_sidelobe_indices()` uses essentially the same peak definition, so the primary artifact masks also have widened/duplicated guard regions around these artificial double peaks.

### Fix

Treat a plateau or adjacent cluster of high-gradient samples as **one physical edge**.

For example:

1. threshold the summed reference gradient;
2. identify connected peak clusters;
3. represent each cluster by its gradient-weighted centroid;
4. construct the guard/sidelobe window around that centroid.

For the benchmark box, add a hard regression assertion:

> two detected edges along RO and two along PE.

That would have caught the current issue immediately.

---

# 3. The scalar edge-location bias still cancels real geometric errors

Even after physical-edge clustering, I would change the aggregate metric.

`_edge_shift()` currently returns:

```python
mean(per_edge_signed_shifts)
```

The response's own values illustrate the problem perfectly:

$$
+0.25,\ +0.49,\ -0.49,\ -0.25
$$

average to approximately zero.

So the headline `edge_location_bias` says “no bias” despite every detected edge moving substantially.

There are two different quantities here:

* **signed mean shift** → global translation;
* **edge-position error** → geometric fidelity.

For the benchmark's error decomposition, I think the second is the important quantity.

I would report something like:

$$
{\rm edge\ location\ RMSE}
=
\sqrt{\frac1N\sum_i\Delta x_i^2}
$$

or mean absolute displacement as the primary scalar, while retaining:

* per-edge signed shifts;
* optional signed mean shift.

The newly exposed per-edge values are useful, but the current scalar remains misleading.

---

# 4. The most important dark-side fix does not yet have a targeted semantic regression test

This is important because the independent scorer did **not** protect against the old defect.

Before this round, both the primary and naive scorers used the same bright-side-only interpretation. They agreed with one another while both were wrong.

Now both have been changed to the both-side interpretation.  Agreement still only proves implementation agreement, not the semantic property that motivated the change.

The current scorer unit tests include alignment zero/half/one, global rotation, amplification, phase masking, etc., but I do not see a test for the exact round-4 failure mode.

I would add one very explicit fixture:

* construct a complex ringing control around an edge;
* construct a method result that removes **only the bright-side sidelobes** while leaving dark-side ringing unchanged;
* assert that the primary full complex residual-energy score remains substantially nonzero;
* assert that the bright-only secondary metric approaches zero.

That directly pins the physical reason for this change.

The response says the cross-check confirms the new behavior, but because both implementations were changed together, this targeted analytic test is the stronger protection.

---

## I would also rerun the actual method acceptance suite

This is not a code defect, but it is important before declaring the benchmark finished.

The definition of the primary PF artifact metrics changed materially—from bright-side only to both-side complex residuals. The response reports 83 Rust tests and 60 Python tests and describes one real-fixture dark-side measurement, but it does not report a fresh full `mrdegibbs`/DIPY acceptance run after this change.

CI cannot substitute for that because those external unringing tools are deliberately absent from the self-contained Python job.

I would rerun the 24-fixture acceptance suite and record at least the clean/unapodized results for:

* no-op;
* `mrdegibbs`;
* DIPY;

using the new **both-side**, contiguous-PF metrics.

There is no need to demand a predetermined ranking. I just want to confirm that the newly defined acceptance rules still behave sensibly on the actual methods they were designed around.

---

## Small documentation cleanup

The spec is much more internally consistent now: `o=2` is correctly described as the shipped, memory-driven default while `o=4` is the accuracy target.

One stale sentence remains, though. Section 6 still says:

> "`o = 2` therefore fits a 16 GB machine"

whereas both the response and CLI now correctly say 16 GB is **tight and data-dependent**.

I would make the spec use the same cautious language.

There is also a tiny docstring typo in `acceptance.py`:

> `**Partial Fourier is now covered by **`

with the metric name missing.

## Verdict

This round did fix the three things I asked for last time, and **the core Gibbs forward model remains in a state I am comfortable with**. The live CI on the exact current head is green.

I would still give **REQUEST CHANGES**, but this is now a very narrow request:

1. correct the motion-path memory estimate to reflect the actual `generate_compartments_moving()` allocation lifecycle;
2. cluster adjacent gradient plateaus into one physical edge;
3. make the scalar edge-position error non-cancelling, while retaining signed per-edge values;
4. add the explicit dark-side regression test;
5. rerun the real-method acceptance suite after the metric-definition change.

I do **not** see a reason to reopen the Fourier model, PF acquisition implementation, phase model, noise model, or benchmark architecture. Once those measurement/guardrail issues are fixed, I expect the next review to be approval rather than another redesign cycle.
