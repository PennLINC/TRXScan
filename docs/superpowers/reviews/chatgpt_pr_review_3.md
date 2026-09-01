I reviewed the attached response and the current PR head, `e6be119f99d5d484cc6ecfa424612933492a8409`, four commits beyond the version I reviewed last time. The PR is still a draft and is mergeable.  The agent's response says all eight prior findings were addressed.

A lot of that is correct. The PR is materially better again. But I **still would not merge it yet**. I found three important issues that the response misses, including one fairly serious memory regression created by fixing my previous production-path blocker.

## What is genuinely fixed

The previous production-path regression is fixed correctly. The signal grid is now chosen **before** Stage A, and motion, SIFT2 weights, κ dispersion, myelin, and dropout operate on that selected grid rather than being applied to a nominal-grid `comp` that is later thrown away.

The PF scoring-axis issue is also substantially improved. Readout and PE axes are now explicit, `residual_alignment`, residual energy, and artifact amplitude are reported separately for RO and PE, and the independent scorer is now tested on the actual symmetric 2-D box on both axes.

The clean acquisition is now correctly used as the Gibbs-artifact template for both clean and noisy runs, and summaries no longer average clean/noisy and Hann/unwindowed conditions together.

The PF mask work itself is also good: the Fiberfox behavior is now explicitly named rather than masquerading as exact 6/8, and a genuinely contiguous mode has been implemented and tested for exact line count and contiguity.

And CI really is green now: both the Python and Rust jobs passed on the current head.

Those are substantial improvements.

# 1. New merge blocker: the memory analysis is now stale because the production signal path changed

This is the biggest new problem.

The response says lowering the production default to `o=2` is the interim memory-safe answer, based on the measured `13.04 GB` `o=4` run and the f64 `generate_compartments` accumulator.

But after fixing my previous review item #1, **the normal no-motion production path no longer uses `generate_compartments`**.

It now uses:

```rust
generate_mixture(...)
signal_from_mixture(...)
```

on the **oversampled signal grid**.

And `generate_mixture()` allocates a dense:

```rust
vec![0.0f64; nvox * nvert]
```

orientation histogram.

The sphere has **321 vertices** at level 3.

For the HBCD dimensions quoted in the PR:

$$
107\times151\times104
$$

the `o=2` grid is approximately:

$$
214\times302\times104
=
6.72\text{ million voxels}.
$$

The histogram alone therefore costs approximately:

$$
6.72{\rm M}\times321\times8
\approx17.3\ {\rm GB}.
$$

And at the end of `generate_mixture`, that f64 histogram is converted into an f32 ODF:

```rust
odf: hist.iter().map(|&x| x as f32).collect()
```

while `hist` is still needed to construct the output, so peak memory around that conversion can approach another:

$$
6.72{\rm M}\times321\times4
\approx8.6\ {\rm GB}.
$$

That is roughly **26 GB just for `hist + odf`**, before the compartment images and other arrays.

At `o=4`, the corresponding dense histogram is about **69 GB f64**, with another roughly **35 GB f32 ODF**.

So the previous 13-GB `o=4` measurement is no longer representative of the production path after this latest fix. It measured the earlier signal-stage implementation.

This also means the current preflight warning is looking at the wrong dominant allocation. It estimates the three f32 compartment volumes.  At `o=2`, that estimate stays under its 8-GB warning threshold even though `generate_mixture()` alone may require >20 GB.

### Required action

I would **remeasure peak RSS on the current head**, through the actual default no-motion/histogram-first path, at `o=1`, `o=2`, and if possible `o=4`.

I expect that measurement to change the default decision again.

More fundamentally, the dense `nvox × 321` histogram probably needs one of:

* z-slab streaming;
* sparse per-voxel orientation storage;
* processing only relevant masked voxels;
* a lower-memory representation;
* or a separate lower-memory signal-only path when weights/κ/microstructure truth are not needed.

Until that is resolved, I would not call `o=2` a memory-safe production default.

The parallel `generate_compartments` concern from the response is real too—it creates a full `nvox × ngrad` f64 buffer per Rayon group.  But for the ordinary no-motion path, the dense mixture histogram is currently an even more immediate problem.

---

# 2. The new scanner-like PF mode is still not actually used by the benchmark

This is the clearest mismatch between the response and the implementation.

The response correctly says:

> `Contiguous` keeps exactly `round(ny*pf)` consecutive lines — what a scanner produces, and the more relevant condition … for PF-aware reconstruction.

The Rust test says essentially the same thing:

> “Contiguous mode is what a scanner produces and what a PF-aware unringing benchmark should default to comparing against.”

But `trxscan-benchmark` constructs each acquisition as:

```rust
Acquisition {
    partial_fourier: p.partial_fourier,
    ...
    ..Acquisition::default()
}
```

and `Acquisition::default()` sets:

```rust
pf_mode: PartialFourierMode::FiberfoxCompatible
```

So the benchmark directories named:

* `pf68`
* `pf78`

are **still using the Fiberfox-specific noncontiguous masks**, not scanner-like 6/8 and 7/8.

The factor grid itself continues describing 0.75 as “6/8.”

That means the PF acceptance numbers being discussed in the response are still not the scanner-like PF experiment that motivated adding `Contiguous` in the first place.

### Required action

For the main Gibbs-unringing benchmark I would explicitly set:

```rust
pf_mode: PartialFourierMode::Contiguous
```

and record `pf_mode` in `factors.json`.

There are two defensible extensions:

1. **Best for the current scientific question:** use Contiguous for `full / 7/8 / 6/8`; separately test Fiberfox-compatible behavior as a robustness condition.
2. Add PF mode as a factor, so the suite explicitly compares both reconstruction masks.

I would not continue calling the existing Fiberfox-mask conditions simply `pf68` and `pf78`.

There is a related production question: the main CLI describes its acquisition as HBCD-like but explicitly hardcodes `FiberfoxCompatible`.  I would probably make the HBCD-like CLI use `Contiguous` and expose Fiberfox compatibility as an option. At minimum, expose a `--pf-mode` argument; currently the newly added mode is not selectable from either user-facing binary.

---

# 3. CI is green, but it still does not compile the code path that was the main subject of the last two reviews

The CI repair itself is good. The sibling path dependencies are now checked out, and the Python environment correctly tests the scorer without accidentally requiring nibabel.

But the Rust job runs:

```text
cargo build
cargo test
```

with **default features only**.

TRXScan has:

```toml
default = []
```

and the three binaries—including `trxscan`—all have:

```toml
required-features = ["cli"]
```

Therefore the green Rust CI job **does not compile `src/bin/trxscan.rs` at all**.

That is particularly problematic here because the biggest bugs in my previous reviews were specifically in the wiring of that binary.

### Required action

Add at least:

```text
cargo check --features cli --bin trxscan
cargo check --features cli --bin trxscan-benchmark
```

I would also compile:

```text
cargo check --features cli,par --bin trxscan
```

because the parallel signal-stage path has materially different memory/code behavior.

You can keep the fast pure-std `cargo test` job. But the production binary must be part of CI for this PR.

---

# 4. The PF acceptance suite is more axis-aware, but not fully axis-aware yet

The response fixes my main axis criticism for `residual_alignment`, `residual_energy`, and `artifact_norm`. Good.

But the other acceptance metrics remain single-axis:

* `oscillatory_residual`;
* `edge_sharpness`;
* `edge_location_bias`.

`score()` still auto-selects one axis for those.

For the symmetric box, that defaults to readout.

This causes two remaining problems.

First, acceptance rule **(a)** still requires methods to beat the no-op control on `oscillatory_residual` for PF data.  But the same file explicitly explains that this Nyquist metric is **not valid as a PF metric**.

I would scope rule (a) to **full Fourier only**.

Second, the anti-blurring rule **(c)** uses the same auto-axis `edge_sharpness`.  A method could blur specifically along phase-encode while leaving the readout edge sharp, and the PF acceptance suite would not catch it.

I would add:

* `edge_sharpness_ro`
* `edge_sharpness_pe`
* ideally `edge_location_bias_ro`
* `edge_location_bias_pe`

and use the PE versions for PF acceptance.

This matters especially because the response's own measurement says zero-filled PF trades some ringing for **PE-axis blur**.

---

# 5. Noisy residual-energy rows are still unsuitable as an artifact-amplification gate

Using the **clean** acquisition as the template for noisy runs fixed the alignment problem.

But:

$$
\|R_m\|
$$

for a noisy method output still contains thermal noise.

Yet acceptance rule (b) applies:

```python
if e > 1.5:
    fail
```

to clean **and noisy** rows.

That makes `residual_energy_pe` simultaneously a Gibbs metric and a noise-energy metric.

I would run the actual Gibbs-removal acceptance rules on `noisy=False`.

Use the paired noisy rows for separate questions such as:

* noise amplification;
* robustness;
* interaction with unringing;
* whether a method converts noise into structured residual.

This would make the interpretation much cleaner and removes the need for the somewhat arbitrary `1.5×` cushion.

---

# 6. The newly important PF metrics are not independently cross-checked

The second scorer remains a very good idea.

But the independent implementation currently recomputes only:

* Nyquist oscillatory amplitude;
* residual alignment.

The new quantities that now participate in PF acceptance:

* `residual_energy_ratio`;
* `artifact_norm`;

do **not** have independent implementations.

The acceptance tests exercise synthetic values for them, but that only tests the acceptance branching, not whether the metrics themselves are correct.

Given the history of scorer bugs in this PR, I would extend the naive scorer to independently calculate both quantities, particularly on the 2-D PF box.

I would consider this important but not necessarily a standalone merge blocker if the simpler direct unit tests are added.

---

# 7. One subtle PF-scoring issue remains: only one of the two box edges is measured

Each of `residual_alignment`, `residual_energy_ratio`, and `artifact_norm` finds:

```python
peak = argmax(...)
```

and scores one window around that single edge.

But the box has two edges per axis, and a one-sided PF transfer function is asymmetric/complex. The two PE edges need not have equivalent sidelobes.

For full Fourier this is mostly redundant because of symmetry. For partial Fourier, I would measure **both PE edges**, ideally separately and jointly.

The unused `_edge_mask()` machinery already points toward a more general multi-edge implementation.

I would not necessarily block this PR solely on that, but it is worth fixing before treating the PF benchmark as a definitive evaluation framework.

---

# Judgment call: Hann-windowed data

I agree with the agent's decision **not to make Hann-windowed performance of Kellner-style methods a hard PF acceptance criterion**. The agent reports that `mrdegibbs` amplifies the PE artifact substantially under Hann apodization and reasonably interprets that as out-of-domain behavior rather than evidence that the simulator is wrong.

I would keep:

* the Hann conditions;
* the quantitative results;
* the prominent warning.

I would not use them to reject the simulator or establish a ranking between methods designed around rectangular truncation.

There is a minor inconsistency in the current implementation, though: only rule (b) explicitly skips non-`None` windows; rules (a) and (c) can still fail a Hann row.

So either:

* make *all* Hann method-performance checks descriptive/out-of-domain; or
* say specifically that only the PF artifact-amplification assertion is waived.

The test named `test_apodized_rows_are_out_of_domain_and_never_fail` currently promises more than the implementation actually guarantees.

---

# Smaller cleanup

A few things should also be corrected:

* The `residual_alignment()` docstring still says `0.0` means the artifact is “entirely gone,” despite the response saying this wording was corrected. The new energy-ratio docstring correctly explains that zero alignment can merely mean orthogonality.
* `available_methods()` can still report `rpg` as available if an RPG executable/module is detected, but `run_method("rpg")` **always** raises `MethodUnavailable` because there is no adapter.  Until the adapter exists, RPG should not be returned by `available_methods()`.
* The agent correctly notes that the preparation workflow still does not produce a simulation-grid myelin map.  The silent-loss bug is fixed—the CLI now errors if the myelin grid is wrong—but the default oversampled workflow is still awkward for `--myelin`.

## Revised verdict

The response is **mostly technically strong**, and most of my previous review was addressed correctly. I no longer have concerns about the core finite-Fourier transform, phase-model integration, clean/noisy pairing, or the basic per-axis PF scoring idea.

But I would still **request changes**, primarily for three reasons:

1. **The current memory analysis no longer describes the current production signal path.** The dense 321-direction mixture field makes even `o=2` potentially a ~25+ GB operation before the DWI compartment arrays are considered.
2. **The new scanner-like PF mode is not actually used anywhere in the benchmark that is reporting “6/8” and “7/8” results.**
3. **Green CI still does not compile the `trxscan` binary whose wiring caused the major previous regressions.**

After those three are fixed, I would address the remaining PF acceptance details—PE-axis sharpness, clean-only artifact gating, both-edge scoring, and independent checks of the new energy metrics. At that point I think this will be very close to an approvable implementation rather than another architectural iteration.
