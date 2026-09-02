I reviewed the agent's response and the current PR head, `222dfb91100af390c5260ebb884b521345ad4f1d`. The PR is still a draft and mergeable.  The response accurately describes most of the changes it made.

The situation is much better than the previous round. **Contiguous scanner-like PF is now really used by the benchmark and is the HBCD-like CLI default, the scorer now handles both axes and multiple edges, the new PF quantities have independent implementations, and CI now actually compiles the production binaries and the parallel path.** The current CI run passed both jobs, including those new binary checks.

I would nevertheless **request one more changes round**. There is one production blocker and two benchmark-validity issues I would resolve before merge.

1. **The new memory “pre-flight” still occurs after the memory-intensive operation it is supposed to protect.** The response correctly remeasured the current `generate_mixture` path at 11.46 GB for the HBCD-sized `o=2` case and correctly identified the much larger dense histogram/ODF upper bound.  But in `trxscan.rs`, `generate_mixture()` and `signal_from_mixture()` execute first; only later, immediately before Stage B, does the code calculate `hist_gb`, `images_gb`, and print the warning.  An allocation failure can therefore occur before the warning is ever reached. Move the calculation to immediately after the simulation grid and gradient scheme are known, **before Stage A begins**. It should distinguish the no-motion mixture path from the motion path because their dominant allocations differ.

   I would also be more cautious about saying "`o=2` fits a 16 GB machine." The arithmetic upper bound for that example is about 25.9 GB of histogram+ODF plus 6 GB of compartment images; the 11.46-GB measurement depends substantially on sparse page commitment for that particular tractogram/mask.  `o=2` can remain the pragmatic default, but the warning should precede allocation and should describe 16 GB as **tight/data-dependent**, not generally safe.

   There is also now dangerous stale CLI help: it still says the `o=4` run peaked at 13 GB and recommends using `o=4` with ≥32 GB, even though the response correctly says that measurement was from the obsolete signal path and the new `o=4` dense bound is roughly 104 GB.

2. **The complex Gibbs metrics still throw away the entire dark-side ringing pattern.** This is the most important scientific issue I found this round. `_nyquist_amplitude`, `residual_alignment`, `residual_energy_ratio`, and `artifact_norm` all restrict their masks to something equivalent to:

   `ref > 0.5 * max(ref)`.

   The justification in `_nyquist_amplitude` says negative ringing is rectified because “these are MAGNITUDE images.”  But that is no longer true of the quantity actually being scored: `score()` explicitly reconstructs complex data as `mag * exp(i*phase)`, and the residual/template metrics operate on those complex arrays.  A negative real Gibbs lobe stored as positive magnitude with phase ≈π is recovered as a negative complex value. The dark-side complex residual is therefore perfectly meaningful even though the **reference phase** at zero magnitude is not.

   This creates a concrete blind spot: a hypothetical method that perfectly corrects Gibbs on the bright side of every edge but leaves all dark-side ringing untouched can obtain essentially zero alignment and zero residual-energy ratio under the current mask. That is especially undesirable for a benchmark intended eventually to evaluate complex-aware unringing.

   I would separate the two concepts:

   * **phase RMSE** remains magnitude-thresholded, exactly as the spec requires;
   * **complex Gibbs residual metrics** use an edge/sidelobe mask on **both sides of the edge**, excluding only the transition guard band.

   If bright-side scores remain useful for comparing traditional magnitude methods, report them as secondary metrics. The primary complex artifact score should not discard half the artifact.

3. **The response says Gibbs-removal acceptance now runs on clean rows only, but the implementation only does that for rule (b).** Rule (b), the PF-aware alignment/energy test, correctly has `if noisy: continue`.  But rule (a), the full-Fourier Nyquist comparison, still evaluates noisy rows, and rule (c), the two-axis anti-blurring test, also evaluates noisy rows. That does not match the response's stated disposition that “Gibbs-removal rules run on clean rows only.”

   I think the response's intended design is the cleaner one: run Gibbs-correction acceptance on clean rows and use the paired noisy rows for robustness/noise-interaction reporting. If that is the policy, add the same clean-row restriction to rules (a) and (c).

   Relatedly, the acceptance gates are calculated from broader populations than the rules they gate. `floor` includes controls from PF, noisy, and Hann conditions even though rule (a) only applies to full-Fourier/unapodized data; `pe_floor` includes noisy and Hann controls even though rule (b) subsequently excludes them.  Calculate each floor from its actual in-domain control population. That will make the “negligible artifact” exemption much more stable and interpretable.

There are also a few smaller cleanup items I would fix in the same pass. The specification is now internally stale: §3.1 still declares **`o=4` the default**, while the newly edited memory section declares `o=2` the default; later in §6 the runtime table again marks `o=4` as default and says z-slab streaming is not warranted, immediately after the preceding paragraph says streaming/sparse storage are now warranted.   That should be reconciled, especially because this PR includes the design document as an authoritative record.

The `residual_alignment()` docstring also still says `0.0` means the artifact is “entirely gone,” despite the response correctly recognizing that zero merely means **orthogonal to the template**; that was the reason `residual_energy_ratio` was added.  The acceptance output has another stale comment saying the PF axis is “descriptive only,” even though PF now has an asserted PE-axis rule.  And the CLI still describes `--noise` as k-space variance even though the implemented convention is reconstructed-image, single-coil, per-component variance.

One more minor metric issue remains: the residual metrics and Nyquist metric now consider all significant edges, which is good, but `_edge_shift()` still uses only the single strongest edge.  For a one-sided PF transfer function, the two PE edges need not have identical apparent bias. I would eventually report edge-location bias per edge or aggregate both, though I would not block this PR on that alone.

### Verdict

The response to round 3 is **substantially correct**, and I consider the previous PF-mode and CI blockers closed. The current CI is green, including the production and parallel binary checks.  The core finite-Fourier forward model itself continues to look sound.

I would now narrow the remaining merge work to three things:

1. move the memory estimate **before Stage A** and remove the obsolete `o=4`/13-GB guidance;
2. make the primary complex Gibbs metrics score **both bright and dark sidelobes**;
3. make the clean/noisy acceptance policy and its gating populations internally consistent.

After those, plus the documentation cleanup, I would be comfortable moving from **request changes** to **approve**, with RPG, sparse/slab storage, calibrated `sigma_rot`, and automatic sim-grid myelin generation remaining clearly documented follow-up work rather than merge blockers.
