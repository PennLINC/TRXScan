# `scripts/` — validation and real-vs-sim analysis

Python helpers that sit *beside* the Rust simulator. None of this is part of the `cargo` build —
it reads the NIfTI the binaries write and (for the comparison tools) real BIDS data.

Environment: `python3` with `numpy`, `scipy`; the comparison and figure scripts also need
`nibabel` and `matplotlib`. The `test_*.py` files run under `pytest` and need no simulator build;
CI runs them on numpy/scipy alone (tests needing more skip themselves). The microstructure
fixtures use a separate conda env (see the repo `CLAUDE.md`), not these scripts.

## Data preparation

- `prepare_acquisition_grid.py` — resamples anatomical probsegs (+ an optional fieldmap and
  myelin map) onto an acquisition grid, and with `--oversample N` also onto the `N×` finer
  simulation grid `trxscan` needs for intrinsic Gibbs ringing. Its `--out` directory
  (`wm/gm/csf/mask.nii.gz` + `sim/`) is what `trxscan --wm ... --sim-wm ...` and the comparison
  scripts' `--grid-dir` point at. `test_oversample.py` checks the refinement is exact (needs the
  reference bundle at `$TRXSCAN_REFERENCE` or `~/projects/trxscan-reference`).
- `make_noise_map.py` — a smooth centre-high noise-level (SD) map from the brain mask, for
  `trxscan --noise-map` (a peripheral head-coil array's g-factor pattern: noise peaks mid-brain).

## Gibbs-ringing benchmark (`trxscan-benchmark`)

`acceptance.py` scores the factor grid `trxscan-benchmark` writes; it drives `run_unringing.py`
(one interface over none / `mrdegibbs` / dipy / RPG unringing) and `score_unringing.py` (splits
the residual into ringing left over vs. edge sharpness lost). `score_crosscheck.py` is a
deliberately simple second scorer that must agree with the first; `behavioural_validation.py`
compares the frequency profile of what `mrdegibbs` removes, sim vs real. All have `test_*.py`.

## Real vs. simulated DWI

The point is a **voxelwise** comparison, so the simulation and the real scan must share a world
frame: simulate in the scanner frame on the real acquisition grid (a `prepare_acquisition_grid.py`
run on the subject's own probsegs, with `--fieldmap` from the subject's own fieldmap estimate).
Then:

| Step | Script | Produces |
|---|---|---|
| 1. tissue maps on the acq grid | `prepare_acquisition_grid.py` | the `--grid-dir` |
| 2. metrics (+ figures) | `compare_real_sim.py` | `<out>.json`, and with `--png` four figures |
| 3. next knob values | `fit_tissue_params.py` | recommended `--tissue-s0` / `--diff-scale` |

`compare_real_sim.py` resamples the *real* data onto the *sim* output grid **by affine, never by
index** (the grid files are LPS while `--fsl-orientation` output is LAS), puts the sim on the real
scale by one global WM-b0 median factor, then reports per shell and per tissue: voxelwise b0
agreement (Pearson r, median absolute % error), tissue-mean b0 levels and S(b)/S0 decay, the
background noise SD and WM SNR, and in phase the circular SD per shell in WM plus the b0 phase-
gradient RMS. Real and sim DWIs are addressed by a BIDS *stem* with `{pe}` / `{part}` placeholders
(`sub-XX_dir-{pe}_part-{part}_dwi`), one `--pe` at a time; `--shells` sets the shell centres. With
`--png` it also writes the montage, the ax/cor/sag magnitude panel, the phase + tight-window
background panel (ghost/noise) and centre-line profiles.

```bash
python compare_real_sim.py \
  --real-dir REAL --sim-dir SIM --grid-dir GRID --pe PA \
  --real-stem 'sub-XXXXX_ses-01_dir-{pe}_part-{part}_dwi' \
  --sim-stem  'sub-XXXXX_ses-01_dir-{pe}_part-{part}_dwi' \
  --label 'sim v3' --png --out cmp_PA
python fit_tissue_params.py cmp_PA.json          # -> the --tissue-s0 / --diff-scale to try next
```

The tuning loop is: compare, adjust the `trxscan` flags (`--noise --noise-map --coils --eddy
--eddy-phase --tissue-s0 --diff-scale --phase-model ...`, see the top-level `README.md`),
re-simulate, re-compare. Phase handling (Siemens integer phase → radians, circular statistics)
comes from `calibrate_phase.py`, whose docstring records where the conversion was verified.

The relaxometry fits that produced the `adult` preset's T2s and the one-subject HTML report live
in the nibs reference kit, not here.
