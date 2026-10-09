#!/usr/bin/env bash
# The re-sync baseline (mrsim-acq docs/plans/2026-10-08-trxscan-resync.md, Task 1): the P0 fixture through
# the trxscan CLI under both feature sets, extended to what main added (partial Fourier modes, the noise
# map, the GRE fieldmap, gradient nonlinearity), one checksum per output file.
# Usage: tools/run_resync_baseline.sh <output-checksum-file>
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
FIX="$REPO/tests/fixtures/p0_baseline"
OUTFILE="${1:?usage: run_resync_baseline.sh <output-checksum-file>}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

COMMON=(--wm "$FIX/wm.nii.gz" --gm "$FIX/gm.nii.gz" --csf "$FIX/csf.nii.gz"
        --mask "$FIX/mask.nii.gz" --fmap "$FIX/fmap.nii.gz"
        --sim-wm "$FIX/sim_wm.nii.gz" --sim-gm "$FIX/sim_gm.nii.gz"
        --sim-csf "$FIX/sim_csf.nii.gz" --sim-mask "$FIX/sim_mask.nii.gz"
        --sim-fmap "$FIX/sim_fmap.nii.gz"
        --streamlines "$FIX/streamlines.tck"
        --bval "$FIX/scheme.bval" --bvec "$FIX/scheme.bvec"
        --seed 20260921)

run_config() {  # $1 = features, $2 = config name, rest = extra flags ("@OUT@" becomes the output prefix)
  local features="$1" name="$2"; shift 2
  local out="$WORK/${name}_$(echo "$features" | tr ',' '_')"
  local args=()
  for a in "$@"; do args+=("${a//@OUT@/$out}"); done
  cargo run --quiet --release --features "$features" --bin trxscan -- \
    "${COMMON[@]}" "${args[@]}" -o "$out" >/dev/null
  for f in "$out"*; do
    [ -f "$f" ] || continue
    # basename only: the directory is a fresh mktemp every run
    printf '%s  %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$(basename "$f")"
  done
}

: > "$OUTFILE"
for FEATURES in "cli" "cli,kspace,par"; do
  # the P0 configurations (the CLI starts from the hbcd preset: partial Fourier 0.75, scanner mode)
  run_config "$FEATURES" plain                                                        >> "$OUTFILE"
  run_config "$FEATURES" eddy     --eddy 0.03 --eddy-quad 0.01 --eddy-phase 0.02      >> "$OUTFILE"
  run_config "$FEATURES" parallel --coils 4 --accel 2                                 >> "$OUTFILE"
  run_config "$FEATURES" mb       --mb 2 --dropout-rate 0.5                           >> "$OUTFILE"
  # P0's "legacy" case: main has no legacy path, so o = 1 runs the oversampled path
  run_config "$FEATURES" o1       --oversample 1 --eddy 0.03 --eddy-quad 0.01 --eddy-phase 0.02 >> "$OUTFILE"
  # what main added or changed
  run_config "$FEATURES" pfcontig --pf-mode contiguous --eddy 0.03                    >> "$OUTFILE"
  run_config "$FEATURES" pffiber  --pf-mode fiberfox --eddy 0.03                      >> "$OUTFILE"
  run_config "$FEATURES" pfscan   --pf-mode scanner --eddy 0.03                       >> "$OUTFILE"
  run_config "$FEATURES" noise    --noise 0.5 --noise-map "$FIX/mask.nii.gz"          >> "$OUTFILE"
  run_config "$FEATURES" gre      --gre-out "@OUT@_gre"                               >> "$OUTFILE"
  run_config "$FEATURES" gnl      --gnl whole-body-80                                 >> "$OUTFILE"
  # the Gibbs benchmark binary (its own slice producer and phase models)
  bench="$WORK/bench_$(echo "$FEATURES" | tr ',' '_')"
  cargo run --quiet --release --features "$FEATURES" --bin trxscan-benchmark -- "$bench" matrix=32 oversample=2 slices=2 >/dev/null
  for f in $(find "$bench" -type f | sort); do
    printf '%s  %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "bench_$(echo "$FEATURES" | tr ',' '_')/${f#$bench/}"
  done >> "$OUTFILE"
done

sort -k2 -o "$OUTFILE" "$OUTFILE"
wc -l "$OUTFILE"
