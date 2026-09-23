#!/usr/bin/env bash
# Run the P0 fixture under both feature sets and emit checksums.
# Usage: tools/run_p0_baseline.sh <output-checksum-file>
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
FIX="$REPO/tests/fixtures/p0_baseline"
OUTFILE="${1:?usage: run_p0_baseline.sh <output-checksum-file>}"
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

run_config() {  # $1 = features, $2 = config name, rest = extra flags
  local features="$1" name="$2"; shift 2
  local out="$WORK/${name}_$(echo "$features" | tr ',' '_')"
  cargo run --quiet --release --features "$features" --bin trxscan -- \
    "${COMMON[@]}" "$@" -o "$out" >/dev/null
  for f in "$out"*; do
    [ -f "$f" ] || continue
    # Basename only. The directory is a fresh mktemp every run, so including it would make
    # every manifest line unique per invocation and Step 7 could never pass. The basename
    # already carries the configuration and feature-set names.
    printf '%s  %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$(basename "$f")"
  done
}

: > "$OUTFILE"
for FEATURES in "cli" "cli,kspace,par"; do
  # 1. plain: oversampled production path (o=2), no eddy, single coil, no accel, no MB
  run_config "$FEATURES" plain                          >> "$OUTFILE"
  # 2. eddy: nonzero linear and quadratic eddy + eddy phase
  run_config "$FEATURES" eddy    --eddy 0.03 --eddy-quad 0.01 --eddy-phase 0.02 >> "$OUTFILE"
  # 3. parallel: multiple coils with GRAPPA
  run_config "$FEATURES" parallel --coils 4 --accel 2   >> "$OUTFILE"
  # 4. multiband: within-volume motion dropout
  run_config "$FEATURES" mb      --mb 2 --dropout-rate 0.5 >> "$OUTFILE"
  # 5. legacy: simulate_acquisition_legacy (o=1), also touched by changes 1, 2, 4, 5.
  #    Eddy must be ON here: the legacy wrapper's own trap, (g/|g|)*|g| != g, is only
  #    reachable through the eddy model, and with eddy off (kspace.rs:393-394) the gate
  #    could not tell Some(g) from Some(bvec*bval).
  run_config "$FEATURES" legacy  --oversample 1 --eddy 0.03 --eddy-quad 0.01 --eddy-phase 0.02 >> "$OUTFILE"
done

sort -k2 -o "$OUTFILE" "$OUTFILE"
wc -l "$OUTFILE"
