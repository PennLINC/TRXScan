#!/usr/bin/env bash
# MEGRE (multi-echo GRE magnitude+phase) -> B0 fieldmap (Hz) + R2*/T2* -> ACPC and TRXScan
# scanner-frame grids -> comparison with the DRBUDDI-derived field (NIBS sub-60501 defaults).
#
# Stages (each skipped when its outputs already exist; FORCE=1 to redo):
#   register    rigid antsRegistration: MEGRE echo-1 magnitude -> ACPC preproc T1w (MI, masked)
#   mask        ACPC brain mask -> MEGRE native space (inverse rigid, NN)
#   fit         complex 5-echo field fit + R2* (fit_megre_field.py fit), native space, raw sign
#   extrapolate nearest-value + smoothed extrapolation of the field beyond the mask (whole FOV)
#   resample    field (extrapolated), R2*, fit mask -> ACPC DWI grid and the scanner acq/sim grids,
#               with EXACTLY the kit's ACPC->scanner call chained with the MEGRE->ACPC rigid
#   compare     sign / offset / per-region agreement / PE Jacobian vs the DRBUDDI field (acq grid)
#   finish      apply the sign+offset from `compare`, taper beyond the mask, T2* = 1/R2*; QC PNGs
#   provenance  append a "megre" entry to qmri/provenance.json
#
# FSL/ANTs run inside the qsiprep container (IMAGE); python via PY (nibabel numpy scipy skimage mpl).
# Usage: scripts/megre_pipeline.sh [stage ...]      (default: all stages in order)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/home/matt/miniforge3/envs/qsiprep/bin/python}
IMAGE=${IMAGE:-pennlinc/qsiprep:unstable}
NTHR=${NTHR:-12}

SUB=${SUB:-sub-60501}; SES=${SES:-ses-01}
BIDS_ANAT=${BIDS_ANAT:-/media/matt/5TB/nibs/BIDS/$SUB/$SES/anat}
QSIPREP=${QSIPREP:-/media/matt/5TB/nibs/derivatives/qsiprep-drbuddi/$SUB/$SES}
KIT=${KIT:-/media/matt/5TB/nibs/derivatives/trxscan-inputs/$SUB/$SES}
QMRI=${QMRI:-$KIT/qmri}
WORK=${WORK:-$QMRI/work-megre}
REPORT=${REPORT:-/home/matt/projects/qsiprep-testdata/outputs/nibs-exp/realism/megre}
TRT=${TRT:-0.0917359}          # DWI TotalReadoutTime (s), HBCD75 PA/AP

P="${SUB}_${SES}"
MEGRE() { echo "$BIDS_ANAT/${P}_acq-QSM_run-01_echo-$1_part-$2_MEGRE.nii.gz"; }
TE_MS="6.34 11.78 17.32 22.81 28.3"     # from the sidecars (EchoTime 0.00634 .. 0.0283 s)
T1W="$QSIPREP/anat/${P}_space-ACPC_desc-preproc_T1w.nii.gz"
BMASK="$QSIPREP/anat/${P}_space-ACPC_desc-brain_mask.nii.gz"
DWI_MASK="$QSIPREP/dwi/${P}_acq-HBCD75_run-01_space-ACPC_desc-brain_mask.nii.gz"
SCANNER_XFM="$KIT/scanner/from-scanner_to-ACPC_xfm.mat"      # qsiprep b0_to_anat (ACPC <- scanner)
SIM_REF="$KIT/scanner/grid/sim/wm.nii.gz";  SIM_MASK="$KIT/scanner/grid/sim/mask.nii.gz"
ACQ_REF="$KIT/scanner/grid/wm.nii.gz";      ACQ_MASK="$KIT/scanner/grid/mask.nii.gz"
ACQ_ASEG="$KIT/scanner/grid/aseg.nii.gz"
ACQ_DRBUDDI="$KIT/scanner/grid/fmap_hz.nii.gz"
REF1MM="$KIT/scanner/ref_1mm_lps.nii.gz";   T1W_SC="$KIT/scanner/t1w.nii.gz"

mkdir -p "$WORK" "$QMRI" "$REPORT"
LOG="$WORK/commands.log"

ctr() {
  echo "[$(date -Is)] $*" >> "$LOG"
  docker run --rm -u "$(id -u):$(id -g)" --entrypoint bash \
    -v "$BIDS_ANAT":"$BIDS_ANAT":ro -v "$QSIPREP":"$QSIPREP":ro -v "$KIT":"$KIT" -v "$WORK":"$WORK" \
    -e OMP_NUM_THREADS="$NTHR" -e ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$NTHR" \
    "$IMAGE" -c "$*"
}
have() { for f in "$@"; do [[ -s $f ]] || return 1; done; [[ ${FORCE:-0} == 0 ]]; }
FIT="$PY $HERE/fit_megre_field.py"

stage_register() {
  have "$WORK/megre2acpc_0GenericAffine.mat" && { echo "[register] exists, skip"; return; }
  # Fixed mask: ACPC brain mask dilated 8 mm (as in the MESE pipeline) so neck/face cannot drive MI.
  ctr "ImageMath 3 $WORK/bmask_dil.nii.gz MD $BMASK 8"
  ctr "antsRegistration -d 3 --float 0 -v 1 -o [$WORK/megre2acpc_,$WORK/echo-1_mag_space-ACPC.nii.gz] \
       -n Linear -w [0.005,0.995] -u 1 -z 1 \
       -r [$T1W,$(MEGRE 1 mag),1] \
       -t Rigid[0.1] -m MI[$T1W,$(MEGRE 1 mag),1,32,Regular,0.25] \
       -c [1000x500x250x100,1e-6,10] -f 8x4x2x1 -s 3x2x1x0vox -x [$WORK/bmask_dil.nii.gz,NULL] \
       > $WORK/antsRegistration.log 2>&1"
  $FIT regcheck --fixed "$T1W" --moving "$WORK/echo-1_mag_space-ACPC.nii.gz" --mask "$BMASK" \
       --title "MEGRE echo-1 magnitude (rigid) on the ACPC T1w" --png "$REPORT/reg_megre_on_acpc_t1w.png"
}

stage_mask() {
  have "$WORK/bmask_megre.nii.gz" && { echo "[mask] exists, skip"; return; }
  ctr "antsApplyTransforms -d 3 -i $BMASK -r $(MEGRE 1 mag) -t [$WORK/megre2acpc_0GenericAffine.mat,1] \
       -n NearestNeighbor -o $WORK/bmask_megre.nii.gz"
}

stage_fit() {
  have "$WORK/fieldmap_raw.nii.gz" "$WORK/R2star.nii.gz" && { echo "[fit] exists, skip"; return; }
  $FIT fit --mag $(for e in 1 2 3 4 5; do MEGRE $e mag; done) \
           --phase $(for e in 1 2 3 4 5; do MEGRE $e phase; done) \
           --te $TE_MS --mask "$WORK/bmask_megre.nii.gz" --mag-floor-frac 0.05 \
           --out-prefix "$WORK/" --fit-json "$WORK/fit_native.json" --qc-png "$REPORT/fit_qc_native.png" \
           | tee "$WORK/fit.log"
}

stage_extrapolate() {
  have "$WORK/fieldmap_raw_extrap.nii.gz" && { echo "[extrapolate] exists, skip"; return; }
  # The outer 1-2 mm of the mask is the noisiest (low |S|, partial volume); extrapolate from 1 vox in.
  $FIT extrapolate --field "$WORK/fieldmap_raw.nii.gz" --mask "$WORK/fitmask.nii.gz" --erode 1 --sigma 3 \
       --out "$WORK/fieldmap_raw_extrap.nii.gz"
}

stage_resample() {
  have "$WORK/fieldmap_raw_scanneracq.nii.gz" "$WORK/fieldmap_raw_scannersim.nii.gz" "$WORK/fieldmap_raw_acpc-dwi.nii.gz" \
    && { echo "[resample] exists, skip"; return; }
  local R="$WORK/megre2acpc_0GenericAffine.mat"
  for img in fieldmap_raw_extrap R2star fitmask; do
    ctr "antsApplyTransforms -d 3 -i $WORK/$img.nii.gz -r $DWI_MASK -t $R -n Linear -o $WORK/${img%_extrap}_acpc-dwi.nii.gz"
  done
  # EXACTLY the kit's ACPC->scanner call (scanner/resample.sh: -t [from-scanner_to-ACPC_xfm.mat,1]) chained
  # with the MEGRE->ACPC rigid, so the native maps are interpolated once (order verified in the MESE provenance).
  for grid in sim acq; do
    ref=$SIM_REF; sp=scannersim; [[ $grid == acq ]] && { ref=$ACQ_REF; sp=scanneracq; }
    for img in fieldmap_raw_extrap R2star fitmask; do
      ctr "antsApplyTransforms -d 3 -i $WORK/$img.nii.gz -r $ref -t [$SCANNER_XFM,1] -t $R -n Linear \
           -o $WORK/${img%_extrap}_${sp}.nii.gz"
    done
  done
  # QC: echo-1 magnitude into the 1 mm scanner frame (same chain) for an edge overlay on scanner/t1w
  ctr "antsApplyTransforms -d 3 -i $(MEGRE 1 mag) -r $REF1MM -t [$SCANNER_XFM,1] -t $R -n Linear \
       -o $WORK/echo-1_mag_scanner1mm.nii.gz"
}

stage_compare() {
  $FIT compare --test "$WORK/fieldmap_raw_scanneracq.nii.gz" --ref "$ACQ_DRBUDDI" --mask "$ACQ_MASK" \
       --aseg "$ACQ_ASEG" --trt "$TRT" --jac-sigma 1.0 --vlim 150 \
       --out-prefix "$WORK/cmp_acq_" --metrics "$REPORT/metrics.json" --report-dir "$REPORT" | tee "$WORK/compare.log"
}

stage_finish() {
  local C="$WORK/cmp_acq_correction.json"
  [[ -s $C ]] || { echo "[finish] needs compare first"; exit 1; }
  # native (MEGRE space): field = fit inside the mask, extrapolated 5 vox beyond, taper 3 vox
  $FIT finish --field "$WORK/fieldmap_raw_extrap.nii.gz" --mask "$WORK/fitmask.nii.gz" --correction-json "$C" \
       --out "$QMRI/${P}_space-MEGRE_fieldmap.nii.gz" \
       --r2s "$WORK/R2star.nii.gz" --coverage "$WORK/fitmask.nii.gz" --out-t2s "$QMRI/${P}_space-MEGRE_T2starmap.nii.gz"
  cp "$WORK/R2star.nii.gz" "$QMRI/${P}_space-MEGRE_R2starmap.nii.gz"
  # ACPC (qsiprep preproc DWI grid, like the MESE maps)
  $FIT finish --field "$WORK/fieldmap_raw_acpc-dwi.nii.gz" --mask "$DWI_MASK" --correction-json "$C" \
       --out "$QMRI/${P}_space-ACPC_fieldmap.nii.gz" \
       --r2s "$WORK/R2star_acpc-dwi.nii.gz" --coverage "$WORK/fitmask_acpc-dwi.nii.gz" \
       --out-t2s "$QMRI/${P}_space-ACPC_T2starmap.nii.gz"
  # scanner grids
  for grid in acq sim; do
    sp=scanneracq; m=$ACQ_MASK; [[ $grid == sim ]] && { sp=scannersim; m=$SIM_MASK; }
    $FIT finish --field "$WORK/fieldmap_raw_${sp}.nii.gz" --mask "$m" --correction-json "$C" \
         --out "$QMRI/${P}_space-scanner_res-${grid}_desc-megre_fieldmap.nii.gz" \
         --r2s "$WORK/R2star_${sp}.nii.gz" --coverage "$WORK/fitmask_${sp}.nii.gz" \
         --out-t2s "$QMRI/${P}_space-scanner_res-${grid}_T2starmap.nii.gz"
  done
  # sidecars
  $PY - "$QMRI/${P}_space-scanner_res-acq_desc-megre_fieldmap.json" "$QMRI/${P}_space-scanner_res-sim_desc-megre_fieldmap.json" \
        "$QMRI/${P}_space-ACPC_fieldmap.json" "$QMRI/${P}_space-MEGRE_fieldmap.json" "$C" "$TRT" <<'EOF'
import json, sys
c = json.load(open(sys.argv[5])); trt = float(sys.argv[6])
for p in sys.argv[1:5]:
    json.dump({"Description": "B0 off-resonance field from the acq-QSM MEGRE (5 echoes, complex fit). Same sign "
                              "convention as the DRBUDDI-derived kit fieldmap: for PhaseEncodingDirection j (PA) "
                              "signal is displaced by +field*TotalReadoutTime voxels along +j (anterior). Offset "
                              "matched to the DRBUDDI field (median difference removed). 0 outside the brain "
                              "mask + 5 voxels (3-voxel taper).",
               "EstimationMethod": "MEGRE complex multi-echo fit (scripts/fit_megre_field.py)",
               "Units": "Hz", "TotalReadoutTime": trt, "PhaseEncodingDirection": "j",
               "SignAppliedToRawMEGRE": c["sign"], "OffsetSubtractedHz": c["offset_hz"],
               "GradientNonlinearity": "MEGRE is DIS3D (vendor-corrected); the DWI/DRBUDDI field is ND"},
              open(p, "w"), indent=1)
EOF
  # QC overlays
  $FIT regcheck --fixed "$T1W_SC" --moving "$WORK/echo-1_mag_scanner1mm.nii.gz" \
       --title "MEGRE echo-1 magnitude (rigid + kit ACPC->scanner chain) on scanner/t1w.nii.gz" \
       --png "$REPORT/reg_megre_on_scanner_t1w.png"
  $FIT regcheck --fixed "$ACQ_REF" --moving "$WORK/R2star_scanneracq.nii.gz" --mask "$ACQ_MASK" \
       --field "$QMRI/${P}_space-scanner_res-acq_desc-megre_fieldmap.nii.gz" \
       --title "MEGRE R2* edges + final field on the scanner acq grid (wm probseg)" --png "$REPORT/scanner_acq_field_over_wm.png"
  $FIT regcheck --fixed "$SIM_REF" --moving "$WORK/R2star_scannersim.nii.gz" --mask "$SIM_MASK" \
       --field "$QMRI/${P}_space-scanner_res-sim_desc-megre_fieldmap.nii.gz" \
       --title "MEGRE R2* edges + final field on the scanner sim grid (wm probseg)" --png "$REPORT/scanner_sim_field_over_wm.png"
}

stage_provenance() {
  "$PY" - "$QMRI" "$WORK" "$REPORT" "$HERE" "$TRT" <<'PYEOF'
import sys, json, subprocess
from pathlib import Path
qmri, work, report, here = map(Path, sys.argv[1:5]); trt = float(sys.argv[5])
J = lambda p: json.load(open(p)) if Path(p).exists() else None
git = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
m = J(report / "metrics.json") or {}
entry = {
 "pipeline": {"driver": str(here / "megre_pipeline.sh"), "fit": str(here / "fit_megre_field.py"), "trxscan_git": git},
 "inputs": {"megre": "acq-QSM run-01, 5 echoes TE 6.34/11.78/17.32/22.81/28.3 ms, TR 33 ms, FA 15, 3D GRE "
                     "(*fl3d5r), 1 mm iso 192x256x144 RAS, GRAPPA 3, PE i, ImageType DIS3D (gradient-"
                     "nonlinearity corrected); phase int16 -4096..4094, radians = value*pi/4096",
            "reference_field": "scanner/grid/fmap_hz.nii.gz (DRBUDDI, sign corrected 2026-09-19)"},
 "registration": "antsRegistration Rigid[0.1], MI(32 bins, 25% sampling), 4 levels; fixed = ACPC preproc T1w "
                 "with brain mask dilated 8 mm; moving = MEGRE echo-1 magnitude",
 "mask": "ACPC brain mask -> MEGRE space (inverse rigid, NN), minus |S1| < 5% of p99 (signal voids)",
 "field_fit": "coarse: echo-2 x conj(echo-1) phase, 3D spatial unwrap (skimage unwrap_phase) in mask, "
              "f = dphi/(2 pi dTE); final: temporal unwrap of all 5 echoes with the coarse field, |S|^2-weighted "
              "linear fit phase vs TE (2 passes), f = slope/2pi; raw sign = stored phase increasing with TE",
 "r2star_fit": "|S|^2-weighted log-linear fit ln|S| vs TE over 5 echoes; T2* clamped to [2, 200] ms",
 "extrapolation": "fit mask eroded 1 vox; nearest-in-mask value + Gaussian sigma 3 vox outside; whole FOV "
                  "before resampling; final maps weighted 1 within (grid mask + 5 vox), taper to 0 over 3 vox",
 "resampling": "antsApplyTransforms Linear; ACPC = qsiprep preproc DWI grid; scanner grids = one-shot chain "
               "-t [scanner/from-scanner_to-ACPC_xfm.mat,1] -t megre2acpc_0GenericAffine.mat (kit call composed "
               "with the MEGRE->ACPC rigid, as for the MESE maps)",
 "sign_and_offset": {"sign_determination": m.get("sign_determination"), "offset": m.get("offset"),
                     "trt_s": trt},
 "comparison_summary": {k: {kk: v[kk] for kk in ("n", "r", "rms_hz", "diff_p5_50_95_hz")}
                        for k, v in (m.get("regions") or {}).items()},
 "pe_jacobian_definition": (m.get("pe_jacobian") or {}).get("definition"),
 "gnl_note": "MEGRE is DIS3D-corrected, the DWI (and hence the DRBUDDI field) is not (ND): expect <= 1-2 mm "
             "peripheral geometric discrepancy; not undone",
 "fit_summary": {k: v for k, v in (J(work / "fit_native.json") or {}).items() if k != "outputs"},
 "report": str(report),
 "commands": (work / "commands.log").read_text().splitlines() if (work / "commands.log").exists() else [],
}
pp = qmri / "provenance.json"
prov = J(pp) or {}
prov["megre"] = entry
pp.write_text(json.dumps(prov, indent=1))
print("wrote", pp, "(appended key 'megre')")
PYEOF
}

stages=("$@"); [[ ${#stages[@]} -eq 0 ]] && stages=(register mask fit extrapolate resample compare finish provenance)
for s in "${stages[@]}"; do echo "== stage $s"; "stage_$s"; done
echo "done"
