#!/usr/bin/env bash
# MESE -> voxelwise T2/S0 -> predicted diffusion b0 pipeline (set SUB/SES + the BIDS_ANAT/QSIPREP/KIT dirs for your data).
#
# Stages (each skipped when its outputs already exist; pass FORCE=1 to redo):
#   topup     FSL topup on AP echo-1 + PA echo-1, applytopup (jacobian) to the 4 AP echoes
#   register  rigid antsRegistration: SDC-corrected echo-3 -> ACPC preproc T2w
#   fit       T2/S0 fit in MESE native (corrected) space, with per-tissue TR correction for
#             echo-4 (dseg resampled into MESE space) and N4 bias field for S0
#   resample  S0 / R2 -> ACPC DWI grid (linear; T2 = 1/R2)
#   predict   b0 prediction at the DWI TE/TR, baseline, metrics, figures (fit_mese_t2.py)
#   scanner   S0 / T2 -> TRXScan scanner-frame sim + acquisition grids (kit transform)
#
# FSL / ANTs run inside the qsiprep container (docker, IMAGE=pennlinc/qsiprep:unstable);
# python steps use PY (needs nibabel, numpy, scipy, matplotlib).
#
# Usage: scripts/mese_pipeline.sh [stage ...]      (default: all stages in order)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
IMAGE=${IMAGE:-pennlinc/qsiprep:unstable}
NTHR=${NTHR:-12}

SUB=${SUB:-sub-01}; SES=${SES:-ses-01}   # override for your subject/session
BIDS_ANAT=${BIDS_ANAT:?set BIDS_ANAT to the subject anat dir (multi-echo GRE/SE-EPI source)}
QSIPREP=${QSIPREP:?set QSIPREP to the qsiprep derivatives dir for the subject}
KIT=${KIT:?set KIT to the TRXScan scanner-frame inputs for the subject}
QMRI=${QMRI:-$KIT/qmri}
WORK=${WORK:-$QMRI/work}
REPORT=${REPORT:-./realism/mese}

P="${SUB}_${SES}"
MESE_AP() { echo "$BIDS_ANAT/${P}_dir-AP_run-01_echo-$1_MESE.nii.gz"; }
MESE_PA1="$BIDS_ANAT/${P}_dir-PA_run-01_echo-1_MESE.nii.gz"
T2W="$QSIPREP/anat/${P}_space-ACPC_desc-preproc_T2w.nii.gz"
DSEG="$QSIPREP/anat/${P}_space-ACPC_dseg.nii.gz"            # 1=CSF 2=GM 3=WM (verified vs probsegs)
BMASK="$QSIPREP/anat/${P}_space-ACPC_desc-brain_mask.nii.gz"
DWI="$QSIPREP/dwi/${P}_acq-HBCD75_run-01_space-ACPC_desc-preproc_dwi.nii.gz"
DWI_MASK="$QSIPREP/dwi/${P}_acq-HBCD75_run-01_space-ACPC_desc-brain_mask.nii.gz"
SCANNER_XFM="$KIT/scanner/from-scanner_to-ACPC_xfm.mat"      # qsiprep b0_to_anat (ACPC <- scanner)
SIM_REF="$KIT/scanner/grid/sim/wm.nii.gz"
ACQ_REF="$KIT/scanner/grid/wm.nii.gz"

mkdir -p "$WORK" "$QMRI" "$REPORT"
LOG="$WORK/commands.log"

# Run a command inside the container with the data roots mounted at their host paths, so every
# path in this script is the same inside and out. The command line is appended to $LOG.
ctr() {
  echo "[$(date -Is)] $*" >> "$LOG"
  docker run --rm -u "$(id -u):$(id -g)" --entrypoint bash \
    -v "$BIDS_ANAT":"$BIDS_ANAT":ro -v "$QSIPREP":"$QSIPREP":ro -v "$KIT":"$KIT" -v "$WORK":"$WORK" \
    -e OMP_NUM_THREADS="$NTHR" -e ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$NTHR" \
    "$IMAGE" -c "$*"
}
have() { for f in "$@"; do [[ -s $f ]] || return 1; done; [[ ${FORCE:-0} == 0 ]]; }

stage_topup() {
  have "$WORK/echo-4_sdc.nii.gz" && { echo "[topup] exists, skip"; return; }
  # AP is PhaseEncodingDirection j- (row 1), PA is j (row 2); TotalReadoutTime 0.01728 s (JSON).
  printf '0 -1 0 0.01728\n0 1 0 0.01728\n' > "$WORK/acqparams.txt"
  "$PY" - "$(MESE_AP 1)" "$MESE_PA1" "$WORK/topup_in.nii.gz" <<'EOF'
import sys, numpy as np, nibabel as nib
a, b = (nib.load(f) for f in sys.argv[1:3])
x = np.stack([np.asanyarray(a.dataobj), np.asanyarray(b.dataobj)], -1).astype(np.float32)
nib.save(nib.Nifti1Image(x, a.affine), sys.argv[3])
EOF
  ctr "topup --imain=$WORK/topup_in.nii.gz --datain=$WORK/acqparams.txt --config=b02b0.cnf \
       --out=$WORK/topup --iout=$WORK/topup_iout.nii.gz --fout=$WORK/topup_fieldHz.nii.gz --nthr=$NTHR -v \
       > $WORK/topup.log 2>&1"
  for e in 1 2 3 4; do
    ctr "applytopup --imain=$(MESE_AP $e) --inindex=1 --datain=$WORK/acqparams.txt \
         --topup=$WORK/topup --method=jac --interp=spline --out=$WORK/echo-${e}_sdc.nii.gz"
  done
  ctr "applytopup --imain=$MESE_PA1 --inindex=2 --datain=$WORK/acqparams.txt \
       --topup=$WORK/topup --method=jac --interp=spline --out=$WORK/echo-1_PA_sdc.nii.gz"
}

stage_register() {
  have "$WORK/mese2acpc_0GenericAffine.mat" && { echo "[register] exists, skip"; return; }
  # Fixed mask: ACPC brain mask dilated 8 mm, so the neck/face in the whole-head T2w cannot drive MI.
  ctr "ImageMath 3 $WORK/bmask_dil.nii.gz MD $BMASK 8"
  ctr "antsRegistration -d 3 --float 0 -v 1 -o [$WORK/mese2acpc_,$WORK/echo-3_sdc_space-ACPC.nii.gz] \
       -n Linear -w [0.005,0.995] -u 1 -z 1 \
       -r [$T2W,$WORK/echo-3_sdc.nii.gz,1] \
       -t Rigid[0.1] -m MI[$T2W,$WORK/echo-3_sdc.nii.gz,1,32,Regular,0.25] \
       -c [1000x500x250x100,1e-6,10] -f 8x4x2x1 -s 3x2x1x0vox -x [$WORK/bmask_dil.nii.gz,NULL] \
       > $WORK/antsRegistration.log 2>&1"
  # Tissue labels and brain mask into the corrected MESE space (inverse of the rigid).
  ctr "antsApplyTransforms -d 3 -i $DSEG -r $WORK/echo-3_sdc.nii.gz \
       -t [$WORK/mese2acpc_0GenericAffine.mat,1] -n NearestNeighbor -o $WORK/dseg_mese.nii.gz"
  ctr "antsApplyTransforms -d 3 -i $BMASK -r $WORK/echo-3_sdc.nii.gz \
       -t [$WORK/mese2acpc_0GenericAffine.mat,1] -n NearestNeighbor -o $WORK/bmask_mese.nii.gz"
}

stage_fit() {
  have "$WORK/T2_mese.nii.gz" "$WORK/S0_mese.nii.gz" && { echo "[fit] exists, skip"; return; }
  # Receive-coil bias from corrected echo-1 (the MESE is not prescan-normalised; the DWI is
  # rec-norm + N4 in qsiprep). The field is common to all echoes so only S0 needs it.
  ctr "N4BiasFieldCorrection -d 3 -i $WORK/echo-1_sdc.nii.gz -x $WORK/bmask_mese.nii.gz \
       -s 2 -b [150] -c [50x50x50x50,1e-6] -o [$WORK/echo-1_sdc_n4.nii.gz,$WORK/bias_mese.nii.gz]"
  "$PY" "$HERE/fit_mese_t2.py" fit \
    --echoes "$WORK"/echo-{1,2,3,4}_sdc.nii.gz --te 15 30 50 100 --tr 5.71 5.71 5.71 6.03 \
    --dseg "$WORK/dseg_mese.nii.gz" --dseg-labels 3 2 1 --brain-mask "$WORK/bmask_mese.nii.gz" \
    --bias "$WORK/bias_mese.nii.gz" --out-prefix "$WORK/" --fit-json "$WORK/fit_native.json"
}

stage_resample() {
  have "$QMRI/${P}_space-ACPC_T2map.nii.gz" && { echo "[resample] exists, skip"; return; }
  ctr "antsApplyTransforms -d 3 -i $WORK/S0_mese.nii.gz -r $DWI_MASK -t $WORK/mese2acpc_0GenericAffine.mat \
       -n Linear -o $WORK/S0_acpc-dwi.nii.gz"
  ctr "antsApplyTransforms -d 3 -i $WORK/S0raw_mese.nii.gz -r $DWI_MASK -t $WORK/mese2acpc_0GenericAffine.mat \
       -n Linear -o $WORK/S0raw_acpc-dwi.nii.gz"
  ctr "antsApplyTransforms -d 3 -i $WORK/R2_mese.nii.gz -r $DWI_MASK -t $WORK/mese2acpc_0GenericAffine.mat \
       -n Linear -o $WORK/R2_acpc-dwi.nii.gz"
  ctr "antsApplyTransforms -d 3 -i $WORK/fitmask_mese.nii.gz -r $DWI_MASK -t $WORK/mese2acpc_0GenericAffine.mat \
       -n Linear -o $WORK/fitmask_acpc-dwi.nii.gz"
  "$PY" "$HERE/fit_mese_t2.py" finish-maps --s0 "$WORK/S0_acpc-dwi.nii.gz" --s0raw "$WORK/S0raw_acpc-dwi.nii.gz" \
    --r2 "$WORK/R2_acpc-dwi.nii.gz" --coverage "$WORK/fitmask_acpc-dwi.nii.gz" \
    --out-t2 "$QMRI/${P}_space-ACPC_T2map.nii.gz" --out-s0 "$QMRI/${P}_space-ACPC_S0map.nii.gz" \
    --out-s0raw "$QMRI/${P}_space-ACPC_desc-raw_S0map.nii.gz"
}

stage_predict() {
  "$PY" "$HERE/fit_mese_t2.py" predict \
    --t2 "$QMRI/${P}_space-ACPC_T2map.nii.gz" --s0 "$QMRI/${P}_space-ACPC_S0map.nii.gz" \
    --dwi "$DWI" --bval "${DWI%.nii.gz}.bval" --dwi-mask "$DWI_MASK" \
    --probseg "$KIT/${P}_space-ACPC_label-WM_probseg.nii.gz" "$KIT/${P}_space-ACPC_label-GM_probseg.nii.gz" \
              "$KIT/${P}_space-ACPC_label-CSF_probseg.nii.gz" \
    --te 88 --tr 4.8 --tr-ref 5.71 \
    --out-pred "$QMRI/${P}_space-ACPC_desc-mese_b0pred.nii.gz" \
    --out-baseline "$QMRI/${P}_space-ACPC_desc-baseline_b0pred.nii.gz" \
    --out-realb0 "$QMRI/${P}_space-ACPC_desc-real_meanb0.nii.gz" \
    --report-dir "$REPORT" --t2w "$T2W" --reg-check "$WORK/echo-3_sdc_space-ACPC.nii.gz" \
    --brain-mask "$BMASK"
}

stage_scanner() {
  # EXACTLY the kit's ACPC->scanner call (scanner/resample.sh: -t [from-scanner_to-ACPC_xfm.mat,1]),
  # verified to reproduce scanner/t1w.nii.gz bit-for-bit from the ACPC T1w. Chained with the
  # MESE->ACPC rigid so the native maps are interpolated once. Transform order verified in
  # provenance (chain vs two-step agree).
  for grid in sim acq; do
    ref=$SIM_REF; sp=scannersim; [[ $grid == acq ]] && { ref=$ACQ_REF; sp=scanneracq; }
    ctr "antsApplyTransforms -d 3 -i $WORK/S0_mese.nii.gz -r $ref -t [$SCANNER_XFM,1] \
         -t $WORK/mese2acpc_0GenericAffine.mat -n Linear -o $WORK/S0_${sp}.nii.gz"
    ctr "antsApplyTransforms -d 3 -i $WORK/R2_mese.nii.gz -r $ref -t [$SCANNER_XFM,1] \
         -t $WORK/mese2acpc_0GenericAffine.mat -n Linear -o $WORK/R2_${sp}.nii.gz"
    ctr "antsApplyTransforms -d 3 -i $WORK/fitmask_mese.nii.gz -r $ref -t [$SCANNER_XFM,1] \
         -t $WORK/mese2acpc_0GenericAffine.mat -n Linear -o $WORK/fitmask_${sp}.nii.gz"
    "$PY" "$HERE/fit_mese_t2.py" finish-maps --s0 "$WORK/S0_${sp}.nii.gz" --r2 "$WORK/R2_${sp}.nii.gz" \
      --coverage "$WORK/fitmask_${sp}.nii.gz" \
      --out-t2 "$QMRI/${P}_space-scanner_res-${grid}_T2map.nii.gz" \
      --out-s0 "$QMRI/${P}_space-scanner_res-${grid}_S0map.nii.gz" \
      --overlay "$ref" --overlay-png "$REPORT/scanner_${grid}_T2_over_wm.png"
  done
}

stage_verify() {
  # (1) the kit transform: regenerate scanner/t1w.nii.gz from the ACPC T1w with the call used for
  #     the S0/T2 maps and diff it; (2) the chain order of the two-transform call: compare the
  #     one-shot chain (both orders) with a two-step resampling via the ACPC 1 mm grid.
  local T1W="$QSIPREP/anat/${P}_space-ACPC_desc-preproc_T1w.nii.gz"
  ctr "antsApplyTransforms -d 3 -i $T1W -r $KIT/scanner/ref_1mm_lps.nii.gz -t [$SCANNER_XFM,1] \
       -n LanczosWindowedSinc -o $WORK/verify_t1w_regen.nii.gz"
  ctr "antsApplyTransforms -d 3 -i $WORK/echo-3_sdc.nii.gz -r $ACQ_REF -t [$SCANNER_XFM,1] \
       -t $WORK/mese2acpc_0GenericAffine.mat -n Linear -o $WORK/verify_chain_used.nii.gz"
  ctr "antsApplyTransforms -d 3 -i $WORK/echo-3_sdc.nii.gz -r $ACQ_REF -t $WORK/mese2acpc_0GenericAffine.mat \
       -t [$SCANNER_XFM,1] -n Linear -o $WORK/verify_chain_swapped.nii.gz"
  ctr "antsApplyTransforms -d 3 -i $WORK/echo-3_sdc_space-ACPC.nii.gz -r $ACQ_REF -t [$SCANNER_XFM,1] \
       -n Linear -o $WORK/verify_twostep.nii.gz"
  "$PY" - "$KIT" "$WORK" <<'PYEOF'
import sys, json, numpy as np, nibabel as nib
kit, work = sys.argv[1:3]
L = lambda f: nib.load(f).get_fdata()
a, b = L(f"{kit}/scanner/t1w.nii.gz"), L(f"{work}/verify_t1w_regen.nii.gz")
wm = L(f"{kit}/scanner/grid/wm.nii.gz") > 0.5
ref = L(f"{work}/verify_twostep.nii.gz")
r = lambda x: float(np.corrcoef(x[wm], ref[wm])[0, 1])
out = {"t1w_regen_max_abs_diff": float(np.abs(a - b).max()),
       "t1w_regen_corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
       "chain_used_vs_twostep_corr_in_WM": r(L(f"{work}/verify_chain_used.nii.gz")),
       "chain_swapped_vs_twostep_corr_in_WM": r(L(f"{work}/verify_chain_swapped.nii.gz"))}
json.dump(out, open(f"{work}/verify.json", "w"), indent=1); print(json.dumps(out, indent=1))
PYEOF
}

stage_provenance() {
  "$PY" - "$QMRI" "$WORK" "$REPORT" "$HERE" <<'PYEOF'
import sys, json, subprocess
from pathlib import Path
qmri, work, report, here = map(Path, sys.argv[1:5])
J = lambda p: json.load(open(p)) if Path(p).exists() else None
git = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
prov = {
 "pipeline": {"driver": str(here / "mese_pipeline.sh"), "fit": str(here / "fit_mese_t2.py"), "trxscan_git": git},
 "inputs": {"mese_AP_echoes": "TE 15/30/50/100 ms, TR 5.71/5.71/5.71/6.03 s, FA 90, SE-EPI, PE j-, "
                              "TRT 0.01728 s, PF 0.875, 256x256x48 @ 0.898x0.898x3 mm",
            "mese_PA": "echo-1 only, PE j, used for topup", "no_prescan_normalize_on_MESE": True},
 "sdc": "FSL topup (b02b0.cnf) on AP echo-1 + PA echo-1; applytopup --method=jac --interp=spline "
        "applied to the 4 AP echoes BEFORE fitting",
 "registration": "antsRegistration Rigid[0.1], MI(32 bins, 25% sampling), 4 levels; fixed = ACPC preproc "
                 "T2w with brain mask dilated 8 mm; moving = topup-corrected echo-3",
 "tr_correction": "echo-4 rescaled to TR 5.71 s with (1-exp(-5.71/T1))/(1-exp(-6.03/T1)); per-voxel T1 "
                  "from the ACPC dseg (1=CSF 2=GM 3=WM) resampled into MESE space (NN); nominal T1 "
                  "WM 0.83 / GM 1.33 / CSF 4.3 s",
 "fit": (J(work / "fit_native.json") or {}).get("model"),
 "bias": "N4BiasFieldCorrection (-s 2 -b [150] -c [50x50x50x50,1e-6]) on corrected echo-1 within the "
         "brain mask; field normalised to brain median 1 and divided out of S0 only (T2 is "
         "bias-invariant); raw S0 kept as desc-raw",
 "resampling": {"S0": "antsApplyTransforms Linear",
                "T2": "R2 = 1/T2 resampled Linear, T2 = 1/R2 where fit-mask coverage > 0.5",
                "acpc_grid": "qsiprep preproc DWI grid (1.7 mm ACPC, LPS)",
                "scanner_grids": "one-shot chain -t [scanner/from-scanner_to-ACPC_xfm.mat,1] "
                                 "-t mese2acpc_0GenericAffine.mat, i.e. the kit's ACPC->scanner call "
                                 "(scanner/resample.sh) composed with the MESE->ACPC rigid"},
 "prediction": (J(report / "metrics.json") or {}).get("prediction"),
 "baseline": (J(report / "metrics.json") or {}).get("baseline"),
 "verification": J(work / "verify.json"),
 "fit_summary": J(work / "fit_native.json"),
 "commands": (work / "commands.log").read_text().splitlines(),
}
(qmri / "provenance.json").write_text(json.dumps(prov, indent=1))
print("wrote", qmri / "provenance.json")
PYEOF
}

stages=("$@"); [[ ${#stages[@]} -eq 0 ]] && stages=(topup register fit resample predict scanner verify provenance)
for s in "${stages[@]}"; do echo "== stage $s"; "stage_$s"; done
echo "done"
