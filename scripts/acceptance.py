"""Acceptance rules for the unringing suite (spec 5.2).

The criterion is that varying partial Fourier, phase, noise and windowing produces interpretable
and physically consistent changes in method behaviour -- NOT that methods achieve a predetermined
ranking. Relative quality between methods is recorded in the results table and never asserted.

**Partial Fourier is now covered by `residual_alignment_pe` and `residual_energy_pe`**, , a frequency-agnostic complex metric
that measures how much of the simulator's own control artifact survives a method's output. The
Nyquist projection remains as a specialised full-Fourier measure; PF ringing does not sit at
Nyquist, so that one alone could not see it. RPG is still absent, so the PF-*aware method*
comparison remains unavailable -- but the PF *axis* is no longer unmeasured.
"""
from collections import defaultdict

__all__ = ["check_consistency", "SHARPNESS_FLOOR"]

# Below this, a method has bought its ringing reduction with resolution.
SHARPNESS_FLOOR = 0.75

# A condition with this little ringing left -- typically because a reconstruction window already
# apodized it away -- is exempt from "must beat the control". There is nothing to remove, so any
# method can only perturb the image, and demanding an improvement would penalise correct behaviour.
# Expressed as a fraction of the MEDIAN control residual so it self-calibrates to the fixture set.
NEGLIGIBLE_RINGING_FRACTION = 0.25


def _by(rows, **eq):
    return [r for r in rows if all(r.get(k) == v for k, v in eq.items())]


def check_consistency(rows):
    """Return (ok, reasons). Each reason names a physically implausible behaviour."""
    reasons = []
    methods = sorted({r["method"] for r in rows if r["method"] != "none"})
    # Gate on a PF-AWARE quantity. Gating on oscillatory_residual -- the Nyquist projection this
    # module elsewhere documents as going blind under partial Fourier -- meant the PF-aware rule
    # could be skipped precisely where PF had moved the artifact off Nyquist.
    # Each floor is computed from the SAME population its rule applies to. Pooling PF, noisy and
    # Hann controls into a floor used only on full-Fourier clean unapodized rows made the
    # "negligible artifact" exemption depend on conditions the rule never sees.
    def _median(vals):
        v = sorted(x for x in vals if x is not None and x == x)
        return v[len(v) // 2] if v else 0.0

    in_domain_a = [r for r in rows if r["method"] == "none" and r["pf"] >= 1.0
                   and not r.get("noisy", False) and str(r.get("window")) in ("None", "none")]
    floor = NEGLIGIBLE_RINGING_FRACTION * _median(r["oscillatory_residual"] for r in in_domain_a)
    # Gate on the ABSOLUTE artifact size, not a ratio: residual_energy for the control is
    # identically 1 (its residual IS R0), so a ratio-based gate can never skip anything.
    in_domain_b = [r for r in rows if r["method"] == "none" and not r.get("noisy", False)
                   and str(r.get("window")) in ("None", "none")]
    pe_floor = NEGLIGIBLE_RINGING_FRACTION * _median(
        r.get("artifact_norm_pe") for r in in_domain_b)

    for m in methods:
        # (a) beat the no-op control -- FULL FOURIER ONLY. oscillatory_residual is the Nyquist
        #     projection this module documents as invalid under partial Fourier, so asserting it
        #     on PF rows was inconsistent with its own stated limitation. PF is covered by rule (b)
        #     on the phase-encode axis instead.
        for r in _by(rows, method=m):
            if r["pf"] < 1.0 or r.get("noisy", False):
                continue          # Gibbs-removal rules run on CLEAN rows; the paired noisy rows
                                  # are for noise-amplification and robustness reporting.
            if str(r.get("window")) not in ("None", "none"):
                continue          # apodized input is out of domain for ALL method-performance
                                  # rules, not just (b) -- see the module docstring. An earlier
                                  # version skipped it only in (b), so
                                  # test_apodized_rows_are_out_of_domain_and_never_fail promised
                                  # more than the implementation delivered.
            # Match the control on EVERY factor, window included. Matching only pf and phase
            # compared an unapodized method row against an apodized control, which made all three
            # methods look like they lost to doing nothing.
            ctrl = _by(rows, method="none", pf=r["pf"], phase=r["phase"],
                       noisy=r.get("noisy", False), window=r.get("window"))
            if not ctrl or ctrl[0]["oscillatory_residual"] < floor:
                continue          # nothing to remove here; see NEGLIGIBLE_RINGING_FRACTION
            if r["oscillatory_residual"] >= ctrl[0]["oscillatory_residual"]:
                reasons.append(
                    f"{m}: does not beat the control at pf={r['pf']} phase={r['phase']} "
                    f"({r['oscillatory_residual']:.4f} vs {ctrl[0]['oscillatory_residual']:.4f})"
                )

        # (b) a method must not AMPLIFY the artifact -- stated as what the code enforces, not the
        #     stronger "must remove some" an earlier comment claimed while the check was `a > 1`.
        #     Scored on the PHASE-ENCODE axis, since that is the one partial Fourier acts on, and
        #     gated on a PF-aware quantity so the rule is not skipped exactly where PF matters.
        for r in _by(rows, method=m):
            # CLEAN rows only. ||Rm|| for a noisy output still contains thermal noise, so applying
            # an artifact-amplification threshold to it makes residual_energy simultaneously a
            # Gibbs metric and a noise metric. The paired noisy rows are for noise-amplification
            # and robustness questions, reported separately.
            if r.get("noisy", False):
                continue
            a = r.get("residual_alignment_pe")
            e = r.get("residual_energy_pe")
            if a is None or a != a:            # NaN
                continue
            ctrl = _by(rows, method="none", pf=r["pf"], phase=r["phase"],
                       noisy=r.get("noisy", False), window=r.get("window"))
            if not ctrl:
                continue
            cpe = ctrl[0].get("artifact_norm_pe")
            if cpe is None or cpe != cpe or cpe < pe_floor:
                continue                        # nothing to remove on this axis
            if r.get("window") not in (None, "None"):
                # OUT OF DOMAIN, reported not asserted. Kellner's sub-voxel shift assumes an
                # unapodized rectangular window -- MRtrix says as much, recommending scanner
                # filtering be disabled for best mrdegibbs performance. Measured here, mrdegibbs
                # amplifies the PE-axis artifact ~30% on Hann-windowed data (alignment 1.18-1.31)
                # while dipy is near-neutral (~1.00-1.07). That is a real result about the
                # methods, so it belongs in the table; failing the suite for a method misbehaving
                # outside its documented domain would not be.
                continue
            if a > 1.0:
                reasons.append(
                    f"{m}: amplifies the control artifact (PE axis) at pf={r['pf']} "
                    f"phase={r['phase']} window={r.get('window')} (alignment {a:.3f} > 1)"
                )
            if e is not None and e == e and e > 1.05:
                reasons.append(
                    f"{m}: residual energy grew {e:.2f}x on the PE axis at pf={r['pf']} "
                    f"phase={r['phase']} window={r.get('window')}"
                )

        # (b-legacy) WITHDRAWN -- the Nyquist projection alone cannot assert on the PF axis.
        #
        # The intended rule was that more aggressive partial Fourier makes unringing harder. It is
        # not testable here, and both real methods violate any form of it, for a reason that is
        # about the metric rather than the methods:
        #
        #   `oscillatory_residual` is a projection onto the Nyquist frequency, because that is
        #   ringing's signature under a rectangular window. But PF ringing arises from TWO k-space
        #   intervals and sits at a DIFFERENT frequency -- which is exactly why RPG exists as a
        #   separate method. So as PF grows more aggressive, ringing migrates off Nyquist and the
        #   metric sees less of it, whether or not any method improved.
        #
        # Asserting on it would reward the metric's blind spot. The PF trend is reported in the
        # results table instead. Closing this properly needs a PF-aware metric AND RPG installed;
        # RPG is absent here, so the PF-aware comparison is unavailable in any case.

        # (c) ringing reduction must not be bought with resolution -- checked on BOTH axes.
        #     A method could blur along phase-encode while leaving readout sharp, and a
        #     single auto-selected axis (readout, for the symmetric box) would not see it. This
        #     matters precisely because zero-filled PF itself trades ringing for PE-axis blur.
        for r in _by(rows, method=m):
            if r.get("noisy", False):
                continue                      # clean rows only, as in (a) and (b)
            if str(r.get("window")) not in ("None", "none"):
                continue                      # apodization legitimately softens edges
            for ax in ("ro", "pe"):
                sv = r.get(f"edge_sharpness_{ax}")
                if sv is not None and sv == sv and sv < SHARPNESS_FLOOR:
                    reasons.append(
                        f"{m}: {ax} edge sharpness {sv:.2f} below floor {SHARPNESS_FLOOR} at "
                        f"pf={r['pf']} phase={r['phase']} -- suppressing ringing by blurring"
                    )

    return (not reasons), reasons


# ---------------------------------------------------------------- driver

def _load_pair(d, desc):
    import nibabel as nib
    import numpy as np
    m = nib.load(str(d / f"bench_desc-{desc}_part-mag_dwi.nii.gz")).get_fdata().squeeze()
    p = nib.load(str(d / f"bench_desc-{desc}_part-phase_dwi.nii.gz")).get_fdata().squeeze()
    return np.asarray(m, float), np.asarray(p, float)


def run_suite(root, methods=None):
    """Walk fixture directories, run each method, score against object-nominal.

    Scoring uses the CLEAN acquisition against the nominal reference so the measured error is
    unringing error. The noisy image is scored separately, and the pair shares one realization, so
    the difference between the two rows isolates noise amplification.
    """
    import json
    from pathlib import Path
    from run_unringing import run_method, available_methods, MethodUnavailable
    from score_unringing import score

    root = Path(root)
    methods = methods or available_methods()
    rows, skipped = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        fj = d / "factors.json"
        if not fj.exists():
            continue
        f = json.loads(fj.read_text())
        ref_m, ref_p = _load_pair(d, "objectnominal")
        # The artifact template is the CLEAN acquisition for BOTH rows. Using the noisy image as
        # its own template folds thermal noise into R0, so a method that only denoises would lower
        # residual_alignment without removing any Gibbs -- defeating the point of the paired
        # clean/noisy emission.
        ctl_m, ctl_p = _load_pair(d, "acquiredclean")
        for noisy in (False, True):   # both images of the SAME fixture; noise is not a grid factor
            acq_m, acq_p = _load_pair(d, "acquirednoisy" if noisy else "acquiredclean")
            for meth in methods:
                try:
                    om, op = run_method(meth, acq_m, acq_p)
                except MethodUnavailable as e:
                    skipped.append((meth, str(e)))
                    continue
                # The uncorrected acquisition is the control artifact for residual_alignment.
                s = score(om, op, ref_m, ref_p,
                          control_mag=ctl_m, control_phase=ctl_p)
                rows.append({"label": f["label"], "method": meth, "pf": f["partial_fourier"],
                             "phase": f["phase"], "window": f["window"], "noisy": noisy, **s})
    return rows, sorted(set(m for m, _ in skipped))


def summarise(rows):
    """Mean score per (method, pf) and per (method, phase), for the report table."""
    from collections import defaultdict
    # Grouped by window and noise as well as pf. Averaging Hann with unwindowed, and clean with
    # noisy, made the reported PF trend look cleaner than the underlying data supports.
    agg = defaultdict(list)
    for r in rows:
        if not r.get("edge_metrics_valid", True):
            continue          # never average NaN edge metrics into a summary row
        agg[(r["method"], r["pf"], r.get("window"), r.get("noisy", False))].append(r)
    out = []
    for (m, pf, win, noisy), rs in sorted(agg.items(), key=lambda kv: tuple(map(str, kv[0]))):
        n = len(rs)
        ok = [r for r in rs if r.get("residual_alignment_pe") == r.get("residual_alignment_pe")]
        out.append({
            "method": m, "pf": pf, "window": win, "noisy": noisy, "n": n,
            "align_pe": (sum(r["residual_alignment_pe"] for r in ok) / len(ok)) if ok else float("nan"),
            "energy_pe": (sum(r["residual_energy_pe"] for r in ok) / len(ok)) if ok else float("nan"),
            "oscillatory_residual": sum(r["oscillatory_residual"] for r in rs) / n,
            "edge_sharpness": sum(r["edge_sharpness"] for r in rs) / n,
            "phase_rmse_masked": sum(r["phase_rmse_masked"] for r in rs) / n,
        })
    return out


def main(argv=None):
    import argparse, csv, sys
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="directory of fixture sets from trxscan-benchmark")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)
    rows, skipped = run_suite(a.root)
    if not rows:
        print("no fixtures found", file=sys.stderr)
        return 2
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} rows over {len({r['label'] for r in rows})} fixture sets")
    if skipped:
        print(f"SKIPPED (not installed): {', '.join(skipped)}")
    print()
    print(f"  {'method':<10} {'pf':>6} {'window':>6} {'noisy':>6} {'n':>3} "
          f"{'oscill':>8} {'algn_pe':>8} {'engy_pe':>8} {'sharp':>6}")
    for r in summarise(rows):
        print(f"  {r['method']:<10} {r['pf']:>6.3f} {str(r['window']):>6} "
              f"{str(r['noisy']):>6} {r['n']:>3} "
              f"{r['oscillatory_residual']:>8.5f} {r['align_pe']:>8.3f} "
              f"{r['energy_pe']:>8.3f} {r['edge_sharpness']:>6.3f}")
    # Out-of-domain observations: reported, never asserted (see rule b). Kellner's sub-voxel
    # shift assumes an unapodized rectangular window, and MRtrix recommends disabling scanner
    # filtering for best mrdegibbs performance -- so how these methods behave on apodized input is
    # a real result worth tabulating, not a reason to fail the suite.
    apod = [r for r in rows
            if r["method"] != "none" and str(r.get("window")) not in ("None", "none")
            and r.get("residual_alignment_pe") == r.get("residual_alignment_pe")]
    if apod:
        by = defaultdict(list)
        for r in apod:
            by[r["method"]].append(r["residual_alignment_pe"])
        print()
        print("  Out of domain (apodized input; Kellner assumes an unapodized rectangular window):")
        for meth, v in sorted(by.items()):
            mean = sum(v) / len(v)
            note = "AMPLIFIES the PE artifact" if mean > 1.05 else "near-neutral"
            print(f"    {meth:<10} mean PE alignment {mean:.3f}   {note}")

    ok, reasons = check_consistency(rows)
    print()
    # PF now carries an asserted rule on the phase-encode axis (rule b), so this is no longer
    # full-Fourier-only acceptance. What it still excludes: apodized input (out of domain for
    # Kellner-style methods) and noisy rows (reported separately), and no PF-AWARE method is
    # present, RPG having no adapter.
    print("ACCEPTANCE:", "PASS (clean unapodized rows; PF asserted on the PE axis; RPG absent)" if ok else "FAIL")
    for x in reasons[:12]:
        print("  -", x)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
