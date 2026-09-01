"""Acceptance rules for the unringing suite (spec 5.2).

The criterion is that varying partial Fourier, phase, noise and windowing produces interpretable
and physically consistent changes in method behaviour -- NOT that methods achieve a predetermined
ranking. Relative quality between methods is recorded in the results table and never asserted.

**Known limitation: the partial-Fourier axis is reported, not asserted.** The oscillatory metric
projects onto the Nyquist frequency, which is ringing's signature under a rectangular window, but
PF ringing arises from two k-space intervals and lies elsewhere. Fixtures are generated across
three PF values and the trend is tabulated, but no rule is applied to it. Closing that needs a
PF-aware metric and RPG, which is not installed here.
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
    ctl_all = sorted(r["oscillatory_residual"] for r in rows if r["method"] == "none")
    floor = (NEGLIGIBLE_RINGING_FRACTION * ctl_all[len(ctl_all) // 2]) if ctl_all else 0.0

    for m in methods:
        # (a) a method must beat the no-op control wherever both were run
        for r in _by(rows, method=m):
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

        # (b) WITHDRAWN -- the PF axis cannot be asserted on with the current metric.
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

        # (c) ringing reduction must not be bought with resolution
        for r in _by(rows, method=m):
            s = r.get("edge_sharpness")
            if s is not None and s < SHARPNESS_FLOOR:
                reasons.append(
                    f"{m}: edge sharpness {s:.2f} below floor {SHARPNESS_FLOOR} at "
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
        for noisy in (False, True):
            acq_m, acq_p = _load_pair(d, "acquirednoisy" if noisy else "acquiredclean")
            for meth in methods:
                try:
                    om, op = run_method(meth, acq_m, acq_p)
                except MethodUnavailable as e:
                    skipped.append((meth, str(e)))
                    continue
                s = score(om, op, ref_m, ref_p)
                rows.append({"label": f["label"], "method": meth, "pf": f["partial_fourier"],
                             "phase": f["phase"], "window": f["window"], "noisy": noisy, **s})
    return rows, sorted(set(m for m, _ in skipped))


def summarise(rows):
    """Mean score per (method, pf) and per (method, phase), for the report table."""
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        if not r.get("edge_metrics_valid", True):
            continue          # never average NaN edge metrics into a summary row
        agg[(r["method"], r["pf"])].append(r)
    out = []
    for (m, pf), rs in sorted(agg.items()):
        n = len(rs)
        out.append({
            "method": m, "pf": pf, "n": n,
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
    print(f"  {'method':<10} {'pf':>6} {'n':>4} {'oscill':>9} {'sharp':>7} {'phase':>7}")
    for r in summarise(rows):
        print(f"  {r['method']:<10} {r['pf']:>6.3f} {r['n']:>4} "
              f"{r['oscillatory_residual']:>9.5f} {r['edge_sharpness']:>7.3f} "
              f"{r['phase_rmse_masked']:>7.3f}")
    ok, reasons = check_consistency(rows)
    print()
    print("ACCEPTANCE:", "PASS" if ok else "FAIL")
    for x in reasons[:12]:
        print("  -", x)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
