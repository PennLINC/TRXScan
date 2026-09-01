"""Acceptance rules for the unringing suite (spec 5.2).

The criterion is that varying partial Fourier, phase, noise and windowing produces interpretable
and physically consistent changes in method behaviour -- NOT that methods achieve a predetermined
ranking. Relative quality between methods is recorded in the results table and never asserted.
"""
from collections import defaultdict

__all__ = ["check_consistency", "SHARPNESS_FLOOR"]

# Below this, a method has bought its ringing reduction with resolution.
SHARPNESS_FLOOR = 0.75


def _by(rows, **eq):
    return [r for r in rows if all(r.get(k) == v for k, v in eq.items())]


def check_consistency(rows):
    """Return (ok, reasons). Each reason names a physically implausible behaviour."""
    reasons = []
    methods = sorted({r["method"] for r in rows if r["method"] != "none"})

    for m in methods:
        # (a) a method must beat the no-op control wherever both were run
        for r in _by(rows, method=m):
            ctrl = _by(rows, method="none", pf=r["pf"], phase=r["phase"],
                       noisy=r.get("noisy", False))
            if ctrl and r["oscillatory_residual"] >= ctrl[0]["oscillatory_residual"]:
                reasons.append(
                    f"{m}: does not beat the control at pf={r['pf']} phase={r['phase']} "
                    f"({r['oscillatory_residual']:.4f} vs {ctrl[0]['oscillatory_residual']:.4f})"
                )

        # (b) more aggressive partial Fourier must not make ringing easier to remove
        byphase = defaultdict(list)
        for r in _by(rows, method=m):
            byphase[r["phase"]].append(r)
        for phase, rs in byphase.items():
            rs = sorted(rs, key=lambda r: -r["pf"])          # 1.0, 0.875, 0.75
            for a, b in zip(rs, rs[1:]):
                if b["oscillatory_residual"] < a["oscillatory_residual"]:
                    reasons.append(
                        f"{m}: residual improves as partial fourier gets more aggressive "
                        f"(pf {a['pf']} -> {b['pf']}, phase={phase}): "
                        f"{a['oscillatory_residual']:.4f} -> {b['oscillatory_residual']:.4f}"
                    )

        # (c) ringing reduction must not be bought with resolution
        for r in _by(rows, method=m):
            s = r.get("edge_sharpness")
            if s is not None and s < SHARPNESS_FLOOR:
                reasons.append(
                    f"{m}: edge sharpness {s:.2f} below floor {SHARPNESS_FLOOR} at "
                    f"pf={r['pf']} phase={r['phase']} -- suppressing ringing by blurring"
                )

    return (not reasons), reasons
