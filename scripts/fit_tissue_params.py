#!/usr/bin/env python3
"""Recommend --tissue-s0 and --diff-scale for trxscan from a compare_real_sim.py JSON.

Separable calibration: --tissue-s0 sets the b0 tissue levels (flat multiplier, no effect on
decay), --diff-scale sets the S(b)/S0 decay (no effect on b0). One or two iterations converge.

  s0_c   *= (real_b0_c / sim_b0_c) relative to WM   (WM is the global-scale reference -> 1.0)
  diff_c *= median_b ADC_real(b)/ADC_sim(b),  ADC = -ln(S/S0)/b   (slows a too-fast sim decay)

Usage: fit_tissue_params.py cmp.json [--s0 w,g,c] [--diff w,g,c]   (current multipliers, default 1,1,1)
"""
import argparse, json, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('json'); ap.add_argument('--s0', default='1,1,1'); ap.add_argument('--diff', default='1,1,1')
a = ap.parse_args()
d = json.load(open(a.json))
cur_s0 = [float(x) for x in a.s0.split(',')]; cur_diff = [float(x) for x in a.diff.split(',')]
tis = ['WM', 'GM', 'CSF']

# s0: match each tissue's b0 relative to WM (WM stays 1.0 as the global-scale anchor)
pt = d['per_tissue']
r_wm = pt['WM']['real_b0_mean']; s_wm = pt['WM']['sim_b0_mean']
new_s0 = []
for i, t in enumerate(tis):
    r_ratio = pt[t]['real_b0_mean'] / r_wm
    s_ratio = pt[t]['sim_b0_mean'] / s_wm
    corr = r_ratio / s_ratio if s_ratio > 0 else 1.0
    new_s0.append(cur_s0[i] * corr)
# normalize so WM stays exactly its current value (WM ratio is 1/1 -> corr 1, but keep explicit)
new_s0[0] = cur_s0[0]

# diff: match ADC per shell (median over shells), tissue by tissue
new_diff = []
for i, t in enumerate(tis):
    ratios = []
    for b, rs in pt[t]['real_S_over_S0'].items():
        b = float(b); ss = pt[t]['sim_S_over_S0'].get(str(int(b)), pt[t]['sim_S_over_S0'].get(b))
        if ss is None or ss <= 0 or rs <= 0: continue
        adc_r = -np.log(rs) / b; adc_s = -np.log(ss) / b
        if adc_s > 0: ratios.append(adc_r / adc_s)
    corr = float(np.median(ratios)) if ratios else 1.0
    new_diff.append(cur_diff[i] * corr)

print("current   s0 =", ','.join(f'{x:.3f}' for x in cur_s0), " diff =", ','.join(f'{x:.3f}' for x in cur_diff))
print("recommend s0 =", ','.join(f'{x:.3f}' for x in new_s0), " diff =", ','.join(f'{x:.3f}' for x in new_diff))
print()
print("  --tissue-s0 " + ','.join(f'{x:.3f}' for x in new_s0) + " --diff-scale " + ','.join(f'{x:.3f}' for x in new_diff))
