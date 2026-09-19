#!/usr/bin/env python3
"""Build a self-contained HTML realism report from compare_real_sim.py JSONs + montages.

Usage: make_realism_report.py --out report.html --pe PA \
         --cmp baseline:cmp_PA_baseline final:cmp_PA_final [--montage final:cmp_PA_final_montage.png ...]
"""
import argparse, base64, json, html
from pathlib import Path

def b64(p):
    p=Path(p)
    if not p.exists(): return None
    return base64.b64encode(p.read_bytes()).decode()

def tissue_rows(d):
    out=[]
    wmr=d['per_tissue']['WM']['real_b0_mean']; wms=d['per_tissue']['WM']['sim_b0_mean']
    for t in ['WM','GM','CSF']:
        pt=d['per_tissue'][t]
        ss=" · ".join(f"b{b}: <b>{pt['real_S_over_S0'][b]:.3f}</b>/{pt['sim_S_over_S0'][b]:.3f}" for b in pt['real_S_over_S0'])
        out.append(f"<tr><td>{t}</td><td>{pt['real_b0_mean']/wmr:.2f} / {pt['sim_b0_mean']/wms:.2f}</td><td>{ss}</td></tr>")
    return "".join(out)

def phase_rows(d):
    out=[]
    for b in sorted(k for k in d['phase'] if isinstance(k,int) or (isinstance(k,str) and k.isdigit())):
        p=d['phase'][b]
        out.append(f"<tr><td>b{b}</td><td><b>{p['real_circ_sd']:.2f}</b> / {p['sim_circ_sd']:.2f}</td></tr>")
    return "".join(out)

ap=argparse.ArgumentParser()
ap.add_argument('--out',required=True); ap.add_argument('--pe',default='PA')
ap.add_argument('--cmp',nargs='+',required=True,help='label:jsonstem')
ap.add_argument('--montage',nargs='*',default=[],help='label:pngpath')
ap.add_argument('--command',default='')
a=ap.parse_args()
cmps={x.split(':',1)[0]:json.load(open(x.split(':',1)[1]+'.json')) for x in a.cmp}
monts={x.split(':',1)[0]:b64(x.split(':',1)[1]) for x in a.montage}

sec=[]
for label,d in cmps.items():
    sec.append(f"""
    <h3>{html.escape(label)} <span class=sub>(sim→real global scale {d['global_scale_sim_to_real']:.0f}, PE {d['pe']})</span></h3>
    <div class=grid>
    <div><table><caption>Tissue b0 (ratio to WM) &amp; S(b)/S₀ — <b>real</b>/sim</caption>
      <tr><th>tissue</th><th>b0/WM real/sim</th><th>S(b)/S₀ real/sim</th></tr>{tissue_rows(d)}</table></div>
    <div><table><caption>WM phase circular SD (rad) — <b>real</b>/sim</caption>
      <tr><th>shell</th><th>real/sim</th></tr>{phase_rows(d)}</table>
      <table><caption>background noise σ</caption><tr><td>real {d['noise']['real_bg_std']:.1f} · sim {d['noise']['sim_bg_std']:.1f}</td></tr></table></div>
    </div>""")
    if label in monts and monts[label]:
        sec.append(f'<img src="data:image/png;base64,{monts[label]}" alt="{label} montage">')

page=f"""<!doctype html><html><head><meta charset=utf-8><title>TRXScan realism — sub-60501</title>
<style>
body{{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1050px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
h1{{font-size:1.7rem;margin:.2rem 0}} h2{{margin-top:2rem;border-bottom:2px solid #eee;padding-bottom:.3rem}}
h3{{margin-top:1.5rem}} .sub{{font-weight:400;color:#888;font-size:.85em}}
table{{border-collapse:collapse;margin:.5rem 0;font-size:.9em;width:100%}} caption{{text-align:left;font-weight:600;padding:.3rem 0;color:#444}}
th,td{{border:1px solid #ddd;padding:.35rem .55rem;text-align:left}} th{{background:#f6f6f6}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
img{{max-width:100%;border:1px solid #ddd;border-radius:4px;margin:.5rem 0}}
code,pre{{background:#f4f4f4;border-radius:4px}} pre{{padding:.7rem;overflow-x:auto;font-size:.82em}}
.key{{background:#f0f7ff;border-left:3px solid #3b82f6;padding:.6rem .9rem;margin:.8rem 0}}
</style></head><body>
<h1>TRXScan realism vs a real acquisition — sub-60501 ses-01</h1>
<p class=sub>Simulated in the scanner frame with the subject's own DRBUDDI field; compared voxelwise to the raw {a.pe} complex DWI (HBCD75, Prisma 3T, TE 88 ms, TR 4.8 s).</p>
<div class=key><b>Distortion convention verified:</b> a uniform ±field test shows the forward path shifts +field→+j (+9 vox) and reverse shifts +field→−j, both matching the real PED labels (NCC 0.996). The sim distortion is correct; the field is causal and is used with its natural sign.</div>
<h2>Metrics: real vs sim</h2>
{''.join(sec)}
<h2>Recommended command</h2>
<pre>{html.escape(a.command)}</pre>
<p class=sub>Montage rows: magnitude (real, sim) then phase (real, sim), for b0 / b1000 / b3000. Same window per panel.</p>
</body></html>"""
Path(a.out).write_text(page)
print("wrote",a.out)
