#!/usr/bin/env python3
"""Generate a smooth, center-high noise-level (SD) map for trxscan --noise-map.

Real complex noise maps peak in the CENTER of the brain (the g-factor of a peripheral head-coil
array: the middle of the brain is farthest from every coil, so parallel-imaging noise is amplified
there). This builds that pattern cleanly from the brain mask alone: noise is highest at the mask's
center of mass and is downweighted smoothly toward the edge.

  sigma(x) = base * (1 + (ratio-1) * w(x)),   w = 1 at the centre, 0 at the brain edge

where w is a smooth function of the distance from the mask centre (falloff^power). `base` is the
peripheral per-component SD (sim intensity units); `ratio` is centre/periphery (real ≈ 15-20).

Usage: make_noise_map.py --mask mask.nii.gz --out sigma.nii.gz [--base 0.15] [--ratio 15]
                         [--power 2] [--smooth 2]
"""
import argparse
import numpy as np, nibabel as nb
from scipy.ndimage import gaussian_filter

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--mask', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--base', type=float, default=0.15, help='peripheral per-component noise SD (sim units)')
ap.add_argument('--ratio', type=float, default=15.0, help='centre/periphery noise ratio (real ~15-20)')
ap.add_argument('--power', type=float, default=2.0, help='falloff exponent (higher = more peaked centre)')
ap.add_argument('--smooth', type=float, default=2.0, help='Gaussian smoothing (voxels)')
a = ap.parse_args()

mim = nb.load(a.mask); M = mim.get_fdata() > 0.5
com = np.array(np.nonzero(M)).mean(1)
sh = M.shape
I, J, K = np.mgrid[0:sh[0], 0:sh[1], 0:sh[2]]
# distance from the mask centre of mass, normalized by the max in-mask distance
d = np.sqrt(((I - com[0]) / sh[0])**2 + ((J - com[1]) / sh[1])**2 + ((K - com[2]) / sh[2])**2)
dn = np.clip(d / (d[M].max() + 1e-9), 0, 1)
w = (1.0 - dn) ** a.power                      # 1 at centre, 0 at edge
sigma = a.base * (1.0 + (a.ratio - 1.0) * w)   # base at edge, base*ratio at centre
sigma = gaussian_filter(sigma.astype(np.float32), a.smooth)
sigma[sigma < a.base] = a.base                 # never below the peripheral floor (noise everywhere)
nb.save(nb.Nifti1Image(sigma.astype(np.float32), mim.affine, mim.header), a.out)
c = float(sigma[M & (dn < 0.2)].mean()); p = float(sigma[M & (dn > 0.7)].mean())
print(f"wrote {a.out}: centre {c:.3f}  periphery {p:.3f}  ratio {c/p:.1f}")
