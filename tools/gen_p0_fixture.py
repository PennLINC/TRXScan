"""Generate the P0 bit-identity fixture on TWO grids.

trxscan's production path is `--oversample 2` (the default), which reads the object from a
finer SIMULATION grid via `--sim-wm/--sim-gm/--sim-csf/--sim-mask/--sim-fmap` and reads the
ACQUISITION grid via `--wm/--gm/--csf/--mask` only for the output matrix
(`src/bin/trxscan.rs:372-405`). Without the `--sim-*` files the binary exits with an error,
and with `--oversample 1` it takes the legacy entry point instead, so a fixture on one grid
cannot gate `simulate_acquisition_oversampled` at all. This writes both grids, plus an FSL
scheme with a b0 row written the FSL way (bval > 0, zero bvec), and a few streamlines.

The phase-encode axis (y) is 32 lines, not 8. `acs_lines` is hard-coded to 24
(`src/bin/trxscan.rs:854`); on an 8-line axis every line is inside the ACS band and GRAPPA
calibrates but synthesizes nothing. On 32 lines it synthesizes lines 1, 3 and 29.

Run inside the simasl micromamba env for nibabel/numpy.
"""
import os
import numpy as np
import nibabel as nib
from nibabel.streamlines import Tractogram, TckFile

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "p0_baseline")
O = 2                                   # in-plane oversampling, trxscan's default
ACQ_DIMS = (8, 32, 4)
SIM_DIMS = (ACQ_DIMS[0] * O, ACQ_DIMS[1] * O, ACQ_DIMS[2])
ACQ_AFFINE = np.diag([2.0, 2.0, 3.0, 1.0])
# Sim cells 2X and 2X+1 tile acquired voxel X, whose centre is at 2X mm, so the sim cell
# centres sit at 2X-0.5 and 2X+0.5 mm. z is never oversampled.
SIM_AFFINE = np.array(
    [[1.0, 0.0, 0.0, -0.5], [0.0, 1.0, 0.0, -0.5], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


def write_volume(name, data, affine):
    img = nib.Nifti1Image(np.ascontiguousarray(data, dtype=np.float32), affine)
    img.to_filename(os.path.join(OUT, name))


def block_mean(v):
    """Simulation grid -> acquisition grid: mean over each O x O in-plane block."""
    x, y, z = v.shape
    return v.reshape(x // O, O, y // O, O, z).mean(axis=(1, 3))


def fieldmap(dims, affine):
    """A smooth, nonzero off-resonance field (Hz), evaluated in WORLD mm so the two grids
    describe the same field rather than two differently-stretched copies of one pattern."""
    ijk = np.indices(dims).reshape(3, -1).astype(float)
    xyz = affine[:3, :3] @ ijk + affine[:3, 3:4]
    f = 12.0 * np.sin(xyz[0] / 6.0) + 7.0 * np.cos(xyz[1] / 5.0)
    return f.reshape(dims)


def main():
    os.makedirs(OUT, exist_ok=True)

    # Tissue fractions on the SIM grid. The block starts at sim cell 5, halfway through
    # acquired voxel 2, so the acquisition grid carries a genuine partial-volume edge and
    # the rim of the FOV stays empty.
    wm = np.zeros(SIM_DIMS, dtype=np.float32)
    gm = np.zeros(SIM_DIMS, dtype=np.float32)
    csf = np.zeros(SIM_DIMS, dtype=np.float32)
    wm[5:12, 8:56, 1:3] = 0.6
    gm[5:12, 8:56, 1:3] = 0.3
    csf[5:12, 8:56, 1:3] = 0.1
    mask = (wm + gm + csf > 0).astype(np.float32)

    for name, v in [("wm", wm), ("gm", gm), ("csf", csf), ("mask", mask)]:
        write_volume(f"sim_{name}.nii.gz", v, SIM_AFFINE)
        acq = block_mean(v)
        if name == "mask":
            acq = (acq > 0).astype(np.float32)
        write_volume(f"{name}.nii.gz", acq, ACQ_AFFINE)

    write_volume("sim_fmap.nii.gz", fieldmap(SIM_DIMS, SIM_AFFINE), SIM_AFFINE)
    write_volume("fmap.nii.gz", fieldmap(ACQ_DIMS, ACQ_AFFINE), ACQ_AFFINE)

    # Scheme: one FSL-style b0 row (bval>0, zero bvec) plus three DWI rows.
    # The b0 row is the change-2 regression case and must not be bval=0.
    bvals = [5.0, 1000.0, 1000.0, 1000.0]
    bvecs = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
    ]
    with open(os.path.join(OUT, "scheme.bval"), "w") as f:
        f.write(" ".join(f"{b:g}" for b in bvals) + "\n")
    with open(os.path.join(OUT, "scheme.bvec"), "w") as f:
        for axis in range(3):
            f.write(" ".join(f"{v[axis]:.16g}" for v in bvecs) + "\n")

    # Streamlines in RAS mm, crossing the live block (x 4.5-11.5, y 7.5-55.5, slices 1-2)
    # along two directions so the rasteriser sees more than one orientation per voxel.
    lines = []
    for y in np.linspace(12.0, 50.0, 6):
        lines.append(np.array([[4.0, y, 4.5], [12.0, y, 4.5]], dtype=np.float32))
    for x in np.linspace(5.5, 10.5, 4):
        lines.append(np.array([[x, 6.0, 4.5], [x, 56.0, 4.5]], dtype=np.float32))
    tractogram = Tractogram(lines, affine_to_rasmm=np.eye(4))
    TckFile(tractogram).save(os.path.join(OUT, "streamlines.tck"))

    print("wrote fixture to", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
