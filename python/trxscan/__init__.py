"""trxscan: diffusion-MRI simulation with acquisition artifacts and ground truth.

Typical use::

    import trxscan as ts
    from dipy.core.gradients import gradient_table

    phantom = ts.Phantom.load("sub-60501")
    gtab = gradient_table(bvals, bvecs)
    proto = ts.Protocol.HBCD.replace(voxel_mm=2.5, coils=8, accel=2)
    sim = phantom.simulate(gtab, proto, ts.Artifacts(noise=2e-4), slices=40)
    sim.magnitude            # nibabel image of the slice, every volume
    sim.kspace.acquired      # per-coil k-space, pre-GRAPPA
"""

from importlib.metadata import PackageNotFoundError, version as _version

from . import _core, objects
from .artifacts import Artifacts, Gre
from .bids import BidsDwi, Dataset
from .kspace import KSpace
from .motion import Motion
from .phantom import Fibers, Object, Phantom, Streamlines
from .protocol import EpiReadout, Protocol, Tissue
from .simulate import DroppedShot, GnlResult, GreResult, Simulation, TrxscanError
from .voxel import Voxel, VoxelSignal, VoxelTruth
from . import data  # noqa: E402  (after phantom/motion, which it imports)

try:
    __version__ = _version("trxscan")
except PackageNotFoundError:  # pragma: no cover
    __version__ = _core.core_version()


def set_threads(n: int) -> bool:
    """Size the parallel thread pool. Must run before the first simulation in a process;
    returns False if the pool already exists."""
    return _core.set_threads(int(n))


def scalar_names() -> list[str]:
    """The 27 ground-truth map names, in the order :meth:`Object.microstructure` returns them."""
    return list(_core.scalar_names())


__all__ = [
    "Artifacts", "BidsDwi", "Dataset", "DroppedShot", "EpiReadout", "Fibers", "GnlResult", "Gre", "GreResult", "KSpace", "Motion", "Object",
    "Phantom", "Protocol", "Simulation", "Streamlines", "Tissue", "TrxscanError", "Voxel", "VoxelSignal", "VoxelTruth",
    "data", "objects", "scalar_names", "set_threads", "__version__",
]
