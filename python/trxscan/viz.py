"""Visualisation hand-offs: write what TRXViz renders (TRX tractograms, ODX orientation files
built from the simulator's own mixture and truth peaks) and drive ``trxviz-cli`` to a PNG.

Needs ``trx-python`` for TRX writing (a base dependency since 0.2.1) and the ``odx`` package (the
Python binding of odx-rs, TRXViz's orientation-data reader) for ODX (``pip install
trxscan[viz]``). Rendering needs a ``trxviz-cli`` binary: ``$TRXVIZ_CLI``, ``trxviz-cli`` on
PATH, or one installed with :func:`install_trxviz`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ._arrays import vectors, volume

if TYPE_CHECKING:  # pragma: no cover
    from .phantom import Streamlines
    from .simulate import Simulation


# ─── TRX ────────────────────────────────────────────────────────────────────


def write_trx(streamlines: "Streamlines", path: str | Path, *, reference: Any = None, weights_name: str = "sift2_weights") -> Path:
    """Write a :class:`~trxscan.phantom.Streamlines` as a TRX file (positions in RAS mm),
    with the weights as a per-streamline field."""
    try:
        from trx.trx_file_memmap import TrxFile
    except ImportError as e:  # pragma: no cover
        raise ImportError("write_trx needs trx-python: pip install trx-python") from e
    from nibabel.streamlines import ArraySequence, Tractogram

    try:  # trx-python type-checks against dipy's StatefulTractogram without importing it
        import dipy.io.stateful_tractogram  # noqa: F401
    except ImportError:
        pass

    off = streamlines.offsets.astype(np.int64)
    seq = ArraySequence()
    seq._data = np.asarray(streamlines.positions, dtype=np.float32)
    seq._offsets = off[:-1]
    seq._lengths = np.diff(off)
    dps = {}
    if streamlines.weights is not None:
        dps[weights_name] = np.asarray(streamlines.weights, dtype=np.float32)
    tg = Tractogram(seq, data_per_streamline=dps, affine_to_rasmm=np.eye(4))
    import nibabel as nib

    if reference is None:
        reference = nib.Nifti1Image(np.zeros((1, 1, 1), np.float32), np.eye(4))
    elif not hasattr(reference, "affine"):
        reference = nib.load(str(reference))
    trx = TrxFile.from_tractogram(tg, reference=reference)
    from trx.trx_file_memmap import save as trx_save

    trx_save(trx, str(path))
    trx.close()
    return Path(path)


# ─── ODX ────────────────────────────────────────────────────────────────────


def _match_hemisphere(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """For each destination vertex, the source vertex nearest up to sign, and the sign.
    Returns (index (n_dst,), sign (n_dst,), worst angle in degrees)."""
    dots = dst @ src.T
    idx = np.abs(dots).argmax(axis=1)
    best = np.abs(dots[np.arange(dst.shape[0]), idx])
    sign = np.sign(dots[np.arange(dst.shape[0]), idx])
    worst = float(np.degrees(np.arccos(np.clip(best.min(), -1, 1))))
    return idx, sign, worst


def mixture_to_odx(mixture: Any, dims: tuple[int, int, int], affine: np.ndarray, *, peaks: Any = None, name: str = "trxscan"):
    """An ``odx.Odx`` holding the mixture's ODF (the orientation histogram, resampled by
    nearest vertex onto the 321-vertex ODF8 hemisphere TRXViz knows), WM/GM/CSF fractions as
    per-voxel scalars and, optionally, the truth peaks as fixels (direction scaled by mass
    fraction).

    ``mixture`` is the handle from ``Simulation.mixture``; ``dims``/``affine`` its grid (the
    simulation grid: ``sim.object.sim_dims``, ``sim.object.sim_affine``). ``peaks`` is the
    truth-peaks image (9 volumes) on the ACQUISITION grid, in which case the ODX is written on
    that grid instead (pass ``dims``/``affine`` of the acquisition grid).
    """
    try:
        import odx
    except ImportError as e:  # pragma: no cover
        raise ImportError("mixture_to_odx needs the odx package: pip install trxscan[viz]") from e
    verts_src = np.asarray(mixture.sphere_vertices(), dtype=np.float64).reshape(-1, 3)
    verts_dst, faces = odx.spheres.dsistudio_odf8()
    verts_dst = np.asarray(verts_dst, dtype=np.float64)
    # TRXScan's icosphere(3) hemisphere and DSI Studio's ODF8 are different tessellations of
    # the same size (321 vertices, up to ~5 degrees apart), so the ODF is resampled onto ODF8 by
    # nearest vertex: piecewise-constant, adequate for glyphs. Anything worse than that is a bug.
    idx, sign, worst = _match_hemisphere(verts_src, verts_dst)
    if worst > 8.0:
        warnings.warn(f"hemisphere vertices differ from ODF8 by up to {worst:.1f} deg; check the sphere", stacklevel=2)
    nvert = verts_src.shape[0]
    odf = np.asarray(mixture.odf(), dtype=np.float32).reshape(-1, nvert)[:, idx]
    wm, gm, csf = (np.asarray(v) for v in mixture.fractions())
    fallback = np.asarray(mixture.fallback())
    mdims = tuple(int(v) for v in mixture.dims)
    mask_f = (fallback == 0) & (wm > 0)                       # flat, F-order (TRXScan)
    # ODX stores the mask and orders per-voxel rows in C order over (i, j, k)
    mask_c = np.ascontiguousarray(volume(mask_f, mdims)).ravel(order="C").astype(np.uint8)
    sel_c = np.flatnonzero(mask_c)
    sel_f = np.ravel_multi_index(np.unravel_index(sel_c, mdims, order="C"), mdims, order="F")
    b = odx.OdxBuilder(np.ascontiguousarray(affine, dtype=np.float64), mdims, mask_c)
    b.set_sphere(np.ascontiguousarray(verts_dst, dtype=np.float32), np.ascontiguousarray(faces, dtype=np.uint32))
    b.set_odf("odf", np.ascontiguousarray(odf[sel_f], dtype=np.float32))
    for nm, arr in (("wm", wm), ("gm", gm), ("csf", csf)):
        b.set_dpv(nm, np.ascontiguousarray(arr[sel_f], dtype=np.float32))
    if peaks is not None:
        pk = np.asarray(peaks.dataobj if hasattr(peaks, "dataobj") else peaks, dtype=np.float32)
        if tuple(pk.shape[:3]) != mdims:
            raise ValueError("peaks must be on the same grid as the mixture (build the mixture on the acquisition grid via Object.microstructure(...) or pass peaks=None)")
        pk = pk.reshape(mdims + (3, 3))
        for v in sel_f:
            x, y, z = np.unravel_index(v, mdims, order="F")
            arr = pk[x, y, z]
            good = np.linalg.norm(arr, axis=-1) > 0
            b.push_voxel_peaks(np.ascontiguousarray(arr[good], dtype=np.float32).reshape(-1, 3))
    else:
        b.skip_all_peaks()
    b.set_metadata({"source": name, "generator": "trxscan"})
    return b.finalize()


# ─── TRXViz ─────────────────────────────────────────────────────────────────


def find_trxviz() -> Path | None:
    """The ``trxviz-cli`` binary: ``$TRXVIZ_CLI``, then PATH, then the local install."""
    env = os.environ.get("TRXVIZ_CLI")
    if env and Path(env).exists():
        return Path(env)
    which = shutil.which("trxviz-cli")
    if which:
        return Path(which)
    local = _install_dir()
    if local.exists():
        for p in local.rglob("trxviz-cli*"):
            if p.is_file() and os.access(p, os.X_OK):
                return p
    return None


def _install_dir() -> Path:
    import pooch

    return Path(pooch.os_cache("trxscan")) / "trxviz"


def install_trxviz(version: str = "latest", *, url: str | None = None) -> Path:
    """Download a TRXViz CLI release (tee-ar-ex/TRXViz) for this platform into the cache.
    Explicit and opt-in; nothing is downloaded otherwise."""
    import platform
    import tarfile
    import zipfile

    import pooch

    if url is None:
        sysname = platform.system().lower()
        arch = platform.machine().lower()
        base = f"https://github.com/tee-ar-ex/TRXViz/releases/{'latest/download' if version == 'latest' else 'download/' + version}"
        tag = {"linux": "linux", "darwin": "macos", "windows": "windows"}.get(sysname, sysname)
        url = f"{base}/trxviz-cli-{tag}-{arch}.tar.gz" if sysname != "windows" else f"{base}/trxviz-cli-{tag}-{arch}.zip"
    dest = _install_dir() / version
    dest.mkdir(parents=True, exist_ok=True)
    archive = pooch.retrieve(url, known_hash=None, path=dest)
    if str(archive).endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest)
    exe = find_trxviz()
    if exe is None:
        raise RuntimeError(f"downloaded {url} but found no trxviz-cli binary in {dest}")
    exe.chmod(0o755)
    return exe


def render(
    *,
    tractogram: str | Path | Sequence[str | Path] | None = None,
    nifti: str | Path | Sequence[str | Path] | None = None,
    odx: str | Path | Sequence[str | Path] | None = None,
    project: str | Path | None = None,
    out: str | Path | None = None,
    width: int = 1200,
    height: int = 900,
    view: str = "3d",
    azimuth: float | None = None,
    elevation: float | None = None,
    distance: float | None = None,
    target: Sequence[float] | None = None,
    env: dict[str, str] | None = None,
) -> bytes:
    """Render with ``trxviz-cli render`` and return the PNG bytes (also written to ``out`` if
    given). Either loose assets or a saved workflow ``project`` (needed for 2-D slice views and
    lighting/colormap settings). On a machine without a GPU, TRXViz needs Mesa llvmpipe
    (``LIBGL_ALWAYS_SOFTWARE=1``)."""
    exe = find_trxviz()
    if exe is None:
        raise RuntimeError("trxviz-cli not found: set $TRXVIZ_CLI, put it on PATH, or call trxscan.viz.install_trxviz()")
    import tempfile

    tmp = None
    if out is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        out = tmp.name
    cmd: list[str] = [str(exe), "render", "--out", str(out), "--width", str(width), "--height", str(height), "--view", view]
    if project is not None:
        cmd += ["--project", str(project)]
    for flag, val in (("--tractogram", tractogram), ("--nifti", nifti), ("--odx", odx)):
        if val is None:
            continue
        for v in ([val] if isinstance(val, (str, Path)) else val):
            cmd += [flag, str(v)]
    for flag, val in (("--azimuth", azimuth), ("--elevation", elevation), ("--distance", distance)):
        if val is not None:
            cmd += [flag, str(val)]
    if target is not None:
        cmd += ["--target", ",".join(str(float(v)) for v in target)]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})})
    if r.returncode != 0:
        raise RuntimeError(f"trxviz-cli failed ({r.returncode}):\n{r.stderr[-2000:]}")
    data = Path(out).read_bytes()
    if tmp is not None:
        os.unlink(tmp.name)
    return data


def render_ipython(**kw: Any):
    """:func:`render` wrapped in an ``IPython.display.Image``."""
    from IPython.display import Image

    return Image(data=render(**kw), format="png")
