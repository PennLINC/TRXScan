"""Unringing method harness for the acceptance suite (spec 5.2).

Every method is run through one interface so the results table compares like with like. The `none`
control is required, not optional: without it the suite cannot distinguish "this method helped"
from "this method changed the image".

A method that is not installed raises `MethodUnavailable`. It must never fall back to another
implementation -- a silent substitution would put one method's numbers in another's row.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

__all__ = ["run_method", "available_methods", "MethodUnavailable", "METHODS"]

METHODS = ("none", "mrdegibbs", "dipy", "rpg")


class MethodUnavailable(RuntimeError):
    """Raised when a requested method is not installed, or is not a known method at all."""


def _has_dipy():
    try:
        from dipy.denoise.gibbs import gibbs_removal  # noqa: F401
        return True
    except Exception:
        return False


def _has_rpg():
    if shutil.which("rpg_degibbs") or shutil.which("rpg"):
        return True
    try:
        import rpg  # noqa: F401
        return True
    except Exception:
        return False


def available_methods():
    """Which methods can actually run here. `none` is always present."""
    out = ["none"]
    if shutil.which("mrdegibbs"):
        out.append("mrdegibbs")
    if _has_dipy():
        out.append("dipy")
    if _has_rpg():
        out.append("rpg")
    return out


def _run_mrdegibbs(mag):
    import nibabel as nib
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.nii"), os.path.join(d, "out.nii")
        # mrdegibbs needs >=3 dims; add a singleton slice axis.
        nib.save(nib.Nifti1Image(mag[:, :, None].astype(np.float32), np.eye(4)), src)
        r = subprocess.run(["mrdegibbs", "-quiet", "-force", src, dst],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"mrdegibbs failed: {r.stderr.strip()[:400]}")
        return np.asarray(nib.load(dst).get_fdata()).squeeze()


def _run_dipy(mag):
    from dipy.denoise.gibbs import gibbs_removal
    try:
        return gibbs_removal(mag, inplace=False, num_processes=1)
    except TypeError:                       # older signatures lack these kwargs
        return gibbs_removal(mag)


def run_method(name, mag, phase):
    """Apply `name` to a magnitude/phase pair, returning the corrected pair.

    Magnitude-only methods return the phase untouched. That is deliberate and is what makes their
    degradation visible on `phase_rmse_masked` when object phase is present.
    """
    mag = np.asarray(mag, float)
    phase = np.asarray(phase, float)
    if name == "none":
        return mag, phase
    if name not in METHODS:
        raise MethodUnavailable(f"unknown method {name!r}; known: {METHODS}")
    if name not in available_methods():
        raise MethodUnavailable(
            f"{name!r} is not installed here (available: {available_methods()}). "
            "Refusing to substitute another method."
        )
    if name == "mrdegibbs":
        return _run_mrdegibbs(mag), phase
    if name == "dipy":
        return _run_dipy(mag), phase
    if name == "rpg":
        raise MethodUnavailable("rpg is detected but no adapter is implemented yet")
    raise MethodUnavailable(name)
