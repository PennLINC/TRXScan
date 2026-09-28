import os

import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "phantom: needs the full sub-60501 phantom (TRXSCAN_RUN_PHANTOM=1)")


@pytest.fixture(scope="session")
def gtab6():
    from dipy.core.gradients import gradient_table

    bvals = np.array([0, 1000, 1000, 1000, 2000, 2000], dtype=float)
    bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]], dtype=float)
    bvecs[1:] /= np.linalg.norm(bvecs[1:], axis=1, keepdims=True)
    return gradient_table(bvals, bvecs=bvecs)


@pytest.fixture(scope="session")
def phantom():
    """The full NIBS phantom from $TRXSCAN_DATA (skipped otherwise)."""
    if not os.environ.get("TRXSCAN_RUN_PHANTOM"):
        pytest.skip("set TRXSCAN_RUN_PHANTOM=1 (and TRXSCAN_DATA) to run phantom tests")
    import trxscan as ts

    return ts.Phantom.load("sub-60501")
