"""Two independent implementations must agree, or one of them is wrong."""
import numpy as np
import pytest
from score_crosscheck import compare_scorers, naive_alignment, naive_oscillatory


def _edge(n=64, ring=0.0, phase=0.0, blur=0.0):
    x = np.arange(n)
    img = (x >= n // 2 + 0.5).astype(float)
    if ring:
        img = img + ring * ((-1.0) ** x) * np.exp(-np.abs(x - n // 2) / 6.0)
    if blur:
        k = np.exp(-0.5 * (np.arange(-9, 10) / blur) ** 2)
        k /= k.sum()
        img = np.convolve(img, k, mode="same")
    return np.tile(img, (n, 1)).astype(complex) * np.exp(1j * phase)


@pytest.mark.parametrize("phase", [0.0, 0.7, 2.5])
@pytest.mark.parametrize("frac", [0.0, 0.5, 1.0])
def test_implementations_agree_across_phase_and_removal(phase, frac):
    ref = _edge(phase=phase)
    ctl = _edge(ring=0.09, phase=phase)
    est = ref + frac * (ctl - ref)
    ok, detail = compare_scorers(est, ref, ctl)
    assert ok, detail


def test_implementations_agree_on_a_blurring_method():
    ref, ctl = _edge(), _edge(ring=0.09)
    ok, detail = compare_scorers(_edge(blur=2.0), ref, ctl)
    assert ok, detail


def test_naive_alignment_matches_its_definition_exactly():
    ref, ctl = _edge(), _edge(ring=0.09)
    assert abs(naive_alignment(ctl, ref, ctl) - 1.0) < 1e-12
    assert abs(naive_alignment(ref, ref, ctl)) < 1e-12


def test_naive_oscillatory_is_zero_for_perfect_recovery():
    ref = _edge()
    assert naive_oscillatory(ref, ref) < 1e-12


def _box(n=64, o=4, ring=True):
    """The benchmark's box, built the way `trxscan-benchmark` builds it.

    Edges sit at `(q + 0.5)` and `(3q + 0.5)` ACQUIRED voxels, and the reference is a block MEAN --
    both matching the Rust fixture. An earlier version placed the box on voxel boundaries and
    point-sampled with `obj[::o, ::o]`, which produces a single-sample gradient and therefore could
    not reproduce the partial-volume `0, 0.5, 1` plateau that made one physical edge detect as two.
    That made `test_the_box_has_exactly_two_edges_per_axis` pass vacuously.
    """
    N = n * o
    q = n // 4
    lo, hi = (q + 0.5) * o, (3 * q + 0.5) * o
    ax = np.arange(N)
    cov = np.clip(np.minimum(ax + 1.0, hi) - np.maximum(ax, lo), 0.0, 1.0)
    obj = cov[:, None] * cov[None, :]
    if not ring:
        return obj.reshape(n, o, n, o).mean(axis=(1, 3)).astype(complex)   # block MEAN
    K = np.fft.fftshift(np.fft.fft2(obj))
    c = K[N // 2 - n // 2:N // 2 + n // 2, N // 2 - n // 2:N // 2 + n // 2] / (o * o)
    return np.fft.ifft2(np.fft.ifftshift(c))


def test_symmetric_box_gradients_tie_so_the_axis_must_be_explicit():
    box = np.abs(_box(ring=False))
    gx = np.abs(np.diff(box, axis=0)).mean()
    gy = np.abs(np.diff(box, axis=1)).mean()
    assert abs(gx - gy) < 1e-12, f"expected an exact tie, got {gx} vs {gy}"


@pytest.mark.parametrize("axis", [0, 1])
def test_implementations_agree_on_the_2d_box_per_axis(axis):
    """Both scorers, both axes, on the actual benchmark phantom.

    Without an explicit axis the primary broke the tie to axis 0 and the naive one to axis 1, so
    they measured different physics while every one-dimensional test still passed.
    """
    ref = _box(ring=False)
    ctl = _box(ring=True)
    est = ref + 0.4 * (ctl - ref)
    ok, detail = compare_scorers(est, ref, ctl, axis=axis)
    assert ok, f"axis {axis}: {detail}"


def test_partial_fourier_breaks_the_axis_symmetry_of_the_box():
    """PF acts on ky only, so it must break the box's exact readout/phase-encode symmetry.

    An earlier version of this test asserted `energy_pe > energy_ro`, on the assumption that PF
    "shows up more on phase-encode". That is measurably false here: zero-filled PF trades ringing
    for blur along PE, and the blur sits at the edge, which the sidelobe guard excludes. Measured
    artifact norms are ro 0.0828 / pe 0.0757 -- PF makes the READOUT axis worse on this metric.

    What is defensible, and is what the per-axis machinery exists to detect, is that a symmetric
    box has exactly equal artifact on both axes under full Fourier, and PF destroys that equality.
    """
    from score_unringing import artifact_norm
    n, o = 64, 4
    N = n * o
    q = N // 4
    obj = np.zeros((N, N))
    obj[q:3 * q, q:3 * q] = 1.0
    K = np.fft.fftshift(np.fft.fft2(obj))
    c = K[N // 2 - n // 2:N // 2 + n // 2, N // 2 - n // 2:N // 2 + n // 2] / (o * o)
    ref = obj[::o, ::o].astype(complex)

    full = np.fft.ifft2(np.fft.ifftshift(c))
    ro_f, pe_f = artifact_norm(ref, full, 0), artifact_norm(ref, full, 1)
    assert abs(ro_f - pe_f) < 1e-9, f"symmetric box must be axis-symmetric: {ro_f} vs {pe_f}"

    pf = c.copy()
    pf[:, : int(n * 0.25)] = 0                    # drop low-ky lines: PF along axis 1
    pf_img = np.fft.ifft2(np.fft.ifftshift(pf))
    ro_p, pe_p = artifact_norm(ref, pf_img, 0), artifact_norm(ref, pf_img, 1)
    assert abs(ro_p - pe_p) > 0.05 * ro_f, (
        f"PF must break the symmetry: ro={ro_p:.5f} pe={pe_p:.5f} (full {ro_f:.5f})")


@pytest.mark.parametrize("axis", [0, 1])
def test_energy_and_artifact_norm_agree_on_the_2d_box(axis):
    """The two metrics that drive PF acceptance, independently reimplemented."""
    from score_crosscheck import naive_artifact_norm, naive_energy_ratio
    from score_unringing import artifact_norm, residual_energy_ratio
    ref, ctl = _box(ring=False), _box(ring=True)
    est = ref + 0.4 * (ctl - ref)
    a = residual_energy_ratio(est, ref, ctl, axis=axis)
    b = naive_energy_ratio(est, ref, ctl, axis=axis)
    assert np.isclose(a, b, rtol=0.02), f"energy axis {axis}: {a} vs {b}"
    a = artifact_norm(ref, ctl, axis)
    b = naive_artifact_norm(ref, ctl, axis=axis)
    assert np.isclose(a, b, rtol=0.02), f"artifact_norm axis {axis}: {a} vs {b}"


def test_partial_fourier_box_cross_checks_on_both_axes():
    """The scanner-like contiguous PF case, both scorers, both axes."""
    n, o = 64, 4
    N = n * o
    q = N // 4
    obj = np.zeros((N, N))
    obj[q:3 * q, q:3 * q] = 1.0
    K = np.fft.fftshift(np.fft.fft2(obj))
    c = K[N // 2 - n // 2:N // 2 + n // 2, N // 2 - n // 2:N // 2 + n // 2] / (o * o)
    ref = obj[::o, ::o].astype(complex)
    ctl = np.fft.ifft2(np.fft.ifftshift(c))
    pf = c.copy()
    pf[:, : n - int(round(n * 0.75))] = 0          # contiguous 6/8 along ky
    est = np.fft.ifft2(np.fft.ifftshift(pf))
    for axis in (0, 1):
        ok, detail = compare_scorers(est, ref, ctl, axis=axis)
        assert ok, f"axis {axis}: {detail}"


def test_the_box_has_exactly_two_edges_per_axis():
    """A rectangle has two boundaries per axis. Detecting four means plateaus are being split.

    The benchmark places its edges at half-voxel positions, so a block-averaged profile reads
    0, 0.5, 1 and its gradient reads 0.5, 0.5 -- two adjacent samples that both pass a naive
    local-maximum test. That reported four PE edges for two, and widened the sidelobe guard bands.
    """
    from score_crosscheck import _peak_indices
    from score_unringing import physical_edges
    box = np.abs(_box(ring=False))
    for axis in (0, 1):
        g = np.abs(np.diff(box, axis=axis)).sum(axis=1 - axis)
        assert len(physical_edges(g)) == 2, f"primary, axis {axis}: {physical_edges(g)}"
        assert len(_peak_indices(list(g))) == 2, f"naive, axis {axis}: {_peak_indices(list(g))}"


def test_box_reference_really_has_partial_volume_edges():
    """Guards the fixture itself: without a 0/0.5/1 transition the edge test proves nothing."""
    box = np.abs(_box(ring=False))
    row = box[box.shape[0] // 2]
    assert np.any(np.isclose(row, 0.5, atol=0.02)), (
        f"expected a partial-volume edge voxel, profile has {sorted(set(np.round(row, 3)))[:5]}")


def test_nyquist_score_does_not_cancel_across_an_edge():
    """The round-6 blocker: Gibbs is antisymmetric about an edge.

    Projecting both sides onto one global (-1)**x and summing annihilates it -- measured +0.019 on
    the dark side and -0.019 on the bright side of the same edge. Each side must be projected
    separately and combined in magnitude.
    """
    from score_unringing import _nyquist_amplitude, physical_edges, sidelobe_runs
    ref, ctl = _box(ring=False), _box(ring=True)
    diff = ctl - ref
    dr = np.abs(np.diff(np.abs(ref), axis=-1)).sum(axis=0)

    assert len(physical_edges(dr)) == 2, "two physical edges per axis"
    runs = sidelobe_runs(dr, ref.shape[-1])
    assert len(runs) == 4, f"two edges x two sides = 4 runs, got {len(runs)}"

    row = ref.shape[0] // 2
    per = []
    for idx in runs:
        per.append(abs((diff[row][idx] * ((-1.0) ** idx)).sum() / idx.size))
    assert all(v > 1e-3 for v in per), f"every side must carry artifact: {per}"

    union = np.concatenate(runs)
    cancelled = abs((diff[row][union] * ((-1.0) ** union)).sum() / union.size)
    combined = _nyquist_amplitude(diff, np.abs(ref))
    assert combined > 5 * max(cancelled, 1e-9), (
        f"per-side aggregation {combined:.5f} must survive where the union projection "
        f"cancels to {cancelled:.5f}")


@pytest.mark.parametrize("n", [60, 64])
def test_edge_centroid_rounding_agrees_across_parities(n):
    """Both scorers must map a fractional centroid the same way, at any matrix size.

    Python's round() is ties-to-even, so a centroid of 14.5 became 14 in the naive scorer while
    the primary mapped it to 15. n=64 hid this -- its centroids happen to round the same way --
    so the parity must be tested at both.
    """
    from score_crosscheck import _peak_indices
    from score_unringing import physical_edges

    q = n // 4
    lo, hi = q + 0.5, 3 * q + 0.5
    ax = np.arange(n)
    cov = np.clip(np.minimum(ax + 1.0, hi) - np.maximum(ax, lo), 0.0, 1.0)
    g = np.abs(np.diff(cov))

    centroids = physical_edges(g)
    assert len(centroids) == 2, f"n={n}: {centroids}"
    assert all(abs(c - round(c)) > 0.4 for c in centroids), (
        f"n={n}: fixture must produce HALF-integer centroids to exercise the parity, got {centroids}")

    primary = [int(np.floor(c + 0.5)) for c in centroids]
    naive = _peak_indices(list(g))
    assert primary == naive, f"n={n}: primary {primary} vs naive {naive}"
