"""Moment calibration, cache validation, and Gaussian-mixture regressions."""

from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import beta, chi2, ncx2, norm

from sonic.kernels import FFTKernel, MatrixKernel, NUFFTKernel
from sonic.statistics import (
    _estimate_kernel_traces,
    _moment_sf,
    _prepare_moment_fit,
    compute_null_params,
    liu_sf,
    spatial_q_test,
)


@pytest.mark.parametrize("backend", ["matrix", "fft2", "rfft2", "nufft"])
@pytest.mark.parametrize("cache", ["default", "method", "prepared"])
def test_null_is_prepared_once_before_chunking_without_mutating_cache(backend, cache):
    from copy import deepcopy

    from sonic import statistics

    rng = np.random.default_rng(22)
    if backend in ("fft2", "rfft2"):
        kernel = FFTKernel((8, 8), method="gaussian", bandwidth=0.7, fft_solver=backend)
        data = rng.normal(size=(8, 8, 7))
    else:
        coords = rng.uniform(0, 8, (40, 2))
        kernel = (
            MatrixKernel.from_coordinates(coords, method="gaussian")
            if backend == "matrix"
            else NUFFTKernel(coords, (8, 8), (1, 1), method="gaussian")
        )
        data = rng.normal(size=(40, 7))
    method = "welch" if backend == "matrix" and cache == "default" else "moments"
    full = compute_null_params(kernel, method=method)
    params = None if cache == "default" else {"method": method}
    if cache == "prepared":
        params = full
    before = deepcopy(params)
    expected = spatial_q_test(data, kernel, null_params=full, chunk_size=-1)
    with (
        patch.object(statistics, "compute_null_params", wraps=compute_null_params) as prepare,
        patch.object(
            statistics, "_fit_standardized_q", wraps=statistics._fit_standardized_q
        ) as fit,
    ):
        actual = spatial_q_test(data, kernel, null_params=params, chunk_size=2)
        assert prepare.call_count == (0 if cache == "prepared" else 1)
        assert fit.call_count == (1 if method == "moments" and cache != "prepared" else 0)
    np.testing.assert_allclose(actual, expected)
    assert params == before

    # Computing Q alone must not collect null moments, including on Fourier backends.
    with patch.object(statistics, "_prepare_q_null", side_effect=AssertionError("unexpected fit")):
        scores = spatial_q_test(data, kernel, return_pval=False, chunk_size=2)
    np.testing.assert_allclose(scores, expected[0])


def test_shared_evaluator_preserves_tail_conventions_and_scale_invariance():
    from sonic.statistics import _q_pvalues

    q = np.array([-2.0, 0.0, 2.0])
    normal = {"family": "normal", "model": "standardized_q", "mu_Q": 0.0, "sigma_Q": 1.0}
    np.testing.assert_allclose(_q_pvalues(q, {"method": "moments", "q_fit": normal}), norm.sf(q))
    np.testing.assert_allclose(
        _q_pvalues(q, {"method": "clt", "mean_Q": 0.0, "var_Q": 1.0}),
        2 * norm.sf(abs(q)),
    )
    tiny = {"method": "clt", "mean_Q": 0.0, "var_Q": 1e-26}
    np.testing.assert_allclose(_q_pvalues(q * 1e-13, tiny), 2 * norm.sf(abs(q)))
    np.testing.assert_array_equal(_q_pvalues(q, {**tiny, "var_Q": 0.0}), np.ones(3))


@pytest.mark.parametrize("m,rank", [(2, 1), (63, 1), (63, 4), (63, 15), (63, 31), (63, 59)])
def test_finite_moments_and_bounded_fallback_against_exact_beta(m, rank):
    # For a rank-r projection, Q/m ~ Beta(r/2, (m-r)/2) exactly.
    lam = np.r_[np.ones(rank), np.zeros(m - rank)]
    coef = _prepare_moment_fit(lam, n=m + 1)
    expected = np.array(beta.stats(rank / 2, (m - rank) / 2, moments="mvsk"))
    expected[:2] *= [m, m**2]
    if coef["family"] == "beta":
        fitted = beta.stats(coef["alpha"], coef["beta"], moments="mvsk")
    else:
        assert coef["dof_x"] > 0 and coef["delta_x"] >= 0
        fitted = ncx2.stats(coef["dof_x"], coef["delta_x"], moments="mvsk")
    np.testing.assert_allclose(
        [coef["mu_Q"], coef["sigma_Q"] ** 2, *fitted[2:]], expected, rtol=1e-11, atol=1e-12
    )
    alpha = np.array([0.05, 0.01, 0.001])
    q = m * beta.isf(alpha, rank / 2, (m - rank) / 2)
    np.testing.assert_allclose(_moment_sf(q, coef), alpha, rtol=0.04, atol=1e-12)


def test_shift_scale_and_omitted_zero_modes():
    m = 63
    lam = np.r_[np.ones(15), np.zeros(m - 15)]
    q = m * beta.isf([0.05, 0.01, 0.001], 7.5, 24)
    expected = _moment_sf(q, _prepare_moment_fit(lam, n=m + 1))
    for scale, shift in [(1e-12, 0), (1e12, 0), (1e-4, 100), (1, 3)]:
        np.testing.assert_allclose(
            _moment_sf(q * scale + m * shift, _prepare_moment_fit(lam * scale + shift, n=m + 1)),
            expected,
            rtol=2e-7,
        )
    np.testing.assert_allclose(_moment_sf(q, _prepare_moment_fit(np.ones(15), n=m + 1)), expected)
    np.testing.assert_allclose(_moment_sf(q, _prepare_moment_fit(np.r_[0, lam], n=m + 1)), expected)
    np.testing.assert_allclose(
        _moment_sf(q, _prepare_moment_fit([1], dofs=[15], n=m + 1)), expected
    )


def test_normal_and_deterministic_limits_and_invalid_inputs():
    lam = np.ones(63)
    np.testing.assert_array_equal(
        _moment_sf([62, 63, 64], _prepare_moment_fit(lam, n=64)), [1, 1, 1]
    )
    # Symmetric spectrum with positive excess kurtosis cannot fit a beta or chi-square.
    lam[:2] = [0, 2]
    coef = _prepare_moment_fit(lam, n=64)
    assert coef["family"] == "normal"
    q = coef["mu_Q"] + np.array([-2, 0, 2]) * coef["sigma_Q"]
    np.testing.assert_allclose(_moment_sf(q, coef), norm.sf([-2, 0, 2]))
    for kwargs in [{"n": 1}, {"n": 2}, {"n": 64, "deltas": np.ones(63)}]:
        with pytest.raises(ValueError):
            _prepare_moment_fit(lam, **kwargs)


@pytest.mark.parametrize("noncentral", [False, True])
def test_unstandardized_mixture_matches_liu_reference_across_scales(noncentral):
    # Liu parameters calculated independently with 60-digit Decimal arithmetic,
    # followed by SciPy's central/noncentral chi-square survival function.
    expected = (
        [0.9427447399105364, 0.6150786517776423, 0.06667771283096502]
        if noncentral
        else [0.7727110822059136, 0.20738656243550757, 0.002438980581802866]
    )
    for scale in (1e-200, 1e-3, 1.0, 1e200):
        actual = liu_sf(
            np.array([1, 5, 20]) * scale,
            np.array([0.3, 1, 2]) * scale,
            deltas=[0.2, 1, 2] if noncentral else None,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13)


@pytest.mark.parametrize("delta", [0.0, 2.0])
def test_liu_single_component_is_exact_across_scales(delta):
    q = np.array([0.5, 5.0, 20.0])
    expected = chi2.sf(q, 3) if delta == 0 else ncx2.sf(q, 3, delta)
    for scale in (1e-200, 1e-3, 1.0, 1e200):
        actual = liu_sf(q * scale, [scale], dofs=[3], deltas=[delta])
        np.testing.assert_allclose(actual, expected, rtol=1e-13)
    np.testing.assert_array_equal(liu_sf([0, 1e-200, 1], [0]), [1, 0, 0])


def test_gaussian_q_calibration_retains_small_eigenvalues():
    rng = np.random.default_rng(11)
    basis, _ = np.linalg.qr(np.column_stack([np.ones(16), rng.normal(size=(16, 15))]))
    matrix = (basis[:, 1:] * np.arange(1, 16)) @ basis[:, 1:].T
    expected = liu_sf([50, 150, 300], np.arange(1, 16))
    for scale in (1.0, 1e-12, 1e-100):
        kernel = MatrixKernel.from_matrix(matrix * scale)
        params = compute_null_params(kernel, method="moments", dirichlet_correction=False)
        np.testing.assert_allclose(
            _moment_sf(np.array([50, 150, 300]) * scale, params["q_fit"]), expected, rtol=1e-12
        )


@pytest.mark.parametrize("backend", ["matrix", "fft2", "rfft2", "nufft"])
def test_gaussian_q_probe_fit_is_scale_invariant(backend):
    rng = np.random.default_rng(15)
    coords = rng.uniform(0, 8, (32, 2))
    results = []
    for scale in (1.0, 1e-12, 1e-100):
        if backend == "matrix":
            kernel = MatrixKernel.from_matrix(np.diag(np.linspace(1, 2, 32)) * scale)
        elif backend == "nufft":
            kernel = NUFFTKernel(coords, (8, 8), (1, 1), method="gaussian")
            kernel._fft_kernel.spectrum *= scale
        else:
            kernel = FFTKernel((8, 8), method="gaussian", fft_solver=backend)
            kernel.spectrum *= scale
        fit = compute_null_params(
            kernel, method="moments", dirichlet_correction=False, n_probes=60
        )["q_fit"]
        if scale == 1.0:
            q = fit["mu_Q"] + np.array([-1, 0, 1, 3]) * fit["sigma_Q"]
        results.append(_moment_sf(q * scale, fit))
    np.testing.assert_allclose(results[1:], [results[0], results[0]], rtol=1e-10)


@pytest.mark.parametrize(
    "method,kwargs",
    [("gaussian", {"bandwidth": 0.4}), ("car", {"rho": 0.9}), ("moran", {"k_neighbors": 4})],
)
def test_gaussian_null_tail_calibration(method, kwargs):
    kernel = FFTKernel((8, 8), method=method, **kwargs)
    lam = kernel.eigenvalues(return_full_layout=True)[1:]
    coef = compute_null_params(kernel, method="moments")["q_fit"]
    rng = np.random.default_rng(2718)
    rejected = 0
    draws = 200_000
    for _ in range(draws // 4000):
        z2 = rng.standard_normal((4000, len(lam))) ** 2
        q = len(lam) * (z2 @ lam) / z2.sum(axis=1)
        rejected += np.count_nonzero(_moment_sf(q, coef) < 0.01)
    assert abs(rejected / draws - 0.01) < 0.001


@pytest.mark.parametrize("backend", ["matrix", "fft2", "rfft2", "nufft"])
def test_moran_moments_are_explicit_upper_tail_with_clt_default(backend):
    rng = np.random.default_rng(41)
    coords = rng.uniform(0, 8, (40, 2))
    if backend == "matrix":
        kernel = MatrixKernel.from_coordinates(coords, method="moran")
    elif backend == "nufft":
        kernel = NUFFTKernel(coords, (8, 8), (1, 1), method="moran")
    else:
        kernel = FFTKernel((8, 8), method="moran", fft_solver=backend)
    data = rng.normal(size=(8, 8, 7) if backend in ("fft2", "rfft2") else (40, 7))

    params = compute_null_params(kernel, method="moments")
    assert params["model"] == "standardized_q"
    assert params["tail"] == "upper"
    assert params["source"] == ("spectrum" if backend in ("fft2", "rfft2") else "probes")
    q, p = spatial_q_test(data, kernel, null_params=params, chunk_size=2)
    np.testing.assert_allclose(
        spatial_q_test(data, kernel, null_params={"method": "moments"}), [q, p]
    )
    np.testing.assert_allclose(p, _moment_sf(q, params["q_fit"]))
    assert np.all(np.isfinite(p) & (p >= 0) & (p <= 1))

    clt = compute_null_params(kernel, method="clt")
    np.testing.assert_allclose(
        spatial_q_test(data, kernel), spatial_q_test(data, kernel, null_params=clt)
    )
    for method in ("welch", "moments"):
        with pytest.raises(ValueError, match="require a PSD kernel"):
            compute_null_params(kernel, method=method, dirichlet_correction=False)
    with pytest.raises(ValueError, match="require a PSD kernel"):
        compute_null_params(kernel, method="welch")


@pytest.mark.parametrize("rank", [4, 31, 59])
def test_signed_moment_fit_matches_beta_and_is_invariant_to_psd_shift(rank):
    # A signed two-level spectrum has an exact affine beta null, including left skew.
    n = 64
    m = n - 1
    rng = np.random.default_rng(15)
    basis, _ = np.linalg.qr(np.column_stack([np.ones(n), rng.normal(size=(n, m))]))
    basis = basis[:, 1:]
    lam = np.r_[np.ones(rank), -np.ones(m - rank)]
    matrix = (basis * lam) @ basis.T
    kernel = MatrixKernel.from_matrix(matrix)
    params = compute_null_params(kernel, method="moments")
    alpha = np.array([0.05, 0.01, 0.001])
    q = m * (2 * beta.isf(alpha, rank / 2, (m - rank) / 2) - 1)
    np.testing.assert_allclose(_moment_sf(q, params["q_fit"]), alpha, rtol=0.04)
    assert params["mean_Q"] == pytest.approx(2 * rank - m)

    # Adding cH shifts Q by mc without changing its shape or upper-tail probabilities.
    shifted = MatrixKernel.from_matrix(matrix + 2 * (np.eye(n) - np.ones((n, n)) / n))
    shifted_params = compute_null_params(shifted, method="moments")
    data = rng.normal(size=(n, 7))
    scores, p = spatial_q_test(data, kernel, null_params=params)
    shifted_scores, shifted_p = spatial_q_test(data, shifted, null_params=shifted_params)
    np.testing.assert_allclose(shifted_scores, scores + 2 * m)
    np.testing.assert_allclose(shifted_p, p, rtol=1e-10)


def test_centered_probes_stay_stable_near_identity_with_two_applications():
    kernel = FFTKernel((16, 16), method="car", rho=0.1)
    lam = kernel.eigenvalues(return_full_layout=True)[1:]
    matrix = kernel.Kx(np.eye(256).reshape(16, 16, 256)).reshape(256, 256)
    m = 255
    errors = []
    for multiplier in (1, 1e-12, 1e-100):
        dense = MatrixKernel.from_matrix(matrix * multiplier)
        truth = lam - lam.mean()
        expected = np.sum(truth**3) / np.sum(truth**2) ** 1.5
        for seed in range(12):
            with patch.object(dense, "Kx", wraps=dense.Kx) as apply:
                traces = _estimate_kernel_traces(dense, rng_seed=seed, centered=True)
                b = traces["powers"]
                # Analytic traces also apply K to the single constant vector.
                assert sum(call.args[0].ndim == 2 for call in apply.call_args_list) == 2
            errors.append(abs(b[3] / b[2] ** 1.5 - expected))
    assert np.median(errors) < 0.003
    for start in (12, 24):
        np.testing.assert_allclose(errors[:12], errors[start : start + 12], atol=1e-12)
    # Constant spectra stay deterministic with both explicit and stored-inverse kernels.
    for mode in ("precomputed", "precomputed_inverse"):
        identity = MatrixKernel(np.eye(m + 1), mode=mode, method="precomputed")
        params = compute_null_params(identity, method="moments", n_probes=60)
        assert params["q_fit"]["sigma_Q"] == 0


@pytest.mark.parametrize("backend", ["fft2", "rfft2", "matrix", "nufft"])
@pytest.mark.parametrize("correction", [False, True])
def test_prepared_caches_are_complete_and_partial_caches_are_rejected(backend, correction):
    rng = np.random.default_rng(81)
    if backend in ("fft2", "rfft2"):
        kernel = FFTKernel((8, 8), method="gaussian", bandwidth=0.7, fft_solver=backend)
        data = rng.normal(size=(8, 8, 7))
    elif backend == "matrix":
        kernel = MatrixKernel.from_coordinates(rng.uniform(0, 8, (40, 2)), method="gaussian")
        data = rng.normal(size=(40, 7))
    else:
        kernel = NUFFTKernel(rng.uniform(0, 8, (40, 2)), (8, 8), (1, 1), method="gaussian")
        kernel._TOEPLITZ_R_THRESHOLD = 0  # Exercise the probe fallback.
        data = rng.normal(size=(40, 7))
    params = compute_null_params(kernel, method="moments", dirichlet_correction=correction)
    expected = spatial_q_test(data, kernel, null_params=params)
    assert set(params) == {"method", "model", "tail", "source", "q_fit", "mean_Q", "var_Q", "var_R"}
    np.testing.assert_allclose(params["var_R"], kernel.square_trace(), rtol=1e-12)
    for missing_fit in [
        {key: value for key, value in params.items() if key != "q_fit"},
        {**params, "q_fit": None},
    ]:
        with pytest.raises(ValueError, match="q_fit"):
            spatial_q_test(data, kernel, null_params=missing_fit)
    if correction:
        np.testing.assert_allclose(
            spatial_q_test(data, kernel, null_params={"method": "moments"}), expected
        )
    if backend == "matrix":
        from sonic.statistics import _q_pvalues

        np.testing.assert_allclose(_q_pvalues(expected[0], params), expected[1])


class _PrecisionKernel(MatrixKernel):
    """Exercise the real precision solves without the production size threshold."""

    def _build_kernel(self):
        self.stores_precision = True
        return self._data


@pytest.mark.parametrize("method", ["welch", "clt", "moments"])
def test_precision_calibration_shares_solves_and_honors_probe_budget(method):
    from unittest.mock import Mock

    from scipy.sparse import diags
    from scipy.sparse.linalg import splu

    matrix = diags([np.linspace(1, 2, 32)], [0], format="csc")
    kernel = _PrecisionKernel.from_matrix(matrix, is_precision=True)
    kernel._lu = Mock(wraps=splu(matrix))
    with (
        patch.object(kernel, "trace", side_effect=AssertionError("separate lower probes")),
        patch.object(kernel, "square_trace", side_effect=AssertionError("separate R probes")),
        patch("scipy.sparse.linalg.eigsh", side_effect=AssertionError("unexpected eigensolve")),
    ):
        params = compute_null_params(kernel, method=method, n_probes=7)
    widths = [call.args[0].shape[1] for call in kernel._lu.solve.call_args_list]
    assert widths == ([1, 7, 7] if method == "moments" else [1, 7])
    assert kernel._trace_rvs_cache["n_vectors"] == 7
    assert params["var_R"] > 0
    # A later moment fit reuses the first seven solves without altering them.
    probes = kernel._trace_rvs_cache["rvs"].copy()
    solutions = kernel._trace_rvs_cache["Y"].copy()
    kernel._lu.solve.reset_mock()
    compute_null_params(kernel, method="moments", n_probes=7)
    assert [c.args[0].shape[1] for c in kernel._lu.solve.call_args_list] == [1, 7]
    np.testing.assert_array_equal(kernel._trace_rvs_cache["rvs"], probes)
    np.testing.assert_array_equal(kernel._trace_rvs_cache["Y"], solutions)


@pytest.mark.parametrize("method", ["welch", "clt", "moments"])
@pytest.mark.parametrize("correction", [True, False])
@pytest.mark.parametrize("centering", [True, False])
def test_shared_precision_moments_and_r_variance_match_exact_traces(method, correction, centering):
    from scipy.linalg import hadamard

    rng = np.random.default_rng(741)
    n = 16
    x = rng.normal(size=(n, n))
    matrix = x @ x.T / n + np.eye(n)
    precision = _PrecisionKernel.from_matrix(
        np.linalg.inv(matrix), is_precision=True, centering=centering
    )
    probes = hadamard(n).astype(float)
    precision._trace_rvs_cache = {"n_vectors": n, "rvs": probes, "Y": matrix @ probes}
    explicit = MatrixKernel.from_matrix(matrix, centering=centering)
    # Orthogonal probes eliminate Monte Carlo error; the raw diagnostic view
    # must not affect calibration of either standardized Q or R.
    kwargs = {"method": method, "dirichlet_correction": correction}
    if not centering and method == "moments":
        # Exact HKH eigenvalues give the reference even for the raw trace view.
        reference = compute_null_params(MatrixKernel.from_matrix(matrix), **kwargs)
    else:
        reference = compute_null_params(explicit, **kwargs)
    actual = compute_null_params(precision, n_probes=n, **kwargs)
    for key in ("mean_Q", "var_Q", "var_R"):
        np.testing.assert_allclose(actual[key], reference[key], rtol=2e-12, atol=1e-12)
    if method == "moments":
        q = actual["mean_Q"] + np.sqrt(actual["var_Q"]) * np.array([-1, 0, 1, 3])
        np.testing.assert_allclose(
            _moment_sf(q, actual["q_fit"]), _moment_sf(q, reference["q_fit"]), rtol=1e-10
        )
    assert precision.centering is centering


@pytest.mark.parametrize("precision", [False, True])
@pytest.mark.parametrize("centered", [False, True])
def test_probe_blocking_preserves_estimates_and_bounds_solve_width(
    monkeypatch, precision, centered
):
    from sonic import statistics

    n, count = 32, 19
    rng = np.random.default_rng(113)
    x = rng.normal(size=(n, n))
    matrix = x @ x.T / n + np.eye(n)
    cls = _PrecisionKernel if precision else MatrixKernel
    stored = np.linalg.inv(matrix) if precision else matrix
    full_kernel = cls.from_matrix(stored, is_precision=precision)
    blocked_kernel = cls.from_matrix(stored, is_precision=precision)
    full = _estimate_kernel_traces(full_kernel, n_probes=count, centered=centered)
    monkeypatch.setattr(statistics, "_TRACE_PROBE_BUDGET_BYTES", 8 * n * 8 * 3)
    with patch.object(
        blocked_kernel, "_apply_K_dense", wraps=blocked_kernel._apply_K_dense
    ) as apply:
        blocked = _estimate_kernel_traces(blocked_kernel, n_probes=count, centered=centered)
    widths = [call.args[0].shape[1] for call in apply.call_args_list]
    assert max(widths) <= 3
    if precision:
        assert sum(widths) == 1 + count * (3 if centered else 2)
        assert not hasattr(blocked_kernel, "_trace_rvs_cache")
    for key in ("mean", "var_R"):
        np.testing.assert_allclose(blocked[key], full[key], rtol=1e-12)
    for order in (2, 3, 4):
        np.testing.assert_allclose(
            blocked["powers"][order] * blocked["scale"] ** order,
            full["powers"][order] * full["scale"] ** order,
            rtol=1e-11,
        )


def test_matrix_null_avoids_unbounded_spectra_but_reuses_cached_spectrum(monkeypatch):
    from scipy.sparse import diags

    from sonic import statistics

    sparse = MatrixKernel.from_matrix(diags([np.linspace(1, 2, 32)], [0], format="csr"))
    with patch("scipy.sparse.linalg.eigsh", side_effect=AssertionError("near-full eigensolve")):
        with pytest.raises(NotImplementedError, match="Full sparse/implicit spectrum"):
            sparse.eigenvalues()
        assert compute_null_params(sparse, method="moments")["source"] == "probes"
    # Exercise the size policy on a small matrix, without allocating a large test fixture.
    dense = MatrixKernel.from_matrix(sparse._K.toarray())
    monkeypatch.setattr(statistics, "_DENSE_NULL_SPECTRUM_LIMIT", 16)
    with patch.object(dense, "eigenvalues", side_effect=AssertionError("large eigensolve")):
        assert compute_null_params(dense, method="moments")["source"] == "probes"
    dense.eigenvalues()
    with patch("numpy.linalg.eigvalsh", side_effect=AssertionError("repeated eigensolve")):
        assert compute_null_params(dense, method="moments")["source"] == "spectrum"


@pytest.mark.parametrize("family", ["gaussian", "matern", "car", "graph_laplacian", "moran"])
@pytest.mark.parametrize("backend", ["matrix", "fft", "nufft"])
def test_direct_null_defaults_match_q_test_defaults(family, backend):
    from sonic.statistics import _prepare_q_null

    coords = np.indices((4, 4)).reshape(2, -1).T.astype(float)
    if backend == "matrix":
        kernel = MatrixKernel.from_coordinates(coords, method=family)
    elif backend == "fft":
        kernel = FFTKernel((4, 4), method=family)
    else:
        kernel = NUFFTKernel(coords, (4, 4), (1, 1), method=family)
    direct = compute_null_params(kernel)
    automatic = _prepare_q_null(kernel)
    assert direct == automatic
    assert direct["method"] == (
        "clt" if family == "moran" else "welch" if backend == "matrix" else "moments"
    )
    assert direct["tail"] == ("two-sided" if family == "moran" else "upper")
    assert direct["model"] == "standardized_q"


def test_precomputed_matrix_default_assumes_psd_unless_labeled_moran():
    matrix = np.diag(np.arange(1, 5, dtype=float))
    assert compute_null_params(MatrixKernel.from_matrix(matrix))["method"] == "welch"
    assert compute_null_params(MatrixKernel.from_matrix(matrix, method="moran"))["method"] == "clt"


@pytest.mark.parametrize(
    "field", ["model", "family", "mu_Q", "sigma_Q", "mu_x", "sigma_x", "dof_x", "delta_x"]
)
def test_incomplete_nested_moment_fit_is_rejected(field):
    kernel = FFTKernel((8, 8), method="car", rho=0.9)
    params = compute_null_params(kernel)
    broken = {
        **params,
        "q_fit": {key: value for key, value in params["q_fit"].items() if key != field},
    }
    with pytest.raises(ValueError, match="q_fit"):
        spatial_q_test(np.ones((8, 8)), kernel, null_params=broken)


@pytest.mark.parametrize("changed", [{"tail": "two-sided"}, {"model": "gaussian_quadratic"}])
def test_contradictory_moment_cache_metadata_is_rejected(changed):
    kernel = FFTKernel((8, 8), method="car", rho=0.9)
    params = compute_null_params(kernel)
    with pytest.raises(ValueError, match="tail|model"):
        spatial_q_test(np.ones((8, 8)), kernel, null_params={**params, **changed})


def test_signed_kernel_checks_the_fitted_model_even_without_top_level_metadata():
    kernel = FFTKernel((8, 8), method="car", rho=0.9)
    params = compute_null_params(kernel, dirichlet_correction=False)
    del params["model"]
    signed = FFTKernel((8, 8), method="moran")
    with pytest.raises(ValueError, match="require a PSD"):
        spatial_q_test(np.ones((8, 8)), signed, null_params=params)


@pytest.mark.parametrize("backend", ["matrix", "sparse", "fft2", "rfft2", "nufft"])
@pytest.mark.parametrize("method", ["clt", "moments"])
@pytest.mark.parametrize("standardized", [False, True])
def test_constant_features_are_uninformative_with_signed_nulls(backend, method, standardized):
    from scipy.sparse import csc_matrix

    rng = np.random.default_rng(95)
    values = np.column_stack([np.zeros(64) if standardized else np.ones(64), rng.normal(size=64)])
    if standardized:
        values[:, 1] = (values[:, 1] - values[:, 1].mean()) / values[:, 1].std(ddof=1)
    if backend in ("matrix", "sparse"):
        # Q=0 lies in the significant tail of this null, but a constant
        # feature cannot be standardized and must still receive p=1.
        kernel = MatrixKernel.from_matrix(np.diag(np.r_[10.0, -np.ones(63)]), method="moran")
        data = csc_matrix(values) if backend == "sparse" else values
    elif backend in ("fft2", "rfft2"):
        kernel = FFTKernel((8, 8), method="moran", fft_solver=backend)
        data = values.reshape(8, 8, 2)
    else:
        kernel = NUFFTKernel(rng.uniform(0, 8, (64, 2)), (8, 8), (1, 1), method="moran")
        data = values
    params = compute_null_params(kernel, method=method)
    q, p = spatial_q_test(data, kernel, null_params=params, is_standardized=standardized)
    assert q[0] == 0 and p[0] == 1
    _, reference = spatial_q_test(
        data[..., 1:] if backend != "sparse" else data[:, 1:],
        kernel,
        null_params=params,
        is_standardized=standardized,
    )
    np.testing.assert_allclose(p[1], np.asarray(reference).item())


@pytest.mark.parametrize("method", ["welch", "clt", "moments"])
@pytest.mark.parametrize("budget", [0, -1, 1.5, np.nan, np.inf, True])
def test_invalid_probe_budgets_fail_before_kernel_work(method, budget):
    kernel = MatrixKernel.from_matrix(np.eye(4))
    with patch.object(kernel, "trace", side_effect=AssertionError("unexpected trace")):
        with pytest.raises(ValueError, match="positive integer"):
            compute_null_params(kernel, method=method, n_probes=budget)
