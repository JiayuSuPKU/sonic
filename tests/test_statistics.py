"""
Unit tests for statistical functions.
"""

import unittest

import anndata as ad
import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix, diags

from sonic import Detector
from sonic.kernels import FFTKernel, MatrixKernel, NUFFTKernel
from sonic.statistics import (
    _moment_sf,
    _prepare_moment_fit,
    apply_bh_correction,
    cauchy_combine,
    compute_null_params,
    liu_sf,
    spatial_q_test,
    spatial_r_test,
)


@pytest.mark.parametrize("backend", ["matrix", "fft", "nufft"])
@pytest.mark.parametrize("test", ["q", "r"])
def test_memory_budget_controls_auto_chunks_without_changing_results(monkeypatch, backend, test):
    import sonic.statistics as statistics

    rng = np.random.default_rng(13)
    coords = rng.uniform(0, 8, (20, 2))
    if backend == "fft":
        kernel = FFTKernel((8, 8))
        shape, per_feature = (8, 8, 5), 64 * 64
    elif backend == "nufft":
        kernel = NUFFTKernel(coords, (8, 8), (1.0, 1.0))
        shape, per_feature = (20, 5), 64 * (64 + 20)
    else:
        kernel = MatrixKernel.from_coordinates(coords, method="gaussian")
        shape, per_feature = (20, 5), 32 * 20
    x, y = rng.normal(size=shape), rng.normal(size=shape)
    fn, args = (spatial_q_test, (x, kernel)) if test == "q" else (spatial_r_test, (x, y, kernel))
    null = compute_null_params(kernel, method="welch")
    # Explicit chunks retain precedence, even over a tiny positive budget.
    expected = fn(*args, null_params=null, chunk_size=-1, memory_budget_bytes=1)
    seen = []
    original = statistics._chunk_last_axis

    def record(array, start, end):
        seen.append(end - start)
        return original(array, start, end)

    monkeypatch.setattr(statistics, "_chunk_last_axis", record)
    actual = fn(*args, null_params=null, memory_budget_bytes=np.int64(2 * per_feature))
    assert seen == ([2, 2, 1] if test == "q" else [2, 2, 2, 2, 1, 1])
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
    with pytest.raises(ValueError, match="cannot fit one feature"):
        fn(*args, null_params=null, memory_budget_bytes=per_feature - 1)


@pytest.mark.parametrize("budget", [0, -1, 1.5, "auto", "0 GB", "1.1 B", "-2 GiB", None, True])
def test_invalid_memory_budget_rejected_even_for_explicit_chunks(budget):
    kernel = MatrixKernel.from_matrix(np.eye(4))
    x = np.arange(4.0)
    for fn, args in [(spatial_q_test, (x, kernel)), (spatial_r_test, (x, x, kernel))]:
        with pytest.raises(ValueError, match="memory_budget_bytes must be positive"):
            fn(*args, chunk_size=-1, memory_budget_bytes=budget)


@pytest.mark.parametrize("backend", ["fft2", "rfft2", "nufft"])
def test_null_routing_on_signed_spectra_even_with_a_psd_name(backend):
    shape = (8, 8)
    rng = np.random.default_rng(0)
    if backend == "nufft":
        kernel = NUFFTKernel(
            rng.uniform(0, 8, (40, 2)),
            grid_shape=shape,
            spacing=(1.0, 1.0),
            method="gaussian",
        )
        fourier_kernel = kernel._fft_kernel
        data = rng.normal(size=kernel.n)
    else:
        kernel = FFTKernel(shape, method="gaussian", fft_solver=backend)
        fourier_kernel = kernel
        data = rng.normal(size=shape)
    cached = {
        "welch": compute_null_params(kernel, method="welch"),
        "moments": compute_null_params(kernel, method="moments", dirichlet_correction=False),
    }
    # Model a legacy/custom spectrum: the family name alone does not certify PSD.
    fourier_kernel.spectrum = FFTKernel(
        shape, method="moran", fft_solver=fourier_kernel.fft_solver
    ).spectrum.copy()
    explicit = compute_null_params(kernel, method="clt")
    np.testing.assert_allclose(
        spatial_q_test(data, kernel), spatial_q_test(data, kernel, null_params=explicit)
    )
    for method in ("welch", "moments"):
        with pytest.raises(ValueError, match="require a PSD kernel"):
            compute_null_params(kernel, method=method, dirichlet_correction=False)
        with pytest.raises(ValueError, match="require a PSD kernel"):
            spatial_q_test(data, kernel, null_params=cached[method])
    params = compute_null_params(kernel, method="moments", n_probes=4)
    assert params["model"] == "standardized_q"
    assert np.isfinite(spatial_q_test(data, kernel, null_params=params)[1])


class TestMultipleTestingHelpers(unittest.TestCase):
    """Public p-value combination and adjustment helpers."""

    def test_helpers_are_exported_from_statistics(self):
        import sonic.statistics as statistics

        assert "apply_bh_correction" in statistics.__all__
        assert "cauchy_combine" in statistics.__all__

    def test_apply_bh_correction_adjusts_raw_pvalues(self):
        pvals = np.array([0.01, 0.04, 0.03, 0.002, 0.8])
        expected = np.array([0.025, 0.05, 0.05, 0.01, 0.8])

        adjusted = apply_bh_correction(pvals)

        np.testing.assert_allclose(adjusted, expected)

    def test_apply_bh_correction_preserves_shape_and_nonfinite_entries(self):
        pvals = np.array([[0.01, np.nan], [0.04, np.inf]])

        adjusted = apply_bh_correction(pvals)

        assert adjusted.shape == pvals.shape
        np.testing.assert_allclose(adjusted[0, 0], 0.02)
        np.testing.assert_allclose(adjusted[1, 0], 0.04)
        assert np.isnan(adjusted[0, 1])
        assert np.isnan(adjusted[1, 1])

    def test_apply_bh_correction_rejects_invalid_raw_pvalues(self):
        with self.assertRaises(ValueError):
            apply_bh_correction([0.1, -0.1, 0.2])

    def test_cauchy_combine_is_stable_for_tiny_pvalues(self):
        moderate = np.array([[0.01, 0.2, 0.5], [0.4, 0.8, 0.9]])
        expected = 0.5 - np.arctan(np.mean(np.tan(np.pi * (0.5 - moderate)), axis=1)) / np.pi

        np.testing.assert_allclose(cauchy_combine(moderate, axis=1), expected, rtol=1e-12)

        combined = cauchy_combine(np.array([[1e-300, 1e-250, 0.5]]), axis=1)
        assert 0.0 < combined[0] < np.finfo(float).eps


class TestStatisticalFunctions(unittest.TestCase):
    """Test cases for statistical functions."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Create a small grid
        self.n = 25
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 4, 5)
        xx, yy = np.meshgrid(x, y)
        self.coords = np.column_stack((xx.ravel(), yy.ravel()))

        # Create test data
        self.data = np.random.randn(self.n)

        # Create a spatial kernel
        self.kernel = MatrixKernel.from_coordinates(self.coords, method="car")

    def test_spatial_q_test_welch(self):
        """Test spatial Q-test with Welch approximation."""
        Q, pval = spatial_q_test(self.data, self.kernel, null_params={"method": "welch"})

        # Q should be a positive number
        self.assertIsInstance(Q, (float, np.floating))
        self.assertGreater(Q, 0)

        # P-value should be between 0 and 1
        self.assertIsInstance(pval, (float, np.floating))
        self.assertGreaterEqual(pval, 0)
        self.assertLessEqual(pval, 1)

    def test_spatial_q_test_moments(self):
        """Test spatial Q-test with Liu approximation."""
        Q, pval = spatial_q_test(self.data, self.kernel, null_params={"method": "moments"})

        # Q should be a positive number
        self.assertIsInstance(Q, (float, np.floating))
        self.assertGreater(Q, 0)

        # P-value should be between 0 and 1
        self.assertIsInstance(pval, (float, np.floating))
        self.assertGreaterEqual(pval, 0)
        self.assertLessEqual(pval, 1)

    def test_liu_sf(self):
        """Test Liu survival function approximation."""
        # Simple case with uniform eigenvalues
        lambs = np.ones(10)
        t = 5.0

        pval = liu_sf(t, lambs)

        # P-value should be between 0 and 1
        self.assertIsInstance(pval, (float, np.floating))
        self.assertGreaterEqual(pval, 0)
        self.assertLessEqual(pval, 1)

    def test_liu_sf_kurtosis_path(self):
        """Test Liu approximation with kurtosis-based branch."""
        lambs = np.ones(10)
        t = 5.0
        pval = liu_sf(t, lambs, kurtosis=True)
        self.assertIsInstance(pval, (float, np.floating))
        self.assertGreaterEqual(pval, 0)
        self.assertLessEqual(pval, 1)

    def test_zero_variance_data(self):
        """Test handling of zero variance data."""
        constant_data = np.ones(self.n)
        Q, pval = spatial_q_test(constant_data, self.kernel)

        # Should handle gracefully
        self.assertEqual(Q, 0.0)
        self.assertEqual(pval, 1.0)

    def test_spatial_q_test_sparse_csr(self):
        """Test spatial_q_test with sparse CSR matrix input."""
        # Create sparse data (CSR format)
        X_dense = np.random.randn(25, 10)
        X_sparse = csr_matrix(X_dense)

        # Compute with dense and sparse
        Q_dense, pval_dense = spatial_q_test(X_dense, self.kernel, return_pval=True)
        Q_sparse, pval_sparse = spatial_q_test(X_sparse, self.kernel, return_pval=True)

        # Results should be identical
        np.testing.assert_allclose(Q_sparse, Q_dense, rtol=1e-10)
        np.testing.assert_allclose(pval_sparse, pval_dense, rtol=1e-10)

    def test_spatial_q_test_sparse_csc(self):
        """Test spatial_q_test with sparse CSC matrix input."""
        # Create sparse data (CSC format)
        X_dense = np.random.randn(25, 10)
        X_sparse = csc_matrix(X_dense)

        # Compute with dense and sparse
        Q_dense, pval_dense = spatial_q_test(X_dense, self.kernel, return_pval=True)
        Q_sparse, pval_sparse = spatial_q_test(X_sparse, self.kernel, return_pval=True)

        # Results should be identical
        np.testing.assert_allclose(Q_sparse, Q_dense, rtol=1e-10)
        np.testing.assert_allclose(pval_sparse, pval_dense, rtol=1e-10)

    def test_spatial_q_test_chunking(self):
        """Test spatial_q_test with chunking for large feature sets."""
        # Create data with many features
        X = np.random.randn(25, 50)

        # Compute without chunking
        Q_full, pval_full = spatial_q_test(X, self.kernel, return_pval=True)

        # Compute with chunking (chunk_size=10)
        Q_chunked, pval_chunked = spatial_q_test(X, self.kernel, chunk_size=10, return_pval=True)

        # Results should be identical
        np.testing.assert_allclose(Q_chunked, Q_full, rtol=1e-10)
        np.testing.assert_allclose(pval_chunked, pval_full, rtol=1e-10)

    def test_spatial_q_test_sparse_with_chunking(self):
        """Test spatial_q_test with sparse input and chunking."""
        # Create sparse data with many features
        X_dense = np.random.randn(25, 50)
        X_sparse = csr_matrix(X_dense)

        # Compute dense without chunking
        Q_dense, pval_dense = spatial_q_test(X_dense, self.kernel, return_pval=True)

        # Compute sparse with chunking
        Q_sparse, pval_sparse = spatial_q_test(
            X_sparse, self.kernel, chunk_size=15, return_pval=True
        )

        # Results should be identical
        np.testing.assert_allclose(Q_sparse, Q_dense, rtol=1e-10)
        np.testing.assert_allclose(pval_sparse, pval_dense, rtol=1e-10)

    def test_spatial_q_test_single_feature_sparse(self):
        """Test spatial_q_test with single feature sparse matrix."""
        # Create single-feature sparse data
        X_dense = np.random.randn(25, 1)
        X_sparse = csr_matrix(X_dense)

        # Should handle single feature correctly
        Q_dense, pval_dense = spatial_q_test(X_dense, self.kernel, return_pval=True)
        Q_sparse, pval_sparse = spatial_q_test(X_sparse, self.kernel, return_pval=True)

        # Results should be identical
        np.testing.assert_allclose(Q_sparse, Q_dense, rtol=1e-10)
        np.testing.assert_allclose(pval_sparse, pval_dense, rtol=1e-10)

    def test_spatial_q_test_progress_bar(self):
        """Test spatial_q_test with progress bar enabled (manual inspection)."""
        # Create data that will require multiple chunks
        X = np.random.randn(25, 30)

        # Should not raise error when show_progress=True
        Q, pval = spatial_q_test(
            X, self.kernel, chunk_size=10, show_progress=False, return_pval=True
        )

        # Verify we got results
        self.assertEqual(len(Q), 30)
        self.assertEqual(len(pval), 30)

    def test_compute_null_params_clt(self):
        """Test compute_null_params with CLT approximation."""
        params = compute_null_params(self.kernel, method="clt")
        self.assertEqual(params["method"], "clt")
        self.assertIn("mean_Q", params)
        self.assertIn("var_Q", params)

    def test_compute_null_params_moments(self):
        """The cache contains the fitted distribution, without intermediate traces."""
        params = compute_null_params(self.kernel, method="moments", k_eigen=5)
        self.assertEqual(params["method"], "moments")
        self.assertEqual(
            set(params), {"method", "model", "tail", "source", "q_fit", "mean_Q", "var_Q", "var_R"}
        )
        self.assertIn(params["q_fit"]["family"], {"ncx2", "beta", "normal"})
        self.assertEqual(params["q_fit"]["mu_Q"], params["mean_Q"])
        self.assertEqual(params["q_fit"]["sigma_Q"] ** 2, params["var_Q"])

    def test_spatial_q_test_kernel_matrix_requires_params(self):
        """Kernel matrices without params should raise when null_params is None."""
        K = self.kernel.realization()
        with self.assertRaises(ValueError):
            spatial_q_test(self.data, K, null_params=None)

    def test_spatial_r_test_basic(self):
        """Test spatial R-test on two vectors."""
        x = np.random.randn(self.n)
        y = np.random.randn(self.n)
        R, pval = spatial_r_test(x, y, self.kernel, return_pval=True)
        self.assertIsInstance(R, (float, np.floating))
        self.assertIsInstance(pval, (float, np.floating))
        self.assertGreaterEqual(pval, 0)
        self.assertLessEqual(pval, 1)

    def test_spatial_r_test_zero_variance(self):
        """Zero-variance inputs should return neutral p-values."""
        x = np.ones(self.n)
        y = np.random.randn(self.n)
        R, pval = spatial_r_test(x, y, self.kernel, return_pval=True)
        self.assertAlmostEqual(R, 0.0, places=8)
        self.assertAlmostEqual(pval, 1.0, places=8)


class TestKernelPrimitivesAndNullParams(unittest.TestCase):
    """Cross-cutting checks: shared signature shape, ``var_R`` hand-off, and
    equivalence of ``spatial_r_test``'s public ``Kx`` path with the legacy
    ``kernel._K``-based computation."""

    def setUp(self):
        np.random.seed(0)
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 4, 5)
        xx, yy = np.meshgrid(x, y)
        self.coords = np.column_stack((xx.ravel(), yy.ravel()))
        # Use raw (centering=False) so ``kernel.Kx(z) == K @ z`` algebraically
        # matches what the K·z primitive tests expect. Centered behavior is
        # covered by the Q-test FPR / power tests in test_kernels.py.
        self.kernel = MatrixKernel.from_coordinates(self.coords, method="matern", centering=False)
        self.n = self.coords.shape[0]

    def test_compute_null_params_populates_var_R(self):
        """compute_null_params should always populate var_R alongside Q-test moments."""
        for method in ("clt", "welch", "moments"):
            params = compute_null_params(self.kernel, method=method)
            self.assertIn("var_R", params)
            self.assertGreater(params["var_R"], 0.0)

    def test_spatial_r_test_consumes_var_R(self):
        """Supplying var_R via null_params should match the on-the-fly path exactly."""
        x = np.random.randn(self.n)
        y = np.random.randn(self.n)
        R_auto, p_auto = spatial_r_test(x, y, self.kernel)
        params = compute_null_params(self.kernel, method="welch")
        R_given, p_given = spatial_r_test(x, y, self.kernel, null_params=params)
        self.assertAlmostEqual(R_auto, R_given, places=10)
        self.assertAlmostEqual(p_auto, p_given, places=10)

    def test_kernel_Kx_matches_dense_matmul(self):
        """Kernel.Kx(z) must equal K @ z for explicit kernels."""
        K = self.kernel.realization()
        z = np.random.randn(self.n, 3)
        np.testing.assert_allclose(self.kernel.Kx(z), K @ z, rtol=1e-10, atol=1e-12)

    def test_kernel_xtKy_matches_paired_diagonal(self):
        """Kernel.xtKy(x, y) must equal the paired diagonal of x^T K y."""
        K = self.kernel.realization()
        X = np.random.randn(self.n, 4)
        Y = np.random.randn(self.n, 4)
        expected = np.einsum("ij,ik,kj->j", X, K, Y)
        np.testing.assert_allclose(self.kernel.xtKy(X, Y), expected, rtol=1e-10, atol=1e-12)

    def test_qtest_sparse_input_matches_dense(self):
        """Phase E: spatial_q_test must produce the same Q on CSR and ndarray inputs.

        The sparse path in `spatial_q_test` densifies **per chunk** (not the full
        slab), so the statistic should be bit-identical up to floating-point noise.
        """
        rng = np.random.default_rng(0)
        X_dense = rng.standard_normal((self.n, 50))
        # Sprinkle zeros to make sparsity meaningful.
        X_dense[X_dense < -0.5] = 0.0
        X_sparse = csr_matrix(X_dense)
        Q_dense = spatial_q_test(X_dense, self.kernel, return_pval=False)
        Q_sparse = spatial_q_test(X_sparse, self.kernel, return_pval=False)
        np.testing.assert_allclose(Q_dense, Q_sparse, rtol=1e-10, atol=1e-12)

    def test_unified_tests_share_signature(self):
        """Public Q/R dispatchers accept the canonical kwargs."""
        import inspect

        from sonic import spatial_q_test, spatial_r_test

        canonical_q = {"null_params", "return_pval", "is_standardized"}
        canonical_r = {"null_params", "return_pval", "is_standardized"}
        for fn in (spatial_q_test,):
            sig = set(inspect.signature(fn).parameters)
            self.assertTrue(canonical_q.issubset(sig), f"{fn.__name__} missing {canonical_q - sig}")
        for fn in (spatial_r_test,):
            sig = set(inspect.signature(fn).parameters)
            self.assertTrue(canonical_r.issubset(sig), f"{fn.__name__} missing {canonical_r - sig}")


def test_identity_kernel_q_is_uninformative():
    rng = np.random.default_rng(0)
    data = ad.AnnData(csc_matrix(rng.normal(size=(80, 2))))
    data.var_names = ["x", "y"]
    data.obsm["spatial"] = np.column_stack([np.arange(80), np.zeros(80)])
    detector = Detector(data, kernel_method="gaussian", bandwidth=0.01).setup_data(data)
    params = compute_null_params(detector.kernel_, method="welch")
    assert params["var_Q"] == 0
    q, p = spatial_q_test(data.X, detector.kernel_, null_params=params)
    np.testing.assert_allclose(q, 79)
    np.testing.assert_array_equal(p, [1, 1])
    result = detector.compute_qstat(n_jobs=1, show_progress=False)
    np.testing.assert_array_equal(result.P_value, [1, 1])
    np.testing.assert_array_equal(_moment_sf(q, _prepare_moment_fit(np.ones(79), n=80)), [1, 1])


@pytest.mark.parametrize("backend", ["fft2", "rfft2", "nufft"])
def test_cached_degenerate_welch_null_on_fourier_backends(backend):
    shape = (8, 8)
    data = np.random.default_rng(0).normal(size=shape)
    if backend == "nufft":
        coords = np.column_stack([a.ravel() for a in np.indices(shape)])
        kernel = NUFFTKernel(coords, grid_shape=shape, spacing=(1.0, 1.0))
        data = data.ravel()
    else:
        kernel = FFTKernel(shape, fft_solver=backend, bandwidth=0.01)
    # A cached deterministic null must take the same guard as uncached moments.
    params = {"method": "welch", "mean_Q": 63.0, "var_Q": 0.0, "scale_g": 0.0, "df_h": 1.0}
    _, p = spatial_q_test(data, kernel, null_params=params)
    assert p == 1.0


@pytest.mark.parametrize("dtype,scale", [(np.int16, 100), (np.int32, 100_000), (np.int64, 10**10)])
@pytest.mark.parametrize("sparse_type", [csr_matrix, csc_matrix])
def test_sparse_integer_q_matches_float(dtype, scale, sparse_type):
    values = np.column_stack([np.arange(8), np.arange(8)[::-1]]) * scale
    counts = sparse_type(values.astype(dtype))
    before = counts.copy()
    coords = np.column_stack([np.arange(8), np.zeros(8)])
    kernel = MatrixKernel.from_coordinates(coords, method="gaussian")
    expected = spatial_q_test(values.astype(float), kernel)
    np.testing.assert_allclose(spatial_q_test(counts, kernel, chunk_size=1), expected)
    assert counts.dtype == dtype
    assert (counts != before).nnz == 0


@pytest.mark.parametrize("sparse_type", [csr_matrix, csc_matrix])
@pytest.mark.parametrize("kernel_kind", ["dense", "sparse", "precision"])
def test_sparse_q_large_offsets_match_dense_without_mutation(sparse_type, kernel_kind):
    rng = np.random.default_rng(85)
    n = 100
    values = rng.normal(size=(n, 3)) + np.array([0.0, 1e6, 1e8])
    values = np.column_stack([values, np.full(n, 1e8)])
    matrix = diags([np.full(n - 1, 0.2), np.ones(n), np.full(n - 1, 0.2)], [-1, 0, 1])
    kernel = MatrixKernel.from_matrix(
        matrix.toarray() if kernel_kind == "dense" else matrix.tocsr(),
        is_precision=kernel_kind == "precision",
    )
    sparse = sparse_type(values)
    before = sparse.copy()
    expected = spatial_q_test(values, kernel)
    actual = spatial_q_test(sparse, kernel, chunk_size=2)
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-9)
    direct = kernel.xtKx_standardized(sparse, values.mean(0), values.std(0, ddof=1))
    np.testing.assert_allclose(direct, expected[0], rtol=2e-7, atol=1e-9)
    np.testing.assert_allclose(
        spatial_r_test(sparse, sparse, kernel),
        spatial_r_test(values, values, kernel),
        rtol=2e-7,
        atol=1e-9,
    )
    np.testing.assert_array_equal(sparse.data, before.data)
    np.testing.assert_array_equal(sparse.indices, before.indices)
    np.testing.assert_array_equal(sparse.indptr, before.indptr)
    assert actual[0][-1] == 0.0
    assert actual[1][-1] == 1.0


def test_sparse_moments_handle_duplicate_entries_without_mutating_input():
    rng = np.random.default_rng(61)
    values = 1e8 + rng.normal(size=(32, 3))
    duplicate = csr_matrix(
        (
            np.repeat(values.ravel() / 2, 2),
            np.tile(np.repeat(np.arange(3), 2), 32),
            np.arange(33) * 6,
        ),
        shape=values.shape,
    )
    before = duplicate.copy()
    kernel = MatrixKernel.from_matrix(np.diag(np.linspace(0.5, 1.5, 32)))
    np.testing.assert_allclose(
        spatial_q_test(duplicate, kernel), spatial_q_test(values, kernel), rtol=2e-7
    )
    assert not duplicate.has_canonical_format
    np.testing.assert_array_equal(duplicate.data, before.data)
    np.testing.assert_array_equal(duplicate.indices, before.indices)
    np.testing.assert_array_equal(duplicate.indptr, before.indptr)


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize("method", ["liu", "unknown"])
def test_invalid_null_methods_raise_before_calibration(method):
    kernel = MatrixKernel.from_matrix(np.eye(4))
    with pytest.raises(ValueError, match="choose 'clt', 'welch', or 'moments'"):
        compute_null_params(kernel, method=method)
    with pytest.raises(ValueError, match="choose 'clt', 'welch', or 'moments'"):
        spatial_q_test(np.arange(4), kernel, null_params={"method": method})


@pytest.mark.parametrize("backend", ["matrix", "fft2", "rfft2", "nufft"])
@pytest.mark.parametrize("method", ["clt", "welch"])
@pytest.mark.parametrize("correction", [False, True])
def test_null_moments_project_raw_kernel_traces(backend, method, correction):
    # Features are centered by the test, independent of the kernel's trace view.
    rng = np.random.default_rng(37)
    coords = rng.uniform(0, 8, (40, 2))
    kernels = []
    for centered in (True, False):
        if backend == "matrix":
            kernel = MatrixKernel.from_coordinates(coords, method="gaussian", centering=centered)
        elif backend == "nufft":
            kernel = NUFFTKernel(coords, (8, 8), (1, 1), method="gaussian", centering=centered)
        else:
            kernel = FFTKernel((8, 8), method="gaussian", fft_solver=backend, centering=centered)
        kernels.append(kernel)
    data = rng.normal(size=(8, 8, 3) if backend in ("fft2", "rfft2") else (40, 3))
    expected = compute_null_params(kernels[0], method=method, dirichlet_correction=correction)
    actual = compute_null_params(kernels[1], method=method, dirichlet_correction=correction)
    for key in ("mean_Q", "var_Q"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10)
    np.testing.assert_allclose(
        spatial_q_test(data, kernels[1], null_params=actual),
        spatial_q_test(data, kernels[0], null_params=expected),
        rtol=1e-10,
    )
    np.testing.assert_allclose(actual["var_R"], expected["var_R"], rtol=1e-10)
    assert kernels[1].centering is False


@pytest.mark.parametrize("backend", ["fft2", "rfft2", "nufft"])
def test_clt_tail_is_invariant_to_small_kernel_scale(backend):
    rng = np.random.default_rng(24)
    coords = rng.uniform(0, 8, (40, 2))
    data = rng.normal(size=(40, 3) if backend == "nufft" else (8, 8, 3))
    pvalues = []
    for scale in (1.0, 1e-15):
        if backend == "nufft":
            kernel = NUFFTKernel(coords, (8, 8), (1, 1), method="gaussian")
            kernel._fft_kernel.spectrum *= scale
        else:
            kernel = FFTKernel((8, 8), method="gaussian", fft_solver=backend)
            kernel.spectrum *= scale
        params = compute_null_params(kernel, method="clt")
        pvalues.append(spatial_q_test(data, kernel, null_params=params)[1])
    np.testing.assert_allclose(pvalues[0], pvalues[1], rtol=1e-10)


@pytest.mark.parametrize("backend", ["matrix", "fft2", "rfft2", "nufft"])
def test_r_null_projects_raw_kernel_without_mutation(backend):
    rng = np.random.default_rng(140)
    coords = rng.uniform(0, 8, (40, 2))
    x = rng.normal(size=(8, 8, 3) if backend in ("fft2", "rfft2") else (40, 3))
    y = x + rng.normal(size=x.shape)
    kernels = []
    for centered in (True, False):
        if backend == "matrix":
            kernel = MatrixKernel.from_coordinates(coords, method="gaussian", centering=centered)
        elif backend == "nufft":
            kernel = NUFFTKernel(coords, (8, 8), (1, 1), method="gaussian", centering=centered)
        else:
            kernel = FFTKernel((8, 8), method="gaussian", fft_solver=backend, centering=centered)
        kernels.append(kernel)
    expected = spatial_r_test(x, y, kernels[0])
    for method in (None, "clt", "welch", "moments"):
        params = None if method is None else compute_null_params(kernels[1], method=method)
        if params is not None:
            np.testing.assert_allclose(params["var_R"], kernels[0].square_trace(), rtol=1e-10)
        actual = spatial_r_test(x, y, kernels[1], null_params=params, chunk_size=1)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
        assert kernels[1].centering is False


@pytest.mark.parametrize("sparse", [False, True])
def test_blocked_centered_traces_preserve_constant_and_projection_limits(sparse):
    n = 1100  # More than one 8 MiB row block.
    for diagonal in (0.0, 1e-9):
        raw = np.full((n, n), 0.1)
        raw[np.diag_indices(n)] += diagonal
        actual_diagonal = raw[0, 0] - raw[0, 1]
        kernel = MatrixKernel.from_matrix(csr_matrix(raw) if sparse else raw)
        # HKH = actual_diagonal * H: its n-1 nonzero eigenvalues are equal.
        np.testing.assert_allclose(
            [kernel.trace(), kernel.square_trace()],
            [(n - 1) * actual_diagonal, (n - 1) * actual_diagonal**2],
            rtol=1e-12,
            atol=0,
        )


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("centering", [False, True])
@pytest.mark.parametrize("bandwidth", [100.0, 10000.0])
def test_nearly_constant_matrix_null_matches_explicit_projection(sparse, centering, bandwidth):
    rng = np.random.default_rng(30)
    coords = rng.uniform(0, 1, (32, 2))
    raw = MatrixKernel.from_coordinates(
        coords, method="gaussian", bandwidth=bandwidth, centering=False
    ).realization()
    original = raw.copy()
    # Use extended precision for the dense oracle so projection roundoff does
    # not dominate the very small centered operator.
    h = np.eye(32, dtype=np.longdouble) - np.longdouble(1) / 32
    centered = np.asarray(h @ raw @ h, dtype=float)
    reference = MatrixKernel.from_matrix(centered)
    kernel = MatrixKernel.from_matrix(csr_matrix(raw) if sparse else raw, centering=centering)
    expected = compute_null_params(reference)
    actual = compute_null_params(kernel)
    for key in ("mean_Q", "var_Q", "var_R"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-6, atol=0)
    x = np.column_stack([coords[:, 0], rng.normal(size=(32, 2))])
    y = rng.normal(size=x.shape)
    np.testing.assert_allclose(
        spatial_q_test(x, kernel), spatial_q_test(x, reference), rtol=1e-5, atol=0
    )
    np.testing.assert_allclose(
        spatial_r_test(x, y, kernel), spatial_r_test(x, y, reference), rtol=1e-5, atol=0
    )
    assert kernel.centering is centering
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize("backend", ["fft2", "rfft2", "nufft"])
def test_r_pvalues_ignore_kernel_amplitude(backend):
    rng = np.random.default_rng(10)
    coords = rng.uniform(0, 8, (40, 2))
    x = rng.normal(size=(40, 3) if backend == "nufft" else (8, 8, 3))
    y = x + 0.5 * rng.normal(size=x.shape)
    pvalues = []
    for scale in (1.0, 1e-20, 0.0):
        if backend == "nufft":
            kernel = NUFFTKernel(coords, (8, 8), (1, 1), method="gaussian")
            kernel._fft_kernel.spectrum *= scale
        else:
            kernel = FFTKernel((8, 8), method="gaussian", fft_solver=backend)
            kernel.spectrum *= scale
        pvalues.append(spatial_r_test(x, y, kernel)[1])
    np.testing.assert_allclose(pvalues[0], pvalues[1], rtol=1e-9)
    np.testing.assert_array_equal(pvalues[2], 1.0)


@pytest.mark.parametrize("sparse_type", [csr_matrix, csc_matrix])
def test_small_sparse_q_uses_dense_validity_cutoff(sparse_type):
    rng = np.random.default_rng(3)
    kernel = MatrixKernel.from_matrix(-np.diag(np.linspace(1, 2, 32)), method="moran")
    data = rng.normal(size=(32, 2)) * [1e-14, 1.0]
    expected = spatial_q_test(data, kernel)
    actual = spatial_q_test(sparse_type(data), kernel)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert actual[0][0] == 0.0
    assert actual[1][0] == 1.0
