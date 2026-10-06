"""
Unit tests for kernel classes and methods.
"""

import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import hadamard

from sonic.kernels import MatrixKernel


class _SmallImplicitKernel(MatrixKernel):
    """Exercise the real implicit builder without a large dense reference."""

    def _build_kernel(self):
        self._implicit_threshold = 0
        return super()._build_kernel()


class TestMatrixKernel(unittest.TestCase):
    """Test cases for MatrixKernel class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Create a small 5x5 grid
        self.n = 25
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 4, 5)
        xx, yy = np.meshgrid(x, y)
        self.coords = np.column_stack((xx.ravel(), yy.ravel()))

    def test_from_coordinates_docstring_example(self):
        """Test from_coordinates classmethod as shown in docstring."""
        # Example from docstring
        coords = np.random.randn(100, 2)
        kernel = MatrixKernel.from_coordinates(coords, method="gaussian", bandwidth=1.5)

        self.assertEqual(kernel.n, 100)
        self.assertEqual(kernel.method, "gaussian")
        self.assertEqual(kernel.params["bandwidth"], 1.5)

    def test_gaussian_kernel(self):
        """Gaussian RBF — raw K properties (symmetric, unit diagonal)."""
        kernel = MatrixKernel(
            self.coords, mode="coords", method="gaussian", bandwidth=1.0, centering=False
        )
        self.assertEqual(kernel.n, self.n)
        self.assertFalse(kernel.stores_precision)
        K = kernel.realization()
        self.assertEqual(K.shape, (self.n, self.n))
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)
        np.testing.assert_array_almost_equal(np.diag(K), np.ones(self.n))

    def test_matern_kernel(self):
        """Matern — raw K properties (symmetric, unit diagonal)."""
        kernel = MatrixKernel(
            self.coords, mode="coords", method="matern", bandwidth=1.0, nu=1.5, centering=False
        )
        self.assertEqual(kernel.n, self.n)
        self.assertFalse(kernel.stores_precision)
        K = kernel.realization()
        self.assertEqual(K.shape, (self.n, self.n))
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)
        np.testing.assert_array_almost_equal(np.diag(K), np.ones(self.n))

    def test_moran_kernel(self):
        """Moran adjacency — raw K sparsity."""
        kernel = MatrixKernel(
            self.coords, mode="coords", method="moran", k_neighbors=4, centering=False
        )
        self.assertEqual(kernel.n, self.n)
        self.assertFalse(kernel.stores_precision)
        K = kernel.realization()
        self.assertEqual(K.shape, (self.n, self.n))
        K_dense = K.toarray() if hasattr(K, "toarray") else K
        self.assertGreater(np.sum(K_dense == 0), self.n)

    def test_graph_laplacian_kernel(self):
        """Graph Laplacian — construction and shape."""
        kernel = MatrixKernel(
            self.coords, mode="coords", method="graph_laplacian", k_neighbors=4, centering=False
        )
        self.assertEqual(kernel.n, self.n)
        self.assertFalse(kernel.stores_precision)
        K = kernel.realization()
        self.assertEqual(K.shape, (self.n, self.n))

    def test_car_explicit(self):
        """CAR (small N, explicit mode) — raw K symmetry."""
        kernel = MatrixKernel(
            self.coords, mode="coords", method="car", k_neighbors=4, rho=0.9, centering=False
        )
        self.assertEqual(kernel.n, self.n)
        self.assertFalse(kernel.stores_precision)
        K = kernel.realization()
        self.assertEqual(K.shape, (self.n, self.n))
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)

    def test_realization_respects_centering(self):
        """``realization()`` returns raw K with centering=False, HKH with True."""
        k_raw = MatrixKernel(self.coords, method="gaussian", bandwidth=1.0, centering=False)
        k_cen = MatrixKernel(self.coords, method="gaussian", bandwidth=1.0)
        assert k_cen.centering is True
        K = k_raw.realization()
        n = k_raw.n
        H = np.eye(n) - np.ones((n, n)) / n
        np.testing.assert_allclose(k_cen.realization(), H @ K @ H, atol=1e-12)

    def test_xtKx_standardized_invariant_under_centering(self):
        """``xtKx_standardized`` returns the same values for centering=True
        and centering=False — z has mean 0 by construction so Hz = z and
        z^T K z = z^T HKH z identically.
        """
        rng = np.random.default_rng(0)
        X_raw = rng.poisson(lam=2.0, size=(self.n, 5)).astype(float)
        X_sparse = sp.csr_matrix(X_raw)
        means = X_raw.mean(axis=0)
        stds = X_raw.std(axis=0, ddof=1)

        k_raw = MatrixKernel(self.coords, method="matern", bandwidth=1.5, centering=False)
        k_cen = MatrixKernel(self.coords, method="matern", bandwidth=1.5, centering=True)

        q_raw = k_raw.xtKx_standardized(X_sparse, means, stds)
        q_cen = k_cen.xtKx_standardized(X_sparse, means, stds)
        np.testing.assert_allclose(q_cen, q_raw, rtol=1e-12)

        # And both match the explicit ``z^T K z`` (same as ``z^T HKH z``).
        Z = (X_raw - means) / stds
        K = k_raw.realization()
        q_explicit = np.einsum("ij,ik,kj->j", Z, K, Z)
        np.testing.assert_allclose(q_cen, q_explicit, rtol=1e-12)

    def test_xtKx_computation(self):
        """Raw ``xᵀ K x`` (``centering=False``)."""
        kernel = MatrixKernel(
            self.coords, mode="coords", method="gaussian", bandwidth=1.0, centering=False
        )
        x = np.random.randn(self.n)
        result = kernel.xtKx(x)
        K = kernel.realization()
        expected = x.T @ K @ x
        np.testing.assert_almost_equal(result, expected, decimal=10)

    def test_xtKx_centered_matches_H_projection(self):
        """Default centered ``xtKx`` equals ``(H x)ᵀ K (H x)``."""
        kernel = MatrixKernel(self.coords, mode="coords", method="gaussian", bandwidth=1.0)
        assert kernel.centering is True
        x = np.random.randn(self.n)
        K = kernel.realization()
        x_c = x - x.mean()
        expected = x_c.T @ K @ x_c
        np.testing.assert_almost_equal(kernel.xtKx(x), expected, decimal=10)

    def test_xtKx_sparse_input(self):
        """Test quadratic form computation with sparse input."""
        kernel = MatrixKernel(
            self.coords, mode="coords", method="gaussian", bandwidth=1.0, centering=False
        )

        # Create sparse and dense versions
        x_dense = np.random.randn(self.n)
        x_sparse = sp.csr_matrix(x_dense.reshape(-1, 1))

        # Compute with both
        result_dense = kernel.xtKx(x_dense)
        result_sparse = kernel.xtKx(x_sparse)

        # Should be identical
        np.testing.assert_almost_equal(result_sparse, result_dense, decimal=10)

    def test_xtKx_sparse_batch(self):
        """Test quadratic form computation with sparse batch input."""
        kernel = MatrixKernel(
            self.coords, mode="coords", method="gaussian", bandwidth=1.0, centering=False
        )

        # Create sparse and dense batches
        X_dense = np.random.randn(self.n, 10)
        X_sparse = sp.csr_matrix(X_dense)

        # Compute with both
        result_dense = kernel.xtKx(X_dense)
        result_sparse = kernel.xtKx(X_sparse)

        # Should be identical
        np.testing.assert_allclose(result_sparse, result_dense, rtol=1e-10)

        # Verify shape
        self.assertEqual(len(result_sparse), 10)
        self.assertEqual(len(result_dense), 10)

    def test_trace_computation(self):
        """Test RAW trace computation (``centering=False``)."""
        kernel = MatrixKernel(
            self.coords, mode="coords", method="gaussian", bandwidth=1.0, centering=False
        )
        trace_result = kernel.trace()
        K = kernel.realization()
        expected_trace = np.trace(K)
        np.testing.assert_almost_equal(trace_result, expected_trace, decimal=10)

    def test_centered_eigenvalues_dense_match_HKH_eigvalsh(self):
        """Centered eigenvalues on dense ``K`` agree with direct ``eigvalsh(HKH)``."""
        kernel = MatrixKernel(self.coords, method="gaussian", bandwidth=1.0)
        assert kernel.centering is True
        K = kernel.realization()
        n = kernel.n
        H = np.eye(n) - np.ones((n, n)) / n
        expected = np.sort(np.linalg.eigvalsh(H @ K @ H))[::-1]
        got = kernel.eigenvalues()
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)

    def test_centered_eigenvalues_implicit_sparse_match_dense(self):
        """Implicit sparse-precision path (CAR) eigenvalues match the dense
        ``eigvalsh(HKH)`` reference (top-k, since eigsh can't return all).
        """
        kernel = _SmallImplicitKernel(self.coords, method="car", k_neighbors=4, rho=0.9)
        self.assertTrue(kernel.stores_precision)
        K = np.linalg.inv(kernel._K.toarray())
        n = kernel.n
        H = np.eye(n) - np.ones((n, n)) / n
        ref_descending = np.sort(np.linalg.eigvalsh(H @ K @ H))[::-1]
        # Request the top-k eigenvalues.
        k_eigen = 6
        got = kernel.eigenvalues(k=k_eigen)
        np.testing.assert_allclose(got, ref_descending[:k_eigen], rtol=1e-6, atol=1e-8)

    def test_centered_trace_is_raw_minus_constant_mode(self):
        """Default ``centering=True`` trace equals ``trace(K) − 𝟏ᵀK𝟏/n``."""
        kernel = MatrixKernel(self.coords, mode="coords", method="gaussian", bandwidth=1.0)
        assert kernel.centering is True
        K = kernel.realization()
        raw = np.trace(K)
        s1 = float(K.sum())  # 𝟏ᵀ K 𝟏
        expected_centered = raw - s1 / kernel.n
        np.testing.assert_almost_equal(kernel.trace(), expected_centered, decimal=10)

    def test_square_trace_computation(self):
        """Test raw trace(K^2) computation (``centering=False``)."""
        kernel = MatrixKernel(
            self.coords, mode="coords", method="gaussian", bandwidth=1.0, centering=False
        )
        sq_trace = kernel.square_trace()

        # Compute directly
        K = kernel.realization()
        expected = np.sum(K**2)

        np.testing.assert_almost_equal(sq_trace, expected, decimal=10)

    def test_car_implicit(self):
        """Cross the production implicit threshold without materializing K."""
        # Create larger dataset
        n_large = 6400
        x = np.linspace(0, 10, int(np.sqrt(n_large)))
        y = np.linspace(0, 10, int(np.sqrt(n_large)))
        xx, yy = np.meshgrid(x, y)
        coords_large = np.column_stack((xx.ravel(), yy.ravel()))

        # Raw mode lets us check the defining precision solve M @ Kx = x.
        kernel = MatrixKernel(
            coords_large,
            mode="coords",
            method="car",
            k_neighbors=4,
            rho=0.9,
            centering=False,
        )
        self.assertEqual(kernel.n, n_large)
        self.assertTrue(kernel.stores_precision)  # Should be implicit due to size

        x = np.random.randn(n_large)
        Kx = kernel.Kx(x)
        np.testing.assert_allclose(kernel._K @ Kx, x, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(kernel.xtKx(x), x @ Kx, rtol=1e-10)


class TestMatrixKernelStandardization(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        # Small grid for explicit inversion path
        n_side = 6
        x = np.linspace(0, 5, n_side)
        y = np.linspace(0, 5, n_side)
        xx, yy = np.meshgrid(x, y)
        self.coords = np.column_stack((xx.ravel(), yy.ravel()))
        self.n = self.coords.shape[0]

    def test_standardize_explicit_diagonal(self):
        """Diagonal of RAW K should be ~1 when standardize=True (centering=False).

        Raw ``realization()`` exposes ``K`` itself; its unit-diagonal is the
        standardize contract. Centered ``realization()`` returns ``HKH`` —
        diagonal is structurally sub-unity (constant-mode projected out).
        """
        kernel = MatrixKernel.from_coordinates(
            self.coords,
            method="car",
            k_neighbors=4,
            rho=0.85,
            standardize=True,
            centering=False,
        )
        self.assertFalse(kernel.stores_precision)
        K = kernel.realization()
        diag = np.diag(K)
        self.assertEqual(K.shape, (self.n, self.n))
        np.testing.assert_allclose(diag, np.ones_like(diag), rtol=1e-5, atol=1e-3)

    def test_no_standardize_diagonal_not_unity(self):
        """Without standardize, raw-K diagonal need not be exactly 1."""
        kernel = MatrixKernel.from_coordinates(
            self.coords,
            method="car",
            k_neighbors=4,
            rho=0.85,
            standardize=False,
            centering=False,
        )
        K = kernel.realization()
        diag = np.diag(K)
        self.assertGreater(np.max(np.abs(diag - 1.0)), 1e-3)


class TestMatrixKernelImplicitTrace(unittest.TestCase):
    """Test trace and square_trace for implicit kernel matrices."""

    def test_implicit_moments_match_dense(self):
        """Orthogonal ±1 probes give exact moments through the real LU solves."""
        coords = np.indices((4, 4)).reshape(2, -1).T
        kernel = _SmallImplicitKernel(
            coords, method="car", k_neighbors=4, rho=0.85, centering=False
        )
        self.assertTrue(kernel.stores_precision)
        raw = np.linalg.inv(kernel._K.toarray())
        H = np.eye(kernel.n) - np.ones((kernel.n, kernel.n)) / kernel.n
        # Replace only the probe draw, eliminating sampling error in this unit test.
        with patch("sonic.kernels.base.np.random.default_rng") as rng:
            rng.return_value.choice.return_value = hadamard(kernel.n)
            for centering in (False, True):
                with self.subTest(centering=centering):
                    kernel.centering = centering
                    expected = H @ raw @ H if centering else raw
                    np.testing.assert_allclose(kernel.realization(), expected, atol=1e-12)
                    np.testing.assert_allclose(
                        [kernel.trace(n_probes=kernel.n), kernel.square_trace(n_probes=kernel.n)],
                        [np.trace(expected), np.sum(expected**2)],
                        rtol=1e-10,
                    )


class TestKernelUtilities(unittest.TestCase):
    """Additional tests to cover utility helpers and error paths."""

    def setUp(self):
        np.random.seed(7)
        x = np.linspace(0, 3, 4)
        y = np.linspace(0, 3, 4)
        xx, yy = np.meshgrid(x, y)
        self.coords = np.column_stack((xx.ravel(), yy.ravel()))

    def test_format_params_repr_str(self):
        """_format_params, __repr__, and __str__ should handle arrays and sparse matrices."""
        kernel = MatrixKernel.from_coordinates(self.coords, method="gaussian", bandwidth=1.2)
        kernel.params["arr"] = np.ones((2, 2))
        kernel.params["sparse"] = sp.csr_matrix(np.eye(2))

        params_str = kernel._format_params()
        self.assertIn("bandwidth=1.2", params_str)
        self.assertIn("array(shape=(2, 2)", params_str)
        self.assertIn("sparse(shape=(2, 2)", params_str)

        repr_str = repr(kernel)
        self.assertIn("MatrixKernel", repr_str)
        self.assertIn("method=gaussian", repr_str)

        str_out = str(kernel)
        self.assertIn("MatrixKernel", str_out)
        self.assertIn("Method: gaussian", str_out)

    def test_from_matrix_precomputed_and_inverse(self):
        """from_matrix should support precomputed kernels and precision matrices.

        Uses ``centering=False`` since the test checks ``realization() == K``
        (raw buffer) rather than the centered operator ``HKH``.
        """
        K = np.array([[2.0, -1.0], [-1.0, 2.0]])
        kernel_pre = MatrixKernel.from_matrix(K, is_precision=False, centering=False)
        np.testing.assert_allclose(kernel_pre.realization(), K, rtol=1e-12)

        # Singular precision triggers pseudo-inverse path
        M = np.array([[1.0, 0.0], [0.0, 0.0]])
        with pytest.warns(RuntimeWarning, match="Precision matrix is singular"):
            kernel_inv = MatrixKernel.from_matrix(
                M, is_precision=True, method="car", standardize=True, centering=False
            )
        K_inv = kernel_inv.realization()
        self.assertEqual(K_inv.shape, (2, 2))
        self.assertTrue(np.allclose(K_inv, K_inv.T, rtol=1e-12))

    def test_invalid_kernel_param_raises(self):
        """Unknown kernel parameter should raise ValueError."""
        with self.assertRaises(ValueError):
            MatrixKernel.from_coordinates(self.coords, method="gaussian", bad_param=1)

    def test_eigenvalues_cache(self):
        """eigenvalues should reuse cached spectrum for smaller k."""
        kernel = MatrixKernel.from_coordinates(self.coords, method="moran", k_neighbors=2)
        vals_full = kernel.eigenvalues(k=4)
        vals_subset = kernel.eigenvalues(k=2)
        # Spectrum is sorted descending, so top-2 are the first 2 elements
        np.testing.assert_allclose(vals_subset, vals_full[:2], rtol=1e-12)

    def test_getstate_setstate_resets_lu(self):
        """__getstate__ and __setstate__ should reset cached LU factorization."""
        kernel = MatrixKernel.from_coordinates(self.coords, method="gaussian", bandwidth=1.0)
        kernel._lu = object()
        state = kernel.__getstate__()
        self.assertIsNone(state.get("_lu"))

        kernel2 = MatrixKernel.from_coordinates(self.coords, method="gaussian", bandwidth=1.0)
        kernel2.__setstate__(state)
        self.assertIsNone(kernel2._lu)


class TestXtKxStandardized(unittest.TestCase):
    """``MatrixKernel.xtKx_standardized`` must match the manual
    ``z = (X - μ)/σ; kernel.xtKx(z)`` pipeline to float-precision — it's an
    algebraic identity, so the tolerance is ~1e-10, not an approximation band.
    """

    def setUp(self):
        rng = np.random.default_rng(0)
        self.coords = rng.standard_normal((100, 2))
        self.kernel = MatrixKernel.from_coordinates(
            self.coords, method="matern", bandwidth=1.0, nu=1.5
        )

    def test_matches_manual_zscore_dense(self):
        """Dense ``X``: ``xtKx_standardized(X, μ, σ)`` equals
        ``xtKx((X - μ)/σ)`` to float-precision."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 5))
        means = X.mean(axis=0)
        stds = X.std(axis=0, ddof=1)
        Z = (X - means) / stds

        q_fast = self.kernel.xtKx_standardized(X, means, stds)
        q_manual = np.asarray(self.kernel.xtKx(Z))
        np.testing.assert_allclose(q_fast, q_manual, rtol=1e-10, atol=1e-12)

    def test_matches_manual_zscore_sparse(self):
        """Sparse ``X``: sparse fast-path agrees with the dense reference to
        float-precision — verifies the ``(K·1, 1ᵀK1)`` expansion."""
        import scipy.sparse as sp

        rng = np.random.default_rng(2)
        X_dense = rng.standard_normal((100, 4))
        # Sparsify by zeroing ~60% of entries; keep sparsity realistic.
        X_dense[rng.random(X_dense.shape) < 0.6] = 0.0
        X_sparse = sp.csc_matrix(X_dense)
        # Means/stds must be computed from the same underlying data the fast
        # path will see (sparse's column stats).
        means = np.asarray(X_sparse.mean(axis=0)).ravel()
        sq = np.asarray(X_sparse.multiply(X_sparse).sum(axis=0)).ravel()
        n = X_sparse.shape[0]
        var = (sq - n * means**2) / max(n - 1, 1)
        var[var < 0] = 0.0
        stds = np.sqrt(var)
        Z = np.zeros_like(X_dense)
        valid = stds > 1e-12
        Z[:, valid] = (X_dense[:, valid] - means[valid]) / stds[valid]

        q_fast = self.kernel.xtKx_standardized(X_sparse, means, stds)
        q_manual = np.asarray(self.kernel.xtKx(Z))
        np.testing.assert_allclose(q_fast, q_manual, rtol=1e-10, atol=1e-12)

    def test_zero_variance_column_returns_zero(self):
        """Constant columns (std=0) must return 0 per the documented contract."""
        X = np.zeros((100, 3))
        X[:, 0] = 1.0  # constant column → std=0
        X[:, 1] = np.linspace(-1, 1, 100)
        X[:, 2] = 7.5  # another constant column
        means = X.mean(axis=0)
        stds = X.std(axis=0, ddof=1)

        q = self.kernel.xtKx_standardized(X, means, stds)
        self.assertAlmostEqual(q[0], 0.0, places=12)
        self.assertAlmostEqual(q[2], 0.0, places=12)
        # Middle column has nonzero std → nonzero Q.
        self.assertGreater(q[1], 0.0)


def test_standardized_precision_lu_survives_pickle():
    # Cross the production threshold for implicit inversion, without a dense matrix.
    n = 5001
    precision = sp.diags(np.linspace(1, 4, n), format="csc")
    kernel = MatrixKernel.from_matrix(
        precision, is_precision=True, method="car", standardize=True, centering=False
    )
    vector = np.linspace(-1, 1, n)
    np.testing.assert_allclose(kernel.Kx(vector), vector, atol=1e-12)
    restored = pickle.loads(pickle.dumps(kernel))
    np.testing.assert_allclose(restored.Kx(vector), kernel.Kx(vector), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
