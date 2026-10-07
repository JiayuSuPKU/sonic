from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.stats import norm
from tqdm import tqdm

from sonic.detectors.base import Detector
from sonic.kernels import Kernel, MatrixKernel
from sonic.kernels.nufft import NUFFTKernel, _standardize_features
from sonic.statistics import (
    _q_pvalues,
    _resolve_q_null_method,
    _sparse_mean_std,
    apply_bh_correction,
    compute_null_params,
    spatial_q_test,
)
from sonic.utils import _DEFAULT_CHUNK_BUDGET, _parse_memory_budget, resolve_parallelism

__all__ = ["DetectorIrregular"]

logger = logging.getLogger(__name__)


# helper function for parallel Q-stat computation
def _qstat_worker(
    X_csc: sp.csc_matrix,
    feature_indices: np.ndarray,
    kernel_obj: Kernel,
    null_params: dict[str, float | np.ndarray],
    means: np.ndarray,
    stds: np.ndarray,
    names: list[str],
    return_pval: bool,
    chunk_size: int = 64,
) -> list[dict[str, str | float]]:
    """
    Worker function for parallel Q-statistic computation.

    Processes a batch of features using pre-computed null distribution parameters.
    Handles zero-variance features gracefully.

    Parameters
    ----------
    X_csc : scipy.sparse.csc_matrix
        Sparse feature matrix (N_samples, N_features_batch). Column-compressed format.
    feature_indices : np.ndarray
        Global indices of features in this batch (for looking up means/stds/names).
    kernel_obj : Kernel
        Pre-constructed kernel object (shared across workers).
    null_params : dict
        Pre-computed null distribution parameters from compute_null_params().
        Keys: 'mean_Q', 'var_Q'.
    means : np.ndarray
        Global feature means (shape: N_total_features).
    stds : np.ndarray
        Global feature standard deviations (shape: N_total_features).
    names : List[str]
        Global feature names (length: N_total_features).
    return_pval : bool
        Whether to compute p-values for each feature.
    chunk_size : int, default 64
        Number of features to process in each internal batch for vectorization.

    Returns
    -------
    list[dict[str, str or float]]
        Each dict contains: {'Feature': str, 'Q': float, 'P_value': float, 'Z_score': float}
    """
    results = []

    n_features = len(feature_indices)
    mu = null_params["mean_Q"]
    sigma = np.sqrt(null_params["var_Q"])
    # Reuse moments; the kernel selects sparse products or a centered dense RHS.
    use_sparse_fastpath = hasattr(kernel_obj, "xtKx_standardized")

    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        local_slice = slice(start, end)
        batch_global_indices = feature_indices[start:end]

        b_means = means[batch_global_indices]
        b_stds = stds[batch_global_indices]
        valid_mask = b_stds > 1e-9

        if use_sparse_fastpath:
            X_batch_sp = X_csc[:, local_slice]
            Q_batch = np.atleast_1d(kernel_obj.xtKx_standardized(X_batch_sp, b_means, b_stds))
            # xtKx_standardized already returns 0.0 for std<=0 columns.
            P_batch = (
                _q_pvalues(Q_batch, null_params) if return_pval else np.full(Q_batch.shape, np.nan)
            )
        else:
            # Fallback (e.g. raw-matrix "kernel"): densify and z-score explicitly.
            X_batch = X_csc[:, local_slice].toarray()
            Z_batch = np.zeros_like(X_batch)
            if np.any(valid_mask):
                Z_batch[:, valid_mask] = (X_batch[:, valid_mask] - b_means[valid_mask]) / b_stds[
                    valid_mask
                ]
            if return_pval:
                Q_batch, P_batch = spatial_q_test(
                    Z_batch,
                    kernel_obj,
                    null_params,
                    return_pval=True,
                    is_standardized=True,
                    chunk_size=-1,
                )
            else:
                Q_batch = spatial_q_test(
                    Z_batch,
                    kernel_obj,
                    null_params,
                    return_pval=False,
                    is_standardized=True,
                    chunk_size=-1,
                )
                P_batch = np.full(np.shape(Q_batch), np.nan)
            Q_batch = np.atleast_1d(Q_batch)
            P_batch = np.atleast_1d(P_batch)

        Z_scores = (Q_batch - mu) / sigma if sigma > 0 else np.zeros_like(Q_batch)

        for k, global_idx in enumerate(batch_global_indices):
            if not valid_mask[k]:
                res = {"Feature": names[global_idx], "Q": 0.0, "P_value": 1.0, "Z_score": 0.0}
            else:
                res = {
                    "Feature": names[global_idx],
                    "Q": Q_batch[k],
                    "P_value": P_batch[k],
                    "Z_score": Z_scores[k],
                }
            results.append(res)

    return results


# optimized R-stat worker with pre-computed K@Y
def _rstat_worker_chunked(
    X_csc: sp.csc_matrix,
    y_chunk_indices: np.ndarray,
    x_indices: np.ndarray,
    kernel_obj: Kernel,
    null_params: dict[str, float],
    means: np.ndarray,
    stds: np.ndarray,
    names: list[str],
    return_pval: bool,
) -> list[dict[str, str | float]]:
    """
    Optimized worker for R-statistic computation with pre-computed K@Y.

    Pre-computes K@Y_chunk once, then reuses for all X features paired with Y chunk.
    This avoids redundant K matrix multiplications.

    Parameters
    ----------
    X_csc : scipy.sparse.csc_matrix
        Sparse feature matrix (N_samples, N_features).
    y_chunk_indices : np.ndarray
        Indices of Y features to process in this chunk.
    x_indices : np.ndarray
        Indices of all X features to pair with Y chunk.
    kernel_obj : Kernel
        Pre-constructed kernel object.
    null_params : dict
        Pre-computed null variance ``'var_R'`` (``trace(K²)``). The null mean is zero.
    means : np.ndarray
        Feature means for standardization.
    stds : np.ndarray
        Feature standard deviations.
    names : List[str]
        Feature names.
    return_pval : bool
        Whether to compute p-values.

    Returns
    -------
    list[dict[str, str or float]]
        Results for all (x, y) pairs in this chunk.
    """
    results = []
    sigma = np.sqrt(null_params["var_R"])

    # Keep sparse products when stable; otherwise center bounded dense blocks.
    Y_block = X_csc[:, y_chunk_indices]  # (n, n_y) sparse
    X_block = X_csc[:, x_indices]  # (n, n_x) sparse

    y_means = means[y_chunk_indices]
    y_stds = stds[y_chunk_indices]
    x_means = means[x_indices]
    x_stds = stds[x_indices]
    y_valid = y_stds > 1e-9
    x_valid = x_stds > 1e-9

    sx = np.where(x_valid, x_stds, 1.0)
    sy = np.where(y_valid, y_stds, 1.0)
    R_block = None
    if sp.issparse(getattr(kernel_obj, "_K", None)) and not kernel_obj.stores_precision:
        KY = kernel_obj._K @ Y_block
        raw = (X_block.T @ KY).toarray()
        K_sum, K_total = kernel_obj._K_column_sums()
        # Keep both cross terms even for a non-symmetric precomputed matrix.
        left = x_means[:, None] * np.asarray(KY.sum(axis=0))
        right = np.asarray(X_block.T @ K_sum).ravel()[:, None] * y_means[None, :]
        offset = np.outer(x_means, y_means) * K_total
        corrected = raw - left - right + offset
        magnitude = (
            (abs(X_block).T @ abs(KY)).toarray() + np.abs(left) + np.abs(right) + np.abs(offset)
        )
        valid = x_valid[:, None] & y_valid[None, :]
        # Match Q's safeguard: recenter if cancellation loses four or more digits.
        if np.all(np.isfinite(corrected)) and np.all(
            np.abs(corrected[valid]) >= 1e-4 * magnitude[valid]
        ):
            R_block = corrected / (sx[:, None] * sy[None, :])
        del KY

    if R_block is None:
        Zy = Y_block.astype(float, copy=False).toarray()
        Zy -= y_means
        Zy /= sy
        Zy[:, ~y_valid] = 0.0
        if hasattr(kernel_obj, "Kx"):
            KZy = kernel_obj.Kx(Zy)
        else:
            KZy = kernel_obj.dot(Zy) if hasattr(kernel_obj, "dot") else np.asarray(kernel_obj @ Zy)
        R_block = np.empty((len(x_indices), len(y_chunk_indices)))
        # Bound X's dense workspace by the same chunk size as Y.
        for start in range(0, len(x_indices), len(y_chunk_indices)):
            sl = slice(start, start + len(y_chunk_indices))
            Zx = X_block[:, sl].astype(float, copy=False).toarray()
            Zx -= x_means[sl]
            Zx /= sx[sl]
            R_block[sl] = Zx.T @ KZy
    R_block[~x_valid, :] = 0.0
    R_block[:, ~y_valid] = 0.0

    # Vectorized p-value / z-score stage
    if return_pval and sigma > 0:
        Z_scores_block = R_block / sigma
        P_block = 2 * norm.sf(np.abs(Z_scores_block))
    elif return_pval:
        Z_scores_block = np.zeros_like(R_block)
        P_block = np.ones_like(R_block)
    else:
        Z_scores_block = np.full_like(R_block, np.nan)
        P_block = np.full_like(R_block, np.nan)

    # Emit one result row per (x, y) pair — preserves the original output shape
    # (y_idx iterates local positions within this chunk).
    for xi, x_idx in enumerate(x_indices):
        name_x = names[x_idx]
        for yj, y_global_idx in enumerate(y_chunk_indices):
            results.append(
                {
                    "Feature_1": name_x,
                    "Feature_2": names[y_global_idx],
                    "R": R_block[xi, yj],
                    "Z_score": Z_scores_block[xi, yj],
                    "P_value": P_block[xi, yj],
                }
            )

    return results


class DetectorIrregular(Detector):
    r"""
    Detect spatial patterns on **irregular** samples (AnnData spots / cells).

    Univariate (Q-test) and bivariate (R-test) kernel-based spatial statistics.
    Supports two backends:

    - ``backend='matrix'`` — :class:`~sonic.MatrixKernel` (dense or implicit
      sparse-precision, auto-selected by ``n``). Good up to ~10⁴ spots.
    - ``backend='nufft'`` — :class:`~sonic.NUFFTKernel`, ``O(n log n)`` quadratic
      forms on arbitrary point sets. Recommended for ≥ 10⁴ spots.

    The core test statistics are:

    - Univariate:  :math:`Q = \\mathbf{x}^T \\mathbf{K} \\mathbf{x}`
    - Bivariate:  :math:`R = \\mathbf{x}^T \\mathbf{K} \\mathbf{y}`

    Workflow
    --------
    1. **Construct** with kernel method + backend + kernel hyperparameters.
    2. **Setup** with :meth:`setup_data` passing the :class:`anndata.AnnData`
       plus spatial source (``obsm_key`` in ``obsm``, or
       ``obsp_key`` for precomputed adjacency / distance).
    3. **Compute** with :meth:`compute_qstat` / :meth:`compute_rstat`.

    Parameters
    ----------
    kernel_method : str, default ``'matern'``
        One of ``'gaussian'``, ``'matern'``, ``'moran'``, ``'graph_laplacian'``,
        ``'car'``.
    backend : {``'matrix'``, ``'nufft'``}, default ``'matrix'``
        Kernel backend.
    **kernel_params
        Method- and backend-specific kernel hyperparameters. Matrix backend:
        ``bandwidth``, ``nu``, ``rho``, ``k_neighbors``, ``standardize``.
        NUFFT backend: ``bandwidth``, ``nu``, ``rho``, ``neighbor_degree``,
        plus grid controls ``grid_shape``, ``spacing``, ``unit_scale``,
        ``oversample``, ``eps``.

    Attributes
    ----------
    backend\_ : {``'matrix'``, ``'nufft'``}
        Which backend was selected at construction.
    adata : :class:`anndata.AnnData` or None
        Input container set by :meth:`setup_data`.
    min_cells : int or None
        Minimum non-zero count per feature; set by :meth:`setup_data`.
    kernel\_ : :class:`~sonic.kernels.Kernel` or None
        The built kernel; populated by :meth:`setup_data`.
    kernel_method\_, kernel_params\_, n
        See :class:`Detector`.

    Examples
    --------
    >>> import anndata as ad, numpy as np
    >>> from sonic import DetectorIrregular
    >>> rng = np.random.default_rng(0)
    >>> adata = ad.AnnData(X=rng.standard_normal((200, 5)))
    >>> adata.obsm["spatial"] = rng.standard_normal((200, 2))
    >>> det = DetectorIrregular(kernel_method="car", rho=0.9, k_neighbors=8)
    >>> det.setup_data(adata, min_cells=5)  # doctest: +ELLIPSIS
    <DetectorIrregular ...>
    >>> # q = det.compute_qstat()
    """

    _NUFFT_ONLY_GRID_KEYS = ("grid_shape", "spacing", "unit_scale", "oversample", "eps")

    def __init__(
        self,
        kernel_method: str = "matern",
        backend: str = "matrix",
        **kernel_params: Any,
    ) -> None:
        if backend not in ("matrix", "nufft"):
            raise ValueError(f"backend must be 'matrix' or 'nufft', got {backend!r}.")
        self.backend_: str = backend
        """Which backend will build the kernel — ``'matrix'`` or ``'nufft'``."""
        super().__init__(kernel_method, **kernel_params)

        # Data-state attrs (populated by setup_data):
        self.adata: Any | None = None
        """Reference to the input :class:`anndata.AnnData`, set by :meth:`setup_data`."""
        self.min_cells: int | None = None
        """Minimum non-zero-count threshold applied in :meth:`setup_data`."""

    def _merge_kernel_defaults(self, method: str, user_params: dict) -> dict:
        """Merge per-method defaults with user overrides.

        Matrix backend uses ``k_neighbors``; NUFFT backend uses
        ``neighbor_degree`` (matching :class:`FFTKernel`) and exposes extra
        grid-spacing controls.
        """
        if self.backend_ == "matrix":
            method_defaults = {
                "gaussian": {"bandwidth": 2.0},
                "matern": {"bandwidth": 2.0, "nu": 1.5},
                "moran": {"k_neighbors": 4},
                "graph_laplacian": {"k_neighbors": 4},
                "car": {"rho": 0.9, "k_neighbors": 4, "standardize": False},
            }
        else:  # nufft
            method_defaults = {
                "gaussian": {"bandwidth": 2.0},
                "matern": {"bandwidth": 2.0, "nu": 1.5},
                "moran": {"neighbor_degree": 1},
                "graph_laplacian": {"neighbor_degree": 1},
                "car": {"rho": 0.9, "neighbor_degree": 1},
            }
        defaults = method_defaults.get(method, {}).copy()
        # NUFFT accepts grid controls in addition to the kernel-method params.
        # None sentinels → auto-infer inside NUFFTKernel.
        if self.backend_ == "nufft":
            defaults.update(
                {
                    "grid_shape": None,
                    "spacing": None,
                    "unit_scale": 1.0,
                    "oversample": 2.0,
                    "eps": 1e-6,
                }
            )
        for k, v in user_params.items():
            if k not in defaults:
                raise ValueError(
                    f"Unknown parameter {k!r} for method {method!r} under "
                    f"backend={self.backend_!r}. Allowed: {sorted(defaults)}."
                )
            defaults[k] = v
        return defaults

    def setup_data(
        self,
        adata: Any,
        *,
        obsm_key: str = "spatial",
        obsp_key: str | None = None,
        is_distance: bool = False,
        min_cells: int = 1,
        min_cells_frac: float | None = None,
    ) -> DetectorIrregular:
        """
        Attach ``adata``, apply feature filters, build the kernel.

        Parameters
        ----------
        adata : :class:`anndata.AnnData`
            Input container. Must have ``adata.obsm[obsm_key]`` (unless
            ``obsp_key`` is provided instead).
        obsm_key : str, default ``'spatial'``
            Key in ``adata.obsm`` holding ``(n_obs, 2)`` spatial coordinates.
            Used when ``obsp_key`` is ``None``.
        obsp_key : str, optional
            If provided, build the kernel from ``adata.obsp[obsp_key]``
            instead of from coordinates. Not compatible with ``backend='nufft'``.
        is_distance : bool, default ``False``
            When ``obsp_key`` is given: treat the matrix as pairwise distances
            (``True``) or adjacency / connectivity (``False``).
        min_cells : int, default 1
            Minimum number of cells with non-zero value for a feature to be
            tested. Clamped to ``[1, n_obs]``.
        min_cells_frac : float, optional
            If provided, overrides ``min_cells`` with
            ``max(1, int(min_cells_frac * n_obs))``.

        Returns
        -------
        self : DetectorIrregular
        """
        self.adata = adata
        self.n = adata.shape[0]
        if min_cells_frac is not None:
            self.min_cells = max(1, int(min_cells_frac * self.n))
        else:
            self.min_cells = min(min_cells, self.n)

        if obsp_key is not None:
            if self.backend_ == "nufft":
                raise ValueError("obsp_key is not supported with backend='nufft'.")
            self._build_kernel_from_obsp(
                obsp_key,
                is_distance=is_distance,
                method=self.kernel_method_,
                **self.kernel_params_,
            )
        else:
            self._build_kernel_from_obsm(obsm_key=obsm_key)

        self._data_ready = True
        return self

    # ------------------------------------------------------------------
    # Auto-tuning helpers
    # ------------------------------------------------------------------
    def _auto_chunk_size(self, n_jobs: int = 1, budget_bytes: int = _DEFAULT_CHUNK_BUDGET) -> int:
        """Resolve a batch within the workspace shared by actual outer jobs."""
        from sonic.utils import auto_chunk_size

        return auto_chunk_size(self.kernel_, n_jobs=n_jobs, budget_bytes=budget_bytes)

    def _build_kernel_from_obsm(self, obsm_key: str = "spatial") -> None:
        """Build the kernel over ``adata.obsm[obsm_key]`` using the
        backend / method / params selected at construction. Called from
        :meth:`setup_data`.
        """
        if obsm_key not in self.adata.obsm:
            raise KeyError(
                f"adata.obsm has no key '{obsm_key}'; "
                f"available: {list(self.adata.obsm.keys())}."
            )
        coords = np.asarray(self.adata.obsm[obsm_key], dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"adata.obsm['{obsm_key}'] must be (n_obs, 2), got {coords.shape}.")
        if coords.shape[0] != self.n:
            raise ValueError(
                f"coords shape {coords.shape} inconsistent with adata.shape=({self.n}, ...)."
            )

        logger.info(
            "Building %s %sKernel over %d spots...",
            self.kernel_method_,
            "Matrix" if self.backend_ == "matrix" else "NUFFT",
            self.n,
        )
        if self.backend_ == "matrix":
            self.kernel_ = MatrixKernel.from_coordinates(
                coords, method=self.kernel_method_, **self.kernel_params_
            )
        else:
            self.kernel_ = NUFFTKernel(
                coords=coords, method=self.kernel_method_, **self.kernel_params_
            )
            # NUFFTKernel resolves its own params (including any None-sentinel
            # auto-infers); snapshot them back so kernel_params_ reflects reality.
            self.kernel_params_ = dict(self.kernel_.params)

    def _build_kernel_from_obsp(  # noqa: C901
        self, key: str, is_distance: bool = False, method: str = "car", **kernel_params: Any
    ) -> None:
        """Build a graph kernel from ``adata.obsp[key]``. Called from
        :meth:`setup_data` when ``obsp_key`` is provided.

        Handles 'precomputed' (matrix IS the kernel), distance matrices
        (Gaussian / Matern transform), and adjacency matrices (Moran /
        graph_laplacian / CAR). Isolated nodes (zero degree) are removed.
        """
        if key not in self.adata.obsp:
            raise KeyError(f"Matrix key '{key}' not found in adata.obsp")

        matrix = self.adata.obsp[key]

        if method not in list(self._available_kernels) + ["precomputed"]:
            raise ValueError(
                f"Method '{method}' not recognized. Must be one of {self._available_kernels} or 'precomputed'."
            )

        # If the user says 'precomputed', they imply the matrix IS the kernel K
        if method == "precomputed":
            logger.info("Using obsp['%s'] directly as kernel matrix (n_samples=%d)...", key, self.n)
            self.kernel_ = MatrixKernel.from_matrix(matrix, is_precision=False)
            self.kernel_params_ = kernel_params
            self.kernel_method_ = method
            return

        logger.info(
            "Building %s kernel from obsp['%s'] (is_distance=%s, n_samples=%d)...",
            method,
            key,
            is_distance,
            self.n,
        )

        # kernel_params already carries defaults merged in __init__; no re-merge needed.
        kernel_params_ = dict(kernel_params)

        # --- Distance Based Transformations ---
        if is_distance:
            if method == "gaussian":
                bw = kernel_params_["bandwidth"]
                # K = exp(-d^2 / 2bw^2)
                if sp.issparse(matrix):
                    matrix = matrix.toarray()  # Gaussian usually requires dense
                K = np.exp(-(matrix**2) / (2 * bw**2))
                self.kernel_ = MatrixKernel.from_matrix(K, is_precision=False)
                self.kernel_params_ = {"bandwidth": bw}
            elif method == "matern":
                from scipy.special import gamma, kv

                bw = kernel_params_["bandwidth"]
                nu = kernel_params_["nu"]
                if sp.issparse(matrix):
                    matrix = matrix.toarray()

                dists = matrix.copy()
                dists[dists == 0] = 1e-15
                factor = (np.sqrt(2 * nu) * dists) / bw
                K = (2 ** (1 - nu) / gamma(nu)) * (factor**nu) * kv(nu, factor)
                np.fill_diagonal(K, 1.0)
                self.kernel_ = MatrixKernel.from_matrix(K, is_precision=False)
                self.kernel_params_ = {"bandwidth": bw, "nu": nu}
            else:
                raise ValueError(f"Method {method} not supported for distance matrices.")

        # --- Connectivity Based Transformations ---
        else:
            # Assume the input matrix is W (adjacency/connectivity)
            W = matrix
            if not sp.issparse(W):
                W = sp.csr_matrix(W)

            # Remove isolated cells (zero-degree nodes)
            row_sums_raw = np.array(W.sum(axis=1)).flatten()
            keep_mask = row_sums_raw > 0
            if not np.all(keep_mask):
                removed = int((~keep_mask).sum())
                logger.info(
                    "Removing %d isolated samples with zero degree from adjacency matrix...",
                    removed,
                )
                W = W[keep_mask][:, keep_mask]
                self.adata = self.adata[keep_mask].copy()
                self.n = W.shape[0]
                self.min_cells = min(self.min_cells, self.n)

            # Symmetrize and symmetric normalization
            W = W.astype(float)
            W_sym = 0.5 * (W + W.T)
            row_sums = np.array(W_sym.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1.0
            inv_D_sqrt = sp.diags(1.0 / np.sqrt(row_sums))
            W_norm = inv_D_sqrt @ W_sym @ inv_D_sqrt

            if method == "moran":
                # Already symmetric and normalized
                K = W_norm
                self.kernel_ = MatrixKernel.from_matrix(K, is_precision=False)
                self.kernel_params_ = {}

            elif method == "graph_laplacian":
                # K = I - W_norm
                I = sp.eye(self.n, format="csr")
                K = I - W_norm
                self.kernel_ = MatrixKernel.from_matrix(K, is_precision=False)
                self.kernel_params_ = {}

            elif method == "car":
                # This is the "Implicit" case.
                # The kernel K = (I - rho*W)^-1.
                # We construct M = (I - rho*W) and tell MatrixKernel it is the INVERSE.
                rho = kernel_params_["rho"]
                standardize = kernel_params_["standardize"]
                I = sp.eye(self.n, format="csc")
                M = I - rho * W_norm  # M is the inverse of K

                # We pass M and set is_precision=True. MatrixKernel will handle
                # standardizing the precision to ensure diag(K)=1 if requested.
                self.kernel_ = MatrixKernel.from_matrix(
                    M,
                    is_precision=True,
                    method="car",
                    rho=rho,
                    standardize=standardize,
                )
                self.kernel_params_ = {"rho": rho, "standardize": standardize}

            else:
                raise ValueError(f"Method {method} not supported for connectivity matrices.")

        self.backend_ = "matrix"

    def _prepare_data(
        self, source: str, keys: list[str] | None, min_cells: int, layer: str | None = None
    ) -> tuple[sp.csr_matrix, list[str], np.ndarray, np.ndarray]:
        """
        Prepare anndata feature data for testing (standardization, filtering).

        Extracts features from .obs or .var, applies quality filters (min non-zero count),
        handles categorical features (one-hot encoding), and computes per-feature statistics.

        Parameters
        ----------
        source : str
            Feature source: 'obs' or 'var'.
        keys : Optional[List[str]]
            Feature names to extract. If None, uses all features in source.
        min_cells : int
            Minimum number of non-zero values required per feature.
        layer : Optional[str]
            If source='var', which layer to use. If None, uses .X.

        Returns
        -------
        X_csc : scipy.sparse.csc_matrix
            Sparse feature matrix (n_samples, n_features), column-compressed.
        names : List[str]
            Feature names in order matching X_csc columns.
        means : np.ndarray
            Per-feature means (shape: n_features).
        stds : np.ndarray
            Per-feature standard deviations (shape: n_features).

        Notes
        -----
        - Categorical features in .obs are automatically one-hot encoded.
        - Features with zero variance or fewer than min_cells non-zeros are filtered.
        - Sparse matrices are preserved for memory efficiency.
        """

        if source == "obs":
            # check if keys are in obs
            if keys is not None:
                valid = [k for k in keys if k in self.adata.obs.columns]
                if len(valid) == 0:
                    raise ValueError("None of the specified keys found in adata.obs.")
                adata_tmp = self.adata.obs[valid].copy()
            else:
                raise ValueError("Keys must be provided when feature source is 'obs'.")

            # check if .obs[keys] are char/categorical
            if any(adata_tmp.dtypes == "object") or any(adata_tmp.dtypes == "category"):
                logger.info(
                    "Categorical features detected in .obs[keys]; performing one-hot encoding..."
                )
                # one-hot encode categorical variables while keeping others unchanged
                dummies = pd.get_dummies(adata_tmp)

                names = dummies.columns.tolist()
                X_dense = dummies.values.astype(np.float64)
                means = X_dense.mean(axis=0)
                stds = X_dense.std(axis=0, ddof=1)
                X_csc = sp.csc_matrix(X_dense)
            else:
                # All numeric - no encoding needed
                names = adata_tmp.columns.tolist()
                X_dense = adata_tmp.values.astype(np.float64)
                means = X_dense.mean(axis=0)
                stds = X_dense.std(axis=0, ddof=1)
                X_csc = sp.csc_matrix(X_dense)

        elif source == "var":
            # check if keys are in var
            if keys is not None:
                valid = [g for g in keys if g in self.adata.var_names]
                adata_tmp = self.adata[:, valid]
            else:
                adata_tmp = self.adata

            # extract feature matrix
            if layer is not None:
                X = adata_tmp.layers[layer]
            else:
                X = adata_tmp.X

            names = adata_tmp.var_names.tolist()

            # compute means and stds (ddof=1 for sample std, consistent with statistics.py)
            if sp.issparse(X):
                X = X.astype(np.float64, copy=False)
                means, stds = _sparse_mean_std(X)
                X_csc = X.tocsc()
            else:
                means = np.mean(X, axis=0)
                stds = np.std(X, axis=0, ddof=1)
                X_csc = sp.csc_matrix(X)
        else:
            raise ValueError("Source must be either 'obs' or 'var'.")

        # Filter constant features and those with too few non-zeros
        to_keep = (stds > 0) & (X_csc.getnnz(axis=0) >= min_cells)
        X_csc = X_csc[:, to_keep]
        names = [names[i] for i in range(len(names)) if to_keep[i]]
        means = means[to_keep]
        stds = stds[to_keep]

        return X_csc, names, means, stds

    def compute_qstat(  # noqa: C901
        self,
        source: str = "var",
        features: list[str] | None = None,
        n_jobs: int = -1,
        layer: str | None = None,
        return_pval: bool = True,
        chunk_size: int | str = "auto",
        show_progress: bool = True,
        workers: int | str | None = "auto",
        *,
        memory_budget_bytes: int | str = _DEFAULT_CHUNK_BUDGET,
    ) -> pd.DataFrame:
        """
        Compute univariate spatial Q-statistic for selected features.

        Tests each feature for significant spatial clustering or dispersion using the
        pre-built kernel. Parallelizes across features and applies Benjamini-Hochberg
        multiple testing correction.

        Parameters
        ----------
        source : str, default 'var'
            Feature source: 'var' (genes) or 'obs' (metadata columns).
        features : Optional[List[str]]
            Feature names to test. If None, tests all features in source.
        n_jobs : int, default -1
            Number of parallel jobs. -1 uses all available cores; 1 for sequential.
        layer : Optional[str]
            If source='var', which layer to use (e.g., 'raw', 'log1p'). If None, uses .X.
        return_pval : bool, default True
            If True, returns p-values and BH-corrected p-values. If False, returns Q only.
        chunk_size : int or ``'auto'``, default ``'auto'``
            Number of features each worker densifies at once (inner batch). ``'auto'``
            shares ``memory_budget_bytes`` across actual outer jobs,
            subject to the backend cap in :func:`sonic.utils.auto_chunk_size`.
        show_progress : bool, default True
            Show a tqdm progress bar over worker chunks.

        workers : int, 'auto', or None, default 'auto'
            FINUFFT threads per job on the NUFFT backend; ignored for matrices.
            Automatic threads use spare CPUs, capped at four. ``None`` uses one.
        memory_budget_bytes : int or str, default 2147483648 (2 GiB)
            Bytes or a size string, e.g. ``"512 MB"`` or ``"2 GiB"``. Units are
            case-insensitive: GB uses powers of 1000, GiB powers of 1024.
            Positive estimated batch-workspace budget shared across actual jobs
            for ``chunk_size='auto'``. More jobs divide this fixed budget; raise
            it explicitly to allow larger batches. Excludes stored inputs,
            kernels/factorizations, retained results and null calibration.
            Explicit chunk sizes bypass sizing; not a total-process memory limit.

        Returns
        -------
        df : pd.DataFrame
            Results sorted by Q (descending). Columns:
            - Feature: feature name
            - Q: test statistic (univariate spatial variability)
            - Z_score: standardized Q by null mean/std
            - P_value: tail probability under null (if return_pval=True)
            - P_adj: Benjamini-Hochberg adjusted p-value (if return_pval=True)

        Raises
        ------
        ValueError
            If kernel not initialized, or source is invalid.

        Notes
        -----
        Under H₀: feature has no spatial structure.
        Under H₁: significant spatial signal (clustering or dispersion).

        Features failing the expression or variance filters are omitted. If no
        features remain, returns an empty result with the usual columns.

        The null approximation depends on both method and backend. Gaussian,
        Matérn, CAR and graph-Laplacian kernels use upper-tail ``welch`` on the
        matrix backend and upper-tail ``moments`` on NUFFT. Moran and other
        recognized signed spectra use two-sided ``clt``. All defaults correct
        for sample standardization. NUFFT moments use analytic lower traces
        and 60 probes for higher traces. This method does not expose a null
        override; use :func:`sonic.statistics.spatial_q_test` with a prepared cache from
        :func:`sonic.statistics.compute_null_params` to select another calibration.

        Examples
        --------
        >>> detector.setup_data(adata)
        >>> results = detector.compute_qstat(source='var', features=['Gene1', 'Gene2'], n_jobs=-1)
        >>> top_genes = results.iloc[:10]
        """

        # 1. Ensure Kernel Exists
        memory_budget_bytes = _parse_memory_budget(memory_budget_bytes)
        self._require_setup()

        auto_chunk = chunk_size == "auto"
        if isinstance(chunk_size, str):
            if not auto_chunk:
                raise ValueError(f"chunk_size must be 'auto' or int, got {chunk_size!r}.")
            chunk_size = self._auto_chunk_size(budget_bytes=memory_budget_bytes)
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        n_features = (
            len(features)
            if features is not None
            else self.adata.n_vars if source == "var" else len(self.adata.obs.columns)
        )
        n_batches = max(1, (n_features + chunk_size - 1) // chunk_size)
        n_jobs, n_workers = resolve_parallelism(
            n_jobs,
            workers if self.backend_ == "nufft" else 1,
            backend="nufft" if self.backend_ == "nufft" else "matrix",
            n_tasks=n_batches,
        )
        if self.backend_ == "nufft":
            self.kernel_.nthreads = n_workers
        if auto_chunk:
            chunk_size = self._auto_chunk_size(n_jobs=n_jobs, budget_bytes=memory_budget_bytes)

        # NUFFT computes Q spectrally at the original observations.
        if self.backend_ == "nufft":
            return self._compute_qstat_nufft(
                source=source,
                features=features,
                n_jobs=n_jobs,
                layer=layer,
                return_pval=return_pval,
                chunk_size=chunk_size,
                show_progress=show_progress,
            )

        # 2. Prepare Data
        X_csc, names, means, stds = self._prepare_data(
            source=source, keys=features, min_cells=self.min_cells, layer=layer
        )
        if not names:
            columns = ["Q", "P_value", "Z_score", "P_adj"] if return_pval else ["Q", "Z_score"]
            return pd.DataFrame(columns=columns, index=pd.Index([], name="Feature"), dtype=float)

        # 3. Compute Null Distribution
        null_method = _resolve_q_null_method(self.kernel_)
        logger.info("Computing null distribution approximation (method=%s)...", null_method)
        null_params = compute_null_params(self.kernel_, method=null_method)

        # 4. Parallel Execution
        n_feats = len(names)

        indices = np.arange(n_feats)
        chunks = np.array_split(indices, max(n_jobs * 4, 1))

        logger.info(
            "Testing %d features (n_jobs=%d, chunk_size=%d)...", n_feats, n_jobs, chunk_size
        )

        chunk_iter = chunks
        if show_progress:
            chunk_iter = tqdm(
                chunks,
                desc=f"Q ({self.kernel_method_})",
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
            )

        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_qstat_worker)(
                X_csc[:, chunk_idxs],
                chunk_idxs,
                self.kernel_,
                null_params,
                means,
                stds,
                names,
                return_pval,
                chunk_size,
            )
            for chunk_idxs in chunk_iter
        )

        # 5. Aggregate Results
        flat_results = [item for sublist in results_list for item in sublist]
        df = pd.DataFrame(flat_results).set_index("Feature")
        if not return_pval:
            df = df.drop(columns=["P_value"])

        # 6. Multiple testing correction (Benjamini-Hochberg)
        if return_pval:
            df["P_adj"] = apply_bh_correction(df["P_value"])

        return df.sort_values(by="Q", ascending=False)

    def compute_rstat(  # noqa: C901
        self,
        features_x: list[str] | None = None,
        features_y: list[str] | None = None,
        source: str = "var",
        n_jobs: int = -1,
        layer: str | None = None,
        return_pval: bool = True,
        chunk_size: int | str = "auto",
        show_progress: bool = True,
        workers: int | str | None = "auto",
        *,
        memory_budget_bytes: int | str = _DEFAULT_CHUNK_BUDGET,
    ) -> pd.DataFrame:
        """
        Compute bivariate spatial R-statistic (cross-spatial correlation) for feature pairs.

        Tests for significant spatial co-variation between pairs of features using
        the pre-built kernel. Supports symmetric (all pairs within one set) or bipartite
        (all X vs Y pairs) modes. Parallelizes computation and applies multiple testing correction.

        Parameters
        ----------
        features_x : Optional[List[str]]
            Feature names for the first set. If None and features_y is None, uses all features (symmetric mode).
        features_y : Optional[List[str]]
            Feature names for the second set. If None, computes all pairwise within features_x.
            If provided, computes all X vs Y pairs (bipartite mode).
        source : str, default 'var'
            Feature source: 'var' (genes) or 'obs' (metadata columns).
        n_jobs : int, default -1
            Number of parallel jobs. -1 uses all available cores; 1 for sequential.
        layer : Optional[str]
            If source='var', which layer to use (e.g., 'raw', 'log1p'). If None, uses .X.
        return_pval : bool, default True
            If True, returns p-values and BH-corrected p-values. If False, returns R only.
        chunk_size : int or ``'auto'``, default ``'auto'``
            Number of Y features to batch together when pre-computing ``K @ Y_chunk``.
            ``'auto'`` shares ``memory_budget_bytes`` across outer jobs;
            integer values override the heuristic.
        show_progress : bool, default True
            Show a tqdm progress bar over the Y-chunk loop.

        workers : int, 'auto', or None, default 'auto'
            FINUFFT threads per job on the NUFFT backend; ignored for matrices.
            Automatic threads use spare CPUs, capped at four. ``None`` uses one.
            NUFFT R comparisons have a sequential outer loop.
        memory_budget_bytes : int or str, default 2147483648 (2 GiB)
            Bytes or a size string, e.g. ``"512 MB"`` or ``"2 GiB"``. Units are
            case-insensitive: GB uses powers of 1000, GiB powers of 1024.
            Positive estimated batch-workspace budget shared across actual jobs
            for ``chunk_size='auto'``. More jobs divide this fixed budget; raise
            it explicitly to allow larger batches. Excludes stored inputs,
            kernels/factorizations, retained results and null calibration,
            including the full standardized X block on the NUFFT path.
            Explicit chunk sizes bypass sizing; not a total-process memory limit.

        Returns
        -------
        df : pd.DataFrame
            Results sorted by absolute Z_score (descending). Columns:

            - Feature_1: name of first feature
            - Feature_2: name of second feature
            - R: test statistic (bivariate spatial correlation, range approximately [-1, 1])
            - Z_score: standardized R by null mean/std
            - P_value: two-tailed p-value under null (if return_pval=True)
            - P_adj: Benjamini-Hochberg adjusted p-value (if return_pval=True)

        Raises
        ------
        ValueError
            If kernel not initialized, features_x is None when features_y is provided, or no valid pairs generated.

        Notes
        -----
        Under H₀: features are spatially independent.
        Under H₁: significant spatial co-clustering or co-dispersion.

        Unlike :func:`sonic.statistics.spatial_r_test`, this method always returns R-statistics
        for all requested feature pairs in the symmetric mode (``features_y=None``). For
        ``features_x=[A, B, C]``, the output contains
        ``(A, A), (A, B), (A, C), (B, A), (B, B), (B, C), (C, A), (C, B), (C, C)``.

        P-value calculation uses a normal approximation based on Tr(K²) and is not
        configurable through this method. For finer control over the null model,
        call :func:`sonic.statistics.spatial_r_test` directly.

        Features failing the expression or variance filters are omitted.
        A zero-variance null returns Z_score=0 and P_value=1.

        Examples
        --------
        >>> detector.setup_data(adata)
        >>> # All pairwise correlations within gene set
        >>> results = detector.compute_rstat(features_x=['Gene1', 'Gene2', 'Gene3'], n_jobs=-1)
        >>> # Cross-correlation between two gene sets
        >>> results = detector.compute_rstat(
        ...     features_x=['Gene1', 'Gene2'],
        ...     features_y=['Gene3', 'Gene4'],
        ...     n_jobs=-1
        ... )
        """
        memory_budget_bytes = _parse_memory_budget(memory_budget_bytes)
        self._require_setup()

        auto_chunk = chunk_size == "auto"
        if isinstance(chunk_size, str):
            if not auto_chunk:
                raise ValueError(f"chunk_size must be 'auto' or int, got {chunk_size!r}.")
            chunk_size = self._auto_chunk_size(budget_bytes=memory_budget_bytes)
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        targets = features_y if features_y is not None else features_x
        n_targets = len(targets) if targets is not None else self.adata.n_vars
        n_batches = max(1, (n_targets + chunk_size - 1) // chunk_size)
        # The NUFFT R loop is sequential; only its transforms use threads.
        n_jobs, n_workers = resolve_parallelism(
            1 if self.backend_ == "nufft" else n_jobs,
            workers if self.backend_ == "nufft" else 1,
            backend="nufft" if self.backend_ == "nufft" else "matrix",
            n_tasks=1 if self.backend_ == "nufft" else n_batches,
        )
        if self.backend_ == "nufft":
            self.kernel_.nthreads = n_workers
        if auto_chunk:
            chunk_size = self._auto_chunk_size(n_jobs=n_jobs, budget_bytes=memory_budget_bytes)

        if self.backend_ == "nufft":
            return self._compute_rstat_nufft(
                features_x=features_x,
                features_y=features_y,
                source=source,
                n_jobs=n_jobs,
                layer=layer,
                return_pval=return_pval,
                chunk_size=chunk_size,
                show_progress=show_progress,
            )

        # 1. Compute Null Params for R
        # R uses Trace(K^2), without the finite-sample Q variance correction.
        logger.info("Computing null distribution for R statistic...")
        q_null = compute_null_params(self.kernel_, method="clt")
        null_params = {
            "var_R": q_null["var_R"],
        }

        # 2. Prepare Data
        # We load all unique features needed
        if features_y is None:
            unique_feats = features_x
            mode = "symmetric"
        else:
            if features_x is None:
                raise ValueError("features_x cannot be None.")
            unique_feats = list(set(features_x) | set(features_y))
            mode = "bipartite"

        X_csc, names, means, stds = self._prepare_data(
            source=source, keys=unique_feats, min_cells=self.min_cells, layer=layer
        )

        # Map names to indices in the prepared matrix
        name_to_idx = {n: i for i, n in enumerate(names)}

        # 3. Generate Y chunks (for pre-computing K@Y)
        if mode == "symmetric":
            valid_feats = (
                names if features_x is None else [f for f in features_x if f in name_to_idx]
            )
            valid_y_indices = [name_to_idx[f] for f in valid_feats]
        else:
            valid_y = [f for f in features_y if f in name_to_idx]
            valid_y_indices = [name_to_idx[f] for f in valid_y]

        # Chunk Y indices for K@Y pre-computation
        y_chunks = []
        for start in range(0, len(valid_y_indices), chunk_size):
            end = min(start + chunk_size, len(valid_y_indices))
            y_chunks.append(valid_y_indices[start:end])

        # 4. Generate X features list
        if mode == "symmetric":
            valid_x_indices = [name_to_idx[f] for f in valid_feats]
        else:
            valid_x = [f for f in features_x if f in name_to_idx]
            valid_x_indices = [name_to_idx[f] for f in valid_x]

        if len(valid_x_indices) == 0 or len(valid_y_indices) == 0:
            raise ValueError("No valid features found after filtering.")

        # 5. Parallel Execution: Process each Y chunk.
        # n_jobs is pre-resolved at the compute_rstat entry point.
        logger.info(
            "Testing %d x %d pairs using %d cores with chunk_size=%d...",
            len(valid_x_indices),
            len(valid_y_indices),
            n_jobs,
            chunk_size,
        )

        results_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_rstat_worker_chunked)(
                X_csc,
                y_chunk,
                valid_x_indices,
                self.kernel_,
                null_params,
                means,
                stds,
                names,
                return_pval,
            )
            for y_chunk in (
                tqdm(
                    y_chunks,
                    desc=f"R ({self.kernel_method_})",
                    bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
                )
                if show_progress
                else y_chunks
            )
        )

        # 6. Aggregate
        flat_results = [item for sublist in results_list for item in sublist]
        df = pd.DataFrame(flat_results)

        # 7. Multiple testing correction (Benjamini-Hochberg)
        if return_pval:
            df["P_adj"] = apply_bh_correction(df["P_value"])

        return df.sort_values(by="Z_score", key=abs, ascending=False)

    # ------------------------------------------------------------------
    # NUFFT backend paths
    # ------------------------------------------------------------------

    def _prepare_features_nufft(
        self, source: str, features: list[str] | None, layer: str | None
    ) -> tuple[sp.csc_matrix, list[str], np.ndarray, np.ndarray]:
        """Pull a sparse ``(n_spots, n_features)`` CSC + names + per-feature
        mean/std. NUFFT-friendly: computes variance from sparse moments and
        never materializes a full dense ``X``."""
        if source == "var":
            X = self.adata.X if layer is None else self.adata.layers[layer]
            X_csc = X.tocsc() if sp.issparse(X) else sp.csc_matrix(X)
            names = list(self.adata.var_names)
            if features is not None:
                selected = [g for g in features if g in names]
                missing = set(features) - set(selected)
                if missing:
                    logger.warning(
                        "Requested features not in adata.var_names: %s", sorted(missing)[:5]
                    )
                idx = [names.index(g) for g in selected]
                X_csc = X_csc[:, idx]
                names = selected
        elif source == "obs":
            if features is None:
                raise ValueError("source='obs' requires `features` (obs column names).")
            cols = [features] if isinstance(features, str) else list(features)
            missing = [c for c in cols if c not in self.adata.obs.columns]
            if missing:
                raise KeyError(f"obs columns missing: {missing}")
            X_csc = sp.csc_matrix(self.adata.obs[cols].to_numpy(dtype=np.float64))
            names = list(cols)
        else:
            raise ValueError(f"source must be 'var' or 'obs', got '{source}'.")

        # Integer count products can overflow before the variance is computed.
        X_csc = X_csc.astype(np.float64, copy=False)
        nnz_per = np.asarray((X_csc != 0).sum(axis=0)).ravel()
        means, stds = _sparse_mean_std(X_csc, ddof=0)
        keep = (stds > 0) & (nnz_per >= self.min_cells)

        X_kept = X_csc[:, keep]
        names_kept = [names[i] for i, k in enumerate(keep) if k]
        return X_kept, names_kept, means[keep], stds[keep]

    def _compute_qstat_nufft(  # noqa: C901
        self,
        source: str,
        features: list[str] | None,
        n_jobs: int,
        layer: str | None,
        return_pval: bool,
        chunk_size: int,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """NUFFT dispatch for :meth:`compute_qstat`. Builds null params once
        (n-point-scaled) and delegates per-feature work to
        :func:`sonic.spatial_q_test` on Path A (spectral).

        The NUFFT Q-test targets the n-point operator ``K``, using analytic
        lower traces from :meth:`NUFFTKernel.trace` and
        :meth:`~NUFFTKernel.square_trace`, plus probes for higher traces.
        All are already n-point-scaled; no ``n / (ny · nx)`` rescaling is needed.
        """
        kernel = self.kernel_
        null_params: dict[str, float | np.ndarray] | None = None
        if return_pval:
            # Analytic lower traces plus probes; signed spectra use CLT.
            nm = _resolve_q_null_method(kernel)
            null_params = compute_null_params(kernel, method=nm)

        logger.info("Preparing %s features (layer=%s)...", source, layer)
        X_kept, names_kept, means, stds = self._prepare_features_nufft(source, features, layer)
        n_feats = len(names_kept)
        if n_feats == 0:
            cols = ["Feature", "Q", "Z_score", "P_value", "P_adj"]
            return pd.DataFrame(columns=cols)

        logger.info("Testing %d features via NUFFT (n_jobs=%s)...", n_feats, n_jobs)

        def _batch(batch_idx: np.ndarray) -> list[dict[str, Any]]:
            # Densify one small block at a time; never materialize full X.
            # Let spatial_q_test standardize at the n irregular observations;
            # the internal Fourier grid does not set the null sample size.
            block = np.asarray(X_kept[:, batch_idx].todense(), dtype=np.float64)
            if return_pval:
                Q_arr, P_arr = spatial_q_test(block, kernel, null_params=null_params, chunk_size=-1)
                Q_arr = np.atleast_1d(Q_arr)
                P_arr = np.atleast_1d(P_arr)
            else:
                Q_arr = np.atleast_1d(
                    spatial_q_test(block, kernel, return_pval=False, chunk_size=-1)
                )
                P_arr = np.full_like(Q_arr, np.nan)
            # Reference trace and trace²-based Z for reporting.
            if null_params is not None:
                # All Q calibration methods report the same mean/variance fields.
                trK = float(null_params["mean_Q"])
                varQ = float(null_params["var_Q"])
                sigma = float(np.sqrt(varQ)) if varQ > 0 else 0.0
                Z_arr = (Q_arr - trK) / sigma if sigma > 0 else np.zeros_like(Q_arr)
            else:
                Z_arr = np.full_like(Q_arr, np.nan)

            out = []
            for local_i, gi in enumerate(batch_idx):
                out.append(
                    {
                        "Feature": names_kept[gi],
                        "Q": float(Q_arr[local_i]),
                        "Z_score": float(Z_arr[local_i]),
                        "P_value": float(P_arr[local_i]),
                    }
                )
            return out

        idx_all = np.arange(n_feats)
        batches = [idx_all[i : i + chunk_size] for i in range(0, n_feats, chunk_size)]
        batch_iter = (
            tqdm(batches, desc=f"Q (NUFFT, {self.kernel_method_})") if show_progress else batches
        )
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_batch)(batch) for batch in batch_iter
        )
        flat = [row for chunk in results for row in chunk]
        df = pd.DataFrame(flat)
        if not return_pval:
            df = df.drop(columns=["P_value", "Z_score"], errors="ignore")
        elif return_pval:
            df["P_adj"] = apply_bh_correction(df["P_value"])
        return df.sort_values("Q", ascending=False).reset_index(drop=True)

    def _compute_rstat_nufft(
        self,
        features_x: list[str] | None,
        features_y: list[str] | None,
        source: str,
        n_jobs: int,
        layer: str | None,
        return_pval: bool,
        chunk_size: int,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """NUFFT dispatch for :meth:`compute_rstat`.

        ``var_R`` comes from :meth:`NUFFTKernel.square_trace` (already
        n-point-scaled analytic default).
        """
        kernel = self.kernel_
        var_R = float(kernel.square_trace()) if return_pval else 0.0

        if features_x is None and features_y is not None:
            raise ValueError("Provide features_x when features_y is specified.")

        X_kept, names_kept, means_x, stds_x = self._prepare_features_nufft(
            source, features_x, layer
        )
        if features_y is None:
            X_y, names_y, means_y, stds_y = X_kept, names_kept, means_x, stds_x
        else:
            X_y, names_y, means_y, stds_y = self._prepare_features_nufft(source, features_y, layer)

        if len(names_kept) == 0 or len(names_y) == 0:
            return pd.DataFrame(
                columns=["Feature_1", "Feature_2", "R", "Z_score", "P_value", "P_adj"]
            )

        logger.info("Testing %d x %d feature pairs via NUFFT...", len(names_kept), len(names_y))

        # Densify + z-score the X block once; the NUFFT bipartite R path is
        # ``Xzᵀ · kernel.Kx(Yz)``. We bypass :func:`spatial_r_test` here so
        # we always get the full ``(M_x, M_y)`` cross matrix regardless of
        # whether ``M_x == M_y`` (the shape-based dispatch in
        # :func:`_r_test_nufft` would otherwise return a paired diagonal in
        # the coincident case).
        X_block = np.asarray(X_kept.todense(), dtype=np.float64)
        Xz = _standardize_features(X_block)

        results: list[dict[str, Any]] = []
        y_chunks = [slice(i, i + chunk_size) for i in range(0, len(names_y), chunk_size)]
        y_iter = (
            tqdm(y_chunks, desc=f"R (NUFFT, {self.kernel_method_})") if show_progress else y_chunks
        )
        for ysl in y_iter:
            Y_block = np.asarray(X_y[:, ysl].todense(), dtype=np.float64)
            Yz = _standardize_features(Y_block)
            KY = kernel.Kx(Yz)  # (n, M_y)
            R_chunk = Xz.T @ KY  # (M_x, M_y)
            for i, name_x in enumerate(names_kept):
                for j, name_y in enumerate(names_y[ysl]):
                    r = float(R_chunk[i, j])
                    row = {"Feature_1": name_x, "Feature_2": name_y, "R": r}
                    if return_pval:
                        z = r / np.sqrt(var_R) if var_R > 0 else 0.0
                        row["Z_score"] = z
                        row["P_value"] = float(2.0 * norm.sf(abs(z)))
                    results.append(row)
        # Silence "unused" warnings for means_x/_y/stds now that grid-space
        # standardization is done inside the NUFFT dispatch.
        del means_x, means_y, stds_x, stds_y

        df = pd.DataFrame(results)
        if return_pval:
            df["P_adj"] = apply_bh_correction(df["P_value"])
        return df.sort_values("R", ascending=False).reset_index(drop=True)
