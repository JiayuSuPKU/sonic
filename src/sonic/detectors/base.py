"""
Shared base class for single-sample spatial pattern detectors.

Concrete detectors follow a three-step workflow:

1. **Construction** — :meth:`Detector.__init__` takes kernel method + backend
   configs + kernel hyperparameters. No data is attached.
2. **Data setup** — :meth:`Detector.setup_data` takes the input container
   (:class:`anndata.AnnData` for :class:`DetectorIrregular`,
   :class:`spatialdata.SpatialData` for :class:`DetectorGrid`), performs
   preprocessing (feature filtering, coordinate / obsp extraction, or
   rasterization), and builds the kernel.
3. **Computation** — :meth:`Detector.compute_qstat` and
   :meth:`Detector.compute_rstat` take feature selections + compute-time knobs
   (``n_jobs``, ``chunk_size``, etc.) and return per-feature results.

The base class owns the attribute contract (``kernel_method_``,
``kernel_params_``, ``kernel_``, ``n``, ``_data_ready``) and enforces the
workflow via :meth:`_require_setup`. Concrete subclasses implement
:meth:`_merge_kernel_defaults`, :meth:`setup_data`, :meth:`compute_qstat`,
and :meth:`compute_rstat`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pandas as pd

from sonic.kernels import Kernel
from sonic.utils import _DEFAULT_CHUNK_BUDGET

if TYPE_CHECKING:
    from sonic.explanation import SpatialExplanation

__all__ = ["Detector"]


class Detector(ABC):
    r"""
    Abstract base for single-sample pattern detectors.

    Attributes
    ----------
    kernel_method\_ : str
        Kernel method name (e.g. ``'matern'``, ``'car'``). Set at construction.
    kernel_params\_ : dict
        Resolved kernel parameters after backend-specific defaults are merged
        with user overrides. Set at construction.
    kernel\_ : :class:`~sonic.kernels.Kernel` or None
        Kernel object built in :meth:`setup_data`. ``None`` before data setup.
    n : int or None
        Effective number of observations after preprocessing. ``None`` before
        data setup.
    """

    _available_kernels: tuple[str, ...] = (
        "gaussian",
        "matern",
        "moran",
        "graph_laplacian",
        "car",
    )

    def __init__(self, kernel_method: str, **kernel_params: Any) -> None:
        if kernel_method not in self._available_kernels:
            raise ValueError(
                f"kernel_method must be one of {self._available_kernels}, got {kernel_method!r}."
            )
        self.kernel_method_: str = kernel_method
        self.kernel_params_: dict = self._merge_kernel_defaults(kernel_method, dict(kernel_params))
        self.kernel_: Kernel | None = None
        self.n: int | None = None
        self._data_ready: bool = False

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def _merge_kernel_defaults(self, method: str, user_params: dict) -> dict:
        """Return ``{default: value}`` after merging user overrides.

        Subclasses must validate unknown keys and fill in backend-specific
        defaults (e.g. ``fft_solver`` / ``spacing`` for
        :class:`DetectorGrid`; ``k_neighbors`` vs ``neighbor_degree`` for
        the matrix vs NUFFT backends of :class:`DetectorIrregular`).
        """

    @abstractmethod
    def setup_data(self, data: Any, **kwargs: Any) -> Detector:
        """Attach ``data``, preprocess features, and build :attr:`kernel_`.

        Must set :attr:`kernel_`, :attr:`n`, and ``self._data_ready = True``
        before returning ``self``.
        """

    @abstractmethod
    def compute_qstat(self, features: list[str] | None = None, **kwargs: Any) -> pd.DataFrame:
        """Univariate Q-test across ``features``."""

    @abstractmethod
    def compute_rstat(self, **kwargs: Any) -> pd.DataFrame:
        """Bivariate R-test. Signature / feature selection are subclass-specific."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def explain_qstat(
        self,
        feature: str,
        *,
        source: str = "var",
        layer: str | None = None,
        memory_budget_bytes: int | str = _DEFAULT_CHUNK_BUDGET,
    ) -> SpatialExplanation:
        """Explain one feature's observed Q as per-observation contributions.

        Uses the same kernel configs and feature preparation as compute_qstat,
        but does not require existing Q/R computation.
        Returns a :class:`sonic.explanation.SpatialExplanation` with original
        values, kernel responses and contributions summing to Q.

        ``source`` ('var' or numeric 'obs') and ``layer`` follow DetectorIrregular;
        DetectorGrid uses its setup-time raster and accepts only source='var',
        layer=None. Categorical obs columns must first be encoded as individual
        numeric columns so a call identifies exactly one feature.
        ``memory_budget_bytes`` bounds estimated workspace and returned arrays.
        Only the requested feature is densified.
        """
        self._require_setup()
        if not isinstance(feature, str):
            raise TypeError("feature must be a single feature name.")
        return self._explain_statistic((feature,), "Q", source, layer, memory_budget_bytes)

    def explain_rstat(
        self,
        feature_x: str,
        feature_y: str,
        *,
        source: str = "var",
        layer: str | None = None,
        memory_budget_bytes: int | str = _DEFAULT_CHUNK_BUDGET,
    ) -> SpatialExplanation:
        """Explain one pair's observed R with a symmetric spatial allocation.

        Contributions are (z_x * Kz_y + z_y * Kz_x) / 2 and sum to R.
        Requires a symmetric real kernel; reconstruction is checked before
        returning. Self-pairs are allowed.
        Selection, memory and result conventions follow :meth:`explain_qstat`.
        """
        self._require_setup()
        if not isinstance(feature_x, str) or not isinstance(feature_y, str):
            raise TypeError("feature_x and feature_y must be single feature names.")
        return self._explain_statistic(
            (feature_x, feature_y), "R", source, layer, memory_budget_bytes
        )

    def _explain_statistic(self, features, statistic, source, layer, memory_budget_bytes):
        """Prepare backend-specific observations and apply the fitted operator."""
        raise NotImplementedError(f"{type(self).__name__} does not implement explanation maps.")

    def _require_setup(self) -> None:
        """Raise if :meth:`setup_data` has not been called."""
        if not self._data_ready or self.kernel_ is None:
            raise RuntimeError(f"{type(self).__name__}: call setup_data(...) before compute_*().")

    def __repr__(self) -> str:
        state = "ready" if self._data_ready else "not set up"
        return (
            f"<{type(self).__name__} kernel_method={self.kernel_method_!r} "
            f"state={state} n={self.n}>"
        )
