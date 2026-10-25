"""Shared spatial contribution maps, independent of detector/comparator containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sonic.utils import _chunk_parameters, _parse_memory_budget


@dataclass
class SpatialExplanation:
    """Observed values, operator responses and additive spatial contributions.

    ``values`` has shape (n_observations, n_features), ``filtered`` has shape
    (n_maps, n_observations, n_features), and ``contributions`` has shape
    (n_maps, n_observations). Detector Q/R explanations contain one map and
    one/two features:

    - ``filtered``: Lx for one feature, or (Lx, Ly) in feature order.
    - ``contributions``: elementwise x * Lx for one feature, or
      (x * Ly + y * Lx) / 2 for a pair.

    Here x and y denote test-prepared features (``values`` after preparation),
    and L is the fitted real symmetric kernel operator, including its scaling.

    ``observation_indices`` maps rows back to the original AnnData passed to
    setup_data, or to the C-order flattened raster for grid detectors.
    ``coordinates`` is optional: AnnData coordinates retain their source column
    order; grid coordinates are (y, x) on the kernel lattice. Units are not
    inferred. ``metadata`` describes the statistic, source and normalization.

    Scores are derived from contributions. Construction by SONIC checks these
    against independently evaluated statistics before discarding the reference.
    No kernels, Fourier coefficients, input containers or p-values are retained.
    """

    feature_names: tuple[str, ...]
    observation_indices: np.ndarray
    coordinates: np.ndarray | None
    values: np.ndarray
    filtered: np.ndarray
    contributions: np.ndarray
    metadata: dict[str, Any]

    @property
    def scores(self) -> np.ndarray:
        """Sum of spatial contributions for each map."""
        return self.contributions.sum(axis=1)


def _check_explanation_budget(kernel, n_features, memory_budget_bytes):
    """Check selected-feature workspace and returned arrays before loading data.

    Excludes resident input/kernel storage, lazy factorization and library
    overhead; this is an allocation estimate rather than an RSS limit.
    """
    budget = _parse_memory_budget(memory_budget_bytes)
    per_feature = _chunk_parameters(kernel)[1]
    estimate = 2 * n_features * per_feature + 8 * kernel.n * (3 * n_features + 5)
    if estimate > budget:
        raise MemoryError(
            f"Estimated explanation workspace {estimate} bytes exceeds budget {budget} bytes."
        )


def _explanation_from_responses(  # noqa: C901 — array boundary validation and reconstruction
    values,
    prepared,
    filtered,
    expected_scores,
    *,
    feature_names,
    observation_indices,
    coordinates=None,
    metadata,
    rtol=1e-9,
):
    """Build maps showing how each observation contributes to a score.

    The caller supplies the values used by the test and their spatially
    filtered values. This helper combines them into contribution maps and
    checks that each map sums to a separately computed score.

    Let x and y be columns of ``prepared``, and Lx and Ly the matching columns
    of ``filtered`` after applying the spatial kernel L. At each observation:

    - One feature contributes x * Lx.
    - A pair contributes (x * Ly + y * Lx) / 2.

    These products are elementwise. The pair formula requires a real symmetric
    kernel. All centering, scaling and filtering happen before this call;
    ``values`` holds the original observed values for display.

    Parameters
    ----------
    values : array_like, shape (n_observations, n_features)
        Original observed values to keep for display. Must be finite and real,
        with at least two observations and one or two feature columns.
    prepared : array_like, shape (n_observations, n_features)
        Values used by the test, after any required centering and scaling.
        Must be finite and real, with the same row and column order as values.
        Used to calculate contributions, but not stored in the result.
    filtered : array_like, shape (n_maps, n_observations, n_features)
        Results of applying each spatial kernel to prepared. Must be finite
        and real, in the same feature order: Lx for one feature, (Lx, Ly) for
        a pair. Include all kernel scaling; at least one map is required.
    expected_scores : array_like, shape (n_maps,)
        Separately computed score for each map. Must be finite and real.
        Used only to check that the contributions sum to the expected scores.
    feature_names : sequence of str
        Names of the one or two features, in column order. Two columns use
        the pair formula, even when both names refer to the same feature.
    observation_indices : array_like of int, shape (n_observations,)
        Nonnegative indices locating these observations in the source data:
        original AnnData rows or raster cells flattened in C order.
    coordinates : array_like, shape (n_observations, 2), optional
        Finite spatial coordinates in row order, or None if unavailable.
        Column order and units are kept as supplied.
    metadata : dict
        Descriptions of the statistic, source and normalization. The dictionary
        is copied, but objects inside it are shared. No specific keys are required.
    rtol : float, default 1e-9
        Finite, nonnegative tolerance for the score check. Each map's allowed
        absolute error is rtol times the larger of the absolute expected score
        and the sum of absolute contributions, plus np.finfo(float).tiny.

    Returns
    -------
    SpatialExplanation
        Contribution maps whose sums match the expected scores within tolerance,
        with copies of the original values, filtered values, source indices and
        optional coordinates.
    """
    if any(np.iscomplexobj(a) for a in (values, prepared, filtered, expected_scores)):
        raise ValueError("Quadratic contribution allocation requires real values and responses.")
    values = np.asarray(values, dtype=float)
    prepared = np.asarray(prepared, dtype=float)
    filtered = np.asarray(filtered, dtype=float)
    expected = np.asarray(expected_scores, dtype=float)
    indices = np.asarray(observation_indices)
    names = tuple(feature_names)
    if values.ndim != 2 or values.shape[1] not in (1, 2) or values.shape[0] < 2:
        raise ValueError(
            "Explanation values must contain at least two observations and one/two features."
        )
    n, g = values.shape
    if prepared.shape != values.shape or len(names) != g:
        raise ValueError("Prepared values and feature names must match explanation values.")
    if filtered.ndim != 3 or filtered.shape[1:] != (n, g) or filtered.shape[0] == 0:
        raise ValueError("Filtered fields must have shape (n_maps, n_observations, n_features).")
    if expected.shape != (filtered.shape[0],):
        raise ValueError("Provide one independent reference score per map.")
    if indices.shape != (n,) or not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0):
        raise ValueError("Observation indices must be nonnegative integers matching the rows.")
    if not all(np.isfinite(a).all() for a in (values, prepared, filtered, expected)):
        raise ValueError("Explanation values, responses and reference scores must be finite.")
    if coordinates is not None:
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape != (n, 2) or not np.isfinite(coordinates).all():
            raise ValueError(
                "Explanation coordinates must be finite with shape (n_observations, 2)."
            )
    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError("Reconstruction tolerance must be finite and nonnegative.")

    if g == 1:
        contributions = prepared[:, 0] * filtered[:, :, 0]
    else:
        contributions = 0.5 * (
            prepared[:, 0] * filtered[:, :, 1] + prepared[:, 1] * filtered[:, :, 0]
        )
    scale = np.maximum(np.abs(expected), np.abs(contributions).sum(axis=1))
    error = np.abs(contributions.sum(axis=1) - expected)
    if not np.isfinite(contributions).all() or np.any(error > rtol * scale + np.finfo(float).tiny):
        raise ArithmeticError(
            "Spatial contributions do not reconstruct the reference statistic. "
            "R allocations require a symmetric kernel; check operator accuracy and support."
        )
    return SpatialExplanation(
        feature_names=names,
        observation_indices=indices.copy(),
        coordinates=None if coordinates is None else coordinates.copy(),
        values=values.copy(),
        filtered=filtered.copy(),
        contributions=contributions,
        metadata=dict(metadata),
    )
