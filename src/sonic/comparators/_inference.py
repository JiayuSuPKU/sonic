"""Shared L2 covariance estimation and null calibration for spectral comparisons."""

from __future__ import annotations

import itertools
import logging
import math
import warnings
from fractions import Fraction
from functools import cache, partial
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import bisect, brentq
from scipy.special import betainc
from scipy.stats import norm

import sonic._davies as _davies
from sonic.comparators import _regression as _lm
from sonic.comparators.normalization import _normalize_shape_apply
from sonic.statistics import liu_sf

logger = logging.getLogger(__name__)
_L2_NULL_OPTIONS = ("permutation", "plugin", "rotation")
_PLUGIN_MIN_DF_NO_WARN = 3

# Bound scratch space for eight float64 arrays; at least one feature is processed.
# Inputs, returned diagnostics, and the bin covariance are outside this target.
_COMPARISON_WORKSPACE_BYTES = 32 * (1 << 20)

# Profile preparation and workspace


def _iter_feature_blocks(n_samples, n_features, n_bins, n_terms=0):
    """Yield feature slices without materializing whole-panel profile/residual arrays."""
    step = max(1, _COMPARISON_WORKSPACE_BYTES // (8 * 8 * max(n_samples, n_terms, 1) * n_bins))
    for start in range(0, n_features, step):
        yield slice(start, min(start + step, n_features))


def _prepare_profile_block(profiles, normalize_shape, *, log_transform=True):
    """Make an owned float64 profile block, logarithmic by default."""
    if normalize_shape:
        block = _normalize_shape_apply(profiles)
    else:
        block = np.array(profiles, dtype=np.float64, copy=True)
        if not np.isfinite(block).all():
            raise ValueError("Tests require finite profiles for observed samples.")
    if log_transform:
        np.maximum(block, 1e-12, out=block)
        np.log(block, out=block)
    return block


def _profile_roundoff(*blocks, per_gene=False):
    """Estimate numerical resolution from the eligible transformed responses."""
    if per_gene:
        scale = np.maximum.reduce([np.max(np.abs(block), axis=(0, 2)) for block in blocks])
        return 32 * np.finfo(float).eps * np.maximum(scale, np.finfo(float).tiny)
    scale = max(float(np.max(np.abs(block), initial=0.0)) for block in blocks)
    return 32 * np.finfo(float).eps * max(scale, np.finfo(float).tiny)


def _iter_mask_groups(masks):
    """Yield a row mask and feature indices for columns with identical masks."""
    if masks.shape[1] == 0:
        return
    _, inverse = np.unique(np.packbits(masks, axis=0).T, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    for features in np.split(order, np.flatnonzero(np.diff(inverse[order])) + 1):
        yield masks[:, features[0]], features


def _resolve_freq_weights(freq_weights: np.ndarray | None, n_bins: int) -> np.ndarray:
    """Return strictly positive bin weights normalized to sum to one."""
    if freq_weights is None:
        return np.full(n_bins, 1.0 / n_bins)
    w = np.asarray(freq_weights, dtype=float).ravel()
    if w.shape != (n_bins,):
        raise ValueError(f"freq_weights must have length n_bins={n_bins}, got shape {w.shape}.")
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("freq_weights must be finite and strictly positive.")
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("freq_weights must have a positive finite sum.")
    weights = w / total
    if np.any(weights == 0):
        raise ValueError("freq_weights must be strictly positive after normalization.")
    return weights


# Gaussian plug-in calibration with estimated covariance


class _PluginNullMetadata(TypedDict, total=False):
    """Optional metadata, using the TypedDict syntax supported by Python 3.10."""

    n_obs_A: int | np.ndarray
    n_obs_B: int | np.ndarray
    design_columns: list[str]
    contrast_vector: np.ndarray
    n_obs: int | np.ndarray
    zero_covariance: np.ndarray
    estimate_per_bin: np.ndarray


class _PluginNullState(_PluginNullMetadata):
    """Internal contract shared by plug-in DF tests and covariance diagnostics.

    The common fields below are returned by every plug-in null estimator and
    by ``Comparator.estimate_null_covariance``. They are intentionally a little
    richer than the minimum needed for p-values. The selected covariance,
    profile-scale roundoff and sample counts also identify contrasts whose
    noise null cannot be estimated; diagnostics expose the weights, effective
    rank and residual degrees of freedom.

    Common fields
    -------------
    covariance
        Effective covariance estimator: ``"pooled"`` or ``"per_gene"``.
    mode
        ``"two_group"`` for binary labels or ``"glm"`` for design-matrix
        contrasts.
    masked
        Whether gene-specific missingness was used.
    observed
        Per-feature L2 statistic on the tested profile scale, shape
        ``(n_features,)``. Masked helpers use ``NaN`` for ineligible genes.
    beta
        Fitted profile coefficients, shape ``(n_terms, n_features, n_bins)``.
        For binary groups, the two rows are the group-A mean and B-minus-A
        difference, with groups ordered by ``np.unique``. Masked paths keep
        ``NaN`` coefficients for ineligible genes.
    sigma_log
        Pooled residual covariance, shape ``(n_bins, n_bins)``. Auto-spectra
        and coherence normally use the log scale, while signed coherency uses
        the original scale. The legacy name is retained for diagnostics.
        Per-gene diagnostics return shape ``(n_features, n_bins, n_bins)``;
        internal tests omit these matrices (``None``) to bound memory.
    roundoff
        Numerical tolerance from eligible transformed profiles in positive-weight
        bins, used to identify a numerically zero residual covariance. Per-gene
        mode uses a separate tolerance for each feature.
    freq_weights
        Normalized positive bin weights, shape ``(n_bins,)``.
    weighted_cov
        ``W^1/2 sigma_log W^1/2``, the covariance whose eigenstructure
        controls the weighted quadratic form. Its shape/storage follows ``sigma_log``.
    eigenvalues
        Liu mixture eigenvalues after applying the contrast variance scale.
        Pooled unmasked helpers return shape ``(n_bins,)``; masked or per-gene
        helpers return ``(n_genes, n_bins)`` with ``NaN`` rows for skipped genes.
    effective_rank
        Participation-ratio rank of the unscaled ``weighted_cov`` eigenvalues;
        a per-feature vector for per-gene covariance.
    contrast_scale
        Scalar contrast variance for unmasked paths; per-gene vector for
        masked paths.
    df_resid
        Residual degrees of freedom, scalar for unmasked paths and per-gene
        vector for masked paths.
    eligible
        Boolean per-feature mask identifying eligible fits; degenerate nonzero
        contrasts can still have NaN p-values.

    zero_covariance
        Per-gene-mode indicator of numerically zero residual variance in all
        positive-weight bins. Used even when full matrices are not stored.

    Two-group-only metadata
    -----------------------
    n_obs_A, n_obs_B
        Sample counts in the two arms. Unmasked paths return scalars; masked
        paths return per-gene vectors.

    GLM-only metadata
    -----------------
    design_columns
        Column names for the resolved design matrix.
    contrast_vector
        Numeric contrast vector aligned to ``design_columns``.
    n_obs
        Number of observed samples. Unmasked paths return a scalar; masked
        paths return a per-gene vector.
    """

    covariance: Literal["pooled", "per_gene"]
    mode: Literal["two_group", "glm"]
    masked: bool
    observed: np.ndarray
    beta: np.ndarray
    sigma_log: np.ndarray | None
    roundoff: float | np.ndarray
    freq_weights: np.ndarray
    # Intermediate cross-spectrum fits defer these fields until after pooling.
    weighted_cov: np.ndarray | None
    eigenvalues: np.ndarray | None
    effective_rank: float | np.ndarray | None
    contrast_scale: float | np.ndarray
    df_resid: int | np.ndarray
    eligible: np.ndarray


def _resolve_covariance(covariance, *, statistic="log_l2", null="plugin"):
    """Validate the covariance option and select the plug-in L2 estimator."""
    if covariance not in (None, "pooled", "per_gene"):
        raise ValueError("covariance must be None, 'pooled', or 'per_gene'.")
    if covariance is not None and (statistic != "log_l2" or null != "plugin"):
        raise ValueError("covariance is only supported with statistic='log_l2', null='plugin'.")
    return "per_gene" if covariance is None else covariance


def _maybe_warn_small_df_plugin(df_resid: int, *, log_transform: bool = True) -> None:
    """Warn the user when running Gaussian plug-in calibration at very small residual df.

    Suppress with ``warnings.filterwarnings('ignore',
    message="log_l2 [+] null='plugin'")`` if you accept the calibration risk.
    """
    if df_resid < _PLUGIN_MIN_DF_NO_WARN:
        rel_noise_pct = 100.0 * (2.0 / max(df_resid, 1)) ** 0.5
        warnings.warn(
            f"{'log_l2' if log_transform else 'l2'} + null='plugin' at residual df={df_resid}: "
            f"Variance estimator σ̂² has ~{rel_noise_pct:.0f}% relative noise, "
            f"so Gaussian plug-in calibration may be anti-conservative. "
            "Check calibration with binary-label permutations when samples are exchangeable.",
            UserWarning,
            stacklevel=3,
        )


def _effective_rank_from_eigenvalues(eigenvalues: np.ndarray) -> float:
    """Participation-ratio rank from covariance eigenvalues."""
    eig = np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)
    total = float(eig.sum())
    if total <= 0.0:
        return float("nan")
    return float((total * total) / np.sum(eig * eig))


def _weighted_covariance_eigensystem(covariance, weights):
    """Return weighted covariance, nonnegative eigenvalues and effective rank."""
    sqrt_weights = np.sqrt(weights)
    weighted_cov = sqrt_weights[:, None] * covariance * sqrt_weights[None, :]
    eigenvalues = np.maximum(np.linalg.eigvalsh(weighted_cov), 0.0)
    return weighted_cov, eigenvalues, _effective_rank_from_eigenvalues(eigenvalues)


def _pooled_covariance_state(covariance, weights, contrast_scale, roundoff, for_pooling):
    """Defer the eigensystem when this covariance contributes to a larger pool."""
    weighted = eigenvalues = rank = None
    if not for_pooling:
        weighted, eigenvalues, rank = _weighted_covariance_eigensystem(covariance, weights)
        eigenvalues = np.asarray(contrast_scale)[..., None] * eigenvalues
    return {
        "sigma_log": covariance,
        "roundoff": roundoff,
        "weighted_cov": weighted,
        "eigenvalues": eigenvalues,
        "effective_rank": rank,
    }


def _init_per_gene_covariance(n_features, n_bins, store_covariance):
    """Retain eigenvalues for tests; full covariance arrays only for diagnostics."""
    return {
        "sigma_log": np.full((n_features, n_bins, n_bins), np.nan) if store_covariance else None,
        "weighted_cov": np.full((n_features, n_bins, n_bins), np.nan) if store_covariance else None,
        "eigenvalues": np.full((n_features, n_bins), np.nan),
        "effective_rank": np.full(n_features, np.nan),
        "roundoff": np.full(n_features, np.nan),
        "zero_covariance": np.zeros(n_features, dtype=bool),
    }


def _store_per_gene_covariance(state, indices, residuals, weights, df_resid, scale, roundoff):
    """Calibrate residual blocks without pooling or inverting singular covariances."""
    genes = range(indices.start, indices.stop) if isinstance(indices, slice) else indices
    n_bins = residuals.shape[-1]
    sqrt_weights = np.sqrt(weights)
    for gene, residual, tolerance in zip(
        genes, residuals.transpose(1, 0, 2), roundoff, strict=True
    ):
        weighted_residual = residual * sqrt_weights
        # The smaller row Gram matrix has the same nonzero mixture eigenvalues.
        gram = (
            weighted_residual @ weighted_residual.T
            if residual.shape[0] < n_bins
            else weighted_residual.T @ weighted_residual
        ) / df_resid
        eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.0)
        state["eigenvalues"][gene] = np.pad(eigenvalues * scale, (n_bins - len(eigenvalues), 0))
        state["effective_rank"][gene] = _effective_rank_from_eigenvalues(eigenvalues)
        state["roundoff"][gene] = tolerance
        variance = np.sum(residual**2, axis=0) / df_resid
        state["zero_covariance"][gene] = np.max(variance) <= tolerance**2
        if state["sigma_log"] is not None:
            covariance = residual.T @ residual / df_resid
            state["sigma_log"][gene] = covariance
            state["weighted_cov"][gene] = sqrt_weights[:, None] * covariance * sqrt_weights


def _estimate_two_group_null_covariance(
    spectra: np.ndarray,
    groups: np.ndarray,
    *,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
    log_transform: bool = True,
    min_samples_per_group: int = 2,
    covariance: str | None = None,
    store_covariance: bool = True,
    for_pooling: bool = False,
) -> _PluginNullState:
    """Fit the unmasked binary null as an intercept-plus-indicator GLM.

    Two group case-control comparison is a special case of the GLM.
    We delegate the actual estimation to ``_estimate_glm_null_covariance``,
    but keep a separate validation rule and metadata.
    ``log_transform=False`` supports signed cross-spectrum profiles.
    """
    covariance = _resolve_covariance(covariance)
    spectra = np.asarray(spectra)
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D (n_samples, n_genes, n_bins), got {spectra.shape}.")
    n_samples = spectra.shape[0]
    groups = np.asarray(groups)
    if groups.shape != (n_samples,):
        raise ValueError(f"groups shape {groups.shape} does not match n_samples={n_samples}.")
    uniq = np.unique(groups)
    if uniq.size != 2:
        raise ValueError(f"groups must contain exactly two distinct values, got {uniq}.")
    indicator = (groups == uniq[1]).astype(float)
    n_b = int(indicator.sum())
    n_a = n_samples - n_b
    if min(n_a, n_b) < min_samples_per_group:
        raise ValueError(f"Each group requires at least {min_samples_per_group} observed samples.")

    state = _estimate_glm_null_covariance(
        spectra,
        np.column_stack([np.ones(n_samples), indicator]),
        np.array([0.0, 1.0]),
        freq_weights=freq_weights,
        normalize_shape=normalize_shape,
        log_transform=log_transform,
        covariance=covariance,
        store_covariance=store_covariance,
        for_pooling=for_pooling,
    )
    for key in ("design_columns", "contrast_vector", "n_obs"):
        del state[key]
    state.update(mode="two_group", n_obs_A=n_a, n_obs_B=n_b)
    return state


def _estimate_two_group_masked_null_covariance(
    spectra: np.ndarray,
    groups: np.ndarray,
    presence: np.ndarray,
    *,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
    log_transform: bool = True,
    allow_empty: bool = False,
    min_samples_per_group: int = 2,
    covariance: str | None = None,
    store_covariance: bool = True,
    for_pooling: bool = False,
) -> _PluginNullState:
    """Estimate a masked two-group null through the shared GLM estimator.

    Two group case-control comparison is a special case of the GLM.
    We delegate the actual estimation to ``_estimate_glm_masked_null_covariance``,
    but keep a separate validation rule that requires each feature to have
    at least ``min_samples_per_group`` unmasked observations in both groups.
    Skipped features retain their sample counts and NaN test results.
    Raises if no features qualify, unless ``allow_empty=True``.
    """
    covariance = _resolve_covariance(covariance)
    if int(min_samples_per_group) < 2:
        raise ValueError(f"min_samples_per_group must be >= 2, got {min_samples_per_group}.")
    spectra = np.asarray(spectra)
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D, got {spectra.shape}.")
    n_samples, n_genes, _ = spectra.shape
    presence = np.asarray(presence, dtype=bool)
    if presence.shape != (n_samples, n_genes):
        raise ValueError(
            f"presence shape {presence.shape} != (n_samples, n_genes) = ({n_samples}, {n_genes})."
        )
    groups = np.asarray(groups)
    if groups.shape != (n_samples,):
        raise ValueError(f"groups shape {groups.shape} does not match n_samples={n_samples}.")
    uniq = np.unique(groups)
    if uniq.size != 2:
        raise ValueError("groups must contain exactly two distinct values.")
    group_b = groups == uniq[1]
    n_obs_a = presence[~group_b].sum(axis=0)
    n_obs_b = presence[group_b].sum(axis=0)
    eligible = (n_obs_a >= min_samples_per_group) & (n_obs_b >= min_samples_per_group)

    state = _estimate_glm_masked_null_covariance(
        spectra,
        np.column_stack([np.ones(n_samples), group_b]),
        np.array([0.0, 1.0]),
        # Exclude threshold failures even when the GLM contrast would be estimable.
        presence & eligible[None, :],
        freq_weights=freq_weights,
        normalize_shape=normalize_shape,
        log_transform=log_transform,
        allow_empty=True,
        covariance=covariance,
        store_covariance=store_covariance,
        for_pooling=for_pooling,
    )
    if not state["eligible"].any() and not allow_empty:
        raise ValueError(
            "estimate_null_covariance: no genes meet "
            f"min_samples_per_group={min_samples_per_group} per arm. "
            "Cannot estimate the null covariance."
        )
    for key in ("design_columns", "contrast_vector", "n_obs"):
        del state[key]
    state.update(
        mode="two_group",
        n_obs_A=n_obs_a,
        n_obs_B=n_obs_b,
        # Preserve the historical residual-df floor for skipped features.
        df_resid=np.maximum(n_obs_a + n_obs_b - 2, 1),
    )
    return state


def _estimate_glm_null_covariance(
    spectra: np.ndarray,
    design: pd.DataFrame | np.ndarray,
    contrast: str | dict[str, float] | np.ndarray,
    *,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
    log_transform: bool = True,
    covariance: str | None = None,
    store_covariance: bool = True,
    for_pooling: bool = False,
) -> _PluginNullState:
    """Compute plug-in null state for unmasked GLM ``log_l2`` tests.

    The design matrix is fit once against all flattened ``(gene, bin)``
    log-spectrum responses. The returned state includes the per-gene contrast
    effect converted to log-L2 ``observed`` values, the per-gene or pooled
    residual covariance, and the
    contrast variance scale ``c'(X'X)^+c`` applied to the Liu eigenvalues.

    Set ``log_transform=False`` for signed cross-spectrum profiles; in that
    case ``sigma_log`` contains covariance on the original profile scale.
    """
    covariance = _resolve_covariance(covariance)
    spectra = np.asarray(spectra)
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D (n_samples, n_genes, n_bins), got {spectra.shape}.")
    prepared_design = _lm._prepare_glm_design(design, contrast, spectra.shape[0])
    return _fit_glm_profiles(
        spectra,
        prepared_design,
        freq_weights=freq_weights,
        normalize_shape=normalize_shape,
        log_transform=log_transform,
        covariance=covariance,
        store_covariance=store_covariance,
        for_pooling=for_pooling,
    )


def _fit_glm_profiles(
    profiles,
    prepared_design,
    *,
    freq_weights=None,
    normalize_shape=False,
    log_transform=True,
    covariance: str | None = None,
    store_covariance: bool = True,
    for_pooling: bool = False,
) -> _PluginNullState:
    """Build the unmasked L2 null state from 3D spectral profiles.

    Parameters
    ----------
    profiles : np.ndarray
        Shape ``(n_samples, n_features, n_bins)``, with samples in the same order
        as the prepared design matrix.
    prepared_design : tuple
        Validated output of ``_prepare_glm_design``: design matrix, column names,
        contrast vector, design pseudoinverse, and rank.
    freq_weights : np.ndarray, optional
        Positive bin weights of shape ``(n_bins,)``, normalized to sum to one.
        Defaults to equal weights.
    normalize_shape : bool, default False
        Normalize each sample/feature profile to sum to one before transformation.
    log_transform : bool, default True
        Apply ``log(max(profile, 1e-12))`` after optional shape normalization.
        Set to False for signed profiles.
    covariance : {None, 'pooled', 'per_gene'}, optional
        Pool residual covariance across features, or estimate it per feature.
        None selects per-gene covariance.
    store_covariance : bool, default True
        Retain full covariance matrices in per-gene mode; otherwise their state
        entries are None. Pooled covariance is always retained.
    for_pooling : bool, default False
        Return pooled covariance for later combination
        across support groups. Defer the eigensystem until that pool is complete.

    Returns
    -------
    state : _PluginNullState
        Weighted L2 distances, signed per-bin contrasts, residual covariance diagnostics,
        contrast-scaled eigenvalues for Liu calibration, and design metadata.
        Fitted ``beta`` coefficients have shape ``(n_terms, n_features, n_bins)``.
        P-values are computed separately from this state.

    Notes
    -----
    Each feature contributes ``n_samples - rank`` residual
    degrees of freedom; pooled covariance combines residual cross-products and
    degrees of freedom across features. Weighted covariance eigenvalues are
    scaled by ``sum((contrast @ pseudoinverse)**2)`` for Liu calibration.
    """
    covariance = _resolve_covariance(covariance)
    n_samples, n_features, n_bins = profiles.shape
    design_matrix, design_columns, contrast_vector, design_inverse, rank = prepared_design
    n_terms = design_matrix.shape[1]
    df_resid = n_samples - rank
    weights = _resolve_freq_weights(freq_weights, n_bins)
    per_gene = covariance == "per_gene"
    covariance_state = (
        _init_per_gene_covariance(n_features, n_bins, store_covariance) if per_gene else {}
    )
    observed = np.empty(n_features)
    estimates = np.empty((n_features, n_bins))
    beta = np.empty((n_terms, n_features, n_bins))
    residual_cov = None if per_gene else np.zeros((n_bins, n_bins))
    roundoff = 0.0
    for block in _iter_feature_blocks(n_samples, n_features, n_bins, n_terms):
        # Transform feature blocks and flatten their feature/bin columns for GLM fitting.
        response = _prepare_profile_block(
            profiles[:, block], normalize_shape, log_transform=log_transform
        ).reshape(n_samples, -1)
        # Compute global or per-feature numerical roundoff for numerically zero residual covariance.
        roundoff = max(roundoff, _profile_roundoff(response.reshape(n_samples, -1, n_bins)))
        tolerance = (
            _profile_roundoff(response.reshape(n_samples, -1, n_bins), per_gene=True)
            if per_gene
            else None
        )
        beta_block, residuals, theta = _lm._fit_glm_response(
            response, design_matrix, design_inverse, contrast_vector, rank
        )
        beta[:, block] = beta_block.reshape(n_terms, -1, n_bins)
        effect = theta.reshape(-1, n_bins)
        observed[block] = np.sqrt(np.sum(weights * effect**2, axis=-1))
        estimates[block] = effect
        if per_gene:
            # Update per-gene covariance with the current block's residuals.
            _store_per_gene_covariance(
                covariance_state,
                block,
                residuals.reshape(n_samples, -1, n_bins),
                weights,
                df_resid,
                float(np.sum((contrast_vector @ design_inverse) ** 2)),
                tolerance,
            )
        else:
            residuals = residuals.reshape(-1, n_bins)
            residual_cov += residuals.T @ residuals
    if not per_gene:
        residual_cov /= n_features * df_resid
    _maybe_warn_small_df_plugin(df_resid, log_transform=log_transform)

    # Calibrate the L2 statistic on the tested profile scale.
    contrast_scale = float(np.sum((contrast_vector @ design_inverse) ** 2))
    if not per_gene:
        covariance_state = _pooled_covariance_state(
            residual_cov, weights, contrast_scale, roundoff, for_pooling
        )

    return {
        "mode": "glm",
        "masked": False,
        "observed": observed,
        "estimate_per_bin": estimates,
        "covariance": covariance,
        "freq_weights": weights,
        **covariance_state,
        "contrast_scale": contrast_scale,
        "df_resid": df_resid,
        "design_columns": design_columns,
        "contrast_vector": contrast_vector,
        "beta": beta,
        "n_obs": n_samples,
        "eligible": np.ones(n_features, dtype=bool),
    }


def _estimate_glm_masked_null_covariance(  # noqa: C901
    spectra: np.ndarray,
    design: pd.DataFrame | np.ndarray,
    contrast: str | dict[str, float] | np.ndarray,
    presence: np.ndarray,
    *,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
    log_transform: bool = True,
    allow_empty: bool = False,
    min_resid_df: int = 1,
    covariance: str | None = None,
    store_covariance: bool = True,
    for_pooling: bool = False,
) -> _PluginNullState:
    """Compute plug-in null state for masked GLM ``log_l2`` tests.

    A separate OLS fit is performed for each gene's observed samples. Eligible
    genes must have enough residual degrees of freedom, an estimable contrast,
    and a positive finite contrast variance. By default each gene has its own
    residual covariance. Both modes return per-gene ``observed`` and contrast-scaled
    ``eigenvalues`` because missingness changes the design.
    ``log_transform=False`` fits signed profiles without the log floor.
    ``allow_empty=True`` returns an ineligible state for an empty covariance
    pool, allowing cross-spectrum callers to report other support groups.
    With ``for_pooling=True``, return covariance only on tested bins and defer
    its eigensystem as in ``_fit_glm_profiles``.
    """
    covariance = _resolve_covariance(covariance)
    if int(min_resid_df) < 1:
        raise ValueError(f"min_resid_df must be >= 1, got {min_resid_df}.")
    spectra = np.asarray(spectra)
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D (n_samples, n_genes, n_bins), got {spectra.shape}.")
    n_samples, n_genes, n_bins = spectra.shape
    presence = np.asarray(presence, dtype=bool)
    if presence.shape != (n_samples, n_genes):
        raise ValueError(
            f"presence shape {presence.shape} != (n_samples, n_genes) = ({n_samples}, {n_genes})."
        )

    # Resolve the design matrix and contrast vector.
    design_matrix, design_columns = _lm._build_design_matrix(design, n_samples)
    contrast_vector = _lm._resolve_contrast(contrast, design_columns)

    n_terms = design_matrix.shape[1]
    if contrast_vector.shape != (n_terms,):
        raise ValueError(f"contrast length {contrast_vector.shape} != design cols ({n_terms},).")

    # Fit each gene on its observed samples.
    weights = _resolve_freq_weights(freq_weights, n_bins)
    per_gene = covariance == "per_gene"
    covariance_state = (
        _init_per_gene_covariance(n_genes, n_bins, store_covariance) if per_gene else {}
    )
    n_obs = np.zeros(n_genes, dtype=int)
    df_resid = np.zeros(n_genes, dtype=int)
    observed = np.full(n_genes, np.nan, dtype=float)
    estimates = np.full((n_genes, n_bins), np.nan)
    contrast_scale = np.full(n_genes, np.nan, dtype=float)
    beta = np.full((n_terms, n_genes, n_bins), np.nan, dtype=float)
    eligible = np.zeros(n_genes, dtype=bool)
    sigma_acc = None if per_gene else np.zeros((n_bins, n_bins))
    pooled_df = 0
    roundoff = 0.0

    for sample_mask, genes in _iter_mask_groups(presence):
        count = int(sample_mask.sum())
        n_obs[genes] = count
        if count == 0:
            continue

        gene_design = design_matrix[sample_mask]
        design_inverse, rank_gene, estimable = _lm._decompose_glm_design(
            gene_design, contrast_vector
        )
        df_gene = count - rank_gene
        df_resid[genes] = df_gene
        if df_gene < int(min_resid_df):
            continue
        if not estimable:
            continue

        scale = float(np.sum((contrast_vector @ design_inverse) ** 2))
        if not np.isfinite(scale) or scale <= 0.0:
            continue

        rows = np.flatnonzero(sample_mask)
        for block in _iter_feature_blocks(count, len(genes), n_bins, n_terms):
            cols = genes[block]
            response = _prepare_profile_block(
                spectra[rows[:, None], cols], normalize_shape, log_transform=log_transform
            )
            roundoff = max(roundoff, _profile_roundoff(response))
            tolerance = _profile_roundoff(response, per_gene=True) if per_gene else None
            gene_beta, residuals, theta = _lm._fit_glm_response(
                response.reshape(count, -1), gene_design, design_inverse, contrast_vector, rank_gene
            )
            effect = theta.reshape(-1, n_bins)
            observed[cols] = np.sqrt(np.sum(weights * effect**2, axis=-1))
            estimates[cols] = effect
            beta[:, cols] = gene_beta.reshape(n_terms, -1, n_bins)
            if per_gene:
                _store_per_gene_covariance(
                    covariance_state,
                    cols,
                    residuals.reshape(count, -1, n_bins),
                    weights,
                    df_gene,
                    scale,
                    tolerance,
                )
            else:
                residuals = residuals.reshape(-1, n_bins)
                sigma_acc += residuals.T @ residuals
        pooled_df += df_gene * len(genes)
        contrast_scale[genes] = scale
        eligible[genes] = True

    if pooled_df == 0 and not allow_empty:
        raise ValueError(
            "estimate_null_covariance: no genes have enough observed "
            "samples and an estimable contrast to estimate the null covariance."
        )

    # Calibrate the L2 statistic on the tested profile scale.
    if eligible.any():
        _maybe_warn_small_df_plugin(int(np.median(df_resid[eligible])), log_transform=log_transform)
    if not per_gene:
        residual_cov = sigma_acc / max(pooled_df, 1)
        covariance_state = _pooled_covariance_state(
            residual_cov, weights, contrast_scale, roundoff, for_pooling
        )

    return {
        "mode": "glm",
        "masked": True,
        "observed": observed,
        "estimate_per_bin": estimates,
        "covariance": covariance,
        "freq_weights": weights,
        **covariance_state,
        "contrast_scale": contrast_scale,
        "df_resid": df_resid,
        "design_columns": design_columns,
        "contrast_vector": contrast_vector,
        "beta": beta,
        "n_obs": n_obs,
        "eligible": eligible,
    }


def _l2_pvalues_from_state(state: _PluginNullState) -> np.ndarray:
    """Evaluate Liu-tail p-values from the plug-in null state.

    The state must already contain the observed L2 statistic and the
    contrast-scaled Liu eigenvalues produced by one of the
    ``_estimate_*_null_covariance`` helpers. Pooled unmasked states carry one
    eigenvalue vector; masked or per-gene states carry an eigenvalue matrix.
    The ``eligible`` mask keeps skipped genes at ``NaN``. The observed distance
    is squared for the mixture-chi-square tail. Only pooled-covariance genes
    with the same contrast scale share a Liu fit. With numerically zero
    residual covariance, zero effects return p=1 and nonzero effects return NaN.
    """
    observed = np.asarray(state["observed"], dtype=float)
    eigenvalues = np.asarray(state["eigenvalues"], dtype=float)
    roundoff = state["roundoff"]
    if state.get("covariance", "pooled") == "per_gene":
        pvals = np.full(observed.shape, np.nan)
        n_observed = state["n_obs"] if "n_obs" in state else state["n_obs_A"] + state["n_obs_B"]
        zero_effect = observed <= roundoff * np.sqrt(state["contrast_scale"] * n_observed)
        pvals[state["eligible"] & state["zero_covariance"] & zero_effect] = 1.0
        for gene in np.flatnonzero(state["eligible"] & ~state["zero_covariance"]):
            pvals[gene] = liu_sf(observed[gene] ** 2, eigenvalues[gene])
        return pvals
    variances = np.diag(state["sigma_log"])
    if np.max(variances, where=state["freq_weights"] > 0, initial=0.0) <= roundoff**2:
        n_observed = state["n_obs"] if "n_obs" in state else state["n_obs_A"] + state["n_obs_B"]
        zero = observed <= roundoff * np.sqrt(state["contrast_scale"] * n_observed)
        return np.where(state["eligible"] & zero, 1.0, np.nan)
    if eigenvalues.ndim == 1:
        return np.asarray(liu_sf(observed**2, eigenvalues), dtype=float)

    pvals = np.full(observed.shape, np.nan, dtype=float)
    indices = np.flatnonzero(state["eligible"])
    if indices.size:
        scales = np.asarray(state["contrast_scale"])[indices]
        order = np.argsort(scales)
        boundaries = np.flatnonzero(np.diff(scales[order])) + 1
        for group in np.split(indices[order], boundaries):
            pvals[group] = liu_sf(observed[group] ** 2, eigenvalues[group[0]])
    return pvals


# Binary label permutations


def _make_label_permutations(
    groups: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    *,
    max_exact_permutations: int = 10000,
) -> tuple[np.ndarray, bool]:
    """Build a null-distribution set of group relabellings.

    For small samples the total number of distinct two-group label
    assignments (``C(n, n_a)``) can be tiny compared to the user's
    requested ``n_perm``. In that regime an **exact** enumeration
    of every possible relabelling is both cheaper and strictly more
    accurate (zero Monte-Carlo noise, sharp p-values).

    Parameters
    ----------
    groups : np.ndarray
        Observed group labels, length ``n_samples`` with exactly two
        unique values.
    n_perm : int
        Number of random shuffles to produce when exact enumeration is
        infeasible. Ignored on the exact path.
    rng : np.random.Generator
        RNG for the sampling fallback.
    max_exact_permutations : int, default 10000
        If ``C(n_samples, n_a)`` is at most this, every distinct relabelling
        is enumerated (``is_exact=True``) and ``n_perm`` is overridden to
        the enumeration count. Otherwise ``n_perm`` random shuffles of
        ``groups`` are returned (``is_exact=False``).

    Returns
    -------
    perm_labels : np.ndarray
        ``(n_used, n_samples)`` array; each row is a valid relabelling
        (same ``n_a`` / ``n_b`` marginals as ``groups``).
    is_exact : bool
        True if every row is a distinct relabelling and together they
        span every possible partition; False if the rows are independent
        random shuffles.
    """
    groups = np.asarray(groups)
    n_samples = len(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    if uniq.size != 2:
        raise ValueError(f"groups must have exactly two unique values, got {uniq}.")
    n_a = int(counts[0])
    total = int(math.comb(n_samples, n_a))
    if total <= max_exact_permutations:
        perm_labels = np.empty((total, n_samples), dtype=groups.dtype)
        a_val, b_val = uniq[0], uniq[1]
        for i, subset in enumerate(itertools.combinations(range(n_samples), n_a)):
            perm_labels[i] = b_val
            perm_labels[i, list(subset)] = a_val
        return perm_labels, True
    perm_labels = np.empty((n_perm, n_samples), dtype=groups.dtype)
    base = groups.copy()
    for i in range(n_perm):
        rng.shuffle(base)
        perm_labels[i] = base
    return perm_labels, False


def _run_l2_permutations(
    profiles: np.ndarray,
    group_codes: np.ndarray,
    perm_labels: np.ndarray,
    *,
    freq_weights: np.ndarray,
    normalize_shape: bool = False,
    is_exact: bool = False,
    log_transform: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return L2 distances, label-permutation tails, and observed A-minus-B bin contrasts.

    ``perm_labels`` is a ``(n_perm_used, n_samples)`` matrix of group
    relabellings (as produced by :func:`_make_label_permutations`).
    Profiles and weights are prepared once; ``log_transform=False`` keeps
    signed profiles on their original scale. Only per-feature exceedance
    counts are retained; no permutations-by-features null matrix is allocated.
    Exhaustive enumeration uses the exact tail; sampled nulls get +1 correction.
    """
    uniq = np.unique(group_codes)
    a_val = uniq[0]
    response = _prepare_profile_block(profiles, normalize_shape, log_transform=log_transform)
    response -= response[:1].copy()
    weights = freq_weights

    def difference(values, group_a):
        return values[group_a].mean(axis=0) - values[~group_a].mean(axis=0)

    group_a = group_codes == a_val
    estimates = difference(response, group_a)
    observed = np.sqrt(np.sum(weights * estimates**2, axis=-1))
    # Comparing two L2 values rounds four group means and two weighted sums over bins.
    bin_scale = np.max(np.abs(response), axis=0)
    scale = np.sqrt(np.sum(weights * bin_scale**2, axis=-1))
    tie_tol = (4 * len(response) + 2 * len(weights)) * np.finfo(float).eps * scale
    threshold = observed - tie_tol
    exceedances = np.zeros(profiles.shape[1], dtype=np.int64)
    for labels in perm_labels:
        effect = difference(response, labels == a_val)
        exceedances += np.sqrt(np.sum(weights * effect**2, axis=-1)) >= threshold
    correction = 0.0 if is_exact else 1.0
    pvals = (exceedances + correction) / (len(perm_labels) + correction)
    return observed, pvals, estimates


def _run_two_group_permutations(
    profiles,
    groups,
    presence,
    *,
    freq_weights,
    normalize_shape,
    min_samples_per_group,
    n_perm,
    max_exact_permutations,
    random_state,
):
    """Run binary permutations, retaining feature order for masked RNG draws."""
    profiles = np.asarray(profiles, dtype=np.float64)
    freq_weights = _resolve_freq_weights(freq_weights, profiles.shape[-1])
    rng = np.random.default_rng(random_state)
    if presence is None:
        labels, exact = _make_label_permutations(
            groups, n_perm, rng, max_exact_permutations=max_exact_permutations
        )
        if exact:
            logger.info(
                "Exact permutation test: enumerated %d distinct relabellings (C(%d, %d)).",
                len(labels),
                len(groups),
                int((groups == 0).sum()),
            )
        return _run_l2_permutations(
            profiles,
            groups,
            labels,
            freq_weights=freq_weights,
            is_exact=exact,
            normalize_shape=normalize_shape,
        )

    observed = np.full(profiles.shape[1], np.nan)
    pvals = np.full(profiles.shape[1], np.nan)
    estimates = np.full(profiles.shape[1:], np.nan)
    for gene in range(profiles.shape[1]):
        mask = presence[:, gene]
        sub_groups = groups[mask]
        if min(np.sum(sub_groups == 0), np.sum(sub_groups == 1)) < min_samples_per_group:
            continue
        labels, exact = _make_label_permutations(
            sub_groups, n_perm, rng, max_exact_permutations=max_exact_permutations
        )
        statistic, tail, effect = _run_l2_permutations(
            profiles[mask, gene : gene + 1],
            sub_groups,
            labels,
            freq_weights=freq_weights,
            is_exact=exact,
            normalize_shape=normalize_shape,
        )
        observed[gene], pvals[gene], estimates[gene] = statistic[0], tail[0], effect[0]
    return observed, pvals, estimates


# Conditional rotation


def _rotation_metadata(n_rotations, random_state):
    """Validate rotation controls and retain the effective seed for replay."""
    if (
        isinstance(n_rotations, (bool, np.bool_))
        or not isinstance(n_rotations, (int, np.integer))
        or n_rotations < 1
    ):
        raise ValueError("n_rotations must be a positive integer.")
    seed = np.random.SeedSequence(random_state).entropy
    return {
        "null": "rotation",
        "covariance": None,
        "n_rotations": int(n_rotations),
        "random_state": int(seed),
        "min_p_value": 1.0 / (int(n_rotations) + 1),
    }


_as_fractions = np.frompyfunc(Fraction, 1, 1)


def _orthogonalize_columns(columns, n_samples):
    """Rational Gram-Schmidt in place, using the sample rows for inner products."""
    samples = columns[:n_samples]
    for j in range(columns.shape[1] - 1):
        column = samples[:, j]
        squared = column @ column
        if squared:
            slopes = (column @ samples[:, j + 1 :]) / squared
            columns[:, j + 1 :] -= columns[:, j, None] * slopes


def _exact_rotation_design(design, contrast, retained, residual_basis):
    """Orthogonalize the retained model exactly for refitting roundoff-sized residuals."""
    augmented = _as_fractions(np.vstack([design, contrast]))
    if len(retained) < design.shape[1]:
        # Restrict coefficients to the same right-singular subspace as the OLS fit.
        augmented = augmented @ _as_fractions(retained.T)

    # Apply the design column operations to the contrast as well. Rational
    # Gram-Schmidt needs no square roots or tolerance-based pivot decisions.
    _orthogonalize_columns(augmented, len(design))
    q = augmented[:-1]
    inverse = q.T / np.sum(q * q, axis=0)[:, None]
    sample_weights = augmented[-1] @ inverse
    norm = math.hypot(*np.asarray(sample_weights, dtype=float))
    # Complete the well-conditioned design basis along the original rotation axes.
    # The original SVD complement can lose residual energy for nearly collinear designs.
    columns = np.asarray(q / np.max(np.abs(q), axis=0), dtype=float)
    basis, r = np.linalg.qr(np.column_stack([columns, residual_basis]))
    residual_axes = basis[:, len(retained) :] * np.copysign(1.0, np.diag(r)[len(retained) :])
    return q, inverse, sample_weights, Fraction(norm), residual_axes


def _rank_one_rotation(effect, residuals):
    """Check exact proportionality to a nonzero contrast across the active bins."""
    if len(effect) == 1:
        return True
    nonzero = np.flatnonzero(effect)
    if not nonzero.size:
        return False
    k = nonzero[0]
    return np.array_equal(residuals * effect[k], residuals[:, k, None] * effect)


def _refine_rotation_fit(profiles, model, sqrt_weights):
    """Recheck a nearly perfect fit when its residuals are as small as rounding error.

    Recompute the per-bin contrast and residuals using exact arithmetic on the
    transformed profile. Scale their weighted values together before converting
    to floats to preserve their relative sizes. Return the contrast, the rotation
    coordinates, and whether a rank-one tail applies. Raise if rounding erases
    the contribution of a nonzero contrast or residual.
    """
    q, inverse, sample_weights, norm, residual_basis = model
    y = _as_fractions(profiles)
    effect = sample_weights @ y
    residuals = y - q @ (inverse @ y)
    rank_one = _rank_one_rotation(effect, residuals)
    weighted = np.vstack([effect / norm, residuals]) * sqrt_weights
    scale = np.max(np.abs(weighted))
    scaled = np.asarray(weighted / scale if scale else weighted, dtype=float)
    coordinates = np.vstack([scaled[0], residual_basis.T @ scaled[1:]])
    if (effect.any() and not np.any(coordinates[0] ** 2)) or (
        residuals.any() and not np.any(coordinates[1:] ** 2)
    ):
        raise RuntimeError(
            "Conditional scatter cannot resolve a nonzero contrast or residual energy."
        )
    return np.asarray(effect, dtype=float), coordinates, rank_one


def _rotation_inertia(matrix):
    """Count negative and zero eigenvalues by exact symmetric elimination."""
    negative = zero = 0
    while matrix.size:
        pivot = matrix[0, 0]
        if pivot:
            negative += pivot < 0
            matrix = matrix[1:, 1:] - np.outer(matrix[1:, 0], matrix[0, 1:] / pivot)
        elif matrix[0].any():
            # A congruence preserves inertia. Combine two axes to get a nonzero pivot.
            j = np.flatnonzero(matrix[0])[0]
            sign = 1 if matrix[j, j] + 2 * matrix[0, j] else -1
            matrix = matrix.copy()
            matrix[:, 0] += sign * matrix[:, j]
            matrix[0, :] += sign * matrix[j, :]
        else:
            zero += 1
            matrix = matrix[1:, 1:]
    return negative, zero


def _refine_rotation_shift(
    feature, gap, *, profiles, weights, exact_design, near_rank_one, shift_tolerance
):
    """Recheck apparent rank-one fits and ties that roundoff can create or hide.

    Use the transformed profiles and original bin-weight ratios to check whether
    the contrast is zero (p=1) or the adjusted bins vary along one direction
    (beta tail). If a scatter eigenvalue is close to the observed statistic,
    check which eigenvalues lie above, below, or exactly at that value. Return
    the resulting tail coefficients, or ``None`` when the computed spectrum suffices.
    """
    ambiguous_shift = gap <= shift_tolerance[feature]
    if not (near_rank_one[feature] or ambiguous_shift):
        return None
    q, inverse, a, _, residual_basis = exact_design()
    y = _as_fractions(profiles[:, feature])
    effect = a @ y
    if not effect.any():
        return np.zeros(1)  # Every rotation attains at least the observed zero.
    contrast_energy = a @ a
    observed = (effect * weights) @ effect / contrast_energy
    residuals = y - q @ (inverse @ y)
    dimension = residual_basis.shape[1] + 1
    if _rank_one_rotation(effect, residuals):
        coefficients = np.full(dimension, -observed, dtype=object)
        coefficients[0] = np.sum(residuals * residuals, axis=0) @ weights
        scaled = np.asarray(coefficients / max(np.abs(coefficients)), dtype=float)
        if np.any((coefficients != 0) & (scaled == 0)):
            raise RuntimeError("Conditional tail cannot resolve the coefficient dynamic range.")
        return scaled
    if not ambiguous_shift:
        return None
    if y.shape[1] < dimension:
        # Outside the profile span the scatter is zero, so every shift is -observed.
        basis = residuals + np.outer(a, effect / contrast_energy)
        _orthogonalize_columns(basis, len(y))
        basis = basis[:, np.any(basis, axis=0)]
    else:
        residual = _as_fractions(residual_basis)
        basis = np.column_stack([a, residual - q @ (inverse @ residual)])
    basis /= np.max(np.abs(basis), axis=0)
    projected = basis.T @ y
    metric = basis.T @ basis
    shifted = (projected * weights) @ projected.T - observed * metric
    return _verified_shift_spectrum(shifted, metric, dimension - len(metric), observed)


def _refine_shift_eigenvalue(shifted, scaled_metric, vector, index, bound, count):
    """Certify the Rayleigh estimate with exact counts, or bisect for a clustered root."""
    # Error in the Rayleigh quotient is second order in the eigenvector error.
    v = _as_fractions(vector)
    estimate = float((v @ shifted @ v) / (v @ scaled_metric @ v))
    left, right = np.nextafter(estimate, [-np.inf, np.inf])
    if estimate and count(left) <= index < count(right):
        return estimate
    value = bisect(
        lambda x: 1 if count(x) > index else -1,
        -bound,
        bound,
        xtol=np.nextafter(0.0, 1.0),
        maxiter=1200,
    )
    if value == 0:
        raise RuntimeError("Conditional tail cannot resolve a nonzero shifted eigenvalue.")
    return value


def _verified_shift_spectrum(shifted, metric, outside, observed):
    """Solve the exact shifted pencil, retaining zero shifts and omitted sample directions."""
    if not shifted.any():
        return np.array([-1.0 if outside else 0.0])

    scale = np.max(np.abs(shifted))
    coefficients, vectors = eigh(
        np.asarray(shifted / scale, dtype=float), np.asarray(metric, dtype=float)
    )
    # A backward-stable eigensolve can lose the sign of a tiny shift. Inertia
    # identifies exact zeros and brackets each remaining ambiguous eigenvalue.
    bound = 2 * np.max(np.abs(coefficients))
    small = np.flatnonzero(np.abs(coefficients) <= np.sqrt(np.finfo(float).eps) * bound)
    if small.size:
        negative, zero = _rotation_inertia(shifted)
        if negative + zero == len(metric):
            return np.array([-1.0])
        if negative == 0 and outside == 0:
            return np.array([1.0])
        scaled_metric = metric * scale

        @cache
        def count(value):
            # The positive definite metric preserves the generalized eigenvalue count.
            return sum(_rotation_inertia(shifted - Fraction(value) * scaled_metric))

        coefficients[negative : negative + zero] = 0.0
        nonzero = (small < negative) | (small >= negative + zero)
        for i in small[nonzero]:
            coefficients[i] = _refine_shift_eigenvalue(
                shifted, scaled_metric, vectors[:, i], i, bound, count
            )
    if outside:
        common = max(scale, observed)
        inside = coefficients * float(scale / common)
        omitted = -float(observed / common)
        if not omitted or np.any((coefficients != 0) & (inside == 0)):
            raise RuntimeError("Conditional tail cannot resolve the coefficient dynamic range.")
        return np.r_[inside, np.full(outside, omitted)]
    return coefficients


def _conditional_scatter(coordinates):
    """Form the small sample-space scatter without dropping null directions.

    Parameters
    ----------
    coordinates : np.ndarray
        Weighted coordinates, shape ``(residual_df + 1, n_features, n_bins)``.
        The first row is the contrast divided by its sample-weight norm;
        the remaining rows lie in the full-model residual basis.

    Returns
    -------
    scatter : np.ndarray
        Shape ``(n_features, residual_df + 1, residual_df + 1)``. Each feature
        is rescaled before squaring; its observed quadratic form is
        ``scatter[:, 0, 0]`` on the same scale. Zero eigenvalue directions
        remain part of the conditional sphere, even when there are fewer bins.
    effective_rank : np.ndarray
        Squared trace divided by squared Frobenius norm; NaN for zero scatter.
    """
    # Per-feature rescaling leaves the tail unchanged and avoids squaring tiny values.
    scale = np.max(np.abs(coordinates), axis=(0, 2))
    coordinates = coordinates / np.where(scale > 0, scale, 1)[None, :, None]
    rows = coordinates.transpose(1, 0, 2)
    gram = rows @ rows.swapaxes(1, 2)
    trace = np.trace(gram, axis1=1, axis2=2)
    squared = np.sum(gram**2, axis=(1, 2))
    effective_rank = np.divide(
        trace**2, squared, out=np.full(len(trace), np.nan), where=squared > 0
    )
    return gram, effective_rank


def _validate_integration_tol(integration_tol):
    """Validate the absolute error target for conditional integration."""
    if (
        isinstance(integration_tol, (bool, np.bool_))
        or not isinstance(integration_tol, (int, float, np.integer, np.floating))
        or not np.isfinite(integration_tol)
        or not 0 < integration_tol < 1
    ):
        raise ValueError("integration_tol must be a finite number strictly between 0 and 1.")


def _saddlepoint_tail(coefficients):
    """Approximate a signed quadratic-form tail with Kuonen's modified normal law.

    Solve ``K'(t)=0`` for ``K(t)=-sum(log(1-2*t*a))/2`` and evaluate the
    modified signed root. Near the mean use its continuous limit. This is an
    approximation, with no numerical bound on its distributional error.
    """
    a = np.asarray(coefficients, dtype=float)
    variance = 2 * np.sum(a * a)
    mean = a.sum()
    if abs(mean) <= 1e-6 * np.sqrt(variance):
        z = -mean / np.sqrt(variance) + 8 * np.sum(a**3) / (6 * variance**1.5)
    else:
        left = np.nextafter(0.5 / a.min(), 0.0)
        right = np.nextafter(0.5 / a.max(), 0.0)
        with np.errstate(divide="ignore", invalid="raise", over="raise"):
            t = brentq(
                lambda t: np.sum(a / (1 - 2 * t * a)),
                left,
                right,
                xtol=np.nextafter(0.0, 1.0),
                rtol=1e-14,
                maxiter=200,
            )
            x = 2 * t * a
            # Avoid cancellation of linear terms near the mean.
            k = (
                t * mean + 0.5 * sum(np.sum(x**order) / order for order in range(2, 9))
                if np.max(np.abs(x)) < 0.01
                else -0.5 * np.log1p(-x).sum()
            )
            w = np.sign(t) * np.sqrt(-2 * k)
            # Form t*sqrt(K''(t)) without squaring tiny terms before multiplying by t.
            v = np.sign(t) * np.sqrt(0.5 * np.sum((x / (1 - x)) ** 2))
            z = w + np.log(v / w) / w
    pvalue = float(norm.sf(z))
    if not np.isfinite(pvalue) or not 0 < pvalue <= 1:
        raise RuntimeError("Saddlepoint tail is nonfinite or below floating-point resolution.")
    return pvalue


def _two_level_tail(positive, negative, n_positive, n_negative):
    """Tail for two signed levels whose magnitudes have a common scale."""
    total = positive + negative
    p = betainc(n_negative / 2, n_positive / 2, positive / total)
    if p > 0.5:
        # The positive ratio can round to one while its complement is still resolved.
        p = 1 - betainc(n_positive / 2, n_negative / 2, negative / total)
    return float(p)


def _conditional_tail(coefficients, *, integration_tol=1e-6):  # noqa: C901 — beta bounds, one retry and marked fallback
    """Evaluate ``P(sum(coefficients * Z**2) >= 0)`` for independent normal Z.

    Parameters
    ----------
    coefficients : np.ndarray
        Finite nonempty vector ``eigenvalues(A) - observed``. Include every
        sample-space direction: a zero eigenvalue of A contributes ``-observed``.
    integration_tol : float
        Initial Davies absolute error target, default 1e-6. Accept an interior
        tail only with ``ifault=0`` and error budget at most ``0.001*p``.
        A valid but insufficiently resolved tail gets one retry at
        ``min(integration_tol, 1e-4*p)``, with 10,000 terms per attempt.

    Returns
    -------
    pvalue, error, method : tuple
        Tail, numerical error budget/estimate, and ``'davies'``, ``'saddlepoint'``,
        ``'beta'`` or ``'exact'``. Davies reports its accepted requested accuracy;
        beta includes the uncertainty from its probability bounds and roundoff.
        Exact boundaries report zero.
        Unresolved Davies calls use marked saddlepoint with NaN error: the
        integration target does not bound approximation error. If this also
        fails, raise RuntimeError instead of returning a numerical zero.

    Notes
    -----
    Davies' AS 155 uses signed coefficients. Scale them before integration;
    never merge distinct levels or discard zero eigenvalue directions of A.
    Two beta laws bound the tail by replacing each signed coefficient with
    the smallest or largest of its sign. Equal levels give the exact beta law.
    """
    _validate_integration_tol(integration_tol)
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1 or not coefficients.size or not np.isfinite(coefficients).all():
        raise ValueError("Conditional coefficients must be a finite nonempty vector.")
    if np.all(coefficients >= 0):
        return 1.0, 0.0, "exact"
    if np.all(coefficients <= 0):
        return 0.0, 0.0, "exact"

    nonzero = coefficients[coefficients != 0]
    scaled = nonzero / np.max(np.abs(nonzero))
    if np.any(scaled == 0):
        raise RuntimeError("Conditional tail cannot resolve the coefficient dynamic range.")
    eps = np.finfo(float).eps
    pos, neg = scaled[scaled > 0], -scaled[scaled < 0]
    p_min, p_max = pos.min(), pos.max()
    n_min, n_max = neg.min(), neg.max()
    two_levels = p_min == p_max and n_min == n_max
    lower = _two_level_tail(p_min, n_max, len(pos), len(neg))
    upper = lower if two_levels else _two_level_tail(p_max, n_min, len(pos), len(neg))
    pvalue = lower + (upper - lower) / 2
    roundoff = max(32 * eps * len(nonzero) * upper, np.nextafter(0.0, 1.0))
    half_width = max(abs(pvalue - lower), abs(upper - pvalue))
    error = half_width + roundoff
    if 0 < pvalue <= 1 and error <= min(integration_tol, 0.001 * pvalue):
        return pvalue, error, "beta"
    # At the roundoff floor, rounded eigenvalues can compare equal in just
    # some sample orders. Apply the same resolution check to either case.
    if two_levels or (pvalue > 0 and half_width <= roundoff):
        raise RuntimeError(
            "Conditional beta tail underflowed or fails absolute/relative resolution."
        )

    accuracy = float(integration_tol)
    for attempt in range(2):
        pvalue, fault = _davies.sf(scaled, accuracy, 10000)
        valid = fault == 0 and np.isfinite(pvalue) and 0 < pvalue <= 1
        error = max(accuracy, 8 * eps)
        if valid and error <= integration_tol and error <= 0.001 * pvalue:
            return pvalue, error, "davies"
        if attempt or not valid or accuracy <= 0.001 * pvalue:
            break
        accuracy = min(accuracy, 1e-4 * pvalue)
    try:
        return _saddlepoint_tail(scaled), np.nan, "saddlepoint"
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        raise RuntimeError(
            f"Conditional tail unresolved by Davies (ifault={fault}) and saddlepoint: {exc}"
        ) from exc


def _integration_metadata(integration_tol):
    """Describe the shared deterministic tail policy, including empty comparisons."""
    _validate_integration_tol(integration_tol)
    return {
        "null": "rotation",
        "covariance": None,
        "method": "davies",
        "integration_tol": float(integration_tol),
        "integration_limit": 10000,
        "n_saddlepoint": 0,
    }


def _conditional_tails(scatter, *, integration_tol=1e-6, rank_one=False, refine=None):
    """Evaluate prepared sphere tails by Davies integration with marked saddlepoint fallback.

    Parameters
    ----------
    scatter : np.ndarray
        Symmetric matrices from :func:`_conditional_scatter`, shape
        ``(n_features, residual_df + 1, residual_df + 1)``. Each observed
        quadratic form is the first diagonal entry.
    integration_tol : float
        Absolute p-value error target; see :func:`_conditional_tail` for acceptance
        checks and explicit resolution failures.
    rank_one : bool or array of bool
        True for each scatter known to have rank at most one. Use exact
        beta coefficients from diagonal energies, without eigendecomposition.
    refine : callable, optional
        Called as ``refine(feature, gap)``, where ``gap`` is the distance from the
        observed statistic to the nearest scatter eigenvalue. It can recheck a
        possible rank-one fit or tie against the profiles. A returned array
        replaces the computed tail coefficients. ``None`` keeps them.

    Returns
    -------
    pvalues, errors, methods : tuple of np.ndarray
        One tail, numerical error budget/estimate and method per feature. Zero eigenvalue
        directions are retained; no covariance moments or sphere draws are used.
    """
    scatter = np.asarray(scatter, dtype=float)
    if (
        scatter.ndim != 3
        or scatter.shape[1] == 0
        or scatter.shape[1] != scatter.shape[2]
        or not np.isfinite(scatter).all()
    ):
        raise ValueError("Conditional scatter must contain finite square matrices.")
    pvalues, errors = np.empty(len(scatter)), np.empty(len(scatter))
    methods = np.empty(len(scatter), dtype=object)
    rank_one = np.broadcast_to(rank_one, (len(scatter),))
    for feature, matrix in enumerate(scatter):
        try:
            if rank_one[feature]:
                coefficients = np.full(len(matrix), -matrix[0, 0])
                # lambda - q is residual energy; avoid subtracting nearly equal totals.
                coefficients[0] = np.diag(matrix)[1:].sum()
            else:
                # Shift before diagonalization to preserve nearly constant spectral differences.
                shifted = matrix.copy()
                shifted.flat[:: len(matrix) + 1] -= matrix[0, 0]
                scale = np.max(np.abs(shifted))
                if scale > 0:
                    shifted /= scale
                coefficients = np.linalg.eigvalsh(shifted)
                verified = None
                if refine is not None and matrix.any():
                    verified = refine(feature, np.min(np.abs(coefficients)) * scale)
                if verified is not None:
                    coefficients = verified
                elif coefficients.max() <= 0 and np.any(shifted[0, 1:] != 0):
                    raise RuntimeError("The upper spectral boundary is numerically unresolved.")
            pvalues[feature], errors[feature], methods[feature] = _conditional_tail(
                coefficients, integration_tol=integration_tol
            )
        except (RuntimeError, np.linalg.LinAlgError) as exc:
            raise RuntimeError(
                f"Conditional integration failed for block feature {feature}: {exc}"
            ) from exc
    return pvalues, errors, methods


def _rotation_tails(scatter, basis, n_rotations, seed):
    """Stream Monte Carlo tails for a prepared conditional scatter.

    Parameters
    ----------
    scatter : np.ndarray
        Shape ``(n_features, residual_df + 1, residual_df + 1)`` from
        :func:`_conditional_scatter`; the observed value is ``scatter[:, 0, 0]``.
    basis : np.ndarray
        Orthonormal sample-space basis aligned with the scatter axes.
    n_rotations : int
        Number of Monte Carlo draws.
    seed : np.random.SeedSequence
        Shared seed for this observation mask, reused across feature blocks.

    Returns
    -------
    np.ndarray
        Plus-one upper-tail p-values, one per feature.
    """
    gram = scatter
    observed = gram[:, 0, 0]
    trace = np.trace(gram, axis1=1, axis2=2)
    if not trace.any():
        return np.ones(len(trace))
    tolerance = 64 * np.finfo(float).eps * trace
    counts = np.zeros(len(trace), dtype=np.int64)
    n_samples, dimension = basis.shape
    draw_block = max(
        1, _COMPARISON_WORKSPACE_BYTES // (32 * (n_samples + dimension + dimension**2 + len(trace)))
    )
    rng = np.random.default_rng(seed)
    for start in range(0, n_rotations, draw_block):
        size = min(draw_block, n_rotations - start)
        # Project sample-space Gaussian draws so basis changes do not change the draws.
        directions = rng.standard_normal((size, n_samples)) @ basis
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        products = (directions[:, :, None] * directions[:, None, :]).reshape(size, -1)
        null = products @ gram.reshape(len(trace), -1).T
        counts += np.count_nonzero(null >= observed - tolerance, axis=0)
    return (counts + 1) / (n_rotations + 1)


def _run_l2_conditional(  # noqa: C901
    profiles,
    design,
    contrast,
    presence=None,
    *,
    freq_weights=None,
    normalize_shape=False,
    log_transform=True,
    min_resid_df=1,
    allow_empty=False,
    integration_tol=1e-6,
    n_rotations=None,
    random_state=None,
):
    """Prepare and evaluate each feature's conditional L2 null in bounded blocks.

    Parameters
    ----------
    profiles : np.ndarray
        Shape ``(n_samples, n_features, n_bins)``, transformed block by block.
    design, contrast
        Design and scalar contrast accepted by ``compare_glm``.
    presence : np.ndarray or None
        Optional sample-by-feature observation mask. False cells are never read.
    freq_weights : array-like, optional
        Positive bin weights, normalized to sum to one and applied after fitting
        to the observed L2 statistic and conditional scatter. None gives equal weights.
    normalize_shape : bool
        If True, scale each nonzero sample-feature profile to unit total across all
        bins before fitting.
    log_transform : bool
        If True, fit ``log(max(profile, 1e-12))`` after any shape normalization.
    min_resid_df : int
        Minimum residual degrees of freedom per observed design.
    allow_empty : bool
        Return skipped rows when no features qualify, for cross-spectrum callers.
    integration_tol : float
        Absolute p-value error target for deterministic integration, the default.
        See :func:`_conditional_tail` for numerical acceptance checks.
    n_rotations : int or None
        If supplied, use this many Monte Carlo draws instead of integration.
        Retained for internal numerical reference checks.
    random_state : int or None
        Monte Carlo seed; ignored by integration. The effective seed is saved
        in metadata for replay.

    Returns
    -------
    dict
        Statistics (weighted L2 distances), signed bin estimates, p-values,
        numerical errors, eligibility, counts, residual df
        and conditional effective ranks in original feature order. Ineligible
        tails remain NaN. Metadata records the integration method/tolerance or
        Monte Carlo seed/draw count/floor; Monte Carlo numerical errors are NaN.

    Notes
    -----
    With ``a = contrast @ pinv(X)``, B contains ``a / ||a||`` and the
    full-model residual basis. For transformed profiles Y and diagonal weights W,
    form ``A = B.T @ Y @ W @ Y.T @ B``. Compare ``L2**2 / (a @ a)`` with
    ``u.T @ A @ u`` for uniform sphere directions u; this preserves the
    nuisance-adjusted cross-bin scatter ``Y.T @ B @ B.T @ Y`` without forming
    that bin-by-bin matrix. The evaluator receives a rescaled A with all
    ``residual_df + 1`` axes retained, including zero eigenvalue directions.
    """
    if n_rotations is None:
        metadata = _integration_metadata(integration_tol)
    else:
        metadata = _rotation_metadata(n_rotations, random_state)
    if int(min_resid_df) < 1:
        raise ValueError(f"min_resid_df must be >= 1, got {min_resid_df}.")
    profiles = np.asarray(profiles)
    if profiles.ndim != 3 or profiles.shape[0] == 0 or profiles.shape[2] == 0:
        raise ValueError("Rotation profiles must be 3D with samples and frequency bins.")
    n_samples, n_features, n_bins = profiles.shape
    masked = presence is not None
    presence = (
        np.ones((n_samples, n_features), bool)
        if presence is None
        else np.asarray(presence, dtype=bool)
    )
    if presence.shape != (n_samples, n_features):
        raise ValueError("presence must have shape (n_samples, n_features).")
    matrix, columns = _lm._build_design_matrix(design, n_samples)
    vector = _lm._resolve_contrast(contrast, columns)
    inverse, rank, estimable = _lm._decompose_glm_design(matrix, vector)
    if not masked:
        if n_samples - rank < int(min_resid_df):
            raise ValueError("Rotation tests require positive residual degrees of freedom.")
        if not estimable:
            raise ValueError("contrast is not estimable from the supplied design matrix.")
    weights = _resolve_freq_weights(freq_weights, n_bins)
    sqrt_weights = np.sqrt(weights)
    exact_sqrt_weights = _as_fractions(sqrt_weights)
    # A common scale cancels from the tail. Rounding each normalized weight can break a tie.
    raw_weights = (
        weights if freq_weights is None else np.asarray(freq_weights, dtype=object).ravel()
    )
    shift_weights = np.array(
        [
            Fraction(int(w)) if isinstance(w, (int, np.integer)) else Fraction(float(w))
            for w in raw_weights
        ]
    )
    observed = np.full(n_features, np.nan)
    estimates = np.full((n_features, n_bins), np.nan)
    pvals = np.full(n_features, np.nan)
    pvalue_error = np.full(n_features, np.nan)
    pvalue_method = np.full(n_features, None, dtype=object)
    effective_rank = np.full(n_features, np.nan)
    eligible = np.zeros(n_features, bool)
    df_resid = np.zeros(n_features, int)
    n_obs = presence.sum(axis=0)
    # Matching observation masks share the fitted design and conditional basis.
    for sample_mask, genes in _iter_mask_groups(presence):
        rows = np.flatnonzero(sample_mask)
        count = len(rows)
        if not count:
            continue
        x = matrix[rows]
        inverse, rank, estimable = _lm._decompose_glm_design(x, vector)
        df = count - rank
        df_resid[genes] = df
        if df < int(min_resid_df) or not estimable:
            continue
        # a = c^T X^+ gives sample weights for the fitted contrast: theta = a @ Y.
        a = vector @ inverse
        contrast_norm = np.linalg.norm(a)
        if not np.isfinite(contrast_norm) or contrast_norm <= 0:
            continue
        # Unit contrast + residual axes span the sample space after nuisance removal.
        u, singular, vt = np.linalg.svd(x, full_matrices=True)
        residual_basis = u[:, rank:]
        basis = np.column_stack([a / contrast_norm, residual_basis])
        exact_design = cache(partial(_exact_rotation_design, x, vector, vt[:rank], residual_basis))
        column_scale = np.max(np.abs(x), axis=0)
        constant_nuisance = rank == x.shape[1] and np.any(
            np.all(x == x[0], axis=0) & (x[0] != 0) & (vector == 0)
        )
        if n_rotations is not None:
            # Reuse this mask's seed across feature blocks to preserve seeded results.
            seed = np.random.SeedSequence(
                metadata["random_state"], spawn_key=(n_samples, *map(int, rows))
            )
        # Bound workspace for both profile arrays and the (df + 1)-dimensional scatter.
        for block in _iter_feature_blocks(count, len(genes), max(n_bins, df + 1), matrix.shape[1]):
            cols = genes[block]
            values = profiles[rows[:, None], cols]
            response = _prepare_profile_block(
                values, normalize_shape, log_transform=log_transform
            ).reshape(count, -1)
            beta, residuals, theta = _lm._fit_glm_response(response, x, inverse, vector, rank)
            effect = theta.reshape(-1, n_bins)
            # Dividing by ||a|| makes the contrast axis unit length; it is not an SE.
            coordinates = np.concatenate(
                [theta[None, :] / contrast_norm, residual_basis.T @ residuals]
            )
            # C = B^T Y W^(1/2): one contrast row and df orthogonal residual rows.
            coordinates = coordinates.reshape(df + 1, len(cols), n_bins) * sqrt_weights
            # Opposing fitted terms can dwarf both the response and its residual.
            fitted_scale = (column_scale @ np.abs(beta)).reshape(1, len(cols), n_bins)
            profiles_block = response.reshape(count, len(cols), n_bins)
            roundoff = _profile_roundoff(
                profiles_block * sqrt_weights, fitted_scale * sqrt_weights, per_gene=True
            ) * np.sqrt(count)
            residual_scale = np.max(np.abs(coordinates[1:]), axis=(0, 2))
            small_residual = residual_scale <= roundoff
            rank_one = np.full(len(cols), n_bins == 1)
            for j in np.flatnonzero(small_residual):
                tested = profiles_block[:, j]
                if constant_nuisance and np.all(tested == tested[0]):
                    effect[j] = 0.0
                    coordinates[:, j] = 0.0
                    continue
                # Refit to distinguish a genuinely small residual from roundoff in an exact fit.
                effect[j], coordinates[:, j], rank_one[j] = _refine_rotation_fit(
                    tested, exact_design(), exact_sqrt_weights
                )
            estimates[cols] = effect
            observed[cols] = np.sqrt(np.sum(weights * effect**2, axis=-1))
            scatter, effective_rank[cols] = _conditional_scatter(coordinates)
            if n_rotations is None:
                # Allow for fitting and projection error when screening the scatter.
                eps = np.finfo(float).eps
                fit_error = np.divide(
                    roundoff, residual_scale, out=np.zeros_like(roundoff), where=~small_residual
                )
                projection_error = 32 * eps * singular[0] / singular[rank - 1]
                tolerance = (
                    np.sqrt(32 * eps * (df + 1)) + projection_error + fit_error * np.sqrt(n_bins)
                )
                trace = np.trace(scatter, axis1=1, axis2=2)
                # Rounding can mimic a rank-one fit or a tie, so verify either on the profiles.
                refine = partial(
                    _refine_rotation_shift,
                    profiles=profiles_block,
                    weights=shift_weights,
                    exact_design=exact_design,
                    near_rank_one=np.sqrt(np.maximum(0, effective_rank[cols] - 1)) <= tolerance,
                    shift_tolerance=tolerance * trace,
                )
                pvals[cols], pvalue_error[cols], pvalue_method[cols] = _conditional_tails(
                    scatter,
                    integration_tol=integration_tol,
                    rank_one=rank_one,
                    refine=refine,
                )
            else:
                pvals[cols] = _rotation_tails(scatter, basis, metadata["n_rotations"], seed)
        eligible[genes] = True
    if not eligible.any() and not allow_empty:
        raise ValueError(
            "Rotation tests: no features have enough observed samples and an estimable contrast."
        )
    if n_rotations is None:
        metadata["n_saddlepoint"] = int(np.count_nonzero(pvalue_method == "saddlepoint"))
    return {
        "metadata": metadata,
        "observed": observed,
        "estimate_per_bin": estimates,
        "pvals": pvals,
        "pvalue_error": pvalue_error,
        "pvalue_method": pvalue_method,
        "effective_rank": effective_rank,
        "eligible": eligible,
        "n_obs": n_obs,
        "df_resid": df_resid,
    }


# Null routing


def _run_l2_test(  # noqa: C901 — binary/GLM and masked/unmasked null dispatch
    profiles,
    *,
    groups=None,
    design=None,
    contrast=None,
    presence=None,
    null,
    covariance=None,
    freq_weights=None,
    normalize_shape=False,
    min_samples_per_group=2,
    min_resid_df=1,
    n_perm=1000,
    max_exact_permutations=10000,
    integration_tol=1e-6,
    random_state=None,
):
    """Select one L2 null for binary groups or a scalar GLM contrast.

    Parameters
    ----------
    profiles : np.ndarray
        Raw spectra, shape ``(n_samples, n_features, n_bins)``.
    groups : np.ndarray or None
        Binary labels encoded as 0 and 1. None selects ``design`` and ``contrast``.
    design, contrast
        GLM inputs used when ``groups`` is None.
    presence : np.ndarray or None
        Optional sample-by-feature observation mask.
    null, covariance, freq_weights, normalize_shape
        Null calibration and profile options from the public comparison.
    min_samples_per_group, min_resid_df
        Binary and GLM eligibility thresholds, respectively, for masked fits.
    n_perm, max_exact_permutations, random_state
        Controls for binary label permutations.
    integration_tol
        Absolute p-value error target for deterministic conditional integration.

    Returns
    -------
    observed, pvals, extra, metadata : tuple
        Per-feature statistics and p-values, bin estimates, sample counts, and null metadata.
        Binary bin estimates use A minus B (the negative of the indicator slope).
    """
    binary = groups is not None
    if not binary and null not in ("plugin", "rotation"):
        raise ValueError("GLM null must be 'plugin' or 'rotation'.")
    if null not in _L2_NULL_OPTIONS:
        raise ValueError(f"Unknown null='{null}'. Options: {list(_L2_NULL_OPTIONS)}.")
    _resolve_covariance(covariance, null=null)
    profiles = np.asarray(profiles)
    if profiles.ndim != 3:
        raise ValueError(f"spectra must be 3D, got {profiles.shape}.")
    extra = {}
    if binary and presence is not None:
        extra = {
            "n_obs_A": presence[groups == 0].sum(axis=0),
            "n_obs_B": presence[groups == 1].sum(axis=0),
        }

    if null == "plugin":
        options = {
            "freq_weights": freq_weights,
            "normalize_shape": normalize_shape,
            "covariance": covariance,
            "store_covariance": False,
        }
        if binary:
            if presence is None:
                state = _estimate_two_group_null_covariance(profiles, groups, **options)
            else:
                state = _estimate_two_group_masked_null_covariance(
                    profiles,
                    groups,
                    presence,
                    min_samples_per_group=min_samples_per_group,
                    **options,
                )
        elif presence is None:
            state = _estimate_glm_null_covariance(profiles, design, contrast, **options)
        else:
            state = _estimate_glm_masked_null_covariance(
                profiles,
                design,
                contrast,
                presence,
                min_resid_df=min_resid_df,
                **options,
            )
        observed = state["observed"]
        estimates = state["estimate_per_bin"]
        pvals = _l2_pvalues_from_state(state)
        metadata = {"covariance": state["covariance"], "null": "plugin"}

    elif null == "rotation":
        if binary:
            design = np.column_stack([np.ones(len(groups)), groups])
            contrast = np.array([0.0, 1.0])
            if presence is not None:
                eligible = (extra["n_obs_A"] >= min_samples_per_group) & (
                    extra["n_obs_B"] >= min_samples_per_group
                )
                presence = presence & eligible[None, :]
        state = _run_l2_conditional(
            profiles,
            design,
            contrast,
            presence,
            freq_weights=freq_weights,
            normalize_shape=normalize_shape,
            min_resid_df=min_resid_df if not binary and presence is not None else 1,
            integration_tol=integration_tol,
        )
        observed, pvals, metadata = state["observed"], state["pvals"], state["metadata"]
        estimates = state["estimate_per_bin"]
        extra["P_value_error"] = state["pvalue_error"]
        extra["P_value_method"] = state["pvalue_method"]

    else:  # Binary label permutations; GLM permutations are not implemented.
        observed, pvals, estimates = _run_two_group_permutations(
            profiles,
            groups,
            presence,
            freq_weights=freq_weights,
            normalize_shape=normalize_shape,
            min_samples_per_group=min_samples_per_group,
            n_perm=n_perm,
            max_exact_permutations=max_exact_permutations,
            random_state=random_state,
        )
        metadata = {}

    if binary and null != "permutation":
        estimates *= -1
    extra["Estimate_per_bin"] = list(estimates)
    if not binary and presence is not None:
        extra.update(n_obs=state["n_obs"], df_resid=state["df_resid"])
    return observed, pvals, extra, metadata
