"""Compare gene spectra or mean expression across biological samples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp  # noqa: F401 (historical calibration-script import)
from scipy.stats import t as _t_dist

from sonic.comparators import _inference as _l2
from sonic.comparators import _regression as _lm
from sonic.statistics import apply_bh_correction, cauchy_combine

__all__ = [
    "compare_two_groups",
    "compare_two_groups_masked",
    "compare_two_groups_scalar",
    "compare_glm",
    "compare_glm_masked",
    "compare_glm_scalar",
]

# Retain the private helpers imported by existing paper analyses.
_effective_rank_from_eigenvalues = _l2._effective_rank_from_eigenvalues
_estimate_glm_null_covariance = _l2._estimate_glm_null_covariance
_estimate_two_group_masked_null_covariance = _l2._estimate_two_group_masked_null_covariance
_estimate_two_group_null_covariance = _l2._estimate_two_group_null_covariance
_l2_pvalues_from_state = _l2._l2_pvalues_from_state

_AVAILABLE_STATISTICS = ("log_l2", "log_t_cauchy")
_BINWISE_COLUMNS = ("Estimate_per_bin", "T_per_bin", "P_value_per_bin")


def _welch_test(
    group_a: np.ndarray, group_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed Welch t-statistic, two-sided p-value, and mean difference along axis 0.

    Works for any trailing feature shape: ``(n_samples, n_features)`` gives a
    ``(n_features,)`` result for scalar DE, while ``(n_samples, n_genes, n_bins)``
    gives ``(n_genes, n_bins)`` per-frequency-bin statistics. The p-values use the
    Welch-Satterthwaite degrees of freedom from the t-distribution tail.
    """
    n_a, n_b = group_a.shape[0], group_b.shape[0]
    if n_a < 2 or n_b < 2:
        raise ValueError("Welch tests require at least two samples in each group.")
    if not np.all(np.isfinite(group_a)) or not np.all(np.isfinite(group_b)):
        raise ValueError("Welch tests require finite input values.")
    # Anchor before reductions: repeated non-binary floats otherwise acquire
    # rounding-dependent means and variances when group sizes differ.
    offset = group_a[0] - group_b[0]
    group_a = group_a - group_a[:1]
    group_b = group_b - group_b[:1]
    mean_diff = offset + (group_a.mean(axis=0) - group_b.mean(axis=0))
    var_a = group_a.var(axis=0, ddof=1)
    var_b = group_b.var(axis=0, ddof=1)
    se2_a = var_a / n_a
    se2_b = var_b / n_b
    se2 = se2_a + se2_b
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = mean_diff / np.sqrt(se2)
    t_stat = np.where((se2 == 0) & (mean_diff == 0), 0.0, t_stat)
    # Normalize variance contributions before squaring: no scale-dependent
    # epsilon, and no underflow/overflow from squaring the variances.
    fraction_a = np.divide(se2_a, se2, out=np.zeros_like(se2), where=se2 > 0)
    df = 1.0 / (fraction_a**2 / (n_a - 1) + (1.0 - fraction_a) ** 2 / (n_b - 1))
    df = np.maximum(df, 1.0)
    pvals = 2.0 * _t_dist.sf(np.abs(t_stat), df)
    # Clip the floor to the smallest representable positive float so
    # Cauchy's tan(pi(0.5 - p)) stays finite.
    return t_stat, np.clip(pvals, np.finfo(float).tiny, 1.0), mean_diff


def _comparison_frame(
    n_genes: int,
    gene_names: Sequence[str] | None,
    observed: np.ndarray,
    pvals: np.ndarray,
    *,
    extra: dict[str, Sequence[Any] | np.ndarray] | None = None,
    covariance: str | None = None,
    attrs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the common comparison result table and apply BH correction."""
    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]
    df = pd.DataFrame(
        {
            "Feature": list(gene_names),
            "Statistic": np.asarray(observed, dtype=float),
            "P_value": np.asarray(pvals, dtype=float),
        }
    )
    if extra:
        for key, value in extra.items():
            df[key] = value
    df["P_adj"] = apply_bh_correction(df["P_value"])
    if covariance is not None:
        df.attrs["covariance"] = covariance
        df.attrs["null"] = "plugin"
    if attrs is not None:
        df.attrs.update(attrs)
    return df.sort_values("Statistic", ascending=False, na_position="last").reset_index(drop=True)


def _maybe_log_expression(
    values: np.ndarray,
    *,
    log_expression: bool,
    eps: float,
) -> np.ndarray:
    """Return scalar values, optionally transformed to ``log(values + eps)``."""
    if not log_expression:
        return values
    eps = float(eps)
    if eps <= 0:
        raise ValueError(f"eps must be positive when log_expression=True, got {eps}.")
    if np.any(values + eps <= 0):
        raise ValueError("log_expression=True requires values + eps to be strictly positive.")
    return np.log(values + eps)


def _resolve_profile_null(null, statistic, covariance):
    if null is not None:
        return null
    if statistic in ("log_l2", "l2"):
        return "rotation" if covariance is None else "plugin"
    return "analytic"


def _compare_two_group_profiles(  # noqa: C901 — input validation and statistic routing
    spectra,
    groups,
    presence=None,
    *,
    gene_names=None,
    statistic="log_l2",
    null=None,
    covariance=None,
    normalize_shape=False,
    freq_weights=None,
    min_samples_per_group=2,
    n_perm=1000,
    max_exact_permutations=10000,
    integration_tol=1e-6,
    random_state=None,
):
    """Validate binary inputs, select the statistic, and build one result table."""
    null = _resolve_profile_null(null, statistic, covariance)
    if statistic not in _AVAILABLE_STATISTICS:
        raise ValueError(
            f"Unknown statistic '{statistic}'. Options: {list(_AVAILABLE_STATISTICS)}."
        )
    if statistic == "log_t_cauchy":
        _validate_log_t_cauchy_options(null, covariance)
    else:
        _l2._resolve_covariance(covariance, statistic=statistic, null=null)
    spectra = np.asarray(spectra)
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D (n_samples, n_genes, n_bins), got {spectra.shape}.")
    groups = np.asarray(groups)
    if groups.shape != (spectra.shape[0],):
        raise ValueError(
            f"groups shape {groups.shape} does not match n_samples={spectra.shape[0]}."
        )
    if pd.isna(groups).any():
        raise ValueError("Binary labels must be nonmissing.")
    labels, counts = np.unique(groups, return_counts=True)
    if len(labels) != 2:
        raise ValueError(f"groups must contain exactly two distinct values, got {labels}.")
    if int(min_samples_per_group) < 2:
        raise ValueError("min_samples_per_group must be >= 2.")
    if statistic == "log_l2" and counts.min() < min_samples_per_group:
        raise ValueError(f"Each group requires at least {min_samples_per_group} observed samples.")
    groups = (groups == labels[1]).astype(int)
    if presence is not None:
        presence = np.asarray(presence, dtype=bool)
        if presence.shape != spectra.shape[:2]:
            raise ValueError("presence must have shape (n_samples, n_genes).")

    if statistic == "log_t_cauchy":
        observed, pvals, extra = _run_t_cauchy_test(
            spectra,
            groups=groups,
            presence=presence,
            normalize_shape=normalize_shape,
            freq_weights=freq_weights,
            min_samples_per_group=min_samples_per_group,
        )
        metadata = {"null": "analytic", "statistic": statistic, "per_bin_test": "welch"}
    else:
        observed, pvals, extra, metadata = _l2._run_l2_test(
            spectra,
            groups=groups,
            presence=presence,
            null=null,
            covariance=covariance,
            freq_weights=freq_weights,
            normalize_shape=normalize_shape,
            min_samples_per_group=min_samples_per_group,
            n_perm=n_perm,
            max_exact_permutations=max_exact_permutations,
            integration_tol=integration_tol,
            random_state=random_state,
        )
    result = _comparison_frame(
        spectra.shape[1], gene_names, observed, pvals, extra=extra, attrs=metadata
    )
    if presence is not None and (statistic != "log_l2" or null == "permutation"):
        # Keep the historical masked permutation/Welch column order.
        columns = ["Feature", "n_obs_A", "n_obs_B", "Statistic", "P_value", "Estimate_per_bin"]
        if statistic != "log_l2":
            columns.extend(["T_per_bin", "P_value_per_bin", "Best_bin"])
        result = result[[*columns, "P_adj"]]
    return result


def _validate_log_t_cauchy_options(null, covariance):
    """Log t-Cauchy uses analytic binwise tests without cross-bin covariance."""
    if null != "analytic":
        raise ValueError("statistic='log_t_cauchy' requires null='analytic'.")
    if covariance is not None:
        raise ValueError("statistic='log_t_cauchy' requires covariance=None.")


def _run_t_cauchy_test(  # noqa: C901 — binary/GLM eligibility and masked fits
    spectra,
    *,
    groups=None,
    design=None,
    contrast=None,
    prepared_design=None,
    unit_fits=None,
    presence=None,
    normalize_shape=False,
    log_transform=True,
    freq_weights=None,
    min_samples_per_group=2,
    min_resid_df=1,
):
    """Fit binwise t tests and combine them on each feature's observed samples.

    Parameters
    ----------
    spectra : np.ndarray
        Profiles of shape ``(n_samples, n_features, n_bins)``.
    groups : np.ndarray, optional
        Validated binary codes (0 or 1). Selects Welch tests; otherwise use OLS.
    design, contrast
        GLM design and one linear contrast, used when ``groups`` is None.
    prepared_design : tuple, optional
        Validated unmasked GLM design from ``_prepare_glm_design``. Avoids
        repeating its decomposition across cross-spectrum bin-support groups.
    unit_fits : dict, optional
        Reuse constant-response fits by sample mask across cross-spectrum blocks.
    presence : np.ndarray, optional
        Sample-by-feature observation mask. Missing entries are never fitted.
    normalize_shape : bool
        Normalize total power before applying the L2 log transformation.
    log_transform : bool
        Apply the L2 log floor; False retains signed cross-spectrum profiles.
    freq_weights : np.ndarray, optional
        Fixed bin weights, renormalized over each feature's informative bins.
    min_samples_per_group, min_resid_df : int
        Masked binary and GLM eligibility thresholds, respectively.

    Returns
    -------
    observed, pvals, extra : tuple
        Maximum absolute t, Cauchy-combined p-values, and per-bin estimates,
        signed t statistics, p-values and best-bin indices, with masked sample
        counts. Skipped features retain NaN arrays and a missing best bin.
    """
    spectra = np.asarray(spectra)
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D (n_samples, n_genes, n_bins), got {spectra.shape}.")
    n_samples, n_genes, n_bins = spectra.shape
    if n_samples == 0 or n_bins == 0:
        raise ValueError("Binwise tests require samples and frequency bins.")
    weights = _l2._resolve_freq_weights(freq_weights, n_bins)
    masked = presence is not None
    if masked:
        presence = np.asarray(presence, dtype=bool)
        if presence.shape != spectra.shape[:2]:
            raise ValueError("presence must have shape (n_samples, n_genes).")
    extra = {}
    binary = groups is not None
    if binary:
        if masked and int(min_samples_per_group) < 2:
            raise ValueError("min_samples_per_group must be >= 2.")
        if masked:
            extra = {
                "n_obs_A": presence[groups == 0].sum(axis=0),
                "n_obs_B": presence[groups == 1].sum(axis=0),
            }
    else:
        if unit_fits is None:
            unit_fits = {}
        if masked and int(min_resid_df) < 1:
            raise ValueError("min_resid_df must be >= 1.")
        if masked:
            matrix, columns = _lm._build_design_matrix(design, n_samples)
            vector = _lm._resolve_contrast(contrast, columns)
            full_fit = _lm._decompose_glm_design(matrix, vector)
            extra = {"n_obs": presence.sum(axis=0), "df_resid": np.zeros(n_genes, dtype=int)}
        else:
            matrix, _, vector, full_fit = (
                prepared_design
                if prepared_design is not None
                else _lm._prepare_glm_design(design, contrast, n_samples)
            )

    observed = np.full(n_genes, np.nan)
    pvals = np.full(n_genes, np.nan)
    binwise = {name: np.full((n_genes, n_bins), np.nan) for name in _BINWISE_COLUMNS}
    best_bin = np.full(n_genes, np.nan)
    mask_groups = (
        _l2._iter_mask_groups(presence)
        if masked
        else [(np.ones(n_samples, dtype=bool), np.arange(n_genes))]
    )
    for sample_mask, genes in mask_groups:
        count = int(sample_mask.sum())
        if count == 0:
            continue
        if binary:
            codes = groups[sample_mask]
            if masked and min(np.sum(codes == 0), np.sum(codes == 1)) < int(min_samples_per_group):
                continue
            n_terms = 0
            if normalize_shape:
                x = np.column_stack([codes == 0, codes == 1])
                inv = x.T / x.sum(axis=0)[:, None]  # Each fitted value is its group mean.
        else:
            fit = (
                full_fit
                if sample_mask.all()
                else _lm._decompose_glm_design(matrix[sample_mask], vector)
            )
            x, inv = fit.matrix, fit.inverse
            df_resid = count - fit.rank
            if masked:
                extra["df_resid"][genes] = df_resid
            scale = float(np.sum(fit.sample_weights**2))
            if (
                df_resid < int(min_resid_df)
                or not fit.estimable
                or not np.isfinite(scale)
                or scale <= 0
            ):
                continue
            n_terms = x.shape[1]
            mask_key = sample_mask.tobytes()

        rows = np.flatnonzero(sample_mask)
        for block in _l2._iter_feature_blocks(count, len(genes), n_bins, n_terms):
            cols = genes[block]
            values = spectra[rows[:, None], cols]
            response = _l2._prepare_profile_block(
                values,
                normalize_shape,
                log_transform=log_transform,
                design_fit=(x, inv) if normalize_shape else None,
            )
            if binary:
                t_stat, bins, effect = _welch_test(response[codes == 0], response[codes == 1])
                informative = np.any(response != response[:1], axis=0) | (effect != 0)
            else:
                t_stat, bins, effect, informative = _lm._test_glm_response(
                    response.reshape(count, -1),
                    fit,
                    unit_fits=unit_fits,
                    mask_key=mask_key,
                )
                informative = informative.reshape(-1, n_bins)
            t_stat = t_stat.reshape(-1, n_bins)
            effect = effect.reshape(-1, n_bins)
            bins = bins.reshape(-1, n_bins)
            tail = cauchy_combine(
                bins,
                axis=-1,
                where=informative,
                weights=weights if freq_weights is not None else None,
            )
            observed[cols] = np.max(np.abs(t_stat), axis=-1)
            pvals[cols] = tail
            binwise["Estimate_per_bin"][cols] = effect
            binwise["T_per_bin"][cols] = t_stat
            binwise["P_value_per_bin"][cols] = bins
            selected = informative & np.isfinite(bins)
            best_bin[cols] = np.where(
                selected.any(axis=-1), np.argmin(np.where(selected, bins, np.inf), axis=-1), np.nan
            )
    return (
        observed,
        pvals,
        {
            **extra,
            **{name: list(values) for name, values in binwise.items()},
            "Best_bin": pd.array(best_bin, dtype="Int64"),
        },
    )


def compare_two_groups(
    spectra: np.ndarray,
    groups: np.ndarray,
    gene_names: Sequence[str] | None = None,
    statistic: str = "log_l2",
    null: str | None = None,
    n_perm: int = 1000,
    max_exact_permutations: int = 10000,
    random_state: int | None = None,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
    *,
    min_samples_per_group: int = 2,
    covariance: str | None = None,
    integration_tol: float = 1e-6,
) -> pd.DataFrame:
    """Compare each gene's power spectrum between two groups.

    Parameters
    ----------
    spectra : np.ndarray
        Spectral power with shape ``(n_samples, n_genes, n_bins)``. Supply
        finite, nonnegative power for observed samples.
    groups : array-like
        Two distinct group labels, one per sample. Group A has the first
        label in sorted order. Group B has the second.
    gene_names : sequence of str, optional
        Names in input gene order. The default labels are zero-based indices
        written as strings.
    statistic : {'log_l2', 'log_t_cauchy'}, default 'log_l2'
        L2 measures the size of the contrast across bins. ``'log_t_cauchy'``
        combines bin p-values and identifies the bin with the smallest
        p-value among those used in the combination.
    null : str, optional
        For L2, use ``'rotation'`` (default) or ``'plugin'`` for Gaussian
        plug-in calibration. Use ``'permutation'`` for two exchangeable groups.
        For ``'log_t_cauchy'``, the only calibration is ``'analytic'`` (default).
    n_perm : int, default 1000
        Number of random group assignments for a sampled permutation test.
    max_exact_permutations : int, default 10000
        Enumerate all group assignments when their number is at most this
        limit. Set it to zero to sample assignments.
    random_state : int, optional
        Random seed for sampled permutations.
    freq_weights : np.ndarray, optional
        Positive, finite weights in fitted bin order.
        The default gives each bin equal weight. Choose weights before testing.
    normalize_shape : bool, default False
        Divide each gene's power in each bin by its total power in that
        sample before taking the logarithm (DF-norm).
    min_samples_per_group : int, default 2
        Minimum number of samples in each group for L2. Welch's t test
        requires two samples in each group.
    covariance : str, optional
        With plug-in calibration, ``None`` or ``'per_gene'`` estimates residual
        covariance across bins for each gene. ``'pooled'`` shares this estimate
        across eligible genes. Setting a covariance selects plug-in calibration
        if you omit ``null``. For other calibrations, leave this as ``None``.
    integration_tol : float, default 1e-6
        Target absolute error for numerical integration of rotation p-values.

    Returns
    -------
    pandas.DataFrame
        One row per gene, sorted by decreasing ``Statistic``. ``Feature`` identifies
        the gene. ``P_value`` tests its contrast. ``P_adj`` gives the
        Benjamini–Hochberg adjustment across tested genes.

        ``Estimate_per_bin`` gives the mean in group A minus the mean in group B.
        Estimates use natural-log power or natural-log power share in fitted bin order.
        L2 reports the weighted root mean square of these estimates.

        For t-Cauchy, ``Statistic`` is the largest absolute t statistic.
        ``T_per_bin`` and ``P_value_per_bin`` give signed t statistics and
        unadjusted two-sided p-values. ``Best_bin`` is the zero-based index
        of the smallest p-value among bins used in the combination.

        For rotation, ``P_value_method`` identifies the calculation.
        ``P_value_error`` gives its numerical error estimate or budget, when available.

    Raises
    ------
    ValueError
        If the input or test options are invalid.
    RuntimeError
        If the rotation p-value cannot be evaluated.

    Notes
    -----
    The binwise test uses Welch's t test. See :doc:`/guides/multisample` for calibration
    assumptions and interpretation.
    """
    return _compare_two_group_profiles(
        spectra,
        groups,
        None,
        gene_names=gene_names,
        statistic=statistic,
        null=null,
        covariance=covariance,
        normalize_shape=normalize_shape,
        freq_weights=freq_weights,
        min_samples_per_group=min_samples_per_group,
        n_perm=n_perm,
        max_exact_permutations=max_exact_permutations,
        integration_tol=integration_tol,
        random_state=random_state,
    )


def compare_two_groups_masked(
    spectra: np.ndarray,
    groups: np.ndarray,
    presence: np.ndarray,
    gene_names: Sequence[str] | None = None,
    statistic: str = "log_l2",
    null: str | None = None,
    n_perm: int = 1000,
    max_exact_permutations: int = 10000,
    random_state: int | None = None,
    min_samples_per_group: int = 2,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
    *,
    covariance: str | None = None,
    integration_tol: float = 1e-6,
) -> pd.DataFrame:
    """Compare each gene's power spectrum between groups using its observed samples.

    Parameters
    ----------
    spectra : np.ndarray
        Spectral power with shape ``(n_samples, n_genes, n_bins)``. Supply
        finite, nonnegative power for observed samples.
    groups : array-like
        Two distinct group labels, one per sample. Group A has the first
        label in sorted order. Group B has the second.
    presence : np.ndarray
        Boolean array with shape ``(n_samples, n_genes)``. True means the
        gene is present in that sample.
    gene_names : sequence of str, optional
        Names in input gene order. The default labels are zero-based indices
        written as strings.
    statistic : {'log_l2', 'log_t_cauchy'}, default 'log_l2'
        L2 measures the size of the contrast across bins. ``'log_t_cauchy'``
        combines bin p-values and identifies the bin with the smallest
        p-value among those used in the combination.
    null : str, optional
        For L2, use ``'rotation'`` (default) or ``'plugin'`` for Gaussian
        plug-in calibration. Use ``'permutation'`` for two exchangeable groups.
        For ``'log_t_cauchy'``, the only calibration is ``'analytic'`` (default).
    n_perm : int, default 1000
        Number of random group assignments for a sampled permutation test.
    max_exact_permutations : int, default 10000
        Enumerate all group assignments when their number is at most this
        limit. Set it to zero to sample assignments.
    random_state : int, optional
        Random seed for sampled permutations.
    min_samples_per_group : int, default 2
        Minimum observed samples in each group for each gene. For L2,
        each group must also meet this minimum before masking.
    freq_weights : np.ndarray, optional
        Positive, finite weights in fitted bin order.
        The default gives each bin equal weight. Choose weights before testing.
    normalize_shape : bool, default False
        Divide each gene's power in each bin by its total power in that
        sample before taking the logarithm (DF-norm).
    covariance : str, optional
        With plug-in calibration, ``None`` or ``'per_gene'`` estimates residual
        covariance across bins for each gene. ``'pooled'`` shares this estimate
        across eligible genes. Setting a covariance selects plug-in calibration
        if you omit ``null``. For other calibrations, leave this as ``None``.
    integration_tol : float, default 1e-6
        Target absolute error for numerical integration of rotation p-values.

    Returns
    -------
    pandas.DataFrame
        One row per gene, sorted by decreasing ``Statistic``. ``Feature`` identifies
        the gene. ``P_value`` tests its contrast. ``P_adj`` gives the
        Benjamini–Hochberg adjustment across tested genes.

        ``Estimate_per_bin`` gives the mean in group A minus the mean in group B.
        Estimates use natural-log power or natural-log power share in fitted bin order.
        L2 reports the weighted root mean square of these estimates.

        For t-Cauchy, ``Statistic`` is the largest absolute t statistic.
        ``T_per_bin`` and ``P_value_per_bin`` give signed t statistics and
        unadjusted two-sided p-values. ``Best_bin`` is the zero-based index
        of the smallest p-value among bins used in the combination.

        For rotation, ``P_value_method`` identifies the calculation.
        ``P_value_error`` gives its numerical error estimate or budget, when available.
        ``n_obs_A`` and ``n_obs_B`` give the sample count in each group.

    Raises
    ------
    ValueError
        If the input or test options are invalid.
    RuntimeError
        If the rotation p-value cannot be evaluated.

    Notes
    -----
    The binwise test uses Welch's t test. See :doc:`/guides/multisample` for calibration
    assumptions and interpretation.
    """
    return _compare_two_group_profiles(
        spectra,
        groups,
        presence,
        gene_names=gene_names,
        statistic=statistic,
        null=null,
        covariance=covariance,
        normalize_shape=normalize_shape,
        freq_weights=freq_weights,
        n_perm=n_perm,
        max_exact_permutations=max_exact_permutations,
        integration_tol=integration_tol,
        random_state=random_state,
        min_samples_per_group=min_samples_per_group,
    )


def compare_glm(
    spectra: np.ndarray,
    design: pd.DataFrame | np.ndarray,
    contrast: str | dict[str, float] | np.ndarray,
    gene_names: Sequence[str] | None = None,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
    *,
    statistic: str = "log_l2",
    covariance: str | None = None,
    null: str | None = None,
    integration_tol: float = 1e-6,
) -> pd.DataFrame:
    """Test a linear contrast in each gene's power spectrum.

    Parameters
    ----------
    spectra : np.ndarray
        Spectral power with shape ``(n_samples, n_genes, n_bins)``. Supply
        finite, nonnegative power for observed samples.
    design : pandas.DataFrame or 2-D np.ndarray
        One row per sample. SONIC adds an intercept to a data frame and encodes
        categorical predictors. Include any intercept in a numeric matrix yourself.
    contrast : str, dict, or np.ndarray
        Name one coefficient, map coefficient names to weights, or supply a
        vector in encoded design-matrix column order. The contrast must be
        estimable from the design.
    gene_names : sequence of str, optional
        Names in input gene order. The default labels are zero-based indices
        written as strings.
    freq_weights : np.ndarray, optional
        Positive, finite weights in fitted bin order.
        The default gives each bin equal weight. Choose weights before testing.
    normalize_shape : bool, default False
        Divide each gene's power in each bin by its total power in that
        sample before taking the logarithm (DF-norm).
    statistic : {'log_l2', 'log_t_cauchy'}, default 'log_l2'
        L2 measures the size of the contrast across bins. ``'log_t_cauchy'``
        combines bin p-values and identifies the bin with the smallest
        p-value among those used in the combination.
    covariance : str, optional
        With plug-in calibration, ``None`` or ``'per_gene'`` estimates residual
        covariance across bins for each gene. ``'pooled'`` shares this estimate
        across eligible genes. Setting a covariance selects plug-in calibration
        if you omit ``null``. For other calibrations, leave this as ``None``.
    null : str, optional
        For L2, use ``'rotation'`` (default) or ``'plugin'`` for Gaussian
        plug-in calibration.
        For ``'log_t_cauchy'``, the only calibration is ``'analytic'`` (default).
    integration_tol : float, default 1e-6
        Target absolute error for numerical integration of rotation p-values.

    Returns
    -------
    pandas.DataFrame
        One row per gene, sorted by decreasing ``Statistic``. ``Feature`` identifies
        the gene. ``P_value`` tests its contrast. ``P_adj`` gives the
        Benjamini–Hochberg adjustment across tested genes.

        ``Estimate_per_bin`` gives the specified signed contrast in each bin.
        Estimates use natural-log power or natural-log power share in fitted bin order.
        L2 reports the weighted root mean square of these estimates.

        For t-Cauchy, ``Statistic`` is the largest absolute t statistic.
        ``T_per_bin`` and ``P_value_per_bin`` give signed t statistics and
        unadjusted two-sided p-values. ``Best_bin`` is the zero-based index
        of the smallest p-value among bins used in the combination.

        For rotation, ``P_value_method`` identifies the calculation.
        ``P_value_error`` gives its numerical error estimate or budget, when available.

    Raises
    ------
    ValueError
        If the input or test options are invalid.
    RuntimeError
        If the rotation p-value cannot be evaluated.
    TypeError
        If the design or contrast has an unsupported type.

    Notes
    -----
    The binwise test uses ordinary least squares. See :doc:`/guides/multisample` for calibration
    assumptions and interpretation.
    """
    null = _resolve_profile_null(null, statistic, covariance)
    if statistic == "log_t_cauchy":
        _validate_log_t_cauchy_options(null, covariance)
        observed, pvals, extra = _run_t_cauchy_test(
            spectra,
            design=design,
            contrast=contrast,
            normalize_shape=normalize_shape,
            freq_weights=freq_weights,
        )
        metadata = {"null": "analytic", "statistic": statistic, "per_bin_test": "ols"}
    elif statistic == "log_l2":
        observed, pvals, extra, metadata = _l2._run_l2_test(
            spectra,
            design=design,
            contrast=contrast,
            null=null,
            covariance=covariance,
            freq_weights=freq_weights,
            normalize_shape=normalize_shape,
            integration_tol=integration_tol,
        )
    else:
        raise ValueError("GLM statistic must be 'log_l2' or 'log_t_cauchy'.")
    return _comparison_frame(
        len(observed), gene_names, observed, pvals, extra=extra, attrs=metadata
    )


def compare_glm_masked(
    spectra: np.ndarray,
    design: pd.DataFrame | np.ndarray,
    contrast: str | dict[str, float] | np.ndarray,
    presence: np.ndarray,
    gene_names: Sequence[str] | None = None,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
    min_resid_df: int = 1,
    *,
    statistic: str = "log_l2",
    covariance: str | None = None,
    null: str | None = None,
    integration_tol: float = 1e-6,
) -> pd.DataFrame:
    """Test a linear contrast in each gene's power spectrum using its observed samples.

    Parameters
    ----------
    spectra : np.ndarray
        Spectral power with shape ``(n_samples, n_genes, n_bins)``. Supply
        finite, nonnegative power for observed samples.
    design : pandas.DataFrame or 2-D np.ndarray
        One row per sample. SONIC adds an intercept to a data frame and encodes
        categorical predictors. Include any intercept in a numeric matrix yourself.
    contrast : str, dict, or np.ndarray
        Name one coefficient, map coefficient names to weights, or supply a
        vector in encoded design-matrix column order. The contrast must be
        estimable from the design.
    presence : np.ndarray
        Boolean array with shape ``(n_samples, n_genes)``. True means the
        gene is present in that sample.
    gene_names : sequence of str, optional
        Names in input gene order. The default labels are zero-based indices
        written as strings.
    freq_weights : np.ndarray, optional
        Positive, finite weights in fitted bin order.
        The default gives each bin equal weight. Choose weights before testing.
    normalize_shape : bool, default False
        Divide each gene's power in each bin by its total power in that
        sample before taking the logarithm (DF-norm).
    min_resid_df : int, default 1
        Minimum residual degrees of freedom for each gene's linear model.
    statistic : {'log_l2', 'log_t_cauchy'}, default 'log_l2'
        L2 measures the size of the contrast across bins. ``'log_t_cauchy'``
        combines bin p-values and identifies the bin with the smallest
        p-value among those used in the combination.
    covariance : str, optional
        With plug-in calibration, ``None`` or ``'per_gene'`` estimates residual
        covariance across bins for each gene. ``'pooled'`` shares this estimate
        across eligible genes. Setting a covariance selects plug-in calibration
        if you omit ``null``. For other calibrations, leave this as ``None``.
    null : str, optional
        For L2, use ``'rotation'`` (default) or ``'plugin'`` for Gaussian
        plug-in calibration.
        For ``'log_t_cauchy'``, the only calibration is ``'analytic'`` (default).
    integration_tol : float, default 1e-6
        Target absolute error for numerical integration of rotation p-values.

    Returns
    -------
    pandas.DataFrame
        One row per gene, sorted by decreasing ``Statistic``. ``Feature`` identifies
        the gene. ``P_value`` tests its contrast. ``P_adj`` gives the
        Benjamini–Hochberg adjustment across tested genes.

        ``Estimate_per_bin`` gives the specified signed contrast in each bin.
        Estimates use natural-log power or natural-log power share in fitted bin order.
        L2 reports the weighted root mean square of these estimates.

        For t-Cauchy, ``Statistic`` is the largest absolute t statistic.
        ``T_per_bin`` and ``P_value_per_bin`` give signed t statistics and
        unadjusted two-sided p-values. ``Best_bin`` is the zero-based index
        of the smallest p-value among bins used in the combination.

        For rotation, ``P_value_method`` identifies the calculation.
        ``P_value_error`` gives its numerical error estimate or budget, when available.
        ``n_obs`` and ``df_resid`` give the sample count and residual degrees
        of freedom.

    Raises
    ------
    ValueError
        If the input or test options are invalid.
    RuntimeError
        If the rotation p-value cannot be evaluated.
    TypeError
        If the design or contrast has an unsupported type.

    Notes
    -----
    The binwise test uses ordinary least squares. See :doc:`/guides/multisample` for calibration
    assumptions and interpretation.
    """
    null = _resolve_profile_null(null, statistic, covariance)
    if statistic == "log_t_cauchy":
        _validate_log_t_cauchy_options(null, covariance)
        observed, pvals, extra = _run_t_cauchy_test(
            spectra,
            design=design,
            contrast=contrast,
            presence=presence,
            min_resid_df=min_resid_df,
            normalize_shape=normalize_shape,
            freq_weights=freq_weights,
        )
        metadata = {"null": "analytic", "statistic": statistic, "per_bin_test": "ols"}
    elif statistic == "log_l2":
        observed, pvals, extra, metadata = _l2._run_l2_test(
            spectra,
            design=design,
            contrast=contrast,
            presence=presence,
            null=null,
            covariance=covariance,
            freq_weights=freq_weights,
            normalize_shape=normalize_shape,
            integration_tol=integration_tol,
            min_resid_df=min_resid_df,
        )
    else:
        raise ValueError("GLM statistic must be 'log_l2' or 'log_t_cauchy'.")
    return _comparison_frame(
        len(observed), gene_names, observed, pvals, extra=extra, attrs=metadata
    )


def compare_two_groups_scalar(
    values: np.ndarray,
    groups: np.ndarray,
    gene_names: Sequence[str] | None = None,
    *,
    log_expression: bool = False,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Compare two groups for each gene with a two-sided Welch t test.

    Parameters
    ----------
    values : np.ndarray
        Sample-level responses with shape ``(n_samples, n_genes)``.
    groups : array-like
        Two distinct group labels, one per sample. Group A has the first
        label in sorted order. Group B has the second.
    gene_names : sequence of str, optional
        Names in input gene order. The default labels are zero-based indices
        written as strings.
    log_expression : bool, default False
        Test ``log(values + eps)`` using the natural logarithm. By default
        the test uses the input values.
    eps : float, default 1e-12
        Positive offset for ``log_expression=True``.

    Returns
    -------
    pandas.DataFrame
        One row per gene, sorted by decreasing ``Statistic``. ``Feature`` identifies
        the gene. ``P_value`` tests its contrast. ``P_adj`` gives the
        Benjamini–Hochberg adjustment across tested genes.
        ``Statistic`` is the absolute t statistic. ``Mean_diff`` gives the mean
        in group A minus the mean in group B on the tested scale.

    Raises
    ------
    ValueError
        If the input or test options are invalid.

    Notes
    -----
    Each group needs at least two observations. Welch's test allows unequal
    group variances.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"values must be 2D (n_samples, n_genes), got {values.shape}.")
    values = _maybe_log_expression(values, log_expression=log_expression, eps=eps)
    n_samples, n_genes = values.shape
    groups = np.asarray(groups)
    if groups.shape != (n_samples,):
        raise ValueError(f"groups length {groups.shape} does not match n_samples={n_samples}.")
    uniq = np.unique(groups)
    if uniq.size != 2:
        raise ValueError(f"groups must contain exactly two distinct values, got {uniq}.")
    group_codes = (groups == uniq[1]).astype(int)

    # Compute the Welch t-test.
    a_vals = values[group_codes == 0]
    b_vals = values[group_codes == 1]
    t_stat, pvals, mean_diff = _welch_test(a_vals, b_vals)
    observed = np.abs(t_stat)

    frame = _comparison_frame(
        n_genes,
        gene_names,
        observed,
        pvals,
        extra={"Mean_diff": mean_diff},
    )
    return frame[["Feature", "Statistic", "Mean_diff", "P_value", "P_adj"]]


def compare_glm_scalar(
    values: np.ndarray,
    design: pd.DataFrame | np.ndarray,
    contrast: str | dict[str, float] | np.ndarray,
    gene_names: Sequence[str] | None = None,
    *,
    log_expression: bool = False,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Test a linear contrast for each gene with a two-sided ordinary least squares t test.

    Parameters
    ----------
    values : np.ndarray
        Sample-level responses with shape ``(n_samples, n_genes)``.
    design : pandas.DataFrame or 2-D np.ndarray
        One row per sample. SONIC adds an intercept to a data frame and encodes
        categorical predictors. Include any intercept in a numeric matrix yourself.
    contrast : str, dict, or np.ndarray
        Name one coefficient, map coefficient names to weights, or supply a
        vector in encoded design-matrix column order. The contrast must be
        estimable from the design.
    gene_names : sequence of str, optional
        Names in input gene order. The default labels are zero-based indices
        written as strings.
    log_expression : bool, default False
        Test ``log(values + eps)`` using the natural logarithm. By default
        the test uses the input values.
    eps : float, default 1e-12
        Positive offset for ``log_expression=True``.

    Returns
    -------
    pandas.DataFrame
        One row per gene, sorted by decreasing ``Statistic``. ``Feature`` identifies
        the gene. ``P_value`` tests its contrast. ``P_adj`` gives the
        Benjamini–Hochberg adjustment across tested genes.
        ``Statistic`` is the absolute t statistic. ``Estimate`` gives the signed
        contrast on the response scale used in the test.

    Raises
    ------
    ValueError
        If the input or test options are invalid.
    TypeError
        If the design or contrast has an unsupported type.

    Notes
    -----
    The usual t reference assumes independent Gaussian errors with a common
    variance across samples for each gene.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"values must be 2D (n_samples, n_genes), got {values.shape}.")
    n_samples, n_genes = values.shape

    values = _maybe_log_expression(values, log_expression=log_expression, eps=eps)

    if not np.isfinite(values).all():
        raise ValueError("Scalar GLM responses must be finite.")
    _, _, _, fit = _lm._prepare_glm_design(design, contrast, n_samples)
    t_stat, pvals, estimate, _ = _lm._test_glm_response(values, fit)
    observed = np.abs(t_stat)

    frame = _comparison_frame(
        n_genes,
        gene_names,
        observed,
        pvals,
        extra={"Estimate": estimate},
    )
    return frame[["Feature", "Statistic", "Estimate", "P_value", "P_adj"]]
