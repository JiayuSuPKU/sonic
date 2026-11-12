"""Public spectral and scalar comparison functions for comparator outputs.

Spectral L2 inference lives in ``_inference``; shared OLS fits live in
``_regression``. These functions validate user options and assemble results.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp  # noqa: F401 (historical calibration-script import)
from scipy.stats import t as _t_dist

from sonic.comparators import _inference as _l2
from sonic.comparators import _regression as _lm
from sonic.comparators.normalization import _normalize_shape_apply
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

logger = logging.getLogger(__name__)
_AVAILABLE_STATISTICS = ("log_l2", "welch_t_cauchy")


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


def _run_welch_t_cauchy_analytic(
    spectra: np.ndarray,
    group_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin Welch t test + Cauchy-combined gene-level p-value.

    Both the per-bin significance and the gene-level combination are
    **analytic**: per-bin p-values come from the Welch t-distribution
    and the gene-level p comes from the Cauchy combination.

    Returns
    -------
    observed_abs_t : np.ndarray
        ``(n_genes, n_bins)`` observed per-bin ``|t|`` — used as the reported
        statistic summary (the max across bins sorts the output table
        sensibly, same convention as before).
    combined_pvals : np.ndarray
        ``(n_genes,)`` Cauchy-combined gene-level p-values built from per-bin
        analytic Welch p-values.
    per_bin_pvals : np.ndarray
        ``(n_genes, n_bins)`` per-bin analytic Welch two-sided p-values.
    """
    a_mask = group_codes == 0
    t_stat, per_bin_pvals, _ = _welch_test(spectra[a_mask], spectra[~a_mask])
    abs_t = np.abs(t_stat)
    # A bin constant across all samples has no information about the contrast.
    # Its p=1 would otherwise dominate the Cauchy sum and suppress other bins.
    informative = np.any(spectra != spectra[:1], axis=0)
    combined = cauchy_combine(per_bin_pvals, axis=-1, where=informative)
    return abs_t, combined, per_bin_pvals


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
        df.attrs["null"] = "analytic"
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


def _compare_two_group_profiles(  # noqa: C901 — input validation and statistic routing
    spectra,
    groups,
    presence=None,
    *,
    gene_names=None,
    statistic="log_l2",
    null="analytic",
    covariance=None,
    normalize_shape=False,
    freq_weights=None,
    min_samples_per_group=2,
    n_perm=1000,
    max_exact_permutations=10000,
    n_rotations=19999,
    random_state=None,
):
    """Validate binary inputs, select the statistic, and build one result table."""
    if statistic not in _AVAILABLE_STATISTICS:
        raise ValueError(
            f"Unknown statistic '{statistic}'. Options: {list(_AVAILABLE_STATISTICS)}."
        )
    if null not in _l2._NULL_OPTIONS:
        raise ValueError(f"Unknown null='{null}'. Options: {list(_l2._NULL_OPTIONS)}.")
    _l2._resolve_covariance(covariance, statistic=statistic, null=null)
    if null == "rotation" and statistic != "log_l2":
        raise ValueError("null='rotation' requires statistic='log_l2'.")
    spectra = np.asarray(spectra)
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D (n_samples, n_genes, n_bins), got {spectra.shape}.")
    groups = np.asarray(groups)
    if groups.shape != (spectra.shape[0],):
        raise ValueError(
            f"groups shape {groups.shape} does not match n_samples={spectra.shape[0]}."
        )
    labels = np.unique(groups)
    if len(labels) != 2:
        raise ValueError(f"groups must contain exactly two distinct values, got {labels}.")
    groups = (groups == labels[1]).astype(int)
    if presence is not None:
        presence = np.asarray(presence, dtype=bool)
        if presence.shape != spectra.shape[:2]:
            raise ValueError("presence must have shape (n_samples, n_genes).")

    if statistic == "welch_t_cauchy":
        if freq_weights is not None:
            logger.debug("freq_weights is ignored by statistic='welch_t_cauchy'.")
        observed, pvals, extra = _run_welch_profiles(
            spectra, groups, presence, normalize_shape, min_samples_per_group
        )
        metadata = {}
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
            n_rotations=n_rotations,
            random_state=random_state,
        )
    result = _comparison_frame(
        spectra.shape[1], gene_names, observed, pvals, extra=extra, attrs=metadata
    )
    if presence is not None and (statistic == "welch_t_cauchy" or null == "permutation"):
        # Keep the historical masked permutation/Welch column order.
        columns = ["Feature", "n_obs_A", "n_obs_B", "Statistic", "P_value"]
        if statistic == "welch_t_cauchy":
            columns.append("P_value_per_bin")
        result = result[[*columns, "P_adj"]]
    return result


def _run_welch_profiles(spectra, groups, presence, normalize_shape, min_samples_per_group):
    """Apply Welch-Cauchy to complete profiles or each feature's observed samples."""
    spectra = np.asarray(spectra, dtype=np.float64)
    if normalize_shape:
        spectra = _normalize_shape_apply(spectra)
    if presence is None:
        statistic, pvals, per_bin = _run_welch_t_cauchy_analytic(spectra, groups)
        return statistic.max(axis=-1), pvals, {"P_value_per_bin": list(per_bin)}

    n_a = presence[groups == 0].sum(axis=0)
    n_b = presence[groups == 1].sum(axis=0)
    observed = np.full(spectra.shape[1], np.nan)
    pvals = np.full(spectra.shape[1], np.nan)
    per_bin = [None] * spectra.shape[1]
    for gene in np.flatnonzero((n_a >= min_samples_per_group) & (n_b >= min_samples_per_group)):
        mask = presence[:, gene]
        statistic, tail, bins = _run_welch_t_cauchy_analytic(
            spectra[mask, gene : gene + 1], groups[mask]
        )
        observed[gene], pvals[gene], per_bin[gene] = float(statistic.max()), tail[0], bins[0]
    return observed, pvals, {"n_obs_A": n_a, "n_obs_B": n_b, "P_value_per_bin": per_bin}


def compare_two_groups(
    spectra: np.ndarray,
    groups: np.ndarray,
    gene_names: Sequence[str] | None = None,
    statistic: str = "log_l2",
    null: str = "analytic",
    n_perm: int = 1000,
    max_exact_permutations: int = 10000,
    random_state: int | None = None,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
    *,
    covariance: str | None = None,
    n_rotations: int = 19999,
) -> pd.DataFrame:
    """Test a spectral difference between two groups for each gene.

    Parameters
    ----------
    spectra : np.ndarray
        Raw spectral powers of shape ``(n_samples, n_genes, n_bins)``;
        not logged. Inputs are not modified.
    groups : np.ndarray
        Labels of shape ``(n_samples,)`` with exactly two distinct values.
        Groups A and B follow the sorted label order.
    gene_names : sequence of str, optional
        Feature names. Defaults to stringified zero-based indices.
    statistic : str
        ``'log_l2'`` (default) is the weighted L2 contrast of log spectra.
        ``'welch_t_cauchy'`` combines per-bin Welch p-values by Cauchy
        combination and reports the largest absolute per-bin t statistic.
        Welch requires two observations per arm and excludes constant bins.
    null : str
        ``'analytic'`` (default) uses the selected covariance and Liu tail.
        ``'rotation'`` conditions on each feature's nuisance-adjusted scatter.
        ``'permutation'`` relabels samples. Welch-Cauchy uses its own analytic
        null with either analytic or permutation selection; rotation requires L2.
    n_perm : int
        Number of sampled label permutations; default 1000. Ignored when
        all assignments are enumerated or another null is selected.
    max_exact_permutations : int
        Enumerate all binary label assignments when their number is at
        most this threshold; default 10000. Zero forces sampling.
    random_state : int, optional
        Seed for resampling. Ignored by analytic tests.
    freq_weights : np.ndarray, optional
        Finite nonnegative weights of length ``n_bins``, normalized to sum
        to one. None gives uniform weights. Used only by L2 tests.
    normalize_shape : bool
        If True, divide each sample/feature spectrum by its total power
        before logging to test spectral shape (DF-norm). Default False.
    covariance : str or None
        ``None`` (default) or ``'pooled'`` shares residual covariance across
        eligible genes. ``'per_gene'`` estimates it separately for each gene.
        An explicit choice requires an analytic L2 test.
    n_rotations : int
        Number of rotation draws; default 19999. Used only with
        ``null='rotation'``, which requires ``covariance=None``.

    Returns
    -------
    pandas.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value`` and ``P_adj``.
        L2 tests report the weighted distance on log spectra.
        Welch-Cauchy also returns ``P_value_per_bin``.
        Rows are sorted by descending statistic; BH uses finite p-values.
        Analytic L2 results record ``null`` and ``covariance`` in ``attrs``.
        Rotation also records ``n_rotations``, the effective ``random_state``,
        and ``min_p_value``.

    Raises
    ------
    ValueError
        If input shapes or options are invalid, or the test cannot be fitted.

    Notes
    -----
    For calibration assumptions and zero-variance behavior, see
    :doc:`/guides/multisample`.
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
        n_perm=n_perm,
        max_exact_permutations=max_exact_permutations,
        n_rotations=n_rotations,
        random_state=random_state,
    )


def compare_two_groups_masked(
    spectra: np.ndarray,
    groups: np.ndarray,
    presence: np.ndarray,
    gene_names: Sequence[str] | None = None,
    statistic: str = "log_l2",
    null: str = "analytic",
    n_perm: int = 1000,
    max_exact_permutations: int = 10000,
    random_state: int | None = None,
    min_samples_per_group: int = 2,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
    *,
    covariance: str | None = None,
    n_rotations: int = 19999,
) -> pd.DataFrame:
    """Test a spectral difference between two groups using observed samples.

    Parameters
    ----------
    spectra : np.ndarray
        Raw spectral powers of shape ``(n_samples, n_genes, n_bins)``;
        not logged. Inputs are not modified.
    groups : np.ndarray
        Labels of shape ``(n_samples,)`` with exactly two distinct values.
        Groups A and B follow the sorted label order.
    presence : np.ndarray
        Boolean observation mask of shape ``(n_samples, n_genes)``.
        False entries exclude that sample from the feature's test.
    gene_names : sequence of str, optional
        Feature names. Defaults to stringified zero-based indices.
    statistic : str
        ``'log_l2'`` (default) is the weighted L2 contrast of log spectra.
        ``'welch_t_cauchy'`` combines per-bin Welch p-values by Cauchy
        combination and reports the largest absolute per-bin t statistic.
        Welch requires two observations per arm and excludes constant bins.
    null : str
        ``'analytic'`` (default) uses the selected covariance and Liu tail.
        ``'rotation'`` conditions on each feature's nuisance-adjusted scatter.
        ``'permutation'`` relabels samples. Welch-Cauchy uses its own analytic
        null with either analytic or permutation selection; rotation requires L2.
    n_perm : int
        Number of sampled label permutations; default 1000. Ignored when
        all assignments are enumerated or another null is selected.
    max_exact_permutations : int
        Enumerate all binary label assignments when their number is at
        most this threshold; default 10000. Zero forces sampling.
    random_state : int, optional
        Seed for resampling. Ignored by analytic tests.
    min_samples_per_group : int
        Minimum observed samples per group for a masked binary test;
        default 2. Features below either threshold receive NaN results.
    freq_weights : np.ndarray, optional
        Finite nonnegative weights of length ``n_bins``, normalized to sum
        to one. None gives uniform weights. Used only by L2 tests.
    normalize_shape : bool
        If True, divide each sample/feature spectrum by its total power
        before logging to test spectral shape (DF-norm). Default False.
    covariance : str or None
        ``None`` (default) or ``'pooled'`` shares residual covariance across
        eligible genes. ``'per_gene'`` estimates it separately for each gene.
        An explicit choice requires an analytic L2 test.
    n_rotations : int
        Number of rotation draws; default 19999. Used only with
        ``null='rotation'``, which requires ``covariance=None``.

    Returns
    -------
    pandas.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value`` and ``P_adj``.
        L2 tests report the weighted distance on log spectra.
        Welch-Cauchy also returns ``P_value_per_bin``.
        Includes ``n_obs_A`` and ``n_obs_B``; skipped genes retain NaN results.
        Rows are sorted by descending statistic; BH uses finite p-values.
        Analytic L2 results record ``null`` and ``covariance`` in ``attrs``.
        Rotation also records ``n_rotations``, the effective ``random_state``,
        and ``min_p_value``.

    Raises
    ------
    ValueError
        If input shapes or options are invalid, or the test cannot be fitted.

    Notes
    -----
    For calibration assumptions and zero-variance behavior, see
    :doc:`/guides/multisample`.
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
        n_rotations=n_rotations,
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
    covariance: str | None = None,
    null: str = "analytic",
    n_rotations: int = 19999,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Test one spectral linear-model contrast for each gene.

    Parameters
    ----------
    spectra : np.ndarray
        Raw spectral powers of shape ``(n_samples, n_genes, n_bins)``;
        not logged. Inputs are not modified.
    design : pandas.DataFrame or np.ndarray
        Sample-level design. A DataFrame adds an intercept and encodes
        categorical columns; column names must remain unique after encoding.
        A numeric array is used as supplied, including any intercept.
    contrast : str, dict, or np.ndarray
        One finite, nonzero, estimable linear contrast. A string names
        an encoded effect; a dictionary maps column names to coefficients;
        an array supplies one coefficient per design column.
    gene_names : sequence of str, optional
        Feature names. Defaults to stringified zero-based indices.
    freq_weights : np.ndarray, optional
        Finite nonnegative weights of length ``n_bins``, normalized to sum
        to one. None gives uniform weights. Used only by L2 tests.
    normalize_shape : bool
        If True, divide each sample/feature spectrum by its total power
        before logging to test spectral shape (DF-norm). Default False.
    covariance : str or None
        ``None`` (default) or ``'pooled'`` shares residual covariance across
        eligible genes. ``'per_gene'`` estimates it separately for each gene.
        An explicit choice requires an analytic L2 test.
    null : str
        ``'analytic'`` (default) uses the selected covariance and Liu tail.
        ``'rotation'`` conditions on each feature's nuisance-adjusted scatter.
        Permutation calibration is available only for binary comparisons.
    n_rotations : int
        Number of rotation draws; default 19999. Used only with
        ``null='rotation'``, which requires ``covariance=None``.
    random_state : int, optional
        Seed for resampling. Ignored by analytic tests.

    Returns
    -------
    pandas.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value`` and ``P_adj``.
        L2 tests report the weighted distance on log spectra.
        Rows are sorted by descending statistic; BH uses finite p-values.
        Analytic L2 results record ``null`` and ``covariance`` in ``attrs``.
        Rotation also records ``n_rotations``, the effective ``random_state``,
        and ``min_p_value``.

    Raises
    ------
    ValueError
        If input shapes or options are invalid, or the test cannot be fitted.
    TypeError
        If the design or contrast has an unsupported type.

    Notes
    -----
    For calibration assumptions and zero-variance behavior, see
    :doc:`/guides/multisample`.
    """
    observed, pvals, extra, metadata = _l2._run_l2_test(
        spectra,
        design=design,
        contrast=contrast,
        null=null,
        covariance=covariance,
        freq_weights=freq_weights,
        normalize_shape=normalize_shape,
        n_rotations=n_rotations,
        random_state=random_state,
    )
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
    covariance: str | None = None,
    null: str = "analytic",
    n_rotations: int = 19999,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Test one spectral linear-model contrast using each gene's observed samples.

    Parameters
    ----------
    spectra : np.ndarray
        Raw spectral powers of shape ``(n_samples, n_genes, n_bins)``;
        not logged. Inputs are not modified.
    design : pandas.DataFrame or np.ndarray
        Sample-level design. A DataFrame adds an intercept and encodes
        categorical columns; column names must remain unique after encoding.
        A numeric array is used as supplied, including any intercept.
    contrast : str, dict, or np.ndarray
        One finite, nonzero, estimable linear contrast. A string names
        an encoded effect; a dictionary maps column names to coefficients;
        an array supplies one coefficient per design column.
    presence : np.ndarray
        Boolean observation mask of shape ``(n_samples, n_genes)``.
        False entries exclude that sample from the feature's test.
    gene_names : sequence of str, optional
        Feature names. Defaults to stringified zero-based indices.
    freq_weights : np.ndarray, optional
        Finite nonnegative weights of length ``n_bins``, normalized to sum
        to one. None gives uniform weights. Used only by L2 tests.
    normalize_shape : bool
        If True, divide each sample/feature spectrum by its total power
        before logging to test spectral shape (DF-norm). Default False.
    min_resid_df : int
        Minimum residual degrees of freedom for a masked GLM test;
        default 1. Ineligible or unestimable features receive NaN results.
    covariance : str or None
        ``None`` (default) or ``'pooled'`` shares residual covariance across
        eligible genes. ``'per_gene'`` estimates it separately for each gene.
        An explicit choice requires an analytic L2 test.
    null : str
        ``'analytic'`` (default) uses the selected covariance and Liu tail.
        ``'rotation'`` conditions on each feature's nuisance-adjusted scatter.
        Permutation calibration is available only for binary comparisons.
    n_rotations : int
        Number of rotation draws; default 19999. Used only with
        ``null='rotation'``, which requires ``covariance=None``.
    random_state : int, optional
        Seed for resampling. Ignored by analytic tests.

    Returns
    -------
    pandas.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value`` and ``P_adj``.
        L2 tests report the weighted distance on log spectra.
        Includes ``n_obs`` and ``df_resid``; skipped genes retain NaN results.
        Rows are sorted by descending statistic; BH uses finite p-values.
        Analytic L2 results record ``null`` and ``covariance`` in ``attrs``.
        Rotation also records ``n_rotations``, the effective ``random_state``,
        and ``min_p_value``.

    Raises
    ------
    ValueError
        If input shapes or options are invalid, or the test cannot be fitted.
    TypeError
        If the design or contrast has an unsupported type.

    Notes
    -----
    For calibration assumptions and zero-variance behavior, see
    :doc:`/guides/multisample`.
    """
    observed, pvals, extra, metadata = _l2._run_l2_test(
        spectra,
        design=design,
        contrast=contrast,
        presence=presence,
        null=null,
        covariance=covariance,
        freq_weights=freq_weights,
        normalize_shape=normalize_shape,
        n_rotations=n_rotations,
        random_state=random_state,
        min_resid_df=min_resid_df,
    )
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
    """Test a scalar mean difference between two groups with Welch's t test.

    Parameters
    ----------
    values : np.ndarray
        Per-sample scalar responses of shape ``(n_samples, n_genes)``.
        Inputs are not modified.
    groups : np.ndarray
        Labels of shape ``(n_samples,)`` with exactly two distinct values.
        Groups A and B follow the sorted label order.
    gene_names : sequence of str, optional
        Feature names. Defaults to stringified zero-based indices.
    log_expression : bool
        If True, fit ``log(values + eps)`` instead of raw values.
        Default False; the reported effect uses the fitted response scale.
    eps : float
        Positive additive offset used only with ``log_expression=True``;
        default 1e-12.

    Returns
    -------
    pandas.DataFrame
        Columns ``Feature``, ``Statistic`` (absolute Welch t), ``Mean_diff``
        (group A minus B), ``P_value`` and ``P_adj``. P-values use the
        two-sided Welch-Satterthwaite t tail.
        Rows are sorted by descending statistic; BH uses finite p-values.

    Raises
    ------
    ValueError
        If input shapes or options are invalid, or the test cannot be fitted.

    Notes
    -----
    Both groups require two observations.
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
    """Test one scalar linear-model contrast with an OLS t test.

    Parameters
    ----------
    values : np.ndarray
        Per-sample scalar responses of shape ``(n_samples, n_genes)``.
        Inputs are not modified.
    design : pandas.DataFrame or np.ndarray
        Sample-level design. A DataFrame adds an intercept and encodes
        categorical columns; column names must remain unique after encoding.
        A numeric array is used as supplied, including any intercept.
    contrast : str, dict, or np.ndarray
        One finite, nonzero, estimable linear contrast. A string names
        an encoded effect; a dictionary maps column names to coefficients;
        an array supplies one coefficient per design column.
    gene_names : sequence of str, optional
        Feature names. Defaults to stringified zero-based indices.
    log_expression : bool
        If True, fit ``log(values + eps)`` instead of raw values.
        Default False; the reported effect uses the fitted response scale.
    eps : float
        Positive additive offset used only with ``log_expression=True``;
        default 1e-12.

    Returns
    -------
    pandas.DataFrame
        Columns ``Feature``, ``Statistic`` (absolute OLS t), ``Estimate``
        (signed contrast), ``P_value`` and ``P_adj``. P-values use the
        two-sided Student t tail with ``n_samples - rank(design)`` df.
        Rows are sorted by descending statistic; BH uses finite p-values.

    Raises
    ------
    ValueError
        If input shapes or options are invalid, or the test cannot be fitted.
    TypeError
        If the design or contrast has an unsupported type.

    Notes
    -----
    With zero residual variance, a zero contrast has p=1 and a nonzero
    contrast has p=0.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"values must be 2D (n_samples, n_genes), got {values.shape}.")
    n_samples, n_genes = values.shape

    values = _maybe_log_expression(values, log_expression=log_expression, eps=eps)

    if not np.isfinite(values).all():
        raise ValueError("Scalar GLM responses must be finite.")
    prepared_design = _lm._prepare_glm_design(design, contrast, n_samples)
    design_matrix, _, contrast_vector, design_inverse, rank = prepared_design
    df_resid = n_samples - rank

    # Fit the OLS model
    _, resid, estimate = _lm._fit_glm_response(
        values, design_matrix, design_inverse, contrast_vector, rank
    )
    sigma2 = np.sum(resid**2, axis=0) / df_resid
    contrast_var = float(np.sum((contrast_vector @ design_inverse) ** 2))
    if contrast_var <= 0:
        raise ValueError("contrast has zero estimated variance under the supplied design.")

    # Compute the OLS contrast t statistic and p-value.
    se = np.sqrt(np.maximum(sigma2, 0.0) * contrast_var)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = estimate / se
    zero_se = se == 0.0
    t_stat[zero_se & (estimate == 0.0)] = 0.0
    perfect_effect = zero_se & (estimate != 0.0)
    t_stat[perfect_effect] = np.sign(estimate[perfect_effect]) * np.inf
    observed = np.abs(t_stat)
    pvals = 2.0 * _t_dist.sf(observed, df_resid)

    frame = _comparison_frame(
        n_genes,
        gene_names,
        observed,
        pvals,
        extra={"Estimate": estimate},
    )
    return frame[["Feature", "Statistic", "Estimate", "P_value", "P_adj"]]
