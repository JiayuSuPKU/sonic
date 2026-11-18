"""Public spectral and scalar comparison functions for comparator outputs.

Spectral L2 inference lives in ``_inference``; shared OLS fits live in
``_regression``. These functions validate user options and assemble results.
"""

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
    integration_tol=1e-6,
    random_state=None,
):
    """Validate binary inputs, select the statistic, and build one result table."""
    if statistic not in _AVAILABLE_STATISTICS:
        raise ValueError(
            f"Unknown statistic '{statistic}'. Options: {list(_AVAILABLE_STATISTICS)}."
        )
    if null not in _l2._NULL_OPTIONS:
        raise ValueError(f"Unknown null='{null}'. Options: {list(_l2._NULL_OPTIONS)}.")
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
    labels = np.unique(groups)
    if len(labels) != 2:
        raise ValueError(f"groups must contain exactly two distinct values, got {labels}.")
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
        if masked and int(min_resid_df) < 1:
            raise ValueError("min_resid_df must be >= 1.")
        if masked:
            matrix, columns = _lm._build_design_matrix(design, n_samples)
            vector = _lm._resolve_contrast(contrast, columns)
            inverse, rank, full_estimable = _lm._decompose_glm_design(matrix, vector)
            extra = {"n_obs": presence.sum(axis=0), "df_resid": np.zeros(n_genes, dtype=int)}
        else:
            matrix, _, vector, inverse, rank = (
                prepared_design
                if prepared_design is not None
                else _lm._prepare_glm_design(design, contrast, n_samples)
            )
            full_estimable = True

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
        else:
            x = matrix[sample_mask]
            inv, rank_gene, estimable = (
                (inverse, rank, full_estimable)
                if sample_mask.all()
                else _lm._decompose_glm_design(x, vector)
            )
            df_resid = count - rank_gene
            if masked:
                extra["df_resid"][genes] = df_resid
            scale = float(np.sum((vector @ inv) ** 2))
            if (
                df_resid < int(min_resid_df)
                or not estimable
                or not np.isfinite(scale)
                or scale <= 0
            ):
                continue
            n_terms = x.shape[1]

        rows = np.flatnonzero(sample_mask)
        for block in _l2._iter_feature_blocks(count, len(genes), n_bins, n_terms):
            cols = genes[block]
            values = spectra[rows[:, None], cols]
            if not np.isfinite(values).all():
                raise ValueError("Observed profiles must contain finite values.")
            response = _l2._prepare_profile_block(
                values, normalize_shape, log_transform=log_transform
            )
            if binary:
                t_stat, bins, effect = _welch_test(response[codes == 0], response[codes == 1])
            else:
                t_stat, bins, effect = _lm._test_glm_response(
                    response.reshape(count, -1), x, inv, vector, rank_gene
                )
            t_stat = t_stat.reshape(-1, n_bins)
            effect = effect.reshape(-1, n_bins)
            abs_t = np.abs(t_stat)
            bins = bins.reshape(-1, n_bins)
            # A constant response can still test a nonzero intercept contrast.
            informative = np.any(response != response[:1], axis=0) | (effect != 0)
            tail = cauchy_combine(
                bins,
                axis=-1,
                where=informative,
                weights=weights if freq_weights is not None else None,
            )
            observed[cols] = np.max(abs_t, axis=-1, where=weights > 0, initial=0.0)
            pvals[cols] = tail
            binwise["Estimate_per_bin"][cols] = effect
            binwise["T_per_bin"][cols] = t_stat
            binwise["P_value_per_bin"][cols] = bins
            selected = informative & (weights > 0) & np.isfinite(bins)
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
    null: str = "analytic",
    n_perm: int = 1000,
    max_exact_permutations: int = 10000,
    random_state: int | None = None,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
    *,
    covariance: str | None = None,
    integration_tol: float = 1e-6,
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
        ``'log_t_cauchy'`` combines per-bin Welch p-values on log profiles
        and also supports localizing the best bin by its smallest p-value
        among positive-weight bins.
        It requires two observations per arm and excludes constant bins.
    null : str
        Null calibration (default ``'analytic'``), selected by ``statistic``:

        - ``'log_l2'``: ``'analytic'`` (selected covariance and Liu tail),
          ``'rotation'`` (deterministic conditional integration), or
          ``'permutation'`` (shuffled binary labels).
        - ``'log_t_cauchy'``: ``'analytic'`` only (per-bin t tails and
          Cauchy combination).
    n_perm : int
        Number of sampled label permutations; default 1000. Ignored when
        all assignments are enumerated or another null is selected.
    max_exact_permutations : int
        Enumerate all binary label assignments when their number is at
        most this threshold; default 10000. Zero forces sampling.
    random_state : int, optional
        Seed for binary label permutations; ignored by other nulls.
    freq_weights : np.ndarray, optional
        Finite nonnegative weights of length ``n_bins``, normalized to sum
        to one. None gives uniform weights. Log t-Cauchy renormalizes weights
        over informative bins; zero weights exclude bins from combination and
        the statistic summary. Per-bin p-values remain unchanged. Choose
        weights independently of the observed signal.
    normalize_shape : bool
        If True, divide each sample/feature spectrum by its total power
        before logging to test spectral shape (DF-norm). Default False.
    covariance : str or None
        ``None`` (default) or ``'pooled'`` shares residual covariance across
        eligible genes. ``'per_gene'`` estimates it separately for each gene.
        An explicit choice requires an analytic L2 test.
    integration_tol : float
        Initial Davies absolute p-value error target for rotation; default 1e-6.
        Use at most 10,000 terms per call and one accuracy retry. Accepted
        integration also requires relative error budget <= 0.001. Unresolved
        tails use marked saddlepoint approximation. Ignored by other nulls.

    Returns
    -------
    pandas.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value`` and ``P_adj``.
        ``Estimate_per_bin`` gives the mean A minus mean B on natural-log profiles
        in original bin order, without frequency weights. For ``Statistic``, L2 reports the
        weighted RMS of these unweighted estimates, similar to an aggregated effect
        size. Log t-Cauchy reports the largest absolute t over positive-weight bins.
        It also adds signed ``T_per_bin`` and unadjusted ``P_value_per_bin``.
        ``Best_bin`` is the zero-based index with the smallest
        p-value among informative, positive-weight bins (first index on ties),
        or missing if none. Skipped features have NaN per-bin arrays.
        Feature-level significance does not establish the selected bin's significance.
        Rows are sorted by descending statistic; BH uses finite p-values.
        Analytic L2 results record ``null`` and ``covariance`` in ``attrs``.
        Rotation adds ``P_value_method`` (``davies``, ``saddlepoint``, ``beta``
        or ``exact``; missing for skipped rows) and ``P_value_error``. The latter
        is the accepted Davies absolute error budget, a roundoff estimate for
        beta, or zero at exact boundaries; it is NaN for saddlepoint/skipped
        rows and does not measure statistical uncertainty. ``attrs`` records
        ``method="davies"``, ``integration_tol``, ``integration_limit=10000``,
        ``n_saddlepoint`` and ``covariance=None``.

    Raises
    ------
    ValueError
        If input shapes or options are invalid, or the test cannot be fitted.
    RuntimeError
        If conditional-tail evaluation fails, including the approximation fallback.

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
        integration_tol=integration_tol,
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
    integration_tol: float = 1e-6,
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
        ``'log_t_cauchy'`` combines per-bin Welch p-values on log profiles
        and also supports localizing the best bin by its smallest p-value
        among positive-weight bins.
        It requires two observations per arm and excludes constant bins.
    null : str
        Null calibration (default ``'analytic'``), selected by ``statistic``:

        - ``'log_l2'``: ``'analytic'`` (selected covariance and Liu tail),
          ``'rotation'`` (deterministic conditional integration), or
          ``'permutation'`` (shuffled binary labels).
        - ``'log_t_cauchy'``: ``'analytic'`` only (per-bin t tails and
          Cauchy combination).
    n_perm : int
        Number of sampled label permutations; default 1000. Ignored when
        all assignments are enumerated or another null is selected.
    max_exact_permutations : int
        Enumerate all binary label assignments when their number is at
        most this threshold; default 10000. Zero forces sampling.
    random_state : int, optional
        Seed for binary label permutations; ignored by other nulls.
    min_samples_per_group : int
        Minimum observed samples per group for a masked binary test;
        default 2. Features below either threshold receive NaN results.
    freq_weights : np.ndarray, optional
        Finite nonnegative weights of length ``n_bins``, normalized to sum
        to one. None gives uniform weights. Log t-Cauchy renormalizes weights
        over informative bins; zero weights exclude bins from combination and
        the statistic summary. Per-bin p-values remain unchanged. Choose
        weights independently of the observed signal.
    normalize_shape : bool
        If True, divide each sample/feature spectrum by its total power
        before logging to test spectral shape (DF-norm). Default False.
    covariance : str or None
        ``None`` (default) or ``'pooled'`` shares residual covariance across
        eligible genes. ``'per_gene'`` estimates it separately for each gene.
        An explicit choice requires an analytic L2 test.
    integration_tol : float
        Initial Davies absolute p-value error target for rotation; default 1e-6.
        Use at most 10,000 terms per call and one accuracy retry. Accepted
        integration also requires relative error budget <= 0.001. Unresolved
        tails use marked saddlepoint approximation. Ignored by other nulls.

    Returns
    -------
    pandas.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value`` and ``P_adj``.
        ``Estimate_per_bin`` gives the mean A minus mean B on natural-log profiles
        in original bin order, without frequency weights. For ``Statistic``, L2 reports the
        weighted RMS of these unweighted estimates, similar to an aggregated effect
        size. Log t-Cauchy reports the largest absolute t over positive-weight bins.
        It also adds signed ``T_per_bin`` and unadjusted ``P_value_per_bin``.
        ``Best_bin`` is the zero-based index with the smallest
        p-value among informative, positive-weight bins (first index on ties),
        or missing if none. Skipped features have NaN per-bin arrays.
        Feature-level significance does not establish the selected bin's significance.
        Includes ``n_obs_A`` and ``n_obs_B``; skipped genes retain NaN results.
        Rows are sorted by descending statistic; BH uses finite p-values.
        Analytic L2 results record ``null`` and ``covariance`` in ``attrs``.
        Rotation adds ``P_value_method`` (``davies``, ``saddlepoint``, ``beta``
        or ``exact``; missing for skipped rows) and ``P_value_error``. The latter
        is the accepted Davies absolute error budget, a roundoff estimate for
        beta, or zero at exact boundaries; it is NaN for saddlepoint/skipped
        rows and does not measure statistical uncertainty. ``attrs`` records
        ``method="davies"``, ``integration_tol``, ``integration_limit=10000``,
        ``n_saddlepoint`` and ``covariance=None``.

    Raises
    ------
    ValueError
        If input shapes or options are invalid, or the test cannot be fitted.
    RuntimeError
        If conditional-tail evaluation fails, including the approximation fallback.

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
    null: str = "analytic",
    integration_tol: float = 1e-6,
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
        to one. None gives uniform weights. Log t-Cauchy renormalizes weights
        over informative bins; zero weights exclude bins from combination and
        the statistic summary. Per-bin p-values remain unchanged. Choose
        weights independently of the observed signal.
    normalize_shape : bool
        If True, divide each sample/feature spectrum by its total power
        before logging to test spectral shape (DF-norm). Default False.
    statistic : str
        ``'log_l2'`` (default) tests the weighted L2 contrast of log spectra.
        ``'log_t_cauchy'`` combines per-bin OLS contrast t-test p-values
        and also supports localizing the best bin by its smallest p-value
        among positive-weight bins.
    covariance : str or None
        ``None`` (default) or ``'pooled'`` shares residual covariance across
        eligible genes. ``'per_gene'`` estimates it separately for each gene.
        An explicit choice requires an analytic L2 test.
    null : str
        Null calibration (default ``'analytic'``), selected by ``statistic``:

        - ``'log_l2'``: ``'analytic'`` (selected covariance and Liu tail),
          ``'rotation'`` (deterministic conditional integration).
        - ``'log_t_cauchy'``: ``'analytic'`` only (per-bin t tails and
          Cauchy combination).
    integration_tol : float
        Initial Davies absolute p-value error target for rotation; default 1e-6.
        Use at most 10,000 terms per call and one accuracy retry. Accepted
        integration also requires relative error budget <= 0.001. Unresolved
        tails use marked saddlepoint approximation. Ignored by other nulls.

    Returns
    -------
    pandas.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value`` and ``P_adj``.
        ``Estimate_per_bin`` gives the mean A minus mean B on natural-log profiles
        in original bin order, without frequency weights. For ``Statistic``, L2 reports the
        weighted RMS of these unweighted estimates, similar to an aggregated effect
        size. Log t-Cauchy reports the largest absolute t over positive-weight bins.
        It also adds signed ``T_per_bin`` and unadjusted ``P_value_per_bin``.
        ``Best_bin`` is the zero-based index with the smallest p-value among
        informative, positive-weight bins (first index on ties), or missing
        if none. Skipped features have NaN per-bin arrays.
        Feature-level significance does not establish the selected bin's significance.
        Rows are sorted by descending statistic; BH uses finite p-values.
        Analytic L2 results record ``null`` and ``covariance`` in ``attrs``.
        Rotation adds ``P_value_method`` (``davies``, ``saddlepoint``, ``beta``
        or ``exact``; missing for skipped rows) and ``P_value_error``. The latter
        is the accepted Davies absolute error budget, a roundoff estimate for
        beta, or zero at exact boundaries; it is NaN for saddlepoint/skipped
        rows and does not measure statistical uncertainty. ``attrs`` records
        ``method="davies"``, ``integration_tol``, ``integration_limit=10000``,
        ``n_saddlepoint`` and ``covariance=None``.

    Raises
    ------
    ValueError
        If input shapes or options are invalid, or the test cannot be fitted.
    RuntimeError
        If conditional-tail evaluation fails, including the approximation fallback.
    TypeError
        If the design or contrast has an unsupported type.

    Notes
    -----
    For calibration assumptions and zero-variance behavior, see
    :doc:`/guides/multisample`.
    """
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
    null: str = "analytic",
    integration_tol: float = 1e-6,
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
        to one. None gives uniform weights. Log t-Cauchy renormalizes weights
        over informative bins; zero weights exclude bins from combination and
        the statistic summary. Per-bin p-values remain unchanged. Choose
        weights independently of the observed signal.
    normalize_shape : bool
        If True, divide each sample/feature spectrum by its total power
        before logging to test spectral shape (DF-norm). Default False.
    min_resid_df : int
        Minimum residual degrees of freedom for a masked GLM test;
        default 1. Ineligible or unestimable features receive NaN results.
    statistic : str
        ``'log_l2'`` (default) tests the weighted L2 contrast of log spectra.
        ``'log_t_cauchy'`` combines per-bin OLS contrast t-test p-values
        and also supports localizing the best bin by its smallest p-value
        among positive-weight bins.
    covariance : str or None
        ``None`` (default) or ``'pooled'`` shares residual covariance across
        eligible genes. ``'per_gene'`` estimates it separately for each gene.
        An explicit choice requires an analytic L2 test.
    null : str
        Null calibration (default ``'analytic'``), selected by ``statistic``:

        - ``'log_l2'``: ``'analytic'`` (selected covariance and Liu tail),
          ``'rotation'`` (deterministic conditional integration).
        - ``'log_t_cauchy'``: ``'analytic'`` only (per-bin t tails and
          Cauchy combination).
    integration_tol : float
        Initial Davies absolute p-value error target for rotation; default 1e-6.
        Use at most 10,000 terms per call and one accuracy retry. Accepted
        integration also requires relative error budget <= 0.001. Unresolved
        tails use marked saddlepoint approximation. Ignored by other nulls.

    Returns
    -------
    pandas.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value`` and ``P_adj``.
        ``Estimate_per_bin`` gives the mean A minus mean B on natural-log profiles
        in original bin order, without frequency weights. For ``Statistic``, L2 reports the
        weighted RMS of these unweighted estimates, similar to an aggregated effect
        size. Log t-Cauchy reports the largest absolute t over positive-weight bins.
        It also adds signed ``T_per_bin`` and unadjusted ``P_value_per_bin``.
        ``Best_bin`` is the zero-based index with the smallest p-value among
        informative, positive-weight bins (first index on ties), or missing
        if none. Skipped features have NaN per-bin arrays.
        Feature-level significance does not establish the selected bin's significance.
        Includes ``n_obs`` and ``df_resid``; skipped genes retain NaN results.
        Rows are sorted by descending statistic; BH uses finite p-values.
        Analytic L2 results record ``null`` and ``covariance`` in ``attrs``.
        Rotation adds ``P_value_method`` (``davies``, ``saddlepoint``, ``beta``
        or ``exact``; missing for skipped rows) and ``P_value_error``. The latter
        is the accepted Davies absolute error budget, a roundoff estimate for
        beta, or zero at exact boundaries; it is NaN for saddlepoint/skipped
        rows and does not measure statistical uncertainty. ``attrs`` records
        ``method="davies"``, ``integration_tol``, ``integration_limit=10000``,
        ``n_saddlepoint`` and ``covariance=None``.

    Raises
    ------
    ValueError
        If input shapes or options are invalid, or the test cannot be fitted.
    RuntimeError
        If conditional-tail evaluation fails, including the approximation fallback.
    TypeError
        If the design or contrast has an unsupported type.

    Notes
    -----
    For calibration assumptions and zero-variance behavior, see
    :doc:`/guides/multisample`.
    """
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
    t_stat, pvals, estimate = _lm._test_glm_response(
        values, design_matrix, design_inverse, contrast_vector, rank
    )
    observed = np.abs(t_stat)

    frame = _comparison_frame(
        n_genes,
        gene_names,
        observed,
        pvals,
        extra={"Estimate": estimate},
    )
    return frame[["Feature", "Statistic", "Estimate", "P_value", "P_adj"]]
