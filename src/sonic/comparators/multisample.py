"""Statistical comparison primitives for comparator outputs.

Spectral feature construction lives in :mod:`sonic.comparators.features`.
Spectrum normalization lives in :mod:`sonic.comparators.normalization`.
This module consumes normalized or raw per-sample arrays and provides:

- ``compare_two_groups`` and ``compare_two_groups_masked`` for binary labels.
- ``compare_glm`` and ``compare_glm_masked`` for design-matrix contrasts.
- Scalar DC-expression companions ``compare_two_groups_scalar`` and
  ``compare_glm_scalar`` with analytic t-distribution tests.

The public comparator classes in :mod:`sonic.comparators` wrap these
array-level functions for AnnData and SpatialData inputs.

Analytic log-L2 comparisons accumulate one pooled covariance in gene blocks,
using a fixed 32 MiB scratch-space target (at least one gene). Inputs, returned
diagnostics and the cross-bin covariance are excluded from that target.
Shape normalization and float64 conversion happen within each block. Genes
with identical observation masks share their design decomposition; p-values
and BH correction still use the complete tested gene set.
"""

from __future__ import annotations

import itertools
import logging
import math
import warnings
from collections.abc import Sequence
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp  # noqa: F401  (exposed for downstream calibration tests)
from scipy.stats import t as _t_dist

from sonic.comparators.normalization import _normalize_shape_apply
from sonic.statistics import apply_bh_correction, cauchy_combine, liu_sf

__all__ = [
    "compare_two_groups",
    "compare_two_groups_masked",
    "compare_two_groups_scalar",
    "compare_glm",
    "compare_glm_masked",
    "compare_glm_scalar",
]

logger = logging.getLogger(__name__)

_AVAILABLE_STATISTICS = ("log_l2", "welch_t_cauchy")
_NULL_OPTIONS = ("permutation", "analytic")

# Bound analytic-comparison scratch space, allowing for eight float64 arrays.
# Inputs, returned diagnostics, and the n_bins² covariance are excluded; at
# least one gene is processed even if it alone exceeds this workspace target.
_COMPARISON_WORKSPACE_BYTES = 32 * (1 << 20)


def _comparison_blocks(n_samples, n_genes, n_bins, n_terms=0):
    """Yield gene slices without materializing whole-panel log/residual arrays."""
    step = max(1, _COMPARISON_WORKSPACE_BYTES // (8 * 8 * max(n_samples, n_terms, 1) * n_bins))
    for start in range(0, n_genes, step):
        yield slice(start, min(start + step, n_genes))


def _log_spectra_block(spectra, normalize_shape):
    """Make an owned float64 log block, preserving the caller's spectra."""
    if normalize_shape:
        block = _normalize_shape_apply(spectra)
    else:
        block = np.array(spectra, dtype=np.float64, copy=True)
    np.maximum(block, 1e-12, out=block)
    np.log(block, out=block)
    return block


def _presence_groups(presence):
    """Yield sample masks and gene indices, grouping complete masks, not counts."""
    if presence.shape[1] == 0:
        return
    _, inverse = np.unique(np.packbits(presence, axis=0).T, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    for genes in np.split(order, np.flatnonzero(np.diff(inverse[order])) + 1):
        yield presence[:, genes[0]], genes


class _AnalyticNullMetadata(TypedDict, total=False):
    """Optional metadata, using the TypedDict syntax supported by Python 3.10."""

    n_obs_A: int | np.ndarray
    n_obs_B: int | np.ndarray
    design_columns: list[str]
    contrast_vector: np.ndarray
    beta: np.ndarray
    n_obs: int | np.ndarray


class _AnalyticNullState(_AnalyticNullMetadata):
    """Internal contract shared by analytic DF tests and covariance diagnostics.

    The common fields below are returned by every analytic null estimator and
    by ``Comparator.estimate_null_covariance``. They are intentionally a little
    richer than the minimum needed for p-values: ``compare_*`` needs only
    ``observed``, ``eigenvalues``, and ``eligible``, while public diagnostics
    also need the pooled covariance, weights, effective rank, and residual
    degrees-of-freedom metadata.

    Common fields
    -------------
    mode
        ``"two_group"`` for binary labels or ``"glm"`` for design-matrix
        contrasts.
    masked
        Whether gene-specific missingness was used.
    observed
        Per-gene log-L2 statistic, shape ``(n_genes,)``. Masked helpers use
        ``NaN`` for ineligible genes.
    sigma_log
        Pooled residual covariance of log-spectra, shape ``(n_bins, n_bins)``.
    freq_weights
        Normalized non-negative bin weights, shape ``(n_bins,)``.
    weighted_cov
        ``W^1/2 sigma_log W^1/2``, the covariance whose eigenstructure
        controls the weighted quadratic form.
    eigenvalues
        Liu mixture eigenvalues after applying the contrast variance scale.
        Unmasked helpers return shape ``(n_bins,)``; masked helpers return
        ``(n_genes, n_bins)`` with ``NaN`` rows for skipped genes.
    effective_rank
        Participation-ratio rank of the unscaled ``weighted_cov`` eigenvalues.
    contrast_scale
        Scalar contrast variance for unmasked paths; per-gene vector for
        masked paths.
    df_resid
        Residual degrees of freedom, scalar for unmasked paths and per-gene
        vector for masked paths.
    eligible
        Boolean per-gene mask identifying rows that receive finite p-values.

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
    beta
        Fitted log-spectrum coefficients, shape ``(n_terms, n_genes, n_bins)``.
        Masked paths keep ``NaN`` coefficients for ineligible genes.
    n_obs
        Number of observed samples. Unmasked paths return a scalar; masked
        paths return a per-gene vector.
    """

    mode: Literal["two_group", "glm"]
    masked: bool
    observed: np.ndarray
    sigma_log: np.ndarray
    freq_weights: np.ndarray
    weighted_cov: np.ndarray
    eigenvalues: np.ndarray
    effective_rank: float
    contrast_scale: float | np.ndarray
    df_resid: int | np.ndarray
    eligible: np.ndarray


# ---------------------------------------------------------------------------
# Test statistics
# ---------------------------------------------------------------------------


def _resolve_freq_weights(freq_weights: np.ndarray | None, n_bins: int) -> np.ndarray:
    """Validate / normalize frequency-bin weights; return a length-``n_bins`` array.

    Passing None yields uniform weights ``1/n_bins`` - recovering the unweighted
    statistic. Any other input is cast to ``float``, required to be
    non-negative and not all-zero, and rescaled to sum-1. Non-uniform
    weights are how users express a kernel-like frequency preference (e.g.,
    low-pass polynomial vs exponential decay) inside the spectral distance.
    """
    if freq_weights is None:
        return np.full(n_bins, 1.0 / n_bins)
    w = np.asarray(freq_weights, dtype=float).ravel()
    if w.shape != (n_bins,):
        raise ValueError(f"freq_weights must have length n_bins={n_bins}, got shape {w.shape}.")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("freq_weights must be finite and non-negative.")
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("freq_weights must have a positive finite sum.")
    return w / total


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


# ---------------------------------------------------------------------------
# Permutation engine
# ---------------------------------------------------------------------------


def _exchangeable_group_labels(
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


def _run_statistic_with_perm(
    stat_name: str,
    spectra: np.ndarray,
    group_codes: np.ndarray,
    perm_labels: np.ndarray,
    *,
    freq_weights: np.ndarray | None = None,
    is_exact: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute observed log-L2 statistics and permutation tail probabilities.

    ``perm_labels`` is a ``(n_perm_used, n_samples)`` matrix of group
    relabellings (as produced by :func:`_exchangeable_group_labels`).
    Log spectra and weights are prepared once. Only per-gene exceedance
    counts are retained; no permutations-by-genes null matrix is allocated.
    Exhaustive enumeration uses the exact tail; sampled nulls get +1 correction.
    """
    if stat_name != "log_l2":
        raise ValueError(f"Permutation statistic must be 'log_l2', got {stat_name!r}.")
    if not np.all(np.isfinite(spectra)):
        raise ValueError("Permutation tests require finite spectra for observed samples.")
    uniq = np.unique(group_codes)
    a_val = uniq[0]
    log_spectra = np.maximum(spectra, 1e-12)
    np.log(log_spectra, out=log_spectra)
    log_spectra -= log_spectra[:1].copy()
    weights = _resolve_freq_weights(freq_weights, spectra.shape[-1])

    def statistic(a: np.ndarray) -> np.ndarray:
        diff = log_spectra[a].mean(axis=0) - log_spectra[~a].mean(axis=0)
        return np.sqrt(np.sum(weights * diff**2, axis=-1))

    observed = statistic(group_codes == a_val)
    exceedances = np.zeros(spectra.shape[1], dtype=np.int64)
    for labels in perm_labels:
        exceedances += statistic(labels == a_val) >= observed
    correction = 0.0 if is_exact else 1.0
    pvals = (exceedances + correction) / (len(perm_labels) + correction)
    return observed, pvals


# ---------------------------------------------------------------------------
# Analytic null for welch t and log_l2 (mixture-χ² tail)
# ---------------------------------------------------------------------------


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


# Minimum residual df below which we issue a calibration warning for the
# analytic path. At df=1 the σ̂² estimator has a 100% relative noise
# (var = 2σ⁴), at df=2 it's 50%; both can produce occasional
# anti-conservative spikes.
_ANALYTIC_MIN_DF_NO_WARN = 3


def _maybe_warn_small_df_analytic(df_resid: int) -> None:
    """Warn the user when running the analytic path at very small residual df.

    Suppress with ``warnings.filterwarnings('ignore',
    message="log_l2 + null='analytic'")`` if you accept the calibration risk.
    """
    if df_resid < _ANALYTIC_MIN_DF_NO_WARN:
        rel_noise_pct = 100.0 * (2.0 / max(df_resid, 1)) ** 0.5
        warnings.warn(
            f"log_l2 + null='analytic' at residual df={df_resid}: "
            f"Variance estimator σ̂² has ~{rel_noise_pct:.0f}% relative noise, "
            f"so the analytic null may be anti-conservative on a per-test basis. "
            f"For n_a + n_b ≤ 4, prefer statistic='welch_t_cauchy' for stricter "
            f"calibration (at the cost of some sensitivity).",
            UserWarning,
            stacklevel=3,
        )


def _log_l2_analytic_pvalues(
    statistic: np.ndarray,
    lambs: np.ndarray,
) -> np.ndarray:
    """Analytic p-values for ``log_l2`` via Liu's mixture-χ² tail.

    ``statistic`` is the ``(n_genes,)`` log-L2 distance between group means
    (square root of the quadratic form ``D'WD``).
    Squaring it here gives the H₀ statistic distributed as ``Σ_k λ_k χ²_1``,
    which Liu's approximation handles directly.
    """
    statistic_sq = np.asarray(statistic, dtype=float) ** 2
    return np.asarray(liu_sf(statistic_sq, lambs), dtype=float)


def _effective_rank_from_eigenvalues(eigenvalues: np.ndarray) -> float:
    """Participation-ratio rank from covariance eigenvalues."""
    eig = np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)
    total = float(eig.sum())
    if total <= 0.0:
        return float("nan")
    return float((total * total) / np.sum(eig * eig))


def _log_l2_pvalues_from_state(state: _AnalyticNullState) -> np.ndarray:
    """Evaluate Liu-tail p-values from shared analytic null state.

    The state must already contain the observed log-L2 statistic and the
    contrast-scaled Liu eigenvalues produced by one of the
    ``_estimate_*_null_covariance`` helpers. Unmasked states carry one
    eigenvalue vector reused for every gene; masked states carry a per-gene
    eigenvalue matrix and an ``eligible`` mask so skipped genes remain ``NaN``.
    Genes with the same contrast scale share one Liu fit.
    """
    observed = np.asarray(state["observed"], dtype=float)
    eigenvalues = np.asarray(state["eigenvalues"], dtype=float)
    if eigenvalues.ndim == 1:
        return _log_l2_analytic_pvalues(observed, eigenvalues)

    pvals = np.full(observed.shape, np.nan, dtype=float)
    indices = np.flatnonzero(state["eligible"])
    if indices.size:
        scales = np.asarray(state["contrast_scale"])[indices]
        order = np.argsort(scales)
        boundaries = np.flatnonzero(np.diff(scales[order])) + 1
        for group in np.split(indices[order], boundaries):
            pvals[group] = liu_sf(observed[group] ** 2, eigenvalues[group[0]])
    return pvals


def _comparison_frame(
    n_genes: int,
    gene_names: Sequence[str] | None,
    observed: np.ndarray,
    pvals: np.ndarray,
    *,
    extra: dict[str, Sequence[Any] | np.ndarray] | None = None,
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
    return df.sort_values("Statistic", ascending=False, na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Generalized GLM analytic path for log_l2 (design matrix + contrast)
# ---------------------------------------------------------------------------


def _build_design_matrix(
    design: pd.DataFrame | np.ndarray, n_samples: int
) -> tuple[np.ndarray, list[str]]:
    """Convert ``design`` to a ``(n_samples, p)`` numeric matrix + column labels.

    - ``np.ndarray`` of shape ``(n_samples, p)`` is accepted as-is; columns
      are labelled ``x0, x1, ...``. The caller is responsible for including
      an intercept column if desired.
    - ``pd.DataFrame``: encoded via :func:`patsy.dmatrix`, adding an intercept
      and one-hot encoding categoricals (Treatment contrast against the first
      level). Column names are treated literally, including punctuation and
      spaces. Missing values raise rather than dropping sample rows. Encoded
      labels must be unique, including the generated ``Intercept`` column.
      If patsy is not installed, raise ``ImportError`` with an install hint.
    """
    if isinstance(design, np.ndarray):
        design_matrix = np.asarray(design, dtype=float)
        if design_matrix.ndim != 2 or design_matrix.shape[0] != n_samples:
            raise ValueError(
                f"design ndarray must be (n_samples, p) = ({n_samples}, p), "
                f"got {design_matrix.shape}."
            )
        return design_matrix, [f"x{i}" for i in range(design_matrix.shape[1])]

    if not isinstance(design, pd.DataFrame):
        raise TypeError(
            f"design must be a numpy ndarray or pandas DataFrame, " f"got {type(design).__name__}."
        )
    if len(design) != n_samples:
        raise ValueError(
            f"design DataFrame length {len(design)} does not match " f"n_samples={n_samples}."
        )
    try:
        import patsy
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Building a design matrix from a pandas DataFrame requires patsy. "
            "Install via `pip install patsy` or pass a pre-built numpy "
            "design matrix instead."
        ) from e
    # Use safe formula identifiers even for spaces, punctuation, keywords or
    # columns named like Patsy builtins. Restore user-facing contrast names.
    aliases = [f"_sonic_column_{i}" for i in range(len(design.columns))]
    formula = "~ " + " + ".join(aliases)
    design_matrix = patsy.dmatrix(
        formula, design.set_axis(aliases, axis=1), return_type="dataframe", NA_action="raise"
    )
    names = list(map(str, design.columns))
    columns = list(design_matrix.columns)
    for alias, name in zip(aliases, names, strict=True):
        term_slice = design_matrix.design_info.term_name_slices[alias]
        columns[term_slice] = [name + c[len(alias) :] for c in columns[term_slice]]
    if "Intercept" in names or len(set(names)) != len(names) or len(set(columns)) != len(columns):
        raise ValueError(
            "Design column names are ambiguous after encoding (including the added "
            "'Intercept'); rename conflicting columns or pass a numeric design matrix."
        )
    return design_matrix.to_numpy().astype(float), columns


def _resolve_contrast(
    contrast: str | dict[str, float] | np.ndarray, design_columns: Sequence[str]
) -> np.ndarray:
    """Map the user-supplied contrast spec to a length-``p`` numeric vector.

    - ``str``: column name in the design matrix. Auto-resolves patsy
      treatment-coded factors (e.g., ``"genotype"`` → ``"genotype[T.TG]"``)
      when there is exactly one matching column. For multi-level factors
      with >1 matching column this raises ``ValueError`` (multi-DOF
      contrasts are out of scope; pass an explicit dict or ndarray).
    - ``dict``: maps column-name → coefficient; missing columns get 0.
    - ``ndarray`` of shape ``(p,)``: used as-is.
    """
    n_terms = len(design_columns)
    if isinstance(contrast, np.ndarray):
        contrast_vector = np.asarray(contrast, dtype=float)
        if contrast_vector.shape != (n_terms,):
            raise ValueError(
                f"contrast ndarray must have length p={n_terms}, got shape {contrast_vector.shape}."
            )
        return contrast_vector
    if isinstance(contrast, str):
        if contrast in design_columns:
            target = contrast
        else:
            matches = [col for col in design_columns if col.startswith(contrast + "[T.")]
            if not matches:
                raise ValueError(
                    f"Contrast '{contrast}' not found in design columns " f"{list(design_columns)}."
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Contrast '{contrast}' is ambiguous — matches "
                    f"{matches}. Pass an explicit dict or ndarray (multi-DOF "
                    f"contrasts are out of scope)."
                )
            target = matches[0]
        contrast_vector = np.zeros(n_terms, dtype=float)
        contrast_vector[list(design_columns).index(target)] = 1.0
        return contrast_vector
    if isinstance(contrast, dict):
        contrast_vector = np.zeros(n_terms, dtype=float)
        for k, v in contrast.items():
            if k not in design_columns:
                raise ValueError(
                    f"Contrast key '{k}' not in design columns " f"{list(design_columns)}."
                )
            contrast_vector[list(design_columns).index(k)] = float(v)
        return contrast_vector
    raise TypeError(f"Contrast must be a str, dict, or ndarray; got {type(contrast).__name__}.")


def _decompose_glm_design(
    design_matrix: np.ndarray, contrast_vector: np.ndarray
) -> tuple[np.ndarray, int, bool]:
    """Use one SVD for the OLS inverse, rank, and contrast estimability.

    Forming X'X squares the condition number. Check the contrast against
    the orthonormal row basis directly, without multiplying X's inverse by X.
    Normalize the contrast for this check so its units cannot change estimability.
    """
    if not np.isfinite(design_matrix).all() or not np.isfinite(contrast_vector).all():
        raise ValueError("Design matrix and contrast must be finite.")
    contrast_size = np.max(np.abs(contrast_vector), initial=0.0)
    if contrast_size == 0:
        raise ValueError("contrast must be nonzero.")
    u, singular, vt = np.linalg.svd(design_matrix, full_matrices=False)
    cutoff = np.finfo(float).eps * max(design_matrix.shape) * singular[0]
    rank = int(np.count_nonzero(singular > cutoff))
    rows = vt[:rank]
    inverse = (rows.T / singular[:rank]) @ u[:, :rank].T
    unit_contrast = contrast_vector / contrast_size
    estimable = np.allclose(rows.T @ (rows @ unit_contrast), unit_contrast, rtol=1e-7, atol=1e-10)
    return inverse, rank, bool(estimable)


def _fit_glm_response(
    response: np.ndarray,
    design: np.ndarray,
    inverse: np.ndarray,
    contrast: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit OLS after removing any representable constant response offset.

    Compute residuals and contrasts before restoring the offset to beta, so
    an intercept cannot leak roundoff into a zero group effect. Preserve
    intercept contrasts and designs that do not span a constant response.
    """
    constant = np.flatnonzero(np.all(design == design[:1], axis=0) & (design[0] != 0))
    offset_beta = np.zeros(design.shape[1])
    if constant.size and rank == design.shape[1]:
        j = constant[0]
        offset_beta[j] = 1.0 / design[0, j]
        offset_effect = contrast[j] / design[0, j]
    else:
        # Group indicators or aliased columns can span an intercept implicitly.
        # Use the retained SVD subspace so rank truncation is respected.
        offset_beta = inverse.sum(axis=1)
        roundoff = np.finfo(float).eps * max(design.shape)
        if not np.allclose(design @ offset_beta, 1.0, rtol=0.0, atol=roundoff):
            offset_beta.fill(0.0)
        offset_effect = float(contrast @ offset_beta)
        tolerance = roundoff * np.max(np.abs(contrast)) * np.sum(np.abs(offset_beta))
        if abs(offset_effect) <= tolerance:
            offset_effect = 0.0
    offset = response[0] if np.any(offset_beta) else np.zeros(response.shape[1])
    residual = response - offset
    beta = inverse @ residual
    residual -= design @ beta
    effect = contrast @ beta + offset_effect * offset
    beta += offset_beta[:, None] * offset
    return beta, residual, effect


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


def _two_group_log_block(log_a, log_b, weights):
    """Consume owned log blocks, returning statistics and residual cross-products."""
    offset = log_a[0] - log_b[0]
    log_a -= log_a[:1].copy()
    log_b -= log_b[:1].copy()
    mean_a = log_a.mean(axis=0)
    mean_b = log_b.mean(axis=0)
    observed = np.sqrt(np.sum(weights * (offset + (mean_a - mean_b)) ** 2, axis=-1))
    log_a -= mean_a
    log_b -= mean_b
    a = log_a.reshape(-1, log_a.shape[-1])
    b = log_b.reshape(-1, log_b.shape[-1])
    return observed, a.T @ a + b.T @ b


def _estimate_two_group_null_covariance(
    spectra: np.ndarray,
    groups: np.ndarray,
    *,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
) -> _AnalyticNullState:
    """Compute analytic null state for unmasked two-group ``log_l2`` tests.

    This is the single implementation of the unmasked binary analytic null.
    It computes the observed per-gene weighted log-L2 statistic, the pooled
    within-group covariance of residual log-spectra, and the contrast-scaled
    Liu eigenvalues used by both :func:`compare_two_groups` and
    :meth:`sonic.comparators.base._ComparatorBase.estimate_null_covariance`.
    """
    spectra = np.asarray(spectra)
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D (n_samples, n_genes, n_bins), got {spectra.shape}.")
    n_samples, n_genes, n_bins = spectra.shape
    groups = np.asarray(groups)
    if groups.shape != (n_samples,):
        raise ValueError(f"groups shape {groups.shape} does not match n_samples={n_samples}.")
    uniq = np.unique(groups)
    if uniq.size != 2:
        raise ValueError(f"groups must contain exactly two distinct values, got {uniq}.")
    group_codes = (groups == uniq[1]).astype(int)

    group_a_mask = group_codes == 0
    n_a = int(group_a_mask.sum())
    n_b = int((~group_a_mask).sum())
    df_resid = n_a + n_b - 2
    if df_resid <= 0:
        raise ValueError("Analytic two-group tests require positive residual degrees of freedom.")
    weights = _resolve_freq_weights(freq_weights, n_bins)

    observed = np.empty(n_genes)
    sigma_log = np.zeros((n_bins, n_bins))
    for block in _comparison_blocks(n_samples, n_genes, n_bins):
        log_a = _log_spectra_block(spectra[group_a_mask, block], normalize_shape)
        log_b = _log_spectra_block(spectra[~group_a_mask, block], normalize_shape)
        observed[block], covariance = _two_group_log_block(log_a, log_b, weights)
        sigma_log += covariance
    sigma_log /= n_genes * df_resid
    _maybe_warn_small_df_analytic(df_resid)

    # Compute eigenvalues of the weighted covariance and contrast scale for p-value calculation.
    sqrt_weights = np.sqrt(weights)
    weighted_cov = sqrt_weights[:, None] * sigma_log * sqrt_weights[None, :]
    weighted_cov_eigenvalues = np.maximum(np.linalg.eigvalsh(weighted_cov), 0.0)
    contrast_scale = (1.0 / max(n_a, 1)) + (1.0 / max(n_b, 1))
    eigenvalues = weighted_cov_eigenvalues * contrast_scale
    return {
        "mode": "two_group",
        "masked": False,
        "observed": observed,
        "sigma_log": sigma_log,
        "freq_weights": weights,
        "weighted_cov": weighted_cov,
        "eigenvalues": eigenvalues,
        "effective_rank": _effective_rank_from_eigenvalues(weighted_cov_eigenvalues),
        "contrast_scale": contrast_scale,
        "df_resid": df_resid,
        "n_obs_A": n_a,
        "n_obs_B": n_b,
        "eligible": np.ones(n_genes, dtype=bool),
    }


def _estimate_two_group_masked_null_covariance(  # noqa: C901
    spectra: np.ndarray,
    groups: np.ndarray,
    presence: np.ndarray,
    *,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
    min_samples_per_group: int = 2,
) -> _AnalyticNullState:
    """Compute analytic null state for masked two-group ``log_l2`` tests.

    Each gene contributes only samples where ``presence[:, gene]`` is true.
    Eligible genes contribute residual cross-products to one pooled covariance;
    their observed statistics and eigenvalues are then computed with their own
    ``1 / n_A + 1 / n_B`` contrast scale. Ineligible genes retain ``NaN`` in
    ``observed``, ``contrast_scale``, and ``eigenvalues``.
    """
    if int(min_samples_per_group) < 2:
        raise ValueError(f"min_samples_per_group must be >= 2, got {min_samples_per_group}.")
    spectra = np.asarray(spectra)
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D, got {spectra.shape}.")
    n_samples, n_genes, n_bins = spectra.shape
    presence = np.asarray(presence, dtype=bool)
    if presence.shape != (n_samples, n_genes):
        raise ValueError(
            f"presence shape {presence.shape} != (n_samples, n_genes) = "
            f"({n_samples}, {n_genes})."
        )
    groups = np.asarray(groups)
    if groups.shape != (n_samples,):
        raise ValueError(f"groups shape {groups.shape} does not match n_samples={n_samples}.")
    uniq = np.unique(groups)
    if uniq.size != 2:
        raise ValueError("groups must contain exactly two distinct values.")
    group_codes = (groups == uniq[1]).astype(int)

    group_a_mask = group_codes == 0
    weights = _resolve_freq_weights(freq_weights, n_bins)
    sigma_acc = np.zeros((n_bins, n_bins), dtype=np.float64)
    pooled_df = 0
    observed = np.full(n_genes, np.nan, dtype=float)
    contrast_scale = np.full(n_genes, np.nan, dtype=float)
    eligible = np.zeros(n_genes, dtype=bool)
    n_obs_a = np.zeros(n_genes, dtype=int)
    n_obs_b = np.zeros(n_genes, dtype=int)
    df_resid = np.zeros(n_genes, dtype=int)

    for sample_mask, genes in _presence_groups(presence):
        idx_a = np.flatnonzero(group_a_mask & sample_mask)
        idx_b = np.flatnonzero(~group_a_mask & sample_mask)
        n_a, n_b = len(idx_a), len(idx_b)
        n_obs_a[genes], n_obs_b[genes] = n_a, n_b
        df_gene = max(n_a + n_b - 2, 1)
        df_resid[genes] = df_gene
        if n_a < min_samples_per_group or n_b < min_samples_per_group:
            continue
        contrast_scale[genes] = 1.0 / n_a + 1.0 / n_b
        eligible[genes] = True
        pooled_df += df_gene * len(genes)
        for block in _comparison_blocks(n_a + n_b, len(genes), n_bins):
            cols = genes[block]
            log_a = _log_spectra_block(spectra[idx_a[:, None], cols], normalize_shape)
            log_b = _log_spectra_block(spectra[idx_b[:, None], cols], normalize_shape)
            observed[cols], covariance = _two_group_log_block(log_a, log_b, weights)
            sigma_acc += covariance

    if pooled_df == 0:
        raise ValueError(
            "estimate_null_covariance: no genes meet "
            f"min_samples_per_group={min_samples_per_group} per arm. "
            "Cannot estimate the pooled covariance."
        )

    if eligible.any():
        _maybe_warn_small_df_analytic(int(np.median(df_resid[eligible])))

    # Compute covariance-dependent eigenvalues after the shared covariance is known.
    sqrt_weights = np.sqrt(weights)
    sigma_log = sigma_acc / pooled_df
    weighted_cov = sqrt_weights[:, None] * sigma_log * sqrt_weights[None, :]
    weighted_cov_eigenvalues = np.maximum(np.linalg.eigvalsh(weighted_cov), 0.0)
    eigenvalues = np.full((n_genes, n_bins), np.nan, dtype=float)
    eigenvalues[eligible] = contrast_scale[eligible, None] * weighted_cov_eigenvalues[None, :]

    return {
        "mode": "two_group",
        "masked": True,
        "observed": observed,
        "sigma_log": sigma_log,
        "freq_weights": weights,
        "weighted_cov": weighted_cov,
        "eigenvalues": eigenvalues,
        "effective_rank": _effective_rank_from_eigenvalues(weighted_cov_eigenvalues),
        "contrast_scale": contrast_scale,
        "df_resid": df_resid,
        "n_obs_A": n_obs_a,
        "n_obs_B": n_obs_b,
        "eligible": eligible,
    }


def _estimate_glm_null_covariance(
    spectra: np.ndarray,
    design: pd.DataFrame | np.ndarray,
    contrast: str | dict[str, float] | np.ndarray,
    *,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
) -> _AnalyticNullState:
    """Compute analytic null state for unmasked GLM ``log_l2`` tests.

    The design matrix is fit once against all flattened ``(gene, bin)``
    log-spectrum responses. The returned state includes the per-gene contrast
    effect converted to log-L2 ``observed`` values, the pooled residual
    covariance across genes, and the single contrast variance scale
    ``c'(X'X)^+c`` applied to the Liu eigenvalues.
    """
    spectra = np.asarray(spectra)
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D (n_samples, n_genes, n_bins), got {spectra.shape}.")
    n_samples, n_genes, n_bins = spectra.shape

    # Resolve the design matrix and contrast vector.
    design_matrix, design_columns = _build_design_matrix(design, n_samples)
    contrast_vector = _resolve_contrast(contrast, design_columns)

    n_terms = design_matrix.shape[1]
    if contrast_vector.shape != (n_terms,):
        raise ValueError(f"contrast length {contrast_vector.shape} != design cols ({n_terms},).")
    design_inverse, rank, estimable = _decompose_glm_design(design_matrix, contrast_vector)
    df_resid = n_samples - rank
    if df_resid <= 0:
        raise ValueError(
            f"design has no residual degrees of freedom: n_samples={n_samples}, rank={rank}."
        )
    if not estimable:
        raise ValueError(
            "contrast is not estimable from the supplied design matrix "
            f"(rank={rank}, n_terms={n_terms})."
        )

    weights = _resolve_freq_weights(freq_weights, n_bins)
    observed = np.empty(n_genes)
    beta = np.empty((n_terms, n_genes, n_bins))
    sigma_log = np.zeros((n_bins, n_bins))
    for block in _comparison_blocks(n_samples, n_genes, n_bins, n_terms):
        response = _log_spectra_block(spectra[:, block], normalize_shape).reshape(n_samples, -1)
        beta_block, residuals, theta = _fit_glm_response(
            response, design_matrix, design_inverse, contrast_vector, rank
        )
        beta[:, block] = beta_block.reshape(n_terms, -1, n_bins)
        observed[block] = np.sqrt(np.sum(weights * theta.reshape(-1, n_bins) ** 2, axis=-1))
        residuals = residuals.reshape(-1, n_bins)
        sigma_log += residuals.T @ residuals
    sigma_log /= n_genes * df_resid
    _maybe_warn_small_df_analytic(df_resid)

    # Compute the weighted log-L2 statistic, contrast scale, and eigenvalues.
    contrast_scale = float(np.sum((contrast_vector @ design_inverse) ** 2))
    sqrt_weights = np.sqrt(weights)
    weighted_cov = sqrt_weights[:, None] * sigma_log * sqrt_weights[None, :]
    weighted_cov_eigenvalues = np.maximum(np.linalg.eigvalsh(weighted_cov), 0.0)
    eigenvalues = weighted_cov_eigenvalues * contrast_scale
    return {
        "mode": "glm",
        "masked": False,
        "observed": observed,
        "sigma_log": sigma_log,
        "freq_weights": weights,
        "weighted_cov": weighted_cov,
        "eigenvalues": eigenvalues,
        "effective_rank": _effective_rank_from_eigenvalues(weighted_cov_eigenvalues),
        "contrast_scale": contrast_scale,
        "df_resid": df_resid,
        "design_columns": design_columns,
        "contrast_vector": contrast_vector,
        "beta": beta,
        "n_obs": n_samples,
        "eligible": np.ones(n_genes, dtype=bool),
    }


def _estimate_glm_masked_null_covariance(  # noqa: C901
    spectra: np.ndarray,
    design: pd.DataFrame | np.ndarray,
    contrast: str | dict[str, float] | np.ndarray,
    presence: np.ndarray,
    *,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
    min_resid_df: int = 1,
) -> _AnalyticNullState:
    """Compute analytic null state for masked GLM ``log_l2`` tests.

    A separate OLS fit is performed for each gene's observed samples. Eligible
    genes must have enough residual degrees of freedom, an estimable contrast,
    and a positive finite contrast variance. Residual cross-products are pooled
    into one covariance estimate, while ``observed`` and contrast-scaled
    ``eigenvalues`` remain per-gene because missingness changes the design.
    """
    if int(min_resid_df) < 1:
        raise ValueError(f"min_resid_df must be >= 1, got {min_resid_df}.")
    spectra = np.asarray(spectra)
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D (n_samples, n_genes, n_bins), got {spectra.shape}.")
    n_samples, n_genes, n_bins = spectra.shape
    presence = np.asarray(presence, dtype=bool)
    if presence.shape != (n_samples, n_genes):
        raise ValueError(
            f"presence shape {presence.shape} != (n_samples, n_genes) = "
            f"({n_samples}, {n_genes})."
        )

    # Resolve the design matrix and contrast vector.
    design_matrix, design_columns = _build_design_matrix(design, n_samples)
    contrast_vector = _resolve_contrast(contrast, design_columns)

    n_terms = design_matrix.shape[1]
    if contrast_vector.shape != (n_terms,):
        raise ValueError(f"contrast length {contrast_vector.shape} != design cols ({n_terms},).")

    # Compute the OLS fit for each gene and estimate the pooled residual covariance.
    weights = _resolve_freq_weights(freq_weights, n_bins)
    sqrt_weights = np.sqrt(weights)
    n_obs = np.zeros(n_genes, dtype=int)
    df_resid = np.zeros(n_genes, dtype=int)
    observed = np.full(n_genes, np.nan, dtype=float)
    contrast_scale = np.full(n_genes, np.nan, dtype=float)
    beta = np.full((n_terms, n_genes, n_bins), np.nan, dtype=float)
    eligible = np.zeros(n_genes, dtype=bool)
    sigma_acc = np.zeros((n_bins, n_bins), dtype=float)
    pooled_df = 0

    for sample_mask, genes in _presence_groups(presence):
        count = int(sample_mask.sum())
        n_obs[genes] = count
        if count == 0:
            continue

        gene_design = design_matrix[sample_mask]
        design_inverse, rank_gene, estimable = _decompose_glm_design(gene_design, contrast_vector)
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
        for block in _comparison_blocks(count, len(genes), n_bins, n_terms):
            cols = genes[block]
            response = _log_spectra_block(spectra[rows[:, None], cols], normalize_shape)
            gene_beta, residuals, theta = _fit_glm_response(
                response.reshape(count, -1), gene_design, design_inverse, contrast_vector, rank_gene
            )
            observed[cols] = np.sqrt(np.sum(weights * theta.reshape(-1, n_bins) ** 2, axis=-1))
            beta[:, cols] = gene_beta.reshape(n_terms, -1, n_bins)
            residuals = residuals.reshape(-1, n_bins)
            sigma_acc += residuals.T @ residuals
        pooled_df += df_gene * len(genes)
        contrast_scale[genes] = scale
        eligible[genes] = True

    if pooled_df == 0:
        raise ValueError(
            "estimate_null_covariance: no genes have enough observed "
            "samples and an estimable contrast to estimate the pooled covariance."
        )

    # Compute the weighted log-L2 statistic, contrast scale, and eigenvalues.
    _maybe_warn_small_df_analytic(int(np.median(df_resid[eligible])))
    sigma_log = sigma_acc / pooled_df
    weighted_cov = sqrt_weights[:, None] * sigma_log * sqrt_weights[None, :]
    weighted_cov_eigenvalues = np.maximum(np.linalg.eigvalsh(weighted_cov), 0.0)
    eigenvalues = np.full((n_genes, n_bins), np.nan, dtype=float)
    eigenvalues[eligible] = contrast_scale[eligible, None] * weighted_cov_eigenvalues[None, :]

    return {
        "mode": "glm",
        "masked": True,
        "observed": observed,
        "sigma_log": sigma_log,
        "freq_weights": weights,
        "weighted_cov": weighted_cov,
        "eigenvalues": eigenvalues,
        "effective_rank": _effective_rank_from_eigenvalues(weighted_cov_eigenvalues),
        "contrast_scale": contrast_scale,
        "df_resid": df_resid,
        "design_columns": design_columns,
        "contrast_vector": contrast_vector,
        "beta": beta,
        "n_obs": n_obs,
        "eligible": eligible,
    }


# ---------------------------------------------------------------------------
# Two-group spectral comparison functions
# ---------------------------------------------------------------------------


def compare_two_groups(  # noqa: C901
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
) -> pd.DataFrame:
    """
    Test, for every gene, whether its spatial-pattern spectrum differs between two groups.

    Parameters
    ----------
    spectra : np.ndarray
        Per-sample spectral features of shape ``(n_samples, n_genes, n_bins)``.
    groups : np.ndarray
        Group labels of length ``n_samples`` taking exactly two distinct values
        (mapped internally to 0/1 in sorted order).
    gene_names : sequence of str, optional
        Names for the gene axis. If None, integer indices are used.
    statistic : {'log_l2', 'welch_t_cauchy'}, default 'log_l2'
        Test statistic:

        - ``'log_l2'`` — (optionally weighted) L2 distance between mean
          log-spectra. Global / summary statistic. Pair with
          ``null='analytic'`` for an analytic mixture-χ² null that bypasses
          the small-n permutation BH-floor; ``null='permutation'``
          falls back to label permutations with exact enumeration when
          ``C(n, n_a) ≤ max_exact_permutations``.
        - ``'welch_t_cauchy'`` — per-bin Welch two-sided t-test with
          **analytic** (t-distribution) p-values combined across bins
          via Cauchy combination. Analytic is the whole point:
          permutation p-values would floor at ``1/(n_perm + 1)`` per
          bin, which would also floor the gene-level combined p-value
          and destroy BH-FDR power across thousands of genes. Yields
          an extra ``P_value_per_bin`` column. Requires at least two samples
          per arm. Bins constant across all samples are excluded from the
          combination; genes with no informative bins receive p=1.
    null : {'analytic', 'permutation'}, default 'analytic'
        Null-distribution method. ``'analytic'`` (the default) uses Liu's
        mixture-χ² approximation for the L2 quadratic form:
        under H₀ the statistic ``T² = D'WD`` is distributed as a
        weighted sum of χ²₁ variables whose tail is integrated via Liu's
        approximation (see :func:`sonic.statistics.liu_sf`). Requires positive
        residual degrees of freedom (``n_a + n_b > 2``).
        ``'permutation'`` uses the empirical sample-label permutation
        null and is the only option that respects the
        ``n_perm`` / ``random_state`` / ``max_exact_permutations`` arguments.
        ``welch_t_cauchy`` carries its own analytic t-distribution null
        and ignores this selector. For selector-controlled tests,
        ``null='analytic'`` is supported with ``statistic='log_l2'``.

        **Sample-size guidance** (residual df = ``n_a + n_b - 2``):

        - df ≥ 4 (n_a + n_b ≥ 6): ``'analytic'`` recommended — strong
          calibration + sensitivity; sweeps the top of our benchmark.
        - df ≥ 3 (n_a + n_b ≥ 5): ``'analytic'`` acceptable.
        - df < 3 (n_a + n_b ≤ 4): ``'analytic'`` emits a ``UserWarning``;
          σ̂² has ≥ 67% relative noise so per-test calibration may be
          anti-conservative. Prefer ``statistic='welch_t_cauchy'``
          (per-bin Welch t with proper df-corrected denominator) or
          stay with ``null='permutation'`` if the cohort allows
          enough exact relabellings.
    n_perm : int, default 1000
        Number of label permutations for the null distribution.
    max_exact_permutations : int, default 10000
        If the total number of distinct two-group relabellings
        ``C(n_samples, n_a)`` is at most this, every possible relabelling
        is enumerated (**exact permutation test**) and ``n_perm`` is
        overridden to the enumeration count.
    random_state : int, optional
        Seed for the permutation RNG.
    freq_weights : np.ndarray, optional
        Only used by ``statistic='log_l2'``. Non-negative weights of length
        ``n_bins`` (the number of frequency bins); internally renormalized to
        sum-1. Lets the user emphasize specific frequencies — e.g., a
        polynomial low-pass shape to mirror a CAR kernel, or an exponential
        high-pass shape to mirror a Gaussian kernel. ``None`` (default)
        means uniform weights.
    normalize_shape : bool, default False
        If True, divide each per-(sample, gene) spectrum by its sum along
        the trailing (frequency) axis before the statistic is computed
        (i.e., apply :func:`sonic.comparators.normalization.normalize_shape`
        to ``spectra`` first). Use to isolate shape-only /
        frequency-redistribution signals independent of overall amplitude.
        Works with every valid ``statistic=`` value.

    Returns
    -------
    pd.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value``, ``P_adj``
        (BH-FDR), sorted by descending statistic. When
        ``statistic='welch_t_cauchy'``, the frame also carries a
        ``P_value_per_bin`` object column — each entry is an
        ``(n_bins,)`` numpy array of per-bin analytic Welch p-values for that gene.

    Raises
    ------
    ValueError
        If ``statistic`` is unknown, ``groups`` does not contain exactly two values,
        or shapes are inconsistent.
    """
    if statistic not in _AVAILABLE_STATISTICS:
        raise ValueError(
            f"Unknown statistic '{statistic}'. Options: {list(_AVAILABLE_STATISTICS)}."
        )
    if null not in _NULL_OPTIONS:
        raise ValueError(f"Unknown null='{null}'. Options: {list(_NULL_OPTIONS)}.")
    spectra = np.asarray(spectra)
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D (n_samples, n_genes, n_bins), got {spectra.shape}.")
    n_samples, n_genes, _ = spectra.shape
    groups = np.asarray(groups)
    if groups.shape != (n_samples,):
        raise ValueError(f"groups shape {groups.shape} does not match n_samples={n_samples}.")
    uniq = np.unique(groups)
    if uniq.size != 2:
        raise ValueError(f"groups must contain exactly two distinct values, got {uniq}.")
    group_codes = (groups == uniq[1]).astype(int)  # 0 = first label sorted, 1 = second

    if statistic != "log_l2" or null != "analytic":
        spectra = np.asarray(spectra, dtype=np.float64)
    rng = np.random.default_rng(random_state)  # ignored if using analytic null

    # Run per-bin t tests and combine into a single gene-level statistic
    # ``welch_t_cauchy`` carries its own analytic null and the ``null`` argument is ignored.
    if statistic == "welch_t_cauchy":
        if normalize_shape:
            spectra = _normalize_shape_apply(spectra)
        if freq_weights is not None:
            logger.debug("freq_weights is ignored by statistic='welch_t_cauchy'.")
        observed, combined_p, per_bin_p = _run_welch_t_cauchy_analytic(spectra, group_codes)
        summary_stat = observed.max(axis=-1)  # reportable scalar per gene
        df = _comparison_frame(
            n_genes,
            gene_names,
            summary_stat,
            combined_p,
            extra={"P_value_per_bin": list(per_bin_p)},
        )
        return df

    # Test the log_l2 statistic with Liu's analytic mixture-chi-square null.
    if null == "analytic":
        state = _estimate_two_group_null_covariance(
            spectra,
            groups,
            freq_weights=freq_weights,
            normalize_shape=normalize_shape,
        )
        pvals = _log_l2_pvalues_from_state(state)
        return _comparison_frame(n_genes, gene_names, state["observed"], pvals)

    # Permutation path: generate the exchangeable label set once, then evaluate
    # the same log_l2 statistic under every relabelling.
    if normalize_shape:
        spectra = _normalize_shape_apply(spectra)
    perm_labels, is_exact = _exchangeable_group_labels(
        group_codes,
        n_perm,
        rng,
        max_exact_permutations=max_exact_permutations,
    )
    if is_exact:
        logger.info(
            "Exact permutation test: enumerated %d distinct relabellings " "(C(%d, %d)).",
            perm_labels.shape[0],
            n_samples,
            int((group_codes == 0).sum()),
        )
    observed, pvals = _run_statistic_with_perm(
        statistic, spectra, group_codes, perm_labels, freq_weights=freq_weights, is_exact=is_exact
    )

    df = _comparison_frame(n_genes, gene_names, observed, pvals)
    return df


def compare_two_groups_masked(  # noqa: C901
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
) -> pd.DataFrame:
    """
    Per-gene two-group pattern test with **incomplete data** across samples.

    For each gene, only the subset of samples with ``presence[:, g] == True``
    contributes to the observed statistic and to the label-permutation null.
    Genes that fail to reach ``min_samples_per_group`` observations in at
    least one group are reported with ``NaN`` p-values and the number of
    observed samples per group, so the user sees why they were skipped.

    Parameters
    ----------
    spectra : np.ndarray
        ``(n_samples, n_genes, n_bins)``.
    groups : np.ndarray
        ``(n_samples,)``, exactly two distinct labels.
    presence : np.ndarray
        ``(n_samples, n_genes)`` boolean mask. ``True`` = gene is observed
        in that sample (contributes); ``False`` = gene is absent (ignored).
    gene_names : sequence of str, optional
    statistic : {'log_l2', 'welch_t_cauchy'}, default 'log_l2'
    null : {'analytic', 'permutation'}, default 'analytic'
        Null-distribution method. ``'analytic'`` (the default) uses a
        Liu mixture-χ² test adapted for the masked
        case via a **mask-aware pooled-Σ** estimator: a single global
        ``(n_bins, n_bins)`` covariance is accumulated across every gene's present
        (sample, gene) cells (each gene contributes ``n_g - 2``
        residual degrees of freedom), and per-gene noncentrality
        scaling ``v_{c,g} = 1/n_a_g + 1/n_b_g`` adjusts the eigenvalues
        for that gene's specific cohort. Cross-bin correlation
        structure is taken to be homogeneous across genes (the same
        A3 assumption used in :func:`compare_two_groups` with the analytic null).
        Empirical calibration on synthetic missingness up to 50%
        matches the unmasked analytic path. Currently supported only with
        ``statistic='log_l2'``.

        ``'permutation'`` runs a per-gene permutation test,
        exact-enumerated when ``C(n_g, n_a_g) <= max_exact_permutations``
        (most genes at small samples).
    n_perm : int, default 1000
        Number of label permutations for the null distribution.
    max_exact_permutations : int, default 10000
        If the total number of distinct two-group relabellings
        ``C(n_samples, n_a)`` is at most this, every possible relabelling
        is enumerated (**exact permutation test**) and ``n_perm`` is
        overridden to the enumeration count.
    random_state : int, optional
        Seed for the permutation RNG.
    min_samples_per_group : int, default 2
        Minimum observed samples in each group for the gene to be tested.
    freq_weights : np.ndarray, optional
        Only consumed by ``statistic='log_l2'`` (same semantics as
        :func:`compare_two_groups`).
    normalize_shape : bool, default False
        If True, divide each per-(sample, gene) spectrum by its sum along
        the trailing (frequency) axis before the statistic is computed
        (same semantics as in :func:`compare_two_groups`). Use to isolate
        shape-only / frequency-redistribution signals. Works with every
        valid ``statistic=`` value.

    Returns
    -------
    pd.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value``, ``P_adj``,
        ``n_obs_A``, ``n_obs_B``. For ``'welch_t_cauchy'`` a
        ``P_value_per_bin`` column is also included (``None`` for skipped
        genes). BH-FDR is computed only over tested genes.
    """
    if statistic not in _AVAILABLE_STATISTICS:
        raise ValueError(
            f"Unknown statistic '{statistic}'. Options: {list(_AVAILABLE_STATISTICS)}."
        )
    if null not in _NULL_OPTIONS:
        raise ValueError(f"Unknown null='{null}'. Options: {list(_NULL_OPTIONS)}.")
    spectra = np.asarray(spectra)
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D, got {spectra.shape}.")
    n_samples, n_genes, _ = spectra.shape
    if presence.shape != (n_samples, n_genes):
        raise ValueError(
            f"presence shape {presence.shape} != (n_samples, n_genes) = "
            f"({n_samples}, {n_genes})."
        )
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if uniq.size != 2:
        raise ValueError("groups must contain exactly two distinct values.")
    group_codes = (groups == uniq[1]).astype(int)

    if statistic != "log_l2" or null != "analytic":
        spectra = np.asarray(spectra, dtype=np.float64)
    rng = np.random.default_rng(random_state)  # ignored if using analytic null

    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]

    # Analytic masked path: precompute global pooled Σ + eigvalsh; then per-gene
    # T², v_c-scaled λ, and Liu-tail p-value. ``welch_t_cauchy`` carries its
    # own analytic null and falls through to the per-gene branch below.
    if null == "analytic" and statistic == "log_l2":
        state = _estimate_two_group_masked_null_covariance(
            spectra,
            groups,
            presence,
            freq_weights=freq_weights,
            normalize_shape=normalize_shape,
            min_samples_per_group=min_samples_per_group,
        )
        pvals = _log_l2_pvalues_from_state(state)
        return _comparison_frame(
            n_genes,
            gene_names,
            state["observed"],
            pvals,
            extra={"n_obs_A": state["n_obs_A"], "n_obs_B": state["n_obs_B"]},
        )

    # Permutation / welch_t_cauchy masked path (per-gene loop).
    # Welch t-test ignores the null argument and always uses its own analytic null.
    if normalize_shape:
        spectra = _normalize_shape_apply(spectra)
    rows: list[dict[str, Any]] = []
    for gene_idx in range(n_genes):
        sample_mask = presence[:, gene_idx]
        group_a = group_codes[sample_mask] == 0
        group_b = group_codes[sample_mask] == 1
        n_a, n_b = int(group_a.sum()), int(group_b.sum())
        row: dict[str, Any] = {
            "Feature": gene_names[gene_idx],
            "n_obs_A": n_a,
            "n_obs_B": n_b,
            "Statistic": np.nan,
            "P_value": np.nan,
        }
        if statistic == "welch_t_cauchy":
            row["P_value_per_bin"] = None

        if n_a < min_samples_per_group or n_b < min_samples_per_group:
            rows.append(row)
            continue

        sub = spectra[sample_mask, gene_idx : gene_idx + 1, :]  # (n_obs, 1, n_bins)
        sub_groups = group_codes[sample_mask]

        if statistic == "welch_t_cauchy":
            # Compute the analytic Welch t-test in the present subset.
            observed, combined_p, per_bin_p = _run_welch_t_cauchy_analytic(sub, sub_groups)
            row["Statistic"] = float(observed.max())
            row["P_value"] = float(combined_p[0])
            row["P_value_per_bin"] = per_bin_p[0]
        else:
            # Per-gene exchange set — enumerate exactly when C(n_obs, n_a_obs)
            # is small, otherwise sample. Subsets are typically smaller than
            # the global one so the exact path kicks in more often here.
            perm_labels, is_exact = _exchangeable_group_labels(
                sub_groups,
                n_perm,
                rng,
                max_exact_permutations=max_exact_permutations,
            )
            observed, pval = _run_statistic_with_perm(
                statistic,
                sub,
                sub_groups,
                perm_labels,
                freq_weights=freq_weights,
                is_exact=is_exact,
            )
            row["Statistic"] = float(observed[0])
            row["P_value"] = float(pval[0])
        rows.append(row)

    df = pd.DataFrame(rows)
    # BH-correction over tested (non-NaN) genes only.
    tested = df["P_value"].notna()
    df["P_adj"] = np.nan
    if tested.any():
        df.loc[tested, "P_adj"] = apply_bh_correction(df.loc[tested, "P_value"])
    return df.sort_values("Statistic", ascending=False, na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# GLM-based continuous spectral comparison functions
# ---------------------------------------------------------------------------


def compare_glm(
    spectra: np.ndarray,
    design: pd.DataFrame | np.ndarray,
    contrast: str | dict[str, float] | np.ndarray,
    gene_names: Sequence[str] | None = None,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
) -> pd.DataFrame:
    """Log-L2 analytic spectral comparison via a design matrix and contrast.

    Generalises :func:`compare_two_groups` from binary group labels to an
    arbitrary GLM design matrix and a single-DOF linear contrast. The
    binary case is recovered exactly by passing
    ``design=pd.DataFrame({"group": groups})`` and ``contrast="group"``;
    p-values match ``compare_two_groups(..., statistic="log_l2",
    null="analytic")`` to machine precision.

    Parameters
    ----------
    spectra : np.ndarray
        ``(n_samples, n_genes, n_bins)`` spectral features (raw, not logged).
    design : pd.DataFrame or np.ndarray
        Sample-level metadata. ``DataFrame`` columns are auto-encoded via
        :mod:`patsy` (treatment-coded categoricals + intercept);
        ``ndarray`` is passed through as the design matrix verbatim
        (caller responsible for the intercept column). Encoded column labels
        must be unique; a metadata column named ``Intercept`` collides
        with the generated intercept and must be renamed.
    contrast : str, dict, or np.ndarray
        Nonzero, finite, estimable linear-contrast specification:

        - ``str`` — name of a design column. Auto-resolves treatment-coded
          categoricals (e.g., ``"genotype"`` matches ``"genotype[T.TG]"``).
          Multi-DOF (3+ level factor) contrasts must be passed as an
          explicit dict or ndarray.
        - ``dict[str, float]`` — coefficient per design column.
        - ``np.ndarray`` of length ``p`` — raw contrast vector.
    gene_names : sequence of str, optional
    freq_weights : np.ndarray, optional
        Optional non-negative weights over frequency bins, same semantics as
        :func:`compare_two_groups` with ``statistic="log_l2"``.
    normalize_shape : bool, default False
        If True, divide each per-(sample, gene) spectrum by its sum along
        the trailing (frequency) axis before the GLM is fit (same
        semantics as in :func:`compare_two_groups`). Use to isolate
        shape-only / frequency-redistribution signals along the design
        contrast independent of overall amplitude.

    Returns
    -------
    pd.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value``, ``P_adj`` —
        same schema as :func:`compare_two_groups`.

    Raises
    ------
    ValueError
        If shapes are inconsistent or ``contrast`` cannot be resolved.
    """
    state = _estimate_glm_null_covariance(
        spectra,
        design,
        contrast,
        freq_weights=freq_weights,
        normalize_shape=normalize_shape,
    )
    pvals = _log_l2_pvalues_from_state(state)
    return _comparison_frame(len(state["observed"]), gene_names, state["observed"], pvals)


def compare_glm_masked(  # noqa: C901
    spectra: np.ndarray,
    design: pd.DataFrame | np.ndarray,
    contrast: str | dict[str, float] | np.ndarray,
    presence: np.ndarray,
    gene_names: Sequence[str] | None = None,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
    min_resid_df: int = 1,
) -> pd.DataFrame:
    """Masked design-matrix contrast test for gene-specific missing spectra.

    This is the incomplete-data analogue of :func:`compare_glm`. For each
    gene, only samples with ``presence[:, g]`` are used to fit the OLS model
    on log-spectra. Genes whose observed design has too little residual
    degrees of freedom, or whose contrast is not estimable after masking, are
    retained in the output with ``NaN`` p-values.

    Parameters
    ----------
    spectra : np.ndarray
        ``(n_samples, n_genes, n_bins)`` spectral features.
    design : pandas.DataFrame or np.ndarray
        Sample-level design, encoded exactly as in :func:`compare_glm`.
    contrast : str, dict, or np.ndarray
        Single linear contrast, resolved exactly as in :func:`compare_glm`.
    presence : np.ndarray
        Boolean ``(n_samples, n_genes)`` mask. ``True`` means the gene is
        observed in that sample and contributes to that gene's model.
    freq_weights : np.ndarray, optional
        Optional non-negative weights over frequency bins, same semantics as
        :func:`compare_glm`.
    normalize_shape : bool, default False
        If True, apply :func:`sonic.comparators.normalization.normalize_shape`
        before fitting each gene model.
    min_resid_df : int, default 1
        Minimum per-gene residual degrees of freedom required for testing.

    Returns
    -------
    pandas.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value``, ``n_obs``,
        ``df_resid``, and ``P_adj``. BH-FDR is computed over finite p-values.
    """
    state = _estimate_glm_masked_null_covariance(
        spectra,
        design,
        contrast,
        presence,
        freq_weights=freq_weights,
        normalize_shape=normalize_shape,
        min_resid_df=min_resid_df,
    )
    pvals = _log_l2_pvalues_from_state(state)
    return _comparison_frame(
        len(state["observed"]),
        gene_names,
        state["observed"],
        pvals,
        extra={"n_obs": state["n_obs"], "df_resid": state["df_resid"]},
    )


# ---------------------------------------------------------------------------
# Pseudo-bulk expression scalar comparison functions
# ---------------------------------------------------------------------------


def compare_two_groups_scalar(
    values: np.ndarray,
    groups: np.ndarray,
    gene_names: Sequence[str] | None = None,
    *,
    log_expression: bool = False,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Per-gene two-sample test on scalar per-sample values (classical DE).

    The natural companion to :func:`compare_two_groups`: tested on the DC scalars
    (per-gene grid means) produced by
    :func:`sonic.comparators.features.compute_sample_spectrum`.

    For each gene, the function reports ``Statistic = abs(t)`` where ``t`` is
    the Welch two-sample t statistic, and ``P_value`` is the analytic two-sided
    tail probability under the Welch-Satterthwaite t-distribution null.
    Both groups must contain at least two samples, and values must be finite.

    Parameters
    ----------
    values : np.ndarray
        Per-sample per-gene scalars of shape ``(n_samples, n_genes)`` — e.g.,
        log-normalized mean expression on each slide.
    groups : np.ndarray
        Group labels of length ``n_samples`` with exactly two distinct values.
    gene_names : sequence of str, optional
        Gene names. Integer indices if None.
    log_expression : bool, default False
        If True, test ``log(values + eps)`` instead of raw scalar
        expression values. ``Mean_diff`` is then reported on the log scale.
    eps : float, default 1e-12
        Additive offset used only when ``log_expression=True``.

    Returns
    -------
    pd.DataFrame
        Columns ``Feature``, ``Statistic`` (``abs(Welch t)``), ``Mean_diff``
        (``mean_groupA − mean_groupB``), ``P_value``, ``P_adj`` (BH-FDR), sorted
        by descending ``Statistic``.

    Raises
    ------
    ValueError
        If shapes are inconsistent, ``groups`` does not contain exactly two
        distinct values.
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

    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]
    df = pd.DataFrame(
        {
            "Feature": list(gene_names),
            "Statistic": observed,
            "Mean_diff": mean_diff,
            "P_value": pvals,
        }
    )
    df["P_adj"] = apply_bh_correction(df["P_value"])
    return df.sort_values("Statistic", ascending=False).reset_index(drop=True)


def compare_glm_scalar(
    values: np.ndarray,
    design: pd.DataFrame | np.ndarray,
    contrast: str | dict[str, float] | np.ndarray,
    gene_names: Sequence[str] | None = None,
    *,
    log_expression: bool = False,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Per-gene scalar linear-model test via a design matrix and contrast.

    This is the scalar-expression companion to :func:`compare_glm`: it fits an
    ordinary least-squares model independently for each gene's per-sample scalar
    values and tests one linear contrast with the usual OLS t statistic.
    ``Statistic`` is ``abs(t)``, and ``P_value`` is the analytic two-sided tail
    probability under a Student t null with ``n_samples - rank(design_matrix)`` residual
    degrees of freedom. Passing ``log_expression=True`` fits the model on
    ``log(values + eps)``, which tests multiplicative changes in non-negative
    expression-like means.
    Responses, design entries and contrast coefficients must be finite. With
    zero residual variance, an exactly zero estimate has p=1 and a nonzero
    estimate has p=0, independent of its units.

    Parameters
    ----------
    values : np.ndarray
        Per-sample per-gene scalars of shape ``(n_samples, n_genes)``.
    design : pandas.DataFrame or np.ndarray
        Sample-level metadata. ``DataFrame`` columns are encoded with the same
        rules as :func:`compare_glm`; ``ndarray`` is used verbatim.
    contrast : str, dict, or np.ndarray
        Linear contrast specification resolved by the same rules as
        :func:`compare_glm`.
    gene_names : sequence of str, optional
    log_expression : bool, default False
        If True, test ``log(values + eps)`` instead of raw scalar expression
        values. ``Estimate`` is then reported on the log scale.
    eps : float, default 1e-12
        Additive offset used only when ``log_expression=True``.

    Returns
    -------
    pandas.DataFrame
        Columns ``Feature``, ``Statistic`` (``abs(t)``), ``Estimate`` (the
        signed contrast estimate), ``P_value``, and ``P_adj``. Rows are sorted by
        descending ``Statistic``.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"values must be 2D (n_samples, n_genes), got {values.shape}.")
    n_samples, n_genes = values.shape

    values = _maybe_log_expression(values, log_expression=log_expression, eps=eps)

    if not np.isfinite(values).all():
        raise ValueError("Scalar GLM responses must be finite.")
    design_matrix, design_columns = _build_design_matrix(design, n_samples)
    contrast_vector = _resolve_contrast(contrast, design_columns)
    design_inverse, rank, estimable = _decompose_glm_design(design_matrix, contrast_vector)
    df_resid = n_samples - rank
    if df_resid <= 0:
        raise ValueError(
            f"design has no residual degrees of freedom: n_samples={n_samples}, rank={rank}."
        )
    if not estimable:
        raise ValueError(
            "contrast is not estimable from the supplied design matrix "
            f"(rank={rank}, n_terms={design_matrix.shape[1]})."
        )

    # Fit the OLS model
    _, resid, estimate = _fit_glm_response(
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

    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]
    df = pd.DataFrame(
        {
            "Feature": list(gene_names),
            "Statistic": observed,
            "Estimate": estimate,
            "P_value": pvals,
        }
    )
    df["P_adj"] = apply_bh_correction(df["P_value"])
    return df.sort_values("Statistic", ascending=False).reset_index(drop=True)
