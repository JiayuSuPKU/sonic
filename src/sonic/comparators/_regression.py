"""Design encoding and stable ordinary-least-squares fits shared by DF and DE."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.stats import t as _t_dist


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
            f"design must be a numpy ndarray or pandas DataFrame, got {type(design).__name__}."
        )
    if len(design) != n_samples:
        raise ValueError(
            f"design DataFrame length {len(design)} does not match n_samples={n_samples}."
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
                    f"Contrast '{contrast}' not found in design columns {list(design_columns)}."
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
                    f"Contrast key '{k}' not in design columns {list(design_columns)}."
                )
            contrast_vector[list(design_columns).index(k)] = float(v)
        return contrast_vector
    raise TypeError(f"Contrast must be a str, dict, or ndarray; got {type(contrast).__name__}.")


@dataclass
class _ScaledDesign:
    """OLS design, contrast and pseudoinverse in the same scaled coordinates."""

    matrix: np.ndarray
    contrast: np.ndarray
    inverse: np.ndarray
    scales: np.ndarray
    rank: int
    estimability_error: str | None

    @property
    def estimable(self):
        return self.estimability_error is None

    @property
    def sample_weights(self):
        return self.contrast @ self.inverse


def _column_scales(matrix):
    # Powers of two preserve the represented values for exact-fit checks.
    largest = np.max(np.abs(matrix), axis=0)
    exponents = np.frexp(largest)[1]
    return np.ldexp(1.0, np.where(largest == 0, 0, exponents - 1))


def _rank_and_cutoff(matrix, singular):
    cutoff = np.finfo(float).eps * max(matrix.shape) * singular[0]
    return int(np.count_nonzero(singular > cutoff)), cutoff


def _contrast_in_row_space(matrix, contrast, rank, largest_singular, cutoff):
    """An estimable contrast adds no independent row at the design's rank tolerance."""
    while True:
        direction = contrast / np.max(np.abs(contrast))
        row = direction * (largest_singular / np.linalg.norm(direction))
        if np.linalg.matrix_rank(np.vstack((matrix, row)), tol=cutoff) != rank:
            return False
        # A large estimable term can hide a smaller aliased term below the cutoff.
        # Check those nonzero terms separately at a scale the rank test can resolve.
        small = (contrast != 0) & (np.abs(row) <= cutoff)
        if not small.any():
            return True
        contrast = np.where(small, contrast, 0.0)


def _centered_design_error(matrix, contrast, fitted_rank):
    """Check rank and contrast after removing offsets represented by the intercept."""
    low, high = matrix.min(axis=0), matrix.max(axis=0)
    origin = np.clip(0.0, low, high)
    origin[low == high] = 0.0
    if not np.any(origin):
        return None
    intercepts = np.flatnonzero((low == high) & (low != 0))
    if intercepts.size:
        i = intercepts[0]
        intercept_weight = contrast[i] / low[i]
    else:
        # Use unshifted columns so large timestamps cannot masquerade as an intercept.
        anchors = matrix[:, origin == 0]
        if not anchors.shape[1]:
            return None
        tolerance = np.finfo(float).eps * max(anchors.shape)
        inverse = np.linalg.pinv(anchors, rcond=tolerance)
        coefficients = inverse.sum(axis=1)
        residual = 1 - anchors @ coefficients
        if np.max(np.abs(residual)) > tolerance:
            return None
        # Refine before applying contrasts such as the difference of two group means.
        coefficients += inverse @ residual
        intercept_weight = contrast[origin == 0] @ coefficients
    centered = matrix - origin
    scales = _column_scales(centered)
    centered /= scales
    singular = np.linalg.svd(centered, compute_uv=False)
    rank, cutoff = _rank_and_cutoff(centered, singular)
    if rank != fitted_rank:
        return (
            f"GLM design rank changes from {fitted_rank} to {rank} after centering. "
            "Center the predictors or supply elapsed times before fitting."
        )
    adjusted = (contrast - intercept_weight * origin) / scales
    if rank < centered.shape[1] and not _contrast_in_row_space(
        centered, adjusted, rank, singular[0], cutoff
    ):
        return "contrast is not estimable from the centered design matrix."
    return None


def _decompose_glm_design(design_matrix: np.ndarray, contrast_vector: np.ndarray) -> _ScaledDesign:
    """Fit in comparable column units so timestamps cannot swamp the intercept.

    Keep the design, pseudoinverse and contrast together through inference.
    Only reported coefficients need converting back to the input units.
    """
    if not np.isfinite(design_matrix).all() or not np.isfinite(contrast_vector).all():
        raise ValueError("Design matrix and contrast must be finite.")
    if not np.any(contrast_vector):
        raise ValueError("contrast must be nonzero.")
    scales = _column_scales(design_matrix)
    matrix = design_matrix / scales
    contrast = contrast_vector / scales
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    rank, cutoff = _rank_and_cutoff(matrix, singular)
    rows = vt[:rank]
    inverse = (rows.T / singular[:rank]) @ u[:, :rank].T
    error = _centered_design_error(matrix, contrast, rank)
    if error is None and rank < matrix.shape[1]:
        if rank == 0 or not _contrast_in_row_space(matrix, contrast, rank, singular[0], cutoff):
            error = "contrast is not estimable from the supplied design matrix."
    return _ScaledDesign(matrix, contrast, inverse, scales, rank, error)


def _prepare_glm_design(design, contrast, n_samples):
    """Resolve and validate one design for reuse across response blocks.

    Returns the original matrix, column names and contrast, plus the scaled design.
    Masked tests still decompose each distinct observed-sample design separately.
    """
    matrix, columns = _build_design_matrix(design, n_samples)
    vector = _resolve_contrast(contrast, columns)
    fit = _decompose_glm_design(matrix, vector)
    if n_samples <= fit.rank:
        raise ValueError(
            f"design has no residual degrees of freedom: n_samples={n_samples}, rank={fit.rank}."
        )
    if fit.estimability_error:
        raise ValueError(fit.estimability_error)
    return matrix, columns, vector, fit


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


def _exact_glm_design(design, contrast, retained=None):
    """Return an exact orthogonal design, its inverse and the contrast's sample weights."""
    augmented = _as_fractions(np.vstack([design, contrast]))
    if retained is not None and len(retained) < design.shape[1]:
        # Restrict coefficients to the same right-singular subspace as the OLS fit.
        augmented = augmented @ _as_fractions(retained.T)
    # Apply the design column operations to the contrast as well.
    _orthogonalize_columns(augmented, len(design))
    q = augmented[:-1]
    inverse = q.T / np.sum(q * q, axis=0)[:, None]
    return q, inverse, augmented[-1] @ inverse


def _fit_glm_response(
    response: np.ndarray,
    fit: _ScaledDesign,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve OLS for an already-transformed 2D response block.

    Parameters
    ----------
    response : np.ndarray
        Shape ``(n_samples, n_responses)``. Columns are already-transformed
        responses, such as flattened feature/bin profiles or scalar outcomes.
    fit : _ScaledDesign
        Scaled design for the observed response rows and requested contrast.

    Returns
    -------
    beta : np.ndarray
        OLS coefficients in scaled design coordinates, shape ``(n_terms, n_responses)``.
    residuals : np.ndarray
        ``response - design @ beta``, shape ``(n_samples, n_responses)``.
    effect : np.ndarray
        ``contrast @ beta``, shape ``(n_responses,)``.

    Notes
    -----
    When the design spans a constant, subtract a response offset before fitting
    and restore its contribution to the coefficients and contrast afterward.
    This reduces cancellation error in residuals and small effects.
    """
    design, inverse, contrast = fit.matrix, fit.inverse, fit.contrast
    constant = np.flatnonzero(np.all(design == design[:1], axis=0) & (design[0] != 0))
    offset_beta = np.zeros(design.shape[1])
    if constant.size and fit.rank == design.shape[1]:
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


def _glm_fit_roundoff(design, inverse, scale):
    """Bound cancellation in X @ (X⁺ @ y), including nearly collinear designs."""
    amplification = np.max(np.abs(design) @ np.sum(np.abs(inverse), axis=1))
    return np.finfo(float).eps * max(design.shape) * (1 + amplification) * scale


def _test_glm_response(response, fit, *, unit_fits=None, mask_key=None):
    """Compute ordinary OLS contrast t tests for a transformed response block.

    Parameters
    ----------
    response : np.ndarray
        Finite responses, shape ``(n_samples, n_responses)``.
    fit : _ScaledDesign
        Validated design and estimable contrast, as in ``_fit_glm_response``.
        The design must have positive residual degrees of freedom.
    unit_fits : dict, optional
        Cache of constant-response fits shared within one comparison and contrast.
    mask_key : bytes, optional
        Observed-sample mask identifying this design in ``unit_fits``.

    Returns
    -------
    t_stat, pvals, estimate, informative : tuple of np.ndarray
        Signed t statistics, two-sided t tails, signed contrasts and a mask
        for bin selection, each of shape ``(n_responses,)``. A nonzero contrast
        or a varying response with residual variation is informative.
    """
    design, rank = fit.matrix, fit.rank
    _, residuals, estimate = _fit_glm_response(response, fit)
    df_resid = len(response) - rank
    sigma2 = np.sum(residuals**2, axis=0) / df_resid
    contrast_var = float(np.sum(fit.sample_weights**2))
    if not np.isfinite(contrast_var) or contrast_var <= 0:
        raise ValueError("contrast has zero or nonfinite variance under the supplied design.")
    se = np.sqrt(np.maximum(sigma2, 0.0) * contrast_var)
    scale = np.max(np.abs(response), axis=0)
    roundoff = _glm_fit_roundoff(design, fit.inverse, scale)
    suspect = (scale > 0) & (np.max(np.abs(residuals), axis=0) <= roundoff)
    has_variation = np.any(response != response[:1], axis=0)
    constants = (scale > 0) & ~has_variation
    if rank == design.shape[1] and np.any(np.all(design == design[:1], axis=0) & (design[0] != 0)):
        # _fit_glm_response already fits constant columns exactly with an explicit intercept.
        constants[:] = False
    refit = np.flatnonzero(suspect & has_variation)
    unit_fit = unit_fits.get(mask_key) if unit_fits is not None else None
    need_unit_fit = constants.any() and unit_fit is None
    if refit.size or need_unit_fit:
        # Verify the contrast and SE before their ratio can turn roundoff into a signal.
        retained = None
        if rank < design.shape[1]:
            retained = np.linalg.svd(design, full_matrices=False)[2][:rank]
        q, exact_inverse, weights = _exact_glm_design(design, fit.contrast, retained)
        se_factor = math.hypot(*weights) / math.sqrt(df_resid)
        if need_unit_fit:
            # Every constant column is a multiple of the all-ones response.
            unit_residual = 1 - q @ exact_inverse.sum(axis=1)
            unit_fit = float(weights.sum()), se_factor * math.hypot(*unit_residual)
            if unit_fits is not None:
                unit_fits[mask_key] = unit_fit
        for j in refit:
            y = _as_fractions(response[:, j])
            estimate[j] = weights @ y
            residual = y - q @ (exact_inverse @ y)
            se[j] = se_factor * math.hypot(*residual)
            # Nuisance covariates can explain all the variation in a nonconstant bin.
            has_variation[j] = residual.any()
    if constants.any():
        levels = response[0, constants]
        estimate[constants] = levels * unit_fit[0]
        se[constants] = np.abs(levels) * unit_fit[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = estimate / se
    zero_se = se == 0.0
    t_stat[zero_se & (estimate == 0.0)] = 0.0
    perfect_effect = zero_se & (estimate != 0.0)
    t_stat[perfect_effect] = np.sign(estimate[perfect_effect]) * np.inf
    pvals = 2.0 * _t_dist.sf(np.abs(t_stat), df_resid)
    return t_stat, pvals, estimate, has_variation | (estimate != 0)
