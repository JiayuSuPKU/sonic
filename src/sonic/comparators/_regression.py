"""Design encoding and stable ordinary-least-squares fits shared by DF and DE."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


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


def _prepare_glm_design(design, contrast, n_samples):
    """Resolve and validate one design for reuse across response blocks.

    Returns the matrix, column names, contrast vector, OLS inverse and rank.
    Masked tests still decompose each distinct observed-sample design separately.
    """
    matrix, columns = _build_design_matrix(design, n_samples)
    vector = _resolve_contrast(contrast, columns)
    inverse, rank, estimable = _decompose_glm_design(matrix, vector)
    if n_samples <= rank:
        raise ValueError(
            f"design has no residual degrees of freedom: n_samples={n_samples}, rank={rank}."
        )
    if not estimable:
        raise ValueError(
            "contrast is not estimable from the supplied design matrix "
            f"(rank={rank}, n_terms={matrix.shape[1]})."
        )
    return matrix, columns, vector, inverse, rank


def _fit_glm_response(
    response: np.ndarray,
    design: np.ndarray,
    inverse: np.ndarray,
    contrast: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve OLS for an already-transformed 2D response block.

    Parameters
    ----------
    response : np.ndarray
        Shape ``(n_samples, n_responses)``. Columns are already-transformed
        responses, such as flattened feature/bin profiles or scalar outcomes.
    design : np.ndarray
        Design matrix of shape ``(n_samples, n_terms)``, aligned with response rows.
    inverse : np.ndarray
        Precomputed pseudoinverse of ``design``, shape ``(n_terms, n_samples)``.
    contrast : np.ndarray
        Linear contrast weights, shape ``(n_terms,)``.
    rank : int
        Design rank from the same decomposition used to compute ``inverse``.

    Returns
    -------
    beta : np.ndarray
        OLS coefficients, shape ``(n_terms, n_responses)``.
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
