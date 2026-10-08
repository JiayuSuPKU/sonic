"""Spectrum normalization primitives for comparator outputs."""

from __future__ import annotations

import numpy as np

__all__ = ["normalize_background", "normalize_covariates", "normalize_shape"]


def normalize_background(
    spectra: np.ndarray,
    *,
    axis: int = -2,
    eps: float = 1e-12,
) -> np.ndarray:
    """Cancel per-sample multiplicative gain via cross-gene geometric-mean centering.

    For each (sample, frequency-bin) pair, every gene's power is divided
    by the geometric mean of the spectrum across the genes axis. Use
    this to correct per-sample multiplicative gain (sequencing depth,
    antibody titre, dewaxing efficiency) that scales every gene's
    spectrum at every frequency by a sample-level factor.

    Parameters
    ----------
    spectra : np.ndarray
        Non-negative spectra :math:`P` with shape
        ``(..., n_genes, n_bins)`` when using the default ``axis=-2``.
        Any leading dimensions (e.g., ``n_samples``) are broadcast over.
    axis : int, default -2
        Axis along which the cross-gene geometric mean is taken
        (the genes axis).
    eps : float, default 1e-12
        Floor :math:`\\varepsilon` added before the logarithm to keep
        zeros finite.

    Returns
    -------
    np.ndarray
        Background-normalized spectra :math:`\\tilde P`, same shape as
        ``spectra``. Never mutates the input.

    Notes
    -----
    Let :math:`P` denote the input spectrum, :math:`G` the number of
    genes (length of ``axis``), :math:`K` the number of frequency
    bins, and :math:`\\varepsilon` the ``eps`` floor. The per-bin
    geometric-mean background is

    .. math::

        b_{k} = \\exp\\!\\Bigl(
            \\tfrac{1}{G} \\sum_{g'=1}^{G}
            \\log\\bigl(P_{g',k} + \\varepsilon\\bigr)
        \\Bigr),

    and the output is the per-bin gene-wise quotient

    .. math::

        \\tilde P_{g,k} = \\frac{P_{g,k}}{b_{k}}.

    Equivalently, in log-space this is per-bin mean centering across
    the genes axis,

    .. math::

        \\log \\tilde P_{g,k}
        = \\log\\bigl(P_{g,k} + \\varepsilon\\bigr)
          - \\tfrac{1}{G} \\sum_{g'=1}^{G}
            \\log\\bigl(P_{g',k} + \\varepsilon\\bigr),

    so after the transform :math:`\\prod_{g} \\tilde P_{g,k} = 1` at
    every bin :math:`k` - the cross-gene geometric mean at every
    frequency is unity.

    The operation is equivalent to a per-bin OLS regression of
    :math:`\\log P_{\\cdot,k}` against a constant (the cross-gene
    mean) followed by exponentiation. With a per-sample one-hot
    covariate stacked across all (sample, gene) rows, the residuals
    match :math:`\\log \\tilde P` row-for-row, so running this
    function sample-by-sample is identical to fitting a one-hot
    sample-ID covariate in log-space and residualizing.

    Companion functions:

    - :func:`normalize_covariates` removes per-bin bias linear in
      user-supplied covariate spectra (cell-type proportion maps,
      tissue domains, housekeeping templates).
    - :func:`normalize_shape` removes per-(sample, gene) amplitude
      by L1-normalizing along the frequency axis.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> spec = rng.lognormal(size=(2, 5, 8))      # (n_samples, n_genes, n_bins)
    >>> P_tilde = normalize_background(spec)
    >>> P_tilde.shape
    (2, 5, 8)
    >>> # Cross-gene geometric mean at each (sample, bin) is unity:
    >>> bool(np.allclose(np.prod(P_tilde, axis=-2), 1.0))
    True
    """
    log_spec = np.log(np.asarray(spectra, dtype=np.float64) + eps)
    bg = log_spec.mean(axis=axis, keepdims=True)
    return np.exp(log_spec - bg)


def normalize_covariates(
    spectra: np.ndarray,
    covariate_spectra: np.ndarray,
    *,
    fit_intercept: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Residualize log-spectra against the log of covariate spectra.

    Each gene's log-spectrum is regressed (per gene, OLS in log-space)
    on the log of the supplied covariate spectra plus an optional
    intercept; the function exponentiates and returns the residual
    spectrum. Use to remove the multiplicative contribution of
    structured per-bin templates (cell-type proportion maps,
    tissue-domain indicators, housekeeping composite expression) from
    every gene's per-frequency power.

    Operating in log-space matches the multiplicative noise model of
    spectral data, keeps the output strictly positive (so the result
    composes cleanly with the downstream ``log_l2`` test), and makes
    this helper commute exactly with :func:`normalize_background`
    (orthogonal projections along orthogonal axes - see Notes).

    Parameters
    ----------
    spectra : np.ndarray
        Non-negative gene spectra :math:`P` of shape ``(n_genes, n_bins)`` to
        residualize.
    covariate_spectra : np.ndarray
        Non-negative covariate spectra :math:`C` of shape
        ``(n_covariates, n_bins)``.
    fit_intercept : bool, default True
        If True, prepend a column of ones to the design matrix
        :math:`X` so per-gene log-amplitude offsets along the
        frequency axis are absorbed.
    eps : float, default 1e-12
        Floor :math:`\\varepsilon` added inside :math:`\\log(\\cdot)`
        on both ``spectra`` and ``covariate_spectra`` to keep zeros
        finite.

    Returns
    -------
    np.ndarray
        Residual spectra :math:`\\tilde P` of shape ``(n_genes, n_bins)``,
        strictly positive. Never mutates the input.

    Raises
    ------
    ValueError
        If the last-axis lengths differ or the covariate design spans all
        frequency bins, leaving no residual dimensions. Reduce the number
        of covariates or increase the number of supported frequency bins.

    Notes
    -----
    Let :math:`P \\in \\mathbb{R}_{\\geq 0}^{G \\times K}` denote the
    input spectra (:math:`G` genes, :math:`K` frequency bins) and
    :math:`C \\in \\mathbb{R}_{\\geq 0}^{n_{\\mathrm{cov}} \\times K}`
    the covariate spectra. Build the log-design matrix

    .. math::

        X = \\bigl[\\, \\mathbf{1}_{K} \\;\\big|\\;
                \\log(C^{\\top} + \\varepsilon) \\,\\bigr]
            \\;\\in\\; \\mathbb{R}^{K \\times (n_{\\mathrm{cov}} + 1)},

    dropping the leading column :math:`\\mathbf{1}_{K}` when
    ``fit_intercept=False``. Fit per-gene OLS coefficients via the
    Moore-Penrose pseudoinverse :math:`X^{+}` against the log of the
    response,

    .. math::

        \\hat\\beta_{g} = X^{+}\\,
            \\bigl[ \\log( P_{g,\\cdot} + \\varepsilon ) \\bigr]^{\\top}
            \\;\\in\\; \\mathbb{R}^{n_{\\mathrm{cov}} + 1},

    and return the exponentiated residual

    .. math::

        \\tilde P_{g,k}
        = \\exp\\!\\Bigl(
            \\log( P_{g,k} + \\varepsilon )
            - X_{k,\\cdot}\\,\\hat\\beta_{g}
          \\Bigr).

    Equivalently,

    .. math::

        \\log \\tilde P_{g,\\cdot}^{\\top}
        = \\bigl( I_{K} - X X^{+} \\bigr)\\,
          \\bigl[ \\log( P_{g,\\cdot} + \\varepsilon ) \\bigr]^{\\top},

    i.e., the orthogonal projection of each gene's **log-spectrum**
    onto the orthogonal complement of the column space of :math:`X`,
    then exponentiated.

    **Commutativity with** :func:`normalize_background`. In log-space
    the two operations are left- vs right-multiplication of the
    :math:`G \\times K` log-spectrum matrix by orthogonal-projection
    matrices on disjoint axes,

    .. math::

        \\mathrm{bg}: \\;\\log P \\;\\mapsto\\;
            \\bigl( I_{G} - \\tfrac{1}{G}\\mathbf{1}_{G}\\mathbf{1}_{G}^{\\top}
            \\bigr)\\,\\log P,
        \\qquad
        \\mathrm{cov}: \\;\\log P \\;\\mapsto\\;
            \\log P \\,\\bigl( I_{K} - X X^{+} \\bigr).

    Left- and right-multiplication trivially commute, so we have the exact identity
    ``normalize_background(normalize_covariates(P)) ==
    normalize_covariates(normalize_background(P))``.

    With ``fit_intercept=True`` and **no** covariates (empty
    ``covariate_spectra``), this reduces to per-gene log-mean centering
    along the frequency axis,

    .. math::

        \\tilde P_{g,k}
        = \\frac{P_{g,k} + \\varepsilon}
                {\\exp\\!\\bigl(\\tfrac{1}{K}
                        \\sum_{k'=1}^{K}\\log(P_{g,k'} + \\varepsilon)
                  \\bigr)},

    i.e., dividing each gene's spectrum by its own cross-bin geometric
    mean - a per-gene companion to :func:`normalize_background`'s
    per-bin cross-gene operation, distinct from
    :func:`normalize_shape`'s arithmetic-mean / sum-1 normalization.

    Companion functions:

    - :func:`normalize_background` removes per-sample multiplicative
      gain via cross-gene geometric-mean centering in log-space
      (perpendicular axis to this function).
    - :func:`normalize_shape` removes per-(sample, gene) amplitude
      by L1-normalizing along the frequency axis.

    Examples
    --------
    >>> import numpy as np
    >>> rng  = np.random.default_rng(0)
    >>> spec = rng.lognormal(size=(20, 8))      # (n_genes, n_bins)
    >>> cov  = rng.lognormal(size=(2, 8))       # (n_covariates, n_bins)
    >>> resid = normalize_covariates(spec, cov)
    >>> resid.shape
    (20, 8)
    >>> bool((resid > 0).all())     # log-space output is strictly positive
    True
    """
    if spectra.shape[-1] != covariate_spectra.shape[-1]:
        raise ValueError(
            f"Last axis must match: spectra has n_bins={spectra.shape[-1]}, "
            f"covariate_spectra has n_bins={covariate_spectra.shape[-1]}."
        )
    n_bins = spectra.shape[-1]
    log_spec = np.log(np.asarray(spectra, dtype=np.float64) + eps)
    log_cov = np.log(np.asarray(covariate_spectra, dtype=np.float64) + eps)

    X = log_cov.T
    if fit_intercept:
        X = np.hstack([np.ones((n_bins, 1)), X])
    beta, _, rank, _ = np.linalg.lstsq(X, log_spec.T, rcond=1e-15)
    if rank >= n_bins:
        raise ValueError(
            f"Covariate design has no residual dimensions: rank={rank}, n_bins={n_bins}. "
            "Use fewer covariates or more supported frequency bins."
        )
    fitted = (X @ beta).T
    # Reuse the fitted-value buffer for residuals and their exponentials.
    np.subtract(log_spec, fitted, out=fitted)
    return np.exp(fitted, out=fitted)


def normalize_shape(
    spectra: np.ndarray,
    *,
    axis: int = -1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Normalize each nonzero spectrum to unit total power along ``axis``.

    Spectra differing only by a positive scalar have identical shapes, up to
    floating-point rounding. All-zero spectra stay zero. Scaling by the largest
    entry before summing avoids overflow and preserves very small powers.

    Parameters
    ----------
    spectra : np.ndarray
        Non-negative spectra. Any leading dimensions are preserved.
    axis : int, default -1
        Frequency axis to normalize.
    eps : float, default 1e-12
        Retained for backward compatibility; no floor is applied to positive
        totals. Zero totals are handled explicitly.

    Returns
    -------
    np.ndarray
        Float64 spectra summing to one wherever power is present, with zero
        spectra unchanged. Never mutates the input.

    Notes
    -----
    Used by comparison functions when ``normalize_shape=True`` to isolate
    frequency redistribution independently of overall amplitude. Unlike
    ``normalize_background``, normalization acts independently on each gene
    within each sample.
    """
    spectra = np.asarray(spectra, dtype=np.float64)
    scale = spectra.max(axis=axis, keepdims=True, initial=0.0)
    scaled = np.divide(spectra, scale, out=np.zeros_like(spectra), where=scale > 0)
    total = scaled.sum(axis=axis, keepdims=True)
    return np.divide(scaled, total, out=scaled, where=total > 0)


def _normalize_shape_apply(spectra: np.ndarray) -> np.ndarray:
    """Apply the comparison-default shape normalization along the frequency axis."""
    return normalize_shape(spectra, axis=-1)
