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
    """Remove the common power spectrum across genes in each sample.

    Parameters
    ----------
    spectra : np.ndarray
        Nonnegative powers with shape ``(..., n_genes, n_bins)`` for the default axis.
    axis : int, default -2
        Gene axis used to estimate the geometric mean.
    eps : float, default 1e-12
        Offset added to power before taking the logarithm.

    Returns
    -------
    np.ndarray
        Powers divided by their geometric mean across genes after adding the
        offset. The output has the same shape as the input.

    Notes
    -----
    Use this adjustment when a sample-wide difference in power is nuisance
    variation. A biological difference shared by many genes can also be removed.
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
    """Remove variation in gene power associated with known spatial covariates.

    Parameters
    ----------
    spectra : np.ndarray
        Nonnegative gene powers with shape ``(n_genes, n_bins)``.
    covariate_spectra : np.ndarray
        Nonnegative covariate powers with shape ``(n_covariates, n_bins)``.
        Use the same frequency bins as the gene spectra.
    fit_intercept : bool, default True
        Include an intercept to remove each gene's mean log power across bins.
    eps : float, default 1e-12
        Offset added to both arrays before taking the natural logarithm.

    Returns
    -------
    np.ndarray
        Exponentiated residuals with shape ``(n_genes, n_bins)``. Each gene's
        regression uses frequency bins as observations and log covariate powers
        as predictors.

    Raises
    ------
    ValueError
        If the frequency axes differ or the design has no residual degrees of freedom.

    Notes
    -----
    Use spatial covariate spectra here. Use the comparison design matrix to
    adjust for sample-level covariates such as batch.
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
    """Express each bin's power as a share of the spectrum's total power.

    Parameters
    ----------
    spectra : np.ndarray
        Finite, nonnegative powers.
    axis : int, default -1
        Frequency axis.
    eps : float, default 1e-12
        Unused. Retained for compatibility.

    Returns
    -------
    np.ndarray
        Power shares with the same shape as the input. Shares sum to one for
        each spectrum with positive total power. A spectrum of zeros stays zero.
    """
    spectra = np.asarray(spectra, dtype=np.float64)
    if not np.isfinite(spectra).all():
        raise ValueError("Shape normalization requires finite profiles for observed samples.")
    if np.any(spectra < 0):
        raise ValueError("Observed spectral powers must be nonnegative.")
    peak = spectra.max(axis=axis, keepdims=True, initial=0.0)
    exponent = np.frexp(peak)[1]
    scaled = np.ldexp(spectra, 1 - exponent, out=np.zeros_like(spectra), where=peak > 0)
    if scaled.size <= 1:
        return (scaled > 0).astype(float)
    # Sum each spectrum in a fixed order so permuting bins preserves rotation ties.
    axes = tuple(range(scaled.ndim)) if axis is None else tuple(np.atleast_1d(axis))
    rows = np.moveaxis(scaled, axes, range(-len(axes), 0))
    ordered = rows.copy(order="C").reshape(peak.size, -1)
    ordered.sort(axis=1)
    total = ordered.sum(axis=1).reshape(peak.shape)
    return np.divide(scaled, total, out=scaled, where=total > 0)


def _normalize_shape_apply(spectra: np.ndarray) -> np.ndarray:
    """Apply the comparison-default shape normalization along the frequency axis."""
    return normalize_shape(spectra, axis=-1)
