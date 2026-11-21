"""Compute spatial power spectra, align directional spectra, and measure variation
across gene profiles. See :mod:`sonic.comparators.binning` for frequency bins."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import scipy.ndimage
from tqdm.auto import tqdm

from sonic.comparators.binning import (
    _adapt_radial_edges,
    _as_full_power_spectrum,
    _bin_radial_spectrum,
    _layout_frequency_grid,
    _polar_bin_center_coords,
    _radial_bin_cell_counts,
    _resolve_power_layout,
    _stream_polar_features,
    _stream_radial_features,
    _validate_grid_shape,
    _validate_spacing,
)
from sonic.kernels.fft import power_spectrum_2d

__all__ = [
    "compute_sample_spectrum",
    "effective_rank",
    "gene_pattern_diversity",
    "power_spectrum_anisotropy",
    "stream_geomean_landmark",
    "estimate_rotations_from_landmarks",
    "apply_rotations_to_spectra",
    "align_spectra_by_rotation",
]


# Compatibility imports from releases before binning became private.
adapt_frequency_edges = _adapt_radial_edges


def _shared_observation_mask(values: np.ndarray) -> np.ndarray:
    """Return the observed raster locations shared by all requested features."""
    observed = np.isfinite(values)
    if np.isinf(values).any() or not np.all(observed == observed[:, :1]):
        raise ValueError(
            "FFT features require identical finite support; only shared NaN holes are allowed"
        )
    return observed[:, 0]


def _center_observations(values: np.ndarray) -> None:
    """Center an observed location-by-feature block in place, keeping constants exactly zero."""
    values -= values[0]
    values -= values.mean(axis=0)


def radial_bin_spectrum(
    spectrum: np.ndarray,
    grid_shape: tuple[int, int],
    n_bins: int = 30,
    fft_solver: str = "rfft2",
    exclude_dc: bool = True,
    spacing: tuple[float, float] | None = None,
    edges: np.ndarray | None = None,
) -> np.ndarray:
    """Average a power spectrum over radial frequency intervals (legacy API).

    Parameters
    ----------
    spectrum : ndarray, shape (..., ny, n_kx)
        Real power in native FFT order. Leading dimensions are preserved.
    grid_shape : pair of int
        Original spatial dimensions (ny, nx).
    n_bins : int, default 30
        Number of equal-width intervals when edges is omitted.
    fft_solver : {'fft2', 'rfft2'}, default 'rfft2'
        Input layout; n_kx is nx or nx // 2 + 1. Implicit rfft2 cells receive
        their full-spectrum multiplicity.
    exclude_dc : bool, default True
        Exclude the zero-frequency cell. Nyquist cells remain eligible.
    spacing : pair of float or None, default None
        Spatial pitches (dy, dx). None means (1, 1), giving cycles per pixel.
    edges : ndarray or None, default None
        Increasing radial edges in cycles per spatial unit. Overrides n_bins.
        Intervals are [left, right), with the final edge included.

    Returns
    -------
    ndarray, shape (..., n_radius)
        Mean power per interval. Out-of-range cells are ignored; intervals
        without eligible cells are NaN.

    Notes
    -----
    Retained for existing imports. For sample analysis, prefer the comparator's
    compute_spectra() method, which resolves shared edges and spacing.
    """
    return _bin_radial_spectrum(
        spectrum,
        grid_shape,
        edges=edges,
        spacing=(1.0, 1.0) if spacing is None else spacing,
        fft_solver=fft_solver,
        n_bins=n_bins,
        exclude_dc=exclude_dc,
    )


def radial_bin_counts(
    grid_shape: tuple[int, int],
    n_bins: int = 30,
    fft_solver: str = "rfft2",
    exclude_dc: bool = True,
    spacing: tuple[float, float] | None = None,
    edges: np.ndarray | None = None,
) -> np.ndarray:
    """Count Fourier cells per radial interval (legacy API).

    Parameters
    ----------
    grid_shape : pair of int
        Original spatial dimensions (ny, nx).
    n_bins : int, default 30
        Number of equal-width intervals when edges is omitted.
    fft_solver : {'fft2', 'rfft2'}, default 'rfft2'
        FFT layout; implicit rfft2 cells count with full-spectrum multiplicity.
    exclude_dc : bool, default True
        Exclude the zero-frequency cell. Nyquist cells remain eligible.
    spacing : pair of float or None, default None
        Spatial pitches (dy, dx). None means (1, 1), giving cycles per pixel.
    edges : ndarray or None, default None
        Increasing radial edges in cycles per spatial unit. Overrides n_bins;
        the final edge is included.

    Returns
    -------
    ndarray, shape (n_radius,)
        Full-spectrum-equivalent cell counts. Zero means an unsupported
        interval; counts are not independent observations or mode pairs.
    """
    return _radial_bin_cell_counts(
        grid_shape,
        edges=edges,
        spacing=(1.0, 1.0) if spacing is None else spacing,
        fft_solver=fft_solver,
        n_bins=n_bins,
        exclude_dc=exclude_dc,
    )


def stream_radial_features(
    spectrum_chunk_fn: Any,
    n_genes: int,
    grid_shape: tuple[int, int],
    *,
    chunk_size: int,
    fft_solver: str,
    spacing: tuple[float, float] | None,
    edges: np.ndarray,
    pbar: Any = None,
) -> np.ndarray:
    """Stream power chunks into radial means (legacy API).

    Parameters
    ----------
    spectrum_chunk_fn : callable
        (start, stop) -> real power, shape (stop-start, ny, n_kx), in native
        FFT order. n_kx is nx for fft2 or nx // 2 + 1 for rfft2.
    n_genes : int
        Total number of features supplied by the callback.
    grid_shape : pair of int
        Original spatial dimensions (ny, nx).
    chunk_size : int
        Features per callback block.
    fft_solver : {'fft2', 'rfft2'}
        Layout supplied by the callback.
    spacing : pair of float or None
        Spatial pitches (dy, dx). None means (1, 1), giving cycles per pixel.
    edges : ndarray, shape (n_radius + 1,)
        Shared radial edges in cycles per spatial unit; final edge included.
    pbar : progress bar, optional
        Updated once per block.

    Returns
    -------
    ndarray, shape (n_genes, n_radius)
        Radial means, excluding DC and retaining Nyquist. Unsupported bins
        are NaN. This function computes no FFTs.
    """
    return _stream_radial_features(
        spectrum_chunk_fn,
        n_genes,
        grid_shape,
        edges=edges,
        spacing=(1.0, 1.0) if spacing is None else spacing,
        fft_solver=fft_solver,
        chunk_size=chunk_size,
        pbar=pbar,
    )


def stream_polar_features(
    spectrum_chunk_fn: Any,
    n_genes: int,
    grid_shape: tuple[int, int],
    angle_deg: float,
    *,
    chunk_size: int,
    freq_edges: np.ndarray,
    spacing: tuple[float, float] | None,
    n_theta: int,
    fft_solver: str,
    pbar: Any = None,
    frequency_padding: int = 0,
) -> np.ndarray:
    """Stream power chunks into aligned polar-sector means (legacy API).

    Parameters
    ----------
    spectrum_chunk_fn : callable
        (start, stop) -> real power, shape (stop-start, ny, n_kx), in native
        FFT order. Padded inputs use (ny + 2*p, nx + 2*p) with fft2 layout.
    n_genes : int
        Total number of features supplied by the callback.
    grid_shape : pair of int
        Original spatial dimensions (ny, nx), before Fourier-mode padding.
    angle_deg : float
        Alignment angle in degrees; sampling uses the inverse rotation.
    chunk_size : int
        Features per callback block.
    freq_edges : ndarray, shape (n_radius + 1,)
        Shared radial edges in cycles per spatial unit.
    spacing : pair of float or None
        Spatial pitches (dy, dx). For compatibility, None means
        (1 / ny, 1 / nx): each axis spans one unit, unlike the radial helper.
    n_theta : int
        Angular sectors over [0, pi), centered at j * pi / n_theta.
    fft_solver : {'fft2', 'rfft2'}
        Input layout. Unpadded n_kx is nx or nx // 2 + 1.
    pbar : progress bar, optional
        Updated once per block.
    frequency_padding : int, default 0
        Evaluated NUFFT modes p on each edge. Padded interpolation uses the
        evaluated border; unpadded FFT interpolation wraps periodically.

    Returns
    -------
    ndarray, shape (n_genes, n_radius * n_theta)
        Area-weighted sector means, with angle varying fastest. Unsupported
        bands are NaN. Inputs must already be centered and normalized.
    """
    if spacing is None:
        spacing = (1.0 / grid_shape[0], 1.0 / grid_shape[1])
    return _stream_polar_features(
        spectrum_chunk_fn,
        n_genes,
        grid_shape,
        edges=freq_edges,
        spacing=spacing,
        fft_solver=fft_solver,
        chunk_size=chunk_size,
        n_theta=n_theta,
        angle_deg=angle_deg,
        frequency_padding=frequency_padding,
        pbar=pbar,
    )


# ---------------------------------------------------------------------------
# Per-sample spectra
# ---------------------------------------------------------------------------


def compute_sample_spectrum(
    sample: np.ndarray,
    fft_solver: str = "rfft2",
    workers: int | None = None,
    return_dc: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute the 2D power spectrum of every gene in a single sample.

    The spatial signal is **mean-centered per gene** in Fourier space so that the
    resulting power spectrum carries only the *AC* component of the pattern —
    i.e. the ``k=0`` (DC) bin is exactly zero and low-``k`` leakage from per-
    sample mean shifts is eliminated. The separated DC scalars (the per-sample
    per-gene grid means) can be returned alongside the spectrum with
    ``return_dc=True`` and are the natural target for a *classical differential
    expression* test complementary to the spectral pattern test.

    Power is divided by ``(ny * nx)**2``: Fourier coefficients represent
    averages over observations, so denser sampling of the same physical field
    does not inflate its amplitude. DC expression means remain unscaled.

    Parameters
    ----------
    sample : np.ndarray
        Rasterized expression of shape ``(n_genes, ny, nx)``.
    fft_solver : {'fft2', 'rfft2'}, default 'rfft2'
        FFT routine forwarded to :func:`sonic.kernels.fft.power_spectrum_2d`.
    workers : int, optional
        Parallel workers forwarded to :mod:`scipy.fft`.
    return_dc : bool, default False
        If True, also return a ``(n_genes,)`` array of per-gene grid means (DC
        scalars of the *uncentered* signal).

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Power spectra of shape ``(n_genes, ny, n_kx)``. If ``return_dc=True``,
        also returns a ``(n_genes,)`` DC array.

    Raises
    ------
    ValueError
        If ``sample`` is not 3D.
    """
    if sample.ndim != 3:
        raise ValueError(f"sample must be 3D (n_genes, ny, nx), got shape {sample.shape}")

    # DC scalars always come from the *uncentered* grid.
    dc = sample.mean(axis=(1, 2))

    # Move feature axis to last so power_spectrum_2d treats it as M.
    moved = np.moveaxis(sample, 0, -1)
    p = power_spectrum_2d(moved, fft_solver=fft_solver, workers=workers, center=True)
    p /= (sample.shape[1] * sample.shape[2]) ** 2
    spec = np.moveaxis(p, -1, 0)

    if return_dc:
        return spec, dc
    return spec


def power_spectrum_anisotropy(
    power: np.ndarray,
    grid_shape: tuple[int, int],
    spacing: tuple[float, float] = (1.0, 1.0),
    *,
    fft_solver: str = "auto",
) -> pd.DataFrame:
    """Second angular moment of raw 2-D power spectra, excluding DC.

    Each input row is treated as one spectrum ``(ny, n_kx)``. The result
    measures how concentrated that spectrum is around a single orientation
    axis using the normalized second angular moment:
    ``|sum_k P(k) exp(2i theta_k)| / sum_k P(k)``. The value is near ``1`` for
    power concentrated along one axis and near ``0`` when power is angularly
    balanced. Angles are axial, so ``0`` and ``180`` degrees are equivalent;
    the reported ``dominant_angle_deg`` lies in ``[-90, 90]``.

    For ``rfft2`` half-plane spectra, the implicit Hermitian half-plane is
    reconstructed so the result matches the equivalent full ``fft2`` spectrum
    on both even and odd grids. Negative powers are clipped to zero before
    aggregation.

    Parameters
    ----------
    power : np.ndarray
        Raw power spectra of shape ``(n_spectra, ny, n_kx)``. ``n_kx`` may be
        ``nx`` for ``fft2`` output or ``nx // 2 + 1`` for ``rfft2`` output.
    grid_shape : tuple[int, int]
        Original image grid shape ``(ny, nx)``.
    spacing : tuple[float, float], default (1.0, 1.0)
        Grid spacing ``(dy, dx)`` used to compute physical frequency angles.
        Anisotropic spacing changes the frequency-space angle of each bin.
    fft_solver : {'auto', 'fft2', 'rfft2'}, default 'auto'
        Spectrum layout. ``'auto'`` infers the layout from ``power.shape[-1]``;
        pass an explicit value for ambiguous very small grids.

    Returns
    -------
    pandas.DataFrame
        One row per input spectrum with columns ``anisotropy``,
        ``dominant_angle_deg``, and ``total_power``.

    Raises
    ------
    ValueError
        If the input shape, grid shape, spacing, or FFT layout is invalid.
    """
    power = np.asarray(power, dtype=float)
    ny, nx = _validate_grid_shape(grid_shape)
    dy, dx = _validate_spacing(spacing)
    if power.ndim != 3 or power.shape[-2] != ny:
        raise ValueError(f"power must have shape (n_spectra, {ny}, n_kx), got {power.shape}.")

    # Resolve whether the spectrum is a full fft2 grid or an rfft2 half-plane;
    # the downstream angular grid depends on this layout.
    n_kx = int(power.shape[-1])
    resolved_solver, expected_kx = _resolve_power_layout(n_kx, nx, fft_solver)
    if resolved_solver == "rfft2":
        # Expanding avoids subtle even-grid Nyquist sign conventions. In
        # particular, doubling the stored half-plane at the stored angle is not
        # equivalent to the full fftfreq layout on the Nyquist row.
        power = _as_full_power_spectrum(power, (ny, nx), fft_solver="rfft2")
        resolved_solver = "fft2"
        expected_kx = nx
    kx_grid, ky_grid, layout_weights = _layout_frequency_grid(
        ny, nx, expected_kx, dy, dx, resolved_solver
    )

    # The zero-frequency bin describes total expression, not orientation, so it
    # is excluded from the angular moment.
    radius = np.sqrt(kx_grid * kx_grid + ky_grid * ky_grid)
    theta = np.arctan2(ky_grid, kx_grid)
    mask = radius > 0

    # Clip small numerical negatives to zero. rfft2 inputs have already been
    # expanded to full fft2 layout, so no half-plane weighting remains here.
    weights = np.maximum(power[:, mask], 0.0) * layout_weights[mask][None, :]
    total = weights.sum(axis=1)

    # The second angular moment is axial: directions separated by 180 degrees
    # are equivalent, which is why the angle enters as 2 * theta.
    cos2 = (weights * np.cos(2.0 * theta[mask])).sum(axis=1)
    sin2 = (weights * np.sin(2.0 * theta[mask])).sum(axis=1)
    anisotropy = np.divide(
        np.sqrt(cos2 * cos2 + sin2 * sin2),
        total,
        out=np.full_like(total, np.nan, dtype=float),
        where=total > 0,
    )
    angle = 0.5 * np.degrees(np.arctan2(sin2, cos2))
    angle = np.where(total > 0, angle, np.nan)
    return pd.DataFrame(
        {
            "anisotropy": anisotropy,
            "dominant_angle_deg": angle,
            "total_power": total,
        }
    )


def effective_rank(
    cov: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    assume_psd: bool = False,
) -> float:
    r"""Effective rank (participation ratio) of a covariance matrix.

    Computes

    .. math::
        K_\mathrm{eff} \;=\; \frac{\big(\sum_k \lambda_k\big)^2}{\sum_k \lambda_k^2}

    where :math:`\lambda_k` are the eigenvalues of ``cov`` (or, when
    ``weights`` is given, of :math:`W^{1/2} \mathrm{cov}\, W^{1/2}` with
    :math:`W = \mathrm{diag}(w)`). The result is bounded by 1 (rank-1,
    all variance on a single direction) and ``K = cov.shape[0]``
    (uniformly spread, all eigenvalues equal). It coincides with the
    standard inverse Herfindahl index of the (normalised) eigenvalue
    distribution and quantifies the "effective number of independent
    components" of a quadratic-form statistic
    :math:`T^2 = X^\top \mathrm{cov} X` where :math:`X \sim \mathcal{N}(0,I)`.

    Parameters
    ----------
    cov : np.ndarray
        Symmetric ``(K, K)`` covariance matrix. Negative eigenvalues
        from numerical noise are clipped to 0.
    weights : np.ndarray, optional
        Non-negative weights of length ``K``. When provided, returns the
        effective rank of the weighted form :math:`W^{1/2} \mathrm{cov}\,W^{1/2}`,
        useful for analysing how a frequency-weighted L2 statistic
        actually distributes its sensitivity across eigen-directions.
    assume_psd : bool, default False
        If the covariance is known to be symmetric positive semidefinite,
        compute ``trace(C)**2 / sum(C*C)`` without an eigendecomposition.
        The default retains clipping of negative eigenvalues. Internal
        covariance estimates use this faster path.

    Returns
    -------
    float
        Effective rank in ``[1, K]``. Returns ``nan`` when the trace
        is non-positive (degenerate covariance).

    Examples
    --------
    >>> import numpy as np
    >>> effective_rank(np.eye(10))
    10.0
    >>> # Rank-1 outer product
    >>> v = np.zeros(10); v[0] = 1.0
    >>> effective_rank(np.outer(v, v))
    1.0
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be a square 2D matrix, got shape {cov.shape}.")
    n_bins = cov.shape[0]
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n_bins,):
            raise ValueError(f"weights must have length n_bins={n_bins}, got shape {w.shape}.")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative.")
        sqW = np.sqrt(w)
        cov = (sqW[:, None] * cov) * sqW[None, :]
    if assume_psd:
        scale = np.max(np.abs(cov), initial=0.0)
        if scale == 0:
            return float("nan")
        scaled = cov / scale
        trace = float(np.trace(scaled))
        if trace <= 0:
            return float("nan")
        return float(trace**2 / np.einsum("ij,ij->", scaled, scaled))
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    s = float(eigvals.sum())
    if s <= 0:
        return float("nan")
    return float(s * s / float((eigvals**2).sum()))


def gene_pattern_diversity(
    spectra: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    eps: float = 1e-12,
) -> float:
    """Measure how variation in log power across genes is distributed across bins.

    Parameters
    ----------
    spectra : np.ndarray
        Power spectra for one sample with shape ``(n_genes, n_bins)``.
    weights : np.ndarray, optional
        Nonnegative weights for the frequency bins. The default gives each
        bin equal weight.
    eps : float, default 1e-12
        Lower bound on power before taking the natural logarithm.

    Returns
    -------
    float
        Effective rank of the covariance of log power across genes. A value
        near one means that one linear combination of bins accounts for most
        of the variance. Equal positive eigenvalues give a rank equal to
        the number of bins.
    """
    if spectra.ndim != 2:
        raise ValueError(f"spectra must be (n_genes, n_bins), got shape {spectra.shape}.")
    log_s = np.log(np.maximum(spectra, eps))
    centred = log_s - log_s.mean(axis=0, keepdims=True)
    G = log_s.shape[0]
    cov = (centred.T @ centred) / max(G - 1, 1)
    return effective_rank(cov, weights=weights, assume_psd=True)


def stream_geomean_landmark(
    spectrum_chunk_fn: Any,
    n_genes: int,
    grid_shape: tuple[int, int],
    *,
    chunk_size: int,
    gene_subset: np.ndarray | None = None,
    eps: float = 1e-12,
    pbar: Any = None,
) -> np.ndarray:
    """Stream a sample's per-(k) **geometric-mean** spectrum as a single landmark.

    Accumulates the across-gene mean of ``log(power + eps)`` one gene-chunk at a
    time, then exponentiates — yielding ``(1, ny, n_kx)``. The geomean is the
    amplitude-invariant consensus orientation template (per-gene brightness
    becomes an additive log constant that cannot move the angular argmax),
    mirroring :func:`sonic.comparators.normalization.normalize_background`'s
    cross-gene geometric mean. Peak
    memory is ``O(chunk · ny · nx)`` plus one ``(ny, n_kx)`` accumulator — the
    full ``(n_genes, ny, n_kx)`` stack is never held.

    Parameters
    ----------
    spectrum_chunk_fn : callable
        ``(start, stop) -> (stop-start, ny, n_kx)`` power-spectrum chunk.
    n_genes : int
        Total genes supplied by the callback.
    grid_shape : pair of int
        Original spatial dimensions (ny, nx).
    chunk_size : int
        Genes per callback block.
    gene_subset : np.ndarray, optional
        If given, the geometric mean is taken over only these gene indices
        (into ``0..n_genes-1``), chunked so the subset is never fully cached.
        Used to build a single geomean landmark from an explicit landmark-gene
        set when caching those genes' full 2D spectra would exceed budget.
        ``None`` (default) averages over all ``n_genes``.
    eps : float, default 1e-12
        Floor added before the log (keeps zeros finite).
    pbar : tqdm, optional
        Progress bar; ``.update(1)`` per chunk.

    Returns
    -------
    np.ndarray
        ``(1, ny, n_kx)`` geometric-mean landmark spectrum.
    """
    log_sum: np.ndarray | None = None
    if gene_subset is None:
        indices: Any = range(0, n_genes, chunk_size)
        count = n_genes

        def _chunk(start: int) -> np.ndarray:
            return spectrum_chunk_fn(start, min(start + chunk_size, n_genes))

    else:
        subset = np.asarray(gene_subset, dtype=int)
        count = int(subset.size)
        indices = range(0, count, chunk_size)

        def _chunk(start: int) -> np.ndarray:
            cols = subset[start : min(start + chunk_size, count)]
            # Materialise this slice of the subset gene-by-gene (each is one
            # spectrum) so the full subset is never held at once.
            return np.concatenate([spectrum_chunk_fn(int(j), int(j) + 1) for j in cols], axis=0)

    for start in indices:
        spec_chunk = _chunk(start)
        contrib = np.log(spec_chunk + eps).sum(axis=0)
        log_sum = contrib if log_sum is None else log_sum + contrib
        del spec_chunk, contrib
        if pbar is not None:
            pbar.update(1)
    if log_sum is None:
        raise ValueError("stream_geomean_landmark: must average over >= 1 gene.")
    return np.exp(log_sum / float(count))[None, ...]


# ---------------------------------------------------------------------------
# Optional 2D rotation alignment
# ---------------------------------------------------------------------------

_ALIGNMENT_N_THETA = 180
_ALIGNMENT_N_RADIUS = 64


def _build_landmark_polar_stack(
    spectra: np.ndarray,
    grid_shape: tuple[int, int],
    fft_solver: str,
    n_theta: int,
    freq_edges: np.ndarray,
    spacing: tuple[float, float],
    frequency_padding: int = 0,
) -> np.ndarray:
    """Build a ``(n_landmarks, n_theta, n_radius)`` polar stack for one sample.

    Each landmark's 2D spectrum is fftshifted (DC at centre), resampled onto
    the polar grid, and zero-meaned along theta so the DC angular component
    doesn't dominate the cross-correlation.
    """
    full = _as_full_power_spectrum(
        spectra, grid_shape, fft_solver=fft_solver, frequency_padding=frequency_padding
    )
    shifted = np.fft.fftshift(full, axes=(-2, -1))
    coords, n_radius = _polar_bin_center_coords(
        grid_shape,
        spacing=spacing,
        edges=freq_edges,
        n_theta=n_theta,
        frequency_padding=frequency_padding,
    )
    out = np.empty((shifted.shape[0], n_theta, n_radius), dtype=float)
    for j in range(shifted.shape[0]):
        polar = (
            scipy.ndimage.map_coordinates(
                shifted[j],
                coords,
                order=1,
                mode="nearest" if frequency_padding else "grid-wrap",
            )
            .reshape(n_radius, n_theta)
            .T
        )
        out[j] = polar - polar.mean(axis=0, keepdims=True)
    return out


def _stream_landmark_polar(
    spectrum_chunk_fn: Any,
    n_genes: int,
    grid_shape: tuple[int, int],
    *,
    chunk_size: int,
    fft_solver: str,
    freq_edges: np.ndarray,
    spacing: tuple[float, float],
    geomean: bool,
    frequency_padding: int = 0,
    pbar: Any = None,
) -> np.ndarray:
    """Reduce power chunks to compact alignment landmarks, discarding each dense chunk.

    Return (landmark, angle, radius), with one landmark for the Cartesian
    geometric mean or one per selected gene. Progress advances once per chunk.
    """

    def polar(power):
        return _build_landmark_polar_stack(
            power,
            grid_shape,
            fft_solver,
            _ALIGNMENT_N_THETA,
            freq_edges,
            spacing,
            frequency_padding,
        )

    if geomean:
        # Geometric averaging must precede interpolation; these operations do not commute.
        return polar(
            stream_geomean_landmark(
                spectrum_chunk_fn,
                n_genes,
                grid_shape,
                chunk_size=chunk_size,
                pbar=pbar,
            )
        )
    out = np.empty((n_genes, _ALIGNMENT_N_THETA, len(freq_edges) - 1))
    for start in range(0, n_genes, chunk_size):
        stop = min(start + chunk_size, n_genes)
        out[start:stop] = polar(spectrum_chunk_fn(start, stop))
        if pbar is not None:
            pbar.update(1)
    return out


def estimate_rotations_from_landmarks(
    landmark_spectra: Sequence[np.ndarray],
    grid_shapes: Sequence[tuple[int, int]],
    *,
    fft_solver: str = "fft2",
    reference_index: int = 0,
    n_theta: int = _ALIGNMENT_N_THETA,
    n_radius: int = _ALIGNMENT_N_RADIUS,
    spacings: Sequence[tuple[float, float]] | None = None,
    progress: bool = False,
    frequency_padding: int = 0,
) -> np.ndarray:
    """
    Estimate the per-sample rotation that best aligns every landmark
    spectrum to the reference sample's corresponding landmark.

    For each non-reference sample the routine picks a single rotation angle
    that maximises the **sum over landmarks** of the per-landmark circular
    cross-correlation along the polar-angle axis — i.e. each landmark
    aligns to its same-index counterpart in the reference (not to a mean
    template). This is strictly stronger than mean-template alignment
    because it ignores cross-landmark noise (the off-diagonal ``i ≠ j``
    terms that mean-of-means picks up) and picks up anisotropy shared
    across every landmark at a common orientation.

    Parameters
    ----------
    landmark_spectra : sequence of np.ndarray
        Per-sample landmark spectra. Shape ``(n_landmarks, ny, n_kx)`` with
        ``(ny, n_kx)`` following ``fft_solver``. The first dimension
        (``n_landmarks``) must match across samples — landmark ``j`` in
        sample A is compared to landmark ``j`` in sample B.
    grid_shapes : sequence of tuple[int, int]
        Per-sample ``(ny, nx)`` of the original rasterized image.
    fft_solver : {'fft2', 'rfft2'}, default 'fft2'
        FFT layout of ``landmark_spectra`` — rfft2 spectra are expanded
        to full 2D before resampling to preserve angular content.
    reference_index : int, default 0
        Which sample's landmarks act as the rotation reference (its angle
        is fixed at 0).
    n_theta : int, default 180
        Angular resolution of the polar resampling. Recovered angles are
        accurate to ``180 / n_theta`` degrees.
    n_radius : int, default 64
        Radial resolution of the polar resampling.
    spacings : sequence of (dy, dx), optional
        Physical pixel spacings for each sample; defaults to unit spacing.
        Landmarks share physical radii up to the smallest sampling Nyquist.
    progress : bool, default False
        If True, show a tqdm bar over non-reference samples.
    frequency_padding : int, default 0
        Evaluated Fourier-mode border on each edge. Padded input has
        shape (n_landmarks, ny + 2*p, nx + 2*p), where p is the border width.
        Grid shapes and spacings describe the original unpadded grid.

    Returns
    -------
    angles_deg : np.ndarray
        ``(n_samples,)`` recovered rotation angles in degrees. Reference
        angle is exactly 0.

    Raises
    ------
    ValueError
        If ``reference_index`` is out of range or any two samples have
        inconsistent ``n_landmarks``.
    """
    n_samples = len(landmark_spectra)
    if reference_index < 0 or reference_index >= n_samples:
        raise ValueError(f"reference_index {reference_index} out of range [0, {n_samples})")
    if len(grid_shapes) != n_samples:
        raise ValueError("grid_shapes must have one entry per sample.")
    if spacings is None:
        spacings = [(1.0, 1.0)] * n_samples
    if len(spacings) != n_samples:
        raise ValueError("spacings must have one entry per sample.")
    spacings = [_validate_spacing(s) for s in spacings]
    f_max = min(0.5 / max(s) for s in spacings)
    freq_edges = np.linspace(0.0, f_max, n_radius + 1)

    n_landmarks = landmark_spectra[reference_index].shape[0]
    for i, s in enumerate(landmark_spectra):
        if s.shape[0] != n_landmarks:
            raise ValueError(
                f"landmark_spectra[{i}] has n_landmarks={s.shape[0]}, "
                f"expected {n_landmarks} (must match across samples)."
            )

    polar = [
        _build_landmark_polar_stack(
            spectrum, shape, fft_solver, n_theta, freq_edges, spacing, frequency_padding
        )
        for spectrum, shape, spacing in zip(landmark_spectra, grid_shapes, spacings, strict=True)
    ]
    return _estimate_rotations_from_polar(polar, reference_index=reference_index, progress=progress)


def _estimate_rotations_from_polar(
    landmarks: Sequence[np.ndarray], *, reference_index: int = 0, progress: bool = False
) -> np.ndarray:
    """Correlate compact (landmark, angle, radius) stacks along the angle axis."""
    ref_hat = np.fft.fft(landmarks[reference_index], axis=1)
    n_theta = ref_hat.shape[1]
    angles = np.zeros(len(landmarks))
    indices = range(len(landmarks))
    if progress:
        indices = tqdm(indices, desc="Rotation estimation")
    for i in indices:
        if i == reference_index:
            continue
        cur_hat = np.fft.fft(landmarks[i], axis=1)
        corr = np.real(np.fft.ifft(ref_hat * np.conj(cur_hat), axis=1))
        total = corr.sum(axis=(0, 2))
        angles[i] = int(np.argmax(total)) * 180.0 / n_theta
    return angles


def apply_rotations_to_spectra(
    spectra: Sequence[np.ndarray],
    grid_shapes: Sequence[tuple[int, int]],
    angles_deg: np.ndarray,
    *,
    fft_solver: str = "fft2",
    spacings: Sequence[tuple[float, float]] | None = None,
    progress: bool = False,
) -> list[np.ndarray]:
    """
    Rotate each sample's 2D power spectra by a per-sample angle.

    Parameters
    ----------
    spectra : sequence of np.ndarray
        Per-sample 2D power spectra — any first-axis dimension (e.g. full
        ``n_genes``). Shape ``(n, ny, n_kx)`` with ``(ny, n_kx)`` matching
        ``fft_solver``.
    grid_shapes : sequence of tuple[int, int]
        Per-sample ``(ny, nx)`` of the original rasterized image.
    angles_deg : np.ndarray
        Per-sample rotation angles in degrees (e.g. produced by
        :func:`estimate_rotations_from_landmarks`). Length must equal
        ``len(spectra)``.
    fft_solver : {'fft2', 'rfft2'}, default 'fft2'
        FFT layout of ``spectra``.
    spacings : sequence of (dy, dx), optional
        Physical pixel spacings per sample; defaults to unit spacing. Rotations
        preserve physical frequencies, including on rectangular FFT grids.
    progress : bool, default False
        Show a tqdm bar across samples.

    Returns
    -------
    rotated : list of np.ndarray
        Per-sample rotated spectra with the same shape as the input.

    Notes
    -----
    Rotation is done on the **2D power spectrum** (fftshifted so DC sits at
    the centre), not back on the spatial image. That is enough for any
    downstream analysis that operates on aligned spectra (radial or 2D-bin
    tests). Samples whose angle is exactly 0 are passed through as-is.
    """
    if len(angles_deg) != len(spectra):
        raise ValueError(
            f"angles_deg length {len(angles_deg)} does not match spectra length {len(spectra)}."
        )
    if len(grid_shapes) != len(spectra):
        raise ValueError(
            f"grid_shapes length {len(grid_shapes)} does not match spectra length {len(spectra)}."
        )
    if spacings is None:
        spacings = [(1.0, 1.0)] * len(spectra)
    if len(spacings) != len(spectra):
        raise ValueError("spacings must have one entry per sample.")
    spacings = [_validate_spacing(s) for s in spacings]
    out: list[np.ndarray] = []
    # strict=False: lengths are already verified above.
    iter_samples: Any = enumerate(zip(spectra, grid_shapes, strict=False))
    if progress:
        iter_samples = tqdm(iter_samples, total=len(spectra), desc="Rotation application")
    for i, (spec_i, shape) in iter_samples:
        angle_deg = float(angles_deg[i])
        if angle_deg == 0.0:
            out.append(np.asarray(spec_i).copy())
            continue
        full = _as_full_power_spectrum(spec_i, shape, fft_solver=fft_solver)  # (n, ny, nx)
        full_shift = np.fft.fftshift(full, axes=(-2, -1))
        # Convert the inverse physical rotation into FFT-pixel coordinates.
        # fftshift's DC is at floor(n/2), including for even dimensions.
        extent_y, extent_x = np.asarray(shape) * spacings[i]
        c, s = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
        matrix = np.eye(3)
        matrix[1:, 1:] = [[c, -s * extent_y / extent_x], [s * extent_x / extent_y, c]]
        center = np.array([0, shape[0] // 2, shape[1] // 2])
        rot = scipy.ndimage.affine_transform(
            full_shift, matrix, offset=center - matrix @ center, order=1, mode="grid-wrap"
        )
        # Interpolate periodically across the last sampled frequency, but do
        # not alias rotated corners from outside the physical Nyquist band.
        yy = (np.arange(shape[0]) - center[1])[:, None]
        xx = (np.arange(shape[1]) - center[2])[None, :]
        outside = (np.abs(matrix[1, 1] * yy + matrix[1, 2] * xx) > shape[0] / 2) | (
            np.abs(matrix[2, 1] * yy + matrix[2, 2] * xx) > shape[1] / 2
        )
        rot[:, outside] = 0.0
        rot = np.fft.ifftshift(rot, axes=(-2, -1))
        if fft_solver == "rfft2":
            ny, nx = shape
            half = nx // 2 + 1
            rot = rot[..., :half]
        out.append(rot)
    return out


def align_spectra_by_rotation(
    landmark_spectra: Sequence[np.ndarray],
    grid_shapes: Sequence[tuple[int, int]],
    *,
    target_spectra: Sequence[np.ndarray] | None = None,
    fft_solver: str = "fft2",
    reference_index: int = 0,
    n_theta: int = _ALIGNMENT_N_THETA,
    n_radius: int = _ALIGNMENT_N_RADIUS,
    spacings: Sequence[tuple[float, float]] | None = None,
    progress: bool = False,
) -> tuple[list[np.ndarray] | None, np.ndarray]:
    """
    Two-step rotation alignment: estimate per-sample rotations from
    **landmark** spectra (whose first dimension must match across samples),
    then apply those rotations to a separate set of **target** spectra (the
    full gene panel for each sample, typically a superset of the
    landmarks).

    This is a convenience wrapper around
    :func:`estimate_rotations_from_landmarks` and
    :func:`apply_rotations_to_spectra`. Calling those directly is the
    right pattern when you want to inspect / cache the per-sample angles
    before applying them.

    Implementation
    --------------
    For every non-reference sample:

    1. Expand each landmark's 2D power spectrum to full-fft2 layout,
       fftshift so DC sits at the centre, and resample onto a polar
       ``(n_theta, n_radius)`` grid.
    2. Compute per-landmark circular cross-correlation along the
       polar-angle axis against the reference sample's same-index
       landmark. **Every landmark contributes its own cross-correlation**
       and the per-sample rotation is the angle that maximises the sum
       across landmarks (and across radii). Mean-template alignment —
       what the previous implementation did — was strictly weaker
       because the off-diagonal ``i ≠ j`` pair terms in
       ``corr(mean(a), mean(b))`` are pure noise.
    3. Rotate every entry of ``target_spectra[i]`` (if supplied) by the
       recovered angle.

    Parameters
    ----------
    landmark_spectra : sequence of np.ndarray
        Per-sample landmark spectra, shape ``(n_landmarks, ny, n_kx)``
        per sample. ``n_landmarks`` must match across samples.
    grid_shapes : sequence of tuple[int, int]
        Per-sample ``(ny, nx)`` of the original rasterized image.
    target_spectra : sequence of np.ndarray, optional
        Per-sample spectra to which the recovered rotations are applied.
        Any first-axis dimension (e.g. full gene panel). If ``None``, only
        the angles are returned.
    fft_solver : {'fft2', 'rfft2'}, default 'fft2'
        FFT layout of both inputs. ``fft2`` is recommended so the full
        angular content is present.
    reference_index : int, default 0
    n_theta : int, default 180
    n_radius : int, default 64
    spacings : sequence of (dy, dx), optional
        Physical pixel spacings per sample; defaults to unit spacing.
    progress : bool, default False

    Returns
    -------
    rotated : list of np.ndarray or None
        Per-sample rotated target spectra (or ``None`` when
        ``target_spectra`` is omitted).
    angles_deg : np.ndarray
        ``(n_samples,)`` recovered rotation angles in degrees. Reference
        angle is 0.

    Raises
    ------
    ValueError
        If ``reference_index`` is out of range, if ``landmark_spectra``
        samples disagree on ``n_landmarks``, or if
        ``target_spectra`` length does not match.
    """
    angles = estimate_rotations_from_landmarks(
        landmark_spectra,
        grid_shapes,
        fft_solver=fft_solver,
        reference_index=reference_index,
        n_theta=n_theta,
        n_radius=n_radius,
        spacings=spacings,
        progress=progress,
    )
    if target_spectra is None:
        return None, angles
    if len(target_spectra) != len(landmark_spectra):
        raise ValueError(
            f"target_spectra length {len(target_spectra)} does not match "
            f"landmark_spectra length {len(landmark_spectra)}."
        )
    rotated = apply_rotations_to_spectra(
        target_spectra,
        grid_shapes,
        angles,
        fft_solver=fft_solver,
        spacings=spacings,
        progress=progress,
    )
    return rotated, angles
