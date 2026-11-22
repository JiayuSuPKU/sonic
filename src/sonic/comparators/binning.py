"""Private frequency grids, support counts and radial/polar spectrum binning.

Auto- and cross-spectral products use the same nonnegative mean operators.
Radial means include every mode within the requested edges except DC by
default, including Nyquist modes. Rotated polar means use adaptive,
area-weighted bilinear quadrature. All geometry helpers require explicit
spatial spacing; alignment remains in features.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix


def _radial_frequency_grid(
    grid_shape: tuple[int, int],
    *,
    spacing: tuple[float, float],
    fft_solver: str,
) -> np.ndarray:
    """Return radial frequencies in cycles per spatial unit, shape (ny, n_kx)."""
    ny, nx = grid_shape
    dy, dx = _validate_spacing(spacing)
    ky = np.fft.fftfreq(ny, d=dy)
    if fft_solver == "fft2":
        kx = np.fft.fftfreq(nx, d=dx)
    elif fft_solver == "rfft2":
        kx = np.fft.rfftfreq(nx, d=dx)
    else:
        raise ValueError(f"fft_solver must be 'fft2' or 'rfft2', got {fft_solver!r}")
    return np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)


def _validate_grid_shape(grid_shape: tuple[int, int]) -> tuple[int, int]:
    """Return a validated ``(ny, nx)`` pair."""
    if len(grid_shape) != 2:
        raise ValueError(f"grid_shape must be a pair (ny, nx), got {grid_shape!r}.")
    ny, nx = int(grid_shape[0]), int(grid_shape[1])
    if ny <= 0 or nx <= 0:
        raise ValueError(f"grid_shape values must be positive, got {grid_shape!r}.")
    return ny, nx


def _validate_spacing(spacing: tuple[float, float]) -> tuple[float, float]:
    """Return a validated ``(dy, dx)`` pair."""
    if spacing is None or len(spacing) != 2:
        raise ValueError(f"spacing must be a pair (dy, dx), got {spacing!r}.")
    dy, dx = float(spacing[0]), float(spacing[1])
    if not np.isfinite((dy, dx)).all() or dy <= 0 or dx <= 0:
        raise ValueError(f"spacing values must be positive and finite, got {spacing!r}.")
    return dy, dx


def _resolve_power_layout(n_kx: int, nx: int, fft_solver: str) -> tuple[str, int]:
    """Resolve a power-spectrum layout and expected ``kx`` length."""
    rfft_kx = nx // 2 + 1
    if fft_solver == "auto":
        if n_kx == nx:
            return "fft2", nx
        if n_kx == rfft_kx:
            return "rfft2", rfft_kx
        raise ValueError(
            f"power last axis {n_kx} is incompatible with grid width {nx}; "
            f"expected {nx} for fft2 or {rfft_kx} for rfft2."
        )
    if fft_solver not in {"fft2", "rfft2"}:
        raise ValueError(f"fft_solver must be 'auto', 'fft2', or 'rfft2', got {fft_solver!r}.")
    expected_kx = nx if fft_solver == "fft2" else rfft_kx
    if n_kx != expected_kx:
        raise ValueError(
            f"power last axis {n_kx} does not match {fft_solver} layout for "
            f"grid width {nx}; expected {expected_kx}."
        )
    return fft_solver, expected_kx


def _layout_frequency_grid(
    ny: int,
    nx: int,
    expected_kx: int,
    dy: float,
    dx: float,
    fft_solver: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frequency angle grid plus full-spectrum-equivalent layout weights."""
    ky = np.fft.fftfreq(ny, d=dy)
    if fft_solver == "fft2":
        kx = np.fft.fftfreq(nx, d=dx)
        layout_weights = np.ones((ny, expected_kx), dtype=float)
    else:
        kx = np.fft.rfftfreq(nx, d=dx)
        col_weights = np.full(expected_kx, 2.0)
        col_weights[0] = 1.0
        if nx % 2 == 0:
            col_weights[-1] = 1.0
        layout_weights = np.broadcast_to(col_weights, (ny, expected_kx))
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    return kx_grid, ky_grid, layout_weights


def _bin_radial_spectrum(
    spectrum: np.ndarray,
    grid_shape: tuple[int, int],
    *,
    edges: np.ndarray | None,
    spacing: tuple[float, float],
    fft_solver: str = "rfft2",
    n_bins: int = 30,
    exclude_dc: bool = True,
) -> np.ndarray:
    """Average an existing power spectrum over radial frequency intervals.

    Parameters
    ----------
    spectrum : ndarray, shape (..., ny, n_kx)
        Real auto-power in native FFT order, with DC at [0, 0]. Leading axes
        are preserved. n_kx is nx for fft2 or nx // 2 + 1 for rfft2.
    grid_shape : pair of int
        Original spatial dimensions (ny, nx).
    edges : ndarray or None
        Increasing radial edges in cycles per spatial unit. Intervals are
        [left, right), with the final edge included. None generates n_bins
        equal-width intervals through the largest represented radius.
    spacing : pair of float
        Explicit spatial pitches (dy, dx), in the same units on both axes.
        Use (1, 1) for cycles per pixel; None is not accepted.
    fft_solver : {'fft2', 'rfft2'}, default 'rfft2'
        Layout of spectrum. Implicit rfft2 cells receive their full-spectrum
        multiplicity so the two layouts yield the same radial power means.
    n_bins : int, default 30
        Number of radial intervals when edges is None.
    exclude_dc : bool, default True
        Exclude the zero-frequency cell. Nyquist cells remain eligible.

    Returns
    -------
    ndarray, shape (..., n_radius)
        Mean power in each interval. Frequencies outside edges are ignored;
        intervals with no eligible Fourier cells are NaN.
    """
    reducer, counts = _radial_bin_reducer(
        grid_shape,
        n_bins=n_bins,
        fft_solver=fft_solver,
        exclude_dc=exclude_dc,
        spacing=spacing,
        edges=edges,
    )

    return _apply_radial_reducer(spectrum, grid_shape, fft_solver, reducer, counts)


def _radial_bin_reducer(
    grid_shape: tuple[int, int],
    *,
    edges: np.ndarray | None,
    spacing: tuple[float, float],
    fft_solver: str = "rfft2",
    n_bins: int = 30,
    exclude_dc: bool = True,
) -> tuple[csr_matrix, np.ndarray]:
    """Build a reusable sparse operator for radial Fourier-cell means.

    The returned operator maps a flattened Cartesian spectrum to radial bins.
    Spacing is explicit and edges are in cycles per spatial unit.
    rfft2 multiplicities match full FFT power. See _bin_radial_spectrum
    for geometry keywords and arguments.

    Returns
    -------
    reducer : scipy.sparse.csr_matrix, shape (n_radius, ny * n_kx)
        Rows average native-order Cartesian Fourier cells. Nonempty rows
        have nonnegative weights summing to one; empty rows are zero.
    cell_counts : ndarray, shape (n_radius,)
        Full-spectrum-equivalent cell counts, including rfft2 multiplicities.
    """
    idx, weights, counts, n_bins, keep = _radial_bin_index_and_counts(
        grid_shape,
        n_bins=n_bins,
        fft_solver=fft_solver,
        exclude_dc=exclude_dc,
        spacing=spacing,
        edges=edges,
    )
    reducer = csr_matrix(
        (weights / counts[idx], (idx, np.flatnonzero(keep))),
        shape=(n_bins, keep.size),
    )
    return reducer, counts


def _apply_radial_reducer(
    spectrum: np.ndarray,
    grid_shape: tuple[int, int],
    fft_solver: str,
    reducer: csr_matrix,
    counts: np.ndarray,
) -> np.ndarray:
    """Reduce a spectrum batch while preserving leading axes and empty bins."""
    ny, nx = grid_shape
    expected_kx = nx if fft_solver == "fft2" else nx // 2 + 1
    if spectrum.shape[-2:] != (ny, expected_kx):
        raise ValueError(
            f"spectrum last two dims {spectrum.shape[-2:]} do not match "
            f"expected ({ny}, {expected_kx}) for fft_solver='{fft_solver}'."
        )
    flat = spectrum.reshape(-1, reducer.shape[1])
    out = (reducer @ flat.T).T
    out[:, counts == 0] = np.nan
    return out.reshape(*spectrum.shape[:-2], reducer.shape[0])


def _radial_bin_index_and_counts(
    grid_shape: tuple[int, int],
    *,
    edges: np.ndarray | None,
    spacing: tuple[float, float],
    fft_solver: str = "rfft2",
    n_bins: int = 30,
    exclude_dc: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """Return bin indices, weights, counts, bin count, and the FFT-cell mask."""
    # Compute the radial frequency for each 2D FFT cell.
    ny, nx = grid_shape
    expected_kx = nx if fft_solver == "fft2" else nx // 2 + 1
    k = _radial_frequency_grid(grid_shape, spacing=spacing, fft_solver=fft_solver)
    k_max = float(k.max())

    # Determine the bin edges
    if edges is None:
        # Edges include 0; right edge slightly past k_max so the last bin is closed.
        # Each bin is equally spaced in frequency, but will contain different numbers of FFT cells.
        edges = np.linspace(0.0, k_max * (1.0 + 1e-9), n_bins + 1)
    else:
        edges = np.asarray(edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError("edges must be a 1D monotonically increasing array of length >= 2.")
        n_bins = len(edges) - 1

    # Bin index for each spectrum cell (0..n_bins-1).
    k_flat = k.ravel()
    # Match histogram intervals: [left, right), with the final edge included.
    keep = (k_flat >= edges[0]) & (k_flat <= edges[-1])
    if exclude_dc:
        keep &= k_flat > 0.0
    idx = np.minimum(np.digitize(k_flat[keep], edges) - 1, n_bins - 1)

    # For rfft2 the negative-kx half is implicit but corresponds to conjugate
    # entries with identical |X|^2. To make per-bin sums match what fft2 would
    # give, double-count interior columns and single-count DC + Nyquist (if even).
    if fft_solver == "rfft2":
        col_weights = np.full(expected_kx, 2.0)
        col_weights[0] = 1.0
        if nx % 2 == 0:
            col_weights[-1] = 1.0
        weights2d = np.broadcast_to(col_weights, (ny, expected_kx)).ravel()  # flattened
    else:
        weights2d = np.ones(ny * expected_kx)

    weights2d = weights2d[keep]

    # Count the number of FFT cells in each bin.
    counts = np.zeros(n_bins)
    np.add.at(counts, idx, weights2d)

    return idx, weights2d, counts, n_bins, keep


def _radial_bin_cell_counts(
    grid_shape: tuple[int, int],
    *,
    edges: np.ndarray | None,
    spacing: tuple[float, float],
    fft_solver: str = "rfft2",
    n_bins: int = 30,
    exclude_dc: bool = True,
) -> np.ndarray:
    """Count eligible Fourier cells in each radial frequency interval.

    Geometry keywords match _bin_radial_spectrum, including explicit spacing,
    edge inclusion and DC exclusion. Nyquist cells remain eligible.

    Returns
    -------
    cell_counts : ndarray, shape (n_radius,)
        Full-spectrum-equivalent counts, including implicit rfft2 cells.
        Zero means no grid support, so the corresponding radial mean is NaN.
        Counts do not depend on expression and are not independent observations.
    """
    if fft_solver not in ("fft2", "rfft2"):
        raise ValueError(f"fft_solver must be 'fft2' or 'rfft2', got {fft_solver!r}")
    ny, nx = grid_shape
    dy, dx = _validate_spacing(spacing)
    if ny > nx:
        ny, nx, dy, dx = nx, ny, dx, dy
    fy = np.fft.rfftfreq(ny, dy)
    fx2 = np.fft.rfftfreq(nx, dx) ** 2
    if edges is None:
        maximum = np.sqrt(fy[-1] ** 2 + fx2[-1])
        edges = np.linspace(0.0, maximum * (1.0 + 1e-9), n_bins + 1)
    else:
        edges = np.asarray(edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError("edges must be a 1D monotonically increasing array of length >= 2.")

    # Frequency radii increase along a quadrant row. Count both signs, with DC
    # and even-axis Nyquist frequencies counted once, for either FFT layout.
    counts = np.zeros(len(edges) - 1)
    for row, y in enumerate(fy):
        radius = np.sqrt(y**2 + fx2)
        stops = np.searchsorted(radius, edges)
        stops[-1] = np.searchsorted(radius, edges[-1], side="right")
        below = 2 * stops - (stops > 0) - ((nx % 2 == 0) & (stops == len(radius)))
        multiplicity = 1 if row == 0 or (ny % 2 == 0 and row == ny // 2) else 2
        counts += multiplicity * np.diff(below)
    if exclude_dc and edges[0] <= 0 <= edges[-1]:
        dc_bin = min(np.searchsorted(edges, 0, side="right") - 1, len(counts) - 1)
        counts[dc_bin] -= 1
    return counts


def _adapt_radial_edges(
    grid_shapes: Sequence[tuple[int, int]],
    spacings: Sequence[tuple[float, float]],
    edges: np.ndarray,
    *,
    fft_solver: str = "rfft2",
) -> np.ndarray:
    """Merge adjacent radial intervals to obtain shared Fourier-grid support.

    Parameters
    ----------
    grid_shapes : sequence of pair of int
        Original (ny, nx) spatial dimensions for each sample.
    spacings : sequence of pair of float
        Explicit (dy, dx) spatial pitches for each sample.
    edges : ndarray, shape (n_radius + 1,)
        Finite, increasing candidate radial edges in cycles per spatial unit.
    fft_solver : {'fft2', 'rfft2'}, default 'rfft2'
        FFT layout used for cell counts. DC is excluded; Nyquist is retained.

    Returns
    -------
    ndarray
        A subset of the candidate boundaries, preserving both outer edges.
        Adjacent intervals are accumulated until every sample has support;
        any unsupported tail joins the last interval. If a sample has no
        support anywhere, one interval spanning the original edges is
        returned; callers must still check support.
    """
    shapes = list(grid_shapes)
    sample_spacings = list(spacings)
    if not shapes or len(shapes) != len(sample_spacings):
        raise ValueError("grid_shapes and spacings must have the same nonzero length.")
    requested_edges = np.asarray(edges, dtype=float)
    if (
        requested_edges.ndim != 1
        or requested_edges.size < 2
        or not np.isfinite(requested_edges).all()
        or np.any(np.diff(requested_edges) <= 0)
    ):
        raise ValueError("edges must be a finite, strictly increasing 1D array.")

    # For each sample and each bin, count number of supporting cells
    counts = np.vstack(
        [
            _radial_bin_cell_counts(
                shape,
                n_bins=requested_edges.size - 1,
                fft_solver=fft_solver,
                spacing=spacing,
                edges=requested_edges,
            )
            for shape, spacing in zip(shapes, sample_spacings, strict=True)
        ]
    )
    # If any sample has zero support for the current bin, merge the bin to its right
    merged = [float(requested_edges[0])]
    running = np.zeros(counts.shape[0], dtype=float)
    for bin_idx in range(requested_edges.size - 1):
        running += counts[:, bin_idx]
        if np.all(running > 0):
            merged.append(float(requested_edges[bin_idx + 1]))
            running.fill(0.0)

    if len(merged) == 1:
        merged.append(float(requested_edges[-1]))
    elif merged[-1] != float(requested_edges[-1]):
        merged[-1] = float(requested_edges[-1])
    return np.asarray(merged)


def _stream_radial_features(
    spectrum_chunk_fn: Any,
    n_features: int,
    grid_shape: tuple[int, int],
    *,
    edges: np.ndarray,
    spacing: tuple[float, float],
    fft_solver: str,
    chunk_size: int,
    pbar: Any = None,
) -> np.ndarray:
    """Average power-spectrum chunks into a radial feature matrix.

    Parameters
    ----------
    spectrum_chunk_fn : callable
        (start, stop) -> real power array of shape (stop-start, ny, n_kx),
        in native FFT order. This function does not compute transforms.
    n_features : int
        Number of features supplied by the callback.
    grid_shape : pair of int
        Original spatial dimensions (ny, nx).
    edges : ndarray, shape (n_radius + 1,)
        Shared radial edges in cycles per spatial unit. The final edge is included.
    spacing : pair of float
        Explicit spatial pitches (dy, dx). Use (1, 1) for cycles per pixel.
    fft_solver : {'fft2', 'rfft2'}
        Layout returned by the callback; n_kx is nx or nx // 2 + 1.
    chunk_size : int
        Features per callback block. Only one block is held at a time.
    pbar : progress bar, optional
        Updated once per block.

    Returns
    -------
    ndarray, shape (n_features, n_radius)
        Radial Fourier-cell means. DC is excluded and Nyquist cells are
        retained. Unsupported intervals are NaN, not zero power.
    """
    if edges is None:
        raise ValueError("_stream_radial_features requires explicit radial bin edges.")

    reducer, counts = _radial_bin_reducer(
        grid_shape, fft_solver=fft_solver, spacing=spacing, edges=edges
    )
    out = np.empty((n_features, reducer.shape[0]), dtype=float)
    for start in range(0, n_features, chunk_size):
        stop = min(start + chunk_size, n_features)
        spec_chunk = spectrum_chunk_fn(start, stop)
        out[start:stop] = _apply_radial_reducer(spec_chunk, grid_shape, fft_solver, reducer, counts)
        del spec_chunk
        if pbar is not None:
            pbar.update(1)
    return out


def _polar_bin_center_coords(
    grid_shape: tuple[int, int],
    *,
    edges: np.ndarray,
    spacing: tuple[float, float],
    n_theta: int,
    angle_deg: float = 0.0,
    frequency_padding: int = 0,
) -> tuple[np.ndarray, int]:
    """Map physical polar-bin centers to fractional fftshifted array indices.

    Parameters
    ----------
    grid_shape : pair of int
        Original spatial dimensions (ny, nx), before Fourier-mode padding.
    edges : ndarray, shape (n_radius + 1,)
        Radial edges in cycles per spatial unit; arithmetic midpoints are sampled.
    spacing : pair of float
        Explicit spatial pitches (dy, dx).
    n_theta : int
        Angular centers at j * pi / n_theta over [0, pi).
    angle_deg : float, default 0
        Alignment angle in degrees; coordinates apply its inverse rotation.
    frequency_padding : int, default 0
        Evaluated mode border added to each FFT axis.

    Returns
    -------
    coords : ndarray, shape (2, n_radius * n_theta)
        Fractional (yy, xx) coordinates in the fftshifted full spectrum.
        Radius varies first and angle fastest. These are bin centers used
        for alignment, not the quadrature points used for sector means.
    n_radius : int
        Number of radial intervals.
    """
    if frequency_padding < 0 or int(frequency_padding) != frequency_padding:
        raise ValueError("frequency_padding must be a non-negative integer.")
    ny, nx = grid_shape
    dy, dx = _validate_spacing(spacing)
    edges = np.asarray(edges, dtype=float)
    radii = 0.5 * (edges[:-1] + edges[1:])  # (n_radius,) cycles/unit
    if np.any(radii > min(0.5 / dy, 0.5 / dx) * (1 + 1e-12)):
        raise ValueError("2D frequency-bin centres must lie within both sampling Nyquist limits.")
    # Sample the original spectrum at the inverse rotation. Applying the
    # angle here preserves physical directions on anisotropic FFT grids.
    thetas = np.linspace(0.0, np.pi, n_theta, endpoint=False) - np.deg2rad(angle_deg)
    coords = np.empty((2, len(radii), n_theta))
    coords[0] = radii[:, None] * (np.sin(thetas) * ny * dy)
    coords[1] = radii[:, None] * (np.cos(thetas) * nx * dx)
    coords[0] += ny // 2 + frequency_padding
    coords[1] += nx // 2 + frequency_padding
    return coords.reshape(2, -1), len(radii)


def _polar_quadrature_counts(grid_shape, *, edges, spacing, n_theta, oversampling=2):
    """Determine sample counts for each radial-angular frequency band.

    To compute power means for a given band, we sample the 2D spectrum at a
    set of radial-angular points with interpolation. The number of modes/supports
    per band is not uniform, and wider bands require higher sampling densities.
    This function determines the appropriate sample counts for each band
    based on the physical Fourier grid.

    Sampling density follows the finer Fourier-grid spacing,
    ``df = 1 / max(ny * dy, nx * dx)``. The target step is ``df / oversampling``
    in both radial distance and angular arc length. With the default oversampling
    of 2, steps are at most half a Fourier cell.

    Returns
    -------
    radial : ndarray of int, shape (n_bands,)
        Number of radial subdivisions within each band, at least one.
    angular : ndarray of int, shape (n_bands,)
        Number of angular subdivisions within each angular bin, at least
        three. Counts use the band's outer radius to bound the arc step.
    supported : ndarray of bool, shape (n_bands,)
        Whether each band's upper edge lies within both sampling Nyquist
        limits, allowing a small tolerance for automatically generated edges.
    """
    shape = np.asarray(grid_shape)
    pitch = np.asarray(_validate_spacing(spacing), dtype=float)
    edges = np.asarray(edges, dtype=float)
    if shape.shape != (2,) or np.any(shape < 1) or np.any(shape != shape.astype(int)):
        raise ValueError("grid_shape must contain two positive integers")
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError("spacing must contain two positive finite values")
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.isfinite(edges).all()
        or edges[0] < 0
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("Frequency edges must be finite, nonnegative and increasing")
    if not isinstance(n_theta, (int, np.integer)) or n_theta < 1:
        raise ValueError("n_theta must be a positive integer")
    if not np.isfinite(oversampling) or oversampling <= 0:
        raise ValueError("oversampling must be positive and finite")
    # Automatic edge grids include a 1e-9 relative margin at Nyquist.
    supported = edges[1:] <= 0.5 / pitch.max() * (1 + 2e-9)
    scale = float(oversampling * np.max(shape * pitch))
    radial = np.maximum(1, np.ceil(scale * np.diff(edges))).astype(np.int64)
    angular = np.maximum(3, np.ceil(scale * edges[1:] * np.pi / n_theta)).astype(np.int64)
    return radial, angular, supported


def _polar_reducer_workspace_bytes(grid_shape, *, edges, spacing, n_theta) -> tuple[int, int]:
    """Estimate bytes retained in CSR/counts and peak bytes while building them."""
    nr, nt, supported = _polar_quadrature_counts(
        grid_shape, edges=edges, spacing=spacing, n_theta=n_theta
    )
    points = nr[supported].astype(float) * nt[supported]
    edges = np.asarray(edges, dtype=float)
    lo, hi = edges[:-1][supported], edges[1:][supported]
    sy, sx = np.asarray(grid_shape, dtype=float) * spacing
    # In grid cells, each band's sectors cover half an elliptical annulus.
    area = 0.5 * np.pi * (hi - lo) * (hi + lo) * sy * sx
    # Bound total travel along both axes for the two arcs and each sector's radial sides.
    boundary = 2 * ((sy + sx) * (hi + lo) + n_theta * np.hypot(sy, sx) * (hi - lo))
    # Each bilinear neighbor's unit cell lies within 1.5 cells of its sector.
    # Sweeping a 3x3 square along its boundary covers all area added outside it.
    entries = int(
        np.ceil(np.minimum(4 * points * n_theta, area + 3 * boundary + 9 * n_theta).sum())
    )
    rows = len(nr) * n_theta
    # Allow 64-bit sparse indices on grids or operators too large for int32.
    ny, nx = map(int, grid_shape)
    index_bytes = 8 if max(entries, rows, (ny + 2) * (nx + 2)) > 2**31 - 1 else 4
    stored = (8 + index_bytes) * entries + index_bytes * (rows + 1) + 8 * rows
    # The builder holds two array headers per supported row, row pointers and CSR metadata.
    row_overhead = 320 * len(points) * n_theta + 8 * rows + 8192
    # Only one sector needs interpolation and sort buffers. The final sector is still
    # live when the compact rows are copied into the assembled CSR matrix.
    build_peak = 2 * stored + row_overhead + 256 * int(points.max(initial=0))
    return stored, build_peak


def _mode_pair_counts(reducer, grid_shape, *, periodic=True):
    """Count contributing source-frequency groups per row, excluding DC.

    This is used for cross-spectral coherency computation. Single-frequency
    complex coherency has unit magnitude for nonzero powers. Conjugates k
    and -k count only once to avoid inflating support.

    FFT boundary aliases share one group. NUFFT frequencies are not periodic:
    an unpaired negative Nyquist mode counts individually.
    """
    ny, nx = grid_shape
    counts = np.zeros(reducer.shape[0])
    for row in range(reducer.shape[0]):
        start, stop = reducer.indptr[row : row + 2]
        columns = reducer.indices[start:stop]
        iy, ix = np.divmod(columns, nx)
        opposite = (-iy % ny) * nx + (-ix % nx)
        pairs = np.minimum(columns, opposite)
        if not periodic:
            unpaired = ((ny % 2 == 0) & (iy == ny // 2)) | ((nx % 2 == 0) & (ix == nx // 2))
            pairs = np.where(unpaired, columns, pairs)
        keep = (columns != 0) & (reducer.data[start:stop] > 64 * np.finfo(float).eps)
        counts[row] = np.unique(pairs[keep]).size
    return counts


def _polar_bin_reducer(
    grid_shape,
    *,
    edges,
    spacing,
    n_theta,
    angle_deg=0.0,
    frequency_padding=0,
    oversampling=2,
):
    """Build a sparse operator for rotation-aligned radial-angular band means.

    The operator maps a flattened Cartesian spectrum to area-weighted polar
    band means. Rows are radial-angular bins; columns are source Fourier cells
    in native FFT order, with DC at [0, 0]. The same nonnegative weights apply
    to auto-power ``abs(F)**2`` and complex cross-products ``Fx * conj(Fy)``.

    Parameters
    ----------
    grid_shape : pair of int
        Original spatial grid dimensions ``(ny, nx)``, before mode padding.
    spacing : pair of float
        Explicit spatial pitches ``(dy, dx)``. Use (1, 1) for cycles per
        pixel; None is not accepted.
    edges : array-like, shape (n_radius + 1,)
        Radial frequency edges in cycles per spatial unit.
    n_theta : int
        Angular bins over a half turn, centered at ``j * pi / n_theta``.
        Each bin has angular width ``pi / n_theta``.
    angle_deg : float, default 0
        Sample alignment angle in degrees. Quadrature locations are mapped
        into the original spectrum using the inverse rotation.
    frequency_padding : int, default 0
        Evaluated Fourier modes added on each edge of the original lattice.
        The source spectrum must have shape ``(ny + 2*p, nx + 2*p)``, where
        ``p = frequency_padding``. Its frequency pitch still follows the
        original spatial domain. Unpadded FFT interpolation wraps periodically;
        padded NUFFT interpolation uses the evaluated boundary modes.
    oversampling : float, default 2
        Internal quadrature density relative to the finer Fourier-cell pitch.
        The default limits radial strip widths and outer angular arc steps
        to half a cell. This changes integration density, not output bin count.

    Returns
    -------
    reducer : scipy.sparse.csr_matrix
        Shape ``(n_radius * n_theta, (ny + 2*p) * (nx + 2*p))``.
        Row ``b * n_theta + j`` produces radial band b, angular bin j.
        Column ``iy * (nx + 2*p) + ix`` addresses source cell ``[iy, ix]``.
        Thus ``reducer @ product.ravel()`` has shape ``(n_radius * n_theta,)``
        and reshapes to ``(n_radius, n_theta)``. Supported rows have
        nonnegative weights summing to one. Unsupported radial bands have
        zero rows, which callers mask as NaN after reduction.
    counts : ndarray, shape (n_radius * n_theta,)
        Number of distinct contributing source-frequency groups up to
        conjugacy in each row, excluding DC and round-off weights. FFT
        boundary aliases share a group; unpaired NUFFT Nyquist modes count
        individually. These are support diagnostics, not quadrature-point
        counts or independent statistical observations.

    Notes
    -----
    Each band is subdivided according to the physical Fourier resolution.
    Radial strips are sampled at their area-centroid radii and angular
    subdivisions at their midpoints. Bilinear interpolation distributes each
    quadrature point's annular-area weight among four neighboring source
    cells. Contributions to the same cell are summed into one sparse row.
    This approximates an area mean under ``r dr dtheta``. Centering and
    spectral normalization occur upstream; centered DC may receive an
    interpolation weight but does not contribute power or support counts.
    """
    if not isinstance(frequency_padding, (int, np.integer)) or frequency_padding < 0:
        raise ValueError("frequency_padding must be a non-negative integer")
    if not np.isfinite(angle_deg):
        raise ValueError("angle_deg must be finite")

    # Match quadrature density to Fourier resolution and reject bands beyond Nyquist.
    radial, angular, supported = _polar_quadrature_counts(
        grid_shape, spacing=spacing, edges=edges, n_theta=n_theta, oversampling=oversampling
    )
    edges = np.asarray(edges, dtype=float)
    shape = np.asarray(grid_shape)
    pitch = np.asarray(_validate_spacing(spacing), dtype=float)
    ny, nx = (int(n + 2 * frequency_padding) for n in shape)
    modes = ny * nx
    n_bins = (len(edges) - 1) * n_theta
    data, indices, indptr = [], [], [0]
    dtheta = np.pi / n_theta

    # Build each annular-sector mean using exact strip-area fractions.
    for band, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if not supported[band]:
            # Preserve every output bin: empty CSR rows have repeated pointers.
            indptr.extend([indptr[-1]] * n_theta)
            continue
        nr, nt = int(radial[band]), int(angular[band])
        strips = np.linspace(lo, hi, nr + 1)
        left, right = strips[:-1], strips[1:]
        # Area-centroid radii integrate linear radial fields exactly under r dr.
        radii = (2 / 3) * (left**2 + left * right + right**2) / (left + right)
        # Share each strip's annular-area fraction among its nt angular samples.
        area = np.repeat((right - left) * (right + left) / ((hi - lo) * (hi + lo) * nt), nt)
        offsets = (np.arange(nt) + 0.5) / nt - 0.5

        # Inverse-rotate sector midpoints into fractional Cartesian FFT locations.
        for sector in range(n_theta):
            theta = (sector + offsets) * dtheta - np.deg2rad(angle_deg)
            y = ny // 2 + radii[:, None] * np.sin(theta) * shape[0] * pitch[0]
            x = nx // 2 + radii[:, None] * np.cos(theta) * shape[1] * pitch[1]
            iy, ix = np.floor(y).astype(np.int64), np.floor(x).astype(np.int64)
            dy, dx = y - iy, x - ix
            # Each point distributes its area fraction to its four neighboring cells.
            weights = np.concatenate(
                [
                    ((1 - dy) * (1 - dx)).ravel(),
                    (dy * (1 - dx)).ravel(),
                    ((1 - dy) * dx).ravel(),
                    (dy * dx).ravel(),
                ]
            ) * np.tile(area, 4)
            # Convert shifted locations to flattened native-order source columns.
            # Modulo wraps FFT edges; NUFFT padding supplies evaluated boundary cells.
            columns = np.concatenate(
                [
                    (((iy + oy - ny // 2) % ny) * nx + (ix + ox - nx // 2) % nx).ravel()
                    for oy, ox in ((0, 0), (1, 0), (0, 1), (1, 1))
                ]
            )
            row = csr_matrix((weights, columns, np.array([0, len(weights)])), shape=(1, modes))
            # Multiple quadrature points can contribute to the same source cell.
            row.sum_duplicates()
            row.eliminate_zeros()
            data.append(row.data.copy())
            indices.append(row.indices.copy())
            indptr.append(indptr[-1] + row.nnz)
    # Stack rows in radial-major order and count source support separately.
    reducer = csr_matrix(
        (
            np.concatenate(data) if data else np.array([], dtype=float),
            np.concatenate(indices) if indices else np.array([], dtype=np.int64),
            np.asarray(indptr),
        ),
        shape=(n_bins, modes),
    )
    return reducer, _mode_pair_counts(reducer, (ny, nx), periodic=frequency_padding == 0)


def _stream_polar_features(
    spectrum_chunk_fn: Any,
    n_features: int,
    grid_shape: tuple[int, int],
    *,
    edges: np.ndarray,
    spacing: tuple[float, float],
    fft_solver: str,
    chunk_size: int,
    n_theta: int,
    angle_deg: float = 0.0,
    frequency_padding: int = 0,
    pbar: Any = None,
) -> np.ndarray:
    """Average power-spectrum chunks over rotation-aligned polar sectors.

    Parameters
    ----------
    spectrum_chunk_fn : callable
        (start, stop) -> real power array of shape (stop-start, ny, n_kx),
        in native FFT order. Padded inputs use the dimensions described below.
        Complex cross-products must use _polar_bin_reducer directly.
    n_features : int
        Number of features supplied by the callback.
    grid_shape : pair of int
        Original spatial dimensions (ny, nx), before Fourier-mode padding.
    edges : ndarray, shape (n_radius + 1,)
        Shared radial edges in cycles per spatial unit.
    spacing : pair of float
        Explicit spatial pitches (dy, dx). Use (1, 1) for cycles per pixel.
    fft_solver : {'fft2', 'rfft2'}
        Layout returned by the callback; unpadded n_kx is nx or nx // 2 + 1.
        rfft2 input is expanded using real power symmetry.
    chunk_size : int
        Features per callback block. The reducer is reused across blocks.
    n_theta : int
        Angular sectors over [0, pi), centered at j * pi / n_theta.
    angle_deg : float, default 0
        Alignment angle in degrees. Sampling uses its inverse rotation in
        physical frequency coordinates, including on anisotropic grids.
    frequency_padding : int, default 0
        Evaluated NUFFT-mode border p on each edge. Padded input requires
        fft2 layout and shape (stop-start, ny + 2*p, nx + 2*p); interpolation
        uses those boundary modes. Unpadded FFT interpolation is periodic.
    pbar : progress bar, optional
        Updated once per block.

    Returns
    -------
    ndarray, shape (n_features, n_radius * n_theta)
        Area-weighted sector means, with angle varying fastest. Column
        b * n_theta + j is radial interval b and angular sector j. Bands
        beyond either Nyquist limit or without non-DC support are NaN.
        Inputs must already be centered and normalized; this helper only
        averages power, using bilinear quadrature under r dr dtheta.
    """
    reducer, counts = _polar_bin_reducer(
        grid_shape,
        spacing=spacing,
        edges=edges,
        n_theta=n_theta,
        angle_deg=angle_deg,
        frequency_padding=frequency_padding,
    )
    out = np.empty((n_features, reducer.shape[0]), dtype=np.float64)
    for start in range(0, n_features, chunk_size):
        stop = min(start + chunk_size, n_features)
        spec_chunk = spectrum_chunk_fn(start, stop)
        full = _as_full_power_spectrum(
            spec_chunk, grid_shape, fft_solver=fft_solver, frequency_padding=frequency_padding
        )
        for j in range(full.shape[0]):
            out[start + j] = reducer @ full[j].ravel()
        del spec_chunk, full
        if pbar is not None:
            pbar.update(1)
    out[:, counts == 0] = np.nan
    return out


def _as_full_power_spectrum(
    power: np.ndarray,
    grid_shape: tuple[int, int],
    *,
    fft_solver: str,
    frequency_padding: int = 0,
) -> np.ndarray:
    """Return full native-order auto-power, expanding an rfft2 half-spectrum.

    Uses the Hermitian symmetry of the FFT of a real signal: ``|X[ky, kx]|² ==
    |X[(ny - ky) % ny, (nx - kx) % nx]|²``. For ``fft2`` input, returns ``power``
    unchanged. This mirrors real power, not complex Fourier coefficients
    or cross-products. Padded input must already contain the full evaluated
    grid with shape (..., ny + 2*p, nx + 2*p).
    """
    if frequency_padding and (
        fft_solver != "fft2"
        or power.shape[-2:] != tuple(n + 2 * frequency_padding for n in grid_shape)
    ):
        raise ValueError(
            "Padded polar spectra require fft2 layout and the evaluated frequency border."
        )
    if fft_solver == "fft2":
        return power
    ny, nx = grid_shape
    half = power.shape[-1]
    full = np.zeros(power.shape[:-1] + (nx,), dtype=power.dtype)
    full[..., :half] = power

    # Build the (-ky)-flipped version of `power` (axis -2): keep ky=0 fixed,
    # reverse the order of ky=1..ny-1.
    flipped_ky = np.empty_like(power)
    flipped_ky[..., 0, :] = power[..., 0, :]
    if ny > 1:
        flipped_ky[..., 1:, :] = power[..., :0:-1, :]

    # Mirror interior columns. Column j (1 <= j < last_interior) lives at column
    # nx - j with the ky axis reversed. Skip DC (j=0) and Nyquist (j=nx/2 when
    # nx is even) since both are self-conjugate.
    last_interior = half - 1 if nx % 2 == 0 else half
    for j in range(1, last_interior):
        full[..., nx - j] = flipped_ky[..., j]
    return full
