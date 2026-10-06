"""
Non-uniform FFT (NUFFT) spectra, kernel and spatial tests for irregular data

When data sit on a regular grid (e.g., a rasterized Visium slide),
:func:`sonic.power_spectrum_2d` computes :math:`|\\hat{x}(k)|^2` with a plain
2D FFT. For data whose spatial coordinates are **irregular** — e.g.,
imaging-based in situ platforms, Slide-seq, or a Visium slide read straight
from ``adata.obsm['spatial']`` without rasterization — :func:`power_spectrum_2d_nufft`
evaluates the type-1 NUFFT

.. math::

   \\hat c(k_y, k_x) = \\sum_{j=1}^{n} c_j \\,
       \\exp\\!\\bigl[-i(k_y\\,y_j + k_x\\,x_j)\\bigr]

on the same uniform ``(ny, nx)`` k-space grid that :func:`power_spectrum_2d`
would produce for a rasterized input of the same physical extent, and returns
:math:`|\\hat c|^2` in the scipy FFT layout (DC at ``[0, 0]``). Anything
downstream — :func:`sonic.comparators.features.radial_bin_spectrum`,
:class:`sonic.ComparatorIrregular` — works identically.

Notation (shared across this module)
------------------------------------

Dimensions:

- ``n``: number of spots (on the irregular grid).
- ``(ny, nx)``: internal uniform k-grid dimensions; ``n' = ny · nx``.
- ``(dy, dx)``: **physical** spacing per k-grid cell, same unit as the spatial coordinates
  after multiplying ``unit_scale``.
- ``unit_scale``: multiplier that converts the input coordinates ``S`` to the same unit as
  ``(dy, dx)`` (e.g., 0.35 if ``S`` are in pixels at 0.35 μm/pixel). Samples from different
  slides and platforms may ship coordinates in different units; this parameter harmonizes them
  onto the same **physical** unit for the internal k-grid and all downstream spectra and tests.

Vectors and matrices:

- ``S``: the ``n × 2`` spatial coordinate matrix of the irregular points, ordered as ``(y, x)``.
- ``K``: the ``n × n`` translation-invariant kernel at the irregular points.
- ``K'``: the ``n' × n'`` grid kernel with FFT eigenvalues
  ``λ(k) = F(K')(k)``.
- ``U``: the ``n × n'`` type-2 NUFFT evaluation matrix; the band-limited
  approximation is ``K ≈ (1/n') · U · diag(λ) · Uᴴ``.
- ``x̂ = Uᴴ x``: type-1 NUFFT of a length-``N`` signal onto the k-grid ``(ny, nx)`` (vectorized).

"""

from __future__ import annotations

import logging

import finufft
import numpy as np
import scipy.fft
import scipy.sparse as sp
from scipy.stats import chi2, norm

from sonic.kernels.base import Kernel
from sonic.kernels.fft import FFTKernel

__all__ = [
    "power_spectrum_2d_nufft",
    "NUFFTKernel",
]

logger = logging.getLogger(__name__)


def _infer_grid_from_coords(
    coords: np.ndarray,
    unit_scale: float = 1.0,
    oversample: float = 2.0,
    padding: float = 1.05,
    min_side: int = 32,
    max_side: int = 1024,
) -> tuple[tuple[int, int], tuple[float, float]]:
    """Pick ``(grid_shape, spacing)`` from coords alone, with no kernel input.

    The k-grid only needs to resolve the signal's sampling Nyquist, which is
    set by the median nearest-neighbor spacing of the coordinates. Finer than
    that is wasted work (aliasing kicks in anyway). Coarser misses kernel
    spectral content. ``oversample=2.0`` is a safe default.

    Returns ``(grid_shape, spacing)`` rounded to FFT-friendly sizes (multiples
    of 8).
    """
    from scipy.spatial import cKDTree

    scaled = np.asarray(coords, dtype=np.float64) * float(unit_scale)
    if scaled.ndim != 2 or scaled.shape[1] != 2:
        raise ValueError(f"coords must be (n, 2), got {scaled.shape}.")
    L_y = float(scaled[:, 0].max() - scaled[:, 0].min()) * padding
    L_x = float(scaled[:, 1].max() - scaled[:, 1].min()) * padding
    if L_y <= 0 or L_x <= 0:
        raise ValueError("coords have zero extent along one or both axes.")
    # Median 1-NN distance — robust proxy for the sampling scale.
    nn = cKDTree(scaled).query(scaled, k=2)[0][:, 1]
    d_nn = float(np.median(nn[nn > 0]))
    spacing_target = d_nn / oversample

    def _round_up(n: float) -> int:
        return int(min(max_side, max(min_side, 8 * int(np.ceil(n / 8)))))

    ny = _round_up(L_y / spacing_target)
    nx = _round_up(L_x / spacing_target)
    return (ny, nx), (L_y / ny, L_x / nx)


def _resolve_k_neighbors_on_coords(
    coords: np.ndarray,
    k_neighbors: int,
    grid_shape: tuple[int, int],
    spacing: tuple[float, float],
    unit_scale: float = 1.0,
) -> int:
    r"""Map ``k_neighbors`` on irregular coords to a grid-ring ``neighbor_degree``.

    Graph-based kernels (``moran`` / ``graph_laplacian`` / ``car``) on
    :class:`MatrixKernel` use a mutual-``k``-nearest-neighbor graph on the
    *actual* coord set — the connectivity of pair ``(i, j)`` depends on
    whether ``y_i`` and ``y_j`` are among each other's ``k`` closest points
    in coordinate space. :class:`FFTKernel`, by contrast, builds a
    translation-invariant kernel on the internal ``(ny, nx)`` lattice and
    routes ``k_neighbors`` through :meth:`FFTKernel._k_neighbors_to_degree`,
    which counts *grid cells* within a ring of the origin — entirely
    divorced from the coord set's density.

    :class:`NUFFTKernel` bridges the two: it builds the FFT grid kernel
    but applies it at the irregular coords. The correct semantic for
    ``k_neighbors`` on this path is ``MatrixKernel``'s — "nearby in
    coord space". This helper enforces that semantic by picking the grid
    ring whose Euclidean radius is closest to the *median k-th nearest
    neighbor distance of the coord set*. Points within that radius
    couple through the kernel; points farther apart do not.

    Parameters
    ----------
    coords : np.ndarray
        ``(n, 2)`` coord array in the same unit as ``coords`` passed to
        :class:`NUFFTKernel.__init__`.
    k_neighbors : int
        Target mutual-k-NN count, in the coord set (not grid cells).
    grid_shape : tuple[int, int]
        Internal FFT grid ``(ny, nx)``.
    spacing : tuple[float, float]
        Internal FFT grid spacing ``(dy, dx)`` in physical units.
    unit_scale : float, default 1.0
        Multiplier so ``coords * unit_scale`` shares its unit with
        ``spacing``. Mirrors :class:`NUFFTKernel`'s ``unit_scale``.

    Returns
    -------
    int
        ``neighbor_degree`` (≥ 1) to forward to :class:`FFTKernel`.

    Raises
    ------
    ValueError
        If ``k_neighbors`` is out of ``[1, n-1]``.
    """
    from scipy.spatial import cKDTree

    k_neighbors = int(k_neighbors)
    n = int(coords.shape[0])
    if k_neighbors < 1 or k_neighbors >= n:
        raise ValueError(f"k_neighbors must be in [1, {n - 1}], got {k_neighbors}.")

    # Median k-th NN distance in the same physical units as ``spacing``.
    coords_phys = np.asarray(coords, dtype=np.float64) * float(unit_scale)
    # ``k=k_neighbors + 1``: the 0-th "neighbor" is the point itself.
    dists, _ = cKDTree(coords_phys).query(coords_phys, k=k_neighbors + 1)
    r_k = float(np.median(dists[:, k_neighbors]))
    r_k_sq = r_k * r_k

    # Unique ring radii² on the square-topology torus grid (mirrors
    # :meth:`FFTKernel._precompute_square_dists` /
    # :meth:`FFTKernel._unique_ring_distances` for the ``topology='square'``
    # case NUFFTKernel always instantiates).
    ny, nx = int(grid_shape[0]), int(grid_shape[1])
    dy, dx = float(spacing[0]), float(spacing[1])
    y = np.arange(ny) * dy
    x = np.arange(nx) * dx
    y = np.minimum(y, (ny * dy) - y)
    x = np.minimum(x, (nx * dx) - x)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    d_sq = (yy**2 + xx**2).ravel()
    flat = np.sort(d_sq)
    tol = 1e-6 * max(1.0, float(flat[-1]))
    keep = np.concatenate([[True], np.diff(flat) > tol])
    unique_dists_sq = flat[keep]  # ascending, starts at 0 (self)

    if unique_dists_sq.size <= 1:
        return 1  # degenerate (e.g. 2×2 grid); FFTKernel will clamp anyway.

    # Closest non-zero ring to the coord-set median k-NN distance.
    # ``argmin`` on a slice starting at index 1 keeps the self-ring out.
    diffs = np.abs(unique_dists_sq[1:] - r_k_sq)
    return int(np.argmin(diffs)) + 1


def power_spectrum_2d_nufft(
    coords: np.ndarray,
    values: np.ndarray,
    grid_shape: tuple[int, int],
    spacing: tuple[float, float],
    unit_scale: float = 1.0,
    eps: float = 1e-6,
    center_coords: bool = True,
) -> np.ndarray:
    """
    Compute the 2D power spectrum via type-1 NUFFT

    This function computes the power spectrum :math:`P(k) = |\\hat{c}(k)|^2` of
    one or more non-uniform spatial signals via a type-1 NUFFT.
    The output has the same ``(ny, nx)`` layout as
    :func:`sonic.kernels.fft.power_spectrum_2d` with ``fft_solver='fft2'``: DC at
    ``[0, 0]``, Nyquist at ``[ny/2, nx/2]`` (when dimensions are even).

    Parameters
    ----------
    coords : np.ndarray
        Non-uniform spatial coordinates, shape ``(n, 2)`` in the order
        ``(y, x)``. Values outside the physical domain implied by
        ``grid_shape`` and ``spacing`` are folded into ``[-π, π)`` by finufft.
    values : np.ndarray
        Signal strengths at each coordinate. Shape ``(n,)`` for a single
        feature, or ``(n, M)`` for ``M`` stacked features (e.g., genes) on the
        same coordinates. Real-valued; promoted to complex internally.
    grid_shape : tuple[int, int]
        ``(ny, nx)`` of the target uniform k-space grid. Match whatever grid
        you use for rasterized samples so the two paths produce comparable
        spectra.
    spacing : tuple[float, float]
        ``(dy, dx)`` physical spacing per cell of the target grid, in the same
        unit as ``unit_scale * coords``. Together with ``grid_shape`` this
        defines the physical domain extent ``(ny · dy, nx · dx)``.
    unit_scale : float, default 1.0
        Multiplier applied to ``coords`` before scaling into ``[-π, π)``. Use
        this to convert per-sample coordinate units into the common unit of
        ``spacing`` (e.g., 0.35 if ``coords`` are in Visium full-res pixels at
        0.35 μm/pixel and ``spacing`` is in μm).
    eps : float, default 1e-6
        NUFFT tolerance forwarded to finufft.
    center_coords : bool, default True
        If True, subtract the mean of ``coords`` before scaling — avoids
        wrapping artefacts when coordinates are stored with an arbitrary origin
        offset (e.g., Visium pixel coordinates start at a few thousand). Power
        spectra are translation-invariant so recentering does not change the
        result.

    Returns
    -------
    np.ndarray
        Power spectrum. Shape ``(ny, nx)`` for 1D ``values`` or
        ``(ny, nx, M)`` for 2D ``values``, with DC at index ``[0, 0]``.

    Raises
    ------
    ImportError
        If :mod:`finufft` is not installed.
    ValueError
        If input shapes are inconsistent.

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.random.default_rng(0).uniform(0, 100, size=(500, 2))
    >>> vals = np.random.default_rng(1).standard_normal(500)
    >>> P = power_spectrum_2d_nufft(coords, vals, grid_shape=(32, 32), spacing=(4.0, 4.0))
    >>> P.shape
    (32, 32)
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (n, 2), got {coords.shape}.")
    if values.shape[0] != coords.shape[0]:
        raise ValueError(
            f"values first dim {values.shape[0]} must match coords n={coords.shape[0]}."
        )

    ny, nx = grid_shape
    dy, dx = spacing
    if ny <= 0 or nx <= 0 or dy <= 0 or dx <= 0:
        raise ValueError(f"grid_shape and spacing must be positive, got {grid_shape}, {spacing}.")

    y = coords[:, 0].astype(np.float64) * unit_scale
    x = coords[:, 1].astype(np.float64) * unit_scale
    if center_coords:
        y = y - y.mean()
        x = x - x.mean()

    # Physical domain extents implied by the target uniform grid.
    Ly = ny * dy
    Lx = nx * dx

    # Scale into finufft's [-π, π) window so that mode index k (centred
    # at zero, range [-n/2, (n-1)/2]) corresponds to physical frequency
    # k / L cycles per unit length — matching np.fft.fftfreq(n, d).
    y_scaled = y * (2.0 * np.pi / Ly)
    x_scaled = x * (2.0 * np.pi / Lx)

    # Batched transforms: finufft accepts shape (n_tr, M) for c.
    squeeze = values.ndim == 1
    if squeeze:
        c = values.astype(np.complex128, copy=False)
    else:
        # finufft expects (n_tr, N_points).
        c = np.ascontiguousarray(values.T.astype(np.complex128, copy=False))

    # type-1 NUFFT: nonuniform points -> uniform k-space grid.
    # Output shape: (ny, nx) or (n_tr, ny, nx). DC at CENTRE ([ny//2, nx//2]).
    f_hat = finufft.nufft2d1(y_scaled, x_scaled, c, n_modes=(ny, nx), eps=eps, isign=-1)

    # Power spectrum.
    power = (f_hat.real**2 + f_hat.imag**2).astype(np.float64)

    # Move DC from the centre to [0, 0] so the layout matches scipy.fft.fft2.
    power = np.fft.ifftshift(power, axes=(-2, -1))

    if squeeze:
        return power
    # Put the feature axis back at the end to match power_spectrum_2d(x=(ny, nx, M)).
    return np.moveaxis(power, 0, -1)


# ---------------------------------------------------------------------------
# NUFFTKernel: translation-invariant kernel on irregular spatial points
# ---------------------------------------------------------------------------


class NUFFTKernel(Kernel):
    """
    Spatial kernel over **irregular** 2D coordinates evaluated via NUFFTs.

    Parallels :class:`sonic.kernels.fft.FFTKernel` (which requires a regular grid) and
    implements the :class:`~sonic.kernels.Kernel` interface so it plugs into
    :func:`sonic.statistics.spatial_q_test` /
    :func:`sonic.statistics.spatial_r_test` the same way.

    The band-limited approximation of the ``n × n`` irregular-point operator is
    ``K ≈ (1/n') · U · diag(λ) · Uᴴ``, where ``U`` is the ``n × n'`` type-2
    NUFFT matrix and ``λ = F(K')`` is the grid kernel's spectrum. Under this
    approximation, Parseval's identity gives the fast quadratic form
    ``xᵀ K x = (1/n') Σ_k λ(k) |x̂(k)|²`` with ``x̂ = Uᴴ x`` (a single type-1 NUFFT).
    The matrix-vector primitive :meth:`Kx` uses the companion two-shot NUFFT
    ``K z = (1/n') · U · (λ ⊙ Uᴴ z)`` and backs the Hutchinson-based
    cumulant estimator (:func:`sonic.statistics._hutchinson_cumulants`)
    and the bipartite R-test cross matrix.

    :func:`sonic.spatial_q_test` always uses the k-space Parseval path
    (:meth:`xtKx`); :func:`sonic.spatial_r_test` dispatches on shape —
    paired diagonal for ``M_x == M_y`` via :meth:`xtKy`, full ``(M_x, M_y)``
    cross matrix via :meth:`Kx` otherwise. The matmul counterparts
    (:meth:`xtKx_matmul` / :meth:`xtKy_matmul`) are exposed for callers that
    want to compute the same form directly at the ``n`` irregular points —
    they agree with the spectral path to NUFFT precision (``eps``) on both
    regular and irregular coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Spot coordinates of shape ``(n, 2)`` in order ``(y, x)``.
    grid_shape : tuple[int, int], optional
        ``(ny, nx)`` of the internal uniform k-grid. If ``None`` (default),
        auto-inferred from ``coords``: the grid is sized to cover the bounding
        box and to resolve the sampling Nyquist set by the median
        nearest-neighbor distance (fully coordinate-driven, kernel-agnostic).
        Override only when you know you need a finer or coarser grid.
    spacing : tuple[float, float], optional
        ``(dy, dx)`` physical spacing per k-grid cell (same unit as ``coords``
        after ``unit_scale``). If ``None`` (default), auto-inferred alongside
        ``grid_shape``. When both are supplied, users are responsible for
        ensuring ``ny · dy``, ``nx · dx`` covers the coordinate extent.
    method : str, default ``'matern'``
        Kernel method forwarded to :class:`FFTKernel`. One of ``'gaussian'``,
        ``'matern'``, ``'moran'``, ``'graph_laplacian'``, ``'car'``.
    unit_scale : float, default 1.0
        Multiplier applied to ``coords`` so they share the same unit as
        ``spacing`` (e.g. ``0.35`` if coords are in pixels at 0.35 μm/pixel).
    oversample : float, default 2.0
        Auto-grid oversampling factor above the sampling Nyquist. Used only
        when ``grid_shape`` / ``spacing`` are auto-derived. Larger values give
        a finer k-grid (more accurate, slower); 2.0 is safe for all tested
        kernels.
    eps : float, default 1e-6
        NUFFT tolerance forwarded to finufft.
    workers : int, optional
        Forwarded to :mod:`scipy.fft` (used by :meth:`Kx_grid`) and reserved
        for future finufft parallelism. ``None`` uses the SciPy default.
    **kwargs
        Method-specific kernel hyperparameters forwarded to the internal
        :class:`FFTKernel`. ``bandwidth`` / ``nu`` for ``gaussian`` /
        ``matern``; ``rho`` for ``car``; plus a *coord-aware*
        ``k_neighbors`` for the graph methods (``moran`` /
        ``graph_laplacian`` / ``car``). Note the
        ``k_neighbors``-to-``neighbor_degree`` mapping differs from that
        of :class:`FFTKernel` due to an oversampled internal grid.
        ``neighbor_degree`` is chosen so the internal-grid-ring cutoff matches
        the median k-th nearest-neighbor distance among the coords — matching
        :class:`~sonic.kernels.MatrixKernel`'s mutual-k-NN graph
        semantic up to the band-limit of the internal grid. See
        :func:`_resolve_k_neighbors_on_coords` for the mapping detail.
        Pass ``neighbor_degree`` directly to bypass this and use the
        raw grid-ring semantic as in :class:`FFTKernel`.

    Attributes
    ----------
    coords : np.ndarray
        Original ``(n, 2)`` coordinates.
    n : int
        Number of spots ``n``.
    grid_shape : tuple[int, int]
        Internal k-grid shape ``(ny, nx)``.
    spacing : tuple[float, float]
        Physical spacing per k-grid cell ``(dy, dx)``.
    method : str
        Kernel method name.
    params : dict
        Resolved kernel hyperparameters (snapshot of the internal FFT kernel).
    workers : int or None
        scipy.fft worker count used by :meth:`Kx_grid`.
    stores_precision : bool
        Always ``False`` — NUFFTKernel never holds an ``n × n`` matrix.
    """

    _available_kernels = ["gaussian", "matern", "moran", "graph_laplacian", "car"]

    def __init__(  # noqa: C901
        self,
        coords: np.ndarray,
        grid_shape: tuple[int, int] | None = None,
        spacing: tuple[float, float] | None = None,
        method: str = "matern",
        unit_scale: float = 1.0,
        oversample: float = 2.0,
        eps: float = 1e-6,
        workers: int | None = None,
        *,
        centering: bool = True,
        **kwargs,
    ) -> None:
        """Construct a translation-invariant kernel over irregular 2D coordinates.

        See the class docstring for a full parameter / attribute reference;
        in brief:

        Parameters
        ----------
        coords : np.ndarray
            ``(n, 2)`` spot coordinates in ``(y, x)`` order.
        grid_shape, spacing : tuple or None
            Internal k-grid shape and per-cell spacing. Both optional;
            auto-inferred from ``coords`` when either is missing.
        method : str, default ``'matern'``
            Kernel method forwarded to the internal :class:`FFTKernel`.
        unit_scale : float, default 1.0
            Coord → physical-unit multiplier so ``coords * unit_scale`` is in
            the same unit as ``spacing``.
        oversample : float, default 2.0
            Auto-grid oversampling above the sampling Nyquist.
        eps : float, default 1e-6
            NUFFT tolerance.
        workers : int, optional
            scipy.fft worker count used by :meth:`Kx_grid`.
        **kwargs
            Method-specific kernel hyperparameters. ``bandwidth`` / ``nu``
            for ``gaussian`` / ``matern``; ``rho`` for ``car``; and a
            coord-aware ``k_neighbors`` (resolved to a grid-ring cutoff
            matching the median k-NN distance of the coords, see the
            class docstring) for ``moran`` / ``graph_laplacian`` /
            ``car``. Pass ``neighbor_degree`` directly to override and
            use the internal grid's ring semantic.

        Raises
        ------
        ValueError
            If ``coords`` has the wrong shape, ``method`` is unknown, or
            ``grid_shape`` / ``spacing`` are invalid.
        """
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be shape (n, 2), got {coords.shape}.")
        if method not in self._available_kernels:
            raise ValueError(f"method must be one of {self._available_kernels}, got '{method}'.")

        # Auto-derive grid_shape / spacing from coords alone when either is missing.
        # Coordinate-driven: picks a k-grid that resolves the sampling Nyquist.
        if grid_shape is None or spacing is None:
            auto_gs, auto_sp = _infer_grid_from_coords(
                coords, unit_scale=unit_scale, oversample=oversample
            )
            if grid_shape is None:
                grid_shape = auto_gs
            if spacing is None:
                spacing = auto_sp
            logger.info(
                "NUFFTKernel auto-inferred grid_shape=%s spacing=%s from %d coords.",
                grid_shape,
                spacing,
                coords.shape[0],
            )

        ny, nx = int(grid_shape[0]), int(grid_shape[1])
        if ny < 4 or nx < 4:
            raise ValueError(f"grid_shape must be at least (4, 4), got ({ny}, {nx}).")
        dy, dx = float(spacing[0]), float(spacing[1])
        if dy <= 0 or dx <= 0:
            raise ValueError(f"spacing must be positive, got ({dy}, {dx}).")

        super().__init__(centering=centering)
        self.coords: np.ndarray = coords
        self.n: int = coords.shape[0]
        self.grid_shape: tuple[int, int] = (ny, nx)
        self.spacing: tuple[float, float] = (dy, dx)
        self.method: str = method
        self._unit_scale: float = float(unit_scale)
        self._eps: float = float(eps)
        self.workers: int | None = workers
        self.stores_precision: bool = False

        # Graph-kernel ``k_neighbors`` semantic: on irregular coords the
        # user intends the MatrixKernel-style mutual-k-NN graph on the
        # actual coord set, not the grid-cell-counting ring that
        # FFTKernel's ``_k_neighbors_to_degree`` would produce.
        # Resolve it here using the coord density so the grid-ring
        # cutoff matches the median k-th NN distance among the coords.
        if method in ("moran", "graph_laplacian", "car") and "k_neighbors" in kwargs:
            if "neighbor_degree" in kwargs:
                raise ValueError(
                    "Pass either 'k_neighbors' (coord-k-NN semantic) or "
                    "'neighbor_degree' (grid-ring semantic), not both."
                )
            k_nn = kwargs.pop("k_neighbors")
            kwargs["neighbor_degree"] = _resolve_k_neighbors_on_coords(
                coords,
                k_neighbors=int(k_nn),
                grid_shape=(ny, nx),
                spacing=(dy, dx),
                unit_scale=unit_scale,
            )
            # Stash the original request so callers / tests can inspect it.
            self._k_neighbors_requested: int | None = int(k_nn)
        else:
            self._k_neighbors_requested = None

        # Internal FFTKernel holds the eigenvalue spectrum on the k-grid. We
        # use fft2 (full spectrum) so the ifftshift trick aligns NUFFT output
        # with the scipy FFT layout (DC at [0, 0]).
        # Internal FFTKernel is always kept on the *raw* spectrum — we own
        # centering at the NUFFT level (the constant mode of K on scattered
        # coords isn't the FFT grid's DC mode in general).
        self._fft_kernel = FFTKernel(
            shape=(ny, nx),
            spacing=(dy, dx),
            topology="square",
            method=method,
            fft_solver="fft2",
            workers=workers,
            centering=False,
            **kwargs,
        )
        self.params: dict = dict(self._fft_kernel.params)
        if self._k_neighbors_requested is not None:
            self.params["k_neighbors"] = self._k_neighbors_requested

        # Pre-scale coords into finufft's [-π, π) window. Centered so we avoid
        # origin-offset phase artefacts.
        y = coords[:, 0] * self._unit_scale
        x = coords[:, 1] * self._unit_scale
        self._y_mean = float(y.mean())
        self._x_mean = float(x.mean())
        self._y_scaled = (y - self._y_mean) * (2.0 * np.pi / (ny * dy))
        self._x_scaled = (x - self._x_mean) * (2.0 * np.pi / (nx * dx))

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<NUFFTKernel method={self.method} n={self.n} "
            f"grid={self.grid_shape} spacing={self.spacing} "
            f"params={self.params}>"
        )

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"NUFFTKernel\n"
            f"- Method: {self.method}\n"
            f"- Number of spots: {self.n}\n"
            f"- k-grid: {self.grid_shape} at spacing {self.spacing}\n"
            f"- Params: {self.params}"
        )

    # ------------------------------------------------------------------
    _TOEPLITZ_R_THRESHOLD: int = 2000
    _TOEPLITZ_LAM_TOL: float = 1e-5

    def _real_spectrum(self) -> np.ndarray:
        """Centered Fourier weights for the real operator returned by Kx.

        Even grids contain -Nyquist but not +Nyquist. Taking the real part
        of Kx splits each weight between a mode and its true negative;
        these are distinct at irregular coordinates.
        """
        ny, nx = self.grid_shape
        lam = np.zeros((2 * (ny // 2) + 1, 2 * (nx // 2) + 1))
        lam[:ny, :nx] = np.fft.fftshift(self._fft_kernel.spectrum.reshape(ny, nx))
        return 0.5 * (lam + lam[::-1, ::-1])

    def _coord_phi(self) -> np.ndarray:
        """Coord-density ``φ(k)`` on a doubled mode range, DC at [0, 0].

        ``φ(Δ) = (1/n) · Σ_i exp(-iΔ·y_i)`` — one type-1 NUFFT of the
        all-ones vector. The range includes every signed mode difference
        of ``_real_spectrum``. Off-grid φ is not periodic at the original
        grid size, so differences must never wrap at that period.
        """
        cache = getattr(self, "_phi_cache", None)
        if cache is not None:
            return cache
        ny, nx = self.grid_shape
        phi_hat_centered = finufft.nufft2d1(
            self._y_scaled,
            self._x_scaled,
            np.ones(self.n, dtype=complex),
            n_modes=(2 * ny + 1, 2 * nx + 1),
            eps=self._eps,
            isign=-1,
        )
        # Odd mode counts pair every frequency with its actual negative.
        phi_hat_centered = 0.5 * (phi_hat_centered + phi_hat_centered[::-1, ::-1].conj())
        phi = np.fft.ifftshift(phi_hat_centered) / self.n
        self._phi_cache = phi
        return phi

    def _eigvals_toeplitz_M(self, lam_tol: float | None = None) -> np.ndarray | None:
        """Full-spectrum eigenvalues via the reduced Toeplitz-``G`` matrix.

        Uses the identity (for PSD ``Λ``) that non-zero eigvals of
        ``K_n = (1/n') U Λ Uᴴ`` equal non-zero eigvals of
        ``M = (1/n') Λ^{1/2} G Λ^{1/2}`` with ``G_{k,k'} = n·φ(k'-k)``
        (Toeplitz in k-space, built once from ``_coord_phi``). Truncate
        ``Λ`` to its support — pick the ``r`` modes with
        ``|λ(k)| > lam_tol · max|λ|`` — form the dense ``r × r`` ``M_r``
        by indexed ``φ`` lookups, ``eigvalsh``.

        For ``centering=True`` we apply the rank-1 shift
        ``G → G_H = G − (1/n) · (Uᴴ𝟏)(Uᴴ𝟏)ᴴ = G − n · conj(φ) φᵀ``,
        which accounts for the y-space ``H = I − 𝟏𝟏ᵀ/n`` projection
        ((HU)ᴴ(HU) = UᴴHU = G − (1/n)(Uᴴ𝟏)(Uᴴ𝟏)ᴴ).

        Returns ``None`` when the formula doesn't apply cleanly:

        - Indefinite ``Λ`` (``moran``, anti-phase eigenmodes) — the
          signed square root makes ``M`` non-Hermitian.
        - Broad spectrum with ``r > _TOEPLITZ_R_THRESHOLD`` — dense
          ``r × r`` eigvalsh becomes the bottleneck; callers should
          route through cumulant-based Liu
          (:func:`compute_null_params(..., liu_n_probes=...)`) instead
          of asking for the full spectrum.

        Returns
        -------
        np.ndarray or None
            Eigenvalues (descending) or ``None`` to signal "no full
            spectrum available for this configuration".
        """
        lam = self._real_spectrum()
        lam_abs_max = float(np.abs(lam).max())
        if lam.min() < -np.finfo(float).eps * lam_abs_max:
            return None  # The PSD square-root factorization cannot drop negative modes.

        lam_flat = lam.ravel()
        lam_max = float(np.abs(lam_flat).max())
        if lam_max == 0:
            return np.zeros(self.n)

        tol_rel = self._TOEPLITZ_LAM_TOL if lam_tol is None else float(lam_tol)
        mask_flat = np.abs(lam_flat) > tol_rel * lam_max
        r = int(mask_flat.sum())
        if r > self._TOEPLITZ_R_THRESHOLD:
            return None  # too broad for dense eigvalsh on r × r reduced M

        ny, nx = self.grid_shape
        nprime = ny * nx
        iy, ix = np.where(mask_flat.reshape(lam.shape))
        iy = iy - lam.shape[0] // 2
        ix = ix - lam.shape[1] // 2
        lam_r_sqrt = np.sqrt(np.maximum(lam_flat[mask_flat], 0.0))

        phi = self._coord_phi()
        # Negative indices select the corresponding signed modes in phi's
        # DC-at-origin layout; the doubled range contains every difference.
        dy = iy[None, :] - iy[:, None]
        dx = ix[None, :] - ix[:, None]
        G_r = self.n * phi[dy, dx]
        if self.centering:
            # (HU)ᴴ(HU) = G − (1/n)(Uᴴ𝟏)(Uᴴ𝟏)ᴴ = G − n · conj(φ_r) φ_rᵀ.
            # Rank-1 outer product on the restricted mode set.
            phi_r = phi[iy, ix]
            G_r = G_r - self.n * np.outer(np.conj(phi_r), phi_r)
        M_r = (lam_r_sqrt[:, None] * G_r * lam_r_sqrt[None, :]) / nprime
        M_r = 0.5 * (M_r + M_r.conj().T)

        vals = np.linalg.eigvalsh(M_r)
        vals = np.real(vals)
        # For PSD λ the operator ``K_n = (1/n') U Λ Uᴴ`` and ``HKH`` are
        # both PSD — any negative eigenvalue from ``eigvalsh`` is pure
        # finite-precision noise on the near-null subspace. Clip to 0.
        vals = np.maximum(vals, 0.0)
        return np.sort(vals)[::-1]

    def eigenvalues(self, k: int | None = None, return_full_layout: bool = False) -> np.ndarray:
        """Eigenvalues of the ``n × n`` irregular-point operator.

        Returns the spectrum of the realized operator ``K_n`` (raw) or
        ``H K_n H`` (centered) at the irregular coordinates — **not** the
        internal FFT-grid spectrum. Purely matrix-free: no dense
        ``n × n`` construction.

        Two paths:

        - **``k`` given — Lanczos top-k.** Wrap :meth:`Kx` as a
          :class:`~scipy.sparse.linalg.LinearOperator` and call
          :func:`~scipy.sparse.linalg.eigsh(which='LM')` for the top
          ``k`` eigenvalues by magnitude. Cost: ``O(k × nufft)``.
          Not cached.
        - **``k=None`` — Toeplitz-M (analytic).** See
          :meth:`_eigvals_toeplitz_M`. Applies when ``Λ`` is PSD and
          the support size
          ``r = #{k : |λ(k)| > 10⁻⁵ · max|λ|}`` is below
          ``_TOEPLITZ_R_THRESHOLD`` (default 2000). Accuracy is limited by
          spectral truncation and NUFFT ``eps``; cost is ``O(r³)`` on a dense
          ``r × r`` reduced matrix. Cached per ``centering`` mode.

        Full-spectrum requests that fall outside Toeplitz-M's reach
        (indefinite ``Λ`` like Moran, or broad-support kernels like CAR
        at strong coupling) raise ``NotImplementedError`` rather than
        falling back to an approximate density reconstruction. For Liu's
        method, use :func:`compute_null_params(..., method='liu', liu_n_probes=...)`
        to get cumulant-based Liu directly from :math:`2m` matvecs.

        Parameters
        ----------
        k : int, optional
            Top-``k`` eigenvalues by magnitude (Lanczos). ``None``
            returns the full spectrum via Toeplitz-M.
        return_full_layout : bool, default False
            Kept for API compatibility with :meth:`FFTKernel.eigenvalues`;
            ignored.

        Returns
        -------
        np.ndarray
            Eigenvalues in descending order. Length ``k`` when ``k`` is
            given, ``n`` for the full-spectrum path.
        """
        del return_full_layout  # kept only for API compatibility
        cache_key = "_spectrum_centered" if self.centering else "_spectrum_raw"
        n = self.n

        # Top-k: Lanczos. Not cached.
        if k is not None:
            from scipy.sparse.linalg import LinearOperator, eigsh

            def _matvec(v: np.ndarray) -> np.ndarray:
                return np.asarray(self.Kx(np.asarray(v, dtype=float)))

            op = LinearOperator(shape=(n, n), matvec=_matvec, dtype=float)
            k_req = min(int(k), n - 2)
            vals, _ = eigsh(op, k=k_req, which="LM")
            return np.sort(np.real(vals))[::-1]

        # Full spectrum via Toeplitz-M (cached).
        cached = getattr(self, cache_key, None)
        if cached is not None and len(cached) == n:
            return cached

        toep = self._eigvals_toeplitz_M()
        if toep is None:
            raise NotImplementedError(
                "Full NUFFT spectrum unavailable for this configuration "
                "(indefinite spectrum or broad spectral support: r > "
                f"{self._TOEPLITZ_R_THRESHOLD}). Use "
                "`compute_null_params(..., method='liu', liu_n_probes=60)` "
                "for Liu's approximation via Hutchinson-estimated "
                "cumulants, or `eigenvalues(k=...)` for a Lanczos "
                "top-k."
            )
        pad = max(0, n - len(toep))
        spectrum = np.concatenate([toep, np.zeros(pad)]) if pad else toep[:n]
        setattr(self, cache_key, spectrum)
        return spectrum

    # ------------------------------------------------------------------
    # Path A primitives — k-space Parseval (default for xtKx / xtKy)
    # ------------------------------------------------------------------
    def _nufft_type1(self, x: np.ndarray) -> np.ndarray:
        """Type-1 NUFFT of ``x`` onto the k-grid at the cached scaled coords.

        Computes ``x̂(k) = Σ_j x_j exp(-i k · r̃_j)`` where ``r̃_j`` are the
        mean-centered, ``2π/(n·d)``-scaled versions of the input coordinates.
        Shared primitive for :meth:`xtKx` (takes ``|·|²``), :meth:`xtKy`
        (complex inner product), :meth:`Kx`, and :meth:`Kx_grid`.

        Parameters
        ----------
        x : np.ndarray
            ``(n,)`` or ``(n, M)``.

        Returns
        -------
        np.ndarray
            Complex ``(M, ny, nx)`` (always 3-D; ``M=1`` for 1-D input). DC
            at the array centre (finufft convention); callers that want the
            scipy-FFT layout must ``ifftshift`` along the last two axes.
        """
        if x.shape[0] != self.n:
            raise ValueError(f"x first dim {x.shape[0]} does not match n={self.n}.")
        ny, nx = self.grid_shape
        x_in = x[:, None] if x.ndim == 1 else x
        c = np.ascontiguousarray(x_in.T.astype(np.complex128))  # (M, N)
        return finufft.nufft2d1(
            self._y_scaled,
            self._x_scaled,
            c,
            n_modes=(ny, nx),
            eps=self._eps,
            isign=-1,
        )  # (M, ny, nx)

    def xtKx(self, x: np.ndarray) -> float | np.ndarray:
        """Quadratic form ``xᵀ K x`` via **k-space Parseval**.

        Implements the default path:

        .. math::

           x^T K x \\;=\\; \\frac{1}{n'} \\sum_k \\lambda(k) \\, |\\hat x(k)|^{2}

        using one type-1 NUFFT of ``x`` and an elementwise Parseval sum.
        Only the real power spectrum ``|x̂|²`` of shape ``(ny, nx)`` is
        materialized — no ``ifft2``, no spatial-grid copy of ``x``.

        Parameters
        ----------
        x : np.ndarray
            ``(n,)`` for one feature or ``(n, M)`` for ``M`` features.

        Returns
        -------
        float or np.ndarray
            Scalar if ``x`` is 1-D, shape ``(M,)`` otherwise.

        See Also
        --------
        xtKx_matmul : Compute ``xᵀ · Kx`` via the length-``n`` matrix product.
        """
        ny, nx = self.grid_shape
        # HKH quadratic form = raw K on H x. Center per-feature (column mean).
        if self.centering:
            if x.ndim == 1:
                x = x - x.mean()
            else:
                x = x - x.mean(axis=0, keepdims=True)
        x_hat_centered = self._nufft_type1(x)  # (M, ny, nx)
        power = x_hat_centered.real**2 + x_hat_centered.imag**2  # (M, ny, nx)
        # Spectrum is stored in scipy FFT layout (DC at [0,0]); fftshift → centered
        # to match the NUFFT output before multiplying.
        lam = np.fft.fftshift(self._fft_kernel.spectrum.reshape(ny, nx))
        Q = np.sum(lam[None, :, :] * power, axis=(1, 2)) / (ny * nx)
        if x.ndim == 1:
            return float(Q[0])
        return Q.astype(np.float64)

    def xtKy(self, x: np.ndarray, y: np.ndarray) -> float | np.ndarray:
        """Bilinear form ``xᵀ K y`` via **cross Parseval**.

        Implements the default path:

        .. math::

           x^T K y \\;=\\; \\frac{1}{n'} \\sum_k \\lambda(k) \\,
               \\overline{\\hat x(k)}\\, \\hat y(k).

        Paired same-``M`` convention — returns the diagonal of ``Xᵀ K Y``
        (shape ``(M,)``) for batched inputs, scalar for 1-D inputs. For the
        bipartite ``(M_x, M_y)`` cross matrix use :meth:`xtKy_matmul` (or
        build it explicitly via ``X.T @ self.Kx(Y)``).

        Parameters
        ----------
        x, y : np.ndarray
            ``(n,)`` or ``(n, M)`` — must share shape.

        Returns
        -------
        float or np.ndarray
            Scalar for 1-D inputs; ``(M,)`` for batched.
        """
        if x.shape[0] != self.n or y.shape[0] != self.n:
            raise ValueError(
                f"x, y first dim must equal n={self.n}; got {x.shape[0]}, {y.shape[0]}."
            )
        if x.shape != y.shape:
            raise ValueError(f"x and y must share shape; got {x.shape} vs {y.shape}.")
        ny, nx = self.grid_shape
        if self.centering:
            # x^T HKH y = (Hx)^T K (Hy) — subtract per-feature means on both sides.
            if x.ndim == 1:
                x = x - x.mean()
                y = y - y.mean()
            else:
                x = x - x.mean(axis=0, keepdims=True)
                y = y - y.mean(axis=0, keepdims=True)
        x_hat = self._nufft_type1(x)  # (M, ny, nx) complex
        y_hat = self._nufft_type1(y)
        lam = np.fft.fftshift(self._fft_kernel.spectrum.reshape(ny, nx))
        cross = np.real(np.conj(x_hat) * y_hat) * lam[None, :, :]
        R = np.sum(cross, axis=(1, 2)) / (ny * nx)
        if x.ndim == 1:
            return float(R[0])
        return R.astype(np.float64)

    # ------------------------------------------------------------------
    # Path B primitives — n-point vector via NUFFT round-trip
    # ------------------------------------------------------------------
    def Kx(self, z: np.ndarray) -> np.ndarray:
        """Matrix–vector product ``K z`` at the ``n`` irregular coordinates.

        Implements the band-limited apply

        .. math::

           K z \\;\\approx\\; \\tfrac{1}{n'} \\, U \\bigl(\\lambda \\odot U^{\\mathsf H} z\\bigr),

        evaluated as type-1 NUFFT → elementwise multiply by ``λ(k) / n'`` →
        type-2 NUFFT. Output length ``n``, same shape as ``z``. Base primitive
        for :meth:`xtKx_matmul`, :meth:`xtKy_matmul`, the Hutchinson
        cumulant estimator used by Liu's null approximation, and the
        bipartite R-test in :class:`DetectorIrregular`.

        Parameters
        ----------
        z : np.ndarray
            ``(n,)`` or ``(n, M)``.

        Returns
        -------
        np.ndarray
            Same shape as ``z``.
        """
        if z.shape[0] != self.n:
            raise ValueError(f"z first dim {z.shape[0]} does not match n={self.n}.")
        ny, nx = self.grid_shape
        lam_centred = np.fft.fftshift(self._fft_kernel.spectrum.reshape(ny, nx))
        squeeze = z.ndim == 1
        # HKH z = H · K · H z: center input pre-NUFFT, center output post.
        if self.centering:
            if squeeze:
                z = z - z.mean()
            else:
                z = z - z.mean(axis=0, keepdims=True)
        z_hat = self._nufft_type1(z)  # (M, ny, nx) complex, DC centred
        out_k = np.ascontiguousarray(
            (lam_centred[None, :, :] * z_hat / (ny * nx)).astype(np.complex128)
        )
        Kz = finufft.nufft2d2(
            self._y_scaled,
            self._x_scaled,
            out_k,
            eps=self._eps,
            isign=+1,
        )  # (M, n)
        Kz = np.real(Kz).T  # (n, M)
        if self.centering:
            if squeeze:
                Kz = Kz - Kz.mean(axis=0, keepdims=True)
            else:
                Kz = Kz - Kz.mean(axis=0, keepdims=True)
        return Kz[:, 0] if squeeze else Kz

    def xtKx_matmul(self, x: np.ndarray) -> float | np.ndarray:
        """Quadratic form ``xᵀ K x`` via **direct matmul**.

        Computes ``Q_B = xᵀ · self.Kx(x)`` end-to-end at the ``n`` irregular
        points. Sparse-aware on ``x`` (``x.multiply(Kx).sum``). ~2× the NUFFT
        work of :meth:`xtKx` per feature; agrees with it to NUFFT precision
        on regular grids and to the torus-BC band (~1–2 %) on irregular ones.

        Parameters
        ----------
        x : np.ndarray or scipy.sparse matrix
            ``(n,)`` or ``(n, M)``.

        Returns
        -------
        float or np.ndarray
            Scalar for 1-D input; ``(M,)`` for batched.
        """
        if sp.issparse(x):
            if x.ndim == 1 or (x.shape[1] == 1 and x.shape[0] == self.n):
                x_sp = x.reshape(-1, 1)
                squeeze = True
            else:
                x_sp = x
                squeeze = False
            Kx_dense = self.Kx(x_sp.toarray())
            result = np.asarray(x_sp.multiply(Kx_dense).sum(axis=0)).ravel()
            return float(result[0]) if squeeze else result
        arr = np.asarray(x, dtype=float)
        Kx_dense = self.Kx(arr)
        if arr.ndim == 1:
            return float(np.dot(arr, Kx_dense))
        return np.sum(arr * Kx_dense, axis=0).astype(np.float64)

    def xtKy_matmul(self, x: np.ndarray | sp.spmatrix, y: np.ndarray) -> float | np.ndarray:
        """Bilinear form ``xᵀ K y`` via **direct matmul**.

        Returns the paired ``(M,)`` diagonal of ``Xᵀ K Y`` (sparse-aware on
        ``x``). For the full ``(M_x, M_y)`` bipartite cross matrix build it
        explicitly as ``X.T @ self.Kx(Y)`` — that's what
        :class:`DetectorIrregular` does for ``compute_rstat`` when the two
        feature blocks have different widths.

        Parameters
        ----------
        x, y : np.ndarray or scipy.sparse matrix
            ``(n,)`` or ``(n, M)``.

        Returns
        -------
        float or np.ndarray
            Scalar for 1-D inputs; ``(M,)`` for batched.
        """
        Ky = self.Kx(y)
        if sp.issparse(x):
            x_in = x.reshape(-1, 1) if x.ndim == 1 else x
            Ky_2d = Ky[:, None] if Ky.ndim == 1 else Ky
            result = np.asarray(x_in.multiply(Ky_2d).sum(axis=0)).ravel()
            return float(result[0]) if result.size == 1 else result
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1 and Ky.ndim == 1:
            return float(np.dot(x_arr, Ky))
        x_mat = x_arr.reshape(-1, 1) if x_arr.ndim == 1 else x_arr
        Ky_mat = Ky.reshape(-1, 1) if Ky.ndim == 1 else Ky
        return np.sum(x_mat * Ky_mat, axis=0).astype(np.float64)

    def Kx_grid(self, x: np.ndarray) -> np.ndarray:
        """Grid-domain companion of :meth:`Kx` — ``(ny, nx)`` spatial output.

        Whereas :meth:`Kx` returns the length-``n`` apply at the irregular
        coordinates, :meth:`Kx_grid` returns the apply evaluated on the
        internal uniform grid. Pipeline: type-1 NUFFT → undo the
        coordinate-centering phase (needed here because we keep complex
        coefficients; the square-magnitude and adjoint-round-trip paths of
        :meth:`xtKx` / :meth:`Kx` absorb it automatically) → multiply by
        ``λ(k)`` → ``ifftshift`` → ``ifft2`` → real.

        Parameters
        ----------
        x : np.ndarray
            ``(n,)`` or ``(n, M)``.

        Returns
        -------
        np.ndarray
            Real ``(ny, nx)`` or ``(ny, nx, M)`` in the scipy FFT layout (DC
            at ``[0, 0]``).
        """
        ny, nx = self.grid_shape
        dy, dx = self.spacing
        squeeze = x.ndim == 1
        x_hat_centered = self._nufft_type1(x)  # (M, ny, nx), modes ∈ [-n/2, n/2-1]
        m_y = np.arange(ny) - ny // 2
        m_x = np.arange(nx) - nx // 2
        phase = (
            np.exp(-1j * m_y * self._y_mean * 2.0 * np.pi / (ny * dy))[:, None]
            * np.exp(-1j * m_x * self._x_mean * 2.0 * np.pi / (nx * dx))[None, :]
        )
        # Apply spectrum + undo centering phase in one pass.
        lam_centred = np.fft.fftshift(self._fft_kernel.spectrum.reshape(ny, nx))
        weighted = x_hat_centered * phase[None, :, :] * lam_centred[None, :, :]
        # Shift DC to [0, 0] and inverse-FFT to spatial grid.
        weighted_shifted = np.fft.ifftshift(weighted, axes=(-2, -1))
        Kx_grid = np.real(
            scipy.fft.ifft2(weighted_shifted, axes=(-2, -1), workers=self.workers)
        )  # (M, ny, nx)
        out = np.moveaxis(Kx_grid, 0, -1)  # (ny, nx, M)
        return out[..., 0] if squeeze else out

    # ------------------------------------------------------------------
    # Null-moment estimators — doubled-grid linear-convolution analytic
    # (default) and Rademacher Hutchinson probe (opt-in second opinion).
    # ------------------------------------------------------------------
    def trace(self) -> float:
        """``trace(K)`` (raw) or ``trace(HKH)`` (centered).

        Closed-form ``(n/n') · Σ_k λ(k)`` — exact because the diagonal
        of ``G = UᴴU`` is ``n`` regardless of coord arrangement, so
        ``trace(K_n) = (n/n') · trace(K_grid)`` is independent of the
        coord layout. Adjusts by ``-s₁/n`` when ``centering=True``
        (``s₁ = 𝟏ᵀ K 𝟏`` via a single ``K·𝟏`` apply in
        :meth:`Kernel._ones_stats`).
        """
        ny, nx = self.grid_shape
        raw = float(self._fft_kernel.trace() * self.n / (ny * nx))
        if not self.centering:
            return raw
        s1, _ = self._ones_stats()
        return raw - s1 / self.n

    def square_trace(self) -> float:
        """``trace(K²)`` (raw) or ``trace((HKH)²)`` (centered).

        Evaluate ``(n²/n'²) · Σ λ(k) λ(k') |φ(k'-k)|²`` by linear
        convolution. The real operator's weights include both Nyquist
        signs, and the doubled mode range contains every difference.
        Centering subtracts ``2·s₂/n − s₁²/n²``.
        """
        ny, nx = self.grid_shape
        nprime = ny * nx
        lam = self._real_spectrum()
        phi2_d = np.abs(self._coord_phi()) ** 2
        lam_pad_centered = np.zeros_like(phi2_d)
        sy = phi2_d.shape[0] // 2 - lam.shape[0] // 2
        sx = phi2_d.shape[1] // 2 - lam.shape[1] // 2
        lam_pad_centered[sy : sy + lam.shape[0], sx : sx + lam.shape[1]] = lam
        lam_pad = np.fft.ifftshift(lam_pad_centered)
        lam_f = np.fft.fft2(lam_pad)
        phi2_f = np.fft.fft2(phi2_d)
        conv = np.fft.ifft2(lam_f * phi2_f).real  # (|φ|² ⋆ λ)(k)
        # Element-wise product with ``lam_pad`` auto-restricts the outer
        # sum to the original k-grid support.
        raw = (self.n**2 / nprime**2) * float(np.sum(lam_pad * conv))
        # ``trace(K²) = Σᵢ μᵢ² ≥ 0`` for any real symmetric ``K``. Negative
        # values here are FFT-cancellation noise on indefinite ``Λ``
        # (e.g. Moran), not a valid result — clip to zero.
        raw = max(raw, 0.0)
        if not self.centering:
            return raw
        s1, s2 = self._ones_stats()
        return max(raw - 2.0 * s2 / self.n + s1**2 / (self.n**2), 0.0)


# ---------------------------------------------------------------------------
# Q-test and R-test on irregular spatial coordinates via NUFFT
# ---------------------------------------------------------------------------


def _standardize_features(X: np.ndarray) -> np.ndarray:
    """Z-score each column (ddof=1), leaving constant columns as zeros.

    Matches :func:`sonic.statistics.spatial_q_test`'s convention. Used by the
    NUFFT dispatch to standardize at the ``n`` irregular points before the
    type-1 NUFFT.
    """
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True, ddof=1)
    out = np.zeros_like(X, dtype=float)
    valid = sd > 1e-12
    np.divide(X - mu, sd, out=out, where=valid)
    return out


def _q_test_nufft(  # noqa: C901
    Xn: np.ndarray,
    kernel: NUFFTKernel,
    null_params: dict | None = None,
    return_pval: bool = True,
    is_standardized: bool = False,
) -> float | np.ndarray | tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """
    Spatial Q-test on irregular 2D coordinates.

    Computes ``Q = xᵀ K x`` via the k-space Parseval identity
    ``Q = (1/n') Σ_k λ(k) · |ẑ(k)|²`` (one type-1 NUFFT of ``z`` + a
    Parseval sum; see :meth:`NUFFTKernel.xtKx`). The length-``n`` matmul
    counterpart :meth:`NUFFTKernel.xtKx_matmul` computes the same form to
    NUFFT precision and is exposed for callers that prefer the direct
    round-trip; :func:`spatial_q_test` always uses the spectral path.

    Null moments route through :func:`sonic.statistics.compute_null_params`,
    which on graph kernels defaults to the empirical moment estimator over
    ``HKH``-centered probes (see :meth:`NUFFTKernel.trace` /
    :meth:`NUFFTKernel.square_trace`).

    Standardization at the ``n`` irregular points is applied internally
    unless ``is_standardized=True``.

    Parameters
    ----------
    Xn : np.ndarray
        ``(n,)`` or ``(n, M)``.
    kernel : NUFFTKernel
    null_params : dict, optional
        Pre-built moments (see :func:`sonic.compute_null_params`). Read
        keys depend on the null approximation selected via
        ``null_params['method']``: ``'mean_Q'`` / ``'var_Q'`` for CLT,
        ``'scale_g'`` / ``'df_h'`` (or ``'mean_Q'`` / ``'var_Q'`` as
        fallback) for Welch, and ``'liu_coef'`` (preferred) or
        ``'cumulants'`` for Liu. Pass ``None`` to auto-build.
    return_pval : bool, default True
    is_standardized : bool, default False

    Returns
    -------
    Q : float or np.ndarray
    pval : float or np.ndarray, optional
    """
    Xn = np.asarray(Xn, dtype=float)
    batched = Xn.ndim == 2
    X_in = Xn if batched else Xn[:, None]
    if X_in.shape[0] != kernel.n:
        raise ValueError(f"Xn first dim {X_in.shape[0]} does not match kernel.n={kernel.n}.")

    z = X_in if is_standardized else _standardize_features(X_in)

    # Spectral Parseval path — agrees with xtKx_matmul to NUFFT precision
    # (verified on both regular and irregular grids). Keeping only one path
    # here avoids a stateful switch on the kernel object.
    Q_arr = np.atleast_1d(kernel.xtKx(z)).ravel()

    if not return_pval:
        return Q_arr if batched else float(Q_arr[0])

    # Dispatch on the user-selected null approximation. This mirrors the
    # MatrixKernel path in `spatial_q_test`: the caller picks one of
    # {'clt', 'welch', 'liu'} via ``null_params['method']``; defaults keep
    # backward-compatible behavior — CLT for Moran (its kernel is
    # indefinite so Welch/Liu are degenerate) and Liu for the PSD kernels.
    if null_params is not None and "method" in null_params:
        null_approx = str(null_params["method"])
    else:
        null_approx = "clt" if kernel.method == "moran" else "liu"
    if kernel.method == "moran" and null_approx != "clt":
        raise ValueError(
            f"Moran's I kernel is indefinite; only null_method='clt' is "
            f"supported for the Q-test. Got method={null_approx!r}."
        )

    def _get_mean_var() -> tuple[float, float]:
        """Mean/var of Q under H0 — from user-supplied params or recompute."""
        if null_params is not None and "mean_Q" in null_params and "var_Q" in null_params:
            return float(null_params["mean_Q"]), float(null_params["var_Q"])
        # Route through compute_null_params to pick up the H-centering +
        # finite-n ratio correction; for NUFFT graph kernels the
        # ``'empirical'`` default on trace()/square_trace() ensures the
        # corrections capture the spreading-kernel smoothing too.
        from sonic.statistics import compute_null_params

        p = compute_null_params(kernel, method="clt")
        return float(p["mean_Q"]), float(p["var_Q"])

    if null_approx == "clt":
        mean_Q, var_Q = _get_mean_var()
        sigma = float(np.sqrt(var_Q))
        if sigma <= 1e-12:
            pvals = np.ones_like(Q_arr)
        else:
            z_scores = (Q_arr - mean_Q) / sigma
            pvals = chi2.sf(z_scores**2, df=1)

    elif null_approx == "welch":
        from sonic.statistics import _welch_apply

        if null_params is not None and "scale_g" in null_params and "df_h" in null_params:
            pvals = _welch_apply(Q_arr, null_params)
        else:
            mean_Q, var_Q = _get_mean_var()
            pvals = _welch_apply(Q_arr, {"mean_Q": mean_Q, "var_Q": var_Q})

    elif null_approx == "liu":
        from sonic.statistics import (
            _hutchinson_cumulants,
            _liu_apply,
            _liu_prepare,
            _liu_prepare_from_cumulants,
        )

        # Prefer cached Liu coefficients from compute_null_params; derive
        # from caller-supplied ``cumulants`` otherwise. ``n`` is passed
        # for the Dirichlet(1/2) variance correction — essential on
        # broad-spectrum PSD kernels (CAR / graph_laplacian) where
        # c_1² ≈ m·c_2 would otherwise inflate sigma_Q by O(10×).
        n_kernel = int(kernel.n)
        coef = None if null_params is None else null_params.get("liu_coef")
        if coef is None and null_params is not None and "cumulants" in null_params:
            coef = _liu_prepare_from_cumulants(null_params["cumulants"], n=n_kernel)
        if coef is None:
            # ``null_params`` empty or just ``{"method": "liu"}`` — build
            # the coef from the kernel directly. Any other caller-supplied
            # keys without ``liu_coef`` / ``cumulants`` is considered
            # malformed and raises.
            if null_params is not None and null_params.keys() - {"method"}:
                raise ValueError(
                    "null_params with method='liu' must contain either "
                    "'liu_coef' (preferred) or 'cumulants'. Build via "
                    "compute_null_params(kernel, method='liu')."
                )
            # Try the exact Toeplitz-M eigendecomposition; if that's
            # unavailable (broad-support or indefinite Λ), fall back
            # to Hutchinson-estimated cumulants.
            try:
                evals = kernel.eigenvalues(return_full_layout=True)
                if evals.min() < -0.1:
                    raise ValueError(
                        "Kernel has significant negative eigenvalues; "
                        "Liu's method may be invalid."
                    )
                sig_evals = evals[evals > 1e-9]
                coef = _liu_prepare(sig_evals, n=n_kernel)
            except NotImplementedError:
                c = _hutchinson_cumulants(kernel, n_probes=60)
                coef = _liu_prepare_from_cumulants(c, n=n_kernel)
        pvals = np.atleast_1d(_liu_apply(Q_arr, coef))

    else:
        raise ValueError(f"Unknown null approximation method: {null_approx!r}")

    if batched:
        return Q_arr, pvals
    return float(Q_arr[0]), float(pvals[0])


def _r_test_nufft(
    Xn: np.ndarray,
    Yn: np.ndarray,
    kernel: NUFFTKernel,
    null_params: dict | None = None,
    return_pval: bool = True,
    is_standardized: bool = False,
) -> float | np.ndarray | tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """
    Spatial R-test on irregular 2D coordinates.

    Dispatches purely on the input shape:

    - Paired (``M_x == M_y``) — returns the ``(M,)`` diagonal of
      ``Xᵀ K Y`` via cross Parseval
      ``R = (1/n') Σ_k λ(k) · conj(x̂(k)) · ŷ(k)`` (one pair of
      type-1 NUFFTs; see :meth:`NUFFTKernel.xtKy`).
    - Bipartite (``M_x != M_y``) — returns the full ``(M_x, M_y)`` cross
      matrix via ``Xᵀ · self.Kx(Y)``.

    In either case ``var_R = kernel.square_trace()`` (the default
    ``centering=True`` returns ``trace((HKH)²)``, which is exactly
    ``Var[Xᵀ K Y]`` on z-scored inputs).

    Parameters
    ----------
    Xn, Yn : np.ndarray
        ``(n,)`` or ``(n, M)``.
    kernel : NUFFTKernel
    null_params : dict, optional
        ``{'var_R': ...}`` in the n-point-operator units.
    return_pval : bool, default True
    is_standardized : bool, default False
    """
    Xn = np.asarray(Xn, dtype=float)
    Yn = np.asarray(Yn, dtype=float)
    if Xn.ndim == 1:
        Xn = Xn[:, None]
    if Yn.ndim == 1:
        Yn = Yn[:, None]
    if Xn.shape[0] != kernel.n or Yn.shape[0] != kernel.n:
        raise ValueError(
            f"Xn, Yn first dim must equal kernel.n={kernel.n}; "
            f"got {Xn.shape[0]}, {Yn.shape[0]}."
        )

    Xz = Xn if is_standardized else _standardize_features(Xn)
    Yz = Yn if is_standardized else _standardize_features(Yn)

    if Xn.shape[1] == Yn.shape[1]:
        # Paired diagonal via cross Parseval — one type-1 NUFFT per column.
        R = np.atleast_1d(kernel.xtKy(Xz, Yz))
    else:
        # Bipartite (M_x, M_y) cross matrix via NUFFT round-trip on Y.
        KY = kernel.Kx(Yz)  # (n, M_y)
        R = Xz.T @ KY  # (M_x, M_y)

    if not return_pval:
        return R.squeeze() if R.size > 1 else float(R)

    if null_params is not None and "var_R" in null_params:
        var_R = float(null_params["var_R"])
    else:
        # kernel.square_trace() returns trace((HKH)²) by default (centering=True),
        # which is exactly Var[R] for Zₓᵀ K Zᵧ with both X, Y z-scored.
        var_R = float(kernel.square_trace())
    sigma = float(np.sqrt(max(var_R, 1e-30)))
    z_scores = R / sigma
    pvals = 2.0 * norm.sf(np.abs(z_scores))
    return R.squeeze(), pvals.squeeze()
