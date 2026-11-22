"""Shared spatial contribution maps, independent of detector/comparator containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sonic.utils import _chunk_parameters, _kernel_workspace_bytes, _parse_memory_budget


@dataclass
class SpatialExplanation:
    """Observed values, operator responses and additive spatial contributions.

    ``values`` has shape (n_observations, n_features), ``filtered`` has shape
    (n_maps, n_observations, n_features), and ``contributions`` has shape
    (n_maps, n_observations). Detector Q/R explanations contain one map and
    one/two features:

    - ``filtered``: Lx for one feature, or (Lx, Ly) in feature order.
    - ``contributions``: elementwise x * Lx for one feature, or
      (x * Ly + y * Lx) / 2 for a pair.

    Here x and y denote test-prepared features (``values`` after preparation),
    and L is the fitted real symmetric kernel operator, including its scaling.
    Comparator maps have one map per selected bin. Their real ``filtered``
    fields include observed normalization and, for polar coherence, the two
    conjugate phase factors. The same elementwise allocation then applies;
    metadata records the representation and retained bin indices.

    ``observation_indices`` maps rows back to the original AnnData passed to
    setup_data, or to the C-order flattened raster for grid detectors.
    ``coordinates`` is optional: AnnData coordinates retain their source column
    order; grid coordinates are (y, x) on the kernel lattice. Units are not
    inferred. ``metadata`` describes the statistic, source and normalization.

    Scores are derived from contributions. Construction by SONIC checks these
    against independently evaluated statistics before discarding the reference.
    No kernels, Fourier coefficients, input containers or p-values are retained.
    """

    feature_names: tuple[str, ...]
    observation_indices: np.ndarray
    coordinates: np.ndarray | None
    values: np.ndarray
    filtered: np.ndarray
    contributions: np.ndarray
    metadata: dict[str, Any]

    @property
    def scores(self) -> np.ndarray:
        """Sum of spatial contributions for each map."""
        return self.contributions.sum(axis=1)

    def select_regions(  # noqa: C901 — candidate validation and greedy window selection
        self,
        map_index=0,
        *,
        size=None,
        n_regions=2,
        direction="absolute",
        min_coverage=0.35,
        min_observations=64,
        memory_budget_bytes="256 MiB",
    ):
        """Select nonoverlapping windows enriched for signed contributions.

        Parameters
        ----------
        map_index : int, default 0
            Position in the returned explanation maps.
        size : float, optional
            Width of the requested square window in stored coordinate units.
            For spectral maps, None uses max(3 / lower_frequency,
            1 / bandwidth), divided by metadata['unit_scale']. Q/R maps and
            bands starting at zero require an explicit size. Smaller explicit
            windows are allowed and flagged when a recommendation is available.
        n_regions : int, default 2
            Maximum number of nonoverlapping windows to return. Must be positive.
        direction : {'absolute', 'positive', 'negative'}, default 'absolute'
            Rank by the absolute signed contribution sum, the positive sum,
            or the negated negative sum. Opposite signs can cancel within a
            window. Zero-score windows and windows of the wrong sign are excluded.
        min_coverage : float, default 0.35
            Minimum occupied fraction of a window's 20-by-20 spatial tiles.
            Must be in (0, 1]. A tile is considered occupied if it contains at least
            one observed location.
        min_observations : int, default 64
            Minimum number of observed locations in each window. Must be positive.
        memory_budget_bytes : int or str, default '256 MiB'
            Estimated selection-workspace budget in bytes or a size string.

        Returns
        -------
        regions : pandas.DataFrame
            Windows in selection order, with up to n_regions rows. An empty
            table means no eligible window was selected. Columns are:

            - region: one-based selection order.
            - coord0_min, coord1_min, coord0_max, coord1_max: bounds in stored
              coordinate-column order, using half-open intervals [min, max).
            - size: requested window width in stored coordinate units.
            - recommended_size: spectral context-width heuristic, or NaN when
              unavailable.
            - coverage: fraction of occupied spatial tiles.
            - n_observations: number of observed locations in the window.
            - contribution_sum: sum of the original signed contributions.
            - context_limited: whether size is smaller than an available
              recommended_size.

        Notes
        -----
        Candidate square windows advance by size / 4 and must fit inside the
        coordinate bounding box. When picking the best windows, a Greedy search
        selects the highest-scoring window, discard candidates overlapping it,
        then select the best remaining window until enough regions are selected
        or no eligible candidates remain.

        Filtering and normalization, when applied, always consider the
        full-sample context, which may not be available in a cropped region.
        """
        import pandas as pd

        coords, contribution = _spatial_map(self, map_index)
        for name, value in (("n_regions", n_regions), ("min_observations", min_observations)):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if direction not in ("absolute", "positive", "negative"):
            raise ValueError("direction must be 'absolute', 'positive', or 'negative'.")
        if not np.isfinite(min_coverage) or not 0 < min_coverage <= 1:
            raise ValueError("min_coverage must be in (0, 1].")
        recommended = _recommended_region_size(self.metadata, map_index)
        size = recommended if size is None else float(size)
        if not np.isfinite(size) or size <= 0:
            raise ValueError(
                "Supply a finite positive size; Q/R and zero-edge bands have no default."
            )
        budget = _parse_memory_budget(memory_budget_bytes)
        columns = [
            "region",
            "coord0_min",
            "coord1_min",
            "coord0_max",
            "coord1_max",
            "size",
            "recommended_size",
            "coverage",
            "n_observations",
            "contribution_sum",
            "context_limited",
        ]
        origin, extent = coords.min(axis=0), np.ptp(coords, axis=0)
        if np.any(extent < size):
            return pd.DataFrame(columns=columns)
        tiles = 20
        pitch = size / tiles
        shape_float = np.floor(extent / pitch)
        cells = float(np.prod(shape_float))
        # Coordinate indexing, weighted binning and integral images peak at different times.
        n = len(coords)
        workspace = 8 * (shape_float.sum() + 2) + max(
            40 * n, 18 * n + 9 * cells, 9 * n + 20 * cells
        )
        if not np.isfinite(workspace) or workspace > budget:
            raise MemoryError(
                "Region tiling exceeds the memory budget; increase size or memory_budget_bytes."
            )
        shape = shape_float.astype(int)
        edges = [origin[d] + np.arange(shape[d] + 1) * pitch for d in (0, 1)]
        ij = np.column_stack(
            [np.searchsorted(edges[d], coords[:, d], side="right") - 1 for d in (0, 1)]
        )
        inside = ((ij >= 0) & (ij < shape)).all(axis=1)
        flat = (ij[:, 0] * shape[1] + ij[:, 1])[inside]
        del ij
        count = np.bincount(flat, minlength=int(cells)).reshape(shape)
        coverage = _window_sum(count > 0, tiles, 5) / tiles**2
        observations = _window_sum(count, tiles, 5)
        del count
        total = np.bincount(flat, weights=contribution[inside], minlength=int(cells)).reshape(shape)
        del flat, inside
        sums = _window_sum(total, tiles, 5)
        del total
        rank = (
            abs(sums) if direction == "absolute" else sums * (1 if direction == "positive" else -1)
        )
        eligible = (coverage >= min_coverage) & (observations >= min_observations) & (rank > 0)
        candidates = np.argwhere(eligible) * 5
        order = np.argsort(-rank[eligible], kind="stable")
        chosen, rows = [], []
        for index in order:
            start = candidates[index]
            # Skip any candidate that overlaps a previously chosen window.
            if any(np.all(abs(start - previous) < tiles) for previous in chosen):
                continue
            lower = np.array([edges[d][start[d]] for d in (0, 1)])
            upper = np.array([edges[d][start[d] + tiles] for d in (0, 1)])
            keep = _region_mask(coords, lower, upper)
            signed_sum = contribution[keep].sum()
            signed_rank = (
                abs(signed_sum)
                if direction == "absolute"
                else signed_sum * (1 if direction == "positive" else -1)
            )
            if signed_rank <= 0:
                continue  # Integral-image roundoff must not create zero or wrong-sign crops.
            chosen.append(start)
            rows.append(
                [
                    len(rows) + 1,
                    *lower,
                    *upper,
                    size,
                    recommended,
                    coverage[tuple(start // 5)],
                    int(keep.sum()),
                    signed_sum,
                    bool(np.isfinite(recommended) and size < recommended),
                ]
            )
            if len(rows) == n_regions:
                break
        return pd.DataFrame(rows, columns=columns)

    def plot(  # noqa: C901 — validated multi-panel rendering
        self,
        map_index=0,
        *,
        regions=None,
        title=None,
        limits=None,
        coordinate_order=None,
        point_size=3.0,
    ):
        """Plot original values, filtered responses and signed contributions.

        Parameters
        ----------
        map_index : int, default 0
            Position in the returned explanation maps.
        regions : pandas.DataFrame, optional
            Table returned by select_regions(), or a table containing finite
            coord0_min, coord1_min, coord0_max and coord1_max bounds in stored
            coordinate-column order. Each half-open crop must have positive
            width and contain observations. None shows only the overview.
        title : str, optional
            Figure heading. Defaults to the selected feature names.
        limits : array_like, shape (2 * n_features + 1, 2), optional
            Finite increasing (low, high) color limits, in panel-column order:
            original values for each feature, filtered responses for each
            feature, then signed contributions. Limits are shared across all
            rows in a column. None uses the 1st/99th percentiles for original
            values and symmetric 99th-percentile absolute limits for signed
            fields. Supply common limits to compare multiple figures.
        coordinate_order : tuple of int, optional
            Stored coordinate columns to use as (horizontal, vertical), either
            (0, 1) or (1, 0). Defaults to (1, 0) for known grid coordinates and
            (0, 1) otherwise. The vertical axis increases downward for known
            grids and upward for other sources, regardless of this ordering.
        point_size : float, default 3.0
            Positive marker area in points squared for zoom-panel observations.

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure with one overview row followed by one zoom row per supplied
            region, and 2 * n_features + 1 map columns with colorbars. The
            figure is neither shown nor saved automatically.

        Notes
        -----
        Overview tiles show observation means on a grid with at most 512 tiles per
        axis while empty tiles stay masked. Zoom panels show every observed location.
        """
        coords, contribution = _spatial_map(self, map_index)
        n, g = self.values.shape
        arrays = [*self.values.T, *self.filtered[map_index].T, contribution]
        if len(arrays) != 2 * g + 1 or any(
            a.shape != (n,) or not np.isfinite(a).all() for a in arrays
        ):
            raise ValueError("Plot fields must be finite and match the observed coordinates.")
        grid = (
            self.metadata.get("backend") == "fft"
            or self.metadata.get("coordinate_system") == "local_raster_yx"
        )
        order = (1, 0) if grid else (0, 1)
        if coordinate_order is not None:
            order = tuple(coordinate_order)
        if order not in ((0, 1), (1, 0)) or any(
            isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) for d in order
        ):
            raise ValueError("coordinate_order must be (0, 1) or (1, 0).")
        if not np.isfinite(point_size) or point_size <= 0:
            raise ValueError("point_size must be finite and positive.")
        bounds = (
            np.empty((0, 4))
            if regions is None
            else np.asarray(
                regions[["coord0_min", "coord1_min", "coord0_max", "coord1_max"]], dtype=float
            )
        )
        if (
            bounds.ndim != 2
            or bounds.shape[1] != 4
            or not np.isfinite(bounds).all()
            or np.any(bounds[:, 2:] <= bounds[:, :2])
        ):
            raise ValueError("Region bounds must be finite with positive width in both dimensions.")
        for box in bounds:
            if not _region_mask(coords, box[:2], box[2:]).any():
                raise ValueError("Every plotted region must contain observed locations.")
        if limits is None:
            limits = []
            for column, array in enumerate(arrays):
                if column < g:
                    lo, hi = np.quantile(array, [0.01, 0.99])
                    if lo == hi:
                        padding = max(0.5, abs(lo) * 0.01)
                        lo, hi = lo - padding, hi + padding
                else:
                    hi = float(np.quantile(abs(array), 0.99)) or 1.0
                    lo = -hi
                limits.append((lo, hi))
        limits = np.asarray(limits, dtype=float)
        if (
            limits.shape != (len(arrays), 2)
            or not np.isfinite(limits).all()
            or np.any(limits[:, 1] <= limits[:, 0])
        ):
            raise ValueError("Provide one finite increasing color-limit pair per panel.")

        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.patches import Rectangle

        labels = [
            *(f"{name}: original" for name in self.feature_names),
            *(f"{name}: filtered response" for name in self.feature_names),
            "Signed contribution",
        ]
        fig, axes = plt.subplots(
            1 + len(bounds),
            len(arrays),
            figsize=(3.4 * len(arrays), 3.2 * (1 + len(bounds))),
            squeeze=False,
            constrained_layout=True,
        )
        origin, end = coords.min(0), coords.max(0)
        extent = end - origin
        resolution = min(512, max(1, int(np.ceil(np.sqrt(n)))))
        pitch = extent.max() / resolution if extent.max() > 0 else 1.0
        edges = [
            np.linspace(origin[d], end[d], max(1, int(np.ceil(extent[d] / pitch))) + 1)
            if extent[d] > 0
            else np.array([origin[d] - 0.5, origin[d] + 0.5])
            for d in (0, 1)
        ]
        count = np.histogram2d(*coords.T, bins=edges)[0]
        x, y = order
        view = [edges[x][0], edges[x][-1], edges[y][0], edges[y][-1]]
        for col, (array, label, clim) in enumerate(zip(arrays, labels, limits, strict=True)):
            cmap = plt.get_cmap("viridis" if col < g else "RdBu_r").copy()
            cmap.set_bad("#eeeeee")
            norm = Normalize(*clim)
            sums = np.histogram2d(*coords.T, bins=edges, weights=array)[0]
            means = np.divide(sums, count, out=np.full_like(sums, np.nan), where=count > 0)
            image = axes[0, col].imshow(
                means.T if x == 0 else means,
                extent=view,
                origin="lower",
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
            )
            axes[0, col].set_title(label, fontsize=10)
            for row, box in enumerate(bounds, 1):
                lower, upper = box[:2], box[2:]
                keep = _region_mask(coords, lower, upper)
                axes[0, col].add_patch(
                    Rectangle(
                        (lower[x], lower[y]),
                        upper[x] - lower[x],
                        upper[y] - lower[y],
                        fill=False,
                        edgecolor="#b07800",
                        linewidth=1.2,
                    )
                )
                axes[0, col].text(lower[x], lower[y], str(row), color="#805400", fontsize=9)
                axes[row, col].scatter(
                    coords[keep, x],
                    coords[keep, y],
                    c=array[keep],
                    s=point_size,
                    linewidths=0,
                    cmap=cmap,
                    norm=norm,
                    rasterized=True,
                )
                axes[row, col].set(xlim=(lower[x], upper[x]), ylim=(lower[y], upper[y]))
                if col == len(arrays) - 1:
                    axes[row, col].set_title(
                        f"Region {row}: sum = {contribution[keep].sum():.3g}; n = {keep.sum()}",
                        fontsize=9,
                    )
            fig.colorbar(image, ax=list(axes[:, col]), shrink=0.5, pad=0.02, extend="both")
        for ax in axes.flat:
            ax.set_aspect("equal")
            ax.set_xlabel(f"Coordinate {x}")
            if grid:
                ax.invert_yaxis()
        for row in range(len(bounds) + 1):
            axes[row, 0].set_ylabel(
                f"{'Overview' if row == 0 else f'Region {row}'}\nCoordinate {y}"
            )
        statistic = self.metadata.get("statistic", "score")
        subtitle = f"{statistic} = {contribution.sum():.4g}"
        intervals = self.metadata.get("frequency_bin_intervals")
        if intervals is not None:
            lo, hi = intervals[map_index]
            subtitle += f"; frequency [{lo:.3g}, {hi:.3g}]"
            angles = self.metadata.get("angle_centers_rad")
            if angles is not None:
                subtitle += f"; aligned angle {np.rad2deg(angles[map_index]):.1f}°"
        fig.suptitle(f"{title or ' / '.join(self.feature_names)}\n{subtitle}", fontsize=12)
        fig.supxlabel(
            "Overview: tile means; zoom: individual observations. Colors may clip tails.\n"
            "Crops retain full-sample context. Region selection is descriptive, without local p-values.",
            fontsize=9,
        )
        return fig


def _spatial_map(explanation, map_index):
    """Validate shared spatial inputs without changing the stored map."""
    if (
        isinstance(map_index, (bool, np.bool_))
        or not isinstance(map_index, (int, np.integer))
        or not 0 <= map_index < len(explanation.contributions)
    ):
        raise ValueError("map_index must index a returned explanation map.")
    if explanation.coordinates is None:
        raise ValueError("Spatial coordinates are required for plotting and region selection.")
    coords = np.asarray(explanation.coordinates, dtype=float)
    contribution = np.asarray(explanation.contributions[map_index], dtype=float)
    if (
        coords.shape != (len(contribution), 2)
        or len(coords) == 0
        or not np.isfinite(coords).all()
        or not np.isfinite(contribution).all()
    ):
        raise ValueError("Coordinates and contributions must be finite and have matching rows.")
    return coords, contribution


def _recommended_region_size(metadata, map_index):
    """Heuristic context width in stored coordinate units, or NaN for Q/R/DC."""
    intervals = metadata.get("frequency_bin_intervals")
    if intervals is None:
        return np.nan
    lo, hi = intervals[map_index]
    scale = metadata.get("unit_scale", 1.0)
    if not np.isfinite([lo, hi, scale]).all() or lo < 0 or hi <= lo or scale <= 0:
        raise ValueError("Frequency intervals and unit_scale must define positive physical scales.")
    return max(3 / lo, 1 / (hi - lo)) / scale if lo > 0 else np.nan


def _region_mask(coords, lower, upper):
    return ((coords >= lower) & (coords < upper)).all(axis=1)


def _window_sum(array, width, step):
    """Sum only candidate windows, using one padded integral image."""
    ny, nx = array.shape
    integral = np.zeros((ny + 1, nx + 1), dtype=np.result_type(array.dtype, np.int64))
    core = integral[1:, 1:]
    core[:] = array
    np.cumsum(core, axis=0, out=core)
    np.cumsum(core, axis=1, out=core)
    sums = integral[width::step, width::step] - integral[:-width:step, width::step]
    sums -= integral[width::step, :-width:step]
    sums += integral[:-width:step, :-width:step]
    return sums


def _check_explanation_budget(kernel, n_features, memory_budget_bytes):
    """Check selected-feature workspace and returned arrays before loading data.

    Excludes resident input/kernel storage, lazy factorization and library
    overhead; this is an allocation estimate rather than an RSS limit.
    """
    budget = _parse_memory_budget(memory_budget_bytes)
    per_feature = _chunk_parameters(kernel)[1]
    transform = max(2 * n_features * per_feature, _kernel_workspace_bytes(kernel, n_features))
    estimate = transform + 8 * kernel.n * (3 * n_features + 5)
    if estimate > budget:
        raise MemoryError(
            f"Estimated explanation workspace {estimate} bytes exceeds budget {budget} bytes."
        )


def _explanation_from_responses(  # noqa: C901 — array boundary validation and reconstruction
    values,
    prepared,
    filtered,
    expected_scores,
    *,
    feature_names,
    observation_indices,
    coordinates=None,
    metadata,
    rtol=1e-9,
):
    """Build maps showing how each observation contributes to a score.

    The caller supplies the values used by the test and their spatially
    filtered values. This helper combines them into contribution maps and
    checks that each map sums to a separately computed score.

    Let x and y be columns of ``prepared``, and Lx and Ly the matching columns
    of ``filtered`` after applying the spatial kernel L. At each observation:

    - One feature contributes x * Lx.
    - A pair contributes (x * Ly + y * Lx) / 2.

    These products are elementwise. Detector pairs use a real symmetric kernel;
    more generally the caller may supply matching adjoint responses for the
    two features (as for polar coherence). All centering, scaling and filtering
    happen before this call;
    ``values`` holds the original observed values for display.

    Parameters
    ----------
    values : array_like, shape (n_observations, n_features)
        Original observed values to keep for display. Must be finite and real,
        with at least two observations and one or two feature columns.
    prepared : array_like, shape (n_observations, n_features)
        Values used by the test, after any required centering and scaling.
        Must be finite and real, with the same row and column order as values.
        Used to calculate contributions, but not stored in the result.
    filtered : array_like, shape (n_maps, n_observations, n_features)
        Results of applying each spatial kernel to prepared. Must be finite
        and real, in the same feature order: Lx for one feature, (Lx, Ly) for
        a pair. Include all kernel scaling; at least one map is required.
    expected_scores : array_like, shape (n_maps,)
        Separately computed score for each map. Must be finite and real.
        Used only to check that the contributions sum to the expected scores.
    feature_names : sequence of str
        Names of the one or two features, in column order. Two columns use
        the pair formula, even when both names refer to the same feature.
    observation_indices : array_like of int, shape (n_observations,)
        Nonnegative indices locating these observations in the source data:
        original AnnData rows or raster cells flattened in C order.
    coordinates : array_like, shape (n_observations, 2), optional
        Finite spatial coordinates in row order, or None if unavailable.
        Column order and units are kept as supplied.
    metadata : dict
        Descriptions of the statistic, source and normalization. The dictionary
        is copied, but objects inside it are shared. No specific keys are required.
    rtol : float, default 1e-9
        Finite, nonnegative tolerance for the score check. Each map's allowed
        absolute error is rtol times the larger of the absolute expected score
        and the sum of absolute contributions, plus np.finfo(float).tiny.

    Returns
    -------
    SpatialExplanation
        Contribution maps whose sums match the expected scores within tolerance,
        with copies of the original values, filtered values, source indices and
        optional coordinates.
    """
    if any(np.iscomplexobj(a) for a in (values, prepared, filtered, expected_scores)):
        raise ValueError("Quadratic contribution allocation requires real values and responses.")
    values = np.asarray(values, dtype=float)
    prepared = np.asarray(prepared, dtype=float)
    filtered = np.asarray(filtered, dtype=float)
    expected = np.asarray(expected_scores, dtype=float)
    indices = np.asarray(observation_indices)
    names = tuple(feature_names)
    if values.ndim != 2 or values.shape[1] not in (1, 2) or values.shape[0] < 2:
        raise ValueError(
            "Explanation values must contain at least two observations and one/two features."
        )
    n, g = values.shape
    if prepared.shape != values.shape or len(names) != g:
        raise ValueError("Prepared values and feature names must match explanation values.")
    if filtered.ndim != 3 or filtered.shape[1:] != (n, g) or filtered.shape[0] == 0:
        raise ValueError("Filtered fields must have shape (n_maps, n_observations, n_features).")
    if expected.shape != (filtered.shape[0],):
        raise ValueError("Provide one independent reference score per map.")
    if indices.shape != (n,) or not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0):
        raise ValueError("Observation indices must be nonnegative integers matching the rows.")
    if not all(np.isfinite(a).all() for a in (values, prepared, filtered, expected)):
        raise ValueError("Explanation values, responses and reference scores must be finite.")
    if coordinates is not None:
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape != (n, 2) or not np.isfinite(coordinates).all():
            raise ValueError(
                "Explanation coordinates must be finite with shape (n_observations, 2)."
            )
    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError("Reconstruction tolerance must be finite and nonnegative.")

    if g == 1:
        contributions = prepared[:, 0] * filtered[:, :, 0]
    else:
        contributions = 0.5 * (
            prepared[:, 0] * filtered[:, :, 1] + prepared[:, 1] * filtered[:, :, 0]
        )
    scale = np.maximum(np.abs(expected), np.abs(contributions).sum(axis=1))
    error = np.abs(contributions.sum(axis=1) - expected)
    if not np.isfinite(contributions).all() or np.any(error > rtol * scale + np.finfo(float).tiny):
        raise ArithmeticError(
            "Spatial contributions do not reconstruct the reference statistic. "
            "R allocations require a symmetric kernel; check operator accuracy and support."
        )
    return SpatialExplanation(
        feature_names=names,
        observation_indices=indices.copy(),
        coordinates=None if coordinates is None else coordinates.copy(),
        values=values.copy(),
        filtered=filtered.copy(),
        contributions=contributions,
        metadata=dict(metadata),
    )
