"""Compare spectra from regular grids stored in :class:`spatialdata.SpatialData`."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

# Suppress known deprecation warnings from SpatialData dependencies BEFORE importing them.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*legacy Dask DataFrame.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

import spatialdata as sd

from sonic._rasterize import _mean_fill_missing
from sonic.comparators.base import (
    _ComparatorBase,
    _validate_common,
)
from sonic.comparators.features import compute_sample_spectrum
from sonic.utils import _gene_indices

__all__ = ["ComparatorGrid"]

logger = logging.getLogger(__name__)


def _grid_spectrum(
    block: np.ndarray,
    *,
    fft_solver: str,
    workers: int | None,
    return_dc: bool = False,
):
    """Mean-fill structural holes and normalize power by observed bins squared."""
    n_observed = (~np.isnan(block)).sum(axis=(1, 2))
    spectrum, dc = compute_sample_spectrum(
        _mean_fill_missing(block, axis=(1, 2)),
        fft_solver=fft_solver,
        workers=workers,
        return_dc=True,
    )
    # compute_sample_spectrum uses the whole grid; holes carry zero centered
    # signal and must not count as observations in the Fourier average.
    spectrum *= ((block.shape[1] * block.shape[2]) / np.maximum(n_observed, 1))[:, None, None] ** 2
    return (spectrum, dc) if return_dc else spectrum


def _observed_nonzero_fraction(block: np.ndarray) -> np.ndarray:
    """Fraction of nonzero values among observed, rather than bounding-box, bins."""
    observed = ~np.isnan(block)
    denominator = observed.sum(axis=(1, 2))
    numerator = (observed & (block != 0)).sum(axis=(1, 2))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(block.shape[0], dtype=float),
        where=denominator > 0,
    )


# ---------------------------------------------------------------------------
# ComparatorGrid — SpatialData / regular bins via rasterize_bins
# ---------------------------------------------------------------------------


class ComparatorGrid(_ComparatorBase):
    """Compare spatial power spectra on regular rectangular grids.

    See :doc:`/comparator_api` for methods and return values.

    Parameters
    ----------
    samples : sequence of spatialdata.SpatialData
        One object per sample. Use the same table and grid keys in each.
    bins : str
        Key for the spatial bin shapes.
    table_name : str
        Key for the expression table in each sample.
    col_key : str
        Column in the expression table's ``obs`` that gives the grid column index.
    row_key : str
        Column in the expression table's ``obs`` that gives the grid row index.
    value_key : str, optional
        Key for the values to rasterize. The default uses the expression matrix.
    gene_names : sequence of str, optional
        Gene names in input column order. By default, use the first sample's
        ``var_names``. All samples must have the same names in the same order.
    feature_mode : {'radial', '2d'}, default 'radial'
        Radial bins average power over directions at each spatial scale.
        Use ``'2d'`` to retain direction after aligning the sample spectra.
    n_radial_bins : int, default 30
        Number of requested radial intervals when ``freq_edges`` is omitted.
    n_theta_bins : int, default 36
        Number of angular intervals over a half turn for ``feature_mode='2d'``.
    fft_solver : {'fft2', 'rfft2'}, default 'rfft2'
        Fourier transform for the grid. The real-input transform uses less memory.
    workers : int, 'auto', or None, default 'auto'
        Threads per sample for the FFT. Use ``None`` for one thread or
        ``'auto'`` to share available CPUs with concurrent sample jobs.
    spacing : pair of floats or sequence of pairs, optional
        Physical height and width of a grid cell, in that order. Supply one
        pair for all samples or one per sample. The default is ``(1, 1)``.
    freq_edges : np.ndarray, optional
        Shared radial bin edges in cycles per spatial unit. The automatic edges
        cover the frequency range common to all samples.
    adaptive_freq_edges : bool, default False
        Merge adjacent radial intervals when a sample lacks frequency support.
        By default, omit unsupported radial intervals. The retained intervals
        are available in ``frequency_bin_intervals_``.
    presence_threshold : float, default 0.0
        Minimum fraction of observed grid cells with nonzero expression for a gene
        to count as present in a sample.
    fft_chunk_size : int or 'auto', default 'auto'
        Genes per transform block. The automatic size uses the memory budget.
    """

    def __init__(
        self,
        samples: Sequence[Any],
        *,
        bins: str,
        table_name: str,
        col_key: str,
        row_key: str,
        value_key: str | None = None,
        gene_names: Sequence[str] | None = None,
        feature_mode: str = "radial",
        n_radial_bins: int = 30,
        n_theta_bins: int = 36,
        fft_solver: str = "rfft2",
        workers: int | str | None = "auto",
        spacing: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
        freq_edges: np.ndarray | None = None,
        adaptive_freq_edges: bool = False,
        presence_threshold: float = 0.0,
        fft_chunk_size: int | str = "auto",
    ) -> None:
        fft_solver = _validate_common(feature_mode, fft_solver, presence_threshold)
        samples_list = list(samples)
        if len(samples_list) == 0:
            raise ValueError("samples must be a non-empty list.")
        for i, s in enumerate(samples_list):
            if not isinstance(s, sd.SpatialData):
                raise TypeError(
                    f"sample {i} is {type(s).__name__}, expected spatialdata.SpatialData."
                )

        resolved = _resolve_spatialdata_gene_names(samples_list, gene_names, table_name)

        self.samples = samples_list
        self.gene_names = list(resolved)
        self.feature_mode = feature_mode
        self.freq_edges = None if freq_edges is None else np.asarray(freq_edges, dtype=float)
        self.freq_edges_requested_ = None if self.freq_edges is None else self.freq_edges.copy()
        self._adaptive_freq_edges = bool(adaptive_freq_edges)
        # Private (internal-config) state.
        self._n_radial_bins = int(n_radial_bins)
        self._n_theta_bins = self._normalize_n_theta_bins(n_theta_bins)
        self._fft_solver = fft_solver
        self._workers_spec = workers
        from sonic.utils import resolve_parallelism

        self._workers = resolve_parallelism(1, workers, backend=self._spectrum_backend)[1]
        self._presence_threshold = float(presence_threshold)
        # 'auto' is resolved lazily in compute_spectra once grid shapes are
        # known (rasterize_bins determines them per sample); an int is fixed now.
        self._fft_chunk_size_spec = self._normalize_chunk_spec(fft_chunk_size)
        self._fft_chunk_size = (
            self._auto_chunk_cap
            if self._fft_chunk_size_spec == "auto"
            else (self._fft_chunk_size_spec)
        )
        self._spectrum_fft_solver = fft_solver

        self._bins = bins
        self._table_name = table_name
        self._col_key = col_key
        self._row_key = row_key
        self._value_key = value_key

        # Grid shape is determined per-sample inside compute_spectra by rasterize_bins
        # (the raster's .shape[-2:] carries it). Record the placeholder now
        # and fill in during _compute_spectra; spacing defaults to (1.0, 1.0)
        # because rasterize_bins outputs one pixel per bin — users can override
        # via the spacing kwarg when the bins encode a physical pitch. Pass a
        # single (dy, dx) to apply to all samples, or one (dy, dx) per sample
        # so heterogeneous bin pitches map to the same physical frequency units.
        self._spacing_override = self._broadcast_pairs(
            spacing, len(samples_list), name="spacing", cast=float
        )
        self._grid_shapes = []  # populated by _compute_spectra
        self._spacings = None  # populated alongside

    # ------------------------------------------------------------------
    def _rasterize_one(self, sdata: Any):
        """Return the lazy raster so spectrum batches can load individual channels."""
        from sonic._rasterize import rasterize_table

        img = rasterize_table(
            sdata,
            bins=self._bins,
            table_name=self._table_name,
            col_key=self._col_key,
            row_key=self._row_key,
            value_key=self._value_key,
            return_region_as_labels=False,
        )
        if img.ndim != 3:
            raise ValueError(
                f"rasterize_bins returned shape {img.shape}, expected (n_genes, ny, nx)."
            )
        return img

    def _batch_raster(self, img: Any, columns) -> np.ndarray:
        """Read resolved columns, preserving label alignment on reordered rasters."""
        axis = img.get_index("c")
        if not axis.is_unique and axis.tolist() == self.gene_names:
            batch = img.isel(c=columns)
        else:
            batch = img.sel(c=[self.gene_names[j] for j in columns])
        return np.asarray(batch.values, dtype=np.float64)

    def _sample_spectrum_chunker(self, i: int, *, gene_indices=None):
        """Stream selected genes' powers and collect their DC/presence in the same pass."""
        img = self._rasterize_one(self.samples[i])
        columns = np.arange(len(self.gene_names)) if gene_indices is None else gene_indices
        n_genes = len(columns)
        dc = np.empty(n_genes, dtype=float)
        presence = np.empty(n_genes, dtype=bool)

        def _spec_chunk(start: int, stop: int) -> np.ndarray:
            block = self._batch_raster(img, columns[start:stop])
            presence[start:stop] = _observed_nonzero_fraction(block) >= self._presence_threshold
            spectrum, dc[start:stop] = _grid_spectrum(
                block,
                fft_solver=self._spectrum_fft_solver,
                workers=self._workers,
                return_dc=True,
            )
            return spectrum

        return _spec_chunk, dc, presence, n_genes, tuple(img.shape[-2:])

    # ------------------------------------------------------------------
    def _covariate_features_from_keys(self, keys: Sequence[str]) -> list[np.ndarray]:
        """Rasterize the selected gene or observation covariates for each sample."""
        from sonic._rasterize import rasterize_table

        keys = list(keys)
        out: list[np.ndarray] = []
        for i, sdata in enumerate(self.samples):
            table = sdata.tables[self._table_name]
            columns = None
            if all(k in table.var_names and k not in table.obs for k in keys):
                columns = _gene_indices(table.var_names, keys)
            img = rasterize_table(
                sdata,
                bins=self._bins,
                table_name=self._table_name,
                col_key=self._col_key,
                row_key=self._row_key,
                value_key=keys if columns is None else None,
                return_region_as_labels=False,
            )
            if columns is not None:
                img = img.isel(c=columns)
            arr = np.asarray(img.data if hasattr(img, "data") else img, dtype=np.float64)
            if arr.ndim == 2:
                # Single key → rasterize_bins drops the leading axis. Re-insert.
                arr = arr[None, :, :]
            if arr.ndim != 3:
                raise ValueError(
                    f"sample {i} covariate raster has shape {arr.shape}; "
                    "expected (n_keys, ny, nx) from rasterize_bins."
                )
            out.append(
                self._covariate_features_from_array(
                    _mean_fill_missing(arr, axis=(1, 2)), sample_index=i
                )
            )
        return out


def _resolve_spatialdata_gene_names(
    samples: list[Any],
    gene_names: Sequence[str] | None,
    table_name: str,
) -> list[str]:
    first = samples[0]
    if table_name not in first.tables:
        raise KeyError(
            f"sample 0 has no table '{table_name}'; available: {list(first.tables.keys())}."
        )
    if gene_names is None:
        gene_names = list(first.tables[table_name].var_names)
    for i, s in enumerate(samples):
        if table_name not in s.tables:
            raise KeyError(f"sample {i} has no table '{table_name}'.")
        tbl_names = list(s.tables[table_name].var_names)
        if tbl_names != list(gene_names):
            raise ValueError(f"sample {i}'s table has var_names that do not match the reference.")
    return list(gene_names)
