"""
:class:`ComparatorGrid` — cross-sample pattern comparison on a list of
:class:`spatialdata.SpatialData` (regular rasterized bins, FFT
backend).
"""

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
    """
    Cross-sample pattern comparison on regular bins via FFT + SpatialData.

    Accepts a list of :class:`spatialdata.SpatialData` (one per sample). For
    each sample, :func:`spatialdata.rasterize_bins` turns the designated bin
    shape + table into a dense ``(n_genes, ny, nx)`` image, which is then fed
    to the batched 2D FFT. All samples are expected to share the same
    rasterization schema (``bins`` / ``table_name`` / ``col_key`` / ``row_key``
    / ``value_key``) — this mirrors :class:`~sonic.DetectorGrid`.

    Parameters
    ----------
    samples : sequence of :class:`spatialdata.SpatialData`
    bins : str
        SpatialElement key for the bin shapes in each ``sdata``.
    table_name : str
        Table key in each ``sdata.tables``.
    col_key, row_key : str
        Column / row-index columns in the table's ``.obs``.
    value_key : str, optional
        Expression column in ``.obs``; defaults to ``None`` (rasterizes counts
        / presence directly off ``.X``).
    gene_names : sequence of str, optional
        If None, inferred from the first sample's table. All samples must
        share ``var_names``.
    feature_mode : {'radial', '2d'}, default 'radial'
        2D features are adaptive, area-weighted radial/angular band means.
    n_radial_bins : int, default 30
        Number of radial frequency intervals used when ``freq_edges`` is not
        supplied. Radial mode excludes only the DC cell from the first interval,
        so the automatic radial feature count is ``n_radial_bins``.
    n_theta_bins : int, default 36
        Number of angular bins on the half-plane ``[0, π)`` for
        ``feature_mode='2d'``. Ignored by radial mode. The 2D feature
        count is ``n_radius_bins * n_theta_bins``, where
        ``n_radius_bins = len(freq_edges) - 1`` (or ``n_radial_bins`` when
        edges are auto-generated).
    fft_chunk_size : int or 'auto', default 'auto'
        Genes per chunk. The rasterised image is always kept **lazy** (dask)
        and materialised one ``chunk``-gene block at a time, FFT'd, and
        immediately reduced (radial-binned in ``feature_mode='radial'``;
        rotated + physical-frequency polar-resampled in ``feature_mode='2d'``).
        Peak memory is ``O(chunk · ny · nx · 8 B)`` and the full ``(n_genes, ny,
        nx)`` raster / 2D spectra are *never* held in either mode.
        ``'auto'`` sizes the chunk from the (lazily-known) lattice shapes via
        :func:`sonic.utils.resolve_chunk_size` — the FFT cache sweet-spot
        cap (32) capped further by the live-memory budget.
        In 2d mode the rotation is learned from a streamed cross-gene geometric-mean
        landmark by default, or from an explicit ``landmark_genes`` set passed
        to :meth:`compute_spectra`.
    workers : int, 'auto', or None, default 'auto'
        FFT threads per sample job. Automatic threads use spare CPUs, capped
        at four; ``None`` selects one. Balanced against ``compute_spectra`` jobs.
    spacing : (dy, dx) or sequence of (dy, dx), optional
        Physical pitch of one rasterised bin. ``rasterize_bins`` emits one
        pixel per bin, so this maps the pixel lattice to physical frequency
        units (for example cycles/μm when spacing is in μm).
        Pass a single ``(dy, dx)`` to apply to all samples, or one ``(dy, dx)``
        per sample (length ``n_samples``) so that sections with different bin pitches
        still bin onto a common physical frequency grid.
        Defaults to ``(1.0, 1.0)`` (cycles per raster bin).
    adaptive_freq_edges : bool, default False
        If True, collapse radial bins that are empty for any sample's Fourier
        grid by merging each empty bin into its right neighbour. Otherwise,
        empty bins will be dropped across all samples. Regardless of the setting,
        the retained final bins will always have support in every sample.
        The fitted comparator records per-sample bin occupancy for the original
        requested grid in ``frequency_bin_counts_``,
        the pre-adaptation edge grid in ``freq_edges_requested_``, and
        feature-axis-aligned intervals in ``frequency_bin_intervals_``.

    Other Parameters
    ----------------
    fft_solver, workers, freq_edges, presence_threshold
        See :class:`_ComparatorBase`.

    Notes
    -----
    The comparator carries no design / contrast state — supply the
    cross-sample contrast directly to :meth:`test_diff_freq` /
    :meth:`test_diff_expr`. A single fitted comparator can therefore
    serve any number of unrelated comparisons on the same spectra.
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
        """Lazy-rasterize one sample using ``spatialdata.rasterize_bins``.

        ``spatialdata.rasterize_bins`` returns a dask-backed ``xarray.DataArray``
        with a named gene coordinate ``c`` and one-gene-per-block chunking. We
        deliberately do **not** materialise the full ``(n_genes, ny, nx)`` spectra.
        Instead the lazy ``DataArray`` is returned so :meth:`_compute_spectra`
        can pull one **gene-name batch** at a time via
        ``img.sel(c=batch).values`` (label-based, so it also handles any
        ``gene_names`` reordering — same idiom as ``DetectorGrid``'s
        ``_qstat_worker_fft``). Returns the lazy ``(n_genes, ny, nx)``
        ``DataArray``.
        """
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

    def _batch_raster(self, img: Any, names: Sequence[str]) -> np.ndarray:
        """Materialise the ``(len(names), ny, nx)`` raster block for ``names``.

        Label-based selection off the lazy ``DataArray`` (``img.sel(c=names)``),
        so only this gene batch is pulled into memory and the result follows
        :attr:`gene_names` order regardless of the raster's own gene order —
        the same idiom as ``DetectorGrid._qstat_worker_fft``.
        """
        return np.asarray(img.sel(c=list(names)).values, dtype=np.float64)

    def _sample_spectrum_chunker(self, i: int, *, gene_indices=None):
        """Stream selected genes' powers and collect their DC/presence in the same pass."""
        img = self._rasterize_one(self.samples[i])
        names = (
            self.gene_names if gene_indices is None else [self.gene_names[j] for j in gene_indices]
        )
        n_genes = len(names)
        dc = np.empty(n_genes, dtype=float)
        presence = np.empty(n_genes, dtype=bool)

        def _spec_chunk(start: int, stop: int) -> np.ndarray:
            block = self._batch_raster(img, names[start:stop])
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
        """Forward ``keys`` to :func:`spatialdata.rasterize_bins` as
        ``value_key`` and turn each per-sample raster into covariate features.

        ``keys`` may name any combination of ``.obs`` columns and
        ``var_names`` in the comparator's ``table_name`` table —
        ``rasterize_bins`` accepts both. Each sample's resulting
        ``(n_keys, ny, nx)`` raster is then funneled through the same
        spectrum + radial-binning pipeline as the gene panel via
        :meth:`_covariate_features_from_array`.
        """
        from sonic._rasterize import rasterize_table

        keys = list(keys)
        out: list[np.ndarray] = []
        for i, sdata in enumerate(self.samples):
            img = rasterize_table(
                sdata,
                bins=self._bins,
                table_name=self._table_name,
                col_key=self._col_key,
                row_key=self._row_key,
                value_key=keys,
                return_region_as_labels=False,
            )
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
