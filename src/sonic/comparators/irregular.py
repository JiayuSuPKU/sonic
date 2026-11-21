"""Compare spectra from irregular locations stored in :class:`anndata.AnnData`."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import anndata as _ad
import numpy as np
import scipy.sparse as sp

from sonic.comparators.base import (
    _ComparatorBase,
    _validate_common,
)
from sonic.comparators.binning import (
    _bin_radial_spectrum,
    _stream_polar_features,
)
from sonic.comparators.features import _center_observations
from sonic.utils import _NUFFT_CHUNK_CAP, _take_feature_columns

__all__ = ["ComparatorIrregular"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ComparatorIrregular — AnnData / irregular spots
# ---------------------------------------------------------------------------


class ComparatorIrregular(_ComparatorBase):
    """Compare spatial power spectra at irregular sample coordinates.

    See :doc:`/comparator_api` for methods and return values.

    Parameters
    ----------
    samples : sequence of anndata.AnnData
        One object per sample. Each expression matrix has locations in rows
        and genes in columns.
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
    obsm_key : str, default 'spatial'
        Key in ``obsm`` for the two spatial coordinates per location.
    layer : str, optional
        Expression layer in each sample. The default uses ``X``.
    unit_scales : sequence of float, optional
        One coordinate multiplier per sample to convert to common physical units.
    grid_shape : pair of ints or sequence of pairs, optional
        Grid height and width for the nonuniform Fourier transform (NUFFT).
        Supply both ``grid_shape`` and ``spacing`` for a manual grid. Otherwise,
        SONIC infers both from each sample's coordinates.
    spacing : pair of floats or sequence of pairs, optional
        Grid spacing in the scaled coordinate units, in ``(dy, dx)`` order.
        For both grid options, supply one pair for all samples or one per sample.
    freq_edges : np.ndarray, optional
        Shared radial bin edges in cycles per spatial unit. The automatic edges
        cover the frequency range common to all samples.
    adaptive_freq_edges : bool, default False
        Merge adjacent radial intervals when a sample lacks frequency support.
        By default, omit unsupported radial intervals. The retained intervals
        are available in ``frequency_bin_intervals_``.
    eps : float, default 1e-6
        Requested NUFFT accuracy.
    presence_threshold : float, default 0.0
        Minimum fraction of spots with nonzero expression for a gene
        to count as present in a sample.
    nufft_chunk_size : int or 'auto', default 'auto'
        Genes per transform block. The automatic size uses the memory budget.
    workers : int, 'auto', or None, default 'auto'
        Threads per sample for the NUFFT. Use ``None`` for one thread or
        ``'auto'`` to share available CPUs with concurrent sample jobs.
    """

    # NUFFT cache sweet-spot cap for nufft_chunk_size='auto' (utils.auto_chunk_size).
    _auto_chunk_cap: int = _NUFFT_CHUNK_CAP
    _spectrum_backend = "nufft"

    def __init__(  # noqa: C901 — flat per-arg config assembly + per-sample grid setup
        self,
        samples: Sequence[Any],
        gene_names: Sequence[str] | None = None,
        *,
        feature_mode: str = "radial",
        n_radial_bins: int = 30,
        n_theta_bins: int = 36,
        obsm_key: str = "spatial",
        layer: str | None = None,
        unit_scales: Sequence[float] | None = None,
        grid_shape: tuple[int, int] | Sequence[tuple[int, int]] | None = None,
        spacing: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
        freq_edges: np.ndarray | None = None,
        adaptive_freq_edges: bool = False,
        eps: float = 1e-6,
        presence_threshold: float = 0.0,
        nufft_chunk_size: int | str = "auto",
        workers: int | str | None = "auto",
    ) -> None:
        fft_solver = _validate_common(feature_mode, "fft2", presence_threshold)
        samples_list = list(samples)
        if len(samples_list) == 0:
            raise ValueError("samples must be a non-empty list.")
        for i, s in enumerate(samples_list):
            if not isinstance(s, _ad.AnnData):
                raise TypeError(f"sample {i} is {type(s).__name__}, expected anndata.AnnData.")

        resolved = _resolve_anndata_gene_names(samples_list, gene_names, layer=layer)

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
        # Resolve 'auto' at execution, when both grids and outer jobs are known.
        self._nufft_chunk_size_spec = self._normalize_chunk_spec(nufft_chunk_size)
        self._nufft_chunk_size = (
            self._auto_chunk_cap
            if self._nufft_chunk_size_spec == "auto"
            else (self._nufft_chunk_size_spec)
        )
        # NUFFT always produces full-2D layout (fft2), regardless of user's
        # ``fft_solver`` (which is moot here).
        self._spectrum_fft_solver = "fft2"

        self._layer = layer
        self._obsm_key = obsm_key
        self._nufft_eps = float(eps)

        # Per-sample coords / grids.
        from sonic.kernels.nufft import _infer_grid_from_coords

        if unit_scales is None:
            unit_scales = [1.0] * len(samples_list)
        if len(unit_scales) != len(samples_list):
            raise ValueError(
                f"unit_scales length {len(unit_scales)} does not match "
                f"n_samples={len(samples_list)}."
            )
        self._unit_scales: list[float] = [float(s) for s in unit_scales]

        # ``grid_shape`` / ``spacing`` may be a single pair (applied to every
        # sample) or one pair per sample, so heterogeneous samples can be pinned
        # to grids whose frequencies share the same physical units. The manual
        # override is taken only when BOTH are supplied; otherwise each sample's
        # k-grid is inferred from its (unit-scaled) coords.
        grid_override = self._broadcast_pairs(
            grid_shape, len(samples_list), name="grid_shape", cast=int
        )
        spacing_override = self._broadcast_pairs(
            spacing, len(samples_list), name="spacing", cast=float
        )

        coords_list: list[np.ndarray] = []
        grids: list[tuple[int, int]] = []
        spacings: list[tuple[float, float]] = []
        for i, ad_s in enumerate(samples_list):
            if obsm_key not in ad_s.obsm:
                raise KeyError(
                    f"sample {i} has no obsm['{obsm_key}']; available: {list(ad_s.obsm.keys())}."
                )
            c = np.asarray(ad_s.obsm[obsm_key], dtype=np.float64)
            if c.ndim != 2 or c.shape[1] != 2:
                raise ValueError(f"sample {i} obsm['{obsm_key}'] must be (n, 2), got {c.shape}.")
            coords_list.append(c)
            if grid_override is not None and spacing_override is not None:
                gs_i = grid_override[i]
                sp_i = spacing_override[i]
            else:
                gs_i, sp_i = _infer_grid_from_coords(c * self._unit_scales[i], oversample=2.0)
            grids.append(gs_i)
            spacings.append(sp_i)
        self._coords = coords_list
        self._grid_shapes = grids
        self._spacings = spacings

    def _sample_spectrum_chunker(self, i: int, *, gene_indices=None):
        """Stream selected genes' powers, collecting DC/presence only for each batch."""
        from sonic.kernels.nufft import power_spectrum_2d_nufft

        adata = self.samples[i]
        X_src = adata.X if self._layer is None else adata.layers[self._layer]
        X_src = X_src.tocsc() if sp.issparse(X_src) else X_src
        columns = (
            np.arange(len(self.gene_names)) if gene_indices is None else np.asarray(gene_indices)
        )
        n_genes = len(columns)
        n_spots = X_src.shape[0]
        grid_i = self._grid_shapes[i]
        dc = np.empty(n_genes, dtype=float)
        presence = np.empty(n_genes, dtype=bool)
        padding = int(self.feature_mode == "2d")

        def _spec_chunk(start: int, stop: int) -> np.ndarray:
            values = _take_feature_columns(X_src, columns[start:stop])
            block = np.array(
                values.toarray() if sp.issparse(values) else values, dtype=float, copy=True
            )
            dc[start:stop] = block.mean(axis=0)
            presence[start:stop] = (
                np.count_nonzero(block, axis=0) / max(n_spots, 1)
            ) >= self._presence_threshold
            _center_observations(block)
            power = power_spectrum_2d_nufft(
                self._coords[i],
                block,
                grid_shape=grid_i,
                spacing=self._spacings[i],
                unit_scale=self._unit_scales[i],
                eps=self._nufft_eps,
                nthreads=self._workers,
                center_coords=True,
                frequency_padding=padding,
            )
            power /= n_spots**2
            return np.moveaxis(power, -1, 0)

        return _spec_chunk, dc, presence, n_genes, grid_i

    # ------------------------------------------------------------------
    def _covariate_features_from_keys(  # noqa: C901 — per-key obs/var dispatch + per-sample loop
        self, keys: Sequence[str]
    ) -> list[np.ndarray]:
        """Per-spot column lookup → per-sample covariate features.

        Each ``key`` is resolved against the first sample as either:

        - an ``adata.obs`` column (per-spot scalar — typical for
          deconvolution outputs, region labels, depth proxies); or
        - an entry in ``adata.var_names`` (treats that gene's
          per-spot expression as the covariate — useful for
          regressing out a housekeeping gene's spatial pattern).

        Resolution prefers ``obs`` when a name appears in both. Every
        subsequent sample must resolve each key to the same source as
        the first sample (i.e., a key is "obs everywhere" or "var
        everywhere") — anything else is treated as a schema mismatch.

        For each sample the resolved per-spot vectors are stacked into
        an ``(n_obs, n_covariates)`` block, mean-centered per column,
        and NUFFTed directly onto the sample's k-grid. The 2-D spectra
        are then radial-binned with the same edges as the gene panel.

        Raises
        ------
        KeyError
            If a key is missing from both ``adata.obs.columns`` and
            ``adata.var_names`` in any sample, or if a key resolves to
            different sources across samples.
        ValueError
            If an obs column cannot be cast to float (e.g., string
            categoricals — encode them first).
        """
        from sonic.kernels.nufft import power_spectrum_2d_nufft

        keys = list(keys)
        # Classify each key once against the first sample; require all
        # later samples to agree on the source.
        first = self.samples[0]
        sources: dict[str, str] = {}
        for k in keys:
            in_obs = k in first.obs.columns
            in_var = k in first.var_names
            if not (in_obs or in_var):
                raise KeyError(
                    f"covariate key {k!r} is in neither obs.columns nor "
                    f"var_names of sample 0. Available obs (first 10): "
                    f"{list(first.obs.columns)[:10]}; available var_names "
                    f"(first 10): {list(first.var_names)[:10]}."
                )
            sources[k] = "obs" if in_obs else "var"

        out: list[np.ndarray] = []
        for i, adata in enumerate(self.samples):
            cols: list[np.ndarray] = []
            for k in keys:
                src = sources[k]
                if src == "obs":
                    if k not in adata.obs.columns:
                        raise KeyError(
                            f"sample {i} resolves covariate {k!r} differently from "
                            f"sample 0 (sample 0 → obs, sample {i} → not in obs)."
                        )
                    try:
                        cols.append(np.asarray(adata.obs[k], dtype=np.float64))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"sample {i} obs[{k!r}] cannot be cast to float "
                            f"({type(exc).__name__}); encode categoricals before passing."
                        ) from exc
                else:  # var_names path
                    if k not in adata.var_names:
                        raise KeyError(
                            f"sample {i} resolves covariate {k!r} differently from "
                            f"sample 0 (sample 0 → var_names, sample {i} → not in "
                            "var_names)."
                        )
                    idx = adata.var_names.get_loc(k)
                    X_src = adata.X if self._layer is None else adata.layers[self._layer]
                    col = X_src[:, idx]
                    if sp.issparse(col):
                        col = col.toarray()
                    cols.append(np.asarray(col, dtype=np.float64).ravel())
            block = np.column_stack(cols)  # (n_obs, n_cov)
            # Mean-centre each column, matching the gene panel's per-column DC removal.
            block = block - block.mean(axis=0, keepdims=True)

            p = power_spectrum_2d_nufft(
                self._coords[i],
                block,
                grid_shape=self._grid_shapes[i],
                spacing=self._spacings[i],
                unit_scale=self._unit_scales[i],
                eps=self._nufft_eps,
                nthreads=self._workers,
                center_coords=True,
                frequency_padding=1 if self.feature_mode == "2d" else 0,
            )
            p /= block.shape[0] ** 2
            # power_spectrum_2d_nufft returns (ny, nx, M) for multi-column values.
            cov_2d = np.moveaxis(p, -1, 0)  # (n_cov, ny, nx)
            ny, nx = self._grid_shapes[i]
            if self.feature_mode == "radial":
                cov_feat = _bin_radial_spectrum(
                    cov_2d,
                    grid_shape=(ny, nx),
                    n_bins=self._n_radial_bins,
                    fft_solver=self._spectrum_fft_solver,
                    spacing=self._spacings[i],
                    edges=self.freq_edges,
                )
            else:
                if self.freq_edges is None:
                    raise RuntimeError("2D covariate features require .compute_spectra() first.")
                angle = 0.0 if self.rotation_angles_ is None else float(self.rotation_angles_[i])

                def _cov_chunk(start: int, stop: int, _cov_2d: np.ndarray = cov_2d) -> np.ndarray:
                    return _cov_2d[start:stop]

                cov_feat = _stream_polar_features(
                    _cov_chunk,
                    cov_2d.shape[0],
                    (ny, nx),
                    angle_deg=angle,
                    chunk_size=max(1, cov_2d.shape[0]),
                    edges=self.freq_edges,
                    spacing=self._spacings[i],
                    n_theta=self._n_theta_bins,
                    fft_solver=self._spectrum_fft_solver,
                    frequency_padding=1,
                )
            out.append(cov_feat)
        return out


# ---------------------------------------------------------------------------
# Gene-name resolution helpers
# ---------------------------------------------------------------------------


def _resolve_anndata_gene_names(
    samples: list[Any],
    gene_names: Sequence[str] | None,
    *,
    layer: str | None,
) -> list[str]:
    first = samples[0]
    if gene_names is None:
        gene_names = list(first.var_names)
    for i, s in enumerate(samples):
        if list(s.var_names) != list(gene_names):
            raise ValueError(
                f"sample {i} has var_names that do not match the reference "
                "(all AnnData samples must share the same gene axis)."
            )
        if layer is not None and layer not in s.layers:
            raise KeyError(f"sample {i} is missing layer '{layer}'.")
    return list(gene_names)
