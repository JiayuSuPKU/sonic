"""Shared methods for the regular-grid and irregular-coordinate comparators.

Fit spectra once, then test the contrasts of interest. See
:doc:`/guides/multisample` for examples and calibration assumptions."""

from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from sonic.comparators.binning import (
    _adapt_radial_edges,
    _bin_radial_spectrum,
    _polar_reducer_workspace_bytes,
    _radial_bin_cell_counts,
    _stream_polar_features,
    _stream_radial_features,
)
from sonic.utils import _DEFAULT_CHUNK_BUDGET, _FFT_CHUNK_CAP, _gene_indices, _parse_memory_budget

# Suppress known deprecation warnings from SpatialData dependencies BEFORE importing them.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*legacy Dask DataFrame.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

from sonic.comparators._inference import (
    _estimate_glm_masked_null_covariance,
    _estimate_glm_null_covariance,
    _estimate_two_group_masked_null_covariance,
    _estimate_two_group_null_covariance,
    _resolve_covariance,
)
from sonic.comparators.features import (
    _ALIGNMENT_N_RADIUS,
    _ALIGNMENT_N_THETA,
    _landmark_angles,
    _landmark_cross_sum,
    _stream_landmark_polar,
    compute_sample_spectrum,
)
from sonic.comparators.features import (
    gene_pattern_diversity as _gene_pattern_diversity,
)
from sonic.comparators.multisample import (
    compare_glm,
    compare_glm_masked,
    compare_glm_scalar,
    compare_two_groups,
    compare_two_groups_masked,
    compare_two_groups_scalar,
)

# Aliased to leading-underscore names to avoid shadowing the like-named
# instance methods on the comparator class below.
from sonic.comparators.normalization import (
    normalize_background as _normalize_background,
)
from sonic.comparators.normalization import (
    normalize_covariates as _normalize_covariates,
)

if TYPE_CHECKING:
    from sonic.comparators._inference import _PluginNullState
    from sonic.comparators.cross_spectrum import CrossSpectra
    from sonic.explanation import SpatialExplanation

__all__: list[str] = []

logger = logging.getLogger(__name__)

# Helper functions for running per-sample computations.


def _run_per_sample(
    worker: Any,
    n_samples_total: int,
    *,
    n_chunks_per_sample: int,
    desc: str,
    n_jobs: int,
    progress: bool,
    shared_output: bool = False,
) -> list[Any]:
    """Preserve sample order and require shared memory for workers that fill caller-owned arrays."""
    out: list[Any] = [None] * n_samples_total

    run_sequential = progress or n_jobs == 1
    if run_sequential:
        n_total = n_samples_total * n_chunks_per_sample
        pbar: tqdm | None = tqdm(total=n_total, desc=desc) if progress else None
        for i in range(n_samples_total):
            if pbar is not None:
                pbar.set_postfix_str(f"sample {i + 1}/{n_samples_total}")
            out[i] = worker(i, pbar)
        if pbar is not None:
            pbar.close()
    else:
        constraints = {"require": "sharedmem"} if shared_output else {}
        results = Parallel(n_jobs=n_jobs, prefer="threads", **constraints)(
            delayed(worker)(i, None) for i in range(n_samples_total)
        )
        for i, r in enumerate(results):
            out[i] = r

    return out


def _unpack_sample_triples(
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Convert ``[(features, dc, presence), ...]`` to comparator return values."""
    feats, dc_list, pres_list = zip(*results, strict=True)
    dc = np.stack([np.asarray(x) for x in dc_list], axis=0)
    presence = np.stack([np.asarray(x) for x in pres_list], axis=0)
    return [np.asarray(x) for x in feats], dc, presence


class _ComparatorBase:
    """Common interface for gene and gene-pair comparisons across samples.

    Attributes
    ----------
    samples : list
        Input samples in design row order.
    gene_names : list of str
        Gene names in input column order.
    feature_mode : {'radial', '2d'}
        Radial bins describe spatial scale. Directional bins also describe angle.
    freq_edges : np.ndarray or None
        Shared radial frequency edges in cycles per spatial unit.
    spectra_ : np.ndarray or None
        Power spectra with axes (sample, gene, bin). In directional mode, bin
        indices run by radius and then by angle.
    dc_ : np.ndarray or None
        Mean expression for each sample and gene.
    presence_ : np.ndarray or None
        Boolean mask identifying genes that meet the sample's presence threshold.
    rotation_angles_ : np.ndarray or None
        Sample alignment angles in degrees for directional spectra.
    freq_edges_requested_ : np.ndarray or None
        Requested radial frequency edges before bin selection.
    frequency_bin_intervals_ : np.ndarray or None
        Lower and upper frequency bounds for each retained radial interval.
    frequency_bin_counts_ : np.ndarray or None
        Number of Fourier cells per sample and requested radial bin.
    cross_spectra_ : CrossSpectra or None
        Most recent result from ``compute_cross_spectra()``.
    """

    # --- public attribute stubs populated by subclass __init__ --------
    samples: list[Any]
    gene_names: list[str]
    feature_mode: str
    freq_edges: np.ndarray | None

    # --- private attribute stubs populated by subclass __init__ -------
    _n_radial_bins: int
    _n_theta_bins: int
    _fft_solver: str
    _workers: int | None
    _workers_spec: int | str | None
    _spectrum_backend: str = "fft"
    _presence_threshold: float
    _spacings: list[tuple[float, float]] | None
    _spectrum_fft_solver: str
    _grid_shapes: list[tuple[int, int]]

    # --- populated by :meth:`compute_spectra` ---------------------------
    spectra_: np.ndarray | None = None
    cross_spectra_: CrossSpectra | None = None
    dc_: np.ndarray | None = None
    presence_: np.ndarray | None = None
    rotation_angles_: np.ndarray | None = None
    _alignment_landmark_genes: tuple[str, ...] | None = None
    frequency_bin_counts_: np.ndarray | None = None
    freq_edges_requested_: np.ndarray | None = None
    frequency_bin_intervals_: np.ndarray | None = None
    _supported_frequency_bin_mask: np.ndarray | None = None

    # Angular resolution of the ``feature_mode='2d'`` polar feature grid:
    # the 2-D feature axis is ``n_radius_bins × _n_theta_bins`` over the
    # half-plane ``[0, π)``. Subclasses may override.
    _n_theta_bins: int = 36

    # ------------------------------------------------------------------
    def _prepare_frequency_grid(self) -> None:
        """Resolve common geometry and bin support without computing gene spectra."""
        if self._spectrum_backend == "fft":
            self._grid_shapes = [tuple(self._rasterize_one(s).shape[-2:]) for s in self.samples]
            self._spacings = self._spacing_override or [(1.0, 1.0)] * len(self.samples)
        self._resolve_freq_edges()
        self._record_frequency_bin_metadata(
            self._fft_cell_counts_for_edges(self.freq_edges),
            self._fft_cell_counts_for_edges(self.freq_edges_requested_),
        )

    def _ensure_alignment(self, n_jobs, progress, landmark_genes=None, *, chunk_size=None):
        """Reuse sample angles or estimate them from compact polar landmarks only.

        This helper populates :attr:`rotation_angles_` and :attr:`_alignment_landmark_genes`
        to be used by :meth:`_compute_spectra` when ``feature_mode='2d'``.
        """
        if self.feature_mode != "2d":  # only applies to 2D polar bins
            return
        names = None if landmark_genes is None else tuple(landmark_genes)
        indices = None
        if names is not None:
            if not names:
                raise ValueError("landmark_genes must contain at least one gene")
            available = set(self.gene_names)
            missing = [g for g in names if g not in available]
            if missing:
                raise KeyError(f"landmark_genes not in gene_names: {missing}")
            indices = _gene_indices(self.gene_names, names)
        if self.rotation_angles_ is not None and (
            names is None or names == self._alignment_landmark_genes
        ):
            return
        if len(self.samples) == 1:
            angles = np.zeros(1)
        else:
            padding = int(self._spectrum_backend == "nufft")
            grids = [(ny + 2 * padding, nx + 2 * padding) for ny, nx in self._grid_shapes]
            spec = self._resolve_chunk_spec(chunk_size)
            n_genes = len(self.gene_names) if names is None else len(names)
            n_jobs, chunk, spare = self._resolve_spectrum_schedule(
                spec,
                grids,
                n_genes,
                n_jobs=n_jobs,
                alignment="geomean" if names is None else "per_gene",
            )
            f_max = min(0.5 / max(spacing) for spacing in self._spacings)
            edges = np.linspace(0.0, f_max, _ALIGNMENT_N_RADIUS + 1)

            def fourier(i, selected):
                read, _, _, count, shape = self._sample_spectrum_chunker(
                    i, gene_indices=selected, chunk_size=chunk, cache_budget=spare[i]
                )
                polar = _stream_landmark_polar(
                    read,
                    count,
                    shape,
                    chunk_size=chunk,
                    fft_solver=self._spectrum_fft_solver,
                    freq_edges=edges,
                    spacing=self._spacings[i],
                    geomean=selected is None,
                    frequency_padding=padding,
                )
                return np.fft.rfft(polar, axis=1)

            # The reference block is shared. Only angular cross-products survive each batch.
            cross = np.zeros((len(self.samples), _ALIGNMENT_N_THETA // 2 + 1), dtype=complex)

            def correlate(i, _pbar, *, reference, selected):
                cross[i + 1] += _landmark_cross_sum(reference, fourier(i + 1, selected))

            starts = (0,) if indices is None else range(0, n_genes, chunk)
            for start in tqdm(starts, desc="Alignment landmarks", disable=not progress):
                selected = None if indices is None else indices[start : start + chunk]
                _run_per_sample(
                    partial(correlate, reference=fourier(0, selected), selected=selected),
                    len(self.samples) - 1,
                    n_chunks_per_sample=1,
                    desc="Alignment landmarks",
                    n_jobs=n_jobs,
                    progress=False,
                    shared_output=True,
                )
            angles = _landmark_angles(cross, _ALIGNMENT_N_THETA)
        self.rotation_angles_ = angles
        self._alignment_landmark_genes = names

    def _compute_spectra(self, n_jobs, progress, landmark_genes=None, *, chunk_size=None):
        """Stream raw feature powers and DC/presence through either backend."""
        self._prepare_frequency_grid()
        padding = int(self.feature_mode == "2d" and self._spectrum_backend == "nufft")
        grids = [(ny + 2 * padding, nx + 2 * padding) for ny, nx in self._grid_shapes]
        spec = self._resolve_chunk_spec(chunk_size)
        self._ensure_alignment(n_jobs, progress, landmark_genes, chunk_size=spec)
        n_jobs, chunk, spare = self._resolve_spectrum_schedule(
            spec, grids, len(self.gene_names), n_jobs=n_jobs, polar=self.feature_mode == "2d"
        )
        setattr(self, f"_{self._spectrum_backend}_chunk_size", chunk)

        # Compute binned and flattened single-feature auto-spectra per sample.
        def _one(i, pbar):
            read, dc, presence, count, shape = self._sample_spectrum_chunker(
                i, chunk_size=chunk, cache_budget=spare[i]
            )
            options = {
                "chunk_size": chunk,
                "fft_solver": self._spectrum_fft_solver,
                "spacing": self._spacings[i],
                "edges": self.freq_edges,
                "pbar": pbar,
            }
            if self.feature_mode == "radial":
                features = _stream_radial_features(read, count, shape, **options)
            else:
                features = _stream_polar_features(
                    read,
                    count,
                    shape,
                    angle_deg=float(self.rotation_angles_[i]),
                    n_theta=self._n_theta_bins,
                    frequency_padding=padding,
                    **options,
                )
            return features, dc, presence

        return _unpack_sample_triples(
            _run_per_sample(
                _one,
                len(self.samples),
                n_chunks_per_sample=int(np.ceil(len(self.gene_names) / chunk)),
                desc="Spectra",
                n_jobs=n_jobs,
                progress=progress,
            )
        )

    # Backend cache sweet-spot cap for ``chunk_size='auto'`` — the empirically
    # tuned caps from :func:`sonic.utils.auto_chunk_size` (FFT → 32,
    # NUFFT → 32). Subclasses override. The live-memory budget (across all
    # workers) is :attr:`_auto_chunk_budget_bytes`.
    _auto_chunk_cap: int = _FFT_CHUNK_CAP
    _auto_chunk_budget_bytes: int = _DEFAULT_CHUNK_BUDGET

    @staticmethod
    def _normalize_chunk_spec(spec: int | str) -> int | str:
        """Accept only 'auto' or a positive integer, at construction and per call."""
        if isinstance(spec, str) and spec == "auto":
            return spec
        if (
            isinstance(spec, (int, np.integer))
            and not isinstance(spec, (bool, np.bool_))
            and spec >= 1
        ):
            return int(spec)
        raise ValueError("chunk_size must be auto or a positive integer")

    def _resolve_chunk_spec(self, chunk_size):
        """Validate a call override without changing the constructor setting."""
        spec = (
            getattr(self, f"_{self._spectrum_backend}_chunk_size_spec")
            if chunk_size is None
            else chunk_size
        )
        return self._normalize_chunk_spec(spec)

    @staticmethod
    def _normalize_n_theta_bins(n_theta_bins: int) -> int:
        """Validate the angular bin count used by ``feature_mode='2d'``."""
        n_theta = int(n_theta_bins)
        if n_theta < 1:
            raise ValueError(f"n_theta_bins must be a positive int, got {n_theta_bins!r}.")
        return n_theta

    @staticmethod
    def _broadcast_pairs(
        value: Any, n_samples: int, *, name: str, cast: Any
    ) -> list[tuple[Any, Any]] | None:
        """Normalise a single ``(a, b)`` pair *or* a per-sample sequence of pairs.

        Returns a length-``n_samples`` list of ``(cast(a), cast(b))`` tuples, or
        ``None`` when ``value`` is ``None``. A single ``(a, b)`` (1-D, length 2)
        is broadcast to every sample; an ``(n_samples, 2)`` sequence is taken
        per-sample. This lets ``spacing`` / ``grid_shape`` be set globally or
        adjusted per sample so the resulting grids — and thus frequencies — are
        in the same physical units across heterogeneous samples.
        """
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.shape[0] == 2:
            return [(cast(arr[0]), cast(arr[1]))] * n_samples
        if arr.ndim == 2 and arr.shape[1] == 2:
            if arr.shape[0] != n_samples:
                raise ValueError(
                    f"per-sample {name} has {arr.shape[0]} rows but n_samples={n_samples}."
                )
            return [(cast(r[0]), cast(r[1])) for r in arr]
        raise ValueError(
            f"{name} must be a 2-tuple (dy, dx) or an (n_samples, 2) sequence of them; "
            f"got shape {tuple(arr.shape)}."
        )

    # ------------------------------------------------------------------
    def _resolve_spectrum_schedule(
        self,
        spec: int | str,
        grid_shapes: list[tuple[int, int]],
        n_genes: int,
        *,
        n_jobs: int = 1,
        polar: bool = False,
        alignment: str | None = None,
    ) -> tuple[int, int, list[int]]:
        """Size concurrent transforms and return the spare bytes for sparse column caches."""
        from bisect import bisect_right

        from sonic.utils import _nufft_workspace_bytes, resolve_parallelism

        reducers = [
            _polar_reducer_workspace_bytes(
                self._grid_shapes[i],
                edges=self.freq_edges,
                spacing=self._spacings[i],
                n_theta=self._n_theta_bins,
            )
            if polar
            else (0, 0)
            for i in range(len(grid_shapes))
        ]

        def reference_bytes(count):
            if alignment is None:
                return 0
            landmarks = count if alignment == "per_gene" else 1
            return 16 * landmarks * _ALIGNMENT_N_RADIUS * (_ALIGNMENT_N_THETA // 2 + 1)

        def workspace(i, count, workers):
            shape = grid_shapes[i]
            if self._spectrum_backend == "nufft":
                transform = _nufft_workspace_bytes(
                    shape, len(self._coords[i]), self._nufft_eps, count, workers
                )
            else:
                transform = 64 * shape[0] * shape[1] * count
            if alignment is not None:
                landmarks = count if alignment == "per_gene" else 1
                polar_bytes = 8 * _ALIGNMENT_N_THETA * _ALIGNMENT_N_RADIUS
                # Interpolation retains Cartesian powers, coordinates and one polar block.
                interpolate = 24 * shape[0] * shape[1] * landmarks + polar_bytes * (landmarks + 2)
                return max(transform, interpolate, polar_bytes * landmarks + reference_bytes(count))
            stored, peak = reducers[i]
            return max(peak, stored + transform)

        minimum = max(workspace(i, 1, 1) for i in range(len(grid_shapes)))
        n_jobs, self._workers = resolve_parallelism(
            n_jobs,
            self._workers_spec,
            backend=self._spectrum_backend,
            n_tasks=len(self.samples) - int(alignment is not None),
            per_feat_bytes=minimum if spec == "auto" else None,
            budget_bytes=max(1, self._auto_chunk_budget_bytes - reference_bytes(1)),
        )

        def total_bytes(count):
            return reference_bytes(count) + n_jobs * max(
                workspace(i, count, self._workers) for i in range(len(grid_shapes))
            )

        chunk = spec
        if spec == "auto":
            if total_bytes(1) > self._auto_chunk_budget_bytes:
                raise ValueError(
                    f"Budget cannot fit one feature per worker ({total_bytes(1)} bytes across {n_jobs} workers)."
                )
            chunk = bisect_right(
                range(1, self._auto_chunk_cap + 1), self._auto_chunk_budget_bytes, key=total_bytes
            )
        count = min(chunk, n_genes)
        per_job = (self._auto_chunk_budget_bytes - reference_bytes(count)) // n_jobs
        spare = [
            max(0, per_job - workspace(i, count, self._workers)) for i in range(len(grid_shapes))
        ]
        return n_jobs, chunk, spare

    # ------------------------------------------------------------------
    def _resolve_freq_edges(self) -> None:
        """Populate :attr:`freq_edges` (shared physical-frequency grid), if unset.

        Uses the per-sample physical ``self._spacings`` (already known before
        the spectra pass for both backends) to build a common bin grid on
        ``[0, min Nyquist]``. Used by **both** feature modes: the radial path
        bins onto these edges, and the 2-D path resamples its polar radius axis
        onto the same edges so heterogeneous lattices stay cross-sample aligned.
        Idempotent and a no-op when ``freq_edges`` is already set or cannot yet
        be derived.
        """
        if not self._spacings:
            return

        # Request linearly spaced frequency edges if no explicit edges are provided.
        if self.freq_edges is None:
            # Find the minimum Nyquist frequency across all samples.
            nyquists = [1.0 / (2.0 * max(dy, dx)) for (dy, dx) in self._spacings]
            f_max = float(min(nyquists))
            # Create the shared frequency edges.
            self.freq_edges = np.linspace(0.0, f_max * (1.0 + 1e-9), self._n_radial_bins + 1)
            logger.info(
                "Auto-generated %d radial bins on [0, %.4g] cycles per unit length.",
                self._n_radial_bins,
                f_max,
            )

        # For diagnostic purposes, keep a copy of the requested frequency edges.
        if self.freq_edges_requested_ is None and self.freq_edges is not None:
            self.freq_edges_requested_ = np.asarray(self.freq_edges, dtype=float).copy()

        # Optionally, merge empty radial bins to ensure bins are supported in all samples.
        if getattr(self, "_adaptive_freq_edges", False):
            self._adapt_freq_edges()

    def _fft_cell_counts_for_edges(self, edges: np.ndarray | None = None) -> np.ndarray | None:
        """Return per-sample FFT-cell counts for a radial-frequency edge grid."""
        if edges is None:
            edges = self.freq_edges
        if edges is None or not self._grid_shapes or not self._spacings:
            return None
        return np.vstack(
            [
                _radial_bin_cell_counts(
                    shape,
                    n_bins=self._n_radial_bins,
                    fft_solver=self._spectrum_fft_solver,
                    spacing=spacing,
                    edges=edges,
                )
                for shape, spacing in zip(self._grid_shapes, self._spacings, strict=True)
            ]
        )

    def _adapt_freq_edges(self) -> None:
        """Merge adjacent radial bins until every retained bin has sample support."""
        if self.freq_edges is None:
            return
        if not self._grid_shapes or not self._spacings:
            return

        edges = np.asarray(self.freq_edges, dtype=float)
        requested = edges.size - 1
        self.freq_edges = _adapt_radial_edges(
            self._grid_shapes,
            self._spacings,
            edges,
            fft_solver=self._spectrum_fft_solver,
        )
        self._n_radial_bins = int(self.freq_edges.size - 1)
        collapsed = requested - self._n_radial_bins
        if collapsed:
            logger.info(
                "Adaptive frequency edges collapsed %d empty radial bins "
                "(requested=%d, actual=%d).",
                collapsed,
                requested,
                self._n_radial_bins,
            )

    def _select_supported_frequency_bins(
        self, features: np.ndarray, supported_bins: np.ndarray
    ) -> np.ndarray:
        """Validate the feature axis and retain supported radial bands."""
        supported_bins = np.asarray(supported_bins, dtype=bool)
        n_radius = supported_bins.size
        n_theta = 1 if self.feature_mode == "radial" else self._n_theta_bins
        expected = n_radius * n_theta
        if features.shape[-1] != expected:
            raise ValueError(
                f"{self.feature_mode} feature axis has length {features.shape[-1]}, "
                f"expected {expected} bins before support filtering"
            )
        if supported_bins.all():
            return features
        if self.feature_mode == "radial":
            return features[..., supported_bins]

        reshaped = features.reshape(*features.shape[:-1], n_radius, n_theta)
        kept = reshaped[..., supported_bins, :]
        return kept.reshape(*features.shape[:-1], int(supported_bins.sum()) * n_theta)

    def _record_frequency_bin_metadata(
        self,
        feature_grid_counts: np.ndarray | None,
        requested_grid_counts: np.ndarray | None,
    ) -> np.ndarray | None:
        """Record bin diagnostics and return the feature-grid support mask."""
        if feature_grid_counts is None:
            self.frequency_bin_counts_ = None  # 1D
            self.frequency_bin_intervals_ = None  # 2D, (start, end) per retained bin
            self._supported_frequency_bin_mask = None  # 1D
            return None

        # Identify frequency bins in the current grid that are supported by all samples.
        supported_bins = (feature_grid_counts > 0).all(axis=0)
        if self.feature_mode == "2d":
            nyquist = min(0.5 / max(spacing) for spacing in self._spacings)
            supported_bins &= np.asarray(self.freq_edges[1:]) <= nyquist * (1 + 2e-9)
        if not np.any(supported_bins):
            raise RuntimeError("No radial frequency bins are supported by every sample.")

        self._supported_frequency_bin_mask = supported_bins.copy()
        self.frequency_bin_counts_ = (
            requested_grid_counts if requested_grid_counts is not None else feature_grid_counts
        )

        # Record the start and end of each retained frequency bin.
        if self.freq_edges is not None:
            intervals = np.column_stack(
                [np.asarray(self.freq_edges[:-1]), np.asarray(self.freq_edges[1:])]
            )
            self.frequency_bin_intervals_ = intervals[supported_bins]
        else:
            self.frequency_bin_intervals_ = None

        dropped = int((~supported_bins).sum())
        if dropped:
            logger.info(
                "Dropped %d radial bins with missing sample support (retained=%d).",
                dropped,
                int(supported_bins.sum()),
            )
        self._n_radial_bins = int(supported_bins.sum())
        return supported_bins

    # ------------------------------------------------------------------
    def compute_spectra(
        self,
        n_jobs: int = -1,
        landmark_genes: Sequence[str] | None = None,
        progress: bool = True,
        *,
        memory_budget_bytes: int | str = _DEFAULT_CHUNK_BUDGET,
        chunk_size: int | str | None = None,
    ) -> _ComparatorBase:
        """Compute a power spectrum for each gene in each sample.

        Parameters
        ----------
        n_jobs : int, default -1
            Number of samples to process in parallel. Use ``-1`` for automatic
            allocation. Set ``progress=False`` to enable parallel sample processing.
        landmark_genes : sequence of str, optional
            Genes that define sample alignment in directional mode. The default
            reuses an existing alignment or estimates one from all genes.
        progress : bool, default True
            Show a progress bar and process one sample at a time.
        memory_budget_bytes : int or str, default '2 GiB'
            Estimated workspace for automatic transform blocks and directional binning
            across sample jobs. Supply bytes or a size such as ``'512 MiB'``.
        chunk_size : int, 'auto', or None, optional
            Genes per transform block. By default, use the constructor setting.
            An explicit size overrides the automatic memory budget.

        Returns
        -------
        self
            Stores binned powers in ``spectra_``, sample mean expression in ``dc_``,
            and gene presence in ``presence_``. Radial bins describe spatial scale.
            Directional bins also describe angle.

        Notes
        -----
        Recomputing gene spectra clears any previously computed cross-spectra.
        """
        chunk_size = self._resolve_chunk_spec(chunk_size)
        # Fit on a shallow copy: old arrays are never mutated by extraction.
        pending = copy.copy(self)
        pending._auto_chunk_budget_bytes = _parse_memory_budget(memory_budget_bytes)
        logger.info(
            "Computing per-sample spectra (n_samples=%d, mean-centered)...",
            len(pending.samples),
        )
        pending.cross_spectra_ = None
        per_sample, pending.dc_, pending.presence_ = pending._compute_spectra(
            n_jobs=1 if progress else n_jobs,
            progress=progress,
            landmark_genes=landmark_genes,
            chunk_size=chunk_size,
        )

        supported_bins = pending._supported_frequency_bin_mask

        # Drop unsupported radial bands consistently across all samples.
        feats = list(per_sample)
        if supported_bins is not None:
            feats = [pending._select_supported_frequency_bins(f, supported_bins) for f in feats]
        pending.spectra_ = np.stack(feats, axis=0)
        if hasattr(self, "_nufft_eps") and len(pending.samples) > 1:
            # Diagnostic only: eps is a transform target, not a statistical
            # variance floor. Keep small real effects and p-values unchanged.
            spread = np.ptp(pending.spectra_, axis=0).max(axis=-1)
            scale = pending.spectra_.max(axis=(0, 2))
            unresolved = (spread > 0) & (spread <= 2 * pending._nufft_eps * scale)
            if pending.presence_ is not None:
                unresolved &= pending.presence_.all(axis=0)
            if unresolved.any():
                warnings.warn(
                    f"{unresolved.sum()} feature(s) have between-sample spectral differences "
                    "below the requested NUFFT accuracy. With negligible replicate variance, "
                    "analytic p-values may reflect transform error. Recompute with a smaller "
                    "eps and check effect-size stability before interpreting significance.",
                    UserWarning,
                    stacklevel=2,
                )
        self.__dict__.update(pending.__dict__)
        return self

    # ------------------------------------------------------------------
    def compute_cross_spectra(
        self,
        pairs: Sequence[tuple[str, str]],
        n_jobs: int = -1,
        landmark_genes: Sequence[str] | None = None,
        progress: bool = True,
        *,
        memory_budget_bytes: int | str = _DEFAULT_CHUNK_BUDGET,
        chunk_size: int | str | None = None,
    ) -> _ComparatorBase:
        """Compute cross-spectra for selected gene pairs on a common frequency grid.

        Parameters
        ----------
        pairs : sequence of (str, str)
            Ordered gene pairs. Each name must identify one gene. Supply each
            pair once, in one order. Self-pairs are available for description.
        n_jobs : int, default -1
            Number of samples to process in parallel. Use ``-1`` for automatic
            allocation. Set ``progress=False`` to enable parallel sample processing.
        landmark_genes : sequence of str, optional
            Genes that define sample alignment in directional mode. The default
            reuses an existing alignment or estimates one from all genes.
        progress : bool, default True
            Show a progress bar and process one sample at a time.
        memory_budget_bytes : int or str, default '2 GiB'
            Estimated workspace for transform blocks and polar reducers across sample
            jobs when chunking automatically. Supply bytes or a size such as ``'512 MiB'``.
        chunk_size : int, 'auto', or None, optional
            Genes per transform block. By default, use the constructor setting.
            An explicit size overrides the automatic memory budget.

        Returns
        -------
        self
            Stores the requested pairs in ``cross_spectra_``, replacing the previous
            set. Cross-spectra have axes (sample, pair, bin). The matching power
            spectra have axes (sample, gene, bin).

        Notes
        -----
        Pair spectra use the original expression values, even after single-gene
        normalization. Both kinds of spectra use the same directional alignment.
        To change landmarks after fitting gene spectra, call ``compute_spectra()`` first.
        """
        from sonic.comparators.cross_spectrum import _compute_cross_spectra

        pending = copy.copy(self)
        pending.cross_spectra_ = _compute_cross_spectra(
            pending,
            pairs,
            n_jobs=n_jobs,
            progress=progress,
            memory_budget_bytes=memory_budget_bytes,
            chunk_size=chunk_size,
            landmark_genes=landmark_genes,
        )
        self.__dict__.update(pending.__dict__)
        return self

    # ------------------------------------------------------------------
    def explain_spectrum(
        self,
        features: str | Sequence[str],
        *,
        sample_index: int = 0,
        bands: Sequence[int],
        representation: str | None = None,
        memory_budget_bytes: int | str = _DEFAULT_CHUNK_BUDGET,
    ) -> SpatialExplanation:
        """Map each location's contribution to selected frequency bins in one sample.

        Parameters
        ----------
        features : str or sequence of str
            One gene for a power map, or an ordered pair for a cross-spectrum map.
        sample_index : int, default 0
            Zero-based sample index in this comparator.
        bands : sequence of int
            Distinct zero-based indices of retained bins, in the desired output
            order. Directional bins use radius first, then angle.
        representation : str, optional
            For one gene, use ``'power'`` (default) or ``'power_norm'`` for its
            share of total power. For a pair, use ``'cross'`` (default, real
            cross-spectrum), ``'coherency'`` (signed real part), or ``'coherence'``
            (squared magnitude).
        memory_budget_bytes : int or str, default '2 GiB'
            Memory budget for reconstruction, including the returned maps.

        Returns
        -------
        sonic.explanation.SpatialExplanation
            One map per selected bin. Local contributions sum to the bin value
            on the selected representation's original scale. Individual
            contributions can be negative, including for coherence.

        Notes
        -----
        Compute gene spectra or cross-spectra first. Maps use the source expression
        values. Irregular samples retain their source coordinates. Grid maps use
        local coordinates with origin zero and the specified grid spacing.
        """
        from sonic.comparators.explanation import _explain_spectrum

        return _explain_spectrum(
            self,
            features,
            sample_index=sample_index,
            bands=bands,
            representation=representation,
            memory_budget_bytes=memory_budget_bytes,
        )

    # ------------------------------------------------------------------
    def subset(self, indices: Sequence[int] | np.ndarray) -> _ComparatorBase:
        """Select samples from this comparator for another comparison.

        Parameters
        ----------
        indices : sequence of int or np.ndarray
            Sample indices in the desired order, or a Boolean mask in sample order.

        Returns
        -------
        ComparatorIrregular or ComparatorGrid
            A comparator with the selected samples and their available fitted
            results. It shares the source sample objects with this comparator.
        """
        idx = np.asarray(indices)
        if idx.dtype == bool:
            if idx.ndim != 1 or idx.size != len(self.samples):
                raise IndexError(
                    f"boolean sample mask must have length {len(self.samples)}, got {idx.size}"
                )
            idx = np.flatnonzero(idx)
        else:
            idx = idx.astype(int, copy=False)
        idx = np.ravel(idx)

        sub = copy.copy(self)
        sub.samples = [self.samples[i] for i in idx]
        sub.spectra_ = None if self.spectra_ is None else self.spectra_[idx]
        sub.dc_ = None if self.dc_ is None else self.dc_[idx]
        sub.presence_ = None if self.presence_ is None else self.presence_[idx]
        sub.rotation_angles_ = None if self.rotation_angles_ is None else self.rotation_angles_[idx]
        if self.cross_spectra_ is not None:
            result = self.cross_spectra_
            sub.cross_spectra_ = replace(
                result,
                cross_spectra=result.cross_spectra[idx],
                auto_spectra=result.auto_spectra[idx],
                coherency_valid=result.coherency_valid[idx],
                presence=result.presence[idx],
                mode_pair_counts=result.mode_pair_counts[idx],
                rotation_angles_deg=result.rotation_angles_deg[idx],
                chunk_schedules=[result.chunk_schedules[i] for i in idx],
            )
        sub.frequency_bin_counts_ = (
            None if self.frequency_bin_counts_ is None else self.frequency_bin_counts_[idx]
        )
        sub.freq_edges_requested_ = self.freq_edges_requested_
        sub.frequency_bin_intervals_ = self.frequency_bin_intervals_
        sub._supported_frequency_bin_mask = self._supported_frequency_bin_mask
        for name in (
            "_spacings",
            "_grid_shapes",
            "_coords",
            "_unit_scales",
            "_spacing_override",
        ):
            values = getattr(self, name, None)
            if values is not None and len(values) == len(self.samples):
                setattr(sub, name, [values[i] for i in idx])
        return sub

    # ------------------------------------------------------------------
    # Post-fit transforms
    # ------------------------------------------------------------------
    def normalize_background(self) -> _ComparatorBase:
        """Adjust power in each frequency bin for the sample-wide background.

        Returns
        -------
        self
            Updates ``spectra_`` using the geometric mean power of the observed
            genes in each bin. Only observed genes enter the adjustment.
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .compute_spectra() before .normalize_background().")
        self.spectra_ = np.asarray(self.spectra_, dtype=np.float64)
        for i in range(self.spectra_.shape[0]):
            observed = slice(None) if self.presence_ is None else self.presence_[i]
            if self.presence_ is None or observed.any():
                self.spectra_[i, observed] = _normalize_background(self.spectra_[i, observed])
        return self

    def normalize_covariates(self, covariates: Sequence[Any]) -> _ComparatorBase:  # noqa: C901
        """Adjust gene power for the spectra of known spatial covariates in each sample.

        Parameters
        ----------
        covariates : sequence of str or sequence of np.ndarray
            Names of spatial covariates shared by the samples. On irregular
            coordinates, SONIC checks ``obs`` first and then ``var_names``. On a
            grid, use keys from the expression table. Alternatively, supply one
            array of shape ``(n_covariates, ny, nx)`` per sample on a grid that
            supports its frequency bins.

        Returns
        -------
        self
            Replaces ``spectra_`` with the exponentiated residuals from regressing
            each gene's log power on the log powers of the spatial covariates.
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .compute_spectra() before .normalize_covariates().")
        items = list(covariates)
        if len(items) == 0:
            raise ValueError("covariates must be a non-empty sequence.")

        first = items[0]
        if isinstance(first, str):
            # Shared key-list mode.
            if not all(isinstance(k, str) for k in items):
                raise TypeError(
                    "Mixed str and non-str entries in `covariates=` — pass either a "
                    "list of column-name strings (shared across samples) or a list "
                    "of per-sample (n_cov, ny, nx) arrays."
                )
            cov_features_per_sample = self._covariate_features_from_keys(items)
            covariate_mode = "key"
        elif isinstance(first, np.ndarray):
            if len(items) != len(self.samples):
                raise ValueError(
                    f"covariates length {len(items)} != n_samples {len(self.samples)}."
                )
            cov_features_per_sample = [
                self._covariate_features_from_array(arr, sample_index=i)
                for i, arr in enumerate(items)
            ]
            covariate_mode = "array"
        else:
            raise TypeError(
                f"covariates[0] is {type(first).__name__}; expected str (column-name "
                "mode) or np.ndarray (per-sample image-array mode)."
            )

        if len(cov_features_per_sample) != len(self.samples):
            raise ValueError(
                f"Subclass returned {len(cov_features_per_sample)} covariate-feature "
                f"sets but the comparator has {len(self.samples)} samples."
            )
        for i, cov_feat in enumerate(cov_features_per_sample):
            if self._supported_frequency_bin_mask is not None:
                # Keep only the frequency bins that are supported by all samples.
                cov_feat = self._select_supported_frequency_bins(
                    cov_feat,
                    self._supported_frequency_bin_mask,
                )
            self._validate_covariate_features(cov_feat, sample_index=i, mode=covariate_mode)
            cov_features_per_sample[i] = cov_feat
        # Keep the original spectra intact if a later sample has a saturated design.
        normalized = np.empty(self.spectra_.shape, dtype=np.float64)
        for i, cov_feat in enumerate(cov_features_per_sample):
            normalized[i] = _normalize_covariates(self.spectra_[i], cov_feat)
        self.spectra_ = normalized
        return self

    def _validate_covariate_features(
        self, cov_feat: np.ndarray, *, sample_index: int, mode: str
    ) -> None:
        """Validate per-sample covariate features before log-space regression."""
        if self.spectra_ is None:
            raise RuntimeError("Call .compute_spectra() before .normalize_covariates().")
        if cov_feat.shape[-1] != self.spectra_.shape[-1]:
            raise ValueError(
                f"sample {sample_index} covariate features have {cov_feat.shape[-1]} "
                f"bins after frequency-bin filtering, expected {self.spectra_.shape[-1]}."
            )
        if not np.isfinite(cov_feat).all():
            mode_label = "pre-rasterized array" if mode == "array" else "key-based"
            raise ValueError(
                f"sample {sample_index} {mode_label} covariate features contain "
                "non-finite values after frequency-bin filtering. This usually means "
                "the covariate grid lacks Fourier support for at least one retained "
                "bin; use covariates on a compatible grid, reduce n_radial_bins, or "
                "enable adaptive_freq_edges."
            )

    # ------------------------------------------------------------------
    def _covariate_features_from_array(self, cov: np.ndarray, sample_index: int) -> np.ndarray:
        """Image-array → ``(n_covariates, n_feature_bins)`` covariate features.

        Shared by both subclasses for the pre-rasterized
        ``(n_covariates, ny, nx)`` input mode. Mirrors the spectrum +
        radial-binning pipeline used on the gene panel itself. The
        sample index is needed only to look up that sample's physical
        ``spacing`` for radial binning.
        """
        if cov.ndim != 3:
            raise ValueError(f"covariate array must be 3D (n_cov, ny, nx), got {cov.shape}.")
        cov_2d = compute_sample_spectrum(
            cov, fft_solver=self._spectrum_fft_solver, workers=self._workers
        )
        # Use the covariate's own raster shape — for the NUFFT path the
        # sample's internal k-grid (self._grid_shapes[i]) is auto-inferred
        # and may differ from the covariate raster. ``freq_edges`` (shared
        # across samples when ``feature_mode='radial'``) is what aligns the
        # bins, not grid_shape.
        cov_shape = cov.shape[-2:]
        spacing = self._spacings[sample_index] if self._spacings is not None else (1.0, 1.0)
        if self.feature_mode == "radial":
            return _bin_radial_spectrum(
                cov_2d,
                grid_shape=cov_shape,
                n_bins=self._n_radial_bins,
                fft_solver=self._spectrum_fft_solver,
                spacing=spacing,
                edges=self.freq_edges,
            )
        if self.freq_edges is None:
            raise RuntimeError("2D covariate features require .compute_spectra() first.")
        angles = self.rotation_angles_
        angle = 0.0 if angles is None else float(angles[sample_index])

        def _cov_chunk(start: int, stop: int) -> np.ndarray:
            return cov_2d[start:stop]

        return _stream_polar_features(
            _cov_chunk,
            cov_2d.shape[0],
            cov_shape,
            angle_deg=angle,
            chunk_size=max(1, cov_2d.shape[0]),
            edges=self.freq_edges,
            spacing=spacing,
            n_theta=self._n_theta_bins,
            fft_solver=self._spectrum_fft_solver,
        )

    def _covariate_features_from_keys(self, keys: Sequence[str]) -> list[np.ndarray]:
        """Column-key list → per-sample ``(n_covariates, n_feature_bins)`` features.

        Subclass hook. ``ComparatorIrregular`` reads ``adata.obs[key]``
        per sample (NUFFT directly on per-spot values), and
        ``ComparatorGrid`` forwards the keys as ``value_key`` to
        :func:`spatialdata.rasterize_bins`.

        The base class raises so the user gets a clear error if the
        subclass doesn't implement it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support column-name covariate input; "
            "pass per-sample (n_cov, ny, nx) arrays instead."
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_diff_freq(  # noqa: C901 — dispatcher over three execution paths
        self,
        design: Any,
        *,
        contrast: str | dict[str, float] | np.ndarray | None = None,
        statistic: str = "log_l2",
        null: str | None = None,
        n_perm: int = 1000,
        random_state: int | None = None,
        freq_weights: np.ndarray | None = None,
        max_exact_permutations: int = 10000,
        normalize_shape: bool = False,
        min_samples_per_group: int = 2,
        min_resid_df: int = 1,
        covariance: str | None = None,
        integration_tol: float = 1e-6,
    ) -> Any:
        """Compare gene power across spatial frequencies in the fitted samples.

        Parameters
        ----------
        design : array-like, pandas.DataFrame, or 2-D np.ndarray
            One row per sample, in fitted sample order. Use one-dimensional labels
            for two groups. Use a design matrix and ``contrast`` for a linear model.
            SONIC adds an intercept to a data frame. Include any intercept in a
            numeric matrix yourself.
        contrast : str, dict, or np.ndarray, optional
            Required for a linear model. Name one coefficient, map coefficient names
            to weights, or supply a vector in encoded design-matrix column order.
            The contrast must be estimable from the design.
        statistic : {'log_l2', 'log_t_cauchy'}, default 'log_l2'
            L2 measures the size of the contrast across bins. ``'log_t_cauchy'``
            combines bin p-values and identifies the bin with the smallest
            p-value among those used in the combination.
        null : str, optional
            For L2, use ``'rotation'`` (default) or ``'plugin'`` for Gaussian
            plug-in calibration. Use ``'permutation'`` for two exchangeable groups
            with one-dimensional labels and no ``contrast``.
            For ``'log_t_cauchy'``, the only calibration is ``'analytic'`` (default).
        n_perm : int, default 1000
            Number of random group assignments for a sampled permutation test.
        random_state : int, optional
            Random seed for sampled permutations.
        freq_weights : np.ndarray, optional
            Positive, finite weights in fitted bin order.
            The default gives each bin equal weight. Choose weights before testing.
        max_exact_permutations : int, default 10000
            Enumerate all group assignments when their number is at most this
            limit. Set it to zero to sample assignments.
        normalize_shape : bool, default False
            Divide each gene's power in each bin by its total power in that
            sample before taking the logarithm (DF-norm).
        min_samples_per_group : int, default 2
            Minimum observed samples in each group for binary L2 tests and
            masked binary t-Cauchy tests.
        min_resid_df : int, default 1
            Minimum residual degrees of freedom for features with missing samples.
            Fully observed features require one residual degree of freedom.
        covariance : str, optional
            With plug-in calibration, ``None`` or ``'per_gene'`` estimates residual
            covariance across bins for each gene. ``'pooled'`` shares this estimate
            across eligible genes. Setting a covariance selects plug-in calibration
            if you omit ``null``. For other calibrations, leave this as ``None``.
        integration_tol : float, default 1e-6
            Target absolute error for numerical integration of rotation p-values.

        Returns
        -------
        pandas.DataFrame
            One row per gene, sorted by decreasing ``Statistic``. ``Feature`` identifies
            the gene. ``P_value`` tests its contrast. ``P_adj`` gives the
            Benjamini-Hochberg adjustment across tested genes.

            ``Estimate_per_bin`` gives the first sorted group's mean minus the second
            group's mean, or the specified linear contrast.
            Estimates use natural-log power or natural-log power share in fitted bin order.
            L2 reports the weighted root mean square of these estimates.

            For t-Cauchy, ``Statistic`` is the largest absolute t statistic.
            ``T_per_bin`` and ``P_value_per_bin`` give signed t statistics and
            unadjusted two-sided p-values. ``Best_bin`` is the zero-based index
            of the smallest p-value among bins used in the combination.

            For rotation, ``P_value_method`` identifies the calculation.
            ``P_value_error`` gives its numerical error estimate or budget, when available.
            Masked comparisons add ``n_obs_A`` and ``n_obs_B`` for binary groups,
            or ``n_obs`` and ``df_resid`` for a linear model.

        Raises
        ------
        RuntimeError
            If spectra are unavailable or the rotation p-value cannot be evaluated.
        ValueError
            If the input or test options are invalid.
        NotImplementedError
            If a linear model requests label permutation.

        Notes
        -----
        Binary labels select binwise Welch t tests. A linear model uses ordinary
        least squares t tests. See :doc:`/guides/multisample` for calibration
        assumptions and interpretation.
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .compute_spectra() before .test_diff_freq().")
        if int(min_samples_per_group) < 2:
            raise ValueError(f"min_samples_per_group must be >= 2, got {min_samples_per_group}.")
        if int(min_resid_df) < 1:
            raise ValueError(f"min_resid_df must be >= 1, got {min_resid_df}.")
        groups, design_obj = _validate_design(design, len(self.samples))
        use_masked = self.presence_ is not None and not self.presence_.all()

        options = {
            "gene_names": self.gene_names,
            "statistic": statistic,
            "freq_weights": freq_weights,
            "normalize_shape": normalize_shape,
            "null": null,
            "covariance": covariance,
            "integration_tol": integration_tol,
        }

        use_glm = (contrast is not None) or (groups is None)
        if use_glm:
            if statistic not in ("log_l2", "log_t_cauchy"):
                raise ValueError(
                    "Only statistic='log_l2' or 'log_t_cauchy' is supported when "
                    "contrast= is provided or `design` is a multi-column / "
                    f"continuous design, got statistic={statistic!r}."
                )
            if statistic == "log_l2" and null == "permutation":
                raise NotImplementedError(
                    "Only null='plugin' or 'rotation' is supported for L2 when "
                    "contrast= is provided or `design` is a multi-column / "
                    "continuous design. For label permutations, pass a 1-D "
                    "binary `design` without `contrast=`."
                )
            if contrast is None:
                raise ValueError(
                    "test_diff_freq() requires `contrast=` when `design` is "
                    "a multi-column / continuous design."
                )
            if use_masked:
                return compare_glm_masked(
                    self.spectra_,
                    design_obj,
                    contrast,
                    self.presence_,
                    min_resid_df=int(min_resid_df),
                    **options,
                )
            return compare_glm(
                self.spectra_,
                design_obj,
                contrast,
                **options,
            )

        # Binary path (1-D design, contrast is None).
        options["random_state"] = random_state
        if use_masked:
            return compare_two_groups_masked(
                self.spectra_,
                groups,
                self.presence_,
                n_perm=n_perm,
                min_samples_per_group=int(min_samples_per_group),
                max_exact_permutations=max_exact_permutations,
                **options,
            )
        return compare_two_groups(
            self.spectra_,
            groups,
            min_samples_per_group=int(min_samples_per_group),
            n_perm=n_perm,
            max_exact_permutations=max_exact_permutations,
            **options,
        )

    def test_cross_diff_freq(
        self,
        design: Any,
        *,
        contrast: str | dict[str, float] | np.ndarray | None = None,
        representation: str = "coherence",
        statistic: str = "l2",
        null: str | None = None,
        n_perm: int = 1000,
        random_state: int | None = None,
        freq_weights: np.ndarray | None = None,
        max_exact_permutations: int = 10000,
        min_samples_per_group: int = 2,
        min_resid_df: int = 1,
        covariance: str | None = None,
        integration_tol: float = 1e-6,
    ):
        """Compare the spatial coupling of each gene pair across samples.

        Parameters
        ----------
        design : array-like, pandas.DataFrame, or 2-D np.ndarray
            One row per sample, in fitted sample order. Use one-dimensional labels
            for two groups. Use a design matrix and ``contrast`` for a linear model.
            SONIC adds an intercept to a data frame. Include any intercept in a
            numeric matrix yourself.
        contrast : str, dict, or np.ndarray, optional
            Required for a linear model. Name one coefficient, map coefficient names
            to weights, or supply a vector in encoded design-matrix column order.
            The contrast must be estimable from the design.
        representation : {'coherence', 'coherency'}, default 'coherence'
            ``'coherence'`` tests the natural logarithm of squared-magnitude coherency.
            ``'coherency'`` tests the signed real part of coherency.
        statistic : {'l2', 't_cauchy'}, default 'l2'
            L2 measures the size of the contrast across bins. ``'t_cauchy'``
            combines bin p-values and identifies the bin with the smallest
            p-value among those used in the combination.
        null : str, optional
            For L2, use ``'rotation'`` (default) or ``'plugin'`` for Gaussian
            plug-in calibration. Use ``'permutation'`` for two exchangeable groups
            with one-dimensional labels and no ``contrast``.
            For ``'t_cauchy'``, the only calibration is ``'analytic'`` (default).
        n_perm : int, default 1000
            Number of random group assignments for a sampled permutation test.
        random_state : int, optional
            Random seed for sampled permutations.
        freq_weights : np.ndarray, optional
            Positive, finite weights in original bin order (radius, then angle).
            The default gives each bin equal weight. Choose weights before testing.
        max_exact_permutations : int, default 10000
            Enumerate all group assignments when their number is at most this
            limit. Set it to zero to sample assignments.
        min_samples_per_group : int, default 2
            Minimum observed samples in each group for binary L2 tests and
            masked binary t-Cauchy tests.
        min_resid_df : int, default 1
            Minimum residual degrees of freedom for features with missing samples.
            Fully observed features require one residual degree of freedom.
        covariance : str, optional
            With plug-in calibration, ``None`` or ``'per_pair'`` estimates residual
            covariance across bins for each pair. ``'pooled'`` shares this estimate
            across eligible pairs. Setting a covariance selects plug-in calibration
            if you omit ``null``. For other calibrations, leave this as ``None``.
        integration_tol : float, default 1e-6
            Target absolute error for numerical integration of rotation p-values.

        Returns
        -------
        pandas.DataFrame
            One row per pair, sorted by decreasing ``Statistic``. ``Feature_1`` and
            ``Feature_2`` identify the pair. ``P_value`` tests its contrast. ``P_adj`` gives
            the Benjamini-Hochberg adjustment across tested pairs.

            ``Estimate_per_bin`` gives the first sorted group's mean minus the second
            group's mean, or the specified linear contrast.
            Estimates use natural-log coherence or real coherency in original bin order.
            L2 reports the weighted root mean square of these estimates.

            For t-Cauchy, ``Statistic`` is the largest absolute t statistic.
            ``T_per_bin`` and ``P_value_per_bin`` give signed t statistics and
            unadjusted two-sided p-values. ``Best_bin`` is the zero-based index
            of the smallest p-value among bins used in the combination.

            For rotation, ``P_value_method`` identifies the calculation.
            ``P_value_error`` gives its numerical error estimate or budget, when available.
            ``N_samples`` and ``N_bins`` give the retained counts. ``Status`` indicates
            whether the pair was tested. ``DF_resid`` gives design residual degrees
            of freedom. Binary comparisons also report ``n_obs_A`` and ``n_obs_B``.
            ``N_pairs_pooled`` counts the pairs used to estimate covariance.
            ``Effective_rank`` measures how many directions account for variation
            in the plug-in covariance or conditional rotation distribution.
            Use ``attrs["original_freq_weights"]`` to repeat the bin weighting.

        Raises
        ------
        RuntimeError
            If cross-spectra are unavailable or the rotation p-value cannot be evaluated.
        ValueError
            If the input or test options are invalid.
        NotImplementedError
            If a linear model requests label permutation.

        Notes
        -----
        Each pair uses samples where both genes are present and bins with valid
        coherency in all those samples. Self-pairs are descriptive only.

        Binary labels select binwise Welch t tests. A linear model uses ordinary
        least squares t tests. See :doc:`/guides/multisample` for calibration
        assumptions and interpretation.
        """
        from sonic.comparators.cross_spectrum import _compare_cross_spectra

        if self.cross_spectra_ is None:
            raise RuntimeError("Call .compute_cross_spectra() before .test_cross_diff_freq().")
        groups, design_obj = _validate_design(design, len(self.samples))
        return _compare_cross_spectra(
            self.cross_spectra_,
            design_obj,
            groups=groups,
            contrast=contrast,
            representation=representation,
            statistic=statistic,
            null=null,
            n_perm=n_perm,
            random_state=random_state,
            freq_weights=freq_weights,
            max_exact_permutations=max_exact_permutations,
            min_samples_per_group=min_samples_per_group,
            min_resid_df=min_resid_df,
            covariance=covariance,
            integration_tol=integration_tol,
        )

    def _covariance_columns(self, features, covariance, memory_budget_bytes):
        """Check the two returned covariance arrays before copying or fitting any profiles."""
        _, n_genes, n_bins = self.spectra_.shape
        columns = slice(None)
        if features is not None:
            names = (features,) if isinstance(features, str) else tuple(features)
            if not names or len(set(names)) != len(names):
                raise ValueError("Select at least one feature and supply each name only once.")
            columns = _gene_indices(self.gene_names, names)
            n_genes = len(columns)
        per_gene = covariance == "per_gene"
        required = 2 * np.dtype(np.float64).itemsize * n_bins**2 * (n_genes if per_gene else 1)
        budget = _parse_memory_budget(memory_budget_bytes)
        if required > budget:
            hint = (
                "Select fewer features or use covariance='pooled'" if per_gene else "Use fewer bins"
            )
            raise MemoryError(
                f"The covariance matrices require {required} bytes (budget {budget} bytes). "
                f"{hint}, or increase memory_budget_bytes."
            )
        return columns

    def estimate_null_covariance(
        self,
        design: Any,
        *,
        contrast: str | dict[str, float] | np.ndarray | None = None,
        freq_weights: np.ndarray | None = None,
        normalize_shape: bool = False,
        min_samples_per_group: int = 2,
        min_resid_df: int = 1,
        covariance: str | None = None,
        features: str | Sequence[str] | None = None,
        memory_budget_bytes: int | str = _DEFAULT_CHUNK_BUDGET,
    ) -> _PluginNullState:
        """Estimate residual covariance across bins for Gaussian plug-in calibration.

        Parameters
        ----------
        design : array-like, pandas.DataFrame, or 2-D np.ndarray
            One row per sample, in fitted sample order. Use one-dimensional labels
            for two groups. Use a design matrix and ``contrast`` for a linear model.
            SONIC adds an intercept to a data frame. Include any intercept in a
            numeric matrix yourself.
        contrast : str, dict, or np.ndarray, optional
            Required for a linear model. Name one coefficient, map coefficient names
            to weights, or supply a vector in encoded design-matrix column order.
            The contrast must be estimable from the design.
        freq_weights : np.ndarray, optional
            Positive, finite weights in fitted bin order. The default gives each
            bin equal weight.
        normalize_shape : bool, default False
            Divide each gene's power in each bin by its total power in that
            sample before taking the logarithm (DF-norm).
        min_samples_per_group : int, default 2
            Minimum number of observed samples in each group for a binary fit.
        min_resid_df : int, default 1
            Minimum residual degrees of freedom for features with missing samples.
            Fully observed features require one residual degree of freedom.
        covariance : str, optional
            Use ``'per_gene'`` (default) for separate gene covariance estimates.
            Use ``'pooled'`` for a shared estimate across eligible genes.
        features : str or sequence of str, optional
            Inspect these genes in the requested order. Pooled covariance uses
            only the selected genes. By default, use every gene in input order.
        memory_budget_bytes : int or str, default '2 GiB'
            Maximum combined size of the returned float64 ``sigma_log`` and
            ``weighted_cov`` matrices. Other outputs and fitting workspace are additional.

        Returns
        -------
        dict
            ``sigma_log`` gives the covariance across bins. Its shape is
            ``(n_genes, n_bins, n_bins)`` for separate estimates and
            ``(n_bins, n_bins)`` for a pooled estimate. ``weighted_cov`` includes the bin
            weights. ``effective_rank`` measures how many directions account for its
            variance. ``eigenvalues`` also includes the contrast variance factor.
            ``observed`` gives the L2 statistic and ``eligible`` identifies tested genes.

            ``beta`` contains fitted coefficients in the original predictor units,
            with shape ``(n_terms, n_genes, n_bins)``. For binary labels the two rows
            give the first sorted group's mean and the second mean minus the first.
            For a linear model, ``design_columns`` identifies the coefficient rows.

        Raises
        ------
        MemoryError
            If the covariance matrices would exceed the budget.
        RuntimeError
            Compute spectra before calling this method.
        ValueError
            If the input or test options are invalid.

        Notes
        -----
        See :doc:`/guides/multisample` for the covariance options.
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .compute_spectra() before .estimate_null_covariance().")
        if int(min_samples_per_group) < 2:
            raise ValueError(f"min_samples_per_group must be >= 2, got {min_samples_per_group}.")
        if int(min_resid_df) < 1:
            raise ValueError(f"min_resid_df must be >= 1, got {min_resid_df}.")

        groups, design_obj = _validate_design(design, len(self.samples))
        covariance = _resolve_covariance(covariance)
        columns = self._covariance_columns(features, covariance, memory_budget_bytes)
        spectra = self.spectra_[:, columns]
        presence = self.presence_[:, columns] if self.presence_ is not None else None
        use_masked = presence is not None and not presence.all()
        options = {
            "freq_weights": freq_weights,
            "normalize_shape": normalize_shape,
            "covariance": covariance,
        }

        use_glm = (contrast is not None) or (groups is None)
        if use_glm:
            if contrast is None:
                raise ValueError(
                    "estimate_null_covariance() requires `contrast=` when "
                    "`design` is a multi-column / continuous design."
                )
            if use_masked:
                state = _estimate_glm_masked_null_covariance(
                    spectra,
                    design_obj,
                    contrast,
                    presence,
                    min_resid_df=int(min_resid_df),
                    **options,
                )
            else:
                state = _estimate_glm_null_covariance(spectra, design_obj, contrast, **options)
        elif use_masked:
            state = _estimate_two_group_masked_null_covariance(
                spectra,
                groups,
                presence,
                min_samples_per_group=int(min_samples_per_group),
                **options,
            )
        else:
            state = _estimate_two_group_null_covariance(
                spectra, groups, min_samples_per_group=int(min_samples_per_group), **options
            )
        del state["estimate_per_bin"]
        return state

    def test_diff_expr(
        self,
        design: Any,
        *,
        contrast: str | dict[str, float] | np.ndarray | None = None,
        log_expression: bool = False,
        eps: float = 1e-12,
    ) -> Any:
        """Compare the sample mean expression of each gene.

        Parameters
        ----------
        design : array-like, pandas.DataFrame, or 2-D np.ndarray
            One row per sample, in fitted sample order. Use one-dimensional labels
            for two groups. Use a design matrix and ``contrast`` for a linear model.
            SONIC adds an intercept to a data frame. Include any intercept in a
            numeric matrix yourself.
        contrast : str, dict, or np.ndarray, optional
            Required for a linear model. Name one coefficient, map coefficient names
            to weights, or supply a vector in encoded design-matrix column order.
            The contrast must be estimable from the design.
        log_expression : bool, default False
            Take the natural logarithm of each sample's mean expression plus ``eps``.
            By default, the test uses mean expression on the input scale.
        eps : float, default 1e-12
            Positive offset for ``log_expression=True``.

        Returns
        -------
        pandas.DataFrame
            One row per gene, sorted by decreasing ``Statistic``. ``Feature`` identifies
            the gene. ``P_value`` tests its contrast. ``P_adj`` gives the
            Benjamini-Hochberg adjustment across tested genes.
            ``Statistic`` is the absolute t statistic. ``Mean_diff`` gives the first
            sorted group's mean minus the second group's mean. A linear model
            returns its signed contrast in ``Estimate``.

        Raises
        ------
        RuntimeError
            Compute spectra before calling this method.
        ValueError
            If the input or test options are invalid.

        Notes
        -----
        Binary labels select Welch's t test. A design matrix or an explicit
        contrast selects an ordinary least squares t test.
        """
        if self.dc_ is None:
            raise RuntimeError("Call .compute_spectra() before .test_diff_expr().")
        groups, design_obj = _validate_design(design, len(self.samples))
        use_glm = (contrast is not None) or (groups is None)
        if use_glm:
            if contrast is None:
                raise ValueError(
                    "test_diff_expr() requires `contrast=` when `design` is "
                    "a multi-column / continuous design."
                )
            return compare_glm_scalar(
                self.dc_,
                design_obj,
                contrast,
                gene_names=self.gene_names,
                log_expression=log_expression,
                eps=eps,
            )
        return compare_two_groups_scalar(
            self.dc_,
            groups,
            gene_names=self.gene_names,
            log_expression=log_expression,
            eps=eps,
        )

    def effective_rank(self, *, weights: np.ndarray | None = None) -> np.ndarray:
        """Measure how the variance across genes is distributed across frequency bins.

        Parameters
        ----------
        weights : np.ndarray, optional
            Nonnegative weights in fitted bin order. By default, use the
            unweighted covariance across bins.

        Returns
        -------
        np.ndarray
            One effective rank per sample. For nonzero covariance, the rank is one
            when a single direction accounts for all the variance. It equals the
            number of bins when all covariance eigenvalues are equal.

        Notes
        -----
        The covariance is calculated from log power across genes in this sample.
        The effective rank is the squared sum of its eigenvalues divided by the
        sum of their squares.
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .compute_spectra() before .effective_rank().")
        n_samples = self.spectra_.shape[0]
        return np.array(
            [_gene_pattern_diversity(self.spectra_[i], weights=weights) for i in range(n_samples)]
        )


# ---------------------------------------------------------------------------
# Shared input-validation helpers
# ---------------------------------------------------------------------------


def _validate_design(  # noqa: C901 — dispatch over three input shapes is essential
    design: Any,
    n_samples: int,
) -> tuple[np.ndarray | None, Any]:
    """Validate + normalise the unified ``design`` test-time argument.

    Accepted input forms:

    - **1-D ``np.ndarray`` / ``pd.Series``** — treated as the binary
      contrast. Must contain exactly two distinct labels. Returned
      both as a ``(n_samples,)`` array (the binary-dispatch signal)
      and wrapped in a single-column ``pd.DataFrame({"group": …})``
      so the GLM path can be taken with an explicit ``contrast=`` too.
    - **2-D ``np.ndarray``** — full design matrix of shape
      ``(n_samples, p)``. Used verbatim as the design; binary
      dispatch is disabled.
    - **``pandas.DataFrame``** — passed straight through (patsy
      encoding happens lazily inside ``compare_glm``). Binary
      dispatch disabled.

    Returns
    -------
    groups : np.ndarray | None
        ``(n_samples,)`` 1-D array iff the user supplied a 1-D binary
        input; ``None`` otherwise. Drives the binary two-group dispatch
        in :meth:`test_diff_freq` and :meth:`test_diff_expr`; ``None``
        selects the GLM path.
    design : pandas.DataFrame | np.ndarray
        Normalised design. Always a ``DataFrame`` for the 1-D and
        DataFrame inputs; a 2-D ``ndarray`` is returned as-is.

    Raises
    ------
    ValueError
        Length mismatch, 1-D input without exactly two levels, or
        2-D input with wrong row count.
    TypeError
        ``design`` is not one of the accepted forms.
    """
    if design is None:
        raise ValueError("`design` must be supplied to the test method.")
    # 1-D array-likes → binary groups path.
    arr = np.asarray(design)
    is_1d_array = arr.ndim == 1
    # pandas import is lazy so the array-only path doesn't pay for it.
    try:
        import pandas as _pd
    except ImportError:
        _pd = None
    if _pd is not None and isinstance(design, _pd.Series):
        is_1d_array = True
        arr = design.to_numpy()
    if is_1d_array:
        if arr.shape != (n_samples,):
            raise ValueError(f"1-D design length {arr.shape} does not match n_samples={n_samples}.")
        if np.unique(arr).size != 2:
            raise ValueError(
                "A 1-D design array must contain exactly two distinct labels "
                "(binary contrast). For continuous or multi-column designs, "
                "wrap in a pandas DataFrame: design=pd.DataFrame({'x': arr})."
            )
        if _pd is None:
            raise ImportError(
                "pandas is required to normalise a 1-D design into a "
                "single-column DataFrame; install it or pass a 2-D ndarray."
            )
        return arr, _pd.DataFrame({"group": arr})
    # DataFrame path.
    if _pd is not None and isinstance(design, _pd.DataFrame):
        if len(design) != n_samples:
            raise ValueError(f"design DataFrame length {len(design)} != n_samples={n_samples}.")
        return None, design
    # 2-D ndarray path.
    if isinstance(design, np.ndarray):
        if design.ndim != 2 or design.shape[0] != n_samples:
            raise ValueError(
                f"design ndarray must be (n_samples, p) = ({n_samples}, p), got {design.shape}."
            )
        return None, design
    raise TypeError(
        "design must be a 1-D array/Series (binary contrast), a 2-D ndarray "
        f"(design matrix), or a pandas DataFrame; got {type(design).__name__}."
    )


def _validate_common(
    feature_mode: str,
    fft_solver: str,
    presence_threshold: float,
) -> str:
    if feature_mode not in ("radial", "2d"):
        raise ValueError(f"feature_mode must be 'radial' or '2d', got '{feature_mode}'.")
    if feature_mode == "2d" and fft_solver != "fft2":
        logger.info("feature_mode='2d' works best with fft_solver='fft2'; switching automatically.")
        fft_solver = "fft2"
    if not 0.0 <= float(presence_threshold) <= 1.0:
        raise ValueError(f"presence_threshold must be in [0, 1], got {presence_threshold}.")
    return fft_solver
