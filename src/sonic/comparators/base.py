"""
Shared mixin and input-validation helpers for the comparator layer.

This module hosts the private :class:`_ComparatorBase` mixin that
:class:`~sonic.ComparatorIrregular` and :class:`~sonic.ComparatorGrid`
inherit from. The mixin owns:

- the ``compute_spectra`` driver that turns per-sample 2-D images
  into the ``(n_samples, n_genes, n_feature_bins)`` ``spectra_`` tensor;
- the chainable preprocessing methods ``normalize_background()`` and
  ``normalize_covariates(covariates)`` — thin wrappers around the
  same-named standalone functions in
  :mod:`sonic.comparators.normalization` that mutate ``spectra_`` in
  place;
- the test methods ``test_diff_freq(design, ...)`` and
  ``test_diff_expr(design, ...)`` — design-at-call-time so a single
  fitted comparator can serve any number of unrelated contrasts on
  the same spectra;
- the diagnostic ``effective_rank()``;

- the ``compute_cross_spectra`` driver that computes pairwise cross-spectra
  for given feature pairs and populates the ``cross_spectra_`` attribute.

The shape-only / sum-1 feature representation is reached via the
``normalize_shape: bool = False`` keyword on :meth:`test_diff_freq`
(forwarded to its dispatch target), not via a chainable method — this
keeps the per-test choice non-destructive.

Concrete classes live in sibling modules:
:mod:`sonic.comparators.irregular` and
:mod:`sonic.comparators.grid`.
"""

from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from sonic.comparators.binning import (
    _adapt_radial_edges,
    _bin_radial_spectrum,
    _radial_bin_cell_counts,
    _stream_polar_features,
    _stream_radial_features,
)
from sonic.utils import _DEFAULT_CHUNK_BUDGET, _FFT_CHUNK_CAP, _parse_memory_budget

# Suppress known deprecation warnings from SpatialData dependencies BEFORE importing them.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*legacy Dask DataFrame.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

from sonic.comparators._inference import (
    _estimate_glm_masked_null_covariance,
    _estimate_glm_null_covariance,
    _estimate_two_group_masked_null_covariance,
    _estimate_two_group_null_covariance,
)
from sonic.comparators.features import (
    _ALIGNMENT_N_RADIUS,
    _ALIGNMENT_N_THETA,
    _estimate_rotations_from_polar,
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
    from sonic.comparators._inference import _AnalyticNullState
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
) -> list[Any]:
    """Invoke ``worker(i, pbar)`` for each sample, preserving sample order."""
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
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
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
    """Shared state + shared methods for NUFFT / FFT pattern comparators.

    Subclasses populate the small public surface and the underscored
    internal state listed below in ``__init__``, then implement
    :meth:`_sample_spectrum_chunker` to fill the post-``compute_spectra``
    attributes. The comparator carries **no design / contrast state**:
    cross-sample comparisons are specified per-call on
    :meth:`test_diff_freq` and :meth:`test_diff_expr`, so a single
    fitted comparator can serve any number of unrelated contrasts on
    the same ``spectra_``.

    Public attributes (user-facing, inspection-worthy)
    --------------------------------------------------
    samples : list
        The exact ``samples`` sequence passed at construction (AnnData
        list for the irregular path, SpatialData list for the grid
        path). Reference-shared with the user's collection.
    gene_names : list[str]
        Per-gene labels (length ``n_genes``).
    feature_mode : {'radial', '2d'}
        Spectral feature representation chosen at construction.
    freq_edges : np.ndarray | None
        Shared radial-frequency bin edges used by both feature modes.
        Auto-derived during :meth:`compute_spectra` if not supplied at
        construction.

    Public attributes set by :meth:`compute_spectra`
    -------------------------------------------------
    spectra_ : np.ndarray | None
        Per-(sample, gene) spectral feature matrix, shape
        ``(n_samples, n_genes, n_feature_bins)``. The trailing axis is
        radial bins in ``feature_mode='radial'`` and flattened
        ``radius × theta`` bins in ``feature_mode='2d'``. This is the
        input to :meth:`test_diff_freq` and the in-place
        :meth:`normalize_background` / :meth:`normalize_covariates`
        preprocessing transforms.
    dc_ : np.ndarray | None
        DC component of the spectrum per (sample, gene), shape
        ``(n_samples, n_genes)``. Equals the sample-grid mean of each
        gene's uncentered expression. Input to :meth:`test_diff_expr`.
    presence_ : np.ndarray | None
        Boolean mask of shape ``(n_samples, n_genes)`` — ``True`` where
        a gene's per-sample spot-presence fraction cleared
        ``_presence_threshold``. Drives the masked variant of
        :meth:`test_diff_freq` when any entry is ``False``.
    rotation_angles_ : np.ndarray | None
        Per-sample rotation angle (degrees) applied during
        rotation-alignment. Populated only when ``feature_mode='2d'``.
    freq_edges_requested_ : np.ndarray | None
        The original auto-resolved or requested 1-D edge grid before adaptive
        merging and support filtering. This is for reference purposes only.
    frequency_bin_intervals_ : np.ndarray | None
        The final radial frequency intervals of shape ``(n_retained_radius_bins, 2)``.
        It is stored as intervals because after dropping empty bins the
        edges may not be contiguous.
    frequency_bin_counts_ : np.ndarray | None
        Number of FFT-cell counts for each sample and originally requested radial
        frequency bin, shape ``(n_samples, len(freq_edges_requested_) - 1)``.
        These counts diagnose requested-bin support and distinguish empty
        Fourier-grid intervals from true zero-power measurements in
        :attr:`spectra_`.

    cross_spectra_ : CrossSpectra | None
        Pairwise result set by ``compute_cross_spectra``. Its ``cross_spectra``
        and matched ``auto_spectra`` use the same retained radial intervals
        as ``spectra_``.
        Successful auto-spectrum recomputation clears this result. Failed
        computations leave all previously fitted results intact.

    Private state (set by subclasses; not part of the user API)
    -----------------------------------------------------------
    ``_n_radial_bins``, ``_n_theta_bins``, ``_fft_solver``, ``_workers``,
    ``_presence_threshold``, ``_spacings``,
    ``_spectrum_fft_solver``, ``_grid_shapes``.
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
        if names is not None:
            if not names:
                raise ValueError("landmark_genes must contain at least one gene")
            missing = [g for g in names if g not in self.gene_names]
            if missing:
                raise KeyError(f"landmark_genes not in gene_names: {missing}")
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
            n_jobs, chunk = self._resolve_spectrum_schedule(spec, grids, n_jobs=n_jobs)
            indices = None if names is None else [self.gene_names.index(g) for g in names]
            n_genes = len(self.gene_names) if names is None else len(names)
            use_geomean = names is None or not self._landmark_cache_fits(n_genes)
            f_max = min(0.5 / max(spacing) for spacing in self._spacings)
            edges = np.linspace(0.0, f_max, _ALIGNMENT_N_RADIUS + 1)

            # Compute interpolated polar spectra for selected landmarks per sample.
            def _one(i, pbar):
                read, _dc, _presence, count, shape = self._sample_spectrum_chunker(
                    i, gene_indices=indices
                )

                return _stream_landmark_polar(
                    read,
                    count,
                    shape,
                    chunk_size=chunk,
                    fft_solver=self._spectrum_fft_solver,
                    freq_edges=edges,
                    spacing=self._spacings[i],
                    geomean=use_geomean,
                    frequency_padding=padding,
                    pbar=pbar,
                )

            landmarks = _run_per_sample(
                _one,
                len(self.samples),
                n_chunks_per_sample=int(np.ceil(n_genes / chunk)),
                desc="Alignment landmarks",
                n_jobs=n_jobs,
                progress=progress,
            )
            angles = _estimate_rotations_from_polar(landmarks, progress=progress)
        self.rotation_angles_ = angles
        self._alignment_landmark_genes = names

    def _compute_spectra(self, n_jobs, progress, landmark_genes=None, *, chunk_size=None):
        """Stream raw feature powers and DC/presence through either backend."""
        self._prepare_frequency_grid()
        padding = int(self.feature_mode == "2d" and self._spectrum_backend == "nufft")
        grids = [(ny + 2 * padding, nx + 2 * padding) for ny, nx in self._grid_shapes]
        spec = self._resolve_chunk_spec(chunk_size)
        n_jobs, chunk = self._resolve_spectrum_schedule(spec, grids, n_jobs=n_jobs)
        setattr(self, f"_{self._spectrum_backend}_chunk_size", chunk)
        self._ensure_alignment(n_jobs, progress, landmark_genes, chunk_size=chunk)

        # Compute binned and flattened single-feature auto-spectra per sample.
        def _one(i, pbar):
            read, dc, presence, count, shape = self._sample_spectrum_chunker(i)
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
        self, spec: int | str, grid_shapes: list[tuple[int, int]], *, n_jobs: int = 1
    ) -> tuple[int, int]:
        """Resolve sample jobs, transform threads and chunks from the same budget.

        Estimate 64 bytes per grid cell plus, for NUFFT, 64 per observation.
        Automatic jobs can shrink to fit one feature per sample job; explicit
        positive jobs retain their limit. Explicit chunks bypass memory sizing.
        """
        from sonic.utils import resolve_chunk_size, resolve_parallelism

        max_lat = max((ny * nx for (ny, nx) in grid_shapes), default=1)
        per_feat = 64 * max(max_lat, 1)
        if self._spectrum_backend == "nufft":
            per_feat += 64 * max(len(coords) for coords in self._coords)
        n_jobs, self._workers = resolve_parallelism(
            n_jobs,
            self._workers_spec,
            backend=self._spectrum_backend,
            n_tasks=len(self.samples),
            per_feat_bytes=per_feat if spec == "auto" else None,
            budget_bytes=self._auto_chunk_budget_bytes,
        )
        chunk = (
            resolve_chunk_size(
                self._auto_chunk_cap,
                per_feat,
                n_jobs=n_jobs,
                budget_bytes=self._auto_chunk_budget_bytes,
            )
            if spec == "auto"
            else spec
        )
        return n_jobs, chunk

    # Compact polar stacks are retained only for the alignment pass.
    _landmark_cache_warn_bytes: int = 8 * 1024**3

    def _landmark_cache_fits(self, n_landmarks: int) -> bool:
        """Bound the total compact landmark cache across samples, excluding transform chunks."""
        cache_bytes = len(self.samples) * n_landmarks * _ALIGNMENT_N_THETA * _ALIGNMENT_N_RADIUS * 8
        if cache_bytes > self._landmark_cache_warn_bytes:
            warnings.warn(
                f"Rotation alignment with {n_landmarks} explicit landmark genes would cache "
                f"{cache_bytes / 1024**3:.1f} GiB of polar spectra "
                f"(limit {self._landmark_cache_warn_bytes / 1024**3:.1f} GiB). Falling back "
                "to a single streamed geometric-mean of the landmark genes to avoid OOM.",
                UserWarning,
                stacklevel=3,
            )
            return False
        return True

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
        """
        Compute binned power spectra per feature per sample.

        Frequency bins are either 1D radial (``feature_mode='radial'``) or
        flattened 2D radial+angular (``feature_mode='2d'``).
        In 2D mode, the per-sample spectra are optionally rotation-aligned using a
        landmark gene set (default: a single geometric-mean landmark over all genes).
        2D bins are interpolated after rotation to handle heterogeneous sample grids,
        while empty 1D radial bins are adaptively merged to ensure every retained bin
        has support in all samples.

        Parameters
        ----------
        n_jobs : int, default -1
            Parallelism over samples for the per-sample spectrum pass. When
            ``progress=True`` the outer loop is sequential (so the tqdm bar is
            accurate). Transform threads use spare CPUs, capped at four by
            default. With automatic chunks, negative job counts are also capped
            by the workspace needed for one feature per concurrent sample.
        landmark_genes : sequence of str, optional
            Only used in ``feature_mode='2d'``. Names of genes (matched against
            :attr:`gene_names`) whose spectra define the rotation-alignment
            landmarks. Recovered rotations are still applied to every gene in
            :attr:`gene_names`. If None (default), a single streamed geometric
            mean over every configured gene is used on the first call, without
            presence filtering. Later calls reuse existing alignment when None
            is passed or the same explicit landmark panel is supplied. A changed
            panel refits alignment. Successful recomputation clears cross_spectra_.
        progress : bool, default True
            Show tqdm progress bars over the three phases (spectrum compute,
            optional rotation alignment, radial binning).
        memory_budget_bytes : int or str, default 2147483648 (2 GiB)
            Bytes or a size string, e.g. ``"512 MB"`` or ``"2 GiB"``. Units are
            case-insensitive: GB uses powers of 1000, GiB powers of 1024.
            Positive estimated transform-workspace budget shared across actual
            sample jobs when ``fft_chunk_size`` / ``nufft_chunk_size`` is
            ``'auto'``. More jobs divide this fixed budget; increase it explicitly
            to allow larger batches. Setting explicit chunk sizes (``fft_chunk_size``
            / ``nufft_chunk_size``) bypasses the budget limitation.

        chunk_size : int, "auto" or None, default None
            Genes per transform block. None uses the constructor setting;
            explicit sizes override the budget for this call only.

        Returns
        -------
        self
            Results replace the previous fit only after successful computation.
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
        """Compute raw cross-spectra using the auto-spectrum grid and alignment.

        Parameters
        ----------
        pairs : sequence of (str, str)
            Ordered feature pairs, using F_x * conj(F_y). Self-pairs are
            allowed; duplicate or reversed duplicate pairs are rejected.
        n_jobs : int, default -1
            Sample parallelism, as in compute_spectra. Progress reporting
            processes samples serially; otherwise jobs share the memory budget.
        landmark_genes : sequence of str, optional
            Shared polar-alignment landmarks, as in compute_spectra. Cold calls
            estimate alignment only but do not populate spectra_. None reuses
            existing angles, or uses all configured genes on the first call.
            If auto-spectra exist, change landmarks through compute_spectra first
            to keep their frame consistent. Cross-spectra always use the same
            alignment as the auto-spectra.
        progress : bool, default True
            Show progress during alignment and pair extraction.
        memory_budget_bytes : int or str, default 2147483648 (2 GiB)
            Estimated transform-workspace budget shared across sample jobs.
        chunk_size : int, "auto" or None, default None
            Features per transform block. None uses the constructor's
            FFT/NUFFT chunk setting; explicit sizes override the budget.

        Returns
        -------
        self
            Raw results are stored in cross_spectra_. Its cross_spectra and
            coherency arrays have axes (sample, pair, bin); auto_spectra has
            axes (sample, selected gene, bin). Radial intervals match
            frequency_bin_intervals_; polar bins flatten radius then angle.

        Notes
        -----
        Existing spectra_ and its normalization are preserved. A successful call
        replaces cross_spectra_ with exactly the requested pairs; failures preserve
        the previous result and alignment. Results are not implicitly accumulated.

        Matching auto_spectra are recomputed from source values and agree
        with the selected genes in unnormalized spectra_ within numerical
        tolerance. They do not follow background or covariate normalization;
        coherency always uses its matched raw cross- and auto-spectra.

        Polar alignment uses one shared axial angle per sample, modulo
        180 degrees. A half turn conjugates the cross-spectrum, leaving its
        real part and coherence unchanged but reversing its imaginary part.
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
        """Explain selected raw spectrum bins in one sample's spatial coordinates.

        This function allocates the observed feature value in each selected frequency bin
        to the source coordinates. It is not a derivative or a local significance test.

        Parameters
        ----------
        features : str or sequence of one/two str
            One feature for power, or an ordered pair for cross-spectrum maps.
        sample_index : int, default 0
            Sample in this comparator (or its subset).
        bands : sequence of int
            Distinct indices into the retained flattened spectrum, in output
            order. Polar bins flatten radius then angle; these are not indices
            into the original, potentially unsupported frequency intervals.
        representation : str, optional
            One feature: 'power' (default) or 'power_norm', which divides by
            the sum of raw power across all retained bins. An all-zero power
            profile remains zero. Pair: 'cross' (default, real co-spectrum),
            'coherency' (real signed coherency), or 'coherence' (squared complex
            magnitude). Invalid normalized pair bins raise ValueError.
        memory_budget_bytes : int or str, default '2 GiB'
            Estimated memory for returned maps, reducers and transform scratch.
            Checked before loading selected features.

        Returns
        -------
        sonic.explanation.SpatialExplanation
            One map per requested bin band. Signed contributions sum to that bin's
            **untransformed** feature (no log floor, test statistic or p-value is
            attributed). When the selected representation requires scaling
            (e.g., 'power_norm'), it is applied with observed normalization
            factors held fixed, and no attribution from the denominator.

        Notes
        -----
        Requires compute_spectra() or compute_cross_spectra() first. Recomputes
        only the selected source features using the fitted grid, bin reducers
        and alignment, without refitting or changing caches. Maps always explain
        raw-source spectra, including after normalize_background/covariates.

        For complex polar cross-spectrum S, let a_i be its complex allocation.
        Coherence allocates Re(conj(S) * a_i) / (P_x * P_y). Thus phase contributes
        to coherence maps, and local contributions can be negative.

        NUFFT maps retain source coordinates and row indices. FFT maps contain
        occupied raster cells, indexed in C order, with local (y, x) coordinates
        from spacing and origin zero, not SpatialData world coordinates. Only
        frequency weights are aligned; spatial coordinates are never rotated.
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
        """Return a lightweight comparator view over a subset of samples.

        The returned comparator is a shallow copy of ``self``: sample containers
        and gene-level configuration are shared, while per-sample fitted arrays
        and per-sample metadata are indexed to ``indices``. This is intended for
        running multiple contrasts against one fitted comparator without
        recomputing spectra.
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
        """Normalize each sample using its observed genes' geometric mean.

        Genes excluded by ``presence_`` neither contribute to the background
        nor get transformed. Samples with no observed genes are left unchanged.
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
        """Regress out per-sample covariate spectra from :attr:`spectra_`.

        Two input modes, detected from the first element's type:

        - **Sequence of str** — column-key list shared across samples.
          Each string key is considered as one covariate applied to every
          sample. Implementation depends on the subclass:

            * :class:`~sonic.ComparatorIrregular` looks each key up in
              ``adata.obs.columns`` first, then ``adata.var_names``
              (preferring obs on collision); the resolved per-spot
              vector is NUFFTed directly onto the sample's k-grid.
            * :class:`~sonic.ComparatorGrid` rasterizes via
              :func:`spatialdata.rasterize_bins` with the keys forwarded
              as ``value_key`` (any combination of ``.obs`` columns and
              ``var_names``).

        - **Sequence of np.ndarray** — per-sample pre-rasterized images,
          one ``(n_covariates, ny_i, nx_i)`` array per sample. Universal
          path; works on either subclass. Use when you want full control
          over rasterization, or when the covariates aren't already
          attached to the sample containers. These rasters must have
          frequency support compatible with the comparator's retained bins.

        Both modes produce the same downstream behaviour: per-sample
        covariate features are reduced to ``(n_covariates, n_feature_bins)``
        (same trailing axis as :attr:`spectra_`) and passed through
        :func:`~sonic.comparators.normalization.normalize_covariates` to
        log-space-residualise each gene's spectrum against them.

        Parameters
        ----------
        covariates : sequence of str or sequence of np.ndarray
            See modes above. Strings are interpreted by the subclass;
            ndarrays are used verbatim (one per sample).

        Returns
        -------
        self : _ComparatorBase
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
        null: str = "analytic",
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
        """Test differential frequency (DF) on the fitted gene spectra.

        Parameters
        ----------
        design : 1-D array, pandas.DataFrame, or 2-D np.ndarray
            Binary labels select the two-group path. A DataFrame or 2D array
            selects a GLM; supplying ``contrast`` also selects a GLM.
            Rows must match the fitted sample order. DataFrames add an intercept;
            numeric matrices are used as supplied.
        contrast : str, dict, or np.ndarray, optional
            One linear contrast, required on the GLM path. Names, coefficient
            dictionaries, and arrays follow :func:`~sonic.comparators.multisample.compare_glm`.
        statistic : str
            ``'log_l2'`` (default) is the weighted L2 contrast of log spectra.
            ``'log_t_cauchy'`` combines per-bin p-values on log profiles using
            Welch for binary labels or OLS contrast t tests for a GLM. It also
            supports localizing the best bin by its smallest per-bin p-value
            among positive-weight bins.
        null : str
            Null calibration (default ``'analytic'``), selected by ``statistic``:

            - ``'log_l2'``: ``'analytic'`` (selected covariance and Liu tail),
              ``'rotation'`` (deterministic conditional integration), or
              ``'permutation'`` (binary labels only, without ``contrast``).
            - ``'log_t_cauchy'``: ``'analytic'`` only (per-bin t tails and
              Cauchy combination).
        n_perm : int
            Number of sampled label permutations; default 1000. Ignored when
            all assignments are enumerated or another null is selected.
        random_state : int, optional
            Seed for binary label permutations; ignored by other nulls.
        freq_weights : np.ndarray, optional
            Nonnegative weights in the fitted frequency-bin order, normalized
            to sum to one. None gives uniform weights. Log t-Cauchy renormalizes
            over informative bins; zero weights exclude bins from combination,
            the statistic summary, and ``Best_bin``. Per-bin outputs are unchanged.
            Choose weights independently of the observed signal.
        max_exact_permutations : int
            Enumerate all binary label assignments when their number is at
            most this threshold; default 10000. Zero forces sampling.
        normalize_shape : bool
            If True, divide each sample/feature spectrum by its total power
            before logging to test spectral shape (DF-norm). Default False.
        min_samples_per_group : int
            Minimum observations per arm in masked binary tests; default 2.
            Ignored for unmasked or GLM comparisons.
        min_resid_df : int
            Minimum residual degrees of freedom in masked GLM tests; default 1.
            Ignored for unmasked or binary comparisons.
        covariance : str or None
            ``None`` (default) or ``'pooled'`` shares residual covariance across
            eligible genes. ``'per_gene'`` estimates it separately for each gene.
            An explicit choice requires an analytic L2 test.
        integration_tol : float
            Initial Davies absolute p-value error target for rotation; default 1e-6.
            Use at most 10,000 terms per call and one accuracy retry. Accepted
            integration also requires relative error budget <= 0.001. Unresolved
            tails use marked saddlepoint approximation. Ignored by other nulls.

        Returns
        -------
        pandas.DataFrame
            ``Feature``, ``Statistic``, ``P_value`` and ``P_adj``, sorted by
            descending statistic. Masked binary tests include ``n_obs_A`` and
            ``n_obs_B``; masked GLMs include ``n_obs`` and ``df_resid``.
            Log t-Cauchy reports the largest absolute t over positive-weight
            bins and adds ``Estimate_per_bin``, signed ``T_per_bin``, and raw
            ``P_value_per_bin`` in original bin order. Estimates are mean A
            minus mean B for binary labels, or the GLM contrast, on log profiles.
            ``Best_bin`` is the zero-based index with the smallest p-value among
            informative, positive-weight bins (first index on ties), or missing
            if none. Skipped genes have NaN per-bin arrays. Feature-level
            significance does not establish the selected bin's significance.
            BH uses finite feature p-values.
            Analytic L2 results record ``null`` and ``covariance`` in ``attrs``.
            Rotation adds ``P_value_method`` (``davies``, ``saddlepoint``, ``beta``
            or ``exact``; missing for skipped rows) and ``P_value_error``. The latter
            is the accepted Davies absolute error budget, a roundoff estimate for
            beta, or zero at exact boundaries; it is NaN for saddlepoint/skipped
            rows and does not measure statistical uncertainty. ``attrs`` records
            ``method="davies"``, ``integration_tol``, ``integration_limit=10000``,
            ``n_saddlepoint`` and ``covariance=None``.

        Raises
        ------
        RuntimeError
            If spectra have not been computed or conditional-tail evaluation fails.
        ValueError
            If shapes, options, or the contrast are invalid.
        NotImplementedError
            If a GLM permutation null is requested.

        Notes
        -----
        Uses :func:`~sonic.comparators.multisample.compare_two_groups` or
        :func:`~sonic.comparators.multisample.compare_glm` and their masked
        variants. ``normalize_shape`` leaves the cached spectra unchanged.
        For calibration assumptions and zero-variance behavior, see
        :doc:`/guides/multisample`.
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
            if statistic == "log_l2" and null not in ("analytic", "rotation"):
                raise NotImplementedError(
                    "Only null='analytic' or 'rotation' is supported when "
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
        null: str = "analytic",
        n_perm: int = 1000,
        random_state: int | None = None,
        freq_weights: np.ndarray | None = None,
        max_exact_permutations: int = 10000,
        min_samples_per_group: int = 2,
        min_resid_df: int = 1,
        covariance: str | None = None,
        integration_tol: float = 1e-6,
    ):
        """Test differential frequency in the coupling profiles of gene pairs.

        Parameters
        ----------
        design : 1-D array, pandas.DataFrame, or 2-D np.ndarray
            Binary labels select the two-group path. A DataFrame or 2D array
            selects a GLM; supplying ``contrast`` also selects a GLM.
            Rows must match the fitted sample order. DataFrames add an intercept;
            numeric matrices are used as supplied.
        contrast : str, dict, or np.ndarray, optional
            One linear contrast, required on the GLM path. Names, coefficient
            dictionaries, and arrays follow :func:`~sonic.comparators.multisample.compare_glm`.
        representation : str
            ``'coherence'`` (default) tests log squared-magnitude coherency.
            ``'coherency'`` tests its signed real part without logging.
        statistic : str
            ``'l2'`` (default) is the weighted L2 contrast on the selected scale.
            ``'t_cauchy'`` combines per-bin p-values using Welch for binary
            labels or OLS contrast t tests for a GLM. It also localizes the
            best bin by its smallest p-value among positive-weight bins.
        null : str
            Null calibration (default ``'analytic'``), selected by ``statistic``:

            - ``'l2'``: ``'analytic'`` (selected covariance and Liu tail),
              ``'rotation'`` (deterministic conditional integration), or
              ``'permutation'`` (binary labels only, without ``contrast``).
            - ``'t_cauchy'``: ``'analytic'`` only (per-bin t tails and
              Cauchy combination).
        n_perm : int
            Number of sampled label permutations; default 1000. Ignored when
            all assignments are enumerated or another null is selected.
        random_state : int, optional
            Seed for binary label permutations; ignored by other nulls.
        freq_weights : np.ndarray, optional
            Nonnegative weights on the original flattened bins (radius then
            angle for polar features); None gives uniform weights. Weights are
            normalized over retained bins, and over informative bins for
            t-Cauchy. Zero weights exclude bins from the test and ``Best_bin``;
            valid per-bin outputs are retained. Choose weights independently
            of the observed signal.
        max_exact_permutations : int
            Enumerate all binary label assignments when their number is at
            most this threshold; default 10000. Zero forces sampling.
        min_samples_per_group : int
            Minimum observations per arm in masked binary tests; default 2.
            Ignored for unmasked or GLM comparisons.
        min_resid_df : int
            Minimum residual degrees of freedom in masked GLM tests; default 1.
            Ignored for unmasked or binary comparisons.
        covariance : str or None
            ``None`` (default) or ``'pooled'`` shares residual covariance across
            eligible pairs and PSD-corrects the pooled matrix. ``'per_pair'``
            estimates it separately on each pair's retained samples and bins.
            An explicit choice requires an analytic L2 test.
        integration_tol : float
            Initial Davies absolute p-value error target for rotation; default 1e-6.
            Use at most 10,000 terms per call and one accuracy retry. Accepted
            integration also requires relative error budget <= 0.001. Unresolved
            tails use marked saddlepoint approximation. Ignored by other nulls.

        Returns
        -------
        pandas.DataFrame
            ``Feature_1``, ``Feature_2``, ``Statistic``, ``P_value`` and ``P_adj``,
            sorted by descending statistic. L2 reports the weighted distance;
            t-Cauchy reports the largest absolute t over positive-weight bins.
            Includes sample/bin counts, ``DF_resid`` (design residual df),
            ``Effective_rank``, ``N_pairs_pooled`` and ``Status``. Binary tests
            also return ``n_obs_A`` and ``n_obs_B``. Skipped and self-pair rows
            have NaN p-values; BH uses finite pair p-values.

            t-Cauchy adds ``Estimate_per_bin``, signed ``T_per_bin``, and raw
            ``P_value_per_bin`` in original bin order, with NaNs for unsupported
            bins or skipped pairs. Estimates are mean A minus mean B for binary
            labels, or the GLM contrast, on the selected representation's scale.
            ``Best_bin`` is the zero-based index with the smallest p-value among
            informative, positive-weight bins (first index on ties), or missing
            if none. Pair-level significance does not establish the selected
            bin's significance. Welch degrees of freedom vary by bin;
            ``DF_resid`` is not the df used for its tails.

            ``attrs`` records retained sample/bin indices, weights, representation
            and design. ``covariance`` is ``'pooled_across_valid_bins_psd'``,
            ``'per_pair'``, or None. Per-pair fits have one pooled contributor;
            skipped, rotation, permutation and t-Cauchy pairs have zero. t-Cauchy has no
            covariance or effective rank and records ``per_bin_test`` as
            ``welch`` or ``ols``. Rotation effective ranks describe the
            conditional scatter. Rotation adds ``P_value_method`` (``davies``, ``saddlepoint``, ``beta``
            or ``exact``; missing for skipped rows) and ``P_value_error``. The latter
            is the accepted Davies absolute error budget, a roundoff estimate for
            beta, or zero at exact boundaries; it is NaN for saddlepoint/skipped
            rows and does not measure statistical uncertainty. ``attrs`` records
            ``method="davies"``, ``integration_tol``, ``integration_limit=10000``,
            ``n_saddlepoint`` and ``covariance=None``.
            Permutation counts/exactness are per-pair when masked, otherwise shared.

        Raises
        ------
        RuntimeError
            If the required spectra have not been computed or conditional-tail evaluation fails.
        ValueError
            If shapes, options, or the contrast are invalid.
        NotImplementedError
            If a GLM permutation null is requested.

        Notes
        -----
        Each pair retains samples where both genes are present, and bins
        valid in every retained sample. Self-pairs are descriptive only.
        For calibration assumptions and zero-variance behavior, see
        :doc:`/guides/multisample`.
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
    ) -> _AnalyticNullState:
        """Estimate the analytic L2 null covariance and fitted coefficients.

        Parameters
        ----------
        design : 1-D array, pandas.DataFrame, or 2-D np.ndarray
            Binary labels select the two-group path. A DataFrame or 2D array
            selects a GLM; supplying ``contrast`` also selects a GLM.
            Rows must match the fitted sample order. DataFrames add an intercept;
            numeric matrices are used as supplied.
        contrast : str, dict, or np.ndarray, optional
            One linear contrast, required on the GLM path. Names, coefficient
            dictionaries, and arrays follow :func:`~sonic.comparators.multisample.compare_glm`.
        freq_weights : np.ndarray, optional
            Nonnegative bin weights, normalized to sum to one.
            None gives uniform weights.
        normalize_shape : bool
            If True, divide each sample/feature spectrum by its total power
            before logging to test spectral shape (DF-norm). Default False.
        min_samples_per_group : int
            Minimum observations per arm in masked binary tests; default 2.
            Ignored for unmasked or GLM comparisons.
        min_resid_df : int
            Minimum residual degrees of freedom in masked GLM tests; default 1.
            Ignored for unmasked or binary comparisons.
        covariance : str or None
            ``None`` (default) or ``'pooled'`` shares residual covariance across
            eligible genes. ``'per_gene'`` estimates it separately for each gene.
            An explicit choice requires an analytic L2 test.

        Returns
        -------
        dict
            Analytic-null state with common keys ``mode``, ``masked``,
            ``observed``, ``beta``, ``sigma_log``, ``freq_weights``, ``weighted_cov``,
            ``eigenvalues``, ``effective_rank``, ``contrast_scale``,
            ``df_resid``, ``eligible``, and the effective ``covariance`` choice.
            With ``covariance="per_gene"``, ``sigma_log`` and ``weighted_cov``
            have shape ``(n_genes, n_bins, n_bins)``, ``eigenvalues`` has shape
            ``(n_genes, n_bins)``, and ``effective_rank`` and ``roundoff`` are
            per-gene vectors. Ineligible covariance/eigenvalue rows are NaN.
            Full per-gene covariance storage is only allocated by this diagnostic
            method; tests retain eigenvalues without storing all matrices.

            Fitted ``beta`` coefficients always have shape
            ``(n_terms, n_genes, n_bins)`` on the tested profile scale. Binary
            tests return two rows: the group-A mean and B-minus-A difference,
            with groups ordered by ``np.unique``. Skipped genes retain NaNs.
            Two-group-only metadata: ``n_obs_A`` and ``n_obs_B``.
            GLM-only metadata: ``design_columns``, ``contrast_vector``, and ``n_obs``.

        Raises
        ------
        RuntimeError
            If the required spectra have not been computed.
        ValueError
            If shapes, options, or the contrast are invalid.

        Notes
        -----
        This diagnostic stores full covariance matrices. Comparison tests
        retain eigenvalues without storing all per-gene covariance matrices.
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .compute_spectra() before .estimate_null_covariance().")
        if int(min_samples_per_group) < 2:
            raise ValueError(f"min_samples_per_group must be >= 2, got {min_samples_per_group}.")
        if int(min_resid_df) < 1:
            raise ValueError(f"min_resid_df must be >= 1, got {min_resid_df}.")

        groups, design_obj = _validate_design(design, len(self.samples))
        use_masked = self.presence_ is not None and not self.presence_.all()
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
                return _estimate_glm_masked_null_covariance(
                    self.spectra_,
                    design_obj,
                    contrast,
                    self.presence_,
                    min_resid_df=int(min_resid_df),
                    **options,
                )
            return _estimate_glm_null_covariance(
                self.spectra_,
                design_obj,
                contrast,
                **options,
            )

        if use_masked:
            return _estimate_two_group_masked_null_covariance(
                self.spectra_,
                groups,
                self.presence_,
                min_samples_per_group=int(min_samples_per_group),
                **options,
            )
        return _estimate_two_group_null_covariance(
            self.spectra_,
            groups,
            **options,
        )

    def test_diff_expr(
        self,
        design: Any,
        *,
        contrast: str | dict[str, float] | np.ndarray | None = None,
        log_expression: bool = False,
        eps: float = 1e-12,
    ) -> Any:
        """Test differential expression (DE) on the fitted sample-level DC means.

        Parameters
        ----------
        design : 1-D array, pandas.DataFrame, or 2-D np.ndarray
            Binary labels select the two-group path. A DataFrame or 2D array
            selects a GLM; supplying ``contrast`` also selects a GLM.
            Rows must match the fitted sample order. DataFrames add an intercept;
            numeric matrices are used as supplied.
        contrast : str, dict, or np.ndarray, optional
            One linear contrast, required on the GLM path. Names, coefficient
            dictionaries, and arrays follow :func:`~sonic.comparators.multisample.compare_glm`.
        log_expression : bool
            If True, fit ``log(dc + eps)`` instead of raw DC means.
            Default False; reported effects use the fitted response scale.
        eps : float
            Positive additive offset used only with ``log_expression=True``;
            default 1e-12.

        Returns
        -------
        pandas.DataFrame
            ``Feature``, ``Statistic`` (absolute t), ``P_value`` and ``P_adj``,
            plus ``Mean_diff`` (group A minus B) for binary comparisons or
            ``Estimate`` (signed contrast) for GLMs. Rows are sorted by
            descending statistic; BH uses finite p-values.

        Raises
        ------
        RuntimeError
            If the required spectra have not been computed.
        ValueError
            If shapes, options, or the contrast are invalid.

        Notes
        -----
        Binary comparisons use :func:`~sonic.comparators.multisample.compare_two_groups_scalar`
        and Welch-Satterthwaite tails. GLMs use
        :func:`~sonic.comparators.multisample.compare_glm_scalar` and Student t tails.
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
        """Per-sample effective rank ``K_eff`` of the gene-spectrum covariance.

        Quantifies how concentrated the spatial-frequency content is along
        the eigen-directions of each sample's cross-gene-spectrum covariance matrix.
        ``K_eff = (Σλ)² / Σλ²`` — bounded by 1 (rank-1, all power on a
        single direction) and ``n_feature_bins`` (uniformly spread).

        Parameters
        ----------
        weights : np.ndarray, optional
            Per-bin weights (same semantics as ``freq_weights``). When
            given, returns the effective rank of
            ``W^{1/2} Σ W^{1/2}`` — useful for analysing how a
            frequency-weighted L2 statistic redistributes its power.

        Returns
        -------
        np.ndarray of shape ``(n_samples,)``.
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
