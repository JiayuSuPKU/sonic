"""Streamed cross-spectra on the comparator frequency grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import issparse

from sonic.comparators.base import _run_per_sample
from sonic.comparators.binning import (
    _mode_pair_counts,
    _polar_bin_reducer,
    _radial_bin_reducer,
)
from sonic.kernels.fft import _fourier_coefficients_2d
from sonic.kernels.nufft import _fourier_coefficients_2d_nufft
from sonic.utils import _parse_memory_budget, resolve_parallelism


@dataclass
class CrossSpectra:
    """Raw binned cross-spectra and their matching auto-spectra.

    Attributes
    ----------
    cross_spectra : ndarray, shape (n_samples, n_pairs, n_bins)
        Ordered products F_x * conj(F_y) / n_observed**2 after mean centering.
        Radial bins contain real co-spectra (stored with zero imaginary part);
        polar bins retain complex phase. Finite raw values are preserved
        even when coherency is masked.
    auto_spectra : ndarray, shape (n_samples, n_genes, n_bins)
        Matching raw powers for the selected genes, in gene_names order.
        These agree with the corresponding unnormalized comparator.spectra_
        within numerical tolerance. They are recomputed from source values
        and do not follow later background or covariate normalization.
    pairs : tuple of (str, str)
        Requested ordered gene pairs.
    gene_names : tuple of str
        Selected genes, retaining the comparator's gene order.
    pair_indices : ndarray, shape (n_pairs, 2)
        Indices into gene_names for each ordered pair.
    coherency_valid : ndarray of bool, shape (n_samples, n_pairs, n_bins)
        Whether coherency can be computed. Requires finite cross-spectra,
        at least two frequency groups, sufficient expression presence and
        positive auto-power above 1e-12 of each gene's peak band power.
    mode_pair_counts : ndarray, shape (n_samples, n_bins)
        Contributing source-frequency groups up to conjugacy, excluding DC.
        These are geometric support counts, not independent observations.
    frequency_bin_intervals : ndarray, shape (n_radius, 2)
        Retained radial intervals in cycles per spatial unit, matching the
        comparator's frequency_bin_intervals_. Polar arrays flatten radius
        then angle, with angle varying fastest.
    angle_centers_rad : ndarray, shape (n_theta,)
        Polar sector centers in radians over [0, pi); empty for radial bins.
    rotation_angles_deg : ndarray, shape (n_samples,)
        A copy of the comparator's shared polar alignment angles in degrees;
        zeros for radial bins. No separate pairwise alignment is estimated.
    feature_mode : {'radial', '2d'}
        Same feature mode as the comparator; '2d' denotes polar bins.
    chunk_schedules : list of dict
        Per-sample chunk_size, estimated_workspace_bytes, n_locations and
        n_freqs (number of source Fourier cells). No FFT cache is retained.

    Notes
    -----
    The same nonnegative bin weights reduce cross-products and both powers.
    Polar alignment is axial, modulo 180 degrees. A half turn conjugates
    the complex cross-spectrum, so the sign of its imaginary part has an
    orientation ambiguity; the real part and coherence are unaffected.
    """

    cross_spectra: np.ndarray
    auto_spectra: np.ndarray
    pairs: tuple
    gene_names: tuple
    pair_indices: np.ndarray
    coherency_valid: np.ndarray
    mode_pair_counts: np.ndarray
    frequency_bin_intervals: np.ndarray
    angle_centers_rad: np.ndarray
    rotation_angles_deg: np.ndarray
    feature_mode: str
    chunk_schedules: list

    @property
    def coherency(self):
        """Cross-spectrum / sqrt(matched auto powers); radial values are signed."""
        left = self.auto_spectra[:, self.pair_indices[:, 0]]
        right = self.auto_spectra[:, self.pair_indices[:, 1]]
        denominator = np.sqrt(left) * np.sqrt(right)
        result = np.full_like(self.cross_spectra, np.nan)
        np.divide(self.cross_spectra, denominator, out=result, where=self.coherency_valid)
        if np.any(np.abs(result[self.coherency_valid]) > 1 + 1e-8):
            raise ArithmeticError("Cross/auto reduction violated the coherence bound")
        return result

    @property
    def coherence(self):
        """Squared magnitude of coherency; invalid bins remain NaN."""
        return np.abs(self.coherency) ** 2


def _make_feature_reader(comparator, sample_index, raster):
    """Return a feature reader and location count; densify only requested columns."""
    if comparator._spectrum_backend == "nufft":
        sample = comparator.samples[sample_index]
        matrix = sample.X if comparator._layer is None else sample.layers[comparator._layer]
        if issparse(matrix):
            matrix = matrix.tocsc()
        n_locations = matrix.shape[0]

        def read_features(columns):
            block = matrix[:, columns]
            return np.asarray(block.toarray() if issparse(block) else block, dtype=float).copy()

        coords = comparator._coords[sample_index]
        if not np.isfinite(coords).all():
            raise ValueError("Coordinates must be finite")
    else:
        n_locations = int(np.prod(raster.shape[-2:]))

        def read_features(columns):
            values = comparator._batch_raster(raster, [comparator.gene_names[j] for j in columns])
            return values.reshape(len(columns), -1).T.copy()

    if n_locations < 2:
        raise ValueError("Each sample must have at least two observations")
    return read_features, n_locations


def _compute_cross_spectra(  # noqa: C901 — validation and two backend schedules
    comparator,
    pairs,
    n_jobs=-1,
    progress=True,
    *,
    memory_budget_bytes="2 GiB",
    chunk_size=None,
    landmark_genes=None,
):
    """Compute raw pair spectra on the comparator's shared grid and alignment.

    Alignment is prepared independently of auto-spectrum extraction and reused
    for 2D polar bins. Radial extraction needs no rotation.

    Each sample centers and transforms feature chunks, then averages
    F_x * conj(F_y) / n_observed**2 and matching auto-powers using identical
    weights. Only two coefficient blocks and one product grid are retained.
    Raw powers are recomputed from source values, not comparator.spectra_,
    which may already have been normalized.

    Parallelism and memory settings follow compute_spectra. chunk_size=None
    uses the constructor's FFT/NUFFT setting; explicit sizes override the
    budget. The returned CrossSpectra masks unstable coherency without
    discarding finite raw spectra.
    """
    # Parse and validate the memory budget and chunk size.
    budget_bytes = _parse_memory_budget(memory_budget_bytes)
    chunk_size = comparator._resolve_chunk_spec(chunk_size)

    # Check the requested pairs and map them to the comparator's gene names.
    all_genes = tuple(comparator.gene_names)
    pairs = tuple(pairs)
    if not pairs or any(isinstance(p, str) or len(p) != 2 for p in pairs):
        raise ValueError("Provide at least one pair of feature names")
    pairs = tuple(tuple(p) for p in pairs)
    pair_genes = {g for p in pairs for g in p}
    if not pair_genes.issubset(all_genes):
        raise ValueError("All pair features must exist in the comparator")
    if len({frozenset(p) for p in pairs}) != len(pairs):
        raise ValueError("Duplicate or reversed duplicate pairs are not allowed")
    genes = tuple(g for g in all_genes if g in pair_genes)
    gene_index = {g: i for i, g in enumerate(genes)}
    pair_indices = np.array([[gene_index[a], gene_index[b]] for a, b in pairs])
    gene_columns = np.array([all_genes.index(g) for g in genes])
    n_samples = len(comparator.samples)

    # Prepare on the public caller's shallow copy, so failed fits leave its state intact.
    previous_intervals = comparator.frequency_bin_intervals_
    comparator._prepare_frequency_grid()
    if comparator.spectra_ is not None and not np.array_equal(
        previous_intervals, comparator.frequency_bin_intervals_
    ):
        raise ValueError(
            "Frequency bins changed; call compute_spectra() before computing cross-spectra"
        )
    edges = np.asarray(comparator.freq_edges, dtype=float).copy()
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.isfinite(edges).all()
        or edges[0] < 0
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("Frequency edges must be finite, nonnegative and increasing")

    frequency_padding = int(
        comparator.feature_mode == "2d" and comparator._spectrum_backend == "nufft"
    )
    sample_sizes = [
        (
            int(np.prod(np.asarray(shape) + 2 * frequency_padding)),
            len(comparator._coords[i])
            if comparator._spectrum_backend == "nufft"
            else int(np.prod(shape)),
        )
        for i, shape in enumerate(comparator._grid_shapes)
    ]
    min_workspace_bytes = max(
        128 * n_freqs + 128 * (n_freqs + n_locations) for n_freqs, n_locations in sample_sizes
    )
    # Keep automatic job counts inside the same aggregate budget as chunks.
    if chunk_size == "auto" and budget_bytes < min_workspace_bytes:
        raise MemoryError("Budget cannot fit two one-feature transform blocks plus scratch")
    n_jobs, workers = resolve_parallelism(
        1 if progress else n_jobs,
        comparator._workers_spec,
        backend=comparator._spectrum_backend,
        n_tasks=n_samples,
        per_feat_bytes=min_workspace_bytes if chunk_size == "auto" else None,
        budget_bytes=budget_bytes,
    )
    sample_budget_bytes = budget_bytes // n_jobs

    # Existing auto features must retain the alignment under which they were computed.
    if comparator.feature_mode == "2d":
        if (
            comparator.spectra_ is not None
            and landmark_genes is not None
            and tuple(landmark_genes) != comparator._alignment_landmark_genes
        ):
            raise ValueError(
                "To change alignment landmarks, call compute_spectra(landmark_genes=...) first"
            )
        comparator._auto_chunk_budget_bytes = budget_bytes
        comparator._ensure_alignment(
            1 if progress else n_jobs, progress, landmark_genes, chunk_size=chunk_size
        )
        rotation_angles_deg = np.asarray(comparator.rotation_angles_, dtype=float)
    else:
        rotation_angles_deg = np.zeros(n_samples)
    if rotation_angles_deg.shape != (n_samples,) or not np.isfinite(rotation_angles_deg).all():
        raise ValueError("Provide one finite shared rotation angle per sample")

    # Compute the cross-spectra and auto-spectra in a streamed, chunked fashion.
    angle_centers_rad = (
        np.arange(comparator._n_theta_bins) * np.pi / comparator._n_theta_bins
        if comparator.feature_mode == "2d"
        else np.array([])
    )
    n_bins = (len(edges) - 1) * max(1, len(angle_centers_rad))
    cross_spectra = np.full((n_samples, len(pairs), n_bins), np.nan + 0j)
    auto_spectra = np.full((n_samples, len(genes), n_bins), np.nan)
    mode_pair_counts = np.zeros((n_samples, n_bins))
    presence = np.zeros((n_samples, len(genes)), dtype=bool)
    chunk_schedules = [None] * n_samples

    def compute_sample(sample_index, pbar):  # noqa: C901 — streamed FFT/NUFFT block products
        shape = comparator._grid_shapes[sample_index]
        spacing = comparator._spacings[sample_index]
        raster = (
            comparator._rasterize_one(comparator.samples[sample_index])
            if comparator._spectrum_backend == "fft"
            else None
        )
        read_features, n_locations = _make_feature_reader(comparator, sample_index, raster)
        n_freqs = int(np.prod(np.asarray(shape) + 2 * frequency_padding))
        # Include two coefficient blocks, transform scratch and one pair-product grid.
        fixed_bytes = 128 * n_freqs
        bytes_per_feature = 128 * (n_freqs + n_locations)
        if chunk_size == "auto":
            capacity = min(len(genes), (sample_budget_bytes - fixed_bytes) // bytes_per_feature)
            if capacity < 1:
                raise MemoryError("Budget cannot fit two one-feature transform blocks plus scratch")
            chunk = int(min(32, capacity))
        else:
            chunk = min(int(chunk_size), len(genes))
        chunk_schedules[sample_index] = {
            "chunk_size": chunk,
            "estimated_workspace_bytes": fixed_bytes + bytes_per_feature * chunk,
            "n_locations": n_locations,
            "n_freqs": n_freqs,
        }
        if comparator.feature_mode == "2d":
            reducer, mode_pair_counts[sample_index] = _polar_bin_reducer(
                shape,
                spacing=spacing,
                edges=edges,
                n_theta=comparator._n_theta_bins,
                angle_deg=rotation_angles_deg[sample_index],
                frequency_padding=frequency_padding,
            )
        else:
            reducer, _ = _radial_bin_reducer(shape, fft_solver="fft2", spacing=spacing, edges=edges)
            mode_pair_counts[sample_index] = _mode_pair_counts(
                reducer, shape, periodic=comparator._spectrum_backend == "fft"
            )
        shared_mask = [None]

        def transform_block(columns):
            block = read_features(gene_columns[columns])
            if comparator._spectrum_backend == "fft":
                observed = np.isfinite(block)
                if np.isinf(block).any() or not np.all(observed == observed[:, :1]):
                    raise ValueError(
                        "FFT features require identical finite support; only shared NaN holes are allowed"
                    )
                mask = observed[:, 0]
                if shared_mask[0] is None:
                    shared_mask[0] = mask.copy()
                elif not np.array_equal(shared_mask[0], mask):
                    raise ValueError("FFT features require identical finite support across chunks")
                n_observed = int(mask.sum())
                if n_observed < 2:
                    raise ValueError("Each sample must have at least two observed locations")
                presence[sample_index, columns] = (
                    np.count_nonzero(block[mask], axis=0) / n_observed
                    >= comparator._presence_threshold
                )
                block -= block[mask].mean(axis=0)
                block[~mask] = 0
            elif not np.isfinite(block).all():
                raise ValueError("Feature values must be finite on common sample support")
            else:
                n_observed = n_locations
                presence[sample_index, columns] = (
                    np.count_nonzero(block, axis=0) / n_locations >= comparator._presence_threshold
                )
                block -= block.mean(axis=0)
            if comparator._spectrum_backend == "nufft":
                coeffs = _fourier_coefficients_2d_nufft(
                    comparator._coords[sample_index],
                    block,
                    shape,
                    spacing,
                    unit_scale=comparator._unit_scales[sample_index],
                    eps=comparator._nufft_eps,
                    nthreads=workers,
                    frequency_padding=frequency_padding,
                )
            else:
                coeffs = _fourier_coefficients_2d(
                    block.reshape(*shape, len(columns)), "fft2", workers, center=True
                )
            coeffs[0, 0] = 0
            coeffs = np.moveaxis(coeffs, -1, 0).reshape(len(columns), -1)
            coeffs /= n_observed
            for local, column in enumerate(columns):
                auto_spectra[sample_index, column] = reducer @ (
                    coeffs[local].real ** 2 + coeffs[local].imag ** 2
                )
            return coeffs

        blocks = [np.arange(i, min(i + chunk, len(genes))) for i in range(0, len(genes), chunk)]
        pairs_by_block = [{} for _ in blocks]
        for pair_idx, (x, y) in enumerate(pair_indices):
            ai, bi = sorted((x // chunk, y // chunk))
            pairs_by_block[ai].setdefault(bi, []).append((pair_idx, x, y))
        for ai, a in enumerate(blocks):
            coeffs_a = transform_block(a)
            for bi, pair_tasks in pairs_by_block[ai].items():
                b = blocks[bi]
                coeffs_b = coeffs_a if ai == bi else transform_block(b)
                for pair_idx, x, y in pair_tasks:
                    if x // chunk == ai:
                        product = coeffs_a[x - a[0]] * np.conj(coeffs_b[y - b[0]])
                    else:
                        product = coeffs_b[x - b[0]] * np.conj(coeffs_a[y - a[0]])
                    binned = reducer @ product
                    cross_spectra[sample_index, pair_idx] = (
                        binned if comparator.feature_mode == "2d" else binned.real
                    )
                    del product
                del coeffs_b
            del coeffs_a
        unsupported = mode_pair_counts[sample_index] == 0
        auto_spectra[sample_index, :, unsupported] = np.nan
        cross_spectra[sample_index, :, unsupported] = np.nan
        if pbar is not None:
            pbar.update(1)

    # Compute per-sample cross and auto spectra in parallel, with progress bar if requested.
    _run_per_sample(
        compute_sample,
        n_samples,
        n_chunks_per_sample=1,
        desc="Cross-spectra",
        n_jobs=n_jobs,
        progress=progress,
    )
    keep = comparator._supported_frequency_bin_mask
    cross_spectra = comparator._select_supported_frequency_bins(cross_spectra, keep)
    auto_spectra = comparator._select_supported_frequency_bins(auto_spectra, keep)
    mode_pair_counts = comparator._select_supported_frequency_bins(mode_pair_counts, keep)
    # Coherency checks preserve raw powers and positive amplitude invariance.
    peak_power = np.max(
        np.where(np.isfinite(auto_spectra), auto_spectra, 0), axis=-1, keepdims=True
    )
    power_valid = (
        np.isfinite(auto_spectra) & (auto_spectra > 1e-12 * peak_power) & (auto_spectra > 0)
    )
    coherency_valid = (
        np.isfinite(cross_spectra)
        & (mode_pair_counts[:, None, :] >= 2)
        & presence[:, pair_indices[:, 0], None]
        & presence[:, pair_indices[:, 1], None]
        & power_valid[:, pair_indices[:, 0]]
        & power_valid[:, pair_indices[:, 1]]
    )
    result = CrossSpectra(
        cross_spectra=cross_spectra,
        auto_spectra=auto_spectra,
        pairs=pairs,
        gene_names=genes,
        pair_indices=pair_indices,
        coherency_valid=coherency_valid,
        mode_pair_counts=mode_pair_counts,
        frequency_bin_intervals=comparator.frequency_bin_intervals_.copy(),
        angle_centers_rad=angle_centers_rad,
        rotation_angles_deg=rotation_angles_deg.copy(),
        feature_mode=comparator.feature_mode,
        chunk_schedules=chunk_schedules,
    )
    _ = result.coherency
    return result
