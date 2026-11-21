"""Pairwise cross-spectrum extraction and cross-sample comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from sonic.comparators._inference import (
    _L2_NULL_OPTIONS,
    _estimate_glm_masked_null_covariance,
    _estimate_two_group_masked_null_covariance,
    _estimate_two_group_null_covariance,
    _fit_glm_profiles,
    _integration_metadata,
    _iter_feature_blocks,
    _iter_mask_groups,
    _l2_pvalues_from_state,
    _make_label_permutations,
    _resolve_freq_weights,
    _run_l2_conditional,
    _run_l2_permutations,
    _weighted_covariance_eigensystem,
)
from sonic.comparators._regression import (
    _build_design_matrix,
    _decompose_glm_design,
    _prepare_glm_design,
    _resolve_contrast,
)
from sonic.comparators.base import _run_per_sample
from sonic.comparators.binning import (
    _mode_pair_counts,
    _polar_bin_reducer,
    _radial_bin_reducer,
)
from sonic.comparators.features import _center_observations, _shared_observation_mask
from sonic.comparators.multisample import (
    _BINWISE_COLUMNS,
    _resolve_profile_null,
    _run_t_cauchy_test,
)
from sonic.kernels.fft import _fourier_coefficients_2d
from sonic.kernels.nufft import _fourier_coefficients_2d_nufft
from sonic.statistics import apply_bh_correction
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
        These agree with the corresponding unnormalized ``comparator.spectra_``
        within numerical tolerance. They are recomputed from source values
        and do not follow later background or covariate normalization.
    pairs : tuple of (str, str)
        Requested ordered gene pairs.
    gene_names : tuple of str
        Selected genes, retaining the comparator's gene order.
    presence : ndarray of bool, shape (n_samples, n_genes)
        Expression presence for selected genes, in gene_names order. Testing
        uses a sample only when both genes in the pair are present.
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
        comparator's ``frequency_bin_intervals_``. Polar arrays flatten radius
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
    presence: np.ndarray
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
        return self._coherency_block(slice(None))

    def _coherency_block(self, pairs):
        """Compute and check coherency for a slice of pairs."""
        left = self.auto_spectra[:, self.pair_indices[pairs, 0]]
        right = self.auto_spectra[:, self.pair_indices[pairs, 1]]
        denominator = np.sqrt(left) * np.sqrt(right)
        cross = self.cross_spectra[:, pairs]
        valid = self.coherency_valid[:, pairs]
        result = np.full_like(cross, np.nan)
        np.divide(cross, denominator, out=result, where=valid)
        if np.any(np.abs(result[valid]) > 1 + 1e-8):
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
    all_genes = pd.Index(comparator.gene_names)
    pairs = tuple(pairs)
    if not pairs or any(isinstance(p, str) or len(p) != 2 for p in pairs):
        raise ValueError("Provide at least one pair of feature names")
    pairs = tuple(tuple(p) for p in pairs)
    pair_genes = {g for p in pairs for g in p}
    if not pair_genes.issubset(all_genes):
        raise ValueError("All pair features must exist in the comparator")
    if len({frozenset(p) for p in pairs}) != len(pairs):
        raise ValueError("Duplicate or reversed duplicate pairs are not allowed")
    gene_columns = np.flatnonzero(all_genes.isin(pair_genes))
    selected = all_genes[gene_columns]
    if not selected.is_unique:
        ambiguous = selected[selected.duplicated()].unique().tolist()
        raise ValueError(
            f"Ambiguous pair feature names {ambiguous}. Use unique feature identifiers."
        )
    genes = tuple(selected)
    pair_indices = selected.get_indexer(np.ravel(pairs)).reshape(-1, 2)
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
            values = block
            if comparator._spectrum_backend == "fft":
                mask = _shared_observation_mask(block)
                if shared_mask[0] is None:
                    shared_mask[0] = mask.copy()
                elif not np.array_equal(shared_mask[0], mask):
                    raise ValueError("FFT features require identical finite support across chunks")
                if not mask.all():
                    values = block[mask]
            elif not np.isfinite(block).all():
                raise ValueError("Feature values must be finite on common sample support")
            n_observed = len(values)
            if n_observed < 2:
                raise ValueError("Each sample must have at least two observed locations")
            presence[sample_index, columns] = (
                np.count_nonzero(values, axis=0) / n_observed >= comparator._presence_threshold
            )
            _center_observations(values)
            if values is not block:
                block[mask] = values
                block[~mask] = 0
            del values
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
        shared_output=True,
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
        presence=presence,
        pair_indices=pair_indices,
        coherency_valid=coherency_valid,
        mode_pair_counts=mode_pair_counts,
        frequency_bin_intervals=comparator.frequency_bin_intervals_.copy(),
        angle_centers_rad=angle_centers_rad,
        rotation_angles_deg=rotation_angles_deg.copy(),
        feature_mode=comparator.feature_mode,
        chunk_schedules=chunk_schedules,
    )
    for block in _iter_feature_blocks(n_samples, len(pairs), cross_spectra.shape[-1]):
        result._coherency_block(block)
    return result


def _prepare_cross_profiles(result, pairs, bins, representation, presence):
    """Materialize only one common-support group of binned pair profiles."""
    # From the raw cross-spectra and auto-spectra, compute the requested
    # coherency or signed co-spectrum.
    rows = np.arange(result.cross_spectra.shape[0])
    left, right = result.pair_indices[pairs].T
    denominator = np.sqrt(result.auto_spectra[np.ix_(rows, left, bins)])
    denominator *= np.sqrt(result.auto_spectra[np.ix_(rows, right, bins)])
    values = result.cross_spectra[np.ix_(rows, pairs, bins)].astype(np.complex128)
    if np.any(presence[..., None] & ~(np.isfinite(denominator) & (denominator > 0))):
        raise ValueError("Valid coherency bins must have finite positive auto-power.")
    # Missing pairs stay NaN and are excluded by the shared masked DF helpers.
    denominator[~presence] = 1
    values[~presence] = np.nan
    values /= denominator
    if np.any(presence[..., None] & ~np.isfinite(values)):
        raise ValueError("Valid coherency bins must contain finite values.")
    if np.any(np.abs(values) > 1 + 1e-8):
        raise ArithmeticError("Cross/auto reduction violated the coherence bound")
    if representation == "coherence":
        return np.minimum(np.abs(values) ** 2, 1.0)
    return np.clip(values.real, -1.0, 1.0)


def _stabilize_cross_covariance(covariance):
    """Return a PSD covariance while preserving the pooled marginal variances.

    Entrywise residual-df pooling can give an indefinite matrix when different
    pairs contribute to different entries. Clip negative eigenvalues of its
    correlation matrix, then rescale to unit diagonal and restore variances.
    Zero-variance bins remain zero; an already PSD matrix is returned unchanged.
    """
    variances = np.diag(covariance)
    bins = np.flatnonzero(variances > 0)
    if not bins.size:
        return covariance
    scales = np.sqrt(variances[bins])
    correlation = covariance[np.ix_(bins, bins)] / scales[:, None] / scales[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    tolerance = 64 * np.finfo(float).eps * len(bins) * np.max(np.abs(eigenvalues))
    if eigenvalues[0] >= -tolerance:
        return covariance
    correlation = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    diagonal = np.sqrt(np.diag(correlation))
    correlation /= diagonal[:, None]
    correlation /= diagonal[None, :]
    stabilized = np.zeros_like(covariance)
    stabilized[np.ix_(bins, bins)] = correlation * scales[:, None] * scales[None, :]
    np.fill_diagonal(stabilized, variances)
    return stabilized


def _cross_pvalues_from_covariance(observed, scales, covariance, weights, roundoff, n_observed):
    """Evaluate one bin group's contrasts using its shared covariance submatrix.

    ``observed``, ``scales`` and ``n_observed`` are per-pair arrays. The square
    ``covariance`` and sum-one ``weights`` use this group's retained bin axis;
    ``roundoff`` is the group's profile-scale tolerance. Return per-pair
    p-values/status and the weighted covariance's effective rank.
    """
    _, eigenvalues, effective_rank = _weighted_covariance_eigensystem(covariance, weights)
    pvals = _l2_pvalues_from_state(
        {
            "covariance": "pooled",
            "observed": observed,
            "sigma_log": covariance,
            "roundoff": roundoff,
            "freq_weights": weights,
            "n_obs": n_observed,
            "eigenvalues": scales[:, None] * eigenvalues,
            "eligible": np.ones(len(observed), dtype=bool),
            "contrast_scale": scales,
        }
    )
    return pvals, effective_rank, np.where(np.isnan(pvals), "degenerate_null", "ok")


def _run_cross_plugin(  # noqa: C901 — shared fits across covariance modes and design paths
    result,
    pair_presence,
    support,
    candidates,
    weights,
    *,
    representation,
    covariance_mode,
    use_masked,
    groups,
    prepared_design,
    min_samples_per_group,
    min_resid_df,
):
    """Fit eligible pairs with pooled or per-pair residual covariance.

    Parameters
    ----------
    result : CrossSpectra
        Cached spectra in original sample, pair, and bin order. The cross
        array has shape ``(n_samples, n_pairs, n_bins)``. Inputs are not modified.
    pair_presence : ndarray of bool, shape (n_samples, n_pairs)
        Sample inclusion mask: ``pair_presence[s, p]`` is True when both
        genes in pair p pass the expression-presence threshold in sample s.
    support : ndarray of bool, shape (n_pairs, n_bins)
        Bins valid in every retained sample.
        Polar bins retain the original radius-then-angle order.
    candidates : ndarray of int, shape (n_candidates,)
        Non-self pairs with supported bins that meet the binary group minimum, if applicable.
    weights : ndarray, shape (n_bins,)
        Finite positive frequency weights on the original bin axis.
        The DF helpers renormalize these weights over each retained bin set.
    representation : {'coherence', 'coherency'}
        Test log magnitude-squared coherency for ``coherence``, or its
        untransformed signed real part for ``coherency``.
    covariance_mode : {'pooled', 'per_pair'}
        Pool across eligible pairs or use each pair's residual covariance.
    use_masked : bool
        Whether any pair in the complete panel has an absent sample.
        This is a global dispatch decision.
    groups : ndarray, shape (n_samples,), or None
        Validated binary sample labels in the original sample order.
        Used when ``prepared_design`` is None; ignored for GLM contrasts.
    prepared_design : tuple or None
        Resolved GLM ``(matrix, column_names, contrast, scaled_fit)``
        for the full sample set, or None to select binary comparison.
        Unmasked fits reuse this inverse. Masked fits refit the matrix on
        each distinct retained-sample set and check contrast estimability.
    min_samples_per_group : int
        Minimum retained samples per arm for a binary comparison.
    min_resid_df : int
        Minimum residual degrees of freedom for masked GLM fits. Ignored
        for unmasked tests and binary comparisons.

    Returns
    -------
    dict of ndarray
        Owned arrays of shape ``(n_pairs,)`` in original pair order:
        ``observed`` (weighted L2 statistic), ``pvals``, ``pooled_pairs``,
        ``effective_rank``, ``df_resid``, and ``status``. Skipped fits retain
        NaN statistics; untestable p-values remain NaN. ``permutation_counts``
        is zero and ``exact_permutations`` is False on this plug-in path.
        ``pooled_pairs`` counts all eligible contributors to the shared matrix;
        the contributors to each individual entry depend on bin overlap.
        Per-pair fits count one contributor for each eligible pair, zero otherwise.
        ``estimate_per_bin`` contains signed contrasts on the original
        bin axis, with NaN for unsupported bins.

    Notes
    -----
    With pooled covariance, bin masks only batch the DF fits; they do not partition
    the covariance pool. Each entry pools residual cross-products from eligible
    pairs that support both bins, divided by their summed residual df. Entries
    with no joint support start at zero. One PSD correction of the shared
    matrix precedes subsetting and frequency weighting for each test.
    Per-pair mode instead uses the shared per-feature estimator and Liu tails
    directly. Its covariance is PSD by construction, without cross-pair repair.
    The existing DF fitting helpers retain their small-residual-df warnings.
    """
    per_pair = covariance_mode == "per_pair"
    estimates = np.full(support.shape, np.nan)
    n_samples, n_pairs, _ = result.cross_spectra.shape
    n_observed = pair_presence.sum(axis=0)
    use_glm = prepared_design is not None
    if use_glm:
        matrix, _, vector, fit = prepared_design
        df_resid = (
            np.zeros(n_pairs, dtype=int) if use_masked else np.full(n_pairs, n_samples - fit.rank)
        )
    else:
        df_resid = np.maximum(n_observed - 2, 0)
    observed = np.full(n_pairs, np.nan)
    pvals = np.full(n_pairs, np.nan)
    pooled_pairs = np.zeros(n_pairs, dtype=int)
    effective_rank = np.full(n_pairs, np.nan)
    status = np.full(n_pairs, "no_common_bins", dtype=object)
    if not per_pair:
        contrast_scale = np.full(n_pairs, np.nan)
        roundoff = np.zeros(n_pairs)
        pooled_bins = np.flatnonzero(support[candidates].any(axis=0))
        covariance = np.zeros((len(pooled_bins), len(pooled_bins)))
        covariance_df = np.zeros_like(covariance)
    # Batch fits by bin support; pooled mode combines shared covariance entries afterward.
    for keep, indices in _iter_mask_groups(support[candidates].T):
        pairs = candidates[indices]
        bins = np.flatnonzero(keep)
        group_presence = pair_presence[:, pairs]
        profiles = _prepare_cross_profiles(result, pairs, bins, representation, group_presence)
        group_weights = weights[bins]
        log_transform = representation == "coherence"
        kwargs = {"freq_weights": group_weights, "log_transform": log_transform}
        if per_pair:
            kwargs.update(covariance="per_gene", store_covariance=False)
        else:
            kwargs.update(covariance="pooled", for_pooling=True)
        if use_masked:
            if use_glm:
                state = _estimate_glm_masked_null_covariance(
                    profiles,
                    matrix,
                    vector,
                    group_presence,
                    min_resid_df=int(min_resid_df),
                    allow_empty=True,
                    **kwargs,
                )
            else:
                state = _estimate_two_group_masked_null_covariance(
                    profiles,
                    groups,
                    group_presence,
                    min_samples_per_group=int(min_samples_per_group),
                    allow_empty=True,
                    **kwargs,
                )
        elif use_glm:
            state = _fit_glm_profiles(profiles, prepared_design, **kwargs)
        else:
            state = _estimate_two_group_null_covariance(profiles, groups, **kwargs)

        # Record fit eligibility before calibrating this bin group.
        eligible = state["eligible"]
        selected = pairs[eligible]
        observed[pairs] = state["observed"]
        effect = state["estimate_per_bin"]
        estimates[np.ix_(pairs, bins)] = effect if use_glm else -effect
        df_resid[pairs] = state["df_resid"]
        status[pairs] = "ineligible_design" if use_glm else "insufficient_samples"
        if not eligible.any():
            continue
        if per_pair:
            pvals[pairs] = _l2_pvalues_from_state(state)
            effective_rank[selected] = state["effective_rank"][eligible]
            pooled_pairs[selected] = 1
            status[selected] = np.where(np.isnan(pvals[selected]), "degenerate_null", "ok")
            continue
        contrast_scale[selected] = np.broadcast_to(state["contrast_scale"], (len(pairs),))[eligible]
        positions = np.searchsorted(pooled_bins, bins)
        block = np.ix_(positions, positions)
        group_df = df_resid[selected].sum()
        covariance[block] += state["sigma_log"] * group_df
        covariance_df[block] += group_df
        roundoff[selected] = state["roundoff"]
        status[selected] = "ok"

    if not per_pair:
        np.divide(covariance, covariance_df, out=covariance, where=covariance_df > 0)
        eligible_pairs = np.flatnonzero(status == "ok")
        pooled_pairs[eligible_pairs] = len(eligible_pairs)
        # Subset only after PSD correction, so every test uses the same covariance.
        tested_groups = list(_iter_mask_groups(support[eligible_pairs].T))
        if len(tested_groups) > 1:
            covariance = _stabilize_cross_covariance(covariance)
        for keep, indices in tested_groups:
            pairs = eligible_pairs[indices]
            bins = np.flatnonzero(keep)
            positions = np.searchsorted(pooled_bins, bins)
            # subset the shared covariance to the current bin group and compute p-values.
            pair_covariance = covariance[np.ix_(positions, positions)]
            group_weights = _resolve_freq_weights(weights[bins], len(bins))
            pvals[pairs], effective_rank[pairs], status[pairs] = _cross_pvalues_from_covariance(
                observed[pairs],
                contrast_scale[pairs],
                pair_covariance,
                group_weights,
                roundoff[pairs].max(),
                n_observed[pairs],
            )
    return {
        "observed": observed,
        "pvals": pvals,
        "pooled_pairs": pooled_pairs,
        "estimate_per_bin": estimates,
        "effective_rank": effective_rank,
        "df_resid": df_resid,
        "status": status,
        "permutation_counts": np.zeros(n_pairs, dtype=int),
        "exact_permutations": np.zeros(n_pairs, dtype=bool),
    }


def _run_cross_permutations(
    result,
    pair_presence,
    support,
    candidates,
    weights,
    group_codes,
    *,
    representation,
    use_masked,
    n_perm,
    max_exact_permutations,
    rng,
):
    """Calibrate binary L2 contrasts by permuting sample labels.

    Shared input shapes, masks, representations and eligibility follow
    :func:`_run_cross_plugin`. Support and weights stay fixed across relabellings.

    Parameters
    ----------
    group_codes : ndarray of int, shape (n_samples,)
        Validated binary labels encoded as 0 and 1, in original sample order.
        Labels are subset by each pair's sample mask before permutation.
    n_perm : int
        Positive number of sampled label permutations when exact enumeration
        is not selected. Ignored when all assignments are enumerated.
    max_exact_permutations : int
        Enumerate all assignments when their count for the retained samples
        is at most this threshold. Zero forces sampling.
    rng : numpy.random.Generator
        Generator used for sampled relabellings. Masked pairs draw separately
        in original pair order; unmasked pairs share one label matrix across
        bin groups, matching DF.

    Returns
    -------
    dict of ndarray
        Arrays of shape ``(n_pairs,)`` in original pair order, with the same
        keys as :func:`_run_cross_plugin`. ``observed`` is weighted L2;
        ``pvals`` uses exact exceedance fractions or sampled tails with a +1
        correction. ``permutation_counts`` and ``exact_permutations`` record
        each pair's calibration. ``df_resid`` records binary-design residual df;
        ``pooled_pairs`` is zero and ``effective_rank`` is NaN. Skipped pairs
        retain NaN statistics and p-values. The caller assigns final statuses
        for self-pairs and entirely absent pairs.
    """
    n_pairs = result.cross_spectra.shape[1]
    estimates = np.full(support.shape, np.nan)
    observed = np.full(n_pairs, np.nan)
    pvals = np.full(n_pairs, np.nan)
    status = np.full(n_pairs, "no_common_bins", dtype=object)
    permutation_counts = np.zeros(n_pairs, dtype=int)
    exact_permutations = np.zeros(n_pairs, dtype=bool)
    # Match DF: masked features draw in original order; unmasked features share labels.
    batches = (np.array([pair]) for pair in candidates) if use_masked else (candidates,)
    for pairs in batches:
        if not pairs.size:
            continue
        sample_mask = pair_presence[:, pairs[0]]
        perm_labels, is_exact = _make_label_permutations(
            group_codes[sample_mask],
            n_perm,
            rng,
            max_exact_permutations=max_exact_permutations,
        )
        for keep, local in _iter_mask_groups(support[pairs].T):
            selected = pairs[local]
            bins = np.flatnonzero(keep)
            profiles = _prepare_cross_profiles(
                result, selected, bins, representation, pair_presence[:, selected]
            )
            observed[selected], pvals[selected], effect = _run_l2_permutations(
                profiles[sample_mask],
                group_codes[sample_mask],
                perm_labels,
                freq_weights=_resolve_freq_weights(weights[bins], len(bins)),
                is_exact=is_exact,
                log_transform=representation == "coherence",
            )
            permutation_counts[selected] = len(perm_labels)
            estimates[np.ix_(selected, bins)] = effect
            exact_permutations[selected] = is_exact
            status[selected] = "ok"
    return {
        "observed": observed,
        "pvals": pvals,
        "pooled_pairs": np.zeros(n_pairs, dtype=int),
        "effective_rank": np.full(n_pairs, np.nan),
        "df_resid": np.maximum(pair_presence.sum(axis=0) - 2, 0),
        "status": status,
        "permutation_counts": permutation_counts,
        "estimate_per_bin": estimates,
        "exact_permutations": exact_permutations,
    }


def _run_cross_rotations(
    result,
    pair_presence,
    support,
    candidates,
    freq_weights,
    *,
    representation,
    use_masked,
    groups,
    prepared_design,
    min_resid_df,
    integration_tol,
):
    """Calibrate weighted L2 contrasts with the conditional rotation null.

    Shared input shapes, sample/bin masks, representations and eligibility
    thresholds follow :func:`_run_cross_plugin`. ``prepared_design`` supplies
    the GLM matrix and contrast; the rotation engine refits retained-sample
    designs. Binary labels use an intercept and a group indicator.

    Parameters
    ----------
    freq_weights : array-like, optional
        Original bin weights. Keep their ratios intact when selecting supported bins.
    integration_tol : float
        Initial Davies absolute p-value error target for rotation; default 1e-6.
        Use at most 10,000 terms per call and one accuracy retry. Accepted
        integration also requires relative error budget <= 0.001. Unresolved
        tails use marked saddlepoint approximation. Ignored by other nulls.

    Returns
    -------
    dict
        Arrays of shape ``(n_pairs,)`` in original pair order, with the same
        keys as :func:`_run_cross_plugin`. ``observed`` is weighted L2 and
        ``pvals`` contains deterministic conditional tails.
        ``effective_rank`` describes each pair's nuisance-adjusted scatter.
        ``pooled_pairs`` and ``permutation_counts`` are zero;
        ``exact_permutations`` is False. Skipped pairs retain NaN statistics
        and p-values. ``extra["P_value_method"]`` identifies each evaluator;
        ``extra["P_value_error"]`` contains numerical error budgets/estimates,
        NaN for saddlepoint or skipped pairs. ``metadata`` records the
        integration method/tolerance and null/covariance labels.

    Notes
    -----
    Each bin-support group delegates to
    :func:`~sonic.comparators._inference._run_l2_conditional`, conditioning on
    each pair's nuisance-adjusted scatter without pooling across pairs.
    The caller assigns final statuses for self-pairs and entirely absent pairs.
    """
    n_samples, n_pairs = pair_presence.shape
    metadata = _integration_metadata(integration_tol)
    estimates = np.full(support.shape, np.nan)
    use_glm = prepared_design is not None
    if use_glm:
        matrix, _, vector, _ = prepared_design
    else:
        group_b = groups == np.unique(groups)[1]
        matrix = np.column_stack([np.ones(n_samples), group_b])
        vector = np.array([0.0, 1.0])
    observed = np.full(n_pairs, np.nan)
    pvals = np.full(n_pairs, np.nan)
    pvalue_error = np.full(n_pairs, np.nan)
    pvalue_method = np.full(n_pairs, None, dtype=object)
    effective_rank = np.full(n_pairs, np.nan)
    df_resid = np.zeros(n_pairs, int)
    status = np.full(n_pairs, "no_common_bins", dtype=object)
    for keep, indices in _iter_mask_groups(support[candidates].T):
        bins = np.flatnonzero(keep)
        for block in _iter_feature_blocks(n_samples, len(indices), len(bins)):
            pairs = candidates[indices[block]]
            group_presence = pair_presence[:, pairs]
            profiles = _prepare_cross_profiles(result, pairs, bins, representation, group_presence)
            rotation = _run_l2_conditional(
                profiles,
                matrix,
                vector,
                group_presence if use_masked else None,
                freq_weights=None if freq_weights is None else freq_weights[bins],
                log_transform=representation == "coherence",
                min_resid_df=min_resid_df if use_glm and use_masked else 1,
                allow_empty=True,
                integration_tol=integration_tol,
            )
            observed[pairs] = rotation["observed"]
            effect = rotation["estimate_per_bin"]
            estimates[np.ix_(pairs, bins)] = effect if use_glm else -effect
            pvals[pairs] = rotation["pvals"]
            pvalue_error[pairs] = rotation["pvalue_error"]
            pvalue_method[pairs] = rotation["pvalue_method"]
            effective_rank[pairs] = rotation["effective_rank"]
            df_resid[pairs] = rotation["df_resid"]
            status[pairs] = "ineligible_design" if use_glm else "insufficient_samples"
            status[pairs[rotation["eligible"]]] = "ok"
    metadata["n_saddlepoint"] = int(np.count_nonzero(pvalue_method == "saddlepoint"))
    return {
        "observed": observed,
        "pvals": pvals,
        "effective_rank": effective_rank,
        "df_resid": df_resid,
        "status": status,
        "pooled_pairs": np.zeros(n_pairs, int),
        "permutation_counts": np.zeros(n_pairs, int),
        "exact_permutations": np.zeros(n_pairs, bool),
        "metadata": metadata,
        "estimate_per_bin": estimates,
        "extra": {"P_value_error": pvalue_error, "P_value_method": pvalue_method},
    }


def _run_cross_t_cauchy(
    result,
    pair_presence,
    support,
    candidates,
    weights,
    *,
    representation,
    use_masked,
    groups,
    prepared_design,
    min_samples_per_group,
    min_resid_df,
):
    """Fit binwise t tests and combine their p-values for each pair.

    Shared input shapes, sample masks, representations and design eligibility
    follow :func:`_run_cross_plugin`. Binary labels select Welch tests;
    ``prepared_design`` selects OLS contrast t tests. Both use the shared
    :func:`~sonic.comparators.multisample._run_t_cauchy_test` engine.

    Parameters
    ----------
    support : ndarray of bool, shape (n_pairs, n_bins)
        Bins valid in every retained sample, in original flattened bin order.
        Pairs with no supported bins are skipped.
    weights : ndarray, shape (n_bins,)
        Strictly positive weights on the original bin axis. Cauchy
        combination renormalizes them over each pair's informative bins.

    Returns
    -------
    dict
        Arrays in original pair order, with the same keys as
        :func:`_run_cross_plugin`. ``observed`` is the maximum absolute t
        over supported bins; ``pvals`` contains Cauchy-combined tails.
        ``df_resid`` records design residual df, not the bin-specific Welch df.
        ``pooled_pairs`` and ``permutation_counts`` are zero,
        ``effective_rank`` is NaN, and ``exact_permutations`` is False.

        ``extra`` contains ``Estimate_per_bin``, signed ``T_per_bin``, and
        raw ``P_value_per_bin`` arrays of length ``n_bins`` for each pair.
        Unsupported bins and skipped pairs have NaN entries. ``Best_bin`` is
        a nullable, zero-based original bin index: the smallest p-value among
        informative bins, with ties taking the first index.
        It is missing when no bin qualifies. Skipped pair statistics and
        p-values remain NaN; the caller assigns final statuses for self-pairs
        and entirely absent pairs.
    """
    n_samples, n_pairs = pair_presence.shape
    n_bins = support.shape[1]
    use_glm = prepared_design is not None
    if use_glm:
        matrix, _, vector, fit = prepared_design
    else:
        codes = (groups == np.unique(groups)[1]).astype(int)
    observed = np.full(n_pairs, np.nan)
    pvals = np.full(n_pairs, np.nan)
    df_resid = np.zeros(n_pairs, int)
    status = np.full(n_pairs, "no_common_bins", dtype=object)
    binwise = {name: np.full((n_pairs, n_bins), np.nan) for name in _BINWISE_COLUMNS}
    best_bin = np.full(n_pairs, np.nan)
    unit_fits = {}
    for keep, indices in _iter_mask_groups(support[candidates].T):
        bins = np.flatnonzero(keep)
        for block in _iter_feature_blocks(n_samples, len(indices), len(bins)):
            pairs = candidates[indices[block]]
            presence = pair_presence[:, pairs]
            profiles = _prepare_cross_profiles(result, pairs, bins, representation, presence)
            observed[pairs], pvals[pairs], extra = _run_t_cauchy_test(
                profiles,
                groups=None if use_glm else codes,
                design=matrix if use_glm else None,
                contrast=vector if use_glm else None,
                prepared_design=prepared_design if not use_masked else None,
                unit_fits=unit_fits,
                presence=presence if use_masked else None,
                log_transform=representation == "coherence",
                freq_weights=weights[bins],
                min_samples_per_group=min_samples_per_group,
                min_resid_df=min_resid_df if use_masked else 1,
            )
            if use_glm:
                df_resid[pairs] = extra["df_resid"] if use_masked else n_samples - fit.rank
            else:
                df_resid[pairs] = np.maximum(presence.sum(axis=0) - 2, 0)
            status[pairs] = np.where(
                np.isfinite(pvals[pairs]),
                "ok",
                "ineligible_design" if use_glm else "insufficient_samples",
            )
            for name in _BINWISE_COLUMNS:
                binwise[name][np.ix_(pairs, bins)] = np.stack(extra[name])
            local_best = extra["Best_bin"].to_numpy(dtype=float, na_value=np.nan)
            eligible = np.isfinite(local_best)
            best_bin[pairs[eligible]] = bins[local_best[eligible].astype(int)]
    return {
        "observed": observed,
        "pvals": pvals,
        "df_resid": df_resid,
        "status": status,
        "pooled_pairs": np.zeros(n_pairs, int),
        "effective_rank": np.full(n_pairs, np.nan),
        "permutation_counts": np.zeros(n_pairs, int),
        "exact_permutations": np.zeros(n_pairs, bool),
        "extra": {
            **{name: list(values) for name, values in binwise.items()},
            "Best_bin": pd.array(best_bin, dtype="Int64"),
        },
    }


def _cross_comparison_frame(
    result,
    inference,
    pair_presence,
    support,
    weights,
    *,
    original_freq_weights,
    representation,
    statistic,
    null,
    covariance_mode,
    use_masked,
    design_metadata,
    count_columns,
):
    """Assemble pair results, apply the existing BH family, and attach metadata.

    ``inference`` contains the result arrays returned by either testing path.
    Rows are sorted only after all pair-indexed columns have been assembled;
    support and permutation metadata stay keyed by the original ordered pairs.
    """
    permutation_counts = inference["permutation_counts"]
    exact_permutations = inference["exact_permutations"]
    frame = (
        pd.DataFrame(
            {
                "Statistic": inference["observed"],
                "P_value": inference["pvals"],
                "Feature_1": [p[0] for p in result.pairs],
                "Feature_2": [p[1] for p in result.pairs],
                "N_samples": pair_presence.sum(axis=0),
                "N_bins": support.sum(axis=1),
                "N_pairs_pooled": inference["pooled_pairs"],
                "DF_resid": inference["df_resid"],
                "Effective_rank": inference["effective_rank"],
                "Status": inference["status"],
                **count_columns,
                **inference.get("extra", {}),
            },
        )
        .sort_values("Statistic", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    frame["P_adj"] = apply_bh_correction(frame["P_value"])
    # Keep original bin indices so downstream plots can recover each tested frequency range.
    frame.attrs.update(
        representation=representation,
        statistic=statistic,
        transformation="log(max(C, 1e-12))" if representation == "coherence" else "real(coherency)",
        null=null,
        covariance=(
            ("per_pair" if covariance_mode == "per_pair" else "pooled_across_valid_bins_psd")
            if null == "plugin"
            else None
        ),
        bin_indices={p: tuple(np.flatnonzero(support[i])) for i, p in enumerate(result.pairs)},
        freq_weights=tuple(weights),
        original_freq_weights=original_freq_weights,
        sample_indices={
            p: tuple(np.flatnonzero(pair_presence[:, i])) for i, p in enumerate(result.pairs)
        },
        n_permutations=(
            dict(zip(result.pairs, permutation_counts, strict=True))
            if use_masked and null == "permutation"
            else int(permutation_counts.max(initial=0))
        ),
        exact_permutations=(
            dict(zip(result.pairs, exact_permutations, strict=True))
            if use_masked and null == "permutation"
            else bool(exact_permutations.any())
        ),
        **design_metadata,
    )
    if statistic == "t_cauchy":
        frame.attrs["per_bin_test"] = "ols" if "contrast_vector" in design_metadata else "welch"
    if null == "rotation":
        frame.attrs.update(inference["metadata"])
    return frame


def _compare_cross_spectra(  # noqa: C901 — validation and dispatch over binary/GLM designs
    result,
    design,
    *,
    groups=None,
    contrast=None,
    representation="coherence",
    statistic="l2",
    null=None,
    n_perm=1000,
    random_state=None,
    freq_weights=None,
    max_exact_permutations=10000,
    min_samples_per_group=2,
    min_resid_df=1,
    covariance=None,
    integration_tol=1e-6,
):
    """Compare pairwise coupling profiles with the shared DF inference helpers.

    Parameters
    ----------
    result : CrossSpectra
        Cached cross-spectra, matching auto-powers, pair indices and validity
        masks. Pair arrays have shape ``(n_samples, n_pairs, n_bins)``. Sample
        order must match the design; the cache is never modified here.
    design : pandas.DataFrame or ndarray
        Design normalized by ``base._validate_design``. Used on the GLM path;
        DataFrames are encoded with an intercept, while numeric matrices are
        used as supplied. Ignored on the binary path without ``contrast``.
    groups : ndarray, optional
        Binary labels of length ``n_samples``. Without an explicit contrast,
        these select the two-group path. Binary L2 requires at least
        ``min_samples_per_group`` observations per arm for each pair.
        Otherwise ``design`` and ``contrast``
        determine the GLM test.
    contrast : str, dict or ndarray, optional
        One finite, nonzero, estimable GLM contrast. Required when ``groups``
        is None; supplying it selects the GLM path even with binary labels.
    representation : {'coherence', 'coherency'}, default 'coherence'
        Let ``k = S_xy / sqrt(S_xx * S_yy)`` using matching binned products.
        ``coherence`` tests ``Y = log(max(abs(k)**2, 1e-12))``;
        ``coherency`` tests ``Y = real(k)`` without a transformation.
    statistic : {'l2', 't_cauchy'}, default 'l2'
        Weighted L2 contrast or binwise t tests with Cauchy combination.
        t-Cauchy uses Welch for binary labels and OLS for a GLM, requires
        ``null='analytic'`` and ``covariance=None``, and returns bin diagnostics
        as documented in ``test_cross_diff_freq``.
    null : str, optional
        For L2, ``'rotation'`` (default) integrates each pair's conditional null.
        ``'plugin'`` uses Gaussian plug-in calibration with the estimated covariance.
        Both support GLM contrasts.
        ``'permutation'`` is available only for binary
        comparisons with group labels and no explicit contrast.
    covariance : {None, 'pooled', 'per_pair'}, optional
        For plug-in L2 calibration, ``None`` or ``'per_pair'`` uses each pair's residual
        covariance. ``'pooled'`` shares it across eligible pairs. Specifying a
        covariance selects plug-in calibration when ``null`` is omitted.
    n_perm : int, default 1000
        Number of sampled relabellings when exact enumeration is too large.
        Used only for sampled permutations.
    random_state : int, optional
        Seed for binary label permutations; ignored by other nulls.
    integration_tol : float
        Initial Davies absolute p-value error target for rotation; default 1e-6.
        Use at most 10,000 terms per call and one accuracy retry. Accepted
        integration also requires relative error budget <= 0.001. Unresolved
        tails use marked saddlepoint approximation. Ignored by other nulls.
    freq_weights : ndarray, optional
        Finite, strictly positive weights for the original bins. None gives uniform weights.
        Weights are renormalized over retained bins (informative bins for
        t-Cauchy). Choose weights independently of the observed signal.
    max_exact_permutations : int, default 10000
        Enumerate all binary label assignments if their number is at most
        this threshold. Zero forces sampling. Used only for permutation tests.
    min_samples_per_group : int, default 2
        Minimum observed samples per arm for binary L2 and masked binary t-Cauchy.
    min_resid_df : int, default 1
        Minimum residual degrees of freedom on the masked GLM path, as in DF.

    Returns
    -------
    pandas.DataFrame
        One row per requested pair, sorted by descending ``Statistic``.
        ``Feature_1`` and ``Feature_2`` identify the pair; ``Statistic`` is
        the weighted L2 contrast or maximum absolute t, with ``P_value`` and
        BH-adjusted ``P_adj``. Both statistics return unweighted, signed
        ``Estimate_per_bin`` on the tested scale. t-Cauchy adds signed t
        statistics, p-values, and the original ``Best_bin`` index.
        ``N_samples`` and ``N_bins`` record the support. ``DF_resid``,
        ``N_pairs_pooled`` and ``Effective_rank`` describe the plug-in fit;
        per-pair fits count one contributor per eligible pair, while
        permutations and t-Cauchy use zero pooled pairs and NaN effective rank.
        Rotation uses zero pooled pairs and the conditional scatter's effective
        rank. ``Status`` is ``ok``, ``self_pair``, ``no_common_bins``,
        ``insufficient_samples``, ``ineligible_design`` or ``degenerate_null``.
        Untestable rows retain NaN p-values. Binary tests also return
        ``n_obs_A`` and ``n_obs_B``. ``attrs`` records original sample/bin
        indices, weights, representation, null and design metadata. Permutation
        counts/exactness are per-pair dictionaries when presence is masked;
        otherwise they are shared scalars. Rotation adds ``P_value_method`` (``davies``, ``saddlepoint``, ``beta``
        or ``exact``; missing for skipped rows) and ``P_value_error``. The latter
        is the Davies accuracy budget, the beta interval half-width plus estimated
        roundoff, or zero at exact boundaries; it is NaN for saddlepoint/skipped
        rows and does not measure statistical uncertainty. ``attrs`` records
        ``method="davies"``, ``integration_tol``, ``integration_limit=10000``,
        ``n_saddlepoint`` and ``covariance=None``.

    Notes
    -----
    A sample is retained for a pair only when both genes are present, using
    the same presence threshold and masked inference as auto-spectrum DF.
    Bins must be valid in every retained sample.

    For transformed profiles Y, let d_p be the difference of group means or
    the fitted GLM contrast for pair p. For L2, the reported statistic is
    ``T_p = sqrt(d_p.T @ W @ d_p)``, with sum-one diagonal bin weights W.
    With ``covariance='pooled'``, the shared estimators fit each pair's retained-sample
    condition/nuisance design and pool each covariance entry as
    ``Sigma[j,k] = sum_p(E_p[:,j].T @ E_p[:,k]) / sum_p(nu_p)``, including
    only eligible pairs supporting both j and k in those sums.

    Note that this covariance is estimated from all eligible pairs even when
    their bin or sample masks differ. Individual covariance entries use only pairs
    supporting both bins. ``N_pairs_pooled`` counts all eligible pairs.
    Self-pairs are excluded from testing and pooling. The resulting
    covariance matrix may be indefinite. In this case, negative correlation
    eigenvalues are clipped and its diagonal rescaled to preserve pooled variances.

    The null of ``T_p**2`` is approximated using the eigenvalues of
    ``v_p * sqrt(W) @ Sigma[B_p,B_p] @ sqrt(W)``, where B_p are pair p's bins
    and ``v_p = 1/n_Ap + 1/n_Bp``
    or ``sum((contrast @ pinv(X_p))**2)`` for that pair's retained design.

    With ``covariance='per_pair'``, use ``S_p = E_p.T @ E_p / nu_p`` on
    each pair's retained samples and bins instead of ``Sigma[B_p,B_p]``.
    These matrices are PSD and may be singular; no covariance inverse or
    cross-pair PSD correction is needed. Both covariance choices are plug-in
    Liu approximations that ignore covariance-estimation uncertainty and assume
    comparable residual covariance across samples within a pair.

    Rotation uses :func:`~sonic.comparators.multisample.compare_glm`'s conditional
    null on the selected profile scale, separately for each retained-sample
    and bin-support group.

    Masked permutations draw separately per pair in original order, as in DF;
    unmasked pairs share labels. Support and weights stay fixed. Exact tails count
    exceedances over all assignments,
    while sampled tails use the usual +1 correction. There is no residual or
    blocked permutation scheme here. BH is applied once across finite pair
    p-values in this call, excluding untestable and self-pairs as in DF.
    """
    if statistic not in ("l2", "t_cauchy"):
        raise ValueError("statistic must be 'l2' or 't_cauchy'.")
    null = _resolve_profile_null(null, statistic, covariance)
    if statistic == "t_cauchy" and (null != "analytic" or covariance is not None):
        raise ValueError("statistic='t_cauchy' requires null='analytic' and covariance=None.")
    if representation not in ("coherence", "coherency"):
        raise ValueError("representation must be 'coherence' or 'coherency'.")
    if statistic == "l2" and null not in _L2_NULL_OPTIONS:
        raise ValueError(f"Unknown null='{null}' for L2. Options: {list(_L2_NULL_OPTIONS)}.")
    if covariance not in (None, "pooled", "per_pair"):
        raise ValueError("covariance must be None, 'pooled', or 'per_pair'.")
    if covariance is not None and null != "plugin":
        raise ValueError("covariance is only supported with null='plugin'.")
    covariance_mode = "per_pair" if covariance is None else covariance
    n_samples, _, n_bins = result.cross_spectra.shape
    if n_samples == 0 or n_bins == 0:
        raise ValueError("Cross-spectrum tests require samples and frequency bins.")
    weights = _resolve_freq_weights(freq_weights, n_bins)
    if freq_weights is not None:
        freq_weights = np.asarray(freq_weights, dtype=object).ravel()
    if int(min_samples_per_group) < 2:
        raise ValueError(f"min_samples_per_group must be >= 2, got {min_samples_per_group}.")
    if int(min_resid_df) < 1:
        raise ValueError(f"min_resid_df must be >= 1, got {min_resid_df}.")
    presence = np.asarray(result.presence, dtype=bool)
    if presence.shape != result.auto_spectra.shape[:2]:
        raise ValueError("presence must match the sample and selected-gene axes of auto_spectra.")
    left, right = result.pair_indices.T
    pair_presence = presence[:, left] & presence[:, right]
    use_masked = not pair_presence.all()
    n_observed = pair_presence.sum(axis=0)
    # Validate the full design, then let masked DF check each retained-sample fit.
    use_glm = contrast is not None or groups is None
    design_metadata = {}
    count_columns = {}
    prepared_design = None
    if use_glm:
        if null == "permutation":
            raise NotImplementedError(
                "Only null='plugin' or 'rotation' is supported for L2 with a design matrix or explicit contrast; "
                "permutations require binary labels without contrast=."
            )
        if contrast is None:
            raise ValueError("test_cross_diff_freq() requires contrast= for a design matrix.")
        if use_masked:
            matrix, columns = _build_design_matrix(design, n_samples)
            vector = _resolve_contrast(contrast, columns)
            prepared_design = matrix, columns, vector, _decompose_glm_design(matrix, vector)
        else:
            prepared_design = _prepare_glm_design(design, contrast, n_samples)
            columns, vector = prepared_design[1:3]
        design_metadata = {"design_columns": tuple(columns), "contrast_vector": tuple(vector)}
    else:
        groups = np.asarray(groups)
        if groups.shape != (n_samples,) or pd.isna(groups).any():
            raise ValueError("Binary labels must be nonmissing and match the sample count.")
        labels, counts = np.unique(groups, return_counts=True)
        if len(labels) != 2:
            raise ValueError("groups must contain exactly two distinct values.")
        group_codes = (groups == labels[1]).astype(int)
        n_a = pair_presence[group_codes == 0].sum(axis=0)
        n_b = pair_presence[group_codes == 1].sum(axis=0)
        count_columns = {"n_obs_A": n_a, "n_obs_B": n_b}
        design_metadata = {"group_labels": tuple(labels), "group_sizes": tuple(counts)}

    rng = np.random.default_rng(random_state) if null == "permutation" else None
    if null == "permutation":
        if not isinstance(n_perm, (int, np.integer)) or n_perm < 1:
            raise ValueError("n_perm must be a positive integer.")
        if not isinstance(max_exact_permutations, (int, np.integer)) or max_exact_permutations < 0:
            raise ValueError("max_exact_permutations must be a nonnegative integer.")

    # Missing expression excludes samples; unsupported frequencies still exclude bins.
    support = (result.coherency_valid | ~pair_presence[..., None]).all(axis=0)
    support &= pair_presence.any(axis=0)[:, None]
    self_pairs = np.array([a == b for a, b in result.pairs], dtype=bool)
    testable = ~self_pairs & support.any(axis=1)
    binary_l2 = statistic == "l2" and not use_glm
    if binary_l2:
        insufficient = (n_a < min_samples_per_group) | (n_b < min_samples_per_group)
        testable &= ~insufficient
    candidates = np.flatnonzero(testable)
    if statistic == "t_cauchy":
        inference = _run_cross_t_cauchy(
            result,
            pair_presence,
            support,
            candidates,
            weights,
            representation=representation,
            use_masked=use_masked,
            groups=groups,
            prepared_design=prepared_design,
            min_samples_per_group=min_samples_per_group,
            min_resid_df=min_resid_df,
        )
    elif null == "permutation":
        inference = _run_cross_permutations(
            result,
            pair_presence,
            support,
            candidates,
            weights,
            group_codes,
            representation=representation,
            use_masked=use_masked,
            n_perm=n_perm,
            max_exact_permutations=max_exact_permutations,
            rng=rng,
        )
    elif null == "rotation":
        inference = _run_cross_rotations(
            result,
            pair_presence,
            support,
            candidates,
            freq_weights,
            representation=representation,
            use_masked=use_masked,
            groups=groups,
            prepared_design=prepared_design,
            min_resid_df=min_resid_df,
            integration_tol=integration_tol,
        )
    else:
        inference = _run_cross_plugin(
            result,
            pair_presence,
            support,
            candidates,
            weights,
            representation=representation,
            covariance_mode=covariance_mode,
            use_masked=use_masked,
            groups=groups,
            prepared_design=prepared_design,
            min_samples_per_group=min_samples_per_group,
            min_resid_df=min_resid_df,
        )
    inference["status"][n_observed == 0] = (
        "ineligible_design" if use_glm else "insufficient_samples"
    )
    if binary_l2:
        inference["status"][insufficient & support.any(axis=1)] = "insufficient_samples"
        inference["df_resid"] = n_observed - (n_a > 0) - (n_b > 0)
    inference["status"][self_pairs] = "self_pair"
    if statistic == "l2":
        inference.setdefault("extra", {})["Estimate_per_bin"] = list(inference["estimate_per_bin"])
    return _cross_comparison_frame(
        result,
        inference,
        pair_presence,
        support,
        weights,
        original_freq_weights=None if freq_weights is None else tuple(freq_weights),
        representation=representation,
        statistic=statistic,
        null=null,
        covariance_mode=covariance_mode,
        use_masked=use_masked,
        design_metadata=design_metadata,
        count_columns=count_columns,
    )
