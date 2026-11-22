"""Per-bin spatial explanations on a comparator's fitted frequency geometry."""

from __future__ import annotations

import finufft
import numpy as np
from scipy.fft import ifft2
from scipy.sparse import issparse

from sonic.comparators.binning import (
    _mode_pair_counts,
    _polar_bin_reducer,
    _polar_reducer_workspace_bytes,
    _radial_bin_reducer,
)
from sonic.comparators.features import _center_observations, _shared_observation_mask
from sonic.explanation import _explanation_from_responses
from sonic.kernels.fft import _fourier_coefficients_2d
from sonic.kernels.nufft import _fourier_coefficients_2d_nufft
from sonic.utils import _gene_indices, _parse_memory_budget, _take_feature_columns


def _check_spectrum_budget(comparator, sample_index, n_features, n_maps, budget):
    """Estimate return arrays, their construction copies, reducers and transforms."""
    shape = comparator._grid_shapes[sample_index]
    polar = comparator.feature_mode == "2d"
    irregular = comparator._spectrum_backend == "nufft"
    n_modes = int(np.prod(np.asarray(shape) + 2 * int(polar and irregular)))
    n_locations = len(comparator._coords[sample_index]) if irregular else int(np.prod(shape))
    estimate = 8 * n_locations * (2 * n_maps * n_features + 3 * n_maps + 5 * n_features + 8)
    estimate += 128 * n_modes * (n_features + 1)
    if polar:
        stored, build_peak = _polar_reducer_workspace_bytes(
            shape,
            edges=comparator.freq_edges,
            spacing=comparator._spacings[sample_index],
            n_theta=comparator._n_theta_bins,
        )
        estimate = max(build_peak, estimate + stored)
    else:
        estimate += 64 * n_modes
    if estimate > budget:
        raise MemoryError(
            f"Estimated explanation workspace {estimate} bytes exceeds budget {budget} bytes."
        )


def _explain_spectrum(  # noqa: C901 — validation, backend adjoints and five related allocations
    comparator, features, *, sample_index, bands, representation, memory_budget_bytes
):
    if comparator.spectra_ is None and comparator.cross_spectra_ is None:
        raise RuntimeError(
            "Call compute_spectra() or compute_cross_spectra() before explaining bins."
        )
    names = (features,) if isinstance(features, str) else tuple(features)
    if len(names) not in (1, 2) or any(not isinstance(g, str) for g in names):
        raise ValueError("Provide one feature name or a pair of feature names.")
    columns = _gene_indices(comparator.gene_names, names)
    choices = ("power", "power_norm") if len(names) == 1 else ("cross", "coherency", "coherence")
    if representation is None:
        representation = choices[0]
    if representation not in choices:
        raise ValueError(f"For {len(names)} feature(s), representation must be one of {choices}.")
    if (
        isinstance(sample_index, (bool, np.bool_))
        or not isinstance(sample_index, (int, np.integer))
        or not 0 <= sample_index < len(comparator.samples)
    ):
        raise ValueError("sample_index must be an integer indexing an existing sample.")
    polar = comparator.feature_mode == "2d"
    n_theta = comparator._n_theta_bins if polar else 1
    n_bins = len(comparator.frequency_bin_intervals_) * n_theta
    bands = np.asarray(bands)
    if (
        bands.ndim != 1
        or bands.size == 0
        or not np.issubdtype(bands.dtype, np.integer)
        or np.any((bands < 0) | (bands >= n_bins))
        or len(np.unique(bands)) != len(bands)
    ):
        raise ValueError("bands must be distinct integer indices into the retained spectrum bins.")
    _check_spectrum_budget(
        comparator, sample_index, len(names), len(bands), _parse_memory_budget(memory_budget_bytes)
    )

    shape = comparator._grid_shapes[sample_index]
    spacing = comparator._spacings[sample_index]
    irregular = comparator._spectrum_backend == "nufft"
    padding = int(polar and irregular)
    mode_shape = tuple(np.asarray(shape) + 2 * padding)
    angle = float(comparator.rotation_angles_[sample_index]) if polar else 0.0
    if polar:
        reducer, counts = _polar_bin_reducer(
            shape,
            edges=comparator.freq_edges,
            spacing=spacing,
            n_theta=n_theta,
            angle_deg=angle,
            frequency_padding=padding,
        )
    else:
        reducer, _ = _radial_bin_reducer(
            shape, edges=comparator.freq_edges, spacing=spacing, fft_solver="fft2"
        )
        counts = _mode_pair_counts(reducer, shape, periodic=not irregular)
    retained = np.repeat(comparator._supported_frequency_bin_mask, n_theta)
    reducer, counts = reducer[retained], counts[retained]
    if np.any(counts[bands] == 0):
        raise ValueError("Requested bins have no supported non-DC Fourier modes.")

    if irregular:
        sample = comparator.samples[sample_index]
        matrix = sample.X if comparator._layer is None else sample.layers[comparator._layer]
        selected = _take_feature_columns(matrix, columns)
        values = np.array(selected.toarray() if issparse(selected) else selected, dtype=float)
        coordinates = comparator._coords[sample_index]
        if not np.isfinite(values).all() or not np.isfinite(coordinates).all():
            raise ValueError("Values and coordinates must be finite on common sample support.")
        indices = np.arange(len(values))
        mask = np.ones(len(values), dtype=bool)
        coordinate_system = "source"
        unit_scale = comparator._unit_scales[sample_index]
    else:
        raster = comparator._rasterize_one(comparator.samples[sample_index])
        if tuple(raster.shape[-2:]) != tuple(shape):
            raise ValueError("Raster geometry changed; recompute spectra before explaining bins.")
        block = comparator._batch_raster(raster, columns).reshape(len(names), -1).T
        mask = _shared_observation_mask(block)
        indices = np.flatnonzero(mask)
        values = block[mask].copy()
        coordinates = np.column_stack(np.unravel_index(indices, shape)) * np.asarray(spacing)
        coordinate_system = "local_raster_yx"
        unit_scale = 1.0
    n = len(values)
    if n < 2:
        raise ValueError("Each sample must have at least two observed locations.")
    prepared = values.copy()
    _center_observations(prepared)
    if irregular:
        coefficients = _fourier_coefficients_2d_nufft(
            coordinates,
            prepared,
            shape,
            spacing,
            unit_scale=unit_scale,
            eps=comparator._nufft_eps,
            nthreads=comparator._workers,
            frequency_padding=padding,
        )
        # Padding extends the mode lattice, not the original physical domain.
        scaled = np.asarray(coordinates, dtype=float) * unit_scale
        scaled -= scaled.mean(axis=0)
        scaled *= 2 * np.pi / (np.asarray(shape) * spacing)
        y_scaled, x_scaled = np.ascontiguousarray(scaled.T)
    else:
        filled = np.zeros((int(np.prod(shape)), len(names)))
        filled[mask] = prepared
        coefficients = _fourier_coefficients_2d(
            filled.reshape(*shape, len(names)), "fft2", comparator._workers, center=True
        )
    coefficients[0, 0] = 0
    coefficients = np.moveaxis(coefficients, -1, 0).reshape(len(names), -1) / n
    auto = (reducer @ (abs(coefficients) ** 2).T).T
    auto[:, counts == 0] = np.nan
    cross = None
    if len(names) == 2:
        cross = reducer @ (coefficients[0] * coefficients[1].conj())
        if not polar:
            cross = cross.real
    if representation in ("coherency", "coherence"):
        peak = np.max(np.where(np.isfinite(auto), auto, 0), axis=1, keepdims=True)
        present = np.count_nonzero(values, axis=0) / n >= comparator._presence_threshold
        valid = (np.isfinite(auto) & (auto > 1e-12 * peak) & (auto > 0)).all(axis=0)
        valid &= (counts >= 2) & present.all() & np.isfinite(cross)
        if not valid[bands].all():
            raise ValueError(
                "Requested bins have invalid coherency: check presence, mode support and auto-power."
            )
    total_power = auto[0].sum() if representation == "power_norm" else None
    if total_power is not None and not np.isfinite(total_power):
        raise ValueError("Normalized power requires finite power in every retained bin.")

    filtered = np.empty((len(bands), n, len(names)))
    scores = np.empty(len(bands))
    for output, band in enumerate(bands):
        weights = reducer.getrow(band).toarray().ravel()
        weighted = (coefficients * weights).reshape(len(names), *mode_shape)
        if irregular:
            response = (
                finufft.nufft2d2(
                    y_scaled,
                    x_scaled,
                    weighted,
                    eps=comparator._nufft_eps,
                    isign=1,
                    modeord=1,
                    nthreads=comparator._workers,
                ).T
                / n
            )
        else:
            response = ifft2(weighted, axes=(-2, -1), workers=comparator._workers)
            response = response.reshape(len(names), -1).T[mask] * (np.prod(shape) / n)
        response -= response.mean(axis=0)
        if representation in ("power", "power_norm"):
            denominator = 1.0 if representation == "power" else total_power
            filtered[output] = response.real / denominator if denominator > 0 else 0
            scores[output] = auto[0, band] / denominator if denominator > 0 else 0
        elif representation == "cross":
            filtered[output] = response.real
            scores[output] = cross[band].real
        else:
            denominator = np.sqrt(auto[0, band]) * np.sqrt(auto[1, band])
            coherency = cross[band] / denominator
            if abs(coherency) > 1 + 1e-8:
                raise ArithmeticError("Cross/auto reduction violated the coherence bound.")
            response /= denominator
            if representation == "coherence":
                # Ordered S = Fx conj(Fy): its two adjoint allocations are conjugates.
                response[:, 0] *= np.conj(coherency)
                response[:, 1] *= coherency
            filtered[output] = response.real
            scores[output] = (
                abs(coherency) ** 2 if representation == "coherence" else coherency.real
            )

    return _explanation_from_responses(
        values,
        prepared,
        filtered,
        scores,
        feature_names=names,
        observation_indices=indices,
        coordinates=coordinates,
        metadata={
            "statistic": representation,
            "sample_index": int(sample_index),
            "bands": bands.copy(),
            "frequency_bin_intervals": comparator.frequency_bin_intervals_[bands // n_theta].copy(),
            "angle_centers_rad": (bands % n_theta) * np.pi / n_theta if polar else None,
            "rotation_angle_deg": angle,
            "feature_mode": comparator.feature_mode,
            "spectrum_source": "recomputed_raw",
            "normalization": "observed_fixed"
            if representation in ("power_norm", "coherency", "coherence")
            else "none",
            "coordinate_system": coordinate_system,
            "unit_scale": unit_scale,
        },
        rtol=max(1e-9, 50 * comparator._nufft_eps) if irregular else 1e-9,
    )
