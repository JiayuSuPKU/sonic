"""
Shared :func:`spatialdata.rasterize_bins` wrappers used by
:class:`sonic.DetectorGrid` and :class:`sonic.ComparatorGrid`.

Both consumers need the same boilerplate:

1. Coerce the table's X matrix to CSC sparse (required by ``rasterize_bins``).
2. Forward to ``spatialdata.rasterize_bins`` with the user-supplied keys.
3. Restore structurally absent grid positions as ``NaN``; SpatialData emits
   zeros for both absent bins and observed zero-valued bins.

Callers differ only in what they do with the rasterized image afterwards —
``DetectorGrid`` stashes it back into ``sdata.images`` under a derived key,
``ComparatorGrid`` extracts the array and reindexes the gene axis — so the
shared helper stops after restoring the raster's structural-missing mask.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import spatialdata as sd

__all__ = ["ensure_csc_table", "rasterize_table"]


def _mean_fill_missing(values: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    """Mean-fill ``NaN`` bins in float64, reusing float64 inputs in place."""
    values = np.asarray(values, dtype=np.float64)
    missing = np.isnan(values)
    if not missing.any():
        return values
    # Detect constants before filling: even float64 summation can perturb them.
    minimum = np.fmin.reduce(values, axis=axis, keepdims=True)
    constant = minimum == np.fmax.reduce(values, axis=axis, keepdims=True)
    n_observed = np.prod([values.shape[i] for i in axis]) - missing.sum(axis=axis, keepdims=True)
    np.copyto(values, 0.0, where=missing)
    sums = values.sum(axis=axis, keepdims=True)
    means = np.full_like(sums, np.nan)
    np.divide(sums, n_observed, out=means, where=n_observed > 0)
    np.copyto(means, minimum, where=constant)
    np.copyto(values, means, where=missing)
    return values


def ensure_csc_table(sdata: Any, table_name: str) -> None:
    """Coerce ``sdata.tables[table_name].X`` to CSC sparse in-place, if sparse.

    ``spatialdata.rasterize_bins`` performs column-wise slicing and requires CSC.
    Dense arrays are left untouched.
    """
    if table_name not in sdata.tables:
        raise ValueError(f"Table {table_name!r} not found in sdata.")
    table = sdata.tables[table_name]
    X = getattr(table, "X", None)
    if X is None or isinstance(X, np.ndarray):
        return
    if sp.issparse(X) and X.format != "csc":
        table.X = X.tocsc()


def rasterize_table(
    sdata: Any,
    *,
    bins: str,
    table_name: str,
    col_key: str,
    row_key: str,
    value_key: str | list[str] | None = None,
    return_region_as_labels: bool = False,
):
    """Rasterize a table while preserving structural missingness as ``NaN``.

    ``spatialdata.rasterize_bins`` initializes its bounding rectangle with
    zeros, making an absent bin indistinguishable from an observed zero. This
    wrapper reconstructs occupancy from ``row_key`` / ``col_key`` and masks
    only absent positions. The mask is applied lazily to image outputs; label
    outputs are returned unchanged.
    """
    ensure_csc_table(sdata, table_name)
    rasterized = sd.rasterize_bins(
        sdata,
        bins=bins,
        table_name=table_name,
        col_key=col_key,
        row_key=row_key,
        value_key=value_key,
        return_region_as_labels=return_region_as_labels,
    )
    if return_region_as_labels:
        return rasterized

    table = sdata.tables[table_name]
    rows = np.asarray(table.obs[row_key])
    cols = np.asarray(table.obs[col_key])
    occupied = np.zeros(rasterized.shape[-2:], dtype=bool)
    occupied[(rows - rows.min()).astype(int), (cols - cols.min()).astype(int)] = True
    return rasterized if occupied.all() else rasterized.where(occupied)
