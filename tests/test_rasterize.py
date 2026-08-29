"""Unit tests for :mod:`sonic._rasterize`.

The two helpers in ``_rasterize.py`` do thin, testable work:

- :func:`ensure_csc_table` — coerces a table's ``X`` to CSC if it is sparse
  and a different format; leaves dense / already-CSC arrays untouched; raises
  on missing table names.
- :func:`rasterize_table` — forwards to :func:`spatialdata.rasterize_bins`
  after the coercion step.

The detector-side integration is exercised in ``test_detector_grid.py`` and
the comparator-side in ``test_comparators.py``; this file pins down the
helpers directly so regressions surface at this lower layer.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
import scipy.sparse as sp
import xarray as xr

from sonic._rasterize import _mean_fill_missing, ensure_csc_table, rasterize_table


class _FakeTable:
    def __init__(self, X, obs=None):
        self.X = X
        self.obs = {} if obs is None else obs


class _FakeSData:
    def __init__(self, tables):
        self.tables = tables


class TestMeanFillMissing:
    def test_fills_in_place_with_each_feature_mean(self):
        values = np.array(
            [
                [[0.0, np.nan], [2.0, 4.0]],
                [[np.nan, 1.0], [3.0, 5.0]],
            ]
        )

        out = _mean_fill_missing(values, axis=(1, 2))

        assert out is values
        np.testing.assert_allclose(
            values,
            np.array(
                [
                    [[0.0, 2.0], [2.0, 4.0]],
                    [[3.0, 1.0], [3.0, 5.0]],
                ]
            ),
        )


class TestEnsureCscTable:
    def test_raises_on_missing_table(self):
        sdata = _FakeSData({})
        with pytest.raises(ValueError, match="not found"):
            ensure_csc_table(sdata, "missing")

    def test_dense_ndarray_untouched(self):
        X = np.arange(12, dtype=np.float64).reshape(4, 3)
        table = _FakeTable(X)
        sdata = _FakeSData({"t": table})
        ensure_csc_table(sdata, "t")
        # Same object, unchanged.
        assert table.X is X

    def test_none_x_is_noop(self):
        table = _FakeTable(None)
        sdata = _FakeSData({"t": table})
        ensure_csc_table(sdata, "t")
        assert table.X is None

    def test_csr_converted_to_csc(self):
        X = sp.random(8, 5, density=0.3, format="csr", random_state=0)
        table = _FakeTable(X)
        sdata = _FakeSData({"t": table})
        ensure_csc_table(sdata, "t")
        assert sp.issparse(table.X) and table.X.format == "csc"
        # Values preserved.
        np.testing.assert_allclose(table.X.toarray(), X.toarray())

    def test_already_csc_not_rewritten(self):
        X = sp.random(8, 5, density=0.3, format="csc", random_state=0)
        table = _FakeTable(X)
        sdata = _FakeSData({"t": table})
        ensure_csc_table(sdata, "t")
        # Same object — no needless conversion.
        assert table.X is X


class TestRasterizeTable:
    def test_forwards_kwargs_and_coerces_first(self):
        X = sp.random(9, 4, density=0.25, format="csr", random_state=1)
        table = _FakeTable(
            X,
            obs={
                "row": np.repeat(np.arange(3), 3),
                "col": np.tile(np.arange(3), 3),
            },
        )
        sdata = _FakeSData({"tbl": table})
        sentinel = xr.DataArray(np.zeros((4, 3, 3)), dims=("c", "y", "x"))

        with patch("sonic._rasterize.sd.rasterize_bins", return_value=sentinel) as mock_rb:
            out = rasterize_table(
                sdata,
                bins="bins",
                table_name="tbl",
                col_key="col",
                row_key="row",
                value_key=None,
                return_region_as_labels=False,
            )

        # Coercion happened before the call.
        assert table.X.format == "csc"
        # Forwarded exactly once with the right kwargs.
        mock_rb.assert_called_once_with(
            sdata,
            bins="bins",
            table_name="tbl",
            col_key="col",
            row_key="row",
            value_key=None,
            return_region_as_labels=False,
        )
        assert out is sentinel

    def test_structural_holes_are_nan_but_observed_zeros_remain_zero(self):
        table = _FakeTable(
            np.zeros((3, 2)),
            obs={"row": np.array([0, 0, 1]), "col": np.array([0, 1, 1])},
        )
        sdata = _FakeSData({"tbl": table})
        raw = da.from_array(
            np.array(
                [
                    [[0.0, 2.0], [0.0, 4.0]],
                    [[5.0, 6.0], [0.0, 8.0]],
                ]
            ),
            chunks=(1, 2, 2),
        )
        sentinel = xr.DataArray(
            raw,
            dims=("c", "y", "x"),
            attrs={"transformations": {"global": "sentinel"}},
        )

        with patch("sonic._rasterize.sd.rasterize_bins", return_value=sentinel):
            out = rasterize_table(
                sdata,
                bins="bins",
                table_name="tbl",
                col_key="col",
                row_key="row",
            )

        assert isinstance(out.data, da.Array)
        assert out.attrs == sentinel.attrs
        assert out.sel(c=0, y=0, x=0).compute().item() == 0.0
        assert np.isnan(out.sel(c=0, y=1, x=0).compute().item())
        assert np.isnan(out.sel(c=1, y=1, x=0).compute().item())

    def test_label_raster_is_returned_unchanged(self):
        table = _FakeTable(np.zeros((1, 1)))
        sdata = _FakeSData({"tbl": table})
        sentinel = types.SimpleNamespace(data=np.zeros((2, 2), dtype=int))

        with patch("sonic._rasterize.sd.rasterize_bins", return_value=sentinel):
            out = rasterize_table(
                sdata,
                bins="bins",
                table_name="tbl",
                col_key="col",
                row_key="row",
                return_region_as_labels=True,
            )

        assert out is sentinel

    def test_raises_on_missing_table(self):
        sdata = _FakeSData({})
        with pytest.raises(ValueError, match="not found"):
            rasterize_table(
                sdata,
                bins="bins",
                table_name="nope",
                col_key="col",
                row_key="row",
            )
