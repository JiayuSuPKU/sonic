"""Synthetic local checks for spatial explanation arrays and detector adapters."""

from dataclasses import fields
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from sonic import DetectorGrid, DetectorIrregular, MatrixKernel
from sonic.explanation import SpatialExplanation, _explanation_from_responses
from tests.test_detector_grid import MockDataArray, MockSpatialData, MockTable


def _adata(sparse=False, n=42):
    rng = np.random.default_rng(721)
    values = rng.uniform(0.2, 2, (n, 3))
    data = ad.AnnData(sp.csr_matrix(values) if sparse else values)
    data.var_names = ["a", "b", "c"]
    data.obsm["positions"] = rng.uniform(0, 7, (n, 2))
    return data


def _detector(data, backend="matrix", method="gaussian"):
    options = {"bandwidth": 1.3} if method == "gaussian" else {}
    if backend == "nufft":
        options.update(grid_shape=(9, 11), spacing=(1, 1), eps=1e-12)
    return DetectorIrregular(kernel_method=method, backend=backend, **options).setup_data(
        data, obsm_key="positions"
    )


def _grid(monkeypatch, data, *, solver="fft2", method="gaussian", **kwargs):
    names = ["a", "b", "c"][: len(data)]
    raster = MockDataArray(data, names)
    table = MockTable(np.nan_to_num(data.reshape(len(data), -1).T), names)
    sdata = MockSpatialData("table", table)
    monkeypatch.setattr("sonic._rasterize.rasterize_table", lambda *args, **kw: raster)
    options = {"bandwidth": 1.1} if method == "gaussian" else {}
    options.update(kwargs)
    detector = DetectorGrid(kernel_method=method, fft_solver=solver, workers=1, **options)
    detector.setup_data(sdata, bins="bins", table_name="table", row_key="row", col_key="col")
    return detector


def _explain(detector, statistic):
    return detector.explain_qstat("a") if statistic == "Q" else detector.explain_rstat("a", "b")


def _reference(detector, statistic, source="var", layer=None):
    options = {"workers": 1, "return_pval": False, "show_progress": False}
    if isinstance(detector, DetectorIrregular):
        options.update(source=source, layer=layer, n_jobs=1)
    elif statistic == "Q":
        options["n_jobs"] = 1
    result = (
        detector.compute_qstat(features=["a"], **options)
        if statistic == "Q"
        else detector.compute_rstat(["a"], ["b"], **options)
    )
    return result.iloc[0][statistic]


@pytest.fixture(params=["matrix", "nufft", "fft"])
def detector(request, monkeypatch):
    if request.param == "fft":
        return _grid(monkeypatch, np.random.default_rng(3).normal(size=(3, 9, 11)))
    return _detector(_adata(), request.param)


class TestSpatialExplanation:
    @pytest.mark.parametrize("pair", [False, True])
    def test_dense_reference_and_array_ownership(self, pair):
        rng = np.random.default_rng(10)
        values = rng.normal(size=(16, 2 if pair else 1))
        z = (values - values.mean(0)) / values.std(0, ddof=1)
        matrices = rng.normal(size=(3, 16, 16))
        matrices += matrices.transpose(0, 2, 1)
        response = np.einsum("bij,jg->big", matrices, z)
        expected = np.einsum("i,bij,j->b", z[:, 0], matrices, z[:, -1])
        coords = rng.normal(size=(16, 2))
        indices = np.arange(16)
        explanation = _explanation_from_responses(
            values,
            z,
            response,
            expected,
            feature_names=("a", "b") if pair else ("a",),
            observation_indices=indices,
            coordinates=coords,
            metadata={"statistic": "R" if pair else "Q"},
        )
        expected_maps = z[:, 0] * response[:, :, 0]
        if pair:
            expected_maps = (z[:, 0] * response[:, :, 1] + z[:, 1] * response[:, :, 0]) / 2
        np.testing.assert_allclose(explanation.contributions, expected_maps)
        np.testing.assert_allclose(explanation.scores, expected)
        assert {f.name for f in fields(SpatialExplanation)} == {
            "feature_names",
            "observation_indices",
            "coordinates",
            "values",
            "filtered",
            "contributions",
            "metadata",
        }
        values[:] = 0
        response[:] = 0
        coords[:] = 0
        indices[:] = 0
        assert (
            explanation.values.any()
            and explanation.filtered.any()
            and explanation.coordinates.any()
        )
        np.testing.assert_array_equal(explanation.observation_indices, np.arange(16))
        explanation.contributions *= 2
        np.testing.assert_allclose(explanation.scores, 2 * expected)

    @pytest.mark.parametrize("corruption", ["reference", "nan", "shape", "indices", "coords"])
    def test_invalid_inputs_and_score_mismatch(self, corruption):
        rng = np.random.default_rng(140)
        values = rng.normal(size=(9, 1))
        response = values[None].copy()
        expected = np.array([np.sum(values**2)])
        indices = np.arange(9)
        coords = np.ones((9, 2))
        error = ValueError
        if corruption == "reference":
            expected *= 2
            error = ArithmeticError
        elif corruption == "nan":
            response[0, 0] = np.nan
        elif corruption == "shape":
            response = response[:, :-1]
        elif corruption == "indices":
            indices = indices.astype(float)
        else:
            coords[0, 0] = np.nan
        with pytest.raises(error):
            _explanation_from_responses(
                values,
                values,
                response,
                expected,
                feature_names=("a",),
                observation_indices=indices,
                coordinates=coords,
                metadata={},
            )

    def test_complex_fields_rejected(self):
        values = np.ones((4, 1))
        with pytest.raises(ValueError, match="real values"):
            _explanation_from_responses(
                values,
                values,
                (1j * values)[None],
                [0],
                feature_names=("a",),
                observation_indices=np.arange(4),
                metadata={},
            )


class TestIrregularExplanation:
    @pytest.mark.parametrize("backend", ["matrix", "nufft"])
    @pytest.mark.parametrize("method", ["gaussian", "moran"])
    @pytest.mark.parametrize("sparse", [False, True])
    @pytest.mark.parametrize("statistic", ["Q", "R"])
    def test_scores_and_input_preservation(self, backend, method, sparse, statistic):
        data = _adata(sparse)
        original = data.X.copy()
        coords = data.obsm["positions"].copy()
        detector = _detector(data, backend, method)
        kernel = detector.kernel_
        original_params = dict(kernel.params)
        if sparse:
            data.X.data.setflags(write=False)
        else:
            data.X.setflags(write=False)
        explanation = _explain(detector, statistic)
        assert explanation.metadata["statistic"] == statistic
        assert explanation.filtered.shape == (1, data.n_obs, 1 if statistic == "Q" else 2)
        np.testing.assert_allclose(
            explanation.scores, [_reference(detector, statistic)], rtol=1e-8, atol=1e-10
        )
        np.testing.assert_array_equal(explanation.observation_indices, np.arange(data.n_obs))
        np.testing.assert_array_equal(explanation.coordinates, coords)
        np.testing.assert_array_equal(
            data.X.toarray() if sparse else data.X, original.toarray() if sparse else original
        )
        assert detector.kernel_ is kernel and kernel.params == original_params

    @pytest.mark.parametrize("backend", ["matrix", "nufft"])
    @pytest.mark.parametrize("source", ["var", "obs"])
    def test_layers_and_numeric_obs(self, backend, source):
        data = _adata(sparse=True)
        rng = np.random.default_rng(134)
        values = rng.normal(size=(data.n_obs, 3))
        data.layers["alternative"] = sp.csr_matrix(values)
        data.obs["a"] = values[:, 0]
        data.obs["b"] = values[:, 1]
        detector = _detector(data, backend)
        layer = "alternative" if source == "var" else None
        result = detector.explain_rstat("a", "b", source=source, layer=layer)
        np.testing.assert_array_equal(result.values, values[:, :2])
        np.testing.assert_allclose(
            result.scores, [_reference(detector, "R", source, layer)], rtol=1e-8
        )
        assert result.metadata["source"] == source
        assert result.metadata["layer"] == layer

    @pytest.mark.parametrize("coordinates", [False, True])
    def test_removed_nodes_keep_source_indices(self, coordinates):
        data = _adata(n=10)
        keep = np.array([0, 2, 5, 7, 9])
        graph = np.zeros((10, 10))
        graph[np.ix_(keep, keep)] = np.ones((5, 5)) - np.eye(5)
        data.obsp["graph"] = sp.csr_matrix(graph)
        if not coordinates:
            del data.obsm["positions"]
        detector = DetectorIrregular(kernel_method="moran").setup_data(
            data, obsp_key="graph", obsm_key="positions"
        )
        result = detector.explain_rstat("a", "b")
        np.testing.assert_array_equal(result.observation_indices, keep)
        np.testing.assert_array_equal(result.values, data.X[keep, :2])
        np.testing.assert_allclose(result.scores, [_reference(detector, "R")], atol=1e-12)
        if coordinates:
            np.testing.assert_array_equal(result.coordinates, data.obsm["positions"][keep])
        else:
            assert result.coordinates is None
        assert data.n_obs == 10

    @pytest.mark.parametrize("statistic", ["Q", "R"])
    def test_precision_kernel_stays_sparse(self, monkeypatch, statistic):
        n = 5001
        data = _adata(sparse=True, n=n)
        data.obsp["graph"] = sp.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], format="csr")
        detector = DetectorIrregular(kernel_method="car", rho=0.8, standardize=False).setup_data(
            data, obsp_key="graph", obsm_key="positions"
        )
        kernel = detector.kernel_
        assert kernel.stores_precision

        def forbidden(*args, **kwargs):
            raise AssertionError("explanations must not materialize or diagonalize K")

        monkeypatch.setattr(kernel, "realization", forbidden)
        monkeypatch.setattr(kernel, "eigenvalues", forbidden)
        result = _explain(detector, statistic)
        x = data.X[:, : 1 if statistic == "Q" else 2].toarray()
        z = (x - x.mean(0)) / x.std(0, ddof=1)
        expected = z[:, 0] @ spsolve(kernel._K, z[:, -1])
        np.testing.assert_allclose(result.scores, [expected], rtol=1e-10)
        assert kernel.stores_precision and sp.issparse(kernel._K)

    @pytest.mark.parametrize("backend", ["matrix", "nufft"])
    @pytest.mark.parametrize("sparse", [False, True])
    def test_only_selected_features_are_densified(self, monkeypatch, backend, sparse):
        detector = _detector(_adata(sparse), backend)
        original_init = sp.csc_matrix.__init__
        original_toarray = sp.csc_matrix.toarray

        def bounded_init(self, arg1, *args, **kwargs):
            if hasattr(arg1, "shape") and len(arg1.shape) == 2:
                assert arg1.shape[1] <= 2
            original_init(self, arg1, *args, **kwargs)

        def bounded_toarray(self, *args, **kwargs):
            assert self.shape[1] <= 2
            return original_toarray(self, *args, **kwargs)

        monkeypatch.setattr(sp.csc_matrix, "__init__", bounded_init)
        monkeypatch.setattr(sp.csc_matrix, "toarray", bounded_toarray)
        detector.explain_qstat("a")
        detector.explain_rstat("a", "b")

    def test_asymmetric_kernel_rejected(self):
        detector = _detector(_adata())
        rng = np.random.default_rng(141)
        detector.kernel_ = MatrixKernel.from_matrix(rng.normal(size=(detector.n, detector.n)))
        with pytest.raises(ArithmeticError, match="symmetric kernel"):
            detector.explain_rstat("a", "b")

    @pytest.mark.parametrize("backend", ["matrix", "nufft"])
    def test_filtered_features_rejected(self, backend):
        adata = _adata()
        adata.X[:, 0] = 0
        with pytest.raises(ValueError, match="excluded"):
            _detector(adata, backend).explain_qstat("a")

    def test_invalid_selection(self):
        detector = _detector(_adata())
        with pytest.raises(TypeError, match="single feature"):
            detector.explain_qstat(["a", "b"])
        with pytest.raises(ValueError, match="Every requested"):
            detector.explain_rstat("a", "unknown")
        detector.adata.obs["celltype"] = pd.Categorical(["A", "B"] * (detector.n // 2))
        with pytest.raises(ValueError, match="categorical"):
            detector.explain_qstat("celltype", source="obs")
        for kwargs in ({"source": "bad"}, {"source": "obs", "layer": "x"}):
            with pytest.raises(ValueError):
                detector.explain_qstat("a", **kwargs)


class TestGridExplanation:
    @pytest.mark.parametrize("solver", ["fft2", "rfft2"])
    @pytest.mark.parametrize("method", ["gaussian", "moran"])
    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("statistic", ["Q", "R"])
    def test_scores_mask_and_coordinates(self, monkeypatch, solver, method, masked, statistic):
        rng = np.random.default_rng(25)
        values = rng.normal(size=(3, 7, 8))
        if masked:
            values[:, 2:5, 1:4] = np.nan
        original = values.copy()
        detector = _grid(monkeypatch, values, solver=solver, method=method, spacing=(2, 3))
        values.setflags(write=False)
        explanation = _explain(detector, statistic)
        reference = _reference(detector, statistic)
        np.testing.assert_allclose(explanation.scores, [reference], rtol=1e-10, atol=1e-12)
        indices = np.flatnonzero(np.isfinite(values[0]).ravel())
        np.testing.assert_array_equal(explanation.observation_indices, indices)
        np.testing.assert_array_equal(
            explanation.coordinates, np.column_stack(np.unravel_index(indices, (7, 8))) * [2, 3]
        )
        np.testing.assert_array_equal(
            explanation.values, values[: 1 if statistic == "Q" else 2, np.isfinite(values[0])].T
        )
        np.testing.assert_array_equal(values, original)

    def test_constant_maps_are_zero(self, monkeypatch):
        data = np.ones((3, 7, 8)) * 0.7
        data[:, 1, 2] = np.nan
        detector = _grid(monkeypatch, data)
        for result in (detector.explain_qstat("a"), detector.explain_rstat("a", "b")):
            np.testing.assert_array_equal(result.contributions, 0)
            np.testing.assert_array_equal(result.values, 0.7)

    def test_invalid_source_and_support(self, monkeypatch):
        values = np.random.default_rng(6).normal(size=(3, 7, 8))
        values[0, 0, 0] = np.nan
        grid = _grid(monkeypatch, values)
        with pytest.raises(ValueError, match="identical"):
            grid.explain_rstat("a", "b")
        with pytest.raises(ValueError, match="setup-time raster"):
            grid.explain_qstat("a", layer="x")
        values[0, 0, 0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            grid.explain_qstat("a")


class TestDetectorExplanation:
    def test_self_pair_and_pair_symmetry(self, detector):
        a = detector.explain_rstat("a", "b")
        b = detector.explain_rstat("b", "a")
        np.testing.assert_allclose(a.contributions, b.contributions, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(a.filtered, b.filtered[:, :, ::-1], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            detector.explain_qstat("a").contributions,
            detector.explain_rstat("a", "a").contributions,
            rtol=1e-10,
            atol=1e-12,
        )

    @pytest.mark.parametrize("backend", ["matrix", "nufft", "fft"])
    @pytest.mark.parametrize("offset,scale", [(1e6, 1), (0, 1e-10), (0, 1e-14)])
    def test_offsets_and_small_signals(self, monkeypatch, backend, offset, scale):
        if backend == "fft":
            values = offset + scale * np.random.default_rng(6).normal(size=(3, 7, 8))
            values[:, 0, 0] = np.nan
            detector = _grid(monkeypatch, values)
        else:
            data = _adata()
            data.X = offset + scale * data.X
            detector = _detector(data, backend)
        for statistic in ("Q", "R"):
            result = _explain(detector, statistic)
            np.testing.assert_allclose(
                result.scores,
                [_reference(detector, statistic)],
                rtol=1e-8,
                atol=1e-25 if scale < 1e-12 else 1e-8,
            )

    def test_budget_checked_before_loading(self, detector):
        loader = (
            "_filter_features"
            if isinstance(detector, DetectorGrid)
            else {"matrix": "_prepare_data", "nufft": "_prepare_features_nufft"}[detector.backend_]
        )
        with patch.object(
            detector, loader, side_effect=AssertionError("loaded before budget check")
        ):
            with pytest.raises(MemoryError, match="exceeds budget"):
                detector.explain_rstat("a", "b", memory_budget_bytes=1)
            with pytest.raises(ValueError, match="positive"):
                detector.explain_qstat("a", memory_budget_bytes=0)

    def test_no_null_fit(self, monkeypatch, detector):
        def forbidden(*args, **kwargs):
            raise AssertionError("explanation must not fit a null distribution")

        monkeypatch.setattr("sonic.statistics._prepare_q_null", forbidden)
        monkeypatch.setattr("sonic.statistics.compute_null_params", forbidden)
        monkeypatch.setattr("sonic.detectors.grid.compute_null_params", forbidden)
        monkeypatch.setattr("sonic.detectors.irregular.compute_null_params", forbidden)
        detector.explain_qstat("a")
        detector.explain_rstat("a", "b")

    @pytest.mark.parametrize("detector_cls", [DetectorIrregular, DetectorGrid])
    def test_setup_required(self, detector_cls):
        with pytest.raises(RuntimeError, match="setup_data"):
            detector_cls(kernel_method="gaussian").explain_qstat("a")
