"""Synthetic checks for shared spatial explanations and detector/comparator adapters."""

import copy
from dataclasses import fields
from unittest.mock import PropertyMock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from sonic import DetectorGrid, DetectorIrregular, MatrixKernel
from sonic.comparators.binning import _polar_bin_reducer, _radial_bin_reducer
from sonic.explanation import SpatialExplanation, _explanation_from_responses
from tests.test_cross_spectrum import EDGES, _irregular_points, make_comparator  # noqa: F401
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
    @pytest.mark.parametrize("sparse,layer", [(False, None), (True, "alternative")])
    def test_matrix_explains_unique_gene_on_duplicate_axis(self, sparse, layer):
        data = _adata(sparse=sparse)
        data.var_names = ["dup", "unique", "dup"]
        if layer is not None:
            data.layers[layer] = data.X[:, [2, 0, 1]].copy()
        detector = _detector(data)
        scores = detector.compute_qstat(
            layer=layer, n_jobs=1, return_pval=False, show_progress=False
        )
        explanation = detector.explain_qstat("unique", layer=layer)
        np.testing.assert_allclose(explanation.scores, [scores.loc["unique", "Q"]])
        with pytest.raises(ValueError, match="Ambiguous.*dup"):
            detector.explain_qstat("dup", layer=layer)

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


class TestComparatorExplanation:
    REPRESENTATIONS = ["power", "power_norm", "cross", "coherency", "coherence"]

    def test_explanation_rejects_only_ambiguous_gene_names(self):
        from sonic import ComparatorIrregular

        sample = _adata()
        sample.var_names = ["a", "a", "c"]
        comparator = ComparatorIrregular(
            [sample], obsm_key="positions", grid_shape=(9, 11), spacing=(1, 1), workers=1
        ).compute_spectra(progress=False)

        for features in ("a", ("c", "a")):
            with pytest.raises(ValueError, match="Ambiguous.*'a'"):
                comparator.explain_spectrum(features, bands=[0])
        result = comparator.explain_spectrum(("c", "c"), bands=[0])
        np.testing.assert_array_equal(result.values, sample.X[:, [2, 2]])

    def test_backed_spectrum_explains_reversed_and_repeated_features(self, tmp_path):
        from sonic import ComparatorIrregular

        data = _adata()
        path = tmp_path / "expression.h5ad"
        data.write_h5ad(path)

        def fit(sample):
            return ComparatorIrregular(
                [sample],
                obsm_key="positions",
                grid_shape=(9, 11),
                spacing=(1, 1),
                freq_edges=EDGES,
                workers=1,
            ).compute_cross_spectra([("a", "c")], progress=False)

        backed = ad.read_h5ad(path, backed="r")
        try:
            memory, disk = fit(data), fit(backed)
            for names in (["c", "a"], ["c", "c"]):
                options = {"bands": [1], "representation": "coherence"}
                expected = memory.explain_spectrum(names, **options)
                actual = disk.explain_spectrum(names, **options)
                np.testing.assert_array_equal(
                    actual.values, data.X[:, data.var_names.get_indexer(names)]
                )
                np.testing.assert_allclose(actual.scores, expected.scores)
                np.testing.assert_allclose(actual.contributions, expected.contributions)
        finally:
            backed.file.close()

    @staticmethod
    def _fit(comparator, angle=17):
        if comparator.feature_mode == "2d":
            comparator.rotation_angles_ = np.full(len(comparator.samples), angle, dtype=float)
        return comparator.compute_spectra(progress=False, n_jobs=1).compute_cross_spectra(
            [("a", "b"), ("a", "a")], progress=False, n_jobs=1
        )

    @staticmethod
    def _explain(comparator, representation, bands, **kwargs):
        features = "a" if representation.startswith("power") else ["a", "b"]
        return comparator.explain_spectrum(
            features, bands=bands, representation=representation, **kwargs
        )

    @staticmethod
    def _expected(comparator, representation):
        result = comparator.cross_spectra_
        if representation.startswith("power"):
            power = comparator.spectra_[0, 0]
            return power / power.sum() if representation == "power_norm" else power
        if representation == "cross":
            return result.cross_spectra[0, 0].real
        if representation == "coherency":
            return result.coherency[0, 0].real
        return result.coherence[0, 0]

    @pytest.mark.parametrize("backend", ["fft", "nufft"])
    @pytest.mark.parametrize("mode", ["radial", "2d"])
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_reconstructs_extracted_features(self, make_comparator, backend, mode, representation):  # noqa: F811
        values = np.random.default_rng(51).normal(size=(3, 16, 18))
        comparator = self._fit(make_comparator(values, backend=backend, mode=mode))
        bands = [2, 0] if mode == "radial" else [7, 0, 11]
        result = self._explain(comparator, representation, bands)
        np.testing.assert_allclose(
            result.scores, self._expected(comparator, representation)[bands], atol=2e-10
        )
        np.testing.assert_array_equal(result.metadata["bands"], bands)
        np.testing.assert_allclose(result.values[:, 0], values[0].ravel())
        assert result.filtered.shape == (len(bands), 16 * 18, result.values.shape[1])
        assert result.metadata["spectrum_source"] == "recomputed_raw"

    @pytest.mark.parametrize("mode", ["radial", "2d"])
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_direct_dft_spatial_oracle(self, mode, representation):
        rng = np.random.default_rng(63)
        shape, spacing, scale = (8, 10), (0.7, 1.1), 2.3
        coords = rng.uniform(-2, 3, size=(37, 2))
        values = rng.normal(size=(37, 3))
        comparator = self._fit(
            _irregular_points(
                values, coords, shape=shape, mode=mode, spacing=spacing, unit_scales=[scale]
            )
        )
        padding = int(mode == "2d")
        modes = np.asarray(shape) + 2 * padding
        fy = np.fft.fftfreq(modes[0]) * modes[0] / (shape[0] * spacing[0])
        fx = np.fft.fftfreq(modes[1]) * modes[1] / (shape[1] * spacing[1])
        frequencies = np.stack(np.meshgrid(fy, fx, indexing="ij"), axis=-1).reshape(-1, 2)
        basis = np.exp(-2j * np.pi * (coords * scale) @ frequencies.T).T
        if mode == "2d":
            reducer, _ = _polar_bin_reducer(
                shape, spacing=spacing, edges=EDGES, n_theta=4, angle_deg=17, frequency_padding=1
            )
        else:
            reducer, _ = _radial_bin_reducer(shape, spacing=spacing, edges=EDGES, fft_solver="fft2")
        retained = np.repeat(comparator._supported_frequency_bin_mask, 4 if mode == "2d" else 1)
        reducer = reducer[retained]
        g = 1 if representation.startswith("power") else 2
        z = values[:, :g] - values[:, :g].mean(0)
        coefficients = basis @ z / len(z)
        coefficients[0] = 0
        auto = reducer @ abs(coefficients) ** 2
        cross = reducer @ (coefficients[:, 0] * coefficients[:, -1].conj())
        if mode == "radial":
            cross = cross.real
        bands = [reducer.shape[0] - 1, 1]
        result = self._explain(comparator, representation, bands)
        for row, band in enumerate(bands):
            weights = reducer.getrow(band).toarray().ravel()
            weights[0] = 0
            matrix = (basis.conj().T * weights) @ basis / len(z) ** 2
            matrix -= matrix.mean(axis=0, keepdims=True)
            response = matrix @ z
            allocation = z[:, 0] * response[:, 0].real
            if g == 2:
                allocation = (z[:, 1] * response[:, 0] + z[:, 0] * response[:, 1].conj()) / 2
                if mode == "radial":
                    allocation = allocation.real
            if representation == "power_norm":
                allocation /= auto[:, 0].sum()
            elif representation in ("coherency", "coherence"):
                denominator = np.sqrt(auto[band, 0]) * np.sqrt(auto[band, 1])
                allocation /= denominator
                if representation == "coherence":
                    allocation = ((cross[band] / denominator).conj() * allocation).real
            np.testing.assert_allclose(result.contributions[row], allocation.real, atol=2e-10)
        np.testing.assert_array_equal(result.coordinates, coords)
        assert result.metadata["unit_scale"] == scale
        if mode == "2d" and g == 2:
            assert np.max(abs(cross.imag)) > 1e-5

    @pytest.mark.parametrize("mode", ["radial", "2d"])
    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_fft_nufft_maps_match(self, make_comparator, mode, representation):  # noqa: F811
        values = np.random.default_rng(80).normal(size=(3, 15, 17))
        fft = self._fit(make_comparator(values, mode=mode))
        nufft = self._fit(make_comparator(values, mode=mode, backend="nufft"))
        a, b = [self._explain(c, representation, [1, 2]) for c in (fft, nufft)]
        np.testing.assert_allclose(a.contributions, b.contributions, atol=2e-10)
        np.testing.assert_allclose(a.filtered, b.filtered, atol=2e-10)
        np.testing.assert_allclose(a.coordinates, b.coordinates)

    @pytest.mark.parametrize("representation", REPRESENTATIONS)
    def test_grid_holes_use_observed_support(self, make_comparator, representation):  # noqa: F811
        values = np.random.default_rng(18).normal(size=(3, 15, 17))
        values[:, 2:5, 4:7] = np.nan
        comparator = self._fit(make_comparator(values, mode="2d", spacing=(0.8, 1.1)))
        result = self._explain(comparator, representation, [1, 3])
        indices = np.flatnonzero(np.isfinite(values[0].ravel()))
        np.testing.assert_array_equal(result.observation_indices, indices)
        np.testing.assert_allclose(
            result.coordinates, np.c_[np.unravel_index(indices, (15, 17))] * [0.8, 1.1]
        )
        np.testing.assert_allclose(
            result.scores, self._expected(comparator, representation)[[1, 3]], atol=1e-12
        )
        assert np.isnan(values[:, 2:5, 4:7]).all()

    @pytest.mark.parametrize("representation", ["coherency", "coherence"])
    def test_pair_swap_self_pair_scaling_translation(self, representation):
        rng = np.random.default_rng(57)
        coords, values = rng.uniform(0, 10, (61, 2)), rng.normal(size=(61, 3))
        original = self._fit(_irregular_points(values, coords, mode="2d"))
        result = self._explain(original, representation, [0, 4])
        swapped = original.explain_spectrum(["b", "a"], bands=[0, 4], representation=representation)
        np.testing.assert_allclose(swapped.contributions, result.contributions, atol=1e-12)
        scaled = self._fit(_irregular_points(values * [3, 0.2, 1], coords + [100, -73], mode="2d"))
        np.testing.assert_allclose(
            self._explain(scaled, representation, [0, 4]).contributions,
            result.contributions,
            atol=2e-10,
        )
        self_pair = original.explain_spectrum(
            ["a", "a"], bands=[0, 4], representation=representation
        )
        np.testing.assert_allclose(self_pair.scores, 1, atol=1e-12)
        negative = self._fit(_irregular_points(values * [-1, 1, 1], coords, mode="2d"))
        sign = -1 if representation == "coherency" else 1
        np.testing.assert_allclose(
            self._explain(negative, representation, [0, 4]).contributions,
            sign * result.contributions,
            atol=2e-10,
        )

    @pytest.mark.parametrize("mode", ["radial", "2d"])
    def test_retained_bin_indices_and_total_normalization(self, make_comparator, mode):  # noqa: F811
        values = np.random.default_rng(43).normal(size=(3, 16, 18))
        edges = [0, 0.01, 0.1, 0.101, 0.3, 0.45, 0.8]
        comparator = self._fit(make_comparator(values, mode=mode, freq_edges=edges))
        assert not comparator._supported_frequency_bin_mask.all()
        n_theta = 4 if mode == "2d" else 1
        result = self._explain(comparator, "power_norm", [1])
        np.testing.assert_allclose(result.scores, self._expected(comparator, "power_norm")[[1]])
        np.testing.assert_array_equal(
            result.metadata["frequency_bin_intervals"],
            comparator.frequency_bin_intervals_[[1 // n_theta]],
        )
        assert result.scores[0] < 1

    def test_no_refit_or_cache_mutation_and_subset(self, make_comparator):  # noqa: F811
        rng = np.random.default_rng(27)
        comparator = self._fit(
            make_comparator([rng.normal(size=(3, 15, 17)) for _ in range(2)], mode="2d")
        )
        reference = comparator.spectra_.copy()
        comparator.normalize_background()
        cached = comparator.spectra_.copy()
        cross = copy.deepcopy(comparator.cross_spectra_)
        angles = comparator.rotation_angles_.copy()
        with patch.object(comparator, "_ensure_alignment", side_effect=AssertionError("refit")):
            maps = comparator.explain_spectrum("a", sample_index=1, bands=[1, 4])
        np.testing.assert_allclose(maps.scores, reference[1, 0, [1, 4]])
        np.testing.assert_array_equal(comparator.spectra_, cached)
        np.testing.assert_array_equal(comparator.rotation_angles_, angles)
        for name in ("cross_spectra", "auto_spectra", "coherency_valid"):
            np.testing.assert_array_equal(
                getattr(comparator.cross_spectra_, name), getattr(cross, name)
            )
        subset = comparator.subset([1])
        np.testing.assert_allclose(
            subset.explain_spectrum("a", bands=[1, 4]).contributions, maps.contributions
        )
        maps.filtered[:] = 0
        maps.values[:] = 0
        maps.coordinates[:] = 0
        np.testing.assert_array_equal(comparator.spectra_, cached)

    def test_cold_cross_fit_and_selected_sparse_layer(self):
        rng = np.random.default_rng(6)
        values, coords = rng.normal(size=(50, 3)), rng.uniform(0, 9, (50, 2))
        base = _irregular_points(values, coords)
        sample = base.samples[0]
        sample.layers["counts"] = sp.csr_matrix(values * 2)
        comparator = type(base)(
            [sample],
            grid_shape=(17, 19),
            spacing=(1, 1),
            freq_edges=EDGES,
            eps=1e-12,
            workers=1,
            layer="counts",
        )
        comparator.compute_cross_spectra([("a", "b")], progress=False)
        assert comparator.spectra_ is None
        original = sp.csr_matrix.toarray
        shapes = []

        def record(matrix, *args, **kwargs):
            shapes.append(matrix.shape)
            return original(matrix, *args, **kwargs)

        with patch.object(sp.csr_matrix, "toarray", record):
            result = self._explain(comparator, "coherence", [0, 2])
        assert all(shape[1] <= 2 for shape in shapes if shape[0] == 50)
        np.testing.assert_allclose(result.values, values[:, :2] * 2)
        np.testing.assert_allclose(
            result.scores, comparator.cross_spectra_.coherence[0, 0, [0, 2]], atol=2e-10
        )
        assert comparator.spectra_ is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"bands": []},
            {"bands": [0.0]},
            {"bands": [True]},
            {"bands": [-1]},
            {"bands": [0, 0]},
            {"bands": [999]},
            {"bands": [[0]]},
            {"sample_index": -1},
            {"sample_index": True},
            {"sample_index": 2},
            {"sample_index": 0.1},
            {"features": []},
            {"features": ["a", "b", "c"]},
            {"features": ["missing"]},
            {"representation": "coherence"},
        ],
    )
    def test_invalid_requests(self, make_comparator, kwargs):  # noqa: F811
        comparator = self._fit(make_comparator(np.random.default_rng(1).normal(size=(3, 9, 11))))
        options = {"features": "a", "bands": [0], **kwargs}
        with pytest.raises(ValueError):
            comparator.explain_spectrum(**options)

    def test_preconditions_budget_and_invalid_normalization(self, make_comparator):  # noqa: F811
        values = np.random.default_rng(13).normal(size=(3, 9, 11))
        values[1] = 1
        comparator = make_comparator(values)
        with pytest.raises(RuntimeError, match="compute_spectra"):
            comparator.explain_spectrum("a", bands=[0])
        self._fit(comparator)
        with patch.object(comparator, "_batch_raster", side_effect=AssertionError("loaded values")):
            with pytest.raises(MemoryError, match="budget"):
                comparator.explain_spectrum("a", bands=[0], memory_budget_bytes=1)
        for representation in ("coherency", "coherence"):
            with pytest.raises(ValueError, match="invalid coherency"):
                self._explain(comparator, representation, [0])
        np.testing.assert_array_equal(
            comparator.explain_spectrum("b", bands=[0], representation="power_norm").scores, [0]
        )
        np.testing.assert_array_equal(self._explain(comparator, "cross", [0]).scores, [0])
        values[1, 0, 0] = np.nan
        with pytest.raises(ValueError, match="identical finite support"):
            self._explain(comparator, "cross", [0])

    def test_even_nufft_nyquist_uses_real_radial_cross(self):
        rng = np.random.default_rng(72)
        coords = rng.uniform(size=(53, 2)) * [8, 10]
        values = rng.normal(size=(53, 3))
        comparator = self._fit(
            _irregular_points(values, coords, shape=(8, 10), freq_edges=[0, 0.25, 0.5])
        )
        frequencies = np.stack(
            np.meshgrid(np.fft.fftfreq(8), np.fft.fftfreq(10), indexing="ij"), axis=-1
        ).reshape(-1, 2)
        basis = np.exp(-2j * np.pi * coords @ frequencies.T).T
        z = values[:, :2] - values[:, :2].mean(0)
        coefficients = basis @ z / len(z)
        radius = np.linalg.norm(frequencies, axis=1)
        weights = ((radius >= 0.25) & (radius <= 0.5)).astype(float)
        weights /= weights.sum()
        cross = weights @ (coefficients[:, 0] * coefficients[:, 1].conj())
        assert abs(cross.imag) > 1e-6  # Even NUFFT lattices have unpaired boundary modes.
        response = basis.conj().T @ (weights[:, None] * coefficients) / len(z)
        response -= response.mean(axis=0)
        allocation = (z[:, 0] * response[:, 1].real + z[:, 1] * response[:, 0].real) / 2
        np.testing.assert_allclose(
            self._explain(comparator, "cross", [1]).contributions[0], allocation, atol=1e-10
        )
        coherence = self._explain(comparator, "coherence", [1])
        np.testing.assert_allclose(
            coherence.scores, self._expected(comparator, "coherence")[[1]], atol=1e-10
        )

    @pytest.mark.parametrize("reason", ["presence", "single_mode"])
    def test_normalized_validity_matches_extraction(self, reason):
        rng = np.random.default_rng(21)
        values, coords = rng.normal(size=(41, 3)), rng.uniform(0, 5, (41, 2))
        if reason == "presence":
            values[2:, 0] = 0
        comparator = self._fit(
            _irregular_points(
                values,
                coords,
                shape=(8, 10),
                spacing=(0.7, 1.1),
                presence_threshold=0.5,
            )
        )
        band = 1 if reason == "presence" else 0
        assert not comparator.cross_spectra_.coherency_valid[0, 0, band]
        assert np.isfinite(self._explain(comparator, "cross", [band]).scores).all()
        with pytest.raises(ValueError, match="invalid coherency"):
            self._explain(comparator, "coherence", [band])

    @pytest.mark.parametrize("representation", ["cross", "coherency", "coherence"])
    def test_half_turn_preserves_real_maps(self, make_comparator, representation):  # noqa: F811
        values = np.random.default_rng(22).normal(size=(3, 17, 19))
        original = self._fit(make_comparator(values, mode="2d"), angle=17)
        a = self._explain(original, representation, [0, 4, 10])
        rotated = self._fit(make_comparator(values, mode="2d"), angle=197)
        b = self._explain(rotated, representation, [0, 4, 10])
        np.testing.assert_allclose(a.contributions, b.contributions, atol=1e-12)
        np.testing.assert_array_equal(a.coordinates, b.coordinates)
        np.testing.assert_allclose(
            original.cross_spectra_.cross_spectra,
            rotated.cross_spectra_.cross_spectra.conj(),
            atol=1e-12,
        )

    @pytest.mark.parametrize("mode", ["radial", "2d"])
    def test_nufft_budget_precedes_source_read(self, make_comparator, mode):  # noqa: F811
        comparator = self._fit(
            make_comparator(
                np.random.default_rng(9).normal(size=(3, 9, 11)), backend="nufft", mode=mode
            )
        )
        with patch.object(
            type(comparator.samples[0]),
            "X",
            new_callable=PropertyMock,
            side_effect=AssertionError("loaded source"),
        ):
            with pytest.raises(MemoryError, match="budget"):
                comparator.explain_spectrum(["a", "b"], bands=[0], memory_budget_bytes=1)


@pytest.fixture
def plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as pyplot

    yield pyplot
    pyplot.close("all")


class TestExplanationRegions:
    @staticmethod
    def _map(contribution=None, *, metadata=None):
        yy, xx = np.indices((61, 61))
        coords = np.c_[yy.ravel(), xx.ravel()].astype(float)
        if contribution is None:
            contribution = np.exp(-((yy - 16) ** 2 + (xx - 16) ** 2) / 12)
            contribution -= 2 * np.exp(-((yy - 46) ** 2 + (xx - 46) ** 2) / 12)
        contribution = np.asarray(contribution).ravel()
        return SpatialExplanation(
            feature_names=("a",),
            observation_indices=np.arange(len(coords)),
            coordinates=coords,
            values=np.ones((len(coords), 1)),
            filtered=contribution[None, :, None].copy(),
            contributions=contribution[None].copy(),
            metadata={"statistic": "Q"} if metadata is None else metadata,
        )

    @pytest.mark.parametrize(
        "direction,center,sign", [("absolute", 46, -1), ("positive", 16, 1), ("negative", 46, -1)]
    )
    def test_signed_ranking_nonoverlap_and_exact_totals(self, direction, center, sign):
        result = self._map()
        regions = result.select_regions(size=20, n_regions=2, direction=direction)
        first = regions.iloc[0]
        assert first.coord0_min <= center < first.coord0_max
        assert first.coord1_min <= center < first.coord1_max
        assert np.sign(first.contribution_sum) == sign
        for row, region in enumerate(regions.itertuples(index=False)):
            lower = np.array([region.coord0_min, region.coord1_min])
            upper = np.array([region.coord0_max, region.coord1_max])
            inside = ((result.coordinates >= lower) & (result.coordinates < upper)).all(axis=1)
            assert region.n_observations == inside.sum()
            assert region.contribution_sum == result.contributions[0, inside].sum()
            assert region.coverage == 1
            for previous in regions.iloc[:row].itertuples(index=False):
                assert not (
                    abs(region.coord0_min - previous.coord0_min) < 20
                    and abs(region.coord1_min - previous.coord1_min) < 20
                )
        repeated = result.select_regions(size=20, n_regions=2, direction=direction)
        pd.testing.assert_frame_equal(regions, repeated)

    def test_translation_and_source_immutability(self):
        result = self._map()
        original = copy.deepcopy(result)
        regions = result.select_regions(size=20)
        shifted = copy.deepcopy(result)
        shift = np.array([123.5, -71.25])
        shifted.coordinates += shift
        translated = shifted.select_regions(size=20)
        for suffix in ("min", "max"):
            np.testing.assert_allclose(
                translated[[f"coord0_{suffix}", f"coord1_{suffix}"]],
                regions[[f"coord0_{suffix}", f"coord1_{suffix}"]] + shift,
            )
        np.testing.assert_allclose(translated.contribution_sum, regions.contribution_sum)
        for name in ("values", "filtered", "contributions", "coordinates", "observation_indices"):
            np.testing.assert_array_equal(getattr(result, name), getattr(original, name))

    def test_scale_heuristic_uses_returned_map_and_source_units(self):
        result = self._map(
            metadata={
                "statistic": "coherency",
                "bands": np.array([7, 12]),
                "frequency_bin_intervals": np.array([[0.05, 0.1], [0.2, 0.21]]),
                "unit_scale": 2.0,
            }
        )
        result.contributions = np.repeat(result.contributions, 2, axis=0)
        result.filtered = np.repeat(result.filtered, 2, axis=0)
        first = result.select_regions(map_index=0, min_coverage=0.1)
        second = result.select_regions(map_index=1, min_coverage=0.1)
        np.testing.assert_allclose(first["size"], 30)
        np.testing.assert_allclose(second["size"], 50)
        assert not first.context_limited.any()
        small = result.select_regions(map_index=1, size=20)
        assert small.context_limited.all()
        np.testing.assert_allclose(small.recommended_size, 50)
        with pytest.raises(ValueError, match="map_index"):
            result.select_regions(map_index=12, size=20)

    def test_empty_low_coverage_cancellation_and_boundaries(self):
        result = self._map(np.ones((61, 61)))
        region = result.select_regions(size=60, min_coverage=1).iloc[0]
        assert region.n_observations == 60**2  # Upper-bound points are excluded.
        assert region.contribution_sum == 60**2
        assert result.select_regions(size=61).empty
        assert result.select_regions(size=20, min_observations=1000).empty
        checker = (-1.0) ** np.indices((61, 61)).sum(axis=0)
        assert self._map(checker).select_regions(size=20).empty
        assert self._map(np.zeros((61, 61))).select_regions(size=20).empty
        keep = np.arange(0, len(result.coordinates), 25)
        sparse = SpatialExplanation(
            result.feature_names,
            result.observation_indices[keep],
            result.coordinates[keep],
            result.values[keep],
            result.filtered[:, keep],
            result.contributions[:, keep],
            result.metadata,
        )
        assert sparse.select_regions(size=20, min_coverage=0.5, min_observations=1).empty

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"size": 0},
            {"size": np.nan},
            {"size": -2},
            {"size": 20, "n_regions": 0},
            {"size": 20, "n_regions": True},
            {"size": 20, "min_observations": 1.5},
            {"size": 20, "min_coverage": 0},
            {"size": 20, "min_coverage": np.nan},
            {"size": 20, "direction": "bad"},
            {"size": 20, "map_index": True},
            {"size": 20, "map_index": -1},
        ],
    )
    def test_invalid_selection(self, kwargs):
        with pytest.raises(ValueError):
            self._map().select_regions(**kwargs)

    def test_sparse_region_tiling_fits_candidate_window_budget(self):
        result = self._map()
        options = {"size": 4, "min_coverage": 0.02, "min_observations": 1}
        expected = result.select_regions(**options)
        assert len(expected) == 2
        actual = result.select_regions(**options, memory_budget_bytes="3 MiB")
        pd.testing.assert_frame_equal(actual, expected)

    def test_missing_coordinates_zero_edge_and_memory_limit(self):
        result = self._map()
        with patch("numpy.bincount", side_effect=AssertionError("allocated before guard")):
            with pytest.raises(MemoryError, match="memory budget"):
                result.select_regions(size=1e-4, memory_budget_bytes="1 MiB")
        result.metadata["frequency_bin_intervals"] = np.array([[0, 0.2]])
        with pytest.raises(ValueError, match="size"):
            result.select_regions()
        assert len(result.select_regions(size=20)) > 0
        result.coordinates = None
        with pytest.raises(ValueError, match="coordinates"):
            result.select_regions(size=20)
        with pytest.raises(ValueError, match="coordinates"):
            result.plot()

    @pytest.mark.parametrize("mode", ["radial", "2d"])
    def test_comparator_local_motif(self, make_comparator, mode):  # noqa: F811
        yy, xx = np.indices((65, 65))
        motif = np.sin(2 * np.pi * 0.2 * xx) * np.exp(-((yy - 22) ** 2 + (xx - 22) ** 2) / 64)
        comparator = make_comparator(np.stack([motif, motif, motif]), mode=mode)
        if mode == "2d":
            comparator.rotation_angles_ = np.array([0.0])
        comparator.compute_spectra(progress=False, n_jobs=1)
        result = comparator.explain_spectrum("a", bands=[1 if mode == "radial" else 4])
        region = result.select_regions(size=15, n_regions=1).iloc[0]
        assert region.coord0_min <= 22 < region.coord0_max
        assert region.coord1_min <= 22 < region.coord1_max
        assert region.context_limited


class TestExplanationPlotting:
    @pytest.mark.parametrize("kind", ["Q", "R", "power", "coherency", "coherence"])
    def test_panels_shared_limits_and_zoom_data(self, plt, monkeypatch, make_comparator, kind):  # noqa: F811
        values = np.random.default_rng(91).normal(size=(3, 25, 27))
        values[:, 3:5, 6:8] = np.nan
        if kind in ("Q", "R"):
            result = _explain(_grid(monkeypatch, values), kind)
        else:
            comparator = make_comparator(values, mode="2d")
            comparator.rotation_angles_ = np.array([19.0])
            comparator.compute_cross_spectra([("a", "b")], progress=False, n_jobs=1)
            result = comparator.explain_spectrum(
                "a" if kind == "power" else ["a", "b"], bands=[4], representation=kind
            )
        before = copy.deepcopy(result)
        regions = result.select_regions(size=10, n_regions=1, min_observations=10, min_coverage=0.1)
        fig = result.plot(regions=regions, title="Synthetic example")
        fig.canvas.draw()
        ncols = 2 * len(result.feature_names) + 1
        axes = fig.axes[: 2 * ncols]
        assert len(axes) == 2 * ncols
        box = regions.iloc[0]
        lower, upper = [box.coord0_min, box.coord1_min], [box.coord0_max, box.coord1_max]
        keep = ((result.coordinates >= lower) & (result.coordinates < upper)).all(axis=1)
        arrays = [*result.values.T, *result.filtered[0].T, result.contributions[0]]
        for column, array in enumerate(arrays):
            image = axes[column].images[0]
            scatter = axes[ncols + column].collections[0]
            assert image.norm is scatter.norm
            assert image.get_array().mask.any()  # Holes are not interpolated.
            np.testing.assert_array_equal(scatter.get_array(), array[keep])
            np.testing.assert_allclose(scatter.get_offsets(), result.coordinates[keep][:, [1, 0]])
            assert axes[column].yaxis_inverted()
            assert len(axes[column].patches) == 1
        assert "sum =" in axes[-1].get_title()
        assert "frequency" in fig._suptitle.get_text() if kind not in ("Q", "R") else True
        for name in ("values", "filtered", "contributions", "coordinates"):
            np.testing.assert_array_equal(getattr(result, name), getattr(before, name))
        np.testing.assert_array_equal(result.scores, before.scores)

    def test_source_coordinate_order_custom_limits_and_no_implicit_crops(self, plt):
        result = TestExplanationRegions._map()
        limits = [(0, 2), (-2, 2), (-3, 3)]
        with patch.object(
            result, "select_regions", side_effect=AssertionError("implicit selection")
        ):
            fig = result.plot(limits=limits)
        assert len(fig.axes) == 6  # Three overview axes and colorbars.
        overview = fig.axes[0]
        assert overview.get_xlabel() == "Coordinate 0"
        assert not overview.yaxis_inverted()
        for axis, clim in zip(fig.axes[:3], limits, strict=True):
            assert axis.images[0].get_clim() == clim
        region = result.select_regions(size=20, n_regions=1)
        fig = result.plot(regions=region, coordinate_order=(1, 0))
        assert fig.axes[0].get_xlabel() == "Coordinate 1"
        fig.canvas.draw()

    def test_overview_is_bounded_observation_mean(self, plt):
        result = TestExplanationRegions._map()
        result.coordinates *= 1e6
        result.coordinates[1] = result.coordinates[0]
        result.values[0, 0], result.values[1, 0] = 1, 3
        fig = result.plot()
        image = fig.axes[0].images[0]
        assert max(image.get_array().shape) <= 512
        assert image.get_array()[0, 0] == 2
        assert np.count_nonzero(~image.get_array().mask) <= len(result.coordinates)
        fig.canvas.draw()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"limits": [(0, 1)]},
            {"limits": [(0, 0)] * 3},
            {"limits": [(0, np.nan)] * 3},
            {"coordinate_order": (0, 0)},
            {"coordinate_order": (0.0, 1.0)},
            {"coordinate_order": (False, True)},
            {"point_size": 0},
            {"map_index": 1},
        ],
    )
    def test_plot_validation_no_figure_leak(self, plt, kwargs):
        with pytest.raises(ValueError):
            TestExplanationRegions._map().plot(**kwargs)
        assert not plt.get_fignums()

    def test_empty_invalid_and_stale_region_summaries(self, plt):
        result = TestExplanationRegions._map()
        empty = result.select_regions(size=100)
        fig = result.plot(regions=empty)
        assert len(fig.axes) == 6
        region = result.select_regions(size=20, n_regions=1)
        region.loc[0, "contribution_sum"] = 99999
        fig = result.plot(regions=region)
        assert "99999" not in fig.axes[5].get_title()  # Recompute summaries from plotted rows.
        for corruption in ("width", "nan", "empty"):
            bad = region.copy()
            if corruption == "width":
                bad.loc[0, "coord0_max"] = bad.loc[0, "coord0_min"]
            elif corruption == "nan":
                bad.loc[0, "coord0_min"] = np.nan
            else:
                bad.loc[0, ["coord0_min", "coord1_min", "coord0_max", "coord1_max"]] = [
                    100,
                    100,
                    110,
                    110,
                ]
            with pytest.raises(ValueError):
                result.plot(regions=bad)

    def test_constant_large_values_and_zero_fields(self, plt):
        result = TestExplanationRegions._map(np.zeros((61, 61)))
        result.values[:] = 1e20
        fig = result.plot()
        fig.canvas.draw()
        assert fig.axes[0].images[0].norm.vmin < 1e20 < fig.axes[0].images[0].norm.vmax
        assert fig.axes[2].images[0].get_clim() == (-1, 1)
