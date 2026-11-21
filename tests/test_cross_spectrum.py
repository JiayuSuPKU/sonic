"""Synthetic local checks for cross-spectrum extraction."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from sonic import ComparatorGrid, ComparatorIrregular
from sonic.comparators.binning import _polar_bin_reducer
from sonic.comparators.cross_spectrum import _compute_cross_spectra
from tests.test_comparators import _grid_sample_to_adata, _install_grid_rasters
from tests.test_features import _polar_area_reference

NAMES = ["a", "b", "c"]
EDGES = np.array([0, 0.15, 0.3, 0.45])


_DEFAULT_OPTIONS = {
    "spacing": (1, 1),
    "freq_edges": EDGES,
    "feature_mode": "radial",
    "n_theta_bins": 4,
    "workers": 1,
}


@pytest.fixture
def make_comparator(monkeypatch):
    """Build either backend from one or more synthetic (gene, y, x) rasters."""

    def make(values, *, backend="fft", mode="radial", **kwargs):
        rasters = [values] if isinstance(values, np.ndarray) else values
        options = {**_DEFAULT_OPTIONS, "feature_mode": mode, **kwargs}
        if backend == "fft":
            return ComparatorGrid(
                _install_grid_rasters(monkeypatch, rasters, NAMES),
                bins="bins",
                table_name="table",
                col_key="col",
                row_key="row",
                **options,
            )
        return ComparatorIrregular(
            [_grid_sample_to_adata(v, NAMES, options["spacing"]) for v in rasters],
            grid_shape=[v.shape[-2:] for v in rasters],
            eps=1e-12,
            **options,
        )

    return make


def _irregular_points(values, coords, *, shape=(17, 19), mode="radial", **kwargs):
    sample = ad.AnnData(values, var=pd.DataFrame(index=NAMES))
    sample.obsm["spatial"] = coords
    return ComparatorIrregular(
        [sample],
        grid_shape=shape,
        eps=1e-12,
        **{**_DEFAULT_OPTIONS, "feature_mode": mode, **kwargs},
    )


def _cross_at_angle(comparator, pairs, *, angle_deg, **kwargs):
    """Use a fixed angle only for direct Fourier/reducer oracles."""
    comparator.rotation_angles_ = np.array([angle_deg], dtype=float)
    return _compute_cross_spectra(comparator, pairs, **kwargs)


class TestFourierConsistency:
    def test_extraction_checks_coherency_in_pair_blocks(self, make_comparator, monkeypatch):
        from sonic.comparators import _inference, cross_spectrum

        values = np.random.default_rng(43).normal(size=(3, 17, 19))
        comp = make_comparator(values)
        monkeypatch.setattr(_inference, "_COMPARISON_WORKSPACE_BYTES", 1)
        original = cross_spectrum.CrossSpectra._coherency_block
        checked = []

        def check(result, pairs):
            checked.append((pairs.start, pairs.stop))
            return original(result, pairs)

        monkeypatch.setattr(cross_spectrum.CrossSpectra, "_coherency_block", check)
        result = _compute_cross_spectra(comp, [("a", "b"), ("b", "c"), ("c", "a")])
        assert checked == [(0, 1), (1, 2), (2, 3)]
        valid = result.coherency_valid[:, -1]
        assert valid.any()
        result.cross_spectra[:, -1][valid] = 1e10
        with pytest.raises(ArithmeticError, match="coherence bound"):
            original(result, slice(2, 3))

    @pytest.mark.parametrize("mode", ["radial", "2d"])
    @pytest.mark.parametrize("shape", [(17, 19), (16, 18)])
    def test_fft_nufft_chunks_and_order(self, make_comparator, mode, shape):
        values = np.random.default_rng(5).normal(size=(3, *shape))
        pairs = [("b", "a"), ("b", "c"), ("c", "a"), ("a", "a")]
        fft = _cross_at_angle(
            make_comparator(values, mode=mode), pairs, angle_deg=11.0, chunk_size=1
        )
        nufft = _cross_at_angle(
            make_comparator(values, backend="nufft", mode=mode), pairs, angle_deg=11.0, chunk_size=3
        )
        for key in ("cross_spectra", "auto_spectra", "coherency", "coherence"):
            np.testing.assert_allclose(getattr(fft, key), getattr(nufft, key), atol=1e-10)
        np.testing.assert_array_equal(fft.frequency_bin_intervals, np.c_[EDGES[:-1], EDGES[1:]])
        np.testing.assert_allclose(fft.cross_spectra[:, 3], fft.auto_spectra[:, 0], atol=1e-12)
        np.testing.assert_allclose(fft.coherence[:, 3], 1, atol=1e-12)
        reversed_pair = _cross_at_angle(
            make_comparator(values, backend="nufft", mode=mode), [("a", "b")], angle_deg=11.0
        )
        np.testing.assert_allclose(
            reversed_pair.cross_spectra[:, 0], nufft.cross_spectra[:, 0].conj(), atol=1e-12
        )

    def test_direct_fourier_oracle_translation_and_r_identity(self):
        rng = np.random.default_rng(12)
        coords = rng.uniform(0, 12, (71, 2))
        values = rng.normal(size=(71, 3))
        result = (
            _irregular_points(values, coords).compute_cross_spectra([("a", "b")]).cross_spectra_
        )
        shifted = (
            _irregular_points(values, coords + [75.2, -13.7])
            .compute_cross_spectra([("a", "b")])
            .cross_spectra_
        )
        np.testing.assert_allclose(result.cross_spectra, shifted.cross_spectra, atol=1e-10)
        fy, fx = np.meshgrid(np.fft.fftfreq(17), np.fft.fftfreq(19), indexing="ij")
        frequencies = np.c_[fy.ravel(), fx.ravel()]
        basis = np.exp(-2j * np.pi * (coords @ frequencies.T))
        z = values - values.mean(axis=0)
        coefficients = z.T @ basis / len(z)
        radius = np.linalg.norm(frequencies, axis=1)
        total = 0.0
        weights = np.zeros(len(radius))
        for b, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:], strict=True)):
            keep = (radius > 0) & (radius >= lo) & (radius < hi)
            cross_spectra = coefficients[0, keep] * coefficients[1, keep].conj()
            np.testing.assert_allclose(
                result.cross_spectra[0, 0, b], cross_spectra.mean().real, atol=1e-10
            )
            # Independent spatial-domain kernel oracle, with a different weight per band.
            weights[keep] = b + 1
            total += (b + 1) * keep.sum() * result.cross_spectra[0, 0, b].real
        kernel = (basis * weights) @ basis.conj().T / len(radius)
        expected_r = z[:, 0] @ kernel.real @ z[:, 1]
        np.testing.assert_allclose(total * len(z) ** 2 / len(radius), expected_r, atol=1e-8)

    @pytest.mark.parametrize("backend", ["fft", "nufft"])
    def test_radial_nyquist_power_is_retained_in_both_paths(self, make_comparator, backend):
        shape = (8, 10)
        yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        values = np.array([(-1.0) ** xx, (-1.0) ** yy, np.ones(shape)])
        comparator = make_comparator(values, backend=backend)
        comparator.freq_edges = np.array([0, 0.25, 0.5])
        comparator.compute_spectra(progress=False)
        result = comparator.compute_cross_spectra(
            [("a", "a"), ("b", "b")], progress=False
        ).cross_spectra_
        np.testing.assert_allclose(result.auto_spectra, comparator.spectra_[:, :2], atol=1e-10)
        assert np.all(result.auto_spectra[0, :, -1] > 0.01)
        np.testing.assert_allclose(result.cross_spectra[0].real, result.auto_spectra[0], atol=1e-10)
        assert np.all(result.cross_spectra.imag == 0)

    def test_radial_nufft_nyquist_matches_direct_cospectrum(self):
        rng = np.random.default_rng(72)
        shape = (8, 10)
        coords = rng.uniform(size=(53, 2)) * shape
        values = rng.normal(size=(53, 3))
        comparator = _irregular_points(values, coords, shape=shape)
        comparator.freq_edges = np.array([0, 0.25, 0.5])
        result = comparator.compute_cross_spectra([("a", "b")], progress=False).cross_spectra_
        fy, fx = np.meshgrid(np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1]), indexing="ij")
        basis = np.exp(-2j * np.pi * (fy[..., None] * coords[:, 0] + fx[..., None] * coords[:, 1]))
        f = basis @ (values - values.mean(axis=0)) / len(values)
        radius = np.hypot(fy, fx)
        boundary = (radius >= 0.25) & (radius <= 0.5)
        cross_spectra = (f[..., 0] * f[..., 1].conj())[boundary].mean()
        assert abs(cross_spectra.imag) > 1e-6
        np.testing.assert_allclose(result.cross_spectra[0, 0, -1], cross_spectra.real, atol=1e-10)
        for feature in (0, 1):
            np.testing.assert_allclose(
                result.auto_spectra[0, feature, -1],
                (abs(f[..., feature]) ** 2)[boundary].mean(),
                atol=1e-10,
            )
        assert np.all(result.coherence[result.coherency_valid] <= 1 + 1e-12)


class TestPolarBinning:
    def test_angular_phase_seam_and_half_turn(self, make_comparator):
        rng = np.random.default_rng(23)
        values = rng.normal(size=(3, 33, 33))
        values[1] = np.roll(values[0], 3, axis=1)
        comparator = make_comparator(values, mode="2d")
        result = _cross_at_angle(comparator, [("a", "b")], angle_deg=0)
        half_turn = _cross_at_angle(comparator, [("a", "b")], angle_deg=180)
        np.testing.assert_allclose(half_turn.cross_spectra, result.cross_spectra.conj(), atol=1e-12)
        f = np.fft.fft2(values[:2] - values[:2].mean(axis=(1, 2), keepdims=True)) / (33 * 33)
        f[:, 0, 0] = 0
        expected = _polar_area_reference(f[0] * f[1].conj(), (33, 33), (1, 1), EDGES, 4, 0)
        np.testing.assert_allclose(result.cross_spectra[0, 0], expected, atol=1e-12)
        for feature in range(2):
            expected_auto = _polar_area_reference(
                abs(f[feature]) ** 2, (33, 33), (1, 1), EDGES, 4, 0
            )
            np.testing.assert_allclose(result.auto_spectra[0, feature], expected_auto, atol=1e-12)
        assert np.max(np.abs(result.cross_spectra.imag)) > 1e-5
        radial = make_comparator(values).compute_cross_spectra([("a", "b")]).cross_spectra_
        assert np.all(radial.cross_spectra.imag == 0)

    @pytest.mark.parametrize("n_theta", [1, 2, 3, 6, 18, 36])
    @pytest.mark.parametrize("padding", [0, 1])
    def test_interpolation_weights_preserve_conjugacy_and_bound(self, n_theta, padding):
        shape = (16, 19)
        spectrum_shape = tuple(n + 2 * padding for n in shape)
        f = np.fft.fft2(np.random.default_rng(41).normal(size=(2, *spectrum_shape)))
        f[:, 0, 0] = 0
        product = (f[0] * f[1].conj()).ravel()
        auto_spectra = (abs(f) ** 2).reshape(2, -1)
        for rotation in (0, 2.5, 10, 45, 90):
            real, mode_pair_counts = _polar_bin_reducer(
                shape,
                spacing=(1, 1),
                edges=EDGES,
                n_theta=n_theta,
                angle_deg=rotation,
                frequency_padding=padding,
            )
            turned, turned_counts = _polar_bin_reducer(
                shape,
                spacing=(1, 1),
                edges=EDGES,
                n_theta=n_theta,
                angle_deg=rotation + 180,
                frequency_padding=padding,
            )
            assert np.all(real.data >= 0)
            np.testing.assert_allclose(np.asarray(real.sum(axis=1)).ravel(), 1)
            assert np.all(mode_pair_counts == mode_pair_counts.astype(int))
            np.testing.assert_array_equal(mode_pair_counts, turned_counts)
            np.testing.assert_allclose(turned @ product, (real @ product).conj(), atol=1e-10)
            denominator = (real @ auto_spectra[0]) * (real @ auto_spectra[1])
            assert np.all(abs(real @ product) ** 2 <= denominator * (1 + 1e-12))

    @pytest.mark.parametrize("shape,spacing", [((8, 10), (1, 1)), ((9, 7), (1, 1.5))])
    @pytest.mark.parametrize("rotation", [0, 73, 180])
    def test_nufft_interpolation_matches_direct_products_at_frequency_boundary(
        self, shape, spacing, rotation
    ):
        rng = np.random.default_rng(42)
        coords = rng.uniform(size=(43, 2)) * np.array(shape) * spacing
        values = rng.normal(size=(43, 3))
        edges = np.linspace(0, 0.5 / max(spacing), 5)
        comparator = _irregular_points(
            values,
            coords,
            shape=shape,
            spacing=spacing,
            freq_edges=edges,
            mode="2d",
            n_theta_bins=12,
        )
        result = _cross_at_angle(comparator, [("a", "b")], angle_deg=rotation)
        frequencies = [
            (np.arange(n + 2) - (n + 2) // 2) / (n * d) for n, d in zip(shape, spacing, strict=True)
        ]
        fy, fx = np.meshgrid(*frequencies, indexing="ij")
        basis = np.exp(-2j * np.pi * (fy[..., None] * coords[:, 0] + fx[..., None] * coords[:, 1]))
        f = basis @ (values - values.mean(axis=0)) / len(values)
        f = np.fft.ifftshift(f, axes=(0, 1))
        f[0, 0] = 0
        expected = _polar_area_reference(
            f[..., 0] * f[..., 1].conj(), shape, spacing, edges, 12, rotation, padded=True
        )
        keep = comparator._supported_frequency_bin_mask
        expected = expected.reshape(len(keep), 12)[keep].ravel()
        np.testing.assert_allclose(result.cross_spectra[0, 0], expected, atol=1e-10)
        for feature in range(2):
            expected_auto = _polar_area_reference(
                abs(f[..., feature]) ** 2, shape, spacing, edges, 12, rotation, padded=True
            )
            expected_auto = expected_auto.reshape(len(keep), 12)[keep].ravel()
            np.testing.assert_allclose(result.auto_spectra[0, feature], expected_auto, atol=1e-10)
        assert result.coherency_valid.all()

    @pytest.mark.parametrize("rotation", [0, 17, 61])
    def test_fine_angular_bins_remain_supported_without_unit_coherence(
        self, make_comparator, rotation
    ):
        shape = (9, 11)
        values = np.random.default_rng(43).normal(size=(3, *shape))
        comparator = make_comparator(values, mode="2d")
        comparator._n_theta_bins = 36
        result = _cross_at_angle(comparator, [("a", "b")], angle_deg=rotation)
        assert result.coherency_valid.all()
        assert np.all(result.mode_pair_counts >= 2)
        assert np.all(result.coherence <= 1 + 1e-12)
        assert np.median(result.coherence) < 0.9
        # The lowest annulus has too few source modes to fill 36 disjoint sectors.
        fy, fx = np.meshgrid(np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1]), indexing="ij")
        radius = np.hypot(fy, fx)
        assert ((radius > 0) & (radius < EDGES[1])).sum() / 2 < 36

    def test_unsupported_frequency_bands_are_dropped(self, make_comparator):
        comparator = make_comparator(np.random.default_rng(44).normal(size=(3, 9, 11)), mode="2d")
        comparator.freq_edges = np.array([0, 0.2, 0.45, 0.6, 0.8])
        result = _cross_at_angle(comparator, [("a", "b")], angle_deg=23)
        assert result.cross_spectra.shape == (1, 1, 8)
        assert result.coherency_valid.all()
        np.testing.assert_array_equal(result.frequency_bin_intervals, [[0, 0.2], [0.2, 0.45]])
        comparator.freq_edges = np.array([0.6, 0.8])
        with pytest.raises(RuntimeError, match="No radial frequency bins"):
            _cross_at_angle(comparator, [("a", "b")], angle_deg=23)


class TestCoherency:
    def test_coherence_sign_scaling_and_degenerate_input(self, make_comparator):
        rng = np.random.default_rng(15)
        values = rng.normal(size=(3, 17, 19))
        values[1] = -4 * values[0]
        values[2] = 1
        comparator = make_comparator(values, backend="nufft")
        r = comparator.compute_cross_spectra([("a", "b"), ("a", "c")]).cross_spectra_
        np.testing.assert_allclose(r.coherency[0, 0], -1, atol=1e-10)
        np.testing.assert_allclose(r.coherence[0, 0], 1, atol=1e-10)
        assert not r.coherency_valid[0, 1].any()
        values[0] *= 300
        scaled = (
            make_comparator(values, backend="nufft")
            .compute_cross_spectra([("a", "b")])
            .cross_spectra_
        )
        np.testing.assert_allclose(scaled.coherency[0, 0], r.coherency[0, 0], atol=1e-10)

    def test_coherency_safeguards_preserve_raw_spectra(self, make_comparator):
        yy, xx = np.meshgrid(np.arange(17), np.arange(19), indexing="ij")
        low = np.cos(2 * np.pi * xx / 19)
        high = np.cos(10 * np.pi * xx / 19)
        values = np.array([low + 1e-8 * high, low, high])
        comparator = make_comparator(values)
        result = comparator.compute_cross_spectra([("a", "a")], progress=False).cross_spectra_
        assert np.isfinite(result.auto_spectra).all()
        assert np.isfinite(result.cross_spectra).all()
        assert result.auto_spectra[0, 0, 1] > 0
        assert not result.coherency_valid[0, 0, 1]
        assert np.isnan(result.coherency[0, 0, 1])
        np.testing.assert_allclose(result.cross_spectra[0, 0].real, result.auto_spectra[0, 0])
        comparator.freq_edges = np.array([0.05, 0.056])
        result = comparator.compute_cross_spectra([("a", "a")], progress=False).cross_spectra_
        assert result.mode_pair_counts[0, 0] == 1
        assert np.isfinite(result.cross_spectra).all()
        assert not result.coherency_valid.any()


class TestInputs:
    @pytest.mark.parametrize("backend", ["fft", "nufft"])
    @pytest.mark.parametrize(
        "chunk", [0, -1, True, False, 1.5, 2.0, "invalid", np.bool_(True), np.array([1, 2])]
    )
    def test_constructor_and_compute_methods_reject_invalid_chunks(
        self, make_comparator, backend, chunk
    ):
        values = np.random.default_rng(122).normal(size=(3, 17, 19))
        with pytest.raises(ValueError, match="positive integer"):
            make_comparator(values, backend=backend, **{f"{backend}_chunk_size": chunk})
        comparator = make_comparator(values, backend=backend)
        with pytest.raises(ValueError, match="positive integer"):
            comparator.compute_spectra(chunk_size=chunk, progress=False)
        with pytest.raises(ValueError, match="positive integer"):
            comparator.compute_cross_spectra([("a", "b")], chunk_size=chunk, progress=False)

    @pytest.mark.parametrize("backend", ["fft", "nufft"])
    @pytest.mark.parametrize("chunk", [1, np.int64(2), "auto"])
    def test_constructor_and_compute_methods_accept_valid_chunks(
        self, make_comparator, backend, chunk
    ):
        values = np.random.default_rng(123).normal(size=(3, 17, 19))
        comparator = make_comparator(values, backend=backend, **{f"{backend}_chunk_size": chunk})
        comparator.compute_spectra(chunk_size=chunk, progress=False)
        comparator.compute_cross_spectra([("a", "b")], chunk_size=chunk, progress=False)
        np.testing.assert_allclose(
            comparator.cross_spectra_.auto_spectra, comparator.spectra_[:, :2], atol=1e-10
        )
        assert getattr(comparator, f"_{backend}_chunk_size_spec") == chunk

    def test_shared_holes_match_observed_point_transform(self, make_comparator):
        values = np.random.default_rng(9).normal(size=(3, 17, 19))
        values[:, 1::3, 1::3] = np.nan
        sample = _grid_sample_to_adata(values, NAMES)
        observed = np.isfinite(sample.X).all(axis=1)
        nufft = _irregular_points(sample.X[observed], sample.obsm["spatial"][observed])
        a = make_comparator(values).compute_cross_spectra([("a", "b")], chunk_size=1).cross_spectra_
        b = nufft.compute_cross_spectra([("a", "b")]).cross_spectra_
        np.testing.assert_allclose(a.cross_spectra, b.cross_spectra, atol=1e-10)
        values[1, 0, 0] = np.nan
        with pytest.raises(ValueError, match="identical finite support"):
            make_comparator(values).compute_cross_spectra([("a", "b")], chunk_size=1)

    def test_sparse_layer_gene_order_and_coordinate_units(self):
        rng = np.random.default_rng(34)
        values = rng.normal(size=(73, 3))
        coords = rng.uniform(0, 12, size=(73, 2))
        pairs = [("a", "c"), ("b", "a")]
        reference = _cross_at_angle(
            _irregular_points(values, coords, mode="2d"), pairs, angle_deg=17
        )
        order = [2, 0, 1]
        sample = ad.AnnData(np.zeros_like(values), var=pd.DataFrame(index=np.array(NAMES)[order]))
        sample.layers["signal"] = csr_matrix(values[:, order])
        sample.obsm["spatial"] = coords * 2
        original = sample.layers["signal"].copy()
        comparator = ComparatorIrregular(
            [sample],
            layer="signal",
            unit_scales=[0.5],
            grid_shape=(17, 19),
            spacing=(1, 1),
            freq_edges=EDGES,
            feature_mode="2d",
            n_theta_bins=4,
            eps=1e-12,
            workers=1,
        )
        result = _cross_at_angle(comparator, pairs, angle_deg=17, chunk_size=1)
        np.testing.assert_allclose(result.cross_spectra, reference.cross_spectra, atol=1e-10)
        np.testing.assert_allclose(result.coherency, reference.coherency, atol=1e-10)
        np.testing.assert_array_equal(sample.layers["signal"].toarray(), original.toarray())
        np.testing.assert_array_equal(sample.X, 0)
        assert result.gene_names == ("c", "a", "b")
        np.testing.assert_array_equal(result.pair_indices, [[1, 0], [2, 1]])

    @pytest.mark.parametrize(
        "pairs, kwargs, message",
        [
            ([], {}, "at least one pair"),
            (["ab"], {}, "at least one pair"),
            ([("a",)], {}, "at least one pair"),
            ([("a", "missing")], {}, "must exist"),
            ([("a", "b"), ("b", "a")], {}, "Duplicate"),
            ([("a", "b")], {"chunk_size": 1.5}, "positive integer"),
        ],
    )
    def test_invalid_requests_fail_clearly(self, make_comparator, pairs, kwargs, message):
        comparator = make_comparator(
            np.random.default_rng(37).normal(size=(3, 17, 19)), backend="nufft"
        )
        with pytest.raises(ValueError, match=message):
            comparator.compute_cross_spectra(pairs, **kwargs)

    def test_nonfinite_values_are_rejected(self, make_comparator):
        values = np.random.default_rng(15).normal(size=(3, 17, 19))
        values[0, 0, 0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            make_comparator(values, backend="nufft").compute_cross_spectra([("a", "b")])


class TestChunking:
    @pytest.mark.parametrize("mode", ["radial", "2d"])
    @pytest.mark.parametrize("backend", ["fft", "nufft"])
    def test_chunk_sizes_and_budget_give_identical_features(self, make_comparator, mode, backend):
        values = np.random.default_rng(32).normal(size=(3, 17, 19))
        comparator = make_comparator(values, backend=backend, mode=mode)
        pairs = [("c", "b"), ("a", "c"), ("b", "a"), ("b", "b")]
        reference = _cross_at_angle(comparator, pairs, angle_deg=23, chunk_size=3)
        budget = reference.chunk_schedules[0]["estimated_workspace_bytes"] - 128 * (
            reference.chunk_schedules[0]["n_locations"] + reference.chunk_schedules[0]["n_freqs"]
        )
        for chunk in (1, 2, "auto"):
            result = _cross_at_angle(
                comparator, pairs, angle_deg=23, chunk_size=chunk, memory_budget_bytes=budget
            )
            np.testing.assert_allclose(result.cross_spectra, reference.cross_spectra, atol=1e-10)
            np.testing.assert_allclose(result.auto_spectra, reference.auto_spectra, atol=1e-10)
            np.testing.assert_array_equal(result.coherency_valid, reference.coherency_valid)
            assert result.chunk_schedules[0]["estimated_workspace_bytes"] <= budget
            assert result.chunk_schedules[0]["chunk_size"] <= 2
        override = _cross_at_angle(
            comparator, pairs, angle_deg=23, chunk_size=3, memory_budget_bytes=1
        )
        np.testing.assert_allclose(override.cross_spectra, reference.cross_spectra, atol=1e-10)
        np.testing.assert_allclose(override.auto_spectra, reference.auto_spectra, atol=1e-10)
        assert override.chunk_schedules[0]["chunk_size"] == 3
        assert override.chunk_schedules[0]["estimated_workspace_bytes"] > 1

    def test_auto_chunk_requires_sufficient_budget(self, make_comparator):
        comparator = make_comparator(
            np.random.default_rng(15).normal(size=(3, 17, 19)), backend="nufft"
        )
        with pytest.raises(MemoryError):
            comparator.compute_cross_spectra([("a", "b")], memory_budget_bytes=100)

    def test_fft_stream_releases_rasters_and_coefficient_blocks(self, make_comparator, monkeypatch):
        import weakref

        import sonic.comparators.cross_spectrum as cross_module

        values = np.random.default_rng(36).normal(size=(3, 17, 19))
        comparator = make_comparator([values] * 3)
        rasterize = comparator._rasterize_one
        coefficient_transform = cross_module._fourier_coefficients_2d
        rasters, coefficients, observed_chunks = [], [], []

        def checked_rasterize(sample):
            assert not any(r() is not None for r in rasters)
            assert not any(r() is not None for r in coefficients)
            result = rasterize(sample)
            rasters.append(weakref.ref(result))
            return result

        def checked_transform(values, *args, **kwargs):
            assert sum(r() is not None for r in coefficients) <= 1
            observed_chunks.append(values.shape[-1])
            result = coefficient_transform(values, *args, **kwargs)
            coefficients.append(weakref.ref(result))
            return result

        monkeypatch.setattr(comparator, "_rasterize_one", checked_rasterize)
        monkeypatch.setattr(cross_module, "_fourier_coefficients_2d", checked_transform)
        result = comparator.compute_cross_spectra(
            [("a", "b"), ("a", "c"), ("b", "c")], chunk_size=1
        ).cross_spectra_
        assert result.cross_spectra.shape[0] == 3
        assert observed_chunks and set(observed_chunks) == {1}
        assert not any(r() is not None for r in rasters + coefficients)

    @pytest.mark.parametrize("backend", ["fft", "nufft"])
    def test_explicit_chunk_overrides_budget_during_initial_alignment(
        self, make_comparator, backend
    ):
        values = np.random.default_rng(63).normal(size=(3, 17, 19))
        comparator = make_comparator(values, backend=backend, mode="2d")
        setting = f"_{backend}_chunk_size_spec"
        configured = getattr(comparator, setting)
        comparator.compute_cross_spectra(
            [("a", "b")],
            n_jobs=1,
            progress=False,
            chunk_size=1,
            memory_budget_bytes=1,
        )
        assert comparator.rotation_angles_ is not None
        assert comparator.cross_spectra_.chunk_schedules[0]["chunk_size"] == 1
        assert getattr(comparator, setting) == configured


class TestComparatorIntegration:
    @pytest.mark.parametrize("mode", ["radial", "2d"])
    @pytest.mark.parametrize("adaptive", [False, True])
    @pytest.mark.parametrize("backend", ["fft", "nufft"])
    def test_fitted_auto_state_and_resolved_grid_are_preserved(
        self, make_comparator, mode, adaptive, backend
    ):
        rng = np.random.default_rng(35)
        values = [rng.normal(size=(3, 17, 19)), rng.normal(size=(3, 9, 11))]
        comparator = make_comparator(
            values,
            backend=backend,
            mode=mode,
            freq_edges=None,
            n_radial_bins=16,
            adaptive_freq_edges=adaptive,
        )
        comparator.compute_spectra(n_jobs=1, progress=False, landmark_genes=["a"])
        state = {
            key: getattr(comparator, key).copy() if getattr(comparator, key) is not None else None
            for key in (
                "spectra_",
                "dc_",
                "presence_",
                "rotation_angles_",
                "freq_edges",
                "freq_edges_requested_",
                "frequency_bin_intervals_",
                "frequency_bin_counts_",
            )
        }
        result = comparator.compute_cross_spectra([("a", "b")]).cross_spectra_
        for key, expected in state.items():
            np.testing.assert_equal(getattr(comparator, key), expected)
        np.testing.assert_array_equal(
            result.frequency_bin_intervals, comparator.frequency_bin_intervals_
        )
        n_bins = comparator.spectra_.shape[-1]
        assert result.cross_spectra.shape == (2, 1, n_bins)
        if mode == "2d":
            np.testing.assert_array_equal(result.rotation_angles_deg, comparator.rotation_angles_)
        assert result.auto_spectra.shape == (2, len(result.gene_names), n_bins)
        comparator.spectra_ *= 10
        repeat = comparator.compute_cross_spectra([("a", "b")]).cross_spectra_
        np.testing.assert_allclose(repeat.cross_spectra, result.cross_spectra)

    @pytest.mark.parametrize("backend", ["fft", "nufft"])
    @pytest.mark.parametrize("shape,spacing", [((16, 18), (1.0, 1.0)), ((17, 19), (1.5, 0.5))])
    @pytest.mark.parametrize("mode", ["radial", "2d"])
    @pytest.mark.parametrize("normalization", ["background", "covariates"])
    def test_standard_autos_match_cross_spectrum_autos(
        self, make_comparator, backend, shape, spacing, mode, normalization
    ):
        values = np.random.default_rng(53).normal(size=(3, *shape))
        comparator = make_comparator(
            values,
            backend=backend,
            mode=mode,
            spacing=spacing,
            freq_edges=np.linspace(0, 0.5 / max(spacing), 9),
            n_theta_bins=12,
        )
        comparator.compute_spectra(progress=False, landmark_genes=["a"])
        result = comparator.compute_cross_spectra(
            [("a", "b"), ("c", "c")], chunk_size=1
        ).cross_spectra_
        np.testing.assert_allclose(result.auto_spectra, comparator.spectra_, atol=1e-10)
        np.testing.assert_array_equal(
            result.frequency_bin_intervals, comparator.frequency_bin_intervals_
        )
        np.testing.assert_allclose(
            result.cross_spectra[0, 1].real, result.auto_spectra[0, 2], atol=1e-10
        )
        if normalization == "background":
            comparator.normalize_background()
        else:
            covariates = [np.random.default_rng(54).normal(size=(1, *shape))]
            comparator.normalize_covariates(covariates)
        normalized = comparator.spectra_.copy()
        assert not np.allclose(normalized, result.auto_spectra)
        repeated = comparator.compute_cross_spectra([("a", "b"), ("c", "c")]).cross_spectra_
        np.testing.assert_array_equal(comparator.spectra_, normalized)
        np.testing.assert_allclose(repeated.auto_spectra, result.auto_spectra, atol=1e-10)
        np.testing.assert_allclose(repeated.cross_spectra, result.cross_spectra, atol=1e-10)
        np.testing.assert_allclose(repeated.coherency, result.coherency, atol=1e-10)

    @pytest.mark.parametrize("backend", ["fft", "nufft"])
    def test_polar_cross_uses_one_shared_alignment_per_sample(
        self, make_comparator, monkeypatch, backend
    ):
        import sonic.comparators.base as base_module

        first = np.random.default_rng(81).normal(size=(3, 33, 33))
        values = [first, 1.2 * np.rot90(first, axes=(-2, -1))]
        comparator = make_comparator(values, backend=backend, mode="2d", n_theta_bins=12)
        estimate = base_module._estimate_rotations_from_polar
        fits = []

        def tracked_estimate(*args, **kwargs):
            angles = estimate(*args, **kwargs)
            fits.append(angles.copy())
            return angles

        monkeypatch.setattr(base_module, "_estimate_rotations_from_polar", tracked_estimate)
        comparator.compute_cross_spectra([("b", "c")], landmark_genes=["a"], progress=False)
        assert len(fits) == 1
        assert comparator.rotation_angles_[0] == 0
        assert comparator.rotation_angles_[1] % 180 == pytest.approx(90, abs=1)
        assert comparator.spectra_ is comparator.dc_ is comparator.presence_ is None
        result = comparator.cross_spectra_
        np.testing.assert_array_equal(result.rotation_angles_deg, fits[0])
        comparator.compute_cross_spectra(
            [("a", "c"), ("c", "b")], landmark_genes=["a"], progress=False
        )
        assert len(fits) == 1
        assert comparator.spectra_ is None
        assert comparator.cross_spectra_ is not result
        np.testing.assert_array_equal(comparator.cross_spectra_.rotation_angles_deg, fits[0])
        comparator.compute_spectra(progress=False)
        assert len(fits) == 1  # Autos reuse the alignment established by a standalone cross call.
        np.testing.assert_allclose(result.auto_spectra, comparator.spectra_[:, [1, 2]], atol=1e-10)
        comparator.normalize_background()
        normalized = comparator.spectra_
        comparator.compute_cross_spectra([("b", "c")], landmark_genes=["a"], progress=False)
        assert len(fits) == 1
        assert comparator.spectra_ is normalized
        result = comparator.cross_spectra_
        with pytest.raises(ValueError, match="compute_spectra"):
            comparator.compute_cross_spectra([("b", "c")], landmark_genes=["b"], progress=False)
        assert comparator.spectra_ is normalized
        assert comparator.cross_spectra_ is result
        comparator.compute_spectra(landmark_genes=["b"], progress=False)
        assert len(fits) == 2
        assert comparator.cross_spectra_ is None
        comparator.compute_cross_spectra([("b", "c")], progress=False)
        assert len(fits) == 2
        np.testing.assert_array_equal(comparator.cross_spectra_.rotation_angles_deg, fits[1])
        np.testing.assert_allclose(
            comparator.cross_spectra_.auto_spectra, comparator.spectra_[:, [1, 2]], atol=1e-10
        )

    def test_single_sample_alignment(self, make_comparator):
        comparator = make_comparator(
            np.random.default_rng(15).normal(size=(3, 17, 19)), backend="nufft", mode="2d"
        )
        result = comparator.compute_cross_spectra([("a", "b")], progress=False).cross_spectra_
        assert result.rotation_angles_deg.shape == (1,)

    @pytest.mark.parametrize("angles", [[0, 1], [np.nan]])
    def test_shared_rotation_angles_are_validated(self, make_comparator, angles):
        comparator = make_comparator(
            np.random.default_rng(82).normal(size=(3, 17, 19)), backend="nufft", mode="2d"
        )
        comparator.compute_spectra(progress=False)
        comparator.rotation_angles_ = np.asarray(angles)
        with pytest.raises(ValueError, match="finite shared rotation angle per sample"):
            comparator.compute_cross_spectra([("a", "b")], progress=False)

    @pytest.mark.parametrize("mode", ["radial", "2d"])
    @pytest.mark.parametrize("backend", ["fft", "nufft"])
    def test_fit_style_api_alignment_parallelism_and_subset(self, make_comparator, mode, backend):
        values = np.random.default_rng(62).normal(size=(3, 17, 19))
        comparator = make_comparator([values, values * 2], backend=backend, mode=mode)
        returned = comparator.compute_cross_spectra(
            [("a", "b")],
            n_jobs=2,
            landmark_genes=["a"] if mode == "2d" else None,
            progress=False,
        )
        assert returned is comparator
        result = comparator.cross_spectra_
        assert result.cross_spectra.shape == (2, 1, 3 * (4 if mode == "2d" else 1))
        np.testing.assert_array_equal(
            result.frequency_bin_intervals, comparator.frequency_bin_intervals_
        )
        comparator.compute_cross_spectra([("a", "b")], n_jobs=1, progress=False)
        np.testing.assert_allclose(
            comparator.cross_spectra_.cross_spectra, result.cross_spectra, atol=1e-10
        )
        subset = comparator.subset([1, 0, 1])
        np.testing.assert_array_equal(
            subset.cross_spectra_.cross_spectra, result.cross_spectra[[1, 0, 1]]
        )
        np.testing.assert_array_equal(
            subset.cross_spectra_.auto_spectra, result.auto_spectra[[1, 0, 1]]
        )
        assert subset.cross_spectra_.chunk_schedules == [
            result.chunk_schedules[i] for i in [1, 0, 1]
        ]
        comparator.compute_spectra(n_jobs=1, progress=False)
        assert comparator.cross_spectra_ is None

    def test_public_cross_signature_matches_fit_arguments(self):
        import inspect

        auto_spectra = inspect.signature(ComparatorGrid.compute_spectra).parameters
        cross_spectra = inspect.signature(ComparatorGrid.compute_cross_spectra).parameters
        for name in ("n_jobs", "landmark_genes", "progress", "memory_budget_bytes", "chunk_size"):
            assert cross_spectra[name].default == auto_spectra[name].default
        assert not {
            "min_modes",
            "power_rtol",
            "rotation_angles",
            "directed_orientation",
        }.intersection(cross_spectra)


class TestAlignment:
    @pytest.mark.parametrize("backend", ["fft", "rfft2", "nufft"])
    @pytest.mark.parametrize("landmarks", [None, ["c", "a"]])
    def test_compact_alignment_matches_dense_landmarks(
        self, make_comparator, monkeypatch, backend, landmarks
    ):
        import sonic.comparators.base as base_module
        from sonic.comparators.features import (
            _build_landmark_polar_stack,
            estimate_rotations_from_landmarks,
        )

        rng = np.random.default_rng(114)
        values = [rng.normal(size=(3, 17, 19)), rng.normal(size=(3, 21, 23))]
        for value in values:
            value[1] = 0  # Absent genes remain in the default geometric mean.
        solver = "rfft2" if backend == "rfft2" else "fft2"
        options = {"fft_solver": solver} if backend != "nufft" else {}
        comparator = make_comparator(
            values,
            backend="fft" if backend == "rfft2" else backend,
            mode="2d",
            presence_threshold=0.5,
            **options,
        )
        expected = []
        shapes = [value.shape[-2:] for value in values]
        for value, shape, sample_index in zip(values, shapes, range(2), strict=True):
            if backend != "nufft":
                from sonic.comparators.grid import _grid_spectrum

                power = _grid_spectrum(value, fft_solver=solver, workers=1)
            else:
                from sonic.kernels.nufft import power_spectrum_2d_nufft

                block = value.reshape(3, -1).T.copy()
                block -= block.mean(axis=0)
                power = (
                    np.moveaxis(
                        power_spectrum_2d_nufft(
                            comparator._coords[sample_index],
                            block,
                            grid_shape=shape,
                            spacing=(1, 1),
                            eps=1e-12,
                            nthreads=1,
                            frequency_padding=1,
                        ),
                        -1,
                        0,
                    )
                    / block.shape[0] ** 2
                )
            expected.append(
                np.exp(np.log(power + 1e-12).mean(axis=0))[None]
                if landmarks is None
                else power[[2, 0]]
            )
        padding = int(backend == "nufft")
        expected_angles = estimate_rotations_from_landmarks(
            expected, shapes, fft_solver=solver, frequency_padding=padding
        )
        expected_polar = [
            _build_landmark_polar_stack(
                p, shape, solver, 180, np.linspace(0, 0.5, 65), (1, 1), padding
            )
            for p, shape in zip(expected, shapes, strict=True)
        ]
        estimate = base_module._estimate_rotations_from_polar

        def check_polar(polar, **kwargs):
            for actual, reference in zip(polar, expected_polar, strict=True):
                np.testing.assert_allclose(actual, reference, rtol=1e-11, atol=1e-14)
            return estimate(polar, **kwargs)

        monkeypatch.setattr(base_module, "_estimate_rotations_from_polar", check_polar)
        comparator.compute_cross_spectra(
            [("a", "c")], landmark_genes=landmarks, chunk_size=2, progress=False
        )
        np.testing.assert_array_equal(comparator.rotation_angles_, expected_angles)
        assert comparator.spectra_ is None

    @pytest.mark.parametrize("backend", ["fft", "nufft"])
    @pytest.mark.parametrize("landmarks,expected_batches", [(None, [2, 1]), (["c", "a"], [2])])
    def test_only_requested_landmarks_are_transformed(
        self, make_comparator, monkeypatch, backend, landmarks, expected_batches
    ):
        import sonic.comparators.grid as grid_module
        import sonic.kernels.nufft as nufft_module

        values = np.random.default_rng(115).normal(size=(3, 17, 19))
        comparator = make_comparator([values, values * 2], backend=backend, mode="2d")
        module, name = (
            (grid_module, "_grid_spectrum")
            if backend == "fft"
            else (nufft_module, "power_spectrum_2d_nufft")
        )
        transform = getattr(module, name)
        batches = []

        def tracked(*args, **kwargs):
            batches.append(args[0].shape[0] if backend == "fft" else args[1].shape[1])
            return transform(*args, **kwargs)

        monkeypatch.setattr(module, name, tracked)
        comparator.compute_cross_spectra(
            [("a", "b")], landmark_genes=landmarks, chunk_size=2, n_jobs=1, progress=False
        )
        assert batches == expected_batches * 2
        batches.clear()
        comparator.compute_cross_spectra([("a", "c")], landmark_genes=landmarks, progress=False)
        assert batches == []  # Changing pairs does not repeat alignment transforms.
        comparator.compute_spectra(chunk_size=2, n_jobs=1, progress=False)
        assert batches == [2, 1] * 2  # Only the auto-feature pass remains.

    @pytest.mark.parametrize("backend", ["fft", "nufft"])
    def test_single_sample_skips_landmark_pass(self, make_comparator, monkeypatch, backend):
        comparator = make_comparator(
            np.random.default_rng(116).normal(size=(3, 17, 19)), backend=backend, mode="2d"
        )

        def unexpected(*args, **kwargs):
            pytest.fail("A single sample needs no landmark transforms")

        monkeypatch.setattr(type(comparator), "_sample_spectrum_chunker", unexpected)
        comparator.compute_cross_spectra([("a", "b")], progress=False)
        np.testing.assert_array_equal(comparator.rotation_angles_, [0])
        assert comparator.spectra_ is None

    @pytest.mark.parametrize("backend", ["fft", "nufft"])
    def test_cache_fallback_keeps_batched_selected_geomean(self, make_comparator, backend):
        values = np.random.default_rng(117).normal(size=(3, 17, 19))
        comparator = make_comparator([values, values * 2], backend=backend, mode="2d")
        comparator._landmark_cache_warn_bytes = 1
        with pytest.warns(UserWarning, match="geometric-mean"):
            comparator.compute_cross_spectra(
                [("a", "b")], landmark_genes=["a", "c"], chunk_size=2, progress=False
            )
        np.testing.assert_allclose(comparator.rotation_angles_, 0)


class TestFitState:
    @pytest.mark.parametrize("backend", ["fft", "nufft"])
    @pytest.mark.parametrize("method", ["compute_spectra", "compute_cross_spectra"])
    def test_failed_extraction_preserves_all_previous_state(
        self, make_comparator, monkeypatch, backend, method
    ):
        import sonic.comparators.base as base_module
        import sonic.comparators.cross_spectrum as cross_module

        values = np.random.default_rng(118).normal(size=(3, 17, 19))
        comparator = make_comparator([values, values * 2], backend=backend, mode="2d")
        comparator.compute_spectra(landmark_genes=["a"], progress=False).normalize_background()
        comparator.compute_cross_spectra([("a", "b")], progress=False)
        before = comparator.__dict__.copy()
        normalized = comparator.spectra_.copy()

        def fail(*args, **kwargs):
            raise RuntimeError("synthetic extraction failure")

        if method == "compute_spectra":
            monkeypatch.setattr(base_module, "_stream_polar_features", fail)
            args, kwargs = (), {"landmark_genes": ["c"]}
        else:
            monkeypatch.setattr(cross_module, "_make_feature_reader", fail)
            args, kwargs = ([("a", "c")],), {}
        with pytest.raises(RuntimeError, match="synthetic extraction failure"):
            getattr(comparator, method)(*args, progress=False, **kwargs)
        assert before.keys() == comparator.__dict__.keys()
        assert all(comparator.__dict__[key] is value for key, value in before.items())
        np.testing.assert_array_equal(comparator.spectra_, normalized)

    @pytest.mark.parametrize("backend", ["fft", "nufft"])
    def test_failed_alignment_preserves_cross_only_result(
        self, make_comparator, monkeypatch, backend
    ):
        import sonic.comparators.base as base_module

        values = np.random.default_rng(119).normal(size=(3, 17, 19))
        comparator = make_comparator([values, values * 2], backend=backend, mode="2d")
        comparator.compute_cross_spectra([("a", "b")], landmark_genes=["a"], progress=False)
        before = comparator.__dict__.copy()

        def fail(*args, **kwargs):
            raise RuntimeError("synthetic alignment failure")

        monkeypatch.setattr(base_module, "_estimate_rotations_from_polar", fail)
        with pytest.raises(RuntimeError, match="synthetic alignment failure"):
            comparator.compute_cross_spectra([("a", "c")], landmark_genes=["c"], progress=False)
        assert all(comparator.__dict__[key] is value for key, value in before.items())

    @pytest.mark.parametrize("backend", ["fft", "nufft"])
    def test_invalid_pairs_fail_before_alignment(self, make_comparator, monkeypatch, backend):
        comparator = make_comparator(
            np.random.default_rng(120).normal(size=(3, 17, 19)), backend=backend, mode="2d"
        )
        before = comparator.__dict__.copy()

        def unexpected(*args, **kwargs):
            pytest.fail("Invalid pairs must fail before alignment")

        monkeypatch.setattr(type(comparator), "_ensure_alignment", unexpected)
        with pytest.raises(ValueError, match="must exist"):
            comparator.compute_cross_spectra(
                [("a", "missing")], landmark_genes=["a"], progress=False
            )
        assert before.keys() == comparator.__dict__.keys()
        assert all(comparator.__dict__[key] is value for key, value in before.items())

    @pytest.mark.parametrize("backend", ["fft", "nufft"])
    @pytest.mark.parametrize("method", ["compute_spectra", "compute_cross_spectra"])
    def test_chunk_override_is_call_local(self, make_comparator, backend, method):
        comparator = make_comparator(
            np.random.default_rng(121).normal(size=(3, 17, 19)),
            backend=backend,
            **{f"{backend}_chunk_size": 3},
        )
        args = () if method == "compute_spectra" else ([("a", "b"), ("b", "c")],)
        getattr(comparator, method)(*args, chunk_size=1, progress=False)
        assert getattr(comparator, f"_{backend}_chunk_size_spec") == 3
        if method == "compute_spectra":
            assert getattr(comparator, f"_{backend}_chunk_size") == 1
        else:
            assert comparator.cross_spectra_.chunk_schedules[0]["chunk_size"] == 1
        getattr(comparator, method)(*args, progress=False)
        if method == "compute_spectra":
            assert getattr(comparator, f"_{backend}_chunk_size") == 3
        else:
            assert comparator.cross_spectra_.chunk_schedules[0]["chunk_size"] == 3

    @pytest.mark.parametrize("backend", ["fft", "nufft"])
    @pytest.mark.parametrize("mode", ["radial", "2d"])
    @pytest.mark.parametrize("operation", ["compute_spectra", "normalize_covariates"])
    def test_mismatched_bins_raise_without_truncation(
        self, make_comparator, monkeypatch, backend, mode, operation
    ):
        values = np.random.default_rng(124).normal(size=(3, 17, 19))
        comparator = make_comparator([values, values * 2], backend=backend, mode=mode)
        comparator.compute_spectra(progress=False)
        comparator.compute_cross_spectra([("a", "b")], progress=False)
        before = comparator.__dict__.copy()
        if operation == "compute_spectra":

            def malformed(self, *args, **kwargs):
                features = list(self.spectra_)
                features[-1] = np.pad(features[-1], ((0, 0), (0, 1)))
                return features, self.dc_, self.presence_

            monkeypatch.setattr(type(comparator), "_compute_spectra", malformed)
            args, kwargs = (), {"progress": False}
        else:

            def malformed(self, keys):
                return [np.ones((1, self.spectra_.shape[-1] + 1)) for _ in self.samples]

            monkeypatch.setattr(type(comparator), "_covariate_features_from_keys", malformed)
            args, kwargs = (["covariate"],), {}
        with pytest.raises(ValueError, match="feature axis"):
            getattr(comparator, operation)(*args, **kwargs)
        assert before.keys() == comparator.__dict__.keys()
        assert all(comparator.__dict__[key] is value for key, value in before.items())
