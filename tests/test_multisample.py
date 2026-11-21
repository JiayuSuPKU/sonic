"""Tests for comparator statistical comparison helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest

from sonic.comparators.multisample import (
    compare_glm,
    compare_glm_masked,
    compare_glm_scalar,
    compare_two_groups,
    compare_two_groups_masked,
    compare_two_groups_scalar,
)
from sonic.comparators.normalization import normalize_shape


def _stub_spectra_groups(seed: int = 0, n_a: int = 4, n_b: int = 4, n_genes: int = 30, K: int = 12):
    """Build reusable spectra and labels for normalize_shape option tests."""

    rng = np.random.default_rng(seed)
    spec = rng.uniform(0.5, 2.0, size=(n_a + n_b, n_genes, K))
    groups = np.array([0] * n_a + [1] * n_b)
    return spec, groups


class TestComparisonOptions:
    """Public options, result schemas, and spectrum transformations."""

    @pytest.mark.parametrize(
        "options,shape_only,bad",
        [
            ({}, False, np.nan),
            ({}, True, -1.0),
            ({"covariance": "pooled"}, True, np.inf),
            ({"null": "plugin"}, False, -1.0),
            ({"null": "permutation"}, True, -np.inf),
            ({"statistic": "log_t_cauchy"}, False, np.nan),
        ],
    )
    def test_comparisons_check_observed_profiles(self, options, bad, shape_only):
        groups = np.repeat([0, 1], 6)
        spectra = np.tile([1.0, 2.0, 3.0], (12, 1, 1))
        spectra[1, 0, 1] = bad
        message = "nonnegative" if np.isfinite(bad) else "finite"
        with pytest.raises(ValueError, match=message):
            compare_two_groups(spectra, groups, normalize_shape=shape_only, **options)
        presence = np.ones((12, 1), dtype=bool)
        presence[1] = False
        masked = compare_two_groups_masked(
            spectra, groups, presence, normalize_shape=shape_only, **options
        )
        assert masked.P_value.item() == 1
        presence[1] = True
        with pytest.raises(ValueError, match=message):
            compare_two_groups_masked(
                spectra, groups, presence, normalize_shape=shape_only, **options
            )

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("shape_only", [False, True])
    @pytest.mark.parametrize(
        "glm,null,covariance",
        [
            (False, "plugin", None),
            (False, "plugin", "per_gene"),
            (False, "rotation", None),
            (False, "permutation", None),
            (True, "plugin", None),
            (True, "plugin", "per_gene"),
            (True, "rotation", None),
        ],
    )
    def test_log_l2_reports_fitted_bin_contrasts(self, glm, null, covariance, masked, shape_only):
        rng = np.random.default_rng(413)
        spectra = rng.lognormal(size=(10, 3, 4))
        spectra[2, 0, 1] = 0
        groups = np.tile(["B", "A"], 5)
        design = np.column_stack((np.ones(10), np.linspace(-1, 1, 10), rng.normal(size=10)))
        contrast = np.array([0.0, -2.0, 0.5])
        weights = np.array([2.0, 0.5, 1.0, 1.0])
        presence = np.ones(spectra.shape[:2], dtype=bool)
        if masked:
            presence[1::3, 1] = False
            presence[:, 2] = False
            spectra[~presence] = np.nan

        if glm:
            compare = compare_glm_masked if masked else compare_glm
            args = [spectra, design, contrast]
        else:
            compare = compare_two_groups_masked if masked else compare_two_groups
            args = [spectra, groups]
        if masked:
            args.append(presence)
        options = {"normalize_shape": shape_only, "freq_weights": weights}
        result = compare(*args, null=null, covariance=covariance, **options).sort_values("Feature")
        cauchy = compare(*args, statistic="log_t_cauchy", **options).sort_values("Feature")
        estimates = np.stack(result.Estimate_per_bin)
        np.testing.assert_allclose(estimates, np.stack(cauchy.Estimate_per_bin), atol=1e-12)
        for gene, observed_samples in enumerate(presence.T):
            if not observed_samples.any():
                assert np.isnan(estimates[gene]).all()
                continue
            values = spectra[observed_samples, gene]
            if shape_only:
                values = values / values.sum(axis=1, keepdims=True)
            response = np.log(np.maximum(values, 1e-12))
            if glm:
                expected = (
                    contrast @ np.linalg.lstsq(design[observed_samples], response, rcond=None)[0]
                )
            else:
                group_a = groups[observed_samples] == "A"
                expected = response[group_a].mean(0) - response[~group_a].mean(0)
            np.testing.assert_allclose(estimates[gene], expected, atol=1e-12)
            assert result.Statistic.iloc[gene] == pytest.approx(
                np.sqrt(np.average(expected**2, weights=weights))
            )

    @pytest.mark.parametrize("stat", ["log_l2", "log_t_cauchy"])
    def test_each_statistic_runs(self, stat):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(6, 8, 6))
        groups = np.array([0, 0, 0, 1, 1, 1])
        df = compare_two_groups(spectra, groups, statistic=stat, n_perm=50, random_state=0)
        assert df.shape[0] == 8
        assert {"Feature", "Statistic", "P_value", "P_adj"} <= set(df.columns)
        assert df["P_value"].between(0, 1).all()
        if stat != "log_l2":
            assert "P_value_per_bin" in df.columns
            per_bin = np.stack(df["P_value_per_bin"].to_numpy())
            assert per_bin.shape == (8, 6)
            assert ((per_bin >= 0) & (per_bin <= 1)).all()

    def test_binary_labels_must_be_observed(self):
        with pytest.raises(ValueError, match="nonmissing"):
            compare_two_groups(np.ones((4, 1, 3)), np.array([0, 0, np.nan, np.nan]))

    def test_unknown_statistic_raises(self):
        with pytest.raises(ValueError, match="Unknown statistic"):
            compare_two_groups(
                np.zeros((4, 3, 5)),
                np.array([0, 0, 1, 1]),
                statistic="bogus",
            )

    def test_log_l2_freq_weights(self):
        rng = np.random.default_rng(0)
        n_samples, n_genes, K = 6, 4, 8
        base = rng.uniform(0.5, 1.5, size=(n_samples, n_genes, K))
        base[3:, 0, :2] *= 3.0
        base[3:, 1, -2:] *= 3.0
        groups = np.array([0, 0, 0, 1, 1, 1])
        df_equal = compare_two_groups(base, groups, statistic="log_l2", n_perm=200, random_state=0)
        low_pass = np.concatenate([np.ones(2), np.full(K - 2, 0.01)])
        df_low = compare_two_groups(
            base,
            groups,
            statistic="log_l2",
            n_perm=200,
            random_state=0,
            freq_weights=low_pass,
        )
        assert df_low["Feature"].iloc[0] == "0"
        assert df_equal["Feature"].iloc[0] != df_low["Feature"].iloc[-1]

    def test_log_l2_freq_weights_validation(self):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(4, 3, 6))
        groups = np.array([0, 0, 1, 1])
        with pytest.raises(ValueError, match="length n_bins="):
            compare_two_groups(
                spectra,
                groups,
                statistic="log_l2",
                n_perm=10,
                freq_weights=np.ones(5),
            )
        with pytest.raises(ValueError, match="strictly positive"):
            compare_two_groups(
                spectra,
                groups,
                statistic="log_l2",
                n_perm=10,
                freq_weights=np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0]),
            )

    def test_implanted_difference_is_recovered(self):
        rng = np.random.default_rng(7)
        n_per = 6
        n_genes, K = 50, 10
        a = rng.normal(loc=1.0, scale=0.1, size=(n_per, n_genes, K))
        b = rng.normal(loc=1.0, scale=0.1, size=(n_per, n_genes, K))
        b[:, :5, :3] += 0.8
        spectra = np.concatenate([a, b], axis=0)
        groups = np.array([0] * n_per + [1] * n_per)
        df = compare_two_groups(spectra, groups, statistic="log_l2", n_perm=400, random_state=0)
        top10 = set(df.head(10)["Feature"].astype(str))
        implanted = {"0", "1", "2", "3", "4"}
        recovered = len(top10 & implanted)
        assert recovered >= 4, f"only recovered {recovered}/5 top-10: {top10}"

    def test_masked_matches_unmasked_when_full(self):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(6, 4, 5))
        groups = np.array([0, 0, 0, 1, 1, 1])
        presence = np.ones((6, 4), dtype=bool)
        df_full = compare_two_groups(spectra, groups, statistic="log_l2", n_perm=50, random_state=0)
        df_mask = compare_two_groups_masked(
            spectra, groups, presence, statistic="log_l2", n_perm=50, random_state=0
        )
        np.testing.assert_allclose(
            df_full.set_index("Feature").loc[df_mask["Feature"], "Statistic"].to_numpy(),
            df_mask["Statistic"].to_numpy(),
            rtol=1e-10,
        )
        assert (df_mask["n_obs_A"] == 3).all()
        assert (df_mask["n_obs_B"] == 3).all()

    def test_gene_skipped_when_below_min_samples_per_group(self):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(6, 3, 5))
        groups = np.array([0, 0, 0, 1, 1, 1])
        presence = np.array(
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [False, True, True],
                [False, True, True],
                [True, True, True],
            ]
        )
        df = compare_two_groups_masked(
            spectra,
            groups,
            presence,
            statistic="log_l2",
            n_perm=20,
            random_state=0,
            min_samples_per_group=2,
        )
        skipped_row = df[df.Feature == "0"].iloc[0]
        assert np.isnan(skipped_row["P_value"])
        assert np.isnan(skipped_row["P_adj"])
        assert skipped_row["n_obs_A"] == 3
        assert skipped_row["n_obs_B"] == 1

    @pytest.mark.parametrize(
        "statistic,null",
        [
            ("log_l2", "plugin"),
            ("log_l2", "permutation"),
            ("log_t_cauchy", "analytic"),
        ],
    )
    def test_two_groups_option_matches_manual_normalization(self, statistic, null):
        spec, groups = _stub_spectra_groups()
        kw = {"statistic": statistic, "null": null, "n_perm": 200, "random_state": 0}
        df_kw = compare_two_groups(spec, groups, **kw, normalize_shape=True)
        df_manual = compare_two_groups(normalize_shape(spec), groups, **kw)
        df_kw = df_kw.sort_values("Feature").reset_index(drop=True)
        df_manual = df_manual.sort_values("Feature").reset_index(drop=True)
        np.testing.assert_allclose(
            df_kw["P_value"].to_numpy(),
            df_manual["P_value"].to_numpy(),
            rtol=1e-12,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            df_kw["Statistic"].to_numpy(),
            df_manual["Statistic"].to_numpy(),
            rtol=1e-12,
            atol=1e-15,
        )

    @pytest.mark.parametrize(
        "statistic,null",
        [
            ("log_l2", "plugin"),
            ("log_l2", "permutation"),
            ("log_t_cauchy", "analytic"),
        ],
    )
    def test_masked_two_groups_option_matches_manual_normalization(self, statistic, null):
        spec, groups = _stub_spectra_groups()
        presence = np.ones((spec.shape[0], spec.shape[1]), dtype=bool)
        kw = {"statistic": statistic, "null": null, "n_perm": 200, "random_state": 0}
        df_kw = compare_two_groups_masked(spec, groups, presence, **kw, normalize_shape=True)
        df_manual = compare_two_groups_masked(normalize_shape(spec), groups, presence, **kw)
        df_kw = df_kw.sort_values("Feature").reset_index(drop=True)
        df_manual = df_manual.sort_values("Feature").reset_index(drop=True)
        np.testing.assert_allclose(
            df_kw["P_value"].to_numpy(),
            df_manual["P_value"].to_numpy(),
            rtol=1e-12,
            atol=1e-15,
        )

    def test_glm_option_matches_manual_normalization(self):
        spec, _ = _stub_spectra_groups()
        design = pd.DataFrame({"x": np.arange(spec.shape[0], dtype=float)})
        df_kw = compare_glm(spec, design, "x", normalize_shape=True)
        df_manual = compare_glm(normalize_shape(spec), design, "x")
        df_kw = df_kw.sort_values("Feature").reset_index(drop=True)
        df_manual = df_manual.sort_values("Feature").reset_index(drop=True)
        np.testing.assert_allclose(
            df_kw["P_value"].to_numpy(),
            df_manual["P_value"].to_numpy(),
            rtol=1e-12,
            atol=1e-15,
        )

    def test_masked_glm_option_matches_manual_normalization(self):
        spec, _ = _stub_spectra_groups()
        design = pd.DataFrame({"x": np.arange(spec.shape[0], dtype=float)})
        presence = np.ones((spec.shape[0], spec.shape[1]), dtype=bool)
        df_kw = compare_glm_masked(spec, design, "x", presence, normalize_shape=True)
        df_manual = compare_glm_masked(normalize_shape(spec), design, "x", presence)
        df_kw = df_kw.sort_values("Feature").reset_index(drop=True)
        df_manual = df_manual.sort_values("Feature").reset_index(drop=True)
        np.testing.assert_allclose(
            df_kw["P_value"].to_numpy(),
            df_manual["P_value"].to_numpy(),
            rtol=1e-12,
            atol=1e-15,
        )

    @pytest.mark.parametrize(
        "mode", ["binary", "masked", "glm", "masked_glm", "permutation", "masked_permutation"]
    )
    @pytest.mark.parametrize("shape_only", [False, True])
    def test_float32_comparisons_match_the_same_values_promoted_to_float64(self, mode, shape_only):
        rng = np.random.default_rng(921)
        spectra = (1e-4 * (1 + 2e-7 * rng.normal(size=(8, 40, 3)))).astype(np.float32)
        original = spectra.copy()
        groups = np.repeat([0, 1], 4)
        presence = np.ones(spectra.shape[:2], dtype=bool)
        presence[0, :10] = False
        presence[4, 10:20] = False
        design = pd.DataFrame({"group": groups})

        def run(values):
            kwargs = {"normalize_shape": shape_only}
            if mode == "glm":
                result = compare_glm(values, design, "group", **kwargs)
            elif mode == "masked_glm":
                result = compare_glm_masked(values, design, "group", presence, **kwargs)
            else:
                kwargs["null"] = "permutation" if "permutation" in mode else "plugin"
                if mode.startswith("masked"):
                    result = compare_two_groups_masked(values, groups, presence, **kwargs)
                else:
                    result = compare_two_groups(values, groups, **kwargs)
            return result.set_index("Feature").sort_index()[["Statistic", "P_value", "P_adj"]]

        np.testing.assert_array_equal(run(spectra), run(spectra.astype(np.float64)))
        np.testing.assert_array_equal(spectra, original)

    @pytest.mark.parametrize(
        "profile,factor",
        [(np.array([1.0, 2.0, 4.0]) * 1e-16, 10), ([0.5, 0.5 + 2.0**-52], 3)],
        ids=["low-power", "exact-threefold"],
    )
    @pytest.mark.parametrize("null", ["rotation", "plugin", "permutation"])
    def test_shape_only_comparison_ignores_amplitude_changes(self, profile, factor, null):
        spectra = np.tile(profile, (8, 1, 1))
        spectra[4:] *= factor
        result = compare_two_groups(spectra, np.repeat([0, 1], 4), normalize_shape=True, null=null)
        np.testing.assert_array_equal(result.Estimate_per_bin.iloc[0], np.zeros(len(profile)))
        np.testing.assert_array_equal(result["Statistic"], 0.0)
        np.testing.assert_array_equal(result["P_value"], 1.0)

    def test_shape_permutation_uses_same_shares_for_mixed_labels(self):
        # Two distinct shapes can fit a 3:5 split even when the supplied labels mix them.
        low = [0.5, 0.5 + 2.0**-52]
        powers = np.array([low] * 3 + [[0.5 - 2.0**-20, 0.5 + 2.0**-20]] * 5)[:, None, :]
        powers[1] *= 3
        expected = powers.copy()
        expected[:3, 0] = np.array(low) / sum(low)
        labels = np.array([0, 1, 0, 1, 1, 0, 1, 1])
        for groups in (labels, 1 - labels):
            result = compare_two_groups(
                powers, groups, null="permutation", normalize_shape=True
            ).iloc[0]
            reference = compare_two_groups(expected, groups, null="permutation").iloc[0]
            np.testing.assert_array_equal(result.Estimate_per_bin, reference.Estimate_per_bin)
            assert result.P_value == reference.P_value

    @pytest.mark.parametrize(
        "path",
        ["two_group", "two_group_masked", "two_group_scalar", "glm", "glm_masked", "glm_scalar"],
    )
    @pytest.mark.parametrize("n_a,n_b", [(6, 6), (10, 30)])
    def test_constant_responses_do_not_become_significant(self, path, n_a, n_b):
        groups = np.repeat([0, 1], [n_a, n_b])
        values = np.broadcast_to([0.1, 2.0, 10.0], (len(groups), 3))
        spectra = np.broadcast_to(values[..., None] * [0.5, 1.0, 2.0], (len(groups), 3, 3))
        presence = np.ones(values.shape, dtype=bool)
        presence[0, 1] = presence[-1, 2] = False
        design = pd.DataFrame({"group": groups})
        if path == "glm_scalar":
            result = compare_glm_scalar(values, design, "group")
        elif path == "glm":
            result = compare_glm(spectra, design, "group")
        elif path == "glm_masked":
            result = compare_glm_masked(spectra, design, "group", presence)
        elif path == "two_group_scalar":
            result = compare_two_groups_scalar(values, groups)
        elif path == "two_group_masked":
            result = compare_two_groups_masked(spectra, groups, presence)
        else:
            result = compare_two_groups(spectra, groups)
        np.testing.assert_array_equal(result["Statistic"], 0.0)
        np.testing.assert_array_equal(result[["P_value", "P_adj"]], 1.0)

    @pytest.mark.parametrize("glm", [False, True])
    def test_scalar_result_builder_preserves_column_order_and_feature_names(self, glm):
        rng = np.random.default_rng(50)
        values = rng.normal(size=(10, 4))
        groups = np.repeat([0, 1], 5)
        if glm:
            result = compare_glm_scalar(values, pd.DataFrame({"group": groups}), "group")
            effect_column = "Estimate"
        else:
            result = compare_two_groups_scalar(values, groups)
            effect_column = "Mean_diff"
        assert list(result.columns) == ["Feature", "Statistic", effect_column, "P_value", "P_adj"]
        assert set(result.Feature) == {"0", "1", "2", "3"}
        assert result.Statistic.is_monotonic_decreasing


class TestGLMComparison:
    """Design encoding, scalar contrasts, and stable OLS fits."""

    def test_binary_design_matches_groups_path_byte_close(self):
        """A binary GLM design should match the two-group plug-in calibration."""
        rng = np.random.default_rng(0)
        spectra = np.exp(rng.standard_normal((8, 200, 25)))
        groups = np.array(["WT"] * 4 + ["TG"] * 4)

        df_g = (
            compare_two_groups(spectra, groups, statistic="log_l2", null="plugin")
            .sort_values("Feature")
            .reset_index(drop=True)
        )

        design = pd.DataFrame({"genotype": groups})
        df_d = (
            compare_glm(spectra, design, contrast="genotype", null="plugin")
            .sort_values("Feature")
            .reset_index(drop=True)
        )

        np.testing.assert_allclose(
            df_g["P_value"].to_numpy(), df_d["P_value"].to_numpy(), atol=1e-10
        )
        np.testing.assert_allclose(
            df_g["Statistic"].to_numpy(), df_d["Statistic"].to_numpy(), atol=1e-10
        )

    def test_continuous_contrast_recovers_planted_signal(self):
        rng = np.random.default_rng(7)
        n, n_genes, K = 8, 100, 25
        x = np.linspace(0.0, 1.0, n)
        beta = np.zeros((n_genes, K))
        beta[:5, :3] = 5.0
        log_y = beta[None, :, :] * x[:, None, None] + 0.3 * rng.standard_normal((n, n_genes, K))
        spectra = np.exp(log_y)
        design = pd.DataFrame({"time": x})
        df = compare_glm(spectra, design, contrast="time")
        top10 = set(df.head(10)["Feature"].tolist())
        assert {"0", "1", "2", "3", "4"} <= top10, f"missing planted: {top10}"

    def test_dict_contrast_normalization(self):
        """A dict contrast should match the equivalent column-name contrast."""
        rng = np.random.default_rng(2)
        spectra = np.exp(rng.standard_normal((8, 30, 20)))
        design = pd.DataFrame({"a": [0, 1, 0, 1, 0, 1, 0, 1], "b": np.arange(8)})
        df_dict = (
            compare_glm(spectra, design, contrast={"a": 1.0})
            .sort_values("Feature")
            .reset_index(drop=True)
        )
        df_str = (
            compare_glm(spectra, design, contrast="a").sort_values("Feature").reset_index(drop=True)
        )
        np.testing.assert_allclose(df_dict["P_value"].to_numpy(), df_str["P_value"].to_numpy())

    def test_ndarray_design_with_intercept_only(self):
        rng = np.random.default_rng(3)
        n, n_genes, K = 6, 20, 18
        spectra = np.exp(rng.standard_normal((n, n_genes, K)))
        X = np.column_stack([np.ones(n), [0, 0, 0, 1, 1, 1]])
        df = compare_glm(spectra, X, contrast=np.array([0.0, 1.0]))
        assert df.shape[0] == n_genes
        assert df["P_value"].between(0, 1).all()

    def test_rank_deficient_estimable_contrast_matches_collapsed_design(self):
        rng = np.random.default_rng(4)
        n, n_genes, K = 8, 20, 12
        x = np.linspace(-1.0, 1.0, n)
        spectra = np.exp(rng.standard_normal((n, n_genes, K)))
        X_full = np.column_stack([np.ones(n), x])
        X_duplicate = np.column_stack([np.ones(n), x, x])

        df_full = (
            compare_glm(spectra, X_full, contrast=np.array([0.0, 1.0]))
            .sort_values("Feature")
            .reset_index(drop=True)
        )
        df_duplicate = (
            compare_glm(spectra, X_duplicate, contrast=np.array([0.0, 1.0, 1.0]))
            .sort_values("Feature")
            .reset_index(drop=True)
        )

        np.testing.assert_allclose(df_duplicate["Statistic"], df_full["Statistic"], rtol=1e-12)
        np.testing.assert_allclose(df_duplicate["P_value"], df_full["P_value"], rtol=1e-12)

    def test_rank_deficient_nonestimable_contrast_raises(self):
        rng = np.random.default_rng(5)
        n, n_genes, K = 8, 10, 6
        x = np.linspace(0.0, 1.0, n)
        spectra = np.exp(rng.standard_normal((n, n_genes, K)))
        X = np.column_stack([np.ones(n), x, x])

        with pytest.raises(ValueError, match="not estimable"):
            compare_glm(spectra, X, contrast=np.array([0.0, 1.0, 0.0]))

    @pytest.mark.parametrize("null", ["rotation", "plugin"])
    @pytest.mark.parametrize("origin", [0, 1_730_000_000], ids=["centered", "unix"])
    @pytest.mark.parametrize("delay", [1 / 8, 1 / 65_536])
    def test_duration_confounded_group_is_not_estimable(self, null, origin, delay):
        group = np.tile([0, 0, 1, 1], 6)
        start = origin + 86_400 * (np.arange(24) - 11.5)
        end = start + 0.5 + group * delay
        np.testing.assert_array_equal((end - start - 0.5) / delay, group)
        design = pd.DataFrame({"group": group, "start": start, "end": end})
        powers = np.random.default_rng(325).lognormal(size=(24, 2, 7))
        with pytest.raises(ValueError, match="contrast is not estimable"):
            compare_glm(powers, design, "group", null=null, normalize_shape=True)

    @pytest.mark.parametrize("null", ["rotation", "plugin"])
    def test_indicator_contrast_confounded_by_duration(self, null):
        group = np.tile([0, 0, 1, 1], 6)
        start = 86_400 * (np.arange(24) - 11.5)
        powers = np.random.default_rng(325).lognormal(size=(24, 1, 3))
        for origin in (0, 1_730_000_000):
            time = start + origin
            design = np.column_stack((1 - group, group, time, time + 0.5 + group / 65_536))
            with pytest.raises(ValueError, match="contrast is not estimable"):
                compare_glm(
                    powers, design, np.array([-1, 1, 0, 0]), null=null, normalize_shape=True
                )

    @pytest.mark.parametrize("null", ["rotation", "plugin"])
    def test_timestamp_centering_changes_rank(self, null):
        group = np.tile([0, 0, 1, 1], 6)
        time = 86_400 * (np.arange(24) - 11.5)
        duration = 2 * group + np.tile([0, 1, 0, 1], 6)
        end = time + 0.5 + duration / 65_536
        powers = np.random.default_rng(325).lognormal(size=(24, 1, 3))

        def fit(start, stop):
            design = pd.DataFrame({"group": group, "start": start, "end": stop})
            return compare_glm(powers, design, "group", null=null, normalize_shape=True)

        with pytest.raises(ValueError, match="rank.*center"):
            fit(time + 1_730_000_000, end + 1_730_000_000)
        np.testing.assert_allclose(fit(time, end).P_value, fit(time, duration).P_value, atol=1e-4)

    @pytest.mark.parametrize("indicators", [False, True], ids=["explicit", "implicit"])
    def test_estimable_prediction_with_shifted_duplicate_predictor(self, indicators):
        time = np.arange(16.0, 28.0)
        if indicators:
            group = np.tile([0, 0, 1], 4)
            design = np.column_stack((time, 1 - group, group))
        else:
            design = np.column_stack((time, np.full_like(time, 2.0)))
        powers = np.random.default_rng(37).lognormal(size=(len(time), 1, 3))
        duplicate = np.column_stack((design, time + 8))
        # A fitted mean at an observed sample is estimable in either encoding.
        reference = compare_glm(powers, design, design[4], null="plugin")
        actual = compare_glm(powers, duplicate, duplicate[4], null="plugin")
        np.testing.assert_allclose(actual.P_value, reference.P_value, rtol=1e-12)

    def test_mixed_contrast_keeps_small_timestamp_terms(self):
        group = np.tile([0, 1], 6)
        time = (1_730_000_000 + 86_400 * np.arange(12)) * 10**9
        design = pd.DataFrame({"group": group, "time": time})
        duplicate = design.assign(time_copy=time)
        powers = np.random.default_rng(37).lognormal(size=(12, 1, 3))
        contrast = {"group": 1, "time": 1}
        with pytest.raises(ValueError, match="contrast is not estimable"):
            compare_glm(powers, duplicate, contrast, null="plugin")
        expected = compare_glm(powers, design, contrast, null="plugin")
        actual = compare_glm(powers, duplicate, {**contrast, "time_copy": 1}, null="plugin")
        np.testing.assert_allclose(actual.P_value, expected.P_value, rtol=1e-9)

    def test_glm_statistic_stays_keyword_only(self):
        import inspect

        for fn in (compare_glm, compare_glm_masked):
            sig = inspect.signature(fn)
            assert sig.parameters["statistic"].default == "log_l2"
            assert sig.parameters["statistic"].kind == inspect.Parameter.KEYWORD_ONLY

    def test_masked_glm_matches_unmasked_when_complete(self):
        """A fully present mask should recover the unmasked GLM result."""
        rng = np.random.default_rng(11)
        spectra = np.exp(rng.standard_normal((8, 80, 18)))
        design = pd.DataFrame({"time": np.linspace(0.0, 1.0, 8)})
        presence = np.ones((8, 80), dtype=bool)

        df_full = (
            compare_glm(spectra, design, contrast="time")
            .sort_values("Feature")
            .reset_index(drop=True)
        )
        df_mask = (
            compare_glm_masked(spectra, design, "time", presence)
            .sort_values("Feature")
            .reset_index(drop=True)
        )

        np.testing.assert_allclose(df_full["Statistic"], df_mask["Statistic"], atol=1e-12)
        np.testing.assert_allclose(df_full["P_value"], df_mask["P_value"], atol=1e-12)
        assert (df_mask["n_obs"] == 8).all()
        assert (df_mask["df_resid"] == 6).all()

    def test_masked_glm_skips_unestimable_gene_contrast(self):
        """Masking can make a contrast unestimable for a single gene."""
        rng = np.random.default_rng(12)
        spectra = np.exp(rng.standard_normal((6, 12, 10)))
        design = pd.DataFrame({"group": [0, 0, 0, 1, 1, 1]})
        presence = np.ones((6, 12), dtype=bool)
        presence[3:, 0] = False  # gene 0 has no observed samples with group == 1

        df = compare_glm_masked(spectra, design, "group", presence)
        skipped = df.set_index("Feature").loc["0"]

        assert skipped["n_obs"] == 3
        assert skipped["df_resid"] == 2
        assert np.isnan(skipped["Statistic"])
        assert np.isnan(skipped["P_value"])
        assert df.loc[df["Feature"] != "0", "P_value"].notna().all()

    @pytest.mark.parametrize(
        "path,covariance_mode",
        [
            ("spectral", None),
            ("masked", None),
            ("scalar", None),
            ("spectral", "per_gene"),
            ("masked", "per_gene"),
        ],
    )
    @pytest.mark.parametrize("offset,scale", [(1e4, 1.0), (0.0, 1e8)])
    def test_glm_nuisance_offset_and_units_preserve_contrast(
        self, path, covariance_mode, offset, scale
    ):
        rng = np.random.default_rng(673)
        n, n_genes, n_bins = 24, 8, 6
        group = np.repeat([0.0, 1.0], n // 2)
        covariate = rng.normal(size=n)
        design = np.column_stack([np.ones(n), group, covariate])
        shifted = design.copy()
        shifted[:, 2] = (covariate + offset) * scale
        contrast = np.array([0.0, 1.0, 0.0])
        values = (
            1.0 + 0.5 * group[:, None] + 0.2 * covariate[:, None] + rng.normal(size=(n, n_genes))
        )
        spectra = np.exp(values[:, :, None] + rng.normal(size=(n, n_genes, n_bins)))
        presence = rng.random((n, n_genes)) > 0.15

        def compare(x):
            if path == "scalar":
                frame = compare_glm_scalar(values, x, contrast)
            elif path == "masked":
                frame = compare_glm_masked(
                    spectra, x, contrast, presence, covariance=covariance_mode
                )
            else:
                frame = compare_glm(spectra, x, contrast, covariance=covariance_mode)
            return frame.set_index("Feature").sort_index()

        pd.testing.assert_frame_equal(compare(shifted), compare(design), rtol=2e-6, atol=1e-10)
        np.testing.assert_array_equal(design[:, 2], covariate)

    @pytest.mark.parametrize(
        "path,covariance_mode",
        [
            ("spectral", None),
            ("masked", None),
            ("scalar", None),
            ("spectral", "per_gene"),
            ("masked", "per_gene"),
        ],
    )
    def test_glm_contrast_scale_preserves_pvalues_and_estimability(self, path, covariance_mode):
        rng = np.random.default_rng(33)
        groups = np.repeat([0, 1], 6)
        design = np.column_stack([np.ones(12), groups])
        values = rng.normal(size=(12, 5)) + 0.8 * groups[:, None]
        spectra = np.exp(values[:, :, None] + rng.normal(size=(12, 5, 3)))
        presence = np.ones((12, 5), dtype=bool)
        presence[0, 0] = False

        def compare(matrix, contrast):
            if path == "scalar":
                result = compare_glm_scalar(values, matrix, contrast)
            elif path == "masked":
                result = compare_glm_masked(
                    spectra, matrix, contrast, presence, covariance=covariance_mode
                )
            else:
                result = compare_glm(spectra, matrix, contrast, covariance=covariance_mode)
            return result.set_index("Feature").sort_index()

        expected = compare(design, np.array([0.0, 1.0]))
        aliased = np.column_stack([design, groups])
        for scale in (1.0, 1e-16, 1e-6, -1e6):
            actual = compare(design, np.array([0.0, scale]))
            np.testing.assert_allclose(
                actual[["P_value", "P_adj"]], expected[["P_value", "P_adj"]], rtol=1e-12
            )
            with pytest.raises(ValueError, match="estimable"):
                compare(aliased, np.array([0.0, scale, 0.0]))
        with pytest.raises(ValueError, match="nonzero"):
            compare(design, np.zeros(2))

    @pytest.mark.parametrize("path", ["spectral", "masked", "scalar"])
    def test_glm_rejects_colliding_encoded_design_names(self, path):
        rng = np.random.default_rng(13)
        x = rng.normal(size=12)
        spectra = rng.lognormal(size=(12, 2, 3))
        designs = [
            pd.DataFrame({"Intercept": x}),
            pd.DataFrame({"Intercept": pd.Categorical(np.repeat(["A", "B"], 6))}),
            pd.DataFrame(np.column_stack([x, x**2]), columns=["x", "x"]),
            pd.DataFrame({"x": x, "group": pd.Categorical(np.repeat(["A", "B"], 6))}).set_axis(
                ["x", "x"], axis=1
            ),
            pd.DataFrame({"group": pd.Categorical(np.repeat(["A", "B"], 6)), "group[T.B]": x}),
        ]
        for design in designs:
            with pytest.raises(ValueError, match="ambiguous after encoding"):
                if path == "scalar":
                    compare_glm_scalar(spectra.mean(axis=-1), design, design.columns[0])
                elif path == "masked":
                    compare_glm_masked(spectra, design, design.columns[0], np.ones((12, 2), bool))
                else:
                    compare_glm(spectra, design, design.columns[0])

    def test_glm_response_centering_preserves_intercepts_and_small_effects(
        self,
    ):
        groups = np.repeat([0.0, 1.0], 6)
        explicit = np.column_stack([np.ones(12), groups])
        implicit = np.column_stack([1 - groups, groups])
        values = np.ones((12, 1))
        # A group-indicator design spans an intercept without a constant column.
        result = compare_glm_scalar(values, implicit, np.array([-1.0, 1.0]))
        assert result.P_value.iloc[0] == 1.0
        # Tiny genuine intercept components must not be rounded out of a contrast.
        result = compare_glm_scalar(values, explicit, np.array([1e-16, 1.0]))
        assert result.Estimate.iloc[0] == 1e-16
        assert result.P_value.iloc[0] == 0.0
        # Without an intercept, a constant response still has a nonzero group coefficient.
        result = compare_glm_scalar(values, groups[:, None], np.ones(1))
        np.testing.assert_allclose(result.Estimate.iloc[0], 1.0)
        assert result.P_value.iloc[0] < 0.05
        # Changing intercept units still controls for the constant response.
        rescaled = np.column_stack([np.full(12, 1e-18), groups])
        actual = compare_glm_scalar(values, rescaled, np.array([0.0, 1.0]))
        assert actual.Estimate.item() == 0 and actual.P_value.item() == 1
        # A nearly constant covariate still does not span an exact intercept.
        near_constant = (1 + 1e-10 * groups)[:, None]
        actual = compare_glm_scalar(values, near_constant, np.ones(1))
        assert np.isfinite(actual.Statistic.iloc[0])
        values[groups == 1] += 1e-9
        result = compare_glm_scalar(values, explicit, np.array([0.0, 1e-16]))
        np.testing.assert_allclose(result.Estimate.iloc[0], (values[-1, 0] - 1) * 1e-16, rtol=1e-12)
        assert result.P_value.iloc[0] < 1e-10

    @pytest.mark.parametrize("path", ["glm", "masked_glm", "scalar"])
    @pytest.mark.parametrize("contrast_kind", ["str", "dict"])
    def test_glm_metadata_names_preserve_contrasts(self, path, contrast_kind):
        rng = np.random.default_rng(6)
        design = pd.DataFrame(
            {
                "age": rng.normal(size=24),
                "treatment": pd.Categorical(np.repeat(["A", "B"], 12)),
                "nuisance": rng.normal(size=24),
            }
        )
        spectra = rng.lognormal(size=(24, 4, 3))
        presence = np.ones((24, 4), dtype=bool)
        presence[0, 0] = False

        def compare(frame, contrast):
            if path == "glm":
                result = compare_glm(spectra, frame, contrast)
            elif path == "masked_glm":
                result = compare_glm_masked(spectra, frame, contrast, presence)
            else:
                result = compare_glm_scalar(spectra.mean(axis=-1), frame, contrast)
            return result.sort_values("Feature").reset_index(drop=True)

        for new_name in ["age-years", "treatment group", "Q", "class", "quote'\"[x]"]:
            # Numeric and categorical names must remain usable in both contrast forms.
            for column, term in [("age", "age"), ("treatment", "treatment[T.B]")]:
                reference_contrast = column if contrast_kind == "str" else {term: 1.0}
                renamed_contrast = (
                    new_name if contrast_kind == "str" else {new_name + term[len(column) :]: 1.0}
                )
                reference = compare(design, reference_contrast)
                actual = compare(design.rename(columns={column: new_name}), renamed_contrast)
                pd.testing.assert_frame_equal(actual, reference)


class TestWelchComparison:
    """Welch replication, constant-bin handling, and scale invariance."""

    def test_log_t_cauchy_is_invariant_to_spectral_units_above_floor(
        self,
    ):
        rng = np.random.default_rng(25)
        spectra = rng.uniform(1, 5, size=(12, 4, 3))
        groups = np.repeat([0, 1], 6)
        presence = np.ones((12, 4), dtype=bool)
        presence[0, 0] = False
        for mask in (None, presence):
            results = []
            for scale in (1.0, 1e-8):
                kwargs = {"statistic": "log_t_cauchy"}
                result = (
                    compare_two_groups(spectra * scale, groups, **kwargs)
                    if mask is None
                    else compare_two_groups_masked(spectra * scale, groups, mask, **kwargs)
                )
                results.append(result.sort_values("Feature")["P_value"].to_numpy())
            np.testing.assert_allclose(*results, rtol=1e-10)

    @pytest.mark.parametrize("n_b", [1, 5])
    @pytest.mark.parametrize("spectral", [False, True])
    def test_welch_requires_replication_in_both_groups(self, n_b, spectral):
        values = np.r_[100.0, np.arange(1.0, n_b + 1)][:, None]
        groups = np.r_[0, np.ones(n_b)]
        with pytest.raises(ValueError, match="at least two samples in each group"):
            if spectral:
                compare_two_groups(values[..., None], groups, statistic="log_t_cauchy")
            else:
                compare_two_groups_scalar(values, groups)

    @pytest.mark.parametrize("masked", [False, True])
    def test_log_t_cauchy_excludes_only_constant_bins(self, masked):
        values = np.r_[np.arange(1.0, 7.0), np.arange(9.0, 15.0)]
        groups = np.repeat([0, 1], 6)
        spectra = np.exp(np.stack([values, np.ones(12), 1 + groups], axis=1)[..., None])
        presence = np.ones((12, 3), dtype=bool)
        presence[0, 0] = False

        def compare(array):
            kwargs = {"statistic": "log_t_cauchy"}
            result = (
                compare_two_groups_masked(array, groups, presence, **kwargs)
                if masked
                else compare_two_groups(array, groups, **kwargs)
            )
            return result.set_index("Feature").sort_index()

        expected = compare(spectra)
        padded = np.concatenate([spectra, np.zeros_like(spectra), np.ones_like(spectra)], axis=-1)
        actual = compare(padded)
        np.testing.assert_allclose(actual.P_value, expected.P_value, rtol=1e-12)
        assert actual.loc["0", "P_value"] < 0.001
        assert actual.loc["1", "P_value"] == 1.0
        assert (
            actual.loc["2", "P_value"] < 1e-100
        )  # Different constants in the two arms remain informative.


class TestScalarTwoGroupComparison:
    """Scalar Welch tests and optional expression transforms."""

    def test_welch_constant_features_have_defined_tails(self):
        values = np.repeat([[2.0, 3.0, 0.0], [2.0, 9.0, 0.0]], 4, axis=0)
        result = compare_two_groups_scalar(values, np.repeat([0, 1], 4)).set_index("Feature")
        np.testing.assert_array_equal(result.loc[["0", "2"], "Statistic"], 0.0)
        np.testing.assert_array_equal(result.loc[["0", "2"], "P_value"], 1.0)
        assert np.isinf(result.loc["1", "Statistic"])
        assert result.loc["1", "P_value"] == np.finfo(float).tiny

    def test_welch_analytic_is_uniform_under_h0(self):
        rng = np.random.default_rng(0)
        n_samples, n_genes = 10, 1000
        values = rng.standard_normal((n_samples, n_genes))
        groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        df = compare_two_groups_scalar(values, groups)
        _, ks_p = kstest(df.P_value.to_numpy(), "uniform")
        assert ks_p > 0.01, f"analytic-Welch p-values not uniform under H0, KS p={ks_p:.4f}"

    @pytest.mark.parametrize("scale", [1.0, 1e-8, 1e-100, 1e100])
    def test_welch_analytic_matches_scipy(self, scale):
        from scipy.stats import ttest_ind

        rng = np.random.default_rng(2)
        n_per, n_genes = 5, 50
        a = rng.normal(0.0, 1.0, size=(n_per, n_genes))
        b = rng.normal(0.3, 1.5, size=(n_per, n_genes))
        values = np.concatenate([a, b], axis=0) * scale
        groups = np.array([0] * n_per + [1] * n_per)
        df = compare_two_groups_scalar(values, groups)
        df = df.set_index("Feature").loc[[str(i) for i in range(n_genes)]]
        scipy_p = ttest_ind(a, b, equal_var=False, axis=0).pvalue
        np.testing.assert_allclose(df["P_value"].to_numpy(), scipy_p, rtol=1e-9, atol=1e-300)

    def test_implanted_mean_shift_recovered(self):
        rng = np.random.default_rng(7)
        n_per, n_genes = 6, 40
        a = rng.normal(loc=0.0, scale=1.0, size=(n_per, n_genes))
        b = rng.normal(loc=0.0, scale=1.0, size=(n_per, n_genes))
        b[:, :5] += 2.0
        values = np.concatenate([a, b], axis=0)
        groups = np.array([0] * n_per + [1] * n_per)
        df = compare_two_groups_scalar(values, groups)
        top5 = set(df.head(5).Feature.astype(str).tolist())
        assert top5 == {"0", "1", "2", "3", "4"}

    def test_log_expression_matches_manual_log_values(self):
        rng = np.random.default_rng(8)
        values = rng.lognormal(mean=1.0, sigma=0.2, size=(8, 6))
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        df_logged = compare_two_groups_scalar(values, groups, log_expression=True)
        df_manual = compare_two_groups_scalar(np.log(values + 1e-12), groups)

        df_logged = df_logged.sort_values("Feature").reset_index(drop=True)
        df_manual = df_manual.sort_values("Feature").reset_index(drop=True)
        np.testing.assert_allclose(df_logged["Mean_diff"], df_manual["Mean_diff"], rtol=1e-12)
        np.testing.assert_allclose(df_logged["P_value"], df_manual["P_value"], rtol=1e-12)

    def test_log_expression_rejects_nonpositive_values(self):
        values = np.array([[1.0, 2.0], [0.5, -1.0], [1.5, 2.5]])
        groups = np.array([0, 0, 1])
        with pytest.raises(ValueError, match="log_expression=True"):
            compare_two_groups_scalar(values, groups, log_expression=True)


class TestScalarGLMComparison:
    """Scalar OLS tests, estimability, and perfect fits."""

    def test_scalar_functions_do_not_expose_null_argument(self):
        """Scalar DE uses fixed t-distribution nulls, not a null selector."""
        import inspect

        assert "null" not in inspect.signature(compare_two_groups_scalar).parameters
        assert "null" not in inspect.signature(compare_glm_scalar).parameters

    def test_matches_manual_ols_contrast(self):
        from scipy.stats import t as t_dist

        rng = np.random.default_rng(11)
        n, n_genes = 9, 6
        x = np.linspace(-1.0, 1.0, n)
        values = rng.normal(size=(n, n_genes)) + x[:, None] * np.linspace(0.0, 1.0, n_genes)
        design = pd.DataFrame({"x": x})

        df = compare_glm_scalar(values, design, contrast="x").set_index("Feature")

        X = np.column_stack([np.ones(n), x])
        xtx_inv = np.linalg.pinv(X.T @ X)
        beta = xtx_inv @ X.T @ values
        resid = values - X @ beta
        df_resid = n - np.linalg.matrix_rank(X)
        sigma2 = np.sum(resid**2, axis=0) / df_resid
        se = np.sqrt(sigma2 * xtx_inv[1, 1])
        t_stat = beta[1] / se
        pvals = 2.0 * t_dist.sf(np.abs(t_stat), df_resid)

        ordered = df.loc[[str(i) for i in range(n_genes)]]
        np.testing.assert_allclose(ordered["Estimate"].to_numpy(), beta[1], rtol=1e-12)
        np.testing.assert_allclose(ordered["P_value"].to_numpy(), pvals, rtol=1e-12)

    def test_continuous_contrast_recovers_planted_scalar_signal(self):
        rng = np.random.default_rng(12)
        n, n_genes = 12, 50
        x = np.linspace(0.0, 1.0, n)
        beta = np.zeros(n_genes)
        beta[:5] = 2.0
        values = 0.2 * rng.standard_normal((n, n_genes)) + x[:, None] * beta[None, :]

        df = compare_glm_scalar(values, pd.DataFrame({"time": x}), contrast="time")

        top10 = set(df.head(10)["Feature"].tolist())
        assert {"0", "1", "2", "3", "4"} <= top10

    def test_log_expression_matches_manual_log_response(self):
        rng = np.random.default_rng(13)
        n, n_genes = 10, 8
        x = np.linspace(0.0, 1.0, n)
        log_values = 1.0 + x[:, None] * np.linspace(0.0, 1.0, n_genes)
        values = np.exp(log_values + 0.05 * rng.standard_normal((n, n_genes)))
        design = pd.DataFrame({"x": x})

        df_logged = compare_glm_scalar(values, design, "x", log_expression=True)
        df_manual = compare_glm_scalar(np.log(values + 1e-12), design, "x")

        df_logged = df_logged.sort_values("Feature").reset_index(drop=True)
        df_manual = df_manual.sort_values("Feature").reset_index(drop=True)
        np.testing.assert_allclose(df_logged["Estimate"], df_manual["Estimate"], rtol=1e-12)
        np.testing.assert_allclose(df_logged["P_value"], df_manual["P_value"], rtol=1e-12)

    def test_log_expression_rejects_nonpositive_values(self):
        values = np.array([[1.0, 2.0], [0.5, -1.0], [1.5, 2.5]])
        design = np.column_stack([np.ones(3), np.arange(3)])
        with pytest.raises(ValueError, match="log_expression=True"):
            compare_glm_scalar(values, design, np.array([0.0, 1.0]), log_expression=True)

    def test_rank_deficient_estimable_contrast_matches_collapsed_design(self):
        rng = np.random.default_rng(14)
        n, n_genes = 9, 7
        x = np.linspace(-1.0, 1.0, n)
        values = rng.normal(size=(n, n_genes)) + x[:, None] * np.linspace(0.0, 1.0, n_genes)
        X_full = np.column_stack([np.ones(n), x])
        X_duplicate = np.column_stack([np.ones(n), x, x])

        df_full = (
            compare_glm_scalar(values, X_full, contrast=np.array([0.0, 1.0]))
            .sort_values("Feature")
            .reset_index(drop=True)
        )
        df_duplicate = (
            compare_glm_scalar(values, X_duplicate, contrast=np.array([0.0, 1.0, 1.0]))
            .sort_values("Feature")
            .reset_index(drop=True)
        )

        np.testing.assert_allclose(df_duplicate["Estimate"], df_full["Estimate"], rtol=1e-12)
        np.testing.assert_allclose(df_duplicate["P_value"], df_full["P_value"], rtol=1e-12)

    def test_rank_deficient_nonestimable_contrast_raises(self):
        values = np.arange(24, dtype=float).reshape(8, 3)
        x = np.linspace(0.0, 1.0, values.shape[0])
        X = np.column_stack([np.ones(values.shape[0]), x, x])

        with pytest.raises(ValueError, match="not estimable"):
            compare_glm_scalar(values, X, contrast=np.array([0.0, 1.0, 0.0]))

    def test_scalar_glm_perfect_fit_preserves_nonzero_effects(
        self,
    ):
        values = np.tile([0.0, 1.0, 1e-9, -1e-9], (4, 1))
        result = compare_glm_scalar(values, np.ones((4, 1)), np.ones(1)).set_index("Feature")
        result = result.loc[["0", "1", "2", "3"]]
        np.testing.assert_array_equal(result["Estimate"], values[0])
        np.testing.assert_array_equal(result["Statistic"], [0, np.inf, np.inf, np.inf])
        np.testing.assert_array_equal(result["P_value"], [1, 0, 0, 0])

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("target", ["response", "design", "contrast"])
    def test_scalar_glm_rejects_nonfinite_inputs(self, value, target):
        values = np.arange(4.0)[:, None]
        design = np.ones((4, 1))
        contrast = np.ones(1)
        {"response": values, "design": design, "contrast": contrast}[target].flat[0] = value
        with pytest.raises(ValueError, match="finite"):
            compare_glm_scalar(values, design, contrast)


class TestLogTCauchy:
    """Direct references for log-profile binwise inference and masked designs."""

    @pytest.mark.parametrize("glm", [False, True], ids=["binary", "glm"])
    def test_df_norm_fixed_share_is_excluded(self, glm):
        powers = np.array([[6, 4, 6], [6, 6, 4]] * 3 + [[6, 5, 5]] * 6)[:, None, :]
        groups = np.repeat([0, 1], 6)
        compare = compare_glm if glm else compare_two_groups
        args = (np.c_[np.ones(12), groups], np.array([0, 1])) if glm else (groups,)
        # All totals are exactly 16, so division by 16 gives the reference shares.
        reference = compare(powers / 16, *args, statistic="log_t_cauchy").iloc[0]
        result = compare(powers, *args, statistic="log_t_cauchy", normalize_shape=True).iloc[0]
        assert result.Estimate_per_bin[0] == result.T_per_bin[0] == 0
        assert result.P_value_per_bin[0] == 1
        assert result.Best_bin == reference.Best_bin != 0
        assert result.P_value == reference.P_value

    def test_df_norm_constant_share_without_intercept(self):
        x = np.tile([-1, 1], 8)
        second = np.where(x > 0, 0.5 - 2.0**-12, 2.0**-12)
        total = 1 + 2.0**-52
        powers = np.column_stack([np.full(len(x), 0.5), second, total - 0.5 - second])
        args = (x[:, None], np.array([1.0]))
        reference = compare_glm(powers[:, None, 1:] / total, *args, statistic="log_t_cauchy").iloc[
            0
        ]
        for values in (powers, powers * np.where(x[:, None] > 0, 3, 1)):
            row = compare_glm(
                values[:, None], *args, statistic="log_t_cauchy", normalize_shape=True
            ).iloc[0]
            assert row.Estimate_per_bin[0] == row.T_per_bin[0] == 0
            assert row.P_value_per_bin[0] == 1 and row.Best_bin == reference.Best_bin + 1
            assert row.P_value == pytest.approx(reference.P_value, rel=1e-12)
            assert row.P_value < 0.01

    @pytest.mark.parametrize("masked", [False, True])
    def test_df_norm_refines_nuisance_fits_without_refining_noisy_features(self, masked):
        from unittest.mock import patch

        from sonic.comparators import _inference as inference

        group = np.tile([0, 1, 0, 1], 6)
        nuisance = np.tile([0, 0, 1, 1], 6)
        design = np.c_[np.ones(len(group)), nuisance, group]
        shape = np.where(nuisance[:, None], [0.25, 0.75], [0.5, 0.5 + 2.0**-52])
        power = shape * np.where(group[:, None], 3.0, 1.0)
        noise = np.random.default_rng(71).uniform(1, 2, size=power.shape)
        values = np.stack([power, noise], axis=1)
        args = [values, design, np.array([0, 0, 1])]
        compare = compare_glm
        if masked:
            values[0, 0] = np.nan
            args.append(np.isfinite(values).all(axis=-1))
            compare = compare_glm_masked
        with patch.object(
            inference, "_exact_shape_shares", wraps=inference._exact_shape_shares
        ) as exact:
            result = compare(*args, normalize_shape=True, statistic="log_t_cauchy").set_index(
                "Feature"
            )
        exact.assert_called_once()
        np.testing.assert_array_equal(exact.call_args.args[0], power[int(masked) :])
        np.testing.assert_array_equal(result.loc["0", "P_value_per_bin"], 1.0)
        assert result.loc["0", "P_value"] == 1 and pd.isna(result.loc["0", "Best_bin"])

    @pytest.mark.parametrize("coding", ["original", "aliased", "nearly_collinear"])
    def test_exact_nuisance_and_tested_effects(self, coding):
        z = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1])
        x = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1])
        design = np.c_[np.ones(len(z)), z, x]
        contrast = np.array([0, 0, 1])
        expected_effect = np.log(2)
        if coding == "aliased":
            design = np.c_[design, z]
            contrast = np.r_[contrast, 0]
        elif coding == "nearly_collinear":
            design[:, 2] = z + x * 2.0**-20
            expected_effect *= 2**20
        spectra = np.exp2(np.column_stack([z, x]))[:, :, None]
        frame = compare_glm(spectra, design, contrast, statistic="log_t_cauchy").set_index(
            "Feature"
        )
        null, effect = frame.loc["0"], frame.loc["1"]
        assert null.Estimate_per_bin[0] == null.T_per_bin[0] == 0
        assert null.P_value_per_bin[0] == 1
        assert null.P_value == 1
        assert pd.isna(null.Best_bin)
        assert effect.Estimate_per_bin[0] == expected_effect
        assert effect.T_per_bin[0] == np.inf
        assert effect.P_value_per_bin[0] == 0
        assert effect.Best_bin == 0

    def test_nuisance_bin_preserves_signal_and_noisy_null(self):
        x = np.repeat([0, 1], 4)
        z = np.tile([0, 0, 1, 1], 2)
        noise = np.tile([-1, 1], 4)
        design = np.c_[np.ones(8), x, z]
        # Each feature has a nuisance-only first bin. The second has real residual variance.
        signal = np.column_stack([z, 4 * x + noise])
        noisy_null = np.column_stack([z, noise])
        spectra = np.exp2(np.stack([signal, noisy_null], axis=1))
        frame = compare_glm(
            spectra, design, np.array([0, 1, 0]), statistic="log_t_cauchy", freq_weights=[9, 1]
        ).set_index("Feature")
        for row in frame.itertuples():
            assert row.P_value_per_bin[0] == 1
            assert row.Best_bin == 1
            assert row.P_value == pytest.approx(row.P_value_per_bin[1], rel=1e-12)
        assert frame.loc["0", "P_value"] < 0.05
        assert frame.loc["1", "P_value"] == pytest.approx(1)

    @pytest.mark.parametrize("contrast", [np.array([-1, 1]), np.array([1, 0])])
    def test_constant_fits_reused_by_observation_mask(self, contrast, monkeypatch):
        from unittest.mock import patch

        from sonic.comparators import _inference, _regression

        groups = np.repeat([0, 1], 4)
        design = np.c_[1 - groups, groups]
        levels = np.array([0.25, 0.5, 1, 2, 4])
        spectra = np.tile(levels, (8, 6, 1))
        presence = np.ones((8, 6), dtype=bool)
        presence[0, ::2] = presence[-1, 1::2] = False
        spectra[~presence] = np.nan
        monkeypatch.setattr(_inference, "_COMPARISON_WORKSPACE_BYTES", 1)
        with patch.object(
            _regression, "_exact_glm_design", wraps=_regression._exact_glm_design
        ) as fit:
            frame = compare_glm_masked(
                spectra, design, contrast, presence, statistic="log_t_cauchy"
            )
        assert fit.call_count == 2  # One unit fit per mask, across all bins and feature blocks.
        expected = np.log(levels) * contrast.sum()
        for row in frame.itertuples():
            np.testing.assert_array_equal(row.Estimate_per_bin, expected)
            np.testing.assert_array_equal(row.P_value_per_bin, expected == 0)
            if contrast.sum():
                assert row.Best_bin == 0
            else:
                assert pd.isna(row.Best_bin)

    @pytest.mark.parametrize("glm", [False, True])
    @pytest.mark.parametrize("masked", [False, True])
    def test_removed_raw_statistic_is_rejected(self, glm, masked):
        spectra = np.ones((8, 2, 3))
        groups = np.repeat([0, 1], 4)
        if glm:
            fn = compare_glm_masked if masked else compare_glm
            args = [spectra, pd.DataFrame({"group": groups}), "group"]
        else:
            fn = compare_two_groups_masked if masked else compare_two_groups
            args = [spectra, groups]
        if masked:
            args.append(np.ones((8, 2), bool))
        with pytest.raises(ValueError, match="log_t_cauchy"):
            fn(*args, statistic="welch_t_cauchy")

    @pytest.mark.parametrize("shape_only", [False, True])
    def test_binary_matches_scipy_welch(self, shape_only):
        from scipy.stats import ttest_ind

        from sonic.statistics import apply_bh_correction, cauchy_combine

        spectra = np.random.default_rng(83).lognormal(size=(11, 4, 5))
        spectra[0, 0, 0] = 0
        spectra[1, 0, 0] = 1e-15
        original = spectra.copy()
        groups = np.r_[np.zeros(4), np.ones(7)]
        response = normalize_shape(spectra) if shape_only else spectra
        response = np.log(np.maximum(response, 1e-12))
        expected = ttest_ind(response[:4], response[4:], equal_var=False, axis=0)
        result = (
            compare_two_groups(
                spectra, groups, statistic="log_t_cauchy", normalize_shape=shape_only
            )
            .set_index("Feature")
            .loc[[str(i) for i in range(4)]]
        )
        np.testing.assert_allclose(np.stack(result.P_value_per_bin), expected.pvalue, rtol=1e-12)
        np.testing.assert_allclose(np.stack(result.T_per_bin), expected.statistic, rtol=1e-12)
        np.testing.assert_allclose(
            np.stack(result.Estimate_per_bin), response[:4].mean(0) - response[4:].mean(0)
        )
        np.testing.assert_array_equal(result.Best_bin, np.argmin(expected.pvalue, axis=-1))
        assert result.Best_bin.dtype == pd.Int64Dtype()
        np.testing.assert_allclose(result.Statistic, np.max(np.abs(expected.statistic), axis=-1))
        np.testing.assert_allclose(result.P_value, cauchy_combine(expected.pvalue))
        np.testing.assert_allclose(result.P_adj, apply_bh_correction(result.P_value))
        np.testing.assert_array_equal(spectra, original)

    @pytest.mark.parametrize("shape_only", [False, True])
    @pytest.mark.parametrize("aliased", [False, True])
    def test_glm_matches_direct_ols_with_nuisance(self, shape_only, aliased):
        from scipy.stats import t

        from sonic.statistics import cauchy_combine

        rng = np.random.default_rng(84)
        x = np.linspace(-1, 1, 13)
        nuisance = 0.7 * x + rng.normal(size=13)
        matrix = np.c_[np.ones(13), x, nuisance]
        contrast = np.array([0.0, 1.0, 0.0])
        if aliased:
            matrix = np.c_[matrix, x]
            contrast = np.r_[contrast, 1.0]
        spectra = np.exp(rng.normal(size=(13, 4, 5)) + 2 * nuisance[:, None, None])
        values = normalize_shape(spectra) if shape_only else spectra
        response = np.log(np.maximum(values, 1e-12)).reshape(13, -1)
        beta = np.linalg.lstsq(matrix, response, rcond=None)[0]
        residual = response - matrix @ beta
        df = 13 - np.linalg.matrix_rank(matrix)
        se = np.sqrt(
            (residual**2).sum(axis=0) / df * np.sum((contrast @ np.linalg.pinv(matrix)) ** 2)
        )
        abs_t = np.abs(contrast @ beta / se).reshape(4, 5)
        expected_p = 2 * t.sf(abs_t, df)
        result = compare_glm(
            spectra, matrix, contrast, statistic="log_t_cauchy", normalize_shape=shape_only
        )
        result = result.set_index("Feature").loc[[str(i) for i in range(4)]]
        np.testing.assert_allclose(np.stack(result.P_value_per_bin), expected_p, rtol=1e-11)
        np.testing.assert_allclose(
            np.stack(result.Estimate_per_bin), (contrast @ beta).reshape(4, 5), rtol=1e-11
        )
        np.testing.assert_allclose(
            np.stack(result.T_per_bin), (contrast @ beta / se).reshape(4, 5), rtol=1e-11
        )
        np.testing.assert_array_equal(result.Best_bin, np.argmin(expected_p, axis=-1))
        np.testing.assert_allclose(result.Statistic, abs_t.max(axis=-1), rtol=1e-11)
        np.testing.assert_allclose(result.P_value, cauchy_combine(expected_p), rtol=1e-11)
        assert result.attrs["per_bin_test"] == "ols"

    @pytest.mark.parametrize("glm", [False, True])
    @pytest.mark.parametrize("shape_only", [False, True])
    def test_masked_matches_separately_subsetted_fits(self, glm, shape_only):
        rng = np.random.default_rng(85)
        spectra = rng.lognormal(size=(12, 5, 4))
        groups = np.repeat([0, 1], 6)
        design = pd.DataFrame({"age": np.arange(12.0), "batch": np.tile([0, 1], 6)})
        presence = np.ones((12, 5), bool)
        presence[0, :2] = False
        presence[7:9, 2:4] = False
        spectra[~presence] = np.nan
        options = {"statistic": "log_t_cauchy", "normalize_shape": shape_only}
        if glm:
            result = compare_glm_masked(spectra, design, "age", presence, **options)
        else:
            result = compare_two_groups_masked(spectra, groups, presence, **options)
        result = result.set_index("Feature")
        for gene in range(5):
            mask = presence[:, gene]
            values = spectra[mask, gene : gene + 1]
            expected = (
                compare_glm(values, design.loc[mask], "age", **options)
                if glm
                else compare_two_groups(values, groups[mask], **options)
            ).iloc[0]
            actual = result.loc[str(gene)]
            np.testing.assert_allclose(actual.Statistic, expected.Statistic, rtol=1e-12)
            np.testing.assert_allclose(actual.P_value, expected.P_value, rtol=1e-12)
            np.testing.assert_allclose(actual.P_value_per_bin, expected.P_value_per_bin, rtol=1e-12)
            np.testing.assert_allclose(actual.T_per_bin, expected.T_per_bin, rtol=1e-12)
            np.testing.assert_allclose(
                actual.Estimate_per_bin, expected.Estimate_per_bin, rtol=1e-12
            )
            assert actual.Best_bin == expected.Best_bin
            if glm:
                assert actual.n_obs == mask.sum()
                assert actual.df_resid == mask.sum() - 3
            else:
                assert actual.n_obs_A == np.sum(mask & (groups == 0))
                assert actual.n_obs_B == np.sum(mask & (groups == 1))

    @pytest.mark.parametrize("glm", [False, True])
    def test_masked_ineligible_rows_keep_nan_and_counts(self, glm):
        spectra = np.random.default_rng(86).lognormal(size=(8, 4, 3))
        groups = np.repeat([0, 1], 4)
        presence = np.ones((8, 4), bool)
        presence[4:, 0] = False
        presence[2:, 1] = False
        presence[:, 2] = False
        spectra[~presence] = np.nan
        if glm:
            result = compare_glm_masked(
                spectra,
                pd.DataFrame({"group": groups}),
                "group",
                presence,
                statistic="log_t_cauchy",
                min_resid_df=2,
            ).set_index("Feature")
            np.testing.assert_array_equal(result.loc[["0", "1", "2"], "df_resid"], [3, 1, 0])
        else:
            result = compare_two_groups_masked(
                spectra, groups, presence, statistic="log_t_cauchy"
            ).set_index("Feature")
        assert result.loc[["0", "1", "2"], ["Statistic", "P_value", "P_adj"]].isna().all().all()
        for column in ("Estimate_per_bin", "T_per_bin", "P_value_per_bin"):
            values = np.stack(result.loc[["0", "1", "2"], column])
            assert values.shape == (3, 3)
            assert np.isnan(values).all()
        assert result.loc[["0", "1", "2"], "Best_bin"].isna().all()
        assert np.isfinite(result.loc["3", "P_value"])
        assert result.loc["3", "P_adj"] == result.loc["3", "P_value"]

    @pytest.mark.parametrize("glm", [False, True])
    def test_constant_bins_and_perfect_effects(self, glm):
        groups = np.repeat([0, 1], 6)
        response = np.c_[np.zeros(12), np.ones(12), groups * 2.0]
        spectra = np.exp(response)[:, :, None]
        args = (pd.DataFrame({"group": groups}), "group") if glm else (groups,)
        compare = compare_glm if glm else compare_two_groups
        expected = compare(spectra, *args, statistic="log_t_cauchy").set_index("Feature")
        padded = np.concatenate([spectra, np.ones_like(spectra), np.zeros_like(spectra)], axis=-1)
        actual = compare(padded, *args, statistic="log_t_cauchy").set_index("Feature")
        np.testing.assert_allclose(actual.P_value, expected.P_value, atol=1e-14)
        assert actual.loc["0", "P_value"] == actual.loc["1", "P_value"] == 1
        assert actual.loc["2", "P_value"] < 1e-100
        assert actual.loc[["0", "1"], "Best_bin"].isna().all()
        assert actual.loc["2", "Best_bin"] == 0
        # Constant log responses still carry an intercept contrast.
        if glm:
            intercept = compare_glm(
                np.exp(np.ones((8, 1, 2))), np.ones((8, 1)), np.ones(1), statistic="log_t_cauchy"
            )
            assert intercept.P_value.iloc[0] < 1e-100

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("glm", [False, True])
    @pytest.mark.parametrize(
        "options,match",
        [
            ({"null": "permutation"}, "analytic"),
            ({"null": "rotation"}, "analytic"),
            ({"null": "plugin"}, "analytic"),
            ({"covariance": "pooled"}, "covariance"),
        ],
    )
    def test_invalid_options_are_rejected(self, masked, glm, options, match):
        spectra = np.ones((8, 2, 3))
        groups = np.repeat([0, 1], 4)
        presence = np.ones((8, 2), bool)
        if glm:
            fn = compare_glm_masked if masked else compare_glm
            args = [spectra, pd.DataFrame({"group": groups}), "group"]
        else:
            fn = compare_two_groups_masked if masked else compare_two_groups
            args = [spectra, groups]
        if masked:
            args.append(presence)
        with pytest.raises(ValueError, match=match):
            fn(*args, statistic="log_t_cauchy", **options)

    def test_explicit_binary_glm_uses_ols_not_welch(self):
        from scipy.stats import ttest_ind

        response = np.random.default_rng(3).normal(size=(16, 2, 3))
        response[:4] *= 3
        groups = np.r_[np.zeros(4), np.ones(12)]
        spectra = np.exp(response)
        glm = (
            compare_glm(spectra, pd.DataFrame({"group": groups}), "group", statistic="log_t_cauchy")
            .set_index("Feature")
            .sort_index()
        )
        binary = (
            compare_two_groups(spectra, groups, statistic="log_t_cauchy")
            .set_index("Feature")
            .sort_index()
        )
        expected = ttest_ind(response[:4], response[4:], axis=0, equal_var=True)
        np.testing.assert_allclose(np.stack(glm.P_value_per_bin), expected.pvalue, rtol=1e-12)
        assert not np.allclose(np.stack(glm.P_value_per_bin), np.stack(binary.P_value_per_bin))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("glm", [False, True])
    def test_observed_nonfinite_values_raise_before_logging(self, bad, glm):
        spectra = np.ones((8, 2, 3))
        spectra[0, 0, 0] = bad
        with pytest.raises(ValueError, match="finite"):
            if glm:
                compare_glm(
                    spectra, pd.DataFrame({"age": np.arange(8.0)}), "age", statistic="log_t_cauchy"
                )
            else:
                compare_two_groups(spectra, np.repeat([0, 1], 4), statistic="log_t_cauchy")

    def test_masked_nonestimable_full_design_skips_without_fitting(self):
        x = np.arange(8.0)
        result = compare_glm_masked(
            np.ones((8, 2, 3)),
            np.c_[np.ones(8), x, x],
            np.array([0.0, 1.0, 0.0]),
            np.ones((8, 2), bool),
            statistic="log_t_cauchy",
        )
        assert result.P_value.isna().all()
        assert (result.df_resid == 6).all()
        with pytest.raises(ValueError, match="not estimable"):
            compare_glm(
                np.ones((8, 2, 3)),
                np.c_[np.ones(8), x, x],
                np.array([0.0, 1.0, 0.0]),
                statistic="log_t_cauchy",
            )

    @pytest.mark.parametrize("glm", [False, True])
    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("shape_only", [False, True])
    def test_weights_match_formula_and_preserve_bin_pvalues(self, glm, masked, shape_only):
        rng = np.random.default_rng(23)
        spectra = rng.lognormal(size=(12, 3, 4))
        spectra[:, :, 1] = 1.0
        groups = np.repeat([0, 1], 6)
        design = pd.DataFrame({"x": np.arange(12.0)})
        presence = np.ones((12, 3), bool)
        if masked:
            presence[0, 0] = False
            presence[6, 1] = False
        if glm:
            fn = compare_glm_masked if masked else compare_glm
            args = [spectra, design, "x"]
        else:
            fn = compare_two_groups_masked if masked else compare_two_groups
            args = [spectra, groups]
        if masked:
            args.append(presence)
        options = {"statistic": "log_t_cauchy", "normalize_shape": shape_only}
        original = fn(*args, **options).set_index("Feature").sort_index()
        weights = np.array([1.0, 10.0, 3.0, 0.25])
        actual = fn(*args, **options, freq_weights=weights).set_index("Feature").sort_index()
        uniform = fn(*args, **options, freq_weights=np.ones(4)).set_index("Feature").sort_index()
        np.testing.assert_allclose(uniform.P_value, original.P_value, rtol=1e-12)
        np.testing.assert_allclose(uniform.Statistic, original.Statistic, rtol=1e-12)
        np.testing.assert_array_equal(
            np.stack(actual.P_value_per_bin), np.stack(original.P_value_per_bin)
        )
        for gene in range(3):
            values = spectra[presence[:, gene], gene : gene + 1]
            if shape_only:
                values = normalize_shape(values)
            response = np.log(np.maximum(values[:, 0], 1e-12))
            selected = np.any(response != response[:1], axis=0)
            assert (
                actual.loc[str(gene), "Best_bin"]
                == np.flatnonzero(selected)[
                    np.argmin(original.loc[str(gene), "P_value_per_bin"][selected])
                ]
            )
            p = original.loc[str(gene), "P_value_per_bin"][selected]
            w = weights[selected] / weights[selected].sum()
            expected = 0.5 - np.arctan(np.sum(w * np.tan(np.pi * (0.5 - p)))) / np.pi
            np.testing.assert_allclose(actual.loc[str(gene), "P_value"], expected, rtol=1e-11)

    @pytest.mark.parametrize("glm", [False, True])
    def test_sole_informative_bin_determines_result(self, glm):
        groups = np.repeat([0, 1], 6)
        spectra = np.exp(np.stack([groups * 2.0, np.zeros(12)], axis=-1))[:, None, :]
        result = (
            compare_glm(
                spectra,
                pd.DataFrame({"group": groups}),
                "group",
                statistic="log_t_cauchy",
                freq_weights=[0.01, 1],
            )
            if glm
            else compare_two_groups(
                spectra, groups, statistic="log_t_cauchy", freq_weights=[0.01, 1]
            )
        ).iloc[0]
        np.testing.assert_allclose(
            result.P_value, result.P_value_per_bin[0], atol=np.finfo(float).tiny
        )
        assert result.P_value < 1e-100
        assert result.Statistic == abs(result.T_per_bin[0])
        assert result.Best_bin == 0

    @pytest.mark.parametrize("glm", [False, True])
    @pytest.mark.parametrize(
        "weights", [np.zeros(3), [1, 0, 1], [1, -1, 1], [1, np.nan, 1], [1, np.inf, 1], [1, 1]]
    )
    def test_invalid_bin_weights_raise(self, glm, weights):
        spectra = np.ones((8, 2, 3))
        with pytest.raises(ValueError, match="weights"):
            if glm:
                compare_glm(
                    spectra,
                    np.c_[np.ones(8), np.arange(8)],
                    np.array([0.0, 1.0]),
                    statistic="log_t_cauchy",
                    freq_weights=weights,
                )
            else:
                compare_two_groups(
                    spectra, np.repeat([0, 1], 4), statistic="log_t_cauchy", freq_weights=weights
                )

    def test_best_bin_uses_smallest_welch_p_not_largest_absolute_t(self):
        groups = np.r_[np.zeros(3), np.ones(12)]
        response = np.zeros((15, 1, 2))
        response[:3, 0, 0] = np.array([-1, 0, 1]) + 2.2 / np.sqrt(3)
        variation = np.linspace(-1, 1, 12)
        response[3:, 0, 1] = variation - 2 * variation.std(ddof=1) / np.sqrt(12)
        result = compare_two_groups(np.exp(response), groups, statistic="log_t_cauchy").iloc[0]
        assert np.argmax(np.abs(result.T_per_bin)) == 0
        assert result.Best_bin == np.argmin(result.P_value_per_bin) == 1
