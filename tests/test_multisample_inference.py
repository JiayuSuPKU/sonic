"""Tests for spectral covariance estimation and null calibration."""

from __future__ import annotations

import itertools
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest

from sonic.comparators import _inference as l2
from sonic.comparators import _regression as lm
from sonic.comparators._inference import (
    _estimate_glm_masked_null_covariance,
    _estimate_two_group_masked_null_covariance,
    _l2_pvalues_from_state,
    _PluginNullState,
)
from sonic.comparators.multisample import (
    compare_glm,
    compare_glm_masked,
    compare_two_groups,
    compare_two_groups_masked,
)
from sonic.comparators.normalization import normalize_shape
from sonic.statistics import apply_bh_correction, liu_sf


class TestLiuCalibration:
    """Gaussian plug-in calibration of L2 tails, masking, and low-df behavior."""

    @pytest.mark.parametrize("masked", [False, True])
    def test_df_norm_plugin_is_invariant_to_duplicate_timestamp(self, masked):
        group = np.tile([0, 0, 1, 1], 6)
        time = (1_730_000_000 + 86_400 * (np.arange(24) - 11.5)) * 1e9
        powers = np.random.default_rng(325).lognormal(size=(24, 4, 7))
        powers[:, 0, 0] *= np.exp(2 * group)
        design = pd.DataFrame({"group": group, "time": time})

        def fit(design):
            options = {"normalize_shape": True, "null": "plugin"}
            if masked:
                presence = np.ones(powers.shape[:2], dtype=bool)
                presence[:3, 1] = False
                result = compare_glm_masked(powers, design, "group", presence, **options)
            else:
                result = compare_glm(powers, design, "group", **options)
            return result.set_index("Feature").sort_index()

        reference = fit(design)
        duplicate = fit(design.assign(time_copy=time))
        np.testing.assert_allclose(
            duplicate[["Statistic", "P_value", "P_adj"]],
            reference[["Statistic", "P_value", "P_adj"]],
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            np.stack(duplicate.Estimate_per_bin), np.stack(reference.Estimate_per_bin), atol=1e-11
        )

    def test_saturated_design_rejects_plugin(self):
        spectra = np.ones((2, 1, 3))
        with pytest.raises(ValueError, match="no residual degrees of freedom"):
            compare_glm(spectra, np.eye(2), np.array([1, -1]), null="plugin")

    def test_synthetic_h0_pvalues_are_uniform(self):
        rng = np.random.default_rng(0)
        n_a, n_b, K = 4, 4, 30
        spectra = np.exp(rng.standard_normal((n_a + n_b, 5000, K)))
        groups = np.array([0] * n_a + [1] * n_b)
        df = compare_two_groups(spectra, groups, statistic="log_l2", covariance="pooled")
        _, ks_p = kstest(df["P_value"].to_numpy(), "uniform")
        assert ks_p > 0.01, f"plug-in p-values not uniform under H0: KS p={ks_p:.4f}"
        assert 0.45 < df["P_value"].mean() < 0.55

    def test_plugin_breaks_permutation_floor_on_implanted_signal(self):
        """Plug-in p-values can beat the exact-permutation floor on strong signals."""
        rng = np.random.default_rng(7)
        n_per = 4
        n_genes, K = 50, 30
        a = np.exp(rng.normal(loc=0.0, scale=0.2, size=(n_per, n_genes, K)))
        b = np.exp(rng.normal(loc=0.0, scale=0.2, size=(n_per, n_genes, K)))
        b[:, :5, :3] *= 5.0
        spectra = np.concatenate([a, b], axis=0)
        groups = np.array([0] * n_per + [1] * n_per)
        df_perm = compare_two_groups(
            spectra, groups, statistic="log_l2", null="permutation", random_state=0
        )
        df_plugin = compare_two_groups(spectra, groups, statistic="log_l2", null="plugin")
        for g in ("0", "1", "2", "3", "4"):
            p_perm = df_perm.loc[df_perm["Feature"] == g, "P_value"].iloc[0]
            p_plugin = df_plugin.loc[df_plugin["Feature"] == g, "P_value"].iloc[0]
            assert p_plugin < p_perm, f"gene {g}: plugin={p_plugin:.3g} not < perm={p_perm:.3g}"
            assert p_plugin < 1.0 / 71.0, f"gene {g}: plugin={p_plugin:.3g} above perm floor"

    def test_log_t_cauchy_analytic_schema(self):
        """Analytic log t-Cauchy retains the feature and per-bin result schema."""
        rng = np.random.default_rng(3)
        spectra = np.exp(rng.standard_normal((6, 50, 20)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        df = compare_two_groups(spectra, groups, statistic="log_t_cauchy", null="analytic")
        assert "P_value_per_bin" in df.columns
        assert df["P_value"].between(0, 1).all()

    def test_unknown_null_raises(self):
        rng = np.random.default_rng(4)
        spectra = np.exp(rng.standard_normal((6, 50, 20)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        with pytest.raises(ValueError, match="Unknown null"):
            compare_two_groups(spectra, groups, statistic="log_l2", null="bootstrap")

    def test_small_df_emits_user_warning(self):
        """Plug-in null warns when residual df makes variance estimates unstable."""
        rng = np.random.default_rng(0)
        spectra = np.exp(rng.standard_normal((3, 50, 20)))
        groups = np.array([0, 1, 1])
        with pytest.warns(UserWarning, match="log_l2 \\+ null='plugin' at residual df=1"):
            compare_glm(spectra, np.c_[np.ones(3), groups], np.array([0, 1]), null="plugin")
        spectra = np.exp(rng.standard_normal((4, 50, 20)))
        groups = np.array([0, 0, 1, 1])
        with pytest.warns(UserWarning, match="log_l2 \\+ null='plugin' at residual df=2"):
            compare_two_groups(spectra, groups, statistic="log_l2", null="plugin")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warnings.filterwarnings("ignore", message="log_l2 [+] null='plugin'")
            compare_two_groups(spectra, groups, statistic="log_l2", null="plugin")
        spectra = np.exp(rng.standard_normal((6, 50, 20)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            compare_two_groups(spectra, groups, statistic="log_l2", null="plugin")

    def test_full_sigma_calibrates_under_correlated_bins(self):
        """Full-covariance plug-in p-values should stay calibrated with correlated bins."""
        rng = np.random.default_rng(42)
        n_a, n_b, G, K = 4, 4, 2000, 30
        # Shared noise creates near rank-1 covariance across bins.
        shared = rng.standard_normal((n_a + n_b, G, 1)) * 1.0
        per_bin = rng.standard_normal((n_a + n_b, G, K)) * 0.2
        log_y = shared + per_bin
        spectra = np.exp(log_y)
        groups = np.array([0] * n_a + [1] * n_b)
        df = compare_two_groups(spectra, groups, statistic="log_l2", covariance="pooled")
        # Four binomial standard errors allow sampling variation in either direction.
        alpha = 0.05
        rejection_rate = (df.P_value < alpha).mean()
        assert abs(rejection_rate - alpha) < 4 * np.sqrt(alpha * (1 - alpha) / G)

    def test_masked_plugin_parity_with_unmasked_when_complete(self):
        """All-present masks should match the unmasked plug-in path."""
        rng = np.random.default_rng(5)
        n_samples, n_genes, K = 6, 100, 20
        spectra = np.exp(rng.standard_normal((n_samples, n_genes, K)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        presence = np.ones((n_samples, n_genes), dtype=bool)
        df_un = (
            compare_two_groups(spectra, groups, statistic="log_l2", null="plugin")
            .sort_values("Feature")
            .reset_index(drop=True)
        )
        df_m = (
            compare_two_groups_masked(spectra, groups, presence, statistic="log_l2", null="plugin")
            .sort_values("Feature")
            .reset_index(drop=True)
        )
        np.testing.assert_allclose(
            df_un["Statistic"].to_numpy(), df_m["Statistic"].to_numpy(), atol=1e-12
        )
        np.testing.assert_allclose(
            df_un["P_value"].to_numpy(), df_m["P_value"].to_numpy(), atol=1e-12
        )

    def test_masked_plugin_calibration_under_synthetic_missingness(self):
        """Random missingness should not strongly distort H0 false positives."""
        rng = np.random.default_rng(0)
        n_samples, n_genes, K = 8, 800, 20
        spectra = np.exp(rng.standard_normal((n_samples, n_genes, K)))
        groups = np.array([0] * 4 + [1] * 4)

        df_full = compare_two_groups(spectra, groups, statistic="log_l2", covariance="pooled")
        fpr_full = (df_full["P_value"] < 0.05).mean()

        presence = rng.uniform(size=(n_samples, n_genes)) >= 0.25
        df_m = compare_two_groups_masked(
            spectra,
            groups,
            presence,
            statistic="log_l2",
            covariance="pooled",
            min_samples_per_group=2,
        )
        valid = df_m["P_value"].notna()
        assert valid.sum() > 100, f"only {valid.sum()} testable genes after masking"
        fpr_m = (df_m.loc[valid, "P_value"] < 0.05).mean()

        assert 0.0 < fpr_m < 0.20, f"masked FPR={fpr_m:.3f} far from nominal"
        assert 0.5 * fpr_full < fpr_m < 2.0 * fpr_full + 0.02, (
            f"masked vs unmasked FPR mismatch: masked={fpr_m:.3f} unmasked={fpr_full:.3f}"
        )

    def test_masked_plugin_skips_genes_with_insufficient_presence(self):
        """Genes below per-group presence thresholds should report NaN p-values."""
        rng = np.random.default_rng(7)
        n_samples, n_genes, K = 6, 20, 12
        spectra = np.exp(rng.standard_normal((n_samples, n_genes, K)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        presence = np.ones((n_samples, n_genes), dtype=bool)
        presence[1:3, 0] = False
        df = compare_two_groups_masked(
            spectra,
            groups,
            presence,
            statistic="log_l2",
            null="plugin",
            min_samples_per_group=2,
        )
        row = df[df["Feature"] == "0"].iloc[0]
        assert np.isnan(row["P_value"]), f"expected NaN, got {row['P_value']}"
        assert row["n_obs_A"] == 1
        valid_genes = df[df["Feature"] != "0"]
        assert valid_genes["P_value"].notna().all()

    def test_masked_plugin_raises_when_no_eligible_genes(self):
        """A fully ineligible mask should raise instead of returning all-NaN results."""
        rng = np.random.default_rng(8)
        n_samples, n_genes, K = 6, 10, 12
        spectra = np.exp(rng.standard_normal((n_samples, n_genes, K)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        presence = np.ones((n_samples, n_genes), dtype=bool)
        presence[0:3, :] = False
        with pytest.raises(ValueError, match="no genes meet"):
            compare_two_groups_masked(
                spectra,
                groups,
                presence,
                statistic="log_l2",
                null="plugin",
                min_samples_per_group=2,
            )

    def test_masked_log_t_cauchy_analytic_schema(self):
        """Masked log t-Cauchy includes per-bin p-values for eligible features."""
        rng = np.random.default_rng(9)
        n_samples, n_genes, K = 6, 10, 12
        spectra = np.exp(rng.standard_normal((n_samples, n_genes, K)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        presence = np.ones((n_samples, n_genes), dtype=bool)
        df = compare_two_groups_masked(
            spectra,
            groups,
            presence,
            statistic="log_t_cauchy",
            null="analytic",
        )
        assert "P_value_per_bin" in df.columns
        assert df["P_value"].between(0, 1).all()

    @pytest.mark.parametrize("mode", ["two_group", "glm"])
    @pytest.mark.parametrize("amplitude", [1.0, 1e-5, 0.0])
    def test_masked_liu_reuses_fits_without_changing_tails(self, mode, amplitude):
        rng = np.random.default_rng(32)
        spectra = np.exp(amplitude * rng.normal(size=(12, 12, 5)))
        groups = np.repeat([0, 1], 6)
        presence = np.ones((12, 12), dtype=bool)
        presence[0, :4] = False
        presence[6, 4:8] = False
        presence[:, -1] = False
        if mode == "two_group":
            state = _estimate_two_group_masked_null_covariance(
                spectra, groups, presence, covariance="pooled"
            )
        else:
            design = np.column_stack([np.ones(12), groups, rng.normal(size=12)])
            state = _estimate_glm_masked_null_covariance(
                spectra, design, np.array([0.0, 1.0, 0.0]), presence, covariance="pooled"
            )
        expected = np.full(12, np.nan)
        for j in np.flatnonzero(state["eligible"]):
            expected[j] = liu_sf(state["observed"][j] ** 2, state["eigenvalues"][j])
        with patch("sonic.comparators._inference.liu_sf", wraps=liu_sf) as fitted:
            actual = _l2_pvalues_from_state(state)
            expected_fits = (
                np.unique(state["contrast_scale"][state["eligible"]]).size if amplitude else 0
            )
            assert fitted.call_count == expected_fits
        # Batched and scalar tail evaluation can differ by floating-point roundoff.
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0, equal_nan=True)


class TestPermutationCalibration:
    """Exact and sampled binary permutation tails."""

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("exact", [False, True])
    def test_permutation_tail_uses_exact_or_sampled_correction(self, masked, exact):
        values = np.array([0, 0, 0, 10, 10, 10], dtype=float)
        groups = np.array([0, 0, 0, 1, 1, 1])
        spectra = np.exp(values[:, None, None])
        n_perm, seed = 17, 2
        if exact:
            assignments = []
            for indices in itertools.combinations(range(6), 3):
                labels = np.ones(6, dtype=int)
                labels[list(indices)] = 0
                assignments.append(labels)
        else:
            rng = np.random.default_rng(seed)
            labels = groups.copy()
            assignments = []
            for _ in range(n_perm):
                rng.shuffle(labels)
                assignments.append(labels.copy())
        ge = sum(abs(values[g == 0].mean() - values[g == 1].mean()) >= 10 for g in assignments)
        expected = ge / len(assignments) if exact else (ge + 1) / (len(assignments) + 1)
        kwargs = {
            "null": "permutation",
            "n_perm": n_perm,
            "random_state": seed,
            "max_exact_permutations": 10000 if exact else 0,
        }
        if masked:
            # Extra missing samples exercise the per-gene exchangeability subset.
            spectra = np.concatenate([spectra, np.ones((2, 1, 1))])
            groups = np.concatenate([groups, [0, 1]])
            presence = np.array([True] * 6 + [False] * 2)[:, None]
            result = compare_two_groups_masked(spectra, groups, presence, **kwargs)
        else:
            result = compare_two_groups(spectra, groups, **kwargs)
        assert result.P_value.iloc[0] == pytest.approx(expected)
        if exact:
            assert expected == 0.1

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("exact", [False, True])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_streamed_permutations_match_materialized_weighted_null(self, masked, exact, dtype):
        rng = np.random.default_rng(37)
        spectra = rng.lognormal(size=(8, 4, 5)).astype(dtype)
        spectra[0, 0, 0] = 0
        spectra[1, 0, 1] = 1e-20
        spectra[:, 2] = 1  # Every permutation ties for this gene.
        original = spectra.copy()
        groups = np.repeat([0, 1], 4)
        weights = np.arange(1.0, 6.0)
        weights /= weights.sum()
        presence = np.ones((8, 4), dtype=bool)
        if masked:
            presence[[0, 4], 1] = False
            presence[:3, 3] = False  # Too few observations in group A.

        def statistic(values, labels):
            a = np.log(np.maximum(values[labels == 0], 1e-12)).mean(axis=0)
            b = np.log(np.maximum(values[labels == 1], 1e-12)).mean(axis=0)
            return np.sqrt(np.sum(weights * (a - b) ** 2, axis=-1))

        # Materialize the reference null, retaining the public API's RNG sequence.
        perm_rng = np.random.default_rng(19)
        expected = []

        def sample_assignments(labels):
            assignments = []
            for _ in range(43):
                perm_rng.shuffle(labels)
                assignments.append(labels.copy())
            return assignments

        shared = None if exact else sample_assignments(groups.copy())
        for gene in range(4):
            keep = presence[:, gene]
            labels = groups[keep].copy()
            if min(np.bincount(labels, minlength=2)) < 2:
                expected.append((np.nan, np.nan))
                continue
            if exact:
                assignments = []
                for indices in itertools.combinations(range(len(labels)), int((labels == 0).sum())):
                    assignment = np.ones(len(labels), dtype=int)
                    assignment[list(indices)] = 0
                    assignments.append(assignment)
            else:
                assignments = shared if keep.all() else sample_assignments(labels)
            observed = statistic(spectra[keep, gene], groups[keep])
            null = np.array(
                [statistic(spectra[keep, gene], assignment) for assignment in assignments]
            )
            correction = 0 if exact else 1
            pvalue = ((null >= observed).sum() + correction) / (len(null) + correction)
            expected.append((observed, pvalue))

        kwargs = {
            "null": "permutation",
            "freq_weights": weights,
            "n_perm": 43,
            "random_state": 19,
            "max_exact_permutations": 10000 if exact else 0,
        }
        if masked:
            result = compare_two_groups_masked(spectra, groups, presence, **kwargs)
        else:
            result = compare_two_groups(spectra, groups, **kwargs)
        result = result.set_index("Feature").loc[[str(i) for i in range(4)]]
        np.testing.assert_allclose(result[["Statistic", "P_value"]], expected, equal_nan=True)
        np.testing.assert_array_equal(spectra, original)

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("poison", ["value", "weight"])
    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_permutation_rejects_nonfinite_observations_and_weights(self, masked, poison, value):
        spectra = np.random.default_rng(12).lognormal(size=(8, 2, 3))
        groups = np.repeat([0, 1], 4)
        weights = np.ones(3)
        if poison == "value":
            spectra[0, 0, 0] = value
        else:
            weights[0] = value
        with pytest.raises(ValueError, match="finite"):
            if masked:
                compare_two_groups_masked(
                    spectra,
                    groups,
                    np.ones((8, 2), dtype=bool),
                    null="permutation",
                    freq_weights=weights,
                )
            else:
                compare_two_groups(spectra, groups, null="permutation", freq_weights=weights)


class TestCovarianceEstimation:
    """Covariance choices, binary adapters, diagnostics, and bounded workspace."""

    def test_existing_paper_helper_imports_are_preserved(self):
        from sonic.comparators import multisample

        for name in (
            "_effective_rank_from_eigenvalues",
            "_estimate_glm_null_covariance",
            "_estimate_two_group_masked_null_covariance",
            "_estimate_two_group_null_covariance",
            "_l2_pvalues_from_state",
        ):
            assert getattr(multisample, name) is getattr(l2, name)

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("shape_only", [False, True])
    def test_per_gene_binary_glm_equivalence_and_contrast_scaling(self, masked, shape_only):
        rng = np.random.default_rng(31)
        groups = np.repeat([0, 1], 5)
        spectra = rng.lognormal(size=(10, 6, 17))
        presence = np.ones((10, 6), bool)
        presence[0, :3] = False
        kwargs = {
            "covariance": "per_gene",
            "normalize_shape": shape_only,
            "freq_weights": np.arange(1, 18),
        }
        binary = (
            compare_two_groups_masked(spectra, groups, presence, **kwargs)
            if masked
            else compare_two_groups(spectra, groups, **kwargs)
        )
        design = np.column_stack([np.ones(10), groups, 2 * groups])  # Estimable despite aliasing.
        contrast = np.array([0.0, 1.0, 2.0])

        def glm(values, c):
            result = (
                compare_glm_masked(values, design, c, presence, **kwargs)
                if masked
                else compare_glm(values, design, c, **kwargs)
            )
            return result.set_index("Feature").sort_index()

        reference = glm(spectra, contrast)
        columns = ["Statistic", "P_value", "P_adj"]
        np.testing.assert_allclose(
            reference[columns], binary.set_index("Feature").sort_index()[columns], atol=1e-12
        )
        scaled = glm(spectra, -3 * contrast)
        np.testing.assert_allclose(scaled.Statistic, 3 * reference.Statistic, atol=1e-12)
        np.testing.assert_allclose(scaled.P_value, reference.P_value, atol=1e-12)
        if not shape_only:
            # A shared bin offset is absorbed by the intercept on the log scale.
            shifted = glm(spectra * np.linspace(1, 2, 17), contrast)
            np.testing.assert_allclose(shifted[columns], reference[columns], atol=1e-12)

    @pytest.mark.parametrize("glm", [False, True])
    @pytest.mark.parametrize("masked", [False, True])
    def test_per_gene_degenerate_profiles_and_feature_independence(self, glm, masked):
        from sonic.comparators import multisample as ms

        rng = np.random.default_rng(19)
        groups = np.repeat([0, 1], 4)
        logs = rng.normal(size=(8, 5, 11))
        logs[:, 0] = 0.0
        logs[:, 1] = groups[:, None] * np.linspace(0, 1, 11)  # Perfect nonzero contrast.
        logs[:, 2] = 0.0
        logs[:, 2, 0] = np.tile([0.25, -0.25], 4)  # One informative bin.
        logs[:, 3] = 1e-7 + 1e-14 * rng.normal(size=(8, 11))
        spectra = np.exp(logs)
        weights = np.ones(11)
        presence = np.ones((8, 5), bool)
        presence[0, :] = not masked
        design = np.column_stack([np.ones(8), groups])
        method = "compare_" + ("glm" if glm else "two_groups") + ("_masked" if masked else "")

        def run(values, masks):
            args = [values, design, np.array([0.0, 1.0])] if glm else [values, groups]
            if masked:
                args.append(masks)
            return (
                getattr(ms, method)(*args, covariance="per_gene", freq_weights=weights)
                .set_index("Feature")
                .sort_index()
            )

        result = run(spectra, presence)
        assert result.loc["0", "P_value"] == 1.0
        assert np.isnan(result.loc["1", "P_value"])
        assert np.isnan(result.loc["1", "P_adj"])
        assert np.isfinite(result.loc["2", "P_value"])
        assert np.isfinite(result.loc["3", "P_value"])
        # Other genes must not supply either covariance or numerical tolerance.
        for j in [3, 4]:
            alone = run(spectra[:, j : j + 1], presence[:, j : j + 1])
            np.testing.assert_allclose(
                result.loc[str(j), ["Statistic", "P_value"]].to_numpy(dtype=float),
                alone.loc["0", ["Statistic", "P_value"]].to_numpy(dtype=float),
                atol=1e-12,
            )
        enlarged = np.concatenate([spectra, np.exp(100 + rng.normal(size=(8, 1, 11)))], axis=1)
        together = run(enlarged, np.column_stack([presence, presence[:, 0]]))
        np.testing.assert_allclose(
            together.loc[result.index, ["Statistic", "P_value"]],
            result[["Statistic", "P_value"]],
            atol=1e-12,
            equal_nan=True,
        )

    def test_per_gene_low_df_retains_plugin_warning(self):
        rng = np.random.default_rng(5)
        with pytest.warns(UserWarning, match="residual df=2"):
            result = compare_two_groups(
                rng.lognormal(size=(4, 5, 29)), np.repeat([0, 1], 2), covariance="per_gene"
            )
        assert result.P_value.between(0, 1).all()

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("covariance", ["pooled", "per_gene"])
    @pytest.mark.parametrize("kwargs", [{"null": "permutation"}, {"statistic": "log_t_cauchy"}])
    def test_covariance_selector_rejects_inapplicable_binary_tests(
        self, masked, covariance, kwargs
    ):
        values = np.ones((8, 2, 3))
        args = [values, np.repeat([0, 1], 4)]
        fn = compare_two_groups
        if masked:
            fn = compare_two_groups_masked
            args.append(np.ones((8, 2), bool))
        with pytest.raises(ValueError, match="covariance"):
            fn(*args, covariance=covariance, **kwargs)

    @pytest.mark.parametrize("glm", [False, True])
    @pytest.mark.parametrize("masked", [False, True])
    def test_covariance_choice_validation_default_and_storage(self, glm, masked):
        from sonic.comparators import multisample as ms

        rng = np.random.default_rng(62)
        values = rng.lognormal(size=(8, 3, 19))
        groups = np.repeat([0, 1], 4)
        args = [values, pd.DataFrame({"group": groups}), "group"] if glm else [values, groups]
        if masked:
            args.append(np.ones((8, 3), bool))
        suffix = "_masked" if masked else ""
        fn = getattr(ms, "compare_" + ("glm" if glm else "two_groups") + suffix)
        with pytest.raises(ValueError, match="covariance must be"):
            fn(*args, covariance="typo")
        pd.testing.assert_frame_equal(fn(*args), fn(*args, null="rotation"))
        with pytest.raises(ValueError, match="plugin"):
            fn(*args, null="analytic")
        plugin = fn(*args, null="plugin")
        pd.testing.assert_frame_equal(plugin, fn(*args, covariance="per_gene"))
        assert plugin.attrs["null"] == "plugin"
        assert plugin.attrs["covariance"] == "per_gene"
        estimate = getattr(
            l2, "_estimate_" + ("glm" if glm else "two_group") + suffix + "_null_covariance"
        )
        state = estimate(*args, covariance="per_gene", store_covariance=False)
        assert state["sigma_log"] is None
        assert state["weighted_cov"] is None
        with patch.object(np.linalg, "eigvalsh", wraps=np.linalg.eigvalsh) as eig:
            fn(*args, covariance="per_gene")
        assert all(call.args[0].shape == (8, 8) for call in eig.call_args_list)

    @pytest.mark.parametrize("covariance", ["pooled", "per_gene"])
    @pytest.mark.parametrize("sizes", [(2, 6), (6, 2), (3, 5)])
    @pytest.mark.parametrize("profile_scale", ["df", "df_norm", "signed"])
    def test_binary_glm_adapter_preserves_null_and_metadata(self, covariance, sizes, profile_scale):

        rng = np.random.default_rng(25)
        n = sum(sizes)
        groups = np.repeat(["control", "treated"], sizes)
        indicator = (groups == "treated").astype(float)
        values = rng.normal(size=(n, 4, 17))
        values[:, 0] = 0.0
        values[:, 1] = indicator[:, None] * np.linspace(0, 1, 17)
        values[:, 2] *= 1e-10
        spectra = values if profile_scale == "signed" else np.exp(values)
        original = spectra.copy()
        spectra.setflags(write=False)
        kwargs = {
            "covariance": covariance,
            "freq_weights": np.arange(1, 18),
            "normalize_shape": profile_scale == "df_norm",
            "log_transform": profile_scale != "signed",
        }
        binary = l2._estimate_two_group_null_covariance(spectra, groups, **kwargs)
        glm = l2._estimate_glm_null_covariance(
            spectra, np.column_stack([np.ones(n), indicator]), np.array([0.0, 1.0]), **kwargs
        )
        for key in (
            "observed",
            "beta",
            "sigma_log",
            "weighted_cov",
            "eigenvalues",
            "effective_rank",
        ):
            np.testing.assert_allclose(binary[key], glm[key], atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            l2._l2_pvalues_from_state(binary),
            l2._l2_pvalues_from_state(glm),
            atol=1e-12,
            equal_nan=True,
        )
        assert binary["mode"] == "two_group"
        assert not binary["masked"]
        assert (binary["n_obs_A"], binary["n_obs_B"]) == sizes
        assert binary["df_resid"] == n - 2
        assert binary["contrast_scale"] == pytest.approx(1 / sizes[0] + 1 / sizes[1])
        assert {"design_columns", "contrast_vector", "n_obs"}.isdisjoint(binary)
        assert binary["beta"].shape == (2, 4, 17)
        np.testing.assert_array_equal(spectra, original)

    @pytest.mark.parametrize("covariance", ["pooled", "per_gene"])
    @pytest.mark.parametrize("minimum", [2, 3])
    @pytest.mark.parametrize("profile_scale", ["df", "df_norm", "signed"])
    def test_masked_binary_glm_adapter_enforces_each_arm(self, covariance, minimum, profile_scale):

        groups = np.repeat([0, 1], 6)
        n_a = np.array([6, 3, 2, 1, 4, 6, 0, 0, 1, 3])
        n_b = np.array([6, 3, 4, 5, 1, 0, 6, 0, 1, 3])
        presence = np.concatenate([np.arange(6)[:, None] < n_a, np.arange(6)[:, None] < n_b])
        presence[:, -1] = presence[::-1, -1]  # Same counts, different observed rows.
        eligible = (n_a >= minimum) & (n_b >= minimum)
        values = np.random.default_rng(27).normal(size=(12, len(n_a), 17))
        spectra = values if profile_scale == "signed" else np.exp(values)
        spectra[~presence] = np.nan
        # Some rejected contrasts are estimable; they must still never enter fitting or pooling.
        spectra[:, ~eligible] = np.nan
        original = spectra.copy()
        spectra.setflags(write=False)
        presence.setflags(write=False)
        kwargs = {
            "covariance": covariance,
            "freq_weights": np.arange(1, 18),
            "normalize_shape": profile_scale == "df_norm",
            "log_transform": profile_scale != "signed",
        }
        binary = l2._estimate_two_group_masked_null_covariance(
            spectra, groups, presence, min_samples_per_group=minimum, **kwargs
        )
        glm = l2._estimate_glm_masked_null_covariance(
            spectra[:, eligible],
            np.column_stack([np.ones(12), groups]),
            np.array([0.0, 1.0]),
            presence[:, eligible],
            **kwargs,
        )
        assert binary["beta"].shape == (2, len(n_a), 17)
        np.testing.assert_allclose(binary["beta"][:, eligible], glm["beta"], atol=1e-12)
        assert np.isnan(binary["beta"][:, ~eligible]).all()
        for key in ("observed", "eigenvalues", "contrast_scale"):
            np.testing.assert_allclose(binary[key][eligible], glm[key], atol=1e-12)
            assert np.isnan(binary[key][~eligible]).all()
        for key in ("sigma_log", "weighted_cov", "effective_rank"):
            actual = binary[key][eligible] if covariance == "per_gene" else binary[key]
            np.testing.assert_allclose(actual, glm[key], atol=1e-12)
        pvalues = l2._l2_pvalues_from_state(binary)
        np.testing.assert_allclose(pvalues[eligible], l2._l2_pvalues_from_state(glm), atol=1e-12)
        assert np.isnan(pvalues[~eligible]).all()
        assert binary["mode"] == "two_group" and binary["masked"]
        assert {"design_columns", "contrast_vector", "n_obs"}.isdisjoint(binary)
        np.testing.assert_array_equal(binary["eligible"], eligible)
        np.testing.assert_array_equal(binary["n_obs_A"], n_a)
        np.testing.assert_array_equal(binary["n_obs_B"], n_b)
        np.testing.assert_array_equal(binary["df_resid"], np.maximum(n_a + n_b - 2, 1))
        np.testing.assert_allclose(
            binary["contrast_scale"][eligible], 1 / n_a[eligible] + 1 / n_b[eligible]
        )
        np.testing.assert_array_equal(spectra, original)

    @pytest.mark.parametrize("covariance", ["pooled", "per_gene"])
    @pytest.mark.parametrize("store_covariance", [False, True])
    def test_masked_binary_glm_adapter_preserves_empty_pool(self, covariance, store_covariance):

        groups = np.repeat([0, 1], 4)
        presence = np.zeros((8, 3), bool)
        presence[:4, 1:] = True
        presence[4, 2] = True  # Estimable with residual df, but only one observation in arm B.
        spectra = np.full((8, 3, 5), np.nan)
        kwargs = {"covariance": covariance, "store_covariance": store_covariance}
        with pytest.raises(ValueError, match="no genes meet min_samples_per_group=2 per arm"):
            l2._estimate_two_group_masked_null_covariance(spectra, groups, presence, **kwargs)
        state = l2._estimate_two_group_masked_null_covariance(
            spectra, groups, presence, allow_empty=True, **kwargs
        )
        assert not state["eligible"].any()
        assert state["beta"].shape == (2, 3, 5)
        for key in ("observed", "beta", "contrast_scale", "eigenvalues"):
            assert np.isnan(state[key]).all()
        assert np.isnan(l2._l2_pvalues_from_state(state)).all()
        np.testing.assert_array_equal(state["n_obs_A"], [0, 4, 4])
        np.testing.assert_array_equal(state["n_obs_B"], [0, 0, 1])
        np.testing.assert_array_equal(state["df_resid"], [1, 2, 3])
        for key in ("sigma_log", "weighted_cov"):
            if covariance == "pooled":
                np.testing.assert_array_equal(state[key], np.zeros((5, 5)))
            elif store_covariance:
                assert np.isnan(state[key]).all()
            else:
                assert state[key] is None
        with pytest.raises(ValueError, match="freq_weights"):
            l2._estimate_two_group_masked_null_covariance(
                spectra, groups, presence, freq_weights=-np.ones(5), allow_empty=True, **kwargs
            )

    def test_plugin_null_metadata_stays_optional_on_python_310(
        self,
    ):
        assert _PluginNullState.__optional_keys__ == {
            "estimate_per_bin",
            "n_obs_A",
            "n_obs_B",
            "design_columns",
            "contrast_vector",
            "n_obs",
            "zero_covariance",
        }
        assert {
            "observed",
            "beta",
            "eigenvalues",
            "eligible",
        } <= _PluginNullState.__required_keys__

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("glm", [False, True])
    @pytest.mark.parametrize("shape_only", [False, True])
    @pytest.mark.parametrize("covariance_mode", ["pooled", "per_gene"])
    @pytest.mark.parametrize("n_bins", [7, 17])
    def test_blocked_covariance_matches_independent_per_gene_fits(  # noqa: C901
        self,
        monkeypatch,
        masked,
        glm,
        shape_only,
        covariance_mode,
        n_bins,
    ):
        from sonic.comparators import multisample as ms

        rng = np.random.default_rng(42)
        spectra = rng.lognormal(size=(12, 23, n_bins)).astype(np.float32)
        labels = np.tile([0, 1], 6)
        design = np.column_stack([np.ones(12), labels, rng.normal(size=12)])
        contrast = np.array([0.0, 1.0, 0.0])
        weights = np.arange(1, n_bins + 1, dtype=float)
        weights /= weights.sum()
        presence = np.ones((12, 23), bool)
        if masked:
            presence[0, :5] = False
            presence[1, 5:10] = False  # Same count, different design.
            presence[:, 17:19] = False
            presence[3:, 19:21] = False  # No GLM residual df; too few in each arm.
            presence[labels == 1, 21:] = False  # Contrast not estimable.
            spectra[~presence] = np.nan
        original = spectra.copy()
        spectra.setflags(write=False)
        observed = np.full(23, np.nan)
        scales = np.full(23, np.nan)
        beta = np.full((3 if glm else 2, 23, n_bins), np.nan)
        covariance = np.zeros((n_bins, n_bins))
        gene_covariances = np.full((23, n_bins, n_bins), np.nan)
        pooled_df = 0
        for j in range(23):
            keep = presence[:, j]
            x = design[keep]
            if not len(x):
                continue
            values = spectra[keep, j].astype(float)
            if shape_only:
                values = normalize_shape(values)
            logs = np.log(np.maximum(values, 1e-12))
            if glm:
                fit, _, rank, _ = np.linalg.lstsq(x, logs, rcond=None)
                inverse = np.linalg.pinv(x)
                if len(x) <= rank or not np.allclose(contrast @ inverse @ x, contrast):
                    continue
                beta[:, j] = fit
                effect = contrast @ fit
                residuals = logs - x @ fit
                scale = np.sum((contrast @ inverse) ** 2)
                df = len(x) - rank
            else:
                groups = labels[keep]
                a, b = groups == 0, groups == 1
                if a.sum() < 2 or b.sum() < 2:
                    continue
                ma, mb = logs[a].mean(axis=0), logs[b].mean(axis=0)
                beta[:, j] = [ma, mb - ma]
                effect = ma - mb
                residuals = logs - np.where(a[:, None], ma, mb)
                scale = 1 / a.sum() + 1 / b.sum()
                df = len(x) - 2
            observed[j] = np.sqrt(np.sum(weights * effect**2))
            scales[j] = scale
            gene_covariances[j] = residuals.T @ residuals / df
            covariance += residuals.T @ residuals
            pooled_df += df
        covariance /= pooled_df
        weighted = np.sqrt(weights[:, None] * weights) * covariance
        eigenvalues = np.maximum(np.linalg.eigvalsh(weighted), 0)
        if covariance_mode == "per_gene":
            covariance = gene_covariances
            eigenvalues = np.full((23, n_bins), np.nan)
            for j in np.flatnonzero(np.isfinite(observed)):
                weighted = np.sqrt(weights[:, None] * weights) * covariance[j]
                eigenvalues[j] = np.maximum(np.linalg.eigvalsh(weighted), 0)
        else:
            eigenvalues = np.broadcast_to(eigenvalues, (23, n_bins))
        pvalues = np.array(
            [
                liu_sf(t**2, eig * s) if np.isfinite(t) else np.nan
                for t, s, eig in zip(observed, scales, eigenvalues, strict=True)
            ]
        )

        # Force small blocks so both grouping and within-group chunk boundaries run.
        monkeypatch.setattr(l2, "_COMPARISON_WORKSPACE_BYTES", 8 * 8 * 12 * n_bins * 2)
        name = "_estimate_" + ("glm" if glm else "two_group")
        name += "_masked_null_covariance" if masked else "_null_covariance"
        args = [spectra, design, contrast] if glm else [spectra, labels]
        if masked:
            args.append(presence)
        with patch.object(lm, "_decompose_glm_design", wraps=lm._decompose_glm_design) as decompose:
            with patch.object(
                l2, "_prepare_profile_block", wraps=l2._prepare_profile_block
            ) as blocks:
                state = getattr(l2, name)(
                    *args,
                    freq_weights=weights,
                    normalize_shape=shape_only,
                    covariance=covariance_mode,
                )
                assert max(call.args[0].shape[1] for call in blocks.call_args_list) <= 3
            if glm:
                expected = sum(mask.any() for mask in np.unique(presence.T, axis=0))
                assert decompose.call_count == expected
        np.testing.assert_allclose(state["observed"], observed, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(state["sigma_log"], covariance, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            l2._l2_pvalues_from_state(state), pvalues, atol=1e-12, equal_nan=True
        )
        np.testing.assert_array_equal(state["eligible"], np.isfinite(observed))
        np.testing.assert_allclose(state["beta"], beta, atol=1e-12, equal_nan=True)
        np.testing.assert_array_equal(spectra, original)

        comparison = "compare_" + ("glm" if glm else "two_groups") + ("_masked" if masked else "")
        result = (
            getattr(ms, comparison)(
                *args,
                null="plugin",
                freq_weights=weights,
                normalize_shape=shape_only,
                covariance=covariance_mode,
            )
            .set_index("Feature")
            .loc[[str(i) for i in range(23)]]
        )
        np.testing.assert_allclose(result.P_value, pvalues, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            result.P_adj, apply_bh_correction(pvalues), atol=1e-12, equal_nan=True
        )
        assert result.attrs["covariance"] == covariance_mode
        assert result.attrs["null"] == "plugin"
        monkeypatch.setattr(l2, "_COMPARISON_WORKSPACE_BYTES", 32 * (1 << 20))
        whole = (
            getattr(ms, comparison)(
                *args,
                null="plugin",
                freq_weights=weights,
                normalize_shape=shape_only,
                covariance=covariance_mode,
            )
            .set_index("Feature")
            .loc[result.index]
        )
        pd.testing.assert_frame_equal(whole, result, rtol=1e-7, atol=1e-12)


class TestConditionalRotation:
    """Conditional tails, invariance, eligibility, and deterministic streaming."""

    @pytest.mark.parametrize("shape_only", [False, True], ids=["DF", "DF-norm"])
    def test_constant_companion_bin_retains_student_tail(self, shape_only):
        a = np.array([-1, -1, 1, 1])
        r = np.array([1, -1, -1, 1])
        from scipy.stats import t

        # Two distinct equal-total spectra span one direction even after normalization.
        y = np.where(a + r > 0, 0.5, 0.25)
        expected = 2 * t.sf(1, df=2)  # Pooled two-sample t for (low, low) vs (low, high).
        spectra = np.column_stack([y, np.full(4, 0.25), 1 - y])[:, None]
        for rows in ([0, 1, 2, 3], [0, 2, 1, 3]):
            row = compare_two_groups(
                spectra[rows], a[rows], normalize_shape=shape_only, freq_weights=[1, 2, 1]
            ).iloc[0]
            assert row.P_value_method == "beta"
            np.testing.assert_allclose(row.P_value, expected, rtol=1e-13)

    @pytest.mark.parametrize("shape_only", [False, True], ids=["DF", "DF-norm"])
    def test_nearly_duplicate_bins_preserve_tail_across_sample_orders(self, shape_only):
        a = np.array([-1, -1, 1, 1])
        r = np.array([1, -1, -1, 1])
        p = 0.049
        y = np.exp(0.05 * ((1 - p) / np.sqrt(p * (2 - p)) * a + r) - 1)
        other = y + 1e-12 * a * r
        # A large constant bin keeps the two normalized signal bins nearly proportional.
        spectra = np.column_stack([y, other, np.full(4, 1e7)])[:, None]
        reference = np.log(y / spectra[:, 0].sum(axis=1)) if shape_only else np.log(y)
        from scipy.stats import ttest_ind

        expected = ttest_ind(reference[a < 0], reference[a > 0], equal_var=True).pvalue
        for rows in ([0, 1, 2, 3], [0, 2, 1, 3]):
            row = compare_two_groups(
                spectra[rows], a[rows], normalize_shape=shape_only, freq_weights=[1, 3, 1]
            ).iloc[0]
            assert row.P_value_method == "beta"
            np.testing.assert_allclose(row.P_value, expected, atol=1e-12, rtol=0)
        # These orders formerly disagreed about whether the levels were equal.
        for rows in ([0, 1, 2, 3], [0, 2, 1, 3]):
            with pytest.raises(RuntimeError, match="resolution"):
                compare_two_groups(
                    spectra[rows],
                    a[rows],
                    normalize_shape=shape_only,
                    freq_weights=[1, 3, 1],
                    integration_tol=1e-20,
                )

    @pytest.mark.parametrize(
        "shape_only,low,high",
        [(False, 3.0, 7.0), (True, 3.0, 7.0), (True, 0.3, 0.7), (True, 0.1, 0.3)],
        ids=["DF", "DF-norm", "DF-norm-decimal", "DF-norm-rounded-total"],
    )
    def test_df_rotation_is_invariant_to_bin_order(self, shape_only, low, high):
        a = np.array([-1, -1, 1, 1])
        r = np.array([1, -1, -1, 1])
        design = np.column_stack([np.ones(4), a, a * r])
        weights = np.array([1, 1, 2, 2])
        # Every row has two of each power. The paired bins have equal energy on orthogonal axes.
        tied = np.where(np.column_stack([a, r, -a, -r]) > 0, high, low)
        nearby = tied.copy()
        nearby[0, 0] += 1
        profiles = np.stack([tied, nearby], axis=1)
        # One residual df gives a circle. Its upper tail is the fraction of angles
        # at or above the observed energy along the contrast direction.
        values = nearby
        if shape_only:
            totals = 2 * (low + high) + np.array([1, 0, 0, 0])
            values = nearby / totals[:, None]
        u, v = np.stack([a, r]) @ np.log(values)
        expected = (
            np.arctan2(2 * abs((u * weights) @ v), (u * weights) @ u - (v * weights) @ v) / np.pi
        )
        tails = []
        for order in itertools.permutations(range(4)):
            bins = list(order)
            frame = compare_glm(
                profiles[:, :, bins],
                design,
                np.array([0, 1, 0]),
                normalize_shape=shape_only,
                freq_weights=weights[bins],
            ).set_index("Feature")
            assert frame.loc["0", "P_value"] == 1
            assert frame.loc["0", "P_value_method"] == "exact"
            tails.append(frame.loc["1", "P_value"])
        assert 0 < expected < 1
        np.testing.assert_allclose(tails, expected, rtol=1e-13)

    def test_df_norm_rotation_checks_ties_after_rescaling(self):
        design = np.column_stack([np.ones(4), [-1, 0, 1, 0]])
        contrast = np.array([0, 1])
        tied = np.where(np.eye(4), 1 + 2.0**-51, 0.5)
        scale = np.array([1, 1, 1, 3])[:, None]
        interior_tie = tied.copy()
        perturbation = 2.0**-12
        interior_tie[1, 0] += perturbation
        nearby = interior_tie.copy()
        nearby[0, 1] += 0.0625
        # Compare against log-L2 on shares computed with the known row totals.
        totals = 2.5 + 2.0**-51 + np.array([0.0625, perturbation, 0, 0])
        expected = compare_glm((nearby / totals[:, None])[:, None], design, contrast).P_value.item()
        assert 0 < expected < 1
        for values in (nearby, nearby * scale):
            result = compare_glm(values[:, None], design, contrast, normalize_shape=True)
            assert result.P_value.item() == pytest.approx(expected, rel=1e-12)

        panel = np.stack([nearby, tied], axis=1)
        result = compare_glm(panel, design, contrast, normalize_shape=True).set_index("Feature")
        assert result.loc["1", "P_value"] == 1
        with pytest.raises(RuntimeError, match="normalization roundoff for input feature 1"):
            compare_glm(panel * scale[:, None], design, contrast, normalize_shape=True)

        # At an interior eigenvalue the solver can resolve the tail or flag it explicitly.
        totals = 2.5 + 2.0**-51 + np.array([0, perturbation, 0, 0])
        expected = compare_glm((interior_tie / totals[:, None])[:, None], design, contrast)
        try:
            scaled = compare_glm(
                (interior_tie * scale)[:, None], design, contrast, normalize_shape=True
            )
        except RuntimeError as exc:
            assert "normalization roundoff for input feature 0" in str(exc)
        else:
            assert scaled.P_value.item() == pytest.approx(expected.P_value.item(), abs=1e-6)

    def test_df_norm_rotation_agrees_for_hours_and_nanoseconds(self):
        powers = np.array(
            [[1.0, 0.8, 1.7, 8.0], [0.8, 0.5, 3.3, 0.9], [3.6, 0.3, 6.4, 1.9], [9.9, 0.4, 1.3, 1.0]]
        )[:, None, :]
        hours = np.arange(4) - 1.5

        def hourly_effect(units_per_hour):
            design = np.column_stack([np.ones(4), hours * units_per_hour])
            contrast = np.array([0.0, units_per_hour])
            return compare_glm(powers, design, contrast, normalize_shape=True).iloc[0]

        reference = hourly_effect(1)
        result = hourly_effect(3_600_000_000_000)
        assert 0 < reference.P_value < 0.01
        np.testing.assert_allclose(result.Estimate_per_bin, reference.Estimate_per_bin, rtol=1e-12)
        assert result.Statistic == pytest.approx(reference.Statistic, rel=1e-12)
        assert result.P_value == pytest.approx(reference.P_value, rel=1e-12)

    @pytest.mark.parametrize("masked", [False, True])
    def test_df_norm_timestamp_units_preserve_the_intercept(self, masked):
        powers = np.random.default_rng(325).lognormal(size=(24, 12, 17))
        days = np.arange(24) - 11.5
        ns_per_day = 86_400_000_000_000
        unix_ns = 1_730_000_000_000_000_000 + (days * ns_per_day).astype(np.int64)
        presence = np.ones(powers.shape[:2], dtype=bool)
        presence[:3, 0] = False

        def fit(time):
            design = pd.DataFrame({"time": time})
            if masked:
                frame = compare_glm_masked(powers, design, "time", presence, normalize_shape=True)
                np.testing.assert_array_equal(frame.df_resid, frame.n_obs - 2)
            else:
                frame = compare_glm(powers, design, "time", normalize_shape=True)
            return frame.set_index("Feature").sort_index()

        reference, actual = fit(days), fit(unix_ns)
        np.testing.assert_allclose(actual.P_value, reference.P_value, rtol=1e-9)
        np.testing.assert_allclose(actual.P_adj, reference.P_adj, rtol=1e-9)
        np.testing.assert_allclose(
            np.stack(actual.Estimate_per_bin) * ns_per_day,
            np.stack(reference.Estimate_per_bin),
            rtol=1e-9,
            atol=1e-12,
        )

    def test_conditional_scatter_retains_zero_directions_and_profile_scales(self):
        coordinates = np.random.default_rng(86).normal(size=(8, 1, 2))
        scales = np.array([1e-100, 1.0, 1e100, 0.0])
        coordinates = coordinates * scales[None, :, None]
        original = coordinates.copy()
        scatter, rank = l2._conditional_scatter(coordinates)

        assert scatter.shape == (4, 8, 8)
        np.testing.assert_allclose(scatter[:3], np.repeat(scatter[1:2], 3, axis=0))
        eigenvalues = np.linalg.eigvalsh(scatter)
        assert eigenvalues.shape == (4, 8)
        np.testing.assert_allclose(eigenvalues[:3, :6], 0, atol=1e-14)
        np.testing.assert_allclose(rank[:3], rank[1])
        assert 1 <= rank[1] <= 2
        np.testing.assert_array_equal(scatter[3], 0)
        assert np.isnan(rank[3])
        np.testing.assert_array_equal(coordinates, original)

    @pytest.mark.parametrize("bins", [2, 13])
    @pytest.mark.parametrize("shape_only", [False, True])
    @pytest.mark.parametrize("log_transform", [False, True])
    def test_shared_construction_matches_nuisance_projection(
        self, bins, shape_only, log_transform, monkeypatch
    ):
        rng = np.random.default_rng(87)
        n, genes = 10, 6
        x = np.c_[np.ones(n), np.linspace(-1, 1, n), rng.normal(size=n)]
        c = np.array([0.0, 1.0, 0.0])
        values = rng.lognormal(size=(n, genes, bins))
        values[:, 0] = 1.0
        presence = np.ones((n, genes), bool)
        presence[0, 1:3] = False
        presence[:, 4] = False
        presence[2:, 5] = False  # No residual df for the retained design.
        values[~presence] = np.nan
        original = values.copy()
        values.setflags(write=False)
        weights = np.arange(1, bins + 1, dtype=float)
        weights /= weights.sum()
        expected_statistic = np.full(genes, np.nan)
        seen = []

        def evaluate_tail(scatter, basis, n_rotations, seed):
            rows = np.array(seed.spawn_key[1:])
            mask = np.zeros(n, bool)
            mask[rows] = True
            cols = np.flatnonzero(np.all(presence == mask[:, None], axis=0))
            seen.extend(cols)
            y = values[rows[:, None], cols]
            if shape_only:
                y = normalize_shape(y)
            if log_transform:
                y = np.log(np.maximum(y, 1e-12))
            inverse = np.linalg.pinv(x[rows])
            a = c @ inverse
            dimension = len(rows) - np.linalg.matrix_rank(x[rows]) + 1
            assert scatter.shape == (len(cols), dimension, dimension)
            np.testing.assert_allclose(basis.T @ basis, np.eye(dimension), atol=1e-12)
            projection = np.eye(len(rows)) - x[rows] @ inverse + np.outer(a, a) / (a @ a)
            np.testing.assert_allclose(basis @ basis.T, projection, atol=1e-12)
            np.testing.assert_allclose(basis[:, 0], a / np.linalg.norm(a), atol=1e-12)
            coordinates = np.einsum("sd,sfb->dfb", basis, y) * np.sqrt(weights)
            coordinates[:, cols == 0] = 0  # The constant feature has zero conditional scatter.
            scale = np.max(np.abs(coordinates), axis=(0, 2))
            coordinates /= np.where(scale > 0, scale, 1)[None, :, None]
            expected = np.einsum("ifb,jfb->fij", coordinates, coordinates)
            np.testing.assert_allclose(scatter, expected, atol=1e-12)
            theta = (a @ y.reshape(len(rows), -1)).reshape(len(cols), bins)
            expected_statistic[cols] = np.sqrt(np.sum(weights * theta**2, axis=1))
            return np.where(np.trace(scatter, axis1=1, axis2=2) > 0, 0.25, 1.0)

        monkeypatch.setattr(l2, "_rotation_tails", evaluate_tail)
        state = l2._run_l2_conditional(
            values,
            x,
            c,
            presence,
            n_rotations=1,
            random_state=0,
            freq_weights=weights,
            normalize_shape=shape_only,
            log_transform=log_transform,
        )
        assert sorted(seen) == [0, 1, 2, 3]
        np.testing.assert_array_equal(state["eligible"], [True, True, True, True, False, False])
        np.testing.assert_allclose(state["observed"], expected_statistic, atol=1e-12)
        np.testing.assert_allclose(state["pvals"], [1.0, 0.25, 0.25, 0.25, np.nan, np.nan])
        np.testing.assert_array_equal(state["n_obs"], presence.sum(axis=0))
        np.testing.assert_array_equal(values, original)

    @staticmethod
    def _reference(y, x, c, weights, seed, draws, original_rows, total_samples):
        inverse = np.linalg.pinv(x)
        a = c @ inverse
        scale = a @ a
        theta = c @ inverse @ y
        observed = np.sum(weights * theta**2)
        nuisance = x @ inverse - np.outer(a, a) / scale
        rng = np.random.default_rng(
            np.random.SeedSequence(seed, spawn_key=(total_samples, *map(int, original_rows)))
        )
        directions = rng.normal(size=(draws, len(y)))
        directions -= directions @ nuisance
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        null = np.sum(weights * (directions @ y) ** 2, axis=1)
        scatter_trace = np.sum(weights * (y - nuisance @ y) ** 2)
        count = np.count_nonzero(
            null >= observed / scale - 64 * np.finfo(float).eps * scatter_trace
        )
        return np.sqrt(observed), (count + 1) / (draws + 1)

    @pytest.mark.parametrize(
        "design_kind", ["binary", "continuous", "aliased", "intercept", "no_intercept"]
    )
    @pytest.mark.parametrize("shape_only", [False, True])
    def test_matches_direct_sample_space_reference(self, design_kind, shape_only):
        rng = np.random.default_rng(341)
        n, genes, bins = 10, 4, 7
        groups = np.repeat([0, 1], 5)
        x = np.column_stack([np.ones(n), groups, rng.normal(size=n)])
        c = np.array([0.0, 1.0, 0.0])
        if design_kind == "binary":
            x, c = x[:, :2], c[:2]
        elif design_kind == "aliased":
            x = np.column_stack([x, 2 * groups])
            c = np.array([0.0, 1.0, 0.0, 2.0])
        elif design_kind == "intercept":
            c = np.array([1.0, 0.0, 0.0])
        elif design_kind == "no_intercept":
            x, c = x[:, 1:], c[1:]
        else:
            x[:, 1] = np.linspace(-1, 1, n)
        values = rng.lognormal(size=(n, genes, bins))
        presence = np.ones((n, genes), bool)
        presence[0, 1:3] = False
        presence[:, 3] = False
        values[~presence] = np.nan
        original = values.copy()
        values.setflags(write=False)
        weights = np.arange(1, bins + 1, dtype=float)
        weights /= weights.sum()
        result = l2._run_l2_conditional(
            values,
            x,
            c,
            presence,
            freq_weights=weights,
            normalize_shape=shape_only,
            n_rotations=499,
            random_state=35,
        )
        expected = []
        for gene in range(3):
            rows = np.flatnonzero(presence[:, gene])
            power = values[rows, gene]
            if shape_only:
                power = normalize_shape(power)
            y = np.log(np.maximum(power, 1e-12))
            expected.append(self._reference(y, x[rows], c, weights, 35, 499, rows, n))
        np.testing.assert_allclose(
            np.column_stack([result["observed"][:3], result["pvals"][:3]]), expected, atol=1e-12
        )
        assert np.isnan(result["pvals"][3])
        np.testing.assert_array_equal(values, original)

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("shape_only", [False, True])
    def test_binary_glm_equivalence_and_unchanged_statistic(self, masked, shape_only):
        from sonic.comparators import multisample as ms

        groups = np.repeat([0, 1], 5)
        values = np.random.default_rng(34).lognormal(size=(10, 5, 9))
        presence = np.ones((10, 5), bool)
        if masked:
            presence[0, :2] = False
            presence[:4, -1] = False  # GLM-estimable, but binary threshold fails.
        kwargs = {"freq_weights": np.arange(1, 10), "normalize_shape": shape_only}
        binary = ms.compare_two_groups_masked if masked else ms.compare_two_groups
        args = [values, groups, presence] if masked else [values, groups]
        result = binary(*args, null="rotation", **kwargs)
        plugin = binary(*args, covariance="per_gene", **kwargs)
        pd.testing.assert_series_equal(
            result.set_index("Feature").Statistic.sort_index(),
            plugin.set_index("Feature").Statistic.sort_index(),
        )
        glm = ms.compare_glm_masked if masked else ms.compare_glm
        glm_args = [values, np.c_[np.ones(10), groups], np.array([0.0, 1.0])]
        if masked:
            glm_args.append(presence & (presence[:5].sum(0) >= 2))
        reference = glm(*glm_args, null="rotation", **kwargs)
        np.testing.assert_allclose(
            result[["Statistic", "P_value", "P_adj"]],
            reference[["Statistic", "P_value", "P_adj"]],
            equal_nan=True,
        )
        assert result.attrs == {
            "null": "rotation",
            "covariance": None,
            "method": "davies",
            "integration_tol": 1e-6,
            "integration_limit": 10000,
            "n_saddlepoint": 0,
        }
        if masked:
            assert result.set_index("Feature").loc["4", "n_obs_A"] == 1

    @pytest.mark.parametrize("n", [4, 12])
    def test_rank_one_matches_exact_t_and_beta(self, n):
        from scipy.stats import beta, t

        rng = np.random.default_rng(19)
        x = np.c_[np.ones(n), np.linspace(-1, 1, n)]
        y = rng.normal(size=(n, 24)) + x[:, 1, None] * np.linspace(0, 2, 24)
        inverse = np.linalg.pinv(x)
        coefficients = inverse @ y
        residuals = y - x @ coefficients
        df = n - 2
        q = coefficients[1] ** 2 / np.sum(inverse[1] ** 2)
        ss = np.sum(residuals**2, axis=0)
        exact = 2 * t.sf(np.sqrt(q * df / ss), df)
        np.testing.assert_allclose(exact, beta.sf(q / (q + ss), 0.5, df / 2))
        result = compare_glm(
            np.exp(y[..., None]),
            x,
            np.array([0.0, 1.0]),
            null="rotation",
            gene_names=list(map(str, range(24))),
        )
        actual = result.set_index("Feature").loc[list(map(str, range(24))), "P_value"].to_numpy()
        np.testing.assert_allclose(actual, exact, rtol=0, atol=1e-10)

    def test_nuisance_units_and_contrast_scaling(self):
        rng = np.random.default_rng(28)
        x = np.c_[np.ones(14), np.linspace(-1, 1, 14), rng.normal(size=14)]
        c = np.array([0.0, 1.0, 0.0])
        y = rng.normal(size=(14, 6, 9))
        kwargs = {"null": "rotation"}
        reference = compare_glm(np.exp(y), x, c, **kwargs).set_index("Feature").sort_index()
        nuisance = rng.normal(size=(3, 54))
        nuisance[1] = 0
        shifted_y = y + (x @ nuisance).reshape(y.shape)
        shifted_x = x.copy()
        shifted_x[:, 2] = 10000 + 100 * x[:, 2]
        actual = (
            compare_glm(np.exp(shifted_y), shifted_x, c, **kwargs).set_index("Feature").sort_index()
        )
        pd.testing.assert_frame_equal(actual, reference, check_exact=False, atol=1e-10)
        for scale in (-3.0, 1e-12, 1e12):
            actual = (
                compare_glm(np.exp(y), x, c * scale, **kwargs).set_index("Feature").sort_index()
            )
            np.testing.assert_allclose(
                actual.Statistic, reference.Statistic * abs(scale), rtol=1e-10
            )
            np.testing.assert_allclose(actual.P_value, reference.P_value, rtol=0, atol=1e-10)

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("contrast_scale", [1.0, 1e-12])
    def test_nuisance_only_profiles_and_nonzero_perfect_fits(self, masked, contrast_scale):
        rng = np.random.default_rng(1)
        z = rng.integers(-8, 9, size=10) / 8
        x = np.c_[np.ones(10), np.arange(10) / 8, z]
        c = np.array([0.0, contrast_scale, 0.0])
        y = np.repeat((3 + z)[:, None, None] * np.array([1.0, 0.5, -0.25]), 3, axis=1)
        y[:, 1] += x[:, 1, None] * np.array([0.75, -0.25, 0.5])
        y[:, 2] += rng.normal(scale=0.1, size=(10, 3))
        presence = np.ones((10, 3), bool)
        if masked:
            presence[0, 0] = False
            y[~presence] = np.nan
        kwargs = {
            "log_transform": False,
            "n_rotations": 999,
            "random_state": 20,
            "freq_weights": np.array([0.7, 0.2, 0.1]),
        }
        result = l2._run_l2_conditional(y, x, c, presence if masked else None, **kwargs)
        assert result["pvals"][0] == 1.0
        assert result["pvals"][1] == 0.001
        assert result["observed"][1] > 0
        reference = l2._run_l2_conditional(y[:, 2:3], x, c, **kwargs)
        assert result["pvals"][2] == reference["pvals"][0]

    def test_roundoff_guard_preserves_tiny_and_large_profile_scales(self):

        rng = np.random.default_rng(52)
        x = np.c_[np.ones(10), np.linspace(-1, 1, 10), rng.normal(size=10)]
        y = rng.normal(size=(10, 1, 4)) + x[:, 1, None, None]
        scales = np.array([1e-100, 1.0, 1e100])
        profiles = y * scales[None, :, None]
        result = l2._run_l2_conditional(
            profiles,
            x,
            np.array([0.0, 1.0, 0.0]),
            log_transform=False,
            freq_weights=np.array([0.2, 0.3, 0.4, 0.1]),
            n_rotations=999,
            random_state=20,
        )
        np.testing.assert_array_equal(result["pvals"], np.repeat(result["pvals"][1], 3))
        assert result["pvals"][1] < 1
        np.testing.assert_allclose(result["observed"] / scales, result["observed"][1])
        assert np.isfinite(result["effective_rank"]).all()

    def test_chunking_and_feature_order(self, monkeypatch):

        rng = np.random.default_rng(51)
        x = np.c_[np.ones(10), np.linspace(-1, 1, 10)]
        values = rng.lognormal(size=(10, 9, 13))
        values[:, 1] = values[:, 0]
        presence = np.ones((10, 9), bool)
        presence[0, 3:7] = False
        kwargs = {
            "null": "rotation",
            "gene_names": list(map(str, range(9))),
        }

        def run(v, mask, names):
            return (
                compare_glm_masked(
                    v, x, np.array([0.0, 1.0]), mask, **(kwargs | {"gene_names": names})
                )
                .set_index("Feature")
                .sort_index()
            )

        expected = run(values, presence, kwargs["gene_names"])
        monkeypatch.setattr(l2, "_COMPARISON_WORKSPACE_BYTES", 4096)
        actual = run(values, presence, kwargs["gene_names"])
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, atol=1e-12)
        np.testing.assert_allclose(actual.P_value, expected.P_value, rtol=0, atol=1e-10)
        assert actual.loc["0", "P_value"] == actual.loc["1", "P_value"]
        order = rng.permutation(9)
        reordered = run(values[:, order], presence[:, order], [str(i) for i in order])
        pd.testing.assert_frame_equal(reordered, expected, check_exact=False, atol=1e-12)

    def test_constant_ties_and_perfect_fit(self):
        groups = np.repeat([0, 1], 3)
        x = np.c_[np.ones(6), groups]
        a = np.linalg.pinv(x)[1]
        basis = np.c_[a / np.linalg.norm(a), np.linalg.svd(x, full_matrices=True)[0][:, 2:]]
        y = np.zeros((6, 3, 5))
        y[:, 0] = 4.2
        y[:, 1, :5] = groups[:, None]
        y[:, 2, :5] = np.sqrt(5) * basis
        result = l2._run_l2_conditional(
            np.exp(y),
            x,
            np.array([0.0, 1.0]),
            n_rotations=199,
            random_state=17,
        )
        np.testing.assert_array_equal(result["pvals"], [1.0, 0.005, 1.0])
        low_df = l2._run_l2_conditional(
            np.exp(y[:3]),
            np.c_[np.ones(3), [0, 1, 1]],
            np.array([0.0, 1.0]),
            n_rotations=19,
            random_state=1,
        )
        assert ((low_df["pvals"] >= 0.05) & (low_df["pvals"] <= 1)).all()

    def test_generated_seed_can_be_replayed(self):
        values = np.random.default_rng(14).lognormal(size=(8, 3, 5))
        x = np.c_[np.ones(8), np.repeat([0, 1], 4)]
        c = np.array([0.0, 1.0])
        result = l2._run_l2_conditional(values, x, c, n_rotations=99)
        replay = l2._run_l2_conditional(
            values, x, c, n_rotations=99, random_state=result["metadata"]["random_state"]
        )
        for key in result.keys() - {"metadata"}:
            np.testing.assert_array_equal(result[key], replay[key])
        assert result["metadata"] == replay["metadata"]

    @pytest.mark.parametrize("draws", [0, -1, 2.5, True])
    def test_invalid_draw_count(self, draws):
        with pytest.raises(ValueError, match="n_rotations must be a positive integer"):
            l2._run_l2_conditional(
                np.ones((6, 2, 3)),
                np.c_[np.ones(6), np.repeat([0, 1], 3)],
                np.array([0.0, 1.0]),
                n_rotations=draws,
            )

    @pytest.mark.parametrize("covariance", ["pooled", "per_gene"])
    @pytest.mark.parametrize("glm", [False, True])
    @pytest.mark.parametrize("masked", [False, True])
    def test_explicit_covariance_is_rejected(self, covariance, glm, masked):
        from sonic.comparators import multisample as ms

        values = np.ones((6, 2, 3))
        groups = np.repeat([0, 1], 3)
        name = "compare_" + ("glm" if glm else "two_groups") + ("_masked" if masked else "")
        args = (
            [values, np.c_[np.ones(6), groups], np.array([0.0, 1.0])] if glm else [values, groups]
        )
        if masked:
            args.append(np.ones((6, 2), bool))
        with pytest.raises(ValueError, match="covariance is only supported"):
            getattr(ms, name)(*args, null="rotation", covariance=covariance)

    def test_inapplicable_statistic_design_and_observations(self):
        values = np.ones((6, 2, 3))
        groups = np.repeat([0, 1], 3)
        with pytest.raises(ValueError, match="requires null='analytic'"):
            compare_two_groups(values, groups, null="rotation", statistic="log_t_cauchy")
        with pytest.raises(ValueError, match="residual degrees"):
            compare_glm(values, np.eye(6), np.ones(6), null="rotation")
        with pytest.raises(ValueError, match="not estimable"):
            compare_glm(
                values, np.c_[np.ones(6), np.ones(6)], np.array([1.0, 0.0]), null="rotation"
            )
        with pytest.raises(ValueError, match="finite profiles"):
            compare_two_groups(np.full_like(values, np.nan), groups, null="rotation")
        with pytest.raises(ValueError, match="no features"):
            compare_two_groups_masked(values, groups, np.zeros((6, 2), bool), null="rotation")


class TestConditionalIntegration:
    """Synthetic accuracy gates for the private deterministic rotation evaluator."""

    @pytest.mark.parametrize("normalize", [False, True], ids=["DF", "DF-norm"])
    def test_exact_fit_next_to_noisy_feature(self, normalize):
        groups = np.tile([0, 1], 6)
        exact = np.array([[1.0, 2.0], [2.0, 1.0]])[groups]
        noisy = np.random.default_rng(92).uniform(1, 2, size=exact.shape)
        spectra = np.stack([exact, noisy], axis=1)
        options = {"normalize_shape": normalize, "freq_weights": [1, 2]}
        result = compare_two_groups(spectra, groups, **options).set_index("Feature")
        alone = compare_two_groups(spectra[:, 1:], groups, **options).iloc[0]
        assert result.loc["0", "P_value"] == result.loc["0", "P_value_error"] == 0
        assert result.loc["0", "P_value_method"] == "exact"
        assert 0 < alone.P_value < 1
        assert result.loc["1", "P_value"] == pytest.approx(alone.P_value, rel=1e-12)

    def test_continuous_exact_fit_and_one_ulp_residual(self):
        from scipy.stats import t

        x = np.arange(8.0)
        design = np.c_[np.ones(8), x]
        response = np.tile((1 + x / 8)[:, None, None], (1, 2, 1))
        response[1, 1, 0] = np.nextafter(response[1, 1, 0], np.inf)
        result = l2._run_l2_conditional(response, design, np.array([0, 1]), log_transform=False)
        assert result["pvals"][0] == result["pvalue_error"][0] == 0
        assert result["pvalue_method"].tolist() == ["exact", "beta"]
        # A perturbation at sample 1 leaves RSS = delta² (1 - leverage[1]).
        delta = np.spacing(1 + x[1] / 8)
        centered = x[1] - x.mean()
        sxx = np.sum((x - x.mean()) ** 2)
        leverage = 1 / len(x) + centered**2 / sxx
        slope = 1 / 8 + delta * centered / sxx
        se = delta * np.sqrt((1 - leverage) / (6 * sxx))
        np.testing.assert_allclose(result["pvals"][1], 2 * t.sf(slope / se, 6), rtol=1e-12, atol=0)

    @pytest.mark.parametrize("aliased", [False, True], ids=["truncated", "aliased"])
    def test_refinement_respects_retained_rank(self, aliased):
        from scipy.stats import t

        g = np.tile([-1, 1], 6)
        noise = np.tile([-1, -1, 1, 1], 3)
        # The third column is either exactly or numerically collinear with the tested term.
        x = np.c_[np.ones(12), g, 2 * g if aliased else g + noise * 2.0**-51]
        c = np.array([0, 1, 2 if aliased else 1])
        y = 0.5 + np.spacing(0.5) * (3 * g + noise)
        result = l2._run_l2_conditional(y[:, None, None], x, c, log_transform=False)
        assert result["df_resid"][0] == 10
        expected = 2 * t.sf(3 * np.sqrt(10), 10)
        np.testing.assert_allclose(result["pvals"], expected, rtol=1e-12, atol=0)

    def test_refined_positive_residual_energy_underflow_is_explicit(self):
        x = np.c_[np.ones(4), [0, 0, 1, 1]]
        y = np.array([1e-200, 0, 1, 1])[:, None, None]
        with pytest.raises(RuntimeError, match="cannot resolve a nonzero contrast or residual"):
            l2._run_l2_conditional(y, x, np.array([0, 1]), log_transform=False)

    @pytest.mark.parametrize(
        "dimension,rank,q",
        [(2, 1, 0.2), (7, 3, 0.5), (7, 6, 0.9), (3, 1, 1 - 1e-8)],
    )
    def test_two_level_spectra_match_exact_beta(self, dimension, rank, q):
        from scipy.stats import beta

        eigenvalues = np.r_[np.ones(rank), np.zeros(dimension - rank)]
        expected = beta.sf(q, rank / 2, (dimension - rank) / 2)
        actual, error, method = l2._conditional_tail(eigenvalues - q)
        assert abs(actual - expected) <= error
        assert error <= min(1e-10, 0.001 * actual)

    def test_zero_shifted_coefficients_cancel_from_the_sign(self):
        actual, error, method = l2._conditional_tail([2.0, 2.0, -3.0, -3.0, 0.0, 0.0])
        assert abs(actual - 0.4) <= error

    @pytest.mark.parametrize("df,score", [(1, 3.0), (10, 1.0), (30, 1e4)])
    def test_rank_one_matches_student_t_including_extreme_tails(self, df, score):
        from scipy.stats import t

        actual, error, method = l2._conditional_tail(np.r_[df, np.full(df, -(score**2))])
        expected = 2 * t.sf(score, df)
        assert abs(actual - expected) <= error
        assert error <= min(1e-10, 0.001 * actual)

    @staticmethod
    def _three_level_reference(levels, counts, q):
        """Positive beta-conditioning integral, independent of Imhof inversion."""
        from scipy.integrate import quad
        from scipy.special import beta, betainc

        lo, middle, hi = levels
        k0, k1, k2 = counts
        fraction = (middle - lo) / (hi - lo)
        gap = (hi - q) / (hi - lo)

        def integrand(v):
            return betainc((k0 + k1) / 2, k2 / 2, gap / (1 - fraction * v))

        value, error = quad(
            integrand,
            0,
            1,
            weight="alg",
            wvar=(k1 / 2 - 1, k0 / 2 - 1),
            epsabs=1e-300,
            epsrel=2e-12,
            limit=1000,
        )
        return value / beta(k1 / 2, k0 / 2), error / beta(k1 / 2, k0 / 2)

    @pytest.mark.parametrize(
        "counts,levels,fraction,method",
        [
            pytest.param((1, 1, 1), (0.0, 0.2, 1.0), 0.25, "saddlepoint", id="low-df"),
            pytest.param((5, 7, 5), (0.0, 0.2, 1.0), 0.9, "davies", id="repeated-levels"),
            pytest.param((1, 1, 1), (0.0, 1e-10, 1.0), 1 - 1e-8, "beta", id="small-tail"),
            pytest.param(
                (2, 2, 3), (1.0, 1 + 2e-10, 1 + 1e-9), 0.25, "davies", id="nearly-constant"
            ),
            pytest.param((2, 2, 3), (0.0, 0.2, 1.0), 1 - 1e-8, "saddlepoint", id="unresolved-tail"),
        ],
    )
    def test_three_levels_match_independent_integral(self, counts, levels, fraction, method):
        q = levels[1] + fraction * (levels[2] - levels[1])
        expected, reference_error = self._three_level_reference(levels, counts, q)
        coefficients = np.repeat(levels, counts) - q
        actual, error, used = l2._conditional_tail(coefficients)
        assert used == method
        if method == "saddlepoint":
            assert np.isnan(error)
            assert actual > 0 and abs(actual / expected - 1) < 0.2
        else:
            assert abs(actual - expected) <= error + reference_error
            assert error <= min(1e-6, 0.001 * actual)

    def test_beta_bounds_require_resolved_probability_width(self):
        levels = np.array([0, 1e-5, 1.0])
        q = 0.9
        reference, reference_error = self._three_level_reference(levels, (1, 1, 1), q)
        for sign, expected in [(1, reference), (-1, 1 - reference)]:
            coefficients = sign * (levels - q)
            p, error, method = l2._conditional_tail(coefficients)
            assert method == "beta" and 1e-9 < error <= min(1e-6, 0.001 * p)
            assert abs(p - expected) <= error + reference_error
            assert l2._conditional_tail(coefficients, integration_tol=1e-10)[2] != "beta"

    @pytest.mark.parametrize(
        "coefficients,reference,integral_error",
        [
            ([-0.7, -0.5, 0.3], 0.184938448031828, 5.46229728115577e-14),
            ([-3, -1, 0.2, 0.8, 2], 0.449828725949695, 2.42861286636753e-14),
            ([-5, -2, -0.7, 0.1, 1, 4, 8], 0.646938782053403, 2.78495619566519e-14),
        ],
    )
    def test_matches_compquadform_reference(self, coefficients, reference, integral_error):
        # CompQuadForm 1.4.4: imhof(0, lambda=a/max(abs(a)), epsabs=1e-13,
        # epsrel=0, limit=10000); its reported abserr is on the integral scale.
        # https://github.com/cran/CompQuadForm/blob/master/src/imhof.cpp
        actual, error, method = l2._conditional_tail(coefficients)
        if len(coefficients) == 3:
            assert method == "saddlepoint" and np.isnan(error)
            assert abs(actual / reference - 1) < 0.04
        else:
            assert method == "davies"
            assert abs(actual - reference) <= error + integral_error / np.pi

    def test_beta_complement_avoids_rounding_argument_to_one(self):
        expected = 1 - 2 * np.arctan(1e-10) / np.pi
        actual, error, method = l2._conditional_tail([1.0, -1e-20])
        assert abs(actual - expected) <= error
        assert actual < 1

    @pytest.mark.parametrize(
        "coefficients,expected",
        [
            ([0.0], 1.0),
            ([0.0, 0.0], 1.0),
            ([0.0, 2.0], 1.0),
            ([-2.0, 0.0], 0.0),
            ([-2.0, -1.0], 0.0),
        ],
    )
    def test_exact_sign_boundaries(self, coefficients, expected):
        assert l2._conditional_tail(coefficients) == (expected, 0.0, "exact")

    def test_constant_and_nearly_constant_scatter(self):
        matrices = np.array([np.zeros((3, 3)), np.eye(3), np.diag([1 + 1e-14, 1, 1])])
        pvalues, errors, methods, saddlepoint_ties = l2._conditional_tails(matrices)
        np.testing.assert_array_equal(pvalues, [1.0, 1.0, 0.0])
        np.testing.assert_array_equal(errors, 0.0)
        assert not saddlepoint_ties.any()
        # The exact upper endpoint differs from the legacy Monte Carlo tie tolerance.
        mc = l2._rotation_tails(matrices[-1:], np.eye(3), 199, np.random.SeedSequence(5))
        assert mc[0] == 1.0

    def test_scale_shift_and_eigenvalue_order_invariance(self):
        rng = np.random.default_rng(721)
        eigenvalues = rng.lognormal(size=7)
        q = eigenvalues.mean()
        expected, _, _ = l2._conditional_tail(eigenvalues - q)
        for scale, shift in ((1e-200, 0), (1e200, 0), (3.0, 7.0)):
            actual, _, _ = l2._conditional_tail(scale * eigenvalues + shift - (scale * q + shift))
            assert abs(actual - expected) < 1e-10
        actual, _, _ = l2._conditional_tail(rng.permutation(eigenvalues) - q)
        assert abs(actual - expected) < 1e-12

    @pytest.mark.parametrize(
        "p,fault", [(0.2, 1), (0.2, 2), (np.nan, 0), (0.0, 0), (-0.1, 0), (1.1, 0)]
    )
    def test_failed_davies_is_marked_without_an_accuracy_retry(self, p, fault, monkeypatch):
        calls = []

        def fail(a, accuracy, limit):
            calls.append((accuracy, limit))
            return p, fault

        monkeypatch.setattr(l2._davies, "sf", fail)
        actual, error, method = l2._conditional_tail([-3, -1, 0.2, 0.8, 2])
        assert 0 < actual < 1
        assert np.isnan(error) and method == "saddlepoint"
        assert calls == [(1e-6, 10000)]

    @pytest.mark.parametrize("second", [(1e-5, 0), (1e-20, 0), (0.2, 1)])
    def test_one_accuracy_retry_is_bounded_and_marked(self, second, monkeypatch):
        calls = []

        def integrate(a, accuracy, limit):
            calls.append((accuracy, limit))
            return (1e-5, 0) if len(calls) == 1 else second

        monkeypatch.setattr(l2._davies, "sf", integrate)
        actual, error, method = l2._conditional_tail([-3, -1, 0.2, 0.8, 2])
        assert len(calls) == 2
        np.testing.assert_allclose(calls, [(1e-6, 10000), (1e-9, 10000)])
        if second == (1e-5, 0):
            assert actual == 1e-5 and error == pytest.approx(1e-9) and method == "davies"
        else:
            assert 0 < actual < 1 and np.isnan(error) and method == "saddlepoint"

    def test_exact_cases_bypass_native_and_saddlepoint(self, monkeypatch):
        def fail(*args):
            raise AssertionError("Exact laws must bypass numerical evaluators")

        monkeypatch.setattr(l2._davies, "sf", fail)
        monkeypatch.setattr(l2, "_saddlepoint_tail", fail)
        assert l2._conditional_tail([0.0, 0.0]) == (1.0, 0.0, "exact")
        p, _, method = l2._conditional_tail([2.0, 2.0, -3.0, -3.0])
        assert p == pytest.approx(0.4) and method == "beta"

    def test_saddlepoint_limit_and_unresolved_approximation(self, monkeypatch):
        # Symmetry gives an independent check at the mean, where the formula is singular.
        assert l2._saddlepoint_tail(np.array([-2, -1, 1, 2]) / 2) == 0.5
        a = np.array([-0.7, -0.5, 1.2])
        assert abs(l2._saddlepoint_tail(a + 1e-8) - l2._saddlepoint_tail(a - 1e-8)) < 1e-6
        monkeypatch.setattr(l2._davies, "sf", lambda *args: (0.0, 0))
        monkeypatch.setattr(l2.norm, "sf", lambda _: 0.0)
        with pytest.raises(RuntimeError, match="Davies.*saddlepoint.*resolution"):
            l2._conditional_tail([-3, -1, 0.2, 0.8, 2])

    def test_native_repeated_and_concurrent_calls(self):
        from concurrent.futures import ThreadPoolExecutor

        from sonic._davies import sf

        vectors = [[-3, -1, 0.2, 0.8, 2], [-5, -2, -0.7, 0.1, 1, 4, 8]]
        references = [0.449828725949695, 0.646938782053403]
        expected = [sf(a, 1e-6, 10000) for a in vectors]
        for (p, fault), reference in zip(expected, references, strict=True):
            assert fault == 0 and abs(p - reference) <= 1e-6
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(lambda i: sf(vectors[i % 2], 1e-6, 10000), range(24)))
        assert actual == expected * 12

    def test_saddlepoint_extreme_scale_does_not_underflow_curvature(self):
        # For two negative directions the small-positive limit is a/(2*sqrt(b1*b2)).
        a = 1e-250
        reference = a / (2 * np.sqrt(0.5))
        p, error, method = l2._conditional_tail([a, -1.0, -0.5])
        assert method == "saddlepoint" and np.isnan(error)
        assert p > 0 and abs(p / reference - 1) < 0.3

    @pytest.mark.parametrize(
        "a,accuracy,limit",
        [([], 1e-6, 10000), ([np.nan], 1e-6, 10000), ([1.0], 0.0, 10000), ([1.0], 1e-6, 0)],
    )
    def test_native_input_validation(self, a, accuracy, limit):
        from sonic._davies import sf

        with pytest.raises(ValueError):
            sf(a, accuracy, limit)

    @pytest.mark.parametrize("shape_only", [False, True])
    def test_public_fallback_marks_eligible_rows_and_preserves_bh(self, shape_only, monkeypatch):
        values = np.random.default_rng(792).lognormal(size=(10, 4, 5))
        values[:, 2] = 1  # Exact zero scatter remains exact even with failed Davies.
        presence = np.ones((10, 4), bool)
        presence[:4, 3] = False
        values[~presence] = np.nan
        monkeypatch.setattr(l2._davies, "sf", lambda *args: (0.0, 1))
        result = (
            compare_two_groups_masked(
                values,
                np.repeat([0, 1], 5),
                presence,
                null="rotation",
                normalize_shape=shape_only,
                freq_weights=np.array([1, 2, 1, 3, 4]),
            )
            .set_index("Feature")
            .sort_index()
        )
        assert result.P_value_method.iloc[:3].tolist() == ["saddlepoint", "saddlepoint", "exact"]
        assert pd.isna(result.P_value_method.iloc[3])
        assert result.P_value_error.iloc[:2].isna().all()
        assert result.P_value_error.iloc[2] == 0
        assert np.isnan(result.P_value.iloc[3]) and np.isnan(result.P_value_error.iloc[3])
        assert result.attrs["n_saddlepoint"] == 2
        np.testing.assert_allclose(
            result.P_adj, apply_bh_correction(result.P_value), equal_nan=True
        )

    @pytest.mark.parametrize("tolerance", [True, 0, -1, 1, np.inf, np.nan, "small"])
    def test_invalid_tolerance(self, tolerance):
        with pytest.raises(ValueError, match="integration_tol"):
            l2._conditional_tail([-1.0, 2.0], integration_tol=tolerance)
        with pytest.raises(ValueError, match="integration_tol"):
            compare_two_groups(
                np.ones((6, 2, 3)),
                np.repeat([0, 1], 3),
                null="rotation",
                integration_tol=tolerance,
            )

    @pytest.mark.parametrize("coefficients", [[], [[1.0]], [np.nan], [np.inf]])
    def test_invalid_coefficients(self, coefficients):
        with pytest.raises(ValueError, match="coefficients"):
            l2._conditional_tail(coefficients)

    @pytest.mark.parametrize(
        "coefficients", [[1e-200, -1, -1, -1, -1], [1e-160, -1, -1, -1, -1], [1e-300, -1e300]]
    )
    def test_unresolved_beta_or_dynamic_range_raises(self, coefficients):
        with pytest.raises(RuntimeError, match="resolution|dynamic range"):
            l2._conditional_tail(coefficients)

    def test_too_tight_tolerance_and_unresolved_matrix_boundary(self, monkeypatch):
        p, error, method = l2._conditional_tail([-0.7, -0.5, 0.3], integration_tol=1e-20)
        assert p > 0 and np.isnan(error) and method == "saddlepoint"
        monkeypatch.setattr(np.linalg, "eigvalsh", lambda _: np.array([-1.0, 0.0]))
        with pytest.raises(RuntimeError, match="block feature 0.*boundary"):
            l2._conditional_tails(np.array([[[1.0, 0.1], [0.1, 0.5]]]))

    @pytest.mark.parametrize(
        "n,glm,masked,shape_only",
        [
            pytest.param(4, True, False, False, id="glm-low-df"),
            pytest.param(12, False, False, True, id="binary-DF-norm"),
            pytest.param(12, False, True, False, id="binary-masked"),
            pytest.param(18, True, True, True, id="glm-masked-DF-norm"),
        ],
    )
    def test_shared_profiles_agree_with_rotation(self, n, glm, masked, shape_only):
        rng = np.random.default_rng(723)
        genes, bins = 4, 9
        groups = np.repeat([0, 1], n // 2)
        x = (
            np.c_[np.ones(n), np.linspace(-1, 1, n), rng.normal(size=n)]
            if glm
            else np.c_[np.ones(n), groups]
        )
        c = np.eye(x.shape[1])[1]
        y = rng.normal(size=(n, genes, bins)) + rng.normal(size=(n, genes, 1))
        if glm:
            y += x[:, 2, None, None] * rng.normal(size=(genes, bins))
        values = np.exp(y)
        presence = np.ones((n, genes), bool)
        if masked:
            presence[0, ::3] = False
            presence[-1, 1::3] = False
            presence[:, -1] = False
            values[~presence] = np.nan
            if not glm:
                presence &= (
                    (presence[groups == 0].sum(0) >= 2) & (presence[groups == 1].sum(0) >= 2)
                )[None, :]
        mask = presence if masked else None
        weights = np.arange(1, bins + 1, dtype=float)
        options = {"freq_weights": weights, "normalize_shape": shape_only}
        result = l2._run_l2_conditional(values, x, c, mask, **options)
        draws = 4999
        reference = l2._run_l2_conditional(
            values, x, c, mask, n_rotations=draws, random_state=724, **options
        )
        for key in ("observed", "eligible", "df_resid", "n_obs", "effective_rank"):
            np.testing.assert_array_equal(result[key], reference[key])
        eligible = result["eligible"]
        tails = result["pvals"][eligible]
        errors = result["pvalue_error"][eligible]
        bound = 6 * np.sqrt(tails * (1 - tails) / draws) + 3 / draws
        assert np.all(np.abs(tails - reference["pvals"][eligible]) <= bound)
        assert np.all(np.isnan(errors) | (errors <= np.minimum(1e-6, 0.001 * tails)))
        assert np.isnan(reference["pvalue_error"]).all()
        if masked:
            assert np.isnan(result["pvals"][-1])
            assert np.isnan(result["pvalue_error"][-1])

    @pytest.mark.parametrize(
        "glm,masked", [(False, False), (True, False), (False, True), (True, True)]
    )
    def test_public_rank_one_extreme_tails(self, glm, masked):
        from scipy.stats import t

        groups = np.repeat([0, 1], 9)
        x = np.c_[np.ones(18), groups]
        if glm:
            x = np.c_[x, np.linspace(-1, 1, 18)]
        c = np.eye(x.shape[1])[1]
        presence = np.ones((18, 3), bool)
        if masked:
            presence[0, 1] = False
        y = np.zeros((18, 3))
        expected = np.ones(3)
        for gene, score in enumerate((20.0, 100.0)):
            rows = presence[:, gene]
            design = x[rows]
            df = rows.sum() - x.shape[1]
            residual = np.linalg.svd(design, full_matrices=True)[0][:, x.shape[1]]
            se_factor = np.linalg.norm(c @ np.linalg.pinv(design)) / np.sqrt(df)
            y[rows, gene] = design[:, 1] + residual / (score * se_factor)
            expected[gene] = 2 * t.sf(score, df)
        values = np.exp(y - y.max(axis=0) - 1)[..., None]
        values[~presence] = np.nan
        options = {"null": "rotation"}

        if glm:
            method = compare_glm_masked if masked else compare_glm
            args = [values, x, c]
        else:
            method = compare_two_groups_masked if masked else compare_two_groups
            args = [values, groups]
        if masked:
            args.append(presence)
        result = method(*args, **options).set_index("Feature").sort_index()
        np.testing.assert_allclose(result.P_value, expected, rtol=2e-11, atol=0)
        assert np.all(result.P_value_error <= np.minimum(1e-10, 0.001 * result.P_value))

    @pytest.mark.parametrize("n", [4, 18])
    def test_one_bin_profiles_match_student_t(self, n):
        from scipy.stats import t

        rng = np.random.default_rng(725)
        x = np.c_[np.ones(n), np.linspace(-1, 1, n)]
        y = rng.normal(size=(n, 6))
        inverse = np.linalg.pinv(x)
        residuals = y - x @ inverse @ y
        score = (inverse[1] @ y) / np.sqrt(
            np.sum(inverse[1] ** 2) * np.sum(residuals**2, axis=0) / (n - 2)
        )
        exact = 2 * t.sf(abs(score), n - 2)
        result = l2._run_l2_conditional(
            np.exp(y[..., None]),
            x,
            np.array([0.0, 1.0]),
        )
        np.testing.assert_allclose(result["pvals"], exact, rtol=0, atol=1e-10)

    @pytest.mark.parametrize("shape_only", [False, True])
    def test_masked_binary_glm_feature_order_shift_and_chunk_invariance(
        self, shape_only, monkeypatch
    ):
        rng = np.random.default_rng(726)
        groups = np.repeat([0, 1], 6)
        values = rng.lognormal(size=(12, 4, 9))
        presence = np.ones((12, 4), bool)
        presence[0, ::2] = False
        values[~presence] = np.nan
        options = {
            "normalize_shape": shape_only,
            "freq_weights": np.arange(1, 10),
            "null": "rotation",
        }
        expected = (
            compare_two_groups_masked(values, groups, presence, **options)
            .set_index("Feature")
            .sort_index()
        )
        glm = (
            compare_glm_masked(
                values, np.c_[np.ones(12), groups], np.array([0.0, -1.0]), presence, **options
            )
            .set_index("Feature")
            .sort_index()
        )
        pd.testing.assert_frame_equal(
            expected,
            glm.drop(columns=["n_obs", "df_resid"]).assign(
                n_obs_A=expected.n_obs_A, n_obs_B=expected.n_obs_B
            )[expected.columns],
        )
        monkeypatch.setattr(l2, "_COMPARISON_WORKSPACE_BYTES", 512)
        order = rng.permutation(4)
        actual = (
            compare_two_groups_masked(
                7 * values[:, order],
                groups,
                presence[:, order],
                gene_names=[str(i) for i in order],
                **options,
            )
            .set_index("Feature")
            .sort_index()
        )
        pd.testing.assert_frame_equal(actual, expected, rtol=0, atol=1e-10)

    def test_public_rotation_controls_and_failure_propagation(self, monkeypatch):
        import inspect

        from sonic.comparators import multisample as ms

        for method in (
            ms.compare_two_groups,
            ms.compare_two_groups_masked,
            ms.compare_glm,
            ms.compare_glm_masked,
        ):
            parameters = inspect.signature(method).parameters
            assert "n_rotations" not in parameters
            assert parameters["integration_tol"].kind == inspect.Parameter.KEYWORD_ONLY
        values = np.random.default_rng(17).lognormal(size=(8, 3, 5))
        groups = np.repeat([0, 1], 4)
        result = compare_two_groups(values, groups, null="rotation", random_state=1)
        replay = compare_two_groups(values, groups, null="rotation", random_state=2)
        pd.testing.assert_frame_equal(result, replay)
        assert result.attrs == replay.attrs
        assert (result.P_value_error.dropna() <= 1e-6).all()
        np.testing.assert_allclose(result.P_adj, apply_bh_correction(result.P_value))
        with pytest.raises(TypeError, match="n_rotations"):
            compare_two_groups(values, groups, null="rotation", n_rotations=99)

        def fail(*args, **kwargs):
            raise RuntimeError("unresolved integration")

        monkeypatch.setattr(l2, "_conditional_tails", fail)
        with pytest.raises(RuntimeError, match="unresolved integration"):
            compare_two_groups(values, groups, null="rotation")


class TestTCauchyCalibration:
    @pytest.mark.parametrize("glm", [False, True])
    @pytest.mark.parametrize("weights", [None, np.array([4.0, 2.0, 1.0, 0.5, 0.25])])
    def test_independent_gaussian_log_errors_have_calibrated_tails(self, glm, weights):
        rng = np.random.default_rng(907)
        spectra = np.exp(rng.normal(size=(20, 6000, 5)))
        if glm:
            design = pd.DataFrame({"x": np.linspace(-1, 1, 20), "batch": np.tile([0, 1], 10)})
            result = compare_glm(
                spectra, design, "x", statistic="log_t_cauchy", freq_weights=weights
            )
        else:
            result = compare_two_groups(
                spectra, np.repeat([0, 1], 10), statistic="log_t_cauchy", freq_weights=weights
            )
        assert 0.035 < np.mean(result.P_value <= 0.05) < 0.065
        assert 0.004 < np.mean(result.P_value <= 0.01) < 0.018
        per_bin = np.stack(result.P_value_per_bin)
        assert 0.04 < np.mean(per_bin <= 0.05) < 0.06
