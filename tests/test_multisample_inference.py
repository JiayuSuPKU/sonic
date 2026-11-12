"""Tests for spectral covariance estimation and null calibration."""

from __future__ import annotations

import itertools
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest

from sonic.comparators import _inference as l2
from sonic.comparators import _regression as lm
from sonic.comparators._inference import (
    _AnalyticNullState,
    _estimate_glm_masked_null_covariance,
    _estimate_two_group_masked_null_covariance,
    _l2_pvalues_from_state,
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
    """Analytic L2 tails, masking, and low-df behavior."""

    def test_no_residual_df_rejects_analytic_but_allows_permutation(self):
        spectra = np.array([[[1.0, 2.0, 3.0]], [[2.0, 3.0, 4.0]]])
        groups = np.array([0, 1])
        with pytest.raises(ValueError, match="positive residual degrees of freedom"):
            compare_two_groups(spectra, groups)
        result = compare_two_groups(spectra, groups, null="permutation")
        np.testing.assert_array_equal(result["P_value"], 1.0)

    def test_synthetic_h0_pvalues_are_uniform(self):
        rng = np.random.default_rng(0)
        n_a, n_b, K = 4, 4, 30
        spectra = np.exp(rng.standard_normal((n_a + n_b, 5000, K)))
        groups = np.array([0] * n_a + [1] * n_b)
        df = compare_two_groups(spectra, groups, statistic="log_l2", null="analytic")
        _, ks_p = kstest(df["P_value"].to_numpy(), "uniform")
        assert ks_p > 0.01, f"analytic p-values not uniform under H0: KS p={ks_p:.4f}"
        assert 0.45 < df["P_value"].mean() < 0.55

    def test_analytic_breaks_permutation_floor_on_implanted_signal(self):
        """Analytic p-values can beat the exact-permutation floor on strong signals."""
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
        df_analytic = compare_two_groups(spectra, groups, statistic="log_l2", null="analytic")
        for g in ("0", "1", "2", "3", "4"):
            p_perm = df_perm.loc[df_perm["Feature"] == g, "P_value"].iloc[0]
            p_analytic = df_analytic.loc[df_analytic["Feature"] == g, "P_value"].iloc[0]
            assert p_analytic < p_perm, (
                f"gene {g}: analytic={p_analytic:.3g} not < perm={p_perm:.3g}"
            )
            assert p_analytic < 1.0 / 71.0, f"gene {g}: analytic={p_analytic:.3g} above perm floor"

    def test_analytic_argument_is_ignored_for_welch_t_cauchy(self):
        """``welch_t_cauchy`` ignores ``null`` and still returns its schema."""
        rng = np.random.default_rng(3)
        spectra = np.exp(rng.standard_normal((6, 50, 20)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        df = compare_two_groups(spectra, groups, statistic="welch_t_cauchy", null="analytic")
        assert "P_value_per_bin" in df.columns
        assert df["P_value"].between(0, 1).all()

    def test_unknown_null_raises(self):
        rng = np.random.default_rng(4)
        spectra = np.exp(rng.standard_normal((6, 50, 20)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        with pytest.raises(ValueError, match="Unknown null"):
            compare_two_groups(spectra, groups, statistic="log_l2", null="bootstrap")

    def test_small_df_emits_user_warning(self):
        """Analytic null warns when residual df makes variance estimates unstable."""
        rng = np.random.default_rng(0)
        spectra = np.exp(rng.standard_normal((3, 50, 20)))
        groups = np.array([0, 1, 1])
        with pytest.warns(UserWarning, match="log_l2 \\+ null='analytic' at residual df=1"):
            compare_two_groups(spectra, groups, statistic="log_l2", null="analytic")
        spectra = np.exp(rng.standard_normal((4, 50, 20)))
        groups = np.array([0, 0, 1, 1])
        with pytest.warns(UserWarning, match="log_l2 \\+ null='analytic' at residual df=2"):
            compare_two_groups(spectra, groups, statistic="log_l2", null="analytic")
        spectra = np.exp(rng.standard_normal((6, 50, 20)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            compare_two_groups(spectra, groups, statistic="log_l2", null="analytic")

    def test_full_sigma_calibrates_under_correlated_bins(self):
        """Full-covariance analytic p-values should stay calibrated with correlated bins."""
        rng = np.random.default_rng(42)
        n_a, n_b, G, K = 4, 4, 2000, 30
        # Shared noise creates near rank-1 covariance across bins.
        shared = rng.standard_normal((n_a + n_b, G, 1)) * 1.0
        per_bin = rng.standard_normal((n_a + n_b, G, K)) * 0.2
        log_y = shared + per_bin
        spectra = np.exp(log_y)
        groups = np.array([0] * n_a + [1] * n_b)
        df = compare_two_groups(spectra, groups, statistic="log_l2", null="analytic")
        ks_p = kstest(df["P_value"].to_numpy(), "uniform")[1]
        fpr05 = (df["P_value"] < 0.05).mean()
        assert fpr05 < 0.10, (
            f"Full-Sigma analytic p-values should be near-calibrated; got Pr(p<.05)={fpr05:.3f}"
        )
        assert ks_p > 1e-3 or fpr05 < 0.07, (
            f"Full-Sigma analytic p-values should be roughly uniform; KS_p={ks_p:.3g} fpr05={fpr05:.3f}"
        )

    def test_masked_analytic_parity_with_unmasked_when_complete(self):
        """All-present masks should match the unmasked analytic path."""
        rng = np.random.default_rng(5)
        n_samples, n_genes, K = 6, 100, 20
        spectra = np.exp(rng.standard_normal((n_samples, n_genes, K)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        presence = np.ones((n_samples, n_genes), dtype=bool)
        df_un = (
            compare_two_groups(spectra, groups, statistic="log_l2", null="analytic")
            .sort_values("Feature")
            .reset_index(drop=True)
        )
        df_m = (
            compare_two_groups_masked(
                spectra, groups, presence, statistic="log_l2", null="analytic"
            )
            .sort_values("Feature")
            .reset_index(drop=True)
        )
        np.testing.assert_allclose(
            df_un["Statistic"].to_numpy(), df_m["Statistic"].to_numpy(), atol=1e-12
        )
        np.testing.assert_allclose(
            df_un["P_value"].to_numpy(), df_m["P_value"].to_numpy(), atol=1e-12
        )

    def test_masked_analytic_calibration_under_synthetic_missingness(self):
        """Random missingness should not strongly distort H0 false positives."""
        rng = np.random.default_rng(0)
        n_samples, n_genes, K = 8, 800, 20
        spectra = np.exp(rng.standard_normal((n_samples, n_genes, K)))
        groups = np.array([0] * 4 + [1] * 4)

        df_full = compare_two_groups(spectra, groups, statistic="log_l2", null="analytic")
        fpr_full = (df_full["P_value"] < 0.05).mean()

        presence = rng.uniform(size=(n_samples, n_genes)) >= 0.25
        df_m = compare_two_groups_masked(
            spectra,
            groups,
            presence,
            statistic="log_l2",
            null="analytic",
            min_samples_per_group=2,
        )
        valid = df_m["P_value"].notna()
        assert valid.sum() > 100, f"only {valid.sum()} testable genes after masking"
        fpr_m = (df_m.loc[valid, "P_value"] < 0.05).mean()

        assert 0.0 < fpr_m < 0.20, f"masked FPR={fpr_m:.3f} far from nominal"
        assert 0.5 * fpr_full < fpr_m < 2.0 * fpr_full + 0.02, (
            f"masked vs unmasked FPR mismatch: masked={fpr_m:.3f} unmasked={fpr_full:.3f}"
        )

    def test_masked_analytic_skips_genes_with_insufficient_presence(self):
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
            null="analytic",
            min_samples_per_group=2,
        )
        row = df[df["Feature"] == "0"].iloc[0]
        assert np.isnan(row["P_value"]), f"expected NaN, got {row['P_value']}"
        assert row["n_obs_A"] == 1
        valid_genes = df[df["Feature"] != "0"]
        assert valid_genes["P_value"].notna().all()

    def test_masked_analytic_raises_when_no_eligible_genes(self):
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
                null="analytic",
                min_samples_per_group=2,
            )

    def test_masked_analytic_argument_is_ignored_for_welch_t_cauchy(self):
        """Masked ``welch_t_cauchy`` ignores ``null`` like the unmasked path."""
        rng = np.random.default_rng(9)
        n_samples, n_genes, K = 6, 10, 12
        spectra = np.exp(rng.standard_normal((n_samples, n_genes, K)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        presence = np.ones((n_samples, n_genes), dtype=bool)
        df = compare_two_groups_masked(
            spectra,
            groups,
            presence,
            statistic="welch_t_cauchy",
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
            state = _estimate_two_group_masked_null_covariance(spectra, groups, presence)
        else:
            design = np.column_stack([np.ones(12), groups, rng.normal(size=12)])
            state = _estimate_glm_masked_null_covariance(
                spectra, design, np.array([0.0, 1.0, 0.0]), presence
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

    def test_log_l2_pvalues_are_uniform_under_h0(self):
        rng = np.random.default_rng(42)
        n_samples, n_genes, K = 8, 200, 12
        spectra = rng.uniform(0.5, 5.0, size=(n_samples, n_genes, K))
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # Force the sampling path (max_exact_permutations=0) so the KS test can check
        # continuous uniformity. Exact enumeration with only C(8,4)=70
        # distinct relabellings produces a discrete distribution on 71
        # values, which KS rejects by construction even under perfect
        # calibration; its own calibration is covered by
        # ``test_exact_permutation_on_small_samples`` below.
        df = compare_two_groups(
            spectra,
            groups,
            statistic="log_l2",
            n_perm=300,
            random_state=0,
            max_exact_permutations=0,
        )
        _, ks_p = kstest(df["P_value"].to_numpy(), "uniform")
        assert ks_p > 0.01, f"p-values not uniform under H0: KS p={ks_p:.4f}"

    def test_exact_permutation_on_small_samples(self):
        """Exact enumeration is deterministic and calibrated despite discrete p-values."""
        rng = np.random.default_rng(0)
        n_genes, K = 50, 10
        spectra = rng.uniform(0.5, 5.0, size=(8, n_genes, K))
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        df_a = compare_two_groups(spectra, groups, statistic="log_l2", n_perm=5000, random_state=0)
        df_b = compare_two_groups(
            spectra, groups, statistic="log_l2", n_perm=5000, random_state=999
        )
        np.testing.assert_allclose(
            df_a.set_index("Feature").loc[df_b["Feature"], "P_value"].to_numpy(),
            df_b["P_value"].to_numpy(),
        )
        unique_pvals = set(df_a["P_value"].round(8).tolist())
        assert len(unique_pvals) <= 71
        assert 0.4 < df_a["P_value"].mean() < 0.6

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
        weights = np.array([0, 1, 2, 3, 4], dtype=float)
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
            elif not masked and gene > 0:
                pass  # Unmasked genes use the same relabellings.
            else:
                assignments = []
                for _ in range(43):
                    perm_rng.shuffle(labels)
                    assignments.append(labels.copy())
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
            "freq_weights": np.arange(17),
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
        logs[:, 2, 0] = rng.normal(size=8)  # Variation only in a zero-weight bin.
        logs[:, 3] = 1e-7 + 1e-14 * rng.normal(size=(8, 11))
        spectra = np.exp(logs)
        weights = np.ones(11)
        weights[0] = 0
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
        assert result.loc["2", "P_value"] == 1.0
        assert np.isfinite(result.loc["3", "P_value"])
        # Other genes must not supply either covariance or numerical tolerance.
        for j in [3, 4]:
            alone = run(spectra[:, j : j + 1], presence[:, j : j + 1])
            np.testing.assert_allclose(
                result.loc[str(j), ["Statistic", "P_value"]],
                alone.loc["0", ["Statistic", "P_value"]],
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

    def test_per_gene_low_df_retains_analytic_warning(self):
        rng = np.random.default_rng(5)
        with pytest.warns(UserWarning, match="residual df=2"):
            result = compare_two_groups(
                rng.lognormal(size=(4, 5, 29)), np.repeat([0, 1], 2), covariance="per_gene"
            )
        assert result.P_value.between(0, 1).all()

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("covariance", ["pooled", "per_gene"])
    @pytest.mark.parametrize("kwargs", [{"null": "permutation"}, {"statistic": "welch_t_cauchy"}])
    def test_covariance_selector_rejects_inapplicable_binary_tests(
        self, masked, covariance, kwargs
    ):
        values = np.ones((8, 2, 3))
        args = [values, np.repeat([0, 1], 4)]
        fn = compare_two_groups
        if masked:
            fn = compare_two_groups_masked
            args.append(np.ones((8, 2), bool))
        with pytest.raises(ValueError, match="covariance is only supported"):
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
        pd.testing.assert_frame_equal(fn(*args), fn(*args, covariance="pooled"))
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
    @pytest.mark.parametrize("sizes", [(1, 7), (7, 1), (3, 5)])
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
            "freq_weights": np.arange(17),
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
            "freq_weights": np.arange(17),
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

    def test_analytic_null_metadata_stays_optional_on_python_310(
        self,
    ):
        assert _AnalyticNullState.__optional_keys__ == {
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
        } <= _AnalyticNullState.__required_keys__

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("glm", [False, True])
    @pytest.mark.parametrize("shape_only", [False, True])
    @pytest.mark.parametrize("covariance_mode", [None, "per_gene"])
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
        weights = np.arange(n_bins, dtype=float)
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
                *args, freq_weights=weights, normalize_shape=shape_only, covariance=covariance_mode
            )
            .set_index("Feature")
            .loc[[str(i) for i in range(23)]]
        )
        np.testing.assert_allclose(result.P_value, pvalues, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            result.P_adj, apply_bh_correction(pvalues), atol=1e-12, equal_nan=True
        )
        assert result.attrs["covariance"] == (covariance_mode or "pooled")
        assert result.attrs["null"] == "analytic"
        monkeypatch.setattr(l2, "_COMPARISON_WORKSPACE_BYTES", 32 * (1 << 20))
        whole = (
            getattr(ms, comparison)(
                *args, freq_weights=weights, normalize_shape=shape_only, covariance=covariance_mode
            )
            .set_index("Feature")
            .loc[result.index]
        )
        np.testing.assert_allclose(whole, result, atol=1e-12, equal_nan=True)


class TestConditionalRotation:
    """Conditional tails, invariance, eligibility, and deterministic streaming."""

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
        weights = np.arange(bins, dtype=float)
        weights /= weights.sum()
        result = compare_glm_masked(
            values,
            x,
            c,
            presence,
            freq_weights=weights,
            normalize_shape=shape_only,
            null="rotation",
            n_rotations=499,
            random_state=35,
        )
        result = result.set_index("Feature").sort_index()
        expected = []
        for gene in range(3):
            rows = np.flatnonzero(presence[:, gene])
            power = values[rows, gene]
            if shape_only:
                power = normalize_shape(power)
            y = np.log(np.maximum(power, 1e-12))
            expected.append(self._reference(y, x[rows], c, weights, 35, 499, rows, n))
        np.testing.assert_allclose(result.iloc[:3][["Statistic", "P_value"]], expected, atol=1e-12)
        assert np.isnan(result.loc["3", "P_value"])
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
        kwargs = {"freq_weights": np.arange(9), "normalize_shape": shape_only}
        binary = ms.compare_two_groups_masked if masked else ms.compare_two_groups
        args = [values, groups, presence] if masked else [values, groups]
        result = binary(*args, null="rotation", n_rotations=199, random_state=4, **kwargs)
        analytic = binary(*args, covariance="per_gene", **kwargs)
        pd.testing.assert_series_equal(
            result.set_index("Feature").Statistic.sort_index(),
            analytic.set_index("Feature").Statistic.sort_index(),
        )
        glm = ms.compare_glm_masked if masked else ms.compare_glm
        glm_args = [values, np.c_[np.ones(10), groups], np.array([0.0, 1.0])]
        if masked:
            glm_args.append(presence & (presence[:5].sum(0) >= 2))
        reference = glm(*glm_args, null="rotation", n_rotations=199, random_state=4, **kwargs)
        np.testing.assert_allclose(
            result[["Statistic", "P_value", "P_adj"]],
            reference[["Statistic", "P_value", "P_adj"]],
            equal_nan=True,
        )
        assert result.attrs == {
            "null": "rotation",
            "covariance": None,
            "n_rotations": 199,
            "random_state": 4,
            "min_p_value": 0.005,
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
            n_rotations=49999,
            random_state=77,
            gene_names=list(map(str, range(24))),
        )
        actual = result.set_index("Feature").loc[list(map(str, range(24))), "P_value"].to_numpy()
        assert np.all(
            np.abs(actual - exact) <= 6 * np.sqrt(exact * (1 - exact) / 50000) + 3 / 50000
        )

    def test_nuisance_units_and_contrast_scaling(self):
        rng = np.random.default_rng(28)
        x = np.c_[np.ones(14), np.linspace(-1, 1, 14), rng.normal(size=14)]
        c = np.array([0.0, 1.0, 0.0])
        y = rng.normal(size=(14, 6, 9))
        kwargs = {"null": "rotation", "n_rotations": 999, "random_state": 11}
        reference = compare_glm(np.exp(y), x, c, **kwargs).set_index("Feature").sort_index()
        nuisance = rng.normal(size=(3, 54))
        nuisance[1] = 0
        shifted_y = y + (x @ nuisance).reshape(y.shape)
        shifted_x = x.copy()
        shifted_x[:, 2] = 10000 + 100 * x[:, 2]
        actual = (
            compare_glm(np.exp(shifted_y), shifted_x, c, **kwargs).set_index("Feature").sort_index()
        )
        np.testing.assert_allclose(actual, reference, atol=1e-10)
        for scale in (-3.0, 1e-12, 1e12):
            actual = (
                compare_glm(np.exp(y), x, c * scale, **kwargs).set_index("Feature").sort_index()
            )
            np.testing.assert_allclose(
                actual.Statistic, reference.Statistic * abs(scale), rtol=1e-10
            )
            np.testing.assert_array_equal(actual.P_value, reference.P_value)

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("contrast_scale", [1.0, 1e-12])
    def test_nuisance_only_profiles_and_nonzero_perfect_fits(self, masked, contrast_scale):
        rng = np.random.default_rng(1)
        z = rng.normal(size=10)
        x = np.c_[np.ones(10), np.linspace(-1, 1, 10), z]
        c = np.array([0.0, contrast_scale, 0.0])
        y = np.repeat((3 + z)[:, None, None] * np.array([1.0, 0.5, -0.2]), 3, axis=1)
        y[:, 1] += x[:, 1, None] * np.array([0.8, -0.3, 0.4])
        y[:, 2] += rng.normal(scale=0.1, size=(10, 3))
        # An ignored bin must not set the numerical-zero threshold.
        y[:, :, -1] = 100 * x[:, 1, None]
        presence = np.ones((10, 3), bool)
        if masked:
            presence[0, 0] = False
            y[~presence] = np.nan
        values = np.exp(y)
        kwargs = {
            "null": "rotation",
            "n_rotations": 999,
            "random_state": 20,
            "freq_weights": np.array([0.8, 0.2, 0.0]),
        }
        if masked:
            result = compare_glm_masked(values, x, c, presence, **kwargs)
        else:
            result = compare_glm(values, x, c, **kwargs)
        result = result.set_index("Feature")
        assert result.loc["0", "P_value"] == 1.0
        assert result.loc["1", "P_value"] == 0.001
        assert result.loc["1", "Statistic"] > 0
        reference = compare_glm(values[:, 2:3], x, c, **kwargs)
        assert result.loc["2", "P_value"] == reference.P_value.iloc[0]

    def test_roundoff_guard_preserves_tiny_and_large_profile_scales(self):

        rng = np.random.default_rng(52)
        x = np.c_[np.ones(10), np.linspace(-1, 1, 10), rng.normal(size=10)]
        y = rng.normal(size=(10, 1, 4)) + x[:, 1, None, None]
        scales = np.array([1e-100, 1.0, 1e100])
        profiles = y * scales[None, :, None]
        profiles[:, :, -1] = 1e100 * x[:, 1, None]
        result = l2._run_l2_rotations(
            profiles,
            x,
            np.array([0.0, 1.0, 0.0]),
            log_transform=False,
            freq_weights=np.array([0.2, 0.3, 0.5, 0.0]),
            n_rotations=999,
            random_state=20,
        )
        np.testing.assert_array_equal(result["pvals"], np.repeat(result["pvals"][1], 3))
        assert result["pvals"][1] < 1
        np.testing.assert_allclose(result["observed"] / scales, result["observed"][1])
        assert np.isfinite(result["effective_rank"]).all()

    def test_chunking_feature_order_and_shared_draws(self, monkeypatch):

        rng = np.random.default_rng(51)
        x = np.c_[np.ones(10), np.linspace(-1, 1, 10)]
        values = rng.lognormal(size=(10, 9, 13))
        values[:, 1] = values[:, 0]
        presence = np.ones((10, 9), bool)
        presence[0, 3:7] = False
        kwargs = {
            "null": "rotation",
            "n_rotations": 499,
            "random_state": 5,
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
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_array_equal(actual.P_value, expected.P_value)
        assert actual.loc["0", "P_value"] == actual.loc["1", "P_value"]
        order = rng.permutation(9)
        reordered = run(values[:, order], presence[:, order], [str(i) for i in order])
        np.testing.assert_allclose(reordered, expected, atol=1e-12)

    def test_constant_ties_perfect_fit_and_zero_weight_bins(self):
        groups = np.repeat([0, 1], 3)
        x = np.c_[np.ones(6), groups]
        a = np.linalg.pinv(x)[1]
        basis = np.c_[a / np.linalg.norm(a), np.linalg.svd(x, full_matrices=True)[0][:, 2:]]
        y = np.zeros((6, 4, 6))
        y[:, 0] = 4.2
        y[:, 1, :5] = groups[:, None]
        y[:, 2, :5] = np.sqrt(5) * basis
        y[:, 3, -1] = np.arange(6)
        result = compare_two_groups(
            np.exp(y),
            groups,
            null="rotation",
            n_rotations=199,
            random_state=17,
            freq_weights=np.r_[np.ones(5), 0],
        ).set_index("Feature")
        np.testing.assert_array_equal(
            result.loc[["0", "1", "2", "3"], "P_value"], [1.0, 0.005, 1.0, 1.0]
        )
        low_df = compare_two_groups(
            np.exp(y[:3]), np.array([0, 1, 1]), null="rotation", n_rotations=19, random_state=1
        )
        assert low_df.P_value.between(0.05, 1).all()

    def test_generated_seed_can_be_replayed(self):
        values = np.random.default_rng(14).lognormal(size=(8, 3, 5))
        groups = np.repeat([0, 1], 4)
        result = compare_two_groups(values, groups, null="rotation", n_rotations=99)
        replay = compare_two_groups(
            values,
            groups,
            null="rotation",
            n_rotations=99,
            random_state=result.attrs["random_state"],
        )
        pd.testing.assert_frame_equal(result, replay)
        assert result.attrs == replay.attrs

    @pytest.mark.parametrize("draws", [0, -1, 2.5, True])
    def test_invalid_draw_count(self, draws):
        with pytest.raises(ValueError, match="n_rotations must be a positive integer"):
            compare_two_groups(
                np.ones((6, 2, 3)), np.repeat([0, 1], 3), null="rotation", n_rotations=draws
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
        with pytest.raises(ValueError, match="requires statistic"):
            compare_two_groups(values, groups, null="rotation", statistic="welch_t_cauchy")
        with pytest.raises(ValueError, match="residual degrees"):
            compare_glm(values, np.eye(6), np.ones(6), null="rotation")
        with pytest.raises(ValueError, match="not estimable"):
            compare_glm(
                values, np.c_[np.ones(6), np.ones(6)], np.array([1.0, 0.0]), null="rotation"
            )
        with pytest.raises(ValueError, match="finite profiles"):
            compare_two_groups(np.full_like(values, np.nan), groups, null="rotation", n_rotations=9)
        with pytest.raises(ValueError, match="no features"):
            compare_two_groups_masked(
                values, groups, np.zeros((6, 2), bool), null="rotation", n_rotations=9
            )
