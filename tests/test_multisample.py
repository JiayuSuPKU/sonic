"""Tests for comparator statistical comparison helpers."""

from __future__ import annotations

import itertools
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest

from sonic.comparators.multisample import (
    _AnalyticNullState,
    _estimate_glm_masked_null_covariance,
    _estimate_two_group_masked_null_covariance,
    _l2_pvalues_from_state,
    compare_glm,
    compare_glm_masked,
    compare_glm_scalar,
    compare_two_groups,
    compare_two_groups_masked,
    compare_two_groups_scalar,
)
from sonic.comparators.normalization import normalize_shape
from sonic.statistics import liu_sf


@pytest.mark.parametrize(
    "mode", ["binary", "masked", "glm", "masked_glm", "permutation", "masked_permutation"]
)
@pytest.mark.parametrize("shape_only", [False, True])
def test_float32_comparisons_match_the_same_values_promoted_to_float64(mode, shape_only):
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
            kwargs["null"] = "permutation" if "permutation" in mode else "analytic"
            if mode.startswith("masked"):
                result = compare_two_groups_masked(values, groups, presence, **kwargs)
            else:
                result = compare_two_groups(values, groups, **kwargs)
        return result.set_index("Feature").sort_index()[["Statistic", "P_value", "P_adj"]]

    np.testing.assert_array_equal(run(spectra), run(spectra.astype(np.float64)))
    np.testing.assert_array_equal(spectra, original)


def test_shape_only_comparison_ignores_low_power_amplitude_changes():
    spectra = np.tile(np.array([1.0, 2.0, 4.0]) * 1e-16, (8, 1, 1))
    spectra[4:] *= 10
    result = compare_two_groups(spectra, np.repeat([0, 1], 4), normalize_shape=True)
    np.testing.assert_array_equal(result["Statistic"], 0.0)
    np.testing.assert_array_equal(result["P_value"], 1.0)


def test_analytic_null_metadata_stays_optional_on_python_310():
    assert _AnalyticNullState.__optional_keys__ == {
        "n_obs_A",
        "n_obs_B",
        "design_columns",
        "contrast_vector",
        "n_obs",
        "zero_covariance",
    }
    assert {"observed", "beta", "eigenvalues", "eligible"} <= _AnalyticNullState.__required_keys__


class TestTwoGroupNullCalibration:
    """Permutation-null calibration checks."""

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


class TestLogL2AnalyticNull:
    """Analytic Liu mixture null for ``log_l2``."""

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
def test_glm_nuisance_offset_and_units_preserve_contrast(path, covariance_mode, offset, scale):
    rng = np.random.default_rng(673)
    n, n_genes, n_bins = 24, 8, 6
    group = np.repeat([0.0, 1.0], n // 2)
    covariate = rng.normal(size=n)
    design = np.column_stack([np.ones(n), group, covariate])
    shifted = design.copy()
    shifted[:, 2] = (covariate + offset) * scale
    contrast = np.array([0.0, 1.0, 0.0])
    values = 1.0 + 0.5 * group[:, None] + 0.2 * covariate[:, None] + rng.normal(size=(n, n_genes))
    spectra = np.exp(values[:, :, None] + rng.normal(size=(n, n_genes, n_bins)))
    presence = rng.random((n, n_genes)) > 0.15

    def compare(x):
        if path == "scalar":
            frame = compare_glm_scalar(values, x, contrast)
        elif path == "masked":
            frame = compare_glm_masked(spectra, x, contrast, presence, covariance=covariance_mode)
        else:
            frame = compare_glm(spectra, x, contrast, covariance=covariance_mode)
        return frame.sort_values("Feature").drop(columns="Feature").to_numpy()

    np.testing.assert_allclose(compare(shifted), compare(design), rtol=2e-6, atol=1e-10)
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
def test_glm_contrast_scale_preserves_pvalues_and_estimability(path, covariance_mode):
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
def test_glm_rejects_colliding_encoded_design_names(path):
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


def test_scalar_glm_perfect_fit_preserves_nonzero_effects():
    values = np.tile([0.0, 1.0, 1e-9, -1e-9], (4, 1))
    result = compare_glm_scalar(values, np.ones((4, 1)), np.ones(1)).set_index("Feature")
    result = result.loc[["0", "1", "2", "3"]]
    np.testing.assert_array_equal(result["Estimate"], values[0])
    np.testing.assert_array_equal(result["Statistic"], [0, np.inf, np.inf, np.inf])
    np.testing.assert_array_equal(result["P_value"], [1, 0, 0, 0])


@pytest.mark.parametrize(
    "path", ["two_group", "two_group_masked", "two_group_scalar", "glm", "glm_masked", "glm_scalar"]
)
@pytest.mark.parametrize("n_a,n_b", [(6, 6), (10, 30)])
def test_constant_responses_do_not_become_significant(path, n_a, n_b):
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


def test_glm_response_centering_preserves_intercepts_and_small_effects():
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
    # A constant column removed by the rank cutoff must not change that model.
    truncated = np.column_stack([np.full(12, 1e-18), groups])
    actual = compare_glm_scalar(values, truncated, np.array([0.0, 1.0]))
    np.testing.assert_allclose(actual[["Estimate", "P_value"]], result[["Estimate", "P_value"]])
    # A nearly constant covariate still does not span an exact intercept.
    near_constant = (1 + 1e-10 * groups)[:, None]
    actual = compare_glm_scalar(values, near_constant, np.ones(1))
    assert np.isfinite(actual.Statistic.iloc[0])
    values[groups == 1] += 1e-9
    result = compare_glm_scalar(values, explicit, np.array([0.0, 1e-16]))
    np.testing.assert_allclose(result.Estimate.iloc[0], (values[-1, 0] - 1) * 1e-16, rtol=1e-12)
    assert result.P_value.iloc[0] < 1e-10


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("target", ["response", "design", "contrast"])
def test_scalar_glm_rejects_nonfinite_inputs(value, target):
    values = np.arange(4.0)[:, None]
    design = np.ones((4, 1))
    contrast = np.ones(1)
    {"response": values, "design": design, "contrast": contrast}[target].flat[0] = value
    with pytest.raises(ValueError, match="finite"):
        compare_glm_scalar(values, design, contrast)


class TestCompareGLM:
    """GLM analytic comparisons for DataFrame, dict, and ndarray designs."""

    def test_binary_design_matches_groups_path_byte_close(self):
        """A binary GLM design should match the two-group analytic path."""
        rng = np.random.default_rng(0)
        spectra = np.exp(rng.standard_normal((8, 200, 25)))
        groups = np.array(["WT"] * 4 + ["TG"] * 4)

        df_g = (
            compare_two_groups(spectra, groups, statistic="log_l2", null="analytic")
            .sort_values("Feature")
            .reset_index(drop=True)
        )

        design = pd.DataFrame({"genotype": groups})
        df_d = (
            compare_glm(spectra, design, contrast="genotype")
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

    def test_glm_functions_do_not_expose_statistic_or_null_arguments(self):
        """GLM primitives are fixed log-L2 analytic tests."""
        import inspect

        for fn in (compare_glm, compare_glm_masked):
            sig = inspect.signature(fn)
            assert "statistic" not in sig.parameters
            assert "null" not in sig.parameters

    def test_masked_glm_matches_unmasked_when_complete(self):
        """A fully present mask should recover the ordinary GLM analytic result."""
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


class TestTwoGroupSignalDetection:
    """Power checks for implanted spectral differences."""

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


class TestStatisticAliases:
    """Statistic selection and frequency-weight behavior."""

    @pytest.mark.parametrize("stat", ["log_l2", "welch_t_cauchy"])
    def test_each_statistic_runs(self, stat):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(6, 8, 6))
        groups = np.array([0, 0, 0, 1, 1, 1])
        df = compare_two_groups(spectra, groups, statistic=stat, n_perm=50, random_state=0)
        assert df.shape[0] == 8
        assert {"Feature", "Statistic", "P_value", "P_adj"} <= set(df.columns)
        assert df["P_value"].between(0, 1).all()
        if stat == "welch_t_cauchy":
            assert "P_value_per_bin" in df.columns
            per_bin = np.stack(df["P_value_per_bin"].to_numpy())
            assert per_bin.shape == (8, 6)
            assert ((per_bin >= 0) & (per_bin <= 1)).all()

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
        low_pass = np.concatenate([np.ones(2), np.zeros(K - 2)])
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
        with pytest.raises(ValueError, match="non-negative"):
            compare_two_groups(
                spectra,
                groups,
                statistic="log_l2",
                n_perm=10,
                freq_weights=np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0]),
            )


class TestScalarTwoGroupComparison:
    """Scalar differential-expression calibration and power."""

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


class TestCompareGLMScalar:
    """Scalar GLM tests for continuous designs and logged responses."""

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


class TestMaskedTwoGroupComparison:
    """Masked two-group comparison behavior for missing features."""

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


def _stub_spectra_groups(seed: int = 0, n_a: int = 4, n_b: int = 4, n_genes: int = 30, K: int = 12):
    """Build reusable spectra and labels for normalize_shape option tests."""

    rng = np.random.default_rng(seed)
    spec = rng.uniform(0.5, 2.0, size=(n_a + n_b, n_genes, K))
    groups = np.array([0] * n_a + [1] * n_b)
    return spec, groups


class TestNormalizeShapeComparisonOptions:
    """``normalize_shape=True`` delegates to manual shape normalization."""

    @pytest.mark.parametrize(
        "statistic,null",
        [
            ("log_l2", "analytic"),
            ("log_l2", "permutation"),
            ("welch_t_cauchy", "permutation"),
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
            ("log_l2", "analytic"),
            ("log_l2", "permutation"),
            ("welch_t_cauchy", "permutation"),
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


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("exact", [False, True])
def test_permutation_tail_uses_exact_or_sampled_correction(masked, exact):
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
def test_streamed_permutations_match_materialized_weighted_null(masked, exact, dtype):
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
        null = np.array([statistic(spectra[keep, gene], assignment) for assignment in assignments])
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


@pytest.mark.parametrize("mode", ["two_group", "glm"])
@pytest.mark.parametrize("amplitude", [1.0, 1e-5, 0.0])
def test_masked_liu_reuses_fits_without_changing_tails(mode, amplitude):
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
    with patch("sonic.comparators.multisample.liu_sf", wraps=liu_sf) as fitted:
        actual = _l2_pvalues_from_state(state)
        expected_fits = (
            np.unique(state["contrast_scale"][state["eligible"]]).size if amplitude else 0
        )
        assert fitted.call_count == expected_fits
    # Batched and scalar tail evaluation can differ by floating-point roundoff.
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0, equal_nan=True)


@pytest.mark.parametrize("path", ["glm", "masked_glm", "scalar"])
@pytest.mark.parametrize("contrast_kind", ["str", "dict"])
def test_glm_metadata_names_preserve_contrasts(path, contrast_kind):
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


def test_welch_cauchy_is_invariant_to_spectral_units():
    rng = np.random.default_rng(25)
    spectra = rng.uniform(1, 5, size=(12, 4, 3))
    groups = np.repeat([0, 1], 6)
    presence = np.ones((12, 4), dtype=bool)
    presence[0, 0] = False
    for mask in (None, presence):
        results = []
        for scale in (1.0, 1e-8):
            kwargs = {"statistic": "welch_t_cauchy"}
            result = (
                compare_two_groups(spectra * scale, groups, **kwargs)
                if mask is None
                else compare_two_groups_masked(spectra * scale, groups, mask, **kwargs)
            )
            results.append(result.sort_values("Feature")["P_value"].to_numpy())
        np.testing.assert_allclose(*results, rtol=1e-10)


@pytest.mark.parametrize("n_b", [1, 5])
@pytest.mark.parametrize("spectral", [False, True])
def test_welch_requires_replication_in_both_groups(n_b, spectral):
    values = np.r_[100.0, np.arange(1.0, n_b + 1)][:, None]
    groups = np.r_[0, np.ones(n_b)]
    with pytest.raises(ValueError, match="at least two samples in each group"):
        if spectral:
            compare_two_groups(values[..., None], groups, statistic="welch_t_cauchy")
        else:
            compare_two_groups_scalar(values, groups)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("poison", ["value", "weight"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_permutation_rejects_nonfinite_observations_and_weights(masked, poison, value):
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


@pytest.mark.parametrize("masked", [False, True])
def test_welch_cauchy_excludes_only_constant_bins(masked):
    values = np.r_[np.arange(1.0, 7.0), np.arange(9.0, 15.0)]
    groups = np.repeat([0, 1], 6)
    spectra = np.stack([values, np.ones(12), 1 + groups], axis=1)[..., None]
    presence = np.ones((12, 3), dtype=bool)
    presence[0, 0] = False

    def compare(array):
        kwargs = {"statistic": "welch_t_cauchy"}
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


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("glm", [False, True])
@pytest.mark.parametrize("shape_only", [False, True])
@pytest.mark.parametrize("covariance_mode", [None, "per_gene"])
@pytest.mark.parametrize("n_bins", [7, 17])
def test_blocked_covariance_matches_independent_per_gene_fits(  # noqa: C901
    monkeypatch, masked, glm, shape_only, covariance_mode, n_bins
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
    monkeypatch.setattr(ms, "_COMPARISON_WORKSPACE_BYTES", 8 * 8 * 12 * n_bins * 2)
    name = "_estimate_" + ("glm" if glm else "two_group")
    name += "_masked_null_covariance" if masked else "_null_covariance"
    args = [spectra, design, contrast] if glm else [spectra, labels]
    if masked:
        args.append(presence)
    with patch.object(ms, "_decompose_glm_design", wraps=ms._decompose_glm_design) as decompose:
        with patch.object(ms, "_prepare_profile_block", wraps=ms._prepare_profile_block) as blocks:
            state = getattr(ms, name)(
                *args, freq_weights=weights, normalize_shape=shape_only, covariance=covariance_mode
            )
            assert max(call.args[0].shape[1] for call in blocks.call_args_list) <= 3
        if glm:
            expected = sum(mask.any() for mask in np.unique(presence.T, axis=0))
            assert decompose.call_count == expected
    np.testing.assert_allclose(state["observed"], observed, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(state["sigma_log"], covariance, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(
        ms._l2_pvalues_from_state(state), pvalues, atol=1e-12, equal_nan=True
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
        result.P_adj, ms.apply_bh_correction(pvalues), atol=1e-12, equal_nan=True
    )
    assert result.attrs["covariance"] == (covariance_mode or "pooled")
    assert result.attrs["null"] == "analytic"
    monkeypatch.setattr(ms, "_COMPARISON_WORKSPACE_BYTES", 32 * (1 << 20))
    whole = (
        getattr(ms, comparison)(
            *args, freq_weights=weights, normalize_shape=shape_only, covariance=covariance_mode
        )
        .set_index("Feature")
        .loc[result.index]
    )
    np.testing.assert_allclose(whole, result, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("glm", [False, True])
def test_scalar_result_builder_preserves_column_order_and_feature_names(glm):
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


class TestPerGeneCovariance:
    """Per-gene covariance equivalence, independence, and edge cases."""

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


class TestCovarianceOptions:
    """Covariance selection, defaults, validation, and storage."""

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
            ms, "_estimate_" + ("glm" if glm else "two_group") + suffix + "_null_covariance"
        )
        state = estimate(*args, covariance="per_gene", store_covariance=False)
        assert state["sigma_log"] is None
        assert state["weighted_cov"] is None
        with patch.object(ms.np.linalg, "eigvalsh", wraps=np.linalg.eigvalsh) as eig:
            fn(*args, covariance="per_gene")
        assert all(call.args[0].shape == (8, 8) for call in eig.call_args_list)


class TestBinaryGLMAdapters:
    """Binary GLM fits preserve coefficients, metadata, and per-arm eligibility."""

    @pytest.mark.parametrize("covariance", ["pooled", "per_gene"])
    @pytest.mark.parametrize("sizes", [(1, 7), (7, 1), (3, 5)])
    @pytest.mark.parametrize("profile_scale", ["df", "df_norm", "signed"])
    def test_binary_glm_adapter_preserves_null_and_metadata(self, covariance, sizes, profile_scale):
        from sonic.comparators import multisample as ms

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
        binary = ms._estimate_two_group_null_covariance(spectra, groups, **kwargs)
        glm = ms._estimate_glm_null_covariance(
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
            ms._l2_pvalues_from_state(binary),
            ms._l2_pvalues_from_state(glm),
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
        from sonic.comparators import multisample as ms

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
        binary = ms._estimate_two_group_masked_null_covariance(
            spectra, groups, presence, min_samples_per_group=minimum, **kwargs
        )
        glm = ms._estimate_glm_masked_null_covariance(
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
        pvalues = ms._l2_pvalues_from_state(binary)
        np.testing.assert_allclose(pvalues[eligible], ms._l2_pvalues_from_state(glm), atol=1e-12)
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
        from sonic.comparators import multisample as ms

        groups = np.repeat([0, 1], 4)
        presence = np.zeros((8, 3), bool)
        presence[:4, 1:] = True
        presence[4, 2] = True  # Estimable with residual df, but only one observation in arm B.
        spectra = np.full((8, 3, 5), np.nan)
        kwargs = {"covariance": covariance, "store_covariance": store_covariance}
        with pytest.raises(ValueError, match="no genes meet min_samples_per_group=2 per arm"):
            ms._estimate_two_group_masked_null_covariance(spectra, groups, presence, **kwargs)
        state = ms._estimate_two_group_masked_null_covariance(
            spectra, groups, presence, allow_empty=True, **kwargs
        )
        assert not state["eligible"].any()
        assert state["beta"].shape == (2, 3, 5)
        for key in ("observed", "beta", "contrast_scale", "eigenvalues"):
            assert np.isnan(state[key]).all()
        assert np.isnan(ms._l2_pvalues_from_state(state)).all()
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
            ms._estimate_two_group_masked_null_covariance(
                spectra, groups, presence, freq_weights=-np.ones(5), allow_empty=True, **kwargs
            )
