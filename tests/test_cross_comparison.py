"""Synthetic checks for aggregate cross-spectrum inference; no cohort data."""

import copy
import itertools
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import hadamard
from scipy.stats import beta, kstest, t, ttest_ind

from sonic import ComparatorIrregular
from sonic.comparators import _inference as l2
from sonic.comparators import _regression as lm
from sonic.comparators import multisample as ms
from sonic.comparators.cross_spectrum import CrossSpectra, _stabilize_cross_covariance
from sonic.statistics import apply_bh_correction, liu_sf
from tests.test_cross_spectrum import make_comparator  # noqa: F401


def _comparison(coherency):
    """Construct feasible disjoint pair spectra with unit matching powers."""
    coherency = np.asarray(coherency, dtype=complex)
    n_samples, n_pairs, n_bins = coherency.shape
    names = tuple(f"g{i}" for i in range(2 * n_pairs))
    comparator = ComparatorIrregular.__new__(ComparatorIrregular)
    comparator.samples = [None] * n_samples
    comparator.cross_spectra_ = CrossSpectra(
        cross_spectra=coherency.copy(),
        auto_spectra=np.ones((n_samples, 2 * n_pairs, n_bins)),
        pairs=tuple(zip(names[::2], names[1::2], strict=True)),
        gene_names=names,
        presence=np.ones((n_samples, 2 * n_pairs), dtype=bool),
        pair_indices=np.arange(2 * n_pairs).reshape(-1, 2),
        coherency_valid=np.ones(coherency.shape, dtype=bool),
        mode_pair_counts=np.full((n_samples, n_bins), 8),
        frequency_bin_intervals=np.column_stack([np.arange(n_bins), np.arange(1, n_bins + 1)]),
        angle_centers_rad=np.empty(0),
        rotation_angles_deg=np.zeros(n_samples),
        feature_mode="radial",
        chunk_schedules=[],
    )
    return comparator


def _ordered(frame):
    return frame.set_index(["Feature_1", "Feature_2"]).sort_index()


def _mask_missing_pairs(cross):
    """Make absent-gene values unusable so accidental inclusion cannot pass."""
    left, right = cross.pair_indices.T
    presence = cross.presence[:, left] & cross.presence[:, right]
    cross.auto_spectra[~cross.presence] = np.nan
    cross.cross_spectra[~presence] = np.nan
    cross.coherency_valid &= presence[..., None]
    return presence


def _direct_shared_covariance(values, presence, support, matrix, contrast, pairs):
    """Independent per-pair OLS reference with entrywise residual-df pooling."""
    n_bins = values.shape[-1]
    numerator = np.zeros((n_bins, n_bins))
    denominator = np.zeros_like(numerator)
    effects, scales = {}, {}
    for pair in pairs:
        bins = np.flatnonzero(support[pair])
        x = matrix[presence[:, pair]]
        y = values[presence[:, pair], pair][:, bins]
        fit = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = y - x @ fit
        numerator[np.ix_(bins, bins)] += residual.T @ residual
        denominator[np.ix_(bins, bins)] += len(x) - np.linalg.matrix_rank(x)
        effects[pair] = contrast @ fit
        scales[pair] = np.sum((contrast @ np.linalg.pinv(x)) ** 2)
    covariance = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )
    return covariance, effects, scales


def _profiles(coherency, representation):
    """Return independent, writable profiles on the scale tested by DF."""
    if representation == "coherence":
        return np.log(np.maximum(np.abs(coherency) ** 2, 1e-12))
    return np.asarray(coherency).real.copy()


def _permutation_reference(profiles, groups, weights, *, exact, n_perm=0, seed=None):
    """Compute a small independent label-permutation reference."""
    profiles = profiles - profiles[:1]

    def statistic(labels):
        difference = profiles[labels == 0].mean(0) - profiles[labels == 1].mean(0)
        return np.sqrt(np.sum(weights * difference**2, axis=-1))

    if exact:
        labels = []
        for subset in itertools.combinations(range(len(groups)), np.count_nonzero(groups == 0)):
            row = np.ones(len(groups), dtype=int)
            row[list(subset)] = 0
            labels.append(row)
    else:
        rng = np.random.default_rng(seed)
        row = groups.copy()
        labels = []
        for _ in range(n_perm):
            rng.shuffle(row)
            labels.append(row.copy())
    observed = statistic(groups)
    tails = np.array([statistic(row) for row in labels])
    correction = int(not exact)
    pvals = ((tails >= observed).sum(0) + correction) / (len(labels) + correction)
    return observed, pvals, len(labels)


@pytest.mark.parametrize("representation", ["coherence", "coherency"])
class TestPluginNull:
    @pytest.mark.parametrize("glm", [False, True])
    def test_pooled_null_matches_direct_ols_with_more_bins_than_samples(self, representation, glm):
        rng = np.random.default_rng(829)
        n, p, b = 12, 9, 19
        comparator = _comparison(rng.uniform(-0.8, 0.8, (n, p, b)))
        groups = np.repeat([0, 1], n // 2)
        matrix = np.column_stack([np.ones(n), groups])
        if glm:
            matrix = np.column_stack([matrix, rng.normal(size=n)])
        contrast = np.eye(matrix.shape[1])[1]
        profiles = _profiles(comparator.cross_spectra_.coherency, representation)
        weights = np.arange(1, b + 1, dtype=float)
        weights /= weights.sum()
        fit = np.linalg.lstsq(matrix, profiles.reshape(n, -1), rcond=None)[0]
        residual = (profiles.reshape(n, -1) - matrix @ fit).reshape(-1, b)
        covariance = residual.T @ residual / (p * (n - matrix.shape[1]))
        effect = (contrast @ fit).reshape(p, b)
        expected_stat = np.sqrt(np.sum(weights * effect**2, axis=1))
        scale = np.sum((contrast @ np.linalg.pinv(matrix)) ** 2)
        eigen = np.maximum(np.linalg.eigvalsh(np.sqrt(weights[:, None] * weights) * covariance), 0)
        expected_p = liu_sf(expected_stat**2, eigen * scale)
        args = {"contrast": contrast} if glm else {}
        actual = comparator.test_cross_diff_freq(
            matrix if glm else groups,
            representation=representation,
            freq_weights=weights,
            covariance="pooled",
            **args,
        )
        actual = actual.set_index("Feature_1").loc[[f"g{i}" for i in range(0, 2 * p, 2)]]
        np.testing.assert_allclose(actual.Statistic, expected_stat, rtol=1e-12)
        np.testing.assert_allclose(
            np.stack(actual.Estimate_per_bin), effect if glm else -effect, atol=1e-12
        )
        np.testing.assert_allclose(actual.P_value, expected_p, rtol=1e-11, atol=1e-13)
        assert (actual.N_pairs_pooled == p).all()
        assert (actual.N_bins == b).all()
        assert (actual.DF_resid == n - matrix.shape[1]).all()

    def test_glm_design_is_reused_across_pair_support_groups(self, representation):
        rng = np.random.default_rng(403)
        n_samples, n_pairs, n_bins = 12, 6, 7
        comparator = _comparison(rng.uniform(0.1, 0.8, (n_samples, n_pairs, n_bins)))
        cross = comparator.cross_spectra_
        cross.coherency_valid[0, :2, 0] = False
        cross.coherency_valid[1, 2:4, 1:3] = False
        design = pd.DataFrame({"condition": np.repeat([0, 1], 6), "age": rng.normal(size=12)})
        with patch.object(lm, "_decompose_glm_design", wraps=lm._decompose_glm_design) as decompose:
            actual = _ordered(
                comparator.test_cross_diff_freq(
                    design,
                    contrast="condition",
                    covariance="pooled",
                    representation=representation,
                )
            )
            assert decompose.call_count == 1

        # A single independently pooled covariance supplies every supported submatrix.
        matrix = np.column_stack([np.ones(n_samples), design.to_numpy()])
        support = cross.coherency_valid.all(axis=0)
        values = _profiles(cross.coherency, representation)
        covariance, effects, scales = _direct_shared_covariance(
            values,
            np.ones((n_samples, n_pairs), dtype=bool),
            support,
            matrix,
            np.array([0, 1, 0]),
            range(n_pairs),
        )
        assert np.linalg.eigvalsh(covariance).min() > 0
        for pair in range(n_pairs):
            bins = np.flatnonzero(support[pair])
            cov = covariance[np.ix_(bins, bins)]
            statistic = np.sqrt(np.mean(effects[pair] ** 2))
            eigenvalues = np.linalg.eigvalsh(cov / len(bins)) * scales[pair]
            row = actual.loc[cross.pairs[pair]]
            np.testing.assert_allclose(row.Statistic, statistic, atol=1e-12)
            np.testing.assert_allclose(row.P_value, liu_sf(statistic**2, eigenvalues), atol=1e-12)
            assert row.N_pairs_pooled == n_pairs
        assert actual.attrs["design_columns"] == ("Intercept", "condition", "age")

    @pytest.mark.parametrize("shared_noise", [0, 1])
    def test_null_tail_calibration_when_bins_exceed_samples(self, representation, shared_noise):
        rng = np.random.default_rng(867)
        n, p, b = 12, 1600, 25
        noise = 0.12 * (rng.normal(size=(n, p, b)) + shared_noise * rng.normal(size=(n, p, 1)))
        # Feasible coherencies with approximately Gaussian tested profiles.
        values = np.exp((-4 + noise) / 2) if representation == "coherence" else noise
        assert np.max(np.abs(values)) < 1
        comparator = _comparison(values)
        frame = comparator.test_cross_diff_freq(
            np.repeat([0, 1], n // 2), representation=representation, covariance="pooled"
        )
        if not shared_noise:
            assert kstest(frame.P_value, "uniform").pvalue > 0.001
        # Liu approximates the upper tail; a dominant eigenvalue can put mass at p=1.
        for alpha in (0.01, 0.05, 0.1):
            margin = 4 * np.sqrt(alpha * (1 - alpha) / p)
            assert abs((frame.P_value < alpha).mean() - alpha) < margin

    @pytest.mark.parametrize(
        "glm,masked,retained,covariance", [(False, True, 4, "per_pair"), (True, False, 6, "pooled")]
    )
    def test_cross_small_df_warning_policy_is_unchanged(
        self, representation, glm, masked, retained, covariance
    ):
        n = retained + 2 * masked
        comparator = _comparison(np.random.default_rng(225).uniform(0.1, 0.8, (n, 2, 3)))
        cross = comparator.cross_spectra_
        if masked:
            cross.presence[[0, n // 2], 0] = False
            _mask_missing_pairs(cross)
        cross.coherency_valid[-1, 0, 0] = False
        groups = np.repeat([0, 1], n // 2)
        design = pd.DataFrame({"condition": groups}) if glm else groups
        kwargs = {"contrast": "condition"} if glm else {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            comparator.test_cross_diff_freq(
                design, representation=representation, covariance=covariance, **kwargs
            )
        messages = [str(w.message) for w in caught if "at residual df=" in str(w.message)]
        if retained == 4:
            assert messages
            prefix = "log_l2" if representation == "coherence" else "l2"
            assert all(f"{prefix} + null='plugin' at residual df=2" in m for m in messages)
        else:
            assert not messages

    @pytest.mark.parametrize("covariance", ["pooled", "per_pair"])
    def test_covariance_chunking_and_readonly_caches(self, monkeypatch, representation, covariance):
        rng = np.random.default_rng(73)
        comparator = _comparison(rng.uniform(0.1, 0.9, (10, 13, 9)))
        original = copy.deepcopy(comparator.cross_spectra_)
        for name in ("cross_spectra", "auto_spectra", "coherency_valid"):
            getattr(comparator.cross_spectra_, name).setflags(write=False)
        groups = np.repeat([0, 1], 5)
        expected = _ordered(
            comparator.test_cross_diff_freq(
                groups, representation=representation, covariance=covariance
            )
        )
        monkeypatch.setattr(l2, "_COMPARISON_WORKSPACE_BYTES", 8 * 8 * 10 * 9 * 2)
        with patch.object(l2, "_prepare_profile_block", wraps=l2._prepare_profile_block) as blocks:
            actual = _ordered(
                comparator.test_cross_diff_freq(
                    groups, representation=representation, covariance=covariance
                )
            )
            assert (
                max(c.args[0].shape[1] for c in blocks.call_args_list if c.args[0].ndim == 3) <= 2
            )
        np.testing.assert_allclose(
            actual[["Statistic", "P_value"]], expected[["Statistic", "P_value"]]
        )
        for name in ("cross_spectra", "auto_spectra", "coherency_valid"):
            np.testing.assert_array_equal(
                getattr(comparator.cross_spectra_, name), getattr(original, name)
            )


class TestCovariancePooling:
    @pytest.mark.parametrize("masked", [False, True])
    def test_pooled_eigensystem_only_uses_tested_bins(self, masked):
        values = np.random.default_rng(42).uniform(0.1, 0.8, (10, 3, 64))
        comp = _comparison(values)
        if masked:
            comp.cross_spectra_.presence[0, 0] = False
            _mask_missing_pairs(comp.cross_spectra_)
        groups = np.repeat([0, 1], 5)
        supported = np.isin(np.arange(64), [4, 25, 53])
        comp.cross_spectra_.coherency_valid[..., ~supported] = False
        weights = np.arange(1, 65)
        with patch.object(np.linalg, "eigvalsh", wraps=np.linalg.eigvalsh) as eig:
            result = _ordered(
                comp.test_cross_diff_freq(groups, freq_weights=weights, covariance="pooled")
            )
        assert [call.args[0].shape for call in eig.call_args_list] == [(3, 3)]
        estimates = np.stack(result.Estimate_per_bin)
        assert np.isfinite(estimates[:, supported]).all()
        assert np.isnan(estimates[:, ~supported]).all()

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    def test_bin_support_groups_share_covariance_without_dropping_samples(self, representation):
        rng = np.random.default_rng(43)
        comparator = _comparison(rng.uniform(0.1, 0.7, (10, 5, 5)))
        cross = comparator.cross_spectra_
        cross.coherency_valid[0, :2, 0] = False
        cross.coherency_valid[3, 2:4, 1] = False
        cross.coherency_valid[0, 4, :] = False
        cross.cross_spectra[~cross.coherency_valid] = np.nan
        groups = np.repeat([0, 1], 5)
        cross.coherency_valid[..., 4] = False
        weights = np.array([1, 2, 3, 4, 5])
        frame = _ordered(
            comparator.test_cross_diff_freq(
                groups,
                representation=representation,
                freq_weights=weights,
                covariance="pooled",
            )
        )
        values = _profiles(cross.coherency, representation)
        support = cross.coherency_valid.all(axis=0) & (weights > 0)
        covariance, effects, scales = _direct_shared_covariance(
            values,
            np.ones((10, 5), dtype=bool),
            support,
            np.column_stack([np.ones(10), groups]),
            np.array([0, 1]),
            range(4),
        )
        assert np.linalg.eigvalsh(covariance).min() >= 0
        for pair in range(4):
            bins = np.flatnonzero(support[pair])
            w = weights[bins] / weights[bins].sum()
            cov = covariance[np.ix_(bins, bins)]
            stat = np.sqrt(np.sum(w * effects[pair] ** 2))
            eigen = np.linalg.eigvalsh(np.sqrt(w[:, None] * w) * cov) * scales[pair]
            np.testing.assert_allclose(frame.iloc[pair].Statistic, stat, rtol=1e-10)
            np.testing.assert_allclose(frame.iloc[pair].P_value, liu_sf(stat**2, eigen), rtol=1e-10)
            assert frame.iloc[pair].N_pairs_pooled == 4
        assert frame.attrs["covariance"] == "pooled_across_valid_bins_psd"
        assert (frame.N_samples == 10).all()
        assert frame.iloc[-1].Status == "no_common_bins"
        assert np.isnan(frame.iloc[-1].P_value)
        assert np.isnan(frame.iloc[-1].P_adj)
        expected_adj = apply_bh_correction(frame.P_value)
        np.testing.assert_allclose(frame.P_adj.iloc[:-1], expected_adj[:-1])
        assert frame.attrs["bin_indices"][("g0", "g1")] == (1, 2, 3)

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    @pytest.mark.parametrize("glm", [False, True])
    def test_shared_covariance_with_unequal_bin_and_sample_masks(self, representation, glm):
        rng = np.random.default_rng(848)
        n, p, b = 16, 6, 4
        values = rng.uniform(0.1, 0.8, (n, p, b))
        comparator = _comparison(values)
        cross = comparator.cross_spectra_
        cross.presence[0, 0] = False
        cross.presence[[1, 9, 10], 2] = False
        cross.presence[[2, 3], 4] = False
        cross.presence[:8, 8] = False  # Ineligible pair must not contribute.
        cross.pairs = cross.pairs[:-1] + (("g10", "g10"),)
        cross.pair_indices[-1] = (10, 10)  # Self-pairs also cannot contribute.
        present = _mask_missing_pairs(cross)
        cross.coherency_valid[4, :2, 0] = False
        cross.coherency_valid[5, 2:4, 1] = False
        cross.cross_spectra[~cross.coherency_valid] = np.nan
        groups = np.repeat([0, 1], 8)
        matrix = np.column_stack([np.ones(n), groups])
        if glm:
            matrix = np.column_stack([matrix, rng.normal(size=n)])
        contrast = np.eye(matrix.shape[1])[1]
        kwargs = {"contrast": contrast} if glm else {}
        weights = np.array([1, 2, 3, 4.0])
        actual = _ordered(
            comparator.test_cross_diff_freq(
                matrix if glm else groups,
                representation=representation,
                freq_weights=weights,
                covariance="pooled",
                **kwargs,
            )
        )
        support = (cross.coherency_valid | ~present[..., None]).all(axis=0)
        transformed = _profiles(values, representation)
        covariance, effects, scales = _direct_shared_covariance(
            transformed, present, support, matrix, contrast, range(4)
        )
        assert np.linalg.eigvalsh(covariance).min() > 0
        for pair in range(4):
            bins = np.flatnonzero(support[pair])
            w = weights[bins] / weights[bins].sum()
            cov = covariance[np.ix_(bins, bins)]
            eigenvalues = np.linalg.eigvalsh(np.sqrt(w[:, None] * w) * cov)
            statistic = np.sqrt(np.sum(w * effects[pair] ** 2))
            row = actual.loc[cross.pairs[pair]]
            np.testing.assert_allclose(row.Statistic, statistic, atol=1e-13)
            np.testing.assert_allclose(
                row.P_value, liu_sf(statistic**2, eigenvalues * scales[pair]), atol=1e-12
            )
            np.testing.assert_allclose(
                row.Effective_rank, eigenvalues.sum() ** 2 / np.sum(eigenvalues**2), atol=1e-12
            )
            assert row.N_pairs_pooled == 4
            assert row.N_samples == present[:, pair].sum()
        assert actual.loc[list(cross.pairs[4:])].P_value.isna().all()
        assert (actual.loc[list(cross.pairs[4:])].N_pairs_pooled == 0).all()

    def test_psd_correction_preserves_variances_zero_bins_and_input(self):
        # Three pairwise correlations of +1, +1, -1 cannot form a PSD matrix.
        correlation = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1.0]])
        scales = np.array([1, 2, 3.0])
        covariance = np.zeros((4, 4))
        covariance[:3, :3] = correlation * scales[:, None] * scales
        original = covariance.copy()
        covariance.setflags(write=False)
        actual = _stabilize_cross_covariance(covariance)
        expected = np.zeros((4, 4))
        expected[:3, :3] = (0.5 * correlation + 0.5 * np.eye(3)) * scales[:, None] * scales
        np.testing.assert_allclose(actual, expected, atol=1e-13)
        np.testing.assert_array_equal(np.diag(actual), np.diag(original))
        np.testing.assert_array_equal(covariance, original)
        assert np.linalg.eigvalsh(actual).min() > -1e-12
        for valid in (np.zeros((4, 4)), expected, np.diag([1, 2, 0, 3.0])):
            np.testing.assert_array_equal(_stabilize_cross_covariance(valid), valid)

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    def test_indefinite_covariance_is_corrected_once_before_pair_subsetting(self, representation):
        groups = np.repeat([0, 1], 3)
        residual = np.tile([-0.1, 0, 0.1], 2)
        profiles = np.zeros((6, 3, 3))
        support = np.array([[True, True, False], [True, False, True], [False, True, True]])
        for pair, signs in enumerate(([1, 1], [1, 1], [1, -1])):
            bins = np.flatnonzero(support[pair])
            profiles[:, pair, bins] = residual[:, None] * signs + groups[:, None] * [0.1, 0.2]
        values = np.exp((-4 + profiles) / 2) if representation == "coherence" else profiles
        comparator = _comparison(values)
        comparator.cross_spectra_.coherency_valid &= support[None]
        comparator.cross_spectra_.cross_spectra[~comparator.cross_spectra_.coherency_valid] = np.nan
        actual = _ordered(
            comparator.test_cross_diff_freq(
                groups, representation=representation, covariance="pooled"
            )
        )
        # The corrected shared correlation has off-diagonals +.5, +.5, -.5.
        expected_covariance = 0.005 * np.array([[2, 1, 1], [1, 2, -1], [1, -1, 2.0]])
        statistic = np.sqrt(np.mean(np.array([0.1, 0.2]) ** 2))
        for pair in range(3):
            bins = np.flatnonzero(support[pair])
            eigenvalues = np.linalg.eigvalsh(expected_covariance[np.ix_(bins, bins)] / 2) * (2 / 3)
            np.testing.assert_allclose(actual.iloc[pair].Statistic, statistic, atol=1e-13)
            np.testing.assert_allclose(
                actual.iloc[pair].P_value, liu_sf(statistic**2, eigenvalues), atol=1e-12
            )
            assert actual.iloc[pair].N_pairs_pooled == 3


class TestRepresentations:
    @pytest.mark.parametrize("glm", [False, True])
    def test_log_coherence_reuses_df_and_signed_test_detects_a_sign_reversal(self, glm):
        rng = np.random.default_rng(91)
        magnitude = rng.uniform(0.3, 0.7, (12, 4, 5))
        values = np.concatenate([magnitude, -magnitude])
        groups = np.repeat([0, 1], 12)
        comparator = _comparison(values)
        design = pd.DataFrame({"condition": groups}) if glm else groups
        kwargs = {"contrast": "condition"} if glm else {}
        coherence = comparator.test_cross_diff_freq(design, **kwargs)
        signed = comparator.test_cross_diff_freq(design, representation="coherency", **kwargs)
        reference = ms.compare_two_groups(values**2, groups)
        np.testing.assert_allclose(
            sorted(coherence.Statistic), sorted(reference.Statistic), atol=1e-14
        )
        np.testing.assert_allclose(coherence.P_value, 1)
        assert (signed.P_value < 1e-10).all()
        assert comparator.spectra_ is None  # Testing needs only cross-spectra.

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    def test_polar_conjugation_and_amplitude_scaling_preserve_inference(self, representation):
        rng = np.random.default_rng(8)
        values = rng.uniform(0.1, 0.8, (10, 4, 8)) * np.exp(
            1j * rng.uniform(-np.pi, np.pi, (10, 4, 8))
        )
        comparator = _comparison(values)
        result = comparator.cross_spectra_
        result.feature_mode = "2d"
        result.frequency_bin_intervals = result.frequency_bin_intervals[:2]
        result.angle_centers_rad = np.arange(4) * np.pi / 4
        groups = np.repeat([0, 1], 5)
        reference = _ordered(comparator.test_cross_diff_freq(groups, representation=representation))
        result.cross_spectra[::2] = result.cross_spectra[::2].conj()
        factors = np.exp(rng.normal(size=result.auto_spectra.shape[:2]))[:, :, None]
        result.auto_spectra *= factors**2
        result.cross_spectra *= (
            factors[:, result.pair_indices[:, 0]] * factors[:, result.pair_indices[:, 1]]
        )
        actual = _ordered(comparator.test_cross_diff_freq(groups, representation=representation))
        np.testing.assert_allclose(
            actual[["Statistic", "P_value"]], reference[["Statistic", "P_value"]], rtol=1e-12
        )

    def test_coherence_detects_changed_imaginary_coupling_with_unchanged_autos(self):
        rng = np.random.default_rng(170)
        magnitudes = rng.uniform(0.15, 0.25, (20, 3, 6))
        magnitudes[10:] += 0.5
        comparator = _comparison(1j * magnitudes)
        result = comparator.cross_spectra_
        result.feature_mode = "2d"
        result.frequency_bin_intervals = result.frequency_bin_intervals[:2]
        result.angle_centers_rad = np.arange(3) * np.pi / 3
        groups = np.repeat([0, 1], 10)
        coherence = comparator.test_cross_diff_freq(groups)
        real = comparator.test_cross_diff_freq(groups, representation="coherency")
        autos = ms.compare_two_groups(result.auto_spectra, groups)
        assert (coherence.P_value < 1e-8).all()
        assert (real.P_value == 1).all()
        assert (autos.P_value == 1).all()


class TestSamplePresence:
    @pytest.mark.parametrize(
        "options",
        [
            {"null": "rotation"},
            {"covariance": "per_pair"},
            {"covariance": "pooled"},
            {"statistic": "t_cauchy"},
        ],
    )
    def test_masked_pairs_do_not_change_complete_glm_fit(self, options):
        values = np.random.default_rng(58).uniform(0.1, 0.8, (6, 3, 4))
        design = pd.DataFrame({"condition": [0, 0, 0, 1, 1, 1]})
        kwargs = {"contrast": "condition", "min_resid_df": 5, **options}
        reference = _comparison(values[:, :1]).test_cross_diff_freq(design, **kwargs).iloc[0]
        comparator = _comparison(values)
        cross = comparator.cross_spectra_
        cross.pairs = (*cross.pairs[:2], ("g4", "g4"))
        cross.pair_indices[2] = [4, 4]
        cross.presence[-1, [2, 4]] = False
        _mask_missing_pairs(cross)
        actual = _ordered(comparator.test_cross_diff_freq(design, **kwargs))
        assert actual.Status.tolist() == ["ok", "ineligible_design", "self_pair"]
        complete = actual.iloc[0]
        assert complete.DF_resid == 4
        assert np.isfinite(complete.P_value)
        np.testing.assert_allclose(complete.P_value, reference.P_value)
        np.testing.assert_allclose(complete.Statistic, reference.Statistic)

    def test_complete_pairs_share_sampled_labels_in_mixed_panel(self):
        values = np.random.default_rng(15).uniform(0.1, 0.8, (10, 3, 4))
        groups = np.repeat([0, 1], 5)
        options = {
            "null": "permutation",
            "random_state": 19,
            "n_perm": 29,
            "max_exact_permutations": 0,
        }
        expected = _ordered(_comparison(values[:, 1:]).test_cross_diff_freq(groups, **options))
        comparator = _comparison(values)
        comparator.cross_spectra_.presence[0, 0] = False
        _mask_missing_pairs(comparator.cross_spectra_)
        actual = _ordered(comparator.test_cross_diff_freq(groups, **options))
        np.testing.assert_array_equal(actual.P_value.iloc[1:], expected.P_value)

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    @pytest.mark.parametrize("glm", [False, True])
    def test_pair_presence_matches_masked_df_and_independent_pooled_ols(self, representation, glm):
        rng = np.random.default_rng(482)
        n, p, b = 12, 4, 19
        values = rng.uniform(-0.8, 0.8, (n, p, b))
        comparator = _comparison(values)
        cross = comparator.cross_spectra_
        cross.presence[0, 0] = False
        cross.presence[1, 1] = False  # Either partner can exclude a sample.
        cross.presence[2, 2] = False
        cross.presence[7, 3] = False  # Same counts, different retained samples.
        cross.presence[:6, 6] = False  # No eligible condition contrast for this pair.
        present = _mask_missing_pairs(cross)
        groups = np.repeat([0, 1], 6)
        matrix = np.column_stack([np.ones(n), groups])
        if glm:
            matrix = np.column_stack([matrix, rng.normal(size=n)])
        contrast = np.eye(matrix.shape[1])[1]
        kwargs = {"covariance": "pooled"}
        if glm:
            kwargs["contrast"] = contrast
        design = matrix if glm else groups
        weights = np.arange(1, b + 1, dtype=float)
        weights /= weights.sum()
        transformed = _profiles(values, representation)
        covariance, effects, scales = _direct_shared_covariance(
            transformed, present, np.ones((p, b), dtype=bool), matrix, contrast, range(3)
        )
        dfs = [
            present[:, pair].sum() - np.linalg.matrix_rank(matrix[present[:, pair]])
            for pair in range(3)
        ]
        stats = [np.sqrt(np.sum(weights * effects[pair] ** 2)) for pair in range(3)]
        eigen = np.maximum(np.linalg.eigvalsh(np.sqrt(weights[:, None] * weights) * covariance), 0)
        expected_p = np.array(
            [
                liu_sf(stat**2, scale * eigen)
                for stat, scale in zip(stats, scales.values(), strict=True)
            ]
        )
        original = copy.deepcopy(cross)
        for name in ("presence", "cross_spectra", "auto_spectra", "coherency_valid"):
            getattr(cross, name).setflags(write=False)
        actual = _ordered(
            comparator.test_cross_diff_freq(
                design, representation=representation, freq_weights=weights, **kwargs
            )
        )
        np.testing.assert_allclose(actual.Statistic.iloc[:3], stats, rtol=1e-12)
        np.testing.assert_allclose(
            actual.P_value.iloc[:3], expected_p.ravel(), rtol=1e-10, atol=1e-13
        )
        np.testing.assert_array_equal(actual.N_samples, present.sum(0))
        np.testing.assert_array_equal(actual.DF_resid.iloc[:3], dfs)
        assert (actual.N_pairs_pooled.iloc[:3] == 3).all()
        assert (actual.N_bins == b).all()
        assert actual.Status.iloc[-1] == ("ineligible_design" if glm else "insufficient_samples")
        assert np.isnan(actual.P_value.iloc[-1])
        if not glm:
            np.testing.assert_array_equal(actual.n_obs_A, present[:6].sum(0))
            np.testing.assert_array_equal(actual.n_obs_B, present[6:].sum(0))
        if representation == "coherence":
            comparator.spectra_ = values**2
            comparator.presence_ = present
            comparator.gene_names = [pair[0] for pair in cross.pairs]
            reference = comparator.test_diff_freq(design, freq_weights=weights, **kwargs).set_index(
                "Feature"
            )
            reference = reference.loc[comparator.gene_names]
            np.testing.assert_allclose(
                actual.Statistic, reference.Statistic, equal_nan=True, atol=1e-12
            )
            np.testing.assert_allclose(
                actual[["P_value", "P_adj"]],
                reference[["P_value", "P_adj"]],
                equal_nan=True,
                atol=1e-12,
            )
        for pair, mask in zip(cross.pairs, present.T, strict=True):
            assert actual.attrs["sample_indices"][pair] == tuple(np.flatnonzero(mask))
        for name in ("presence", "cross_spectra", "auto_spectra", "coherency_valid"):
            np.testing.assert_array_equal(getattr(cross, name), getattr(original, name))

    @pytest.mark.parametrize("glm", [False, True])
    @pytest.mark.parametrize("covariance", ["pooled", "per_pair"])
    def test_empty_masked_pool_does_not_abort_other_bin_groups(self, glm, covariance):
        comparator = _comparison(np.random.default_rng(28).uniform(0.1, 0.8, (10, 2, 4)))
        cross = comparator.cross_spectra_
        cross.presence[:5, 0] = False
        _mask_missing_pairs(cross)
        cross.coherency_valid[0, 1, 0] = False  # Different bin support from the ineligible pair.
        groups = np.repeat([0, 1], 5)
        design = pd.DataFrame({"condition": groups}) if glm else groups
        kwargs = {"contrast": "condition"} if glm else {}
        kwargs["covariance"] = covariance
        actual = _ordered(comparator.test_cross_diff_freq(design, **kwargs))
        assert np.isnan(actual.P_value.iloc[0])
        assert actual.N_pairs_pooled.iloc[0] == 0
        assert actual.Status.iloc[1] == "ok"
        assert actual.N_bins.iloc[1] == 3
        assert actual.N_samples.iloc[1] == 10
        assert np.isfinite(actual.P_value.iloc[1])
        cross.presence[-1, 2] = False
        _mask_missing_pairs(cross)
        kwargs["min_resid_df" if glm else "min_samples_per_group"] = 20
        empty = comparator.test_cross_diff_freq(design, **kwargs)
        assert empty.P_value.isna().all()
        assert (empty.N_pairs_pooled == 0).all()

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    def test_masked_degenerate_and_absent_pairs(self, representation):
        comparator = _comparison(np.zeros((8, 2, 3)))
        cross = comparator.cross_spectra_
        cross.presence[0, 0] = False
        cross.presence[:, 2] = False
        _mask_missing_pairs(cross)
        groups = np.repeat([0, 1], 4)
        actual = _ordered(
            comparator.test_cross_diff_freq(groups, representation=representation, null="plugin")
        )
        assert actual.P_value.iloc[0] == 1  # Valid zero coupling remains observed.
        assert actual.N_samples.iloc[0] == 7
        assert actual.Status.iloc[1] == "insufficient_samples"
        assert actual.N_samples.iloc[1] == 0
        assert actual.N_bins.iloc[1] == 0
        cross.cross_spectra[4:, 0] = 0.5
        changed = _ordered(
            comparator.test_cross_diff_freq(groups, representation=representation, null="plugin")
        )
        assert changed.Status.iloc[0] == "degenerate_null"
        assert np.isnan(changed.P_value.iloc[0])


@pytest.mark.parametrize("representation", ["coherence", "coherency"])
class TestPermutations:
    def test_equal_maxima_count_across_support_groups(self, representation):
        groups = np.repeat([0, 1], [2, 7])
        values = np.full((9, 2, 2), 0.4)
        values[[0, 1, 5], 0, 0] = 0.9
        if representation == "coherency":
            values[:, 0, 0] *= 2.0**-40
        comp = _comparison(values)
        for supported in (True, False):
            comp.cross_spectra_.coherency_valid[:, 1, 1] = supported
            result = _ordered(
                comp.test_cross_diff_freq(groups, null="permutation", representation=representation)
            )
            assert result.P_value.iloc[0] == 3 / 36
        comp.cross_spectra_.cross_spectra[5, 0, 0] *= 1 - 1e-8
        result = _ordered(
            comp.test_cross_diff_freq(groups, null="permutation", representation=representation)
        )
        assert result.P_value.iloc[0] == 1 / 36

    @pytest.mark.parametrize("exact", [False, True])
    def test_permutations_match_exhaustive_or_sampled_label_reference(self, representation, exact):
        rng = np.random.default_rng(98)
        comparator = _comparison(rng.uniform(-0.9, 0.9, (6, 4, 3)))
        groups = np.repeat([0, 1], 3)
        profiles = _profiles(comparator.cross_spectra_.coherency, representation)
        weights = np.array([1, 2, 4]) / 7
        observed, expected, n_used = _permutation_reference(
            profiles, groups, weights, exact=exact, n_perm=39, seed=17
        )
        frame = _ordered(
            comparator.test_cross_diff_freq(
                groups,
                representation=representation,
                null="permutation",
                freq_weights=weights,
                n_perm=39,
                random_state=17,
                max_exact_permutations=100 if exact else 0,
            )
        )
        np.testing.assert_allclose(frame.Statistic, observed, atol=1e-14)
        np.testing.assert_allclose(frame.P_value, expected)
        assert frame.attrs["n_permutations"] == n_used
        assert frame.attrs["exact_permutations"] == exact

    @pytest.mark.parametrize("exact", [False, True])
    def test_masked_permutations_use_only_present_samples(self, representation, exact):
        rng = np.random.default_rng(632)
        comparator = _comparison(rng.uniform(-0.8, 0.8, (8, 2, 4)))
        cross = comparator.cross_spectra_
        cross.presence[[0, 7], 0] = False
        cross.presence[0, 2] = False
        cross.presence[7, 3] = False
        present = _mask_missing_pairs(cross)
        groups = np.repeat([0, 1], 4)
        mask = present[:, 0]
        values = _profiles(cross.cross_spectra[mask], representation)
        perm_rng = np.random.default_rng(57)
        references = [
            _permutation_reference(
                values[:, pair : pair + 1],
                groups[mask],
                np.full(values.shape[-1], 1 / values.shape[-1]),
                exact=exact,
                n_perm=49,
                seed=perm_rng,
            )
            for pair in range(values.shape[1])
        ]
        observed = np.array([r[0][0] for r in references])
        expected = np.array([r[1][0] for r in references])
        n_used = references[0][2]
        actual = _ordered(
            comparator.test_cross_diff_freq(
                groups,
                representation=representation,
                null="permutation",
                n_perm=49,
                random_state=57,
                max_exact_permutations=30 if exact else 0,
            )
        )
        np.testing.assert_allclose(actual.Statistic, observed, atol=1e-14)
        np.testing.assert_allclose(actual.P_value, expected)
        assert set(actual.attrs["n_permutations"].values()) == {n_used}
        assert set(actual.attrs["exact_permutations"].values()) == {exact}
        if representation == "coherence":
            comparator.spectra_ = np.abs(cross.cross_spectra) ** 2
            comparator.presence_ = present
            comparator.gene_names = [pair[0] for pair in cross.pairs]
            reference = (
                comparator.test_diff_freq(
                    groups,
                    null="permutation",
                    n_perm=49,
                    random_state=57,
                    max_exact_permutations=30 if exact else 0,
                )
                .set_index("Feature")
                .loc[comparator.gene_names]
            )
            np.testing.assert_array_equal(
                actual[["Statistic", "P_value", "P_adj"]],
                reference[["Statistic", "P_value", "P_adj"]],
            )

    @pytest.mark.parametrize("exact", [False, True])
    def test_masked_permutations_follow_pair_order_across_bin_groups(self, representation, exact):
        comparator = _comparison(np.random.default_rng(57).uniform(-0.8, 0.8, (8, 4, 4)))
        cross = comparator.cross_spectra_
        cross.presence[np.ix_([0, 7], [0, 4])] = False
        cross.presence[np.ix_([1, 6], [2, 6])] = False
        present = _mask_missing_pairs(cross)
        cross.coherency_valid[2, [0, 2], 0] = False
        cross.coherency_valid[2, [1, 3], 1] = False
        groups = np.repeat([0, 1], 4)
        with patch(
            "sonic.comparators.cross_spectrum._make_label_permutations",
            wraps=l2._make_label_permutations,
        ) as permute:
            actual = _ordered(
                comparator.test_cross_diff_freq(
                    groups,
                    representation=representation,
                    null="permutation",
                    n_perm=49,
                    random_state=13,
                    max_exact_permutations=10000 if exact else 0,
                )
            )
            assert permute.call_count == 4
        perm_rng = np.random.default_rng(13)
        for i, pair in enumerate(cross.pairs):
            mask = present[:, i]
            bins = cross.coherency_valid[mask, i].all(0)
            values = _profiles(cross.cross_spectra[mask, i][:, bins], representation)
            observed, expected, _ = _permutation_reference(
                values,
                groups[mask],
                np.full(bins.sum(), 1 / bins.sum()),
                exact=exact,
                n_perm=49,
                seed=perm_rng,
            )
            assert actual.loc[pair, "N_samples"] == mask.sum()
            np.testing.assert_allclose(actual.loc[pair, "Statistic"], observed, atol=1e-14)
            np.testing.assert_allclose(actual.loc[pair, "P_value"], expected)


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs,error,match",
        [
            ({"representation": "cross"}, ValueError, "representation"),
            ({"null": "foo"}, ValueError, "null"),
            ({"null": "analytic"}, ValueError, "plugin"),
            ({"covariance": "per_gene"}, ValueError, "covariance"),
            ({"covariance": "typo"}, ValueError, "covariance"),
            ({"null": "permutation", "covariance": "per_pair"}, ValueError, "plugin"),
            ({"null": "permutation", "covariance": "pooled"}, ValueError, "plugin"),
            ({"freq_weights": [0, 0, 0]}, ValueError, "strictly positive"),
            ({"freq_weights": [1, np.nan, 1]}, ValueError, "finite"),
            ({"null": "permutation", "n_perm": 0}, ValueError, "positive integer"),
            ({"null": "permutation", "n_perm": 1.5}, ValueError, "positive integer"),
            (
                {"null": "permutation", "max_exact_permutations": -1},
                ValueError,
                "nonnegative integer",
            ),
            ({"null": "permutation", "contrast": "group"}, NotImplementedError, "plugin"),
        ],
    )
    def test_invalid_test_options_fail_before_testing(self, kwargs, error, match):
        comparator = _comparison(np.full((8, 2, 3), 0.5))
        with pytest.raises(error, match=match):
            comparator.test_cross_diff_freq(np.repeat([0, 1], 4), **kwargs)

    def test_invalid_or_missing_design_and_uncomputed_spectra(self):
        comparator = _comparison(np.full((8, 2, 3), 0.5))
        with pytest.raises(ValueError, match="contrast="):
            comparator.test_cross_diff_freq(pd.DataFrame({"group": np.repeat([0, 1], 4)}))
        with pytest.raises(ValueError, match="nonmissing"):
            comparator.test_cross_diff_freq([0] * 4 + [np.nan] * 4)
        with pytest.raises(ValueError, match="residual degrees"):
            comparator.test_cross_diff_freq(np.eye(8), contrast=np.ones(8))
        with pytest.raises(ValueError, match="not estimable"):
            comparator.test_cross_diff_freq(np.ones((8, 2)), contrast=np.array([1.0, -1.0]))
        comparator.cross_spectra_ = None
        with pytest.raises(RuntimeError, match="compute_cross_spectra"):
            comparator.test_cross_diff_freq(np.repeat([0, 1], 4))

    @pytest.mark.parametrize(
        "corruption,error,match",
        [
            ("nonfinite", ValueError, "finite values"),
            ("zero_power", ValueError, "positive auto-power"),
            ("bound", ArithmeticError, "coherence bound"),
        ],
    )
    def test_corrupted_valid_bins_are_rejected(self, corruption, error, match):
        comparator = _comparison(np.full((8, 2, 3), 0.5))
        if corruption == "zero_power":
            comparator.cross_spectra_.auto_spectra[0, 0, 0] = 0
        else:
            comparator.cross_spectra_.cross_spectra[0, 0, 0] = (
                np.nan if corruption == "nonfinite" else 1.5
            )
        with pytest.raises(error, match=match):
            comparator.test_cross_diff_freq(np.repeat([0, 1], 4))

    @pytest.mark.parametrize("argument,value", [("min_samples_per_group", 1), ("min_resid_df", 0)])
    def test_cross_presence_threshold_controls_are_validated(self, argument, value):
        comparator = _comparison(np.full((8, 1, 3), 0.5))
        with pytest.raises(ValueError, match=argument):
            comparator.test_cross_diff_freq(np.repeat([0, 1], 4), **{argument: value})

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    @pytest.mark.parametrize("glm", [False, True])
    def test_constant_profiles_self_pairs_and_degenerate_noise(self, representation, glm):
        groups = np.repeat([0, 1], 4)
        design = pd.DataFrame({"condition": groups}) if glm else groups
        kwargs = {"contrast": "condition"} if glm else {}
        comparator = _comparison(np.full((8, 2, 4), 0.7))
        cross = comparator.cross_spectra_
        cross.pairs = (("g0", "g0"), ("g2", "g3"))
        cross.pair_indices[0] = (0, 0)
        frame = _ordered(
            comparator.test_cross_diff_freq(
                design, representation=representation, null="plugin", **kwargs
            )
        )
        assert frame.iloc[0].Status == "self_pair"
        assert np.isnan(frame.iloc[0].P_value)
        assert frame.iloc[1].P_value == 1
        cross.cross_spectra[4:, 1] = 0.2
        frame = _ordered(
            comparator.test_cross_diff_freq(
                design, representation=representation, null="plugin", **kwargs
            )
        )
        assert frame.iloc[1].Status == "degenerate_null"
        assert np.isnan(frame.iloc[1].P_value)

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    def test_zero_valid_coupling_is_distinct_from_invalid_denominator(self, representation):
        comparator = _comparison(np.zeros((8, 2, 3)))
        comparator.cross_spectra_.coherency_valid[:, 1] = False
        comparator.cross_spectra_.auto_spectra[:, 2:] = 0
        frame = _ordered(
            comparator.test_cross_diff_freq(np.repeat([0, 1], 4), representation=representation)
        )
        assert frame.iloc[0].Status == "ok"
        assert frame.iloc[0].P_value == 1
        assert frame.iloc[1].Status == "no_common_bins"


@pytest.mark.parametrize("backend", ["fft", "nufft"])
@pytest.mark.parametrize("mode", ["radial", "2d"])
class TestIntegration:
    def test_extraction_to_testing_on_both_backends(self, make_comparator, backend, mode):  # noqa: F811
        rng = np.random.default_rng(9)
        samples = [rng.normal(size=(3, 17, 19)) for _ in range(8)]
        comparator = make_comparator(samples, backend=backend, mode=mode)
        comparator.compute_cross_spectra([("a", "b"), ("b", "c")], n_jobs=1, progress=False)
        groups = np.repeat([0, 1], 4)
        for representation in ("coherence", "coherency"):
            frame = comparator.test_cross_diff_freq(groups, representation=representation)
            assert len(frame) == 2
            assert (frame.Status == "ok").all()
            assert frame.P_value.between(0, 1).all()
            assert frame.attrs["null"] == "rotation"
            assert (frame.N_pairs_pooled == 0).all()

    def test_pair_presence_extraction_and_subsetting(self, make_comparator, backend, mode):  # noqa: F811
        rng = np.random.default_rng(73)
        samples = [rng.normal(size=(3, 17, 19)) for _ in range(8)]
        samples[0][1] = 0
        samples[1][2] = 0
        # An absent unselected gene must not exclude a pair's sample.
        samples[2][0] = 0
        comparator = make_comparator(samples, backend=backend, mode=mode, presence_threshold=0.1)
        comparator.compute_cross_spectra([("c", "b")], progress=False)
        cross = comparator.cross_spectra_
        assert cross.gene_names == ("b", "c")
        expected = np.ones((8, 2), dtype=bool)
        expected[0, 0] = expected[1, 1] = False
        np.testing.assert_array_equal(cross.presence, expected)
        selected = [7, 0, 2, 7, 1]
        np.testing.assert_array_equal(
            comparator.subset(selected).cross_spectra_.presence, expected[selected]
        )
        actual = comparator.test_cross_diff_freq(np.repeat([0, 1], 4))
        assert actual.N_samples.iloc[0] == 6
        assert actual.n_obs_A.iloc[0] == 2
        assert actual.n_obs_B.iloc[0] == 4
        assert actual.Status.iloc[0] == "ok"
        comparator.compute_spectra(progress=False)
        np.testing.assert_array_equal(cross.presence, comparator.presence_[:, [1, 2]])


@pytest.mark.parametrize("representation", ["coherence", "coherency"])
@pytest.mark.parametrize("glm", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("jitter", [0.0, 1e-15, 1e-3])
def test_df_and_cross_share_degenerate_null_policy(representation, glm, masked, jitter):
    rng = np.random.default_rng(962)
    values = np.full((8, 3, 3), 0.4)
    values[:4, 0, :2] = [0.7, 0.6]
    values[4:, 0, :2] = [0.2, 0.3]
    values[:, 0, :2] += jitter * rng.normal(size=(8, 2))
    comparator = _comparison(values)
    cross = comparator.cross_spectra_
    if masked:
        cross.presence[0, 0] = False
        cross.presence[:, 4] = False
    comparator.presence_ = _mask_missing_pairs(cross)
    comparator.spectra_ = values**2 if representation == "coherence" else np.exp(values)
    comparator.gene_names = [pair[0] for pair in cross.pairs]
    groups = np.repeat([0, 1], 4)
    design = pd.DataFrame({"condition": groups}) if glm else groups
    kwargs = {"contrast": "condition"} if glm else {}
    kwargs["covariance"] = "pooled"
    kwargs["freq_weights"] = np.array([1.0, 2.0, 0.5])
    auto = (
        comparator.test_diff_freq(design, **kwargs).set_index("Feature").loc[comparator.gene_names]
    )
    pair = _ordered(
        comparator.test_cross_diff_freq(design, representation=representation, **kwargs)
    )
    np.testing.assert_allclose(
        auto[["Statistic", "P_value", "P_adj"]],
        pair[["Statistic", "P_value", "P_adj"]],
        atol=1e-12,
        equal_nan=True,
    )
    assert auto.P_value.iloc[1] == pair.P_value.iloc[1] == 1
    if jitter <= 1e-15:
        assert np.isnan(auto.P_value.iloc[0]) and np.isnan(auto.P_adj.iloc[0])
        assert pair.Status.iloc[0] == "degenerate_null"
    else:
        assert np.isfinite(auto.P_value.iloc[0])
        assert pair.Status.iloc[0] == "ok"


@pytest.mark.parametrize("null", ["rotation", "plugin", "permutation"])
@pytest.mark.parametrize("sizes,minimum", [((1, 3), 2), ((3, 2), 3)])
def test_binary_l2_minimum_is_independent_of_panel_masks(sizes, minimum, null):
    groups = np.repeat([0, 1], sizes)
    values = np.random.default_rng(720).uniform(0.1, 0.8, (len(groups), 2, 3))
    comparator = _comparison(values)
    cross = comparator.cross_spectra_
    comparator.spectra_ = values**2
    comparator.gene_names = [pair[0] for pair in cross.pairs]
    options = {"null": null, "min_samples_per_group": minimum}
    for masked in (False, True):
        cross.presence[0, 2] = not masked
        comparator.presence_ = _mask_missing_pairs(cross)
        for normalize in (False, True):
            with pytest.raises(ValueError, match=f"at least {minimum} observed"):
                comparator.test_diff_freq(groups, normalize_shape=normalize, **options)
        for representation in ("coherence", "coherency"):
            result = comparator.test_cross_diff_freq(
                groups, representation=representation, **options
            )
            assert result.Status.eq("insufficient_samples").all()
            expected_df = result.N_samples - result.n_obs_A.gt(0) - result.n_obs_B.gt(0)
            np.testing.assert_array_equal(result.DF_resid, expected_df)
            assert result[["Statistic", "P_value", "P_adj"]].isna().all(axis=None)


@pytest.mark.parametrize("null", ["rotation", "plugin", "permutation"])
def test_binary_l2_keeps_feature_at_group_minimum(null):
    groups = np.repeat([0, 1], [3, 4])
    values = np.random.default_rng(99).uniform(0.1, 0.8, (7, 2, 3))
    comparator = _comparison(values)
    cross = comparator.cross_spectra_
    comparator.spectra_ = values**2
    comparator.gene_names = [pair[0] for pair in cross.pairs]
    first = []
    for masked in (False, True):
        cross.presence[0, 2] = not masked
        comparator.presence_ = _mask_missing_pairs(cross)
        options = {"null": null, "min_samples_per_group": 3}
        tails = []
        for normalize in (False, True):
            result = comparator.test_diff_freq(groups, normalize_shape=normalize, **options)
            tails.append(result.set_index("Feature").loc["g0", "P_value"])
            assert result.P_value.notna().sum() == (1 if masked else 2)
        for representation in ("coherence", "coherency"):
            result = _ordered(
                comparator.test_cross_diff_freq(groups, representation=representation, **options)
            )
            tails.append(result.P_value.iloc[0])
            assert result.Status.iloc[1] == ("insufficient_samples" if masked else "ok")
        first.append(tails)
    np.testing.assert_allclose(first[0], first[1], rtol=1e-12)


@pytest.mark.parametrize(
    "glm,null",
    [
        (False, "analytic"),
        (True, "analytic"),
        (False, "rotation"),
        (False, "plugin"),
        (False, "permutation"),
        (True, "rotation"),
        (True, "plugin"),
    ],
)
def test_profile_tests_reject_zero_bin_weights(glm, null):
    groups = np.repeat([0, 1], 3)
    values = np.ones((6, 2, 3))
    comparator = _comparison(values)
    comparator.spectra_ = values
    comparator.gene_names = ["a", "b"]
    comparator.cross_spectra_.presence[0, 2] = False
    comparator.presence_ = _mask_missing_pairs(comparator.cross_spectra_)
    options = {"null": null, "freq_weights": [1, 0, 2]}
    design = pd.DataFrame({"age": np.arange(6)}) if glm else groups
    if glm:
        options["contrast"] = "age"
    if null == "analytic":
        options["statistic"] = "log_t_cauchy"
    for normalize in (False, True):
        with pytest.raises(ValueError, match="strictly positive"):
            comparator.test_diff_freq(design, normalize_shape=normalize, **options)
    comparator.cross_spectra_.coherency_valid[..., 1] = False
    if null == "analytic":
        options["statistic"] = "t_cauchy"
    for representation in ("coherence", "coherency"):
        with pytest.raises(ValueError, match="strictly positive"):
            comparator.test_cross_diff_freq(design, representation=representation, **options)


@pytest.mark.parametrize("representation", ["coherence", "coherency"])
@pytest.mark.parametrize("glm", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_per_pair_matches_independent_ols_with_sample_and_bin_masks(representation, glm, masked):
    rng = np.random.default_rng(711)
    n, p, b = 12, 5, 19
    values = rng.uniform(0.1, 0.8, (n, p, b)) * np.exp(1j * rng.uniform(-np.pi, np.pi, (n, p, b)))
    comparator = _comparison(values)
    cross = comparator.cross_spectra_
    cross.coherency_valid[0, :2, 1] = False
    cross.coherency_valid[1, 2:4, 2:4] = False
    if masked:
        cross.presence[0, :2] = False
        cross.presence[8, 2:4] = False
        cross.presence[:, -2:] = False
    presence = _mask_missing_pairs(cross)
    weights = np.arange(1, b + 1, dtype=float)
    support = (cross.coherency_valid | ~presence[..., None]).all(0) & (weights > 0)
    support &= presence.any(0)[:, None]
    groups = np.repeat([0, 1], n // 2)
    design = pd.DataFrame({"group": groups, "age": rng.normal(size=n)}) if glm else groups
    matrix = (
        np.column_stack([np.ones(n), design.to_numpy()])
        if glm
        else np.column_stack([np.ones(n), groups])
    )
    contrast = np.array([0.0, 0.0, 1.0]) if glm else np.array([0.0, 1.0])
    kwargs = {"contrast": "age"} if glm else {}
    # Per-pair mode never constructs or repairs the entrywise pooled matrix.
    with patch(
        "sonic.comparators.cross_spectrum._stabilize_cross_covariance",
        side_effect=AssertionError("unexpected pooling"),
    ):
        with patch.object(
            l2, "_init_per_gene_covariance", wraps=l2._init_per_gene_covariance
        ) as init:
            frame = comparator.test_cross_diff_freq(
                design,
                representation=representation,
                covariance="per_pair",
                freq_weights=weights,
                **kwargs,
            )
            assert init.call_count
            assert all(call.args[2] is False for call in init.call_args_list)
    assert frame.attrs["covariance"] == "per_pair"
    assert frame.attrs["null"] == "plugin"
    actual = _ordered(frame).loc[list(cross.pairs)]
    profiles = _profiles(values, representation)
    for j, pair in enumerate(cross.pairs):
        row = actual.loc[pair]
        keep = presence[:, j]
        bins = np.flatnonzero(support[j])
        assert actual.attrs["sample_indices"][pair] == tuple(np.flatnonzero(keep))
        assert actual.attrs["bin_indices"][pair] == tuple(bins)
        if not keep.any():
            assert np.isnan(row.P_value)
            assert row.N_pairs_pooled == 0
            continue
        x = matrix[keep]
        y = profiles[keep, j][:, bins]
        fit = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = y - x @ fit
        df = len(x) - np.linalg.matrix_rank(x)
        covariance = residual.T @ residual / df
        w = weights[bins] / weights[bins].sum()
        eigen = np.maximum(np.linalg.eigvalsh(np.sqrt(w[:, None] * w) * covariance), 0)
        scale = np.sum((contrast @ np.linalg.pinv(x)) ** 2)
        expected_stat = np.sqrt(np.sum(w * (contrast @ fit) ** 2))
        np.testing.assert_allclose(row.Statistic, expected_stat, atol=1e-12)
        np.testing.assert_allclose(row.P_value, liu_sf(expected_stat**2, scale * eigen), atol=1e-12)
        np.testing.assert_allclose(
            row.Effective_rank, eigen.sum() ** 2 / np.sum(eigen**2), atol=1e-12
        )
        assert row.DF_resid == df
        assert row.N_samples == keep.sum()
        assert row.N_bins == len(bins)
        assert row.N_pairs_pooled == 1
        assert row.Status == "ok"
    np.testing.assert_allclose(actual.P_adj, apply_bh_correction(actual.P_value), equal_nan=True)
    plugin = comparator.test_cross_diff_freq(
        design, representation=representation, null="plugin", freq_weights=weights, **kwargs
    )
    pd.testing.assert_frame_equal(plugin, frame)
    assert plugin.attrs == frame.attrs


@pytest.mark.parametrize("representation", ["coherence", "coherency"])
@pytest.mark.parametrize("glm", [False, True])
def test_per_pair_degeneracy_and_ineligible_statuses(representation, glm):
    rng = np.random.default_rng(518)
    groups = np.repeat([0, 1], 4)
    values = rng.uniform(0.1, 0.8, (8, 8, 5))
    values[:, 1] = 0.4  # Zero residual covariance and zero effect.
    values[:, 2] = np.where(groups[:, None] == 0, 0.2, 0.6)  # Perfect nonzero fit.
    comparator = _comparison(values)
    cross = comparator.cross_spectra_
    pairs = list(cross.pairs)
    pairs[3] = ("g6", "g6")
    cross.pairs = tuple(pairs)
    cross.pair_indices[3] = [6, 6]
    cross.coherency_valid[:, 4] = False
    cross.presence[:4, 10] = False  # No control arm; GLM contrast unestimable.
    cross.presence[1:, 12] = False  # No residual df.
    cross.presence[:, 14] = False  # No observed samples.
    _mask_missing_pairs(cross)
    design = pd.DataFrame({"group": groups}) if glm else groups
    kwargs = {"contrast": "group"} if glm else {}
    actual = _ordered(
        comparator.test_cross_diff_freq(
            design, representation=representation, covariance="per_pair", **kwargs
        )
    ).loc[list(cross.pairs)]
    skipped = "ineligible_design" if glm else "insufficient_samples"
    assert list(actual.Status) == [
        "ok",
        "ok",
        "degenerate_null",
        "self_pair",
        "no_common_bins",
        skipped,
        skipped,
        skipped,
    ]
    assert np.isfinite(actual.P_value.iloc[0])
    assert actual.P_value.iloc[1] == 1.0
    assert actual.P_value.iloc[2:].isna().all()
    assert actual.P_adj.iloc[2:].isna().all()
    np.testing.assert_array_equal(actual.N_pairs_pooled, [1, 1, 1, 0, 0, 0, 0, 0])


@pytest.mark.parametrize("representation", ["coherence", "coherency"])
@pytest.mark.parametrize("glm", [False, True])
def test_per_pair_raw_results_do_not_borrow_from_other_pairs(representation, glm):
    rng = np.random.default_rng(831)
    comparator = _comparison(rng.uniform(0.1, 0.8, (10, 4, 13)))
    cross = comparator.cross_spectra_
    cross.coherency_valid[0, 0, :2] = False
    cross.presence[0, 0] = False
    _mask_missing_pairs(cross)
    groups = np.repeat([0, 1], 5)
    design = pd.DataFrame({"group": groups, "age": rng.normal(size=10)}) if glm else groups
    kwargs = {"contrast": "age"} if glm else {}
    kwargs.update(representation=representation, covariance="per_pair")
    full = _ordered(comparator.test_cross_diff_freq(design, **kwargs))
    # Retain the same pair's samples, bins, and design while removing the rest.
    cross.cross_spectra = cross.cross_spectra[:, :1]
    cross.coherency_valid = cross.coherency_valid[:, :1]
    cross.pairs = cross.pairs[:1]
    cross.pair_indices = cross.pair_indices[:1]
    alone = _ordered(comparator.test_cross_diff_freq(design, **kwargs))
    columns = ["Statistic", "P_value", "Effective_rank", "DF_resid"]
    np.testing.assert_allclose(alone[columns], full.loc[alone.index, columns], atol=1e-12)
    # A singleton pooled fit uses the same covariance as the per-pair fit.
    pooled = _ordered(
        comparator.test_cross_diff_freq(design, **(kwargs | {"covariance": "pooled"}))
    )
    np.testing.assert_allclose(alone[columns], pooled[columns], atol=1e-12)


@pytest.mark.parametrize("representation", ["coherence", "coherency"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize(
    "glm,null,covariance",
    [
        (False, "plugin", None),
        (False, "plugin", "per_pair"),
        (False, "rotation", None),
        (False, "permutation", None),
        (True, "plugin", None),
        (True, "plugin", "per_pair"),
        (True, "rotation", None),
    ],
)
def test_l2_bin_estimates_match_t_cauchy(representation, masked, glm, null, covariance):
    rng = np.random.default_rng(729)
    values = rng.uniform(-0.7, 0.7, (10, 5, 6)) + 0.15j
    values[1, 0, 0] = 0
    comparator = _comparison(values)
    cross = comparator.cross_spectra_
    groups = np.tile(["B", "A"], 5)
    weights = np.array([1, 1, 2, 3, 1, 4])
    cross.coherency_valid[0, 0, [0, 2]] = False
    cross.coherency_valid[:, 1] = False
    cross.pairs = (*cross.pairs[:-1], ("g8", "g8"))
    cross.pair_indices[-1] = [8, 8]
    if masked:
        cross.presence[0, 0] = False
        cross.presence[1:, 6] = False
        _mask_missing_pairs(cross)
    design = np.c_[np.ones(10), np.arange(10), rng.normal(size=10)] if glm else groups
    options = {"representation": representation, "freq_weights": weights}
    if glm:
        options["contrast"] = np.array([0, -2, 0.5])
    l2_options = {"null": null, "covariance": covariance}
    if null == "permutation":
        l2_options.update(random_state=781, n_perm=99, max_exact_permutations=0)
    actual = _ordered(comparator.test_cross_diff_freq(design, **l2_options, **options))
    reference = _ordered(comparator.test_cross_diff_freq(design, statistic="t_cauchy", **options))
    effects = np.stack(actual.Estimate_per_bin)
    np.testing.assert_allclose(effects, np.stack(reference.Estimate_per_bin), atol=1e-12)
    assert np.isnan(effects[[1, 4]]).all()
    if masked:
        assert np.isnan(effects[3]).all()
    assert np.isfinite(effects[0, 3])
    for effect, row in zip(effects, actual.itertuples(), strict=True):
        tested = np.isfinite(effect) & (weights > 0)
        if tested.any():
            assert row.Statistic == pytest.approx(
                np.sqrt(np.average(effect[tested] ** 2, weights=weights[tested]))
            )
            assert row.N_bins == tested.sum()


class TestCrossRotation:
    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    def test_streaming_and_pair_order(self, monkeypatch, representation):
        import sonic.comparators.cross_spectrum as cs

        values = np.random.default_rng(18).uniform(-0.8, 0.8, (10, 8, 7))
        comparator = _comparison(values)
        design = np.c_[np.ones(10), np.linspace(-1, 1, 10)]
        kwargs = {
            "contrast": np.array([0.0, 1.0]),
            "representation": representation,
            "null": "rotation",
            "integration_tol": 5e-11,
        }
        expected = comparator.test_cross_diff_freq(design, **kwargs)
        prepared_sizes = []
        prepare = cs._prepare_cross_profiles

        def record(result, pairs, *args):
            prepared_sizes.append(len(pairs))
            return prepare(result, pairs, *args)

        monkeypatch.setattr(cs, "_prepare_cross_profiles", record)
        monkeypatch.setattr(l2, "_COMPARISON_WORKSPACE_BYTES", 4096)
        actual = comparator.test_cross_diff_freq(design, min_resid_df=100, **kwargs)
        pd.testing.assert_frame_equal(actual, expected)
        assert max(prepared_sizes) == 1
        assert actual.attrs == expected.attrs

        cross = comparator.cross_spectra_
        order = np.arange(len(cross.pairs))[::-1]
        cross.cross_spectra = cross.cross_spectra[:, order]
        cross.coherency_valid = cross.coherency_valid[:, order]
        cross.pair_indices = cross.pair_indices[order]
        cross.pairs = tuple(cross.pairs[i] for i in order)
        reordered = comparator.test_cross_diff_freq(design, **kwargs)
        pd.testing.assert_frame_equal(_ordered(reordered), _ordered(expected))
        assert reordered.attrs == expected.attrs

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    @pytest.mark.parametrize("glm", [False, True])
    def test_single_retained_bin_matches_extreme_student_t(self, representation, glm):
        groups = np.repeat([0, 1], 9)
        x = np.c_[np.ones(18), groups]
        if glm:
            x = np.c_[x, np.linspace(-1, 1, 18)]
        c = np.eye(x.shape[1])[1]
        y = np.random.default_rng(29).uniform(-0.3, 0.3, (18, 2, 3))
        expected, dfs = [], []
        for pair, (bin_index, score) in enumerate(zip([0, 2], [20, 100], strict=True)):
            rows = np.arange(pair, 18)
            subx = x[rows]
            df = len(rows) - x.shape[1]
            residual = np.linalg.svd(subx, full_matrices=True)[0][:, x.shape[1]]
            se = np.linalg.norm(c @ np.linalg.pinv(subx)) / np.sqrt(df)
            # A unit residual direction gives an exact, prespecified t statistic.
            y[rows, pair, bin_index] = 0.2 * (subx[:, 1] + residual / (score * se))
            if glm:
                y[rows, pair, bin_index] += 0.1 * subx[:, 2]
            expected.append(2 * t.sf(score, df))
            dfs.append(df)
        values = np.exp((y - 2) / 2 + 0.4j) if representation == "coherence" else y + 0.2j
        comparator = _comparison(values)
        cross = comparator.cross_spectra_
        cross.presence[0, 2] = False
        _mask_missing_pairs(cross)
        cross.coherency_valid[..., 1] = False
        cross.coherency_valid[0, 0, 2] = False
        cross.coherency_valid[1, 1, 0] = False
        result = _ordered(
            comparator.test_cross_diff_freq(
                x if glm else groups,
                **({"contrast": c} if glm else {}),
                representation=representation,
                freq_weights=np.array([2.0, 1.0, 5.0]),
            )
        )
        np.testing.assert_allclose(result.P_value, expected, rtol=2e-11, atol=0)
        np.testing.assert_allclose(result.Statistic, 0.2, atol=1e-14)
        np.testing.assert_array_equal(result.DF_resid, dfs)
        assert (result.N_bins == 1).all()
        assert list(result.attrs["bin_indices"].values()) == [(0,), (2,)]
        assert (result.P_value_error <= np.minimum(1e-10, 0.001 * result.P_value)).all()
        np.testing.assert_allclose(result.P_adj, apply_bh_correction(expected), rtol=2e-11, atol=0)

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    def test_multibin_low_df_reports_accuracy_against_beta(self, representation):
        groups = np.array([0, 0, 0, 1, 1])
        x = np.c_[np.ones(5), groups]
        y = np.zeros((5, 2, 3))
        y[:, :, 1] = np.linspace(-0.4, 0.4, 5)[:, None]
        expected, statistics = [], []
        for pair, q in enumerate([0.25, 0.9]):
            rows = np.arange(pair, 5)
            subx = x[rows]
            a = np.array([0.0, 1.0]) @ np.linalg.pinv(subx)
            residual = np.linalg.svd(subx, full_matrices=True)[0][:, 2:]
            basis = np.column_stack([a / np.linalg.norm(a), residual])
            coordinates = np.zeros((basis.shape[1], 2))
            coordinates[0, 0], coordinates[-1, 0] = np.sqrt(q), np.sqrt(1 - q)
            coordinates[1, 1] = 1
            # Two equal eigenvalues: sphere projection energy has a beta law.
            profiles = 0.2 * (basis @ coordinates) / np.sqrt([0.25, 0.75])
            y[rows, pair, 0], y[rows, pair, 2] = profiles.T
            expected.append(beta.sf(q, 1, (basis.shape[1] - 2) / 2))
            statistics.append(0.2 * np.linalg.norm(a) * np.sqrt(q))
        values = np.exp((y - 2) / 2 + 0.4j) if representation == "coherence" else y + 0.2j
        comparator = _comparison(values)
        comparator.cross_spectra_.presence[0, 2] = False
        _mask_missing_pairs(comparator.cross_spectra_)
        comparator.cross_spectra_.coherency_valid[..., 1] = False
        result = _ordered(
            comparator.test_cross_diff_freq(
                groups,
                representation=representation,
                null="rotation",
                freq_weights=np.array([1.0, 2.0, 3.0]),
            )
        )
        for row, reference in zip(result.itertuples(), expected, strict=True):
            if row.P_value_method == "saddlepoint":
                # Roundoff separates repeated levels; low df can exhaust Davies' budget.
                assert np.isnan(row.P_value_error)
                assert abs(row.P_value / reference - 1) < 0.03
            else:
                assert abs(row.P_value - reference) <= row.P_value_error + 1e-12
        np.testing.assert_allclose(result.Statistic, statistics, atol=1e-14)
        np.testing.assert_array_equal(result.DF_resid, [3, 2])
        assert (result.N_bins == 2).all()
        assert (result.P_value_error.dropna() <= 1e-6).all()

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    @pytest.mark.parametrize("glm", [False, True])
    @pytest.mark.parametrize("masked", [False, True])
    def test_matches_shared_engine_on_retained_samples_and_bins(self, representation, glm, masked):
        values = np.random.default_rng(183).uniform(-0.8, 0.8, (10, 3, 7))
        comparator = _comparison(values)
        cross = comparator.cross_spectra_
        if masked:
            cross.presence[0, 0] = False
            cross.presence[1, 2] = False
            cross.presence[1:, 4] = False
        presence = _mask_missing_pairs(cross)
        cross.coherency_valid[0, 1, 2] = False
        original = copy.deepcopy(cross)
        groups = np.repeat([0, 1], 5)
        x = np.c_[np.ones(10), np.linspace(-1, 1, 10) if glm else groups]
        c = np.array([0.0, 1.0])
        kwargs = {"contrast": c} if glm else {}
        weights = np.arange(1, 8, dtype=float)
        result = _ordered(
            comparator.test_cross_diff_freq(
                x if glm else groups,
                representation=representation,
                null="rotation",
                freq_weights=weights,
                **kwargs,
            )
        )
        transformed = _profiles(values, representation)
        for i, pair in enumerate(cross.pairs):
            rows = np.flatnonzero(presence[:, i])
            bins = np.array(result.attrs["bin_indices"][pair])
            expected_bins = np.flatnonzero(
                original.coherency_valid[rows, i].all(axis=0) & (weights > 0)
            )
            np.testing.assert_array_equal(bins, expected_bins)
            np.testing.assert_array_equal(result.attrs["sample_indices"][pair], rows)
            if masked and i == 2:
                assert np.isnan(result.loc[pair, "P_value"])
                assert np.isnan(result.loc[pair, "P_value_error"])
                continue
            reference = l2._run_l2_conditional(
                transformed[:, i : i + 1, bins],
                x,
                c,
                presence[:, i : i + 1] if masked else None,
                freq_weights=weights[bins],
                log_transform=False,
            )
            np.testing.assert_allclose(
                result.loc[pair, ["Statistic", "P_value"]].to_numpy(dtype=float),
                [reference["observed"][0], reference["pvals"][0]],
                atol=1e-12,
            )
            assert (
                result.loc[pair, "P_value_error"] <= 1e-6
                or result.loc[pair, "P_value_method"] == "saddlepoint"
            )
            # Integration must agree with independent draws within sampling uncertainty.
            y = transformed[rows, i][:, bins]
            inverse = np.linalg.pinv(x[rows])
            a = c @ inverse
            nuisance = x[rows] @ inverse - np.outer(a, a) / (a @ a)
            seed = np.random.SeedSequence(39, spawn_key=(len(x), *map(int, rows)))
            directions = np.random.default_rng(seed).normal(size=(499, len(rows)))
            directions -= directions @ nuisance
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            w = weights[bins] / weights[bins].sum()
            null = np.sum(w * (directions @ y) ** 2, axis=1)
            observed = np.sum(w * (a @ y) ** 2) / (a @ a)
            trace = np.sum(w * (y - nuisance @ y) ** 2)
            count = np.count_nonzero(null >= observed - 64 * np.finfo(float).eps * trace)
            pvalue = result.loc[pair, "P_value"]
            bound = 6 * np.sqrt(pvalue * (1 - pvalue) / 499) + 3 / 499
            assert abs(pvalue - (count + 1) / 500) <= bound
        assert result.attrs["null"] == "rotation"
        assert result.attrs["covariance"] is None
        assert result.attrs["method"] == "davies"
        assert result.attrs["integration_tol"] == 1e-6
        assert not {"n_rotations", "random_state", "min_p_value"} & result.attrs.keys()
        np.testing.assert_array_equal(result.attrs["freq_weights"], weights / weights.sum())
        assert (result.N_pairs_pooled == 0).all()
        np.testing.assert_allclose(result.P_adj, apply_bh_correction(result.P_value))
        for name in ("cross_spectra", "auto_spectra", "presence", "coherency_valid"):
            np.testing.assert_array_equal(getattr(cross, name), getattr(original, name))

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    @pytest.mark.parametrize("masked", [False, True])
    def test_nuisance_only_profiles_have_no_rotation_signal(self, representation, masked):
        rng = np.random.default_rng(7)
        groups = np.tile([0, 1], 6)
        z = rng.integers(-8, 9, size=12) / 16
        h = rng.integers(0, 2, size=12)
        x = np.c_[np.ones(12), groups, z, z + h / 4096]
        # Each pair has just two levels, also after the coherence log transform.
        values = np.stack([0.25 + h / 4, 0.25 + groups / 4], axis=1)[:, :, None]
        comparator = _comparison(values)
        if masked:
            comparator.cross_spectra_.presence[0, 0] = False
            _mask_missing_pairs(comparator.cross_spectra_)
        result = _ordered(
            comparator.test_cross_diff_freq(
                x,
                contrast=np.array([0.0, 1.0, 0.0, 0.0]),
                representation=representation,
            )
        )
        np.testing.assert_array_equal(result.P_value, [1.0, 0.0])
        assert (result.P_value_method == "exact").all()
        assert np.isnan(result.Effective_rank.iloc[0])
        assert result.Effective_rank.iloc[1] == pytest.approx(1.0)
        assert (result.Status == "ok").all()

    @pytest.mark.parametrize("with_nuisance", [False, True], ids=["binary", "glm"])
    def test_exact_coherency_contrast_survives_shared_offset(self, with_nuisance):
        groups = np.tile([0, 1], 6)
        step = 2.0**-50
        profiles = np.zeros((len(groups), 2, 2))
        profiles[:, 0] = groups[:, None] * np.array([-step, step])
        design, options = groups, {}
        if with_nuisance:
            nuisance = np.repeat(np.arange(6), 2) / 32
            profiles += nuisance[:, None, None]
            design = np.column_stack([np.ones(len(groups)), groups, nuisance])
            options = {"contrast": np.array([0.0, 1.0, 0.0])}

        for offset in (0.0, 0.5):
            result = _ordered(
                _comparison(profiles + offset).test_cross_diff_freq(
                    design, representation="coherency", **options
                )
            )
            np.testing.assert_allclose(result.Statistic, [step, 0.0], rtol=1e-14, atol=0)
            np.testing.assert_array_equal(result.P_value, [0.0, 1.0])
            np.testing.assert_array_equal(result.P_value_error, [0.0, 0.0])
            assert (result.P_value_method == "exact").all()

    def test_noisy_coherency_contrast_after_nuisance_recoding(self):
        groups = np.tile([0, 1], 6)
        z, w = np.random.default_rng(16).integers(-8, 9, size=(2, 12)) / 16
        pattern = 8 * groups
        pattern[0] += 1
        contrast = np.array([0, 1, 0, 0])
        design = np.c_[np.ones(12), groups, z, w]
        # Remove the exact nuisance and common scale analytically for the t reference.
        inverse = np.linalg.pinv(design)
        beta = inverse @ pattern
        se = np.linalg.norm(inverse[1]) * np.linalg.norm(pattern - design @ beta) / np.sqrt(8)
        expected = 2 * t.sf(abs(beta[1]) / se, 8)
        # Duplicate bins still span one direction even with unequal weights.
        profiles = np.repeat((w / 2 + pattern * 2.0**-53)[:, None, None], 2, axis=2)
        comparator = _comparison(profiles)
        for nuisance in (w, z + w / 512, z + w * 2.0**-44):
            design[:, -1] = nuisance
            row = comparator.test_cross_diff_freq(
                design, contrast=contrast, representation="coherency", freq_weights=[1, 3]
            ).iloc[0]
            np.testing.assert_allclose(row.P_value, expected, rtol=1e-10, atol=0)
            assert row.P_value_method == "beta"

    @pytest.mark.parametrize("order", [[0, 1, 2, 3], [2, 0, 3, 1]])
    def test_constant_and_one_ulp_nonconstant_rotation(self, order):
        groups = np.array([0, 0, 1, 1])
        design = np.c_[np.ones(4), groups, [-1, 1, -1, 1]]
        a, r = 2 * groups - 1, np.array([1, -1, -1, 1])
        constant = np.column_stack([a, r]) / 4
        nearby = constant.copy()
        nearby[0, 0] = np.nextafter(nearby[0, 0], np.inf)
        delta = nearby[0, 0] - constant[0, 0]
        # The circle coordinates change from (1,0), (0,1) to (1-d,0), (d,1).
        # The common factor of 2d cancels from the circle-tail angle.
        expected = np.arctan2(1 - delta, -1) / np.pi
        profiles = np.stack([constant, nearby], axis=1)
        result = _ordered(
            _comparison(profiles[order]).test_cross_diff_freq(
                design[order], contrast=np.array([0, 1, 0]), representation="coherency"
            )
        )
        assert result.P_value.iloc[0] == 1
        assert result.P_value_error.iloc[0] == 0
        assert result.P_value_method.tolist() == ["exact", "beta"]
        assert abs(result.P_value.iloc[1] - expected) <= result.P_value_error.iloc[1]

    def test_constant_rotation_boundaries_with_recoded_nuisance(self):
        intercept, a, z, w, x, y, r, s = hadamard(8)
        design = np.column_stack([intercept, a, z, z + w * 2.0**-40, x, y])
        # a, r and s span the sphere. Change only the energy along r by one ULP.
        levels = [0.25, np.nextafter(0.25, np.inf), np.nextafter(0.25, 0)]
        profiles = np.stack(
            [np.column_stack([a / 4, r * level, s / 4]) for level in levels], axis=1
        )
        result = _ordered(
            _comparison(profiles).test_cross_diff_freq(
                design, contrast=np.array([0, 1, 0, 0, 0, 0]), representation="coherency"
            )
        )
        np.testing.assert_array_equal(result.P_value, [1, 1, 0])
        np.testing.assert_array_equal(result.P_value_error, 0)
        assert result.P_value_method.tolist() == ["exact"] * 3

    def test_weighted_rotation_tie_preserves_ratios_after_bin_selection(self):
        a = np.array([-1, -1, 1, 1])
        r = np.array([1, -1, -1, 1])
        design = np.column_stack([np.ones(4), a, a * r])
        comp = _comparison(np.column_stack([3 * a, r, r])[:, None, :] / 4)
        comp.cross_spectra_.coherency_valid[0, 0, 1] = False
        options = {"contrast": np.array([0, 1, 0]), "representation": "coherency"}
        # The retained energies are 9 and w. Every rotation ties for w=9.
        large = 2**53 + 1
        for weights, expected in [
            (None, 0),
            ([1, 10, 9], 1),
            ([large, 10 * large, 9 * large], 1),
            ([1, 10, np.nextafter(9.0, 0)], 0),
        ]:
            result = comp.test_cross_diff_freq(design, freq_weights=weights, **options)
            row = result.iloc[0]
            assert (row.P_value, row.P_value_error, row.P_value_method) == (expected, 0, "exact")
            saved = result.attrs["original_freq_weights"]
            assert saved == (None if weights is None else tuple(weights))
            replay = comp.test_cross_diff_freq(design, freq_weights=saved, **options)
            pd.testing.assert_frame_equal(replay, result, check_exact=True)

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    def test_rotation_with_duplicate_nearly_duplicate_or_constant_bin(self, representation):
        a = np.array([-1, -1, 1, 1])
        r = np.array([1, -1, -1, 1])
        expected = 0.049
        ratio = (1 - expected) / np.sqrt(expected * (2 - expected))
        signal = 0.05 * (ratio * a + r)
        y = np.exp((signal - 1) / 2) if representation == "coherence" else 0.5 + signal
        companions = [y, np.ones(4), y + 1e-12 * a * r]
        profiles = np.stack([np.column_stack([y, other]) for other in companions], axis=1)
        for rows in ([0, 1, 2, 3], [0, 2, 1, 3]):
            frame = _ordered(
                _comparison(profiles[rows]).test_cross_diff_freq(
                    a[rows], representation=representation, freq_weights=[1, 3]
                )
            )
            assert (frame.P_value_method == "beta").all()
            np.testing.assert_allclose(frame.P_value.iloc[:2], expected, rtol=1e-13)
            np.testing.assert_allclose(frame.P_value.iloc[2], expected, atol=1e-12, rtol=0)

    def test_isolated_endpoint_shift_matches_beta_tail(self):
        axes = hadamard(16)[1:].T
        delta = 2.0**-15
        profiles = axes / 8
        profiles[:, 0] = axes[:, 0] / 2 + delta * axes[:, 1]
        # Only the first two directions mix. Solve their shifted eigenvalues
        # without subtracting the nearly equal largest eigenvalue and statistic.
        gap = 1 / 4 - 1 / 64 - delta**2
        positive = delta**2 / (2 * (np.hypot(gap, delta) + gap))
        expected = beta.cdf(positive / (positive + 1 / 4 - 1 / 64), 7, 0.5)
        row = (
            _comparison(profiles[:, None])
            .test_cross_diff_freq(axes[:, 0], representation="coherency")
            .iloc[0]
        )
        np.testing.assert_allclose(row.P_value, expected, rtol=1e-7, atol=0)

    def test_repeated_rotation_endpoints_and_nearby_interior(self):
        a = np.array([-1, -1, 1, 1])
        r = np.array([1, -1, -1, 1])
        z = a * r
        upper = np.column_stack([a / 2, r / 2, z / 4])
        lower = np.column_stack([a / 4, r / 4, z / 2])
        delta = 2.0**-40
        nearby = upper.copy()
        nearby[:, 0] += delta * z
        # The r direction still ties. The remaining two directions give a circle tail.
        expected = np.arctan2(delta, 3 / 16 - delta**2) / np.pi
        profiles = np.stack([upper, lower, nearby], axis=1)
        for rows in ([0, 1, 2, 3], [2, 3, 0, 1]):
            frame = _ordered(
                _comparison(profiles[rows]).test_cross_diff_freq(
                    a[rows], representation="coherency"
                )
            )
            assert frame.P_value_method.tolist() == ["exact", "exact", "beta"]
            np.testing.assert_array_equal(frame.P_value.iloc[:2], [0, 1])
            np.testing.assert_array_equal(frame.P_value_error.iloc[:2], 0)
            np.testing.assert_allclose(frame.P_value.iloc[2], expected, rtol=1e-12, atol=0)

    def test_two_bin_rotation_matches_circle_tail_after_offset(self):
        groups = np.array([0, 0, 1, 1])
        design = np.c_[np.ones(4), groups, [-1, 1, -1, 1]]
        a, r = np.array([-1, -1, 1, 1]), np.array([1, -1, -1, 1])
        pattern = a[:, None] * [5, 9] + r[:, None] * [2, -1]
        weights = np.array([1, 3])
        # In two sample dimensions the tail is the angle between equal-level directions / pi.
        u, v = a @ pattern, r @ pattern
        expected = (
            np.arctan2(2 * abs((u * weights) @ v), (u * weights) @ u - (v * weights) @ v) / np.pi
        )
        for offset in (0.0, 0.5):
            values = offset + np.spacing(0.5) * pattern
            row = (
                _comparison(values[:, None])
                .test_cross_diff_freq(
                    design,
                    contrast=np.array([0, 1, 0]),
                    representation="coherency",
                    freq_weights=weights,
                )
                .iloc[0]
            )
            np.testing.assert_allclose(row.P_value, expected, atol=1e-14)

    def test_rotation_endpoint_with_fewer_bins_than_sample_directions(self):
        a = np.array([-1, -1, 1, 1])
        r = np.array([1, -1, -1, 1])
        pairs = np.stack([np.column_stack([a / 2, r / 2]), np.column_stack([a / 4, r / 2])], axis=1)
        frame = _ordered(_comparison(pairs).test_cross_diff_freq(a, representation="coherency"))
        # On the second pair the shifts are proportional to (0, 3, -1).
        assert frame.P_value_method.tolist() == ["exact", "beta"]
        assert frame.P_value.iloc[0] == 0
        np.testing.assert_allclose(frame.P_value.iloc[1], 2 / 3, rtol=1e-14)

    @pytest.mark.parametrize("glm", [False, True])
    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    def test_empty_support_and_perfect_fit(self, glm, representation):
        groups = np.tile([0, 1], 6)
        values = np.full((12, 3, 4), 0.3)
        values[:, 1, :2] = np.array([[0.25, 0.5], [0.5, 0.25]])[groups]
        comparator = _comparison(values)
        comparator.cross_spectra_.presence[:, 4] = False
        _mask_missing_pairs(comparator.cross_spectra_)
        design = np.c_[np.ones(12), groups] if glm else groups
        kwargs = {"contrast": np.array([0.0, 1.0])} if glm else {}
        result = _ordered(
            comparator.test_cross_diff_freq(
                design,
                representation=representation,
                null="rotation",
                freq_weights=np.array([1.0, 1.0, 2.0, 2.0]),
                **kwargs,
            )
        )
        np.testing.assert_array_equal(result.P_value, [1.0, 0.0, np.nan])
        assert result.P_value_method.iloc[:2].tolist() == ["exact", "exact"]
        assert np.isnan(result.P_value_error.iloc[-1])
        comparator.cross_spectra_.coherency_valid[:] = False
        empty = comparator.test_cross_diff_freq(
            design, null="rotation", integration_tol=5e-11, **kwargs
        )
        assert empty.P_value.isna().all()
        assert empty.P_value_error.isna().all()
        assert empty.attrs["method"] == "davies"
        assert empty.attrs["integration_tol"] == 5e-11

    @pytest.mark.parametrize("covariance", ["pooled", "per_pair"])
    def test_rejects_covariance_selection(self, covariance):
        comparator = _comparison(np.ones((6, 2, 3)))
        with pytest.raises(ValueError, match="covariance is only supported"):
            comparator.test_cross_diff_freq(
                np.repeat([0, 1], 3), null="rotation", covariance=covariance
            )

    def test_controls_self_pairs_and_integration_failure(self, monkeypatch):
        import sonic.comparators.cross_spectrum as cs

        comparator = _comparison(np.random.default_rng(195).uniform(-0.8, 0.8, (8, 2, 5)))
        cross = comparator.cross_spectra_
        cross.pairs = (("g0", "g0"), cross.pairs[1])
        cross.pair_indices[0, 1] = 0
        groups = np.repeat([0, 1], 4)
        options = {"null": "rotation", "integration_tol": 5e-11}
        result = comparator.test_cross_diff_freq(groups, random_state=1, **options)
        replay = comparator.test_cross_diff_freq(groups, random_state=2, **options)
        pd.testing.assert_frame_equal(result, replay)
        assert result.attrs == replay.attrs
        self_pair = result[result.Status == "self_pair"].iloc[0]
        assert np.isnan(self_pair.P_value) and np.isnan(self_pair.P_value_error)
        assert result.loc[result.Status == "ok", "P_value_error"].dropna().le(5e-11).all()
        with pytest.raises(TypeError, match="n_rotations"):
            comparator.test_cross_diff_freq(groups, null="rotation", n_rotations=99)

        def fail(*args, **kwargs):
            raise RuntimeError("unresolved integration")

        monkeypatch.setattr(cs, "_run_l2_conditional", fail)
        with pytest.raises(RuntimeError, match="unresolved integration"):
            comparator.test_cross_diff_freq(groups, **options)
        cross.coherency_valid[:] = False
        for tolerance in (0, True, np.nan):
            with pytest.raises(ValueError, match="integration_tol"):
                comparator.test_cross_diff_freq(groups, null="rotation", integration_tol=tolerance)

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    @pytest.mark.parametrize("glm", [False, True])
    def test_marked_fallback_with_masks_bins_and_self_pairs(self, representation, glm, monkeypatch):
        values = np.random.default_rng(811).uniform(0.1, 0.8, (10, 4, 5))
        values[:, 3] = 0.3
        comparator = _comparison(values)
        cross = comparator.cross_spectra_
        cross.presence[0, 0] = False
        _mask_missing_pairs(cross)
        cross.coherency_valid[:, 1] = False
        cross.coherency_valid[2, 0, 2] = False
        cross.pairs = (*cross.pairs[:2], ("g4", "g4"), cross.pairs[3])
        cross.pair_indices[2, 1] = 4
        groups = np.repeat([0, 1], 5)
        x = np.c_[np.ones(10), groups, np.linspace(-1, 1, 10)]
        monkeypatch.setattr(l2._davies, "sf", lambda *args: (0.0, 1))
        result = _ordered(
            comparator.test_cross_diff_freq(
                x if glm else groups,
                **({"contrast": np.array([0, 1, 0])} if glm else {}),
                representation=representation,
                null="rotation",
                freq_weights=np.array([1, 1, 2, 3, 4]),
            )
        )
        assert result.P_value_method.iloc[[0, 3]].tolist() == ["saddlepoint", "exact"]
        assert result.P_value_method.iloc[1:3].isna().all()
        assert result.P_value_error.iloc[:3].isna().all()
        assert result.P_value_error.iloc[3] == 0
        assert result.Status.tolist() == ["ok", "no_common_bins", "self_pair", "ok"]
        assert result.attrs["n_saddlepoint"] == 1
        assert result.attrs["integration_limit"] == 10000
        np.testing.assert_allclose(
            result.P_adj, apply_bh_correction(result.P_value), equal_nan=True
        )


class TestCrossTCauchy:
    """Weighted binwise inference on both cross-spectrum representations."""

    def test_coherency_exact_fits_and_one_ulp_residual(self):
        z = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1])
        x = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1])
        design = np.c_[np.ones(len(z)), z, x]
        contrast = np.array([0, 0, 1])
        step = np.spacing(0.5)
        impulse = np.eye(len(z))[2]
        profiles = np.column_stack([z / 2, z / 2 - step * x, z / 2 + step * impulse])
        # Removing the exact nuisance and dividing by one ULP leaves the unit impulse.
        inverse = np.linalg.pinv(design)
        beta = inverse @ impulse
        df = len(z) - 3
        se = np.linalg.norm(inverse[2]) * np.linalg.norm(impulse - design @ beta) / np.sqrt(df)
        for rows in (slice(None), slice(None, None, -1)):
            frame = _ordered(
                _comparison(profiles[rows, :, None]).test_cross_diff_freq(
                    design[rows],
                    contrast=contrast,
                    representation="coherency",
                    statistic="t_cauchy",
                )
            )
            null, perfect, nearby = frame.itertuples()
            assert null.T_per_bin[0] == null.Estimate_per_bin[0] == 0
            assert null.P_value_per_bin[0] == 1
            assert null.P_value == 1 and pd.isna(null.Best_bin)
            assert perfect.Estimate_per_bin[0] == -step
            assert perfect.T_per_bin[0] == -np.inf
            assert perfect.P_value_per_bin[0] == 0
            np.testing.assert_allclose(nearby.Estimate_per_bin, [step * beta[2]], atol=0)
            np.testing.assert_allclose(nearby.T_per_bin, [beta[2] / se])
            np.testing.assert_allclose(nearby.P_value, 2 * t.sf(abs(beta[2]) / se, df))

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    def test_nuisance_bin_excluded_with_masked_samples_and_bin_support(self, representation):
        x = np.repeat([0, 1], 6)
        z = np.tile([0, 0, 0, 1, 1, 1], 2)
        signal = 0.3 + 0.4 * x + np.tile([-0.05, 0, 0.05], 4)
        values = np.column_stack([0.25 + z / 4, signal, signal])
        comparator = _comparison(values[:, None, :])
        cross = comparator.cross_spectra_
        cross.presence[0, 0] = False
        _mask_missing_pairs(cross)
        cross.coherency_valid[1, 0, 1] = False
        row = comparator.test_cross_diff_freq(
            np.c_[np.ones(len(x)), x, z],
            contrast=np.array([0, 1, 0]),
            representation=representation,
            statistic="t_cauchy",
            freq_weights=[8, 4, 1],
        ).iloc[0]
        assert row.P_value_per_bin[0] == 1 and np.isnan(row.P_value_per_bin[1])
        assert row.Best_bin == 2
        assert row.P_value == pytest.approx(row.P_value_per_bin[2], rel=1e-12, abs=0)
        assert 0 < row.P_value < 0.05

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    @pytest.mark.parametrize("glm", [False, True])
    @pytest.mark.parametrize("masked", [False, True])
    def test_matches_independent_binwise_fits(self, representation, glm, masked):
        rng = np.random.default_rng(143)
        values = rng.uniform(-0.6, 0.6, (16, 4, 6)) + 0.2j
        comparator = _comparison(values)
        cross = comparator.cross_spectra_
        cross.coherency_valid[0, 0, 1] = False
        cross.coherency_valid[2, 1, 3] = False
        groups = np.repeat(["A", "B"], [5, 11])
        age = np.linspace(-1, 1, 16)
        nuisance = age + rng.normal(size=16)
        design = pd.DataFrame({"age": age, "batch": nuisance})
        matrix = np.c_[np.ones(16), age, nuisance]
        if masked:
            cross.presence[0, :2] = False
            cross.presence[6:8, 2:6] = False
        presence = _mask_missing_pairs(cross)
        original = copy.deepcopy(cross)
        weights = np.array([0.5, 1, 2, 3, 0.25, 4.0])
        args = {"contrast": "age"} if glm else {}
        result = _ordered(
            comparator.test_cross_diff_freq(
                design if glm else groups,
                representation=representation,
                statistic="t_cauchy",
                null="analytic",
                freq_weights=weights,
                **args,
            )
        )
        expected_p = []
        profiles = _profiles(values, representation)
        for pair, row in enumerate(result.itertuples()):
            mask = presence[:, pair]
            supported = (cross.coherency_valid[:, pair] | ~mask[:, None]).all(0)
            bins = np.flatnonzero(supported)
            y = profiles[mask, pair][:, bins]
            if glm:
                x = matrix[mask]
                beta = np.linalg.lstsq(x, y, rcond=None)[0]
                df = len(x) - np.linalg.matrix_rank(x)
                variance = ((y - x @ beta) ** 2).sum(0) / df
                estimate = beta[1]
                signed_t = estimate / np.sqrt(variance * np.sum(np.linalg.pinv(x)[1] ** 2))
                per_bin_p = 2 * t.sf(np.abs(signed_t), df)
            else:
                a, b = y[groups[mask] == "A"], y[groups[mask] == "B"]
                estimate = a.mean(0) - b.mean(0)
                signed_t, per_bin_p = ttest_ind(a, b, equal_var=False, axis=0)
                df = mask.sum() - 2
            for column, expected in zip(
                ("Estimate_per_bin", "T_per_bin", "P_value_per_bin"),
                (estimate, signed_t, per_bin_p),
                strict=True,
            ):
                actual = getattr(row, column)
                assert actual.shape == (6,)
                assert np.isnan(actual[~supported]).all()
                np.testing.assert_allclose(actual[bins], expected, rtol=1e-11, atol=1e-14)
            w = weights[bins] / weights[bins].sum()
            combined = np.sum(w / np.tan(np.pi * per_bin_p))
            p = 0.5 - np.arctan(combined) / np.pi
            expected_p.append(p)
            np.testing.assert_allclose(row.P_value, p, rtol=1e-10)
            np.testing.assert_allclose(row.Statistic, np.abs(signed_t).max())
            assert row.Best_bin == bins[np.argmin(per_bin_p)]
            assert row.DF_resid == df
            assert row.N_bins == len(bins)
            assert row.Status == "ok"
            assert result.attrs["bin_indices"][row.Index] == tuple(bins)
        np.testing.assert_allclose(result.P_adj, apply_bh_correction(expected_p), rtol=1e-10)
        assert result.Best_bin.dtype == pd.Int64Dtype()
        assert result.attrs["statistic"] == "t_cauchy"
        assert result.attrs["null"] == "analytic"
        assert result.attrs["per_bin_test"] == ("ols" if glm else "welch")
        assert result.attrs["covariance"] is None
        assert (result.N_pairs_pooled == 0).all()
        assert result.Effective_rank.isna().all()
        for name in ("cross_spectra", "auto_spectra", "coherency_valid", "presence"):
            np.testing.assert_array_equal(getattr(cross, name), getattr(original, name))

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    @pytest.mark.parametrize("glm", [False, True])
    def test_best_bin_keeps_original_index_and_first_tie(self, representation, glm):
        values = np.random.default_rng(144).uniform(0.1, 0.8, (12, 1, 6))
        values[:, :, 2:] = values[:, :, 2:3]
        comparator = _comparison(values)
        comparator.cross_spectra_.coherency_valid[:, :, :2] = False
        comparator.cross_spectra_.feature_mode = "polar"
        comparator.cross_spectra_.angle_centers_rad = np.array([0, np.pi / 2])
        groups = np.repeat([0, 1], 6)
        design = pd.DataFrame({"x": np.arange(12.0)})
        args = {"contrast": "x"} if glm else {}
        row = comparator.test_cross_diff_freq(
            design if glm else groups,
            statistic="t_cauchy",
            representation=representation,
            freq_weights=[1, 1, 0.25, 2, 5, 1],
            **args,
        ).iloc[0]
        assert row.Best_bin == 2
        assert np.isnan(row.P_value_per_bin[:2]).all()
        assert np.isfinite(row.P_value_per_bin[2:]).all()
        np.testing.assert_allclose(row.P_value, row.P_value_per_bin[2])

    @pytest.mark.parametrize("glm", [False, True])
    def test_combination_uses_informative_bins(self, glm):
        groups = np.repeat([0, 1], 6)
        values = np.full((12, 3, 3), 0.5)
        values[:, 1, 0] = 0.25 * groups
        values[:, 2, :] = 0.25 * groups[:, None]
        comparator = _comparison(values)
        design = pd.DataFrame({"group": groups})
        args = {"contrast": "group"} if glm else {}
        result = _ordered(
            comparator.test_cross_diff_freq(
                design if glm else groups,
                statistic="t_cauchy",
                representation="coherency",
                freq_weights=[0.25, 1, 2],
                **args,
            )
        )
        assert result.P_value.iloc[0] == 1
        assert result.Statistic.iloc[0] == 0
        assert pd.isna(result.Best_bin.iloc[0])
        assert (result.P_value.iloc[1:] < 1e-100).all()
        assert (result.Statistic.iloc[1:] > 1e14).all()
        assert result.Best_bin.iloc[1:].tolist() == [0, 0]

    @pytest.mark.parametrize("glm", [False, True])
    def test_skipped_pairs_keep_nan_outputs_and_bh_family(self, glm):
        comparator = _comparison(np.random.default_rng(145).uniform(0.1, 0.8, (8, 6, 4)))
        cross = comparator.cross_spectra_
        cross.pairs = (("g0", "g0"),) + cross.pairs[1:]
        cross.pair_indices[0] = [0, 0]
        cross.coherency_valid[:, 1] = False
        cross.presence[4:, 4:6] = False  # only one group, nonestimable GLM
        cross.presence[1:4, 6:8] = False
        cross.presence[5:, 6:8] = False  # two samples, no GLM residual df
        cross.presence[:, 8:10] = False
        _mask_missing_pairs(cross)
        groups = np.repeat([0, 1], 4)
        args = {"contrast": "group"} if glm else {}
        result = (
            comparator.test_cross_diff_freq(
                pd.DataFrame({"group": groups}) if glm else groups,
                statistic="t_cauchy",
                **args,
            )
            .set_index("Feature_1")
            .loc[[f"g{i}" for i in range(0, 12, 2)]]
        )
        ineligible = "ineligible_design" if glm else "insufficient_samples"
        assert result.Status.tolist() == [
            "self_pair",
            "no_common_bins",
            ineligible,
            ineligible,
            ineligible,
            "ok",
        ]
        for column in ("Estimate_per_bin", "T_per_bin", "P_value_per_bin"):
            assert np.isnan(np.stack(result[column].iloc[:5])).all()
        assert result.Best_bin.iloc[:5].isna().all()
        assert result.P_value.iloc[:5].isna().all()
        assert result.P_adj.iloc[:5].isna().all()
        assert result.P_value.iloc[-1] == result.P_adj.iloc[-1]

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    def test_unmasked_glm_reuses_design_across_bin_support_groups(self, representation):
        comparator = _comparison(np.random.default_rng(146).uniform(0.1, 0.8, (12, 3, 4)))
        comparator.cross_spectra_.coherency_valid[0, 0, 0] = False
        comparator.cross_spectra_.coherency_valid[0, 1, 1] = False
        with patch.object(lm, "_decompose_glm_design", wraps=lm._decompose_glm_design) as decompose:
            comparator.test_cross_diff_freq(
                pd.DataFrame({"x": np.arange(12.0)}),
                contrast="x",
                statistic="t_cauchy",
                representation=representation,
            )
            assert decompose.call_count == 1

    @pytest.mark.parametrize(
        "options",
        [
            {"statistic": "log_t_cauchy"},
            {"statistic": "invalid"},
            {"null": "plugin"},
            {"null": "rotation"},
            {"null": "permutation"},
            {"covariance": "pooled"},
            {"covariance": "per_pair"},
            {"freq_weights": [1, -1, 1]},
            {"freq_weights": [0, 0, 0]},
            {"freq_weights": [1, np.nan, 1]},
            {"freq_weights": [1, 1]},
        ],
    )
    def test_invalid_options_raise(self, options):
        comparator = _comparison(np.full((8, 1, 3), 0.5))
        with pytest.raises(ValueError):
            comparator.test_cross_diff_freq(
                np.repeat([0, 1], 4), **{"statistic": "t_cauchy", **options}
            )

    def test_unmasked_welch_requires_two_samples_per_arm(self):
        comparator = _comparison(np.full((5, 1, 3), 0.5))
        with pytest.raises(ValueError, match="at least two"):
            comparator.test_cross_diff_freq([0, 1, 1, 1, 1], statistic="t_cauchy")

    @pytest.mark.parametrize("representation", ["coherence", "coherency"])
    def test_default_is_l2_rotation(self, representation):
        comparator = _comparison(np.random.default_rng(147).uniform(0.1, 0.8, (12, 3, 4)))
        groups = np.repeat([0, 1], 6)
        default = comparator.test_cross_diff_freq(groups, representation=representation)
        explicit = comparator.test_cross_diff_freq(
            groups, statistic="l2", null="rotation", representation=representation
        )
        pd.testing.assert_frame_equal(default, explicit)
        assert default.attrs == explicit.attrs
        assert default.attrs["covariance"] is None
        assert "Best_bin" not in default

    @pytest.mark.parametrize("glm", [False, True])
    def test_masked_eligibility_thresholds(self, glm):
        comparator = _comparison(np.random.default_rng(148).uniform(0.1, 0.8, (8, 2, 3)))
        comparator.cross_spectra_.presence[:2, :2] = False
        _mask_missing_pairs(comparator.cross_spectra_)
        groups = np.repeat([0, 1], 4)
        options = {"contrast": "group", "min_resid_df": 5} if glm else {"min_samples_per_group": 3}
        result = _ordered(
            comparator.test_cross_diff_freq(
                pd.DataFrame({"group": groups}) if glm else groups,
                statistic="t_cauchy",
                **options,
            )
        )
        assert np.isnan(result.P_value.iloc[0])
        assert pd.isna(result.Best_bin.iloc[0])
        assert np.isfinite(result.P_value.iloc[1])

    def test_no_supported_bins_skips_pair(self):
        comparator = _comparison(np.random.default_rng(149).uniform(0.1, 0.8, (8, 1, 3)))
        comparator.cross_spectra_.coherency_valid[:] = False
        result = comparator.test_cross_diff_freq(
            np.repeat([0, 1], 4),
            statistic="t_cauchy",
            freq_weights=[1, 2, 3],
        ).iloc[0]
        assert result.Status == "no_common_bins"
        assert result.N_bins == 0
        assert np.isnan(result.P_value)
        assert np.isnan(result.P_value_per_bin).all()
        assert pd.isna(result.Best_bin)

    def test_cross_and_df_share_coherence_bin_outputs_with_default_weights(self):
        values = np.random.default_rng(150).uniform(0.1, 0.8, (12, 4, 3))
        comparator = _comparison(values)
        groups = np.repeat([0, 1], 6)
        actual = _ordered(comparator.test_cross_diff_freq(groups, statistic="t_cauchy"))
        expected = (
            ms.compare_two_groups(values**2, groups, statistic="log_t_cauchy")
            .set_index("Feature")
            .sort_index()
        )
        for name in ("Estimate_per_bin", "T_per_bin", "P_value_per_bin"):
            np.testing.assert_allclose(np.stack(actual[name]), np.stack(expected[name]))
        for name in ("Statistic", "P_value", "P_adj"):
            np.testing.assert_allclose(actual[name], expected[name])
        np.testing.assert_array_equal(actual.Best_bin, expected.Best_bin)
