"""Tests for sgi.analysis module."""

import numpy as np
import pandas as pd
import pytest

from sgi.analysis import (
    EffectSizeResult,
    compute_cohens_d,
    compute_correlation_matrix,
    compute_effect_size,
    compute_pairwise_correlations,
    compute_roc_curve,
    compute_topk_overlap,
    compute_topk_overlap_matrix,
    summarize_cross_model_validation,
)


class TestEffectSizeResult:
    """Tests for the EffectSizeResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = EffectSizeResult(
            metric_name="sgi",
            grounded_mean=0.8,
            grounded_std=0.1,
            hallucinated_mean=1.2,
            hallucinated_std=0.15,
            cohens_d=2.5,
            p_value=0.001,
            auroc=0.75,
            significant=True,
        )
        d = result.to_dict()
        assert d["metric"] == "sgi"
        assert d["grounded_mean"] == 0.8
        assert d["cohens_d"] == 2.5
        assert d["significant"] is True


class TestComputeCohensD:
    """Tests for the compute_cohens_d function."""

    def test_identical_groups_zero(self):
        """Identical groups should have zero effect size."""
        group = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = compute_cohens_d(group, group.copy())
        assert np.isclose(d, 0.0, atol=1e-10)

    def test_positive_effect(self):
        """Higher second group should give positive d."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        d = compute_cohens_d(group1, group2)
        assert d > 0

    def test_negative_effect(self):
        """Lower second group should give negative d."""
        group1 = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        group2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = compute_cohens_d(group1, group2)
        assert d < 0

    def test_large_effect(self):
        """Well-separated groups should have large effect."""
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 100)
        group2 = np.random.normal(2, 1, 100)
        d = compute_cohens_d(group1, group2)
        # d should be around 2 (2 std dev difference)
        assert 1.5 < d < 2.5

    def test_no_variance_returns_zero(self):
        """Constant groups should return 0."""
        group1 = np.array([5.0, 5.0, 5.0])
        group2 = np.array([5.0, 5.0, 5.0])
        d = compute_cohens_d(group1, group2)
        assert np.isclose(d, 0.0)


class TestComputeEffectSize:
    """Tests for the compute_effect_size function."""

    def test_returns_effect_size_result(self):
        """Should return EffectSizeResult."""
        np.random.seed(42)
        values = np.concatenate([np.random.normal(0.8, 0.1, 50), np.random.normal(1.2, 0.15, 50)])
        labels = np.array([True] * 50 + [False] * 50)
        result = compute_effect_size(values, labels, "test_metric")
        assert isinstance(result, EffectSizeResult)

    def test_grounded_vs_hallucinated_means(self):
        """Should correctly compute group means."""
        np.random.seed(42)
        grounded_vals = np.random.normal(0.8, 0.1, 50)
        halluc_vals = np.random.normal(1.2, 0.1, 50)
        values = np.concatenate([grounded_vals, halluc_vals])
        labels = np.array([True] * 50 + [False] * 50)

        result = compute_effect_size(values, labels)
        assert np.isclose(result.grounded_mean, grounded_vals.mean(), rtol=0.01)
        assert np.isclose(result.hallucinated_mean, halluc_vals.mean(), rtol=0.01)

    def test_significant_difference(self):
        """Well-separated groups should be significant."""
        np.random.seed(42)
        values = np.concatenate([np.random.normal(0.5, 0.1, 100), np.random.normal(1.5, 0.1, 100)])
        labels = np.array([True] * 100 + [False] * 100)
        result = compute_effect_size(values, labels)
        assert result.significant
        assert result.p_value < 0.05

    def test_auroc_range(self):
        """AUROC should be in [0, 1]."""
        np.random.seed(42)
        values = np.random.rand(100)
        labels = np.random.choice([True, False], 100)
        result = compute_effect_size(values, labels)
        assert 0 <= result.auroc <= 1


class TestComputeCorrelationMatrix:
    """Tests for the compute_correlation_matrix function."""

    def test_diagonal_ones(self, sample_sgi_dataframe):
        """Diagonal should be 1.0."""
        columns = ["sgi_model_a", "sgi_model_b", "sgi_model_c"]
        corr = compute_correlation_matrix(sample_sgi_dataframe, columns)
        for col in columns:
            assert np.isclose(corr.loc[col, col], 1.0)

    def test_symmetry(self, sample_sgi_dataframe):
        """Matrix should be symmetric."""
        columns = ["sgi_model_a", "sgi_model_b", "sgi_model_c"]
        corr = compute_correlation_matrix(sample_sgi_dataframe, columns)
        np.testing.assert_allclose(corr.values, corr.values.T)

    def test_spearman_method(self, sample_sgi_dataframe):
        """Should support Spearman correlation."""
        columns = ["sgi_model_a", "sgi_model_b"]
        corr = compute_correlation_matrix(sample_sgi_dataframe, columns, method="spearman")
        assert corr.shape == (2, 2)

    def test_values_in_range(self, sample_sgi_dataframe):
        """All correlations should be in [-1, 1]."""
        columns = ["sgi_model_a", "sgi_model_b", "sgi_model_c"]
        corr = compute_correlation_matrix(sample_sgi_dataframe, columns)
        assert np.all(corr.values >= -1)
        assert np.all(corr.values <= 1)


class TestComputePairwiseCorrelations:
    """Tests for the compute_pairwise_correlations function."""

    def test_summary_stats(self, sample_sgi_dataframe):
        """Should compute summary statistics."""
        columns = ["sgi_model_a", "sgi_model_b", "sgi_model_c"]
        corr = compute_correlation_matrix(sample_sgi_dataframe, columns)
        stats = compute_pairwise_correlations(corr)

        assert "mean" in stats
        assert "min" in stats
        assert "max" in stats
        assert "std" in stats
        assert "n_pairs" in stats
        assert stats["n_pairs"] == 3  # 3 unique pairs for 3 models

    def test_min_max_range(self, sample_sgi_dataframe):
        """Min should be <= mean <= max."""
        columns = ["sgi_model_a", "sgi_model_b", "sgi_model_c"]
        corr = compute_correlation_matrix(sample_sgi_dataframe, columns)
        stats = compute_pairwise_correlations(corr)
        assert stats["min"] <= stats["mean"] <= stats["max"]


class TestComputeTopkOverlap:
    """Tests for the compute_topk_overlap function."""

    def test_identical_columns(self, sample_sgi_dataframe):
        """Identical columns should have perfect overlap."""
        sample_sgi_dataframe["sgi_model_copy"] = sample_sgi_dataframe["sgi_model_a"]
        jaccard, overlap_pct = compute_topk_overlap(
            sample_sgi_dataframe, "sgi_model_a", "sgi_model_copy", k_percent=10
        )
        assert np.isclose(jaccard, 1.0)
        assert np.isclose(overlap_pct, 100.0)

    def test_jaccard_range(self, sample_sgi_dataframe):
        """Jaccard should be in [0, 1]."""
        jaccard, _ = compute_topk_overlap(
            sample_sgi_dataframe, "sgi_model_a", "sgi_model_b", k_percent=10
        )
        assert 0 <= jaccard <= 1

    def test_overlap_pct_range(self, sample_sgi_dataframe):
        """Overlap percentage should be in [0, 100]."""
        _, overlap_pct = compute_topk_overlap(
            sample_sgi_dataframe, "sgi_model_a", "sgi_model_b", k_percent=10
        )
        assert 0 <= overlap_pct <= 100


class TestComputeTopkOverlapMatrix:
    """Tests for the compute_topk_overlap_matrix function."""

    def test_output_shape(self, sample_sgi_dataframe):
        """Output should be square matrix."""
        columns = ["sgi_model_a", "sgi_model_b", "sgi_model_c"]
        matrix = compute_topk_overlap_matrix(sample_sgi_dataframe, columns, k_percent=10)
        assert matrix.shape == (3, 3)

    def test_diagonal_ones(self, sample_sgi_dataframe):
        """Diagonal should be 1.0 (perfect self-overlap)."""
        columns = ["sgi_model_a", "sgi_model_b", "sgi_model_c"]
        matrix = compute_topk_overlap_matrix(sample_sgi_dataframe, columns, k_percent=10)
        for col in columns:
            assert np.isclose(matrix.loc[col, col], 1.0)

    def test_symmetry(self, sample_sgi_dataframe):
        """Matrix should be symmetric."""
        columns = ["sgi_model_a", "sgi_model_b", "sgi_model_c"]
        matrix = compute_topk_overlap_matrix(sample_sgi_dataframe, columns, k_percent=10)
        np.testing.assert_allclose(matrix.values, matrix.values.T)


class TestComputeROCCurve:
    """Tests for the compute_roc_curve function."""

    def test_output_types(self):
        """Should return correct tuple elements."""
        np.random.seed(42)
        values = np.concatenate([np.random.normal(0.8, 0.1, 50), np.random.normal(1.2, 0.1, 50)])
        labels = np.array([True] * 50 + [False] * 50)
        fpr, tpr, thresholds, auroc = compute_roc_curve(values, labels)

        assert isinstance(fpr, np.ndarray)
        assert isinstance(tpr, np.ndarray)
        assert isinstance(thresholds, np.ndarray)
        assert isinstance(auroc, float)

    def test_fpr_tpr_range(self):
        """FPR and TPR should be in [0, 1]."""
        np.random.seed(42)
        values = np.random.rand(100)
        labels = np.random.choice([True, False], 100)
        fpr, tpr, _, _ = compute_roc_curve(values, labels)

        assert np.all(fpr >= 0) and np.all(fpr <= 1)
        assert np.all(tpr >= 0) and np.all(tpr <= 1)

    def test_auroc_range(self):
        """AUROC should be in [0, 1]."""
        np.random.seed(42)
        values = np.random.rand(100)
        labels = np.random.choice([True, False], 100)
        _, _, _, auroc = compute_roc_curve(values, labels)
        assert 0 <= auroc <= 1

    def test_perfect_classifier(self):
        """Perfect separation should give AUROC = 1."""
        values = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 1.0])
        labels = np.array([True, True, True, False, False, False])
        _, _, _, auroc = compute_roc_curve(values, labels)
        assert np.isclose(auroc, 1.0, atol=0.01)


class TestSummarizeCrossModelValidation:
    """Tests for the summarize_cross_model_validation function."""

    def test_returns_string(self, sample_effect_sizes):
        """Should return a string."""
        effect_sizes_result = {
            name: EffectSizeResult(
                metric_name="sgi",
                grounded_mean=0.8,
                grounded_std=0.1,
                hallucinated_mean=1.2,
                hallucinated_std=0.1,
                cohens_d=data["cohens_d"],
                p_value=data["p_value"],
                auroc=data["auroc"],
                significant=data["significant"],
            )
            for name, data in sample_effect_sizes.items()
        }
        pearson = {"mean": 0.85, "min": 0.80, "max": 0.90, "std": 0.05}
        spearman = {"mean": 0.82, "min": 0.78, "max": 0.88, "std": 0.04}

        result = summarize_cross_model_validation(effect_sizes_result, pearson, spearman)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_key_sections(self, sample_effect_sizes):
        """Summary should contain key sections."""
        effect_sizes_result = {
            name: EffectSizeResult(
                metric_name="sgi",
                grounded_mean=0.8,
                grounded_std=0.1,
                hallucinated_mean=1.2,
                hallucinated_std=0.1,
                cohens_d=data["cohens_d"],
                p_value=data["p_value"],
                auroc=data["auroc"],
                significant=data["significant"],
            )
            for name, data in sample_effect_sizes.items()
        }
        pearson = {"mean": 0.85, "min": 0.80, "max": 0.90, "std": 0.05}
        spearman = {"mean": 0.82, "min": 0.78, "max": 0.88, "std": 0.04}

        result = summarize_cross_model_validation(effect_sizes_result, pearson, spearman)
        assert "CROSS-MODEL VALIDATION SUMMARY" in result
        assert "Models tested" in result
        assert "Effect Size" in result
        assert "INTERPRETATION" in result
