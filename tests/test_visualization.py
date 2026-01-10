"""Tests for sgi.visualization module."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from sgi.visualization import (
    create_summary_figure,
    plot_correlation_heatmap,
    plot_effect_sizes,
    plot_roc_curves,
    plot_scatter_comparison,
    plot_sgi_distributions,
    set_publication_style,
)


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


class TestSetPublicationStyle:
    """Tests for set_publication_style function."""

    def test_sets_rcparams(self):
        """Should modify matplotlib rcParams."""
        original_fontsize = plt.rcParams["font.size"]
        set_publication_style()
        # Check some expected values
        assert plt.rcParams["figure.dpi"] == 150
        assert plt.rcParams["savefig.dpi"] == 300
        assert plt.rcParams["axes.spines.top"] is False


class TestPlotEffectSizes:
    """Tests for plot_effect_sizes function."""

    def test_returns_axes(self, sample_effect_sizes):
        """Should return matplotlib Axes."""
        ax = plot_effect_sizes(sample_effect_sizes)
        assert isinstance(ax, plt.Axes)

    def test_uses_provided_axes(self, sample_effect_sizes):
        """Should use provided axes."""
        fig, ax = plt.subplots()
        result = plot_effect_sizes(sample_effect_sizes, ax=ax)
        assert result is ax

    def test_cohens_d_metric(self, sample_effect_sizes):
        """Should plot Cohen's d metric."""
        ax = plot_effect_sizes(sample_effect_sizes, metric="cohens_d")
        assert "Cohen's d" in ax.get_ylabel()

    def test_auroc_metric(self, sample_effect_sizes):
        """Should plot AUROC metric."""
        ax = plot_effect_sizes(sample_effect_sizes, metric="auroc")
        assert "AUROC" in ax.get_ylabel()

    def test_custom_title(self, sample_effect_sizes):
        """Should use custom title."""
        title = "Custom Effect Size Title"
        ax = plot_effect_sizes(sample_effect_sizes, title=title)
        assert ax.get_title() == title


class TestPlotCorrelationHeatmap:
    """Tests for plot_correlation_heatmap function."""

    @pytest.fixture
    def sample_corr_matrix(self):
        """Create sample correlation matrix."""
        data = {
            "model_a": [1.0, 0.85, 0.82],
            "model_b": [0.85, 1.0, 0.88],
            "model_c": [0.82, 0.88, 1.0],
        }
        return pd.DataFrame(data, index=["model_a", "model_b", "model_c"])

    def test_returns_axes(self, sample_corr_matrix):
        """Should return matplotlib Axes."""
        ax = plot_correlation_heatmap(sample_corr_matrix)
        assert isinstance(ax, plt.Axes)

    def test_uses_provided_axes(self, sample_corr_matrix):
        """Should use provided axes."""
        fig, ax = plt.subplots()
        result = plot_correlation_heatmap(sample_corr_matrix, ax=ax)
        assert result is ax

    def test_custom_title(self, sample_corr_matrix):
        """Should use custom title."""
        title = "Custom Correlation Title"
        ax = plot_correlation_heatmap(sample_corr_matrix, title=title)
        assert ax.get_title() == title


class TestPlotSGIDistributions:
    """Tests for plot_sgi_distributions function."""

    @pytest.fixture
    def distribution_df(self):
        """Create sample DataFrame for distribution plots."""
        np.random.seed(42)
        n = 50
        return pd.DataFrame(
            {
                "is_grounded": [True] * n + [False] * n,
                "sgi_model_a": np.concatenate(
                    [np.random.normal(0.8, 0.2, n), np.random.normal(1.2, 0.2, n)]
                ),
                "sgi_model_b": np.concatenate(
                    [np.random.normal(0.75, 0.15, n), np.random.normal(1.15, 0.18, n)]
                ),
            }
        )

    def test_returns_axes(self, distribution_df):
        """Should return matplotlib Axes."""
        ax = plot_sgi_distributions(distribution_df, ["model_a", "model_b"])
        assert isinstance(ax, plt.Axes)

    def test_uses_provided_axes(self, distribution_df):
        """Should use provided axes."""
        fig, ax = plt.subplots()
        result = plot_sgi_distributions(distribution_df, ["model_a"], ax=ax)
        assert result is ax

    def test_sets_labels(self, distribution_df):
        """Should set axis labels."""
        ax = plot_sgi_distributions(distribution_df, ["model_a"])
        assert ax.get_xlabel() == "SGI"
        assert ax.get_ylabel() == "Density"


class TestPlotScatterComparison:
    """Tests for plot_scatter_comparison function."""

    @pytest.fixture
    def scatter_df(self):
        """Create sample DataFrame for scatter plots."""
        np.random.seed(42)
        n = 50
        return pd.DataFrame(
            {
                "is_grounded": [True] * n + [False] * n,
                "sgi_model_a": np.random.uniform(0.5, 1.5, n * 2),
                "sgi_model_b": np.random.uniform(0.5, 1.5, n * 2),
            }
        )

    def test_returns_axes(self, scatter_df):
        """Should return matplotlib Axes."""
        ax = plot_scatter_comparison(scatter_df, "model_a", "model_b")
        assert isinstance(ax, plt.Axes)

    def test_uses_provided_axes(self, scatter_df):
        """Should use provided axes."""
        fig, ax = plt.subplots()
        result = plot_scatter_comparison(scatter_df, "model_a", "model_b", ax=ax)
        assert result is ax

    def test_custom_title(self, scatter_df):
        """Should use custom title."""
        title = "Custom Scatter Title"
        ax = plot_scatter_comparison(scatter_df, "model_a", "model_b", title=title)
        assert ax.get_title() == title

    def test_auto_title_with_correlation(self, scatter_df):
        """Auto title should include correlation."""
        ax = plot_scatter_comparison(scatter_df, "model_a", "model_b")
        assert "r=" in ax.get_title()


class TestPlotROCCurves:
    """Tests for plot_roc_curves function."""

    @pytest.fixture
    def roc_data(self):
        """Create sample ROC data."""
        return {
            "model_a": (
                np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                np.array([0.0, 0.5, 0.7, 0.85, 0.95, 1.0]),
                0.78,
            ),
            "model_b": (
                np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                np.array([0.0, 0.45, 0.65, 0.8, 0.9, 1.0]),
                0.72,
            ),
        }

    def test_returns_axes(self, roc_data):
        """Should return matplotlib Axes."""
        ax = plot_roc_curves(roc_data)
        assert isinstance(ax, plt.Axes)

    def test_uses_provided_axes(self, roc_data):
        """Should use provided axes."""
        fig, ax = plt.subplots()
        result = plot_roc_curves(roc_data, ax=ax)
        assert result is ax

    def test_custom_title(self, roc_data):
        """Should use custom title."""
        title = "Custom ROC Title"
        ax = plot_roc_curves(roc_data, title=title)
        assert ax.get_title() == title

    def test_sets_labels(self, roc_data):
        """Should set axis labels."""
        ax = plot_roc_curves(roc_data)
        assert ax.get_xlabel() == "False Positive Rate"
        assert ax.get_ylabel() == "True Positive Rate"


class TestCreateSummaryFigure:
    """Tests for create_summary_figure function."""

    @pytest.fixture
    def summary_data(self, sample_sgi_dataframe, sample_effect_sizes):
        """Create data for summary figure."""
        columns = ["sgi_model_a", "sgi_model_b", "sgi_model_c"]
        pearson = sample_sgi_dataframe[columns].corr(method="pearson")
        spearman = sample_sgi_dataframe[columns].corr(method="spearman")
        # Rename to match model names without prefix
        pearson.index = ["model_a", "model_b", "model_c"]
        pearson.columns = ["model_a", "model_b", "model_c"]
        spearman.index = ["model_a", "model_b", "model_c"]
        spearman.columns = ["model_a", "model_b", "model_c"]

        return {
            "df": sample_sgi_dataframe,
            "effect_sizes": sample_effect_sizes,
            "pearson": pearson,
            "spearman": spearman,
            "model_names": ["model_a", "model_b", "model_c"],
        }

    def test_returns_figure(self, summary_data):
        """Should return matplotlib Figure."""
        fig = create_summary_figure(
            df=summary_data["df"],
            effect_sizes=summary_data["effect_sizes"],
            pearson_matrix=summary_data["pearson"],
            spearman_matrix=summary_data["spearman"],
            model_names=summary_data["model_names"],
        )
        assert isinstance(fig, plt.Figure)

    def test_has_four_subplots(self, summary_data):
        """Should create 2x2 subplot grid."""
        fig = create_summary_figure(
            df=summary_data["df"],
            effect_sizes=summary_data["effect_sizes"],
            pearson_matrix=summary_data["pearson"],
            spearman_matrix=summary_data["spearman"],
            model_names=summary_data["model_names"],
        )
        # Figure should have 4 axes (2x2 grid) plus potential colorbar axes
        assert len(fig.axes) >= 4
