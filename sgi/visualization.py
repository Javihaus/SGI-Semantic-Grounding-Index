"""
Visualization Utilities

Publication-quality figures for cross-model validation experiments.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def plot_effect_sizes(
    effect_sizes: Dict[str, dict],
    ax: Optional[plt.Axes] = None,
    metric: str = "cohens_d",
    title: str = "Effect Size Across Embedding Models",
) -> plt.Axes:
    """
    Bar plot of effect sizes across models.

    Args:
        effect_sizes: Dict mapping model names to effect size results
        ax: Matplotlib axes (created if None)
        metric: 'cohens_d' or 'auroc'
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    model_names = list(effect_sizes.keys())
    values = [effect_sizes[m][metric] for m in model_names]

    # Color by threshold
    if metric == "cohens_d":
        colors = ["#2ecc71" if v > 0.5 else "#f39c12" if v > 0.2 else "#e74c3c" for v in values]
        thresholds = [(0.8, "Large (0.8)"), (0.5, "Medium (0.5)"), (0.2, "Small (0.2)")]
    else:  # auroc
        colors = ["#2ecc71" if v > 0.7 else "#f39c12" if v > 0.6 else "#e74c3c" for v in values]
        thresholds = [(0.7, "Good (0.7)"), (0.5, "Random (0.5)")]

    bars = ax.bar(model_names, values, color=colors, edgecolor="black", linewidth=0.5)

    # Reference lines
    for thresh, label in thresholds:
        ax.axhline(thresh, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Cohen's d" if metric == "cohens_d" else "AUROC")
    ax.set_title(title)
    ax.set_xticklabels(model_names, rotation=45, ha="right")

    return ax


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "Cross-Model SGI Correlation",
    cmap: str = "RdYlGn",
) -> plt.Axes:
    """
    Heatmap of correlation matrix.

    Args:
        corr_matrix: Correlation matrix as DataFrame
        ax: Matplotlib axes (created if None)
        title: Plot title
        cmap: Colormap name

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap=cmap,
        center=0.5,
        vmin=0,
        vmax=1,
        square=True,
        fmt=".2f",
        ax=ax,
        cbar_kws={"label": "Correlation"},
        linewidths=0.5,
    )
    ax.set_title(title)

    return ax


def plot_sgi_distributions(
    df: pd.DataFrame,
    model_names: List[str],
    ax: Optional[plt.Axes] = None,
    title: str = "SGI Distributions by Model",
) -> plt.Axes:
    """
    Overlaid KDE plots of SGI distributions.

    Args:
        df: DataFrame with SGI columns and is_grounded column
        model_names: List of model names
        ax: Matplotlib axes (created if None)
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    for i, model_name in enumerate(model_names):
        sgi_col = f"sgi_{model_name}"
        grounded = df[df["is_grounded"]][sgi_col]
        hallucinated = df[~df["is_grounded"]][sgi_col]

        sns.kdeplot(
            grounded,
            ax=ax,
            color=colors[i],
            linestyle="-",
            alpha=0.7,
            label=f"{model_name} (grounded)",
        )
        sns.kdeplot(hallucinated, ax=ax, color=colors[i], linestyle="--", alpha=0.7)

    ax.axvline(1.0, color="black", linestyle=":", alpha=0.5, label="SGI=1")
    ax.set_xlabel("SGI")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 3)

    return ax


def plot_scatter_comparison(
    df: pd.DataFrame,
    model1: str,
    model2: str,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Scatter plot comparing SGI scores from two models.

    Args:
        df: DataFrame with SGI columns and is_grounded column
        model1: First model name
        model2: Second model name
        ax: Matplotlib axes (created if None)
        title: Plot title (auto-generated if None)

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    col1, col2 = f"sgi_{model1}", f"sgi_{model2}"
    grounded_mask = df["is_grounded"]

    ax.scatter(
        df[~grounded_mask][col1],
        df[~grounded_mask][col2],
        alpha=0.5,
        c="#e74c3c",
        label="Hallucinated",
        s=30,
    )
    ax.scatter(
        df[grounded_mask][col1],
        df[grounded_mask][col2],
        alpha=0.5,
        c="#2ecc71",
        label="Grounded",
        s=30,
    )

    # Identity line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3)

    ax.set_xlabel(f"SGI ({model1})")
    ax.set_ylabel(f"SGI ({model2})")

    if title is None:
        corr = df[col1].corr(df[col2])
        title = f"{model1} vs {model2} (r={corr:.3f})"
    ax.set_title(title)
    ax.legend()

    return ax


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    ax: Optional[plt.Axes] = None,
    title: str = "ROC Curves by Embedding Model",
) -> plt.Axes:
    """
    Plot ROC curves for multiple models.

    Args:
        roc_data: Dict mapping model names to (fpr, tpr, auroc) tuples
        ax: Matplotlib axes (created if None)
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))

    for (model_name, (fpr, tpr, auroc)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{model_name} (AUC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    return ax


def create_summary_figure(
    df: pd.DataFrame,
    effect_sizes: Dict[str, dict],
    pearson_matrix: pd.DataFrame,
    spearman_matrix: pd.DataFrame,
    model_names: List[str],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create comprehensive 2x2 summary figure.

    Args:
        df: DataFrame with all results
        effect_sizes: Effect size results per model
        pearson_matrix: Pearson correlation matrix
        spearman_matrix: Spearman correlation matrix
        model_names: List of model names
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Effect sizes
    plot_effect_sizes(
        effect_sizes, axes[0, 0], metric="cohens_d", title="Effect Size (Cohen's d) Across Models"
    )

    # 2. AUROC
    plot_effect_sizes(
        effect_sizes, axes[0, 1], metric="auroc", title="Classification Performance (AUROC)"
    )

    # 3. SGI distributions
    plot_sgi_distributions(
        df,
        model_names,
        axes[1, 0],
        title="SGI Distributions: Grounded (solid) vs Hallucinated (dashed)",
    )

    # 4. Spearman correlation
    plot_correlation_heatmap(spearman_matrix, axes[1, 1], title="Ranking Agreement (Spearman ρ)")

    fig.suptitle(
        "Cross-Model Validation: Is SGI Fundamental?", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    return fig
