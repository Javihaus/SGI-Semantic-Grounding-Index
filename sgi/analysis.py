"""
Statistical Analysis Utilities

Effect size computation, cross-model validation, and statistical tests
for hallucination detection experiments.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


@dataclass
class EffectSizeResult:
    """Results from effect size analysis."""

    metric_name: str
    grounded_mean: float
    grounded_std: float
    hallucinated_mean: float
    hallucinated_std: float
    cohens_d: float
    p_value: float
    auroc: float
    significant: bool

    def to_dict(self) -> Dict:
        return {
            "metric": self.metric_name,
            "grounded_mean": self.grounded_mean,
            "grounded_std": self.grounded_std,
            "hallucinated_mean": self.hallucinated_mean,
            "hallucinated_std": self.hallucinated_std,
            "cohens_d": self.cohens_d,
            "p_value": self.p_value,
            "auroc": self.auroc,
            "significant": self.significant,
        }


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.

    Args:
        group1: First group values
        group2: Second group values

    Returns:
        Cohen's d (positive = group2 > group1)
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return float((group2.mean() - group1.mean()) / pooled_std)


def compute_effect_size(
    values: np.ndarray,
    labels: np.ndarray,
    metric_name: str = "metric",
    positive_class_is_hallucinated: bool = True,
) -> EffectSizeResult:
    """
    Compute effect size and classification metrics.

    Args:
        values: Metric values (e.g., SGI scores)
        labels: Boolean labels (True = grounded, False = hallucinated)
        metric_name: Name for reporting
        positive_class_is_hallucinated: If True, higher values indicate hallucination

    Returns:
        EffectSizeResult with all computed statistics
    """
    grounded = values[labels]
    hallucinated = values[~labels]

    # Effect size
    cohens_d = compute_cohens_d(grounded, hallucinated)

    # Statistical test
    _, p_value = stats.ttest_ind(hallucinated, grounded, equal_var=False)

    # AUROC (for detecting hallucinations)
    y_true: np.ndarray = (~labels).astype(int)  # 1 = hallucinated
    if positive_class_is_hallucinated:
        auroc = roc_auc_score(y_true, values)
    else:
        auroc = roc_auc_score(y_true, -values)

    return EffectSizeResult(
        metric_name=metric_name,
        grounded_mean=float(grounded.mean()),
        grounded_std=float(grounded.std()),
        hallucinated_mean=float(hallucinated.mean()),
        hallucinated_std=float(hallucinated.std()),
        cohens_d=float(cohens_d),
        p_value=float(p_value),
        auroc=float(auroc),
        significant=p_value < 0.05,
    )


def compute_correlation_matrix(
    df: pd.DataFrame, columns: List[str], method: str = "pearson"
) -> pd.DataFrame:
    """
    Compute correlation matrix for specified columns.

    Args:
        df: DataFrame with data
        columns: Column names to correlate
        method: 'pearson' or 'spearman'

    Returns:
        Correlation matrix as DataFrame
    """
    return df[columns].corr(method=method)


def compute_pairwise_correlations(corr_matrix: pd.DataFrame) -> Dict[str, float]:
    """
    Compute summary statistics for off-diagonal correlations.

    Args:
        corr_matrix: Correlation matrix

    Returns:
        Dictionary with mean, min, max, std of off-diagonal elements
    """
    n = len(corr_matrix)
    off_diag = []

    for i in range(n):
        for j in range(i + 1, n):
            off_diag.append(corr_matrix.iloc[i, j])

    off_diag = np.array(off_diag)

    return {
        "mean": float(off_diag.mean()),
        "min": float(off_diag.min()),
        "max": float(off_diag.max()),
        "std": float(off_diag.std()),
        "n_pairs": len(off_diag),
    }


def compute_topk_overlap(
    df: pd.DataFrame, col1: str, col2: str, k_percent: float = 10.0
) -> Tuple[float, float]:
    """
    Compute overlap between top-k% flagged by two metrics.

    Args:
        df: DataFrame with metric columns
        col1: First metric column
        col2: Second metric column
        k_percent: Percentage of top samples to compare

    Returns:
        Tuple of (jaccard_similarity, overlap_percentage)
    """
    n = len(df)
    k = int(n * k_percent / 100)

    top_k_1 = set(df.nlargest(k, col1).index)
    top_k_2 = set(df.nlargest(k, col2).index)

    intersection = len(top_k_1 & top_k_2)
    union = len(top_k_1 | top_k_2)

    jaccard = intersection / union if union > 0 else 0
    overlap_pct = intersection / k * 100 if k > 0 else 0

    return jaccard, overlap_pct


def compute_topk_overlap_matrix(
    df: pd.DataFrame, columns: List[str], k_percent: float = 10.0
) -> pd.DataFrame:
    """
    Compute pairwise top-k overlap matrix.

    Args:
        df: DataFrame with metric columns
        columns: Column names to compare
        k_percent: Percentage of top samples

    Returns:
        Overlap matrix as DataFrame
    """
    n = len(columns)
    matrix = np.zeros((n, n))

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i <= j:
                jaccard, _ = compute_topk_overlap(df, col1, col2, k_percent)
                matrix[i, j] = jaccard
                matrix[j, i] = jaccard

    return pd.DataFrame(matrix, index=columns, columns=columns)


def compute_roc_curve(
    values: np.ndarray, labels: np.ndarray, positive_class_is_hallucinated: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve.

    Args:
        values: Metric values
        labels: Boolean labels (True = grounded)
        positive_class_is_hallucinated: If True, higher values indicate hallucination

    Returns:
        Tuple of (fpr, tpr, thresholds, auroc)
    """
    y_true: np.ndarray = (~labels).astype(int)  # 1 = hallucinated
    y_scores = values if positive_class_is_hallucinated else -values

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auroc = auc(fpr, tpr)

    return fpr, tpr, thresholds, auroc


def compute_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        scores: Prediction scores (e.g., SGI values)
        labels: Binary labels (True = grounded, False = hallucinated)
        n_bins: Number of calibration bins

    Returns:
        Tuple of (ece, bin_accuracies, bin_confidences, bin_counts)
    """
    # Normalize scores to [0, 1] - higher score = more likely hallucinated
    scores_min, scores_max = scores.min(), scores.max()
    scores_norm = (scores - scores_min) / (scores_max - scores_min + 1e-8)

    # Convert labels: 1 = hallucinated (positive class for calibration)
    y_true = (~labels).astype(int)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        # Handle last bin edge case
        if i == n_bins - 1:
            mask = (scores_norm >= bin_boundaries[i]) & (scores_norm <= bin_boundaries[i + 1])
        else:
            mask = (scores_norm >= bin_boundaries[i]) & (scores_norm < bin_boundaries[i + 1])

        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            bin_accuracies[i] = y_true[mask].mean()
            bin_confidences[i] = scores_norm[mask].mean()
        else:
            bin_confidences[i] = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2

    # Compute ECE (weighted average of |accuracy - confidence|)
    total = bin_counts.sum()
    if total > 0:
        ece = sum(
            (count / total) * abs(acc - conf)
            for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
            if count > 0
        )
    else:
        ece = 0.0

    return float(ece), bin_accuracies, bin_confidences, bin_counts


def compute_stratified_analysis(
    df: pd.DataFrame,
    stratify_col: str,
    sgi_col: str = "sgi",
    label_col: str = "is_grounded",
    n_bins: int = 3,
) -> pd.DataFrame:
    """
    Compute effect sizes stratified by a continuous variable.

    Args:
        df: DataFrame with SGI scores and labels
        stratify_col: Column to stratify by (e.g., 'theta_qc')
        sgi_col: Column name for SGI scores
        label_col: Column name for grounding labels
        n_bins: Number of strata (default: 3 for terciles)

    Returns:
        DataFrame with effect sizes per stratum
    """
    # Create strata labels
    stratum_labels = ["Low", "Medium", "High"] if n_bins == 3 else [f"Q{i+1}" for i in range(n_bins)]
    df = df.copy()
    df["stratum"] = pd.qcut(df[stratify_col], n_bins, labels=stratum_labels, duplicates="drop")

    results = []
    for stratum in df["stratum"].unique():
        subset = df[df["stratum"] == stratum]

        grounded = subset[subset[label_col]][sgi_col].values
        hallucinated = subset[~subset[label_col]][sgi_col].values

        if len(grounded) < 2 or len(hallucinated) < 2:
            continue

        d = compute_cohens_d(hallucinated, grounded)

        # AUROC for detecting hallucinations (higher SGI = more likely hallucinated)
        y_true = (~subset[label_col]).astype(int).values
        try:
            auroc = roc_auc_score(y_true, subset[sgi_col].values)
        except ValueError:
            auroc = 0.5

        results.append(
            {
                "stratum": stratum,
                "n": len(subset),
                "range_min": float(subset[stratify_col].min()),
                "range_max": float(subset[stratify_col].max()),
                "sgi_grounded": float(grounded.mean()),
                "sgi_hallucinated": float(hallucinated.mean()),
                "cohens_d": float(d),
                "auroc": float(auroc),
            }
        )

    return pd.DataFrame(results)


def compute_subgroup_analysis(
    df: pd.DataFrame,
    feature_col: str,
    sgi_col: str = "sgi",
    label_col: str = "is_grounded",
    n_bins: int = 3,
) -> pd.DataFrame:
    """
    Compute effect sizes for subgroups defined by a feature.

    Args:
        df: DataFrame with SGI scores, labels, and feature column
        feature_col: Column to create subgroups from
        sgi_col: Column name for SGI scores
        label_col: Column name for grounding labels
        n_bins: Number of subgroups

    Returns:
        DataFrame with effect sizes per subgroup
    """
    subgroup_labels = ["Short", "Medium", "Long"] if n_bins == 3 else [f"G{i+1}" for i in range(n_bins)]
    df = df.copy()
    df["subgroup"] = pd.qcut(df[feature_col], n_bins, labels=subgroup_labels, duplicates="drop")

    results = []
    for subgroup in df["subgroup"].unique():
        subset = df[df["subgroup"] == subgroup]

        grounded = subset[subset[label_col]][sgi_col].values
        hallucinated = subset[~subset[label_col]][sgi_col].values

        if len(grounded) < 2 or len(hallucinated) < 2:
            continue

        d = compute_cohens_d(hallucinated, grounded)

        y_true = (~subset[label_col]).astype(int).values
        try:
            auroc = roc_auc_score(y_true, subset[sgi_col].values)
        except ValueError:
            auroc = 0.5

        results.append(
            {
                "subgroup": subgroup,
                "feature": feature_col,
                "n": len(subset),
                "feature_mean": float(subset[feature_col].mean()),
                "sgi_grounded": float(grounded.mean()),
                "sgi_hallucinated": float(hallucinated.mean()),
                "cohens_d": float(d),
                "auroc": float(auroc),
            }
        )

    return pd.DataFrame(results)


def summarize_cross_model_validation(
    effect_sizes: Dict[str, EffectSizeResult],
    pearson_corr: Dict[str, float],
    spearman_corr: Dict[str, float],
) -> str:
    """
    Generate summary interpretation of cross-model validation.

    Args:
        effect_sizes: Effect size results per model
        pearson_corr: Pearson correlation summary
        spearman_corr: Spearman correlation summary

    Returns:
        Interpretation string
    """
    n_models = len(effect_sizes)
    n_significant = sum(1 for r in effect_sizes.values() if r.significant)
    mean_spearman = spearman_corr["mean"]

    d_values = [r.cohens_d for r in effect_sizes.values()]
    auc_values = [r.auroc for r in effect_sizes.values()]

    lines = [
        "=" * 70,
        "CROSS-MODEL VALIDATION SUMMARY",
        "=" * 70,
        f"",
        f"Models tested: {n_models}",
        f"Models with significant effect (p<0.05): {n_significant}/{n_models}",
        f"",
        f"Effect Size (Cohen's d):",
        f"  Mean: {np.mean(d_values):.3f}",
        f"  Range: [{min(d_values):.3f}, {max(d_values):.3f}]",
        f"",
        f"Classification (AUROC):",
        f"  Mean: {np.mean(auc_values):.3f}",
        f"  Range: [{min(auc_values):.3f}, {max(auc_values):.3f}]",
        f"",
        f"Cross-Model Agreement:",
        f"  Pearson r (mean): {pearson_corr['mean']:.3f}",
        f"  Spearman ρ (mean): {spearman_corr['mean']:.3f}",
        f"",
        "-" * 70,
        "INTERPRETATION:",
        "-" * 70,
    ]

    if n_significant == n_models and mean_spearman > 0.7:
        lines.append(
            """
✓ STRONG EVIDENCE: SGI captures something fundamental.
  - Effect is significant across ALL embedding models
  - High ranking agreement suggests models see the same signal
  - The geometric pattern is not an artifact of any single embedding space
"""
        )
    elif n_significant >= n_models * 0.8 and mean_spearman > 0.5:
        lines.append(
            """
✓ MODERATE EVIDENCE: SGI likely captures something real.
  - Effect is significant in most models
  - Reasonable ranking agreement
  - Some variation may be due to embedding model differences
"""
        )
    elif n_significant >= n_models * 0.5:
        lines.append(
            """
⚠ WEAK EVIDENCE: SGI may be partially model-dependent.
  - Effect is significant in some but not all models
  - Consider investigating why certain models show weaker signal
"""
        )
    else:
        lines.append(
            """
✗ INSUFFICIENT EVIDENCE: SGI may be an artifact.
  - Effect is not consistent across models
  - The signal may be specific to certain embedding spaces
"""
        )

    lines.append("=" * 70)

    return "\n".join(lines)
