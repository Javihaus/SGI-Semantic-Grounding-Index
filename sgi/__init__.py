"""
SGI: Semantic Grounding Index

Geometric methods for hallucination detection in language model outputs.

This package provides tools for computing and validating the Semantic
Grounding Index (SGI), a metric that detects when language model responses
"drift" from source context based on angular relationships in embedding space.

Reference:
    Marín, J. (2024). "Semantic Grounding Index: Geometric Hallucination
    Detection via Angular Distance Ratios in Embedding Space"
    arXiv:2512.13771

Example:
    >>> from sgi import compute_sgi, load_halueval_qa
    >>> from sentence_transformers import SentenceTransformer
    >>>
    >>> encoder = SentenceTransformer('all-mpnet-base-v2')
    >>> cases = load_halueval_qa(max_samples=100)
    >>>
    >>> for case in cases:
    ...     q_emb = encoder.encode(case.question)
    ...     c_emb = encoder.encode(case.context)
    ...     r_emb = encoder.encode(case.response)
    ...     result = compute_sgi(q_emb, c_emb, r_emb)
    ...     print(f"SGI: {result.sgi:.3f}, Grounded: {case.is_grounded}")
"""

__version__ = "0.1.0"
__author__ = "Javier Marín"

from .analysis import (
    EffectSizeResult,
    compute_cohens_d,
    compute_correlation_matrix,
    compute_effect_size,
    compute_pairwise_correlations,
    compute_roc_curve,
    compute_topk_overlap,
)
from .data import (
    TestCase,
    get_dataset_stats,
    load_halueval_dialogue,
    load_halueval_qa,
    load_truthfulqa,
    print_dataset_summary,
)
from .metrics import (
    SGIResult,
    angular_distance,
    compute_sgi,
    compute_sgi_batch,
)
from .visualization import (
    create_summary_figure,
    plot_correlation_heatmap,
    plot_effect_sizes,
    plot_roc_curves,
    plot_scatter_comparison,
    plot_sgi_distributions,
    set_publication_style,
)

__all__ = [
    # Metrics
    "compute_sgi",
    "compute_sgi_batch",
    "angular_distance",
    "SGIResult",
    # Data
    "load_halueval_qa",
    "load_halueval_dialogue",
    "load_truthfulqa",
    "TestCase",
    "get_dataset_stats",
    "print_dataset_summary",
    # Analysis
    "compute_effect_size",
    "compute_cohens_d",
    "compute_correlation_matrix",
    "compute_pairwise_correlations",
    "compute_topk_overlap",
    "compute_roc_curve",
    "EffectSizeResult",
    # Visualization
    "set_publication_style",
    "plot_effect_sizes",
    "plot_correlation_heatmap",
    "plot_sgi_distributions",
    "plot_scatter_comparison",
    "plot_roc_curves",
    "create_summary_figure",
]
