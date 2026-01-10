# Semantic Grounding Index (SGI)

[![Tests](https://github.com/Javihaus/SEMANTIC_GROUNDING_INDEX/workflows/Tests/badge.svg)](https://github.com/Javihaus/SEMANTIC_GROUNDING_INDEX/actions)
[![codecov](https://codecov.io/gh/Javihaus/SEMANTIC_GROUNDING_INDEX/branch/main/graph/badge.svg)](https://codecov.io/gh/Javihaus/SEMANTIC_GROUNDING_INDEX)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Geometric methods for hallucination detection in RAG systems, based on angular relationships in embedding space.

## Research Question

> Does the angular geometry of embeddings reveal when language models disengage from source context?

We answer this by computing the **Semantic Grounding Index (SGI)**, which measures relative angular proximity of responses to questions versus contexts.

## Key Finding: Semantic Laziness

Hallucinated responses exhibit **semantic laziness** — they remain angularly proximate to questions rather than departing toward contexts. This manifests as higher SGI scores for hallucinations.

## Results

### Main Finding: Cross-Model Effect Sizes (Table 1)

On HaluEval QA (n=5,000), SGI achieves strong discrimination:

| Model | SGI (Valid) | SGI (Halluc) | Cohen's d | AUROC |
|-------|-------------|--------------|-----------|-------|
| mpnet | 1.142 | 0.921 | +0.92 | 0.776 |
| minilm | 1.203 | 0.856 | +1.28 | 0.824 |
| bge | 1.231 | 0.948 | +1.27 | 0.823 |
| e5 | 1.138 | 0.912 | +1.03 | 0.794 |
| gte | 1.224 | 0.927 | +1.13 | 0.811 |
| **Mean** | **1.188** | **0.913** | **+1.13** | **0.806** |

### Cross-Model Robustness

Five embedding models with distinct architectures agree on SGI scores with:
- **Pearson r = 0.85** (linear agreement)
- **Spearman ρ = 0.87** (ranking agreement)

This indicates SGI captures a property of text, not an artifact of embedding geometry.

### Triangle Inequality Confirmation (Table 2)

Effect size increases monotonically with question-context separation θ(q,c):

| θ(q,c) Tercile | Cohen's d | AUROC |
|----------------|-----------|-------|
| Low | +0.61 | 0.721 |
| Medium | +0.90 | 0.768 |
| High | +1.27 | 0.832 |

### Operational Boundaries (Table 3)

SGI excels on:
- Long responses (d = 2.05)
- Short questions (d = 1.22)

SGI fails on:
- TruthfulQA (AUC = 0.478) — angular geometry cannot discriminate factual accuracy

### Calibration

Expected Calibration Error (ECE) = 0.10, indicating SGI scores can serve as probability estimates for risk stratification.

### Displacement Consistency (AUROC ~0.98-1.0)

Advanced trajectory-based methods achieve near-perfect discrimination:

| Method | Cross-Domain AUROC | Within-Domain AUROC |
|--------|-------------------|---------------------|
| Displacement Consistency | 1.0 | 1.0 |
| Expected Response Dev. | 1.0 | 1.0 |
| Displacement Magnitude | 0.9999 | 0.9998 |
| Tangent Space Residual | 0.9990 | 0.9961 |
| LSGI (Local SGI) | 0.914 | 0.807 |

## Installation

```bash
git clone https://github.com/Javihaus/SEMANTIC_GROUNDING_INDEX.git
cd SEMANTIC_GROUNDING_INDEX
pip install -e .
```

## Quick Start

```python
from sgi import compute_sgi, load_halueval_qa
from sentence_transformers import SentenceTransformer

# Load data
cases = load_halueval_qa(max_samples=100)

# Initialize encoder
encoder = SentenceTransformer('all-mpnet-base-v2')

# Compute SGI
for case in cases:
    q_emb = encoder.encode(case.question)
    c_emb = encoder.encode(case.context)
    r_emb = encoder.encode(case.response)

    result = compute_sgi(q_emb, c_emb, r_emb)
    print(f"SGI: {result.sgi:.3f}, Grounded: {case.is_grounded}")
```

## The SGI Metric

$$\text{SGI} = \frac{\theta(r, q)}{\theta(r, c)}$$

Where:
- $\theta$ is the angular distance (arccos of cosine similarity)
- $r$, $q$, $c$ are L2-normalized embeddings of response, question, and context

**Interpretation:**
- SGI < 1: Response closer to context than question (grounded)
- SGI > 1: Response closer to question than context (potentially hallucinated)
- SGI ≈ 1: Equidistant (ambiguous)

## Experiments

This repository contains three main experiment notebooks:

| Notebook | Description |
|----------|-------------|
| `01_sgi_semantic_laziness.ipynb` | Main paper experiments (Tables 1-5, Figures 2-5) |
| `02_cross_model_validation.ipynb` | Cross-model robustness validation (n=5,000) |
| `03_displacement_consistency.ipynb` | Displacement geometry experiments (AUROC ~0.98) |

### Running Experiments

```bash
cd experiments
jupyter notebook 01_sgi_semantic_laziness.ipynb
```

Or run all cells:

```bash
jupyter nbconvert --to notebook --execute 01_sgi_semantic_laziness.ipynb
```


## Models Tested

| Model | Dimension | Source |
|-------|-----------|--------|
| `all-mpnet-base-v2` | 768 | Sentence-Transformers |
| `all-MiniLM-L6-v2` | 384 | Sentence-Transformers |
| `bge-base-en-v1.5` | 768 | BAAI |
| `e5-base-v2` | 768 | Microsoft |
| `gte-base` | 768 | Alibaba |

## Limitations

- SGI measures **topical engagement**, not factual accuracy
- Requires source context; not applicable to open-domain generation
- Performance varies with response length and question-context separation
- Cross-domain transfer requires revalidation

## Reference

```bibtex
@misc{marín2025semanticgroundingindexgeometric,
      title={Semantic Grounding Index: Geometric Bounds on Context Engagement in RAG Systems},
      author={Javier Marín},
      year={2025},
      eprint={2512.13771},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.13771},
}
```

## License

Apache 2.0
