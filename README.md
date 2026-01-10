# SGI Cross-Model Validation


[![Tests](https://github.com/Javihaus/SEMANTIC_GROUNDING_INDEX/workflows/Tests/badge.svg)](https://github.com/Javihaus/SEMANTIC_GROUNDING_INDEX/actions)
[![codecov](https://codecov.io/gh/Javihaus/SEMANTIC_GROUNDING_INDEX/branch/main/graph/badge.svg)](https://codecov.io/gh/Javihaus/SEMANTIC_GROUNDING_INDEX)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Empirical validation that the Semantic Grounding Index (SGI) captures a property of text rather than artifacts of specific embedding geometries.

## Research Question

> Does SGI measure something fundamental about language model outputs, or is it an artifact of particular embedding spaces?

We answer this by computing SGI across five embedding architectures with different training regimes and dimensionalities. If the signal is real, it should be consistent across models.

## Key Finding

**SGI scores correlate at r=0.85 across five embedding architectures.**

This suggests the "semantic laziness" pattern—where hallucinations remain angularly proximate to questions rather than traveling toward context—is a property of the text itself, not an artifact of any single embedding space.

## Repository Structure

```
sgi-validation/
├── sgi/                          # Core library
│   ├── __init__.py              # Package exports
│   ├── metrics.py               # SGI computation (angular distances)
│   ├── data.py                  # Dataset loaders (HaluEval, TruthfulQA)
│   ├── analysis.py              # Statistical tests (effect sizes, correlations)
│   └── visualization.py         # Publication-quality figures
├── experiments/
│   └── cross_model_validation.ipynb  # Main experiment notebook
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/yourusername/sgi-validation.git
cd sgi-validation
pip install -r requirements.txt
python -m spacy download en_core_web_sm
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

## Models Tested

| Model | Dimension | Source |
|-------|-----------|--------|
| `all-mpnet-base-v2` | 768 | Sentence-Transformers |
| `all-MiniLM-L6-v2` | 384 | Sentence-Transformers |
| `bge-base-en-v1.5` | 768 | BAAI |
| `e5-base-v2` | 768 | Microsoft |
| `gte-base` | 768 | Alibaba |

## Validation Tests

1. **Effect Size Consistency**: Cohen's d significant across all models
2. **Cross-Model Correlation**: Pearson r between SGI scores from different models
3. **Ranking Agreement**: Spearman ρ for sample rankings
4. **Top-K Overlap**: Jaccard similarity of flagged samples
5. **Component Analysis**: Which angular distance drives the signal

## Running the Experiment

```bash
cd experiments
jupyter notebook cross_model_validation.ipynb
```

Or run all cells:

```bash
jupyter nbconvert --to notebook --execute cross_model_validation.ipynb
```

## Results

Typical results on HaluEval QA (500 samples):

| Metric | Value |
|--------|-------|
| Models with significant effect (p<0.05) | 5/5 |
| Mean Cohen's d | ~0.35 |
| Mean AUROC | ~0.65 |
| Mean Pearson r (cross-model) | ~0.85 |
| Mean Spearman ρ (ranking agreement) | ~0.85 |

## Limitations

- SGI alone achieves moderate AUROC (~0.65-0.70); stronger performance requires ensemble methods
- Cross-domain transfer fails: models trained on one domain don't generalize
- Requires source context; not applicable to open-domain generation

## Reference

```bibtex
@article{marin2024sgi,
  title={Semantic Grounding Index: Geometric Hallucination Detection 
         via Angular Distance Ratios in Embedding Space},
  author={Mar{\'\i}n, Javier},
  journal={arXiv preprint arXiv:2512.13771},
  year={2024}
}
```

## License

MIT
