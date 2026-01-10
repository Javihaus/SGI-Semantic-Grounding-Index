"""Pytest configuration and shared fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    np.random.seed(42)
    dim = 384  # Common embedding dimension

    # Create normalized embeddings
    q_emb = np.random.randn(dim)
    q_emb = q_emb / np.linalg.norm(q_emb)

    c_emb = np.random.randn(dim)
    c_emb = c_emb / np.linalg.norm(c_emb)

    r_emb = np.random.randn(dim)
    r_emb = r_emb / np.linalg.norm(r_emb)

    return q_emb, c_emb, r_emb


@pytest.fixture
def batch_embeddings():
    """Generate batch embeddings for testing."""
    np.random.seed(42)
    n_samples = 10
    dim = 384

    q_embs = np.random.randn(n_samples, dim)
    c_embs = np.random.randn(n_samples, dim)
    r_embs = np.random.randn(n_samples, dim)

    return q_embs, c_embs, r_embs


@pytest.fixture
def sample_test_cases():
    """Generate sample TestCase objects for testing."""
    from sgi.data import TestCase

    cases = [
        TestCase(
            id="test_1",
            question="What is the capital of France?",
            context="Paris is the capital and largest city of France.",
            response="The capital of France is Paris.",
            is_grounded=True,
            source="test",
        ),
        TestCase(
            id="test_2",
            question="What is the capital of France?",
            context="Paris is the capital and largest city of France.",
            response="The capital of France is London.",
            is_grounded=False,
            source="test",
        ),
        TestCase(
            id="test_3",
            question="Who wrote Hamlet?",
            context="William Shakespeare wrote Hamlet around 1600.",
            response="William Shakespeare authored Hamlet.",
            is_grounded=True,
            source="test",
        ),
        TestCase(
            id="test_4",
            question="Who wrote Hamlet?",
            context="William Shakespeare wrote Hamlet around 1600.",
            response="Charles Dickens wrote Hamlet in 1850.",
            is_grounded=False,
            source="test",
        ),
    ]
    return cases


@pytest.fixture
def sample_sgi_dataframe():
    """Generate sample DataFrame with SGI scores for testing."""
    np.random.seed(42)
    n_samples = 100

    # Create DataFrame with SGI scores from multiple models
    df = pd.DataFrame(
        {
            "is_grounded": np.random.choice([True, False], n_samples),
            "sgi_model_a": np.random.uniform(0.5, 1.5, n_samples),
            "sgi_model_b": np.random.uniform(0.5, 1.5, n_samples),
            "sgi_model_c": np.random.uniform(0.5, 1.5, n_samples),
        }
    )

    # Make hallucinated cases have higher SGI on average
    df.loc[~df["is_grounded"], "sgi_model_a"] += 0.3
    df.loc[~df["is_grounded"], "sgi_model_b"] += 0.25
    df.loc[~df["is_grounded"], "sgi_model_c"] += 0.35

    return df


@pytest.fixture
def sample_effect_sizes():
    """Generate sample effect size results for testing."""
    return {
        "model_a": {
            "cohens_d": 0.75,
            "auroc": 0.72,
            "p_value": 0.001,
            "significant": True,
        },
        "model_b": {
            "cohens_d": 0.68,
            "auroc": 0.69,
            "p_value": 0.003,
            "significant": True,
        },
        "model_c": {
            "cohens_d": 0.82,
            "auroc": 0.75,
            "p_value": 0.0001,
            "significant": True,
        },
    }
