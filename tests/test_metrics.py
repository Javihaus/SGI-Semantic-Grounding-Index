"""Tests for sgi.metrics module."""

import numpy as np
import pytest

from sgi.metrics import (
    SGIResult,
    angular_distance,
    compute_sgi,
    compute_sgi_batch,
    euclidean_distance_normalized,
    normalize,
)


class TestNormalize:
    """Tests for the normalize function."""

    def test_normalize_unit_vector(self):
        """Normalized vector should have unit length."""
        v = np.array([3.0, 4.0])
        result = normalize(v)
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_normalize_preserves_direction(self):
        """Normalization should preserve direction."""
        v = np.array([3.0, 4.0])
        result = normalize(v)
        expected = np.array([0.6, 0.8])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_normalize_already_unit(self):
        """Already unit vectors should be unchanged."""
        v = np.array([1.0, 0.0])
        result = normalize(v)
        np.testing.assert_allclose(result, v, rtol=1e-5)

    def test_normalize_high_dimensional(self):
        """Should work on high-dimensional vectors."""
        v = np.random.randn(768)
        result = normalize(v)
        assert np.isclose(np.linalg.norm(result), 1.0)


class TestAngularDistance:
    """Tests for the angular_distance function."""

    def test_identical_vectors_zero_distance(self):
        """Identical vectors should have near-zero angular distance."""
        v = np.array([1.0, 0.0, 0.0])
        # Allow small tolerance due to epsilon in normalization
        assert np.isclose(angular_distance(v, v), 0.0, atol=1e-4)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have pi/2 angular distance."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert np.isclose(angular_distance(v1, v2), np.pi / 2, atol=1e-10)

    def test_opposite_vectors(self):
        """Opposite vectors should have pi angular distance."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        assert np.isclose(angular_distance(v1, v2), np.pi, atol=1e-10)

    def test_symmetry(self):
        """Angular distance should be symmetric."""
        v1 = np.random.randn(128)
        v2 = np.random.randn(128)
        assert np.isclose(angular_distance(v1, v2), angular_distance(v2, v1))

    def test_range(self):
        """Angular distance should be in [0, pi]."""
        for _ in range(10):
            v1 = np.random.randn(64)
            v2 = np.random.randn(64)
            dist = angular_distance(v1, v2)
            assert 0 <= dist <= np.pi + 1e-10


class TestEuclideanDistanceNormalized:
    """Tests for the euclidean_distance_normalized function."""

    def test_identical_vectors(self):
        """Identical vectors should have zero distance."""
        v = np.array([1.0, 2.0, 3.0])
        assert np.isclose(euclidean_distance_normalized(v, v), 0.0, atol=1e-10)

    def test_opposite_vectors(self):
        """Opposite normalized vectors should have distance 2."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        assert np.isclose(euclidean_distance_normalized(v1, v2), 2.0, atol=1e-10)

    def test_orthogonal_vectors(self):
        """Orthogonal unit vectors should have distance sqrt(2)."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert np.isclose(euclidean_distance_normalized(v1, v2), np.sqrt(2), atol=1e-10)

    def test_range(self):
        """Euclidean distance on unit sphere should be in [0, 2]."""
        for _ in range(10):
            v1 = np.random.randn(64)
            v2 = np.random.randn(64)
            dist = euclidean_distance_normalized(v1, v2)
            assert 0 <= dist <= 2 + 1e-10


class TestSGIResult:
    """Tests for the SGIResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SGIResult(
            sgi=1.5,
            theta_rq=0.5,
            theta_rc=0.33,
            theta_qc=0.4,
            d_rq=0.7,
            d_rc=0.5,
        )
        d = result.to_dict()
        assert d["sgi"] == 1.5
        assert d["theta_rq"] == 0.5
        assert d["theta_rc"] == 0.33
        assert d["theta_qc"] == 0.4
        assert d["d_rq"] == 0.7
        assert d["d_rc"] == 0.5
        assert len(d) == 6


class TestComputeSGI:
    """Tests for the compute_sgi function."""

    def test_returns_sgi_result(self, sample_embeddings):
        """compute_sgi should return an SGIResult."""
        q_emb, c_emb, r_emb = sample_embeddings
        result = compute_sgi(q_emb, c_emb, r_emb)
        assert isinstance(result, SGIResult)

    def test_sgi_positive(self, sample_embeddings):
        """SGI should be positive."""
        q_emb, c_emb, r_emb = sample_embeddings
        result = compute_sgi(q_emb, c_emb, r_emb)
        assert result.sgi > 0

    def test_response_close_to_context_high_sgi(self):
        """Response close to context should have high SGI (large theta_rq / small theta_rc)."""
        # Question pointing one direction
        q_emb = np.array([1.0, 0.0, 0.0])
        # Context pointing another
        c_emb = np.array([0.0, 1.0, 0.0])
        # Response close to context
        r_emb = np.array([0.1, 0.99, 0.0])

        result = compute_sgi(q_emb, c_emb, r_emb)
        # SGI = theta_rq / theta_rc
        # theta_rq is large (far from question), theta_rc is small (close to context)
        # So SGI > 1
        assert result.sgi > 1.0

    def test_response_close_to_question_low_sgi(self):
        """Response staying near question should have low SGI (small theta_rq / large theta_rc)."""
        # Question pointing one direction
        q_emb = np.array([1.0, 0.0, 0.0])
        # Context pointing another
        c_emb = np.array([0.0, 1.0, 0.0])
        # Response close to question
        r_emb = np.array([0.99, 0.1, 0.0])

        result = compute_sgi(q_emb, c_emb, r_emb)
        # SGI = theta_rq / theta_rc
        # theta_rq is small (close to question), theta_rc is large (far from context)
        # So SGI < 1
        assert result.sgi < 1.0

    def test_angular_distances_positive(self, sample_embeddings):
        """All angular distances should be non-negative."""
        q_emb, c_emb, r_emb = sample_embeddings
        result = compute_sgi(q_emb, c_emb, r_emb)
        assert result.theta_rq >= 0
        assert result.theta_rc >= 0
        assert result.theta_qc >= 0

    def test_euclidean_distances_in_range(self, sample_embeddings):
        """Euclidean distances should be in [0, 2]."""
        q_emb, c_emb, r_emb = sample_embeddings
        result = compute_sgi(q_emb, c_emb, r_emb)
        assert 0 <= result.d_rq <= 2
        assert 0 <= result.d_rc <= 2


class TestComputeSGIBatch:
    """Tests for the compute_sgi_batch function."""

    def test_output_shapes(self, batch_embeddings):
        """Output arrays should have correct shapes."""
        q_embs, c_embs, r_embs = batch_embeddings
        sgi, theta_rq, theta_rc = compute_sgi_batch(q_embs, c_embs, r_embs)

        n_samples = q_embs.shape[0]
        assert sgi.shape == (n_samples,)
        assert theta_rq.shape == (n_samples,)
        assert theta_rc.shape == (n_samples,)

    def test_all_positive(self, batch_embeddings):
        """All SGI values should be positive."""
        q_embs, c_embs, r_embs = batch_embeddings
        sgi, theta_rq, theta_rc = compute_sgi_batch(q_embs, c_embs, r_embs)

        assert np.all(sgi > 0)
        assert np.all(theta_rq >= 0)
        assert np.all(theta_rc >= 0)

    def test_consistency_with_single(self, batch_embeddings):
        """Batch results should match single computation."""
        q_embs, c_embs, r_embs = batch_embeddings
        sgi_batch, theta_rq_batch, theta_rc_batch = compute_sgi_batch(q_embs, c_embs, r_embs)

        # Compare with single computation for first sample
        result_single = compute_sgi(q_embs[0], c_embs[0], r_embs[0])

        assert np.isclose(sgi_batch[0], result_single.sgi, rtol=1e-5)
        assert np.isclose(theta_rq_batch[0], result_single.theta_rq, rtol=1e-5)
        assert np.isclose(theta_rc_batch[0], result_single.theta_rc, rtol=1e-5)

    def test_empty_batch(self):
        """Should handle empty batches gracefully."""
        q_embs = np.zeros((0, 128))
        c_embs = np.zeros((0, 128))
        r_embs = np.zeros((0, 128))

        sgi, theta_rq, theta_rc = compute_sgi_batch(q_embs, c_embs, r_embs)
        assert len(sgi) == 0
        assert len(theta_rq) == 0
        assert len(theta_rc) == 0
