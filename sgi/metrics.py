"""
Semantic Grounding Index (SGI) Computation

Core geometric metrics for hallucination detection based on angular
relationships in embedding space.

Reference: Marín (2024) "Semantic Grounding Index" arXiv:2512.13771
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SGIResult:
    """Container for SGI computation results."""
    sgi: float
    theta_rq: float  # Angular distance: response to question
    theta_rc: float  # Angular distance: response to context
    theta_qc: float  # Angular distance: question to context
    d_rq: float      # Euclidean distance: response to question (normalized)
    d_rc: float      # Euclidean distance: response to context (normalized)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'sgi': self.sgi,
            'theta_rq': self.theta_rq,
            'theta_rc': self.theta_rc,
            'theta_qc': self.theta_qc,
            'd_rq': self.d_rq,
            'd_rc': self.d_rc,
        }


def normalize(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """L2 normalize a vector."""
    norm = np.linalg.norm(v)
    return v / (norm + eps)


def angular_distance(emb1: np.ndarray, emb2: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute angular distance (geodesic) on unit hypersphere.
    
    Args:
        emb1: First embedding vector
        emb2: Second embedding vector
        eps: Small constant for numerical stability
        
    Returns:
        Angular distance in radians [0, π]
    """
    emb1_norm = normalize(emb1, eps)
    emb2_norm = normalize(emb2, eps)
    
    cos_sim = np.clip(np.dot(emb1_norm, emb2_norm), -1.0, 1.0)
    return float(np.arccos(cos_sim))


def euclidean_distance_normalized(emb1: np.ndarray, emb2: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute Euclidean distance between L2-normalized vectors.
    
    Args:
        emb1: First embedding vector
        emb2: Second embedding vector
        eps: Small constant for numerical stability
        
    Returns:
        Euclidean distance [0, 2]
    """
    emb1_norm = normalize(emb1, eps)
    emb2_norm = normalize(emb2, eps)
    return float(np.linalg.norm(emb1_norm - emb2_norm))


def compute_sgi(
    q_emb: np.ndarray,
    c_emb: np.ndarray,
    r_emb: np.ndarray,
    eps: float = 1e-10
) -> SGIResult:
    """
    Compute Semantic Grounding Index and related metrics.
    
    The SGI measures whether a response is "grounded" in the context
    or "lazy" (staying close to the question without engaging context).
    
    SGI = θ(r, q) / θ(r, c)
    
    - SGI < 1: Response closer to context than question (grounded)
    - SGI > 1: Response closer to question than context (potentially hallucinated)
    - SGI ≈ 1: Response equidistant (ambiguous)
    
    Args:
        q_emb: Question embedding
        c_emb: Context embedding  
        r_emb: Response embedding
        eps: Small constant for numerical stability
        
    Returns:
        SGIResult with all computed metrics
    """
    # Angular distances
    theta_rq = angular_distance(r_emb, q_emb, eps)
    theta_rc = angular_distance(r_emb, c_emb, eps)
    theta_qc = angular_distance(q_emb, c_emb, eps)
    
    # SGI ratio
    sgi = theta_rq / (theta_rc + eps)
    
    # Euclidean distances (normalized space)
    d_rq = euclidean_distance_normalized(r_emb, q_emb, eps)
    d_rc = euclidean_distance_normalized(r_emb, c_emb, eps)
    
    return SGIResult(
        sgi=sgi,
        theta_rq=theta_rq,
        theta_rc=theta_rc,
        theta_qc=theta_qc,
        d_rq=d_rq,
        d_rc=d_rc,
    )


def compute_sgi_batch(
    q_embs: np.ndarray,
    c_embs: np.ndarray,
    r_embs: np.ndarray,
    eps: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SGI for batches of embeddings (vectorized).
    
    Args:
        q_embs: Question embeddings (N, D)
        c_embs: Context embeddings (N, D)
        r_embs: Response embeddings (N, D)
        eps: Small constant for numerical stability
        
    Returns:
        Tuple of (sgi_scores, theta_rq, theta_rc) arrays
    """
    # Normalize
    q_norm = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + eps)
    c_norm = c_embs / (np.linalg.norm(c_embs, axis=1, keepdims=True) + eps)
    r_norm = r_embs / (np.linalg.norm(r_embs, axis=1, keepdims=True) + eps)
    
    # Cosine similarities
    cos_rq = np.clip(np.sum(r_norm * q_norm, axis=1), -1.0, 1.0)
    cos_rc = np.clip(np.sum(r_norm * c_norm, axis=1), -1.0, 1.0)
    
    # Angular distances
    theta_rq = np.arccos(cos_rq)
    theta_rc = np.arccos(cos_rc)
    
    # SGI
    sgi = theta_rq / (theta_rc + eps)
    
    return sgi, theta_rq, theta_rc
