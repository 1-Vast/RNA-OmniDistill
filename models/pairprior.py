"""Biological pair-prior matrix for RNA secondary structure prediction.

Provides build_pair_prior_matrix() which produces a [L, L] prior matrix
that can be added to pair logits before Nussinov decoding.
"""
from __future__ import annotations
import numpy as np
from typing import Optional

BASE_TO_IDX = {'A': 0, 'U': 1, 'G': 2, 'C': 3}

# Canonical pair compatibility matrix [4x4]
CANONICAL_COMPAT = np.array([
    [0.0, 1.0, 0.0, 0.0],  # A-A, A-U, A-G, A-C
    [1.0, 0.0, 0.6, 0.0],  # U-A, U-U, U-G, U-C
    [0.0, 0.6, 0.0, 1.0],  # G-A, G-U, G-G, G-C
    [0.0, 0.0, 1.0, 0.0],  # C-A, C-U, C-G, C-C
], dtype=np.float32)

DEFAULT_CONFIG = {
    "alpha": 0.5,
    "canonical_bonus": 1.0,
    "wobble_bonus": 0.6,
    "incompatible_penalty": -1.0,
    "short_distance_penalty": -0.8,
    "min_loop_length": 3,
    "long_range_bonus": 0.0,
    "stem_continuity_bonus": 0.0,
    "isolated_pair_penalty": 0.0,
    "symmetrize": True,
}


def build_pair_prior_matrix(
    seq: str,
    config: Optional[dict] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Build a biological pair-prior matrix for an RNA sequence.
    
    Args:
        seq: RNA sequence (A/C/G/U).
        config: Prior configuration dict (see DEFAULT_CONFIG).
        alpha: Overall prior strength (overrides config['alpha'] if config is None).
    
    Returns:
        [L, L] float32 prior matrix (symmetric, upper triangle filled).
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if config is None or 'alpha' not in config:
        cfg['alpha'] = alpha
    
    L = len(seq)
    if L < 2:
        return np.zeros((L, L), dtype=np.float32)
    
    prior = np.zeros((L, L), dtype=np.float32)
    ml = int(cfg.get('min_loop_length', 3))
    
    for i in range(L):
        for j in range(i + ml, L):
            bi = seq[i].upper().replace('T', 'U')
            bj = seq[j].upper().replace('T', 'U')
            
            if bi in BASE_TO_IDX and bj in BASE_TO_IDX:
                compat = CANONICAL_COMPAT[BASE_TO_IDX[bi], BASE_TO_IDX[bj]]
                if compat > 0:
                    # AU/GC pairs get canonical_bonus, GU gets wobble_bonus
                    is_wobble = (bi == 'G' and bj == 'U') or (bi == 'U' and bj == 'G')
                    bonus = cfg['wobble_bonus'] if is_wobble else cfg['canonical_bonus']
                    prior[i, j] += bonus * compat
                elif compat == 0:
                    prior[i, j] += cfg.get('incompatible_penalty', -1.0)
            
            # Short distance penalty
            d = j - i
            if d < ml:
                prior[i, j] += cfg.get('short_distance_penalty', -0.8) * (ml - d) / ml
            
            # Long range bonus
            if cfg.get('long_range_bonus', 0) != 0 and d > L * 0.6:
                prior[i, j] += cfg['long_range_bonus'] * (d - L * 0.6) / (L * 0.4)
    
    # Symmetrize
    if cfg.get('symmetrize', True):
        prior = (prior + prior.T) * 0.5
    
    # Apply alpha
    prior *= cfg['alpha']
    
    return prior.astype(np.float32)


def apply_pair_prior_to_logits(
    pair_logits: np.ndarray,
    seq: str,
    config: Optional[dict] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Apply pair prior to logits matrix.
    
    Args:
        pair_logits: [L, L] pair logits from model.
        seq: RNA sequence.
        config: Prior config.
        alpha: Prior strength.
    
    Returns:
        Enhanced [L, L] logits.
    """
    prior = build_pair_prior_matrix(seq, config, alpha)
    L = min(len(seq), pair_logits.shape[0])
    return pair_logits[:L, :L] + prior[:L, :L]
