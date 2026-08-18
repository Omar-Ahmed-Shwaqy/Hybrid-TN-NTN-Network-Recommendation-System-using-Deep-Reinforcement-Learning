# =============================================================================
# FILE: Scr/utils/stats/reward.py
# PURPOSE: Reward statistics.
# =============================================================================

import numpy as np
from typing import List, Dict
from utils.stats.confidence import compute_confidence_interval

def calculate_reward_statistics(
    rewards: List[float],
    confidence: float = 0.95
) -> Dict[str, float]:
    """Calculate statistics for rewards."""
    if not rewards:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0, 'sum': 0.0, 'count': 0}
    mean, ci_lower, ci_upper = compute_confidence_interval(rewards, confidence)
    return {
        'mean': mean,
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'median': np.median(rewards),
        'sum': np.sum(rewards),
        'count': len(rewards),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence': confidence
    }