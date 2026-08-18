# =============================================================================
# FILE: Scr/utils/stats/confidence.py
# PURPOSE: Confidence interval calculation.
# =============================================================================

import numpy as np
from scipy.stats import norm, t
from typing import List, Tuple, Union

def compute_confidence_interval(
    data: Union[List[float], np.ndarray],
    confidence: float = 0.95,
    method: str = 't'
) -> Tuple[float, float, float]:
    """Compute confidence interval for data."""
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if n == 0:
        return (0.0, 0.0, 0.0)
    if n == 1:
        return (mean, mean, mean)
    se = std / np.sqrt(n)
    if method == 't':
        critical = t.ppf((1 + confidence) / 2, n - 1)
    else:
        critical = norm.ppf((1 + confidence) / 2)
    margin = critical * se
    return (mean, mean - margin, mean + margin)