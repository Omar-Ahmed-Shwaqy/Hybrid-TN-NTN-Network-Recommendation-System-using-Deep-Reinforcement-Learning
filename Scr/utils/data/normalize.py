# =============================================================================
# FILE: Scr/utils/data/normalize.py
# PURPOSE: Data normalization utilities.
# =============================================================================

import numpy as np
from typing import Dict, Union

def safe_divide(
    numerator: Union[float, int],
    denominator: Union[float, int],
    default: float = 0.0
) -> float:
    """Safe division with default value for division by zero."""
    if denominator == 0 or np.isnan(denominator):
        return default
    return numerator / denominator

def normalize_dict_values(
    data: Dict,
    method: str = 'min_max'
) -> Dict:
    """Normalize dictionary values."""
    values = np.array(list(data.values()))
    if method == 'min_max':
        min_val = values.min()
        max_val = values.max()
        if max_val - min_val > 0:
            normalized = (values - min_val) / (max_val - min_val)
        else:
            normalized = np.ones_like(values)
    elif method == 'z_score':
        mean_val = values.mean()
        std_val = values.std()
        if std_val > 0:
            normalized = (values - mean_val) / std_val
        else:
            normalized = np.zeros_like(values)
    elif method == 'robust':
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        if iqr > 0:
            normalized = (values - q1) / iqr
        else:
            normalized = np.zeros_like(values)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return {key: float(val) for key, val in zip(data.keys(), normalized)}