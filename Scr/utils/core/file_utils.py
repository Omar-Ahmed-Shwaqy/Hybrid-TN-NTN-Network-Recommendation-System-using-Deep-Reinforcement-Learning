# =============================================================================
# FILE: Scr/utils/core/file_utils.py
# PURPOSE: File utilities.
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def ensure_directory(path: str) -> None:
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)

def save_json(data: Dict, path: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    ensure_directory(os.path.dirname(path))
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        return obj
    
    serializable = convert_to_serializable(data)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=indent, ensure_ascii=False)
    logger.info(f"JSON saved to: {path}")

def load_json(path: str) -> Dict:
    """Load data from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"JSON loaded from: {path}")
    return data