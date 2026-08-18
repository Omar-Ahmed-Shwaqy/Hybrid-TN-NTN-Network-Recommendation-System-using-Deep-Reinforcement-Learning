# =============================================================================
# FILE: Scr/utils/core/seed_utils.py
# PURPOSE: Random seed utilities.
# =============================================================================

import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    logger.info(f"Random seed set to: {seed}")