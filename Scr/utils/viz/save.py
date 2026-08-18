# =============================================================================
# FILE: Scr/utils/viz/save.py
# PURPOSE: Figure saving utilities.
# =============================================================================

import os
import matplotlib.pyplot as plt
from typing import List
import logging

logger = logging.getLogger(__name__)

def save_figure(
    fig,
    filename: str,
    output_dir: str = 'results/figures/',
    dpi: int = 300,
    bbox_inches: str = 'tight',
    formats: List[str] = ['png', 'pdf']
) -> None:
    """Save figure in multiple formats."""
    os.makedirs(output_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, format=fmt)
        logger.info(f"Figure saved: {path}")