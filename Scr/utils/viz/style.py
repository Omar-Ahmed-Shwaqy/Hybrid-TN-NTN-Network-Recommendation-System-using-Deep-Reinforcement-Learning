# =============================================================================
# FILE: Scr/utils/viz/style.py
# PURPOSE: Plot styling utilities.
# =============================================================================

import matplotlib.pyplot as plt
from typing import List
import logging

logger = logging.getLogger(__name__)

def set_plot_style(style: str = 'seaborn-v0_8-whitegrid') -> None:
    """Set consistent plot style."""
    try:
        plt.style.use(style)
    except:
        plt.style.use('default')
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def create_color_palette(n_colors: int, palette_name: str = 'husl') -> List[str]:
    """Create a color palette with n colors."""
    if palette_name == 'husl':
        import seaborn as sns
        return sns.color_palette('husl', n_colors).as_hex()
    elif palette_name == 'viridis':
        import seaborn as sns
        return sns.color_palette('viridis', n_colors).as_hex()
    elif palette_name == 'plasma':
        import seaborn as sns
        return sns.color_palette('plasma', n_colors).as_hex()
    else:
        palette = [
            '#2E86AB', '#A23B72', '#F18F01', '#6A994E', '#C73E1D',
            '#D4A373', '#3D5A80', '#B5838D', '#6D597A', '#BEE9E8'
        ]
        return palette[:n_colors]