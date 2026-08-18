# =============================================================================
# FILE: Scr/utils/__init__.py
# PURPOSE: Main utilities package initialization.
#          Exposes all utility functions from submodules for easy import.
# =============================================================================

from typing import List, Dict, Tuple, Union, Optional, Any

# =============================================================================
# Import from core submodules
# =============================================================================

from utils.core.time_utils import format_time, get_timestamp
from utils.core.seed_utils import set_seed
from utils.core.file_utils import ensure_directory, save_json, load_json

# =============================================================================
# Import from stats submodules
# =============================================================================

from utils.stats.confidence import compute_confidence_interval
from utils.stats.accuracy import compute_accuracy, compute_confusion_matrix, compute_f1_score
from utils.stats.reward import calculate_reward_statistics
from utils.stats.handover import calculate_handover_rate
from utils.stats.area import calculate_area_accuracy

# =============================================================================
# Import from viz submodules
# =============================================================================

try:
    from .viz.style import set_plot_style, create_color_palette
    from .viz.save import save_figure
except ImportError:  # Fallback for non-package execution / tooling resolution
    from utils.viz.style import set_plot_style, create_color_palette
    from utils.viz.save import save_figure

# =============================================================================
# Import from data submodules
# =============================================================================

from utils.data.normalize import normalize_dict_values, safe_divide
from utils.data.network import (
    get_network_index,
    get_area_index,
    is_ntn_network,
    is_tn_network,
    get_network_colors,
    get_area_colors
)

# =============================================================================
# Import from helpers (legacy - for backward compatibility)
# =============================================================================

from utils.helpers import print_section, print_dict

# =============================================================================
# Package metadata
# =============================================================================

__version__ = "1.0.0"
__author__ = "Hybrid Network Recommendation System Team"
__description__ = "Utility functions for TN-NTN hybrid network recommendation system"

# =============================================================================
# Public API - List of all exported functions
# =============================================================================

__all__ = [
    # Core utilities
    'format_time',
    'get_timestamp',
    'set_seed',
    'ensure_directory',
    'save_json',
    'load_json',
    
    # Statistical utilities
    'compute_confidence_interval',
    'compute_accuracy',
    'compute_confusion_matrix',
    'compute_f1_score',
    'calculate_reward_statistics',
    'calculate_handover_rate',
    'calculate_area_accuracy',
    
    # Visualization utilities
    'set_plot_style',
    'create_color_palette',
    'save_figure',
    
    # Data utilities
    'normalize_dict_values',
    'safe_divide',
    'get_network_index',
    'get_area_index',
    'is_ntn_network',
    'is_tn_network',
    'get_network_colors',
    'get_area_colors',
    
    # Legacy helpers
    'print_section',
    'print_dict',
]


# =============================================================================
# Convenience functions
# =============================================================================

def import_all_utils():
    """
    Import all utility functions into global namespace.
    Useful for interactive sessions and notebooks.
    
    Usage:
        from utils import import_all_utils
        import_all_utils()
        
        # Now you can use:
        format_time(100)
        compute_accuracy(pred, true)
        set_plot_style()
    """
    import sys
    import inspect
    
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if not name.startswith('_') and callable(obj) and name not in ['import_all_utils']:
            globals()[name] = obj
            print(f"   Imported: {name}")
    
    print("✅ All utilities imported to global namespace")


def check_dependencies():
    """
    Check if all required dependencies are installed.
    
    Returns:
        Dict: Status of each dependency
    """
    dependencies = {
        'numpy': 'np',
        'pandas': 'pd',
        'matplotlib': 'plt',
        'scipy': 'scipy',
        'seaborn': 'sns',
        'torch': 'torch',
        'sklearn': 'sklearn'
    }
    
    status = {}
    
    for dep_name, import_name in dependencies.items():
        try:
            __import__(import_name)
            status[dep_name] = True
        except ImportError:
            status[dep_name] = False
    
    return status


def print_package_info():
    """
    Print package information and dependency status.
    """
    print("="*80)
    print("UTILS PACKAGE INFORMATION")
    print("="*80)
    print(f"   Version: {__version__}")
    print(f"   Author: {__author__}")
    print(f"   Description: {__description__}")
    print()
    print("📦 Dependencies:")
    deps = check_dependencies()
    for dep, installed in deps.items():
        status = "✅" if installed else "❌"
        print(f"   {status} {dep}")
    print("="*80)


# =============================================================================
# Run package info when file is executed directly
# =============================================================================

if __name__ == "__main__":
    print_package_info()