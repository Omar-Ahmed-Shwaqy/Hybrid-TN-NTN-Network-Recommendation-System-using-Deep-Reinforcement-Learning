# =============================================================================
# FILE: Scr/utils/data/network.py
# PURPOSE: Network and area utilities.
# =============================================================================

from typing import Dict

# Import constants if available
try:
    from utils.constants import NETWORK_TYPES, AREA_TYPES, NETWORK_COLORS, AREA_COLORS
except ImportError:
    NETWORK_TYPES = ['NR_5G', 'WiFi', 'SAT (LEO)', 'HAPS', 'UAV']
    AREA_TYPES = ['Urban', 'Indoor', 'Rural', 'Highway', 'Maritime', 'Desert']
    NETWORK_COLORS = {
        'NR_5G': '#2E86AB', 'WiFi': '#A23B72', 'SAT (LEO)': '#F18F01',
        'HAPS': '#C73E1D', 'UAV': '#6A994E'
    }
    AREA_COLORS = {
        'Urban': '#2C3E50', 'Indoor': '#E67E22', 'Rural': '#27AE60',
        'Highway': '#2980B9', 'Maritime': '#1ABC9C', 'Desert': '#F39C12'
    }

def get_network_index(network_name: str) -> int:
    """Get index of a network type."""
    if network_name in NETWORK_TYPES:
        return NETWORK_TYPES.index(network_name)
    return -1

def get_area_index(area_name: str) -> int:
    """Get index of an area type."""
    if area_name in AREA_TYPES:
        return AREA_TYPES.index(area_name)
    return -1

def is_ntn_network(network_name: str) -> bool:
    """Check if a network is Non-Terrestrial."""
    ntn_networks = ['SAT (LEO)', 'HAPS', 'UAV']
    return network_name in ntn_networks

def is_tn_network(network_name: str) -> bool:
    """Check if a network is Terrestrial."""
    tn_networks = ['NR_5G', 'WiFi']
    return network_name in tn_networks

def get_network_colors() -> Dict[str, str]:
    """Get network colors from constants."""
    return NETWORK_COLORS

def get_area_colors() -> Dict[str, str]:
    """Get area colors from constants."""
    return AREA_COLORS