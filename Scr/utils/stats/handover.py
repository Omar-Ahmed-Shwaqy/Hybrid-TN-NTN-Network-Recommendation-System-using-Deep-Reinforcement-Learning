# =============================================================================
# FILE: Scr/utils/stats/handover.py
# PURPOSE: Handover rate calculation.
# =============================================================================

from typing import List, Dict

def calculate_handover_rate(
    network_sequence: List[str]
) -> Dict[str, float]:
    """Calculate handover rate from network sequence."""
    if len(network_sequence) <= 1:
        return {'rate': 0.0, 'count': 0, 'total': len(network_sequence)}
    handovers = sum(
        1 for i in range(1, len(network_sequence))
        if network_sequence[i] != network_sequence[i-1]
    )
    transitions = {}
    for i in range(1, len(network_sequence)):
        from_net = network_sequence[i-1]
        to_net = network_sequence[i]
        key = f"{from_net}->{to_net}"
        transitions[key] = transitions.get(key, 0) + 1
    return {
        'rate': handovers / (len(network_sequence) - 1) if len(network_sequence) > 1 else 0.0,
        'count': handovers,
        'total': len(network_sequence),
        'transitions': transitions
    }