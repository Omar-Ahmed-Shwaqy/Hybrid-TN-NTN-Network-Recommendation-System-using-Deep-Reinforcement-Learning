# =============================================================================
# FILE: Scr/utils/stats/area.py
# PURPOSE: Area accuracy calculation.
# =============================================================================

from typing import List, Dict

def calculate_area_accuracy(
    results: List[Dict],
    area_key: str = 'area',
    correct_key: str = 'correct'
) -> Dict[str, float]:
    """Calculate accuracy per area from episode data."""
    area_data = {}
    for step in results:
        area = step.get(area_key, 'Unknown')
        correct = step.get(correct_key, False)
        if area not in area_data:
            area_data[area] = {'correct': 0, 'total': 0}
        area_data[area]['total'] += 1
        if correct:
            area_data[area]['correct'] += 1
    return {
        area: data['correct'] / data['total'] if data['total'] > 0 else 0.0
        for area, data in area_data.items()
    }