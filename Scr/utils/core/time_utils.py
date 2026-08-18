# =============================================================================
# FILE: Scr/utils/core/time_utils.py
# PURPOSE: Time utilities.
# =============================================================================

from datetime import datetime

def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS format."""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"

def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')