# =============================================================================
# FILE: Scr/utils/helpers.py
# PURPOSE: Utility functions used across the entire project.
#          Provides common helper functions for data processing, visualization,
#          logging, and general utilities.
# =============================================================================

import os
import sys
import json
import random
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm, t
import logging
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import constants if available
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: TIME UTILITIES
# =============================================================================

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


# =============================================================================
# SECTION 2: STATISTICAL UTILITIES
# =============================================================================

def compute_confidence_interval(
    data: Union[List[float], np.ndarray],
    confidence: float = 0.95,
    method: str = 't'
) -> Tuple[float, float, float]:
    """Compute confidence interval for data."""
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if n == 0:
        return (0.0, 0.0, 0.0)
    if n == 1:
        return (mean, mean, mean)
    se = std / np.sqrt(n)
    if method == 't':
        critical = t.ppf((1 + confidence) / 2, n - 1)
    else:
        critical = norm.ppf((1 + confidence) / 2)
    margin = critical * se
    return (mean, mean - margin, mean + margin)


def compute_accuracy(
    predictions: List,
    ground_truth: List,
    return_details: bool = False
) -> Union[float, Dict]:
    """Compute accuracy between predictions and ground truth."""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    if len(predictions) == 0:
        return 0.0 if not return_details else {'accuracy': 0.0, 'correct': 0, 'total': 0}
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    total = len(predictions)
    accuracy = correct / total
    if return_details:
        return {'accuracy': accuracy, 'correct': correct, 'total': total, 'error_rate': 1 - accuracy}
    return accuracy


def compute_confusion_matrix(
    predictions: List,
    ground_truth: List,
    labels: Optional[List] = None,
    normalize: bool = False
) -> Dict[str, Any]:
    """Compute confusion matrix."""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    if labels is None:
        labels = sorted(set(ground_truth + predictions))
    n = len(labels)
    matrix = np.zeros((n, n), dtype=np.float64)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    for p, g in zip(predictions, ground_truth):
        i = label_to_idx.get(p, 0)
        j = label_to_idx.get(g, 0)
        matrix[i][j] += 1
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, where=row_sums != 0)
    return {
        'labels': labels,
        'matrix': matrix.tolist(),
        'accuracy': np.trace(matrix) / np.sum(matrix) if np.sum(matrix) > 0 else 0,
        'per_class_accuracy': {
            labels[i]: matrix[i][i] / matrix[i].sum() if matrix[i].sum() > 0 else 0
            for i in range(n)
        }
    }


def compute_f1_score(
    predictions: List,
    ground_truth: List,
    average: str = 'binary'
) -> Union[float, Dict]:
    """Compute F1 score."""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    labels = sorted(set(ground_truth + predictions))
    if len(labels) == 2 and average == 'binary':
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == 1 and g == 1)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == 1 and g == 0)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p == 0 and g == 1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    per_class_f1 = {}
    total_f1 = 0
    for label in labels:
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p != label and g == label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        per_class_f1[label] = f1
        total_f1 += f1
    if average == 'micro':
        total_tp = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        total_fp = sum(1 for p, g in zip(predictions, ground_truth) if p != g)
        return total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    elif average == 'macro':
        return total_f1 / len(labels) if labels else 0
    elif average == 'weighted':
        counts = Counter(ground_truth)
        weights = {label: counts[label] / len(ground_truth) for label in labels}
        return sum(per_class_f1[label] * weights[label] for label in labels)
    return per_class_f1


# =============================================================================
# SECTION 3: VISUALIZATION UTILITIES
# =============================================================================

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


def get_network_colors() -> Dict[str, str]:
    """Get network colors from constants."""
    return NETWORK_COLORS


def get_area_colors() -> Dict[str, str]:
    """Get area colors from constants."""
    return AREA_COLORS


# =============================================================================
# SECTION 4: RANDOM SEED UTILITIES
# =============================================================================

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


# =============================================================================
# SECTION 5: DATA UTILITIES
# =============================================================================

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


# =============================================================================
# SECTION 6: METRIC UTILITIES
# =============================================================================

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


def calculate_reward_statistics(
    rewards: List[float],
    confidence: float = 0.95
) -> Dict[str, float]:
    """Calculate statistics for rewards."""
    if not rewards:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0, 'sum': 0.0, 'count': 0}
    mean, ci_lower, ci_upper = compute_confidence_interval(rewards, confidence)
    return {
        'mean': mean,
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'median': np.median(rewards),
        'sum': np.sum(rewards),
        'count': len(rewards),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence': confidence
    }


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


# =============================================================================
# SECTION 7: NETWORK AND AREA UTILITIES
# =============================================================================

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


# =============================================================================
# SECTION 8: LOGGING AND PRINTING UTILITIES
# =============================================================================

def print_section(
    title: str,
    char: str = '=',
    width: int = 80,
    newline: bool = True
) -> None:
    """Print a formatted section header."""
    if newline:
        print()
    padding = (width - len(title) - 2) // 2
    if padding > 0:
        print(f"{char * padding} {title} {char * (width - len(title) - 2 - padding)}")
    else:
        print(f"{title}")


def print_dict(
    data: Dict,
    indent: int = 2,
    prefix: str = '',
    max_depth: int = 3,
    current_depth: int = 0
) -> None:
    """Pretty print a dictionary."""
    if current_depth > max_depth:
        print(f"{prefix}... ({len(data)} items)")
        return
    spaces = ' ' * indent
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_dict(value, indent, prefix + spaces, max_depth, current_depth + 1)
        elif isinstance(value, (list, tuple)):
            print(f"{prefix}{key}: {len(value)} items")
            if len(value) > 0 and current_depth < max_depth:
                print(f"{prefix}{spaces}First: {value[0]}")
                if len(value) > 1:
                    print(f"{prefix}{spaces}Last: {value[-1]}")
        else:
            print(f"{prefix}{key}: {value}")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING HELPERS")
    print("="*80)
    
    print("\nTime Utilities:")
    print(f"   format_time(3661.5): {format_time(3661.5)}")
    print(f"   get_timestamp(): {get_timestamp()}")
    
    print("\nStatistical Utilities:")
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    mean, lower, upper = compute_confidence_interval(data)
    print(f"   Confidence Interval (95%): mean={mean:.2f}, [{lower:.2f}, {upper:.2f}]")
    
    predictions = [1, 0, 1, 1, 0]
    ground_truth = [1, 0, 0, 1, 1]
    acc = compute_accuracy(predictions, ground_truth)
    print(f"   Accuracy: {acc:.2f}")
    
    cm = compute_confusion_matrix(predictions, ground_truth, labels=[0, 1])
    print(f"   Confusion Matrix: {cm['matrix']}")
    
    print("\nVisualization Utilities:")
    set_plot_style()
    colors = create_color_palette(5)
    print(f"   Color palette: {colors}")
    
    print("\nData Utilities:")
    test_dict = {'a': 1, 'b': 2, 'c': 3}
    normalized = normalize_dict_values(test_dict, method='min_max')
    print(f"   Normalized: {normalized}")
    
    print("\nMetric Utilities:")
    networks = ['5G', 'WiFi', '5G', 'WiFi', 'LEO']
    hr = calculate_handover_rate(networks)
    print(f"   Handover Rate: {hr['rate']:.2f}")
    
    rewards = [0.5, 0.6, 0.7, 0.8, 0.9]
    stats_dict = calculate_reward_statistics(rewards)
    print(f"   Reward Stats: mean={stats_dict['mean']:.2f}, std={stats_dict['std']:.2f}")
    
    print("\nNetwork Utilities:")
    print(f"   Is LEO NTN? {is_ntn_network('SAT (LEO)')}")
    print(f"   Is 5G TN? {is_tn_network('NR_5G')}")
    
    print("\nHelpers test completed successfully!")