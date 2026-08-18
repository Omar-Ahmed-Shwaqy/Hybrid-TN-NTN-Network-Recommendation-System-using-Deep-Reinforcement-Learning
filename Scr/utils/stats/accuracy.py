# =============================================================================
# FILE: Scr/utils/stats/accuracy.py
# PURPOSE: Accuracy metrics.
# =============================================================================

import numpy as np
from typing import List, Dict, Any, Optional, Union
from collections import Counter

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