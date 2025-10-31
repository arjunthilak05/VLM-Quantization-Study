"""
Metric computation utilities for VLM evaluation
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Compute accuracy

    Args:
        predictions: List of predicted answers
        targets: List of ground truth answers

    Returns:
        Accuracy score
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")

    correct = sum(p == t for p, t in zip(predictions, targets))
    accuracy = correct / len(predictions)

    return accuracy


def compute_exact_match(predictions: List[str], targets: List[str]) -> float:
    """
    Compute exact match score

    Args:
        predictions: List of predicted answers
        targets: List of ground truth answers

    Returns:
        Exact match score
    """
    return compute_accuracy(predictions, targets)


def compute_vqa_accuracy(predictions: List[str], targets: List[List[str]]) -> float:
    """
    Compute VQA-style accuracy (answer must match at least 3 out of 10 annotators)

    Args:
        predictions: List of predicted answers
        targets: List of lists of ground truth answers (multiple annotators)

    Returns:
        VQA accuracy score
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")

    correct = 0
    for pred, target_list in zip(predictions, targets):
        # Normalize answer
        pred_norm = normalize_answer(pred)

        # Count matches
        matches = sum(1 for t in target_list if normalize_answer(t) == pred_norm)

        # VQA accuracy: min(matches / 3, 1.0)
        correct += min(matches / 3.0, 1.0)

    accuracy = correct / len(predictions)
    return accuracy


def compute_classification_metrics(
    predictions: List[int],
    targets: List[int]
) -> Dict[str, float]:
    """
    Compute classification metrics (precision, recall, F1)

    Args:
        predictions: List of predicted labels
        targets: List of ground truth labels

    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='macro', zero_division=0)
    recall = recall_score(targets, predictions, average='macro', zero_division=0)
    f1 = f1_score(targets, predictions, average='macro', zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison

    Args:
        answer: Answer string

    Returns:
        Normalized answer
    """
    # Convert to lowercase
    answer = answer.lower()

    # Remove punctuation
    answer = answer.strip().strip('.,!?;:')

    # Remove articles
    articles = ['a', 'an', 'the']
    answer_words = answer.split()
    answer_words = [w for w in answer_words if w not in articles]
    answer = ' '.join(answer_words)

    return answer


def compute_metrics(
    predictions: List[Any],
    targets: List[Any],
    metric_type: str = "accuracy"
) -> Dict[str, float]:
    """
    Compute metrics based on type

    Args:
        predictions: List of predictions
        targets: List of targets
        metric_type: Type of metric to compute

    Returns:
        Dictionary of computed metrics
    """
    if metric_type == "accuracy":
        return {"accuracy": compute_accuracy(predictions, targets)}
    elif metric_type == "vqa_accuracy":
        return {"accuracy": compute_vqa_accuracy(predictions, targets)}
    elif metric_type == "classification":
        return compute_classification_metrics(predictions, targets)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


def aggregate_results(results_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate results from multiple runs

    Args:
        results_list: List of result dictionaries

    Returns:
        Aggregated results with mean and std
    """
    if not results_list:
        return {}

    aggregated = {}

    # Get all metric names
    metric_names = set()
    for results in results_list:
        metric_names.update(results.keys())

    # Compute mean and std for each metric
    for metric in metric_names:
        values = [r[metric] for r in results_list if metric in r]

        if values:
            aggregated[f"{metric}_mean"] = np.mean(values)
            aggregated[f"{metric}_std"] = np.std(values)
            aggregated[f"{metric}_min"] = np.min(values)
            aggregated[f"{metric}_max"] = np.max(values)

    return aggregated
