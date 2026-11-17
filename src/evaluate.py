"""Reusable validation and evaluation helpers for the UPI analyses."""
from collections import Counter
import math

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

LABELS = ("negative", "neutral", "positive")


def _labels(values, name):
    values = list(values)
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(value is None or pd.isna(value) for value in values):
        raise ValueError(f"{name} contains missing values")
    return values


def _aligned_labels(y_true, y_pred):
    truth = list(y_true)
    prediction = list(y_pred)
    if len(truth) != len(prediction):
        raise ValueError("y_true and y_pred must have equal lengths")
    truth = _labels(truth, "y_true")
    prediction = _labels(prediction, "y_pred")
    return truth, prediction


def classification_metrics(y_true, y_pred, labels=LABELS):
    """Return fixed-label metrics and a confusion matrix for non-empty aligned data."""
    truth, prediction = _aligned_labels(y_true, y_pred)
    unexpected = (set(truth) | set(prediction)) - set(labels)
    if unexpected:
        raise ValueError(f"Unexpected classification labels: {sorted(unexpected)}")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=list(labels), zero_division=0
    )
    return {
        "n": len(truth),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(truth, prediction, labels=list(labels)).tolist(),
        "classification_report": classification_report(
            truth, prediction, labels=list(labels), output_dict=True, zero_division=0
        ),
    }


def mcnemar_test(y_true, prediction_a, prediction_b):
    """McNemar's continuity-corrected statistic and two-sided chi-square p-value."""
    truth, pred_a = _aligned_labels(y_true, prediction_a)
    truth_b, pred_b = _aligned_labels(y_true, prediction_b)
    if truth != truth_b:
        raise ValueError("All McNemar inputs must have equal lengths and order")
    b = sum(a == actual and second != actual for actual, a, second in zip(truth, pred_a, pred_b))
    c = sum(a != actual and second == actual for actual, a, second in zip(truth, pred_a, pred_b))
    discordant = b + c
    statistic = 0.0 if discordant == 0 else (max(abs(b - c) - 1, 0) ** 2) / discordant
    return {
        "n": len(truth),
        "a_correct_b_wrong": int(b),
        "a_wrong_b_correct": int(c),
        "discordant_pairs": int(discordant),
        "statistic": float(statistic),
        "p_value": float(chi2.sf(statistic, df=1)),
        "continuity_corrected": True,
    }


def wilson_interval(successes, n, alpha=0.05):
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if isinstance(successes, bool) or isinstance(n, bool):
        raise ValueError("successes and n must be integers")
    if int(successes) != successes or int(n) != n or n <= 0 or successes < 0 or successes > n:
        raise ValueError("Require integer counts with 0 <= successes <= n and n > 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    z = float(norm.ppf(1 - alpha / 2))
    proportion = successes / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    lower = 0.0 if successes == 0 else max(0.0, center - margin)
    upper = 1.0 if successes == n else min(1.0, center + margin)
    return lower, upper


def sentiment_distribution(values):
    """Return counts and proportions for the three supported sentiment labels."""
    values = _labels(values, "sentiment labels")
    unexpected = set(values) - set(LABELS)
    if unexpected:
        raise ValueError(f"Unexpected sentiment labels: {sorted(unexpected)}")
    counts = Counter(values)
    return {
        "n": len(values),
        "counts": {label: int(counts[label]) for label in LABELS},
        "proportions": {label: counts[label] / len(values) for label in LABELS},
    }


def validate_sampling(frame, rating_col="score", target_per_rating=1000):
    """Validate benchmark sample sizes for all five rating strata."""
    if rating_col not in frame.columns:
        raise KeyError(f"Missing rating column {rating_col!r}")
    ratings = frame[rating_col]
    invalid = ratings.notna() & ~ratings.isin([1, 2, 3, 4, 5])
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} invalid ratings")
    counts = ratings.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    if (counts > target_per_rating).any():
        raise ValueError("A rating stratum exceeds the requested sample target")
    return {int(rating): int(count) for rating, count in counts.items()}


def topic_distribution(assignments, num_topics=None):
    """Summarize non-negative integer topic assignments and reject invalid IDs."""
    values = _labels(assignments, "topic assignments")
    if any(
        value is None
        or pd.isna(value)
        or isinstance(value, bool)
        or int(value) != value
        or value < 0
        for value in values
    ):
        raise ValueError("Topic assignments must be non-negative integers")
    values = [int(value) for value in values]
    num_topics = max(values) + 1 if num_topics is None else num_topics
    if isinstance(num_topics, bool) or int(num_topics) != num_topics or num_topics <= max(values):
        raise ValueError("num_topics must be an integer greater than every topic ID")
    counts = Counter(values)
    return {
        topic: {"count": int(counts[topic]), "proportion": counts[topic] / len(values)}
        for topic in range(int(num_topics))
    }
