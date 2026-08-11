"""Tests protecting the verified v1 regression metric formulae."""

from __future__ import annotations

import math

import numpy as np

from src.metrics import calculate_regression_metrics, concordance_index


def _brute_concordance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    score = 0.0
    comparable = 0
    for left in range(len(y_true)):
        for right in range(left + 1, len(y_true)):
            if y_true[left] == y_true[right]:
                continue
            comparable += 1
            true_direction = np.sign(y_true[left] - y_true[right])
            predicted_direction = np.sign(y_pred[left] - y_pred[right])
            if predicted_direction == 0:
                score += 0.5
            elif predicted_direction == true_direction:
                score += 1.0
    return score / comparable


def test_metrics_match_known_values() -> None:
    metrics = calculate_regression_metrics([1, 2, 3], [1, 3, 2])
    assert math.isclose(metrics["mse"], 2 / 3)
    assert math.isclose(metrics["rmse"], math.sqrt(2 / 3))
    assert math.isclose(metrics["mae"], 2 / 3)
    assert math.isclose(metrics["concordance_index"], 2 / 3)
    assert math.isclose(metrics["pearson_correlation"], 0.5)


def test_fast_concordance_matches_brute_force_with_ties() -> None:
    generator = np.random.default_rng(42)
    y_true = generator.integers(0, 5, size=50).astype(float)
    y_pred = generator.integers(0, 7, size=50).astype(float)
    assert math.isclose(
        concordance_index(y_true, y_pred),
        _brute_concordance(y_true, y_pred),
    )
