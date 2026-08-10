"""Evaluation metrics for drug–target affinity regression."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _prepare_arrays(
    true_values,
    predicted_values,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to validated one-dimensional NumPy arrays."""

    y_true = np.asarray(true_values, dtype=float).reshape(-1)
    y_pred = np.asarray(predicted_values, dtype=float).reshape(-1)

    if y_true.size == 0:
        raise ValueError("Metric inputs cannot be empty.")

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "True and predicted values must have the same shape."
        )

    if not np.isfinite(y_true).all():
        raise ValueError("True values contain non-finite values.")

    if not np.isfinite(y_pred).all():
        raise ValueError("Predicted values contain non-finite values.")

    return y_true, y_pred


def mean_squared_error(
    true_values,
    predicted_values,
) -> float:
    """Calculate mean squared error."""

    y_true, y_pred = _prepare_arrays(
        true_values,
        predicted_values,
    )

    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(
    true_values,
    predicted_values,
) -> float:
    """Calculate root mean squared error."""

    return float(
        np.sqrt(
            mean_squared_error(
                true_values,
                predicted_values,
            )
        )
    )


def mean_absolute_error(
    true_values,
    predicted_values,
) -> float:
    """Calculate mean absolute error."""

    y_true, y_pred = _prepare_arrays(
        true_values,
        predicted_values,
    )

    return float(np.mean(np.abs(y_true - y_pred)))


class _FenwickTree:
    """Fenwick tree used for efficient concordance calculation."""

    def __init__(self, size: int) -> None:
        self.tree = np.zeros(size + 1, dtype=np.int64)

    def update(self, index: int, value: int = 1) -> None:
        """Add a value at a one-based index."""

        while index < len(self.tree):
            self.tree[index] += value
            index += index & -index

    def query(self, index: int) -> int:
        """Return the cumulative count up to a one-based index."""

        total = 0

        while index > 0:
            total += int(self.tree[index])
            index -= index & -index

        return total


def concordance_index(
    true_values,
    predicted_values,
) -> float:
    """
    Calculate the concordance index efficiently.

    A pair is concordant when the sample with the higher true affinity
    also has the higher predicted affinity.

    Prediction ties contribute 0.5.
    Pairs with identical true values are excluded.
    """

    y_true, y_pred = _prepare_arrays(
        true_values,
        predicted_values,
    )

    if y_true.size < 2:
        return float("nan")

    prediction_values = np.unique(y_pred)

    prediction_ranks = (
        np.searchsorted(prediction_values, y_pred) + 1
    )

    sorted_indices = np.argsort(
        y_true,
        kind="mergesort",
    )

    sorted_true = y_true[sorted_indices]
    sorted_ranks = prediction_ranks[sorted_indices]

    tree = _FenwickTree(len(prediction_values))

    concordant_score = 0.0
    comparable_pairs = 0
    previously_added = 0

    group_start = 0
    total_samples = len(sorted_true)

    while group_start < total_samples:
        group_end = group_start + 1

        while (
            group_end < total_samples
            and sorted_true[group_end] == sorted_true[group_start]
        ):
            group_end += 1

        # Compare this true-value group only against lower true values.
        for rank in sorted_ranks[group_start:group_end]:
            lower_predictions = tree.query(rank - 1)
            equal_predictions = (
                tree.query(rank) - tree.query(rank - 1)
            )

            concordant_score += (
                lower_predictions
                + 0.5 * equal_predictions
            )

            comparable_pairs += previously_added

        # Add this group after comparisons to exclude true-value ties.
        for rank in sorted_ranks[group_start:group_end]:
            tree.update(int(rank))

        previously_added += group_end - group_start
        group_start = group_end

    if comparable_pairs == 0:
        return float("nan")

    return float(concordant_score / comparable_pairs)


def pearson_correlation(
    true_values,
    predicted_values,
) -> float:
    """Calculate Pearson correlation."""

    y_true, y_pred = _prepare_arrays(
        true_values,
        predicted_values,
    )

    if y_true.size < 2:
        return float("nan")

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")

    correlation_matrix = np.corrcoef(y_true, y_pred)

    return float(correlation_matrix[0, 1])


def calculate_regression_metrics(
    true_values,
    predicted_values,
) -> Dict[str, float]:
    """Calculate all regression metrics used by the project."""

    return {
        "mse": mean_squared_error(
            true_values,
            predicted_values,
        ),
        "rmse": root_mean_squared_error(
            true_values,
            predicted_values,
        ),
        "mae": mean_absolute_error(
            true_values,
            predicted_values,
        ),
        "concordance_index": concordance_index(
            true_values,
            predicted_values,
        ),
        "pearson_correlation": pearson_correlation(
            true_values,
            predicted_values,
        ),
    }
