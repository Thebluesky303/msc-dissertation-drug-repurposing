"""Evaluation metrics for drug-target affinity prediction."""

import numpy as np
import torch


def _to_numpy(values) -> np.ndarray:
    """Convert tensors or array-like values to a one-dimensional NumPy array."""

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()

    return np.asarray(values, dtype=float).reshape(-1)


def _prepare_values(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Prepare and validate true and predicted affinity values."""

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "y_true and y_pred must contain the same number of values."
        )

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if y_true.size == 0:
        raise ValueError("No valid values are available for evaluation.")

    return y_true, y_pred


def mean_squared_error(y_true, y_pred) -> float:
    """Calculate mean squared error."""

    y_true, y_pred = _prepare_values(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred) -> float:
    """Calculate root mean squared error."""

    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true, y_pred) -> float:
    """Calculate mean absolute error."""

    y_true, y_pred = _prepare_values(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson_correlation(y_true, y_pred) -> float:
    """Calculate Pearson correlation between true and predicted values."""

    y_true, y_pred = _prepare_values(y_true, y_pred)

    if y_true.size < 2:
        return float("nan")

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def concordance_index(y_true, y_pred) -> float:
    """
    Calculate the concordance index.

    A value of 1.0 indicates perfect ranking, 0.5 indicates random ranking,
    and values below 0.5 indicate poor ranking agreement.
    """

    y_true, y_pred = _prepare_values(y_true, y_pred)

    if y_true.size < 2:
        return float("nan")

    # Rank the predicted values so comparisons can be counted efficiently.
    unique_predictions = np.unique(y_pred)
    prediction_ranks = np.searchsorted(unique_predictions, y_pred) + 1

    order = np.argsort(y_true, kind="mergesort")
    sorted_true = y_true[order]
    sorted_ranks = prediction_ranks[order]

    tree = np.zeros(len(unique_predictions) + 1, dtype=np.int64)

    def update(index: int) -> None:
        while index < len(tree):
            tree[index] += 1
            index += index & -index

    def query(index: int) -> int:
        total = 0
        while index > 0:
            total += tree[index]
            index -= index & -index
        return total

    concordant = 0.0
    comparable = 0
    previous_count = 0

    start = 0

    while start < len(sorted_true):
        end = start + 1

        while (
            end < len(sorted_true)
            and sorted_true[end] == sorted_true[start]
        ):
            end += 1

        # Compare this true-value group only with lower true values.
        for rank in sorted_ranks[start:end]:
            lower_predictions = query(rank - 1)
            equal_predictions = query(rank) - lower_predictions

            concordant += lower_predictions
            concordant += 0.5 * equal_predictions
            comparable += previous_count

        # Add the current group only after all comparisons are complete,
        # so pairs with equal true affinity are not counted.
        for rank in sorted_ranks[start:end]:
            update(int(rank))

        previous_count += end - start
        start = end

    if comparable == 0:
        return float("nan")

    return float(concordant / comparable)


def calculate_metrics(y_true, y_pred) -> dict:
    """Calculate all regression metrics used in the project."""

    y_true, y_pred = _prepare_values(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    ci = concordance_index(y_true, y_pred)
    pearson = pearson_correlation(y_true, y_pred)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "ci": ci,
        "pearson": pearson,
    }
