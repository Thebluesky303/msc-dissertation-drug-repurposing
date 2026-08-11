"""Utility functions used across the project."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the best available PyTorch device."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_folder(path: Path | str) -> Path:
    """Create a folder if it does not already exist."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, file_path: Path | str) -> None:
    """Save a dictionary as a JSON file."""

    file_path = Path(file_path)
    create_folder(file_path.parent)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_results_csv(
    results: dict,
    file_path: Path | str,
) -> None:
    """Save one experiment result as a single-row CSV file."""

    file_path = Path(file_path)
    create_folder(file_path.parent)

    dataframe = pd.DataFrame([results])
    dataframe.to_csv(file_path, index=False)


def save_predictions_csv(
    predictions: pd.DataFrame,
    file_path: Path | str,
) -> None:
    """Save model predictions to a CSV file."""

    file_path = Path(file_path)
    create_folder(file_path.parent)

    predictions.to_csv(file_path, index=False)


def print_experiment_summary(results: dict) -> None:
    """Print the main experiment results in a readable format."""

    print("\nExperiment summary")
    print("-" * 40)

    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("-" * 40)
