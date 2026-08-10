"""General utility functions for the DTA project."""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import (
    RANDOM_SEED,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    PLOTS_DIR,
)


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_project_directories() -> None:
    """Create directories used for generated project outputs."""

    directories = [
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        PLOTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable model parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def get_timestamp() -> str:
    """Return the current date and time for experiment logging."""

    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_json(data: Any, output_path: str | Path) -> None:
    """Save a dictionary or list as a formatted JSON file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)


def load_json(input_path: str | Path) -> Any:
    """Load data from a JSON file."""

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"JSON file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)
