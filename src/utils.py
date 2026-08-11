"""General utility functions for the DTA project."""

import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import (
    RANDOM_SEED,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SPLITS_DIR,
    ESM2_CACHE_DIR,
    MODELS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    PLOTS_DIR,
    SCREENING_RESULTS_DIR,
)


def set_random_seed(
    seed: int = RANDOM_SEED,
    *,
    deterministic_algorithms: bool = True,
) -> None:
    """Seed Python, NumPy and PyTorch and request deterministic kernels."""

    if seed < 0:
        raise ValueError("seed must be non-negative.")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_data_loader_worker(worker_id: int) -> None:
    """Seed NumPy and Python inside a PyTorch DataLoader worker."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SPLITS_DIR,
        ESM2_CACHE_DIR,
        MODELS_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        PLOTS_DIR,
        SCREENING_RESULTS_DIR,
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


def sha256_file(input_path: str | Path) -> str:
    """Return a SHA-256 digest for provenance checks on a local artifact."""

    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
