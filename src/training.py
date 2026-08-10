"""Training, evaluation and experiment logging utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from config import (
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MINIMUM_IMPROVEMENT,
    EXPERIMENT_LOG_PATH,
)
from src.metrics import calculate_regression_metrics
from src.utils import count_trainable_parameters, get_timestamp


def train_one_epoch(
    model: nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """Train the model for one epoch and return average MSE loss."""

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in data_loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        predictions = model(batch)
        true_values = batch.y.float().view(-1)

        loss = loss_function(predictions, true_values)

        loss.backward()
        optimizer.step()

        batch_size = true_values.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training DataLoader contains no samples.")

    return total_loss / total_samples


@torch.no_grad()
def predict(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions and return:

        true_values
        predicted_values
        sample_indices
    """

    model.eval()

    all_true_values: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_sample_indices: list[np.ndarray] = []

    for batch in data_loader:
        batch = batch.to(device)

        predictions = model(batch)

        all_predictions.append(
            predictions.detach().cpu().numpy().reshape(-1)
        )

        if hasattr(batch, "y"):
            all_true_values.append(
                batch.y.detach().cpu().numpy().reshape(-1)
            )

        # if hasattr(batch, "sample_index"):
        #     all_sample_indices.append(
        #         batch.sample_index
        #         .detach()
        #         .cpu()
        #         .numpy()
        #         .reshape(-1)
        #     )
        if hasattr(batch, "sample_id"):
            all_sample_indices.append(
                batch.sample_id
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

    if not all_predictions:
        raise ValueError("Prediction DataLoader contains no samples.")

    predicted_values = np.concatenate(all_predictions)

    if all_true_values:
        true_values = np.concatenate(all_true_values)
    else:
        true_values = np.array([], dtype=float)

    if all_sample_indices:
        sample_indices = np.concatenate(all_sample_indices)
    else:
        sample_indices = np.arange(len(predicted_values))

    return true_values, predicted_values, sample_indices


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a model using all project regression metrics."""

    true_values, predicted_values, _ = predict(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    if true_values.size == 0:
        raise ValueError(
            "Evaluation requires affinity labels."
        )

    return calculate_regression_metrics(
        true_values=true_values,
        predicted_values=predicted_values,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_metrics: dict[str, float],
    checkpoint_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model weights and experiment information."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": validation_metrics,
        "metadata": metadata or {},
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load saved weights into an existing model instance."""

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return checkpoint


def fit_model(
    model: nn.Module,
    train_loader,
    validation_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    checkpoint_path: str | Path,
    num_epochs: int = NUM_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    minimum_improvement: float = MINIMUM_IMPROVEMENT,
    metadata: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Train with validation-based early stopping.

    The checkpoint with the lowest validation MSE is retained.
    """

    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1.")

    if patience < 1:
        raise ValueError("patience must be at least 1.")

    model = model.to(device)

    loss_function = nn.MSELoss()

    history = {
        "epoch": [],
        "train_mse": [],
        "validation_mse": [],
        "validation_rmse": [],
        "validation_mae": [],
        "validation_concordance_index": [],
        "validation_pearson_correlation": [],
    }

    best_validation_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_mse = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
        )

        validation_metrics = evaluate_model(
            model=model,
            data_loader=validation_loader,
            device=device,
        )

        history["epoch"].append(epoch)
        history["train_mse"].append(train_mse)
        history["validation_mse"].append(
            validation_metrics["mse"]
        )
        history["validation_rmse"].append(
            validation_metrics["rmse"]
        )
        history["validation_mae"].append(
            validation_metrics["mae"]
        )
        history["validation_concordance_index"].append(
            validation_metrics["concordance_index"]
        )
        history["validation_pearson_correlation"].append(
            validation_metrics["pearson_correlation"]
        )

        if verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"Train MSE: {train_mse:.4f} | "
                f"Validation MSE: "
                f"{validation_metrics['mse']:.4f} | "
                f"RMSE: {validation_metrics['rmse']:.4f} | "
                f"MAE: {validation_metrics['mae']:.4f}"
            )

        improvement = (
            best_validation_mse
            - validation_metrics["mse"]
        )

        if improvement > minimum_improvement:
            best_validation_mse = validation_metrics["mse"]
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_metrics=validation_metrics,
                checkpoint_path=checkpoint_path,
                metadata=metadata,
            )

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            if verbose:
                print(
                    "Early stopping triggered at "
                    f"epoch {epoch}."
                )
            break

    checkpoint = load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_metrics": checkpoint[
            "validation_metrics"
        ],
        "checkpoint_path": str(checkpoint_path),
    }


def append_experiment_log(
    *,
    dataset: str,
    model_name: str,
    protein_encoder: str,
    split_type: str,
    seed: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    best_validation_mse: float,
    test_metrics: dict[str, float],
    model_path: str | Path,
    model: nn.Module | None = None,
    log_path: str | Path = EXPERIMENT_LOG_PATH,
) -> None:
    """Append one completed experiment to experiment_log.csv."""

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = (
        f"{dataset}_{model_name}_{split_type}_"
        f"seed{seed}_{get_timestamp()}"
    )

    row = {
        "run_id": run_id,
        "timestamp": get_timestamp(),
        "dataset": dataset,
        "model": model_name,
        "protein_encoder": protein_encoder,
        "split_type": split_type,
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "best_validation_mse": best_validation_mse,
        "test_mse": test_metrics["mse"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "concordance_index": test_metrics[
            "concordance_index"
        ],
        "pearson_correlation": test_metrics[
            "pearson_correlation"
        ],
        "model_path": str(model_path),
    }

    if model is not None:
        row["trainable_parameters"] = (
            count_trainable_parameters(model)
        )

    existing_header = []

    if log_path.exists() and log_path.stat().st_size > 0:
        with log_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.reader(file)
            existing_header = next(reader, [])

    fieldnames = [
        "run_id",
        "timestamp",
        "dataset",
        "model",
        "protein_encoder",
        "split_type",
        "seed",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "best_validation_mse",
        "test_mse",
        "test_rmse",
        "test_mae",
        "concordance_index",
        "pearson_correlation",
        "model_path",
    ]

    if existing_header:
        fieldnames = existing_header

    file_exists = (
        log_path.exists()
        and log_path.stat().st_size > 0
    )

    with log_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
