"""Training functions for drug-target affinity models."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import (
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_IMPROVEMENT,
    WEIGHT_DECAY,
)
from .metrics import calculate_metrics



def _forward_batch(
    model: nn.Module,
    batch,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Move one batch to the selected device and run the model.

    Baseline batches are PyTorch Geometric Data objects.

    Advanced batches are dictionaries containing:
        graph
        protein_embeddings
        protein_mask
        y
    """

    if isinstance(batch, dict):
        required_keys = {
            "graph",
            "protein_embeddings",
            "protein_mask",
            "y",
        }

        missing_keys = required_keys.difference(
            batch.keys()
        )

        if missing_keys:
            raise KeyError(
                "Advanced batch is missing required keys: "
                f"{sorted(missing_keys)}"
            )

        graph = batch["graph"].to(device)

        protein_embeddings = batch[
            "protein_embeddings"
        ].to(device)

        protein_mask = batch[
            "protein_mask"
        ].to(device)

        targets = batch["y"].to(
            device=device,
            dtype=torch.float,
        ).view(-1)

        predictions = model(
            graph,
            protein_embeddings,
            protein_mask,
        )

    else:
        batch = batch.to(device)

        predictions = model(batch)

        targets = batch.y.view(-1).float()

    predictions = predictions.view(-1)

    if predictions.shape != targets.shape:
        raise ValueError(
            "Predictions and targets must have matching shapes. "
            f"Found predictions {tuple(predictions.shape)} and "
            f"targets {tuple(targets.shape)}."
        )

    return predictions, targets

def train_one_epoch(
    model: nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """Train the model for one epoch and return the mean loss."""

    model.train()

    total_loss = 0.0
    total_examples = 0

    for batch in data_loader:
        optimizer.zero_grad()

        predictions, targets = _forward_batch(
            model=model,
            batch=batch,
            device=device,
        )

        loss = loss_function(predictions, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Training data loader is empty.")

    return total_loss / total_examples


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    data_loader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the mean loss on a validation or test data loader."""

    model.eval()

    total_loss = 0.0
    total_examples = 0

    for batch in data_loader:
        predictions, targets = _forward_batch(
            model=model,
            batch=batch,
            device=device,
        )

        loss = loss_function(predictions, targets)

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Evaluation data loader is empty.")

    return total_loss / total_examples


@torch.no_grad()
def predict(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect true and predicted affinity values from a data loader."""

    model.eval()

    true_values = []
    predicted_values = []

    for batch in data_loader:
        predictions, targets = _forward_batch(
            model=model,
            batch=batch,
            device=device,
        )

        true_values.append(
            targets.detach().cpu().numpy()
        )
        predicted_values.append(
            predictions.detach().cpu().numpy()
        )

    if not true_values:
        raise ValueError("Prediction data loader is empty.")

    y_true = np.concatenate(true_values)
    y_pred = np.concatenate(predicted_values)

    return y_true, y_pred


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    checkpoint_path: Path | str,
) -> None:
    """Save a model checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path | str,
    device: torch.device,
) -> dict:
    """Load a saved model checkpoint."""

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

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    return checkpoint


def fit_model(
    model: nn.Module,
    train_loader,
    validation_loader,
    device: torch.device,
    checkpoint_path: Path | str,
    max_epochs: int = MAX_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = EARLY_STOPPING_PATIENCE,
    min_improvement: float = MIN_IMPROVEMENT,
    loss_function: nn.Module | None = None,
    verbose: bool = True,
) -> dict:
    """
    Train a model with early stopping.

    The checkpoint with the lowest validation loss is kept on disk.
    The returned dictionary uses the same key names for every model.
    """

    if max_epochs <= 0:
        raise ValueError("max_epochs must be greater than zero.")

    if patience <= 0:
        raise ValueError("patience must be greater than zero.")

    model = model.to(device)

    if loss_function is None:
        loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = []

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
        )

        validation_loss = evaluate_loss(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function,
            device=device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_loss": float(
                    validation_loss
                ),
            }
        )

        improvement = (
            best_validation_loss
            - validation_loss
        )

        if improvement > min_improvement:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                checkpoint_path=checkpoint_path,
            )

        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"Train MSE: {train_loss:.4f} | "
                f"Validation MSE: {validation_loss:.4f}"
            )

        if epochs_without_improvement >= patience:
            if verbose:
                print(
                    "Early stopping at epoch "
                    f"{epoch}. Best epoch: {best_epoch}"
                )
            break

    if best_epoch == 0:
        # This is unlikely, but ensures a checkpoint exists even if the
        # requested minimum improvement is unusually large.
        final_validation_loss = history[-1][
            "validation_loss"
        ]

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=len(history),
            validation_loss=final_validation_loss,
            checkpoint_path=checkpoint_path,
        )

        best_epoch = len(history)
        best_validation_loss = final_validation_loss

    load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    return {
        "best_epoch": int(best_epoch),
        "best_validation_mse": float(
            best_validation_loss
        ),
        "completed_epochs": len(history),
        "history": history,
        "checkpoint_path": str(
            Path(checkpoint_path)
        ),
    }


def evaluate_model(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> dict:
    """Evaluate a trained model using all project metrics."""

    y_true, y_pred = predict(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }
