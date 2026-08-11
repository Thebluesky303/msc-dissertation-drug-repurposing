"""Training, evaluation, experiment orchestration, and model selection.

The low-level training functions retain the verified v1 regression behaviour.
The v2 runners add reproducible data loading, validation-only checkpoint
selection, final test evaluation, and machine-readable experiment artifacts.
Importing this module never starts a training job.
"""

from __future__ import annotations

import csv
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from config import (
    ADVANCED_GINE_LAYERS,
    ADVANCED_HIDDEN_DIM,
    ADVANCED_PROTEIN_CNN_LAYERS,
    BOND_FEATURE_DIM,
    CROSS_ATTENTION_FFN_DIM,
    CROSS_ATTENTION_HEADS,
    DAVIS_RAW_DIR,
    DROPOUT,
    DRUG_HIDDEN_DIM,
    EARLY_STOPPING_PATIENCE,
    ESM2_EMBEDDING_DIM,
    ESM2_MODEL_NAME,
    EXPERIMENT_LOG_PATH,
    FUSION_HIDDEN_DIM,
    KIBA_RAW_DIR,
    METRICS_DIR,
    MINIMUM_IMPROVEMENT,
    MODELS_DIR,
    NODE_FEATURE_DIM,
    NUM_EPOCHS,
    PREDICTIONS_DIR,
    PROTEIN_EMBEDDING_DIM,
    PROTEIN_HIDDEN_DIM,
    TEST_RATIO,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    ExperimentConfig,
    processed_pairs_path,
)
from src.dataset import AdvancedDTADataset, DTADataset
from src.metrics import calculate_regression_metrics
from src.models import create_advanced_dta_model, create_dta_model
from src.preprocessing import (
    build_drug_graph_cache,
    build_protein_tensor_cache,
    ensure_pair_ids,
    pair_id_sha256,
    prepare_dataset,
    scientific_pairs_sha256,
)
from src.splits import create_split, validate_split_integrity
from src.utils import (
    count_trainable_parameters,
    get_device,
    get_timestamp,
    seed_data_loader_worker,
    set_random_seed,
)


BASELINE_MODEL_NAMES = ("gcn", "gat", "gin")
EXPERIMENT_CSV_FIELDS = (
    "run_id",
    "timestamp",
    "dataset",
    "affinity_label",
    "split",
    "split_type",
    "model",
    "protein_encoder",
    "seed",
    "batch_size",
    "epochs",
    "epochs_trained",
    "learning_rate",
    "weight_decay",
    "best_epoch",
    "best_validation_mse",
    "test_mse",
    "test_rmse",
    "test_mae",
    "concordance_index",
    "pearson_correlation",
    "trainable_parameters",
    "trainable_parameter_count",
    "training_runtime_seconds",
    "checkpoint_path",
    "predictions_path",
    "record_path",
    "split_fingerprint",
    "split_manifest_path",
    "split_manifest_sha256",
    "data_fingerprint",
    "data_provenance",
    "runtime_provenance",
    "allow_unverified_residue_cache",
    "split_counts",
    "configuration",
    "configuration_used",
)


def _package_version(*distribution_names: str) -> str | None:
    for name in distribution_names:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_provenance() -> dict[str, Any]:
    """Capture local software and Git state without network access."""

    repository = Path(__file__).resolve().parents[1]

    def git_value(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() or None

    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        dirty: bool | None = bool(status_result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "package_versions": {
            "torch": torch.__version__,
            "torch_geometric": _package_version("torch-geometric"),
            "rdkit": _package_version("rdkit", "rdkit-pypi"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": _package_version("scikit-learn"),
            "transformers": _package_version("transformers"),
            "huggingface_hub": _package_version("huggingface-hub"),
        },
        "git": {
            "head": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "dirty": dirty,
        },
    }


def train_one_epoch(
    model: nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """Train the model for one epoch and return sample-weighted mean loss."""

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
    """Return true values, predictions, and stable loader sample indices."""

    model.eval()
    all_true_values: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_sample_indices: list[np.ndarray] = []

    for batch in data_loader:
        batch = batch.to(device)
        predictions = model(batch)
        all_predictions.append(predictions.detach().cpu().numpy().reshape(-1))

        if hasattr(batch, "y"):
            all_true_values.append(batch.y.detach().cpu().numpy().reshape(-1))
        if hasattr(batch, "sample_id"):
            all_sample_indices.append(
                batch.sample_id.detach().cpu().numpy().reshape(-1)
            )

    if not all_predictions:
        raise ValueError("Prediction DataLoader contains no samples.")

    predicted_values = np.concatenate(all_predictions)
    true_values = (
        np.concatenate(all_true_values)
        if all_true_values
        else np.array([], dtype=float)
    )
    sample_indices = (
        np.concatenate(all_sample_indices)
        if all_sample_indices
        else np.arange(len(predicted_values))
    )
    return true_values, predicted_values, sample_indices


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a model using the unchanged project regression metrics."""

    true_values, predicted_values, _ = predict(
        model=model,
        data_loader=data_loader,
        device=device,
    )
    if true_values.size == 0:
        raise ValueError("Evaluation requires affinity labels.")
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
    """Save model weights, optimizer state, and validation information."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_metrics": validation_metrics,
            "metadata": metadata or {},
        },
        checkpoint_path,
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load saved weights into an existing model instance."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
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
    """Train with early stopping and retain lowest-validation-MSE weights."""

    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1.")
    if patience < 1:
        raise ValueError("patience must be at least 1.")

    model = model.to(device)
    loss_function = nn.MSELoss()
    history: dict[str, list[float | int]] = {
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
        history["validation_mse"].append(validation_metrics["mse"])
        history["validation_rmse"].append(validation_metrics["rmse"])
        history["validation_mae"].append(validation_metrics["mae"])
        history["validation_concordance_index"].append(
            validation_metrics["concordance_index"]
        )
        history["validation_pearson_correlation"].append(
            validation_metrics["pearson_correlation"]
        )

        if verbose:
            print(
                f"Epoch {epoch:03d} | Train MSE: {train_mse:.4f} | "
                f"Validation MSE: {validation_metrics['mse']:.4f} | "
                f"RMSE: {validation_metrics['rmse']:.4f} | "
                f"MAE: {validation_metrics['mae']:.4f}"
            )

        improvement = best_validation_mse - validation_metrics["mse"]
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
                print(f"Early stopping triggered at epoch {epoch}.")
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
        "best_validation_metrics": checkpoint["validation_metrics"],
        "checkpoint_path": str(checkpoint_path),
    }


def _json_safe(value: Any) -> Any:
    """Convert paths, NumPy values, and non-finite floats for strict JSON."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _csv_value(value: Any) -> Any:
    """Serialize nested record values into one deterministic CSV cell."""

    safe_value = _json_safe(value)
    if isinstance(safe_value, (dict, list)):
        return json.dumps(safe_value, sort_keys=True, separators=(",", ":"))
    return safe_value


def _append_record_csv(record: Mapping[str, Any], log_path: str | Path) -> None:
    """Append one record, extending a legacy header without losing old rows."""

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {field: _csv_value(record.get(field)) for field in EXPERIMENT_CSV_FIELDS}

    existing_rows: list[dict[str, str]] = []
    existing_header: list[str] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            existing_header = list(reader.fieldnames or [])
            existing_rows = list(reader)

    fieldnames = list(existing_header)
    fieldnames.extend(
        field for field in EXPERIMENT_CSV_FIELDS if field not in fieldnames
    )
    if not fieldnames:
        fieldnames = list(EXPERIMENT_CSV_FIELDS)

    if existing_header != fieldnames and existing_header:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
        temporary_path.replace(path)

    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


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
    best_epoch: int | None = None,
    training_runtime_seconds: float | None = None,
    configuration_used: Mapping[str, Any] | None = None,
    affinity_label: str | None = None,
    predictions_path: str | Path | None = None,
    record_path: str | Path | None = None,
) -> None:
    """Append a completed experiment while preserving the v1 call signature."""

    timestamp = get_timestamp()
    parameter_count = (
        count_trainable_parameters(model) if model is not None else None
    )
    record = {
        "run_id": f"{dataset}_{model_name}_{split_type}_seed{seed}_{timestamp}",
        "timestamp": timestamp,
        "dataset": dataset,
        "affinity_label": affinity_label,
        "split": split_type,
        "split_type": split_type,
        "model": model_name,
        "protein_encoder": protein_encoder,
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "epochs_trained": best_epoch,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation_mse,
        "test_mse": test_metrics["mse"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "concordance_index": test_metrics["concordance_index"],
        "pearson_correlation": test_metrics["pearson_correlation"],
        "trainable_parameters": parameter_count,
        "trainable_parameter_count": parameter_count,
        "training_runtime_seconds": training_runtime_seconds,
        "checkpoint_path": str(model_path),
        "predictions_path": predictions_path,
        "record_path": record_path,
        "configuration": configuration_used or {},
        "configuration_used": configuration_used or {},
    }
    _append_record_csv(record, log_path)


def _expected_affinity_metadata(dataset: str) -> dict[str, str]:
    if dataset == "davis":
        return {
            "affinity_type": "pKd",
            "affinity_target": "pKd",
            "affinity_transformation": "9 - log10(Kd_nM)",
            "affinity_semantics": (
                "Negative base-10 logarithm of molar dissociation constant, "
                "computed from GraphDTA Kd in nM as 9-log10(Kd_nM)."
            ),
        }
    return {
        "affinity_type": "KIBA_score",
        "affinity_target": "KIBA score",
        "affinity_transformation": (
            "GraphDTA KIBA score retained without pKd conversion"
        ),
        "affinity_semantics": (
            "GraphDTA KIBA integrated activity score retained on its native "
            "scale; it is not a pKd value."
        ),
    }


def _validate_pair_semantics(
    pairs: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    """Normalize and strictly validate one benchmark's affinity semantics."""

    result = pairs.copy()
    expected = _expected_affinity_metadata(dataset)["affinity_type"]
    if "affinity_type" not in result.columns:
        raise ValueError(
            "Experiment pair data must explicitly declare affinity_type; "
            "numeric labels cannot be safely inferred or relabelled."
        )
    observed = set(result["affinity_type"].astype(str).str.strip())
    if observed != {expected}:
        raise ValueError(
            f"{dataset} affinity_type must be exactly {expected!r}; "
            f"observed {sorted(observed)}."
        )
    return result


def _read_processed_pairs_with_provenance(
    input_path: Path,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a processed CSV only when its strict preparation sidecar matches."""

    metadata_path = input_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Processed benchmark CSV requires its provenance sidecar: "
            f"{metadata_path}. Re-run prepare_dataset."
        )
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, Mapping):
        raise ValueError("Processed-data metadata must be a JSON object.")
    required = {
        "preprocessing_schema_version",
        "dataset",
        "affinity_type",
        "affinity_target",
        "affinity_transformation",
        "affinity_semantics",
        "raw_source_files",
        "processed_csv_file",
        "processed_csv_bytes",
        "processed_csv_sha256",
        "processed_csv_rows",
        "pair_id_sha256",
        "scientific_data_sha256",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(
            f"Processed-data metadata is missing fields: {sorted(missing)}"
        )
    if int(metadata["preprocessing_schema_version"]) != 1:
        raise ValueError("Unsupported processed-data metadata schema version.")
    if str(metadata["processed_csv_file"]) != input_path.name:
        raise ValueError("Processed-data metadata names a different CSV file.")
    if int(metadata["processed_csv_bytes"]) != input_path.stat().st_size:
        raise ValueError("Processed CSV byte size does not match its metadata.")
    digest = _sha256_path(input_path)
    if str(metadata["processed_csv_sha256"]).lower() != digest:
        raise ValueError("Processed CSV SHA-256 does not match its metadata.")
    if str(metadata["dataset"]).strip().lower() != config.dataset:
        raise ValueError("Processed-data metadata dataset does not match config.")
    expected = _expected_affinity_metadata(config.dataset)
    for field, value in expected.items():
        if str(metadata[field]) != value:
            raise ValueError(
                f"Processed-data metadata {field} does not match "
                f"{config.dataset} benchmark semantics."
            )
    source_files = metadata["raw_source_files"]
    if not isinstance(source_files, Mapping) or set(source_files) != {
        "ligands_can.txt",
        "proteins.txt",
        "Y",
    }:
        raise ValueError("Processed-data metadata lacks the three raw sources.")
    for source_name, source in source_files.items():
        if not isinstance(source, Mapping) or not {
            "bytes",
            "sha256",
        }.issubset(source):
            raise ValueError(f"Raw-source provenance is incomplete for {source_name}.")
        if len(str(source["sha256"])) != 64 or int(source["bytes"]) < 1:
            raise ValueError(f"Raw-source provenance is invalid for {source_name}.")

    pairs = pd.read_csv(
        input_path,
        dtype={
            "dataset": str,
            "pair_id": str,
            "drug_id": str,
            "target_id": str,
            "smiles": str,
            "protein_sequence": str,
            "affinity_type": str,
        },
    )
    pairs = _validate_pair_semantics(pairs, config.dataset)
    pairs = ensure_pair_ids(pairs)
    if int(metadata["processed_csv_rows"]) != len(pairs):
        raise ValueError("Processed CSV row count does not match its metadata.")
    if str(metadata["pair_id_sha256"]) != pair_id_sha256(pairs):
        raise ValueError("Processed pair-ID fingerprint does not match metadata.")
    data_fingerprint = scientific_pairs_sha256(pairs)
    if str(metadata["scientific_data_sha256"]) != data_fingerprint:
        raise ValueError("Processed scientific-data fingerprint does not match metadata.")
    return pairs, {
        "mode": "verified_processed_csv",
        "data_sha256": data_fingerprint,
        "path": str(input_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "csv_sha256": digest,
        "metadata": _json_safe(metadata),
    }


def _load_experiment_pairs(
    config: ExperimentConfig,
    pairs_dataframe: pd.DataFrame | None,
) -> pd.DataFrame:
    """Load one benchmark only and establish deterministic pair identifiers."""

    if pairs_dataframe is None:
        input_path = processed_pairs_path(config.dataset)
        if not input_path.exists():
            raw_directory = {
                "davis": DAVIS_RAW_DIR,
                "kiba": KIBA_RAW_DIR,
            }[config.dataset]
            if not raw_directory.exists():
                raise FileNotFoundError(
                    f"Processed {config.dataset} pair data was not found at "
                    f"{input_path}, and its raw directory was not found at "
                    f"{raw_directory}. Stage the GraphDTA raw files, run data "
                    "preparation, or pass pairs_dataframe."
                )
            prepare_dataset(
                dataset_name=config.dataset,
                raw_data_directory=raw_directory,
                output_path=input_path,
            )
        pairs, data_provenance = _read_processed_pairs_with_provenance(
            input_path, config
        )
    else:
        pairs = pairs_dataframe.copy()
        data_provenance = {"mode": "in_memory_dataframe"}

    if pairs.empty:
        raise ValueError("Experiment pair data cannot be empty.")
    if "dataset" not in pairs.columns:
        pairs.insert(0, "dataset", config.dataset)
    observed_datasets = set(pairs["dataset"].astype(str).str.strip().str.lower())
    if observed_datasets != {config.dataset}:
        raise ValueError(
            "Experiment data must contain exactly one configured dataset; "
            f"observed {sorted(observed_datasets)}."
        )
    pairs["dataset"] = config.dataset
    pairs = ensure_pair_ids(pairs)
    pairs = _validate_pair_semantics(pairs, config.dataset)

    required = {"drug_id", "target_id", "smiles", "protein_sequence", "affinity"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"Experiment pair data is missing columns: {sorted(missing)}")
    affinities = pd.to_numeric(pairs["affinity"], errors="coerce").to_numpy()
    if not np.isfinite(affinities).all():
        raise ValueError("Experiment affinity values must be finite and numeric.")
    pairs["affinity"] = affinities.astype(float)
    pairs = pairs.reset_index(drop=True)
    data_fingerprint = scientific_pairs_sha256(pairs)
    data_provenance.update(
        {
            "data_sha256": data_fingerprint,
            "pair_id_sha256": pair_id_sha256(pairs),
            "rows": int(len(pairs)),
            "scientific_columns": [
                column
                for column in (
                    "dataset",
                    "pair_id",
                    "drug_id",
                    "target_id",
                    "smiles",
                    "protein_sequence",
                    "raw_affinity",
                    "affinity",
                    "affinity_type",
                )
                if column in pairs.columns
            ],
            "affinity_semantics": _expected_affinity_metadata(config.dataset),
        }
    )
    pairs.attrs["data_provenance"] = data_provenance
    return pairs


def _resolve_splits(
    pairs: pd.DataFrame,
    config: ExperimentConfig,
    split_manifest_path: str | Path | None,
) -> tuple[
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    dict[str, Any],
]:
    """Load an explicit manifest or reconstruct and optionally persist one."""

    manifest_path = Path(split_manifest_path) if split_manifest_path else None
    if manifest_path is not None and manifest_path.exists():
        from src.splits import apply_split_manifest, load_split_manifest

        provenance = _validate_manifest_metadata(manifest_path, config, pairs)
        splits = apply_split_manifest(pairs, load_split_manifest(manifest_path))
    else:
        splits = create_split(
            dataframe=pairs,
            split_type=config.split,
            random_seed=config.seed,
        )
        if manifest_path is not None:
            from src.splits import build_split_manifest, save_split_manifest

            manifest = build_split_manifest(
                train_dataframe=splits[0],
                validation_dataframe=splits[1],
                test_dataframe=splits[2],
                dataset_name=config.dataset,
            )
            save_split_manifest(
                manifest,
                manifest_path,
                metadata={
                    "dataset": config.dataset,
                    "split_type": config.split,
                    "seed": config.seed,
                    "ratios": {
                        "train": TRAIN_RATIO,
                        "validation": VALIDATION_RATIO,
                        "test": TEST_RATIO,
                    },
                    "source_rows": len(pairs),
                    "source_pair_id_sha256": pair_id_sha256(pairs),
                    "source_data_sha256": scientific_pairs_sha256(pairs),
                },
            )
            provenance = _validate_manifest_metadata(manifest_path, config, pairs)
        else:
            provenance = {
                "mode": "deterministically_reconstructed",
                "path": None,
                "metadata_path": None,
                "sha256": None,
            }

    validate_split_integrity(*splits, split_type=config.split)
    return splits, provenance


def _validate_manifest_metadata(
    manifest_path: Path,
    config: ExperimentConfig,
    pairs: pd.DataFrame,
) -> dict[str, Any]:
    """Fail closed if explicit-manifest provenance disagrees with the run."""

    metadata_path = manifest_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            "An explicit split manifest requires its provenance sidecar: "
            f"{metadata_path}. Re-materialize the manifest for this seed."
        )
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, Mapping):
        raise ValueError("Split-manifest metadata must be a JSON object.")

    required = {
        "dataset",
        "split_type",
        "seed",
        "ratios",
        "manifest_file",
        "manifest_sha256",
        "schema_version",
        "source_rows",
        "source_pair_id_sha256",
        "source_data_sha256",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(
            "Split-manifest metadata is missing provenance fields: "
            f"{sorted(missing)}."
        )
    observed_dataset = str(metadata["dataset"]).strip().lower()
    observed_split = str(metadata["split_type"]).strip().lower().replace("-", "_")
    try:
        observed_seed = int(metadata["seed"])
    except (TypeError, ValueError) as error:
        raise ValueError("Split-manifest seed must be an integer.") from error
    if observed_dataset != config.dataset:
        raise ValueError(
            f"Split-manifest dataset {observed_dataset!r} does not match "
            f"configured dataset {config.dataset!r}."
        )
    if int(metadata["schema_version"]) != 1:
        raise ValueError("Unsupported split-manifest metadata schema version.")
    if int(metadata["source_rows"]) != len(pairs):
        raise ValueError("Split-manifest source row count does not match data.")
    if str(metadata["source_pair_id_sha256"]) != pair_id_sha256(pairs):
        raise ValueError("Split-manifest source pair-ID fingerprint differs.")
    if str(metadata["source_data_sha256"]) != scientific_pairs_sha256(pairs):
        raise ValueError("Split-manifest source scientific-data fingerprint differs.")
    if observed_split != config.split:
        raise ValueError(
            f"Split-manifest type {observed_split!r} does not match "
            f"configured split {config.split!r}."
        )
    if observed_seed != config.seed:
        raise ValueError(
            f"Split-manifest seed {observed_seed} does not match configured "
            f"seed {config.seed}."
        )
    if str(metadata["manifest_file"]) != manifest_path.name:
        raise ValueError("Split-manifest metadata names a different CSV file.")

    ratios = metadata["ratios"]
    if not isinstance(ratios, Mapping):
        raise ValueError("Split-manifest ratios must be a JSON object.")
    expected_ratios = {
        "train": TRAIN_RATIO,
        "validation": VALIDATION_RATIO,
        "test": TEST_RATIO,
    }
    for partition, expected in expected_ratios.items():
        try:
            observed = float(ratios[partition])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Split-manifest ratios are missing numeric {partition!r}."
            ) from error
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Split-manifest {partition} ratio {observed} does not match "
                f"configured ratio {expected}."
            )

    digest = _sha256_path(manifest_path)
    if str(metadata["manifest_sha256"]).lower() != digest:
        raise ValueError("Split-manifest SHA-256 does not match its metadata.")
    return {
        "mode": "explicit_manifest",
        "path": str(manifest_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "sha256": digest,
        "metadata": _json_safe(metadata),
    }


def _split_fingerprint(
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    data_fingerprint: str,
) -> str:
    """Hash ordered pair membership so an experiment can cite its exact split."""

    payload = {
        "source_data_sha256": data_fingerprint,
        "partitions": {
            partition: dataframe["pair_id"].astype(str).tolist()
            for partition, dataframe in zip(
                ("train", "validation", "test"), splits
            )
        },
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _make_loader(
    dataset,
    config: ExperimentConfig,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Create one portable PyG loader with deterministic worker seeds."""

    if len(dataset) == 0:
        raise ValueError("Train, validation, and test datasets must be non-empty.")
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        generator=generator,
        worker_init_fn=seed_data_loader_worker,
    )


def _artifact_paths(
    config: ExperimentConfig,
    checkpoint_path: str | Path | None,
    record_path: str | Path | None,
    predictions_path: str | Path | None,
) -> tuple[Path, Path, Path]:
    """Resolve default paths without creating files before a run starts."""

    return (
        Path(checkpoint_path or MODELS_DIR / f"{config.run_name}.pt"),
        Path(record_path or METRICS_DIR / f"{config.run_name}.json"),
        Path(predictions_path or PREDICTIONS_DIR / f"{config.run_name}.csv"),
    )


def _save_predictions(
    test_dataframe: pd.DataFrame,
    sample_indices: np.ndarray,
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    affinity_label: str,
    output_path: Path,
) -> None:
    """Save predictions joined back to stable pair IDs and source identifiers."""

    indices = np.asarray(sample_indices, dtype=int).reshape(-1)
    if len(indices) != len(predicted_values) or len(indices) != len(true_values):
        raise ValueError("Prediction arrays and sample IDs must have equal length.")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("Test prediction sample IDs must be unique.")
    if np.any(indices < 0) or np.any(indices >= len(test_dataframe)):
        raise IndexError("A prediction sample ID is outside the test dataframe.")

    predictions = test_dataframe.iloc[indices].reset_index(drop=True).copy()
    predictions["observed_affinity"] = true_values
    predictions["predicted_affinity"] = predicted_values
    predictions["affinity_label"] = affinity_label
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)


def _write_record_json(record: Mapping[str, Any], output_path: Path) -> None:
    """Write one standards-compliant JSON experiment record."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(record), file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def _configuration_used(
    config: ExperimentConfig,
    *,
    model_family: str,
    custom_model_supplied: bool,
    split_manifest_provenance: Mapping[str, Any],
    data_provenance: Mapping[str, Any] | None = None,
    runtime_provenance: Mapping[str, Any] | None = None,
    residue_embedding_dim: int | None = None,
    residue_cache_metadata: Mapping[str, Any] | None = None,
    allow_unverified_residue_cache: bool = False,
) -> dict[str, Any]:
    """Capture experiment and architecture constants that affect a run."""

    values = config.to_dict()
    values.update(
        {
            "model_family": model_family,
            "custom_model_supplied": custom_model_supplied,
            "split_ratios": {
                "train": TRAIN_RATIO,
                "validation": VALIDATION_RATIO,
                "test": TEST_RATIO,
            },
            "atom_feature_dim": NODE_FEATURE_DIM,
            "bond_feature_dim": BOND_FEATURE_DIM,
            "dropout": DROPOUT,
            "optimizer": "torch.optim.Adam",
            "loss_function": "torch.nn.MSELoss",
            "checkpoint_selection_metric": "validation_mse",
            "test_evaluation_policy": "after_best_validation_checkpoint",
            "split_manifest": dict(split_manifest_provenance),
            "data_provenance": dict(data_provenance or {}),
            "runtime_provenance": dict(
                runtime_provenance or _runtime_provenance()
            ),
        }
    )
    if model_family == "baseline":
        values["architecture"] = {
            "drug_hidden_dim": DRUG_HIDDEN_DIM,
            "protein_embedding_dim": PROTEIN_EMBEDDING_DIM,
            "protein_hidden_dim": PROTEIN_HIDDEN_DIM,
            "fusion_hidden_dim": FUSION_HIDDEN_DIM,
            "protein_encoder": "v1 token embedding + 2-layer 1D CNN",
        }
    elif model_family == "advanced":
        cache_metadata = dict(residue_cache_metadata or {})
        verified_model_name = (
            cache_metadata.get("model_name")
            if cache_metadata.get("provenance_verified") is True
            else None
        )
        values["architecture"] = {
            "hidden_dim": ADVANCED_HIDDEN_DIM,
            "gine_layers": ADVANCED_GINE_LAYERS,
            "protein_cnn_layers": ADVANCED_PROTEIN_CNN_LAYERS,
            "attention_heads": CROSS_ATTENTION_HEADS,
            "attention_feedforward_dim": CROSS_ATTENTION_FFN_DIM,
            "regression_hidden_dim": 512,
            "regression_bottleneck_dim": 256,
            "esm2_model_name": verified_model_name,
            "configured_default_esm2_model_name": ESM2_MODEL_NAME,
            "configured_esm2_embedding_dim": ESM2_EMBEDDING_DIM,
            "residue_embedding_dim_used": residue_embedding_dim,
            "esm2_execution_during_training": False,
            "residue_cache_provenance": cache_metadata,
            "allow_unverified_residue_cache": bool(
                allow_unverified_residue_cache
            ),
        }
    else:  # pragma: no cover - internal contract
        raise ValueError("model_family must be baseline or advanced.")
    return values


def _execute_experiment(
    *,
    config: ExperimentConfig,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    test_dataframe: pd.DataFrame,
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    protein_encoder: str,
    configuration_used: Mapping[str, Any],
    split_manifest_provenance: Mapping[str, Any],
    selected_device: torch.device,
    checkpoint_path: Path,
    record_path: Path,
    predictions_path: Path,
    log_path: str | Path,
    verbose: bool,
) -> dict[str, Any]:
    """Fit on train/validation, then evaluate the reloaded best model on test."""

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    data_provenance = dict(configuration_used.get("data_provenance", {}))
    data_fingerprint = str(data_provenance.get("data_sha256", ""))
    if len(data_fingerprint) != 64:
        raise ValueError("Experiment configuration lacks a valid data fingerprint.")
    split_fingerprint = _split_fingerprint(splits, data_fingerprint)
    metadata = {
        "configuration_used": dict(configuration_used),
        "affinity_label": config.affinity_label,
        "data_fingerprint": data_fingerprint,
        "data_provenance": data_provenance,
        "runtime_provenance": dict(
            configuration_used.get("runtime_provenance", {})
        ),
        "split_fingerprint": split_fingerprint,
        "split_manifest": dict(split_manifest_provenance),
    }
    start_time = time.perf_counter()
    fit_result = fit_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        device=selected_device,
        checkpoint_path=checkpoint_path,
        num_epochs=config.epochs,
        patience=config.early_stopping_patience,
        minimum_improvement=config.minimum_improvement,
        metadata=metadata,
        verbose=verbose,
    )
    training_runtime_seconds = time.perf_counter() - start_time

    # This is deliberately the first access to test labels/predictions. The
    # model now contains the checkpoint chosen solely by validation MSE.
    true_values, predicted_values, sample_indices = predict(
        model=fit_result["model"],
        data_loader=test_loader,
        device=selected_device,
    )
    test_metrics = calculate_regression_metrics(true_values, predicted_values)

    stored_predictions_path: str | None = None
    if config.save_predictions:
        _save_predictions(
            test_dataframe=test_dataframe,
            sample_indices=sample_indices,
            true_values=true_values,
            predicted_values=predicted_values,
            affinity_label=config.affinity_label,
            output_path=predictions_path,
        )
        stored_predictions_path = str(predictions_path.resolve())

    timestamp = get_timestamp()
    parameter_count = count_trainable_parameters(fit_result["model"])
    history = fit_result["history"]
    record: dict[str, Any] = {
        "run_id": f"{config.run_name}_{timestamp}",
        "timestamp": timestamp,
        "dataset": config.dataset,
        "affinity_label": config.affinity_label,
        "split": config.split,
        "split_type": config.split,
        "model": config.model,
        "protein_encoder": protein_encoder,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "epochs_trained": len(history["epoch"]),
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "best_epoch": fit_result["best_epoch"],
        "best_validation_mse": fit_result["best_validation_metrics"]["mse"],
        "test_mse": test_metrics["mse"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "concordance_index": test_metrics["concordance_index"],
        "pearson_correlation": test_metrics["pearson_correlation"],
        "trainable_parameters": parameter_count,
        "trainable_parameter_count": parameter_count,
        "training_runtime_seconds": training_runtime_seconds,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "predictions_path": stored_predictions_path,
        "record_path": str(record_path.resolve()),
        "split_fingerprint": metadata["split_fingerprint"],
        "split_manifest_path": split_manifest_provenance.get("path"),
        "split_manifest_sha256": split_manifest_provenance.get("sha256"),
        "split_manifest_provenance": dict(split_manifest_provenance),
        "data_fingerprint": data_fingerprint,
        "data_provenance": data_provenance,
        "runtime_provenance": dict(
            configuration_used.get("runtime_provenance", {})
        ),
        "allow_unverified_residue_cache": bool(
            configuration_used.get("architecture", {}).get(
                "allow_unverified_residue_cache", False
            )
        ),
        "split_counts": {
            "train": len(splits[0]),
            "validation": len(splits[1]),
            "test": len(splits[2]),
        },
        "configuration": dict(configuration_used),
        "configuration_used": dict(configuration_used),
        "validation_metrics": fit_result["best_validation_metrics"],
        "test_metrics": test_metrics,
        "history": history,
    }
    _write_record_json(record, record_path)
    _append_record_csv(record, log_path)
    return _json_safe(record)


def _validate_config(config: ExperimentConfig, allowed_models: set[str]) -> None:
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig instance.")
    if config.model not in allowed_models:
        allowed = ", ".join(sorted(allowed_models))
        raise ValueError(f"This runner requires config.model to be one of: {allowed}.")


def run_baseline_experiment(
    config: ExperimentConfig,
    pairs_dataframe: pd.DataFrame | None = None,
    *,
    drug_graph_cache: Mapping[str, Any] | None = None,
    protein_tensor_cache: Mapping[str, torch.Tensor] | None = None,
    model: nn.Module | None = None,
    device: str | torch.device | None = None,
    split_manifest_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    record_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
    log_path: str | Path = EXPERIMENT_LOG_PATH,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one independently configured GCN/GAT/GIN baseline experiment.

    No test information participates in training or checkpoint selection.
    Passing caches allows several fair model/split runs to reuse preprocessing.
    """

    _validate_config(config, set(BASELINE_MODEL_NAMES))
    set_random_seed(config.seed)
    pairs = _load_experiment_pairs(config, pairs_dataframe)
    data_provenance = dict(pairs.attrs["data_provenance"])
    runtime_provenance = _runtime_provenance()
    splits, split_manifest_provenance = _resolve_splits(
        pairs, config, split_manifest_path
    )

    graph_cache = (
        drug_graph_cache
        if drug_graph_cache is not None
        else build_drug_graph_cache(pairs)
    )
    sequence_cache = (
        protein_tensor_cache
        if protein_tensor_cache is not None
        else build_protein_tensor_cache(
            pairs, max_length=config.max_protein_length
        )
    )
    datasets = tuple(
        DTADataset(partition, graph_cache, sequence_cache)
        for partition in splits
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loaders = (
        _make_loader(datasets[0], config, shuffle=True, generator=generator),
        _make_loader(datasets[1], config, shuffle=False),
        _make_loader(datasets[2], config, shuffle=False),
    )
    selected_model = model if model is not None else create_dta_model(config.model)
    selected_device = torch.device(device) if device is not None else get_device()
    paths = _artifact_paths(
        config, checkpoint_path, record_path, predictions_path
    )
    return _execute_experiment(
        config=config,
        model=selected_model,
        train_loader=loaders[0],
        validation_loader=loaders[1],
        test_loader=loaders[2],
        test_dataframe=splits[2],
        splits=splits,
        protein_encoder="Protein CNN (v1 token representation)",
        configuration_used=_configuration_used(
            config,
            model_family="baseline",
            custom_model_supplied=model is not None,
            split_manifest_provenance=split_manifest_provenance,
            data_provenance=data_provenance,
            runtime_provenance=runtime_provenance,
        ),
        split_manifest_provenance=split_manifest_provenance,
        selected_device=selected_device,
        checkpoint_path=paths[0],
        record_path=paths[1],
        predictions_path=paths[2],
        log_path=log_path,
        verbose=verbose,
    )


def run_advanced_experiment(
    config: ExperimentConfig,
    residue_embedding_cache: Mapping[str, torch.Tensor],
    pairs_dataframe: pd.DataFrame | None = None,
    *,
    drug_graph_cache: Mapping[str, Any] | None = None,
    model: nn.Module | None = None,
    device: str | torch.device | None = None,
    split_manifest_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    record_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
    log_path: str | Path = EXPERIMENT_LOG_PATH,
    verbose: bool = True,
    allow_unverified_residue_cache: bool = False,
) -> dict[str, Any]:
    """Run one advanced GINE/frozen-ESM-2 cross-attention experiment.

    The supplied cache must contain residue-level tensors. ESM-2 is never
    loaded or executed by this runner, keeping its parameters frozen by design.
    """

    _validate_config(config, {"advanced"})
    if not residue_embedding_cache:
        raise ValueError("residue_embedding_cache cannot be empty.")
    set_random_seed(config.seed)
    pairs = _load_experiment_pairs(config, pairs_dataframe)
    data_provenance = dict(pairs.attrs["data_provenance"])
    runtime_provenance = _runtime_provenance()
    splits, split_manifest_provenance = _resolve_splits(
        pairs, config, split_manifest_path
    )
    from src.preprocessing import validate_residue_embedding_cache

    residue_cache_metadata = validate_residue_embedding_cache(
        dataframe=pairs,
        residue_embedding_cache=residue_embedding_cache,
        max_length=config.max_protein_length,
        expected_model_name=None,
    )
    if (
        residue_cache_metadata.get("provenance_verified") is not True
        and not allow_unverified_residue_cache
    ):
        raise ValueError(
            "Advanced experiments require a verified residue cache by default. "
            "Regenerate/load a cache with its provenance sidecar, or explicitly "
            "set allow_unverified_residue_cache=True for an auditable unsafe run."
        )
    graph_cache = (
        drug_graph_cache
        if drug_graph_cache is not None
        else build_drug_graph_cache(pairs)
    )
    datasets = tuple(
        AdvancedDTADataset(
            partition,
            graph_cache,
            residue_embedding_cache,
            max_length=config.max_protein_length,
        )
        for partition in splits
    )
    embedding_dimensions = {dataset.esm_embedding_dim for dataset in datasets}
    if len(embedding_dimensions) != 1:
        raise ValueError("Advanced splits inferred inconsistent ESM dimensions.")

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loaders = (
        _make_loader(datasets[0], config, shuffle=True, generator=generator),
        _make_loader(datasets[1], config, shuffle=False),
        _make_loader(datasets[2], config, shuffle=False),
    )
    residue_embedding_dim = next(iter(embedding_dimensions))
    selected_model = (
        model
        if model is not None
        else create_advanced_dta_model(
            esm_embedding_dim=residue_embedding_dim,
            embedding_dim=ADVANCED_HIDDEN_DIM,
            attention_heads=CROSS_ATTENTION_HEADS,
            attention_feedforward_dim=CROSS_ATTENTION_FFN_DIM,
            regression_hidden_dim=512,
            regression_bottleneck_dim=256,
            dropout=DROPOUT,
        )
    )
    selected_device = torch.device(device) if device is not None else get_device()
    paths = _artifact_paths(
        config, checkpoint_path, record_path, predictions_path
    )
    return _execute_experiment(
        config=config,
        model=selected_model,
        train_loader=loaders[0],
        validation_loader=loaders[1],
        test_loader=loaders[2],
        test_dataframe=splits[2],
        splits=splits,
        protein_encoder="Frozen ESM-2 residues + length-preserving 3-layer CNN",
        configuration_used=_configuration_used(
            config,
            model_family="advanced",
            custom_model_supplied=model is not None,
            split_manifest_provenance=split_manifest_provenance,
            data_provenance=data_provenance,
            runtime_provenance=runtime_provenance,
            residue_embedding_dim=residue_embedding_dim,
            residue_cache_metadata=residue_cache_metadata,
            allow_unverified_residue_cache=allow_unverified_residue_cache,
        ),
        split_manifest_provenance=split_manifest_provenance,
        selected_device=selected_device,
        checkpoint_path=paths[0],
        record_path=paths[1],
        predictions_path=paths[2],
        log_path=log_path,
        verbose=verbose,
    )


def _records_dataframe(
    records: pd.DataFrame | Sequence[Mapping[str, Any]] | str | Path,
) -> pd.DataFrame:
    """Normalize in-memory, CSV, or JSON experiment records."""

    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, (str, Path)):
        path = Path(records)
        if not path.exists():
            raise FileNotFoundError(f"Experiment records not found: {path}")
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)
            return pd.DataFrame(loaded if isinstance(loaded, list) else [loaded])
        raise ValueError("Experiment records path must be CSV or JSON.")
    return pd.DataFrame(list(records))


def _parse_record_configuration(value: Any) -> dict[str, Any]:
    """Parse a nested configuration from a returned record or CSV cell."""

    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                "configuration_used contains invalid JSON."
            ) from error
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _validate_strict_baseline_evidence(working: pd.DataFrame) -> None:
    """Require complete task coverage and matched non-model protocols."""

    expected_cells = {
        (dataset, split)
        for dataset in ("davis", "kiba")
        for split in ("random", "cold_drug", "cold_target")
    }
    for seed, seed_records in working.groupby("seed", sort=True):
        observed_cells = set(
            seed_records[["dataset", "split"]].itertuples(
                index=False, name=None
            )
        )
        if observed_cells != expected_cells:
            missing_cells = sorted(expected_cells.difference(observed_cells))
            extra_cells = sorted(observed_cells.difference(expected_cells))
            raise ValueError(
                f"Seed {seed} does not cover the complete Davis/KIBA random, "
                "cold-drug, and cold-target protocol; "
                f"missing={missing_cells}, extra={extra_cells}."
            )

    protocol_columns = (
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "split_fingerprint",
    )
    missing_protocol = set(protocol_columns).difference(working.columns)
    if missing_protocol:
        raise ValueError(
            "Strict baseline selection requires protocol fields: "
            f"{sorted(missing_protocol)}."
        )
    numeric_protocol = (
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
    )
    for column in numeric_protocol:
        values = pd.to_numeric(working[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Protocol field {column} must be finite and numeric.")
        working[column] = values
    if working["split_fingerprint"].isna().any() or (
        working["split_fingerprint"].astype(str).str.strip() == ""
    ).any():
        raise ValueError("split_fingerprint must be recorded for every baseline.")

    evidence_columns = ["dataset", "split", "seed"]
    for column in protocol_columns:
        mismatched = working.groupby(evidence_columns, sort=True)[column].nunique(
            dropna=False
        )
        if (mismatched != 1).any():
            examples = [tuple(index) for index in mismatched[mismatched != 1].index]
            raise ValueError(
                f"Baseline protocol field {column!r} differs across models for "
                f"evidence groups such as {examples[:3]}."
            )

    configuration_column = (
        "configuration_used"
        if "configuration_used" in working.columns
        else "configuration" if "configuration" in working.columns else None
    )
    if configuration_column is None:
        raise ValueError(
            "Strict baseline selection requires configuration_used records."
        )
    configurations = working[configuration_column].map(
        _parse_record_configuration
    )
    safety_policy_fields = {
        "optimizer": "torch.optim.Adam",
        "loss_function": "torch.nn.MSELoss",
        "checkpoint_selection_metric": "validation_mse",
        "test_evaluation_policy": "after_best_validation_checkpoint",
    }
    for field, expected in safety_policy_fields.items():
        extracted = configurations.map(lambda values: values.get(field))
        if (extracted != expected).any():
            raise ValueError(
                f"configuration_used must record {field!r} as {expected!r}."
            )
        comparison = working[evidence_columns].copy()
        comparison[field] = extracted.to_numpy()
        if (
            comparison.groupby(evidence_columns, sort=True)[field].nunique()
            != 1
        ).any():  # pragma: no cover - exact-value validation catches this first
            raise ValueError(
                f"Baseline safety-policy field {field!r} differs across models."
            )

    if "affinity_label" not in working.columns:
        raise ValueError("Strict baseline selection requires affinity_label.")
    expected_labels = working["dataset"].map(
        {"davis": "pKd", "kiba": "KIBA score"}
    )
    observed_labels = working["affinity_label"].astype(str).str.strip()
    if (observed_labels != expected_labels).any():
        raise ValueError(
            "affinity_label does not match the dataset-specific pKd/KIBA scale."
        )

    for index, configuration in configurations.items():
        dataset = str(working.loc[index, "dataset"])
        provenance = configuration.get("data_provenance")
        semantics = (
            provenance.get("affinity_semantics")
            if isinstance(provenance, Mapping)
            else None
        )
        if semantics != _expected_affinity_metadata(dataset):
            raise ValueError(
                "configuration_used data provenance must record exact "
                f"{dataset} affinity semantics."
            )

    configuration_fields = (
        "max_protein_length",
        "early_stopping_patience",
        "minimum_improvement",
    )
    for field in configuration_fields:
        extracted = configurations.map(lambda values: values.get(field))
        numeric = pd.to_numeric(extracted, errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(
                f"configuration_used must record numeric {field!r}."
            )
        comparison = working[evidence_columns].copy()
        comparison[field] = numeric.to_numpy()
        mismatched = comparison.groupby(evidence_columns, sort=True)[field].nunique()
        if (mismatched != 1).any():
            raise ValueError(
                f"Baseline configuration field {field!r} differs across models."
            )


def rank_baseline_models_by_validation(
    records: pd.DataFrame | Sequence[Mapping[str, Any]] | str | Path,
    *,
    require_complete_evidence: bool = True,
) -> pd.DataFrame:
    """Rank baselines within each dataset/split/seed, then aggregate ranks.

    Davis pKd MSE and KIBA-score MSE are never averaged together. Repeated
    records for the same model/evidence cell use the latest row. If incomplete
    evidence is allowed, only the intersection shared by every model is used.
    """

    dataframe = _records_dataframe(records)
    if "split" not in dataframe.columns and "split_type" in dataframe.columns:
        dataframe["split"] = dataframe["split_type"]
    required = {"dataset", "split", "model", "seed", "best_validation_mse"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Experiment records are missing columns: {sorted(missing)}")

    working = dataframe.copy()
    for column in ("dataset", "split", "model"):
        working[column] = working[column].astype(str).str.strip().str.lower()
    working["split"] = working["split"].str.replace("-", "_", regex=False)
    numeric_seeds = pd.to_numeric(working["seed"], errors="coerce")
    if (
        numeric_seeds.isna().any()
        or not np.isfinite(numeric_seeds.to_numpy()).all()
        or not np.equal(numeric_seeds, np.floor(numeric_seeds)).all()
    ):
        raise ValueError("seed must contain finite integer values.")
    working["seed"] = numeric_seeds.astype(int)
    working = working[working["model"].isin(BASELINE_MODEL_NAMES)].copy()
    if working.empty:
        raise ValueError("No GCN, GAT, or GIN baseline records were provided.")
    working["best_validation_mse"] = pd.to_numeric(
        working["best_validation_mse"], errors="coerce"
    )
    if not np.isfinite(working["best_validation_mse"].to_numpy()).all():
        raise ValueError("best_validation_mse must contain finite numeric values.")

    cell_columns = ["dataset", "split", "seed", "model"]
    working = working.drop_duplicates(subset=cell_columns, keep="last")
    evidence_columns = ["dataset", "split", "seed"]
    evidence_by_model = {
        model_name: set(
            map(tuple, group[evidence_columns].itertuples(index=False, name=None))
        )
        for model_name, group in working.groupby("model", sort=True)
    }
    model_names = sorted(evidence_by_model)
    expected_model_names = set(BASELINE_MODEL_NAMES)
    if require_complete_evidence and set(model_names) != expected_model_names:
        raise ValueError(
            "Complete baseline selection requires exactly GCN, GAT, and GIN; "
            f"observed {model_names}."
        )
    if len(model_names) < 2:
        raise ValueError("At least two baseline architectures are required.")

    evidence_sets = list(evidence_by_model.values())
    if require_complete_evidence and any(
        evidence != evidence_sets[0] for evidence in evidence_sets[1:]
    ):
        coverage = {model: len(groups) for model, groups in evidence_by_model.items()}
        raise ValueError(
            "Baseline models do not have identical validation evidence: "
            f"{coverage}."
        )
    shared_evidence = set.intersection(*evidence_sets)
    if not shared_evidence:
        raise ValueError("Baseline models share no dataset/split/seed evidence.")
    working = working[
        working[evidence_columns].apply(tuple, axis=1).isin(shared_evidence)
    ].copy()
    if require_complete_evidence:
        _validate_strict_baseline_evidence(working)

    working["validation_rank"] = working.groupby(
        evidence_columns, sort=True
    )["best_validation_mse"].rank(method="average", ascending=True)
    minimum_ranks = working.groupby(evidence_columns)["validation_rank"].transform(
        "min"
    )
    working["first_place"] = working["validation_rank"].eq(minimum_ranks)

    summary = (
        working.groupby("model", as_index=False)
        .agg(
            mean_validation_rank=("validation_rank", "mean"),
            median_validation_rank=("validation_rank", "median"),
            validation_rank_std=("validation_rank", lambda values: values.std(ddof=0)),
            first_place_count=("first_place", "sum"),
            evidence_groups=("validation_rank", "size"),
        )
        .sort_values(
            [
                "mean_validation_rank",
                "median_validation_rank",
                "first_place_count",
                "model",
            ],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    selection_scores = [
        (
            float(row.mean_validation_rank),
            float(row.median_validation_rank),
            -int(row.first_place_count),
        )
        for row in summary.itertuples(index=False)
    ]
    ordered_scores = sorted(set(selection_scores))
    score_to_rank = {
        score: position + 1 for position, score in enumerate(ordered_scores)
    }
    summary["selection_rank"] = [
        score_to_rank[score] for score in selection_scores
    ]
    summary["selected"] = summary["selection_rank"].eq(1)
    return summary


def select_best_baseline_architecture(
    records: pd.DataFrame | Sequence[Mapping[str, Any]] | str | Path,
    *,
    require_complete_evidence: bool = True,
) -> str:
    """Return the unique validation-rank winner, refusing unresolved ties."""

    ranking = rank_baseline_models_by_validation(
        records,
        require_complete_evidence=require_complete_evidence,
    )
    contenders = ranking[ranking["selected"]]
    if len(contenders) != 1:
        names = contenders["model"].tolist()
        raise ValueError(
            "Validation evidence does not identify one baseline architecture; "
            f"tied models: {names}."
        )
    return str(contenders.iloc[0]["model"])
