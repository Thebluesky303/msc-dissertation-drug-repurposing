"""Fast tests for validation-only selection and reusable experiment runners."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from config import ExperimentConfig
import src.training as training
from src.preprocessing import (
    ResidueEmbeddingCache,
    clean_esm2_sequence,
    validate_residue_embedding_cache,
    prepare_dataset,
)


class _TinyGraphRegressor(nn.Module):
    """One-parameter regressor used to test orchestration, not model quality."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch) -> torch.Tensor:
        number_of_graphs = int(batch.batch.max().item()) + 1
        return self.bias.expand(number_of_graphs)


def _toy_pairs() -> pd.DataFrame:
    drugs = {
        "D0": "C",
        "D1": "CC",
        "D2": "CCC",
        "D3": "O",
        "D4": "CO",
        "D5": "N",
    }
    targets = {
        "T0": "ACDE",
        "T1": "FGHI",
        "T2": "KLMN",
        "T3": "PQRS",
        "T4": "TVWY",
        "T5": "AAAA",
    }
    rows = []
    for drug_index, (drug_id, smiles) in enumerate(drugs.items()):
        for target_index, (target_id, sequence) in enumerate(targets.items()):
            rows.append(
                {
                    "dataset": "davis",
                    "drug_id": drug_id,
                    "target_id": target_id,
                    "smiles": smiles,
                    "protein_sequence": sequence,
                    "affinity": 5.0 + 0.1 * drug_index + 0.05 * target_index,
                    "affinity_type": "pKd",
                }
            )
    return pd.DataFrame(rows)


def _artifact_paths(root: Path, name: str) -> dict[str, Path]:
    return {
        "checkpoint_path": root / f"{name}.pt",
        "record_path": root / f"{name}.json",
        "predictions_path": root / f"{name}_predictions.csv",
        "log_path": root / f"{name}_experiments.csv",
        "split_manifest_path": root / f"{name}_manifest.csv",
    }


@pytest.mark.parametrize("runner_name", ["baseline", "advanced"])
def test_experiment_runners_write_complete_records(
    tmp_path: Path,
    runner_name: str,
) -> None:
    pairs = _toy_pairs()
    model_name = "gcn" if runner_name == "baseline" else "advanced"
    config = ExperimentConfig(
        dataset="davis",
        split="random",
        model=model_name,
        seed=42,
        batch_size=8,
        epochs=1,
        early_stopping_patience=1,
        max_protein_length=8,
        save_predictions=True,
    )
    paths = _artifact_paths(tmp_path, runner_name)
    common_arguments = {
        "config": config,
        "pairs_dataframe": pairs,
        "model": _TinyGraphRegressor(),
        "device": "cpu",
        "verbose": False,
        **paths,
    }

    if runner_name == "baseline":
        record = training.run_baseline_experiment(**common_arguments)
    else:
        residue_cache = {
            target_id: torch.ones(4, 8) * (index + 1)
            for index, target_id in enumerate(
                pairs["target_id"].drop_duplicates().astype(str)
            )
        }
        record = training.run_advanced_experiment(
            residue_embedding_cache=residue_cache,
            allow_unverified_residue_cache=True,
            **common_arguments,
        )

    required_fields = {
        "dataset",
        "split",
        "model",
        "seed",
        "best_epoch",
        "best_validation_mse",
        "test_mse",
        "test_rmse",
        "test_mae",
        "concordance_index",
        "pearson_correlation",
        "trainable_parameters",
        "training_runtime_seconds",
        "checkpoint_path",
        "configuration_used",
    }
    assert required_fields.issubset(record)
    assert record["affinity_label"] == "pKd"
    assert record["best_epoch"] == 1
    assert sum(record["split_counts"].values()) == len(pairs)
    assert record["configuration_used"]["optimizer"] == "torch.optim.Adam"
    assert len(record["data_fingerprint"]) == 64
    assert record["data_provenance"]["mode"] == "in_memory_dataframe"
    assert record["runtime_provenance"]["python"]
    assert "torch" in record["runtime_provenance"]["package_versions"]
    assert record["split_manifest_path"] == str(
        paths["split_manifest_path"].resolve()
    )
    assert len(record["split_manifest_sha256"]) == 64
    if runner_name == "advanced":
        cache_provenance = record["configuration_used"]["architecture"][
            "residue_cache_provenance"
        ]
        assert cache_provenance["provenance_verified"] is False
        assert record["configuration_used"]["architecture"][
            "allow_unverified_residue_cache"
        ] is True
        assert record["allow_unverified_residue_cache"] is True
    assert paths["checkpoint_path"].exists()
    assert paths["predictions_path"].exists()
    assert paths["split_manifest_path"].exists()

    with paths["record_path"].open("r", encoding="utf-8") as file:
        saved_record = json.load(file)
    assert saved_record["split_fingerprint"] == record["split_fingerprint"]
    checkpoint = torch.load(
        paths["checkpoint_path"], map_location="cpu", weights_only=False
    )
    assert checkpoint["metadata"]["data_fingerprint"] == record[
        "data_fingerprint"
    ]

    experiment_log = pd.read_csv(paths["log_path"])
    expected_columns = {
        "dataset",
        "split",
        "model",
        "seed",
        "best_validation_mse",
        "test_mse",
        "test_rmse",
        "test_mae",
        "concordance_index",
        "pearson_correlation",
        "trainable_parameters",
        "training_runtime_seconds",
        "configuration_used",
    }
    assert expected_columns.issubset(experiment_log.columns)
    assert len(experiment_log) == 1


def test_fit_model_selects_only_lowest_validation_mse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    validation_mse = iter([4.0, 1.0, 3.0])

    monkeypatch.setattr(training, "train_one_epoch", lambda **_: 2.0)

    def fake_evaluate_model(**_) -> dict[str, float]:
        mse = next(validation_mse)
        return {
            "mse": mse,
            "rmse": mse**0.5,
            "mae": mse,
            "concordance_index": 0.5,
            "pearson_correlation": 0.0,
        }

    monkeypatch.setattr(training, "evaluate_model", fake_evaluate_model)
    result = training.fit_model(
        model=model,
        train_loader=object(),
        validation_loader=object(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        checkpoint_path=tmp_path / "best.pt",
        num_epochs=3,
        patience=3,
        minimum_improvement=0.0,
        verbose=False,
    )

    assert result["best_epoch"] == 2
    assert result["best_validation_metrics"]["mse"] == 1.0
    assert result["history"]["validation_mse"] == [4.0, 1.0, 3.0]


def test_verified_residue_cache_model_provenance_is_recorded() -> None:
    pairs = _toy_pairs()
    target_sequences = (
        pairs[["target_id", "protein_sequence"]]
        .drop_duplicates("target_id")
        .set_index("target_id")["protein_sequence"]
        .to_dict()
    )
    tensors = {
        str(target_id): torch.ones(4, 8) for target_id in target_sequences
    }
    cache = ResidueEmbeddingCache(
        tensors,
        metadata={
            "schema_version": 2,
            "model_name": "verified/example-esm",
            "resolved_model_revision": "fixture-revision",
            "max_length": 8,
            "embedding_dimension": 8,
            "model_provenance_verified": True,
            "artifact_integrity_verified": True,
            "cached_target_ids": sorted(tensors),
            "target_residue_lengths": {
                str(target_id): 4 for target_id in target_sequences
            },
            "tensor_file": "fixture.pt",
            "tensor_file_bytes": 1,
            "tensor_file_sha256": hashlib.sha256(b"fixture").hexdigest(),
            "target_sequence_sha256": {
                str(target_id): hashlib.sha256(
                    clean_esm2_sequence(sequence, max_length=8).encode("utf-8")
                ).hexdigest()
                for target_id, sequence in target_sequences.items()
            },
        },
    )
    metadata = validate_residue_embedding_cache(
        pairs,
        cache,
        max_length=8,
        expected_model_name=None,
    )
    configuration = training._configuration_used(
        ExperimentConfig(
            dataset="davis",
            split="random",
            model="advanced",
            max_protein_length=8,
        ),
        model_family="advanced",
        custom_model_supplied=False,
        split_manifest_provenance={"mode": "test"},
        residue_embedding_dim=8,
        residue_cache_metadata=metadata,
    )

    architecture = configuration["architecture"]
    assert architecture["esm2_model_name"] == "verified/example-esm"
    assert architecture["residue_cache_provenance"][
        "provenance_verified"
    ] is True


def _selection_record(
    dataset: str,
    split: str,
    model: str,
    validation_mse: float,
) -> dict:
    return {
        "dataset": dataset,
        "split": split,
        "model": model,
        "affinity_label": "pKd" if dataset == "davis" else "KIBA score",
        "seed": 42,
        "best_validation_mse": validation_mse,
        "batch_size": 32,
        "epochs": 30,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "split_fingerprint": f"{dataset}:{split}:seed42",
        "configuration_used": {
            "max_protein_length": 1000,
            "early_stopping_patience": 7,
            "minimum_improvement": 1e-4,
            "optimizer": "torch.optim.Adam",
            "loss_function": "torch.nn.MSELoss",
            "checkpoint_selection_metric": "validation_mse",
            "test_evaluation_policy": "after_best_validation_checkpoint",
            "data_provenance": {
                "affinity_semantics": training._expected_affinity_metadata(
                    dataset
                )
            },
        },
    }


def _complete_selection_records() -> list[dict]:
    records = []
    for dataset in ("davis", "kiba"):
        for split in ("random", "cold_drug", "cold_target"):
            model_values = (
                {"gcn": 0.10, "gat": 0.20, "gin": 0.30}
                if dataset == "davis"
                # KIBA's larger scale would dominate a raw-MSE average.
                else {"gcn": 100.0, "gat": 1000.0, "gin": 1.0}
            )
            for model, validation_mse in model_values.items():
                records.append(
                    _selection_record(
                        dataset, split, model, validation_mse
                    )
                )
    return records


def test_baseline_selection_uses_within_dataset_validation_ranks() -> None:
    records = _complete_selection_records()

    ranking = training.rank_baseline_models_by_validation(records)
    assert ranking.iloc[0]["model"] == "gcn"
    assert ranking.iloc[0]["mean_validation_rank"] == pytest.approx(1.5)
    assert training.select_best_baseline_architecture(records) == "gcn"


def test_strict_baseline_selection_rejects_protocol_mismatch() -> None:
    records = _complete_selection_records()
    for record in records:
        if (
            record["dataset"] == "davis"
            and record["split"] == "random"
            and record["model"] == "gat"
        ):
            record["batch_size"] = 64
            break

    with pytest.raises(ValueError, match="batch_size.*differs"):
        training.rank_baseline_models_by_validation(records)


def test_strict_baseline_selection_rejects_unsafe_checkpoint_policy() -> None:
    records = _complete_selection_records()
    records[0]["configuration_used"]["checkpoint_selection_metric"] = "test_mse"
    with pytest.raises(ValueError, match="checkpoint_selection_metric"):
        training.rank_baseline_models_by_validation(records)


def test_baseline_ranking_rejects_unequal_evidence_coverage() -> None:
    records = [
        {
            "dataset": "davis",
            "split": "random",
            "model": "gcn",
            "seed": 42,
            "best_validation_mse": 1.0,
        },
        {
            "dataset": "davis",
            "split": "random",
            "model": "gat",
            "seed": 42,
            "best_validation_mse": 2.0,
        },
        {
            "dataset": "kiba",
            "split": "random",
            "model": "gcn",
            "seed": 42,
            "best_validation_mse": 3.0,
        },
        {
            "dataset": "davis",
            "split": "random",
            "model": "gin",
            "seed": 42,
            "best_validation_mse": 2.5,
        },
    ]
    with pytest.raises(ValueError, match="identical validation evidence"):
        training.rank_baseline_models_by_validation(records)


def test_complete_baseline_selection_requires_gcn_gat_and_gin() -> None:
    incomplete_models = [
        {
            "dataset": "davis",
            "split": "random",
            "model": model,
            "seed": 42,
            "best_validation_mse": value,
        }
        for model, value in (("gcn", 1.0), ("gat", 2.0))
    ]
    with pytest.raises(ValueError, match="exactly GCN, GAT, and GIN"):
        training.rank_baseline_models_by_validation(incomplete_models)


def test_explicit_manifest_rejects_wrong_seed_metadata(tmp_path: Path) -> None:
    config = ExperimentConfig(
        dataset="davis",
        split="random",
        model="gcn",
        seed=42,
        epochs=1,
    )
    pairs = training._load_experiment_pairs(config, _toy_pairs())
    manifest_path = tmp_path / "davis_random_seed42_manifest.csv"
    _, provenance = training._resolve_splits(pairs, config, manifest_path)
    assert provenance["sha256"]

    metadata_path = manifest_path.with_suffix(".metadata.json")
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    metadata["seed"] = 123
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file)

    with pytest.raises(ValueError, match="seed 123 does not match"):
        training._resolve_splits(pairs, config, manifest_path)


def test_explicit_manifest_rejects_changed_scientific_data(tmp_path: Path) -> None:
    config = ExperimentConfig(
        dataset="davis", split="random", model="gcn", seed=42
    )
    pairs = training._load_experiment_pairs(config, _toy_pairs())
    manifest_path = tmp_path / "davis_random_seed42_manifest.csv"
    training._resolve_splits(pairs, config, manifest_path)
    changed = pairs.copy()
    changed.loc[0, "affinity"] += 0.5
    changed.attrs["data_provenance"] = pairs.attrs["data_provenance"]
    with pytest.raises(ValueError, match="scientific-data fingerprint"):
        training._resolve_splits(changed, config, manifest_path)


def test_in_memory_pairs_require_explicit_affinity_semantics() -> None:
    pairs = _toy_pairs().drop(columns="affinity_type")
    config = ExperimentConfig(dataset="davis", split="random", model="gcn")
    with pytest.raises(ValueError, match="explicitly declare affinity_type"):
        training._load_experiment_pairs(config, pairs)


def test_scientific_data_fingerprint_binds_values() -> None:
    config = ExperimentConfig(dataset="davis", split="random", model="gcn")
    original = training._load_experiment_pairs(config, _toy_pairs())
    changed = _toy_pairs()
    changed.loc[0, "protein_sequence"] = "CCCC"
    modified = training._load_experiment_pairs(config, changed)
    assert original.attrs["data_provenance"]["data_sha256"] != modified.attrs[
        "data_provenance"
    ]["data_sha256"]


def test_runtime_provenance_records_clean_git_as_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(arguments, **kwargs):
        del kwargs
        if arguments[1:3] == ["status", "--porcelain"]:
            return Result("")
        if arguments[1:3] == ["rev-parse", "HEAD"]:
            return Result("a" * 40 + "\n")
        if arguments[1:3] == ["branch", "--show-current"]:
            return Result("v2-refactor\n")
        raise AssertionError(arguments)

    monkeypatch.setattr(training.subprocess, "run", fake_run)
    provenance = training._runtime_provenance()
    assert provenance["git"] == {
        "head": "a" * 40,
        "branch": "v2-refactor",
        "dirty": False,
    }


def test_advanced_runner_rejects_unverified_cache_by_default(tmp_path: Path) -> None:
    pairs = _toy_pairs()
    config = ExperimentConfig(
        dataset="davis",
        split="random",
        model="advanced",
        epochs=1,
        max_protein_length=8,
    )
    cache = {
        target_id: torch.ones(4, 8)
        for target_id in pairs["target_id"].drop_duplicates()
    }
    with pytest.raises(ValueError, match="verified residue cache"):
        training.run_advanced_experiment(
            config,
            cache,
            pairs,
            model=_TinyGraphRegressor(),
            device="cpu",
            verbose=False,
            **_artifact_paths(tmp_path, "unverified"),
        )


def test_processed_csv_requires_matching_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ligands_can.txt").write_text(
        json.dumps({"D1": "C", "D2": "CC"}), encoding="utf-8"
    )
    (raw / "proteins.txt").write_text(
        json.dumps({"T1": "ACD", "T2": "EFG"}), encoding="utf-8"
    )
    import pickle
    import numpy as np

    with (raw / "Y").open("wb") as file:
        pickle.dump(np.array([[1.0, 2.0], [3.0, 4.0]]), file)
    processed = tmp_path / "davis_pairs.csv"
    prepare_dataset("davis", raw, processed)
    monkeypatch.setattr(training, "processed_pairs_path", lambda _: processed)
    config = ExperimentConfig(dataset="davis", split="random", model="gcn")
    loaded = training._load_experiment_pairs(config, None)
    assert loaded.attrs["data_provenance"]["mode"] == "verified_processed_csv"

    processed.write_text(processed.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="byte size"):
        training._load_experiment_pairs(config, None)
