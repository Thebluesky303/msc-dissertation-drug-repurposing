"""Dataset splitting utilities for DTA experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RANDOM_SEED,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
)


SplitTuple = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

SPLIT_TYPES = ("random", "cold_drug", "cold_target")
PARTITIONS = ("train", "validation", "test")
MANIFEST_REQUIRED_COLUMNS = (
    "dataset",
    "pair_id",
    "partition",
    "partition_index",
    "drug_id",
    "target_id",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_split_type(split_type: str) -> str:
    """Return the canonical spelling used in artifact names and metadata."""

    normalized = str(split_type).strip().lower().replace("-", "_")
    if normalized not in SPLIT_TYPES:
        raise ValueError(f"split_type must be one of {sorted(SPLIT_TYPES)}.")
    return normalized


def _normalize_dataset_name(dataset_name: str) -> str:
    """Validate and normalize a dataset label used in stable pair IDs."""

    normalized = str(dataset_name).strip().lower()
    if not normalized:
        raise ValueError("dataset_name must be non-empty.")
    return normalized


def _make_pair_id(dataset: str, drug_id: Any, target_id: Any) -> str:
    """Mirror the repository's stable ``dataset::drug::target`` identifier."""

    normalized_dataset = _normalize_dataset_name(dataset)
    normalized_drug = str(drug_id).strip()
    normalized_target = str(target_id).strip()
    if not normalized_drug or not normalized_target:
        raise ValueError("drug_id and target_id must be non-empty.")
    return f"{normalized_dataset}::{normalized_drug}::{normalized_target}"


def _with_pair_ids(
    dataframe: pd.DataFrame,
    dataset_name: str,
    drug_column: str = "drug_id",
    target_column: str = "target_id",
    pair_id_column: str = "pair_id",
) -> pd.DataFrame:
    """Return a copy with validated deterministic and unique pair IDs."""

    missing = {
        column
        for column in (drug_column, target_column)
        if column not in dataframe.columns
    }
    if missing:
        raise ValueError(f"Missing pair identifier columns: {sorted(missing)}")

    dataset_name = _normalize_dataset_name(dataset_name)
    result = dataframe.copy()

    if "dataset" in result.columns:
        observed_datasets = {
            str(value).strip().lower() for value in result["dataset"]
        }
        if observed_datasets and observed_datasets != {dataset_name}:
            raise ValueError(
                "Dataset column does not match dataset_name: "
                f"{sorted(observed_datasets)} versus {dataset_name!r}."
            )
        result["dataset"] = dataset_name
    else:
        result.insert(0, "dataset", dataset_name)

    generated = [
        _make_pair_id(dataset_name, drug_id, target_id)
        for drug_id, target_id in zip(
            result[drug_column],
            result[target_column],
        )
    ]

    if pair_id_column in result.columns:
        existing = result[pair_id_column].astype(str).tolist()
        if existing != generated:
            raise ValueError(
                "Existing pair_id values do not match dataset/drug/target IDs."
            )
    else:
        insertion_index = 1 if "dataset" in result.columns else 0
        result.insert(insertion_index, pair_id_column, generated)

    if result[pair_id_column].duplicated().any():
        raise ValueError("pair_id values must be unique.")
    return result


def validate_split_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    """Validate that split ratios are positive and sum to one."""

    ratios = [train_ratio, validation_ratio, test_ratio]

    if any(ratio <= 0 or ratio >= 1 for ratio in ratios):
        raise ValueError(
            "Train, validation and test ratios must be between 0 and 1."
        )

    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(
            "Train, validation and test ratios must sum to 1."
        )


def _check_pair_duplicates(
    dataframe: pd.DataFrame,
    drug_column: str,
    target_column: str,
) -> None:
    """Prevent identical drug-target pairs from leaking across splits."""

    duplicate_count = dataframe.duplicated(
        subset=[drug_column, target_column]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Dataset contains {duplicate_count} duplicated "
            "drug-target pairs."
        )


def _reset_indices(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> SplitTuple:
    """Reset dataframe indices after splitting."""

    return (
        train_dataframe.reset_index(drop=True),
        validation_dataframe.reset_index(drop=True),
        test_dataframe.reset_index(drop=True),
    )


def create_random_split(
    dataframe: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
    drug_column: str = "drug_id",
    target_column: str = "target_id",
) -> SplitTuple:
    """
    Split individual drug-target pairs randomly.

    Drugs and targets may appear in more than one split, but an identical
    pair cannot appear in multiple splits.
    """

    validate_split_ratios(
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    _check_pair_duplicates(
        dataframe,
        drug_column,
        target_column,
    )

    train_dataframe, temporary_dataframe = train_test_split(
        dataframe,
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True,
    )

    relative_validation_ratio = (
        validation_ratio / (validation_ratio + test_ratio)
    )

    validation_dataframe, test_dataframe = train_test_split(
        temporary_dataframe,
        train_size=relative_validation_ratio,
        random_state=random_seed,
        shuffle=True,
    )

    return _reset_indices(
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )


def _split_unique_entities(
    entity_ids: list[str],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> tuple[set[str], set[str], set[str]]:
    """Split unique drug or target identifiers into three disjoint sets."""

    if len(entity_ids) < 3:
        raise ValueError(
            "At least three unique entities are required."
        )

    train_entities, temporary_entities = train_test_split(
        entity_ids,
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True,
    )

    relative_validation_ratio = (
        validation_ratio / (validation_ratio + test_ratio)
    )

    validation_entities, test_entities = train_test_split(
        temporary_entities,
        train_size=relative_validation_ratio,
        random_state=random_seed,
        shuffle=True,
    )

    return (
        set(train_entities),
        set(validation_entities),
        set(test_entities),
    )


def create_cold_drug_split(
    dataframe: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
    drug_column: str = "drug_id",
    target_column: str = "target_id",
) -> SplitTuple:
    """
    Create a cold-drug split.

    No drug identifier appears in more than one split.
    """

    validate_split_ratios(
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    _check_pair_duplicates(
        dataframe,
        drug_column,
        target_column,
    )

    unique_drugs = sorted(
        dataframe[drug_column].astype(str).unique().tolist()
    )

    train_drugs, validation_drugs, test_drugs = (
        _split_unique_entities(
            unique_drugs,
            train_ratio,
            validation_ratio,
            test_ratio,
            random_seed,
        )
    )

    drug_ids = dataframe[drug_column].astype(str)

    train_dataframe = dataframe[drug_ids.isin(train_drugs)]
    validation_dataframe = dataframe[
        drug_ids.isin(validation_drugs)
    ]
    test_dataframe = dataframe[drug_ids.isin(test_drugs)]

    return _reset_indices(
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )


def create_cold_target_split(
    dataframe: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
    drug_column: str = "drug_id",
    target_column: str = "target_id",
) -> SplitTuple:
    """
    Create a cold-target split.

    No target identifier appears in more than one split.
    """

    validate_split_ratios(
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    _check_pair_duplicates(
        dataframe,
        drug_column,
        target_column,
    )

    unique_targets = sorted(
        dataframe[target_column].astype(str).unique().tolist()
    )

    train_targets, validation_targets, test_targets = (
        _split_unique_entities(
            unique_targets,
            train_ratio,
            validation_ratio,
            test_ratio,
            random_seed,
        )
    )

    target_ids = dataframe[target_column].astype(str)

    train_dataframe = dataframe[
        target_ids.isin(train_targets)
    ]
    validation_dataframe = dataframe[
        target_ids.isin(validation_targets)
    ]
    test_dataframe = dataframe[
        target_ids.isin(test_targets)
    ]

    return _reset_indices(
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )


def _validate_row_conservation(
    source_dataframe: pd.DataFrame,
    splits: SplitTuple,
    drug_column: str,
    target_column: str,
) -> None:
    """Ensure a split assigns every source pair exactly once."""

    source_pairs = set(
        zip(
            source_dataframe[drug_column].astype(str),
            source_dataframe[target_column].astype(str),
        )
    )
    assigned_pairs = [
        (str(drug_id), str(target_id))
        for partition in splits
        for drug_id, target_id in zip(
            partition[drug_column],
            partition[target_column],
        )
    ]

    if sum(len(partition) for partition in splits) != len(source_dataframe):
        raise ValueError("Split row counts do not conserve the source rows.")
    if len(assigned_pairs) != len(set(assigned_pairs)):
        raise ValueError("A drug-target pair was assigned more than once.")
    if set(assigned_pairs) != source_pairs:
        raise ValueError("Split membership does not conserve the source pairs.")


def create_split(
    dataframe: pd.DataFrame,
    split_type: str,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
    drug_column: str = "drug_id",
    target_column: str = "target_id",
) -> SplitTuple:
    """Dispatch to a v1-compatible split algorithm and verify conservation.

    This function intentionally delegates to the original public split
    functions without changing their algorithms: random pairs use the same
    two-stage ``train_test_split`` calls, while cold splits sort unique entity
    IDs before applying those same calls.
    """

    normalized_split_type = _normalize_split_type(split_type)
    split_function = {
        "random": create_random_split,
        "cold_drug": create_cold_drug_split,
        "cold_target": create_cold_target_split,
    }[normalized_split_type]

    splits = split_function(
        dataframe=dataframe,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        drug_column=drug_column,
        target_column=target_column,
    )
    _validate_row_conservation(
        source_dataframe=dataframe,
        splits=splits,
        drug_column=drug_column,
        target_column=target_column,
    )
    return splits


def _pairwise_overlap_report(
    partition_sets: Mapping[str, set[Any]],
    value_name: str,
) -> Dict[str, int | bool]:
    """Count overlap for every pair of train/validation/test partitions."""

    report: Dict[str, int | bool] = {}
    counts = []
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        count = len(partition_sets[left].intersection(partition_sets[right]))
        report[f"{value_name}_{left}_{right}_overlap_count"] = count
        counts.append(count)
    report[f"{value_name}_overlap"] = any(count > 0 for count in counts)
    return report


def _validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized manifest after strict schema and ID checks."""

    if not isinstance(manifest, pd.DataFrame):
        raise TypeError("manifest must be a pandas DataFrame.")
    missing = set(MANIFEST_REQUIRED_COLUMNS).difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}.")

    normalized = manifest.loc[:, list(MANIFEST_REQUIRED_COLUMNS)].copy()
    for column in (
        "dataset",
        "pair_id",
        "partition",
        "drug_id",
        "target_id",
    ):
        if normalized[column].isna().any():
            raise ValueError(f"Manifest column {column!r} contains null values.")
        normalized[column] = normalized[column].astype(str).str.strip()
        if normalized[column].eq("").any():
            raise ValueError(f"Manifest column {column!r} contains empty values.")

    datasets = {
        _normalize_dataset_name(value) for value in normalized["dataset"]
    }
    if len(datasets) != 1:
        raise ValueError("A split manifest must contain exactly one dataset.")
    dataset_name = next(iter(datasets))
    normalized["dataset"] = dataset_name

    invalid_partitions = set(normalized["partition"]).difference(PARTITIONS)
    if invalid_partitions:
        raise ValueError(
            f"Manifest has invalid partitions: {sorted(invalid_partitions)}."
        )
    if set(normalized["partition"]) != set(PARTITIONS):
        raise ValueError("Manifest must contain train, validation, and test rows.")

    numeric_indices = pd.to_numeric(
        normalized["partition_index"],
        errors="coerce",
    )
    if numeric_indices.isna().any() or (numeric_indices < 0).any():
        raise ValueError("partition_index values must be non-negative integers.")
    if (numeric_indices % 1 != 0).any():
        raise ValueError("partition_index values must be integers.")
    normalized["partition_index"] = numeric_indices.astype(int)

    if normalized["pair_id"].duplicated().any():
        raise ValueError("Manifest pair_id values must be globally unique.")
    if normalized.duplicated(
        subset=["partition", "partition_index"]
    ).any():
        raise ValueError(
            "partition_index values must be unique within each partition."
        )

    for partition in PARTITIONS:
        observed = sorted(
            normalized.loc[
                normalized["partition"].eq(partition),
                "partition_index",
            ].tolist()
        )
        if observed != list(range(len(observed))):
            raise ValueError(
                f"{partition} partition_index values must be contiguous from 0."
            )

    expected_pair_ids = [
        _make_pair_id(dataset_name, drug_id, target_id)
        for drug_id, target_id in zip(
            normalized["drug_id"],
            normalized["target_id"],
        )
    ]
    if normalized["pair_id"].tolist() != expected_pair_ids:
        raise ValueError(
            "Manifest pair_id values do not match dataset/drug/target IDs."
        )

    partition_order = {name: index for index, name in enumerate(PARTITIONS)}
    normalized["_partition_order"] = normalized["partition"].map(
        partition_order
    )
    normalized = normalized.sort_values(
        ["_partition_order", "partition_index"],
        kind="stable",
    ).drop(columns="_partition_order")
    return normalized.reset_index(drop=True)


def build_split_manifest(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    dataset_name: str,
    drug_column: str = "drug_id",
    target_column: str = "target_id",
    pair_id_column: str = "pair_id",
) -> pd.DataFrame:
    """Build a compact, deterministic manifest for one three-way split."""

    dataset_name = _normalize_dataset_name(dataset_name)
    manifest_parts = []
    for partition, dataframe in zip(
        PARTITIONS,
        (train_dataframe, validation_dataframe, test_dataframe),
    ):
        with_pair_ids = _with_pair_ids(
            dataframe=dataframe,
            dataset_name=dataset_name,
            drug_column=drug_column,
            target_column=target_column,
            pair_id_column=pair_id_column,
        )
        part = pd.DataFrame(
            {
                "dataset": dataset_name,
                "pair_id": with_pair_ids[pair_id_column].astype(str),
                "partition": partition,
                "partition_index": range(len(with_pair_ids)),
                "drug_id": with_pair_ids[drug_column].astype(str),
                "target_id": with_pair_ids[target_column].astype(str),
            }
        )
        manifest_parts.append(part)

    manifest = pd.concat(manifest_parts, ignore_index=True)
    return _validate_manifest(manifest)


def save_split_manifest(
    manifest: pd.DataFrame,
    path: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Validate and save a manifest, plus optional auditable JSON metadata."""

    normalized_manifest = _validate_manifest(manifest)
    output_path = Path(path)
    if output_path.suffix.lower() != ".csv":
        raise ValueError("Split manifest path must have a .csv suffix.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_manifest.to_csv(output_path, index=False, lineterminator="\n")

    if metadata is not None:
        digest = _sha256_file(output_path)
        payload = dict(metadata)
        payload.setdefault("schema_version", 1)
        payload.update(
            {
                "manifest_file": output_path.name,
                "manifest_sha256": digest,
                "manifest_rows": len(normalized_manifest),
                "partition_rows": {
                    partition: int(
                        normalized_manifest["partition"].eq(partition).sum()
                    )
                    for partition in PARTITIONS
                },
            }
        )
        metadata_path = output_path.with_suffix(".metadata.json")
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True, default=str)
            file.write("\n")
    return output_path


def load_split_manifest(path: str | Path) -> pd.DataFrame:
    """Load and strictly validate a saved CSV split manifest."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    manifest = pd.read_csv(
        manifest_path,
        dtype={
            "dataset": str,
            "pair_id": str,
            "partition": str,
            "drug_id": str,
            "target_id": str,
        },
    )
    return _validate_manifest(manifest)


def apply_split_manifest(
    dataframe: pd.DataFrame,
    manifest: pd.DataFrame,
    drug_column: str = "drug_id",
    target_column: str = "target_id",
    pair_id_column: str = "pair_id",
) -> SplitTuple:
    """Recreate split dataframes in recorded order from a manifest.

    The source and manifest pair-ID sets must match exactly. This fail-closed
    behavior prevents stale manifests from silently dropping or duplicating
    rows after data preparation changes.
    """

    normalized_manifest = _validate_manifest(manifest)
    dataset_name = normalized_manifest["dataset"].iloc[0]
    source = _with_pair_ids(
        dataframe=dataframe,
        dataset_name=dataset_name,
        drug_column=drug_column,
        target_column=target_column,
        pair_id_column=pair_id_column,
    )

    source_pair_ids = set(source[pair_id_column].astype(str))
    manifest_pair_ids = set(normalized_manifest["pair_id"])
    if source_pair_ids != manifest_pair_ids:
        missing_from_manifest = source_pair_ids.difference(manifest_pair_ids)
        missing_from_source = manifest_pair_ids.difference(source_pair_ids)
        raise ValueError(
            "Manifest/source pair IDs differ: "
            f"{len(missing_from_manifest)} source-only and "
            f"{len(missing_from_source)} manifest-only."
        )

    source_by_pair_id = source.set_index(pair_id_column, drop=False)
    for row in normalized_manifest.itertuples(index=False):
        source_row = source_by_pair_id.loc[row.pair_id]
        if (
            str(source_row[drug_column]).strip() != row.drug_id
            or str(source_row[target_column]).strip() != row.target_id
        ):
            raise ValueError(
                f"Manifest identifiers disagree for pair {row.pair_id!r}."
            )

    reconstructed = []
    for partition in PARTITIONS:
        pair_ids = normalized_manifest.loc[
            normalized_manifest["partition"].eq(partition),
            "pair_id",
        ].tolist()
        reconstructed.append(
            source_by_pair_id.loc[pair_ids].reset_index(drop=True)
        )

    splits = tuple(reconstructed)
    _validate_row_conservation(
        source_dataframe=source,
        splits=splits,
        drug_column=drug_column,
        target_column=target_column,
    )
    return splits  # type: ignore[return-value]


def materialize_split_manifests(
    dataframe: pd.DataFrame,
    dataset_name: str,
    output_root: str | Path,
    seed: int = RANDOM_SEED,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
    drug_column: str = "drug_id",
    target_column: str = "target_id",
    pair_id_column: str = "pair_id",
) -> Dict[str, Path]:
    """Create and save all v1-compatible split manifests for a dataset.

    Metadata records pairwise ID leakage checks and structural-equivalence
    diagnostics. When only a raw ``smiles`` column exists, values are
    canonicalized solely for diagnostics; memberships remain ID based.
    """

    dataset_name = _normalize_dataset_name(dataset_name)
    source = _with_pair_ids(
        dataframe=dataframe,
        dataset_name=dataset_name,
        drug_column=drug_column,
        target_column=target_column,
        pair_id_column=pair_id_column,
    )
    output_directory = Path(output_root)
    output_directory.mkdir(parents=True, exist_ok=True)

    if "canonical_smiles" in source.columns:
        canonical_smiles_column = "canonical_smiles"
        canonicalize_smiles_values = False
    elif "smiles" in source.columns:
        canonical_smiles_column = "smiles"
        canonicalize_smiles_values = True
    else:
        canonical_smiles_column = None
        canonicalize_smiles_values = False
    protein_sequence_column = (
        "protein_sequence" if "protein_sequence" in source.columns else None
    )

    source_fingerprint = hashlib.sha256(
        "\n".join(source[pair_id_column].astype(str)).encode("utf-8")
    ).hexdigest()
    source_data_columns = sorted(source.columns.astype(str).tolist())
    try:
        from src.preprocessing import scientific_pairs_sha256

        source_data_fingerprint = scientific_pairs_sha256(source)
        source_data_fingerprint_algorithm = "scientific_pairs_sha256_v1"
    except ValueError:
        source_data_fingerprint = hashlib.sha256(
            source[source_data_columns]
            .to_csv(index=False, lineterminator="\n")
            .encode("utf-8")
        ).hexdigest()
        source_data_fingerprint_algorithm = "ordered_all_available_columns_sha256_v1"
    manifest_paths: Dict[str, Path] = {}

    for split_type in SPLIT_TYPES:
        splits = create_split(
            dataframe=source,
            split_type=split_type,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            random_seed=seed,
            drug_column=drug_column,
            target_column=target_column,
        )
        integrity = validate_split_integrity(
            train_dataframe=splits[0],
            validation_dataframe=splits[1],
            test_dataframe=splits[2],
            split_type=split_type,
            drug_column=drug_column,
            target_column=target_column,
            canonical_smiles_column=canonical_smiles_column,
            protein_sequence_column=protein_sequence_column,
            canonicalize_smiles_values=canonicalize_smiles_values,
        )
        manifest = build_split_manifest(
            train_dataframe=splits[0],
            validation_dataframe=splits[1],
            test_dataframe=splits[2],
            dataset_name=dataset_name,
            drug_column=drug_column,
            target_column=target_column,
            pair_id_column=pair_id_column,
        )
        path = output_directory / (
            f"{dataset_name}_{split_type}_seed{seed}_manifest.csv"
        )
        manifest_paths[split_type] = save_split_manifest(
            manifest=manifest,
            path=path,
            metadata={
                "dataset": dataset_name,
                "split_type": split_type,
                "seed": int(seed),
                "ratios": {
                    "train": train_ratio,
                    "validation": validation_ratio,
                    "test": test_ratio,
                },
                "source_rows": len(source),
                "source_pair_id_sha256": source_fingerprint,
                "source_pair_id_fingerprint_algorithm": "ordered_pair_ids_sha256_v1",
                "source_data_sha256": source_data_fingerprint,
                "source_data_columns": source_data_columns,
                "source_data_fingerprint_algorithm": (
                    source_data_fingerprint_algorithm
                ),
                "integrity": integrity,
                "membership_policy": "ID-based v1-compatible split",
                "structural_diagnostics_change_membership": False,
            },
        )
    return manifest_paths


def _canonical_smiles_set(
    dataframe: pd.DataFrame,
    column: str,
    canonicalize_values: bool,
) -> set[str]:
    """Return distinct canonical SMILES, optionally canonicalizing with RDKit."""

    values = {
        str(value).strip()
        for value in dataframe[column].dropna()
        if str(value).strip()
    }
    if not canonicalize_values:
        return values

    try:
        from rdkit import Chem
    except ImportError as error:  # pragma: no cover - dependency is required
        raise ImportError(
            "RDKit is required to canonicalize SMILES diagnostics."
        ) from error

    canonical_values: set[str] = set()
    for value in values:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(
                f"Cannot canonicalize invalid SMILES value {value!r}."
            )
        canonical_values.add(Chem.MolToSmiles(molecule, canonical=True))
    return canonical_values


def _protein_sequence_set(
    dataframe: pd.DataFrame,
    column: str,
) -> set[str]:
    """Return whitespace-insensitive, upper-case protein sequences."""

    return {
        "".join(str(value).split()).upper()
        for value in dataframe[column].dropna()
        if "".join(str(value).split())
    }


def structural_overlap_diagnostics(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    canonical_smiles_column: str | None = None,
    protein_sequence_column: str | None = None,
    canonicalize_smiles_values: bool = False,
) -> Dict[str, int | bool]:
    """Report structural-equivalence overlap without changing membership.

    If ``canonical_smiles_column`` already stores canonical strings, leave
    ``canonicalize_smiles_values`` false. Set it true for a raw SMILES column;
    only the diagnostic values are canonicalized and the ID-based split is
    never modified. Protein sequences are compared after removing whitespace
    and converting to upper case.
    """

    partitions = {
        "train": train_dataframe,
        "validation": validation_dataframe,
        "test": test_dataframe,
    }
    report: Dict[str, int | bool] = {
        "canonical_smiles_diagnostics_available": (
            canonical_smiles_column is not None
        ),
        "protein_sequence_diagnostics_available": (
            protein_sequence_column is not None
        ),
    }

    if canonical_smiles_column is not None:
        missing = [
            name
            for name, dataframe in partitions.items()
            if canonical_smiles_column not in dataframe.columns
        ]
        if missing:
            raise ValueError(
                f"Column {canonical_smiles_column!r} is missing from "
                f"partitions: {missing}."
            )
        smiles_sets = {
            name: _canonical_smiles_set(
                dataframe,
                canonical_smiles_column,
                canonicalize_smiles_values,
            )
            for name, dataframe in partitions.items()
        }
        report.update(
            _pairwise_overlap_report(smiles_sets, "canonical_smiles")
        )

    if protein_sequence_column is not None:
        missing = [
            name
            for name, dataframe in partitions.items()
            if protein_sequence_column not in dataframe.columns
        ]
        if missing:
            raise ValueError(
                f"Column {protein_sequence_column!r} is missing from "
                f"partitions: {missing}."
            )
        sequence_sets = {
            name: _protein_sequence_set(dataframe, protein_sequence_column)
            for name, dataframe in partitions.items()
        }
        report.update(
            _pairwise_overlap_report(sequence_sets, "protein_sequence")
        )

    return report


def validate_split_integrity(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    split_type: str,
    drug_column: str = "drug_id",
    target_column: str = "target_id",
    canonical_smiles_column: str | None = None,
    protein_sequence_column: str | None = None,
    canonicalize_smiles_values: bool = False,
) -> Dict[str, int | bool]:
    """Check pairwise leakage and report optional structural diagnostics."""

    normalized_split_type = _normalize_split_type(split_type)
    partitions = {
        "train": train_dataframe,
        "validation": validation_dataframe,
        "test": test_dataframe,
    }

    for name, dataframe in partitions.items():
        missing = {
            column
            for column in (drug_column, target_column)
            if column not in dataframe.columns
        }
        if missing:
            raise ValueError(
                f"{name} partition is missing columns: {sorted(missing)}."
            )
        if dataframe.duplicated(
            subset=[drug_column, target_column]
        ).any():
            raise ValueError(
                f"{name} partition contains duplicate drug-target pairs."
            )

    pair_sets = {
        name: set(
            zip(
                dataframe[drug_column].astype(str),
                dataframe[target_column].astype(str),
            )
        )
        for name, dataframe in partitions.items()
    }
    drug_sets = {
        name: set(dataframe[drug_column].astype(str))
        for name, dataframe in partitions.items()
    }
    target_sets = {
        name: set(dataframe[target_column].astype(str))
        for name, dataframe in partitions.items()
    }

    report: Dict[str, int | bool] = {
        "train_rows": len(train_dataframe),
        "validation_rows": len(validation_dataframe),
        "test_rows": len(test_dataframe),
        "total_rows": sum(len(dataframe) for dataframe in partitions.values()),
    }
    report.update(_pairwise_overlap_report(pair_sets, "pair"))
    report.update(_pairwise_overlap_report(drug_sets, "drug"))
    report.update(_pairwise_overlap_report(target_sets, "target"))

    if report["pair_overlap"]:
        raise ValueError("Identical drug-target pairs overlap across splits.")
    if normalized_split_type == "cold_drug" and report["drug_overlap"]:
        raise ValueError("Drug identifiers overlap in the cold-drug split.")
    if normalized_split_type == "cold_target" and report["target_overlap"]:
        raise ValueError("Target identifiers overlap in the cold-target split.")

    report.update(
        structural_overlap_diagnostics(
            train_dataframe=train_dataframe,
            validation_dataframe=validation_dataframe,
            test_dataframe=test_dataframe,
            canonical_smiles_column=canonical_smiles_column,
            protein_sequence_column=protein_sequence_column,
            canonicalize_smiles_values=canonicalize_smiles_values,
        )
    )
    return report


def save_splits(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    output_directory: str | Path,
    dataset_name: str,
) -> None:
    """Save train, validation and test dataframes as CSV files."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    dataset_name = dataset_name.strip().lower()

    train_dataframe.to_csv(
        output_directory / f"{dataset_name}_train.csv",
        index=False,
    )

    validation_dataframe.to_csv(
        output_directory / f"{dataset_name}_validation.csv",
        index=False,
    )

    test_dataframe.to_csv(
        output_directory / f"{dataset_name}_test.csv",
        index=False,
    )
