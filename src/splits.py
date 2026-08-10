"""Dataset splitting utilities for DTA experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RANDOM_SEED,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
)


SplitTuple = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


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


def validate_split_integrity(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    split_type: str,
    drug_column: str = "drug_id",
    target_column: str = "target_id",
) -> Dict[str, int | bool]:
    """Check overlap and report split statistics."""

    split_type = split_type.strip().lower()

    valid_split_types = {
        "random",
        "cold_drug",
        "cold_target",
    }

    if split_type not in valid_split_types:
        raise ValueError(
            f"split_type must be one of {sorted(valid_split_types)}."
        )

    def pair_set(dataframe: pd.DataFrame) -> set[tuple[str, str]]:
        return set(
            zip(
                dataframe[drug_column].astype(str),
                dataframe[target_column].astype(str),
            )
        )

    train_pairs = pair_set(train_dataframe)
    validation_pairs = pair_set(validation_dataframe)
    test_pairs = pair_set(test_dataframe)

    pair_overlap = bool(
        train_pairs.intersection(validation_pairs)
        or train_pairs.intersection(test_pairs)
        or validation_pairs.intersection(test_pairs)
    )

    train_drugs = set(
        train_dataframe[drug_column].astype(str)
    )
    validation_drugs = set(
        validation_dataframe[drug_column].astype(str)
    )
    test_drugs = set(
        test_dataframe[drug_column].astype(str)
    )

    train_targets = set(
        train_dataframe[target_column].astype(str)
    )
    validation_targets = set(
        validation_dataframe[target_column].astype(str)
    )
    test_targets = set(
        test_dataframe[target_column].astype(str)
    )

    drug_overlap = bool(
        train_drugs.intersection(validation_drugs)
        or train_drugs.intersection(test_drugs)
        or validation_drugs.intersection(test_drugs)
    )

    target_overlap = bool(
        train_targets.intersection(validation_targets)
        or train_targets.intersection(test_targets)
        or validation_targets.intersection(test_targets)
    )

    if pair_overlap:
        raise ValueError(
            "Identical drug-target pairs overlap across splits."
        )

    if split_type == "cold_drug" and drug_overlap:
        raise ValueError(
            "Drug identifiers overlap in the cold-drug split."
        )

    if split_type == "cold_target" and target_overlap:
        raise ValueError(
            "Target identifiers overlap in the cold-target split."
        )

    return {
        "train_rows": len(train_dataframe),
        "validation_rows": len(validation_dataframe),
        "test_rows": len(test_dataframe),
        "pair_overlap": pair_overlap,
        "drug_overlap": drug_overlap,
        "target_overlap": target_overlap,
    }


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
