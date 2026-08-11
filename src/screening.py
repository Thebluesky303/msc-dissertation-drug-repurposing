"""Reusable preparation, inference, ranking and audit utilities for AML screens.

Both architecture-neutral callbacks and strictly provenance-checked v2 training
checkpoints are supported. This module performs inference only: it never trains
models, downloads artifacts or executes ESM-2.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from config import (
    ADVANCED_GINE_LAYERS,
    ADVANCED_PROTEIN_CNN_LAYERS,
    BOND_FEATURE_DIM,
    DATASET_AFFINITY_LABELS,
    DRUG_HIDDEN_DIM,
    FUSION_HIDDEN_DIM,
    NODE_FEATURE_DIM,
    PROTEIN_EMBEDDING_DIM,
    PROTEIN_HIDDEN_DIM,
)
from src.dataset import AdvancedDTADataset, DTADataset
from src.models import create_advanced_dta_model, create_dta_model
from src.preprocessing import (
    build_drug_graph_cache,
    build_protein_tensor_cache,
    smiles_to_graph,
)
from src.training import predict
from src.utils import get_device


SCREENING_RESULT_DESCRIPTION = (
    "computationally prioritised drug-repurposing candidates"
)

_DEFAULT_SCORE_COLUMN = "predicted_binding_affinity_score"
_STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

__all__ = [
    "SCREENING_RESULT_DESCRIPTION",
    "ScreeningComparisonResult",
    "DualModelScreeningResult",
    "LoadedScreeningCheckpoint",
    "ScreeningRankingResult",
    "build_screening_pairs",
    "compare_screening_models",
    "create_aml_screening_pairs",
    "load_screening_checkpoint",
    "predict_advanced_checkpoint",
    "predict_baseline_checkpoint",
    "predict_screening_pairs",
    "prepare_aml_targets",
    "prepare_approved_drug_library",
    "prepare_approved_drugs",
    "rank_screening_predictions",
    "run_screening_inference",
    "run_dual_model_aml_screening",
    "screening_wording_metadata",
    "screening_input_provenance",
    "stable_screening_pair_id",
    "validate_aml_targets",
    "validate_approved_drug_dataframe",
    "validate_screening_checkpoint_compatibility",
]


@dataclass(frozen=True)
class ScreeningRankingResult:
    """Ranked predictions and the three requested AML shortlists."""

    ranked: pd.DataFrame
    top_10_overall: pd.DataFrame
    top_20_overall: pd.DataFrame
    top_5_per_target: pd.DataFrame
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ScreeningComparisonResult:
    """Simple baseline-versus-advanced AML screening comparison."""

    pair_comparison: pd.DataFrame
    shared_top_drugs: pd.DataFrame
    top_pair_overlap: pd.DataFrame
    target_agreement: pd.DataFrame
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class LoadedScreeningCheckpoint:
    """Reconstructed inference model plus verified training provenance."""

    model: nn.Module
    path: Path
    sha256: str
    configuration_sha256: str
    model_family: str
    model_name: str
    training_dataset: str
    affinity_label: str
    configuration: Mapping[str, Any]
    data_fingerprint: str
    data_provenance: Mapping[str, Any]
    runtime_provenance: Mapping[str, Any]
    split_fingerprint: str
    split_manifest: Mapping[str, Any]
    best_epoch: int
    best_validation_mse: float
    device: torch.device

    @property
    def provenance(self) -> dict[str, Any]:
        """Return serialisable checkpoint provenance for result artifacts."""

        return {
            "checkpoint_path": str(self.path),
            "checkpoint_sha256": self.sha256,
            "checkpoint_configuration_sha256": self.configuration_sha256,
            "checkpoint_model_family": self.model_family,
            "checkpoint_model_name": self.model_name,
            "training_dataset": self.training_dataset,
            "affinity_label": self.affinity_label,
            "checkpoint_configuration": dict(self.configuration),
            "checkpoint_data_fingerprint": self.data_fingerprint,
            "checkpoint_data_provenance": dict(self.data_provenance),
            "checkpoint_runtime_provenance": dict(self.runtime_provenance),
            "checkpoint_split_fingerprint": self.split_fingerprint,
            "checkpoint_split_manifest": dict(self.split_manifest),
            "checkpoint_best_epoch": self.best_epoch,
            "checkpoint_best_validation_mse": self.best_validation_mse,
        }


@dataclass(frozen=True)
class DualModelScreeningResult:
    """Checkpoint-backed baseline and advanced AML screening artifacts."""

    baseline_predictions: pd.DataFrame
    advanced_predictions: pd.DataFrame
    baseline_rankings: ScreeningRankingResult
    advanced_rankings: ScreeningRankingResult
    comparison: ScreeningComparisonResult
    provenance: Mapping[str, Any]


def screening_wording_metadata() -> dict[str, str]:
    """Return safe, consistent wording for AML screening outputs."""

    return {
        "result_description": SCREENING_RESULT_DESCRIPTION,
        "evidence_type": "in-silico model prediction",
        "interpretation": (
            "These rankings are computational hypotheses for follow-up; they "
            "are not experimentally or clinically validated treatments."
        ),
    }


def _require_dataframe(dataframe: pd.DataFrame, name: str) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if dataframe.empty:
        raise ValueError(f"{name} must contain at least one row.")


def _require_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    name: str,
) -> None:
    missing = sorted(set(columns).difference(dataframe.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _validate_nonempty_strings(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    name: str,
) -> None:
    for column in columns:
        invalid = ~dataframe[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        if invalid.any():
            rows = dataframe.index[invalid].tolist()[:5]
            raise ValueError(
                f"{name}.{column} contains missing/non-string values "
                f"at rows such as {rows}."
            )


def validate_approved_drug_dataframe(
    dataframe: pd.DataFrame,
    *,
    drug_id_column: str = "drug_id",
    drug_name_column: str = "drug_name",
    smiles_column: str = "smiles",
    require_prepared: bool = False,
) -> None:
    """Validate the schema and identifier mapping of an approved-drug table.

    Approval and provenance columns are intentionally not hard-coded because
    public drug sources expose different metadata schemas. All extra columns
    are retained by :func:`prepare_approved_drugs`.
    """

    name = "approved_drugs"
    _require_dataframe(dataframe, name)
    _require_columns(
        dataframe,
        [drug_id_column, drug_name_column, smiles_column],
        name,
    )
    _validate_nonempty_strings(
        dataframe,
        [drug_id_column, drug_name_column, smiles_column],
        name,
    )

    if require_prepared:
        _require_columns(dataframe, ["canonical_smiles", "inchikey"], name)
        if dataframe[drug_id_column].duplicated().any():
            raise ValueError("Prepared drug IDs must be unique.")
        if dataframe["canonical_smiles"].duplicated().any():
            raise ValueError("Prepared canonical structures must be unique.")


def _mol_from_smiles_without_rdkit_stderr(smiles: str) -> Chem.Mol | None:
    """Parse SMILES while keeping expected invalid-record noise out of logs."""

    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(smiles)


def _inchi_key(molecule: Chem.Mol) -> Any:
    """Calculate an InChIKey when the installed RDKit build supports InChI."""

    try:
        value = Chem.MolToInchiKey(molecule)
    except (AttributeError, RuntimeError, ValueError):
        return pd.NA
    return value or pd.NA


def prepare_approved_drugs(
    dataframe: pd.DataFrame,
    *,
    drug_id_column: str = "drug_id",
    drug_name_column: str = "drug_name",
    smiles_column: str = "smiles",
    allow_multicomponent: bool = False,
    graph_builder: Callable[[str], Any] = smiles_to_graph,
    return_rejections: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Clean and validate an approved small-molecule screening library.

    Invalid structures, multi-component records (salts/mixtures by default),
    graph-incompatible structures and duplicate canonical structures are
    excluded. The first occurrence of a duplicate structure is retained in
    source order. Source approval/provenance fields and all other input metadata
    are preserved.

    Args:
        dataframe: Source approved-drug records.
        allow_multicomponent: Opt in to structures containing multiple RDKit
            fragments. The default is appropriate for model-ready parent drugs.
        graph_builder: Validation callback, normally the project's verified
            ``smiles_to_graph`` implementation.
        return_rejections: Also return rejected source rows with a
            ``rejection_reason`` column for a reproducible audit trail.
    """

    _require_dataframe(dataframe, "approved_drugs")
    _require_columns(
        dataframe,
        [drug_id_column, drug_name_column, smiles_column],
        "approved_drugs",
    )
    if not callable(graph_builder):
        raise TypeError("graph_builder must be callable.")

    source = dataframe.copy(deep=True)
    accepted_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    canonical_to_drug_id: dict[str, str] = {}
    accepted_by_drug_id: dict[str, tuple[str, str]] = {}

    for source_position, (_, row) in enumerate(source.iterrows()):
        record = row.to_dict()
        raw_drug_id = record[drug_id_column]
        raw_drug_name = record[drug_name_column]
        raw_smiles = record[smiles_column]
        required_values = {
            "drug_id": raw_drug_id,
            "drug_name": raw_drug_name,
        }
        normalised_values: dict[str, str] = {}
        invalid_field: str | None = None
        for field_name, value in required_values.items():
            if isinstance(value, (Mapping, Sequence)) and not isinstance(
                value,
                (str, bytes),
            ):
                invalid_field = field_name
                break
            try:
                missing = bool(pd.isna(value))
            except (TypeError, ValueError):
                missing = True
            if missing:
                text = ""
            elif isinstance(value, (float, np.floating)) and float(
                value
            ).is_integer():
                # CSV readers commonly promote integer IDs to float when a
                # column contains one missing value. Preserve identifier text.
                text = str(int(value))
            else:
                text = str(value).strip()
            if not text:
                invalid_field = field_name
                break
            normalised_values[field_name] = text
        if invalid_field is not None:
            record["source_row"] = source_position
            record["rejection_reason"] = f"missing_or_invalid_{invalid_field}"
            rejected_records.append(record)
            continue
        if not isinstance(raw_smiles, str):
            record["source_row"] = source_position
            record["rejection_reason"] = "missing_or_invalid_smiles"
            rejected_records.append(record)
            continue
        input_smiles = raw_smiles.strip()
        if not input_smiles:
            record["source_row"] = source_position
            record["rejection_reason"] = "missing_or_invalid_smiles"
            rejected_records.append(record)
            continue

        normalised_drug_id = normalised_values["drug_id"]
        normalised_drug_name = normalised_values["drug_name"]
        if raw_drug_id != normalised_drug_id:
            record[f"source_{drug_id_column}"] = raw_drug_id
        if raw_drug_name != normalised_drug_name:
            record[f"source_{drug_name_column}"] = raw_drug_name
        record[drug_id_column] = normalised_drug_id
        record[drug_name_column] = normalised_drug_name
        record["input_smiles"] = input_smiles
        record["source_row"] = source_position

        molecule = _mol_from_smiles_without_rdkit_stderr(input_smiles)
        if molecule is None or molecule.GetNumAtoms() == 0:
            record["rejection_reason"] = "invalid_smiles"
            rejected_records.append(record)
            continue

        fragments = Chem.GetMolFrags(molecule)
        if len(fragments) != 1 and not allow_multicomponent:
            record["rejection_reason"] = "multicomponent_structure"
            rejected_records.append(record)
            continue

        canonical_smiles = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
        try:
            graph = graph_builder(canonical_smiles)
            if graph is None:
                raise ValueError("graph builder returned None")
            number_of_nodes = getattr(graph, "num_nodes", None)
            if number_of_nodes is not None and int(number_of_nodes) < 1:
                raise ValueError("graph contains no atoms")
        except Exception as error:  # source records must be audited, not fatal
            record["rejection_reason"] = (
                f"graph_incompatible:{type(error).__name__}"
            )
            rejected_records.append(record)
            continue

        drug_id = str(record[drug_id_column])
        drug_name = str(record[drug_name_column])
        previous_id_record = accepted_by_drug_id.get(drug_id)
        if previous_id_record is not None:
            previous_smiles, previous_name = previous_id_record
            record["canonical_smiles"] = canonical_smiles
            record["duplicate_of_drug_id"] = drug_id
            record["rejection_reason"] = (
                "duplicate_drug_id"
                if (canonical_smiles, drug_name)
                == (previous_smiles, previous_name)
                else "conflicting_drug_id"
            )
            rejected_records.append(record)
            continue

        duplicate_of = canonical_to_drug_id.get(canonical_smiles)
        if duplicate_of is not None:
            record["canonical_smiles"] = canonical_smiles
            record["duplicate_of_drug_id"] = duplicate_of
            record["rejection_reason"] = "duplicate_structure"
            rejected_records.append(record)
            continue

        canonical_to_drug_id[canonical_smiles] = drug_id
        accepted_by_drug_id[drug_id] = (canonical_smiles, drug_name)

        if "inchikey" in record:
            record["source_inchikey"] = record["inchikey"]
        record["canonical_smiles"] = canonical_smiles
        record["inchikey"] = _inchi_key(molecule)
        # Downstream project utilities conventionally read ``smiles``. Keep the
        # original value in input_smiles and expose the canonical parent here.
        record[smiles_column] = canonical_smiles
        record.pop("rejection_reason", None)
        accepted_records.append(record)

    prepared = pd.DataFrame(accepted_records)
    rejected = pd.DataFrame(rejected_records)
    if prepared.empty:
        reasons = (
            rejected.get("rejection_reason", pd.Series(dtype=str))
            .value_counts()
            .to_dict()
        )
        raise ValueError(
            "No approved-drug records passed preparation. "
            f"Rejection counts: {reasons}"
        )

    prepared = prepared.drop(columns=["source_row"], errors="ignore")
    prepared = prepared.reset_index(drop=True)
    rejected = rejected.reset_index(drop=True)
    validate_approved_drug_dataframe(
        prepared,
        drug_id_column=drug_id_column,
        drug_name_column=drug_name_column,
        smiles_column=smiles_column,
        require_prepared=True,
    )

    preparation_summary = {
        "input_records": int(len(source)),
        "accepted_unique_structures": int(len(prepared)),
        "rejected_records": int(len(rejected)),
        "allow_multicomponent": bool(allow_multicomponent),
        "rejection_counts": (
            rejected.get("rejection_reason", pd.Series(dtype=str))
            .value_counts()
            .to_dict()
        ),
    }
    prepared.attrs["preparation_summary"] = preparation_summary
    rejected.attrs["preparation_summary"] = preparation_summary

    if return_rejections:
        return prepared, rejected
    return prepared


def validate_aml_targets(
    dataframe: pd.DataFrame,
    *,
    target_id_column: str = "target_id",
    sequence_column: str = "protein_sequence",
) -> None:
    """Validate AML target identifiers and sequences without transforming them."""

    name = "aml_targets"
    _require_dataframe(dataframe, name)
    _require_columns(dataframe, [target_id_column, sequence_column], name)
    _validate_nonempty_strings(
        dataframe,
        [target_id_column, sequence_column],
        name,
    )
    if dataframe[target_id_column].duplicated().any():
        duplicates = (
            dataframe.loc[
                dataframe[target_id_column].duplicated(keep=False),
                target_id_column,
            ]
            .astype(str)
            .unique()
            .tolist()[:5]
        )
        raise ValueError(f"AML target IDs must be unique; found {duplicates}.")

    for index, sequence in dataframe[sequence_column].items():
        if sequence != sequence.strip() or not sequence.isalpha():
            raise ValueError(
                f"AML target sequence at row {index} must contain only "
                "unspaced amino-acid letters."
            )
        if sequence != sequence.upper():
            raise ValueError(
                f"AML target sequence at row {index} must already be uppercase; "
                "the preparation step never changes established sequences."
            )


def prepare_aml_targets(
    dataframe: pd.DataFrame,
    *,
    target_id_column: str = "target_id",
    sequence_column: str = "protein_sequence",
) -> pd.DataFrame:
    """Return validated AML targets without altering biological identity.

    Non-null scalar IDs inferred as numeric by a CSV reader are normalised to
    their stable string representation. Their source values are retained in a
    ``source_<target_id_column>`` audit column; sequences are never transformed.
    """

    _require_dataframe(dataframe, "aml_targets")
    _require_columns(
        dataframe,
        [target_id_column, sequence_column],
        "aml_targets",
    )
    prepared = dataframe.copy(deep=True).reset_index(drop=True)
    normalised_ids: list[str] = []
    source_ids: list[Any] = []
    source_id_changed = False
    for index, value in prepared[target_id_column].items():
        if isinstance(value, (Mapping, Sequence)) and not isinstance(
            value,
            (str, bytes),
        ):
            raise ValueError(
                f"aml_targets.{target_id_column} contains an invalid value "
                f"at row {index}."
            )
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = True
        if missing:
            normalised = ""
        elif isinstance(value, (float, np.floating)) and float(
            value
        ).is_integer():
            normalised = str(int(value))
        else:
            normalised = str(value).strip()
        if not normalised:
            raise ValueError(
                f"aml_targets.{target_id_column} contains a missing/blank "
                f"value at row {index}."
            )
        source_ids.append(value)
        normalised_ids.append(normalised)
        source_id_changed = source_id_changed or value != normalised
    if source_id_changed:
        source_column = f"source_{target_id_column}"
        if source_column in prepared.columns:
            raise ValueError(
                f"AML targets already contain reserved audit column "
                f"{source_column!r}."
            )
        prepared[source_column] = source_ids
    prepared[target_id_column] = normalised_ids
    validate_aml_targets(
        prepared,
        target_id_column=target_id_column,
        sequence_column=sequence_column,
    )
    calculated_lengths = prepared[sequence_column].str.len().astype(int)
    if "sequence_length" in prepared.columns:
        existing = pd.to_numeric(prepared["sequence_length"], errors="coerce")
        if existing.isna().any() or not np.array_equal(
            existing.to_numpy(dtype=int),
            calculated_lengths.to_numpy(),
        ):
            raise ValueError("Existing sequence_length values are inconsistent.")
    else:
        prepared["sequence_length"] = calculated_lengths

    prepared["contains_nonstandard_residue"] = prepared[sequence_column].map(
        lambda sequence: not set(sequence).issubset(_STANDARD_AMINO_ACIDS)
    )
    return prepared


def stable_screening_pair_id(
    drug_id: str,
    target_id: str,
    *,
    prefix: str = "AMLPAIR",
) -> str:
    """Create an order-independent, reproducible ID for one drug-target pair."""

    if not isinstance(drug_id, str) or not drug_id.strip():
        raise ValueError("drug_id must be a non-empty string.")
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("target_id must be a non-empty string.")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string.")

    payload = json.dumps(
        [drug_id, target_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = sha256(payload).hexdigest()[:20].upper()
    return f"{prefix}_{digest}"


def build_screening_pairs(
    approved_drugs: pd.DataFrame,
    aml_targets: pd.DataFrame,
    *,
    drug_id_column: str = "drug_id",
    target_id_column: str = "target_id",
    sequence_column: str = "protein_sequence",
    pair_id_prefix: str = "AMLPAIR",
) -> pd.DataFrame:
    """Build the complete approved-drug x AML-target Cartesian product."""

    _require_dataframe(approved_drugs, "approved_drugs")
    _require_columns(approved_drugs, [drug_id_column], "approved_drugs")
    _validate_nonempty_strings(
        approved_drugs,
        [drug_id_column],
        "approved_drugs",
    )
    if approved_drugs[drug_id_column].duplicated().any():
        raise ValueError("Approved drug IDs must be unique before pairing.")

    validate_aml_targets(
        aml_targets,
        target_id_column=target_id_column,
        sequence_column=sequence_column,
    )

    drugs = approved_drugs.copy(deep=True)
    targets = aml_targets.copy(deep=True)
    pairs = drugs.merge(
        targets,
        how="cross",
        suffixes=("_drug", "_target"),
        sort=False,
    )
    pairs.insert(0, "pair_index", np.arange(1, len(pairs) + 1, dtype=int))
    pairs.insert(
        0,
        "pair_id",
        [
            stable_screening_pair_id(
                str(drug_id),
                str(target_id),
                prefix=pair_id_prefix,
            )
            for drug_id, target_id in zip(
                pairs[drug_id_column],
                pairs[target_id_column],
                strict=True,
            )
        ],
    )
    if pairs["pair_id"].duplicated().any():
        raise RuntimeError("Stable pair-ID collision detected.")
    if len(pairs) != len(approved_drugs) * len(aml_targets):
        raise RuntimeError("Screening Cartesian product has an invalid row count.")
    return pairs


def _coerce_prediction_output(
    output: Any,
    pairs: pd.DataFrame,
    score_column: str,
) -> np.ndarray:
    """Normalise a callback result to scores in the input pair order."""

    if isinstance(output, pd.DataFrame):
        _require_columns(output, [score_column], "predictor output")
        if "pair_id" in output.columns:
            if output["pair_id"].duplicated().any():
                raise ValueError("Predictor output pair_id values must be unique.")
            expected = set(pairs["pair_id"].astype(str))
            observed = set(output["pair_id"].astype(str))
            if expected != observed:
                raise ValueError("Predictor output pair IDs do not match the input.")
            lookup = output.set_index(output["pair_id"].astype(str))[score_column]
            values = lookup.loc[pairs["pair_id"].astype(str)].to_numpy()
        else:
            values = output[score_column].to_numpy()
    elif isinstance(output, pd.Series):
        values = output.to_numpy()
    else:
        values = np.asarray(output)

    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) != len(pairs):
        raise ValueError(
            "Predictor must return exactly one score per screening pair; "
            f"expected {len(pairs)}, received {len(values)}."
        )
    if not np.isfinite(values).all():
        raise ValueError("Predictor returned missing or non-finite scores.")
    return values


def predict_screening_pairs(
    pairs: pd.DataFrame,
    predictor: Callable[[pd.DataFrame], Any],
    *,
    score_column: str = _DEFAULT_SCORE_COLUMN,
    model_name: str | None = None,
    batch_size: int | None = None,
) -> pd.DataFrame:
    """Run architecture-agnostic screening inference through a callback.

    The callback receives a copy of the pair table (or successive batches) and
    returns one numeric score per row. It may instead return a DataFrame with
    ``pair_id`` and ``score_column``; those results are safely realigned.
    """

    _require_dataframe(pairs, "pairs")
    _require_columns(pairs, ["pair_id"], "pairs")
    if pairs["pair_id"].duplicated().any():
        raise ValueError("Screening pair IDs must be unique.")
    if not callable(predictor):
        raise TypeError("predictor must be callable.")
    if score_column in pairs.columns:
        raise ValueError(f"pairs already contains score column {score_column!r}.")
    if batch_size is not None and (
        not isinstance(batch_size, int) or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer or None.")

    if batch_size is None:
        scores = _coerce_prediction_output(
            predictor(pairs.copy(deep=True)),
            pairs,
            score_column,
        )
    else:
        chunks: list[np.ndarray] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs.iloc[start : start + batch_size].copy(deep=True)
            chunks.append(
                _coerce_prediction_output(
                    predictor(batch),
                    batch,
                    score_column,
                )
            )
        scores = np.concatenate(chunks)

    predictions = pairs.copy(deep=True)
    predictions[score_column] = scores
    if model_name is not None:
        predictions["screening_model"] = str(model_name)
    predictions.attrs["screening_metadata"] = screening_wording_metadata()
    return predictions


def rank_screening_predictions(
    predictions: pd.DataFrame,
    *,
    score_column: str = _DEFAULT_SCORE_COLUMN,
    higher_is_better: bool = True,
    model_name: str | None = None,
) -> ScreeningRankingResult:
    """Rank one model's scores and create top-10/top-20/top-5 outputs."""

    _require_dataframe(predictions, "predictions")
    required = ["pair_id", "drug_id", "target_id", score_column]
    _require_columns(predictions, required, "predictions")
    if predictions["pair_id"].duplicated().any():
        raise ValueError("Prediction pair IDs must be unique.")
    numeric_scores = pd.to_numeric(predictions[score_column], errors="coerce")
    if numeric_scores.isna().any() or not np.isfinite(numeric_scores).all():
        raise ValueError("Prediction scores must be finite numeric values.")

    ranked = predictions.copy(deep=True)
    ranked[score_column] = numeric_scores.astype(float)
    ranked["_stable_pair_id"] = ranked["pair_id"].astype(str)
    ranked = ranked.sort_values(
        [score_column, "_stable_pair_id"],
        ascending=[not higher_is_better, True],
        kind="mergesort",
    ).drop(columns="_stable_pair_id")
    ranked = ranked.reset_index(drop=True)
    ranked["overall_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    ranked["target_rank"] = (
        ranked.groupby("target_id", sort=False).cumcount() + 1
    ).astype(int)

    wording = screening_wording_metadata()
    metadata: dict[str, Any] = {
        **wording,
        "model_name": model_name,
        "score_column": score_column,
        "ranking_direction": (
            "higher score ranks first"
            if higher_is_better
            else "lower score ranks first"
        ),
        "number_of_pairs": int(len(ranked)),
        "number_of_drugs": int(ranked["drug_id"].nunique()),
        "number_of_targets": int(ranked["target_id"].nunique()),
    }
    outputs = [
        ranked,
        ranked.head(10).copy(),
        ranked.head(20).copy(),
        ranked.loc[ranked["target_rank"] <= 5].copy(),
    ]
    for output in outputs:
        output.attrs["screening_metadata"] = metadata

    return ScreeningRankingResult(
        ranked=outputs[0],
        top_10_overall=outputs[1],
        top_20_overall=outputs[2],
        top_5_per_target=outputs[3],
        metadata=metadata,
    )


def _best_drug_rows(
    ranked: pd.DataFrame,
    *,
    top_n: int,
    score_column: str,
    prefix: str,
) -> pd.DataFrame:
    """Return the best overall occurrence of each drug in a top-pair list."""

    columns = ["drug_id", "overall_rank", score_column]
    if "drug_name" in ranked.columns:
        columns.insert(1, "drug_name")
    result = (
        ranked.loc[ranked["overall_rank"] <= top_n, columns]
        .sort_values("overall_rank", kind="mergesort")
        .drop_duplicates("drug_id", keep="first")
    )
    rename = {
        "overall_rank": f"{prefix}_best_overall_rank",
        score_column: f"{prefix}_score",
    }
    if "drug_name" in result.columns:
        rename["drug_name"] = f"{prefix}_drug_name"
    return result.rename(columns=rename)


def compare_screening_models(
    baseline_predictions: pd.DataFrame,
    advanced_predictions: pd.DataFrame,
    *,
    baseline_score_column: str = _DEFAULT_SCORE_COLUMN,
    advanced_score_column: str = _DEFAULT_SCORE_COLUMN,
    higher_is_better: bool = True,
    top_n: int = 20,
    per_target_n: int = 5,
) -> ScreeningComparisonResult:
    """Compare baseline and advanced AML rankings without external validation."""

    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    if not isinstance(per_target_n, int) or per_target_n < 1:
        raise ValueError("per_target_n must be a positive integer.")

    baseline_provenance = _prediction_checkpoint_provenance(
        baseline_predictions,
        "baseline_predictions",
    )
    advanced_provenance = _prediction_checkpoint_provenance(
        advanced_predictions,
        "advanced_predictions",
    )
    if (baseline_provenance is None) != (advanced_provenance is None):
        raise ValueError(
            "Checkpoint provenance must be present on both model outputs or neither."
        )
    if baseline_provenance is not None and advanced_provenance is not None:
        if baseline_provenance["checkpoint_model_family"] != "baseline":
            raise ValueError("Baseline predictions have non-baseline provenance.")
        if advanced_provenance["checkpoint_model_family"] != "advanced":
            raise ValueError("Advanced predictions have non-advanced provenance.")
        if (
            baseline_provenance["training_dataset"]
            != advanced_provenance["training_dataset"]
        ):
            raise ValueError(
                "Baseline and advanced predictions were produced on different "
                "training datasets."
            )
        if (
            baseline_provenance["affinity_label"]
            != advanced_provenance["affinity_label"]
        ):
            raise ValueError(
                "Baseline and advanced predictions use different affinity labels."
            )
        for field in (
            "checkpoint_training_split",
            "checkpoint_seed",
            "checkpoint_max_protein_length",
            "checkpoint_data_fingerprint",
            "screening_pairs_sha256",
            "screening_pair_count",
            "screening_drug_count",
            "screening_target_count",
        ):
            if baseline_provenance[field] != advanced_provenance[field]:
                raise ValueError(
                    "Baseline and advanced prediction provenance differs for "
                    f"{field!r}."
                )
        if (
            baseline_provenance["checkpoint_split_fingerprint"]
            != advanced_provenance["checkpoint_split_fingerprint"]
        ):
            raise ValueError(
                "Baseline and advanced predictions use different split fingerprints."
            )

    baseline = rank_screening_predictions(
        baseline_predictions,
        score_column=baseline_score_column,
        higher_is_better=higher_is_better,
        model_name="baseline",
    ).ranked
    advanced = rank_screening_predictions(
        advanced_predictions,
        score_column=advanced_score_column,
        higher_is_better=higher_is_better,
        model_name="advanced",
    ).ranked

    baseline_ids = set(baseline["pair_id"].astype(str))
    advanced_ids = set(advanced["pair_id"].astype(str))
    if baseline_ids != advanced_ids:
        raise ValueError(
            "Baseline and advanced predictions must contain identical pair IDs."
        )

    baseline_identity = (
        baseline.assign(pair_id=baseline["pair_id"].astype(str))
        .set_index("pair_id")[["drug_id", "target_id"]]
        .astype(str)
        .sort_index()
    )
    advanced_identity = (
        advanced.assign(pair_id=advanced["pair_id"].astype(str))
        .set_index("pair_id")[["drug_id", "target_id"]]
        .astype(str)
        .sort_index()
    )
    if not baseline_identity.equals(advanced_identity):
        raise ValueError(
            "Pair IDs must map to the same drug and target in both models."
        )

    identity_columns = ["pair_id", "drug_id", "target_id"]
    if "drug_name" in baseline.columns:
        identity_columns.append("drug_name")
    baseline_view = baseline[
        identity_columns
        + [baseline_score_column, "overall_rank", "target_rank"]
    ].rename(
        columns={
            baseline_score_column: "baseline_score",
            "overall_rank": "baseline_overall_rank",
            "target_rank": "baseline_target_rank",
        }
    )
    advanced_view = advanced[
        [
            "pair_id",
            "drug_id",
            "target_id",
            advanced_score_column,
            "overall_rank",
            "target_rank",
        ]
    ].rename(
        columns={
            advanced_score_column: "advanced_score",
            "overall_rank": "advanced_overall_rank",
            "target_rank": "advanced_target_rank",
        }
    )
    pair_comparison = baseline_view.merge(
        advanced_view,
        on=["pair_id", "drug_id", "target_id"],
        how="inner",
        validate="one_to_one",
    )
    pair_comparison["overall_rank_difference"] = (
        pair_comparison["advanced_overall_rank"]
        - pair_comparison["baseline_overall_rank"]
    )
    pair_comparison["target_rank_difference"] = (
        pair_comparison["advanced_target_rank"]
        - pair_comparison["baseline_target_rank"]
    )
    pair_comparison["score_difference"] = (
        pair_comparison["advanced_score"]
        - pair_comparison["baseline_score"]
    )

    baseline_top_pair_ids = set(
        baseline.loc[baseline["overall_rank"] <= top_n, "pair_id"].astype(str)
    )
    advanced_top_pair_ids = set(
        advanced.loc[advanced["overall_rank"] <= top_n, "pair_id"].astype(str)
    )
    shared_pair_ids = baseline_top_pair_ids.intersection(advanced_top_pair_ids)
    top_pair_overlap = pair_comparison.loc[
        pair_comparison["pair_id"].astype(str).isin(shared_pair_ids)
    ].sort_values(
        ["baseline_overall_rank", "advanced_overall_rank"],
        kind="mergesort",
    )

    baseline_drugs = _best_drug_rows(
        baseline,
        top_n=top_n,
        score_column=baseline_score_column,
        prefix="baseline",
    )
    advanced_drugs = _best_drug_rows(
        advanced,
        top_n=top_n,
        score_column=advanced_score_column,
        prefix="advanced",
    )
    shared_top_drugs = baseline_drugs.merge(
        advanced_drugs,
        on="drug_id",
        how="inner",
        validate="one_to_one",
    )
    shared_top_drugs["best_rank_difference"] = (
        shared_top_drugs["advanced_best_overall_rank"]
        - shared_top_drugs["baseline_best_overall_rank"]
    )
    shared_top_drugs = shared_top_drugs.sort_values(
        ["baseline_best_overall_rank", "advanced_best_overall_rank"],
        kind="mergesort",
    ).reset_index(drop=True)

    target_rows: list[dict[str, Any]] = []
    target_ids = sorted(
        set(baseline["target_id"].astype(str)).union(
            advanced["target_id"].astype(str)
        )
    )
    for target_id in target_ids:
        baseline_target = baseline.loc[
            (baseline["target_id"].astype(str) == target_id)
            & (baseline["target_rank"] <= per_target_n)
        ].sort_values("target_rank")
        advanced_target = advanced.loc[
            (advanced["target_id"].astype(str) == target_id)
            & (advanced["target_rank"] <= per_target_n)
        ].sort_values("target_rank")
        baseline_set = set(baseline_target["drug_id"].astype(str))
        advanced_set = set(advanced_target["drug_id"].astype(str))
        overlap = sorted(baseline_set.intersection(advanced_set))
        union = baseline_set.union(advanced_set)
        baseline_top = (
            str(baseline_target.iloc[0]["drug_id"])
            if not baseline_target.empty
            else None
        )
        advanced_top = (
            str(advanced_target.iloc[0]["drug_id"])
            if not advanced_target.empty
            else None
        )
        target_rows.append(
            {
                "target_id": target_id,
                "baseline_top_drug_id": baseline_top,
                "advanced_top_drug_id": advanced_top,
                "same_top_candidate": baseline_top == advanced_top,
                "top_drug_overlap_count": len(overlap),
                "top_drug_union_count": len(union),
                "top_drug_jaccard": len(overlap) / len(union) if union else 1.0,
                "overlapping_drug_ids": "|".join(overlap),
            }
        )
    target_agreement = pd.DataFrame(target_rows)

    pair_union = baseline_top_pair_ids.union(advanced_top_pair_ids)
    baseline_top_drug_ids = set(baseline_drugs["drug_id"].astype(str))
    advanced_top_drug_ids = set(advanced_drugs["drug_id"].astype(str))
    drug_union = baseline_top_drug_ids.union(advanced_top_drug_ids)
    wording = screening_wording_metadata()
    summary: dict[str, Any] = {
        **wording,
        "top_n_pairs_compared": top_n,
        "per_target_n_compared": per_target_n,
        "shared_top_pair_count": len(shared_pair_ids),
        "top_pair_jaccard": (
            len(shared_pair_ids) / len(pair_union) if pair_union else 1.0
        ),
        "shared_top_drug_count": len(
            baseline_top_drug_ids.intersection(advanced_top_drug_ids)
        ),
        "top_drug_jaccard": (
            len(baseline_top_drug_ids.intersection(advanced_top_drug_ids))
            / len(drug_union)
            if drug_union
            else 1.0
        ),
        "targets_with_same_top_candidate": int(
            target_agreement["same_top_candidate"].sum()
        ),
        "number_of_targets": int(len(target_agreement)),
    }
    for output in [
        pair_comparison,
        shared_top_drugs,
        top_pair_overlap,
        target_agreement,
    ]:
        output.attrs["screening_metadata"] = summary

    return ScreeningComparisonResult(
        pair_comparison=pair_comparison.reset_index(drop=True),
        shared_top_drugs=shared_top_drugs,
        top_pair_overlap=top_pair_overlap.reset_index(drop=True),
        target_agreement=target_agreement,
        summary=summary,
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_copy(value: Any, name: str) -> Any:
    """Return an immutable-safe JSON round trip or reject unsafe metadata."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Checkpoint {name} must be strict JSON data.") from error


def _required_mapping_value(
    mapping: Mapping[str, Any],
    key: str,
    mapping_name: str,
) -> Any:
    if key not in mapping:
        raise ValueError(f"Checkpoint {mapping_name} is missing {key!r}.")
    return mapping[key]


def _positive_int(value: Any, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"Checkpoint {name} must be a positive integer.")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError(f"Checkpoint {name} must be a non-negative integer.")
    return int(value)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Checkpoint {name} must be finite and numeric.")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Checkpoint {name} must be finite and numeric."
        ) from error
    if not np.isfinite(converted):
        raise ValueError(f"Checkpoint {name} must be finite and numeric.")
    return converted


def _validate_sha256(value: Any, name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"Checkpoint {name} must be a SHA-256 hex digest.")
    return digest


def _reconstruct_checkpoint_model(
    configuration: Mapping[str, Any],
) -> tuple[nn.Module, str, str]:
    """Reconstruct only the two supported standard model families."""

    model_family = str(
        _required_mapping_value(configuration, "model_family", "configuration")
    ).strip().lower()
    model_name = str(
        _required_mapping_value(configuration, "model", "configuration")
    ).strip().lower()
    custom_model = _required_mapping_value(
        configuration,
        "custom_model_supplied",
        "configuration",
    )
    if custom_model is not False:
        raise ValueError(
            "Screening cannot reconstruct a checkpoint trained with a custom model."
        )

    dropout = _finite_float(
        _required_mapping_value(configuration, "dropout", "configuration"),
        "configuration.dropout",
    )
    if not 0.0 <= dropout < 1.0:
        raise ValueError("Checkpoint configuration.dropout must be in [0, 1).")
    atom_feature_dim = _positive_int(
        _required_mapping_value(
            configuration,
            "atom_feature_dim",
            "configuration",
        ),
        "configuration.atom_feature_dim",
    )
    bond_feature_dim = _positive_int(
        _required_mapping_value(
            configuration,
            "bond_feature_dim",
            "configuration",
        ),
        "configuration.bond_feature_dim",
    )
    if atom_feature_dim != NODE_FEATURE_DIM or bond_feature_dim != BOND_FEATURE_DIM:
        raise ValueError(
            "Checkpoint graph feature dimensions do not match this source version."
        )

    architecture = _required_mapping_value(
        configuration,
        "architecture",
        "configuration",
    )
    if not isinstance(architecture, Mapping):
        raise ValueError("Checkpoint configuration.architecture must be a mapping.")

    if model_family == "baseline":
        if model_name not in {"gcn", "gat", "gin"}:
            raise ValueError(
                "Baseline screening checkpoints must identify GCN, GAT, or GIN."
            )
        expected_dimensions = {
            "drug_hidden_dim": DRUG_HIDDEN_DIM,
            "protein_embedding_dim": PROTEIN_EMBEDDING_DIM,
            "protein_hidden_dim": PROTEIN_HIDDEN_DIM,
            "fusion_hidden_dim": FUSION_HIDDEN_DIM,
        }
        for key, expected in expected_dimensions.items():
            observed = _positive_int(
                _required_mapping_value(
                    architecture,
                    key,
                    "configuration.architecture",
                ),
                f"configuration.architecture.{key}",
            )
            if observed != expected:
                raise ValueError(
                    f"Checkpoint {key}={observed} does not match source "
                    f"dimension {expected}."
                )
        return create_dta_model(model_name, dropout=dropout), model_family, model_name

    if model_family != "advanced" or model_name != "advanced":
        raise ValueError(
            "Checkpoint model_family/model must be baseline + GCN/GAT/GIN "
            "or advanced + advanced."
        )

    gine_layers = _positive_int(
        _required_mapping_value(
            architecture,
            "gine_layers",
            "configuration.architecture",
        ),
        "configuration.architecture.gine_layers",
    )
    protein_layers = _positive_int(
        _required_mapping_value(
            architecture,
            "protein_cnn_layers",
            "configuration.architecture",
        ),
        "configuration.architecture.protein_cnn_layers",
    )
    if gine_layers != ADVANCED_GINE_LAYERS:
        raise ValueError("Advanced checkpoint does not use the required GINE depth.")
    if protein_layers != ADVANCED_PROTEIN_CNN_LAYERS:
        raise ValueError("Advanced checkpoint does not use the required CNN depth.")

    esm2_model_name = str(architecture.get("esm2_model_name") or "").strip()
    cache_provenance = architecture.get("residue_cache_provenance")
    if not esm2_model_name:
        raise ValueError(
            "Advanced checkpoint lacks verified ESM-2 model provenance."
        )
    if not isinstance(cache_provenance, Mapping) or (
        cache_provenance.get("provenance_verified") is not True
    ):
        raise ValueError(
            "Advanced checkpoint lacks verified residue-cache provenance."
        )
    if cache_provenance.get("schema_version") != 2:
        raise ValueError(
            "Advanced checkpoint requires residue-cache provenance schema 2."
        )
    if cache_provenance.get("artifact_integrity_verified") is not True:
        raise ValueError(
            "Advanced checkpoint residue-cache artifact integrity is unverified."
        )
    if str(cache_provenance.get("model_name", "")).strip() != esm2_model_name:
        raise ValueError(
            "Advanced checkpoint residue-cache and ESM-2 model provenance differ."
        )
    if not str(cache_provenance.get("resolved_model_revision") or "").strip():
        raise ValueError(
            "Advanced checkpoint lacks a resolved ESM-2 model revision."
        )

    residue_embedding_dimension = _positive_int(
        _required_mapping_value(
            architecture,
            "residue_embedding_dim_used",
            "configuration.architecture",
        ),
        "configuration.architecture.residue_embedding_dim_used",
    )
    maximum_protein_length = _positive_int(
        _required_mapping_value(
            configuration,
            "max_protein_length",
            "configuration",
        ),
        "configuration.max_protein_length",
    )
    if (
        _positive_int(
            cache_provenance.get("embedding_dimension"),
            "configuration.architecture.residue_cache_provenance."
            "embedding_dimension",
        )
        != residue_embedding_dimension
    ):
        raise ValueError(
            "Advanced checkpoint residue-cache embedding dimension is inconsistent."
        )
    if (
        _positive_int(
            cache_provenance.get("max_length"),
            "configuration.architecture.residue_cache_provenance.max_length",
        )
        != maximum_protein_length
    ):
        raise ValueError(
            "Advanced checkpoint residue-cache max length is inconsistent."
        )
    sequence_hashes = cache_provenance.get("target_sequence_sha256")
    if not isinstance(sequence_hashes, Mapping) or not sequence_hashes:
        raise ValueError(
            "Advanced checkpoint residue-cache target sequence hashes are missing."
        )
    for target_id, sequence_digest in sequence_hashes.items():
        if not str(target_id).strip():
            raise ValueError(
                "Advanced checkpoint residue-cache target IDs must be non-empty."
            )
        _validate_sha256(
            sequence_digest,
            "configuration.architecture.residue_cache_provenance."
            f"target_sequence_sha256[{target_id!r}]",
        )
    tensor_file = str(cache_provenance.get("tensor_file") or "").strip()
    if not tensor_file:
        raise ValueError(
            "Advanced checkpoint residue-cache tensor filename is missing."
        )
    _positive_int(
        cache_provenance.get("tensor_file_bytes"),
        "configuration.architecture.residue_cache_provenance.tensor_file_bytes",
    )
    _validate_sha256(
        cache_provenance.get("tensor_file_sha256"),
        "configuration.architecture.residue_cache_provenance.tensor_file_sha256",
    )

    advanced_arguments = {
        "atom_input_dim": atom_feature_dim,
        "bond_feature_dim": bond_feature_dim,
        "esm_embedding_dim": residue_embedding_dimension,
        "embedding_dim": _positive_int(
            _required_mapping_value(
                architecture,
                "hidden_dim",
                "configuration.architecture",
            ),
            "configuration.architecture.hidden_dim",
        ),
        "attention_heads": _positive_int(
            _required_mapping_value(
                architecture,
                "attention_heads",
                "configuration.architecture",
            ),
            "configuration.architecture.attention_heads",
        ),
        "attention_feedforward_dim": _positive_int(
            _required_mapping_value(
                architecture,
                "attention_feedforward_dim",
                "configuration.architecture",
            ),
            "configuration.architecture.attention_feedforward_dim",
        ),
        "regression_hidden_dim": _positive_int(
            _required_mapping_value(
                architecture,
                "regression_hidden_dim",
                "configuration.architecture",
            ),
            "configuration.architecture.regression_hidden_dim",
        ),
        "regression_bottleneck_dim": _positive_int(
            _required_mapping_value(
                architecture,
                "regression_bottleneck_dim",
                "configuration.architecture",
            ),
            "configuration.architecture.regression_bottleneck_dim",
        ),
        "dropout": dropout,
    }
    return (
        create_advanced_dta_model(**advanced_arguments),
        model_family,
        model_name,
    )


def load_screening_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_model_family: str | None = None,
    expected_training_dataset: str | None = None,
    expected_affinity_label: str | None = None,
    device: str | torch.device | None = None,
) -> LoadedScreeningCheckpoint:
    """Strictly validate provenance and reconstruct an inference checkpoint.

    Only v2 checkpoints from the standard experiment runners are accepted.
    PyTorch's restricted ``weights_only`` loader is used; unsafe pickle
    fallback and custom-model reconstruction are deliberately refused.
    """

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Screening checkpoint not found: {path}")
    if not path.is_file():
        raise ValueError(f"Screening checkpoint is not a file: {path}")
    selected_device = (
        torch.device(device) if device is not None else get_device()
    )
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ValueError(
            "Checkpoint could not be loaded with PyTorch's restricted "
            "weights-only loader."
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint root must be a mapping.")
    state_dict = _required_mapping_value(
        checkpoint,
        "model_state_dict",
        "root",
    )
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Checkpoint model_state_dict must be a non-empty mapping.")
    metadata = _required_mapping_value(checkpoint, "metadata", "root")
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint metadata must be a mapping.")

    raw_configuration = _required_mapping_value(
        metadata,
        "configuration_used",
        "metadata",
    )
    if not isinstance(raw_configuration, Mapping):
        raise ValueError("Checkpoint configuration_used must be a mapping.")
    configuration = _strict_json_copy(
        dict(raw_configuration),
        "configuration_used",
    )
    configuration_json = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    configuration_digest = sha256(configuration_json.encode("utf-8")).hexdigest()

    training_dataset = str(
        _required_mapping_value(configuration, "dataset", "configuration")
    ).strip().lower()
    if training_dataset not in DATASET_AFFINITY_LABELS:
        raise ValueError("Checkpoint training dataset must be Davis or KIBA.")
    training_split = str(
        _required_mapping_value(configuration, "split", "configuration")
    ).strip().lower().replace("-", "_")
    if training_split not in {"random", "cold_drug", "cold_target"}:
        raise ValueError("Checkpoint training split is unsupported.")
    training_seed = _required_mapping_value(
        configuration,
        "seed",
        "configuration",
    )
    if (
        isinstance(training_seed, bool)
        or not isinstance(training_seed, int)
        or training_seed < 0
    ):
        raise ValueError("Checkpoint training seed must be a non-negative integer.")
    _positive_int(
        _required_mapping_value(
            configuration,
            "max_protein_length",
            "configuration",
        ),
        "configuration.max_protein_length",
    )
    affinity_label = str(
        _required_mapping_value(metadata, "affinity_label", "metadata")
    ).strip()
    configured_label = str(
        _required_mapping_value(
            configuration,
            "affinity_label",
            "configuration",
        )
    ).strip()
    expected_dataset_label = DATASET_AFFINITY_LABELS[training_dataset]
    if affinity_label != configured_label or affinity_label != expected_dataset_label:
        raise ValueError(
            "Checkpoint dataset and affinity-label provenance are inconsistent."
        )
    if configuration.get("checkpoint_selection_metric") != "validation_mse":
        raise ValueError(
            "Checkpoint must record validation_mse as its selection metric."
        )
    if (
        configuration.get("test_evaluation_policy")
        != "after_best_validation_checkpoint"
    ):
        raise ValueError("Checkpoint test-evaluation policy is missing or unsafe.")

    data_fingerprint = _validate_sha256(
        _required_mapping_value(metadata, "data_fingerprint", "metadata"),
        "metadata.data_fingerprint",
    )
    data_provenance = _required_mapping_value(
        metadata,
        "data_provenance",
        "metadata",
    )
    configured_data_provenance = configuration.get("data_provenance")
    if not isinstance(data_provenance, Mapping) or not data_provenance:
        raise ValueError("Checkpoint data_provenance must be a non-empty mapping.")
    if not isinstance(configured_data_provenance, Mapping):
        raise ValueError("Checkpoint configuration lacks data_provenance.")
    data_provenance_copy = _strict_json_copy(
        dict(data_provenance),
        "data_provenance",
    )
    if data_provenance_copy != _strict_json_copy(
        dict(configured_data_provenance),
        "configuration.data_provenance",
    ):
        raise ValueError(
            "Checkpoint metadata/configuration data provenance are inconsistent."
        )
    if str(data_provenance_copy.get("data_sha256", "")).lower() != data_fingerprint:
        raise ValueError("Checkpoint data fingerprint is internally inconsistent.")

    runtime_provenance = _required_mapping_value(
        metadata,
        "runtime_provenance",
        "metadata",
    )
    configured_runtime_provenance = configuration.get("runtime_provenance")
    if not isinstance(runtime_provenance, Mapping) or not runtime_provenance:
        raise ValueError("Checkpoint runtime_provenance must be non-empty.")
    if not isinstance(configured_runtime_provenance, Mapping):
        raise ValueError("Checkpoint configuration lacks runtime_provenance.")
    runtime_provenance_copy = _strict_json_copy(
        dict(runtime_provenance),
        "runtime_provenance",
    )
    if runtime_provenance_copy != _strict_json_copy(
        dict(configured_runtime_provenance),
        "configuration.runtime_provenance",
    ):
        raise ValueError(
            "Checkpoint metadata/configuration runtime provenance are inconsistent."
        )

    split_fingerprint = _validate_sha256(
        _required_mapping_value(metadata, "split_fingerprint", "metadata"),
        "metadata.split_fingerprint",
    )
    split_manifest = _required_mapping_value(
        metadata,
        "split_manifest",
        "metadata",
    )
    if not isinstance(split_manifest, Mapping):
        raise ValueError("Checkpoint split_manifest must be a mapping.")
    split_manifest_copy = _strict_json_copy(
        dict(split_manifest),
        "split_manifest",
    )
    for key in ("mode", "path", "metadata_path", "sha256"):
        if key not in split_manifest_copy:
            raise ValueError(f"Checkpoint split_manifest is missing {key!r}.")

    epoch = _positive_int(
        _required_mapping_value(checkpoint, "epoch", "root"),
        "epoch",
    )
    validation_metrics = _required_mapping_value(
        checkpoint,
        "validation_metrics",
        "root",
    )
    if not isinstance(validation_metrics, Mapping):
        raise ValueError("Checkpoint validation_metrics must be a mapping.")
    best_validation_mse = _finite_float(
        _required_mapping_value(
            validation_metrics,
            "mse",
            "validation_metrics",
        ),
        "validation_metrics.mse",
    )
    if best_validation_mse < 0:
        raise ValueError("Checkpoint validation MSE cannot be negative.")

    model, model_family, model_name = _reconstruct_checkpoint_model(configuration)
    if expected_model_family is not None and model_family != str(
        expected_model_family
    ).strip().lower():
        raise ValueError(
            f"Expected {expected_model_family!r} checkpoint, found {model_family!r}."
        )
    if expected_training_dataset is not None and training_dataset != str(
        expected_training_dataset
    ).strip().lower():
        raise ValueError(
            "Checkpoint training dataset does not match the requested dataset."
        )
    if expected_affinity_label is not None and affinity_label != str(
        expected_affinity_label
    ).strip():
        raise ValueError(
            "Checkpoint affinity label does not match the requested semantics."
        )

    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            "Checkpoint weights do not match the reconstructed model/configuration."
        ) from error
    model.to(selected_device)
    model.eval()

    return LoadedScreeningCheckpoint(
        model=model,
        path=path,
        sha256=_file_sha256(path),
        configuration_sha256=configuration_digest,
        model_family=model_family,
        model_name=model_name,
        training_dataset=training_dataset,
        affinity_label=affinity_label,
        configuration=configuration,
        data_fingerprint=data_fingerprint,
        data_provenance=data_provenance_copy,
        runtime_provenance=runtime_provenance_copy,
        split_fingerprint=split_fingerprint,
        split_manifest=split_manifest_copy,
        best_epoch=epoch,
        best_validation_mse=best_validation_mse,
        device=selected_device,
    )


def _coerce_loaded_checkpoint(
    checkpoint: str | Path | LoadedScreeningCheckpoint,
    *,
    expected_model_family: str,
    device: str | torch.device | None,
) -> LoadedScreeningCheckpoint:
    if isinstance(checkpoint, LoadedScreeningCheckpoint):
        if checkpoint.model_family != expected_model_family:
            raise ValueError(
                f"Expected a {expected_model_family} screening checkpoint, "
                f"found {checkpoint.model_family}."
            )
        if device is not None and torch.device(device) != checkpoint.device:
            raise ValueError(
                "A loaded screening checkpoint is already bound to another "
                "device; reload it from its path on the requested device."
            )
        return checkpoint
    return load_screening_checkpoint(
        checkpoint,
        expected_model_family=expected_model_family,
        device=device,
    )


def validate_screening_checkpoint_compatibility(
    baseline: LoadedScreeningCheckpoint,
    advanced: LoadedScreeningCheckpoint,
) -> None:
    """Fail closed unless two checkpoints predict the same affinity semantics."""

    if not isinstance(baseline, LoadedScreeningCheckpoint) or not isinstance(
        advanced,
        LoadedScreeningCheckpoint,
    ):
        raise TypeError("Both values must be loaded screening checkpoints.")
    if baseline.model_family != "baseline":
        raise ValueError("The first checkpoint must be a baseline model.")
    if advanced.model_family != "advanced":
        raise ValueError("The second checkpoint must be an advanced model.")
    if baseline.training_dataset != advanced.training_dataset:
        raise ValueError(
            "Baseline and advanced checkpoints were trained on different "
            "datasets; their affinity scores cannot be compared."
        )
    if baseline.affinity_label != advanced.affinity_label:
        raise ValueError(
            "Baseline and advanced checkpoints use different affinity labels."
        )
    if baseline.data_fingerprint != advanced.data_fingerprint:
        raise ValueError(
            "Baseline and advanced checkpoints use different training-data "
            "fingerprints."
        )
    required_configuration_matches = (
        "split",
        "seed",
        "max_protein_length",
    )
    for field in required_configuration_matches:
        baseline_value = _required_mapping_value(
            baseline.configuration,
            field,
            "baseline configuration",
        )
        advanced_value = _required_mapping_value(
            advanced.configuration,
            field,
            "advanced configuration",
        )
        if baseline_value != advanced_value:
            raise ValueError(
                "Baseline and advanced checkpoint configurations differ for "
                f"{field!r}."
            )
    if baseline.split_fingerprint != advanced.split_fingerprint:
        raise ValueError(
            "Baseline and advanced checkpoints use different split fingerprints."
        )


def _validate_screening_pair_inputs(pairs: pd.DataFrame) -> None:
    _require_dataframe(pairs, "pairs")
    _require_columns(
        pairs,
        [
            "pair_id",
            "drug_id",
            "target_id",
            "smiles",
            "protein_sequence",
        ],
        "pairs",
    )
    _validate_nonempty_strings(
        pairs,
        [
            "pair_id",
            "drug_id",
            "target_id",
            "smiles",
            "protein_sequence",
        ],
        "pairs",
    )
    if pairs["pair_id"].duplicated().any():
        raise ValueError("Screening pair IDs must be unique.")


def _validate_inference_settings(
    batch_size: int,
    num_workers: int,
) -> None:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer.")
    if (
        isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 0
    ):
        raise ValueError("num_workers must be a non-negative integer.")


def _checkpoint_prediction_values(
    checkpoint: LoadedScreeningCheckpoint,
    dataset: DTADataset | AdvancedDTADataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
    )
    _, predicted_values, sample_indices = predict(
        model=checkpoint.model,
        data_loader=loader,
        device=checkpoint.device,
    )
    indices = np.asarray(sample_indices, dtype=int).reshape(-1)
    predictions = np.asarray(predicted_values, dtype=float).reshape(-1)
    expected_indices = np.arange(len(dataset), dtype=int)
    if len(indices) != len(expected_indices) or len(predictions) != len(
        expected_indices
    ):
        raise RuntimeError("Checkpoint inference returned an invalid row count.")
    if len(np.unique(indices)) != len(indices) or not np.array_equal(
        np.sort(indices),
        expected_indices,
    ):
        raise RuntimeError("Checkpoint inference returned invalid sample indices.")
    if not np.isfinite(predictions).all():
        raise ValueError("Checkpoint inference produced non-finite scores.")
    ordered = np.empty_like(predictions)
    ordered[indices] = predictions
    return ordered


def screening_input_provenance(pairs: pd.DataFrame) -> dict[str, Any]:
    """Fingerprint the ordered molecular/protein inputs used for screening."""

    _validate_screening_pair_inputs(pairs)
    records: list[list[str]] = []
    for row in pairs.itertuples(index=False):
        row_values = row._asdict()
        input_smiles = str(
            row_values.get("canonical_smiles", row_values["smiles"])
        )
        molecule = _mol_from_smiles_without_rdkit_stderr(input_smiles)
        if molecule is None or molecule.GetNumAtoms() < 1:
            raise ValueError("Screening inputs contain an invalid molecular structure.")
        canonical_smiles = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
        records.append(
            [
                str(row_values["pair_id"]),
                str(row_values["drug_id"]),
                canonical_smiles,
                str(row_values["target_id"]),
                str(row_values["protein_sequence"]),
            ]
        )
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "screening_pairs_sha256": sha256(encoded).hexdigest(),
        "screening_pair_count": int(len(pairs)),
        "screening_drug_count": int(pairs["drug_id"].astype(str).nunique()),
        "screening_target_count": int(pairs["target_id"].astype(str).nunique()),
    }


def _attach_checkpoint_provenance(
    pairs: pd.DataFrame,
    scores: np.ndarray,
    checkpoint: LoadedScreeningCheckpoint,
    *,
    score_column: str,
) -> pd.DataFrame:
    if score_column in pairs.columns:
        raise ValueError(f"pairs already contains score column {score_column!r}.")
    result = pairs.copy(deep=True)
    result[score_column] = np.asarray(scores, dtype=float)
    input_provenance = screening_input_provenance(pairs)
    columns: dict[str, Any] = {
        "training_dataset": checkpoint.training_dataset,
        "affinity_label": checkpoint.affinity_label,
        "checkpoint_model_family": checkpoint.model_family,
        "checkpoint_model_name": checkpoint.model_name,
        "checkpoint_run_name": str(
            checkpoint.configuration.get("run_name")
            or (
                f"{checkpoint.training_dataset}_{checkpoint.model_name}_"
                f"{checkpoint.configuration['split']}_"
                f"seed{checkpoint.configuration['seed']}"
            )
        ),
        "checkpoint_sha256": checkpoint.sha256,
        "checkpoint_configuration_sha256": checkpoint.configuration_sha256,
        "checkpoint_data_fingerprint": checkpoint.data_fingerprint,
        "checkpoint_training_split": checkpoint.configuration["split"],
        "checkpoint_seed": checkpoint.configuration["seed"],
        "checkpoint_max_protein_length": checkpoint.configuration[
            "max_protein_length"
        ],
        "checkpoint_split_fingerprint": checkpoint.split_fingerprint,
        "checkpoint_best_epoch": checkpoint.best_epoch,
        "checkpoint_best_validation_mse": checkpoint.best_validation_mse,
        **input_provenance,
    }
    for name, value in columns.items():
        result[name] = value
    result.attrs["checkpoint_provenance"] = checkpoint.provenance
    result.attrs["screening_input_provenance"] = input_provenance
    result.attrs["screening_metadata"] = screening_wording_metadata()
    return result


def predict_baseline_checkpoint(
    pairs: pd.DataFrame,
    checkpoint: str | Path | LoadedScreeningCheckpoint,
    *,
    drug_graph_cache: Mapping[str, Any] | None = None,
    protein_tensor_cache: Mapping[str, torch.Tensor] | None = None,
    score_column: str = _DEFAULT_SCORE_COLUMN,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    """Predict AML pairs directly from a standard GCN/GAT/GIN checkpoint."""

    _validate_screening_pair_inputs(pairs)
    _validate_inference_settings(batch_size, num_workers)
    loaded = _coerce_loaded_checkpoint(
        checkpoint,
        expected_model_family="baseline",
        device=device,
    )
    maximum_length = _positive_int(
        _required_mapping_value(
            loaded.configuration,
            "max_protein_length",
            "configuration",
        ),
        "configuration.max_protein_length",
    )
    graph_cache = (
        drug_graph_cache
        if drug_graph_cache is not None
        else build_drug_graph_cache(pairs)
    )
    token_cache = (
        protein_tensor_cache
        if protein_tensor_cache is not None
        else build_protein_tensor_cache(pairs, max_length=maximum_length)
    )
    inference_pairs = pairs.copy(deep=True)
    inference_pairs["_screening_dummy_affinity"] = 0.0
    dataset = DTADataset(
        inference_pairs,
        dict(graph_cache),
        dict(token_cache),
        affinity_column="_screening_dummy_affinity",
    )
    scores = _checkpoint_prediction_values(
        loaded,
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return _attach_checkpoint_provenance(
        pairs,
        scores,
        loaded,
        score_column=score_column,
    )


def predict_advanced_checkpoint(
    pairs: pd.DataFrame,
    checkpoint: str | Path | LoadedScreeningCheckpoint,
    residue_embedding_cache: Mapping[str, torch.Tensor],
    *,
    drug_graph_cache: Mapping[str, Any] | None = None,
    score_column: str = _DEFAULT_SCORE_COLUMN,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    """Predict AML pairs using cached, provenance-verified ESM-2 residues."""

    _validate_screening_pair_inputs(pairs)
    _validate_inference_settings(batch_size, num_workers)
    loaded = _coerce_loaded_checkpoint(
        checkpoint,
        expected_model_family="advanced",
        device=device,
    )
    architecture = loaded.configuration["architecture"]
    expected_model_name = str(architecture.get("esm2_model_name") or "").strip()
    if not expected_model_name:
        raise ValueError(
            "Advanced checkpoint has no verified ESM-2 model provenance."
        )
    maximum_length = _positive_int(
        _required_mapping_value(
            loaded.configuration,
            "max_protein_length",
            "configuration",
        ),
        "configuration.max_protein_length",
    )
    expected_dimension = _positive_int(
        _required_mapping_value(
            architecture,
            "residue_embedding_dim_used",
            "configuration.architecture",
        ),
        "configuration.architecture.residue_embedding_dim_used",
    )
    from src.preprocessing import validate_residue_embedding_cache

    cache_metadata = validate_residue_embedding_cache(
        dataframe=pairs,
        residue_embedding_cache=residue_embedding_cache,
        max_length=maximum_length,
        expected_model_name=expected_model_name,
    )
    if cache_metadata.get("provenance_verified") is not True:
        raise ValueError(
            "Advanced inference requires a provenance-verified residue cache."
        )
    if cache_metadata.get("artifact_integrity_verified") is not True:
        raise ValueError(
            "Advanced inference requires a residue cache reloaded with verified "
            "tensor-artifact integrity."
        )
    if cache_metadata.get("schema_version") != 2:
        raise ValueError("Advanced inference requires residue-cache schema 2.")
    if str(cache_metadata.get("model_name", "")).strip() != expected_model_name:
        raise ValueError("Residue-cache ESM-2 model provenance is inconsistent.")
    checkpoint_cache_provenance = architecture["residue_cache_provenance"]
    checkpoint_revision = str(
        checkpoint_cache_provenance.get("resolved_model_revision") or ""
    ).strip()
    cache_revision = str(
        cache_metadata.get("resolved_model_revision") or ""
    ).strip()
    if not checkpoint_revision or cache_revision != checkpoint_revision:
        raise ValueError(
            "Residue-cache resolved ESM-2 revision does not match the checkpoint."
        )
    if int(cache_metadata.get("embedding_dimension", -1)) != expected_dimension:
        raise ValueError(
            "Residue-cache embedding dimension does not match the checkpoint."
        )
    if int(cache_metadata.get("max_length", -1)) != maximum_length:
        raise ValueError(
            "Residue-cache max length does not match the checkpoint."
        )
    if not isinstance(cache_metadata.get("target_sequence_sha256"), Mapping):
        raise ValueError("Residue cache has no validated target-sequence hashes.")
    for field in ("tensor_file", "tensor_file_bytes", "tensor_file_sha256"):
        if field not in cache_metadata:
            raise ValueError(
                f"Residue cache provenance is missing required field {field!r}."
            )
    if not str(cache_metadata["tensor_file"]).strip():
        raise ValueError("Residue cache tensor_file must be non-empty.")
    _positive_int(
        cache_metadata["tensor_file_bytes"],
        "residue_cache.tensor_file_bytes",
    )
    _validate_sha256(
        cache_metadata["tensor_file_sha256"],
        "residue_cache.tensor_file_sha256",
    )

    graph_cache = (
        drug_graph_cache
        if drug_graph_cache is not None
        else build_drug_graph_cache(pairs)
    )
    inference_pairs = pairs.copy(deep=True)
    inference_pairs["_screening_dummy_affinity"] = 0.0
    dataset = AdvancedDTADataset(
        inference_pairs,
        graph_cache,
        residue_embedding_cache,
        max_length=maximum_length,
        esm_embedding_dim=expected_dimension,
        affinity_column="_screening_dummy_affinity",
    )
    scores = _checkpoint_prediction_values(
        loaded,
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    predictions = _attach_checkpoint_provenance(
        pairs,
        scores,
        loaded,
        score_column=score_column,
    )
    predictions.attrs["residue_cache_provenance"] = dict(cache_metadata)
    return predictions


_CHECKPOINT_PROVENANCE_COLUMNS = frozenset(
    {
        "training_dataset",
        "affinity_label",
        "checkpoint_model_family",
        "checkpoint_model_name",
        "checkpoint_run_name",
        "checkpoint_sha256",
        "checkpoint_configuration_sha256",
        "checkpoint_data_fingerprint",
        "checkpoint_training_split",
        "checkpoint_seed",
        "checkpoint_max_protein_length",
        "checkpoint_split_fingerprint",
        "screening_pairs_sha256",
        "screening_pair_count",
        "screening_drug_count",
        "screening_target_count",
    }
)


def _prediction_checkpoint_provenance(
    dataframe: pd.DataFrame,
    name: str,
) -> dict[str, Any] | None:
    present = _CHECKPOINT_PROVENANCE_COLUMNS.intersection(dataframe.columns)
    if not present:
        return None
    missing = _CHECKPOINT_PROVENANCE_COLUMNS.difference(dataframe.columns)
    if missing:
        raise ValueError(
            f"{name} has incomplete checkpoint provenance columns: "
            f"{sorted(missing)}"
        )

    values: dict[str, Any] = {}
    for column in _CHECKPOINT_PROVENANCE_COLUMNS:
        unique = dataframe[column].drop_duplicates()
        if len(unique) != 1 or pd.isna(unique.iloc[0]):
            raise ValueError(
                f"{name}.{column} must contain one non-missing provenance value."
            )
        values[column] = unique.iloc[0]

    values["checkpoint_sha256"] = _validate_sha256(
        values["checkpoint_sha256"],
        f"{name}.checkpoint_sha256",
    )
    values["checkpoint_configuration_sha256"] = _validate_sha256(
        values["checkpoint_configuration_sha256"],
        f"{name}.checkpoint_configuration_sha256",
    )
    values["checkpoint_data_fingerprint"] = _validate_sha256(
        values["checkpoint_data_fingerprint"],
        f"{name}.checkpoint_data_fingerprint",
    )
    values["checkpoint_split_fingerprint"] = _validate_sha256(
        values["checkpoint_split_fingerprint"],
        f"{name}.checkpoint_split_fingerprint",
    )
    values["screening_pairs_sha256"] = _validate_sha256(
        values["screening_pairs_sha256"],
        f"{name}.screening_pairs_sha256",
    )

    dataset = str(values["training_dataset"]).strip().lower()
    affinity_label = str(values["affinity_label"]).strip()
    if dataset not in DATASET_AFFINITY_LABELS:
        raise ValueError(f"{name} training_dataset must be Davis or KIBA.")
    if affinity_label != DATASET_AFFINITY_LABELS[dataset]:
        raise ValueError(f"{name} dataset and affinity label are inconsistent.")
    training_split = str(values["checkpoint_training_split"]).strip().lower()
    if training_split not in {"random", "cold_drug", "cold_target"}:
        raise ValueError(f"{name} checkpoint training split is unsupported.")
    for count_column in (
        "screening_pair_count",
        "screening_drug_count",
        "screening_target_count",
    ):
        values[count_column] = _positive_int(
            values[count_column],
            f"{name}.{count_column}",
        )
    values["checkpoint_seed"] = _nonnegative_int(
        values["checkpoint_seed"],
        f"{name}.checkpoint_seed",
    )
    values["checkpoint_max_protein_length"] = _positive_int(
        values["checkpoint_max_protein_length"],
        f"{name}.checkpoint_max_protein_length",
    )
    values["training_dataset"] = dataset
    values["affinity_label"] = affinity_label
    values["checkpoint_training_split"] = training_split
    values["checkpoint_model_family"] = str(
        values["checkpoint_model_family"]
    ).strip().lower()
    return values


def run_dual_model_aml_screening(
    pairs: pd.DataFrame,
    baseline_checkpoint: str | Path | LoadedScreeningCheckpoint,
    advanced_checkpoint: str | Path | LoadedScreeningCheckpoint,
    residue_embedding_cache: Mapping[str, torch.Tensor],
    *,
    drug_graph_cache: Mapping[str, Any] | None = None,
    baseline_protein_tensor_cache: Mapping[str, torch.Tensor] | None = None,
    score_column: str = _DEFAULT_SCORE_COLUMN,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: str | torch.device | None = None,
) -> DualModelScreeningResult:
    """Run comparable baseline and advanced checkpoint screens end to end.

    Drug graphs are built once and reused by both model families. The function
    performs no training, checkpoint discovery, ESM-2 execution or network IO.
    """

    _validate_screening_pair_inputs(pairs)
    _validate_inference_settings(batch_size, num_workers)
    baseline_loaded = _coerce_loaded_checkpoint(
        baseline_checkpoint,
        expected_model_family="baseline",
        device=device,
    )
    advanced_loaded = _coerce_loaded_checkpoint(
        advanced_checkpoint,
        expected_model_family="advanced",
        device=device,
    )
    validate_screening_checkpoint_compatibility(
        baseline_loaded,
        advanced_loaded,
    )

    shared_graph_cache = (
        drug_graph_cache
        if drug_graph_cache is not None
        else build_drug_graph_cache(pairs)
    )
    maximum_length = _positive_int(
        baseline_loaded.configuration["max_protein_length"],
        "configuration.max_protein_length",
    )
    baseline_tokens = (
        baseline_protein_tensor_cache
        if baseline_protein_tensor_cache is not None
        else build_protein_tensor_cache(pairs, max_length=maximum_length)
    )
    baseline_predictions = predict_baseline_checkpoint(
        pairs,
        baseline_loaded,
        drug_graph_cache=shared_graph_cache,
        protein_tensor_cache=baseline_tokens,
        score_column=score_column,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    advanced_predictions = predict_advanced_checkpoint(
        pairs,
        advanced_loaded,
        residue_embedding_cache,
        drug_graph_cache=shared_graph_cache,
        score_column=score_column,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    baseline_rankings = rank_screening_predictions(
        baseline_predictions,
        score_column=score_column,
        model_name=baseline_loaded.model_name,
    )
    advanced_rankings = rank_screening_predictions(
        advanced_predictions,
        score_column=score_column,
        model_name=advanced_loaded.model_name,
    )
    comparison = compare_screening_models(
        baseline_predictions,
        advanced_predictions,
        baseline_score_column=score_column,
        advanced_score_column=score_column,
    )
    provenance = {
        **screening_wording_metadata(),
        "training_dataset": baseline_loaded.training_dataset,
        "affinity_label": baseline_loaded.affinity_label,
        "training_data_fingerprint": baseline_loaded.data_fingerprint,
        "screening_input": screening_input_provenance(pairs),
        "baseline_checkpoint": baseline_loaded.provenance,
        "advanced_checkpoint": advanced_loaded.provenance,
    }
    return DualModelScreeningResult(
        baseline_predictions=baseline_predictions,
        advanced_predictions=advanced_predictions,
        baseline_rankings=baseline_rankings,
        advanced_rankings=advanced_rankings,
        comparison=comparison,
        provenance=provenance,
    )


# Readable aliases for notebook workflows.
prepare_approved_drug_library = prepare_approved_drugs
create_aml_screening_pairs = build_screening_pairs
run_screening_inference = predict_screening_pairs
