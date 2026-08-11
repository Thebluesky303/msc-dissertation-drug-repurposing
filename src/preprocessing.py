"""Data preprocessing utilities for drug--target affinity prediction.

The Davis/KIBA transformations, identifier ordering, six atom features and
1,000-residue baseline encoding intentionally retain the verified v1 logic.
V2 adds stable pair identifiers, stricter graph validation, and an optional
frozen ESM-2 residue-embedding cache used only by the advanced model.
"""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data

from config import (
    BOND_FEATURE_DIM,
    ESM2_MODEL_NAME,
    MAX_PROTEIN_LENGTH,
    NODE_FEATURE_DIM,
)


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_TO_INDEX = {
    amino_acid: index + 1 for index, amino_acid in enumerate(AMINO_ACIDS)
}
ESM2_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZOU")


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


class ResidueEmbeddingCache(dict[str, torch.Tensor]):
    """Dict-compatible residue cache carrying validated provenance metadata."""

    def __init__(
        self,
        values: Mapping[str, torch.Tensor] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(values or {})
        self.metadata = dict(metadata or {})

HYBRIDIZATION_TO_INDEX = {
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
    Chem.rdchem.HybridizationType.SP3D: 4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
}


def kd_nm_to_pkd(kd_nm: float | np.ndarray) -> float | np.ndarray:
    """Convert Davis dissociation constants in nM using ``9-log10(Kd)``."""

    values = np.asarray(kd_nm, dtype=float)
    if np.any(values <= 0):
        raise ValueError("Kd values must be greater than zero.")

    converted = 9.0 - np.log10(values)
    return float(converted) if converted.ndim == 0 else converted


def canonicalize_smiles(smiles: str) -> str:
    """Return RDKit canonical SMILES for one valid, non-empty structure."""

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True)


def atom_to_feature_vector(atom: Chem.Atom) -> list[float]:
    """Return the verified six-feature v1 atom representation."""

    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(atom.GetIsAromatic()),
        float(HYBRIDIZATION_TO_INDEX.get(atom.GetHybridization(), 0)),
    ]


def bond_to_feature_vector(bond: Chem.Bond) -> list[float]:
    """Return six bond features used by the bond-aware GINE encoder."""

    bond_type = bond.GetBondType()
    return [
        float(bond_type == Chem.rdchem.BondType.SINGLE),
        float(bond_type == Chem.rdchem.BondType.DOUBLE),
        float(bond_type == Chem.rdchem.BondType.TRIPLE),
        float(bond_type == Chem.rdchem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def validate_graph_tensors(graph: Data) -> None:
    """Validate node, edge and bond-feature tensors on a molecular graph."""

    if graph.x.ndim != 2 or graph.x.shape[1] != NODE_FEATURE_DIM:
        raise ValueError(
            f"Expected graph.x shape [N, {NODE_FEATURE_DIM}], "
            f"received {tuple(graph.x.shape)}."
        )
    if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if graph.edge_index.dtype != torch.long:
        raise TypeError("edge_index must use torch.long indices.")
    if graph.edge_attr is None:
        raise ValueError("Molecular graphs must contain edge_attr.")
    if graph.edge_attr.ndim != 2:
        raise ValueError("edge_attr must have shape [E, bond_feature_dim].")
    if graph.edge_attr.shape != (
        graph.edge_index.shape[1],
        BOND_FEATURE_DIM,
    ):
        raise ValueError(
            "edge_attr must have one row per directed edge and "
            f"{BOND_FEATURE_DIM} columns; received "
            f"{tuple(graph.edge_attr.shape)}."
        )
    if not torch.isfinite(graph.x).all() or not torch.isfinite(
        graph.edge_attr
    ).all():
        raise ValueError("Graph features must contain only finite values.")


def smiles_to_graph(smiles: str) -> Data:
    """Convert SMILES to a bidirectional PyTorch Geometric graph."""

    canonical_smiles = canonicalize_smiles(smiles)
    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"Unable to create molecule from SMILES: {smiles}")

    x = torch.tensor(
        [atom_to_feature_vector(atom) for atom in molecule.GetAtoms()],
        dtype=torch.float32,
    )

    edge_indices: list[list[int]] = []
    edge_features: list[list[float]] = []
    for bond in molecule.GetBonds():
        source = bond.GetBeginAtomIdx()
        target = bond.GetEndAtomIdx()
        features = bond_to_feature_vector(bond)
        edge_indices.extend(([source, target], [target, source]))
        edge_features.extend((features, features))

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=molecule.GetNumAtoms(),
        canonical_smiles=canonical_smiles,
    )
    validate_graph_tensors(graph)
    return graph


def protein_sequence_to_tensor(
    sequence: str,
    max_length: int = MAX_PROTEIN_LENGTH,
) -> torch.Tensor:
    """Encode a sequence with v1 tokenisation, truncation and zero padding."""

    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError("Protein sequence must be a non-empty string.")
    if max_length < 1:
        raise ValueError("max_length must be positive.")

    encoded = torch.zeros(max_length, dtype=torch.long)
    for position, residue in enumerate(sequence.strip().upper()[:max_length]):
        encoded[position] = AMINO_ACID_TO_INDEX.get(residue, 0)
    return encoded


def clean_esm2_sequence(
    sequence: str,
    max_length: int = MAX_PROTEIN_LENGTH,
) -> str:
    """Clean a protein sequence for ESM-2, replacing unsupported residues by X."""

    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError("Protein sequence must be a non-empty string.")
    if max_length < 1:
        raise ValueError("max_length must be positive.")

    compact = "".join(sequence.split()).upper()
    cleaned = "".join(
        residue if residue in ESM2_AMINO_ACIDS else "X" for residue in compact
    )
    return cleaned[:max_length]


def make_pair_id(dataset: str, drug_id: Any, target_id: Any) -> str:
    """Return a stable, human-readable identifier for a benchmark pair."""

    normalized_dataset = str(dataset).strip().lower()
    normalized_drug = str(drug_id).strip()
    normalized_target = str(target_id).strip()
    if not normalized_dataset or not normalized_drug or not normalized_target:
        raise ValueError("dataset, drug_id, and target_id must be non-empty.")
    return f"{normalized_dataset}::{normalized_drug}::{normalized_target}"


def ensure_pair_ids(
    dataframe: pd.DataFrame,
    dataset_column: str = "dataset",
    pair_id_column: str = "pair_id",
) -> pd.DataFrame:
    """Return a copy containing deterministic, unique pair identifiers."""

    required = {dataset_column, "drug_id", "target_id"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing pair-ID columns: {sorted(missing)}")

    result = dataframe.copy()
    generated = [
        make_pair_id(dataset, drug, target)
        for dataset, drug, target in zip(
            result[dataset_column], result["drug_id"], result["target_id"]
        )
    ]

    if pair_id_column in result.columns:
        existing = result[pair_id_column].astype(str).tolist()
        if existing != generated:
            raise ValueError("Existing pair_id values do not match identifiers.")
    else:
        result.insert(1, pair_id_column, generated)

    if result[pair_id_column].duplicated().any():
        raise ValueError("pair_id values must be unique.")
    return result


def pair_id_sha256(dataframe: pd.DataFrame) -> str:
    """Hash ordered pair identifiers exactly as consumed by an experiment."""

    if "pair_id" not in dataframe.columns:
        raise ValueError("pair_id is required to fingerprint pair membership.")
    return hashlib.sha256(
        "\n".join(dataframe["pair_id"].astype(str)).encode("utf-8")
    ).hexdigest()


def scientific_pairs_sha256(dataframe: pd.DataFrame) -> str:
    """Hash ordered identifiers, structures, sequences, labels and affinities."""

    columns = [
        "dataset",
        "pair_id",
        "drug_id",
        "target_id",
        "smiles",
        "protein_sequence",
        "raw_affinity",
        "affinity",
        "affinity_type",
    ]
    selected = [column for column in columns if column in dataframe.columns]
    required = set(columns).difference({"raw_affinity"})
    missing = required.difference(selected)
    if missing:
        raise ValueError(
            "Cannot fingerprint scientific pair data; missing columns: "
            f"{sorted(missing)}"
        )
    normalized = dataframe[selected].copy()
    numeric_columns = [
        column for column in ("raw_affinity", "affinity") if column in selected
    ]
    for column in numeric_columns:
        numeric = pd.to_numeric(normalized[column], errors="coerce")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"{column} must be finite to fingerprint pair data.")
        normalized[column] = numeric.astype(float)
    for column in set(selected).difference(numeric_columns):
        normalized[column] = normalized[column].astype(str)
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json_dictionary(file_path: str | Path) -> dict:
    """Load a JSON dictionary stored in a GraphDTA text file."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON dictionary in {path}.")
    return value


def load_affinity_matrix(file_path: str | Path) -> np.ndarray:
    """Load a pickled GraphDTA affinity matrix without changing its values."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Affinity matrix not found: {path}")
    with path.open("rb") as file:
        matrix = pickle.load(file, encoding="latin1")
    return np.asarray(matrix, dtype=float)


def load_graphdta_pairs(
    dataset_name: str,
    raw_data_directory: str | Path,
) -> pd.DataFrame:
    """Create pair-level Davis or KIBA data using verified v1 ordering.

    Davis nM Kd values are converted to pKd.  KIBA values are retained as the
    scores supplied by the GraphDTA matrix; Davis and KIBA are never combined.
    """

    dataset = dataset_name.strip().lower()
    if dataset not in {"davis", "kiba"}:
        raise ValueError("dataset_name must be either 'davis' or 'kiba'.")

    directory = Path(raw_data_directory)
    ligands = load_json_dictionary(directory / "ligands_can.txt")
    proteins = load_json_dictionary(directory / "proteins.txt")
    affinity_matrix = load_affinity_matrix(directory / "Y")

    drug_ids = list(ligands.keys())
    target_ids = list(proteins.keys())
    expected_shape = (len(drug_ids), len(target_ids))
    if affinity_matrix.shape != expected_shape:
        raise ValueError(
            f"Affinity matrix shape {affinity_matrix.shape} does not match "
            f"identifier shape {expected_shape}."
        )

    records: list[dict[str, Any]] = []
    for drug_index, drug_id in enumerate(drug_ids):
        for target_index, target_id in enumerate(target_ids):
            raw_affinity = affinity_matrix[drug_index, target_index]
            if not np.isfinite(raw_affinity):
                continue

            if dataset == "davis":
                affinity = kd_nm_to_pkd(raw_affinity)
                affinity_type = "pKd"
            else:
                affinity = float(raw_affinity)
                affinity_type = "KIBA_score"

            records.append(
                {
                    "dataset": dataset,
                    "pair_id": make_pair_id(dataset, drug_id, target_id),
                    "drug_id": str(drug_id),
                    "target_id": str(target_id),
                    "smiles": ligands[drug_id],
                    "protein_sequence": proteins[target_id],
                    "raw_affinity": float(raw_affinity),
                    "affinity": float(affinity),
                    "affinity_type": affinity_type,
                }
            )

    pairs = pd.DataFrame.from_records(records)
    if pairs.empty:
        raise ValueError(f"No valid drug--target pairs loaded for {dataset}.")
    validate_pairs_dataframe(pairs, raise_on_error=True)
    return pairs


def _check_unique_mapping(
    dataframe: pd.DataFrame,
    identifier_column: str,
    value_column: str,
) -> None:
    counts = dataframe.groupby(identifier_column)[value_column].nunique()
    if (counts > 1).any():
        raise ValueError(
            f"Some {identifier_column} values map to multiple {value_column} values."
        )


def build_drug_graph_cache(
    dataframe: pd.DataFrame,
    drug_id_column: str = "drug_id",
    smiles_column: str = "smiles",
) -> Dict[str, Data]:
    """Build one validated molecular graph for each unique drug identifier."""

    _check_unique_mapping(dataframe, drug_id_column, smiles_column)
    unique = dataframe[[drug_id_column, smiles_column]].drop_duplicates(
        subset=[drug_id_column]
    )
    return {
        str(row[drug_id_column]): smiles_to_graph(row[smiles_column])
        for _, row in unique.iterrows()
    }


def build_protein_tensor_cache(
    dataframe: pd.DataFrame,
    target_id_column: str = "target_id",
    sequence_column: str = "protein_sequence",
    max_length: int = MAX_PROTEIN_LENGTH,
) -> Dict[str, torch.Tensor]:
    """Build one baseline token tensor for each unique target identifier."""

    _check_unique_mapping(dataframe, target_id_column, sequence_column)
    unique = dataframe[[target_id_column, sequence_column]].drop_duplicates(
        subset=[target_id_column]
    )
    return {
        str(row[target_id_column]): protein_sequence_to_tensor(
            row[sequence_column], max_length=max_length
        )
        for _, row in unique.iterrows()
    }


def _load_tensor_cache(path: Path) -> dict[str, torch.Tensor]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "This project requires restricted torch.load(weights_only=True); "
            "upgrade to the supported PyTorch version."
        ) from error
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Expected a mapping in ESM-2 cache {path}.")
    cache: dict[str, torch.Tensor] = {}
    for key, value in loaded.items():
        target_id = str(key).strip()
        tensor = torch.as_tensor(value).float()
        if not target_id:
            raise ValueError(f"ESM-2 cache {path} contains an empty target ID.")
        if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] < 1:
            raise ValueError(
                f"ESM-2 cache value for {target_id} must have shape [L, E]."
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"ESM-2 cache value for {target_id} contains non-finite values."
            )
        cache[target_id] = tensor.contiguous()
    if not cache:
        raise ValueError(f"ESM-2 cache {path} is empty.")
    dimensions = {int(tensor.shape[1]) for tensor in cache.values()}
    if len(dimensions) != 1:
        raise ValueError(f"ESM-2 cache {path} contains inconsistent dimensions.")
    return cache


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one local artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def residue_embedding_cache_metadata_path(cache_path: str | Path) -> Path:
    """Return the JSON sidecar path associated with a tensor-cache path."""

    return Path(cache_path).with_suffix(".metadata.json")


def _read_residue_cache_metadata(cache_path: str | Path) -> dict[str, Any]:
    metadata_path = residue_embedding_cache_metadata_path(cache_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Residue cache metadata not found: "
            f"{metadata_path}. Regenerate with the v2 cache builder or load "
            "explicitly with require_metadata=False and treat provenance as "
            "unverified."
        )
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected a JSON object in {metadata_path}.")
    return metadata


def load_residue_embedding_cache(
    cache_path: str | Path,
    *,
    require_metadata: bool = True,
) -> ResidueEmbeddingCache:
    """Load a frozen ESM-2 residue cache through one public entry point.

    Values are returned on CPU with shape ``[num_residues, esm_dim]``.  The
    advanced dataset performs the remaining length, finiteness and dimension
    checks before a tensor is used for training or inference.
    """

    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Residue embedding cache not found: {path}")
    if require_metadata:
        metadata = _read_residue_cache_metadata(path)
        if int(metadata.get("schema_version", -1)) != 2:
            raise ValueError(
                "Unsupported residue cache metadata schema version; regenerate "
                "the cache with the current v2 builder."
            )
        required = {
            "schema_version",
            "model_name",
            "max_length",
            "embedding_dimension",
            "cached_target_ids",
            "target_sequence_sha256",
            "target_residue_lengths",
            "tensor_file",
            "tensor_file_bytes",
            "tensor_file_sha256",
        }
        missing = required.difference(metadata)
        if missing:
            raise ValueError(
                f"Residue cache metadata is missing fields: {sorted(missing)}"
            )
        if str(metadata["tensor_file"]) != path.name:
            raise ValueError("Residue cache metadata names a different tensor file.")
        if int(metadata["tensor_file_bytes"]) != path.stat().st_size:
            raise ValueError("Residue cache byte size disagrees with metadata.")
        if str(metadata["tensor_file_sha256"]).lower() != _sha256_file(path):
            raise ValueError("Residue cache SHA-256 disagrees with metadata.")
        tensors = _load_tensor_cache(path)
        if set(map(str, metadata["cached_target_ids"])) != set(tensors):
            raise ValueError("Residue cache tensor keys and metadata IDs differ.")
        dimensions = {int(tensor.shape[1]) for tensor in tensors.values()}
        if dimensions != {int(metadata["embedding_dimension"])}:
            raise ValueError("Residue cache dimension disagrees with metadata.")
        sequence_hashes = metadata["target_sequence_sha256"]
        if not isinstance(sequence_hashes, dict) or not set(tensors).issubset(
            map(str, sequence_hashes)
        ):
            raise ValueError("Residue cache metadata lacks target sequence hashes.")
        recorded_lengths = metadata["target_residue_lengths"]
        if not isinstance(recorded_lengths, dict) or set(map(str, recorded_lengths)) != set(
            tensors
        ):
            raise ValueError("Residue cache metadata lacks exact residue lengths.")
        for target_id, tensor in tensors.items():
            if int(recorded_lengths[target_id]) != int(tensor.shape[0]):
                raise ValueError(
                    f"Residue cache length disagrees with metadata for {target_id}."
                )
        metadata["artifact_integrity_verified"] = True
        metadata["provenance_verified"] = bool(
            metadata.get("model_provenance_verified", False)
            and metadata["artifact_integrity_verified"]
        )
    else:
        tensors = _load_tensor_cache(path)
        metadata = {
            "schema_version": None,
            "provenance_verified": False,
            "provenance_warning": "Metadata sidecar was not required or validated.",
            "embedding_dimension": next(iter(tensors.values())).shape[1],
            "cached_target_ids": sorted(tensors),
        }
    return ResidueEmbeddingCache(tensors, metadata=metadata)


def validate_residue_embedding_cache(
    dataframe: pd.DataFrame,
    residue_embedding_cache: Mapping[str, torch.Tensor],
    *,
    target_id_column: str = "target_id",
    sequence_column: str = "protein_sequence",
    max_length: int = MAX_PROTEIN_LENGTH,
    expected_model_name: str | None = None,
) -> dict[str, Any]:
    """Validate cache coverage, dimensions and sequence/model provenance."""

    _check_unique_mapping(dataframe, target_id_column, sequence_column)
    unique = dataframe[[target_id_column, sequence_column]].drop_duplicates(
        subset=[target_id_column]
    )
    sequences = {
        str(row[target_id_column]): clean_esm2_sequence(
            row[sequence_column], max_length=max_length
        )
        for _, row in unique.iterrows()
    }
    missing = set(sequences).difference(map(str, residue_embedding_cache))
    if missing:
        raise KeyError(
            "Residue embedding cache is missing target IDs such as: "
            f"{sorted(missing)[:5]}"
        )
    for target_id, sequence in sequences.items():
        tensor = torch.as_tensor(residue_embedding_cache[target_id])
        if tensor.ndim != 2 or int(tensor.shape[0]) != len(sequence):
            raise ValueError(
                f"Residue cache length does not match cleaned sequence for {target_id}."
            )
        if tensor.shape[1] < 1 or not torch.isfinite(tensor).all():
            raise ValueError(f"Residue cache tensor is invalid for {target_id}.")

    metadata = dict(getattr(residue_embedding_cache, "metadata", {}) or {})
    if not metadata:
        return {
            "provenance_verified": False,
            "provenance_warning": "In-memory mapping has no v2 metadata sidecar.",
            "cached_target_count": len(residue_embedding_cache),
        }

    if int(metadata.get("max_length", -1)) != int(max_length):
        raise ValueError(
            "Residue cache max_length does not match the experiment configuration."
        )
    if expected_model_name is not None and metadata.get("model_name") != (
        expected_model_name
    ):
        raise ValueError(
            "Residue cache model does not match the configured ESM-2 model."
        )
    if metadata.get("model_provenance_verified") is True:
        verified_required = {
            "schema_version",
            "model_name",
            "resolved_model_revision",
            "embedding_dimension",
            "cached_target_ids",
            "target_residue_lengths",
            "tensor_file",
            "tensor_file_bytes",
            "tensor_file_sha256",
        }
        missing_verified = verified_required.difference(metadata)
        if missing_verified or int(metadata.get("schema_version", -1)) != 2:
            raise ValueError(
                "A provenance-verified residue cache requires complete schema-2 "
                f"metadata; missing {sorted(missing_verified)}."
            )
        if not str(metadata.get("resolved_model_revision") or "").strip():
            raise ValueError("Verified residue cache has no resolved model revision.")
        dimensions = {
            int(torch.as_tensor(value).shape[1])
            for value in residue_embedding_cache.values()
        }
        if dimensions != {int(metadata["embedding_dimension"])}:
            raise ValueError("Residue cache embedding dimension metadata differs.")
        if set(map(str, metadata["cached_target_ids"])) != set(
            map(str, residue_embedding_cache)
        ):
            raise ValueError("Residue cache IDs differ from verified metadata.")
    recorded_hashes = metadata.get("target_sequence_sha256")
    if not isinstance(recorded_hashes, dict):
        raise ValueError("Residue cache metadata has no sequence-hash mapping.")
    for target_id, sequence in sequences.items():
        expected_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        if recorded_hashes.get(target_id) != expected_hash:
            raise ValueError(
                f"Residue cache sequence hash does not match target {target_id}."
            )
        recorded_lengths = metadata.get("target_residue_lengths")
        if isinstance(recorded_lengths, Mapping) and int(
            recorded_lengths.get(target_id, -1)
        ) != len(sequence):
            raise ValueError(
                f"Residue cache metadata length does not match target {target_id}."
            )
    metadata["provenance_verified"] = bool(
        metadata.get("model_provenance_verified", False)
        and (
            metadata.get("artifact_integrity_verified", False)
            or metadata.get("provenance_verified", False)
        )
    )
    metadata["validated_target_count"] = len(sequences)
    return metadata


def _atomic_save_residue_cache(
    cache: Mapping[str, torch.Tensor],
    output_path: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically replace tensor and JSON files, leaving old files on failure."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_temporary = output_path.with_name(f".{output_path.name}.tmp")
    metadata_path = residue_embedding_cache_metadata_path(output_path)
    metadata_temporary = metadata_path.with_name(f".{metadata_path.name}.tmp")
    torch.save(dict(cache), tensor_temporary)
    enriched_metadata = dict(metadata)
    enriched_metadata.update(
        {
            "tensor_file": output_path.name,
            "tensor_file_bytes": tensor_temporary.stat().st_size,
            "tensor_file_sha256": _sha256_file(tensor_temporary),
        }
    )
    with metadata_temporary.open("w", encoding="utf-8") as file:
        json.dump(
            enriched_metadata,
            file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        file.write("\n")
    tensor_temporary.replace(output_path)
    metadata_temporary.replace(metadata_path)
    return enriched_metadata


def build_esm2_residue_embedding_cache(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    *,
    target_id_column: str = "target_id",
    sequence_column: str = "protein_sequence",
    model_name: str = ESM2_MODEL_NAME,
    model_revision: str | None = None,
    max_length: int = MAX_PROTEIN_LENGTH,
    batch_size: int = 1,
    checkpoint_interval_batches: int = 25,
    device: str | torch.device | None = None,
    resume: bool = True,
    tokenizer: Any | None = None,
    model: torch.nn.Module | None = None,
) -> ResidueEmbeddingCache:
    """Precompute frozen, residue-level ESM-2 embeddings.

    The function is intentionally opt-in and never runs during import or model
    training.  Supplying ``tokenizer`` and ``model`` supports offline testing;
    otherwise Hugging Face Transformers is imported lazily.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if checkpoint_interval_batches < 1:
        raise ValueError("checkpoint_interval_batches must be positive.")
    _check_unique_mapping(dataframe, target_id_column, sequence_column)

    unique = dataframe[[target_id_column, sequence_column]].drop_duplicates(
        subset=[target_id_column]
    )
    if unique.empty:
        raise ValueError("Cannot build an ESM-2 cache from an empty dataframe.")
    sequence_by_target = {
        str(row[target_id_column]): clean_esm2_sequence(
            row[sequence_column], max_length=max_length
        )
        for _, row in unique.iterrows()
    }
    sequence_hashes = {
        target_id: hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        for target_id, sequence in sequence_by_target.items()
    }
    sequence_fingerprint = hashlib.sha256(
        json.dumps(sequence_hashes, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()

    output = Path(output_path)
    loaded_cache = (
        load_residue_embedding_cache(output)
        if resume and output.exists()
        else None
    )
    cache = dict(loaded_cache or {})
    previous_metadata: dict[str, Any] | None = None
    if cache:
        previous_metadata = dict(loaded_cache.metadata)
        if previous_metadata.get("model_name") != model_name:
            raise ValueError("Cannot resume: ESM-2 model name has changed.")
        if previous_metadata.get("model_revision") != model_revision:
            raise ValueError("Cannot resume: ESM-2 model revision has changed.")
        if int(previous_metadata.get("max_length", -1)) != int(max_length):
            raise ValueError("Cannot resume: protein max_length has changed.")
        if previous_metadata.get("input_sequence_sha256") != sequence_fingerprint:
            raise ValueError("Cannot resume: target IDs or protein sequences changed.")
        if set(cache).difference(sequence_by_target):
            raise ValueError("Cannot resume: cache contains targets absent from input.")

    created_model = tokenizer is None and model is None
    if (tokenizer is None) != (model is None):
        raise ValueError("Provide both tokenizer and model, or neither.")
    if created_model:
        try:
            from transformers import AutoTokenizer, EsmModel
        except ImportError as error:
            raise ImportError(
                "transformers is required to build ESM-2 embeddings."
            ) from error
        load_kwargs = {"revision": model_revision} if model_revision else {}
        tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        model = EsmModel.from_pretrained(model_name, **load_kwargs)

    assert tokenizer is not None and model is not None
    selected_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(selected_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    resolved_model_revision = (
        getattr(getattr(model, "config", None), "_commit_hash", None)
        or model_revision
    )
    if previous_metadata is not None and previous_metadata.get(
        "resolved_model_revision"
    ) != resolved_model_revision:
        raise ValueError("Cannot resume: resolved ESM-2 model revision changed.")

    pending = [
        (target_id, sequence)
        for target_id, sequence in sequence_by_target.items()
        if target_id not in cache
    ]

    source_datasets = (
        sorted(dataframe["dataset"].astype(str).str.strip().str.lower().unique())
        if "dataset" in dataframe.columns
        else []
    )
    source_pair_fingerprint = (
        pair_id_sha256(dataframe)
        if "pair_id" in dataframe.columns
        else None
    )

    def metadata_for(current_cache: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        dimensions = {int(tensor.shape[1]) for tensor in current_cache.values()}
        if len(dimensions) != 1:
            raise ValueError("Cached ESM-2 embeddings have inconsistent dimensions.")
        return {
            "schema_version": 2,
            "cache_type": "frozen_esm2_residue_embeddings",
            "residue_level": True,
            "model_name": model_name,
            "model_revision": model_revision,
            "resolved_model_revision": resolved_model_revision,
            "model_source": (
                "huggingface_pretrained" if created_model else "supplied_model"
            ),
            "model_class": type(model).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "model_parameters_frozen": True,
            "model_provenance_verified": bool(
                created_model and resolved_model_revision
            ),
            "library_versions": {
                "torch": torch.__version__,
                "transformers": _installed_version("transformers"),
                "huggingface_hub": _installed_version("huggingface-hub"),
            },
            "max_length": int(max_length),
            "embedding_dimension": next(iter(dimensions)),
            "dtype": "float32",
            "input_rows": int(len(dataframe)),
            "input_target_count": len(sequence_by_target),
            "cached_target_count": len(current_cache),
            "cached_target_ids": sorted(current_cache),
            "target_residue_lengths": {
                target_id: int(current_cache[target_id].shape[0])
                for target_id in sorted(current_cache)
            },
            "target_sequence_sha256": sequence_hashes,
            "input_sequence_sha256": sequence_fingerprint,
            "source_datasets": source_datasets,
            "source_pair_id_sha256": source_pair_fingerprint,
            "source_pair_id_fingerprint_algorithm": (
                "ordered_pair_ids_sha256_v1"
            ),
            "checkpoint_interval_batches": int(checkpoint_interval_batches),
        }

    total_batches = (len(pending) + batch_size - 1) // batch_size
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        target_ids = [item[0] for item in batch]
        sequences = [item[1] for item in batch]
        tokens = tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length + 2,
            return_special_tokens_mask=True,
        )
        special_tokens_mask = tokens.pop("special_tokens_mask").bool()
        model_inputs = {
            name: value.to(selected_device)
            for name, value in tokens.items()
            if isinstance(value, torch.Tensor)
        }

        with torch.inference_mode():
            hidden = model(**model_inputs).last_hidden_state.detach().cpu()

        attention_mask = tokens["attention_mask"].bool()
        residue_mask = attention_mask & ~special_tokens_mask
        for index, target_id in enumerate(target_ids):
            embeddings = hidden[index, residue_mask[index]].float()
            expected_length = len(sequences[index])
            if embeddings.shape[0] != expected_length:
                raise ValueError(
                    f"ESM-2 produced {embeddings.shape[0]} residues for "
                    f"{target_id}; expected {expected_length}."
                )
            if embeddings.ndim != 2 or not torch.isfinite(embeddings).all():
                raise ValueError(f"Invalid ESM-2 embeddings for {target_id}.")
            cache[target_id] = embeddings.contiguous()

        completed_batches = start // batch_size + 1
        if (
            completed_batches % checkpoint_interval_batches == 0
            or completed_batches == total_batches
        ):
            _atomic_save_residue_cache(cache, output, metadata_for(cache))

    if not pending:
        _atomic_save_residue_cache(cache, output, metadata_for(cache))
    metadata = _read_residue_cache_metadata(output)
    metadata["artifact_integrity_verified"] = True
    metadata["provenance_verified"] = bool(
        metadata.get("model_provenance_verified", False)
    )
    return ResidueEmbeddingCache(cache, metadata=metadata)


def validate_pairs_dataframe(
    dataframe: pd.DataFrame,
    *,
    raise_on_error: bool = False,
) -> dict[str, int]:
    """Validate processed pair data and return reproducible summary counts."""

    required = {
        "drug_id",
        "target_id",
        "smiles",
        "protein_sequence",
        "affinity",
    }
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    duplicate_pairs = int(
        dataframe.duplicated(subset=["drug_id", "target_id"]).sum()
    )
    duplicate_pair_ids = int(
        dataframe["pair_id"].duplicated().sum()
        if "pair_id" in dataframe.columns
        else 0
    )
    empty_sequences = int(
        dataframe["protein_sequence"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )
    affinities = pd.to_numeric(dataframe["affinity"], errors="coerce")
    non_finite = int((~np.isfinite(affinities.to_numpy())).sum())
    invalid_smiles = sum(
        Chem.MolFromSmiles(str(smiles)) is None
        for smiles in dataframe["smiles"].dropna().unique()
    )

    summary = {
        "rows": int(len(dataframe)),
        "unique_drugs": int(dataframe["drug_id"].nunique()),
        "unique_targets": int(dataframe["target_id"].nunique()),
        "duplicate_pairs": duplicate_pairs,
        "duplicate_pair_ids": duplicate_pair_ids,
        "invalid_smiles": int(invalid_smiles),
        "empty_protein_sequences": empty_sequences,
        "non_finite_affinities": non_finite,
    }
    if raise_on_error and any(
        summary[key]
        for key in (
            "duplicate_pairs",
            "duplicate_pair_ids",
            "invalid_smiles",
            "empty_protein_sequences",
            "non_finite_affinities",
        )
    ):
        raise ValueError(f"Pair-data validation failed: {summary}")
    return summary


def save_pairs_dataframe(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Validate and save processed pair data without changing row order."""

    validate_pairs_dataframe(dataframe, raise_on_error=True)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def prepare_dataset(
    dataset_name: str,
    raw_data_directory: str | Path,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, validate and optionally save one GraphDTA benchmark dataset.

    This thin orchestration hook is intended for notebooks and Colab.  The
    verified Davis/KIBA parsing and affinity transformations remain in
    :func:`load_graphdta_pairs`.
    """

    dataset = dataset_name.strip().lower()
    raw_directory = Path(raw_data_directory)
    pairs = load_graphdta_pairs(
        dataset_name=dataset,
        raw_data_directory=raw_directory,
    )
    validation_summary = validate_pairs_dataframe(pairs, raise_on_error=True)
    source_files: dict[str, dict[str, Any]] = {}
    for name in ("ligands_can.txt", "proteins.txt", "Y"):
        path = raw_directory / name
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        source_files[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    affinity_matrix = load_affinity_matrix(raw_directory / "Y")
    pair_id_fingerprint = pair_id_sha256(pairs)
    data_fingerprint = scientific_pairs_sha256(pairs)
    affinity_type = "pKd" if dataset == "davis" else "KIBA_score"
    affinity_semantics = (
        "Negative base-10 logarithm of molar dissociation constant, "
        "computed from GraphDTA Kd in nM as 9-log10(Kd_nM)."
        if dataset == "davis"
        else "GraphDTA KIBA integrated activity score retained on its native "
        "scale; it is not a pKd value."
    )
    summary: dict[str, Any] = {
        **validation_summary,
        "preprocessing_schema_version": 1,
        "dataset": dataset,
        "affinity_type": affinity_type,
        "affinity_target": "pKd" if dataset == "davis" else "KIBA score",
        "affinity_semantics": affinity_semantics,
        "affinity_transformation": (
            "9 - log10(Kd_nM)"
            if dataset == "davis"
            else "GraphDTA KIBA score retained without pKd conversion"
        ),
        "missing_affinity_cells": int((~np.isfinite(affinity_matrix)).sum()),
        "pair_id_sha256": pair_id_fingerprint,
        "scientific_data_sha256": data_fingerprint,
        "raw_source_files": source_files,
    }
    if output_path is not None:
        output = Path(output_path)
        save_pairs_dataframe(pairs, output)
        summary.update(
            {
                "processed_csv_file": output.name,
                "processed_csv_bytes": output.stat().st_size,
                "processed_csv_sha256": _sha256_file(output),
                "processed_csv_rows": int(len(pairs)),
            }
        )
        metadata_path = output.with_suffix(".metadata.json")
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, sort_keys=True, allow_nan=False)
            file.write("\n")
        summary["metadata_path"] = str(metadata_path.resolve())
    return pairs, summary
