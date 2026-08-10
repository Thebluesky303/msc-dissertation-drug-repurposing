"""Data preprocessing utilities for drug–target affinity prediction."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data

# from config import MAX_PROTEIN_LENGTH, NODE_FEATURE_DIM
from config import (
    MAX_PROTEIN_LENGTH,
    NODE_FEATURE_DIM,
    BOND_FEATURE_DIM,
)


# ============================================================
# PROTEIN ENCODING
# ============================================================

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

AMINO_ACID_TO_INDEX = {
    amino_acid: index + 1
    for index, amino_acid in enumerate(AMINO_ACIDS)
}


# ============================================================
# DRUG ATOM FEATURES
# ============================================================

HYBRIDIZATION_TO_INDEX = {
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
    Chem.rdchem.HybridizationType.SP3D: 4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
}


def kd_nm_to_pkd(kd_nm: float | np.ndarray) -> float | np.ndarray:
    """
    Convert dissociation constant values from nanomolar Kd to pKd.

    Formula:
        pKd = 9 - log10(Kd in nM)
    """

    values = np.asarray(kd_nm, dtype=float)

    if np.any(values <= 0):
        raise ValueError("Kd values must be greater than zero.")

    pkd_values = 9.0 - np.log10(values)

    if pkd_values.ndim == 0:
        return float(pkd_values)

    return pkd_values


def canonicalize_smiles(smiles: str) -> str:
    """Convert a SMILES string into its canonical RDKit representation."""

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    return Chem.MolToSmiles(molecule, canonical=True)


def atom_to_feature_vector(atom: Chem.Atom) -> list[float]:
    """
    Convert one RDKit atom into the six baseline numerical features.

    Features:
        1. Atomic number
        2. Atom degree
        3. Formal charge
        4. Total hydrogen count
        5. Aromaticity
        6. Hybridisation category
    """

    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(atom.GetIsAromatic()),
        float(HYBRIDIZATION_TO_INDEX.get(atom.GetHybridization(), 0)),
    ]


# def smiles_to_graph(smiles: str) -> Data:
#     """
#     Convert a SMILES string into a PyTorch Geometric molecular graph.

#     Each bond is represented in both directions.
#     """

#     canonical_smiles = canonicalize_smiles(smiles)
#     molecule = Chem.MolFromSmiles(canonical_smiles)

#     if molecule is None or molecule.GetNumAtoms() == 0:
#         raise ValueError(f"Unable to create molecule from SMILES: {smiles}")

#     node_features = [
#         atom_to_feature_vector(atom)
#         for atom in molecule.GetAtoms()
#     ]

#     x = torch.tensor(node_features, dtype=torch.float32)

#     if x.shape[1] != NODE_FEATURE_DIM:
#         raise ValueError(
#             f"Expected {NODE_FEATURE_DIM} atom features, "
#             f"but created {x.shape[1]}."
#         )

#     directed_edges: list[list[int]] = []

#     # for bond in molecule.GetBonds():
#     #     start_atom = bond.GetBeginAtomIdx()
#     #     end_atom = bond.GetEndAtomIdx()

#     #     directed_edges.append([start_atom, end_atom])
#     #     directed_edges.append([end_atom, start_atom])
#     edge_indices = []
#     edge_features = []
    
#     for bond in molecule.GetBonds():
#         source_atom = bond.GetBeginAtomIdx()
#         target_atom = bond.GetEndAtomIdx()
    
#         bond_features = (
#             bond_to_feature_vector(
#                 bond
#             )
#         )
    
#         # Forward direction
#         edge_indices.append(
#             [
#                 source_atom,
#                 target_atom,
#             ]
#         )
    
#         edge_features.append(
#             bond_features
#         )
    
#         # Reverse direction
#         edge_indices.append(
#             [
#                 target_atom,
#                 source_atom,
#             ]
#         )
    
#         edge_features.append(
#             bond_features
#         )

#         if edge_indices:
#             edge_index = torch.tensor(
#                 edge_indices,
#                 dtype=torch.long,
#             ).t().contiguous()
        
#             edge_attr = torch.tensor(
#                 edge_features,
#                 dtype=torch.float32,
#             )
        
#         else:
#             edge_index = torch.empty(
#                 (2, 0),
#                 dtype=torch.long,
#             )
        
#             edge_attr = torch.empty(
#                 (
#                     0,
#                     cfg.BOND_FEATURE_DIM,
#                 ),
#                 dtype=torch.float32,
#             )




        

#     if directed_edges:
#         edge_index = (
#             torch.tensor(directed_edges, dtype=torch.long)
#             .t()
#             .contiguous()
#         )
#     else:
#         edge_index = torch.empty((2, 0), dtype=torch.long)

#     return Data(
#         x=x,
#         edge_index=edge_index,
#         num_nodes=molecule.GetNumAtoms(),
#         edge_attr=edge_attr,
#     )


def smiles_to_graph(smiles: str) -> Data:
    """
    Convert a SMILES string into a PyTorch Geometric molecular graph.

    Each chemical bond is represented in both directions.

    Returns:
        Data object containing:
            x         -> atom features
            edge_index -> graph connectivity
            edge_attr  -> bond features
    """

    canonical_smiles = canonicalize_smiles(smiles)

    molecule = Chem.MolFromSmiles(
        canonical_smiles
    )

    if (
        molecule is None
        or molecule.GetNumAtoms() == 0
    ):
        raise ValueError(
            f"Unable to create molecule from SMILES: {smiles}"
        )

    # ========================================================
    # Node / atom features
    # ========================================================

    node_features = [
        atom_to_feature_vector(atom)
        for atom in molecule.GetAtoms()
    ]

    x = torch.tensor(
        node_features,
        dtype=torch.float32,
    )

    if x.shape[1] != NODE_FEATURE_DIM:
        raise ValueError(
            f"Expected {NODE_FEATURE_DIM} atom features, "
            f"but created {x.shape[1]}."
        )

    # ========================================================
    # Edge / bond features
    # ========================================================

    edge_indices = []
    edge_features = []

    for bond in molecule.GetBonds():

        source_atom = (
            bond.GetBeginAtomIdx()
        )

        target_atom = (
            bond.GetEndAtomIdx()
        )

        bond_features = (
            bond_to_feature_vector(
                bond
            )
        )

        # Forward direction
        edge_indices.append(
            [
                source_atom,
                target_atom,
            ]
        )

        edge_features.append(
            bond_features
        )

        # Reverse direction
        edge_indices.append(
            [
                target_atom,
                source_atom,
            ]
        )

        edge_features.append(
            bond_features
        )

    # ========================================================
    # Convert to tensors
    # ========================================================

    if edge_indices:

        edge_index = torch.tensor(
            edge_indices,
            dtype=torch.long,
        ).t().contiguous()

        edge_attr = torch.tensor(
            edge_features,
            dtype=torch.float32,
        )

    else:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

        edge_attr = torch.empty(
            (
                0,
                BOND_FEATURE_DIM,
            ),
            dtype=torch.float32,
        )

    # ========================================================
    # Final molecular graph
    # ========================================================

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=molecule.GetNumAtoms(),
    )

def protein_sequence_to_tensor(
    sequence: str,
    max_length: int = MAX_PROTEIN_LENGTH,
) -> torch.Tensor:
    """
    Convert an amino-acid sequence into a fixed-length integer tensor.

    Index 0 is used for padding and unsupported amino-acid characters.
    """

    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError("Protein sequence must be a non-empty string.")

    sequence = sequence.strip().upper()

    encoded_sequence = torch.zeros(
        max_length,
        dtype=torch.long,
    )

    for position, amino_acid in enumerate(sequence[:max_length]):
        encoded_sequence[position] = AMINO_ACID_TO_INDEX.get(
            amino_acid,
            0,
        )

    return encoded_sequence

def bond_to_feature_vector(
    bond,
) -> list[float]:
    """Convert one RDKit bond into six features."""

    bond_type = bond.GetBondType()

    return [
        float(
            bond_type
            == Chem.rdchem.BondType.SINGLE
        ),
        float(
            bond_type
            == Chem.rdchem.BondType.DOUBLE
        ),
        float(
            bond_type
            == Chem.rdchem.BondType.TRIPLE
        ),
        float(
            bond_type
            == Chem.rdchem.BondType.AROMATIC
        ),
        float(
            bond.GetIsConjugated()
        ),
        float(
            bond.IsInRing()
        ),
    ]

# ============================================================
# GRAPHDTA DATA LOADING
# ============================================================

def load_json_dictionary(file_path: str | Path) -> dict:
    """Load a JSON dictionary stored in a GraphDTA text file."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_affinity_matrix(file_path: str | Path) -> np.ndarray:
    """Load the pickled GraphDTA affinity matrix."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Affinity matrix not found: {file_path}")

    with file_path.open("rb") as file:
        affinity_matrix = pickle.load(file, encoding="latin1")

    return np.asarray(affinity_matrix, dtype=float)


def load_graphdta_pairs(
    dataset_name: str,
    raw_data_directory: str | Path,
) -> pd.DataFrame:
    """
    Load Davis or KIBA GraphDTA raw files and create pair-level data.

    Expected files:
        ligands_can.txt
        proteins.txt
        Y

    Davis Kd values are converted from nM to pKd.
    KIBA scores are retained without conversion.
    """

    dataset_name = dataset_name.strip().lower()

    if dataset_name not in {"davis", "kiba"}:
        raise ValueError(
            "dataset_name must be either 'davis' or 'kiba'."
        )

    raw_data_directory = Path(raw_data_directory)

    ligands = load_json_dictionary(
        raw_data_directory / "ligands_can.txt"
    )

    proteins = load_json_dictionary(
        raw_data_directory / "proteins.txt"
    )

    affinity_matrix = load_affinity_matrix(
        raw_data_directory / "Y"
    )

    drug_ids = list(ligands.keys())
    target_ids = list(proteins.keys())

    expected_shape = (len(drug_ids), len(target_ids))

    if affinity_matrix.shape != expected_shape:
        raise ValueError(
            f"Affinity matrix shape {affinity_matrix.shape} does not "
            f"match expected shape {expected_shape}."
        )

    records: list[dict] = []

    for drug_index, drug_id in enumerate(drug_ids):
        smiles = ligands[drug_id]

        for target_index, target_id in enumerate(target_ids):
            raw_affinity = affinity_matrix[
                drug_index,
                target_index,
            ]

            # KIBA contains missing drug–target measurements.
            if not np.isfinite(raw_affinity):
                continue

            if dataset_name == "davis":
                model_affinity = kd_nm_to_pkd(raw_affinity)
                affinity_type = "pKd"
            else:
                model_affinity = float(raw_affinity)
                affinity_type = "KIBA_score"

            records.append(
                {
                    "dataset": dataset_name,
                    "drug_id": str(drug_id),
                    "target_id": str(target_id),
                    "smiles": smiles,
                    "protein_sequence": proteins[target_id],
                    "raw_affinity": float(raw_affinity),
                    "affinity": float(model_affinity),
                    "affinity_type": affinity_type,
                }
            )

    pairs_dataframe = pd.DataFrame(records)

    if pairs_dataframe.empty:
        raise ValueError(
            f"No valid drug–target pairs were loaded for {dataset_name}."
        )

    return pairs_dataframe


# ============================================================
# FEATURE CACHES
# ============================================================

def _check_unique_mapping(
    dataframe: pd.DataFrame,
    identifier_column: str,
    value_column: str,
) -> None:
    """Ensure that each identifier maps to only one value."""

    value_counts = dataframe.groupby(identifier_column)[
        value_column
    ].nunique()

    inconsistent_ids = value_counts[value_counts > 1]

    if not inconsistent_ids.empty:
        raise ValueError(
            f"Some {identifier_column} values map to multiple "
            f"{value_column} values."
        )


def build_drug_graph_cache(
    dataframe: pd.DataFrame,
    drug_id_column: str = "drug_id",
    smiles_column: str = "smiles",
) -> Dict[str, Data]:
    """Create one molecular graph for every unique drug."""

    _check_unique_mapping(
        dataframe,
        drug_id_column,
        smiles_column,
    )

    unique_drugs = dataframe[
        [drug_id_column, smiles_column]
    ].drop_duplicates(subset=[drug_id_column])

    graph_cache: Dict[str, Data] = {}

    for row in unique_drugs.itertuples(index=False):
        drug_id = str(getattr(row, drug_id_column))
        smiles = getattr(row, smiles_column)

        graph_cache[drug_id] = smiles_to_graph(smiles)

    return graph_cache


def build_protein_tensor_cache(
    dataframe: pd.DataFrame,
    target_id_column: str = "target_id",
    sequence_column: str = "protein_sequence",
    max_length: int = MAX_PROTEIN_LENGTH,
) -> Dict[str, torch.Tensor]:
    """Create one encoded tensor for every unique protein target."""

    _check_unique_mapping(
        dataframe,
        target_id_column,
        sequence_column,
    )

    unique_targets = dataframe[
        [target_id_column, sequence_column]
    ].drop_duplicates(subset=[target_id_column])

    protein_cache: Dict[str, torch.Tensor] = {}

    for row in unique_targets.itertuples(index=False):
        target_id = str(getattr(row, target_id_column))
        sequence = getattr(row, sequence_column)

        protein_cache[target_id] = protein_sequence_to_tensor(
            sequence,
            max_length=max_length,
        )

    return protein_cache


# ============================================================
# VALIDATION AND SAVING
# ============================================================

def validate_pairs_dataframe(
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    """Validate the processed drug–target pair dataframe."""

    required_columns = {
        "drug_id",
        "target_id",
        "smiles",
        "protein_sequence",
        "affinity",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    duplicate_pairs = dataframe.duplicated(
        subset=["drug_id", "target_id"]
    ).sum()

    empty_sequences = (
        dataframe["protein_sequence"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    non_finite_affinities = (
        ~np.isfinite(
            pd.to_numeric(
                dataframe["affinity"],
                errors="coerce",
            )
        )
    ).sum()

    invalid_smiles = 0

    for smiles in dataframe["smiles"].dropna().unique():
        if Chem.MolFromSmiles(str(smiles)) is None:
            invalid_smiles += 1

    return {
        "rows": int(len(dataframe)),
        "unique_drugs": int(dataframe["drug_id"].nunique()),
        "unique_targets": int(dataframe["target_id"].nunique()),
        "duplicate_pairs": int(duplicate_pairs),
        "invalid_smiles": int(invalid_smiles),
        "empty_protein_sequences": int(empty_sequences),
        "non_finite_affinities": int(non_finite_affinities),
    }


def save_pairs_dataframe(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save a processed drug–target pair dataframe as CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(output_path, index=False)
