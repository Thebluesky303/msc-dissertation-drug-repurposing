"""Data preparation functions for drug-target affinity experiments."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch.nn.utils.rnn import pad_sequence

from config import MAX_PROTEIN_LENGTH, RANDOM_SEED


# ---------------------------------------------------------------------
# Protein sequence encoding
# ---------------------------------------------------------------------

# 20 standard amino acids plus common ambiguous or rare symbols.
PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYBXZOU"
AMINO_ACID_TO_INDEX = {
    amino_acid: index + 1
    for index, amino_acid in enumerate(PROTEIN_ALPHABET)
}

PROTEIN_VOCAB_SIZE = len(AMINO_ACID_TO_INDEX) + 1  # +1 for padding


# ---------------------------------------------------------------------
# Molecular graph feature sizes
# ---------------------------------------------------------------------

ATOM_FEATURE_DIM = 9
BOND_FEATURE_DIM = 6


def clean_protein_sequence(sequence: str) -> str:
    """Return an uppercase protein sequence without spaces."""

    if not isinstance(sequence, str):
        raise TypeError("Protein sequence must be a string.")

    return "".join(sequence.upper().split())


def encode_protein_sequence(
    sequence: str,
    max_length: int = MAX_PROTEIN_LENGTH,
) -> torch.Tensor:
    """
    Encode a protein sequence as integer indices.

    Index 0 is reserved for padding. Unknown amino-acid symbols are also
    encoded as 0 so the input remains safe for the embedding layer.
    """

    sequence = clean_protein_sequence(sequence)

    encoded = torch.zeros(max_length, dtype=torch.long)

    for position, amino_acid in enumerate(sequence[:max_length]):
        encoded[position] = AMINO_ACID_TO_INDEX.get(amino_acid, 0)

    return encoded


# ---------------------------------------------------------------------
# Drug molecular graphs
# ---------------------------------------------------------------------

def atom_to_feature_vector(atom: Chem.Atom) -> list[float]:
    """Create a small numerical feature vector for one atom."""

    hybridization = atom.GetHybridization()

    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetTotalDegree() / 6.0,
        float(atom.GetFormalCharge()),
        atom.GetTotalNumHs() / 4.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(hybridization == Chem.rdchem.HybridizationType.SP),
        float(hybridization == Chem.rdchem.HybridizationType.SP2),
        float(hybridization == Chem.rdchem.HybridizationType.SP3),
    ]


def bond_to_feature_vector(bond: Chem.Bond) -> list[float]:
    """Create a bond feature vector used by the GINE model."""

    bond_type = bond.GetBondType()

    return [
        float(bond_type == Chem.rdchem.BondType.SINGLE),
        float(bond_type == Chem.rdchem.BondType.DOUBLE),
        float(bond_type == Chem.rdchem.BondType.TRIPLE),
        float(bond_type == Chem.rdchem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def smiles_to_graph(smiles: str) -> Data:
    """
    Convert a SMILES string into a PyTorch Geometric molecular graph.

    Atoms become nodes and chemical bonds become bidirectional edges.
    Bond features are stored in edge_attr so the same graph can be used
    by GCN, GAT, GIN, and GINE models.
    """

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")

    node_features = [
        atom_to_feature_vector(atom)
        for atom in molecule.GetAtoms()
    ]

    edge_indices = []
    edge_features = []

    for bond in molecule.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_to_feature_vector(bond)

        # PyTorch Geometric represents an undirected bond using two edges.
        edge_indices.append([start, end])
        edge_indices.append([end, start])

        edge_features.append(features)
        edge_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)

    if edge_indices:
        edge_index = torch.tensor(
            edge_indices,
            dtype=torch.long,
        ).t().contiguous()

        edge_attr = torch.tensor(
            edge_features,
            dtype=torch.float,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty(
            (0, BOND_FEATURE_DIM),
            dtype=torch.float,
        )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def _read_json_file(file_path: Path) -> dict:
    """Read a JSON dictionary from disk."""

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _load_affinity_matrix(file_path: Path) -> np.ndarray:
    """Load the affinity matrix used by the standard DAVIS/KIBA files."""

    with open(file_path, "rb") as file:
        try:
            matrix = pickle.load(file)
        except UnicodeDecodeError:
            file.seek(0)
            matrix = pickle.load(file, encoding="latin1")

    return np.asarray(matrix, dtype=float)


def kd_to_pkd(values) -> np.ndarray:
    """
    Convert dissociation constants in nM to pKd.

    pKd = -log10(Kd in molar units)
    """

    values = np.asarray(values, dtype=float)

    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & (values > 0)

    result[valid] = 9.0 - np.log10(values[valid])

    return result


def load_deepdta_dataset(
    dataset_dir: Path | str,
    dataset_name: str,
    transform_davis: bool = True,
) -> pd.DataFrame:
    """
    Load DAVIS or KIBA data stored in the common DeepDTA format.

    Expected files:
        ligands_can.txt
        proteins.txt
        Y

    The returned table contains:
        drug_id
        target_id
        smiles
        protein_sequence
        affinity
    """

    dataset_dir = Path(dataset_dir)
    dataset_name = dataset_name.lower().strip()

    ligand_file = dataset_dir / "ligands_can.txt"
    protein_file = dataset_dir / "proteins.txt"
    affinity_file = dataset_dir / "Y"

    required_files = [
        ligand_file,
        protein_file,
        affinity_file,
    ]

    missing_files = [
        file.name
        for file in required_files
        if not file.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing files in {dataset_dir}: {missing_files}"
        )

    ligands = _read_json_file(ligand_file)
    proteins = _read_json_file(protein_file)
    affinity_matrix = _load_affinity_matrix(affinity_file)

    drug_ids = list(ligands.keys())
    target_ids = list(proteins.keys())

    expected_shape = (
        len(drug_ids),
        len(target_ids),
    )

    if affinity_matrix.shape != expected_shape:
        raise ValueError(
            "Affinity matrix shape does not match the number of drugs "
            f"and targets. Expected {expected_shape}, "
            f"found {affinity_matrix.shape}."
        )

    if dataset_name == "davis" and transform_davis:
        affinity_matrix = kd_to_pkd(affinity_matrix)

    rows = []

    for drug_index, drug_id in enumerate(drug_ids):
        smiles = ligands[drug_id]

        for target_index, target_id in enumerate(target_ids):
            affinity = affinity_matrix[
                drug_index,
                target_index,
            ]

            if not np.isfinite(affinity):
                continue

            rows.append(
                {
                    "drug_id": str(drug_id),
                    "target_id": str(target_id),
                    "smiles": smiles,
                    "protein_sequence": proteins[target_id],
                    "affinity": float(affinity),
                }
            )

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        raise ValueError(
            f"No valid interactions were loaded from {dataset_dir}."
        )

    return dataframe.reset_index(drop=True)


def load_interaction_csv(
    file_path: Path | str,
) -> pd.DataFrame:
    """
    Load an already prepared interaction CSV.

    Required columns:
        drug_id
        target_id
        smiles
        protein_sequence
        affinity
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    dataframe = pd.read_csv(file_path)

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
            "Interaction CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe[
        [
            "drug_id",
            "target_id",
            "smiles",
            "protein_sequence",
            "affinity",
        ]
    ].copy()

    dataframe["drug_id"] = dataframe["drug_id"].astype(str)
    dataframe["target_id"] = dataframe["target_id"].astype(str)
    dataframe["affinity"] = pd.to_numeric(
        dataframe["affinity"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=[
            "smiles",
            "protein_sequence",
            "affinity",
        ]
    )

    return dataframe.reset_index(drop=True)


# ---------------------------------------------------------------------
# Cached drug and protein representations
# ---------------------------------------------------------------------

def _check_unique_mapping(
    dataframe: pd.DataFrame,
    id_column: str,
    value_column: str,
) -> None:
    """Check that each identifier maps to only one value."""

    mapping_counts = (
        dataframe[[id_column, value_column]]
        .drop_duplicates()
        .groupby(id_column)[value_column]
        .nunique()
    )

    inconsistent = mapping_counts[
        mapping_counts > 1
    ]

    if not inconsistent.empty:
        examples = inconsistent.index.astype(str).tolist()[:5]

        raise ValueError(
            f"Some {id_column} values map to more than one "
            f"{value_column}. Examples: {examples}"
        )


def build_drug_graph_cache(
    dataframe: pd.DataFrame,
    drug_id_column: str = "drug_id",
    smiles_column: str = "smiles",
) -> dict[str, Data]:
    """Create one molecular graph for every unique drug."""

    _check_unique_mapping(
        dataframe,
        drug_id_column,
        smiles_column,
    )

    unique_drugs = dataframe[
        [drug_id_column, smiles_column]
    ].drop_duplicates()

    graph_cache = {}

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
) -> dict[str, torch.Tensor]:
    """Create one encoded tensor for every unique protein target."""

    _check_unique_mapping(
        dataframe,
        target_id_column,
        sequence_column,
    )

    unique_targets = dataframe[
        [target_id_column, sequence_column]
    ].drop_duplicates()

    tensor_cache = {}

    for row in unique_targets.itertuples(index=False):
        target_id = str(getattr(row, target_id_column))
        sequence = getattr(row, sequence_column)

        tensor_cache[target_id] = encode_protein_sequence(
            sequence,
            max_length=max_length,
        )

    return tensor_cache


# ---------------------------------------------------------------------
# Train, validation, and test splits
# ---------------------------------------------------------------------

def _validate_split_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    """Check that split ratios are valid."""

    ratios = [
        train_ratio,
        validation_ratio,
        test_ratio,
    ]

    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("All split ratios must be greater than zero.")

    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must add up to 1.0.")


def random_split(
    dataframe: pd.DataFrame,
    train_ratio: float = 0.80,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a reproducible random interaction split."""

    _validate_split_ratios(
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    shuffled = dataframe.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    validation_end = train_end + int(
        total * validation_ratio
    )

    train_df = shuffled.iloc[:train_end].copy()
    validation_df = shuffled.iloc[
        train_end:validation_end
    ].copy()
    test_df = shuffled.iloc[
        validation_end:
    ].copy()

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _group_split(
    dataframe: pd.DataFrame,
    group_column: str,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split interactions by unique drugs or unique targets."""

    _validate_split_ratios(
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    groups = dataframe[
        group_column
    ].drop_duplicates().to_numpy()

    rng = np.random.default_rng(seed)
    groups = rng.permutation(groups)

    total_groups = len(groups)

    if total_groups < 3:
        raise ValueError(
            f"At least three unique values are required in {group_column}."
        )

    train_end = max(1, int(total_groups * train_ratio))
    validation_count = max(
        1,
        int(total_groups * validation_ratio),
    )
    validation_end = min(
        train_end + validation_count,
        total_groups - 1,
    )

    train_groups = set(groups[:train_end])
    validation_groups = set(
        groups[train_end:validation_end]
    )
    test_groups = set(groups[validation_end:])

    train_df = dataframe[
        dataframe[group_column].isin(train_groups)
    ].copy()

    validation_df = dataframe[
        dataframe[group_column].isin(validation_groups)
    ].copy()

    test_df = dataframe[
        dataframe[group_column].isin(test_groups)
    ].copy()

    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError(
            f"The {group_column} split produced an empty partition."
        )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def cold_drug_split(
    dataframe: pd.DataFrame,
    train_ratio: float = 0.80,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the data so drugs do not overlap between partitions."""

    return _group_split(
        dataframe=dataframe,
        group_column="drug_id",
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


def cold_target_split(
    dataframe: pd.DataFrame,
    train_ratio: float = 0.80,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the data so protein targets do not overlap between partitions."""

    return _group_split(
        dataframe=dataframe,
        group_column="target_id",
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


def check_cold_split(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    column: str,
) -> bool:
    """Return True when the selected identifiers do not overlap."""

    train_ids = set(train_df[column])
    validation_ids = set(validation_df[column])
    test_ids = set(test_df[column])

    return (
        train_ids.isdisjoint(validation_ids)
        and train_ids.isdisjoint(test_ids)
        and validation_ids.isdisjoint(test_ids)
    )


# ---------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------

class InteractionDataset(Dataset):
    """
    Pair cached drug graphs with protein tensors and affinity values.

    Each item is returned as a PyTorch Geometric Data object. This allows
    torch_geometric.loader.DataLoader to batch graphs, proteins, and labels
    together.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        drug_graph_cache: dict[str, Data],
        protein_tensor_cache: dict[str, torch.Tensor],
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.drug_graph_cache = drug_graph_cache
        self.protein_tensor_cache = protein_tensor_cache

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Data:
        row = self.dataframe.iloc[index]

        drug_id = str(row["drug_id"])
        target_id = str(row["target_id"])

        if drug_id not in self.drug_graph_cache:
            raise KeyError(
                f"Drug graph not found for drug_id: {drug_id}"
            )

        if target_id not in self.protein_tensor_cache:
            raise KeyError(
                f"Protein tensor not found for target_id: {target_id}"
            )

        graph = self.drug_graph_cache[drug_id].clone()

        # Shape [1, sequence_length] gives [batch, sequence_length]
        # after PyTorch Geometric batching.
        graph.protein = self.protein_tensor_cache[
            target_id
        ].unsqueeze(0)

        graph.y = torch.tensor(
            [float(row["affinity"])],
            dtype=torch.float,
        )

        return graph

# ---------------------------------------------------------------------
# Frozen ESM-2 protein embeddings
# ---------------------------------------------------------------------

def build_esm_embedding_cache(
    dataframe: pd.DataFrame,
    tokenizer,
    esm_model,
    device: torch.device,
    target_id_column: str = "target_id",
    sequence_column: str = "protein_sequence",
    max_length: int = MAX_PROTEIN_LENGTH,
    batch_size: int = 8,
    storage_dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """
    Create one frozen ESM-2 residue embedding tensor per unique protein.

    The returned dictionary maps each target ID to a tensor with shape
    [protein_length, esm_embedding_dim].

    ESM-2 is used only as a frozen feature extractor. Special tokens and
    padding tokens are removed before the residue embeddings are cached.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    _check_unique_mapping(
        dataframe,
        target_id_column,
        sequence_column,
    )

    unique_targets = (
        dataframe[
            [target_id_column, sequence_column]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if unique_targets.empty:
        raise ValueError(
            "No protein sequences are available for ESM-2 embedding."
        )

    esm_model = esm_model.to(device)
    esm_model.eval()

    for parameter in esm_model.parameters():
        parameter.requires_grad = False

    embedding_cache = {}

    total_targets = len(unique_targets)

    with torch.inference_mode():
        for start in range(0, total_targets, batch_size):
            batch_df = unique_targets.iloc[
                start : start + batch_size
            ]

            target_ids = (
                batch_df[target_id_column]
                .astype(str)
                .tolist()
            )

            sequences = [
                clean_protein_sequence(sequence)[:max_length]
                for sequence in batch_df[sequence_column]
            ]

            tokens = tokenizer(
                sequences,
                padding=True,
                truncation=True,
                max_length=max_length + 2,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )

            special_tokens_mask = tokens.pop(
                "special_tokens_mask"
            )

            model_inputs = {
                key: value.to(device)
                for key, value in tokens.items()
            }

            outputs = esm_model(**model_inputs)

            hidden_states = (
                outputs.last_hidden_state
                .detach()
                .cpu()
            )

            attention_mask = (
                tokens["attention_mask"]
                .detach()
                .cpu()
                .bool()
            )

            special_tokens_mask = (
                special_tokens_mask
                .detach()
                .cpu()
                .bool()
            )

            for index, target_id in enumerate(target_ids):
                residue_mask = (
                    attention_mask[index]
                    & ~special_tokens_mask[index]
                )

                residue_embeddings = hidden_states[
                    index
                ][residue_mask]

                if residue_embeddings.size(0) == 0:
                    raise ValueError(
                        "No residue embeddings were created for "
                        f"target_id: {target_id}"
                    )

                embedding_cache[target_id] = (
                    residue_embeddings
                    .to(dtype=storage_dtype)
                    .contiguous()
                )

            completed = min(
                start + batch_size,
                total_targets,
            )

            print(
                f"ESM-2 embeddings: "
                f"{completed}/{total_targets}"
            )

    return embedding_cache


def save_esm_embedding_cache(
    embedding_cache: dict[str, torch.Tensor],
    file_path: Path | str,
) -> None:
    """Save cached ESM-2 residue embeddings to disk."""

    file_path = Path(file_path)
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cpu_cache = {
        str(target_id): embedding.detach().cpu()
        for target_id, embedding in embedding_cache.items()
    }

    torch.save(
        cpu_cache,
        file_path,
    )


def load_esm_embedding_cache(
    file_path: Path | str,
) -> dict[str, torch.Tensor]:
    """Load cached ESM-2 residue embeddings from disk."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"ESM-2 embedding cache not found: {file_path}"
        )

    cache = torch.load(
        file_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(cache, dict):
        raise ValueError(
            "The ESM-2 embedding cache must be a dictionary."
        )

    return {
        str(target_id): embedding
        for target_id, embedding in cache.items()
    }


def check_esm_embedding_cache(
    dataframe: pd.DataFrame,
    embedding_cache: dict[str, torch.Tensor],
    target_id_column: str = "target_id",
) -> bool:
    """Return True when every target has a valid cached embedding."""

    required_targets = set(
        dataframe[target_id_column]
        .astype(str)
        .unique()
    )

    cached_targets = set(
        str(target_id)
        for target_id in embedding_cache
    )

    if not required_targets.issubset(cached_targets):
        return False

    for target_id in required_targets:
        embedding = embedding_cache[target_id]

        if (
            not isinstance(embedding, torch.Tensor)
            or embedding.ndim != 2
            or embedding.size(0) == 0
            or embedding.size(1) == 0
        ):
            return False

    return True


# ---------------------------------------------------------------------
# Advanced dataset for GINE + frozen ESM-2
# ---------------------------------------------------------------------

class AdvancedInteractionDataset(Dataset):
    """
    Pair drug graphs with cached ESM-2 residue embeddings.

    This dataset is used by the advanced model. Variable-length protein
    embeddings are padded only when a batch is created.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        drug_graph_cache: dict[str, Data],
        protein_embedding_cache: dict[str, torch.Tensor],
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.drug_graph_cache = drug_graph_cache
        self.protein_embedding_cache = protein_embedding_cache

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict:
        row = self.dataframe.iloc[index]

        drug_id = str(row["drug_id"])
        target_id = str(row["target_id"])

        if drug_id not in self.drug_graph_cache:
            raise KeyError(
                f"Drug graph not found for drug_id: {drug_id}"
            )

        if target_id not in self.protein_embedding_cache:
            raise KeyError(
                "ESM-2 embedding not found for "
                f"target_id: {target_id}"
            )

        graph = self.drug_graph_cache[
            drug_id
        ].clone()

        protein_embedding = (
            self.protein_embedding_cache[
                target_id
            ]
            .detach()
            .cpu()
        )

        return {
            "graph": graph,
            "protein_embedding": protein_embedding,
            "affinity": float(row["affinity"]),
            "drug_id": drug_id,
            "target_id": target_id,
        }


def advanced_collate_fn(items: list[dict]) -> dict:
    """
    Create one batch for the advanced model.

    Molecular graphs are batched with PyTorch Geometric. Protein residue
    embeddings are padded to the longest sequence in the current batch,
    and a boolean mask identifies the real residues.
    """

    if not items:
        raise ValueError("Cannot create a batch from an empty list.")

    graph_batch = Batch.from_data_list(
        [
            item["graph"]
            for item in items
        ]
    )

    protein_embeddings = [
        item["protein_embedding"].float()
        for item in items
    ]

    protein_lengths = torch.tensor(
        [
            embedding.size(0)
            for embedding in protein_embeddings
        ],
        dtype=torch.long,
    )

    padded_embeddings = pad_sequence(
        protein_embeddings,
        batch_first=True,
        padding_value=0.0,
    )

    max_length = padded_embeddings.size(1)

    protein_mask = (
        torch.arange(max_length)
        .unsqueeze(0)
        < protein_lengths.unsqueeze(1)
    )

    affinities = torch.tensor(
        [
            item["affinity"]
            for item in items
        ],
        dtype=torch.float,
    )

    return {
        "graph": graph_batch,
        "protein_embeddings": padded_embeddings,
        "protein_mask": protein_mask,
        "y": affinities,
        "drug_ids": [
            item["drug_id"]
            for item in items
        ],
        "target_ids": [
            item["target_id"]
            for item in items
        ],
    }

