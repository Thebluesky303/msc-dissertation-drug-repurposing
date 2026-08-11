"""PyTorch Geometric dataset for drug–target affinity prediction."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from config import MAX_PROTEIN_LENGTH
from src.preprocessing import validate_graph_tensors


class DTADataset(Dataset):
    """
    Dataset containing drug graphs, protein sequences and affinity labels.

    Each returned PyTorch Geometric Data object contains:

        x
            Drug atom features.

        edge_index
            Drug molecular graph edges.

        target
            Encoded protein sequence with shape [1, max_length].

        y
            Affinity value with shape [1].

        sample_id
            Position of the sample in the dataset dataframe.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        drug_graph_cache: Dict[str, Data],
        protein_tensor_cache: Dict[str, torch.Tensor],
        drug_id_column: str = "drug_id",
        target_id_column: str = "target_id",
        affinity_column: str | None = "affinity",
    ) -> None:
        super().__init__()

        self.dataframe = dataframe.reset_index(drop=True).copy()

        self.drug_graph_cache = drug_graph_cache
        self.protein_tensor_cache = protein_tensor_cache

        self.drug_id_column = drug_id_column
        self.target_id_column = target_id_column
        self.affinity_column = affinity_column

        self._validate_dataset()

    def _validate_dataset(self) -> None:
        """Validate dataframe columns, caches and affinity labels."""

        required_columns = {
            self.drug_id_column,
            self.target_id_column,
        }

        if self.affinity_column is not None:
            required_columns.add(self.affinity_column)

        missing_columns = required_columns.difference(
            self.dataframe.columns
        )

        if missing_columns:
            raise ValueError(
                f"Missing dataset columns: {sorted(missing_columns)}"
            )

        dataframe_drug_ids = set(
            self.dataframe[self.drug_id_column]
            .astype(str)
            .unique()
        )

        dataframe_target_ids = set(
            self.dataframe[self.target_id_column]
            .astype(str)
            .unique()
        )

        missing_drug_graphs = dataframe_drug_ids.difference(
            self.drug_graph_cache.keys()
        )

        missing_protein_tensors = dataframe_target_ids.difference(
            self.protein_tensor_cache.keys()
        )

        if missing_drug_graphs:
            examples = sorted(missing_drug_graphs)[:5]

            raise KeyError(
                "Drug graph cache is missing IDs such as: "
                f"{examples}"
            )

        if missing_protein_tensors:
            examples = sorted(missing_protein_tensors)[:5]

            raise KeyError(
                "Protein tensor cache is missing IDs such as: "
                f"{examples}"
            )

        for drug_id in dataframe_drug_ids:
            validate_graph_tensors(self.drug_graph_cache[drug_id])

        for target_id in dataframe_target_ids:
            tensor = self.protein_tensor_cache[target_id]
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1:
                raise ValueError(
                    f"Protein tensor for {target_id} must have shape [length]."
                )
            if tensor.dtype not in (torch.int32, torch.int64):
                raise TypeError(
                    f"Protein tensor for {target_id} must contain integer tokens."
                )

        if "pair_id" in self.dataframe.columns:
            if self.dataframe["pair_id"].astype(str).duplicated().any():
                raise ValueError("pair_id values must be unique within a dataset.")

        if self.affinity_column is not None:
            affinity_values = pd.to_numeric(
                self.dataframe[self.affinity_column],
                errors="coerce",
            )

            if affinity_values.isna().any():
                raise ValueError(
                    "Affinity column contains missing or "
                    "non-numeric values."
                )

            if not np.isfinite(affinity_values.to_numpy()).all():
                raise ValueError(
                    "Affinity column contains non-finite values."
                )

    def len(self) -> int:
        """Return the number of drug–target pairs."""

        return len(self.dataframe)

    def get(self, index: int) -> Data:
        """Return one drug–target pair as a PyG Data object."""

        row = self.dataframe.iloc[index]

        drug_id = str(row[self.drug_id_column])
        target_id = str(row[self.target_id_column])

        drug_graph = self.drug_graph_cache[drug_id].clone()

        protein_tensor = self.protein_tensor_cache[
            target_id
        ].clone()

        drug_graph.target = protein_tensor.unsqueeze(0)

        drug_graph.sample_id = torch.tensor(
            [index],
            dtype=torch.long,
        )

        if "pair_id" in self.dataframe.columns:
            drug_graph.pair_id = str(row["pair_id"])

        if self.affinity_column is not None:
            affinity_value = float(row[self.affinity_column])

            drug_graph.y = torch.tensor(
                [affinity_value],
                dtype=torch.float32,
            )

        return drug_graph


class AdvancedDTADataset(Dataset):
    """Dataset for GINE plus cached residue-level ESM-2 embeddings.

    Each graph receives a padded ``protein_embeddings`` tensor with shape
    ``[1, max_length, esm_dim]`` and a boolean valid-residue mask with shape
    ``[1, max_length]``.  PyG concatenates that leading dimension when it
    forms a mini-batch, producing the model contract ``[B, L, E]`` and
    ``[B, L]`` without running ESM-2 during training.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        drug_graph_cache: Mapping[str, Data],
        residue_embedding_cache: Mapping[str, torch.Tensor],
        *,
        max_length: int = MAX_PROTEIN_LENGTH,
        esm_embedding_dim: int | None = None,
        drug_id_column: str = "drug_id",
        target_id_column: str = "target_id",
        affinity_column: str | None = "affinity",
    ) -> None:
        super().__init__()
        if max_length < 1:
            raise ValueError("max_length must be positive.")
        if dataframe.empty:
            raise ValueError("AdvancedDTADataset cannot be empty.")

        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.drug_graph_cache = drug_graph_cache
        self.residue_embedding_cache = residue_embedding_cache
        self.max_length = int(max_length)
        self.drug_id_column = drug_id_column
        self.target_id_column = target_id_column
        self.affinity_column = affinity_column
        self.esm_embedding_dim = self._validate_dataset(esm_embedding_dim)

    def _validate_dataset(self, configured_dim: int | None) -> int:
        required = {self.drug_id_column, self.target_id_column}
        if self.affinity_column is not None:
            required.add(self.affinity_column)
        missing_columns = required.difference(self.dataframe.columns)
        if missing_columns:
            raise ValueError(f"Missing dataset columns: {sorted(missing_columns)}")

        drug_ids = set(self.dataframe[self.drug_id_column].astype(str))
        target_ids = set(self.dataframe[self.target_id_column].astype(str))
        missing_drugs = drug_ids.difference(self.drug_graph_cache)
        missing_targets = target_ids.difference(self.residue_embedding_cache)
        if missing_drugs:
            raise KeyError(
                "Drug graph cache is missing IDs such as: "
                f"{sorted(missing_drugs)[:5]}"
            )
        if missing_targets:
            raise KeyError(
                "Residue embedding cache is missing IDs such as: "
                f"{sorted(missing_targets)[:5]}"
            )

        dimensions: set[int] = set()
        for drug_id in drug_ids:
            validate_graph_tensors(self.drug_graph_cache[drug_id])
        for target_id in target_ids:
            embeddings = self.residue_embedding_cache[target_id]
            if not isinstance(embeddings, torch.Tensor):
                raise TypeError(f"Residue cache value for {target_id} is not a tensor.")
            if embeddings.ndim != 2 or embeddings.shape[0] < 1:
                raise ValueError(
                    f"Residue cache value for {target_id} must have shape [L, E]."
                )
            if not embeddings.is_floating_point():
                raise TypeError(
                    f"Residue cache value for {target_id} must be floating point."
                )
            if not torch.isfinite(embeddings).all():
                raise ValueError(
                    f"Residue cache value for {target_id} contains non-finite values."
                )
            dimensions.add(int(embeddings.shape[1]))

        if len(dimensions) != 1:
            raise ValueError("All cached residue embeddings must share one feature size.")
        inferred_dim = dimensions.pop()
        if configured_dim is not None and int(configured_dim) != inferred_dim:
            raise ValueError(
                f"Configured ESM dimension {configured_dim} does not match cache "
                f"dimension {inferred_dim}."
            )

        if self.affinity_column is not None:
            affinities = pd.to_numeric(
                self.dataframe[self.affinity_column], errors="coerce"
            ).to_numpy()
            if not np.isfinite(affinities).all():
                raise ValueError("Affinity column contains non-finite values.")
        if "pair_id" in self.dataframe.columns:
            if self.dataframe["pair_id"].astype(str).duplicated().any():
                raise ValueError("pair_id values must be unique within a dataset.")
        return inferred_dim

    def len(self) -> int:
        """Return the number of drug-target pairs."""

        return len(self.dataframe)

    def get(self, index: int) -> Data:
        """Return one graph with padded cached protein inputs."""

        row = self.dataframe.iloc[index]
        drug_id = str(row[self.drug_id_column])
        target_id = str(row[self.target_id_column])
        graph = self.drug_graph_cache[drug_id].clone()
        residues = self.residue_embedding_cache[target_id][
            : self.max_length
        ].detach().clone().float()

        padded = torch.zeros(
            (self.max_length, self.esm_embedding_dim), dtype=torch.float32
        )
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        length = int(residues.shape[0])
        padded[:length] = residues
        mask[:length] = True
        graph.protein_embeddings = padded.unsqueeze(0)
        graph.protein_mask = mask.unsqueeze(0)
        graph.sample_id = torch.tensor([index], dtype=torch.long)
        if "pair_id" in self.dataframe.columns:
            graph.pair_id = str(row["pair_id"])
        if self.affinity_column is not None:
            graph.y = torch.tensor(
                [float(row[self.affinity_column])], dtype=torch.float32
            )
        return graph
