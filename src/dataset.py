"""PyTorch Geometric dataset for drug–target affinity prediction."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset


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

        sample_index
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

        # drug_graph.sample_index = torch.tensor(
        #     [index],
        #     dtype=torch.long,
        # )
        drug_graph.sample_id = torch.tensor(
            [index],
            dtype=torch.long,
        )

        if self.affinity_column is not None:
            affinity_value = float(row[self.affinity_column])

            drug_graph.y = torch.tensor(
                [affinity_value],
                dtype=torch.float32,
            )

        return drug_graph
