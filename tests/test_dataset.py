"""Tests for baseline and cached-residue PyG datasets."""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader

from src.dataset import AdvancedDTADataset, DTADataset
from src.preprocessing import smiles_to_graph


def _pairs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_id": ["demo::D1::T1", "demo::D2::T2"],
            "drug_id": ["D1", "D2"],
            "target_id": ["T1", "T2"],
            "affinity": [7.0, 8.0],
        }
    )


def test_baseline_dataset_batches_token_sequences() -> None:
    dataset = DTADataset(
        _pairs(),
        {"D1": smiles_to_graph("CC"), "D2": smiles_to_graph("O")},
        {"T1": torch.tensor([1, 2, 0]), "T2": torch.tensor([3, 4, 5])},
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    assert batch.target.shape == (2, 3)
    assert batch.y.shape == (2,)
    assert batch.sample_id.tolist() == [0, 1]


def test_advanced_dataset_builds_boolean_residue_masks() -> None:
    dataset = AdvancedDTADataset(
        _pairs(),
        {"D1": smiles_to_graph("CC"), "D2": smiles_to_graph("O")},
        {"T1": torch.ones(2, 8), "T2": torch.ones(4, 8)},
        max_length=3,
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    assert batch.protein_embeddings.shape == (2, 3, 8)
    assert batch.protein_mask.dtype == torch.bool
    assert batch.protein_mask.tolist() == [
        [True, True, False],
        [True, True, True],
    ]
    assert dataset.esm_embedding_dim == 8


def test_advanced_dataset_rejects_nonfinite_cache() -> None:
    bad = torch.ones(2, 8)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        AdvancedDTADataset(
            _pairs().iloc[:1],
            {"D1": smiles_to_graph("CC")},
            {"T1": bad},
            max_length=3,
        )
