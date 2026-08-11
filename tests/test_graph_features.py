"""Molecular graph and GINE edge-feature tests."""

from __future__ import annotations

import pytest
import torch

from src.preprocessing import smiles_to_graph, validate_graph_tensors


def test_double_bond_is_bidirectional_and_encoded() -> None:
    graph = smiles_to_graph("C=C")
    assert graph.x.shape == (2, 6)
    assert graph.edge_index.shape == (2, 2)
    assert graph.edge_attr.shape == (2, 6)
    torch.testing.assert_close(
        graph.edge_attr,
        torch.tensor([[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]).float(),
    )


def test_aromatic_ring_features_are_scientifically_consistent() -> None:
    graph = smiles_to_graph("c1ccccc1")
    assert graph.edge_attr.shape == (12, 6)
    assert torch.all(graph.edge_attr[:, 3] == 1)
    assert torch.all(graph.edge_attr[:, 4] == 1)
    assert torch.all(graph.edge_attr[:, 5] == 1)


def test_single_atom_graph_has_valid_empty_edge_tensors() -> None:
    graph = smiles_to_graph("[He]")
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_attr.shape == (0, 6)
    validate_graph_tensors(graph)


def test_graph_validator_rejects_edge_attribute_mismatch() -> None:
    graph = smiles_to_graph("CC")
    graph.edge_attr = graph.edge_attr[:, :-1]
    with pytest.raises(ValueError, match="6 columns"):
        validate_graph_tensors(graph)
