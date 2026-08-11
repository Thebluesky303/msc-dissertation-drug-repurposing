"""Fast synthetic tests for baseline and advanced DTA models."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from src.models import (
    AdvancedDTAModel,
    BidirectionalCrossAttention,
    ResidualGINEEncoder,
    ResidueCNNEncoder,
    create_dta_model,
    masked_mean_pool,
    pack_atom_embeddings,
)


torch.set_num_threads(1)


def _graph_tensors() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Return two small molecular graphs in one PyG-style batch."""

    generator = torch.Generator().manual_seed(42)
    x = torch.randn(5, 6, generator=generator)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 3, 4],
            [1, 0, 2, 1, 4, 3],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.randn(6, 6, generator=generator)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    return x, edge_index, edge_attr, batch


def _baseline_batch() -> SimpleNamespace:
    x, edge_index, edge_attr, batch = _graph_tensors()
    target = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0],
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        ],
        dtype=torch.long,
    )
    return SimpleNamespace(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        target=target,
    )


def _advanced_inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    x, edge_index, edge_attr, batch = _graph_tensors()
    generator = torch.Generator().manual_seed(123)
    protein_embeddings = torch.randn(2, 5, 8, generator=generator)
    protein_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )
    return (
        x,
        edge_index,
        edge_attr,
        batch,
        protein_embeddings,
        protein_mask,
    )


def _small_advanced_model() -> AdvancedDTAModel:
    return AdvancedDTAModel(
        atom_input_dim=6,
        bond_feature_dim=6,
        esm_embedding_dim=8,
        embedding_dim=16,
        attention_heads=4,
        attention_feedforward_dim=32,
        regression_hidden_dim=32,
        regression_bottleneck_dim=16,
        dropout=0.0,
    )


class BaselineModelTests(unittest.TestCase):
    """Protect the verified v1 baseline model interfaces."""

    def test_baseline_factory_forward_shapes(self) -> None:
        batch = _baseline_batch()
        for model_name in ("gcn", "gat", "gin", "gine"):
            with self.subTest(model=model_name):
                model = create_dta_model(model_name, dropout=0.0).eval()
                with torch.no_grad():
                    prediction = model(batch)
                self.assertEqual(prediction.shape, (2,))
                self.assertTrue(torch.isfinite(prediction).all())

    def test_baseline_factory_rejects_unknown_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_name"):
            create_dta_model("not-a-model")


class AdvancedComponentTests(unittest.TestCase):
    """Check advanced encoder, masking and cross-attention contracts."""

    def test_residual_gine_returns_per_atom_embeddings(self) -> None:
        x, edge_index, edge_attr, _ = _graph_tensors()
        encoder = ResidualGINEEncoder(
            input_dim=6, embedding_dim=16, edge_dim=6, dropout=0.0
        ).eval()
        with torch.no_grad():
            output = encoder(x, edge_index, edge_attr)
        self.assertEqual(len(encoder.convolutions), 4)
        self.assertEqual(output.shape, (5, 16))
        self.assertTrue(torch.isfinite(output).all())

    def test_gine_validates_edge_feature_shape(self) -> None:
        x, edge_index, edge_attr, _ = _graph_tensors()
        encoder = ResidualGINEEncoder(
            input_dim=6, embedding_dim=16, edge_dim=6, dropout=0.0
        )
        with self.assertRaisesRegex(ValueError, "requires edge_attr"):
            encoder(x, edge_index, None)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "rows must match"):
            encoder(x, edge_index, edge_attr[:-1])
        with self.assertRaisesRegex(ValueError, "exactly 6"):
            encoder(x, edge_index, edge_attr[:, :-1])

    def test_pack_atom_embeddings_returns_dense_tensor_and_mask(self) -> None:
        atom_embeddings = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        batch = torch.tensor([0, 0, 0, 1, 1])
        dense, mask = pack_atom_embeddings(atom_embeddings, batch)

        self.assertEqual(dense.shape, (2, 3, 4))
        self.assertEqual(mask.dtype, torch.bool)
        torch.testing.assert_close(
            mask,
            torch.tensor([[True, True, True], [True, True, False]]),
        )
        torch.testing.assert_close(dense[0], atom_embeddings[:3])
        torch.testing.assert_close(dense[1, :2], atom_embeddings[3:])
        torch.testing.assert_close(dense[1, 2], torch.zeros(4))

    def test_protein_cnn_preserves_length_and_ignores_padding_values(self) -> None:
        *_, protein_embeddings, protein_mask = _advanced_inputs()
        altered = protein_embeddings.clone()
        altered[~protein_mask] = 1_000_000.0
        encoder = ResidueCNNEncoder(
            input_dim=8, output_dim=16, dropout=0.0
        ).eval()

        with torch.no_grad():
            original_output = encoder(protein_embeddings, protein_mask)
            altered_output = encoder(altered, protein_mask)

        self.assertEqual(len(encoder.convolutions), 3)
        self.assertEqual(original_output.shape, (2, 5, 16))
        torch.testing.assert_close(original_output, altered_output)
        torch.testing.assert_close(
            original_output[~protein_mask], torch.zeros(2, 16)
        )

    def test_cross_attention_masks_keys_and_zeroes_padded_queries(self) -> None:
        generator = torch.Generator().manual_seed(7)
        atoms = torch.randn(2, 3, 16, generator=generator)
        proteins = torch.randn(2, 5, 16, generator=generator)
        atom_mask = torch.tensor(
            [[True, True, True], [True, True, False]]
        )
        protein_mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, True],
            ]
        )
        altered_atoms = atoms.clone()
        altered_proteins = proteins.clone()
        altered_atoms[~atom_mask] = -1_000_000.0
        altered_proteins[~protein_mask] = 1_000_000.0
        block = BidirectionalCrossAttention(
            embedding_dim=16,
            num_heads=4,
            feedforward_dim=32,
            dropout=0.0,
        ).eval()

        with torch.no_grad():
            updated_atoms, updated_proteins = block(
                atoms, proteins, atom_mask, protein_mask
            )
            altered_updated_atoms, altered_updated_proteins = block(
                altered_atoms,
                altered_proteins,
                atom_mask,
                protein_mask,
            )

        self.assertEqual(updated_atoms.shape, atoms.shape)
        self.assertEqual(updated_proteins.shape, proteins.shape)
        torch.testing.assert_close(updated_atoms, altered_updated_atoms)
        torch.testing.assert_close(updated_proteins, altered_updated_proteins)
        torch.testing.assert_close(
            updated_atoms[~atom_mask], torch.zeros(1, 16)
        )
        torch.testing.assert_close(
            updated_proteins[~protein_mask], torch.zeros(2, 16)
        )

    def test_masked_mean_pool_ignores_padding(self) -> None:
        values = torch.tensor(
            [
                [[1.0, 3.0], [3.0, 5.0], [99_999.0, -99_999.0]],
                [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
            ]
        )
        mask = torch.tensor(
            [[True, True, False], [True, True, True]]
        )
        pooled = masked_mean_pool(values, mask)
        expected = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        torch.testing.assert_close(pooled, expected)


class AdvancedModelTests(unittest.TestCase):
    """Exercise the complete advanced model with synthetic cached embeddings."""

    def test_default_regression_head_matches_proposed_architecture(self) -> None:
        model = AdvancedDTAModel(dropout=0.0)
        linear_layers = [
            layer for layer in model.regressor if isinstance(layer, nn.Linear)
        ]
        dimensions = [
            (layer.in_features, layer.out_features) for layer in linear_layers
        ]
        self.assertEqual(dimensions, [(256, 512), (512, 256), (256, 1)])

    def test_forward_data_api_and_intermediate_shapes(self) -> None:
        inputs = _advanced_inputs()
        x, edge_index, edge_attr, batch, proteins, protein_mask = inputs
        data = SimpleNamespace(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            protein_embeddings=proteins,
            protein_mask=protein_mask,
        )
        model = _small_advanced_model().eval()

        with torch.no_grad():
            atoms, residues, atom_mask, returned_protein_mask = (
                model.encode_interactions(*inputs)
            )
            prediction = model(data)

        self.assertEqual(atoms.shape, (2, 3, 16))
        self.assertEqual(residues.shape, (2, 5, 16))
        self.assertEqual(atom_mask.shape, (2, 3))
        torch.testing.assert_close(returned_protein_mask, protein_mask)
        self.assertEqual(prediction.shape, (2,))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_complete_model_ignores_padded_protein_values(self) -> None:
        x, edge_index, edge_attr, batch, proteins, protein_mask = (
            _advanced_inputs()
        )
        altered = proteins.clone()
        altered[~protein_mask] = 1_000_000.0
        model = _small_advanced_model().eval()

        with torch.no_grad():
            original_prediction = model.forward_tensors(
                x,
                edge_index,
                edge_attr,
                batch,
                proteins,
                protein_mask,
            )
            altered_prediction = model.forward_tensors(
                x,
                edge_index,
                edge_attr,
                batch,
                altered,
                protein_mask,
            )

        torch.testing.assert_close(original_prediction, altered_prediction)

    def test_forward_reports_missing_cached_protein_fields(self) -> None:
        x, edge_index, edge_attr, batch = _graph_tensors()
        incomplete = SimpleNamespace(
            x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch
        )
        with self.assertRaisesRegex(AttributeError, "protein_embeddings"):
            _small_advanced_model()(incomplete)


if __name__ == "__main__":
    unittest.main()
