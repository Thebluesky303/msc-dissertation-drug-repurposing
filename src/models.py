"""Neural network architectures for drug-target affinity prediction.

The baseline GCN, GAT and GIN models intentionally retain the v1 architecture.
The advanced model consumes cached, residue-level embeddings from a frozen
ESM-2 model; ESM-2 itself is kept outside the trainable DTA model.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    GINEConv,
    global_mean_pool,
)

from config import (
    BOND_FEATURE_DIM,
    DROPOUT,
    DRUG_HIDDEN_DIM,
    FUSION_HIDDEN_DIM,
    NODE_FEATURE_DIM,
    PROTEIN_EMBEDDING_DIM,
    PROTEIN_HIDDEN_DIM,
    PROTEIN_VOCAB_SIZE,
)


def _create_gin_mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    """Create the two-layer MLP used inside a GIN/GINE layer."""

    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(),
        nn.Linear(output_dim, output_dim),
    )


def _validate_edge_features(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    *,
    edge_dim: int,
    num_nodes: int,
) -> None:
    """Validate the graph tensors required by a bond-aware GINE layer."""

    if not isinstance(edge_index, torch.Tensor):
        raise TypeError("edge_index must be a torch.Tensor.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise TypeError("edge_index must contain integer node indices.")
    if edge_attr is None:
        raise ValueError(
            "GINE requires edge_attr containing molecular bond features."
        )
    if not isinstance(edge_attr, torch.Tensor):
        raise TypeError("edge_attr must be a torch.Tensor.")
    if edge_attr.ndim != 2:
        raise ValueError("edge_attr must have shape [num_edges, edge_dim].")
    if edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError(
            "edge_attr rows must match the number of columns in edge_index."
        )
    if edge_attr.shape[1] != edge_dim:
        raise ValueError(
            f"edge_attr must contain exactly {edge_dim} bond features; "
            f"received {edge_attr.shape[1]}."
        )
    if not edge_attr.is_floating_point():
        raise TypeError("edge_attr must be a floating-point tensor.")
    if not torch.isfinite(edge_attr).all():
        raise ValueError("edge_attr contains non-finite values.")
    if edge_index.numel() > 0:
        minimum = int(edge_index.min().item())
        maximum = int(edge_index.max().item())
        if minimum < 0 or maximum >= num_nodes:
            raise ValueError("edge_index contains an out-of-range node index.")


def _validate_batch_vector(batch: torch.Tensor, num_nodes: int) -> None:
    """Validate a standard, graph-contiguous PyG batch vector."""

    if not isinstance(batch, torch.Tensor):
        raise TypeError("batch must be a torch.Tensor.")
    if batch.ndim != 1 or batch.shape[0] != num_nodes:
        raise ValueError("batch must have shape [num_nodes].")
    if batch.dtype not in (torch.int32, torch.int64):
        raise TypeError("batch must contain integer graph indices.")
    if num_nodes == 0:
        raise ValueError("At least one atom is required.")
    if int(batch.min().item()) < 0:
        raise ValueError("batch graph indices cannot be negative.")
    if torch.any(batch[1:] < batch[:-1]):
        raise ValueError("batch must group atoms contiguously by graph.")

    graph_ids = torch.unique_consecutive(batch)
    expected = torch.arange(
        graph_ids.numel(), device=batch.device, dtype=batch.dtype
    )
    if not torch.equal(graph_ids, expected):
        raise ValueError("batch graph indices must be contiguous and start at zero.")


def _validate_sequence_mask(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    name: str,
    feature_dim: int | None = None,
) -> None:
    """Validate a padded sequence tensor and its boolean valid-position mask."""

    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if values.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, length, features].")
    if not values.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor.")
    if feature_dim is not None and values.shape[-1] != feature_dim:
        raise ValueError(
            f"{name} must have feature dimension {feature_dim}; "
            f"received {values.shape[-1]}."
        )
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} mask must be a torch.Tensor.")
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} mask must use boolean dtype.")
    if mask.device != values.device:
        raise ValueError(f"{name} mask must be on the same device as its values.")
    if mask.shape != values.shape[:2]:
        raise ValueError(
            f"{name} mask must have shape {tuple(values.shape[:2])}."
        )
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} cannot have an empty batch or length dimension.")
    if not torch.all(mask.any(dim=1)):
        raise ValueError(f"Every {name} sample must contain a valid position.")
    if not torch.isfinite(values[mask]).all():
        raise ValueError(f"{name} contains non-finite values at valid positions.")


# ---------------------------------------------------------------------------
# Verified v1 baseline drug encoders
# ---------------------------------------------------------------------------


class DrugGCNEncoder(nn.Module):
    """Encode molecular graphs using the verified three-layer v1 GCN."""

    def __init__(
        self,
        input_dim: int = NODE_FEATURE_DIM,
        hidden_dim: int = DRUG_HIDDEN_DIM,
        output_dim: int = DRUG_HIDDEN_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout
        self.output_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one graph-level embedding per drug."""

        del edge_attr
        x = F.dropout(
            F.relu(self.conv1(x, edge_index)),
            p=self.dropout,
            training=self.training,
        )
        x = F.dropout(
            F.relu(self.conv2(x, edge_index)),
            p=self.dropout,
            training=self.training,
        )
        x = F.relu(self.conv3(x, edge_index))
        return global_mean_pool(x, batch)


class DrugGATEncoder(nn.Module):
    """Encode molecular graphs using the verified three-layer v1 GAT."""

    def __init__(
        self,
        input_dim: int = NODE_FEATURE_DIM,
        hidden_dim: int = DRUG_HIDDEN_DIM,
        output_dim: int = DRUG_HIDDEN_DIM,
        heads: int = 4,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by the number of heads.")

        hidden_per_head = hidden_dim // heads
        self.conv1 = GATConv(
            input_dim,
            hidden_per_head,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.conv2 = GATConv(
            hidden_dim,
            hidden_per_head,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.conv3 = GATConv(
            hidden_dim,
            output_dim,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.dropout = dropout
        self.output_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one graph-level embedding per drug."""

        del edge_attr
        x = F.dropout(
            F.elu(self.conv1(x, edge_index)),
            p=self.dropout,
            training=self.training,
        )
        x = F.dropout(
            F.elu(self.conv2(x, edge_index)),
            p=self.dropout,
            training=self.training,
        )
        x = F.elu(self.conv3(x, edge_index))
        return global_mean_pool(x, batch)


class DrugGINEncoder(nn.Module):
    """Encode molecular graphs using the verified three-layer v1 GIN."""

    def __init__(
        self,
        input_dim: int = NODE_FEATURE_DIM,
        hidden_dim: int = DRUG_HIDDEN_DIM,
        output_dim: int = DRUG_HIDDEN_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.conv1 = GINConv(_create_gin_mlp(input_dim, hidden_dim))
        self.conv2 = GINConv(_create_gin_mlp(hidden_dim, hidden_dim))
        self.conv3 = GINConv(_create_gin_mlp(hidden_dim, output_dim))
        self.dropout = dropout
        self.output_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one graph-level embedding per drug."""

        del edge_attr
        x = F.dropout(
            F.relu(self.conv1(x, edge_index)),
            p=self.dropout,
            training=self.training,
        )
        x = F.dropout(
            F.relu(self.conv2(x, edge_index)),
            p=self.dropout,
            training=self.training,
        )
        x = F.relu(self.conv3(x, edge_index))
        return global_mean_pool(x, batch)


class GINEDrugEncoder(nn.Module):
    """Legacy pooled three-layer GINE encoder used by the v1 experiment.

    This class remains available so existing ``create_dta_model("gine")``
    calls keep working. The advanced atom-level encoder is
    :class:`ResidualGINEEncoder`.
    """

    def __init__(
        self,
        input_dim: int = NODE_FEATURE_DIM,
        hidden_dim: int = DRUG_HIDDEN_DIM,
        output_dim: int = DRUG_HIDDEN_DIM,
        edge_dim: int = BOND_FEATURE_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.conv1 = GINEConv(
            nn=_create_gin_mlp(input_dim, hidden_dim), edge_dim=edge_dim
        )
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GINEConv(
            nn=_create_gin_mlp(hidden_dim, hidden_dim), edge_dim=edge_dim
        )
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GINEConv(
            nn=_create_gin_mlp(hidden_dim, output_dim), edge_dim=edge_dim
        )
        self.norm3 = nn.BatchNorm1d(output_dim)
        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.output_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return one graph-level embedding per drug."""

        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x must have shape [num_atoms, {self.input_dim}]."
            )
        _validate_edge_features(
            edge_index,
            edge_attr,
            edge_dim=self.edge_dim,
            num_nodes=x.shape[0],
        )
        _validate_batch_vector(batch, x.shape[0])

        x = F.dropout(
            F.relu(self.norm1(self.conv1(x, edge_index, edge_attr))),
            p=self.dropout,
            training=self.training,
        )
        x = F.dropout(
            F.relu(self.norm2(self.conv2(x, edge_index, edge_attr))),
            p=self.dropout,
            training=self.training,
        )
        x = F.relu(self.norm3(self.conv3(x, edge_index, edge_attr)))
        return global_mean_pool(x, batch)


# ---------------------------------------------------------------------------
# Verified v1 baseline protein encoder and complete baseline model
# ---------------------------------------------------------------------------


class ProteinCNNEncoder(nn.Module):
    """Encode padded amino-acid tokens using the verified v1 protein CNN."""

    def __init__(
        self,
        vocabulary_size: int = PROTEIN_VOCAB_SIZE,
        embedding_dim: int = PROTEIN_EMBEDDING_DIM,
        hidden_dim: int = PROTEIN_HIDDEN_DIM,
        output_dim: int = PROTEIN_HIDDEN_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.conv1 = nn.Conv1d(
            embedding_dim, hidden_dim, kernel_size=7, padding=3
        )
        self.conv2 = nn.Conv1d(
            hidden_dim, output_dim, kernel_size=7, padding=3
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = dropout
        self.output_dim = output_dim

    def forward(self, protein_tensor: torch.Tensor) -> torch.Tensor:
        """Return one fixed-length embedding per protein sequence."""

        x = self.embedding(protein_tensor.long()).transpose(1, 2)
        x = F.dropout(
            F.relu(self.conv1(x)),
            p=self.dropout,
            training=self.training,
        )
        x = F.relu(self.conv2(x))
        return self.pool(x).squeeze(-1)


class DTAModel(nn.Module):
    """Combine a baseline graph encoder and protein CNN for regression."""

    def __init__(
        self,
        drug_encoder: nn.Module,
        protein_encoder: nn.Module,
        drug_encoder_name: str,
        fusion_hidden_dim: int = FUSION_HIDDEN_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.drug_encoder = drug_encoder
        self.protein_encoder = protein_encoder
        self.drug_encoder_name = str(drug_encoder_name).strip().lower()
        combined_dim = drug_encoder.output_dim + protein_encoder.output_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, data: Any) -> torch.Tensor:
        """Predict one affinity value for each drug-target pair."""

        if self.drug_encoder_name == "gine":
            drug_embedding = self.drug_encoder(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
        else:
            drug_embedding = self.drug_encoder(
                data.x, data.edge_index, data.batch
            )
        protein_embedding = self.protein_encoder(data.target)
        prediction = self.fusion(
            torch.cat([drug_embedding, protein_embedding], dim=1)
        )
        return prediction.squeeze(-1)


def create_dta_model(
    model_name: str,
    dropout: float = DROPOUT,
) -> DTAModel:
    """Create a baseline GCN/GAT/GIN model or the legacy pooled GINE model."""

    model_name = model_name.strip().lower()
    encoders: dict[str, type[nn.Module]] = {
        "gcn": DrugGCNEncoder,
        "gat": DrugGATEncoder,
        "gin": DrugGINEncoder,
        "gine": GINEDrugEncoder,
    }
    try:
        drug_encoder = encoders[model_name](dropout=dropout)
    except KeyError as error:
        raise ValueError(
            "model_name must be one of: 'gcn', 'gat', 'gin', or 'gine'."
        ) from error

    return DTAModel(
        drug_encoder=drug_encoder,
        protein_encoder=ProteinCNNEncoder(dropout=dropout),
        drug_encoder_name=model_name,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# Advanced GINE + frozen ESM-2 residue model
# ---------------------------------------------------------------------------


class ResidualGINEEncoder(nn.Module):
    """Four-layer bond-aware GINE encoder returning per-atom embeddings.

    The input atom features are projected to ``embedding_dim`` before the four
    residual GINE blocks. LayerNorm is used instead of graph-level pooling so
    the returned tensor retains shape ``[num_atoms, embedding_dim]``.
    """

    num_layers = 4

    def __init__(
        self,
        input_dim: int = NODE_FEATURE_DIM,
        embedding_dim: int = DRUG_HIDDEN_DIM,
        edge_dim: int = BOND_FEATURE_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or embedding_dim <= 0 or edge_dim <= 0:
            raise ValueError("All GINE dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

        self.input_projection = nn.Linear(input_dim, embedding_dim)
        self.convolutions = nn.ModuleList(
            [
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(embedding_dim, embedding_dim),
                        nn.GELU(),
                        nn.Linear(embedding_dim, embedding_dim),
                    ),
                    edge_dim=edge_dim,
                    train_eps=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.normalisations = nn.ModuleList(
            [nn.LayerNorm(embedding_dim) for _ in range(self.num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.output_dim = embedding_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-atom embeddings with shape ``[num_atoms, output_dim]``."""

        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor.")
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x must have shape [num_atoms, {self.input_dim}]."
            )
        if x.shape[0] == 0:
            raise ValueError("At least one atom is required.")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor.")
        if not torch.isfinite(x).all():
            raise ValueError("x contains non-finite atom features.")
        _validate_edge_features(
            edge_index,
            edge_attr,
            edge_dim=self.edge_dim,
            num_nodes=x.shape[0],
        )

        hidden = self.input_projection(x)
        for convolution, normalisation in zip(
            self.convolutions, self.normalisations, strict=True
        ):
            update = convolution(hidden, edge_index, edge_attr)
            update = self.dropout(F.gelu(update))
            hidden = normalisation(hidden + update)

        if not torch.isfinite(hidden).all():
            raise FloatingPointError("GINE produced non-finite atom embeddings.")
        return hidden


def pack_atom_embeddings(
    atom_embeddings: torch.Tensor,
    batch: torch.Tensor,
    *,
    max_atoms: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack PyG atom embeddings into a dense tensor and valid-atom mask.

    Parameters
    ----------
    atom_embeddings:
        Concatenated PyG node embeddings with shape ``[num_atoms, features]``.
    batch:
        Graph index for each atom. Atoms must be grouped by graph, as in a
        standard :class:`torch_geometric.data.Batch`.
    max_atoms:
        Optional output length. It cannot be shorter than the largest graph.

    Returns
    -------
    dense, mask:
        A zero-padded ``[B, N_atoms, features]`` tensor and a boolean valid
        mask with shape ``[B, N_atoms]``.
    """

    if not isinstance(atom_embeddings, torch.Tensor):
        raise TypeError("atom_embeddings must be a torch.Tensor.")
    if atom_embeddings.ndim != 2:
        raise ValueError("atom_embeddings must have shape [num_atoms, features].")
    if not atom_embeddings.is_floating_point():
        raise TypeError("atom_embeddings must be a floating-point tensor.")
    if not torch.isfinite(atom_embeddings).all():
        raise ValueError("atom_embeddings contains non-finite values.")
    _validate_batch_vector(batch, atom_embeddings.shape[0])
    if batch.device != atom_embeddings.device:
        raise ValueError(
            "batch must be on the same device as atom_embeddings."
        )

    num_graphs = int(batch[-1].item()) + 1
    counts = torch.bincount(batch.to(torch.int64), minlength=num_graphs)
    required_atoms = int(counts.max().item())
    if max_atoms is None:
        max_atoms = required_atoms
    if max_atoms <= 0:
        raise ValueError("max_atoms must be positive.")
    if max_atoms < required_atoms:
        raise ValueError(
            f"max_atoms={max_atoms} is smaller than the largest graph "
            f"({required_atoms} atoms)."
        )

    dense = atom_embeddings.new_zeros(
        (num_graphs, max_atoms, atom_embeddings.shape[1])
    )
    mask = torch.zeros(
        (num_graphs, max_atoms), dtype=torch.bool, device=batch.device
    )
    offsets = torch.cumsum(counts, dim=0) - counts
    positions = torch.arange(
        atom_embeddings.shape[0], device=batch.device, dtype=torch.int64
    ) - torch.repeat_interleave(offsets, counts)
    graph_indices = batch.to(torch.int64)
    dense[graph_indices, positions] = atom_embeddings
    mask[graph_indices, positions] = True
    return dense, mask


class ResidueCNNEncoder(nn.Module):
    """Map frozen ESM-2 residue embeddings to length-preserving features.

    Cached ESM-2 embeddings are detached by default to guarantee that this
    module never backpropagates into an upstream ESM-2 model. Padding is reset
    to zero before and after every convolution.
    """

    num_layers = 3

    def __init__(
        self,
        input_dim: int = 320,
        output_dim: int = PROTEIN_HIDDEN_DIM,
        kernel_size: int = 3,
        dropout: float = DROPOUT,
        detach_input: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Protein dimensions must be positive.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

        channels = [input_dim, output_dim, output_dim, output_dim]
        padding = kernel_size // 2
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(
                    channels[index],
                    channels[index + 1],
                    kernel_size=kernel_size,
                    padding=padding,
                )
                for index in range(self.num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.detach_input = detach_input

    def forward(
        self,
        protein_embeddings: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return residue features with shape ``[B, L, output_dim]``."""

        _validate_sequence_mask(
            protein_embeddings,
            protein_mask,
            name="protein_embeddings",
            feature_dim=self.input_dim,
        )
        values = (
            protein_embeddings.detach()
            if self.detach_input
            else protein_embeddings
        )
        values = values.to(dtype=self.convolutions[0].weight.dtype)
        values = values.masked_fill(~protein_mask.unsqueeze(-1), 0.0)
        values = values.transpose(1, 2)
        channel_mask = protein_mask.unsqueeze(1)

        for convolution in self.convolutions:
            values = values.masked_fill(~channel_mask, 0.0)
            values = self.dropout(F.gelu(convolution(values)))
            values = values.masked_fill(~channel_mask, 0.0)

        values = values.transpose(1, 2)
        if not torch.isfinite(values).all():
            raise FloatingPointError("Protein CNN produced non-finite features.")
        return values


class BidirectionalCrossAttention(nn.Module):
    """Update atom and residue features with masked cross-attention.

    Atoms query residues in one attention module while residues query atoms in
    a second module. Each direction has independent residual attention,
    LayerNorm and feed-forward residual sublayers.
    """

    def __init__(
        self,
        embedding_dim: int = DRUG_HIDDEN_DIM,
        num_heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or feedforward_dim <= 0:
            raise ValueError("Attention dimensions must be positive.")
        if num_heads <= 0 or embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

        self.atom_to_protein = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.protein_to_atom = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.atom_attention_norm = nn.LayerNorm(embedding_dim)
        self.protein_attention_norm = nn.LayerNorm(embedding_dim)
        self.atom_ffn_norm = nn.LayerNorm(embedding_dim)
        self.protein_ffn_norm = nn.LayerNorm(embedding_dim)
        self.atom_feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
        )
        self.protein_feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
        )
        self.residual_dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim

    @staticmethod
    def _zero_padding(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return values.masked_fill(~mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        atom_features: torch.Tensor,
        protein_features: torch.Tensor,
        atom_mask: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return updated atom and protein tensors without changing shapes."""

        _validate_sequence_mask(
            atom_features,
            atom_mask,
            name="atom_features",
            feature_dim=self.embedding_dim,
        )
        _validate_sequence_mask(
            protein_features,
            protein_mask,
            name="protein_features",
            feature_dim=self.embedding_dim,
        )
        if atom_features.shape[0] != protein_features.shape[0]:
            raise ValueError("Atom and protein batch sizes must match.")
        if atom_features.device != protein_features.device:
            raise ValueError("Atom and protein features must be on the same device.")
        if atom_mask.device != atom_features.device:
            raise ValueError("atom_mask must be on the same device as atom_features.")
        if protein_mask.device != protein_features.device:
            raise ValueError(
                "protein_mask must be on the same device as protein_features."
            )

        atoms = self._zero_padding(atom_features, atom_mask)
        proteins = self._zero_padding(protein_features, protein_mask)

        atom_update, _ = self.atom_to_protein(
            query=atoms,
            key=proteins,
            value=proteins,
            key_padding_mask=~protein_mask,
            need_weights=False,
        )
        protein_update, _ = self.protein_to_atom(
            query=proteins,
            key=atoms,
            value=atoms,
            key_padding_mask=~atom_mask,
            need_weights=False,
        )

        atoms = self.atom_attention_norm(
            atoms + self.residual_dropout(atom_update)
        )
        proteins = self.protein_attention_norm(
            proteins + self.residual_dropout(protein_update)
        )
        atoms = self._zero_padding(atoms, atom_mask)
        proteins = self._zero_padding(proteins, protein_mask)

        atoms = self.atom_ffn_norm(
            atoms + self.residual_dropout(self.atom_feedforward(atoms))
        )
        proteins = self.protein_ffn_norm(
            proteins + self.residual_dropout(self.protein_feedforward(proteins))
        )
        atoms = self._zero_padding(atoms, atom_mask)
        proteins = self._zero_padding(proteins, protein_mask)

        if not torch.isfinite(atoms).all() or not torch.isfinite(proteins).all():
            raise FloatingPointError("Cross-attention produced non-finite features.")
        return atoms, proteins


def masked_mean_pool(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool valid sequence positions while ignoring all padding."""

    _validate_sequence_mask(values, mask, name="values")
    clean_values = values.masked_fill(~mask.unsqueeze(-1), 0.0)
    denominator = mask.sum(dim=1, keepdim=True).to(values.dtype)
    pooled = clean_values.sum(dim=1) / denominator
    if not torch.isfinite(pooled).all():
        raise FloatingPointError("Masked pooling produced non-finite features.")
    return pooled


class AdvancedDTAModel(nn.Module):
    """GINE-ESM-2 bidirectional cross-attention affinity regressor.

    ``forward(data)`` expects a PyG batch with the usual graph attributes plus:

    - ``protein_embeddings``: cached frozen ESM-2 residue embeddings ``[B,L,E]``
    - ``protein_mask``: boolean valid-residue mask ``[B,L]``

    The scalar output is a dataset-specific binding-affinity score. It must not
    be described universally as pKd because Davis and KIBA use different scales.
    """

    output_name = "Predicted Binding Affinity Score"

    def __init__(
        self,
        atom_input_dim: int = NODE_FEATURE_DIM,
        bond_feature_dim: int = BOND_FEATURE_DIM,
        esm_embedding_dim: int = 320,
        embedding_dim: int = DRUG_HIDDEN_DIM,
        attention_heads: int = 4,
        attention_feedforward_dim: int = 512,
        regression_hidden_dim: int = 512,
        regression_bottleneck_dim: int = 256,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        if regression_hidden_dim <= 0 or regression_bottleneck_dim <= 0:
            raise ValueError("Regression dimensions must be positive.")

        self.drug_encoder = ResidualGINEEncoder(
            input_dim=atom_input_dim,
            embedding_dim=embedding_dim,
            edge_dim=bond_feature_dim,
            dropout=dropout,
        )
        self.protein_encoder = ResidueCNNEncoder(
            input_dim=esm_embedding_dim,
            output_dim=embedding_dim,
            dropout=dropout,
            detach_input=True,
        )
        self.cross_attention = BidirectionalCrossAttention(
            embedding_dim=embedding_dim,
            num_heads=attention_heads,
            feedforward_dim=attention_feedforward_dim,
            dropout=dropout,
        )
        self.regressor = nn.Sequential(
            nn.Linear(2 * embedding_dim, regression_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(regression_hidden_dim, regression_bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(regression_bottleneck_dim, 1),
        )
        self.embedding_dim = embedding_dim

    def encode_interactions(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        protein_embeddings: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode graph/residue inputs and return both masks with interactions."""

        _validate_batch_vector(batch, x.shape[0])
        atom_embeddings = self.drug_encoder(x, edge_index, edge_attr)
        atoms, atom_mask = pack_atom_embeddings(atom_embeddings, batch)
        proteins = self.protein_encoder(protein_embeddings, protein_mask)
        if atoms.shape[0] != proteins.shape[0]:
            raise ValueError(
                "The number of molecular graphs must match the protein batch size."
            )
        atoms, proteins = self.cross_attention(
            atoms, proteins, atom_mask, protein_mask
        )
        return atoms, proteins, atom_mask, protein_mask

    def forward_tensors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        protein_embeddings: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Tensor-level forward API used by tests and reusable training code."""

        atoms, proteins, atom_mask, protein_mask = self.encode_interactions(
            x,
            edge_index,
            edge_attr,
            batch,
            protein_embeddings,
            protein_mask,
        )
        drug_vector = masked_mean_pool(atoms, atom_mask)
        protein_vector = masked_mean_pool(proteins, protein_mask)
        prediction = self.regressor(
            torch.cat([drug_vector, protein_vector], dim=-1)
        ).squeeze(-1)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Regressor produced non-finite predictions.")
        return prediction

    def forward(self, data: Any) -> torch.Tensor:
        """Predict affinity from a PyG batch containing cached protein inputs."""

        required_attributes = (
            "x",
            "edge_index",
            "edge_attr",
            "batch",
            "protein_embeddings",
            "protein_mask",
        )
        missing = [name for name in required_attributes if not hasattr(data, name)]
        if missing:
            raise AttributeError(
                "AdvancedDTAModel input is missing required attributes: "
                + ", ".join(missing)
            )
        return self.forward_tensors(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
            data.protein_embeddings,
            data.protein_mask,
        )


def create_advanced_dta_model(**kwargs: Any) -> AdvancedDTAModel:
    """Create the advanced DTA model without loading ESM-2 or starting training."""

    return AdvancedDTAModel(**kwargs)
