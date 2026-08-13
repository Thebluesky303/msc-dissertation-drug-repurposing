"""Neural network models for drug-target affinity prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    GINEConv,
    global_mean_pool,
)

from .preprocessing import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    PROTEIN_VOCAB_SIZE,
)
from torch_geometric.utils import to_dense_batch


# ---------------------------------------------------------------------
# Protein encoder shared by all baseline models
# ---------------------------------------------------------------------

class ProteinCNN(nn.Module):
    """Encode a protein sequence using an embedding layer and 1D CNN."""

    def __init__(
        self,
        vocab_size: int = PROTEIN_VOCAB_SIZE,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.20,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.conv1 = nn.Conv1d(
            embedding_dim,
            hidden_dim,
            kernel_size=7,
            padding=3,
        )
        self.conv2 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=5,
            padding=2,
        )
        self.conv3 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
        )

        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, protein: torch.Tensor) -> torch.Tensor:
        """
        Encode protein sequences.

        Expected input shape:
            [batch_size, sequence_length]

        Output shape:
            [batch_size, hidden_dim]
        """

        protein = protein.long()

        x = self.embedding(protein)
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)

        x = self.pool(x)
        x = x.squeeze(-1)

        return x


# ---------------------------------------------------------------------
# Drug graph encoders
# ---------------------------------------------------------------------

class GCNDrugEncoder(nn.Module):
    """Encode a molecular graph using three GCN layers."""

    def __init__(
        self,
        input_dim: int = ATOM_FEATURE_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.20,
    ):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return one vector for each drug graph in the batch."""

        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.batch_norm3(x)
        x = F.relu(x)

        return global_mean_pool(x, batch)


class GATDrugEncoder(nn.Module):
    """Encode a molecular graph using three graph attention layers."""

    def __init__(
        self,
        input_dim: int = ATOM_FEATURE_DIM,
        hidden_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.20,
    ):
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by the number of GAT heads."
            )

        head_dim = hidden_dim // heads

        self.conv1 = GATConv(
            input_dim,
            head_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.conv2 = GATConv(
            hidden_dim,
            head_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.conv3 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return one vector for each drug graph in the batch."""

        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.batch_norm3(x)
        x = F.elu(x)

        return global_mean_pool(x, batch)


def _make_gin_mlp(
    input_dim: int,
    hidden_dim: int,
) -> nn.Sequential:
    """Create the small MLP used inside a GIN layer."""

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class GINDrugEncoder(nn.Module):
    """Encode a molecular graph using three GIN layers."""

    def __init__(
        self,
        input_dim: int = ATOM_FEATURE_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.20,
    ):
        super().__init__()

        self.conv1 = GINConv(
            _make_gin_mlp(input_dim, hidden_dim)
        )
        self.conv2 = GINConv(
            _make_gin_mlp(hidden_dim, hidden_dim)
        )
        self.conv3 = GINConv(
            _make_gin_mlp(hidden_dim, hidden_dim)
        )

        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return one vector for each drug graph in the batch."""

        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.batch_norm3(x)
        x = F.relu(x)

        return global_mean_pool(x, batch)


# ---------------------------------------------------------------------
# Shared prediction head
# ---------------------------------------------------------------------

class AffinityPredictor(nn.Module):
    """Combine drug and protein vectors and predict one affinity value."""

    def __init__(
        self,
        drug_dim: int = 128,
        protein_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.30,
    ):
        super().__init__()

        combined_dim = drug_dim + protein_dim

        self.layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        drug_vector: torch.Tensor,
        protein_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Predict affinity from the drug and protein representations."""

        combined = torch.cat(
            [drug_vector, protein_vector],
            dim=1,
        )

        return self.layers(combined).squeeze(-1)


# ---------------------------------------------------------------------
# Baseline DTA models
# ---------------------------------------------------------------------

class GCNCNN(nn.Module):
    """Drug GCN + protein CNN model for affinity regression."""

    def __init__(
        self,
        drug_hidden_dim: int = 128,
        protein_hidden_dim: int = 128,
        dropout: float = 0.20,
    ):
        super().__init__()

        self.drug_encoder = GCNDrugEncoder(
            hidden_dim=drug_hidden_dim,
            dropout=dropout,
        )

        self.protein_encoder = ProteinCNN(
            hidden_dim=protein_hidden_dim,
            dropout=dropout,
        )

        self.predictor = AffinityPredictor(
            drug_dim=drug_hidden_dim,
            protein_dim=protein_hidden_dim,
        )

    def forward(self, data) -> torch.Tensor:
        """Predict one affinity value for each drug-target pair."""

        drug_vector = self.drug_encoder(
            data.x,
            data.edge_index,
            data.batch,
        )

        protein_vector = self.protein_encoder(
            data.protein,
        )

        return self.predictor(
            drug_vector,
            protein_vector,
        )


class GATCNN(nn.Module):
    """Drug GAT + protein CNN model for affinity regression."""

    def __init__(
        self,
        drug_hidden_dim: int = 128,
        protein_hidden_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.20,
    ):
        super().__init__()

        self.drug_encoder = GATDrugEncoder(
            hidden_dim=drug_hidden_dim,
            heads=heads,
            dropout=dropout,
        )

        self.protein_encoder = ProteinCNN(
            hidden_dim=protein_hidden_dim,
            dropout=dropout,
        )

        self.predictor = AffinityPredictor(
            drug_dim=drug_hidden_dim,
            protein_dim=protein_hidden_dim,
        )

    def forward(self, data) -> torch.Tensor:
        """Predict one affinity value for each drug-target pair."""

        drug_vector = self.drug_encoder(
            data.x,
            data.edge_index,
            data.batch,
        )

        protein_vector = self.protein_encoder(
            data.protein,
        )

        return self.predictor(
            drug_vector,
            protein_vector,
        )


class GINCNN(nn.Module):
    """Drug GIN + protein CNN model for affinity regression."""

    def __init__(
        self,
        drug_hidden_dim: int = 128,
        protein_hidden_dim: int = 128,
        dropout: float = 0.20,
    ):
        super().__init__()

        self.drug_encoder = GINDrugEncoder(
            hidden_dim=drug_hidden_dim,
            dropout=dropout,
        )

        self.protein_encoder = ProteinCNN(
            hidden_dim=protein_hidden_dim,
            dropout=dropout,
        )

        self.predictor = AffinityPredictor(
            drug_dim=drug_hidden_dim,
            protein_dim=protein_hidden_dim,
        )

    def forward(self, data) -> torch.Tensor:
        """Predict one affinity value for each drug-target pair."""

        drug_vector = self.drug_encoder(
            data.x,
            data.edge_index,
            data.batch,
        )

        protein_vector = self.protein_encoder(
            data.protein,
        )

        return self.predictor(
            drug_vector,
            protein_vector,
        )


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

# ---------------------------------------------------------------------
# Advanced GINE + frozen ESM-2 + cross-attention model
# ---------------------------------------------------------------------

# def _make_gine_mlp(hidden_dim: int) -> nn.Sequential:
#     """Create the MLP used inside a GINE layer."""

#     return nn.Sequential(
#         nn.Linear(hidden_dim, hidden_dim),
#         nn.GELU(),
#         nn.Linear(hidden_dim, hidden_dim),
#     )

def _make_gine_mlp(hidden_dim: int) -> nn.Sequential:
    """Create the MLP used inside a GINE layer."""

    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim * 2),
        nn.GELU(),
        nn.Linear(hidden_dim * 2, hidden_dim),
    )


class GINEDrugEncoder(nn.Module):
    """Encode atoms with four bond-aware GINE layers."""

    def __init__(
        self,
        input_dim: int = ATOM_FEATURE_DIM,
        edge_dim: int = BOND_FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.20,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.layers = nn.ModuleList(
            [
                GINEConv(
                    nn=_make_gine_mlp(hidden_dim),
                    edge_dim=edge_dim,
                    train_eps=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(hidden_dim)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded atom features and a mask of real atoms."""

        if edge_attr is None:
            raise ValueError(
                "GINE requires bond features in edge_attr."
            )

        x = self.input_projection(x)

        for layer, norm in zip(
            self.layers,
            self.norms,
        ):
            residual = x

            x = layer(
                x,
                edge_index,
                edge_attr,
            )

            x = self.dropout(F.gelu(x))
            x = norm(x + residual)

        atom_features, atom_mask = to_dense_batch(
            x,
            batch,
        )

        return atom_features, atom_mask


class ESMProteinCNN(nn.Module):
    """Process cached frozen ESM-2 residue embeddings with a 1D CNN."""

    def __init__(
        self,
        esm_dim: int = 320,
        hidden_dim: int = 128,
        dropout: float = 0.20,
    ):
        super().__init__()

        self.projection = nn.Linear(
            esm_dim,
            hidden_dim,
        )

        self.conv1 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=7,
            padding=3,
        )
        self.conv2 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=5,
            padding=2,
        )
        self.conv3 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        protein_embeddings: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return length-preserving residue features of size 128."""

        if protein_embeddings.ndim != 3:
            raise ValueError(
                "protein_embeddings must have shape "
                "[batch, length, esm_dim]."
            )

        if protein_mask.ndim != 2:
            raise ValueError(
                "protein_mask must have shape "
                "[batch, length]."
            )

        if (
            protein_embeddings.shape[:2]
            != protein_mask.shape
        ):
            raise ValueError(
                "protein_embeddings and protein_mask must share "
                "batch and length dimensions."
            )

        protein_mask = protein_mask.bool()
        x = self.projection(
            protein_embeddings.float()
        )

        residual = x
        x = self.conv1(
            x.transpose(1, 2)
        ).transpose(1, 2)
        x = self.dropout(F.gelu(x))
        x = self.norm1(x + residual)

        residual = x
        x = self.conv2(
            x.transpose(1, 2)
        ).transpose(1, 2)
        x = self.dropout(F.gelu(x))
        x = self.norm2(x + residual)

        residual = x
        x = self.conv3(
            x.transpose(1, 2)
        ).transpose(1, 2)
        x = F.gelu(x)
        x = self.norm3(x + residual)

        return x * protein_mask.unsqueeze(-1)


class CrossAttentionBlock(nn.Module):
    """Apply bidirectional attention between atoms and residues."""

    def __init__(
        self,
        hidden_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.20,
        ff_dim: int = 256,
    ):
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by heads."
            )

        self.atom_queries_protein = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.protein_queries_atom = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.atom_attention_norm = nn.LayerNorm(
            hidden_dim
        )
        self.protein_attention_norm = nn.LayerNorm(
            hidden_dim
        )

        self.atom_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.protein_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.atom_ffn_norm = nn.LayerNorm(hidden_dim)
        self.protein_ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        atom_features: torch.Tensor,
        protein_features: torch.Tensor,
        atom_mask: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return features after bidirectional cross-attention."""

        atom_mask = atom_mask.bool()
        protein_mask = protein_mask.bool()

        atom_attention, _ = self.atom_queries_protein(
            query=atom_features,
            key=protein_features,
            value=protein_features,
            key_padding_mask=~protein_mask,
            need_weights=False,
        )

        protein_attention, _ = self.protein_queries_atom(
            query=protein_features,
            key=atom_features,
            value=atom_features,
            key_padding_mask=~atom_mask,
            need_weights=False,
        )

        atoms = self.atom_attention_norm(
            atom_features + atom_attention
        )
        proteins = self.protein_attention_norm(
            protein_features + protein_attention
        )

        atoms = self.atom_ffn_norm(
            atoms + self.atom_ffn(atoms)
        )
        proteins = self.protein_ffn_norm(
            proteins + self.protein_ffn(proteins)
        )

        atoms = atoms * atom_mask.unsqueeze(-1)
        proteins = proteins * protein_mask.unsqueeze(-1)

        return atoms, proteins


class MaskedAttentionPooling(nn.Module):
    """Pool variable-length features using learned attention weights."""

    def __init__(
        self,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one pooled vector for each example."""

        mask = mask.bool()

        if not torch.all(mask.any(dim=1)):
            raise ValueError(
                "Every example must contain at least one valid position."
            )

        scores = self.score(features).squeeze(-1)

        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min,
        )

        weights = torch.softmax(
            scores,
            dim=1,
        )

        weights = weights * mask.float()
        weights = weights / (
            weights.sum(dim=1, keepdim=True)
            .clamp_min(1e-8)
        )

        return torch.sum(
            features * weights.unsqueeze(-1),
            dim=1,
        )


class AdvancedDTAModel(nn.Module):
    """GINE + frozen ESM-2/CNN + bidirectional cross-attention model."""

    def __init__(
        self,
        esm_dim: int = 320,
        hidden_dim: int = 128,
        attention_heads: int = 4,
        dropout: float = 0.20,
        regressor_dropout: float = 0.30,
    ):
        super().__init__()

        self.drug_encoder = GINEDrugEncoder(
            hidden_dim=hidden_dim,
            num_layers=4,
            dropout=dropout,
        )

        self.protein_encoder = ESMProteinCNN(
            esm_dim=esm_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.cross_attention = CrossAttentionBlock(
            hidden_dim=hidden_dim,
            heads=attention_heads,
            dropout=dropout,
            ff_dim=hidden_dim * 2,
        )

        self.drug_pool = MaskedAttentionPooling(
            hidden_dim
        )
        self.protein_pool = MaskedAttentionPooling(
            hidden_dim
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(regressor_dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(regressor_dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        data,
        protein_embeddings: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one affinity value for each drug-target pair."""

        atom_features, atom_mask = self.drug_encoder(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch,
        )

        protein_features = self.protein_encoder(
            protein_embeddings,
            protein_mask,
        )

        atom_features, protein_features = (
            self.cross_attention(
                atom_features,
                protein_features,
                atom_mask,
                protein_mask,
            )
        )

        drug_vector = self.drug_pool(
            atom_features,
            atom_mask,
        )

        protein_vector = self.protein_pool(
            protein_features,
            protein_mask,
        )

        combined = torch.cat(
            [drug_vector, protein_vector],
            dim=1,
        )

        return self.regressor(combined).squeeze(-1)
