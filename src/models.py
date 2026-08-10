"""Neural network architectures for drug–target affinity prediction."""

from __future__ import annotations

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
    NODE_FEATURE_DIM,
    PROTEIN_VOCAB_SIZE,
    DRUG_HIDDEN_DIM,
    PROTEIN_EMBEDDING_DIM,
    PROTEIN_HIDDEN_DIM,
    FUSION_HIDDEN_DIM,
    DROPOUT,
)
import config as cfg

# ============================================================
# DRUG GRAPH ENCODERS
# ============================================================

class DrugGCNEncoder(nn.Module):
    """Encode molecular graphs using graph convolutional layers."""

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
        edge_attr=None,
    ) -> torch.Tensor:
        """Return one graph-level embedding per drug."""

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        return global_mean_pool(x, batch)


class DrugGATEncoder(nn.Module):
    """Encode molecular graphs using graph attention layers."""

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
            raise ValueError(
                "hidden_dim must be divisible by the number of heads."
            )

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
        edge_attr=None,
    ) -> torch.Tensor:
        """Return one graph-level embedding per drug."""

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.elu(x)

        return global_mean_pool(x, batch)


def _create_gin_mlp(
    input_dim: int,
    output_dim: int,
) -> nn.Sequential:
    """Create the multilayer perceptron used inside a GIN layer."""

    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(),
        nn.Linear(output_dim, output_dim),
    )


class DrugGINEncoder(nn.Module):
    """Encode molecular graphs using graph isomorphism layers."""

    def __init__(
        self,
        input_dim: int = NODE_FEATURE_DIM,
        hidden_dim: int = DRUG_HIDDEN_DIM,
        output_dim: int = DRUG_HIDDEN_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.conv1 = GINConv(
            _create_gin_mlp(input_dim, hidden_dim)
        )

        self.conv2 = GINConv(
            _create_gin_mlp(hidden_dim, hidden_dim)
        )

        self.conv3 = GINConv(
            _create_gin_mlp(hidden_dim, output_dim)
        )

        self.dropout = dropout
        self.output_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr=None,
    ) -> torch.Tensor:
        """Return one graph-level embedding per drug."""

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        return global_mean_pool(x, batch)

class GINEDrugEncoder(nn.Module):
    """Bond-aware GINE encoder for molecular graphs."""

    def __init__(
        self,
        input_dim: int = cfg.NODE_FEATURE_DIM,
        hidden_dim: int = cfg.DRUG_HIDDEN_DIM,
        output_dim: int = cfg.DRUG_HIDDEN_DIM,
        edge_dim: int = cfg.BOND_FEATURE_DIM,
        dropout: float = cfg.DROPOUT,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.output_dim = output_dim
        # --------------------------------------------------
        # GINE layer 1
        # --------------------------------------------------
        mlp_1 = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        self.conv1 = GINEConv(
            nn=mlp_1,
            edge_dim=edge_dim,
        )

        self.norm1 = nn.BatchNorm1d(
            hidden_dim
        )

        # --------------------------------------------------
        # GINE layer 2
        # --------------------------------------------------
        mlp_2 = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        self.conv2 = GINEConv(
            nn=mlp_2,
            edge_dim=edge_dim,
        )

        self.norm2 = nn.BatchNorm1d(
            hidden_dim
        )

        # --------------------------------------------------
        # GINE layer 3
        # --------------------------------------------------
        mlp_3 = nn.Sequential(
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                output_dim,
                output_dim,
            ),
        )

        self.conv3 = GINEConv(
            nn=mlp_3,
            edge_dim=edge_dim,
        )

        self.norm3 = nn.BatchNorm1d(
            output_dim
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
    ):
        if edge_attr is None:
            raise ValueError(
                "GINE requires edge_attr "
                "containing molecular bond features."
            )

        if (
            edge_attr.shape[0]
            != edge_index.shape[1]
        ):
            raise ValueError(
                "edge_attr rows must match "
                "the number of graph edges."
            )

        # Layer 1
        x = self.conv1(
            x,
            edge_index,
            edge_attr,
        )

        x = self.norm1(x)
        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # Layer 2
        x = self.conv2(
            x,
            edge_index,
            edge_attr,
        )

        x = self.norm2(x)
        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # Layer 3
        x = self.conv3(
            x,
            edge_index,
            edge_attr,
        )

        x = self.norm3(x)
        x = F.relu(x)

        # One molecular embedding per graph
        x = global_mean_pool(
            x,
            batch,
        )

        return x
# ============================================================
# PROTEIN ENCODER
# ============================================================

class ProteinCNNEncoder(nn.Module):
    """Encode padded amino-acid sequences using a one-dimensional CNN."""

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
            in_channels=embedding_dim,
            out_channels=hidden_dim,
            kernel_size=7,
            padding=3,
        )

        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=output_dim,
            kernel_size=7,
            padding=3,
        )

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.dropout = dropout
        self.output_dim = output_dim

    def forward(
        self,
        protein_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Return one fixed-length embedding per protein sequence."""

        protein_tensor = protein_tensor.long()

        x = self.embedding(protein_tensor)

        # Conv1d expects [batch, channels, sequence length].
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.pool(x).squeeze(-1)

        return x


# ============================================================
# COMPLETE DTA MODEL
# ============================================================

class DTAModel(nn.Module):
    """
    Complete drug–target affinity regression model.

    The model combines:
        drug graph encoder
        protein sequence encoder
        fully connected affinity prediction layers
    """

    def __init__(
        self,
        drug_encoder: nn.Module,
        protein_encoder: nn.Module,
        drug_encoder_name,
        fusion_hidden_dim: int = FUSION_HIDDEN_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.drug_encoder = drug_encoder
        self.protein_encoder = protein_encoder

        self.drug_encoder_name = (
            str(
                drug_encoder_name
            )
            .strip()
            .lower()
        )


        combined_dim = (
            drug_encoder.output_dim
            + protein_encoder.output_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        """Predict one affinity value for each drug–target pair."""

        # drug_embedding = self.drug_encoder(
        #     data.x,
        #     data.edge_index,
        #     data.edge_attr,
        #     data.batch,
        # )
        if self.drug_encoder_name == "gine":

            drug_embedding = self.drug_encoder(
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
    
        else:
    
            drug_embedding = self.drug_encoder(
                data.x,
                data.edge_index,
                data.batch,
            )


        protein_embedding = self.protein_encoder(
            data.target,
        )

        combined_embedding = torch.cat(
            [drug_embedding, protein_embedding],
            dim=1,
        )

        prediction = self.fusion(combined_embedding)

        return prediction.squeeze(-1)


# ============================================================
# MODEL FACTORY
# ============================================================

# def create_dta_model(
#     model_name: str,
#     dropout: float = DROPOUT,
# ) -> DTAModel:
#     """
#     Create a GCN + CNN, GAT + CNN or GIN + CNN model.

#     Parameters
#     ----------
#     model_name:
#         One of: gcn, gat or gin.
#     """

#     model_name = model_name.strip().lower()

#     if model_name == "gcn":
#         drug_encoder = DrugGCNEncoder(dropout=dropout)

#     elif model_name == "gat":
#         drug_encoder = DrugGATEncoder(dropout=dropout)

#     elif model_name == "gin":
#         drug_encoder = DrugGINEncoder(dropout=dropout)
        
#     elif model_name == "gine":
#     drug_encoder = GINEDrugEncoder(
#         input_dim=cfg.NODE_FEATURE_DIM,
#         hidden_dim=cfg.DRUG_HIDDEN_DIM,
#         output_dim=cfg.DRUG_HIDDEN_DIM,
#         edge_dim=cfg.BOND_FEATURE_DIM,
#         dropout=dropout,
#     )

#     else:
#         raise ValueError(
#             "model_name must be one of: 'gcn', 'gat', or 'gin' or 'gine'."
#         )

#     protein_encoder = ProteinCNNEncoder(dropout=dropout)

#     return DTAModel(
#         drug_encoder=drug_encoder,
#         protein_encoder=protein_encoder,
#         dropout=dropout,
#     )

def create_dta_model(
    model_name: str,
    dropout: float = DROPOUT,
) -> DTAModel:
    """
    Create a GCN + CNN, GAT + CNN,
    GIN + CNN, or GINE + CNN model.
    """

    model_name = model_name.strip().lower()

    if model_name == "gcn":

        drug_encoder = DrugGCNEncoder(
            dropout=dropout,
        )

    elif model_name == "gat":

        drug_encoder = DrugGATEncoder(
            dropout=dropout,
        )

    elif model_name == "gin":

        drug_encoder = DrugGINEncoder(
            dropout=dropout,
        )

    elif model_name == "gine":

        drug_encoder = GINEDrugEncoder(
            input_dim=cfg.NODE_FEATURE_DIM,
            hidden_dim=cfg.DRUG_HIDDEN_DIM,
            output_dim=cfg.DRUG_HIDDEN_DIM,
            edge_dim=cfg.BOND_FEATURE_DIM,
            dropout=dropout,
        )

    else:

        raise ValueError(
            "model_name must be one of: "
            "'gcn', 'gat', 'gin', or 'gine'."
        )

    protein_encoder = ProteinCNNEncoder(
        dropout=dropout,
    )

    return DTAModel(
        drug_encoder=drug_encoder,
        protein_encoder=protein_encoder,
        drug_encoder_name=model_name,
        dropout=dropout,
    )