"""Central configuration for the v2 DTA workflow.

The defaults deliberately preserve the verified v1 experiment settings.  Paths
can be redirected with environment variables so the same source code works in
a local checkout, Google Colab's ``/content`` workspace, or Google Drive.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any


def _path_from_environment(variable: str, default: Path) -> Path:
    """Return an expanded, absolute path from an optional environment value."""

    configured = os.environ.get(variable)
    if configured:
        return Path(configured).expanduser().resolve()
    return default.resolve()


# Project and artifact paths -------------------------------------------------

_SOURCE_PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = _path_from_environment(
    "AML_GNN_PROJECT_ROOT",
    _SOURCE_PROJECT_ROOT,
)

DATA_DIR = _path_from_environment(
    "AML_GNN_DATA_DIR",
    PROJECT_ROOT / "data",
)
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
LEGACY_SPLITS_DIR = PROCESSED_DATA_DIR / "splits"
AML_DATA_DIR = DATA_DIR / "aml"
ESM2_CACHE_DIR = PROCESSED_DATA_DIR / "esm2"

DAVIS_RAW_DIR = RAW_DATA_DIR / "davis"
KIBA_RAW_DIR = RAW_DATA_DIR / "kiba"

MODELS_DIR = _path_from_environment(
    "AML_GNN_MODELS_DIR",
    PROJECT_ROOT / "models",
)
RESULTS_DIR = _path_from_environment(
    "AML_GNN_RESULTS_DIR",
    PROJECT_ROOT / "results",
)
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PLOTS_DIR = RESULTS_DIR / "plots"
SCREENING_RESULTS_DIR = RESULTS_DIR / "screening"
EXPERIMENT_LOG_PATH = METRICS_DIR / "experiment_log.csv"


# Reproducibility and data settings -----------------------------------------

RANDOM_SEED = 42
EXPERIMENT_SEEDS = [42, 123, 2026]

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

MAX_PROTEIN_LENGTH = 1000
PROTEIN_VOCAB_SIZE = 21

# These dimensions are part of the verified v1 input representation.
NODE_FEATURE_DIM = 6
BOND_FEATURE_DIM = 6
BOND_FEATURE_NAMES = [
    "is_single",
    "is_double",
    "is_triple",
    "is_aromatic",
    "is_conjugated",
    "is_in_ring",
]

DATASET_AFFINITY_LABELS = {
    "davis": "pKd",
    "kiba": "KIBA score",
}


# Model settings -------------------------------------------------------------

DRUG_HIDDEN_DIM = 128
PROTEIN_EMBEDDING_DIM = 128
PROTEIN_HIDDEN_DIM = 128
FUSION_HIDDEN_DIM = 128
DROPOUT = 0.20

ESM2_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
ESM2_EMBEDDING_DIM = 320
ADVANCED_HIDDEN_DIM = 128
ADVANCED_GINE_LAYERS = 4
ADVANCED_PROTEIN_CNN_LAYERS = 3
CROSS_ATTENTION_HEADS = 4
CROSS_ATTENTION_FFN_DIM = 256


# Training settings ----------------------------------------------------------

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 7
MINIMUM_IMPROVEMENT = 1e-4

# Zero workers remains the portable notebook/Colab development default.
NUM_WORKERS = 0
PIN_MEMORY = False


@dataclass(frozen=True)
class ExperimentConfig:
    """Serializable configuration for one independently runnable experiment."""

    dataset: str
    split: str
    model: str
    seed: int = RANDOM_SEED
    batch_size: int = BATCH_SIZE
    epochs: int = NUM_EPOCHS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    minimum_improvement: float = MINIMUM_IMPROVEMENT
    num_workers: int = NUM_WORKERS
    pin_memory: bool = PIN_MEMORY
    max_protein_length: int = MAX_PROTEIN_LENGTH
    save_predictions: bool = True

    def __post_init__(self) -> None:
        dataset = self.dataset.strip().lower()
        split = self.split.strip().lower().replace("-", "_")
        model = self.model.strip().lower()

        if dataset not in {"davis", "kiba"}:
            raise ValueError("dataset must be 'davis' or 'kiba'.")
        if split not in {"random", "cold_drug", "cold_target"}:
            raise ValueError(
                "split must be 'random', 'cold_drug', or 'cold_target'."
            )
        if model not in {"gcn", "gat", "gin", "advanced"}:
            raise ValueError("model must be gcn, gat, gin, or advanced.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning_rate must be positive and weight_decay non-negative."
            )
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive.")
        if self.minimum_improvement < 0:
            raise ValueError("minimum_improvement must be non-negative.")
        if self.max_protein_length < 1:
            raise ValueError("max_protein_length must be positive.")

        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "model", model)

    @property
    def run_name(self) -> str:
        """Return a deterministic, human-readable run name."""

        return f"{self.dataset}_{self.model}_{self.split}_seed{self.seed}"

    @property
    def affinity_label(self) -> str:
        """Return the dataset-specific interpretation of model output."""

        return DATASET_AFFINITY_LABELS[self.dataset]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""

        values = asdict(self)
        values["affinity_label"] = self.affinity_label
        values["run_name"] = self.run_name
        return values


def processed_pairs_path(dataset: str) -> Path:
    """Return the expected processed-pair CSV for a benchmark dataset."""

    normalized = dataset.strip().lower()
    if normalized not in {"davis", "kiba"}:
        raise ValueError("dataset must be 'davis' or 'kiba'.")
    return PROCESSED_DATA_DIR / f"{normalized}_pairs.csv"
