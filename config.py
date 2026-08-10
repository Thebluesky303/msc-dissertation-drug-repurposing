from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AML_DATA_DIR = DATA_DIR / "aml"

DAVIS_RAW_DIR = RAW_DATA_DIR / "davis"
KIBA_RAW_DIR = RAW_DATA_DIR / "kiba"

SPLITS_DIR = PROCESSED_DATA_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PLOTS_DIR = RESULTS_DIR / "plots"

EXPERIMENT_LOG_PATH = RESULTS_DIR / "experiment_log.csv"


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42
EXPERIMENT_SEEDS = [42, 123, 2026]


# ============================================================
# DATA SETTINGS
# ============================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

MAX_PROTEIN_LENGTH = 1000
PROTEIN_VOCAB_SIZE = 21

# Current baseline atom-feature dimension.
NODE_FEATURE_DIM = 6


# ============================================================
# MODEL SETTINGS
# ============================================================

DRUG_HIDDEN_DIM = 128
PROTEIN_EMBEDDING_DIM = 128
PROTEIN_HIDDEN_DIM = 128
FUSION_HIDDEN_DIM = 128

# Molecular bond features used by GINE
BOND_FEATURE_DIM = 6
NODE_FEATURE_DIM = 6






BOND_FEATURE_NAMES = [
    "is_single",
    "is_double",
    "is_triple",
    "is_aromatic",
    "is_conjugated",
    "is_in_ring",
]

DROPOUT = 0.20


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 32
NUM_EPOCHS = 30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

EARLY_STOPPING_PATIENCE = 7
MINIMUM_IMPROVEMENT = 1e-4
MIN_IMPROVEMENT = 1e-4

# Zero is reliable for notebooks and macOS.
NUM_WORKERS = 0
PIN_MEMORY = False
