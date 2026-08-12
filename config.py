"""Project configuration for final_aml_gnn_dta."""

from pathlib import Path


# ---------------------------------------------------------
# Project folders
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"

DAVIS_DIR = DATA_DIR / "davis"
KIBA_DIR = DATA_DIR / "kiba"
AML_DIR = DATA_DIR / "aml"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

RANDOM_SEED = 42


# ---------------------------------------------------------
# Data split settings
# ---------------------------------------------------------

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10


# ---------------------------------------------------------
# Training settings
# ---------------------------------------------------------

BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
MIN_IMPROVEMENT = 1e-4


# ---------------------------------------------------------
# Protein sequence setting
# ---------------------------------------------------------

MAX_PROTEIN_LENGTH = 1000


# ---------------------------------------------------------
# Create result folders
# ---------------------------------------------------------

for folder in [
    RESULTS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    FIGURES_DIR,
    CHECKPOINTS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)
