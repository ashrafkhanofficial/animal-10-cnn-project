import os
# ==========================
# Project Paths
# ==========================

DATA_DIR = "data"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
DATASET_DIR = "/content/data/raw/raw-img"

CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR,"plots",)
CONFUSION_MATRIX_DIR = os.path.join(RESULTS_DIR,"confusion_matrices",)
GRADCAM_DIR = os.path.join(RESULTS_DIR,"gradcam",)
EXPERIMENTS_CSV = os.path.join(RESULTS_DIR,"metrics","experiments.csv",)

# ==========================
# Reproducibility
# ==========================

RANDOM_SEED = 42

# ==========================
# Dataset
# ==========================

NUM_CLASSES = 10