# ============================================================
# CONFIG
# ============================================================

from pathlib import Path


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

DATASET_DIR = Path("split_dataset")

TRAIN_IMAGE_DIR = DATASET_DIR / "train" / "images"
TRAIN_XML_DIR = DATASET_DIR / "train" / "annotations"

VAL_IMAGE_DIR = DATASET_DIR / "val" / "images"
VAL_XML_DIR = DATASET_DIR / "val" / "annotations"

TEST_IMAGE_DIR = DATASET_DIR / "test" / "images"
TEST_XML_DIR = DATASET_DIR / "test" / "annotations"

CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Classes
# ------------------------------------------------------------

PATTERN_CLASSES = [
    "Blue Plaid",
    "Brown Plaid",
    "Dot Pattern",
    "Floral Print1",
    "Floral Print2",
    "Floral Print3",
    "Gingham",
    "Gray Plaid",
    "Houndstooth",
    "Knot Pattern",
    "Pattern1",
    "Pattern2",
    "Pattern3",
    "Pattern4",
    "Red Plaid",
    "Thick Stripe",
    "Thin Stripe",
    "Twill Plaid",
    "White Plain",
]

DEFECT_CLASSES = [
    "non-defective",
    "defective",
]


PATTERN_TO_IDX = {
    name: idx
    for idx, name in enumerate(PATTERN_CLASSES)
}

DEFECT_TO_IDX = {
    name: idx
    for idx, name in enumerate(DEFECT_CLASSES)
}


# ------------------------------------------------------------
# Image
# ------------------------------------------------------------

IMAGE_SIZE = 224


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

BATCH_SIZE = 64
NUM_WORKERS = 4

EPOCHS = 30

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05

SEED = 42


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

NUM_PATTERN_CLASSES = len(PATTERN_CLASSES)
NUM_DEFECT_CLASSES = len(DEFECT_CLASSES)

EMBED_DIM = 192
DEPTH = 6
NUM_HEADS = 3
MLP_RATIO = 4.0

DROPOUT = 0.1


# ------------------------------------------------------------
# Loss weighting
# ------------------------------------------------------------

PATTERN_LOSS_WEIGHT = 1.0
DEFECT_LOSS_WEIGHT = 1.0


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = "cuda"