"""config.py — single source of truth for paths and hyperparameters."""
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

for d in (PROCESSED_DIR, SYNTHETIC_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

FULL_SOURCE_CAP = 10_000

TEST_SIZE = 0.20
LABEL_REAL = 0
LABEL_FAKE = 1

LIAR_LABEL_MAP = {
    "true": LABEL_REAL,
    "mostly-true": LABEL_REAL,
    "half-true": LABEL_REAL,
    "barely-true": LABEL_FAKE,
    "false": LABEL_FAKE,
    "pants-fire": LABEL_FAKE,
}

TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 50_000

SVM_KERNEL = "linear"
LR_SOLVER = "liblinear"

CNN_EMBED_DIM = 100
CNN_MAX_LEN = 300
CNN_BATCH_SIZE = 32
CNN_OPTIMIZER = "adam"
CNN_LR = 1e-3
CNN_EPOCHS = 5
CNN_FILTER_SIZES = (3, 4, 5)
CNN_NUM_FILTERS = 100
CNN_DROPOUT = 0.5
CNN_VOCAB_SIZE = 30_000

BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LEN = 512
BERT_BATCH_SIZE = 16
BERT_LR = 2e-5
BERT_EPOCHS = 3

OPENAI_MODEL = "gpt-4o-mini"
TRANSFORMATIONS = [
    "fact_manipulation",
    "context_distortion",
    "tone_adjustment",
    "selective_omission",
]

COMPOSITIONS = ["real_real", "half_synthetic", "full_synthetic"]
