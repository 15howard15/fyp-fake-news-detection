
from pathlib import Path

from dotenv import load_dotenv

# ----------------------------------------------------------------------
# Paths (resolve relative to this file so scripts work from any directory)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# Load OPENAI_API_KEY (and anything else) from .env at the project root, if
# present. Every script imports config first, so this runs exactly once,
# before any script checks os.getenv("OPENAI_API_KEY").
load_dotenv(ROOT / ".env")
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

for d in (PROCESSED_DIR, SYNTHETIC_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
SEED = 42

# Character cap for the source_text stored ALONGSIDE each generated row
# (audit trail only -- NOT the prompt, which stays bounded by
# truncate_article's default for cost). Was 1000, which truncated most
# ISOT articles mid-body and made after-the-fact fact-change
# verification impossible for edits appearing later in the article.
# 10000 covers the full length of essentially every ISOT/LIAR item
# while still guarding against a pathologically long outlier.
FULL_SOURCE_CAP = 10_000

# ----------------------------------------------------------------------
# Data split (Section 3.2.5)
# ----------------------------------------------------------------------
TEST_SIZE = 0.20          # 80/20 split
LABEL_REAL = 0
LABEL_FAKE = 1

# LIAR has 6 truth labels. Collapse to binary (Section 3.2.3).
# Tweak this mapping if your supervisor prefers a different cut.
LIAR_LABEL_MAP = {
    "true": LABEL_REAL,
    "mostly-true": LABEL_REAL,
    "half-true": LABEL_REAL,
    "barely-true": LABEL_FAKE,
    "false": LABEL_FAKE,
    "pants-fire": LABEL_FAKE,
}

# ----------------------------------------------------------------------
# TF-IDF for LR / SVM (Section 3.4.1)
# ----------------------------------------------------------------------
TFIDF_NGRAM_RANGE = (1, 2)     # unigram + bigram
TFIDF_MAX_FEATURES = 50_000

# Traditional model settings (Table 3.5.3.1)
SVM_KERNEL = "linear"
LR_SOLVER = "liblinear"

# ----------------------------------------------------------------------
# CNN (Section 3.4.1 / 3.5.3)
# ----------------------------------------------------------------------
CNN_EMBED_DIM = 100            # GloVe 100d
CNN_MAX_LEN = 300              # tokens per article (ISOT articles are long)
CNN_BATCH_SIZE = 32
CNN_OPTIMIZER = "adam"
CNN_LR = 1e-3
CNN_EPOCHS = 5
CNN_FILTER_SIZES = (3, 4, 5)   # n-gram windows
CNN_NUM_FILTERS = 100
CNN_DROPOUT = 0.5
CNN_VOCAB_SIZE = 30_000

# ----------------------------------------------------------------------
# BERT (Table 3.5.3.1)
# ----------------------------------------------------------------------
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LEN = 512
BERT_BATCH_SIZE = 16
BERT_LR = 2e-5
BERT_EPOCHS = 3

# ----------------------------------------------------------------------
# Synthetic generation (Section 3.3)
# ----------------------------------------------------------------------
OPENAI_MODEL = "gpt-4o-mini"   # cheap + good enough for fact manipulation
# The four transformation strategies from Table 3.3.1
TRANSFORMATIONS = [
    "fact_manipulation",
    "context_distortion",
    "tone_adjustment",
    "selective_omission",
]

# ----------------------------------------------------------------------
# The three data compositions (your core experiment)
# ----------------------------------------------------------------------
COMPOSITIONS = ["real_real", "mixed", "real_syn"]
