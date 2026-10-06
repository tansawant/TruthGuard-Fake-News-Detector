"""Central configuration for the Fake News Detection project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Public Hugging Face mirror of the ISOT "Fake and Real News" dataset
# (same content as Kaggle's Fake.csv / True.csv; no login needed).
HF_DATASET = "ErfanMoosaviMonazzah/fake-news-detection-dataset-English"

LABELS = {0: "FAKE", 1: "REAL"}          # 0 = Fake, 1 = Real
LABEL2ID = {v: k for k, v in LABELS.items()}

SEED = 42

# Transformer hyper-parameters (starting points, override via CLI)
TRANSFORMER_DEFAULTS = {
    "max_len": 256,
    "batch_size": 16,
    "epochs": 2,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "dropout": 0.3,
}

# Friendly short names -> Hugging Face checkpoints
MODEL_ALIASES = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
}
