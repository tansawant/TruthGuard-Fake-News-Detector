"""Baseline: TF-IDF + Logistic Regression.

    python -m src.train_baseline            # full dataset (fast, ~1 min on CPU)
    python -m src.train_baseline --sample 3000
"""
from __future__ import annotations

import argparse
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import MODELS_DIR, SEED
from .data import get_splits
from .metrics import compute_metrics, save_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--name", default="tfidf_logreg")
    args = p.parse_args()

    train_df, val_df, test_df = get_splits(sample=args.sample)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                                  min_df=2, max_df=0.9, max_features=200_000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=4.0, max_iter=2000, random_state=SEED)),
    ])
    t0 = time.time()
    pipe.fit(train_df["content"], train_df["label"])
    train_time = time.time() - t0

    probs = pipe.predict_proba(test_df["content"])
    metrics = compute_metrics(test_df["label"].values, probs.argmax(1), probs[:, 1])
    save_results(args.name, metrics, {"base_model": "TF-IDF (1-2 gram) + Logistic Regression",
                                      "train_size": len(train_df), "train_time_sec": round(train_time, 1)})
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / f"{args.name}.joblib")
    print(f"[baseline] saved -> {MODELS_DIR / (args.name + '.joblib')}")


if __name__ == "__main__":
    main()
