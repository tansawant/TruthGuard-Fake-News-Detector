"""Collect every reports/*_metrics.json into one comparison table + bar chart.

    python -m src.compare
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR, REPORTS_DIR

PRETTY = {
    "tfidf_logreg": "TF-IDF + LogReg",
    "rnn": "Simple RNN",
    "lstm": "BiLSTM",
    "bert": "BERT",
    "roberta": "RoBERTa",
    "distilbert": "DistilBERT"
}
ORDER = ["tfidf_logreg", "rnn", "lstm", "distilbert", "bert", "roberta"]


def load_all() -> pd.DataFrame:
    rows = []
    for f in REPORTS_DIR.glob("*_metrics.json"):
        m = json.loads(f.read_text())
        rows.append({
            "key": m["model"], "Model": PRETTY.get(m["model"], m["model"]),
            "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"],
            "F1 Score": m["f1"], "ROC-AUC": m.get("roc_auc"),
            "Train size": m.get("train_size"), "Test size": m.get("n_test"),
            "Train time (s)": m.get("train_time_sec"),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["_o"] = df["key"].map(lambda k: ORDER.index(k) if k in ORDER else 99)
    return df.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def main():
    df = load_all()
    if df.empty:
        print("No metrics found. Train some models first.")
        return
    cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
    table = df[["Model", *cols, "ROC-AUC", "Train size", "Test size", "Train time (s)"]]
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    md = table.copy()
    for c in cols + ["ROC-AUC"]:
        md[c] = md[c].map(lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "—")
    lines = ["| " + " | ".join(md.columns) + " |", "|" + "---|" * len(md.columns)]
    lines += ["| " + " | ".join("—" if pd.isna(v) else str(v) for v in row) + " |" for row in md.values]
    (REPORTS_DIR / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    table.to_csv(REPORTS_DIR / "comparison.csv", index=False)

    x = np.arange(len(cols))
    w = 0.8 / len(df)
    palette = ["#8b8fa3", "#ffb86b", "#7c5cff", "#00d4a6", "#ff5c8a"]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
    for i, r in df.iterrows():
        bars = ax.bar(x + i * w - 0.4 + w / 2, [r[c] for c in cols], w, label=r["Model"],
                      color=palette[i % len(palette)])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    lo = max(0.0, df[cols].values.min() - 0.08)
    ax.set_ylim(lo, 1.0)
    ax.set_xticks(x, cols)
    ax.set_ylabel("score (test set)")
    ax.set_title("Model Comparison — Fake News Detection", weight="bold")
    ax.legend(ncol=len(df), loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", facecolor="white")
    print(f"\nSaved {REPORTS_DIR / 'comparison.csv'} and figures/model_comparison.png")


if __name__ == "__main__":
    main()
