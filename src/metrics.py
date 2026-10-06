"""Evaluation metrics and plots (Accuracy, Precision, Recall, F1, Confusion Matrix)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from .config import FIGURES_DIR, LABELS, REPORTS_DIR


def compute_metrics(y_true, y_pred, y_prob_real=None) -> dict:
    """Precision / Recall / F1 are reported for the positive class `FAKE` (0),
    because catching fake news is the goal of the system."""
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "report": classification_report(y_true, y_pred, labels=[0, 1],
                                        target_names=[LABELS[0], LABELS[1]],
                                        output_dict=True, zero_division=0),
        "n_test": int(len(y_true)),
    }
    if y_prob_real is not None and len(set(y_true)) == 2:
        m["roc_auc"] = roc_auc_score(y_true, y_prob_real)
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}


def plot_confusion_matrix(cm, title: str, path):
    cm = np.asarray(cm)
    pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    annot = np.array([[f"{cm[i, j]:,}\n({pct[i, j]:.1%})" for j in range(2)] for i in range(2)])

    plt.figure(figsize=(5.2, 4.4), dpi=130)
    sns.heatmap(pct, annot=annot, fmt="", cmap="magma", vmin=0, vmax=1, cbar=False,
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"],
                annot_kws={"size": 12, "weight": "bold"}, linewidths=2, linecolor="#0b0f19")
    plt.xlabel("Predicted", fontsize=11)
    plt.ylabel("Actual", fontsize=11)
    plt.title(title, fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig(path, facecolor="white")
    plt.close()


def plot_history(history: list[dict], title: str, path):
    if not history:
        return
    steps = [h["step"] for h in history]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=130)
    ax.plot(steps, [h["train_loss"] for h in history], label="train loss", color="#7c5cff")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.grid(alpha=0.3)
    val = [(h["step"], h["val_acc"]) for h in history if h.get("val_acc") is not None]
    if val:
        ax2 = ax.twinx()
        ax2.plot(*zip(*val), "o-", color="#00d4a6", label="val accuracy")
        ax2.set_ylabel("validation accuracy")
        ax2.set_ylim(0, 1.02)
    ax.set_title(title, weight="bold")
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def save_results(name: str, metrics: dict, extra: dict | None = None):
    """Write reports/<name>_metrics.json and the confusion-matrix heatmap."""
    payload = {"model": name, **metrics, **(extra or {})}
    (REPORTS_DIR / f"{name}_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_confusion_matrix(metrics["confusion_matrix"], f"Confusion Matrix — {name}",
                          FIGURES_DIR / f"{name}_confusion_matrix.png")
    print(f"\n[{name}] accuracy={metrics['accuracy']:.4f}  precision={metrics['precision']:.4f}  "
          f"recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}")
    print(f"[{name}] confusion matrix (rows=actual Fake/Real): {metrics['confusion_matrix']}")
