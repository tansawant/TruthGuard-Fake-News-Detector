"""Train PyTorch Simple RNN and BiLSTM baseline models for fake news detection."""
from __future__ import annotations

import argparse
import json
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import MODELS_DIR, SEED, TRANSFORMER_DEFAULTS
from .data import get_splits, set_seed
from .metrics import compute_metrics, save_results
from .models_rnn_lstm import SimpleVocab, RNNClassifier, LSTMClassifier


def train_model(arch: str = "lstm", name: str | None = None, sample: int | None = 3000,
                epochs: int = 3, batch_size: int = 32, lr: float = 1e-3, max_len: int = 128):
    set_seed(SEED)
    name = name or arch
    print(f"=== Training {arch.upper()} Model: '{name}' ===")

    train_df, val_df, test_df = get_splits(sample=sample)
    vocab = SimpleVocab(max_vocab=20000)
    vocab.build_vocab(train_df["content"].tolist())

    def prepare_tensor(df):
        ids = [vocab.encode(t, max_len) for t in df["content"]]
        labels = df["label"].tolist()
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    X_train, y_train = prepare_tensor(train_df)
    X_val, y_val = prepare_tensor(val_df)
    X_test, y_test = prepare_tensor(test_df)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = len(vocab.word2idx)

    if arch.lower() == "rnn":
        model = RNNClassifier(vocab_size=vocab_size).to(device)
    else:
        model = LSTMClassifier(vocab_size=vocab_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = model(x_b)
                v_loss += criterion(logits, y_b).item()
        val_loss = v_loss / len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    train_latency = (time.time() - start_time) / epochs

    # Test evaluation
    model.eval()
    all_preds, all_probs = [], []
    eval_start = time.time()
    with torch.no_grad():
        for x_b, _ in test_loader:
            x_b = x_b.to(device)
            logits = model(x_b)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(probs.argmax(axis=-1))

    inference_ms = ((time.time() - eval_start) / len(test_df)) * 1000

    probs_arr = np.array(all_probs)
    metrics = compute_metrics(test_df["label"].values, all_preds, probs_arr[:, 1])
    print(f"\n[{name}] Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

    out_dir = MODELS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    vocab.save(out_dir / "vocab.json")

    meta = {
        "name": name,
        "arch": arch,
        "base_model": f"PyTorch {arch.upper()}",
        "max_len": max_len,
        "sample": sample,
        "epochs": epochs,
        "train_size": len(train_df),
        "test_metrics": {k: metrics[k] for k in ("accuracy", "precision", "recall", "f1")},
        "inference_ms": round(inference_ms, 2)
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    save_results(name, metrics, history)
    print(f"[{name}] Model & metrics saved to {out_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["rnn", "lstm"], default="lstm")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    train_model(arch=args.arch, name=args.name, sample=args.sample, epochs=args.epochs)
