"""Fine-tune BERT / RoBERTa for Fake vs Real news classification.

Examples
--------
Quick CPU run (small subset, ~10-15 min):
    python -m src.train_transformer --model bert --sample 3000 --epochs 1 --max-len 128

Full GPU run (Google Colab):
    python -m src.train_transformer --model bert --epochs 2 --max-len 256
    python -m src.train_transformer --model roberta --epochs 2 --max-len 256
"""
from __future__ import annotations

import argparse
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import FIGURES_DIR, MODEL_ALIASES, MODELS_DIR, SEED, TRANSFORMER_DEFAULTS
from .data import get_splits
from .metrics import compute_metrics, plot_history, save_results
from .model import FakeNewsClassifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(df, tokenizer, max_len, batch_size, shuffle):
    records = list(zip(df["content"].tolist(), df["label"].tolist()))

    def collate(batch):
        texts, labels = zip(*batch)
        # Tokenizer produces input_ids + attention_mask (1 = real token, 0 = [PAD])
        enc = tokenizer(list(texts), truncation=True, max_length=max_len,
                        padding="longest", return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    return DataLoader(records, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, loader, device, use_amp=False):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        probs.append(torch.softmax(logits.float(), dim=-1).cpu())
        labels.append(batch["labels"])
    probs = torch.cat(probs).numpy()
    return probs, torch.cat(labels).numpy()


def main():
    d = TRANSFORMER_DEFAULTS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="bert", help="alias (bert/roberta/distilbert) or HF checkpoint")
    p.add_argument("--name", default=None, help="output name (default: alias)")
    p.add_argument("--sample", type=int, default=None, help="limit total dataset rows (for CPU)")
    p.add_argument("--max-len", type=int, default=d["max_len"])
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument("--lr", type=float, default=d["lr"])
    p.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    p.add_argument("--warmup-ratio", type=float, default=d["warmup_ratio"])
    p.add_argument("--dropout", type=float, default=d["dropout"])
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    set_seed(SEED)
    base = MODEL_ALIASES.get(args.model, args.model)
    name = args.name or (args.model if args.model in MODEL_ALIASES else base.split("/")[-1])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print(f"[train] model={base}  name={name}  device={device}")

    # ---------------- data ----------------
    train_df, val_df, test_df = get_splits(sample=args.sample)
    print(f"[train] train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    tokenizer = AutoTokenizer.from_pretrained(base)
    train_dl = make_loader(train_df, tokenizer, args.max_len, args.batch_size, True)
    val_dl = make_loader(val_df, tokenizer, args.max_len, args.batch_size * 2, False)
    test_dl = make_loader(test_df, tokenizer, args.max_len, args.batch_size * 2, False)

    # ---------------- model ---------------
    model = FakeNewsClassifier(base, dropout=args.dropout).to(device)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    total_steps = math.ceil(len(train_dl)) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    loss_fn = nn.CrossEntropyLoss()        # Loss = -Σ y_i log(p_i)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---------------- training loop -------
    history, step, best_acc = [], 0, -1.0
    out_dir = MODELS_DIR / name
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        for batch in train_dl:
            step += 1
            ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(ids, mask)                 # prediction
                loss = loss_fn(logits.float(), labels)    # calculate loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()                 # backpropagation
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)                        # update parameters
            scaler.update()
            scheduler.step()
            running.append(loss.item())

            if step % args.log_every == 0 or step == total_steps:
                avg = float(np.mean(running[-args.log_every:]))
                history.append({"step": step, "epoch": epoch, "train_loss": avg})
                rate = (time.time() - t0) / step
                print(f"  epoch {epoch} step {step:>5}/{total_steps}  loss={avg:.4f}  "
                      f"eta={rate * (total_steps - step) / 60:.1f} min", flush=True)

        # ---- validation at end of epoch ----
        vprobs, vlabels = predict(model, val_dl, device, use_amp)
        vacc = float((vprobs.argmax(1) == vlabels).mean())
        history.append({"step": step, "epoch": epoch, "train_loss": float(np.mean(running)), "val_acc": vacc})
        print(f"[train] epoch {epoch} done — val_acc={vacc:.4f}")
        if vacc > best_acc:
            best_acc = vacc
            model.save(out_dir, tokenizer, {"base_model": base, "name": name, "max_len": args.max_len,
                                            "dropout": args.dropout})
            print(f"[train] saved best checkpoint -> {out_dir}")

    train_time = time.time() - t0

    # ---------------- test evaluation -----
    model, _, _ = FakeNewsClassifier.load(out_dir, device)
    probs, labels = predict(model, test_dl, device, use_amp)
    metrics = compute_metrics(labels, probs.argmax(1), probs[:, 1])
    extra = {"base_model": base, "train_size": len(train_df), "val_size": len(val_df),
             "epochs": args.epochs, "max_len": args.max_len, "lr": args.lr,
             "batch_size": args.batch_size, "best_val_acc": best_acc,
             "train_time_sec": round(train_time, 1), "device": device,
             "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
             "history": history}
    save_results(name, metrics, extra)
    plot_history(history, f"Training — {name}", FIGURES_DIR / f"{name}_training.png")

    # store test metrics alongside the checkpoint so the app can show them
    import json
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["test_metrics"] = {k: metrics[k] for k in ("accuracy", "precision", "recall", "f1")}
    meta["train_size"] = len(train_df)
    meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
